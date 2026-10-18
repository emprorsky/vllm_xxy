"""Conservative event-confirmed GPU prefixes for one immutable model pool.

Dense full attention, TP=1, one group, 16-token full blocks only. Partial,
LoRA/salted/multimodal and remote entries are excluded. BlockRemoved lacks a
physical-copy identity: removing on any removal can underestimate duplicates.
BlockStored can be emitted at allocation, before GPU writes complete: hits
also require a successful router-observed request covering the same prefix.
"""

import hashlib
from collections import OrderedDict

from .policies import CachedBlock, PrefixDirectory


def native_key(value):
    return "bytes:" + value.hex() if isinstance(value, bytes) else "int:" + str(value)


class EventDirectory(PrefixDirectory):
    def __init__(self, capacity_tokens, block_size=16, ttl_s=60):
        super().__init__(capacity_tokens, block_size, ttl_s)
        self.ready = set()
        self.identities = OrderedDict()
        self.sequence = None
        self.seen = OrderedDict()
        self.gaps = 0
        self.epochs = 0
        self.unknown_parents = 0
        self.rejected = 0

    def clear(self):
        super().clear()
        self.ready.clear()
        self.identities.clear()

    def match(self, keys, now):
        self._expire(now)
        count = 0
        for key in keys:
            if key not in self.blocks or key not in self.ready:
                break
            count += self.block_size
        return count

    def _expire(self, now):
        super()._expire(now)
        self.ready.intersection_update(self.blocks)

    def observe(self, keys, now):
        # A successful request can refresh frequency, never invent residency.
        self._expire(now)
        for key in keys:
            if key in self.blocks:
                previous = self.blocks.pop(key)
                self.blocks[key] = CachedBlock(now, previous.references + 1)
                self.ready.add(key)

    def ingest(self, sequence, events, now, payload_digest):
        if self.sequence is not None and sequence <= self.sequence:
            if self.seen.get(sequence) == payload_digest:
                return "duplicate"
            self.clear()
            self.seen.clear()
            self.sequence = None
            self.epochs += 1
        gap = self.sequence is not None and sequence != self.sequence + 1
        if gap:
            self.gaps += 1
            self.clear()
        self.sequence = sequence
        self.seen[sequence] = payload_digest
        while len(self.seen) > 10000:
            self.seen.popitem(last=False)
        for event in events:
            kind = event["type"]
            if kind == "AllBlocksCleared":
                self.clear()
                continue
            if (
                event.get("medium") not in {None, "GPU"}
                or event.get("group_idx") not in {None, 0}
                or event.get("locality") == "REMOTE"
            ):
                continue
            if kind == "BlockRemoved":
                for block in event["block_hashes"]:
                    key = self.identities.get(native_key(block))
                    self.blocks.pop(key, None)
                    self.ready.discard(key)
                continue
            if kind != "BlockStored":
                continue
            size = event["block_size"]
            blocks = event["block_hashes"]
            tokens = event["token_ids"]
            if (
                size != self.block_size
                or len(tokens) != size * len(blocks)
                or event.get("lora_id") not in {None, 0}
                or event.get("lora_name") is not None
                or any(event.get("extra_keys") or [])
                or event.get("kv_cache_spec_sliding_window") is not None
            ):
                self.rejected += 1
                continue
            parent_native = event.get("parent_block_hash")
            parent = b""
            if parent_native is not None:
                parent_key = self.identities.get(native_key(parent_native))
                if parent_key is None:
                    self.unknown_parents += 1
                    continue
                parent = bytes.fromhex(parent_key)
            for index, block in enumerate(blocks):
                raw = b"".join(
                    token.to_bytes(4, "little", signed=False)
                    for token in tokens[index * size : (index + 1) * size]
                )
                parent = hashlib.blake2b(parent + raw, digest_size=16).digest()
                key = parent.hex()
                native = native_key(block)
                self.identities[native] = key
                self.identities.move_to_end(native)
                old = self.blocks.pop(key, None)
                self.blocks[key] = CachedBlock(now, old.references if old else 1)
            # Missing native-parent identities only produce false negatives.
            while len(self.identities) > self.capacity_blocks * 8:
                self.identities.popitem(last=False)
            while len(self.blocks) > self.capacity_blocks:
                key, _ = self.blocks.popitem(last=False)
                self.ready.discard(key)
        return "gap" if gap else "applied"
