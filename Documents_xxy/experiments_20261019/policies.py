"""Pure placement and capacity-conditioned write-admission policies.

Inspired by Preble (arXiv:2407.00023) and EfficientAgent
(arXiv:2609.33762). These are explicitly scoped experimental adaptations.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class TierTelemetry:
    resident_chunks: int
    capacity_chunks: int
    recent_evictions: int
    timestamp: float


@dataclass(frozen=True)
class AdmissionDecision:
    save: bool
    pressure: bool
    working_set_bytes: float
    active_sessions: int
    new_chunks: int
    telemetry_fresh: bool
    above_capacity: bool
    full_and_evicting: bool


class WriteAdmission:
    """Freeze one decision per logical request; retain sessions during tool gaps."""

    def __init__(
        self,
        capacity_bytes: int,
        bytes_per_token: int,
        chunk_tokens: int = 256,
        mode: Literal["none", "fixed", "conditioned"] = "conditioned",
        write_threshold: int = 8,
        window_s: float = 60.0,
        occupancy_threshold: float = 0.95,
        telemetry_max_age_s: float = 5.0,
    ):
        if min(capacity_bytes, bytes_per_token, chunk_tokens) <= 0:
            raise ValueError("capacity, KV bytes/token and chunk size must be positive")
        if mode not in {"none", "fixed", "conditioned"}:
            raise ValueError(f"unknown admission mode: {mode}")
        if write_threshold < 0 or window_s <= 0:
            raise ValueError("invalid threshold or observation window")
        if not 0 < occupancy_threshold <= 1 or telemetry_max_age_s <= 0:
            raise ValueError("invalid occupancy or telemetry age")
        self.capacity_bytes = capacity_bytes
        self.bytes_per_token = bytes_per_token
        self.chunk_tokens = chunk_tokens
        self.mode = mode
        self.write_threshold = write_threshold
        self.window_s = window_s
        self.occupancy_threshold = occupancy_threshold
        self.telemetry_max_age_s = telemetry_max_age_s
        self.sessions: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self.decisions: dict[str, AdmissionDecision] = {}

    def decide(
        self,
        request_id: str,
        session_id: str,
        prompt_tokens: int,
        host_prefix_tokens: int,
        telemetry: TierTelemetry | None,
        now: float | None = None,
    ) -> AdmissionDecision:
        if request_id in self.decisions:
            return self.decisions[request_id]
        now = time.monotonic() if now is None else now
        if prompt_tokens < 0 or not 0 <= host_prefix_tokens <= prompt_tokens:
            raise ValueError("invalid prompt or CPU prefix length")
        while (
            self.sessions
            and next(iter(self.sessions.values()))[0] < now - self.window_s
        ):
            self.sessions.popitem(last=False)
        self.sessions[session_id] = (now, prompt_tokens)
        self.sessions.move_to_end(session_id)
        active = len(self.sessions)
        mean = sum(n for _, n in self.sessions.values()) / max(active, 1)
        working_set = max(active - 1, 0) * mean * self.bytes_per_token
        fresh = (
            telemetry is not None
            and 0 <= now - telemetry.timestamp <= self.telemetry_max_age_s
            and telemetry.capacity_chunks > 0
        )
        full_evicting = bool(
            fresh
            and telemetry is not None
            and telemetry.resident_chunks / telemetry.capacity_chunks
            >= self.occupancy_threshold
            and telemetry.recent_evictions > 0
        )
        above = working_set > self.capacity_bytes
        pressure = above and full_evicting
        new_chunks = max(
            prompt_tokens // self.chunk_tokens
            - host_prefix_tokens // self.chunk_tokens,
            0,
        )
        filter_active = self.mode == "fixed" or (
            self.mode == "conditioned" and pressure
        )
        decision = AdmissionDecision(
            save=not (filter_active and new_chunks > self.write_threshold),
            pressure=pressure,
            working_set_bytes=working_set,
            active_sessions=active,
            new_chunks=new_chunks,
            telemetry_fresh=fresh,
            above_capacity=above,
            full_and_evicting=full_evicting,
        )
        self.decisions[request_id] = decision
        return decision

    def finish(self, request_id: str) -> None:
        self.decisions.pop(request_id, None)


def prefix_keys(tokens: list[int], block_size: int = 16) -> list[str]:
    """Hash complete blocks chained to their entire preceding token prefix."""
    if block_size <= 0:
        raise ValueError("block size must be positive")
    parent = b""
    keys = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        raw = b"".join(
            t.to_bytes(4, "little", signed=False)
            for t in tokens[start : start + block_size]
        )
        parent = hashlib.blake2b(parent + raw, digest_size=16).digest()
        keys.append(parent.hex())
    return keys


@dataclass
class CachedBlock:
    timestamp: float
    references: int = 1


class PrefixDirectory:
    """Bounded history-based cache estimate, NEVER an engine residency oracle.

    Baselines share this same directory. Backend restarts require clear().
    Engine-observed hit counts are logged separately to measure model error.
    """

    def __init__(self, capacity_tokens: int, block_size: int = 16, ttl_s: float = 60):
        if capacity_tokens < block_size or block_size <= 0 or ttl_s <= 0:
            raise ValueError("invalid directory capacity, block size or TTL")
        self.capacity_blocks = capacity_tokens // block_size
        self.block_size = block_size
        self.ttl_s = ttl_s
        self.blocks: OrderedDict[str, CachedBlock] = OrderedDict()

    def _expire(self, now: float) -> None:
        while (
            self.blocks
            and next(iter(self.blocks.values())).timestamp < now - self.ttl_s
        ):
            self.blocks.popitem(last=False)

    def match(self, keys: list[str], now: float) -> int:
        self._expire(now)
        count = 0
        for key in keys:
            if key not in self.blocks:
                break
            count += self.block_size
        return count

    def observe(self, keys: list[str], now: float) -> None:
        self._expire(now)
        for key in keys:
            previous = self.blocks.pop(key, None)
            self.blocks[key] = CachedBlock(
                now, previous.references + 1 if previous else 1
            )
        while len(self.blocks) > self.capacity_blocks:
            self.blocks.popitem(last=False)

    def eviction_penalty_tokens(self, keys: list[str], now: float) -> float:
        self._expire(now)
        protected = set(keys)
        incoming = sum(key not in self.blocks for key in keys)
        shortage = max(incoming - (self.capacity_blocks - len(self.blocks)), 0)
        cost = 0.0
        for key, block in self.blocks.items():
            if shortage == 0:
                break
            if key not in protected:
                cost += self.block_size * min(block.references, 16)
                shortage -= 1
        return cost

    def clear(self) -> None:
        self.blocks.clear()


@dataclass(frozen=True)
class PlacementScore:
    worker: int
    cached_tokens_est: int
    inflight: int
    queue_ms: float
    prefill_ms: float
    eviction_ms: float
    total_ms: float


class PlacementPolicy:
    """A synchronous reservation ledger makes concurrent dispatch atomic."""

    def __init__(
        self,
        workers: int,
        policy: str = "cost",
        capacity_tokens: int = 8192,
        block_size: int = 16,
        prefill_ms_per_token: float = 0.05,
        decode_ms_per_token: float = 8,
        balance_abs: int = 4,
        balance_rel: float = 1.5,
        cache_threshold: float = 0.5,
        eviction_weight: float = 1.0,
        ttl_s: float = 60,
    ):
        if workers <= 0 or policy not in {
            "rr",
            "load",
            "prefix",
            "threshold",
            "work",
            "cost",
            "work_remaining",
            "cost_remaining",
        }:
            raise ValueError("invalid worker count or policy")
        if min(prefill_ms_per_token, decode_ms_per_token) < 0:
            raise ValueError("negative execution cost")
        self.policy = policy
        self.directories = [
            PrefixDirectory(capacity_tokens, block_size, ttl_s) for _ in range(workers)
        ]
        self.prefill_ms_per_token = prefill_ms_per_token
        self.decode_ms_per_token = decode_ms_per_token
        self.balance_abs = balance_abs
        self.balance_rel = balance_rel
        self.cache_threshold = cache_threshold
        self.eviction_weight = eviction_weight
        self.reservations: dict[str, tuple[int, float]] = {}
        self.output_progress: dict[str, tuple[int, int]] = {}
        self.cursor = 0
        self.completions: deque[tuple[float, int, int]] = deque(maxlen=4096)

    def choose(
        self,
        request_id: str,
        tokens: list[int],
        output_budget: int,
        now: float | None = None,
    ) -> tuple[int, list[PlacementScore]]:
        if request_id in self.reservations:
            raise ValueError("request already reserved")
        if not tokens or output_budget < 0:
            raise ValueError("invalid prompt or output budget")
        now = time.monotonic() if now is None else now
        keys = prefix_keys(tokens, self.directories[0].block_size)
        scores = []
        for i, directory in enumerate(self.directories):
            outstanding = [
                work for worker, work in self.reservations.values() if worker == i
            ]
            cached = directory.match(keys, now)
            prefill = max(len(tokens) - cached, 0) * self.prefill_ms_per_token
            eviction = (
                directory.eviction_penalty_tokens(keys, now)
                * self.prefill_ms_per_token
                * self.eviction_weight
            )
            queue = sum(outstanding)
            scores.append(
                PlacementScore(
                    i,
                    cached,
                    len(outstanding),
                    queue,
                    prefill,
                    eviction,
                    queue + prefill + eviction,
                )
            )
        least = min(scores, key=lambda s: (s.inflight, s.worker))
        affinity = min(
            scores, key=lambda s: (-s.cached_tokens_est, s.inflight, s.worker)
        )
        if self.policy == "rr":
            chosen = self.cursor % len(scores)
            self.cursor += 1
        elif self.policy == "load":
            chosen = least.worker
        elif self.policy == "prefix":
            chosen = affinity.worker
        elif self.policy == "threshold":
            difference = max(s.inflight for s in scores) - least.inflight
            ratio = max(s.inflight for s in scores) / max(least.inflight, 1)
            imbalanced = difference >= self.balance_abs and ratio >= self.balance_rel
            match = affinity.cached_tokens_est / len(tokens)
            chosen = (
                least.worker
                if imbalanced or match < self.cache_threshold
                else affinity.worker
            )
        else:
            chosen = min(
                scores,
                key=lambda s: (
                    s.queue_ms
                    + s.prefill_ms
                    + (s.eviction_ms if self.policy.startswith("cost") else 0),
                    s.worker,
                ),
            ).worker
        selected = scores[chosen]
        # The decode budget is a visible upper bound, not an oracle prediction.
        reserve = selected.prefill_ms + output_budget * self.decode_ms_per_token
        self.reservations[request_id] = (chosen, reserve)
        if self.policy.endswith("_remaining"):
            self.output_progress[request_id] = (output_budget, 0)
        return chosen, scores

    def observe_output(self, request_id: str, emitted_tokens: int) -> None:
        """Subtract observed output, never elapsed wall time or future output."""
        if request_id not in self.output_progress or emitted_tokens <= 0:
            return
        budget, previous = self.output_progress[request_id]
        seen = min(max(previous, emitted_tokens), budget)
        self.output_progress[request_id] = (budget, seen)
        worker, _ = self.reservations[request_id]
        # A real output token proves prefill has finished. The remaining
        # visible budget remains an upper bound when early EOS is possible.
        self.reservations[request_id] = (
            worker,
            (budget - seen) * self.decode_ms_per_token,
        )

    def finish(
        self,
        request_id: str,
        tokens: list[int],
        success: bool,
        now: float | None = None,
    ) -> None:
        reservation = self.reservations.pop(request_id, None)
        self.output_progress.pop(request_id, None)
        if reservation is None:
            return
        now = time.monotonic() if now is None else now
        worker, _ = reservation
        if success:
            self.directories[worker].observe(
                prefix_keys(tokens, self.directories[worker].block_size), now
            )
            self.completions.append((now, worker, len(tokens)))


def decision_dict(decision: AdmissionDecision) -> dict:
    return asdict(decision)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q / 100
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
