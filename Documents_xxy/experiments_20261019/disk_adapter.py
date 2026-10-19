"""Persistent-tier admission with bounded CPU staging and measured file I/O.

CPU and disk copies overlap: only persistent disk capacity enters the policy.
Buffered file I/O may be served by the OS page cache; this is not a NAND timer.
"""

import hashlib
import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict

from .lmcache_adapter import WorkingSetLMCacheConnector
from .policies import TierTelemetry


class DiskWorkingSetLMCacheConnector(WorkingSetLMCacheConnector):
    def __init__(self, *args, **kwargs):
        self._verify_file_integrity = os.environ.get("KVEXP_VERIFY_FILE_PAYLOAD") == "1"
        self._file_digests = {}
        super().__init__(*args, **kwargs)
        self._telemetry_path = self._telemetry_path.with_name(
            self._telemetry_path.name.replace("cpu-", "disk-")
        )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        adapter = self._lmcache_engine
        if result[0] is None or result[0] <= 0 or adapter._max_tokens_per_load <= 0:
            return result
        spec = adapter.load_specs.get(request.request_id)
        if spec is None:
            return result
        expected = (
            spec.lmcache_cached_tokens
            - spec.vllm_cached_tokens
            - int(spec.lmcache_cached_tokens == request.num_tokens)
        )
        if expected != result[0]:
            # The upstream cap rounds the absolute endpoint down to a chunk,
            # but returns the unrounded delta when the GPU prefix is partial.
            if not 0 <= result[0] - expected < self._chunk or expected < 0:
                raise ValueError("Unexpected capped-load endpoint mismatch")
            self._log.write(
                json.dumps(
                    {
                        "event": "load_cap_boundary_reconciled",
                        "request_id": request.request_id,
                        "reported": result[0],
                        "reconciled": expected,
                        "gpu_prefix": spec.vllm_cached_tokens,
                        "load_endpoint": spec.lmcache_cached_tokens,
                    }
                )
                + "\n"
            )
            result = (expected, result[1])
        return result

    def _start_observer(self):
        if self._observer_started:
            return
        engine = getattr(self._lmcache_engine, "lmcache_engine", None)
        manager = getattr(engine, "storage_manager", None)
        backends = getattr(manager, "storage_backends", {})
        disk = next(
            (b for b in backends.values() if type(b).__name__ == "LocalDiskBackend"),
            None,
        )
        if disk is None:
            self._log.write(json.dumps({"event": "disk_telemetry_unavailable"}) + "\n")
            return
        if disk.use_odirect or disk.max_cache_size != self._capacity:
            raise ValueError("Disk probe requires buffered I/O and exact capacity")
        self._telemetry_path = self._telemetry_path.with_name(
            self._telemetry_path.name.replace("cpu-", "disk-")
        )
        evictions = deque()
        original_remove = disk.remove
        original_write = disk.write_file

        def remove(key, force=True):
            removed = original_remove(key, force=force)
            if removed and not force:
                evictions.append(time.monotonic())
                self._log.write(
                    json.dumps({"event": "capacity_eviction", "removed_chunks": 1})
                    + "\n"
                )
            return removed

        def write(buffer, path):
            start = time.monotonic()
            verify = getattr(self, "_verify_file_integrity", False)
            digest = hashlib.sha256(buffer).hexdigest() if verify else None
            result = original_write(buffer, path)
            if verify:
                self._file_digests[str(path)] = digest
            self._log.write(
                json.dumps(
                    {
                        "event": "disk_write",
                        "bytes": len(buffer),
                        "elapsed_s": time.monotonic() - start,
                        "timestamp": time.time(),
                        "sha256": digest,
                    }
                )
                + "\n"
            )
            return result

        def read(key, buffer, path):
            # A short or missing read must never leave stale staging bytes usable.
            start = time.monotonic()
            with open(path, "rb", buffering=0) as source:
                count = source.readinto(buffer)
            if count != len(buffer):
                raise EOFError(f"KV file read {count}/{len(buffer)} bytes")
            verify = getattr(self, "_verify_file_integrity", False)
            if verify:
                digest = hashlib.sha256(buffer).hexdigest()
                if digest != self._file_digests.get(str(path)):
                    raise ValueError("KV file digest mismatch or unknown origin")
            self._log.write(
                json.dumps(
                    {
                        "event": "disk_read",
                        "bytes": count,
                        "elapsed_s": time.monotonic() - start,
                        "timestamp": time.time(),
                        "verified": verify,
                    }
                )
                + "\n"
            )

        disk.remove, disk.write_file, disk.read_file = remove, write, read
        self._observer_started = True

        def observe():
            while not self._stop_event.is_set():
                now = time.monotonic()
                while evictions and evictions[0] < now - 60:
                    evictions.popleft()
                with disk.disk_lock:
                    resident = len(disk.dict)
                value = asdict(
                    TierTelemetry(
                        resident,
                        max(self._capacity // (self._kv_bytes * self._chunk), 1),
                        len(evictions),
                        now,
                    )
                )
                temporary = self._telemetry_path.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(value))
                temporary.replace(self._telemetry_path)
                self._stop_event.wait(0.5)

        threading.Thread(
            target=observe, daemon=True, name="kvexp-disk-observer"
        ).start()
