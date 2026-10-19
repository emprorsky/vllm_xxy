"""Optional LMCache connector with capacity-conditioned CPU write admission.

Keeps LMCache's data movement and eviction paths. The worker reports actual
CPU resident chunks; this is distinct from transfer-pinned memory utilization.
Scope: TP=1, dense full-attention models, local CPU only, non-layerwise mode.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)

from .policies import TierTelemetry, WriteAdmission


def dense_payload_bytes_per_token(kv_cache_config):
    """Use cache specs, including inline quantization scales, not weight dtype."""
    if kv_cache_config is None:
        return None
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    total = 0
    for group in kv_cache_config.kv_cache_groups:
        specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        for name in group.layer_names:
            spec = specs[name] if specs is not None else group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec) or spec.tokens_per_state != 1:
                raise ValueError(
                    "CPU admission payload accounting requires dense attention"
                )
            if spec.unpadded_page_size_bytes % spec.block_size:
                raise ValueError("non-integral KV payload bytes per token")
            total += spec.unpadded_page_size_bytes // spec.block_size
    return total or None


class WorkingSetLMCacheConnector(LMCacheConnectorV1):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        config = vllm_config.kv_transfer_config
        extra = config.kv_connector_extra_config
        self._optional_cpu_in_multi = bool(
            extra.get("admission_optional_in_multi", False)
        )
        if self._optional_cpu_in_multi:
            own_pd = getattr(self._lmcache_engine._manager.config, "enable_pd", None)
            if own_pd is not False:
                raise ValueError(
                    "Optional CPU admission requires LMCache enable_pd=False"
                )
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("working-set experiment currently requires TP=1")
        if extra.get("use_layerwise", False):
            raise ValueError("working-set telemetry requires non-layerwise mode")
        self._exp_role = role
        self._exp_dir = Path(os.environ.get("KVEXP_STATS_DIR", "/tmp/kvexp"))
        self._exp_dir.mkdir(parents=True, exist_ok=True)
        engine_id = str(
            config.engine_id or os.environ.get("KVEXP_ENGINE_ID", "default")
        )
        self._telemetry_path = self._exp_dir / f"cpu-{engine_id}.json"
        self._log_path = self._exp_dir / f"admission-{role.name}-{os.getpid()}.jsonl"
        self._log = self._log_path.open("a", encoding="utf-8", buffering=1)
        hf = vllm_config.model_config.hf_text_config
        heads = hf.num_key_value_heads
        head_size = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        dtype_bytes = torch.empty(
            (), dtype=vllm_config.model_config.dtype
        ).element_size()
        self._kv_bytes = 2 * hf.num_hidden_layers * heads * head_size * dtype_bytes
        measured_bytes = dense_payload_bytes_per_token(kv_cache_config)
        if measured_bytes is not None:
            self._kv_bytes = measured_bytes
        elif vllm_config.cache_config.cache_dtype not in {"auto", "bfloat16"}:
            raise ValueError("Non-BF16 KV payload accounting requires cache specs")
        self._chunk = int(extra.get("admission_chunk_tokens", 256))
        self._capacity = int(
            extra.get(
                "admission_capacity_bytes", extra.get("admission_cpu_bytes", 1024**3)
            )
        )
        self._admission = WriteAdmission(
            self._capacity,
            self._kv_bytes,
            self._chunk,
            mode=extra.get("admission_mode", "conditioned"),
            write_threshold=int(extra.get("admission_write_threshold", 8)),
            window_s=float(extra.get("admission_window_s", 60)),
        )
        self._latest_telemetry = None
        self._last_read = 0.0
        self._stop_event = threading.Event()
        self._observer_started = False
        if role == KVConnectorRole.WORKER:
            self._start_observer()

    def _start_observer(self):
        if self._observer_started:
            return
        engine = getattr(self._lmcache_engine, "lmcache_engine", None)
        manager = getattr(engine, "storage_manager", None)
        backends = getattr(manager, "storage_backends", {})
        cpu = next(
            (b for b in backends.values() if type(b).__name__ == "LocalCPUBackend"),
            None,
        )
        if cpu is None:
            # Missing telemetry is observable and the admission policy fails open.
            self._log.write(
                json.dumps(
                    {"event": "telemetry_unavailable", "backends": list(backends)}
                )
                + "\n"
            )
            return
        if getattr(cpu, "layerwise", False):
            raise ValueError("layerwise CPU allocator is outside observer scope")
        evictions = deque()
        original_remove = cpu.batched_remove

        def tracked_remove(keys, force=True):
            keys = list(keys)
            # force=False is the non-layerwise allocator's capacity eviction path.
            count_before = len(cpu.hot_cache)
            result = original_remove(keys, force=force)
            removed = max(count_before - len(cpu.hot_cache), 0)
            if not force and removed:
                evictions.append((time.monotonic(), removed))
                self._log.write(
                    json.dumps(
                        {
                            "event": "capacity_eviction",
                            "removed_chunks": removed,
                            "timestamp": time.time(),
                        }
                    )
                    + "\n"
                )
            return result

        cpu.batched_remove = tracked_remove
        self._observer_started = True

        def observe():
            while not self._stop_event.is_set():
                now = time.monotonic()
                while evictions and evictions[0][0] < now - 60:
                    evictions.popleft()
                with cpu.cpu_lock:
                    resident = len(cpu.hot_cache)
                value = asdict(
                    TierTelemetry(
                        resident,
                        max(self._capacity // (self._kv_bytes * self._chunk), 1),
                        sum(n for _, n in list(evictions)),
                        now,
                    )
                )
                temp = self._telemetry_path.with_suffix(f".{os.getpid()}.tmp")
                temp.write_text(json.dumps(value))
                temp.replace(self._telemetry_path)
                self._stop_event.wait(0.5)

        threading.Thread(target=observe, daemon=True, name="kvexp-cpu-observer").start()

    def register_kv_caches(self, kv_caches):
        if self._exp_role == KVConnectorRole.WORKER:
            engine = self._lmcache_engine.lmcache_engine
            connector = engine.gpu_connector
            packed = any(cache.ndim == 4 for cache in kv_caches.values())
            if packed:
                from lmcache.v1.gpu_connector.gpu_connectors import (
                    VLLMPagedMemGPUConnectorV3,
                )

                if not isinstance(connector, VLLMPagedMemGPUConnectorV3):
                    raise ValueError(
                        "Packed K/V requires LMCACHE_USE_GPU_CONNECTOR_V3=True; "
                        "V2's split-K/V buffer silently corrupts this layout."
                    )
                # Discover the fused plane BEFORE storage allocators capture
                # metadata shapes. V3's first-transfer lazy init is too late.
                connector.initialize_kvcaches_ptr(kvcaches=list(kv_caches.values()))
                connector._initialize_kv_cache_pointers()
                actual_bytes = sum(
                    shape.numel() * dtype.itemsize
                    for shape, dtype in zip(
                        engine.metadata.get_shapes(1),
                        engine.metadata.get_dtypes(),
                        strict=True,
                    )
                )
                if actual_bytes != self._kv_bytes:
                    raise ValueError(
                        "KV spec/transfer payload mismatch: "
                        f"{self._kv_bytes}/{actual_bytes}"
                    )
                self._log.write(
                    json.dumps(
                        {
                            "event": "packed_transfer_layout_ready",
                            "connector": type(connector).__name__,
                            "format": str(connector.engine_kv_format),
                            "kv_payload_bytes_per_token": actual_bytes,
                            "cpu_shapes": [
                                list(shape)
                                for shape in engine.metadata.get_shapes(self._chunk)
                            ],
                        }
                    )
                    + "\n"
                )
        super().register_kv_caches(kv_caches)
        if self._exp_role == KVConnectorRole.WORKER:
            self._start_observer()

    def _telemetry(self):
        now = time.monotonic()
        if now - self._last_read >= 0.5:
            self._last_read = now
            try:
                self._latest_telemetry = TierTelemetry(
                    **json.loads(self._telemetry_path.read_text())
                )
            except (OSError, ValueError, TypeError):
                self._latest_telemetry = None
        return self._latest_telemetry

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if result[0] is None or request.request_id in self._admission.decisions:
            return result
        params = request.kv_transfer_params or {}
        if params.get("experiment.phase") == "validation":
            return result
        if "disagg_spec" in params or (
            params.get("do_remote_decode")
            and not getattr(self, "_optional_cpu_in_multi", False)
        ):
            self._log.write(
                json.dumps({"event": "pd_bypass", "request_id": request.request_id})
                + "\n"
            )
            return result
        load_spec = getattr(self._lmcache_engine, "load_specs", {}).get(
            request.request_id
        )
        if load_spec is None:
            # No backend-confirmed prefix information: do not invent an admission.
            return result
        prompt = len(request.prompt_token_ids)
        host = min(max(load_spec.lmcache_cached_tokens, 0), prompt)
        session = str(params.get("experiment.session_id", request.request_id))
        decision = self._admission.decide(
            request.request_id,
            session,
            prompt,
            host,
            self._telemetry(),
        )
        if not decision.save:
            sampling = request.sampling_params
            extra = dict(sampling.extra_args or {})
            transfer = dict(extra.get("kv_transfer_params") or {})
            transfer["lmcache.skip_save"] = True
            extra["kv_transfer_params"] = transfer
            sampling.extra_args = extra
        self._log.write(
            json.dumps(
                {
                    "event": "decision",
                    "request_id": request.request_id,
                    "session_id": session,
                    "prompt_tokens": prompt,
                    "host_prefix_tokens": host,
                    "timestamp": time.time(),
                    **asdict(decision),
                }
            )
            + "\n"
        )
        return result

    def request_finished(self, request, block_ids):
        try:
            return super().request_finished(request, block_ids)
        finally:
            self._admission.finish(request.request_id)

    def shutdown(self):
        self._stop_event.set()
        try:
            super().shutdown()
        finally:
            self._log.close()
