# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy, generation-scoped scheduler features derived from KV state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.metrics.stats import SchedulingFeatureStats
from vllm.v1.request import Request


@dataclass(frozen=True)
class CheapRequestFeatures:
    """Request metadata that never consults KV state."""

    priority: int
    arrival_time: float
    num_preemptions: int
    num_computed_tokens: int

    @property
    def is_resumed(self) -> bool:
        return self.num_preemptions > 0

    def age_s(self, now_s: float) -> float:
        return max(now_s - self.arrival_time, 0.0)


@dataclass(frozen=True)
class LocalPrefixFeature:
    """Event-free local prefix result valid for one KV generation."""

    blocks: KVCacheBlocks
    cached_tokens: int
    shared_prefix_boundary: int


LocalPrefixResolver = Callable[[Request], tuple[KVCacheBlocks, int, int]]


class SchedulingFeatureContext:
    """Memoize KV-derived scheduling features until the next KV mutation."""

    def __init__(self, local_prefix_resolver: LocalPrefixResolver) -> None:
        self._local_prefix_resolver = local_prefix_resolver
        self._local_prefix: dict[int, tuple[Request, LocalPrefixFeature]] = {}
        self._kv_generation = 0
        self._stats = SchedulingFeatureStats()

    @property
    def kv_generation(self) -> int:
        return self._kv_generation

    @staticmethod
    def cheap(request: Request) -> CheapRequestFeatures:
        return CheapRequestFeatures(
            priority=request.priority,
            arrival_time=request.arrival_time,
            num_preemptions=request.num_preemptions,
            num_computed_tokens=request.num_computed_tokens,
        )

    def local_prefix(self, request: Request) -> tuple[LocalPrefixFeature, bool]:
        """Return the local prefix feature and whether it was memoized."""
        self._stats.prefix_requests += 1
        key = id(request)
        cached = self._local_prefix.get(key)
        if cached is not None and cached[0] is request:
            self._stats.prefix_cache_hits += 1
            return cached[1], True

        blocks, cached_tokens, shared_prefix_boundary = self._local_prefix_resolver(
            request
        )
        feature = LocalPrefixFeature(
            blocks,
            cached_tokens,
            shared_prefix_boundary,
        )
        self._local_prefix[key] = (request, feature)
        self._stats.prefix_resolutions += 1
        return feature, False

    def remaining_prefill_tokens(self, request: Request) -> int:
        """Resolve the request's remaining local prefill work."""
        cheap = self.cheap(request)
        if cheap.num_computed_tokens > 0:
            cached_tokens = cheap.num_computed_tokens
        else:
            cached_tokens = self.local_prefix(request)[0].cached_tokens
        return max(request.num_tokens - cached_tokens, 0)

    def invalidate_kv_features(self) -> None:
        """Advance the KV generation and discard all derived KV features."""
        self._kv_generation += 1
        self._local_prefix.clear()
        self._stats.invalidations += 1

    def take_stats(self) -> SchedulingFeatureStats:
        """Return counters accumulated since the previous stats flush."""
        stats = self._stats
        self._stats = SchedulingFeatureStats()
        return stats
