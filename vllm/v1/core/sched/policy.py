# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler decision policies for preemption and waiting order.

Design doc: Documents_xxy/vllm_kv_aware_project_design.md section 5/6.

Two policies:
- ``DefaultSchedulingDecisionPolicy``: preserves base-commit behavior exactly
  (FCFS preempts the newest request; PRIORITY preempts the lowest user
  priority, tie-broken by arrival time).
- ``RecomputeAwareSchedulingDecisionPolicy``: KV-aware victim selection.
  Hard constraint first (user priority tier), then minimize recompute cost
  (``num_computed_tokens``), then stable tie-breaks.

Both policies are read-only. All request state mutation (freeing blocks,
resetting computed tokens, re-queueing) remains in the Scheduler and queues.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from vllm.v1.core.sched.request_queue import SchedulingPolicy

if TYPE_CHECKING:
    from vllm.v1.request import Request


RequestOrderKey = tuple[Any, ...]
ReclaimableResolver = Callable[["Request"], int]


@dataclass(frozen=True)
class AdmissionCandidate:
    """Read-only admission features for one bounded waiting candidate."""

    request: Request
    base_position: int
    local_cached_tokens: int

    @property
    def remaining_prefill_tokens(self) -> int:
        return max(self.request.num_tokens - self.local_cached_tokens, 0)


class SchedulingDecisionPolicy(Protocol):
    """Read-only scheduling decisions shared by Scheduler and its queues."""

    def select_preemption_victim(
        self,
        running: list[Request],
        *,
        allocation_shortfall_blocks: int = 0,
        reclaimable_resolver: ReclaimableResolver | None = None,
    ) -> Request: ...

    def waiting_order_key(self, request: Request) -> RequestOrderKey: ...

    def choose_admission(
        self,
        candidates: list[AdmissionCandidate],
        now_s: float,
        aging_threshold_s: float,
    ) -> AdmissionCandidate: ...


class DefaultSchedulingDecisionPolicy:
    """Baseline victim selection, identical to base commit."""

    def __init__(self, policy: SchedulingPolicy, admission_policy: str) -> None:
        self.policy = policy
        self.admission_policy = admission_policy

    def select_preemption_victim(
        self,
        running: list[Request],
        *,
        allocation_shortfall_blocks: int = 0,
        reclaimable_resolver: ReclaimableResolver | None = None,
    ) -> Request:
        del allocation_shortfall_blocks, reclaimable_resolver
        if self.policy == SchedulingPolicy.PRIORITY:
            # Lowest user priority (highest value); ties -> latest arrival.
            return max(running, key=lambda r: (r.priority, r.arrival_time))
        # FCFS: preempt the newest request (last in the running list).
        return running[-1]

    def waiting_order_key(self, request: Request) -> RequestOrderKey:
        """Return the baseline priority-queue order captured at enqueue."""
        return (
            request.priority,
            request.arrival_time,
            request.request_id,
            id(request),
        )

    def choose_admission(
        self,
        candidates: list[AdmissionCandidate],
        now_s: float,
        aging_threshold_s: float,
    ) -> AdmissionCandidate:
        """Choose within one bounded, same-priority candidate window."""
        if not candidates:
            raise ValueError("admission candidates must not be empty")
        if self.admission_policy != "cache_affinity":
            return candidates[0]

        aged = [
            candidate
            for candidate in candidates
            if now_s - candidate.request.arrival_time >= aging_threshold_s
        ]
        if aged:
            return min(aged, key=lambda candidate: candidate.base_position)

        return min(
            candidates,
            key=lambda candidate: (
                candidate.request.num_preemptions == 0,
                candidate.remaining_prefill_tokens,
                candidate.base_position,
            ),
        )


class RecomputeAwareSchedulingDecisionPolicy(DefaultSchedulingDecisionPolicy):
    """Recompute-aware victim selection (design doc section 6).

    Selection is layered so that user priority remains a hard constraint:

    1. Victim must come from the worst user-priority tier present in
       ``running`` (highest ``priority`` value; lower value = more important).
    2. Within that tier, prefer requests never preempted before
       (``num_preemptions == 0``). Without this, a preempted request
       re-enters ``running`` with ``num_computed_tokens`` reset to zero and
       becomes the global minimum again, so the same victim can be preempted
       repeatedly ("victim thrashing") and its user sees a stalled stream.
    3. Then pick the smallest recompute cost, approximated by
       ``num_computed_tokens`` (tokens that would have to be recomputed after
       re-admission). P0 deliberately avoids raw block counts because blocks
       may be shared and not reclaimable.
    4. Ties break toward the latest arrival, then request ID, so the victim is
       stable regardless of running-list position.
    """

    @staticmethod
    def _worst_priority_candidates(running: list[Request]) -> list[Request]:
        worst_tier = max(r.priority for r in running)
        return [r for r in running if r.priority == worst_tier]

    @staticmethod
    def _select_by_recompute_cost(candidates: list[Request]) -> Request:
        if len(candidates) == 1:
            return candidates[0]
        return min(
            candidates,
            key=lambda r: (
                r.num_preemptions > 0,
                r.num_computed_tokens,
                -r.arrival_time,
                r.request_id,
            ),
        )

    def select_preemption_victim(
        self,
        running: list[Request],
        *,
        allocation_shortfall_blocks: int = 0,
        reclaimable_resolver: ReclaimableResolver | None = None,
    ) -> Request:
        del allocation_shortfall_blocks, reclaimable_resolver
        candidates = self._worst_priority_candidates(running)
        return self._select_by_recompute_cost(candidates)

    def waiting_order_key(self, request: Request) -> RequestOrderKey:
        """Prioritize resumed requests without crossing user priorities."""
        return (
            request.priority,
            request.num_preemptions == 0,
            request.arrival_time,
            request.request_id,
            id(request),
        )


class ReclaimableAwareSchedulingDecisionPolicy(
    RecomputeAwareSchedulingDecisionPolicy
):
    """Prefer victims that make the most immediate progress on KV shortfall."""

    def select_preemption_victim(
        self,
        running: list[Request],
        *,
        allocation_shortfall_blocks: int = 0,
        reclaimable_resolver: ReclaimableResolver | None = None,
    ) -> Request:
        candidates = self._worst_priority_candidates(running)
        if allocation_shortfall_blocks <= 0 or reclaimable_resolver is None:
            return self._select_by_recompute_cost(candidates)

        reclaimable = {
            id(request): max(reclaimable_resolver(request), 0)
            for request in candidates
        }
        max_progress = max(
            min(reclaimable[id(request)], allocation_shortfall_blocks)
            for request in candidates
        )
        candidates = [
            request
            for request in candidates
            if min(reclaimable[id(request)], allocation_shortfall_blocks)
            == max_progress
        ]
        return self._select_by_recompute_cost(candidates)


def create_decision_policy(
    policy: SchedulingPolicy,
    preemption_policy: str,
    admission_policy: str = "default",
) -> SchedulingDecisionPolicy:
    """Create the configured read-only scheduling decision policy."""
    if preemption_policy == "reclaimable_aware":
        return ReclaimableAwareSchedulingDecisionPolicy(policy, admission_policy)
    if preemption_policy == "recompute_aware":
        return RecomputeAwareSchedulingDecisionPolicy(policy, admission_policy)
    return DefaultSchedulingDecisionPolicy(policy, admission_policy)
