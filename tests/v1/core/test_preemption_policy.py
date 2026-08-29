# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for preemption victim selection policies (design doc 17.1).

Covers: default compatibility, strict user-priority constraint,
recompute-aware victim choice, and priority-beats-recompute-cost.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.core.sched.policy import AdmissionCandidate, create_decision_policy
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue


def make_request(
    rid: str,
    priority: int = 0,
    arrival_time: float = 0.0,
    num_computed_tokens: int = 0,
    num_preemptions: int = 0,
    num_tokens: int = 100,
) -> Any:
    return SimpleNamespace(
        request_id=rid,
        priority=priority,
        arrival_time=arrival_time,
        num_computed_tokens=num_computed_tokens,
        num_preemptions=num_preemptions,
        num_tokens=num_tokens,
    )


class TestDefaultPolicy:
    """Test A: default compatibility with base commit."""

    def test_fcfs_default_picks_newest(self):
        policy = create_decision_policy(SchedulingPolicy.FCFS, "default")
        reqs = [
            make_request("a", arrival_time=1.0),
            make_request("b", arrival_time=2.0),
            make_request("c", arrival_time=3.0),
        ]
        assert policy.select_preemption_victim(reqs) is reqs[-1]

    def test_priority_default_picks_lowest_priority(self):
        policy = create_decision_policy(SchedulingPolicy.PRIORITY, "default")
        reqs = [
            make_request("a", priority=0, arrival_time=1.0),
            make_request("b", priority=5, arrival_time=1.0),
            make_request("c", priority=5, arrival_time=9.0),
        ]
        # Lowest user priority (highest value); tie -> latest arrival.
        assert policy.select_preemption_victim(reqs) is reqs[2]


class TestRecomputeAwarePolicy:
    """Test B/D/E: recompute-aware victim selection."""

    def test_same_tier_picks_min_recompute_cost(self):
        """Test D: A (128 computed) vs B (2048 computed) -> victim is A."""
        policy = create_decision_policy(SchedulingPolicy.FCFS, "recompute_aware")
        reqs = [
            make_request("a", num_computed_tokens=2048, arrival_time=1.0),
            make_request("b", num_computed_tokens=128, arrival_time=2.0),
        ]
        assert policy.select_preemption_victim(reqs) is reqs[1]

    def test_priority_is_hard_constraint(self):
        """Test B/E: low-priority high-cost request is preempted even when a
        high-priority request has a much smaller recompute cost."""
        policy = create_decision_policy(SchedulingPolicy.FCFS, "recompute_aware")
        reqs = [
            make_request("important", priority=0, num_computed_tokens=128),
            make_request("cheap", priority=7, num_computed_tokens=4096),
        ]
        assert policy.select_preemption_victim(reqs) is reqs[1]

    def test_tie_break_latest_arrival(self):
        policy = create_decision_policy(SchedulingPolicy.FCFS, "recompute_aware")
        reqs = [
            make_request("old", num_computed_tokens=500, arrival_time=1.0),
            make_request("new", num_computed_tokens=500, arrival_time=2.0),
        ]
        assert policy.select_preemption_victim(reqs) is reqs[1]

    def test_single_request(self):
        policy = create_decision_policy(SchedulingPolicy.FCFS, "recompute_aware")
        req = make_request("only", num_computed_tokens=100)
        assert policy.select_preemption_victim([req]) is req

    def test_all_same_tier_mixed_costs(self):
        """Victim is argmin over the whole running list when priorities tie."""
        policy = create_decision_policy(SchedulingPolicy.PRIORITY, "recompute_aware")
        reqs = [
            make_request("a", priority=0, arrival_time=1.0, num_computed_tokens=100),
            make_request("b", priority=0, arrival_time=2.0, num_computed_tokens=50),
            make_request("c", priority=0, arrival_time=3.0, num_computed_tokens=75),
        ]
        assert policy.select_preemption_victim(reqs) is reqs[1]

    def test_anti_thrashing_never_preempted_wins(self):
        """A previously-preempted request with the smallest recompute cost
        must NOT be picked again while a never-preempted candidate exists
        (prevents victim thrashing)."""
        policy = create_decision_policy(SchedulingPolicy.FCFS, "recompute_aware")
        reqs = [
            make_request("thrashed", num_computed_tokens=16, num_preemptions=2),
            make_request("fresh", num_computed_tokens=2048, num_preemptions=0),
        ]
        assert policy.select_preemption_victim(reqs) is reqs[1]

    def test_anti_thrashing_all_preempted_falls_back_to_recompute_cost(self):
        """When every candidate has been preempted at least once, fall back
        to the min recompute cost ordering."""
        policy = create_decision_policy(SchedulingPolicy.FCFS, "recompute_aware")
        reqs = [
            make_request("a", num_computed_tokens=900, num_preemptions=2),
            make_request("b", num_computed_tokens=100, num_preemptions=2),
        ]
        assert policy.select_preemption_victim(reqs) is reqs[1]

    def test_resume_protection_is_binary(self):
        policy = create_decision_policy(SchedulingPolicy.FCFS, "recompute_aware")
        reqs = [
            make_request("once", num_computed_tokens=900, num_preemptions=1),
            make_request("many", num_computed_tokens=100, num_preemptions=3),
        ]
        assert policy.select_preemption_victim(reqs) is reqs[1]

    def test_victim_final_tie_is_deterministic(self):
        policy = create_decision_policy(SchedulingPolicy.FCFS, "recompute_aware")
        first = make_request("a", num_computed_tokens=100, arrival_time=1.0)
        second = make_request("b", num_computed_tokens=100, arrival_time=1.0)

        assert policy.select_preemption_victim([first, second]) is first
        assert policy.select_preemption_victim([second, first]) is first


class TestReclaimableAwarePolicy:
    @staticmethod
    def select(reqs, shortfall, blocks):
        policy = create_decision_policy(SchedulingPolicy.FCFS, "reclaimable_aware")
        return policy.select_preemption_victim(
            reqs,
            allocation_shortfall_blocks=shortfall,
            reclaimable_resolver=lambda request: blocks[request.request_id],
        )

    def test_sufficient_reclaimability_beats_lower_recompute_cost(self):
        expensive = make_request("expensive", num_computed_tokens=1000)
        cheap = make_request("cheap", num_computed_tokens=10)

        assert self.select([expensive, cheap], 3, {"expensive": 3, "cheap": 0}) is (
            expensive
        )

    def test_reclaimability_is_capped_at_shortfall(self):
        expensive = make_request("expensive", num_computed_tokens=1000)
        cheap = make_request("cheap", num_computed_tokens=10)

        assert self.select([expensive, cheap], 3, {"expensive": 8, "cheap": 3}) is (
            cheap
        )

    def test_larger_partial_reduction_wins(self):
        larger = make_request("larger", num_computed_tokens=1000)
        cheaper = make_request("cheaper", num_computed_tokens=10)

        assert self.select([larger, cheaper], 3, {"larger": 2, "cheaper": 1}) is (
            larger
        )

    def test_zero_progress_falls_back_to_recompute_order(self):
        expensive = make_request("expensive", num_computed_tokens=1000)
        cheap = make_request("cheap", num_computed_tokens=10)

        assert self.select([expensive, cheap], 3, {"expensive": 0, "cheap": 0}) is (
            cheap
        )

    def test_resume_protection_breaks_equal_progress_tie(self):
        resumed = make_request(
            "resumed", num_computed_tokens=10, num_preemptions=1
        )
        fresh = make_request("fresh", num_computed_tokens=1000)

        assert self.select([resumed, fresh], 2, {"resumed": 2, "fresh": 2}) is fresh

    def test_priority_is_hard_constraint_and_bounds_resolution(self):
        important = make_request("important", priority=0, num_computed_tokens=10)
        low_priority = make_request("low", priority=5, num_computed_tokens=1000)
        resolved = []
        policy = create_decision_policy(SchedulingPolicy.PRIORITY, "reclaimable_aware")

        victim = policy.select_preemption_victim(
            [important, low_priority],
            allocation_shortfall_blocks=2,
            reclaimable_resolver=lambda request: resolved.append(request.request_id)
            or 0,
        )

        assert victim is low_priority
        assert resolved == ["low"]


class TestWaitingOrder:
    def test_fcfs_prepend_still_resumes_first(self):
        policy = create_decision_policy(SchedulingPolicy.FCFS, "recompute_aware")
        queue = create_request_queue(SchedulingPolicy.FCFS, policy.waiting_order_key)
        fresh = make_request("fresh")
        resumed = make_request("resumed", num_preemptions=1)

        queue.add_request(fresh)
        queue.prepend_request(resumed)

        assert list(queue) == [resumed, fresh]

    def test_resumed_request_precedes_fresh_request_in_same_priority(self):
        policy = create_decision_policy(SchedulingPolicy.PRIORITY, "recompute_aware")
        queue = create_request_queue(
            SchedulingPolicy.PRIORITY, policy.waiting_order_key
        )
        fresh = make_request("fresh", arrival_time=1.0)
        resumed = make_request("resumed", arrival_time=2.0, num_preemptions=1)

        queue.add_request(fresh)
        queue.add_request(resumed)

        assert list(queue) == [resumed, fresh]

    def test_resume_does_not_cross_user_priority(self):
        policy = create_decision_policy(SchedulingPolicy.PRIORITY, "recompute_aware")
        queue = create_request_queue(
            SchedulingPolicy.PRIORITY, policy.waiting_order_key
        )
        important = make_request("important", priority=0)
        resumed = make_request("resumed", priority=1, num_preemptions=1)

        queue.add_request(resumed)
        queue.add_request(important)

        assert list(queue) == [important, resumed]

    def test_default_priority_readmission_is_unchanged(self):
        policy = create_decision_policy(SchedulingPolicy.PRIORITY, "default")
        queue = create_request_queue(
            SchedulingPolicy.PRIORITY, policy.waiting_order_key
        )
        fresh = make_request("fresh", arrival_time=1.0)
        resumed = make_request("resumed", arrival_time=2.0, num_preemptions=1)

        queue.add_request(resumed)
        queue.add_request(fresh)

        assert list(queue) == [fresh, resumed]

    def test_priority_heap_key_is_captured_at_enqueue(self):
        policy = create_decision_policy(SchedulingPolicy.PRIORITY, "recompute_aware")
        queue = create_request_queue(
            SchedulingPolicy.PRIORITY, policy.waiting_order_key
        )
        request = make_request("late", arrival_time=2.0)
        earlier = make_request("early", arrival_time=1.0)
        queue.add_request(request)
        request.num_preemptions = 1
        queue.add_request(earlier)

        assert queue.peek_request() is earlier

        queue.remove_request(request)
        queue.add_request(request)
        assert queue.peek_request() is request


class TestAdmissionPolicy:
    def test_default_preserves_base_order(self):
        policy = create_decision_policy(
            SchedulingPolicy.FCFS, "default", "default"
        )
        first = AdmissionCandidate(make_request("first"), 0, 0)
        warm = AdmissionCandidate(make_request("warm"), 1, 80)

        assert policy.choose_admission([first, warm], 10.0, 30.0) is first

    def test_cache_affinity_prefers_resume_then_remaining_work(self):
        policy = create_decision_policy(
            SchedulingPolicy.FCFS, "default", "cache_affinity"
        )
        cold = AdmissionCandidate(make_request("cold"), 0, 0)
        warm = AdmissionCandidate(make_request("warm"), 1, 80)
        resumed = AdmissionCandidate(
            make_request("resumed", num_preemptions=1), 2, 0
        )

        chosen = policy.choose_admission([cold, warm, resumed], 10.0, 30.0)

        assert chosen is resumed

    def test_aged_candidate_returns_to_base_order(self):
        policy = create_decision_policy(
            SchedulingPolicy.FCFS, "default", "cache_affinity"
        )
        aged = AdmissionCandidate(make_request("aged", arrival_time=1.0), 0, 0)
        warm = AdmissionCandidate(make_request("warm", arrival_time=9.0), 1, 80)

        chosen = policy.choose_admission([aged, warm], 10.0, 5.0)

        assert chosen is aged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
