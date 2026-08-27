# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for preemption victim selection policies (design doc 17.1).

Covers: default compatibility, strict user-priority constraint,
recompute-aware victim choice, and priority-beats-recompute-cost.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.core.sched.policy import create_decision_policy
from vllm.v1.core.sched.request_queue import SchedulingPolicy


def make_request(
    rid: str,
    priority: int = 0,
    arrival_time: float = 0.0,
    num_computed_tokens: int = 0,
    num_preemptions: int = 0,
) -> Any:
    return SimpleNamespace(
        request_id=rid,
        priority=priority,
        arrival_time=arrival_time,
        num_computed_tokens=num_computed_tokens,
        num_preemptions=num_preemptions,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
