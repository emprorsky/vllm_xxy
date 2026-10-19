"""Behavior checks independent of CUDA and optional serving dependencies."""

from .policies import (
    MemoryDemand,
    PlacementPolicy,
    PrefixDirectory,
    TierTelemetry,
    WriteAdmission,
    memory_queue_wait,
    prefix_keys,
)


def test_capacity_conditioning_preserves_extensions_and_releases_filter():
    policy = WriteAdmission(1024, 8, 16, write_threshold=2)
    full = TierTelemetry(8, 8, 5, 1)
    assert policy.decide("r0", "a", 256, 0, full, now=1).save
    decision = policy.decide("r1", "b", 256, 0, full, now=1)
    assert decision.pressure and not decision.save
    assert policy.decide("r2", "a", 256, 240, full, now=1).save
    empty = TierTelemetry(1, 8, 0, 2)
    assert policy.decide("r3", "c", 256, 0, empty, now=2).save


def test_repeat_lookup_does_not_change_decision_or_count_session_twice():
    policy = WriteAdmission(1024, 8, 16, mode="fixed", write_threshold=2)
    first = policy.decide("r", "a", 256, 0, None, now=1)
    assert not first.save
    assert policy.decide("r", "a", 256, 256, None, now=2) is first
    assert len(policy.sessions) == 1
    policy.finish("r")
    assert policy.decide("r", "a", 256, 256, None, now=2).save


def test_stale_telemetry_fails_open_and_tool_gap_sessions_expire():
    policy = WriteAdmission(16, 8, 16, window_s=10)
    telemetry = TierTelemetry(8, 8, 5, 0)
    policy.decide("r1", "a", 256, 0, telemetry, now=0)
    stale = policy.decide("r2", "b", 256, 0, telemetry, now=6)
    assert stale.above_capacity and stale.save and not stale.telemetry_fresh
    decision = policy.decide("r3", "c", 256, 0, None, now=20)
    assert decision.active_sessions == 1


def test_prefix_identity_depends_on_all_ancestors_and_expires():
    first = prefix_keys([1, 2, 3, 4], 2)
    changed = prefix_keys([9, 2, 3, 4], 2)
    assert first[1] != changed[1]
    directory = PrefixDirectory(4, block_size=2, ttl_s=5)
    directory.observe(first, 0)
    assert directory.match(first, 1) == 4
    assert directory.match(changed, 1) == 0
    assert directory.match(first, 6) == 0


def test_atomic_inflight_reservations_distribute_a_burst_and_release_once():
    policy = PlacementPolicy(2, "work", block_size=2)
    assert policy.choose("a", [1, 2, 3, 4], 10, now=1)[0] == 0
    assert policy.choose("b", [1, 2, 3, 4], 10, now=1)[0] == 1
    policy.finish("a", [1, 2, 3, 4], False, now=2)
    policy.finish("a", [1, 2, 3, 4], True, now=2)
    assert len(policy.reservations) == 1
    assert len(policy.directories[0].blocks) == 0


def test_eviction_penalty_avoids_destroying_a_hot_prefix():
    policy = PlacementPolicy(
        2,
        "cost",
        capacity_tokens=4,
        block_size=2,
        prefill_ms_per_token=1,
        decode_ms_per_token=0,
    )
    hot = prefix_keys([1, 2, 3, 4], 2)
    for _ in range(5):
        policy.directories[0].observe(hot, 1)
    worker, scores = policy.choose("r", [7, 8, 9, 10], 1, now=2)
    assert worker == 1
    assert scores[0].eviction_ms > scores[1].eviction_ms


def test_remaining_work_uses_monotonic_observed_tokens_and_releases_progress():
    policy = PlacementPolicy(
        2, "work_remaining", prefill_ms_per_token=1, decode_ms_per_token=10
    )
    policy.choose("old", [1] * 32, 10, now=1)
    assert policy.reservations["old"] == (0, 132)
    policy.observe_output("old", 9)
    assert policy.reservations["old"] == (0, 10)
    policy.observe_output("old", 3)
    assert policy.reservations["old"] == (0, 10)
    policy.choose("other", [2] * 32, 1, now=1)
    worker, scores = policy.choose("new", [3] * 32, 10, now=1)
    assert worker == 0 and scores[0].queue_ms == 10
    policy.observe_output("old", 999)
    assert policy.reservations["old"] == (0, 0)
    policy.finish("old", [1] * 32, False, now=2)
    policy.observe_output("old", 9)
    assert "old" not in policy.output_progress


def test_threshold_feedback_balances_unequal_work_at_equal_request_counts():
    import copy

    policy = PlacementPolicy(2, "rr", decode_ms_per_token=10, balance_abs=1)
    tokens = [1] * 32
    for index in range(8):
        policy.choose(str(index), tokens, 128 if index % 2 == 0 else 32, now=1)
    for directory in policy.directories:
        directory.observe(prefix_keys(tokens, directory.block_size), 1)
    baseline = copy.deepcopy(policy)
    baseline.policy = "threshold"
    policy.policy = "threshold_remaining"
    assert baseline.choose("next", tokens, 64, now=2)[0] == 0
    worker, scores = policy.choose("next", tokens, 64, now=2)
    assert scores[0].inflight == scores[1].inflight == 4
    assert worker == 1


def test_profiled_reservations_use_measured_cold_hit_and_decode_costs():
    profile = {
        "decode_ms": 20,
        "bins": [{"input_tokens": 32, "cold_ms": 100, "cached_ms": 10}],
    }
    policy = PlacementPolicy(2, "work_remaining", cost_profile=profile)
    tokens = [1] * 32
    policy.directories[1].observe(prefix_keys(tokens, 16), 0)
    worker, scores = policy.choose("r", tokens, 10, now=1)
    assert worker == 1
    assert scores[0].prefill_ms == 100 and scores[1].prefill_ms == 10
    assert policy.reservations["r"] == (1, 210)
    policy.observe_output("r", 5)
    assert policy.reservations["r"] == (1, 100)


def test_memory_queue_counts_shared_prefix_once_and_waits_for_capacity():
    shared = ("root", "shared")
    jobs = [MemoryDemand(4, shared, 100), MemoryDemand(2, shared, 20)]
    # Eight tokens fit concurrently; the new private suffix waits for the
    # shorter request to release its two tokens, not the full 120 ms sum.
    assert memory_queue_wait(jobs, MemoryDemand(2, shared, 0), 8, 1, 4) == 20
    assert memory_queue_wait(jobs, MemoryDemand(2, shared, 0), 100, 1, 1) == 120
    assert memory_queue_wait([], MemoryDemand(10, (), 0), 8, 1, 4) == 1e12


def test_generated_kv_is_not_freed_when_output_budget_decreases():
    policy = PlacementPolicy(2, "memory_remaining", capacity_tokens=128, max_batch=2)
    policy.choose("r", [1] * 32, 16, now=1)
    memory = policy.memory_demands["r"]
    policy.observe_output("r", 15)
    assert policy.memory_demands["r"] == memory
    assert policy.reservations["r"][1] == policy.decode_ms_per_token
    policy.finish("r", [1] * 32, False, now=2)
    assert not policy.memory_demands and not policy.output_progress
