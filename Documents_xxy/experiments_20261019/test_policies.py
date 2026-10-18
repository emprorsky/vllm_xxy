"""Behavior checks independent of CUDA and optional serving dependencies."""

from .policies import (
    PlacementPolicy,
    PrefixDirectory,
    TierTelemetry,
    WriteAdmission,
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
