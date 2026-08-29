# SPDX-License-Identifier: Apache-2.0
"""Measure CPU cost of vLLM preemption victim decisions."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.policy import create_decision_policy
from vllm.v1.core.sched.request_queue import SchedulingPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-counts",
        default="1,8,16,64,256",
        help="Comma-separated running-list sizes.",
    )
    parser.add_argument("--blocks-per-request", type=int, default=34)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--target-batch-ms", type=float, default=100.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Documents_xxy/phase6_scheduler_cpu.json"),
    )
    return parser.parse_args()


def calibrate_iterations(fn: Callable[[], Any], target_s: float) -> int:
    iterations = 1
    while iterations < 1_000_000:
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        elapsed_s = (time.perf_counter_ns() - start) / 1e9
        if elapsed_s >= target_s:
            break
        iterations *= max(2, min(10, int(target_s / max(elapsed_s, 1e-9))))
    return iterations


def measure(
    fn: Callable[[], Any], repeats: int, target_batch_ms: float
) -> dict[str, Any]:
    for _ in range(20):
        fn()
    iterations = calibrate_iterations(fn, target_batch_ms / 1000)
    samples_ns = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for _ in range(iterations):
                fn()
            samples_ns.append((time.perf_counter_ns() - start) / iterations)
    finally:
        if gc_enabled:
            gc.enable()
    ordered = sorted(samples_ns)
    return {
        "iterations_per_repeat": iterations,
        "samples_ns": samples_ns,
        "median_us": statistics.median(samples_ns) / 1000,
        "min_us": ordered[0] / 1000,
        "max_us": ordered[-1] / 1000,
    }


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    args = parse_args()
    candidate_counts = [int(value) for value in args.candidate_counts.split(",")]
    max_candidates = max(candidate_counts)
    prompt_tokens = args.blocks_per_request * args.block_size
    num_blocks = max_candidates * args.blocks_per_request + 128

    scheduler = create_scheduler(
        max_num_seqs=max_candidates,
        num_blocks=num_blocks,
        block_size=args.block_size,
        preemption_policy="reclaimable_aware",
    )
    requests = create_requests(
        max_candidates,
        num_tokens=prompt_tokens,
        max_tokens=1,
        block_size=args.block_size,
    )
    for index, request in enumerate(requests):
        computed_blocks, _, _ = scheduler.kv_cache_manager.get_computed_blocks(
            request
        )
        allocated = scheduler.kv_cache_manager.allocate_slots(
            request,
            prompt_tokens,
            new_computed_blocks=computed_blocks,
        )
        if allocated is None:
            raise RuntimeError(f"failed to allocate benchmark request {index}")
        request.num_computed_tokens = prompt_tokens + index

    default_policy = create_decision_policy(SchedulingPolicy.FCFS, "default")
    recompute_policy = create_decision_policy(
        SchedulingPolicy.FCFS, "recompute_aware"
    )
    results = []
    for candidate_count in candidate_counts:
        running = requests[:candidate_count]
        scheduler.running = running
        reclaimable_counts = [
            scheduler.kv_cache_manager.estimate_reclaimable_blocks(request)
            for request in running
        ]
        if reclaimable_counts != [args.blocks_per_request] * candidate_count:
            raise RuntimeError(
                f"unexpected reclaimability at N={candidate_count}: "
                f"{reclaimable_counts}"
            )

        measurements = {
            "default": measure(
                lambda running=running: default_policy.select_preemption_victim(
                    running
                ),
                args.repeats,
                args.target_batch_ms,
            ),
            "recompute_aware": measure(
                lambda running=running: recompute_policy.select_preemption_victim(
                    running
                ),
                args.repeats,
                args.target_batch_ms,
            ),
            "reclaimable_aware": measure(
                lambda: scheduler._select_reclaimable_preemption_victim(1),
                args.repeats,
                args.target_batch_ms,
            ),
        }
        reclaimable_us = measurements["reclaimable_aware"]["median_us"]
        recompute_us = measurements["recompute_aware"]["median_us"]
        results.append(
            {
                "candidate_count": candidate_count,
                "candidate_blocks": sum(reclaimable_counts),
                "measurements": measurements,
                "reclaimable_minus_recompute_us": reclaimable_us - recompute_us,
                "reclaimable_over_recompute": reclaimable_us / recompute_us,
            }
        )

    payload = {
        "metadata": {
            "git_head": git_head(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "candidate_counts": candidate_counts,
            "blocks_per_request": args.blocks_per_request,
            "block_size": args.block_size,
            "shortfall_blocks": 1,
            "repeats": args.repeats,
            "target_batch_ms": args.target_batch_ms,
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print("candidates  default_us  recompute_us  reclaimable_us  extra_us")
    for row in results:
        measurements = row["measurements"]
        print(
            f"{row['candidate_count']:>10}  "
            f"{measurements['default']['median_us']:>10.3f}  "
            f"{measurements['recompute_aware']['median_us']:>12.3f}  "
            f"{measurements['reclaimable_aware']['median_us']:>14.3f}  "
            f"{row['reclaimable_minus_recompute_us']:>8.3f}"
        )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
