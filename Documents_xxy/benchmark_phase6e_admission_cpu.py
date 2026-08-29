# SPDX-License-Identifier: Apache-2.0
"""Measure CPU cost of bounded cache-affinity admission decisions."""

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

from tests.v1.core.utils import EOS_TOKEN_ID, create_scheduler
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-counts",
        default="2,4,8,16",
        help="Comma-separated cache-affinity window sizes.",
    )
    parser.add_argument("--prefix-blocks", type=int, default=3)
    parser.add_argument("--suffix-blocks", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--target-batch-ms", type=float, default=100.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Documents_xxy/phase6e_admission_cpu.json"),
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


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    max_tokens: int,
) -> Request:
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=max_tokens)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def make_output(scheduler: Any) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[request.request_id for request in scheduler.running],
        req_id_to_index={
            request.request_id: index for index, request in enumerate(scheduler.running)
        },
        sampled_token_ids=[[1000]] * len(scheduler.running),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def create_benchmark_scheduler(
    candidate_count: int,
    prefix_tokens: int,
    suffix_tokens: int,
    block_size: int,
    admission_policy: str,
) -> Any:
    scheduler = create_scheduler(
        max_num_seqs=candidate_count,
        max_num_batched_tokens=max(8192, candidate_count * prefix_tokens),
        num_blocks=candidate_count * (prefix_tokens // block_size + 2) + 32,
        block_size=block_size,
        enable_prefix_caching=True,
        admission_policy=admission_policy,
        kv_aware_candidate_window=candidate_count,
        kv_aware_aging_threshold_s=10.0,
    )
    if admission_policy == "cache_affinity":
        seeds = [
            make_request(
                f"seed-{index}",
                [index + 1] * prefix_tokens,
                block_size,
                max_tokens=1,
            )
            for index in range(1, candidate_count)
        ]
        for seed in seeds:
            scheduler.add_request(seed)
        output = scheduler.schedule()
        scheduler.update_from_output(output, make_output(scheduler))
        if scheduler.running:
            raise RuntimeError("prefix seed requests did not finish")

    candidates = [
        make_request(
            f"candidate-{index}",
            [index + 1] * prefix_tokens + [10_000 + index] * suffix_tokens,
            block_size,
            max_tokens=16,
        )
        for index in range(candidate_count)
    ]
    for request in candidates:
        scheduler.add_request(request)
    return scheduler


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    args = parse_args()
    init_none_hash(sha256)
    candidate_counts = [int(value) for value in args.candidate_counts.split(",")]
    prefix_tokens = args.prefix_blocks * args.block_size
    suffix_tokens = args.suffix_blocks * args.block_size
    results = []

    for candidate_count in candidate_counts:
        affinity = create_benchmark_scheduler(
            candidate_count,
            prefix_tokens,
            suffix_tokens,
            args.block_size,
            "cache_affinity",
        )
        default = create_benchmark_scheduler(
            candidate_count,
            prefix_tokens,
            suffix_tokens,
            args.block_size,
            "default",
        )
        affinity_queue = affinity.waiting
        affinity_base = affinity_queue.peek_request()
        default_queue = default.waiting
        default_base = default_queue.peek_request()
        context = affinity.scheduling_feature_context
        if context is None:
            raise RuntimeError("cache-affinity feature context was not created")
        now_s = time.time()

        def select_default(
            scheduler=default,
            queue=default_queue,
            base=default_base,
            timestamp=now_s,
        ) -> Any:
            return scheduler._select_cache_affinity_request(
                queue, base, set(), timestamp
            )

        def select_hot(
            scheduler=affinity,
            queue=affinity_queue,
            base=affinity_base,
            timestamp=now_s,
        ) -> Any:
            return scheduler._select_cache_affinity_request(
                queue, base, set(), timestamp
            )

        def select_cold(ctx=context, select=select_hot) -> Any:
            ctx.invalidate_kv_features()
            return select()

        select_cold()
        selected = select_hot()[1]
        if selected is affinity_base:
            raise RuntimeError("benchmark did not activate cache-affinity reordering")

        measurements = {
            "default_fast_path": measure(
                select_default, args.repeats, args.target_batch_ms
            ),
            "cache_affinity_hot": measure(
                select_hot, args.repeats, args.target_batch_ms
            ),
            "cache_affinity_cold": measure(
                select_cold, args.repeats, args.target_batch_ms
            ),
        }
        results.append(
            {
                "candidate_count": candidate_count,
                "measurements": measurements,
                "cold_minus_default_us": (
                    measurements["cache_affinity_cold"]["median_us"]
                    - measurements["default_fast_path"]["median_us"]
                ),
                "cold_minus_hot_us": (
                    measurements["cache_affinity_cold"]["median_us"]
                    - measurements["cache_affinity_hot"]["median_us"]
                ),
            }
        )

    payload = {
        "metadata": {
            "git_head": git_head(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "candidate_counts": candidate_counts,
            "prefix_blocks": args.prefix_blocks,
            "suffix_blocks": args.suffix_blocks,
            "block_size": args.block_size,
            "repeats": args.repeats,
            "target_batch_ms": args.target_batch_ms,
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print("candidates  default_us  hot_us  cold_us  cold_extra_us")
    for row in results:
        measurements = row["measurements"]
        print(
            f"{row['candidate_count']:>10}  "
            f"{measurements['default_fast_path']['median_us']:>10.3f}  "
            f"{measurements['cache_affinity_hot']['median_us']:>7.3f}  "
            f"{measurements['cache_affinity_cold']['median_us']:>8.3f}  "
            f"{row['cold_minus_default_us']:>13.3f}"
        )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
