# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUPTI benchmark for the FlashAttention KV-cache write kernels."""

import argparse
import math
import statistics

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm import _custom_ops as ops
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)

TOKEN_COUNTS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _run_benchmark(
    implementation: str,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    trials: int,
) -> tuple[float, list[float]]:
    num_blocks = max(64, math.ceil(num_tokens / block_size))
    key = torch.randn(
        num_tokens,
        num_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value = torch.randn_like(key)
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.long, device="cuda"
    )[:num_tokens]
    k_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    v_scale = torch.ones_like(k_scale)
    key_cache = torch.zeros(
        num_blocks,
        block_size,
        num_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value_cache = torch.zeros_like(key_cache)

    if implementation == "cuda":

        def run_kernel() -> None:
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                "auto",
                k_scale,
                v_scale,
            )

    else:

        def run_kernel() -> None:
            triton_reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                "auto",
                k_scale,
                v_scale,
            )

    run_kernel()
    torch.cuda.synchronize()
    flat_key_cache = key_cache.view(-1, num_heads, head_size)
    flat_value_cache = value_cache.view(-1, num_heads, head_size)
    torch.testing.assert_close(flat_key_cache[slot_mapping], key)
    torch.testing.assert_close(flat_value_cache[slot_mapping], value)

    for _ in range(25):
        run_kernel()
    torch.cuda.synchronize()

    trial_us = []
    for _ in range(trials):
        samples_ms = bench_gpu_time_with_cupti(
            run_kernel,
            dry_run_time_ms=10,
            repeat_time_ms=30,
            cold_l2_cache=True,
            use_cuda_graph=True,
        )
        trial_us.append(statistics.median(samples_ms) * 1e3)
    return statistics.median(trial_us), trial_us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation", choices=("cuda", "triton"), default="cuda"
    )
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--num-tokens", type=int, nargs="+", default=TOKEN_COUNTS)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.cuda.get_device_properties(0)
    print(
        f"gpu={device.name} capability={device.major}.{device.minor} "
        f"torch={torch.__version__} cuda={torch.version.cuda}"
    )
    print(
        f"implementation={args.implementation} dtype=bfloat16 layout=NHD "
        f"heads={args.num_heads} head_size={args.head_size} "
        f"block_size={args.block_size} cold_l2=true cuda_graph=true"
    )
    print("tokens median_us trial_us effective_gbps")
    for num_tokens in args.num_tokens:
        median_us, trial_us = _run_benchmark(
            args.implementation,
            num_tokens,
            args.num_heads,
            args.head_size,
            args.block_size,
            args.trials,
        )
        elements = num_tokens * args.num_heads * args.head_size
        minimum_bytes = 4 * elements * torch.bfloat16.itemsize + 8 * num_tokens
        gbps = minimum_bytes / (median_us * 1e3)
        trials = ",".join(f"{value:.3f}" for value in trial_us)
        print(f"{num_tokens} {median_us:.3f} {trials} {gbps:.1f}")


if __name__ == "__main__":
    main()
