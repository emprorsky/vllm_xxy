# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUPTI benchmark for the in-place vLLM rotary embedding kernel."""

import argparse
import statistics

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm import _custom_ops as ops

TOKEN_COUNTS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _reference(
    positions: torch.Tensor,
    tensor: torch.Tensor,
    cache: torch.Tensor,
) -> torch.Tensor:
    original_shape = tensor.shape
    tensor = tensor.view(positions.numel(), -1, cache.shape[-1])
    cos, sin = cache[positions].chunk(2, dim=-1)
    half = tensor.shape[-1] // 2
    left = tensor[..., :half].float()
    right = tensor[..., half:].float()
    cos = cos[:, None, :].float()
    sin = sin[:, None, :].float()
    return (
        torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)
        .to(tensor.dtype)
        .reshape(original_shape)
    )


def _check_correctness(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> None:
    torch.manual_seed(0)
    positions = torch.randperm(256, device="cuda", dtype=torch.long)[:17]
    qkv = torch.randn(
        17,
        (num_q_heads + 2 * num_kv_heads) * head_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    query, key, _ = qkv.split(
        [
            num_q_heads * head_size,
            num_kv_heads * head_size,
            num_kv_heads * head_size,
        ],
        dim=-1,
    )
    cache = torch.randn(256, head_size, device="cuda", dtype=torch.bfloat16)
    expected_query = _reference(positions, query, cache)
    expected_key = _reference(positions, key, cache)

    ops.rotary_embedding(positions, query, key, head_size, cache, True)
    torch.cuda.synchronize()
    torch.testing.assert_close(query, expected_query, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(key, expected_key, atol=2e-2, rtol=2e-2)


def _run_benchmark(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    trials: int,
) -> tuple[float, list[float]]:
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.long)
    qkv = torch.zeros(
        num_tokens,
        (num_q_heads + 2 * num_kv_heads) * head_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    query, key, _ = qkv.split(
        [
            num_q_heads * head_size,
            num_kv_heads * head_size,
            num_kv_heads * head_size,
        ],
        dim=-1,
    )
    cache = torch.randn(
        max(num_tokens, 16), head_size, device="cuda", dtype=torch.bfloat16
    )

    def run_kernel() -> None:
        ops.rotary_embedding(positions, query, key, head_size, cache, True)

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
    parser.add_argument("--label", default="current")
    parser.add_argument("--num-q-heads", type=int, default=28)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--num-tokens", type=int, nargs="+", default=TOKEN_COUNTS)
    args = parser.parse_args()

    _check_correctness(args.num_q_heads, args.num_kv_heads, args.head_size)
    device = torch.cuda.get_device_properties(0)
    print(
        f"gpu={device.name} capability={device.major}.{device.minor} "
        f"torch={torch.__version__} cuda={torch.version.cuda}"
    )
    print(
        f"label={args.label} dtype=bfloat16 q_heads={args.num_q_heads} "
        f"kv_heads={args.num_kv_heads} head_size={args.head_size} "
        "style=neox cold_l2=true cuda_graph=true correctness=pass"
    )
    print("tokens median_us trial_us logical_gbps")
    for num_tokens in args.num_tokens:
        median_us, trial_us = _run_benchmark(
            num_tokens,
            args.num_q_heads,
            args.num_kv_heads,
            args.head_size,
            args.trials,
        )
        elements = num_tokens * (args.num_q_heads + args.num_kv_heads) * args.head_size
        logical_bytes = 4 * elements + num_tokens * (8 + 2 * args.head_size)
        gbps = logical_bytes / (median_us * 1e3)
        trials = ",".join(f"{value:.3f}" for value in trial_us)
        print(f"{num_tokens} {median_us:.3f} {trials} {gbps:.1f}")


if __name__ == "__main__":
    main()
