# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUPTI benchmark for vLLM's fused add + RMSNorm kernel."""

import argparse
import statistics

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm import _custom_ops as ops

TOKEN_COUNTS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def _check_correctness(hidden_size: int, epsilon: float) -> None:
    torch.manual_seed(0)
    x = torch.randn(17, hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    expected_residual = x.float() + residual.float()
    variance = expected_residual.square().mean(dim=-1, keepdim=True)
    expected_x = (
        expected_residual * torch.rsqrt(variance + epsilon) * weight.float()
    ).to(torch.bfloat16)

    ops.fused_add_rms_norm(x, residual, weight, epsilon)
    torch.cuda.synchronize()
    torch.testing.assert_close(residual, expected_residual.to(torch.bfloat16))
    torch.testing.assert_close(x, expected_x, atol=2e-2, rtol=2e-2)


def _run_benchmark(
    num_tokens: int,
    hidden_size: int,
    epsilon: float,
    trials: int,
) -> tuple[float, list[float]]:
    # Zero is a fixed point of the in-place fused operation, so CUDA graph
    # replays time only the kernel instead of cloning/resetting its inputs.
    x = torch.zeros(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = torch.zeros_like(x)
    weight = torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)

    def run_kernel() -> None:
        ops.fused_add_rms_norm(x, residual, weight, epsilon)

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
    parser.add_argument("--hidden-size", type=int, default=3584)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--num-tokens", type=int, nargs="+", default=TOKEN_COUNTS)
    args = parser.parse_args()

    _check_correctness(args.hidden_size, args.epsilon)
    device = torch.cuda.get_device_properties(0)
    print(
        f"gpu={device.name} capability={device.major}.{device.minor} "
        f"torch={torch.__version__} cuda={torch.version.cuda}"
    )
    print(
        f"label={args.label} dtype=bfloat16 hidden_size={args.hidden_size} "
        "cold_l2=true cuda_graph=true correctness=pass"
    )
    print("tokens median_us trial_us logical_gbps")
    for num_tokens in args.num_tokens:
        median_us, trial_us = _run_benchmark(
            num_tokens, args.hidden_size, args.epsilon, args.trials
        )
        # Input/residual are each read and written, and weight is read once per
        # token. This is logical traffic, not a claim about DRAM transactions.
        logical_bytes = 5 * num_tokens * args.hidden_size * 2
        gbps = logical_bytes / (median_us * 1e3)
        trials = ",".join(f"{value:.3f}" for value in trial_us)
        print(f"{num_tokens} {median_us:.3f} {trials} {gbps:.1f}")


if __name__ == "__main__":
    main()
