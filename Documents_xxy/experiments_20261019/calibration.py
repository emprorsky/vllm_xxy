"""Measure a small, disjoint cold/APC profile at concurrency one."""

import json
import statistics
import time

import httpx


async def calibrate(endpoint, output):
    records = []
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        for length in [128, 512, 1024, 2048, 3072]:
            for repeat in range(4):
                # New first block prevents accidental cross-case prefix reuse.
                root = 4000 + length + repeat
                prompt = [root] * 16 + [1000 + (i % 97) for i in range(length - 16)]
                for phase in ["cold", "cached"]:
                    start = time.perf_counter()
                    timestamps, usage, tokens = [], {}, 0
                    done = False
                    async with client.stream(
                        "POST",
                        endpoint + "/v1/completions",
                        json={
                            "model": "experiment",
                            "prompt": prompt,
                            "max_tokens": 32,
                            "ignore_eos": True,
                            "temperature": 0,
                            "stream": True,
                            "return_token_ids": True,
                            "stream_options": {"include_usage": True},
                        },
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            if line == "data: [DONE]":
                                done = True
                                continue
                            message = json.loads(line[6:])
                            if message.get("usage"):
                                usage = message["usage"]
                            for choice in message.get("choices", []):
                                ids = choice.get("token_ids") or []
                                if ids:
                                    timestamps.append(time.perf_counter())
                                    tokens += len(ids)
                    if not done or tokens != 32 or len(timestamps) < 2:
                        raise ValueError("incomplete profile stream")
                    cached = usage.get("prompt_tokens_details", {}).get(
                        "cached_tokens", 0
                    )
                    if (phase == "cold" and cached) or (
                        phase == "cached" and cached < length - 16
                    ):
                        raise ValueError(
                            f"unexpected profile cache coverage: {phase} {cached}"
                        )
                    records.append(
                        {
                            "length": length,
                            "repeat": repeat,
                            "phase": phase,
                            "warmup_excluded": repeat == 0,
                            "cached_tokens": cached,
                            "ttft_ms": (timestamps[0] - start) * 1000,
                            "tpot_ms": (timestamps[-1] - timestamps[0]) * 1000 / 31,
                        }
                    )
    bins = []
    for length in [128, 512, 1024, 2048, 3072]:
        values = [
            r for r in records if r["length"] == length and not r["warmup_excluded"]
        ]
        bins.append(
            {
                "input_tokens": length,
                "cold_ms": statistics.median(
                    r["ttft_ms"] for r in values if r["phase"] == "cold"
                ),
                "cached_ms": statistics.median(
                    r["ttft_ms"] for r in values if r["phase"] == "cached"
                ),
            }
        )
    result = {
        "scope": "C1_disjoint_tokens_native_Triton_BF16_3B_not_batch_predictor",
        "bins": bins,
        "decode_ms": statistics.median(
            r["tpot_ms"] for r in records if not r["warmup_excluded"]
        ),
        "records": records,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return {
        "passed": True,
        "profile": str(output),
        "bins": bins,
        "decode_ms": result["decode_ms"],
    }


def prefill_cost(profile, tokens, cached):
    bins = profile["bins"]
    if not bins:
        raise ValueError("empty cost profile")
    # Nearest length bucket; partial hits interpolate between measured endpoints.
    bucket = min(bins, key=lambda row: abs(row["input_tokens"] - tokens))
    warm = max(bucket["cached_ms"], 0)
    cold = max(bucket["cold_ms"], warm)
    fraction = min(max(cached / max(tokens, 1), 0), 1)
    return warm + (cold - warm) * (1 - fraction)
