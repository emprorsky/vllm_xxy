"""Dependency-preserving replay with token-level streaming observations."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx

from .policies import percentile


async def send_one(client, endpoint, row, model, output_file):
    start = time.perf_counter()
    first = None
    timestamps = []
    generated = []
    text = ""
    usage = {}
    error = None
    worker = None
    done = False
    body = {
        "model": model,
        "prompt": row["prompt_token_ids"],
        "max_tokens": row.get("requested_output_budget", 64),
        "temperature": 0,
        "ignore_eos": row.get("ignore_eos", True),
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "kv_transfer_params": {
            "experiment.session_id": row["session_id"],
            "experiment.prompt_tokens": len(row["prompt_token_ids"]),
        },
    }
    try:
        async with client.stream(
            "POST",
            endpoint.rstrip("/") + "/v1/completions",
            json=body,
            headers={"X-Request-ID": row["request_id"]},
        ) as response:
            worker = response.headers.get("x-worker-index")
            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {(await response.aread())[:1000]!r}"
                )
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    continue
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("text", "")
                    ids = choice.get("token_ids") or []
                    if delta or ids:
                        timestamp = time.perf_counter()
                        first = timestamp if first is None else first
                        timestamps.append(timestamp)
                        generated.extend(ids)
                        text += delta
        if not done:
            raise RuntimeError("stream ended without DONE")
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        error = str(exc)
    end = time.perf_counter()
    tokens = usage.get("completion_tokens", len(generated))
    result = {
        "request_id": row["request_id"],
        "session_id": row["session_id"],
        "turn_id": row.get("turn_id", 0),
        "worker": worker,
        "success": error is None,
        "error": error,
        "input_tokens": len(row["prompt_token_ids"]),
        "output_tokens": tokens,
        "ttft_s": first - start if first is not None else None,
        "e2e_s": end - start,
        "tpot_s": (end - first) / (tokens - 1)
        if first is not None and tokens > 1
        else None,
        "chunk_itl_s": [b - a for a, b in zip(timestamps, timestamps[1:])],
        "output_token_ids": generated,
        "text": text,
        "usage": usage,
        "clock_start": start,
        "clock_end": end,
    }
    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


async def replay(args):
    raw = args.trace.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not rows:
        raise ValueError("trace has no requests")
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("trace request IDs must be unique")
    groups = defaultdict(list)
    for row in rows:
        groups[row["session_id"]].append(row)
    for session in groups.values():
        session.sort(key=lambda row: row.get("turn_id", 0))
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("refusing to overwrite completed run")
    semaphore = asyncio.Semaphore(args.concurrency)
    results = []
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        # Short disjoint prefixes warm the execution path without storing a full
        # LMCache chunk or warming the evaluation's shared prefixes.
        for endpoint in args.metrics_endpoints or [args.endpoint]:
            for index in range(getattr(args, "warmup_requests", 2)):
                warmup = {
                    "model": args.model,
                    "prompt": [1000 + index] + rows[0]["prompt_token_ids"][:32],
                    "max_tokens": 4,
                    "temperature": 0,
                    "kv_transfer_params": {"lmcache.skip_save": True},
                }
                response = await client.post(
                    endpoint.rstrip("/") + "/v1/completions", json=warmup
                )
                response.raise_for_status()
        before = {}
        for endpoint in args.metrics_endpoints or [args.endpoint]:
            try:
                before[endpoint] = (
                    await client.get(endpoint.rstrip("/") + "/metrics")
                ).text
            except httpx.HTTPError as exc:
                before[endpoint] = str(exc)
        start = time.perf_counter()
        with (args.output / "requests.jsonl").open("w", buffering=1) as out:

            async def session_run(session):
                async with semaphore:
                    for row in session:
                        await asyncio.sleep(
                            row.get("observed_gap_s", 0) * args.gap_scale
                        )
                        result = await send_one(
                            client, args.endpoint, row, args.model, out
                        )
                        results.append(result)
                        if not result["success"]:
                            break

            await asyncio.gather(*(session_run(session) for session in groups.values()))
        elapsed = time.perf_counter() - start
        after = {}
        for endpoint in before:
            try:
                after[endpoint] = (
                    await client.get(endpoint.rstrip("/") + "/metrics")
                ).text
            except httpx.HTTPError as exc:
                after[endpoint] = str(exc)
    (args.output / "metrics.json").write_text(
        json.dumps({"before": before, "after": after})
    )
    good = [r for r in results if r["success"]]
    qualified = [
        r
        for r in good
        if r["ttft_s"] is not None
        and r["ttft_s"] <= args.slo_ttft
        and (r["tpot_s"] is None or r["tpot_s"] <= args.slo_tpot)
    ]
    summary = {
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "requests_expected": len(rows),
        "requests_attempted": len(results),
        "successful": len(good),
        "elapsed_s": elapsed,
        "measurement": "dependency_preserving_full_makespan",
        "output_tok_s": sum(r["output_tokens"] for r in good) / elapsed,
        "request_goodput": len(qualified) / elapsed,
        "slo_ttft_s": args.slo_ttft,
        "slo_tpot_s": args.slo_tpot,
        "qualified_requests": len(qualified),
        "warmup_requests_per_backend": getattr(args, "warmup_requests", 2),
        "qualified_output_tok_s": sum(r["output_tokens"] for r in qualified) / elapsed,
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    for name in ["ttft_s", "tpot_s", "e2e_s"]:
        values = [r[name] for r in good if r[name] is not None]
        summary[name] = {f"p{q}": percentile(values, q) for q in [50, 95, 99]}
    (args.output / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--metrics-endpoints", nargs="*")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="experiment")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--gap-scale", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--slo-ttft", type=float, default=5)
    parser.add_argument("--slo-tpot", type=float, default=0.1)
    parser.add_argument("--warmup-requests", type=int, default=2)
    args = parser.parse_args()
    if args.concurrency < 1 or args.gap_scale < 0:
        parser.error("invalid concurrency or gap scale")
    asyncio.run(replay(args))


if __name__ == "__main__":
    main()
