"""Summarize retained evidence; never treat failed runs as zero-valued successes."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from contextlib import suppress
from pathlib import Path

SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([^\s]+)")


def samples(text):
    values = Counter()
    for line in text.splitlines():
        match = SAMPLE.match(line)
        if match and not match[1].endswith("_created"):
            with suppress(ValueError):
                values[match[1]] += float(match[2])
    return values


def counters(path):
    source = json.loads(path.read_text())
    result = defaultdict(float)
    for endpoint, after in source["after"].items():
        before = samples(source["before"].get(endpoint, ""))
        for name, value in samples(after).items():
            if name.endswith(("_total", "_count", "_sum")):
                result[name] += value - before.get(name, 0)
    return dict(result)


def collect(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    row = {"run": directory.name, "status": manifest["status"]}
    environment = manifest["config"].get("env", {})
    cpu = environment.get("LMCACHE_LOCAL_CPU", "").lower() == "true"
    v3 = environment.get("LMCACHE_USE_GPU_CONNECTOR_V3", "").lower() == "true"
    row["transfer_layout"] = (
        "packed_v3" if cpu and v3 else "legacy_v2" if cpu else "none"
    )
    row["quarantined"] = cpu and not v3
    if manifest["status"] != "completed":
        row["error"] = manifest.get("error", "")
        return row
    data = json.loads((directory / "replay/result.json").read_text())
    row.update(
        {
            key: data[key]
            for key in [
                "successful",
                "elapsed_s",
                "output_tok_s",
                "request_goodput",
                "qualified_requests",
            ]
        }
    )
    row["trace_sha256"] = data["trace_sha256"]
    row["gpus"] = len(manifest["config"]["workers"])
    for key in ["ttft_s", "tpot_s", "e2e_s"]:
        for quantile, value in data[key].items():
            row[f"{key}_{quantile}"] = value
    metric = counters(directory / "replay/metrics.json")
    (directory / "counter_deltas.json").write_text(json.dumps(metric, indent=2))
    for name in [
        "num_preemptions",
        "kv_preemption_resume_recompute_tokens",
        "kv_preemption_preempted_computed_tokens",
        "prefix_cache_queries",
        "prefix_cache_hits",
        "external_prefix_cache_queries",
        "external_prefix_cache_hits",
    ]:
        row[name] = metric.get(f"vllm:{name}_total", 0)
    queries = row["prefix_cache_queries"]
    row["gpu_prefix_hit_ratio"] = (
        row["prefix_cache_hits"] / queries if queries else None
    )
    worker_logs = [
        p.read_text(errors="replace") for p in sorted(directory.glob("server_*.log"))
    ]
    logs = "\n".join(worker_logs)
    row["gpu_kv_tokens_per_worker"] = [
        int(match[-1].replace(",", "")) if match else None
        for text in worker_logs
        for match in [re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)]
    ]
    # These are completed data movement logs, not repeated lookup estimates.
    row["cpu_stored_tokens"] = sum(
        int(n) for n in re.findall(r"Stored (\d+) out of total", logs)
    )
    row["cpu_retrieved_tokens"] = sum(
        int(n) for n in re.findall(r"Retrieved (\d+) out of", logs)
    )
    decisions = []
    eviction_chunks = 0
    disk_io = Counter()
    for path in (directory / "admission").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "decision":
                decisions.append(event)
            elif event.get("event") == "capacity_eviction":
                eviction_chunks += event["removed_chunks"]
            elif event.get("event") in {"disk_read", "disk_write"}:
                kind = event["event"]
                disk_io[kind + "_bytes"] += event["bytes"]
                disk_io[kind + "_calls"] += 1
                disk_io[kind + "_seconds_sum"] += event["elapsed_s"]
    row["admission_decisions"] = len(decisions)
    row["admission_skips"] = sum(not e["save"] for e in decisions)
    row["admission_pressure"] = sum(e["pressure"] for e in decisions)
    row["cpu_eviction_chunks_observed"] = eviction_chunks
    row.update(disk_io)
    path = directory / "router_decisions.jsonl"
    if path.exists():
        events = [json.loads(x) for x in path.read_text().splitlines()]
        for kind in ["dispatch", "prefill", "decode"]:
            selected = [e for e in events if e.get("event") == kind]
            if selected:
                row[f"{kind}_workers"] = dict(Counter(e["worker"] for e in selected))
        row["router_errors"] = sum(e.get("event") == "error" for e in events)
    return row


def summarize(root, output):
    rows = []
    for manifest in sorted(root.glob("*/manifest.json")):
        try:
            rows.append(collect(manifest.parent))
        except (ValueError, KeyError, OSError) as exc:
            rows.append(
                {
                    "run": manifest.parent.name,
                    "status": "summary_error",
                    "error": str(exc),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs.json").write_text(json.dumps(rows, indent=2))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "completed":
            groups[re.sub(r"_rep\d+$", "", row["run"])].append(row)
    lines = [
        "# 实验结果（自动汇总）",
        "",
        (
            "仅包含 completed 运行；中位数，单次仅供筛选。"
            "不同 trace/容量/硬件配置不可直接作策略 A/B。"
        ),
        "",
        (
            "旧 V2 CPU offload 的打包 K/V 复制错误已确认；隔离结果不得用于收益结论。"
            "packed_v3 为修复后重跑；completed 不代表全面质量验收。"
        ),
        "",
        (
            "| 配置 | n | tok/s | TTFT P95(s) | TPOT P95(ms) |"
            " SLO goodput(req/s) | CPU读tokens | 保存过滤 | CPU布局 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    aggregate = {}
    for name, group in groups.items():
        metrics = {}
        for key in [
            "output_tok_s",
            "ttft_s_p95",
            "tpot_s_p95",
            "request_goodput",
            "cpu_retrieved_tokens",
            "admission_skips",
        ]:
            values = [r[key] for r in group]
            metrics[key] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        aggregate[name] = {
            "n": len(group),
            "metrics": metrics,
            "quarantined": any(r["quarantined"] for r in group),
            "transfer_layout": sorted({r["transfer_layout"] for r in group}),
        }
        med = {key: value["median"] for key, value in metrics.items()}
        lines.append(
            f"| {name} | {len(group)} | {med['output_tok_s']:.2f} | {med['ttft_s_p95']:.3f} | {med['tpot_s_p95'] * 1000:.2f} | {med['request_goodput']:.3f} | {med['cpu_retrieved_tokens']:.0f} | {med['admission_skips']:.0f} | {'隔离' if aggregate[name]['quarantined'] else ','.join(aggregate[name]['transfer_layout'])} |"  # noqa: E501
        )
    failures = [r for r in rows if r["status"] != "completed"]
    if failures:
        lines += ["", "## 失败或仍在运行", ""] + [
            f"- {r['run']}: {r['status']} {r.get('error', '')}" for r in failures
        ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    print(
        json.dumps(
            {
                "runs": len(rows),
                "completed": sum(r["status"] == "completed" for r in rows),
                "output": str(output),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)


if __name__ == "__main__":
    main()
