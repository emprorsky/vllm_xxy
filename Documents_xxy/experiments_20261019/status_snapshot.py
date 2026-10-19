"""Produce a compact, read-only progress page from retained run manifests."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def snapshot(root):
    counts = Counter()
    active = []
    for path in sorted((root / "runs").glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        counts[manifest["status"]] += 1
        if manifest["status"] not in {"starting", "replaying"}:
            continue
        requests = path.parent / "replay/requests.jsonl"
        rows = []
        if requests.exists():
            for line in requests.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        trace = Path(manifest["config"]["trace"])
        expected = sum(1 for line in trace.read_text().splitlines() if line.strip())
        active.append(
            {
                "run": path.parent.name,
                "status": manifest["status"],
                "gpus": [w["gpu"] for w in manifest["config"]["workers"]],
                "elapsed_s": round(time.time() - manifest["started_at"], 1),
                "completed_responses": len(rows),
                "successful_responses": sum(row["success"] for row in rows),
                "expected_responses": expected,
            }
        )
    checks = {}
    for name in ["cpu_prefix_fidelity_v3_long", "quality_bf16", "quality_awq"]:
        path = root / "checks" / name / "manifest.json"
        if path.exists():
            value = json.loads(path.read_text())
            checks[name] = value.get("result", value.get("status"))
    result = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "run_counts": dict(counts),
        "active": active,
        "checks": checks,
    }
    (root / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 实验当前进度",
        "",
        result["timestamp_utc"],
        "",
        f"运行状态累计：{dict(counts)}（包含保留的历史失败和隔离轮）",
        "",
        "| 正在执行 | GPU | 阶段 | 已响应/预期 | 已用秒 |",
        "|---|---|---|---:|---:|",
    ]
    for row in active:
        lines.append(
            f"| {row['run']} | {row['gpus']} | {row['status']} | "
            f"{row['completed_responses']}/{row['expected_responses']} | "
            f"{row['elapsed_s']} |"
        )
    lines += [
        "",
        "CPU V3 恢复与 BF16/AWQ 小规模代码探测状态见 status.json。",
        "旧 V2 offload 性能结果隔离；单轮差异不作为稳定收益。",
    ]
    (root / "status.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=7200)
    args = parser.parse_args()
    deadline = time.monotonic() + args.seconds
    while True:
        print(json.dumps(snapshot(args.root), ensure_ascii=False), flush=True)
        if args.interval <= 0 or time.monotonic() >= deadline:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
