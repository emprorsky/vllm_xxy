"""Export declared comparisons with every retained repetition and safety gate."""

import argparse
import json
import statistics
from pathlib import Path

from .summarize import collect

COMPARISONS = [
    ("CPU 保存", "V3_O_seq2_none_cpu0p5", "V3_O_seq2_conditioned_cpu0p5"),
    ("CPU 关闭参照", "V3_O_seq2_off_cpu0p5", "V3_O_seq2_conditioned_cpu0p5"),
    ("CPU 独立任务集", "VALID_HOLDOUT_O_none", "VALID_HOLDOUT_O_conditioned"),
    ("本地 BF16 自然容量", "CONF_QNAT_BF16_L0", "CONF_QNAT_BF16_L1"),
    ("本地 AWQ 自然容量", "CONF_QNAT_AWQ_L0", "CONF_QNAT_AWQ_L1"),
    ("本地热点机制", "FIX_LHOT_L0", "FIX_LHOT_L1"),
    ("本地独立任务集", "VALID_HOLDOUT_L0", "VALID_HOLDOUT_L1"),
    ("aging 5/10", "CONF_FAIR_L1_age10", "CONF_FAIR_L1_age5"),
    (
        "剩余工作路由",
        "CONF_REMAIN_public_coding_threshold",
        "CONF_REMAIN_public_coding_work_remaining",
    ),
    (
        "三层完整组合",
        "V3_DR_threshold_none_L0",
        "V3_DR_work_remaining_conditioned_L1",
    ),
    (
        "路由×CPU组合",
        "V3_DR_threshold_none_L0",
        "V3_DR_work_remaining_conditioned_L0",
    ),
    (
        "CPU改造之上增加路由",
        "V3_DR_threshold_conditioned_L0",
        "V3_DR_work_remaining_conditioned_L0",
    ),
    (
        "R/O之上增加本地调度",
        "V3_DR_work_remaining_conditioned_L0",
        "V3_DR_work_remaining_conditioned_L1",
    ),
]
for trace in ["public_coding", "public_coding_8k"]:
    COMPARISONS.append(
        (
            "内存/批处理路由 " + trace,
            f"CONF_MEMORY_{trace}_threshold",
            f"CONF_MEMORY_{trace}_memory_remaining",
        )
    )
for trace in ["holdout_coding", "holdout_coding_8k"]:
    COMPARISONS.append(
        (
            "内存路由独立任务集 " + trace,
            f"VALID_HOLDMEM_{trace}_threshold",
            f"VALID_HOLDMEM_{trace}_memory_remaining",
        )
    )
COMPARISONS += [
    ("Graph下内存感知路由", "CONF_GRAPHR_threshold", "CONF_GRAPHR_memory_remaining"),
    (
        "Graph下队列模型独立增量",
        "CONF_GRAPHR_work_remaining",
        "CONF_GRAPHR_memory_remaining",
    ),
    ("Graph下本地调度", "CONF_GRAPHN_L0", "CONF_GRAPHN_L1"),
    ("Graph下CPU准入", "CONF_GRAPHO_none", "CONF_GRAPHO_conditioned"),
    ("Graph下关闭CPU参照", "CONF_GRAPHO_off", "CONF_GRAPHO_conditioned"),
    ("文件缓存写入准入", "V3_DISK_none", "V3_DISK_conditioned"),
    (
        "PD 布局筛选",
        "SCREEN_PDLAYOUT_PD_LBHNC",
        "SCREEN_PDLAYOUT_PD_BLNHC",
    ),
    (
        "PD CPU 筛选",
        "SCREEN_PDCPU_none",
        "SCREEN_PDCPU_conditioned",
    ),
    ("Graph 筛选", "SCREEN_GRAPH_eager_L0", "SCREEN_GRAPH_graph_L0"),
    ("Graph 内本地策略", "SCREEN_GRAPH_graph_L0", "SCREEN_GRAPH_graph_L1"),
]


def fold(root, name):
    rows = []
    for path in sorted(root.glob(name + "_rep*/manifest.json")):
        try:
            row = collect(path.parent)
        except (OSError, ValueError, KeyError):
            continue
        if row["status"] == "completed" and not row["quarantined"]:
            rows.append(row)
    return rows


def compare(root, label, baseline, treatment):
    left, right = fold(root, baseline), fold(root, treatment)
    result = {"label": label, "baseline": baseline, "treatment": treatment}
    result["runs"] = {"baseline": left, "treatment": right}
    result["n"] = [len(left), len(right)]
    if not left or not right:
        return {**result, "status": "pending_or_failed"}
    hashes = {r["trace_sha256"] for r in left + right}
    if len(hashes) != 1:
        raise ValueError(f"Mismatched trace in {label}")
    if len({r["gpus"] for r in left + right}) != 1:
        raise ValueError(f"Mismatched GPU count in {label}")
    metrics = {}
    for key in [
        "output_tok_s",
        "ttft_s_p95",
        "tpot_s_p95",
        "request_goodput",
        "cpu_stored_tokens",
        "cpu_retrieved_tokens",
        "gpu_prefix_hit_ratio",
        "num_preemptions",
    ]:
        values = [
            [r[key] for r in group if r[key] is not None] for group in (left, right)
        ]
        if not all(values):
            continue
        medians = [statistics.median(v) for v in values]
        metrics[key] = {
            "values": values,
            "medians": medians,
            "ranges": [[min(v), max(v)] for v in values],
            "delta_pct": (medians[1] / medians[0] - 1) * 100 if medians[0] else None,
        }
    pairs = {r["run"].rsplit("_rep", 1)[1]: r["output_tok_s"] for r in left}
    gains = [
        (r["output_tok_s"] / pairs[r["run"].rsplit("_rep", 1)[1]] - 1) * 100
        for r in right
        if r["run"].rsplit("_rep", 1)[1] in pairs
    ]
    result.update(
        status="repeated" if min(len(left), len(right)) >= 3 else "screening",
        metrics=metrics,
        paired_throughput_gains_pct=gains,
        all_pairs_positive=bool(gains) and min(gains) > 0,
    )
    return result


def export(root, output):
    results = [compare(root, *item) for item in COMPARISONS]
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence.json").write_text(json.dumps(results, indent=2))
    lines = [
        "# 可复核的对照证据",
        "",
        "只纳入完成且未隔离的运行。三轮仍是小样本；范围不是置信区间。",
        "正向吞吐不代表尾延迟、质量或同资源架构性能也改善。",
        "配置、工作集和执行模式的适用范围以各轮 manifest 为准。",
        "",
        (
            "| 对照 | n 基线/改造 | tok/s 中位数 | 吞吐变化 |"
            " TTFT P95变化 | 配对吞吐变化 |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if "metrics" not in r:
            lines.append(
                f"| {r['label']} | {r['n']} | 待完成或检查未通过 | — | — | — |"
            )
            continue
        throughput = r["metrics"]["output_tok_s"]
        ttft = r["metrics"]["ttft_s_p95"]["delta_pct"]
        before, after = throughput["medians"]
        gains = ", ".join(f"{v:+.2f}%" for v in r["paired_throughput_gains_pct"])
        lines.append(
            f"| {r['label']} | {r['n']} | {before:.2f} → {after:.2f} | "
            f"{throughput['delta_pct']:+.2f}% | {ttft:+.2f}% | {gains} |"
        )
    lines += [
        "",
        "每轮原始数值、读写计数、命中率、抢占数和运行名称见 evidence.json。",
        "旧 V2 offload 结果不参与此表；INT8 CPU 恢复未通过固定门限，性能运行被阻止。",
    ]
    (output / "evidence.md").write_text("\n".join(lines) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = export(args.root, args.output)
    print(
        json.dumps(
            {"comparisons": len(results), "statuses": [r["status"] for r in results]}
        )
    )


if __name__ == "__main__":
    main()
