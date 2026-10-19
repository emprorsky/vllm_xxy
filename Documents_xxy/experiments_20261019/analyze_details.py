"""Request/session groups, router estimation error and counterfactual decisions."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .policies import percentile
from .summarize import samples


def describe(rows):
    success = [r for r in rows if r["success"]]
    result = {"requests": len(rows), "success": len(success)}
    for metric in ["ttft_s", "tpot_s", "e2e_s"]:
        values = [r[metric] for r in success if r.get(metric) is not None]
        result[metric] = {f"p{q}": percentile(values, q) for q in [50, 95, 99]}
        result[metric]["max"] = max(values, default=None)
    return result


def analyze(run):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "completed":
        return None
    trace = [
        json.loads(x)
        for x in Path(manifest["config"]["trace"]).read_text().splitlines()
    ]
    trace = {r["request_id"]: r for r in trace}
    rows = [
        json.loads(x) for x in (run / "replay/requests.jsonl").read_text().splitlines()
    ]
    by_id = {r["request_id"]: r for r in rows}
    groups = defaultdict(list)
    sessions = defaultdict(list)
    for row in rows:
        meta = trace[row["request_id"]]
        groups["first_turn" if row["turn_id"] == 0 else "later_turn"].append(row)
        groups[
            "long_input_ge_2048"
            if row["input_tokens"] >= 2048
            else "short_input_lt_2048"
        ].append(row)
        if "group" in meta:
            groups["hot_group_0" if meta["group"] == 0 else "other_groups"].append(row)
        sessions[row["session_id"]].append(row)
    origin = min(r["clock_start"] for r in rows)
    sessions = {
        name: {
            "calls": len(values),
            "active_span_s": max(r["clock_end"] for r in values)
            - min(r["clock_start"] for r in values),
            "finished_from_run_first_dispatch_s": max(r["clock_end"] for r in values)
            - origin,
        }
        for name, values in sessions.items()
    }
    result = {
        "groups": {name: describe(values) for name, values in groups.items()},
        "sessions": sessions,
        "output_budget_mismatches": [
            r["request_id"]
            for r in rows
            if r["success"]
            and r["output_tokens"] != trace[r["request_id"]]["requested_output_budget"]
        ],
    }
    if (run / "router_decisions.jsonl").exists():
        events = [
            json.loads(x)
            for x in (run / "router_decisions.jsonl").read_text().splitlines()
        ]
        dispatches = [e for e in events if e.get("event") == "dispatch"]
        errors = []
        counterfactual_difference = 0
        for event in dispatches:
            scores = event["scores"]
            work = min(
                scores, key=lambda s: (s["queue_ms"] + s["prefill_ms"], s["worker"])
            )
            cost = min(scores, key=lambda s: (s["total_ms"], s["worker"]))
            counterfactual_difference += work["worker"] != cost["worker"]
            row = by_id.get(event["request_id"])
            if row and row["success"]:
                estimated = scores[event["worker"]]["cached_tokens_est"]
                observed = (
                    row["usage"]
                    .get("prompt_tokens_details", {})
                    .get("cached_tokens", 0)
                )
                errors.append(estimated - observed)
        result["router"] = {
            "source": sorted(
                {e.get("directory_source", "unknown") for e in dispatches}
            ),
            "dispatches": len(dispatches),
            "cost_work_argmin_disagreements_same_state": counterfactual_difference,
            "cached_tokens_mean_absolute_error": sum(abs(e) for e in errors)
            / len(errors)
            if errors
            else None,
            "cached_tokens_overprediction_fraction": sum(e > 0 for e in errors)
            / len(errors)
            if errors
            else None,
            "usage_scope": (
                "GPU-only for no-offload runs; combined may include external hits"
            ),
        }
        result["pd_pairs"] = dict(
            Counter(
                f"P{e['prefill_worker']}->D{e['worker']}"
                for e in events
                if e.get("event") == "decode"
            )
        )
        prefills = {e["request_id"]: e for e in events if e.get("event") == "prefill"}
        joined = [
            (prefills[e["request_id"]]["elapsed_s"], e["elapsed_s"])
            for e in events
            if e.get("event") == "decode"
            and e["request_id"] in by_id
            and e["request_id"] in prefills
        ]
        if joined:
            result["pd_phase_joined_by_request"] = {
                "prefill_queue_and_compute_s": {
                    f"p{q}": percentile([p for p, total in joined], q)
                    for q in [50, 95, 99]
                },
                "decode_transfer_queue_and_generation_s": {
                    f"p{q}": percentile([max(total - p, 0) for p, total in joined], q)
                    for q in [50, 95, 99]
                },
                "scope": "proxy timestamps; not a kernel-only compute or DMA timer",
            }
    metrics = json.loads((run / "replay/metrics.json").read_text())
    roles = {}
    for worker in manifest["config"]["workers"]:
        endpoint = f"http://127.0.0.1:{worker['port']}"
        before = samples(metrics["before"].get(endpoint, ""))
        after = samples(metrics["after"].get(endpoint, ""))
        selected = {}
        for name in after:
            if name.endswith(("_total", "_count", "_sum")) and any(
                key in name
                for key in [
                    "nixl_",
                    "request_prefill_kv_computed_tokens",
                    "request_queue_time_seconds",
                    "request_prefill_time_seconds",
                    "request_decode_time_seconds",
                    "resume_recompute_tokens",
                    "num_preemptions",
                ]
            ):
                selected[name] = after[name] - before.get(name, 0)
        roles[f"{worker.get('role', 'replica')}_gpu{worker['gpu']}"] = selected
    result["engine_counters_by_role"] = roles
    (run / "details.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    count = 0
    for path in args.root.glob("*/manifest.json"):
        if analyze(path.parent) is not None:
            count += 1
    print(f"Analyzed {count} completed runs")


if __name__ == "__main__":
    main()
