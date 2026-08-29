#!/usr/bin/env python3
"""Capture causal serving metrics from a running vLLM server."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')

COUNTERS = {
    "prompt_tokens": "vllm:prompt_tokens_total",
    "prompt_tokens_cached": "vllm:prompt_tokens_cached_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "preemptions": "vllm:num_preemptions_total",
    "preempted_computed_tokens": (
        "vllm:kv_preemption_preempted_computed_tokens_total"
    ),
    "resume_recompute_tokens": (
        "vllm:kv_preemption_resume_recompute_tokens_total"
    ),
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "admission_selection_calls": "vllm:kv_admission_selection_calls_total",
    "admission_candidate_probes": "vllm:kv_admission_candidate_probes_total",
    "admission_reordered": "vllm:kv_admission_reordered_total",
    "admission_aged_selections": "vllm:kv_admission_aged_selections_total",
    "admission_admitted": "vllm:kv_admission_admitted_total",
    "admission_admitted_reordered": (
        "vllm:kv_admission_admitted_reordered_total"
    ),
    "admission_admitted_aged": "vllm:kv_admission_admitted_aged_total",
}

HISTOGRAMS = {
    "iteration_tokens": "vllm:iteration_tokens_total",
    "request_queue_time_seconds": "vllm:request_queue_time_seconds",
    "request_prefill_time_seconds": "vllm:request_prefill_time_seconds",
    "request_decode_time_seconds": "vllm:request_decode_time_seconds",
    "request_inference_time_seconds": "vllm:request_inference_time_seconds",
    "request_prefill_kv_computed_tokens": (
        "vllm:request_prefill_kv_computed_tokens"
    ),
}


def _parse_labels(raw_labels: str | None) -> dict[str, str]:
    if raw_labels is None:
        return {}
    return {
        key: bytes(value, "utf-8").decode("unicode_escape")
        for key, value in LABEL_RE.findall(raw_labels)
    }


def parse_samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if match is None:
            continue
        samples.append(
            (
                match.group("name"),
                _parse_labels(match.group("labels")),
                float(match.group("value")),
            )
        )
    return samples


def _matches_engine(
    labels: dict[str, str], engine: str, model_name: str | None
) -> bool:
    return labels.get("engine") == engine and (
        model_name is None or labels.get("model_name") == model_name
    )


def build_snapshot(
    text: str,
    *,
    endpoint: str,
    engine: str,
    model_name: str | None,
    label: str | None,
) -> dict[str, object]:
    samples = [
        sample
        for sample in parse_samples(text)
        if _matches_engine(sample[1], engine, model_name)
    ]
    by_name: dict[str, list[tuple[dict[str, str], float]]] = {}
    for name, labels, value in samples:
        by_name.setdefault(name, []).append((labels, value))

    required_counters = (
        "preempted_computed_tokens",
        "resume_recompute_tokens",
    )
    missing = [key for key in required_counters if COUNTERS[key] not in by_name]
    if missing:
        raise RuntimeError(f"server does not expose required counters: {missing}")

    counters = {
        key: by_name.get(metric, [({}, 0.0)])[0][1]
        for key, metric in COUNTERS.items()
    }
    prompt_by_source = {
        labels["source"]: value
        for labels, value in by_name.get("vllm:prompt_tokens_by_source_total", [])
        if "source" in labels
    }

    histograms: dict[str, dict[str, float]] = {}
    for key, metric in HISTOGRAMS.items():
        count = by_name.get(f"{metric}_count", [({}, 0.0)])[0][1]
        total = by_name.get(f"{metric}_sum", [({}, 0.0)])[0][1]
        histograms[key] = {
            "count": count,
            "sum": total,
            "mean": total / count if count else 0.0,
        }

    local_compute = prompt_by_source.get("local_compute", 0.0)
    resume_recompute = counters["resume_recompute_tokens"]
    total_local_compute = local_compute + resume_recompute
    total_model_compute = total_local_compute + counters["generation_tokens"]
    iteration_count = histograms["iteration_tokens"]["count"]
    preempted_compute = counters["preempted_computed_tokens"]
    derived = {
        "resume_recompute_fraction_of_local_compute": (
            resume_recompute / total_local_compute if total_local_compute else 0.0
        ),
        "total_local_compute_including_resume_recompute": total_local_compute,
        "total_model_compute_tokens": total_model_compute,
        "model_compute_tokens_per_engine_output": (
            total_model_compute / iteration_count if iteration_count else 0.0
        ),
        "resume_recompute_per_preempted_computed_token": (
            resume_recompute / preempted_compute if preempted_compute else 0.0
        ),
        "prefix_cache_hit_rate": (
            counters["prefix_cache_hits"] / counters["prefix_cache_queries"]
            if counters["prefix_cache_queries"]
            else 0.0
        ),
    }
    return {
        "schema_version": 2,
        "captured_at": datetime.now(UTC).isoformat(),
        "endpoint": endpoint,
        "engine": engine,
        "model_name": model_name,
        "label": label,
        "counters": counters,
        "prompt_tokens_by_source": prompt_by_source,
        "histograms": histograms,
        "derived": derived,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--engine", default="0")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--label")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opener = build_opener(ProxyHandler({}))
    with opener.open(args.endpoint, timeout=10) as response:
        text = response.read().decode("utf-8")
    snapshot = build_snapshot(
        text,
        endpoint=args.endpoint,
        engine=args.engine,
        model_name=args.model_name,
        label=args.label,
    )
    args.output.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
