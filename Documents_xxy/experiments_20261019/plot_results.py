"""Export scientific figures from completed-run medians and observed ranges."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def comparison(aggregate, names, title, path):
    names = [name for name in names if name in aggregate]
    if not names:
        return
    figure, axes = plt.subplots(1, 2, figsize=(max(9, len(names) * 1.1), 4.5))
    for axis, metric, ylabel in zip(
        axes,
        ["output_tok_s", "ttft_s_p95"],
        ["Output tokens / second", "TTFT P95 (seconds)"],
    ):
        values = [aggregate[n]["metrics"][metric]["median"] for n in names]
        lower = [
            v - aggregate[n]["metrics"][metric]["min"] for n, v in zip(names, values)
        ]
        upper = [
            aggregate[n]["metrics"][metric]["max"] - v for n, v in zip(names, values)
        ]
        x = np.arange(len(names))
        axis.bar(x, values, yerr=[lower, upper], capsize=4, color="#377eb8", alpha=0.85)
        axis.set_xticks(
            x,
            [f"{n}\nn={aggregate[n]['n']}" for n in names],
            rotation=35,
            ha="right",
            fontsize=8,
        )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        title + "\nMedian; whiskers = observed min/max, not confidence intervals",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=160)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = json.loads(args.aggregate.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    panels = [
        (
            "single",
            ["S1_L0", "S1_L1"],
            "Public coding: local scheduling, fixed GPU KV capacity",
        ),
        (
            "offload",
            [f"O_seq2_{m}_cpu0p5" for m in ["off", "none", "fixed", "conditioned"]],
            "Public coding: CPU write admission, resident request limit 2",
        ),
        (
            "routing",
            [f"R_{p}" for p in ["load", "threshold", "work", "cost"]],
            "Controlled hot prefixes: two replicas, identical directory",
        ),
        (
            "mixed",
            [f"RMIX_{p}" for p in ["load", "threshold", "work", "cost"]],
            "Hot prefixes with declared mixed output budgets",
        ),
        (
            "combined",
            [n for n in aggregate if n.startswith("D_")],
            "Public coding: routing / offload / local scheduling ablation",
        ),
        (
            "pd1",
            [f"P1_P{p}D{d}" for p, d in [(0, 0), (1, 0), (0, 1), (1, 1)]]
            + ["REPLICA_2"],
            "Two GPUs: 1P1D role ablation and full replicas",
        ),
        (
            "pd2",
            [
                f"P2_{p}_{d}"
                for p, d in [
                    ("rr", "rr"),
                    ("cost", "rr"),
                    ("rr", "work"),
                    ("cost", "work"),
                ]
            ]
            + ["REPLICA_4"],
            "Four GPUs: 2P2D placement and full replicas",
        ),
        (
            "chunking",
            ["C_512_L0", "C_512_L1", "S1_L0", "S1_L1", "C_2048_L0", "C_2048_L1"],
            "Chunked prefill token budget x local policy",
        ),
    ]
    for name, selected, title in panels:
        comparison(aggregate, selected, title, args.output / name)
    print(f"Figures exported to {args.output}")


if __name__ == "__main__":
    main()
