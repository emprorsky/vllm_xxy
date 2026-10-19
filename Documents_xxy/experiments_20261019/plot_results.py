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
        title
        + (
            "\nOFFLOAD FIDELITY FAILED: performance observations quarantined"
            if any(
                aggregate[n].get("quarantined", n.startswith(("O_", "D_", "LONG_")))
                for n in names
            )
            else ""
        )
        + "\nMedian; whiskers = observed min/max, not confidence intervals",
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
    panels += [
        (
            "architecture_final",
            ["CONF_ARCH_replica4", "CONF_ARCH_2P2D_rr", "CONF_ARCH_2P2D_work"],
            "Four GPUs: full replicas vs pair-validated 2P2D",
        ),
        (
            "v3_offload",
            [f"V3_O_seq2_{m}_cpu0p5" for m in ["off", "none", "fixed", "conditioned"]],
            "Corrected packed KV transfer: CPU admission, fixed GPU/CPU capacity",
        ),
        (
            "v3_combined",
            [n for n in aggregate if n.startswith("V3_DR_")],
            "Corrected transfer: routing x CPU write admission x local policy",
        ),
        (
            "local_hot_confirm",
            ["FIX_LHOT_L0", "FIX_LHOT_L1"],
            "Hot prefix mechanism: GPU KV 4096, max context 2048",
        ),
        (
            "routing_confirm",
            [f"CONF_R_{p}" for p in ["threshold", "work", "cost"]],
            "Independent repetitions: controlled hot-prefix routing",
        ),
        (
            "remaining_public_confirm",
            [f"CONF_REMAIN_public_coding_{p}" for p in ["threshold", "work_remaining"]],
            "Public coding: count threshold vs observed remaining work",
        ),
        (
            "quant_natural_confirm",
            [f"CONF_QNAT_{q}_L{local}" for q in ["BF16", "AWQ"] for local in [0, 1]],
            "8K coding, 32 sessions, same GPU memory budget, natural KV capacity",
        ),
        (
            "holdout_local",
            ["VALID_HOLDOUT_L0", "VALID_HOLDOUT_L1"],
            "Disjoint tasks: local policy, natural KV capacity, 32 sessions",
        ),
        (
            "holdout_offload",
            [f"VALID_HOLDOUT_O_{m}" for m in ["none", "conditioned"]],
            "Disjoint tasks: corrected CPU admission",
        ),
        (
            "holdout_router",
            [f"VALID_HOLDOUT_R_{p}" for p in ["threshold", "work_remaining"]],
            "Disjoint tasks: frozen routing policies and coefficients",
        ),
        (
            "events",
            [
                f"EVENT_{mode}_{policy}"
                for mode in ["history", "events"]
                for policy in ["threshold", "work", "cost"]
            ],
            "Same event publisher/subscriber: history vs event + completion directory",
        ),
        (
            "long",
            [f"LONG_{mode}_cpu1" for mode in ["off", "none", "fixed", "conditioned"]],
            "8K public coding: GPU KV 18432, CPU 1GiB, write threshold 8",
        ),
        (
            "chunk_switch",
            [f"CSWITCH_{enabled}_L{local}" for enabled in [0, 1] for local in [0, 1]],
            "Chunking switch with matched 8192-token budget",
        ),
        (
            "natural",
            ["NATURAL_L0", "NATURAL_L1"],
            "Natural GPU KV profiling capacity: local scheduling control",
        ),
        (
            "quantization",
            [
                f"V_{precision}_fixed_L{local}"
                for precision in ["BF16", "AWQ"]
                for local in [0, 1]
            ],
            "Weight precision x local policy: only output-gated completed runs",
        ),
        (
            "pd3",
            [
                f"PD3_{p}P{3 - p}D_{route}"
                for p in [1, 2]
                for route in ["rr_rr", "cost_work"]
            ]
            + ["REPLICA_3"],
            "Three GPUs: P/D ratio and full replicas",
        ),
    ]
    panels += [
        (
            "fairness_confirm",
            ["CONF_FAIR_L0", "CONF_FAIR_L1_age10", "CONF_FAIR_L1_age5"],
            "8K, C32: cache affinity with different aging cutoffs",
        ),
        (
            "graph_local_confirm",
            ["CONF_GRAPHN_L0", "CONF_GRAPHN_L1"],
            "Decode CUDA graphs: local policy, natural KV capacity, C32",
        ),
        (
            "graph_cpu_confirm",
            [f"CONF_GRAPHO_{m}" for m in ["off", "none", "conditioned"]],
            "Decode CUDA graphs: corrected CPU admission, fixed capacity",
        ),
        (
            "disk_confirm",
            ["V3_DISK_none", "V3_DISK_conditioned"],
            "Buffered file KV: persistent tier 1GiB, CPU staging 0.125GiB",
        ),
        (
            "pd_layout_screen",
            [n for n in aggregate if n.startswith("SCREEN_PDLAYOUT_")],
            "Existing layouts: PD vs full replicas, transfer/attention control",
        ),
        (
            "pd_cpu_screen",
            [n for n in aggregate if n.startswith("SCREEN_PDCPU_")],
            "Mandatory NIXL handoff plus optional CPU write admission",
        ),
    ]
    for trace in ["public_coding", "public_coding_8k"]:
        panels.append(
            (
                "memory_confirm_" + trace,
                [f"CONF_MEMORY_{trace}_{p}" for p in ["threshold", "memory_remaining"]],
                "FIFO batch/KV queue approximation: " + trace,
            )
        )
    for trace in ["holdout_coding", "holdout_coding_8k"]:
        panels.append(
            (
                "holdout_memory_" + trace,
                [
                    f"VALID_HOLDMEM_{trace}_{p}"
                    for p in ["threshold", "memory_remaining"]
                ],
                "Disjoint tasks, frozen FIFO batch/KV queue approximation: " + trace,
            )
        )
    for name, selected, title in panels:
        comparison(aggregate, selected, title, args.output / name)
    print(f"Figures exported to {args.output}")


if __name__ == "__main__":
    main()
