"""Bounded sequential experiments; a failed configuration does not stop the suite."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from .run_experiment import run

MODEL = (
    "/root/autodl-tmp/models/hub/hub/models--Qwen--Qwen2.5-3B-Instruct/"
    "snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
)
ROOT = Path("/root/autodl-tmp/outputs/kv_aware_20261019")


def configuration(trace, gpus, blocks=512, budget=1024):
    return {
        "model": MODEL,
        "trace": str(ROOT / "traces" / f"{trace}.jsonl"),
        "concurrency": 16,
        "gap_scale": 1,
        "slo_ttft": 5,
        "slo_tpot": 0.1,
        "env": {"VLLM_USE_V2_MODEL_RUNNER": "0"},
        "workers": [
            {"gpu": gpu, "port": 8210 + index} for index, gpu in enumerate(gpus)
        ],
        "server_args": [
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "8",
            "--max-num-batched-tokens",
            str(budget),
            "--num-gpu-blocks-override",
            str(blocks),
            "--block-size",
            "16",
            "--dtype",
            "bfloat16",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.85",
            "--enable-prefix-caching",
            "--enable-prompt-tokens-details",
            "--attention-backend",
            "TRITON_ATTN",
            "--compilation-config",
            '{"custom_ops":["none"]}',
            "--kernel-config",
            json.dumps(
                {
                    "ir_op_priority": {
                        "rms_norm": ["native"],
                        "fused_add_rms_norm": ["native"],
                    }
                }
            ),
        ],
        "scope": "controlled_GPU_KV_capacity_16_tokens_per_block",
    }


def local_policy(config):
    config["server_args"] += [
        "--admission-policy",
        "cache_affinity",
        "--preemption-policy",
        "recompute_aware",
        "--kv-aware-candidate-window",
        "8",
        "--kv-aware-aging-threshold-s",
        "10",
    ]


def offload(config, mode, capacity_gib):
    config["env"].update(
        {
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(capacity_gib),
            "LMCACHE_CHUNK_SIZE": "256",
        }
    )
    config["server_args"] += [
        "--kv-transfer-config",
        json.dumps(
            {
                "kv_connector": "WorkingSetLMCacheConnector",
                "kv_connector_module_path": (
                    "Documents_xxy.experiments_20261019.lmcache_adapter"
                ),
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "admission_mode": mode,
                    "admission_cpu_bytes": int(capacity_gib * 1024**3),
                    "admission_chunk_tokens": 256,
                    "admission_write_threshold": 2,
                    "use_layerwise": False,
                },
            }
        ),
    ]


def disaggregate(config, prefill_count):
    config["env"]["UCX_TLS"] = "tcp,sm,cuda_copy"
    for index, worker in enumerate(config["workers"]):
        role = "P" if index < prefill_count else "D"
        worker["role"] = role
        worker["env"] = {"VLLM_NIXL_SIDE_CHANNEL_PORT": str(5650 + index)}
        worker["args"] = [
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "NixlConnector",
                    "kv_role": "kv_producer" if role == "P" else "kv_consumer",
                    "kv_load_failure_policy": "fail",
                }
            ),
        ]
    config["pd"] = {
        "port": 8291,
        "prefill_policy": "rr",
        "decode_policy": "rr",
        "capacity_tokens": 8192,
        "prefill_ms": 0.1,
        "decode_ms": 35,
    }


def cases(stage):
    result = []
    if stage == "single":
        for name, local in [("S1_L0", False), ("S1_L1", True)]:
            config = configuration("public_coding", [3])
            if local:
                local_policy(config)
            result.append((name, config))
    elif stage == "offload":
        for mode in ["none", "fixed", "conditioned"]:
            config = configuration("public_coding", [3])
            offload(config, mode, 0.5)
            result.append((f"O_{mode}_cpu0p5", config))
    elif stage == "capacity":
        for capacity in [0.125, 3.0]:
            for mode in ["none", "conditioned"]:
                config = configuration("public_coding", [3])
                offload(config, mode, capacity)
                result.append((f"O_{mode}_cpu{capacity}", config))
    elif stage == "routing":
        for policy in ["load", "threshold", "work", "cost"]:
            config = configuration("hot", [2, 3])
            config["router"] = {
                "policy": policy,
                "port": 8290,
                "capacity_tokens": 8192,
                "prefill_ms": 0.1,
                "decode_ms": 35,
                "balance_abs": 4,
                "balance_rel": 1.5,
                "eviction_weight": 1,
            }
            result.append((f"R_{policy}", config))
    elif stage == "combined":
        for router, admission, local in [
            ("threshold", "none", False),
            ("cost", "none", False),
            ("threshold", "conditioned", False),
            ("cost", "conditioned", False),
            ("cost", "conditioned", True),
        ]:
            config = configuration("public_coding", [2, 3])
            offload(config, admission, 0.25)
            if local:
                local_policy(config)
            config["router"] = {
                "policy": router,
                "port": 8290,
                "capacity_tokens": 8192,
                "prefill_ms": 0.1,
                "decode_ms": 35,
            }
            result.append((f"D_{router}_{admission}_L{int(local)}", config))
    elif stage == "chunking":
        for budget in [512, 2048]:
            for local in [False, True]:
                config = configuration("public_coding", [3], budget=budget)
                if local:
                    local_policy(config)
                result.append((f"C_{budget}_L{int(local)}", config))
    elif stage == "pd1":
        for p_local, d_local in [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]:
            config = configuration("public_coding", [2, 3])
            disaggregate(config, 1)
            for worker, enabled in zip(config["workers"], [p_local, d_local]):
                if enabled:
                    local = {"server_args": []}
                    local_policy(local)
                    worker["args"] += local["server_args"]
            result.append((f"P1_P{int(p_local)}D{int(d_local)}", config))
    elif stage == "pd2":
        for p_policy, d_policy in [
            ("rr", "rr"),
            ("cost", "rr"),
            ("rr", "work"),
            ("cost", "work"),
        ]:
            config = configuration("public_coding", [0, 1, 2, 3])
            disaggregate(config, 2)
            config["pd"].update(prefill_policy=p_policy, decode_policy=d_policy)
            result.append((f"P2_{p_policy}_{d_policy}", config))
    elif stage == "replicas":
        for count in [2, 4]:
            config = configuration(
                "public_coding", [2, 3] if count == 2 else [0, 1, 2, 3]
            )
            config["router"] = {
                "policy": "threshold",
                "port": 8290,
                "capacity_tokens": 8192,
                "prefill_ms": 0.1,
                "decode_ms": 35,
            }
            result.append((f"REPLICA_{count}", config))
    else:
        raise ValueError(stage)
    return result


async def execute(args):
    args.output.mkdir(parents=True, exist_ok=True)
    selected = cases(args.stage)
    if args.only:
        selected = [(n, c) for n, c in selected if args.only in n]
    if not selected:
        raise ValueError("no selected cases")
    status = {"stage": args.stage, "started_at": time.time(), "runs": []}
    status_file = args.output / f"suite_{args.stage}.json"
    for repeat in range(args.repeats):
        ordered = selected if repeat % 2 == 0 else list(reversed(selected))
        for name, config in ordered:
            run_id = f"{name}_rep{repeat + 1:02d}"
            destination = args.output / run_id
            if (destination / "manifest.json").exists():
                previous = json.loads((destination / "manifest.json").read_text())
                if previous.get("status") == "completed":
                    print(
                        json.dumps({"event": "skip_completed", "run_id": run_id}),
                        flush=True,
                    )
                    continue
                print(
                    json.dumps({"event": "skip_existing_failure", "run_id": run_id}),
                    flush=True,
                )
                continue
            config_path = args.output / f"{run_id}.config.json"
            config_path.write_text(json.dumps(config, indent=2))
            print(
                json.dumps({"event": "start", "stage": args.stage, "run_id": run_id}),
                flush=True,
            )
            entry = {"run_id": run_id, "started_at": time.time()}
            try:
                await asyncio.wait_for(
                    run(argparse.Namespace(config=config_path, output=destination)),
                    timeout=args.timeout,
                )
                entry["status"] = "completed"
            except Exception as exc:
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            entry["ended_at"] = time.time()
            status["runs"].append(entry)
            status_file.write_text(json.dumps(status, indent=2))
            print(json.dumps({"event": "finish", **entry}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "single",
            "offload",
            "capacity",
            "routing",
            "combined",
            "chunking",
            "pd1",
            "pd2",
            "replicas",
        ],
        required=True,
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
