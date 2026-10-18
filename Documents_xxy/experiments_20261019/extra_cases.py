"""Generate reviewable optional sensitivity configurations, without running GPUs."""

import argparse
import hashlib
import json
from pathlib import Path

from .suite import MODEL, ROOT, configuration, disaggregate, local_policy, offload

AWQ_MODEL = (
    "/root/autodl-tmp/models/hub/hub/models--Qwen--Qwen2.5-3B-Instruct-AWQ/"
    "snapshots/3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd"
)


def natural(config):
    args = config["server_args"]
    index = args.index("--num-gpu-blocks-override")
    del args[index : index + 2]
    config["scope"] = "natural_GPU_KV_profile_same_memory_utilization"


def mixed_trace():
    original = ROOT / "traces/hot.jsonl"
    path = ROOT / "traces/hot_mixed.jsonl"
    rows = [json.loads(line) for line in original.read_text().splitlines()]
    for row in rows:
        # A declared service budget, independent of the recorded true output.
        value = hashlib.sha256(row["request_id"].encode()).digest()[0]
        row["requested_output_budget"] = [32, 64, 128][value % 3]
        row["budget_origin"] = "declared_hash_assigned_service_budget_not_output_oracle"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "source_trace_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "budgets": [32, 64, 128],
        "rows": len(rows),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))


def configurations(stage):
    result = []
    if stage == "natural":
        for local in [False, True]:
            config = configuration("public_coding", [3])
            natural(config)
            if local:
                local_policy(config)
            result.append((f"NATURAL_L{int(local)}", config))
    elif stage == "quantization":
        for precision, model in [("BF16", MODEL), ("AWQ", AWQ_MODEL)]:
            for local in [False, True]:
                config = configuration("public_coding", [3])
                config["model"] = model
                if precision == "AWQ":
                    config["server_args"] += ["--quantization", "auto_awq"]
                if local:
                    local_policy(config)
                result.append((f"V_{precision}_fixed_L{int(local)}", config))
    elif stage == "kvfp8":
        for local in [False, True]:
            config = configuration("public_coding", [3])
            config["server_args"] += [
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--calculate-kv-scales",
            ]
            if local:
                local_policy(config)
            result.append((f"V_KVFP8_fixed_L{int(local)}", config))
    elif stage == "long":
        for mode in ["off", "none", "fixed", "conditioned"]:
            config = configuration("public_coding_8k", [3], blocks=1152)
            index = config["server_args"].index("--max-model-len")
            config["server_args"][index + 1] = "8192"
            if mode == "off":
                config["server_args"] += ["--max-num-seqs", "2"]
            else:
                offload(config, mode, 1.0)
                index = config["server_args"].index("--kv-transfer-config")
                transfer = json.loads(config["server_args"][index + 1])
                transfer["kv_connector_extra_config"]["admission_write_threshold"] = 8
                config["server_args"][index + 1] = json.dumps(transfer)
            config["scope"] = "long_trace_GPU_KV_18432_CPU_1GiB_threshold8"
            result.append((f"LONG_{mode}_cpu1", config))
    elif stage == "chunk_switch":
        for enabled in [False, True]:
            for local in [False, True]:
                config = configuration("public_coding", [3], budget=8192)
                flag = (
                    "--enable-chunked-prefill"
                    if enabled
                    else "--no-enable-chunked-prefill"
                )
                config["server_args"] += [flag]
                if local:
                    local_policy(config)
                result.append((f"CSWITCH_{int(enabled)}_L{int(local)}", config))
    elif stage == "pd3":
        for p_count in [1, 2]:
            for p_policy, d_policy in [("rr", "rr"), ("cost", "work")]:
                config = configuration("public_coding", [1, 2, 3])
                disaggregate(config, p_count)
                config["pd"].update(prefill_policy=p_policy, decode_policy=d_policy)
                result.append(
                    (f"PD3_{p_count}P{3 - p_count}D_{p_policy}_{d_policy}", config)
                )
        config = configuration("public_coding", [1, 2, 3])
        config["router"] = {
            "policy": "threshold",
            "port": 8290,
            "capacity_tokens": 8192,
            "prefill_ms": 0.1,
            "decode_ms": 35,
        }
        result.append(("REPLICA_3", config))
    elif stage == "events":
        mixed_trace()
        for mode in ["history", "events"]:
            for policy in ["threshold", "work", "cost"]:
                config = configuration("hot_mixed", [2, 3])
                for index, worker in enumerate(config["workers"]):
                    worker["args"] = [
                        "--kv-events-config",
                        json.dumps(
                            {
                                "enable_kv_cache_events": True,
                                "publisher": "zmq",
                                "endpoint": f"tcp://*:{8350 + index}",
                                "replay_endpoint": f"tcp://127.0.0.1:{8450 + index}",
                            }
                        ),
                    ]
                config["env"].update(
                    {
                        "KVEXP_EVENT_ENDPOINTS": json.dumps(
                            [f"tcp://127.0.0.1:{8350 + index}" for index in range(2)]
                        ),
                        "KVEXP_EVENT_REPLAY_ENDPOINTS": json.dumps(
                            [f"tcp://127.0.0.1:{8450 + index}" for index in range(2)]
                        ),
                        "KVEXP_DIRECTORY_MODE": mode,
                    }
                )
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
                result.append((f"EVENT_{mode}_{policy}", config))
    elif stage == "mixed":
        mixed_trace()
        for policy in ["load", "threshold", "work", "cost"]:
            config = configuration("hot_mixed", [2, 3])
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
            result.append((f"RMIX_{policy}", config))
    else:
        raise ValueError(stage)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "natural",
            "quantization",
            "kvfp8",
            "mixed",
            "events",
            "long",
            "chunk_switch",
            "pd3",
        ],
        required=True,
    )
    parser.add_argument("--output", type=Path, default=ROOT / "extra_configs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, config in configurations(args.stage):
        path = args.output / f"{name}.json"
        path.write_text(json.dumps(config, indent=2))
        print(path)


if __name__ == "__main__":
    main()
