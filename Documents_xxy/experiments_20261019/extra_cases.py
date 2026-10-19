"""Generate reviewable optional sensitivity configurations, without running GPUs."""

import argparse
import copy
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
    if stage == "pd_cpu":
        for mode in ["off", "none", "conditioned"]:
            config = configuration("public_coding", [2, 3])
            disaggregate(config, 1)
            config["server_args"] += ["--max-num-seqs", "2"]
            config["coding_quality_probes"] = True
            config["pd_fidelity_probes"] = True
            if mode != "off":
                temporary = configuration("public_coding", [0])
                offload(temporary, mode, 0.5)
                config["env"].update(temporary["env"])
                at = temporary["server_args"].index("--kv-transfer-config")
                cpu = json.loads(temporary["server_args"][at + 1])
                cpu["kv_connector_extra_config"]["admission_optional_in_multi"] = True
                for worker in config["workers"]:
                    at = worker["args"].index("--kv-transfer-config")
                    nixl = json.loads(worker["args"][at + 1])
                    worker["args"][at + 1] = json.dumps(
                        {
                            "kv_connector": "MultiConnector",
                            "kv_role": "kv_both",
                            "kv_load_failure_policy": "fail",
                            "kv_connector_extra_config": {"connectors": [nixl, cpu]},
                        }
                    )
                config["required_checks"] = [
                    str(ROOT / "checks/cpu_prefix_fidelity_hnd/manifest.json")
                ]
            config["scope"] = "Nixl_mandatory_handoff_CPU_optional_write_admission_seq2"
            result.append((f"PDCPU_{mode}", config))
    elif stage == "fairness":
        for name, source in configurations("quant_natural"):
            if name == "QNAT_BF16_L0":
                result.append(("FAIR_L0", source))
            elif name == "QNAT_BF16_L1":
                result.append(("FAIR_L1_age10", source))
                config = copy.deepcopy(source)
                index = config["server_args"].index("--kv-aware-aging-threshold-s")
                config["server_args"][index + 1] = "5"
                result.append(("FAIR_L1_age5", config))
    elif stage == "memory_router":
        for trace, blocks, context in [
            ("public_coding", 512, 4096),
            ("public_coding_8k", 1152, 8192),
        ]:
            for policy in ["threshold", "work_remaining", "memory_remaining"]:
                config = configuration(trace, [2, 3], blocks=blocks)
                config["server_args"] += ["--max-model-len", str(context)]
                config["router"] = {
                    "policy": policy,
                    "port": 8290,
                    "capacity_tokens": blocks * 16,
                    "prefill_ms": 0.1,
                    "decode_ms": 35,
                    "max_batch": 8,
                }
                config["scope"] = (
                    "FIFO_batch_slots_and_estimated_pinned_KV_queue_proxy_O_OFF"
                )
                result.append((f"MEMORY_{trace}_{policy}", config))
    elif stage == "pd_layout":
        for architecture in ["PD", "REPLICA"]:
            for layout in ["LBHNC", "BLNHC"]:
                config = configuration("public_coding", [2, 3])
                config["env"]["VLLM_KV_CACHE_LAYOUT"] = layout
                config["coding_quality_probes"] = True
                if architecture == "PD":
                    disaggregate(config, 1)
                    config["pd_fidelity_probes"] = True
                else:
                    config["router"] = {
                        "policy": "threshold",
                        "port": 8290,
                        "capacity_tokens": 8192,
                        "prefill_ms": 0.1,
                        "decode_ms": 35,
                    }
                config["scope"] = (
                    "existing_KV_layout_transfer_vs_attention_ablation_TP1"
                )
                result.append((f"PDLAYOUT_{architecture}_{layout}", config))
    elif stage == "kvint8":
        for precision in ["BF16", "INT8"]:
            for local in [False, True]:
                config = configuration("public_coding", [0])
                if precision == "INT8":
                    config["server_args"] += ["--kv-cache-dtype", "int8_per_token_head"]
                if local:
                    local_policy(config)
                config["coding_quality_probes"] = True
                config["scope"] = "KV_precision_fixed_GPU_tokens_8192_local_L0_L1"
                result.append((f"KVINT8_{precision}_L{int(local)}", config))
    elif stage == "kvint8_offload":
        for mode in ["none", "conditioned"]:
            config = configuration("public_coding", [0])
            offload(config, mode, 0.5)
            config["server_args"] += ["--kv-cache-dtype", "int8_per_token_head"]
            config["required_checks"] = [
                str(ROOT / "checks/transfer_roundtrip_quant.manifest.json"),
                str(ROOT / "checks/cpu_prefix_fidelity_int8/manifest.json"),
            ]
            config["scope"] = "INT8_KV_inline_scales_fixed_CPU_bytes_0p5GiB"
            result.append((f"KVINT8_O_{mode}", config))
    elif stage == "holdout_offload":
        for mode in ["none", "conditioned"]:
            config = configuration("holdout_coding", [0])
            offload(config, mode, 0.5)
            config["scope"] = "disjoint_task_validation_frozen_O_WS_threshold2"
            result.append((f"HOLDOUT_O_{mode}", config))
    elif stage == "holdout_local":
        for local in [False, True]:
            config = configuration("holdout_coding_8k", [1])
            natural(config)
            config["concurrency"] = 32
            config["server_args"] += ["--max-model-len", "8192", "--max-num-seqs", "32"]
            if local:
                local_policy(config)
            config["scope"] = "disjoint_task_validation_frozen_L_natural_C32"
            result.append((f"HOLDOUT_L{int(local)}", config))
    elif stage == "holdout_router":
        for policy in ["threshold", "work_remaining"]:
            config = configuration("holdout_coding", [2, 3])
            config["router"] = {
                "policy": policy,
                "port": 8290,
                "capacity_tokens": 8192,
                "prefill_ms": 0.1,
                "decode_ms": 35,
            }
            config["scope"] = "disjoint_task_validation_frozen_R_coefficients_0p1_35"
            result.append((f"HOLDOUT_R_{policy}", config))
    elif stage == "combined_remaining":
        for policy, mode, local in [
            ("threshold", "none", False),
            ("work_remaining", "none", False),
            ("threshold", "conditioned", False),
            ("work_remaining", "conditioned", False),
            ("work_remaining", "conditioned", True),
        ]:
            config = configuration("public_coding", [2, 3])
            offload(config, mode, 0.25)
            if local:
                local_policy(config)
            config["router"] = {
                "policy": policy,
                "port": 8290,
                "capacity_tokens": 8192,
                "prefill_ms": 0.1,
                "decode_ms": 35,
            }
            result.append((f"DR_{policy}_{mode}_L{int(local)}", config))
    elif stage == "local_hot":
        for local in [False, True]:
            config = configuration("hot", [1], blocks=256)
            # The block pool reserves a null block: 256 blocks cannot serve
            # max_model_len=4096. Every trace request fits within 2048.
            config["server_args"] += ["--max-model-len", "2048"]
            config["scope"] = "controlled_hot_prefixes_GPU_KV4096_local_mechanism"
            if local:
                local_policy(config)
            result.append((f"LHOT_L{int(local)}", config))
    elif stage in {"remaining", "threshold_feedback", "profiled"}:
        for trace in ["hot_mixed", "public_coding"]:
            if trace == "hot_mixed":
                mixed_trace()
            policies = (
                ["threshold", "threshold_remaining", "work_remaining"]
                if stage == "profiled"
                else ["threshold", "threshold_remaining"]
                if stage == "threshold_feedback"
                else ["threshold", "work", "work_remaining", "cost_remaining"]
            )
            for policy in policies:
                config = configuration(trace, [2, 3])
                config["router"] = {
                    "policy": policy,
                    "port": 8290,
                    "capacity_tokens": 8192,
                    "prefill_ms": 0.1,
                    "decode_ms": 35,
                }
                if stage == "profiled":
                    config["router"]["cost_profile"] = str(
                        ROOT / "checks/cost_profile/cost_profile.json"
                    )
                prefix = (
                    "PROFILED"
                    if stage == "profiled"
                    else "TFEEDBACK"
                    if stage == "threshold_feedback"
                    else "REMAIN"
                )
                result.append((f"{prefix}_{trace}_{policy}", config))
    elif stage == "quant_natural":
        for precision, model in [("BF16", MODEL), ("AWQ", AWQ_MODEL)]:
            for local in [False, True]:
                config = configuration("public_coding_8k", [1])
                natural(config)
                config["concurrency"] = 32
                config["coding_quality_probes"] = True
                config["model"] = model
                if precision == "AWQ":
                    config["server_args"] += ["--quantization", "auto_awq"]
                config["server_args"] += [
                    "--max-model-len",
                    "8192",
                    "--max-num-seqs",
                    "32",
                ]
                if local:
                    local_policy(config)
                result.append((f"QNAT_{precision}_L{int(local)}", config))
    elif stage == "natural":
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
            ]
            config["scope"] = "KV_FP8_E4M3_unit_scales_compatibility_probe"
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
            "remaining",
            "threshold_feedback",
            "profiled",
            "memory_router",
            "combined_remaining",
            "quant_natural",
            "local_hot",
            "fairness",
            "holdout_offload",
            "holdout_local",
            "holdout_router",
            "kvint8",
            "kvint8_offload",
            "pd_layout",
            "pd_cpu",
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
