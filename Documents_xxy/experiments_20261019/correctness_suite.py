"""Controlled single-batch cold/GPU/CPU checks, including prefix-dependent facts."""

import argparse
import asyncio
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from .run_experiment import wait_ready
from .suite import MODEL, ROOT, configuration, offload
from .validate_outputs import validate


def first_distribution(response):
    return response["choices"][0]["logprobs"]["top_logprobs"][0]


def difference(left, right):
    a, b = first_distribution(left), first_distribution(right)
    shared = a.keys() & b.keys()
    return {
        "topk_overlap": len(shared) / max(len(a), 1),
        "max_shared_logprob_delta": max(
            (abs(a[k] - b[k]) for k in shared), default=None
        ),
        "mean_shared_logprob_delta": sum(abs(a[k] - b[k]) for k in shared)
        / max(len(shared), 1),
        "first_token_equal": left["choices"][0].get("token_ids", [None])[:1]
        == right["choices"][0].get("token_ids", [None])[:1],
    }


async def protocol(model, endpoint, output, server_log):
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows = [
        json.loads(x)
        for x in (ROOT / "traces/public_coding.jsonl").read_text().splitlines()
    ]
    firsts = list(
        {r["session_id"]: r for r in reversed(rows) if r["turn_id"] == 0}.values()
    )
    firsts.sort(key=lambda r: r["session_id"])
    selected = [
        (f"coding-{i:02d}", firsts[i]["prompt_token_ids"], None)
        for i in range(0, len(firsts), 4)
    ]
    filler = (
        "Unrelated documentation: this section describes "
        "ordinary files and routine checks.\n"
    ) * 120
    for index, value in enumerate(["marigold", "cobalt", "orchid"]):
        text = (
            f"The project constant PROJECT_SENTINEL is '{value}'.\n"
            + filler
            + "\nReturn only the value of PROJECT_SENTINEL defined at the beginning."
        )
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        selected.append((f"needle-{index}", tokens, value))
    noise_base = json.loads((ROOT / "traces/hot.jsonl").read_text().splitlines()[0])[
        "prompt_token_ids"
    ]
    records = []
    # These exploratory numerical tolerances are frozen before observing results.
    specification = {
        "batch_size": 1,
        "topk": 20,
        "minimum_overlap": 0.8,
        "logprob_delta_floor": 0.3,
        "gpu_baseline_delta_multiplier": 2,
        "noise_requests": 12,
        "noise_skip_save": True,
        "scope": (
            "next-token top-k and three prefix-dependent facts; "
            "not full quality evaluation"
        ),
    }
    (output / "specification.json").write_text(json.dumps(specification, indent=2))
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:

        async def query(tokens, rid, skip=False, budget=1, logprobs=True):
            body = {
                "model": "experiment",
                "prompt": tokens,
                "max_tokens": budget,
                "temperature": 0,
                "return_token_ids": True,
                "kv_transfer_params": {
                    "lmcache.skip_save": skip,
                    "experiment.phase": "validation",
                },
            }
            if logprobs:
                body.update(logprobs=20, return_tokens_as_token_ids=True)
            response = await client.post(
                endpoint + "/v1/completions", json=body, headers={"X-Request-ID": rid}
            )
            response.raise_for_status()
            return response.json()

        for case_index, (name, tokens, expected) in enumerate(selected):
            budget = 16 if expected else 1
            cold = await query(tokens, f"proof-{name}-cold", budget=budget)
            gpu = await query(tokens, f"proof-{name}-gpu", budget=budget)
            for index in range(12):
                # Unique first block forces distinct full-prefix hashes. Total
                # noise exceeds 8192 GPU slots; no optional CPU writes happen.
                noise = [2000 + case_index * 16 + index] * 16 + noise_base[16:]
                await query(
                    noise, f"proof-{name}-noise-{index}", skip=True, logprobs=False
                )
            restored = await query(tokens, f"proof-{name}-restored", budget=budget)
            baseline = difference(cold, gpu)
            comparison = difference(cold, restored)
            limit = max(0.3, 2 * (baseline["max_shared_logprob_delta"] or 0))
            numeric = (
                comparison["max_shared_logprob_delta"] is not None
                and comparison["max_shared_logprob_delta"] <= limit
                and comparison["topk_overlap"] >= min(0.8, baseline["topk_overlap"])
            )
            texts = [
                result["choices"][0]["text"].strip().strip("`'\" .\n").casefold()
                for result in [cold, gpu, restored]
            ]
            content = expected is None or all(text == expected for text in texts)
            logs = server_log.read_text(errors="replace")
            actual_reads = sum(
                int(n)
                for n in re.findall(
                    (
                        rf"\[req_id=cmpl-proof-{re.escape(name)}-restored[^\]]*\] "
                        r"Retrieved (\d+) out of"
                    ),
                    logs,
                )
            )
            record = {
                "name": name,
                "input_tokens": len(tokens),
                "expected": expected,
                "cold_vs_gpu": baseline,
                "cold_vs_cpu": comparison,
                "numerical_limit": limit,
                "numerical_gate": numeric,
                "content_gate": content,
                "cpu_retrieved_tokens": actual_reads,
                "cold": cold,
                "gpu": gpu,
                "restored": restored,
            }
            records.append(record)
            (output / "probes.json").write_text(
                json.dumps(records, indent=2, ensure_ascii=False)
            )
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in [
                            "name",
                            "numerical_gate",
                            "content_gate",
                            "cpu_retrieved_tokens",
                        ]
                    }
                ),
                flush=True,
            )
    passed = all(
        r["numerical_gate"] and r["content_gate"] and r["cpu_retrieved_tokens"] > 0
        for r in records
    )
    return {
        "passed": passed,
        "probes": len(records),
        "content_probes": 3,
        "numeric_alerts": [r["name"] for r in records if not r["numerical_gate"]],
        "failed_content": [r["name"] for r in records if not r["content_gate"]],
        "missing_cpu_reads": [
            r["name"] for r in records if not r["cpu_retrieved_tokens"]
        ],
    }


async def execute(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if args.wait_manifest:
        while True:
            if args.wait_manifest.exists():
                value = json.loads(args.wait_manifest.read_text())
                if value["status"] in {"completed", "failed"} and "ended_at" in value:
                    break
            await asyncio.sleep(2)
    config = configuration("public_coding", [args.gpu])
    offload(config, "none", 0.5)
    endpoint = f"http://127.0.0.1:{args.port}"
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--served-model-name",
        "experiment",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ] + config["server_args"]
    env = {
        **os.environ,
        **config["env"],
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HUB_OFFLINE": "1",
        "KVEXP_STATS_DIR": str(args.output / "admission"),
        "PATH": str(Path(sys.executable).parent)
        + os.pathsep
        + os.environ.get("PATH", ""),
    }
    manifest = {
        "status": "starting",
        "started_at": time.time(),
        "argv": argv,
        "config": config,
    }

    def save():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    save()
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(
            argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            await wait_ready(args.port, process)
            await validate(MODEL, [endpoint], args.output / "basic_output_gate.json")
            result = await protocol(
                MODEL, endpoint, args.output, args.output / "server.log"
            )
            manifest.update(
                status="completed" if result["passed"] else "failed", result=result
            )
        except BaseException as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.to_thread(process.wait, timeout=20)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            manifest["ended_at"] = time.time()
            save()
    print(json.dumps(manifest.get("result", {})), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8270)
    parser.add_argument("--wait-manifest", type=Path)
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
