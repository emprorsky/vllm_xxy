"""Compare real P/D generation with aggregated P references and transfer counters."""

import json
import re

import httpx
from transformers import AutoTokenizer

from .correctness_suite import difference
from .suite import ROOT
from .summarize import samples


async def validate_pd(model, proxy, prefill, decoders, output):
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows = [
        json.loads(line)
        for line in (ROOT / "traces/public_coding.jsonl").read_text().splitlines()
    ]
    firsts = list(
        {r["session_id"]: r for r in reversed(rows) if r["turn_id"] == 0}.values()
    )
    firsts.sort(key=lambda row: row["session_id"])
    probes = [
        (f"coding-{i}", firsts[i]["prompt_token_ids"], None)
        for i in range(0, len(firsts), 4)
    ]
    filler = "Unrelated documentation about ordinary files and routine checks.\n" * 140
    for index, value in enumerate(["marigold", "cobalt", "orchid"]):
        prompt = (
            f"Case {index}: The project constant PROJECT_SENTINEL is '{value}'.\n"
            + filler
            + "\nReturn only the value of PROJECT_SENTINEL defined at the beginning."
        )
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        probes.append((f"fact-{index}", tokens, value))
    marker = tokenizer.encode(
        "<|im_start|>system\nTransportDiagnostic: "
        "validate KV movement only.<|im_end|>\n",
        add_special_tokens=False,
    )
    probes = [(name, marker + tokens, expected) for name, tokens, expected in probes]
    evaluation_roots = {tuple(row["prompt_token_ids"][:16]) for row in rows}
    if any(tuple(tokens[:16]) in evaluation_roots for _, tokens, _ in probes):
        raise ValueError("diagnostic prefix overlaps performance trace")
    records = []
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        before = {
            endpoint: samples((await client.get(endpoint + "/metrics")).text)
            for endpoint in decoders
        }

        async def query(endpoint, tokens, rid):
            response = await client.post(
                endpoint + "/v1/completions",
                json={
                    "model": "experiment",
                    "prompt": tokens,
                    "max_tokens": 16,
                    "temperature": 0,
                    "return_token_ids": True,
                    "logprobs": 20,
                    "return_tokens_as_token_ids": True,
                    "kv_transfer_params": {
                        "lmcache.skip_save": True,
                        "experiment.phase": "validation",
                    },
                },
                headers={"X-Request-ID": rid},
            )
            response.raise_for_status()
            return response.json()

        for name, tokens, expected in probes:
            cold = await query(prefill, tokens, f"pdfidelity-{name}-cold")
            gpu = await query(prefill, tokens, f"pdfidelity-{name}-gpu")
            pd = await query(proxy, tokens, f"pdfidelity-{name}-pd")
            baseline, actual = difference(cold, gpu), difference(cold, pd)
            limit = max(0.3, 2 * (baseline["max_shared_logprob_delta"] or 0))
            numeric = (
                actual["max_shared_logprob_delta"] is not None
                and actual["max_shared_logprob_delta"] <= limit
                and actual["topk_overlap"] >= min(0.8, baseline["topk_overlap"])
            )
            content = expected is None or all(
                re.search(rf"\b{expected}\b", value["choices"][0]["text"].casefold())
                for value in [cold, gpu, pd]
            )
            records.append(
                {
                    "name": name,
                    "input_tokens": len(tokens),
                    "expected": expected,
                    "cold": cold,
                    "gpu": gpu,
                    "pd": pd,
                    "cold_vs_gpu": baseline,
                    "cold_vs_pd": actual,
                    "numerical_limit": limit,
                    "numerical_gate": numeric,
                    "content_gate": bool(content),
                    "output_token_ids_match": cold["choices"][0].get("token_ids")
                    == pd["choices"][0].get("token_ids"),
                }
            )
        delta = {}
        for endpoint in decoders:
            after = samples((await client.get(endpoint + "/metrics")).text)
            for name, value in after.items():
                if name.endswith(("_sum", "_total", "_count")):
                    delta[name] = (
                        delta.get(name, 0) + value - before[endpoint].get(name, 0)
                    )
    transfer = delta.get("vllm:nixl_bytes_transferred_sum", 0)
    failures = delta.get("vllm:nixl_num_failed_transfers_total", 0)
    prefill_work = delta.get("vllm:request_prefill_kv_computed_tokens_sum", 0)
    result = {
        "scope": "C1_next_token_probabilities_and_prefix_facts_not_full_model_quality",
        "passed": all(r["numerical_gate"] and r["content_gate"] for r in records)
        and transfer > 0
        and failures == 0
        and prefill_work <= 16 * len(records),
        "probes": len(records),
        "records": records,
        "decoder_counter_deltas": delta,
        "actual_transfer_bytes": transfer,
        "decoder_local_prefill_tokens": prefill_work,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return {
        key: result[key]
        for key in [
            "passed",
            "probes",
            "actual_transfer_bytes",
            "decoder_local_prefill_tokens",
        ]
    }
