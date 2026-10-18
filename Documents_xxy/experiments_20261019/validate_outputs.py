"""Small non-forced-output gate, outside timed replay and not an accuracy suite."""

import json
from pathlib import Path

import httpx
from transformers import AutoTokenizer

PROBES = [
    ("Compute 2+2. Respond with only the integer.", "4"),
    ("Compute 3+5. Respond with only the integer.", "8"),
    ("Compute 4*5. Respond with only the integer.", "20"),
]


async def validate(model, endpoints, destination: Path):
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    records = []
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        for worker, endpoint in enumerate(endpoints):
            for index, (prompt, expected) in enumerate(PROBES):
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                )
                body = {
                    "model": "experiment",
                    "prompt": ids,
                    "max_tokens": 16,
                    "temperature": 0,
                    "return_token_ids": True,
                    "kv_transfer_params": {
                        "lmcache.skip_save": True,
                        "experiment.phase": "validation",
                    },
                }
                response = await client.post(
                    endpoint + "/v1/completions",
                    json=body,
                    headers={"X-Request-ID": f"quality-w{worker}-{index}"},
                )
                response.raise_for_status()
                result = response.json()
                actual = result["choices"][0]["text"].strip()
                records.append(
                    {
                        "worker": worker,
                        "prompt": prompt,
                        "expected": expected,
                        "actual": actual,
                        "passed": actual == expected,
                        "output_token_ids": result["choices"][0].get("token_ids"),
                    }
                )
    destination.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    if not all(row["passed"] for row in records):
        raise RuntimeError(f"basic output gate failed: {destination}")
    return records
