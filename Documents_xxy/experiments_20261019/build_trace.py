"""Build explicitly labelled coding mechanisms or public trajectory replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def save(args, rows, metadata):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    args.output.write_text(content)
    lengths = [len(r["prompt_token_ids"]) for r in rows]
    metadata.update(
        {
            "requests": len(rows),
            "sessions": len({r["session_id"] for r in rows}),
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "model": args.model,
            "seed": args.seed,
            "input_tokens_min": min(lengths) if lengths else 0,
            "input_tokens_max": max(lengths) if lengths else 0,
            "input_tokens_mean": sum(lengths) / max(len(lengths), 1),
            "output_budget": args.output_tokens,
        }
    )
    args.output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


def synthetic(args, tokenizer):
    rng = random.Random(args.seed)
    sources = [
        p
        for p in sorted(Path("vllm/v1/core/sched").glob("*.py"))
        if p.stat().st_size >= 8192
    ]
    if not sources:
        raise ValueError("run from the vLLM repository root")
    group_prefixes = []
    for group in range(args.groups):
        source = sources[group % len(sources)]
        raw = source.read_text()
        ids = tokenizer.encode(raw, add_special_tokens=False)
        offset = (group // len(sources) * 193) % max(len(ids) - args.prefix_tokens, 1)
        snippet = tokenizer.decode(ids[offset : offset + args.prefix_tokens])
        group_prefixes.append(f"Repository snapshot group={group} version=1\n{snippet}")
    rows = []
    for session in range(args.sessions):
        group = (
            0
            if args.workload == "hot" and rng.random() < 0.65
            else rng.randrange(args.groups)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a coding assistant. Inspect code, "
                    "propose a minimal fix, and explain tests."
                ),
            },
            {
                "role": "user",
                "content": group_prefixes[group]
                + f"\nTask {session}: explain the scheduling invariants "
                "and suggest one useful test.",
            },
        ]
        for turn in range(args.turns):
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            if len(ids) + args.output_tokens > args.max_model_len:
                break
            rows.append(
                {
                    "request_id": f"s{session:04d}-t{turn:02d}",
                    "session_id": f"s{session:04d}",
                    "turn_id": turn,
                    "source_id": "local_vllm_code_semi_synthetic",
                    "prompt_token_ids": ids,
                    "requested_output_budget": args.output_tokens,
                    "observed_gap_s": 0 if turn == 0 else args.gap,
                    "gap_origin": "synthetic",
                    "group": group,
                    "ignore_eos": True,
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "I will inspect the implementation "
                        "and test the state transition."
                    ),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool observation {turn}:\n"
                    + group_prefixes[group][-600:]
                    + f"\nTest case {session}-{turn} reports a cache miss. "
                    "Refine the explanation.",
                }
            )
    save(
        args,
        rows,
        {
            "kind": "semi_synthetic_coding_mechanism",
            "workload": args.workload,
            "groups": args.groups,
            "input_source": [str(p) for p in sources],
            "limitations": [
                "not production arrival data",
                "recorded synthetic assistant history",
                "no task-success claim",
            ],
        },
    )


def trajectories(args, tokenizer):
    rows = []
    sources = []
    filtered = 0
    # Exclude CTF and explicit test fixtures from the coding demonstration subset.
    paths = [
        p
        for p in args.trajectory_dir.rglob("*")
        if p.is_file()
        and p.name.endswith((".traj", ".traj.json"))
        and "/ctf/" not in str(p)
    ]
    for session, path in enumerate(sorted(paths)[: args.sessions]):
        data = json.loads(path.read_text())
        messages = []
        turn = 0
        for message in data.get("messages", data.get("history", [])):
            role = message.get("role")
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if message.get("tool_calls"):
                content += "\nTool calls: " + json.dumps(message["tool_calls"])
            if role == "tool":
                role, content = "user", "Tool result:\n" + content
            if role not in {"system", "user", "assistant"}:
                continue
            if role == "assistant" and messages and messages[-1]["role"] == "user":
                ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                if len(ids) + args.output_tokens <= args.max_model_len:
                    rows.append(
                        {
                            "request_id": f"public-{session:04d}-{turn:03d}",
                            "session_id": f"public-{session:04d}",
                            "turn_id": turn,
                            "source_id": str(path),
                            "prompt_token_ids": ids,
                            "requested_output_budget": args.output_tokens,
                            "observed_gap_s": 0 if turn == 0 else args.gap,
                            "gap_origin": "synthetic_no_recorded_timestamps",
                            "ignore_eos": True,
                        }
                    )
                    turn += 1
                else:
                    filtered += 1
                if turn >= args.turns:
                    break
            messages.append({"role": role, "content": content})
        sources.append(
            {
                "path": str(path),
                "instance_id": data.get("instance_id", path.name),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "accepted_calls": turn,
            }
        )
    save(
        args,
        rows,
        {
            "kind": "public_coding_trajectory_text_replay",
            "sources": sources,
            "filtered_over_context_calls": filtered,
            "limitations": [
                (
                    "task identity is source instance_id; "
                    "demonstration variants may overlap"
                ),
                "tool roles normalized to textual user observations",
                "tool gaps simulated",
                "outputs not historical forced outputs",
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--sessions", type=int, default=24)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gap", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workload", choices=["hot", "uniform"], default="hot")
    args = parser.parse_args()
    if (
        min(
            args.sessions,
            args.turns,
            args.groups,
            args.prefix_tokens,
            args.output_tokens,
        )
        < 1
    ):
        parser.error("counts must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.trajectory_dir:
        trajectories(args, tokenizer)
    else:
        synthetic(args, tokenizer)


if __name__ == "__main__":
    main()
