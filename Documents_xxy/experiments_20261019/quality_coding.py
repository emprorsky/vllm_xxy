"""Small non-forced code-generation regression probe; not a benchmark score."""

import asyncio
import json
import subprocess
import sys

import httpx
from transformers import AutoTokenizer

TASKS = [
    (
        "sum_even",
        "Return the sum of even integers in nums. Empty input returns 0.",
        [([[]], 0), ([[1, 2, 3, 4]], 6), ([[-4, -3, 0, 8]], 4)],
    ),
    (
        "reverse_words",
        "Return words of text in reverse order, joined by single spaces.",
        [([""], ""), (["a b c"], "c b a"), (["  one   two "], "two one")],
    ),
    (
        "unique_ordered",
        "Return unique integers from nums in first occurrence order.",
        [([[]], []), ([[3, 1, 3, 2, 1]], [3, 1, 2]), ([[0, 0]], [0])],
    ),
    (
        "flatten_once",
        "Given a list of lists, flatten exactly one level and return a list.",
        [([[]], []), ([[[1, 2], [], [3]]], [1, 2, 3]), ([[[[1]], [2]]], [[1], 2])],
    ),
    (
        "is_balanced",
        "Return whether parentheses in text are balanced. Ignore other characters.",
        [([""], True), (["a(b(c)d)"], True), ([")("], False), (["(()"], False)],
    ),
    (
        "first_duplicate",
        "Return the first repeated integer encountered in nums, or None if absent.",
        [([[]], None), ([[2, 1, 3, 1, 2]], 1), ([[0, 0]], 0), ([[1, 2]], None)],
    ),
]

CHILD = r"""
import ast, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
payload = json.load(sys.stdin)
try:
    tree = ast.parse(payload["code"])
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            raise ValueError("imports/classes excluded by probe contract")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder access excluded")
    names = ["len", "range", "enumerate", "sum", "min", "max", "abs", "any", "all",
             "zip", "sorted", "reversed", "int", "str", "float", "bool", "dict",
             "list", "tuple", "set", "isinstance", "ValueError", "TypeError"]
    built = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    namespace = {"__builtins__": {name: built[name] for name in names}}
    exec(compile(tree, "<generated>", "exec"), namespace)
    fn = namespace[payload["name"]]
    results = [fn(*args) == expected for args, expected in payload["tests"]]
    print(json.dumps({"passed": all(results), "assertions": results}))
except BaseException as exc:
    print(json.dumps({"passed": False, "error": type(exc).__name__ + ": " + str(exc)}))
"""


def execute_code(code, name, tests):
    if "```" in code:
        parts = code.split("```")
        code = parts[1].removeprefix("python").strip()
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", CHILD],
            input=json.dumps({"code": code, "name": name, "tests": tests}),
            text=True,
            capture_output=True,
            timeout=4,
        )
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, ValueError) as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


async def evaluate(model, endpoints, output):
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    records = []
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        for worker, endpoint in enumerate(endpoints):
            for name, requirement, tests in TASKS:
                prompt = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": (
                                f"Write Python function {name}. {requirement} "
                                "Return only the function code. Use no imports."
                            ),
                        }
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                response = await client.post(
                    endpoint + "/v1/completions",
                    json={
                        "model": "experiment",
                        "prompt": prompt,
                        "max_tokens": 256,
                        "temperature": 0,
                        "kv_transfer_params": {
                            "lmcache.skip_save": True,
                            "experiment.phase": "validation",
                        },
                    },
                )
                response.raise_for_status()
                generated = response.json()
                code = generated["choices"][0]["text"]
                outcome = await asyncio.to_thread(execute_code, code, name, tests)
                records.append(
                    {
                        "worker": worker,
                        "task": name,
                        "requirement": requirement,
                        "tests": tests,
                        "response": generated,
                        **outcome,
                    }
                )
    result = {
        "scope": "six_handwritten_code_probes_not_HumanEval_or_agent_success",
        "tasks": len(records),
        "passed": sum(r["passed"] for r in records),
        "records": records,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return {k: result[k] for k in ["scope", "tasks", "passed"]}
