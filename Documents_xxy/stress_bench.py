"""压力场景 benchmark：触发抢占 + 前缀共享，测 KV-aware 调度收益。

场景设计要点（对应设计文档的压力测试章节）：
1. 共享前缀池：8 个长 system prompt (~900 tokens)，请求随机选一个，
   为 prefix cache 提供命中空间
2. 混合输出长度 [512, 1024, 1536] → 长短请求竞争 KV cache，解码期持续膨胀
3. 服务端压小 KV cache 触发抢占（关键配置，客户端无法做到）：
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-7B-Instruct \
       --gpu-memory-utilization 0.75 \
       --num-gpu-blocks-override 1250 \
       --max-model-len 8192 --enable-prefix-caching
   ※ 坑：新版本 vLLM 默认 scheduler_reserve_full_isl=True，准入时检查完整输入序列，
     单纯高并发/KV 压到 45k 都触发不了抢占；必须 --num-gpu-blocks-override 压到
     20k tokens（1250 块）才稳定复现抢占
4. 从 /metrics 抓 vllm:num_preemptions_total 和 prefix cache 命中率（差值）

用法: python stress_bench.py [输出json路径]
"""

import asyncio
import hashlib
import json
import os
import random
import sys
import time
from contextlib import suppress

import aiohttp

API = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ---- 场景参数（可用环境变量覆盖，便于 A/B 不同压力形态）----
NUM_PREFIX_POOLS = int(os.environ.get("BENCH_PREFIX_POOLS", 8))  # 共享前缀池数量
PREFIX_TOKENS = 900  # 每个 system prompt 约 900 token
SHARED_RATIO = float(os.environ.get("BENCH_SHARED_RATIO", 0.7))  # 共享前缀请求比例
CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", 48))  # 并发数（压满 KV cache）
TOTAL_REQUESTS = int(os.environ.get("BENCH_TOTAL_REQUESTS", 192))  # 总请求数
POOL_MODE = os.environ.get(
    "BENCH_POOL_MODE", "random"
)  # random=随机选池 / cyclic=轮转扫描
GROUP_SIZE = int(os.environ.get("BENCH_GROUP_SIZE", 8))  # cyclic 模式下每池连续请求数
OUTPUT_LENS = [
    int(value)
    for value in os.environ.get("BENCH_OUTPUT_LENS", "512,1024,1536").split(",")
]
RNG_SEED = 42
REQUEST_SEED_BASE = int(os.environ.get("BENCH_REQUEST_SEED_BASE", 100_000))
TEMPERATURE = float(os.environ.get("BENCH_TEMPERATURE", 0.7))
IGNORE_EOS = os.environ.get("BENCH_IGNORE_EOS", "0").lower() in {
    "1",
    "true",
    "yes",
}
WARMUP_REQUESTS = int(os.environ.get("BENCH_WARMUP_REQUESTS", 1))

random.seed(RNG_SEED)

# 预生成共享前缀（模拟多轮对话/同 system prompt 的业务负载）
SHARED_PREFIXES = []
for i in range(NUM_PREFIX_POOLS):
    # 重复文本凑 ~900 token，内容带编号确保 hash 不同
    base = (
        f"你是一个专业领域助手（角色 {i}）。你的任务是处理第 {i} 类业务问题，"
        "包括需求分析、方案设计、实施规划和结果评估等多个环节。"
    ) * 30
    SHARED_PREFIXES.append(base)

UNIQUE_PREFIX = "以下是一次性独立任务的上下文，不与其他请求共享："


def build_prompt(req_id: int) -> tuple[str, int]:
    """返回 (prompt, prefix_pool_id)；-1 表示独享前缀"""
    if POOL_MODE == "cyclic":
        # 循环/扫描型负载：每 BENCH_GROUP_SIZE 个连续请求用同一个池，
        # 池编号随请求推进轮转。LRU 的经典失效场景——某池被再次需要时
        # 恰好位于 LRU 队尾。
        pid = (req_id // GROUP_SIZE) % NUM_PREFIX_POOLS
        prefix = SHARED_PREFIXES[pid]
    elif random.random() < SHARED_RATIO:
        pid = random.randrange(NUM_PREFIX_POOLS)
        prefix = SHARED_PREFIXES[pid]
    else:
        pid = -1
        prefix = UNIQUE_PREFIX
    user_msg = f"[请求{req_id}] 请基于以上背景，详细阐述你的方案。"
    return f"{prefix}\n{user_msg}", pid


async def fetch_metrics(session: aiohttp.ClientSession) -> dict:
    """抓 vLLM /metrics 中与 KV/抢占相关的指标"""
    ret = {}
    try:
        async with session.get(f"{API}/metrics") as r:
            text = await r.text()
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for key in (
                "vllm:num_preemptions_total",
                "vllm:gpu_prefix_cache_hits_total",
                "vllm:gpu_prefix_cache_queries_total",
                "vllm:prefix_cache_hits_total",
                "vllm:prefix_cache_queries_total",
                "vllm:kv_retention_resolver_calls_total",
                "vllm:kv_retention_candidates_total",
                "vllm:kv_retention_candidates_with_hits_total",
                "vllm:kv_retention_blocks_total",
                "vllm:kv_retention_normal_blocks_total",
                "vllm:kv_retention_resumed_blocks_total",
                "vllm:kv_retention_high_priority_blocks_total",
                "vllm:kv_retention_avoided_evictions_total",
                "vllm:kv_retention_fallback_blocks_total",
                "vllm:kv_admission_selection_calls_total",
                "vllm:kv_admission_candidates_total",
                "vllm:kv_admission_candidate_probes_total",
                "vllm:kv_admission_candidates_with_hits_total",
                "vllm:kv_admission_reordered_total",
                "vllm:kv_admission_aged_selections_total",
                "vllm:kv_admission_selected_cached_tokens_total",
                "vllm:kv_admission_admitted_total",
                "vllm:kv_admission_admitted_reordered_total",
                "vllm:kv_admission_admitted_aged_total",
                "vllm:kv_admission_admitted_cached_tokens_total",
                "vllm:kv_preemption_shortfall_events_total",
                "vllm:kv_preemption_shortfall_blocks_total",
                "vllm:kv_preemption_candidate_estimates_total",
                "vllm:kv_preemption_deferred_candidates_total",
                "vllm:kv_preemption_reclaimable_blocks_total",
                "vllm:kv_preemption_selected_reclaimable_blocks_total",
                "vllm:kv_preemption_sufficient_selections_total",
                "vllm:kv_preemption_zero_progress_selections_total",
                "vllm:scheduling_feature_prefix_requests_total",
                "vllm:scheduling_feature_prefix_resolutions_total",
                "vllm:scheduling_feature_prefix_cache_hits_total",
                "vllm:scheduling_feature_invalidations_total",
            ):
                if line.startswith(key + "{") or line.startswith(key + " "):
                    fields = line.rsplit(" ", 1)
                    if len(fields) == 2:
                        with suppress(ValueError):
                            ret[key] = float(fields[1])
                    break
    except Exception as e:
        ret["error"] = str(e)
    return ret


async def warmup(session: aiohttp.ClientSession):
    """预热首次采样/JIT 路径；指标在预热后取基准值。"""
    async def one_warmup(req_id: int):
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": f"[benchmark warmup {req_id}] 仅用于预热采样路径。",
                }
            ],
            "max_tokens": 16,
            "temperature": TEMPERATURE,
            "seed": REQUEST_SEED_BASE - req_id - 1,
            "ignore_eos": True,
            "stream": False,
        }
        async with session.post(f"{API}/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"warmup HTTP {resp.status}: {(await resp.text())[:200]}"
                )
            await resp.read()

    await asyncio.gather(*(one_warmup(i) for i in range(WARMUP_REQUESTS)))


async def one_request(
    session: aiohttp.ClientSession, sem: asyncio.Semaphore, req_id: int, results: list
):
    prompt, pid = build_prompt(req_id)
    out_len = random.choice(OUTPUT_LENS)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": out_len,
        "temperature": TEMPERATURE,
        "seed": REQUEST_SEED_BASE + req_id,
        "ignore_eos": IGNORE_EOS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    rec = {
        "req_id": req_id,
        "prefix_pool": pid,
        "output_len": out_len,
        "ttft": None,
        "itl_mean": None,
        "itl_p99": None,
        "completion_tokens": None,
        "content_chunks": 0,
        "output_sha256": None,
        "finish_reason": None,
        "elapsed": None,
        "error": None,
    }
    t0 = time.perf_counter()
    content_hasher = hashlib.sha256()
    content_stamps = []
    try:
        async with (
            sem,
            session.post(f"{API}/v1/chat/completions", json=payload) as resp,
        ):
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                results.append(rec)
                return
            async for line in resp.content:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if usage is not None:
                    rec["completion_tokens"] = usage.get("completion_tokens")
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    rec["finish_reason"] = choice["finish_reason"]
                content = choice.get("delta", {}).get("content")
                if content:
                    now = time.perf_counter()
                    if not content_stamps:
                        rec["ttft"] = now - t0
                    content_stamps.append(now)
                    content_hasher.update(content.encode())
                    rec["content_chunks"] += 1
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["elapsed"] = time.perf_counter() - t0
    rec["output_sha256"] = content_hasher.hexdigest()
    if len(content_stamps) > 1:
        itls = [
            current - previous
            for previous, current in zip(content_stamps, content_stamps[1:])
        ]
        if itls:
            itls.sort()
            rec["itl_mean"] = sum(itls) / len(itls)
            rec["itl_p99"] = itls[min(len(itls) - 1, int(0.99 * len(itls)))]
    results.append(rec)


def pct(xs: list, p: float):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


async def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "stress_bench_baseline.json"
    print(
        f"[stress_bench] 并发={CONCURRENCY} 总请求={TOTAL_REQUESTS} "
        f"共享前缀池={NUM_PREFIX_POOLS}(~{PREFIX_TOKENS}tok) 共享比例={SHARED_RATIO}"
    )
    print(f"[stress_bench] 输出长度混合: {OUTPUT_LENS}")
    print(
        f"[stress_bench] temperature={TEMPERATURE} ignore_eos={IGNORE_EOS} "
        f"warmup={WARMUP_REQUESTS}"
    )

    sem = asyncio.Semaphore(CONCURRENCY)
    results = []
    conn = aiohttp.TCPConnector(limit=CONCURRENCY + 8)
    async with aiohttp.ClientSession(
        connector=conn, timeout=aiohttp.ClientTimeout(total=1800)
    ) as s:
        # 等服务就绪
        for _ in range(60):
            try:
                async with s.get(f"{API}/v1/models"):
                    break
            except Exception:
                await asyncio.sleep(2)
        else:
            print("服务不可达")
            sys.exit(1)

        await warmup(s)
        m_before = await fetch_metrics(s)
        print(f"[stress_bench] 指标(前): {m_before}")

        t_start = time.perf_counter()
        await asyncio.gather(
            *[one_request(s, sem, i, results) for i in range(TOTAL_REQUESTS)]
        )
        wall = time.perf_counter() - t_start

        m_after = await fetch_metrics(s)
        print(f"[stress_bench] 指标(后): {m_after}")

    ok = [r for r in results if r["error"] is None]
    err = [r for r in results if r["error"] is not None]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    itl_means = [r["itl_mean"] for r in ok if r["itl_mean"] is not None]
    itl_p99s = [r["itl_p99"] for r in ok if r["itl_p99"] is not None]
    completion_tokens = [
        r["completion_tokens"] for r in ok if r["completion_tokens"] is not None
    ]
    total_tokens = sum(completion_tokens)
    total_content_chunks = sum(r["content_chunks"] for r in ok)

    def delta(key):
        if key not in m_before or key not in m_after:
            return None
        return m_after[key] - m_before[key]

    def first_delta(*keys):
        for key in keys:
            value = delta(key)
            if value is not None:
                return value
        return None

    summary = {
        "config": {
            "concurrency": CONCURRENCY,
            "total_requests": TOTAL_REQUESTS,
            "prefix_pools": NUM_PREFIX_POOLS,
            "shared_ratio": SHARED_RATIO,
            "output_lens": OUTPUT_LENS,
            "shape_seed": RNG_SEED,
            "request_seed_base": REQUEST_SEED_BASE,
            "temperature": TEMPERATURE,
            "ignore_eos": IGNORE_EOS,
            "warmup_requests": WARMUP_REQUESTS,
        },
        "wall_time_s": wall,
        "ok": len(ok),
        "errors": len(err),
        "ttft": {
            "mean": sum(ttfts) / len(ttfts) if ttfts else None,
            "p50": pct(ttfts, 0.5),
            "p90": pct(ttfts, 0.9),
            "p99": pct(ttfts, 0.99),
        },
        "chunk_itl": {
            "mean": sum(itl_means) / len(itl_means) if itl_means else None,
            "p99": pct(itl_p99s, 0.99),
        },
        "usage_missing": len(ok) - len(completion_tokens),
        "completion_tokens": total_tokens,
        "content_chunks": total_content_chunks,
        "output_throughput_tok_s": total_tokens / wall if wall else None,
        "content_chunk_throughput_s": total_content_chunks / wall if wall else None,
        "request_throughput": len(ok) / wall if wall else None,
        "preemptions_delta": delta("vllm:num_preemptions_total"),
        "prefix_hits_delta": first_delta(
            "vllm:gpu_prefix_cache_hits_total",
            "vllm:prefix_cache_hits_total",
        ),
        "prefix_queries_delta": first_delta(
            "vllm:gpu_prefix_cache_queries_total",
            "vllm:prefix_cache_queries_total",
        ),
        "retention": {
            "resolver_calls": delta("vllm:kv_retention_resolver_calls_total"),
            "candidates": delta("vllm:kv_retention_candidates_total"),
            "candidates_with_hits": delta(
                "vllm:kv_retention_candidates_with_hits_total"
            ),
            "blocks": delta("vllm:kv_retention_blocks_total"),
            "normal_blocks": delta("vllm:kv_retention_normal_blocks_total"),
            "resumed_blocks": delta("vllm:kv_retention_resumed_blocks_total"),
            "high_priority_blocks": delta(
                "vllm:kv_retention_high_priority_blocks_total"
            ),
            "avoided_evictions": delta("vllm:kv_retention_avoided_evictions_total"),
            "fallback_blocks": delta("vllm:kv_retention_fallback_blocks_total"),
        },
        "admission": {
            "selection_calls": delta("vllm:kv_admission_selection_calls_total"),
            "candidates": delta("vllm:kv_admission_candidates_total"),
            "candidate_probes": delta("vllm:kv_admission_candidate_probes_total"),
            "candidates_with_hits": delta(
                "vllm:kv_admission_candidates_with_hits_total"
            ),
            "reordered": delta("vllm:kv_admission_reordered_total"),
            "aged_selections": delta(
                "vllm:kv_admission_aged_selections_total"
            ),
            "selected_cached_tokens": delta(
                "vllm:kv_admission_selected_cached_tokens_total"
            ),
            "admitted": delta("vllm:kv_admission_admitted_total"),
            "admitted_reordered": delta(
                "vllm:kv_admission_admitted_reordered_total"
            ),
            "admitted_aged": delta("vllm:kv_admission_admitted_aged_total"),
            "admitted_cached_tokens": delta(
                "vllm:kv_admission_admitted_cached_tokens_total"
            ),
        },
        "kv_preemption": {
            "shortfall_events": delta(
                "vllm:kv_preemption_shortfall_events_total"
            ),
            "shortfall_blocks": delta(
                "vllm:kv_preemption_shortfall_blocks_total"
            ),
            "candidate_estimates": delta(
                "vllm:kv_preemption_candidate_estimates_total"
            ),
            "deferred_candidates": delta(
                "vllm:kv_preemption_deferred_candidates_total"
            ),
            "reclaimable_blocks": delta(
                "vllm:kv_preemption_reclaimable_blocks_total"
            ),
            "selected_reclaimable_blocks": delta(
                "vllm:kv_preemption_selected_reclaimable_blocks_total"
            ),
            "sufficient_selections": delta(
                "vllm:kv_preemption_sufficient_selections_total"
            ),
            "zero_progress_selections": delta(
                "vllm:kv_preemption_zero_progress_selections_total"
            ),
        },
        "scheduling_features": {
            "prefix_requests": delta(
                "vllm:scheduling_feature_prefix_requests_total"
            ),
            "prefix_resolutions": delta(
                "vllm:scheduling_feature_prefix_resolutions_total"
            ),
            "prefix_cache_hits": delta(
                "vllm:scheduling_feature_prefix_cache_hits_total"
            ),
            "invalidations": delta(
                "vllm:scheduling_feature_invalidations_total"
            ),
        },
        "errors_detail": [r["error"] for r in err[:5]],
    }
    with open(out_path, "w") as f:
        json.dump(
            {"summary": summary, "detail": results}, f, indent=2, ensure_ascii=False
        )

    print("\n===== 压力场景基线 =====")
    print(f"完成 {len(ok)}/{TOTAL_REQUESTS}  (失败 {len(err)})  总耗时 {wall:.1f}s")
    if ttfts:
        ttft = summary["ttft"]
        print(
            f"TTFT  mean={ttft['mean']:.2f}s  p50={ttft['p50']:.2f}s  "
            f"p90={ttft['p90']:.2f}s  p99={ttft['p99']:.2f}s"
        )
    if itl_means:
        print(
            f"chunk-ITL mean={summary['chunk_itl']['mean'] * 1000:.1f}ms  "
            f"p99={summary['chunk_itl']['p99'] * 1000:.1f}ms"
        )
    print(
        f"吞吐  output={summary['output_throughput_tok_s']:.1f} tok/s  "
        f"request={summary['request_throughput']:.2f} req/s  "
        f"usage_missing={summary['usage_missing']}"
    )
    print(f"抢占  preemptions_delta={summary['preemptions_delta']}")
    print(
        f"前缀  hits={summary['prefix_hits_delta']}  "
        f"queries={summary['prefix_queries_delta']}"
    )
    print(f"保留  {summary['retention']}")
    print(f"准入  {summary['admission']}")
    print(f"KV抢占  {summary['kv_preemption']}")
    print(f"特征  {summary['scheduling_features']}")
    if err:
        print(f"错误示例: {summary['errors_detail'][:2]}")
    print(f"\n结果已存 {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
