# Phase 6i 实施报告：Cache-affinity 性能机制归因

日期：2026-08-29（UTC）

## 1. 目标

Phase 6h 已在 16-prefix workload 完成两对反向顺序确认，output throughput
均值提升 `4.204%`。但 prefix-cache hits 和 preemption 次数都不能稳定解释收益，
所以 Phase 6i 不再调整 heuristic，而是回答两个问题：

1. cache-affinity 是否减少了抢占恢复产生的真实重复计算；
2. 它是否改变了 iteration batch shape 和 queue/prefill/decode 时间分布。

## 2. C3 审计轮发现的测量盲区

先运行了一轮 fresh-process control C3。正式 192 请求的 output throughput 为
`852.378 tok/s`，服务累计 48 warmup + 192 正式请求，共发生 `423` 次 preemption。

现有指标却给出：

- `prompt_tokens_by_source{source="local_compute"}=72,470`；
- `request_prefill_kv_computed_tokens_sum=72,470`；
- 两者差值为 0。

源码审计确认这不是“没有重复计算”，而是指标定义如此：`prefill_stats` 只记录首次
scheduled prefill，明确排除了 post-preemption repeat prefill。因此，原计划用上述
差值估算重复 prefill 是错误的，C3 只能作为 telemetry 审计样本，不能用于机制结论。

原始产物：

- `phase6i_benchserve_1250_prefix16_default_c3.json`；
- `phase6i_metrics_1250_prefix16_default_c3.json`（旧 schema，仅保留审计证据）。

## 3. 最小只读 telemetry

为避免把新策略逻辑与测量混在一起，只在已有 `KVPreemptionStats`/Prometheus
counter 通路增加两个累计量：

- `kv_preemption_preempted_computed_tokens`：每次抢占把 request computed-token
  frontier 归零时，累计该 frontier；
- `kv_preemption_resume_recompute_tokens`：恢复调度区间与抢占前旧 frontier 的
  重叠 token 数，即真正重复调度的计算。

第二个计数不把“恢复后尚未做过的新进度”误算为重算，也会扣除 prefix cache 已恢复
的前缀。若恢复途中再次被抢占，未恢复完的旧 frontier 会保留，避免漏计。

`capture_phase6i_metrics.py` 只读取 `/metrics`，并显式禁用 localhost proxy。schema
v2 会在服务没有新 counter 时直接报错，不再把缺失指标静默当 0。它同时保留：

- iteration tokens 的 sum/count/mean；
- queue/prefill/decode/inference time 的 sum/count/mean；
- prefix cache、preemption、admission counters；
- 首次 prefill local compute，以及加上 resume recompute 后的总本地计算估计。

## 4. 验证

单测覆盖三类边界：实际抢占 frontier 累计、cache hit 与新进度不误计、恢复途中再次
抢占不丢失旧 frontier。完成结果：

- Scheduler 定向测试：`3 passed`；
- metrics 序列化测试：`12 passed`；
- 相关 Python 文件 ruff：通过；
- snapshot parser schema-v2 smoke：通过。

## 5. C4→T4 attribution 结果

使用 Phase 6h 的固定 16-prefix 配置执行一对 fresh-process C4→T4：

1. 每轮 48 warmup + 192 正式请求；
2. 正式请求结束后、停服前抓取 schema-v2 snapshot；
3. 唯一变量是 default admission 与 cache-affinity W=8/aging=10s；
4. 两轮正式请求均 192/192 成功，input/output tokens 完全一致。

正式 benchmark 结果：

| 指标 | C4 default | T4 cache-affinity | 变化 |
|---|---:|---:|---:|
| duration (s) | 235.729 | 229.543 | -2.624% |
| output throughput (tok/s) | 834.044 | 856.521 | **+2.695%** |
| TTFT mean (ms) | 21192.300 | 19793.473 | **-6.601%** |
| TTFT p50 (ms) | 23299.554 | 22648.854 | -2.793% |
| TTFT p99 (ms) | 38872.666 | 38103.583 | -1.978% |
| TPOT mean (ms) | 30.477 | 30.469 | -0.026% |
| TPOT p99 (ms) | 62.671 | 53.386 | **-14.816%** |
| ITL p99 (ms) | 100.611 | 102.573 | +1.950% |

Prometheus snapshot 包含 warmup + 正式请求，共 240 个请求：

| 机制指标 | C4 | T4 | 变化 |
|---|---:|---:|---:|
| preemptions | 407 | 458 | +12.531% |
| preempted computed-token frontiers | 483,302 | 541,238 | +11.988% |
| **resume recompute tokens** | **197,286** | **186,390** | **-5.523%** |
| first-prefill local compute | 74,566 | 76,294 | +2.317% |
| **total local context compute** | **271,852** | **262,684** | **-3.372%** |
| generation + local context compute | 517,612 | 508,444 | -1.771% |
| engine outputs | 15,495 | 14,501 | -6.415% |
| compute tokens / engine output | 33.405 | 35.063 | +4.962% |
| request queue time mean (s) | 16.842 | 15.683 | -6.877% |
| request decode time mean (s) | 32.397 | 31.950 | -1.382% |
| request inference time mean (s) | 32.640 | 32.197 | -1.357% |
| request prefill time mean (s) | 0.243 | 0.248 | +1.897% |

T4 还记录了 12,566 次 selection、98,307 次 candidate probe、690 次成功 admission，
其中 80 次真正绕过基础队头，证明策略已激活并改变了执行顺序。

## 6. 机制结论

结果排除了“收益来自更多 APC hits”或“收益来自更少 preemption”这两种解释：T4
首次 prefill cache hits 少 1,728，preemption 反而多 51 次。真实变化是：

1. 少掉的 cache hits 让首次 prefill 多计算 1,728 tokens；
2. admission 顺序改变后，恢复重算少 10,896 tokens；
3. 因而净减少 9,168 个 local context compute tokens（-3.372%）；
4. 同时 engine output 次数减少 994 次，每次 output 承载的估算实际计算量提高
   4.962%，queue time 降低 6.877%。

所以当前最可信的解释是：cache-affinity 不是简单减少抢占次数，而是改善抢占恢复的
成本和 batch shape，用更少的重复上下文计算、更少的 engine steps 完成相同输出。
该结论来自一对 attribution A/B，作为 Phase 6h 两对性能确认的机制支持；它不应被
表述为已证明唯一因果路径。

需要特别说明：原生 `iteration_tokens_total` 的 sum 仍使用首次 prefill stats，排除
post-preemption recompute。报告中的 `compute tokens / engine output` 使用
`first local compute + resume recompute + generation` 重建，适用于本轮无 speculative
decode 的固定 workload。

原始产物：

- `phase6i_benchserve_1250_prefix16_default_c4.json`；
- `phase6i_benchserve_1250_prefix16_cache_affinity_aging10_t4.json`；
- `phase6i_metrics_1250_prefix16_default_c4.json`；
- `phase6i_metrics_1250_prefix16_cache_affinity_aging10_t4.json`。

该轮用于 attribution，不替代 Phase 6h 已完成的 performance confirmation。
