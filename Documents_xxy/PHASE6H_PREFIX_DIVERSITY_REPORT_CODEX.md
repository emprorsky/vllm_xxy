# Phase 6h 实施报告：Prefix Diversity 泛化确认

日期：2026-08-29（UTC）

## 1. 结论

Phase 6h 不再调整 admission heuristic，而是把 prefix-repetition workload 的 prefix
种类从 8 增到 16，降低单个 prefix 的复用频率。固定 W=8/aging=10s 的
cache-affinity 完成两对 fresh-process 反向顺序 A/B：C1→T1、T2→C2。

两对 output throughput 分别提升 `+5.632%`、`+2.758%`，均为正；C/T 两轮均值
从 `828.435` 提升到 `863.265 tok/s`，即 `+4.204%`，通过预设的“每对为正且
均值至少 +2%”泛化确认 Gate。

与 8-prefix workload 不同，16-prefix 下延迟也整体改善：TTFT mean/p50/p99 均值
分别 `-0.682%/-5.009%/-2.235%`，mean/p99 TPOT `-4.070%/-15.432%`。代价是
ITL p99 `+4.206%`，因此不能宣称所有延迟指标都改善。

这证明收益不只存在于恰好 8 个 prefix 的输入分布。安全结论是：在同一 RTX 4090、
Qwen2.5-7B、1250-block 压力条件下，cache-affinity 在 8-prefix 和 16-prefix 两种
workload shape 均通过两对反向顺序吞吐确认；但 1000-block 外推仍失败，不能概括
为跨 KV budget 普遍提升。

## 2. 实验边界

四轮均使用独立 fresh server process，共同配置：

- RTX 4090 / Qwen2.5-7B-Instruct；
- prefix caching、1250 GPU KV blocks；
- recompute-aware preemption、LRU retention；
- 16 个 prefix、prefix 900 tokens、suffix 64 tokens、output 1024 tokens；
- 48 warmup + 192 正式请求、max concurrency 48；
- temperature 0.7、ignore-eos、seed 42；
- 每轮 192/192 成功，185,094 input tokens、196,608 output tokens。

唯一变量：C 使用 default admission；T 使用 cache-affinity、window=8、aging=10s。
执行顺序为 C1→T1、T2→C2，以降低顺序偏差。

## 3. 完整结果

| 指标 | C1 | T1 | T2 | C2 |
|---|---:|---:|---:|---:|
| duration (s) | 235.808 | 223.235 | 232.450 | 238.861 |
| output throughput (tok/s) | 833.763 | 880.722 | 845.809 | 823.108 |
| TTFT mean (ms) | 19886.033 | 20415.653 | 20950.621 | 21764.374 |
| TTFT p50 (ms) | 22589.121 | 22240.340 | 22267.088 | 24265.480 |
| TTFT p99 (ms) | 40573.452 | 38835.265 | 40897.094 | 40981.720 |
| TPOT mean (ms) | 31.820 | 29.466 | 30.163 | 30.339 |
| TPOT p99 (ms) | 63.103 | 52.911 | 52.758 | 61.849 |
| ITL p99 (ms) | 104.504 | 105.367 | 106.024 | 98.355 |

| 指标 | Pair 1：C1→T1 | Pair 2：C2→T2 | C mean | T mean | 均值变化 |
|---|---:|---:|---:|---:|---:|
| duration | -5.332% | -2.684% | 237.334 s | 227.842 s | -3.999% |
| output throughput | +5.632% | +2.758% | 828.435 | 863.265 | +4.204% |
| TTFT mean | +2.663% | -3.739% | 20825.204 ms | 20683.136 ms | -0.682% |
| TTFT p50 | -1.544% | -8.236% | 23427.300 ms | 22253.714 ms | -5.009% |
| TTFT p99 | -4.284% | -0.206% | 40777.586 ms | 39866.180 ms | -2.235% |
| TPOT mean | -7.398% | -0.579% | 31.080 ms | 29.815 ms | -4.070% |
| TPOT p99 | -16.150% | -14.699% | 62.476 ms | 52.835 ms | -15.432% |
| ITL p99 | +0.826% | +7.797% | 101.430 ms | 105.696 ms | +4.206% |

16-prefix control 相比 8-prefix control 更慢，说明更高 prefix diversity 实际降低了
locality，不能复用 Phase 6d 的 control。本报告只在同一 16-prefix workload 内做
fresh-process 配对和组均值比较。

## 4. 机制数据

| 指标 | C1 | T1 | T2 | C2 |
|---|---:|---:|---:|---:|
| prefix cache queries | 231,366 | 231,366 | 231,366 | 231,366 |
| prefix cache hits | 156,368 | 154,112 | 152,784 | 153,776 |
| preemptions | 464 | 400 | 428 | 421 |
| selection calls | 0 | 12,282 | 12,891 | 0 |
| candidate probes | 0 | 97,146 | 100,858 | 0 |
| candidates with hits | 0 | 59,379 | 67,104 | 0 |
| raw reordered selections | 0 | 292 | 330 | 0 |
| aged selection attempts | 0 | 11,478 | 12,074 | 0 |
| successful admissions | 0 | 631 | 657 | 0 |
| admitted reordered | 0 | 82 | 81 | 0 |
| admitted aged | 0 | 517 | 545 | 0 |
| admitted cached tokens | 0 | 437,696 | 462,640 | 0 |

两轮 treatment 共 163 次 successful reordered admission，证明策略真实改变
admission。reorder ratio 分别为 13.0% 和 12.3%。

值得注意的是，T1/T2 的 prefix-cache hits 均没有高于相邻 control，收益不能简化为
“APC hit 数更多”。T1 比 C1 少 64 次 preemption，而 T2 比 C2 多 7 次，说明
preemption counter 也不能单独解释两对收益。更可能的机制是请求执行顺序改变了
decode/prefill 竞争和抢占时序。

Phase 6i 随后补充了此前缺失的 post-preemption recompute telemetry。在一对独立
C4→T4 attribution A/B 中，cache-affinity 吞吐 +2.695%，同时 resume recompute
-5.523%、total local context compute -3.372%、engine outputs -6.415%。treatment
preemption 反而多 12.531%，因此机制证据支持“降低每次恢复的实际重算成本并改善
batch shape”，而不是“减少抢占次数”。详见
`PHASE6I_MECHANISM_ATTRIBUTION_REPORT_CODEX.md`。

## 5. 决策

- 16-prefix 两对反向顺序确认通过，纳入最终可复现性能证据；
- W=8/aging=10s 继续作为唯一 balanced opt-in preset，默认 admission 不变；
- 不继续调 window/aging heuristic；
- 简历可以陈述 8-prefix 和 16-prefix 两种 workload shape 均获得正向吞吐收益，
  但必须限定 RTX 4090、Qwen2.5-7B、1250 KV blocks；
- Phase 6i 机制归因已完成；下一步优先换模型做最小泛化，不再扫描参数。

原始产物：

- `phase6h_benchserve_1250_prefix16_default_c1.json`；
- `phase6h_benchserve_1250_prefix16_cache_affinity_aging10_t1.json`；
- `phase6h_benchserve_1250_prefix16_cache_affinity_aging10_t2.json`；
- `phase6h_benchserve_1250_prefix16_default_c2.json`；
- `phase6h_admission_metrics.json`。

简历可安全表述为：

> 为 vLLM V1 实现有界 cache-affinity admission 与 aging 公平性保护；在官方
> `vllm bench serve` 的 RTX 4090/Qwen2.5-7B/1250-block 高 KV 压力测试中，
> 8-prefix workload 两对反向 A/B 吞吐均值提升 3.2%，16-prefix workload 提升
> 4.2%；后者同时使 TTFT p99 降低 2.2%、mean TPOT 降低 4.1%。
