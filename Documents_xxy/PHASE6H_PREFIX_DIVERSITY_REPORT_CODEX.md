# Phase 6h 探索报告：Prefix Diversity 泛化筛选

日期：2026-08-29（UTC）

## 1. 结论

Phase 6h 不再调整 admission heuristic，而是把 prefix-repetition workload 的 prefix
种类从 8 增到 16，降低单个 prefix 的复用频率。在一次 fresh-process C→T 中，固定
W=8/aging=10s 的 cache-affinity 得到：

| 指标 | default C1 | cache-affinity T1 | T vs C |
|---|---:|---:|---:|
| output throughput | 833.763 tok/s | 880.722 tok/s | **+5.632%** |
| TTFT mean | 19886.033 ms | 20415.653 ms | +2.663% |
| TTFT p50 | 22589.121 ms | 22240.340 ms | -1.544% |
| TTFT p99 | 40573.452 ms | 38835.265 ms | -4.284% |
| TPOT mean | 31.820 ms | 29.466 ms | -7.398% |

这组 screening 通过预设的 `>= +2% throughput` 泛化门槛，而且 TTFT p50/p99、
mean/p99 TPOT 同向改善，优于 8-prefix workload 的 throughput/fairness 折中形态。
它证明收益不只存在于恰好 8 个 prefix 的输入分布。

不过当前只有一对 C→T，尚未做反向顺序确认。安全结论是“第二个 workload shape
上的初步正向泛化证据”，不能把 +5.6% 写成已确认均值。

## 2. 实验边界

两轮使用独立 fresh server process，共同配置：

- RTX 4090 / Qwen2.5-7B-Instruct；
- prefix caching、1250 GPU KV blocks；
- recompute-aware preemption、LRU retention；
- 16 个 prefix、prefix 900 tokens、suffix 64 tokens、output 1024 tokens；
- 48 warmup + 192 正式请求、max concurrency 48；
- temperature 0.7、ignore-eos、seed 42；
- 192/192 成功，185,094 input tokens、196,608 output tokens。

唯一变量：C1 使用 default admission；T1 使用 cache-affinity、window=8、
aging=10s。

## 3. 完整结果

| 指标 | C1 | T1 | T vs C |
|---|---:|---:|---:|
| duration (s) | 235.808 | 223.235 | -5.332% |
| output throughput (tok/s) | 833.763 | 880.722 | +5.632% |
| TTFT mean (ms) | 19886.033 | 20415.653 | +2.663% |
| TTFT p50 (ms) | 22589.121 | 22240.340 | -1.544% |
| TTFT p99 (ms) | 40573.452 | 38835.265 | -4.284% |
| TPOT mean (ms) | 31.820 | 29.466 | -7.398% |
| TPOT p99 (ms) | 63.103 | 52.911 | -16.150% |
| ITL p99 (ms) | 104.504 | 105.367 | +0.826% |

16-prefix control 相比 8-prefix control 更慢，说明更高 prefix diversity 实际降低了
locality，不能复用 Phase 6d 的 control。这里的比较只使用同一 16-prefix workload
的 fresh C1/T1。

## 4. 机制数据

| 指标 | C1 | T1 |
|---|---:|---:|
| prefix cache queries | 231,366 | 231,366 |
| prefix cache hits | 156,368 | 154,112 |
| preemptions | 464 | 400 |
| selection calls | 0 | 12,282 |
| candidate probes | 0 | 97,146 |
| candidates with hits | 0 | 59,379 |
| raw reordered selections | 0 | 292 |
| aged selection attempts | 0 | 11,478 |
| successful admissions | 0 | 631 |
| admitted reordered | 0 | 82 |
| admitted aged | 0 | 517 |
| admitted cached tokens | 0 | 437,696 |

T1 的 successful reorder ratio 为 13.0%，证明策略真实改变 admission。值得注意的
是 T1 prefix-cache hits 反而比 C1 少 2,256 tokens，但 preemption 少 64 次、TPOT
明显改善。因此收益不能简化为“APC hit 数更多”；更可能来自请求执行顺序改变后，
decode/prefill 竞争与抢占轨迹更有利。该因果解释仍需要 profiler 或更细的时序计数。

## 5. 决策

- 将 16-prefix 单对结果保留为正向泛化 screening；
- 不改变默认 admission；
- 不继续调 heuristic；
- 下一步优先做 T→C 反向顺序第二对。若第二对仍为正且两对均值 >= +2%，再将
  “跨两个 prefix-diversity workload 均有收益”写入最终简历表述；
- 若反向对不复现，则只保留 8-prefix 两对确认结果。

原始产物：

- `phase6h_benchserve_1250_prefix16_default_c1.json`；
- `phase6h_benchserve_1250_prefix16_cache_affinity_aging10_t1.json`；
- `phase6h_admission_metrics.json`。
