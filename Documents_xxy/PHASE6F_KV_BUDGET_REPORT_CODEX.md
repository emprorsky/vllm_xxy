# Phase 6f 探索报告：Cache-affinity 的 KV Budget 外推性

日期：2026-08-29（UTC）

## 1. 结论

Phase 6f 将 Phase 6d 的 balanced preset 原样从 1250 blocks 移到更紧的 1000
blocks，只改变 admission policy。一次 fresh-process C→T screening 中，aging=10s、
window=8 的 cache-affinity 未复现 1250-block 的收益：

| 配置 | output throughput | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean |
|---|---:|---:|---:|---:|---:|
| default admission C1 | 840.643 tok/s | 12168.743 ms | 11702.468 ms | 35297.110 ms | 39.329 ms |
| cache-affinity T1 | 835.602 tok/s | 12980.668 ms | 13927.249 ms | 36092.778 ms | 38.030 ms |
| T vs C | **-0.600%** | +6.672% | +19.011% | +2.254% | -3.303% |

这组结果没有通过跨 KV budget Gate。它不能证明算法在 1000 blocks 下稳定负收益，
因为这里只做了一对 screening；但已经足以否定“1250-block 的约 3.2% 收益可以
直接外推到更高压力”的假设，因此不投入第二对复验。

当前可复现结论必须限定为：RTX 4090、Qwen2.5-7B、固定 prefix-repetition
workload、1250 KV blocks。不能写成普遍提升。

## 2. 实验边界

两轮共同使用：

- prefix caching 开启；
- `preemption_policy=recompute_aware`；
- `prefix_cache_eviction_policy=lru`，排除 retention 混杂；
- 1000 GPU KV blocks；
- 8 个 prefix、prefix 900 tokens、suffix 64 tokens、output 1024 tokens；
- 48 warmup + 192 正式请求、max concurrency 48；
- `temperature=0.7`、`ignore_eos=true`、seed 42；
- 每轮新服务进程，192/192 请求成功，185,097 input tokens、196,608 output
  tokens。

唯一实验变量：

- C1：`admission_policy=default`；
- T1：`admission_policy=cache_affinity`、window=8、aging=10s。

## 3. 完整结果

| 指标 | C1 | T1 | T vs C |
|---|---:|---:|---:|
| duration (s) | 233.878 | 235.289 | +0.603% |
| output throughput (tok/s) | 840.643 | 835.602 | -0.600% |
| TTFT mean (ms) | 12168.743 | 12980.668 | +6.672% |
| TTFT p50 (ms) | 11702.468 | 13927.249 | +19.011% |
| TTFT p99 (ms) | 35297.110 | 36092.778 | +2.254% |
| TPOT mean (ms) | 39.329 | 38.030 | -3.303% |
| TPOT p99 (ms) | 60.495 | 60.828 | +0.551% |
| ITL p99 (ms) | 95.881 | 96.949 | +1.114% |

## 4. 机制解释

1000-block T1 共发生 13,699 次 candidate selection，但只有 862 次 admission
成功，其中 102 次真正越过基础队头，successful reorder ratio 为 11.8%。12,830
次 selection（93.7%）因 aged request 回退到基础顺序。

相比 1250-block aging=10s 的两轮：

- selection call 从约 9.3k 增到 13.7k，说明 KV 更紧时调度反复尝试更多；
- successful reorder ratio 从 15.9%/20.1% 降到 11.8%；
- 策略大部分时间被 aging 保护钳制，实际 locality 重排空间变小。

按 Phase 6e 的 W=8 冷路径 `78.466 us/decision` 上界估算，T1 admission CPU
约 1.075s，占 235.289s wall time 的 0.457%。因此 CPU probe 成本可能贡献噪声，
但不足以单独解释算法为何没有吞吐收益；更合理的解释是紧 KV budget 下可利用的
重排机会减少，而队头等待和调度重试成本上升。

## 5. 决策

- 1000-block 外推 Gate 未通过，停止该点的 paired confirmation；
- 不将 1250-block 性能数字概括为跨 budget 收益；
- 不继续扫描 aging magic threshold；
- 下一步只筛选一个结构变量：在 1250 blocks 固定 aging=10s，将 window 从 8
  缩至 4，检验更小重排范围能否降低 TTFT 代价；
- 若 W=4 不能同时保留可见吞吐收益并改善公平性，停止 admission heuristic
  调参，转向 workload/模型泛化或 profiler。

原始产物：

- `phase6f_benchserve_1000_default_admission_c1.json`；
- `phase6f_benchserve_1000_cache_affinity_aging10_t1.json`；
- `phase6f_admission_metrics.json`。
