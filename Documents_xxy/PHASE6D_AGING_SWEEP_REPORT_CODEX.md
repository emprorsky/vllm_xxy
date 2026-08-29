# Phase 6d 探索报告：Cache-affinity Aging 折中筛选

日期：2026-08-29（UTC）

## 1. 结论

Phase 6d 在 Phase 6c 的固定官方 workload 上进行 treatment-only 单轮筛选：window
保持 8，只把 aging 从 30s 分别改为 5s 和 10s。

结果形成清晰的 throughput/fairness 曲线：

| 配置 | throughput vs C mean | TTFT p99 vs C mean | TPOT mean vs C mean |
|---|---:|---:|---:|
| default admission（C 均值） | baseline | baseline | baseline |
| cache-affinity, aging 5s | +0.275% | +6.416% | -5.966% |
| cache-affinity, aging 10s | +3.648% | +29.060% | -5.297% |
| cache-affinity, aging 30s（T 均值） | +4.286% | +58.975% | -9.190% |

5s 基本收回 p99 TTFT，但也几乎消除吞吐收益；10s 单轮保留约 3.6% 吞吐收益，
并把 30s 下约 59% 的 p99 退化缩到约 29%，是下一轮唯一值得复验的候选。

这两轮是相对 Phase 6c 两次 control 均值的 screening，不是 paired confirmation。
不能把 aging=10s 的 `+3.648%` 作为最终简历数字。Phase 6c 已完成的 30s 双向 A/B
`+4.286%` 才是当前可复现吞吐数据。

## 2. 精确结果

Control mean 来自 Phase 6c C1/C2。

| 指标 | C mean | aging 5s | aging 10s | 30s T mean |
|---|---:|---:|---:|---:|
| duration (s) | 186.850 | 186.336 | 180.271 | 179.251 |
| output throughput | 1052.240 | 1055.128 | 1090.623 | 1097.342 |
| TTFT mean (ms) | 4635.464 | 6389.805 | 5885.943 | 6675.115 |
| TTFT p50 (ms) | 2010.885 | 4235.148 | 3373.414 | 3165.107 |
| TTFT p99 (ms) | 17572.826 | 18700.233 | 22679.488 | 27936.397 |
| TPOT mean (ms) | 36.169 | 34.011 | 34.253 | 32.845 |
| TPOT p99 (ms) | 49.029 | 47.583 | 49.006 | 48.634 |
| ITL p99 (ms) | 98.983 | 101.640 | 97.734 | 101.705 |

## 3. Aging 激活解释

| 指标 | aging 5s | aging 10s | aging 30s T1/T2 |
|---|---:|---:|---:|
| successful admissions | 956 | 834 | 898 / 942 |
| admitted reordered | 128 | 133 | 592 / 651 |
| reordered ratio | 13.4% | 15.9% | 65.9% / 69.1% |
| admitted aged | 774 | 619 | 137 / 199 |
| aged selection attempts | 9,428 | 8,415 | 2,763 / 3,700 |

5s/10s 下大量请求很早进入 aged tier，策略恢复基础队列顺序，所以实际成功重排
骤降。10s 的单轮吞吐仍明显高于 control，但只发生 133 次成功重排，说明该点的
run-to-run 方差可能较大，必须 fresh-process 复验。

## 4. 下一步与停止条件

下一轮只做 aging=10s 的反向 paired confirmation，不继续扩大 sweep：

- 若两对 throughput 都相对 control 为正，且均值至少 +2%，保留 10s 作为
  balanced preset 候选；
- 若方向反转或均值低于 +2%，保留 30s throughput mode 和 5s fairness endpoint，
  不再用 magic threshold 追逐单轮数据；
- 无论结果如何，默认 admission 保持不变。

原始产物：

- `phase6d_benchserve_1250_cache_affinity_aging5.json`；
- `phase6d_benchserve_1250_cache_affinity_aging10.json`；
- `phase6d_aging_metrics.json`。
