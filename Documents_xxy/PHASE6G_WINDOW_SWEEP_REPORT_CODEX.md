# Phase 6g 探索报告：Cache-affinity Window=4 筛选

日期：2026-08-29（UTC）

## 1. 结论

Phase 6g 在 1250 blocks、aging=10s 下将 candidate window 从 8 缩至 4。单轮
treatment-only screening 相对 Phase 6d 两轮 fresh control 均值为：

- output throughput `-0.955%`；
- TTFT mean/p50 `+4.179%/+43.459%`；
- TTFT p99 `-8.874%`；
- mean TPOT 基本持平（`-0.001%`）。

W=4 确实将 W=8 的 TTFT 代价明显收回，p99 甚至优于 control，但同时消除了可见
吞吐收益。它没有通过“保留至少 +2% throughput，同时改善 TTFT”的筛选门槛，
因此不进入 paired confirmation。

结合 Phase 6d：W=8/aging=10s 是已确认的 opt-in throughput/fairness 折中；W=4
是更接近 default 的公平性端点，不是更好的性能 preset。到此停止 window/aging
heuristic sweep。

## 2. 实验配置

与 Phase 6d 共同配置一致：RTX 4090、Qwen2.5-7B、1250 KV blocks、prefix
caching、recompute-aware preemption、LRU retention、8 个 prefix、900/64/1024
tokens、48 warmup + 192 正式请求、max concurrency 48、temperature 0.7、
ignore-eos、seed 42。

本轮 treatment：`admission_policy=cache_affinity`、window=4、aging=10s。正式请求
192/192 成功，185,097 input tokens、196,608 output tokens。

## 3. 结果

| 指标 | Phase 6d C mean | W=4 screening | W=4 vs C |
|---|---:|---:|---:|
| duration (s) | 185.918 | 187.702 | +0.959% |
| output throughput (tok/s) | 1057.544 | 1047.448 | -0.955% |
| TTFT mean (ms) | 4749.861 | 4948.374 | +4.179% |
| TTFT p50 (ms) | 1825.964 | 2619.514 | +43.459% |
| TTFT p99 (ms) | 18067.598 | 16464.349 | -8.874% |
| TPOT mean (ms) | 35.855 | 35.855 | -0.001% |
| TPOT p99 (ms) | 49.392 | 48.621 | -1.560% |
| ITL p99 (ms) | 98.880 | 102.428 | +3.589% |

相对已确认的 W=8 treatment 均值，W=4 throughput 下降 `3.998%`，TTFT
mean/p50/p99 分别改善 `21.1%/28.5%/25.7%`。这说明 window 是真实的
throughput/fairness 控制杆，而不是 Scheduler CPU 优化杆。

## 4. 激活数据

| 指标 | W=4 |
|---|---:|
| prefix cache queries / hits | 231,375 / 201,664 |
| preemptions | 737 |
| selection calls | 9,762 |
| candidate probes | 38,828 |
| candidates with hits | 36,517 |
| raw reordered selections | 481 |
| aged selection attempts | 8,630 |
| successful admissions | 950 |
| admitted reordered | 206 |
| admitted aged | 645 |
| admitted cached tokens | 849,824 |

successful reorder ratio 为 21.7%，高于 W=8 两轮的 15.9%/20.1%，但每次只能在
更小窗口中选候选，累计 locality 排序空间仍较小。88.4% selection attempts 触发
aged tier。机制确实激活，但激活次数本身没有转化为吞吐收益。

## 5. 决策

- W=4 不保留为性能 preset，不做第二对；
- W=8/aging=10s 继续作为唯一 balanced opt-in preset 候选；
- 默认 admission 仍保持不变；
- 停止 aging/window 参数扫描，避免为了单 workload 过拟合；
- 下一步若继续追求可见收益，应换模型或 workload 验证泛化，或用 profiler 定位
  prefill/decode 调度收益来源，而不是叠加更多排序项。

原始产物：

- `phase6g_benchserve_1250_cache_affinity_w4_aging10.json`；
- `phase6g_admission_metrics.json`。
