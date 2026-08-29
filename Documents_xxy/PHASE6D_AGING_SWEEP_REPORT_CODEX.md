# Phase 6d 实施报告：Cache-affinity Aging 折中确认

日期：2026-08-29（UTC）

## 1. 结论

Phase 6d 在 Phase 6c 的固定官方 workload 上先进行 treatment-only 筛选，再对
aging=10s 完成两对 fresh-process 反向顺序 A/B。window 始终为 8，唯一变化是
aging threshold。

结果形成清晰的 throughput/fairness 曲线：

| 配置 | throughput vs C mean | TTFT p99 vs C mean | TPOT mean vs C mean |
|---|---:|---:|---:|
| default admission（C 均值） | baseline | baseline | baseline |
| cache-affinity, aging 5s | +0.275% | +6.416% | -5.966% |
| cache-affinity, aging 10s（paired mean） | +3.169% | +22.678% | -5.722% |
| cache-affinity, aging 30s（T 均值） | +4.286% | +58.975% | -9.190% |

5s 基本收回 p99 TTFT，但也几乎消除吞吐收益。10s 的两对 throughput 分别
`+2.436%` 和 `+3.913%`，两次均值 `+3.169%`，通过预设的“每对为正且均值至少
+2%”确认门槛；同时把 30s 下约 59% 的 p99 TTFT 退化缩到约 22.7%。

因此 aging=10s 可以保留为比 30s 更平衡的 opt-in preset 候选。不过它仍不是默认
策略：TTFT mean/p50 均值分别退化 32.0%/100.5%，只是在 p99 和 throughput 之间
形成了更好的折中。

## 2. 精确结果

### 2.1 5s 单轮 screening

5s screening 相对 Phase 6c C1/C2 均值为：throughput `+0.275%`、TTFT p99
`+6.416%`、mean TPOT `-5.966%`。它作为 fairness endpoint 有意义，但没有可见
吞吐收益，因此不进入 paired confirmation。

### 2.2 10s 两对反向 A/B

执行顺序为 T3→C3、C4→T4。每轮都使用全新服务进程、48 warmup 和 192 正式
请求；四轮均为 192/192 成功、185,097 input tokens、196,608 output tokens。

| 指标 | T3 | C3 | C4 | T4 |
|---|---:|---:|---:|---:|
| duration (s) | 180.271 | 184.663 | 187.174 | 180.126 |
| output throughput | 1090.623 | 1064.686 | 1050.402 | 1091.503 |
| TTFT mean (ms) | 5885.943 | 5105.418 | 4394.303 | 6651.941 |
| TTFT p50 (ms) | 3373.414 | 1774.227 | 1877.701 | 3948.783 |
| TTFT p99 (ms) | 22679.488 | 20249.248 | 15885.948 | 21650.291 |
| TPOT mean (ms) | 34.253 | 35.336 | 36.374 | 33.354 |
| TPOT p99 (ms) | 49.006 | 49.342 | 49.441 | 49.632 |
| ITL p99 (ms) | 97.734 | 98.044 | 99.715 | 97.939 |

| 指标 | Pair 1：C3→T3 | Pair 2：C4→T4 | C mean | T mean | 均值变化 |
|---|---:|---:|---:|---:|---:|
| duration | -2.378% | -3.766% | 185.918 s | 180.199 s | -3.077% |
| output throughput | +2.436% | +3.913% | 1057.544 | 1091.063 | +3.169% |
| TTFT mean | +15.288% | +51.376% | 4749.860 ms | 6268.942 ms | +31.982% |
| TTFT p50 | +90.134% | +110.299% | 1825.964 ms | 3661.099 ms | +100.502% |
| TTFT p99 | +12.002% | +36.286% | 18067.598 ms | 22164.889 ms | +22.678% |
| TPOT mean | -3.067% | -8.301% | 35.855 ms | 33.804 ms | -5.722% |
| TPOT p99 | -0.681% | +0.385% | 49.392 ms | 49.319 ms | -0.147% |
| ITL p99 | -0.316% | -1.781% | 98.879 ms | 97.837 ms | -1.055% |

## 3. Aging 激活解释

| 指标 | aging 5s | aging 10s T3/T4 | aging 30s T1/T2 |
|---|---:|---:|---:|
| successful admissions | 956 | 834 / 791 | 898 / 942 |
| admitted reordered | 128 | 133 / 159 | 592 / 651 |
| reordered ratio | 13.4% | 15.9% / 20.1% | 65.9% / 69.1% |
| admitted aged | 774 | 619 / 577 | 137 / 199 |
| aged selection attempts | 9,428 | 8,415 / 8,408 | 2,763 / 3,700 |

10s 两轮共 292 次 successful reordered admission，证明机制实际激活；数量明显
低于 30s 的 1,243 次，所以让更少请求越过基础队头，符合 TTFT p99 退化收敛的
方向。prefix hits 在 C3/C4 都是 201,664，T3/T4 为 200,704/199,808；吞吐收益
依然不是更多 APC hits 导致。

## 4. 最终决策

- aging=10s 通过 paired throughput Gate，保留为 balanced throughput preset 候选；
- aging=30s 保留为更激进的最大吞吐实验模式；
- aging=5s 不形成可见吞吐收益，不继续复验；
- 默认 admission 和默认 aging 配置不改；
- 到此停止 threshold sweep，下一步应测 Scheduler admission CPU/probe cost 或换
  KV budget/模型验证外推性，而不是继续追逐 7s/12s 等 magic threshold。

原始产物：

- `phase6d_benchserve_1250_cache_affinity_aging5.json`；
- `phase6d_benchserve_1250_cache_affinity_aging10.json`；
- `phase6d_benchserve_1250_default_admission_c3.json`；
- `phase6d_benchserve_1250_default_admission_c4.json`；
- `phase6d_benchserve_1250_cache_affinity_aging10_t4.json`；
- `phase6d_aging_metrics.json`。

简历可安全表述为：

> 为 vLLM V1 实现有界 cache-affinity admission 与 aging 公平性保护；在官方
> `vllm bench serve` 的 RTX 4090/Qwen2.5-7B 高 KV 压力两对反向 A/B 中，10s
> balanced preset 的 output throughput 均值提升 3.2%、mean TPOT 降低 5.7%，
> 并将激进 30s 模式的 TTFT p99 退化从 59.0% 收敛到 22.7%。
