# Phase 6e 实施报告：Cache-affinity Scheduler CPU 成本

日期：2026-08-29（UTC）

## 1. 结论

Phase 6e 完成了 cache-affinity admission 的真实 Scheduler CPU microbenchmark。
在 serving 使用的 window 8 下：

- default admission fast path：`0.172 us/decision`；
- 同 KV generation 热缓存：`30.433 us/decision`；
- KV mutation 后冷解析：`78.466 us/decision`；
- 相对 default 的冷路径增量：`78.294 us/decision`。

Phase 6d aging=10s 的一轮 serving 有 9,322 次 selection call、73,244 次 candidate
probe。按 W=8 冷路径估算，累计 admission decision CPU 时间约：

```text
9,322 * 78.466 us = 0.731 s
0.731 / 180.271 = 0.406% serving wall time
```

因此 probe 放大是可观测的控制面低效，但不是当前约 3% throughput 改善或 TTFT
变化的主因。现阶段不值得为了亚百分之一 wall time 进入复杂跨代缓存或减少正确性
失效点；下一步应验证性能能否跨 KV budget 外推。

## 2. 方法

新增 `Documents_xxy/benchmark_phase6e_admission_cpu.py`，使用真实：

- `Scheduler` 与 FCFS waiting queue；
- vLLM `Request` 和 block hash；
- prefix cache seed、event-free local-prefix resolver；
- `_select_cache_affinity_request()`；
- `SchedulingFeatureContext` 的热缓存和 KV generation 失效。

每个候选 prompt 为 3 个共享 prefix blocks + 1 个独立 suffix block。为了确保机制
激活，基础队头是 cold request，其余请求有真实本地 prefix hit；benchmark 验证
cache-affinity 实际绕过队头。

配置：

- CPU 固定到 core 0，`OMP_NUM_THREADS=1`；
- candidates/window：2、4、8、16；
- 每个点自适应迭代到约 100ms，9 次重复，报告 run median；
- 正式计时批次关闭 GC；
- default/hot/cold 使用相同规模的真实 Request queue。

## 3. 结果

| candidates | default (us) | affinity hot (us) | affinity cold (us) | cold - default (us) |
|---:|---:|---:|---:|---:|
| 2 | 0.178 | 11.196 | 21.539 | 21.360 |
| 4 | 0.173 | 17.356 | 40.481 | 40.308 |
| 8 | 0.172 | 30.433 | 78.466 | 78.294 |
| 16 | 0.175 | 56.517 | 156.213 | 156.038 |

成本随候选数近似线性。冷路径比热路径多出的部分来自真实 prefix resolution；热
路径本身仍需构造候选特征、执行策略排序和更新统计。

default fast path 保持约 0.17us，证明实验开关关闭时不会进入候选窗口和 prefix
probe。

## 4. 决策

- 保留当前 window=8 的有界范围；
- 不减少 Phase 4 的 KV mutation 失效点；
- 不做跨 generation 的 stale prefix cache；
- 暂不优化临时 candidate list/dict，因为最大可回收量级不足 1% wall time；
- 下一步在 1000 blocks 下隔离复验 aging=10s balanced preset。

原始产物：

- `benchmark_phase6e_admission_cpu.py`；
- `phase6e_admission_cpu.json`。
