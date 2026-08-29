# Phase 6b 实施报告：抢占策略反事实因果诊断

日期：2026-08-29（UTC）

分支：`project/kv-aware-scheduling`

起点：`b6c775ba6`（Phase 6）

## 1. 结论

Phase 6b 完成了 Phase 5c 的因果归因检查，结论是否定但很明确：

> 在本项目已使用的三个高压 serving workload 中，`reclaimable_aware` 一次也没有
> 改变 `recompute_aware` 本来会选择的 victim。此前测得的 `+2.8%` 自定义压测和
> `+2.1%` 官方压测吞吐差异是真实的 run-level 测量值，但不能归因于 Phase 5c
> 算法，只能按运行波动处理。

三个 workload 共发生 1,922 次 shortfall/preemption decision：

| workload | KV blocks | shortfall events | changed victims | change rate |
|---|---:|---:|---:|---:|
| 官方 `vllm bench serve` | 1250 | 779 | 0 | 0.000% |
| 官方 `vllm bench serve` | 1000 | 655 | 0 | 0.000% |
| 自定义 `stress_bench.py` | 1250 | 488 | 0 | 0.000% |
| **合计** | - | **1,922** | **0** | **0.000%** |

这不是实现错误。定向单测证明策略在有区分度的数据上能改选 victim，并能准确记录
增加或避免的重算成本。问题在于当前 decode-heavy workload 的每次 allocation
shortfall 都恰好为 1 block，而原 `recompute_aware` victim 每次已经至少能立即释放
1 block。于是所有可行候选的：

```text
min(immediately_reclaimable, shortfall) = 1
```

第二层排序键完全相同，策略必然退化回原 recompute 排序。

因此 Phase 5c 的正确结论应改为：correctness 和 observability Gate 通过；当前
workload activation/performance Gate 未通过。它是经过验证的实验框架，不是已验证
有性能收益的优化。

## 2. 新增反事实观测

Scheduler 在同一次真实 allocation failure 中只读计算：

1. `recompute_aware` 将选择的 baseline victim；
2. `reclaimable_aware` 实际选择的 victim；
3. 两者的 immediate reclaimable blocks 和 computed-token cost；
4. 是否改选、增加/避免的重算 token。

baseline 与实际策略共享同一个候选 refcount memo，不重复扫描相同 request。它只
增加观测，不改变 production victim selection、free 顺序或请求状态。

新增 Prometheus counters：

- `changed_selections`；
- `baseline_reclaimable_blocks` 与 `reclaimability_gain_blocks`；
- `selected_computed_tokens` 与 `baseline_computed_tokens`；
- `additional_recompute_tokens` 与 `avoided_recompute_tokens`。

## 3. 因果结果

### 3.1 官方 benchmark

| 指标 | 1250 blocks | 1000 blocks |
|---|---:|---:|
| shortfall events / blocks | 779 / 779 | 655 / 655 |
| candidate estimates | 21,512 | 14,635 |
| sufficient / zero-progress | 779 / 0 | 655 / 0 |
| selected reclaimable blocks | 13,788 | 15,084 |
| baseline reclaimable blocks | 13,788 | 15,084 |
| changed selections | 0 | 0 |
| reclaimability gain | 0 | 0 |
| selected computed tokens | 894,569 | 768,880 |
| baseline computed tokens | 894,569 | 768,880 |
| additional / avoided recompute | 0 / 0 | 0 / 0 |

新增 treatment run 的吞吐为：

| KV blocks | Phase 6 control | Phase 6 treatment | Phase 6b treatment repeat |
|---:|---:|---:|---:|
| 1250 | 1068.475 | 1090.813 | 1076.246 tok/s |
| 1000 | 874.952 | 869.739 | 852.768 tok/s |

同一策略路径的 treatment repeat 自身已有明显波动。结合 0 次改选，Phase 6 的
control/treatment 差值不具备算法因果含义。

### 3.2 自定义 benchmark

自定义 1250-block treatment repeat：

- 192/192 成功，198,656 output tokens；
- 吞吐 1035.9 tok/s，旧两次 treatment 均值为 1036.869 tok/s；
- 488 次 shortfall 全部为 1 block；
- 13,336 次 candidate estimate；
- `changed_selections=0`，两种选择的 reclaimable/computed-token totals 完全一致。

该 repeat 同时复现了旧 treatment 的吞吐水平和“0 次策略改选”。所以旧 `+2.797%`
数字不能继续作为 Phase 5c 性能收益使用。

## 4. CPU 成本复查

加入反事实 telemetry 后重跑相同 microbenchmark：

| candidates | Phase 6 reclaimable (us) | Phase 6b reclaimable (us) |
|---:|---:|---:|
| 1 | 17.873 | 19.971 |
| 8 | 120.264 | 129.391 |
| 16 | 246.947 | 244.377 |
| 64 | 983.002 | 972.340 |
| 256 | 3807.596 | 3997.635 |

同量级波动，没有出现第二次 refcount 全扫描。现阶段真正的问题不是 Scheduler CPU
成本，而是策略键在目标 workload 上没有区分度。

## 5. 验证

所有 Python 命令均使用 `/root/miniconda3/envs/vllm-dev/bin/python`，未安装依赖。

| 验证范围 | 结果 |
|---|---:|
| 完整 Scheduler（排除单卡不支持的 PP=2 用例） | `178 passed, 1 deselected` |
| policy/prefix/deferred/priority/metrics/CLI 相关回归 | `258 passed` |
| reclaimable 反事实定向测试 | `4 passed` |
| metrics stats | `12 passed` |
| Ruff | passed |
| `git diff --check` | passed |

定向测试覆盖两种真正改选的反例：改选造成额外 990 个 computed tokens，以及改选
避免 990 个 computed tokens。这证明 0 次改选来自 workload，而不是计数器失效。

## 6. 下一步决策

1. 保留 `reclaimable_aware` 实验开关和只读 telemetry，但不默认启用、不宣称收益；
2. 不继续为当前 one-block shortfall 调权重，因为 capacity key 在数学上已退化；
3. 只有找到真实 multi-block shortfall/speculative-decode workload 后，才重新评估
   Phase 5c；
4. 短期性能路线转向已经能实际改变行为的 admission/retention/recompute policy，
   要求每个吞吐结果同时附带 activation/counterfactual 证据。

原始产物：

- `phase6b_counterfactual_metrics.json`；
- `phase6b_benchserve_1250_reclaimable.json`；
- `phase6b_benchserve_1000_reclaimable.json`；
- `phase6b_stress_reclaimable.json`；
- `phase6b_scheduler_cpu.json`。

简历当前可安全表述为：

> 为 vLLM V1 实现 refcount-aware KV 抢占原型与反事实可观测性，在 1,922 次真实
> 高压抢占决策中识别出 one-block shortfall 导致的策略退化，排除约 2% run-level
> 波动的错误因果归因，并用 178 项 Scheduler 回归和标准 serving benchmark 验证。
