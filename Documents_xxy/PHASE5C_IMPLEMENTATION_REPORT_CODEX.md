# Phase 5c 实施报告：Allocation-shortfall-aware KV 抢占

日期：2026-08-29（UTC）

分支：`project/kv-aware-scheduling`

起点：`0f141ae95`（Phase 5b）

## 1. 结论

Phase 5c correctness Gate 已完成，并在当前固定 RTX 4090 高 KV 压力场景通过
performance Gate。

本阶段新增独立实验开关 `preemption_policy=reclaimable_aware`。它只在真实 KV
allocation failure 后工作：读取本次分配缺少的精确 physical-block 数，在最差
user-priority tier 内优先保留“抢占后不能立即推进 allocation retry”的请求，再沿用
Phase 1 的 resume protection、recompute cost 和稳定 tie-break。

两对反向顺序 A/B 的结果为：

- 吞吐两对分别提升 2.886% 和 2.709%，两次均值提升 2.797%；
- wall time 两对分别下降 2.805% 和 2.638%，两次均值下降 2.722%；
- 抢占次数两对分别下降 2.326% 和 12.719%，两次均值下降 8.258%；
- prefix hits 两次均值完全持平；
- TTFT mean 两次均值变化 +0.049%，TTFT p99 +0.474%，基本持平。

因此可确认的结论是：在该固定高压场景中，Phase 5c 提高了输出吞吐并减少了
抢占，没有牺牲 prefix-hit 均值；目前不能宣称它全面改善尾延迟，也不能从单卡、
单模型、单压力形状外推为普适收益。

## 2. 策略设计

### 2.1 精确 allocation shortfall

`KVCacheManager.allocate_slots()` 仅在最终返回 `None` 时，通过只读 observer 报告：

```text
shortfall = required physical blocks - currently available physical blocks
```

该信息是本次真实 allocation retry 的 deficit，不是根据请求 token 长度猜测的容量。
observer 只在 `reclaimable_aware` 开启时传入；默认策略和 `recompute_aware` 不创建
candidate reclaimability scan。

### 2.2 Victim 层级

策略按以下顺序选择 victim：

```text
1. 最差 user-priority tier                    硬约束
2. 最大化 min(immediately_reclaimable, shortfall)
3. 优先保护已经被抢占过的 resumed request
4. 最小化需要重算的 token 数
5. arrival_time / request_id 稳定 tie-break
```

第二层用 `min(reclaimable, shortfall)`，避免把超出当前 deficit 的容量继续当作额外
收益。所有候选都不能立即推进时，排序结果与 `recompute_aware` 完全一致。

### 2.3 Eventual 与 immediate 边界

Phase 5b estimator 表示 official free 后最终回到 BlockPool 的物理页数。Phase 5c
在 Scheduler 中增加立即可用性 gate：

```text
defer_block_free
+ request.last_sched_seq > processed_step_seq
=> immediately_reclaimable = 0
```

因此 GPU write fence 尚未完成时，即使 Phase 5b 的 eventual estimate 大于 0，策略
也不会把它当成本次 allocation retry 可以使用的容量。相关候选随后参与原有
recompute fallback，而不是改变 correctness-critical free 流程。

## 3. 代码改动

| 文件 | 改动 |
|---|---|
| `vllm/config/scheduler.py` | typed config/CLI 增加独立 `reclaimable_aware` 策略及语义说明 |
| `vllm/v1/core/kv_cache_manager.py` | allocation 最终失败时通过 observer 报告精确 physical-block shortfall |
| `vllm/v1/core/sched/policy.py` | 新增 reclaimable-aware 只读决策策略；抽取并复用 priority/recompute fallback |
| `vllm/v1/core/sched/scheduler.py` | 只在实验策略失败路径解析 immediate reclaimability；处理 deferred-free gate 和统计 |
| `vllm/v1/metrics/stats.py` | 新增 `KVPreemptionStats`，按 stats flush 周期传递策略活动 |
| `vllm/v1/metrics/loggers.py` | 暴露 8 个 `vllm:kv_preemption_*` Prometheus counter |
| `Documents_xxy/stress_bench.py` | 抓取、打印并保存 Phase 5c 遥测 |
| `tests/v1/core/test_preemption_policy.py` | priority 硬约束、shortfall progress、recompute fallback、稳定性单测 |
| `tests/v1/core/test_prefix_caching.py` | allocation failure observer 的精确 deficit 与成功路径无回调测试 |
| `tests/v1/core/test_scheduler.py` | immediate/deferred 选择、遥测、真实抢占与策略开关集成测试 |
| `tests/v1/core/test_priority_preemption_bug.py` | 新策略加入既有任意 victim-index regression |
| `tests/v1/core/utils.py` | Scheduler test factory 支持传入 preemption policy |
| `tests/engine/test_arg_utils.py` | CLI 解析回归 |
| `tests/v1/metrics/test_stats.py` | 新统计的 msgpack round-trip 回归 |

没有修改：

- Phase 5a `priority_aware` retention tier；
- Phase 5b refcount estimator 的 eventual 语义；
- Scheduler 对 Request/KV 状态的正式 preempt/free mutation；
- 默认与 `recompute_aware` 的 victim 排序；
- default fast path 的候选 KV block-table 扫描成本。

## 4. 正确性验证

所有命令均直接使用 `/root/miniconda3/envs/vllm-dev/bin/python`，没有运行 uv、
pip 或任何安装命令。

### 4.1 最终回归

| 测试范围 | 结果 |
|---|---:|
| 完整 `tests/v1/core/test_scheduler.py`，排除单卡无法构造的 PP=2 既知用例 | `177 passed, 1 deselected` |
| policy + prefix/refcount + deferred free + priority regression + metrics + CLI | `258 passed` |
| Phase 5c policy 定向测试 | `25 passed` |
| allocation observer/reclaimable prefix 定向测试 | `5 passed` |
| Scheduler reclaimable 定向测试 | `3 passed` |

此前不排除 PP=2 用例的 Scheduler 全量结果为 `177 passed, 1 failed`；唯一失败是
RTX 4090 单卡环境无法构造 world size 2 的
`test_async_scheduling_pp_allows_rescheduling_with_output_placeholders`，与 Phase 5c
逻辑无关。排除该环境不支持用例后为全绿。

### 4.2 静态检查

```text
python -m ruff check <全部本阶段修改的 Python 文件>
All checks passed!

git diff --check
passed
```

## 5. 两对反向顺序 GPU A/B

### 5.1 方法

- 模型：`Qwen/Qwen2.5-7B-Instruct`
- GPU：RTX 4090
- 共同策略：prefix caching、Phase 5a `priority_aware` retention、Phase 3
  `cache_affinity` admission、aging 30s、candidate window 8
- 对照 C：`preemption_policy=recompute_aware`
- 实验 T：`preemption_policy=reclaimable_aware`
- GPU blocks：1250；max model length：8192
- 客户端：192 请求、并发 48、输出长度 `512/1024/1536`、48 个 warmup、
  `temperature=0.7`、`ignore_eos=true`、shape/request seed 固定
- 执行顺序：第一对 C1→T1，第二对 T2→C2
- 每轮使用全新服务进程；轮间确认服务进程退出、GPU 显存释放
- 四轮均为 192/192 成功、0 错误、198,656 completion tokens
- 四轮统一设置 `VLLM_USE_FLASHINFER_SAMPLER=0`

### 5.2 单次结果

| 指标 | C1 | T1 | T2 | C2 |
|---|---:|---:|---:|---:|
| wall time (s) | 197.791 | 192.242 | 190.946 | 196.119 |
| throughput (tok/s) | 1004.373 | 1033.362 | 1040.377 | 1012.935 |
| TTFT mean (s) | 67.953 | 68.450 | 67.903 | 68.333 |
| TTFT p50 (s) | 59.051 | 59.264 | 58.995 | 59.383 |
| TTFT p90 (s) | 121.207 | 121.915 | 124.139 | 139.872 |
| TTFT p99 (s) | 152.519 | 154.131 | 162.856 | 162.972 |
| chunk-ITL mean (ms) | 34.558 | 32.279 | 32.138 | 32.468 |
| preemptions | 473 | 462 | 549 | 629 |
| prefix hits | 129,392 | 128,320 | 128,320 | 127,248 |

### 5.3 逐对方向与两次均值

| 指标 | Pair 1：C1→T1 | Pair 2：C2→T2 | C 两次均值 | T 两次均值 | 均值变化 |
|---|---:|---:|---:|---:|---:|
| wall time | -2.805% | -2.638% | 196.955 s | 191.594 s | -2.722% |
| throughput | +2.886% | +2.709% | 1008.654 | 1036.869 | +2.797% |
| TTFT mean | +0.730% | -0.629% | 68.143 s | 68.176 s | +0.049% |
| TTFT p50 | +0.362% | -0.654% | 59.217 s | 59.130 s | -0.147% |
| TTFT p90 | +0.584% | -11.248% | 130.539 s | 123.027 s | -5.755% |
| TTFT p99 | +1.057% | -0.071% | 157.745 s | 158.493 s | +0.474% |
| chunk-ITL mean | -6.597% | -1.018% | 33.513 ms | 32.208 ms | -3.894% |
| preemptions | -2.326% | -12.719% | 551.0 | 505.5 | -8.258% |
| prefix hits | -0.828% | +0.842% | 128,320 | 128,320 | 0.000% |

这里的分位数均值是两次 run-level percentile 的简单均值，不是把两轮请求样本
重新合并后的总体 percentile。自定义 harness 的 chunk-ITL 是诊断指标，不能当作
标准 `vllm bench serve` global ITL。

### 5.4 Phase 5c 激活遥测

| 指标 | T1 | T2 | 两次均值 |
|---|---:|---:|---:|
| shortfall events | 462 | 549 | 505.5 |
| shortfall blocks | 462 | 549 | 505.5 |
| candidate estimates | 12,823 | 15,199 | 14,011 |
| deferred candidates | 0 | 0 | 0 |
| reclaimable blocks across candidates | 435,039 | 517,311 | 476,175 |
| selected reclaimable blocks | 11,186 | 11,066 | 11,126 |
| sufficient selections | 462 | 549 | 505.5 |
| zero-progress selections | 0 | 0 | 0 |

每次 shortfall 都恰好是 1 block，且选中的 victim 都能提供立即容量。由于第二层
使用 `min(reclaimable, shortfall)`，该场景实际验证的是“排除 0-progress victim，
然后按原 recompute 规则选择”，而不是大 deficit 下对多个 reclaimable 数值的
排序效果。这个边界必须保留在性能结论中。

原始产物：

- `stress_bench_phase5c_recompute.json`
- `stress_bench_phase5c_reclaimable.json`
- `stress_bench_phase5c_reclaimable_repeat.json`
- `stress_bench_phase5c_recompute_repeat.json`

## 6. 验收与下一步

Phase 5c 验收分为两层：

- correctness：通过。精确 shortfall、priority 硬约束、immediate/eventual 边界、
  recompute fallback、真实抢占 mutation、CLI 和 metrics 均有回归保护；
- performance：通过当前固定压力场景 Gate。两对 reverse-order A/B 的 throughput、
  wall time 和 preemption 方向一致，prefix-hit 均值持平。

下一步不建议立刻再叠加新的调度 heuristic。优先顺序应是：

1. 进入 Phase 6，测量 default/recompute/reclaimable 的 Scheduler CPU decision cost，
   量化每次 shortfall 平均约 28 次 candidate estimator 的代价；
2. 增加一个能产生多 block deficit 或更高 shared-KV 比例的独立 workload，验证
   ranking 的适用边界；
3. 用 `vllm bench serve` 补一套标准 serving 指标，并在另一模型或另一 KV budget
   上复现方向；
4. 若目标是简历展示，可准确表述为“实现 refcount-aware、deferred-free-safe 的
   allocation-shortfall victim policy；RTX 4090/Qwen2.5-7B 固定高压 reverse-order
   A/B 中吞吐 +2.8%、抢占 -8.3%，435 项 CPU 回归通过”，不要写成普适提升。
