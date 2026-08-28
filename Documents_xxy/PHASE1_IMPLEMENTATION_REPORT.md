# Phase 1 实施与验收记录：可恢复的抢占策略

> 状态：功能验收完成；性能数据已留档，但尚不能据此宣称稳定收益。
>
> 日期：2026-08-28 (UTC)
> 
> 范围：`recompute_aware` 抢占策略的语义收紧，以及 Priority 调度下
> 的 resume-aware 重入队。Phase 2 的 KV 保留、准入控制和 aging 尚未开始。

## 1. 本阶段要解决什么

已有的策略抽象已经把“选谁抢占”和 Scheduler 中的状态变更分开。Phase 1
在这个边界上补齐两个语义：

1. 同一 user-priority 层内，已经被抢占过的请求不应因其
   `num_preemptions` 较小/较大而反复成为 victim；它们先获得二元的
   resume protection。
2. 已抢占请求重新进入等待队列时，在不越过更高 user priority 的前提下，
   应先于同层的新请求恢复，避免刚释放的请求被同层新工作持续挤压。

默认策略保持原始 vLLM 行为，所有新行为仅由
`preemption_policy="recompute_aware"` 启用。

## 2. 最终调度规则

| 场景 | `default` | `recompute_aware` |
|---|---|---|
| 选择抢占 victim（FCFS） | running 列表最后一个请求 | 同 `default` |
| 选择抢占 victim（Priority） | 最低 user priority 层中的最后一个请求 | 只在最低 user-priority 层选择；先选从未被抢占的请求，再选计算量 `num_computed_tokens` 最小者，最后以较晚 arrival 和 request ID 稳定打破平局 |
| Priority 等待队列 | `(priority, arrival_time, request_id)` | `(priority, 是否为新请求, arrival_time, request_id)`；已抢占请求在同层优先 |
| 跨 `waiting` / `skipped_waiting` 队列取下一项 | 当前 Priority head | 比较每项**入队时冻结**的 key，避免 `num_preemptions` 后续变化破坏 heap 顺序 |

其中 priority 是硬约束：数值更小的 user priority 始终优先，Phase 1 不允许
“恢复请求”跨越更高优先级的新请求。

## 3. 代码改动

| 文件 | 改动 |
|---|---|
| `vllm/v1/core/sched/policy.py` | 扩展只读 `SchedulingDecisionPolicy`：同时提供 victim 选择和 waiting order key；recompute-aware 使用二元 resume protection，并加入 request ID 稳定 tie-break。 |
| `vllm/v1/core/sched/request_queue.py` | Priority heap 改为保存 `(frozen key, request)`；新增 `peek_order_key()`。这避免请求状态变化后直接影响已在 heap 中的比较键。 |
| `vllm/v1/core/sched/scheduler.py` | Scheduler 通过 policy 创建所有 Priority waiting 队列；在 normal/skipped waiting 队列之间按冻结 key 选取。 |
| `vllm/config/scheduler.py` | 补充 `preemption_policy` 的二元保护与同层恢复优先语义。 |
| `tests/v1/core/test_preemption_policy.py` | 覆盖二元保护、稳定平局、FCFS / Priority 恢复顺序、严格 priority 和 heap key 冻结。 |
| `tests/v1/core/test_scheduler.py` | 覆盖真实 Scheduler 下 default/优化策略、严格 priority，以及 waiting/skipped 队列合并时的冻结 key。 |
| `tests/v1/core/test_priority_preemption_bug.py` | 对 default 和 recompute-aware 同时验证 Priority 抢占的状态回滚：victim 从 running 移除、计算 token 清零、抢占计数加一且仅重入 waiting 一次。 |

关键边界没有改变：释放 KV block、回滚 token budget、更新请求状态、放回等待队列
仍由 `Scheduler` 完成；policy 和 queue 只作只读决策或排序。

## 4. 验证结果

使用环境：`/root/miniconda3/envs/vllm-dev/bin/python`，torch
`2.13.0+cu130`，editable vLLM。

| 验证项 | 结果 |
|---|---|
| `test_preemption_policy.py` + `test_priority_preemption_bug.py` | **18 passed**，4.72 s（14 条既有 torch deprecation warnings） |
| Scheduler 定向回归：`test_scheduler.py -k 'preempt or priority or resume or waiting_queue'` | **40 passed, 114 deselected**，57.34 s |
| Priority 随机调度回归 | **12 passed**，268.32 s |
| worker slot overflow 回归 | **3 passed** |
| deferred block free 回归 | **12 passed** |
| `ruff check`（7 个改动代码/测试文件） | 通过 |
| `ruff format --check`（7 个改动代码/测试文件） | 通过，7 files already formatted |
| `compileall`（scheduler 与 config） | 通过 |
| `git diff --check` | 通过 |
| 服务关闭检查 | 无 vLLM/API/EngineCore 进程，GPU 无 compute app |

## 5. 4090 压测：同配置单次对照

模型为本地缓存的 `Qwen/Qwen2.5-7B-Instruct`，GPU 为 RTX 4090。服务使用
`max-model-len=8192`、`gpu-memory-utilization=0.75`、APC 开启；压测固定为
48 并发、192 请求、8 个 prefix pool、70% 共享前缀、输出上限
`[512, 1024, 1536]`、seed 42。两次均成功完成 192 个请求、错误数为 0。

| 指标 | `default` | `recompute_aware` | 相对变化 |
|---|---:|---:|---:|
| 墙钟时间 | 32.121 s | 30.151 s | -6.13% |
| 请求吞吐 | 5.977 req/s | 6.368 req/s | +6.53% |
| content-chunk 吞吐 | 1679.79 chunk/s | 1714.47 chunk/s | +2.06% |
| TTFT mean | 9.697 s | 9.584 s | -1.16% |
| TTFT p99 | 22.015 s | 22.149 s | +0.61% |
| 请求级 mean ITL 的均值 | 23.11 ms | 22.75 ms | -1.57% |
| 请求级 p99 ITL 的跨请求 p99 | 3.359 s | 0.989 s | -70.56% |
| scheduler preemptions | 74 | 79 | +6.76% |
| prefix cache hit/query | 141152 / 154312 | 141136 / 154312 | 基本持平 |

原始结果可复查：
[default JSON](stress_bench_phase1_default.json) 与
[recompute-aware JSON](stress_bench_phase1_recompute_aware.json)。

### 如何解读这张表

这是一对固定输入的**单次诊断运行**，不是性能结论。它确认两种策略都能在
高 KV pressure 下完整执行，也观察到本次尾部 ITL、墙钟时间和吞吐的正向信号；
但优化策略的抢占次数是 79，高于 baseline 的 74。因此不能把这组数据描述为
“减少抢占”或“稳定提升”。

此外，现有 `stress_bench.py` 的 `n_tokens` 统计的是 SSE content chunk，
不是 tokenizer token；表中“请求级 p99 ITL 的跨请求 p99”也不是全局 token ITL
p99。后续性能 Gate 应先完善指标，再采用交替顺序的多轮 paired run，并以
`vllm bench serve` 结果作为可对外比较的主指标。

## 6. Phase 1 结论与下一步

Phase 1 的 correctness 目标已经完成：默认路径兼容、优化路径的二元反复抢占
保护、Priority 同层恢复优先、严格 user-priority 以及 heap 不变量均有测试覆盖。
本阶段的性能数据作为后续对照基线留存，性能收益仍待重复实验确认。

下一阶段按统一实施计划进入 Phase 2：只为 waiting 请求按需保留可复用的 KV
前缀，并先建立 block accounting、cancel/finish 清理和 pressure A/B 验收。
