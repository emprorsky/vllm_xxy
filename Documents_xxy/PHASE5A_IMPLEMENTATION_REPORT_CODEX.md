# Phase 5a 实施报告：Priority/Demand-aware Prefix Retention

日期：2026-08-28（UTC）

分支：`project/kv-aware-scheduling`

起点：`c045abd3b`（Phase 4）

## 1. 结论

Phase 5a correctness Gate 已完成。实现没有覆盖原有 Phase 2 行为，而是保留
`waiting_queue_aware` 作为二值对照，新增显式实验开关
`prefix_cache_eviction_policy=priority_aware`。

新策略把原来的 `set[block_id]` 演进为：

```text
tier 0: 无近期 waiting demand
tier 1: 普通 near-head demand
tier 2: resumed near-head demand
tier 3: 相对高 user-priority 的 near-head demand
```

BlockPool 始终先淘汰低 tier，tier 内保持原始 LRU；空间不足时依次回退到所有
tier，因此不会改变 allocation feasibility。默认 `lru` 路径不创建 hint、
不解析 waiting demand，行为不变。

两对反向执行顺序的固定工作量 A/B 中，tier 策略的平均 wall time 下降
1.954%、吞吐提高 1.970%。但是 prefix hits 和 preemption 的逐对方向相反，
TTFT p50 的两次均值还退化 10.417%。所以本阶段只验收 correctness；性能信号
偏正但混合，不能宣称稳定 KV 效率收益。

## 2. Tier 语义

### 2.1 User priority

只有配置为 `scheduling_policy=priority` 时，Request priority 才影响 retention
tier。最佳 priority 必须严格优于窗口内另一个有效候选，才被提升到 tier 3：

```text
priority scheduling
+ near-head 候选存在至少两个不同 priority
+ candidate.priority == min(window priorities)
=> HIGH_PRIORITY
```

这样没有硬编码“priority 0 就是高优先级”，也不会把所有请求都使用默认
priority=0 的窗口整体误判为高优先级。FCFS 下 Request 上即使带 priority 值，
retention 也不会使用一个 Scheduler 本身不执行的优先级语义。

### 2.2 Resume 和重叠 demand

非高优先级候选中，`num_preemptions > 0` 为 tier 2，普通请求为 tier 1。
同一物理 block 可能被多个 waiting request 的 prefix 命中，最终使用所有需求的
最大 tier：

```text
block_tier[id] = max(existing_tier, candidate_tier)
```

因此普通请求与 resumed 请求共享 prefix 时，该 block 会提升到 resumed tier。

## 3. 淘汰算法

`FreeKVCacheBlockQueue` 的选择顺序为：

```text
unhashed / unmentioned (tier 0), original LRU
normal demand          (tier 1), original LRU
resumed demand         (tier 2), original LRU
high-priority demand   (tier 3), original LRU
```

实现仍然原地 unlink 选中的 free block；未选 block 的全局相对顺序和双向链表
保持不变。如果 tier 0 足以满足请求，扫描找到足够 block 后即可停止；只有需要
fallback 时才扫描到队尾并按 tier bucket 补足。

二值 `waiting_queue_aware` 也使用同一底层接口，但把所有 demand block 标为
tier 1，因此选择顺序与 Phase 2 完全一致。

## 4. 代码改动

| 文件 | 改动 |
|---|---|
| `vllm/config/scheduler.py` | 新增 `priority_aware` typed config/CLI 值及语义说明 |
| `vllm/v1/core/kv_cache_utils.py` | 新增 `BlockRetentionTier`；hint 改为 lazy `block_id -> tier` mapping；free queue 实现 tiered stable-LRU 选择 |
| `vllm/v1/core/block_pool.py` | 接入 tier 选择，同时保留正式 eviction accounting 和 soft fallback 统计 |
| `vllm/v1/core/kv_cache_manager.py` | allocation-scoped resolver 类型升级为 tier mapping |
| `vllm/v1/core/sched/scheduler.py` | bounded waiting candidate 生成 normal/resumed/high tier，重叠 block 取最大值 |
| `vllm/v1/metrics/stats.py` | retention 增加 normal/resumed/high-priority block 统计 |
| `vllm/v1/metrics/loggers.py` | 暴露 3 个新 Prometheus counter |
| `Documents_xxy/stress_bench.py` | 抓取和保存 tier 激活数据 |
| `tests/v1/core/test_prefix_caching.py` | tier 顺序、tier 内 LRU、未选顺序、全 tier fallback 和正式 eviction 回归 |
| `tests/v1/core/test_scheduler.py` | 相对 priority、FCFS priority 忽略、resume、重叠 demand promotion 测试 |
| `tests/engine/test_arg_utils.py` | `priority_aware` CLI 解析回归 |
| `tests/v1/metrics/test_stats.py` | tier stats msgpack 序列化回归 |

新增指标：

- `vllm:kv_retention_normal_blocks_total`
- `vllm:kv_retention_resumed_blocks_total`
- `vllm:kv_retention_high_priority_blocks_total`

## 5. 正确性验证

所有命令直接使用 `/root/miniconda3/envs/vllm-dev/bin/python`，没有运行 uv、
pip 或安装命令。

| 检查 | 结果 |
|---|---|
| scheduler + policy + CLI + metrics 组合回归 | `299 passed, 1 deselected` |
| 完整 `test_prefix_caching.py` | `103 passed` |
| Phase 5a prefix 定向测试 | `8 passed` |
| Phase 5a Scheduler 定向测试 | `4 passed` |
| CLI/metrics 定向测试 | `2 passed` |
| ruff（全部本阶段 Python 改动） | passed |
| compileall（生产代码与压测脚本） | passed |
| `git diff --check` | passed |

组合回归排除的仍是单卡机器无法构造 PP=2 配置的既知测试
`test_async_scheduling_pp_allows_rescheduling_with_output_placeholders`，它没有
进入 Phase 5a 路径。

## 6. 两对反向顺序 GPU A/B

### 6.1 方法

- 模型：`Qwen/Qwen2.5-7B-Instruct`
- GPU：RTX 4090
- 共同策略：prefix caching、`recompute_aware`、cache-affinity admission、
  aging 30s、candidate window 8
- GPU blocks：1250；max model length：8192
- 客户端：192 请求、并发 48、输出长度 `512/1024/1536`、48 个 warmup、
  `temperature=0.7`、`ignore_eos=true`、每请求固定 seed
- B：`waiting_queue_aware` 二值 retention
- T：`priority_aware` tier retention
- 执行顺序：第一对 B1→T1，第二对 T2→B2
- 四组均为 192/192 成功、0 错误、198,656 completion tokens，全部按长度结束
- 四组统一设置 `VLLM_USE_FLASHINFER_SAMPLER=0`

### 6.2 单次结果

| 指标 | B1 | T1 | T2 | B2 |
|---|---:|---:|---:|---:|
| wall time (s) | 198.835 | 193.079 | 190.565 | 192.455 |
| throughput (tok/s) | 999.100 | 1028.886 | 1042.460 | 1032.219 |
| TTFT mean (s) | 68.613 | 68.021 | 66.882 | 68.554 |
| TTFT p50 (s) | 59.081 | 59.143 | 71.605 | 59.332 |
| TTFT p90 (s) | 124.406 | 121.235 | 124.892 | 122.063 |
| TTFT p99 (s) | 172.699 | 153.320 | 155.876 | 161.129 |
| preemptions | 553 | 465 | 519 | 465 |
| prefix hits | 126,176 | 128,320 | 127,248 | 128,320 |

### 6.3 逐对方向与两次均值

| 指标 | Pair 1：B1→T1 | Pair 2：B2→T2 | B 两次均值 | T 两次均值 | 均值变化 |
|---|---:|---:|---:|---:|---:|
| wall time | -2.895% | -0.982% | 195.645 s | 191.822 s | -1.954% |
| throughput | +2.981% | +0.992% | 1015.660 | 1035.673 | +1.970% |
| TTFT mean | -0.863% | -2.440% | 68.583 s | 67.451 s | -1.651% |
| TTFT p50 | +0.105% | +20.685% | 59.207 s | 65.374 s | +10.417% |
| TTFT p90 | -2.549% | +2.318% | 123.235 s | 123.063 s | -0.139% |
| TTFT p99 | -11.221% | -3.260% | 166.914 s | 154.598 s | -7.379% |
| preemptions | -15.913% | +11.613% | 509 | 492 | -3.340% |
| prefix hits | +1.699% | -0.835% | 127,248 | 127,784 | +0.421% |

这里的“均值”是两次运行结果或两次 run-level percentile 的简单均值，不是把
所有请求样本重新合并后的总体 percentile。

wall time、throughput、TTFT mean 和 p99 在两对中方向一致；但 p50 明显退化，
p90 基本持平，最重要的 KV 机制指标 preemption 和 prefix hits 在第二对反向。
因此目前不能把平均改善解释成已经稳定的算法收益。

### 6.4 Tier 激活

| 指标 | B1 | T1 | T2 | B2 |
|---|---:|---:|---:|---:|
| normal blocks | 867,891 | 56,472 | 34,543 | 1,027,866 |
| resumed blocks | 0 | 933,505 | 693,844 | 0 |
| high-priority blocks | 0 | 0 | 0 | 0 |
| avoided evictions | 183 | 572 | 189 | 593 |
| fallback blocks | 8,760 | 8,841 | 8,665 | 8,530 |

压力服务使用 FCFS，因此 high-priority tier 为 0 是预期结果；该 tier 的行为由
priority Scheduler 单元测试覆盖。T1/T2 中 94% 以上的最终 demand mapping 属于
resumed tier，证明 Phase 5a 的 resume 分级在高抢占负载中被真实触发。

`avoided_evictions` 同样随整个调度轨迹明显波动，不能脱离 prefix hits 和
preemption 单独作为收益证明。

原始产物：

- `stress_bench_phase5a_binary_retention.json`
- `stress_bench_phase5a_priority_retention.json`
- `stress_bench_phase5a_priority_retention_repeat.json`
- `stress_bench_phase5a_binary_retention_repeat.json`

## 7. 验收与下一步

Phase 5a 的结论分为两部分：

- correctness：通过。默认 LRU、二值对照、tier 顺序、LRU 稳定性、soft
  fallback、priority 和 resume 语义都有测试保护；
- performance：暂不通过稳定收益 Gate。两次吞吐方向偏正，但 KV 指标和
  TTFT 分位数混合。

建议保留 `priority_aware` 实验开关，下一步进入 Phase 5b reclaimable-KV
estimate 前先保持策略边界独立。Phase 5b 只提供经过 refcount 测试的只读信息，
不要把 Phase 5a 的混合性能结果当作继续叠加复杂 heuristic 的理由。
