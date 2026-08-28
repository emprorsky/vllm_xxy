# Phase 3 实施报告：有界 Cache-Affinity 准入与 Aging

> 状态：实现、正确性测试和两组固定工作量 GPU A/B 已完成。
>
> 日期：2026-08-28 (UTC)
>
> 起始提交：`69e8763be`（Phase 2 已提交并推送）
>
> 结论：Gate 3 正确性通过；性能结果为“压力场景延迟/吞吐偏正、无抢占场景无明显开销”，但只有单次配对数据，且压力场景 prefix hit 略降，因此暂不宣称总体性能 Gate 通过。

## 1. 本阶段解决什么问题

Phase 1 决定“KV 不够时抢占谁”，Phase 2 决定“分配新块时优先保留哪些等待请求可能复用的缓存”。Phase 3 处理下一层问题：

> 当多个等待请求都可以准入时，不再永远只看队头；在严格受限的窗口和优先级范围内，允许更可能复用 KV、剩余 prefill 更少的请求先进入运行集。

实现必须同时满足三个约束：

1. 不能跨越队头的用户优先级；
2. 不能扫描整个 waiting queue，只看真实队列顺序前 `W` 个请求；
3. 不能让冷请求永久饿死，等待超过阈值后恢复基础队列顺序。

## 2. 最终策略

新增配置：

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `admission_policy` | `default` | 默认完全保持原准入顺序；`cache_affinity` 开启 Phase 3 |
| `kv_aware_aging_threshold_s` | `30.0` | 同优先级请求等待达到该秒数后进入 aged tier |
| `kv_aware_candidate_window` | `8` | Phase 2/3 共用的有界候选窗口 |

`cache_affinity` 的选择顺序是：

1. 只取合并后的真实等待顺序前 `W` 个请求；
2. 到第一个不同用户优先级的请求立即停止；
3. 若存在 aged 请求，在 aged 请求中按基础队列顺序选择；
4. 否则 resumed 请求优先于 fresh 请求；
5. 再选 `request.num_tokens - local_cached_tokens` 更小者；
6. 最后按基础队列位置稳定打破平局。

每次成功分配 KV 后，下一轮重新取候选和探测，不跨 KV mutation 保留旧排名。默认策略不会取候选窗口，也不会做推测性 prefix probe。

## 3. 关键实现取舍

### 3.1 Priority 队列是真正的有界 top-W

Priority queue 的底层是 heap，直接复制 heap 再排序会变成 `O(total waiting)`。本次使用 heap frontier：从根开始，只把已弹出节点的子节点放入临时 frontier，得到前 `W` 个元素，复杂度为 `O(W log W)`、临时空间 `O(W)`。

Priority item 在入队时捕获不可变排序键。`peek_n_with_keys()` 返回该冻结键，`waiting` 与 `skipped_waiting` 的全局合并也使用冻结键，而不是根据可能已变化的 Request 字段重新计算。

### 3.2 任意候选删除保持 O(log N)

cache-affinity 可能选择窗口中间的请求。Priority queue 新增 `request identity -> heap index` 映射，以及自维护的 swap/sift/pop-at，使删除选中请求保持 `O(log N)`，并在每次 heap 交换时同步索引。

FCFS 仍使用 deque；队头走原来的快速 pop，中间候选使用已有 remove。

### 3.3 失败分配不破坏队列

策略只选择候选，不提前删除。只有真实 `allocate_slots()` 成功后才移除请求；分配返回 `None` 时，选中请求和其他候选仍保持在原队列中，也不记录成功准入或真实 prefix-cache usage。

### 3.4 区分“选择尝试”和“成功准入”

压力场景中，同一批 waiting 请求可能在很多 scheduler step 被反复考虑，但因为 KV/slot 不足无法真正准入。为避免把尝试误读为效果，指标分成两组：

- 尝试：`selection_calls`、`candidates`、`candidate_probes`、`reordered`、`aged_selections`、`selected_cached_tokens`；
- 成功：`admitted`、`admitted_reordered`、`admitted_aged`、`admitted_cached_tokens`。

这些字段通过 `SchedulerStats` 和 Prometheus `vllm:kv_admission_*_total` 暴露，并由压测脚本按运行前后差值写入 JSON。

## 4. 代码改动

| 文件 | 改动 |
|---|---|
| `vllm/config/scheduler.py` | 新增 typed admission policy 和 aging threshold，默认关闭 Phase 3 |
| `vllm/engine/arg_utils.py` | 新增 CLI 参数并传入 `SchedulerConfig` |
| `vllm/v1/core/sched/policy.py` | 新增只读 `AdmissionCandidate` 和分层准入决策 |
| `vllm/v1/core/sched/request_queue.py` | 有界 heap frontier、冻结键读取、任意元素 O(log N) 删除和索引维护 |
| `vllm/v1/core/sched/scheduler.py` | 合并 waiting/skipped top-W、ready/LoRA/priority 过滤、探测、选择、成功后删除与统计 |
| `vllm/v1/metrics/stats.py` | 新增 `KVAdmissionStats` |
| `vllm/v1/metrics/loggers.py` | 新增 11 个 Phase 3 Prometheus counter |
| `Documents_xxy/stress_bench.py` | 抓取并汇总 Phase 3 尝试/成功指标 |
| `tests/engine/test_arg_utils.py` | 默认值与 CLI override 测试 |
| `tests/v1/core/test_preemption_policy.py` | default、resume/cache、aging 的纯策略测试 |
| `tests/v1/core/test_scheduler.py` | 窗口、优先级、aging、默认 fast path、分配失败、heap 删除、冻结键和统计集成测试 |
| `tests/v1/core/utils.py` | 测试 scheduler factory 参数透传 |
| `tests/v1/metrics/test_stats.py` | admission stats 的 msgpack 序列化回归 |

生产代码没有修改 CUDA/Triton kernel、Request 状态机或 KV allocation math。

## 5. 正确性与静态检查

最终回归命令使用 `/root/miniconda3/envs/vllm-dev/bin/python`，没有执行安装命令。

| 检查 | 结果 |
|---|---|
| scheduler + policy + CLI + metrics 回归 | `293 passed, 1 deselected` |
| Phase 3/Priority 定向回归 | `7 passed` |
| ruff（全部本阶段 Python 改动） | passed |
| compileall（生产代码与压测脚本） | passed |
| `git diff --check` | passed |
| 服务/GPU 残留检查 | 无服务、无 GPU compute process |

未过滤执行完整 `test_scheduler.py` 时为 `165 passed, 1 failed`。唯一失败是 `test_async_scheduling_pp_allows_rescheduling_with_output_placeholders` 在创建 `pipeline_parallel_size=2` 配置时发现机器只有 1 张 GPU，尚未进入本阶段 Scheduler 路径。最终组合回归只排除了这个硬件不满足的用例。

## 6. 固定工作量 GPU A/B

### 6.1 共同方法

- 模型：`Qwen/Qwen2.5-7B-Instruct`
- GPU：RTX 4090
- prefix caching：开启
- Phase 1：`recompute_aware`
- Phase 2：`waiting_queue_aware`
- candidate window：`8`
- 客户端：`temperature=0.7`、`ignore_eos=true`、每请求独立 seed
- 所有请求都以 `finish_reason=length` 完成，usage token 完整
- 每个 A/B 服务都独立冷启动，运行前有 48 个短 warmup 请求

比较项只有：

- D：`admission_policy=default`
- E：`admission_policy=cache_affinity`，aging `30 s`

### 6.2 KV 压力场景

配置：192 请求、并发 48、输出长度混合 `512/1024/1536`、`num_gpu_blocks_override=1250`。两边都准确生成 198,656 completion tokens，0 失败。

| 指标 | D：default | E：cache-affinity | E 相对 D |
|---|---:|---:|---:|
| wall time | 200.737 s | 197.422 s | -1.651% |
| output throughput | 989.635 tok/s | 1006.248 tok/s | +1.679% |
| TTFT mean | 72.928 s | 68.737 s | -5.746% |
| TTFT p50 | 67.098 s | 59.441 s | -11.412% |
| TTFT p90 | 159.176 s | 122.286 s | -23.176% |
| TTFT p99 | 163.303 s | 162.460 s | -0.516% |
| preemptions | 445 | 447 | +2 |
| prefix hits / queries | 131,040 / 154,312 | 127,616 / 154,312 | hits -2.613% |

E 的成功准入可观测结果：

| 指标 | 值 |
|---|---:|
| selection calls / candidates / probes | 8,359 / 65,195 / 65,195 |
| attempted reordered / aged | 2,003 / 5,431 |
| admitted | 627 |
| admitted reordered / aged | 362 / 266 |
| admitted cached tokens | 294,288 |

这里最重要的新信息是：只有 `627 / 8,359 = 7.50%` 的选择尝试最终成功准入，平均每次成功准入对应约 104 次候选 probe。当前策略在高压下会反复探测相同 waiting 集合，这是 Phase 4 lazy feature generation/cache 要解决的明确问题。

本场景的 wall time、吞吐和 TTFT 方向偏正，preemption 基本持平；但 prefix hits 下降 2.6%。另外一次探索性 E 重复运行是 197.919 s / 566 preemptions，说明抢占计数存在明显单次运行波动。因此不能用当前一对数据宣称稳定总体收益。

原始产物：

- `stress_bench_phase3_d_default_admission.json`
- `stress_bench_phase3_e_cache_affinity_success_metrics.json`
- `stress_bench_phase3_e_cache_affinity.json`（增加成功指标前的探索性重复，仅作稳定性参考）

### 6.3 无抢占、slot 限制场景

配置：96 请求、并发 32、`max_num_seqs=16`、输出长度混合 `256/512/768`、`num_gpu_blocks_override=2000`。两边都准确生成 49,920 completion tokens，0 失败、0 抢占。由于本地没有 `ninja`，这对 A/B 同时设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，避免 FlashInfer 为新 warmup shape 触发 JIT 构建；两边服务参数完全相同。

| 指标 | D：default | E：cache-affinity | E 相对 D |
|---|---:|---:|---:|
| wall time | 62.246 s | 62.197 s | -0.078% |
| output throughput | 801.985 tok/s | 802.607 tok/s | +0.078% |
| TTFT mean | 23.088 s | 22.587 s | -2.172% |
| TTFT p50 | 22.922 s | 22.662 s | -1.136% |
| TTFT p90 | 44.989 s | 44.970 s | -0.043% |
| TTFT p99 | 49.427 s | 49.420 s | -0.013% |
| prefix hits / queries | 68,432 / 79,235 | 68,432 / 79,235 | 完全相同 |
| preemptions | 0 | 0 | 相同 |

cache-affinity 在此场景完成 94 次成功准入，其中 77 次真实重排；尽管如此，wall time 和 throughput 仅相差 0.078%，没有观察到可见吞吐税。TTFT mean 的小幅改善不能在单次结果上解释为稳定收益，而 p90/p99 实际相同。

原始产物：

- `stress_bench_phase3_slot_default.json`
- `stress_bench_phase3_slot_cache_affinity.json`

## 7. Gate 3 判定

| Gate 3 条件 | 结果 |
|---|---|
| top-W 有界，不扫描/复制整个 Priority heap | 通过 |
| 不跨用户优先级 | 通过 |
| aging 防饿死 | 通过 |
| default 无推测性 prefix probe | 通过 |
| 中间候选删除不破坏 heap | 通过 |
| allocation failure 不丢失/重复请求 | 通过 |
| Priority waiting/skipped 合并使用冻结键 | 通过 |
| 配置、统计、序列化链路完整 | 通过 |

因此 **Phase 3 correctness Gate 通过**。性能方面保留为“有希望但尚未验收”：

- 正面信号：压力场景吞吐 +1.68%，mean/p50/p90 TTFT 改善；无抢占场景约等于 baseline；
- 负面/不确定信号：prefix hits -2.61%，单次 preemption 波动明显，只有一对严格 A/B；
- 已确认开销源：高压下大量失败前重复候选 probe。

## 8. 下一步建议

按总计划进入 Phase 4，但先保持范围很小：

1. 引入带 KV generation 的 lazy feature context；
2. 同一 KV generation 内复用候选的 local prefix 特征；
3. 在成功 allocation、preemption/free、cache reset 或外部 invalidation 后失效；
4. 用 `candidate_probes / admitted` 和 scheduler CPU profile 作为优化主指标；
5. 再做至少 3 轮交错 D/E 配对，确认吞吐、TTFT、prefix hits 和 preemptions 的分布。

不建议现在调整 aging 阈值或堆叠更多评分项。当前最清晰的问题是重复特征解析，而不是已有排序规则缺少更多启发式。
