# Phase 4 实施报告：代际化 Lazy SchedulingFeatureContext

日期：2026-08-28（UTC）

分支：`project/kv-aware-scheduling`

起点：`466a93184`（Phase 3）

## 1. 结论

Phase 4 的 correctness Gate 已通过。新实现把 Phase 2 retention 和 Phase 3
cache-affinity 重复使用的 local-prefix 信息集中到一个延迟求值、按 KV
generation 失效的 `SchedulingFeatureContext` 中：

- 默认 admission + LRU 路径不创建 context，不增加 prefix lookup；
- cheap feature 只读 Request 元数据，不访问 KV；
- expensive local-prefix feature 在同一 KV generation 内按 Request 身份复用；
- allocation、free、eviction、prefix reset 等可能改变 KV 可见性的操作后立即
  清空 memoization；
- feature 不写入 `Request`，不会变成跨迭代的隐式状态；
- 预计算 prefix 进入真实 admission 时仍走原有事件上报逻辑。

固定压力 A/B 中，cache-affinity 侧共有 154,192 次 prefix feature read，实际
解析 141,943 次，memoization 命中 12,249 次，即减少 **7.944%** 的 resolver
调用。功能目标成立，但 Phase 3 的主要高压开销没有完全消失：成功 admission
为 635 次，候选实际 probe 为 64,864 次，仍是 **102.15 probes/admission**。

因此本阶段的准确评价是：**抽象和安全代内复用有效；跨 KV mutation 的反复
失败选择仍需后续策略处理。** 单轮性能信号偏正，但不足以宣称稳定收益。

## 2. 为什么采用 KV generation

prefix hit 由物理 block 的占用、缓存、引用和驱逐状态决定。一次 allocation
即使最终返回失败，也可能先执行 skipped-block 清理；free、CoW、异步 KV
接收、invalid-block eviction 等同样会改变后续 lookup 结果。因此不能按
“scheduler iteration”或固定 TTL 缓存。

本实现采用保守规则：

```text
local_prefix(request)
        |
        +-- 当前 generation 已解析：直接复用
        |
        +-- 未解析：event-free KV lookup -> memoize

任何潜在 KV mutation
        |
        +-- generation += 1
        +-- clear all KV-derived features
```

这保证不会为了提高命中率而复用可能已经过期的 block 对象。缓存项同时保存
Request 对象并校验身份，避免只使用 `id(request)` 时的对象 ID 重用问题。

## 3. 代码改动

| 文件 | 改动 |
|---|---|
| `vllm/v1/core/sched/feature_context.py` | 新增 cheap/local-prefix feature、延迟 resolver、Request 身份 memoization、KV generation 失效和统计 |
| `vllm/v1/core/sched/scheduler.py` | 两个现有 consumer 接入 context；默认 fast path 不创建 context；覆盖 KV mutation 失效点；真实 admission 复用已解析结果 |
| `vllm/v1/core/kv_cache_manager.py` | 新增返回完整三元组的 event-free peek；真实 lookup 可接收当前 generation 的预计算结果，同时保留 full-report 事件 |
| `vllm/v1/metrics/stats.py` | 新增 `SchedulingFeatureStats` |
| `vllm/v1/metrics/loggers.py` | 新增 4 个 Prometheus counter |
| `Documents_xxy/stress_bench.py` | 抓取并汇总 Phase 4 feature 指标 |
| `tests/v1/core/test_scheduler.py` | 增加 memoization、失效、cheap fast path、非 Request 状态测试，并验证真实 admission 不重复 lookup |
| `tests/v1/core/test_prefix_caching.py` | 验证预计算复用不会丢失 full-report KV 事件 |
| `tests/v1/metrics/test_stats.py` | 增加新统计的 msgpack 序列化回归 |

新增 Prometheus 指标为：

- `vllm:scheduling_feature_prefix_requests_total`
- `vllm:scheduling_feature_prefix_resolutions_total`
- `vllm:scheduling_feature_prefix_cache_hits_total`
- `vllm:scheduling_feature_invalidations_total`

## 4. 失效边界

Scheduler 在下列操作后调用统一的
`_invalidate_scheduling_features()`：

| 类别 | 覆盖点 |
|---|---|
| allocation | running 和 waiting 两个 `allocate_slots` 路径；成功和失败都失效 |
| free | 直接 request free、deferred block pop/free、CoW retained block free |
| partial tail | 取得 partial-tail offload 后 |
| prefix cache | 成功 reset、connector 完成后的 skipped-block removal/cache/free |
| remote/async KV | 接收成功后的 cache，失败后的 cache/free |
| eviction | invalid/failed KV load block eviction |

这里故意选择 correctness 优先的保守失效。例如不能只在
`allocate_slots()` 成功时失效，因为失败路径也可能已经清理 skipped blocks。

## 5. 正确性验证

所有命令都直接使用：

```bash
/root/miniconda3/envs/vllm-dev/bin/python
```

没有运行 uv、pip 或任何安装命令。核心新增测试证明：

- 同 generation 的两个 consumer 只触发一次 resolver；
- KV allocate/free 后 generation 前进，旧 prefix 不再复用；
- priority、arrival、resume、computed-token 等 cheap feature 零 KV 调用；
- 完成一次真实 scheduler 流程后，Request 上没有 feature-cache 字段；
- default admission 路径的 context 为 `None`；
- Phase 3 选中 candidate 的真实 admission 复用预计算结果，不发生第三次
  coordinator lookup；
- full-report 模式下，event-free peek 本身无事件，而真实 admission 使用同一
  预计算结果时仍产生原有 `BlockStored` 事件。

最终检查结果：

| 检查 | 结果 |
|---|---|
| scheduler + policy + CLI + metrics 组合回归 | `297 passed, 1 deselected` |
| 完整 `test_prefix_caching.py` | `102 passed` |
| 最终身份缓存定向回归 | `5 passed, 165 deselected` |
| ruff（本阶段全部 Python 改动） | passed |
| compileall（生产代码与压测脚本） | passed |
| `git diff --check` | passed |

组合回归继续排除唯一需要 2 张 GPU 的
`test_async_scheduling_pp_allows_rescheduling_with_output_placeholders`；本机只有
1 张 RTX 4090，该用例在构造 PP=2 配置时失败，未进入本阶段代码路径。

## 6. 固定工作量 GPU A/B

### 6.1 方法

- 模型：`Qwen/Qwen2.5-7B-Instruct`
- GPU：RTX 4090
- 服务共同配置：prefix caching、`recompute_aware`、
  `waiting_queue_aware`、candidate window 8、GPU blocks 1250
- 客户端：192 请求、并发 48、输出长度 `512/1024/1536`、48 个 warmup、
  `temperature=0.7`、`ignore_eos=true`、每请求独立 seed
- 比较项：D 为 default admission；E 为 cache-affinity admission
- D/E 都是 192/192 成功、0 错误、198,656 completion tokens，全部
  `finish_reason=length`

代码 hash 变化后，FlashInfer sampler 尝试 JIT 时发现本机没有 `ninja`，首次
服务启动在创建 Scheduler 前失败。没有安装依赖；最终 D/E 都统一设置
`VLLM_USE_FLASHINFER_SAMPLER=0`，其余参数相同。两组结束后服务均已关闭。

### 6.2 性能和缓存结果

| 指标 | D：default | E：cache-affinity | E 相对 D |
|---|---:|---:|---:|
| wall time | 200.162 s | 197.010 s | -1.575% |
| output throughput | 992.474 tok/s | 1008.357 tok/s | +1.600% |
| TTFT mean | 72.742 s | 68.596 s | -5.700% |
| TTFT p50 | 66.931 s | 59.317 s | -11.376% |
| TTFT p90 | 158.733 s | 122.025 s | -23.126% |
| TTFT p99 | 162.848 s | 162.119 s | -0.448% |
| preemptions | 445 | 455 | +10 |
| prefix hits / queries | 131,040 / 154,312 | 127,616 / 154,312 | hits -2.613% |

结果方向与 Phase 3 的同类单轮压力测试基本一致：wall time、吞吐和 TTFT
改善，prefix hit 略降，preemption 没有改善。由于这是单轮测试，不能把这些
差异解释为稳定因果收益。

### 6.3 FeatureContext 结果

| 指标 | D：default | E：cache-affinity |
|---|---:|---:|
| prefix feature reads | 78,964 | 154,192 |
| actual resolutions | 78,964 | 141,943 |
| memoization hits | 0 | 12,249 |
| resolver reduction | 0% | 7.944% |
| KV invalidations | 208,014 | 207,610 |

E 的 154,192 次 read 可以完整分解：

```text
64,864 admission candidate reads
+ 8,347 selected-candidate real-admission reads
+ 80,981 retention candidate reads
= 154,192
```

12,249 次 memoization hit 中：

- 8,347 次来自“选择阶段已解析，真实 admission 直接复用”；
- 3,902 次来自同 generation 内 admission 与 retention consumer 的交叉复用。

D 只有 retention consumer。一次 allocation 后必须立即失效，而 retention
resolver 自身已对单次选择去重，因此 D 的 memoization hit 为 0 是预期结果，
不是 context 未生效。

### 6.4 遗留的 probe 放大

E 的 Phase 3/4 admission 指标为：

| 指标 | 值 |
|---|---:|
| selection calls | 8,347 |
| candidates / actual probes | 64,864 / 64,864 |
| successful admissions | 635 |
| reordered / aged selections | 2,030 / 5,382 |
| admitted reordered / aged | 364 / 275 |
| probes per successful admission | 102.15 |

高压时许多选择后紧接着发生 allocation 尝试；无论成功失败，都可能改变 KV
状态，因此 context 必须失效。下一次重试不能安全复用旧物理 block 结果。
所以 Phase 4 消除了同一安全代际内的重复 lookup，却不能靠缓存解决跨 mutation
的重试放大。若要降低这一项，需要改变“何时值得重新选/探测”的 admission
控制流或增加可证明安全的版本粒度，而不是放宽失效条件。

原始产物：

- `stress_bench_phase4_d_default_features.json`
- `stress_bench_phase4_e_cache_affinity_features.json`

## 7. 下一步建议

可以按统一计划进入 Phase 5，但应保持实验性开关和停止条件：

1. 先做 Phase 5a 的 priority/demand-aware retention tier，不改变默认 LRU；
2. 单元测试明确 user priority、resume tier、稳定 LRU fallback 和 allocation
   feasibility；
3. 再考虑 Phase 5b reclaimable-block estimate，严格区分“最终可回收”和
   “本次 allocation retry 立即可用”；
4. 每个阶段都保留固定工作量 A/B，若 prefix hit、preemption、吞吐与 TTFT
   无一致改善，就停止继续叠加策略复杂度。

Phase 4 不建议通过减少失效点来追求更高 cache-hit 数字；那会把一个可观测的
效率问题变成潜在的 stale-block correctness 问题。
