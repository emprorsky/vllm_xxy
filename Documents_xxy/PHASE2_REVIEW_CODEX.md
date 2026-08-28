# Phase 2 独立复核：基线一致性、确定性复测与算法判断

> 状态：复核、可观测性补强和固定工作量复测完成；Phase 2 correctness
> 基本成立，但性能 Gate 仍未通过。
>
> 日期：2026-08-28 (UTC)
>
> 复核对象：`PHASE1_IMPLEMENTATION_REPORT.md`、
> `PHASE2_IMPLEMENTATION_REPORT_GLM.md`、Phase 2 当前代码与压测产物。
>
> 本文是独立复核记录，不修改或覆盖原 Phase 1/Phase 2 报告。

## 1. 结论摘要

1. 历史 baseline、Codex Phase 1 和 GLM Phase 2 的单次性能数据都不能组成
   严格可比的 A/B/C。压测只固定了请求形状，没有固定每个请求的模型采样；
   `temperature=0.7` 下，不同调度时序会生成不同长度的实际输出。
2. GLM 的 `default + LRU = 59.45 s / 1039 preemptions` 是异常 thrash
   运行，不能作为 Phase 1 的性能基线。
3. Codex Phase 1 报告中的 `32.12 s -> 30.15 s` 也不能证明性能提升。
   该报告原本已将结果限定为单次诊断数据，本次复核进一步确认其不适合用于
   policy 间的严格性能比较。
4. 在为每个请求显式设置独立生成 seed 后，Phase 1 的可信信号是
   preemptions 从 100 降到 64；墙钟时间和 TTFT 没有显示稳定收益。
5. 确定性 Phase 2 对照中，waiting-queue-aware 与 LRU 的 prefix
   hit/query 完全相同；墙钟差异处于运行噪声范围，preemptions 反而更高。
6. Phase 2 没有明显的核心 correctness bug。局部机制在人工构造场景中能够
   按设计保护目标块，但当前启发式没有把这种局部行为转化为真实 workload
   的额外 KV 命中。因此应判定为：**实现基本正确，测试偏局部，算法效果验证
   失败，Phase 2 性能 Gate 未通过。**

## 2. 为什么历史结果不一致

### 2.1 请求形状固定，但生成工作量没有固定

`Documents_xxy/stress_bench.py` 在模块加载时使用 `RNG_SEED=42`，它只固定：

- 请求是否使用共享前缀；
- 共享前缀池 ID；
- 每个请求的 `max_tokens` 上限。

模型请求仍然使用 `temperature=0.7`，payload 中没有 per-request `seed`。
因此 scheduler policy 一旦改变执行时序，采样 RNG 的消费和实际停止位置也可能
改变，导致参与比较的实际生成 workload 不同。

逐请求检查历史 JSON，所有运行的 `(req_id, prefix_pool, output_len)` 均一致，
但与 Phase 1 default 相比，`n_tokens` 相同的请求数为：

| 运行 | `n_tokens` 相同请求数 | 总请求数 |
|---|---:|---:|
| Phase 1 recompute-aware | 46 | 192 |
| Phase 2 default | 2 | 192 |
| Phase 2 recompute-aware + LRU | 3 | 192 |
| Phase 2 waiting-queue-aware | 3 | 192 |

这里的 `n_tokens` 还不是 tokenizer token，而是客户端收到的 SSE content
chunk 数。它只能进一步证明输出流形态不同，不能作为真实生成 token 数使用。

### 2.2 现有 throughput 与 ITL 口径不可靠

当前脚本每收到一个非空 content delta 就执行一次 `n_tokens += 1`，因此
`output_throughput_tok_s` 实际是 content-chunk/s。即使请求使用相同 seed，
不同调度下的流式发送/解码分组也可能造成 chunk 数不同。

ITL 也存在两个问题：

- 统计单位是 content chunk，不是 token；
- 第一个 content chunk 到第二个 content chunk 的间隔没有进入差分数组。

报告中的 `itl_mean.p99` 还是“每请求 p99 chunk ITL 的跨请求 p99”，不是标准的
全局 token ITL p99。因此历史 throughput/ITL 只能用于观察压力形态，不能用于
精确的 policy 性能结论。

### 2.3 单次运行存在明显机器状态噪声

本次在相同当前代码、相同模型、相同请求 seed 下，recompute-aware + LRU
分别出现约 40.57 s 和 32.52 s 两种运行状态。两次 preemptions 和 prefix
hit/query 完全相同，说明约 8 s 的墙钟差异并非来自调度语义变化。

因此所有单次 run 都不足以支撑吞吐/延迟结论，尤其不能用一个异常 default
run 与另一个正常 recompute run 计算 Phase 1 的收益。

## 3. 本次确定性复测方法

### 3.1 公共服务配置

当前提交：`1cb3b2f00`。

模型与服务配置：

```text
model: Qwen/Qwen2.5-7B-Instruct
GPU: NVIDIA GeForce RTX 4090
max_model_len: 8192
gpu_memory_utilization: 0.75
num_gpu_blocks_override: 1250
GPU KV cache: 20,000 tokens
enable_prefix_caching: true
model runner: V2
```

压测形状保持原主场景：

```text
concurrency: 48
total_requests: 192
prefix_pools: 8
shared_ratio: 0.7
output_lens: [512, 1024, 1536]
temperature: 0.7
request seed: 100000 + req_id
```

per-request seed 通过运行时替换客户端请求函数加入，没有修改仓库文件。本次临时
结果写入 `/tmp/kvaware_seeded_*.json`，未作为正式 benchmark artifact 提交。

### 3.2 服务模式

| 名称 | preemption policy | prefix eviction policy |
|---|---|---|
| A | `default` | `lru` |
| B | `recompute_aware` | `lru` |
| C | `recompute_aware` | `waiting_queue_aware` |

每次实验都重新启动服务；实验结束后已关闭 API server 和 EngineCore，GPU 无
compute app 残留。

## 4. 确定性复测结果

### 4.1 原始运行结果

| 运行 | 模式 | 墙钟 | TTFT mean | TTFT p99 | preemptions | prefix hit/query |
|---|---|---:|---:|---:|---:|---:|
| A | default + LRU | 32.152 s | 9.625 s | 21.653 s | 100 | 141168 / 154312 |
| B1 | recompute-aware + LRU | 40.571 s | 17.361 s | 30.279 s | 64 | 141168 / 154312 |
| C1 | B + waiting-queue-aware | 40.610 s | 17.414 s | 30.532 s | 84 | 141168 / 154312 |
| B2 | recompute-aware + LRU | 32.519 s | 9.739 s | 22.238 s | 64 | 141168 / 154312 |
| C2 | B + waiting-queue-aware | 31.392 s | 9.725 s | 22.441 s | 78 | 141168 / 154312 |

### 4.2 Phase 1：A 与正常运行态 B2

| 指标 | A: default | B2: Phase 1 | 相对变化 |
|---|---:|---:|---:|
| 墙钟 | 32.152 s | 32.519 s | +1.14% |
| TTFT mean | 9.625 s | 9.739 s | +1.18% |
| TTFT p99 | 21.653 s | 22.238 s | +2.70% |
| preemptions | 100 | 64 | **-36.0%** |
| prefix hit/query | 141168 / 154312 | 141168 / 154312 | 持平 |

这组结果支持 Phase 1 减少抢占的方向，但不支持吞吐或延迟已经改善。由于仍然
只有一对正常运行态对照，`-36% preemptions` 也应在多轮交替实验中复核。

### 4.3 Phase 2：B/C 两组运行态内对照

| 指标 | B1 -> C1 | B2 -> C2 |
|---|---:|---:|
| 墙钟变化 | +0.10% | -3.47% |
| TTFT mean 变化 | +0.31% | -0.15% |
| TTFT p99 变化 | +0.84% | +0.92% |
| preemptions | 64 -> 84（+31.25%） | 64 -> 78（+21.88%） |
| prefix hit/query | 完全相同 | 完全相同 |

两组 B/C 都没有增加任何 prefix hit。墙钟和 TTFT 的小幅正负变化不能与已观察
到的机器运行态噪声区分；preemptions 则在两组中都高于 LRU。

因此 GLM 报告中的“C 比 B 墙钟慢 13.9%”幅度不能复现，但它关于 Phase 2
没有产生 prefix-cache 收益的核心判断得到确认。

## 5. Phase 2 代码复核

### 5.1 没有发现明显 correctness bug

默认 `lru` 模式下，Scheduler 的 retention resolver 为 `None`；BlockPool
直接调用原有 `free_block_queue.popleft_n(num_blocks)`。Phase 2 不会在默认模式
下进入 `popleft_n_avoiding`。

aware 模式的主要不变量也成立：

- retained block 只是软偏好，全部 free block 都 retained 时仍会按原顺序回退；
- 只移除实际分配的 block，没有永久 pin 或容量泄漏；
- 未选 block 保持相对链表顺序；
- prefix peek 不 touch block、不改 refcount、不生成事件；
- resolver 在一次 `allocate_slots` transaction 内惰性求值并缓存；
- 当前正在 admission 的请求会被排除，因为其真实 hit 会在分配内被 touch。

Phase 2 的 Scheduler 集成单测还构造了一个可生效场景：LRU 下等待请求只能命中
32 tokens，aware 下能命中 48 tokens。这证明代码能够实现设计的局部选择语义。

据此，GLM 的 `1039 preemptions` 更像 default policy 在不受控生成负载下进入
thrash 异常态，而不是 Phase 2 默认 LRU 分支被改坏。

### 5.2 报告中的一处机制解释不准确

原报告将更多 preemptions 解释为“跳过 retained block 让可用块更紧”。
这不准确：soft retention 不减少 free block 数量，也不改变 allocation 的可行性；
当非 retained block 不足时会使用 retained block 回退。

它可能通过改变缓存内容、后续命中、admission 时序或 Scheduler 执行耗时，间接
改变抢占轨迹；本次 B/C hit 完全相同，因此更多 preemptions 也可能只是调度时序
扰动。需要内部 telemetry 才能进一步归因，不能直接解释为容量减少。

## 6. 为什么算法没有转化成收益

### 6.1 目标前缀经常已经被 active request 自然保护

当前 workload 只有 8 个共享 prefix pool，却有 48 个并发请求。热门前缀很可能
同时被 running 请求引用，其 block `ref_cnt > 0`，本来就不在可淘汰 free queue
中。waiting-demand resolver 即使识别到这些 block，也没有额外保护价值。

### 6.2 保护窗口太短

waiting request 成为当前 admission 对象后，真实 cache lookup 会先 touch 其
命中块，随后该请求从 resolver 中排除。Phase 2 只保护“当前分配与候选未来
admission 之间”的短窗口。在 vLLM 当前 prefill/admission 顺序下，这个窗口内
真正会被 LRU 淘汰、随后又被候选命中的 block 很少。

### 6.3 候选信息没有价值排序

实现把窗口内候选能命中的所有物理 block ID 做等权并集，没有考虑：

- 候选在队列中的距离；
- 相同 prefix 的等待需求次数；
- 预计 admission 时间；
- block 的 recompute cost；
- retained 集合大小及其对其他缓存项的机会成本。

`window=16` 只限制请求数量，不限制 retained block 数量。长前缀或多 group 模型
可能一次保留较大的集合。

### 6.4 soft retention 不是性能意义上的 LRU fallback

当 retained 集合只是 free queue 的一部分时，`popleft_n_avoiding` 会跳过 LRU
队头的 retained block，转而淘汰更靠后的 non-retained block。后者可能更新、
更热门，或者具有更高的未来复用价值。

因此 soft retention 只保证“不会导致分配失败”，不保证“最坏性能退化为 LRU”。
当前策略没有衡量被保护收益与被迫淘汰其他缓存项的损失。

## 7. 测试覆盖评价

现有测试较完整地覆盖了局部 correctness：选择顺序、fallback、链表不变量、
只读 peek、resolver 调用次数、官方 eviction accounting、候选窗口和 admission
顺序。

缺少的是“策略有效性”与反例覆盖：

1. retained block 后续是否真的产生了 realized cache hit；
2. 跳过 retained block 是否淘汰了未来价值更高的 non-retained block；
3. 多个 active request 共享同一前缀时 resolver 是否基本为 no-op；
4. retained 集合很大时的扫描成本和 cache pollution；
5. 确定性、真实 token 计数的端到端 LRU/aware 对照；
6. 多轮 paired run 的统计稳定性。

所以“单测全绿”只能说明机制按预期执行，不能说明策略对真实 workload 有收益。

## 8. 最终定性

| 阶段 | Correctness | 当前性能结论 | 状态 |
|---|---|---|---|
| Phase 1 | 通过；victim、resume protection、Priority 重入队已有覆盖 | preemptions 有下降信号；吞吐和延迟收益未证明 | 功能完成，性能待复验 |
| Phase 2 | 局部机制与安全不变量基本通过 | prefix hit 无增加，preemptions 偏高，无稳定吞吐/延迟收益 | **性能 Gate 未通过** |

Phase 2 结果不能证明“等待队列感知 KV”整个方向无效，但已经证明当前
“分配时跳过 retained free block”的弱启发式在现有 workload 上没有产生目标
收益。按照统一实施计划的 stop condition，不应把本实现描述为性能优化成功。

## 9. 后续建议

在继续修改算法之前，先修正性能验证基础：

1. 每请求固定独立 model sampling seed，并保存输出 token 数或输出 hash；
2. 使用 API usage 或 tokenizer 统计 completion tokens，不以 SSE chunk 代替 token；
3. 修复 token-level ITL 口径；
4. A/B/C 交替顺序运行至少 5 轮，记录 GPU/CPU 状态；
5. 增加 `retention_candidates`、`retained_blocks`、`avoided_evictions`、
   `realized_retained_hits` 等内部计数；
6. 如果 `realized_retained_hits` 仍接近零，应停止打磨当前选择器，改为有预算的
   value-ranked retention，或进入 Phase 3 的 bounded admission + aging 实验。

## 10. 后续实施：修正压测口径

### 10.1 为什么仅固定 seed 仍不够

本轮先把每个请求固定为 `seed=100000+req_id`，又额外尝试
`temperature=0`。两种方式下，不同 policy 之间仍分别出现大量 completion token
数量或输出 hash 不一致。V2 runner 的动态 batch、CUDA graph 和 GPU 数值路径仍会
使逐 token 生成发生分叉。因此 seed/hash 适合审计输出，不能单独保证调度实验的
decode 工作量相同。

最终采用以下控制方法：

- 请求携带 `ignore_eos=true`，每个请求必须生成其预设 `max_tokens`；
- 从流式响应的 `usage.completion_tokens` 读取真实 token 数；
- 保存每请求输出 SHA-256、finish reason、prefix pool 和预设长度；
- 正式计时前发起 48 路短请求，预热并发采样/JIT 路径；
- 预热完成后才读取 `/metrics` 基准值，因此预热不进入正式统计窗口；
- chunk 数和真实 token 数分开，历史 chunk throughput 不再冒充 tok/s。

未固定长度的探索结果保留为
`stress_bench_phase2_telemetry_{lru,aware}.json`（temperature 0.7）和
`stress_bench_phase2_greedy_{lru,aware}.json`（temperature 0）。它们用于证明
seed/greedy 仍不足以固定工作量，不用于最终性能表。

严格复测命令中的客户端环境为：

```bash
BENCH_TEMPERATURE=0.7 BENCH_IGNORE_EOS=1 BENCH_WARMUP_REQUESTS=48 \
  /root/miniconda3/envs/vllm-dev/bin/python \
  Documents_xxy/stress_bench.py <output.json>
```

服务端继续使用 §3 的 20,000-token KV 压力配置。LRU、W4、W8、W16 每轮都
重新启动服务；除 eviction policy 和 aware 的 candidate window 外，参数相同。

### 10.2 严格一致性检查

同预热 LRU、aware W4/W8/W16 四轮均满足：

- 192/192 请求成功，`usage_missing=0`；
- 每轮 completion tokens 总数均为 `198656`；
- 192/192 请求均有 `completion_tokens == output_len`；
- 192/192 请求均以 `finish_reason=length` 结束；
- LRU 与 aware 的 `(req_id, prefix_pool, output_len)` 差异为 0；
- LRU 与 W16 的 completion token 数差异为 0。

输出 hash 仍会随调度改变，LRU 与 W16 有 150/192 个请求不同。这不再改变
prompt/decode token 数、KV 占用形状或结束条件；生成内容本身也不会被后续请求
复用，因此不破坏本次 scheduler/KV 工作量对照。

## 11. 后续实施：Phase 2 telemetry

新增并接入以下 Prometheus counter，LRU 路径保持为 0：

| 指标 | 含义 |
|---|---|
| `kv_retention_resolver_calls_total` | waiting-demand resolver 实际执行次数 |
| `kv_retention_candidates_total` | 被探测的等待请求引用次数 |
| `kv_retention_candidates_with_hits_total` | 探测时存在本地命中的候选引用次数 |
| `kv_retention_blocks_total` | 每次 resolver 得到的 retained block 并集大小之和 |
| `kv_retention_avoided_evictions_total` | 本次选择确实跳过的 LRU 队头 retained block 数 |
| `kv_retention_fallback_blocks_total` | non-retained 不足后不得不分配的 retained block 数 |

实现沿 `BlockRetentionHint -> BlockPool -> KVCacheManager -> SchedulerStats ->
Prometheus` 传递一次 allocation selection 的结果，不新增全局状态，也不改变
默认 LRU 选择。相关单测覆盖 observer 精确计数、Scheduler 聚合/清零和 msgpack
序列化。

## 12. 固定工作量复测结果

### 12.1 同预热可比结果

| 模式 | 墙钟 | 相对 LRU | output tok/s | TTFT mean / p99 | preemptions | prefix hit/query |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 202.376 s | 基准 | 981.62 | 72.934 / 163.404 s | 449 | 130464 / 154312 |
| aware W4 | 200.758 s | -0.80% | 989.53 | 72.932 / 163.314 s | 445 | 130464 / 154312 |
| aware W8 | 200.709 s | -0.82% | 989.77 | 72.913 / 163.265 s | 445 | 131040 / 154312 |
| aware W16 | 199.898 s | -1.22% | 993.78 | 72.609 / 162.601 s | 445 | 131040 / 154312 |

另有一轮只做单请求预热的 LRU：202.221 s / 470 preemptions /
127248 hits。该轮正式负载开始时仍发生首次并发 sampling JIT，因此不用于上表的
墙钟 A/B，只保留为冷态诊断 artifact。

### 12.2 retention 内部信号

| 模式 | resolver calls | candidates | retained block refs | avoided evictions | fallback blocks |
|---|---:|---:|---:|---:|---:|
| W4 | 11875 | 40540 | 890220 | 183 | 7744 |
| W8 | 11875 | 79552 | 1489604 | 247 | 7801 |
| W16 | 11877 | 152325 | 2418013 | 292 | 8110 |

W8 相对 W16：

- candidates 减少 47.8%；
- retained block refs 减少 38.4%；
- prefix hits 和 preemptions 完全相同；
- avoided evictions 从 292 降到 247，但没有损失最终可观察命中。

因此将 `kv_aware_candidate_window` 的实验分支默认值从 16 调整为 8；CLI
仍可显式覆盖，不改变默认 `prefix_cache_eviction_policy=lru`。

原始文件：

- `stress_bench_phase2_fixed_work_lru_warm.json`；
- `stress_bench_phase2_fixed_work_aware_w4.json`；
- `stress_bench_phase2_fixed_work_aware_w8.json`；
- `stress_bench_phase2_fixed_work_aware.json`（W16）。

## 13. 结果解释与当前决策

严格工作量测试修正了此前“Phase 2 一定是反效果”的判断：当前机制确实能在真实
压力下改变淘汰，并产生小幅正向信号。W8 相比 LRU 多命中 576 prompt tokens，
少 4 次抢占，单轮吞吐高 0.83%。因此实现不是纯 no-op，也没有证据表明存在导致
整体退化的 correctness bug。

但该信号仍不足以通过性能 Gate：

- 0.8%--1.2% 的单轮墙钟差异很可能落在机器运行噪声内；
- W8 的 1489604 次 retained block 引用只对应 247 次实际跳过队头，机会非常稀疏；
- 多出的 576 hit tokens 仅相当于 36 个 16-token blocks；
- 目前没有追踪“某个被保护 block 后来真的被哪次 admission 命中”的因果 ID，
  hit 增量只能通过 paired LRU 间接观察；
- 只有一个模型、一个压力形状和每个 policy 单轮，不足以证明泛化或统计显著性。

当前决策是：

1. 保留 Phase 2 correctness 实现、telemetry 和严格压测器；
2. 默认策略继续是 LRU，waiting-queue-aware 仍是显式实验选项；
3. 采用 W8 作为实验默认值，不继续在本场景过拟合窗口参数；
4. 暂不投入复杂 value-ranked retention，因为真实可利用机会仍过少；
5. 下一步优先进入 Phase 3 的 bounded admission + aging，让 KV 信息影响请求准入，
   同时用现有固定工作量 benchmark 做 paired、多轮 Gate；
6. 若 Phase 3 也不能稳定提高 prefix hits/吞吐，则按实施计划停止继续叠加复杂度。

## 14. 本轮代码改动与验证

主要改动位置：

| 文件 | 改动 |
|---|---|
| `Documents_xxy/stress_bench.py` | 固定请求 seed；支持固定长度、并发预热和可配置输出长度；读取 usage 真 token；保存 hash/finish reason；区分 token 与 SSE chunk；抓 retention metrics |
| `vllm/v1/metrics/stats.py` | 新增 `KVRetentionStats`，随 `SchedulerStats` 传输 |
| `vllm/v1/metrics/loggers.py` | 暴露 6 个 `vllm:kv_retention_*_total` counter |
| `vllm/v1/core/kv_cache_utils.py` | `BlockRetentionHint` 支持 selection observer |
| `vllm/v1/core/block_pool.py` | 统计实际跳过的 LRU retained block 与 fallback block |
| `vllm/v1/core/kv_cache_manager.py` | allocation transaction 透传 observer |
| `vllm/v1/core/sched/scheduler.py` | 聚合 resolver/selection telemetry，并在 stats flush 后清零 |
| `vllm/config/scheduler.py` | 将实验默认 candidate window 从 16 调整为 8 |
| `tests/v1/core/*`、`tests/v1/metrics/test_stats.py` | 增加选择计数、Scheduler flush/reset 和序列化断言 |
| `tests/engine/test_arg_utils.py` | 冻结 LRU/W8 默认值及 CLI 覆盖行为 |

最终检查：

- 所有改动 Python 文件 `ruff check`：通过；
- 所有改动 Python 文件 `compileall`：通过；
- `git diff --check`：通过；
- Phase 2 prefix retention/peek 定向测试：10 passed；
- Scheduler retention/queue 定向测试：5 passed；
- telemetry stats 序列化：1 passed；
- CLI/default 及相邻格式用例：3 passed；
- 四轮严格 GPU workload：全部 192/192 成功，无 usage 缺失；
- 最终 API server/EngineCore 已关闭，GPU 无计算进程残留。
