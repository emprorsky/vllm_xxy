# Phase 2 实施与验收记录：等待队列感知的前缀 KV 保留

> 状态：功能验收完成（单测全绿）；4090 压测显示本阶段在已测负载下**无净收益**，
> 且 default 基线出现一次与 Phase 1 不一致的异常运行，待 Codex 复核（见 §5.3、§8）。
>
> 日期：2026-08-28 (UTC)
>
> 范围：`prefix_cache_eviction_policy="waiting_queue_aware"` 的完整实现——
> 等待队列近头部请求的前缀需求感知、块分配时的软保留（soft retention）、
> 以及配套的只读探测与队列 peek 原语。Phase 3 的准入控制与 aging 未开始。
>
> 执行者：GLM（Trae 会话）。与 Phase 1 报告（Codex 执行）区分，
> 本文件命名为 `PHASE2_IMPLEMENTATION_REPORT_GLM.md`。

## 1. 本阶段要解决什么

Phase 1 之后，抢占已经做到"选计算量最小的 victim"。但在高 KV pressure 下，
即使请求未被抢占，其**可复用的前缀缓存块**也可能在为 running 请求分配新块时
被 LRU 淘汰；随后等待队列中本可命中该前缀的请求只能全量重算 prefill。

Phase 2 的目标：在从 free queue 取块时，**跳过**近头部等待请求的前缀命中块
（软保留），只有当非保留块不足时才回退使用它们。默认路径（LRU）行为完全不变。

## 2. 最终行为与配置

新增两个 `SchedulerConfig` 字段（同步暴露到 `EngineArgs` / CLI）：

| 参数 | 默认值 | 语义 |
|---|---|---|
| `prefix_cache_eviction_policy` | `"lru"` | `"lru"`：原始行为；`"waiting_queue_aware"`：分配时软保留等待队列近头部请求的前缀命中块 |
| `kv_aware_candidate_window` | `16`（≥1） | 扫描等待队列近头部多少个请求作为"需求候选" |

行为规则：

| 场景 | `lru`（默认） | `waiting_queue_aware` |
|---|---|---|
| `BlockPool.get_new_blocks` | `free_block_queue.popleft_n(n)`（原路径，零开销） | 若 free 队头 n 块内无 cached block，走原 fast path；否则解析 retained 集合，`popleft_n_avoiding(n, retained)` 跳过保留块，不足时回退取保留块 |
| retained 集合来源 | — | 对候选窗口内每个 `num_computed_tokens==0` 且非 blocked 状态的等待请求，`peek_computed_blocks` 只读探测其可命中前缀块 ID 的并集（排除当前正在调度的请求） |
| 候选窗口取法 | — | FCFS：`skipped_waiting` + `waiting` 各 `peek_n`；Priority：按**入队时冻结**的 order key `heapq.merge` 两队列取前 window 个（与 Phase 1 的冻结 key 语义一致） |
| Mamba / CrossAttention manager | 原路径 | `allocate_new_blocks` / `allocate_external_computed_blocks` 透传 `retention_hint`（含 Mamba align 模式） |

`peek_computed_blocks` 与 `get_computed_blocks` 的区别：**只读**——不 touch
cached block、不改 LRU 顺序、不产生副作用（有专门单测 `test_prefix_peek_is_read_only` 断言 free queue 前后一致）。

## 3. 代码改动

整体规模：17 个文件 `+1225 / -90`（其中核心代码+测试 16 文件 `+1210/-84`；
`AGENTS.md +8` 为本机 LOCAL OVERRIDE，与 Phase 2 无关）。

| 文件 | 改动 |
|---|---|
| `vllm/v1/core/kv_cache_utils.py` | 新增 `BlockRetentionHint`（惰性 resolver 包装）与 `FreeKVCacheBlockQueue.popleft_n_avoiding(n, retained_ids)`：从头扫描跳过保留块，不足 n 时按顺序回退取保留块。 |
| `vllm/v1/core/block_pool.py` | `get_new_blocks` 增加 `retention_hint` 参数；用 `itertools.islice` 探测 free 队头 n 块内是否存在 cached block（无则直接走原 `popleft_n` fast path，避免无谓 resolve）。 |
| `vllm/v1/core/kv_cache_manager.py` | 新增 `peek_computed_blocks(request)`（只读探测，跨 hash group）；`allocate_slots` 接受 `retention_resolver` 并包装为 `BlockRetentionHint` 下传；保证"先 touch 本地命中、再分配外部块"的两阶段顺序。 |
| `vllm/v1/core/kv_cache_coordinator.py` | 透传 retention hint 到各 single-type manager。 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `allocate_new_blocks` / `allocate_external_computed_blocks` 增加 `retention_hint`；`MambaManager`（含 align 模式分支）与 `CrossAttentionManager` 签名对齐。 |
| `vllm/v1/core/sched/scheduler.py` | 新增 `_iter_waiting_demand_candidates(window)`（有界、不改队列）与 `_make_waiting_demand_retention_resolver(exclude)`（策略门控 + 惰性闭包）；在调度路径把 resolver 传入 `allocate_slots`。 |
| `vllm/v1/core/sched/request_queue.py` | `RequestQueue` / `PriorityRequestQueue` 新增 `peek_n(n)`（FCFS 直接切片；Priority 用冻结 key 的 heap 扫描），保证 peek 顺序 == pop 顺序。 |
| `vllm/config/scheduler.py` | 新增 `prefix_cache_eviction_policy`（`Literal["lru","waiting_queue_aware"]`）与 `kv_aware_candidate_window` 字段及校验。 |
| `vllm/engine/arg_utils.py` | 暴露 `--prefix-cache-eviction-policy`、`--kv-aware-candidate-window` 到 EngineArgs/CLI。 |
| `tests/v1/core/utils.py` | `create_scheduler` 支持两个新参数（此处曾漏签名导致 NameError，已修）。 |
| `tests/v1/core/test_prefix_caching.py` | 新增 11 条用例：软保留淘汰顺序、fast path、只读 peek、跨 group 并集、hint 为 None 兼容、official accounting 等。 |
| `tests/v1/core/test_scheduler.py` | 新增 4 条用例：真实 Scheduler 下保留近头部命中块（vs LRU 基线淘汰）、`peek_n` 与 pop 顺序一致性等。 |
| `Documents_xxy/stress_bench.py` | 负载参数化：`BENCH_PREFIX_POOLS` / `BENCH_SHARED_RATIO` / `BENCH_TOTAL_REQUESTS` / `BENCH_POOL_MODE`（random/cyclic）/ `BENCH_GROUP_SIZE` 环境变量。 |

关键设计（为什么这样做）：

- **惰性 resolver**：retained 集合在每次 `allocate_slots` 时才计算，且只有
  free 队头存在 cached block 时才会触发——避免每次调度都扫描等待队列。
- **软保留而非硬锁定**：保留块仍可被回退使用，不会造成死锁或分配失败；
  最坏情况退化为 LRU。
- **默认零侵入**：`prefix_cache_eviction_policy="lru"` 时 resolver 为 `None`，
  走原始代码路径。

## 4. 验证结果（Gate 2）

环境：`/root/miniconda3/envs/vllm-dev/bin/python`，torch `2.13.0+cu130`，
editable vLLM（遵守本机 LOCAL OVERRIDE：未用 uv / pip）。

| 验证项 | 结果 |
|---|---|
| `tests/v1/core/test_prefix_caching.py` | **102 passed**（含 11 条新增） |
| `tests/v1/core/test_scheduler.py` | **158 passed, 1 failed**；failed 为 `test_async_scheduling_pp...`，需 2 卡，本机单卡属环境性失败 |
| 其他 v1 core 测试（`test_preemption_policy.py` 等） | **422 passed, 2 errors**；errors 同为多卡/环境相关 |
| `ruff check` / `ruff format --check`（改动文件） | 通过 |
| `compileall`（scheduler、config 等） | 通过 |

过程中修复的问题：`MambaManager`/`CrossAttentionManager` 签名缺 `retention_hint`；
`create_scheduler` 漏参数导致 `NameError`；测试断言误用 `request_id`（应为
`req_id`）；hit token 统计混入 scheduled tokens 导致断言错；ruff import 顺序
与未用变量。

## 5. 4090 压测

### 5.1 环境与参数

模型本地缓存的 `Qwen/Qwen2.5-7B-Instruct`，RTX 4090 单卡。所有主场景 server
共用（等价命令）：

```bash
/root/miniconda3/envs/vllm-dev/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.75 \
  --num-gpu-blocks-override 1250 --enable-prefix-caching \
  --preemption-policy <default|recompute_aware> \
  --prefix-cache-eviction-policy <lru|waiting_queue_aware> \
  > Documents_xxy/server_X.log 2>&1
```

压测端（`Documents_xxy/stress_bench.py`，输出路径为第一个位置参数）：

```bash
BENCH_PREFIX_POOLS=8 BENCH_SHARED_RATIO=0.7 BENCH_TOTAL_REQUESTS=192 \
  /root/miniconda3/envs/vllm-dev/bin/python Documents_xxy/stress_bench.py \
  Documents_xxy/stress_bench_phase2_c_aware.json
```

固定项：并发 48（`BENCH_CONCURRENCY`）、每个共享前缀约 900 token、输出长度
混合 `[512, 1024, 1536]`、seed 42、`BENCH_POOL_MODE=random`。
preemptions / prefix hits / queries 为 server `/metrics` 累计计数器在压测窗口内的 **delta**。
指标口径注意（沿 Phase 1 报告）：`n_tokens` 统计的是 SSE content chunk
非 tokenizer token；"ITL p99" 是请求级 p99 ITL 的跨请求 p99。

### 5.2 主场景 A/B/C（同 seed 同负载单次对照）

负载：48 并发、192 请求、8 前缀池、70% 共享比例、random 模式。

| 指标 | A: default+LRU | B: recompute_aware+LRU | C: B+waiting_queue_aware | C vs B |
|---|---:|---:|---:|---:|
| 墙钟时间 | 59.45 s | 30.33 s | 34.53 s | **+13.9%** |
| 请求吞吐 | 3.229 req/s | 6.330 req/s | 5.560 req/s | -12.2% |
| 输出吞吐 | 934.3 chunk/s | 1717.3 chunk/s | 1566.9 chunk/s | -8.8% |
| TTFT mean | 18.89 s | 8.92 s | 9.29 s | +4.1% |
| TTFT p99 | 42.05 s | 21.12 s | 22.47 s | +6.4% |
| 请求级 mean ITL 均值 | 39.5 ms | 22.0 ms | 22.6 ms | +2.7% |
| 请求级 p99 ITL 跨请求 p99 | 8.514 s | 0.305 s | 2.129 s | **+598%** |
| scheduler preemptions | 1039 | 78 | 101 | +29.5% |
| prefix hits / queries | 293248 / 305134 | 141152 / 154312 | 141152 / 154312 | 持平 |

结论：**C 相对 B 无收益，多数指标反向**。B 相对 A 的大幅改善
（-49% 墙钟、+96% 吞吐、ITL p99 -96%）是 Phase 1 抢占策略的收益，
不是本阶段的。三次运行均 192/192 成功、0 错误。

### 5.3 与 Phase 1 报告的不一致（重要，待复核）

Phase 1（Codex 测）与本阶段（GLM 测）在**相同 server flags、相同压测配置**下：

| 运行 | 代码 | preemption policy | 墙钟 | preemptions |
|---|---|---|---:|---:|
| phase1_default（Codex） | Phase 1 | default | 32.12 s | 74 |
| baseline / default_run2（早期运行） | 更早 | default | 31.61 / 31.54 s | 135 / 135 |
| **phase2_a_default（本阶段）** | Phase 2 | default | **59.45 s** | **1039** |
| phase1_recompute（Codex） | Phase 1 | recompute_aware | 30.15 s | 79 |
| phase2_b_recompute（本阶段） | Phase 2 | recompute_aware | 30.33 s | 78 |

- recompute 路径前后几乎一致（30.15→30.33 s，79→78 preemptions），
  说明 Phase 2 改动在 recompute+LRU 路径上无回归。
- **A 运行是唯一异常点**：1039 次抢占、ITL p99 8.5 s、hits/queries 翻倍
  （抢占→resume→重查 prefix cache 的连锁效应），呈典型 default 策略
  thrash 形态。代码层面 default 路径的 resolver 为 `None`、走原始分支，
  理论上行为不变。
- 两个假设：①（更可能）default 抢占策略在该压力点存在**双稳态/抖动**，
  单次运行落入哪个态受时序噪声影响——早期两次 default 也出现过 74 vs 135
  的波动，本次落到极端态；② Phase 2 改动以未知方式影响了 default 路径。
- **给 Codex 的复核建议**：default 与 recompute 各 ≥3 次重复、交替顺序运行
  主场景；若 default 仍复现 1000+ preemptions，用 `git stash` Phase 2 diff
  在 Phase 1 代码上跑同配置做二分定位。

### 5.4 扩展负载（均为单次运行，供参考）

| 场景 | 负载参数 | LRU（B 侧） | aware（C 侧） | 结论 |
|---|---|---|---|---|
| 高压随机 hp | 64 池、90% 共享、192 请求、random | 96.94 s / 75 preempt / ITL p99 1.493 s / hits 52928 | 99.26 s / 81 / **0.742 s** / hits 50896 | ITL p99 减半是唯一正向信号；墙钟 +2.4%、hits -3.8% 反向 |
| 循环扫描 cyc | 16 池、90% 共享、384 请求、`BENCH_POOL_MODE=cyclic`、`BENCH_GROUP_SIZE=8` | 83.55 s / 226 preempt / hits 380080 | 82.69 s / 277 / hits 380080 | 墙钟 -1.0%、preempt +22.6%、hits 完全相同，基本持平 |
| 槽位受限 slot | 16 池、90%、384 请求、`--num-gpu-blocks-override 2000 --max-num-seqs 16` | 173.75 s / 0 preempt | 173.46 s / 0 preempt | 对照实验：无 KV 压力时保留机制为 no-op，符合预期 |

（hp 场景即 LRU 的经典失效负载——某池再次被需要时恰在 LRU 队尾——
aware 在该场景 hits 反而略降，说明软保留的重排没有转化为命中。）

### 5.5 诊断：为什么 Phase 2 没有收益

用临时插桩（`retention-dbg` 日志，**已从当前代码移除**，仅存于 dbg 日志）
得到的事实：

1. **多数分配时刻无可保留对象**：dbg2 运行中 8784 次 resolver 调用里
   3881 次（44%）candidates=0——等待队列为空或候选均已有 computed tokens。
2. **retained 非零时也改变不了命中**：典型 retained 集合为 2 或 69 块，
   但 B/C 两列 hits 逐 token 相同（141152 / 380080）。机制解释：
   v1 调度序（prefill 优先、分配前先 touch 本地命中）下，被保留的块在
   候选请求真正被调度时，要么不在 free queue 头部扫描窗口内
   （`popleft_n_avoiding` 触不到），要么其前缀块已被更早的分配消费。
   软保留只是**重排淘汰顺序**，不**阻止**淘汰。
3. **副作用**：跳过 retained 块让可用块更紧，preempt 略增（101 vs 78；
   277 vs 226），并解释了 C 的 ITL p99 恶化。

## 6. 产物文件清单（`Documents_xxy/`）

命名规则：`stress_bench_*.json` = 压测结果（`summary` + 每请求 `detail`）；
`server_*.log` = 对应 vLLM server 完整日志（含启动 flags，见各 log 第 11 行
`non-default args`）。**JSON 与 log 按后缀成对**。

### Phase 2 主场景（本次，GLM）

| JSON | server log | 配置 |
|---|---|---|
| `stress_bench_phase2_a_default.json` | `server_a.log` | default 抢占 + LRU 淘汰；8 池/0.7/192 请求。**异常运行，见 §5.3** |
| `stress_bench_phase2_b_recompute.json` | `server_b.log` | recompute_aware + LRU；同上负载 |
| `stress_bench_phase2_c_aware.json` | `server_c.log` | recompute_aware + waiting_queue_aware；同上负载 |

### Phase 2 扩展负载（本次，GLM）

| JSON | server log | 配置 |
|---|---|---|
| `stress_bench_hp_b1_lru.json` | `server_b1.log` | 高压随机：64 池/0.9/192 请求，recompute+LRU |
| `stress_bench_hp_c1_aware.json` | `server_c1.log` | 同上，+waiting_queue_aware |
| `stress_bench_cyc_b1_lru.json` | `server_cyc_b1.log` | 循环扫描：16 池/0.9/384 请求/cyclic/GROUP=8，recompute+LRU |
| `stress_bench_cyc_c1_aware.json` | `server_cyc_c1.log` | 同上，+waiting_queue_aware |
| `stress_bench_slot_b_lru.json` | `server_slot_b.log` | 槽位受限：2000 blocks + max_num_seqs 16，recompute+LRU，0 抢占 |
| `stress_bench_slot_c_aware.json` | `server_slot_c.log` | 同上，+waiting_queue_aware。注意：此 run 带 dbg 插桩（log 内有 16018 行 retention-dbg），B 侧无，对比仅供参考 |

### 诊断运行（本次，GLM）

| JSON | server log | 配置 |
|---|---|---|
| `stress_bench_dbg.json` | `server_dbg.log` | 16 池/0.9/96 请求，aware + retention-dbg 插桩（3226 行诊断日志） |
| `stress_bench_dbg2.json` | `server_dbg2.log` | 16 池/0.9/384 请求，aware + 插桩（19745 行）。§5.5 的统计来源 |

### Phase 1 及更早（历史留存，供对照）

| JSON | server log | 配置 |
|---|---|---|
| `stress_bench_phase1_default.json` | （未留存） | Phase 1 验收运行（Codex）：default+LRU，32.12 s / 74 preempt |
| `stress_bench_phase1_recompute_aware.json` | （未留存） | Phase 1 验收运行（Codex）：recompute+LRU，30.15 s / 79 preempt |
| `stress_bench_baseline.json` / `stress_bench_default_run2.json` / `stress_bench_recompute_v2.json` | （未留存） | 更早探索运行（default×2 / recompute×1），用于观察 default 波动性 |

其他文件：`IMPLEMENTATION_PLAN.md`（三阶段总计划）、
`PHASE1_IMPLEMENTATION_REPORT.md`（Phase 1 报告）、`stress_bench.py`
（压测脚本本体）、其余 `*.md` 为项目设计文档，与本次运行无关。

## 7. 结论

- **Correctness 目标达成**：默认 LRU 路径零行为变化（有单测锁定）；
  waiting_queue_aware 的软保留语义、只读 peek、队列不变量均有覆盖；
  lint / 编译 / 编译期检查全部通过。
- **性能目标未达成**：在 random / cyclic / 高压 / 槽位受限四类负载下，
  waiting_queue_aware 相对 LRU 无净收益（最好情况持平，主场景 +13.9% 墙钟）。
  根因是软保留只重排淘汰顺序，而 v1 调度时序使保留块很少处于
  "即将被淘汰且候选随后命中"的位置。
- 本阶段压测中观察到的 B vs A 巨大差距全部来自 Phase 1 的 recompute_aware
  抢占策略；A 运行自身的异常（1039 preempt）需按 §5.3 复核后方可用作基线。

## 8. 下一步（供 Codex / 后续会话）

1. **复核 A 异常**：default vs recompute 各 ≥3 次交替重复；必要时 stash
   Phase 2 diff 二分。
2. 若确认 Phase 2 无收益的机制诊断，改进方向二选一：
   - 把保留信息用于 **evictable 排序**（touch/LRU 顺序）而非分配时跳过，
     使保留块在被淘汰前有更长的存活窗口；
   - 直接进入 **Phase 3**（等待队列准入控制 + aging），按
     `IMPLEMENTATION_PLAN.md` 推进。
3. 性能 Gate 沿用 Phase 1 报告的建议：先完善指标（真实 token 计数）、
   多轮 paired run，再以 `vllm bench serve` 作为对外可比较的主指标。
