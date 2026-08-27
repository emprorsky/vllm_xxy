# vLLM V1 Adaptive KV-Aware Serving 项目 Coding Agent 完整提示词

你现在要协助我完成一个基于 **最新 vLLM V1 源码** 的推理引擎工程项目。

这是一个用于秋招推理引擎 / LLM Serving / Inference Engine 岗位的核心项目，因此目标不是简单“能跑”，而是：

- 改动真实 vLLM 核心路径；
- 设计必须有工程价值；
- correctness 必须可靠；
- 测试必须完整；
- Git history 必须清晰；
- 后续 benchmark 必须能真实说明优化效果；
- 所有设计我本人后续都需要能够理解并在面试中完整讲清楚。

---

# 一、你必须先阅读的两份设计文档

仓库中会提供两份设计文档：

```text
vllm_kv_aware_project_design.md
vllm_kv_aware_project_advanced_design.md
```

它们的关系是：

```text
vllm_kv_aware_project_design.md
=
基础版
=
第一阶段必须完成的强项目版本

vllm_kv_aware_project_advanced_design.md
=
增量进阶版
=
在基础版架构上继续深化
```

**进阶版不是推翻基础版。**

你的任务是：

> 同时阅读两份文档，并结合我当前实际冻结的 vLLM commit，设计一套可以从基础版自然增量到进阶版的统一实现方案。

注意：

- 不允许机械照抄设计文档中的类名、函数名；
- 必须以当前仓库实际源码为准；
- 如果设计文档和当前源码不一致，明确指出；
- 不允许为了匹配文档而更新 vLLM upstream；
- 不允许擅自升级/降级当前 base commit。

---

# 二、当前项目总体目标

最终项目目标是：

## Adaptive KV-Value-Aware Scheduling and Cache Management for vLLM V1

中文：

## vLLM V1 自适应 KV Cache 感知调度与缓存管理优化

项目最终希望覆盖三层：

```text
Control Plane
├── Admission
├── Priority
├── Aging
├── Re-admission
└── Preemption

Memory Plane
├── Prefix Cache
├── KV Retention
├── Reclaimable KV
└── Cache Eviction

Performance / Data Plane
├── Scheduler CPU Hot Path
└── Optional KV Cache Write Kernel
```

---

# 三、项目资源和现实约束

当前资源：

```text
GPU:
单卡 RTX 4090 24GB

开发：
AutoDL Linux
GitHub fork
vLLM latest main base commit

主要开发语言：
Python

主要目标代码：
vLLM V1 Scheduler / KVCacheManager / BlockPool
```

现实约束：

- 我需要尽快投秋招；
- 基础代码改造需要几天内形成可运行版本；
- 整体项目可以之后继续深化；
- 不做科研数据集；
- 不重新设计 benchmark framework；
- 尽量使用 vLLM repo 原有 pytest / benchmark；
- 不做多 GPU 系统；
- 第一阶段不改 CUDA/C++；
- 不为了“功能多”乱加不相关 feature；
- 普通新模型适配不是当前优先级。

---

# 四、非常重要：你现在不要直接改代码

你的第一阶段任务只有：

```text
阅读设计
+
阅读当前源码
+
生成统一 IMPLEMENTATION_PLAN.md
```

在我确认 `IMPLEMENTATION_PLAN.md` 之前：

**禁止修改任何业务代码。**

允许做：

- 查看源码；
- grep/ripgrep；
- 查看 git history；
- 阅读 tests；
- 阅读 config；
- 阅读 docs；
- 阅读当前项目已有 benchmark；
- 运行只读检查；
- 运行 baseline tests；
- 做代码调用链分析。

不允许：

- 修改 scheduler；
- 修改 KV cache manager；
- 修改 BlockPool；
- 新增策略；
- 写 kernel；
- 大规模重构；
- 自动提交 commit。

---

# 五、第一步：检查 Git / Base Commit

首先执行并记录：

```bash
git status
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
git remote -v
```

必须确认：

1. 当前 branch；
2. 当前 base commit；
3. working tree 是否 clean；
4. `origin` 是否为我的 fork；
5. `upstream` 是否为官方 vllm-project/vllm。

如果工作区不干净：

> 立即停止，不要 reset / restore / stash，先告诉我有哪些修改。

不要未经允许处理我的未提交代码。

---

# 六、必须完整阅读当前真实代码路径

至少阅读并建立调用关系：

```text
vllm/v1/core/sched/scheduler.py
vllm/v1/core/sched/request_queue.py
vllm/v1/request.py

vllm/v1/core/kv_cache_manager.py
vllm/v1/core/block_pool.py
vllm/v1/core/kv_cache_utils.py

vllm/config/scheduler.py
```

以及与它们直接相关的：

```text
KV cache coordinator
request status
scheduler output
prefix caching
hybrid KV manager
KV connector
```

实际文件名以当前 commit 为准。

---

# 七、你必须回答清楚当前 Scheduler 的真实运行链

不要只说“Scheduler 调度 request”。

需要精确分析：

```text
Scheduler.schedule()
    ↓
running requests 怎么处理？
    ↓
waiting requests 怎么处理？
    ↓
token budget 怎么控制？
    ↓
prefix cache 在哪个阶段查询？
    ↓
KVCacheManager.allocate_slots()
    ↓
什么时候返回 None？
    ↓
Scheduler 如何决定 preemption？
    ↓
victim 如何选择？
    ↓
_preempt_request 做了什么？
    ↓
RequestStatus 怎么改变？
    ↓
num_preemptions 在哪里更新？
    ↓
request 怎样重新回 waiting queue？
    ↓
下一轮怎样 re-admission？
```

请精确到：

```text
文件
类
函数
关键代码区域
状态变化
```

---

# 八、必须分析 RequestQueue

需要确认当前：

```text
FCFS
Priority
```

具体怎么实现。

包括：

- deque / heap / custom queue？
- `peek()` / `pop()` 怎么工作？
- iterator 是否等于真实 scheduling order？
- heap 内部 array 顺序能不能直接拿来当 top-K？
- `Request.__lt__` 如何比较？
- priority 数值越大还是越小优先？
- arrival time 的 tie-break？
- request id 的 tie-break？
- `num_preemptions` 当前是否参与排序？
- request 在 heap 内部时修改比较字段会不会破坏 heap invariant？

这一部分必须非常严谨。

---

# 九、必须分析 KVCacheManager

重点理解：

```text
get_computed_blocks(...)
allocate_slots(...)
free(...)
get_blocks(...)
get_block_ids(...)
```

以及当前实际版本对应的方法。

重点回答：

### 1. allocation 为什么失败？

```text
需要多少 block？
目前多少 free block？
watermark 有什么作用？
cached blocks 算 free 吗？
hybrid group 怎么算？
```

### 2. preemption 之后 KV 怎么处理？

需要区分：

```text
active referenced block
free block
free-but-cached block
shared prefix block
```

### 3. Request free 后是否一定能真正释放所有 block？

不能猜。

要阅读：

```text
ref_cnt
shared block
prefix cache
block pool
```

真实行为。

---

# 十、必须分析 BlockPool / Prefix Cache

重点回答：

```text
BlockPool 如何初始化 physical block？
free_block_queue 的真实语义是什么？
cached block 为什么可能 ref_cnt == 0？
cached block 什么时候还留在 hash table？
真正 eviction 发生在哪里？
LRU 顺序如何维护？
get_new_blocks() 怎么选择 block？
free_blocks() 怎么重新插队？
_maybe_evict_cached_block() 做什么？
```

需要特别分析：

```text
prefix cache lookup 是否有副作用？
```

例如是否会：

- 修改 LRU；
- 修改 refcnt；
- 产生 event；
- 更新 metrics；
- 触发 KV connector；
- 影响 hash state。

这是后续 read-only prefix probe 是否能复用现有函数的关键。

---

# 十一、基础版必须最终实现的内容

基础版 P0：

## 1. Lightweight Scheduler Policy Abstraction

目的：

> 将“策略决定选谁”与 Scheduler correctness-critical 状态修改分离。

至少未来能自然支持：

```text
Admission Decision
Preemption Decision
Re-admission Ordering
Cache Retention Hint
```

但：

**不要实现完整 Plugin Framework。**

禁止过度设计：

```text
dynamic plugin loader
entry points
registry framework
callback lifecycle framework
out-of-tree plugins
```

只做当前项目需要的最小抽象。

---

## 2. Recompute-Aware Preemption

第一版：

```text
recompute_cost(request)
≈
request.num_computed_tokens
```

严格：

```text
User Priority
>
Recompute Cost
```

也就是：

```text
先选择最差 user priority tier
再在该 tier 内选低 recompute cost victim
```

不能让低优先级 request 因 recompute cost 高而保护自己、反过来抢高优先级 request。

---

## 3. Resume-Aware Re-admission

同一 user priority tier：

```text
preempted/resume request
>
cold never-run request
```

但不能跨越 user priority。

需要认真设计：

```text
priority
resume state
arrival time
request id
```

之间的排序。

不能简单创建一个完全绕过 Priority semantics 的 preempted queue。

---

## 4. Waiting-Queue-Informed Prefix Cache Retention

利用：

```text
waiting queue near-head requests
```

作为：

```text
near-future prefix demand
```

调整 cached free block eviction preference。

例如：

```text
Original LRU:
A B C D E

Retained:
B D

Temporary preference:
A C E B D
```

但：

```text
B/D 不是 pinned
```

内存仍不够：

```text
最终允许淘汰 B/D
```

必须保证：

> 新策略不会让一个原来可成功的 allocation 因 retention 而失败。

---

# 十二、进阶版后续目标

这些不是第一批同时实现，但基础架构必须考虑未来可以自然增量。

## 1. Cache-Affinity Admission

例如：

```text
A:
prompt = 4000
cached prefix = 0

B:
prompt = 4000
cached prefix = 3500
```

则：

```text
remaining prefill:
A = 4000
B = 500
```

可以在有限 candidate window 内优先考虑 B。

目的：

> 尽快兑现已有 Prefix Cache 价值，减少实际 prefill compute。

## 2. Bounded Candidate Window

不能全队列排序。

只允许在：

```text
waiting queue 前 W 个候选
```

做 cache-aware reordering。

W 具体值由实现和测试决定。

目的是：

- 防 starvation；
- 控制 CPU overhead；
- 保持 scheduling semantics 可解释。

## 3. Aging / Anti-Starvation

不能：

```text
warm request 永远压 cold request
```

需要设计 aging guard。

第一版优先：

```text
tier / lexicographic rule
```

不要先写拍脑袋加权公式。

## 4. Lazy Per-Iteration SchedulingFeatures

最终多个策略会需要：

```text
priority
computed_tokens
num_preemptions
cached_prefix_tokens
remaining_prefill_tokens
waiting_time
reclaimable_blocks
```

不能每个策略重复进行 expensive KV lookup。

希望最终有类似：

```text
RequestSchedulingFeatures
```

特点：

```text
per-iteration
read-only
lazy
bounded
no duplicated source of truth
```

cheap fields：

```text
priority
computed_tokens
arrival time
num_preemptions
```

expensive fields：

```text
cached prefix
reclaimable KV
```

只在真正需要时 resolve。

## 5. Priority + Demand-Aware Prefix Retention

基础版：

```text
near-head request needs block
→ retain later
```

进阶版：

```text
near-term demand
+
request priority
+
resume state
+
saved compute
```

共同决定 KV block value。

优先考虑 tier：

```text
Tier 0 普通 cached
Tier 1 near-head reuse
Tier 2 near-head resumed
Tier 3 near-head high-priority
```

组内仍保持 LRU。

## 6. Reclaimable-KV-Aware Preemption

需要区分：

```text
allocated blocks
!=
真正抢占后可释放 blocks
```

shared prefix/refcount 会影响 reclaimability。

未来可能增加：

```python
get_reclaimable_block_count(request)
```

只能是 read-only estimate。

高级 victim 逻辑推荐：

```text
1. user priority tier
2. reclaimability feasibility
3. minimize recompute cost
4. stable tie-break
```

不要第一版直接拍一个：

```text
reclaimable / recompute
```

比例公式。

---

# 十三、为什么不要设计万能数学 score

禁止在没有数据依据时做：

```text
score =
0.31 * cache
+ 0.27 * age
+ 0.18 * recompute
...
```

优先：

- hierarchy
- tier
- lexicographic ordering
- simple cost proxy
- bounded candidate selection

原因：

- 更容易解释；
- 更容易测试；
- 不需要训练；
- 不需要 tuning dataset；
- 更适合 Scheduler hot path。

---

# 十四、Scheduler CPU Hot Path

进阶版还必须考虑：

> 新增智能策略不能把 Scheduler 自己拖慢。

因此统一 Plan 中需要分析：

```text
当前 Scheduler CPU 热点
```

以及新增设计可能增加：

```text
prefix lookup
queue scan
block scan
temporary allocation
sorting
```

的成本。

需要考虑：

```text
bounded candidate
lazy feature
default fast path
duplicate lookup elimination
```

---

# 十五、GPU Kernel 方向：现在只调查，不实现

需要调查：

```text
reshape_and_cache
KV cache write path
```

原因：

它与项目主线直接相关：

```text
Scheduler
→ KV allocation
→ slot mapping
→ physical KV block
→ K/V write
```

但目前只做 reconnaissance。

---

# 十六、Kernel 调查必须回答

请找到：

```text
reshape_and_cache
```

或当前 base commit 对应的 KV cache write kernel。

分析：

- Python dispatch 在哪里？
- CUDA/Triton 实现在哪里？
- 是否存在多 backend？
- 4090 SM89 走哪一个？
- 当前支持 dtype？
- head_dim/block_size specialization？
- 是否有 vectorized load/store？
- 是否有现成 benchmark？

重点搜索：

```text
benchmarks/kernels/
```

找到当前真实 benchmark。

---

# 十七、Kernel 禁止事项

当前阶段禁止：

- 重写 GEMM；
- 重写 FlashAttention；
- 重写 PagedAttention；
- 做 MoE fused GEMM；
- 直接提交新 Triton kernel；
- 未 benchmark 就声称优化。

未来只有在：

```text
4090 existing benchmark
```

发现真实可重复性能 gap 后，才考虑：

```text
vectorized memory access
address calculation specialization
SM89 common-shape specialization
Triton launch config
```

---

# 十八、你生成的 IMPLEMENTATION_PLAN.md 必须包含以下结构

## A. Repository State

```text
branch
commit hash
git status
remote
vLLM version metadata
```

## B. Current Architecture Map

精确到：

```text
文件
类
函数
调用关系
状态变化
```

必须覆盖：

```text
Scheduler.schedule
waiting admission
running scheduling
KV allocation
allocation failure
preemption
request status
re-admission
prefix lookup
block allocation
cached eviction
```

## C. Request Queue Semantics

说明：

```text
FCFS
Priority
heap/deque
top-K 如何安全获取
comparison key
mutation invariant
```

## D. KV Cache Lifecycle

画清楚：

```text
allocate
reference
share
free
cached-free
evict
reuse
```

## E. Gap Analysis

逐条：

```text
设计文档假设
vs
当前 base commit 实际情况
```

如果不一致必须明确。

## F. Unified Architecture Proposal

设计：

```text
Policy
Feature access
Admission
Preemption
Re-admission
Retention
```

如何组合。

要求：

> 基础阶段接口不要阻碍进阶阶段。

但不得设计完整 plugin framework。

## G. Exact Phase 1 Patch

精确到：

```text
文件
类
函数
新增参数
返回值
配置
测试
```

只描述，不生成完整实现。

## H. Exact Phase 2 Patch

Queue-aware Prefix Retention。

同样精确到函数。

## I. Phase 3+ Integration Points

只指出：

```text
Cache-Affinity Admission
Aging
Lazy SchedulingFeatures
Priority Retention
Reclaimable KV
```

未来挂在哪里。

此时不实现。

## J. Feature Cost Map

至少输出表格：

```text
Feature
Source
Complexity
Side Effects
Cheap/Expensive
Lazy?
```

例如：

```text
priority
computed_tokens
num_preemptions
cached_prefix_tokens
remaining_prefill
waiting age
reclaimable blocks
```

复杂度必须以源码真实实现分析。

## K. Correctness Invariants

详细说明如何保护：

```text
user priority
heap invariant
request status
KV refcount
shared prefix
block hash
free queue order
prefix event
metrics
fallback allocation
```

## L. Test Plan

每个 test 必须写：

```text
test file
test name
initial state
operation
expected result
```

---

# 十九、至少需要考虑这些 Scheduler Tests

## Default compatibility

新 feature 关闭时行为与当前 base commit 一致。

## Strict priority

高 user priority 不能因为其他 KV heuristic 被抢。

## Recompute victim

同 priority：

```text
A computed=128
B computed=2048
```

优化策略应优先抢 A。

## Re-admission

同 priority：

```text
preempted
>
cold
```

## Heap invariant

修改 `num_preemptions` 等状态后，Priority heap 仍正确。

---

# 二十、至少需要考虑这些 Prefix Cache Tests

## Retained ordering

```text
LRU:
A B C D E

retained:
B D

eviction:
A C E B D
```

## Fallback

non-retained 不够时 retained 必须仍可淘汰。

## Read-only probe

prefix probe 不改变：

```text
refcount
LRU
hash
metrics
event
request state
```

## Fast path

有足够普通 free blocks 时：

```text
不要做 waiting prefix scan
```

---

# 二十一、进阶阶段未来需要的 Tests

Plan 里先预留。

## Cache-Affinity

warm request 在 bounded candidate window 内获得合理排序。

## Bounded

window 外请求不能无限跨越前排。

## Aging

持续到来的 warm request 不能让 cold request 永久 starvation。

## Lazy Feature

不用 cached prefix 时：

```text
resolver 不被调用
```

同一 iteration：

```text
expensive feature 最多 resolve 一次
```

## Reclaimable KV

shared prefix 下：

```text
allocated block count
>
reclaimable block count
```

正确估计。

---

# 二十二、配置设计

需要能够：

```text
baseline
vs
optimized
```

不要只有全局 hard-coded 行为。

但配置也不能爆炸。

请提出最小 typed configuration。

优先使用当前 vLLM 配置体系。

不要为了方便加大量环境变量。

---

# 二十三、Backward Compatibility

默认配置下：

> 尽量保持 vLLL 原行为。

新策略应该是：

```text
opt-in
```

至少项目开发阶段如此。

这样：

- baseline 清楚；
- regression 容易定位；
- upstream PR 更容易 review；
- benchmark 能公平对比。

---

# 二十四、性能验证

不开发新数据集。

优先使用当前 repo 已有：

```text
scheduler tests
prefix caching tests
benchmark_prefix_caching
vllm bench serve
prefix_repetition
kernel benchmarks
```

实际命令必须由你基于当前 repo 确认。

---

# 二十五、4090 Benchmark 原则

目标不是用巨大模型撑满 24GB。

而是：

```text
正常 3B / 7B 模型
+
限制 KV cache budget
+
较高 concurrency
+
较长 prompt
+
prefix repetition
```

主动制造：

```text
KV pressure
```

触发：

- preemption
- re-admission
- cached eviction
- cache affinity

---

# 二十六、最终关注指标

Serving：

```text
throughput
TTFT
TPOT / ITL
```

Work efficiency：

```text
prefix cache hit
recomputed tokens
preemption count
resume delay
cached block eviction
```

Scheduler overhead：

```text
decision latency
prefix lookup count
candidate count
expensive feature resolve count
```

不要为了获取某个指标写大型 instrumentation。

---

# 二十七、Git 实施原则

代码阶段必须：

```text
one logical change
→ test
→ diff review
→ commit
→ push
```

禁止一次 Agent 自动改完全部阶段。

推荐 commit：

```text
feat(scheduler): add lightweight scheduling decision policy

feat(scheduler): add recompute-aware preemption selection

feat(scheduler): prioritize preempted requests on re-admission

feat(kv-cache): add waiting-queue-informed prefix retention

feat(scheduler): add bounded cache-affinity admission

feat(scheduler): add aging guard for cache-aware ordering

refactor(scheduler): add lazy per-iteration scheduling features

feat(kv-cache): weight prefix retention by request demand

feat(kv-cache): expose reclaimable block estimate

feat(scheduler): use reclaimable KV in preemption decisions
```

性能 commit 必须有 benchmark 证据才保留。

---

# 二十八、严格禁止 scope creep

除非我明确要求，否则不要做：

```text
完整 Scheduler Plugin Framework
MLFQ
SJF/EWSJF
VTC fairness
DistServe
P/D disaggregation
multi-GPU routing
TP/PP modifications
KV CPU offload
KV compression
speculative decoding algorithm
new model adaptation
dashboard
dataset
new load generator
FlashAttention rewrite
PagedAttention rewrite
GEMM rewrite
MoE kernel rewrite
```

---

# 二十九、Stop Rules

出现以下情况，必须停止扩展并报告：

- 需要大改 vLLM request state machine；
- Phase 1 correctness 还没稳定；
- core patch 预计明显超过约 600 行却只是基础功能；
- 为 Cache-Affinity 需要扫描整个 waiting queue；
- prefix lookup 无法实现可靠 read-only；
- reclaimable block 在当前 Hybrid KV Manager 下无法可靠定义；
- kernel benchmark 没有真实提升；
- default path 出现明显性能退化。

宁愿缩 scope，不要堆一个难以维护的项目。

---

# 三十、你生成 Plan 后先不要继续执行

最终输出：

```text
IMPLEMENTATION_PLAN.md
```

完成以后：

1. 给我总结 Plan；
2. 标出最高风险点；
3. 标出文档假设与源码不一致的地方；
4. 给出 Phase 1 预计修改文件和代码量；
5. 给出 Phase 1 预计测试范围；
6. 明确告诉我：

```text
“当前尚未修改业务代码，等待确认后进入 Phase 1。”
```

然后停止。

**不要自动开始编码。**

---

# 三十一、我最终期待的实施路线

```text
Phase 0
Unified Inspection & Plan
        ↓
Phase 1
Policy
+ Recompute Preemption
+ Resume
        ↓
Phase 2
Queue-Informed Prefix Retention
        ↓
—— 到这里已经形成基础版可投项目 ——
        ↓
Phase 3
Cache-Affinity Admission
+ Bounded Window
+ Aging
        ↓
Phase 4
Lazy SchedulingFeatures
        ↓
Phase 5
Priority/Demand Retention
+ Reclaimable KV
+ Advanced Preemption
        ↓
—— 到这里形成强力 Adaptive KV-aware Serving 项目 ——
        ↓
Phase 6
Scheduler CPU Performance
        ↓
Phase 7
Optional RTX4090 KV-write Kernel
```

代码必须按阶段推进。

---

# 三十二、最终设计原则

请始终遵循以下原则：

> **策略负责 decision，Scheduler Core 负责 correctness。**

> **User Priority 是硬约束，KV heuristic 只能在允许范围内优化。**

> **Future Cache Demand 是 hint，不是 hard pin。**

> **Read-only feature lookup 不得偷偷修改 KV/cache state。**

> **智能策略必须 bounded + lazy，不能让 Scheduler hot path 无限制变重。**

> **能用 tier / lexicographic rule 解决的，不要先造复杂数学 score。**

> **不重新发明 vLLM，只优化真实存在的决策缝隙。**

> **Kernel 是否更快只由真实 GPU benchmark 决定，不由代码看起来多高级决定。**

> **项目的价值来自 correctness、系统深度、真实性能证据和清晰工程设计，而不是 feature 数量。**
