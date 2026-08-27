# vLLM V1 Adaptive KV-Aware Serving：进阶版项目设计说明

> **定位**：本文是 `vllm_kv_aware_project_design.md` 的**增量进阶版**，不是替代基础版。
>
> 基础版解决：
>
> 1. Mini Scheduler Policy Abstraction
> 2. Recompute-Aware Preemption & Re-admission
> 3. Waiting-Queue-Informed Prefix Cache Eviction
>
> 本进阶版在其上继续向三个层次扩展：
>
> - **Control Plane**：Cache-Affinity Admission、Aging、统一 Scheduling Features
> - **Memory Plane**：Priority/Demand-Aware Retention、Reclaimable-KV-Aware Preemption
> - **Performance/Data Plane**：Scheduler CPU Hot Path、可选 KV Cache Write Kernel 优化
>
> 最终目标不是“堆更多 feature”，而是形成一个完整的：
>
> **Adaptive KV-Value-Aware Scheduling and Cache Management for vLLM V1**
>
> 即：根据请求已经完成的计算、当前可复用 Prefix、等待时间、用户优先级和可回收 KV Memory，对 admission、preemption、resume、cache eviction 做轻量协同，并控制这些策略本身给 Scheduler hot path 带来的开销。

---

# 0. 最重要的问题：基础版和进阶版怎么衔接？

## 0.1 不建议的方式

不建议：

```text
先完全实现基础版
    ↓
所有接口都写死
    ↓
确认完成
    ↓
再让 Agent 阅读进阶版
    ↓
重新改 Policy / Queue / Prefix lookup 接口
```

这样容易产生二次重构：

- 基础版 Policy 只考虑 preemption，进阶版 admission 又需要新接口；
- 基础版 Prefix Cache retention 每次自己查 prefix，进阶版才发现多个策略重复查询；
- 基础版 waiting ordering 写死，进阶版 Cache-Affinity/Aging 又得重写；
- 基础版 SchedulerConfig 只有一两个 bool，后续逐渐膨胀。

---

## 0.2 也不建议直接一次性实现全部进阶功能

同样不要：

```text
Agent 读完本文
    ↓
一次修改 1500 行
    ↓
Policy + Admission + Eviction + Feature Cache + Kernel 全部一起上
```

这会导致：

- 很难确定哪个功能破坏 correctness；
- benchmark 不知道收益来自哪里；
- Git history 无法解释；
- Agent 更容易把 vLLM 状态机改坏；
- 一周内很可能无法收尾。

---

## 0.3 推荐方式：统一设计，分阶段落地

**现在就把基础版和进阶版一起给 Agent。**

Agent 第一阶段只做：

```text
BASE DESIGN
+
ADVANCED DESIGN
+
实际冻结的 vLLM commit
        ↓
生成统一 IMPLEMENTATION_PLAN.md
```

这个 Plan 必须考虑最终会加入：

- Admission policy
- Preemption policy
- Resume ordering
- Prefix retention
- Lazy SchedulingFeatures

所以基础接口不要设计成未来无法扩展的死胡同。

但是**代码落地仍然严格分阶段**：

```text
Phase 1：基础正确性
    ↓
Phase 2：Cache-Affinity Admission
    ↓
Phase 3：Lazy Scheduling Features
    ↓
Phase 4：更深 KV-aware 策略
    ↓
Phase 5：CPU Performance
    ↓
Phase 6：可选 GPU Kernel
```

原则：

> **架构一次看远一点，代码一次只走一步。**

---

# 1. 最终项目定位

推荐最终名称：

## English

**Adaptive KV-Value-Aware Scheduling and Cache Management for vLLM V1**

或更短：

**Adaptive KV-Aware Serving for vLLM V1**

## 中文

**vLLM V1 自适应 KV Cache 感知调度与缓存管理优化**

---

# 2. 为什么进阶版比基础版更有含金量？

基础版解决的是一个完整但相对局部的问题：

```text
KV pressure
→ 抢谁
→ 被抢请求怎么恢复
→ 哪些 Prefix 不要过早淘汰
```

进阶版把问题提升为：

```text
在每一个 Scheduler iteration，
如何根据“请求价值”和“KV 价值”
统一决定：

1. 谁先 admission？
2. 内存不够抢谁？
3. 被抢请求何时恢复？
4. 哪些 cache 值得短期保留？
5. 这些决策自身最多允许花多少 CPU 时间？
```

最终形成三层：

```text
┌─────────────────────────────────────────────┐
│ Control Plane                               │
│ Admission / Priority / Aging / Preemption   │
├─────────────────────────────────────────────┤
│ Memory Plane                                │
│ Prefix Value / Retention / Reclaimable KV   │
├─────────────────────────────────────────────┤
│ Performance / Data Plane                    │
│ CPU Scheduler Hot Path / Optional KV Kernel │
└─────────────────────────────────────────────┘
```

这比“实现若干 heuristic”更像推理引擎系统设计。

---

# 3. 项目总架构

```text
                    Incoming / Waiting Requests
                              │
                              ▼
                 Bounded Candidate Window
                              │
                              ▼
                 Lazy Scheduling Features
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   User Priority        Cache Affinity         Waiting Age
        │                     │                     │
        ├─────────────────────┼─────────────────────┤
        │                     │                     │
        ▼                     ▼                     ▼
    Admission            Re-admission          Anti-starvation
      Policy                Policy
        │                     │
        └──────────┬──────────┘
                   ▼
              Scheduler Core
                   │
           KV allocation attempt
                   │
          ┌────────┴────────┐
          │                 │
        success           failure
          │                 │
          ▼                 ▼
      Execute        Preemption Policy
                          │
                ┌─────────┼──────────┐
                │         │          │
                ▼         ▼          ▼
           user QoS   recompute   reclaimable
                        cost          KV
                │         │          │
                └─────────┴──────────┘
                          │
                          ▼
                   selected victim
                          │
                          ▼
                    KVCacheManager
                          │
                          ▼
                       BlockPool
                          │
               cached-block pressure
                          │
                          ▼
                Prefix Retention Policy
                          │
              ┌───────────┼────────────┐
              ▼           ▼            ▼
          near-term    priority    saved compute
           demand
                          │
                          ▼
                 actual eviction order
```

---

# 4. 第一阶段：基础版仍然是地基

基础版三个方向仍然保留。

## 4.1 Mini Policy Abstraction

但进阶版要求它不要只抽：

```text
select_preemption_victim()
```

而要让最终结构至少能自然容纳：

```text
Admission Decision
Preemption Decision
Re-admission Ordering
Cache Retention Hint
```

**注意：不是现在就实现完整 plugin framework。**

可以是一两个简单的 strategy/helper 类，但命名和职责不能把未来 admission 扩展堵死。

---

## 4.2 Recompute-Aware Preemption

P0 仍使用：

```text
recompute_cost ≈ num_computed_tokens
```

并保持：

```text
User Priority
>
Recompute Cost
```

这是后续 Reclaimable-KV-Aware cost model 的第一版。

---

## 4.3 Resume-Aware Re-admission

同一 user-priority tier：

```text
preempted/resume
>
cold request
```

但不能跨越 user priority。

后续 Cache-Affinity Admission 会和这一规则组合，所以不要简单实现成一个独立“永远优先的 preempted deque”。

---

## 4.4 Queue-Informed Prefix Retention

基础版仍然保留：

```text
waiting queue near head
→ 近期可能需要的 prefix
→ cached block soft retention
```

必须：

- read-only probe
- retained != pinned
- capacity 不够时 fallback
- 正式 eviction accounting 仍走原路径

进阶版会在其上加入 Priority/Demand Value。

---

# 5. 进阶核心一：Cache-Affinity Admission

这是最应该加入的增量。

## 5.1 基础版只保护 cache，还没有利用 cache 决定“谁先上 GPU”

例如：

```text
Waiting A
prompt = 4000
cached prefix = 0

Waiting B
prompt = 4000
cached prefix = 3500
```

如果其他条件接近：

```text
A remaining prefill ≈ 4000
B remaining prefill ≈ 500
```

普通 FCFS 可能：

```text
A → B
```

Cache-affinity admission 可以考虑：

```text
B → A
```

原因不是单纯“B cache 多”，而是：

> B 可以用很少的新计算快速完成 prefill/admission，占用较少额外计算资源，并尽早兑现已经存在的 Prefix Cache 价值。

---

## 5.2 不能全局按 cached_tokens 排序

错误做法：

```text
waiting queue 所有请求
→ cached_tokens 最大的永远先跑
```

会造成 starvation。

例如：

```text
A cold request 已经等 10 秒

不断出现：
B1 warm
B2 warm
B3 warm
...
```

如果 warm 永远压过 cold，A 可能无限等待。

---

# 6. Bounded Cache-Affinity + Aging

推荐两层约束。

## 6.1 Bounded Candidate Window

只在 waiting queue 前：

```text
W = 8 / 16 / max_num_running_reqs
```

之类有限范围内做重排。

具体 W 不提前锁死，由 base commit 和测试决定。

作用：

- 避免 O(N) 全队列扫描；
- 保持整体 FCFS/Priority 语义可理解；
- 限制策略 CPU overhead；
- 防止后排 warm request 无限跨越前排 request。

---

## 6.2 Aging

candidate 内也不能只看 cache。

概念上：

```text
priority
    ↓
resume state
    ↓
aging + cache affinity
```

可以设计成分层规则，而不是一开始就复杂公式。

例如：

```text
Tier 0：必须尊重 user priority
Tier 1：达到 starvation/aging threshold 的 request
Tier 2：preempted/resume request
Tier 3：cache-affinity ranking
Tier 4：arrival order
```

或者：

```text
score =
cache benefit
+
bounded aging bonus
```

**第一版优先使用分层/tier rule，少用难解释的连续权重公式。**

---

# 7. Cache-Affinity 的特征

最直接：

```text
cached_prefix_tokens
```

但更有意义的是：

```text
remaining_prefill_tokens
=
prompt_tokens - cached_prefix_tokens
```

例如：

```text
A:
prompt=4096
cached=0
remaining=4096

B:
prompt=4096
cached=3584
remaining=512
```

B 的 admission 价值更容易解释为：

> **完成它还需要多少真实计算？**

---

## 7.1 可选 admission value

第一版不必复杂：

```text
cache_saved_compute ≈ cached_prefix_tokens
```

或：

```text
remaining_work ≈ uncached_prefill_tokens
```

随后再加：

```text
aging
priority
resume state
```

---

# 8. 进阶核心二：Lazy Per-Iteration Scheduling Features

这是架构含金量最高的扩展之一。

随着策略增加，会需要：

```text
request.priority
request.num_computed_tokens
request.num_preemptions
cached_prefix_tokens
remaining_prefill_tokens
waiting_time
reclaimable_blocks
```

如果每个策略各查一次：

```text
Admission → prefix lookup
Eviction → prefix lookup
Preemption → KV block lookup
...
```

会增加 Scheduler CPU overhead。

---

## 8.1 目标

设计一个只存在于当前 scheduler iteration 的：

```text
RequestSchedulingFeatures
```

概念字段：

```text
priority
is_preempted
num_preemptions
computed_tokens
cached_prefix_tokens       # lazy
remaining_prefill_tokens   # derived
waiting_time               # cheap
reclaimable_blocks         # optional lazy
```

---

## 8.2 必须是 read-only snapshot / cache

它不是：

```text
新的持久 Request 状态真源
```

而是：

```text
当前 scheduling iteration
对真实 Request/KV state 的只读派生视图
```

避免两份 source of truth。

---

## 8.3 Lazy 的意义

例如当前 iteration 只有 preemption 发生：

```text
只需要 computed_tokens
```

那么：

```text
不要查询 cached_prefix_tokens
```

只有 admission policy 真正比较 warm/cold candidate 时：

```text
才做 prefix lookup
```

只有高级 victim scoring 开启时：

```text
才计算 reclaimable_blocks
```

---

## 8.4 Bounded + Lazy 是一起的

理想复杂度不是：

```text
所有 waiting requests × 所有 features × 所有 policies
```

而是：

```text
bounded candidates
×
only requested features
```

这样才有资格把更多智能策略放进 Scheduler hot path。

---

# 9. SchedulingFeatures 的建议语义

Agent 需要精确设计，但需要守这些 invariant。

## 9.1 Per-iteration 生命周期

每轮重新构建或使用明确 generation/version。

不要跨 scheduler iteration 缓存那些会变化的：

- cached prefix
- refcount
- reclaimable block
- waiting time

除非有完整 invalidation，当前项目不建议做。

## 9.2 Cheap field 可直接引用

例如：

```text
priority
num_computed_tokens
num_preemptions
arrival_time
```

可以直接读 Request。

## 9.3 Expensive field lazy resolve

例如：

```text
cached_prefix_tokens
reclaimable_blocks
```

才使用 resolver。

## 9.4 Resolver 必须只读

同基础版：

- 不改 refcount
- 不改 LRU
- 不计真实 hit stats
- 不触发 KV connector
- 不改变 request state

---

# 10. 进阶核心三：Priority + Demand-Aware Prefix Retention

基础版只知道：

```text
near-head waiting request 需要 B
→ B 暂时 retained
```

进阶版加入“这个未来需求本身有多重要”。

---

## 10.1 Prefix / KV Block Value 的直觉

可以近似看成：

```text
KV value
≈
near-term reuse
×
saved compute
×
request importance
```

不要第一版真的写概率乘法。

优先做 tier。

---

## 10.2 推荐 Tier Retention

例如：

```text
Tier 0：普通 cached free blocks

Tier 1：
near-head waiting request 可能复用

Tier 2：
near-head + preempted/resume request 可能复用

Tier 3：
near-head + high-user-priority request 可能复用
```

Eviction preference：

```text
Tier 0
→ Tier 1
→ Tier 2
→ Tier 3
```

每个 tier 内仍保持原有 LRU 顺序。

---

## 10.3 为什么 Tier 比复杂 score 更适合第一版？

因为：

- correctness 好验证；
- user priority 语义更明确；
- 可解释；
- benchmark 容易做 ablation；
- 不需要调权重；
- 不会因为一个错误归一化把策略搞怪。

---

# 11. 进阶核心四：Reclaimable-KV-Aware Preemption

这是基础版 `num_computed_tokens` 的深入版。

## 11.1 基础问题

只比较：

```text
recompute_cost
```

还缺一个问题：

> 抢它到底能腾出多少显存？

例如：

```text
A:
computed=100
reclaimable=100 blocks

B:
computed=3000
reclaimable=20 blocks
```

A 明显是更好的 victim：

```text
损失计算少
+
释放内存多
```

---

## 11.2 Raw allocated blocks 不能代表 reclaimable blocks

因为：

- shared prefix
- refcount > 1
- hybrid KV groups
- padding/null blocks
- cached-but-free 与 active reference 语义不同

所以不能：

```text
reclaimable = len(request.block_ids)
```

---

## 11.3 推荐 helper

概念：

```python
get_reclaimable_block_count(request) -> int
```

只读地估计：

> 如果现在按当前正式 free semantics 释放该 request，有多少 block 会真正从 active referenced 状态转成可重新分配状态？

Agent 必须先研究当前 base commit 的 refcount/group semantics。

---

## 11.4 第一版高级 victim 选择不要急着做 ratio

不建议直接：

```text
score = reclaimable / recompute
```

因为：

- ratio 对小 denominator 敏感；
- 很难处理“必须释放至少 K blocks”的约束；
- 多 request 连续 preempt 时可能局部最优但整体差。

更稳的方式：

```text
Step 1:
保持 user-priority tier

Step 2:
根据当前 allocation deficit，
选择能有效释放 KV 的候选

Step 3:
在可行候选中最小化 recompute cost
```

如果一次 victim 不够，Scheduler 仍按原循环继续 preempt。

---

# 12. 统一 Request/KV Value 的最终抽象

做到这里，项目的研究/工程思想可以统一成：

## Admission Value

```text
高：
- cached prefix 多
- remaining work 少
- 已经被 preempt，存在 sunk work
- 等待时间长
- user priority 高
```

## Preemption Cost

```text
高：
- computed work 多
- 可回收 KV 少
- user priority 高
```

## Cache Retention Value

```text
高：
- near-head request 即将复用
- 能节省大量 prefix compute
- 对高 priority/resume request 有帮助
```

不要为了统一强行写一个“万能公式”。

**统一的是特征和设计原则，不一定是同一个数学 score。**

---

# 13. 为什么不要一开始就做复杂 Cost Model？

面试中真正加分的是：

> 你知道为什么选择简单、稳定、低 overhead 的 signal。

而不是：

```text
0.31 * cached_tokens
+ 0.19 * waiting_time
+ 0.42 * priority
...
```

这种没有数据标定的“拍脑袋公式”反而减分。

第一版优先：

- lexicographic ordering
- tiers
- bounded window
- simple derived work estimates

如果 benchmark 发现需要，再调。

---

# 14. Performance 方向一：Scheduler CPU Hot-Path Optimization

完成调度/KV策略之后，进一步的性能工程不一定先写 GPU kernel。

**Scheduler CPU hot path 是更稳的进阶方向。**

---

## 14.1 为什么值得做？

GPU 越快：

```text
单 step GPU compute 越短
```

CPU scheduler / Python metadata overhead 越容易占比明显。

尤其你增加了：

- Cache affinity
- Feature lookup
- Prefix retention

更应该证明：

> 智能策略没有把 Scheduler 自己拖慢。

---

## 14.2 首先测自己新增的 overhead

至少做 source-level microbenchmark / timing：

```text
baseline schedule decision
vs
feature-enabled schedule decision
```

关注：

- 1 / 8 / 16 / 64 / 256 waiting requests
- candidate window 开/关
- cache lookup 是否触发
- retention resolver 是否 lazy

目标不是造新的科研 benchmark，而是可以使用已有测试/benchmark 基础，或非常小的 source-level profiling。

---

## 14.3 优先优化方向

可能包括：

### A. 避免重复 prefix lookup

由 Lazy Features 解决。

### B. bounded candidate iteration

避免全 waiting queue 排序/扫描。

### C. fast path

当：

```text
feature disabled
或
无 prefix cache
或
没有 memory pressure
```

应尽量走原路径。

### D. 减少临时对象 / 大列表分配

如果 profiling 真证明是热点再改。

### E. 避免无意义的完整 block-table scan

必须先真实 profile 当前 base commit，不要假设。

---

# 15. Performance 方向二：可选 GPU Operator Optimization

## 15.1 能做，但定位必须正确

不能设目标：

> “一周让 AI 重写 FlashAttention，并比成熟实现更快。”

不现实。

可以设：

> “针对 RTX 4090 / SM89 某个常见 KV-related shape，基于已有 benchmark 找到一个真实 gap，用 specialization/vectorization/launch tuning 优化一个小型 memory-bound operator。”

这完全可能。

---

# 16. 不建议优先做的 Kernel

## 16.1 GEMM

不建议。

很难短期超过：

- cuBLAS
- CUTLASS
- 已调优 Triton GEMM

## 16.2 FlashAttention/PagedAttention 主 kernel

风险极高：

- online softmax
- register/shared-memory pressure
- warp specialization
- tensor core
- tiling
- numerical correctness

不适合当前一周主线。

## 16.3 Fused MoE GEMM

也不是第一选择。

shape/routing/quantization 变量太多。

---

# 17. 最推荐尝试的算子：KV Cache Write / reshape-and-cache

原因：

1. 和当前项目逻辑完全连得上；
2. vLLM repo 已有 kernel benchmark；
3. 它通常是 memory movement + addressing 型 kernel；
4. 可以做 GPU-specific fast path；
5. 不必挑战 attention/GEMM 算法核心。

---

# 18. reshape-and-cache 的系统位置

新 token 生成当前层 K/V：

```text
K / V
shape roughly:
[num_tokens, num_kv_heads, head_size]
```

Scheduler/KV manager 已经决定：

```text
slot mapping
physical KV block
offset
```

然后 kernel：

```text
new K/V
   │
   ▼
slot mapping
   │
   ▼
block id / offset
   │
   ▼
paged KV cache layout
```

因此项目可以从：

```text
Scheduler admission
```

一路讲到：

```text
physical KV cache write
```

---

# 19. KV-write Kernel 可探索的方向

**只在 benchmark 证明有空间后做。**

## 19.1 Vectorized memory access

检查是否能安全使用更宽 load/store：

```text
32-bit
→ 64/128-bit vectorized access
```

需要满足 alignment / dtype / head dimension 条件。

## 19.2 Address calculation specialization

常见：

```text
block_id = slot // block_size
offset   = slot % block_size
```

当：

```text
block_size = 16
head_dim = 128
```

等编译期固定时，可以探索 constexpr / bit operation / specialized mapping。

## 19.3 Common-shape fast path

例如只针对：

```text
SM89
BF16/FP16
head_dim=128
block_size=16
```

提供 specialized path。

Generic path 保持不变。

## 19.4 Triton launch configuration

探索：

```text
num_warps
program size
tokens/program
vector width
```

但必须通过真实 4090 benchmark 选择。

---

# 20. AI 在算子优化中应该扮演什么角色？

AI 很适合：

```text
读现有 kernel
→ 建立 memory access map
→ 提候选优化
→ 生成多个 Triton/CUDA candidate
→ 写 correctness cross-check
→ 调用已有 benchmark sweep
→ 汇总结果
```

AI 不应该直接宣称：

```text
这个 kernel 会更快
```

只有 GPU benchmark 有权回答。

---

# 21. 算子优化的验收条件

没有性能数据就不算优化。

至少要求：

```text
correctness:
output bitwise / tolerance correct

performance:
existing vLLM kernel benchmark
baseline vs candidate
多个 representative shape
```

必须避免只挑一个极小 shape 显示 30% 提升，而其他常见 shape 全退化。

理想策略：

```text
specialized fast path 只在明确 shape 命中
否则 fallback baseline
```

---

# 22. 进阶版项目的推荐优先级

| Priority | 项目 | 价值 | 风险 |
|---|---|---:|---:|
| P0 | 基础 Policy 架构 | 后续所有策略地基 | 低 |
| P0 | Recompute-aware Preemption | 核心调度 | 中 |
| P0 | Resume-aware Re-admission | 补完整生命周期 | 低 |
| P0 | Queue-informed Prefix Retention | 核心 KV | 中 |
| P0 | Cache-Affinity Admission + Aging | 明显提升项目层次 | 中 |
| P1 | Lazy Bounded SchedulingFeatures | 架构+CPU性能 | 中 |
| P1 | Priority/Demand-aware Retention | Cache Value 深化 | 中 |
| P1 | Reclaimable-KV-aware Preemption | Memory efficiency 深化 | 中高 |
| P2 | Scheduler CPU hot-path profiling/optimization | 真正性能工程 | 中 |
| P2 | SM89 KV-write kernel optimization | Data Plane 亮点 | 高 |
| P3 | 普通新模型适配 | 对本项目增益较小 | 低中 |

---

# 23. 推荐实施阶段

## Phase 0：Unified Inspection / Plan

**现在就做。**

Agent 同时阅读：

```text
基础版 MD
+
本进阶版 MD
+
用户实际 base commit
```

输出一个统一：

```text
IMPLEMENTATION_PLAN.md
```

但不要改代码。

---

## Phase 1：Stable Foundation

落地：

1. Mini Policy abstraction
2. Recompute-aware preemption
3. Resume-aware re-admission

要求：

- 默认行为 compatibility
- targeted scheduler tests
- clean commit history

---

## Phase 2：Base KV Coordination

落地：

4. Queue-informed Prefix Cache retention

要求：

- read-only prefix probe
- soft retention
- fallback eviction
- BlockPool tests
- Scheduler/KV integration tests

做到这里，**基础版项目已经完整可投简历**。

---

## Phase 3：Advanced Admission

落地：

5. Cache-Affinity Admission
6. Bounded Candidate Window
7. Aging / starvation protection

要求：

- strict user priority
- bounded reordering
- cache-warm request test
- cold request starvation test

---

## Phase 4：Scheduling Features

落地：

8. Lazy per-iteration SchedulingFeatures

将：

```text
cached tokens
remaining work
computed tokens
resume state
waiting age
```

统一成 read-only feature access。

注意：

> 如果 Agent 在 Phase 0 判断没有提前做一个很小 FeatureContext 就会导致 Phase 1/2 重构，可以在 Phase 1 先加入“最小骨架”，但 expensive feature 本身仍在 Phase 4 才正式启用。

---

## Phase 5：KV Value Deepening

落地：

9. Priority/Demand-aware Prefix Retention
10. Reclaimable-KV count
11. Advanced victim selection

做完这里，项目已经是非常完整的 Adaptive KV-aware serving system。

---

## Phase 6：CPU Performance Engineering

- profile scheduler
- quantify new policy overhead
- eliminate duplicate lookup
- verify bounded/lazy fast path
- optimizations based on evidence

---

## Phase 7：Optional GPU Data Plane

只有前面稳定后：

- inspect `reshape_and_cache`
- run existing kernel benchmark
- 找真实 4090 gap
- candidate kernel
- correctness
- performance
- specialized fallback

**如果没测出正收益，宁可不写简历，也不要硬保留一个慢 kernel。**

---

# 24. 为什么这是最佳衔接顺序？

因为依赖关系是：

```text
Policy abstraction
      │
      ├──► Recompute Preemption
      ├──► Resume
      └──► Admission
              │
              ▼
      Lazy SchedulingFeatures
              │
      ┌───────┴────────┐
      ▼                ▼
Cache Affinity   Prefix Retention
                       │
                       ▼
             Reclaimable-KV Info
                       │
                       ▼
             Advanced Preemption
```

GPU kernel则几乎独立于策略 correctness：

```text
KV allocation / slot mapping
      ↓
KV write operator
```

所以最后做最安全。

---

# 25. 对 Agent 的架构要求：不要为了进阶版提前过度设计

Agent 看到未来需求后，可能会想直接做：

```text
ISchedulingFeatureProvider
IAdmissionPlugin
IPreemptionPlugin
ICachePolicyPlugin
PluginRegistry
DynamicLoader
...
```

禁止。

目标仍然是：

> **最小结构能支持当前 5~8 个策略点。**

如果两个函数 + 一个小 context 足够，就不要建框架帝国。

---

# 26. 建议 Feature API 形态（仅概念）

可能类似：

```python
features = SchedulingFeatures(request, context)

features.priority
features.computed_tokens
features.is_preempted

# lazy:
features.cached_prefix_tokens()
features.remaining_prefill_tokens()
features.reclaimable_blocks()
```

也可能：

```python
feature_ctx.get(request, Feature.CACHED_PREFIX)
```

实际选哪一个以当前 vLLM coding style 为准。

要求：

- no duplicate source of truth
- no side effect
- per-iteration
- lazy
- bounded
- easy to unit test

---

# 27. Admission Policy 的建议语义

**先使用 lexicographic / tiered rule。**

例如：

```text
1. user priority
2. starvation/aging guard
3. resume state
4. cache affinity / remaining prefill
5. arrival time
```

或者根据当前 vLLM semantics 调整 2/3 顺序。

必须通过 test 明确。

---

# 28. Advanced Retention 的建议语义

不要用“prefix popularity”替代 near-future demand。

优先：

```text
waiting near-head demand
```

而不是历史 LFU。

推荐：

```text
Eviction Group 0:
no near-term demand

Group 1:
near-term normal request

Group 2:
near-term resumed request

Group 3:
near-term high-priority request
```

Group 内保持 LRU。

---

# 29. Advanced Preemption 的建议语义

推荐逐步演进：

## v0

```text
user priority
→ arrival time
```

base behavior。

## v1

```text
user priority
→ recompute cost
→ stable tie-break
```

## v2

```text
user priority
→ reclaimability feasibility
→ recompute cost
→ stable tie-break
```

不要 v1 一上来就写复杂多目标优化。

---

# 30. Fairness / Starvation 必须成为进阶版一等公民

Cache-aware 策略最容易被问：

> 会不会冷请求饿死？

必须有明确答案：

- bounded window
- aging
- user priority hard constraint
- deterministic fallback

至少新增测试：

```text
不断加入 warm requests
cold request waiting time 增长
最终 aging guard 使 cold request 获得 admission
```

如果无法在真实 scheduler test 中方便模拟时间，可使用可控 arrival time / iteration-based age proxy，但要保持语义明确。

---

# 31. 性能指标应该分三类

## 31.1 Serving Metrics

已有工具支持时：

- throughput
- TTFT
- TPOT/ITL

## 31.2 Work Efficiency Metrics

更贴近本项目：

- prefix cached tokens hit
- redundant recompute tokens
- preemption count
- resumed request delay
- cached block eviction / retained hit

## 31.3 Scheduler Overhead

进阶版必须关注：

- decision latency
- prefix lookup count
- candidate count
- expensive feature resolver calls

---

# 32. Benchmark Workload 不需要科研数据集

继续坚持：

> 只使用 vLLM 原有测试和 benchmark 工具。

优先：

```text
prefix_repetition
benchmark_prefix_caching.py
已有 scheduler tests
已有 kernel benchmarks
```

可以调：

- concurrency
- prompt length
- repeated prefix ratio
- KV cache budget
- priority mix

这些是 workload 参数，不是新数据集。

---

# 33. 4090 如何做这些实验？

24GB 完全够。

关键不是塞最大模型，而是制造：

```text
KV pressure
```

方法：

```text
使用较小 3B/7B 模型
+
限制 KV cache budget
+
提升并发
+
较长 prompt
+
prefix repetition
```

这样能稳定触发：

- preemption
- prefix eviction
- cache-affinity
- re-admission

---

# 34. Ablation 建议

进阶版很适合做简单 ablation。

最低：

```text
A baseline
B base scheduler only
C + queue-aware cache
D + cache-affinity admission
E + advanced features
```

如果时间紧：

```text
baseline
vs
full optimized
```

先保证。

---

# 35. 不要追求“所有 workload 都更快”

正确目标：

```text
No pressure:
≈ baseline

Memory pressure + reuse:
optimized > baseline
```

如果无 KV pressure 时策略产生明显退化，说明 CPU overhead/ordering 设计有问题。

这也是 Lazy + Bounded 的价值。

---

# 36. Kernel 方向的 Go / No-Go Gate

不要默认一定做 kernel。

只有满足：

```text
1. 基础/进阶 scheduler 已稳定
2. 已有 benchmark 能运行
3. 4090 上至少找到一个可重复 gap
4. candidate 优化能保持 correctness
```

才进入 kernel implementation。

否则：

> 将 Kernel Exploration 写进 README “future work”，不要硬塞简历。

---

# 37. CPU Optimization 的 Go / No-Go Gate

同理：

如果 profile 显示：

```text
Scheduler CPU overhead 完全不是瓶颈
```

不要为了“有 CPU 优化”乱改。

可以把成果写成：

> 对新增策略进行 hot-path profiling，确认 bounded/lazy feature lookup 将额外开销控制在 XXX。

这本身也是性能工程。

---

# 38. 推荐 Git 分支策略

不用新开“advanced-project”重做。

仍然：

```text
main
└── project/kv-aware-scheduling
```

同一项目分支按 commit 演进。

可以在危险实验时短期开：

```text
experiment/cache-affinity
experiment/kv-write-sm89
```

验证后 merge/cherry-pick 回项目分支。

---

# 39. 推荐 Commit 序列

基础：

```text
feat(scheduler): add lightweight scheduling decision policy

feat(scheduler): add recompute-aware preemption selection

feat(scheduler): prioritize preempted requests on re-admission

feat(kv-cache): add waiting-queue-informed prefix retention
```

进阶：

```text
feat(scheduler): add bounded cache-affinity admission

feat(scheduler): add aging guard for cache-aware ordering

refactor(scheduler): add lazy per-iteration scheduling features

feat(kv-cache): weight prefix retention by request demand

feat(kv-cache): expose reclaimable block estimate

feat(scheduler): use reclaimable KV in preemption decisions
```

性能：

```text
perf(scheduler): avoid duplicate prefix feature lookup

perf(kv-cache): specialize reshape-and-cache for sm89 common shapes
```

最后一条只有真实更快才 commit 到主项目分支。

---

# 40. SSP 级项目真正靠什么？

不是 feature 数量。

更有价值的是：

1. **真实主干代码**
2. **跨 Scheduler/KV/BlockPool 模块**
3. **清晰 correctness invariants**
4. **真实 workload motivation**
5. **有 baseline 和 ablation**
6. **有 hot-path overhead 意识**
7. **最好能提交 upstream RFC/PR**
8. **能从 control plane 讲到 data plane**

如果这些做到了，项目的面试价值远高于：

```text
适配 3 个普通模型
+
写 2 个 demo kernel
```

---

# 41. 最终简历叙事可以升级成什么？

建议最终项目标题：

## vLLM V1 Adaptive KV-Aware Serving Engine Optimization

下面不要写 8 条 feature，还是压缩成 3~4 条主线。

### Bullet 1：Architecture / Control Plane

> 基于 vLLM V1 构建轻量级 KV-aware scheduling policy，解耦 admission/preemption 决策与 correctness-critical request/KV 状态管理；设计 bounded cache-affinity admission、aging 和 resume-aware ordering，在保持用户 Priority 语义的同时利用 Prefix Cache locality 降低有效 prefill work。

### Bullet 2：Preemption / Memory Efficiency

> 设计 recomputation-aware / reclaimable-KV-aware preemption，在同优先级候选中综合已计算 token 与可回收 KV blocks 选择 victim，减少高重算成本请求的无效抢占，并通过 re-admission 优先恢复已有执行进度。

### Bullet 3：Cache Management

> 实现 waiting-queue / priority-aware Prefix Cache retention，以近期请求需求和 saved compute 对 cached free blocks 分级，保持组内 LRU，并通过 soft retention + fallback eviction 保证显存不足时不改变原始 allocation feasibility。

### Bullet 4：Performance Engineering（有真实结果再写）

> 通过 bounded candidate window 与 lazy per-iteration scheduling features 复用 Prefix/KV metadata，控制新增策略的 Scheduler hot-path overhead；进一步基于 vLLM 原生 kernel benchmark 对 RTX 4090 KV Cache write path 进行 shape-specific tuning，XXX。

第四条没有真实结果之前不要放 kernel 数字。

---

# 42. Agent 下一步的明确任务

现在不要让 Agent“先完全实现旧 MD”。

正确 prompt 是：

> 同时阅读 `vllm_kv_aware_project_design.md`（基础版）与 `vllm_kv_aware_project_advanced_design.md`（增量进阶版）。基础版定义 P0 correctness 和第一阶段功能，进阶版定义后续增量目标。先对当前冻结的 vLLM base commit 做完整源码勘察，生成一个统一 `IMPLEMENTATION_PLAN.md`。Plan 必须保证第一阶段的接口不会阻碍后续 Cache-Affinity Admission、Lazy SchedulingFeatures 和 Advanced Retention，但不得为了未来扩展实现完整 plugin framework。代码实施按 Phase 1→Phase 7 分阶段，每阶段独立测试、commit、验收；未经当前阶段验收不得批量实现下一阶段。

---

# 43. IMPLEMENTATION_PLAN.md 进阶版必须额外包含

除基础版要求外，再加入：

## A. Admission path map

精确找出：

- waiting requests 如何被 peek/pop
- token budget 检查
- KV prefix lookup 在 admission 前后的时机
- 哪个位置能做 bounded candidate selection
- Priority queue 怎样安全枚举 top-K

## B. Feature cost map

列出每个候选 feature：

```text
feature
source
time complexity
side effect
是否 lazy
```

例如：

```text
priority                Request                O(1)
computed_tokens         Request                O(1)
waiting_age             Request/clock          O(1)
cached_prefix_tokens    KV cache lookup        expensive
reclaimable_blocks      KV/refcount scan       potentially expensive
```

实际复杂度以当前代码为准。

## C. Fast path

明确：

```text
默认 policy
无 prefix caching
waiting 少于阈值
KV 无 pressure
```

分别是否绕过新增逻辑。

## D. Fairness proof / tests

说明：

- priority 不被跨越
- bounded reordering
- aging 如何避免 starvation
- resume 如何和 aging/cache affinity 组合

## E. Performance measurement

明确哪些现有 benchmark 能用于：

- admission
- retention
- preemption
- scheduler overhead
- kernel

## F. Kernel reconnaissance（只调查，不实现）

Agent 需要：

- 找 `reshape_and_cache` 的真实实现
- 找 backend dispatch
- 找现有 benchmark
- 确认 SM89 是否已有 specialization
- 列出可能优化空间
- **不得在本阶段写新 kernel**

---

# 44. 阶段验收 Gate

## Gate 1：基础 Scheduler

必须：

- default tests pass
- new preemption tests pass
- user priority correctness

## Gate 2：Base KV Retention

必须：

- prefix tests pass
- retention read-only
- fallback correctness

## Gate 3：Cache-Affinity Admission

必须：

- warm request can benefit
- cold starvation test
- bounded candidate test
- default off compatibility

## Gate 4：Features

必须：

- no duplicate expensive lookup in same iteration
- lazy resolver not called on fast path
- no state mutation

## Gate 5：Advanced KV

必须：

- reclaimable estimate correct under shared block test
- advanced victim correctness
- priority retention correctness

## Gate 6：Performance

必须：

- overhead measured
- optimization evidence

## Gate 7：Kernel

必须：

- correctness
- real speedup
- no common-shape regression or gated specialization

---

# 45. 项目 Stop Rules

一旦出现以下情况，停止扩功能：

- Phase 1/2 correctness 还没稳定；
- 新 feature 需要大改 vLLM 状态机；
- 核心 patch 超过约 1000 行且无法解释；
- Cache-Affinity 需要扫描全部 waiting queue；
- Prefix probe 有不可控副作用；
- Reclaimable KV 在当前 Hybrid Manager 下无法可靠定义；
- Kernel 连续多轮 benchmark 都无收益。

宁可少一个 feature，也不要把主项目变成不可维护实验分支。

---

# 46. 与基础版文档的最终关系

可以理解为：

```text
基础版
=
Minimum Viable Strong Project
=
一周内一定要保底完成


进阶版
=
Depth Upgrade
=
让项目从“真实 vLLM 改进”
升级为“Adaptive KV-aware Serving System”
```

基础版不是废稿。

它是：

```text
Phase 1 + Phase 2
```

进阶版是在其上追加：

```text
Phase 3 ~ Phase 7
```

---

# 47. 推荐时间策略

如果离投简历非常近：

```text
先完成 Phase 1 + 2
→ 马上写简历/投递
→ 代码继续 Phase 3+
```

简历项目可以持续更新。

到真正面试时：

```text
GitHub 项目已经比投递时更深
```

完全正常。

不要为了“等所有高级功能做完再投”错过秋招窗口。

---

# 48. 最后的路线图

```text
              ┌────────────────────────────┐
              │ Phase 0 Unified Inspection │
              └─────────────┬──────────────┘
                            ▼
              ┌────────────────────────────┐
              │ Phase 1 Scheduler Base     │
              │ Policy / Preempt / Resume  │
              └─────────────┬──────────────┘
                            ▼
              ┌────────────────────────────┐
              │ Phase 2 Base KV Retention  │
              └─────────────┬──────────────┘
                            │
                ← 已经是可投简历版本 →
                            │
                            ▼
              ┌────────────────────────────┐
              │ Phase 3 Cache Admission    │
              │ Affinity + Aging + Bound   │
              └─────────────┬──────────────┘
                            ▼
              ┌────────────────────────────┐
              │ Phase 4 Lazy Features      │
              └─────────────┬──────────────┘
                            ▼
              ┌────────────────────────────┐
              │ Phase 5 KV Value Deepening │
              │ Priority + Reclaimability  │
              └─────────────┬──────────────┘
                            │
              ← 强力推理引擎项目版本 →
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
        ┌──────────────────┐  ┌──────────────────┐
        │ Phase 6 CPU Perf │  │ Phase 7 GPU Op   │
        └──────────────────┘  └──────────────────┘
                  │                   │
                  └─────────┬─────────┘
                            ▼
               Control → Memory → Data Plane
```

---

# 49. 一句话原则

> **不要“先做旧方案，再重做新方案”；现在就用新方案指导接口设计，但让旧方案作为第一阶段功能逐步落地。**

> **不要把进阶理解成更多 heuristic；真正的进阶是把 admission、preemption、cache retention 统一到 KV/work value 视角，同时控制 Scheduler 自身 overhead。**

> **Kernel 是 bonus，不是主线救命稻草；只有真实 benchmark 证明更快，它才是优化。**

> **为了 SSP，深度、correctness、performance evidence 和代码质量，比 feature 数量更重要。**
