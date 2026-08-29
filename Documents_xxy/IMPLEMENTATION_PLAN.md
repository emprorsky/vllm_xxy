# vLLM V1 Adaptive KV-Aware Serving 统一实施计划

> 状态：Phase 1、Phase 2、Phase 3、Phase 4、Phase 5a correctness Gate 已完成；
> Phase 2 性能 Gate 未通过，Phase 3/4/5a 的性能信号尚未完成稳定统计验收。详见
> [`PHASE1_IMPLEMENTATION_REPORT.md`](PHASE1_IMPLEMENTATION_REPORT.md)、
> [`PHASE2_REVIEW_CODEX.md`](PHASE2_REVIEW_CODEX.md) 和
> [`PHASE3_IMPLEMENTATION_REPORT_CODEX.md`](PHASE3_IMPLEMENTATION_REPORT_CODEX.md)、
> [`PHASE4_IMPLEMENTATION_REPORT_CODEX.md`](PHASE4_IMPLEMENTATION_REPORT_CODEX.md)、
> [`PHASE5A_IMPLEMENTATION_REPORT_CODEX.md`](PHASE5A_IMPLEMENTATION_REPORT_CODEX.md)。
> 适用仓库：`/root/autodl-tmp/repos/vllm`
> 勘察日期：2026-08-28 (UTC)
> 原则：Policy 只做 decision；Scheduler/KV core 仍然负责所有 correctness-critical mutation。

## 0. Executive decision

本项目应当**沿着已有实现继续，但不把当前 `a51216af4` 视为已经验收的 Phase 1**。

已有工作中应保留的部分：

- `Documents_xxy/stress_bench.py` 已能在 4090 上稳定制造 KV pressure，证明了项目问题真实存在。
- `vllm/v1/core/sched/policy.py` 已建立“policy 选 victim，Scheduler 删 running entry、回滚 budget、free KV、改 status”的正确边界。
- `Scheduler.schedule()` 中针对任意 victim index 的 cursor/budget 回滚是必要修正，而且已有专门的 upstream regression test 场景可用。
- typed config + CLI 能清晰地做 baseline/optimized 对照。

当前实现需要在 Gate 1 前收紧的部分：

- 现有 victim key 实际是 `(priority, num_preemptions, computed, -arrival)`；它按抢占的**精确次数**排序，不是文档和 docstring 所说的 binary anti-thrashing。
- 需把 anti-thrashing 明确化为同 user-priority tier 内的 binary resume-protection tier，然后才比 recompute cost；并增加 request-id deterministic tie-break。
- 当前只有 victim policy，尚无 Priority 模式的 resume-aware re-admission。FCFS 因 `appendleft` 已自然优先 resume，Priority 的 `prepend_request()` 却只是普通 `heappush`。
- 现有 policy tests 主要测 `SimpleNamespace` 的纯函数，还不足以证明 running-list cursor、budget rollback、request status、Priority heap 和 re-admission 集成语义。
- 当前 benchmark 的 `n_tokens` 实际计数 SSE content chunks，不一定等于 tokenizer tokens；`itl_mean.p99` 也是“每请求 p99 的跨请求 p99”，不能当作标准 global ITL p99。它应保留为 pressure/diagnostic harness，最终 serving 数据以 `vllm bench serve` 为主。

因此路线为：

```text
Existing stress baseline + policy seed
                 |
                 v
Phase 1: stabilize victim policy + resume-aware queue semantics
                 |
                 v
Phase 2: lazy waiting-demand prefix retention
                 |
       minimum strong project gate
                 |
                 v
Phase 3+: bounded admission, aging, lazy features, KV value deepening
```

---

## A. Repository state

### A.1 Git state before creating this plan

| Item | Observed value |
|---|---|
| Branch | `project/kv-aware-scheduling` |
| Project HEAD | `a5548fb53c7c5cdc477fc28ad5e8d0a7cc01425f` |
| HEAD subject | `docs: move project docs into Documents_xxy` |
| Frozen upstream/base commit | `fd57c4b7afebc0b43d25ed7f5848fc35786463d0` |
| Base subject | `[Rocm][CI] add dockerfile.xpu to rocm ci artifact (#53949)` |
| Merge-base with `origin/main` | `fd57c4b7afebc0b43d25ed7f5848fc35786463d0` |
| Working tree | clean before this file was created |
| Tracking branch | `origin/project/kv-aware-scheduling` at the same HEAD |
| `origin` | `https://github.com/emprorsky/vllm_xxy.git` (fork) |
| `upstream` | `https://github.com/vllm-project/vllm.git` (official) |
| Version metadata | `vllm/_version.py`: `0.1.dev20460+gfd57c4b7a` |
| GPU observed | NVIDIA GeForce RTX 4090, SM 8.9, 24564 MiB |

`Project HEAD` 和 `Frozen upstream/base commit` 必须在后续报告中分开：前者包含项目自有提交，后者是用于评估 default compatibility 的 vLLM 基线。不更新 upstream，不更换 base。

### A.2 Existing project commits

| Commit | Content | Decision |
|---|---|---|
| `f6a832a13` | setup + base design | retain |
| `543ebbf43` | pressure benchmark + 135-preemption baseline | retain as diagnostic evidence |
| `a51216af4` | policy seed + recompute-aware victim + tests + result JSON | evolve in place; Gate 1 not yet complete |
| `d60b8a0ec` | advanced design and machine notes | retain |
| `a5548fb53` | move docs under `Documents_xxy` | retain |

Relative to `origin/main`, business-code changes currently touch:

```text
vllm/config/scheduler.py
vllm/engine/arg_utils.py
vllm/v1/core/sched/policy.py
vllm/v1/core/sched/scheduler.py
tests/v1/core/test_preemption_policy.py
```

### A.3 Environment/test state

This AutoDL project uses the pre-provisioned `vllm-dev` conda environment.
All Python commands in this plan must use:

```bash
/root/miniconda3/envs/vllm-dev/bin/python
```

Do not create a `.venv` or run dependency installation commands unless the
user explicitly requests it. Historical test results never substitute for a
fresh Gate run at the current HEAD.

---

## B. Current architecture map

### B.1 One scheduler iteration

```text
Scheduler.schedule()                                  scheduler.py:505
  |
  +-- initialize token/input/encoder budgets
  +-- kv_cache_manager.new_step_starts()
  |
  +-- RUNNING loop                                    scheduler.py:552-762
  |     |
  |     +-- compute num_new_tokens from
  |     |   num_tokens_with_spec - num_computed_tokens
  |     +-- cap by token/input/model/encoder/Mamba/MTP constraints
  |     +-- KVCacheManager.allocate_slots()           scheduler.py:660
  |           |
  |           +-- success -> record scheduled work and budgets
  |           +-- None -> select/remove victim
  |                         -> rollback same-step bookkeeping if needed
  |                         -> _preempt_request()
  |                         -> retry allocation
  |
  +-- WAITING loop (only if no preemption this step)  scheduler.py:782-1200
  |     |
  |     +-- select waiting/skipped queue
  |     +-- peek queue head
  |     +-- handle blocked/stale/LoRA constraints
  |     +-- local prefix lookup when num_computed_tokens == 0
  |     +-- optional connector lookup
  |     +-- calculate remaining prefill and budgets
  |     +-- KVCacheManager.allocate_slots()           scheduler.py:1068
  |           +-- None -> stop waiting admission
  |           +-- success -> pop queue, append running,
  |                         WAITING/PREEMPTED -> RUNNING
  |
  +-- assert budgets/running limits
  +-- build SchedulerOutput
```

There is no separate decode scheduler and prefill scheduler. A request is work
whose `num_computed_tokens` must catch up to `num_tokens_with_spec`; the same loop
covers chunked prefill, decode, speculative tokens and resumed recomputation.

### B.2 Running allocation failure and preemption

Current call chain:

```text
Scheduler.schedule(): running request
  -> KVCacheManager.allocate_slots(request, num_new_tokens, ...)
       -> coordinator.remove_skipped_blocks(...)
       -> coordinator.get_num_blocks_to_allocate(...)
       -> compare required_blocks against
          block_pool.get_num_free_blocks() - reserved_blocks
       -> None when insufficient
  -> decision_policy.select_preemption_victim(self.running)
  -> remove arbitrary running index
  -> if victim was already scheduled this iteration:
       restore token/input/encoder budgets and remove its output bookkeeping
  -> Scheduler._preempt_request(victim)
       -> _free_request_blocks()
       -> encoder_cache_manager.free()
       -> status RUNNING -> PREEMPTED
       -> num_computed_tokens = 0
       -> clear spec tokens/output placeholders
       -> num_preemptions += 1
       -> waiting.prepend_request(request)
       -> add request id to reset_preempted_req_ids
  -> retry allocate_slots()
```

Important current fact: `allocate_slots()` is not a pure dry-run. Before its
capacity check it may call `remove_skipped_blocks()` and free out-of-window blocks.
Any future policy must not assume a failed allocation left every KV structure
byte-for-byte unchanged.

### B.3 Waiting admission and prefix lookup

At `scheduler.py:795-1171`:

1. `_select_waiting_queue_for_scheduling()` chooses between `waiting` and
   `skipped_waiting`.
2. The scheduler peeks; it does not pop until allocation succeeds or the request
   is explicitly moved to the step-skipped queue.
3. For a normal WAITING/PREEMPTED request with `num_computed_tokens == 0`,
   `_get_local_prefix_cache_hit()` calls one of:
   - `KVCacheManager.get_computed_blocks()` for local/no-connector semantics;
   - `get_computed_blocks_for_connector()` for hybrid connector semantics.
4. Connector hits, encoder readiness, token budget, chunking, Mamba alignment,
   MTP lookahead and full-sequence reservation are resolved before allocation.
5. `allocate_slots()` adopts local cached blocks only after the capacity check.
6. Prefix-cache stats are recorded only after successful admission.
7. The queue item is then popped; `WAITING` becomes a new request and
   `PREEMPTED` becomes a resumed request; both transition to `RUNNING`.

This order is the correct insertion point for future cache-affinity admission:
candidate prefix probes must occur before the final chosen candidate is popped,
and the real hit must be consumed immediately or invalidated after any KV mutation.

### B.4 Allocation accounting

`KVCacheManager.allocate_slots()` currently accounts for:

- already-held request blocks;
- local computed blocks found in prefix cache;
- local-hit blocks with `ref_cnt == 0` that will leave the free queue when touched;
- external connector-computed tokens;
- speculative lookahead;
- encoder/cross-attention blocks;
- hybrid groups with potentially different block sizes;
- sliding/chunked-local recycling and admission caps;
- CoW reservation for a partial local hit;
- async-load reserved blocks;
- an optional waiting/preempted watermark;
- full-sequence admission (`scheduler_reserve_full_isl`, default true).

`num_blocks_to_allocate` is summed across all coordinator groups because they
share one physical `BlockPool`. `get_num_free_blocks()` includes both never-cached
free pages and `ref_cnt == 0` cached eviction candidates. There is no separate
watermark hidden in `BlockPool`; watermark is added by `KVCacheManager` only for
eligible waiting/preempted admissions.

### B.5 Coordinator and cache-manager layers

```text
KVCacheManager
  -> KVCacheCoordinator
       +-- KVCacheCoordinatorNoPrefixCache
       +-- UnitaryKVCacheCoordinator
       +-- HybridKVCacheCoordinator
       |
       +-- one SingleTypeKVCacheManager per KV group
            +-- FullAttentionManager
            +-- SlidingWindowManager / RSWAManager
            +-- ChunkedLocalAttentionManager
            +-- MambaManager
            +-- CrossAttentionManager
       |
       +-- shared BlockPool
```

The design must therefore pass a retention hint through manager/coordinator
allocation without teaching any attention-specific manager about Scheduler
queues. The hint applies to physical block IDs in the shared pool.

### B.6 Prefix lookup side effects

`BlockPool.get_cached_block()` and all examined
`SingleTypeKVCacheManager.find_longest_cache_hit()` implementations only read
hash tables/block metadata and construct result lists. They do not touch blocks,
change refcounts or reorder the free queue.

However, public `KVCacheManager.get_computed_blocks()` is **not universally
side-effect-free**: if KV cache events are enabled and the request uses
`kv_cache_report_mode == "full"`, it emits `BlockStored` events for reused blocks.
The real admission path also separately records prefix stats after allocation.
Therefore Phase 2 must expose/factor an explicit event-free, stats-free local
peek API; it must not call connectors and must not call `block_pool.touch()`.

### B.7 Request state transitions relevant to this project

```text
new request
  -> WAITING / WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
  -> (successful admission) RUNNING
  -> (allocation pressure) PREEMPTED
       - KV request refs freed
       - num_computed_tokens reset to 0
       - num_preemptions incremented
       - request re-enqueued
  -> (successful re-admission) RUNNING
       - local prefix is probed again
       - num_computed_tokens set to the valid hit length
  -> FINISHED_*
       - request/encoder/KV bookkeeping freed
```

Blocked statuses live in `skipped_waiting`; they are not an independent user
priority tier. Async/deferred block free can delay when pages actually return to
the free queue even after request bookkeeping is popped.

---

## C. Request queue semantics

### C.1 FCFS

- `FCFSRequestQueue` subclasses `deque`.
- `add_request()` appends right; `peek_request()`/`pop_request()` use the left.
- `prepend_request()` uses `appendleft()`.
- Consequently the current FCFS preemption path already gives a preempted
  request immediate re-admission preference over cold requests.
- Iteration is true queue order and an `islice` of the first W items is O(W).
- The skipped queue is considered before the regular queue in FCFS mode.

### C.2 Priority

- `PriorityRequestQueue` stores a binary heap of `Request` objects.
- `Request.__lt__` compares:

```text
priority ascending (smaller number = higher user priority)
arrival_time ascending
request_id lexicographically
id(object) as final collision breaker
```

- `num_preemptions` and `RequestStatus` do not currently participate.
- `prepend_request()` is only `heappush()`, so it does not give a resumed request
  any special treatment.
- The raw `_heap` array is **not scheduling order** and must never be sliced as
  top-K.
- `PriorityRequestQueue.__iter__()` is scheduling order, but it first copies the
  entire heap and ultimately pops the entire copy. Even `islice(iter(queue), W)`
  still pays O(N) for the copy; it is unsuitable for a strict bounded hot path.
- Mutating `priority`, `arrival_time` or any future comparison field while the
  request is inside the heap breaks the invariant unless the queue stores an
  immutable key or is rebuilt.
- `_preempt_request()` increments `num_preemptions` before re-enqueueing, so an
  immutable key snapshot built at insertion can safely include binary resume
  state.

### C.3 Required queue evolution

Phase 1 should make ordering explicit instead of modifying `Request.__lt__`
globally:

- let `PriorityRequestQueue` store a small immutable heap item containing an
  insertion-time key plus the `Request`;
- preserve the default key exactly;
- allow the selected decision policy to supply an optimized waiting key;
- expose queue-head ordering so `_select_waiting_queue_for_scheduling()` compares
  the same key across `waiting` and `skipped_waiting`;
- create every transient queue (`step_skipped_waiting` included) through one
  Scheduler helper so it cannot silently use a different comparator.

Phase 3 should add `peek_n(n)`:

- deque: `islice`, O(n);
- priority heap: frontier traversal starting at heap index 0, pushing only child
  indices of popped nodes, O(n log n) time and O(n) temporary space, without
  copying/scanning the whole waiting heap;
- return true scheduling order without mutating the queue.

---

## D. KV cache lifecycle

### D.1 Physical block states

| State | `ref_cnt` | Hash metadata | In free queue | Allocatable now? |
|---|---:|---|---|---|
| null block | unmanaged special case | none | no | no |
| active uncached | `> 0` | none | no | no |
| active cached/shared | `> 0` | present | no | no |
| free uncached | `0` | none | yes, at/near front | yes, no eviction |
| free-but-cached | `0` | present | yes, LRU region | yes, after formal eviction |

The pool reserves physical block 0 as the null block, so a `num_blocks=N` test
has N-1 normal pages.

### D.2 Allocate -> cache -> share

1. `get_new_blocks()` removes free pages from the linked queue.
2. If a selected page is cached, `_maybe_evict_cached_block()` removes all hash
   aliases, updates metrics and emits `BlockRemoved` events.
3. The allocated page's `ref_cnt` becomes 1.
4. When a full request block becomes cacheable, `cache_full_blocks()` installs
   its hash metadata and optional events while the request still references it.
5. A later prefix hit initially only returns block objects. Actual adoption in
   `allocate_new_computed_blocks()` calls `BlockPool.touch()`, removes a free
   cached page from the free queue if necessary and increments its refcount.

### D.3 Free -> cached-free -> evict

`SingleTypeKVCacheManager.free()` releases request blocks in reverse order.
`BlockPool.free_blocks()` decrements each reference:

- a block that remains referenced is not put on the free queue;
- an uncached block reaching zero is prepended for LIFO/locality reuse;
- a cached block reaching zero is appended for FIFO/LRU eviction;
- a shared prefix is only free when all references have been released.

Freeing a request therefore does **not** mean all of its block IDs immediately
become unreferenced, and with deferred free they may not return to the pool in
the current scheduler iteration. A cached page remains in the hash table at
`ref_cnt == 0` until selected by `get_new_blocks()`, explicitly evicted/reset, or
prefix cache reset.

### D.4 Retention semantics required by Phase 2

Retention is only an allocation-time preference over `ref_cnt == 0` cached pages:

```text
original free eviction order: A B C D E
retained IDs:                B D
selection preference:       A C E B D
```

It must not:

- increment a refcount;
- remove a retained page from the free queue unless it is actually allocated;
- create a second pin/reference mechanism;
- skip `_maybe_evict_cached_block()` for an allocated page;
- make an otherwise feasible allocation return `None`.

The selected implementation should perform a stable, non-destructive-to-
unselected-order selection: scan free pages only until enough non-retained pages
are found, remember skipped retained candidates, and fall back to those candidates
in original LRU order if required. Expected scan is O(requested pages + retained
pages encountered), not an unconditional O(total pool) stable partition.

---

## E. Gap analysis: design assumptions vs frozen source

| Design assumption | Frozen source / project reality | Consequence |
|---|---|---|
| Scheduler has FCFS/Priority and running/waiting | true, plus a separate `skipped_waiting` and blocked statuses | every ordering API must cover the two-queue merge |
| Priority heap iterator can provide top-K | ordered, but copies all N heap entries | add a bounded frontier `peek_n`; never slice `_heap` |
| Re-admission must be added everywhere | FCFS already prepends preempted requests; Priority does not | change only the semantics that are missing; preserve FCFS default |
| `get_computed_blocks()` can be a read-only probe | lookup core is read-only, public method can emit full-report events | factor an explicit `peek_computed_blocks()` with no events/stats/connectors |
| Prefix blocks are all full, uniform blocks | current code supports partial hashes, CoW, sparse managers, hybrid group sizes, EAGLE and Mamba | retention operates on returned physical IDs and ignores null blocks; no custom hash logic |
| Allocation only checks free count | it also removes skipped blocks, accounts for evictable hit blocks, watermark, reservations, CoW and full-ISL gate | no policy may duplicate allocation math |
| Freeing a request releases all its blocks | shared refs and deferred-free fences can prevent immediate reclaim | advanced reclaimability must define immediate vs eventual reclaim and initially fall back on deferred paths |
| BlockPool LRU is a deque | it is an intrusive doubly-linked `FreeKVCacheBlockQueue` | add the smallest queue primitive; preserve O(1) removal and link invariants |
| Watermark is a future feature | current source already has `SchedulerConfig.watermark` and waiting/preempted admission logic | reuse; do not invent another headroom mechanism |
| Prefix stats happen at lookup | current scheduler records local stats only after successful admission | speculative probes must not record stats |
| Base victim is always last running item | FCFS uses last; Priority uses max `(priority, arrival)` | default policy must reproduce both exactly |
| Existing P1 is binary anti-thrashing | code orders by raw `num_preemptions` | convert to explicit boolean tier or document a different policy; this plan chooses boolean |
| Existing P1 has stable final tie-break | equal cost/arrival falls back to running-list position | add request ID tie-break and integration coverage |
| Benchmark output count is tokenizer-token throughput | custom script counts streaming chunks | use it only for controlled pressure/preemption diagnostics; use built-in bench for final metrics |

---

## F. Unified architecture proposal

### F.1 Component boundaries

```text
Scheduler Core
  |
  +-- SchedulingDecisionPolicy
  |     +-- waiting_order_key(request)            Phase 1
  |     +-- select_preemption_victim(running)     Phase 1
  |     +-- choose_admission(candidates, ctx)     Phase 3
  |
  +-- SchedulingFeatureContext
  |     +-- cheap request fields                  Phase 1/3
  |     +-- lazy local prefix feature             Phase 3/4
  |     +-- lazy reclaimability                   Phase 5
  |     +-- invalidate KV-derived fields on mutation
  |
  +-- local PrefixDemandResolver                  Phase 2
        -> event-free KVCacheManager peek
        -> per-allocation memoized retention IDs

KVCacheManager / Coordinator
  +-- owns allocation feasibility and request block tables
  +-- transports allocation-scoped retention hint

BlockPool
  +-- owns stable soft-retention selection
  +-- owns real eviction/refcount/hash/event/metrics mutation
```

Do not implement registries, entry points, dynamic plugins, out-of-tree loading,
or lifecycle callbacks.

### F.2 Policy hierarchy

Default mode:

- byte/behavior-compatible FCFS and Priority queue order;
- default victim selection;
- default LRU;
- no candidate prefix probes;
- no new queue scans.

Base KV-aware mode:

```text
preemption victim key:
  1. worst user-priority tier (largest priority value)
  2. fresh victim before previously preempted victim (binary protection)
  3. minimum num_computed_tokens
  4. latest arrival
  5. request_id deterministic tie-break

waiting order:
  1. user priority
  2. PREEMPTED/resume before cold
  3. arrival time
  4. request_id
```

The binary protection tier is retained because the existing controlled pressure
experiment exposed repeat-victim thrashing. It is not allowed to cross user
priority, and it must not order by the exact number of past preemptions.

Advanced admission mode, within a bounded window and one user-priority tier:

```text
  1. aged requests (then base order among aged)
  2. resumed requests
  3. smaller remaining prefill work
  4. base queue order
```

No weighted universal score is planned.

### F.3 Minimal typed configuration

Keep existing `SchedulerConfig`/`EngineArgs` plumbing. Add fields only when their
phase lands:

| Phase | Field | Default | Meaning |
|---|---|---|---|
| existing/1 | `preemption_policy: Literal["default", "recompute_aware"]` | `default` | selects victim policy; optimized value also enables same-tier resume ordering |
| 2 | `prefix_cache_eviction_policy: Literal["lru", "waiting_queue_aware"]` | `lru` | enables soft retention |
| 2/3 | `kv_aware_candidate_window: int` | 8 (tuned from the proposed 16) | common upper bound for retention/admission candidates |
| 3 | `admission_policy: Literal["default", "cache_affinity"]` | `default` | enables bounded cache-affinity ordering |
| 3 | `kv_aware_aging_threshold_s: float` | 30.0 s（暂定） | promotes starved same-tier candidates |

Do not expose block-selection internals or feature-cache implementation details
as CLI flags. If review finds `preemption_policy` too narrow a name for its
resume behavior, rename it once in Phase 1 before more public surface is added;
do not keep two aliases indefinitely during this project branch.

### F.4 Feature lifetime and invalidation

Cheap Request fields can be read throughout an iteration. KV-derived fields
cannot be blindly cached for the entire iteration because an earlier admission,
preemption, free or eviction changes the cache.

Phase 4 context must therefore maintain a KV-state generation:

- a prefix/reclaimability result is memoized only for the current generation;
- Scheduler invalidates KV-derived values immediately after every
  `allocate_slots()` transaction that can mutate state, preemption/free, reset or
  external invalidation;
- repeated policy consumers before the next mutation reuse the result;
- cheap fields remain available without copying Request state.

This preserves the intended per-iteration object while avoiding stale prefix
decisions.

---

## G. Exact Phase 1 patch: stable scheduler foundation

### G.1 Scope

Phase 1 contains exactly:

1. stabilize the lightweight decision-policy abstraction already present;
2. stabilize recompute-aware victim selection;
3. add resume-aware Priority re-admission;
4. add integration tests for state/list/heap correctness;
5. rerun the existing pressure diagnostic and report both gains and regressions.

It does not touch `KVCacheManager`, `BlockPool`, CUDA/Triton code or admission
cache affinity.

### G.2 Production files and exact changes

#### `vllm/v1/core/sched/policy.py`

Retain the current factory and decision-only responsibility, then:

- define a small common policy type/protocol used by Scheduler and queues;
- keep `DefaultSchedulingDecisionPolicy.select_preemption_victim()` exactly
  equivalent to the frozen base;
- change the optimized anti-thrashing component from raw
  `request.num_preemptions` to a boolean `request.num_preemptions > 0` tier;
- add explicit final request-ID tie-break;
- expose `waiting_order_key(request)` for Priority queues:
  - default: `(priority, arrival_time, request_id, id(request))`;
  - optimized: `(priority, not_is_resumed, arrival_time, request_id, id(request))`;
- do not mutate Request, queues, status or KV state.

The exact optimized victim behavior must be tested as a lexicographic hierarchy,
not described as a mathematical score.

#### `vllm/v1/core/sched/request_queue.py`

- Preserve the public `RequestQueue` operations used by Scheduler.
- Replace direct `Request` heap entries with an internal immutable heap item
  whose comparison key is captured when the request is enqueued.
- Let `create_request_queue()` accept the selected Priority ordering key.
- Make `peek_request`, `pop_request`, iteration and remove operations return/use
  the underlying Request identity.
- Add a queue-level `order_key(request)` or equivalent head comparison method so
  Scheduler never falls back to `Request.__lt__` when optimized ordering is on.
- Keep `Request.__lt__` unchanged for compatibility with callers outside the
  queue.
- Do not add bounded `peek_n` yet unless it is needed to avoid duplicating the
  heap-item representation; admission selection remains Phase 3.

Snapshot keys are essential: a mutable field cannot corrupt a heap that stores
the old ordering. `num_preemptions` changes before optimized re-enqueue, so the
new key is captured correctly.

#### `vllm/v1/core/sched/scheduler.py`

- Construct `decision_policy` before queues, as current code does.
- Add `_create_waiting_queue()` and use it for `waiting`, `skipped_waiting` and
  the per-step `step_skipped_waiting` queue.
- Update `_select_waiting_queue_for_scheduling()` in Priority mode to compare the
  configured queue key, not raw `Request.__lt__`.
- Keep all victim removal, cursor adjustment, scheduled-work rollback and
  `_preempt_request()` calls in Scheduler.
- Keep `_preempt_request()`'s mutation order: free bookkeeping, set PREEMPTED,
  reset progress, increment count, then enqueue.
- Add no new direct Request mutation for ordering.

The existing arbitrary-victim removal logic at `scheduler.py:672-711` must remain
covered. It fixes the case where a selected victim precedes the current
`req_index` and would otherwise cause a running request to be skipped.

#### `vllm/config/scheduler.py` and `vllm/engine/arg_utils.py`

- Retain typed opt-in configuration and CLI plumbing.
- Clarify that the optimized mode is a coordinated preemption/re-admission mode,
  or perform the one-time field rename discussed in F.3.
- Default remains baseline-compatible.
- No graph hash factor is required because queue decisions do not change the
  compiled model graph.

#### `vllm/v1/request.py`

No planned modification. In particular, do not add resume ordering to
`Request.__lt__`, because that would change default Priority semantics globally
and cannot be toggled per Scheduler.

### G.3 Phase 1 tests

Prefer extending existing nearby files over creating more one-off suites. The
already-created `tests/v1/core/test_preemption_policy.py` is justified as a
focused policy unit suite; Scheduler integration belongs in existing scheduler
tests.

| Test file | Proposed test | Initial state | Operation | Expected result |
|---|---|---|---|---|
| `tests/v1/core/test_preemption_policy.py` | `test_default_fcfs_victim_compatibility` | three running requests | call default selector | newest/list-tail victim |
| same | `test_default_priority_victim_compatibility` | mixed priority and arrival | call default selector | largest priority, then latest arrival |
| same | `test_recompute_victim_same_tier` | fresh A=128, fresh B=2048 | optimized selector | A victim |
| same | `test_user_priority_precedes_recompute` | high-priority cheap vs low-priority expensive | optimized selector | low user-priority request victim |
| same | `test_resume_protection_is_binary` | same tier; preemption counts 1 and 3 with different costs | optimized selector | both are same resume tier; recompute cost decides |
| same | `test_fresh_victim_before_resumed` | same priority; fresh expensive vs resumed cheap | optimized selector | fresh victim, documenting anti-thrashing hierarchy |
| same | `test_victim_final_tie_is_deterministic` | equal priority/state/cost/arrival, different IDs and list order | selector twice | same ID wins independent of running-list order |
| `tests/v1/core/test_scheduler.py` | `test_recompute_policy_removes_earlier_running_victim_without_skip` | arbitrary victim occurs before current cursor | run schedule under block pressure | later request still scheduled; budgets/output maps consistent |
| same | `test_recompute_policy_preemption_state_transition` | one selected running victim with KV | trigger allocation failure | RUNNING -> PREEMPTED; KV refs released by official path; computed=0; count+1; queued once |
| same | `test_resume_order_same_priority` | optimized Priority queue with cold older and PREEMPTED newer | schedule one slot | preempted request admitted first |
| same | `test_resume_does_not_cross_user_priority` | high-priority cold and low-priority preempted | schedule one slot | high-priority cold admitted first |
| same | `test_default_priority_readmission_unchanged` | default mode with preempted and cold requests | inspect/schedule queue | frozen-base `(priority, arrival, id)` behavior |
| same | `test_priority_heap_key_snapshot` | enqueue optimized request, mutate only fields not in key; requeue after real preemption for changed resume key | iterate/pop | heap remains ordered; changed resume field only takes effect on re-enqueue |
| `tests/v1/core/test_priority_preemption_bug.py` | existing regression | A/B/C mixed priority and arbitrary victim | run throttle-prefill schedule | C is not silently skipped |
| `tests/v1/core/test_priority_scheduler_random.py` | existing blast matrix | random scheduling with/without APC/spec decode | 20k iterations | no invalid output/state invariant |

### G.4 Commands and Gate 1

After environment setup:

```bash
/root/miniconda3/envs/vllm-dev/bin/python -m pytest tests/v1/core/test_preemption_policy.py -v
/root/miniconda3/envs/vllm-dev/bin/python -m pytest tests/v1/core/test_priority_preemption_bug.py -v
/root/miniconda3/envs/vllm-dev/bin/python -m pytest tests/v1/core/test_scheduler.py -k \
  'preempt or priority or resume or waiting_queue' -v
/root/miniconda3/envs/vllm-dev/bin/python -m pytest tests/v1/core/test_worker_slot_overflow.py -v
/root/miniconda3/envs/vllm-dev/bin/python -m pytest tests/v1/core/test_deferred_block_free.py -v
/root/miniconda3/envs/vllm-dev/bin/python -m pytest tests/v1/core/test_priority_scheduler_random.py -v
pre-commit run ruff-check --files \
  vllm/v1/core/sched/policy.py \
  vllm/v1/core/sched/request_queue.py \
  vllm/v1/core/sched/scheduler.py \
  vllm/config/scheduler.py \
  vllm/engine/arg_utils.py \
  tests/v1/core/test_preemption_policy.py \
  tests/v1/core/test_scheduler.py
```

Gate 1 passes only if:

- default behavior tests pass;
- strict user priority and binary protection are explicit;
- resume ordering passes for both FCFS and Priority semantics;
- arbitrary victim index does not skip or double-schedule a request;
- no request is present twice across running/waiting queues;
- random preemption blast passes;
- pressure benchmark has no errors and its raw result JSON is retained.

### G.5 Estimated Phase 1 size

Relative to the current project HEAD:

| Area | Production delta | Test delta |
|---|---:|---:|
| policy stabilization | 20-45 lines | 35-60 lines |
| keyed Priority queue + queue factory | 55-95 lines | 50-90 lines |
| Scheduler queue integration | 20-45 lines | 70-120 lines |
| config/docs cleanup | 5-20 lines | 0-15 lines |
| Total | about 100-205 lines | about 155-285 lines |

Cumulative business production code relative to frozen upstream should remain
roughly 220-350 lines for Phase 1. If the queue abstraction pushes cumulative
production changes past about 450 lines, first drop convenience APIs; do not
introduce a general scheduler plugin framework.

---

## H. Exact Phase 2 patch: waiting-queue-informed prefix retention

### H.1 Supported P0 surface

Initial supported path:

- single-node/local GPU;
- decoder-only generation;
- local prefix caching enabled;
- unitary full-attention model first for integration evidence;
- generic BlockPool behavior remains correct for hybrid physical IDs;
- connector lookup is not used for retention probes.

Hybrid/SWA/Mamba correctness tests must still pass. If the event-free local peek
cannot be made semantically reliable for a specific coordinator, that path
returns no hint and uses LRU; it must not guess.

### H.2 Production files and exact changes

#### `vllm/v1/core/kv_cache_manager.py`

Add an explicit method, final name chosen to match local style, with this
contract:

```python
peek_computed_blocks(request: Request) -> KVCacheBlocks
```

or a small result containing blocks and exact local token count.

Requirements:

- use the same coordinator hit semantics and `num_tokens - 1` cap as real local
  admission;
- return empty when caching/lookup is disabled;
- do not emit cache events;
- do not record hit/query stats;
- do not call a connector;
- do not touch/refcount blocks;
- do not mutate `request.shared_prefix_boundary`;
- factor shared lookup code so the real method and peek cannot drift.

Add an allocation-scoped lazy hint object (private is sufficient) that memoizes
`set[int]` once. Extend `allocate_slots()` with an optional retention context;
default `None` must take the original path. The same context is passed through
all physical allocations in that transaction so hybrid groups do not re-probe
the waiting queue.

#### `vllm/v1/core/kv_cache_coordinator.py`

- Thread the optional allocation-scoped retention context through
  `allocate_new_computed_blocks()` and `allocate_new_blocks()`.
- Do not add Scheduler imports or queue knowledge.
- Preserve the existing two-phase “touch every local hit before allocating
  external blocks” ordering; this is a correctness constraint from issue
  #33775 and must not be disturbed.

#### `vllm/v1/core/single_type_kv_cache_manager.py`

- Pass the context to every `BlockPool.get_new_blocks()` call that may consume
  cached-free pages, including external-computed allocation and partial-hit CoW.
- Keep manager-specific block math unchanged.
- Do not rank or interpret waiting requests here.

#### `vllm/v1/core/kv_cache_utils.py`

Add the smallest linked-queue primitive needed for stable soft-retention
selection. Its contract must be independently testable:

```text
select/remove N free blocks while avoiding retained IDs when possible;
fallback to retained IDs in original queue order;
leave every unselected block in the same relative order;
maintain num_free_blocks and all prev/next links.
```

Do not permanently move retained blocks to the tail and do not rebuild the full
queue on every allocation.

#### `vllm/v1/core/block_pool.py`

Extend `get_new_blocks()` with an optional allocation-scoped retention context:

1. if feature is disabled, call the current `popleft_n()` fast path;
2. inspect only the first N free candidates;
3. if they are all unhashed, call current fast path without resolving waiting
   demand;
4. when cached eviction is actually possible, resolve the retained set once;
5. select non-retained pages first with retained fallback;
6. for every selected page, still call `_maybe_evict_cached_block()`, increment
   refcount and update the existing metrics path.

Retention applies only when `block.block_hash is not None`, `ref_cnt == 0` and
the block is truly in the free queue. An ID hint cannot protect an active block
or the null block.

#### `vllm/v1/core/sched/scheduler.py`

Add a private bounded candidate iterator that reflects actual near-term order:

- FCFS: ready/step-skipped order consistent with the current two-queue logic;
- Priority: merge queue heads/order keys safely;
- skip requests that cannot use local prefix cache;
- cap request count by `kv_aware_candidate_window`;
- no full waiting-queue scan.

For every local `allocate_slots()` call that may consume cached-free pages
(running growth and waiting admission), pass a lazy closure/context only when
`prefix_cache_eviction_policy == "waiting_queue_aware"`. For waiting admission,
exclude the request currently being allocated from the future-demand scan: its
actual hit blocks are adopted/touched before new pages are selected and do not
need a second speculative probe. The closure:

- gets at most W near-head candidates;
- calls event-free local peek;
- unions non-null physical block IDs;
- does not mutate candidates or scheduler queues;
- memoizes once for that allocation transaction.

Do not eagerly resolve the closure at scheduler-step start.

#### `vllm/config/scheduler.py` and `vllm/engine/arg_utils.py`

Add only the Phase 2 fields in F.3, with LRU/default behavior unless explicitly
enabled. Validate window > 0 when waiting-aware mode is selected.

### H.3 Phase 2 deterministic tests

| Test file | Proposed test | Initial state | Operation | Expected result |
|---|---|---|---|---|
| `tests/v1/core/test_prefix_caching.py` | `test_soft_retention_eviction_order` | free cached LRU A B C D E; retain B,D | allocate five one at a time or selection API | A,C,E,B,D |
| same | `test_soft_retention_falls_back` | N requested > non-retained free pages | allocate N | allocation succeeds and consumes retained in original order |
| same | `test_soft_retention_preserves_unselected_order` | mixed cached/uncached queue, partial allocation | allocate with hint | relative order and links of every unselected page unchanged |
| same | `test_soft_retention_ignores_active_and_null_ids` | hint includes active and null IDs | allocate | no corruption; only free cached pages affected |
| same | `test_retention_disabled_is_exact_lru` | same LRU, no context | allocate | frozen-base IDs/events/metrics order |
| same | `test_retention_resolver_not_called_on_uncached_fast_path` | first N free pages unhashed | allocate with mock resolver | allocation succeeds; resolver call count zero |
| same | `test_retention_resolver_called_once_per_allocation` | hybrid/multiple manager allocations under cached pressure | one `allocate_slots()` | resolver call count one |
| same | `test_prefix_peek_is_read_only` | cached request with refcounts, free order, hash map, stats and event queue snapshotted | call peek | all snapshots unchanged; returned hit IDs correct |
| same | `test_prefix_peek_full_report_emits_no_event` | events enabled and report mode full | call peek then real lookup/admission | peek emits none; real path retains current event behavior |
| same | `test_prefix_peek_does_not_touch_connector` | connector mock present | local peek | zero connector calls |
| same | `test_retained_eviction_uses_official_accounting` | events/metrics enabled cached victim | fallback allocation | selected retained page loses hashes and emits normal removal/metrics exactly once |
| `tests/v1/core/test_scheduler.py` | `test_waiting_prefix_demand_retains_near_head_hit` | waiting X hits B; a running allocation needs cached page | schedule allocation | non-demand page evicted before B; X's later local hit survives |
| same | `test_retention_candidate_window_is_bounded` | >W waiting requests, unique cached prefixes | trigger resolver | exactly first eligible W are probed; no request outside W retained |
| same | `test_retention_does_not_change_admission_order` | same queue with LRU vs retention mode | schedule without capacity difference | admitted request order identical |
| same | `test_retention_allocation_feasibility` | all free cached pages are retained and allocation needs them | schedule | same success/failure as LRU baseline; retained pages evicted as fallback |

Also rerun existing prefix suites covering:

- computed hit blocks not evicted before touch;
- duplicate hash entries;
- hybrid combinations and EAGLE;
- partial tail/CoW;
- event generation and prefix reset;
- deferred block free.

### H.4 Commands and Gate 2

```bash
/root/miniconda3/envs/vllm-dev/bin/python -m pytest tests/v1/core/test_prefix_caching.py -v
/root/miniconda3/envs/vllm-dev/bin/python -m pytest tests/v1/core/test_single_type_kv_cache_manager.py -v
/root/miniconda3/envs/vllm-dev/bin/python -m pytest tests/v1/core/test_scheduler.py -k \
  'prefix or retention or preempt or waiting' -v
/root/miniconda3/envs/vllm-dev/bin/python -m pytest tests/v1/core/test_deferred_block_free.py -v
/root/miniconda3/envs/vllm-dev/bin/python -m pytest tests/v1/core/test_swa_inflight_window_free.py -v
pre-commit run ruff-check --files <all-phase-2-changed-files>
```

Gate 2 passes only if the read-only and fallback properties are proved, default
LRU remains unchanged, and no full waiting scan occurs on the uncached/free fast
path.

### H.5 Estimated Phase 2 size and stop condition

| Area | Production delta | Test delta |
|---|---:|---:|
| event-free peek/factoring | 25-55 | 50-90 |
| allocation hint threading | 35-75 | 30-60 |
| stable queue selection + BlockPool | 55-100 | 90-150 |
| Scheduler resolver/config | 45-85 | 90-150 |
| Total | about 160-315 | about 260-450 |

If production work exceeds about 350 lines because every coordinator needs
custom behavior, reduce P0 to the unitary local full-attention path with explicit
fallback. If an event-free lookup cannot be guaranteed, stop Phase 2 rather than
using real admission lookup as a probe.

---

## I. Phase 3+ integration points

### I.1 Phase 3: bounded cache-affinity admission + aging

> 实施状态（2026-08-28）：已完成并通过 Gate 3 correctness。实现采用
> `W=8`、aging `30 s`，default fast path 不探测 prefix；Priority top-W
> 使用 heap frontier 和冻结入队键，中间候选为 `O(log N)` 删除。单次固定工作量
> A/B 显示压力场景吞吐约 `+1.68%`、TTFT mean 约 `-5.75%`，无抢占场景
> wall time 约 `-0.08%`；但压力场景 prefix hits 约 `-2.61%` 且 probe 放大
> 明显，性能 Gate 留待 Phase 4 优化和多轮配对复测。完整证据见
> `PHASE3_IMPLEMENTATION_REPORT_CODEX.md`。

Add only after Gate 2.

Files/functions:

- `request_queue.py`: `peek_n(W)` and remove-chosen-candidate support using the
  heap frontier algorithm described in C.3.
- `policy.py`: choose one candidate from a same-user-priority candidate window.
- `scheduler.py` waiting admission section: get candidates, resolve readiness
  constraints, rank one candidate, then immediately use its local prefix result
  in the normal admission/allocation path.
- `scheduler.py`: rebuild/revalidate after every successful allocation; never
  keep a KV-derived ranking across a cache mutation.
- `scheduler.py`: preserve skipped/blocked/LoRA/connector rules. A candidate
  that is not currently schedulable is moved through existing skip semantics,
  not silently bypassed forever.

Candidate selection:

1. obtain first W requests in true queue order;
2. never cross the head's user-priority tier;
3. requests whose wait exceeds the aging threshold are ordered by base order;
4. otherwise resume requests precede cold requests;
5. compare `remaining_prefill_tokens = request.num_tokens - local_cached_tokens`;
6. final tie-break is the base queue position.

Required tests:

| Test file | Proposed test | Initial state | Operation | Expected result |
|---|---|---|---|---|
| `tests/v1/core/test_scheduler.py` | `test_cache_affinity_prefers_less_remaining_prefill` | same-priority A cold, B warm, both inside W | schedule one free slot | B admitted first |
| same | `test_cache_affinity_respects_user_priority` | high-priority cold A, low-priority warm/resumed B | schedule one slot | A admitted first |
| same | `test_cache_affinity_does_not_cross_window` | warm request at index W, cold requests in `[0,W)` | select candidate | outside request cannot jump window |
| same | `test_cache_affinity_aging_prevents_starvation` | cold head plus repeated same-tier warm arrivals; controlled wall-clock arrivals | advance time/schedule repeatedly | cold request becomes aged and is admitted |
| same | `test_default_admission_does_not_probe_prefix` | cache available but admission policy default | schedule | base order; resolver call count zero |
| same | `test_failed_affinity_admission_preserves_queue` | chosen candidate cannot allocate | run waiting admission | no duplicate/removal corruption and no prefix stats recorded twice |
| `tests/v1/core/test_scheduler.py` or focused queue suite if one exists by then | `test_priority_peek_n_matches_pop_order` | randomized Priority heap | compare `peek_n(W)` with pop order from a copy | identical first W without mutating source |

Go/no-go: no implementation that copies/scans the entire Priority heap is
accepted as “bounded.”

### I.2 Phase 4: lazy SchedulingFeatureContext

> 实施状态（2026-08-28）：已完成 correctness Gate。上下文仅在 Phase 2/3
> KV-aware consumer 开启时创建；昂贵 prefix feature 按 Request 身份和 KV
> generation 延迟解析、代内复用，并在 allocation/free/eviction/reset 等 KV
> mutation 后失效。压力 A/B 中 154,192 次 feature read 有 12,249 次命中
> memoization，实际 KV resolver 调用减少 7.944%；但每次成功 admission 仍约有
> 102.15 次候选 probe，因此 Phase 4 解决了安全代内重复解析，没有解决跨 KV
> mutation 的反复失败选择。完整结果见
> [`PHASE4_IMPLEMENTATION_REPORT_CODEX.md`](PHASE4_IMPLEMENTATION_REPORT_CODEX.md)。

Introduce the context only after there are two real consumers of expensive
prefix data. It is a derived view, never Request state.

Suggested responsibilities:

```text
cheap(request): priority, arrival, num_preemptions, computed tokens
lazy(request, CACHED_PREFIX): event-free local hit at current KV generation
derived(request, REMAINING_PREFILL): num_tokens - cached prefix
lazy(request, RECLAIMABLE_BLOCKS): Phase 5 only
invalidate_kv_features(): after allocation/free/eviction/reset
```

Required tests:

| Test file | Proposed test | Initial state | Operation | Expected result |
|---|---|---|---|---|
| `tests/v1/core/test_scheduler.py` | `test_feature_context_default_fast_path` | all KV-aware consumers off | schedule | prefix resolver never called |
| same | `test_feature_context_memoizes_within_generation` | one request, two consumers, no KV mutation | read prefix feature twice | one resolver call, identical result |
| same | `test_feature_context_invalidates_after_kv_mutation` | cached feature at generation g | allocate/free then read again | generation advances and resolver runs again |
| same | `test_feature_context_cheap_fields_do_not_touch_kv` | request with cheap metadata | read priority/computed/resume/age | zero KV manager calls |
| same | `test_feature_context_is_not_request_state` | one completed scheduler iteration | inspect Request/context references | no feature cache persisted on Request |

### I.3 Phase 5a: priority/demand-aware retention

> 实施状态（2026-08-28）：correctness Gate 已完成。保留
> `waiting_queue_aware` 作为 Phase 2 二值对照，新增显式实验策略
> `priority_aware`；底层统一为 `block_id -> tier`，按 tier 0→3 淘汰，tier
> 内保持 LRU，并回退遍历所有 tier。两对反向顺序压力 A/B 的平均 wall time
> 为 195.645→191.822 s（-1.954%），吞吐 +1.970%；但 preemption/prefix-hit
> 的逐对方向不一致，且 TTFT p50 均值退化，因此只验收 correctness，不宣称
> 稳定 KV 效率收益。详见
> [`PHASE5A_IMPLEMENTATION_REPORT_CODEX.md`](PHASE5A_IMPLEMENTATION_REPORT_CODEX.md)。

Evolve `set[int]` into a small `block_id -> retention_tier` hint:

```text
tier 0: no near-term demand
tier 1: normal near-head demand
tier 2: resumed near-head demand
tier 3: high-user-priority near-head demand
```

Evict lower tiers first; preserve LRU within each tier; always fall back through
all tiers. Define “high user priority” relative to the candidate set/configured
priority semantics, not a hard-coded magic numeric value.

### I.4 Phase 5b: reclaimable-KV estimate

> 实施状态（2026-08-29）：只读信息层 correctness Gate 已完成。新增
> `KVCacheManager.estimate_reclaimable_blocks(request)`，按请求在所有 group
> block table 及 request-scoped partial-tail pins 中持有的引用重数，与物理
> block 当前 `ref_cnt` 做相等比较；只计算无外部引用的唯一非 null 物理页。
> estimator 不接入 `SchedulingFeatureContext`、preemption policy 或 Phase 5a
> retention heuristic，并明确只表示 official free 后的 eventual pool return。
> deferred-free 测试证明 fence 完成前这些页并不立即可用于 allocation retry。
> 详见
> [`PHASE5B_IMPLEMENTATION_REPORT_CODEX.md`](PHASE5B_IMPLEMENTATION_REPORT_CODEX.md)。

Add a read-only `estimate_reclaimable_blocks(request)` in `KVCacheManager` only
after refcount semantics are proven by tests.

For each unique non-null physical block across group tables, count it if the
number of references owned by the request equals the block's current refcount;
shared blocks with other requests do not count. Account for repeated physical
IDs, partial-tail pins and CoW ownership.

Critically, distinguish:

- eventual reclaimable pages after official free;
- immediately reusable pages in this allocation retry.

When deferred block free is enabled, the P0 advanced policy must fall back to
recompute-only selection unless it can prove pages return before the retry. Do
not claim an eventual page fixes an immediate allocation deficit.

Advanced victim hierarchy:

```text
1. worst user-priority tier
2. candidates that can satisfy/currently reduce allocation deficit
3. binary resume protection
4. minimum recompute cost
5. stable tie-break
```

Tests must include a shared prefix where allocated block count is greater than
reclaimable count, duplicated references, null blocks, hybrid groups and the
deferred-free fallback.

| Test file | Proposed test | Initial state | Operation | Expected result |
|---|---|---|---|---|
| `tests/v1/core/test_prefix_caching.py` | `test_reclaimable_excludes_shared_prefix` | A/B share prefix; A has private suffix | estimate A | only A-exclusive physical pages counted |
| same | `test_reclaimable_counts_request_multiplicity_once` | one physical ID appears multiple times in request bookkeeping where supported | estimate then official free | estimate matches pages that actually enter free queue |
| same | `test_reclaimable_ignores_null_blocks` | sparse/hybrid table contains null padding | estimate | null ID not counted |
| same | `test_reclaimable_hybrid_groups_match_actual_free_delta` | request owns multiple group pages | snapshot free count, estimate, officially free | estimate equals supported-path free-count delta |
| `tests/v1/core/test_deferred_block_free.py` | `test_reclaimable_policy_falls_back_when_free_is_deferred` | in-flight write fence prevents immediate return | select victim/estimate | advanced feasibility is disabled or reports zero immediate pages; recompute fallback used |
| `tests/v1/core/test_preemption_policy.py` | `test_advanced_victim_uses_feasibility_before_recompute` | same priority; only one candidate reduces deficit, another has lower recompute | select victim | feasible candidate chosen, then recompute tie-break within feasible set |

### I.5 Phase 6: Scheduler CPU performance

Measure before optimizing. Use request counts 1/8/16/64/256 and report:

- default decision latency;
- enabled decision latency;
- candidate count;
- prefix resolver calls;
- KV-generation invalidations;
- allocations/temporary objects only if profiling identifies them.

Primary goals:

- default path approximately baseline;
- cost scales with W, not total waiting count;
- no duplicate expensive lookup in one valid generation;
- no retention resolver call without cached-block pressure.

Do not keep a perf patch without repeatable evidence.

### I.6 Phase 7: optional RTX 4090 KV-write kernel

Reconnaissance at this commit found:

```text
Scheduler block tables / slot mapping
  -> vllm/v1/worker/gpu/block_table.py
  -> attention backend do_kv_cache_update()
  -> FlashAttention backend on ordinary CUDA/SM89 by priority
  -> vllm/v1/attention/backends/fa_utils.py
  -> vllm._custom_ops.reshape_and_cache_flash
  -> torch.ops._C_cache_ops.reshape_and_cache_flash
  -> csrc/libtorch_stable/cache_kernels.cu
```

For ordinary non-MLA Qwen/Llama-like workloads on the observed RTX 4090, CUDA
backend priority selects FlashAttention when supported; SM89 falls back to FA2
for attention, while its KV write is the native custom
`reshape_and_cache_flash_kernel`. Explicit Triton attention uses
`triton_reshape_and_cache_flash`; other backends (FlashInfer, Flex, ROCm, MLA,
quantized modes) have their own dispatch variations.

Current native flash writer already:

- supports NHD and HND layouts via strides;
- supports scalar or per-head K/V scales;
- dispatches native and FP8 cache dtypes (NVFP4 is SM100+ only);
- uses vectorized alignment-aware copy with vector size 8 for 16-bit inputs and
  4 for 32-bit inputs;
- launches one CUDA block per token, up to 512 threads;
- computes block ID and offset from slot mapping at runtime.

Current Triton writer already specializes `num_heads`, `head_size` and
`block_size`, uses a smaller tile on pre-SM90 CUDA, and supports native/FP8
dtypes. Therefore “add vectorization” is not itself a valid future proposal;
any candidate must identify a measured gap.

Existing benchmarks:

```text
benchmarks/kernels/benchmark_reshape_and_cache.py
benchmarks/kernels/benchmark_reshape_and_cache_flash.py
tests/kernels/attention/test_cache.py
```

The flash benchmark compares CUDA and Triton for NHD/HND across powers-of-two
token counts, common head sizes, block sizes 16/32 and native/FP8 storage. Its
current timing synchronizes every iteration with `time.perf_counter`; before
drawing a kernel conclusion, use the repository kernel-microbenchmark rules:
warm compile/autotune, correctness first, isolate the op, use CUPTI/device timing
where available, report bytes/GB/s for this memory-bound operation, and record
GPU/dtype/layout/shape/commit/env.

Go/no-go:

- run native CUDA and Triton baselines on actual model-representative KV head
  counts (not only benchmark default 128 heads), head size 128, block size 16,
  BF16/FP16 and token counts reflecting decode and prefill;
- inspect PTX/SASS/profiler only after a reproducible gap;
- candidate may be an SM89/common-shape gated fast path with generic fallback;
- require correctness against existing tests and repeatable gain over multiple
  representative shapes;
- if no gain or any common-shape regression exists, do not merge a kernel
  change and list reconnaissance as future work only.

---

## J. Feature cost map

Complexities below use W for candidate window, B for request block/hash count,
G for KV groups and R for physical request references.

| Feature | Source | Cost at current commit | Side effects | Class | Lazy? |
|---|---|---:|---|---|---|
| user priority | `Request.priority` | O(1) | none | cheap | no |
| arrival/base age input | `Request.arrival_time` | O(1) | none | cheap | no |
| waiting age | `time.time() - Request.arrival_time` in the same clock domain | O(1) | none; clock read | cheap | no |
| computed tokens | `Request.num_computed_tokens` | O(1) | none | cheap | no |
| resume state | `status == PREEMPTED` or `num_preemptions > 0` | O(1) | none | cheap | no |
| preemption count | `Request.num_preemptions` | O(1) | none on read | cheap | no |
| prompt/total tokens | Request properties/lists | O(1) length | none | cheap | no |
| true FCFS top-W | deque `islice` | O(W) | none | bounded | yes, only policy use |
| current Priority iteration | heap copy + pops | O(N + N log N) to exhaust; O(N + W log N) even with `islice` | temp copy | expensive/unbounded | do not use |
| proposed Priority top-W | heap frontier | O(W log W), O(W) temp | none | bounded | yes |
| local cached prefix | coordinator `find_longest_cache_hit` | full attention about O(min(B, hit+1)); sparse/hybrid may scan/reconcile more, multiplied by group/spec behavior | core lookup none; public `get_computed_blocks` may emit full-report events | expensive | yes, event-free API |
| cached prefix tokens | prefix result | lookup cost then O(1) | same caveat | expensive | yes |
| remaining prefill | request tokens - cached prefix | O(1) after prefix feature | none | derived | yes through dependency |
| retained physical IDs | union peeked group blocks for <=W requests | O(sum hit blocks), bounded by probed candidates but potentially long prefixes | none in resolver | expensive | only under cached pressure |
| allocated block IDs | coordinator tables | O(R) to materialize | none | medium | yes |
| reclaimable blocks | group/pin table + refcount/multiplicity scan | O(R) expected with a temporary counter/set | none if correctly implemented | expensive | yes, Phase 5 |
| free block count | `BlockPool.get_num_free_blocks` | O(1) | none | cheap | no |
| first N free-page cache state | intrusive queue walk | O(N) | none | cheap for requested N | only with retention context |
| soft-retention block selection | queue walk until N non-retained or exhaustion | O(N + retained encountered), worst O(free pages) only when fallback requires it | removes selected queue nodes | pressure-only | yes |

---

## K. Correctness invariants

### K.1 User priority

- Smaller numeric value remains higher user priority.
- KV/recompute/resume/aging heuristics may only reorder within an allowed user
  priority tier.
- A low-priority resumed/warm request never preempts or jumps a high-priority
  cold request.

### K.2 Queue and heap

- Never treat heap array layout as sorted order.
- Heap comparison uses immutable insertion-time keys.
- Any field that changes ordering takes effect only on remove/re-enqueue.
- Every Scheduler-created waiting/skipped/transient queue uses the same policy
  key.
- A request exists in exactly one logical location: running, one waiting queue,
  or finished/connector-owned transition state.
- Arbitrary victim removal updates `req_index` and all same-step budget/output
  bookkeeping exactly once.

### K.3 Request status and output

- Only RUNNING requests can enter `_preempt_request()`.
- Preemption uses existing stale-output, spec-token, encoder and worker-reset
  behavior.
- `num_preemptions` increments once per real preemption.
- A resume is identified from existing state; no duplicate persistent source of
  truth is added.

### K.4 KV references and sharing

- Only BlockPool/manager official paths change refcounts.
- A lookup/probe never touches a block.
- Shared blocks do not become free until all owning refs are released.
- Null blocks are never retained, freed or counted reclaimable.
- Partial-hit CoW source/destination retention and deferred fences remain intact.

### K.5 Hashes, LRU and free queue

- Every allocated cached-free page passes `_maybe_evict_cached_block()`.
- All hash aliases are removed by existing helpers; no policy edits hash maps.
- `BlockRemoved`/`BlockStored` behavior remains on real operations only.
- Soft retention preserves relative LRU order within equal tiers.
- Unselected queue links and `num_free_blocks` remain consistent.
- Retained pages are fallback candidates, never pins.

### K.6 Metrics/events/connectors

- Speculative prefix probes do not record queries/hits, emit events or call
  connectors.
- Real successful admission records prefix stats once through the current path.
- Real eviction updates metrics/events once through the current path.
- Unsupported connector/hybrid paths fall back to default behavior rather than
  inventing approximate remote demand.

### K.7 Allocation feasibility and fast paths

- With identical free/request state, enabling retention cannot turn a successful
  allocation into `None`.
- Default policy/LRU execute the original path without closures, candidate
  objects or prefix scans.
- Waiting-aware mode does not resolve demand when the requested pages can be
  satisfied by the leading uncached free pages.
- Full-sequence admission, watermark, reserved blocks and manager block math
  remain owned by `KVCacheManager`.

---

## L. End-to-end test and evaluation plan

### L.1 Test design answers

- Module purpose: improve only decision points under KV pressure while core
  allocation/state machinery remains authoritative.
- I/O contract: policy returns request/order; peek returns read-only local hit;
  retention returns a soft preference; Scheduler/BlockPool execute mutation.
- Failures guarded: priority inversion, victim-loop skip, heap corruption,
  request duplication, refcount/hash/event corruption, allocation false failure,
  starvation and hot-path blow-up.
- Cheapest levels: pure policy/queue/block-pool unit tests first; existing
  scheduler integration tests second; one 4090 serving benchmark last.

### L.2 Regression layers

1. Pure deterministic policy and queue tests.
2. Existing `test_scheduler.py` focused selection with real Request/KV manager.
3. Prefix/BlockPool and hybrid manager suites.
4. Random priority scheduling blast.
5. Existing built-in offline/online prefix workloads.
6. Controlled 4090 baseline/ablation.

### L.3 Serving experiment matrix

Use the same model, server flags, prompt seed, request rate and cache budget for
every cell:

| Cell | Preemption/resume | Retention | Admission |
|---|---|---|---|
| A baseline | default | LRU | default |
| B scheduler | recompute-aware | LRU | default |
| C cache | default | waiting-aware | default |
| D base full | recompute-aware | waiting-aware | default |
| E advanced | recompute-aware | demand-aware | cache-affinity + aging |

The existing diagnostic launch recipe should be preserved with exact commit and
server log, including `--num-gpu-blocks-override 1250`, but final metrics should
come from a built-in workload such as:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.75 \
  --num-gpu-blocks-override 1250 \
  <policy flags for the matrix cell>

vllm bench serve \
  --backend openai \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 900 \
  --prefix-repetition-suffix-len 64 \
  --prefix-repetition-num-prefixes 8 \
  --prefix-repetition-output-len 512 \
  --num-prompts 192 \
  --max-concurrency 48 \
  --seed 42 \
  --save-result
```

Confirm exact CLI availability with the current environment before treating the
example as a recorded command. Do not silently substitute a different model or
base commit.

`benchmarks/benchmark_prefix_caching.py` is useful for offline APC sanity and
elapsed-time comparison, but it does not expose the same online concurrency and
latency breakdown. Use fixed random prompts (`temperature=0`, fixed seed) rather
than a new dataset.

### L.4 Metrics and interpretation

Serving:

- request/output-token throughput from the built-in benchmark;
- TTFT mean/p50/p90/p99;
- ITL/TPOT mean and percentiles.

Work efficiency:

- preemption count;
- GPU prefix cache hits/queries;
- recomputed tokens if available without a large instrumentation patch;
- resume delay and retained-hit/eviction counts only if existing stats or tiny
  scoped counters can provide them.

Scheduler overhead:

- decision latency;
- candidate count;
- prefix/reclaimability resolver calls;
- default fast-path overhead.

Report pressure and no-pressure workloads. Expected claim shape is:

```text
no pressure: approximately baseline
KV pressure + prefix reuse: less wasted work and/or better latency/throughput
```

Do not claim overall improvement from the current diagnostic alone: its three
recorded runs show preemptions 135/135/96 and output-chunk throughput roughly
1705/1709/1710 per second, but optimized `itl_mean.p99` changed from about
1.342 s to 1.525 s. This is a mixed result and must be investigated, not hidden.

### L.5 Phase gates and stop rules

| Gate | Required evidence | Stop/fallback |
|---|---|---|
| 1 Scheduler | default + priority + recompute + resume + random blast pass | no KV feature work until stable |
| 2 Base retention | pure peek, stable order, fallback, fast path, hybrid regressions pass | fallback to unitary local full attention or stop |
| 3 Admission | bounded top-W, strict priority, aging/starvation tests | reject any O(total waiting) implementation |
| 4 Features | lazy once/generation, invalidation, no persistent duplicate truth | keep phase-local resolvers if abstraction is larger than benefit |
| 5 Advanced KV | shared/repeated/deferred reclaim tests | disable on unsupported hybrid/deferred path |
| 6 CPU perf | measured overhead and evidence-backed patch | document profile only if no bottleneck |
| 7 Kernel | correctness + repeated representative 4090 speedup | no kernel commit if gain is absent/mixed |

Global stop conditions:

- Phase 1/2 correctness is not green;
- a feature requires major Request state-machine changes;
- base functionality exceeds roughly 600 production lines without a clear
  reduction path;
- a candidate policy needs a whole waiting scan;
- read-only prefix semantics cannot be guaranteed;
- immediate reclaimability cannot be defined on the active path;
- default-path CPU regression is material;
- kernel evidence is not repeatable.

---

## M. Commit and review sequence

Continue on `project/kv-aware-scheduling`; do not rewrite the already-pushed
history merely to make the project look linear. Each next commit is one logical
change with its tests/results in the commit or PR description.

Recommended sequence from the current HEAD:

```text
fix(scheduler): stabilize recompute-aware victim ordering
feat(scheduler): add resume-aware priority queue ordering
test(scheduler): cover recompute preemption and re-admission integration

feat(kv-cache): add side-effect-free local prefix probe
feat(kv-cache): add soft block retention with fallback
feat(scheduler): derive prefix retention from bounded waiting demand

feat(scheduler): add bounded cache-affinity admission
feat(scheduler): add aging guard for cache-aware admission
refactor(scheduler): share lazy scheduling features

feat(kv-cache): tier retention by near-term request demand
feat(kv-cache): estimate immediately reclaimable request blocks
feat(scheduler): use reclaimability in victim feasibility

perf(scheduler): <only evidence-backed hot-path change>
perf(kv-cache): <only evidence-backed SM89 specialization>
```

Before proposing any upstream PR, run mandatory duplicate-work checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search '<issue_number> in:body'
gh pr list --repo vllm-project/vllm --state open --search '<short area keywords>'
```

If an open PR duplicates a patch, do not open another. A future PR description
must state duplicate-check results, all test commands/results, model evals when
applicable, and that AI assistance was used. The human submitter must review and
understand every changed line.

---

## N. Main risks, mitigations and immediate next step

### Highest risks

1. **Read-only lookup drift**: public lookup can emit events and hybrid lookup
   semantics are complex. Mitigation: factor shared pure lookup core and prove
   event/stats/refcount/queue snapshots.
2. **Priority heap correctness**: resume state is mutable. Mitigation: immutable
   insertion key, one queue factory and real integration tests.
3. **Retention hot-path cost**: a stable partition of the entire free pool or a
   heap copy would defeat the design. Mitigation: lazy pressure trigger,
   bounded request candidates and scan-until-enough physical selection.
4. **Stale cached features**: KV state can mutate several times within one
   Scheduler iteration. Mitigation: generation invalidation or transaction-
   scoped results, never an unversioned iteration-wide cache.
5. **Reclaimability overclaim**: shared refs and deferred frees mean allocated
   is not immediately reclaimable. Mitigation: define immediate semantics and
   fall back on unsupported paths.
6. **Benchmark misinterpretation**: current custom “token” and ITL metrics are
   nonstandard. Mitigation: keep it as a pressure harness and use built-in bench
   output for final claims.

### First implementation action after approval

Start only Gate 1:

1. create the required `.venv` and reproduce current policy tests;
2. add the two missing tests that expose raw-count vs binary protection and
   nondeterministic final ties;
3. stabilize policy key;
4. implement keyed Priority re-admission without modifying `Request.__lt__`;
5. run the full Gate 1 test list and review the diff;
6. commit/push one logical change at a time.

No Phase 2 code begins until Gate 1 is reviewed and accepted.

---

## O. Phase 0 conclusion

The current work is a useful and directionally correct foundation. The best
engineering choice is to continue from it, while explicitly treating the
existing recompute-aware commit as a measured prototype whose policy semantics,
queue integration and evaluation methodology still need stabilization.

**No business code was modified during this Phase 0 inspection. Await approval
before entering Phase 1.**
