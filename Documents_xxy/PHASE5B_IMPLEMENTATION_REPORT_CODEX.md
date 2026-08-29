# Phase 5b 实施报告：只读 Reclaimable-KV Estimate

日期：2026-08-29（UTC）

分支：`project/kv-aware-scheduling`

起点：`040a879d3`（Phase 5a）

## 1. 结论

Phase 5b 的只读信息层 correctness Gate 已完成。

本阶段只新增 `KVCacheManager.estimate_reclaimable_blocks(request)`，没有把
估算结果接入抢占选择、admission、`SchedulingFeatureContext` 或 Phase 5a 的
`priority_aware` retention。这样可以独立验证 refcount 语义，避免在 Phase 5a
性能结果仍混合时继续叠加 heuristic。

该接口返回的是：如果现在通过 vLLM 正式 free 路径释放该请求，最终会回到
BlockPool 的唯一物理 block 数。它不是“当前 allocation retry 立即可用页数”。

## 2. 估算语义

对请求在所有 KV group block table 以及 request-scoped partial-tail pins 中持有的
引用进行聚合：

```text
owned_refs[physical_block_id] = 该请求持有该物理页的引用次数

reclaimable(block) =
    block 非 null
    且 owned_refs[block_id] == block.ref_cnt
```

最终按唯一物理 `block_id` 计数，而不是按 block-table entry 计数。

因此：

- A/B 共享 prefix 时，共享页的 `ref_cnt` 大于 A 或 B 单独持有的引用数，不计入；
- 同一请求 bookkeeping 中重复出现同一物理页时，先统计引用重数，物理页只计一次；
- null padding 永不计入；
- request-scoped partial-tail pin 会随请求正式释放，因此计入 owned refs；
- CoW copy fence 或其他 operation retention 带来的额外引用不属于请求，相关页不计入；
- 查询只读，不改 refcount、free queue、hash、事件、指标或 Request 状态。

## 3. Eventual 与 Immediate 的边界

estimator 只描述 official free 后的 eventual pool return。

Scheduler 开启 deferred block free 时，请求 bookkeeping 可以先从 manager 中移除，
但物理页要等 GPU write fence 完成后才回到 free queue。新增测试验证：

```text
estimate > 0
请求 finish，但 fence 未完成       -> free queue 增量 = 0
fence 完成并 drain deferred frees  -> free queue 增量 = estimate
```

所以后续若实现 reclaimability-aware victim feasibility，必须由策略层结合
`defer_block_free`、request fence 状态和当前 allocation deficit 判断；不能直接把
本接口返回值当作立即容量。

## 4. 代码改动

| 文件 | 改动 |
|---|---|
| `vllm/v1/core/kv_cache_manager.py` | 新增只读 `estimate_reclaimable_blocks`，按物理 ID 聚合请求引用重数并与 `ref_cnt` 比较 |
| `tests/v1/core/test_prefix_caching.py` | 增加 shared prefix、重复引用、request pin/operation retention、hybrid/null 与真实 free delta 测试 |
| `tests/v1/core/test_deferred_block_free.py` | 增加 eventual estimate 与 deferred immediate capacity 的边界测试 |
| `Documents_xxy/IMPLEMENTATION_PLAN.md` | 记录 Phase 5b 信息层完成状态和策略隔离边界 |

没有修改：

- `prefix_cache_eviction_policy`，`priority_aware` 仍是独立实验开关；
- preemption/admission policy；
- Scheduler victim selection；
- `SchedulingFeatureContext`；
- metrics、CLI 和 benchmark harness。

## 5. 验证结果

所有命令均直接使用 `/root/miniconda3/envs/vllm-dev/bin/python`，没有运行 uv、
pip 或安装命令。

### 5.1 修改前基线

```text
tests/v1/core/test_prefix_caching.py
103 passed, 14 warnings
```

### 5.2 Phase 5b 定向与完整回归

| 测试 | 结果 |
|---|---:|
| `test_prefix_caching.py -k reclaimable -vv` | 4 passed |
| `test_deferred_block_free.py -k reclaimable -vv` | 1 passed |
| 完整 `test_prefix_caching.py -q` | 107 passed |
| 完整 `test_deferred_block_free.py -q` | 13 passed |
| 完整 `test_single_type_kv_cache_manager.py -q` | 11 passed |

静态检查：

```text
python -m ruff check <3 changed Python files>
All checks passed!

git diff --check
passed
```

## 6. 性能与下一步

本阶段没有 GPU A/B，因为 estimator 尚未被任何 runtime policy 调用，默认路径和
Phase 5a 路径都不会产生新的扫描开销。此时做 serving A/B 不能验证该 abstraction，
也容易把环境波动误判为收益。

建议下一步先停在清晰边界上审查和提交 Phase 5b。后续若进入基于 reclaimability
的 victim feasibility，应作为新的独立实验策略进行：

1. 只在真实 allocation deficit 下解析 reclaimability；
2. deferred/connector 路径无法证明立即归还时回退到现有 `recompute_aware`；
3. user priority 仍是硬约束；
4. 先用 deterministic policy tests 验证 feasibility，再做 GPU A/B；
5. 不修改 Phase 5a retention tier，也不把其混合结果作为叠加复杂度的依据。
