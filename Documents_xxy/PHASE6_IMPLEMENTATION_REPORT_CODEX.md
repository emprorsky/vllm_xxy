# Phase 6 实施报告：Scheduler CPU 成本与标准 Serving 复现

日期：2026-08-29（UTC）

分支：`project/kv-aware-scheduling`

起点：`8c53b0648`（Phase 5c）

## 1. 结论

> **Phase 6b 因果复核修正（2026-08-29）**：新增 counterfactual victim telemetry
> 证明官方 1250/1000-block 两轮分别 779/655 次决策均为 0 次改选。因此本报告记录的
> `+2.091%` 与 `-0.596%` 是 run-level 差异，不能归因于 Phase 5c，也不能据此判断
> KV-budget 有效区间。最终结论以
> [`PHASE6B_CAUSAL_DIAGNOSTIC_REPORT_CODEX.md`](PHASE6B_CAUSAL_DIAGNOSTIC_REPORT_CODEX.md)
> 为准。

Phase 6 measurement Gate 已完成，得到两个需要同时保留的结论。

第一，Phase 5c 的 CPU 成本可测但不是当前 serving 收益/退化的主因：

- 34 blocks/candidate、单核固定 CPU 的真实 refcount estimator benchmark 中，
  N=8/16/64/256 的决策中位耗时分别为 120.264/246.947/983.002/3807.596 us；
- 开销近似随候选数和 block references 线性增长；
- 按 Phase 5c 自定义 A/B 的实际平均约 28 candidates/shortfall、约 506 次
  shortfall 估算，累计 CPU 时间约 0.22 s，仅约 192 s wall time 的 0.11%；
- 因此必须保留“只在真实 allocation failure 下扫描”的边界，但当前 2% 量级的
  serving 变化不能用约 0.1% 的 estimator CPU overhead 解释。

第二，标准 serving 复现证明收益存在 KV-budget 有效区间：

- 官方 `vllm bench serve`、1250 blocks：吞吐 `+2.091%`，wall time
  `-2.048%`；
- 相同 workload、1000 blocks：吞吐 `-0.596%`，wall time `+0.599%`；
- 两个 budget 均为 192/192 成功，input/output token 总数完全一致；
- TTFT、TPOT、ITL 和 E2EL 分位数方向混合，不能宣称全面延迟改善。

Phase 6b 后的最终判断是：CPU 成本测量仍然有效；标准工具复现了 run-level 波动，
没有复现可归因于 Phase 5c 的吞吐收益。跨 KV budget performance Gate 未通过。

## 2. Scheduler CPU microbenchmark

### 2.1 方法

新增 `Documents_xxy/benchmark_phase6_scheduler_cpu.py`，使用真实：

- vLLM `Request`；
- `KVCacheManager` block table；
- physical block `ref_cnt`；
- `estimate_reclaimable_blocks()`；
- Scheduler `_select_reclaimable_preemption_victim()` 与 stats 更新。

配置：

- CPU：Intel Xeon Gold 6430；进程固定到 CPU 0；`OMP_NUM_THREADS=1`；
- candidates：`1/8/16/64/256`；
- 每个请求 34 个独占 physical blocks、block size 16；
- shortfall：1 block；
- 每个数据点自适应迭代至约 100 ms，重复 9 次，报告 run median；
- GC 在正式计时批次中关闭；
- default/recompute/reclaimable 使用相同 Request 对象和 running-list slice。

34 blocks/request 来自 Phase 5c 两轮实际 telemetry：所有 candidate reclaimable
blocks / candidate estimates 约为 34，因而比任意选择一个小 block table 更贴近该
serving workload。

执行命令：

```text
taskset -c 0 env OMP_NUM_THREADS=1 \
  /root/miniconda3/envs/vllm-dev/bin/python \
  -m Documents_xxy.benchmark_phase6_scheduler_cpu
```

### 2.2 结果

| candidates | default (us) | recompute-aware (us) | reclaimable-aware (us) | reclaimable - recompute (us) |
|---:|---:|---:|---:|---:|
| 1 | 0.127 | 0.530 | 17.873 | 17.344 |
| 8 | 0.127 | 2.274 | 120.264 | 117.990 |
| 16 | 0.127 | 3.539 | 246.947 | 243.407 |
| 64 | 0.127 | 12.184 | 983.002 | 970.818 |
| 256 | 0.128 | 47.189 | 3807.596 | 3760.407 |

这不是“零开销”策略：N=256 已达到 3.8 ms，因此 estimator 不能进入每个 scheduler
step 或 default fast path。当前只在 allocation failure 下执行，使实际累计成本维持
在亚秒级。

按 N=16 和 N=64 之间线性插值，Phase 5c 的约 27.7 candidates/shortfall 对应约
0.43 ms；乘以约 505.5 次 shortfall 得到约 0.22 s。这个估算用于量级判断，不是
profiler 给出的精确累计 CPU time。

原始产物：`phase6_scheduler_cpu.json`。

## 3. 标准 `vllm bench serve` 复现

### 3.1 共同配置

- 模型/GPU：`Qwen/Qwen2.5-7B-Instruct` / RTX 4090；
- server：prefix caching、Phase 5a `priority_aware` retention、Phase 3
  `cache_affinity` admission、candidate window 8、aging 30s；
- C：`preemption_policy=recompute_aware`；
- T：`preemption_policy=reclaimable_aware`；
- dataset：仓库内置 `prefix_repetition`；
- 8 个 prefix；prefix 900 tokens；suffix 64 tokens；output 1024 tokens；
- 192 正式请求、48 warmup、max concurrency 48、`request-rate=inf`；
- `temperature=0.7`、`ignore_eos=true`、seed 42；
- 每轮全新服务进程，轮间确认服务和 GPU 显存释放；
- 1250 blocks 执行 C→T，1000 blocks 执行 T→C，以反向顺序降低时序偏差；
- 四轮均为 192/192 成功、185,097 input tokens、196,608 output tokens。

### 3.2 1250-block 结果

| 指标 | recompute C | reclaimable T | 变化 |
|---|---:|---:|---:|
| duration (s) | 184.008 | 180.240 | -2.048% |
| output throughput (tok/s) | 1068.475 | 1090.813 | +2.091% |
| TTFT mean (ms) | 5839.698 | 5869.960 | +0.518% |
| TTFT p50 (ms) | 2597.666 | 3298.902 | +26.995% |
| TTFT p90 (ms) | 14807.332 | 14070.210 | -4.978% |
| TTFT p99 (ms) | 26049.037 | 36111.673 | +38.630% |
| TPOT mean (ms) | 33.928 | 34.309 | +1.122% |
| TPOT p99 (ms) | 51.521 | 50.299 | -2.372% |
| ITL p99 (ms) | 100.220 | 100.966 | +0.744% |
| E2EL mean (ms) | 40548.492 | 40968.219 | +1.035% |
| E2EL p99 (ms) | 58347.431 | 65230.441 | +11.797% |

服务端 fresh-process counters（包含 initial probe、48 warmup 和 192 正式请求）：

- C preemptions：742；
- T preemptions/shortfall events：815；
- T candidate estimates：21,790，约 26.7 candidates/shortfall；
- T sufficient selections：815，zero-progress selections：0。

Phase 6b 证明这一轮的实际 victim 与 recompute baseline 完全相同，因此这里的吞吐、
preemption 和 TTFT 差异都不能归因于 reclaimable victim ranking。

### 3.3 1000-block 结果

| 指标 | recompute C | reclaimable T | 变化 |
|---|---:|---:|---:|
| duration (s) | 224.707 | 226.054 | +0.599% |
| output throughput (tok/s) | 874.952 | 869.739 | -0.596% |
| TTFT mean (ms) | 14148.849 | 12778.631 | -9.684% |
| TTFT p50 (ms) | 14399.318 | 12118.165 | -15.842% |
| TTFT p90 (ms) | 32266.201 | 30259.357 | -6.220% |
| TTFT p99 (ms) | 39362.811 | 42299.580 | +7.461% |
| TPOT mean (ms) | 35.183 | 37.465 | +6.485% |
| TPOT p99 (ms) | 55.581 | 60.140 | +8.202% |
| ITL p99 (ms) | 95.322 | 93.805 | -1.591% |
| E2EL mean (ms) | 50141.121 | 51104.994 | +1.922% |
| E2EL p99 (ms) | 72134.588 | 74900.548 | +3.834% |

服务端 fresh-process counters：

| 指标 | C | T | 变化 |
|---|---:|---:|---:|
| preemptions | 687 | 644 | -6.259% |
| prefix cache queries | 231,369 | 231,369 | 0.000% |
| prefix cache hits | 206,464 | 209,152 | +1.302% |

T 中 644 次 shortfall 全部为 1 block，15,770 次 candidate estimate，约 24.5
candidates/shortfall；644 次选择均被预测为 sufficient，zero-progress 为 0。

Phase 6 当时把 1000-block 反转推测为 reclaimability/recompute cost 冲突。Phase 6b
直接否定了该解释：644 次原 run 未保存反事实信息，但新 repeat 的 655 次选择全部
与 baseline 相同，additional/avoided recompute 均为 0。原解释应撤销。

原始产物：

- `phase6_benchserve_1250_recompute.json`
- `phase6_benchserve_1250_reclaimable.json`
- `phase6_benchserve_1000_reclaimable.json`
- `phase6_benchserve_1000_recompute.json`

## 4. 验收与下一步

Phase 6 不修改生产策略，只完成测量与边界判断：

- CPU measurement Gate：通过；开销线性、可解释，在实际候选规模下不是主要瓶颈；
- standard serving reproduction：只复现了 run-level 数值差异，没有算法因果激活；
- cross-budget generalized performance Gate：未通过；1000 blocks 出现小幅负收益；
- latency Gate：未通过；不同分位数和 budget 的方向混合。

下一步不要直接调整权重或加入更多 heuristic。应先补只读 telemetry：

1. 记录 reclaimable 策略实际改变 victim 的次数；
2. 同时记录原 recompute victim 与实际 victim 的 computed-token cost；
3. 记录每个 shortfall 为满足 1 block 容量实际连续抢占的 victim 数和总 wasted
   recompute tokens；
4. 用这些数据验证 1000-block 反转究竟来自单次高成本 victim、共享 refcount 的
   动态变化，还是请求队列轨迹放大；
5. 只有证据明确后，再设计 cost-aware feasibility，而不是在结果上调 magic
   threshold。

本报告原简历表述作废。更新建议：

> 为 vLLM V1 实现 refcount-aware KV 抢占与反事实可观测性；通过单核
> microbenchmark 量化 Scheduler 决策成本，并在官方 `vllm bench serve` 中识别
> one-block shortfall 导致的策略退化，排除约 2% run-level 波动的错误性能归因。
