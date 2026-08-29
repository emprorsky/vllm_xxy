# Phase 6c 探索报告：Cache-affinity 标准吞吐复现

日期：2026-08-29（UTC）

分支：`project/kv-aware-scheduling`

代码起点：`b6c775ba6` + 未提交的 Phase 6b 只读 telemetry

## 1. 结论

Phase 6c 找到了当前项目第一组同时具备“真实行为变化”和“双向顺序复现”的标准
serving 吞吐收益：

> 在 RTX 4090 / Qwen2.5-7B / 1250 KV blocks 的固定 prefix-repetition workload
> 中，开启 `admission_policy=cache_affinity` 后，两对反向顺序 A/B 的 output
> throughput 分别提高 **6.933%** 和 **1.659%**，两轮均值提高 **4.286%**。

这次与 Phase 5c 不同，机制确实激活：两轮 treatment 分别有 592 和 651 次成功
admission 绕过基础队头，占成功 admission 的 65.9% 和 69.1%。因此吞吐差异具备
明确的行为前提，不是“策略一次都没改选”的伪归因。

但 latency Gate 未通过：TTFT mean/p50/p99 的两轮均值分别退化 44.0%、57.4%、
59.0%。与此同时 mean TPOT 改善 9.19%。准确解释是：cache-affinity 改善已开始
生成请求的 decode efficiency/批处理组成和总完成时间，但让部分请求等待更久。

因此当前策略可以作为有数据支持的实验性“throughput mode”，不能默认开启，也
不能宣称全面延迟改善。下一步目标是通过更早 aging 或更小 candidate window 收回
TTFT，同时保留一部分吞吐收益。

## 2. 隔离实验设计

共同配置：

- 模型/GPU：`Qwen/Qwen2.5-7B-Instruct` / RTX 4090；
- prefix caching 开启，`num_gpu_blocks_override=1250`；
- `preemption_policy=recompute_aware`；
- `prefix_cache_eviction_policy=lru`，关闭 Phase 2/5a retention 混杂；
- C：`admission_policy=default`；
- T：`admission_policy=cache_affinity`、window 8、aging 30s；
- 官方 `vllm bench serve` / `prefix_repetition`；
- 8 个 prefix、prefix 900 tokens、suffix 64 tokens、output 1024 tokens；
- 192 正式请求、48 warmup、并发 48、request rate inf；
- `temperature=0.7`、`ignore_eos=true`、seed 42；
- 每轮全新服务进程，执行顺序 C1→T1、T2→C2；
- 四轮均为 192/192 成功、185,097 input tokens、196,608 output tokens。

唯一实验变量是 admission policy。

## 3. 标准 benchmark 结果

### 3.1 单轮结果

| 指标 | C1 | T1 | T2 | C2 |
|---|---:|---:|---:|---:|
| duration (s) | 187.551 | 175.390 | 183.111 | 186.149 |
| output throughput (tok/s) | 1048.291 | 1120.973 | 1073.712 | 1056.189 |
| TTFT mean (ms) | 4959.581 | 6766.949 | 6583.281 | 4311.347 |
| TTFT p50 (ms) | 2334.203 | 3487.582 | 2842.633 | 1687.566 |
| TTFT p99 (ms) | 18045.925 | 24571.283 | 31301.511 | 17099.727 |
| TPOT mean (ms) | 36.043 | 32.504 | 33.186 | 36.294 |
| TPOT p99 (ms) | 48.624 | 48.577 | 48.692 | 49.434 |
| ITL p99 (ms) | 100.439 | 100.156 | 103.254 | 97.527 |

### 3.2 两对方向与均值

| 指标 | Pair 1：C1→T1 | Pair 2：C2→T2 | C 均值 | T 均值 | 均值变化 |
|---|---:|---:|---:|---:|---:|
| duration | -6.484% | -1.632% | 186.850 s | 179.251 s | -4.067% |
| output throughput | +6.933% | +1.659% | 1052.240 | 1097.342 | +4.286% |
| TTFT mean | +36.443% | +52.700% | 4635.464 ms | 6675.115 ms | +44.001% |
| TTFT p50 | +49.409% | +68.442% | 2010.885 ms | 3165.107 ms | +57.399% |
| TTFT p99 | +36.161% | +83.053% | 17572.826 ms | 27936.397 ms | +58.975% |
| TPOT mean | -9.819% | -8.564% | 36.169 ms | 32.845 ms | -9.190% |
| TPOT p99 | -0.097% | -1.501% | 49.029 ms | 48.634 ms | -0.806% |
| ITL p99 | -0.282% | +5.872% | 98.983 ms | 101.705 ms | +2.750% |

## 4. 机制激活

| 指标 | T1 | T2 |
|---|---:|---:|
| selection calls | 9,111 | 9,418 |
| candidate probes | 71,723 | 72,886 |
| successful admissions | 898 | 942 |
| admitted reordered | 592 | 651 |
| admitted reordered / admitted | 65.9% | 69.1% |
| admitted aged | 137 | 199 |
| selected cached tokens | 8,265,168 | 8,718,512 |
| admitted cached tokens | 794,432 | 828,640 |

控制组 admission 指标全部为 0，符合 default fast path。

两轮 C 的 prefix hits 都是 201,664；T1/T2 分别是 200,832/200,000，说明吞吐
收益不是来自更多 APC hits。preemptions 的 run-level 波动也很大：C1/C2 为
657/552，T1/T2 为 676/726，不能作为收益解释。

更可信的机制是 admission 先选择 resumed/剩余 prefill 更少的候选，改变 running
batch 组成，使已运行请求的 mean TPOT 降低并缩短总完成时间；代价是基础队头等待
更久，直至 30s aging 恢复基础顺序。

## 5. 下一步

> Phase 6d 已完成 5s/10s 单轮筛选：5s 基本收回 p99 TTFT 但吞吐收益降至
> +0.275%；10s 为 +3.648% throughput / +29.060% TTFT p99，是下一轮唯一待
> paired confirmation 的候选。详见
> [`PHASE6D_AGING_SWEEP_REPORT_CODEX.md`](PHASE6D_AGING_SWEEP_REPORT_CODEX.md)。

Phase 6d 只优化已证实的 throughput/fairness 边界，不叠加新的 KV heuristic：

1. treatment-only 筛选 aging 5s/10s 和 window 4；
2. 目标是相对本报告 control 均值保持至少 +2% throughput，同时显著降低 TTFT
   mean/p99 退化；
3. 最优候选再做一对反向 fresh-process A/B；
4. 同时把 `candidate_probes / admitted`（当前约 76--80）作为 Scheduler CPU
   放大指标；
5. 若公平性无法改善，则明确保留为 opt-in throughput mode，而不是默认策略。

原始产物：

- `phase6c_benchserve_1250_default_admission_c1.json`；
- `phase6c_benchserve_1250_cache_affinity_t1.json`；
- `phase6c_benchserve_1250_cache_affinity_t2.json`；
- `phase6c_benchserve_1250_default_admission_c2.json`；
- `phase6c_admission_metrics.json`。

当前简历可安全表述为：

> 为 vLLM V1 实现有界 cache-affinity admission，在官方 `vllm bench serve` 的
> RTX 4090/Qwen2.5-7B/高 KV 压力双向顺序 A/B 中，通过 1,243 次成功队头重排
> 验证机制激活，output throughput 均值提升 4.3%、mean TPOT 降低 9.2%；同时
> 识别 TTFT 公平性退化并以 aging 作为后续优化边界。
