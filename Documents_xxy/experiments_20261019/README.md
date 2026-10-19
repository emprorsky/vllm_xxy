# 四卡 KV 感知调度实验

完整设计见 [`20261019任务规划.md`](../20261019任务规划.md)。当前状态见 `progress.json`。

本目录为原型与实验工具，行为参考 Preble、EfficientAgent；不是两篇论文的完整复现。路由目录提供历史估计和事件+完成确认两种路径。实际引擎命中通过逐请求 usage 与 Prometheus 快照另行记录。

## 已实现的入口

- `policies.py`：请求账本、工作量/淘汰代理评分、条件化保存决定与精确前缀身份。
- `router.py`：支持流式 completions、统一目录下的六种路由对照。
- `lmcache_adapter.py`：自定义 connector，实际 CPU 遥测和请求级保存控制；需真实服务验收后才视为可用。
- `build_trace.py`：半合成代码机制负载与公开示范轨迹转换，来源与限制写入 manifest。
- `replay_client.py`：保持会话顺序和工具间隔，记录流式 token、错误、usage、时延及指标快照。
- `run_experiment.py`：启动/清理本轮进程，保留命令、日志和配置。

## 解释器与测试

在仓库根运行，解释器为 `/root/miniconda3/envs/vllm-dev/bin/python`。不需要新环境。为子进程补齐同一环境的 `bin` PATH。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /root/miniconda3/envs/vllm-dev/bin/python -m pytest --noconftest Documents_xxy/experiments_20261019/test_policies.py -q
/root/miniconda3/envs/vllm-dev/bin/python -m Documents_xxy.experiments_20261019.run_experiment --config CONFIG.json --output NEW_RUN_DIRECTORY
```

运行目录已存在时拒绝覆盖。完整 trace、模型、原始服务日志位于 `/root/autodl-tmp`，不提交大型数据。

## 正确性排查记录

2026-10-19 本机验收发现：默认自定义算子/FlashAttention 路径下，0.6B 与 3B 返回异常重复字符；V1/V2 runner 均可复现。相同 3B 权重的 Transformers eager reference 正常。0.6B 在禁用自定义算子、使用 native RMSNorm 和 Triton attention 后通过基本输出检查。

具体故障算子尚待隔离，不能把排查用异常服务作为性能基线。工作配置通过 3B 与 KV 恢复检查后再固定用于所有对照。

另已定位独立的 CPU KV 恢复错误：LMCache 0.5.6 默认 V2 的 `[2,L,T,H*D]` 缓冲与本分支 Triton 的打包 K/V 不兼容。逐元素 round-trip 中分离布局正常、打包布局错误；V3 使用 `[1,L,T,H*2D]` 后通过。自定义 connector 会拒绝 V2 打包布局，并在 CPU allocator 初始化前完成 V3 格式发现；配置必须包含 `LMCACHE_USE_GPU_CONNECTOR_V3=True`。这不是降低精度或改动模型权重。

`check_kv_transfer.py` 保留复制检查；`correctness_suite.py` 的 8 个 Coding 提示和 3 个较长前缀事实提示已通过 CPU 恢复验收，包含实际读取与冻结概率阈值。旧 V2 offload 性能结果隔离，修复后使用 `V3_` 前缀重跑，不混入原轮统计。

当前 Triton 路径在 SM86 拒绝 FP8 KV；参数修正后的启动日志已保留。不能把它与 AWQ 权重量化混为同一种限制。

## 分阶段执行与复核

```bash
/root/miniconda3/envs/vllm-dev/bin/python -m Documents_xxy.experiments_20261019.suite --stage offload --repeats 3
/root/miniconda3/envs/vllm-dev/bin/python -m Documents_xxy.experiments_20261019.campaign --stages routing combined pd1 pd2 replicas capacity chunking
/root/miniconda3/envs/vllm-dev/bin/python -m Documents_xxy.experiments_20261019.case_runner --stages natural events mixed quantization kvfp8
/root/miniconda3/envs/vllm-dev/bin/python -m Documents_xxy.experiments_20261019.summarize --root /root/autodl-tmp/outputs/kv_aware_20261019/runs --output /root/autodl-tmp/outputs/kv_aware_20261019/summary
/root/miniconda3/envs/vllm-dev/bin/python -m Documents_xxy.experiments_20261019.analyze_details --root /root/autodl-tmp/outputs/kv_aware_20261019/runs
/root/miniconda3/envs/vllm-dev/bin/python -m Documents_xxy.experiments_20261019.plot_results --aggregate /root/autodl-tmp/outputs/kv_aware_20261019/summary/aggregate.json --output /root/autodl-tmp/outputs/kv_aware_20261019/figures
```

已存在的运行不覆盖；需要独立复验时使用新 prefix，例如 `case_runner --prefix F_ --stages offload --repeats 3`。重复轮按正序/逆序交错。`--names` 可从给定 stages 中选择具体配置。screen 与 F_ 复验结果需分开解读。

所有正式轮先经过三个非强制输出算术探测，再运行短前缀预热。客户端连续 90 秒没有请求完成时取消本轮，进程组由 runner 清理；错误轮保留 manifest 和逐请求/服务日志。

## 策略与既有能力的区别

- vLLM 原有 APC、分块、connector 的复制/恢复实现继续使用。本轮新增的是跨实例放置目标、保存准入决定与跨组件状态管理。
- `work` 使用排队工作代理 + 剩余 prefill 的 argmin；`cost` 只额外加淘汰损失代理。它不是“保持 threshold 规则，仅替换计数”的严格单项消融，不能把 threshold→work 的全部变化只归因于负载估计。
- `work_remaining`/`cost_remaining` 在上述评分外，只根据流中观察到的 token ID 数单调扣减预算，首个实际输出后移除 prefill；没有可靠 token 数时保守保留原预算。只支持单输出 `n=1`，不按等待时长或未来输出猜进度。未建模 continuous batching 重叠，仍是排序代理。
- O-WS 的开关落点是既有 `lmcache.skip_save`；新工作是工作集估计、真实容量/淘汰反馈、一次决定冻结与生命周期适配。不能把 skip_save 机制本身称为发明。
- offload 公开轨迹对照统一最多 2 个驻留请求；8 个驻留请求的强压力版本曾出现无进展，失败日志保留。seq2 的 O-OFF 才是同配置关闭 offload 的参照。
- GPU KV 固定为 512 × 16 = 8192 tokens，以构造可控容量压力。自然显存容量是单独对照；不能把人工限容下的数字推广到默认显存配置。

## KV event 目录扩展

`event_directory.py`/`events_router.py` 支持真实 ZMQ KV events、启动重放、序列缺口失效、payload 变化的 epoch 检测及原子重放发布。history 与 events 消融均开启相同引擎发布器和订阅器。

`BlockStored` 可在 GPU 写入前发出，因此事件目录额外要求路由观察到覆盖该前缀的请求成功完成，才用于命中估计。移除、缺口与清空会撤销该确认。它是事件索引与完成确认的保守组合，仍有传输延迟，不能称为零延迟物理驻留 oracle。事件缺少物理重复副本 ID，移除按前缀撤销，可能低估仍存活的重复副本。

限定：固定模型池、TP=1、dense full attention、单 group、16-token 完整块。salt/LoRA/MM、部分块和 REMOTE 条目排除。暂不叠加 CPU tier 身份或引擎热更新。事件目录只用于选路，计算复用始终由 vLLM 验证。
