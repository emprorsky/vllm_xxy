# 四卡 KV 感知调度实验

完整设计见 [`20261019任务规划.md`](../20261019任务规划.md)。当前状态见 `progress.json`。

本目录为原型与实验工具，行为参考 Preble、EfficientAgent；不是两篇论文的完整复现。当前路由目录是明确标注的历史估计，不是实际 GPU 驻留目录。实际引擎命中通过逐请求 usage 与 Prometheus 快照另行记录。

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
