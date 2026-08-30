# Phase 7：RTX 4090 KV 写入 Kernel 优化报告

> 日期：2026-08-30
> 分支：`project/kv-aware-scheduling`
> 实现提交：`3eab64ecd`
> 基准提交：实现提交的同工具链回退版本
> 结论：**通过 Phase 7 Gate，停止继续堆叠 kernel heuristic。**

## 1. 一句话结论

Qwen2.5-7B 的每个 token 只有 4 个 KV head、head size 为 128。原 CUDA
KV 写入 kernel 固定启动 512 个线程，但 16-byte 向量化后实际只需要 64 个线程，
导致 16 个 warp 中只有 2 个做有效拷贝。把生产路径在严格安全条件下改为按向量数
选择 block size 后，原生 CUDA kernel 在 1024-token 形状上由 9.569 us 降至
6.592 us，达到 **1.45x**；在 2048-token 形状上达到 **1.39x**，并在
128 token 以上全面追平或超过当前 Triton 实现。

这是一项真实生产算子优化，但它只优化每层一次的 KV 写入，不应把 1.45x
微基准提升等同于 1.45x 服务吞吐提升。

## 2. 环境与测量口径

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，SM89，128 SM |
| Python | `/root/miniconda3/envs/vllm-dev/bin/python` |
| PyTorch | 2.13.0+cu130 |
| 编译 CUDA toolkit | 12.8 |
| 测量 | FlashInfer CUPTI device timing，CUDA Graph，cold L2 |
| 测量重复 | 25 次 warmup；每个形状 3 个 trial，取 trial 中位数 |
| 模型代表形状 | BF16、NHD、4 KV heads、head size 128、block size 16 |
| 对比方法 | 同一源码树、同一编译器和同一构建参数，仅切换 launch 配置 |

普通 CUDA/SM89 上的实际调用链是：

```text
FlashAttention backend
  -> fa_utils.do_kv_cache_update
  -> vllm._custom_ops.reshape_and_cache_flash
  -> torch.ops._C_cache_ops.reshape_and_cache_flash
  -> csrc/libtorch_stable/cache_kernels.cu
```

因此本次改动不是孤立 demo kernel，而是直接作用在 vLLM 的生产 KV 写入路径。

## 3. 为什么原配置浪费线程

Qwen2.5-7B 的一个 token 有 `4 * 128 = 512` 个 BF16 元素。现有 kernel 已经以
16-byte 为单位向量化，每个线程一次搬 8 个 BF16，因此只需：

```text
512 elements / 8 elements per vector = 64 useful threads
```

但旧 host launcher 使用：

```cpp
block_threads = min(num_heads * head_size, 512);  // 512 threads
```

线程 64～511 不会执行任何向量拷贝。它们不降低理论 occupancy，却会让每个 block
占用更多 resident-thread 配额，使单个 SM 同时驻留的 token block 从 24 个降至 3 个。

独立 probe 由 ptxas 报告 26 registers/thread、0 spill、0 shared memory。CUDA
occupancy API 的实测结果如下：

| threads/block | active blocks/SM | 理论 occupancy | 有效 warp 比例 |
|---:|---:|---:|---:|
| 64 | 24 | 100% | 100% |
| 128 | 12 | 100% | 50% |
| 256 | 6 | 100% | 25% |
| 512 | 3 | 100% | 12.5% |

这里最重要的不是传统 occupancy 百分比，而是同样 100% occupancy 下有多少 warp
真正搬数据，以及能同时容纳多少独立 token block。probe 在 1024 token 上从
7.778 us 降至 6.306 us，也验证了 64-thread 配置的方向。

## 4. 全局内存访问审计

| 访问 | 模式 | 结论 |
|---|---|---|
| key/value 读取 | warp 内连续的 16-byte pack | 已合并访问，无需改布局 |
| key/value cache 写入 | 单 token 内连续 16-byte pack | 已合并访问；token 间 slot 可随机 |
| slot mapping | 每 token 一次 64-bit 读取 | 开销很小，随机性由调度语义决定 |
| shared memory | 无 | 纯拷贝不需要引入 shared memory |

现有 kernel 本身已经正确向量化；本阶段没有重复做“新增向量化”，而是让 launch
配置匹配已有向量化宽度。Qwen 代表形状的输入行和 cache 行跨度都是 16-byte
对齐，因此可以安全走 64-thread 路径。

## 5. 代码改动

### 5.1 生产 launcher

修改 `csrc/libtorch_stable/cache_kernels.cu`：

- 保留通用默认值 `min(num_elements, 512)`；
- 仅在 native `auto` dtype、NHD 连续 head layout、scalar scale 下考虑缩小 block；
- 同时校验 key、value、两个 cache 的基地址和相关 stride 均满足 16-byte
  向量对齐；
- 按 `num_elements / vector_size` 计算真正需要的线程数，向上对齐至整 warp；
- 至少保留 64 threads，最多 512 threads；
- FP8、HND、per-head scale、非对齐 row 全部走原 fallback。

Qwen 代表形状最终 launch 为：

```text
grid  = (num_tokens, 1, 1)
block = (64, 1, 1)
smem  = 0
```

grid 仍是一 token 一 block，slot mapping 和写入语义完全不变。没有添加
`__launch_bounds__`，因为同一模板 kernel 仍需支持通用 512-thread fallback；强制
launch bound 会限制通用路径，得不偿失。

### 5.2 可复现工具

- `benchmarks/kernels/benchmark_reshape_and_cache_flash_cupti.py`：生产 CUDA
  与 Triton 的 CUPTI 微基准，内置 correctness check、cold L2 和有效带宽计算；
- `benchmarks/kernels/phase7_kv_write_launch_config.cu`：隔离 64/128/256/512
  threads 配置并读取 occupancy 的 BF16/NHD probe。

## 6. 精确 A/B 结果

延迟单位为 us。`相对 Triton` 为候选 CUDA 的延迟优势；负数表示候选仍较慢。

| tokens | 原 CUDA | 候选 CUDA | 加速比 | 延迟下降 | Triton | 相对 Triton |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.593 | 2.560 | 1.013x | 1.3% | 1.953 | -31.1% |
| 4 | 2.529 | 2.465 | 1.026x | 2.5% | 1.888 | -30.6% |
| 8 | 2.624 | 2.529 | 1.038x | 3.6% | 1.952 | -29.6% |
| 16 | 2.657 | 2.624 | 1.013x | 1.2% | 1.984 | -32.3% |
| 32 | 2.720 | 2.656 | 1.024x | 2.4% | 2.048 | -29.7% |
| 64 | 2.880 | 2.752 | 1.047x | 4.4% | 2.464 | -11.7% |
| 128 | 3.232 | 3.105 | 1.041x | 3.9% | 3.200 | +3.0% |
| 256 | 4.160 | 4.000 | 1.040x | 3.8% | 4.033 | +0.8% |
| 512 | 6.177 | 5.088 | 1.214x | 17.6% | 5.506 | +7.6% |
| 1024 | 9.569 | 6.592 | **1.452x** | **31.1%** | 8.578 | +23.2% |
| 2048 | 16.495 | 11.871 | **1.390x** | **28.0%** | 15.263 | +22.2% |
| 4096 | 30.333 | 28.446 | 1.066x | 6.2% | 34.397 | +17.3% |

候选 CUDA 的有效带宽在 512/1024/2048/4096 token 时分别为
413.0/637.5/708.0/590.9 GB/s。小 token 区域仍主要受 host wrapper 和 launch
固定开销影响；继续改内存拷贝 kernel 并不能消除这部分开销。

## 7. 正确性和 fallback 验证

已完成以下检查：

```text
代表 CUDA/NHD/auto 用例 + 全部 unaligned-row 用例：13 passed, 4 skipped
FP8 tensor scale + HND auto + per-head scale fallback：3 passed
完整 CUDA/tensor/NHD/auto 子矩阵：36 passed, 828 deselected
非连续 QKV unbind + slot=-1 padding 自定义检查：PASS
所有 CUPTI benchmark 形状的写入结果：PASS
git diff --check：PASS
```

候选 `.so` 与保存的验收 artifact SHA256 均为：

```text
690e31537c69e120ee33d70eaa531be70c2c56ca6179cba1bdb01e72ac134f88
```

## 8. NCU 限制

本机已有 Nsight Compute 2025.1.1，但容器没有 GPU performance-counter 权限，
运行时返回 `ERR_NVGPUCTRPERM`。这不是缺少安装包，而是宿主机
`RmProfilingAdminOnly=1` 且容器缺少相应 capability。CUPTI timing、ptxas
register/spill 信息和 CUDA occupancy API 均可用，足以确认本次 launch-config
问题；报告不伪造 stall reason、DRAM throughput 等拿不到的 NCU counter。

## 9. 对端到端性能的诚实解释

Qwen2.5-7B 有 28 层。按单次 kernel 差值粗略累加：

- 1024 token：约节省 `2.977 us * 28 = 83 us`；
- 2048 token：约节省 `4.624 us * 28 = 129 us`；
- decode 小 batch：每层差值很小，总体只有数微秒量级。

prefill 的 attention/GEMM 远大于这部分写入时间，因此常规 `vllm bench serve`
中的端到端变化预计低于 1%，容易被请求调度和 GPU 波动淹没。本阶段可严谨声称：

> 在 RTX 4090 上优化 vLLM 生产 KV-cache 写入 kernel 的 launch configuration，
> Qwen2.5-7B 代表形状微基准最高 1.45x，并保持通用 fallback 与完整正确性。

不能声称“服务吞吐提升 45%”。

## 10. 复现命令

```bash
/root/miniconda3/envs/vllm-dev/bin/python \
  benchmarks/kernels/benchmark_reshape_and_cache_flash_cupti.py \
  --implementation cuda

/root/miniconda3/envs/vllm-dev/bin/python \
  benchmarks/kernels/benchmark_reshape_and_cache_flash_cupti.py \
  --implementation triton

/root/miniconda3/envs/vllm-dev/bin/python -m pytest -q \
  tests/kernels/attention/test_cache.py::test_reshape_and_cache_flash \
  -k 'cuda and tensor and NHD and auto and not triton'
```

## 11. Phase 7 停止规则

本轮已经满足“真实 gap、生产路径、通用 fallback、可复现提升、正确性通过”五项
Gate。下一种可能优化需要处理小 batch 的 wrapper/launch 固定成本，属于新的融合或
dispatch 课题，而不是继续调整 block size。它的端到端上限又很低，因此本项目在
这里停止 Phase 7，不继续添加 Phase 7b/7c，也不把它与调度 heuristic 继续耦合。
