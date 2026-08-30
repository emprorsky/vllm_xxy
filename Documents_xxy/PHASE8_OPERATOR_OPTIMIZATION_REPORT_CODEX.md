# Phase 8：RMSNorm 与 RoPE 生产算子优化报告

> 日期：2026-08-30
> 分支：`project/kv-aware-scheduling`
> GPU：NVIDIA GeForce RTX 4090（SM89）
> 模型代表形状：Qwen2.5-7B，BF16，hidden size 3584，28 Q heads，4 KV
> heads，head size 128
> 结论：RMSNorm 获得 **1.01x～1.04x** 的低 token 小收益；RoPE 获得
> **1.13x～1.84x** 的全形状稳定加速。两项均通过生产 op 正确性和同二进制
> 反向 A/B Gate，Phase 8 到此停止。

## 1. 为什么选择这两个算子

Phase 7 已经优化 KV-cache 写入。继续做算子优化时，没有直接重写 GEMM、Attention
或 MoE，因为它们已有 cuBLAS、CUTLASS、FlashAttention 等成熟实现，单卡短周期内
重写的收益风险比很差。先沿 Qwen2.5-7B 的真实每层调用链寻找：

```text
Decoder layer
  -> fused_add_rms_norm       残差相加并归一化
  -> QKV projection           GEMM，成熟库负责
  -> rotary_embedding         对 Q/K 加位置旋转
  -> attention                FlashAttention 负责
  -> reshape_and_cache_flash  Phase 7 已优化
```

选择标准是：

| 标准 | fused add + RMSNorm | RoPE |
|---|---|---|
| Qwen 每层真实调用 | 是 | 是 |
| 当前走 vLLM 自有 CUDA kernel | 是 | 是 |
| 能建立精确 correctness reference | 是 | 是 |
| 能隔离 kernel 边界测量 | 是 | 是 |
| 源码存在可验证的 launch/访存疑点 | 1024 threads 中大量线程无 vector task | 标量旋转、固定一 token 一 block |

## 2. 测量纪律

- Python：`/root/miniconda3/envs/vllm-dev/bin/python`；
- PyTorch：2.13.0+cu130；编译 CUDA toolkit：12.8；
- CUPTI device timing、CUDA Graph、cold L2；
- 25 次 warmup，每个 shape 5 个 trial，表格取 trial 中位数；
- baseline 与 candidate 使用同一源码树、同一工具链，只替换保存的稳定 ABI `.so`；
- RoPE 的 Q/K 从一个 fused-QKV tensor 中 `split`，保留真实的非连续 token stride；
- 性能循环使用全零输入，因为两个 in-place op 对零输入保持不变，避免把 clone/reset
  计入 kernel；随机输入另做 correctness check。

新增可复现脚本：

- `benchmarks/kernels/benchmark_fused_add_rms_norm_cupti.py`；
- `benchmarks/kernels/benchmark_rotary_embedding_cupti.py`。

## 3. 算子一：fused add + RMSNorm

### 3.1 原实现的问题

Qwen hidden size 为 3584。现有 BF16 对齐路径每线程处理 8 个元素，所以每个 token
只有：

```text
3584 / 8 = 448 个 vector tasks
```

但小 token launcher 启动 1024 threads。线程 448～1023 不再处理 vector task，
只是参与 block 的线程资源分配。旧 launcher 是在选择 scalar/vector 分支之前统一
计算 block size，因此 vector path 没有利用已经知道的 vector width。

### 3.2 最小改动

只在已经满足对齐条件的 vector path 中使用：

```text
block_threads = min(hidden_size / vector_width, max_block_size)
```

generic、非对齐和 batch-invariant fallback 仍使用原来的 scalar 线程数。没有改归约
公式、数据类型或 API，也没有引入 shared memory。

### 3.3 精确 A/B

| tokens | 原 1024-thread 配置 | 候选 448-thread 配置 | 加速 |
|---:|---:|---:|---:|
| 1 | 2.720 us | 2.687 us | 1.012x |
| 4 | 2.720 us | 2.656 us | 1.024x |
| 8 | 2.752 us | 2.720 us | 1.012x |
| 16 | 3.008 us | 2.880 us | **1.044x** |
| 32 | 3.648 us | 3.584 us | 1.018x |
| 64 | 4.927 us | 4.864 us | 1.013x |
| 128 | 6.463 us | 6.463 us | 1.000x |
| 256～2048 | 基本持平 | 基本持平 | 约 1.00x |

结论不是“线程减半所以大幅加速”。该 kernel 还受 block reduction 和固定 launch
成本影响，原实现也已经接近 FlashInfer。它只是一项风险很低、对 decode/小 batch
有 1%～4% 收益的修正。

### 3.4 正确性

运行 vLLM 现有 layernorm 核心矩阵：

```text
324 passed, 649 deselected
```

覆盖 FP32/FP16/BF16、带权/无权、带残差/不带残差、连续/切片输入和多种 hidden
size。

## 4. 算子二：RoPE

### 4.1 RoPE 做什么

RoPE 把一个 head 的前后半段两两看成二维向量：

```text
x' = x * cos(theta) - y * sin(theta)
y' = y * cos(theta) + x * sin(theta)
```

它不是 GEMM：没有大矩阵乘法，也不使用 Tensor Core；主要工作是读 Q/K、读
cos/sin cache、做少量逐元素乘加，再写回 Q/K。

### 4.2 如何发现瓶颈

原 vLLM kernel 的配置为：

```text
grid  = num_tokens
block = min(num_q_heads * rotary_dim / 2, 512)
```

Qwen 代表形状因此固定为一 token 一个 512-thread block。每个线程处理一个标量
旋转 pair，并通过循环覆盖 28 个 Q head；K 的 4 个 head 又走第二段循环。这有三个
疑点：

1. BF16 地址连续且 16-byte 对齐，却按标量 load/store；
2. 小 token 时 grid 只有几个 block，128 个 SM 很难被填满；
3. 大 token 时可以让同一 token 的线程复用 cos/sin，但原映射没有利用 vector
   宽度。

先用完全相同 Qwen shape 比较 vLLM 与现有 FlashInfer RoPE，发现 FlashInfer 在
1～2048 token 大多快 1.27x～1.74x，证明差距真实。随后阅读 FlashInfer
`pos_enc.cuh`，只吸收两条通用方法：每线程处理 16 bytes，以及根据工作量在
head-parallel/token-parallel 之间切换。

没有直接重新启用 vLLM 中被注释掉的 FlashInfer 路径，因为现有源码明确记录了
历史 failures，而且该路径要求 FP32 cos/sin cache。新实现保留 vLLM 当前 BF16
cache、稳定 ABI 和 fallback。

### 4.3 16-byte 向量映射

head size 为 128，BF16 每个元素 2 bytes。一个线程一次加载 8 个 BF16：

```text
128 elements / 8 elements = 16 threads per head
```

每线程加载自己的 8 个元素、与之配对的另 8 个元素，以及 8 个 cos/sin，使用
FP32 做乘加并转回 BF16。16 个线程共同覆盖完整 head；地址天然形成连续 16-byte
事务。

### 4.4 双并行策略

小/中 token 使用 head-parallel：

```text
grid.x = ceil(num_tokens / 8)
grid.y = num_q_heads + num_kv_heads
block  = (16, 8)
```

一个 CTA 处理一个 head 和最多 8 个 token。即使只有 1 token，也有 32 个 CTA，
比旧实现的 1 个 CTA 更能利用 128 个 SM。

较大 token 使用 token-parallel：

```text
grid.x = ceil(num_tokens / 8)
block  = (16, 8)
```

同一组 16 个线程先加载该 token 的 cos/sin，再循环处理所有 Q/K heads，减少重复
cache 读取。实测两条曲线在约 4096 token 汇合，因此最终规则为：

```text
num_tokens < 4096 -> head-parallel
otherwise         -> token-parallel
```

### 4.5 一次被数据否决的阈值

首轮假设在 256 token 切换到 token-parallel。它在 1～128 token 很快，但 256
token 从 baseline 的约 10.6 us 退化到 16.1 us，反而慢约 52%。原因是此时 token
grid 仍不足以补偿 head 维并行度的损失。

第二轮把阈值移到 1024，修复 256/512；最后只扫描 1024/2048/4096，发现
head-parallel 在 1024/2048 仍明显更快，到 4096 才与 token-parallel 汇合。于是
固定 4096 并停止，不继续无边界调参。

### 4.6 安全 fast path

只有以下条件全部满足才进入新 kernel：

- key 存在，positions 为生产路径的一维连续 tensor；
- NeoX style、head size = rotary dim = 128、offset = 0；
- Q/K/cache dtype 相同且元素宽度为 2 bytes；
- Q/K/cache 基地址 16-byte 对齐；
- token/head stride 能被 8 个元素整除。

FP32、交错式 RoPE、partial rotary、二维 positions、无 key、非对齐或其他 head
size 全部保留原 kernel。fast path 同时支持 fused-QKV split 后的非连续 token
stride。

### 4.7 最终精确 A/B

以下是 fused-QKV view 的 5-trial 中位数：

| tokens | 原 scalar kernel | 新 vector kernel | 加速 |
|---:|---:|---:|---:|
| 1 | 3.584 us | 1.952 us | **1.836x** |
| 4 | 3.616 us | 2.016 us | **1.794x** |
| 8 | 3.776 us | 2.080 us | **1.815x** |
| 16 | 3.936 us | 2.304 us | 1.708x |
| 32 | 4.416 us | 2.816 us | 1.568x |
| 64 | 5.472 us | 3.712 us | 1.474x |
| 128 | 8.032 us | 4.735 us | 1.696x |
| 256 | 10.559 us | 6.431 us | 1.642x |
| 512 | 19.103 us | 11.296 us | 1.691x |
| 1024 | 31.919 us | 20.895 us | 1.528x |
| 2048 | 55.965 us | 43.775 us | 1.278x |
| 4096 | 97.596 us | 86.460 us | 1.129x |

### 4.8 正确性

新增 Qwen fast-path 测试覆盖：

```text
FP16/BF16 × 1/257/4096 tokens：6 passed
```

Q/K 来自同一 fused-QKV tensor，覆盖真实非连续 token stride，并分别跨过
head-parallel 和 token-parallel 分支。

原通用 RoPE 矩阵：

```text
384 passed, 7 deselected
```

覆盖 NeoX/交错式、BF16/FP32、不同 head size、partial rotary、带/不带 key、flat/
四维/切片布局，确认 fallback 未回归。

## 5. 论文和开源实现分别提供了什么

- RoFormer 论文给出旋转位置编码的数学语义；本项目没有改变其公式；
- FlashInfer 论文提供 serving-native kernel 的总体设计背景；
- FlashInfer `pos_enc.cuh` 提供“16-byte vector + 依据并行度切换映射”的公开实现
  证据；
- 具体的 BF16 cache 兼容、vLLM stable ABI 接入、Qwen fused-QKV stride、安全条件、
  4096 阈值和 A/B 数据均由本项目在 RTX 4090 上独立实现和验证。

参考：

- RoFormer：https://arxiv.org/abs/2104.09864
- FlashInfer：https://arxiv.org/abs/2501.01005
- FlashInfer RoPE source：
  https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/pos_enc.cuh

## 6. 对端到端收益的边界

Qwen2.5-7B 有 28 层。仅按 RoPE kernel 差值粗略累加：

- decode 1 token：每层约省 1.632 us，28 层约 45.7 us；
- 512 token：每层约省 7.807 us，28 层约 219 us；
- 1024 token：每层约省 11.024 us，28 层约 309 us。

这只是 kernel 上限估算，不包含 GEMM、Attention、调度和 CPU wrapper，也不能直接
换算成服务吞吐。严谨表述是：

> 在 RTX 4090/Qwen2.5-7B 代表形状上，将 vLLM 生产 RoPE 从标量一 token 一
> block 改为 alignment-gated 16-byte vector 双并行 kernel，CUPTI 微基准获得
> 1.13x～1.84x 加速；RMSNorm 低 token 获得最高 1.04x。

## 7. 复现命令

```bash
/root/miniconda3/envs/vllm-dev/bin/python \
  benchmarks/kernels/benchmark_fused_add_rms_norm_cupti.py

/root/miniconda3/envs/vllm-dev/bin/python \
  benchmarks/kernels/benchmark_rotary_embedding_cupti.py

/root/miniconda3/envs/vllm-dev/bin/python -m pytest -q \
  tests/kernels/core/test_layernorm.py \
  -k 'test_rms_norm or test_rms_norm_weightless'

/root/miniconda3/envs/vllm-dev/bin/python -m pytest -q \
  tests/kernels/core/test_pos_encoding.py \
  -k qwen_rotary_embedding_fast_path
```

## 8. Phase 8 停止规则

本阶段只做两个算子，并已满足：真实生产调用、可解释瓶颈、同工具链 A/B、随机输入
正确性、通用 fallback、可复现脚本。没有继续优化第三个算子，也没有把微基准冒充
端到端吞吐。下一步若继续，应先做一次完整 serving A/B 或跨 GPU 验证，而不是继续
增加 shape heuristic。
