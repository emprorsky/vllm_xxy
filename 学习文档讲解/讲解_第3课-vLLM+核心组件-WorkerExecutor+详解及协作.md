# 讲解 | 第3课 vLLM 核心组件：Worker/Executor 详解及协作

> 本文是对 [第3课-vLLM+核心组件-WorkerExecutor+详解及协作.md](../学习文档/第3课-vLLM+核心组件-WorkerExecutor+详解及协作.md) 的精髓提炼，
> 源码注释已同步添加到 vllm 仓库（搜索 `【第3课` 即可找到全部 54 处注释，分布在 7 个文件）。

---

## 🎯 精髓一句话总结

**Engine 不算模型，Executor 不算模型，Worker 才算模型。三层分工 + 一套"广播下发、单点回传"的类 RPC 机制：**

- **Engine（EngineCoreProc）**：总调度台——收请求、做调度、把执行计划交给 Executor；
- **Executor（MultiprocExecutor）**：二房东——spawn 一批 Worker 子进程（一个进程一张卡），把任务**广播**下去、把结果**收回来**，自己不碰 GPU；
- **Worker**：真正干活的人——持有权重分片 + KV cache + 一张 GPU，执行前向计算；
- 通信全靠 **1 + N 条消息队列**：1 条 `rpc_broadcast_mq` 广播下行（共享内存 + ZMQ 混合传输），N 条 `worker_response_mq` 独立回传结果。

> 💡 比喻：Executor 像包工头在群里 @所有人 发任务（一条广播），Worker 们各自干活，但只有"指定发言人"（output_rank）把答案私聊发回给包工头。

---

## 1️⃣ 三层分工：谁在调度，谁在干活

| 组件 | 进程 | 职责 | 不做什么 |
|---|---|---|---|
| **EngineCoreProc** | Engine 进程 | 收前端请求、调度（组 batch）、调 `model_executor` | 不做模型计算 |
| **Executor** | Engine 进程内 | 选 Worker 类、spawn 子进程、广播任务、收结果 | 不碰 GPU、不持有权重 |
| **Worker** | **独立子进程** | 绑定一张 GPU、持有权重分片 + KV cache、执行 `execute_model` 前向 | 不管调度 |

Worker 的核心接口（`WorkerBase`）：`execute_model`（前向）、`sample_tokens`（采样）、`load_model`、`get_kv_cache_spec` / `initialize_from_config` / `compile_or_warm_up_model`（KV Cache 三件套）、`add_lora` 等。

---

## 2️⃣ Executor 家族与 world_size

### 三种 Executor（`Executor.get_class()` 按 `distributed_executor_backend` 选择）@

| 后端 | 类 | 场景 |
|---|---|---|
| `"uni"` | UniProcExecutor | 单 GPU / 简单测试，Executor 与 Worker 同进程 |
| `"mp"` ✅ | **MultiprocExecutor** | **单机多 GPU**，用 Python multiprocessing 拉起子进程，避免 Ray 开销 |
| `"ray"` | RayDistributedExecutor | 跨机分布式，Ray 负责多机拉起 |
| `"external_launcher"` | ExecutorWithExternalLauncher | torchrun/slurm 拉起全部进程 |

---------------
### 【补充解释：四种后端到底什么意思？】

想象你要开一家工厂（推理服务），有四种管理方式：

- **`"uni"`（单 GPU 模式）**：老板亲自下车间。Executor 和 Worker 在**同一个进程**里，没有子进程、没有消息队列。适合只有 1 张卡的开发/测试环境，最简单直接。
- **`"mp"`（多进程模式，最常用）**：老板在办公室，派多个车间主任（Worker 子进程）各管一张卡。用 Python 自带的 `multiprocessing` 创建子进程，**只能在单机内**。这是本课的主角 MultiprocExecutor。
- **`"ray"`（Ray 分布式模式）**：工厂跨了好几栋楼（多台机器），Ray 是跨机调度管家，负责在多台机器上拉起 Worker 进程、传递对象。适合多机集群。
- **`"external_launcher"`（外部启动器模式）**：不用 vLLM 自己拉进程，而是由 `torchrun` 或 `slurm` 在外面**一次性把所有进程拉起来**，vLLM 只是"被拉起来的一个进程"。适合超算/集群环境已有统一进程管理工具的场景。

一句话：**单卡选 uni，单机多卡选 mp，跨机选 ray，已有集群管理工具选 external_launcher。**
---------------

**注意**：`get_class()` 返回的是**类**不是实例。真正实例化发生在 EngineCore 后台进程里：@
`AsyncLLM.from_vllm_config → EngineCoreClient.make_async_mp_client → EngineCoreProc → EngineCore.__init__ → executor_class(vllm_config)`

### world_size 两个口径@

| 属性 | 公式 | 含义 | 例（TP=2, PP=2, DP=2） |
|---|---|---|---|
| `world_size` | TP × PP × PCP | **单个 DP 副本**（一个模型副本）内的 Worker 数 | 4 |
| `world_size_across_dp` | world_size × DP | 所有 DP 副本的总进程数（多机总规模） | 8 |

---------------
### 【补充解释：PCP 是什么？外部 DP 和 DP 有什么区别？】

**PCP（Prefill Context Parallel，预填充上下文并行）**：
- 当用户发来一个**超长 prompt**（比如 10 万 token），一张卡的显存装不下整个 prompt 的 KV cache。
- PCP 把这个长 prompt **沿序列方向切成几段**，分给多张卡同时算，最后再合并。
- 默认 PCP=1（不启用），只有处理超长上下文时才开。
- 简单理解：TP 是"把模型切碎"，PCP 是"把输入切碎"。

**DP vs 外部 DP（ExternalDP）的区别**：
- **DP（Data Parallel，数据并行）**：你有多个**完整的模型副本**（每个副本都有 TP×PP×PCP 个 Worker），不同副本处理不同用户的请求。就像开两家分店，各自独立接客。
- **外部 DP（ExternalDP）**：这是 vLLM 与外部训练框架（如 verl）集成时的概念。外部框架自己管着多个 DP 副本，vLLM 只是被它调用的推理引擎。ExternalDP 这个维度告诉 vLLM："你只是更大系统里的一环"。
- 一句话：**DP 是 vLLM 自己管多个副本；ExternalDP 是外部系统管的多个副本，vLLM 不操心**。
---------------

- `external_launcher` 时 `world_size` 还要 × DP（外部启动器看到的是全局 rank）。
- Python multiprocessing **不能跨机器** spawn 进程 → 跨机 DP 必须 Ray / Slurm / torchrun 解决"跨机拉起 + rank/地址对齐"。

---

## 3️⃣ 并行分组：一个 reshape 搞定 TP/PP/DP/EP@

`initialize_model_parallel` 把一维 rank 列表重塑为 **5D 张量**，再用"**转置目标维到最后 → reshape 2D → unbind 按行拆**"三步取组：

```text
all_ranks = torch.arange(world_size).reshape(ExternalDP, DP, PP, PCP, TP)
```

以 8 卡 TP=2 / PP=2 / DP=2 为例（TP 在最内侧）：

| 并行组 | 取组方法 | 分组结果 | 用途 |
|---|---|---|---|
| **TP 组** | 直接连续切（最内侧维度） | `[g0,g1] [g2,g3] [g4,g5] [g6,g7]` | 层内权重切分，NCCL **all-reduce** 同步 |
| **PP 组** | `transpose(2,4)` 后切 | `[g0,g2] [g1,g3] [g4,g6] [g5,g7]` | 层间 **send/recv** 传激活值 |
| **DP 组** | `transpose(1,4)` 后切 | `[g0,g4] [g1,g5] [g2,g6] [g3,g7]` | 相同位置、不同副本对齐（推理时基本不通信） |
| **EP 组**（MoE） | `transpose(1,2)` 按 DP×PCP×TP 切 | `[g0,g1,g4,g5] [g2,g3,g6,g7]` | 同一 PP stage 上联合承载专家权重 |

---------------
### 【补充解释：分组结果怎么读？】

先把 8 张卡编号 g0~g7，初始排列写成一个方阵（行=DP×PP，列=TP）：

```
         TP_0  TP_1
DP_0 PP_0  g0    g1
DP_0 PP_1  g2    g3
DP_1 PP_0  g4    g5
DP_1 PP_1  g6    g7
```

现在用"**谁需要通信，谁排在一行**"的思路来理解每种分组：

- **TP 组 `[g0,g1] [g2,g3] [g4,g5] [g6,g7]`**：同一行（同一张表的左右两列）的卡，它们合起来算一整层的权重。比如 g0 和 g1 算同一层，算完要 all-reduce 合并结果，所以它们必须通信 → 分一组。共 4 组，每组 2 卡。

- **PP 组 `[g0,g2] [g1,g3] [g4,g6] [g5,g7]`**：同一列上下相邻的卡，它们是"流水线上的上下游"。g0 算完模型的前半部分（PP stage 0），把激活值**传给** g2 接着算后半部分（PP stage 1）。所以 g0 和 g2 要通信 → 分一组。每个 TP 分片各成一条链。

- **DP 组 `[g0,g4] [g1,g5] [g2,g6] [g3,g7]`**：跨两个模型副本、但处于**相同位置**的卡。g0 在副本 0 当 TP_0/PP_0，g4 在副本 1 也当 TP_0/PP_0。它们干同样的活、处理不同用户，推理时基本不通信，但需要隔离保证 TP/PP 通信不串线。

- **EP 组 `[g0,g1,g4,g5] [g2,g3,g6,g7]`**：MoE 模型专属。把同一层（同一 PP stage）上的所有卡合并成一组，一起承载分散的专家权重。组 0 = PP stage 0 的全部 4 卡，组 1 = PP stage 1 的全部 4 卡。

> 💡 一句话记忆：**TP=同一层横向合伙、PP=流水线纵向传递、DP=不同副本相同位置、EP=同一层所有卡合伙放专家**。
---------------

> 💡 记住一点：**TP 切层内、PP 切层间、DP 是多个完整副本**。TP=2 时每个 Worker 持有**每层权重的一半**（不是一半的层）。

---

## 4️⃣ Worker 初始化三阶段（每个 Worker 进程独立执行）@

```text
WorkerWrapperBase.__init__()          ← 先建"壳"，记住 worker 类名
  └─ init_worker()                    ← 加载插件、解析 worker_cls、动态混入 extension
      └─ init_device()                ← 【1. Init Device】
      │     绑 GPU(local_rank) + 初始化 NCCL 通信组 + 创建 ModelRunner/InputBatch
      └─ load_model()                 ← 【2. Load Model】
      │     实例化模型 + 按 TP/PP 切分加载权重 + eval() + 可选 torch.compile
      └─ initialize_from_config()     ← 【3. Initialize KV Cache】
            get_kv_cache_spec → dummy run 估算显存 → 分配 KV block → 绑定注意力层
            （之后 compile_or_warm_up_model 做 warmup + CUDA Graph 捕获）
```

初始化完成后 Worker **不退出**，进入 `worker_busy_loop` 无限循环等待任务。

---

## 5️⃣ 通信机制：1 + N 队列 + 混合传输（本课核心）

### 队列结构（TP=2 时共 3 条）@

```text
                    ┌─────────────────────────────┐
   Executor ──广播──▶  rpc_broadcast_mq（1 条）    ├──▶ Worker0（dequeue 拿到相同任务）
                    │  = ShmRingBuffer + ZMQ XPUB │
                    │                             ├──▶ Worker1（dequeue 拿到相同任务）
                    └─────────────────────────────┘
   Executor ◀──单点── worker_response_mq[0]（Worker0 本地新建） ◀── Worker0 enqueue
   Executor ◀──单点── worker_response_mq[1]（Worker1 本地新建） ◀── Worker1 enqueue
```

- **下行 `rpc_broadcast_mq`**：Executor 是 Writer；组内 Worker 都是 Reader，拿到**相同**的任务（TP/PP/PCP 协同处理同一个 SchedulerOutput，广播一次就够）。
- **上行 `worker_response_mq`**：每个 Worker **本地新建**，把 handle 通过 ready_pipe 回传给 Executor，Executor 再 `create_from_handle` 反向接入——保证各 Worker 结果互不混淆。

### MessageQueue 双通道混合传输（`shm_broadcast.py`）

| 通道 | 条件 | 特点 |
|---|---|---|
| **ShmRingBuffer 共享内存** | payload < `max_chunk_bytes`（默认 **16MB**，`VLLM_MQ_MAX_CHUNK_BYTES_MB`） | 零拷贝，多个 Reader 各自维护读游标 |
| **ZMQ XPUB/SUB** | payload ≥ 阈值 或 跨节点 | 共享内存首字节写 `1`（溢出标记），真实数据走 socket；跨节点走 TCP PUB/SUB |

Reader 端 `dequeue` 先看**首字节**：`0` = 数据在共享内存直接解析；`1` = 溢出，从 SUB socket 接收。

---

## 6️⃣ 两次握手（别混淆）

| 握手 | 函数 | 层级 | 过程 |
|---|---|---|---|
| **① 进程级** | `WorkerProc.wait_for_ready`（Executor 端） | 子进程初始化 | Worker 完成 init_device + load_model + 建队列后，通过 **ready_pipe** 发 `{status: "READY", handle: ...}`；Executor 由此拿到 response_mq 的 handle |
| **② 队列级** | `MessageQueue.wait_until_ready`（双方都调） | ZMQ 连接配对 | 依赖 `XPUB_VERBOSE=True`：每个 Reader 订阅时 Writer 收到一条通知 → Writer 收齐 N 条后广播 `b"READY"` → Reader 收到即通道就绪。类似 TCP 三次握手 |

顺序：先 ① 拿到 handle，再 ② 确认通道连通，之后 Worker 才进入 busy loop。

---

## 7️⃣ 一次 execute_model 的完整生命周期@

```text
EngineCore.step() 调度出 SchedulerOutput
    │
    ▼
Executor.execute_model(scheduler_output)
    └─ collective_rpc("execute_model", args=(scheduler_output,),
                      unique_reply_rank=self.output_rank)
        │
        ├─ ① rpc_broadcast_mq.enqueue((方法名, args, kwargs, output_rank))
        │      小数据 → SHM / 大数据 → ZMQ XPUB
        │
        ▼ ═══════════ 广播到组内所有 Worker ═══════════
Worker0 / Worker1（各自进程的 worker_busy_loop）
    ├─ ② rpc_broadcast_mq.dequeue() → 还原 (method, args, kwargs, output_rank)
    ├─ ③ getattr(self.worker, method)(*args) → 前向计算（TP 用 NCCL 内部同步）
    └─ ④ 只有 rank == output_rank 的 Worker 把结果 enqueue 到 worker_response_mq
        │
        ▼
Executor 从 response_mqs[output_rank].dequeue() 拿到 ModelRunnerOutput → 返回 EngineCore
```

---------------
### 【补充解释：这个图怎么看？（初学者版）】

把整个图想象成**餐厅接单流程**：

**第 0 步：调度器出菜单** —— `EngineCore.step()` 把当前要处理的请求打包成一份 `SchedulerOutput`（就像服务员把桌号和菜品写在订单上）。

**第 1 步：Executor 接单 + 广播** —— `Executor.execute_model()` 拿到订单，调用 `collective_rpc()`，把"方法名(execute_model) + 参数(scheduler_output) + 谁负责回结果(output_rank)"这四样东西打包，塞进广播队列 `rpc_broadcast_mq`。小订单走共享内存（快），大订单走 ZMQ（能装多）。

---------------
### 【补充解释：共享内存和 ZMQ 分别是什么？】@

**共享内存（Shared Memory，SHM）**：
- 正常情况下，两个进程之间传数据，要先"复制"一份——就像你要给同事一份文件，得先复印再递过去。
- 共享内存是操作系统提供的一种"**两个进程看同一块物理内存**"的机制。Executor 把数据写进去，Worker 直接从同一块内存读，**不用复制**，所以特别快。
- 但共享内存有个限制：**大小是预先分配好的**（这里默认 16MB 一格）。如果你要传的数据超过这个大小，就放不下了。
- 比喻：共享内存就像一张**共享白板**，写上去大家都能看到，但白板面积有限，写不下大文件。

**ZMQ（ZeroMQ 套接字）**：
- ZMQ 是一个高性能的消息传递库，类似"加强版的 socket"。它能传任意大小的数据，不受固定内存大小限制。
- 当数据太大（≥ 16MB），共享内存放不下时，就改用 ZMQ 的 XPUB/SUB 模式发送：Executor 发到 XPUB，Worker 用 SUB 订阅接收。
- 比喻：ZMQ 就像一个**快递公司**，能寄大包裹，虽然比白板慢一点，但什么都能寄。

**vLLM 的聪明之处：自动二选一**
- 数据 < 16MB → 走共享内存白板（快，零拷贝）
- 数据 ≥ 16MB → 走 ZMQ 快递（能装大件）
- Worker 读取时先看共享内存的**第一个字节**：`0` 表示"数据在白板上，直接读"；`1` 表示"数据太大溢出了，去 ZMQ 快递柜取"。
---------------

**第 2 步：所有 Worker 同时收到订单** —— 图中 `═══ 广播 ═══` 那条线就是"广播"。Worker0 和 Worker1 在各自的 `worker_busy_loop` 里 `dequeue()`，**同时**拿到一份一模一样的订单。因为是广播，发一次就够了，不用逐个发。

**第 3 步：每个 Worker 各自干活** —— `getattr(self.worker, method)` 就是"找到 Worker 身上的 execute_model 方法并调用"。Worker0 和 Worker1 各自用自己那张卡算前向传播。如果是 TP 模式，它们算的是**同一层的不同部分**，算完用 NCCL all-reduce 合并。

**第 4 步：只有"指定发言人"回结果** —— `rank == output_rank` 的那个 Worker（通常是最有一个 PP stage 的第一个 TP rank）把最终结果塞进自己的 `worker_response_mq`。其他 Worker 算完就算了，不回传。因为 TP 模式下大家算出来的 logits 一样，收一份就够，省带宽。

**第 5 步：Executor 收货 + 交差** —— Executor 从 `response_mqs[output_rank]` 里 `dequeue()` 拿到 `ModelRunnerOutput`，交给 `EngineCore` 去做后续（采样、返回给用户）。

> 核心记住四个字：**广播执行、单点回传**。发任务时一声令下所有人都听，收结果时只听一个人汇报。@
---------------

**output_rank = world_size - tp_size**（最后一个 PP stage 的第一个 TP rank）。TP 模式下各 rank 经 all-reduce 后 logits 完全一致，**只收一份就够**——这就是"广播执行、单点回传"。

---

## 8️⃣ vLLM 源码对应位置一览（已加中文注释，搜"【第3课"）

### 📄 `vllm/v1/executor/multiproc_executor.py`（21 处，本课主战场）

| 知识点 | 函数/位置 |
|---|---|
| world_size 校验、创建广播队列、导出 Handle | `_init_executor` |
| 1+N 队列结构、队列级握手调用 | `_init_executor`（response_mqs 收集段） |
| 广播执行 + 单点回传 | `execute_model` |
| 类 RPC 核心逻辑（enqueue 四元组、等 response） | `collective_rpc` |
| output_rank 的选取（= world_size - tp_size） | `_get_output_rank` |
| spawn Worker 子进程（一进程一卡） | `WorkerProc.make_worker_process` |
| 进程级握手（收 READY + handle） | `WorkerProc.wait_for_ready` |
| Worker 生命周期 + 双向握手 + busy loop | `WorkerProc.worker_main` |
| 重建广播队列 Reader 端 / 新建回传队列 | `WorkerProc._init_message_queues` |
| Worker 事件循环（dequeue → 执行 → 按 rank 回传） | `worker_busy_loop` / `_execute_worker_rpc` |

### 📄 `vllm/distributed/device_communicators/shm_broadcast.py`（11 处）

| 知识点 | 函数 |
|---|---|
| 双通道混合传输总览 | `MessageQueue` 类注释 |
| XPUB + XPUB_VERBOSE 订阅通知 | `MessageQueue.__init__` |
| Handle 导出 / Reader 端重建 | `export_handle` / `create_from_handle` |
| 队列级双向握手 | `wait_until_ready` |
| 首字节 0/1 分流（SHM vs ZMQ） | `enqueue` / `dequeue` |

### 📄 `vllm/v1/executor/abstract.py`（8 处）

| 知识点 | 函数 |
|---|---|
| 三层分工定位 | `Executor` 类 docstring |
| uni / mp / ray / external_launcher 选择 | `Executor.get_class` |

### 📄 `vllm/v1/worker/gpu_worker.py`（4 处）

| 知识点 | 函数 |
|---|---|
| Worker 三阶段初始化 1/2/3 | `init_device` / `load_model` / `initialize_from_config` |

### 📄 `vllm/v1/worker/worker_base.py`（2 处）

| 知识点 | 函数 |
|---|---|
| 初始化调用链 + 动态混入 extension | `WorkerWrapperBase` docstring / `init_worker` |

### 📄 `vllm/distributed/parallel_state.py`（5 处）

| 知识点 | 函数 |
|---|---|
| 5D reshape + TP/PP/DP/EP 四种分组 | `initialize_model_parallel` |

### 📄 `vllm/config/parallel.py`（3 处）

| 知识点 | 位置 |
|---|---|
| world_size = TP×PP×PCP、external_launcher ×DP | `ParallelConfig.__post_init__` |
| world_size_across_dp | property |

---

## 9️⃣ 一张速记表@@

```text
启动期：
  get_class 选 Executor ──▶ EngineCore 进程内实例化 MultiprocExecutor
      │
      ├─ 创建 rpc_broadcast_mq（SHM + XPUB）── export_handle ──┐
      ├─ spawn N 个 Worker 子进程（一进程一卡） ◀───────────────┘ handle 传入
      │       └─ Worker: init_device → load_model → init_kv_cache
      │                → ready_pipe 发 READY + response_mq handle（握手①）
      └─ wait_for_ready 收 handle → wait_until_ready ZMQ 配对（握手②）

运行期（每个 step）：
  SchedulerOutput ─▶ collective_rpc ─▶ broadcast_mq 广播(所有 Worker 执行)
                                      └▶ 仅 output_rank 回传 response_mq
  大数据走 ZMQ(首字节=1) / 小数据走共享内存(首字节=0)
```

---------------
### 【补充解释：速记表怎么看？（初学者版）】

这张表把 vLLM 多卡推理的**一生**浓缩成两个阶段：**启动期**（只发生一次）和**运行期**（每次推理重复）。

**启动期（开机只做一次，做好就不再重来了）：**

1. `get_class 选 Executor`：根据你设的后端（mp/ray/uni），选一个 Executor 类。
2. `实例化 MultiprocExecutor`：在 EngineCore 进程里把 Executor 造出来。
3. `创建 rpc_broadcast_mq`：Executor 先建一条广播队列，内部含共享内存 + ZMQ 两种通道。`export_handle` 是把这条队列的"连接地址"打包成可传递的小卡片。
4. `spawn N 个 Worker 子进程`：用 multiprocessing 创建 N 个子进程，每个绑一张卡。上面打包的 handle 卡片随创建过程传给每个 Worker。
5. `Worker 初始化`：每个子进程依次做三件事——绑设备(`init_device`)、加载模型权重(`load_model`)、初始化 KV cache(`init_kv_cache`)。
6. `握手①`：Worker 初始化完后，通过 `ready_pipe`（一根管道）告诉 Executor "我准备好了(READY)"，同时把自己的回传队列 handle 塞进去。
7. `握手②`：双方再调 `wait_until_ready`，确认 ZMQ 收发通道真正连通（类似 TCP 三次握手）。

**运行期（每收到一波请求就循环一次）：**

8. `SchedulerOutput`：调度器决定这轮处理哪些请求，产出一份执行计划。
9. `collective_rpc`：Executor 把计划塞进广播队列。
10. `广播`：所有 Worker 从队列里同时拿到同样的计划，各自执行前向计算。
11. `仅 output_rank 回传`：只有"指定发言人" Worker 把结果塞回 Executor。
12. `大数据/小数据`：传输时按 16MB 分界，小的走共享内存（快），大的走 ZMQ（能装）。

> 一句话总结：**启动期建管道 + 拉进程 + 握手 → 运行期广播任务 + 单点回结果**。启动期像装修店铺（做好不变），运行期像日常接客（循环往复）。
---------------

---

## ⚠️ 可跳过的高级内容

课程中以下内容属于多机/MoE 扩展场景，入门时先跳过，把"单机 mp + TP"主线跑通即可：

- **elastic EP**（弹性专家并行，stateless group + TCP store）与 **EPLB** 专家负载均衡；
- **PCP/DCP**（Prefill/Decode 上下文并行，超长 prompt 沿序列维切分）；
- **多节点**的 `inner_dp_world_group` / `peer_worker_response_mqs`（远端 Reader 走 TCP PUB/SUB）；
- **external_launcher**、KV/EC output aggregator（收集所有 rank 输出再聚合）。

---

## 🔍 调试小技巧（来自课程）

- Executor 初始化发生在**启动阶段**（用户请求来之前就完成了），**attach 调试赶不上**——要用 launch 模式：`.vscode/launch.json` 以 `vllm.entrypoints.openai.api_server` 为 module 启动，断点打在 `vllm/v1/executor/abstract.py` 的 `get_class`；
- TP=2 启动后，`distributed_executor_backend='mp'` 会自动选中 `MultiprocExecutor`，`worker_cls='vllm.v1.worker.gpu_worker.Worker'`；
- 想.Debug Worker 的 busy loop，断点打在 `multiproc_executor.py` 的 `worker_busy_loop` / `_execute_worker_rpc`，用 curl 发请求触发：

```bash
curl -s --noproxy '*' http://127.0.0.1:13333/v1/chat/completions \
 -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "用20字介绍vLLM"}],
    "max_tokens": 30,
    "temperature": 0.6
 }'
```
