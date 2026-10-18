# 讲解 | 第2课 vLLM 核心组件：引擎模块和流式执行

> 本文是对 [第2课-vLLM+核心组件-引擎模块和流式执行.md](../学习文档/第2课-vLLM+核心组件-引擎模块和流式执行.md) 的精髓提炼，
> 源码注释已同步添加到 vllm 仓库（搜索 `【第2课` 即可找到全部 16 处注释）。

---

## 🎯 精髓一句话总结

**vLLM 把前端 AsyncLLM 和后端 Engine 拆成两个独立进程，靠 ZeroMQ 的三条通道配合：**

- **DEALER ↔ ROUTER**：AsyncLLM 把请求（ROUTER bind）路由到 Engine（DEALER connect）；
- **zmq.Poller**：单线程同时看多个 DEALER 套接字（I/O 多路复用）；
- **PUSH → PULL**：Engine 每生成一批 token 就推回前端；
- **每条请求一条私有队列（RequestOutputCollector）** + `generate()` 里 `yield`，就做到了真正的流式输出。

---

## 1️⃣ ZeroMQ 三种通信模式（为什么不用 REQ/REP）

| 模式 | 特点 | vLLM 中的用途 |
|---|---|---|
| **REQ/REP** | 严格一问一答，同步阻塞 ❌ | 不用 —— 吞吐量低、不能双工、不支持多客户端并发 |
| **DEALER/ROUTER** ✅ | 异步、双向、可按 identity 路由；ROUTER 自动把消息加上发送方标识 | **请求下发通道**：ROUTER（AsyncLLM/前端）→ DEALER（Engine） |
| **PUSH/PULL** ✅ | 单向数据流，天然适合流式推送 | **结果回推通道**：PUSH（Engine）→ PULL（AsyncLLM） |
| **zmq.Poller** ✅ | I/O 多路复用（内核态等待），一个线程看多个 socket | **Engine 输入线程**监听多个 DEALER 套接字的可读事件 |

> 💡 比喻：
> - REQ/REP = 你打电话必须等对方说一句你才能说，连续说两句直接挂。
> - DEALER/ROUTER = 微信群：每个人(DEALER)发消息自带昵称(identity)，群主(ROUTER)可以精准@回复。
> - PUSH/PULL = 直播间：主播(Engine)只管推流，观众(AsyncLLM)只管看，不需要回话。

### 关键机制补充

1. **identity 路由**：Engine 的 DEALER 启动时先发一条 ready 消息"报到"，让前端 ROUTER 记住自己的 identity；之后 ROUTER 发请求时在消息头带上目标 identity，就能精确投递到对应 Engine。

2. **ready 消息还捎带配置**：`EngineCoreReadyResponse` 里包含 `max_model_len`、`num_gpu_blocks` 等启动后才能确定的配置，一并同步给前端。

3. **multipart 消息**：ROUTER 收到的消息是 `[identity, 业务帧...]` 的多段式结构；业务帧又分为 `类型帧(ADD/ABORT/UTILITY) + 数据帧`。

---

## 2️⃣ 数据完整链路（4 段关键函数）

```text
用户 HTTP 请求
    │
    ▼
【AsyncLLM.generate()】                          ← async_llm.py
  ├─ add_request() → 创建【请求专属队列 q】
  ├─ _add_request()
  │   ├─ ① output_processor.add_request(注册状态+绑定 q)   ← output_processor.py
  │   └─ ② engine_core.add_request_async(发往 Engine)     ← core_client.py
  │          └─ _send_input() → ROUTER 发: (identity + type + payload)
  │
  ▼═══════════════ DEALER/ROUTER (ZMQ) ═══════════════╗
  ║                                                     ║
Engine 端 (EngineCoreProc，独立进程)                    ║
  ├─ 【process_input_sockets】← DEALER 收请求           ← core.py  输入线程
  │   · zmq.Poller 监听 → recv_multipart → 解码 → 入 input_queue
  │
  ├─ 【_process_input_queue】← 主循环消费 input_queue   ← core.py  主循环
  │   · get() → _handle_client_request → 调度器 → Executor → 模型 step
  │
  ├─ 模型每 step 产出 → 入 output_queue
  │
  └─ 【process_output_sockets】← PUSH 推结果             ← core.py  输出线程
      · get() → 按 client_index 选 socket → send_multipart
  ║                                                     ║
  ▼═══════════════ PUSH/PULL (ZMQ) ════════════════════╝

前端 AsyncLLM 端
  ├─ 【AsyncMPClient.process_outputs_socket】← PULL 接  ← core_client.py
  │   · recv_multipart → 解码 → 入公共 outputs_queue
  │
  ├─ 【AsyncLLM.output_handler】后台任务循环             ← async_llm.py
  │   · get_output_async() → output_processor.process_outputs()
  │
  ├─ 【OutputProcessor.process_outputs】                 ← output_processor.py
  │   · 按 request_id → 找 RequestState → detokenize → 判断 stop
  │   · req_state.queue.put(RequestOutput)  ← 放入请求私有队列
  │
  └─ 【generate() while 循环】
      · q.get_nowait() or await q.get() → yield 一个分块  ← 流式返回给用户
```

### Engine 里的生产者-消费者（两条线程 + 一个队列）

| 角色 | 函数 | 职责 |
|---|---|---|
| **生产者** | `process_input_sockets`（后台输入线程） | 从 ZMQ socket 收请求 → 反序列化 → 写入 `input_queue` |
| **消费者** | `_process_input_queue`（Engine 主循环） | 从 `input_queue` 取请求 → 交给调度器和执行器 |

两者通过 `input_queue` 解耦：网络 IO 与引擎调度互不阻塞。

---

## 3️⃣ 流式返回三步曲（课程精髓）

```text
1️⃣ 提交请求 → q = add_request()       // 为 request_id 创建私有队列 q
2️⃣ 等待就绪 → out = await q.get()     // output_handler 把结果推入 q
3️⃣ 即时产出 → yield out               // 拿到一块就返回，不累积
```

核心代码（generate 里的循环，最重要的几行）：

```python
q = await self.add_request(...)          # 拿到该请求的专属队列
finished = False
while not finished:
    out = q.get_nowait() or await q.get()  # 队列有数据立即取
    finished = out.finished
    yield out                              # 每拿一块就返回一块 —— 流式!
```

**关键点**：Engine 每 step 一次（可能多个请求各产 1 token）就 PUSH 一次 → 前端解包后按 request_id 分别入各自私有队列 → `generate()` 端谁的队列有数据就 `yield` 谁的，互不阻塞。这就是你在网页上看到"字一个一个蹦"的原因。

---

## 4️⃣ vLLM 源码对应位置一览（已加中文注释，搜"【第2课"）

### 📄 `vllm/v1/engine/core.py`（Engine 进程端）

| 知识点 | 函数 | 对应课程节 |
|---|---|---|
| 输入队列消费者（主循环取请求） | `_process_input_queue` | 生产者-消费者 / input_queue |
| DEALER 接收请求 + Poller + ready 报到 + 入队 | `process_input_sockets` | DEALER/ROUTER、zmq.Poller、Engine 接收请求 |
| PUSH 发送推理结果（输出线程） | `process_output_sockets` | PUSH/PULL、Engine 发送数据 |

### 📄 `vllm/v1/engine/core_client.py`（AsyncMPClient 前端端）

| 知识点 | 函数 | 对应课程节 |
|---|---|---|
| ROUTER socket bind | `__init__`（两处 make_zmq_socket） | AsyncLLM 发送请求 / ROUTER 端 |
| PULL 端接收 Engine 回推 → 入公共 outputs_queue | `process_outputs_socket` | AsyncLLM 接收结果 / 拉取结果 |
| 请求打包成消息帧 | `_send_input` | AsyncLLM 转发请求 / 封装 |
| ROUTER 按 identity 发送 multipart | `_send_input_message` | ROUTER → DEALER 路由投递 |
| ADD 请求入口（打 client_index 标记） | `add_request_async` | AsyncLLM 发送请求 |

### 📄 `vllm/v1/engine/async_llm.py`（前端 AsyncLLM 主类）

| 知识点 | 函数 | 对应课程节 |
|---|---|---|
| 为每个请求创建**专属输出队列** | `add_request` | 创建结果接收队列 / RequestOutputCollector |
| 双轨：本地注册 + 远程转发 | `_add_request` | 转发请求 + OutputProcessor.add_request |
| 拉取 EngineCoreOutputs → process_outputs 分发 | `_run_output_handler` 里的 `output_handler` | 结果放入队列 / output_handler |
| 流式三步：提交 → 等 → yield | `generate` | 流式返回 / generate() yield |

### 📄 `vllm/v1/engine/output_processor.py`（请求状态管理 + 分发）

| 知识点 | 函数 | 对应课程节 |
|---|---|---|
| request_id → RequestState（绑定 queue） | `add_request` | OutputProcessor 注册请求状态 |
| 主循环：按 request_id 拆 → 解码 → 判断 stop → 入私有队列 | `process_outputs` | 结果放入队列 / process_outputs 四步 |

---

## 5️⃣ 一张速记表

```text
请求方向：  AsyncLLM(ROUTER.bind) ──identity 路由──▶ Engine(DEALER.connect)
            ↑                                                 │
            │  先 ready_payload 报到（让 ROUTER 记住 identity）│
            │                                                 ▼
            │                                          input_queue.put
            │                                                 │
            │                                         _process_input_queue.get
            │                                                 ▼
            │                                          调度器 + Executor 推理
结果方向：  AsyncLLM(PULL) ◀────────────────────────── Engine(PUSH)
            ▲                                                 │
            │                                       output_queue.put
            │                                                 │
  process_outputs_socket.recv_multipart          process_output_sockets.send_multipart
            │
            ▼
  outputs_queue(公共) → output_handler → process_outputs(request_id 查表)
                                              │
                                              ▼
                         req_state.queue.put(RequestOutput)   ← 每条请求私有队列
                                              │
                                              ▼
                         generate() await q.get() + yield    ← 流式吐给用户
```

---

## ⚠️ 可跳过的高级内容

课程后半段提到的 **DP coordinator / XSUB**、**FaultTolerance**、**P0/P1 MM cache drift**、**elastic EP scaling** 属于多前端 + 数据并行扩展场景，入门时可以先跳过，先把"单 Engine + 单 AsyncLLM"主线跑通即可。

---

## 🔍 调试小技巧（来自课程）

- 断点打在 `vllm/v1/engine/core.py` 的 `process_input_sockets` / `process_output_sockets`；
- attach 的目标是 **EngineCore worker 所在进程**（看 `nvidia-smi` 中占显存的 PID），而不是 `vllm.pid` 输出的 PID；
- 用 curl 发请求即可触发断点：

```bash
curl -s --noproxy '*' http://127.0.0.1:13311/v1/chat/completions \
 -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "用20字介绍vLLM"}],
    "max_tokens": 30,
    "temperature": 0.6
 }'
```
