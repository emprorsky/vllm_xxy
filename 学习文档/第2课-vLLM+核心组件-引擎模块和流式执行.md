> 🔗 原文链接：[‌‍⁣⁢⁣‍⁣​‌⁡第2课-vLLM 核心组件-引擎模块和流式执行](https://tvle9mq8jh.feishu.cn/docx/O8hSdQCTCo0WPFxaKhLczrP5nzf)

> ⏰ 剪存时间：2026-07-24 22:53:07

> ✂️ 本文档由 [游侠飞书剪存](https://pwwjpto7tva.feishu.cn/wiki/space/7517832277555544092) 一键生成

> 💖 更多好物请访问[游侠创客](https://uibot.cn) 微信：xuefuta

⚠️温馨提示：文档中包含【1个】暂不支持的区域，请通过搜索关键字【暂不支持的文档区域】进行后续处理

## 从零入门的Ai-Infra课程

1. [📄《自制大模型推理框架》课程目录-支持LLama3和Qwen3](https://tvle9mq8jh.feishu.cn/docx/AGb0dpqwfohQ9oxx4QycqbCjnJh)

2. [📄《速通OpenAI Triton：手写轻量级多模态推理框架》课程目录](https://tvle9mq8jh.feishu.cn/docx/QaYhdI0VzofUXyxGWvKcpyEMnih)

3. [📄目录-面向AI-Infra的Cuda零基础入门](https://tvle9mq8jh.feishu.cn/docx/BnqMdyaJ9oyXb1xwktgc7esMn4c)

**课程介绍请点击上方链接**

![](data:image/jpeg;base64...)

在 vLLM 的架构中，Engine 模块根据使用场景分为用于离线推理的 LLMEngine 和支持在线服务的AsyncLLM，本章将重点聚焦于在线推理部分。

## ZMQ 通信

如前所述，AsyncLLM 负责从客户端接收输入请求，并将其转发至底层实际执行推理任务的引擎组件，随后以异步方式获取执行结果。二者之间的通信依托于 ZeroMQ（ZMQ）实现。

### REQ/REP通信模式

为了让大家更直观地理解 ZMQ 的多种通信模式，我们将在本例中启动一个 server 和一个 client：client 端负责向 server 端发送 "hello" 字符串，server 端回复同样的字符串作为响应。第一个例子采用**请求-响应（REQ/REP）模式**，client 端将以同步阻塞的方式等待 server 端的响应回复。

![](data:image/png;base64...)

**启动命令：**python vllm\_learn/code/course1/req/server.py

```
# server.py - ZeroMQ Echo 服务端 (REP)
import zmq

def echo_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP socket
    socket.bind("tcp://*:5555")  # 绑定到所有IP的5555端口

    print("Echo 服务端启动，等待客户端连接...")

    while True:
        # 等待接收消息
        message = socket.recv()  # 阻塞等待
        print(f"收到消息: {message.decode()}")

        # 回显相同的消息
        socket.send(message)

if __name__ == "__main__":
    try:
        echo_server()
    except KeyboardInterrupt:
        print("\n服务端关闭。")
```

server 将绑定到 5555 端口，监听客户端的连接请求。

在接收到客户端连接后，服务端会以阻塞方式接收其发送的数据，即当前线程将暂停执行，直至数据到达。如同以下 client 代码所示，客户端主动连接到本机的 5555 端口，并在连接成功之后发送字符串消息到 server 端。

**启动命令：**python vllm\_learn/code/course1/req/client.py

```
import zmq

def echo_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # REQ socket
    socket.connect("tcp://localhost:5555")  # 连接到服务端

    # 发送消息
    msg = "Hello, ZeroMQ!"
    print(f"发送消息: {msg}")

    socket.send(msg.encode())  # 发送消息
    reply = socket.recv()  # 接收回复
    print(f"收到回显: {reply.decode()}")

if __name__ == "__main__":
    echo_client()
```

server 端的输出：

```
(base) test_fss@node4:~/code/vllm_learn$ python /home/test_fss/code/vllm_learn/code/course1/req/client.py
发送消息: Hello, ZeroMQ!
收到回显: Hello, ZeroMQ!
```

在本例中，`zmq.REP` 与 `zmq.REQ` 是成对使用的典型请求-应答模式：客户端发送请求（REQ），服务端接收请求并返回响应（REP）。这种模式要求通信双方严格遵循**一问一答**的交互流程——客户端必须先发送请求并等待回复，不能连续发送两次请求，否则会引发异常。如下例所示，我们就在没有接收服务端请求的时候又发送了一条新消息，导致了整个客户端进行的错误崩溃。

```
def echo_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # REQ socket
    socket.connect("tcp://localhost:5555")  # 连接到服务端

    # 发送消息
    msg = "Hello, ZeroMQ!"
    print(f"发送消息: {msg}")

    socket.send(msg.encode())  # 发送消息
    socket.send(msg.encode())  # 发送消息
    reply = socket.recv()  # 接收回复
    print(f"收到回显: {reply.decode()}")
```

```
(base) test_fss@node4:~/code/vllm_learn/code/course1$ python client.py
发送消息: Hello, ZeroMQ!
Traceback (most recent call last):
  File "/home/test_fss/code/vllm_learn/code/course1/client.py", line 21, in <module>
    echo_client()
  File "/home/test_fss/code/vllm_learn/code/course1/client.py", line 15, in echo_client
    socket.send(msg.encode())  # 发送消息
    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/test_fss/anaconda3/lib/python3.12/site-packages/zmq/sugar/socket.py", line 696, in send
    return super().send(data, flags=flags, copy=copy, track=track)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "zmq/backend/cython/socket.pyx", line 742, in zmq.backend.cython.socket.Socket.send
  File "zmq/backend/cython/socket.pyx", line 789, in zmq.backend.cython.socket.Socket.send
  File "zmq/backend/cython/socket.pyx", line 255, in zmq.backend.cython.socket._send_copy
  File "zmq/backend/cython/socket.pyx", line 250, in zmq.backend.cython.socket._send_copy
  File "zmq/backend/cython/checkrc.pxd", line 28, in zmq.backend.cython.checkrc._check_rc
zmq.error.ZMQError: Operation cannot be accomplished in current state
```

同样，在服务端 `socket.recv()` 是一个阻塞调用，它会使当前线程暂停执行，直到有来自客户端的消息到达。这意味着，在单线程环境下，REP 套接字只能顺序处理一个请求-响应周期，必须完整完成一次 `recv → send` 操作后，才能开始处理下一个请求。

尽管多个客户端可以同时连接到该服务端，ZeroMQ 会在传输层缓存它们的请求，但服务端仍以串行方式逐个处理这些请求。另外，服务端无法主动向客户端发起连接，因此这种模式不符合我们在 vLLM 中的通信需求。

Engine 和 AsyncLLM 的通信需要是双工的：Engine 既要向 AsyncLLM 发送推理输出，也要从 AsyncLLM 中获取用户的请求；而且不能是一问一答的同步模式，多个请求和响应的出现顺序并不严格要求前后对应。

### DEALER/ROUTER通信模式

`REQ/REP` 是 ZeroMQ 提供的一种同步请求-响应通信模式，其核心特征是**严格的交替会话机制**：每个 `REQ` 客户端发送一个请求后，必须等待对应的 `REP` 响应，之后才能发起下一次请求；服务端必须处理完一个请求并返回响应后，才能接收下一个请求。这种锁步行为本质上限制了并发能力，无法支持异步通信或多路复用。在大规模分布式系统（如AI推理服务）中，客户端数量众多、请求频繁且具有高度并发性。若采用 `REQ/REP` 模式，将导致以下问题：

1. **吞吐量受限**：单个客户端无法连续发送多个请求，必须等待响应，形成串行瓶颈；

2. **服务端阻塞**：服务端被绑定在同步处理流程中，难以并行处理多个客户端请求；

3. **缺乏路由信息**：`REP` 套接字自动封装了路由路径，但不暴露原始客户端身份，不利于会话追踪与状态管理；

4. **无法双向通信**：服务端不能主动向客户端推送消息。

为解决上述问题，ZeroMQ 提供了更灵活的`DEALER/ROUTER`模式：

* `ROUTER` 套接字能够接收来自多个客户端的消息，并自动在消息前附加**唯一标识符（Identity）**，从而识别消息来源；

* `DEALER` 套接字支持异步双向通信，允许客户端连续发送请求，无需等待响应；

#### 客户端

`DEALER` 与 `ROUTER` 模式的结合，能够构建一个**完全异步、高并发、可扩展**的通信架构，特别适用于像 **AsyncLLM** 这类需要客户端批量提交请求、服务端并行处理并按需响应的场景。为了更好地展示该模式的实际应用，我们实现了一个示例系统：服务端在 `localhost:6666` 端口上通过 `ROUTER` 套接字绑定并监听来自多个客户端的连接请求（完整代码见 `vllm_learn/code/course1/deal/server.py`）。在客户端侧，我们首先为每个客户端设置一个唯一的身份标识（Identity）：

```
socket.setsockopt(zmq.IDENTITY, b"Client-1001")
```

这一标识由 ZeroMQ 内部机制自动附加到所有消息中，使得服务端能够准确识别消息来源，并在后续将响应精确路由回对应的客户端。随后，客户端通过 `DEALER` 套接字向服务端发送请求。每条请求消息包含三个关键字段：

1. `msg_id`：请求的唯一标识符（UUID 或递增 ID），用于追踪请求生命周期。在后续课程中我们将看到，在 AsyncLLM 架构中，当客户端向 `EngineCore` 提交生成任务时，也会携带类似的请求编号，以支持异步回调和状态管理；

2. `query`：实际的输入内容，例如用户提问或提示词（prompt）；

3. `timestamp`：时间戳。

构造完请求后，客户端调用`socket.send_json()`将消息发送至服务端。由于`DEALER`的异步特性，客户端无需等待响应即可继续发送下一条请求，从而实现**高吞吐、非阻塞的消息提交**。

服务端的`ROUTER`套接字在接收到消息时，会自动在消息前附加发送方的标识，形成多段式（multipart）消息结构：

![](data:image/png;base64...)

```
def client(client_id):
    context = zmq.Context.instance()
    socket = context.socket(zmq.DEALER)

    # 为每个客户端设置唯一身份标识（ZeroMQ 会自动使用此身份进行路由）
    identity = f"Client-{client_id}".encode("utf-8")
    socket.setsockopt(zmq.IDENTITY, identity)

    socket.connect("tcp://localhost:6666")
    print(f"[Client {client_id}] 启动，连接到引擎...")

    # 发送多个异步请求
    for i in range(5):
        request = {
            "msg_id": str(uuid.uuid4()),
            "query": f"请求 {i} 来自 {client_id}",
            "timestamp": time.time(),
        }
        socket.send_json(request)
        print(f"[Client {client_id}] 发送请求: {request['msg_id']}")

        # 不等待响应，继续发送下一个请求（异步）
        time.sleep(0.1)

    # 接收响应（异步接收，顺序不一定与发送一致）
    for _ in range(5):
        try:
            response = socket.recv_json(flags=zmq.NOBLOCK)  # 非阻塞接收
            print(f"[Client {client_id}] 收到响应: {response}")
        except zmq.Again:
            time.sleep(0.1)
            continue

    socket.close()
```

#### 服务端

![](data:image/png;base64...)

以下的 `engine` 程序即为服务端实现。它在`localhost:6666`端口上创建并绑定一个监听套接字，使用 ZeroMQ 的`ROUTER`模式接收来自多个客户端的连接与请求。当套接字成功绑定后，服务端进入主事件循环，持续监听并接收客户端消息。由于 `ROUTER` 套接字的特性，所有收到的消息均为**多段式（multipart）消息**，其结构通常包含两个主要部分：

1. **客户端标识（Identity）**：由客户端在初始化时设置，用于唯一标识其身份；

2. **实际消息内容（Message payload）**：即客户端发送的 JSON 数据或其他格式的有效载荷。

服务端从接收到的 multipart 消息中提取出核心数据部分（即 `message`），然后将其交由模拟推理函数 `process_inference` 进行处理——该函数模拟了实际的模型推理任务。当输入的请求数据处理完成后，服务端再构造响应消息体，并通过 `frontend.send_multipart()` 将结果回传给原始客户端。

```
def engine():
    context = zmq.Context.instance()
    frontend = context.socket(zmq.ROUTER)  # 接收客户端请求
    frontend.bind("tcp://*:6666")

    print("推理引擎启动，监听端口 6666...")

    def process_inference(data):
        """模拟异步推理任务"""
        time.sleep(0.5)  # 模拟计算延迟
        return {
            "status": "success",
            "result": f"推理完成: {data['query']}",
            "processed_at": time.time(),
        }

    try:
        while True:
            # ROUTER 接收格式: [identity, delimiter, message]
            # 注意：DEALER 发送的消息会被 ROUTER 自动加上客户端 identity
            multipart = frontend.recv_multipart()
            if not multipart:
                continue

            identity = multipart[0]
            message = multipart[-1]  # 中间可能有空帧（如果用了 delimiter），但我们忽略
            request = zmq.utils.jsonapi.loads(message)

            print(f"引擎收到来自 {identity.decode()} 的请求: {request['msg_id']}")

            # 异步处理推理（这里简化为同步模拟）
            result = process_inference(request)

            # 构造响应：先发 identity，再发响应内容（ROUTER 自动路由）
            response = {
                "msg_id": request["msg_id"],
                "reply": result,
                "from_engine": "Engine-0",
            }
            frontend.send_multipart([identity, zmq.utils.jsonapi.dumps(response)])

    except KeyboardInterrupt:
        print("\n引擎关闭.")
    finally:
        frontend.close()
```

#### C/S通信

我们将客户端与服务端分别运行在**独立的进程**中，以模拟真实分布式环境下的通信场景。在客户端进程中，我们创建了三个独立线程，每个线程代表一个并发客户端，分别调用 `client(client_id)` 函数向服务端发送请求并区分彼此。这三个线程并行运行，模拟多个用户**同时发起请求**的高并发场景。与此同时，在服务端进程中，我们启动一个独立的服务线程，其目标函数为 `engine()`，负责监听 `6666` 端口，接收来自各客户端的消息，并进行处理与响应。

```
if __name__ == "__main__":
    # 启动引擎线程
    engine_thread = threading.Thread(target=engine, daemon=True)
    engine_thread.start()

    time.sleep(1)  # 等待引擎启动

    # 启动多个客户端
    client_threads = []
    for i in range(3):
        t = threading.Thread(target=client, args=(i,), daemon=True)
        t.start()
        client_threads.append(t)

    try:
        # 等待所有客户端完成
        for t in client_threads:
            t.join()
        time.sleep(2)
    except KeyboardInterrupt:
        print("\n主程序退出.")
```

以下为执行的结果：

```
(base) test_fss@node4:~/code/vllm_learn/code/course1/deal$ python server.py 1
推理引擎启动，监听端口 6666...
[Client 0] 启动，连接到引擎...
[Client 1] 启动，连接到引擎...
[Client 0] 发送请求: ef882aee-a43c-4f65-a9ad-d6335ec5f77c
引擎收到来自 Client-0 的请求: ef882aee-a43c-4f65-a9ad-d6335ec5f77c
[Client 1] 发送请求: c4d4ef36-7374-4a89-b7db-afd5ccef0283
[Client 2] 启动，连接到引擎...
[Client 2] 发送请求: ae2af078-5712-46f8-928d-4dc8f46647d2
[Client 0] 发送请求: dc173735-b76b-47e1-a35c-da938bef951f
[Client 1] 发送请求: e6726b67-06e9-4add-b373-2df179657355
[Client 2] 发送请求: 3256abd1-39b7-4271-92e8-4f7d0414cfca
[Client 0] 发送请求: df973b08-f74c-4218-892c-8e1ccf6913d0
[Client 1] 发送请求: fa31d892-e838-4d33-bf3c-e7f218d16530
[Client 2] 发送请求: 7d56bb75-0e66-4fbf-84f4-dfcb70768650
[Client 0] 发送请求: ecad7999-3582-4962-a410-0630ad166000
[Client 1] 发送请求: 6de99f2b-3040-4001-8144-77abec8ffc0e
[Client 2] 发送请求: 52e62e55-bd8f-4384-9a82-dbec094a614e
[Client 0] 发送请求: bf0281fd-5c6b-4a04-8270-aa3206ce6961
[Client 1] 发送请求: d63b0b5f-2b15-4377-b9a5-2e77e16e4a08
[Client 2] 发送请求: 04502be0-53e3-4e7a-92b7-f5a7647bea48
引擎收到来自 Client-0 的请求: dc173735-b76b-47e1-a35c-da938bef951f
[Client 0] 收到响应: {'msg_id': 'ef882aee-a43c-4f65-a9ad-d6335ec5f77c', 'reply': {'status': 'success', 'result': '推理完成: 请求 0 来自 0', 'processed_at': 1757399169.5695212}, 'from_engine': 'Engine-0'}
引擎收到来自 Client-1 的请求: c4d4ef36-7374-4a89-b7db-afd5ccef0283
[Client 0] 收到响应: {'msg_id': 'dc173735-b76b-47e1-a35c-da938bef951f', 'reply': {'status': 'success', 'result': '推理完成: 请求 1 来自 0', 'processed_at': 1757399170.069937}, 'from_engine': 'Engine-0'}
引擎收到来自 Client-2 的请求: ae2af078-5712-46f8-928d-4dc8f46647d2
[Client 1] 收到响应: {'msg_id': 'c4d4ef36-7374-4a89-b7db-afd5ccef0283', 'reply': {'status': 'success', 'result': '推理完成: 请求 0 来自 1', 'processed_at': 1757399170.5703325}, 'from_engine': 'Engine-0'}
引擎收到来自 Client-0 的请求: df973b08-f74c-4218-892c-8e1ccf6913d0
[Client 2] 收到响应: {'msg_id': 'ae2af078-5712-46f8-928d-4dc8f46647d2', 'reply': {'status': 'success', 'result': '推理完成: 请求 0 来自 2', 'processed_at': 1757399171.0706155}, 'from_engine': 'Engine-0'}
引擎收到来自 Client-1 的请求: e6726b67-06e9-4add-b373-2df179657355
```

#### 使用 `zmq.Poller` 等待多个套接字事件

> vllm当中也是**重点使用的**

![](data:image/png;base64...)

现在有一个基于 DEALER/ROUTER 的基本通信架构，但当前的 `engine()` 函数虽然能够处理来自不同客户端的请求，但其单线程阻塞式接收（`recv_multipart()` 会一直等待）仍存在性能瓶颈。

**我们通过引入** `zmq.Poller` **来实现事件驱动的多套接字监听机制。它是 ZeroMQ 提供的核心 I/O 多路复用机制，**`poll()` **会等待事件或超时返回；等待期间应用线程不处理后续代码。**它允许应用程序在一个线程中同时监控多个套接字（socket）的状态变化，例如是否有消息到达（可读）或是否可以发送数据（可写），而无需为每个连接创建独立的线程。

简单理解，就是将套接字是否就绪的监控从用户态转移到了内核态；当内核发现其监控的某个套接字上有事件发生时，再将控制权转移回用户态的代码。

```
poller = zmq.Poller()
poller.register(frontend, zmq.POLLIN)  # 监听是否有消息可读
...
...
socks = dict(poller.poll(timeout=1000))
# 检查 frontend 是否有来自客户端的消息
if frontend in socks and socks[frontend] == zmq.POLLIN:
    multipart = frontend.recv_multipart()
    identity = multipart[0]
    message = multipart[-1]
    request = zmq.utils.jsonapi.loads(message)
```

以上代码`code/course1/deal/server_poller.py`，poller.poll返回的时候一般是存在以下的两种情况：

1. 在指定的超时时间（1000ms）内，没有任何被监听的套接字就绪，此时返回空结果，程序可继续轮询或执行其他任务；

2. 有一个或多个套接字变为就绪状态（例如可读），表明有来自客户端的消息到达，此时即可执行 `recv_multipart()` 读取。

与传统的阻塞式 `recv()` 不同，`zmq.Poller` 通过封装操作系统底层的高性能事件通知机制（如 Linux 的 epoll、macOS 的 kqueue 或跨平台的 poll），在指定超时时间内监听多个套接字的就绪状态。当任意一个套接字准备好进行 I/O 操作时，`zmq.Poller` 会立即返回该套接字及其事件类型，使程序能够及时处理消息，避免不必要的等待。

### PULL/PUSH通信模式

![](data:image/png;base64...)

PUSH 与 PULL 是 ZeroMQ 所提供的单向数据流通信模式，构成一个无回复、单向的消息管道。

消息从 PUSH 端发送，由 PULL 端接收。该模式是实现流式数据传输的理想底层通信机制之一。在流式推理场景中，生产端 Engine 需要连续生成输出，模型每推理出一个 token 就可以通过 PUSH 端发送一次，消费端 AsyncLLM 则持续监听并接收新的数据分块。

由于生产端无需等待或获取消费端的响应，且消息的传递可靠性与到达顺序由 ZeroMQ 保障，因此该模式非常适合在模型推理过程中，将逐个生成的 token 实时推送给消费端。**详见code/course1/push/****server.py**

```
def generate_tokens():
    # 模拟逐个生成 token
    for token in ["Hello", " world", ",", " how", " are", " you", "?"]:
        yield token
        time.sleep(0.1)  # 模拟生成延迟

def main():
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:7777")

    print("PUSH: Sending tokens...")

    for token in generate_tokens():
        sender.send_string(token)

    # 发送结束标记
    sender.send_string("END")
    print("PUSH: All tokens sent.")

    sender.close()
    context.term()
```

我们使用上述代码模拟了 PUSH/PULL 的通信过程：生产者端将获取到的 token 逐个推送，在整段推理结果输出完成后发送结束标志。这一流程与 vLLM 中的流式输出机制基本一致，我们将在下文中进一步分析。

## Engine接收数据

在前文中我们学习DEALER/ROUTER这种通信模式，它的出现能够构建一个**完全异步、高并发、可扩展**的通信架构，特别适用于像 **AsyncLLM** 这类需要客户端批量提交请求、服务端并行处理并按需响应的场景。

所以我们现在要做的事情就是在Engine的代码中找到接收客户端单次/批量提交推理请求的部分。最简单的方法就是在V1版本的Engine中搜索DEALER这个关键字就可以，以下代码详见vllm/v1/engine/core.py。

![](data:image/png;base64...)

```
def process_input_sockets(self, input_addresses: list[str],
                          coord_input_address: Optional[str],
                          identity: bytes):
    """Input socket IO thread."""
    # Msgpack serialization decoding.
    add_request_decoder = MsgpackDecoder(EngineCoreRequest)
    generic_decoder = MsgpackDecoder()
    with ExitStack() as stack, zmq.Context() as ctx:
        # 1. 创建zmq套接字和客户端连接
        input_sockets = [
            stack.enter_context(
                make_zmq_socket(ctx,
                                input_address,
                                zmq.DEALER,
                                identity=identity,
                                bind=False))
            for input_address in input_addresses
        ]

        # Register sockets with poller.
        poller = zmq.Poller()
        ready_response = EngineCoreReadyResponse(
            max_model_len=self.vllm_config.model_config.max_model_len,
            num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
            dp_stats_address=self.frontend_stats_publish_address,
            dtype=str(self.vllm_config.model_config.dtype).removeprefix("torch."),
        )
        ready_payload = msgspec.msgpack.encode(ready_response)
        for input_socket in input_sockets:
            # Send initial message to each input socket - this is requ
            # before the front-end ROUTER socket can send input messag
            # back to us.
            input_socket.send(ready_payload)
            # 2. 将套接字注册到poller中，让内核关注套接字的读写事件
            poller.register(input_socket, zmq.POLLIN)
        if coord_socket is not None:
            poller.register(coord_socket, zmq.POLLIN)
        while True:
            for input_socket, _ in poller.poll():
                # (RequestType, RequestData)
                # 3. 接收来自客户端的请求数据
                type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                request_type = EngineCoreRequestType(
                    bytes(type_frame.buffer))
                # Deserialize the request data.
                decoder = add_request_decoder if (
                    request_type
                    == EngineCoreRequestType.ADD) else generic_decoder
                # 4. 解压缩请求
                request = decoder.decode(data_frames)
                # Push to input queue for core busy loop.
                # 5. 放入到engine的待处理队列中
                self.input_queue.put_nowait((request_type, request))
```

`process_input_sockets` 在 `EngineCoreProc` 初始化时作为后台输入线程启动。它会为 `input_addresses` 创建多个 `DEALER` 套接字，用于接收前端转发的输入请求，并通过 `input_sockets` 持续轮询这些请求；在后续事件循环中，哪个 socket 上有数据，就从哪个 socket 读取消息。

这里的角色划分与很多人直觉中的“服务端 bind、客户端 connect”并不完全一致。在 vLLM 中，Engine 侧使用 `DEALER`，前端侧使用 `ROUTER`。这是因为 `DEALER`/`ROUTER` 是一组异步、可路由的通信模式，哪一侧承担客户端或服务端的语义，更多取决于系统中的逻辑职责与消息路由关系，而不只是物理连接方向。

在课程里可以先做一个简化理解：vLLM 中，Engine 与前端通过 `DEALER`/`ROUTER` 建立异步通信。启动时，Engine 创建 `DEALER` socket 并连接前端 `ROUTER`，随后发送一条包含启动后配置的 ready 消息完成初始化，使前端识别该 Engine 的 identity 并同步必要配置。

之后，前端即可将请求路由到对应的 Engine；Engine 侧由 `process_input_sockets` 接收消息，并在反序列化后放入内部输入队列，交由主循环处理。

我们现在在这里打个断点 vllm/v1/engine/core.py，随后用 code/course1/server.sh 启动服务。

![](data:image/jpeg;base64...)

### 创建连接

![](data:image/png;base64...)

```
input_sockets = [
  stack.enter_context(
    # 1. 创建zmq套接字和客户端连接
    make_zmq_socket(ctx,
                    input_address,
                    zmq.DEALER,
                    identity=identity,
                    bind=False))
  for input_address in input_addresses
]
```

就如同上文所说，这里为每个输入地址创建一个带 identity 的 DEALER 连接，并统一保存到 `input_sockets` 中，供后续通过 poller 持续监听和接收请求。

### 配置非阻塞监听

正如前文所述，这里通过引入 `zmq.Poller`，实现了一个非阻塞、事件驱动的多套接字监听机制。`Poller` 是 ZeroMQ 提供的核心 I/O 多路复用组件，可用于高效监控多个 socket 的状态变化，例如是否有消息可读、是否具备发送条件等。借助这一机制，应用程序可以在单线程内同时管理多个 socket，而无需为每个连接单独创建线程。

随后这一步发送 ready 消息非常关键。它会先向前端的 `ROUTER` 发送一条 `EngineCoreReadyResponse`，使 `ROUTER` 感知并记录当前 `DEALER` 的 identity，同时把 `max_model_len`、`num_gpu_blocks` 等启动后配置同步给前端。

可以将其理解为：Engine 主动向前端报到，告知“这个 identity 对应的 Engine 已经上线”。完成这一步之后，前端 `AsyncLLM` 就可以基于该 identity，将后续请求准确路由回对应的 Engine。

```
for input_socket in input_sockets:
    # Send initial message to each input socket - this is requ
    # before the front-end ROUTER socket can send input messag
    # back to us.
    input_socket.send(ready_payload)
    poller.register(input_socket, zmq.POLLIN)
```

### 接收用户请求

随后，如果 `AsyncLLM` 可以将消息推送回来，当 `poller`（即这里的 `EngineCore`）返回时，说明系统检测到了某个套接字有可读事件。此时，应打开对应的输入套接字，即 `input_socket`，并开始接收数据。

接收到的数据被存储在 `data_frames` 中；由于传输前对请求数据进行了编码压缩以降低带宽占用，因此需要调用 `decoder.decode` 对其进行解码。解码完成后，将完整的请求放入输入队列 `input_queue` 中。

也就是说来自前端的请求，请求会以 (request\_type, request) 的形式通过 self.input\_queue.put\_nowait(...) 放入输入队列`self.input_queue`。

![](data:image/png;base64...)

```
while True:
  for input_socket, _ in poller.poll():
    # (RequestType, RequestData)
     # 3. 接收来自客户端的请求数据
    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
    request_type = EngineCoreRequestType(bytes(type_frame.buffer))
    # Deserialize the request data.
    decoder = ...
    request = decoder.decode(data_frames)
    # Push to input queue for core busy loop.
    self.input_queue.put_nowait((request_type, request))
```

### 调试

> 我们仍采用第一节的方法将 Python 进程 attach，然后点击 VS Code 中的Debug->Run with Debug按钮启动执行。当成功 attach 后，在终端使用 curl 向 vLLM 服务发起请求，断点便会停在相应的位置。**但是调试的时候要attach worker所在的进程，最简单的方法就是查看nvidia-smi中占用显存的进程，而不是vllm.pid中输出的pid。**

或者直接 Attach 到 EngineCoreProc 进程进行调试

![](data:image/jpeg;base64...)

随后在终端上发送一个请求，vLLM调试就会落在断点上，注意下方标红的地方要改成服务的ip和端口：

```
curl -s --noproxy '*' http://127.0.0.1:13311/v1/chat/completions
 -H "Content-Type: application/json"   -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "用20字介绍vLLM"}],
    "max_tokens": 30,
    "temperature": 0.6
  }' | jq -r '.choices[0].message.content'
```

```
+----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         1857996      C   ...est_fss/anaconda3/bin/python3      23830MiB |
+-----------------------------------------------------------------------------------------+
```

在这里我们要抓取的进程就是1857996

```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Attach by PID",
            "type": "debugpy",
            "request": "attach",
            // ① 让 VS Code 弹出进程列表，你手动选 PID
            "processId": "${command:pickProcess}",
            // ② 也可以写死 PID，例如 "processId": 12345
            "justMyCode": false, // 想跳进第三方库就关
            "subProcess": true // 如果目标会 fork 子进程也一起调试
        }
    ]
}
```

我们主要关注两个线程的断点：一个是在 EngineCoreProc 后台输入线程中通过 DEALER/ROUTER 链路接收前端请求，解码后将其放入 Engine 的输入队列；另一个线程则从 Engine 的输入队列中持续读取新到达的请求，并将请求提交至调度器（Scheduler）。调度器的具体调度过程将在后续课程中介绍。

**Engine中的生产者和消费者**

1. `process_input_sockets` 是生产者
   它负责不断从外部 socket 接收请求，然后把请求放入 `input_queue`

2. Engine 主循环中的 `_process_input_queue` 是消费者
   它负责从 `input_queue` 中取出请求，再交给后续调度与执行逻辑处理

**线程1：**以同步的方式从Engine的输入队列`input_queue`中获取新的请求，并在随后用`_handle_client_request`进一步放入该请求并做进一步的调度。

```
def _process_input_queue(self):
    """Exits when an engine step needs to be performed."""
    waited = False
    while not self.has_work() and self.is_running():
        if self.input_queue.empty():
            if logger.isEnabledFor(DEBUG):
                logger.debug("EngineCore waiting for work.")
                waited = True
        block = self.process_input_queue_block
        try:
            req = self.input_queue.get(block=block)
            self._handle_client_request(*req)
        except queue.Empty:
            break
        if not block:
            break
```

**线程2：**线程2中运行的函数就是刚才所说的`process_input_sockets`，以异步非阻塞的方式从客户端AsyncLLM中获取新的请求。线程1和线程2两者的工作协作都围绕着`input_queue`展开。

![](data:image/png;base64...)

`process_input_sockets` 与 `_process_input_queue` 分别处于 Engine 请求处理链路的不同阶段。

1. 前者运行在后台输入线程中，负责通过 ZMQ socket 接收前端发送的请求消息，并在完成反序列化后将其放入内部输入队列 `input_queue`；

2. 后者则运行在 Engine 主循环中，负责从 `input_queue` 中取出这些请求，并进一步交由调度器和执行器处理。

因此，`process_input_sockets` 主要承担外部通信与入队的职责，而 `_process_input_queue` 主要承担内部取队列与分发处理的职责。

## Engine发送数据

上一节已经提到，Engine 会通过 ZMQ 从前端接收输入请求。当请求经过调度器完成调度，并由执行器 Executor 完成模型执行后，Engine 还需要将对应的推理结果返回给用户。在 vLLM 中，这条返回链路采用的是 PUSH/PULL 模式：Engine 作为结果的产生者，负责将输出主动推送回前端。

因此，如果我们要定位这一部分实现，可以直接在 `vllm_source/v1/engine/core.py` 中全局搜索 `PUSH`，从而找到负责结果回传的输出模块。

虽然前面的输入通道已经使用了 DEALER，但这里的 PUSH 属于另一条独立的输出通道。DEALER/ROUTER 主要用于前端按 identity 将请求路由到指定 Engine，而 PUSH 则用于 Engine 将处理结果直接推送回前端，因此两者面向的是不同方向、不同职责的通信流程。

经过搜索，非常轻易地定位到了输出回推的函数：

![](data:image/png;base64...)

同样可以把断点停在推回输出的函数上：

![](data:image/jpeg;base64...)

上一节已经提到，Engine 会通过 ZMQ 从前端接收输入请求。当请求经过调度器完成调度，并由执行器 Executor 完成模型执行后，Engine 还需要将对应的推理结果返回给用户。

在 vLLM 中，这条返回链路采用的是 PUSH/PULL 模式：Engine 作为结果的产生者，负责将输出主动推送回前端。

因此，如果我们要定位这一部分实现，可以直接在 `vllm_source/v1/engine/core.py` 中全局搜索 `PUSH`，从而找到负责结果回传的输出模块。

```
def process_output_sockets(self, output_paths: list[str],
                           coord_output_path: Optional[str],
                           engine_index: int):
  """Output socket IO thread."""

  # Msgpack serialization encoding.
  encoder = MsgpackEncoder()
  # Send buffers to reuse.
  reuse_buffers: list[bytearray] = []
  # Keep references to outputs and buffers until zmq is finished
  # with them (outputs may contain tensors/np arrays whose
  # backing buffers were extracted for zero-copy send).
  pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

  # We must set linger to ensure the ENGINE_CORE_DEAD
  # message is sent prior to closing the socket.
  with ExitStack() as stack, zmq.Context() as ctx:
   # 1. 创建ZMQ PUSH模式的套接字
    sockets = [
      stack.enter_context(
        make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000))
      for output_path in output_paths
    ]

    coord_socket = stack.enter_context(
      make_zmq_socket(
        ctx, coord_output_path, zmq.PUSH, bind=False,
        linger=4000)) if coord_output_path is not None else None
    max_reuse_bufs = len(sockets) + 1

    while True: # 从主循环开始看
      # 2. 等待从输出队列中获取结果
      output = self.output_queue.get()
      assert not isinstance(output, bytes)
      # 3. 确定区分发送的client是哪一个
      client_index, outputs = output
      outputs.engine_index = engine_index

      while pending and pending[-1][0].done:
        reuse_buffers.append(pending.pop()[2])

      buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
      # 4. 对结果进行序列化
      buffers = encoder.encode_into(outputs, buffer)
      # 5. 发送到对应的客户端
      tracker = sockets[client_index].send_multipart(buffers,
                                                       copy=False,
                                                       track=True)
```

### 创建连接

在前文所述中，ZeroMQ 提供的`PUSH`与`PULL`模式构成了一种单向数据流通信机制，形成一条无响应、单向的消息管道：消息由 `PUSH` 端发出，由 `PULL` 端接收。该模式天然适用于流式数据传输场景，是构建高效异步通信链路的理想底层机制之一。

```
with ExitStack() as stack, zmq.Context() as ctx:
    sockets = [
      stack.enter_context(
        make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000))
      for output_path in output_paths
    ]
```

在流式推理场景中，推理引擎（Engine）作为生产者，需在模型每生成一个或多个 token 时，将结果通过 ZMQ 的 `PUSH` 套接字发送出去；前端的 AsyncMPClient 持有对应的 `PULL` 端，持续接收并处理逐步到达的输出分块。因此，如上代码所示，推理引擎一侧需创建一条以 `PUSH` 套接字为出口的单向通信管道，确保生成的 token 流可以高效、有序地推送至下游消费端，实现低延迟、高吞吐的流式输出能力。

可以看到，这里的 `output_path` 是由外部传入的，即在 AsyncLLM 初始化 Engine 时传递，目的是在 AsyncLLM 与 Engine 之间建立一条 ZMQ 连接，以便在后续生成输出时，能够成功将数据推送到 AsyncLLM 所在的客户端进程中。

![](data:image/png;base64...)

### 等待结果产生

当用于推送推理结果的套接字创建并打开后，系统进入循环，等待引擎输出就绪。具体而言，当 `Engine` 完成一个或多个请求的推理后，会将结果写入其内部的 `output_queue`。`process_output_sockets` 函数中的循环正是用于等待该队列中有数据可读。

一旦有输出就绪，`output = self.output_queue.get()` 调用立即返回，随后通过 ZMQ 的 `PUSH` 套接字将结果（通常为新生成的 token）发送PUSH出去。该过程分为以下步骤：

1. 从 `Engine` 的输出队列中获取结果 `output`；

2. 根据 `output` 中携带的元信息，确定该结果对应的是哪个客户端，获取其 `client_index`；

3. 根据 `client_index` 找到对应的 ZMQ 套接字，将新生成的结果通过 `PUSH` 模式发送到对应的AsyncLLM客户端。

```
while True:
  # 1. 从输出队列中获取结果
  output = self.output_queue.get()
  assert not isinstance(output, bytes)
  # 2. 区分发送的client是哪一个
  client_index, outputs = output
  outputs.engine_index = engine_index

  buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
  # 3. 对结果进行压缩
  buffers = encoder.encode_into(outputs, buffer)
  # 4. 发送到对应的客户端
  tracker = sockets[client_index].send_multipart(buffers,
                                                   copy=False,
                                                   track=True)
```

![](data:image/png;base64...)

## AsyncLLM发送请求

有接收请求，就必然有客户端发送请求。我们知道 Engine 端用的是 DEALER，那么客户端就是 ROUTER，因此只要在代码里搜 ROUTER 就能定位。很快找到了AsyncLLM在客户端侧用 ROUTER 套接字转发请求，同时为每个请求维护一个队列，用来接收 Engine 回传的新 token，后续AsyncLLM会通过一个EngineCoreClient中的input\_socket来向Engine转发来自客户端的请求。

![](data:image/png;base64...)

```
self.engine_core = EngineCoreClient.make_async_mp_client(
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=self.log_stats,
    client_addresses=client_addresses,
    client_index=client_index,
)

# self.input_socket是self.engine_core的中一个变量，也就是EngineCoreClient中的一个变量
self.input_socket = self.resources.input_socket = make_zmq_socket(
                self.ctx, input_address, zmq.ROUTER, bind=True)
```

**在非 DP 场景下，AsyncLLM 中创建的 EngineCoreClient 实例是 AsyncMPClient**；如果启用 data parallel，则会使用 DPAsyncMPClient 或 DPLBAsyncMPClient 这类 DP 客户端。AsyncMPClient 中有一个 input\_socket 用于向 Engine 转发请求，也就是像上面代码写的那样。

也就是说，客户端侧的核心并不是 HTTP Server 本身，而是 AsyncLLM 这一层：

* AsyncLLM 初始化时，会创建一个 EngineCoreClient。

* 这个 EngineCoreClient 内部持有一个 input\_socket，用于把前端请求转发给后端 EngineCore。

* 其中 input\_socket 使用的正是 ROUTER socket。

> 因此这次要 attach 的是客户端进程，所以需要在attach process的时候选择如下的进程。我这里因为开了两个vllm服务，所以大家看一下对应的运行端口（port）是哪个。

![](data:image/jpeg;base64...)

```
# 为了突出重点，以下的代码有省略
async def add_request(
  self,
  request_id: str,
  prompt: PromptType,
  params: Union[SamplingParams, PoolingParams],
  arrival_time: Optional[float] = None,
  lora_request: Optional[LoRARequest] = None,
  tokenization_kwargs: Optional[dict[str, Any]] = None,
  trace_headers: Optional[Mapping[str, str]] = None,
  priority: int = 0,
  data_parallel_rank: Optional[int] = None,
) -> RequestOutputCollector:
  """Add new request to the AsyncLLM."""

  if self.errored:
    raise EngineDeadError()

    is_pooling = isinstance(params, PoolingParams)

    # Create a new output collector for the request.
    # 为每个请求创建一条队列
    queue = RequestOutputCollector(output_kind=params.output_kind)
    # 将请求request和新创建的队列queue传递到_add_request
    await self._add_request(request, prompt_str, None, 0, queue)

    return queue
```

在客户端 AsyncLLM 的 add\_request 函数中：

1. 为每个请求创建一个用于接收输出结果的队列 queue。这个队列并不会发送给 EngineCoreProc，而是保留在前端，用来缓存该请求后续返回的增量结果。

2. 调用 \_add\_request() 处理该请求，在这个函数里：
   1. **一方面会把 request 注册到前端的 OutputProcessor 中，并与对应的 queue 绑定；**
   2. 另一方面，会调用 engine\_core.add\_request\_async(request)，将 request 通过 EngineCoreClient 的 input\_socket 发往 EngineCoreProc 所在进程执行推理。

![](data:image/png;base64...)![](data:image/png;base64...)![](data:image/jpeg;base64...)

### 创建结果的接收队列

结果的接收队列就是上一节中的 queue。它与请求一一对应，用于在前端接收来自 Engine 的推理输出。每个请求都会创建一个独立的队列，这些队列并不是单独以一个"队列表"的形式保存，而是作为请求状态的一部分，按 request\_id 组织并统一维护在 AsyncLLM 的 OutputProcessor 中。调用链如下图所示：

1. 当 Engine 产生输出后，前端后台的 output\_handler 会持续从 EngineCore 拉取 EngineCoreOutputs，并将其交给 OutputProcessor 处理。

2. OutputProcessor 会根据 request\_id 找到对应的请求状态，以及其中绑定的 queue，再将新生成的 token 或阶段性输出放入该队列。

3. 上层 generate() 再从这个 queue 中不断取出结果，对外形成流式输出。

![](data:image/png;base64...)

```
class AsyncLLM():
  async def _add_request(self, request: EngineCoreRequest,
                         prompt: Optional[str],
                         parent_req: Optional[ParentRequest], index: int,
                         queue: RequestOutputCollector):

    self.output_processor.add_request(request, prompt, parent_req, index, queue)

class OutputProcessor:
    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: Optional[str],
        parent_req: Optional[ParentRequest] = None,
        request_index: int = 0,
        queue: Optional[RequestOutputCollector] = None,
    ) -> None:
        # 1. 每个请求独有自己独立的req id
        request_id = request.request_id
        if request_id in self.request_states:
            raise ValueError(f"Request id {request_id} already running.")
        tokenizer = None if not self.tokenizer else \
            self.tokenizer.get_lora_tokenizer(request.lora_request)
        req_state = RequestState.from_new_request(tokenizer=tokenizer,
                                                  request=request,
                                                  prompt=prompt,
                                                  parent_req=parent_req,
                                                  request_index=request_index,
                                                  queue=queue,
                                                  log_stats=self.log_stats)
        # 2. 对每个请求对应的队列进行缓存
        self.request_states[request_id] = req_state
```

到这里位置，我们当前请求对应的队列就已经初始化完毕了。

### 转发请求

当 AsyncLLM 接收到请求并创建对应的队列之后，如前文所述，就通过 self.engine\_core 的 add\_request\_async 对请求通过 ZMQ 进行转发，转发到 Engine 对应的套接字上。在非 DP 场景下，engine\_core 是 **AsyncMPClient 的实例**；启用 data parallel 时，它会是 DPAsyncMPClient 或 DPLBAsyncMPClient。

```
async def _add_request(self, request: EngineCoreRequest,
                       prompt: Optional[str],
                       parent_req: Optional[ParentRequest], index: int,
                       queue: RequestOutputCollector):
    # Add the request to OutputProcessor (this process).
    self.output_processor.add_request(request, prompt, parent_req, index,
                                      queue)
    # Add the EngineCoreRequest to EngineCore (separate process).
    await self.engine_core.add_request_async(request)
```

![](data:image/png;base64...)

1. 流程进入 AsyncMPClient.add\_request\_async() 后，会先将请求对象序列化编码，即通过 self.encoder.encode 转换成可跨进程传输的消息帧。

2. 随后客户端会将 engine identity、request type 以及编码后的请求数据组装成一条 multipart 消息，并通过 self.input\_socket 发送到 Engine 端。

3. 至此，请求从前端转发到 Engine 的流程就闭环了，也与前面介绍的 ROUTER -> DEALER 通信链路对应起来。

```
async def add_request_async(self, request: EngineCoreRequest) -> None:
    # 给请求打上客户端标记
    request.client_index = self.client_index
    # 将 ADD 类型的请求发往 Engine
    await self._send_input(EngineCoreRequestType.ADD, request)

def _send_input():
    # 将 request 序列化，并与请求类型一起封装成消息
    message = (request_type.value, *self.encoder.encode(request))
    return self._send_input_message(message, engine, request)

def _send_input_message():
    # 通过 ZMQ 套接字把消息发送到 Engine
    future = self.input_socket.send_multipart(msg, copy=False, track=True)
    return future
```

## AsyncLLM接收结果

在 AsyncLLM 中，每个请求都会创建一个专属队列，并在请求注册阶段绑定到 OutputProcessor 维护的请求状态上。后续 Engine 返回结果时，需要先经过 OutputProcessor 的拆分与后处理，再分发到对应请求的专属队列中。整体流程如下：

1. **拉取结果：**AsyncMPClient 使用 PULL socket 持续从 Engine 侧接收推理结果，并将接收到的 EngineCoreOutputs 放入本地的 `outputs_queue` 中。

2. **结果处理：**AsyncLLM 后台的 output\_handler 持续从 `outputs_queue` 中取出结果，并交给 OutputProcessor 进行拆分和后处理。

3. **放入请求队列：**OutputProcessor 根据结果中携带的 request\_id 找到对应请求状态中的专属 queue，并将新生成的 token 或阶段性输出放入该队列。

4. **对外返回：**上层 generate() 再从这个专属 queue 中不断取出结果，从而形成流式输出。

### 拉取结果

1. 这里的 output\_socket 是前端 AsyncMPClient 持有的 PULL 套接字，它与引擎端的 PUSH 套接字配对，用于接收 Engine 返回的推理结果。

2. 在协程 process\_outputs\_socket 中，这个 output\_socket 会持续监听来自引擎端的推送消息。一旦收到数据，就立即调用 recv\_multipart() 完成接收，并通过 decoder.decode(frames) 将消息帧反序列化为 EngineCoreOutputs。

3. 解码完成后，这些结果不会直接分发到各个请求队列中，而是会先放入 AsyncMPClient 的 `outputs_queue`。后续再由 AsyncLLM 后台的 output\_handler 从 `outputs_queue` 中取出结果，并交给 OutputProcessor 做进一步处理和分发。

![](data:image/png;base64...)

```
self.resources.output_socket = make_zmq_socket(
    self.ctx, output_address, zmq.PULL
)

async def process_outputs_socket():
    try:
        while True:
            # 1. 持续从 Engine 端拉取结果。
            # Engine 会把推理输出按 multipart message 的形式推送过来。
            frames = await output_socket.recv_multipart(copy=False)

            # 检查 Engine 是否仍然存活，必要时抛出异常。
            resources.validate_alive(frames)

            # 2. 将收到的消息帧反序列化为 EngineCoreOutputs 对象。
            outputs: EngineCoreOutputs = decoder.decode(frames)

            # 3. 放入 AsyncMPClient 的 outputs_queue。
            # 注意：这里还没有按 request_id 分发到各请求自己的 queue，
            # 只是先放入客户端的公共输出队列，等待后续 AsyncLLM 消费。
            if outputs.outputs or outputs.scheduler_stats:
                outputs_queue.put_nowait(outputs)

    except Exception as e:
        # 如果后台拉取结果时出现异常，也放入 outputs_queue，
        # 这样上层在消费 outputs_queue 时就能感知并处理错误。
        outputs_queue.put_nowait(e)
```

### 结果放入队列

上一节中，把 Engine 的输出拉到 AsyncMPClient 公共的 `outputs_queue` 里，只完成了结果回传的第一步。接下来，还需要把这些公共结果进一步分发到各个请求自己的专属队列中。

1. AsyncLLM 后台的 output\_handler 会持续调用 engine\_core.get\_output\_async()，从 AsyncMPClient 的公共 `outputs_queue` 中取出新到达的 EngineCoreOutputs。

2. 取出的结果随后交给 output\_processor.process\_outputs() 处理。这个过程不仅会按 request\_id 拆分不同请求的输出，还会完成 detokenize、停止条件判断等后处理逻辑。

目前为止的数据流是：

1. **Engine**

-> PUSH 输出结果

2. **AsyncMPClient**

-> AsyncMPClient.output\_socket(PULL) 接收结果

-> 反序列化为 EngineCoreOutputs

-> 放入 AsyncMPClient 的公共 `outputs_queue`

-> AsyncLLM.output\_handler 调用 engine\_core.get\_output\_async() 取出结果

-> output\_processor.process\_outputs() 按 request\_id 拆分，并完成 detokenize、停止条件判断等后处理

![](data:image/png;base64...)

```
def _run_output_handler(self):
    """后台循环：持续从 EngineCore 拉取输出，并分发给各请求。"""

    engine_core = self.engine_core
    output_processor = self.output_processor

    async def output_handler():
        try:
            while True:
                # 1. 从 engine_core 拉取最新一批结果。
                # 从 AsyncMPClient的公共 outputs_queue 中取出一个 EngineCoreOutputs。
                outputs = await engine_core.get_output_async()

                # 当前这一批结果里包含多少条 request 输出
                num_outputs = len(outputs.outputs)

                # 如果开启了统计功能，则为这一轮结果处理准备统计对象
                iteration_stats = IterationStats() if (
                    log_stats and num_outputs) else None

                # 这里为了说明主流程，省略了真实代码中的分块逻辑；
                # 可以把它理解为"逐批处理本轮返回的所有输出"
                slices = (outputs.outputs, )

                for i, outputs_slice in enumerate(slices):
                    # 2. 对这一批 EngineCoreOutputs 做前端处理。
                    # process_outputs() 会完成几件事：
                    # - 按 request_id 区分不同请求的结果
                    # - 将新 token 做 detokenize
                    # - 判断是否触发 stop 条件
                    # - 将处理后的结果放入对应请求绑定的专属 queue 中
                    processed_outputs = output_processor.process_outputs(
                        outputs_slice, outputs.timestamp, iteration_stats)
```

在 `process_outputs` 里，入口参数 `engine_core_outputs` 就是刚从 Engine 拉到的最新一批输出，每条 `EngineCoreOutput` 都带 `request_id` 字段。函数只保留关键四步：

1. 接收已经由 AsyncLLM.output\_handler 从 engine\_core.get\_output\_async() 取出的 `engine_core_outputs`；

2. 用 `request_id` 到 OutputProcessor 的注册表里找到该请求的唯一队列；

3. 把原始输出进行解码并且封装成 `RequestOutput`；

4. 将解码后的结果塞进这条私有队列，等待客户端消费。

![](data:image/png;base64...)

```
def process_outputs(
    self,
    engine_core_outputs: list[EngineCoreOutput],
    engine_core_timestamp: Optional[float] = None,
    iteration_stats: Optional[IterationStats] = None,
) -> OutputProcessorOutput:

    # 我们在这里只保留关键流程
    for engine_core_output in engine_core_outputs:
        # 找到唯一的队列
        req_id = engine_core_output.request_id
        req_state = self.request_states.get(req_id)
        # 解码输出的token_id
        stop_string = req_state.detokenizer.update(
                    new_token_ids, finish_reason == FinishReason.STOP)

        # 如果解码中包括了stop token就需要加上停止符号，用于控制客户端后续逇输出
        if stop_string:
            finish_reason = FinishReason.STOP
            stop_reason = stop_string

        if request_output := req_state.make_request_output(
                new_token_ids, pooling_output, finish_reason,
                kv_transfer_params, num_cached_tokens):
            if req_state.queue is not None:
                # 找到唯一的队列后放入当前解码后的输出
                req_state.queue.put(request_output)
```

## 流式返回

在 `AsyncLLM.generate` 中，流式返回的核心机制可概括为“提交-等待-即时产出”三步：

1. 提交请求：`add_request` 为当前 `request_id` 创建专属的异步队列 `q`；

1. 等待就绪：循环执行 `q.get_nowait()` 或 `await q.get()` —— 当 `output_handler` 中的 `process_outputs` 将解码后的输出推入队列时，立即获取；

1. 即时产出：拿到一条 `RequestOutput` 就 `yield` 一条，不累积完整响应，实现真正的实时流式输出。

![](data:image/png;base64...)

```
async def generate(
  self,
  prompt: PromptType,
  sampling_params: SamplingParams,
  request_id: str,
  lora_request: Optional[LoRARequest] = None,
  trace_headers: Optional[Mapping[str, str]] = None,
  priority: int = 0,
  data_parallel_rank: Optional[int] = None,
) -> AsyncGenerator[RequestOutput, None]:
  # 增加请求
  q = await self.add_request(
    request_id,
    prompt,
    sampling_params,
    lora_request=lora_request,
    trace_headers=trace_headers,
    priority=priority,
    data_parallel_rank=data_parallel_rank,
  )

  finished = False
  while not finished:
    # 等待队列就绪
    out = q.get_nowait() or await q.get()
    finished = out.finished
    # 每个请求对应的队列有数据立即返回
    yield out
```

## 整体流程

1. 用户调用 `AsyncLLM.generate(...)`，为这个 `request_id` 注册一个专属输出队列/stream。

1. `AsyncLLM` 通过 `EngineCoreClient.add_request_async(request)` 将请求序列化后发送给后台 `EngineCore`。

1. EngineCore 执行推理，生成 token，并把结果打包成带 `request_id` 字段的 `EngineCoreOutput` 批量返回。

1. 后台协程 `_run_output_handler` 持续拉取输出，调用 `await engine_core.get_output_async()` 获取最新一批结果。

1. `output_processor.process_outputs(...)` 处理并分发结果推送到该请求自己的队列。

1. `generate()` 前台异步消费结果，该请求的专属队列中不断取出 `RequestOutput`，每拿到一条就 `yield` 给用户，直到 `finished=True` 结束。

⚠️ 暂不支持的文档区域，【文档小组件】

* Frontend：面向用户的调用接口层，负责接收生成请求并以流式方式返回结果，例如 `generate()`。

* Engine：底层推理执行层，负责请求调度、模型计算以及 token 生成。

* Background：异步输出处理层，负责从 Engine 拉取推理结果、完成结果处理与分发，并将结果投递回前端请求对应的输出队列。