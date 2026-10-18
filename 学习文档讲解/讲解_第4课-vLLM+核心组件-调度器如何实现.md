# 讲解 | 第4课 vLLM 核心组件：调度器如何实现

> 本文是对 [第4课-vLLM+核心组件-调度器如何实现.md](../学习文档/第4课-vLLM+核心组件-调度器如何实现.md) 的精髓提炼，
> 所有知识点均已在本地 vLLM 仓库中核对到对应源码（见第 8 节代码锚点，可直接点击跳转）。
> 课程里的代码片段是**简化版**，本文同时标注了当前仓库代码与课程片段的差异。

---

## 🎯 精髓一句话总结

**调度器（Scheduler）每个 step 回答三个问题：这轮让谁算？（选人）、各算多少 token？（定量）、KV cache 放得下吗？（分块，放不下就抢占）。**

- 它**只做决策、不碰模型**：产出一张任务清单 `SchedulerOutput`，交给第3课讲的 Executor/Worker 去执行；
- 核心心法只有一句（源码作者 woosuk 的注释）：**调度器眼里没有"prefill 阶段 / decode 阶段"，只有"每个请求的已算 token 数（`num_computed_tokens`）努力追上它的目标 token 数（`num_tokens_with_spec`）"**。chunked prefill、prefix cache、投机解码全是这套模型的特例；
- 排队顺序两条铁律：**RUNNING 优先于 WAITING（保流式输出不卡）**；同队列内 **FCFS 按到达顺序 / PRIORITY 数字小者优先**；
- 两类硬约束：**token budget**（单轮 token 总量上限）+ **KV cache block**（物理显存档位），后者不足时触发**抢占**。

> 💡 比喻：调度器像高铁检票口。
> `waiting` = 候车大厅排队的乘客（行李大小 = prompt 长短）；
> `running` = 已经上车就座的乘客；
> `token_budget` = 本班列车的剩余座位；
> KV cache block = 行李架格子；
> **抢占** = 行李架满了，把最晚进站的乘客请下车回大厅等下一班（FCFS），或请优先级最低的 VIP 下车（PRIORITY）。

---

## 1️⃣ 调度器的"户型图"：三个队列 + 一个注册表

调度器内部就这几个核心容器（[scheduler.py L213-216](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L213-L216)）：

```python
self.waiting = create_request_queue(self.policy)          # 正常排队的新请求/被抢占请求
self.skipped_waiting = create_request_queue(self.policy)  # "条件不满足"的候补区
self.running: list[Request] = []                          # 已上车的请求（普通 list）
self.requests: dict[str, Request] = {}                    # req_id -> Request 的总登记表
```

| 队列 | 数据结构 | 装什么 |
|---|---|---|
| **waiting** | FCFS 时是 deque；PRIORITY 时是小顶堆 | 还没开始算的新请求、被抢占后等恢复的请求 |
| **skipped_waiting** | 同上 | 暂时"上不了车"的请求：等 grammar、等远端 KV 传输、等多模态编码、LoRA 名额满等 |
| **running** | 普通 list | 已被接纳、正在持续推进的请求 |
| **requests** | dict | 所有在管请求的户籍册，按 `request_id` 随时查 |

请求状态机（[request.py L364-372](file:///F:/Projects/vllm/vllm/v1/request.py#L364-L372)）：

```text
WAITING ──首次调度成功──▶ RUNNING ──生成完毕──▶ FINISHED
   ▲                       │
   │                       │ KV block 不足，被抢占
   └───── PREEMPTED ◀──────┘
          （放回 waiting 队首，清零进度，等下轮重新调度）
```

> 💡 注意：`skipped_waiting` 不是独立状态，请求的 `status` 仍然是 WAITING 系的各种 blocked 子状态，只是暂时挪到这个队列里，下一轮再"尝试解冻"。

### 【补充解释：为什么需要两个 waiting 队列？】

排队买奶茶时，如果你点的饮品原料没了，店员不会让你堵在柜台前——会让你站到旁边等原料，原料来了再回来。`skipped_waiting` 就是这个"旁边等位区"：

- 比如**结构化输出（grammar）**的约束还没编译好、**多模态图片**还在编码、**远端 KV cache**还在传输——这些请求强行排进正常队列只会反复检查、空耗 CPU；
- 调度器每轮会对 skipped 里的请求"试探解冻"（`_try_promote_blocked_waiting_request`），解冻成功就重新参与调度。

对入门来说，只要记住"**waiting 是正常排队，skipped_waiting 是条件不齐的候补区**"即可。

---

## 2️⃣ 请求进门：从网络到 waiting 队列

新请求**不会**一到就触发推理，要走完这条"挂号"链路：

```text
前端 AsyncLLM
  │  ZMQ ROUTER-DEALER（异步）
  ▼
EngineCoreProc 输入线程：写入 input_queue
  │
  ▼ 主循环 run_busy_loop() 每轮：
_process_input_queue()      先把到达的请求全取完
  └─ _handle_client_request(ADD)
       └─ EngineCore.add_request()    做校验（id 类型/pooling/KV transfer 配置）
            └─ Scheduler.add_request()
                 └─ _enqueue_waiting_request()
                      ├─ blocked 状态 → skipped_waiting
                      └─ 普通请求    → waiting 队尾
```

对应代码：

| 步骤 | 位置 |
|---|---|
| EngineCore 主循环（先收请求、再走一步） | [core.py L1411-1422](file:///F:/Projects/vllm/vllm/v1/engine/core.py#L1411-L1422) `run_busy_loop` |
| 消费 input_queue（阻塞等活 → 非阻塞排空） | [core.py L1437-1475](file:///F:/Projects/vllm/vllm/v1/engine/core.py#L1437-L1475) `_process_input_queue` |
| 校验后转交调度器 | [core.py L452-496](file:///F:/Projects/vllm/vllm/v1/engine/core.py#L452-L496) `add_request`（核心调用 `self.scheduler.add_request(request)` 在 [L492](file:///F:/Projects/vllm/vllm/v1/engine/core.py#L492)） |
| 入 waiting / skipped_waiting | [scheduler.py L2824-2850](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L2824-L2850) `add_request`、[L2408-2412](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L2408-L2412) `_enqueue_waiting_request` |

> 💡 一个请求里有多个 token。用户输入 "Hi!" 会被 tokenizer 切成整数 ID（如 `[1234, 5678]`），Request 里存的是 **token ID 序列**，不是字符串。vLLM **不要求一轮算完整个请求**——这正是 chunked prefill 的基础。

---

## 3️⃣ 两条硬约束：token budget 和 KV block

每个 scheduler step，调度器手里只有两种"资源额度"：

### 约束 1：token budget（本轮算力额度）

```python
token_budget = self.max_num_scheduled_tokens          # scheduler.py L631
```

- 默认等于配置 `max_num_batched_tokens`（[scheduler.py L130-134](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L130-L134)）；开投机解码时可能调小；
- 含义：**这一个 step 所有请求加起来最多调度多少个 token**。每收一个请求的 token，就 `token_budget -= num_new_tokens`，扣完为止；
- 类比：这班列车只剩 3 个座位，先来的人先坐，坐满即止，剩下的等下一班。

**课程同款例子（budget = 3）：**

```text
R1（running，decode 只需 1 token）  → 收！剩余预算 2
R2（waiting，prompt "Hi!" = 2 token）→ 1+2=3 正好！剩余预算 0
R3（waiting，prompt = 5 token）      → 预算已尽，本轮不调度，留在 waiting
```

### 约束 2：KV cache block（物理显存格子）

- KV cache **不以单个 token 为单位分配，而是固定大小的 block**（大小由 `block_size` 决定）；
- 调度器为每个请求调 `kv_cache_manager.allocate_slots(...)` 申请 block（[kv_cache_manager.py L400](file:///F:/Projects/vllm/vllm/v1/core/kv_cache_manager.py#L400)）：
  - 返回 `new_blocks` → 有位置，本轮可以上车；
  - 返回 `None` → 没位置，**running 请求触发抢占，waiting 请求直接放弃本轮**。

### 还有第三条"人数"约束

```python
if len(self.running) == self.max_num_running_reqs:   # = max_num_seqs
    break
```

（[scheduler.py L931-933](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L931-L933)）并发请求数达到 `max_num_seqs` 后，不再接纳新请求。

> 💡 为什么 running 优先？decode 中的请求每轮只差 1 个 token 就能给用户吐出一个字，一旦被饿住，流式输出肉眼可见地卡顿；而新请求晚一轮上车用户几乎无感。**保连续性 > 接新客**，这是整个调度顺序设计的出发点。

---

## 4️⃣ schedule() 主流程：一张图看完

`scheduler.schedule()`（[scheduler.py L611](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L611)）结构非常清晰：

```text
                          schedule() 一个 step
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 ① 初始化临时账本          ② 调度 RUNNING           ③ 调度 WAITING
 token_budget = 上限      （L658-876）             （L922-1376）
 4 个请求列表             逐个请求：               前提：本轮没发生抢占
 req_to_new_blocks        算 num_new_tokens       且 token_budget 有余量
 num_scheduled_tokens     → allocate_slots        逐个请求：
                          → 不足则抢占循环         查 prefix cache（本地+外部）
                                                  → allocate_slots
                                                  → 出队进 running
                                │
                                ▼
                       ④ assert 校验（L1378-1390）
                          总 token ≤ budget、人数 ≤ max_num_seqs
                                │
                                ▼
                       ⑤ 构造 SchedulerOutput（L1402-1434）
                          NewRequestData / CachedRequestData
                                │
                                ▼
                       ⑥ _update_after_schedule()
                          乐观推进 num_computed_tokens
```

四个临时账本（L624-631）：

| 变量 | 含义 |
|---|---|
| `scheduled_new_reqs` | 本轮**首次**上车的请求（waiting 中 WAITING 状态的） |
| `scheduled_resumed_reqs` | 本轮**恢复**的请求（被抢占后 PREEMPTED → 重新上车） |
| `scheduled_running_reqs` | 本来就在车上、本轮继续推进的请求 |
| `preempted_reqs` | 本轮被踢下车的请求（**它非空时，本轮绝不再接纳 waiting**） |
| `num_scheduled_tokens` | `req_id -> 本轮给它算几个 token` |
| `req_to_new_blocks` | `req_id -> 本轮分到的 KV block` |

### 【补充解释：为什么抢占发生后就不接新请求了？】

代码 L922 一个 `if not preempted_reqs` 把关。理由很朴素：**抢占本身就是"显存已经不够了"的红色警报**——把老乘客都赶下了车，这时候再放新乘客上车只会雪上加霜、继续抢占。所以一旦本轮踢过人，剩余 token budget 宁可空着，也不接 waiting。

---

## 5️⃣ 调度 RUNNING：算 token → 分 block → 不够就抢

对 running 队列里的每个请求（[scheduler.py L660-876](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L660-L876)）：

**第 1 步：本轮还差多少 token？**

```python
num_new_tokens = (
    request.num_tokens_with_spec        # 目标位置：prompt + 已生成 + 投机 token
    + request.num_output_placeholders   # 异步调度占位（入门可先当 0）
    - request.num_computed_tokens       # 已算到的位置
)
```

（[scheduler.py L693-697](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L693-L697)。课程片段里没有 `num_output_placeholders`，那是新版异步调度加的，主线逻辑不变。）

直觉理解：`目标位置 − 当前位置 = 还差的距离`。普通 decode 请求这个差值就是 1；长 prompt 第一次调度时差值可能是几百上千。

**第 2 步：三道截断**

```python
# ① 长 prefill 单请求限速（可选）
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold
# ② 不能超过本轮剩余预算（新版同时受 input_budget 约束）
num_new_tokens = min(num_new_tokens, token_budget, ...)
# ③ 不能超过模型最大长度
num_new_tokens = min(num_new_tokens, max_model_len - num_computed_tokens - ...)
```

（[L698-711](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L698-L711)）截断后若为 0，`continue` 跳过这个请求、去看 running 里的下一个（注意是 continue 不是 break，允许"插队式"跳过）。

**第 3 步：申请 KV block，进入"申请—抢占"重试循环**（[L777-866](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L777-L866)）：

```text
while True:
    new_blocks = allocate_slots(request, num_new_tokens, ...)
    if new_blocks is not None:  break          # ✓ 分到了，跳出
    # ✗ 没分到 → 选一个受害者抢占，释放它的 block
    # 然后 while 回到开头，再试一次 allocate_slots
    if preempted_req == request:  break         # 把自己都抢了还不够 → 放弃本轮
if new_blocks is None:  break                   # 整个 running 扫描到此为止
```

**第 4 步：成功，记账**（[L868-876](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L868-L876)）：

```python
scheduled_running_reqs.append(request)
req_to_new_blocks[request_id] = new_blocks
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
```

### 【补充解释：allocate_slots 分配的 block 数为什么不是简单的除法？】

课程里特意提醒：block 数**不是** `num_computed_tokens / block_size` 向上取整那么简单。`allocate_slots` 还要综合考虑：

- 请求**已经占用**的 block（追加分配即可，不重复给）；
- 本轮新增 token 是否正好跨 block 边界；
- lookahead token（投机解码预留槽位）；
- **prefix cache 命中**：前缀对应的 block 可直接引用，不用新分；
- 外部 KV（connector）正在异步搬运的 block。

入门阶段只需把它理解成一个黑盒："**告诉它请求是谁、本轮要算几个 token，它要么还我一组 block，要么还我 None**"。

---

## 6️⃣ 抢占机制：谁下车、下车后去哪

### 选受害者（victim selection）

| 策略 | 规则 | 代码 |
|---|---|---|
| **FCFS**（默认） | 踢 running **队尾**（最晚进站的） | [scheduler.py L852](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L852) `self.running.pop()` |
| **PRIORITY** | 踢优先级最低者：`priority` 数值**最大**的；并列踢到达更晚的 | [policy.py L84-86](file:///F:/Projects/vllm/vllm/v1/core/sched/policy.py#L84-L86) `max(running, key=lambda r: (r.priority, r.arrival_time))` |

> 💡 为什么 FCFS 踢队尾？越早进站的请求已投入的算力越多，打断它损失最大；队尾请求刚上车，被抢的"重算代价"最小。这和排队上车本身是同一套先来先服务哲学。

### 抢占动作 `_preempt_request`（[scheduler.py L1553-1601](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1553-L1601)）

```python
self._free_request_blocks(request)      # ① 释放它占用的全部 KV block
self.encoder_cache_manager.free(request)# ② 释放多模态 encoder cache
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0         # ③ 计算进度清零！恢复时 prompt 要重算
request.spec_token_ids = []             # ④ 清空投机 token
request.num_preemptions += 1            # ⑤ 计数+1（统计/防饿死用）
self.waiting.prepend_request(request)   # ⑥ 放回 waiting 队首
```

两个要点：

1. **进度清零 = 重算**：vLLM 的抢占采用 **recompute（重算）** 策略，不保留 KV，恢复时从头再跑一遍 prompt（prefix cache 可能帮忙省掉一部分）；
2. **回队首而不是队尾**：FCFS 队列是 deque，`prepend_request` = `appendleft`（[request_queue.py L110-112](file:///F:/Projects/vllm/vllm/v1/core/sched/request_queue.py#L110-L112)），被抢的请求下轮优先恢复，不会被永久饿死。PRIORITY 堆里则按优先级重新排序，没有"队首"概念。

### 边界情况：抢到自己

```python
if preempted_req == request:
    break        # L860-862
```

能抢的都抢了（极端情况下把自己也从 running 里踢出去了）仍然分不到 block → 这个请求本轮调度失败，整个 running 扫描停止。

### 【补充解释：FCFS 和 PRIORITY 队列的数据结构】

见 [request_queue.py](file:///F:/Projects/vllm/vllm/v1/core/sched/request_queue.py)：

- **FCFSRequestQueue**：直接继承 `deque`——`append` 入队尾、`popleft` 出队头、`appendleft` 恢复插队，O(1)；
- **PriorityRequestQueue**：手写小顶堆，入队时锁定排序键 `(priority, arrival_time, request_id, id(request))`（[L177-184](file:///F:/Projects/vllm/vllm/v1/core/sched/request_queue.py#L177-L184)）。priority 数值小的排堆顶先被服务；`arrival_time` 做同优先级的二级排序。还维护了 `request -> 堆下标` 的索引，支持 O(log n) 任意删除。

工厂函数 `create_request_queue(policy)`（[L328-338](file:///F:/Projects/vllm/vllm/v1/core/sched/request_queue.py#L328-L338)）按策略二选一。

---

## 7️⃣ 调度 WAITING：查缓存 → 分 block → 上车

只有满足三个前提才会进入 waiting 段（[scheduler.py L922-933](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L922-L933)）：

1. `not preempted_reqs`：本轮没发生抢占；
2. `_pause_state == UNPAUSED`：服务没有被暂停；
3. `token_budget > 0` 且 running 人数未达 `max_num_seqs`。

每个候选请求的处理（[L938-1345](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L938-L1345)）：

```text
peek 队首请求（只看不取，分不到 block 就留给下轮）
   │
   ├─ blocked 状态？尝试解冻，失败 → 挪进 skipped_waiting
   ├─ stale output 在途 / LoRA 名额满？→ 挪进 skipped_waiting
   │
   ▼
查"已经算过的 token" num_computed_tokens
   ├─ == 0：查本地 prefix cache（get_computed_blocks，算公共前缀）
   │         + 查外部 KV（connector.get_num_new_matched_tokens）
   └─ > 0：通常是远端 KV 异步接收已完成，直接复用
   │
   ▼
num_new_tokens = request.num_tokens − num_computed_tokens
   再用 token_budget 截断
   （没开 chunked prefill 且一个预算装不下整个 prompt → break，等下轮）
   │
   ▼
allocate_slots(..., new_computed_blocks=命中的前缀block, ...)
   ├─ None → 直接 break！注意：waiting 请求分不到 block 【不会】触发抢占
   └─ 成功 → pop 出 waiting，append 进 running
             WAITING  → scheduled_new_reqs（新客）
             PREEMPTED → scheduled_resumed_reqs（恢复）
             status = RUNNING；num_computed_tokens 记账；预算扣减
```

### 三个和 running 段的关键差异

1. **waiting 段分不到 block 不抢占**（[L1242-1249](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1242-L1249) 直接 `break`）——抢占有可能腾出 block 吗？理论上可能，但课程基线策略选择"新客不能把老客挤下车"，只在 running 请求自己需要时才允许抢占；
2. **先 peek 后 pop**：分 block 成功才真正出队，失败时请求原样留在队首；
3. **上车时把命中的前缀 block"认领"过来**：`allocate_slots` 接收 `new_computed_blocks`，这些 block 不用重新计算也不用重新分配，直接挂进该请求的 block table——这就是 **Prefix Caching** 省油的地方。

### 【补充解释：Prefix Caching 在这里到底发生了什么？】

假设两个请求都以"请翻译以下段落："开头。第二个请求 peek 出来时 `num_computed_tokens == 0`，调度器调 `get_computed_blocks` 在已有 block 里按 hash 找公共前缀，发现这 10 个 token 的 KV 之前算过且还在显存 → `num_new_local_computed_tokens = 10`，于是：

- 本轮只需算 `总 token − 10` 个，省算力、省延迟；
- 那 10 个 token 对应的 block 被**引用计数式地共享**（不是拷贝），refcount +1；
- 若还配了外部 KV 存储（connector，如多机 PD 分离），还可能从远端命中更多 token，异步搬运期间请求进入 `WAITING_FOR_REMOTE_KVS`，本轮 `num_new_tokens = 0`，只搬不算。

---

## 8️⃣ SchedulerOutput：调度器交给 Worker 的"任务清单"

调度的全部决策被打包成 `SchedulerOutput`（[output.py L209-285](file:///F:/Projects/vllm/vllm/v1/core/sched/output.py#L209-L285)），在 EngineCore.step 里交给 Executor（[core.py L608-609](file:///F:/Projects/vllm/vllm/v1/engine/core.py#L608-L609)）——正好接上第3课的 `collective_rpc("execute_model", ...)`。

清单回答三件事：

| 问题 | 字段 | 说明 |
|---|---|---|
| **做什么** | `scheduled_new_reqs` / `scheduled_cached_reqs` | 本轮上车的请求（[NewRequestData L36](file:///F:/Projects/vllm/vllm/v1/core/sched/output.py#L36)：首次出现，**全量**发送 prompt token、采样参数、block_ids）+ 已缓存的请求（[CachedRequestData L132](file:///F:/Projects/vllm/vllm/v1/core/sched/output.py#L132)：**只发增量 diff**，省通信） |
| **做多少** | `num_scheduled_tokens` / `total_num_scheduled_tokens` | `req_id -> token 数` 及总和（总和必须 ≤ budget，有 assert 兜底，[L1379-1380](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1379-L1380)） |
| **用哪些缓存** | block_ids（含在上面两类 data 中） | 传的是 **block 编号，不是 KV 张量本身**！Worker 按编号定位物理显存、更新自己的 block table |

辅助字段：`scheduled_spec_decode_tokens`（投机 token）、`scheduled_encoder_inputs`（多模态）、`preempted_req_ids`（本轮被抢的，通知 Worker 清状态）、`finished_req_ids`（已完成的）、`new_block_ids_to_zero`（新 block 先清零防脏数据）、`num_common_prefix_blocks`（公共前缀，可用于 cascade attention）等。

> 💡 **为什么要分 New / Cached 两种数据？** 新请求的 prompt 可能几千 token，每轮都全量广播太贵（第3课讲过广播队列的 16MB 分界）。所以请求元数据**只在首次调度时发一次**，Worker 缓存住；之后每轮只发"新增了哪几个 block、本轮算几个 token"的增量。这和 HTTP 缓存里的"全量首包 + 增量更新"是一个思路。

### 调度之后的状态更新（两个回调，别混淆）

| 回调 | 时机 | 干什么 |
|---|---|---|
| `_update_after_schedule`（[L1622-1648](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1622-L1648)） | 清单构造完、**GPU 还没算完**时 | **乐观预推进**：`num_computed_tokens += num_scheduled_token`，并标记是否还在 prefill chunk 中 |
| `update_from_output`（[L1976](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1976)） | GPU 前向 + 采样完成后（[core.py L622](file:///F:/Projects/vllm/vllm/v1/engine/core.py#L622)） | 把真正采样出的 token 追加进请求、判断是否结束、投机 token 被拒绝时回滚多算的进度 |

### 【补充解释：GPU 还没算完，凭什么先加 num_computed_tokens？】

这是故意的"乐观更新"，源码注释给了三条理由（L1623-1631）：

1. 本轮发给 Worker 的清单必须记下原始调度量；
2. 提前推进后，**下一轮立刻可以继续调度同一请求的下一个 prefill 分片**（chunked prefill 流水化，不用干等 GPU）；
3. 万一投机 token 被拒绝导致"预支"了，`update_from_output` 拿到真实采样结果后再纠偏。

如果不提前记账，下一轮调度时就不知道这个请求已经算到哪，会重复调度同一段 token，导致 KV cache 写穿位置。

---

## 9️⃣ 一个完整 step 的数值演练（把全文串起来）

```text
初始：max_num_scheduled_tokens(budget) = 8，block_size = 4
  waiting: [R3(prompt=5, VIP=0)] [R4(prompt=6, 普通)]
  running: [R1(decode 中)] [R2(prompt=14，上轮已算 4，chunked prefill 中)]

① RUNNING 段
  R1：num_new_tokens = 目标3 − 已算2 = 1；allocate_slots ✓
      budget 8→7，scheduled_running = [R1]
  R2：目标14 − 已算4 = 10；long_prefill_threshold/预算截断 → 本轮给 7
      allocate_slots(7 token ≈ 2 block) ✓（跨 block 边界按需追加）
      budget 7→0，scheduled_running = [R1, R2]

② WAITING 段：token_budget == 0 → while 条件不成立，R3/R4 本轮不上车

③ SchedulerOutput：
     cached_reqs: R1(tokens=1), R2(tokens=7, new_block_ids=[...])
     total = 8 ≤ 8 ✓

④ _update_after_schedule：
     R1.num_computed_tokens 2→3；R2 4→11（还没算完，仍是 prefill chunk）

─── GPU 执行 + 采样 ───

⑤ update_from_output：R1 吐出 1 个真 token，追加到输出；
     R2 本 chunk 无输出 token（还在 prefill）；检查是否 finish
```

**抢占分支演练**：若 ① 中 R1 的 `allocate_slots` 返回 None：

```text
allocate_slots → None
  → FCFS: victim = running.pop() = R2（队尾）
  → _preempt_request(R2)：free 全部 block、num_computed_tokens=0、
     status=PREEMPTED、prepend 回 waiting 队首、preempted_reqs=[R2]
  → 重试 allocate_slots(R1) ✓ → R1 正常记账
② 因 preempted_reqs 非空，waiting 段整个跳过（R2/R3/R4 都等下轮）
③ 输出中带 preempted_req_ids={R2}，Worker 同步清掉 R2 的缓存状态
```

---

## 🔟 vLLM 源码对应位置一览

> 结论：**课程讲到的每个机制在 vLLM 项目中都有真实对应代码**，变量命名也基本一致。
> 注意：当前本地分支是 `project/kv-aware-scheduling`，在基线之上做了 KV-aware 扩展（见文末说明）；下表行号基于当前工作区文件。

### 📄 调度器主文件 [vllm/v1/core/sched/scheduler.py](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py)（主战场，约 3000 行）

| 知识点 | 函数 / 行号 |
|---|---|
| 三个队列 waiting / skipped_waiting / running | [`__init__` L213-216](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L213-L216) |
| token budget 默认值 = max_num_batched_tokens | [L128-134](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L128-L134) |
| schedule 主入口 + woosuk 的"无阶段论"注释 | [`schedule` L611-622](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L611-L622) |
| RUNNING 调度全段 | [L658-910](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L658-L910) |
| num_new_tokens 计算 + 三道截断 | [L693-711](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L693-L711) |
| allocate_slots + 抢占重试循环 | [L777-866](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L777-L866) |
| FCFS 踢队尾 | [L852](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L852) |
| "本轮有抢占就不接 waiting" 门禁 | [L922](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L922) |
| WAITING 调度全段 + max_num_seqs 门禁 | [L922-1376](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L922-L1376) |
| 本地 + 外部 prefix cache 查询 | [L996-1101](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L996-L1101) |
| waiting 请求 num_new_tokens | [L1116-1157](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1116-L1157) |
| waiting 分 block 失败直接 break（不抢占） | [L1223-1249](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1223-L1249) |
| 出 waiting 进 running、状态 NEW/RESUMED 分流 | [L1324-1345](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1324-L1345) |
| 三条 assert 校验 | [L1378-1390](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1378-L1390) |
| 构造 SchedulerOutput | [L1402-1434](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1402-L1434) |
| 抢占：释放 block / 清零 / 回 waiting 队首 | [`_preempt_request` L1553-1601](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1553-L1601) |
| 调度后乐观推进 num_computed_tokens | [`_update_after_schedule` L1622-1648](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1622-L1648) |
| GPU 返回后对账（采样 token、finish、纠偏） | [`update_from_output` L1976](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1976) |
| blocked 请求分流到 skipped_waiting | [`_enqueue_waiting_request` L2408-2412](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L2408-L2412) |
| FCFS 时 skipped 优先于 waiting 的选取规则 | [`_select_waiting_queue_for_scheduling` L2420-2432](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L2420-L2432) |
| 请求注册入队 | [`add_request` L2824-2850](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L2824-L2850) |

### 📄 [vllm/v1/core/sched/request_queue.py](file:///F:/Projects/vllm/vllm/v1/core/sched/request_queue.py)

| 知识点 | 位置 |
|---|---|
| FCFS / PRIORITY 枚举 | [L16-20](file:///F:/Projects/vllm/vllm/v1/core/sched/request_queue.py#L16-L20) |
| FCFS 队列（deque：append / popleft / appendleft） | [`FCFSRequestQueue` L93-158](file:///F:/Projects/vllm/vllm/v1/core/sched/request_queue.py#L93-L158) |
| PRIORITY 队列（堆 + 排序键 priority→arrival_time） | [`PriorityRequestQueue` L167-325](file:///F:/Projects/vllm/vllm/v1/core/sched/request_queue.py#L167-L325) |
| 策略工厂 | [`create_request_queue` L328-338](file:///F:/Projects/vllm/vllm/v1/core/sched/request_queue.py#L328-L338) |

### 📄 [vllm/v1/core/sched/output.py](file:///F:/Projects/vllm/vllm/v1/core/sched/output.py)

| 知识点 | 位置 |
|---|---|
| 新请求全量数据包 | [`NewRequestData` L36](file:///F:/Projects/vllm/vllm/v1/core/sched/output.py#L36) |
| 老请求增量数据包 | [`CachedRequestData` L132](file:///F:/Projects/vllm/vllm/v1/core/sched/output.py#L132) |
| 任务清单本体 | [`SchedulerOutput` L209-285](file:///F:/Projects/vllm/vllm/v1/core/sched/output.py#L209-L285) |

### 📄 上下游配合文件

| 知识点 | 文件 / 位置 |
|---|---|
| EngineCore 主循环 / 消费 input_queue / ADD 分发 | [vllm/v1/engine/core.py L1411-1560](file:///F:/Projects/vllm/vllm/v1/engine/core.py#L1411-L1560) |
| EngineCore.add_request 校验 | [core.py L452-496](file:///F:/Projects/vllm/vllm/v1/engine/core.py#L452-L496) |
| **schedule → execute_model → update_from_output 三连**（调度器与第3课 Executor 的接缝） | [core.py L597-627](file:///F:/Projects/vllm/vllm/v1/engine/core.py#L597-L627) |
| block 分配黑盒 | [vllm/v1/core/kv_cache_manager.py `allocate_slots` L400](file:///F:/Projects/vllm/vllm/v1/core/kv_cache_manager.py#L400) |
| 请求状态枚举 WAITING/RUNNING/PREEMPTED | [vllm/v1/request.py L364-372](file:///F:/Projects/vllm/vllm/v1/request.py#L364-L372) |
| num_tokens / num_tokens_with_spec | [request.py L287-293](file:///F:/Projects/vllm/vllm/v1/request.py#L287-L293) |
| priority / arrival_time 字段 | [request.py L84-96](file:///F:/Projects/vllm/vllm/v1/request.py#L84-L96) |

### ⚠️ 课程片段 vs 当前仓库代码的差异（避免对不上号）

1. 课程是**简化后的教学片段**（如 running 段只写了 `token_budget`，实际还有 `input_budget`、encoder budget、spec padding、Mamba 对齐等），主干逻辑一致；
2. PRIORITY 的抢占选择已被抽到 [policy.py 的 `DefaultSchedulingDecisionPolicy`](file:///F:/Projects/vllm/vllm/v1/core/sched/policy.py#L69-L97)，基线行为（`max(priority, arrival_time)`、FCFS 取 `running[-1]`）与课程完全一致；
3. 课程提到的调度演示 Demo `code/course3/sche.py` 属于**课程配套仓库**，本地 vLLM 仓库中没有该文件（已确认不存在）；vLLM 官方仓库另有 `tests/v1/entrypoints/` 与 `benchmarks/` 下的调度相关用例可参考。

---

## 1️⃣1️⃣ 一张速记表

```text
请求进门（随时发生）：
  ZMQ → input_queue → EngineCore.add_request（校验）
       → Scheduler.add_request → waiting 队尾（blocked 进 skipped_waiting）

每个 step（schedule()）：
  预算 token_budget = max_num_scheduled_tokens
  │
  ├─ ① RUNNING 优先（保 decode 连续）
  │     每个请求：num_new_tokens = 目标位置 − 已算位置
  │               → 三道截断（长prefill阈值/预算/max_model_len）
  │               → allocate_slots
  │                    ├─ 成功：记账，预算扣减
  │                    └─ None：抢占！FCFS 踢队尾 / PRIORITY 踢优先级最低
  │                             释放 block + 进度清零 + PREEMPTED + 回 waiting 队首
  │                             抢到自己都不够 → 本轮放弃
  │
  ├─ ② WAITING（仅当本轮无抢占、预算有余、人数未满）
  │     peek → 解冻检查 → 本地/外部 prefix cache 命中
  │          → allocate_slots（失败直接 break，不抢占）
  │          → 出队进 running：WAITING→new_reqs / PREEMPTED→resumed_reqs
  │
  ├─ ③ assert：总 token ≤ budget、并发 ≤ max_num_seqs
  ├─ ④ 打包 SchedulerOutput：新客全量、老客增量、只传 block 编号
  └─ ⑤ _update_after_schedule：乐观推进 num_computed_tokens

GPU 算完：
  update_from_output：追加采样 token、finish 判定、投机拒绝则纠偏
```

一句话：**running 先保命、waiting 看余量；block 是硬通货，不足就抢占重算；每轮一张清单，编号代替张量。**

---

## ⚠️ 可跳过的高级内容

课程与代码中以下内容属于扩展场景，入门时先抓住"running 优先 + budget 截断 + block 抢占"主线即可：

- **投机解码（spec decode）**：`num_tokens_with_spec`、`spec_token_ids`、新 decode 请求 padding 到统一投机长度、`scheduled_spec_decode_tokens`；
- **异步调度（async scheduling）**：`num_output_placeholders`、`num_in_flight_tokens`、`next_decode_eligible_step`、PP 节奏控制（主线里把它们都当 0 即可）；
- **外部 KV / KVConnector**：`WAITING_FOR_REMOTE_KVS`、`load_kv_async` 只搬不算、producer partial-tail offload；
- **skipped_waiting 的各种解冻条件**：grammar bitmask、encoder cache、LoRA `max_loras`、streaming input、stale output 在途；
- **多模态/混合模型细节**：encoder compute budget、Mamba block 对齐切分、MTP prefill lookahead；
- **DP prefill 均衡**：`throttle_prefills` / `defer_prefills`；cascade attention 的 `num_common_prefix_blocks`；
- **本地分支 KV-aware 扩展（非上游基线！）**：[policy.py](file:///F:/Projects/vllm/vllm/v1/core/sched/policy.py) 中除 `DefaultSchedulingDecisionPolicy` 外的 `RecomputeAwareSchedulingDecisionPolicy`（抢占时优先选重算代价小的受害者）、`ReclaimableAware...`（优先选能最快补齐 block 缺口的受害者）和 cache_affinity 准入/老化策略，都是本仓库 `project/kv-aware-scheduling` 分支自己的研究改动，读课时先忽略，对照基线行为只看 Default 策略。

---

## 🔍 调试小技巧

- **断点位置**：打在 [scheduler.py L611](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L611) `schedule()` 入口，单步观察 `self.waiting` / `self.running` / `token_budget` / `num_scheduled_tokens` 四个账本逐行变化；抢占分支直接在 [L777](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L777) 的 while 循环和 [`_preempt_request` L1553](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1553) 下断点；
- **触发方式**：launch 模式启动 `vllm.entrypoints.openai.api_server`（attach 赶不上启动期），用 curl 连续发 2-3 个**长 prompt + 小 `max_num_batched_tokens`**（如 256）的请求，几轮内必现 chunked prefill；把 `gpu_memory_utilization` 调低或并发拉高即可复现抢占；
- **观测指标**：开 stats 后日志里的 ` preempted` 计数、KV cache usage；代码侧可看 `request.num_preemptions`（被抢次数）和 `kv_preemption_stats`（[L1569-1575](file:///F:/Projects/vllm/vllm/v1/core/sched/scheduler.py#L1569-L1575)）；
- **验证 PRIORITY**：请求中带不同 `priority` 字段，在 `PriorityRequestQueue` 堆里观察 `(priority, arrival_time)` 排序，以及抢占时 `max(key=...)` 选出的受害者；
- 与第3课联调：在 [core.py L608](file:///F:/Projects/vllm/vllm/v1/engine/core.py#L608) 看 `SchedulerOutput` 如何作为参数进入 `execute_model` → 就是第3课广播队列里被序列化下发的那个对象。

```bash
curl -s --noproxy '*' http://127.0.0.1:13333/v1/completions \
 -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen3-1.7B",
    "prompt": "请用五百字介绍一下人工智能的发展历史",
    "max_tokens": 30,
    "temperature": 0.6
 }'
```
