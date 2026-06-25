# 编排控制层

> 源码位置：`areal/infra/controller/`, `areal/infra/workflow_executor.py` 等 文件数：10 个 |
> 总行数：6392 行

## 1. 模块定位

编排控制层是 AReaL 分布式 RL 训练框架的**中枢调度系统**。它负责协调训练侧（TrainController） 与推理侧（RolloutController /
RemoteInfEngine）之间的异步协作，实现训练与数据采集的流水线 重叠。核心设计目标：

- **训练-推理解耦**：训练与 rollout 采样在独立进程/节点上异步执行
- **版本过期控制**：通过 StalenessManager 保证 on-policy 训练的数据新鲜度
- **批量任务调度**：BatchTaskDispatcher 实现生产者-消费者模式的高吞吐任务派发
- **权重同步协调**：RolloutCallback 通过 HTTP 回调 + NCCL 集合通信实现非阻塞权重更新

## 2. 文件清单与行数

| 文件                               | 行数 | 核心职责                           |
| ---------------------------------- | ---- | ---------------------------------- |
| `controller/__init__.py`           | 11   | 包导出                             |
| `controller/rollout_controller.py` | 1131 | 推理侧编排控制器                   |
| `controller/train_controller.py`   | 868  | 训练侧编排控制器                   |
| `controller/rollout_callback.py`   | 182  | 训练侧到推理侧的 HTTP 回调代理     |
| `workflow_executor.py`             | 1372 | 工作流执行器 + BatchTaskDispatcher |
| `remote_inf_engine.py`             | 1457 | 远程推理引擎（HTTP 客户端）        |
| `staleness_manager.py`             | 182  | 版本过期 / 容量管理                |
| `async_task_runner.py`             | 685  | 通用异步任务执行器                 |
| `dist_rollout.py`                  | 263  | 分布式 rollout 数据重分布          |
| `workflow_context.py`              | 241  | 工作流执行上下文 + HTTP 客户端管理 |

## 3. 架构总览

```
+--------------------------------------------------------------------+
|                        训练循环 (Workflow)                           |
|  prepare_batch() / train_batch() / update_weights()                 |
+-------+------------------+-----------------------------------------+
        |                  |
        v                  v
+---------------+   +------------------+
| TrainController|   | RolloutController|
| (L174-868)     |   | (L72-1131)       |
+-------+-------+   +--------+---------+
        |                     |
        | connect_engine()    | _start_callback_server()
        | RolloutCallback     | Flask HTTP Server
        v                     v
+---------------+   +-------------------+
| RolloutCallback|-->| Callback Endpoints|
| (HTTP Proxy)   |   | /callback/*      |
| (L18-182)      |   +-------------------+
+---------------+          |
                           v
                  +-------------------+       +------------------+
                  |BatchTaskDispatcher|------>| AsyncTaskRunner  |
                  | (L262-732)        |       | (L73-685)        |
                  +--------+----------+       +--------+---------+
                           |                           |
                           v                           v
                  +-------------------+       +------------------+
                  | StalenessManager  |       | uvloop EventLoop |
                  | (L20-182)         |       | (背景线程)        |
                  +-------------------+       +------------------+
                                                       |
                                              +--------v---------+
                                              | WorkflowExecutor |
                                              | (L746-1372)      |
                                              +--------+---------+
                                                       |
                                              +--------v---------+
                                              | RemoteInfEngine  |
                                              | (L327-1457)      |
                                              +--------+---------+
                                                       |
                                                       v
                                              HTTP (aiohttp/uvloop)
                                                       |
                                              +--------v---------+
                                              | SGLang / vLLM     |
                                              | 推理服务器         |
                                              +------------------+
```

## 4. 核心组件详解

### 4.1 RolloutController -- 推理侧编排控制器

**文件**：`controller/rollout_controller.py`（1131 行） **类定义**：L72

RolloutController 管理一组分布式推理 Worker，提供统一的 rollout 采样接口。

**核心状态**：

| 字段                 | 类型                        | 说明                                  |
| -------------------- | --------------------------- | ------------------------------------- |
| `workers`            | `list[Worker]`              | 通过 Scheduler 创建的推理 Worker 列表 |
| `_dispatcher`        | `BatchTaskDispatcher`       | 批量任务分发器（L105-107）            |
| `_staleness_manager` | `StalenessManager`          | 版本过期管理器（L102）                |
| `_callback_app`      | `Flask`                     | HTTP 回调服务器（L109）               |
| `_pending_futures`   | `dict[int, asyncio.Future]` | 任务完成回调 Future（L119）           |
| `proxy_workers`      | `list[Worker]`              | AgentWorkflow 代理 Worker（L123）     |

**初始化流程** (`initialize`，L158-232)：

```
initialize(role, server_args)
    |
    +-- 1. 计算实例规模: TP x PP = instance_size
    |
    +-- 2. 通过 Scheduler.create_workers() 创建 Worker
    |       每个 Worker 对应一个推理引擎实例
    |
    +-- 3. _async_initialize():
    |       - create_engine() on each Worker (动态导入引擎类)
    |       - launch_server() 启动推理服务
    |       - engine.initialize() 连接到推理服务
    |
    +-- 4. 创建 StalenessManager (L207-216)
    |       max_concurrent_rollouts, consumer_batch_size, max_staleness
    |
    +-- 5. 创建 BatchTaskDispatcher (L219-229)
    |       task_factory = _create_submit_callback
    |
    +-- 6. _start_callback_server() 启动 Flask HTTP 回调服务
```

**任务提交与等待模式** (`submit`/`wait`，L866-911)：

```
submit(data, workflow, ...)           wait(count, timeout)
    |                                      |
    v                                      v
_RemoteRolloutTaskInput               dispatcher.wait_results(count)
    |                                      |
    v                                      v
dispatcher.submit_task_input()        从 _pending_results 取出结果
    |                                 按 create_time 排序后随机打乱
    v
_pending_inputs deque
```

**submit-then-wait 回调模式** (`_create_submit_callback`，L778-861)：

这是 RolloutController 最核心的异步协调逻辑。每个 rollout 任务通过以下流程执行：

```
_create_submit_callback(pending_task) -> async _submit_then_wait()
    |
    +-- 1. round-robin 选择 Worker
    |
    +-- 2. 创建 asyncio.Future 并注册到 _pending_futures[task_id]
    |
    +-- 3. scheduler.async_call_engine("submit", ...)
    |       将任务提交到远程 Worker 上的 RemoteInfEngine
    |       携带 callback_addr 指向本地 Flask 回调服务
    |
    +-- 4. await asyncio.wait_for(future, timeout)
    |       阻塞等待 Flask 回调 /callback/rollout_complete
    |       远程 Worker 完成后 POST 回调，触发 _resolve_task_future()
    |
    +-- 5. scheduler.async_call_engine("wait_for_task", ...)
    |       获取实际结果数据
    |
    +-- 6. staleness_manager.on_rollout_accepted/rejected()
```

**权重同步回调服务器** (`_start_callback_server`，L546-654)：

Flask 服务器注册 6 个回调端点，处理来自 TrainController (通过 RolloutCallback) 的请求：

| 端点                            | 方法 | 作用                       |
| ------------------------------- | ---- | -------------------------- |
| `/callback/init_weights_group`  | POST | 初始化 NCCL 权重更新通信组 |
| `/callback/update_weights_xccl` | POST | 通过 NCCL 分布式接收权重   |
| `/callback/update_weights_disk` | POST | 从磁盘加载权重             |
| `/callback/pause_generation`    | POST | 暂停推理生成（权重更新前） |
| `/callback/continue_generation` | POST | 恢复推理生成（权重更新后） |
| `/callback/rollout_complete`    | POST | rollout 任务完成通知       |

**Proxy 子系统**（L347-544）：

为 AgentWorkflow 提供代理 Worker，通过 `start_proxy()` 在推理 Worker 旁创建 ProxyRolloutServer， 并可通过
`start_proxy_gateway()` 启动统一网关供外部访问。

### 4.2 TrainController -- 训练侧编排控制器

**文件**：`controller/train_controller.py`（868 行） **类定义**：L174

TrainController 管理分布式训练 Worker 的生命周期和数据分发。

**核心设计**：

- 每个 Worker 对应一个 GPU rank（`LOCAL_RANK=0`，每进程单 GPU）
- 仅 DP Head Worker 接收数据，非 DP Head 通过 broadcast 获取
- 通过 `_custom_function_call` 统一分发 RPC 请求

**训练-推理桥接** (`connect_engine`，L600-610)：

```python
def connect_engine(self, rollout: RolloutController, meta: WeightUpdateMeta):
    self.rollout = rollout
    engine = RolloutCallback(controller_addr=rollout.callback_addr)
    self._custom_function_call("connect_engine", engine=engine, meta=meta)
```

这是训练侧与推理侧的关键连接点。通过序列化 `RolloutCallback` dataclass 到每个训练 Worker，使训练引擎可以通过 HTTP
回调触发推理侧的权重同步操作。

**数据分发流水线** (`_custom_function_call`，L471-598)：

```
_custom_function_call(method, *args, **kwargs)
    |
    +-- _prepare_dispatch(*args, **kwargs)
    |       |
    |       +-- _is_tensor_like(args)? --Yes--> _partition_inputs()
    |       |       |
    |       |       +-- _dispatch_tensors(item_list, dp_size, group_size)
    |       |       |       使用 balanced_greedy_partition 按 token 数均衡分配
    |       |       |
    |       |       +-- 返回 (splits, group_indices)
    |       |
    |       +-- _is_tensor_like(args)? --No--> _replicate_inputs()
    |               复制到所有 DP 组
    |
    +-- _call_workers(method, dp_split_args, dp_split_kwargs)
    |       仅 DP Head 获得数据切片，其他 Worker 获得空参数
    |       通过 scheduler.async_call_engine 并行调用
    |
    +-- _collect_results(results, group_indices)
            tensor dispatch: _merge_tensors 按原始顺序重排
            scalar dispatch: 返回第一个 DP head 的结果
```

**销毁顺序**（`destroy`，L403-469）：

销毁顺序经过精心设计以避免 NCCL `TCPStore.recvValue failed` 警告：

1. 先调用所有引擎的 `destroy()` 使每个 rank 执行 `dist.destroy_process_group()`
1. 以反序删除 Worker（rank-0 即 TCPStore owner 最后删除）

### 4.3 RolloutCallback -- 非阻塞 HTTP 回调代理

**文件**：`controller/rollout_callback.py`（182 行） **类定义**：L18

RolloutCallback 是训练 Worker 用于触发推理侧操作的代理。它是一个 `@dataclass`， 可以序列化后传递到远程 Worker。

**关键设计约束**（L26-28）：

> IMPORTANT: Methods that return Future must be non-blocking to avoid deadlocks. NCCL
> operations are collective - both train and inference sides must participate
> concurrently.

**方法分类**：

| 方法                              | 阻塞性 | 返回类型       | 原因                          |
| --------------------------------- | ------ | -------------- | ----------------------------- |
| `init_weights_update_group`       | 非阻塞 | `Future[None]` | 训练侧需同时创建 NCCL 组      |
| `update_weights_from_distributed` | 非阻塞 | `Future[None]` | 双侧需并行参与 NCCL broadcast |
| `update_weights_from_disk`        | 非阻塞 | `Future[None]` | 一致性设计                    |
| `pause_generation`                | 阻塞   | `None`         | 必须在权重更新前完成          |
| `continue_generation`             | 阻塞   | `None`         | 必须在返回前完成              |

非阻塞方法通过 `_post_nowait` 提交到后台线程池执行 HTTP 请求。

### 4.4 BatchTaskDispatcher -- 批量任务调度器

**文件**：`workflow_executor.py`（L262-732） **泛型类**：`BatchTaskDispatcher[TInput, TResult]`

BatchTaskDispatcher 是编排控制层的**任务调度心脏**，同时被 RolloutController 和 WorkflowExecutor
使用，但参数化不同的输入/输出类型。

**三线程架构**：

```
主线程                    生产者线程 (_commit_loop)     消费者线程 (_fetch_loop)
  |                           |                             |
  |  submit_task_input()      |                             |
  +-- _pending_inputs --->    |                             |
  |   (deque, 无界)           |                             |
  |                     _get_next_task_for_submission()      |
  |                           |                             |
  |                     检查 staleness 容量                  |
  |                     检查 runner 队列未满                  |
  |                           |                             |
  |                     task_factory(task_input)             |
  |                     runner.submit(task_fn)               |
  |                     staleness_manager.on_submitted()     |
  |                           |                             |
  |                           |    AsyncTaskRunner 执行      |
  |                           |         |                   |
  |                           |         v                   |
  |                           |    uvloop event loop        |
  |                           |    asyncio.create_task()    |
  |                           |         |                   |
  |                           |         v                   |
  |                           |    结果 -> output_queue     |
  |                           |                             |
  |                           |                   runner.wait()
  |                           |                   _pending_results[task_id]
  |                           |                   触发回调 (if registered)
  |                           |                             |
  |  wait_results(count)      |                             |
  +-- _result_cv.wait() <---------------------------------+
  |                           |                             |
  |  返回 results             |                             |
```

**容量控制双重约束** (`_has_runner_capacity`，L328-333)：

```python
def _has_runner_capacity(self) -> bool:
    return (
        not self.runner.paused.is_set()                              # 未暂停
        and self.staleness_manager.get_capacity() > 0                # staleness 容量
        and self.runner.get_input_queue_size() < self.runner.max_queue_size  # 队列未满
    )
```

**active_submit_and_wait**（L634-731）：

这是异步训练的核心方法，持续从 dataloader 提交任务并等待结果：

```
while True:
    1. 计算可提交容量:
       cap_staleness = pending_limit - pending_inputs
       cap_queue = max_queue_size - (input_queue_size + batch_size)
       capacity = min(cap_staleness, cap_queue)

    2. 提交 min(batch_size, capacity) 个任务

    3. wait_results(batch_size - accepted_cnt, timeout=1)
       - dynamic_bs=True: 达到 batch_size 次尝试即返回
       - dynamic_bs=False: 必须收集 batch_size 个 accepted 结果
```

**异常传播**（L314-326）：

后台线程异常通过 `_set_thread_exception` / `_check_thread_exception` 实现 fail-fast 行为，
任何一个线程失败都会通过 `RuntimeError` 传播到主线程。

### 4.5 WorkflowExecutor -- 工作流执行器

**文件**：`workflow_executor.py`（L746-1372） **类定义**：L746

WorkflowExecutor 是 RemoteInfEngine 内部的 rollout 执行编排器。与 RolloutController 使用
BatchTaskDispatcher 的方式类似，但运行在推理 Worker 进程内部。

**与 BatchTaskDispatcher 的关系**：

```
WorkflowExecutor
    |
    +-- 持有 BatchTaskDispatcher[_RolloutTaskInput, _RolloutResult]
    |
    +-- _create_workflow_task()  -->  task_factory 回调
    |       |
    |       +-- 设置 workflow_context (ContextVar)
    |       +-- workflow.arun_episode(engine, data)
    |       +-- check_trajectory_format() 验证
    |       +-- InteractionWithTokenLogpReward 转换
    |       +-- should_accept_fn 过滤
    |       +-- _dump_trajectory() 持久化
    |       +-- staleness_manager.on_accepted/rejected()
    |
    +-- submit() / wait() / prepare_batch()  -->  委托给 dispatcher
```

**轨迹格式验证**（`check_trajectory_format`，L47-226）：

验证 `arun_episode` 返回数据的三种合法格式：

1. `None` -- 被拒绝的轨迹
1. `Dict[str, InteractionWithTokenLogpReward]` -- Agent 交互结果
1. `Dict[str, torch.Tensor]` -- 标准张量格式，需含 `input_ids` + `attention_mask`

### 4.6 RemoteInfEngine -- 远程推理引擎

**文件**：`remote_inf_engine.py`（1457 行） **类定义**：L327

RemoteInfEngine 是 `InferenceEngine` 的 HTTP 客户端实现，通过组合模式注入
`RemoteInfBackendProtocol`（L125）适配不同后端（SGLang/vLLM）。

**核心职责**：

```
RemoteInfEngine
    |
    +-- 服务发现与健康检查 (initialize, L393-469)
    |       地址来源优先级: 参数 > 本地子进程 > name_resolve > 环境变量
    |
    +-- 推理请求路由 (agenerate, L719-884)
    |       - round-robin 选择服务器
    |       - RID 缓存 (128) 保证 KV cache 复用
    |       - 中断恢复循环: 暂停时等待, abort 时续生成
    |
    +-- 权重更新协调 (init_weights_update_group, L886-943)
    |       - update_weights_from_distributed (NCCL/XCCL, L945-973)
    |       - update_weights_from_disk (磁盘, L975-1021)
    |       - 均通过 ProcessPoolExecutor 异步执行
    |
    +-- Workflow 解析 (_resolve_workflow, L541-655)
    |       支持 5 种形式: 实例/类/字符串/Agent 类/Agent 实例
    |
    +-- WorkflowExecutor 委托
            submit/wait/rollout_batch/prepare_batch
```

**推理中断恢复**（`agenerate`，L787-860）：

```python
while stop_reason not in ["stop", "tool_calls", "length"]
      and len(accumulated_output_tokens) < ori_max_new_tokens:
    # 暂停时等待 (权重更新期间)
    while self.workflow_executor.is_paused():
        await asyncio.sleep(0.5)
    # 构建并发送请求
    http_req = self.backend.build_generation_request(req, ...)
    result = await arequest_with_retry(session, ...)
    # 累积 tokens 并更新请求 (KV cache 续生成)
    req.input_ids += gen_result.output_tokens
    req.gconfig.max_new_tokens -= len(gen_result.output_tokens)
```

**`RemoteInfBackendProtocol`**（L125-325）：

Protocol 接口抽象后端差异，包含 11 个方法：

| 方法                                         | 用途                   |
| -------------------------------------------- | ---------------------- |
| `build_generation_request`                   | 构建生成 HTTP 请求     |
| `parse_generation_response`                  | 解析生成响应           |
| `build_disk_weight_update_requests`          | 构建磁盘权重加载请求   |
| `build_distributed_weight_update_requests`   | 构建 NCCL 权重更新请求 |
| `build_init_weights_group_request`           | 构建通信组初始化请求   |
| `get_pause_request` / `get_resume_request`   | 暂停/恢复生成          |
| `get_health_check_request`                   | 健康检查               |
| `get_offload_request` / `get_onload_request` | 模型内存管理           |
| `launch_server`                              | 启动推理服务子进程     |

### 4.7 StalenessManager -- 版本过期管理器

**文件**：`staleness_manager.py`（182 行） **类定义**：L20

StalenessManager 解决异步 RL 训练中的**数据新鲜度问题**：rollout 采样时使用的模型版本 可能已经落后于当前训练版本，过期数据会降低
on-policy 算法的效果。

**容量计算公式**（`get_capacity`，L79-113）：

```
concurrency_capacity = max_concurrent_rollouts - running
staleness_capacity   = (max_staleness + current_version + 1) * consumer_bs - sample_cnt
capacity             = min(concurrency_capacity, staleness_capacity)
```

其中 `sample_cnt = accepted + running`。

**设计意图**：确保当样本最终被训练消费时，其生成时的模型版本与当前版本的差距不超过 `max_staleness`。

**状态转移**：

```
                on_rollout_enqueued()
新提交  -----> enqueued++
                    |
                on_rollout_submitted()
                    |----> enqueued--, running++
                    |
              +-----+-----+
              |            |
     on_rollout_accepted() on_rollout_rejected()
              |            |
         running--         running--
         accepted++        rejected++
```

**Checkpoint 恢复**（`on_version_recovered`，L115-131）：

当从 checkpoint 恢复时，version 从 0 跳到 N，需要设置 `accepted = version * consumer_bs`
以防止容量公式产生突发提交。

### 4.8 AsyncTaskRunner -- 通用异步任务执行器

**文件**：`async_task_runner.py`（685 行） **类定义**：L73

AsyncTaskRunner 是一个**无 AReaL 依赖**的通用异步执行引擎，基于 uvloop 在后台线程运行 asyncio 事件循环。

**核心循环**（`_run_async_loop`，L310-416）：

```
while not exiting:
    if paused: sleep; continue

    _drain_pending_inputs(running_tasks)    # 从 input_queue 取出任务
        对每个 task_input:
        asyncio.create_task(async_fn(*args))  # 创建协程任务

    if not running_tasks:
        _wait_for_new_tasks()               # 等待新输入 (asyncio.Event)
        continue

    done, _ = await asyncio.wait(           # 等待任意任务完成
        tasks, timeout=poll_wait_time,
        return_when=FIRST_COMPLETED)

    for task in done:
        result -> TimedResult(create_time, data, task_id)
        output_queue.put_nowait(result)
```

**线程安全设计**：

- `input_queue` / `output_queue`：`queue.Queue`（线程安全）
- `_active_task_ids`：`set` + `Lock`
- `_input_event`：`asyncio.Event`，用于 `loop.call_soon_threadsafe` 跨线程唤醒
- `_thread_exception`：后台线程异常捕获，在 `_check_thread_health` 中传播

### 4.9 DistRolloutCoordinator -- 分布式 Rollout 协调器

**文件**：`dist_rollout.py`（263 行）

DistRolloutCoordinator 解决多 DP Worker 场景下 rollout 数据的收集和重分布问题。

**redistribute_trajectories**（L29-94）：

```
1. all_gather_tensor_container()    -- 从所有 rank 收集轨迹
2. 计算每条轨迹的 seqlen
3. split_and_unpad_tensor()         -- 去除 padding
4. balanced_greedy_partition()      -- 按 token 数均衡分配到各 rank
5. 返回 RedistributedData(all_data, data, rank, group_indices)
```

**\_broadcast_and_redistribute_trajectories**（L102-150）：

```
仅 DP Head 持有数据
    |
    v
redistribute_trajectories()         -- DP 组内负载均衡
    |
    v
current_platform.synchronize() + dist.barrier()
    |
    v
broadcast_tensor_container()         -- 广播到 context + model 并行组
    |
    v
current_platform.synchronize() + dist.barrier()
    |
    v
所有 rank 均持有分配到的数据
```

### 4.10 WorkflowContext -- 工作流执行上下文

**文件**：`workflow_context.py`（241 行）

**ContextVar 机制**（L24-53）：

```python
@dataclass(frozen=True)
class WorkflowContext:
    is_eval: bool = False
    task_id: int | None = None

_current_context: ContextVar[WorkflowContext] = ContextVar(...)
```

通过 Python `contextvars.ContextVar` 在每个异步任务中传递上下文信息，使 `arun_episode` 内部可以区分 eval 和 train
模式。

**HttpClientManager**（L66-241）：

每个线程维护独立的 `aiohttp.ClientSession` 和 `httpx.AsyncClient`，实现连接池复用。 通过
`register_loop_cleanup` 在事件循环关闭时自动清理。如果检测到事件循环切换（如测试 场景），会自动关闭旧客户端并创建新的。

## 5. RolloutController 与 TrainController 的异步协作机制

### 5.1 训练-推理流水线

```
时间轴 -->

TrainController:   |--train_batch--|--update_weights--|--train_batch--|
                                   |                  |
                          pause_generation()  continue_generation()
                                   |                  |
RolloutController: |---rollout----|x|----wait----|x|---rollout----|
                                   |  NCCL broadcast  |
                                   +--权重同步---------+
```

### 5.2 权重同步详细流程

```
TrainController                   RolloutCallback             RolloutController
     |                                 |                            |
     |-- engine.update_weights(meta) --|                            |
     |                                 |                            |
     |   train_engine 内部:            |                            |
     |   1. pause_generation()  ------>|-- POST /pause_generation ->|
     |      (同步, 等待推理侧暂停)      |                            |-- pause 所有 Worker
     |                                 |                            |
     |   2. init_weights_group(meta) ->|-- POST /init_weights_group |
     |      (异步, 返回 Future)         |                            |-- 在每个 Worker 上
     |      训练侧同时创建 NCCL 组      |                            |   初始化 NCCL 通信组
     |      future.result() 等待       |                            |
     |                                 |                            |
     |   3. update_weights(meta, specs)|-- POST /update_weights_xccl|
     |      (异步, 返回 Future)         |                            |-- 每个 Worker 执行
     |      训练侧同时 NCCL broadcast   |                            |   NCCL recv
     |      双侧并行参与集合通信         |                            |
     |      future.result() 等待       |                            |
     |                                 |                            |
     |   4. continue_generation() ---->|-- POST /continue_gen ----->|
     |      (同步, 确认推理恢复)         |                            |-- resume 所有 Worker
     |                                 |                            |
```

**关键设计**：`init_weights_update_group` 和 `update_weights_from_distributed` 必须 是非阻塞的（返回
`Future`），因为 NCCL 集合通信需要训练侧和推理侧同时参与。如果 这些方法阻塞，训练侧就无法启动自己的 NCCL 操作，导致死锁。

## 6. 关键流程：异步 Rollout 的 submit/wait 模式

### 6.1 本地模式（RemoteInfEngine + WorkflowExecutor）

适用于推理引擎直连场景（eval、单机训练）：

```
RemoteInfEngine.submit(data, workflow)
    |
    +-- _resolve_workflow(workflow, kwargs, group_size)
    |       支持 5 种 workflow 形式
    |
    +-- WorkflowExecutor.submit(data, resolved_workflow)
    |       |
    |       +-- TaskIdGenerator.next()
    |       +-- dispatcher.submit_task_input(_RolloutTaskInput)
    |               -> _pending_inputs.append()
    |               -> staleness_manager.on_rollout_enqueued()
    |
    +-- 返回 task_id

RemoteInfEngine.wait(count)
    |
    +-- WorkflowExecutor.wait(count)
            |
            +-- dispatcher.wait_results(count)
                    -> _result_cv.wait() 直到足够结果
                    -> 按 create_time 排序, 随机 shuffle
                    -> 返回 [result.data for result in selected]
```

### 6.2 远程模式（TrainController + RolloutController）

适用于分布式训练场景：

```
TrainController.prepare_batch(dataloader, workflow)
    |
    +-- self.rollout.prepare_batch(dataloader, workflow)
            |  (RolloutController)
            |
            +-- dispatcher.active_submit_and_wait(data_generator, batch_size)
                    |
                    +-- 持续提交直到满足容量约束
                    |       submit_task_input() -> _pending_inputs
                    |
                    +-- wait_results(batch_size - accepted_cnt, timeout=1)
                    |       -> dispatcher._commit_loop 从 deque 取出
                    |       -> task_factory() 创建 _submit_then_wait
                    |       -> runner.submit() -> AsyncTaskRunner
                    |       -> _submit_then_wait:
                    |           - round-robin 选 Worker
                    |           - scheduler.async_call_engine("submit")
                    |           - await callback future
                    |           - scheduler.async_call_engine("wait_for_task")
                    |       -> dispatcher._fetch_loop 收集结果
                    |
                    +-- 返回 accepted 轨迹列表
```

## 7. 设计模式与工程决策

### 7.1 组合优于继承

- `RemoteInfEngine` 通过注入 `RemoteInfBackendProtocol` 适配不同后端
- `BatchTaskDispatcher` 通过 `task_factory` 回调解耦任务创建逻辑
- `StalenessManager` 通过 `VersionProvider` Protocol 获取版本信息

### 7.2 生产者-消费者 + Condition Variable

BatchTaskDispatcher 的三线程模型使用 `threading.Condition` 实现精确唤醒：

- `_input_cv`：主线程通知生产者有新任务 / 生产者等待容量
- `_result_cv`：消费者通知主线程有新结果 / 主线程等待足够结果
- `_shutdown_event`：统一关闭信号

### 7.3 Fail-Fast 异常传播

后台线程通过 `_set_thread_exception` / `_check_thread_exception` 实现跨线程异常传播。 任何后台线程失败后，主线程在下次
`submit` / `wait` 时立即抛出 `RuntimeError`。

### 7.4 NCCL 死锁预防

RolloutCallback 的非阻塞设计（`_post_nowait` 返回 `Future`）确保训练侧可以立即启动 自己的 NCCL
操作，而不是等待推理侧完成。这是分布式集合通信的基本约束：所有参与方必须 同时进入 collective operation。

### 7.5 版本一致性

StalenessManager 的容量公式 `(max_staleness + version + 1) * consumer_bs` 确保即使
在异步采样场景下，已采集数据在被消费时的过期度也不超过 `max_staleness`。

## 8. 外部依赖与集成

### 8.1 上游依赖

| 模块                           | 用途                                                           |
| ------------------------------ | -------------------------------------------------------------- |
| `areal.api`                    | `InferenceEngine`, `TrainEngine`, `Scheduler`, `Worker` 等抽象 |
| `areal.infra.rpc`              | RPC 序列化 (`serialize_value`, `deserialize_value`)            |
| `areal.infra.utils.concurrent` | `run_async_task`, `get_executor`                               |
| `areal.infra.utils.http`       | `arequest_with_retry`, HTTP 客户端工具                         |
| `areal.utils.data`             | `cycle_dataloader`, `concat_padded_tensors` 等                 |
| `areal.utils.seqpack`          | `balanced_greedy_partition` 负载均衡算法                       |

### 8.2 下游消费者

| 模块                         | 使用方式                                              |
| ---------------------------- | ----------------------------------------------------- |
| `areal.workflow.*`           | 通过 TrainController + RolloutController 编排训练循环 |
| `areal.engine.fsdp_engine`   | 作为 TrainEngine 被 TrainController 管理              |
| `areal.engine.sglang_engine` | 提供 RemoteInfBackendProtocol 实现                    |

### 8.3 第三方库

| 库      | 用途                    | 使用位置                         |
| ------- | ----------------------- | -------------------------------- |
| Flask   | 回调 HTTP 服务器        | RolloutController (L546)         |
| uvloop  | 高性能 asyncio 事件循环 | AsyncTaskRunner (L287)           |
| aiohttp | 异步 HTTP 客户端        | RemoteInfEngine, WorkflowContext |
| httpx   | 备选异步 HTTP 客户端    | WorkflowContext (L158)           |
| Ray     | Worker 调度（可选后端） | 通过 Scheduler 抽象              |
