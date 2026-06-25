# 实验性微服务架构

> 源码位置：`areal/experimental/inference_service/`, `training_service/`, `agent_service/`,
> `weight_update/` 文件数：114 个 Python 文件 | 总行数：16,378 行

## 1. 模块概览

本模块实现了 AReaL 的第二代（V2）分布式架构——基于 HTTP 微服务的推理、训练、Agent 及权重更新服务。与 V1 架构（SPMD + RPC 直连）相比，V2
将每个关注点拆分为独立进程， 通过标准 HTTP/JSON 协议互通，实现了更好的故障隔离、弹性伸缩和可观测性。

### 四个服务的行数分布

```
inference_service/   37 文件   7,382 行  (45.1%)  -- 推理微服务
training_service/    31 文件   3,515 行  (21.5%)  -- 训练微服务
weight_update/       18 文件   3,306 行  (20.2%)  -- 权重更新服务
agent_service/       28 文件   2,175 行  (13.3%)  -- Agent 微服务
```

### 核心设计理念

- **Gateway-Router-Worker 三层模式**：所有四个服务共享统一的微服务分层
- **RPCGuard 进程管理**：通过轻量 Guard 进程 fork 子服务，继承 GPU 环境
- **HTTP 松耦合**：服务间全部通过 HTTP/JSON 通信，无 RPC 直连依赖
- **Duck-type 兼容**：V2 Controller 与 V1 Controller 鸭子类型兼容，训练器无需改码

## 2. 架构总览

### V1 vs V2 架构对比

```
V1 (SPMD + RPC)                          V2 (微服务 + HTTP)
+------------------+                     +------------------+
| TrainController  |                     | RolloutCtrlV2    |
|  (同进程引擎)    |                     |  (纯 HTTP 客户端)|
+--------+---------+                     +--------+---------+
         |                                        |
    RPC (gRPC/ZMQ)                           HTTP/JSON
         |                                        |
+--------+---------+              +---------------+----------------+
| SGLang/vLLM      |              |   Gateway     |    Router      |
| (直接加载模型)   |              |   (FastAPI)   |    (FastAPI)   |
+------------------+              +-------+-------+--------+-------+
                                          |                |
                                    +-----+------+   +----+-----+
                                    | DataProxy  |   | DataProxy|
                                    | (FastAPI)  |   | (FastAPI)|
                                    +-----+------+   +----+-----+
                                          |                |
                                    +-----+------+   +----+-----+
                                    | SGLang     |   | SGLang   |
                                    | (后端进程) |   | (后端进程)|
                                    +------------+   +----------+
```

### 三层微服务请求流

```
                           +-------------------------------------------+
                           |           Controller (编排层)              |
                           |  初始化、版本管理、批次提交、生命周期管理  |
                           +----+-----+-----+-----+-----+------+------+
                                |     |     |     |     |      |
                     fork via   |     |     |     |     |      |  fork via
                     RPCGuard   v     v     v     v     v      v  RPCGuard
                           +----+-----+-----+-----+----+------+------+
                           |                                          |
     +-----------+    +----v----+                              +------v------+
     | 外部客户端|--->| Gateway |                              | Guard (N个) |
     |  (OpenAI) |    | (入口)  |                              | (进程管理)  |
     +-----------+    +----+----+                              +------+------+
                           |                                          |
                     POST /route                                fork/kill
                           |                                     子进程
                      +----v----+
                      | Router  |
                      | (路由)  |
                      +----+----+
                           |
                   worker_addr
                           |
            +--------------+--------------+
            |              |              |
       +----v----+    +----v----+    +----v----+
       |DataProxy|    |DataProxy|    |DataProxy|
       |   #0    |    |   #1    |    |   #N    |
       +----+----+    +----+----+    +----+----+
            |              |              |
       +----v----+    +----v----+    +----v----+
       | Backend |    | Backend |    | Backend |
       | SGLang  |    | SGLang  |    | vLLM    |
       +---------+    +---------+    +---------+
```

## 3. Inference Service（推理微服务）

> 源码：`areal/experimental/inference_service/` 37 文件，7,382 行 — 四个服务中最大、最复杂的

### 3.1 文件结构与行数

| 子目录/文件                | 行数  | 职责                                 |
| -------------------------- | ----- | ------------------------------------ |
| `controller/controller.py` | 1,802 | RolloutControllerV2 — 编排入口       |
| `data_proxy/app.py`        | 818   | DataProxy — 会话管理 + 推理桥接      |
| `data_proxy/session.py`    | 517   | SessionStore — 会话生命周期          |
| `gateway/app.py`           | 633   | Gateway — HTTP 入口代理              |
| `gateway/streaming.py`     | 497   | SSE 流式转发 + Router 通信           |
| `router/app.py`            | 505   | Router — 路由决策 + Worker 注册      |
| `router/state.py`          | 264   | 注册表（Worker/Session/Model/Group） |
| `inf_bridge.py`            | 264   | InfBridge — 后端无关推理桥接         |
| `controller/workflow.py`   | 263   | InferenceServiceWorkflow             |
| `sglang/scheduler.py`      | 251   | AwexSchedulerBridge + PP 入口        |
| `sglang/pp_bridge.py`      | 234   | PPSchedulerBridge — PP 感知权重更新  |
| `vllm/bridge.py`           | 152   | VLLMBridgeBackend                    |
| `sglang/bridge.py`         | 148   | SGLangBridgeBackend                  |
| `sglang/awex.py`           | 141   | /awex/\* HTTP 端点注册               |
| 其余 23 文件               | 893   | 配置、__main__、Guard、auth 等       |

### 3.2 RolloutControllerV2（controller.py, 1,802 行）

这是推理微服务的编排核心，与 V1 的 `RolloutController` **鸭子类型兼容**但完全独立 实现。训练器可以无需改码地使用任一版本。

**初始化流水线**（`_async_initialize`）：

```
Step 0: 通过 Scheduler 创建 RPCGuard workers
        |
        +---> self._workers_ready.set()  <-- 主线程可继续
        |
Step 1+2 (并行):
        +---> fork Router (on guard_0)
        +---> fork 推理后端进程 (SGLang/vLLM)
        |
Step 3+4 (并行):
        +---> fork DataProxy x dp_size
        +---> fork Gateway (on guard_0)
        |
Step 5: 在 Router 中注册所有 DataProxy
Step 6: 创建 WorkflowExecutor + StalenessManager
Step 7: 注册模型
```

**关键设计决策**：

1. **流水线初始化**：`initialize()` 在后台线程中执行完整初始化。一旦 Guard workers 就绪（~Step 0）就立即通过
   `_workers_ready.set()` 通知主线程，使训练侧可以提前 开始自己的初始化，两侧并行降低启动延迟。

1. **多节点推理**：当 `tp_size * pp_size > n_gpus_per_node` 时，自动计算 `nnodes_per_instance` 并跨节点
   fork 推理进程，head 节点分配 rendezvous 端口。

1. **在线/离线双模式**：离线模式通过 `InferenceServiceWorkflow` 驱动完整的 start_session → agent.run →
   set_reward → export 流程；在线模式通过 callback server 接收 DataProxy 推送的就绪通知。

### 3.3 Gateway（gateway/app.py, 633 行）

轻量 FastAPI HTTP 代理，**不持有任何 Worker 状态**——所有状态由 Router 管理。

**路由端点**：

| 端点                          | 认证              | 行为                         |
| ----------------------------- | ----------------- | ---------------------------- |
| `POST /chat/completions`      | admin/session key | 流式或非流式转发至 DataProxy |
| `POST /v1/chat/completions`   | 同上              | OpenAI 兼容别名              |
| `POST /rl/start_session`      | admin only        | 创建会话 → Router 注册       |
| `POST /rl/set_reward`         | session/admin key | 设置奖励分数                 |
| `POST /export_trajectories`   | admin only        | 导出轨迹 → 清理 session      |
| `POST /register_model`        | admin only        | 注册模型 → 广播至 DataProxy  |
| `POST /pause_generation/{id}` | admin only        | 暂停指定 worker 生成         |
| `POST /set_version/{id}`      | admin only        | 设置指定 worker 版本号       |

**流式代理**（`forward_sse_stream`）：使用 `httpx.AsyncClient.stream()` 实现真正的 SSE 逐 chunk
透传，客户端可实时看到 token。上游错误时注入 SSE error event。

### 3.4 Router（router/app.py, 505 行）

纯路由决策服务，**绝不代理流量**——只回答"去哪个 Worker"。

**四层注册表**（`router/state.py`, 264 行）：

```
WorkerRegistry   -- worker_addr -> WorkerInfo (health, active_requests)
SessionRegistry  -- session_api_key -> worker_addr (会话固定)
                    session_id -> worker_addr
ModelRegistry    -- model_name -> ModelInfo (url, api_key, data_proxy_addrs)
GroupRegistry    -- group_id -> GroupInfo (worker_addr, session_ids)
```

**路由决策流程**（`POST /route`）：

```
1. 解析 model -> 候选 worker_addrs (ModelRegistry)
2. session_id 存在? -> 返回固定 worker (SessionRegistry)
3. api_key 为 session key? -> 返回固定 worker (SessionRegistry)
4. api_key 为 admin key? -> strategy.pick(candidates)
5. 均不匹配 -> 404
```

**路由策略**（`router/strategies.py`, 44 行）：当前实现 `RoundRobinStrategy`， `least_busy` 策略预留但未接入
`active_requests` 追踪。

**健康检查**：后台 `_poll_workers` 协程按 `poll_interval` 周期 GET 每个 Worker 的 `/health`，更新
`is_healthy` 标记。

### 3.5 DataProxy（data_proxy/app.py, 818 行）

推理微服务的"大脑"——会话管理、推理桥接、轨迹存储三合一。

**核心职责**：

1. **会话管理**：创建/销毁 RL 会话，API key 认证
1. **推理调度**：通过 `InfBridge` 发送生成请求到后端
1. **轨迹收集**：通过 `SessionStore` 存储 interaction → 组装为训练轨迹

**InfBridge**（`inf_bridge.py`, 264 行）：

后端无关的 HTTP 客户端，实现 `_AsyncGenerateEngine` 协议。核心是 **pause/abort/resubmit 循环**：

```
for attempt in range(max_resubmit_retries):
    while paused:          # <-- 权重更新期间等待
        await sleep(0.5)
    remaining = budget - len(accumulated_tokens)
    backend.patch_generation_request(http_req, accumulated_tokens, remaining)
    result = await _send_request(http_req)
    accumulated_tokens.extend(result.output_tokens)
    if stop_reason in ("stop", "tool_calls", "length"):
        break
    # stop_reason == "abort" -> 继续循环（重新提交）
```

这个设计使得权重更新可以中途中断推理，推理完成后自动拼接 token，对上层完全透明。

**SessionStore**（`session.py`, 517 行）：

```
SessionData
  |-- active_completions: InteractionCache  (当前进行中)
  |-- ready_trajectories: OrderedDict[int, ReadyTrajectory]  (已就绪)
  |-- _next_trajectory_id: int  (递增)
```

支持离线（单次 set_reward → export）和在线（重复 set_reward → export）两种模式。 会话超时 3600 秒自动清理。

### 3.6 InfBridgeBackend 抽象

`backend.py`（94 行）定义 `InfBridgeBackend` Protocol，隔离推理后端差异：

| 方法                        | SGLang 实现                       | vLLM 实现                 |
| --------------------------- | --------------------------------- | ------------------------- |
| `build_generation_request`  | `POST /generate`                  | `POST /v1/completions`    |
| `parse_generation_response` | `meta_info.output_token_logprobs` | `choices[0].logprobs`     |
| `get_pause_request`         | `/pause_generation`               | `/areal_pause_generation` |
| `get_offload_request`       | `/release_memory_occupation`      | `/sleep`                  |
| `get_onload_request`        | `/resume_memory_occupation`       | `/wake_up`                |

### 3.7 SGLang 桥接

**AwexSchedulerBridge**（`sglang/scheduler.py`, 251 行）：

在 SGLang `Scheduler.__init__()` 之后创建，通过 `setattr` 将 `awex_*` 方法注入到 scheduler
实例上（无继承、无猴子补丁）。方法列表：

```
awex_report_weight_meta      -- 收集模型参数元数据
awex_report_parallelism      -- 报告并行策略
awex_init_weights_update_group  -- 初始化 NCCL 组
awex_execute_weight_update   -- 执行权重同步
awex_batch_isend_irecv       -- 批量 P2P 通信
awex_get_parameters          -- 保存参数（调试）
awex_init_colocate_weight_update  -- 共置模式初始化
awex_execute_colocate_weight_update  -- 共置模式执行
awex_release_memory / awex_resume_memory  -- 内存管理
```

数据返回通过 ZMQ PUSH socket（仅 `tp_rank==0 && dp_rank==0`）。

**PPSchedulerBridge**（`sglang/pp_bridge.py`, 234 行）：

Pipeline Parallel 场景下的权重更新路由。训练侧每个 PP stage 创建独立 NCCL group （命名
`update_weight_group_{pp_rank}`），此桥接拦截 `init_weights_update_group`、
`update_weights_from_distributed`、`destroy_weights_update_group` 三个方法：

```
if pp_rank_from_group_name != worker.pp_rank:
    model_runner._model_update_group[name] = None  # sentinel
    return "skipped"
else:
    return original_method(recv_req)  # 正常参与
```

包含 NCCL 初始化 watchdog——30/60/120 秒无响应打印警告，辅助 hang 诊断。

**AWEX HTTP 端点**（`sglang/awex.py`, 141 行）：

在 SGLang FastAPI app 上注册 `/awex/*` 端点，将 HTTP 请求转发为 `rpc_proxy.collective_rpc()` ZMQ 调用。

## 4. Training Service（训练微服务）

> 源码：`areal/experimental/training_service/` 31 文件，3,515 行

### 4.1 文件结构与行数

| 子目录/文件                | 行数  | 职责                               |
| -------------------------- | ----- | ---------------------------------- |
| `controller/controller.py` | 1,021 | GatewayTrainController             |
| `data_proxy/dispatcher.py` | 403   | DP-aware 张量分发                  |
| `worker/engine.py`         | 308   | Engine Blueprint — HTTP 化引擎接口 |
| `worker/app.py`            | 214   | Worker Flask 应用                  |
| `worker/awex.py`           | 211   | AWEX 端点注册                      |
| `data_proxy/engine.py`     | 164   | DataProxy Engine 封装              |
| `router/app.py`            | 166   | Router — API key→模型地址          |
| `gateway/engine.py`        | 154   | Gateway Engine 封装                |
| `gateway/app.py`           | 124   | Gateway FastAPI 应用               |
| `data_proxy/topology.py`   | 102   | WorkerTopology 发现                |
| 其余 21 文件               | 648   | 配置、__main__、auth、streaming 等 |

### 4.2 GatewayTrainController（controller.py, 1,021 行）

训练微服务编排器。注意文件头的 TODO：**当前 V2 尚未完全替代 TrainController， PPO/GRPO 路径仍使用 V1**。

**初始化流程**：

```
Step 0: 创建 world_size 个 Guard (一个 per GPU rank)
Step 1: 分配 NCCL master_addr/master_port
Step 1.5: 通过 /set_env 设置每个 Guard 的 NCCL 环境变量
Step 2: fork train-worker x world_size (并行)
Step 3: 在所有 Worker 上 create_engine + create_process_group + initialize
Step 4: fork Router
Step 5: fork DataProxy
Step 6: fork Gateway
Step 7: 在 Router 中注册 DataProxy + API key
```

**关键差异（vs 推理服务）**：

- 训练侧每个 Guard 分配一个 GPU（推理侧可多 GPU per Guard）
- 引擎通过 HTTP `/create_engine` 远程实例化（动态导入 `engine_class`）
- NCCL 进程组通过 `/create_process_group` 远程协调创建

### 4.3 Dispatcher（data_proxy/dispatcher.py, 403 行）

训练 DataProxy 的核心——DP-aware 张量分发器，复制了 V1 `TrainController` 的分发语义。

**两种请求模式**：

1. **`dispatch(path)`** — DP-aware 分区分发：

   ```
   POST body -> 检测是否包含张量
     |
     +-- 含张量 -> _dispatch_tensors 按 DP 分区
     |             DP heads 收到分片, non-heads 收到空信号
     |             收集 DP head 结果 -> _merge_tensors 合并
     |
     +-- 纯标量 -> _scalar_fan_out
                   所有 workers 收到请求（NCCL collective 参与）
                   返回第一个 DP head 的结果
   ```

1. **`broadcast(path)`** — 全 worker 广播：

   ```
   相同请求发往所有 workers, 返回所有响应
   ```

**关键**：non-DP-head workers 必须收到空信号（`_empty_payload`），否则无法参与 intra-group NCCL collective
导致 hang。

### 4.4 Worker Engine Blueprint（worker/engine.py, 308 行）

通过 Flask Blueprint 将 `TrainEngine` 的方法暴露为 HTTP 端点。采用 `_register_compute_route` 和
`_register_engine_route` 两种模式批量注册：

**计算路由**（需要张量序列化 + broadcast）：

```
/train_batch, /forward_batch, /eval_batch
/sft/train, /sft/evaluate
/ppo/actor/compute_logp, /ppo/actor/update
/ppo/critic/compute_values, /ppo/critic/update
/rw/train, /rw/evaluate
```

**引擎路由**（直接调用引擎方法）：

```
/create_engine, /destroy_engine, /configure
/train, /eval, /initialize, /topology
/set_version, /get_version, /save, /load
/offload, /onload, /optimizer_zero_grad, /optimizer_step
/export_stats, /get_device_stats, /get_param_info
```

### 4.5 WorkerTopology（topology.py, 102 行）

启动时并行 GET 所有 Worker 的 `/topology` 端点，构建全局拓扑视图：

```python
@dataclass
class WorkerTopology:
    workers: list[WorkerInfo]     # 所有 worker
    dp_heads: list[int]           # DP head 索引列表
    dp_size: int                  # 数据并行度
    dp_groups: list[list[int]]    # dp_rank -> worker 索引列表
    pp_size, tp_size, cp_size, ep_size: int  # 其余并行维度
```

## 5. Agent Service（Agent 微服务）

> 源码：`areal/experimental/agent_service/` 28 文件，2,175 行

### 5.1 文件结构与行数

| 子目录/文件                | 行数 | 职责                            |
| -------------------------- | ---- | ------------------------------- |
| `controller/controller.py` | 583  | AgentController — 弹性伸缩编排  |
| `protocol.py`              | 313  | WebSocket 帧协议                |
| `gateway/app.py`           | 185  | Gateway FastAPI 应用            |
| `gateway/bridge.py`        | 172  | OpenResponsesBridge             |
| `data_proxy/app.py`        | 155  | DataProxy — 会话→Worker 代理    |
| `router/app.py`            | 91   | Router — 极简路由               |
| `worker/app.py`            | 92   | Worker — Agent 执行器           |
| `types.py`                 | 66   | 共享类型定义                    |
| 其余 20 文件               | 518  | 配置、client、auth、__main__ 等 |

### 5.2 AgentController（controller.py, 583 行）

Agent 服务独有的特性是 **弹性伸缩**——`scale_up(n)` / `scale_down(n)` 动态增减 Worker+DataProxy 对。

**初始化流程**：

```
Step 1: Scheduler 创建 Guard workers
Step 2: fork Router (on guard_0)
Step 3: scale_up(1) -- fork Worker+DataProxy 对, 注册到 Router
Step 4: fork Gateway (on guard_0)
Step 5: 启动健康监控线程
```

**弹性伸缩**：

```python
# scale_up: 增加 Worker+DataProxy 对
def scale_up(count):
    for i in range(count):
        fork agent-worker-{i} (on guard, round-robin)
        fork agent-proxy-{i}  (on same guard)
        register proxy in Router
        # 失败回滚: cleanup_pair_forks

# scale_down: 移除 Worker+DataProxy 对 (LIFO 顺序)
def scale_down(count):
    for pair in reversed_pairs[:count]:
        unregister from Router (3 次重试)
        drain active sessions (等待 active_sessions=0)
        kill DataProxy + Worker
```

**健康监控**：后台线程按 `_DEFAULT_HEALTH_POLL_INTERVAL`（5 秒）周期检查所有 proxy 的 `/health`，使用
`ThreadPoolExecutor` 并行探测。

### 5.3 Protocol（protocol.py, 313 行）

实现类 OpenClaw 的 WebSocket 帧协议，三种帧类型：

```
RequestFrame (client -> gateway)
  .id: str              -- 请求 ID
  .method: "agent"      -- 调用方法
  .params.message       -- 用户消息
  .params.sessionKey    -- 会话亲和键
  .params.queueMode     -- "collect" | "followup"

ResponseFrame (gateway -> client)
  .id: str              -- 匹配请求 ID
  .ok: bool             -- 是否成功
  .payload.runId        -- 运行 ID
  .payload.status       -- "accepted" | "complete" | "failed"

EventFrame (gateway -> client, streaming)
  .event: str           -- 事件类别 (e.g. "agent")
  .payload.runId        -- 运行 ID
  .payload.delta        -- 增量文本
  .payload.toolCall     -- 工具调用
```

`QueueMode.COLLECT` 合并排队消息为一个追加轮次，`FOLLOWUP` 排队为下一轮。

### 5.4 OpenResponsesBridge（gateway/bridge.py, 172 行）

将 OpenAI Responses API (`POST /v1/responses`) 翻译为 DataProxy 会话轮次：

```
1. 从 input_items 提取 message（支持 input_text, function_call_output）
2. 从 user + model 派生 session_key: "agent:{model}:{user}"
3. POST Router /route 获取 data_proxy_addr
4. POST DataProxy /session/{key}/turn 发送消息
5. 将 DataProxy 结果转换为 OpenAI Response 格式
```

## 6. Weight Update Service（权重更新服务）

> 源码：`areal/experimental/weight_update/` 18 文件，3,306 行

### 6.1 文件结构与行数

| 子目录/文件                | 行数 | 职责                          |
| -------------------------- | ---- | ----------------------------- |
| `gateway/app.py`           | 806  | Gateway — 连接管理 + 更新协调 |
| `awex/sglang_adapter.py`   | 641  | SGLang 推理侧 AWEX 适配器     |
| `awex/megatron_adapter.py` | 550  | Megatron 训练侧 AWEX 适配器   |
| `awex/fsdp_adapter.py`     | 342  | FSDP 训练侧 AWEX 适配器       |
| `nccl_group.py`            | 225  | NCCL 进程组管理               |
| `controller/controller.py` | 208  | WeightUpdateController        |
| `training_adapter.py`      | 86   | AwexTrainingAdapter Protocol  |
| `inference_adapter.py`     | 86   | AwexInferenceAdapter Protocol |
| `gateway/kv_store.py`      | 64   | 参数元数据 KV 存储            |
| 其余 9 文件                | 298  | 配置、pair_registry、auth 等  |

### 6.2 AWEX 协议概述

AWEX (Asynchronous Weight Exchange) 是 AReaL 的权重同步协议，通过 NCCL P2P 操作直接在训练 GPU 和推理 GPU
之间传输模型参数，避免磁盘 I/O。

```
+------------------+                    +------------------+
| Training Engine  |                    | Inference Engine |
| (FSDP/Megatron)  |                    | (SGLang/vLLM)    |
+--------+---------+                    +--------+---------+
         |                                       |
    AwexTrainingAdapter                  AwexInferenceAdapter
    (报告 shard 元数据)                  (报告 shard 元数据)
         |                                       |
         +--------->  Weight Update  <-----------+
                      Gateway (HTTP)
                           |
                      TransferPlan
                      (awex 库计算)
                           |
                 NCCL P2P isend/irecv
                    (直接 GPU 传输)
```

### 6.3 Weight Update Gateway（gateway/app.py, 806 行）

权重更新的中心协调器。支持三种模式：

**模式一：AWEX（默认）**

`POST /connect` 流程：

```
1. GET 训练/推理端的 /awex/report_parallelism -- 获取并行策略
2. POST 训练/推理端的 /awex/report_weight_meta -- 收集参数元数据
3. _merge_training_meta_by_name -- 合并 DP 副本的分片信息
4. 存入 KV Store (pair_name -> meta)
5. POST /awex/init_weights_update_group -- 初始化 NCCL 进程组
   - 推理侧 rank: [0, num_engines * infer_world_size)
   - 训练侧 rank: [total_infer_ranks, total_world_size)
6. POST /awex/batch_isend_irecv -- 活性检查（避免后续 hang）
```

`POST /update_weights` 流程：

```
1. POST 推理端 /pause_generation -- 暂停推理
2. POST 训练端 /awex/update_weights -- 触发 NCCL send
3. POST 推理端 /awex/update_weights -- 触发 NCCL recv (与 send 配对)
4. POST 推理端 /continue_generation -- 恢复推理
5. POST 推理端 /set_version -- 更新版本号
```

**模式二：Colocate（共置）**

训练和推理在同一 GPU 上时，使用 CUDA IPC 传输代替 NCCL P2P。Gateway 协调 `init_colocate_weight_update` +
`execute_colocate_weight_update`。

**模式三：Disk（磁盘）**

通过共享文件系统保存/加载权重。训练侧 `save_pretrained` → 推理侧 `update_weights _from_disk`。最简单但最慢。

### 6.4 NCCL 进程组管理（nccl_group.py, 225 行）

`init_custom_process_group` 允许在已有全局进程组之外创建独立 NCCL 组，解决 torchrun 的
`TORCHELASTIC_USE_AGENT_STORE=True` 阻止创建新 TCP store 的问题。

`setup_batch_isend_irecv` 在 NCCL 组建立后执行一次简单 P2P 通信作为活性检查：

- 偶数 world_size：前半 recv from 后半，后半 send to 前半
- 奇数 world_size：环形通信

### 6.5 AWEX 适配器

**AwexTrainingAdapter Protocol**（86 行）与 **AwexInferenceAdapter Protocol**（86 行）：

镜像设计——训练侧 build_send_ops，推理侧 build_recv_ops。

```
                 共同方法
+-----------------------------------------------+
| parallelism_strategy -> dict                   |
| get_weight_metadata -> list[ParameterMeta]     |
| get_local_shard_parameters -> dict[str,Tensor] |
| init_weight_update_group(...)                  |
| execute_weight_update(version)                 |
| batch_isend_irecv(**kwargs)                    |
| teardown_weight_update_group()                 |
| init_colocate_weight_update(...)               |
| execute_colocate_weight_update(version)        |
| release_memory(tags) / resume_memory(tags)     |
+-----------------------------------------------+
```

**三个具体适配器**：

| 适配器                | 行数 | 训练/推理 | 引擎             | 分片策略                 |
| --------------------- | ---- | --------- | ---------------- | ------------------------ |
| `AwexFSDPAdapter`     | 342  | Training  | FSDPEngine       | DTensor Shard placements |
| `AwexMegatronAdapter` | 550  | Training  | MegatronEngine   | TP all_gather + PP stage |
| `AwexSGLangAdapter`   | 641  | Inference | SGLang Scheduler | sglang_sharding_strategy |

**AwexFSDPAdapter** 处理 DTensor 分片元数据：

```python
for name, param in model.named_parameters():
    if isinstance(param.data, DTensor):
        shard_meta = _extract_dtensor_shard_meta(name, tensor, rank_info)
    else:
        shard_meta = _extract_plain_shard_meta(name, tensor, rank_info)
```

**AwexMegatronAdapter** 特殊处理：

- PP：`get_named_parameters` 仅返回当前 stage 的层（全局索引正确）
- TP：`all_gather_param` 先在 TP 组内聚合再转换为 HF 格式
- `dp_replicated=True` 告知 awex 同一 DP 组内 TP ranks 持有相同完整张量

**AwexSGLangAdapter** 特殊处理：

- 通过 `get_sglang_rank_info` / `get_sglang_sharding_strategy` 获取 SGLang 内部的分片信息
- 支持 colocate 模式的 CUDA IPC 反序列化
- 内存管理：`release_memory` 释放 KV cache + 模型权重到 CPU

## 7. 共享设计模式

### 7.1 RPCGuard 进程管理

所有四个服务共享相同的 Guard 模式：

```
Controller
    |-- Scheduler.create_workers(job) --> Guard 进程 (持有 GPU/端口)
    |-- POST guard/alloc_ports          --> 获取可用端口
    |-- POST guard/fork                 --> fork 子服务进程
    |-- POST guard/set_env              --> 设置环境变量（训练侧 NCCL）
    |-- POST guard/kill_forked_worker   --> 清理子进程
```

Guard 退出时自动清理所有 forked children（`cleanup_forked_children`）。

### 7.2 认证模型

所有服务使用统一的两级认证：

```
Admin API Key (全局管理密钥)
  |-- 控制面操作: register_model, start_session, pause, set_version
  |-- HMAC constant-time 比较 (hmac.compare_digest)
  |
Session API Key (会话级临时密钥)
  |-- 数据面操作: chat/completions, set_reward
  |-- 由 start_session 返回, 绑定到特定 DataProxy
```

### 7.3 Pause/Resume 协议

权重更新期间的推理暂停：

```
WeightUpdate Gateway                   DataProxy              InfBridge
     |                                    |                      |
     +-- POST /pause_generation --------->|                      |
     |                                    +-- PauseState=True -->|
     |                                    +-- POST backend       |
     |                                    |   /pause_generation  |
     |                                    |                      |
     |     (NCCL 权重传输...)             |                      |
     |                                    |   InfBridge.agenerate|
     |                                    |   while paused:      |
     |                                    |     await sleep(0.5) |
     |                                    |                      |
     +-- POST /continue_generation ------>|                      |
     |                                    +-- POST backend       |
     |                                    |   /continue          |
     |                                    +-- PauseState=False ->|
     |                                    |                      |
     |                                    |   resubmit with      |
     |                                    |   accumulated tokens |
```

### 7.4 版本管理

```
TrainController                 WeightUpdate          InfController
     |                          Gateway                    |
     +-- train step N -------->  |                         |
     |                          +-- /update_weights -----> |
     |                          |                     (NCCL recv)
     |                          +-- POST /set_version ---> |
     |                          |        version=N         |
     |                          |                     DataProxy
     |                          |                     InfBridge._version=N
     |                          |                     每个 token 标记版本
```

每个生成的 token 带版本标签 `output_versions`，`StalenessManager` 据此判断
轨迹是否过期（`max_head_offpolicyness`）。

## 8. 关键数据流

### 8.1 离线 RL 训练循环

```
prepare_batch(dataloader, workflow)
  |
  +-- for item in batch:
  |     submit(item, InferenceServiceWorkflow)
  |       |
  |       +-- _start_session(gateway) --> session_id, api_key
  |       +-- agent.run(data, base_url=gateway, api_key=session_api_key)
  |       |     |
  |       |     +-- POST gateway/chat/completions
  |       |     |     |-> Router /route -> DataProxy addr
  |       |     |     |-> DataProxy -> InfBridge -> SGLang /generate
  |       |     |     |<- response tokens + logprobs
  |       |     +-- (agent 逻辑: 工具调用、多轮对话)
  |       |
  |       +-- _set_last_reward(gateway, reward, session_api_key)
  |       +-- _export_interactions(gateway, session_ids, group_id)
  |             |-> DataProxy 组装 trajectory dict
  |             |-> Router 清理 session 注册
  |
  +-- wait(count=batch_size)
  |     |-> 返回 list[trajectory_dict]
  |
  +-- TrainEngine.train_batch(trajectories)
  |
  +-- WeightUpdate.update_weights(version)
        |-> pause -> NCCL P2P -> resume -> set_version
```

### 8.2 在线 Agent 模式

```
外部客户端 --POST /chat/completions--> Gateway
                                         |
                                    Router /route
                                    (session key 固定)
                                         |
                                    DataProxy
                                    (InfBridge -> SGLang)
                                         |
外部客户端 --POST /rl/set_reward------> DataProxy
                                    (标记 trajectory ready)
                                         |
                                    callback -> Controller
                                    (wait_for_online_trajectory)
                                         |
                                    _export_interactions
                                    (训练消费)
```

## 9. 设计约束与待改进

### 当前限制

1. **训练 V2 不完整**：`GatewayTrainController` TODO 注释明确指出 PPO/GRPO 路径 尚未迁移到 V2，需补齐
   `connect_engine`、`prepare_batch/rollout_batch`、 `update_weights` 的完整实现

1. **路由策略有限**：仅实现 `RoundRobinStrategy`，`least_busy` 需要接入 `active_requests` 追踪

1. **状态全内存**：Router 的四个注册表均为内存数据结构（`router/state.py` 第 6 行注释："All state is in-memory
   (lost on restart)"），不支持 Router 重启恢复

1. **Agent 弹性有限**：当前 `scale_up/scale_down` 仅支持 Worker+DataProxy 对级别 的伸缩，不支持 Router 或
   Gateway 的水平扩展

### 扩展点

- `InfBridgeBackend` Protocol 可扩展新的推理后端
- `AwexTrainingAdapter` / `AwexInferenceAdapter` Protocol 可扩展新的引擎类型
- `RoutingStrategy` Protocol 可插入自定义路由策略
- Worker Blueprint 的 `_register_compute_route` / `_register_engine_route` 可轻松添加新的训练端点
