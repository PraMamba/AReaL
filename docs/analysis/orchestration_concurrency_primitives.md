# AReaL 编排与并发原语深度分析

> 基于源码的详细分析，覆盖 asyncio + HTTP RPC 通信层、可插拔 Ray/Slurm 调度器、容错机制和性能权衡。

---

## 目录

1. [架构总览](#1-架构总览)
2. [asyncio + HTTP RPC 通信层](#2-asyncio--http-rpc-通信层)
   - 2.1 [RPC Server 架构](#21-rpc-server-架构)
   - 2.2 [HTTP 客户端与重试机制](#22-http-客户端与重试机制)
   - 2.3 [RTensor 远程张量传输](#23-rtensor-远程张量传输)
   - 2.4 [异步任务执行器 (AsyncTaskRunner)](#24-异步任务执行器-asynctaskrunner)
   - 2.5 [与 Ray Actor 模型的延迟对比](#25-与-ray-actor-模型的延迟对比)
3. [可插拔的 Ray/Slurm 编排底座](#3-可插拔的-rayslurm-编排底座)
   - 3.1 [双层部署架构：Launcher vs Scheduler](#31-双层部署架构launcher-vs-scheduler)
   - 3.2 [Launcher 系统](#32-launcher-系统)
   - 3.3 [Scheduler 抽象层](#33-scheduler-抽象层)
   - 3.4 [三种 Scheduler 实现](#34-三种-scheduler-实现)
   - 3.5 [名称解析服务 (NameResolve)](#35-名称解析服务-nameresolve)
   - 3.6 [端到端部署流程](#36-端到端部署流程)
4. [容错与优雅关闭](#4-容错与优雅关闭)
5. [并发模式总结](#5-并发模式总结)
6. [代码质量发现](#6-代码质量发现)
7. [设计权衡总结](#7-设计权衡总结)

---

## 1. 架构总览

AReaL 的编排系统采用 **分层混合设计**：

```
┌─────────────────────────────────────────────────────────┐
│                    用户入口 (CLI)                        │
│  python -m areal.infra.launcher.{local,slurm,ray}      │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Launcher 层 (作业提交)                      │
│  LocalLauncher / SlurmLauncher / RayLauncher            │
│  职责: 进程启动、GPU分配、环境变量、日志路由              │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│             Scheduler 层 (运行时控制面)                   │
│  LocalScheduler / SlurmScheduler / RayScheduler          │
│  职责: Worker 生命周期、引擎创建、RPC 调用、健康检查       │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│            Controller 层 (训练编排)                       │
│  TrainController / RolloutController                     │
│  职责: 数据分发、结果合并、权重同步、容量管理              │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              RPC / 传输层                                │
│  HTTP (Flask + aiohttp) / Ray Actor / NCCL               │
│  职责: 方法调用序列化、张量传输、集合通信                  │
└─────────────────────────────────────────────────────────┘
```

核心设计理念：**控制面用 HTTP/Ray RPC（灵活、可调试），数据面用 NCCL（高性能、零拷贝）**。

---

## 2. asyncio + HTTP RPC 通信层

### 2.1 RPC Server 架构

**源码**: `areal/infra/rpc/rpc_server.py`

每个 Worker 进程运行一个 Flask HTTP Server（Werkzeug `make_server`, `threaded=True`），作为该 Worker 的 RPC 端点。

#### 引擎线程模型

这是 RPC Server 最关键的设计：**所有引擎操作通过单一后台线程串行执行**，以保证 NCCL 线程安全。

```python
# rpc_server.py:46-51
# Engine thread for executing all engine-related endpoints serially
# This ensures NCCL compatibility by running engine operations in a single thread,
# while allowing /data/ endpoints to be processed concurrently
_engine_thread: Thread | None = None
_engine_work_queue: Queue[tuple[Callable, tuple, dict, Future]] | None = None
_engine_thread_lock = Lock()
```

工作流程：

```
Flask Handler Thread(s)        Engine Worker Thread
         │                            │
    POST /call ──────────────→ work_queue.get()
         │                     func(*args, **kwargs)
    future.result() ←──────── future.set_result(result)
         │                            │
    POST /data/xxx  ─→ 直接处理（不经过引擎线程）
```

```python
# rpc_server.py:119-126
def _submit_to_engine_thread(func_name: str, func: Callable, *args, **kwargs) -> Any:
    _init_engine_thread()
    future = Future()
    _engine_work_queue.put((func, args, kwargs, future, func_name))
    return future.result()  # Block until result is available
```

**设计亮点**:
- **控制面/数据面分离**: `/call` 端点的引擎方法调用通过引擎线程串行化，而 `/data/<shard_id>` 的 PUT/GET 端点可并发处理，不经过引擎线程。这在保证 NCCL 安全的同时，不阻塞张量数据传输。
- **基于名称的服务发现**: Worker 启动时通过 `name_resolve.add()` 注册自己的 `host:port`（`rpc_server.py:1032`），Controller 通过键路径查找 Worker 地址，实现了 Worker 启动与 Controller 感知的解耦。

**端点总览**:

| 端点 | HTTP 方法 | 走引擎线程 | 功能 |
|------|----------|-----------|------|
| `/health` | GET | 否 | 健康检查，返回引擎数量和名称 |
| `/alloc_ports` | POST | 否 | 分配空闲端口 |
| `/fork` | POST | 否 | 派生子进程（用于 colocation） |
| `/kill_forked_worker` | POST | 否 | 终止派生子进程 |
| `/configure` | POST | 否 | 推送实验配置 |
| `/set_env` | POST | 否 | 设置环境变量 |
| `/create_engine` | POST | **是** | 创建引擎实例 |
| `/call` | POST | **是** | 调用引擎方法 |
| `/data/<shard_id>` | PUT/GET/DELETE | 否 | 张量分片存取 |

### 2.2 HTTP 客户端与重试机制

**源码**: `areal/infra/utils/http.py`

异步 HTTP 客户端 `arequest_with_retry()` 使用 `aiohttp`，支持超时和重试：

```python
# http.py:20-30
async def arequest_with_retry(
    addr: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    session: aiohttp.ClientSession | None = None,
    method: str = "POST",
    max_retries: int | None = None,    # 默认 1（实际只请求 1 次）
    timeout: float | None = None,       # 默认 3600s
    retry_delay: float = 1.0,
    verbose=False,
) -> dict | str | bytes:
```

**连接器配置**:

```python
# http.py:16-17
def get_default_connector():
    return aiohttp.TCPConnector(limit=0, use_dns_cache=False, force_close=True)
```

`force_close=True` 意味着每次请求后销毁 TCP 连接，无法利用 HTTP Keep-Alive。这在 Controller 频繁调用同一 Worker 的场景下会引入不必要的 TCP 握手开销。

### 2.3 RTensor 远程张量传输

**源码**: `areal/infra/rpc/rtensor.py`

`RTensor` 是分布式张量传输的核心抽象，支持 **双后端透明切换**：

```python
# rtensor.py:20-61
class TensorBackend(Protocol):
    def fetch(self, shards: list[TensorShardInfo]) -> list[torch.Tensor]: ...
    def store(self, tensor: torch.Tensor) -> Any: ...
    async def delete(self, node_addr: str, shard_ids: list[Any]) -> None: ...
```

两个后端实现：

| 后端 | 存储机制 | 序列化方式 | 适用场景 |
|------|---------|-----------|---------|
| `HttpTensorBackend` | Worker 进程内存 + HTTP PUT/GET | Base64 + JSON（via `orjson`） | Local/Slurm Scheduler |
| `RayTensorBackend` | Ray Object Store（共享内存 + Arrow） | 零拷贝（同节点） | Ray Scheduler |

**自动后端检测**:

```python
# rtensor.py:151-158 (概念)
def get_backend() -> TensorBackend:
    if _backend is None:
        if ray.is_initialized():
            _backend = RayTensorBackend()
        else:
            _backend = HttpTensorBackend()
    return _backend
```

**延迟物化设计**: `RTensor.from_batched()` 将分片存储到远端，仅保留 meta tensor 在本地。实际数据传输推迟到 `to_local()` 调用时。这对数据并行分发非常重要——Controller 拆分和路由数据时无需将张量全部加载到内存。

**序列长度感知分区**: `data_parallel_dispatch` 使用 `balanced_greedy_partition` 按序列长度进行贪心平衡分区，避免某个 DP 组因不成比例的长序列而成为瓶颈。这对 LLM 负载尤为关键。

### 2.4 异步任务执行器 (AsyncTaskRunner)

**源码**: `areal/infra/async_task_runner.py`

`AsyncTaskRunner[T]` 是一个通用的异步任务执行器，用于并发处理 rollout 等异步任务：

```python
# async_task_runner.py:71
class AsyncTaskRunner(Generic[T]):
    """Generic asynchronous task runner with queue management and pause/resume control."""
```

**核心特性**:
- 使用 **uvloop** 替代默认 asyncio 事件循环，提供 ~2-4x 性能提升
- 线程安全的输入/输出队列（`queue.Queue`）
- 暂停/恢复控制（`threading.Event`）
- 健康监控（后台线程存活检测）
- 重复任务检测
- 50ms 轮询等待 + 500ms 轮询间隔的可配置参数

**跨线程唤醒机制**:

```python
# 概念: async_task_runner.py 中的信号传递
# 主线程提交任务 → loop.call_soon_threadsafe(input_event.set) → uvloop 线程被唤醒
```

`loop.call_soon_threadsafe()` 是高效的跨线程唤醒原语，避免了轮询带来的延迟。

### 2.5 与 Ray Actor 模型的延迟对比

#### 张量数据传输

| 维度 | HTTP 路径 | Ray 路径 |
|------|----------|---------|
| **序列化** | Base64 + JSON（~33% 膨胀 + bfloat16→float32 转换） | Arrow 零拷贝（同节点） |
| **传输** | HTTP PUT/GET（TCP） | Ray Object Store（共享内存/gRPC） |
| **连接管理** | 每次请求新建连接（`force_close=True`） | 持久 gRPC 通道 |
| **开销估算** | 大张量 ~2-5x 额外延迟 | 接近原始内存拷贝速度 |

#### 方法调用（RPC）

| 维度 | HTTP 路径 | Ray 路径 |
|------|----------|---------|
| **调用方式** | `aiohttp.post("/call", json=payload)` | `actor.call.remote(method, *args)` |
| **序列化** | JSON（orjson） | Ray 内部序列化（cloudpickle + Arrow） |
| **线程模型** | Flask 多线程 + 引擎线程串行化 | Ray Actor 天然单线程 |
| **并发数据端点** | `/data/` 端点可并发 | 所有操作都在 Actor 线程上串行 |

**关键差异**:

Ray Actor 天然单线程的特性意味着所有操作（包括张量广播和引擎方法调用）在同一线程上串行执行（`ray_rpc_server.py`）。HTTP 路径则通过引擎线程仅串行化引擎操作，允许数据端点并发处理。

```
HTTP 路径的并发优势:

Thread-1: POST /data/shard_1 (PUT)  ──→ 直接写入 _storage
Thread-2: POST /data/shard_2 (PUT)  ──→ 直接写入 _storage   ← 并发!
Thread-3: POST /call (train_step)   ──→ 引擎线程排队 → 执行

Ray 路径的序列化:

Actor Thread: call("train_step")    ──→ 执行
              broadcast_tensors()    ──→ 等待上一步完成
              call("get_metrics")    ──→ 等待上一步完成    ← 全部串行
```

**延迟优势总结**:

- **HTTP 路径延迟优势**: 在控制面操作（方法调用、配置推送）和并行数据传输场景下，HTTP 路径的多线程模型提供了更好的并发性。
- **Ray 路径延迟优势**: 在张量密集型传输（大模型权重同步）场景下，Ray Object Store 的零拷贝和共享内存机制显著减少序列化开销。
- **总体**: 对于 AReaL 的核心场景（RL 训练中的周期性权重同步 + 大量小型 RPC 调用），HTTP 路径在灵活性和可调试性上的优势往往超过 Ray 在数据传输上的性能优势。这也是 AReaL 以 HTTP 为默认路径的原因。

---

## 3. 可插拔的 Ray/Slurm 编排底座

### 3.1 双层部署架构：Launcher vs Scheduler

AReaL 有两个独立但互补的编排系统：

| 系统 | 目的 | 生命周期 | 实现 |
|------|------|---------|------|
| **Launcher** | 作业启动（从 CLI 到进程创建） | 从用户命令开始，到所有进程退出结束 | `areal/infra/launcher/{local,slurm,ray}.py` |
| **Scheduler** | 运行时控制面（Worker 管理） | 在训练进程内部，贯穿整个训练过程 | `areal/infra/scheduler/{local,slurm,ray}.py` |

这两个系统**不互斥**——可以混合使用。例如：用 Slurm Launcher 分配节点，然后在 Controller 进程内用 LocalScheduler 管理该节点上的 GPU Worker。

### 3.2 Launcher 系统

**入口**: 通过选择不同的 Python 模块来选择 Launcher（无运行时工厂）：

```bash
python -m areal.infra.launcher.local   examples/gsm8k_grpo.py --config ...
python -m areal.infra.launcher.slurm   examples/gsm8k_grpo.py --config ...
python -m areal.infra.launcher.ray     examples/gsm8k_grpo.py --config ...
```

所有 Launcher 遵循相同的启动流程：

```python
# 以 slurm_main(config, run_id) 为例：
1. parse_cli_args(sys.argv[1:])              # 解析命令行参数
2. to_structured_cfg(config)                  # OmegaConf → 结构化 dataclass
3. validate_config_for_distributed_launcher() # 验证配置
4. name_resolve.reconfigure(config.cluster.name_resolve) # 配置名称解析后端
5. name_resolve.clear_subtree(trial_root())   # 清除上次运行的陈旧条目
6. AllocationMode.from_str(config.allocation_mode) # 解析分配模式
7. submit_array("llm_server", ...)            # 先启动推理服务器
8. wait_llm_server_addrs(...)                 # 等待推理服务器注册地址
9. submit_array("trainer", ...)               # 再启动训练进程
10. wait(...)                                  # 阻塞直到完成
```

三种 Launcher 实现：

| Launcher | 进程管理 | GPU 分配 | 训练启动方式 |
|----------|---------|---------|------------|
| **LocalLauncher** | `subprocess.Popen` | 轮询分配 `CUDA_VISIBLE_DEVICES` | `torchrun --nnodes 1` |
| **SlurmLauncher** | `sbatch` 脚本提交 | Slurm `--gres=gpu:N` | `srun torchrun` |
| **RayLauncher** | `@ray.remote` 任务 | Placement Group (`PACK` 策略) | 直接在 Ray 任务中执行 |

**RayLauncher 的特殊考量** (`areal/infra/launcher/ray.py`):

RayLauncher 使用 `ray.remote` 而非 `ray job submit`，原因是后者不支持 Placement Group，而 Placement Group 是获取节点 IP 以初始化 `torch.distributed` 的必要条件。

```python
# ray.py: Placement Group 策略
# submit_array 为每个逻辑作业创建一个 PlacementGroup
# 使用 "PACK" 策略将任务按节点分组
# bundle_index = i // tasks_per_node
# env_hook 回调查询 PlacementGroup 的实际节点 IP
# 填入 MASTER_ADDR 和 MASTER_PORT
```

### 3.3 Scheduler 抽象层

**源码**: `areal/api/scheduler_api.py`

`Scheduler` ABC 定义了统一的运行时控制面接口：

```python
class Scheduler(abc.ABC):
    # Worker 生命周期管理
    def create_workers(self, job: Job) -> list[str]: ...
    def get_workers(self, role: str, timeout: int | None = None) -> list[Worker]: ...
    def delete_workers(self, role: str | None = None): ...
    def fork_workers(self, role: str, target_role: str, command: str | None = None) -> list[str]: ...

    # 引擎操作（控制面）
    async def create_engine(self, worker_id, engine, engine_name, *args, **kwargs) -> Any: ...
    async def set_worker_env(self, worker_id: str, env: dict[str, str]) -> None: ...

    # 引擎方法调用（数据面）
    def call_engine(self, worker_id, method, engine_name, *args, **kwargs) -> Any: ...
    async def async_call_engine(self, worker_id, method, engine_name, *args, **kwargs) -> Any: ...
```

**关键数据结构**:

```python
@dataclass
class Worker:
    id: str              # "rollout/0", "actor/1"
    ip: str              # Worker 所在 IP
    worker_ports: list[str]  # Worker 通信端口
    engine_ports: list[str]  # 引擎通信端口

@dataclass
class Job:
    role: str            # "rollout", "actor", "ref"
    replicas: int        # 副本数
    tasks: list[SchedulingSpec]      # 每个任务的资源规格
    scheduling_strategy: SchedulingStrategy  # 调度策略
```

**Scheduler 工厂**: 在 `areal/trainer/sft_trainer.py` 中通过简单的 if/elif 链实现：

```python
def _init_scheduler(self) -> Scheduler:
    cfg = self.config.scheduler
    if cfg.type == "local":
        return LocalScheduler(exp_config=self.config)
    elif cfg.type == "ray":
        return RayScheduler(exp_config=self.config)
    elif cfg.type == "slurm":
        return SlurmScheduler(exp_config=self.config)
```

### 3.4 三种 Scheduler 实现

#### LocalScheduler

**源码**: `areal/infra/scheduler/local.py`

```
Controller 进程
│
├── LocalScheduler
│   ├── create_workers("rollout", replicas=2)
│   │   ├── subprocess.Popen("python -m areal.infra.rpc.rpc_server",
│   │   │                     env={CUDA_VISIBLE_DEVICES="0"})
│   │   └── subprocess.Popen("python -m areal.infra.rpc.rpc_server",
│   │                         env={CUDA_VISIBLE_DEVICES="1"})
│   │
│   ├── get_workers("rollout")
│   │   └── HTTP GET /health → 轮询直到所有 Worker 就绪
│   │
│   ├── create_engine("rollout/0", "areal.engine.fsdp_engine.FSDPLMEngine")
│   │   └── HTTP POST /create_engine → Worker 的引擎线程创建引擎
│   │
│   └── call_engine("rollout/0", "generate_sequences", batch)
│       └── HTTP POST /call → Worker 的引擎线程执行方法
```

- GPU 分配: 轮询分配 (`gpu_counter % len(gpu_devices)`)
- 端口分配: `find_free_ports(count, exclude_ports=_allocated_ports)`
- Colocation: 通过 HTTP POST `/fork` 到目标 Worker，由 Worker 进程 fork 子进程

#### SlurmScheduler

**源码**: `areal/infra/scheduler/slurm.py`

```
Controller 进程（运行在某个 Slurm 作业节点上）
│
├── SlurmScheduler
│   ├── create_workers("rollout", replicas=2)
│   │   └── sbatch → Slurm 分配节点 → Worker 进程启动
│   │              → Worker 注册到 name_resolve:
│   │                name_resolve.add("workers/rollout/0", "10.0.1.5:8000")
│   │
│   ├── get_workers("rollout")
│   │   └── name_resolve.get("workers/rollout/0") → 轮询直到发现所有 Worker
│   │   └── HTTP POST /configure → 推送实验配置到 Worker
│   │
│   ├── Colocation:
│   │   └── squeue --nodelist → 获取目标作业节点
│   │   └── sbatch --nodelist=<same_nodes> → 强制共同放置
│   │   或: HTTP POST /fork → 通过 RPC Server fork 子进程
│   │
│   └── call_engine("rollout/0", "generate_sequences", batch)
│       └── HTTP POST /call
```

- 发现机制: Worker 通过 `name_resolve` 注册，Scheduler 轮询发现
- 状态缓存: `_check_job_status` 使用 5 秒 TTL 缓存 Slurm 作业状态
- Colocation: 两种方式——`--nodelist` 强制共同放置，或 `/fork` RPC 端点

#### RayScheduler

**源码**: `areal/infra/scheduler/ray.py`

```
Controller 进程（Ray Driver）
│
├── RayScheduler
│   ├── create_workers("rollout", replicas=2)
│   │   ├── PlacementGroup(bundles=[{"GPU": 0.9}], strategy="PACK")
│   │   ├── RayRPCServer.options(num_gpus=0.9,
│   │   │                        placement_group=pg).remote()
│   │   └── ray.wait([actor.ping.remote()]) → 等待 Actor 就绪
│   │
│   ├── Colocation (fork=True):
│   │   └── RayRPCServer.options(num_gpus=0.01,  ← 微量 GPU 资源
│   │                            placement_group=target_pg).remote()
│   │   注: 0.9 + 0.01 = 0.91，共享同一 GPU
│   │
│   ├── Colocation (fork=False):
│   │   └── 直接复用目标 Actor 句柄，不创建新 Actor
│   │
│   └── call_engine("rollout/0", "generate_sequences", batch)
│       └── actor.call.remote(method, *args) → ray.get(ref, timeout=...)
│           ├── GetTimeoutError → 重试（指数退避）
│           └── RayActorError → 立即抛出（Worker 已死亡）
```

- 放置策略: `DeferredDeviceRayPlacementStrategy` / `SeparatedRayPlacementStrategy` / `SharedRayPlacementStrategy`
- 资源隔离: 每个非共置 Worker 有独立的 Placement Group
- 重试机制: 指数退避 (`retry_delay * (2 ** (attempt - 1))`)，区分瞬态错误和致命错误

### 3.5 名称解析服务 (NameResolve)

**源码**: `areal/utils/name_resolve.py`, `areal/utils/names.py`

名称解析是 AReaL 实现 Worker 发现的核心机制，本质上是一个**分布式 KV 多映射**。

#### 后端选择

```python
# name_resolve.py: make_repository(config)
#   "nfs"   → NfsNameRecordRepository    # 文件系统：{root}/{name}/ENTRY
#   "etcd3" → Etcd3NameRecordRepository  # etcd3 with TTL 续约
#   "ray"   → RayNameResolveRepository   # Ray Actor (@ray.remote)
```

| 后端 | 存储方式 | 适用场景 | 优点 | 缺点 |
|------|---------|---------|------|------|
| NFS | 文件系统 `{root}/{name}/ENTRY` | 默认，无外部依赖 | 简单、可靠 | 依赖共享文件系统 |
| etcd3 | KV 存储 + TTL 租约 | 大规模生产环境 | 高可用、自动过期 | 需要 etcd 集群 |
| Ray | Ray Actor 内存 | Ray 环境 | 与 Ray 集成 | 依赖 Ray 运行时 |

#### 键路径层级

```python
# names.py: 键路径构造
trial_root(exp, trial)
    → "{USER_NAMESPACE}/{exp}/{trial}"

gen_servers(exp, trial)
    → "{USER_NAMESPACE}/{exp}/{trial}/gen_servers"

worker_discovery(exp, trial, role, task_id)
    → "{USER_NAMESPACE}/{exp}/{trial}/workers/{role}/{task_id}"
```

#### 使用模式

1. **推理服务器公告**: 服务器进程启动后注册地址到 `gen_servers(...)` 子树。Launcher 通过 `wait_llm_server_addrs()` 轮询 `name_resolve.get_subtree(name)` 直到期望数量的地址出现（超时 1200s）。

2. **Slurm Worker 发现**: Worker RPC Server 启动时注册到 `worker_discovery(exp, trial, role, task_id)`。Scheduler 轮询此键发现 IP 和端口，然后开始 HTTP 通信。

#### 全局重配置

```python
# 每个 Launcher 启动时调用:
name_resolve.reconfigure(config.cluster.name_resolve)
# 这会重新绑定模块级函数到新后端的方法，
# 所有已有的 name_resolve.get(...) 调用自动使用新后端
```

### 3.6 端到端部署流程

#### SPMD 模式（最常见）: Launcher 驱动

```
用户: python -m areal.infra.launcher.slurm examples/gsm8k_grpo.py --config config.yaml
│
├─ slurm_main(config)
│   ├─ name_resolve.reconfigure(...)           # 配置名称解析后端
│   ├─ name_resolve.clear_subtree(trial_root)  # 清除陈旧状态
│   ├─ AllocationMode.from_str(...)            # 解析分配模式
│   │
│   ├─ [如果 gen_backend == "sglang"]
│   │   └─ SlurmLauncher.submit_array("llm_server", ...)   # sbatch → N 个推理节点
│   │       wait_llm_server_addrs(...)                      # 轮询 name_resolve
│   │
│   ├─ SlurmLauncher.submit_array("trainer", ...)           # sbatch → M 个训练节点
│   │   cmd = "torchrun --nnodes=M --node-rank=$i ..."
│   │   env: BASE_ENVIRONS + AREAL_LLM_SERVER_ADDRS + AREAL_SPMD_MODE=1
│   │
│   └─ SlurmLauncher.wait(...)                              # 阻塞直到完成
│
每个训练节点内部:
│
└─ torchrun → 每 GPU 一个进程
    ├─ torch.distributed.init_process_group()   # via MASTER_ADDR/$head_node_ip
    ├─ AREAL_LLM_SERVER_ADDRS → SGLang HTTP 客户端
    └─ RLVRWorkflow.run()                       # 训练循环
```

#### Single-Controller 模式: Scheduler 驱动

```
Controller 进程 (训练编排循环)
│
├─ _init_scheduler() → 选择 Scheduler 实现
│
├─ scheduler.create_workers(Job(role="rollout", replicas=N))
│   └─ 按 Scheduler 类型: subprocess / sbatch / Ray Actor
│
├─ scheduler.get_workers("rollout")          # 等待健康检查通过
│   └─ 返回 [Worker(id="rollout/0", ip=..., ports=[...]), ...]
│
├─ await scheduler.create_engine("rollout/0", "areal.engine.fsdp_engine.FSDPLMEngine", config)
│   └─ RPC: HTTP POST /create_engine 或 actor.create_engine.remote(...)
│
└─ 训练循环:
    ├─ scheduler.call_engine("rollout/0", "generate_sequences", batch)
    ├─ scheduler.call_engine("actor/0", "compute_loss", batch)
    └─ ...
```

#### 组件透明性验证

Controller 层（`TrainController`, `RolloutController`）**完全通过 `Scheduler` ABC 交互**，不包含任何 Scheduler 特定代码：

```python
# train_controller.py:103-176 (initialize 方法)
self.scheduler.create_workers(job=job)          # 不知道底层是 subprocess/sbatch/Ray
self.scheduler.get_workers(role=job.role)        # 统一返回 Worker 对象
self.scheduler.create_engine(...)                # 统一的引擎创建接口
self.scheduler.async_call_engine(...)            # 统一的异步方法调用
```

这意味着切换编排底座只需修改 `SchedulerConfig.type` 配置，Controller 代码无需任何变更。

---

## 4. 容错与优雅关闭

### 4.1 重试机制

**HTTP 层**: `arequest_with_retry` 提供可配置的重试逻辑。

**Scheduler 层**: 各 Scheduler 实现各自的重试策略：

```python
# Ray Scheduler (ray.py:663-699):
# - GetTimeoutError → 指数退避重试 (delay * 2^attempt)
# - RayActorError → 立即抛出（Worker 已死亡，重试无意义）

# Local/Slurm Scheduler:
# - 委托给 arequest_with_retry 处理 HTTP 错误
```

### 4.2 健康检查

```
启动阶段:
├── LocalScheduler: HTTP GET /health → 轮询直到返回 200
├── SlurmScheduler: name_resolve.get() → 轮询直到地址出现 → HTTP GET /health
└── RayScheduler: ray.wait([actor.ping.remote()]) → 等待 Actor 响应
```

**注意**: 初始化完成后没有持续的健康检查。Worker 死亡只在下次 RPC 调用失败时被发现。

### 4.3 优雅关闭

#### RPC Server 关闭顺序 (`rpc_server.py:1047-1051`)

```python
finally:
    perf_tracer.save(force=True)
    cleanup_forked_children()  # 1. 终止所有派生子进程
    cleanup_engine_thread()    # 2. 停止引擎线程（发送 None 哨兵）
    cleanup_engines()          # 3. 销毁所有引擎实例
    server.shutdown()          # 4. 关闭 Flask 服务器
```

#### TrainController 关闭 (`train_controller.py:256-298`)

```python
# 1. 销毁引擎（释放 GPU 内存）
await asyncio.gather(*[destroy_engine(w) for w in workers], return_exceptions=True)
# 2. 删除 Worker 进程
self.scheduler.delete_workers(self._worker_role)
# 3. 销毁进程组
dist.destroy_process_group(group)
```

`return_exceptions=True` 确保一个引擎销毁失败不会阻止其他引擎的清理。

#### RolloutController 关闭 (`rollout_controller.py:301-333`)

按顺序关闭: 调度器 → 回调服务器 → Worker 引擎 → Worker 进程 → 代理基础设施

### 4.4 Staleness-aware 容量管理

**源码**: `areal/infra/staleness_manager.py`

`StalenessManager` 实现双重约束容量模型，用于防止异步 RL 训练中的 off-policy 漂移：

```python
# staleness_manager.py:97-111
def get_capacity(self) -> int:
    with self.lock:
        current_version = self.version_provider.get_version()
        # 并发约束
        concurrency_capacity = max_concurrent_rollouts - running

        # 陈旧度约束
        staleness_capacity = (max_staleness + current_version + 1) * consumer_bs - sample_cnt

        # 取两者最小值
        return min(concurrency_capacity, staleness_capacity)
```

**数学原理**: 假设当前模型版本为 $v$，消费者批大小为 $B$，最大允许陈旧度为 $S$。已接受/运行中的样本总数为 $n$。则可接受的最大样本数为 $(S + v + 1) \times B$，剩余容量为 $(S + v + 1) \times B - n$。这确保了当样本被消费时，它们的版本差不会超过 $S$。

---

## 5. 并发模式总结

| 模式 | 使用位置 | 目的 |
|------|---------|------|
| **单线程引擎队列** | `rpc_server.py` | NCCL 线程安全 |
| **asyncio + uvloop** | `async_task_runner.py` | 高性能异步任务执行 |
| **线程安全队列** | `AsyncTaskRunner` 输入/输出 | 跨线程任务提交和结果收集 |
| **ThreadPoolExecutor** | `concurrent.py` | sync→async 桥接 |
| **threading.Lock** | `StalenessManager`, RTensor `_storage` | 共享状态保护 |
| **threading.Event** | `AsyncTaskRunner` 暂停/恢复 | 流控制 |
| **WeakKeyDictionary** | `concurrent.py` 事件循环注册 | GC 安全的生命周期管理 |
| **asyncio.gather()** | `TrainController` | 并发多 Worker 操作 |
| **Ray Actor 单线程** | `ray_rpc_server.py` | 天然串行化保证 |
| **Placement Group** | `RayScheduler` | GPU 资源隔离与共置 |
| **Daemon Thread** | 引擎线程、回调服务器、后台事件循环 | 生命周期跟随主进程 |

---

## 6. 代码质量发现

### Critical 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 1 | `scheduler/ray.py` | 580 | `fork_workers` 调用 `_create_forked_workers`（不存在），应为 `_create_forked_workers_internal`。Ray 后端的 fork 流程运行时必崩 |
| 2 | `rpc/rpc_server.py` | 98-112 | 引擎线程外层 `except` 引用可能未绑定的 `func_name`。若 `Queue.get()` 返回异常项，引擎线程将因 `UnboundLocalError` 静默死亡 |

### High 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 3 | `rpc/rpc_server.py` | 57, 163-165 | `_allocated_ports`（普通 `set`）在多 Flask 线程间无同步访问 |
| 4 | `utils/http.py` | 43-112 | `aiohttp.ClientSession` 在未捕获异常时泄漏（`ValueError` 等不在 except 范围内） |
| 5 | `utils/concurrent.py` | 215-230 | `_patched_loop_close` 中若回调抛异常，`_cleanup_orig_close()` 可能永远不被调用 |
| 6 | `scheduler/ray.py` | 586-610 | `_cleanup_workers` 中 `actor.destroy.remote()` 是 fire-and-forget，但随即删除 Placement Group。Actor 可能在销毁引擎中途被杀 |

### Moderate 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 7 | `rpc/rpc_server.py` | 1047-1051 | 关闭顺序：先停引擎线程再销毁引擎，但引擎销毁可能需要 NCCL 操作 |
| 8 | `staleness_manager.py` | 97-111 | 锁持有期间调用外部 `version_provider.get_version()`，若其涉及 I/O 则可能成为瓶颈 |
| 9 | `async_task_runner.py` | 358-365 | 任务异常被静默转换为 `None`，违反 `AsyncTaskRunner[T]` 的类型契约 |
| 10 | `scheduler_api.py` | 77 | ABC 中 `timeout: int` vs 实现中 `timeout: float`，类型不一致 |
| 11 | `rpc/rtensor.py` | 151-158 | `get_backend()` 非线程安全，两个线程可能竞争创建后端实例 |
| 12 | `utils/concurrent.py` | 56-69 | `run_async_task` 在非异步上下文中每次调用创建/销毁事件循环 |

---

## 7. 设计权衡总结

### 为什么以 HTTP RPC 为主而非纯 Ray？

| 维度 | HTTP RPC | Ray Actor |
|------|----------|-----------|
| **部署依赖** | 仅需 Python + Flask | 需要 Ray 集群 |
| **调试性** | curl 即可测试端点 | 需要 Ray Dashboard |
| **灵活性** | 可与任意集群管理器配合 | 绑定 Ray 生态 |
| **张量传输效率** | 较低（Base64 + JSON） | 高（零拷贝共享内存） |
| **并发模型** | 多线程 + 引擎线程隔离 | 单线程 Actor |
| **容错** | 需要自行实现 | Ray 提供 Actor 重启 |

**核心取舍**: AReaL 选择以 HTTP 为默认路径，牺牲张量传输效率，换取部署灵活性和可调试性。在 RL 训练场景中，大部分时间花在 GPU 计算（前向/反向传播）和 NCCL 集合通信上，HTTP RPC 的开销相对较小。真正的数据密集传输（模型权重同步）使用 NCCL 而非 HTTP。

### 可插拔设计的实际效果

```
┌──────────────────────────────────────────────────┐
│               Controller 层                       │
│  TrainController / RolloutController              │
│  ↕ 仅通过 Scheduler ABC 交互                      │
├──────────────────────────────────────────────────┤
│ ┌─────────────┬─────────────┬──────────────────┐ │
│ │   Local     │   Slurm     │     Ray          │ │
│ │ subprocess  │  sbatch     │  Ray Actor       │ │
│ │ HTTP RPC    │  HTTP RPC   │  Ray RPC         │ │
│ │ 本地端口    │  name_resolve│  Placement Group │ │
│ └─────────────┴─────────────┴──────────────────┘ │
│          Scheduler 实现层                         │
├──────────────────────────────────────────────────┤
│ ┌─────────────┬─────────────┬──────────────────┐ │
│ │   NFS       │   etcd3     │     Ray KV       │ │
│ └─────────────┴─────────────┴──────────────────┘ │
│          NameResolve 后端层                        │
└──────────────────────────────────────────────────┘
```

三层解耦确保了：
1. **Controller 对 Scheduler 透明**: 切换 `SchedulerConfig.type` 即可
2. **Scheduler 对 NameResolve 透明**: 通过 `reconfigure()` 动态切换后端
3. **Launcher 独立于 Scheduler**: 可以混合使用（如 Slurm Launcher + Local Scheduler）

这种设计使得 AReaL 能够在开发环境（Local）、HPC 集群（Slurm）和云环境（Ray）之间无缝切换，同时保持核心训练逻辑不变。
