# 启动器与调度器层

> 源码位置：`areal/infra/launcher/`, `areal/infra/scheduler/` 文件数：11 个 | 总行数：6999 行

## 1. 模块定位

启动器与调度器层是 AReaL 分布式部署的基础设施核心，负责将训练任务和推理服务从"单机配置"映射到"集群可执行"。该层分为两个子系统：

- **Launcher（启动器）**：面向 SPMD（单程序多数据）模式的遗留启动入口，直接拉起训练进程和推理服务器，随后阻塞等待直到所有作业结束或失败。Launcher
  正处于废弃迁移期（见 `FutureWarning` 于 `local.py:274`、`ray.py:352`、`slurm.py:415`），新代码应使用
  Scheduler。
- **Scheduler（调度器）**：面向 Single-Controller 模式的现代部署接口，实现了
  `areal.api.scheduler_api.Scheduler` 抽象基类（`scheduler_api.py:43`），提供 Worker
  生命周期管理、Engine 远程创建与 RPC 方法调用的统一 API。

```
+--------------------------------------------------------------------+
|                          用户 / Workflow                             |
+--------------------------------------------------------------------+
        |                                          |
        v                                          v
+------------------+                    +---------------------+
|    Launcher      |                    |     Scheduler       |
|  (SPMD 遗留模式)  |                    |  (Single-Controller) |
|  local_main()    |                    |  create_workers()   |
|  ray_main()      |                    |  get_workers()      |
|  slurm_main()    |                    |  create_engine()    |
+------------------+                    |  call_engine()      |
        |                               |  delete_workers()   |
        v                               +---------------------+
+------------------+                            |
| 推理服务器 Wrapper |                            v
| SGLangWrapper    |                    +---------------------+
| vLLMWrapper      |                    |   RPC Server        |
+------------------+                    |  (HTTP / Ray Actor) |
        |                               +---------------------+
        v                                       |
+--------------------------------------------------------------------+
|               底层资源：GPU / 端口 / 容器 / Slurm                     |
+--------------------------------------------------------------------+
```

## 2. 文件清单与行数

| 文件路径                    | 行数 | 核心职责                                 |
| --------------------------- | ---- | ---------------------------------------- |
| `launcher/__init__.py`      | 22   | 导出符号表                               |
| `launcher/local.py`         | 444  | LocalLauncher + `local_main()` SPMD 入口 |
| `launcher/ray.py`           | 648  | RayLauncher + `ray_main()` SPMD 入口     |
| `launcher/slurm.py`         | 697  | SlurmLauncher + `slurm_main()` SPMD 入口 |
| `launcher/sglang_server.py` | 269  | SGLangServerWrapper 推理服务管理         |
| `launcher/vllm_server.py`   | 303  | vLLMServerWrapper 推理服务管理           |
| `scheduler/__init__.py`     | 11   | 导出符号表                               |
| `scheduler/exceptions.py`   | 128  | 11 种调度器异常类型                      |
| `scheduler/local.py`        | 1791 | LocalScheduler 单节点调度器              |
| `scheduler/ray.py`          | 872  | RayScheduler Ray 集群调度器              |
| `scheduler/slurm.py`        | 1814 | SlurmScheduler Slurm 集群调度器          |

## 3. Launcher vs Scheduler 职责划分

### 3.1 设计哲学差异

| 维度        | Launcher（启动器）                  | Scheduler（调度器）                         |
| ----------- | ----------------------------------- | ------------------------------------------- |
| 编程模型    | SPMD：一个入口函数拉起所有进程      | Single-Controller：外部编排器调用 API       |
| 进程粒度    | 直接管理子进程/Ray Future/Slurm Job | 管理 Worker 抽象，Worker 内运行 RPC Server  |
| Engine 创建 | 由 Trainer 入口函数自行创建         | 由 Scheduler 通过 HTTP/Ray Actor 远程创建   |
| 方法调用    | 进程内直接调用                      | `call_engine()` / `async_call_engine()` RPC |
| GPU 分配    | 启动时一次性分配（round-robin）     | 按 Job 动态分配，支持 Colocation/Fork       |
| 迁移状态    | 标记为 `FutureWarning`，将被移除    | 当前推荐方式                                |

### 3.2 公共 API 对比

Launcher 没有统一基类，各实现直接提供 `submit/submit_array/stop/wait` 方法。Scheduler 实现 `Scheduler`
抽象基类（`scheduler_api.py:43`），强制要求以下接口：

```
Scheduler (ABC)
  |-- n_gpus_per_node      # property
  |-- create_workers(job)  # 创建 Worker 进程
  |-- get_workers(role)    # 等待 Worker 就绪
  |-- delete_workers(role) # 删除 Worker 并清理
  |-- fork_workers(role, target_role)  # Fork Worker
  |-- create_engine(worker_id, engine) # 远程创建 Engine
  |-- set_worker_env(worker_id, env)   # 设置环境变量
  |-- call_engine(worker_id, method)   # 同步 RPC 调用
  |-- async_call_engine(worker_id, method) # 异步 RPC 调用
```

## 4. 三种部署模式的差异

### 4.1 Local 模式

```
LocalLauncher (local.py:87)          LocalScheduler (scheduler/local.py:92)
  |                                    |
  |-- 进程模型: subprocess.Popen       |-- 进程模型: subprocess.Popen + RPC Server
  |-- GPU 分配: round-robin            |-- GPU 分配: round-robin (_allocate_gpus L203)
  |   _gpu_counter (L97)               |-- 端口分配: find_free_ports (_allocate_ports L234)
  |-- 作业跟踪: dict[str, Popen]       |-- Worker 跟踪: dict[str, list[WorkerInfo]]
  |-- 日志: tee -a 到文件              |-- 日志: run_with_streaming_logs
  |-- 等待: psutil 轮询进程状态        |-- 健康检查: HTTP /health 端点轮询
  +-- 恢复: RECOVER_TIME_INTERVAL=10s  +-- Fork: /fork HTTP 端点创建子进程
```

**GPU 分配策略（Local）**：

```python
# LocalLauncher (local.py:136-144) - round-robin 分配
for _ in range(gpu):
    available_device_id = self._gpu_counter % len(self._gpu_devices)
    self._gpu_counter += 1
    visible_devices.append(available_device_id)
env[CUDA_VISIBLE_DEVICES] = ",".join(visible_devices)

# LocalScheduler (scheduler/local.py:203-218) - 相同策略
def _allocate_gpus(self, num_gpus):
    for _ in range(num_gpus):
        gpu_id = self.gpu_devices[self._gpu_counter % len(self.gpu_devices)]
        allocated.append(gpu_id)
        self._gpu_counter += 1
```

### 4.2 Ray 模式

```
RayLauncher (ray.py:80)              RayScheduler (scheduler/ray.py:58)
  |                                    |
  |-- 进程模型: ray.remote() Future    |-- 进程模型: RayRPCServer Actor
  |-- GPU 分配: PlacementGroup         |-- GPU 分配: PlacementGroup + Strategy
  |   bundles=[{GPU:N}]*nodes          |   (Deferred/Separated/Shared)
  |-- 作业跟踪: dict[str, Future]      |-- Worker 跟踪: dict[str, list[RayWorkerInfo]]
  |-- 等待: ray.get(timeout=0.1) 轮询  |-- 健康检查: actor.ping.remote()
  |-- NPU 支持: resources={"NPU": N}  |-- Fork: 共享 PlacementGroup, GPU=0.01
  +-- 全局单例: RAY_LAUNCHER           +-- 销毁: 三阶段 (destroy -> wait -> remove PG)
```

**PlacementGroup 策略（Ray Scheduler）**：

Ray Scheduler 支持三种放置策略（`scheduler/ray.py:145-165`）：

- `DeferredDeviceRayPlacementStrategy`：延迟设备绑定
- `SeparatedRayPlacementStrategy`：每个 Worker 独占 GPU 资源
- `SharedRayPlacementStrategy`：多个 Worker 共享 GPU 资源

Fork Worker 时使用 `num_gpus=0.01`（`scheduler/ray.py:283`），与主 Worker 的 `num_gpus=0.9` 共享同一
PlacementGroup。

### 4.3 Slurm 模式

```
SlurmLauncher (slurm.py:52)          SlurmScheduler (scheduler/slurm.py:72)
  |                                    |
  |-- 进程模型: sbatch + srun          |-- 进程模型: sbatch + srun + RPC Server
  |-- GPU 分配: --gres=gpu:N           |-- GPU 分配: --gres=gpu:N + CUDA_VISIBLE_DEVICES
  |-- 作业跟踪: dict[int, JobInfo]     |-- Worker 发现: name_resolve 注册/查询
  |   (slurm_job_id -> JobInfo)        |   (_discover_worker_network L349)
  |-- 容器支持: apptainer / none       |-- 容器支持: apptainer / native
  |-- 等待: squeue 轮询作业状态        |-- 健康检查: HTTP + squeue 双重检查
  |-- 跨节点: torchrun nnodes>1        |-- 销毁: 两阶段 (engine.destroy -> scancel)
  +-- SBATCH 脚本生成: 模板填充        +-- SBATCH 脚本生成: 动态构建
```

**Slurm 作业状态查询与缓存（Slurm Scheduler）**：

```python
# scheduler/slurm.py:216-258 - 带 TTL 缓存的状态查询
_job_status_cache: dict[int, tuple[JobState, float]]
_status_cache_ttl = 5.0  # 5 秒缓存

def _check_job_status(self, role):
    # 先检查缓存
    if cached_time < self._status_cache_ttl:
        return cached_state
    # 缓存过期则调用 squeue
    job_infos = query_jobs(slurm_ids=[job_id])
```

### 4.4 三模式横向对比

| 维度        | Local                   | Ray                     | Slurm                          |
| ----------- | ----------------------- | ----------------------- | ------------------------------ |
| 进程启动    | `subprocess.Popen`      | `ray.remote()`          | `sbatch`                       |
| 进程终止    | `SIGKILL` 递归子树      | `ray.cancel(force)`     | `scancel --signal`             |
| GPU 隔离    | `CUDA_VISIBLE_DEVICES`  | PlacementGroup bundle   | `--gres=gpu` + `SLURM_LOCALID` |
| 多节点      | 不支持（单节点）        | 自动跨节点              | torchrun `--nnodes`            |
| Worker 发现 | 本地直连                | PlacementGroup IP       | `name_resolve` NFS/etcd3       |
| 日志管理    | `tee -a`                | Ray 标准输出            | srun + `tee` + merged.log      |
| 端口管理    | `find_free_ports`       | PlacementGroup 内分配   | RPC Server 自动分配            |
| 恢复重试    | `RECOVER_TIME_INTERVAL` | `RECOVER_TIME_INTERVAL` | `RECOVER_TIME_INTERVAL`        |

## 5. Worker 生命周期

### 5.1 创建流程

```
create_workers(Job)
  |
  |-- [1] 解析调度策略
  |   |-- SchedulingStrategyType.separation -> 独立创建
  |   +-- SchedulingStrategyType.colocation -> 复用/Fork
  |
  |-- [2] 资源分配
  |   |-- GPU: _allocate_gpus() / PlacementGroup / --gres
  |   |-- 端口: _allocate_ports() / alloc_ports RPC
  |   +-- 环境: get_env_vars() + get_thread_env_vars() + get_tms_env_vars()
  |
  |-- [3] 进程启动
  |   |-- Local:  subprocess.Popen -> RPC Server
  |   |-- Ray:    RayRPCServer.options().remote()
  |   +-- Slurm:  sbatch -> srun -> RPC Server
  |
  |-- [4] 就绪等待
  |   |-- Local:  HTTP /health 轮询 (startup_timeout=30s)
  |   |-- Ray:    actor.ping.remote() (startup_timeout=30s)
  |   +-- Slurm:  name_resolve 发现 + /health (startup_timeout=300s)
  |
  +-- [5] 配置推送
      +-- HTTP POST /configure (exp_config, role, rank)
```

### 5.2 Fork 流程

Fork 是 Scheduler 的关键特性，允许在已有 Worker 的同一 GPU 上创建新的 Worker 进程（例如为 Proxy/Critic 复用 Actor 的
GPU）。

```
fork_workers(role, target_role)
  |
  |-- [1] 获取目标 Worker 列表
  |
  |-- [2] 并发异步 Fork (aiohttp / Ray Actor)
  |   |-- Local/Slurm: POST /alloc_ports -> POST /fork -> 等待 /health
  |   +-- Ray: RayRPCServer.options(num_gpus=0.01, same PG)
  |
  |-- [3] 注册为 colocated_roles[role] = target_role
  |
  +-- [4] 失败回滚: 清理已成功的 Fork Worker
```

### 5.3 删除流程

删除遵循两/三阶段协议，确保 NCCL/TCPStore 无竞争关闭：

```
delete_workers(role, reverse_order=True)
  |
  +-- Local Scheduler (3 阶段):
  |   |-- Phase 1: 释放端口（同步，无 I/O）
  |   |-- Phase 2: 并发 SIGTERM（多线程同时发送）
  |   +-- Phase 3: 等待清理线程完成（join_timeout=10s）
  |
  +-- Ray Scheduler (3 阶段):
  |   |-- Phase 1: 并发 actor.destroy.remote()
  |   |-- Phase 2: ray.wait() 等待 barrier 完成（timeout=30s）
  |   +-- Phase 3: remove_placement_group()
  |
  +-- Slurm Scheduler (2 阶段):
      |-- Phase 1: HTTP POST /call destroy 引擎（并发）
      +-- Phase 2: scancel SIGTERM -> 检查 -> SIGKILL
```

**为何需要多阶段协议**（`scheduler/local.py:1145-1176` 注释详解）：

旧实现串行地对每个 Worker 调用 `kill_process_tree(timeout=3, graceful=True)`，4-rank 作业需要约 12
秒。在此窗口期只有一个 rank 在执行 `engine.destroy()`，导致 `FSDPEngine.destroy()` 中的 CPU barrier 无法同步，产生
`TCPStore.recvValue failed` 错误。新实现改为：先并发发送 SIGTERM，让所有 rank 同时进入 destroy 路径，再统一等待。

## 6. 推理服务器（SGLang/vLLM）的启动与健康检查

### 6.1 SGLangServerWrapper (`sglang_server.py:89`)

```
SGLangServerWrapper.run()
  |
  |-- [1] 计算拓扑
  |   |-- gpus_per_server = allocation_mode.gen_instance_size
  |   |-- cross_nodes = (gpus_per_server > n_gpus_per_node)
  |   +-- n_servers_per_node = n_gpus_per_node // gpus_per_server
  |
  |-- [2] 端口分配
  |   |-- 端口范围: (10000, 32767)
  |   |-- 每服务器独占: ports_per_server = 22767 // n_servers_per_node
  |   +-- find_free_ports(2, port_range) -> (server_port, dist_init_port)
  |
  |-- [3] 并发启动 (ThreadPoolExecutor)
  |   |-- launch_server_cmd() -> subprocess.Popen
  |   |-- 设置唯一 TRITON_CACHE_PATH 避免冲突
  |   +-- 每服务器独立随机种子: base_seed + server_local_idx
  |
  |-- [4] 健康检查
  |   +-- wait_for_server(): GET /v1/models -> 200 OK
  |       (轮询间隔 1s，成功后额外等待 5s)
  |
  |-- [5] 服务注册
  |   +-- name_resolve.add_subentry(name, "host:port")
  |       (仅 node_rank=0 时注册)
  |
  +-- [6] 进程监控
      +-- _monitor_server_processes(): 1s 轮询 process.poll()
          任一服务器退出则 sys.exit(1)
```

### 6.2 vLLMServerWrapper (`vllm_server.py:88`)

```
vLLMServerWrapper.run()
  |
  |-- [1] 计算拓扑（不支持跨节点）
  |   +-- gpus_per_server > n_gpus_per_node -> NotImplementedError
  |
  |-- [2] 端口分配（同 SGLang）
  |
  |-- [3] 并发启动 (ThreadPoolExecutor)
  |   |-- 设置唯一 TRITON_CACHE_PATH
  |   |-- 设置唯一 VLLM_CACHE_ROOT 避免编译缓存冲突
  |   +-- VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
  |
  |-- [4] 信号处理 (vllm_server.py:109-123)
  |   |-- SIGTERM/SIGINT -> _handle_shutdown_signal()
  |   +-- _cleanup_all_servers() -> kill_process_tree(graceful=True)
  |       (防止重入: _is_shutting_down 标志)
  |
  +-- [5] 进程监控
      +-- while not _shutdown_requested: 1s 轮询
          任一服务器退出 -> _cleanup_all_servers() -> sys.exit(1)
```

### 6.3 SGLang vs vLLM 对比

| 维度         | SGLang                              | vLLM                                      |
| ------------ | ----------------------------------- | ----------------------------------------- |
| 跨节点       | 支持（`AREAL_SGLANG_MULTI_NODE_*`） | 不支持                                    |
| 信号处理     | 依赖 `finally: kill_process_tree()` | 注册 SIGTERM/SIGINT handler + 防重入      |
| 缓存隔离     | TRITON_CACHE_PATH                   | TRITON_CACHE_PATH + VLLM_CACHE_ROOT       |
| LoRA         | -                                   | `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`   |
| GPU 设备控制 | 由 SGLang 参数 `base_gpu_id` 控制   | 由 `CUDA_VISIBLE_DEVICES` 环境变量控制    |
| 服务注册     | 仅 `node_rank=0` 注册               | 每个服务器进程都注册                      |
| 退出策略     | `sys.exit(1)` 直接退出              | `_cleanup_all_servers()` 后 `sys.exit(1)` |

## 7. 故障恢复机制

### 7.1 Launcher 层恢复

三种 Launcher 共享相同的恢复模式（`RECOVER_TIME_INTERVAL = 10` 秒）：

```python
# 恢复流程模板 (local.py:423-440, ray.py:616-642, slurm.py:666-691)
try:
    launcher.wait(check_status=(...))
except (KeyboardInterrupt, JobException, TimeoutError) as e:
    launcher.stop_all(...)
    if isinstance(e, JobException):
        recover_this = (
            e.reason in recover_states          # 可恢复状态
            and run_id < config.recover.retries  # 未超最大重试
            and config.recover.mode in ("on", "auto")  # 恢复模式启用
        )
        if recover_this:
            time.sleep(RECOVER_TIME_INTERVAL)    # 等待 10 秒
            xxx_main(config, run_id=run_id + 1)  # 递归重试
```

各模式的可恢复状态差异：

| 模式  | 可恢复状态                   | 原因                                          |
| ----- | ---------------------------- | --------------------------------------------- |
| Local | FAILED, NOT_FOUND, COMPLETED | 无法区分成功完成和失败（`local.py:427` 注释） |
| Ray   | FAILED                       | 可精确区分失败和完成                          |
| Slurm | FAILED, NOT_FOUND            | 作业可能从队列中消失                          |

### 7.2 Scheduler 层异常体系

调度器通过 `exceptions.py`（128 行）定义了 11 种结构化异常类型：

```
SchedulerError (基类)
  |
  |-- WorkerCreationError    # Worker 创建失败（含 worker_key, reason, details）
  |-- WorkerConfigurationError # Worker 配置失败
  |-- WorkerFailedError      # Worker 进程退出（含 exit_code, stderr）
  |-- WorkerNotFoundError    # Worker 不存在
  |-- WorkerTimeoutError     # 等待 Worker 超时（含 timeout 值）
  |
  |-- EngineCreationError    # Engine 创建失败（含 HTTP status_code）
  |-- EngineCallError        # Engine 方法调用失败（含 method, attempt）
  |-- EngineImportError      # Engine 类导入失败（含 import_path）
  |
  |-- PortAllocationError    # 端口分配失败
  |-- GPUAllocationError     # GPU 资源不足
  +-- RPCConnectionError     # RPC 连接失败（含 host, port）
```

### 7.3 端口冲突重试

LocalScheduler 在 Worker 启动时支持端口冲突重试（`scheduler/local.py:59`）：

```python
_MAX_STARTUP_PORT_CONFLICT_RETRIES = 3

# scheduler/local.py:799-873
for attempt in range(1, _MAX_STARTUP_PORT_CONFLICT_RETRIES + 1):
    ports = self._allocate_ports(scheduling.port_count)
    process = run_with_streaming_logs(cmd, ...)
    if process.poll() is None:  # 进程存活
        break
    stderr = self._read_log_tail(log_file)
    if self._is_port_conflict_error(stderr):  # 检测端口冲突
        time.sleep(0.1 * attempt)  # 指数退避
        continue
```

端口冲突检测（`scheduler/local.py:248-255`）匹配以下模式：

- `"address already in use"`
- `"errno 98"` / `"errno 48"`
- `"port ... is in use by another program"`

## 8. 关键设计决策与约束

### 8.1 SPMD 到 Single-Controller 迁移

所有 Launcher 文件中均包含 `FutureWarning`（如 `local.py:274-283`）：

> SPMD launchers use the deprecated `_AllocationMode` parser which will be removed. Bare
> dimension strings (e.g., 'd4t2') are NO LONGER ACCEPTED. Migrate to single-controller
> mode (scheduler.type=local) with per-engine 'backend' fields.

`_AllocationMode.from_str()` 解析如 `"fsdp:d4"` 和 `"sglang:d4t2"` 的分配字符串，但这种扁平化的资源描述正被
Scheduler 的 `Job` + `SchedulingSpec` + `SchedulingStrategy` 所取代。

### 8.2 Ray Launcher 的全局单例

```python
# ray.py:49 - 全局变量，用于恢复运行时复用 PlacementGroup
RAY_LAUNCHER = None

# ray.py:375-385
if RAY_LAUNCHER is None:
    assert run_id == 0
    launcher = RayLauncher(...)
    RAY_LAUNCHER = launcher
else:
    launcher = RAY_LAUNCHER  # 恢复时复用
```

这避免了恢复重试时重新创建 PlacementGroup 的开销，但也意味着一个 Python 进程中只能有一个 RayLauncher 实例。

### 8.3 Colocation 的两种模式

Scheduler 的 `create_workers()` 在 `SchedulingStrategyType.colocation` 下支持两种行为：

1. **共享模式**（`strategy.fork = False`）：直接复用目标 Role 的 Worker，不创建新进程。适用于 Actor 和 Reference
   共享同一个 FSDPEngine 的场景。

1. **Fork 模式**（`strategy.fork = True`）：在目标 Worker 的同一 GPU 上 Fork 新进程。适用于需要独立进程但共享 GPU
   的场景（如 Proxy Server）。

### 8.4 Slurm Worker 发现机制

Slurm 模式下 Worker 的 IP 和端口在 sbatch 提交时未知，需要异步发现：

```
sbatch 提交 -> srun 启动 RPC Server -> Server 自注册到 name_resolve
                                                |
Scheduler._discover_worker_network() <---------+
  |-- name_resolve.get(worker_discovery_key)
  |-- 解析 ip:port
  |-- 标记 worker_info.discovered = True
  +-- alloc_ports 分配额外端口
```

### 8.5 线程环境变量管理

所有三种部署模式都通过 `get_thread_env_vars()` 统一管理线程相关的环境变量，确保 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`
等设置与分配的 CPU 数量一致，避免多进程间的 CPU 资源竞争。

### 8.6 Slurm 容器化支持

SlurmLauncher 和 SlurmScheduler 支持两种容器类型：

- `apptainer`：通过 `singularity exec --nv` 运行，环境变量通过 `--env` 传递
- `none`（native）：直接在宿主环境运行，环境变量通过 `srun --export` 传递
