# 基础设施平台与工具层

> 源码位置：`areal/infra/platforms/`, `areal/infra/utils/`, `areal/infra/sandbox/` 文件数：18 个 |
> 总行数：2590 行 最后更新：2026-06-13

______________________________________________________________________

## 1. 模块职责概述

本层为 AReaL 分布式 RL 训练框架提供三类基础能力：

1. **平台抽象层** (`platforms/`)：屏蔽 NVIDIA CUDA、华为 NPU、CPU 等不同硬件的差异，
   为上层引擎（FSDPEngine、ArchonEngine、MegatronEngine）和调度器（Ray/Slurm/Local）
   提供统一的设备元数据、内存管理、环境变量控制接口。全局单例 `current_platform` 以惰性 代理模式延迟初始化，避免在 import 时触发 CUDA。

1. **基础设施工具** (`utils/`)：覆盖六个子领域——

   - HTTP 客户端与重试策略（aiohttp / httpx）
   - 异步并发与事件循环清理
   - 进程树管理与流式日志
   - Ray 放置组策略（Shared / Separated / DeferredDevice）
   - Slurm 作业脚本生成、查询、取消与节点解析
   - 启动器环境变量与配置校验

1. **沙箱集成** (`sandbox/`)：为 rollout 中的代码执行提供 Daytona 云沙箱后端， 包含异步客户端管理器和同步 Runner
   封装，支持基于镜像或快照的沙箱创建。

```
+------------------------------------------------------------------+
|                    上层调用方                                      |
|  FSDPEngine / ArchonEngine / MegatronEngine / Scheduler / ...    |
+------------------------------------------------------------------+
        |                  |                    |
        v                  v                    v
+----------------+  +----------------+  +------------------+
|  platforms/    |  |  utils/        |  |  sandbox/        |
|  Platform      |  |  http          |  |  DaytonaClient   |
|  CudaPlatform  |  |  concurrent    |  |  DaytonaRunner   |
|  CpuPlatform   |  |  proc          |  |  DaytonaRunResult|
|  NPUPlatform   |  |  launcher      |  +------------------+
|  UnknownPlat.  |  |  ray*.py       |
+----------------+  |  slurm         |
                    |  exp_metadata  |
                    +----------------+
```

______________________________________________________________________

## 2. 文件清单

### 2.1 platforms/ — 平台抽象层（6 文件，500 行）

| 文件          | 行数 | 职责                                                                                     |
| ------------- | ---- | ---------------------------------------------------------------------------------------- |
| `platform.py` | 149  | 平台基类 `Platform`：定义 7 个类属性（device_name/type/dispatch_key 等）和 9 个接口方法  |
| `cuda.py`     | 101  | `CudaPlatform`：NVIDIA GPU 实现，NUMA 亲和、cuBLAS 工作空间清理、vLLM Worker 加载        |
| `cpu.py`      | 56   | `CpuPlatform`：纯 CPU 回退，所有设备操作为 no-op                                         |
| `npu.py`      | 30   | `NPUPlatform`：华为昇腾 NPU，HCCL 后端                                                   |
| `unknown.py`  | 66   | `UnknownPlatform`：无法识别的 CUDA 设备，回退到基本 CUDA 操作                            |
| `__init__.py` | 98   | 自动检测逻辑 `_init_platform()` + 惰性代理 `_LazyPlatform` + 全局单例 `current_platform` |

### 2.2 utils/ — 基础设施工具（8 文件，1309 行）

| 文件                     | 行数 | 职责                                                                                  |
| ------------------------ | ---- | ------------------------------------------------------------------------------------- |
| `http.py`                | 238  | HTTP 客户端工具：httpx/aiohttp 连接池、重试装饰器、管理员 API Key 校验、请求函数      |
| `concurrent.py`          | 259  | 线程池管理、同步/异步桥接、事件循环清理回调注册（asyncio-atexit 模式）                |
| `launcher.py`            | 276  | 启动器工具：环境变量组装、JobState/JobInfo/JobException、配置校验、LLM 服务器地址等待 |
| `ray_placement_group.py` | 236  | Ray 放置组策略：3 种策略类 + bundle 规格计算 + 资源类型自动检测                       |
| `proc.py`                | 229  | 进程管理：流式日志命令构建、`kill_process_tree` 进程树清理                            |
| `slurm.py`               | 231  | Slurm 集成：sbatch/srun 脚本模板、作业查询/取消、节点列表解析、IP 获取                |
| `ray.py`                 | 36   | Ray 基础工具：放置组 master IP/端口获取、资源规格字典构建                             |
| `exp_metadata.py`        | 62   | 实验元数据：版本信息 JSON 的保存与加载                                                |
| `__init__.py`            | 1    | 空文件（仅 license 头）                                                               |

### 2.3 sandbox/ — Daytona 沙箱集成（3 文件，522 行）

| 文件          | 行数 | 职责                                                                           |
| ------------- | ---- | ------------------------------------------------------------------------------ |
| `runner.py`   | 371  | `DaytonaRunner`：同步沙箱执行器，内部维护独立事件循环线程                      |
| `_client.py`  | 138  | `DaytonaClientManager`：进程级 AsyncDaytona 客户端单例管理，支持跨事件循环重建 |
| `__init__.py` | 13   | 公共 API 导出                                                                  |

______________________________________________________________________

## 3. 核心数据结构与接口

### 3.1 Platform 抽象基类（`platform.py` 第 12-149 行）

```
+---------------------------------------------------+
| Platform (抽象基类)                                 |
+---------------------------------------------------+
| 类属性 (7 个)                                      |
|   device_name: str     # "NVIDIA" / "CPU" / "NPU" |
|   device_type: str     # "cuda" / "cpu" / "npu"   |
|   dispatch_key: str    # "CUDA" / "CPU" / "NPU"   |
|   ray_device_key: str  # "GPU" / "CPU" / "NPU"    |
|   device_control_env_var: str                      |
|   ray_experimental_noset: str                      |
|   communication_backend: str  # "nccl"/"gloo"/...  |
+---------------------------------------------------+
| 实例方法                                           |
|   __getattr__(key) -> 委托 torch.<device_type>     |
|   clear_memory()                                   |
+---------------------------------------------------+
| 类方法 (抽象/可覆写)                                |
|   clear_cublas_workspaces()                        |
|   get_vllm_worker_class()                          |
|   set_allocator_settings()                         |
|   set_numa_affinity(local_rank)                    |
|   get_custom_env_vars() -> dict                    |
|   update_env_vars_for_visible_devices(env, ranks)  |
|   get_visible_devices() -> list                    |
+---------------------------------------------------+
```

关键设计：`__getattr__` 将未知属性访问委托给 `torch.<device_type>` 模块（第 68-88 行）， 使调用方可以直接通过
`current_platform.synchronize()` 等方式透明调用底层 PyTorch 设备 API， 而无需关心具体硬件类型。

### 3.2 CudaPlatform（`cuda.py` 第 15-101 行）

NVIDIA 平台的完整实现，关键特性：

- **NUMA 亲和绑定**（第 63-88 行）：通过 `pynvml` 库调用 `nvmlDeviceSetCpuAffinity`， 将进程绑定到 GPU 物理相邻的
  CPU 核心，减少跨 NUMA 节点内存访问延迟。 错误处理包括 pynvml 未安装和运行时异常两种降级路径。

- **vLLM Worker 加载**（第 34-56 行）：优先尝试 vLLM V1 API (`vllm.v1.worker.gpu_worker`)， 失败后回退 V0
  API (`vllm.worker.worker`)，兼容不同 vLLM 版本。

- **自定义环境变量**（第 91-97 行）：设置 `TORCHINDUCTOR_COMPILE_THREADS=2` 限制编译线程、
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 启用可扩展内存段。

### 3.3 JobState / JobInfo / JobException（`launcher.py` 第 168-198 行）

```
+---------------------------+
| JobState (enum.Enum)      |
+---------------------------+
| NOT_FOUND = 0             |
| PENDING   = 1             |
| RUNNING   = 2             |
| COMPLETED = 3             |
| FAILED    = 4             |
| CANCELLED = 5             |
+---------------------------+
| active() -> bool          |
|  (PENDING or RUNNING)     |
+---------------------------+

+---------------------------+
| JobInfo (dataclass)       |
+---------------------------+
| name: str                 |
| state: JobState           |
| host: str | None          |
| submit_time: str | None   |
| start_time: str | None    |
| slurm_id: int | None      |
+---------------------------+

+---------------------------+
| JobException (Exception)  |
+---------------------------+
| run_name: str             |
| worker_type: str          |
| host: str                 |
| reason: JobState          |
+---------------------------+
```

`JobState` 被 Slurm 模块的 `STATUS_MAPPING`（`slurm.py` 第 28-38 行）引用， 将 Slurm 原生状态（RUNNING /
COMPLETING / PENDING / OUT_OF_MEMORY 等）映射到统一枚举。

### 3.4 HTTP 客户端工具（`http.py`）

核心组件：

| 组件                       | 位置          | 说明                                                                                            |
| -------------------------- | ------------- | ----------------------------------------------------------------------------------------------- |
| `create_httpx_client()`    | 第 47-53 行   | 工厂函数：创建带连接池（4096 连接）和传输级重试（3 次）的 `httpx.AsyncClient`                   |
| `get_default_connector()`  | 第 124-125 行 | aiohttp 连接器：无限制连接数、禁用 DNS 缓存、强制关闭连接                                       |
| `async_http_retry`         | 第 105-110 行 | tenacity 装饰器：最多 8 次、指数退避 1-30s、捕获 aiohttp/OS/Runtime 错误                        |
| `async_httpx_retry`        | 第 113-118 行 | tenacity 装饰器：最多 4 次、指数退避 0.5-4s、捕获 httpx 传输错误                                |
| `arequest_with_retry()`    | 第 128-224 行 | 通用异步请求函数：支持 GET/POST/PUT/DELETE、自动内容类型解析、超时重试                          |
| `validate_admin_api_key()` | 第 64-102 行  | 安全检查：拒绝在非回环地址上使用默认管理 API Key，可通过 `AREAL_ALLOW_DEFAULT_ADMIN_KEY=1` 覆写 |

连接池容量常量：

```
HTTPX_MAX_CONNECTIONS          = 4096   (最大连接数)
HTTPX_MAX_KEEPALIVE_CONNECTIONS = 1024  (最大保活连接)
HTTPX_KEEPALIVE_EXPIRY         = 30    (保活超时 秒)
HTTPX_RETRIES                  = 3     (传输级重试)
UVICORN_BACKLOG                = 4096  (服务端积压)
UVICORN_LIMIT_CONCURRENCY      = 4096  (服务端并发限制)
```

### 3.5 Ray 放置组策略层次（`ray_placement_group.py`）

```
+-------------------------------------------+
| RayPlacementStrategy (ABC, dataclass)     |
|-------------------------------------------|
| _placement_groups: list[PlacementGroup]   |
|-------------------------------------------|
| create_placement_group() [抽象]            |
| actor_resources() [抽象]                   |
+-------------------------------------------+
          |                  |
          v                  v
+-----------------------+  +-----------------------------+
| SharedRayPlacement    |  | SeparatedRayPlacement       |
| Strategy              |  | Strategy                    |
|-----------------------|  |-----------------------------|
| 用途: 训练             |  | 用途: 推理 rollout           |
| 多 worker 共享 1 个 PG |  | 每个 rollout 独立 1 个 PG    |
| PACK 策略              |  | PACK 策略                    |
| 按 bundle index 分配   |  | 按 PG index 轮转分配          |
+-----------------------+  +-----------------------------+
                                      |
                                      v
                           +-----------------------------+
                           | DeferredDeviceRayPlacement  |
                           | Strategy                    |
                           |-----------------------------|
                           | 用途: 推理服务器启动时自行     |
                           | 获取加速器                    |
                           | bundle 按节点拆分              |
                           | actor 资源请求为 (0,0,0)      |
                           +-----------------------------+
```

`SharedRayPlacementStrategy.actor_resources()` 中 GPU 乘数默认 0.9
（`MAIN_WORKER_GPU_FRAC_FOR_COLOCATION = 0.9`，第 22 行），允许主 worker 与辅助 进程共享同一张 GPU。

### 3.6 DaytonaRunner / DaytonaClientManager（`sandbox/`）

```
+-------------------------------------------+
| DaytonaClientManager (类方法单例)           |
|-------------------------------------------|
| _client: AsyncDaytona | None              |
| _loop: AbstractEventLoop | None           |
| _config_overrides: dict                   |
|-------------------------------------------|
| configure(**kwargs)                       |
| get_client() -> AsyncDaytona              |
| close()                                   |
+-------------------------------------------+
          |
          | 被引用
          v
+-------------------------------------------+
| DaytonaRunner (实例化使用)                  |
|-------------------------------------------|
| snapshot / image / resources              |
| env_vars / language / ephemeral           |
| default_timeout / create_timeout          |
|-------------------------------------------|
| start() -> DaytonaRunner                  |
| run(code, timeout, ...) -> DaytonaRunResult|
| close()                                   |
| __enter__ / __exit__ (上下文管理器)         |
+-------------------------------------------+

+-------------------------------------------+
| DaytonaRunResult (dataclass)              |
|-------------------------------------------|
| stdout: str                               |
| stderr: str                               |
| exit_code: int                            |
| charts: list[Chart]                       |
| error: str | None                         |
+-------------------------------------------+
```

______________________________________________________________________

## 4. 算法与逻辑详解

### 4.1 平台自动检测机制（`platforms/__init__.py` 第 22-45 行）

检测优先级链：

```
_init_platform()
    |
    +---> torch.cuda.is_available()?
    |         |
    |         +---> YES: get_device_name().upper()
    |         |         |
    |         |         +---> 包含 "NVIDIA"? --> CudaPlatform()
    |         |         |
    |         |         +---> 否 -----------> UnknownPlatform()
    |         |
    |         +---> NO: is_npu_available?
    |                   |
    |                   +---> YES: NPUPlatform()
    |                   |
    |                   +---> NO:  CpuPlatform()
```

`_LazyPlatform` 代理类（第 48-88 行）通过 `__getattr__` / `__setattr__` 将所有属性 访问委托给底层平台实例。私有属性（以
`_` 开头）留在代理自身，其余转发。首次属性访问 触发 `_ensure_initialized()`，此后直接使用缓存实例。

`is_npu_available` 在模块顶层（第 19 行）通过
`transformers.utils.import_utils.is_torch_npu_available()` 提前检测，避免在运行时重复探测。

### 4.2 进程树管理 `kill_process_tree`（`proc.py` 第 140-229 行）

两种终止模式：

```
graceful=True (默认):
    +---------------------+
    | 重置 SIGCHLD handler |  (仅主线程)
    +---------------------+
              |
              v
    +---------------------+
    | psutil 获取进程树     |
    | parent.children()   |
    | (recursive=True)    |
    +---------------------+
              |
              v
    +---------------------+
    | 过滤 skip_pid       |
    +---------------------+
              |
              v
    +---------------------+
    | SIGTERM -> children  |
    | SIGTERM -> parent    |
    +---------------------+
              |
              v
    +---------------------+
    | wait_procs(timeout) |
    +---------------------+
              |
       +------+------+
       |             |
    gone          alive
                     |
                     v
              +-------------+
              | SIGKILL 强杀 |
              | wait_procs   |
              | (timeout=1)  |
              +-------------+

graceful=False:
    直接 SIGKILL 所有子进程
    若 parent == 当前进程: kill + sys.exit(0)
    否则: kill + SIGQUIT (应对 PID=1 的 K8s 场景)
```

关键细节：

- 第 148-149 行：仅在主线程中重置 `SIGCHLD` 为 `SIG_DFL`，避免日志噪音。
- 第 152-154 行：`parent_pid=None` 时默认当前进程，且 `include_parent=False`。
- 第 225-228 行：非优雅模式下对 PID=1（Kubernetes 容器 init 进程）发送额外 `SIGQUIT`， 因为 PID=1 默认忽略
  `SIGKILL`。

### 4.3 流式日志命令构建（`proc.py` 第 26-89 行）

构建 shell 管道实现多路日志输出：

```
[env_vars] stdbuf -oL <cmd> 2>&1
    |
    +---> tee -a <role_log>         (无前缀，角色专属日志)
    |
    +---> stdbuf -oL sed 's/^/[role]  /' >> <merged_log>  (带前缀，合并日志)
```

`stdbuf -oL` 强制行缓冲以实现实时流式输出，macOS 下不可用时自动降级。

### 4.4 Ray 放置组 bundle 拆分策略（`ray_placement_group.py` 第 38-69 行）

`_create_bundle_specs_split` 将 GPU 资源按节点物理布局拆分为多个 bundle：

```
输入: n_gpus_per_node=8, cpu=64, gpu=20, mem=128

步骤 1: divmod(20, 8) = (2 个完整节点, 余 4 GPU)
步骤 2: total_nodes = 3
步骤 3: gpu_per_node = [8, 8, 4]
步骤 4: 按 GPU 比例分配 CPU/MEM:
         节点 0: gpu=8, cpu=round(64*8/20)=26, mem=round(128*8/20)=51
         节点 1: gpu=8, cpu=round(64*8/20)=26, mem=round(128*8/20)=51
         节点 2: gpu=4, cpu=64-26-26=12,        mem=128-51-51=26  (最后一个取余量)

输出: [
    {"CPU": 26, "GPU": 8.0, "memory": 51*1024^3},
    {"CPU": 26, "GPU": 8.0, "memory": 51*1024^3},
    {"CPU": 12, "GPU": 4.0, "memory": 26*1024^3},
]
```

最后一个节点取余量（第 58-59 行 `cpu_left`, `mem_left`），保证总量精确。

### 4.5 Slurm 节点解析与作业管理

#### 节点列表解析（`slurm.py` 第 210-223 行）

```python
parse_slurm_nodelist("node[001-003,005]")
    |
    v
subprocess: scontrol show hostnames node[001-003,005]
    |
    v
["node001", "node002", "node003", "node005"]
```

直接委托 Slurm 原生工具 `scontrol`，避免自行实现复杂的节点名展开规则。

#### sbatch 脚本模板（`slurm.py` 第 40-135 行）

`SBATCH_SCRIPT_TEMPLATE` 生成完整的多节点训练脚本，核心流程：

```
1. 声明 bg_pids 数组 + cleanup_bg_jobs 函数 + trap EXIT
2. scontrol show hostnames 获取节点名
3. srun 在 head_node 获取 IP 和随机空闲端口 (10000-60000)
4. 为每个节点获取 master_addr 和 master_port
5. 执行 srun 命令 (由外部注入 {srun_cmds})
6. 监控循环：轮询 bg_pids，任一失败则 exit 1 触发 trap 清理
```

空闲端口查找使用 `comm -23` 比较 seq 序列与 `ss -tan` 已占用端口的差集， 再 `shuf | head -n 1` 随机选取。

#### 作业查询（`slurm.py` 第 176-207 行）

`query_jobs()` 使用自定义分隔符 `__PSI__` 调用 `squeue -O`：

- 避免字段内容包含空格导致解析错误
- 输出首行为表头，跳过后逐行解析
- 将 Slurm 状态通过 `STATUS_MAPPING` 映射到 `JobState` 枚举

### 4.6 并发工具：线程池与事件循环清理（`concurrent.py`）

#### 命名线程池（第 22-60 行）

```python
get_executor(scope="default", max_workers=4)
```

按 `scope` 名称缓存线程池，双重检查锁保证线程安全。不同 scope 使用独立池， 防止同一池内任务同步等待导致死锁。进程退出时通过 `atexit` 注册的
`_shutdown_all_executors()` 非阻塞关闭所有池。

#### asyncio-atexit 模式（第 101-259 行）

为异步资源（aiohttp session、httpx client、数据库连接）提供事件循环级清理回调：

```
register_loop_cleanup(callback)
    |
    v
_register_loop(loop)
    |
    +---> 已注册? 跳过
    |
    +---> 首次: 保存 loop.close 原始引用
    |          替换 loop.close 为 _patched_loop_close
    |          创建 _LoopCleanupEntry 加入 WeakKeyDictionary
    |
    v
loop.close() 被调用时:
    +---> _patched_loop_close
    |         |
    |         v
    |     run_until_complete(_run_cleanup_callbacks)
    |         |
    |         v
    |     LIFO 逆序执行所有回调 (同步/异步均可)
    |         |
    |         v
    |     调用原始 loop._cleanup_orig_close()
```

使用 `WeakKeyDictionary` 存储注册表，事件循环被垃圾回收时自动清除对应条目。

### 4.7 启动器环境变量组装（`launcher.py`）

#### 缓存目录初始化（第 17-50 行）

```
AREAL_CACHE_DIR (默认 /tmp/areal-<user>)
    |
    +---> PYTORCH_KERNEL_CACHE_PATH  = <cache>/.cache/<user>/torch/kernels/
    +---> VLLM_CACHE_ROOT            = <cache>/.cache/<user>/vllm/
    +---> TRITON_CACHE_PATH          = <cache>/.cache/<user>/triton/
```

模块加载时即 `os.makedirs(exist_ok=True)` 创建。

#### 线程环境变量优先级（`get_thread_env_vars()`，第 65-112 行）

```
优先级 (高 -> 低):
    1. existing_env_vars (用户通过 SchedulingSpec.env_vars 配置)
    2. os.environ (用户在 shell 预设)
    3. cpus_per_task (动态计算)
    4. THREAD_NUM_DEFAULT_WHEN_UNKNOWN = 8 (兜底)

控制变量:
    OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS
    VECLIB_MAXIMUM_THREADS / NUMEXPR_NUM_THREADS
```

#### PYTHONPATH 动态构建（`get_env_vars()`，第 132-165 行）

动态捕获当前 `sys.path` 并合并到 `PYTHONPATH`，确保 Ray 远程 worker 能 import 到与主进程相同的自定义模块（如
`examples/` 下的配置类）。 使用 `dict.fromkeys()` 保序去重。

### 4.8 Daytona 沙箱执行流程

#### DaytonaRunner 内部事件循环（`runner.py` 第 164-190 行）

```
DaytonaRunner._ensure_loop()
    |
    +---> 已关闭? raise RuntimeError
    |
    +---> 已有 loop? 直接返回
    |
    +---> 创建 new_event_loop + daemon Thread
    |     |
    |     +---> Thread: set_event_loop -> run_forever -> shutdown_asyncgens -> close
    |     |
    |     +---> 主线程: ready.wait() 等待 loop 就绪
    |
    v
_run(coro):
    run_coroutine_threadsafe(coro, loop) -> future.result()
```

`DaytonaRunner` 在独立守护线程中运行自有事件循环，通过 `run_coroutine_threadsafe`
桥接同步调用和异步操作，避免与调用方的事件循环冲突。

#### SDK 延迟加载（`_client.py` 第 17-33 行，`runner.py` 第 30-82 行）

两个文件各自实现 `_load_daytona_sdk()`，使用全局变量缓存导入结果：

- `_client.py` 版本返回 `(AsyncDaytona, DaytonaConfig)` 元组
- `runner.py` 版本返回包含 5 个类的字典（含 3 种异常类型）

均在首次调用时触发 `import daytona`，未安装则抛出含安装指令的 `ImportError`。

______________________________________________________________________

## 5. 数据流

### 5.1 平台初始化数据流

```
import areal.infra.platforms
    |
    v
current_platform = _LazyPlatform()   (此时不触发 CUDA)
    |
    v
<首次属性访问> e.g. current_platform.device_type
    |
    v
_LazyPlatform.__getattr__("device_type")
    |
    v
_ensure_initialized()
    |
    v
_init_platform()
    |
    +---> 检测硬件 --> 返回具体 Platform 子类实例
    |
    v
缓存到 self._platform, self._initialized = True
    |
    v
getattr(self._platform, "device_type")
    |
    v
"cuda"  (如果是 NVIDIA GPU)
```

### 5.2 HTTP 请求数据流

```
调用方
    |
    v
arequest_with_retry(addr, endpoint, payload)
    |
    v
split_hostport(addr) --> format_hostport --> url
    |
    v
创建或复用 aiohttp.ClientSession
    |
    +---> connector: limit=0, no_dns_cache, force_close
    +---> timeout: total=sock_connect=connect=3600s (默认)
    +---> read_bufsize: 10MB
    |
    v
for attempt in range(max_retries):
    |
    +---> POST/GET/PUT/DELETE
    |
    +---> 成功: 按 content_type 解析 (json/text/bytes)
    |           关闭 session (若非外部传入)
    |           返回结果
    |
    +---> 超时/ClientError: 日志 warning + sleep(retry_delay)
    |
    v
全部重试失败: raise RuntimeError (含 payload/addr/endpoint 信息)
```

### 5.3 Slurm 作业生命周期

```
validate_config_for_distributed_launcher(config)
    |
    v
生成 SBATCH_SCRIPT_TEMPLATE + SRUN_CMD_TEMPLATE
    |
    v
sbatch 提交 --> Slurm 调度
    |
    v
query_jobs(slurm_ids) --> squeue -O --> 解析 --> [JobInfo]
    |                                              |
    |                                              v
    |                                         STATUS_MAPPING
    |                                         "RUNNING" -> JobState.RUNNING
    |                                         "OUT_OF_MEMORY" -> JobState.FAILED
    v
cancel_jobs(slurm_ids, signal="SIGKILL")
    |
    v
scancel -s SIGKILL <ids>
```

### 5.4 Ray 放置组数据流

```
Scheduler
    |
    v
选择策略: SharedRay / SeparatedRay / DeferredDeviceRay
    |
    v
create_placement_group(role, schedulings, n_gpus_per_node)
    |
    +---> SharedRay: 合并所有 spec 为 bundles -> 1 个 PG
    |
    +---> SeparatedRay: 每个 spec 独立创建 1 个 PG
    |
    +---> DeferredDevice: 按节点拆分 bundle + 独立 PG
    |
    v
placement_group(bundles, strategy="PACK")
    |
    v
ray.get(pg.ready(), timeout) --> 超时则日志 ray.nodes() 并 raise
    |
    v
actor_resources(spec)
    |
    +---> 返回 (resource_dict, PlacementGroupSchedulingStrategy)
    +---> 用于 ray.remote() 的参数
```

______________________________________________________________________

## 6. 关键设计决策与不变量

### 6.1 惰性平台初始化

**决策**：全局 `current_platform` 使用 `_LazyPlatform` 代理，而非模块级直接初始化。

**原因**：避免 `import areal` 时触发 CUDA 初始化。在 CPU 调度节点上执行的 launcher 脚本不应要求 GPU
驱动可用。惰性初始化确保只有实际需要设备信息时才执行探测。

**不变量**：

- `current_platform` 始终提供与具体 `Platform` 子类相同的属性接口
- 初始化后 `_platform` 实例不可变，不会被替换

### 6.2 双重 HTTP 库共存

**决策**：同时使用 `aiohttp`（`arequest_with_retry`）和 `httpx`（`create_httpx_client`）。

**原因**：`aiohttp` 用于已有的 RPC 通信路径（历史遗留），`httpx` 用于新的推理服务 网关转发（更现代的
API，原生支持传输级重试）。两者连接池参数保持对齐 （MAX_CONNECTIONS=4096）。

### 6.3 进程树优雅清理优先

**决策**：`kill_process_tree` 默认 `graceful=True`（SIGTERM 优先，超时后 SIGKILL）。

**原因**：分布式训练子进程可能持有 NCCL 通信组、GPU 内存映射等资源。 优雅关闭给予进程释放资源的机会，减少 GPU 内存泄漏和端口残留。 对 K8s PID=1
场景额外发送 SIGQUIT 是防御性编程。

### 6.4 三种 Ray 放置策略分离

**决策**：训练用 `SharedRayPlacementStrategy`，推理用 `SeparatedRayPlacementStrategy`， 推理服务器用
`DeferredDeviceRayPlacementStrategy`。

**原因**：

- 训练 worker 之间需要 NCCL 通信，物理上应紧密放置（共享同一 PG，PACK 策略）
- 推理 rollout 之间相互独立，各自占用独立资源不互相干扰
- SGLang/vLLM 推理服务器在启动时自行管理 GPU，launcher 只需预留资源（bundle 按节点拆分）， actor 本身请求 0 GPU（第
  235-236 行 `_get_resource_spec` 返回 `(0,0,0)`）

### 6.5 Slurm 脚本模板而非 API

**决策**：通过字符串模板生成 sbatch 脚本，而非使用 Slurm Python 绑定。

**原因**：

- 减少对 Slurm 版本的依赖（不同集群 Slurm 版本差异大）
- 脚本可人工审查和调试
- sbatch 脚本本身包含进程监控和清理逻辑（trap EXIT + bg_pids 轮询）， 不依赖外部监控

### 6.6 Daytona SDK 延迟导入

**决策**：`_load_daytona_sdk()` 使用全局变量缓存，首次调用时才 `import daytona`。

**原因**：Daytona 为可选依赖（`uv sync --extra sandbox`），大多数用户不会安装。 延迟导入确保核心框架不因可选依赖缺失而报错。

### 6.7 DaytonaRunner 独立事件循环

**决策**：`DaytonaRunner` 在独立守护线程中运行自有 `asyncio` 事件循环。

**原因**：调用方可能来自同步代码或已有事件循环的异步代码。独立循环避免嵌套 `asyncio.run()` 导致 "cannot run nested event
loop" 错误，同时与 `DaytonaClientManager`（绑定到调用方循环）解耦。

### 6.8 管理员 API Key 安全守卫

**决策**：`validate_admin_api_key()` 拒绝在非回环地址使用默认 key。

**原因**：默认 key `areal-admin-key` 公开在源码中（第 23 行），在公网暴露等同无认证。 强制用户在生产环境设置自定义 key，或显式通过环境变量
`AREAL_ALLOW_DEFAULT_ADMIN_KEY=1` 确认风险。

______________________________________________________________________

## 7. 已知问题与限制

1. **NPUPlatform 实现不完整**（`npu.py` 仅 30 行）：缺少 `clear_cublas_workspaces`、
   `get_vllm_worker_class`、`set_allocator_settings`、`get_custom_env_vars` 的实现。
   调用这些方法将触发基类的 `raise NotImplementedError()`。 代码第 22 行注释标记 "TODO: NPU"。

1. **UnknownPlatform 与 CudaPlatform 代码重复**：`unknown.py` 的 `clear_cublas_workspaces`、
   `get_vllm_worker_class`、`set_allocator_settings` 与 `cuda.py` 实现几乎相同 （尤其是
   `get_vllm_worker_class` 第 33-55 行完全一致），但 `UnknownPlatform` 缺少 `set_numa_affinity` 和
   `synchronize` 方法。

1. **aiohttp connector 无连接上限**（`http.py` 第 125 行 `limit=0`）：在极端高并发场景 可能导致文件描述符耗尽。

1. **Slurm 端口查找非原子操作**（`slurm.py` 第 86 行）：`comm -23` + `shuf` 选取空闲端口 与实际绑定之间存在 TOCTOU
   竞态条件，多作业同时启动时可能冲突。

1. **`kill_process_tree` 的 PID=1 处理**（`proc.py` 第 225-228 行）：对 PID=1 发送 `SIGQUIT`
   是启发式方案，并非所有容器运行时的 init 进程都响应此信号。

1. **exp_metadata 版本信息依赖 git**（`exp_metadata.py`）：在非 git 仓库环境 （如 pip install 后的纯包安装）下
   `version_info.commit` 等字段可能为空。

1. **DaytonaClientManager 锁为类级别 asyncio.Lock**（`_client.py` 第 49 行）： `asyncio.Lock`
   绑定到创建它的事件循环。在跨循环场景下首次 `get_client()` 可能因锁与当前循环不匹配而产生问题（虽然代码通过检测循环变化来缓解）。

1. **CpuPlatform 缺少 `set_allocator_settings` 和 `get_vllm_worker_class`**
   （`cpu.py`）：这些基类方法未被覆写，调用会触发 `NotImplementedError`。 CPU 模式下通常不需要 vLLM，但接口一致性上存在缺口。

______________________________________________________________________

## 8. 相关测试覆盖

| 被测模块                       | 测试文件                                     | 行数 | 测试要点                                            |
| ------------------------------ | -------------------------------------------- | ---- | --------------------------------------------------- |
| `sandbox/_client.py`           | `tests/infra/sandbox/test_client_manager.py` | 91   | DaytonaClientManager 单例管理、配置覆写、跨循环重建 |
| `sandbox/runner.py`            | `tests/infra/sandbox/test_runner.py`         | 114  | DaytonaRunner 生命周期、run 结果解析、异常处理      |
| `utils/slurm.py`               | `tests/test_slurm_scheduler.py`              | 511  | Slurm 调度器集成、作业状态查询、cancel 逻辑         |
| `utils/ray_placement_group.py` | `tests/test_ray_scheduler.py`                | -    | `_create_bundle_specs_split` 的 bundle 拆分正确性   |
| `utils/proc.py`                | `tests/test_local_scheduler.py`              | -    | `kill_process_tree` 在 LocalScheduler 中的清理行为  |
| `utils/concurrent.py`          | `tests/test_sglang_pp_distributed.py`        | -    | `run_async_task` 桥接同步/异步调用                  |

**覆盖空白**：

- `platforms/` 子模块（`Platform`、`CudaPlatform`、`_LazyPlatform` 等）无直接单元测试，
  仅通过引擎和调度器的集成测试间接覆盖
- `utils/http.py` 的 `arequest_with_retry`、`validate_admin_api_key` 无独立测试
- `utils/launcher.py` 的 `get_env_vars`、`get_thread_env_vars`、配置校验函数无独立测试
- `utils/exp_metadata.py` 无测试
- `utils/concurrent.py` 的事件循环清理回调（asyncio-atexit 模式）无独立测试
