# 数据集服务层

> 源码位置：`areal/infra/data_service/` 文件数：22 个 | 总行数：2100 行

## 1. 概述

数据集服务层是 AReaL 分布式训练框架中的**远程数据加载子系统**，采用经典的 Gateway/Router/Worker
微服务架构，将数据集的加载、采样、分发从训练进程中解耦。 其核心抽象 `RDataset`（Remote Dataset）为训练侧提供了与本地
`torch.utils.data.Dataset` 完全兼容的 map-style 接口，使训练代码无需感知数据来自本地还是远程 Worker。

设计目标：

- **训练无阻塞**：后台预取线程按采样器顺序提前拉取数据，`__getitem__` 近零延迟
- **弹性扩缩容**：Worker 数量由调度器动态管理，Router 自动健康检查与剔除
- **多数据集复用**：同一组 Worker 可同时承载多个数据集，按 dataset_id 隔离
- **认证隔离**：Gateway 层通过 API Key 注册机制隔离管理面（Admin）与消费面（Dataset）

## 2. 目录结构与文件清单

```
areal/infra/data_service/
|-- __init__.py               (  11 行)  公共导出：DataController, DataServiceConfig, RDataset
|-- types.py                  (  44 行)  Pydantic 请求模型（6 种）
|-- rdataset.py               ( 330 行)  RDataset 远程数据集代理 + 预取缓冲区
|
|-- controller/
|   |-- __init__.py            (   1 行)
|   |-- config.py              (  44 行)  DataServiceConfig 数据类
|   +-- controller.py          ( 618 行)  DataController 编排器（核心）
|
|-- gateway/
|   |-- __init__.py            (   1 行)
|   |-- config.py              (  15 行)  GatewayConfig 数据类
|   |-- auth.py                (  64 行)  DatasetKeyRegistry + Bearer Token 认证
|   |-- app.py                 ( 361 行)  Gateway FastAPI 应用（路由转发/广播）
|   +-- __main__.py            (  47 行)  CLI 入口：python -m ...gateway
|
|-- guard/
|   |-- __init__.py            (   1 行)
|   |-- app.py                 (  12 行)  复用 areal.infra.rpc.guard（进程监护）
|   +-- __main__.py            (   8 行)  CLI 入口
|
|-- router/
|   |-- __init__.py            (   1 行)
|   |-- config.py              (  14 行)  RouterConfig 数据类
|   |-- app.py                 ( 148 行)  Router FastAPI 应用（服务注册/负载均衡）
|   +-- __main__.py            (  49 行)  CLI 入口
|
|-- worker/
    |-- __init__.py            (   1 行)
    |-- config.py              (  15 行)  DataWorkerConfig 数据类
    |-- app.py                 ( 265 行)  Worker FastAPI 应用（数据集加载/采样）
    +-- __main__.py            (  50 行)  CLI 入口
```

各子模块行数占比：

| 子模块      | 行数 | 占比  | 职责           |
| ----------- | ---- | ----- | -------------- |
| controller/ | 663  | 31.6% | 生命周期编排   |
| gateway/    | 488  | 23.2% | 认证与请求转发 |
| rdataset.py | 330  | 15.7% | 客户端代理     |
| worker/     | 331  | 15.8% | 数据加载执行   |
| router/     | 212  | 10.1% | 服务发现与路由 |
| guard/      | 21   | 1.0%  | 进程监护复用   |
| 顶层        | 55   | 2.6%  | 导出与类型定义 |

## 3. 架构总览

```
+------------------------------------------------------------------+
|                       训练进程 (Trainer)                          |
|  +-----------------------------------------------------------+   |
|  |  RDataset                                                  |   |
|  |  - connect(controller, dataset_id, ...)                    |   |
|  |  - __getitem__(idx) --> _PrefetchBuffer.get(idx)           |   |
|  |  - __len__() --> total_samples                             |   |
|  +--------|--------------------------------------------------+   |
|            | HTTP POST /v1/samples/fetch                          |
|  +---------v-------------------------------------------------+   |
|  | DataController (controller.py:35)                          |   |
|  |  - initialize(role) --> 启动 Guard/Worker/Router/Gateway   |   |
|  |  - register_dataset() --> POST /v1/datasets/register       |   |
|  |  - unregister_dataset() / destroy()                        |   |
|  +--------|--------------------------------------------------+   |
+------------|-----------------------------------------------------+
             | HTTP (aiohttp)
             v
+------------------------------------------------------------------+
|                     微服务集群 (Guard 进程管理)                    |
|                                                                   |
|  +--------------------+    +-----------------+                    |
|  |   Gateway (:8090)  |    |  Router (:8091) |                    |
|  |  (gateway/app.py)  |    | (router/app.py) |                    |
|  |                    |    |                  |                    |
|  | /v1/datasets/*     |<-->| /route (RR)     |                    |
|  | /v1/samples/fetch  |    | /register       |                    |
|  | /v1/epochs/advance |    | /workers        |                    |
|  | /v1/state/*        |    | /health         |                    |
|  | /v1/shutdown       |    +-----------------+                    |
|  +-------|------------+              |                            |
|          | forward                   | health poll                |
|          v                           v                            |
|  +---------------+  +---------------+  +---------------+         |
|  | Worker-0      |  | Worker-1      |  | Worker-N      |         |
|  | (:auto)       |  | (:auto)       |  | (:auto)       |         |
|  | rank=0        |  | rank=1        |  | rank=N        |         |
|  |               |  |               |  |               |         |
|  | /datasets/load|  | /datasets/load|  | /datasets/load|         |
|  | /v1/samples/* |  | /v1/samples/* |  | /v1/samples/* |         |
|  | /epoch/reset  |  | /epoch/reset  |  | /epoch/reset  |         |
|  | /state/save   |  | /state/save   |  | /state/save   |         |
|  | /state/load   |  | /state/load   |  | /state/load   |         |
|  +---------------+  +---------------+  +---------------+         |
+------------------------------------------------------------------+
```

## 4. 核心组件详解

### 4.1 DataController -- 生命周期编排器

**文件**：`controller/controller.py`（618 行）、`controller/config.py`（44 行）

DataController 是整个数据服务层的入口与编排中心，遵循与 `RolloutControllerV2` 相同的
`__init__(config, scheduler) -> initialize(role)` 模式。

#### 4.1.1 初始化流水线（两阶段异步）

```
initialize(role)                   [主线程]
    |
    +-- _bg_initialize()           [线程池 ctrl_init]
         |
         +-- _async_initialize()   [asyncio 事件循环]
              |
              |  Wave 1: 并行启动
              |  +-- fork DataWorker-0 ... DataWorker-N
              |  +-- fork Router
              |
              |  Wave 2: 并行完成
              |  +-- fork Gateway
              |  +-- register Workers --> Router
              |
              +-- 全部就绪
```

关键设计点：

- **流水线化**（第 77-110 行）：`initialize()` 提交后台线程，主线程只阻塞等待 Guard Worker 就绪（`_workers_ready`
  Event，超时 30 秒），后续的 Gateway/Router/Worker fork 在后台完成
- **两波并行**（第 197-289 行）：Wave 1 用 `asyncio.gather` 并行 fork 所有 DataWorker 和 Router；Wave 2
  并行 fork Gateway 和向 Router 注册所有 Worker
- **回滚保护**（第 290-312 行）：任一步骤失败时，反序 kill 所有已 fork 的服务、 删除调度器 Worker、清空内部状态
- **Guard 进程模型**：每个 Worker 节点运行一个 RPCGuard 进程，DataWorker/Router/Gateway 作为子进程通过 Guard 的
  `/fork` 端点启动，Guard 负责端口分配与进程监护

#### 4.1.2 数据集注册/注销

```python
# controller.py 第 316-397 行
def register_dataset(self, dataset_id, dataset_path, ...) -> dict:
    # 1. 确保初始化完成
    self._ensure_initialized()
    # 2. POST /v1/datasets/register 到 Gateway
    # 3. Gateway 广播 /datasets/load 到所有 Worker
    # 4. 返回 {api_key, dataset_id, total_samples, num_workers}

def unregister_dataset(self, dataset_id) -> None:
    # POST /v1/datasets/unregister 到 Gateway
    # Gateway 广播 /datasets/unload 到所有 Worker
```

注册返回的 `api_key` 是后续数据消费的凭证，由 Gateway 的 `DatasetKeyRegistry` 生成。

#### 4.1.3 销毁流程

`destroy()` 方法（第 433-478 行）按照如下顺序清理：

1. 设置 `_shutdown_requested` 事件，取消未完成的初始化 Future
1. POST `/v1/shutdown` 到 Gateway（触发所有数据集卸载）
1. 反序 kill 所有已 fork 的子服务
1. 反序删除调度器 Worker
1. 清空内部状态

### 4.2 Gateway -- 认证网关与请求代理

**文件**：`gateway/app.py`（361 行）、`gateway/auth.py`（64 行）、`gateway/config.py`（15 行）

Gateway 是数据服务的唯一对外入口，提供基于 FastAPI 的 HTTP API。核心职责：

1. **认证分离**：Admin 操作（注册/注销/关闭）需要 Admin API Key；数据消费操作 （fetch/epoch/state）使用 Dataset API
   Key
1. **请求路由**：fetch_samples 请求通过 Router 获取目标 Worker 地址后单点转发；
   广播类操作（register/unregister/epoch/state）扇出到所有 Worker
1. **失败回滚**：register_dataset 若部分 Worker 失败，自动对成功的 Worker 发送 unload 回滚

#### 4.2.1 API 端点总览

| 端点                      | 方法 | 认证    | 转发模式 | 说明           |
| ------------------------- | ---- | ------- | -------- | -------------- |
| `/health`                 | GET  | 无      | 本地     | 健康检查       |
| `/v1/datasets/register`   | POST | Admin   | 广播     | 加载数据集     |
| `/v1/datasets/unregister` | POST | Admin   | 广播     | 卸载数据集     |
| `/v1/shutdown`            | POST | Admin   | 关闭     | 关闭服务       |
| `/v1/workers`             | GET  | Admin   | 查询路由 | 列出 Worker    |
| `/v1/samples/fetch`       | POST | Dataset | 单点路由 | 按索引取样本   |
| `/v1/epochs/advance`      | POST | Dataset | 广播     | 推进 epoch     |
| `/v1/state/save`          | POST | Dataset | 广播     | 保存采样器状态 |
| `/v1/state/load`          | POST | Dataset | 广播     | 加载采样器状态 |
| `/v1/status`              | GET  | Dataset | 单点路由 | 查询状态       |

#### 4.2.2 认证机制

```
gateway/auth.py
+-------------------------------------+
|       DatasetKeyRegistry            |
|-------------------------------------|
| _admin_key: str                     |
| _key_to_dataset: {api_key -> id}    |
| _dataset_to_key: {id -> api_key}    |
|-------------------------------------|
| generate_key(dataset_id) -> api_key |  # "ds-" + uuid4 hex[:16]
| resolve(api_key) -> dataset_id      |
| revoke(dataset_id) -> api_key       |
| is_admin(api_key) -> bool           |  # hmac.compare_digest 防时序攻击
+-------------------------------------+
```

认证流程：

1. 所有请求从 `Authorization: Bearer <token>` 头提取 token（第 51-57 行）
1. Admin 端点通过 `require_admin_key()` 用 `hmac.compare_digest` 验证（防时序攻击）
1. Dataset 端点通过 `_resolve_dataset_key()` 在注册表中查找对应 dataset_id

### 4.3 Router -- 服务注册与负载均衡

**文件**：`router/app.py`（148 行）、`router/config.py`（14 行）

Router 维护 Worker 注册表，提供轮询（Round-Robin）路由和周期性健康检查。

#### 4.3.1 核心机制

```
+----------------------------------------------+
|               DataRouter                      |
|----------------------------------------------|
| registered_workers: list[str]                 |  # 有序 Worker 地址列表
| worker_healthy: dict[str, bool]               |  # 健康状态表
| rr_idx: int                                   |  # 轮询计数器
|----------------------------------------------|
| POST /register   -- 添加 Worker 到列表       |
| POST /unregister -- 从列表移除 Worker         |
| POST /route      -- RR 选择健康 Worker 返回   |
| GET  /workers    -- 列出所有 Worker 及健康状态 |
| GET  /health     -- 返回 Worker 总数/健康数   |
+----------------------------------------------+
```

- **健康检查**（第 55-72 行）：后台 asyncio Task 每 `poll_interval`（默认 5 秒） 并行检查所有已注册 Worker 的
  `/health` 端点
- **路由策略**（第 122-136 行）：仅在健康 Worker 中做 Round-Robin，`rr_idx` 单调递增 取模，保证均匀分布
- **生命周期**（第 74-84 行）：通过 FastAPI `lifespan` 上下文管理器启动/取消 后台轮询任务

### 4.4 Worker -- 数据加载执行引擎

**文件**：`worker/app.py`（265 行）、`worker/config.py`（15 行）

Worker 是实际执行数据集加载和采样的节点，每个 Worker 持有独立的数据集分片。

#### 4.4.1 数据集加载三阶段协议

```
POST /datasets/load
    |
    Phase 1: Reserve (datasets_lock)    [快速]
    |   - 检查 dataset_id 未重复
    |   - 加入 _loading_ids 占位集合
    |
    Phase 2: Load (无锁, asyncio.to_thread)  [慢速 I/O]
    |   - load_hf_processor_and_tokenizer()
    |   - _get_custom_dataset(path, type, split, ...)
    |   - 创建 DistributedSampler(world_size, rank)
    |   - 创建 StatefulDataLoader(batch_size=1)
    |
    Phase 3: Store (datasets_lock)      [快速]
        - 从 _loading_ids 移除
        - 存入 datasets 字典
        - 返回 {dataset_size, steps_per_epoch}
```

关键设计：

- **两级锁**（第 53-63 行）：`datasets_lock` 保护字典级变更，`state.lock` 保护 单数据集操作。锁序为
  `datasets_lock -> state.lock`，避免死锁
- **Loading 占位**：`_loading_ids` 集合防止并发重复加载同一 dataset_id
- **Unloading 标志**：`_DatasetState.unloading` 布尔位防止卸载中的数据集被访问

#### 4.4.2 采样与状态管理

| 端点                | 功能                                                                   |
| ------------------- | ---------------------------------------------------------------------- |
| `/v1/samples/fetch` | 按索引列表直接访问 `raw_dataset[idx]`，经 `serialize_value` 序列化返回 |
| `/epoch/reset`      | 设置 sampler epoch，重置 exhausted 标志                                |
| `/state/save`       | pickle 序列化 `dataloader.state_dict()` 到磁盘                         |
| `/state/load`       | 从磁盘反序列化恢复 dataloader 状态                                     |
| `/data/clear`       | 清理缓存（当前为空操作占位）                                           |

状态保存路径按 Worker rank 分片：`{path}/worker_{rank}.pkl`，支持 checkpoint 恢复时精确还原每个 Worker 的采样位置。

### 4.5 RDataset -- 远程数据集代理

**文件**：`rdataset.py`（330 行）

RDataset 是训练侧的核心抽象，对外暴露标准 map-style 接口 `__len__` / `__getitem__`， 内部通过 HTTP 从远程 Worker
获取数据。

#### 4.5.1 生命周期状态机

```
 [Unconnected]                       [Connected]                     [Closed]
  创建时仅存                          可正常使用                      资源已释放
  元数据                              __getitem__/__len__

  RDataset(path, type, ...)
       |
       | connect(controller, dataset_id, ...)
       |   - controller.register_dataset()
       |   - 获得 api_key + total_samples
       |   - 创建 _PrefetchBuffer
       |
       +---> [Connected]
                  |
                  | close()
                  |   - _PrefetchBuffer.stop()
                  |   - controller.unregister_dataset()
                  |
                  +---> [Closed]
```

#### 4.5.2 预取缓冲区（\_PrefetchBuffer）

```
_PrefetchBuffer (rdataset.py 第 36-155 行)
+---------------------------------------------------+
|  _cache: dict[int, Any]    # idx -> sample 缓存   |
|  _indices: list[int]       # 本 epoch 采样顺序    |
|  _pos: int                 # 当前预取位置          |
|  _chunk_size: int = 64     # 每次 HTTP 批量大小    |
|  _max_cached: int = 512    # 缓存上限（反压机制）  |
+---------------------------------------------------+
|  set_index_order(indices)  # 每 epoch 开始时重置   |
|  get(idx) -> sample        # 缓存命中则 pop        |
|  stop()                    # 停止后台线程          |
+---------------------------------------------------+

后台线程 _run():
    while not stop:
        chunk = indices[pos : pos + chunk_size]    # 取一批索引
        wait_for(space_available)                  # 反压：缓存满时暂停
        samples = fetch_fn(chunk)                  # HTTP 批量获取
        cache[idx] = sample for each               # 存入缓存

消费侧 get(idx):
    if idx in cache:
        pop(idx)              # 缓存命中 --> 释放空间 --> set(space_available)
    else:
        fetch_fn([idx])       # 缓存未命中 --> 阻塞单次 HTTP 请求
```

设计要点：

- **顺序预取**：与 `_PrefetchAwareSampler` 配合，预取线程严格按照采样器将要请求的 索引顺序拉取，理想情况下缓存命中率接近 100%
- **反压机制**：缓存达到 `max_cached` 时预取线程暂停，由 `get()` 的 `pop` 操作 释放空间后恢复
- **失败重试**：预取失败时回退 `_pos`，等待 0.5 秒后重试同一批次

#### 4.5.3 \_PrefetchAwareSampler

```python
# rdataset.py 第 309-331 行
class _PrefetchAwareSampler(DistributedSampler):
    """在 set_epoch 时触发 RDataset 预取"""

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        indices = list(super().__iter__())      # 生成本 epoch 索引序列
        self._rdataset._start_prefetch(indices) # 传递给预取缓冲区
```

该采样器继承 `DistributedSampler`，在 `set_epoch` 时生成确定性索引序列并传递给 `_PrefetchBuffer`，使预取线程能在
DataLoader 实际请求之前开始拉取数据。

### 4.6 Guard -- 进程监护

**文件**：`guard/app.py`（12 行）、`guard/__main__.py`（8 行）

Guard 子模块完全复用 `areal.infra.rpc.guard` 基础设施，无自定义逻辑。 RPCGuard 在每个 Worker 节点上运行，提供：

- `/alloc_ports`：分配可用端口
- `/fork`：启动子进程（DataWorker/Router/Gateway）
- `/kill_forked_worker`：终止指定子进程

### 4.7 类型定义

**文件**：`types.py`（44 行）

6 个 Pydantic `BaseModel` 请求体，用于 Worker HTTP API 的参数校验：

| 类名                         | 字段                                                                  | 用途         |
| ---------------------------- | --------------------------------------------------------------------- | ------------ |
| `WorkerLoadDatasetRequest`   | dataset_id, path, type, split, max_length, shuffle, drop_last, kwargs | 加载数据集   |
| `WorkerUnloadDatasetRequest` | dataset_id                                                            | 卸载数据集   |
| `WorkerEpochResetRequest`    | dataset_id, epoch                                                     | 重置 epoch   |
| `WorkerStateSaveRequest`     | dataset_id, path                                                      | 保存状态     |
| `WorkerStateLoadRequest`     | dataset_id, path                                                      | 加载状态     |
| `FetchSamplesRequest`        | dataset_id, indices                                                   | 按索引取样本 |

## 5. 数据流与交互序列

### 5.1 数据集注册完整流程

```
Trainer            DataController       Gateway          Router         Worker-0..N
  |                     |                  |               |               |
  | register_dataset()  |                  |               |               |
  |-------------------->|                  |               |               |
  |                     | POST /v1/datasets/register       |               |
  |                     |----------------->|               |               |
  |                     |                  | GET /workers   |               |
  |                     |                  |-------------->|               |
  |                     |                  |<-------------|               |
  |                     |                  |  [worker_addrs]              |
  |                     |                  |                              |
  |                     |                  | POST /datasets/load (广播)    |
  |                     |                  |----------------------------->|
  |                     |                  |<-----------------------------|
  |                     |                  |  {dataset_size, steps}       |
  |                     |                  |                              |
  |                     |                  | generate_key(dataset_id)     |
  |                     |                  |  => "ds-xxxx"                |
  |                     |                  |                              |
  |                     |<-----------------|                              |
  |                     |  {api_key, dataset_id, total_samples}           |
  |<--------------------|                                                 |
  |  {api_key, dataset_id, total_samples}                                 |
```

### 5.2 样本获取数据流

```
RDataset.__getitem__(idx)
  |
  +-- _PrefetchBuffer.get(idx)
       |
       +-- [缓存命中] --> pop(idx), return sample
       |
       +-- [缓存未命中] --> _fetch_samples([idx])
                              |
                              +-- DataController._gateway_post("/v1/samples/fetch")
                                     |
                                     +-- Gateway.fetch_samples()
                                          |
                                          +-- Router.route() --> worker_addr (RR)
                                          |
                                          +-- Worker.fetch_samples(dataset_id, indices)
                                               |
                                               +-- raw_dataset[idx]
                                               +-- serialize_value(sample)
                                          |
                                     <-- {samples: [...]}
                              |
                              +-- deserialize_value(s) for each sample
                         |
                    <-- sample
```

### 5.3 初始化时序（DataController 内部）

```
时间线 --->

主线程: initialize() -----> 等待 _workers_ready -----> 返回 Future
                  |
后台线程:         +-- scheduler.create_workers(guard_job)
                  |   RPCGuard-0..N 就绪
                  |   _workers_ready.set() ------------> 主线程继续
                  |
                  +-- Wave 1: asyncio.gather(
                  |       fork(Worker-0), fork(Worker-1), ..., fork(Router)
                  |   )
                  |   等待所有 /health 200
                  |
                  +-- Wave 2: asyncio.gather(
                  |       fork(Gateway),
                  |       register(Worker-0..N -> Router)
                  |   )
                  |
                  +-- 初始化完成
```

## 6. 配置体系

### 6.1 DataServiceConfig

```python
# controller/config.py 第 11-30 行
@dataclass
class DataServiceConfig:
    num_workers: int = 1                    # 数据 Worker 数量
    scheduling_spec: SchedulingSpec = ...   # Guard 进程资源规格
    scheduling_strategy: SchedulingStrategy # 始终 "separation" 策略
    setup_timeout: float = 120.0            # 初始化超时（秒）
    dataloader_num_workers: int = 4         # PyTorch DataLoader 子进程数
    seed: int = 42                          # 随机种子
```

通过 `from_dataset_config(dataset_config, seed)` 静态方法从用户配置转换。

### 6.2 微服务配置

| 配置类             | 关键字段                                                        | 默认值              |
| ------------------ | --------------------------------------------------------------- | ------------------- |
| `GatewayConfig`    | host, port, router_addr, admin_api_key, forward_timeout         | 0.0.0.0:8090, 60s   |
| `RouterConfig`     | host, port, admin_api_key, poll_interval, worker_health_timeout | 0.0.0.0:8091, 5s/3s |
| `DataWorkerConfig` | host, port, rank, world_size, dataloader_num_workers, seed      | 0.0.0.0:auto, 4, 42 |

所有配置均为 `@dataclass`，CLI 参数在各 `__main__.py` 中通过 argparse 注入。

## 7. 外部集成点

### 7.1 调用方

数据集服务层被以下 Trainer 使用：

| Trainer      | 文件                     | 使用方式                           |
| ------------ | ------------------------ | ---------------------------------- |
| `RLTrainer`  | `trainer/rl_trainer.py`  | 训练集 + 验证集各创建一个 RDataset |
| `SFTTrainer` | `trainer/sft_trainer.py` | 同上                               |
| `DPOTrainer` | `trainer/dpo_trainer.py` | 同上                               |
| `RWTrainer`  | `trainer/rw_trainer.py`  | 同上                               |

通用流程（以 `rl_trainer.py` 第 212-217 行为例）：

```python
if is_single_controller() and isinstance(train_dataset, RDataset):
    ds_cfg = DataServiceConfig.from_dataset_config(dataset_config, seed=seed)
    controller = DataController(ds_cfg, self.scheduler)
    controller.initialize(role, num_dataset_workers=ds_cfg.num_workers)
    train_dataset.connect(controller, dataset_id="train", ...)
```

### 7.2 依赖

| 依赖模块                        | 用途                                    |
| ------------------------------- | --------------------------------------- |
| `areal.infra.rpc.guard`         | Guard 进程管理（fork/kill/alloc_ports） |
| `areal.infra.rpc.serialization` | 样本序列化/反序列化                     |
| `areal.api.scheduler_api`       | Scheduler/Worker/Job 抽象               |
| `areal.api.cli_args`            | SchedulingSpec, \_DatasetConfig         |
| `areal.dataset`                 | `_get_custom_dataset()` 数据集工厂      |
| `areal.utils.dataloader`        | `create_dataloader()` 集成 RDataset     |
| `areal.utils.network`           | `format_hostport()` 地址格式化          |
| `areal.utils.seeding`           | Worker 启动时种子初始化                 |

### 7.3 DataLoader 集成

`areal/utils/dataloader.py`（第 34-47 行）根据 dataset 类型自动选择采样器：

```
dataset 类型       | dataset_config 类型    | 采样器类
RDataset           | ValidDatasetConfig     | _PrefetchAwareEvalSampler
RDataset           | 其他                   | _PrefetchAwareSampler
普通 Dataset       | ValidDatasetConfig     | EvalDistributedSampler
普通 Dataset       | 其他                   | DistributedSampler
```

## 8. 设计模式与关键决策

### 8.1 微服务分层模式

```
+------------------------------------------+
|  Gateway  (认证 + 路由 + 转发)            |
|  - 唯一对外入口                           |
|  - Admin vs Dataset 双认证域             |
|  - 广播 vs 单点两种转发模式              |
+------------------------------------------+
|  Router  (服务发现 + 负载均衡)            |
|  - Worker 注册/注销                       |
|  - 周期性健康检查 + 自动剔除不健康节点   |
|  - Round-Robin 路由                       |
+------------------------------------------+
|  Worker  (数据加载 + 采样)                |
|  - 多数据集隔离（dataset_id 分区）       |
|  - 两级锁保护并发                        |
|  - StatefulDataLoader 支持状态持久化     |
+------------------------------------------+
```

这种三层架构将**认证/路由/执行**关注点完全分离：

- Gateway 不持有数据，仅做请求转发与认证
- Router 不了解数据集语义，仅做 Worker 发现与健康管理
- Worker 不处理认证与路由，专注数据加载

### 8.2 关键设计决策

| 决策                               | 原因                                                     |
| ---------------------------------- | -------------------------------------------------------- |
| 客户端预取而非服务端推送           | 预取顺序与采样器绑定，客户端能预知完整索引序列           |
| Guard 复用 RPC 基础设施            | 避免重复实现进程管理、端口分配、子进程监护               |
| Admin Key 使用 hmac.compare_digest | 防止时序侧信道攻击                                       |
| Worker 两级锁 + loading_ids 集合   | 数据集加载是慢速 I/O，不能持锁；占位集合防止并发重复加载 |
| StatefulDataLoader + pickle 持久化 | 支持 checkpoint 恢复时精确还原每个 Worker 的采样位置     |
| batch_size=1 的 Worker DataLoader  | Worker 侧不做 batch，由训练侧控制 batch 大小             |
| 分离调度策略（separation）         | 数据服务需在训练引擎之前启动，不能与引擎共享调度资源     |

### 8.3 并发模型

```
DataController:
    主线程 --> 提交到 ctrl_init 线程池 --> asyncio 事件循环
    _init_lock  : 保护 _init_future 单次初始化
    _workers_ready : 主线程与后台线程同步点

Worker (FastAPI + uvicorn):
    asyncio 事件循环 (单进程)
    datasets_lock (asyncio.Lock)  : 保护 datasets dict + _loading_ids
    state.lock (asyncio.Lock)     : 保护单数据集状态操作
    锁序: datasets_lock --> state.lock

Router (FastAPI + uvicorn):
    asyncio 事件循环 (单进程)
    lock (asyncio.Lock)           : 保护 Worker 列表变更与路由选择
    后台 Task                      : 周期性健康检查

RDataset._PrefetchBuffer:
    后台守护线程 + threading.Lock
    _space_available (Event)      : 反压信号
    _stop (Event)                 : 终止信号
```
