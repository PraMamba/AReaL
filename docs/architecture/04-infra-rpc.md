# RPC 与序列化层

> 源码位置：`areal/infra/rpc/` 文件数：10 个 | 总行数：3252 行 最后更新：2026-06-13

______________________________________________________________________

## 1. 模块定位

RPC 层是 AReaL 分布式 RL 训练框架的**远程过程调用与张量传输**基础设施。它解决了 两个核心问题：

1. **跨节点引擎调用** -- 控制器（TrainController）需要在远端 Worker 上创建、配置 并调用 TrainEngine /
   InferenceEngine 的方法，同时保证 NCCL 线程安全。
1. **大规模张量传输** -- rollout 生成的轨迹数据（可达数十 GB）需要在控制器与多个 训练 Worker 之间高效传输，并支持按 DP head 广播。

RPC 层提供两条并行通道：

- **HTTP 通道**（Flask + werkzeug）：同步 RPC 服务器，用于 SLURM / 本地调度。
- **Ray 通道**（Ray Actor）：基于 Ray Object Store 的异步 RPC 服务器，用于 Ray 调度。

两条通道共享同一套序列化协议（`serialization.py`）和张量引用抽象（`RTensor`）。

______________________________________________________________________

## 2. 目录结构与文件清单

```
areal/infra/rpc/
|-- __init__.py              13 行   公共导出：RTensor, TensorShardInfo, serialize/deserialize
|-- rtensor.py              646 行   分布式张量引用(RTensor)与后端抽象
|-- serialization.py        778 行   序列化协议：Tensor/NDArray/Dataclass/Tokenizer/Processor/Image
|-- rpc_server.py            66 行   HTTP RPC 入口（组合 Guard + Data + Engine 蓝图）
|-- ray_rpc_server.py       222 行   Ray Actor RPC 服务器
|
+-- guard/                           Guard 子系统 (进程管理 + Flask 蓝图)
    |-- __init__.py          38 行   Guard 公共导出
    |-- __main__.py          35 行   独立 Guard 入口 (python -m areal.infra.rpc.guard)
    |-- app.py              649 行   Guard 核心：GuardState + Flask 路由 + 服务生命周期
    |-- data_blueprint.py   199 行   数据平面蓝图：/data/* 张量分片存取
    |-- engine_blueprint.py 606 行   引擎平面蓝图：/create_engine, /call, 引擎线程
```

______________________________________________________________________

## 3. 架构总览

```
                           TrainController / RayScheduler
                                     |
                    +----------------+----------------+
                    |  HTTP 通道                       |  Ray 通道
                    v                                  v
         +-------------------+              +--------------------+
         |   rpc_server.py   |              | ray_rpc_server.py  |
         |  (Flask WSGI)     |              |  (Ray Actor)       |
         +---+------+--------+              +--------+-----------+
             |      |                                |
    +--------+  +---+--------+                       |
    | Guard  |  | data_bp    |                       |
    | app.py |  | (Blueprint)|                       |
    +--------+  +------------+                       |
                | engine_bp  |                       |
                | (Blueprint)|                       |
                +-----+------+                       |
                      |                              |
                      v                              v
               +-------------+              +--------------+
               |  RTensor    |              |   RTensor    |
               | HttpBackend |              | RayBackend   |
               +------+------+              +------+-------+
                      |                            |
                      v                            v
              _storage (dict)             Ray Object Store
               (进程内存)                  (共享内存)
```

### 数据流概要

```
 Controller                Worker (HTTP)                 Worker (Ray)
    |                          |                              |
    |--- serialize_value() --->|                              |
    |    (Tensor->base64)      |                              |
    |                    deserialize_value()                   |
    |                    RTensor.localize()                    |
    |                          |                              |
    |                   engine.method()                       |
    |                          |                              |
    |                   RTensor.remotize()                    |
    |<-- serialize_value() ----|                              |
    |    (结果返回)              |                              |
    |                          |                              |
    |--- ray.remote.call() ---------------------------------->|
    |                                          RTensor.localize()
    |                                          engine.method()
    |                                          RTensor.remotize()
    |<------------------------------------- ray.get(result) ---|
```

______________________________________________________________________

## 4. 核心组件详解

### 4.1 RTensor -- 分布式张量引用 (`rtensor.py`, 646 行)

RTensor 是整个 RPC 层的核心数据抽象。它将一个 `torch.Tensor` 转换为\*\*元数据引用

- 远端存储\*\*的两阶段表示，使大张量无需随 HTTP JSON 请求体一起传输。

#### 4.1.1 类结构

```
RTensorBackend (Protocol)         <-- 后端抽象接口
|-- fetch(shards) -> list[Tensor]
|-- store(tensor) -> shard_id
|-- delete(node_addr, shard_ids)
|
+-- HttpRTensorBackend            <-- HTTP 后端 (第 88-271 行)
|   |-- _fetch_tensor()           单分片 HTTP GET
|   |-- _fetch_shard_group()      批量 POST /data/batch
|   |-- fetch()                   并发获取, 按 node 分组
|   |-- store()                   本地 _storage + UUID
|   +-- delete()                  HTTP DELETE /data/clear
|
+-- RayRTensorBackend             <-- Ray 后端 (第 273-286 行)
    |-- fetch()                   ray.get(ObjectRef)
    |-- store()                   ray.put(tensor)
    +-- delete()                  ray.internal.free()

RTensor (dataclass)               <-- 张量引用 (第 371-580 行)
|-- shard: TensorShardInfo
|-- data: torch.Tensor            (meta tensor 或已物化)
|
|-- to_local()                    延迟物化, 先查 fetch_buffer
|-- remotize(obj, node_addr)      递归: Tensor -> RTensor
|-- localize(obj)                 递归: RTensor -> Tensor, 批量预取
|-- collect_shards(obj)           收集所有分片 ID (按 node 分组)
+-- clear_node(node_addr, ids)    清理远端 + fetch_buffer
```

#### 4.1.2 后端自动选择

```python
# rtensor.py 第 292-299 行
def get_backend() -> RTensorBackend:
    global _backend
    if _backend is None:
        if ray.is_initialized():
            _backend = RayRTensorBackend()
        else:
            _backend = HttpRTensorBackend()
    return _backend
```

启动时根据 `ray.is_initialized()` 自动选择后端。HTTP 后端用于 SLURM/本地调度， Ray 后端用于 Ray 调度。

#### 4.1.3 remotize / localize 对称操作

| 操作       | 方向              | 输入               | 输出               | 关键行为                                                         |
| ---------- | ----------------- | ------------------ | ------------------ | ---------------------------------------------------------------- |
| `remotize` | Tensor -> RTensor | 嵌套结构含 Tensor  | 嵌套结构含 RTensor | 1) detach+cpu 2) backend.store 3) data 置为 meta                 |
| `localize` | RTensor -> Tensor | 嵌套结构含 RTensor | 嵌套结构含 Tensor  | 1) 收集所有 meta RTensor 2) 查 fetch_buffer 3) 批量 fetch 缺失项 |

**紧凑优化**（第 422-435 行）：`remotize` 检测含 `attention_mask` 的轨迹字典， 自动调用
`split_and_unpad_tensor` 裁剪尾部 padding，减少存储和传输开销。

#### 4.1.4 客户端 Fetch Buffer

```
_fetch_buffer: dict[shard_id -> Tensor]   (第 321 行)
```

进程内缓存，避免同一分片被多次网络获取。清理遵循三点不变式（第 315-319 行注释）：

1. **Controller 端**：`RTensor.clear_node()` 弹出本地缓存
1. **存储所有者 Worker**：`remove()` 弹出 `_storage` 和 `_fetch_buffer`
1. **跨节点消费者**：`clear_batches` RPC 触发 `clear_fetch_buffer(sids)`

#### 4.1.5 本地存储 (HTTP 后端专用)

```python
# rtensor.py 第 593-646 行
_storage: dict[str, torch.Tensor] = {}     # shard_id -> Tensor
_storage_lock = Lock()
_storage_stats: dict[str, int] = {}        # shard_id -> nbytes
```

HTTP 后端的 `store()` 将张量写入进程内 `_storage` 字典；`data_blueprint` 通过 `/data/<shard_id>` 路由提供
HTTP 存取接口。

______________________________________________________________________

### 4.2 序列化协议 (`serialization.py`, 778 行)

序列化层定义了 RPC 通信中所有数据类型的 JSON 编码协议。每种类型通过 Pydantic 模型封装，带有 `type` 字段标签实现运行时多态反序列化。

#### 4.2.1 序列化类型矩阵

| Pydantic 模型            | type 标签          | 编码方式                                       | 行号    |
| ------------------------ | ------------------ | ---------------------------------------------- | ------- |
| `SerializedTensor`       | `"tensor"`         | base64(numpy.tobytes) + shape + dtype          | 49-174  |
| `SerializedNDArray`      | `"ndarray"`        | base64(contiguous.tobytes) + shape + dtype.str | 177-218 |
| `SerializedPILImage`     | `"pil_image"`      | base64(PNG) + mode                             | 221-249 |
| `SerializedRayObjectRef` | `"ray_object_ref"` | base64(cloudpickle)                            | 252-269 |
| `SerializedDataclass`    | `"dataclass"`      | class_path + 递归序列化字段                    | 272-336 |
| `SerializedTokenizer`    | `"tokenizer"`      | base64(ZIP/Zstd archive) + name_or_path        | 339-442 |
| `SerializedProcessor`    | `"processor"`      | base64(ZIP/Zstd archive) + name_or_path        | 445-551 |
| `torch.dtype`            | `"torch_dtype"`    | str(dtype)                                     | 640-641 |
| `enum.Enum`              | `"enum"`           | class_path + value                             | 644-650 |

#### 4.2.2 Tensor 序列化流程

```
serialize_value(tensor)
    |
    v
SerializedTensor.from_tensor()
    |-- tensor.detach().cpu()
    |-- bfloat16 ? upcast to float32 (第 100-101 行)
    |-- numpy().tobytes() -> base64
    +-- 保存 shape + 原始 dtype

deserialize_value(dict)
    |
    v
SerializedTensor.to_tensor()
    |-- base64 -> bytes
    |-- np.frombuffer(dtype=_torch_dtype_to_numpy(dtype))
    |-- np_array.copy()  # 确保可写
    |-- torch.from_numpy().reshape(shape)
    +-- .to(dtype)  # 恢复原始 dtype (如 bfloat16)
```

**bfloat16 处理**：NumPy 不原生支持 bfloat16，序列化时上转为 float32 存储缓冲 区，在 dtype 元数据中保留原始
`torch.bfloat16`，反序列化时通过 `.to(dtype)` 恢复 （第 99-101 行, 第 133-134 行）。

#### 4.2.3 Tokenizer/Processor 压缩策略

```
save_pretrained() -> tmpdir
    |
    v
total_size < 512KB?  --> ZIP_STORED (无压缩)
total_size >= 512KB? --> ZIP_DEFLATED (level=6)
    |
    v
blob > 20MB 且有 zstandard? --> Zstd level=3 二次压缩
    |
    v
base64 编码 -> JSON 传输
```

常量定义（第 41-42 行）：

- `TOKENIZER_ARCHIVE_INLINE_THRESHOLD = 512 * 1024`（512KB）
- `TOKENIZER_ZSTD_THRESHOLD = 20 * 1024 * 1024`（20MB）

#### 4.2.4 serialize_value / deserialize_value 递归调度

`serialize_value`（第 554-653 行）和 `deserialize_value`（第 656-778 行）是两个 对称的递归函数，通过
`isinstance` 链判断类型并分派到对应的 Pydantic 模型。

**类型优先级**（serialize 顺序）：

1. None -> None
1. torch.Tensor -> SerializedTensor
1. np.ndarray -> SerializedNDArray
1. PIL.Image -> SerializedPILImage
1. ray.ObjectRef -> SerializedRayObjectRef
1. dataclass (非 type) -> SerializedDataclass（递归字段）
1. Tokenizer -> SerializedTokenizer
1. Processor -> SerializedProcessor
1. dict -> 递归 values
1. list/tuple -> 递归 elements
1. subprocess.Popen -> None（跳过）
1. torch.dtype -> `{"type": "torch_dtype", "value": str}`
1. enum.Enum -> `{"type": "enum", "class_path": ..., "value": ...}`
1. 原语（int/float/str/bool）-> 透传

______________________________________________________________________

### 4.3 Guard 进程架构 (`guard/app.py`, 649 行)

Guard 是 HTTP 通道的**进程管理网关**，提供子进程 fork、端口分配、健康检查等基础 设施。Guard 通过 Flask Blueprint
模式实现数据平面和引擎平面的可插拔组合。

#### 4.3.1 GuardState -- 共享可变状态

```python
# guard/app.py 第 41-115 行
class GuardState:
    # 服务身份
    server_host, server_port
    experiment_name, trial_name, fileroot
    name_resolve_type, nfs_record_root, etcd3_addr
    role, worker_index

    # 线程安全资源
    allocated_ports: set[int]          + Lock
    forked_children: list[Popen]       + Lock
    forked_children_map: dict[(role, index) -> Popen]

    # Hook 系统
    _health_hooks: list[HealthHook]       # () -> dict
    _configure_hooks: list[ConfigureHook] # (dict) -> dict
    _cleanup_hooks: list[CleanupHook]     # () -> None
```

Hook 系统使 Guard 核心与蓝图解耦：蓝图通过注册 hook 扩展 `/health`、`/configure` 和关闭清理逻辑，Guard
本身不感知引擎或数据存储的存在。

#### 4.3.2 Guard 核心路由

| 路由                  | 方法 | 功能                          | 行号    |
| --------------------- | ---- | ----------------------------- | ------- |
| `/health`             | GET  | 健康检查 + hook 扩展字段      | 180-191 |
| `/alloc_ports`        | POST | 分配空闲端口（排斥已分配）    | 193-225 |
| `/fork`               | POST | 派生子 Worker 进程（raw_cmd） | 227-328 |
| `/kill_forked_worker` | POST | 终止指定子进程                | 330-416 |
| `/set_env`            | POST | 设置环境变量                  | 418-447 |
| `/configure`          | POST | 配置 Worker（分派到 hook）    | 449-485 |

#### 4.3.3 服务生命周期 (`run_server`, 第 581-649 行)

```
make_base_parser() -> parse CLI args
    |
    v
configure_state_from_args()
    |-- 解析 host (0.0.0.0 -> gethostip())
    |-- 解析 worker_index (支持 SLURM_PROCID 覆盖)
    |-- 填充 name_resolve 配置
    |
    v
create_app(state)
    |-- 注册核心路由
    |
    v
app.register_blueprint(data_bp)    # 可选
app.register_blueprint(engine_bp)  # 可选
register_engine_hooks(state)       # 可选
    |
    v
run_server(state, app, bind_host, port)
    |-- werkzeug.make_server(threaded=True)
    |-- name_resolve.add(key, node_addr)   # 注册服务发现
    |-- SIGTERM -> SystemExit 转换
    |-- server.serve_forever()
    |-- finally: cleanup_hooks + cleanup_forked_children + server.shutdown
```

______________________________________________________________________

### 4.4 数据平面蓝图 (`guard/data_blueprint.py`, 199 行)

数据平面蓝图为 `HttpRTensorBackend` 提供 HTTP 张量存取接口。

#### 4.4.1 路由

| 路由               | 方法   | 功能             | 行号    |
| ------------------ | ------ | ---------------- | ------- |
| `/data/<shard_id>` | PUT    | 存储单个张量分片 | 58-75   |
| `/data/<shard_id>` | GET    | 获取单个张量分片 | 78-104  |
| `/data/batch`      | POST   | 批量获取多个分片 | 107-163 |
| `/data/clear`      | DELETE | 清除指定分片     | 166-199 |

#### 4.4.2 请求验证

使用 Pydantic 模型验证请求体（第 39-50 行）：

```python
class ShardListRequest(BaseModel):
    shard_ids: list[str]

class BatchShardRequest(ShardListRequest): ...
class ClearShardRequest(ShardListRequest): ...
```

#### 4.4.3 编解码路径

```
PUT /data/<id>:   request.get_data() -> orjson.loads -> deserialize_value -> rtensor.store
GET /data/<id>:   rtensor.fetch -> serialize_value -> orjson.dumps -> Response
POST /data/batch: [rtensor.fetch(id) for id] -> serialize_value -> orjson.dumps
DELETE /data/clear: [rtensor.remove(id) for id] -> storage_stats
```

______________________________________________________________________

### 4.5 引擎平面蓝图 (`guard/engine_blueprint.py`, 606 行)

引擎平面蓝图管理 TrainEngine / InferenceEngine 的生命周期和方法调用。其核心设计 约束是**所有引擎操作必须在单一线程中串行执行**，以保证
NCCL 兼容性。

#### 4.5.1 引擎线程模型

```
Flask 请求线程 (werkzeug threaded=True)
    |
    |--- /create_engine -->  _submit_to_engine_thread()
    |--- /call          -->  _submit_to_engine_thread()
    |--- /set_env       -->  _submit_to_engine_thread()
    |                             |
    |                             v
    |                    +------------------+
    |                    | EngineWorker     |  <-- daemon Thread (第 152 行)
    |                    | (单线程串行)      |
    |                    |                  |
    |                    | while True:      |
    |                    |   work = Q.get() |
    |                    |   result = fn()  |
    |                    |   future.set()   |
    |                    +------------------+
    |
    |<--- future.result() --- (阻塞等待)
```

关键点：

- `_engine_work_queue`（Queue）作为请求线程与引擎线程的通信管道
- `_submit_to_engine_thread`（第 157-167 行）封装 Future 模式
- 引擎线程 daemon=True，跟随主进程退出
- 关闭信号：向队列发送 None（第 253 行）

#### 4.5.2 路由

| 路由             | 方法 | 功能                     | 行号    |
| ---------------- | ---- | ------------------------ | ------- |
| `/set_env`       | POST | 在引擎线程中设置环境变量 | 270-298 |
| `/create_engine` | POST | 动态导入+实例化引擎      | 301-408 |
| `/call`          | POST | 调用引擎方法             | 411-606 |

#### 4.5.3 /call 请求处理流程

```
POST /call {method, engine_name, args, kwargs, rpc_meta}
    |
    v
1. Pydantic 验证 (CallEngineRequest)
    |
    v
2. deserialize_value(args/kwargs)     -- 恢复 Tensor/Dataclass
    |
    v
3. RTensor.localize(args/kwargs)      -- 拉取远端张量到本地
    |
    v
4. _submit_to_engine_thread()         -- 提交到引擎线程
    |
    v (引擎线程内)
5. _should_broadcast_payload()        -- 决定是否 DP 广播
    |-- TrainEngine 且已初始化 -> 默认广播
    |-- rpc_meta.broadcast 可覆盖
    |
    v
6. broadcast_tensor_container()       -- 广播到所有 DP rank
    |
    v
7. engine.method(*args, **kwargs)     -- 执行引擎方法
    |
    v
8. perf_tracer.trace_scope()          -- 性能追踪
    |
    v (回到请求线程)
9. RTensor.remotize(result)           -- 仅 DP head 执行
    |-- 非 DP head -> serialize(None)  -- 避免 RSS 泄漏 (#1209)
    |
    v
10. serialize_value(result) -> JSON 响应
```

**DP head 优化**（第 577-602 行）：只有 DP head rank 执行 `remotize`，非 DP head 直接返回 None。这是因为控制器的
`_collect_results` 会丢弃非 DP head 结果，若 非 DP head 也执行 remotize，则存储的张量永远不会被 `clear_batches`
清理，导致每步 训练 RSS 泄漏。

#### 4.5.4 性能追踪分类

引擎蓝图根据方法名自动分类性能追踪事件（第 503-529 行）：

| 类别        | 匹配关键词                                                                      |
| ----------- | ------------------------------------------------------------------------------- |
| `scheduler` | submit, wait                                                                    |
| `comm`      | update_weights, broadcast                                                       |
| `io`        | save, load                                                                      |
| `compute`   | train, eval, forward, compute, step, update, optimizer, zero_grad, lr_scheduler |
| `misc`      | 其他                                                                            |

#### 4.5.5 Hook 注册

`register_engine_hooks(state)`（第 175-188 行）在 Guard 上注册三类 hook：

| Hook 类型 | 注册函数                 | 功能                             |
| --------- | ------------------------ | -------------------------------- |
| health    | `_engine_health_hook`    | 返回 engine_count + engines 列表 |
| configure | `_engine_configure_hook` | 在引擎线程内设置随机种子         |
| cleanup   | `cleanup_engine_thread`  | 关闭引擎线程                     |
| cleanup   | `cleanup_engines`        | 销毁所有引擎实例                 |

______________________________________________________________________

### 4.6 RayRPCServer (`ray_rpc_server.py`, 222 行)

RayRPCServer 是 HTTP 通道的 Ray 替代方案，以 `@ray.remote` Actor 形式部署在 Ray 集群中。

#### 4.6.1 与 HTTP 通道的对比

| 维度     | HTTP 通道 (rpc_server)           | Ray 通道 (RayRPCServer)     |
| -------- | -------------------------------- | --------------------------- |
| 部署方式 | 独立进程 (werkzeug)              | Ray Actor                   |
| 调度器   | LocalScheduler / SlurmScheduler  | RayScheduler                |
| 张量后端 | HttpRTensorBackend               | RayRTensorBackend           |
| 张量存储 | 进程内 \_storage + HTTP API      | Ray Object Store (共享内存) |
| 线程模型 | 引擎线程(Queue) + Flask 请求线程 | Ray Actor 单线程串行        |
| 服务发现 | name_resolve (NFS/etcd)          | Ray Actor Handle            |

#### 4.6.2 核心方法

```python
# ray_rpc_server.py
class RayRPCServer:
    _engines: dict[str, TrainEngine | InferenceEngine]

    ping()                    -> "ok"              # 健康检查
    alloc_ports(count)        -> list[int]          # 端口分配
    configure(config, role, rank)                   # 配置 + 随机种子
    set_env(env)                                    # 设置环境变量
    create_engine(engine, *, engine_name, **kw)     # 动态导入+实例化
    call(method, *args, engine_name, rpc_meta, **kw)# 引擎方法调用
    destroy()                                       # 销毁引擎 + 退出 Actor
```

#### 4.6.3 call() 流程

```
call(method, *args, engine_name, rpc_meta, **kwargs)
    |
    v
1. 解析 engine (默认 fallback 到第一个引擎)
    |
    v
2. RTensor.localize(args/kwargs)          -- ray.get(ObjectRef)
    |
    v
3. _should_broadcast_payload()            -- 决定是否广播
    |
    v
4. broadcast_tensor_container()           -- DP 广播
    |
    v
5. engine.method(*args, **kwargs)         -- 执行
    |-- Future? -> .result()              -- 等待异步结果
    |
    v
6. RTensor.remotize(result, node_addr="") -- ray.put(tensor)
    |
    v
7. tensor_container_to(result, "cpu")     -- 模拟 HTTP 编解码行为
```

注意第 7 步：Ray 通道在返回前将结果移至 CPU，以保持与 HTTP 通道一致的语义。

______________________________________________________________________

### 4.7 HTTP RPC 入口 (`rpc_server.py`, 66 行)

rpc_server.py 是最薄的一层胶水代码，将 Guard + 数据蓝图 + 引擎蓝图组合为完整的 RPC 服务器。

```python
# rpc_server.py 第 33-62 行 (简化)
def main():
    parser = make_base_parser()
    parser.add_argument("--werkzeug-log-level", ...)
    args, _ = parser.parse_known_args()

    state = GuardState()
    bind_host = configure_state_from_args(state, args)

    app = create_app(state)           # Guard 核心路由
    app.register_blueprint(data_bp)   # 数据平面 /data/*
    app.register_blueprint(engine_bp) # 引擎平面 /create_engine, /call
    register_engine_hooks(state)      # 引擎 hook

    run_server(state, app, bind_host, args.port)
```

调度器通过以下方式启动 RPC 服务器：

```bash
python -m areal.infra.rpc.rpc_server \
    --experiment-name exp1 --trial-name trial1 \
    --role actor --worker-index 0
```

______________________________________________________________________

## 5. 关键设计决策

### 5.1 HTTP vs Ray 双后端

**问题**：SLURM 集群无法使用 Ray，Ray 集群不需要 HTTP 开销。

**决策**：通过 `RTensorBackend` Protocol 抽象后端差异，运行时自动选择。HTTP 后端 使用 `aiohttp` 异步并发获取 +
按节点分组批量请求（`max_shards_per_request=32`）。 Ray 后端直接走 Object Store 共享内存，零拷贝（同节点）。

### 5.2 引擎线程隔离

**问题**：NCCL 通信要求所有集合操作在同一线程中执行；Flask threaded 模式的请求 线程不确定。

**决策**：引入专用 `EngineWorker` daemon 线程 + Queue + Future 模式。所有引擎
操作（create、call、configure）均通过 `_submit_to_engine_thread` 串行化。数据
平面路由（/data/\*）不经过引擎线程，可并发处理。

### 5.3 DP Head Only Remotize

**问题**：非 DP head rank 的引擎方法返回值会被控制器丢弃，但 `remotize` 仍会在 `_storage` 中存储张量，且
`clear_batches` 不会清理这些分片 -- RSS 逐步泄漏。

**决策**（engine_blueprint.py 第 577-602 行）：非 DP head 的 `/call` 响应直接返回
`serialize_value(None)`，跳过 `remotize`。仅 DP head 或非 TrainEngine 执行完整的 remotize 路径。

### 5.4 Fetch Buffer 三点清理

**问题**：`_fetch_buffer` 缓存加速了重复读取，但若不及时清理则 RSS 增长。

**决策**：每步训练结束时，三条清理路径确保所有进程的 fetch buffer 被排空：

1. Controller: `RTensor.clear_node()`
1. 存储所有者: `rtensor.remove()`（同时弹出 `_fetch_buffer`）
1. 跨节点消费者: `clear_fetch_buffer(sids)` RPC

### 5.5 bfloat16 序列化

**问题**：NumPy 不原生支持 bfloat16，`tobytes()` 会失败。

**决策**：序列化时将 bfloat16 上转为 float32 的 numpy buffer，但在元数据中保留 原始 `torch.bfloat16`
dtype。反序列化时先以 float32 读取 buffer，再 `.to(dtype)` 恢复。

### 5.6 Guard Hook 可扩展架构

**问题**：Guard 需要在不同场景下（RPC 服务器、推理服务、训练服务）复用核心逻辑， 但各场景需要不同的引擎管理和数据存储行为。

**决策**：Guard 核心仅提供进程管理路由，通过 Hook 系统（health/configure/cleanup） 让蓝图注入扩展行为。Blueprint
模式使数据平面和引擎平面可独立组合。

______________________________________________________________________

## 6. 外部依赖与被依赖关系

### 6.1 向外依赖

| 依赖           | 用途                       | 文件                                                     |
| -------------- | -------------------------- | -------------------------------------------------------- |
| `flask`        | HTTP 路由 + Blueprint      | app.py, data_blueprint.py, engine_blueprint.py           |
| `werkzeug`     | WSGI 服务器                | app.py                                                   |
| `aiohttp`      | 异步 HTTP 张量获取         | rtensor.py                                               |
| `orjson`       | 高性能 JSON 编解码         | data_blueprint.py, rtensor.py                            |
| `pydantic`     | 请求/响应模型验证          | serialization.py, data_blueprint.py, engine_blueprint.py |
| `ray`          | Ray Actor + Object Store   | ray_rpc_server.py, rtensor.py, serialization.py          |
| `torch`        | 张量操作                   | rtensor.py, serialization.py                             |
| `numpy`        | 张量序列化桥接             | serialization.py                                         |
| `transformers` | Tokenizer/Processor 序列化 | serialization.py                                         |
| `PIL`          | 图像序列化 (VLM)           | serialization.py                                         |
| `zstandard`    | 大 Tokenizer 压缩 (可选)   | serialization.py                                         |

### 6.2 被谁依赖

| 调用方                                    | 使用内容                                   |
| ----------------------------------------- | ------------------------------------------ |
| `areal.infra.scheduler.local`             | 启动 `rpc_server.py` 作为 Worker 进程      |
| `areal.infra.scheduler.ray`               | 创建 `RayRPCServer` Actor                  |
| `areal.experimental.inference_service`    | 复用 Guard + data_blueprint                |
| `areal.experimental.training_service`     | 复用 Guard + serialization                 |
| `areal.experimental.engine.archon_engine` | 使用 `clear_fetch_buffer`                  |
| `areal.experimental.openai.proxy`         | 使用 `serialize_value`/`deserialize_value` |
| `areal.experimental.weight_update`        | 使用 `deserialize_value`                   |
| `areal.infra.data_service.rdataset`       | 类比 RTensor 的设计模式                    |

______________________________________________________________________

## 7. 扩展指南

### 7.1 添加新的序列化类型

1. 在 `serialization.py` 中定义新的 `SerializedXxx(BaseModel)`，设置 `type: Literal["xxx"]` 标签
1. 实现 `from_xxx()` 类方法（序列化）和 `to_xxx()` 实例方法（反序列化）
1. 在 `serialize_value()` 的 isinstance 链中添加分支（注意优先级）
1. 在 `deserialize_value()` 的 `value.get("type")` 链中添加分支

### 7.2 添加新的 Blueprint

1. 创建 `guard/xxx_blueprint.py`，定义 `xxx_bp = Blueprint("xxx", __name__)`
1. 在 `rpc_server.py` 中 `app.register_blueprint(xxx_bp)`
1. 如需扩展 `/health`、`/configure` 或关闭逻辑，使用 `state.register_xxx_hook()`

### 7.3 添加新的 RTensor 后端

1. 实现 `RTensorBackend` Protocol（fetch/store/delete 三个方法）
1. 在 `get_backend()` 中添加选择逻辑或通过 `set_backend()` 手动注入

______________________________________________________________________

## 8. 附录：关键行号速查

| 内容                               | 文件                      | 行号    |
| ---------------------------------- | ------------------------- | ------- |
| RTensorBackend Protocol 定义       | rtensor.py                | 25-66   |
| HttpRTensorBackend 批量获取        | rtensor.py                | 157-248 |
| RayRTensorBackend                  | rtensor.py                | 273-286 |
| 后端自动选择                       | rtensor.py                | 289-304 |
| \_fetch_buffer 清理不变式注释      | rtensor.py                | 307-319 |
| RTensor.remotize (含 padding 紧凑) | rtensor.py                | 391-444 |
| RTensor.localize (批量预取)        | rtensor.py                | 446-488 |
| \_storage 本地存储                 | rtensor.py                | 587-646 |
| SerializedTensor (bfloat16 处理)   | serialization.py          | 49-174  |
| SerializedDataclass (类型保留)     | serialization.py          | 272-336 |
| SerializedTokenizer (ZIP/Zstd)     | serialization.py          | 339-442 |
| serialize_value 递归入口           | serialization.py          | 554-653 |
| deserialize_value 递归入口         | serialization.py          | 656-778 |
| GuardState 定义                    | guard/app.py              | 41-115  |
| Flask 路由工厂 create_app          | guard/app.py              | 156-487 |
| run_server 服务生命周期            | guard/app.py              | 581-649 |
| 引擎线程初始化                     | guard/engine_blueprint.py | 114-154 |
| /call 路由 (DP head 优化)          | guard/engine_blueprint.py | 411-606 |
| RayRPCServer.call()                | ray_rpc_server.py         | 128-208 |
| 数据平面路由                       | guard/data_blueprint.py   | 58-199  |
