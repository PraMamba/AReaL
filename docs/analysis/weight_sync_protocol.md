# AReaL 权重同步协议深度分析

> 基于源码的详细分析，覆盖 NCCL Chunked 分块广播策略、显存带宽峰值消耗、
> Safetensors 文件系统同步一致性机制，以及两种路径的完整权衡。

---

## 目录

1. [架构总览](#1-架构总览)
2. [NCCL Chunked 分块传输协议](#2-nccl-chunked-分块传输协议)
   - 2.1 [进程组初始化](#21-进程组初始化)
   - 2.2 [分块策略详解](#22-分块策略详解)
   - 2.3 [Per-Bucket 广播协议](#23-per-bucket-广播协议)
   - 2.4 [接收端处理](#24-接收端处理)
   - 2.5 [显存带宽峰值消耗分析](#25-显存带宽峰值消耗分析)
   - 2.6 [三种引擎实现差异](#26-三种引擎实现差异)
3. [Safetensors 文件系统同步协议](#3-safetensors-文件系统同步协议)
   - 3.1 [写入路径](#31-写入路径)
   - 3.2 [读取路径](#32-读取路径)
   - 3.3 [多节点一致性保证](#33-多节点一致性保证)
   - 3.4 [低延迟机制](#34-低延迟机制)
4. [协调机制：Callback 与 Pause/Resume](#4-协调机制callback-与-pauseresume)
5. [XCCL vs Disk 路径权衡](#5-xccl-vs-disk-路径权衡)
6. [代码质量发现](#6-代码质量发现)
7. [设计总结](#7-设计总结)

---

## 1. 架构总览

AReaL 支持两种权重同步路径：

| 路径 | 传输介质 | 延迟 | 适用场景 |
|------|---------|------|---------|
| **XCCL (NCCL/HCCL)** | GPU→GPU 集合通信 | 低 | 同集群、有 NCCL 互联 |
| **Disk (Safetensors)** | 文件系统序列化 | 高 | 跨集群、无直接互联 |

通过 `WeightUpdateMeta.type` 字段选择（`"xccl"` 或 `"disk"`）。

### 协调拓扑

```
TrainController ──RPC──→ TrainEngine (rank 0..N-1)
       │                      │
       │               connect_engine()
       │                      │
       │               RolloutCallback (HTTP 代理)
       │                      │ _post_nowait() ← 非阻塞！
       ▼                      ▼
RolloutController ←──HTTP── Flask 回调服务器
       │
       ├──RPC──→ RemoteInfEngine (rank 0..M-1)
       │              │
       │         HTTP ──→ SGLang/vLLM (GPU Workers)
       │                      │
       └──────── NCCL ────────┘  ← 直接 GPU-to-GPU
                broadcast
```

**核心源码文件**：

| 文件 | 职责 |
|------|------|
| `areal/api/io_struct.py:164-258` | `WeightUpdateMeta`, `ParamSpec` 数据结构 |
| `areal/engine/fsdp_engine.py:999-1163` | FSDP 引擎权重同步实现 |
| `areal/engine/megatron_engine.py:1032-1347` | Megatron 引擎权重同步实现 |
| `areal/experimental/engine/archon_weight_sync.py` | Archon 引擎权重同步实现 |
| `areal/infra/controller/rollout_callback.py` | 训练→推理 HTTP 回调代理 |
| `areal/engine/core/distributed.py:25-90` | 自定义 NCCL 进程组创建 |
| `areal/infra/remote_inf_engine.py:870-989` | 推理侧权重更新入口 |

---

## 2. NCCL Chunked 分块传输协议

### 2.1 进程组初始化

**源码**: `areal/engine/fsdp_engine.py:1048-1078`

XCCL 进程组采用**星形拓扑**：训练引擎的 rank 0 作为唯一的广播源（src=0），
所有推理 Worker 作为接收方。

```python
def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta):
    assert meta.type == "xccl"

    # 1. Rank 0 发现自己的 IP 和空闲端口
    meta.nccl_master_address = self.weight_update_master_addr = gethostip()
    meta.nccl_master_port = self.weight_update_master_port = find_free_ports(1)[0]

    # 2. 绕过 torchrun 的弹性 Agent Store（会与自定义 TCP Store 冲突）
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)

    if dist.get_rank() == 0:
        # 3. 非阻塞通知推理侧初始化（HTTP → RolloutController → 所有推理 Worker）
        fut = self.rollout_engine.init_weights_update_group(meta)

        # 4. 创建 NCCL 进程组
        #    world_size = 推理 world_size + 1（训练 rank 0）
        self.weight_update_group = init_custom_process_group(
            backend=current_platform.communication_backend,  # NCCL/HCCL
            world_size=meta.alloc_mode.gen.world_size + 1,
            init_method=f"tcp://{meta.nccl_master_address}:{meta.nccl_master_port}",
            rank=0,
            group_name=meta.nccl_group_name,
            timeout=DIST_GROUP_DEFAULT_TIMEOUT,
        )

        # 5. 等待推理侧完成初始化
        fut.result()
```

**进程组创建** (`areal/engine/core/distributed.py:25-90`):

```python
def init_custom_process_group(backend, world_size, init_method, rank, group_name, ...):
    # 使用 PrefixStore 隔离命名空间，防止与训练主进程组冲突
    store = PrefixStore(group_name, store)
    pg, _ = _new_process_group_helper(world_size, rank, [], backend, store, ...)
    return pg
```

**关键设计点**：
- 使用 `PrefixStore(group_name, ...)` 隔离命名空间，避免与训练主进程组的 key 冲突
- 每次权重更新会话复用同一个进程组（初始化一次，后续跳过）
- `TORCHELASTIC_USE_AGENT_STORE=False` 是必要的 workaround——torchrun 会自动创建 Agent Store，
  与自定义 TCP Store 冲突

### 2.2 分块策略详解

**源码**: `areal/engine/fsdp_engine.py:1081-1137`

#### 分块参数

```python
# io_struct.py:173 — 默认 1024 MB (1 GB)
weight_chunked_mem_mb: int = 1024
```

可通过 `WeightUpdateMeta.from_fsdp_xccl(weight_chunked_mem_mb=512)` 调整。
官方 OOM 处理文档建议在显存紧张时降至 512 MB。

#### 贪心装箱算法

```python
@trace_perf("fsdp_engine.update_weights_from_distributed", category="comm")
def _update_weights_from_distributed(self, meta: WeightUpdateMeta):
    weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024  # 字节
    main_rank = dist.get_rank() == 0

    buffer_size = 0
    named_tensors: list[tuple[str, torch.Tensor]] = []

    for name, param in self._get_model_name_parameters():
        # ① 所有训练 rank 共同参与 DTensor → full tensor 重构
        tensor = self._get_full_tensor(param)

        # ② 只有 rank 0 累积到 bucket 中
        if not main_rank:
            continue

        tensor_size = tensor.numel() * tensor.element_size()

        # ③ 当累积超过阈值时，刷出当前 bucket
        if tensor_size + buffer_size > weight_chunked_mem_size:
            self._update_bucket_weights_from_distributed(meta, named_tensors)
            buffer_size = 0

        named_tensors.append((name, tensor))
        buffer_size += tensor_size

    # ④ 处理最后一个不满的 bucket
    if named_tensors:
        self._update_bucket_weights_from_distributed(meta, named_tensors)
```

**分块流程图**：

```
模型参数遍历:
  param_1 (200MB) → 累积: 200MB
  param_2 (300MB) → 累积: 500MB
  param_3 (600MB) → 500+600=1100 > 1024 → 刷出 [param_1, param_2] → 重置
                     累积 param_3: 600MB
  param_4 (300MB) → 累积: 900MB
  param_5 (200MB) → 900+200=1100 > 1024 → 刷出 [param_3, param_4] → 重置
                     累积 param_5: 200MB
  ...（最终刷出剩余）

每次刷出 = 一次 _update_bucket_weights_from_distributed() 调用
         = N 次 NCCL broadcast + 1 次 HTTP 回调
```

**重要细节**：

1. **所有训练 rank 参与 `_get_full_tensor()`**：FSDP2 的参数是 `DTensor`（分片张量），
   重构完整张量需要隐式的 `all_gather`。非 rank 0 的进程做了有用的工作（聚合自己的分片），
   但随后丢弃结果（`if not main_rank: continue`）。

2. **先检查后追加**：注意条件是 `tensor_size + buffer_size > threshold` 而非 `>=`。
   这意味着如果单个参数超过 `weight_chunked_mem_mb`（如大型 embedding 层），
   该参数仍会被追加到空 bucket 中并单独广播，实际 bucket 大小可能超过配置阈值。

3. **LoRA 优化**：当 `use_lora=True` 时，只遍历 `requires_grad=True` 的参数，
   大幅减少传输量。

### 2.3 Per-Bucket 广播协议

**源码**: `areal/engine/fsdp_engine.py:999-1046`

```python
def _update_bucket_weights_from_distributed(self, meta, named_tensors):
    if not named_tensors:
        return

    # ① 构建参数规格列表（名称、形状、dtype）
    param_specs = [
        ParamSpec(name=name, shape=tuple(tensor.shape),
                  dtype=str(tensor.dtype).split("torch.")[1])
        for name, tensor in named_tensors
    ]

    # ② 非阻塞通知推理侧：即将广播这些参数
    #    推理侧据此分配接收缓冲区并启动接收
    fut = self.rollout_engine.update_weights_from_distributed(meta, param_specs)

    # ③ 发起异步 NCCL broadcast（bucket 内所有参数并行）
    handles = []
    for _, tensor in named_tensors:
        handles.append(
            dist.broadcast(tensor, src=0, group=self.weight_update_group, async_op=True)
        )

    # ④ 等待所有 broadcast 完成
    for handle in handles:
        handle.wait()

    # ⑤ 等待推理侧确认接收完毕
    fut.result()

    # ⑥ 清空 bucket，准备下一轮
    named_tensors.clear()
```

**时序图**：

```
训练 Rank 0                           推理 Worker
     │                                     │
     │──── HTTP: param_specs ──────────→   │
     │     (非阻塞, Future)                │
     │                                     │ 分配接收缓冲区
     │                                     │
     │──── NCCL broadcast(param_1) ──→     │ dist.broadcast(recv, src=0)
     │──── NCCL broadcast(param_2) ──→     │ dist.broadcast(recv, src=0)
     │──── NCCL broadcast(param_3) ──→     │ dist.broadcast(recv, src=0)
     │     (async_op=True, 并行发起)        │
     │                                     │
     │ handle.wait() × 3                   │ 等待接收完成
     │                                     │
     │ ←─── HTTP: 确认完成 ───────────     │ load_weights()
     │     fut.result()                    │
     │                                     │
```

**关键**：`_post_nowait()` 的非阻塞设计是避免 NCCL 死锁的核心：

```python
# rollout_callback.py:23-27 — 关键注释
# IMPORTANT: Methods that return Future must be non-blocking to avoid deadlocks.
# NCCL operations are collective - both train and inference sides must participate
# concurrently. If these methods blocked, the train side couldn't start its NCCL
# operations while waiting for the inference side, causing a deadlock.
```

### 2.4 接收端处理

#### SGLang 后端

SGLang 通过单步 HTTP 请求 `/update_weights_from_distributed` 接收参数名、dtype、shape
和 group_name。服务器内部执行 NCCL receive 并将权重加载到模型中。

#### vLLM 后端

**源码**: `areal/engine/vllm_ext/vllm_worker_extension.py:119-150`

vLLM 使用两步协议：
1. `set_weight_meta` — 存储参数元数据
2. `update_weight_xccl` — 逐参数同步广播并加载

```python
# vLLM Worker 接收循环（简化）
for name, dtype, shape in zip(names, dtypes, shapes):
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.broadcast(tensor, src=0, group=group, async_op=False)  # 同步！
    self.model_runner.model.load_weights(weights=[(name, tensor)])
```

**注意**：vLLM 使用 `async_op=False`（同步广播），且逐参数串行处理。
这比训练侧的批量异步广播效率低——没有 broadcast 与 load_weights 之间的流水线重叠。

### 2.5 显存带宽峰值消耗分析

#### 训练侧（发送方）

以 70B 参数 bf16 模型为例（总权重 ~140 GB）：

```
               训练侧 Rank 0 显存峰值
┌──────────────────────────────────────────────────┐
│                                                  │
│  模型参数 (FSDP 分片)         ~140/N GB           │
│  ─────────────────────                           │
│  + _get_full_tensor() 临时缓冲  ~param_size      │
│    (DTensor all_gather 结果)                      │
│  ─────────────────────                           │
│  + Bucket 累积缓冲              ≤ 1 GB (默认)     │
│    (weight_chunked_mem_mb)                        │
│  ─────────────────────                           │
│  = 额外峰值开销               ≈ 1 GB + param_size │
│                                                  │
└──────────────────────────────────────────────────┘
```

**详细分析**：

| 开销来源 | 大小 | 生命周期 | 说明 |
|----------|------|---------|------|
| FSDP all_gather 临时张量 | 单个参数大小 | 每参数一次 | `_get_full_tensor()` 触发 |
| Bucket 累积张量 | ≤ `weight_chunked_mem_mb` | 刷出时释放 | 贪心装箱 |
| NCCL 发送缓冲区 | NCCL 内部管理 | 透明 | 通常 ~几十 MB |

**NIC 带宽消耗**（70B bf16，8 训练 rank + 8 推理 Worker）：

```
Rank 0 的 NIC 总流量:
  ① FSDP all_gather 接收: ~140 GB × (N-1)/N ≈ ~122 GB（从其他 7 个 rank）
  ② NCCL broadcast 发送:  ~140 GB（到 8 个推理 Worker）
  ───────────────────
  总计: ~262 GB 通过 Rank 0 的 NIC

  在 200 Gbps (25 GB/s) NIC 下: ~262/25 ≈ 10.5 秒
  在 400 Gbps (50 GB/s) NIC 下: ~262/50 ≈ 5.2 秒
```

**Rank 0 是带宽瓶颈**：所有训练 rank 参与 all_gather，但只有 rank 0 参与跨组 broadcast。
这造成了 fan-in（all_gather 接收）+ fan-out（broadcast 发送）的双重压力。

#### 推理侧（接收方）

```
               推理侧 Worker 显存峰值
┌──────────────────────────────────────────────────┐
│                                                  │
│  模型参数 (已加载)              ~140/M GB          │
│  ─────────────────────                           │
│  + 接收缓冲区                  ~单个参数大小       │
│    (torch.empty per param)                        │
│  ─────────────────────                           │
│  = 额外峰值开销               ≈ 几百 MB           │
│                                                  │
└──────────────────────────────────────────────────┘
```

vLLM 逐参数处理，每次只分配一个参数大小的接收缓冲区；SGLang 可能批量处理。

#### Chunk Size 与峰值显存的关系

```
chunk_size (MB)    训练侧额外峰值    Bucket 轮次 (70B)    延迟影响
──────────────────────────────────────────────────────────────
256                ~256 MB           ~547 轮              多 HTTP 往返
512                ~512 MB           ~274 轮              中等
1024 (默认)         ~1 GB            ~137 轮              较少
2048               ~2 GB            ~69 轮               最少
```

每增大一倍 chunk_size，bucket 轮次减半（减少 HTTP 回调开销），
但训练侧峰值显存增加一倍。默认 1 GB 是合理的平衡点。

### 2.6 三种引擎实现差异

| 维度 | FSDP | Megatron | Archon |
|------|------|----------|--------|
| 参数重构 | `DTensor.full_tensor()` (隐式 all_gather) | `all_gather_param()` + 格式转换 | `_get_full_tensor()` (DTensor) |
| 广播参与者 | rank 0 only | PP head only | rank 0 only |
| 锁保护 | **无锁** | `DistributedLock` (acquire/release) | `DistributedLock` (with 语句) |
| 格式转换 | 无需（HF 原生格式） | `convert_to_hf()` (名称/形状转换) | `state_dict_adapter` |
| MoE 支持 | N/A | `_update_bucket_expert_weights_from_distributed()` + expert all_gather | 通过 adapter |
| FP8 支持 | N/A | `dequantize_param_if_fp8()` | N/A |

**Megatron 额外复杂性**：
- Pipeline 并行：只有 PP head（PP rank 0）参与跨组 broadcast
- Expert 并行（MoE）：expert 权重需先在 EP 组内 `all_gather`，再转换格式后 broadcast
- FP8 量化：传输前需反量化到 fp16/bf16

---

## 3. Safetensors 文件系统同步协议

### 3.1 写入路径

**源码**: `areal/engine/fsdp_engine.py:1139-1163`

```python
@trace_perf("fsdp_engine.update_weights_from_disk", category="io")
def _update_weights_from_disk(self, meta: WeightUpdateMeta):
    fut = Future()

    # ① Rank 0 非阻塞通知推理侧准备加载
    if dist.get_rank() == 0:
        fut = self.rollout_engine.update_weights_from_disk(meta)

    # ② 所有训练 rank 保存模型到 HF 格式
    self._save_model_to_hf(meta.path, self.tokenizer, self.processor)
    # 内部包含 dist.barrier()

    # ③ Rank 0 发布完成信号（通过 name_resolve）
    if dist.get_rank() == 0:
        update_name = names.update_weights_from_disk(
            self.config.experiment_name,
            self.config.trial_name,
            self.get_version(),
        )
        name_resolve.add(update_name, str(datetime.now().timestamp()), keepalive_ttl=120)

        # ④ 等待推理侧确认加载完成
        fut.result()

    current_platform.synchronize()
    dist.barrier(group=self.cpu_group)
```

#### `_save_model_to_hf` 保存过程

**源码**: `areal/engine/fsdp_engine.py:1165-1188`

```python
def _save_model_to_hf(self, path, tokenizer, processor):
    # FSDP2: 聚合完整 state_dict 到 CPU（rank 0）
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = get_model_state_dict(self.model, options=options)

    # 只有 rank 0 写入文件
    if dist.get_rank() == 0:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path, state_dict=state_dict)  # → safetensors
        self.model_config.save_pretrained(path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)

    dist.barrier(group=self.cpu_group)  # 同步所有 rank
```

**写入格式**：HuggingFace `save_pretrained` 默认使用 **safetensors** 格式，
将模型权重以零拷贝 mmap 友好的方式写入磁盘。

### 3.2 读取路径

**源码**: `areal/infra/remote_inf_engine.py:1279-1324`

```python
def _update_weights_from_disk(meta, addresses, backend, ...):
    # ① 等待训练侧发布完成信号
    name_resolve.wait(update_name, timeout=120)

    # ② 构建后端特定的 HTTP 请求
    reqs = backend.build_disk_weight_update_requests(meta)

    # ③ 并发发送到所有推理服务器
    async def _fn():
        await asyncio.gather(*[
            arequest_with_retry(addr, req.endpoint, req.payload, ...)
            for addr, req in zip(addresses, reqs)
        ])

    # ④ 清理检查点文件（可选）
    if meta.clear_checkpoint_after_load:
        shutil.rmtree(meta.path, ignore_errors=True)
```

### 3.3 多节点一致性保证

Disk 路径依赖两个同步原语来保证多节点并发读取的一致性：

#### 同步原语 1: `dist.barrier()` — 训练侧写入完整性

```
训练 Rank 0                训练 Rank 1..N-1
    │                           │
    │ get_model_state_dict()    │ get_model_state_dict()
    │   (FSDP all_gather)       │   (FSDP all_gather)
    │                           │
    │ save_pretrained()         │ (无操作)
    │   → safetensors 写入      │
    │                           │
    ├───── dist.barrier() ──────┤  ← 保证写入完成后再继续
    │                           │
```

`dist.barrier()` 确保 rank 0 的 `save_pretrained()` 完成后，
所有训练 rank 才继续执行。这保证文件写入在信号发布前完成。

#### 同步原语 2: `name_resolve` — 跨进程组发布/等待

```
训练 Rank 0                          推理 Worker
    │                                    │
    │ name_resolve.add(                  │ name_resolve.wait(
    │   "update_weights/.../v5",         │   "update_weights/.../v5",
    │   timestamp,                       │   timeout=120
    │   keepalive_ttl=120                │ )
    │ )                                  │
    │                                    │ ← 收到信号后开始加载
    │                                    │
    │                                    │ load_weights(meta.path)
    │                                    │
```

#### 一致性分析

```
时间线:

  T1: Rank 0 完成 save_pretrained()
  T2: dist.barrier() 通过 ← 保证 T1 已完成
  T3: name_resolve.add() 发布信号
  T4: 推理 Worker 收到信号
  T5: 推理 Worker 开始读取文件

  一致性保证: T5 > T4 > T3 > T2 > T1
  即: 读取一定发生在写入完全完成之后
```

**潜在风险 — 分布式文件系统缓存一致性**：

上述保证假设文件系统提供 **read-after-write 一致性**。
这在本地 NVMe/SSD 上成立，但在某些分布式文件系统上可能不成立：

| 文件系统 | 一致性 | 风险 |
|----------|--------|------|
| 本地 NVMe/SSD | 强一致 | 无 |
| NFS (默认) | 弱一致（属性缓存） | 可能读到陈旧数据 |
| Lustre | 取决于挂载选项 | 需要 `lazystatfs=0` |
| GPFS/Spectrum Scale | 强一致 | 无 |
| CephFS | 强一致 | 无 |

**缓解措施**：HuggingFace `save_pretrained` 内部调用 Python 的 `open().write().close()`，
会触发内核级 flush，但不保证 `fsync()`。对于 NFS，可能需要额外的 `os.sync()` 调用。

### 3.4 低延迟机制

#### 版本化路径

**源码**: `areal/api/io_struct.py:185-197`

```python
def with_version(self, version: int) -> "WeightUpdateMeta":
    new_meta = copy.copy(self)
    new_meta.version = version
    if self.path is not None:
        base_dir = os.path.dirname(self.path)
        new_meta.path = os.path.join(base_dir, f"weight_update_v{version}")
    return new_meta
```

每个版本使用独立目录 `weight_update_v{version}`，避免写入与清理的竞争条件。

#### 并发读取

推理侧使用 `asyncio.gather` 并发请求所有推理服务器同时加载：

```python
# remote_inf_engine.py — 所有服务器并行加载
await asyncio.gather(*[
    arequest_with_retry(addr, req.endpoint, req.payload, ...)
    for addr in self.addresses
])
```

#### 异步清理

加载完成后的文件清理在回调中异步执行：

```python
# 不阻塞训练主流程
if meta.clear_checkpoint_after_load:
    shutil.rmtree(meta.path, ignore_errors=True)
```

#### CPU Offload

`StateDictOptions(cpu_offload=True)` 将 state_dict 聚合到 CPU 内存，
避免 GPU 显存峰值。代价是增加 CPU→磁盘的写入时间。

---

## 4. 协调机制：Callback 与 Pause/Resume

### Pause/Resume 协议

**源码**: `areal/engine/fsdp_engine.py:1089-1092`, `areal/infra/remote_inf_engine.py:1172-1180`

在任何权重更新（XCCL 或 Disk）之前，训练侧必须暂停推理生成：

```python
# 训练侧（fsdp_engine.py:1089-1092）
if dist.get_rank() == 0:
    self.rollout_engine.pause_generation()  # 同步 HTTP → 推理服务器
dist.barrier(group=self.cpu_group)          # 等待所有 rank

# ... 执行权重更新 ...

dist.barrier(group=self.cpu_group)
if dist.get_rank() == 0:
    self.rollout_engine.continue_generation()  # 恢复生成
```

`pause_generation()` 在推理侧的行为：

```python
# remote_inf_engine.py:1172-1180
def pause_generation(self):
    pause_req = self.backend.get_pause_request()
    self._run_request_on_all_servers(pause_req)  # HTTP → 所有推理服务器

    # 等待宽限期（启发式）
    time.sleep(self.config.pause_grace_period)  # 默认 0.0s
```

**设计意图**：暂停防止推理服务器在权重更新期间使用不一致的模型状态进行推理。

**注意**: `pause_grace_period` 默认为 0.0，在高负载下可能不足以等待所有 in-flight 请求完成。

### RolloutCallback 的非阻塞设计

**源码**: `areal/infra/controller/rollout_callback.py:60-82`

```python
def _post_nowait(self, endpoint, payload) -> Future[dict]:
    """非阻塞 HTTP POST — 提交到后台线程池，立即返回 Future。

    这对 NCCL 协调至关重要：训练和推理双方必须同时参与
    集合操作。如果这些方法阻塞，训练侧就无法在等待推理侧
    的同时启动自己的 NCCL 操作，导致死锁。"""
    return get_executor().submit(self._post, endpoint, payload)
```

这是整个权重同步协议中最关键的并发设计。每个 bucket 的时序：

```
训练 Rank 0:                     推理侧 (via callback):
fut = rollout_engine              ← HTTP POST (后台线程)
  .update_weights_from_distributed()
    │                              │
    ├─ dist.broadcast(p1, async)   │ dist.broadcast(p1, recv)
    ├─ dist.broadcast(p2, async)   │ dist.broadcast(p2, recv)
    ├─ dist.broadcast(p3, async)   │ dist.broadcast(p3, recv)
    │                              │   ↑ 双方同时参与 NCCL 集合操作
    ├─ handle.wait() × 3           │
    │                              │ load_weights(...)
    └─ fut.result()                │ ← HTTP 响应
```

如果 `_post_nowait` 改为阻塞：训练侧会等待推理侧响应 → 推理侧等待 NCCL broadcast →
训练侧无法发起 broadcast → **死锁**。

---

## 5. XCCL vs Disk 路径权衡

| 维度 | XCCL 路径 | Disk 路径 |
|------|----------|----------|
| **延迟** | 低（直接 GPU→GPU） | 高（序列化→写盘→读盘→反序列化） |
| **训练侧峰值显存** | +1 GB（chunk 缓冲）+ DTensor all_gather | CPU 上 full state_dict |
| **推理侧峰值显存** | +单参数接收缓冲 | +模型加载临时内存 |
| **NIC 带宽** | Rank 0 NIC 饱和风险 | 无（走磁盘 I/O） |
| **磁盘 I/O** | 无 | 全模型写入/读取 |
| **基础设施要求** | 训练-推理 NCCL 互联 | 共享文件系统 |
| **LoRA 支持** | 完整（vLLM）/ 部分（SGLang） | 完整 |
| **容错** | NCCL 超时 → 硬失败 | 文件系统故障 → 可重试 |
| **复杂性** | 高（进程组、回调协调） | 低（save/load + 信号） |
| **适用规模** | ≤70B 参数效果好 | 任意规模（受 I/O 限制） |

### 延迟估算对比（70B bf16 模型）

| 阶段 | XCCL | Disk |
|------|------|------|
| 参数聚合 | ~3s (FSDP all_gather) | ~5s (full state_dict + CPU offload) |
| 传输/写入 | ~5s (NIC broadcast @ 400Gbps) | ~20s (写入 NVMe @ 7 GB/s) |
| 信号/协调 | ~0.1s (HTTP 回调) | ~0.1s (name_resolve) |
| 加载 | 包含在传输中 | ~15s (读取 + 反序列化) |
| **总计** | **~8s** | **~40s** |

> 注：以上为粗略估算，实际取决于硬件配置、网络拓扑和模型结构。

---

## 6. 代码质量发现

### Critical 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 1 | `api/io_struct.py` | 166 | `Literal["disk", "nccl"]` 类型标注错误，应为 `Literal["disk", "xccl"]`。所有工厂方法和消费代码使用 `"xccl"`，但类型声明写的是 `"nccl"` |

### High 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 2 | `engine/fsdp_engine.py` | 999-1046 | FSDP 引擎的 bucket broadcast 无锁保护，与 Megatron/Archon 引擎不一致 |
| 3 | `engine/megatron_engine.py` | 1041/1068, 1239/1259 | `engine_lock.acquire()/release()` 未使用 `try/finally`，异常时锁永不释放 |
| 4 | `engine/fsdp_engine.py` | 1089-1092 | `pause_generation()` 依赖启发式 `pause_grace_period`（默认 0.0s），无法保证 in-flight 请求全部完成 |

### Medium 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 5 | `engine/fsdp_engine.py` | 1034-1044 | NCCL broadcast 部分失败时无回滚机制，推理引擎可能处于新旧权重混合状态 |
| 6 | `engine/fsdp_engine.py` | 1120 | 单参数超过 `weight_chunked_mem_mb` 时，实际 bucket 大小不受限 |
| 7 | `engine/fsdp_engine.py` | 1139-1163 | Disk 路径无 `fsync()`，在 NFS 等弱一致文件系统上可能读到不完整数据 |
| 8 | `engine/vllm_ext/vllm_worker_extension.py` | 119-150 | vLLM 接收端同步广播 + 逐参数加载，无流水线重叠 |

### 设计优化建议

| 优先级 | 建议 | 预期收益 |
|--------|------|---------|
| 高 | **双缓冲流水线**：bucket N broadcast 的同时 all_gather bucket N+1 | ~50% 时间缩减 |
| 高 | **分布式 broadcast**：每个训练 rank 广播自己的 DTensor 分片，推理侧本地重构 | 消除 rank 0 NIC 瓶颈 |
| 中 | Disk 路径添加 `os.fsync()` | 强一致性保证 |
| 中 | vLLM 接收端改用 `async_op=True` + 流水线 | 减少接收延迟 |
| 低 | Delta 压缩：只传输显著变化的参数 | 后期训练大幅减少传输量 |

---

## 7. 设计总结

### NCCL Chunked 分块策略核心

```
                    分块传输时间线 (70B 模型, 1GB chunk)

    Bucket 1        Bucket 2        Bucket 3        ...
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │all_gather│    │all_gather│    │all_gather│
    │  (FSDP) │    │  (FSDP) │    │  (FSDP) │
    ├─────────┤    ├─────────┤    ├─────────┤
    │broadcast │    │broadcast │    │broadcast │     ← 每 bucket 内
    │ (NCCL)  │    │ (NCCL)  │    │ (NCCL)  │       参数并行 async
    ├─────────┤    ├─────────┤    ├─────────┤
    │fut.result│    │fut.result│    │fut.result│     ← bucket 间串行
    └─────────┘    └─────────┘    └─────────┘       （无流水线）

    总轮次: ~140 GB / 1 GB = 140 轮
    每轮: all_gather + broadcast + HTTP 确认
```

### Safetensors 一致性保证核心

```
    训练侧                     推理侧
      │                          │
      │ save_pretrained()        │
      │   (safetensors 写入)     │
      │                          │
      ├── dist.barrier() ────────┤  ← 保证写入完成
      │                          │
      │ name_resolve.add()       │
      │   (发布完成信号)          │
      │                          │ name_resolve.wait()
      │                          │   (等待信号)
      │                          │
      │                          │ load_weights()
      │                          │   (safetensors 读取)
      │                          │
      │ ←── fut.result() ────────│  ← 确认加载完成
      │                          │
      │                          │ shutil.rmtree()
      │                          │   (异步清理)
```

### 一句话总结

> AReaL 的权重同步采用**分 bucket 的 NCCL 异步广播 + 非阻塞 HTTP 回调**避免死锁，
> 以 1 GB 默认 chunk size 在**显存峰值（+1GB）与传输轮次（~140 轮/70B 模型）之间取得平衡**；
> 文件系统路径通过 **dist.barrier() + name_resolve 发布/等待**提供跨进程组一致性，
> 但在弱一致文件系统上需要额外的 fsync 保证。两条路径通过 `WeightUpdateMeta.type` 一键切换。
