# FSDP2 Unshard 机制、内存布局对齐与异构 Process Group 桥接深度分析

> 基于源码的底层分析，覆盖 FSDP2 参数切片的解除机制、展平张量到 HF 格式的还原过程、
> 以及训练与推理集群间 NCCL Communicator 的 P2P 通道建立。

---

## 目录

1. [FSDP2 参数存储模型：DTensor 而非 FlatParameter](#1-fsdp2-参数存储模型dtensor-而非-flatparameter)
2. [底层切片汇聚：`_get_full_tensor()` 的 Unshard 机制](#2-底层切片汇聚_get_full_tensor-的-unshard-机制)
3. [32B/70B 模型的 OOM 防护策略](#3-32b70b-模型的-oom-防护策略)
4. [内存布局对齐：DTensor → HuggingFace 格式的还原路径](#4-内存布局对齐dtensor--huggingface-格式的还原路径)
5. [异构 Process Group 桥接：双独立 NCCL Communicator 的 P2P 通道](#5-异构-process-group-桥接双独立-nccl-communicator-的-p2p-通道)
6. [完整数据流：从 FSDP2 Shard 到 vLLM 模型权重](#6-完整数据流从-fsdp2-shard-到-vllm-模型权重)
7. [代码质量发现](#7-代码质量发现)
8. [设计总结](#8-设计总结)

---

## 1. FSDP2 参数存储模型：DTensor 而非 FlatParameter

### 1.1 FSDP1 vs FSDP2 的关键差异

| 维度 | FSDP1 (`FullyShardedDataParallel`) | FSDP2 (`fully_shard`) |
|------|-----------------------------------|----------------------|
| 参数存储 | **FlatParameter**（1D 展平拼接） | **DTensor**（保持原始形状） |
| 分片语义 | 扁平化后按位置切分 | `Shard(0)` 沿第 0 维切分 |
| 形状信息 | 需要手动记录和还原 | **DTensor 自动维护** |
| 与 TP 组合 | 复杂（需要额外的 flatten/unflatten 层） | **原生支持**（多维 placement） |

### 1.2 AReaL 中 FSDP2 的参数结构

**源码**: `areal/engine/fsdp_utils/__init__.py:52-88` (`apply_fsdp2`)

```python
def apply_fsdp2(model, fsdp_kwargs, wrap_policy):
    # 逐层包装 transformer block
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # 包装根模型（捕获剩余参数）
    fully_shard(model, **fsdp_kwargs)
```

`fully_shard()` 将每个 `nn.Parameter` **原地转换为 DTensor**：

```
包装前:
  model.layers.0.attention.wq.weight: nn.Parameter [4096, 4096]  ← 普通张量

包装后:
  model.layers.0.attention.wq.weight: DTensor [4096, 4096]
    .device_mesh = DeviceMesh["dp_sp"] (8 ranks)
    .placements = (Shard(0),)  ← 沿 dim 0 分片
    ._local_tensor: Tensor [512, 4096]  ← 本地分片 (4096/8=512)
```

**关键**: 参数的**全局形状**（[4096, 4096]）被 DTensor 元数据完整保留。
本地只存储分片 `[4096/N, 4096]`，但调用者看到的 `.shape` 仍然是 `[4096, 4096]`。

### 1.3 当 TP + FSDP2 组合时的多维 DTensor

**源码**: `areal/engine/fsdp_utils/parallel.py:368-394`

```python
def parallelize_model(model, config, model_config, nd_device_mesh, parallel_helper, ...):
    if tp_enabled:
        apply_non_moe_tp(model, model_config, parallel_helper, nd_device_mesh["tp"])
    # TP 先应用 → 参数变为 TP DTensor

    fsdp_kwargs = {"mesh": nd_device_mesh["dp_sp"], ...}
    apply_fsdp2(model, fsdp_kwargs, wrap_policy)
    # FSDP2 再包装 → 参数变为 TP+FSDP 双层 DTensor
```

应用 TP 后再应用 FSDP2，参数变为**多维 DTensor**：

```
apply_non_moe_tp() 后:
  wq.weight: DTensor [4096, 4096]
    .device_mesh = DeviceMesh["tp"] (2 ranks)
    .placements = (Shard(0),)  ← ColwiseParallel: 按头数切分
    ._local_tensor: Tensor [2048, 4096]

apply_fsdp2() 后:
  wq.weight: DTensor [4096, 4096]
    .device_mesh = DeviceMesh["dp_sp", "tp"] 或复合 mesh
    .placements = (Shard(0), Shard(0))  ← FSDP shard + TP shard
    ._local_tensor: Tensor [256, 4096]  ← 4096/(8*2)=256
```

---

## 2. 底层切片汇聚：`_get_full_tensor()` 的 Unshard 机制

### 2.1 核心实现

**源码**: `areal/engine/fsdp_engine.py:979-997`

```python
def _get_full_tensor(self, param: nn.Parameter) -> torch.Tensor:
    """Get full tensor from a parameter, handling DTensor and CPU offloaded tensors."""
    tensor = param.data
    if isinstance(tensor, DTensor):
        # 路径 A: GPU 上的 DTensor → 直接 full_tensor()
        if tensor.device.type != "cpu":
            return tensor.full_tensor()

        # 路径 B: CPU offload 的 DTensor → 重建后 full_tensor()
        temp_dtensor = DTensor.from_local(
            tensor.to_local(),        # 提取本地分片（CPU 张量）
            device_mesh=tensor.device_mesh,
            placements=tensor.placements,
        )
        return temp_dtensor.full_tensor()
    else:
        # 路径 C: 非 DTensor → 直接移到 GPU
        if tensor.device.type == "cpu":
            tensor = tensor.to(current_platform.device_type)
        return tensor
```

### 2.2 `DTensor.full_tensor()` 的内部机制

`full_tensor()` 是 PyTorch DTensor API 的核心方法。它检查 `placements` 元组中的每个维度：

```
对于 placements = (Shard(0),) 在 dp_sp mesh (8 ranks) 上:

  Rank 0: _local_tensor [512, 4096]  ─┐
  Rank 1: _local_tensor [512, 4096]  ─┤
  Rank 2: _local_tensor [512, 4096]  ─┤  NCCL AllGather
  Rank 3: _local_tensor [512, 4096]  ─┤  ──────────────→  full_tensor [4096, 4096]
  Rank 4: _local_tensor [512, 4096]  ─┤                   (每个 rank 都有)
  Rank 5: _local_tensor [512, 4096]  ─┤
  Rank 6: _local_tensor [512, 4096]  ─┤
  Rank 7: _local_tensor [512, 4096]  ─┘
```

- `Shard(dim)` placement → 触发 **NCCL all_gather** 在对应 mesh 维度上
- `Replicate()` placement → 无通信
- 多个 `Shard` placement → **逐维度串行 all_gather**（先 FSDP 维度，再 TP 维度）

**返回值**: 一个**普通 `torch.Tensor`**（非 DTensor），保持原始参数形状，
内存连续。这就是为什么 AReaL 不需要任何 flatten/unflatten 操作。

### 2.3 CPU Offload 路径的特殊处理

当 `is_offload=True` 时，FSDP2 将参数移到 CPU。CPU 上的 DTensor 无法直接调用
`full_tensor()`（NCCL 需要 GPU 内存）。

解决方案（line 987-993）：

```
CPU DTensor:
  ._local_tensor: CPU Tensor [512, 4096]
  .device_mesh: 指向 GPU mesh
  .placements: (Shard(0),)

  ↓ tensor.to_local()         → CPU Tensor [512, 4096]（纯本地分片）

  ↓ DTensor.from_local(       → 新 DTensor（GPU 上）
      local_tensor,            → 隐式 CPU→GPU 拷贝
      device_mesh=...,
      placements=...,
  )

  ↓ .full_tensor()            → NCCL all_gather → 完整 GPU Tensor [4096, 4096]
```

**`torch_memory_saver` 的配合** (`fsdp_engine.py:457-468`):

```python
def update_weights(self, meta: WeightUpdateMeta):
    if meta.type == "xccl":
        tms_context = (
            torch_memory_saver.disable()  # ← 临时禁用 GPU→CPU 自动卸载
            if self.is_offload and not torch.version.hip
            else nullcontext()
        )
        with tms_context:
            self._update_weights_from_distributed(meta)
```

`torch_memory_saver.disable()` 防止 TMS 的 LD_PRELOAD 钩子在 weight sync 期间
把 GPU 上的张量自动移到 CPU，确保 `_get_full_tensor()` 能正常执行 NCCL 操作。

---

## 3. 32B/70B 模型的 OOM 防护策略

### 3.1 峰值显存分析

对于一个形状为 `[H, H]` 的参数（如 70B 模型的 attention 矩阵，H=8192）：

```
参数全尺寸: H × H × 2 bytes (bf16) = 8192 × 8192 × 2 = 128 MB

FSDP 8-way 分片后:
  每 rank 本地分片: 128 / 8 = 16 MB

full_tensor() 调用期间:
  本地分片:   16 MB  (原有)
  all_gather 输出: 128 MB  (新分配)
  ──────────────────────
  峰值增量:  128 MB  (= 原参数全尺寸)
```

**对于 70B 模型的单个参数**，峰值增量 = 参数全尺寸。但由于分块传输，
系统**不会同时 unshard 所有参数**。

### 3.2 三层 OOM 防护机制

#### 防护层 1: 逐参数 Unshard + Bucket 累积

**源码**: `fsdp_engine.py:1094-1129`

```python
weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024  # 默认 1 GB

for name, param in param_iterator:
    tensor = self._get_full_tensor(param)  # 逐个 unshard

    if not main_rank:
        continue  # 非 rank 0: 参与 all_gather 但不累积

    tensor_size = tensor.numel() * tensor.element_size()

    if tensor_size + buffer_size > weight_chunked_mem_size:
        self._update_bucket_weights_from_distributed(meta, named_tensors)
        buffer_size = 0  # 刷出后释放 bucket 内张量的引用

    named_tensors.append((name, tensor))
    buffer_size += tensor_size
```

**关键**: `_get_full_tensor()` 是逐参数调用的，每次只 unshard 一个参数。
当 bucket 累积到 `weight_chunked_mem_mb`（默认 1 GB）时刷出并释放。

**Rank 0 的峰值额外显存**:

```
= max(单参数 full_tensor 大小) + bucket 累积大小
≤ max_param_size + weight_chunked_mem_mb
≈ 128 MB (largest param) + 1024 MB (bucket)
≈ ~1.15 GB
```

**非 Rank 0 的峰值额外显存**:

```
= max(单参数 full_tensor 大小)  ← 参与 all_gather 产生临时完整张量
= ~128 MB (largest param)
  → 但因为 `continue` 后不保持引用，Python GC 可在下次迭代回收
```

#### 防护层 2: `reshard_after_forward` 策略

**源码**: `fsdp_utils/parallel.py:392`

```python
fsdp_kwargs = {
    "mesh": nd_device_mesh["dp_sp"],
    "reshard_after_forward": True,  # ← 默认开启
}
```

训练期间，FSDP2 在前向传播后自动 reshard（all_gather 的逆操作），
释放临时的完整参数。权重同步发生在训练步之间（`pause()` → `update_weights()` → `resume()`），
此时模型参数处于 sharded 状态，显存占用最小。

#### 防护层 3: Disk 路径的 CPU Offload

**源码**: `fsdp_engine.py:1178-1179`

```python
options = StateDictOptions(full_state_dict=True, cpu_offload=True)
state_dict = get_model_state_dict(self.model, options=options)
```

Disk 路径使用 `cpu_offload=True`，将完整 state_dict 聚合到 **CPU 内存**而非 GPU 显存。
对于 70B bf16 模型（~140 GB），这需要 ~140 GB CPU 内存但不占用 GPU 显存。

### 3.3 不同模型规模的峰值显存估算

| 模型规模 | 最大单参数 | FSDP 分片数 | Rank 0 峰值增量 | 非 Rank 0 峰值增量 |
|----------|-----------|-----------|----------------|-------------------|
| 7B (bf16) | ~128 MB | 8 | ~1.15 GB | ~128 MB |
| 32B (bf16) | ~256 MB | 16 | ~1.25 GB | ~256 MB |
| 70B (bf16) | ~512 MB | 32 | ~1.5 GB | ~512 MB |

> Bucket 大小可调：`weight_chunked_mem_mb=512` 将 Rank 0 峰值降至 ~0.6-1.0 GB。

---

## 4. 内存布局对齐：DTensor → HuggingFace 格式的还原路径

### 4.1 核心问题

FSDP2 的 DTensor **保持原始形状**，不像 FSDP1 那样展平。但存在两个对齐问题：

1. **参数命名**：内部命名 vs HuggingFace 命名
2. **参数结构**：某些引擎（如 Megatron/Archon）将多个 HF 参数合并为一个

### 4.2 FSDP Engine 的直接路径（无展平问题）

**源码**: `fsdp_engine.py:937-977` (`_get_model_name_parameters`)

```python
def _get_model_name_parameters(self):
    for name, value in self.model.named_parameters():
        new_name = name
        # 简单的名称映射（处理 vision model 前缀等）
        if new_name.startswith("language_model."):
            new_name = new_name.replace("language_model.", "", 1)
        yield new_name, value
```

对于 FSDP Engine（使用 HuggingFace 原生模型），参数形状和命名**天然匹配 HF 格式**：

```
_get_full_tensor(param) 返回:
  name = "model.layers.0.self_attn.q_proj.weight"
  tensor.shape = [4096, 4096]  ← 直接是 HF 格式的 2D 权重矩阵

→ ParamSpec(name="model.layers.0.self_attn.q_proj.weight",
            shape=(4096, 4096), dtype="bfloat16")

→ dist.broadcast(tensor [4096, 4096], src=0, group=weight_update_group)
```

**没有 flatten/unflatten 步骤**——FSDP2 的 DTensor 保留了原始形状，
`full_tensor()` 直接返回原始形状的连续张量。

### 4.3 Archon Engine 的适配器路径（需要格式转换）

**源码**: `areal/experimental/engine/archon_weight_sync.py:132-164`

```python
for name, param in engine._get_model_name_parameters():
    tensor = _get_full_tensor(param)

    if engine.state_dict_adapter is not None:
        # Archon 内部格式 → HF 格式（可能一对多）
        hf_pairs = engine.state_dict_adapter.convert_single_to_hf(name, tensor)
    else:
        hf_pairs = [(name, tensor)]

    for hf_name, hf_tensor in hf_pairs:
        # ... bucket 累积 + broadcast
```

**典型的格式转换**（以 Qwen3 MoE 为例）：

```
Archon 内部:
  name = "layers.0.moe.experts.w1"
  tensor.shape = [64, 18432, 7168]  ← 3D: [num_experts, out_dim, in_dim]

convert_single_to_hf() →

HF 格式 (64 个独立参数):
  ("model.layers.0.mlp.experts.0.gate_proj.weight", [9216, 7168])
  ("model.layers.0.mlp.experts.0.up_proj.weight",   [9216, 7168])
  ("model.layers.0.mlp.experts.1.gate_proj.weight", [9216, 7168])
  ...
  ("model.layers.0.mlp.experts.63.up_proj.weight",  [9216, 7168])
```

**转换过程**（`state_dict_adapter.py`）：
1. `torch.unbind(dim=0)` 沿 expert 维度拆分 3D → 64 个 2D 张量
2. 每个 2D 张量再沿 out_dim 拆分为 gate_proj 和 up_proj（SwiGLU 结构）
3. 重命名为 HF 标准命名

### 4.4 数据流总结

```
FSDP2 DTensor                    full_tensor()              Broadcast
[512, 4096]                    [4096, 4096]             [4096, 4096]
(Shard(0) 本地分片)      →    (完整参数，原始形状)   →   (NCCL 广播到推理 Worker)
                          ↑                            ↑
                      NCCL AllGather               weight_update_group
                      (FSDP2 内部)                   (自定义跨集群组)

         名称映射                  推理侧接收
  "model.layers.0..."     →     torch.empty(shape)
  (HF 格式或转换后)              dist.broadcast(recv, src=0)
                                 model.load_weights([(name, tensor)])
```

---

## 5. 异构 Process Group 桥接：双独立 NCCL Communicator 的 P2P 通道

### 5.1 问题定义

```
训练集群                              推理集群
┌─────────────────────┐              ┌─────────────────────┐
│ NCCL Group A:       │              │ NCCL Group B:       │
│   torchrun 的默认 PG │              │   vLLM/SGLang 的 PG  │
│   FSDP2 all-gather  │              │   TP all-reduce     │
│   gradient reduce   │              │                     │
│                     │     ???      │                     │
│   Rank 0 ──────────────────────────── Rank 0..N          │
│                     │              │                     │
└─────────────────────┘              └─────────────────────┘

问题: 如何在两个完全独立的 NCCL communicator 之间建立通信通道？
```

### 5.2 解决方案：第三个独立的 NCCL Process Group

AReaL 的方案不是在现有组之间建立 P2P，而是**创建一个全新的第三个 NCCL Process Group**，
让训练的 rank 0 和所有推理 Worker 共同加入。

**源码**: `areal/engine/core/distributed.py:25-90`

```python
def init_custom_process_group(
    backend=None, init_method=None, timeout=None,
    world_size=-1, rank=-1, store=None, group_name=None, ...
):
    # ① 创建全新的 TCP Store（独立于 torchrun 的 store）
    rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
    store, rank, world_size = next(rendezvous_iterator)

    # ② PrefixStore 隔离命名空间（防止 key 冲突）
    store = PrefixStore(group_name, store)

    # ③ 创建全新的 NCCL communicator
    pg, _ = _new_process_group_helper(
        world_size, rank, [], backend, store,
        group_name=group_name, timeout=timeout,
    )

    # ④ 注册到 PyTorch 全局 PG 注册表
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg
```

### 5.3 三层隔离机制

```
┌─────────────── 训练进程 ────────────────┐   ┌──── 推理 Worker ────┐
│                                         │   │                     │
│  PG-A: FSDP2 (torchrun 默认 PG)        │   │  PG-B: vLLM TP/PP   │
│    TCP Store A (MASTER_ADDR:MASTER_PORT)│   │    TCP Store B       │
│    NCCL Comm A (ncclUniqueId_A)         │   │    NCCL Comm B       │
│                                         │   │                     │
│  ───────────── 完全独立 ──────────────  │   │  ─── 完全独立 ────  │
│                                         │   │                     │
│  PG-C: Weight Update Group              │   │  PG-C: Weight Update │
│    TCP Store C (gethostip():free_port)  │   │    (同一个 Store C)  │
│    PrefixStore("update_weight_group_0") │   │    相同 prefix       │
│    NCCL Comm C (ncclUniqueId_C)         │   │    NCCL Comm C       │
│    rank = 0 (broadcast src)             │   │    rank = 1..N       │
│                                         │   │                     │
└─────────────────────────────────────────┘   └─────────────────────┘

三层隔离:
  ① TCP Store 隔离: PG-C 使用独立的 TCP 端点
  ② PrefixStore 隔离: key 命名空间不冲突
  ③ NCCL Communicator 隔离: 独立的 ncclUniqueId
```

### 5.4 初始化时序

**训练侧** (`fsdp_engine.py:1048-1078`):

```python
def _init_weight_update_from_distributed(self, meta):
    # 1. 训练 rank 0 获取自己的 IP 和空闲端口
    meta.nccl_master_address = gethostip()
    meta.nccl_master_port = find_free_ports(1)[0]

    # 2. 绕过 torchrun 的 Agent Store
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)

    if dist.get_rank() == 0:
        # 3. 非阻塞通知推理侧（HTTP POST → RolloutController → 所有 Worker）
        fut = self.rollout_engine.init_weights_update_group(meta)

        # 4. 训练 rank 0 加入 PG-C（rank=0, world_size=gen_ws+1）
        self.weight_update_group = init_custom_process_group(
            backend="nccl",
            world_size=meta.alloc_mode.gen.world_size + 1,
            init_method=f"tcp://{meta.nccl_master_address}:{meta.nccl_master_port}",
            rank=0,
            group_name=meta.nccl_group_name,
        )

        # 5. 等待推理侧全部加入
        fut.result()
```

**推理侧** — vLLM Worker (`vllm_worker_extension.py:263-288`):

```python
def init_update_weight_group(self, master_address, master_port, rank_offset,
                              world_size, backend, group_name):
    # 每个 vLLM Worker 加入 PG-C
    # rank = self.rank + rank_offset (self.rank 是 Worker 在 vLLM 内部的 TP/PP rank)
    group = init_custom_process_group(
        backend=backend,
        world_size=world_size,
        init_method=f"tcp://{master_address}:{master_port}",
        rank=self.rank + rank_offset,
        group_name=group_name,
    )
    self.weight_update_groups[group_name] = group
```

**推理侧** — SGLang (`sglang_remote.py:183-201`):

```python
rank_offset = 1 + server_idx * meta.alloc_mode.gen.tp_size
# Server 0 的 TP Workers: ranks [1, 1+tp_size)
# Server 1 的 TP Workers: ranks [1+tp_size, 1+2*tp_size)
```

### 5.5 Rank 映射关系

```
PG-C (weight_update_group):

  Rank 0:  训练集群 dist.get_rank()==0 (FSDP2 全局 rank 0)
  Rank 1:  推理 Server 0, TP Worker 0
  Rank 2:  推理 Server 0, TP Worker 1
  ...
  Rank K:  推理 Server 1, TP Worker 0
  ...
  Rank N:  推理 Server M, TP Worker (tp_size-1)

  world_size = 1 + num_servers * tp_size
```

### 5.6 组的生命周期

**创建时机**: 首次 `connect_engine()` 时创建，由 `weight_update_group_initialized` 标志守护。

**复用**: PG-C 创建一次后在**所有后续 weight sync 中复用**。不会每次 sync 重建。

**销毁**: **未显式销毁**。`fsdp_engine.py:destroy()` 只销毁默认 PG (`dist.destroy_process_group()`)，
PG-C 作为独立组不受影响，NCCL communicator 在进程退出时由 OS 回收。

### 5.7 共存安全性

**FSDP2 的 all_gather 与 weight_update broadcast 是否可能冲突？**

**不会**——两者在 Python 控制流中严格串行：

```
_update_weights_from_distributed():
  for each param:
    tensor = _get_full_tensor(param)        ← FSDP2 all_gather (PG-A)
    if not main_rank: continue              ← 非 rank 0 不继续
    # 累积到 bucket...
    if bucket_full:
      _update_bucket_weights_from_distributed()  ← Weight broadcast (PG-C)
```

同一参数的 PG-A all_gather 和 PG-C broadcast 是**顺序执行**的。
虽然它们使用不同的 NCCL communicator 和 CUDA stream，但 Python GIL + 显式 `handle.wait()`
确保了不会并发执行。

**NIC 带宽不会争抢**——因为是串行的，不会出现 PG-A 和 PG-C 同时发包的情况。

---

## 6. 完整数据流：从 FSDP2 Shard 到 vLLM 模型权重

```
训练 Rank 0              训练 Rank 1..7            推理 vLLM Worker 0..N

  ① param.data = DTensor      param.data = DTensor
     [512, 4096]               [512, 4096]
     (Shard(0) on dp_sp)       (Shard(0) on dp_sp)

  ② _get_full_tensor()         _get_full_tensor()
     ↓                          ↓
     DTensor.full_tensor()      DTensor.full_tensor()
     ↓                          ↓
     ═══ PG-A: NCCL AllGather (FSDP2 内部) ═══
     ↓                          ↓
     full [4096, 4096]          full [4096, 4096]
     (保留引用)                  (丢弃: if not main_rank)

  ③ 累积到 bucket:
     named_tensors.append(
       ("model...q_proj.weight",
        tensor [4096, 4096])
     )
     buffer_size += 128 MB

  ④ bucket_size > 1 GB → 刷出:
     ParamSpec = {
       name: "model...q_proj.weight"
       shape: (4096, 4096)
       dtype: "bfloat16"
     }
     ↓
     HTTP POST /callback/update_weights_xccl
     payload = {meta, param_specs}          →       fut = rollout_engine
                                                       .update_weights_from_distributed()
                                                    ↓
                                                    HTTP → vLLM Worker:
                                                    set_weight_meta(names, dtypes, shapes)
                                                    ↓
  ⑤ for tensor in bucket:                           for (name, dtype, shape):
       dist.broadcast(                                tensor = torch.empty(shape, dtype)
         tensor,                                      dist.broadcast(
         src=0,               ← PG-C 广播 →             tensor,
         group=weight_update_group,                     src=0,
         async_op=True                                  group=weight_update_group,
       )                                                async_op=False
                                                      )
                                                      model.load_weights([(name, tensor)])

  ⑥ for handle in handles:
       handle.wait()
     fut.result()             ←── HTTP 确认 ────     完成加载
     named_tensors.clear()
```

---

## 7. 代码质量发现

### Medium 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 1 | `fsdp_engine.py` | 374-389 | `weight_update_group` 在 `destroy()` 中未显式销毁。NCCL communicator 泄漏（低影响，进程退出时 OS 回收） |
| 2 | `fsdp_engine.py` | 1058 | `TORCHELASTIC_USE_AGENT_STORE` 设置是进程全局的，虽有守护但可改为 context manager |
| 3 | `fsdp_engine.py` | 985 | `full_tensor()` 返回值未断言 `is_contiguous()`。`dist.broadcast()` 要求连续张量 |

### Low 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 4 | `distributed.py` | 75 | `str(torch.__version__) >= "2.6"` 字符串比较对 "2.10" 等版本不正确 |
| 5 | `fsdp_engine.py` | 1051, 1084 | 注释 "Reset weight weight meta" 中 "weight" 重复 |

---

## 8. 设计总结

### 底层切片汇聚

> FSDP2 使用 DTensor（非 FSDP1 的 FlatParameter），**保持原始参数形状**。
> 切片解除通过 `DTensor.full_tensor()` 实现，内部触发 NCCL all_gather
> 在 FSDP2 的 `dp_sp` mesh 上（如果有 TP，还会在 TP mesh 上额外 all_gather）。
> OOM 防护依赖**逐参数 unshard + bucket 累积 + 可配置 chunk_size**，
> Rank 0 峰值额外显存 ≤ `weight_chunked_mem_mb` + 最大单参数大小（~1.15-1.5 GB）。

### 内存布局对齐

> **不存在"一维展平张量还原回 3D/4D"的问题**——这是 FSDP1 的问题，不是 FSDP2 的。
> FSDP2 的 DTensor 天然保持原始形状。`full_tensor()` 直接返回原始形状的连续 GPU 张量。
> 对于 Archon 引擎，`state_dict_adapter.convert_single_to_hf()` 处理内部格式到 HF 格式的
> 名称映射和结构转换（如 3D MoE 权重拆分为 64 个独立 expert 的 2D 权重）。

### 异构 Process Group 桥接

> AReaL 不在现有 PG 之间建立 P2P——而是创建**第三个完全独立的 NCCL Process Group (PG-C)**。
> PG-C 使用独立的 TCP Store（独立端点）、PrefixStore（命名空间隔离）和 NCCL Communicator
> （独立 ncclUniqueId）。训练 rank 0 加入为 `rank=0`（broadcast src），
> 所有推理 Worker 加入为 `rank=1..N`（broadcast dst）。
> PG-C 创建一次后复用，与 FSDP2 的 PG-A 和 vLLM 的 PG-B 完全不干涉。
