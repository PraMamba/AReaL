# FSDP2 多维并行扩展与 MoE 专家切片策略深度分析

> 基于源码的底层分析，覆盖 DeviceMesh 三维并行初始化、梯度通信重叠效率、
> 以及 MoE 专家在 FSDP2 下的定向切片机制如何保持稀疏性。

---

## 目录

1. [DeviceMesh 三维并行初始化](#1-devicemesh-三维并行初始化)
2. [TP + SP 与 FSDP2 的组合机制](#2-tp--sp-与-fsdp2-的组合机制)
3. [梯度通信重叠分析](#3-梯度通信重叠分析)
4. [MoE 专家的切片陷阱与 AReaL 的解决方案](#4-moe-专家的切片陷阱与-areal-的解决方案)
5. [完整前向传播通信时序](#5-完整前向传播通信时序)
6. [代码质量发现](#6-代码质量发现)
7. [设计总结](#7-设计总结)

---

## 1. DeviceMesh 三维并行初始化

### 1.1 FSDP Engine 的 3D Mesh

**源码**: `areal/engine/fsdp_utils/parallel.py:78-139`

FSDP Engine（非 Archon）创建 3 维 DeviceMesh，约束为 `dp × sp × tp = world_size`：

```python
# parallel.py:123-139 (_build_mesh_without_ep)
def _build_mesh_without_ep(self) -> DeviceMesh:
    dp, sp, tp = (self._ps.dp_size, self._ps.cp_size, self._ps.tp_size)

    mesh = init_device_mesh(
        current_platform.device_type,
        mesh_shape=(dp, sp, tp),
        mesh_dim_names=("dp", "sp", "tp"),
    )

    # 派生子 Mesh:
    mesh["dp", "sp"]._flatten(mesh_dim_name="dp_sp")   # ← FSDP 使用此 mesh
    mesh["sp", "tp"]._flatten(mesh_dim_name="sp_tp")    # ← 模型并行组
    return mesh
```

**关键设计**: FSDP 在 `dp_sp`（而非 `dp`）上做参数分片。这意味着 **SP rank 也参与 FSDP 权重分片**，
进一步降低每 rank 的显存占用。

```
示例: 8 GPU, dp=2, sp=2, tp=2

DeviceMesh [dp=2, sp=2, tp=2]:
  GPU 0: (dp=0, sp=0, tp=0)    GPU 1: (dp=0, sp=0, tp=1)
  GPU 2: (dp=0, sp=1, tp=0)    GPU 3: (dp=0, sp=1, tp=1)
  GPU 4: (dp=1, sp=0, tp=0)    GPU 5: (dp=1, sp=0, tp=1)
  GPU 6: (dp=1, sp=1, tp=0)    GPU 7: (dp=1, sp=1, tp=1)

  dp_sp mesh (FSDP 分片): 4 ranks → 每 rank 存 1/4 参数
  tp mesh: 2 ranks → 每 rank 存 1/2 heads
  sp mesh: 2 ranks → 每 rank 处理 1/2 序列
```

### 1.2 Archon Engine 的 4D/5D Mesh

**源码**: `areal/experimental/models/archon/parallel_dims.py:179-293`

Archon 添加 PP 维度，以及可选的 EP 维度：

```
无 EP: [pp, dp_shard, cp, tp]             — 4D
有 EP: [pp, dp_shard_mod_ep, dp_shard_in_ep, cp, tp] — 5D
```

**Archon 特有子 Mesh**:

| 子 Mesh | 组成 | 用途 |
|---------|------|------|
| `dp_shard_cp` | `dp_shard × cp` | Dense 参数 FSDP |
| `dp_shard_mod_ep` | `dp_shard×cp×tp/ep` (etp=1) | Expert 参数 FSDP |
| `pp_cp_tp` | `pp × cp × tp` | 数据广播 |
| `ep` | `dp_in_ep × cp × etp` | EP 通信 |
| `ep_tp` | `[ep, tp]` 2D | ExpertTensorParallel |

### 1.3 两种引擎的 Mesh 对比

| 维度 | FSDP Engine | Archon Engine |
|------|------------|---------------|
| Root 维度 | 3D `(dp, sp, tp)` | 4D/5D `(pp, dp_shard, cp, tp, [ep])` |
| FSDP mesh | `dp_sp` (dp×sp) | `dp_shard_cp` (dp_shard×cp) |
| TP mesh | `tp` | `tp` |
| SP/CP mesh | `sp` | `cp` |
| PP 支持 | 无 | 有（正交维度） |
| EP 支持 | 有限（通过 `dp_mod_ep` 子 mesh） | 完整（5D mesh） |

---

## 2. TP + SP 与 FSDP2 的组合机制

### 2.1 应用顺序：TP → FSDP2

**源码**: `areal/engine/fsdp_utils/parallel.py:368-394`

```python
def parallelize_model(model, config, model_config, nd_device_mesh, parallel_helper, ...):
    # ① 先应用 TP
    if tp_enabled:
        apply_non_moe_tp(model, model_config, parallel_helper, nd_device_mesh["tp"])

    # ② 再应用 FSDP2（在已经 TP 分片的 DTensor 之上）
    fsdp_kwargs = {
        "mesh": nd_device_mesh["dp_sp"],    # ← DP+SP 的 flatten mesh
        "mp_policy": MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32),
        "reshard_after_forward": True,
    }
    apply_fsdp2(model, fsdp_kwargs, wrap_policy)
```

### 2.2 TP 的 DTensor 布局

**源码**: `parallel.py:217-365` (`apply_non_moe_tp`)

TP 使用 PyTorch 的 `parallelize_module` 在 `tp` 子 mesh 上创建 DTensor 布局：

```python
# TP 计划（简化）:
model_tp_plan = {
    "layers.*.self_attn.q_proj": ColwiseParallel(),        # 按列切分（头数维度）
    "layers.*.self_attn.o_proj": RowwiseParallel(output=Shard(1)),  # 按行切分
    "layers.*.input_layernorm":  SequenceParallel(),        # 序列并行
    "layers.*.mlp.gate_proj":    ColwiseParallel(),
    "layers.*.mlp.down_proj":    RowwiseParallel(output=Shard(1)),
}
```

**SequenceParallel 模式**:
- LayerNorm 层标记为 `SequenceParallel()`——在序列维分片的数据上操作
- Attention/MLP 的输入用 `PrepareModuleInput` 从 `Shard(1)` → `Replicate()`（all-gather）
- 输出用 `RowwiseParallel(output=Shard(1))` 从 `Replicate()` → `Shard(1)`（reduce-scatter）

### 2.3 FSDP2 在 TP DTensor 之上的分片

TP 应用后，参数变为 TP DTensor。FSDP2 再在其上添加分片：

```
TP 应用后:
  q_proj.weight: DTensor [4096, 4096]
    .device_mesh = tp_mesh (2 ranks)
    .placements = (Shard(0),)         ← ColwiseParallel
    ._local_tensor = [2048, 4096]     ← 每 TP rank 存一半 heads

FSDP2 应用后:
  q_proj.weight: DTensor [4096, 4096]
    .device_mesh = 复合 mesh (dp_sp × tp)
    .placements = (Shard(?), Shard(0)) ← FSDP shard + TP shard
    ._local_tensor = [2048/dp_sp, 4096] 或 更小
```

**关键**: FSDP2 在 `dp_sp` mesh 上分片时，操作的是已经 TP 分片后的**本地张量**。
所以 FSDP 的 all-gather 只恢复 TP 本地分片的 FSDP 部分，不恢复完整的 TP 前张量。

### 2.4 SP (Ulysses 序列并行) 的实现

**FSDP Engine**: `sp` 维度复用为 `SequenceParallel()` TP 计划的一部分（`parallel.py:256,273`）。

**Archon Engine**: 独立的 `apply_cp()` 函数（`parallelize.py:500-528`）：

```python
def apply_cp(model, cp_group, tp_size):
    for transformer_block in model.layers.values():
        transformer_block.attention.set_cp_group(cp_group)  # 设置 Ulysses CP 组
```

Ulysses SP 的通信模式：
1. **Forward**: `gather_seq_scatter_heads` — All-to-All 聚合序列、分散头
2. **Attention**: 每 rank 在完整序列上计算自己的头子集
3. **Forward**: `gather_heads_scatter_seq` — All-to-All 聚合头、分散序列

---

## 3. 梯度通信重叠分析

### 3.1 单层的完整通信时序

```
                     一个 Transformer Layer 的通信时序
                     (TP=2, SP=2, FSDP on dp_sp=4)

  时间 →

  [FSDP AllGather] → [TP AllGather] → [Attention Compute] → [TP ReduceScatter]
  (dp_sp group)      (tp group)       (local GPU)           (tp group)
  恢复本地参数         Shard(1)→Rep    Q/K/V → Score → AV    Rep→Shard(1)

                      [SP All2All]  → [Attn Core] → [SP All2All]
                      (sp group)      (local)       (sp group)
                      scatter head    full seq,     gather head
                      gather seq      local heads   scatter seq

  [SequenceParallel LayerNorm] → [TP AllGather] → [MLP Compute] → [TP ReduceScatter]
  (local, seq-sharded)            (tp group)       (local)         (tp group)

  [FSDP Reshard]
  (dp_sp group)
  释放完整参数
```

### 3.2 通信重叠机会

**FSDP2 前向预取 vs TP 通信**:

```
Layer i                               Layer i+1
  ├─ TP AllReduce (end)               ├─ FSDP AllGather (start)
  │   → tp group                      │   → dp_sp group
  │   → 不同 NCCL communicator!       │   → 可以重叠!
```

FSDP2 的隐式前向预取机制在 layer `i` 开始前向时，就启动 layer `i+1` 的 AllGather。
因为 TP 和 FSDP 使用**不同的 NCCL communicator** 和**不同的 CUDA stream**，
它们可以在硬件级别并发执行。

**但存在限制**:
- GPU SM 饱和时，计算和通信无法有效重叠
- 同一 NIC 上的两个 NCCL 集合操作会共享带宽
- NCCL 内部的 stream 同步可能引入隐式依赖

### 3.3 反向传播的通信模式

```
反向传播（per layer, 逆序）:

  ① FSDP AllGather (unshard 权重用于梯度计算)
  ② TP AllGather (对应前向的 ReduceScatter 的反向)
  ③ 梯度计算 (local)
  ④ TP ReduceScatter (对应前向的 AllGather 的反向)
  ⑤ SP All-to-All (Ulysses 反向)
  ⑥ FSDP ReduceScatter (梯度在 dp_sp group 上累积)

  FSDP 反向预取: layer i 的 ReduceScatter 与 layer i-1 的 AllGather 重叠
```

### 3.4 与纯 DP 的效率对比

| 并行模式 | 每层额外通信 | 通信组大小 | 重叠能力 |
|----------|------------|-----------|---------|
| 纯 DP (FSDP) | AllGather(fwd) + ReduceScatter(bwd) | `world_size` | FSDP 预取重叠 |
| DP + TP | + 2×TP AllGather + 2×TP ReduceScatter | `tp_size` | TP/FSDP 可并发 |
| DP + SP | + 2×SP All2All; FSDP 在 `dp×sp` 上 | `sp_size`, `dp×sp` | SP/FSDP 可并发 |
| DP + SP + TP | + 2×TP + 2×SP; FSDP 在 `dp×sp` 上 | 各自的组 | 三者可部分并发 |

**关键洞察**: TP/SP 的通信在层**内部**是串行的（数据依赖），但与 FSDP 的层**间**预取可以并发。
总体效率取决于 TP/SP 通信量与计算量的比值——当模型足够大（每 TP rank 计算量大），
通信可被计算隐藏。

### 3.5 梯度裁剪的同步点

**源码**: `areal/engine/fsdp_utils/grad.py:225-268`

反向传播完成后，梯度裁剪需要两次小型 all-reduce：

```python
# 1. FSDP 组内聚合梯度范数（因为梯度是分片的）
dist.all_reduce(total_norm, op=ReduceOp.SUM, group=fsdp_group)

# 2. TP 组内聚合（处理 TP 复制的参数）
dist.all_reduce(total_norm, op=ReduceOp.SUM, group=tp_group)
```

这是**标量 all-reduce**（单个 float32），延迟极小但构成全局同步点。

### 3.6 优化器步骤

**无额外通信**。FSDP2 的设计保证了 reduce-scatter 后每 rank 持有正确的梯度分片，
优化器直接在本地分片上操作。

---

## 4. MoE 专家的切片陷阱与 AReaL 的解决方案

### 4.1 陷阱描述

**朴素 FSDP2 + MoE 的问题**:

```
假设 64 个专家，8 GPU，FSDP2 默认 Shard(0):

  Expert 权重 w1: [64, 18432, 7168]  (3D 张量)

  FSDP2 Shard(0) 在 8 GPU 上:
    GPU 0: w1[0:8, :, :]     ← 专家 0-7 的完整权重
    GPU 1: w1[8:16, :, :]    ← 专家 8-15 的完整权重
    ...

  前向传播时: 每个 token 的路由可能指向任意专家
  → FSDP2 必须 AllGather 恢复完整的 [64, 18432, 7168] 张量
  → 每个 GPU 持有所有 64 个专家的完整权重
  → 完全破坏了 MoE 的稀疏性优势！
```

### 4.2 AReaL 的解决方案：EP 先分区 + FSDP 后分片

**核心思路**: 不让 FSDP2 直接看到完整的 64 个专家。
先用 EP 将专家分区到不同 GPU 组，FSDP2 只对**本地专家**做内存优化分片。

**两层分片**:

```
第一层: Expert Parallelism (EP)
  ep_mesh (8 ranks, EP=8):
    GPU 0: 8 个本地专家 (experts 0-7)
    GPU 1: 8 个本地专家 (experts 8-15)
    ...
    GPU 7: 8 个本地专家 (experts 56-63)

  权重形状变化:
    全局: w1 [64, 18432, 7168]
    本地: w1._local_tensor [8, 18432, 7168]  ← Shard(0) on EP mesh

第二层: FSDP2 (在 dp_shard_mod_ep mesh 上)
  dp_shard_mod_ep mesh (更小的 DP mesh):
    GPU 0 的本地专家 [8, 18432, 7168] 在 dp_mod_ep ranks 上进一步分片
    → 每 rank 存 [8, 18432/dp_mod_ep, 7168] 或更小

  前向传播时的 AllGather:
    FSDP2 只恢复本地的 8 个专家 → [8, 18432, 7168]
    NOT 全部 64 个专家！
    → 稀疏性完全保持！
```

### 4.3 具体实现

#### Step 1: EP 分区专家权重

**源码**: `areal/experimental/models/archon/expert_parallel.py:87-102`

```python
class ExpertParallel(BaseExpertParallel):
    def _partition_fn(self, name, module, device_mesh):
        for param_name, param in module.named_parameters(recurse=False):
            # Shard(0) 在 EP mesh 上 → 按专家维度分区
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
            module.register_parameter(param_name, dist_param)
```

这里的 `Shard(0)` 操作在 **EP mesh** 上，沿第 0 维（专家维度）分区。
64 个专家在 EP=8 的 mesh 上分成每 rank 8 个。

#### Step 2: FSDP2 对本地专家做内存分片

**源码**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:366-399`

```python
def apply_fsdp(model, dp_mesh, dp_mod_ep_mesh=None, ep_degree=1, ...):
    for transformer_block in model.layers.values():
        if moe_enabled and ep_degree > 1 and dp_mod_ep_mesh is not None:
            fsdp_ep_config = fsdp_config.copy()
            fsdp_ep_config["mesh"] = dp_mod_ep_mesh  # ← 不同于 dense 参数的 mesh！

            # 当 dp_mod_ep × ep > num_experts 时，
            # dim-0 元素太少无法有效分片，改用 dim-1
            _experts_shard_placement_fn = None
            if dp_mod_ep_mesh.size() * ep_degree > num_experts:
                _experts_shard_placement_fn = lambda param: Shard(1)

            # 对专家模块单独 FSDP 包装（使用更小的 mesh）
            fully_shard(
                transformer_block.moe.experts,
                **fsdp_ep_config,
                shard_placement_fn=_experts_shard_placement_fn,
            )

            # 设置梯度归一化因子
            transformer_block.moe.experts.set_gradient_divide_factor(
                gradient_divide_factor  # = dp_shard × cp
            )

        # 整个 transformer block 用标准 mesh 包装
        fully_shard(transformer_block, **fsdp_config)
```

**三个关键设计**:

1. **不同 Mesh**: Expert 用 `dp_shard_mod_ep`（更小），Dense 用 `dp_shard_cp`（更大）
2. **自适应分片维度**: 当 `dp_mod_ep × ep > num_experts` 时改为 `Shard(1)` 避免 dim-0 过小
3. **梯度归一化**: 强制设置 `gradient_divide_factor = dp_shard × cp`，确保与 Dense 参数一致

#### Step 3: EP 的 All-to-All Token Dispatch

**源码**: `expert_parallel.py:104-181`

前向传播时，MoE 的数据流完全保持稀疏性：

```
Token 路由:
  Router → top-k 专家选择 → 每个 token 知道要去哪个 GPU

Token Dispatch (All-to-All):
  GPU 0 的 token → 需要 expert 32 的 → 发送到 GPU 4
  GPU 4 的 token → 需要 expert 5 的 → 发送到 GPU 0
  (每个 GPU 只接收要在本地专家处理的 token)

本地计算:
  每个 GPU 只在本地 8 个专家上计算
  → FSDP AllGather 只恢复这 8 个专家的参数
  → 不涉及其他 56 个专家

Token Combine (All-to-All 反向):
  结果发回原始 GPU
```

### 4.4 为什么朴素 FSDP2 会破坏稀疏性

```
朴素 FSDP2:
  FSDP AllGather → 恢复全部 64 个专家 → 所有 GPU 持有全部权重
  Router → 每 token 选 2 个专家 → 但全部 64 个已在显存中
  → 显存: O(总专家数)，无论路由结果如何

AReaL 的 EP + FSDP2:
  EP 分区 → 每 GPU 持有 8 个专家
  FSDP AllGather → 只恢复这 8 个专家的 FSDP 分片
  All-to-All → token 路由到正确的 GPU
  本地计算 → 只在本地 8 个专家上
  → 显存: O(总专家数 / EP_size)，线性缩减
```

### 4.5 基础 FSDP Engine 的 MoE 支持现状

**源码**: `areal/engine/fsdp_utils/__init__.py:52-87`

基础 FSDP Engine 的 `apply_fsdp2()` **完全没有 MoE 逻辑**——只按 transformer block 级别做统一包装。
如果用它训练 MoE 模型，**确实会陷入朴素 FSDP2 的切片陷阱**。

**MoE 感知的 FSDP2 完全存在于 Archon Engine 中**（`qwen3/infra/parallelize.py`）。

### 4.6 EP-Aware FSDP 预取

**源码**: `parallelize.py:434-497` (`_setup_fsdp_prefetch`)

EP 的 All-to-All 包含 D2H 同步（计算 split sizes 需要 `tolist()`），
这会干扰 FSDP2 的隐式预取。解决方案是显式建立预取链：

```python
def _setup_fsdp_prefetch(model):
    # 前向预取链:
    # tok_embeddings → block[0] → block[1] (+ experts) → ... → final layers
    for i, block in enumerate(blocks):
        next_modules = [blocks[i+1]]
        if hasattr(blocks[i+1], "moe"):
            next_modules.append(blocks[i+1].moe.experts)  # ← 显式预取专家权重
        block.set_modules_to_forward_prefetch(next_modules)

    # 反向预取链:
    # final → block[last] → block[last-1] (+ experts) → ... → tok_embeddings
```

---

## 5. 完整前向传播通信时序

### Dense Layer (TP + SP + FSDP2)

```
时间 →

Layer i:
  [FSDP AG]  [TP AG]  [SP A2A] [Attn] [SP A2A] [TP RS]  [LayerNorm]
  dp_sp grp  tp grp   sp grp   local  sp grp   tp grp   local(Shard1)

  [TP AG]  [MLP] [TP RS]  [LayerNorm]  [FSDP Reshard]
  tp grp   local tp grp   local         dp_sp grp
                                         ↓
                                    FSDP Prefetch Layer i+1
                                    (可与 Layer i 的 TP RS 重叠)

AG = AllGather, RS = ReduceScatter, A2A = All-to-All
```

### MoE Layer (EP + TP + SP + FSDP2)

```
时间 →

MoE Layer i:
  [FSDP AG dense]  [TP AG]  [SP A2A] [Attn] [SP A2A] [TP RS]
  dp_shard_cp grp  tp grp   sp grp   local  sp grp   tp grp

  [FSDP AG experts]  [Router]  [EP A2A dispatch]  [Expert Compute]  [EP A2A combine]
  dp_mod_ep grp      local     ep grp             local (8 experts) ep grp

  [TP RS]  [FSDP Reshard experts]  [FSDP Reshard dense]
  tp grp   dp_mod_ep grp           dp_shard_cp grp
```

---

## 6. 代码质量发现

### Medium 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 1 | `fsdp_utils/__init__.py` | 52-87 | 基础 FSDP Engine 的 `apply_fsdp2()` 无 MoE 感知。若使用此 Engine 训练 MoE 模型将陷入稀疏性破坏陷阱 |
| 2 | `parallelize.py` | 578-585 | `SequenceParallel` on norms 与 `torch.compile` 不兼容，这些 norms 被排除在编译外，产生小气泡 |

### Low 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 3 | `parallel.py` | 392 | `reshard_after_forward=True` 是硬编码的，不像 Archon 有策略选择（always/never/default） |
| 4 | Ulysses 实现 | model.py:224-244 | Q/K/V 的三次 `gather_seq_scatter_heads` 是串行的，可考虑批量化 |

### Positive 发现

| # | 文件 | 行号 | 亮点 |
|---|------|------|------|
| 5 | `parallelize.py` | 379-383 | `shard_placement_fn = lambda: Shard(1)` 自适应切换——当 dim-0 元素不足时自动改为 dim-1 分片 |
| 6 | `parallelize.py` | 434-497 | EP-aware FSDP 显式预取链——正确解决了 D2H sync 干扰隐式预取的问题 |
| 7 | `parallel_dims.py` | 361-368 | `fsdp_gradient_divide_factor` 确保 Dense/Expert 梯度缩放一致 |

---

## 7. 设计总结

### DeviceMesh 初始化

> AReaL 通过 **扁平化子 Mesh** 将 DP 和 SP 合并为 `dp_sp`（FSDP Engine）或
> `dp_shard_cp`（Archon），使 FSDP2 在扩展的 DP 组上分片。SP rank **参与** FSDP 权重分片，
> 进一步降低每 rank 显存。TP 使用独立子 Mesh，其 DTensor 布局在 FSDP2 之前创建。
> FSDP2 在 TP DTensor 之上再添加 DP 维度的分片，形成多维 DTensor。

### 3D 梯度通信重叠

> **是的，仍然高效**。TP 和 FSDP2 使用**不同的 NCCL communicator**（各自的子 Mesh），
> 可在不同 CUDA stream 上并发。FSDP2 的前向预取机制在 layer `i` 计算时就启动
> layer `i+1` 的 AllGather。反向传播中，FSDP2 的 ReduceScatter 与上一层的
> AllGather 重叠。主要限制是同层内的 TP↔SP 通信是串行的（数据依赖）。

### MoE 切片陷阱

> **AReaL 的 Archon Engine 完全解决了此问题**。通过 **EP 先分区 + FSDP2 后分片** 的两层设计：
> - EP 的 `Shard(0)` 在 EP mesh 上按**专家维度**分区（每 GPU 持有 `num_experts/ep_size` 个专家）
> - FSDP2 的 `Shard(0/1)` 在**更小的 `dp_shard_mod_ep` mesh** 上对**本地专家**做内存优化分片
> - 前向 AllGather 只恢复本地专家，不涉及远端专家
> - Token 通过 EP All-to-All 路由到正确的 GPU，保持 MoE 稀疏性
>
> **基础 FSDP Engine 没有此逻辑**——直接用它训练 MoE 会陷入稀疏性破坏的陷阱。
> MoE 感知的 FSDP2 完全存在于 Archon Engine 的 `parallelize_qwen3` 中。
