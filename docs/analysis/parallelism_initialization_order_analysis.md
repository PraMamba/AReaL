# 并行化初始化顺序深度分析：TP → CP → AC → compile → FSDP

## 目录

- [1. 概述](#1-概述)
- [2. 规范顺序的源码定义](#2-规范顺序的源码定义)
- [3. DeviceMesh 拓扑基础](#3-devicemesh-拓扑基础)
- [4. 第一步：TP（张量并行）——为什么必须首先执行](#4-第一步tp张量并行为什么必须首先执行)
- [5. 第二步：CP（上下文并行）——为什么在 TP 之后、FSDP 之前](#5-第二步cp上下文并行为什么在-tp-之后fsdp-之前)
- [6. 第三步：AC（激活检查点）——为什么在 TP/CP 之后、compile 之前](#6-第三步ac激活检查点为什么在-tpcp-之后compile-之前)
- [7. 第四步：torch.compile——为什么在 AC 之后、FSDP 之前](#7-第四步torchcompile为什么在-ac-之后fsdp-之前)
- [8. 第五步：FSDP——为什么必须最后执行](#8-第五步fsdp为什么必须最后执行)
- [9. 依赖关系 DAG](#9-依赖关系-dag)
- [10. 两条引擎路径的对比](#10-两条引擎路径的对比)
- [11. 反模式分析：违反顺序会怎样](#11-反模式分析违反顺序会怎样)

---

## 1. 概述

AReaL 框架在模型初始化时按严格顺序应用五种并行化/优化技术：

```
TP (Tensor Parallelism)
  → CP (Context Parallelism / Ulysses SP)
    → AC (Activation Checkpointing)
      → torch.compile
        → FSDP (Fully Sharded Data Parallelism)
```

这个顺序不是任意选择，而是由 **PyTorch 分布式基础设施的不变量约束** 严格决定的。每一步都会改变模型的内部表示（参数类型、模块层次、计算图结构），后续步骤的正确性依赖于前面步骤已经建立的不变量。

本文档基于 AReaL 源码，从架构约束��代码实现和故障模式三个维度深入分析这一顺序背后的原因。

---

## 2. 规范顺序的源码定义

### 2.1 ArchonEngine（完整 5 步序列）

顺序在 `parallelize_qwen3` 和 `parallelize_qwen2` 函数的文档字符串中明确定义：

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:100-106`

```python
"""
Order of operations:
1. Apply non-MoE TP (Tensor Parallelism for dense layers)
2. Apply MoE EP+TP (Expert Parallelism + MoE-specific TP)
3. Apply CP (Context Parallelism / Ulysses SP)
4. Apply AC (Activation Checkpointing) - must be after TP/EP
5. Apply torch.compile - must be after AC, before FSDP
6. Apply FSDP (Fully Sharded Data Parallelism)
"""
```

**文件**: `areal/experimental/models/archon/qwen2/infra/parallelize.py:85-91`

```python
"""
Order of operations:
1. Apply TP (Tensor Parallelism)
2. Apply CP (Context Parallelism / Ulysses SP)
3. Apply AC (Activation Checkpointing) - must be after TP
4. Apply torch.compile - must be after AC, before FSDP
5. Apply FSDP (Fully Sharded Data Parallelism)
"""
```

### 2.2 编排代码

以 `parallelize_qwen3` 为例，核心编排逻辑在 `parallelize.py:132-193`：

```python
# Step 1: Apply non-MoE TP first
tp_mesh = parallel_dims.get_mesh("tp") if parallel_dims.tp_enabled else None
if tp_mesh is not None:
    apply_non_moe_tp(model, tp_mesh, loss_parallel=loss_parallel)

# Step 1b: Apply MoE EP+TP
ep_mesh = parallel_dims.get_mesh("ep") if parallel_dims.ep_enabled else None
ep_tp_mesh = parallel_dims.get_mesh("ep_tp") if parallel_dims.etp_enabled else None
if tp_mesh is not None or ep_mesh is not None:
    apply_moe_ep_tp(model, tp_mesh, ep_mesh, etp=parallel_dims.etp, ep_tp_mesh=ep_tp_mesh)

# Step 2: Apply CP
if parallel_dims.cp_enabled:
    cp_group = parallel_dims.get_group("cp")
    apply_cp(model, cp_group, tp_size=parallel_dims.tp)

# Step 3: AC must be after TP/CP
if ac_config is not None and ac_config.mode != "none":
    apply_ac(model, ac_config, model_compile_enabled=enable_compile,
             op_sac_save_list=_get_op_sac_save_list())

# Step 4: torch.compile must be after AC, before FSDP
if enable_compile:
    _apply_compile(model, ep_enabled=parallel_dims.ep > 1)

# Step 5: Apply FSDP
dp_mesh = parallel_dims.get_mesh("dp_shard_cp")
if dp_mesh is not None:
    apply_fsdp(model, dp_mesh, ...)
```

---

## 3. DeviceMesh 拓扑基础

理解并行化顺序的前提是理解 DeviceMesh 的构建方式。

### 3.1 无 EP 的 Mesh（FSDPEngine）

**文件**: `areal/engine/fsdp_utils/parallel.py:123-139`

```python
mesh = init_device_mesh(
    device_type,
    mesh_shape=(dp, sp, tp),
    mesh_dim_names=("dp", "sp", "tp"),
)
# 扁平化子网格
mesh["dp", "sp"]._flatten(mesh_dim_name="dp_sp")    # 用于 FSDP
mesh["sp", "tp"]._flatten(mesh_dim_name="sp_tp")    # 用于 CP+TP 通信组
```

### 3.2 有 EP 的 Mesh

**文件**: `areal/engine/fsdp_utils/parallel.py:84-121`

```python
mesh = init_device_mesh(
    device_type,
    mesh_shape=(dp_mod_ep, dp_in_ep, sp, tp),
    mesh_dim_names=("dp_mod_ep", "dp_in_ep", "sp", "tp"),
)
mesh["dp_mod_ep", "dp_in_ep"]._flatten(mesh_dim_name="dp")
mesh["dp_mod_ep", "dp_in_ep", "sp"]._flatten(mesh_dim_name="dp_sp")
mesh["sp", "tp"]._flatten(mesh_dim_name="sp_tp")
```

### 3.3 ArchonEngine Mesh

**文件**: `areal/experimental/models/archon/parallel_dims.py`

```python
mesh = init_device_mesh(
    device_type,
    (pp, dp_shard, cp, tp),
    mesh_dim_names=("pp", "dp_shard", "cp", "tp"),
)
# 关键扁平化
meshes["dp_shard_cp"] = mesh["dp_shard", "cp"]._flatten(...)   # 用于 FSDP
meshes["pp_cp_tp"] = mesh["pp", "cp", "tp"]._flatten(...)      # 用于广播
```

**核心洞察**: FSDP 使用的 `dp_shard_cp` mesh 将数据并行和上下文并行维度合并。这意味着 **CP 的 rank 同时参与 FSDP 的权重分片**，CP 拓扑必须在 FSDP 之前确定。

---

## 4. 第一步：TP（张量并行）——为什么必须首先执行

### 4.1 TP 对模型做了什么

TP 通过 `torch.distributed.tensor.parallel.parallelize_module()` 对模型进行**破坏性变换**。

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:196-297`

```python
def apply_non_moe_tp(model, tp_mesh, loss_parallel=True):
    # 根级别（embedding, norm, lm_head）
    root_plan = {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=True,
        ),
    }
    parallelize_module(model, tp_mesh, root_plan)

    # 每个 Transformer Block 的 TP plan
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), Replicate(), ...),
                desired_input_layouts=(Replicate(), Replicate(), ...),
            ),
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=True),
            "attention.q_norm": SequenceParallel(sequence_dim=2),
            "attention.k_norm": SequenceParallel(sequence_dim=2),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }
        parallelize_module(transformer_block, tp_mesh, layer_plan)
```

### 4.2 为什么 TP 必须第一

**原因 A：参数张量被不可逆地转换为 DTensor**

`parallelize_module()` 将 `nn.Parameter`（普通 `torch.Tensor`）替换为 `DTensor`（分布式张量）对象。例如，`ColwiseParallel()` 作用于 `q_proj` 会将 `q_proj.weight`（形状 `[n_heads * head_dim, dim]`）替换为一个沿维度 0 分片的 DTensor，每个 rank 只持有 `1/tp_size` 的列。这是对参数存储布局的**破坏性、不可逆变换**。

**原因 B：安装输入/输出再分布 hook**

TP 通过 `PrepareModuleInput` 安装 hook，指定激活值如何在分片和复制布局之间流动。例如，注意力模块的输入需要通过 all-gather 变为 `Replicate()` 布局，输出通过 reduce-scatter 变为 `Shard(1)` 布局。这些 hook 必须在后续变换（CP、AC、compile、FSDP）之前就位，因为后续变换基于当前模块图操作。

**原因 C：下游变换依赖 TP 的效果**

CP 的约束验证需要知道 TP 分割后的"本地" head 数量：

**文件**: `areal/experimental/models/archon/utils.py:66-67`

```python
q_heads = n_heads // tp_size
kv_heads = n_kv_heads // tp_size
```

如果 TP 在 CP 之后才执行，CP 会基于错误的（全局）head 数量进行配置。

---

## 5. 第二步：CP（上下文并行）——为什么在 TP 之后、FSDP 之前

### 5.1 CP 做了什么

CP 在 AReaL 中实现的是 **Ulysses 序列并行**，本质上是在注意力计算中通过 All-to-All 通信将 head 维度和 sequence 维度进行交换。

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:500-528`

```python
def apply_cp(model, cp_group, tp_size=1):
    """Apply context parallelism (Ulysses SP) to Qwen3 model."""
    cp_size = dist.get_world_size(cp_group)
    validate_cp_constraints(model.model_args, cp_size, tp_size)
    for transformer_block in model.layers.values():
        transformer_block.attention.set_cp_group(cp_group)
```

注意力前向传播中的实际通信：
- **注意力前**: `gather_seq_scatter_heads` — `[batch, seq/cp, heads, dim]` → `[batch, seq, heads/cp, dim]`
- **注意力后**: `gather_heads_scatter_seq` — 反向交换，恢复原始分区

### 5.2 为什么 CP 必须在 TP 之后

**原因 A：CP 约束验证依赖 TP 分割后的 head 数量**

**文件**: `areal/experimental/models/archon/utils.py:44-91`

```python
def validate_cp_constraints(model_args, cp_size, tp_size=1):
    q_heads = n_heads // tp_size       # ← 依赖 TP 的结果
    kv_heads = n_kv_heads // tp_size   # ← 依赖 TP 的结果
    # 约束 1: q_heads 必须能被 cp_size 整除
    if q_heads % cp_size != 0:
        raise ValueError(...)
    # 约束 2: kv_heads 必须能被 cp_size 整除或整除 cp_size
    ...
```

如果 CP 先于 TP 执行，验证会使用全局 head 数而非 TP 本地 head 数，导致错误配置。

**原因 B：TP 的再分布 hook 与 CP 的 All-to-All 在注意力 forward 内有严格的嵌套关系**

TP 的 `Shard(1)` 输出布局假设它直接喂入注意力 block 的 `PrepareModuleInput`（期望 `Shard(1)` 输入）。CP 的 All-to-All 必须发生在**注意力 forward 内部**——即在 TP 包裹的 q/k/v 投影和 TP 包裹的输出投影之间。如果 CP 在 TP 之前执行，TP 的 `parallelize_module` 会遇到已配置 CP 进程组的注意力模块，其 `PrepareModuleInput` hook 会与 CP 的 All-to-All 再分布模式冲突。

### 5.3 为什么 CP 必须在 FSDP 之前

**原因 A：FSDP mesh 整合了 CP 维度**

FSDP 使用 `dp_shard_cp` mesh，它将数据并行和上下文并行维度扁平化在一起。CP 的 rank **参与 FSDP 的权重分片**。如果 CP 在 FSDP 之后配置，FSDP 的 mesh 构建就无法正确纳入 CP 的通信拓扑。

**原因 B：FSDP 之后模块被 FSDPModule 封装**

FSDP 用 `FSDPModule` 封装模块后，虽然 `set_cp_group` 仍可调用，但 FSDP 的 pre-forward/post-forward hook 已经安装完毕，任何额外的通信模式引入都可能干扰 FSDP 的参数 unshard/reshard 逻辑。

---

## 6. 第三步：AC（激活检查点）——为什么在 TP/CP 之后、compile 之前

### 6.1 AC 做了什么

AC 通过 `checkpoint_wrapper` 包裹每个 Transformer Block，根本性地改变模块层次结构。

**文件**: `areal/experimental/models/archon/activation_checkpoint.py:247-305`

```python
def apply_ac(model, ac_config, *, model_compile_enabled=False, op_sac_save_list=None):
    """Apply activation checkpointing to the model."""
    if ac_config.mode == "memory_budget":
        assert model_compile_enabled, "Memory budget mode requires model to be compiled"
        torch._functorch.config.activation_memory_budget = ac_config.memory_budget
        return

    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(
            transformer_block, ac_config, base_fqn=f"layers.{layer_id}", ...
        )
        model.layers.register_module(layer_id, transformer_block)
```

支持四种模式：

| 模式 | 机制 | 效果 |
|------|------|------|
| `"full"` | `checkpoint_wrapper()` | 丢弃所有中间激活，反向时重新计算 |
| `"selective"` + `"op"` | `create_selective_checkpoint_contexts()` | 按算子级别决定保存/重算 |
| `"selective"` + `"N"` | 每 N 层 `checkpoint_wrapper()` | 稀疏层级检查点 |
| `"memory_budget"` | `activation_memory_budget` config | 编译器级别内存预算 |

### 6.2 为什么 AC 必须在 TP/CP 之后

**原因 A：选择性 AC 的 op 列表包含 TP/CP 的通信算子**

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:57-81`

```python
def _get_op_sac_save_list():
    return {
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,   # TP 的 reduce-scatter
        torch.ops._c10d_functional.all_to_all_single.default,       # CP 的 all-to-all
        torch._higher_order_ops.flex_attention,
        ...
    }
```

这些算子**只有在 TP 和 CP 被应用之后**才存在于计算图中。如果 AC 先于 TP/CP，选择性检查点策略将无法感知这些通信算子，因此无法对其做出正确的 save/recompute 决策。

**原因 B：`checkpoint_wrapper` 改变模块层次**

`checkpoint_wrapper` 创建 `CheckpointWrapper` 模块，将原始 block 作为 `_checkpoint_wrapped_module` 包含其中。如果 AC 在 TP 之前执行，`parallelize_module` 需要穿透 `CheckpointWrapper` 才能找到线性层。TP plan 使用路径如 `"attention.wq"`，这些路径**无法穿透 wrapper 的间接引用解析**。

代码中有明确的排序注释：

```python
# AC must be after TP/CP           (qwen2/parallelize.py:123)
# AC must be after TP/CP           (qwen3/parallelize.py:155)
```

### 6.3 为什么 AC 必须在 compile 之前

**原因 A：compile 需要捕获 AC 的重计算逻辑**

`torch.compile` 追踪模块的 forward 函数并生成优化代码。AC 的 `checkpoint_wrapper` 在 forward 中插入重计算逻辑。如果 compile 先于 AC：
- compile 编译的是**没有重计算逻辑的原始 forward**
- 之后用 AC 包裹已编译的模块时，重计算逻辑无法正确与编译图交织
- `fullgraph=True` 下，编译图是不透明的，AC 的 wrapper 无法有效地介入

**原因 B：`memory_budget` 模式显式要求 compile**

```python
# activation_checkpoint.py:269
assert model_compile_enabled, "Memory budget mode requires model to be compiled"
```

`memory_budget` 模式设置 `torch._functorch.config.activation_memory_budget`，这个配置**被 compile 框架的 autograd 引擎消费**。AC 设置预算，compile 依据预算做优化——两者有明确的数据流依赖。

---

## 7. 第四步：torch.compile——为什么在 AC 之后、FSDP 之前

### 7.1 compile 做了什么

**文件**: `areal/experimental/models/archon/compile.py:24-45`

```python
def apply_compile(model: Compilable) -> None:
    """Must be called AFTER TP and AC, BEFORE FSDP."""
    for name, block in model.layers.items():
        model.layers[name] = torch.compile(
            block, backend="inductor", fullgraph=True,
        )
```

关键参数是 `fullgraph=True`——要求整个 block 能被捕获为**单一计算图**，没有 graph break。

对于 MoE 模型（Qwen3），compile 更为精细：

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:531-629`

```python
def _apply_compile(model, ep_enabled=False):
    """Must be called AFTER TP and AC, BEFORE FSDP."""
    for name, block in model.layers.items():
        if getattr(block, "moe_enabled", False):
            # MoE 层: 逐子模块编译, 避免 FSDP(GroupedExperts) 引起的 graph break
            if isinstance(block, CheckpointWrapper):
                inner_block = block._checkpoint_wrapped_module   # 感知 AC 的 wrapper!
            else:
                inner_block = block
            # 逐子模块编译, 跳过 experts（有 graph break）
            for attr_name, submod in inner_block.named_children():
                if isinstance(submod, moe_module.MoE):
                    for moe_attr, moe_submod in submod.named_children():
                        if moe_attr == "experts":
                            continue  # 跳过 experts
                        setattr(submod, moe_attr,
                            torch.compile(moe_submod, backend="inductor", fullgraph=True))
                else:
                    setattr(inner_block, attr_name,
                        torch.compile(submod, backend="inductor", fullgraph=True))
        else:
            # 非 MoE: 整块编译
            model.layers[name] = torch.compile(block, backend="inductor", fullgraph=True)
```

**注意**：MoE 的 compile 显式检查 `isinstance(block, CheckpointWrapper)` 来穿透 AC wrapper——这证明 **compile 感知并依赖 AC 的存在**。

### 7.2 为什么 compile 必须在 FSDP 之前

**原因 A：FSDP 引入 graph-breaking hook**

FSDP 的 `fully_shard()` 安装 pre-forward 和 post-forward hook，执行参数 all-gather（unshard）和 reduce-scatter（reshard）。这些 hook 涉及：
- 异步集合通信操作
- 参数的动态物化/去物化（参数在分片形态和完整形态之间切换）
- 通信的动态缓冲区管理

`fullgraph=True` 下，`torch.compile` **无法追踪这些 FSDP hook**，因为它们包含 Python 级别的控制流、动态张量大小和通信原语。如果 compile 在 FSDP 之后执行，每个 block 的 FSDP hook 都会导致**即时 graph break**，使 `fullgraph=True` 不可能实现。

**原因 B：FSDP2 专为包裹已编译模块设计**

PyTorch 的 FSDP2 (`fully_shard`) 被设计为在已编译模块的**外层**安装 hook。当 FSDP 包裹一个 `torch.compile` 产生的 `OptimizedModule` 时，它的 hook 运行在编译 forward 的**外层**——编译内核作为不透明单元在 FSDP 的 unshard 和 reshard 操作之间运行。这是 PyTorch 分布式 + 编译栈的**预期组合模式**。

**原因 C：MoE 代码中的显式证据**

Qwen3 的 compile 逻辑中有注释（`qwen3/infra/parallelize.py:534`）：

```python
# MoE 层: 逐子模块编译以避免 FSDP(GroupedExperts) hook 引起的 graph break
# FSDP on GroupedExperts uses torch._dynamo.disable
```

这揭示了 FSDP 在 `GroupedExperts` 上使用 `torch._dynamo.disable` 来抑制编译——确认 **FSDP hook 与编译本质上不兼容**，必须在编译之后应用。

---

## 8. 第五步：FSDP——为什么必须最后执行

### 8.1 FSDP 做了什么

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:300-431`

```python
def apply_fsdp(model, dp_mesh, ...):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # 1. 包裹 embedding
    fully_shard(model.tok_embeddings, **fsdp_config, ...)

    # 2. 包裹每个 Transformer Block
    for transformer_block in model.layers.values():
        # MoE experts 用不同 mesh (dp_mod_ep) 分片
        if moe_enabled and ep_degree > 1 and dp_mod_ep_mesh:
            fsdp_ep_config = fsdp_config.copy()
            fsdp_ep_config["mesh"] = dp_mod_ep_mesh
            fully_shard(transformer_block.moe.experts, **fsdp_ep_config, ...)
        fully_shard(transformer_block, **fsdp_config, ...)

    # 3. 包裹最终层（norm + output/score）
    fully_shard(final_layers, **fsdp_config, ...)

    # 4. 包裹根模型
    fully_shard(model, **fsdp_config)

    # 5. EP 场景下设置显式 prefetch
    if ep_degree > 1:
        _setup_fsdp_prefetch(model)
```

每次 `fully_shard()` 调用会：
1. **将所有参数分片** — 参数形状从 `[H, D]` 变为 `[H/fsdp_size, D]`
2. **安装通信 hook** — pre-forward: all-gather; post-forward: reduce-scatter; backward 类似
3. **配置混合精度** — 在存储 dtype 和计算 dtype 之间转换
4. **（可选）设置 CPU offloading** — 计算间将参数移至 CPU

### 8.2 为什么 FSDP 之后不能再做任何变换

**原因 A：参数变为分片 DTensor**

FSDP 之后，`module.weight` 不再是普通 `nn.Parameter`——它是一个**分片 DTensor**。任何检查参数形状、名称或存储的变换都会看到分片后的表示。TP 的 `parallelize_module` 会失败：
- 它通过名称匹配参数（如 `"layers.*.self_attn.q_proj"`），但期望完整大小的张量
- 它会尝试对**已经分片**的参数再次分片，导致双重分片或布局冲突

**原因 B：模块层次结构被冻结**

FSDP 创建了特定的 wrapper 层次和 pre/post forward hook。任何后续的 `model.layers.register_module()`（AC 的操作）或 `model.layers[name] = torch.compile(block)`（compile 的操作）会：
- 用未包裹的模块替换 FSDP 包裹的模块，丢失分片信息
- 破坏 FSDP 的参数到模块映射的内部状态

**原因 C：通信拓扑已建立**

FSDP 设置了显式的 prefetch 链（特别是 EP 场景）。这些链引用**具体的模块实例**：

```python
# parallelize.py:451-462
model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])
block.set_modules_to_forward_prefetch([next_block, next_block.moe.experts])
```

FSDP 之后修改模块层次会破坏这些引用。

### 8.3 唯一的 FSDP 后操作：权重绑定

代码中**唯一**在 FSDP 之后的变更是权重绑定：

```python
# parallelize.py:189-191
if getattr(model.model_args, "enable_weight_tying", False):
    if model.output is not None and model.tok_embeddings is not None:
        model.output.weight = model.tok_embeddings.weight
```

这是支持的操作，因为它操作的是 FSDP 感知的 DTensor 引用。

---

## 9. 依赖关系 DAG

### 9.1 约束总结

```
TP  ────→  CP  ────→  AC  ────→  compile  ────→  FSDP
│          │          │           │                │
│          │          │           │                └─ 分片参数，安装通信 hook
│          │          │           └─ 追踪 forward 为优化内核
│          │          └─ 用重计算逻辑包裹模块
│          └─ 配置注意力 All-to-All 通信
└─ 用 DTensor 替换参数，安装再分布 hook
```

### 9.2 每阶段的不变量

| 阶段 | 输入不变量 | 输出不变量 |
|------|-----------|-----------|
| **TP** | 原始模型，完整大小参数 | 参数为 DTensor；模块有再分布 hook |
| **CP** | TP 分割后的本地 head 数量可用 | 注意力模块配置了 CP 进程组 |
| **AC** | 所有通信算子（TP reduce-scatter, CP all-to-all）存在于 forward 图中 | 模块包裹了检查点逻辑；选择性策略引用正确算子 |
| **compile** | 模块 forward 包含 AC 重计算和通信；无 FSDP hook | forward 编译为优化内核，`fullgraph=True` |
| **FSDP** | 所有模块变换完成 | 参数分片；通信 hook 安装；模型可训练 |

### 9.3 ASCII 流程图

```
 ┌─────────────────────────────────────────────────────────────────────────────────────┐
 │                         并行化初始化顺序 (ArchonEngine)                              │
 └─────────────────────────────────────────────────────────────────────────────────────┘

 ┌───────────────────┐
 │    原始模型         │  nn.Module + nn.Parameter (完整大小张量)
 │  (Raw Model)       │
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐     变换效果                          为什么必须第一
 │  Step 1: TP        │     ● Parameter → DTensor (不可逆)    ● 后续步骤依赖 DTensor 布局
 │  张量并行           │     ● 安装 PrepareModuleInput hook    ● CP 需要 TP 本地 head 数
 │  parallelize_      │     ● ColwiseParallel / RowwiseP.     ● AC 的 plan 路径需穿透
 │  module()          │     ● SequenceParallel on norms         原始模块结构
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐     变换效果                          为什么在 TP 后
 │  Step 2: CP        │     ● 配置 Attention All-to-All       ● 验证需 TP 本地 head 数:
 │  上下文并行         │       进程组 (set_cp_group)              q_heads = n_heads // tp
 │  Ulysses SP        │     ● 运行时执行:                     ● All-to-All 必须嵌套在
 │  apply_cp()        │       gather_seq_scatter_heads            TP 的再分布 hook 之间
 │                    │       gather_heads_scatter_seq
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐     变换效果                          为什么在 TP/CP 后
 │  Step 3: AC        │     ● checkpoint_wrapper 包裹         ● 选择性 AC op 列表包含:
 │  激活检查点         │       每个 TransformerBlock              reduce_scatter (TP)
 │  checkpoint_       │     ● 改变模块层次:                      all_to_all (CP)
 │  wrapper()         │       Block → CheckpointWrapper         这些算子仅在 TP/CP 后存在
 │                    │         └─ _checkpoint_wrapped_module  ● wrapper 会阻断 TP plan 路径
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐     变换效果                          为什么在 AC 后、FSDP 前
 │  Step 4: compile   │     ● forward → inductor 优化内核     ● 需捕获 AC 重计算逻辑
 │  torch.compile     │     ● fullgraph=True (无 graph break) ● 显式检查 CheckpointWrapper
 │  backend=inductor  │     ● MoE: 逐子模块编译               ● FSDP hook 会导致
 │                    │       (跳过 experts 避免 break)          fullgraph=True 失败
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐     变换效果                          为什么必须最后
 │  Step 5: FSDP      │     ● 参数分片: [H,D] → [H/N,D]      ● 之后 TP 遇到 FlatParameter
 │  全分片数据并行      │     ● 安装 unshard/reshard hook          plan 匹配失败
 │  fully_shard()     │     ● 设置 prefetch 链                ● register_module 会破坏
 │                    │     ● 混合精度 + CPU offload             分片状态和 hook 引用
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐
 │   可训练的并行模型   │  参数已分片 + 通信 hook 已安装 + 计算图已优化
 │  (Ready to Train)  │
 └───────────────────┘


 ═══════════════════════════════════════════════════════════════════════════════════════

 依赖约束链 (每条箭头表示"必须在...之前"):

     TP ──────► CP ──────► AC ──────► compile ──────► FSDP
     │          │          │           │                │
     │          │          │           │                └─ 分片参数, 安装通信 hook
     │          │          │           │                   冻结模块层次
     │          │          │           │
     │          │          │           └─ 追踪 forward 为优化内核
     │          │          │              fullgraph=True 不容忍 FSDP hook
     │          │          │
     │          │          └─ 用重计算逻辑包裹模块
     │          │             选择性策略需引用 TP/CP 通信算子
     │          │
     │          └─ 配置 Attention All-to-All 通信
     │             验证依赖 TP 的本地 head 数
     │
     └─ 用 DTensor 替换参数
        安装再分布 hook

 ═══════════════════════════════════════════════════════════════════════════════════════

 反向依赖 (如果违反顺序会怎样):

     FSDP 在 TP 前?  ──► parallelize_module 遇到 FlatParameter → 路径匹配失败 ✗
     CP 在 TP 前?    ──► 用全局 head 数验证 → 配置错误 ✗
     AC 在 TP 前?    ──► TP plan 无法穿透 CheckpointWrapper → 找不到模块 ✗
     compile 在 AC 前? ─► 编译原始 forward, AC 无法介入编译图 → AC 失效 ✗
     FSDP 在 compile 前? ► FSDP hook 导致 fullgraph=True graph break → 编译失败 ✗
```

---

## 10. 两条引擎路径的对比

AReaL 有两条引擎路径，并行化实现有显著差异：

| 方面 | FSDPEngine | ArchonEngine |
|------|-----------|-------------|
| **模型来源** | HuggingFace `from_pretrained` | 自定义 Archon 模型类 |
| **TP 机制** | `parallelize_module()` 作用于 HF 子模块 | `parallelize_module()` 作用于自定义模型层 |
| **CP 机制** | **全局猴子补丁**（替换 flash attention 函数） | 逐层 `set_cp_group()` |
| **AC 机制** | HF `gradient_checkpointing_enable()`（Flag 方式） | PyTorch `checkpoint_wrapper()`（4 种模式） |
| **torch.compile** | **不支持**（配置项存在但未实现） | 逐 block 编译，MoE 感知 |
| **FSDP 包裹** | 基于类名的自动包裹 | 显式逐组件包裹 |
| **EP 支持** | 仅 mesh 级别 | 完整 EP（token dispatch + expert 分片） |
| **PP 支持** | 不支持（`assert pp_size == 1`） | 完整 PP |
| **实际执行顺序** | AC → CP → TP → FSDP | TP → CP → AC → compile → FSDP |

### 10.1 FSDPEngine 的顺序差异

FSDPEngine 的实际执行顺序是 **AC → CP(monkey-patch) → TP → FSDP**，与规范顺序不同：

```python
# fsdp_engine.py
self._create_device_model()    # AC 在这里通过 HF API 发生
apply_monkey_patch(...)        # CP 通过猴子补丁发生
parallelize_model(...)         # TP + FSDP 在这里发生
```

这在 FSDPEngine 中是**功能安全的**，因为 HF 的 `gradient_checkpointing_enable()` 只设置一个 flag 动态修改 forward 行为（不是模块 wrapper），monkey-patch CP 也是运行时替换函数而非改变模块结构。但这与 ArchonEngine 的严格模块包裹方式在架构上不一致。

---

## 11. 反模式分析：违反顺序会怎样

### 11.1 如果 FSDP 在 TP 之前

```
❌ parallelize_module() 遇到 FlatParameter，
   无法通过 "layers.*.self_attn.q_proj" 路径匹配参数
   → 参数匹配失败，TP plan 无法应用
```

### 11.2 如果 CP 在 TP 之前

```
❌ validate_cp_constraints() 使用全局 head 数而非 TP 本地数
   → 可能允许无效配置或拒绝有效配置
❌ TP 的 PrepareModuleInput hook 与 CP 的 All-to-All 冲突
   → 注意力输入布局不正确，产生数值错误
```

### 11.3 如果 AC 在 TP 之前

```
❌ TP plan 路径 "attention.wq" 无法穿透 CheckpointWrapper
   → parallelize_module() 找不到目标模块
❌ 选择性 AC 的 op 列表中 reduce_scatter/all_to_all 尚不存在
   → 选择性策略退化为全部重算，丧失性能优势
```

### 11.4 如果 compile 在 FSDP 之后

```
❌ FSDP hook 中的 all-gather/reduce-scatter 导致即时 graph break
   → fullgraph=True 直接失败
❌ FSDP 的 torch._dynamo.disable 在某些模块上禁用编译
   → 编译范围不完整，无法实现预期优化
```

### 11.5 如果 FSDP 在 compile 之前

```
✓ 功能上可能工作（FSDP2 设计为可在 compile 前使用）
⚠ 但 fullgraph=True 不可能（FSDP hook 导致 graph break）
⚠ 需要切换到 fullgraph=False，大幅降低编译优化效果
```

### 11.6 如果 compile 在 AC 之前

```
❌ compile 编译原始 forward（无重计算逻辑）
❌ 之后 AC 包裹编译模块时，重计算无法正确介入编译图
❌ memory_budget 模式的 activation_memory_budget config 被 compile 忽略
   → AC 形同虚设
```

---

## 附录 A：代码中所有排序相关注释和断言

| 文件位置 | 注释/断言 |
|---------|----------|
| `qwen2/infra/parallelize.py:85-91` | Docstring: "Order of operations: 1.TP, 2.CP, 3.AC, 4.compile, 5.FSDP" |
| `qwen2/infra/parallelize.py:123` | `# AC must be after TP/CP` |
| `qwen2/infra/parallelize.py:132` | `# torch.compile must be after AC, before FSDP` |
| `qwen3/infra/parallelize.py:100-106` | Docstring: "Order of operations: 1.non-MoE TP, 2.MoE EP+TP, 3.CP, 4.AC, 5.compile, 6.FSDP" |
| `qwen3/infra/parallelize.py:155` | `# AC must be after TP/CP` |
| `qwen3/infra/parallelize.py:164` | `# torch.compile must be after AC, before FSDP` |
| `compile.py:31` | `"Must be called AFTER TP and AC, BEFORE FSDP."` |
| `qwen3/infra/parallelize.py:537` | `"Must be called AFTER TP and AC, BEFORE FSDP."` |
| `activation_checkpoint.py:269` | `assert model_compile_enabled, "Memory budget mode requires model to be compiled"` |

**注意**：目前唯一的**运行时断言**是 `memory_budget` 模式要求 compile 启用。其余排序约束仅通过代码结构和注释保证。

## 附录 B：关键源文件索引

| 文件 | 内容 |
|------|------|
| `areal/experimental/models/archon/qwen3/infra/parallelize.py` | Qwen3 完整并行化（含 MoE） |
| `areal/experimental/models/archon/qwen2/infra/parallelize.py` | Qwen2 并行化（Dense） |
| `areal/experimental/models/archon/compile.py` | torch.compile 应用（Dense） |
| `areal/experimental/models/archon/activation_checkpoint.py` | AC 配置和应用 |
| `areal/experimental/models/archon/utils.py` | TP/CP 约束验证 |
| `areal/experimental/models/archon/parallel_dims.py` | DeviceMesh 构建 |
| `areal/engine/fsdp_utils/parallel.py` | FSDPEngine 的 TP + FSDP |
| `areal/engine/fsdp_utils/__init__.py` | `apply_fsdp2` 通用 FSDP 包裹 |
| `areal/engine/fsdp_engine.py` | FSDPEngine 初始化流程 |
