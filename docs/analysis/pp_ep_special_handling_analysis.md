# PP 与 EP 为何不在主并行化初始化顺序中——深度源码分析

## 目录

- [1. 问题背景](#1-问题背景)
- [2. 核心结论](#2-核心结论)
- [3. PP（流水线并行）——在序列之上，而非序列之中](#3-pp流水线并行在序列之上而非序列之中)
- [4. EP（专家并行）——融入 TP 步骤，而非独立步骤](#4-ep专家并行融入-tp-步骤而非独立步骤)
- [5. 完整初始化顺序 ASCII 图](#5-完整初始化顺序-ascii-图)
- [6. DeviceMesh 维度分析](#6-devicemesh-维度分析)
- [7. 反模式分析：如果把 PP/EP 硬塞进主序列会怎样](#7-反模式分析如果把-ppep-硬塞进主序列会怎样)
- [附录 A：关键源文件索引](#附录-a关键源文件索引)

---

## 1. 问题背景

AReaL 的并行化文档描述了一个 5 步初始化顺序：

```
TP → CP → AC → compile → FSDP
```

但框架实际支持 **7 种**并行化/优化技术：TP、CP、AC、compile、FSDP、**PP**、**EP**。

PP 和 EP 为什么不在这个序列中？它们是被遗忘了，还是有本质上不同的处理方式？

**答案：两者都不是"缺失"，而是在本质不同的抽象层次上运作。**

---

## 2. 核心结论

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    三个抽象层次                                    │
 ├──────────────┬──────────────────┬────────────────────────────────┤
 │   层次        │   成员            │   操作对象                      │
 ├────���─────────┼──────────────────┼────────────────────────────────┤
 │ 结构分解层    │   PP             │ 模型拓扑（哪些层存在）             │
 │ (engine 级)  │                  │ 深拷贝 + 裁剪 → N 个独立模型     │
 ├──────────────┼──────────────────┼────────────────────────────────┤
 │ 参数/计算     │   TP, EP, CP,    │ 参数存储、计算图、内存布局         │
 │ 变换层        │   AC, compile,   │ DTensor, hook, wrapper         │
 │ (model 级)   │   FSDP           │ 就地变换单个模型实例              │
 ├──────────────┼──────────────────┼────────────────────────────────┤
 │ 执行编排层    │   PP Schedules   │ 前向/反向执行顺序                 │
 │ (runtime 级) │  (1F1B, ZBV...)  │ 微批次调度                      │
 └──────────────┴──────────────────┴────────────────────────────────┘
```

- **PP 不在序列中**：因为它在序列**之上**——先把模型切成 N 个 stage，然后**每个 stage 独立执行** TP→EP→CP→AC→compile→FSDP 全流程
- **EP 不单独列出**：因为它**融入了 TP 步骤**——EP 本质上是"MoE 层的 TP"，与 Dense 层的 TP 属于同一类操作（`parallelize_module` + DTensor 分片），二者共享同一个逻辑阶段

---

## 3. PP（流水线并行）——在序列之上，而非序列之中

### 3.1 PP 在 Engine 层做分支决策

PP 不是 `parallelize_qwen3()` 内的一个步骤，而是在更上层的 `ArchonEngine` 中决定**走哪条路径**。

**文件**: `areal/experimental/engine/archon_engine.py:755-763`

```python
def _setup_parallelism(self, ac_config, enable_compile) -> None:
    if self.parallel_dims.pp_enabled:
        self._apply_pipeline_parallelism(ac_config, enable_compile)  # 走 PP 路径
    else:
        self._apply_parallelism(ac_config, enable_compile)           # 走普通路径
```

这是两条**互斥路径**，不是在同一个流水线中多加一步。

### 3.2 PP 的三阶段工作流

**文件**: `areal/experimental/models/archon/pipeline_parallel.py:347-492`

```python
def pipeline_llm(model, device, parallel_dims, archon_config,
                 parallelize_fn, **parallelize_kwargs):
    """
    Workflow:
    1. Generate module names for each virtual stage
    2. Split model into stages
    3. Apply parallelization (TP, FSDP) to each model part   ← 注意这里！
    """
```

**阶段 1：决定每个 stage 包含哪些层**（`generate_llm_fqn_per_model_part`）

```python
# 例如 4 层 + 2 个 PP stage:
# Stage 0: ['tok_embeddings', 'layers.0', 'layers.1']
# Stage 1: ['layers.2', 'layers.3', 'norm', 'output']
```

**阶段 2：深拷贝 + 裁剪**（`pipeline_module_split`）

**文件**: `pipeline_parallel.py:198-280`

```python
def pipeline_module_split(whole_model, pp_mesh, pp_schedule, device, module_names_per_stage):
    def _build_stage_from_modules(stage_idx, module_names, num_stages):
        model = copy.deepcopy(whole_model)         # 深拷贝整个模型
        modules_to_keep = set(module_names)
        for module_name, module_value in list(model.named_children()):
            if isinstance(module_value, nn.ModuleDict):
                for layer_key in list(module_value.keys()):
                    if layer_key not in layers_to_keep:
                        del module_value[layer_key]  # 删掉不属于这个 stage 的层
            else:
                if module_name not in modules_to_keep:
                    setattr(model, module_name, None)  # 不属于这个 stage 的设为 None
        # 封装为 PipelineStage
        return PipelineStage(model, stage_idx, num_stages, device, ...)
```

PP 是一个**结构性操作**——深拷贝模型，删掉不属于本 stage 的层。这与 TP/FSDP 等就地变换参数的操作**本质不同**。

**阶段 3：对每个 stage 独立执行完整并行化序列**

**文件**: `pipeline_parallel.py:476-481`

```python
# 3. Apply parallelization to each model part
for i, m in enumerate(model_parts):
    m = parallelize_fn(m, parallel_dims, **parallelize_kwargs)  # ← 这里！
    model_parts[i] = m
    stages[i].submod = m  # 更新 stage 引用
```

**`parallelize_fn` 就是 `parallelize_qwen3`**，它内部执行完整的 TP→EP→CP→AC→compile→FSDP 序列。

PP 对每个 model part 调用一次 `parallelize_fn`，而不是一次处理整个模型。这就是 PP 必须在序列**之外**的根本原因。

### 3.3 PP 只在 ArchonEngine 中支持

**文件**: `areal/engine/fsdp_utils/parallel.py:41`

```python
assert fsdp_ps.pp_size == 1, "Pipeline parallelism is not supported in FSDP"
```

FSDPEngine 通过硬断言拒绝 PP。原因：PP 需要模型感知的层拆分逻辑（知道 `tok_embeddings`、`layers.N`、`norm`、`output` 的结构），而 FSDPEngine 使用 HuggingFace 通用模型，缺少这种结构性知识。

### 3.4 PP 反向影响 FSDP 的行为

PP 虽然在序列之外，但作为**上下文参数**传入序列内部，影响 FSDP 的 `reshard_after_forward` 策略：

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:344-356`

```python
match reshard_after_forward_policy:
    case "default":
        # PP 场景下默认不 reshard，避免每个微批次的 all-gather 开销
        reshard_after_forward = not pp_enabled
```

PP 处理多个微批次，如果每个微批次 forward 后都 reshard，会产生大量无法与计算重叠的 all-gather。

### 3.5 PP 的 Schedule 调度

PP 不仅在初始化时拆分模型，还在运行时改变执行模式。支持的调度器：

| Schedule | 每 rank stage 数 | 特点 |
|----------|----------------|------|
| `1F1B` | 1 | 基础流水线，单 stage |
| `Interleaved1F1B` | ≥2 | 交错式，减少气泡 |
| `ZBVZeroBubble` | 2 | V 形零气泡调度 |
| `DualPipeV` | 2 | 双管道 V 形调度 |

这些调度器来自 PyTorch 的 `torch.distributed.pipelining.schedules`，在运行时通过 `schedule.step()` 编排微批次的前向/反向执行顺序——这是 TP/CP/AC/compile/FSDP 完全不涉及的运行时执行模型。

---

## 4. EP（专家并行）——融入 TP 步骤，而非独立步骤

### 4.1 EP 实际上已经在序列中了

在 `parallelize_qwen3` 的文档字符串中，EP 是**步骤 2**：

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:100-106`

```python
"""
Order of operations:
1. Apply non-MoE TP (Tensor Parallelism for dense layers)      ← Dense 层的 TP
2. Apply MoE EP+TP (Expert Parallelism + MoE-specific TP)      ← MoE 层的 EP+TP
3. Apply CP (Context Parallelism / Ulysses SP)
4. Apply AC (Activation Checkpointing) - must be after TP/EP   ← 注意: "TP/EP"
5. Apply torch.compile - must be after AC, before FSDP
6. Apply FSDP (Fully Sharded Data Parallelism)
"""
```

EP 并非"缺失"，而是与 TP 合并为同一个**逻辑阶段**：`张量级分区`。

### 4.2 EP 本质上是"MoE 层的 TP"

对比 `apply_non_moe_tp()` 和 `apply_moe_ep_tp()`——两者使用**同一个 PyTorch API**：

**Dense TP**（`apply_non_moe_tp`，第 291-295 行）：

```python
parallelize_module(
    module=transformer_block,
    device_mesh=tp_mesh,
    parallelize_plan=layer_plan,          # ColwiseParallel, RowwiseParallel, ...
)
```

**MoE EP**（`apply_moe_ep_tp`，第 730-735 行）：

```python
parallelize_module(
    module=moe.experts,
    device_mesh=experts_mesh,             # tp_mesh 或 ep_mesh 或 ep_tp_mesh
    parallelize_plan=experts_plan,        # ExpertParallel, TensorParallel, 或 ExpertTensorParallel
)
```

两者都调用 `parallelize_module()`，都将参数转换为 DTensor，都安装通信 hook。区别仅在于：

| 对比维度 | Dense TP (`apply_non_moe_tp`) | MoE EP+TP (`apply_moe_ep_tp`) |
|---------|------|------|
| **目标模块** | Attention, FFN, Norms, Embedding | MoE Experts, Router, MoE I/O |
| **分片方式** | `ColwiseParallel`, `RowwiseParallel` | `ExpertParallel`, `TensorParallel`, `ExpertTensorParallel` |
| **通信模式** | 隐式（DTensor 自动 all-reduce/reduce-scatter） | 显式 All-to-All forward hook（动态 token dispatch） |
| **Mesh** | 始终 `tp_mesh` | `tp_mesh` 或 `ep_mesh` 或 2D `ep_tp_mesh` |

### 4.3 EP 的 5 种策略

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:646-655`

```python
"""
| EP  | TP  | etp | Strategy              | Expert Weight Sharding         |
|-----|-----|-----|-----------------------|--------------------------------|
| 1   | 1   | -   | None                  | Replicate                      |
| 1   | >1  | -   | TensorParallel        | [Shard(1/2)]                   |
| >1  | 1   | -   | ExpertParallel        | [Shard(0)]                     |
| >1  | >1  | 1   | ExpertParallel        | [Shard(0)] (TP borrowed by EP) |
| >1  | >1  | tp  | ExpertTensorParallel  | [Shard(0), Shard(1/2)]         |
"""
```

策略选择代码（第 719-728 行）：

```python
experts_mesh, experts_plan = None, None
if ep_mesh is None:
    experts_mesh = tp_mesh
    experts_plan = TensorParallel()              # 无 EP，用 TP 切 expert
elif tp_mesh is None or etp == 1:
    experts_mesh = ep_mesh
    experts_plan = ExpertParallel()              # 纯 EP（或 TP 被 EP 借走）
else:
    experts_mesh = ep_tp_mesh
    experts_plan = ExpertTensorParallel()        # EP + TP 2D 切分
```

EP 和 TP 在**同一个策略空间**中——它们不是正交的并行维度，而是同一类操作（张量分片）的不同实例化。

### 4.4 ExpertParallel 的具体实现

**文件**: `areal/experimental/models/archon/expert_parallel.py:68-237`

`ExpertParallel` 是 `ParallelStyle` 的子类（与 `ColwiseParallel`、`RowwiseParallel` 同级），通过 `distribute_module()` 实现：

```python
class ExpertParallel(BaseExpertParallel):
    """Expert Parallelism with ETP=1."""

    def _partition_fn(self, name, module, device_mesh):
        """Shard expert weights on expert dimension (dim 0)."""
        for param_name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(param_name, dist_param)

    def _token_dispatch(self, module, inputs, device_mesh):
        """Dispatch tokens to EP ranks via all-to-all."""
        # Step 1: 交换 token 计数
        num_tokens_per_expert_received = all_to_all_single(num_tokens_per_expert, ...)
        # Step 2: 交换实际 token
        dispatched_input = all_to_all_single_autograd(routed_input, ...)
        # Step 3: 排列对齐
        dispatched_input = _permute(dispatched_input, ...)
        return dispatched_input, local_num_tokens_per_expert

    def _token_combine(self, module, output, device_mesh):
        """Combine expert outputs back via reverse all-to-all."""
        combined = _unpermute(output, ...)
        combined = all_to_all_single_autograd(combined, ...)
        return combined
```

与 TP 的关键区别是 EP 的通信模式是**动态的**（All-to-All 的 split 大小取决于路由结果），而 TP 是**静态的**（每次 all-reduce/reduce-scatter 大小固定）。但两者的 PyTorch API 入口是相同的。

### 4.5 EP 对 FSDP 的特殊影响

EP 虽然与 TP 合并为同一步骤，但它在 FSDP 步骤中有**特殊处理**——MoE experts 使用不同的 mesh 进行 FSDP 分片：

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:365-399`

```python
for transformer_block in model.layers.values():
    if moe_enabled and ep_degree > 1 and dp_mod_ep_mesh is not None:
        fsdp_ep_config = fsdp_config.copy()
        fsdp_ep_config["mesh"] = dp_mod_ep_mesh     # ← experts 用 dp_mod_ep mesh

        fully_shard(transformer_block.moe.experts, **fsdp_ep_config, ...)

        # 梯度除因子：保证与数据并行一致
        if gradient_divide_factor is not None:
            transformer_block.moe.experts.set_gradient_divide_factor(
                gradient_divide_factor,
            )

    fully_shard(transformer_block, **fsdp_config, ...)  # 其余部分用标准 dp_shard_cp mesh
```

**为什么需要 `gradient_divide_factor`？**

**文件**: `areal/experimental/models/archon/parallel_dims.py:361-368`

```python
@property
def fsdp_gradient_divide_factor(self) -> int:
    """Gradient divide factor for FSDP.

    Although the FSDP sharding of experts is done on a mesh of a different
    size than other parameters, the gradient division factor should be
    consistent with data parallelism degree.
    """
    return self.dp_shard * self.cp
```

FSDP 在 `reduce_scatter` 时默认除以 mesh size。experts 的 mesh (`dp_mod_ep`) 与其他参数的 mesh (`dp_shard_cp`) 大小不同，如果不覆写除因子，experts 的梯度缩放会不一致。

### 4.6 EP 不能独立为一步的根本原因

如果把 EP 从 TP 步骤中拆出来成为独立步骤，需要回答：放在哪里？

- **TP 之前？** 不行。EP 的 `PrepareModuleInputOutput` 需要 TP 的 `SequenceParallel` 输出布局信息
- **TP 之后、CP 之前？** 可以，但无意义。EP 与 TP 操作同一模块（transformer block）的不同子模块（dense FFN vs MoE experts），使用同一 API（`parallelize_module`），产生同一种产物（DTensor 参数）。拆分只会增加概念复杂度而无实际收益
- **CP 之后？** 不行。AC 的注释写 `"must be after TP/EP"`，AC 的选择性 op 列表包含 EP 的 `all_to_all_single`

EP 与 TP 共享同一个"必须在 CP/AC/compile/FSDP 之前"的约束，它们在依赖图中处于同一位置，因此被合并为同一个逻辑阶段。

---

## 5. 完整初始化顺序 ASCII 图

```
 ═══════════════════════════════════════════════════════════════════════════
                          完整初始化顺序
 ═══════════════════════════════════════════════════════════════════════════

 ┌──────────────────────────────────────────────────────────────────────┐
 │                    Engine 层 (ArchonEngine)                          │
 │                                                                      │
 │   _setup_parallelism()                                               │
 │      │                                                               │
 │      ├── PP disabled? ──→ 直接调用 parallelize_fn(model, ...)        │
 │      │                                                               │
 │      └── PP enabled?  ──→ pipeline_llm(model, ..., parallelize_fn)   │
 │                              │                                       │
 │                              ├── 1) generate_llm_fqn_per_model_part  │
 │                              │     分配层到各 stage                    │
 │                              │                                       │
 │                              ├── 2) pipeline_module_split             │
 │                              │     deepcopy + 裁剪 → N 个 model part  │
 │                              │                                       │
 │                              └── 3) for each model_part:             │
 │                                     parallelize_fn(model_part, ...)   │
 │                                         │                            │
 └─────────────────────────────────────────┼────────────────────────────┘
                                           │
                                           ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                    Model 层 (parallelize_qwen3)                      │
 │                                                                      │
 │   对单个 model/model_part 执行以下序列:                                │
 │                                                                      │
 │   ┌─────────────────────────────────────────────────────┐            │
 │   │  Step 1a: Non-MoE TP                                │            │
 │   │  apply_non_moe_tp()                                 │            │
 │   │  · parallelize_module() on attention, FFN, norms    │            │
 │   │  · ColwiseParallel, RowwiseParallel, SeqParallel    │            │
 │   │  · Parameter → DTensor                              │            │
 │   └──────────────────────┬──────────────────────────────┘            │
 │                          │                                           │
 │                          ▼                                           │
 │   ┌─────────────────────────────────────────────────────┐            │
 │   │  Step 1b: MoE EP+TP  (仅 MoE 模型)                  │            │
 │   │  apply_moe_ep_tp()                                  │            │
 │   │  · parallelize_module() on MoE experts, router      │            │
 │   │  · ExpertParallel / TensorParallel / ExpertTensorP. │            │
 │   │  · All-to-All token dispatch/combine hooks          │            │
 │   │  · 与 Step 1a 使用同一 PyTorch API                   │            │
 │   │  · 共同构成"张量级分区"逻辑阶段                        │            │
 │   └──────────────────────┬──────────────────────────────┘            │
 │                          │                                           │
 │                          ▼                                           │
 │   ┌─────────────────────────────────────────────────────┐            │
 │   │  Step 2: CP (上下文并行)                              │            │
 │   │  apply_cp()                                         │            │
 │   │  · set_cp_group() on each attention layer           │            │
 │   └──────────────────────┬──────────────────────────────┘            │
 │                          │                                           │
 │                          ▼                                           │
 │   ┌─────────────────────────────────────────────────────┐            │
 │   │  Step 3: AC (激活检查点)                              │            │
 │   │  apply_ac()                                         │            │
 │   │  · checkpoint_wrapper() on transformer blocks       │            │
 │   │  · "must be after TP/EP" ← 注意包含 EP!              │            │
 │   └──────────────────────┬──────────────────────────────┘            │
 │                          │                                           │
 │                          ▼                                           │
 │   ┌─────────────────────────────────────────────────────┐            │
 │   │  Step 4: torch.compile                              │            │
 │   │  _apply_compile()                                   │            │
 │   │  · MoE 层: 逐子模块编译 (跳过 experts 避免 break)     │            │
 │   └──────────────────────┬──────────────────────────────┘            │
 │                          │                                           │
 │                          ▼                                           │
 │   ┌─────────────────────────────────────────────────────┐            │
 │   │  Step 5: FSDP                                       │            │
 │   │  apply_fsdp()                                       │            │
 │   │  · Dense params: fully_shard(dp_shard_cp mesh)      │            │
 │   │  · MoE experts:  fully_shard(dp_mod_ep mesh)  ← EP! │            │
 │   │  · gradient_divide_factor for expert gradients       │            │
 │   │  · EP 场景: _setup_fsdp_prefetch()                   │            │
 │   └─────────────────────────────────────────────────────┘            │
 └──────────────────────────────────────────────────────────────────────┘


 ═══════════════════════════════════════════════════════════════════════════
                     PP 与 EP 的位置总结
 ═══════════════════════════════════════════════════════════════════════════

     PP ─────────── 在序列之上 (engine 层)
     │               ● 拆分模型 → N 个 stage
     │               ● 每个 stage 独立走完整序列
     │               ● 只在 ArchonEngine，FSDPEngine 不支持
     │
     └──→ 序列内部 (model 层):
            │
            ├── TP + EP ── 张量级分区 (同一逻辑阶段)
            │    ├── Dense TP:  apply_non_moe_tp()
            │    └── MoE EP+TP: apply_moe_ep_tp()
            │         ● EP 是"MoE 层的 TP"
            │         ● 使用同一 parallelize_module API
            │         ● 在 FSDP 步骤中有特殊 mesh 处理
            │
            ├── CP ──── 注意力级通信
            ├── AC ──── 模块级包裹
            ├── compile ── 计算图级优化
            └── FSDP ── 参数级分片
```

---

## 6. DeviceMesh 维度分析

PP 和 EP 在 DeviceMesh 中也有本质不同的存在方式。

### 6.1 PP 拥有独立的最外层维度

**文件**: `areal/experimental/models/archon/parallel_dims.py:183-184`

```
无 EP:  mesh = [pp, dp_shard, cp, tp]       ← PP 是最外层
有 EP:  mesh = [pp, dp_shard_mod_ep, dp_shard_in_ep, cp, tp]
```

PP 是 mesh 的最外层维度，意味着它将 GPU 划分为独立的组，每组内部独立运行 TP/CP/FSDP。

### 6.2 EP 没有独立维度——它借用其他维度

**文件**: `areal/experimental/models/archon/parallel_dims.py:215-230`

```python
def _build_mesh_with_ep(self):
    if self.etp == self.tp:
        # etp=tp: EP 借用 dp_shard_in_ep * cp
        dp_shard_mod_ep = self.dp_shard * self.cp // self.ep
        dp_shard_in_ep = self.ep // self.cp
    else:
        # etp=1:  EP 借用 dp_shard_in_ep * cp * tp
        dp_shard_mod_ep = self.dp_shard * self.cp * self.tp // self.ep
        dp_shard_in_ep = self.ep // (self.cp * self.tp)
```

EP 不增加新的 mesh 维度。总 GPU 数公式始终是 `pp × dp_shard × cp × tp = world_size`。EP 通过**重新解释** `dp_shard` 维度来获取设备：

```
 无 EP:                           有 EP (etp=tp):
 ┌─────────────────────────┐      ┌─────────────────────────────────────┐
 │ pp │ dp_shard │ cp │ tp │      │ pp │ dp_mod_ep │ dp_in_ep │ cp │ tp │
 └─────────────────────────┘      └─────────────────────────────────────┘
       ↑                                ↑            ↑
       └── 统一 DP                       └── 剩余 DP   └── 借给 EP
                                                          (与 cp 一起
                                                           构成 ep mesh)
```

这正是 EP 与 PP 的本质差异在 mesh 层面的体现：
- **PP**：拥有独立维度，增加了 mesh 的"宽度"
- **EP**：借用现有维度，改变了 mesh 的内部"划分方式"

### 6.3 EP 的 Mesh 扁平化

```python
# etp=tp 时: ep = dp_shard_in_ep × cp
mesh["dp_shard_in_ep", "cp"]._flatten(mesh_dim_name="ep")

# etp=1 时:  ep = dp_shard_in_ep × cp × tp
mesh["dp_shard_in_ep", "cp", "tp"]._flatten(mesh_dim_name="ep")

# FSDP 用于 expert 的 mesh:
mesh["dp_shard_mod_ep"]._flatten(mesh_dim_name="dp_shard_mod_ep")  # experts 的 FSDP
mesh["dp_shard_mod_ep", "dp_shard_in_ep", "cp"]._flatten(mesh_dim_name="dp_shard_cp")  # 其他参数
```

---

## 7. 反模式分析：如果把 PP/EP 硬塞进主序列会怎样

### 7.1 如果把 PP 放在 TP 之前（作为 Step 0）

```
PP → TP → CP → AC → compile → FSDP
```

这在逻辑上不成立。PP 不是"对一个模型执行一次的操作"，而是"把一个模型变成 N 个"。你不能在 `parallelize_qwen3()` 内部"先 PP 再 TP"，因为 PP 之后每个 stage 都需要**独立执行** TP/CP/AC/compile/FSDP。PP 是循环的外层，不是循环体内的一步。

### 7.2 如果把 PP 放在 FSDP 之后（作为 Step 6）

```
TP → CP → AC → compile → FSDP → PP
```

更不可行。FSDP 已经把参数分片并安装了 hook，此时再 `deepcopy + 裁剪` 会：
- 复制分片的 DTensor（每个 copy 只有 1/N 的参数）
- 破坏 FSDP 的通信 hook（hook 引用的 process group 不再匹配裁剪后的模型）

### 7.3 如果把 EP 从 TP 中拆出放在 CP 之后

```
TP → CP → EP → AC → compile → FSDP
```

不安全。原因：
- EP 的 `apply_moe_ep_tp` 中 `PrepareModuleInputOutput` 依赖 TP 的 `SequenceParallel` 输出布局
- 但更关键的是：EP 和 TP 操作同一个 transformer block 的不同子模块。CP 的 `set_cp_group` 会修改 attention 模块——如果 EP 在 CP 之后，EP 的 `PrepareModuleInputOutput` 会看到 CP 已修改的 attention 模块，可能与 CP 的 All-to-All 模式冲突

### 7.4 如果把 EP 完全独立出来（单独的 Step）

```
TP → EP → CP → AC → compile → FSDP
```

**在当前实现下功能上可行**（因为 EP 确实在 TP 之后、CP 之前执行），但没有意义：
- EP 和 TP 使用相同的 `parallelize_module` API
- EP 和 TP 的产出相同（DTensor 参数 + 通信 hook）
- EP 和 TP 共享同一个排序约束（"必须在 CP/AC/compile/FSDP 之前"）
- 拆分只会增加概念负担，制造"这是一个不同类型的操作"的错误印象

---

## 附录 A：关键源文件索引

| 文件 | PP/EP 相关内容 |
|------|--------------|
| `areal/experimental/engine/archon_engine.py:755-763` | PP vs 非 PP 路径分支 |
| `areal/experimental/models/archon/pipeline_parallel.py:83-195` | Stage 层分配 |
| `areal/experimental/models/archon/pipeline_parallel.py:198-280` | 模型深拷贝 + 裁剪 |
| `areal/experimental/models/archon/pipeline_parallel.py:347-492` | PP 主入口 `pipeline_llm` |
| `areal/experimental/models/archon/pipeline_parallel.py:476-481` | **每个 stage 独立并行化** |
| `areal/experimental/models/archon/qwen3/infra/parallelize.py:100-106` | 6 步顺序文档（含 EP） |
| `areal/experimental/models/archon/qwen3/infra/parallelize.py:632-746` | `apply_moe_ep_tp()` |
| `areal/experimental/models/archon/qwen3/infra/parallelize.py:719-735` | EP 策略选择 |
| `areal/experimental/models/archon/qwen3/infra/parallelize.py:365-399` | FSDP 中 EP 的特殊处理 |
| `areal/experimental/models/archon/expert_parallel.py:68-237` | `ExpertParallel` 实现 |
| `areal/experimental/models/archon/expert_parallel.py:262-328` | `TensorParallel`（MoE 用） |
| `areal/experimental/models/archon/expert_parallel.py:331-433` | `ExpertTensorParallel` |
| `areal/experimental/models/archon/expert_parallel.py:436-503` | `ReordererSequenceParallel` |
| `areal/experimental/models/archon/parallel_dims.py:179-213` | 无 EP 的 4D mesh |
| `areal/experimental/models/archon/parallel_dims.py:215-293` | 有 EP 的 5D mesh |
| `areal/experimental/models/archon/parallel_dims.py:361-368` | `fsdp_gradient_divide_factor` |
| `areal/engine/fsdp_utils/parallel.py:41` | FSDPEngine 拒绝 PP |
