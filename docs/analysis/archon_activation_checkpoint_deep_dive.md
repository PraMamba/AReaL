# Archon Activation Checkpoint 深度解析

> 源文件：`areal/experimental/models/archon/activation_checkpoint.py`（312 行）
> 核心导出：`ActivationCheckpointConfig` · `apply_ac`

---

[TOC]

---

# 1. 白话解释

## 1.1 一句话总结

Activation Checkpoint（激活检查点，简称 AC）是一种 **用计算换显存** 的技术：前向传播时丢弃中间激活值、反向传播时重新计算它们，从而大幅降低 GPU 显存峰值，让同一张卡能训练更大的模型或使用更大的 batch。

## 1.2 现实类比

```text
想象你是一个考试阅卷老师，有 100 份试卷要批改：

方案 A：不用 AC（保存所有草稿纸）
  → 改每道题时，把中间计算步骤全写在草稿纸上
  → 最后对答案时，直接翻草稿纸
  → 问题：草稿纸堆满桌子（= GPU 显存爆满）

方案 B：Full AC（不保存草稿纸）
  → 改完就扔掉草稿纸
  → 对答案时，重新算一遍
  → 省纸（省显存），但多花时间（多计算 ~33%）

方案 C：Selective AC（有选择地保留部分草稿纸）
  ┌──────────────────────────────────────────────────┐
  │ 仅保留 "贵" 的草稿纸（比如矩阵乘法的结果）       │
  │ 扔掉 "便宜" 的草稿纸（比如 ReLU 的输出）          │
  │ 甚至同类操作也交替保留（奇数次 mm 保留，偶数次丢弃）│
  └──────────────────────────────────────────────────┘
  → 省纸且只多花一点时间：最佳性价比

方案 D：Memory Budget AC（给编译器一个预算，让它自动决定）
  → "桌���上最多放 50% 的草稿纸，你自己决定留哪些"
  → 需要 torch.compile，编译器自动帕累托最优
```

## 1.3 这个文件做了什么

```text
activation_checkpoint.py 在整个系统中的角色：

CLI (cli_args.py)                    配置 (archon_utils.py)
   ac_mode="selective"     →    build_ac_config()
   selective_ac_option="op"           ↓
   ac_memory_budget=0.5        ActivationCheckpointConfig
                                      ↓
                              parallelize_qwen3()
                                Step 1: TP
                                Step 2: EP+TP
                                Step 3: CP
                              → Step 4: apply_ac()  ← 本文件的入口
                                Step 5: torch.compile
                                Step 6: FSDP

apply_ac() 的工作：
  遍历 model.layers，给每个 TransformerBlock 包上 CheckpointWrapper
  CheckpointWrapper 拦截 forward：
    前向时 → 不记录中间激活（torch.no_grad()）
    反向时 → 重新执行 forward 得到中间激活
```

## 1.4 核心不变量

1. **AC 必须在 TP/EP/CP 之后、torch.compile 和 FSDP 之前应用**
2. **apply_ac 仅操作 `model.layers`（ModuleDict），逐层包装**
3. **4 种模式互斥：`none` / `full` / `selective` / `memory_budget`**
4. **selective 模式有 2 个子模式：`"op"`（算子级）和数字字符串如 `"2"`（层级）**

---

# 2. 前置概念

## 2.1 为什么需要 Activation Checkpoint

在训练深度神经网络时，前向传播产生的 **中间激活值** 必须保留到反向传播才能计算梯度。对于一个 L 层 Transformer：

```text
内存消耗 ∝ L × batch_size × seq_len × hidden_dim

例：72 层 Qwen3, batch=4, seq=8192, hidden=8192
  ≈ 72 × 4 × 8192 × 8192 × 2 bytes (bf16)
  ≈ 36.5 GB  仅激活值
```

AC 的核心思想是：**不保存中间激活，需要时重新计算**。

| 方案 | 显存消耗 | 计算开销 | 适用场景 |
|------|---------|---------|---------|
| 无 AC | 100%（所有激活） | 100%（1 次前向） | 小模型/小 batch |
| Full AC | ~√L（仅层边界） | ~133%（2 次前向） | 极端显存受限 |
| Selective AC (op) | ~50-70% | ~105-115% | **生产推荐** |
| Selective AC (层级) | 取决于频率 | 取决于频率 | 粗粒度折中 |
| Memory Budget | 自动调节 | 自动调节 | 需 torch.compile |

## 2.2 PyTorch 的两种 Checkpoint 机制

### 2.2.1 `checkpoint_wrapper`（全量包装）

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

# 包装一个模块：
wrapped = ptd_checkpoint_wrapper(module, preserve_rng_state=False)

# 等价于：
# wrapped.forward(x) 时：
#   前向：在 torch.no_grad() 下运行原始 forward，只保存输入
#   反向：重新执行 forward 得到中间激活，再计算梯度
```

关键参数：
- `preserve_rng_state`：保存/恢复 RNG 状态（确保 dropout 等随机操作在重算时结果一致）
- `determinism_check`：验证重算结果与原始一致（`"default"` / `"none"` / `"throw"`）
- `early_stop`：所有需要的激活都已重建后提前停止重算
- `context_fn`：**选择性 AC 的注入点**（见下文）

### 2.2.2 `create_selective_checkpoint_contexts`（选择性策略）

```python
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

def my_policy(ctx, func, *args, **kwargs):
    if func == torch.ops.aten.mm.default:
        return CheckpointPolicy.MUST_SAVE      # 保留此 op 的输出
    return CheckpointPolicy.PREFER_RECOMPUTE    # 丢弃此 op 的输出

# 作为 context_fn 传入 checkpoint_wrapper：
ptd_checkpoint_wrapper(module, context_fn=lambda: create_selective_checkpoint_contexts(my_policy))
```

这是更精细的 API：对前向传播中 **每个 ATen 算子** 单独决定"保留"还是"重算"。

## 2.3 `_get_op_sac_save_list()`：模型特定的保留列表

定义在 `qwen3/infra/parallelize.py`（57-81 行，其中 57-60 行为函数签名和 import，61-81 行为 op 集合），列出了 **值得保留** 的 op：

```python
def _get_op_sac_save_list() -> set[torch._ops.OpOverload]:
    return {
        torch.ops.aten.mm.default,                              # 矩阵乘法
        torch.ops.aten._scaled_dot_product_efficient_attention.default,  # 注意力
        torch.ops.aten._scaled_dot_product_flash_attention.default,      # Flash Attention
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,      # cuDNN 注意力
        torch.ops.aten._scaled_dot_product_attention_math.default,       # 数学注意力
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,        # FSDP 通信
        torch.ops._c10d_functional.all_to_all_single.default,            # MoE EP 通信
        torch.ops.aten.max.default,                              # 量化缩放因子
        torch._higher_order_ops.flex_attention,                  # Flex Attention
        torch.ops.areal._varlen_attn.default,                   # AReaL 自定义 op
    }
```

> **设计分离**：AC 模块是模型无关的；哪些 op 值得保留是模型特定的知识，放在 parallelize 模块中。

## 2.4 配置流转

```text
用户命令行                   ArchonEngineConfig (cli_args.py:445-479)
  --ac_mode selective    →     ac_mode: str = "selective"
  --selective_ac_option op →   selective_ac_option: str = "op"
  --ac_memory_budget 0.5  →   ac_memory_budget: float = 0.5
  --ac_preserve_rng_state →   ac_preserve_rng_state: bool = False
  --ac_debug              →   ac_debug: bool = False
                                       ↓
                          build_ac_config() (archon_utils.py:108-140)
                            检查 gradient_checkpointing 是否开启
                            检查 mode 是否为 "none"
                            构建 ActivationCheckpointConfig 实例
                                       ↓
                          parallelize_qwen3() Step 4 (parallelize.py:156-162)
                            if ac_config is not None and ac_config.mode != "none":
                                apply_ac(model, ac_config, ...)
```

## 2.5 AC 在并行化流水线中的位置及原因

```text
parallelize_qwen3() 的 6 步流水线 (parallelize.py:100-106 注释，实现在 132-187 行):

  Step 1: TP   → 分片权重，注册 DTensor hooks
  Step 2: EP   → 分片专家权重，注册 dispatch/combine hooks
  Step 3: CP   → 配置 Ulysses 注意力组
  ─── AC 必须在这里 ───
  Step 4: AC   → 用 CheckpointWrapper 包装 TransformerBlock
  ─── 编译必须在 AC 之后 ───
  Step 5: compile → torch.compile 编译 TransformerBlock
  ─── FSDP 必须在最外层 ───
  Step 6: FSDP  → fully_shard 分片参数
```

**为什么 AC 在 TP/EP/CP 之后？**
- CheckpointWrapper 包装 module 的 forward 方法
- TP/EP/CP 通过 `distribute_module` 注册的 hooks 在内层 module 上
- 反向传播时 CheckpointWrapper 重新执行内层 forward，TP hooks 被正确重新调用

**为什么 AC 在 compile 之前？**
- CheckpointWrapper 创建 autograd 边界
- 编译器必须感知这些边界，否则可能跨边界融合 op 导致 AC 失效

**为什么 AC 在 FSDP 之前？**
- FSDP 的 pre-forward/post-forward hooks 管理参数 all-gather 和 reshard
- 如果 AC 在 FSDP 之后，checkpoint 重算时会触发冗余的 all-gather

## 2.6 CheckpointWrapper 对 state_dict 的影响

```python
# 包装前：
model.layers.0.attention.wq.weight

# 包装后：
model.layers.0._checkpoint_wrapped_module.attention.wq.weight
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           CheckpointWrapper 插入的前缀

# state_dict_adapter.py 的修复（qwen3/model/state_dict_adapter.py:215-217）：
name = name.replace("._checkpoint_wrapped_module", "")
```

## 2.7 Zero-Bubble Schedule 兼容性

```text
Zero-bubble 流水线调度（如 ScheduleInterleavedZeroBubble）使用 split backward + retain_graph=True，
与某些 AC 模式冲突：

  冲突模式           →  自动降级策略 (archon_utils.py:192-201)
  ─────────────────────────────────────────────────────
  selective (op 级)  →  降级为 full AC
  memory_budget      →  降级为 full AC
  selective (层级)   →  不冲突，正常使用
  full               →  不冲突，正常使用
```

---

# 3. 源码逐行地图

## 3.1 文件结构总览

```text
activation_checkpoint.py (312 行)
├── import 区域 (1-20)
├── _get_logger() (23-27)           ← rank-aware 日志
├── _layer_sac_count (31)           ← 全局层计数器
├── ActivationCheckpointConfig (34-82) ← 配置 dataclass
├── _apply_layer_sac (85-110)       ← 层级选择性 AC
├── _apply_op_sac (113-194)         ← 算子级选择性 AC (核心)
├── _apply_full_ac (197-216)        ← 全量 AC
├── _apply_ac_to_transformer_block (219-244) ← 路由分发
├── apply_ac (247-305)              ← 主入口 ★
└── __all__ (308-311)               ← 公开 API
```

## 3.2 `_get_logger()`（23-27 行）

```python
@functools.cache  # 同一 rank 只创建一次 logger
def _get_logger() -> logging.Logger:
    """Get rank-aware logger for this module."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    return logging.getLogger(f"[Archon ActivationCheckpoint Rank {rank}]")
```

- 使用 `@functools.cache` 保证 logger 只创建一次
- rank-aware：每个 GPU 进程的日志带 rank 编号
- 遵循项目规范：使用 `areal.utils.logging.getLogger`

## 3.3 `ActivationCheckpointConfig`（34-82 行）

```python
@dataclass
class ActivationCheckpointConfig:
    mode: str = "selective"
    # 可选值: "none" | "full" | "selective" | "memory_budget"

    selective_ac_option: str = "op"
    # "op" → 算子级选择性 AC
    # "1"  → 每层都 checkpoint（等同 full）
    # "2"  → 每 2 层 checkpoint 一次
    # "N"  → 每 N 层 checkpoint 一次

    per_op_sac_force_recompute_mm_shapes_by_fqns: list[str] = field(
        default_factory=lambda: ["moe.router.gate"]
    )
    # 强制重算这些 nn.Linear 的矩阵乘法（按 FQN 匹配）

    early_stop: bool = False        # 所有激活重建后提前停止重算
    memory_budget: float = 0.5      # memory_budget 模式的预算 (0.0~1.0)
    visualize_memory_budget_pareto: bool = False  # 可视化帕累托曲线
    preserve_rng_state: bool = False  # 保存 RNG 状态（更慢但确定性）
    determinism_check: str = "default"  # 确定性检查
    debug: bool = False              # 调试信息
```

### `__post_init__` 验证（67-82 行）

```python
def __post_init__(self):
    # 1. 验证 mode 合法性
    valid_modes = ("none", "full", "selective", "memory_budget")
    if self.mode not in valid_modes:
        raise ValueError(...)

    # 2. 仅在 selective 模式下验证 selective_ac_option
    if self.mode == "selective":
        if (
            self.selective_ac_option != "op"
            and not self.selective_ac_option.isdigit()
        ):
            raise ValueError(...)
```

**注意**：`selective_ac_option` 的验证仅在 `mode == "selective"` 时触发。其他模式下可以是任意值。

## 3.4 `_apply_layer_sac()`（85-110 行）— 层级选择性 AC

```python
# 全局计数器
_layer_sac_count = 0   # 第 31 行

def _apply_layer_sac(
    module: nn.Module,
    ac_config: ActivationCheckpointConfig,
) -> nn.Module:
    global _layer_sac_count
    _layer_sac_count += 1                    # 每调用一次 +1
    ac_freq = int(ac_config.selective_ac_option)  # 例如 "2" → 2

    if not ac_freq or _layer_sac_count % ac_freq == 0:
        # ac_freq=0 → 每层都 checkpoint
        # ac_freq=2, count=2 → 2%2==0 → checkpoint
        # ac_freq=2, count=3 → 3%2==1 → 不 checkpoint
        return ptd_checkpoint_wrapper(
            module,
            preserve_rng_state=ac_config.preserve_rng_state,
            determinism_check=ac_config.determinism_check,
            early_stop=ac_config.early_stop,
            debug=ac_config.debug,
        )
    else:
        return module  # 不包装，原样返回
```

**关键细节**：
- `_layer_sac_count` 是全局变量，在 `apply_ac()` 入口被重置为 0（第 291 行）
- `ac_freq=2` 时第 2、4、6... 层被 checkpoint（从 1 开始计数）
- 不传入 `context_fn`，所以是全量重算（等同 full AC 但只对部分层）

## 3.5 `_apply_op_sac()`（113-194 行）— 算子级选择性 AC ★★★

这是最核心、最复杂的函数。分两部分讲解：

### 3.5.1 收集强制重算的矩阵乘法形状（131-152 行）

```python
mm_recompute_shapes = set()
if len(ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns) > 0:
    for module_fqn, submod in module.named_modules():
        # 构建完整 FQN
        fqn = module_fqn
        if base_fqn is not None:
            fqn = f"{base_fqn}.{module_fqn}"

        # 检查 FQN 是否匹配配置中的过滤器
        if not any(
            filter_fqn in fqn
            for filter_fqn in ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns
        ):
            continue

        # 匹配到的必须是 nn.Linear
        if not isinstance(submod, nn.Linear):
            raise ValueError(...)

        # 提取权重形状 (out_features, in_features) → 保存 (in_features, out_features)
        out_f, in_f = submod.weight.shape
        mm_recompute_shapes.add((in_f, out_f))
        # nn.Linear.weight 的 shape 是 (out, in)
        # 但 mm 操作时 input @ weight.T，所以 rhs 形状是 (in, out)
```

**为什么默认是 `["moe.router.gate"]`？**
- MoE router 的 gate 是一个 `nn.Linear(hidden_dim, num_experts)`
- 它的输出维度很小（num_experts 通常是 64），重算代价极低
- 但它的激活值会占用显存（batch × seq_len × num_experts）
- 强制重算 gate 的 mm 能节省显存且几乎不增加计算

### 3.5.2 自定义策略函数（154-181 行）

```python
def _get_custom_policy(meta):
    def _custom_policy(ctx, func, *args, **kwargs):
        # 规则 1：CUDA→CPU 拷贝始终保留
        if (
            func == torch.ops.aten._to_copy.default
            and "cuda" in str(args[0].device)
            and "device" in kwargs
            and str(kwargs["device"]) == "cpu"
        ):
            return CheckpointPolicy.MUST_SAVE

        # 跟踪 mm 计数（forward 和 recompute 分开计数）
        mode = "recompute" if ctx.is_recompute else "forward"
        mm_count_key = f"{mode}_mm_count"

        if func == torch.ops.aten.mm.default:
            # 规则 2：匹配强制重算形状的 mm 始终重算
            if args[1].shape in mm_recompute_shapes:
                return CheckpointPolicy.PREFER_RECOMPUTE
            meta[mm_count_key] += 1

        # 规则 3：保留列表中的 op，但每第 2 次 mm 重算
        to_save = func in op_sac_save_list and not (
            func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
        )
        return (
            CheckpointPolicy.MUST_SAVE
            if to_save
            else CheckpointPolicy.PREFER_RECOMPUTE
        )
    return _custom_policy
```

**策略决策树**：

```text
                    ┌── CUDA→CPU 拷贝？──→ MUST_SAVE
                    │
func 被调用 ────────┤
                    │            ┌── 形状在 mm_recompute_shapes 中？──→ PREFER_RECOMPUTE
                    ├── mm？─────┤
                    │            └── 第偶数次 mm？──→ PREFER_RECOMPUTE
                    │                 第奇数次 mm？──→ 检查 op_sac_save_list
                    │
                    └── 其他 op ──→ 在 save_list 中？──→ MUST_SAVE
                                    不在 save_list 中？──→ PREFER_RECOMPUTE
```

**为什么交替保留 mm？**
Transformer FFN 中典型计算是 SwiGLU：
```text
output = silu(x @ w1.T) * (x @ w3.T)    ← 2 次 mm
output = output @ w2.T                   ← 1 次 mm
```
三次 mm 中保留第 1、3 次（奇数），重算第 2 次（偶数），是接近最优的权衡。

### 3.5.3 组装 context_fn 并包装（183-194 行）

```python
def selective_checkpointing_context_fn():
    meta = defaultdict(int)  # 每次进入 forward 时重新创建 meta
    return create_selective_checkpoint_contexts(_get_custom_policy(meta))

return ptd_checkpoint_wrapper(
    module,
    context_fn=selective_checkpointing_context_fn,
    preserve_rng_state=ac_config.preserve_rng_state,
    determinism_check=ac_config.determinism_check,
    early_stop=ac_config.early_stop,
    debug=ac_config.debug,
)
```

**关键**：`meta = defaultdict(int)` 在 `selective_checkpointing_context_fn()` 内部创建，确保每次进入 checkpoint 区域时 mm 计数器从 0 开始。`meta` 通过闭包传入 `_get_custom_policy`，而 `_custom_policy` 再通过闭包访问 `meta`。

## 3.6 `_apply_full_ac()`（197-216 行）

```python
def _apply_full_ac(module, ac_config):
    return ptd_checkpoint_wrapper(
        module,
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )
```

最简单的模式：不传 `context_fn`，所有中间激活都丢弃并重算。

## 3.7 `_apply_ac_to_transformer_block()`（219-244 行）— 路由分发

```python
def _apply_ac_to_transformer_block(
    module: nn.Module,
    ac_config: ActivationCheckpointConfig,
    *,
    base_fqn: str | None = None,
    model_compile_enabled: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
) -> nn.Module:
    # 1. 验证模式（只接受 full 和 selective）
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(...)

    # 2. 分发到具体实现
    if ac_config.mode == "full":
        return _apply_full_ac(module, ac_config)

    # 3. selective 模式二分路
    if ac_config.selective_ac_option == "op":
        op_sac_save_list = op_sac_save_list or set()
        return _apply_op_sac(
            module, ac_config, base_fqn=base_fqn, op_sac_save_list=op_sac_save_list
        )

    return _apply_layer_sac(module, ac_config)
```

**注意**：此函数不处理 `none` 和 `memory_budget` 模式——它们在 `apply_ac()` 中提前处理。

## 3.8 `apply_ac()`（247-305 行）— 主入口 ★★★

```python
def apply_ac(
    model: nn.Module,
    ac_config: ActivationCheckpointConfig,
    *,
    model_compile_enabled: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
    base_folder: str = "",
) -> None:
```

### 3.8.1 `memory_budget` 模式的处理（268-279 行）

```python
if ac_config.mode == "memory_budget":
    assert model_compile_enabled, "Memory budget mode requires model to be compiled"

    if ac_config.visualize_memory_budget_pareto:
        pareto_dir = os.path.join(base_folder, "memory_budget_pareto")
        if not os.path.exists(pareto_dir):
            os.makedirs(pareto_dir, exist_ok=True)
        torch._functorch.config.memory_budget_pareto_dir = pareto_dir
        torch._functorch.config.visualize_memory_budget_pareto = True

    torch._functorch.config.activation_memory_budget = ac_config.memory_budget
    _get_logger().info(f"Selected {ac_config.memory_budget} budget option")
    return
```

**特殊性**：
- 不包装任何 module，而是设置 `torch._functorch.config` 全局配置
- 由 torch.compile 的编译器自动决定保留/重算策略
- 硬性要求 `model_compile_enabled=True`
- 可选输出帕累托最优曲线 SVG

### 3.8.2 `none` 模式的处理（281-283 行）

```python
if ac_config.mode == "none":
    _get_logger().debug("Activation checkpointing is disabled")
    return
```

### 3.8.3 遍历 model.layers 应用 AC（285-305 行）

```python
# 要求 model 有 layers 属性
if not hasattr(model, "layers"):
    raise ValueError(
        "Model must have a 'layers' attribute (ModuleDict) to apply AC"
    )

# 重置全局层计数器
global _layer_sac_count
_layer_sac_count = 0

# 遍历每个 TransformerBlock
for layer_id, transformer_block in model.layers.named_children():
    transformer_block = _apply_ac_to_transformer_block(
        transformer_block,
        ac_config,
        base_fqn=f"layers.{layer_id}",       # 传入 FQN 供 op_sac 匹配
        model_compile_enabled=model_compile_enabled,
        op_sac_save_list=op_sac_save_list,
    )
    model.layers.register_module(layer_id, transformer_block)
    # ↑ 替换原始 module 为包装后的 CheckpointWrapper

_get_logger().info(
    f"Applied {ac_config.mode} activation checkpointing to the model"
)
```

**调用链总结**：

```text
apply_ac()
  ├── mode == "memory_budget" → 设置 functorch config → return
  ├── mode == "none" → return
  └── mode == "full" / "selective"
       └── for layer in model.layers:
            └── _apply_ac_to_transformer_block()
                 ├── mode == "full" → _apply_full_ac()
                 └── mode == "selective"
                      ├── option == "op" → _apply_op_sac()
                      └── option.isdigit() → _apply_layer_sac()
```

---

# 4. 验证方法

## 4.1 理解检验题

### 题 1：模式判断
> 给定以下配置，apply_ac 会执行什么操作？
> ```python
> config = ActivationCheckpointConfig(mode="selective", selective_ac_option="3")
> apply_ac(model, config)  # model 有 6 层
> ```

**答案**：层级选择性 AC，每 3 层 checkpoint 一次。第 3 层和第 6 层被包装为 CheckpointWrapper，其余层不变。（计数器从 1 开始：count%3==0 时 checkpoint）

### 题 2：策略函数
> 在 op 级 selective AC 中，如果连续遇到 3 个 `aten.mm` 操作，且都在 `op_sac_save_list` 中，哪些会被保留、哪些会被重算？

**答案**：
- 第 1 个 mm：count=1, 1%2=1（奇数）→ `MUST_SAVE`
- 第 2 个 mm：count=2, 2%2=0（偶数）→ `PREFER_RECOMPUTE`
- 第 3 个 mm：count=3, 3%2=1（奇数）→ `MUST_SAVE`

### 题 3：强制重算
> `per_op_sac_force_recompute_mm_shapes_by_fqns=["moe.router.gate"]` 如何工作？

**答案**：
1. 遍历 module 的所有子模块，找到 FQN 包含 `"moe.router.gate"` 的 `nn.Linear`
2. 提取其权重形状 `(out_features, in_features)`，存为 `(in_features, out_features)`
3. 在策略函数中，若 mm 的 rhs 形状匹配，直接返回 `PREFER_RECOMPUTE`
4. 这些 mm 不参与奇偶计数

### 题 4：顺序依赖
> 如果把 AC 移到 FSDP 之后会怎样？

**答案**：反向传播时 CheckpointWrapper 重新执行 forward，会触发 FSDP 的 pre-forward hook 执行冗余 all-gather（参数已经在上一步 reshard 了）。导致通信量翻倍，训练变慢甚至 OOM。

## 4.2 运行测试

```bash
# 运行单元测试（不需要 GPU）
uv run pytest tests/experimental/archon/test_activation_checkpoint.py -v

# 测试覆盖：
# - TestActivationCheckpointConfig：配置验证（11 个测试）
# - TestApplyAc：apply_ac 各模式（9 个测试）
#   包括 none/full/selective(层级)/selective(op)/forward/backward 正确性
```

## 4.3 交互式验证

### 验证 1：观察 CheckpointWrapper 包装效果

```python
import torch.nn as nn
from areal.experimental.models.archon.activation_checkpoint import (
    ActivationCheckpointConfig, apply_ac,
)

class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
    def forward(self, x):
        return self.linear(x)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({str(i): DummyBlock() for i in range(4)})
    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

model = DummyModel()
print("包装前:", type(model.layers["0"]))
# → <class '__main__.DummyBlock'>

config = ActivationCheckpointConfig(mode="full")
apply_ac(model, config)
print("包装后:", type(model.layers["0"]))
# → <class 'torch.distributed...CheckpointWrapper'>

# 查看 state_dict key 的变化：
for k in model.state_dict().keys():
    print(k)
# layers.0._checkpoint_wrapped_module.linear.weight
# layers.0._checkpoint_wrapped_module.linear.bias
# ...
```

### 验证 2：观察层级选择性 AC 的选择模式

```python
config = ActivationCheckpointConfig(mode="selective", selective_ac_option="2")
model = DummyModel()  # 4 层
apply_ac(model, config)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
for name, layer in model.layers.named_children():
    wrapped = isinstance(layer, CheckpointWrapper)
    print(f"Layer {name}: {'✓ Wrapped' if wrapped else '✗ Not wrapped'}")
# Layer 0: ✗ Not wrapped
# Layer 1: ✓ Wrapped
# Layer 2: ✗ Not wrapped
# Layer 3: ✓ Wrapped
```

### 验证 3：forward/backward 一致性

```python
model = DummyModel()
x = torch.randn(2, 8)

# 不包装的输出
with torch.no_grad():
    ref = model(x.clone())

# 包装后的输出
config = ActivationCheckpointConfig(mode="full")
apply_ac(model, config)
with torch.no_grad():
    wrapped = model(x.clone())

print("输出一致:", torch.allclose(ref, wrapped))
# → True
```

## 4.4 常见误区

| 误区 | 正解 |
|-----|------|
| "AC 改变了模型的输出" | AC 不改变前向结果，只改变反向传播的中间激活值获取方式 |
| "Full AC 最省显存" | Memory Budget 模式可能更优（编译器帕累托优化） |
| "selective_ac_option='0' 等同 'none'" | `'0'` 意味着 `int('0')=0`，`not 0 = True`，实际等同 full AC |
| "AC 和 torch.compile 不兼容" | 兼容，但 AC 必须在 compile 之前应用 |
| "preserve_rng_state 默认开启" | 默认 `False`，开启会降低性能 |

---

# 5. 附录

## 5.1 完整数据流图

```text
┌─────────────────────────────────────────────────────────────────┐
│                      正常 Forward 传播                          │
│                                                                 │
│  Input → [Layer 0] → act₀ → [Layer 1] → act₁ → ... → Output   │
│            保存 act₀    保存 act₁    保存 actₙ                   │
│                                                                 │
│  Backward: 直接使用保存的 act₀, act₁, ... 计算梯度              │
│  显存: O(L) 个激活值                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Full AC 的 Forward                        │
│                                                                 │
│  Input → [CW Layer 0] → [CW Layer 1] → ... → Output            │
│          只保存 Input₀   只保存 Input₁                           │
│          丢弃中间激活     丢弃中间激活                             │
│                                                                 │
│  Backward: 用保存的 Input 重新执行 forward 得到中间激活           │
│  显存: O(1) per layer（仅输入边界）                               │
│  计算: ~1.33x（多一次前向）                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Selective AC (op) 的 Forward                 │
│                                                                 │
│  [CW Layer 0] 内部：                                            │
│    op₁(mm) → SAVE       ← 奇数次 mm，保留                      │
│    op₂(silu) → RECOMPUTE ← 不在 save_list 中                   │
│    op₃(mm) → RECOMPUTE  ← 偶数次 mm，重算                      │
│    op₄(mm) → SAVE       ← 奇数次 mm，保留                      │
│    op₅(attention) → SAVE ← 在 save_list 中                     │
│                                                                 │
│  Backward: 仅重算 op₂ 和 op₃，其余直接使用保存的值                │
│  显存: ~50-70%                                                   │
│  计算: ~105-115%                                                 │
└─────────────────────────────────────────────────────────────────┘

CW = CheckpointWrapper
```

## 5.2 文件依赖关系

```text
activation_checkpoint.py
  导入:
  ├── torch.distributed.algorithms._checkpoint.checkpoint_wrapper
  │     → ptd_checkpoint_wrapper (包装 module)
  ├── torch.utils.checkpoint
  │     → CheckpointPolicy (MUST_SAVE / PREFER_RECOMPUTE)
  │     → create_selective_checkpoint_contexts (创建选择性策略上下文)
  ├── torch._functorch.config
  │     → activation_memory_budget (memory_budget 模式)
  │     → memory_budget_pareto_dir (可视化)
  │     → visualize_memory_budget_pareto (可视化)
  └── areal.utils.logging → getLogger

  被导入:
  ├── archon_engine.py → ActivationCheckpointConfig (类型)
  ├── archon_utils.py → ActivationCheckpointConfig (build_ac_config)
  ├── model_spec.py → ActivationCheckpointConfig (ParallelizeFn 协议)
  ├── qwen2/infra/parallelize.py → apply_ac
  ├── qwen3/infra/parallelize.py → apply_ac
  └── tests/test_activation_checkpoint.py → 两者都导入
```

## 5.3 四种模式完整对比表

| 维度 | none | full | selective (op) | selective (N) | memory_budget |
|------|------|------|---------------|--------------|---------------|
| 显存节省 | 0% | ~50% | ~30-50% | 取决于 N | 自动 |
| 计算增加 | 0% | ~33% | ~5-15% | 取决于 N | 自动 |
| 包装方式 | 不包装 | ptd_checkpoint_wrapper | ptd_checkpoint_wrapper + context_fn | ptd_checkpoint_wrapper (部分层) | 不包装，设置 config |
| 需要 compile | 否 | 否 | 否 | 否 | **是** |
| 需要 save_list | 否 | 否 | **是** | 否 | 否 |
| zero-bubble 兼容 | 是 | 是 | **否 (降级)** | 是 | **否 (降级)** |
| 生产推荐度 | 小模型 | 极端场景 | **首选** | 粗粒度折中 | 实验性 |

## 5.4 op_sac_save_list 详解

| Op | 类别 | 保留原因 |
|----|------|---------|
| `aten.mm` | 矩阵乘法 | 主要计算量来源，选择性保留（奇数次） |
| `_scaled_dot_product_*_attention` (5 种) | 注意力 | 重算代价极高（O(n²)复杂度） |
| `reduce_scatter_tensor` | FSDP 通信 | 重算意味着重做通信 |
| `all_to_all_single` | MoE EP 通信 | 重算意味着重做 All-to-All |
| `aten.max` | 量化 | 用于计算 FP8 缩放因子，重算可能导致不一致 |
| `flex_attention` | 注意力 | Flex Attention 高阶 op |
| `areal._varlen_attn` | 自定义 | AReaL 注册的变长注意力 op |

## 5.5 验证清单

- [ ] 理解 4 种 AC 模式的区别和适用场景
- [ ] 理解算子级选择性 AC 的策略函数工作原理
- [ ] 理解 mm 奇偶交替保留/重算的优化逻辑
- [ ] 理解 `per_op_sac_force_recompute_mm_shapes_by_fqns` 的形状匹配机制
- [ ] 理解 AC 在并行化流水线中的位置约束（Step 4）及原因
- [ ] 理解 CheckpointWrapper 对 state_dict key 的影响
- [ ] 理解 memory_budget 模式与 torch.compile 的关系
- [ ] 理解 zero-bubble schedule 的兼容性限制
- [ ] 能运行测试并理解每个测试用例的意图
- [ ] 能手动验证 forward/backward 一致性
