# AReaL 源码走读：FP8 Blockwise 训练实现解析

> 在大模型 RL 训练场景中，显存始终是横亘在 scaling 路上的核心瓶颈。BF16 训练下，参数、梯度和优化器状态三者叠加，动辄消耗数十 GB 显存。FP8（8-bit
> 浮点）训练是近年来最具实用价值的"显存 + 吞吐"双赢手段之一——权重和激活值从 16-bit 降至 8-bit，GEMM 算力直接翻倍。但 FP8 并非简单地"把
> dtype 换一下"：per-tensor 量化的精度损失太大，blockwise 量化带来了新的 scale 管理和对齐约束；而当 FP8 遇上 FSDP2
> 分片、Tensor Parallel 切分、Pipeline Parallel stage 拆分时，整个 checkpoint、通信和初始化链路都需要重新设计。
>
> AReaL 在两个训练引擎（Megatron 和 Archon）中分别实现了 FP8 Blockwise 训练支持，走的是两条截然不同的技术路线：Megatron 依赖
> Transformer Engine (TE) 的原生 FP8 能力，Archon 则通过 monkey patch + torchao prototype 实现零侵入式的
> forward 替换。本文不展开 FP8 数值格式或 Blockwise Quantization 的理论原理，而是聚焦源码，分析 AReaL 如何把 FP8
> Blockwise 训练接入现有训练链路，以及它带来了哪些收益和限制。

______________________________________________________________________

# 前言

## 业务 / 工程背景

FP8 训练出现在以下场景交汇处：

- **显存瓶颈**：大模型 RL 训练（如 GRPO）需要同时运行 actor、critic、reference 模型，显存极为紧张
- **吞吐需求**：Hopper (H100/H800) GPU 的 FP8 Tensor Core 提供 2× BF16 的算力上限
- **精度约束**：per-tensor FP8 量化（整个权重矩阵共享一个 scale）对 outlier 权重损失过大，需要更细粒度的 blockwise（128×128
  分块独立量化）方案

## 核心矛盾

AReaL 的 FP8 Blockwise 训练面临三重工程矛盾：

1. **两套引擎、两条路线**：Megatron 引擎依赖 Transformer Engine (TE) 的闭源 FP8 实现，Archon 引擎依赖 torchao 的
   prototype API——两者的量化时机、scale 存储格式、checkpoint 格式和通信方式完全不同
1. **分布式分片 vs 块对齐**：FSDP2 / TP / PP 会将权重按不同维度切分到不同 rank，但 FP8 128×128 blockwise kernel
   要求权重维度必须是 128 的整数倍——分片后的 local shape 可能破坏这个约束
1. **权重格式在训练与推理之间的转换**：训练侧（TE 的 `Float8BlockwiseQTensor` 或 torchao 的 on-the-fly
   量化）与推理侧（HuggingFace safetensors FP8 格式）的 scale 存储方式不同，每次 weight sync 都需要格式转换

## 本文主线

本文分以下几章分析 AReaL 中 FP8 Blockwise 训练的实现：

- **一、配置体系**：两套 FP8 配置如何定义、校验和传递
- **二、Archon 引擎的 Monkey Patch 方案**：forward 替换、shard 对齐、compile 互斥
- **三、Megatron 引擎的 TE 集成方案**：配置映射、FP8 参数、优化器交互
- **四、FP8 Blockwise 量化核心**：Triton kernel、UE8M0 格式、scale 管理
- **五、Checkpoint 加载与保存**：FP8 权重的检测、反量化、直通转换
- **六、通信与并行交互**：TP all-gather、FSDP shard/unshard、序列对齐
- **七、完整主路径串联**：一次真实训练调用的端到端执行流
- **八、显存、性能与通信分析**
- **九、测试覆盖与缺口**
- **十、局限性与已知优化点**

## 不展开的内容

- 不讲 FP8 e4m3fn / e5m2 数值格式的理论（参阅 IEEE FP8 spec）
- 不讲 Blockwise Quantization 的数学推导（参阅 DeepSeek-V3 论文）
- 不讲 FSDP2 / Megatron 的基本原理（参阅本系列其他走读文章）
- 不讲 Transformer Engine 内部实现（TE 是闭源 + 部分开源的外部库）

## 核心文件表

| 文件                                                      | 职责                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------- |
| `areal/api/cli_args.py`                                   | `ArchonFP8Config` (L466) 和 `FP8EngineConfig` (L731) 配置定义 |
| `areal/experimental/models/archon/fp8.py`                 | Archon 引擎：nn.Linear forward monkey patch、shard 对齐校验   |
| `areal/experimental/models/archon/fp8_checkpoint.py`      | Archon 引擎：FP8 checkpoint 检测、加载、反量化                |
| `areal/experimental/engine/archon_engine.py`              | Archon 引擎初始化：FP8 patch 注入和后校验                     |
| `areal/engine/megatron_engine.py`                         | Megatron 引擎：FP8 配置映射、high_precision_init_val 修复     |
| `areal/engine/megatron_utils/fp8/kernels.py`              | Triton kernel：blockwise 量化和反量化                         |
| `areal/engine/megatron_utils/fp8/tensor_helper.py`        | `FP8BlockwiseTensorHelper`：FP8 数据 + scale 联动操作         |
| `areal/engine/megatron_utils/fp8/quantize.py`             | 高层量化/反量化 API（HF 权重转换）                            |
| `areal/engine/megatron_utils/megatron.py`                 | Megatron TP all-gather FP8 tensor 支持                        |
| `areal/experimental/models/archon/moe/grouped_experts.py` | MoE expert FP8 for-loop fallback                              |

______________________________________________________________________

# 一、配置体系：两套 FP8 配置的设计差异

## 1.1 设计哲学与核心问题

AReaL 支持两个训练引擎——Megatron 和 Archon——它们的 FP8 支持走的是完全不同的技术路线。配置体系必须反映这个差异：Megatron 需要将十余个
FP8 参数传递给 Transformer Engine 的 `TransformerConfig`，而 Archon 只需要一个 mode 开关加少量模块排除规则。

如果没有分开的配置体系，用户要么需要在一个巨大的配置类中区分"这个字段归 Megatron 用还是归 Archon 用"，要么需要面对大量无效字段的困惑。

## 1.2 源码入口与关键对象

```
areal/api/cli_args.py
  - ArchonFP8Config (L466)：Archon 引擎 FP8 配置，4 个字段
  - FP8EngineConfig  (L731)：Megatron 引擎 FP8 配置，13 个字段
  - ArchonEngineConfig.fp8_config (L654)：始终实例化，通过 .enabled 判断
  - MegatronEngineConfig.fp8_config (L915)：Optional，None 表示禁用
```

## 1.3 主流程拆解

### Archon FP8 配置

```python
# areal/api/cli_args.py:466-535
@dataclass
class ArchonFP8Config:
    mode: str = "disabled"           # "disabled" | "blockwise"
    exclude_modules: list[str] = ["output", "router", "score"]
    include_experts: bool = False
    use_triton: bool = True          # 当前必须为 True

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"
```

配置极其简洁：只有一种 FP8 模式（blockwise 128×128），通过 `mode` 字段开关。`exclude_modules` 控制哪些 `nn.Linear`
保留 BF16（默认排除 LM head、MoE router、critic score head 等精度敏感模块）。

`__post_init__` 中有一个关键校验（L523-535）：`use_triton` 在 FP8 启用时**必须为 True**。原因是 torchao 的
blockwise FP8 使用了 **mixed per-operand scaling**（激活值 1×128，权重 128×128），PyTorch 的
`torch._scaled_mm`（cuBLAS 路径）不支持这种混合 scale 模式，只能走 Triton kernel
`triton_fp8_gemm_1x128_128x128`。

### Megatron FP8 配置

```python
# areal/api/cli_args.py:731-833
@dataclass
class FP8EngineConfig:
    mode: str = "e4m3"         # "e4m3" | "hybrid"
    recipe: str = "delayed"    # "tensorwise" | "delayed" | "mxfp8" | "blockwise"
    param: bool = False        # 是否将参数保持为 FP8 格式
    margin: int = 0            # scaling factor 安全裕量
    amax_history_len: int = 1  # delayed scaling 的 amax 历史窗口
    amax_compute_algo: str = "most_recent"  # amax 选择算法
    wgrad: bool = True         # 权重梯度是否用 FP8 计算
    direct_convert: bool = True # 直通 FP8 ↔ PyTorch FP8 转换
    # ... 还有 dot_product_attention, multi_head_attention, tp_only_amax_red,
    #     first_last_layers_bf16, num_layers_at_start_in_bf16, num_layers_at_end_in_bf16
```

Megatron 配置丰富得多，因为 TE 本身支持多种 FP8 recipe（tensorwise / delayed / blockwise / mxfp8）。对于
blockwise 训练，关键组合是 `recipe="blockwise"` + `param=True`（权重也存为 FP8）+
`direct_convert=True`（save/load 直通，不走反量化-再量化中间步骤）。

### 用户如何启用

**Archon 引擎**（YAML 示例 `gsm8k_sft_archon_fp8.yaml`）：

```yaml
actor:
  archon:
    fp8_config:
      mode: blockwise
```

**Megatron 引擎**（YAML 示例 `gsm8k_grpo_megatron_fp8.yaml`）：

```yaml
actor:
  megatron:
    fp8_config:
      mode: hybrid
      recipe: blockwise
      param: true
    ddp:
      fp8_param_gather: true
```

## 1.4 关键细节与误区澄清

**误区一：`FP8EngineConfig` 与 `ArchonFP8Config` 的 `mode` 字段含义完全不同。**

- `ArchonFP8Config.mode` 的取值是 `"disabled"` / `"blockwise"`——它控制 FP8 的开/关
- `FP8EngineConfig.mode` 的取值是 `"e4m3"` / `"hybrid"`——它控制前向/反向使用的 FP8 数值格式，而不是开关（开关通过
  `MegatronEngineConfig.fp8_config` 是否为 `None` 判断）

同名字段、不同语义，容易造成混淆。

**误区二：Archon 的 `exclude_modules` 在 YAML 中设置时是替换而非追加。**

配置的 help 信息明确警告（L492）：*"WARNING: Setting this in YAML replaces the entire default list
(does not extend it)"*。如果你只想额外排除一个模块，必须把默认的三个也写上。

## 1.5 本章小结

> 💡 小结
>
> - AReaL 为两个引擎分别设计了 FP8 配置类，Archon 简洁（4 字段）、Megatron 丰富（13 字段）
> - Archon FP8 当前只支持 blockwise 模式 + Triton kernel，cuBLAS 路径因 mixed scaling 限制暂不可用
> - Megatron FP8 通过 `recipe` 字段选择量化策略，`blockwise` 是其中之一
> - 两套配置的 `mode` 字段含义不同，是初学者的常见误区

______________________________________________________________________

# 二、Archon 引擎的 Monkey Patch 方案：零侵入还是维护风险？

## 2.1 设计哲学与核心问题

Archon 引擎选择了一条非常"轻量"的 FP8 集成路线：**不替换模块类，只替换 forward 方法**。标准的 `nn.Linear` 保持不变，权重仍以 BF16
存储和 FSDP 分片，FP8 量化仅在每次 forward 时 on-the-fly 发生。

这个设计的核心考量是：

- FSDP2 对 `nn.Linear` 有原生的 shard/unshard 支持。如果引入新的 `FP8Linear` 类，FSDP2 的 wrap 策略和
  DTensor placement 都需要适配
- Pipeline Parallel 的 stage 拆分使用 `copy.deepcopy`，monkey patch 通过 `types.MethodType`
  绑定可以安全地跟着 deepcopy 走
- 权重保持 BF16 意味着优化器不需要特殊处理——Adam 直接在 BF16 master weight 上更新

如果没有这一层，Archon 引擎要么需要引入 TE 依赖（与 torchao 路线冲突），要么需要定义新的 FP8 Module 子类并处理与 FSDP2 的兼容。

## 2.2 源码入口与关键对象

```
areal/experimental/models/archon/fp8.py
  - enable_fp8_linear()     (L18)：遍历模型，patch 符合条件的 nn.Linear
  - enable_fp8_experts()    (L57)：patch MoE GroupedExperts 模块
  - _is_eligible()          (L119)：判断模块是否可被 FP8 patch
  - _patch_fp8_forward()    (L134)：核心 patch 函数，替换 forward
  - _patch_fp8_experts_forward() (L91)：expert 模块的 patch
  - validate_fp8_shard_alignment() (L166)：并行后校验 shard 对齐

areal/experimental/engine/archon_engine.py
  - initialize() (L316-368)：FP8 patch 注入点和后校验

areal/experimental/engine/archon_utils.py
  - prepare_training_config() (L335)：FP8 与 torch.compile 互斥处理
```

## 2.3 主流程拆解

### 时序：FP8 patch 在初始化链路中的位置

```
archon_engine.initialize()
  │
  ├─ Step 1: _create_device_model()       # 在 meta device 上构建模型
  │
  ├─ Step 2: FP8 Patch 注入 (L316-338)    # ← 必须在 meta device 上
  │     ├─ dtype 校验：必须是 bfloat16
  │     ├─ enable_fp8_linear(model, exclude_fqns, use_triton)
  │     └─ enable_fp8_experts(model, use_triton)  # 仅当 include_experts=True
  │
  ├─ Step 3: prepare_training_config()     # torch.compile 被 FP8 禁用
  │
  ├─ Step 4: _setup_parallelism()          # 应用 TP / PP / FSDP2
  │
  ├─ Step 5: FP8 Shard 对齐校验 (L362-368) # ← 必须在并行后
  │     └─ validate_fp8_shard_alignment(model_parts)
  │
  ├─ Step 6: _materialize_and_load_weights()
  └─ Step 7: _create_optimizer()           # 标准 Adam，无 FP8 特殊处理
```

关键约束：**patch 必须发生在 meta device 上（Step 2），在并行之前**。这是因为 `_patch_fp8_forward` 通过
`types.MethodType` 绑定到模块实例上，而 PP 的 stage split 会 deepcopy 模块——`MethodType` 绑定的方法可以正确地跟着
deepcopy。如果在并行之后 patch，PP 的各个 stage 已经是独立的模块拷贝，需要逐个 patch。

### `_patch_fp8_forward` 的核心逻辑

```python
# areal/experimental/models/archon/fp8.py:146-161
def _fp8_linear_fwd(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    leading = x.shape[:-1]           # 保存前导维度
    x = x.reshape(-1, x.shape[-1])   # 展平为 2D: [tokens, hidden]
    m = x.shape[0]
    pad = (self._fp8_block - m % self._fp8_block) % self._fp8_block
    if pad > 0:
        x = F.pad(x, (0, 0, 0, pad)) # M 维度 pad 到 128 的整数倍
    weight = self.weight
    if hasattr(weight, "to_local"):
        weight = weight.to_local()    # DTensor → local tensor（TP 场景）
    out = self._fp8_mm.apply(         # torchao fp8_blockwise_mm
        x, weight, self._fp8_block, x.dtype, self._fp8_use_triton
    )
    if pad > 0:
        out = out[:m]                 # 裁掉 padding
    return out.view(*leading, -1)
```

这段代码完成了四件事：

1. **维度展平**：将任意形状的输入（`[batch, seq, hidden]` 或 `[tokens, hidden]`）统一为 2D
1. **M 维度 padding**：token 维度不一定是 128 的整数倍，需要 pad（权重维度的对齐在 `validate_fp8_shard_alignment`
   中提前保证）
1. **DTensor 解包**：TP 场景下权重是 DTensor，`to_local()` 取出 local shard
1. **FP8 matmul**：调用
   `torchao.prototype.blockwise_fp8_training.linear.fp8_blockwise_mm`——这是一个
   `torch.autograd.Function`，内部同时处理 forward 量化和 backward 反传

### MoE Expert 的 FP8 路径

MoE 的 FP8 实现走的是 **per-expert for-loop fallback**（`grouped_experts.py:114-181`），因为
`torch._grouped_mm` 尚不支持 FP8：

```python
# areal/experimental/models/archon/moe/grouped_experts.py:160-166
# SwiGLU: silu(x @ w1.T) * (x @ w3.T) @ w2.T
h1 = fp8_blockwise_mm.apply(x_e, w1_e, block_size, x_e.dtype, use_triton)
h3 = fp8_blockwise_mm.apply(x_e, w3_e, block_size, x_e.dtype, use_triton)
h = F.silu(h1) * h3
h2 = fp8_blockwise_mm.apply(h, w2_e, block_size, h.dtype, use_triton)
```

每个 expert 的 token 独立 pad 到 128 对齐后，执行三次 FP8 matmul（SwiGLU 模式）。注意 L135 有一个 **GPU→CPU
sync**（`num_tokens_per_expert.tolist()`），这是 per-expert for-loop 的固有开销，BF16 路径使用
`grouped_mm` 可以避免。

### torch.compile 互斥

`archon_utils.py:335-340` 明确处理了 FP8 与 `torch.compile` 的不兼容：

```python
if config.archon.fp8_config.enabled and enable_compile:
    logger.warning("FP8 blockwise training is incompatible with torch.compile. "
                   "Disabling torch.compile.")
    enable_compile = False
```

这是因为 torchao 的 prototype FP8 kernel 尚未被 `torch.compile` 的 dynamo/inductor 追踪器支持。

## 2.4 关键细节与误区澄清

**误区三：`_fp8_linear_fwd` 中的 `weight.to_local()` 不是 FSDP unshard。**

`to_local()` 是 DTensor 的方法，用于从 DTensor 包装中取出 local tensor。FSDP2 的 unshard（从分片状态恢复完整权重到
local device）在 forward 被调用时**已经发生**——这是 FSDP2 `fully_shard` 的标准行为。`to_local()` 只是去掉
DTensor 的 metadata wrapper，不触发通信。在非 TP 场景下，`weight` 不是
DTensor，`hasattr(weight, "to_local")` 返回 False，直接使用原始 weight。

**误区四：Archon 的 FP8 权重并不是真正的 FP8 存储。**

与 Megatron 的 `fp8_param=True` 不同，Archon 的权重始终以 BF16 存储和分片。FP8 量化只发生在
`fp8_blockwise_mm.apply()` 内部的 forward/backward 过程中（on-the-fly）。这意味着：

- 优化器状态和 master weight 都是 BF16/FP32，没有量化误差累积
- 显存节省仅来自 GEMM 计算过程中的临时激活值压缩，参数本身不节省
- FSDP2 的 shard/unshard 对 BF16 权重透明操作，不需要 FP8 感知

## 2.5 本章小结

> 💡 小结
>
> - Archon 通过 `types.MethodType` 替换 forward，不引入新类，对 FSDP2/PP 透明
> - Patch 必须在 meta device 上、并行之前执行；shard 对齐校验必须在并行之后
> - 权重保持 BF16，FP8 是纯计算优化（on-the-fly），不节省参数显存
> - MoE expert 走 per-expert for-loop（有 GPU→CPU sync 开销），等待 `grouped_mm` FP8 支持
> - FP8 与 torch.compile 互斥

______________________________________________________________________

# 三、Megatron 引擎的 TE 集成方案：配置映射与优化器陷阱

## 3.1 设计哲学与核心问题

Megatron 引擎走的是完全不同的路线：依赖 NVIDIA 的 Transformer Engine (TE) 提供 FP8 能力。TE 的
`Float8BlockwiseQTensor` 是一等公民——权重可以真正存储为 FP8，TE 的 `fp8_autocast` context manager
管理前向/反向的量化行为。AReaL 的职责是把用户配置**正确映射**到 TE 的 `TransformerConfig`，以及处理 HF 预训练权重加载时的 FP8
兼容问题。

如果没有这一层映射，用户需要直接面对 Megatron-Core + TE 的十几个分散的 FP8 配置字段，且无法享受 AReaL 的 HF 模型直接加载能力。

## 3.2 源码入口与关键对象

```
areal/engine/megatron_engine.py
  - __init__() (L176-181)：FP8 配置读取和开关设置
  - _check_and_apply_fp8_config() (L1028-1060)：配置字段映射到 TE
  - _validate_fp8_consistency() (L1062-1082)：FP8 训练 vs 权重格式一致性校验
  - initialize() (L345-367)：high_precision_init_val 清理
  - _create_optimizer() (L1179)：fp8_recipe 传递给优化器
  - _prepare_micro_batches() (L1740-1744)：FP8 序列对齐
```

## 3.3 主流程拆解

### 配置映射

`_check_and_apply_fp8_config()` 将 `FP8EngineConfig` 的字段逐一映射到 Megatron-Core 的
`TransformerConfig`：

```python
# areal/engine/megatron_engine.py:1028-1060
special_mappings = {"mode": "fp8"}        # mode → tf_config.fp8
same_fields = {"tp_only_amax_red", "first_last_layers_bf16", ...}  # 同名直传
# 其余字段加 fp8_ 前缀：recipe → fp8_recipe, param → fp8_param, ...
```

映射策略清晰：特殊映射表 → 同名字段集 → 默认加 `fp8_` 前缀。如果某个字段在 `TransformerConfig` 中不存在，会打 warning。

### `high_precision_init_val` 陷阱修复

这是 Megatron FP8 集成中最容易踩的坑，值得详细展开（`megatron_engine.py:345-367`）：

```python
# 问题背景：
# 1. Megatron 的分布式优化器用 high_precision_init_val 初始化 main_params（FP32 master weight）
# 2. TE 在模型初始化时，用 init_method 的随机值设置了 high_precision_init_val
# 3. 但 AReaL 在模型初始化之后才从 HF checkpoint 加载真实权重
# 4. 结果：优化器的 master weight 用的是随机值，而不是加载的真实权重
#
# 解决方案：加载 HF 权重后，清除所有 FP8 参数的 high_precision_init_val
for model in self.model:
    for _, param in model.named_parameters():
        if hasattr(param, "get_high_precision_init_val"):
            param.clear_high_precision_init_val()
            delattr(param, "get_high_precision_init_val")
            delattr(param, "clear_high_precision_init_val")
```

如果这段代码不存在，FP8 训练的优化器 master weight 会被初始化为**随机值**而不是预训练权重——训练结果完全错误。

### 权重格式一致性校验

`_validate_fp8_consistency()` 强制要求：如果训练使用 FP8，模型权重也必须是 FP8
格式（`quantization_config.quant_method == "fp8"`）。这意味着 Megatron FP8 训练**要求使用预量化的 FP8
模型**（如 `Qwen3-1.7B-FP8`），不支持从 BF16 模型直接开始 FP8 训练。

这与 Archon 引擎形成鲜明对比——Archon 可以使用 BF16 模型（如 `Qwen3-1.7B`），FP8 是 on-the-fly 计算。

### FP8 序列对齐

```python
# areal/engine/megatron_engine.py:1740-1744
align_to_multiple_of = (
    math.lcm(align_to_multiple_of, DEFAULT_VECTORIZED_ALIGNMENT_BYTES)  # 16
    if self.enable_fp8
    else align_to_multiple_of
)
```

FP8 kernel 的向量化操作要求序列长度额外对齐到 16 字节边界。这通过 `math.lcm` 取现有对齐值和 16 的最小公倍数实现。

## 3.4 关键细节与误区澄清

**误区五：`FP8EngineConfig.param=True` 不等于"所有参数都是 FP8"。**

配置的 help 信息明确说（L758-760）：*"Not all parameters will be converted to fp8; for example,
biases will remain unchanged."* 此外，`first_last_layers_bf16=True` 可以让首尾 N 层保留
BF16。真正决定哪些参数用 FP8 的是 TE 的内部逻辑，AReaL 只传递配置。

**误区六：Megatron FP8 的 forward/backward 量化逻辑不在 AReaL 源码中。**

AReaL 的 `areal/engine/megatron_utils/fp8/kernels.py` 中的 Triton kernel 只用于 **checkpoint
转换**（quantize/dequantize HF 权重）。训练过程中的 FP8 forward/backward 完全由 TE 的 `fp8_autocast`
context manager 和 `Float8BlockwiseQTensor` 内部管理。AReaL 不实现训练用的 FP8 autograd.Function。

## 3.5 本章小结

> 💡 小结
>
> - Megatron FP8 通过字段映射将 AReaL 配置转为 TE TransformerConfig，不自行实现 FP8 计算
> - `high_precision_init_val` 清理是避免优化器 master weight 为随机值的关键修复
> - Megatron FP8 要求预量化模型（`quant_method=fp8`），Archon 不需要
> - AReaL 的 Triton kernel 不参与训练计算，只用于 checkpoint 格式转换

______________________________________________________________________

# 四、FP8 Blockwise 量化核心：Triton Kernel 与 Scale 管理

## 4.1 设计哲学与核心问题

Blockwise 量化的核心思想是：将一个大矩阵分成 128×128 的小块，每个块独立计算 max 绝对值并导出 scale。与 per-tensor
量化（整个矩阵共享一个 scale）相比，blockwise 可以更精确地捕捉每个局部区域的数值范围，显著降低量化误差。

但这带来了新的工程问题：

- Scale 不再是一个标量，而是一个 `[ceil(M/128), ceil(N/128)]` 的 2D 张量
- 对权重做 view / chunk / split / cat 时，scale 必须做对应的变换
- TP all-gather 需要同时 gather 数据和 scale
- 不同硬件（Hopper vs Blackwell）对 scale 格式有不同要求

## 4.2 源码入口与关键对象

```
areal/engine/megatron_utils/fp8/kernels.py
  - _blockwise_cast_to_fp8_triton (L14)：Triton JIT 量化 kernel
  - blockwise_cast_to_fp8_triton  (L56)：Python wrapper
  - weight_dequant_kernel (L116)：Triton JIT 反量化 kernel
  - weight_dequant (L130)：Python wrapper

areal/engine/megatron_utils/fp8/tensor_helper.py
  - FP8BlockwiseTensorHelper (L11)：数据 + scale 联动的 tensor wrapper

areal/engine/megatron_utils/fp8/ue8m0.py
  - ceil_to_ue8m0 (L33)：scale 取最近的 2 的幂
  - per_block_cast_to_fp8_ue8m0 (L100)：Blackwell UE8M0 量化
  - quant_weight_ue8m0 (L130)：高层接口
  - transform_scale_ue8m0 (L169)：转为 TMA 对齐的 packed 格式
```

## 4.3 主流程拆解

### 量化 Kernel：从 BF16 到 FP8+Scale

Triton kernel `_blockwise_cast_to_fp8_triton`（`kernels.py:14-53`）的逻辑非常直白：

```
对每个 128×128 block (pid_m, pid_n):
  1. 加载 block 数据到寄存器（带边界 mask）
  2. 计算 block 内最大绝对值：_absmax = max(abs(block))
  3. 计算 scale：x_s = _absmax / 448.0  (448.0 = fp8_e4m3fn 的最大可表示值)
  4. 量化：y_q = clamp(block / x_s, -448.0, 448.0)  → cast to float8_e4m3fn
  5. 存储量化数据和 scale
```

wrapper 函数创建输出 tensor（`kernels.py:75-78`）：

```python
y = torch.empty(M, N, device=x.device, dtype=fp8_dtype)        # 量化后数据
s = torch.empty(ceil_div(M, 128), ceil_div(N, 128), dtype=float32)  # scale
```

一个有趣的优化细节：对连续内存 tensor 使用 8 warps / 2 stages，非连续 tensor 使用 1 warp / 4 stages（L83-96）。

### 反量化 Kernel

`weight_dequant_kernel`（`kernels.py:116-127`）更简单：

```
对每个 128×128 block:
  y = fp8_data.to(float32) * scale[block_row, block_col]
```

来源标注为 Alibaba Pai-Megatron-Patch，这是 DeepSeek-V3 开源生态中广泛使用的反量化方案。

### FP8BlockwiseTensorHelper：数据与 Scale 的联动

这是整个 FP8 工具链中最精巧的组件。`FP8BlockwiseTensorHelper`（`tensor_helper.py:11-414`）是
`torch.Tensor` 的子类，同时持有 `_rowwise_data`（FP8）和
`_rowwise_scale_inv`（FP32），并在所有张量操作中自动维护两者的对应关系。

为什么需要这个？因为 Megatron 的 HF 权重转换涉及大量张量操作：

- **QKV 分离**：`[hidden, 3*head_dim*num_heads]` → chunk 成 Q、K、V
- **GLU 合并**：gate 和 up 投影 cat 成 `fc1`
- **Expert 拆分**：3D expert weight 的 split
- **TP 维度变换**：view 成 `[num_groups, heads_per_group, head_dim, hidden]`

每个操作都必须同步作用于数据和 scale。`FP8BlockwiseTensorHelper` 的 `_compute_scale_shape()`
方法（L56-95）负责推断 view/reshape 后 scale 的正确形状：

```python
# 规则：最后两个维度需要 ÷ block_size，前面的维度原样传递
# 例如：data [8, 512, 4096] → scale [8, 4, 32]  (512/128=4, 4096/128=32)
```

`__torch_dispatch__`（L380-414）拦截 `torch.cat`、`torch.split`、`torch.chunk` 等 aten 操作，确保对
Helper 实例的操作也能正确联动。

### UE8M0：Blackwell 专属 Scale 格式

在 Blackwell (SM100+) GPU 上，DeepGEMM 要求 scale 为 **UE8M0 格式**（Unsigned Exponent, 8-bit,
Mantissa=0）——本质上是把 scale 限制为 2 的幂。

```python
# areal/engine/megatron_utils/fp8/ue8m0.py:33-45
def ceil_to_ue8m0(x):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))
```

精度上的代价很小（scale 最多放大 2×），但换来硬件友好的存储格式：FP32 scale 的指数位直接提取为 uint8，4 个 uint8 pack 成 int32，按
TMA 对齐的 MN-major 布局存放（`get_mn_major_tma_aligned_packed_ue8m0_tensor`）。

决策链路（`deepgemm.py:82-101`）：

```
should_deepgemm_weight_requant_ue8m0()
  → 检查 DeepGEMM 是否可用（SM90+ + deep_gemm 包）
  → 检查是否为 Blackwell GPU
  → 检查 weight_block_size 是否非空
```

`_compute_enable_deep_gemm()` 和 `_is_deepgemm_blackwell()` 使用 `@functools.cache`
缓存结果，全进程生命周期只计算一次。

## 4.4 关键细节与误区澄清

**误区七：AReaL 的 Triton kernel 在两个引擎中的角色完全不同。**

- **在 Archon 引擎**中：训练期间的 FP8 matmul 由 **torchao 的 `fp8_blockwise_mm`** 执行，AReaL 的 Triton
  kernel（`kernels.py`）只用于**加载 FP8 checkpoint 时的反量化**
- **在 Megatron 引擎**中：训练的 FP8 计算由 **TE 内部管理**，AReaL 的 Triton kernel 用于 **HF 权重的 save/load
  转换**（BF16 ↔ FP8）

换句话说，`kernels.py` 中的量化/反量化 kernel 从不出现在训练 step 的热路径上。

## 4.5 本章小结

> 💡 小结
>
> - Blockwise 量化为每个 128×128 block 独立维护 scale，精度优于 per-tensor
> - `FP8BlockwiseTensorHelper` 是数据 + scale 联动操作的核心抽象
> - AReaL 的 Triton kernel 仅用于 checkpoint 转换，不在训练热路径上
> - Blackwell GPU 额外引入 UE8M0 scale 格式，以适配 DeepGEMM 的硬件要求

______________________________________________________________________

# 五、Checkpoint 加载与保存：FP8 权重的检测与转换

## 5.1 设计哲学与核心问题

FP8 checkpoint 处理面临的核心矛盾是：HuggingFace 的 FP8 模型使用 `weight` + `weight_scale_inv` 的键值对存储（如
DeepSeek-V3、Qwen3-FP8），但两个引擎的内部表示完全不同——Megatron 用 TE 的 `Float8BlockwiseQTensor`，Archon
用标准 `nn.Linear` 的 BF16 权重。加载和保存时都需要格式转换。

## 5.2 源码入口与关键对象

```
Archon 路径：
  areal/experimental/models/archon/fp8_checkpoint.py
    - _detect_fp8_checkpoint()   (L34)：检测是否为 FP8 checkpoint
    - _prepare_fp8_state_dict()  (L47)：DCP 加载前的 dtype 占位符修改
    - weight_dequant_cpu()       (L119)：CPU 上的纯 PyTorch 反量化
    - _dequant_dtensor()         (L152)：FSDP sharded FP8 DTensor 反量化
    - dequant_fp8_state_dict()   (L229)：主入口，反量化所有 FP8 权重

Megatron 路径：
  areal/models/mcore/hf_load.py
    - 检测 _scale_inv 后缀键 → 直接加载 FP8 或先反量化再加载
  areal/models/mcore/hf_save.py
    - TE FP8 → FP8BlockwiseTensorHelper → PyTorch FP8 safetensors
  areal/engine/megatron_utils/megatron.py
    - all_gather_param()：TP 场景下 FP8 tensor all-gather
```

## 5.3 主流程拆解

### Archon 加载 FP8 Checkpoint 的完整流水线

```
detect → prepare → DCP load → dequant → from_hf → Archon model

具体步骤：
  1. _detect_fp8_checkpoint(path)
       → 读 model.safetensors.index.json，找 *_scale_inv 键
       → 有则判定为 FP8 checkpoint

  2. _prepare_fp8_state_dict(hf_state_dict, path)
       → 将 weight 占位符 dtype 从 BF16 改为 float8_e4m3fn
       → 插入 float32 scale_inv 占位符（shape = ceil(M/128) × ceil(N/128))
       → 返回修改后的 state_dict（DCP 只加载 dict 中已有的键）

  3. dcp.load(state_dict)
       → 分布式加载：每个 rank 只加载自己负责的 shard

  4. dequant_fp8_state_dict(hf_state_dict)
       → 遍历所有 dtype=float8_e4m3fn 的权重
       → 找到对应的 *_scale_inv 键
       → 如果是 DTensor：调用 _dequant_dtensor()（处理 shard 边界对齐）
       → 如果在 GPU：使用 Triton weight_dequant kernel
       → 如果在 CPU：使用 weight_dequant_cpu()（纯 PyTorch fallback）
       → 删除所有 *_scale_inv 键
       → state_dict 回到纯 BF16

  5. state_dict_adapter.from_hf() → 标准的 BF16 加载路径
```

关键设计：Archon **在加载时将 FP8 权重反量化为 BF16**，因为 Archon 的 FP8 是 on-the-fly 计算——权重以 BF16
存储，forward 时再量化。

### `_dequant_dtensor` 的 shard 边界问题

这是 checkpoint 加载中最精妙的部分（`fp8_checkpoint.py:152-226`）。FSDP2 按 `Shard(0)` 切分权重后，每个 rank
拿到一段行切片。但 shard 边界不一定对齐到 128 block 边界——scale 的行索引需要特殊处理：

```python
# 计算 local shard 在全局矩阵中的起始行
start_row = global_offset[0]
local_M = local_fp8.shape[0]

# scale 的行范围：可能跨越 block 边界
block_start = start_row // block_size             # 向下取整
block_end = (start_row + local_M + block_size - 1) // block_size  # 向上取整
local_scale = scale_inv[block_start:block_end, :]

# 如果 shard 起始不在 block 边界上，需要 pad
offset = start_row % block_size
if offset > 0:
    local_fp8 = F.pad(local_fp8, (0, 0, offset, 0))  # 顶部补零
```

反量化后再裁掉 padding：`local_bf16 = local_bf16[offset:]`。

当前限制：只支持 `Shard(0)` placement。`Shard(1)`（column-sharded）需要按列维度切 scale，尚未实现，代码中有 TODO
标注（L170-173）。

### Megatron 的 FP8 Weight Sync（训练 → 推理）

Megatron 引擎在训练后需要将 FP8 权重同步给推理引擎。核心路径在 `megatron.py:63-152`：

```
all_gather_param(name, param, fp8_direct_convert, quantization_config)
  │
  ├─ 检测 param 是否为 TE Float8Tensor
  │
  ├─ 如果 fp8_direct_convert=True 且 param 是 FP8：
  │     └─ _all_gather_fp8_tensor_and_concat()
  │           ├─ 分别 all-gather _rowwise_data 和 _rowwise_scale_inv
  │           └─ 包装为 FP8BlockwiseTensorHelper
  │
  └─ 否则：
        └─ param.data → all-gather BF16（TE 自动反量化）
```

`_all_gather_fp8_tensor_and_concat`（L63-91）的关键在于：**数据和 scale 沿同一个 partition_dim 分别
all-gather**，然后在接收端重组为 `FP8BlockwiseTensorHelper`。这保证了 FP8 直通路径下通信量约为 BF16 的一半（uint8 数据
\+ float32 scale）。

## 5.4 关键细节与误区澄清

**误区八：Archon 加载 FP8 checkpoint 后，权重仍是 FP8 格式——这是错的。**

Archon 在 `dequant_fp8_state_dict()` 中将所有 FP8 权重转回 BF16，并删除 `*_scale_inv` 键。加载完成后，模型完全是
BF16 的。FP8 只在训练 forward 时 on-the-fly 发生。

**误区九：`weight_dequant_cpu` 看起来只是 Triton kernel 的 CPU fallback，但它在 DCP cpu_offload
场景下是主路径。**

DCP（Distributed Checkpoint）可以配置 `cpu_offload`，此时 state_dict 中的 tensor 在 CPU
上。`dequant_fp8_state_dict` 会自动检测 tensor 设备并选择 CPU 或 GPU 反量化路径。这不是"备用路径"，而是生产配置。

## 5.5 本章小结

> 💡 小结
>
> - Archon 加载 FP8 checkpoint 的流程是 detect → prepare → DCP load → dequant → 纯 BF16
> - FSDP shard 边界不对齐 128 block 时，需要 pad + 裁剪处理
> - Megatron 的 FP8 直通路径分别 all-gather 数据和 scale，通信量约为 BF16 的一半
> - `Shard(1)` FP8 反量化尚未实现，是当前的功能缺口

______________________________________________________________________

# 六、通信与并行交互：FP8 如何穿越分布式边界

## 6.1 设计哲学与核心问题

FP8 Blockwise 训练引入了一个新的张量结构：数据是 `float8_e4m3fn`（uint8 存储），scale 是 `float32` 2D
张量。当分布式并行（TP、PP、FSDP、DP）需要通信这些张量时，必须同时处理数据和 scale，且保证 block 边界对齐。

## 6.2 各并行维度的 FP8 交互

### Tensor Parallelism (TP)

TP 沿权重的行或列维度切分参数。在 Megatron 引擎中（`megatron.py:63-91`），FP8 tensor 的 all-gather 分两步：

```
TP group: [rank0, rank1, rank2, rank3]
                              partition_dim=0 (row parallel)

rank0: data_shard=[M/4, K], scale_shard=[ceil(M/4/128), ceil(K/128)]
rank1: data_shard=[M/4, K], scale_shard=[ceil(M/4/128), ceil(K/128)]
...

all-gather data:  concat([data_0, data_1, data_2, data_3], dim=0)  → [M, K]
all-gather scale: concat([scale_0, scale_1, scale_2, scale_3], dim=0) → [ceil(M/128), ceil(K/128)]
```

数据和 scale 沿**相同维度** all-gather——因为 block_size 在该维度上的 scale 数量与数据的行/列数成正比。

在 Archon 引擎中，TP 的处理更隐式：`_fp8_linear_fwd` 中的 `weight.to_local()` 取出 DTensor 的 local
shard（已经是 TP-sharded 的 BF16 权重），FP8 量化在 local shard 上 on-the-fly 执行。FSDP2 的 all-gather 对
BF16 权重透明。

### Pipeline Parallelism (PP)

PP 将模型按层切分到不同 stage。FP8 与 PP 的交互主要是：

- **Archon**：monkey patch 通过 `types.MethodType` 绑定，PP 的 `copy.deepcopy` stage split
  可以正确保留（L138-141 的注释明确说明了这一点）
- **Megatron**：TE 的 FP8 modules 本身支持 PP，AReaL 不需要额外处理

### FSDP2 (Full Sharded Data Parallelism)

FSDP2 的 shard/unshard 是 Archon FP8 的关键交互点：

```
训练前：权重以 BF16 DTensor 形式 FSDP-sharded
forward 时：
  1. FSDP2 自动 all-gather 权重到 local device（仍为 BF16）
  2. _fp8_linear_fwd 读取 unsharded BF16 weight
  3. fp8_blockwise_mm 内部 on-the-fly 量化为 FP8 执行 GEMM
  4. 输出 BF16 gradient
backward 后：FSDP2 reduce-scatter 梯度（BF16）
```

FSDP2 全程操作 BF16 tensor，**完全不感知 FP8 的存在**。这是 Archon "零侵入" 设计的直接结果。

### Shard 对齐校验

`validate_fp8_shard_alignment()`（`fp8.py:166-231`）在并行设置完成后检查所有 FP8-patched 模块的 local
权重维度是否仍为 128 的整数倍。检查对象包括：

- **nn.Linear**：2D 权重 `[out_dim, in_dim]`，两个维度都必须对齐
- **GroupedExperts**：3D 权重 `[num_experts, dim_a, dim_b]`，per-expert slice
  `(dim_a, dim_b)` 必须对齐

如果 TP degree 不合适（例如 hidden_size=4096 被 TP=3 切分，得到 1365），校验会报 `ValueError` 并建议调整 TP
degree 或将该模块加入 `exclude_modules`。

## 6.3 梯度的 FP8 行为

**关键结论：梯度不以 FP8 通信。**

- **Archon**：权重是 BF16，`fp8_blockwise_mm` 的 autograd Function 输出 BF16 梯度。FSDP2 的
  reduce-scatter 操作 BF16 梯度
- **Megatron**：TE 内部可能在 backward 时使用 FP8 计算 weight gradient（由 `wgrad` 字段控制），但梯度
  reduction（`finalize_model_grads`）操作的是训练精度（BF16/FP32）的梯度 buffer

`fp8_param_gather`（`cli_args.py:727`）控制 Megatron DDP 是否以 FP8 格式 gather
参数——这影响参数通信量，但不影响梯度通信。

## 6.4 本章小结

> 💡 小结
>
> - TP all-gather FP8 需要分别 gather 数据和 scale，沿相同维度
> - FSDP2 对 Archon FP8 完全透明——它只看到 BF16 权重和梯度
> - 梯度在两个引擎中都以 BF16 通信，FP8 仅影响计算精度
> - Shard 对齐校验是安全网，在并行后运行，防止运行时 kernel crash

______________________________________________________________________

# 七、完整主路径串联

## 7.1 Archon 引擎 FP8 Blockwise 训练的完整调用栈

```
用户配置: archon.fp8_config.mode = "blockwise"
  │
  ├─ Step 1: 配置加载与校验
  │     ├─ ArchonFP8Config.__post_init__()：mode ∈ {disabled, blockwise}, use_triton=True
  │     └─ ArchonEngineConfig 将 fp8_config 作为子字段持有
  │
  ├─ Step 2: 模型构建（meta device）
  │     └─ _create_device_model()：标准 BF16 nn.Module
  │
  ├─ Step 3: FP8 Forward Patch 注入
  │     ├─ dtype 校验：必须 bfloat16
  │     ├─ enable_fp8_linear(model, exclude_fqns={"output","router","score"})
  │     │     └─ 遍历 nn.Linear → _is_eligible() → _patch_fp8_forward()
  │     │           └─ mod.forward = types.MethodType(_fp8_linear_fwd, mod)
  │     └─ [可选] enable_fp8_experts(model)
  │           └─ 遍历 GroupedExperts → _patch_fp8_experts_forward()
  │
  ├─ Step 4: 训练配置准备
  │     └─ prepare_training_config()：FP8 → 禁用 torch.compile
  │
  ├─ Step 5: 并行设置（TP / PP / FSDP2）
  │     └─ _setup_parallelism()：FSDP2 fully_shard, TP DTensor, PP stage split
  │
  ├─ Step 6: Shard 对齐校验
  │     └─ validate_fp8_shard_alignment(model_parts)
  │           └─ 检查每个 FP8 模块的 local weight 维度 % 128 == 0
  │
  ├─ Step 7: 权重加载
  │     └─ _materialize_and_load_weights()
  │           ├─ [FP8 checkpoint] detect → prepare → DCP load → dequant → from_hf
  │           └─ [BF16 checkpoint] 标准 DCP 加载
  │
  ├─ Step 8: 优化器创建
  │     └─ 标准 Adam/AdamW，无 FP8 特殊处理
  │
  ├─ Step 9: 训练循环（每个 step）
  │     └─ forward:
  │           └─ _fp8_linear_fwd(x):
  │                 ├─ reshape → pad(M→128对齐) → to_local()
  │                 ├─ fp8_blockwise_mm.apply(x, weight, 128, bf16, triton=True)
  │                 │     ├─ forward: 量化 x(1×128) + 量化 w(128×128) → FP8 GEMM → BF16 output
  │                 │     └─ backward: 重新量化 saved_x/w → FP8 梯度计算 → BF16 grad
  │                 └─ unpad → reshape
  │
  └─ Step 10: Checkpoint 保存
        └─ 保存 BF16 state_dict（FP8 只在 forward 临时存在）
```

## 7.2 Megatron 引擎 FP8 Blockwise 训练的完整调用栈

```
用户配置: megatron.fp8_config.recipe = "blockwise", param = true
  │
  ├─ Step 1: 配置映射
  │     └─ _check_and_apply_fp8_config()
  │           └─ FP8EngineConfig 字段 → TransformerConfig (mode→fp8, recipe→fp8_recipe, ...)
  │
  ├─ Step 2: 权重格式校验
  │     └─ _validate_fp8_consistency()：quantization_config.quant_method == "fp8"
  │
  ├─ Step 3: HF FP8 权重加载
  │     └─ _load_model_from_hf()
  │           ├─ 检测 *_scale_inv 键
  │           ├─ [fp8_direct_convert=True]
  │           │     └─ FP8BlockwiseTensorHelper → to_te_fp8_inplace()
  │           └─ [fp8_direct_convert=False]
  │                 └─ dequantize_params() → BF16 → 加载
  │
  ├─ Step 4: high_precision_init_val 清理（关键！）
  │     └─ 遍历所有 FP8 param → clear_high_precision_init_val()
  │
  ├─ Step 5: 优化器创建
  │     └─ fp8_recipe 传递给 MCoreOptimizerConfig
  │
  ├─ Step 6: 序列对齐
  │     └─ align_to_multiple_of = lcm(tp*cp*2, 16)  # FP8 额外 16 字节对齐
  │
  ├─ Step 7: 训练循环
  │     └─ TE fp8_autocast context manager 管理 forward/backward
  │
  └─ Step 8: Weight Sync（训练→推理）
        └─ _collect_param() → all_gather_param()
              ├─ [fp8_direct_convert] 分别 gather data + scale
              └─ convert_to_hf() → FP8BlockwiseTensorHelper → PyTorch FP8 safetensors
```

## 7.3 哪些逻辑不在主路径

| 看似相关的函数/文件             | 容易误解的原因       | 实际是否在主流程               | 正确理解                         |
| ------------------------------- | -------------------- | ------------------------------ | -------------------------------- |
| `kernels.py` 的 Triton kernel   | 文件名暗示是核心计算 | **不在训练热路径**             | 只用于 checkpoint 转换           |
| `quantize_params()`             | 名字暗示训练时量化   | **不在训练 forward**           | 用于 HF save 时将 BF16 转 FP8    |
| `per_block_cast_to_fp8_ue8m0()` | Blackwell 量化       | **大多数环境不触发**           | 仅 Blackwell GPU + DeepGEMM 组合 |
| `weight_dequant_cpu()`          | 看似 fallback        | **DCP cpu_offload 时是主路径** | 不仅仅是 fallback                |

______________________________________________________________________

# 八、显存、性能与通信分析

## 8.1 显存收益范围

### Archon 引擎

| 内容        | 是否节省   | 原因                                                         |
| ----------- | ---------- | ------------------------------------------------------------ |
| 参数        | ❌         | 权重以 BF16 存储，FSDP shard/unshard 也是 BF16               |
| 激活值      | ✅（部分） | `fp8_blockwise_mm` 内部临时激活值为 FP8，但输入输出仍为 BF16 |
| Logits      | ❌         | LM head 被 `exclude_modules` 排除，始终 BF16                 |
| 优化器状态  | ❌         | Adam 在 BF16/FP32 master weight 上操作                       |
| 输入 batch  | ❌         | 输入 embedding 是 BF16                                       |
| 中间 buffer | ✅（临时） | GEMM 计算过程中权重和激活的 FP8 副本在临时 buffer 中         |

**结论**：Archon FP8 的显存节省主要来自 **GEMM 计算过程中的临时激活值压缩**，不节省参数和优化器状态。主要收益是 **计算吞吐**（FP8 Tensor
Core 2× 算力）。

### Megatron 引擎

| 内容       | 是否节省           | 原因                                                                         |
| ---------- | ------------------ | ---------------------------------------------------------------------------- |
| 参数       | ✅（`param=True`） | 权重以 FP8 存储，约为 BF16 的一半                                            |
| 激活值     | ✅                 | TE fp8_autocast 管理激活值的 FP8 量化                                        |
| Logits     | ❌ / ✅            | 取决于 `first_last_layers_bf16` 配置                                         |
| 优化器状态 | ❌                 | FP32 master weight 仍需维护（可通过 precision_aware_optimizer 压缩 exp_avg） |
| FP8 scale  | 额外开销           | 每个参数额外存储 `[ceil(M/128), ceil(N/128)]` 的 FP32 scale                  |
| 输入 batch | ❌                 | 输入 embedding 不受 FP8 影响                                                 |

## 8.2 通信开销

### Archon 引擎

FP8 **不增加任何通信**。FSDP2 的 all-gather 和 reduce-scatter 操作 BF16 tensor，FP8 仅在 local 计算中出现。

### Megatron 引擎

| 通信操作              | 发生场景            | FP8 影响                                                                                      |
| --------------------- | ------------------- | --------------------------------------------------------------------------------------------- |
| TP all-gather         | 每个 forward 的每层 | `fp8_param_gather=True` 时以 uint8+scale gather，通信量 ≈ BF16 的 60%（数据 50% + scale 10%） |
| AMAX reduce           | FP8 delayed scaling | `tp_only_amax_red=True` 可限制在 TP 域内 reduce                                               |
| DDP reduce-scatter    | 梯度同步            | BF16 梯度，不受 FP8 影响                                                                      |
| Weight sync broadcast | 训练→推理           | `fp8_direct_convert=True` 时以 FP8 格式 broadcast，节省约 50% 带宽                            |

## 8.3 性能取舍

| 收益                                          | 代价                                                         |
| --------------------------------------------- | ------------------------------------------------------------ |
| FP8 Tensor Core 2× 计算吞吐                   | Triton kernel 比 cuBLAS BF16 优化程度低（torchao prototype） |
| 参数通信量降低（Megatron `fp8_param_gather`） | 额外的 scale 通信和计算                                      |
| 权重显存减半（Megatron `fp8_param=True`）     | FP32 scale 张量的额外显存                                    |
| 与现有 FSDP2/PP 透明兼容（Archon）            | 无法使用 `torch.compile`（Archon）                           |
| on-the-fly 量化无需预量化模型（Archon）       | 每次 forward 都有量化开销                                    |
| 预量化模型直接加载（Megatron）                | 要求 FP8 格式的 HF checkpoint                                |
| MoE expert FP8 支持（Archon）                 | per-expert for-loop 有 GPU→CPU sync 开销                     |

______________________________________________________________________

# 九、测试覆盖与缺口

## 9.1 已覆盖路径

### Archon 引擎测试（`tests/experimental/archon/fp8/`）

| 测试                             | 覆盖的行为                                                          |
| -------------------------------- | ------------------------------------------------------------------- |
| `test_fp8_linear.py`             | Forward 正确性（cos>0.9）、Backward 正确性（cos>0.5）、多 step 收敛 |
| `test_conversion.py`             | `enable_fp8_linear` patch 逻辑、默认排除、维度不对齐跳过、bias 跳过 |
| `test_dequant.py`                | CPU 反量化正确性、GPU 反量化、state_dict 批量反量化                 |
| `test_checkpoint_detect.py`      | FP8 checkpoint 检测（有/无 scale_inv 键）                           |
| `test_checkpoint_prepare.py`     | DCP 加载前 dtype 占位符修改                                         |
| `test_checkpoint_integration.py` | 完整 detect→prepare→load→dequant 流水线                             |
| `test_checkpoint_e2e.py`         | 使用真实模型（Qwen3-1.7B-FP8）的端到端测试                          |
| `test_dequant_distributed.py`    | 分布式 sharded FP8 反量化（2/4 GPU）                                |
| `test_moe_dispatch.py`           | MoE FP8 per-expert 计算正确性、空 expert、不对齐 token 数           |
| `test_fp8_scale_layout.py`       | torchao scale layout 验证（column-major strides）                   |

### Megatron 引擎测试（`tests/fp8/`）

| 测试                          | 覆盖的行为                                                        |
| ----------------------------- | ----------------------------------------------------------------- |
| `test_fp8_tensor.py`          | `FP8BlockwiseTensorHelper` 的 chunk/view/split/cat/to_pytorch_fp8 |
| `test_fp8_conversion.py`      | TE Float8BlockwiseQTensor GEMM 正确性（需要 TE 安装）             |
| `test_fp8_bf16_comparison.py` | FP8 vs BF16 的 logits（cos>0.99）、梯度（cos>0.94）、文本生成     |

## 9.2 未覆盖风险

| 风险点                         | 当前是否有测试         | 可能后果                                      |
| ------------------------------ | ---------------------- | --------------------------------------------- |
| `Shard(1)` FP8 反量化          | ❌ 未实现              | TP/ETP column-sharded FP8 checkpoint 无法加载 |
| FP8 + torch.compile 同时启用   | ✅ 有互斥逻辑          | 但没有测试在禁用 compile 后的性能回归         |
| 多机分布式 FP8 训练            | ❌ 仅 2/4 GPU 测试     | 多机 NCCL 通信 + FP8 可能有对齐问题           |
| FP8 训练后的 checkpoint resume | ❌ 没有 resume 测试    | FP8→保存→加载→继续训练可能有精度漂移          |
| FP8 + 特殊模型（VL、MLA）      | ❌ 测试仅使用 Qwen3    | MLA attention 的 FP8 兼容性未验证             |
| 性能 benchmark                 | ❌ 没有吞吐/显存测试   | 无法量化 FP8 的实际收益                       |
| UE8M0 Blackwell 路径           | ❌ 没有 Blackwell 测试 | UE8M0 scale 转换的正确性未验证                |
| FP8 + LoRA                     | ❌ 没有组合测试        | FP8 patch + LoRA adapter 可能冲突             |
| `test_profile_gemm_kernels`    | ⏭ 永久 skip            | 仅用于调试，不在 CI 运行                      |
| `test_rmsnorm_from_file`       | ⏭ 永久 skip            | 依赖本地保存的 activation 文件                |

______________________________________________________________________

# 十、局限性与已知优化点

## 10.1 硬约束

- **GPU 要求**：FP8 需要 SM90+（H100/H800 Hopper 或更新）。所有 FP8 测试在无 Hopper GPU 时自动 skip
- **权重维度对齐**：`in_features` 和 `out_features` 都必须是 128 的整数倍。不满足的模块自动跳过（有
  warning），但如果关键层被跳过，训练效果不确定
- **无 bias**：有 bias 的 `nn.Linear` 自动跳过 FP8 patch（`_is_eligible` L123-124）
- **Archon FP8 + torch.compile 互斥**：当前 torchao prototype 不被 dynamo 追踪器支持
- **Megatron FP8 要求预量化模型**：`quant_method` 必须为 `"fp8"`，不支持从 BF16 启动
- **Archon `Shard(1)` FP8 checkpoint 不支持**：column-sharded 的 FP8 DTensor 反量化未实现

## 10.2 维护成本

- **torchao prototype 依赖**：`torchao.prototype.blockwise_fp8_training.linear` 是实验性
  API，可能在 torchao 版本升级时被移动或重命名。`pyproject.toml` 固定了 `torchao==0.15.0`
- **Monkey patch 脆弱性**：如果 `nn.Linear` 的 signature 或 FSDP2 的 forward hook
  机制变化，`_fp8_linear_fwd` 可能失效
- **TE 版本耦合**：`FP8BlockwiseTensorHelper.from_te()` 和 `to_te_fp8_inplace()` 依赖 TE 的
  `_rowwise_data` / `_rowwise_scale_inv` 内部属性，不是公开 API
- **多路径 scale 格式**：FP32 scale（Hopper Triton）vs UE8M0 packed scale（Blackwell DeepGEMM）vs
  TE internal scale——三种格式在不同路径中转换，增加维护复杂度

## 10.3 性能瓶颈

- **MoE per-expert for-loop**：每次 forward 的 `num_tokens_per_expert.tolist()` 引入 GPU→CPU
  sync，是串行瓶颈。需要等 `torch._grouped_mm` 获得 FP8 支持
- **on-the-fly 量化开销（Archon）**：每次 forward 都重新量化权重，对于大模型每层额外的量化计算不可忽略
- **M 维度 padding**：token 数不是 128 倍数时需要 pad，可能浪费计算（最坏 padding 127 行）
- **Triton kernel vs cuBLAS**：torchao 的 Triton FP8 GEMM 可能不如 cuBLAS 的 BF16 GEMM 优化程度高，在小
  batch 场景下 FP8 可能不比 BF16 快

## 10.4 已知优化点

- **源码 TODO**：`_dequant_dtensor` L170-173 标注了 `Shard(1)` 支持为 "Phase 2"
- **`grouped_mm` FP8**：`_run_experts_fp8_for_loop` 注释（L123-130）明确说明是因为 `grouped_mm` 不支持
  FP8 而采用 for-loop，待上游支持后可去除
- **cuBLAS 路径**：`ArchonFP8Config.use_triton` 的 help 文本（L514）指出 "Revisit when torchao
  stabilizes mixed-mode cuBLAS dispatch"——待 `torch._scaled_mm` 支持 mixed scaling 后可切换
- **`torch.compile` 兼容**：当 torchao 的 FP8 算子被 torch inductor 原生支持后，可恢复 compile
- **权重预量化缓存（Archon）**：当前每次 forward 都 on-the-fly 量化权重。如果权重在连续 step 中不变（如 frozen
  layers），可以缓存 FP8 权重避免重复量化

______________________________________________________________________

# 小结与展望

AReaL 的 FP8 Blockwise 训练实现可以用几个关键词概括。

**关键词一：双路线并行**

Megatron 依赖 TE 的 FP8 生态（预量化模型、FP8 参数存储、TE autocast），Archon 依赖 torchao prototype 的
on-the-fly 量化（BF16 master weight、forward monkey patch）。两条路线各有取舍：Megatron 节省参数显存但要求 FP8
模型，Archon 对模型格式无要求但不节省参数显存。

**关键词二：零侵入 Monkey Patch**

Archon 引擎通过 `types.MethodType` 替换 `nn.Linear.forward`，不引入新类、不修改模型定义、不影响 FSDP2
分片。这带来了极好的兼容性，但也引入了 patch 脆弱性和 torch.compile 互斥。

**关键词三：数据-Scale 联动**

`FP8BlockwiseTensorHelper` 是整个实现中最精巧的组件。它将 FP8 数据和 blockwise scale 封装为一个 tensor 子类，使得
chunk / split / cat / view 等操作自动维护数据-scale 的对应关系，这对 Megatron 的 HF 权重转换（QKV 分离、GLU 合并、TP
all-gather）至关重要。

**关键词四：128 对齐约束贯穿全链路**

从配置校验（`__post_init__`）、patch
资格检查（`_is_eligible`）、并行后校验（`validate_fp8_shard_alignment`）到运行时
padding（`_fp8_linear_fwd`），128 对齐是整个 FP8 实现的"硬约束红线"，任何一个环节的遗漏都会导致 Triton kernel crash。

**关键词五：Checkpoint 三段式**

FP8 checkpoint 加载经历 detect → prepare → dequant 三个阶段，每个阶段解决一个问题：检测格式、修改占位符、反量化。Archon 将
FP8 权重反量化为 BF16 后使用，Megatron 可以 FP8 直通加载。

**适用场景**：H100+ GPU 上的大模型 RL 训练，目标是在精度可接受（cosine similarity >0.99）的前提下获得 GEMM
吞吐提升。Megatron 路线适合有预量化 FP8 模型的场景，Archon 路线适合从 BF16 模型零配置切换的场景。

**不适用场景**：Hopper 以下 GPU；需要 `torch.compile` 加速的 Archon 训练；对精度要求极高的场景（如 reward model
输出层）；需要 `Shard(1)` FP8 checkpoint 加载的 TP/ETP 配置。

**后续值得继续走读的方向**：

- torchao 的 `fp8_blockwise_mm` 内部实现（Triton GEMM kernel 的具体 tiling 策略、backward 的重量化策略）
- Transformer Engine 的 `fp8_autocast` 和 `Float8BlockwiseQTensor` 内部实现
- 当 `torch._grouped_mm` 获得 FP8 支持后，MoE expert 路径的重构
- Blackwell GPU 的 UE8M0 + DeepGEMM 路径的实际性能表现
