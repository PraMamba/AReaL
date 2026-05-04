# AReaL 源码走读：Qwen3.5 混合注意力模型（GatedDeltaNet + GatedAttention）实现解析

> 在 Transformer 走向更长上下文、更高效推理的演进中，**线性注意力**以 O(n) 的推理复杂度成为替代标准 softmax
> 注意力的一个重要方向。然而，纯线性注意力在表达能力上与标准注意力仍有差距，不能简单"一换了之"。Qwen3.5
> 的做法是——不全换：大部分层用线性注意力（GatedDeltaNet），关键位置保留全注意力（GatedAttention），形成**混合注意力架构**。本文聚焦
> AReaL 框架的 Archon 引擎，拆解这套混合注意力模型在训练侧的实现：配置怎么读、模型怎么建、权重怎么转、并行怎么做、forward 怎么跑、checkpoint
> 怎么存，以及当前有哪些限制。

______________________________________________________________________

## 前言

### 业务与工程背景

Qwen3.5 是一个"混合注意力"模型。它的每一层 Transformer Block 不再千篇一律使用标准 Multi-Head Attention，而是根据
`layer_types` 配置列表，逐层决定用哪种注意力：

- **`full_attention`** 层使用 `GatedAttention`——标准 GQA + Flash Attention + 输出门控
- **`linear_attention`** 层使用 `GatedDeltaNet`——基于 Gated Delta Rule 的线性注意力

典型配置是**每 4 层中 3 层用线性注意力、1 层用全注意力**。这样既获得线性注意力在推理时的效率优势，又在关键位置保留全注意力的表达能力。

### 核心矛盾

这种混合架构给训练框架带来了三个核心矛盾：

1. **两种注意力的接口不对称**：GatedAttention 需要 RoPE、Flash Attention、cu_seqlens；GatedDeltaNet 需要
   causal conv1d、chunk_gated_delta_rule kernel、seq_idx。同一个 TransformerBlock 要在 forward
   中根据 layer_type 切换这两套完全不同的计算路径。

1. **权重命名空间分裂**：HuggingFace 的 Qwen3.5 模型中，full_attention 层的权重前缀是
   `self_attn.*`，linear_attention 层的前缀是 `linear_attn.*`，且 GatedDeltaNet 有
   `A_log`、`dt_bias` 这类不带 `.weight` 后缀的"裸参数"。State dict adapter 必须正确处理这两套命名。

1. **并行策略残缺**：标准 Attention 的 Tensor Parallelism 通过切分 QKV head 实现，但 GatedDeltaNet 的 Q/K/V
   维度、head 数与标准注意力不同，TP/CP/EP 都尚未支持，只能走 FSDP。

### 本文主线

本文按以下脉络拆解实现：

1. **模型注册与配置归一化**——从 HuggingFace config 到 Archon 内部表示
1. **混合 TransformerBlock 的调度机制**——两种注意力如何共存于同一个 Block
1. **GatedDeltaNet 的计算链路**——线性注意力的 forward 全过程
1. **GatedAttention 的计算链路**——门控全注意力的设计
1. **权重转换的双通道适配器**——State Dict 在 HF 与 Archon 之间的双向转换
1. **并行化与编译**——当前 FSDP-only 的并行策略
1. **完整调用栈串联**——从 ArchonEngine 初始化到训练前向
1. **显存、性能与通信分析**
1. **测试覆盖与缺口**
1. **局限性与已知优化点**

### 不展开的内容

本文**不讲** Gated Delta Rule 的数学原理（请参考原论文），**不讲** FSDP2 的底层分片机制，**不讲** Flash Attention
的算法细节。只讲 AReaL/Archon 如何在工程上把这些东西接入统一的训练管线。

### 核心文件表

| 文件                                                                   | 职责                                                                    |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `areal/experimental/models/archon/qwen3_5/model/model.py`              | 核心模型：GatedDeltaNet、GatedAttention、TransformerBlock、Qwen3_5Model |
| `areal/experimental/models/archon/qwen3_5/model/args.py`               | 模型参数 dataclass，从 HF config 解析混合架构配置                       |
| `areal/experimental/models/archon/qwen3_5/model/state_dict_adapter.py` | HF ↔ Archon 双向权重转换，处理混合层命名和 MoE 权重                     |
| `areal/experimental/models/archon/qwen3_5/model/rope.py`               | Partial RoPE：只对 head_dim 的 25% 做旋转位置编码                       |
| `areal/experimental/models/archon/qwen3_5/spec.py`                     | ModelSpec 注册，绑定模型类、参数类、适配器类、并行化函数                |
| `areal/experimental/models/archon/qwen3_5/infra/parallelize.py`        | 并行化（当前仅 FSDP）+ torch.compile                                    |
| `areal/experimental/engine/archon_engine.py`                           | ArchonEngine：模型生命周期管理（创建、加载、训练、保存）                |
| `areal/experimental/engine/archon_checkpoint.py`                       | Checkpoint 加载/保存，通过 DCP + StateDictAdapter                       |

______________________________________________________________________

## 一、模型注册与配置归一化：从 HF Config 到 Archon 表示

### 1.1 设计哲学与核心问题

Archon 引擎的设计目标是**运行时不依赖 HuggingFace 模型类**。它只用 `AutoConfig.from_pretrained()`
读取配置，然后用自己的模型类构建计算图。这意味着，从 HF 的 `PretrainedConfig` 到 Archon 内部的
`Qwen3_5ModelArgs`，必须有一步**配置归一化**。

对于混合注意力模型，这步归一化的挑战在于：Qwen3.5 的 HF config 比普通 Transformer 多出了一整组 `linear_*` 前缀的参数（conv
kernel、key/value head 维度和数量等），以及关键的 `layer_types` 列表。

### 1.2 源码入口与关键对象

```
areal/experimental/models/archon/qwen3_5/spec.py
  - QWEN3_5_SPEC：ModelSpec 实例，注册支持的 model_type
  - register_model_spec()：导入时自动注册

areal/experimental/models/archon/qwen3_5/model/args.py
  - Qwen3_5ModelArgs：模型参数 dataclass
  - from_hf_config()：HF config → Archon args 的转换入口

areal/experimental/models/archon/model_spec.py
  - get_model_spec()：根据 model_type 查找已注册的 ModelSpec
```

### 1.3 主流程拆解

**注册链路**。当 `areal.experimental.models.archon` 包被导入时，`__init__.py`（第 32 行）会
`import areal.experimental.models.archon.qwen3_5.spec`，触发 `spec.py` 末尾的
`register_model_spec(QWEN3_5_SPEC)`。这样，`{"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}`
这四个 model_type 就被注册到全局 `_MODEL_SPECS` 字典中。

**配置转换链路**。在 ArchonEngine 的 `__init__` 中（`archon_engine.py:157`）：

```
AutoConfig.from_pretrained(path) → model_config
get_model_spec(model_config.model_type) → spec
```

然后在 `_create_device_model()` 中：

```
spec.model_args_class.from_hf_config(model_config) → model_args
spec.model_class(model_args) → model（在 meta device 上）
```

`from_hf_config()` 的核心逻辑（`args.py:98-182`）做了这几件事：

1. **VLM 复合配置降级**：如果 config 有 `text_config` 属性（视觉-语言模型），取 `text_config`
1. **partial_rotary_factor 提取**：可能在顶层，也可能嵌套在 `rope_parameters` 字典中，默认 0.25
1. **MoE 判断**：通过 `num_experts` 或 `num_local_experts` 字段判断是否为 MoE 模型
1. **layer_types
   截断**：`list(hf_config.layer_types)[:num_hidden_layers]`——这一步很关键，因为部分加载（只加载前 N
   层）时，`layer_types` 仍是完整列表，需要截断到实际层数

**参数校验**。`__post_init__`（`args.py:91-96`）做了一个关键检查：如果提供了 `layer_types`，其长度必须等于
`n_layers`。否则抛出 `ValueError`。

### 1.4 关键细节与误区澄清

**误区一：`head_dim` 是由 `hidden_size // num_attention_heads` 计算得到的。**

这在 Qwen3.5 中是**错误的**。Qwen3.5 的 `head_dim` 是**显式配置**的（默认 256），而不是从 hidden_size 推导。0.8B 的
Qwen3.5 有 `hidden_size=3072`、`num_attention_heads=24`，按传统公式应该是 `128`，但实际
`head_dim=256`。这意味着 Q 投影的输出维度是 `24 × 256 × 2 = 12288`（乘以 2 是因为 GatedAttention 的 Q 投影同时输出
query 和 gate），远大于 hidden_size。`from_hf_config` 中通过 `getattr(hf_config, "head_dim", ...)`
正确处理了这一点（`args.py:148-152`）。

**误区二：两种注意力层共享 head 配置（n_heads、n_kv_heads）。**

**不是**。full_attention 层使用 `n_heads`（24）、`n_kv_heads`（4）、`head_dim`（256）；linear_attention
层使用完全独立的
`linear_num_key_heads`（16）、`linear_num_value_heads`（32）、`linear_key_head_dim`（128）、`linear_value_head_dim`（128）。两套注意力的
head 数量、head 维度都不同——这不是简单的"换一种 attention kernel"，而是完全不同的投影配置。

💡 **小结**

- Qwen3.5 通过 `layer_types` 列表实现逐层混合注意力配置，配置从 HF config 原样读取后截断到实际层数
- `head_dim` 是显式配置的 256，不是从 hidden_size/n_heads 计算的——这是与 Qwen2/Qwen3 的关键差异
- 模型注册是导入时自动完成的，ArchonEngine 通过 `get_model_spec()` 根据 model_type 查找对应实现

______________________________________________________________________

## 二、混合 TransformerBlock 的调度机制：两种注意力如何共存

### 2.1 设计哲学与核心问题

混合注意力模型的核心设计问题是：同一个 `TransformerBlock` 类要支持两种完全不同的注意力子模块，但又不能让 Block 变成臃肿的"万能 Block"。

AReaL 的做法是**编译期决定、运行时静态分派**：每个 `TransformerBlock` 实例在 `__init__` 时就根据自己的 `layer_type`
确定使用哪个注意力模块。不使用的注意力模块不会被创建，也不占用内存。

### 2.2 源码入口与关键对象

```
areal/experimental/models/archon/qwen3_5/model/model.py
  - TransformerBlock (line 459)：混合 Block，根据 layer_type 分派
  - Qwen3_5Model (line 551)：顶层模型，管理所有 Block + 嵌入 + RoPE
```

### 2.3 主流程拆解

`TransformerBlock.__init__`（`model.py:462-498`）的关键分派逻辑：

```python
self.layer_type = model_args.layer_types[layer_id]

if self.layer_type == "full_attention":
    self.attention = GatedAttention(model_args)
    self.linear_attn = None
else:
    self.attention = None
    self.linear_attn = GatedDeltaNet(model_args, layer_idx=layer_id)
```

这不是一个 if/else 选择器模式——每个 Block 实例只有一个注意力模块不为 None。这意味着：

- FSDP wrapping 时，每个 Block 的参数量取决于它的 layer_type
- torch.compile 时，两种 Block 会被分别编译
- 模型遍历时，参数名空间是 `layers.{i}.attention.*` 或 `layers.{i}.linear_attn.*`

`TransformerBlock.forward`（`model.py:500-530`）同样是静态分派：

```python
if self.layer_type == "full_attention":
    x = x + self.attention(self.attention_norm(x), rope_cache, positions, cu_seqlens, max_seqlen)
else:
    x = x + self.linear_attn(self.attention_norm(x), cu_seqlens=cu_seqlens, seq_idx=seq_idx)
```

注意两个分支的参数签名完全不同：

- `GatedAttention` 需要 `rope_cache`、`positions`、`max_seqlen`
- `GatedDeltaNet` 需要 `seq_idx`，不需要 RoPE（线性注意力不使用位置编码）

但 `TransformerBlock.forward`
的函数签名是两者的**并集**：`(x, rope_cache, positions, cu_seqlens, max_seqlen, seq_idx=None)`。这是一种"宽接口"设计——调用方统一传递所有参数，Block
内部各取所需。

**顶层模型的 seq_idx 计算**。`Qwen3_5Model.forward`（`model.py:631-668`）做了一件重要的事：

```python
# Compute seq_idx ONCE for all linear_attention layers.
seq_idx = cu_seqlens_to_seq_idx(cu_seqlens, h.shape[1])
```

`seq_idx` 是一个从 `cu_seqlens` 推导的整数张量，标记每个 token 属于哪个序列。它用于 `causal_conv1d_fn`
的跨序列隔离。模型在顶层计算一次，通过 forward 参数传递给所有 Block——无论该 Block 是否是 linear_attention
层。full_attention 层收到 `seq_idx` 但不使用它。

### 2.4 关键细节与误区澄清

**误区三：GatedDeltaNet 不需要位置编码，是因为它内在编码了位置。**

部分正确。GatedDeltaNet 的 "位置感知" 来自两个机制：(1) **causal conv1d** 提供局部位置感知，(2) **gated delta rule
的 decay 和 beta** 提供一种衰减的"伪位置"信号。但它没有显式的 RoPE 或 ALiBi。这意味着 GatedDeltaNet 层的 Q/K
不经过旋转编码——这与 full_attention 层形成对比，后者使用 Partial RoPE（只旋转 head_dim 的
25%）。混合架构中，全注意力层在关键位置"锚定"全局位置信息，线性注意力层则靠局部特征和衰减实现位置感知。

**FeedForward 的选择与注意力类型无关**。无论 layer_type 是什么，MoE vs Dense FFN 的选择由
`model_args.moe_enabled` 控制，不与注意力类型耦合。这意味着可以有"线性注意力 + MoE"或"全注意力 + Dense FFN"的任意组合。

💡 **小结**

- TransformerBlock 通过"编译期决定、运行时静态分派"实现混合注意力，每个 Block 只实例化一种注意力模块
- 两种注意力的 forward 签名完全不同，通过宽接口统一传参
- `seq_idx` 由顶层模型计算一次，传递给所有层——full_attention 层忽略它
- FFN 的 MoE/Dense 选择与注意力类型正交

______________________________________________________________________

## 三、GatedDeltaNet 的计算链路：线性注意力的 forward 全过程

### 3.1 设计哲学与核心问题

GatedDeltaNet 是 Qwen3.5 中占比最大的注意力模块（典型配置中 75% 的层使用它）。它与标准注意力的根本区别在于：标准注意力是
`softmax(QK^T)V` 的显式注意力矩阵计算；GatedDeltaNet 是基于**状态递推**（state recurrence）的线性注意力——训练时用
chunk 分块并行计算，推理时可以像 RNN 一样逐步递推。

在训练框架中，GatedDeltaNet 的核心计算由外部库 `fla`（flash-linear-attention）的 `chunk_gated_delta_rule`
Triton kernel 完成。AReaL 的工作是：构建正确的输入张量、处理 packed sequence 的隔离、实现 per-head gated
normalization、并把这些接入统一的 TransformerBlock。

### 3.2 源码入口与关键对象

```
areal/experimental/models/archon/qwen3_5/model/model.py
  - GatedDeltaNet (line 141)：线性注意力模块
  - compute_decay_beta (line 115)：计算 decay 和 beta 参数
  - Qwen3_5RMSNormGated (line 62)：门控 RMSNorm（fla 不可用时的 fallback）
  - cu_seqlens_to_seq_idx (line 94)：packed sequence 的 sequence index 计算
```

### 3.3 主流程拆解

`GatedDeltaNet.forward`（`model.py:203-308`）是一个**八步流水线**：

**Step 1：输入投影**（`model.py:233-236`）

```python
mixed_qkv = self.in_proj_qkv(x)   # [B, T, conv_dim]  where conv_dim = key_dim*2 + value_dim
z = self.in_proj_z(x)              # [B, T, value_dim]  → 门控信号
a = self.in_proj_a(x)              # [B, T, num_v_heads] → decay 输入
b = self.in_proj_b(x)              # [B, T, num_v_heads] → beta 输入
```

这里有一个关键设计：QKV 不是三个独立投影，而是一个融合投影 `in_proj_qkv`，输出维度是 `key_dim + key_dim + value_dim`。而
`z`、`a`、`b` 是三个额外的小投影——这比标准注意力多了三个线性层。

**Step 2：Causal Convolution**（`model.py:239-265`）

```python
mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, conv_dim, T]
if causal_conv1d_fn is not None:
    mixed_qkv = causal_conv1d_fn(mixed_qkv, self.conv1d.weight.squeeze(1),
                                  bias=None, activation="silu", seq_idx=_seq_idx)
else:
    # Fallback: nn.Conv1d + SiLU（处理 packed sequence 需要按段拆分）
```

这是 GatedDeltaNet 与标准注意力的第一个重大差异：QKV 在投影之后、拆分之前，先经过一个 **depthwise causal
conv1d**。这个卷积的作用是引入局部上下文信息。

关键细节：

- `causal_conv1d_fn` 来自 `causal-conv1d` 库，是一个 CUDA 优化的 depthwise causal convolution，支持
  `seq_idx` 进行 packed sequence 隔离
- 卷积核大小默认为 4（`linear_conv_kernel_dim`），且是 depthwise 的（`groups=conv_dim`）
- Fallback 路径（`causal_conv1d_fn is None`）使用 `nn.Conv1d + SiLU`，但 packed sequence
  需要按段拆分处理以避免跨序列污染

**Step 3：Q/K/V 拆分与 reshape**（`model.py:268-275`）

```python
query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
```

注意 Q 和 K 的 head 数使用 `num_k_heads`（默认 16），V 的 head 数使用 `num_v_heads`（默认 32）。这是一种**不对称
head 配置**——V 的 head 数是 Q/K 的 2 倍。

**Step 4：Decay 和 Beta 计算**（`model.py:278`）

```python
beta, g = compute_decay_beta(self.A_log, self.dt_bias, a, b)
```

`compute_decay_beta`（`model.py:115-138`）的数学：

- `beta = sigmoid(b)` → 范围 (0, 1)，控制新信息写入强度
- `g = -exp(A_log) * softplus(a + dt_bias)` → **始终为负数**，是状态的指数衰减率

`A_log` 和 `dt_bias` 是两个**per-head 裸参数**（没有 `.weight` 后缀），在 state dict 中作为
`layers.{i}.linear_attn.A_log` 和 `layers.{i}.linear_attn.dt_bias` 存在。

**Step 5：Head Grouping**（`model.py:281-284`）

```python
if self.num_v_heads > self.num_k_heads:
    repeats = self.num_v_heads // self.num_k_heads
    query = query.repeat_interleave(repeats, dim=2)  # [B, T, num_v_heads, head_k_dim]
    key = key.repeat_interleave(repeats, dim=2)
```

类似 GQA（Grouped Query Attention），但方向相反：GQA 是 Q 多 KV 少，这里是 V 多 QK
少。`chunk_gated_delta_rule` 要求 Q/K/V 的 head 数相同，所以 Q/K 要扩展到 `num_v_heads`。默认配置下
`num_v_heads=32, num_k_heads=16`，所以 `repeat=2`。

**Step 6：Gated Delta Rule Kernel**（`model.py:291-299`）

```python
core_attn_out, _ = chunk_gated_delta_rule(
    query, key, value,
    g=g, beta=beta,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=cu_seqlens,
)
```

这是整个 forward 的**核心计算**，由 `fla` 库的 Triton kernel 完成。如果 `fla` 未安装，这里会直接 assert 失败——没有纯
PyTorch 的 fallback。

`use_qk_l2norm_in_kernel=True` 表示 kernel 内部会对 Q、K 做 L2 normalization，这是 Gated Delta Rule
的一个数值稳定性技巧。

**Step 7：Per-Head Gated Normalization**（`model.py:302-305`）

```python
core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
z = z.reshape(-1, self.head_v_dim)
core_attn_out = self.norm(core_attn_out, z)  # gated RMSNorm
core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
```

这是一步**per-head 维度**的门控 RMSNorm。注意 norm 的维度是 `head_v_dim`（每个 head 的值维度），不是全量
hidden_size。`z` 是 Step 1 中投影得到的门控信号，用于 `weight * norm(x) * silu(gate)` 的计算。

Norm 实现有两条路径（`model.py:188-194`）：

- 优先使用 `fla.modules.FusedRMSNormGated`（fused CUDA kernel）
- Fallback 到 `Qwen3_5RMSNormGated`（纯 PyTorch 实现）

**Step 8：输出投影**（`model.py:308`）

```python
return self.out_proj(core_attn_out)  # [B, T, hidden_size]
```

### 3.4 Shape 流分析

以 Qwen3.5-0.8B
默认参数为例（`dim=3072, num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128, conv_kernel=4`）：

```
输入: x [B, T, 3072]

Step 1 - 投影:
  mixed_qkv: [B, T, 6144]    (key_dim=2048 + key_dim=2048 + value_dim=4096)
  z:         [B, T, 4096]
  a:         [B, T, 32]
  b:         [B, T, 32]

Step 2 - Causal Conv1d:
  mixed_qkv: [B, 6144, T] → conv1d → silu → [B, 6144, T] → [B, T, 6144]

Step 3 - 拆分 + reshape:
  query:  [B, T, 16, 128]
  key:    [B, T, 16, 128]
  value:  [B, T, 32, 128]

Step 4 - Decay/Beta:
  beta: [B, T, 32]
  g:    [B, T, 32]   (负数)

Step 5 - Head Grouping (repeat_interleave 2x):
  query:  [B, T, 32, 128]
  key:    [B, T, 32, 128]

Step 6 - chunk_gated_delta_rule:
  core_attn_out: [B, T, 32, 128]

Step 7 - Per-head Gated Norm:
  reshape → [B*T*32, 128] → norm(x, gate) → reshape → [B, T, 4096]

Step 8 - 输出投影:
  output: [B, T, 3072]
```

### 3.5 关键细节与误区澄清

**误区四：`chunk_gated_delta_rule` 有纯 PyTorch 的 fallback。**

**没有**。`model.py:287-289` 明确 assert：如果 `chunk_gated_delta_rule is None`，直接报错。这与
`causal_conv1d_fn`（有 fallback）和 `FusedRMSNormGated`（有
fallback）不同。也就是说，`fla`（flash-linear-attention）是 Qwen3.5 训练的**硬依赖**。

**误区五：`conv1d` 权重在 state dict 中的形状包含 bias。**

`conv1d` 的 `bias=False`（`model.py:178`）。在 `causal_conv1d_fn` 调用中也显式传
`bias=None`。这个卷积只学权重，不学 bias。

💡 **小结**

- GatedDeltaNet 的 forward 是一个八步流水线：投影 → causal conv1d → 拆分 → decay/beta → head grouping
  → chunk kernel → per-head gated norm → 输出投影
- 核心计算 `chunk_gated_delta_rule` 无 PyTorch fallback，`fla` 是硬依赖
- Q/K 与 V 的 head 数不对称（默认 16 vs 32），Q/K 通过 repeat_interleave 扩展
- 两个裸参数 `A_log` 和 `dt_bias` 的命名不带 `.weight` 后缀

______________________________________________________________________

## 四、GatedAttention 的计算链路：门控全注意力的设计

### 4.1 设计哲学与核心问题

Qwen3.5 的全注意力层不是标准的 Multi-Head Attention——它在输出处加了一个**sigmoid 门控**。Q 的投影维度是 2 倍，一半是
query，一半是 gate。最终 `output = attn_output * sigmoid(gate)`。

这个设计的工程意义在于：给全注意力层增加了一个可学习的"开关"，模型可以学会在某些位置抑制注意力输出的影响。这与 GatedDeltaNet 中通过 `z`
实现的门控形成呼应——两种注意力都有某种形式的输出门控。

### 4.2 源码入口与关键对象

```
areal/experimental/models/archon/qwen3_5/model/model.py
  - GatedAttention (line 322)：门控全注意力

areal/experimental/models/archon/attention/varlen.py
  - VarlenAttentionWrapper (line 259)：Flash Attention 的 packed sequence wrapper
```

### 4.3 主流程拆解

`GatedAttention.forward`（`model.py:361-427`）是一个**十步流水线**：

```
1. Q 投影 (2x width) → 拆分为 query + gate
2. K, V 投影
3. Q/K RMSNorm（(1+weight)*norm(x) 语义）
4. Partial RoPE（只旋转前 25% 维度）
5. GQA 扩展（KV repeat）
6. transpose → [B, H, T, D]
7. Flash Attention via VarlenAttentionWrapper
8. transpose back + reshape
9. 输出门控：output = attn_output * sigmoid(gate)
10. 输出投影
```

**Partial RoPE 是 Qwen3.5 的另一个特色**。`rope.py:90-126` 中的 `apply_rotary_emb` 只对 head_dim 的前
`rotary_dim = int(head_dim * partial_rotary_factor)` 个维度做旋转，剩余维度直接 pass-through：

```python
rotary_dim = int(256 * 0.25) = 64  # 只有 64/256 = 25% 参与旋转

xq_rot, xq_pass = xq[..., :64], xq[..., 64:]
xq_out = cat([(xq_rot * cos) + (rotate_half(xq_rot) * sin), xq_pass], dim=-1)
```

这意味着 256 维的 head 中，**只有 64 维参与位置编码**，其余 192 维是纯内容特征。

### 4.4 关键细节与误区澄清

**误区六：GatedAttention 的 `wq` 输出维度是 `n_heads * head_dim`。**

**不是**。它是 `n_heads * head_dim * 2`（`model.py:342-344`）。Q 投影同时输出 query 和 gate，然后通过
`torch.chunk(qg, 2, dim=-1)` 拆分。对于 0.8B 模型，这个投影的输出维度是 `24 * 256 * 2 = 12288`。这使得
GatedAttention 的 Q 投影参数量是普通 MHA 的两倍。

**`VarlenAttentionWrapper` 不是 monkey patch**。它是一个 PyTorch custom
op（`torch.library.custom_op("areal::_varlen_attn")`，`varlen.py:18`），通过
`@torch.library.custom_op` 和 `@_varlen_attn.register_fake` / `setup_context` /
`backward` 手动注册 forward 和 backward，直接调用 `torch.ops.aten._flash_attention_forward`。这样做而不用
monkey patch 的原因是：custom op 对 `torch.compile` 友好——编译器可以把它当作一个不透明的算子节点处理，而 monkey patch
可能破坏 autograd 图的可追踪性。这与 FSDPEngine 中使用 monkey patch 替换 `_flash_attention_forward`
的方式形成鲜明对比——Archon 引擎选择了"compile-first"的设计路线。

💡 **小结**

- GatedAttention 的 Q 投影输出 2x width，一半是 query 一半是 gate，最终输出 = attn * sigmoid(gate)
- Partial RoPE 只旋转 head_dim 的 25%（64/256），其余维度直接 pass-through
- Archon 引擎的注意力后端是 custom op 而非 monkey patch，与 FSDPEngine 的实现路径不同

______________________________________________________________________

## 五、权重转换的双通道适配器：State Dict 的双向转换

### 5.1 设计哲学与核心问题

Archon 引擎在内部使用自己的命名规范（如 `layers.0.attention.wq.weight`），但 checkpoint 需要以 HuggingFace
格式保存/加载（如 `model.layers.0.self_attn.q_proj.weight`）。对于混合注意力模型，这个转换的复杂度翻倍——full_attention
层和 linear_attention 层的 HF 键名前缀完全不同。

### 5.2 源码入口与关键对象

```
areal/experimental/models/archon/qwen3_5/model/state_dict_adapter.py
  - Qwen3_5StateDictAdapter (line 21)：双向 state dict 转换器
  - from_hf()：HF → Archon
  - to_hf()：Archon → HF
  - convert_single_to_hf()：增量更新用（单个权重转换）
```

### 5.3 主流程拆解

`__init__`（`state_dict_adapter.py:34-133`）构建了一个 `from_hf_map` 字典，包含约 30
条映射规则。这些规则覆盖三类键名模式：

**full_attention 层（标准 MHA 命名）**：

```python
"model.layers.{}.self_attn.q_proj.weight" → "layers.{}.attention.wq.weight"
"model.layers.{}.self_attn.k_proj.weight" → "layers.{}.attention.wk.weight"
"model.layers.{}.self_attn.q_norm.weight" → "layers.{}.attention.q_norm.weight"
```

**linear_attention 层（GatedDeltaNet 命名）**：

```python
"model.layers.{}.linear_attn.in_proj_qkv.weight" → "layers.{}.linear_attn.in_proj_qkv.weight"
"model.layers.{}.linear_attn.conv1d.weight"       → "layers.{}.linear_attn.conv1d.weight"
"model.layers.{}.linear_attn.A_log"               → "layers.{}.linear_attn.A_log"      # 无 .weight
"model.layers.{}.linear_attn.dt_bias"             → "layers.{}.linear_attn.dt_bias"     # 无 .weight
```

**被跳过的键**：

```python
"model.layers.{}.self_attn.rotary_emb.inv_freq" → None  # 运行时计算，不加载
```

`from_hf()`（`state_dict_adapter.py:163-232`）处理 HF → Archon 转换时，对 MoE 权重有特殊逻辑：

1. 先尝试解析**新格式**（3D fused expert weights，如 `model.layers.0.mlp.experts.gate_up_proj`）
1. 再尝试解析**旧格式**（2D per-expert weights，如 `model.layers.0.mlp.experts.0.gate_proj.weight`）
1. 旧格式需要收集所有 expert 的权重后 stack 成 3D tensor

对于新格式，`gate_up_proj` 会被 split 成 `w1`（gate）和 `w3`（up），这个 split 点在 `dim=1` 的中间：

```python
half = value.shape[1] // 2
state_dict[f"layers.{layer_id}.moe.experts.w1"] = value[:, :half, :]
state_dict[f"layers.{layer_id}.moe.experts.w3"] = value[:, half:, :]
```

### 5.4 关键细节与误区澄清

**误区七：linear_attention 层的 Archon 键名与 HF 键名不同。**

**大部分相同**。对比 `from_hf_map` 可以看到，linear_attention 层的键映射大多是
`model.layers.{}.linear_attn.X` → `layers.{}.linear_attn.X`——只是去掉了 `model.`
前缀。命名空间本身保持一致（`linear_attn`）。而 full_attention 层则有实质性重命名（如 `self_attn.q_proj` →
`attention.wq`）。

**裸参数的处理**。`A_log` 和 `dt_bias` 在 HF 和 Archon 中都没有 `.weight` 后缀。这意味着
`model.named_parameters()` 返回的 name 是 `layers.0.linear_attn.A_log` 而不是
`layers.0.linear_attn.A_log.weight`。`convert_single_to_hf` 在增量转换时会自动 strip
`_checkpoint_wrapped_module` 和 `_orig_mod` 前缀（来自 AC 和 torch.compile 的 wrapper）。

💡 **小结**

- State dict adapter 用一张 `from_hf_map` 字典覆盖 full_attention 和 linear_attention 两套命名空间
- `A_log` 和 `dt_bias` 是裸参数，键名不带 `.weight` 后缀
- MoE 权重支持新旧两种 HF 格式（3D fused 和 2D per-expert）
- `rotary_emb.inv_freq` 被映射到 None，加载时跳过

______________________________________________________________________

## 六、并行化与编译：FSDP-Only 的当前状态

### 6.1 设计哲学与核心问题

Qwen3.5 的并行化是**当前最大的功能缺口**。与 Qwen3 的 `parallelize_qwen3`（757 行，支持
FSDP+TP+CP+EP）相比，`parallelize_qwen3_5` 只有 301 行，且**只实现了 FSDP**。

原因很直接：GatedDeltaNet 的 Q/K/V 维度与标准注意力不同，TP 需要在 head 维度切分，CP 需要在序列维度切分——这些都需要为
GatedDeltaNet 定制新的并行策略。`chunk_gated_delta_rule` kernel 是否支持分布式计算也是一个未解决的问题。

### 6.2 源码入口与关键对象

```
areal/experimental/models/archon/qwen3_5/infra/parallelize.py
  - parallelize_qwen3_5 (line 56)：入口函数
  - apply_fsdp (line 146)：FSDP2 wrapper
  - _apply_compile (line 226)：torch.compile（混合架构感知）
```

### 6.3 主流程拆解

`parallelize_qwen3_5`（`parallelize.py:56-143`）的执行顺序：

```
1. TP（不支持，打 warning 跳过）
2. CP（不支持，打 warning 跳过）
3. EP（不支持，打 warning 跳过）
4. AC（Activation Checkpointing）→ apply_ac()
5. torch.compile → _apply_compile()
6. FSDP → apply_fsdp()
7. Weight Tying（如果配置了 enable_weight_tying）
```

**FSDP wrapping 策略**（`parallelize.py:146-221`）：

```
tok_embeddings → fully_shard (单独 wrap)
每个 TransformerBlock → fully_shard (单独 wrap，不区分 layer_type)
[norm, output/score] → fully_shard (最后一组，不 reshard after forward)
model (root) → fully_shard
```

注意：FSDP wrapping 不区分 full_attention 和 linear_attention Block。两种 Block 都被同等 wrap。这是合理的，因为
FSDP 只关心参数分片，不关心 forward 的具体计算。

**torch.compile 策略**（`parallelize.py:226-295`）是**混合架构感知**的：

```python
for name, block in model.layers.items():
    if getattr(block, "moe_enabled", False):
        # MoE: 分别 compile 子模块（避免 GroupedExperts 导致 graph break）
        for attr_name, submod in inner_block.named_children():
            if isinstance(submod, moe_module.MoE):
                # MoE 的 experts 跳过，其他子模块单独 compile
            else:
                compile(submod)
    else:
        # 非 MoE：整个 Block 一起 compile
        model.layers[name] = torch.compile(block, backend="inductor", fullgraph=True)
```

对于 MoE Block，还有一个细节：`attention_norm` 和 `ffn_norm` 不会被单独
compile（`parallelize.py:266-267`），因为它们在外层 Block compile 时已经包含在图中。

对于非 MoE 的 Block（无论是 full_attention 还是 linear_attention），整个 Block 被
`torch.compile(fullgraph=True)` 编译。这意味着 GatedDeltaNet 中的 `chunk_gated_delta_rule`（Triton
kernel）和 `causal_conv1d_fn`（CUDA kernel）都需要能被 torch.compile 正确捕获。

### 6.4 关键细节与误区澄清

**TP/CP/EP 不是"报错"而是"静默忽略"**。如果用户配置了 `tensor_parallel_size > 1`，框架不会报错，只会打 warning：

```
"Qwen3.5 does not yet support Tensor Parallelism. TP will be ignored."
```

但模型仍然会创建和训练——只是 TP 维度不生效。这是一个**静默降级**行为，可能导致用户以为 TP 已生效但实际全用 FSDP。

💡 **小结**

- Qwen3.5 当前只支持 FSDP 并行，TP/CP/EP 请求会被静默忽略（打 warning 但不报错）
- FSDP wrapping 不区分 layer_type，两种 Block 同等处理
- torch.compile 是混合架构感知的：MoE Block 分子模块编译，非 MoE Block 整体编译
- 这是当前最大的功能缺口——缺少 TP 意味着大模型（70B+）训练需要更多 GPU

______________________________________________________________________

## 七、完整主路径串联

### 7.1 完整调用栈

```
User: 指定 model_path = "Qwen/Qwen3.5-0.8B", backend = "archon"
  │
  ├─ Step 1: ArchonEngine.__init__()
  │     ├─ AutoConfig.from_pretrained() → model_config (包含 layer_types, linear_* 等)
  │     ├─ _validate_model_type() → 检查 is_supported_model()
  │     └─ get_model_spec("qwen3_5") → QWEN3_5_SPEC
  │
  ├─ Step 2: create_process_group()
  │     ├─ dist.init_process_group() → NCCL 初始化
  │     └─ ArchonParallelDims() → 建立 device mesh (dp_shard only for Qwen3.5)
  │
  ├─ Step 3: initialize()
  │     ├─ _create_device_model()
  │     │     ├─ Qwen3_5ModelArgs.from_hf_config(model_config) → model_args
  │     │     └─ Qwen3_5Model(model_args) → model (on meta device, zero memory)
  │     │           ├─ 36 个 TransformerBlock (根据 layer_types 分别持有 GatedAttention 或 GatedDeltaNet)
  │     │           ├─ tok_embeddings, norm, output
  │     │           └─ rope_cache (precomputed Partial RoPE)
  │     │
  │     ├─ _create_state_dict_adapter() → Qwen3_5StateDictAdapter
  │     │
  │     ├─ _setup_parallelism()
  │     │     └─ parallelize_qwen3_5()
  │     │           ├─ TP/CP/EP → warnings, skip
  │     │           ├─ apply_ac() → 激活检查点
  │     │           ├─ _apply_compile() → torch.compile (混合架构感知)
  │     │           └─ apply_fsdp() → fully_shard 每个 Block + root
  │     │
  │     ├─ _materialize_and_load_weights()
  │     │     └─ load_model_from_hf()
  │     │           ├─ model state dict → to_hf() → HF key space
  │     │           ├─ DCP load from HF checkpoint
  │     │           ├─ from_hf() → back to Archon key space
  │     │           └─ model.load_state_dict()
  │     │
  │     └─ _create_optimizer() → AdamW + LR scheduler
  │
  ├─ Step 4: train_batch() (每个 training step)
  │     ├─ optimizer_zero_grad()
  │     ├─ _prepare_mb_list() → micro-batch 准备
  │     ├─ runner.run() → forward_backward through micro-batches
  │     │     └─ Qwen3_5Model.forward()
  │     │           ├─ tok_embeddings(tokens) → [B, T, dim]
  │     │           ├─ cu_seqlens_to_seq_idx() → seq_idx (一次性计算)
  │     │           ├─ for layer in layers:
  │     │           │     └─ TransformerBlock.forward()
  │     │           │           ├─ if full_attention:
  │     │           │           │     └─ GatedAttention(norm(x), rope, pos, cu_seqlens, max_seqlen)
  │     │           │           └─ else:
  │     │           │                 └─ GatedDeltaNet(norm(x), cu_seqlens, seq_idx)
  │     │           ├─ norm(h)
  │     │           └─ output(h) → logits [B, T, vocab_size]
  │     └─ optimizer_step() → gradient clipping + Adam update
  │
  └─ Step 5: save_model_to_hf() (checkpoint 保存)
        ├─ model state dict
        ├─ to_hf() → 转为 HF 格式 (包含 self_attn.* 和 linear_attn.* 混合键)
        └─ DCP save
```

### 7.2 每一层做了什么

| 步骤                            | 输入         | 输出                 | 状态变化        | 通信                 | 显存影响             | 频率                 |
| ------------------------------- | ------------ | -------------------- | --------------- | -------------------- | -------------------- | -------------------- |
| AutoConfig.from_pretrained      | model_path   | PretrainedConfig     | 无              | 无                   | CPU only             | 初始化 1 次          |
| Qwen3_5ModelArgs.from_hf_config | HF config    | Archon args          | 无              | 无                   | CPU only             | 初始化 1 次          |
| Qwen3_5Model (meta)             | args         | model (meta)         | 创建模型结构    | 无                   | 0 GPU memory         | 初始化 1 次          |
| parallelize_qwen3_5             | model        | model (FSDP wrapped) | FSDP hooks 注册 | 无                   | 0 (still meta)       | 初始化 1 次          |
| load_model_from_hf              | model + path | model (materialized) | 参数加载到 GPU  | DCP all-gather       | 峰值显存             | 初始化 1 次          |
| TransformerBlock.forward        | \[B,T,dim\]  | \[B,T,dim\]          | 无              | FSDP unshard/reshard | 激活值               | 每 step 每层         |
| chunk_gated_delta_rule          | Q,K,V,g,beta | attn_out             | 无              | 无                   | chunk 状态           | 每 step 每 linear 层 |
| VarlenAttentionWrapper          | Q,K,V        | attn_out             | 无              | 无                   | Flash Attn workspace | 每 step 每 full 层   |

### 7.3 哪些逻辑不在主路径

| 函数 / 文件                        | 为什么容易误解              | 实际是否在主流程                                                  |
| ---------------------------------- | --------------------------- | ----------------------------------------------------------------- |
| `GatedDeltaNet.init_weights()`     | 看起来像初始化路径          | 只在 from-scratch 训练时调用，从 checkpoint 加载时不执行          |
| `Qwen3_5Model.init_buffers()`      | 看起来每次 forward 前要调用 | 只在初始化时调用一次，将 rope_cache 和 MoE buffer 移到 GPU        |
| causal_conv1d_fn fallback          | 看起来是重要的兼容路径      | Docker 镜像自带 causal-conv1d，生产环境不走 fallback              |
| `Qwen3_5RMSNormGated`              | 看起来是主要的 norm 实现    | 当 fla 可用时用 FusedRMSNormGated，纯 PyTorch 版只在 fla 缺失时用 |
| `_split_moe_experts_distributed()` | 看起来很复杂                | 只在 DCP 分布式保存 MoE 权重时触发                                |

______________________________________________________________________

## 八、显存、性能与通信分析

### 8.1 显存收益分析

| 内容             | GatedDeltaNet 是否节省 | 说明                                                                                                                                                           |
| ---------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 参数量           | ❌ 不节省，反而更多    | GatedDeltaNet 有 6 个投影层（in_proj_qkv/z/a/b + conv1d + out_proj）+ 2 个裸参数 + gated norm。标准 MHA 只有 4 个（QKVO）。但 GatedDeltaNet 的投影维度可能更小 |
| 前向激活值       | ✅ 节省                | 标准注意力的显存瓶颈是 O(n²) 的注意力矩阵；GatedDeltaNet 是 O(n) 的 chunk 状态。序列越长，节省越明显                                                           |
| KV Cache（推理） | ✅ 大幅节省            | 线性注意力不需要 KV cache，改为固定大小的 recurrent state。但本文只关注训练                                                                                    |
| Optimizer State  | ❌ 不节省              | Adam 对每个参数都存 m/v，参数量不减甚至略增                                                                                                                    |
| 通信量（FSDP）   | ❌ 不节省              | FSDP 按参数量通信，参数量不减                                                                                                                                  |

**真正的显存大头**：对于长序列训练，激活值是显存瓶颈。在 Qwen3.5 的混合架构中，75% 的层使用线性注意力（O(n) 激活值），只有 25%
的层使用全注意力（O(n²) 激活值）。这意味着激活值显存大约是纯全注意力模型的 25%+75%×c（其中 c \< 1，取决于 chunk 大小和序列长度的比值）。

### 8.2 通信开销

当前 Qwen3.5 只使用 FSDP 并行，通信模式如下：

| 通信类型       | 时机                                                       | 频率         | 通信组   |
| -------------- | ---------------------------------------------------------- | ------------ | -------- |
| All-Gather     | 每个 TransformerBlock forward 前（unshard）                | 每 step 每层 | dp_shard |
| Reduce-Scatter | 每个 TransformerBlock backward 后（reshard + grad reduce） | 每 step 每层 | dp_shard |
| All-Reduce     | optimizer step 中的 gradient clipping                      | 每 step 1 次 | dp       |

由于不支持 TP/CP，所有通信都在 dp_shard 组内发生。没有 head 维度的 All-to-All（TP），也没有序列维度的 All-to-All（Ulysses
CP）。

**GatedDeltaNet 特有的通信特征**：`chunk_gated_delta_rule` kernel 本身不触发任何分布式通信——它是一个纯本地计算。这与
Ulysses 序列并行中的全注意力不同（后者需要 All-to-All 切换 head/seq 维度）。但这也意味着当前的 GatedDeltaNet
无法受益于序列并行——整个序列必须在单个 rank 上计算。

### 8.3 性能取舍

```
混合注意力的收益：
  ✅ 75% 的层用 O(n) 线性注意力代替 O(n²) softmax 注意力
  ✅ 推理时线性注意力层可以用 RNN 式递推，不需要 KV cache
  ✅ 长序列训练时激活值显存显著减少

混合注意力的代价：
  ❌ GatedDeltaNet 每层有 6 个投影 + 2 个裸参数 + 1 个 conv1d + 1 个 gated norm
  ❌ 依赖 fla 和 causal-conv1d 两个外部 CUDA/Triton 库
  ❌ 不支持 TP/CP/EP，大模型训练受限
  ❌ torch.compile + Triton kernel 的兼容性可能有隐患
  ❌ 两种 Block 的 compile cache 各自独立，编译时间可能翻倍
```

______________________________________________________________________

## 九、配置项、边界条件与坑点

| 配置项                                            | 影响的源码路径                  | 行为变化                                     | 风险/坑点                                                                                   |
| ------------------------------------------------- | ------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `layer_types`                                     | `TransformerBlock.__init__`     | 决定每层用 GatedAttention 还是 GatedDeltaNet | 长度必须等于 n_layers，否则 ValueError。如果为空列表，所有层都走 else 分支（GatedDeltaNet） |
| `partial_rotary_factor`                           | `rope.py:precompute_rope_cache` | 控制 RoPE 覆盖的 head_dim 比例               | 默认 0.25，只影响 full_attention 层。如果设为 1.0 则退化为标准 RoPE                         |
| `linear_num_value_heads` / `linear_num_key_heads` | `GatedDeltaNet.__init__`        | V heads 必须是 K heads 的整数倍              | 不整除会在 head grouping 时出 bug（无显式检查）                                             |
| `head_dim`                                        | `GatedAttention.__init__`       | 决定 Q 投影维度（= n_heads × head_dim × 2）  | Qwen3.5 的 head_dim=256 远大于 hidden_size/n_heads，Q 投影参数量很大                        |
| `tp > 1`                                          | `parallelize_qwen3_5`           | **静默忽略**，打 warning                     | 用户可能以为 TP 生效但实际没有                                                              |
| `cp > 1`                                          | `parallelize_qwen3_5`           | **静默忽略**，打 warning                     | 同上                                                                                        |
| `ep > 1`                                          | `parallelize_qwen3_5`           | **静默忽略**，打 warning                     | MoE 模型可能需要 EP 但不可用                                                                |
| `fla` 未安装                                      | `GatedDeltaNet.forward`         | **Hard assert 失败**                         | 训练会直接挂掉，不是静默降级                                                                |
| `causal-conv1d` 未安装                            | `GatedDeltaNet.forward`         | 走纯 PyTorch fallback                        | 性能下降但功能正确                                                                          |
| `moe_enabled`                                     | `TransformerBlock.__init__`     | 选择 MoE 或 Dense FFN                        | 与注意力类型正交，可以任意组合                                                              |

**静默失效条件**：如果 `layer_types` 是空列表（`[]`），`__post_init__` 不会报错（因为 `if self.layer_types` 为
False 就跳过检查）。注意 `layer_types` 的默认值就是 `field(default_factory=list)` 即空列表（`args.py:72`）。此时
`TransformerBlock.__init__` 在 `model_args.layer_types[layer_id]`（`model.py:466`）处会抛
`IndexError`——但报错信息是 "list index out of range"，不会提示用户是 `layer_types`
配置缺失。这是一个验证缺口：`__post_init__` 应该对空列表也做检查。

______________________________________________________________________

## 十、测试覆盖与缺口

### 10.1 已覆盖路径

| 测试 / 示例                                      | 覆盖的行为                                                                  | 说明                 |
| ------------------------------------------------ | --------------------------------------------------------------------------- | -------------------- |
| `test_qwen3_5.py::TestFromHfConfig` (#1-#4)      | 配置解析（dense/MoE/VLM/非法）                                              | CPU 测试，覆盖全面   |
| `test_qwen3_5.py::TestRMSNorm` (#5-#6)           | 两种 RMSNorm 与 HF 的数值一致性                                             | CPU 测试             |
| `test_qwen3_5.py::TestGatedDeltaNet` (#14-#19b)  | GatedDeltaNet 的 shape/HF parity/packing/backward/cross-oracle              | GPU 测试，覆盖最核心 |
| `test_qwen3_5.py::TestGatedAttention` (#20-#22)  | 门控逻辑/HF parity/backward                                                 | GPU 测试             |
| `test_qwen3_5.py::TestConv1dBoundary` (#23-#23b) | causal_conv1d 的 packed sequence 隔离                                       | GPU 测试             |
| `test_qwen3_5.py` Integration (#29-#35)          | TransformerBlock forward (两种层)/Model forward/backward/HF backward parity | GPU 测试，4 层模型   |
| `test_hf_parity_qwen3_5.py`                      | Dense 模型 E2E HF parity（cosine sim > 0.95）                               | GPU 测试，真实权重   |
| `test_hf_parity_qwen3_5_moe.py`                  | MoE 模型 E2E HF parity（cosine sim > 0.90）                                 | GPU 测试，真实权重   |
| `test_state_dict_adapter.py` (Qwen3.5 部分)      | State dict adapter roundtrip                                                | CPU 测试，包含裸参数 |

> 所有测试文件位于 `tests/experimental/archon/` 目录下。Qwen3.5 测试需要 `transformers >= 5.2.0`（见
> `conftest.py`）。

### 10.2 未覆盖风险

| 风险点                                   | 当前是否有测试 | 可能后果                                                                      |
| ---------------------------------------- | -------------- | ----------------------------------------------------------------------------- |
| 分布式训练（多 GPU FSDP）                | ❌ 无          | FSDP wrapping 后 GatedDeltaNet 的 forward/backward 可能有 shape 或 dtype 问题 |
| torch.compile + GatedDeltaNet            | ❌ 无          | Triton kernel 在 compile 图中可能 graph break                                 |
| 大序列长度（>4K）                        | ❌ 无          | chunk_gated_delta_rule 在长序列下的数值稳定性未验证                           |
| Checkpoint save/load roundtrip           | ❌ 无          | 混合层 state dict 的 DCP 保存+加载可能丢键或错位                              |
| Activation Checkpointing + GatedDeltaNet | ❌ 无          | AC 重计算 causal_conv1d 可能有副作用                                          |
| 空 layer_types 列表                      | ❌ 无          | 会抛 IndexError 但报错不够明确                                                |
| FP8 + GatedDeltaNet                      | ❌ 无          | FP8 量化是否能正确应用到 GatedDeltaNet 的投影层                               |
| TP/CP 静默忽略                           | ❌ 无          | 用户可能不知道并行策略未生效                                                  |
| MoE + 线性注意力组合                     | 部分覆盖       | MoE roundtrip 测试存在，但未测试 MoE + GatedDeltaNet 的联合 forward           |
| Pipeline Parallelism                     | ❌ 无          | `pipelining_fn=pipeline_llm` 已注册但未测试                                   |

**关键缺口**：没有任何分布式测试和 compile 测试。这是生产环境部署前必须补充的。

______________________________________________________________________

## 十一、局限性与已知优化点

### 11.1 硬约束

- `linear_num_value_heads` 必须能被 `linear_num_key_heads` 整除（head grouping 要求）
- `layer_types` 长度必须等于 `n_layers`
- `fla`（flash-linear-attention）是硬依赖，不可缺失
- 当前只支持 FSDP 并行，TP/CP/EP 不可用
- `head_dim` 对于全注意力层是显式配置的，不能从 hidden_size 推导
- `transformers >= 5.2.0` 是测试依赖（HF Qwen3.5 模型类需要）

### 11.2 维护成本

- **双注意力路径**：每次修改 TransformerBlock 或模型 forward 时，都需要同时考虑两种注意力的行为。State dict adapter
  也需要维护两套映射。
- **外部库依赖**：`fla` 和 `causal-conv1d` 都是从源码构建的（见 Dockerfile），版本升级可能破坏 Triton kernel
  行为。`pyproject.toml` 中这两个依赖标记为 `sys_platform == 'never'`，只通过 Docker 安装。
- **TP/CP 待实现**：当前 TP/CP 的缺失意味着大模型（如 Qwen3.5-72B MoE）无法高效训练。实现 TP 需要为 GatedDeltaNet
  的每个投影层定义 shard 策略。
- **并行策略静默降级**：TP/CP/EP 请求不报错而是打 warning，容易被用户忽略。

### 11.3 性能瓶颈

- **FSDP 每层通信**：每个 TransformerBlock 的 forward 都触发一次 all-gather（unshard），backward 触发
  reduce-scatter。36 层模型每 step 通信 72 次。这是 FSDP 的固有开销，但不支持 TP 意味着不能通过减少 FSDP world_size
  来降低通信量。
- **GatedDeltaNet 不能序列并行**：`chunk_gated_delta_rule` 是纯本地计算，不支持跨 rank
  的序列切分。长序列场景下，整个序列必须在单个 rank 上处理。
- **编译时间翻倍**：两种 Block 类型意味着 torch.compile 需要分别编译两个 graph，首次训练的编译时间可能翻倍。
- **head grouping 的 repeat_interleave**：每次 forward 都对 Q/K 做
  `repeat_interleave`，这是一个显存复制操作。对于长序列，这个开销不可忽视。

### 11.4 已知优化点

1. **TP 实现**：最高优先级。GatedDeltaNet 的 `in_proj_qkv` 可以按 output dim 切分（类似标准 QKV 的
   TP），`out_proj` 按 input dim 切分。但 `in_proj_a`、`in_proj_b` 的输出维度是 `num_v_heads`，切分后需要确保
   chunk_gated_delta_rule kernel 能处理 partial heads。
1. **序列并行**：`chunk_gated_delta_rule` 的 chunk 之间存在状态传递（delta rule state），理论上可以用 ring
   通信实现跨 rank 的 chunk 传递，但需要修改 fla kernel。
1. **融合 head grouping**：`repeat_interleave` 可以通过 kernel 融合避免显式复制。
1. **FusedRMSNormGated 的更优实现**：当前 fla 提供的 fused kernel 已经比纯 PyTorch 快，但可能还有 cutlass
   等更优实现。
1. **conv1d 和 chunk kernel 的 overlap**：当前 causal_conv1d 和 chunk_gated_delta_rule
   是串行调用的，理论上不同 head 组可以 pipeline overlap。

______________________________________________________________________

## 小结与展望

AReaL 的 Qwen3.5 混合注意力实现可以用几个关键词概括。

**关键词一：编译期静态分派**

TransformerBlock 在 `__init__` 时确定 layer_type，只创建一种注意力模块。运行时不存在动态切换——这保证了 torch.compile
可以为每种 Block 生成完整的 fullgraph。这是一种简单但有效的设计：通过在 Python 层面消除条件分支，将"混合"的复杂度封锁在模型构建阶段。

**关键词二：裸参数与不对称 head**

GatedDeltaNet 引入了 `A_log`、`dt_bias` 两个没有 `.weight` 后缀的裸参数，以及 Q/K head 数少于 V head
数的不对称配置。这些都是线性注意力特有的设计，要求 state dict adapter、FSDP wrapping、torch.compile 都能正确处理。

**关键词三：外部 kernel 硬依赖**

核心计算 `chunk_gated_delta_rule` 完全依赖 `fla` 库的 Triton kernel，没有 PyTorch fallback。这与
`causal_conv1d`（有 fallback）和 `FusedRMSNormGated`（有 fallback）形成对比。框架选择了"性能优先、可移植性次之"的策略。

**关键词四：FSDP-only 的现实约束**

当前实现只支持 FSDP 并行。TP/CP/EP 都还未实现。这意味着 Qwen3.5 在 AReaL 中的训练规模受限于单机 GPU 数量——每个 rank
必须能放下完整的模型层（FSDP 分片参数，但每层的激活值不分片）。对于 35B-A3B 的 MoE 模型，这可能是一个实际瓶颈。

**关键词五：门控是统一设计语言**

GatedDeltaNet 用 `z` 做 gated RMSNorm，GatedAttention 用 `sigmoid(gate)`
做输出门控——两种注意力都有"门控"机制。这不是巧合，而是 Qwen3.5 架构的统一设计哲学：通过可学习的门控让模型自适应地调节每个位置的信息流强度。

**适用场景**：当前实现最适合在**单机 8 卡**的场景下训练中小规模 Qwen3.5 模型（如 0.8B dense）。长序列训练可以显著受益于混合注意力的显存节省。

**不适合的场景**：大规模 MoE 模型（需要 EP）、超大模型（需要 TP）、超长序列（需要 CP/SP）——这些都需要等待 TP/CP/EP 的实现。

**后续值得继续走读的方向**：

1. `fla` 库的 `chunk_gated_delta_rule` 内部实现——了解 chunk 之间的状态传递机制
1. Qwen3 的 `parallelize_qwen3`——理解完整的 TP/CP/EP 实现，为 Qwen3.5 补全并行策略提供参考
1. `LightningSelfAttention`（BailingMoe 的线性注意力）——另一种线性注意力在 Megatron-Core 中的 TP/CP 实现，可能为
   Qwen3.5 的并行化提供经验
1. ArchonEngine 的 weight sync 机制——理解训练过程中模型权重如何同步到推理引擎
