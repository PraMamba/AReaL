# Archon Ulysses 序列并行深度解析

> 源文件：`areal/experimental/models/archon/ulysses.py`（81 行）+ `areal/models/fsdp/ulysses.py`（283 行）
> 核心函数：`gather_seq_scatter_heads` · `gather_heads_scatter_seq` · `ulysses_slice_inputs` · `ulysses_gather_output`

---

[TOC]

---

# 1. 白话解释

## 1.1 一句话总结

Ulysses 序列并行通过 **All-to-All 通信在注意力头维度切分**，让每个 GPU 只计算部分注意力头的完整序列，从而支持超长序列训练——不同于 Ring Attention 在序列维度切分，Ulysses 实现更简单且与 Tensor Parallel 自然兼容。

## 1.2 现实类比

```text
想象一个 8 人团队阅读一本 8000 页的书并做笔记：

方案 A：每人读全书，各自做笔记
  → 太慢，一个人读不完（= 单 GPU 放不下长序列）

方案 B：Ring Attention（序列切分）
  ┌─────────────────────────────────────────────────────┐
  │ 每人负责 1000 页，但要看全书的笔记才能理解上下文    │
  │ → 需要环形传递笔记，每人轮流看其他人的部分         │
  │ → 实现复杂，需要处理边界                           │
  └─────────────────────────────────────────────────────┘

方案 C：Ulysses（注意力头切分）
  ┌─────────────────────────────────────────────────────┐
  │ 每人读全书（8000 页），但只做 1/8 的笔记类型        │
  │ 例如：                                              │
  │   - 人 0 只记录"人物关系"                           │
  │   - 人 1 只记录"时间线"                             │
  │   - 人 2 只记录"地点"                               │
  │   ...                                               │
  │                                                     │
  │ 阅读前：All-to-All 交换书页                         │
  │   每人把自己的 1000 页分成 8 份，交换后每人拿到     │
  │   所有人的第 i 份（共 8000 页）                     │
  │                                                     │
  │ 阅读后：All-to-All 交换笔记                         │
  │   每人把笔记分成 8 份，交换后每人拿到完整笔记的     │
  │   第 i 部分（1000 页的所有笔记类型）                │
  └─────────────────────────────────────────────────────┘

Ulysses 的优势：
  - 每人读完整上下文（全序列），理解不受限
  - 只需两次 All-to-All（前后各一次），通信模式规整
  - 与 TP 兼容（TP 切 hidden，Ulysses 切 heads）
```

## 1.3 这个文件做了什么

```text
ulysses.py (Archon 封装层)
  │
  ├── ulysses_slice_inputs()        ← Engine 调用：切分输入序列
  │     输入: input_ids [bs, seq_len], labels [bs, seq_len]
  │     输出: input_ids [bs, seq_len/cp], labels [bs, seq_len/cp]
  │
  ├── gather_seq_scatter_heads()    ← Attention 调用：前向 All-to-All
  │     输入: xq [bs, seq/cp, heads, dim]
  │     输出: xq [bs, seq, heads/cp, dim]
  │     通信: 每个 rank 的序列片段 → 所有 rank 的头片段
  │
  ├── gather_heads_scatter_seq()    ← Attention 调用：反向 All-to-All
  │     输入: output [bs, seq, heads/cp, dim]
  │     输出: output [bs, seq/cp, heads, dim]
  │     通信: 每个 rank 的头片段 → 所有 rank 的序列片段
  │
  └── ulysses_gather_output()       ← Engine 调用：聚合输出
        输入: logprobs [bs, seq/cp, vocab]
        输出: logprobs [bs, seq, vocab]
        通信: All-Gather 拼接所有 rank 的序列片段

底层实现 (areal/models/fsdp/ulysses.py)
  │
  ├── all_to_all_tensor()           ← 核心通信原语
  │     使用 all_to_all_single_autograd（torch.compile 兼容）
  │     自动处理 autograd
  │
  ├── _pad_tensor() / _unpad_tensor()  ← 对齐处理
  │     确保序列长度能被 cp_size 整除
  │
  └── slice_input_tensor()          ← 输入切分
        按 cp_rank 切分序列维度
```

## 1.4 核心不变量

```
1. 序列长度必须能被 cp_size 整除（通过 padding 保证）
2. 注意力头数必须能被 cp_size 整除（通过 repeat_kv 保证 GQA）
3. All-to-All 前后数据总量不变：
   gather_seq_scatter_heads: [bs, seq/cp, heads, d] → [bs, seq, heads/cp, d]
   gather_heads_scatter_seq: [bs, seq, heads/cp, d] → [bs, seq/cp, heads, d]
```

---

# 2. 前置概念

## 2.1 All-to-All 通信模式

All-to-All 是一种集体通信原语，每个 rank 向所有其他 rank 发送不同的数据，并从所有 rank 接收数据。

```text
示例：4 个 rank，每个 rank 有 4 个数据块

发送前：
  Rank 0: [A0, A1, A2, A3]
  Rank 1: [B0, B1, B2, B3]
  Rank 2: [C0, C1, C2, C3]
  Rank 3: [D0, D1, D2, D3]

All-to-All 后：
  Rank 0: [A0, B0, C0, D0]  ← 收集所有 rank 的第 0 块
  Rank 1: [A1, B1, C1, D1]  ← 收集所有 rank 的第 1 块
  Rank 2: [A2, B2, C2, D2]  ← 收集所有 rank 的第 2 块
  Rank 3: [A3, B3, C3, D3]  ← 收集所有 rank 的第 3 块

关键特性：
  - 每个 rank 发送 N 块，接收 N 块（N = world_size）
  - 数据总量不变，只是重新分布
  - 支持不同维度的 scatter/gather
```

## 2.2 Ulysses vs Ring Attention

| 特性 | Ring Attention | Ulysses (Archon) |
| --- | --- | --- |
| 切分维度 | 序列维度（seq） | 注意力头维度（heads） |
| 通信模式 | P2P 环形通信 | All-to-All 集体通信 |
| 通信次数 | O(cp_size) 次 P2P | 2 次 All-to-All |
| 计算特性 | 块级注意力（需处理边界） | 全序列注意力（无边界） |
| 实现复杂度 | 高（自定义 CUDA kernel） | 低（PyTorch 原生 API） |
| TP 兼容性 | 需特殊处理 | 自然兼容（TP 切 hidden，CP 切 heads） |
| 适用场景 | 极长序列（>1M tokens） | 长序列（10K-100K tokens） |

**为什么 Ulysses 更简单？**

Ring Attention 需要：
1. 自定义 CUDA kernel 实现块级注意力
2. 处理块边界的 KV cache 传递
3. 复杂的环形通信调度

Ulysses 只需：
1. 两次 All-to-All（PyTorch 原生支持）
2. 标准的 Flash Attention（无需修改）
3. 简单的 padding/unpadding

## 2.3 GQA（Grouped Query Attention）与 Ulysses

GQA 模型的 KV 头数少于 Q 头数（例如 Qwen2-7B：32 Q heads, 4 KV heads）。Ulysses 要求头数能被 cp_size 整除，因此需要 **repeat_kv** 操作。

```text
示例：cp_size=4, Q heads=32, KV heads=4

原始：
  xq: [bs, seq/4, 32, dim]  ← 每个 rank 持有 32/4=8 个 Q 头
  xk: [bs, seq/4,  4, dim]  ← 每个 rank 持有 4/4=1 个 KV 头

问题：4 < 4，无法均分！

解决：repeat_kv(xk, repeats=4/4=1) 不需要重复
      但如果 cp_size=8，则 repeat_kv(xk, repeats=8/4=2)
      xk: [bs, seq/8,  8, dim]  ← 重复后可以均分

gather_seq_scatter_heads 后：
  xq: [bs, seq, 8, dim]   ← 每个 rank 持有完整序列的 8 个 Q 头
  xk: [bs, seq, 1, dim]   ← 每个 rank 持有完整序列的 1 个 KV 头（或重复后的）
```

**关键代码**（qwen2/model/model.py:142-148）：
```python
if self._sp_enabled:
    kv_heads = xk.size(2)
    if kv_heads < self._cp_size:
        repeats = self._cp_size // kv_heads
        xk = repeat_kv(xk, repeats)
        xv = repeat_kv(xv, repeats)
```

## 2.4 Padding 与对齐

Ulysses 要求序列长度能被 cp_size 整除。如果不满足，需要 padding。

```text
示例：seq_len=1000, cp_size=4

1000 % 4 = 0  ✓ 无需 padding

示例：seq_len=1001, cp_size=4

1001 % 4 = 1  ✗ 需要 padding
padding_size = 4 - 1 = 3
padded_seq_len = 1001 + 3 = 1004

切分后每个 rank：1004 / 4 = 251 tokens

gather 后需要 unpad：
  gathered_seq_len = 1004
  unpadded_seq_len = 1001
  unpad 3 个 tokens
```

**Padding 时机**：
- **输入 padding**：`ulysses_slice_inputs` 调用 `_ulysses_slice_tensor` 前，Engine 已通过 `pad_mb_list` 对齐到 `seq_len_divisor = tp * cp * 2`
- **All-to-All padding**：`gather_heads_scatter_seq` 内部调用 `_pad_tensor` 确保序列能被 cp_size 整除
- **输出 unpadding**：`gather_seq_scatter_heads` 使用 `unpadded_dim_size` 参数在 gather 后移除 padding

## 2.5 Ulysses 在 Archon 中的集成点

```text
┌─────────────────────────────────────────────────────────────┐
│                    Archon 训练流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 初始化阶段                                               │
│     parallelize_qwen2/3()                                   │
│       └─ apply_cp(model, cp_group)                          │
│            └─ layer.attention.set_cp_group(cp_group)        │
│                 ├─ self._cp_group = cp_group                │
│                 ├─ self._cp_size = dist.get_world_size(cp_group)│
│                 └─ self._cp_rank = dist.get_rank(cp_group)     │
│                                                             │
│  2. 前向传播前（Engine）                                     │
│     if parallel_dims.cp_enabled:                            │
│         inputs, labels = ulysses_slice_inputs(              │
│             inputs, labels, cp_rank, cp_size                │
│         )                                                   │
│     # 输入从 [bs, seq] 切成 [bs, seq/cp]                    │
│                                                             │
│  3. Attention 前向（Model）                                  │
│     xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)         │
│     xq, xk = apply_rotary_emb(xq, xk, ...)                  │
│                                                             │
│     if self._sp_enabled:                                    │
│         # 3.1 Repeat KV heads if needed                     │
│         if kv_heads < self._cp_size:                        │
│             xk = repeat_kv(xk, self._cp_size // kv_heads)   │
│             xv = repeat_kv(xv, self._cp_size // kv_heads)   │
│                                                             │
│         # 3.2 All-to-All: 序列 → 头                         │
│         xq = gather_seq_scatter_heads(xq, ...)              │
│         xk = gather_seq_scatter_heads(xk, ...)              │
│         xv = gather_seq_scatter_heads(xv, ...)              │
│         # [bs, seq/cp, heads, d] → [bs, seq, heads/cp, d]   │
│                                                             │
│     # 3.3 标准 Flash Attention                              │
│     output = self.packed_attn(xq, xk, xv, ...)             │
│                                                             │
│     if self._sp_enabled:                                    │
│         # 3.4 All-to-All: 头 → 序列                         │
│         output = gather_heads_scatter_seq(output, ...)      │
│         # [bs, seq, heads/cp, d] → [bs, seq/cp, heads, d]   │
│                                                             │
│  4. 前向传播后（Engine）                                     │
│     if self._cp_group is not None:                          │
│         logprobs = ulysses_gather_output(logprobs, ...)     │
│         entropy = ulysses_gather_output(entropy, ...)       │
│         vocab_min = ulysses_gather_output(vocab_min, ...)   │
│         vocab_max = ulysses_gather_output(vocab_max, ...)   │
│     # 输出从 [bs, seq/cp, vocab] 聚合成 [bs, seq, vocab]    │
│                                                             │
│  5. 参考模型 / Critic 推理后（Engine）                       │
│     if self._cp_group is not None:                          │
│         result = ulysses_gather_output(result, ...)         │
│         values = ulysses_gather_output(values, ...)         │
│     # 同理聚合参考 logprobs 和 critic values                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 2.6 seq_len_divisor 的作用

`ArchonParallelDims.seq_len_divisor` 返回 `tp * cp * 2`，确保序列长度满足所有并行策略的对齐要求：

```python
@property
def seq_len_divisor(self) -> int:
    return self.tp * self.cp * 2
```

**为什么是 `tp * cp * 2`？**

1. **TP 要求**：Sequence Parallel（TP 的一部分）要求 `seq_len % tp == 0`
2. **CP 要求**：Ulysses 要求 `seq_len % cp == 0`
3. **× 2**：Varlen attention 的因果掩码对齐需求（Flash Attention 内部优化）

**使用场景**：
- `pad_mb_list` 在数据加载时对齐 batch 序列长度
- `_ulysses_slice_tensor` 验证输入已对齐（否则抛出 ValueError）

---

# 3. 源码逐行地图

## 3.1 Archon 封装层：`areal/experimental/models/archon/ulysses.py`（81 行）

### 导入与公开 API（1-21 行）

```python
# 第 1-7 行: 标准库和 PyTorch 分布式 API
from typing import Any
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_F  # 用于 all_gather
from torch import Tensor
from torch.distributed import ProcessGroup

# 第 9-14 行: 从 FSDP 版 Ulysses 导入核心 All-to-All 函数
# 注意: Archon 复用 FSDP 引擎的底层实现，仅提供更简洁的接口
from areal.models.fsdp.ulysses import (
    _gather_heads_scatter_seq as gather_heads_scatter_seq,
)
from areal.models.fsdp.ulysses import (
    _gather_seq_scatter_heads as gather_seq_scatter_heads,
)

# 第 16-21 行: 公开 API
__all__ = [
    "ulysses_slice_inputs",
    "gather_seq_scatter_heads",
    "gather_heads_scatter_seq",
    "ulysses_gather_output",
]
```

**设计要点**：Archon 的 ulysses.py 是一个薄封装层，直接复用 `areal/models/fsdp/ulysses.py` 中的底层 All-to-All 实现。区别在于 Archon 版本使用显式�� `cp_group` 参数传递（通过 `set_cp_group` 设置），而 FSDP 版本使用全局变量 `_ULYSSES_SEQUENCE_PARALLEL_GROUP`。

### `_ulysses_slice_tensor`（24-42 行）

```python
def _ulysses_slice_tensor(
    tensor: Tensor,     # 待切分的张量
    cp_rank: int,       # 当前 CP rank
    cp_size: int,       # CP 并行度
) -> Tensor:
    """Slice a tensor along the last dimension for Ulysses SP."""
    total_len = tensor.shape[-1]   # 第 30 行: 取最后一维长度

    # 第 32-36 行: 对齐验证——确保 Engine 已通过 pad_mb_list 对齐
    if total_len % cp_size != 0:
        raise ValueError(
            f"Tensor length {total_len} not aligned to cp_size {cp_size}. "
            "Ensure pad_mb_list is called with batch_align_to=seq_len_divisor first."
        )

    # 第 38-42 行: 按 cp_rank 切分
    chunk_size = total_len // cp_size
    start = cp_rank * chunk_size
    end = start + chunk_size
    return tensor[..., start:end].contiguous()
    # .contiguous() 确保内存连续，避免后续计算出错
```

**图解**：

```text
cp_size=4, seq_len=8000

tensor: [t₀, t₁, t₂, ..., t₇₉₉₉]
         |---------|---------|---------|---------|
         chunk_0    chunk_1    chunk_2    chunk_3
         (2000)     (2000)     (2000)     (2000)

cp_rank=0: tensor[..., 0:2000]
cp_rank=1: tensor[..., 2000:4000]
cp_rank=2: tensor[..., 4000:6000]
cp_rank=3: tensor[..., 6000:8000]
```

### `ulysses_slice_inputs`（45-63 行）

```python
def ulysses_slice_inputs(
    inputs: dict[str, Any],   # 包含 input_ids, position_ids 等
    labels: Tensor,            # 标签张量
    cp_rank: int,
    cp_size: int,
) -> tuple[dict[str, Any], Tensor]:
    """Slice inputs and labels for Ulysses SP."""

    # 第 52-53 行: cp_size <= 1 时直接返回（不做切分）
    if cp_size <= 1:
        return inputs, labels

    # 第 55-61 行: 切分 input_ids, labels, position_ids
    inputs = dict(inputs)  # 浅拷贝避免修改原始字典
    inputs["input_ids"] = _ulysses_slice_tensor(inputs["input_ids"], cp_rank, cp_size)
    labels = _ulysses_slice_tensor(labels, cp_rank, cp_size)
    inputs["position_ids"] = _ulysses_slice_tensor(inputs["position_ids"], cp_rank, cp_size)

    return inputs, labels
```

**为什么 `position_ids` 也要切？** 因为 RoPE（旋转位置编码）依赖绝对位置。切分后每个 rank 持有的序列片段对应的位置也需要对齐：`[0,1,...,1999]` 对 rank 0，`[2000,...,3999]` 对 rank 1。

### `ulysses_gather_output`（66-80 行）

```python
def ulysses_gather_output(
    output: Tensor,                    # 待聚合的输出张量
    cp_group: ProcessGroup | None,     # CP 进程组
    seq_dim: int = 0,                  # 序列维度索引（默认 0）
) -> Tensor:
    """Gather output tensor from all CP ranks after forward pass."""

    # 第 72-77 行: 安全检查
    if cp_group is None:
        return output
    cp_size = dist.get_world_size(cp_group)
    if cp_size <= 1:
        return output

    # 第 79-80 行: All-Gather + 拼接
    gathered = dist_F.all_gather(output, group=cp_group)
    return torch.cat(gathered, dim=seq_dim)
```

**`All-Gather` vs `All-to-All`**：

| 操作 | 通信模式 | 使用场景 |
| --- | --- | --- |
| `All-to-All` | 每人发 N 块、收 N 块 | Attention 内部（头↔序列交换） |
| `All-Gather` | 每人发 1 块、收 N 块 | 输出聚合（拼接所有序列片段） |

为什么输出用 `All-Gather` 而不是 `All-to-All`？因为输出不需要维度交换——每个 rank 持有完整的 `heads` 维度输出，只需在序列维度上拼接。

---

## 3.2 底层实现：`areal/models/fsdp/ulysses.py`（283 行）

### 核心通信原语 `all_to_all_tensor`（151-202 行）

```python
def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,      # 待分散的维度
    gather_dim: int,       # 待聚合的维度
    group: dist.ProcessGroup | None = None,
) -> Tensor:
    """All-to-all communication for multi-dimensional tensors."""

    # 第 178 行: 沿 scatter_dim 切成 world_size 份
    chunks = list(torch.chunk(local_input, sp_world_size, dim=scatter_dim))

    # 第 181 行: 堆叠成 [world_size, ...] 张量
    stacked = torch.stack(chunks, dim=0)

    # 第 185 行: 展平为 1D（all_to_all_single 要求 1D 输入）
    stacked_flat = stacked.reshape(-1).contiguous()

    # 第 188-193 行: 执行 All-to-All（内置 autograd 支持）
    received_flat = all_to_all_single_autograd(
        stacked_flat,
        output_split_sizes=None,   # 等分模式（每个 rank 发送相同大小）
        input_split_sizes=None,
        group=group,
    )

    # 第 196 行: 恢复为多维形状
    received = received_flat.reshape(stacked_shape)

    # 第 199-200 行: 沿 gather_dim 拼接
    chunks_received = torch.unbind(received, dim=0)
    output = torch.cat(chunks_received, dim=gather_dim)

    return output.contiguous()
```

**为什么用 `all_to_all_single_autograd` 而不是 `dist.all_to_all`？**

1. **torch.compile 兼容**：`all_to_all_single_autograd` 来自 `torch.distributed._functional_collectives`，是 PyTorch 的 compile-friendly 集体通信 API
2. **内置 autograd**：自动处理反向传播的梯度通信，无需手动注册 backward hook
3. **1D 接口**：只接受 1D 张量，需要手动 reshape（第 185/196 行）

### 数据流图解

```text
gather_seq_scatter_heads: [bs, seq/4, 8heads, d] → [bs, seq, 2heads, d]
  (scatter_dim=head_dim=2, gather_dim=seq_dim=1)

4 个 rank，每个持有 seq/4 序列、8 heads：

chunk (沿 head_dim 切成 4 份)：
  Rank 0: [bs, seq/4, 2h, d] × 4 → 发给 Rank 0,1,2,3

All-to-All 后：
  Rank 0 收到: [bs, seq/4, 2h, d] × 4 (来自 Rank 0,1,2,3)

cat (沿 seq_dim 拼接)：
  Rank 0: [bs, seq, 2h, d] ← 完整序列，1/4 的 heads
```

### `_gather_seq_scatter_heads`（44-66 行）

```python
def _gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,            # 序列维度
    head_dim: int,           # 头维度
    unpadded_dim_size: int = 0,  # 原始未 padding 的序列长度
    group: ProcessGroup | None = None,
) -> Tensor:
    """All-to-All: [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]"""

    # 核心 All-to-All
    x = all_to_all_tensor(x, scatter_dim=head_dim, gather_dim=seq_dim, group=group)

    # 第 61-64 行: 移除 padding（如果原始序列不能被 cp_size 整除）
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        if padding_size > 0:
            x = _unpad_tensor(x, seq_dim, padding_size)

    return x
```

### `_gather_heads_scatter_seq`（69-88 行）

```python
def _gather_heads_scatter_seq(
    x: Tensor,
    head_dim: int,
    seq_dim: int,
    group: ProcessGroup | None = None,
) -> Tensor:
    """All-to-All: [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]"""

    # 第 83-86 行: Padding（确保序列能被 cp_size 整除）
    dim_size = x.size(seq_dim)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)

    # 核心 All-to-All
    return all_to_all_tensor(x, scatter_dim=seq_dim, gather_dim=head_dim, group=group)
```

**不对称性**：注意 `_gather_seq_scatter_heads` 在 All-to-All **之后** unpad，而 `_gather_heads_scatter_seq` 在 All-to-All **之前** pad。因为：
- 前向：序列可能被 pad 过，gather 后需要移除多余 padding
- 后向：序列可能不整除 cp_size，需要先 pad 对齐再通信

### Padding 辅助函数（116-130 行）

```python
def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    """在指定维度末尾���充零。"""
    if padding_size == 0:
        return x
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)

def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    """移除指定维度末尾的填充。"""
    if padding_size == 0:
        return x
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[tuple(slc)]
```

---

## 3.3 集成代码：`apply_cp`

### `validate_cp_constraints`（utils.py:44-91 行）

```python
def validate_cp_constraints(model_args, cp_size, tp_size=1):
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads if model_args.n_kv_heads is not None else n_heads
    n_rep = n_heads // n_kv_heads    # GQA 重复因子

    q_heads = n_heads // tp_size      # TP 后的 Q 头数
    kv_heads = n_kv_heads // tp_size  # TP 后的 KV 头数

    # 约束 1: Q 头必须能被 cp_size 整除（Q 不做 repeat）
    if q_heads % cp_size != 0:
        raise ValueError(...)

    # 约束 2: KV 头的处理分两种情况
    if kv_heads >= cp_size:
        # KV 头多于 cp_size → 必须能整除
        if kv_heads % cp_size != 0:
            raise ValueError(...)
    else:
        # KV 头少于 cp_size → cp_size 必须能被 kv_heads 整除
        if cp_size % kv_heads != 0:
            raise ValueError(...)
        # repeat 次数必须是 n_rep 的因子
        repeats = cp_size // kv_heads
        if n_rep % repeats != 0:
            raise ValueError(...)
```

**约束总结**：

| 条件 | 要求 | 原因 |
| --- | --- | --- |
| `q_heads % cp_size == 0` | Q 头数可被 CP 整除 | All-to-All 切分 Q 头 |
| `kv_heads >= cp_size` 时 | `kv_heads % cp_size == 0` | All-to-All 切分 KV 头 |
| `kv_heads < cp_size` 时 | `cp_size % kv_heads == 0` 且 `n_rep % repeats == 0` | 需要 repeat_kv 对齐 |

**示例**：

| 模型 | n_heads | n_kv_heads | tp_size | cp_size | q_heads | kv_heads | 需 repeat? | 有效? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2-7B | 28 | 4 | 1 | 4 | 28 | 4 | 否（4==4） | 28%4=0 ✓ |
| Qwen2-7B | 28 | 4 | 2 | 2 | 14 | 2 | 否（2==2） | 14%2=0 ✓ |
| Qwen2-7B | 28 | 4 | 1 | 8 | 28 | 4 | 是（×2） | 28%8≠0 ✗ |
| Qwen2-72B | 64 | 8 | 4 | 4 | 16 | 2 | 是（×2） | 16%4=0 ✓ |

### `apply_cp`（parallelize.py:330-356 行）

```python
def apply_cp(
    model: nn.Module,
    cp_group: ProcessGroup,
    tp_size: int = 1,
) -> None:
    """Apply context parallelism (Ulysses SP) to Qwen2 model."""
    cp_size = dist.get_world_size(cp_group)
    validate_cp_constraints(model.model_args, cp_size, tp_size)

    # 遍历所有 Transformer 层，设置 CP 进程组
    for transformer_block in model.layers.values():
        transformer_block.attention.set_cp_group(cp_group)
```

**调用位置**（parallelize_qwen2:119-121）：

```python
if parallel_dims.cp_enabled:
    cp_group = parallel_dims.get_group("cp")
    apply_cp(model, cp_group, tp_size=parallel_dims.tp)
```

---

# 4. 我该怎么验证自己真的懂了

## 4.1 纸面练习

### 练习 1：手画 All-to-All 数据流

给定 `cp_size=2, seq_len=8, heads=4, head_dim=64`，手动画出 `gather_seq_scatter_heads` 的数据流。

```text
预期答案:

初始状态：每个 rank 持有 seq/2=4 的序列片段和全部 4 个 heads
  Rank 0: [1, 4, 4, 64]  (bs=1, seq_chunk=4, heads=4, dim=64)
  Rank 1: [1, 4, 4, 64]

scatter_dim=head_dim=2, gather_dim=seq_dim=1

步骤 1: 沿 head_dim 切成 2 份
  Rank 0: chunk_0=[1,4,2,64], chunk_1=[1,4,2,64]
  Rank 1: chunk_0=[1,4,2,64], chunk_1=[1,4,2,64]

步骤 2: All-to-All 交换
  Rank 0 收到: [1,4,2,64] (from R0), [1,4,2,64] (from R1)
  Rank 1 收到: [1,4,2,64] (from R0), [1,4,2,64] (from R1)

步骤 3: 沿 seq_dim 拼接
  Rank 0: [1, 8, 2, 64]  ← 完整 8 tokens, 前 2 heads
  Rank 1: [1, 8, 2, 64]  ← 完整 8 tokens, 后 2 heads
```

### 练习 2：GQA repeat_kv 触发条件

给定 `n_heads=32, n_kv_heads=4, tp=2`，确定以下 cp_size 的情况：

```text
q_heads = 32/2 = 16, kv_heads = 4/2 = 2

cp_size=2:
  q_heads%2=0 ✓
  kv_heads>=cp_size? 2>=2 → 是
  kv_heads%2=0 ✓
  需要 repeat? 否
  结论: 合法，无需 repeat

cp_size=4:
  q_heads%4=0 ✓
  kv_heads>=cp_size? 2>=4 → 否
  cp_size%kv_heads=0? 4%2=0 ✓
  repeats=4/2=2
  n_rep=32/4=8, 8%2=0 ✓
  结论: 合法，需要 repeat_kv(xk, 2)

cp_size=8:
  q_heads%8=0? 16%8=0 ✓
  kv_heads>=cp_size? 2>=8 → 否
  cp_size%kv_heads=0? 8%2=0 ✓
  repeats=8/2=4
  n_rep=8, 8%4=0 ✓
  结论: 合法，需要 repeat_kv(xk, 4)
```

### 练习 3：理解输入切分 vs All-to-All 切分

```text
为什么输入要切分（ulysses_slice_inputs）？
  → 减少每个 rank 的计算量和显存占用
  → input_ids [bs, 8000] → [bs, 2000]（cp=4）
  → Linear 层（wq/wk/wv）的输入更小

为什么 All-to-All 是在 Linear 层之后、Attention 之前？
  → Linear 层（wq/wk/wv）是 point-wise 操作，不需要全序列
  → Attention 需要全序列上下文（Q 和所有 K、V 交互）
  → 所以先用小序列做 Linear，再 All-to-All 恢复全序列做 Attention
```

### 练习 4：为什么输出聚合用 All-Gather 而不是 All-to-All？

```text
Attention 输出经过 gather_heads_scatter_seq 后：
  每个 rank: [bs, seq/cp, all_heads, dim]
  wo 线性层后: [bs, seq/cp, hidden]

此时每个 rank 持有 seq/cp 序列片段的完整 hidden 表示。
要得到完整输出，只需沿序列维度拼接 → All-Gather。

如果用 All-to-All，需要先沿某维度切再拼另一维度。
但输出只需要拼序列，不需要切任何维度 → All-Gather 更直接。
```

## 4.2 运行测试

```bash
# 搜索相关测试
find tests/ -name "*ulysses*" -o -name "*cp*" -o -name "*context_parallel*" 2>/dev/null

# Archon 模型单元测试（如果存在 Ulysses 测试）
uv run pytest tests/experimental/archon/ -v -k "ulysses or cp"

# FSDP Ulysses 底层测试
uv run pytest tests/ -v -k "ulysses"
```

## 4.3 交互式验证

```python
import torch
from areal.experimental.models.archon.ulysses import _ulysses_slice_tensor

# 验证 1: 切分正确性
tensor = torch.arange(12).unsqueeze(0)  # [1, 12]
for rank in range(4):
    sliced = _ulysses_slice_tensor(tensor, cp_rank=rank, cp_size=4)
    print(f"Rank {rank}: {sliced}")
# 预期:
#   Rank 0: tensor([[ 0,  1,  2]])
#   Rank 1: tensor([[ 3,  4,  5]])
#   Rank 2: tensor([[ 6,  7,  8]])
#   Rank 3: tensor([[ 9, 10, 11]])

# 验证 2: 不对齐时抛出异常
try:
    _ulysses_slice_tensor(torch.randn(1, 10), cp_rank=0, cp_size=3)
except ValueError as e:
    print(f"✓ Caught: {e}")

# 验证 3: ulysses_slice_inputs 的 dict 安全性
from areal.experimental.models.archon.ulysses import ulysses_slice_inputs
original_inputs = {
    "input_ids": torch.randn(1, 8),
    "position_ids": torch.arange(8).unsqueeze(0),
}
original_labels = torch.randn(1, 8)
sliced_inputs, sliced_labels = ulysses_slice_inputs(
    original_inputs, original_labels, cp_rank=0, cp_size=2
)
assert sliced_inputs["input_ids"].shape == torch.Size([1, 4])
assert sliced_labels.shape == torch.Size([1, 4])
assert original_inputs["input_ids"].shape == torch.Size([1, 8])  # 原始未修改
print("✓ ulysses_slice_inputs works correctly")
```

## 4.4 常见理解误区

| 误区 | 正确理解 |
| --- | --- |
| "Ulysses 切序列维度" | Ulysses 在 Attention 内部切**头维度**（通过 All-to-All 交换序列↔头），输入切序列只是减少 Linear 计算量 |
| "All-to-All 增加了数据量" | All-to-All 只是重新分布数据，总量不变：scatter 一个维度，gather 另一个维度 |
| "GQA 模型不能用 Ulysses" | 可以，只要 KV 头数满足约束条件（通过 repeat_kv 对齐即可） |
| "ulysses_gather_output 是 All-to-All" | 不是，它是 All-Gather（每个 rank 发送自己的片段，接收所有片段并拼接） |
| "Archon 的 Ulysses 是全新实现" | 不是，底层 All-to-All 函数复用 FSDP 引擎的 `areal/models/fsdp/ulysses.py` |
| "Ring Attention 和 Ulysses 功能相同" | 通信模式完全不同：Ring 用 P2P 环形传递 KV，Ulysses 用 All-to-All 交换头↔序列 |
| "seq_len_divisor 只与 Ulysses 有关" | 它是 `tp * cp * 2`，同时服务 TP（Sequence Parallel）、CP（Ulysses）和 Varlen attention 对齐 |

---

# 5. 附录

## 5.1 Ulysses 完整数据流时序

```text
时间轴 →

           Rank 0                              Rank 1
           ──────                              ──────
 输入切分   input_ids[0:L/2]                    input_ids[L/2:L]
     ↓      position_ids[0:L/2]                 position_ids[L/2:L]
     ↓
 Embedding  x: [bs, L/2, hidden]               x: [bs, L/2, hidden]
     ↓
 RMSNorm    x: [bs, L/2, hidden]               x: [bs, L/2, hidden]
     ↓
 wq/wk/wv   xq: [bs, L/2, H, d]                xq: [bs, L/2, H, d]
             xk: [bs, L/2, Hkv, d]              xk: [bs, L/2, Hkv, d]
             xv: [bs, L/2, Hkv, d]              xv: [bs, L/2, Hkv, d]
     ↓
 RoPE       xq, xk = apply_rotary_emb(...)     xq, xk = apply_rotary_emb(...)
     ↓
 repeat_kv  (if kv_heads < cp_size)             (if kv_heads < cp_size)
     ↓      xk: [bs, L/2, Hkv, d] → [bs, L/2, H, d]
            xv: [bs, L/2, Hkv, d] → [bs, L/2, H, d]
     ↓
 ┌──── All-to-All (gather_seq_scatter_heads) ────────────────────────┐
 │  scatter heads, gather seq                                        │
 │  xq: [bs, L/2, H, d] → [bs, L, H/2, d]                         │
 │  xk: [bs, L/2, H, d] → [bs, L, H/2, d]  (after repeat_kv)      │
 │  xv: [bs, L/2, H, d] → [bs, L, H/2, d]  (after repeat_kv)      │
 └───────────────────────────────────────────────────────────────────┘
     ↓
 Attention   flash_attn(xq, xk, xv)            flash_attn(xq, xk, xv)
             output: [bs, L, H/2, d]           output: [bs, L, H/2, d]
     ↓
 ┌──── All-to-All (gather_heads_scatter_seq) ────────────────────────┐
 │  scatter seq, gather heads                                        │
 │  output: [bs, L, H/2, d] → [bs, L/2, H, d]                     │
 └───────────────────────────────────────────────────────────────────┘
     ↓
 wo          output: [bs, L/2, hidden]          output: [bs, L/2, hidden]
     ↓
 ... (FFN, 下一层) ...
     ↓
 Output      logprobs: [bs, L/2, vocab]         logprobs: [bs, L/2, vocab]
            entropy:  [bs, L/2, vocab]         entropy:  [bs, L/2, vocab]
            vocab_min/max: [bs, L/2, vocab]    vocab_min/max: [bs, L/2, vocab]
     ↓
 ┌──── All-Gather (ulysses_gather_output) ───────────────────────────┐
 │  logprobs:  [bs, L/2, vocab] → [bs, L, vocab]                    │
 │  entropy:   [bs, L/2, vocab] → [bs, L, vocab]                    │
 │  vocab_min: [bs, L/2, vocab] → [bs, L, vocab]                    │
 │  vocab_max: [bs, L/2, vocab] → [bs, L, vocab]                    │
 │  (参考模型/Critic 推理时同理聚合 result/values)                    │
 └───────────────────────────────────────────────────────────────────┘
```

## 5.2 文件依赖关系

```text
areal/experimental/models/archon/ulysses.py (Archon 封装层, 81 行)
  │
  ├── 复用底层实现 ──────────────────────┐
  │                                       │
  │                                       ▼
  │                            areal/models/fsdp/ulysses.py (底层实现, 283 行)
  │                              ├── all_to_all_tensor()
  │                              ├── _gather_seq_scatter_heads()
  │                              ├── _gather_heads_scatter_seq()
  │                              ├── _pad_tensor() / _unpad_tensor()
  │                              └── all_to_all_single_autograd (PyTorch 原生)
  │
  ├── 被 Engine 调用 ───────────────────┐
  │                                      │
  │                                      ▼
  │                            archon_engine.py
  │                              ├── ulysses_slice_inputs()   (前向前)
  │                              └── ulysses_gather_output()  (前向后)
  │
  ├── 被 Attention 模块使用 ────────────┐
  │                                      │
  │                                      ▼
  │                            qwen2/model/model.py  (qwen3 同理)
  │                              ├── set_cp_group()            (初始化)
  │                              ├── gather_seq_scatter_heads() (Attn 前)
  │                              └── gather_heads_scatter_seq() (Attn 后)
  │
  └── 由 parallelize 注入 ─────────────┐
                                        │
                                        ▼
                               qwen2/infra/parallelize.py
                                 ├── validate_cp_constraints()
                                 └── apply_cp() → set_cp_group()
```

## 5.3 验证清单

- [ ] 理解 All-to-All 与 All-Gather 的区别（能画出两者的数据流图）
- [ ] 理解 `gather_seq_scatter_heads` 的维度变换（给定输入 shape 能算出输出 shape）
- [ ] 理解 GQA 场景下 `repeat_kv` 的触发条件和约束验证
- [ ] 理解为什么 Ulysses 与 TP 自然兼容（TP 切 hidden，Ulysses 切 heads）
- [ ] 理解 padding/unpadding 的时机和不对称性（前向 unpad 在后，后向 pad 在前）
- [ ] 理解 `seq_len_divisor = tp * cp * 2` 的三重含义
- [ ] 理解 Archon 封装层 vs FSDP 底层实现的设计关系（显式 group vs 全局变量）
