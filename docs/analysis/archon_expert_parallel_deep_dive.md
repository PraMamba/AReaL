# Archon Expert Parallel 深度解析

> 源文件：`areal/experimental/models/archon/expert_parallel.py`（513 行）
> 核心类：`BaseExpertParallel` · `ExpertParallel` · `ExpertTensorParallel` · `TensorParallel` · `ReordererSequenceParallel`

---

[TOC]

---

# 1. 白话解释

## 1.1 一句话总结

Expert Parallel 通过 **All-to-All 通信在专家维度切分 MoE 模型**，让每个 GPU 只持有部分专家的权重，从而支持超大规模 MoE 训练——不同于数据并行复制所有专家，EP 实现了专家级别的模型并行。

## 1.2 现实类比

```text
想象一个医院有 64 位专家医生，每天接诊 1000 位病人：

方案 A：数据并行（每个诊室配齐 64 位专家）
  → 4 个诊室，每个诊室都有全部 64 位专家的副本
  → 每个诊室接诊 250 位病人
  → 问题：专家太多，诊室放不下（= GPU 显存不够）

方案 B：Expert Parallel（专家分散到不同诊室）
  ┌─────────────────────────────────────────────────────┐
  │ 4 个诊室，每个诊室只有 16 位专家                     │
  │   诊室 0: 专家 0-15                                  │
  │   诊室 1: 专家 16-31                                 │
  │   诊室 2: 专家 32-47                                 │
  │   诊室 3: 专家 48-63                                 │
  │                                                     │
  │ 病人流程：                                           │
  │   1. 分诊台（Router）决定每个病人看哪 2 位专家       │
  │   2. All-to-All 转运：把病人送到对应诊室             │
  │      - 诊室 0 收到：需要看专家 0-15 的所有病人       │
  │      - 诊室 1 收到：需要看专家 16-31 的所有病人      │
  │   3. 各诊室内部：专家给病人看病                      │
  │   4. All-to-All 转运回：把病人送回原来的位置         │
  └─────────────────────────────────────────────────────┘

方案 C：Expert Parallel + Tensor Parallel（专家 + 助手分工）
  ┌─────────────────────────────────────────────────────┐
  │ 每位专家还配 2 位助手（TP），分担诊断工作            │
  │   诊室 0: 专家 0-15，每位专家有 2 位助手             │
  │   → 专家权重在两个维度上切分：                       │
  │      - 维度 1（专家维度）：切成 4 份（EP）           │
  │      - 维度 2（助手维度）：切成 2 份（TP）           │
  │   → 2D 切分：[Shard(0), Shard(1)]                    │
  └─────────────────────────────────────────────────────┘
```

## 1.3 这个文件做了什么

```text
expert_parallel.py (513 行)
  │
  ├── BaseExpertParallel (27-66)        ← 抽象基类
  │     定义三个核心接口：
  │     - _partition_fn: 如何切分专家权重
  │     - _token_dispatch: 如何分发 token 到专家
  │     - _token_combine: 如何聚合专家输出
  │
  ├── ExpertParallel (68-237)           ← EP 实现（etp=1）
  │     ├── _partition_fn: Shard(0) 按专家维度切分
  │     ├── _token_dispatch: All-to-All 分发 token
  │     │     1. 交换 token 数量（all_to_all_single）
  │     │     2. 变长 All-to-All 分发 token
  │     │     3. _permute 对齐 grouped_mm
  │     ├── _token_combine: All-to-All 聚合输出
  │     │     1. _unpermute 恢复顺序
  │     │     2. 反向 All-to-All
  │     └── _apply: 通过 distribute_module 注册 hooks
  │
  ├── apply_expert_parallel (240-259)   ← 便捷函数
  │     快速应用 EP 到 GroupedExperts 模块
  │
  ├── TensorParallel (262-328)          ← TP-only（EP 禁用时）
  │     ├── _partition_fn:
  │     │     w1/w3: Shard(1) 列切分
  │     │     w2: Shard(2) 行切分
  │     ├── _prepare_input_fn: Replicate → Partial
  │     └── _apply: 注册 partition_fn + input_fn
  │
  ├── ExpertTensorParallel (331-433)    ← EP + TP 组合（etp=tp）
  │     继承 ExpertParallel，扩展为 2D 切分
  │     ├── _partition_fn:
  │     │     w1: [Shard(0), Shard(1)]
  │     │     w2: [Shard(0), Shard(2)]
  │     │     w3: [Shard(0), Shard(1)]
  │     ├── _token_dispatch: TP 输入准备 + EP All-to-All
  │     └── _token_combine: EP All-to-All（使用 ep mesh）
  │
  └── ReordererSequenceParallel (436-503) ← TokenReorderer 的序列并行
        用于 etp=1 场景（TP 被 EP 借用）
        ├── _prepare_input_fn: 按 TP rank 切分 token
        ├── _prepare_output_fn: 调整 token 索引到全局
        └── _apply: 注册 input_fn + output_fn
```

## 1.4 核心不变量

```
1. 所有专家恰好分配到一个 EP rank，不重复、不遗漏
2. All-to-All 前后数据总量不变（只是重新分布）
3. Token dispatch 和 combine 是互逆操作
4. 权重切分维度与通信模式严格对应
```

---

# 2. 前置概念

## 2.1 MoE（Mixture of Experts）架构

MoE 模型用多个"专家"FFN 替代标准 Transformer 的单个 FFN：

```text
标准 Transformer 层:
  x → Attention → FFN → output
                   ↑
              单个 FFN

MoE Transformer 层:
  x → Attention → MoE → output
                   ↑
              ┌────┴────┐
              │ Router  │ ← 决定每个 token 去哪些专家
              └────┬────┘
         ┌─────────┼─────────┐
      Expert0  Expert1  ...  Expert63
         ↑         ↑           ↑
    每个 token 只激活 top_k 个专家（如 k=2）
```

**关键特性**：
- **稀疏激活**：每个 token 只计算 top_k 个专家，不是全部
- **参数扩展**：64 个专家 = 64 倍参数，但计算量只增加 k 倍
- **负载均衡**：需要确保专家负载均匀，避免某些专家过载

## 2.2 为什么需要 Expert Parallel

**问题**：单 GPU 放不下所有专家的权重

```text
示例：Qwen3-MoE 模型
  - 64 个专家，每个专家 ~1GB 权重
  - 总专家权重：64GB
  - 单张 A100 (80GB)：勉强能放下
  - 单张 H100 (80GB)：勉强能放下
  - 但加上激活值、梯度、优化器状态 → 放不下！

解决方案：Expert Parallel
  - 4 个 GPU，每个 GPU 持有 16 个专家
  - 每个 GPU 专家权重：16GB
  - 通过 All-to-All 通信交换 token
```

## 2.3 All-to-All 通信模式

All-to-All 是 EP 的核心通信原语：

```text
示例：4 个 rank，每个 rank 有不同数量的 token 要发送

发送前（每个 rank 持有需要发送给所有 rank 的 token）:
  Rank 0: [T0→R0, T0→R1, T0→R2, T0→R3]  (发送 [10, 5, 8, 3] 个 token)
  Rank 1: [T1→R0, T1→R1, T1→R2, T1→R3]  (发送 [7, 12, 4, 9] 个 token)
  Rank 2: [T2→R0, T2→R1, T2→R2, T2→R3]  (发送 [6, 8, 11, 5] 个 token)
  Rank 3: [T3→R0, T3→R1, T3→R2, T3→R3]  (发送 [9, 3, 7, 10] 个 token)

All-to-All 后（每个 rank 收到所有 rank 发来的 token）:
  Rank 0: [T0→R0, T1→R0, T2→R0, T3→R0]  (收到 10+7+6+9 = 32 个 token)
  Rank 1: [T0→R1, T1→R1, T2→R1, T3→R1]  (收到 5+12+8+3 = 28 个 token)
  Rank 2: [T0→R2, T1→R2, T2→R2, T3→R2]  (收到 8+4+11+7 = 30 个 token)
  Rank 3: [T0→R3, T1→R3, T2→R3, T3→R3]  (收到 3+9+5+10 = 27 个 token)

关键特性：
  - 变长通信：每个 rank 发送/接收的数量可以不同
  - 数据总量守恒：总发送量 = 总接收量
  - 双向操作：dispatch 和 combine 是互逆的
```

## 2.4 distribute_module 与 Hook 机制

PyTorch 的 `distribute_module` 允许在模块前后自动插入通信逻辑：

```python
distribute_module(
    module,
    device_mesh,
    partition_fn=...,    # 如何切分权重
    input_fn=...,        # forward 前的 pre-hook（token dispatch）
    output_fn=...,       # forward 后的 hook（token combine）
)
```

**执行流程**：
```text
1. partition_fn 执行：切分权重到各 GPU
2. 用户调用 module.forward(x)
3. input_fn 自动执行：All-to-All dispatch token
4. 本地专家计算
5. output_fn 自动执行：All-to-All combine 输出
6. 返回结果
```

**优势**：用户无需手动调用通信，forward 自动触发。

## 2.5 DTensor 与 Shard Placement

`DTensor` 是 PyTorch 的分布式张量抽象：

```python
# 1D 切分（EP-only）
w1 = distribute_tensor(w1, ep_mesh, [Shard(0)])
# w1 shape: (num_experts, hidden_dim, dim)
# Shard(0) → 在 expert 维度（dim 0）切分

# 2D 切分（EP + TP）
w1 = distribute_tensor(w1, ep_tp_mesh, [Shard(0), Shard(1)])
# Shard(0) → 在 expert 维度切分（EP）
# Shard(1) → 在 hidden_dim 维度切分（TP）
```

**Placement 类型**：
- `Shard(dim)`: 在指定维度切分
- `Replicate()`: 所有 rank 持有完整副本
- `Partial()`: 部分结果，需要 reduce 聚合

## 2.6 _permute / _unpermute 对齐机制

`torch._grouped_mm` 要求 token 数量对齐到特定倍数（8/16/32）：

```text
问题：专家收到的 token 数量不规则
  Expert 0: 13 tokens
  Expert 1: 7 tokens
  Expert 2: 19 tokens
  Expert 3: 5 tokens

grouped_mm 要求：每个专家的 token 数必须是 8 的倍数

解决：_permute 添加 padding
  1. 计算需要的 padding：
     Expert 0: 13 → 16 (pad 3)
     Expert 1: 7 → 8 (pad 1)
     Expert 2: 19 → 24 (pad 5)
     Expert 3: 5 → 8 (pad 3)
  2. 添加 padding row（全零）
  3. 生成 permuted_indices 记录重排序
  4. 计算后用 _unpermute 移除 padding
```

## 2.7 EP 策略选择矩阵

| EP | TP | etp | 策略类 | 专家权重切分 | 通信模式 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | - | None | Replicate | 无 |
| 1 | >1 | - | `TensorParallel` | `[Shard(1/2)]` | All-reduce（TP） |
| >1 | 1 | - | `ExpertParallel` | `[Shard(0)]` | All-to-All（EP） |
| >1 | >1 | 1 | `ExpertParallel` | `[Shard(0)]` | All-to-All（EP），TP 被借用 |
| >1 | >1 | tp | `ExpertTensorParallel` | `[Shard(0), Shard(1/2)]` | All-to-All（EP）+ All-reduce（TP） |

**关键区别**：
- **etp=1**：TP 维度被 EP "借用"，专家只用 EP 切分，TP rank 处理不同 token
- **etp=tp**：TP 维度独立，专家用 2D 切分（EP + TP），TP rank 持有同一专家的不同切片

## 2.8 ReordererSequenceParallel 的特殊作用

当 `etp=1` 时，TP 维度被 EP 借用，需要 `ReordererSequenceParallel` 确保：

```text
问题：如果不做序列并行
  - 所有 TP rank 都对全部 token 做 reorder
  - 生成相同的 token_indices_sorted
  - EP All-to-All 会发送重复数据！

解决：ReordererSequenceParallel
  1. 输入切分：每个 TP rank 只处理 1/tp 的 token
     Rank 0: tokens[0:500]
     Rank 1: tokens[500:1000]
  2. 各自 reorder，生成局部索引
  3. 输出调整：局部索引 → 全局索引
     Rank 0 的索引 + 0
     Rank 1 的索引 + 500
  4. EP All-to-All 不会重复发送
```

---

# 3. 源码逐行地图

## 3.1 导入与依赖（1-25 行）

```python
# 第 1 行: 源自 TorchTitan 项目
# Adapted from torchtitan: torchtitan/distributed/expert_parallel.py

# 第 3-5 行: 标准库
from __future__ import annotations
from abc import ABC, abstractmethod

# 第 7-8 行: PyTorch 核心
import torch
from torch import nn

# 第 9-12 行: 分布式集体通信
from torch.distributed._functional_collectives import (
    all_to_all_single,         # 不含 autograd 的 all-to-all（用于 token 数量交换）
    all_to_all_single_autograd, # 含 autograd 的 all-to-all（用于 token 数据交换）
)
# 区别：
#   all_to_all_single: 用于 metadata（token count），不需要梯度
#   all_to_all_single_autograd: 用于 tensor 数据，需要反向传播梯度

# 第 13-21 行: DTensor 相关
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    DTensor,        # 分布式张量抽象
    Partial,        # 梯度累加放置策略
    Replicate,      # 复制放置策略
    Shard,          # 切分放置策略
    distribute_module,  # 自动注册 hook 的模块分发
    distribute_tensor,  # 将普通张量转为 DTensor
)
from torch.distributed.tensor.parallel.style import ParallelStyle
# ParallelStyle 是 TP API 的基类，提供 _apply 接口

# 第 24 行: MoE 工具函数
from areal.experimental.models.archon.moe.utils import _permute, _unpermute
# _permute: 重排 token 并添加 grouped_mm 对齐 padding
# _unpermute: 恢复 token 顺序并移除 padding
```

**设计要点**：文件从 `torch.distributed._functional_collectives` 导入 `all_to_all_single_autograd`，这是 `torch.compile` 兼容的集体通信 API，与 Ulysses 的 `all_to_all_single_autograd` 是同一个函数。

## 3.2 BaseExpertParallel 抽象基类（27-66 行）

```python
class BaseExpertParallel(ParallelStyle, ABC):
    """Abstract base class for Expert Parallelism styles."""

    @abstractmethod
    def _partition_fn(
        self,
        name: str,           # 模块名称（未使用，但 distribute_module 要求此签名）
        module: nn.Module,   # 待切分的 GroupedExperts 模块
        device_mesh: DeviceMesh,  # 设备网格
    ) -> None:
        """Partition expert weights across devices."""

    @abstractmethod
    def _token_dispatch(
        self,
        module: nn.Module,   # GroupedExperts 模块
        inputs: tuple[torch.Tensor, torch.Tensor],
        # inputs = (routed_input, num_tokens_per_expert)
        # routed_input: [total_tokens, dim] 按专家排序的 token
        # num_tokens_per_expert: [num_experts] 每个专家的 token 数
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to devices holding their assigned experts."""

    @abstractmethod
    def _token_combine(
        self,
        module: nn.Module,
        output: torch.Tensor,    # 专家计算输出
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        """Combine expert outputs back to original token locations."""
```

**设计哲学**：三个抽象方法恰好对应 `distribute_module` 的三个回调参数：
- `_partition_fn` → `partition_fn`（权重切分）
- `_token_dispatch` → `input_fn`（前向 pre-hook）
- `_token_combine` → `output_fn`（前向 hook）

## 3.3 ExpertParallel 类（68-237 行）

### 3.3.1 初始化（79-85 行）

```python
class ExpertParallel(BaseExpertParallel):
    def __init__(self) -> None:
        super().__init__()
        # 状态变量：dispatch 时保存，combine 时使用
        self.input_splits: list[int] | None = None    # 发送给各 rank 的 token 数
        self.output_splits: list[int] | None = None   # 从各 rank 接收的 token 数
        self.input_shape: tuple[int, ...] | None = None  # permute 前的形状
        self.permuted_indices: torch.Tensor | None = None  # permute 索引
```

**为什么需要保存状态？** `_token_dispatch` 和 `_token_combine` 是分开调用的（分别作为 pre-hook 和 hook），但 combine 需要 dispatch 时计算的 `input_splits` 和 `output_splits` 来执行反向 All-to-All。

### 3.3.2 _partition_fn（87-102 行）

```python
    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        # 遍历所有参数（w1, w2, w3），在 expert 维度（dim 0）切分
        for param_name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
            module.register_parameter(param_name, dist_param)
```

**图解**：
```text
原始权重（64 个专家，4 个 EP rank）：
  w1: [64, hidden_dim, dim]

Shard(0) 切分后：
  Rank 0: w1_local = [16, hidden_dim, dim]  ← 专家 0-15
  Rank 1: w1_local = [16, hidden_dim, dim]  ← 专家 16-31
  Rank 2: w1_local = [16, hidden_dim, dim]  ← 专家 32-47
  Rank 3: w1_local = [16, hidden_dim, dim]  ← 专家 48-63

w2, w3 同理。
```

### 3.3.3 _token_dispatch（104-181 行）——核心通信逻辑

这是整个文件最复杂的函数，分三个步骤：

**步骤 1：交换 token 数量（128-158 行）**

```python
        routed_input, num_tokens_per_expert = inputs
        group = device_mesh.get_group()
        ep_degree = device_mesh.size()
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree
        # 例：64 experts / 4 ranks = 16 local experts per rank

        # 第 128-140 行: 无梯度的 All-to-All 交换每个专家的 token 数量
        with torch.no_grad():
            num_tokens_per_expert_received = all_to_all_single(
                num_tokens_per_expert,
                None,   # 等分模式
                None,
                group=group,
            )
            # 必须显式等待，因为下游操作无法识别 AsyncCollectiveTensor 需要解包
            # （torch.compile 兼容性要求）
            num_tokens_per_expert_received = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_received
            )
```

**图解（步骤 1）**：
```text
4 个 rank，每个 rank 有 64 个专家的 token 计数

发送前（每个 rank 知道所有 64 个专家的 token 数量）：
  Rank 0: [3,5,2,7, | 4,1,6,3, | 8,2,5,1, | 3,6,4,2]
           ↑ rank 0    rank 1     rank 2     rank 3
           的专家      的专家     的专家     的专家

All-to-All 后（每个 rank 只知道自己专家的 token 数量，但来自所有 rank）：
  Rank 0: [3,5,2,7, | 2,3,5,1, | 4,2,7,3, | 6,1,4,5]
           ↑ from R0   from R1    from R2    from R3

含义：Rank 0 的 4 个专家分别从 R0/R1/R2/R3 收到多少 token
```

**步骤 1 续：计算变长 All-to-All 的 splits（142-158 行）**

```python
            # 将 [64] 的计数 reshape 成 [4 ranks, 16 local_experts]
            counts_view = num_tokens_per_expert.view(ep_degree, num_local_experts)
            received_view = num_tokens_per_expert_received.view(
                ep_degree, num_local_experts
            )

            # input_splits: 发送给各 rank 的总 token 数
            # output_splits: 从各 rank 接收的总 token 数
            self.input_splits = counts_view.sum(dim=1).to(
                torch.device("cpu"), non_blocking=True
            )
            self.output_splits = received_view.sum(dim=1).to(
                torch.device("cpu"), non_blocking=False
            )
            self.input_splits = self.input_splits.tolist()
            self.output_splits = self.output_splits.tolist()
```

**为什么 `non_blocking=True` 和 `non_blocking=False`？** `input_splits` 先计算，用 `non_blocking=True` 异步拷贝到 CPU。`output_splits` 最后计算，用 `non_blocking=False` 同步等待（因为 `.tolist()` 需要 CPU 数据）。

**步骤 2：变长 All-to-All 分发 token（160-166 行）**

```python
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,   # 各 rank 接收的数量
            self.input_splits,    # 各 rank 发送的数量
            group,
        )
```

**注意**：`output_splits` 和 `input_splits` 的位置是**相对于当前操作**的。`output_splits` 是我要接收的量（输出），`input_splits` 是我要发送的量（输入）。

**步骤 3：Permute 对齐 grouped_mm（168-181 行）**

```python
        (
            self.input_shape,        # 原始形状（含 padding row）
            routed_input,            # permuted + padded 的 token
            self.permuted_indices,   # permute 索引（用于 unpermute）
            aligned_num_tokens,      # 对齐后的每专家 token 数
        ) = _permute(
            routed_input,
            num_tokens_per_expert_received,
            ep_degree,
            num_local_experts,
        )

        return routed_input, aligned_num_tokens
```

### 3.3.4 _token_combine（183-212 行）

```python
    def _token_combine(
        self,
        module: nn.Module,
        output: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        group = device_mesh.get_group()

        # 步骤 1: 恢复 token 顺序，移除 padding
        output = _unpermute(output, self.input_shape, self.permuted_indices)

        # 步骤 2: 反向 All-to-All（注意 splits 互换！）
        output = all_to_all_single_autograd(
            output,
            self.input_splits,   # dispatch 时的 input_splits 变成 combine 的 output
            self.output_splits,  # dispatch 时的 output_splits 变成 combine 的 input
            group,
        )

        return output
```

**splits 互换的直觉**：
```text
dispatch: Rank 0 发送 [10, 5, 8, 3] 个 token 给 R0/R1/R2/R3
          Rank 0 接收 [10, 7, 6, 9] 个 token 从 R0/R1/R2/R3

combine:  Rank 0 发送 [10, 7, 6, 9] 个 token 给 R0/R1/R2/R3  ← 原来的 output
          Rank 0 接收 [10, 5, 8, 3] 个 token 从 R0/R1/R2/R3  ← 原来的 input
```

### 3.3.5 _apply（214-237 行）

```python
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,     # 切分 w1/w2/w3 到 Shard(0)
            input_fn=self._token_dispatch,       # 注册为 forward pre-hook
            output_fn=self._token_combine,       # 注册为 forward hook
        )
```

## 3.4 apply_expert_parallel 便捷函数（240-259 行）

```python
def apply_expert_parallel(
    experts_module: nn.Module,
    ep_mesh: DeviceMesh,
) -> None:
    ep_style = ExpertParallel()
    ep_style._apply(experts_module, ep_mesh)
```

**用途**：简化 EP 应用为一行调用。

## 3.5 TensorParallel 类（262-328 行）——EP 禁用时的 TP

```python
class TensorParallel(ParallelStyle):
    """Tensor Parallelism for experts when EP is disabled."""
```

**注意**：这个 `TensorParallel` 类**不继承** `BaseExpertParallel`，它直接继承 `ParallelStyle`。因为 TP-only 场景不需要 All-to-All token dispatch/combine。

### 3.5.1 _prepare_input_fn（278-295 行）

```python
    def _prepare_input_fn(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routed_input, num_tokens_per_expert = inputs
        # 前向：输入是 Replicate 的（每个 TP rank 持有相同的 token）
        # 反向：梯度用 Partial 累加（需要 all-reduce）
        routed_input = DTensor.from_local(
            routed_input, device_mesh, (Replicate(),)
        ).to_local(grad_placements=(Partial(),))
        return routed_input, num_tokens_per_expert
```

**为什么 `Replicate()` + `Partial()` 组合？**
```text
前向传播：
  所有 TP rank 持有相同的 routed_input
  w1 被 Shard(1) 切分 → 每个 rank 计算部分输出
  w2 被 Shard(2) 切分 → 行切分，输出需要 All-reduce

反向传播：
  Partial() 告诉 autograd：
  "梯度是部分结果，需要通过 all-reduce 聚合"
  → 自动生成正确的梯度通信
```

### 3.5.2 _partition_fn（297-319 行）

```python
    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        # w1: (num_experts, hidden_dim, dim) → Shard(1) 列切分
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)]))
        )
        # w2: (num_experts, dim, hidden_dim) → Shard(2) 行切分
        module.register_parameter(
            "w2", nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)]))
        )
        # w3: (num_experts, hidden_dim, dim) → Shard(1) 列切分
        module.register_parameter(
            "w3", nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)]))
        )
```

**为什么 w1/w3 切 dim 1，w2 切 dim 2？**

```text
SwiGLU 计算: silu(x @ w1.T) * (x @ w3.T) @ w2.T

权重 shape:
  w1: [E, hidden, dim]    → Shard(1) 切 hidden → 列切分（ColwiseParallel）
  w3: [E, hidden, dim]    → Shard(1) 切 hidden → 列切分
  w2: [E, dim, hidden]    → Shard(2) 切 hidden → 行切分（RowwiseParallel）

等价于标准 TP 的 ColwiseParallel/RowwiseParallel：
  x @ w1.T → [tokens, dim] @ [dim, hidden/tp] = [tokens, hidden/tp]  ✓ 无需通信
  ... @ w2.T → [tokens, hidden/tp] @ [hidden/tp, dim] = [tokens, dim]  需 all-reduce
```

### 3.5.3 _apply（321-328 行）

```python
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._prepare_input_fn,
            # 注意：没有 output_fn！
            # TP 的 all-reduce 由 DTensor 自动处理
        )
```

**为什么不需要 `output_fn`？** TP 的梯度 all-reduce 由 `Partial()` placement 自动触发，不需要显式 hook。而 EP 的 `output_fn` 执行的是 All-to-All（结构上不同于 all-reduce），必须手动注册。

## 3.6 ExpertTensorParallel 类（331-433 行）——EP + TP 组合

```python
class ExpertTensorParallel(ExpertParallel):
    """EP + TP: 继承 ExpertParallel，扩展为 2D 切分。"""
```

**继承关系**：`ExpertTensorParallel → ExpertParallel → BaseExpertParallel → ParallelStyle`

### 3.6.1 _token_dispatch 覆写（351-372 行）

```python
    def _token_dispatch(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,  # 这里是 2D mesh: [ep, tp]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routed_input, num_tokens_per_expert = inputs

        # 第 365-367 行: TP 输入准备（与 TensorParallel._prepare_input_fn 相同）
        routed_input = DTensor.from_local(
            routed_input, device_mesh["tp"], (Replicate(),)
        ).to_local(grad_placements=(Partial(),))

        # 第 370-372 行: 调用父类的 EP dispatch（使用 ep 子网格）
        return super()._token_dispatch(
            module, (routed_input, num_tokens_per_expert), device_mesh["ep"]
        )
```

**关键设计**：`device_mesh` 是 2D `[ep, tp]`，但 EP 通信只需要 `device_mesh["ep"]`（1D ep 子网格），TP 输入准备使用 `device_mesh["tp"]`（1D tp 子网格）。

### 3.6.2 _partition_fn 覆写（374-406 行）

```python
    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,  # 2D mesh: [ep, tp]
    ) -> None:
        # 2D 切分：第一维是 EP（Shard(0)），第二维是 TP（Shard(1) 或 Shard(2)）
        module.register_parameter(
            "w1",
            nn.Parameter(
                distribute_tensor(module.w1, device_mesh, [Shard(0), Shard(1)])
            ),
        )
        module.register_parameter(
            "w2",
            nn.Parameter(
                distribute_tensor(module.w2, device_mesh, [Shard(0), Shard(2)])
            ),
        )
        module.register_parameter(
            "w3",
            nn.Parameter(
                distribute_tensor(module.w3, device_mesh, [Shard(0), Shard(1)])
            ),
        )
```

**2D 切分图解**：
```text
w1 原始: [64, hidden_dim, dim]

2D mesh: ep=4, tp=2 → 8 个 rank
  [Shard(0), Shard(1)] =
    EP 维度切 dim 0: 64/4 = 16 experts per EP rank
    TP 维度切 dim 1: hidden_dim/2 per TP rank

结果（Rank 0, ep_rank=0, tp_rank=0）:
  w1_local: [16, hidden_dim/2, dim]

结果（Rank 1, ep_rank=0, tp_rank=1）:
  w1_local: [16, hidden_dim/2, dim]  ← 相同专家的另一半
```

### 3.6.3 _token_combine 和 _apply 覆写（408-433 行）

```python
    def _token_combine(
        self,
        module: nn.Module,
        output: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        # 使用 EP 子网格进行反向 All-to-All
        return super()._token_combine(module, output, device_mesh["ep"])

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # device_mesh 是 2D [ep, tp]
        return distribute_module(
            module,
            device_mesh,          # 传入 2D mesh
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )
```

**与 `ExpertParallel._apply` 的区别**：`ExpertParallel` 传入 1D ep mesh，`ExpertTensorParallel` 传入 2D ep_tp mesh。

## 3.7 ReordererSequenceParallel 类（436-503 行）

```python
class ReordererSequenceParallel(ParallelStyle):
    """Sequence Parallel for TokenReorderer when etp=1."""
```

**使用场景**：仅在 `EP>1` 且 `etp=1` 时使用。此时 TP 维度被 EP 借用，需要确保各 TP rank 处理不同的 token。

### 3.7.1 _prepare_input_fn（447-471 行）

```python
    def _prepare_input_fn(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_scores, selected_indices = inputs
        num_tokens = top_scores.shape[0]
        tp_size = device_mesh.size()
        tp_rank = device_mesh.get_local_rank()

        # 验证 token 数能被 TP 整除
        if num_tokens % tp_size != 0:
            raise ValueError(...)

        # 按 TP rank 切分 token
        local_num_tokens = num_tokens // tp_size
        offset = tp_rank * local_num_tokens

        return (
            top_scores[offset : offset + local_num_tokens],
            selected_indices[offset : offset + local_num_tokens],
        )
```

**图解**：
```text
1000 个 token, tp=2

Rank 0（tp_rank=0）：处理 token[0:500]
  top_scores[0:500], selected_indices[0:500]

Rank 1（tp_rank=1）：处理 token[500:1000]
  top_scores[500:1000], selected_indices[500:1000]
```

### 3.7.2 _prepare_output_fn（473-494 行）

```python
    def _prepare_output_fn(
        self,
        module: nn.Module,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        top_scores_sorted, token_indices_sorted, num_tokens_per_expert = outputs
        tp_rank = device_mesh.get_local_rank()

        # 局部索引 → 全局索引
        # token_indices_sorted 是在局部空间（num_tokens/tp * top_k）中的索引
        # 需要加上偏移量变成全局索引
        token_indices_global = (
            token_indices_sorted + top_scores_sorted.shape[0] * tp_rank
        )

        return top_scores_sorted, token_indices_global, num_tokens_per_expert
```

**图解（全局索引调整）**：
```text
1000 tokens, tp=2, top_k=2
每个 TP rank 处理 500 tokens → 1000 个 (token, expert) 对

Rank 0 的 reorderer 输出:
  token_indices_sorted: [0, 4, 2, 8, ...]  ← 局部索引（0-999 范围）
  加上偏移 0 → [0, 4, 2, 8, ...]           ← 全局索引不变

Rank 1 的 reorderer 输出:
  token_indices_sorted: [0, 3, 7, 1, ...]  ← 局部索引（0-999 范围）
  加上偏移 1000 → [1000, 1003, 1007, 1001, ...]  ← 全局索引

这样 MoE 的 gather/scatter 使用全局索引时，
Rank 0 和 Rank 1 填充不同的位置，不会重叠。
```

### 3.7.3 _apply（496-503 行）

```python
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            input_fn=self._prepare_input_fn,
            output_fn=self._prepare_output_fn,
            # 注意：没有 partition_fn！TokenReorderer 没有参数需要切分
        )
```

## 3.8 集成代码：apply_moe_ep_tp（parallelize.py:632-747 行）

```python
def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp: int = 1,
    ep_tp_mesh: DeviceMesh | None = None,
) -> None:

    # 对每个 MoE 层:
    for transformer_block in model.layers.values():
        if not getattr(transformer_block, "moe_enabled", False):
            continue
        moe = transformer_block.moe

        # 1. TP 相关处理（input/output DTensor 转换 + router gate 复制）
        if tp_mesh is not None:
            moe_tp_plan = {
                "moe": PrepareModuleInputOutput(...),
                "moe.router.gate": ReplicateParallel(),
            }
            # etp=1 时添加 ReordererSequenceParallel
            if ep_mesh is not None and etp == 1:
                moe_tp_plan["moe.reorderer"] = ReordererSequenceParallel()
            parallelize_module(transformer_block, tp_mesh, moe_tp_plan)

        # 2. 策略选择
        if ep_mesh is None:
            experts_mesh = tp_mesh
            experts_plan = TensorParallel()                # TP-only
        elif tp_mesh is None or etp == 1:
            experts_mesh = ep_mesh
            experts_plan = ExpertParallel()                # EP-only (或 TP 被借用)
        else:
            experts_mesh = ep_tp_mesh
            experts_plan = ExpertTensorParallel()           # EP + TP

        # 3. 应用到 experts 模块
        if experts_mesh is not None:
            parallelize_module(moe.experts, experts_mesh, experts_plan)
```

---

# 4. 我该怎么验证自己真的懂了

## 4.1 纸面练习

### 练习 1：手画 token dispatch 数据流

给定 `ep_degree=2, num_experts=4, num_local_experts=2`，每个 rank 有以下 token 分配：

```text
预期答案:

初始状态：
  Rank 0 持有发给所有 4 个专家的 token:
    Expert 0: 3 tokens, Expert 1: 5 tokens (Rank 0 的专家)
    Expert 2: 2 tokens, Expert 3: 4 tokens (Rank 1 的专家)
  Rank 1 同理

num_tokens_per_expert = [3, 5, 2, 4]

步骤 1: All-to-All 交换 token 数量
  Rank 0 发送: [3,5] 给 R0, [2,4] 给 R1
  Rank 0 接收: [3,5] from R0, [x,y] from R1
  → 知道本地专家 0,1 分别从各 rank 收到多少 token

步骤 2: 计算 splits
  counts_view = [[3,5], [2,4]]  (reshape [4] → [2,2])
  input_splits = [3+5, 2+4] = [8, 6]  ← 发给 R0: 8, 发给 R1: 6
  output_splits 由接收方计算

步骤 3: 变长 All-to-All dispatch
  Rank 0 发送: 前 8 个 token 给 R0, 后 6 个 token 给 R1
  Rank 0 接收: (由 output_splits 决定)

步骤 4: _permute 对齐 grouped_mm
  对收到的 token 按专家重排 + padding
```

### 练习 2：理解 ExpertTensorParallel 的 2D 切分

给定 `ep=4, tp=2, num_experts=64, hidden_dim=1024, dim=512`：

```text
预期答案:

w1 原始 shape: [64, 1024, 512]

2D mesh [ep=4, tp=2]:
  EP 切分 dim 0: 64/4 = 16 专家
  TP 切分 dim 1: 1024/2 = 512

每个 rank 的 w1_local: [16, 512, 512]
  - 16 个专家（EP 维度）
  - 512 hidden_dim（TP 维度，原来 1024 的一半）
  - 512 dim（完整）

_token_dispatch 执行顺序:
  1. DTensor.from_local(routed_input, tp_mesh, Replicate())
     → 标记输入在 TP 维度上是复制的
  2. super()._token_dispatch(...)  使用 ep_mesh
     → EP 维度的 All-to-All
```

### 练习 3：ReordererSequenceParallel 为什么只在 etp=1 时使用

```text
预期答案:

etp=1 时：TP 被 EP 借用
  - TP rank 0 和 rank 1 本来做同一专家的列/行切分
  - 现在 TP 被 EP 借走，rank 0/1 持有不同的专家
  - 如果 reorderer 不做序列并行：
    - 两个 rank 对全部 1000 tokens 做 reorder
    - 生成相同的 token_indices
    - EP All-to-All 发送重复的 token！
  - ReordererSequenceParallel 解决：
    - Rank 0 只处理 token[0:500]
    - Rank 1 只处理 token[500:1000]
    - 不重叠 → 不重复

etp=tp 时：TP 保持独立
  - TP rank 0 和 rank 1 持有同一专家的不同切片
  - 两个 rank 需要看到相同的 token（因为要做列切分计算）
  - 所以 reorderer 不需要切分 → 不使用 ReordererSequenceParallel
  - 输入标记为 Replicate()，两个 rank 处理相同 token
```

### 练习 4：理解 _apply 的 Hook 注册差异

```text
ExpertParallel._apply:
  partition_fn ✓  (切分 w1/w2/w3 到 Shard(0))
  input_fn ✓     (token dispatch)
  output_fn ✓    (token combine)

TensorParallel._apply:
  partition_fn ✓  (切分 w1/w2/w3 到 Shard(1/2))
  input_fn ✓     (Replicate + Partial)
  output_fn ✗    (TP all-reduce 由 DTensor 自动处理)

ExpertTensorParallel._apply:
  partition_fn ✓  (2D 切分到 [Shard(0), Shard(1/2)])
  input_fn ✓     (TP 输入准备 + EP dispatch)
  output_fn ✓    (EP combine)

ReordererSequenceParallel._apply:
  partition_fn ✗  (TokenReorderer 没有参数)
  input_fn ✓     (token 切分)
  output_fn ✓    (索引调整)
```

## 4.2 运行测试

```bash
# 单元测试（不需要 GPU）
uv run pytest tests/experimental/archon/test_parallel_dims.py -v -k "ep"

# 分布式 EP 测试（需要 GPU）
torchrun --nproc_per_node=2 tests/experimental/archon/torchrun/run_ep_tests.py

# EP + TP 组合测试（需要 4 GPU）
torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_ep_tests.py
```

## 4.3 交互式验证

```python
# 验证 1: ExpertParallel 的权重切分
from areal.experimental.models.archon.expert_parallel import ExpertParallel
import torch

# 模拟 GroupedExperts 的 3D 权重
class FakeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(64, 1024, 512))
        self.w2 = torch.nn.Parameter(torch.randn(64, 512, 1024))
        self.w3 = torch.nn.Parameter(torch.randn(64, 1024, 512))

# 在 torchrun 环境中
# ep = ExpertParallel()
# 验证 _partition_fn 后 w1 变成 DTensor[Shard(0)]
# assert isinstance(module.w1, DTensor)
# assert module.w1.placements == (Shard(0),)
# assert module.w1.to_local().shape[0] == 64 // ep_size

# 验证 2: splits 互换
# dispatch 后 self.input_splits = [10, 5, 8, 3]
# dispatch 后 self.output_splits = [10, 7, 6, 9]
# combine 时:
#   发送: self.input_splits = [10, 5, 8, 3]  ← 变成 combine 的 output_splits
#   接收: self.output_splits = [10, 7, 6, 9]  ← 变成 combine 的 input_splits
# 总量守恒: 10+5+8+3 = 26 (dispatch 发送) = 10+5+8+3 (combine 接收) ✓

# 验证 3: ReordererSequenceParallel 全局索引
# tp=2, num_tokens=1000, top_k=2
# Rank 0 处理 500 tokens → local_indices 范围 [0, 1000)
# Rank 1 处理 500 tokens → local_indices 范围 [0, 1000)
# 调整后:
#   Rank 0: global_indices = local_indices + 0      → [0, 1000)
#   Rank 1: global_indices = local_indices + 1000   → [1000, 2000)
# 不重叠 ✓
```

## 4.4 常见理解误区

| 误区 | 正确理解 |
| --- | --- |
| "EP 是一个独立的新并行维度" | EP 是从 dp_shard/cp/tp 维度"借用" GPU，不新增 GPU（详见 parallel_dims 深度解析） |
| "ExpertTensorParallel 先做 EP 再做 TP" | 两者同时生效：权重用 2D DTensor 切分，dispatch 用 EP mesh，梯度归约用 TP mesh |
| "TensorParallel 类和 PyTorch 的 TP 是同一个" | 这里的 `TensorParallel` 是专家专用的 TP（3D 权重 w1/w2/w3），不是通用 TP |
| "_token_combine 的 splits 和 dispatch 一样" | splits 是互换的！dispatch 的 input_splits 变成 combine 的 output_splits |
| "ReordererSequenceParallel 在所有 EP 场景都使用" | 仅 etp=1 时使用。etp=tp 时 TP rank 需要看到相同 token（Replicate）|
| "all_to_all_single 和 all_to_all_single_autograd 一样" | 前者不支持 autograd（用于 metadata），后者支持（用于 tensor 数据）|
| "ExpertParallel 需要每个专家的 token 数对齐" | All-to-All 支持变长通信（input_splits/output_splits），对齐只是给 grouped_mm 用的 |
| "_permute 是排序操作" | _permute 同时做重排序和 padding 对齐，是 grouped_mm 的前置操作 |

---

# 5. 附录

## 5.1 EP Token Dispatch/Combine 完整数据流

```text
时间轴 →

                    MoE Forward Pass (2 EP ranks, 4 experts)
                    ═══════════════════════════════════════

1. Router 阶段（每个 rank 独立）
   ┌───────────────────────────────────────────────────────────────┐
   │ 输入: x_flat [bs*slen, dim]                                   │
   │ 输出: top_scores [bs*slen, top_k],                            │
   │       selected_indices [bs*slen, top_k],                      │
   │       num_tokens_per_expert [num_experts]                     │
   └───────────────────────────────────────────────────────────────┘

2. Reorderer 阶段（如果 etp=1，受 ReordererSequenceParallel 影响）
   ┌───────────────────────────────────────────────────────────────┐
   │ 输入: top_scores, selected_indices                            │
   │ 输出: top_scores_sorted, token_indices_sorted, num_per_expert │
   │                                                               │
   │ etp=1 时: 每个 TP rank 只处理 1/tp 的 token                   │
   │ etp=tp 时: 每个 TP rank 处理全部 token（Replicate）            │
   └───────────────────────────────────────────────────────────────┘

3. Token Gather（收集路由的 token）
   ┌───────────────────────────────────────────────────────────────┐
   │ routed_input = x_flat[token_indices_sorted // top_k]          │
   │ 形状: [total_routed_tokens, dim]                              │
   └───────────────────────────────────────────────────────────────┘

4. EP Token Dispatch (ExpertParallel._token_dispatch)
   ┌───────────────────────────────────────────────────────────────┐
   │                                                               │
   │  4a. All-to-All 交换 token 数量                               │
   │      Rank 0                          Rank 1                   │
   │      [3,5,│2,4]  ───All-to-All───►  [?,?,│?,?]               │
   │       ↑ E0,E1  E2,E3                 E0,E1  E2,E3 ↑         │
   │       R0 专家  R1 专家               R0 专家  R1 专家         │
   │                                                               │
   │  4b. 计算 input_splits / output_splits                       │
   │      input_splits = sum(counts per target rank)               │
   │      output_splits = sum(received per source rank)            │
   │                                                               │
   │  4c. 变长 All-to-All dispatch                                │
   │      Rank 0 发送: [8 tokens → R0, 6 tokens → R1]             │
   │      Rank 0 接收: [8 tokens ← R0, ? tokens ← R1]            │
   │                                                               │
   │  4d. _permute 对齐 grouped_mm                                │
   │      重排 + pad → aligned_num_tokens                          │
   │                                                               │
   └───────────────────────────────────────────────────────────────┘

5. 本地 Expert 计算 (GroupedExperts.forward)
   ┌───────────────────────────────────────────────────────────────┐
   │ SwiGLU: silu(x @ w1.T) * (x @ w3.T) @ w2.T                  │
   │ 使用 grouped_mm（所有本地专家批量计算）                        │
   └───────────────────────────────────────────────────────────────┘

6. EP Token Combine (ExpertParallel._token_combine)
   ┌───────────────────────────────────────────────────────────────┐
   │  6a. _unpermute: 恢复 token 顺序，移除 padding                │
   │  6b. 反向 All-to-All（splits 互换）                           │
   │      Rank 0 发送: [? tokens → R0, ? tokens → R1]             │
   │      Rank 0 接收: [8 tokens ← R0, 6 tokens ← R1]            │
   └───────────────────────────────────────────────────────────────┘

7. Unsort + Combine
   ┌───────────────────────────────────────────────────────────────┐
   │ 将专家输出恢复到原始 token 位置                                │
   │ 与 routing scores 加权求和                                    │
   │ 加上 shared_experts 输出（如果有）                             │
   └───────────────────────────────────────────────────────────────┘
```

## 5.2 文件依赖关系

```text
areal/experimental/models/archon/expert_parallel.py (513 行)
  │
  ├── 依赖底层工具 ──────────────────────┐
  │                                       │
  │                                       ▼
  │                            areal/.../moe/utils.py
  │                              ├── _permute()      ← grouped_mm 对齐
  │                              ├── _unpermute()     ← 恢复 token 顺序
  │                              └── generate_permute_indices()  ← CUDA kernel
  │
  ├── 被 parallelize 调用 ──────────────┐
  │                                      │
  │                                      ▼
  │                            qwen3/infra/parallelize.py
  │                              ├── apply_moe_ep_tp()     (632-747 行)
  │                              │     策略选择:
  │                              │     ├── TensorParallel
  │                              │     ├── ExpertParallel
  │                              │     └── ExpertTensorParallel
  │                              │
  │                              └── parallelize_module()
  │                                    ├── experts → EP/TP/ETP
  │                                    └── reorderer → ReordererSequenceParallel
  │
  ├── 作用于 MoE 模块 ─────────────────┐
  │                                      │
  │                                      ▼
  │                            moe/moe.py
  │                              ├── MoE.forward()
  │                              │     ├── router()
  │                              │     ├── reorderer()     ← ReordererSequenceParallel
  │                              │     ├── experts()       ← ExpertParallel hooks
  │                              │     │     ├── pre-hook: _token_dispatch
  │                              │     │     ├── forward: GroupedExperts
  │                              │     │     └── hook: _token_combine
  │                              │     └── shared_experts()
  │                              │
  │                              └── GroupedExperts
  │                                    ├── w1, w2, w3  ← _partition_fn 切分
  │                                    └── forward()   ← DTensor.to_local()
  │
  └── 由 ArchonParallelDims 提供 Mesh ─┐
                                        │
                                        ▼
                               parallel_dims.py
                                 ├── get_mesh("ep")       → 1D EP mesh
                                 ├── get_mesh("ep_tp")    → 2D [EP, TP] mesh
                                 ├── get_mesh("tp")       → 1D TP mesh
                                 └── get_mesh("dp_shard_mod_ep") → EP 的 FSDP mesh
```

## 5.3 EP vs Ulysses：两种 All-to-All 的对比

| 特性 | EP (Expert Parallel) | Ulysses (Context Parallel) |
| --- | --- | --- |
| 切分维度 | 专家维度（expert dim） | 注意力头维度（head dim） |
| 通信目的 | 把 token 送到正确的专家 | 把序列片段合成完整序列 |
| 变长通信 | 是（不同专家的 token 数量不同） | 否（等分，每个 chunk 大小相同） |
| 通信次数 | 2 次（dispatch + combine） | 2 次（gather_seq + scatter_seq） |
| 对齐要求 | grouped_mm 需要（8/16/32 倍数） | cp_size 整除（padding 处理） |
| autograd | `all_to_all_single_autograd` | `all_to_all_single_autograd` |
| 使用位置 | MoE 层的 expert 前后 | Attention 层的 Q/K/V 前后 |

## 5.4 GroupedExperts.forward 中的 DTensor 处理

```python
# grouped_experts.py:176-198
def forward(self, x, num_tokens_per_expert):
    # 如果权重是 DTensor（被 EP/TP 切分过），转为本地张量
    if isinstance(self.w1, DTensor):
        w1 = self.w1.to_local()   # [num_local_experts, ...]
        w2 = self.w2.to_local()
        w3 = self.w3.to_local()
    else:
        w1, w2, w3 = self.w1, self.w2, self.w3

    if self.use_grouped_mm:
        # EP 场景：hooks 已经处理了 permute/padding
        # 非 EP 场景：需要 indices_padding_wrapper
        if (
            not isinstance(self.w1, DTensor)
            or "ep" not in self.w1.device_mesh.mesh_dim_names
        ):
            run_experts_fn = indices_padding_wrapper(_run_experts_grouped_mm)
        else:
            run_experts_fn = _run_experts_grouped_mm
        return run_experts_fn(w1, w2, w3, x, num_tokens_per_expert)
    else:
        return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)
```

**关键判断**：`"ep" not in self.w1.device_mesh.mesh_dim_names` 用于区分 EP 和非 EP 场景——EP 场景的 `_token_dispatch` hook 已经处理了 `_permute`，不需要再套 `indices_padding_wrapper`。

## 5.5 验证清单

- [ ] 理解 ExpertParallel 的三步 dispatch 流程（交换数量 → 变长 All-to-All → permute）
- [ ] 理解 splits 在 dispatch 和 combine 之间互换的原因（逆操作）
- [ ] 理解 TensorParallel 的 Replicate + Partial 语义（前向复制，反向归约）
- [ ] 理解 ExpertTensorParallel 如何组合 EP 和 TP（2D mesh、2D Shard）
- [ ] 理解 ReordererSequenceParallel 为什么只在 etp=1 时使用
- [ ] 理解 _permute/_unpermute 的对齐目的（grouped_mm 要求）
- [ ] 理解策略选择矩阵的完整逻辑（5 种 EP/TP/ETP 组合）
- [ ] 理解 distribute_module 的三个回调如何自动注册 hook
- [ ] 理解 GroupedExperts.forward 中 DTensor 检测和 padding wrapper 的判断逻辑
