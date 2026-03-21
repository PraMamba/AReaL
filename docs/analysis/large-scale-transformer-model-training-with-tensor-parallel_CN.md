# 大规模 Transformer 模型的张量并行（TP）训练

本教程演示如何使用**张量并行（Tensor Parallel）** 和**完全分片数据并行（Fully Sharded Data Parallel）** 在数百到数千个 GPU 上训练大型 Transformer 模型。

**前置知识：**

- PyTorch 2.3.0 或更高版本（已安装 CUDA/Linux）
- [Tensor Parallel API](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [DeviceMesh 入门教程](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)
- [Fully Sharded Data Parallel 入门教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

---

## 一、张量并行的工作原理

张量并行（TP）最初在 [Megatron-LM 论文](https://arxiv.org/abs/1909.08053) 中提出，是一种高效的模型并行技术，用于训练大规模 Transformer 模型。本教程中提到的**序列并行（Sequence Parallel, SP）** 是张量并行的一种变体，它在序列维度上对 `nn.LayerNorm` 或 `RMSNorm` 进行分片，以进一步节省训练期间的激活内存。随着模型变大，激活内存成为瓶颈，因此在张量并行训练中通常会对 `LayerNorm` 或 `RMSNorm` 层应用序列并行。

![Megatron-LM TP 分片示意图](https://docs.pytorch.org/tutorials/_images/megatron_lm.png)

> **图 1**：Transformer 模型 MLP 和 Self-Attention 层的张量并行分片方式。注意力和 MLP 中的矩阵乘法均通过分片计算完成。（图片来源：Megatron-LM 论文）

### 1.1 高层工作流程

**分片初始化阶段：**

- 确定每一层应用哪种 `ParallelStyle`，并通过调用 `parallelize_module` 对初始化的模块进行分片。
- 并行化后的模块的模型参数会被转换为 DTensor，由 DTensor 负责使用分片计算运行并行化模块。

**运行时前向/反向传播阶段：**

- 根据用户为每种 `ParallelStyle` 指定的输入/输出 DTensor 布局，运行适当的通信操作（如 `allreduce`、`allgather`、`reduce_scatter`）来转换 DTensor 布局。
- 对并行化的层运行分片计算，以节省计算和内存（例如 `nn.Linear`、`nn.Embedding`）。

---

## 二、何时以及为什么应用张量并行

PyTorch FSDP 已经具备将模型训练扩展到特定数量 GPU 的能力。然而，在进一步扩展模型大小和 GPU 数量时，会出现许多额外挑战，可能需要将张量并行与 FSDP 结合使用：

1. **环形延迟瓶颈**：当 world size（GPU 数量）过大（超过 128/256 个 GPU）时，FSDP 的集合通信（如 `allgather`）会被环形延迟主导。在 FSDP 之上实施 TP/SP，可将 FSDP 的 world size 缩小 8 倍（仅在主机间应用 FSDP），从而将延迟成本降低相同倍数。

2. **数据并行极限**：当由于收敛性和 GPU 内存限制，全局批量大小无法超过 GPU 数量时，张量/序列并行是"维持"全局批量大小并继续使用更多 GPU 进行扩展的唯一已知方法。

3. **矩阵运算优化**：对于某些类型的模型，当本地批量大小变小时，TP/SP 可以产生更适合浮点运算（FLOPS）优化的矩阵乘法形状。

### 2.1 实际案例

- **Llama 2 70B**：使用 2000 个 GPU 训练了 35 天，在 2K 规模下需要多维并行。
- 当 Transformer 模型变大时（如 Llama 2 70B），即使 `batch_size=1` 也无法仅用 FSDP 训练。Llama 2 的全局批量大小为 1K，因此在 2K GPU 下不能仅使用数据并行。

---

## 三、如何应用张量并行

PyTorch 张量并行 API 提供了一组模块级原语（`ParallelStyle`），用于配置模型各层的分片方式：

| API | 作用 |
|:---|:---|
| `ColwiseParallel` / `RowwiseParallel` | 按列或行方式分片 `nn.Linear` 和 `nn.Embedding` |
| `SequenceParallel` | 对 `nn.LayerNorm`、`nn.Dropout`、`RMSNorm` 等执行分片计算 |
| `PrepareModuleInput` / `PrepareModuleOutput` | 配置模块输入/输出的分片布局及相应通信操作 |

### 3.1 初始化 DeviceMesh

张量并行是一种 SPMD 分片算法，底层利用 DTensor 执行分片，并使用 DeviceMesh 抽象进行设备管理。TP 通常在单主机内工作：

```python
from torch.distributed.device_mesh import init_device_mesh

tp_mesh = init_device_mesh("cuda", (8,))
```

### 3.2 FeedForward 层的分片

以 Llama 2 模型的 `FeedForward` 层为例，它包含三个 Linear 层，执行 SwiGLU 风格的 MLP：

```python
# FeedForward 层的前向函数
def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

`w1` 和 `w3` 并行执行矩阵乘法，然后将结果输入 `w2`。根据张量并行论文的思路：对 `w1`/`w3` 按**列分片**，对 `w2` 按**行分片**，这样三个层结束后只需一次 `allreduce` 通信：

```python
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

layer_tp_plan = {
    # ColwiseParallel 默认输入布局为 Replicate
    # RowwiseParallel 默认输出布局为 Replicate
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}
```

### 3.3 Attention 层的分片

Attention 层包含 `wq`、`wk`、`wv` 线性层（投影到 q/k/v）和输出投影 `wo`。对 q/k/v 投影按**列分片**，对 `wo` 按**行分片**：

```python
layer_tp_plan = {
    "attention.wq": ColwiseParallel(use_local_output=False),
    "attention.wk": ColwiseParallel(use_local_output=False),
    "attention.wv": ColwiseParallel(use_local_output=False),
    "attention.wo": RowwiseParallel(),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}
```

> **关键细节**：列分片后，Linear 层的输出在最后一个张量维度上被分片。对于 Llama 模型的注意力层，存在与形状相关的 view 操作。`wq`/`wk`/`wv` 的列并行分片使激活张量在 `num_heads` 维度上被分片。设置 `use_local_output=False` 确保输出为 DTensor，DTensor 会自动处理 `num_heads` 维度的变化。

### 3.4 应用分片计划

调用 `parallelize_module` API 使每个 `TransformerBlock` 的计划生效：

```python
for layer_id, transformer_block in enumerate(model.layers):
    layer_tp_plan = {...}  # 上面定义的计划

    parallelize_module(
        module=transformer_block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_tp_plan,
    )
```

### 3.5 Embedding 和最终投影层

对第一个 `nn.Embedding` 层和最后的 `nn.Linear` 投影层也需要指定分片：

```python
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
        ),
        "output": ColwiseParallel(
            output_layouts=Replicate(),
        ),
    }
)
```

> **提示**：如果模型太大无法放入 CPU 内存，可以使用 `meta` 设备初始化（先在 meta 设备上初始化模型，再分片各层，最后物化模型），或在 Transformer 模型初始化过程中逐层并行化 `TransformerBlock`。

---

## 四、对 LayerNorm/RMSNorm 层应用序列并行

序列并行建立在上述张量并行之上。与基础张量并行（仅分片 Attention 和 FeedForward 模块内的张量，保持模块输入输出复制）不同，序列并行在**序列维度**上保持它们的分片状态。

典型的 `TransformerBlock` 前向函数：

```python
# TransformerBlock 的前向函数
def forward(self, x):
    h = x + self.attention(self.attention_norm(x))
    out = h + self.feed_forward(self.ffn_norm(h))
    return out
```

在大多数场景中，Attention 和 FeedForward 模块外部的激活（和梯度）形状为 `[batch_size, sequence_length, hidden_dimension]`。用 DTensor 的术语来说，序列并行使用 `Shard(1)` 布局进行模块前向/反向的激活计算。

### 4.1 启用序列并行的计划

```python
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    SequenceParallel,
)

layer_tp_plan = {
    # SequenceParallel 的输入输出使用 Shard(1) 布局
    # 表示在序列维度上分片
    "attention_norm": SequenceParallel(),
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1), Replicate()),
        desired_input_layouts=(Replicate(), Replicate()),
    ),
    "attention.wq": ColwiseParallel(use_local_output=False),
    "attention.wk": ColwiseParallel(use_local_output=False),
    "attention.wv": ColwiseParallel(use_local_output=False),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    "ffn_norm": SequenceParallel(),
    "feed_forward": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    "feed_forward.w3": ColwiseParallel(),
}
```

这里使用 `PrepareModuleInput` 将 Attention 和 FeedForward 层的模块输入布局从 `Shard(1)` 转换为 `Replicate()`，并将输出布局标记为 `Shard(1)`。与张量并行一样，用户只需指定输入输出的张量分片布局，层间通信会自动发生。

> **注意**：使用序列并行时，假设 `TransformerBlock` 的输入输出始终在序列维度上分片，以便多个 `TransformerBlock` 可以无缝连接。

### 4.2 Embedding 和最终层的配置

```python
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate()
        ),
    }
)
```

---

## 五、应用 Loss Parallel

Loss Parallel 是一种在计算损失函数时**节省内存和通信**的相关技术，因为模型输出通常非常大。在 Loss Parallel 中，当模型输出在（通常很大的）词表维度上被分片时，交叉熵损失可以高效计算，无需将所有模型输出聚集到每个 GPU 上。这不仅显著减少了内存消耗，还通过减少通信开销和并行执行分片计算来提高训练速度。

![Loss Parallel 示意图](https://docs.pytorch.org/tutorials/_images/loss_parallel.png)

> **图 2**：单个 GPU 上使用 Loss Parallel 的交叉熵损失前向计算。蓝色表示分片张量；绿色表示复制张量；黄色表示部分值张量（待 all-reduce）。黑色箭头为本地计算；红色箭头为 GPU 间的集合通信。

在 PyTorch 张量并行 API 中，Loss Parallel 可通过上下文管理器 `loss_parallel` 启用，使用时可以直接调用 `torch.nn.functional.cross_entropy` 或 `torch.nn.CrossEntropyLoss`，无需修改其他代码。

### 5.1 配置模型输出

模型预测（形状通常为 `[batch_size, sequence_length, vocabulary_size]`）应在词表维度上被分片：

```python
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            # 使用 DTensor 作为输出
            use_local_output=False,
        ),
    },
)
```

### 5.2 计算损失

使用 `loss_parallel` 上下文管理器，反向传播也需要在该上下文中进行：

```python
import torch.nn.functional as F
from torch.distributed.tensor.parallel import loss_parallel

pred = model(input_ids)
with loss_parallel():
    # 假设 pred 和 labels 的形状为 [batch, seq, vocab]
    loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
    loss.backward()
```

---

## 六、张量并行与完全分片数据并行的结合

由于张量并行会产生阻塞计算的通信，应确保其在快速通信通道（如 NVLink）内运行。实践中通常在**主机内**应用张量并行，在**主机间**应用 FSDP。

![FSDP + TP 示意图](https://docs.pytorch.org/tutorials/_images/fsdp_tp.png)

> **图 3**：FSDP 和 TP 在不同设备维度上工作。FSDP 通信发生在主机间，TP 通信发生在主机内。

### 6.1 2D 并行的实现

这种 2D 并行模式可通过 2D DeviceMesh 轻松表达：

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.distributed.fsdp import fully_shard

# 2D mesh 为 [dp, tp]，在 64 个 GPU 上进行 8 路 DP 和 8 路 TP
mesh_2d = init_device_mesh("cuda", (8, 8))
tp_mesh = mesh_2d["tp"]  # 连接主机内设备的子网格
dp_mesh = mesh_2d["dp"]  # 连接主机间设备的子网格

model = Model(...)

tp_plan = {...}

# 在 tp_mesh 上应用主机内张量并行
model_tp = parallelize_module(model, tp_mesh, tp_plan)
# 在 dp_mesh 上应用主机间 FSDP
model_2d = fully_shard(model_tp, mesh=dp_mesh, ...)
```

这使得我们可以在主机内轻松应用张量并行，在主机间应用 FSDP，且**无需修改 Llama 模型的任何代码**。张量（模型）并行与数据并行技术的结合，提供了使用大量 GPU 持续增加模型规模并高效训练的能力。

---

## 七、总结

本教程演示了如何使用张量并行结合完全分片数据并行，在数百到数千个 GPU 上训练大型 Transformer 模型。它解释了如何将张量并行应用到模型的不同部分，且无需修改模型本身的代码。张量并行是大规模训练的高效模型并行技术。

完整的端到端代码示例请参考 [pytorch/examples](https://github.com/pytorch/examples) 仓库中的 Tensor Parallel 示例。