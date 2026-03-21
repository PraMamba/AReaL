# DTensor — 现状、设计与未来展望

> 本文概述了 PyTorch DTensor 的当前状态、设计原则以及未来的工作方向。

---

## 一、总体设计目标与原则

DTensor 是 PyTorch 原生的**张量分片（Tensor Sharding）原语**。有关其动机和使用场景，请参考原始 RFC 和设计文档：

- [GitHub Issue #88838 — RFC PyTorch DistributedTensor](https://github.com/pytorch/pytorch/issues/88838)

DTensor 遵循以下几项核心设计目标与原则：

### 1.1 简洁的 SPMD 分片原语

目标是构建一个简洁的 SPMD（Single Program, Multiple Data）分片原语，以加速基于 PyTorch 的分布式训练/推理研究创新，并为分布式算法提供简洁的编程模型。

### 1.2 完全 PyTorch 原生

DTensor 在理论上应与任何 PyTorch 子系统无缝协作，包括 Autograd 引擎、`torch.compile`、嵌套子类（nested subclasses）、低精度数据类型（low-precision dtypes）、自定义算子（custom operators）等。

### 1.3 单设备语义（Single Device Semantic）

编写分布式算法时，就如同编写单设备程序一样，且具有相同的收敛性。单设备语义对 DTensor 至关重要——它是确保 SPMD 分片算法**收敛性/数值正确性**的唯一可靠方式。所有 API 均遵循单设备语义。

### 1.4 少即是多（Less is More）

DTensor 专注于构建正确的原语，而非高层分布式算法 API。倾向于暴露最少量且必要的简洁 API，为用户（即系统开发者）提供灵活性，使其能在此基础上进行构建。

---

## 二、用户体验

> **注意**：所有公开 API 目前均记录在 [torch.distributed.tensor 官方文档](https://pytorch.org/docs/main/distributed.tensor.html) 中，对于非实验性 API，我们承诺向后兼容性。

### 2.1 DeviceMesh：多维分片的基础

`DeviceMesh` 在 SPMD 编程模型中充当"**设备管理器**"或"**通信器**"的角色。在需要多维并行、且并行方案日趋复杂的世界中，甚至通信的设置都变得极具挑战性且难以理解。能够描述设备布局是实现多维分片的基本步骤。

作为 DTensor 开发的一部分，我们构建了 `DeviceMesh` 来管理集群设备布局，并在集群中的设备之间初始化通信器（ProcessGroup）。自 PyTorch 2.2 版本起，`DeviceMesh` 已作为核心分布式抽象发布为 Beta 版。

- 教程参考：[Getting Started with DeviceMesh — PyTorch Tutorials](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)

### 2.2 分片 API（Sharding APIs）

分片 API 自最初引入以来一直相当稳定。主要提供以下功能：构造 DTensor、转换 DTensor 布局、与本地 `torch.Tensor` 交互、调试工具，以及一些实验性 API（如自定义算子注册）。

#### 2.2.1 构造 DTensor 的三种方式

1. **`distribute_tensor`**
   用于叶节点的完整张量（即 `nn.Parameter` 或 Buffer），主要用于参数分片初始化。为正确保持单设备语义，默认在所有 mesh 维度上从 `group_rank=0` 进行广播/分散。最近引入了 `src_data_rank` 关键字参数来控制源数据的 rank，若传入 `None` 则跳过通信。

2. **`DTensor.from_local`**
   静态方法，允许从本地张量初始化。此 API 支持 autograd，主要用于张量计算过程中。

3. **原生 DTensor 构造函数**
   如 `torch.distributed.tensor.zeros/ones` 等，通过 `device_mesh` 和 `placements` 指定分片方式。与方式 1 相比，原生构造函数无需对完整张量执行分片（即 scatter/broadcast），可通过适当的 RNG 支持直接初始化分片数据（详见随机算子部分）。

#### 2.2.2 三种 Placement 类型

| Placement 类型 | 描述 |
|:---|:---|
| `Shard(dim)` | 在当前 mesh 维度上对张量的第 `dim` 维进行分片。分片使用 `torch.chunk` 语义处理不均匀情况 |
| `Replicate()` | 在当前 mesh 维度上进行复制 |
| `Partial()` | 表示待归约（pending reduction）的部分张量状态。通常来自中间计算（如 `aten.mm.default` 的输出）。因为 PyTorch 是 eager 优先的，所以每个状态都可以被终端用户检查，因此选择将其公开 |

> **重要规则**：对于每个算子，如果一个输入张量是 DTensor，则要求该算子的**所有**张量输入都必须是 DTensor。这是因为需要知道所有输入操作数的分片布局，才能推导出正确的输出分片布局。

#### 2.2.3 模块级 API：`distribute_module`

虽然 DTensor 主要提供张量级分片原语 API，但也添加了模块级 API `distribute_module`，以帮助开发者编写应用于 `nn.Module` 的分片算法。它接受三个函数作为参数：

- **`partition_fn`**：定义如何对 `nn.Module` 内的模型参数或缓冲区进行分区/分片的可调用对象。提供灵活性让开发者定义详细的分片行为（如对不同参数调用 `distribute_tensor` 或 `DTensor.from_local`）。
- **`input_fn`**：定义如何处理输入张量的可调用对象。可用于将普通 `torch.Tensor` 转换为具有所需分片注解的 DTensor（如使用 `DTensor.from_local`），或将输入 DTensor 重新分发到另一种分片布局等。
- **`output_fn`**：定义如何处理输出张量的可调用对象。可调用 `redistribute` 将输出更改为不同的分片布局，或对输出 DTensor 调用 `to_local` 以使输出退出 DTensor 计算区域等。

---

## 三、DTensor 布局转换（"集合通信"）

在 SPMD 编程模型中，如果没有 DTensor 这样的高级抽象，用户必须手动执行张量分片、编写不同的可微分集合通信操作以保持单设备语义，并在各处跟踪张量分片状态。这不仅繁琐且容易出错，在处理多维分片时更是异常复杂。

`redistribute` API 通过将不同的集合通信调用建模为 **DTensor 布局转换** 来抽象这些复杂性。用户可以直接处理张量分片布局转换，无需操心如何编写底层的集合通信实现来达到所需的分片布局。

### 3.1 单维度上的常见转换

| 转换 | 对应的集合通信 |
|:---|:---|
| `Shard(dim)` → `Replicate()` | `all_gather` |
| `Shard(src_dim)` → `Shard(dst_dim)` | `all_to_all` |
| `Replicate()` → `Shard(dim)` | 本地 `torch.chunk` |
| `Partial()` → `Replicate()` | `all_reduce` |
| `Partial()` → `Shard(dim)` | `reduce_scatter` |

### 3.2 多维 DeviceMesh 上的转换

`redistribute` API 的另一个重要特性是能够在**多维 DeviceMesh** 上计算出正确的转换步骤。这对于高级用户来说，手动使用集合通信操作将变得极为复杂。

**示例**：在 2D DeviceMesh 上的转换目标：

```
[Shard(0), Shard(1)] → [Shard(1), Shard(0)]
```

`redistribute` 会将其分解为多个步骤，以确保数据在正确的 rank 上完成正确的布局转换：

```
[Shard(0), Shard(1)] → [Shard(0), Replicate()]
[Shard(0), Replicate()] → [Shard(1), Replicate()]
[Shard(1), Replicate()] → [Shard(1), Shard(0)]
```

即使是 2D DeviceMesh 也需要多个步骤，对于 3D 或更高维度的 DeviceMesh 则更加复杂。`redistribute` API 简化了转换过程，使用户可以专注于构建分片算法。

- 详细的多维 mesh 重分发算法变更请参考：[PR #131210](https://github.com/pytorch/pytorch/pull/131210)

### 3.3 注意事项

1. `redistribute` 只是一个执行标准化 DTensor 布局转换的 API，用户仍可以对 DTensor 输入编写手动集合通信实现，并使用适当的 DTensor 布局构造输出 DTensor。
2. 目前仅支持在**同一 DeviceMesh 内**转换 DTensor 布局，尚不支持跨 DeviceMesh 转换。但根据不同需求，可以实现跨不同子网格（submesh）的布局转换（例如用于 MPMD 或检查点重分片场景）。

---

## 四、随机算子（Random Operators）

PyTorch 中的随机算子与其他算子不同，因为它涉及随机数生成的工作方式。对于 DTensor 这样的抽象，需要关注不同分片类型下应如何生成随机数，以保证输出张量分片仍然有意义。

### 4.1 关键属性

以 `aten.dropout.default` 为例，分片场景下需要保持的重要属性：

- **复制张量输入**：应产生复制张量输出，即在某个 mesh 维度的各 rank 上产生**相同**的数据。
- **分片张量输入**：应产生分片张量输出，即在某个 mesh 维度的各 rank 上产生**不同**的数据。理想情况下，不同分片产生的数据应如同在单设备上为"全局/完整"张量生成的数据一样。

这不仅适用于运行时随机算子，也适用于张量创建算子（如 `torch.randn`、`torch.rand` 等）。

### 4.2 实现方式

利用 **PhiloxRNG**，在从随机分布中采样时同时使用 seed 和 offset。具体实现了一个 `OffsetRNGTracker`：

1. 从 `rank=0` 懒同步/广播一个种子（seed），作为"全局种子"。
2. 对于每个随机算子，DTensor 使用 RNGTracker 计算相对于全局种子的偏移量（offset），并根据输入 DTensor 的分片情况移动/跟踪偏移量。

**示例**：假设从 `(global_seed, offset)` 状态开始，在分片的 DTensor 输入上运行 sharded dropout 计算：

- 执行期间，分片/rank `i` 的起始状态为：`(global_seed, offset + (i-1) * numel_local_shard)`
- 执行后，每个 rank 上的偏移量移动到：`offset + numel_global_tensor`

这样确保了采样行为模拟单设备语义。

---

## 五、自定义算子（Custom Operators）

### 5.1 动机

DTensor 一直致力于支持大量原生 PyTorch 算子，但可扩展性对 PyTorch 至关重要。如果用户定义了自定义算子，它也应该能与 DTensor 配合工作。

### 5.2 机制

由于 DTensor 不了解自定义算子的数学语义（例如它是逐点运算还是归约运算），自定义算子的作者需要提供一个"**分片公式（sharding formula）**"，告诉 DTensor 如何处理分片。这类似于新自定义算子 API 所需的 shape formula。

### 5.3 API：`register_sharding`

已实现一个实验性 API `register_sharding`，允许用户定义自定义算子的分片策略，使 DTensor 能够根据输入分片推导出输出分片。后续应征求关于此 API 的反馈并使其稳定化，还可以考虑暴露一些常见的分片策略（如 pointwise、follow strategy）。

---

## 六、`local_map`：与本地张量交互

### 6.1 使用场景

DTensor 允许用户编写如同单设备程序的分布式程序。但开发者在某些场景下仍然希望能"深入一层"，编写手动集合通信代码：

- **场景 1**：大部分模型计算使用 DTensor，但对 1–2 个层编写自定义优化代码，提前调用集合通信以实现更好的通信计算重叠。
- **场景 2**：编写自定义 Triton 代码，希望直接在 Python 中调用而无需通过自定义算子注册流程，已知预期的输入/输出分片，只想让 Triton 内核在本地张量上运行。

### 6.2 API 设计

`local_map` 是一个**函数装饰器**：

1. 提取 DTensor 的本地分片（调用 `to_local`）
2. 使用本地张量直接调用函数
3. 根据用户提供的 `out_placements`，将输出重建为对应的 DTensor

### 6.3 实际案例

一个典型的使用案例是 [torchtitan](https://github.com/pytorch/torchtitan) 中的**自定义 RMSNorm Triton 内核**：通过 `local_map`，可以轻松地在自定义 RMSNorm 层上运行 SequenceParallel，同时保留所有其他 DTensor 分片。

---

## 七、调试能力（Debuggability）

任何抽象或 API 都会隐藏底层实现的某些细节。高级用户（即分布式系统开发者）希望深入了解底层发生了什么。

### 7.1 分片可视化

帮助开发者理解分片（尤其是多维分片）如何执行，以及分片与设备之间的映射关系。

- API：`visualize_sharding`（目前为初步工具，尚未完善）

### 7.2 通信与计算追踪

追踪 DTensor 区域执行时发生的通信和计算——不仅包括每个算子执行的次数，还包括完整调试分析所需的执行顺序。

- 工具：`CommDebugMode`

`CommDebugMode` 本身可以作为独立功能发布。此工具对于使用/开发复杂并行方案的用户非常有用，他们需要确保通信在正确的位置以正确的顺序发生。例如，在构建 FSDP/TP 等高级并行方案时，开发者会频繁使用此工具。

---

## 八、XLA 后端

torchxla 最近开发了其 SPMD 分片 API，底层利用了 XLA 编译器的 GSPMD 分区器。DTensor 与 torchxla 的 SPMD 分片 API 进行了集成，因为二者共享相似的分片概念。

- 参考 Issue：[GitHub Issue #92909 — RFC XLA Lazy Backend Support In DistributedTensor API](https://github.com/pytorch/pytorch/issues/92909)

---

## 九、并行方案编写（Parallelism Authoring）

DTensor 是帮助并行方案编写变得更简单的分片原语。能够在其基础上构建并行方案，证明了 DTensor 的有效性。目前已在此基础上开发了多种并行方案。

### 9.1 FSDP2

FSDP2 使用 DTensor 作为数据抽象层，带来以下优势：

- 更简单的检查点保存/加载（DTensor + DCP 是推荐方案）
- 对单个参数的便捷操作
- 更好的模型并行支持（2D/3D），原生集成 PyTorch TP，支持 2D/3D 并行的分片 `state_dict`
- 更好的张量级功能支持，如 meta device 初始化、梯度范数裁剪、量化等

此外，还有 **SimpleFSDP** 探索，它提供了一种使用纯编译优化来实现 FSDP 的替代方式，可以进一步简化 FSDP 实现（仍在探索阶段）。

### 9.2 张量并行 / 序列并行（Tensor/Sequence Parallel）

TP API 以更直接的方式使用 DTensor，要求用户为模型参数和输入标注分片方式，然后用这些分片信息构造 DTensor 并运行分片计算。这意味着：

- 所有输入/模型参数都是 DTensor
- 所有计算通过 DTensor 的 `__torch_dispatch__` 进行
- 检查点保存/加载直接与 DTensor 交互（DCP 已支持）

TP API 已发布为 Beta 版。未来的工作包括：

- 使用 TP 分片卷积层（如支持 `aten.conv` 算子，用于 ViT 等模型的卷积层分片）
- 更多量化（混合精度）支持，配合 torchao
- 通过自定义算子注册或 `local_map` 支持更多自定义化
- 融合 QKV 分片（可在 TP API 内部支持），在训练中较少使用（GPT2 除外），但对推理可能有用

### 9.3 上下文并行（Context Parallel）

当前 CP 实现利用 DTensor 的 dispatch 机制，允许**非模型侵入式**的更改。目前仅支持 FlashAttention，具体是 `F.scaled_dot_product_attention` API 及其 FlashAttention 内核路径。

未来改进方向：

- **FlexAttention**：鉴于注意力机制有许多创新变体，应以更细粒度的方式支持上下文并行。DTensor 用于 FlexAttention 可能更加自然。
- 现有方案应良好支持 `F.sdpa` 路径（包括 flash attention 和 memory efficient attention），并向用户暴露适当的控制参数（如选择 allgather/ring 通信方式），根据用户反馈改进 API。

### 9.4 专家并行（Expert Parallel）

EP 工作目前正在进行中，对 DTensor 基础设施的需求尚未完全明确。

- 参考文档：[PT-D MoE & EP exploration](https://docs.google.com/document/d/1-gw6sGBW_VNR1MxPc8cJYO52TKL_bXCNR4qB7Ywn1Mo/) （Tianyu Liu 撰写的进展文档）

---

## 十、性能

### 10.1 Tensor 子类开销

DTensor 是基于 `__torch_dispatch__` 的 Tensor 子类，因此继承了 Tensor 子类的所有优缺点。对于 CPU 敏感型工作负载（如参数量大但计算量极低的模型），CPU 上可能存在显著开销。

**原因分析**：对于 Tensor 子类输入执行 `torch.add`，PyTorch 调度器需要经过如下路径：

```
torch.add → torch.ops.aten.add.default → aten:add
→ subclass 的 __torch_dispatch__（通过 torch_dispatch_key）
→ aten.add.default → aten:add → cuda:add
```

与普通 `torch.Tensor` 相比，Tensor 子类引入了至少 2 次额外的往返：1 次从 C++ 到 Python，1 次从 Python 回到 C++。

**完全消除**子类开销的方式只有两种：

1. **`torch.compile`**：为 DTensor 启用编译器
2. **CUDA Graph**：让 eager 模式的 CUDAGraph API 与 DTensor 配合工作

二者并不互斥（如 `torch.compile` 有 cudagraph 后端）。

### 10.2 TorchDispatch 中的 CPU 开销优化

在不借助 `torch.compile` 或 CUDA Graph 的情况下，可以优化调度逻辑。目前已进行了多项改进，包括在多个层级进行适当的缓存、仅在需要时进行展平（flattening）。

主要开销来源是缓存层——`OpSchema` 的 `__hash__` 和 `__eq__` 方法用于检查是否复用分片传播结果。未来可探索的方向：

- 进一步改进调度逻辑的 Python 性能（更多缓存、改进 `__hash__` 和 `__eq__` 性能）
- 实验性地将部分实现移至 C++（如 `DeviceMesh`、`Placement`、`DTensorSpec`），观察是否能改善哈希性能
- 检查 PyTorch 核心中的 Tensor 子类重调度逻辑，寻找改进点

### 10.3 torch.compile + DTensor

启用 `torch.compile` 与 DTensor 配合，可以**完全消除 CPU 开销**。目前已启用此功能并实现了多项优化：

- 完全移除所有与子类相关的 CPU 开销
- 应用了 DTensor 的模型可直接获得 TorchInductor 的计算融合优化
- DTensor 编写的分片算法可在 TorchInductor 内部进一步优化（如 Async TP 和 SimpleFSDP 均实现了通信重叠/重排序优化）
- 保持分片逻辑合理简洁，并利用编译器执行更细粒度的优化

后续需要加固 `torch.compile` + DTensor 路径：

- 确保所有单元测试与 `torch.compile` 兼容，编译器覆盖 DTensor 的每个新/现有功能
- 端到端的编译集成测试（2D/3D 并行），以防护编译时间回退和性能回退

---

## 十一、未来工作

DTensor 已作为公开 API 发布，越来越多用户开始尝试使用，预计会有更多问题被提出。直接处理用户报告的问题对于满足社区需求并使其稳定化非常重要。

### 11.1 集合通信算子（Collective Operators）

**现状**：用户在 DTensor 上调用集合通信时基本无法工作（抛出"算子不支持"错误），官方建议使用 `redistribute` API。这对于不关心底层实现、只想处理分片的用户来说已足够。

**需求**：许多系统工程师已熟悉集合通信概念，希望将 DTensor 与显式集合通信 API 结合使用。`local_map` 可以实现，但直接让集合通信算子对 DTensor 生效则无需学习新 API。

**解决方案**：支持特定集合通信算子（从 functional collective 开始）直接接受 DTensor 输入：

| 集合通信算子 | 输入 | 输出 |
|:---|:---|:---|
| `all_gather_into_tensor` | `Shard(dim)` | `Replicate()` |
| `all_reduce` | `Partial()` | `Replicate()` |
| `reduce_scatter` | `Partial()` | `Shard(dim)` |
| `shard_dim_all_to_all` | `Shard(src_dim)` | `Shard(dst_dim)` |

同时需对 `process_group` 参数进行检查，确保其与 DTensor 的 `device_mesh` 匹配。

### 11.2 批量集合通信（Batched Collectives）

批量集合通信是逐参数分片的重要优化（如 FSDP2 需要高效的批量 `all_gather` 来预取下一个 bucket）。目前仅限 FSDP2 内部使用。

应添加的支持：

- `allgather_copy_in`：接收一组 `Shard` DTensor
- `allgather_copy_out`：产生一组 `Replicate` DTensor
- 等效的 `reduce_scatter` 批量集合通信支持
- 等效的 `all_reduce` 批量集合通信支持

### 11.3 不均匀分片 + 填充（Uneven Sharding + Padding）

**当前问题**：对于参数维度不能被 mesh 维度大小整除的情况，默认不对本地分片进行填充，仅在执行 `allgather` 或 `redistribute` 等集合通信时才填充。这种方式虽然实现更简单，但会引入额外的拷贝开销。当需要将多个参数组合在一起（如 FSDP 算法中的通信重叠）时，填充多个参数的拷贝开销变得昂贵。

**改进方案**：改为**默认填充**的分片方式：

1. `distribute_tensor` 应先将张量填充到 mesh 维度大小的倍数，再进行分片和分散。
2. DTensor 的 `_local_tensor` 应始终保持填充状态，使通信操作可以直接使用本地张量而无需额外拷贝。
3. `DTensorSpec` 可以记录/计算当前 rank 的 `_local_tensor` 是否已填充、如何取消填充，作为方法或属性。
4. 分布式算子需要推理填充数据——由于是算子特定的（不同归约算子需要不同的填充值以保证数学正确性），像 GSPMD 那样枚举所有算子的方法对 PyTorch 庞大的算子库不太可行。
5. **替代方案**：仅保持 `_local_tensor`（或 data）默认填充，但在分片传播时动态取消填充（类似 `redistribute_local_tensor`）。在执行实际计算前先取消 `_local_tensor` 的填充。

### 11.4 支持自定义化（Support Customizations）

良好的编程模型应让用户能利用高级抽象的同时，不阻止他们深入底层做任何想做的事。需要重点支持自定义化和 PyTorch 扩展子系统，原因包括：

- 扩展系统在许多场景下很流行，可定制性对高级用户至关重要
- 许多创新发生在此领域（如用户编写优化算子如 `FusedRMSNorm`，注册为 PyTorch 算子或在模型中直接调用）
- 正确的扩展点允许按节奏开发，同时不阻塞用户

**三个重要的自定义化点**：

1. **`register_sharding`**：加固此 API 使其稳定，改善用户体验
2. **`local_map`**：允许用户编写手动集合通信或内核而无需注册为算子；应收集用户反馈评估此 API 的灵活性
3. **集合通信 / 自定义集合通信**：系统开发者调用手动集合通信时应能保持一致的行为

### 11.5 PyTorch 算子覆盖率

PyTorch 有 2000+ 个算子，DTensor 目前支持约 300+ 个常用算子，距离覆盖完整算子集还有较大差距。

为每个算子编写分片策略并非易事。除了并行化工作外，最好能找到一种**创新方法**来加速这项工作：

- 部分 PyTorch 算子有复杂的数学语义（如 conv、sdpa），确定正确的分片策略需要大量非平凡的工作
- 维护所有分片策略并逐一修复 bug，工程成本巨大

**解决思路**：利用 PT2 中开发的**分解函数（decomposition functions）**，在分解后的函数上直接进行分片传播。由于大多数基本数学算子已启用，对于复杂数学算子，可以直接利用分解后的分片传播来生成分片策略，从而显著加速分片策略的编写。

### 11.6 分片 Placement 顺序（Shard Placement Order）

**背景**：已实现了私有的 `_StridedShard` Placement，作为使 FSDP2 + TP 的 `state_dict` 正确运行的一部分。`_StridedShard` 与普通 `Shard` 的不同在于它有一个 `split_factor`，记录分片以 strided 方式发生。虽然 `_StridedShard` 解决了 FSDP2 + TP 的 2D 分片问题，但用户很难理解什么是 strided 分片。

**根本动机**：在 n 维 mesh 分片中，同一张量维度在不同设备 mesh 维度上的分片可能有**不同的顺序**。

**改进方案**（追求简洁性）：

1. 在 `DTensorSpec` 中添加一个可选的 `shard_order` 列表，描述 "placement" 到 "mesh dimension" 的顺序。对于普通分片为 `None`（或 `[1, 2, 3…]`）。
2. 对于 strided 分片情况，可以从此 `shard_order` 列表中直接得知同一张量维度在多个 mesh 维度上被分片，且分片顺序是 strided 的。
3. 所有相关的 `_StridedShard` 通信逻辑可以复用于此 `shard_order` 方案，无需引入单独的 Placement 类型。

---

> **总结**：DTensor 作为 PyTorch 原生的张量分片原语，通过简洁的 SPMD 编程模型、单设备语义保证和灵活的扩展机制，为分布式训练/推理提供了强大的基础抽象。未来将持续在算子覆盖率、性能优化、自定义化支持和调试工具等方面进行改进，推动其走向稳定发布。