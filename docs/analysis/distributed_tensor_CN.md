# torch.distributed.tensor API 参考

> **注意**：`torch.distributed.tensor` 目前处于 Alpha 阶段，正在积极开发中。文档中列出的大多数 API 承诺向后兼容，但必要时可能会进行 API 变更。

---

## 一、PyTorch DTensor（分布式张量）

PyTorch DTensor 提供简洁灵活的张量分片原语，透明处理分布式逻辑，包括分片存储、算子计算和跨设备/主机的集合通信。`DTensor` 可用于构建不同的并行方案，并在多维分片场景下支持分片 `state_dict` 表示。

基于 `DTensor` 构建的 PyTorch 原生并行方案：

- [Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [FSDP2](https://pytorch.org/docs/stable/fsdp.html)

`DTensor` 遵循 **SPMD**（单程序多数据）编程模型，使用户能像编写单设备程序一样编写分布式程序，并具有相同的收敛性。它通过 `DeviceMesh` 和 `Placement` 提供统一的张量分片布局（DTensor Layout）：

- **`DeviceMesh`**：使用 n 维数组表示设备拓扑和集群通信器。
- **`Placement`**：描述逻辑张量在 `DeviceMesh` 上的分片布局。支持三种类型：`Shard`、`Replicate` 和 `Partial`。

---

## 二、DTensor 类 API

`DTensor` 是 `torch.Tensor` 的子类。一旦创建，可以像 `torch.Tensor` 一样使用，包括运行各种 PyTorch 算子（如同在单设备上运行），从而实现正确的分布式计算。

### 2.1 DTensor 类

```python
class torch.distributed.tensor.DTensor(**kwargs)
```

`DTensor`（分布式张量）是 `torch.Tensor` 的子类，提供单设备式的抽象来编程多设备 `torch.Tensor`。它通过 `DeviceMesh` 和 `Placement` 类型描述分布式张量分片布局。

调用 PyTorch 算子时，`DTensor` 会覆盖算子以执行分片计算，并在必要时发起通信。算子计算过程中，`DTensor` 会根据算子语义正确地转换或传播 placement（DTensor Layout），并生成新的 `DTensor` 输出。

> **注意**：为确保分片计算的数值正确性，`DTensor` 要求算子的每个 Tensor 参数都是 DTensor。

> **注意**：不建议直接使用 Tensor 子类构造函数创建 `DTensor`（它不能正确处理 autograd）。请参考后文的 DTensor 创建方式。

### 2.2 关键方法与属性

**`full_tensor(*, grad_placements=None)`**

返回此 DTensor 的完整张量。它会执行必要的集合通信，从 DeviceMesh 中的其他 rank 收集本地张量并拼接。语法糖等价于：

```python
dtensor.redistribute(placements=[Replicate()] * mesh.ndim).to_local()
```

**`placements` 属性**（只读）

```python
@property
placements: tuple[Placement, ...]
```

描述此 DTensor 在其 DeviceMesh 上的布局。此属性为只读，不可设置。

---

## 三、DeviceMesh 作为分布式通信器

`DeviceMesh` 源自 DTensor，作为描述集群设备拓扑和表示多维通信器（基于 `ProcessGroup`）的抽象。详细的创建和使用方式请参考 [DeviceMesh 入门指南](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)。

---

## 四、DTensor Placement 类型

DTensor 在每个 `DeviceMesh` 维度上支持以下 `Placement` 类型：

### 4.1 Shard

```python
class torch.distributed.tensor.placement_types.Shard
```

`Shard(dim)` 描述 DTensor 在张量维度 `dim` 上沿对应 `DeviceMesh` 维度进行分片，每个 rank 仅持有全局张量的一个分片。遵循 `torch.chunk(dim)` 语义——当张量维度不能被 DeviceMesh 维度大小整除时，最后几个分片可能为空。

> **警告**：在张量维度大小不能被 DeviceMesh 维度整除的维度上分片，目前是实验性功能，可能会变更。

**静态方法：**

```python
static local_shard_size_and_offset(curr_local_size, num_chunks, rank)
    -> tuple[int, int]
```

给定当前本地张量大小（可能已在某些维度上分片），计算期望分块数下的新本地分片大小和偏移量。返回 `(新本地分片大小, 偏移量)`。

> **注意**：新本地分片偏移量是相对于当前分片张量的，而非全局张量。计算全局偏移请参见 `_utils.compute_local_shape_and_global_offset`。

### 4.2 Replicate

```python
class torch.distributed.tensor.placement_types.Replicate
```

`Replicate()` 描述 DTensor 在对应 `DeviceMesh` 维度上进行复制，每个 rank 持有全局张量的一个副本。可被所有 DTensor API 使用。

### 4.3 Partial

```python
class torch.distributed.tensor.placement_types.Partial(reduce_op="sum")
```

`Partial(reduce_op)` 描述在指定 `DeviceMesh` 维度上待归约的 DTensor，每个 rank 持有全局张量的部分值。用户可通过 `redistribute` 将 `Partial` DTensor 转换为 `Replicate` 或 `Shard(dim)` placement，这会触发底层的通信操作（如 `allreduce`、`reduce_scatter`）。

> **注意**：`Partial` placement 可作为 DTensor 算子的结果生成，且只能由 `DTensor.from_local` API 使用。

**参数：**

- `reduce_op`（`str`，可选）：用于生成 Replicated/Sharded DTensor 的归约操作。支持逐元素归约操作："sum"、"avg"、"product"、"max"、"min"。默认："sum"。

### 4.4 MaskPartial

```python
class torch.distributed.tensor.placement_types.MaskPartial(
    reduce_op=None, mask_buffer=None, offset_shape=None, offset_dim=0, *args, **kwargs
)
```

为行分片 Embedding 算子设计的部分掩码 placement，用于掩码和调整索引到本地 Embedding 分片。Embedding 掩码是 `Partial` placement 的一种特殊类型。

> **注意**：此 `MaskPartial` placement 的生命周期跟随对应 DTensor 的生命周期，即 `indices_mask` 仅在 DTensor 存活期间有效。

### 4.5 Placement 基类

```python
class torch.distributed.tensor.placement_types.Placement
```

Placement 类型的基类，描述 DTensor 如何放置在 `DeviceMesh` 上。`Placement` 和 `DeviceMesh` 共同描述 DTensor Layout。它是 `Shard`、`Replicate` 和 `Partial` 三种主要 Placement 类型的基类，不建议直接使用。

```python
is_shard(self, dim=None) -> bool
```

---

## 五、创建 DTensor 的不同方式

有三种方式构造 `DTensor`：

1. **`distribute_tensor()`**：从每个 rank 上的逻辑/"全局" `torch.Tensor` 创建 `DTensor`，用于分片叶节点张量（如模型参数/缓冲区和输入）。
2. **`DTensor.from_local()`**：从每个 rank 上的本地 `torch.Tensor` 创建 `DTensor`，用于从非叶节点张量（如前向/反向过程中的中间激活张量）创建。
3. **DTensor 工厂函数**（如 `empty()`、`ones()`、`randn()` 等）：通过直接指定 `DeviceMesh` 和 `Placement` 创建，可直接在设备上物化分片内存，而非在初始化逻辑张量后再执行分片。

### 5.1 从逻辑 torch.Tensor 创建

在 `torch.distributed` 的 SPMD 编程模型中，通过 `torchrun` 启动多个进程执行同一程序，模型会在不同进程上先初始化。`DTensor` 提供 `distribute_tensor()` API 将模型权重或张量分片为 DTensor，使创建的 DTensor 遵循单设备语义。

#### distribute_tensor

```python
torch.distributed.tensor.distribute_tensor(
    tensor, device_mesh=None, placements=None, *, src_data_rank=0
)
```

将叶节点 `torch.Tensor`（如 `nn.Parameter`/缓冲区）按指定的 `placements` 分布到 `device_mesh` 上。`device_mesh` 和 `placements` 的秩必须相同。`tensor` 是逻辑/"全局"张量，API 默认使用 DeviceMesh 维度上第一个 rank 的张量作为数据源，以保持单设备语义。

> **注意**：使用 `xla` 设备类型初始化 DeviceMesh 时，`distribute_tensor` 返回 `XLAShardedTensor`（实验性功能）。

**参数：**

- `tensor`（`torch.Tensor`）：要分布的张量。在不能被 mesh 维度设备数整除的维度上分片时，使用 `torch.chunk` 语义（实验性行为）。
- `device_mesh`（`DeviceMesh`，可选）：分布张量的 DeviceMesh。若未指定，须在 DeviceMesh 上下文管理器内调用。默认：`None`。
- `placements`（`List[Placement]`，可选）：描述张量在 DeviceMesh 上的放置方式，元素数必须等于 `device_mesh.ndim`。若未指定，默认从每个维度的第一个 rank 复制张量。

**关键字参数：**

- `src_data_rank`（`int`，可选）：逻辑/全局张量的源数据 rank。默认使用每个 DeviceMesh 维度上的 `group_rank=0` 作为源数据。若显式传入 `None`，则直接使用本地数据而不通过 scatter/broadcast 保持单设备语义。默认：`0`。

**返回值：** `DTensor` 或 `XLAShardedTensor` 对象。

#### distribute_module

```python
torch.distributed.tensor.distribute_module(
    module, device_mesh=None, partition_fn=None, input_fn=None, output_fn=None
)
```

暴露三个函数来控制模块的参数/输入/输出：

1. 通过 `partition_fn` 在运行前对模块进行分片（将模块参数转换为 DTensor 参数）。
2. 通过 `input_fn` 和 `output_fn` 在运行时控制模块的输入或输出（如将输入转为 DTensor，将输出转回 `torch.Tensor`）。

**参数：**

- `module`（`nn.Module`）：要分区的用户模块。
- `device_mesh`（`DeviceMesh`）：放置模块的设备网格。
- `partition_fn`（`Callable`）：分区参数的函数。未指定时默认在 mesh 上复制所有参数。
- `input_fn`（`Callable`）：指定输入分布。将作为 `forward_pre_hook` 安装。
- `output_fn`（`Callable`）：指定输出分布。将作为 `forward_hook` 安装。

**返回值：** 包含全部 DTensor 参数/缓冲区的模块。

### 5.2 DTensor 工厂函数

DTensor 提供专用的张量工厂函数，通过额外指定 `DeviceMesh` 和 `Placement` 直接创建 DTensor：

#### zeros

```python
torch.distributed.tensor.zeros(
    *size, requires_grad=False, dtype=None, layout=torch.strided,
    device_mesh=None, placements=None
)
```

返回填充标量值 0 的 `DTensor`。

#### ones

```python
torch.distributed.tensor.ones(
    *size, dtype=None, layout=torch.strided, requires_grad=False,
    device_mesh=None, placements=None
)
```

返回填充标量值 1 的 `DTensor`。

#### empty

```python
torch.distributed.tensor.empty(
    *size, dtype=None, layout=torch.strided, requires_grad=False,
    device_mesh=None, placements=None
)
```

返回填充未初始化数据的 `DTensor`。

#### full

```python
torch.distributed.tensor.full(
    size, fill_value, *, dtype=None, layout=torch.strided, requires_grad=False,
    device_mesh=None, placements=None
)
```

返回填充 `fill_value` 的 `DTensor`。

#### rand

```python
torch.distributed.tensor.rand(
    *size, requires_grad=False, dtype=None, layout=torch.strided,
    device_mesh=None, placements=None
)
```

返回填充 `[0, 1)` 均匀分布随机数的 `DTensor`。

#### randn

```python
torch.distributed.tensor.randn(
    *size, requires_grad=False, dtype=None, layout=torch.strided,
    device_mesh=None, placements=None
)
```

返回填充标准正态分布（均值 0，方差 1）随机数的 `DTensor`。

**以上工厂函数的通用关键字参数：**

- `size`（`int...`）：定义输出 DTensor 形状的整数序列。
- `dtype`（`torch.dtype`，可选）：期望的数据类型。默认使用全局默认值。
- `layout`（`torch.layout`，可选）：期望的布局。默认 `torch.strided`。
- `requires_grad`（`bool`，可选）：是否记录 autograd 操作。默认 `False`。
- `device_mesh`：`DeviceMesh` 类型，包含 rank 的 mesh 信息。
- `placements`：`Placement` 类型序列（`Shard`、`Replicate`）。

---

## 六、随机操作

DTensor 提供分布式 RNG 功能，确保分片张量上的随机操作获得唯一值，复制张量上的随机操作获得相同值。此系统要求所有参与 rank（如 SPMD rank）在每次 DTensor 随机操作前使用相同的生成器状态，操作完成后也保持相同状态。随机操作期间不执行 RNG 状态同步通信。

接受 `generator` 关键字参数的算子会使用用户传入的生成器（若有），否则使用设备的默认生成器。使用的生成器在 DTensor 操作后会被推进。同一生成器可同时用于 DTensor 和非 DTensor 操作，但需确保非 DTensor 操作在所有 rank 上等量推进生成器状态。

与 Pipeline Parallelism 配合使用时，每个流水线阶段的 rank 应使用不同的种子，同一流水线阶段内的 rank 应使用相同的种子。

> DTensor 的 RNG 基础设施基于 Philox RNG 算法，支持任何基于 Philox 的后端（CUDA 及类 CUDA 设备），目前尚不支持 CPU 后端。

---

## 七、调试

### 7.1 日志

启动程序时，可通过 `TORCH_LOGS` 环境变量开启额外日志：

| 设置 | 效果 |
|:---|:---|
| `TORCH_LOGS=+dtensor` | 显示 `logging.DEBUG` 及以上级别的消息 |
| `TORCH_LOGS=dtensor` | 显示 `logging.INFO` 及以上级别的消息 |
| `TORCH_LOGS=-dtensor` | 显示 `logging.WARNING` 及以上级别的消息 |

### 7.2 调试工具

#### CommDebugMode

```python
class torch.distributed.tensor.debug.CommDebugMode
```

`CommDebugMode` 是一个上下文管理器，用于统计其上下文中的 functional collective 数量（通过 `TorchDispatchMode` 实现）。

> **注意**：尚未支持所有 collective。

**用法示例：**

```python
mod = ...
comm_mode = CommDebugMode()
with comm_mode:
    mod.sum().backward()
print(comm_mode.get_comm_counts())
```

**方法：**

```python
log_comm_debug_tracing_table_to_file(file_name='comm_mode_log.txt', noise_level=3)
```

将 CommDebugMode 输出写入用户指定的文件（替代控制台输出）。

#### visualize_sharding

```python
torch.distributed.tensor.debug.visualize_sharding(dtensor, header='', use_rich=False)
```

在终端中可视化 1D 或 2D `DTensor` 的分片情况。

> **注意**：需要 `tabulate` 包，或 `rich` 和 `matplotlib`。空张量不会打印分片信息。

---

## 八、实验性功能

DTensor 还提供一组实验性功能，处于原型阶段或基本功能已完成但正在征求用户反馈。如有反馈请向 [PyTorch 提交 Issue](https://github.com/pytorch/pytorch/issues)。

### 8.1 context_parallel

```python
torch.distributed.tensor.experimental.context_parallel(
    mesh, *, buffers=None, buffer_seq_dims=None, no_restore_buffers=None
)
```

启用上下文并行（CP）的实验性 API。执行两个操作：

1. 将 SDPA（`torch.nn.functional.scaled_dot_product_attention`）替换为 CP 版本。
2. 沿序列维度分片 `buffers`，每个 rank 保留对应分片。

> **警告**：此为 PyTorch 原型功能，API 可能变更。

**参数：**

- `mesh`（`DeviceMesh`）：上下文并行的设备网格。
- `buffers`（`Optional[List[torch.Tensor]]`）：依赖序列维度的缓冲区（如输入批次、标签、位置编码缓冲区）。分片会就地（in-place）发生，上下文结束后恢复。不应包含 `nn.Parameter`。
- `buffer_seq_dims`（`Optional[List[int]]`）：缓冲区的序列维度。
- `no_restore_buffers`（`Optional[Set[torch.Tensor]]`）：上下文退出后不恢复的缓冲区集合（必须是 `buffers` 的子集）。

**返回类型：** `Generator[None, None, None]`

### 8.2 local_map

```python
torch.distributed.tensor.experimental.local_map(
    func=None, out_placements=None, in_placements=None,
    in_grad_placements=None, device_mesh=None, *, redistribute_inputs=False
)
```

允许用户将 DTensor 传入为 `torch.Tensor` 编写的函数。它提取 DTensor 的本地分量、调用函数、并根据 `out_placements` 将输出包装为 DTensor。

**参数：**

- `func`（`Callable`）：应用于 DTensor 本地分片的函数。
- `out_placements`：函数扁平化输出中 DTensor 的期望 placement。非 Tensor 输出应为 `None`。
- `in_placements`（可选）：函数扁平化输入中 DTensor 的期望 placement。若指定但不匹配且 `redistribute_inputs=False`，将抛出异常。
- `in_grad_placements`（可选）：输入 DTensor 梯度的 placement 提示。
- `device_mesh`（`DeviceMesh`，可选）：输出 DTensor 的设备网格。默认从第一个输入 DTensor 推断。

**关键字参数：**

- `redistribute_inputs`（`bool`，可选）：输入 DTensor placement 与要求不同时是否重分片。默认 `False`。

**示例：**

```python
>>> @local_map
... def mm_allreduce_forward(device_mesh, W, X):
...     partial_sum_tensor = torch.mm(W, X)
...     reduced_tensor = funcol.all_reduce(partial_sum_tensor, "sum", device_mesh)
...     return reduced_tensor
>>>
>>> # 使用 local_map 包装
>>> local_mm_allreduce_forward = local_map(
...     mm_allreduce_forward,
...     out_placements=[Replicate()],
...     in_placements=[col_wise, row_wise],
...     device_mesh=device_mesh,
... )
>>>
>>> W_dt = distribute_tensor(W, device_mesh, (col_wise))
>>> X_dt = distribute_tensor(X, device_mesh, (row_wise))
>>> Y_dt = local_mm_allreduce_forward(device_mesh, W_dt, X_dt)
```

> **注意**：此 API 目前为实验性，可能变更。

### 8.3 register_sharding

```python
torch.distributed.tensor.experimental.register_sharding(op)
```

允许用户为算子注册分片策略（当张量输入输出为 DTensor 时）。适用场景：

1. 算子没有默认分片策略（如 DTensor 不支持的自定义算子）。
2. 用户想覆盖现有算子的默认分片策略。

**参数：**

- `op`（`Union[OpOverload, List[OpOverload]]`）：要注册自定义分片函数的算子或算子列表。

**返回值：** 函数装饰器，用于包装定义算子分片策略的函数。

**示例：**

```python
>>> @register_sharding(aten._softmax.default)
... def custom_softmax_sharding(x, dim, half_to_float):
...     softmax_dim = dim if dim >= 0 else dim + x.ndim
...     acceptable_shardings = []
...
...     all_replicate = ([Replicate()], [Replicate(), None, None])
...     acceptable_shardings.append(all_replicate)
...
...     for sharding_dim in range(x.ndim):
...         if sharding_dim != softmax_dim:
...             all_sharded = (
...                 [Shard(sharding_dim)],
...                 [Shard(sharding_dim), None, None],
...             )
...             acceptable_shardings.append(all_sharded)
...
...     return acceptable_shardings
```

> **注意**：此 API 目前为实验性，可能变更。

---

## 九、混合 Tensor 与 DTensor 操作

遇到以下错误信息时：

```
got mixed torch.Tensor and DTensor, need to convert all
torch.Tensor to DTensor before calling distributed operators!
```

### 情况 1：用户错误

最常见的原因是创建了普通 Tensor（通过工厂函数）后执行了 Tensor-DTensor 操作：

```python
tensor = torch.arange(10)
return tensor + dtensor  # 错误！
```

DTensor 不允许混合 Tensor-DTensor 操作：如果算子的任何输入是 DTensor，则所有 Tensor 输入都必须是 DTensor。这是因为语义不明确——不知道 `tensor` 在各 rank 上是相同还是不同的。

**如果每个 rank 有相同的 tensor**，构造复制的 DTensor：

```python
tensor = torch.arange(10)
tensor = DTensor.from_local(tensor, placements=(Replicate(),))
return tensor + dtensor
```

**如果要创建分片的 DTensor**（语义上表示数据分布在各分片上，操作作用于"完整堆叠数据"）：

```python
tensor = torch.full([], RANK)
tensor = DTensor.from_local(tensor, placements=(Shard(0),))
return tensor + dtensor
```

### 情况 2：来自 PyTorch 框架代码的错误

有时是 PyTorch 框架代码尝试执行混合 Tensor-DTensor 操作，这属于 PyTorch 的 Bug，请 [提交 Issue](https://github.com/pytorch/pytorch/issues)。

用户端只能避免使用导致问题的操作并提交 Bug 报告。

**PyTorch 开发者的修复方法：**

1. 重写框架代码以避免混合 Tensor-DTensor 操作。
2. 在框架代码的适当位置启用 DTensor 隐式复制（implicit replication）——任何混合操作将假定非 DTensor 可被复制。需谨慎使用，可能导致静默的不正确结果。