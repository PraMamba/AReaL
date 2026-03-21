# DeviceMesh + DTensor：理解 "PyTorch-native" 的关键

---

## 一、设计哲学：为什么需要 DTensor

### 1.1 核心问题

分布式训练中，张量分散在多张卡上。传统方案（如 Megatron-LM）需要手动插入 all-reduce/all-gather——代码跟单卡版本完全不同。

PyTorch DTensor 的设计目标：**写分布式代码就像写单卡代码**。

> "Simple SPMD primitive + single-device semantic"
> —— DTensor 设计文档（灵感来自 GSPMD / OneFlow / TF DTensor）

### 1.2 两层抽象

```
┌─────────────────────────────────────────────────────┐
│  DeviceMesh                                         │
│  = N 维设备拓扑 + 每个维度的 ProcessGroup            │
│  解决: "哪些 GPU 之间要通信，走哪个维度"              │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│  DTensor                                            │
│  = torch.Tensor 子类 + DeviceMesh + Placement       │
│  解决: "这个 tensor 在各 GPU 上怎么分布的"            │
│  自动为算子插入通信                                   │
└─────────────────────────────────────────────────────┘
```

---

## 二、DeviceMesh 详解

### 2.1 什么是 DeviceMesh

DeviceMesh = **设备的 N 维数组** + **每个维度自动创建的 ProcessGroup**。

```python
from torch.distributed.device_mesh import init_device_mesh

# 1D mesh: 2 个 GPU 组成的简单线性拓扑
mesh_1d = init_device_mesh("cuda", (2,))
# 内部: ranks = [0, 1], 一个 ProcessGroup 覆盖两者

# 2D mesh: 4 个 GPU 排成 2×2
mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
# 内部排布:
#        tp=0  tp=1
# dp=0 [ GPU0  GPU1 ]    ← tp_group_0 = {0, 1}
# dp=1 [ GPU2  GPU3 ]    ← tp_group_1 = {2, 3}
#        │      │
#        dp_group_0={0,2}  dp_group_1={1,3}
```

### 2.2 DeviceMesh 取子 mesh

```python
tp_mesh = mesh_2d["tp"]   # 每个 rank 得到自己所在的 tp 子 mesh
dp_mesh = mesh_2d["dp"]   # 每个 rank 得到自己所在的 dp 子 mesh

# 例如 rank=1:
#   tp_mesh 包含 {0, 1}
#   dp_mesh 包含 {1, 3}
```

### 2.3 DeviceMesh 与上层 API 的关系

| 上层 API | 接收什么 mesh | 在哪个维度通信 |
|---------|-------------|-------------|
| `fully_shard(model, mesh=dp_mesh)` | 1D dp mesh | all-gather / reduce-scatter 走 dp 维 |
| `parallelize_module(model, tp_mesh, plan)` | 1D tp mesh | all-reduce / all-gather 走 tp 维 |
| `fully_shard(model, mesh=mesh_2d)` | 2D mesh | HSDP: 0维replicate + 1维shard |

---

## 三、DTensor 的三种 Placement

DTensor 的布局由 `(DeviceMesh, [Placement, ...])` 完整描述。每个 mesh 维度对应一个 Placement。

### 3.1 Shard(dim)

张量在 `dim` 维度上被切片分给各 rank。

```
全局逻辑张量 (4×4):                    Shard(dim=1) 在 2 个 rank:
┌──────────────────┐
│ a  b │ c  d      │                   Rank 0 持有:     Rank 1 持有:
│ e  f │ g  h      │                   ┌───────┐        ┌───────┐
│ i  j │ k  l      │      ───→         │ a  b  │        │ c  d  │
│ m  n │ o  p      │                   │ e  f  │        │ g  h  │
└──────────────────┘                   │ i  j  │        │ k  l  │
                                       │ m  n  │        │ o  p  │
                                       └───────┘        └───────┘
                                       shape: (4,2)     shape: (4,2)
```

**关键**：`dtensor.shape` 返回的是**全局逻辑 shape** `(4,4)`，不是本地 shape。
用 `dtensor.to_local()` 才能拿到本地 shard 的 `(4,2)` tensor。

### 3.2 Replicate()

每个 rank 都持有完整副本。

```
全局张量 (4×4):                    Replicate() 在 2 个 rank:
┌────────────┐
│ a  b  c  d │                     Rank 0 持有:     Rank 1 持有:
│ e  f  g  h │       ───→          (完整 4×4)       (完整 4×4)
│ i  j  k  l │                     完全一样          完全一样
│ m  n  o  p │
└────────────┘
```

### 3.3 Partial(reduce_op)

每个 rank 持有**部分值**（shape 相同），需要做 reduce 才能得到正确结果。

```
全局张量应该是 A+B:

Rank 0 持有: A (4×4)      Rank 1 持有: B (4×4)
┌────────────┐            ┌────────────┐
│ 部分值...   │            │ 部分值...   │
└────────────┘            └────────────┘
     └───── all_reduce(SUM) ─────┘  →  完整张量 A+B
```

**Partial 什么时候出现？** 矩阵乘法！当 `Y = X @ W` 中 W 按行切（RowwiseParallel），每个 rank 算出的 Y 只是部分和，需要 all-reduce 才能得到完整 Y。

### 3.4 三种 Placement 之间的转换 = 通信

| 从 → 到 | 需要的通信 | 用在哪 |
|---------|----------|-------|
| Shard → Replicate | **all-gather** | FSDP forward 前拼回完整参数 |
| Replicate → Shard | **本地 chunk** (无通信) | FSDP 初始化时分片 |
| Partial → Replicate | **all-reduce** | TP 的 RowwiseParallel 输出 |
| Partial → Shard | **reduce-scatter** | FSDP backward 的梯度分片 |
| Shard(0) → Shard(1) | **all-to-all** | 改变切分维度 |

这就是 DTensor `redistribute()` 的核心逻辑。

---

## 四、DTensor 如何让算子"自动正确"

### 4.1 Sharding Propagation

当你对 DTensor 调用 PyTorch 算子（如 `torch.mm`），DTensor 会：

1. 查 `OpStrategy`：这个算子支持哪些 (input_placement → output_placement) 组合
2. 选最优策略（通信代价最低的）
3. 如果当前 placement 不匹配，先 `redistribute`（插入通信）
4. 在本地 tensor 上执行算子
5. 包装输出为新的 DTensor，带正确的 placement

**例子：矩阵乘法 Y = X @ W**

```
情况: X 是 Replicate, W 是 Shard(dim=0)

策略选择:
  输入: X=Replicate, W=Shard(0)
  本地计算: Y_local = X_local @ W_local   (每个 rank 算部分)
  输出: Y=Partial(SUM)                    (需要 reduce 才完整)

如果下游需要 Y 是 Replicate:
  → 自动插入 all-reduce
如果下游需要 Y 是 Shard(0):
  → 自动插入 reduce-scatter
```

### 4.2 为什么这是 "single-device semantic"

你的代码长这样：

```python
# 看起来跟单卡一模一样！
output = model(input)
loss = F.cross_entropy(output, target)
loss.backward()
optimizer.step()
```

通信全部被 DTensor **封装在算子内部**。你永远不需要手动写 `dist.all_reduce()`。

---

## 五、Tensor Parallel 如何利用 DTensor

### 5.1 ColwiseParallel：按列切权重

```
nn.Linear(D_in=4, D_out=4) 的权重 W shape = (4, 4)

ColwiseParallel 在 2 个 rank 上:
  W 变为 DTensor, placement = Shard(0)  ← 按 out_features 维度切

  Rank 0 持有 W 的前2行: W[:2, :] shape=(2,4)
  Rank 1 持有 W 的后2行: W[2:, :] shape=(2,4)

  forward: input(Replicate) @ W(Shard(0))^T
    → 每个 rank 算出 output 的一部分列
    → output placement = Shard(-1)  (在最后维度上 sharded)
```

### 5.2 RowwiseParallel：按行切权重

```
nn.Linear(D_in=4, D_out=4) 的权重 W shape = (4, 4)

RowwiseParallel 在 2 个 rank 上:
  W 变为 DTensor, placement = Shard(1)  ← 按 in_features 维度切

  Rank 0 持有 W 的左半: W[:, :2] shape=(4,2)
  Rank 1 持有 W 的右半: W[:, 2:] shape=(4,2)

  forward: input(Shard(-1)) @ W(Shard(1))^T
    → 每个 rank 算出 output 的部分和
    → output placement = Partial(SUM)
    → 自动 all-reduce → output 变为 Replicate
```

### 5.3 经典配对：ColwiseParallel + RowwiseParallel

这就是 Megatron-LM 论文的核心思想，用 DTensor 语言表达：

```
input [Replicate]
    │
    ▼ ColwiseParallel(w1)
output_1 [Shard(-1)]       ← 每个 rank 只有部分列, 无需通信
    │
    ▼ activation (SiLU 等)
    │
    ▼ RowwiseParallel(w2)   ← 输入 Shard(-1) 正好匹配!
output_2 [Partial]          ← 部分和
    │
    ▼ all-reduce (自动)
output_2 [Replicate]        ← 完整结果
```

**整个 MLP 只需要一次 all-reduce！** 这就是 TP 高效的原因。

---

## 六、parallelize_module 的实际工作

```python
parallelize_module(model, tp_mesh, {
    "w1": ColwiseParallel(),
    "w2": RowwiseParallel(),
})
```

这行代码做了什么：

1. **找到 `model.w1`** 这个 `nn.Linear`
2. **把 `w1.weight` 转成 DTensor**，placement = `Shard(0)` (列切)
3. **把 `w1.bias` 转成 DTensor**，placement = `Shard(0)`
4. **注册 input hook**：forward 前把输入转成 `Replicate` 的 DTensor
5. **注册 output hook**：forward 后按 `output_layouts` 处理输出

对 `w2` (RowwiseParallel) 同理，但 placement 不同。

**结果**：`model.w1.weight` 不再是普通 Tensor，而是一个 DTensor：

```python
param = model.w1.weight
print(type(param))         # <class 'torch.distributed.tensor.DTensor'>
print(param.shape)          # torch.Size([8, 4])  ← 全局 shape!
print(param.placements)     # (Shard(dim=0),)
print(param.to_local().shape)  # torch.Size([4, 4])  ← 本地 shard
```

---

## 七、2D Mesh 下 FSDP + TP 是怎么组合的

```python
mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))

# TP 用 tp 子 mesh
tp_mesh = mesh_2d["tp"]
parallelize_module(block, tp_mesh, tp_plan)

# FSDP 用整个 2D mesh (或 dp 子 mesh)
fully_shard(block, mesh=mesh_2d)
```

此时一个参数的 DTensor placement 可能是 **2 维的**：

```
mesh_2d shape = (2, 2), dim_names = ("dp", "tp")

参数 placement = (Shard(0), Shard(0))
  意思: 在 dp 维度上按 dim-0 分片 (FSDP)
        在 tp 维度上按 dim-0 分片 (TP column-wise)

或者: (Replicate(), Shard(0))
  意思: 在 dp 维度上复制 (DDP 语义)
        在 tp 维度上按 dim-0 分片 (TP column-wise)
```

**这就是"PyTorch-native composability"**——不同并行方案通过 mesh 维度正交组合。

---

## 八、推荐学习资源

### 必读官方文档

1. **DTensor API Reference** — Placement 定义、API 全集
   https://docs.pytorch.org/docs/stable/distributed.tensor.html

2. **DeviceMesh Tutorial** — mesh 创建、子 mesh、HSDP 示例
   https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html

3. **Tensor Parallel Tutorial (Llama 2)** — 完整 TP plan 示例
   https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

4. **FSDP2 Tutorial** — DTensor 在 FSDP 中的具体表现
   https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html

5. **DTensor README (PyTorch repo)** — 设计动机与 GSPMD 对比
   https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/README.md

### 深入理解

6. **Understanding DTensor Redistribute** (Kavya G) — redistribute 的 7 种路径
   http://gkavya.in/dtensor-redistribute/

7. **DTensor DeepWiki** — dispatch/propagation/autograd 全流程
   https://deepwiki.com/pytorch/pytorch/4.2-symmetric-memory-system

---

## 九、附带实验说明

### 实验 1: `dtensor_shard_demo.py` (2 卡)

创建 1D mesh，把一个 `4×6` 矩阵按列 `Shard(1)` 分到 2 个 rank，验证：
- `dtensor.shape` 是全局 shape `(4, 6)`
- `dtensor.to_local().shape` 是本地 shard `(4, 3)`
- `redistribute` 到 `Replicate` 后每个 rank 都有完整矩阵
- 两个 DTensor 相乘时 placement 如何自动传播

### 实验 2: `tp_linear_demo.py` (4 卡)

创建 2D mesh `(2, 2)` = `[dp, tp]`，用 `parallelize_module` 对一个两层 MLP 做 TP，验证：
- `parallelize_module` 后参数变成 DTensor
- 参数的 `placements` 和本地 shape 的变化
- forward 产生正确的输出（与单卡对比）

```bash
# 实验 1
torchrun --nproc_per_node=2 dtensor_shard_demo.py

# 实验 2
torchrun --nproc_per_node=4 tp_linear_demo.py
```
