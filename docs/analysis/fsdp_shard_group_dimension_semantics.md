# FSDP 分片组维度语义综合报告：为什么是 `dp_shard_cp` 而不是 `dp_shard_cp_tp_pp_ep`

## 目录

- [1. 概述](#1-%E6%A6%82%E8%BF%B0)
- [2. 本轮问题与结论摘要](#2-%E6%9C%AC%E8%BD%AE%E9%97%AE%E9%A2%98%E4%B8%8E%E7%BB%93%E8%AE%BA%E6%91%98%E8%A6%81)
- [3. 源码证据：AReaL / ArchonEngine 如何定义 `dp_shard_cp`](#3-%E6%BA%90%E7%A0%81%E8%AF%81%E6%8D%AEareal--archonengine-%E5%A6%82%E4%BD%95%E5%AE%9A%E4%B9%89-dp_shard_cp)
- [4. 第一层问题：为什么把 CP 并进 FSDP shard group](#4-%E7%AC%AC%E4%B8%80%E5%B1%82%E9%97%AE%E9%A2%98%E4%B8%BA%E4%BB%80%E4%B9%88%E6%8A%8A-cp-%E5%B9%B6%E8%BF%9B-fsdp-shard-group)
- [5. 第二层问题：为什么不继续并入 TP / PP / EP](#5-%E7%AC%AC%E4%BA%8C%E5%B1%82%E9%97%AE%E9%A2%98%E4%B8%BA%E4%BB%80%E4%B9%88%E4%B8%8D%E7%BB%A7%E7%BB%AD%E5%B9%B6%E5%85%A5-tp--pp--ep)
- [6. DeviceMesh 语义：哪些维度适合给 FSDP，哪些维度应保留给模型并行](#6-devicemesh-%E8%AF%AD%E4%B9%89%E5%93%AA%E4%BA%9B%E7%BB%B4%E5%BA%A6%E9%80%82%E5%90%88%E7%BB%99-fsdp%E5%93%AA%E4%BA%9B%E7%BB%B4%E5%BA%A6%E5%BA%94%E4%BF%9D%E7%95%99%E7%BB%99%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C)
- [7. 具体例子：`dp_shard=2, cp=4, tp=2, pp=2, ep=4`](#7-%E5%85%B7%E4%BD%93%E4%BE%8B%E5%AD%90dp_shard2-cp4-tp2-pp2-ep4)
- [8. 通信与调度代价](#8-%E9%80%9A%E4%BF%A1%E4%B8%8E%E8%B0%83%E5%BA%A6%E4%BB%A3%E4%BB%B7)
- [9. 初始化顺序依赖](#9-%E5%88%9D%E5%A7%8B%E5%8C%96%E9%A1%BA%E5%BA%8F%E4%BE%9D%E8%B5%96)
- [10. PyTorch 官方机制与 AReaL 自定义命名的关系](#10-pytorch-%E5%AE%98%E6%96%B9%E6%9C%BA%E5%88%B6%E4%B8%8E-areal-%E8%87%AA%E5%AE%9A%E4%B9%89%E5%91%BD%E5%90%8D%E7%9A%84%E5%85%B3%E7%B3%BB)
- [11. 反模式分析](#11-%E5%8F%8D%E6%A8%A1%E5%BC%8F%E5%88%86%E6%9E%90)
- [12. 最终总结](#12-%E6%9C%80%E7%BB%88%E6%80%BB%E7%BB%93)
- [13. 本 session 验证记录](#13-%E6%9C%AC-session-%E9%AA%8C%E8%AF%81%E8%AE%B0%E5%BD%95)

______________________________________________________________________

## 1. 概述

本报告整理本 session 中围绕 AReaL / ArchonEngine 并行拓扑的两轮追问：

1. 为什么要构造 `dp_shard_cp = flatten(dp_shard, cp)`，让 CP rank 也参与 FSDP 参数、梯度和优化器状态分片？
1. 为什么 FSDP shard group 只扩大到 `dp_shard × cp`，而不是继续把 `tp`、`pp`、`ep` 也并进同一个 FSDP shard
   group？

核心结论是：

> FSDP shard group 能并入哪些维度，取决于这些维度上的 rank 是否仍然拥有“同一批参数的同一逻辑语义”。`dp_shard` 和 `cp`
> 满足这个条件；`tp`、`pp`、`ep` 通常已经改变了参数张量布局、层归属或 expert 归属，因此不能简单并入同一个 dense FSDP shard group。

换句话说，`dp_shard_cp` 的本质不是“CP 变成 DP”，而是：

> CP rank 在 sequence/context 维度上分工计算，同时在模型状态维度上参与 FSDP 分片。

______________________________________________________________________

## 2. 本轮问题与结论摘要

### 2.1 第一轮问题：为什么要有 `dp_shard_cp`

AReaL / ArchonEngine 会构造：

```python
mesh = init_device_mesh(
    device_type,
    (pp, dp_shard, cp, tp),
    mesh_dim_names=("pp", "dp_shard", "cp", "tp"),
)

fsdp_mesh = mesh["dp_shard", "cp"]._flatten(
    mesh_dim_name="dp_shard_cp"
)
```

这个 mesh 作为 FSDP2 `fully_shard()` 的 `mesh` 参数使用。这样 FSDP shard degree 从：

```text
dp_shard
```

扩大为：

```text
dp_shard × cp
```

因此每张 GPU 上持久保存的参数、梯度和优化器状态理论上从：

```text
1 / dp_shard
```

降低到：

```text
1 / (dp_shard × cp)
```

### 2.2 第二轮问题：为什么不是 `dp_shard_cp_tp_pp_ep`

原因是 CP 与 TP / PP / EP 在参数语义上完全不同：

| 维度       | 并行对象                | 是否改变参数语义 | 能否简单并入 dense FSDP shard group |
| ---------- | ----------------------- | ---------------- | ----------------------------------- |
| `dp_shard` | 数据并行 / 参数状态分片 | 否               | 可以                                |
| `cp`       | sequence / context      | 否               | 可以                                |
| `tp`       | 参数矩阵 row / column   | 是               | 通常不可以                          |
| `pp`       | layer / stage           | 是               | 不可以                              |
| `ep`       | expert ownership        | 是               | 不能简单并入                        |

最终一句话：

> 是 `dp_shard_cp`，不是 `dp_shard_cp_tp_pp_ep`，因为 CP 只切输入上下文，仍然保留同一组 dense 参数语义；而 TP、PP、EP
> 分别改变了参数张量布局、层归属和 expert 归属，不能被当作普通 FSDP 数据并行分片维度。

______________________________________________________________________

## 3. 源码证据：AReaL / ArchonEngine 如何定义 `dp_shard_cp`

### 3.1 `ArchonParallelDims` 的语义定义

**文件**: `areal/experimental/models/archon/parallel_dims.py:44-47`

```python
Mesh semantics:
    - total_gpu = pp × dp_shard × cp × tp
    - fsdp_size = dp_shard × cp  (CP ranks participate in weight sharding)
```

这段注释明确给出两个关键事实：

1. ArchonEngine 的全局拓扑由 `pp × dp_shard × cp × tp` 组成。
1. FSDP 分片度不是 `dp_shard`，而是 `dp_shard × cp`。

### 3.2 无 EP 场景下的 mesh 构造

**文件**: `areal/experimental/models/archon/parallel_dims.py:181-204`

```python
dims = [self.pp, self.dp_shard, self.cp, self.tp]
names = ["pp", "dp_shard", "cp", "tp"]

mesh = init_device_mesh(
    self.device_type, tuple(dims), mesh_dim_names=tuple(names)
)

# dp_shard_cp mesh: for FSDP param sharding
self._meshes["dp_shard_cp"] = mesh["dp_shard", "cp"]._flatten(
    mesh_dim_name="dp_shard_cp"
)
```

这里的 `dp_shard_cp` 是 AReaL / ArchonEngine 自定义的子 mesh 名称。它不是 PyTorch 官方固定策略名，但使用的是 PyTorch
官方 `DeviceMesh` 的 slicing 和 flatten 机制。

### 3.3 有 EP 场景下的 mesh 构造

**文件**: `areal/experimental/models/archon/parallel_dims.py:224-259`

```python
if self.etp == self.tp:
    # ETP=TP: ep = dp_shard_in_ep * cp (NOT including tp)
    dp_shard_mod_ep = self.dp_shard * self.cp // self.ep
    dp_shard_in_ep = self.ep // self.cp
else:
    # ETP=1: ep = dp_shard_in_ep * cp * tp
    dp_shard_mod_ep = self.dp_shard * self.cp * self.tp // self.ep
    dp_shard_in_ep = self.ep // (self.cp * self.tp)

# dp_shard_cp mesh: for FSDP param sharding
self._meshes["dp_shard_cp"] = mesh[
    "dp_shard_mod_ep", "dp_shard_in_ep", "cp"
]._flatten(mesh_dim_name="dp_shard_cp")
```

注意：有 EP 时，`dp_shard_cp` 不再物理上直接来自 `mesh["dp_shard", "cp"]`，而是来自：

```text
dp_shard_mod_ep × dp_shard_in_ep × cp
```

但它的逻辑大小仍然是：

```text
dp_shard × cp
```

### 3.4 Qwen3 parallelize 中的 FSDP 使用

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:171-188`

```python
# Apply FSDP
# dp_shard_cp mesh for FSDP sharding of dense params
dp_mesh = parallel_dims.get_mesh("dp_shard_cp")
if dp_mesh is not None:
    # dp_shard_mod_ep mesh for MoE experts FSDP sharding (only when EP enabled)
    dp_mod_ep_mesh = parallel_dims.get_mesh("dp_shard_mod_ep")

    apply_fsdp(
        model,
        dp_mesh,
        ...,
        dp_mod_ep_mesh=dp_mod_ep_mesh,
        gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
    )
```

这说明：

- dense 参数主路径使用 `dp_shard_cp`；
- EP 启用时，MoE experts 另有 `dp_shard_mod_ep` 路径。

______________________________________________________________________

## 4. 第一层问题：为什么把 CP 并进 FSDP shard group

### 4.1 CP 切的是 sequence / context，不是参数矩阵

Context Parallelism 的主要目标是把长上下文序列切到多个 rank 上，例如：

```text
原始 tokens: [B, S]

CP=4 后：
  cp0: [B, S/4]
  cp1: [B, S/4]
  cp2: [B, S/4]
  cp3: [B, S/4]
```

每个 CP rank 处理不同 token 片段，但它们执行的是同一套 transformer 层：

```text
Layer 0 attention
Layer 0 MLP
Layer 1 attention
Layer 1 MLP
...
```

因此 CP rank 之间仍然拥有同一组 dense layer 参数语义。

### 4.2 CP rank 可以共同分片同一批 dense 参数

FSDP 需要的是一组 rank 共同拥有同一个 logical parameter 的不同 shard。CP rank 满足这个条件：

```text
同一 PP stage
同一 TP lane
同一 dense layer
不同 CP rank
```

它们之间的区别在于输入上下文片段不同，而不是参数归属不同。

因此可以构造：

```python
fsdp_mesh = mesh["dp_shard", "cp"]._flatten("dp_shard_cp")
fully_shard(transformer_block, mesh=fsdp_mesh)
```

其工程含义是：

```text
在固定 pp 和 tp 坐标下：
FSDP group size = dp_shard × cp
```

### 4.3 显存收益

假设某个 FSDP unit 的 dense 参数总大小为 `P`，优化器状态大小为 `O`，梯度大小为 `G`。

只用 `dp_shard`：

```text
参数 shard:     P / dp_shard
梯度 shard:     G / dp_shard
优化器状态:     O / dp_shard
```

使用 `dp_shard_cp`：

```text
参数 shard:     P / (dp_shard × cp)
梯度 shard:     G / (dp_shard × cp)
优化器状态:     O / (dp_shard × cp)
```

所以当 `cp=4` 时，在持久模型状态上相对只用 `dp_shard` 又可以理论节省约 4 倍。

______________________________________________________________________

## 5. 第二层问题：为什么不继续并入 TP / PP / EP

### 5.1 为什么 TP 通常不能简单并入同一个 FSDP group

TP，即 Tensor Parallelism，已经改变了参数张量布局。

例如一个线性层权重：

```text
W: [hidden, 4 * hidden]
```

Column Parallel 后：

```text
tp0: W[:, 0:2h]
tp1: W[:, 2h:4h]
```

Row Parallel 后：

```text
tp0: W[0:h/2, :]
tp1: W[h/2:h, :]
```

这意味着 TP rank 之间持有的是同一个 logical weight 的不同 row / column shard。它们不是“同一份参数副本”。

FSDP 的普通 sharding 语义则是：

```text
一组 rank 共同 shard 同一个 logical parameter；
forward 前 all-gather 参数；
backward 后 reduce-scatter 梯度。
```

如果把 TP rank 也直接并进 dense FSDP group，会混淆两层切分：

```text
TP 切的是参数张量的模型并行维度；
FSDP 切的是同一逻辑参数副本的数据并行维度。
```

正确组合方式通常是：

```python
# 先 TP，改变参数布局
parallelize_module(model, tp_mesh, tp_plan)

# 再在每个固定 TP lane 内做 FSDP
fsdp_mesh = mesh["dp_shard", "cp"]._flatten("dp_shard_cp")
fully_shard(block, mesh=fsdp_mesh)
```

换句话说：

> TP 维度保留给张量切分；FSDP 在每个固定 TP coordinate 内跨 `dp_shard × cp` 分片。

### 5.2 为什么 PP 不能并入同一个 FSDP group

PP，即 Pipeline Parallelism，切的是 layer / stage。

例如 `pp=2`：

```text
pp0: layer 0-11
pp1: layer 12-23
```

如果对 `layer 3.weight` 做 FSDP：

```text
layer 3 只存在于 pp0
pp1 根本没有 layer 3
```

此时 PP rank 之间不拥有同一批参数。一个没有某层参数的 rank 不可能参与该参数的 FSDP 分片。

因此 FSDP group 必须限制在同一个 pipeline stage 内：

```text
固定 pp coordinate 后，再在 dp_shard × cp 上做 FSDP。
```

### 5.3 为什么 EP / MoE Expert Parallel 不能简单并入 dense FSDP group

EP，即 Expert Parallelism，改变的是 expert 参数归属。

MoE 模型中可以粗略分为两类参数：

```text
Dense 参数:
  attention weights
  router weights
  layernorm weights
  shared dense weights

Expert 参数:
  expert 0 FFN
  expert 1 FFN
  expert 2 FFN
  expert 3 FFN
```

EP 会把不同 experts 放到不同 rank 或 rank group 上：

```text
ep0: expert 0
ep1: expert 1
ep2: expert 2
ep3: expert 3
```

因此 EP rank 之间并不一定拥有同一组 expert 参数。

AReaL / ArchonEngine 的 Qwen3 FSDP 逻辑明确区分这两类参数：

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:367-390`

```python
for transformer_block in model.layers.values():
    # When EP is enabled, MoE experts are sharded with dp_mod_ep_mesh
    # while the rest of the transformer block uses dp_mesh (dp_shard_cp)
    if (
        getattr(transformer_block, "moe_enabled", False)
        and ep_degree > 1
        and dp_mod_ep_mesh is not None
    ):
        fsdp_ep_config = fsdp_config.copy()
        fsdp_ep_config["mesh"] = dp_mod_ep_mesh

        # FSDP wrap the MoE experts with dp_mod_ep_mesh
        fully_shard(
            transformer_block.moe.experts,
            **fsdp_ep_config,
            ...
        )
```

工程含义是：

```text
dense 参数:  使用 dp_shard_cp
expert 参数: 使用 expert-aware 的单独 mesh / 特殊 FSDP 策略
```

如果把 EP rank 直接混进 dense FSDP group，会把“dense 参数所有 rank 同构”和“expert 参数按 expert
分布”两种语义混在一起，导致参数归属和梯度缩放都更复杂。

______________________________________________________________________

## 6. DeviceMesh 语义：哪些维度适合给 FSDP，哪些维度应保留给模型并行

### 6.1 判断标准

判断某个维度能否并入 FSDP shard group，可以问一个问题：

> 这个维度上的 rank 是否拥有同一个 module / parameter 的同一逻辑语义？

如果答案是“是”，它可以作为 FSDP 数据并行分片维度。

如果答案是“否”，它通常应该保留给 TP / PP / EP 自己的模型并行语义。

### 6.2 维度分类表

| 维度       | rank 之间的差异                 | 参数语义是否同构 | FSDP 处理方式                |
| ---------- | ------------------------------- | ---------------- | ---------------------------- |
| `dp_shard` | 不同数据样本 / 数据并行副本     | 是               | 经典 FSDP shard 维度         |
| `cp`       | 不同 sequence/context 片段      | 是               | 可并入 FSDP shard group      |
| `tp`       | 同一权重的不同 row/column shard | 否               | 固定 TP lane 后再 FSDP       |
| `pp`       | 不同 layer/stage                | 否               | 固定 PP stage 后再 FSDP      |
| `ep`       | 不同 expert ownership           | 否               | experts 使用单独 mesh / 策略 |

### 6.3 DeviceMesh 示意

无 EP 时，Archon 的基础 mesh 可以表示为：

```text
mesh[pp, dp_shard, cp, tp]
```

合理的 FSDP dense 参数 mesh 是：

```text
固定 pp
固定 tp
flatten(dp_shard, cp) -> dp_shard_cp
```

伪代码：

```python
world_mesh = init_device_mesh(
    "cuda",
    (pp, dp_shard, cp, tp),
    mesh_dim_names=("pp", "dp_shard", "cp", "tp"),
)

# 合理：同一 PP stage、同一 TP lane 内扩大 FSDP sharding
fsdp_mesh = world_mesh["dp_shard", "cp"]._flatten("dp_shard_cp")
fully_shard(dense_block, mesh=fsdp_mesh)
```

不合理示意：

```python
# 错误示意：混入已经改变参数语义的维度
bad_mesh = world_mesh["dp_shard", "cp", "tp", "pp"]._flatten(
    "dp_shard_cp_tp_pp"
)
fully_shard(model, mesh=bad_mesh)
```

这里的问题不是 PyTorch 不能表达某种 mesh，而是这个 mesh 不再符合模型参数归属语义。

______________________________________________________________________

## 7. 具体例子：`dp_shard=2, cp=4, tp=2, pp=2, ep=4`

假设某个模型同时启用：

```text
dp_shard = 2
cp       = 4
tp       = 2
pp       = 2
ep       = 4
```

### 7.1 只用 `dp_shard`

```text
FSDP group size = 2
```

在每个固定的：

```text
pp stage
tp lane
cp rank
```

内部，只跨 `dp_shard=2` 做参数分片。

结果：

```text
每卡 dense 参数 shard ≈ 1/2
CP=4 上存在重复的 FSDP shard
```

这能工作，但显存利用不充分。

### 7.2 使用 `dp_shard × cp`

```text
FSDP group size = 2 × 4 = 8
```

在固定：

```text
pp stage
tp lane
dense 参数语义
```

下，让 `dp_shard` 和 `cp` 共同参与 FSDP。

结果：

```text
每卡 dense 参数 shard ≈ 1/8
```

这是 AReaL / ArchonEngine 对 dense 参数采用的合理策略。

### 7.3 错误尝试 `dp_shard × cp × tp`

```text
错误 FSDP group size = 2 × 4 × 2 = 16
```

问题：`tp=2` 上的 rank 不是同一份 dense weight 副本。

例如：

```text
tp0: W[:, 0:half]
tp1: W[:, half:end]
```

TP 已经把 `W` 切成不同张量 shard。如果 FSDP 再把 tp0 和 tp1 当成同一个 ordinary FSDP group 的 peers，就会混淆：

```text
Tensor Parallel shard
vs
FSDP parameter shard
```

### 7.4 错误尝试 `dp_shard × cp × pp`

```text
错误 FSDP group size = 2 × 4 × 2 = 16
```

问题：PP stage 不拥有同一批 layers。

```text
pp0: layer 0-11
pp1: layer 12-23
```

`pp1` 没有 `layer 3.weight`，所以不能参与 `layer 3.weight` 的 FSDP 分片。

### 7.5 错误尝试 `dp_shard × cp × ep`

```text
错误 FSDP group size = 2 × 4 × 4 = 32
```

问题：EP rank 之间拥有不同 expert 参数。

```text
ep0: expert 0
ep1: expert 1
ep2: expert 2
ep3: expert 3
```

expert 参数不是所有 EP rank 都有同一份。因此 expert 参数不能简单使用 dense 参数的 FSDP group。AReaL / ArchonEngine
采取的是：

```text
dense params:  dp_shard_cp
expert params: dp_shard_mod_ep / ep-aware mesh
```

______________________________________________________________________

## 8. 通信与调度代价

### 8.1 `dp_shard_cp` 的收益不是免费的

把 FSDP group 从 `dp_shard` 扩大到 `dp_shard × cp` 后，FSDP 的 collective 也变大：

```text
forward 前:  all-gather 参数
forward 后: reshard 参数
backward 后: reduce-scatter 梯度
```

通信从只跨 `dp_shard` 维度变成跨 `dp_shard × cp` 维度。

### 8.2 主要代价

| 代价                  | 说明                                                           |
| --------------------- | -------------------------------------------------------------- |
| collective group 更大 | all-gather / reduce-scatter 参与 rank 更多                     |
| 网络拓扑更敏感        | flatten 后的 group 可能跨节点或慢链路                          |
| 与 CP 通信竞争        | CP 本身可能需要 Ulysses all-to-all                             |
| overlap 更难          | FSDP pre-forward / post-forward hook 与 attention 通信都要调度 |
| PP 下更敏感           | pipeline microbatch 可能放大反复 all-gather 成本               |

源码中 Qwen3 的 FSDP 逻辑也对 PP 做了特殊处理：

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:346-354`

```python
case "default":
    # For PP, by default do not reshard after forward to avoid per-microbatch
    # all-gathers, which can be expensive and non-overlapped
    reshard_after_forward = not pp_enabled
```

这说明扩大 FSDP group 后，通信调度不是附属问题，而是核心性能风险之一。

______________________________________________________________________

## 9. 初始化顺序依赖

`dp_shard_cp` 也解释了为什么 CP 必须在 FSDP 之前确定。

AReaL / ArchonEngine 的典型顺序是：

```text
TP / EP
  → CP
    → AC
      → torch.compile
        → FSDP
```

Qwen3 编排代码中对应步骤为：

**文件**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:134-173`

```python
# Apply non-MoE TP first
apply_non_moe_tp(...)

# Apply MoE EP+TP
apply_moe_ep_tp(...)

# Apply CP
apply_cp(...)

# AC must be after TP/CP
apply_ac(...)

# torch.compile must be after AC, before FSDP
_apply_compile(...)

# Apply FSDP
# dp_shard_cp mesh for FSDP sharding of dense params
dp_mesh = parallel_dims.get_mesh("dp_shard_cp")
```

原因是：

1. TP / EP 会先改变参数或 expert 的布局语义。
1. CP 要先确定 sequence/context 通信组。
1. FSDP `fully_shard()` 会把传入的 mesh 固化进参数 DTensor / FSDPModule hook。
1. 如果 FSDP 之后才决定 CP 是否参与分片，就需要重建参数 shard layout，而不是简单添加一个 process group。

______________________________________________________________________

## 10. PyTorch 官方机制与 AReaL 自定义命名的关系

### 10.1 官方机制

PyTorch 提供：

- `DeviceMesh` 多维设备拓扑；
- mesh slicing，例如 `mesh["dp_shard", "cp"]`；
- flatten 子 mesh，例如 `._flatten(mesh_dim_name="dp_shard_cp")`；
- FSDP2 `fully_shard(module, mesh=...)`；
- DTensor placement，例如 `Shard(...)`、`Replicate()`、`Partial()`。

这些是官方机制。

### 10.2 AReaL 自定义策略

`dp_shard_cp` 这个名字不是 PyTorch 官方固定名称，而是 AReaL / ArchonEngine 为自己的拓扑语义定义的名字。

更准确地说：

```text
PyTorch 提供 DeviceMesh / DTensor / fully_shard 机制；
AReaL 决定把 dp_shard 和 cp flatten 成名为 dp_shard_cp 的 FSDP mesh。
```

### 10.3 注意：不要说 FSDP2 只能接受 1D mesh

本 session 中 code-reviewer 子代理指出一个重要措辞风险：当前 PyTorch FSDP2 并不是只能接受 1D mesh，2D mesh 可用于
HSDP 语义。

AReaL / ArchonEngine 这里 flatten `dp_shard × cp` 的原因更准确地说是：

> Archon 当前 `dp_replicate=1`，使用的是纯 FSDP sharding 语义；因此把 `dp_shard × cp` flatten 成一个用于
> dense 参数分片的一维 mesh，而不是因为 PyTorch 完全不能处理二维 mesh。

______________________________________________________________________

## 11. 反模式分析

### 11.1 反模式 A：只用 `dp_shard`

```python
fsdp_mesh = mesh["dp_shard"]
fully_shard(block, mesh=fsdp_mesh)
```

问题：

```text
CP rank 之间重复持有模型状态 shard，显存节省不足。
```

适用场景：

```text
模型较小、通信瓶颈强、CP 只想用于 attention 计算而不想扩大 FSDP group。
```

但这不是 ArchonEngine 当前 dense 参数的默认设计。

### 11.2 反模式 B：把 TP 并入 dense FSDP group

```python
bad_mesh = mesh["dp_shard", "cp", "tp"]._flatten("dp_shard_cp_tp")
fully_shard(block, mesh=bad_mesh)
```

问题：

```text
TP rank 已经持有不同 row/column shard；
FSDP 又把它们当成同一 logical parameter 的 sharding peers；
两种参数切分语义冲突。
```

### 11.3 反模式 C：把 PP 并入 dense FSDP group

```python
bad_mesh = mesh["pp", "dp_shard", "cp"]._flatten("pp_dp_shard_cp")
fully_shard(model, mesh=bad_mesh)
```

问题：

```text
不同 PP rank 拥有不同 layer；
某些 rank 根本没有目标参数，不能参与该参数的 FSDP shard。
```

### 11.4 反模式 D：把 EP 并入统一 dense FSDP group

```python
bad_mesh = mesh["dp_shard", "cp", "ep"]._flatten("dp_shard_cp_ep")
fully_shard(model, mesh=bad_mesh)
```

问题：

```text
EP rank 的 expert ownership 不同；
expert 参数和 dense 参数需要不同 sharding 语义；
统一 group 会混淆 expert 参数归属与 dense 参数分片。
```

______________________________________________________________________

## 12. 最终总结

### 12.1 哪些维度可以扩大 FSDP 数据并行分片组

可以扩大 FSDP shard group 的维度，必须满足：

```text
不改变参数张量布局；
不改变 layer 归属；
不改变 expert ownership；
rank 之间仍然对应同一批 module / parameter 语义。
```

在本讨论中，这类维度是：

```text
dp_shard
cp
```

### 12.2 哪些维度不能直接并入同一个 dense FSDP shard group

不能直接并入的维度包括：

```text
tp: 切参数张量 row/column
pp: 切 layer/stage
ep: 切 expert ownership
```

它们不是“更多可用的数据并行 rank”，而是已经承担了模型并行语义。

### 12.3 一句话结论

> AReaL / ArchonEngine 使用 `dp_shard_cp`，而不是 `dp_shard_cp_tp_pp_ep`，因为 CP 只切
> sequence/context，仍然保留同一组 dense 参数语义；而 TP、PP、EP 分别改变了参数张量布局、层归属和 expert 归属，不能被当作普通 FSDP
> 数据并行分片维度。

______________________________________________________________________

## 13. 本 session 验证记录

本报告基于本 session 的两轮问答、四个子代理分析结果和本地源码核验整理。

### 13.1 子代理结果

本 session 第一轮分析中使用并等待了四个子代理：

| 子代理        | 主要结论                                                                                                               |
| ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Explore       | 确认 `dp_shard_cp` 定义于 `parallel_dims.py`，并被 Qwen2 / Qwen3 / Qwen3.5 parallelize 路径传给 FSDP。                 |
| architect     | 确认 CP rank 是训练参与者，不只是 attention helper；`dp_shard × cp` 降低模型状态显存但扩大 FSDP collective。           |
| code-reviewer | 指出“FSDP2 只能接受 1D mesh”表述不准确；应强调 AReaL 当前选择 flatten 是纯 FSDP sharding 策略，而非 PyTorch 能力限制。 |
| analyst       | 梳理中文回答结构，强调 CP / TP / PP / EP 的参数语义差异。                                                              |

### 13.2 本地源码核验

本 session 中执行过本地 grep / `nl -ba` 核验，确认：

- `areal/experimental/models/archon/parallel_dims.py:44-47` 明确写出
  `fsdp_size = dp_shard × cp`。
- `areal/experimental/models/archon/parallel_dims.py:201-204` 无 EP 时构造
  `mesh["dp_shard", "cp"]._flatten("dp_shard_cp")`。
- `areal/experimental/models/archon/parallel_dims.py:256-259` 有 EP 时通过
  `dp_shard_mod_ep × dp_shard_in_ep × cp` 构造逻辑 `dp_shard_cp`。
- `areal/experimental/models/archon/qwen3/infra/parallelize.py:171-188` 将 `dp_shard_cp`
  作为 dense FSDP mesh。
- `areal/experimental/models/archon/qwen3/infra/parallelize.py:367-390` 在 EP 启用时对 MoE
  experts 使用 `dp_mod_ep_mesh` 单独处理。

### 13.3 文档风格对齐

本报告参考 `docs/analysis/parallelism_initialization_order_analysis.md` 的行文规范：

- 使用中文标题与编号章节；
- 在开头提供目录；
- 每个关键结论配源码文件路径和代码片段；
- 同时包含工程推理、DeviceMesh 示意、反模式分析和最终总结。
