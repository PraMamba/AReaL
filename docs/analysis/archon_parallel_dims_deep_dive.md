# Archon ParallelDims 深度解析

> 源文件：`areal/experimental/models/archon/parallel_dims.py`（419 行）
> 核心类：`ArchonParallelDims`

---

[TOC]

---

# 1. 通俗解释

## 1.1 一句话总结

`ArchonParallelDims` 是 Archon 引擎的 **GPU 拓扑代数系统**——它接收用户的并行配置（TP=2, CP=2, PP=4 等），自动将所有 GPU 排列成一个多维网格（DeviceMesh），并为每种并行策略派发正确的 GPU 分组。

## 1.2 类比理解

想象你有 32 台快递分拣机器（GPU），需要同时执行 5 种不同的协作模式：

| 协作模式 | 现实类比 | 并行策略 |
| --- | --- | --- |
| **流水线** | 工位 A 拧螺丝 → 工位 B 焊接 → 工位 C 质检 | PP (Pipeline Parallel) |
| **数据分片** | 每个工人拿一部分零件加工 | DP (FSDP Data Parallel) |
| **张量切分** | 一块大钢板切成 4 片，每人焊一片 | TP (Tensor Parallel) |
| **序列切分** | 一篇长文档分成 4 段，每人读一段 | CP (Context Parallel / Ulysses) |
| **专家分配** | MoE 模型中不同专家分给不同机器 | EP (Expert Parallel) |

`ArchonParallelDims` 的工作就是：给定每种模式需要多少台机器，把 32 台机器排成一个高维网格，让每种模式都能找到自己的"队友"。

## 1.3 核心不变量

```
world_size = pp × dp_shard × cp × tp
```

所有 GPU 恰好被分配一次，无冗余、无遗漏。

## 1.4 这个文件解决的核心问题

1. **配置验证**：用户输入 `tp=3, cp=2, world_size=8` 会在初始化时立即报错（`3×2` 无法整除 8）。
2. **Mesh 构建**：将一维 GPU 列表 `[0,1,...,31]` 变成多维网格 `mesh[pp][dp][cp][tp]`。
3. **子网格派发**：为 FSDP 提供 `dp_shard_cp` 网格，为 Ulysses 提供 `cp` 进程组，��流水线提供 `pp` 网格——每个并行模块只需调 `get_mesh("名称")`。
4. **EP 维度借用**：MoE 模型启用专家并行时，EP 不新增 GPU，而是从已有维度"借用"（重新解释 DP/CP/TP 维度的 GPU 角色）。

---

# 2. 基础概念

## 2.1 DeviceMesh：多维 GPU 网格

PyTorch 的 `init_device_mesh` 将物理 GPU 排列成逻辑上的 N 维张量。例如 8 个 GPU 排成 `(pp=1, dp_shard=2, cp=2, tp=2)`：

```text
mesh[pp=0]:
              tp=0    tp=1
  dp=0, cp=0  GPU0    GPU1     ← TP 组: {GPU0, GPU1}
  dp=0, cp=1  GPU2    GPU3     ← TP 组: {GPU2, GPU3}
  dp=1, cp=0  GPU4    GPU5     ← TP 组: {GPU4, GPU5}
  dp=1, cp=1  GPU6    GPU7     ← TP 组: {GPU6, GPU7}
```

**三种核心操作：**

| 操作 | 语法 | 含义 |
| --- | --- | --- |
| **取子网格** | `mesh["tp"]` | 沿 TP 维度切出 1D 子网格，同一组内的 GPU 做张量并行通信 |
| **多维切片** | `mesh["dp_shard", "cp"]` | 取出 2D 子网格，包含同一 PP 和 TP 位置但不同 DP/CP 的所有 GPU |
| **展平** | `._flatten(mesh_dim_name="dp_shard_cp")` | 将多维子网格压成 1D，注册为新名称供 FSDP 等模块使用 |

**为什么需要展平？** PyTorch 的 `fully_shard()` 和集体通信原语只接受 1D mesh。当 FSDP 需要跨 `dp_shard × cp` 维度分片权重时，必须先把这两个维度压成一维。

## 2.2 5D 并行维度

Archon 支持 5 种并行策略的任意组合：

```text
┌────────────────────────────────────────────────────────────────┐
│                    5D 并行维度全景图                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  PP (Pipeline Parallel)                                        │
│  ├── 模型按层切成多个 stage，每个 stage 在不同 GPU 上           │
│  ├── 通信：Send/Recv 传递激活值                                 │
│  └── 好处：突破单 GPU 显存限制                                  │
│                                                                │
│  DP (FSDP Data Parallel, dp_shard)                             │
│  ├── 每个 GPU 持有完整模型权重的一个分片                        │
│  ├── 通信：All-Gather 聚合权重 + Reduce-Scatter 聚合梯度        │
│  └── 好处：降低每 GPU 的权重显存占用                            │
│                                                                │
│  CP (Context Parallel / Ulysses SP)                            │
│  ├── 序列维度切分，但在注意力头维度上聚合                       │
│  ├── 通信：All-to-All（头↔序列交换）                            │
│  └── 好处：支持超长序列训练                                     │
│                                                                │
│  TP (Tensor Parallel)                                          │
│  ├── 权重矩阵按列/行切分到多个 GPU                              │
│  ├── 通信：All-Reduce（前向/反向同步结果）                       │
│  └── 好处：单层计算并行化                                       │
│                                                                │
│  EP (Expert Parallel)                                          │
│  ├── MoE 专家分布到不同 GPU，每个 GPU 持有部分专家              │
│  ├── 通信：All-to-All（token dispatch/combine）                 │
│  └── 好处：支持超大规模 MoE 模型                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 2.3 EP 维度"借用"机制

EP 不是一个独立的新维度。它通过**重新解释**已有维度的 GPU 角色来实现。这是本文件最核心的设计创新。

### 无 EP 时的 4D Mesh

```text
维度:  [pp, dp_shard, cp, tp]
GPU:   world_size = pp × dp_shard × cp × tp
```

### 有 EP 时的 5D Mesh

dp_shard 被拆成两部分：

```text
维度:  [pp, dp_shard_mod_ep, dp_shard_in_ep, cp, tp]

dp_shard_mod_ep  →  仍然做数据并行的 GPU（同一专家子集的不同数据副本）
dp_shard_in_ep   →  被 EP "借走" 的 GPU（持有不同专家子集）
```

**两种 ETP 模式的本质区别：**

| 模式 | EP 借用的维度 | 专家权重切分 | EP 组构成 |
| --- | --- | --- | --- |
| `etp=1`（TP 被吞并） | `dp_shard × cp × tp` | `[Shard(0)]` 仅按专家维度 | `dp_shard_in_ep × cp × tp` |
| `etp=tp`（TP 保持独立） | `dp_shard × cp` | `[Shard(0), Shard(1/2)]` 2D 切分 | `dp_shard_in_ep × cp` |

### 具体 GPU 分配示例

**配置**: `pp=1, dp_shard=2, cp=1, tp=2, ep=2, world_size=4`

```text
无 EP 的逻辑布局:
              tp=0    tp=1
  dp_shard=0  GPU0    GPU1     ← TP 组: {GPU0, GPU1}
  dp_shard=1  GPU2    GPU3     ← TP 组: {GPU2, GPU3}
                ↑               ↑
          FSDP 组:          FSDP 组:
          {GPU0, GPU2}      {GPU1, GPU3}
```

**Case A: etp=1（TP 被 EP 借用）**

```text
公式: dp_shard_mod_ep = 2×1×2/2 = 2,  dp_shard_in_ep = 2/(1×2) = 1
EP 组 = flatten(dp_shard_in_ep=1, cp=1, tp=2) = 大小 2

Dense 层:  GPU0, GPU1 做 TP（各持一半权重矩阵）
Expert 层: GPU0, GPU1 变成 EP 组（各持不同专家）
           TP 关系暂停，权重仅用 Shard(0) 切分

  EP 组 A: {GPU0, GPU1}  ← 原来的 TP 伙伴
  EP 组 B: {GPU2, GPU3}  ← 原来的 TP 伙伴
```

**Case B: etp=tp=2（TP 保持独立）**

```text
公式: dp_shard_mod_ep = 2×1/2 = 1,  dp_shard_in_ep = 2/1 = 2
EP 组 = flatten(dp_shard_in_ep=2) = 大小 2

Expert 层: GPU0, GPU2 做 EP（各持不同专家）
           GPU0, GPU1 仍做 TP（各持专家权重的不同切片）
           权重用 [Shard(0), Shard(1/2)] 2D 切分

  EP 组 A: {GPU0, GPU2}  ← 原来的 dp_shard 伙伴
  EP 组 B: {GPU1, GPU3}  ← 原来的 dp_shard 伙伴
  TP 组 (仍活跃): {GPU0, GPU1} 和 {GPU2, GPU3}
```

**核心权衡**：`etp=1` 获得更大的 EP 组（借用更多维度），但失去了专家层的张量并行；`etp=tp` 保留专家的 TP，但 EP 池更小。

## 2.4 复合 Mesh 语义

| Mesh 名称 | 构成 | 用途 | 使用方 |
| --- | --- | --- | --- |
| `dp` | `dp_shard`（无 EP）或 `dp_shard_mod_ep × dp_shard_in_ep` | 数据加载：每个 dp rank 加载不同 micro-batch | `ArchonEngine` |
| `dp_shard_cp` | `dp_shard × cp` | FSDP 权重分片：CP rank 也参与权重分片以节省显存 | `parallelize_qwen2/3` |
| `dp_cp` | `dp_shard × cp` | Loss all-reduce：CP rank 也需参与梯度规约 | `ArchonEngine` |
| `pp_cp_tp` | `pp × cp × tp` | 模型并行广播组：同组 GPU 处理相同数据批次 | `ArchonEngine` |
| `ep` | `dp_shard_in_ep × cp [× tp]` | Expert 并行：All-to-All token dispatch | `parallelize_qwen3` |
| `ep_tp` | `[ep, tp]`（2D，仅 etp=tp 时） | Expert 的 2D 权重切分 | `ExpertTensorParallel` |
| `dp_shard_mod_ep` | （仅 EP 启用时） | Expert 参数的 FSDP 分片 | `parallelize_qwen3` |

**为什么 `dp_shard_cp` 和 `dp_cp` 是同一组 GPU 却要分开命名？**

当前 `dp_replicate=1`（无 HSDP），两者物理上相同。但语义不同：
- `dp_shard_cp`：FSDP 参数分片用
- `dp_cp`：Loss/梯度规约用

如果未来支持 HSDP（`dp_replicate > 1`），`dp_cp` 需要包含 replicate 维度（所有数据并行 GPU 都要规约梯度），而 `dp_shard_cp` 不需要。提前分离避免了将来的重构。

## 2.5 懒加载构建模式

```text
                    ArchonParallelDims 生命周期
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Phase 1: 配置阶段                                              │
│  ├── __init__ / __post_init__ 执行                              │
│  ├── 纯算术验证（不需要 GPU，不需要 dist.init_process_group）    │
│  ├── 可在 CPU 控制节点创建、序列化、通过 RPC 传递                │
│  └── Mesh 尚未构建（_world_mesh = None）                        │
│                                                                │
│  Phase 2: 首次访问 world_mesh                                   │
│  ├── property 触发 build_mesh()                                 │
│  ├── 调用 init_device_mesh()（需要 dist 已初始化）               │
│  ├── 构建所有子网格并缓存到 _meshes 字典                         │
│  └── 后续访问直接返回缓存                                       │
│                                                                │
│  Phase 3: 使用阶段                                              │
│  ├── get_mesh("名称") → 返回缓存的子网格或 None                  │
│  ├── get_group("名称") → 返回进程组或 None                       │
│  └── 各 property (tp_enabled, pp_enabled...) 纯布尔判断          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**为什么懒加载？** 分离配置验证和分布式运行时。`ArchonParallelDims` 可能在调度器的 CPU 节点上被创建，然后传到 GPU Worker 上才真正构建 Mesh。

---

# 3. 逐行源码映射

## 3.1 导入与日志（1-21 行）

```python
# 第 1 行: 说明代码源自 TorchTitan 项目
# Adapted from torchtitan: torchtitan/distributed/parallel_dims.py

# 第 3-4 行: functools.cache 用于日志单例，dataclass 用于配置声明
import functools
from dataclasses import dataclass, field

# 第 6-7 行: PyTorch 分布式 API
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# 第 9 行: AReaL 自定义日志（非 stdlib logging）
from areal.utils import logging

# 第 12-16 行: Rank 感知日志
# @functools.cache 确保每个进程只创建一个 logger 实例
# 日志名格式: "[Archon ParallelDims Rank 0]"
@functools.cache
def _get_logger() -> logging.Logger:
    rank = dist.get_rank() if dist.is_initialized() else 0
    return logging.getLogger(f"[Archon ParallelDims Rank {rank}]")
```

## 3.2 类定义与字段（24-121 行）

```python
@dataclass
class ArchonParallelDims:
    # ── 用户配置字段 ──────────────────────────────────────────
    dp_replicate: int = 1   # 109 行: HSDP 复制维度，暂不支持，固定为 1
    dp_shard: int = -1      # 110 行: FSDP 分片维度，-1 表示自动计算
    cp: int = 1             # 111 行: Context Parallel（Ulysses SP）
    tp: int = 1             # 112 行: Tensor Parallel
    pp: int = 1             # 113 行: Pipeline Parallel
    ep: int = 1             # 114 行: Expert Parallel
    etp: int = 1            # 115 行: Expert Tensor Parallel（必须为 1 或等于 tp）
    world_size: int = 1     # 116 行: 总进程数
    device_type: str = "cuda"  # 117 行: 设备类型（cuda / npu）

    # ── 内部状态 ──────────────────────────────────────────────
    _world_mesh: DeviceMesh | None = field(default=None, repr=False)  # 120 行: 懒加载
    _meshes: dict[str, DeviceMesh] = field(default_factory=dict, repr=False)  # 121 行: 子网格缓存
```

**设计要点**：`_world_mesh` 和 `_meshes` 使用 `field(repr=False)` 避免在日志中打印巨大的 mesh 对象。

## 3.3 `__post_init__` 验证（123-166 行）

```python
def __post_init__(self):
    # ── 第 124-125 行: dp_shard 自动计算 ──
    # 如果用户设 dp_shard=-1，从 world_size 反推
    if self.dp_shard < 0:
        self.dp_shard = self.world_size // (self.tp * self.cp * self.pp)
    # 示例: world_size=32, tp=2, cp=2, pp=4 → dp_shard = 32/(2×2×4) = 2

    # ── 第 127-133 行: 核心不变量验证 ──
    expected_world_size = self.dp_shard * self.tp * self.cp * self.pp
    if expected_world_size != self.world_size:
        raise ValueError(...)
    # 确保所有 GPU 恰好被分配一次

    # ── 第 136-139 行: ETP 约束 ──
    if self.etp not in (1, self.tp):
        raise ValueError(...)
    # etp 只能是 1（TP 被 EP 吞并）或等于 tp（TP 保持独立）
    # 不支持部分借用（如 etp=2, tp=4）

    # ── 第 142-166 行: EP 借用约束 ──
    if self.ep > 1:
        if self.etp == self.tp:
            # etp=tp 模式: EP 从 dp_shard × cp 借用
            # 要求: ep % cp == 0              → dp_shard_in_ep 为整数
            # 要求: (dp_shard*cp) % ep == 0   → dp_shard_mod_ep 为整数
        else:
            # etp=1 模式: EP 从 dp_shard × cp × tp 借用
            # 要求: ep % (cp*tp) == 0         → dp_shard_in_ep 为整数
            # 要求: (dp_shard*cp*tp) % ep == 0 → dp_shard_mod_ep 为整数
```

**验证约束一览表：**

| 条件 | etp=tp | etp=1 | 原因 |
| --- | --- | --- | --- |
| 被借用的维度 | dp_shard × cp | dp_shard × cp × tp | etp=1 时 TP 也被借用 |
| EP 整除要求 | `ep % cp == 0` | `ep % (cp × tp) == 0` | 确保 `dp_shard_in_ep` 为整数 |
| 剩余 DP 整除要求 | `(dp_shard×cp) % ep == 0` | `(dp_shard×cp×tp) % ep == 0` | 确保 `dp_shard_mod_ep` 为整数 |

**关键不变量**（`__post_init__` 不直接检查但由整除约束保证）：

```
etp=tp: dp_shard_mod_ep × dp_shard_in_ep × cp = dp_shard × cp
etp=1:  dp_shard_mod_ep × dp_shard_in_ep × cp × tp = dp_shard × cp × tp
```

## 3.4 `_build_mesh_without_ep`（179-213 行）

无 EP 时构建 4D Mesh：

```python
def _build_mesh_without_ep(self) -> DeviceMesh:
    # 183-184 行: 定义 4D 维度（即使某维度=1 也保留，确保子网格可提取）
    dims = [self.pp, self.dp_shard, self.cp, self.tp]
    names = ["pp", "dp_shard", "cp", "tp"]

    # 187-189 行: 调用 PyTorch 原生 API 创建设备网格
    mesh = init_device_mesh(self.device_type, tuple(dims), mesh_dim_names=tuple(names))

    # 191-194 行: 缓存 4 个基础 1D 子网格
    self._meshes["pp"]       = mesh["pp"]
    self._meshes["dp_shard"] = mesh["dp_shard"]
    self._meshes["cp"]       = mesh["cp"]
    self._meshes["tp"]       = mesh["tp"]

    # 197 行: dp（数据加载用，无 EP 时就是 dp_shard 的重命名）
    self._meshes["dp"] = mesh["dp_shard"]._flatten(mesh_dim_name="dp")

    # 200-202 行: dp_shard_cp（FSDP 权重分片用）
    # flatten [dp_shard, cp] → 1D，大小 = dp_shard × cp
    self._meshes["dp_shard_cp"] = mesh["dp_shard", "cp"]._flatten(
        mesh_dim_name="dp_shard_cp"
    )

    # 204-205 行: dp_cp（Loss all-reduce 用）
    # 当前与 dp_shard_cp 物理相同，但语义独立（为 HSDP 预留）
    self._meshes["dp_cp"] = mesh["dp_shard", "cp"]._flatten(mesh_dim_name="dp_cp")

    # 208-210 行: pp_cp_tp（模型并行广播组）
    # flatten [pp, cp, tp] → 1D，大小 = pp × cp × tp
    # 同组内 GPU 处理同一批数据，仅分工不同
    self._meshes["pp_cp_tp"] = mesh["pp", "cp", "tp"]._flatten(
        mesh_dim_name="pp_cp_tp"
    )
```

**图示（8 GPU, pp=1, dp_shard=2, cp=2, tp=2）：**

```text
4D mesh [1, 2, 2, 2]:
  GPU 排列: [[[0,1],[2,3]], [[4,5],[6,7]]]

子网格:
  tp:          {0,1}, {2,3}, {4,5}, {6,7}    ← 4 个 TP 组
  dp_shard:    {0,4}, {1,5}, {2,6}, {3,7}    ← 4 个 DP 组
  cp:          {0,2}, {1,3}, {4,6}, {5,7}    ← 4 个 CP 组
  dp_shard_cp: {0,2,4,6}, {1,3,5,7}          ← 2 个 FSDP 组（大小 4）
  pp_cp_tp:    {0,1,2,3}, {4,5,6,7}          ← 2 个广播组
```

## 3.5 `_build_mesh_with_ep`（215-293 行）

有 EP 时构建 5D Mesh：

```python
def _build_mesh_with_ep(self) -> DeviceMesh:
    # ── 第 223-230 行: 计算 EP 拆分维度 ──

    if self.etp == self.tp:
        # etp=tp: EP 仅借用 dp_shard × cp（TP 保持独立）
        dp_shard_mod_ep = self.dp_shard * self.cp // self.ep    # 225 行
        dp_shard_in_ep = self.ep // self.cp                      # 226 行
    else:
        # etp=1: EP 借用 dp_shard × cp × tp（TP 也被吞并）
        dp_shard_mod_ep = self.dp_shard * self.cp * self.tp // self.ep  # 229 行
        dp_shard_in_ep = self.ep // (self.cp * self.tp)                  # 230 行

    # ── 第 232-240 行: 创建 5D mesh ──
    dims = [self.pp, dp_shard_mod_ep, dp_shard_in_ep, self.cp, self.tp]
    names = ["pp", "dp_shard_mod_ep", "dp_shard_in_ep", "cp", "tp"]
    mesh = init_device_mesh(...)

    # ── 第 242-267 行: 缓存基础和复合子网格 ──
    # dp: flatten [dp_shard_mod_ep, dp_shard_in_ep] → 还原完整 dp_shard
    # dp_shard_cp: flatten [dp_shard_mod_ep, dp_shard_in_ep, cp] → FSDP 用
    # dp_cp: 同上但用于 Loss all-reduce
    # pp_cp_tp: flatten [pp, cp, tp]

    # ── 第 269-290 行: EP 网格构建（关键差异点）──

    if self.etp == self.tp:
        # etp=tp: EP = flatten(dp_shard_in_ep [, cp])
        # TP 不参与 EP（因为 TP 仍然在做权重切分）
        ep_mesh_dims = ["dp_shard_in_ep"]
        if self.cp > 1:                    # 274 行: 仅当 CP > 1 时加入
            ep_mesh_dims.append("cp")
        mesh[tuple(ep_mesh_dims)]._flatten(mesh_dim_name="ep")
        self._meshes["ep"] = mesh["ep"]

        # 285 行: 创建 2D ep_tp 网格，用于 ExpertTensorParallel 的 2D 权重切分
        self._meshes["ep_tp"] = mesh["ep", "tp"]
    else:
        # etp=1: EP = flatten(dp_shard_in_ep, cp, tp)
        # TP 参与 EP（因为 TP 已被 EP 借用，不再做权重切分）
        self._meshes["ep"] = mesh["dp_shard_in_ep", "cp", "tp"]._flatten(
            mesh_dim_name="ep"
        )
        # 注意: etp=1 时不创建 ep_tp 网格（不需要 2D 切分）
```

**为什么 etp=tp 时 TP 不加入 EP mesh？**

因为 TP 仍然在做自己的工作——每个 TP rank 持有同一个专家的权重切片（列切分/行切分），它们不是持有不同专家。All-to-All token dispatch 不应该把 token 发给持有同一专家不同切片的 GPU。

**为什么 etp=1 时 TP 加入 EP mesh？**

因为 TP 已经被"吞并"——原来做 TP 的 GPU 现在持有不同的专家。权重只用 `Shard(0)`（按专家维度切分），不再有列/行切分。All-to-All 必须包含这些 GPU 来交换 token。

## 3.6 属性与工具方法（295-418 行）

### Enabled 标志（306-354 行）

```python
@property
def dp_shard_enabled(self) -> bool: return self.dp_shard > 1   # FSDP 是否激活
def dp_enabled(self) -> bool: ...                               # 任意数据并行
def fsdp_enabled(self) -> bool: return self.dp_shard_enabled or self.cp_enabled
    # ↑ 注意: fsdp_enabled 包含 CP，因为 CP rank 参与 FSDP 权重分片
    # 但这个属性在 parallelize 函数中并未使用！实际条件是 get_mesh("dp_shard_cp") is not None
def cp_enabled(self) -> bool: return self.cp > 1
def tp_enabled(self) -> bool: return self.tp > 1
def pp_enabled(self) -> bool: return self.pp > 1
def ep_enabled(self) -> bool: return self.ep > 1
def etp_enabled(self) -> bool: return self.etp > 1
```

### 工具属性（360-378 行）

```python
@property
def fsdp_gradient_divide_factor(self) -> int:
    return self.dp_shard * self.cp
    # FSDP 梯度除法因子 = 数据并行度
    # 对于 MoE expert 参数尤其关键：
    # 虽然 expert 的 FSDP 分片在更小的 mesh 上（dp_shard_mod_ep），
    # 但梯度除法必须与整体数据并行度一致

@property
def seq_len_divisor(self) -> int:
    return self.tp * self.cp * 2
    # 序列长度最小因子
    # tp: Sequence Parallel 要求 seq_len % tp == 0
    # cp: Ulysses 要求 seq_len % cp == 0
    # × 2: varlen attention 因果掩码的对齐需求

@property
def context_and_model_parallel_size(self) -> int:
    return self.cp * self.tp * self.pp
    # 模型并行度 = 所有非数据并行维度
    # 用于确定 pp_cp_tp 组大小和数据广播范围
```

### get_mesh / get_group（380-418 行）

```python
def get_mesh(self, name: str) -> DeviceMesh | None:
    _ = self.world_mesh   # 触发懒加载，确保 mesh 已构建
    return self._meshes.get(name)  # dict.get() 不存在返回 None（防御式）

def get_group(self, name: str) -> dist.ProcessGroup | None:
    submesh = self.get_mesh(name)
    if submesh is None:
        return None
    return submesh.get_group()
    # 注意: 即使 mesh 大小=1 也会返回有效 ProcessGroup
    # 调用方应先检查 *_enabled 标志再使用
```

**为什么用 `_ = self.world_mesh` 而不是 `self.build_mesh()`？**

`world_mesh` 是 `@property`，内含缓存逻辑（`if self._world_mesh is None`）。直接调 `build_mesh()` 会每次重建网格。

---

# 4. 如何验证理解

## 4.1 纸面验证：手动计算 GPU 分配

**练习 1**：给定 `pp=2, dp_shard=2, cp=1, tp=2, ep=1, world_size=8`，画出 4D mesh 并列出所有子网格的成员。

```text
预期答案:
  4D mesh [2, 2, 1, 2]:
    pp=0: [[GPU0, GPU1], [GPU2, GPU3]]
    pp=1: [[GPU4, GPU5], [GPU6, GPU7]]

  tp 组: {0,1}, {2,3}, {4,5}, {6,7}
  dp_shard 组: {0,2}, {1,3}, {4,6}, {5,7}  ← 注意是跨 dp_shard 维度
  pp 组: {0,4}, {1,5}, {2,6}, {3,7}
  dp_shard_cp (= dp_shard): {0,2}, {1,3}, {4,6}, {5,7}
  pp_cp_tp: {0,1,4,5}, {2,3,6,7}
```

**练习 2**：给定 `pp=1, dp_shard=4, cp=2, tp=2, ep=4, etp=2, world_size=16`，计算 `dp_shard_mod_ep` 和 `dp_shard_in_ep`，画出 5D mesh。

```text
预期答案:
  etp=tp → EP 借用 dp_shard × cp
  dp_shard_mod_ep = 4 × 2 / 4 = 2
  dp_shard_in_ep = 4 / 2 = 2
  5D mesh [1, 2, 2, 2, 2]

  EP mesh = flatten(dp_shard_in_ep=2, cp=2) = 大小 4
  ep_tp mesh = 2D [ep=4, tp=2]
```

## 4.2 代码验证：运行测试

```bash
# 单元测试（不需要 GPU）
uv run pytest tests/experimental/archon/test_parallel_dims.py -v

# 分布式测试（需要 GPU）
# 2 GPU: EP mesh 测试
torchrun --nproc_per_node=2 tests/experimental/archon/torchrun/run_parallel_dims.py

# 4 GPU: ETP 2D mesh 测试
torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_parallel_dims.py
```

## 4.3 交互式验证：检查关键不变量

```python
import torch.distributed as dist
from areal.experimental.models.archon.parallel_dims import ArchonParallelDims

# 在 torchrun 环境中运行
dist.init_process_group("nccl")

# 验证不变量 1: world_size 分解
dims = ArchonParallelDims(dp_shard=2, tp=2, cp=2, pp=1, world_size=8)
assert dims.dp_shard * dims.tp * dims.cp * dims.pp == dims.world_size

# 验证不变量 2: EP 维度分裂守恒（etp=tp）
dims = ArchonParallelDims(dp_shard=4, cp=2, tp=2, ep=4, etp=2, world_size=16)
dp_shard_cp = dims.dp_shard * dims.cp  # = 8
mesh = dims.world_mesh
mod_ep_size = mesh.size(1)  # dp_shard_mod_ep = 2
in_ep_size = mesh.size(2)   # dp_shard_in_ep = 2
assert mod_ep_size * in_ep_size * dims.cp == dp_shard_cp  # 2 × 2 × 2 == 8 ✓

# 验证不变量 3: EP mesh 维度正确
ep_mesh = dims.get_mesh("ep")
assert ep_mesh is not None
assert ep_mesh.ndim == 1
ep_tp_mesh = dims.get_mesh("ep_tp")
assert ep_tp_mesh is not None
assert ep_tp_mesh.ndim == 2  # 2D: [ep, tp]

# 验证不变量 4: 安全 get_mesh
assert dims.get_mesh("nonexistent") is None  # 不会抛异常

# 验证不变量 5: 无 EP 时不存在 ep 相关 mesh
dims_no_ep = ArchonParallelDims(dp_shard=4, tp=2, cp=1, pp=1, world_size=8)
assert dims_no_ep.get_mesh("ep") is None
assert dims_no_ep.get_mesh("ep_tp") is None
```

## 4.4 常见理解误区

| 误区 | 正确理解 |
| --- | --- |
| "EP 是第 5 个独立维度" | EP 不是独立维度，它是从 dp_shard/cp/tp 借用 GPU 重新分配的结果 |
| "dp_shard_cp 和 dp_cp 是不同的 GPU 组" | 当前它们是同一组 GPU，只是语义不同（为将来 HSDP 预留） |
| "fsdp_enabled 属性控制 FSDP 是否应用" | parallelize 函数实际检查 `get_mesh("dp_shard_cp") is not None`，该 mesh 总是被创建的 |
| "etp=1 意味着不做张量并行" | Dense 层仍然做 TP，仅 MoE expert 层的 TP 被 EP 借用 |
| "world_size=1 时代码会出错" | 完全合法——所有维度为 1，所有 mesh 大小为 1，集体通信变成 no-op |

---

# 5. 附录

## 5.1 ArchonParallelDims 在系统中的集成点

```text
ParallelStrategy (CLI 配置)
        │
        ▼
ArchonEngine.create_process_group()   ← 唯一实例化点
        │
        ├── self.parallel_dims = ArchonParallelDims(...)
        ├── self._world_mesh = self.parallel_dims.world_mesh  ← 触发 build_mesh()
        │
        ├── 数据加载: get_mesh("dp")
        ├── PP 设置:  get_mesh("pp")
        ├── CP/TP 组: get_group("cp"), get_group("tp")
        └── 广播组:   get_group("pp_cp_tp")
                │
                ▼
        parallelize_qwen2/3()
                │
                ├── get_mesh("tp")           → apply_non_moe_tp()
                ├── get_mesh("ep")           → apply_moe_ep_tp()
                ├── get_mesh("ep_tp")        → ExpertTensorParallel
                ├── get_group("cp")          → apply_cp()
                ├── get_mesh("dp_shard_cp")  → apply_fsdp()
                └── get_mesh("dp_shard_mod_ep") → apply_fsdp() (expert 专用)
```

## 5.2 EP 策略选择矩阵

| EP | TP | etp | 策略类 | Expert 权重切分 | EP mesh 来源 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | - | None | Replicate | 无 |
| 1 | >1 | - | TensorParallel | `[Shard(1/2)]` | 无 |
| >1 | 1 | - | ExpertParallel | `[Shard(0)]` | dp_shard_in_ep × cp |
| >1 | >1 | 1 | ExpertParallel | `[Shard(0)]`（TP 被 EP 借用） | dp_shard_in_ep × cp × tp |
| >1 | >1 | tp | ExpertTensorParallel | `[Shard(0), Shard(1/2)]` | dp_shard_in_ep [× cp] |

## 5.3 完整 Mesh 名称速查表

| Mesh 名称 | 无 EP 时 | 有 EP 时 | 大小 |
| --- | --- | --- | --- |
| `pp` | ✅ | ✅ | pp |
| `dp_shard` | ✅ | ❌ | dp_shard |
| `dp_shard_mod_ep` | ❌ | ✅ | dp_shard×cp÷ep 或 dp_shard×cp×tp÷ep |
| `dp_shard_in_ep` | ❌ | ✅ | ep÷cp 或 ep÷(cp×tp) |
| `cp` | ✅ | ✅ | cp |
| `tp` | ✅ | ✅ | tp |
| `dp` | ✅ | ✅ | dp_shard |
| `dp_shard_cp` | ✅ | ✅ | dp_shard × cp |
| `dp_cp` | ✅ | ✅ | dp_shard × cp |
| `pp_cp_tp` | ✅ | ✅ | pp × cp × tp |
| `ep` | ❌ | ✅ | ep |
| `ep_tp` | ❌ | ✅（仅 etp=tp） | 2D: [ep, tp] |
