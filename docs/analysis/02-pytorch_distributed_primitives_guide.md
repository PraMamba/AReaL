# PyTorch Distributed 三个底层名词：rank、group、collective

---

## 一、世界观：world_size / rank / local_rank

### 1.1 物理场景

假设你有 **2 台机器（Node）**，每台 **4 张 GPU**：

```
Node 0                          Node 1
┌─────────────────────────┐    ┌─────────────────────────┐
│ GPU0  GPU1  GPU2  GPU3  │    │ GPU0  GPU1  GPU2  GPU3  │
│ rank0 rank1 rank2 rank3 │    │ rank4 rank5 rank6 rank7 │
│ local  local local local│    │ local  local local local│
│ rank0  rank1 rank2 rank3│    │ rank0  rank1 rank2 rank3│
└─────────────────────────┘    └─────────────────────────┘
              world_size = 8
```

| 概念 | 定义 | 上图中的值 |
|------|------|-----------|
| **world_size** | 总进程数（通常 = 总 GPU 数） | 8 |
| **rank**（global rank） | 每个进程的全局唯一 ID，范围 [0, world_size) | 0-7 |
| **local_rank** | 进程在**本机内**的 ID，范围 [0, 本机GPU数) | 0-3（每台机器各自从0开始） |
| **node_rank** | 机器编号 | 0 或 1 |

**换算关系**：`rank = node_rank × gpus_per_node + local_rank`

### 1.2 为什么需要 local_rank？

`local_rank` 决定进程绑定哪张物理 GPU。你会在代码里看到：

```python
torch.cuda.set_device(local_rank)  # 把这个进程"钉"到对应的本地 GPU
```

`rank`（global）则用于进程间通信——"我是谁，我要跟谁说话"。

### 1.3 torchrun 帮你设好一切

```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=192.168.1.1 --master_port=29500 train.py
```

torchrun 自动设置环境变量 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`，你的代码里直接读即可：

```python
import os
rank       = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
```

---

## 二、Process Group 是什么

### 2.1 定义

Process Group（进程组）= **一组可以互相通信的进程的集合** + **底层通信后端（NCCL/Gloo/MPI）**。

```python
import torch.distributed as dist

# 初始化"默认"进程组——包含所有进程
dist.init_process_group(backend="nccl")

# 之后所有 collective 默认在这个组上执行
dist.all_reduce(tensor)  # 隐含 group=默认组（即所有进程）
```

### 2.2 创建子组

你可以创建**只包含部分 rank** 的子组：

```python
# 只让 rank 0 和 rank 1 组成一个子组
sub_group = dist.new_group(ranks=[0, 1])

# 在子组上做 collective——只有 rank 0 和 1 参与
if rank in [0, 1]:
    dist.all_reduce(tensor, group=sub_group)
```

### 2.3 为什么需要子组？—— 为多维并行铺路

分布式训练经常需要**不同维度**的通信：

```
8 GPUs，组织为 (2, 4) 的 2D mesh：

          TP group (同行)
         ┌─────────────────────┐
  rank 0 │ rank 1 │ rank 2 │ rank 3     ← TP group 0: {0,1,2,3}
  rank 4 │ rank 5 │ rank 6 │ rank 7     ← TP group 1: {4,5,6,7}
         └─────────────────────┘
    │        │        │        │
    DP group (同列)
    {0,4}  {1,5}  {2,6}  {3,7}
```

- **TP（Tensor Parallel）** 通信在同一行的 group 里做 all-reduce
- **DP（Data Parallel）** 通信在同一列的 group 里做 all-reduce
- 这就是 **DeviceMesh** 帮你管理的事情

### 2.4 DeviceMesh — 更高层的抽象

```python
from torch.distributed.device_mesh import init_device_mesh

# 把 8 个 GPU 排成 (2, 4) 的二维网格
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))

# 自动创建好所有子 process group
tp_group = mesh_2d.get_group(mesh_dim="tp")    # 4 个 rank 的组
dp_group = mesh_2d.get_group(mesh_dim="dp")    # 2 个 rank 的组
```

DeviceMesh 就是 "process group 的管理器"。FSDP2、TP、PP 都接收 DeviceMesh 作为参数。

---

## 三、Collective 通信操作详解

所有 collective 都是**组内所有进程必须一起调用**的操作。缺一个进程就会死锁。

### 3.1 一图总览

```
假设 world_size = 2, rank 0 有 [A], rank 1 有 [B]

┌──────────────────┬──────────────────────────────┬─────────────────────────┐
│    操作           │      rank 0 结果              │     rank 1 结果          │
├──────────────────┼──────────────────────────────┼─────────────────────────┤
│ broadcast(src=0) │      [A]                     │      [A]                │
│ reduce(dst=0)    │      [A+B]                   │      [B] (不变)          │
│ all_reduce       │      [A+B]                   │      [A+B]              │
│ gather(dst=0)    │      [A, B]                  │      [B] (不变)          │
│ all_gather       │      [A, B]                  │      [A, B]             │
│ scatter(src=0)   │      [A₀]                    │      [A₁]               │
│ reduce_scatter   │      chunk₀(A+B)             │      chunk₁(A+B)        │
│ all_to_all       │      [A₀, B₀]                │      [A₁, B₁]           │
└──────────────────┴──────────────────────────────┴─────────────────────────┘
```

### 3.2 逐个讲解

#### All-Reduce — 最核心，DDP 的基石

**语义**：每个 rank 有一个 tensor → 对所有 rank 的 tensor 做 reduce（如求和）→ **每个 rank 都拿到完整的结果**。

```
Rank 0: [1, 2, 3]     Rank 1: [4, 5, 6]
           │                      │
           └──── all_reduce(SUM) ─┘
                      │
           Rank 0: [5, 7, 9]     Rank 1: [5, 7, 9]
```

**在训练中的用途**：DDP 在 backward 结束后对所有梯度做 all-reduce 求平均。

```python
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
tensor /= world_size  # 手动除以 world_size 得到平均值
```

#### All-Gather — FSDP 的灵魂操作

**语义**：每个 rank 有一个**小** tensor → 拼接成一个**大** tensor → **每个 rank 都拿到完整的大 tensor**。

```
Rank 0: [A₀]          Rank 1: [A₁]
           │                    │
           └── all_gather ──────┘
                    │
        Rank 0: [A₀, A₁]      Rank 1: [A₀, A₁]
```

**在训练中的用途**：FSDP 把参数分片存储，forward 前用 all-gather 把完整参数拼回来。

```python
# FSDP 内部逻辑（简化）
# 每个 rank 只存 1/G 的参数
sharded_param = full_param.chunk(world_size)[rank]
# forward 前
full_param = torch.empty(full_size)
dist.all_gather_into_tensor(full_param, sharded_param)
# 用 full_param 算 forward
output = F.linear(input, full_param)
# forward 后释放 full_param，省内存
del full_param
```

#### Reduce-Scatter — FSDP backward 的灵魂操作

**语义**：reduce + scatter。先对所有 rank 的 tensor 做 reduce，然后把结果**分片**发给每个 rank。

```
Rank 0: [a0, a1, a2, a3]      Rank 1: [b0, b1, b2, b3]
              │                            │
              └── reduce_scatter(SUM) ─────┘
                          │
           Rank 0: [a0+b0, a1+b1]    Rank 1: [a2+b2, a3+b3]
                   (前半chunk)                 (后半chunk)
```

**在训练中的用途**：FSDP backward 时，每个 rank 有完整的梯度 tensor。reduce-scatter 把梯度求和并分片回去，每个 rank 只拿到自己那片梯度。

#### All-to-All — Tensor Parallel / Expert Parallel

**语义**：每个 rank 有 N 份数据（N=world_size），第 i 份发给 rank i。同时从所有 rank 各收一份。

```
Rank 0: [A₀→R0, A₁→R1]       Rank 1: [B₀→R0, B₁→R1]
              │                            │
              └─── all_to_all ─────────────┘
                        │
           Rank 0: [A₀, B₀]          Rank 1: [A₁, B₁]
                   (来自R0, R1)              (来自R0, R1)
```

**在训练中的用途**：MoE（Mixture of Experts）中把 token 路由到对应 expert 所在的 rank。

### 3.3 关键等式

```
all_reduce  =  reduce_scatter  +  all_gather
```

这个等式非常重要！FSDP 就是把 DDP 的 all-reduce 拆成了 reduce-scatter（backward 时）+ all-gather（forward 时），从而让参数可以分片存储。

### 3.4 数据量与通信量

| 操作 | 每个 rank 发送 | 每个 rank 接收 | 通信量(Ring 算法) |
|------|--------------|--------------|-----------------|
| all-reduce | N | N | 2N(G-1)/G |
| all-gather | N/G | N | N(G-1)/G |
| reduce-scatter | N | N/G | N(G-1)/G |
| all-to-all | N | N | N(G-1)/G |

（N = 总数据量，G = group 大小）

---

## 四、Process Group 与分布式训练技术的映射

理解了三个底层名词，现在看看上层技术如何使用它们：

| 技术 | 用了什么 collective | 用在什么 group 上 |
|------|-------------------|-----------------|
| **DDP** | all-reduce（梯度同步）| 默认组（所有 rank） |
| **FSDP forward** | all-gather（拼参数）| FSDP shard group |
| **FSDP backward** | reduce-scatter（梯度分片）| FSDP shard group |
| **Tensor Parallel** | all-reduce 或 all-gather | TP group（同一层内的 rank） |
| **Pipeline Parallel** | P2P send/recv | PP group（流水线相邻 stage） |
| **Context Parallel** | all-to-all 或 ring send/recv | CP group |
| **MoE Expert Parallel** | all-to-all | EP group |

---

## 五、推荐学习资源

### 官方文档（按推荐顺序）

1. **PyTorch Distributed Overview** — 总入口，所有分布式功能的路线图
   - https://docs.pytorch.org/docs/stable/distributed.html

2. **Writing Distributed Applications with PyTorch** — 从零手写 collective 的教程
   - https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html

3. **Getting Started with DeviceMesh** — 理解多维 mesh 与子 group
   - https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html

4. **NCCL Collective Operations** — 精确的 collective 语义定义和图示
   - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html

### 实操向

5. **PyTorch DDP Examples** — torchrun 启动的最小示例
   - https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md

6. **torchtitan FSDP 文档** — FSDP2 的设计动机和 API 对比
   - https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md

---

## 六、附录：配合实验脚本阅读

运行附带的 `collective_demo.py` 脚本：

```bash
# 用 gloo 后端在 CPU 上启动 2 个进程（不需要 GPU）
torchrun --nproc_per_node=2 collective_demo.py
```

脚本会依次演示 all-reduce、all-gather、reduce-scatter、all-to-all，打印每个 rank 操作前后的 tensor 值和 shape，让你亲眼看到数据的流动。
