# DeviceMesh 入门指南

**前置知识：**

- [分布式通信包 - torch.distributed](https://pytorch.org/docs/stable/distributed.html)
- Python 3.8 - 3.11
- PyTorch 2.2

在分布式训练中，设置分布式通信器（如 NCCL 通信器）是一项重大挑战。当用户需要组合不同的并行方案时，必须为每种并行方案手动设置和管理 NCCL 通信器（例如 `ProcessGroup`）。这一过程复杂且容易出错。`DeviceMesh` 可以简化这一流程，使其更易管理、更不容易出错。

---

## 一、什么是 DeviceMesh

`DeviceMesh` 是管理 `ProcessGroup` 的**高级抽象**。它允许用户轻松创建节点间和节点内的进程组，无需担心如何为不同的子进程组正确设置 rank。用户还可以通过 `DeviceMesh` 方便地管理多维并行所需的底层进程组和设备。

![PyTorch DeviceMesh 示意图](https://docs.pytorch.org/tutorials/_images/device_mesh.png)

---

## 二、为什么 DeviceMesh 有用

DeviceMesh 在**多维并行**（如 3D 并行）场景中尤为有用，这些场景要求并行方案的可组合性。例如，并行方案需要同时进行跨主机通信和主机内通信。上图展示了如何创建一个 2D mesh：在每台主机内连接设备，同时将每个设备与其他主机上的对应设备相连。

### 2.1 没有 DeviceMesh 的传统设置

如果不使用 DeviceMesh，用户需要在应用任何并行方案之前，手动设置 NCCL 通信器和 CUDA 设备。以下代码展示了**不使用 DeviceMesh** 的混合分片 2D 并行设置：

```python
import os

import torch
import torch.distributed as dist

# 了解全局拓扑
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
print(f"Running example on {rank=} in a world with {world_size=}")

# 创建进程组来管理 2D 并行模式
dist.init_process_group("nccl")
torch.cuda.set_device(rank)

# 创建分片组（如 (0, 1, 2, 3), (4, 5, 6, 7)）
# 并为每个 rank 分配正确的分片组
num_node_devices = torch.cuda.device_count()
shard_rank_lists = list(range(0, num_node_devices // 2)), list(range(num_node_devices // 2, num_node_devices))
shard_groups = (
    dist.new_group(shard_rank_lists[0]),
    dist.new_group(shard_rank_lists[1]),
)
current_shard_group = (
    shard_groups[0] if rank in shard_rank_lists[0] else shard_groups[1]
)

# 创建复制组（如 (0, 4), (1, 5), (2, 6), (3, 7)）
# 并为每个 rank 分配正确的复制组
current_replicate_group = None
shard_factor = len(shard_rank_lists[0])
for i in range(num_node_devices // 2):
    replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
    replicate_group = dist.new_group(replicate_group_ranks)
    if rank in replicate_group_ranks:
        current_replicate_group = replicate_group
```

将上述代码保存为 `2d_setup.py`，然后运行：

```bash
torchrun --nproc_per_node=8 --rdzv_id=100 --rdzv_endpoint=localhost:29400 2d_setup.py
```

> 为了演示简便，这里在单节点上模拟 2D 并行。此代码同样适用于多主机环境。

### 2.2 使用 DeviceMesh 的简化设置

借助 `init_device_mesh()`，上述 2D 设置仅需**两行代码**，且仍可按需访问底层 `ProcessGroup`：

```python
from torch.distributed.device_mesh import init_device_mesh
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))

# 通过 get_group API 访问底层进程组
replicate_group = mesh_2d.get_group(mesh_dim="replicate")
shard_group = mesh_2d.get_group(mesh_dim="shard")
```

保存为 `2d_setup_with_device_mesh.py`，然后运行：

```bash
torchrun --nproc_per_node=8 2d_setup_with_device_mesh.py
```

---

## 三、如何将 DeviceMesh 用于 HSDP

**混合分片数据并行（HSDP）** 是一种 2D 策略：在主机内执行 FSDP，在主机间执行 DDP。

以下示例展示了如何使用 DeviceMesh 将 HSDP 应用到模型上，无需手动创建和管理分片组与复制组：

```python
import torch
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard as FSDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


# HSDP: MeshShape(2, 4)
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp_replicate", "dp_shard"))
model = FSDP(
    ToyModel(), device_mesh=mesh_2d
)
```

保存为 `hsdp.py`，然后运行：

```bash
torchrun --nproc_per_node=8 hsdp.py
```

---

## 四、如何将 DeviceMesh 用于自定义并行方案

在大规模训练中，可能需要更复杂的自定义并行组合。例如，可能需要从 mesh 中切出子网格用于不同的并行方案。DeviceMesh 允许用户从父 mesh 切出子 mesh，并**复用**父 mesh 初始化时已创建的 NCCL 通信器。

```python
from torch.distributed.device_mesh import init_device_mesh
mesh_3d = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp"))

# 从父 mesh 切出子 mesh
hsdp_mesh = mesh_3d["replicate", "shard"]
tp_mesh = mesh_3d["tp"]

# 通过 get_group API 访问底层进程组
replicate_group = hsdp_mesh["replicate"].get_group()
shard_group = hsdp_mesh["shard"].get_group()
tp_group = tp_mesh.get_group()
```

---

## 五、总结

本文介绍了 `DeviceMesh` 和 `init_device_mesh()` 的使用方法，以及如何用它们描述集群中设备的布局。

**更多资料：**

- [2D 并行：张量/序列并行与 FSDP 的结合](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [可组合的 PyTorch 分布式与 PT2](https://pytorch.org/get-started/pytorch-2.0/)