"""
实验 2: 2D DeviceMesh + parallelize_module 切一层 MLP

用法: torchrun --nproc_per_node=4 tp_linear_demo.py
需要 4 个进程。可用 gloo+CPU 或 nccl+CUDA。

演示:
  1) 创建 2D mesh [dp=2, tp=2]
  2) 用 parallelize_module 对 MLP 做 TP
  3) 检查参数变成 DTensor 后的 shape / placement
  4) 验证 forward 输出与单卡一致
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)


class SimpleMLP(nn.Module):
    """两层 MLP: w1 (D→4D) + SiLU + w2 (4D→D)"""

    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4, bias=False)
        self.w2 = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


def separator(title: str):
    if dist.get_rank() == 0:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    dist.barrier()


def demo_2d_mesh():
    """创建 2D mesh 并展示子 mesh"""
    separator("1. 创建 2D DeviceMesh (dp=2, tp=2)")

    rank = dist.get_rank()

    mesh_2d = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))

    tp_mesh = mesh_2d["tp"]
    dp_mesh = mesh_2d["dp"]

    print(f"  [Rank {rank}] mesh_2d = {mesh_2d}")
    print(f"  [Rank {rank}] tp_mesh = {tp_mesh}  (我的 TP 伙伴)")
    print(f"  [Rank {rank}] dp_mesh = {dp_mesh}  (我的 DP 伙伴)")

    dist.barrier()
    if rank == 0:
        print("""
  ✅ 2D mesh 布局:
               tp=0  tp=1
      dp=0  [ Rank0  Rank1 ]  ← TP group {0,1}
      dp=1  [ Rank2  Rank3 ]  ← TP group {2,3}
               │      │
             DP group  DP group
             {0,2}     {1,3}
  """)

    return mesh_2d


def demo_parallelize_module(mesh_2d):
    """用 parallelize_module 对 MLP 做 Tensor Parallel"""
    separator("2. parallelize_module: MLP 的 TP 切分")

    rank = dist.get_rank()
    tp_mesh = mesh_2d["tp"]
    tp_size = tp_mesh.size()

    # 在所有 rank 上创建相同的模型（用相同的 seed）
    torch.manual_seed(42)
    model = SimpleMLP(dim=8)

    # 保存原始参数用于对比
    w1_orig = model.w1.weight.clone()
    w2_orig = model.w2.weight.clone()

    if rank == 0:
        print(f"  原始模型参数:")
        print(f"    w1.weight: shape={list(w1_orig.shape)}, type={type(w1_orig).__name__}")
        print(f"    w2.weight: shape={list(w2_orig.shape)}, type={type(w2_orig).__name__}")

    dist.barrier()

    # TP plan: w1 列切(ColwiseParallel), w2 行切(RowwiseParallel)
    # 这就是 Megatron 论文的经典 MLP 切法
    tp_plan = {
        "w1": ColwiseParallel(),     # W1 按 out_features 切
        "w2": RowwiseParallel(),     # W2 按 in_features 切
    }

    # 应用 TP!
    model = parallelize_module(model, tp_mesh, tp_plan)

    # 检查参数变化
    print(f"\n  [Rank {rank}] parallelize_module 后:")
    for name, param in model.named_parameters():
        is_dtensor = isinstance(param, DTensor)
        local_shape = list(param.to_local().shape) if is_dtensor else list(param.shape)
        global_shape = list(param.shape)  # DTensor.shape 返回全局 shape
        placements = param.placements if is_dtensor else "N/A"

        print(f"    {name}:")
        print(f"      type       = {type(param).__name__}")
        print(f"      global shape = {global_shape}")
        print(f"      local shape  = {local_shape}")
        print(f"      placements   = {placements}")

    dist.barrier()
    if rank == 0:
        print(f"""
  ✅ 解读:
    w1.weight: 原始 (32, 8) → 全局仍 (32, 8), 本地 (16, 8)
      placement = Shard(0): 按 out_features 切, 每个 TP rank 持有一半输出神经元

    w2.weight: 原始 (8, 32) → 全局仍 (8, 32), 本地 (8, 16)
      placement = Shard(1): 按 in_features 切, 每个 TP rank 持有一半输入神经元

  📌 ColwiseParallel → Shard(0), RowwiseParallel → Shard(1)
  📌 全局 shape 不变! 只有 to_local() 才看到切分后的大小
  📌 MLP 只需在 w2 输出后做一次 all-reduce（DTensor 自动处理）
  """)

    return model, w1_orig, w2_orig


def demo_forward_correctness(model, w1_orig, w2_orig, mesh_2d):
    """验证 TP 后的 forward 与单卡结果一致"""
    separator("3. Forward 正确性验证")

    rank = dist.get_rank()
    tp_mesh = mesh_2d["tp"]

    # 构造输入: 所有 rank 用相同输入
    torch.manual_seed(123)
    x = torch.randn(2, 8)  # batch=2, dim=8

    # 单卡参考计算
    with torch.no_grad():
        ref_output = F.silu(x @ w1_orig.T) @ w2_orig.T

    # 分布式计算
    with torch.no_grad():
        tp_output = model(x)

    print(f"  [Rank {rank}] 输入 x shape: {list(x.shape)}")
    print(f"  [Rank {rank}] TP 输出 shape: {list(tp_output.shape)}")
    print(f"  [Rank {rank}] TP 输出类型: {type(tp_output).__name__}")
    print(f"  [Rank {rank}] TP 输出值:\n{tp_output}")

    dist.barrier()

    if rank == 0:
        match = torch.allclose(tp_output, ref_output, atol=1e-5)
        print(f"\n  单卡参考输出:\n{ref_output}")
        print(f"\n  TP 输出与单卡参考匹配: {match}")
        print("""
  ✅ parallelize_module 后，model(x) 的输出与单卡完全一致!
  📌 通信 (all-reduce) 被 DTensor 自动插入在 w2 的输出处
  📌 用户代码完全不需要修改——这就是 "single-device semantic"
  """)


def demo_inspect_local_shards(model):
    """仔细查看每个 rank 上的本地 shard 内容"""
    separator("4. 查看各 rank 上的本地 shard")

    rank = dist.get_rank()

    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            local = param.to_local()
            print(
                f"  [Rank {rank}] {name}: "
                f"global={list(param.shape)}, "
                f"local={list(local.shape)}, "
                f"local_sum={local.sum().item():.4f}"
            )

    dist.barrier()
    if rank == 0:
        print("""
  ✅ 同一 TP group 内的 rank 持有互补的 shard
  📌 Rank 0 和 Rank 1 的 local_sum 不同——它们各自持有不同的参数片段
  📌 但 all-gather 后拼起来就是完整参数
  """)


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert world_size == 4, f"需要 4 个进程, 当前 world_size={world_size}"

    if rank == 0:
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║  实验 2: 2D DeviceMesh + parallelize_module (TP)              ║")
        print("║  4 进程, gloo 后端, CPU tensors                                ║")
        print("╚════════════════════════════════════════════════════════════════╝")

    dist.barrier()

    mesh_2d = demo_2d_mesh()
    model, w1_orig, w2_orig = demo_parallelize_module(mesh_2d)
    demo_forward_correctness(model, w1_orig, w2_orig, mesh_2d)
    demo_inspect_local_shards(model)

    dist.barrier()
    if rank == 0:
        print(f"{'='*70}")
        print("  全部实验完成！")
        print(f"{'='*70}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
