"""
实验 1: DTensor 基础 — Shard / Replicate / Partial + redistribute

用法: torchrun --nproc_per_node=2 dtensor_shard_demo.py
不需要 GPU，可用 gloo + CPU。如有 GPU 改 "cpu" → "cuda"。
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate, Partial, distribute_tensor


def separator(title: str):
    if dist.get_rank() == 0:
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")
    dist.barrier()


def demo_shard_by_column():
    """把一个 4×6 矩阵按列 Shard(1) 分到 2 个 rank"""
    separator("1. Shard(dim=1) — 按列切分矩阵")

    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (2,))

    # 创建全局逻辑张量 (每个 rank 提供相同的全局 tensor)
    global_tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    if rank == 0:
        print(f"  全局逻辑张量 shape={list(global_tensor.shape)}:")
        print(f"  {global_tensor}")

    dist.barrier()

    # distribute_tensor: 把全局 tensor 按 placement 分发到 mesh 上
    dt = distribute_tensor(global_tensor, mesh, placements=[Shard(1)])

    print(f"\n  [Rank {rank}] DTensor 信息:")
    print(f"    type        = {type(dt).__name__}")
    print(f"    dt.shape    = {list(dt.shape)}        ← 全局逻辑 shape (4×6)")
    print(f"    placements  = {dt.placements}")
    print(f"    to_local().shape = {list(dt.to_local().shape)}  ← 本地 shard shape")
    print(f"    本地数据:\n{dt.to_local()}")

    dist.barrier()
    if rank == 0:
        print("\n  ✅ Shard(1) 把 4×6 矩阵沿 dim=1 切成两个 4×3")
        print("  📌 dt.shape 返回全局 shape (4,6), to_local() 返回本地 shard (4,3)")

    return dt, mesh


def demo_redistribute(dt, mesh):
    """把 Shard(1) 的 DTensor redistribute 成 Replicate"""
    separator("2. redistribute: Shard → Replicate (= all-gather)")

    rank = dist.get_rank()

    # redistribute 会自动执行 all-gather
    dt_replicate = dt.redistribute(mesh, placements=[Replicate()])

    print(f"  [Rank {rank}] redistribute 后:")
    print(f"    placements  = {dt_replicate.placements}")
    print(f"    to_local().shape = {list(dt_replicate.to_local().shape)}  ← 完整 (4×6)!")
    print(f"    本地数据:\n{dt_replicate.to_local()}")

    dist.barrier()
    if rank == 0:
        print("\n  ✅ Shard(1) → Replicate 触发了 all-gather，每个 rank 拿到完整矩阵")
        print("  📌 这就是 FSDP forward 前拼回完整参数的操作")


def demo_shard_by_row():
    """按行 Shard(0) 切分"""
    separator("3. Shard(dim=0) — 按行切分矩阵")

    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (2,))

    global_tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    dt = distribute_tensor(global_tensor, mesh, placements=[Shard(0)])

    print(f"  [Rank {rank}] Shard(0):")
    print(f"    dt.shape    = {list(dt.shape)}    ← 全局 (4,6)")
    print(f"    to_local().shape = {list(dt.to_local().shape)}  ← 本地 (2,6)")
    print(f"    本地数据:\n{dt.to_local()}")

    dist.barrier()
    if rank == 0:
        print("\n  ✅ Shard(0) 按 dim=0 切：Rank0 拿前2行, Rank1 拿后2行")
        print("  📌 FSDP2 默认用 Shard(0) 切参数")


def demo_replicate():
    """创建 Replicate 的 DTensor"""
    separator("4. Replicate — 每个 rank 持有完整副本")

    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (2,))

    global_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dt = distribute_tensor(global_tensor, mesh, placements=[Replicate()])

    print(f"  [Rank {rank}] Replicate:")
    print(f"    dt.shape    = {list(dt.shape)}")
    print(f"    to_local().shape = {list(dt.to_local().shape)}  ← 跟全局一样")
    print(f"    本地数据:\n{dt.to_local()}")

    dist.barrier()
    if rank == 0:
        print("\n  ✅ Replicate: 每个 rank 持有完整数据, shape 不变")
        print("  📌 DDP 的模型参数就是 Replicate 状态")


def demo_matmul_propagation():
    """
    DTensor 矩阵乘法: 展示 placement 自动传播
    Y = X @ W, X=Replicate, W=Shard(0) → Y=Partial(SUM)
    """
    separator("5. 矩阵乘法: placement 自动传播")

    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (2,))

    # X: 2×4 矩阵, Replicate
    X_global = torch.ones(2, 4)
    X_dt = distribute_tensor(X_global, mesh, placements=[Replicate()])

    # W: 4×6 矩阵, Shard(0) — 按行切
    W_global = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    W_dt = distribute_tensor(W_global, mesh, placements=[Shard(0)])

    print(f"  [Rank {rank}] X: shape={list(X_dt.shape)}, placements={X_dt.placements}")
    print(f"  [Rank {rank}] W: shape={list(W_dt.shape)}, placements={W_dt.placements}")
    print(f"  [Rank {rank}] W local shape: {list(W_dt.to_local().shape)}")

    dist.barrier()

    # DTensor 矩阵乘法
    Y_dt = X_dt @ W_dt

    print(f"\n  [Rank {rank}] Y = X @ W:")
    print(f"    Y.shape      = {list(Y_dt.shape)}       ← 全局 (2,6)")
    print(f"    Y.placements = {Y_dt.placements}  ← 注意!")
    print(f"    Y.to_local() =\n{Y_dt.to_local()}")

    dist.barrier()

    # 把 Y 变成 Replicate (触发 all-reduce)
    Y_full = Y_dt.redistribute(mesh, placements=[Replicate()])
    print(f"\n  [Rank {rank}] Y redistribute to Replicate:")
    print(f"    Y_full.to_local() =\n{Y_full.to_local()}")

    dist.barrier()
    if rank == 0:
        # 单卡验证
        Y_ref = X_global @ W_global
        print(f"\n  单卡参考结果:\n{Y_ref}")
        print(f"  匹配: {torch.allclose(Y_full.to_local(), Y_ref)}")
        print("\n  ✅ X(Replicate) @ W(Shard(0)) → Y(Partial)")
        print("  📌 Partial 说明每个 rank 只有部分和, 需 all-reduce 才完整")
        print("  📌 这正是 TP RowwiseParallel 的数学本质!")


def demo_full_tensor():
    """用 full_tensor() 在所有 rank 上恢复完整张量"""
    separator("6. full_tensor() — 方便调试的 API")

    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (2,))

    global_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    dt = distribute_tensor(global_tensor, mesh, placements=[Shard(1)])

    # full_tensor() = 内部做 all-gather + 拼接
    full = dt.full_tensor()

    print(f"  [Rank {rank}] full_tensor() 结果:")
    print(f"    shape = {list(full.shape)}")
    print(f"    data  =\n{full}")
    print(f"    与原始相同: {torch.equal(full, global_tensor)}")

    dist.barrier()
    if rank == 0:
        print("\n  ✅ full_tensor() 可以随时查看完整数据（调试用）")
        print("  📌 checkpoint 保存时就用 full_tensor() 或 DCP")


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()

    if rank == 0:
        print("╔═════════════════════════════════════════════════════════════╗")
        print("║  实验 1: DTensor 基础 — Shard / Replicate / Partial        ║")
        print("║  2 进程, gloo 后端, CPU tensors                             ║")
        print("╚═════════════════════════════════════════════════════════════╝")

    dist.barrier()

    dt, mesh = demo_shard_by_column()
    demo_redistribute(dt, mesh)
    demo_shard_by_row()
    demo_replicate()
    demo_matmul_propagation()
    demo_full_tensor()

    dist.barrier()
    if rank == 0:
        print(f"\n{'='*65}")
        print("  全部实验完成！")
        print(f"{'='*65}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
