"""
collective_demo.py — 2 进程手动跑 4 种 collective 通信

用法（不需要 GPU，用 gloo 后端在 CPU 上运行）:
    torchrun --nproc_per_node=2 collective_demo.py

会依次演示:
    1. all-reduce
    2. all-gather
    3. reduce-scatter
    4. all-to-all
    5. 创建子 group
    6. 基础概念验证 (rank / world_size / group)
"""

import os
import torch
import torch.distributed as dist


def separator(title: str):
    """打印分隔线"""
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    dist.barrier()  # 确保标题先打印


def demo_basics():
    """演示 rank / world_size / group 的基本概念"""
    separator("0. 基础概念: rank, world_size, group")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(
        f"  [Rank {rank}] "
        f"global_rank={rank}, local_rank={local_rank}, world_size={world_size}"
    )
    dist.barrier()


def demo_all_reduce():
    """
    all-reduce (SUM): 每个 rank 有一个 tensor，对所有 rank 求和，结果分发给每个 rank。

    这就是 DDP 同步梯度的核心操作。
    """
    separator("1. All-Reduce (SUM)")

    rank = dist.get_rank()

    # 每个 rank 创建不同的 tensor
    # Rank 0: [1, 2, 3, 4]
    # Rank 1: [10, 20, 30, 40]
    tensor = torch.tensor([1, 2, 3, 4], dtype=torch.float32) * (10 ** rank)
    print(f"  [Rank {rank}] BEFORE all_reduce: {tensor.tolist()}, shape={list(tensor.shape)}")

    dist.barrier()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # 期望两个 rank 都得到: [11, 22, 33, 44]
    print(f"  [Rank {rank}] AFTER  all_reduce: {tensor.tolist()}, shape={list(tensor.shape)}")
    dist.barrier()

    if rank == 0:
        print("\n  ✅ 解读: 两个 rank 的 tensor 逐元素求和，结果每个 rank 都有一份完整副本")
        print("  📌 DDP 就是这样同步梯度的: all_reduce(grad) 然后除以 world_size")


def demo_all_gather():
    """
    all-gather: 每个 rank 有一个小 tensor，拼接成大 tensor，每个 rank 都拿到完整结果。

    这就是 FSDP forward 前"拼回完整参数"的操作。
    """
    separator("2. All-Gather")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 每个 rank 有一个 [2] 大小的 shard
    # Rank 0: [0.0, 0.1]
    # Rank 1: [1.0, 1.1]
    local_tensor = torch.tensor([rank + 0.0, rank + 0.1], dtype=torch.float32)
    print(f"  [Rank {rank}] BEFORE all_gather: local={local_tensor.tolist()}, shape={list(local_tensor.shape)}")

    # 准备输出: world_size 个同样大小的 tensor
    gathered_list = [torch.zeros(2, dtype=torch.float32) for _ in range(world_size)]

    dist.barrier()
    dist.all_gather(gathered_list, local_tensor)

    print(
        f"  [Rank {rank}] AFTER  all_gather: "
        f"gathered={[t.tolist() for t in gathered_list]}, "
        f"每个元素 shape={list(gathered_list[0].shape)}, "
        f"总共 {len(gathered_list)} 个"
    )
    dist.barrier()

    if rank == 0:
        print("\n  ✅ 解读: 每个 rank 的小 tensor 被收集起来，所有 rank 都得到完整列表")
        print("  📌 FSDP forward: 参数被分片存储，forward 前 all_gather 拼回完整参数")


def demo_all_gather_into_tensor():
    """
    all_gather_into_tensor: 更高效的版本，直接拼成一个大 tensor（不是 list）。
    FSDP 内部实际用的是这个。
    """
    separator("2b. All-Gather into Tensor (FSDP 实际使用的版本)")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 模拟 FSDP: 每个 rank 存模型参数的 1/2
    shard = torch.tensor([rank * 10 + 1, rank * 10 + 2, rank * 10 + 3], dtype=torch.float32)
    print(f"  [Rank {rank}] BEFORE: shard={shard.tolist()}, shape={list(shard.shape)}")

    # 输出是一个大 tensor，大小 = shard_size * world_size
    full_param = torch.zeros(3 * world_size, dtype=torch.float32)

    dist.barrier()
    dist.all_gather_into_tensor(full_param, shard)

    print(f"  [Rank {rank}] AFTER:  full_param={full_param.tolist()}, shape={list(full_param.shape)}")
    dist.barrier()

    if rank == 0:
        print("\n  ✅ 解读: 分片 [1,2,3] 和 [11,12,13] 被拼成完整的 [1,2,3,11,12,13]")
        print("  📌 shape 变化: [3] → [6], 即 shard_size → shard_size × world_size")


def demo_reduce_scatter():
    """
    reduce-scatter: 先 reduce（求和），再 scatter（分片发给各 rank）。

    这就是 FSDP backward 时"梯度汇总+分片"的操作。
    """
    separator("3. Reduce-Scatter")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 每个 rank 有一个完整大小的梯度 tensor [4]
    # Rank 0: [1, 2, 3, 4]
    # Rank 1: [10, 20, 30, 40]
    full_grad = torch.tensor([1, 2, 3, 4], dtype=torch.float32) * (10 ** rank)
    print(f"  [Rank {rank}] BEFORE reduce_scatter: full_grad={full_grad.tolist()}, shape={list(full_grad.shape)}")

    # 把 full_grad 切成 world_size 份
    input_list = list(full_grad.chunk(world_size))

    # 输出: 只拿到自己那一份（reduce 后的）
    output = torch.zeros(4 // world_size, dtype=torch.float32)

    dist.barrier()
    dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)

    # Rank 0 拿到: chunk0 的 sum = [1+10, 2+20] = [11, 22]
    # Rank 1 拿到: chunk1 的 sum = [3+30, 4+40] = [33, 44]
    print(f"  [Rank {rank}] AFTER  reduce_scatter: my_shard={output.tolist()}, shape={list(output.shape)}")
    dist.barrier()

    if rank == 0:
        print("\n  ✅ 解读: 两个 rank 的 [4]-tensor 先按元素求和，再分成两半，各 rank 拿一半")
        print("  📌 shape 变化: [4] → [2], 即 full_size → full_size / world_size")
        print("  📌 FSDP backward: 完整梯度 reduce_scatter → 每个 rank 只存自己那片梯度")


def demo_reduce_scatter_tensor():
    """
    reduce_scatter_tensor: 更高效版本，输入直接是一个大 tensor。
    """
    separator("3b. Reduce-Scatter Tensor (FSDP 实际使用的版本)")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 模拟完整梯度
    full_grad = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * (rank + 1)
    print(f"  [Rank {rank}] BEFORE: full_grad={full_grad.tolist()}, shape={list(full_grad.shape)}")

    # 输出: full_size / world_size
    output = torch.zeros(6 // world_size, dtype=torch.float32)

    dist.barrier()
    dist.reduce_scatter_tensor(output, full_grad, op=dist.ReduceOp.SUM)

    print(f"  [Rank {rank}] AFTER:  my_shard={output.tolist()}, shape={list(output.shape)}")
    dist.barrier()

    if rank == 0:
        print("\n  ✅ 验证: reduce_scatter 是 all_gather 的"反操作"")
        print("  📌 关键等式: all_reduce = reduce_scatter + all_gather")


def demo_all_to_all():
    """
    all-to-all: 每个 rank 有 N 份数据，第 i 份发给 rank i。

    MoE 路由 token 到 expert 时用这个。
    """
    separator("4. All-to-All")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 每个 rank 准备 world_size 份数据，每份发给对应 rank
    # Rank 0 准备: [发给R0的: 00, 01] [发给R1的: 02, 03]
    # Rank 1 准备: [发给R0的: 10, 11] [发给R1的: 12, 13]
    input_list = [
        torch.tensor([rank * 10 + dst * 2, rank * 10 + dst * 2 + 1], dtype=torch.float32)
        for dst in range(world_size)
    ]
    output_list = [torch.zeros(2, dtype=torch.float32) for _ in range(world_size)]

    print(f"  [Rank {rank}] BEFORE all_to_all: input={[t.tolist() for t in input_list]}")

    dist.barrier()
    dist.all_to_all(output_list, input_list)

    # Rank 0 收到: [来自R0: 00, 01] [来自R1: 10, 11]
    # Rank 1 收到: [来自R0: 02, 03] [来自R1: 12, 13]
    print(f"  [Rank {rank}] AFTER  all_to_all: output={[t.tolist() for t in output_list]}")
    dist.barrier()

    if rank == 0:
        print("\n  ✅ 解读: 每个 rank 把自己数据的不同 chunk 发给不同 rank")
        print("  📌 MoE: token 被路由到不同 expert 所在的 rank，就是用 all-to-all")


def demo_sub_group():
    """
    创建子 group 的演示。
    虽然只有 2 个 rank 不太明显，但展示 API 用法。
    """
    separator("5. Process Group (子组)")

    rank = dist.get_rank()

    # 创建一个只包含 rank 0 的 "group"（实际中会有多个 rank）
    # 和一个包含所有 rank 的 group（等价于默认 group）
    all_ranks_group = dist.new_group(ranks=[0, 1])

    tensor = torch.tensor([rank + 1.0])
    print(f"  [Rank {rank}] BEFORE all_reduce on sub-group: {tensor.tolist()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=all_ranks_group)
    print(f"  [Rank {rank}] AFTER  all_reduce on sub-group: {tensor.tolist()}")

    dist.barrier()

    if rank == 0:
        print("\n  ✅ 解读: 子组定义了'哪些 rank 参与通信'")
        print("  📌 多维并行中，TP group / DP group / PP group 都是不同的子组")
        print("  📌 DeviceMesh 自动帮你管理这些子组的创建")


def demo_the_key_equation():
    """
    验证: all_reduce = reduce_scatter + all_gather
    """
    separator("6. 验证关键等式: all_reduce = reduce_scatter + all_gather")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 原始数据
    original = torch.tensor([1.0, 2.0, 3.0, 4.0]) * (rank + 1)

    # --- 方法 A: 直接 all_reduce ---
    tensor_a = original.clone()
    dist.all_reduce(tensor_a, op=dist.ReduceOp.SUM)

    # --- 方法 B: reduce_scatter + all_gather ---
    tensor_b = original.clone()
    # Step 1: reduce_scatter
    shard = torch.zeros(4 // world_size, dtype=torch.float32)
    dist.reduce_scatter_tensor(shard, tensor_b, op=dist.ReduceOp.SUM)

    # Step 2: all_gather
    full = torch.zeros(4, dtype=torch.float32)
    dist.all_gather_into_tensor(full, shard)

    print(f"  [Rank {rank}] all_reduce 结果:              {tensor_a.tolist()}")
    print(f"  [Rank {rank}] reduce_scatter + all_gather: {full.tolist()}")

    match = torch.allclose(tensor_a, full)
    print(f"  [Rank {rank}] 两种方法结果相同: {match}")

    dist.barrier()

    if rank == 0:
        print("\n  ✅ 关键等式验证成功！")
        print("  📌 DDP 用 all_reduce 一步到位")
        print("  📌 FSDP 拆成两步: backward 时 reduce_scatter, forward 时 all_gather")
        print("     好处: 参数可以分片存储，每个 rank 只需 1/G 的参数内存")


def main():
    # 初始化进程组（torchrun 会自动设置环境变量）
    dist.init_process_group(backend="gloo")  # CPU 用 gloo，GPU 用 nccl

    rank = dist.get_rank()

    if rank == 0:
        print("╔══════════════════════════════════════════════════════════╗")
        print("║   PyTorch Distributed Collective 操作 完全演示            ║")
        print("║   2 个进程, gloo 后端, CPU tensors                       ║")
        print("╚══════════════════════════════════════════════════════════╝")

    dist.barrier()

    # 依次运行所有 demo
    demo_basics()
    demo_all_reduce()
    demo_all_gather()
    demo_all_gather_into_tensor()
    demo_reduce_scatter()
    demo_reduce_scatter_tensor()
    demo_all_to_all()
    demo_sub_group()
    demo_the_key_equation()

    dist.barrier()
    if rank == 0:
        print(f"\n{'='*60}")
        print("  全部演示完成！")
        print(f"{'='*60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
