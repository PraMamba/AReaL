# 只学 FSDP2，把它吃透

---

## 一、FSDP 解决什么问题

回顾第一阶段的内存公式（混合精度 + AdamW）：

```
单卡训练总内存 ≈ 16N + 激活
  = 2N (bf16参数) + 4N (fp32 master) + 8N (Adam m+v) + 2N (bf16梯度) + 激活
```

**7B 模型 → 16×7.6B = 121.6 GB**，单张 A100 80GB 放不下。

DDP 的做法：每张卡存完整的 16N → 不省任何参数/优化器内存。

FSDP 的做法：把 16N **分片到 G 张卡**，每张卡只存 16N/G。

```
                DDP (4 卡)                    FSDP (4 卡)
          ┌─────────────────┐          ┌──────────────────┐
  Rank 0  │ 全量参数 16N     │   Rank 0 │ 1/4 参数 = 4N    │
  Rank 1  │ 全量参数 16N     │   Rank 1 │ 1/4 参数 = 4N    │
  Rank 2  │ 全量参数 16N     │   Rank 2 │ 1/4 参数 = 4N    │
  Rank 3  │ 全量参数 16N     │   Rank 3 │ 1/4 参数 = 4N    │
          └─────────────────┘          └──────────────────┘
  总内存     4 × 16N = 64N               4 × 4N = 16N ✅
```

---

## 二、核心机制：三个阶段的参数状态

这是本阶段最重要的一张图：

```
时间线 ──────────────────────────────────────────────────────►

                    Layer i                    Layer i+1
              ┌──────────────────┐       ┌──────────────────┐
    Forward   │  ① all-gather    │       │  ① all-gather    │
              │    (拼回完整参数)  │       │                  │
              │  ② 计算 forward  │       │  ② 计算 forward  │
              │  ③ 释放完整参数   │       │  ③ 释放完整参数   │
              │    (回到 sharded) │       │    (回到 sharded) │
              └──────────────────┘       └──────────────────┘

              ┌──────────────────┐       ┌──────────────────┐
    Backward  │  ① all-gather    │       │  ① all-gather    │
    (逆序)    │    (再次拼回)     │       │                  │
              │  ② 计算梯度      │       │  ② 计算梯度      │
              │  ③ reduce-scatter│       │  ③ reduce-scatter│
              │    (梯度分片回去)  │       │                  │
              │  ④ 释放完整参数   │       │  ④ 释放完整参数   │
              └──────────────────┘       └──────────────────┘

    Optimizer  每个 rank 只对自己那 1/G 的参数做 Adam 更新
```

### 参数在各阶段的状态

| 阶段 | 参数状态 | DTensor placement | 每 rank 内存 |
|------|---------|------------------|-------------|
| **平时（idle）** | 分片 | `Shard(0)` | param_size / G |
| **forward 中** | 完整（临时） | plain Tensor（all-gather 后） | param_size（短暂） |
| **forward 后** | 立即释放完整版，回到分片 | `Shard(0)` | param_size / G |
| **backward 中** | 再次 all-gather 拼回完整 | plain Tensor | param_size（短暂） |
| **backward 后** | 梯度 reduce-scatter → 分片梯度 | `Shard(0)` | grad_size / G |
| **optimizer step** | 分片参数 + 分片梯度 + 分片状态 | `Shard(0)` | (param+grad+opt) / G |

### 关键洞察：峰值内存

FSDP 的峰值出现在**某一层 forward/backward 时**：此刻该层的完整参数被 all-gather 到所有 rank，加上已有的分片参数和激活。

```
峰值内存 ≈ 分片的全部参数/梯度/优化器 (16N/G)
          + 当前层的完整参数 (2 × param_per_layer)    ← 这一层临时 all-gather 回来
          + 激活内存
```

对比 DDP 的 `16N + 激活`，FSDP 大约节省到 `16N/G + 2×param_per_layer + 激活`。

---

## 三、FSDP2 的代码结构

### 3.1 基本用法

```python
from torch.distributed.fsdp import fully_shard

model = Transformer()

# ❶ 自底向上: 先 shard 每个子模块
for layer in model.layers:
    fully_shard(layer)

# ❷ 再 shard 根模块 (包含 embedding、final norm、lm_head)
fully_shard(model)

# ❸ optimizer 必须在 fully_shard 之后创建!
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ❹ 训练循环跟单卡完全一样
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 3.2 fully_shard 到底做了什么

```python
fully_shard(layer)
```

这一行做了 5 件事：

1. **参数转 DTensor**：`layer.weight` 从 `Tensor` → `DTensor(Shard(0))`，按 dim-0 切到各 rank
2. **注册 pre-forward hook**：forward 前 all-gather 拼回完整参数，转为普通 Tensor
3. **注册 post-forward hook**：forward 后释放完整参数，转回 DTensor(Shard(0))
4. **注册 pre-backward hook**：backward 前再次 all-gather
5. **注册 post-backward hook**：梯度 reduce-scatter → 分片梯度

### 3.3 嵌套 fully_shard 的意义

```python
for layer in model.layers:
    fully_shard(layer)      # 每个 TransformerBlock 是一个 FSDP unit
fully_shard(model)          # 根模块收集剩余参数 (embedding, norm, lm_head)
```

**为什么要分层 shard？** 如果只 shard 根模块，forward 时会 all-gather **整个模型**的参数——峰值内存等于完整模型，没有省到。

分层后，每次只 all-gather **一层**的参数 → 峰值 = 分片总量 + 一层完整参数。

### 3.4 reshard_after_forward

```python
fully_shard(layer, reshard_after_forward=True)   # 默认: forward 后释放
fully_shard(layer, reshard_after_forward=False)   # forward 后保留完整参数
fully_shard(model, reshard_after_forward=False)   # 根模块常用 False
```

| 设置 | 含义 | 内存 | 通信 |
|------|------|------|------|
| `True` | forward 后释放，backward 要再 all-gather | 省内存 | 多一次通信 |
| `False` | forward 后保留，backward 直接用 | 多一层参数内存 | 少一次通信 |

**推荐**：子模块用 `True`（省内存），根模块用 `False`（因为 backward 立刻就要用）。

---

## 四、为什么 FSDP 能省内存——精确计算

以 Qwen2-7B (7.6B 参数) 在 8 卡上为例：

| 项目 | DDP (每卡) | FSDP (每卡, G=8) |
|------|-----------|-----------------|
| bf16 参数 | 15.2 GB | 1.9 GB |
| fp32 master weights | 30.4 GB | 3.8 GB |
| Adam m + v | 60.8 GB | 7.6 GB |
| bf16 梯度 | 15.2 GB | 1.9 GB |
| **参数相关小计** | **121.6 GB** ❌ | **15.2 GB** ✅ |
| 临时 all-gather (1层) | 0 | ~0.9 GB |
| 激活 | ~10 GB | ~10 GB |
| **总计** | ~131 GB | **~26 GB** ✅ |

8 卡 A100 80GB 下，FSDP 让 7B 模型训练绰绰有余。

### 关键公式

```
FSDP 每卡内存 ≈ 16N/G + 2×(最大单层参数) + 激活

其中:
  16N/G  = 分片的参数+梯度+优化器
  2×单层  = forward 时临时 all-gather 回来的一层完整参数
  激活    = 不被 FSDP 减少 (需要 AC 来减)
```

---

## 五、FSDP 的通信模式

### 5.1 DDP vs FSDP 的通信对比

```
DDP (一个 step):
  forward:  无通信（每卡有完整参数）
  backward: all-reduce (梯度同步)     ← 通信量 = 2N × (G-1)/G

FSDP (一个 step):
  forward:  L 次 all-gather (每层拼参数)   ← 通信量 = N × (G-1)/G
  backward: L 次 all-gather + L 次 reduce-scatter
                                          ← 通信量 = 2N × (G-1)/G
  总通信量 = 3N × (G-1)/G               ← 比 DDP 多 50%
```

**FSDP 通信量比 DDP 多 ~50%**，但通过 prefetch（计算和通信重叠）可以大幅掩盖。

### 5.2 Prefetch 重叠

```
时间 →
              Layer 0 计算    Layer 1 计算    Layer 2 计算
计算流:       ████████████    ████████████    ████████████
通信流:    AG(0)  AG(1)          AG(2)
              ↑
              Layer 0 的 all-gather 早就完成了
              Layer 1 的 all-gather 与 Layer 0 计算重叠
```

FSDP2 默认开启 implicit prefetching：提前发起下一层的 all-gather。

---

## 六、FSDP vs TP：互补而非替代

### 6.1 本质区别

| 维度 | FSDP | TP |
|------|------|-----|
| **切什么** | 参数/梯度/优化器状态 | 单层权重矩阵 + 激活 |
| **什么时候切** | 计算外分片，计算时临时拼回 | 计算中就是分片状态 |
| **通信在哪** | 模块边界（层与层之间） | 算子内部（矩阵乘法中间） |
| **省什么内存** | 参数相关 (16N/G) | 参数 + **激活** |
| **通信特征** | all-gather 大块数据，可与计算重叠 | all-reduce 小块数据，阻塞计算 |
| **适合范围** | 跨机 (inter-node) | 机内 (intra-node, 需 NVLink) |

### 6.2 为什么需要组合

**场景 1: 模型大，GPU 多 (>128)**

纯 FSDP 在 256 卡上：all-gather 涉及 256 个 rank，ring latency 巨大。
FSDP + TP(8)：FSDP 只需在 32 个 rank 间通信，TP 在 8 卡 NVLink 内完成。

```
256 GPUs = 32 nodes × 8 GPUs/node

纯 FSDP:     all-gather across 256 ranks (慢)
FSDP + TP:   TP intra-node (8 ranks, NVLink 快)
             FSDP inter-node (32 ranks, 通信量少)
```

**场景 2: batch size 受限**

纯 FSDP 要求全局 batch_size ≥ world_size。如果 Llama2 训练只允许 global_batch=1024 但你有 2048 卡，FSDP 不够用。TP 可以把同一 batch 的计算分到多卡，不增加 batch size。

**场景 3: 单层太大**

即使分片后，FSDP forward 时仍需 all-gather **整层**参数。如果单层参数就超过单卡内存，FSDP 无能为力。TP 在计算时就保持分片，不需要拼回完整层。

### 6.3 用 DeviceMesh 组合

```python
mesh_2d = init_device_mesh("cuda", (dp_size, tp_size),
                            mesh_dim_names=("dp", "tp"))

# 先 TP (在 tp 子 mesh 上切层内矩阵)
for block in model.layers:
    parallelize_module(block, mesh_2d["tp"], tp_plan)

# 再 FSDP (在整个 2D mesh 上分片参数)
for block in model.layers:
    fully_shard(block, mesh=mesh_2d)
fully_shard(model, mesh=mesh_2d)
```

参数的 DTensor placement 变成 2 维：`(Shard(0), Shard(0))` = dp 维 FSDP 分片 + tp 维 TP 分片。

---

## 七、FSDP2 vs FSDP1

| 特性 | FSDP1 | FSDP2 |
|------|-------|-------|
| 参数表示 | FlatParameter (flatten+concat) | **DTensor per-parameter Shard(0)** |
| 状态字典 | 需要 all-gather 才能得到 | 直接是 sharded DTensor，免通信 |
| 内存管理 | record_stream (不确定性) | **无 record_stream，确定性低内存** |
| API 形式 | Module wrapper | **就地注册 hook，不改变 module 类型** |
| 与 TP 组合 | 复杂 | **通过 DeviceMesh 自然组合** |
| checkpoint | 需要专门 API | **DCP 直接存 DTensor** |

---

## 八、Mixed Precision in FSDP2

```python
from torch.distributed.fsdp import MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # all-gather 后参数 cast 到 bf16 做计算
    reduce_dtype=torch.float32,     # reduce-scatter 梯度用 fp32 保精度
)

fully_shard(layer, mp_policy=mp_policy)
```

**优势**：FSDP2 在模块边界统一 cast，比 `torch.amp` 更可控。

---

## 九、训练循环完整伪代码

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

# 1. 初始化
dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (dist.get_world_size(),))

# 2. 建模型
model = Transformer(config)

# 3. 自底向上 fully_shard
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for layer in model.layers:
    fully_shard(layer, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=False)

# 4. 此时参数已经是 DTensor(Shard(0))
for p in model.parameters():
    assert isinstance(p, DTensor)
    assert p.placements == (Shard(0),)

# 5. optimizer 在 fully_shard 之后创建
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 6. 训练循环 (跟单卡一样!)
for batch in dataloader:
    input_ids, labels = batch
    loss = model(input_ids, labels)    # 内部自动 all-gather/释放
    loss.backward()                     # 内部自动 all-gather + reduce-scatter
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # DTensor 兼容
    optimizer.step()                    # 在分片参数上更新
    optimizer.zero_grad()
```

---

## 十、自测三问

### Q1: 参数在 forward 前后分别是什么状态？

**forward 前**: DTensor `Shard(0)` — 每个 rank 只持有 1/G 的参数。pre-forward hook 触发 all-gather，临时拼成完整 Tensor。

**forward 后**: post-forward hook 释放完整参数，回到 `Shard(0)`。（`reshard_after_forward=True` 时）

**backward 中**: 再次 all-gather 拼回完整参数，计算梯度后 reduce-scatter 把梯度分片。

**optimizer step**: 在分片的参数、分片的梯度、分片的 Adam 状态上做更新。

### Q2: 为什么 FSDP 能省内存？

DDP 每卡存 `16N`（完整参数+梯度+优化器）。FSDP 把这些全部 shard 到 G 卡，每卡只存 `16N/G`。

代价是 forward/backward 时要额外通信（all-gather 拼参数、reduce-scatter 分梯度），通信量比 DDP 多 ~50%，但可通过 prefetch 重叠。

峰值时只需临时持有一层完整参数（~几百 MB），远小于整个模型。

### Q3: 为什么 FSDP 跟 TP 互补？

| | FSDP 擅长 | FSDP 不擅长 |
|--|----------|-----------|
| | 减 16N/G 参数内存 | 不减激活内存 |
| | 跨机扩展 | 大 world_size 时 latency 高 |
| | 不改计算逻辑 | 不减单层峰值 |

| | TP 擅长 | TP 不擅长 |
|--|--------|---------|
| | 减激活内存 | 需要 NVLink 高带宽 |
| | 机内低延迟 | 不适合跨机 |
| | 计算时就保持分片 | 实现复杂 |

**互补**：FSDP 跨机分摊参数内存，TP 机内分摊计算和激活。组合后通过 2D DeviceMesh 正交协作。

---

## 十一、推荐学习资源

### 必读

1. **FSDP2 Official Tutorial** — 完整训练流程 + prefetch + mixed precision + checkpoint
   https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html

2. **fully_shard API Reference** — user contract 详细说明
   https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html

3. **torchtitan FSDP doc** — FSDP1→FSDP2 迁移 + 设计决策
   https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md

4. **TP + FSDP Tutorial** — 为什么需要组合 + DeviceMesh 用法
   https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

### 深入

5. **PyTorch FSDP: Experiences on Scaling** (Zhao et al.) — 官方论文
   https://arxiv.org/abs/2304.11277

6. **ZeRO: Memory Optimizations** (Rajbhandari et al.) — FSDP 的理论基础
   https://arxiv.org/abs/1910.02054

7. **Meta Engineering Blog: FSDP** — 直觉性最好的介绍
   https://engineering.fb.com/2021/07/15/open-source/fsdp/
