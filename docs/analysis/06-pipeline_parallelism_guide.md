# Pipeline Parallelism：先懂 Bubble，再懂 Schedule

---

## 一、PP 的基本模型

PP 把模型的 **层** 分配到不同 GPU（而非把单层切分）：

```
32 层 Transformer, PP=4:

  Stage 0 (GPU 0)     Stage 1 (GPU 1)     Stage 2 (GPU 2)     Stage 3 (GPU 3)
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ Layer  0 ~ 7 │───►│ Layer  8 ~15 │───►│ Layer 16 ~23 │───►│ Layer 24 ~31 │
  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
    embed + L0-7         L8-15               L16-23             L24-31 + lm_head

  通信: Stage 之间只传递 激活 (P2P send/recv)
        不传参数、不传梯度 → 通信量极小
```

### PP vs FSDP vs TP 的通信特征

| 并行方式 | 通信内容 | 通信量 | 通信模式 |
|---------|---------|-------|---------|
| FSDP | 参数 + 梯度 | ~3N | all-gather + reduce-scatter (集合) |
| TP | 激活的部分和 | ~每层 2×all-reduce | all-reduce (集合，阻塞) |
| **PP** | **中间激活** | **~B×S×D (很小)** | **P2P send/recv (点对点)** |

PP 通信量最小，适合**跨机**（慢速网络）。

---

## 二、三个核心概念

### 2.1 Microbatch

一个 mini-batch 被拆成 M 个 microbatch，依次喂入 pipeline：

```
Mini-batch (B=8)  →  拆成 M=4 个 microbatch (每个 b=2)

  μ₀=[样本0,1]  μ₁=[样本2,3]  μ₂=[样本4,5]  μ₃=[样本6,7]
```

**为什么要拆？** 如果不拆，Stage 0 做完 forward 后才能给 Stage 1 → 同一时刻只有 1 个 GPU 在工作！拆成 M 份后，Stage 0 处理完 μ₀ 就能把激活传给 Stage 1，自己立刻处理 μ₁ → 多个 stage 可以**同时**工作。

### 2.2 Pipeline Bubble

即使用了 microbatch，仍然有 GPU **空闲等待**的时间——这就是 bubble。

以 GPipe (fill-drain) schedule 为例，PP=4, M=4：

```
时间 ──────────────────────────────────────────────────────────────────►

Stage 0: │F₀│F₁│F₂│F₃│░░░░░░░░░░░░│B₃│B₂│B₁│B₀│
Stage 1: │░░│F₀│F₁│F₂│F₃│░░░░░░░░│B₃│B₂│B₁│B₀│░░│
Stage 2: │░░░░│F₀│F₁│F₂│F₃│░░░░│B₃│B₂│B₁│B₀│░░░░│
Stage 3: │░░░░░░│F₀│F₁│F₂│F₃│B₃│B₂│B₁│B₀│░░░░░░│

         ← fill →            ← drain →
         ░ = bubble (空闲)    F = forward    B = backward
```

**bubble 的来源**：
- **Fill 阶段**：Stage 1 要等 Stage 0 传来 μ₀ 的激活才能开始 → Stage 越靠后等越久
- **Drain 阶段**：backward 是逆序的，Stage 0 要等到最后才能开始 backward

### 2.3 Bubble Rate 公式

```
GPipe bubble rate = (P - 1) / M

P = 4 stages, M = 4 microbatches → bubble = 75%  ← 灾难!
P = 4 stages, M = 32 microbatches → bubble = 9.4%  ← 可以接受
```

**关键洞察**：减少 bubble 的方法只有两个方向：
1. **增大 M**（更多 microbatch）→ 但受 batch size 和内存限制
2. **改进 schedule**（让空闲时间被有用计算填满）

### 2.4 Stage Assignment

决定哪些层放在哪个 GPU 上。最简单的均匀分配：

```
L 层, P stages → 每个 stage 分 L/P 层
```

实际中有**不均衡**问题：
- Stage 0 多了 embedding 层
- Stage P-1 多了 lm_head + loss 计算
- 这导致某些 stage 更慢 → 整个 pipeline 被最慢的 stage 拖累

torchtitan 的做法：`L = k × P - 2`，把 embed 和 lm_head 的计算量考虑进去。

---

## 三、Schedule 进化史

### 3.1 GPipe (Fill-Drain)

```
Stage 0: │F₀│F₁│F₂│F₃│░░░░░░░░░░░░│B₃│B₂│B₁│B₀│
Stage 1: │░░│F₀│F₁│F₂│F₃│░░░░░░░░│B₃│B₂│B₁│B₀│░░│
Stage 2: │░░░░│F₀│F₁│F₂│F₃│░░░░│B₃│B₂│B₁│B₀│░░░░│
Stage 3: │░░░░░░│F₀│F₁│F₂│F₃│B₃│B₂│B₁│B₀│░░░░░░│
```

- 先跑完所有 F，再跑所有 B
- **bubble = (P-1)/M**
- **内存**：Stage 0 需保存所有 M 个 microbatch 的激活（因为 B 在最后才跑）

### 3.2 1F1B (One Forward, One Backward)

```
Stage 0: │F₀│F₁│F₂│F₃│B₀│B₁│B₂│B₃│░░░░░░│
Stage 1: │░░│F₀│F₁│F₂│B₀│F₃│B₁│B₂│B₃│░░░░│
Stage 2: │░░░░│F₀│F₁│B₀│F₂│B₁│F₃│B₂│B₃│░░│
Stage 3: │░░░░░░│F₀│B₀│F₁│B₁│F₂│B₂│F₃│B₃│
```

关键改进：**稳态阶段交替 1F-1B**，边 forward 边 backward。

- **bubble = (P-1)/M**（跟 GPipe 一样！）
- **内存大幅改善**：每个 stage 最多保存 P 个 microbatch 的激活（而非 M 个）
- 因为 backward 提前开始，激活可以提前释放

### 3.3 Interleaved 1F1B

核心思想：每个 GPU 不只负责 1 个 stage，而是负责**多个不连续的 chunk**。

```
PP=4, V=2 (每个 rank 分 2 个 virtual stage):

  Rank 0: Layer 0-3  + Layer 16-19    (chunk 0 和 chunk 4)
  Rank 1: Layer 4-7  + Layer 20-23    (chunk 1 和 chunk 5)
  Rank 2: Layer 8-11 + Layer 24-27    (chunk 2 和 chunk 6)
  Rank 3: Layer 12-15 + Layer 28-31   (chunk 3 和 chunk 7)
```

数据在 pipeline 中走 "V 字形"：Rank 0→1→2→3→0→1→2→3

- **bubble = (P-1) / (V×M)**，V 越大 bubble 越小
- **代价**：通信次数增多（每个 chunk 都要 P2P）
- 需要更多 microbatch 来摊薄通信

### 3.4 Schedule 对比

| Schedule | Bubble Rate | 内存 (激活) | 通信量 |
|----------|------------|-----------|-------|
| GPipe | (P-1)/M | M 个 microbatch | 最少 |
| 1F1B | (P-1)/M | **P** 个 microbatch | 同 GPipe |
| Interleaved 1F1B | **(P-1)/(V×M)** | P 个 | V倍通信 |
| **Zero Bubble** | **≈ 0** | P~2P 个 | 同 1F1B |

---

## 四、Zero Bubble：把 Backward 一拆为二

### 4.1 核心观察

传统 backward (B) 实际做两件事：

```
backward 一个层 y = f(x, W):

  B (input grad):  dL/dx = (∂f/∂x)ᵀ × dL/dy     ← 依赖上游梯度，需要传给前一层
  W (weight grad):  dL/dW = (∂f/∂W)ᵀ × dL/dy     ← 不依赖其他 stage，可以延后

传统做法：B 和 W 绑在一起执行
Zero Bubble：拆开！B 单独执行，W 延后到 bubble 里执行
```

### 4.2 为什么拆开能减 bubble

在 1F1B 中，bubble 出现在 pipeline 的首尾阶段。如果把 W 延后：

```
传统 1F1B (F 和 B 各占 1 个 time slot):

Stage 0: │F₀│F₁│F₂│F₃│B₀│B₁│B₂│B₃│░░░░░░│
                                      ^^^^^^ bubble

Zero Bubble (拆分后, F / B / W 各占约 1/2 time slot):

Stage 0: │F₀│F₁│F₂│F₃│B₀│B₁│B₂│B₃│W₀│W₁│W₂│W₃│
                                      ^^^^^^^^^^ 原来的 bubble 被 W 填满！
```

### 4.3 时间比例

对于 Transformer 的一个层（主要是 matmul）：

```
T_F : T_B : T_W ≈ 1 : 1 : 1

其中:
  T_F = forward 时间
  T_B = input gradient 时间 (需要传给前一个 stage)
  T_W = weight gradient 时间 (不需要传递，可以自由安排)

更精确地: T_B + T_W = 2 × T_F
```

**W 不在关键路径上**——它不产生需要传给其他 stage 的数据，所以可以安排在任何空闲时间。

### 4.4 ZB-H1 和 ZB-H2 图示

**ZB-H1**（bubble 降为 1F1B 的 1/3，内存不增加）：

```
Stage 0: │F₀│F₁│F₂│F₃│B₀│W₀│B₁│W₁│B₂│W₂│B₃│W₃│
Stage 1: │░░│F₀│F₁│F₂│B₀│F₃│W₀│B₁│W₁│B₂│W₂│B₃│W₃│
Stage 2: │░░░░│F₀│F₁│B₀│F₂│W₀│B₁│F₃│W₁│B₂│W₂│B₃│W₃│
Stage 3: │░░░░░░│F₀│B₀│F₁│W₀│B₁│F₂│W₁│B₂│F₃│W₂│B₃│W₃│
```

**ZB-H2**（接近零 bubble，内存可能增加）：把整个 schedule 排成**平行四边形**，所有空隙被 W 填满。

### 4.5 ZB-V Schedule

给每个 rank 分配 2 个 virtual stage（类似 Interleaved），依赖关系形成 "V" 字形。在 `T_F = T_B = T_W` 时，**真正实现零 bubble**，且内存与 1F1B 相同。

---

## 五、PyTorch 中的 PP API

### 5.1 模型分割

```python
# 方法 1: 手动构建每个 stage 的 module
class Stage0(nn.Module):
    def __init__(self):
        self.embed = nn.Embedding(V, D)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(8)])
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x

# 方法 2: torchtitan 风格 — 模型用 ModuleDict 使切割容易
class Transformer(nn.Module):
    def __init__(self):
        self.tok_embeddings = nn.Embedding(V, D)
        self.layers = nn.ModuleDict({
            str(i): TransformerBlock() for i in range(32)
        })
        self.norm = RMSNorm(D)
        self.output = nn.Linear(D, V)
```

### 5.2 创建 PipelineStage

```python
from torch.distributed.pipelining import PipelineStage

# 每个 rank 创建自己负责的 stage
stage = PipelineStage(
    module=my_stage_module,          # 该 stage 的 nn.Module
    stage_index=rank,                 # 第几个 stage
    num_stages=world_size,            # 总 stage 数
    device=torch.device(f"cuda:{rank}"),
    input_args=(example_input,),      # 示例输入（用于 shape 推断）
)
```

### 5.3 选择 Schedule

```python
from torch.distributed.pipelining import (
    ScheduleGPipe,
    Schedule1F1B,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
)

# GPipe
schedule = ScheduleGPipe(stage, n_microbatches=8)

# 1F1B
schedule = Schedule1F1B(stage, n_microbatches=8)

# Interleaved 1F1B (多个 virtual stage per rank)
schedule = ScheduleInterleaved1F1B(stages=[stage0, stage1], n_microbatches=8)

# Zero Bubble
schedule = ScheduleInterleavedZeroBubble(stages=[stage0, stage1], n_microbatches=8)

# 运行
if rank == 0:
    schedule.step(input_batch)
else:
    output = schedule.step()
```

---

## 六、PP 的内存特征

PP 不像 FSDP 那样减参数内存，它减的是**每卡存的层数**：

```
无 PP: 每卡存全部 32 层参数 + 优化器
PP=4: 每卡只存 8 层参数 + 优化器    ← 参数内存减为 1/4
```

但 PP 引入了额外的**激活内存**开销：

```
需要保存 "in-flight" microbatch 的激活:
  GPipe:  M × (每 microbatch 激活)     ← M 可能很大
  1F1B:   P × (每 microbatch 激活)     ← 只有 P 个，好很多
  ZB-V:   P × (每 microbatch 激活)     ← 同 1F1B
```

---

## 七、PP 与 FSDP / TP 的组合（3D 并行）

```
64 GPUs = 2 nodes × 32 GPUs/node
每个 node 有 8×NVLink GPU

3D 配置: PP=4, TP=8, DP(FSDP)=2

  ┌─── Node 0 ──────────────────────────────────────┐
  │  Stage 0          Stage 1                        │
  │  ┌──────────┐    ┌──────────┐                   │
  │  │TP=8 GPUs │    │TP=8 GPUs │    (FSDP 跨 node) │
  │  │GPU 0-7   │    │GPU 8-15  │                   │
  │  └──────────┘    └──────────┘                   │
  └─────────────────────────────────────────────────┘
  ┌─── Node 1 ──────────────────────────────────────┐
  │  Stage 2          Stage 3                        │
  │  ┌──────────┐    ┌──────────┐                   │
  │  │TP=8 GPUs │    │TP=8 GPUs │    (FSDP 跨 node) │
  │  │GPU 32-39 │    │GPU 40-47 │                   │
  │  └──────────┘    └──────────┘                   │
  └─────────────────────────────────────────────────┘

  PP: 跨 node (通信少，慢网络也行)
  TP: node 内 (通信多，需 NVLink)
  FSDP: 相同 stage 跨 node (分摊参数)
```

**各并行切的维度**：

| 并行 | 切什么 | 适合网络 | DeviceMesh 维度 |
|------|-------|---------|---------------|
| TP | 层内矩阵 | NVLink (快) | tp |
| PP | 层间分配 | 跨机网络 (慢也行) | pp |
| FSDP | 参数/梯度/优化器 | 取决于 scope | dp |

---

## 八、Bubble 是 PP 的核心瓶颈——数值感受

PP=8, 不同 schedule, M=24:

| Schedule | Bubble Rate | 相对吞吐 |
|----------|------------|---------|
| GPipe | (8-1)/24 = 29.2% | 70.8% |
| 1F1B | 29.2% (同上) | 70.8% (内存更好) |
| Interleaved 1F1B (V=2) | 14.6% | 85.4% |
| ZB-H1 | ~9.7% | ~90.3% |
| ZB-H2 / ZB-V | **≈ 0%** | **≈ 100%** |

Bubble 从 29% 降到 0%——这就是 schedule 演进的意义。

---

## 九、自测题

**Q1: 为什么需要 microbatch？**
→ 不拆 microbatch 的话，同一时刻只有一个 stage 在工作（其余空闲）。拆成 M 个后，可以流水线起来让多个 stage 同时工作。

**Q2: 1F1B 比 GPipe 的优势在哪？**
→ Bubble rate 一样，但 1F1B 的激活内存从 O(M) 降到 O(P)，因为 backward 提前开始，激活可以提前释放。

**Q3: Zero Bubble 的核心 idea 是什么？**
→ 把 backward 拆成 B(input grad) 和 W(weight grad)。B 在关键路径上（要传给前一个 stage），W 不在（可以延后）。把 W 安排到 bubble 中执行，填满空闲时间。

---

## 十、推荐学习资源

1. **PyTorch Pipeline Parallelism API** — 所有 schedule 的官方定义
   https://docs.pytorch.org/docs/stable/distributed.pipelining.html

2. **PyTorch PP Tutorial** — 最小 2 卡 GPipe 示例
   https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html

3. **GPipe Paper** (Huang et al. 2019) — fill-drain schedule 的原始论文
   https://arxiv.org/abs/1811.06965

4. **Megatron-LM v2** (Narayanan et al. 2021) — 1F1B + Interleaved 1F1B
   https://arxiv.org/abs/2104.04473

5. **Zero Bubble PP** (Qi et al. 2024) — B/W 拆分 + 自动 schedule
   https://arxiv.org/abs/2401.10241

6. **torchtitan FSDP+PP+TP 文档** — 工程实践中的 3D 并行
   https://github.com/pytorch/torchtitan
