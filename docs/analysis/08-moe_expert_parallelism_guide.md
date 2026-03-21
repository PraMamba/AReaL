# MoE：先学单机，再学 Expert Parallelism

---

## 一、MoE 替换的是什么

MoE 替换的是 Transformer 中的 **FFN（MLP）层**。Attention 不变。

```
Dense Transformer:                    MoE Transformer:
┌─────────────────┐                  ┌─────────────────┐
│   Attention     │                  │   Attention     │
├─────────────────┤                  ├─────────────────┤
│   FFN (1个)     │      ───→       │   Router        │
│   gate+up+down  │                  │   ↓ top-k选择   │
│                 │                  │   Expert 0 (FFN)│
│                 │                  │   Expert 1 (FFN)│
│                 │                  │   ...           │
│                 │                  │   Expert N (FFN)│
└─────────────────┘                  └─────────────────┘
参数: 3×D×I                          参数: 3×D×I × N (但每token只用 k 个)
```

**关键性质**：总参数量 = N × 单 expert 参数（巨大），但每个 token 只激活 k 个 expert → 计算量 ≈ k/N × 总参数。

---

## 二、单机 MoE 的四个核心概念

### 2.1 Router（门控网络）

```python
class Router(nn.Module):
    def __init__(self, D, num_experts):
        self.gate = nn.Linear(D, num_experts, bias=False)  # 一个线性层

    def forward(self, x):  # x: [B*S, D]
        logits = self.gate(x)          # [B*S, num_experts]
        scores = softmax(logits)       # 每个 token 对每个 expert 的亲和度
        top_k_scores, top_k_indices = topk(scores, k=2)  # 选最高的 k 个
        return top_k_scores, top_k_indices
```

Router 就是一个 `Linear(D → E)` + softmax + top-k。

### 2.2 Top-k 选择

| 策略 | 含义 | 代表模型 |
|------|------|---------|
| top-1 | 每 token 只选 1 个 expert | Switch Transformer |
| top-2 | 每 token 选 2 个 expert，加权求和 | GShard, Mixtral |
| top-k | 更一般化 | DeepSeek-V3 (top-8 from 256) |

**加权合并**：

```
output_token = Σ (gate_score_i × Expert_i(token))    (只对被选中的 k 个求和)
```

### 2.3 Capacity Factor

因为 tensor shape 必须静态确定，但 token→expert 的分配是动态的：

```
expert_capacity = capacity_factor × (total_tokens / num_experts)

例: 1024 tokens, 8 experts, CF=1.25
    → 每个 expert 最多处理 1024/8 × 1.25 = 160 tokens
    → 如果某 expert 收到 >160 tokens，多余的被 drop（跳过，走残差）
```

**trade-off**：CF 太大 → 浪费内存和计算；CF 太小 → 丢 token 伤性能。

### 2.4 Load Balancing

如果不干预，router 会收敛到只用少数几个 "热门" expert → **routing collapse**。

**传统方案：auxiliary loss**

```
L_aux = α × Σ_i (f_i × P_i)

f_i = 被路由到 expert i 的 token 比例
P_i = router 给 expert i 的平均概率
α = 超参数（太大伤性能，太小不管用）
```

**DeepSeek-V3 方案：auxiliary-loss-free**

```
routing_score = sigmoid(token @ expert_centroid) + bias_i

训练中动态调 bias:
  expert_i 过载 → bias_i -= γ
  expert_i 空闲 → bias_i += γ

bias 只影响路由决策，不影响 gate 权重计算 → 不伤模型性能
```

---

## 三、单个 Token 的完整生命周期（单机）

```
Token "hello" 进入 MoE 层:

① hidden = [1, D]  (来自 Attention 的输出)
                │
② Router:       gate_logits = hidden @ W_gate    → [1, E]   (E=8 experts)
                scores = softmax(gate_logits)     → [0.05, 0.3, 0.01, ..., 0.4, ...]
                top2 = 选最高的 2 个               → expert_3 (score=0.4), expert_1 (score=0.3)
                normalize: g₃=0.4/0.7=0.57, g₁=0.3/0.7=0.43
                │
③ Dispatch:     把 hidden 发送给 expert_3 和 expert_1
                │
④ Expert 计算:  expert_1(hidden) → out₁ = FFN_1(hidden)   [1, D]
                expert_3(hidden) → out₃ = FFN_3(hidden)   [1, D]
                │
⑤ Combine:      output = g₁ × out₁ + g₃ × out₃
                       = 0.43 × out₁ + 0.57 × out₃        [1, D]
                │
⑥              output 送入残差连接 → 下一层
```

---

## 四、批量 Token 的 Dispatch 矩阵视角

实际中不是一个一个 token 处理，而是用**稀疏矩阵**批量路由：

```
6 tokens, 4 experts, top-1:

Token   Router选择    Dispatch Matrix (one-hot)
  t₀  →  Expert 2     [0, 0, 1, 0]
  t₁  →  Expert 0     [1, 0, 0, 0]
  t₂  →  Expert 2     [0, 0, 1, 0]
  t₃  →  Expert 3     [0, 0, 0, 1]
  t₄  →  Expert 1     [0, 1, 0, 0]
  t₅  →  Expert 2     [0, 0, 1, 0]  ← Expert 2 收到 3 个 token!

Expert 2 的 capacity = CF × 6/4 = 1.25 × 1.5 = 2 (向上取整)
→ t₅ 被 drop! (超过 capacity)
```

---

## 五、演进路线

| 模型 | 年份 | Expert 数 | Top-k | 关键贡献 |
|------|------|----------|-------|---------|
| Sparsely-Gated MoE | 2017 | 数千(LSTM) | k=2-4 | noisy top-k gating + aux loss |
| GShard | 2020 | 2048 | k=2 | 分布式 MoE + capacity factor |
| Switch Transformer | 2021 | 128 | **k=1** | 简化为 top-1 + 更简单的 aux loss |
| Expert Choice | 2022 | — | 反转 | **expert 选 token**，天然负载均衡 |
| Mixtral 8×7B | 2024 | 8 | k=2 | 开源标杆，证明 MoE 实用性 |
| DeepSeek-V3 | 2024 | 256+1 shared | k=8 | **auxiliary-loss-free** + fine-grained experts |

---

## 六、Expert Parallelism (EP)

当 expert 太多放不下单卡时，把不同 expert 分到不同 GPU。

### 6.1 核心机制：两次 all-to-all

```
4 GPUs, 8 experts → 每 GPU 放 2 个 expert

Step 1: Router (每卡本地计算)
  每卡对自己的 tokens 跑 router → 知道每个 token 该去哪个 expert

Step 2: all-to-all DISPATCH (token → expert 所在 GPU)
  ┌────────────┐                        ┌────────────┐
  │ GPU 0      │  tokens for E0,E1 ←──  │ GPU 1      │
  │ holds E0,E1│  tokens for E0,E1 ←──  │ holds E2,E3│
  │            │  ──→ tokens for E2,E3  │            │
  │            │  ──→ tokens for E2,E3  │            │
  └────────────┘                        └────────────┘
  (类似 Ulysses 的 all-to-all，但按 expert 目标而非 head)

Step 3: Expert 计算 (每卡独立)
  GPU 0: E0 处理收到的 tokens, E1 处理收到的 tokens
  GPU 1: E2 处理收到的 tokens, E3 处理收到的 tokens

Step 4: all-to-all COMBINE (结果 → token 原来所在 GPU)
  把 expert 输出发回 token 来源 GPU
  每 GPU 拿到自己 tokens 的 expert 输出，加权合并
```

### 6.2 完整数据流（追踪一个 token）

```
Token "hello" 在 GPU 0 上:

① GPU 0 本地 Router: "hello" → Expert 5 (在 GPU 2)
                │
② all-to-all DISPATCH:
   GPU 0 把 "hello" 的 hidden 发给 GPU 2
   (同时 GPU 0 从其他 GPU 收到要给 E0/E1 的 tokens)
                │
③ GPU 2 上: Expert 5 计算 FFN("hello") → result
                │
④ all-to-all COMBINE:
   GPU 2 把 result 发回 GPU 0
                │
⑤ GPU 0: 收到 result，乘以 gate score，加入 output
```

### 6.3 EP 的通信量

```
每次 all-to-all: ~B×S×D (所有 token 的 hidden)
两次 all-to-all: ~2×B×S×D

对比 TP: 每层 2× all-reduce ~4×B×S×D
→ EP 通信量更小（且 all-to-all 可以更好重叠）
```

### 6.4 EP 与其他并行的关系

```
典型 MoE 模型的并行配置:

  Attention 层:  TP (层内切矩阵) + FSDP (跨机分参数)
  MoE FFN 层:    EP (不同 expert 在不同 GPU)

  ┌─── Node 0 (8 GPUs) ─────────────────────┐
  │  Attention: TP=8 (所有 GPU 协作)         │
  │  MoE:      EP=8 (每 GPU 放不同 experts)  │
  │            all-to-all dispatch/combine    │
  └──────────────────────────────────────────┘
```

---

## 七、Expert Tensor Parallelism (ETP)

当**单个 expert 太大**（如 DeepSeek-V3 的 fine-grained experts）时，对 expert 内部再做 TP：

```
ETP = EP + TP inside each expert

例: 256 experts, 64 GPUs, ETP=2
  → EP 把 256 experts 分到 32 组 (每组 8 experts)
  → 每组 2 个 GPU 对 8 个 experts 各做 TP (ColwiseParallel + RowwiseParallel)

Expert FFN:
  GPU 0: gate_proj 的前半列, up_proj 的前半列, down_proj 的前半行
  GPU 1: gate_proj 的后半列, up_proj 的后半列, down_proj 的后半行
  → expert 内部需要 all-reduce (跟普通 TP 一样)
```

---

## 八、DeepSeek-V3 的 DualPipe

DeepSeek-V3 把 MoE 层的 forward 拆成 4 个组件来重叠通信和计算：

```
一个 chunk 的 forward:
  ① Attention (计算)
  ② all-to-all dispatch (通信)     ← 可与下一个 chunk 的 Attention 重叠
  ③ MLP / Expert (计算)
  ④ all-to-all combine (通信)       ← 可与下一个 chunk 的 MLP 重叠

backward 中 Attention 和 MLP 再拆 B/W (类似 Zero Bubble)
→ 通信-计算比从 1:1 降低到接近全重叠
```

---

## 九、总结表：一个 token 在 MoE + EP 中的完整路径

| 步骤 | 发生在 | 操作 | 通信 |
|------|--------|------|------|
| 1. Attention | 本地 GPU (TP group) | QKV → attn → O proj | all-reduce (TP) |
| 2. Router | 本地 GPU | `gate(hidden) → top-k` | 无 |
| 3. Dispatch | EP group | token hidden → expert 所在 GPU | **all-to-all** |
| 4. Expert FFN | expert 所在 GPU | `down(SiLU(gate(x)) * up(x))` | (ETP: all-reduce) |
| 5. Combine | EP group | expert output → token 原 GPU | **all-to-all** |
| 6. 加权合并 | 本地 GPU | `Σ gate_score × output` | 无 |
| 7. 残差连接 | 本地 GPU | `x + moe_output` | 无 |

---

## 十、推荐资源

**基础**
1. **Visual Guide to MoE** (Maarten Grootendorst) — 图解从 router 到 capacity
   https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts
2. **HuggingFace MoE Explained** — 从 Sparsely-Gated 到 Mixtral 的完整演进
   https://huggingface.co/blog/moe
3. **MoE Load Balancing Review** — GShard→Switch→DeepSeek-V3 负载均衡演进
   https://huggingface.co/blog/NormalUhr/moe-balance

**论文**
4. **Sparsely-Gated MoE** (Shazeer et al. 2017) — 奠基论文
   https://arxiv.org/abs/1701.06538
5. **Switch Transformer** (Fedus et al. 2021) — top-1 + simplified aux loss
   https://arxiv.org/abs/2101.03961
6. **DeepSeek-V3 Technical Report** — auxiliary-loss-free + DualPipe
   https://arxiv.org/abs/2412.19437

**EP 实现**
7. **Distributed MoE and EP** (Bruno Magalhaes) — EP all-to-all 详解
   https://brunomaga.github.io/Mixture-of-Experts
