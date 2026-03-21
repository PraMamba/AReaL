# Context Parallelism：只盯 Ulysses，画清一张图

---

## 一、Ulysses 一句话

> 输入按 **seq 维度**切给 P 张卡 → attention 前用 **all-to-all** 换成按 **head 维度**切 → 每卡算完整序列的部分 head → attention 后再 **all-to-all** 换回按 seq 切。

所有魔法就在这两次 all-to-all 里。

---

## 二、为什么需要 Context Parallelism

Attention 的核心计算是 `softmax(Q @ K^T) @ V`，其中 **每个 token 的 Q 都要跟所有 token 的 K 交互**。

如果序列按 seq 维度切给 P 张卡，每张卡只有 S/P 个 token：

```
Rank 0 有 Q[0:S/P]，但它需要 K[0:S] 的全部才能算 attention！
```

这就是 CP 要解决的核心矛盾：**seq 被切了，但 attention 需要完整 seq**。

Ulysses 的解法：不去拼完整的 K，而是**换一种切法**——按 head 切。每卡拿到所有 token 但只算部分 head。

---

## 三、核心图：两次 all-to-all 的布局变换

以 **S=8 tokens, H=4 heads, P=2 GPUs** 为例。

### 3.1 初始状态：按 seq 切

```
输入 hidden_states 按序列维度切分（这是 MLP/Norm 的自然布局）：

  Rank 0: tokens 0-3 的 hidden [4, D]     ← seq 的前半
  Rank 1: tokens 4-7 的 hidden [4, D]     ← seq 的后半
```

### 3.2 QKV 投影（每卡独立）

```
  Rank 0:                              Rank 1:
  Q₀ = hidden₀ @ Wq                   Q₁ = hidden₁ @ Wq
  K₀ = hidden₀ @ Wk                   K₁ = hidden₁ @ Wk
  V₀ = hidden₀ @ Wv                   V₁ = hidden₁ @ Wv

  Q₀: [4, H, d] = [4, 4, d]          Q₁: [4, 4, d]
       seq=0~3, 全部4个head                seq=4~7, 全部4个head
```

此时每卡有**部分 seq × 全部 head**。

### 3.3 第一次 all-to-all：seq 切 → head 切

```
━━━ all-to-all 前 ━━━                    ━━━ all-to-all 后 ━━━

  Rank 0:                                Rank 0:
  Q: [seq 0-3, head 0-3, d]             Q: [seq 0-7, head 0-1, d]
      ▲ 部分seq, 全部head                    ▲ 全部seq, 部分head

  Rank 1:                                Rank 1:
  Q: [seq 4-7, head 0-3, d]             Q: [seq 0-7, head 2-3, d]
      ▲ 部分seq, 全部head                    ▲ 全部seq, 部分head
```

**all-to-all 做了什么？** 把"按 seq 分片"变成"按 head 分片"：

```
 all-to-all 前:                          all-to-all 后:
 ┌─────────────────────┐                ┌─────────────────────┐
 │  Rank 0             │                │  Rank 0             │
 │  seq: ████░░░░      │                │  seq: ████████      │
 │  head: h0 h1 h2 h3  │    all-to-all  │  head: h0 h1        │
 ├─────────────────────┤   ──────────►  ├─────────────────────┤
 │  Rank 1             │                │  Rank 1             │
 │  seq: ░░░░████      │                │  seq: ████████      │
 │  head: h0 h1 h2 h3  │                │  head: h2 h3        │
 └─────────────────────┘                └─────────────────────┘

 █ = 数据所在位置
```

**同时对 Q, K, V 都做这个变换。** 通信量 = 3 × S × H × d（QKV 三者）。

### 3.4 Attention 计算（每卡完全独立！）

```
  Rank 0: 拥有 seq 0-7 的 head 0-1
    → softmax(Q[0:8, h0:h2] @ K[0:8, h0:h2]^T) @ V[0:8, h0:h2]
    → attn_out₀: [8, 2, d]    ← 全序列，但只有 head 0-1

  Rank 1: 拥有 seq 0-7 的 head 2-3
    → softmax(Q[0:8, h2:h4] @ K[0:8, h2:h4]^T) @ V[0:8, h2:h4]
    → attn_out₁: [8, 2, d]    ← 全序列，但只有 head 2-3

  *** 无需跨卡通信！每卡独立计算自己负责的 head ***
  *** 兼容 FlashAttention、causal mask、sparse attention ***
```

### 3.5 第二次 all-to-all：head 切 → seq 切（换回来）

```
 all-to-all 前:                          all-to-all 后:
 ┌─────────────────────┐                ┌─────────────────────┐
 │  Rank 0             │                │  Rank 0             │
 │  seq: ████████      │                │  seq: ████░░░░      │
 │  head: h0 h1        │    all-to-all  │  head: h0 h1 h2 h3  │
 ├─────────────────────┤   ──────────►  ├─────────────────────┤
 │  Rank 1             │                │  Rank 1             │
 │  seq: ████████      │                │  seq: ░░░░████      │
 │  head: h2 h3        │                │  head: h0 h1 h2 h3  │
 └─────────────────────┘                └─────────────────────┘
```

现在回到了 **部分 seq × 全部 head** 的布局 → 可以继续做 out_proj、MLP、Norm 等逐 token 操作。

### 3.6 O proj + MLP（每卡独立）

```
  Rank 0: attn_out[seq 0-3, all heads] → o_proj → MLP → hidden₀'
  Rank 1: attn_out[seq 4-7, all heads] → o_proj → MLP → hidden₁'

  *** 无需通信，o_proj 和 MLP 没有 seq 维度上的依赖 ***
```

---

## 四、完整一层的布局变化时间线

```
             布局状态                 所在维度         通信
             ─────────                ─────────         ─────
  输入       [S/P, D]                 seq 切            —
             │
  QKV proj   [S/P, H, d] × 3         seq 切            —
             │
             ▼ ━━ all-to-all #1 ━━                      all-to-all
             │
  attention  [S, H/P, d]             head 切            —  (本地计算)
             │
             ▼ ━━ all-to-all #2 ━━                      all-to-all
             │
  out_proj   [S/P, H, d] → [S/P, D]  seq 切            —
             │
  MLP        [S/P, D]                seq 切            —
             │
  输出       [S/P, D]                seq 切            —
             └→ 送入下一层（同样的布局，循环往复）
```

**每层通信：2 次 all-to-all**。

---

## 五、通信量分析：为什么 Ulysses 比 Megatron-SP 好

### all-to-all 的通信量

```
每次 all-to-all 总数据量 = S × H × d = S × D  （因为 H × d = D）
每条链路传输量 = S × D / P × (P-1)/P ≈ S × D / P
```

两次 all-to-all（attention 前后）：

```
Ulysses 每层通信 = 2 × S × D / P （per link）
```

### Megatron-SP（TP 附属的序列并行）

```
每层通信 = 4 × S × D （2次 all-gather + 2次 reduce-scatter, per link 不随 P 减小）
```

**关键差异**：Ulysses 通信量随 P 增大**线性减小**（S 也按 P 切了），Megatron-SP **不变**。

```
S/P 固定（序列和设备同比扩展）时:
  Ulysses:    通信量 = 常数     ← 完美扩展!
  Megatron-SP: 通信量 = O(P)    ← 越多卡通信越多
```

---

## 六、Ulysses 的约束：head 数量

Ulysses 按 head 维度切分 → **P 必须整除 H**（attention heads 数量）。

```
Qwen2-7B: H=28, KV_H=4 (GQA)
  最大 Ulysses CP = min(28, 4) = 4  ← 受 KV heads 限制!
```

解决方案：**Ulysses + Ring Attention** 混合。先用 Ulysses 切 head（P₁ 路），再用 Ring 切剩余 seq（P₂ 路），总 CP = P₁ × P₂。

---

## 七、Ulysses vs Ring Attention 对比

| | Ulysses | Ring Attention |
|--|---------|---------------|
| **通信方式** | all-to-all (两次) | P2P ring (多轮) |
| **通信量** | 2SD/P (随 P 减小) | 2SD (不随 P 减小) |
| **限制** | P ≤ H (head 数) | 无限制 |
| **计算模式** | 每卡：全序列 × 部分 head | 每卡：部分序列 × 全部 head (KV 轮转) |
| **FlashAttn** | 直接兼容 | 需要特殊集成 |
| **负载均衡** | 天然均衡 | causal mask 下不均衡 |

---

## 八、与 TP 的关系

Ulysses 和 TP 都按 head 维度切 → **竞争同一资源**。

```
H = 32 heads
如果 TP=8 (每卡 4 heads)，Ulysses 最多再切 CP=4
如果 TP=4 (每卡 8 heads)，Ulysses 最多 CP=8
```

在 4D 并行中的典型配置：

```
mesh_4d = init_device_mesh("cuda", (dp, cp, tp, pp),
                            mesh_dim_names=("dp", "cp", "tp", "pp"))

例: 64 GPUs = DP=2 × CP=4 × TP=4 × PP=2
```

---

## 九、DTensor 视角下的 Ulysses

用前面学过的 DTensor 语言重新描述：

```
QKV 投影后, Q 的布局:
  mesh: (cp=P,)
  placement: Shard(0)    ← 按 seq 维度切 (dim=0 of [S, H, d])

第一次 all-to-all:
  redistribute: Shard(0) → Shard(1)    ← 从按 seq 切 变成 按 head 切
  (all-to-all 就是 Shard(dim_a) → Shard(dim_b) 的 redistribute!)

attention 后, output 布局:
  placement: Shard(1)    ← 按 head 切

第二次 all-to-all:
  redistribute: Shard(1) → Shard(0)    ← 换回按 seq 切
```

**all-to-all 本质上就是 DTensor 的 `Shard(dim_a) → Shard(dim_b)` redistribute**。

---

## 十、自测

**Q: Ulysses 的 all-to-all 在做什么布局变换？**
→ QKV 从 `[S/P, H, d]`（按 seq 切）变成 `[S, H/P, d]`（按 head 切），attention 算完后反向变回。

**Q: 为什么 attention 计算不需要跨卡通信？**
→ 因为每卡拥有完整序列的部分 head，而 multi-head attention 中不同 head 之间本来就独立计算。

**Q: Ulysses 的约束是什么？**
→ CP 并行度 ≤ attention heads 数量（GQA 下受 KV heads 限制）。

---

## 十一、推荐资源

1. **DeepSpeed Ulysses Paper** — 原论文，Fig.2 是核心图
   https://arxiv.org/abs/2309.14509

2. **Snowflake Arctic Ulysses Blog** — 训练+推理视角，通信量对比清晰
   https://www.snowflake.com/en/engineering-blog/ulysses-low-latency-llm-inference/

3. **Context Parallelism Intro** (insujang) — Ulysses vs Ring 对比
   https://insujang.github.io/2024-09-20/introducing-context-parallelism/

4. **Ulysses + Ring Attention 实践** (HuggingFace blog) — 融合方案
   https://huggingface.co/blog/exploding-gradients/ulysses-ring-attention

5. **HF Accelerate SP 文档** — Ulysses 集成指南
   https://huggingface.co/docs/accelerate/en/concept_guides/sequence_parallelism
