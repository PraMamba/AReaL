# Tensor Parallelism：从矩阵乘法视角彻底搞懂

> **目标**：能手动画出 TP 下每一步的 tensor shape、切分维度、通信操作。

---

## 一、TP 的核心思想（一句话）

把 `Y = X @ W` 中的 **权重矩阵 W** 切给多张卡，每张卡算一部分，最后通过通信拼回正确结果。

切法只有两种：**按列切（Column）** 和 **按行切（Row）**。

---

## 二、Column Parallel：按列切权重

### 2.1 数学推导

```
Y = X @ W        X: [B,S, D_in]    W: [D_in, D_out]    Y: [B,S, D_out]
```

把 W 沿 `D_out` 维度（列）切成 T 份（T = TP size）：

```
W = [W₀ | W₁]    W₀: [D_in, D_out/T]    W₁: [D_in, D_out/T]
```

每个 rank 拿一片 Wᵢ，**输入 X 保持完整（Replicate）**：

```
Rank 0: Y₀ = X @ W₀    →  Y₀: [B,S, D_out/T]    ← 只算了输出的前一半列
Rank 1: Y₁ = X @ W₁    →  Y₁: [B,S, D_out/T]    ← 只算了输出的后一半列
```

全局正确结果 = 把 Y₀ 和 Y₁ 沿最后维度拼起来：`Y = [Y₀ | Y₁]`

### 2.2 完整图示（TP=2）

```
              Rank 0                              Rank 1
         ┌──────────────┐                    ┌──────────────┐
   X     │ ████████████ │  (完整, Replicate)  │ ████████████ │
[B,S,D]  └──────┬───────┘                    └──────┬───────┘
                │                                    │
                ▼                                    ▼
   W₀    ┌──────────┐                        ┌──────────┐    W₁
[D,D/2]  │ ████████ │  (Shard(0) 列切)       │ ████████ │  [D,D/2]
         └──────┬───┘                        └──────┬───┘
                │  local matmul                      │  local matmul
                ▼                                    ▼
   Y₀    ┌──────────┐                        ┌──────────┐    Y₁
[B,S,D/2]│ ████████ │                        │ ████████ │  [B,S,D/2]
         └──────────┘                        └──────────┘
                │                                    │
                └─────── 输出 Shard(-1) ─────────────┘
                         (沿最后维度被切)
```

**通信**: forward **不需要通信**！输出自然就是 `Shard(-1)`。
**PyTorch DTensor 语言**: `input=Replicate, weight=Shard(0) → output=Shard(-1)`

### 2.3 对应 PyTorch API

```python
ColwiseParallel()
# weight: Shard(0) — 按 out_features 切
# input:  Replicate (默认)
# output: Shard(-1) — 按最后维度切
```

---

## 三、Row Parallel：按行切权重

### 3.1 数学推导

```
Z = Y @ W        Y: [B,S, D_in]    W: [D_in, D_out]    Z: [B,S, D_out]
```

把 W 沿 `D_in` 维度（行）切成 T 份：

```
W = [W₀]    W₀: [D_in/T, D_out]
    [W₁]    W₁: [D_in/T, D_out]
```

相应地，**输入 Y 也必须沿最后维度被切**（正好接 ColwiseParallel 的输出！）：

```
Y = [Y₀ | Y₁]    Y₀: [B,S, D_in/T]    Y₁: [B,S, D_in/T]
```

每个 rank 做局部乘法：

```
Rank 0: Z₀ = Y₀ @ W₀    →  Z₀: [B,S, D_out]    ← 部分和！
Rank 1: Z₁ = Y₁ @ W₁    →  Z₁: [B,S, D_out]    ← 部分和！
```

全局正确结果 = `Z = Z₀ + Z₁` → 需要 **all-reduce(SUM)**！

### 3.2 完整图示（TP=2）

```
              Rank 0                              Rank 1
   Y₀    ┌──────────┐                        ┌──────────┐    Y₁
[B,S,D/2]│ ████████ │  (Shard(-1), 来自上层)  │ ████████ │  [B,S,D/2]
         └──────┬───┘                        └──────┬───┘
                │                                    │
                ▼                                    ▼
   W₀    ┌──────────┐                        ┌──────────┐    W₁
[D/2,D]  │ ████████ │  (Shard(1) 行切)       │ ████████ │  [D/2,D]
         └──────┬───┘                        └──────┬───┘
                │  local matmul                      │  local matmul
                ▼                                    ▼
   Z₀    ┌──────────┐                        ┌──────────┐    Z₁
[B,S,D]  │ 部分和    │                        │ 部分和    │  [B,S,D]
         └──────┬───┘                        └──────┬───┘
                │                                    │
                └──────── all-reduce (SUM) ──────────┘
                                 │
                                 ▼
                          Z = Z₀ + Z₁
              Rank 0: [B,S,D] ✅ 完整       Rank 1: [B,S,D] ✅ 完整
                          (Replicate)
```

**通信**: forward 需要 **1 次 all-reduce**。
**DTensor 语言**: `input=Shard(-1), weight=Shard(1) → output=Partial(SUM) → Replicate`

### 3.3 对应 PyTorch API

```python
RowwiseParallel()
# weight: Shard(1) — 按 in_features 切
# input:  Shard(-1) (默认，接 ColwiseParallel 输出)
# output: Replicate (默认，all-reduce 后)
```

---

## 四、经典配对：Column → Row = 只需 1 次 all-reduce

这是 Megatron-LM 论文的核心贡献。两层 Linear 串联时：

```
X ──→ [ColwiseParallel W₁] ──→ activation ──→ [RowwiseParallel W₂] ──→ Z
      无通信                                    1次 all-reduce
```

ColwiseParallel 的输出是 `Shard(-1)` → 正好是 RowwiseParallel 需要的输入！
**中间不需要任何通信**，只在最后 all-reduce 一次。

---

## 五、MLP (SwiGLU) 的完整 TP 数据流

以 Qwen2 MLP 为例：`down_proj( SiLU(gate_proj(x)) * up_proj(x) )`

```
                    Rank 0                          Rank 1
                    ──────                          ──────
  输入 x          [B,S,D] Replicate               [B,S,D] Replicate
                     │                                │
   ┌─────────────────┤                                ├─────────────────┐
   │                 │                                │                 │
   ▼                 ▼                                ▼                 ▼
  gate_proj        up_proj                          gate_proj        up_proj
  (Colwise)        (Colwise)                        (Colwise)        (Colwise)
  W_gate₀          W_up₀                           W_gate₁          W_up₁
  [D,I/2]          [D,I/2]                         [D,I/2]          [D,I/2]
   │                 │                                │                 │
   ▼                 ▼                                ▼                 ▼
  g₀ [B,S,I/2]    u₀ [B,S,I/2]                    g₁ [B,S,I/2]    u₁ [B,S,I/2]
   │                 │                                │                 │
   ▼                 │                                ▼                 │
  SiLU(g₀)          │                               SiLU(g₁)          │
   │                 │                                │                 │
   └──── × ─────────┘                                └──── × ─────────┘
         │                                                  │
         ▼                                                  ▼
  mid₀ [B,S,I/2]    Shard(-1)                      mid₁ [B,S,I/2]   Shard(-1)
         │                                                  │
         ▼                                                  ▼
  down_proj (Rowwise)                               down_proj (Rowwise)
  W_down₀ [I/2,D]                                  W_down₁ [I/2,D]
         │                                                  │
         ▼                                                  ▼
  z₀ [B,S,D] Partial                               z₁ [B,S,D] Partial
         │                                                  │
         └──────────── all-reduce (SUM) ────────────────────┘
                              │
                              ▼
                    z [B,S,D] Replicate ✅
                    (送入残差连接)

通信: 整个 MLP 只有 1 次 all-reduce (在 down_proj 输出后)
```

### PyTorch TP Plan

```python
tp_plan = {
    "mlp.gate_proj": ColwiseParallel(),   # W: Shard(0), out: Shard(-1)
    "mlp.up_proj":   ColwiseParallel(),   # W: Shard(0), out: Shard(-1)
    "mlp.down_proj": RowwiseParallel(),   # W: Shard(1), in: Shard(-1), out: Replicate
}
```

---

## 六、Attention 的完整 TP 数据流

Attention 有天然的并行性：多个 head 可以分给不同 rank！

### 6.1 QKV 投影 (ColwiseParallel)

```
输入 x [B,S,D]  (Replicate)

  Rank 0:                                Rank 1:
  q_proj 持有前 H/2 个 head 的权重          q_proj 持有后 H/2 个 head 的权重
  W_q₀ [D, (H/2)×d]                      W_q₁ [D, (H/2)×d]

  Q₀ = x @ W_q₀  →  [B, S, (H/2)×d]     Q₁ = x @ W_q₁  →  [B, S, (H/2)×d]

  同理 K₀, V₀                             同理 K₁, V₁
```

**每个 rank 只负责一半的 attention heads**。`wq/wk/wv` 全部 ColwiseParallel。

### 6.2 Attention 计算（完全本地！）

```
  Rank 0:                                Rank 1:
  Q₀,K₀,V₀ → reshape 为 H/2 个 head      Q₁,K₁,V₁ → reshape 为 H/2 个 head
  → softmax(Q₀K₀ᵀ/√d) @ V₀               → softmax(Q₁K₁ᵀ/√d) @ V₁
  → attn_out₀ [B, S, (H/2)×d]            → attn_out₁ [B, S, (H/2)×d]

  *** 无需跨 rank 通信！每个 rank 独立算自己的 head ***
```

GQA 也适用：如果 TP=2 且有 4 个 KV heads，每个 rank 分 2 个 KV head。

### 6.3 Output 投影 (RowwiseParallel)

```
  Rank 0:                                Rank 1:
  attn_out₀ [B,S,(H/2)×d]  Shard(-1)     attn_out₁ [B,S,(H/2)×d]  Shard(-1)
       │                                       │
       ▼                                       ▼
  o_proj (Rowwise)                         o_proj (Rowwise)
  W_o₀ [(H/2)×d, D]                       W_o₁ [(H/2)×d, D]
       │                                       │
       ▼                                       ▼
  out₀ [B,S,D] Partial                    out₁ [B,S,D] Partial
       │                                       │
       └──────── all-reduce (SUM) ─────────────┘
                        │
                        ▼
              out [B,S,D] Replicate ✅
              (送入残差连接)
```

### 6.4 Attention 完整图

```
x [B,S,D] Replicate
     │
     ├──→ q_proj (Colwise) ──→ Q [B,S,(H/2)d] Shard(-1)  ─┐
     ├──→ k_proj (Colwise) ──→ K [B,S,(KVH/2)d] Shard(-1)  ├─→ local attention
     └──→ v_proj (Colwise) ──→ V [B,S,(KVH/2)d] Shard(-1)  ─┘  (无通信)
                                                                │
                                                                ▼
                                              attn_out [B,S,(H/2)d] Shard(-1)
                                                                │
                                                                ▼
                                              o_proj (Rowwise) + all-reduce
                                                                │
                                                                ▼
                                              out [B,S,D] Replicate ✅

通信: 整个 Attention 也只有 1 次 all-reduce (在 o_proj 输出后)
```

### PyTorch TP Plan

```python
tp_plan = {
    "self_attn.q_proj": ColwiseParallel(),
    "self_attn.k_proj": ColwiseParallel(),
    "self_attn.v_proj": ColwiseParallel(),
    "self_attn.o_proj": RowwiseParallel(),
}
```

---

## 七、完整 TransformerBlock 在 TP 下的数据流

```
x [B,S,D]  Replicate
│
├─ residual ─────────────────────────────────────────────────────────┐
│                                                                    │
▼                                                                    │
RMSNorm (input_layernorm)      ← 需要完整 D 维度, 输入必须 Replicate  │
│                                                                    │
▼                                                                    │
┌──── Attention TP Block ────────────────────────────────┐           │
│ q/k/v_proj (Colwise) → local attn → o_proj (Rowwise)  │           │
│                                     ↓ all-reduce       │           │
└────────────────────────────────────────────────────────┘           │
│ output: Replicate                                                  │
▼                                                                    │
+ residual ◄─────────────────────────────────────────────────────────┘
│
├─ residual ─────────────────────────────────────────────────────────┐
│                                                                    │
▼                                                                    │
RMSNorm (post_attn_layernorm)                                       │
│                                                                    │
▼                                                                    │
┌──── MLP TP Block ─────────────────────────────────────┐           │
│ gate/up_proj (Colwise) → SiLU×up → down_proj (Rowwise)│           │
│                                     ↓ all-reduce      │           │
└───────────────────────────────────────────────────────┘           │
│ output: Replicate                                                  │
▼                                                                    │
+ residual ◄─────────────────────────────────────────────────────────┘
│
▼
x' [B,S,D]  Replicate   ← 可以直接送入下一层！

通信总计: forward 2 次 all-reduce（Attention 1 次 + MLP 1 次）
         backward 2 次 all-reduce（对称）
         = 每层 4 次 all-reduce
```

---

## 八、Sequence Parallel (SP) — 减激活内存

### 8.1 问题

上面的 TP 中，RMSNorm 的输入是 `Replicate`——每个 rank 都存完整的 `[B,S,D]` 激活。
如果 S=8192, D=4096, B=4, bf16：每个 rank 存 `4×8192×4096×2 = 256 MB`，L 层就是 GB 级。

**SP 的思路**：RMSNorm 和 Dropout 是逐元素操作，不需要跨 D 维度的信息。可以把激活沿 **序列维度 S** 切分！

### 8.2 SP 下的数据流变化

```
之前 (纯 TP):
  RMSNorm 输入: [B,S,D] Replicate    ← 每个 rank 存完整激活
  Attention 输入: [B,S,D] Replicate

之后 (TP + SP):
  RMSNorm 输入: [B,S/T,D] Shard(1)  ← 每个 rank 只存 1/T 的序列
  Attention 输入前: all-gather 拼回 [B,S,D] Replicate
  Attention 输出后: reduce-scatter 切回 [B,S/T,D] Shard(1)
```

### 8.3 TP+SP 的通信变化

| 操作 | 纯 TP | TP + SP |
|------|-------|---------|
| Attn 输出 | all-reduce | **reduce-scatter** (输出变 Shard(1)) |
| MLP 输出 | all-reduce | **reduce-scatter** (输出变 Shard(1)) |
| Attn 输入 | 无 | **all-gather** (拼回 Replicate) |
| MLP 输入 | 无 | **all-gather** (拼回 Replicate) |

通信量不变！`all-reduce = reduce-scatter + all-gather`。
但 SP 把激活内存减为 1/T —— 这对长序列训练至关重要。

### 8.4 PyTorch API

```python
tp_plan = {
    "input_layernorm":        SequenceParallel(),       # Shard(1) in/out
    "self_attn": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),            # all-gather 拼回
    ),
    "self_attn.q_proj":       ColwiseParallel(),
    "self_attn.k_proj":       ColwiseParallel(),
    "self_attn.v_proj":       ColwiseParallel(),
    "self_attn.o_proj":       RowwiseParallel(output_layouts=Shard(1)),  # reduce-scatter
    "post_attention_layernorm": SequenceParallel(),
    "mlp": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "mlp.gate_proj":          ColwiseParallel(),
    "mlp.up_proj":            ColwiseParallel(),
    "mlp.down_proj":          RowwiseParallel(output_layouts=Shard(1)),
}
```

---

## 九、总结表：每一层的切分与通信

以 Qwen2 DecoderLayer 为例，TP=2：

| 组件 | Parallel Style | 权重 placement | 权重 local shape | 输入 layout | 输出 layout | 通信 |
|------|---------------|---------------|-----------------|------------|------------|------|
| input_layernorm | SP | — (D 个参数) | [D] | Shard(1) | Shard(1) | 无 |
| **→ 进 Attention** | PrepareInput | — | — | Shard(1) → Replicate | — | **all-gather** |
| q_proj | Colwise | Shard(0) | [H×d/T, D] | Replicate | Shard(-1) | 无 |
| k_proj | Colwise | Shard(0) | [KVH×d/T, D] | Replicate | Shard(-1) | 无 |
| v_proj | Colwise | Shard(0) | [KVH×d/T, D] | Replicate | Shard(-1) | 无 |
| attention 计算 | local | — | — | Shard (per head) | Shard (per head) | 无 |
| o_proj | Rowwise | Shard(1) | [D, H×d/T] | Shard(-1) | Shard(1) | **reduce-scatter** |
| residual add | — | — | — | Shard(1) | Shard(1) | 无 |
| post_attn_norm | SP | — | [D] | Shard(1) | Shard(1) | 无 |
| **→ 进 MLP** | PrepareInput | — | — | Shard(1) → Replicate | — | **all-gather** |
| gate_proj | Colwise | Shard(0) | [I/T, D] | Replicate | Shard(-1) | 无 |
| up_proj | Colwise | Shard(0) | [I/T, D] | Replicate | Shard(-1) | 无 |
| SiLU × up | local | — | — | Shard(-1) | Shard(-1) | 无 |
| down_proj | Rowwise | Shard(1) | [D, I/T] | Shard(-1) | Shard(1) | **reduce-scatter** |
| residual add | — | — | — | Shard(1) | Shard(1) | 无 |

**每层通信**: 2× all-gather + 2× reduce-scatter = 等价于 2× all-reduce

---

## 十、Qwen2-7B 数值示例 (TP=2)

| 参数 | 全局 shape | Rank 0 本地 shape | 切法 |
|------|-----------|-----------------|------|
| q_proj.weight | [3584, 3584] | [1792, 3584] | Shard(0): 28→14 heads |
| k_proj.weight | [512, 3584] | [256, 3584] | Shard(0): 4→2 KV heads |
| v_proj.weight | [512, 3584] | [256, 3584] | Shard(0) |
| o_proj.weight | [3584, 3584] | [3584, 1792] | Shard(1) |
| gate_proj.weight | [18944, 3584] | [9472, 3584] | Shard(0) |
| up_proj.weight | [18944, 3584] | [9472, 3584] | Shard(0) |
| down_proj.weight | [3584, 18944] | [3584, 9472] | Shard(1) |
| RMSNorm.weight | [3584] | [3584] | 不切 (太小) |

**TP=2 后每 rank 参数内存**: 原来 ~232M/层 → ~116M/层，减半。
**激活内存** (with SP): 也减半。

---

## 十一、为什么 wq/wk/wv/w1/w3 列切，wo/w2 行切？

**根本原因**：让中间激活的 Shard 方向**自然衔接**，避免额外通信。

```
Column (w1/w3)          Row (w2)
输入: Replicate     →   输入: Shard(-1)   ← 正好接上!
输出: Shard(-1)     →   输出: Partial → Replicate
无通信                   1次 all-reduce/reduce-scatter
```

如果反过来（w1 行切、w2 列切），中间需要额外通信，效率更差。

对于 Attention：QKV 列切后每个 rank 得到部分 head，attention 完全本地计算，O 投影行切产生部分和 → 一次 all-reduce 恢复完整输出。

---

## 十二、推荐学习资源

1. **PyTorch TP Tutorial (Llama 2)** — 官方完整 TP plan + SP
   https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

2. **Megatron-LM Paper** (Shoeybi et al. 2019) — TP 的原始论文，Fig.2 是经典
   https://arxiv.org/abs/1909.08053

3. **Tensor Parallelism by Hand** (DEV Community) — 手动推导 forward+backward
   https://dev.to/lewis_won/tensor-parallelism-by-hand-3eh

4. **TP/SP Detailed Analysis** (insujang) — Attention 下 TP 和 SP 的区别
   https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/

5. **Reducing Activation Recomputation** (Korthikanti et al.) — SP 的来源论文
   https://arxiv.org/abs/2205.05198
