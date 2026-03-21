# 单卡 Transformer 训练到底在算什么

> **目标**：读完这份文档后，再看 FSDP / Activation Checkpointing / Context Parallelism 时，能立刻指出"它们在救哪一类内存"。

---

## 一、全景图：一次训练迭代做了什么

```
input_ids                            (int64, shape: [B, S])
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  embed_tokens   (Embedding lookup)                       │
│  input_ids → hidden_states  [B, S, D]                    │
└──────────────────────────────────────────────────────────┘
    │
    ▼  rotary_emb(hidden_states, position_ids) → (cos, sin)
    │
    ▼  ════════ × L layers (Qwen2DecoderLayer) ════════
    │
    │  ┌──────────────────────────────────────────────┐
    │  │ residual = hidden_states                     │
    │  │ hidden_states = input_layernorm(hidden_states)│  ← RMSNorm
    │  │ hidden_states = self_attn(hidden_states, ...) │  ← Attention
    │  │ hidden_states = residual + hidden_states      │  ← Add
    │  │                                               │
    │  │ residual = hidden_states                     │
    │  │ hidden_states = post_attn_layernorm(hidden_states)│← RMSNorm
    │  │ hidden_states = mlp(hidden_states)            │  ← SwiGLU MLP
    │  │ hidden_states = residual + hidden_states      │  ← Add
    │  └──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  norm  (final RMSNorm)                                   │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  lm_head  (Linear: D → Vocab)  → logits [B, S, V]       │
└──────────────────────────────────────────────────────────┘
    │
    ▼
  cross_entropy_loss(logits, labels) → scalar loss
    │
    ▼
  loss.backward()     ← 反向传播，计算所有参数的梯度
    │
    ▼
  optimizer.step()    ← AdamW 更新参数
  optimizer.zero_grad()
```

---

## 二、Qwen2 模型结构详解（对应 HuggingFace 源码）

### 2.1 顶层结构

```python
class Qwen2ForCausalLM:
    self.model = Qwen2Model(config)       # 核心 Transformer
    self.lm_head = nn.Linear(D, V, bias=False)  # 输出投影，权重与 embed_tokens 绑定(tied)

class Qwen2Model:
    self.embed_tokens = nn.Embedding(V, D)           # Token embedding
    self.layers = nn.ModuleList([                     # L 个 DecoderLayer
        Qwen2DecoderLayer(config, i) for i in range(L)
    ])
    self.norm = Qwen2RMSNorm(D)                      # 最终 LayerNorm
    self.rotary_emb = Qwen2RotaryEmbedding(config)   # RoPE（无可训练参数）
```

> **对照你提到的 torchtitan 命名**：`tok_embeddings` = `embed_tokens`，`ModuleDict layers` = `self.layers`，`norm` = `self.norm`，`output` = `lm_head`。

### 2.2 单个 DecoderLayer

```python
class Qwen2DecoderLayer:
    self.input_layernorm    = Qwen2RMSNorm(D)     # Attention 前的 Norm
    self.self_attn          = Qwen2Attention(...)  # 注意力
    self.post_attention_layernorm = Qwen2RMSNorm(D) # MLP 前的 Norm
    self.mlp                = Qwen2MLP(...)        # FFN
```

**forward 伪代码（Pre-Norm + 残差）**：

```python
def forward(self, hidden_states, ...):
    # ─── Attention Block ───
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(hidden_states, ...)
    hidden_states = residual + hidden_states       # 残差连接

    # ─── MLP Block ───
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states       # 残差连接

    return hidden_states
```

### 2.3 Attention 详解

```python
class Qwen2Attention:
    # 投影矩阵（GQA：KV heads 可以少于 Q heads）
    self.q_proj = nn.Linear(D, num_heads * head_dim, bias=True)     # 注意：有 bias
    self.k_proj = nn.Linear(D, num_kv_heads * head_dim, bias=True)
    self.v_proj = nn.Linear(D, num_kv_heads * head_dim, bias=True)
    self.o_proj = nn.Linear(num_heads * head_dim, D, bias=False)    # 无 bias
```

**forward 流程**：

```
hidden_states [B, S, D]
    │
    ├─ q_proj ──→ Q [B, S, num_heads, head_dim] ──→ transpose ──→ [B, num_heads, S, head_dim]
    ├─ k_proj ──→ K [B, S, num_kv_heads, head_dim] ──→ transpose
    └─ v_proj ──→ V [B, S, num_kv_heads, head_dim] ──→ transpose
    │
    ▼  apply_rotary_pos_emb(Q, K, cos, sin)   ← RoPE
    │
    ▼  （如果有 KV cache，更新 past_key_values）
    │
    ▼  repeat_kv(K, V)   ← 将 KV heads 复制到与 Q heads 数量匹配（GQA）
    │
    ▼  attn_weights = Q @ K^T * scaling        [B, num_heads, S, S]
    ▼  attn_weights += causal_mask
    ▼  attn_weights = softmax(attn_weights)
    ▼  attn_weights = dropout(attn_weights)     （训练时）
    ▼  attn_output  = attn_weights @ V          [B, num_heads, S, head_dim]
    │
    ▼  reshape → [B, S, D]
    ▼  o_proj → [B, S, D]
```

**Qwen2 与标准 Llama 的关键差异**：
- QKV 投影**有 bias**（Llama 没有）
- 使用 **GQA**（Grouped Query Attention）：`num_kv_heads < num_heads`，节省 KV cache
- 部分层使用 **Sliding Window Attention**（减少注意力的计算和内存）

### 2.4 MLP 详解（SwiGLU）

```python
class Qwen2MLP:
    self.gate_proj = nn.Linear(D, intermediate_size, bias=False)
    self.up_proj   = nn.Linear(D, intermediate_size, bias=False)
    self.down_proj = nn.Linear(intermediate_size, D, bias=False)
    self.act_fn    = SiLU()  # hidden_act = "silu"

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

```
x [B, S, D]
    │
    ├── gate_proj ──→ [B, S, I] ──→ SiLU() ──→ [B, S, I] ──┐
    │                                                         │  element-wise multiply
    └── up_proj  ──→ [B, S, I] ──────────────────────────────┘
                                    │
                                    ▼
                              [B, S, I]
                                    │
                                    ▼  down_proj
                              [B, S, D]
```

> SwiGLU 用了两个"上投影"（gate + up），所以 MLP 有 3 个权重矩阵，参数量 = `3 × D × I`。

### 2.5 RMSNorm

```python
class Qwen2RMSNorm:
    self.weight = nn.Parameter(torch.ones(D))  # 可训练缩放参数，只有 D 个

    def forward(self, hidden_states):
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return self.weight * hidden_states
```

参数量极少（每个 Norm 只有 D 个参数），但**它产生的激活在 backward 时需要保存**。

---

## 三、参数量计算（以 Qwen2-7B 为例）

| 组件 | 公式 | Qwen2-7B 参数 |
|------|------|---------------|
| embed_tokens | V × D | 151,936 × 3,584 ≈ 545M |
| 每层 q_proj | D × (H × d) + D（bias） | 3,584 × 3,584 + 3,584 ≈ 12.8M |
| 每层 k_proj | D × (KV_H × d) + KV_H×d | 3,584 × 512 + 512 ≈ 1.84M |
| 每层 v_proj | 同 k_proj | ≈ 1.84M |
| 每层 o_proj | (H × d) × D | 3,584 × 3,584 ≈ 12.8M |
| 每层 gate_proj | D × I | 3,584 × 18,944 ≈ 67.9M |
| 每层 up_proj | D × I | ≈ 67.9M |
| 每层 down_proj | I × D | ≈ 67.9M |
| 每层 2×RMSNorm | 2 × D | 7,168 |
| final norm | D | 3,584 |
| lm_head | D × V（与 embed_tokens 绑定）| 0（tied）|
| **合计** | embed + 28层 × 每层 | **≈ 7.6B** |

**符号说明**：B=batch, S=seq_len, D=hidden_size=3584, H=num_heads=28, KV_H=num_kv_heads=4, d=head_dim=128, I=intermediate_size=18944, L=28, V=vocab_size=151936

---

## 四、Forward 时哪些 Tensor 会留到 Backward

### 4.1 核心原理

PyTorch autograd 的规则：**为了计算某个算子的梯度，需要保存该算子的部分/全部输入（有时也保存输出）**。

例如：
- `y = A @ x` → 需要保存 `A` 和 `x` 来算 `dL/dA` 和 `dL/dx`
- `y = softmax(x)` → 保存输出 `y`，因为 `dsoftmax/dx` 可用 `y` 表达
- `y = SiLU(x)` → 保存输入 `x`
- `y = dropout(x, mask)` → 保存 dropout mask
- `y = x1 + x2`（残差）→ 不需要保存任何东西（加法梯度=1）

### 4.2 逐算子分析（单个 Decoder Layer）

下面追踪一个 layer 的 forward，标记哪些 tensor 需要被 autograd 保存：

```
步骤                              需要保存的 Tensor           大小（bytes, bf16）
────────────────────────────────────────────────────────────────────────────────
1. input_layernorm(x)
   - RMSNorm                      输入 x, rsqrt结果           2BSD + BF16开销
                                   (归一化后用于backward)

2. q_proj(normed_x)               normed_x                   2BSD (Linear输入)
   k_proj(normed_x)               (共享同一输入，只存一次)
   v_proj(normed_x)

3. apply_rotary_pos_emb(Q,K)      Q和K的原始值(或cos/sin)      开销较小

4. Q @ K^T  (GEMM)                Q 和 K                     2BS·H·d + 2BS·KV_H·d
                                                              (GQA下K较小)

5. softmax(scores)                softmax 输出 P              2B·H·S·S
                                                              ← 最大的一块！

6. dropout(P)                     dropout mask                B·H·S·S (1 byte/元素)

7. P @ V  (GEMM)                  P 和 V                     已在5/6保存P;
                                                              V: 2BS·KV_H·d

8. o_proj(attn_out)               attn_out                   2BSD

9. residual + attn_out            无需保存                    0

10. post_attn_layernorm           残差后的hidden_states        2BSD

11. gate_proj(normed_x2)          normed_x2                  2BSD (共享)
    up_proj(normed_x2)

12. SiLU(gate_out)                gate_out                   2BSI

13. silu_out * up_out             silu_out 和 up_out          2BSI × 2

14. down_proj(mlp_mid)            mlp_mid                    2BSI

15. residual + mlp_out            无需保存                    0
```

### 4.3 每层激活内存总结公式

引用 Nvidia 的经典公式（Reducing Activation Recomputation in Large Transformer Models, 2022）：

**每个 Transformer 层的激活内存** ≈

```
activation_per_layer = S × B × D × (10 + 24/t + 5×a×S/D/t)    （单位：字节，bf16）
```

简化版（单卡 t=1，不考虑 tensor parallel）：

| 子模块 | 需保存的激活 | 大小 |
|--------|-------------|------|
| LayerNorm ×2 | 输入 ×2 | 4BSD |
| QKV proj | 输入（共享）| 2BSD |
| Q, K after RoPE | Q, K | 2BS(H+KV_H)d |
| Attention score softmax | softmax 输出 | 2B·H·S² |
| Dropout mask | mask | B·H·S² |
| V (for P@V backward) | V | 2BS·KV_H·d |
| o_proj | 输入 | 2BSD |
| MLP gate/up proj | 输入（共享）| 2BSD |
| SiLU gate output | gate_out | 2BSI |
| Gate*up product | 两个操作数 | 4BSI |
| down_proj | 输入 | 2BSI |

> **关键洞察**：softmax 输出的 `2B·H·S²` 随序列长度**二次增长**，是长序列训练的核心瓶颈。

---

## 五、GPU 内存的四大居民

训练时 GPU 内存被四类数据占据：

### 5.1 模型参数（Parameters）

```
纯 bf16 训练：  2 bytes × N_param
混合精度训练：  2 bytes × N_param (bf16 副本，用于 forward/backward)
              + 4 bytes × N_param (fp32 master copy，用于 optimizer update)
```

**Qwen2-7B 举例**：7.6B × 2 bytes = **15.2 GB**（bf16）

### 5.2 优化器状态（Optimizer State）

AdamW 需要为每个参数维护：
- **m**（first moment / 动量）：fp32，4 bytes/param
- **v**（second moment / 方差）：fp32，4 bytes/param

```
AdamW 状态 = 8 bytes × N_param
```

加上 fp32 master weights：

```
总 optimizer 相关 = 4 (master) + 4 (m) + 4 (v) = 12 bytes/param
```

**Qwen2-7B 举例**：7.6B × 12 = **91.2 GB**（⚠️ 单张 A100 80GB 放不下！）

> 这就是为什么需要 **ZeRO / FSDP** — 它们把 optimizer state 分片到多张卡上。

### 5.3 梯度（Gradients）

```
bf16 梯度：2 bytes × N_param
fp32 梯度：4 bytes × N_param
```

梯度只在 backward 到 optimizer.step() 之间存在。如果用 gradient accumulation，则需要在多个 micro-batch 间持久存储。

**Qwen2-7B 举例**：7.6B × 2 = **15.2 GB**（bf16 梯度）

### 5.4 激活（Activations）

```
激活内存 ∝ B × S × D × L    （线性部分）
         + B × H × S² × L   （注意力分数，二次部分）
```

激活是**唯一随 batch size 和 seq_len 显著变化的部分**。

### 5.5 总内存公式（混合精度 + AdamW）

```
M_total = M_params + M_optimizer + M_gradients + M_activations

        = 2N        （bf16 参数）
        + 12N       （fp32 master + Adam m + Adam v）
        + 2N        （bf16 梯度）
        + M_act     （激活，取决于 B, S, L）

稳态 ≈ 16N + M_activations
峰值 ≈ 18N + M_activations   （backward 开始时，梯度 + 激活同时存在）
```

**Qwen2-7B 数值**：

| 类别 | 公式 | 大小 |
|------|------|------|
| bf16 参数 | 2 × 7.6B | 15.2 GB |
| fp32 master weights | 4 × 7.6B | 30.4 GB |
| Adam m + v | 8 × 7.6B | 60.8 GB |
| bf16 梯度 | 2 × 7.6B | 15.2 GB |
| **参数相关小计** | 16 × 7.6B | **121.6 GB** |
| 激活（B=1, S=4096）| 估算 | ~5-15 GB |

> 不算激活就已经超过单张 A100 的 80GB。这解释了为什么 7B 模型的全量训练**必须**用多卡或 offload。

### 5.6 KV Cache（推理专属 vs 训练中的角色）

**推理时**：KV cache 是核心内存消耗。每层缓存 K 和 V：
```
KV cache per layer = 2 × 2 × B × S × KV_H × d   （K和V各一份，bf16）
总 KV cache = L × 4 × B × S × KV_H × d bytes
```

**训练时**：通常不需要 KV cache（full attention 每次从头算）。但 KV 作为中间激活依然存在于显存中，被 autograd 保存用于 backward。

GQA 的好处在这里体现：`KV_H < H`，所以 K 和 V 占的激活内存更少。

---

## 六、详细数据流图（单层完整 Forward + 需保存标记）

```
┌─────────────────────────────────────────────────────────────────┐
│                   Qwen2 DecoderLayer[i]                         │
│                                                                 │
│  输入: x [B, S, D]                                              │
│                                                                 │
│  ┌─── Attention Sub-Block ─────────────────────────────────┐    │
│  │                                                         │    │
│  │  x_norm = RMSNorm(x)              💾 保存: x (for grad) │    │
│  │                                                         │    │
│  │  Q = q_proj(x_norm) + reshape     💾 保存: x_norm       │    │
│  │  K = k_proj(x_norm) + reshape     (QKV共享输入)          │    │
│  │  V = v_proj(x_norm) + reshape                           │    │
│  │                                                         │    │
│  │  Q, K = RoPE(Q, K, cos, sin)                            │    │
│  │                                                         │    │
│  │  scores = Q @ K^T × scale        💾 保存: Q, K          │    │
│  │  P = softmax(scores + mask)       💾 保存: P (softmax输出)│   │
│  │  P = dropout(P)                   💾 保存: dropout mask  │    │
│  │  ctx = P @ V                      💾 保存: V             │    │
│  │                                                         │    │
│  │  out = o_proj(ctx.reshape)        💾 保存: ctx           │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  h = x + out                          （加法不需保存）           │
│                                                                 │
│  ┌─── MLP Sub-Block ──────────────────────────────────────┐     │
│  │                                                        │     │
│  │  h_norm = RMSNorm(h)             💾 保存: h (for grad)  │     │
│  │                                                        │     │
│  │  gate = gate_proj(h_norm)         💾 保存: h_norm        │     │
│  │  up   = up_proj(h_norm)           (gate/up共享输入)      │     │
│  │  gate_act = SiLU(gate)            💾 保存: gate          │     │
│  │  mid = gate_act * up              💾 保存: gate_act, up  │     │
│  │  out = down_proj(mid)             💾 保存: mid           │     │
│  │                                                        │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
│  output = h + out                     （加法不需保存）           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

💾 = autograd 需要保存到 backward 的 tensor

---

## 七、为什么分布式技术各救不同的内存 — 预览

理解了上述四类内存后，各种技术的目标一目了然：

| 技术 | 主要目标 | 它减的是哪类内存 |
|------|---------|----------------|
| **FSDP (ZeRO-3)** | 把参数、梯度、优化器状态分片到多卡 | **参数 + 梯度 + 优化器状态** (16N → 16N/G，G=GPU数) |
| **ZeRO-1** | 只分片优化器状态 | **优化器状态** (12N → 12N/G) |
| **ZeRO-2** | 分片优化器状态 + 梯度 | **优化器状态 + 梯度** |
| **Activation Checkpointing (AC)** | 不保存中间激活，backward时重算 | **激活** (大幅减少，代价是多算一次forward) |
| **Context Parallelism (CP)** | 把序列维度切分到多卡 | **激活**（尤其是 S² 的注意力分数部分）|
| **Tensor Parallelism (TP)** | 切分单层的权重矩阵到多卡 | **参数 + 激活**（每卡只算部分 head）|
| **Pipeline Parallelism (PP)** | 不同层放不同卡 | **参数 + 优化器状态**（但不减激活）|
| **Mixed Precision** | 用 bf16 算 forward/backward | **参数内存减半，激活减半** |
| **Gradient Accumulation** | 小 micro-batch 多次累积 | **激活**（通过减小有效 B）|
| **CPU Offload** | 把优化器状态移到 CPU | **优化器状态（GPU显存）** |

### 关键直觉

- **模型小、序列长** → 激活是瓶颈 → AC + CP 最有效
- **模型大、序列短** → 参数/优化器是瓶颈 → FSDP / TP 最有效
- **两者都大** → 需要组合（3D parallelism + AC）

---

## 八、推荐学习资源

### 必读

1. **Transformer Memory Arithmetic** (erees.dev) — 用 nanoGPT 逐字节验证内存公式
2. **Reducing Activation Recomputation in Large Transformer Models** (Korthikanti et al., MLSys 2023) — Nvidia 的经典论文，推导了每层激活公式
3. **Transformer Math 101** (EleutherAI Blog) — 参数/显存/FLOPs 的完整计算指南
4. **HuggingFace Efficient Training on a Single GPU** — 实操向的单卡优化指南

### 源码

5. **Qwen2 modeling 源码**：`transformers/models/qwen2/modeling_qwen2.py` — 直接读 Qwen2Model / Qwen2DecoderLayer / Qwen2Attention / Qwen2MLP
6. **PyTorch Autograd Tutorial: Saved Tensors Hooks** — 理解 forward 时到底保存了什么

### 进阶（读完本文后）

7. **PyTorch Activation Checkpointing Blog** — 详解 AC 的 speed-memory tradeoff
8. **Qwen2 Technical Report** (arXiv:2407.10671) — 官方模型设计细节

---

## 九、快速自测

1. **Qwen2 的 MLP 有几个权重矩阵？为什么？**
   → 3个（gate_proj, up_proj, down_proj），因为用了 SwiGLU 门控结构。

2. **softmax 输出的 shape 是什么？为什么它是长序列的瓶颈？**
   → [B, H, S, S]，内存随 S 的平方增长。

3. **AdamW 优化器需要多少额外内存？**
   → 12 bytes/param（fp32 master + momentum + variance）。

4. **Activation Checkpointing 是减少哪类内存？代价是什么？**
   → 减少激活内存，代价是 backward 时要多做一次 forward 重算。

5. **FSDP 是减少哪类内存？**
   → 参数 + 梯度 + 优化器状态，通过分片到多卡。

6. **GQA 相比 MHA 节省了什么？**
   → 减少 KV heads 数量，推理时减少 KV cache，训练时减少 K/V 的激活内存。

7. **为什么 backward 刚开始时显存达到峰值？**
   → 此时所有 forward 保存的激活还在，同时开始产生梯度。
