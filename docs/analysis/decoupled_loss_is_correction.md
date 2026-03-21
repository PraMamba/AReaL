# Decoupled Loss 重要性采样解耦：从旧权重数据到正确梯度的数学挽救

> 基于源码的端到端分析，完整追踪一个在旧权重、旧 KV-Cache 下生成的陈旧 token
> 如何通过三策略分离、版本感知插值和多层方差控制，最终产生数学上正确的策略梯度。

---

## 目录

1. [问题定义：为什么旧数据需要"挽救"](#1-问题定义为什么旧数据需要挽救)
2. [核心思想：三策略分离与 IS 因式分解](#2-核心思想三策略分离与-is-因式分解)
3. [端到端数据流：从生成到梯度](#3-端到端数据流从生成到梯度)
4. [逐步数学推导](#4-逐步数学推导)
5. [单个陈旧 Token 的完整追踪](#5-单个陈旧-token-的完整追踪)
6. [多层方差控制机制](#6-多层方差控制机制)
7. [Proximal 策略近似：零开销的数学技巧](#7-proximal-策略近似零开销的数学技巧)
8. [代码正确性验证](#8-代码正确性验证)
9. [设计总结](#9-设计总结)

---

## 1. 问题定义：为什么旧数据需要"挽救"

在 AReaL 的异步架构中，推理引擎和训练引擎并行运行。当训练步 $k$ 处理 rollout 数据时，
这些数据可能是在训练步 $k-3$ 的权重下生成的。

```
时间线:

  推理引擎:  生成 token_1..100 (权重 v=5)  →  被中止  →  续生成 token_101..200 (权重 v=6)
                                                 ↑ 权重更新
  训练引擎:  训练步 v=5 → 训练步 v=6 → ... → 训练步 v=8 消费这条轨迹
                                                              ↑
                                                    此时 token_1..100 落后 3 步
                                                    token_101..200 落后 2 步
```

**标准 PPO 假设**：rollout 数据由 $\pi_{\text{old}}$ 生成，$\pi_{\text{old}} = \pi_{\theta_{k-1}}$（上一步的策略）。

**异步现实**：rollout 数据由 $\pi_{\text{behave}}$ 生成，$\pi_{\text{behave}}$ 可能是 $\pi_{\theta_{k-3}}$，
甚至同一条轨迹内不同 token 来自不同版本。

**直接用标准 PPO 的后果**：

$$r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{behave}}(a_t|s_t)}$$

由于 $\pi_\theta$ 已经经过多步更新远离 $\pi_{\text{behave}}$，
$r_t$ 在训练开始时就远离 1.0——PPO 的 clip 机制失效（所有 token 都被 clip，梯度为零）。

---

## 2. 核心思想：三策略分离与 IS 因式分解

### 2.1 三个策略

**源码**: `areal/utils/functional/functional.py:227-233`

```python
"""
When decoupled loss is enabled, proximal_logprobs is the recomputed logp,
old_logprobs is produced by the inference engine.
"""
```

| 策略 | 符号 | 版本 | 代码变量 | 来源 |
|------|------|------|---------|------|
| 行为策略 $\pi_{\text{behave}}$ | `old_logprobs` | $v_{\text{behave}}$ (每 token 不同) | `input_data["logprobs"]` | 推理引擎在生成时缓存 |
| 近端策略 $\pi_{\text{prox}}$ | `proximal_logprobs` | $v_\theta - 1$ | `prox_logp` (重计算或近似) | 当前策略的前一步 |
| 当前策略 $\pi_\theta$ | `logprobs` | $v_\theta$ | 训练前向传播（有梯度） | 正在优化的策略 |

### 2.2 IS 因式分解

$$\frac{\pi_\theta(a|s)}{\pi_{\text{behave}}(a|s)} = \underbrace{\frac{\pi_\theta(a|s)}{\pi_{\text{prox}}(a|s)}}_{\text{内层: PPO ratio (可 clip)}} \times \underbrace{\frac{\pi_{\text{prox}}(a|s)}{\pi_{\text{behave}}(a|s)}}_{\text{外层: 行为 IS 权重 (乘法修正)}}$$

**为什么要分解**：
- **内层** $r_t = \pi_\theta / \pi_{\text{prox}}$ 始终接近 1.0（只差一步），PPO clip **有效**
- **外层** $w_t = \pi_{\text{prox}} / \pi_{\text{behave}}$ 修正数据来源偏差，**不参与 clip**

---

## 3. 端到端数据流：从生成到梯度

### 3.1 完整流水线

```
Stage 1: Token 生成 (推理引擎)
  │ remote_inf_engine.py:822-824
  │ accumulated_versions.extend([self.get_version()] * len(tokens))
  │ → 每 token 打版本戳
  ▼
Stage 2: Workflow 打包 (rlvr.py:160-172)
  │ logprobs = [0.0] * input_len + output_logprobs    ← log π_behave
  │ versions = [-1]  * input_len + output_versions     ← v_behave per token
  ▼
Stage 3: 近端 logp 计算 (rl_trainer.py:363-372)
  │ if should_compute_prox_logp():
  │     rollout_batch["prox_logp"] = actor.compute_logp(batch)  ← log π_prox
  │ (或跳过，使用 loglinear 近似)
  ▼
Stage 4: 优势计算 (actor.py:129-228)
  │ loss_mask = roll(loss_mask, -1)     ← 对齐到训练约定
  │ old_logp = roll(logprobs, -1)       ← π_behave (对齐后)
  │ GAE(rewards, values) → advantages
  ▼
Stage 5: PPO 更新 (actor.py:358-546)
  │ 训练前向传播 → logprobs (log π_theta, 有梯度)
  │
  │ _resolve_proximal_logp() → prox_logp
  │   ├─ recompute: 使用 Stage 3 的结果
  │   └─ loglinear: α = (v_prox - v_behave)/(v_theta - v_behave)
  │                 log π_prox ≈ (1-α)·log π_behave + α·log π_theta
  │
  │ _apply_m2po_masking() → 过滤高方差 token (可选)
  │
  │ ppo_actor_loss_fn(logprobs, prox_logp, old_logp, advantages, ...)
  │   ├─ ratio = exp(logprobs - prox_logp)       ← π_theta / π_prox
  │   ├─ clipped_loss = max(-A·r, -A·clip(r))     ← PPO 截断
  │   ├─ w = exp(prox_logp - old_logp)            ← π_prox / π_behave
  │   ├─ w_capped = cap(w, mode, threshold)        ← 方差控制
  │   └─ loss = clipped_loss × w_capped            ← 最终损失
  ▼
Stage 6: 反向传播
  │ loss.backward() → 梯度流过 logprobs (π_theta) → 更新模型参数
```

### 3.2 `torch.roll()` 对齐逻辑

**源码**: `actor.py:160,172`

推理引擎的约定：`logprobs[t]` = 生成位置 $t$ 的 token 的 logprob。
训练的约定：`logprobs[t]` = 给定位置 $0..t$ 的 token，生成位置 $t+1$ 的 token 的 logprob。

```python
loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)   # 左移 1
old_logp  = torch.roll(data["logprobs"], shifts=-1, dims=-1)  # 左移 1
```

移位后 `old_logp[t]` = 原始 `logprobs[t+1]` = $\log \pi_{\text{behave}}(a_{t+1}|s_{0..t})$，
与训练前向传播的 `logprobs[t]` 对齐。

**`prox_logp`**（来自 `compute_logp` 前向传播或近似）已经是训练约定，无需移位。

---

## 4. 逐步数学推导

### 4.1 标准 PPO 目标

$$L^{\text{PPO}}(\theta) = \mathbb{E}_{a \sim \pi_{\text{old}}}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_t\right)\right]$$

其中 $r_t = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$。

### 4.2 Decoupled PPO 目标

**源码**: `functional.py:260,267-276,296`

$$L^{\text{decoupled}}(\theta) = \mathbb{E}_{a \sim \pi_{\text{behave}}}\left[w_t^{\text{cap}} \cdot \min\left(r_t A_t, \text{clip}(r_t) A_t\right)\right]$$

其中：

$$r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{prox}}(a_t|s_t)} \quad \text{(内层, line 260)}$$

$$w_t = \frac{\pi_{\text{prox}}(a_t|s_t)}{\pi_{\text{behave}}(a_t|s_t)} \quad \text{(外层, line 192)}$$

$$w_t^{\text{cap}} = \begin{cases} \min(w_t, C) & \text{truncate 模式} \\ w_t \cdot \mathbb{1}[w_t \leq C] & \text{mask 模式 (默认, C=5.0)} \end{cases}$$

### 4.3 梯度正确性证明

忽略 clip 和 cap（分析纯 IS 部分），对 $\theta$ 求梯度：

$$\nabla_\theta L = \mathbb{E}_{a \sim \pi_{\text{behave}}}\left[\frac{\pi_{\text{prox}}}{\pi_{\text{behave}}} \cdot (-A_t) \cdot \nabla_\theta \frac{\pi_\theta}{\pi_{\text{prox}}}\right]$$

由于 $\nabla_\theta (\pi_\theta / \pi_{\text{prox}}) = (\pi_\theta / \pi_{\text{prox}}) \cdot \nabla_\theta \log \pi_\theta$：

$$= \mathbb{E}_{a \sim \pi_{\text{behave}}}\left[\frac{\pi_{\text{prox}}}{\pi_{\text{behave}}} \cdot \frac{\pi_\theta}{\pi_{\text{prox}}} \cdot (-A_t) \cdot \nabla_\theta \log \pi_\theta\right]$$

$$= \mathbb{E}_{a \sim \pi_{\text{behave}}}\left[\frac{\pi_\theta}{\pi_{\text{behave}}} \cdot (-A_t) \cdot \nabla_\theta \log \pi_\theta\right]$$

$$= \mathbb{E}_{a \sim \pi_\theta}\left[(-A_t) \cdot \nabla_\theta \log \pi_\theta\right]$$

**这正是 $\pi_\theta$ 下的正确策略梯度**。IS 因式分解在数学上是精确的——
$\pi_{\text{prox}}$ 在分子分母中约掉了。

**分解的意义不在于数学等价性，而在于实践中的可控性**：
- 内层 $r_t = \pi_\theta / \pi_{\text{prox}}$ 接近 1.0 → PPO clip 有效 → 限制单步策略变化
- 外层 $w_t = \pi_{\text{prox}} / \pi_{\text{behave}}$ 可单独 cap → 控制陈旧度引入的方差

---

## 5. 单个陈旧 Token 的完整追踪

### 设定

- Token 位置 $t$，动作 $a_t$
- 生成版本 $v_{\text{behave}} = 5$，训练版本 $v_\theta = 8$
- $v_{\text{prox}} = v_\theta - 1 = 7$
- 推理引擎缓存的 logprob: $\log \pi_5(a_t|s_t) = -2.3$
- 当前策略 logprob: $\log \pi_8(a_t|s_t) = -1.8$

### Stage 1: 版本戳打标

**源码**: `remote_inf_engine.py:822-824`

```python
accumulated_versions.extend([self.get_version()] * len(gen_result.output_tokens))
# → versions[t] = 5
```

### Stage 2: Workflow 打包

**源码**: `rlvr.py:160-162`

```python
logprobs[t] = -2.3          # log π_behave(a_t|s_t)
versions[t] = 5             # v_behave = 5
loss_mask[t] = 1            # 生成的 token
```

### Stage 3: 近端 logp 解析

**方法 A: 重计算** — `actor.compute_logp()` 前向传播得到 $\log \pi_7(a_t|s_t) = -2.0$

**方法 B: Loglinear 近似** (`actor.py:598-617`):

```python
alpha = (v_prox - v_behave) / (v_theta - v_behave)
      = (7 - 5) / (8 - 5) = 2/3

prox_logp = old_logp + alpha * (logprobs - old_logp)
          = -2.3 + (2/3) * (-1.8 - (-2.3))
          = -2.3 + (2/3) * 0.5
          = -2.3 + 0.333
          = -1.967
```

近似值 $-1.967$ vs 真实值 $-2.0$，误差 0.033（~1.7%）。

### Stage 4: PPO Ratio 计算

**源码**: `functional.py:260`

```python
ratio = exp(logprobs - proximal_logprobs)
      = exp(-1.8 - (-2.0))   # 使用重计算值
      = exp(0.2)
      = 1.22
```

$r_t = \pi_8 / \pi_7 = 1.22$ — 接近 1.0，PPO clip (ε=0.2) 有效：
$\text{clip}(1.22, 0.8, 1.2) = 1.2$

### Stage 5: 行为 IS 权重

**源码**: `functional.py:177,192`

```python
behave_imp_weight = exp(proximal_logprobs - old_logprobs)
                  = exp(-2.0 - (-2.3))
                  = exp(0.3)
                  = 1.35
```

$w_t = \pi_7 / \pi_5 = 1.35$ — 版本差 2 步的修正因子。$1.35 < 5.0$ (cap)，不被截断。

### Stage 6: 最终损失

**源码**: `functional.py:273-276,296`

假设 $A_t = 0.5$（正优势）：

```python
pg_loss1 = -A_t * ratio       = -0.5 * 1.22 = -0.61
pg_loss2 = -A_t * clipped     = -0.5 * 1.2  = -0.60
pg_loss  = max(-0.61, -0.60)  = -0.60        # clip 生效

final_loss = pg_loss * behave_imp_weight
           = -0.60 * 1.35
           = -0.81
```

### Stage 7: 梯度

```
∂L_t/∂θ = w_t · (-A_t) · ∂[clip(r_t)]/∂θ
        = 1.35 · (-0.5) · ∂[1.2]/∂θ
        = 0  (clip 后 ratio 是常数，梯度为零)

当 ratio 未被 clip 时 (r_t ∈ [0.8, 1.2]):
∂L_t/∂θ = w_t · (-A_t) · r_t · ∇θ log πθ
        = 1.35 · (-0.5) · 1.22 · ∇θ log πθ
        ≈ (πθ/π_behave) · (-A_t) · ∇θ log πθ   ← 正确的 IS 策略梯度
```

### 对比：不使用 Decoupled Loss

```
不分解的朴素 ratio = πθ/π_behave = exp(-1.8-(-2.3)) = exp(0.5) = 1.65
clip(1.65, 0.8, 1.2) = 1.2  ← 被 clip！

但 1.65 已经在训练开始时就超出 [0.8, 1.2]
→ 每个梯度步都被 clip → 无法学习
→ 这就是"标准 PPO 在异步下失效"的原因
```

---

## 6. 多层方差控制机制

### 6.1 层级结构

```
                    方差控制层级
                        │
    ┌───────────────────┼───────────────────┐
    ▼                   ▼                   ▼
Layer 1:          Layer 2:              Layer 3:
Depth Bounding    IS Cap                M2PO
(基础设施层)       (算法层)              (算法层, 可选)

max_head_         behave_imp_           m2_threshold
offpolicyness     weight_cap=5.0

限制最大版本差    截断/屏蔽极端          基于二阶矩
→ 从源头控制     IS 权重                过滤高方差 token
  staleness      → 单 token 方差        → 全局方差
  上界             ≤ 25x                  ≤ threshold
```

### 6.2 IS Cap 的方差分析

**源码**: `functional.py:194-206`

设 $w = \pi_{\text{prox}} / \pi_{\text{behave}}$，真实梯度估计的方差：

$$\text{Var}(w \cdot g) = \mathbb{E}[w^2] \cdot \text{Var}(g)$$

**Truncate 模式** ($w^{\text{cap}} = \min(w, C)$):

$$\text{Var}(w^{\text{cap}} \cdot g) \leq C^2 \cdot \text{Var}(g) = 25 \cdot \text{Var}(g)$$

方差被**硬性限制在 25 倍**内。

**Mask 模式** ($w^{\text{cap}} = w \cdot \mathbb{1}[w \leq C]$, 默认):

$$\text{Var}(w^{\text{cap}} \cdot g) \leq C^2 \cdot P(w \leq C) \cdot \text{Var}(g) < 25 \cdot \text{Var}(g)$$

被屏蔽的 token 不贡献任何方差——比 truncate 方差更低，但偏差更大。

### 6.3 M2PO 的二阶矩控制

**源码**: `actor.py:719-800`

M2PO 直接控制 $\mathbb{E}[(\log r)^2]$，而非单 token 的 $r$ 上界：

```python
delta = old_logp - prox_logp                    # log(π_behave / π_prox)
m2 = delta * delta                               # 二阶矩
sorted_m2 = sort(m2, descending=True)
# 找到最小 k 使得 mean(sorted_m2[k:]) < m2_threshold
# 屏蔽 top-k 个高 m2 token
```

**数学依据**: 通过 Jensen 不等式：

$$\mathbb{E}[w^2] = \mathbb{E}[e^{2\log w}] \leq e^{2\mathbb{E}[(\log w)^2]}$$

控制 $\mathbb{E}[(\log w)^2]$ 即控制 $\mathbb{E}[w^2]$，进而控制梯度方差。

**M2PO vs IS Cap 互补**:

| 机制 | 控制目标 | 适用 staleness |
|------|---------|---------------|
| IS Cap (C=5.0) | 单 token $w$ 上界 | 中等 (2-8 步) |
| M2PO | 全 batch 平均 $(\log w)^2$ | 极端 (16-256 步) |

---

## 7. Proximal 策略近似：零开销的数学技巧

### 7.1 问题：额外前向传播的开销

重计算 $\pi_{\text{prox}}$ 需要一次完整的 `model.forward()`（eval 模式，无梯度）。
对于 70B 模型，这约占训练步时间的 30-40%。

### 7.2 解决：版本感知的对数线性插值

**源码**: `actor.py:579-617`

```python
v_proximal = current_version - 1
alpha = (v_proximal - v_behave) / (v_theta - v_behave)
alpha = clamp(alpha, 0.0, 1.0)

# Log-linear (对数空间线性 = 概率空间几何平均)
log_pi_prox ≈ (1 - alpha) · log_pi_behave + alpha · log_pi_theta
```

等价于概率空间的几何插值：

$$\pi_{\text{prox}} \approx \pi_{\text{behave}}^{1-\alpha} \cdot \pi_\theta^{\alpha}$$

### 7.3 假设与误差

**假设**: $\log \pi$ 在版本空间中近似线性演化。

**合理性**: PPO clip 限制每步策略变化 $\leq \varepsilon = 0.2$，
所以 $\log \pi$ 的每步变化有界。对于多步间隔，线性外推是一阶 Taylor 近似。

**实测误差** (文档 `docs/en/algorithms/prox_approx.md`):
> 在 8 步 staleness 下，loglinear 与精确重计算的性能差距 ≤ 2%。
> 训练时间节省 27%（163 min vs 207 min / 300 步）。

### 7.4 Per-Token Alpha 的物理含义

```
v_behave=3, v_prox=7, v_theta=8:
  alpha = (7-3)/(8-3) = 0.8
  → 80% 来自 π_theta, 20% 来自 π_behave
  → π_prox 与 π_theta 接近（只差 1 步）

v_behave=7, v_prox=7, v_theta=8:
  alpha = (7-7)/(8-7) = 0.0
  → 100% 来自 π_behave
  → π_behave 就是 π_prox（新鲜数据，无需修正）

v_behave=0, v_prox=99, v_theta=100:
  alpha = 99/100 = 0.99
  → 99% 来自 π_theta
  → 极度陈旧的 token，近似几乎完全依赖当前策略
```

---

## 8. 代码正确性验证

### 8.1 IS 比率计算 — 正确

**`functional.py:260`**: `ratio = exp(logprobs - proximal_logprobs)` = $\pi_\theta / \pi_{\text{prox}}$ ✓

**`functional.py:177,192`**: `behave_imp_weight = exp(proximal_logprobs - old_logprobs)` = $\pi_{\text{prox}} / \pi_{\text{behave}}$ ✓

### 8.2 梯度流 — 正确

`behave_imp_weight` 由 `proximal_logprobs`（detached）和 `old_logprobs`（detached）计算，
**不携带梯度图**。乘以 `pg_loss`（通过 `logprobs` 携带梯度）时，
`behave_imp_weight` 作为常数乘子——梯度只流过 `logprobs`。✓

### 8.3 `torch.roll` 对齐 — 正确

- `old_logp[t-1]` = 原始 `logprobs[t]` = $\log \pi_{\text{behave}}(a_t|s_{0..t-1})$（移位后训练约定）
- `prox_logp[t-1]` = $\log \pi_{\text{prox}}(a_t|s_{0..t-1})$（forward 输出已是训练约定）
- `logprobs[t-1]` = $\log \pi_\theta(a_t|s_{0..t-1})$（训练 forward 输出）

三者在位置 $t-1$ 都表示"生成位置 $t$ 的 token"的对数概率。✓

### 8.4 M2PO 排序-前缀和算法 — 正确

降序排列后，suffix average `avg_m2_suffix[k]` 是移除 top-$k$ 后的平均值。
由于降序排列，移除 top-$k$ 单调降低剩余平均值。
第一个 $k$ 使得 `avg_m2_suffix[k] < threshold` 就是最小需要移除的数量。✓

### 8.5 Loss 归一化 — 设计选择

**`functional.py:299`**: `pg_loss = sum(pg_loss × loss_mask) / loss_mask_count`

`loss_mask_count` 是**原始** valid token 数（含被 cap mask 掉的 token）。
被 IS cap mask 的 token 贡献 $0$，但分母不减少。
效果：IS masking 比例越高，有效学习率越低。这是**隐式的学习率退火**——
陈旧度越严重，被 mask 的 token 越多，实际步长越小。

---

## 9. 设计总结

### 如何"挽救"旧权重下生成的落后数据

```
                    旧权重数据的"挽救"管线

    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  ① Per-Token 版本戳 (remote_inf_engine.py:822)          │
    │     → 每个 token 知道自己的生成版本 v_behave             │
    │                                                          │
    │  ② Proximal 近似 (actor.py:598-617)                     │
    │     → 利用版本信息对数线性插值 π_prox                     │
    │     → 无需额外前向传播                                    │
    │                                                          │
    │  ③ IS 因式分解 (functional.py:260,296)                   │
    │     → 内层 r = π_theta/π_prox (近 1.0, clip 有效)        │
    │     → 外层 w = π_prox/π_behave (乘法修正)                │
    │     → 数学上精确: r × w = π_theta/π_behave               │
    │                                                          │
    │  ④ IS Cap (functional.py:194-206)                        │
    │     → w > 5.0 的 token 被截断/屏蔽                       │
    │     → 方差硬性限制在 25× 以内                             │
    │                                                          │
    │  ⑤ M2PO 二阶矩过滤 (actor.py:719-800)                  │
    │     → 控制全 batch 的 (log w)^2 平均值                   │
    │     → 极端陈旧场景 (staleness>16) 的补充保障              │
    │                                                          │
    │  结果: 数学上正确的策略梯度 + 可控的方差                   │
    │        E[∇L] = E_{π_theta}[-A · ∇ log π_theta]          │
    │        Var[∇L] ≤ 25 · Var[标准 PPO 梯度]                │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```

### 一句话总结

> AReaL 通过将 IS 比 $\pi_\theta/\pi_{\text{behave}}$ 分解为
> **可 clip 的近端比** $\pi_\theta/\pi_{\text{prox}}$ 和
> **可 cap 的行为修正** $\pi_{\text{prox}}/\pi_{\text{behave}}$，
> 使得旧权重生成的每个 token 都能获得数学上正确的策略梯度修正，
> 同时通过 per-token 版本感知的对数线性插值省去了额外前向传播。
> 三层方差控制（Depth Bounding + IS Cap + M2PO）确保即使在 256 步 staleness 下
> 梯度方差仍然可控。
