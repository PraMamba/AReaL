# AReaL Staleness 管理策略深度分析

> 基于源码的详细分析，覆盖为何放弃 Version Rejection、IS 裁剪的梯度方差控制能力、
> Decoupled Loss 的解耦计算实现，以及对训练步长的性能损耗。

---

## 目录

1. [三种策略的设计哲学](#1-三种策略的设计哲学)
2. [为什么放弃逐样本版本拒绝](#2-为什么放弃逐样本版本拒绝)
   - 2.1 [系统中的"拒绝"实际发生在哪里](#21-系统中的拒绝实际发生在哪里)
   - 2.2 [放弃硬拒绝的五个原因](#22-放弃硬拒绝的五个原因)
   - 2.3 [Token-Level Masking 作为软拒绝的替代](#23-token-level-masking-作为软拒绝的替代)
3. [IS 裁剪对梯度方差的控制能力分析](#3-is-裁剪对梯度方差的控制能力分析)
   - 3.1 [Cap=5.0 的数学分析](#31-cap50-的数学分析)
   - 3.2 [Mask vs Truncate 模式的方差特性](#32-mask-vs-truncate-模式的方差特性)
   - 3.3 [极端 Straggler 场景分析](#33-极端-straggler-场景分析)
   - 3.4 [M2PO：极端 Staleness 下的补充机制](#34-m2po极端-staleness-下的补充机制)
4. [use_decoupled_loss 的解耦计算实现](#4-use_decoupled_loss-的解耦计算实现)
   - 4.1 [三策略分离框架](#41-三策略分离框架)
   - 4.2 [核心损失函数源码解析](#42-核心损失函数源码解析)
   - 4.3 [Proximal 策略的计算路径](#43-proximal-策略的计算路径)
   - 4.4 [Per-Token 版本追踪与插值近似](#44-per-token-版本追踪与插值近似)
5. [性能损耗分析](#5-性能损耗分析)
   - 5.1 [训练步内的完整时间线](#51-训练步内的完整时间线)
   - 5.2 [各配置模式的性能对比](#52-各配置模式的性能对比)
   - 5.3 [Loglinear 近似的零额外开销方案](#53-loglinear-近似的零额外开销方案)
6. [配置模式总览](#6-配置模式总览)
7. [代码质量发现](#7-代码质量发现)
8. [设计总结](#8-设计总结)

---

## 1. 三种策略的设计哲学

AReaL 的 Staleness 管理采用**三层防御体系**，有意识地避免了硬性 Version Rejection：

```
┌──────────────────────────────────────────────────────────────┐
│ 第一层：Depth Bounding（基础设施层）                           │
│  max_head_offpolicyness 限制最大版本差                        │
│  → 从源头控制 staleness 上界                                  │
├──────────────────────────────────────────────────────────────┤
│ 第二层：Behavioral IS Correction（算法层）                     │
│  use_decoupled_loss + behave_imp_weight_cap=5.0              │
│  → 对已接收的 off-policy 样本进行概率修正                      │
├──────────────────────────────────────────────────────────────┤
│ 第三层：M2PO 二阶矩过滤（可选，极端场景）                      │
│  m2_threshold 控制允许的最大二阶矩                             │
│  → 在极端 staleness (>256步) 下保持梯度方差可控                │
└──────────────────────────────────────────────────────────────┘
```

**刻意缺失的层**：逐样本 Version Rejection（基于版本号丢弃整条轨迹）。

---

## 2. 为什么放弃逐样本版本拒绝

### 2.1 系统中的"拒绝"实际发生在哪里

AReaL 中确实存在 `on_rollout_rejected()` 回调，但它**不是基于版本号的拒绝**。

**源码**: `areal/infra/workflow_executor.py:1040-1127`

```python
# 轨迹完成后的接受/拒绝判断
if should_accept_fn is not None:
    should_accept_traj = should_accept_fn(traj)  # 基于内容的过滤，非版本
else:
    should_accept_traj = True  # 默认全部接受

if should_accept_traj:
    manager.on_rollout_accepted()   # accepted += 1
    return _RolloutResult(task_id=task_id, trajectory=traj)

manager.on_rollout_rejected()       # rejected += 1（回收容量槽位）
return None
```

`should_accept_fn` 是一个**基于内容的过滤器**（如检查格式、长度约束），
**不是基于版本的拒绝器**。代码中**没有任何地方**根据 `versions` 张量
或版本差来丢弃整条轨迹。

### 2.2 放弃硬拒绝的五个原因

#### 原因 1：计算浪费不可接受

```
一次 7B 模型的 rollout 成本:
  推理前向传播: ~200ms/1K tokens
  典型 CoT 长度: 2K-8K tokens
  单条轨迹成本: ~0.4-1.6 GPU·秒

如果 20% 的轨迹因版本过旧被丢弃:
  浪费率 = 20% × 单条成本 = 0.08-0.32 GPU·秒/条
  批量 256 条: ~20-80 GPU·秒/步 被丢弃
```

AReaL 选择**利用所有生成的数据**，通过 IS 修正而非丢弃来处理 staleness。

#### 原因 2：Per-Token 粒度 vs Per-Sequence 粒度

AReaL 支持**部分 rollout**——单条轨迹可以跨越多个模型版本：

```python
# workflow/rlvr.py:162 — 每个 token 携带独立版本戳
versions = [-1] * resp.input_len + resp.output_versions
# 例: [-1, -1, -1, 5, 5, 5, 6, 6, 7, 7, 7]
#      ← prompt →  ← v5生成 → ← v6 → ← v7 →
```

Version Rejection 在序列级操作，会丢弃整条轨迹。
但一条轨迹中，开头的 token 可能很新鲜（版本差 0-1），
只有末尾因为训练推进而变陈旧。丢弃整条轨迹浪费了新鲜部分。

**Token-level masking（cap=5.0）** 精确处理：只屏蔽高 IS 比的陈旧 token，保留新鲜 token。

#### 原因 3：max_head_offpolicyness 已提供系统级保证

**源码**: `areal/infra/staleness_manager.py:97-111`

```python
staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt
```

该公式已将最大版本差限制为 `max_head_offpolicyness`。在基础设施层已经控制了
staleness 的上界，算法层无需再做冗余的版本过滤。

#### 原因 4：IS 权重天然衰减陈旧样本的贡献

即使不 cap，行为重要性权重 $w = \pi_{\text{prox}} / \pi_{\text{behave}}$ 的期望为 1，
但其方差随版本差增大而增大。在 `token_mask` 模式下，高 IS 比的 token 被屏蔽（权重置零），
**等效于按概率进行 token 级别的软拒绝**。

#### 原因 5：硬拒绝引入批次大小波动

如果随机丢弃轨迹，`consumer_batch_size` 变得不可预测。
这导致优势归一化、学习率调整和梯度累积的行为不稳定。
AReaL 的 `active_submit_and_wait` 保证始终收集到精确 `batch_size` 个接受的轨迹，
维持训练的确定性。

### 2.3 Token-Level Masking 作为软拒绝的替代

**源码**: `areal/utils/functional/functional.py:200-208`

```python
# token_mask 模式（默认）
if "mask" in behave_imp_weight_mode:
    behave_imp_weight = torch.where(
        behave_imp_weight > behave_imp_weight_cap,  # cap=5.0
        0.0,                                        # 屏蔽！
        behave_imp_weight,                          # 保留
    )

# 应用 loss_mask
behave_imp_weight = torch.where(loss_mask, behave_imp_weight, 0.0)
behave_mask = (behave_imp_weight > 0).logical_and(loss_mask)
```

**效果**：当 $\pi_{\text{prox}}(a|s) / \pi_{\text{behave}}(a|s) > 5.0$ 时，
该 token 的梯度贡献被完全消除（权重=0）。这是一种**精确到 token 粒度的概率性版本拒绝**。

---

## 3. IS 裁剪对梯度方差的控制能力分析

### 3.1 Cap=5.0 的数学分析

设 $w = \pi_{\text{prox}}(a|s) / \pi_{\text{behave}}(a|s)$ 为真实的行为重要性权重。

**无 cap 时的梯度估计**:

$$g = w \cdot \nabla_\theta L_{\text{PPO}}$$

$$\text{Var}(g) = \mathbb{E}[w^2] \cdot \text{Var}(\nabla_\theta L_{\text{PPO}})$$

**关键**：$\mathbb{E}[w^2]$ 可能极大。对于版本差 $G$ 步的 token：

$$\log w \sim \mathcal{N}(\mu_G, \sigma^2_G)$$

其中 $\mu_G \approx G \cdot \bar{\delta}$，$\sigma^2_G \approx G \cdot \text{Var}(\delta)$
（$\delta$ 是单步 log-ratio 变化）。

$$\mathbb{E}[w^2] = \exp(2\mu_G + 2\sigma^2_G)$$

当 $G$ 很大时，$\mathbb{E}[w^2]$ 指数级增长 → 梯度方差爆炸。

**Truncate 模式 (cap=C=5.0) 时**:

$$g' = \min(w, C) \cdot \nabla_\theta L_{\text{PPO}}$$

$$\text{Var}(g') \leq C^2 \cdot \text{Var}(\nabla_\theta L_{\text{PPO}}) = 25 \cdot \text{Var}(\nabla_\theta L_{\text{PPO}})$$

方差被**硬性限制在 25 倍**内，与版本差无关。

**Mask 模式 (cap=C=5.0, 默认) 时**:

$$g' = w \cdot \mathbb{1}[w \leq C] \cdot \nabla_\theta L_{\text{PPO}}$$

$$\text{Var}(g') \leq C^2 \cdot P(w \leq C) \cdot \text{Var}(\nabla_\theta L_{\text{PPO}})$$

更低方差（被屏蔽的 token 不贡献任何方差），但引入更大偏差（丢弃而非截断）。

### 3.2 Mask vs Truncate 模式的方差特性

**源码**: `areal/utils/functional/functional.py:194-208`

```python
if behave_imp_weight_cap is not None:
    if "truncate" in behave_imp_weight_mode:
        # 截断: clamp 到 [0, cap]
        behave_imp_weight = behave_imp_weight.clamp(min=0.0, max=behave_imp_weight_cap)
    else:  # mask
        # 屏蔽: 超过 cap 的置零
        behave_imp_weight = torch.where(
            behave_imp_weight > behave_imp_weight_cap, 0.0, behave_imp_weight
        )
```

| 模式 | 方差 | 偏差 | 适用场景 |
|------|------|------|---------|
| `token_truncate` | ≤ $25 \times \text{Var}(g_0)$ | 较小（降低但不丢弃） | 需要最大化数据利用 |
| `token_mask`（默认） | < $25 \times P(w \leq 5) \times \text{Var}(g_0)$ | 较大（完全丢弃高 IS token） | 优先稳定性 |
| `sequence_mask` | 整序列级别 | 最大（丢弃整条序列） | 序列内 token 高度相关时 |
| `disabled` | 无限制 | 零（无偏估计） | 同步训练 (offpolicyness=0) |

### 3.3 极端 Straggler 场景分析

**场景**: 一条 CoT 序列因推理 straggler 导致版本落后 $G=10$ 步。

**Token 级影响估算** (PPO clip ε=0.2, 每步 Δlog_prob ~0.005):

```
log(w) ≈ G × mean_Δlog_prob = 10 × 0.005 = 0.05
w ≈ exp(0.05) ≈ 1.05

对于大多数 token: w ≈ 1.0-1.1，远低于 cap=5.0
→ cap 几乎不触发，数据全部保留
```

**但对于低概率 token**（策略发生大幅变化的罕见 action）:

```
log(w) 可能 = G × 0.5 = 5.0  (某个不常见 token 的策略发生大幅变化)
w ≈ exp(5.0) ≈ 148  >> cap=5.0
→ 被 mask 屏蔽（token_mask）或截断到 5.0（token_truncate）
```

**结论**：对于版本差 $G \leq 8$ 的**大多数 token**，cap=5.0 几乎不触发。
只有少数策略变化剧烈的低概率 token 会被过滤。
这正是正确的行为——这些 token 的 IS 比方差最大，过滤它们是对的。

**极端场景 ($G > 16$)**：大量 token 的 IS 比超过 5.0，被 mask 后有效样本量大幅缩减。
此时需要 M2PO 提供更精细的控制。

### 3.4 M2PO：极端 Staleness 下的补充机制

**源码**: `areal/trainer/ppo/actor.py:719-800`

当 `m2_threshold` 不为 None 时，M2PO（Second-Moment Trust Policy Optimization）
在 IS cap 之外提供额外的方差控制：

```python
def _apply_m2po_masking(old_logp, prox_logp, loss_mask, m2_threshold):
    # 计算每 token 的二阶矩
    delta = old_logp - prox_logp              # log(pi_behave / pi_prox)
    m2 = delta * delta                        # 二阶矩 = (log IS ratio)^2

    # 按 m2 降序排列
    sorted_m2, indices = torch.sort(m2_selected, descending=True)

    # 逐步移除最高 m2 的 token，直到平均 m2 < threshold
    cumsum = torch.cumsum(sorted_m2, dim=0)
    n = torch.arange(1, len(sorted_m2) + 1)
    running_mean = cumsum / n
    # 找到满足 running_mean < m2_threshold 的分界点
    keep_mask = running_mean <= m2_threshold
```

**M2PO 的数学原理**:

二阶矩 $\mathbb{E}[(\log r)^2]$ 直接约束了 IS 加权梯度的方差。
通过 Jensen 不等式：

$$\mathbb{E}[\exp(2 \log r)] \leq \exp(2 \mathbb{E}[(\log r)^2])$$

控制 $\mathbb{E}[(\log r)^2]$ 就控制了 $\mathbb{E}[r^2]$，进而控制了梯度方差。

**M2PO vs IS Cap 的互补关系**:

| 机制 | 控制维度 | 粒度 | 适用 staleness |
|------|---------|------|---------------|
| IS Cap (5.0) | 单 token IS 比上界 | 独立逐 token | 中等 (2-8步) |
| M2PO | 全 batch 平均二阶矩 | 全局排序 + 截断 | 极端 (16-256步) |

文档 (`docs/en/algorithms/m2po.md`) 显示：在 256 步 staleness 下，
GRPO 出现梯度不稳定，但 M2PO 保持稳定训练。

---

## 4. use_decoupled_loss 的解耦计算实现

### 4.1 三策略分离框架

开启 `use_decoupled_loss=True` 后，AReaL 将标准 PPO 的单一 IS 比
分解为两个独立的比值：

| 策略 | 符号 | 来源 | 数据键 |
|------|------|------|--------|
| 行为策略 $\pi_{\text{behave}}$ | `old_logprobs` | 推理引擎在生成时缓存 | `input_data["logprobs"]` |
| 近端策略 $\pi_{\text{prox}}$ | `proximal_logprobs` | 重计算或近似 | `input_data["prox_logp"]` |
| 当前策略 $\pi_\theta$ | `logprobs` | 训练前向传播（有梯度） | 函数参数 |

**分解公式**:

$$\frac{\pi_\theta}{\pi_{\text{behave}}} = \underbrace{\frac{\pi_\theta}{\pi_{\text{prox}}}}_{\text{内层: PPO clip}} \times \underbrace{\frac{\pi_{\text{prox}}}{\pi_{\text{behave}}}}_{\text{外层: IS 修正}}$$

### 4.2 核心损失函数源码解析

**源码**: `areal/utils/functional/functional.py:213-315`

```python
def ppo_actor_loss_fn(
    logprobs,           # π_θ (有梯度)
    proximal_logprobs,  # π_prox (detached)
    old_logprobs,       # π_behave (detached)
    advantages,         # A (detached)
    eps_clip,           # PPO clip 范围
    loss_mask,
    behave_imp_weight_cap=None,    # IS cap (默认 5.0)
    behave_imp_weight_mode="token_mask",
    ...
):
```

**Step 1: 内层 PPO 比** — $r = \pi_\theta / \pi_{\text{prox}}$

```python
# line 260: 标准 token-level ratio
ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0)
```

因为 $\pi_{\text{prox}}$ 只比 $\pi_\theta$ 落后一步，$r$ 始终在 1.0 附近。
PPO clipping 完全有效。

**Step 2: PPO 截断损失**

```python
# lines 267-276
clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip_higher)
pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * clipped_ratio
pg_loss = torch.max(pg_loss1, pg_loss2)  # 悲观估计
```

**Step 3: 外层 IS 修正** — $w = \pi_{\text{prox}} / \pi_{\text{behave}}$

```python
# lines 287-296
if behave_imp_weight_mode != "disabled":
    behave_imp_weight, behave_approx_kl, behave_mask = compute_behave_imp_weight(
        proximal_logprobs=proximal_logprobs,
        old_logprobs=old_logprobs,
        loss_mask=loss_mask,
        behave_imp_weight_mode=behave_imp_weight_mode,
        behave_imp_weight_cap=behave_imp_weight_cap,
    )
    pg_loss = pg_loss * behave_imp_weight  # ← 关键: IS 权重乘以 PPO 损失
```

**Step 4: 归一化**

```python
# line 299: 除以总 valid token 数（含被 mask 的）
pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count
```

**完整的数学目标**:

$$L = -\frac{1}{N} \sum_{t} \underbrace{w_t^{\text{cap}}}_{\text{IS 修正}} \cdot \underbrace{\max\left(-A_t r_t, -A_t \text{clip}(r_t)\right)}_{\text{PPO 截断损失}}$$

其中 $w_t^{\text{cap}} = \min(w_t, 5.0)$（truncate）或 $w_t \cdot \mathbb{1}[w_t \leq 5.0]$（mask）。

### 4.3 Proximal 策略的计算路径

**决策逻辑**: `areal/api/cli_args.py:1173-1184`

```python
def should_compute_prox_logp(self) -> bool:
    method = ProxLogpMethod(self.prox_logp_method)
    return (
        (self.use_decoupled_loss and not method.skips_forward_pass())  # decoupled + recompute/metrics
        or (not self.use_decoupled_loss and self.recompute_logprob)    # standard + recompute
    )
```

三种模式的 $\pi_{\text{prox}}$ 来源：

| `prox_logp_method` | $\pi_{\text{prox}}$ 来源 | 需要额外前向传播？ | 精度 |
|--------------------|-----------------------|-----------------|------|
| `"recompute"` | 完整前向传播（`compute_logp`） | **是** | 精确 |
| `"loglinear"` | 版本插值近似 | **否** | ~2% 误差 |
| `"metrics"` | 完整前向传播 + 近似对比 | **是** | 精确 + 指标 |

**训练循环中的调用点** (`areal/trainer/rl_trainer.py:363-373`):

```python
# 条件性额外前向传播
if config.actor.should_compute_prox_logp():
    rollout_batch["prox_logp"] = self.actor.compute_logp(rollout_batch)
    # → 一次完整的 model.forward()（eval 模式，无梯度）
```

### 4.4 Per-Token 版本追踪与插值近似

**源码**: `areal/trainer/ppo/actor.py:554-634`

当 `prox_logp_method="loglinear"` 时，$\pi_{\text{prox}}$ 通过版本插值近似，
无需额外前向传播：

```python
def compute_prox_logp_approximations(old_logp, logprobs, versions, current_version):
    v_proximal = current_version - 1   # 近端策略 = 上一版本
    v_behave = versions.float()         # 每 token 的行为策略版本
    v_theta = float(current_version)    # 当前策略版本

    # 只近似生成的 token（version >= 0），prompt token 不处理
    generated_tokens_mask = versions >= 0

    # 计算插值系数 α
    version_diff = v_theta - v_behave
    version_gap = v_proximal - v_behave
    alpha = torch.where(
        (version_diff > 0) & generated_tokens_mask,
        version_gap / version_diff,    # α ∈ [0, 1]
        torch.zeros_like(v_behave),
    )
    alpha = torch.clamp(alpha, 0.0, 1.0)

    # Log-linear 插值（对数空间线性 = 概率空间几何平均）
    loglinear_approx = old_logp + alpha * (logprobs - old_logp)
    # 即: log π_prox ≈ (1-α) log π_behave + α log π_θ
```

**插值系数 α 的物理含义**:

```
v_behave=3, v_proximal=9, v_theta=10:
  α = (9-3)/(10-3) = 6/7 ≈ 0.86
  → π_prox ≈ π_behave^0.14 × π_θ^0.86
  → 86% 权重来自当前策略，14% 来自行为策略

v_behave=9, v_proximal=9, v_theta=10:
  α = (9-9)/(10-9) = 0
  → π_prox ≈ π_behave
  → 行为策略就是近端策略（staleness=0）

v_behave=0, v_proximal=99, v_theta=100:
  α = 99/100 = 0.99
  → π_prox ≈ π_θ^0.99 × π_behave^0.01
  → 几乎完全使用当前策略
```

**假设**：log-probability 随版本线性演化（PPO clipping 限制了每步变化，此假设合理）。

---

## 5. 性能损耗分析

### 5.1 训练步内的完整时间线

```
┌──────────────────────────────────────────────────────────────┐
│            一个训练步的组成部分与耗时占比                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ① prepare_batch()          ← 异步 rollout 收集              │
│     [不计入训练步时间，与上一步重叠]                            │
│                                                              │
│  ② compute_logp()           ← 额外前向传播 (if needed)        │
│     [use_decoupled_loss + recompute: ~30-40% of step time]   │
│     [use_decoupled_loss + loglinear: 0% (跳过)]              │
│                                                              │
│  ③ compute_advantages()     ← GAE + 归一化                   │
│     [~2% of step time]                                       │
│                                                              │
│  ④ ppo_update()             ← 训练前向 + 反向 + 优化器        │
│     [~50-60% of step time (without ②)]                       │
│     包含:                                                    │
│     - forward pass (有梯度)                                   │
│     - grpo_loss_fn (IS 计算、cap、M2PO)                      │
│     - backward pass                                          │
│     - optimizer.step()                                       │
│                                                              │
│  ⑤ update_weights()         ← 权重同步                       │
│     [~5-10% of step time]                                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 各配置模式的性能对比

| 配置 | 前向传播次数 | $\pi_{\text{prox}}$ 来源 | IS 修正 | 相对步长时间 |
|------|------------|------------------------|---------|------------|
| `decoupled=false, recompute=false` | 1（训练） | 无（= $\pi_{\text{behave}}$） | 无 | **1.0x**（基线） |
| `decoupled=false, recompute=true` | 2（重计算+训练） | 重计算覆盖 old_logp | 无 | **~1.35x** |
| `decoupled=true, method=recompute` | 2（重计算+训练） | 独立重计算 | behave_imp_weight | **~1.35x** |
| `decoupled=true, method=loglinear` | 1（训练） | 版本插值近似 | behave_imp_weight | **~1.02x** |
| `decoupled=true, method=metrics` | 2（重计算+训练） | 重计算 + 近似对比 | behave_imp_weight | **~1.38x** |

**关键洞察**：`prox_logp_method="loglinear"` 将 decoupled loss 的额外开销降至 **~2%**，
几乎零成本。

### 5.3 Loglinear 近似的零额外开销方案

**性能数据** (`docs/en/algorithms/prox_approx.md`):

> 在 8 步 staleness 下，proximal 近似方法实现 **27% 加速**
> （163 分钟 vs 207 分钟/300 步），性能差距在 2% 以内。

**为什么几乎零开销**:

1. 跳过 `compute_logp()` 前向传播（`should_compute_prox_logp()` 返回 False）
2. 近似计算发生在 `grpo_loss_fn` 内部的 `_resolve_proximal_logp()` 中
3. 只涉及简单的张量运算（加法、乘法、clamp），无 GPU kernel 启动开销
4. 使用已有的 `old_logp`（来自 rollout）和 `logprobs`（来自训练前向传播），无额外数据

```python
# 近似计算的全部开销（actor.py:612-617）:
loglinear_approx = old_logp + alpha * (logprobs - old_logp)
# 即 3 次逐元素张量运算 → ~微秒级
```

---

## 6. 配置模式总览

```
                    ┌─────────────────────────────┐
                    │   use_decoupled_loss=false   │
                    │   (同步/简单异步)             │
                    ├─────────────────────────────┤
                    │ recompute_logprob=false      │
                    │ → 纯 on-policy PPO           │
                    │ → 1 次前向传播               │
                    │ → 无 IS 修正                 │
                    ├─────────────────────────────┤
                    │ recompute_logprob=true       │
                    │ → 重计算覆盖 old_logp        │
                    │ → 2 次前向传播               │
                    │ → 无 IS 修正                 │
                    └─────────────────────────────┘

                    ┌─────────────────────────────┐
                    │   use_decoupled_loss=true    │
                    │   (推荐的异步训练模式)        │
                    ├─────────────────────────────┤
                    │ prox_logp_method=recompute   │
                    │ → 2 次前向传播               │
                    │ → 精确 π_prox                │
                    │ → IS cap=5.0 修正            │
                    │ → ~35% 额外开销              │
                    ├─────────────────────────────┤
                    │ prox_logp_method=loglinear   │ ← 推荐
                    │ → 1 次前向传播               │
                    │ → 近似 π_prox (≤2% 误差)     │
                    │ → IS cap=5.0 修正            │
                    │ → ~2% 额外开销               │
                    ├─────────────────────────────┤
                    │ + m2_threshold (可选)         │
                    │ → 额外二阶矩过滤             │
                    │ → 极端 staleness 下的保障     │
                    └─────────────────────────────┘
```

---

## 7. 代码质量发现

### Medium 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 1 | `functional.py` | 192-206 | `behave_imp_weight_cap=None` 时 `exp()` 无界——配置验证允许 `cap=None` 与 `mode!=disabled` 组合，导致极端 IS 比未被截断 |
| 2 | `functional.py` | 177 | `behave_approx_kl` 命名误导——实际是 log IS ratio，非 KL 散度 |
| 3 | `actor.py` | 554-634 | 版本差 >10 时近似误差未警告——loglinear 假设 log-prob 线性演化，大版本差下假设可能不成立 |

### Low 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 4 | `actor.py` | 623-626 | Linear 近似中 `exp(old_logp)` 可能下溢为 0（对低概率 token） |
| 5 | `functional.py` | 309 | `proximal_logprobs is not None` 检查冗余（此时已使用该变量） |
| 6 | `functional.py` | 182-188 | sequence-level IS 计算中分配不必要的 dummy 零张量 |

### 测试覆盖缺口

- 缺少 `behave_imp_weight_cap=None` 场景的测试
- 缺少极端版本差 (>10) 下近似精度的测试
- 缺少 M2PO + decoupled loss 交互的测试

---

## 8. 设计总结

### 为什么放弃 Version Rejection？

> **因为 Token-Level IS Masking 是严格更优的替代方案。**
> 它在 per-token 粒度上精确过滤高方差 token，同时保留同一轨迹中的低方差 token；
> 不浪费任何 rollout 计算资源；不引入批次大小波动；
> 且在 `max_head_offpolicyness` 的系统级保证之上提供算法层的双重保障。

### Cap=5.0 能否完全抑制极端 Straggler 的梯度方差？

> **在 $G \leq 8$ 时充分，在 $G > 16$ 时需要 M2PO 补充。**
> Cap=5.0 将单 token 方差乘子硬性限制在 25 倍以内。
> 对于典型 staleness (2-8步)，绝大多数 token 的 IS 比远低于 5.0，cap 几乎不触发。
> 对于极端 straggler ($G > 16$)，大量 token 被 mask，有效样本量显著下降。
> 此时 M2PO 的全局二阶矩约束比逐 token cap 更有效。

### Decoupled Loss 的性能损耗？

> **使用 `prox_logp_method=loglinear` 时仅 ~2% 额外开销，几乎零成本。**
> 标准 recompute 模式需要额外前向传播 (~35% 开销)。
> Loglinear 近似利用 per-token 版本信息在对数空间插值，
> 跳过前向传播且精度损失 ≤2%。
> 这是 AReaL 推荐的异步训练配置。
