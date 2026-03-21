# AReaL Rollout Buffer 设计与 Staleness 控制深度分析

> 基于源码的详细分析，覆盖 `max_head_offpolicyness` 容量公式推导、动态门控机制原理、
> 与 RL 算法层的交互，以及面对变长 CoT 序列时的吞吐量优势。

---

## 目录

1. [系统架构总览](#1-系统架构总览)
2. [max_head_offpolicyness 容量公式详解](#2-max_head_offpolicyness-容量公式详解)
   - 2.1 [配置定义](#21-配置定义)
   - 2.2 [核心公式推导](#22-核心公式推导)
   - 2.3 [双重约束的动态平衡](#23-双重约束的动态平衡)
   - 2.4 [Rollout 生命周期状态机](#24-rollout-生命周期状态机)
   - 2.5 [公式随训练进程的演变](#25-公式随训练进程的演变)
3. [多层队列架构与数据流](#3-多层队列架构与数据流)
   - 3.1 [五层流水线架构](#31-五层流水线架构)
   - 3.2 [BatchTaskDispatcher 调度核心](#32-batchtaskdispatcher-调度核心)
   - 3.3 [AsyncTaskRunner 异步执行引擎](#33-asynctaskrunner-异步执行引擎)
   - 3.4 [反压机制汇总](#34-反压机制汇总)
4. [训练循环中的 Pause/Resume 协议](#4-训练循环中的-pauseresume-协议)
5. [与 RL 算法层的交互](#5-与-rl-算法层的交互)
   - 5.1 [Per-Token 版本追踪](#51-per-token-版本追踪)
   - 5.2 [Decoupled PPO 目标函数](#52-decoupled-ppo-目标函数)
   - 5.3 [行为重要性权重截断](#53-行为重要性权重截断)
   - 5.4 [Proximal 策略近似](#54-proximal-策略近似)
6. [动态门控 vs 固定深度队列：面对变长 CoT 的吞吐量分析](#6-动态门控-vs-固定深度队列面对变长-cot-的吞吐量分析)
   - 6.1 [固定深度队列的局限性](#61-固定深度队列的局限性)
   - 6.2 [动态门控的四大优势](#62-动态门控的四大优势)
   - 6.3 [吞吐量提升的定量分析](#63-吞吐量提升的定量分析)
   - 6.4 [实际配置与经验数据](#64-实际配置与经验数据)
7. [代码质量发现](#7-代码质量发现)
8. [设计总结](#8-设计总结)

---

## 1. 系统架构总览

AReaL 的 Rollout Buffer 不是一个简单的队列，而是一个**五层流水线系统**，核心由
`StalenessManager` 提供基于公式的动态容量门控。

```
PPOTrainer.train()                    ← 训练循环（消费者）
    │
    ▼
RolloutController / WorkflowExecutor  ← 编排层（版本管理、Worker 调度）
    │
    ▼
BatchTaskDispatcher                   ← 调度层（容量门控、生产者-消费者线程）
    │
    ▼
StalenessManager                      ← 准入控制（staleness 公式计算）
    │
    ▼
AsyncTaskRunner                       ← 执行层（uvloop 异步任务引擎）
```

两条并行部署路径共享同一个 `BatchTaskDispatcher` + `StalenessManager`：

| 路径 | 适用场景 | 任务执行方式 |
|------|---------|------------|
| **Single-Controller 模式** | 分布式集群 | `RolloutController` → RPC 到远程 Worker |
| **SPMD 模式** | 单机/本地 | `WorkflowExecutor` → 进程内直接执行 Workflow |

---

## 2. max_head_offpolicyness 容量公式详解

### 2.1 配置定义

**源码**: `areal/api/cli_args.py:1625-1632`

```python
@dataclass
class InferenceEngineConfig:
    max_head_offpolicyness: int = field(
        default=0,
        metadata={
            "help": "Maximum off-policyness for the head. "
            "If the current version is more than this many versions behind, "
            "the request will not be accepted.",
        },
    )
    consumer_batch_size: int = field(default=1, ...)
    max_concurrent_rollouts: None | int = field(default=None, ...)  # 默认等于 consumer_batch_size
    queue_size: None | int = field(default=None, ...)               # 默认 max_concurrent_rollouts × 16
```

**关键配置参数**:

| 参数 | 含义 | 默认值 | 影响 |
|------|------|--------|------|
| `max_head_offpolicyness` (η) | 允许的最大版本陈旧度 | 0（同步） | 控制吞吐量-稳定性权衡 |
| `consumer_batch_size` (B) | 每个训练步消费的样本数 | 1 | 缩放容量预算 |
| `max_concurrent_rollouts` (C) | 最大并发 rollout 数 | B | GPU 利用率上限 |
| `queue_size` | AsyncTaskRunner 队列大小 | C × 16 | 执行管线深度 |

### 2.2 核心公式推导

**源码**: `areal/infra/staleness_manager.py:77-111`

`StalenessManager.get_capacity()` 是整个系统的核心准入函数。它计算两个独立约束的最小值：

```python
def get_capacity(self) -> int:
    with self.lock:
        current_version = self.version_provider.get_version()

        # 约束 1: 并发约束
        concurrency_capacity = max(1, max_concurrent_rollouts) - running

        # 约束 2: 陈旧度约束
        ofp = max_staleness  # 即 max_head_offpolicyness
        sample_cnt = accepted + running
        staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt

        # 取两者最小值
        return min(concurrency_capacity, staleness_capacity)
```

#### 公式拆解

设：
- $V$ = `current_version`（当前模型版本，每训练步 +1）
- $S$ = `max_staleness`（即 `max_head_offpolicyness`）
- $B$ = `consumer_batch_size`（每训练步消费的样本数）
- $C$ = `max_concurrent_rollouts`（最大并发数）
- $n_{\text{accepted}}$ = 历史累计已接受样本数（单调递增）
- $n_{\text{running}}$ = 当前正在运行的 rollout 数

**约束 1: 并发约束**

$$\text{concurrency\_capacity} = C - n_{\text{running}}$$

物理含义：限制同时占用推理 GPU 的任务数，防止推理服务器过载。

**约束 2: 陈旧度约束（核心公式）**

$$\text{staleness\_capacity} = (S + V + 1) \times B - (n_{\text{accepted}} + n_{\text{running}})$$

**推导过程**：

1. 在版本 $V$ 时，训练已经完成 $V$ 步，共消费了 $V \times B$ 个样本
2. 系统允许的最大陈旧度为 $S$，即允许存在版本差最多为 $S$ 的未消费样本
3. 因此，系统生命周期内允许的总样本数上限为：

$$N_{\max}(V) = (S + V + 1) \times B$$

其中 $+1$ 是因为当前训练步本身需要一个 batch。

4. 已生产（含运行中）的样本总数为 $n_{\text{accepted}} + n_{\text{running}}$
5. 剩余容量 = 上限 - 已生产 = $(S + V + 1) \times B - (n_{\text{accepted}} + n_{\text{running}})$

**最终容量**:

$$\text{capacity} = \min(\text{concurrency\_capacity}, \text{staleness\_capacity})$$

最严格的约束胜出。

#### 稳态分析

在稳态下，$n_{\text{accepted}} \approx V \times B$（每步消费 $B$ 个样本），因此：

$$\text{staleness\_capacity}_{\text{steady}} \approx (S + V + 1) \times B - V \times B = (S + 1) \times B$$

这意味着**稳态下的有效缓冲区深度恒为 $(S+1) \times B$**，独立于训练进度。

### 2.3 双重约束的动态平衡

```
                     容量
                      ▲
                      │
   (S+1)×B ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ← 陈旧度稳态上限
                      │          ╱
         C ─ ─ ─ ─ ─ ┼─ ─ ─ ─╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ← 并发上限 C
                      │      ╱   (并发约束成为瓶颈)
                      │    ╱
                      │  ╱ (陈旧度约束是瓶颈)
                      │╱
                      ┼───────────────────────────────→ 版本 V
                    V=0
```

| 训练阶段 | 瓶颈约束 | 原因 |
|----------|---------|------|
| 早期（低 V） | 陈旧度 | $(S + V + 1) \times B$ 较小 |
| 中后期（高 V） | 并发 | 陈旧度容量已远超 $C$ |
| 权重更新期间 | 暂停 | `pause()` 阻止所有新提交 |
| 恢复后首步 | 陈旧度 | $n_{\text{accepted}}=0$ 但 $V$ 可能很大 → 容量突然很大，并发约束兜底 |

### 2.4 Rollout 生命周期状态机

**源码**: `areal/infra/staleness_manager.py:113-146`

```
           submit_task_input()          runner.submit()
  创建 ───────────────────→ enqueued ──────────────────→ running
                              │                            │
                              │ on_rollout_enqueued()       │ on_rollout_submitted()
                              │ enqueued += 1               │ enqueued -= 1
                              │                             │ running += 1
                              │                             │
                              │                     ┌───────┴───────┐
                              │                     │               │
                              │              on_accepted()    on_rejected()
                              │              running -= 1     running -= 1
                              │              accepted += 1    rejected += 1
                              │                     │               │
                              │                     ▼               ▼
                              │                  accepted       rejected
                              │               (计入容量)     (回收容量)
```

**关键设计**：被拒绝的样本不增加 `accepted` 计数器，这意味着它们的容量槽位被"回收"。
这是正确的——被拒绝的样本不会被训练消费，不应计入陈旧度预算。

### 2.5 公式随训练进程的演变

以 $S=2$，$B=8$，$C=32$ 为例：

| 版本 V | 陈旧度容量 $(S+V+1) \times B$ | 并发容量 $C$ | 有效容量 | 瓶颈 |
|--------|------|------|---------|------|
| 0 | $(2+0+1) \times 8 = 24$ | 32 | 24 | 陈旧度 |
| 1 | $(2+1+1) \times 8 = 32$ | 32 | 32 | 持平 |
| 2 | $(2+2+1) \times 8 = 40$ | 32 | 32 | 并发 |
| 5 | $(2+5+1) \times 8 = 64$ | 32 | 32 | 并发 |
| 10 | $(2+10+1) \times 8 = 104$ | 32 | 32 | 并发 |

> 注：稳态下 $n_{\text{accepted}} \approx V \times B$，所以实际陈旧度容量 $\approx (S+1) \times B = 24$，
> 但随着 $V$ 增长，暂时的脉冲（如恢复后的突发）有更大的缓冲空间。

### 2.6 辅助公式：get_pending_limit()

**源码**: `areal/infra/staleness_manager.py:67-75`

```python
def get_pending_limit(self) -> int:
    return (self.max_staleness + 1) * self.consumer_batch_size
```

这是一个**版本无关的静态上限**，用于 `active_submit_and_wait()` 中限制输入 deque 的大小：

$$\text{pending\_limit} = (S + 1) \times B$$

与 `get_capacity()` 的区别：

| 方法 | 版本感知 | 用途 | 调用者 |
|------|---------|------|--------|
| `get_capacity()` | 是 | 控制 runner 提交 | 生产者线程 `_commit_loop` |
| `get_pending_limit()` | 否 | 控制 pending deque 大小 | `active_submit_and_wait` |

---

## 3. 多层队列架构与数据流

### 3.1 五层流水线架构

```
主线程 (训练循环)
    │ submit_task_input()
    ▼
┌─────────────────────────────┐
│   _pending_inputs (deque)   │  ← 无界输入缓冲（受 get_pending_limit() 软限制）
│   受 staleness 门控         │
└──────────┬──────────────────┘
           │ 生产者线程 (_commit_loop)
           │ 检查: get_capacity() > 0 AND 不暂停 AND 有待处理项
           ▼
┌─────────────────────────────┐
│ AsyncTaskRunner.input_queue │  ← 有界队列 (size = queue_size)
│    (queue.Queue)            │
└──────────┬──────────────────┘
           │ uvloop 后台线程
           │ 执行异步任务 (aiohttp RPC / 本地 workflow)
           ▼
┌─────────────────────────────┐
│ AsyncTaskRunner.output_queue│  ← 有界队列 (size = queue_size)
│    (queue.Queue)            │
└──────────┬──────────────────┘
           │ 消费者线程 (_fetch_loop)
           │ 50ms 超时轮询
           ▼
┌─────────────────────────────┐
│ _pending_results (dict)     │  ← 无界结果缓冲
│    {task_id: TimedResult}   │
└──────────┬──────────────────┘
           │ 主线程 wait_results() / active_submit_and_wait()
           ▼
       训练循环消费
```

### 3.2 BatchTaskDispatcher 调度核心

**源码**: `areal/infra/workflow_executor.py:257-726`

三个关键协调点：

#### (1) 生产者线程 - `_commit_loop` (line 354-388)

```python
def _commit_loop(self) -> None:
    while not self._shutdown_event.is_set():
        task_input = self._get_next_task_for_submission()  # 阻塞直到有容量
        if task_input is None:
            continue
        task_fn = self.task_factory(task_input)
        try:
            self.runner.submit(task_fn, task_id=task_input.task_id)
            self.staleness_manager.on_rollout_submitted()  # enqueued--, running++
        except TaskQueueFullError:
            self._pending_inputs.appendleft(task_input)    # 放回队列
            self._input_cv.wait_for(lambda: ... _has_runner_capacity())
```

#### (2) 容量门控 - `_get_next_task_for_submission` (line 431-444)

```python
def _get_next_task_for_submission(self) -> TInput | None:
    with self._input_cv:
        while not self._shutdown_event.is_set():
            if (
                not self.runner.paused.is_set()          # 未暂停
                and self.staleness_manager.get_capacity() > 0  # 有容量
                and self._pending_inputs                 # 有待处理项
            ):
                return self._pending_inputs.popleft()
            self._input_cv.wait()  # 阻塞等待条件变化
    return None
```

#### (3) 流水线提交 - `active_submit_and_wait` (line 629-726)

这是吞吐量优化的核心方法。与"提交一批 → 等待全部完成 → 再提交"的串行模式不同，
它实现了**持续提交与结果收集的重叠**：

```python
def active_submit_and_wait(self, input_generator, batch_size, dynamic_bs=False):
    while True:
        # 1. 计算提交容量
        pending_inputs = len(self._pending_inputs)
        cap_staleness = staleness_manager.get_pending_limit() - pending_inputs
        cap_queue = runner.max_queue_size - (runner.get_input_queue_size() + batch_size)
        capacity = min(cap_staleness, cap_queue)

        # 2. 持续提交新任务（不等待旧任务完成）
        if capacity > 0:
            for _ in range(min(batch_size, capacity)):
                self.submit_task_input(next(input_generator))

        # 3. 非阻塞收集结果（1s 超时）
        try:
            arrived = self.wait_results(count=batch_size - accepted_cnt, timeout=1)
        except TimeoutError:
            arrived = []

        # 4. 处理结果，直到满足 batch_size
        for res in arrived:
            if res is not None:
                accepted_cnt += 1
                results.append(res)
            if accepted_cnt >= batch_size:
                break
```

### 3.3 AsyncTaskRunner 异步执行引擎

**源码**: `areal/infra/async_task_runner.py`

| 特性 | 实现 |
|------|------|
| 事件循环 | **uvloop**（比默认 asyncio ~2-4x 快） |
| 队列 | `queue.Queue(maxsize=max_queue_size)` × 2（输入/输出） |
| 任务管理 | `asyncio.Task` + `asyncio.wait()` |
| 跨线程唤醒 | `loop.call_soon_threadsafe(input_event.set)` |
| 暂停/恢复 | `threading.Event` |
| 重复检测 | `_active_task_ids: set[int]` |
| 轮询参数 | 50ms wait + 500ms sleep |

### 3.4 反压机制汇总

系统有四层反压机制，从外到内：

| 层级 | 机制 | 源码位置 | 触发条件 |
|------|------|---------|---------|
| **1. Staleness 容量** | `get_capacity() <= 0` | `staleness_manager.py:110` | 样本总量超过版本允许上限 |
| **2. Pending Limit** | `get_pending_limit() - pending <= 0` | `workflow_executor.py:673` | 待处理输入超过静态上限 |
| **3. Runner 队列满** | `input_queue.qsize() >= max_queue_size` | `workflow_executor.py:327,678` | 异步执行管线饱和 |
| **4. Pause/Resume** | `runner.paused.is_set()` | `workflow_executor.py:437` | 权重更新期间全面暂停 |

---

## 4. 训练循环中的 Pause/Resume 协议

**源码**: `areal/trainer/rl_trainer.py:444-527`

训练步内的完整时序：

```
┌──────────────────────────────────────────────────────────────┐
│                    一个训练步的时序                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. prepare_batch()                                          │
│     └─ active_submit_and_wait() ← 异步提交 + 收集结果        │
│        └─ 持续调用 get_capacity() 门控提交                    │
│                                                              │
│  2. compute_advantages()                                     │
│     └─ GAE + 优势归一化                                      │
│                                                              │
│  3. ppo_update()                                             │
│     └─ 分微批次计算损失 + 梯度更新                             │
│                                                              │
│  4. ★ self.rollout.pause()   ← 暂停所有新提交                │
│     │                          (运行中任务继续完成)            │
│     │                                                        │
│  5. │ update_weights()                                       │
│     │   ├─ new_version = global_step + 1                     │
│     │   ├─ actor.update_weights(versioned_meta)  ← NCCL 广播 │
│     │   ├─ actor.set_version(new_version)                    │
│     │   ├─ critic.set_version(new_version)                   │
│     │   └─ rollout.set_version(new_version)                  │
│     │       └─ version_provider._version = new_version       │
│     │          → StalenessManager 下次 get_capacity() 将看到  │
│     │            新版本 → 容量预算增加 B                       │
│     │                                                        │
│  6. │ save_checkpoint()                                      │
│  7. │ evaluate()                                             │
│  8. │ clear_batches()                                        │
│     │                                                        │
│  9. ★ self.rollout.resume()  ← 恢复提交                      │
│     └─ 新容量 = (S + V_new + 1) × B - accepted_old          │
│        → 立即可提交新一轮 rollout                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

这形成了自然的 **"吸气-呼气"节律**：
- **呼气**（步骤 1-3）：持续生成和消费 rollout
- **吸气**（步骤 4-9）：暂停生成，更新权重，扩展容量预算，恢复生成

---

## 5. 与 RL 算法层的交互

`max_head_offpolicyness` 不仅控制缓冲区容量，还直接影响 RL 算法的损失计算。
AReaL 提供了多层算法修正来处理 off-policy 数据。

### 5.1 Per-Token 版本追踪

**源码**: `areal/infra/remote_inf_engine.py:822-823`, `areal/workflow/rlvr.py:162`

每个生成的 token 携带生成时的模型版本戳：

```python
# remote_inf_engine.py: 生成时记录版本
accumulated_versions.extend(
    [self.get_version()] * len(gen_result.output_tokens)
)

# rlvr.py: 打包到轨迹张量中
versions = [-1] * resp.input_len + resp.output_versions
# prompt tokens 标记为 -1，生成 tokens 标记为实际版本号
```

在异步训练中，一条轨迹可能跨越多个模型版本（部分 rollout），
因此 per-token 版本追踪是精确 off-policy 修正的基础。

### 5.2 Decoupled PPO 目标函数

**源码**: `areal/trainer/ppo/actor.py:358-546`

标准 PPO 假设 $\pi_{\text{old}} = \pi_{\text{proximal}}$（rollout 策略 = 上一步策略），
但在异步训练中这不成立。Decoupled PPO 区分三个策略：

| 策略 | 符号 | 来源 |
|------|------|------|
| 行为策略 $\pi_{\text{behave}}$ | `old_logprobs` | Rollout 时缓存的 logprobs |
| 近端策略 $\pi_{\text{proximal}}$ | `proximal_logprobs` | 训练前一步的策略（重计算或近似） |
| 当前策略 $\pi_\theta$ | `logprobs` | 当前训练前向传播 |

损失函数变为：

$$L = -\mathbb{E}\left[\frac{\pi_{\text{prox}}}{\pi_{\text{behave}}} \cdot \min\left(r \cdot A, \text{clip}(r) \cdot A\right)\right]$$

其中 $r = \pi_\theta / \pi_{\text{prox}}$ 是标准 PPO ratio，
$\pi_{\text{prox}} / \pi_{\text{behave}}$ 是**行为重要性权重**，
用于修正行为策略与近端策略之间的分布偏移。

### 5.3 行为重要性权重截断

**源码**: `areal/utils/functional/functional.py:144-210`

为防止极端 off-policy 修正导致训练不稳定，提供 `behave_imp_weight_cap`（默认 5.0）：

| 模式 | 行为 |
|------|------|
| `token_mask` | 将 $\pi_{\text{prox}} / \pi_{\text{behave}} > \text{cap}$ 的 token 权重置零 |
| `token_truncate` | 将比率截断到 $[0, \text{cap}]$ |
| `sequence_mask` | 序列级别置零 |
| `sequence_truncate` | 序列级别截断 |

### 5.4 Proximal 策略近似

**源码**: `areal/trainer/ppo/actor.py:554-604`

计算 $\pi_{\text{proximal}}$ 通常需要额外的前向传播。AReaL 提供基于版本的插值近似：

```python
# 假设 v_proximal = current_version - 1
v_proximal = current_version - 1

# 计算插值因子 alpha
# v_behave == v_proximal → alpha=0（使用 old_logp）
# v_behave == v_theta → alpha=1（使用 logprobs）
alpha = (v_proximal - v_behave) / (v_theta - v_behave)
alpha = clamp(alpha, 0.0, 1.0)

# 对数线性插值
loglinear_approx = old_logp + alpha * (logprobs - old_logp)
```

根据文档（`docs/en/algorithms/prox_approx.md`），在 8 步陈旧度下，
近似方法与精确重计算的性能差距在 2% 以内，同时节省 27% 的训练时间。

### 5.5 Staleness 与算法的完整交互链

```
max_head_offpolicyness = S
        │
        ├──→ StalenessManager: 控制缓冲区容量 → 决定样本的最大版本差
        │
        ├──→ Per-Token Versions: 记录每个 token 的生成版本
        │       │
        │       ├──→ Proximal 近似: alpha = (v_prox - v_behave) / (v_theta - v_behave)
        │       │    └─ S 越大 → alpha 范围越大 → 近似误差可能增大
        │       │
        │       └──→ 版本陈旧度统计: v_theta - v_behave 的分布
        │
        ├──→ Behave Importance Weight: pi_prox / pi_behave
        │       └─ S 越大 → 版本差越大 → 比率偏离 1.0 越多 → 需要更强的截断
        │
        └──→ M2PO (可选): 基于二阶矩约束过滤高方差 token
                └─ 在极端 S（如 256）下保持训练稳定性
```

---

## 6. 动态门控 vs 固定深度队列：面对变长 CoT 的吞吐量分析

### 6.1 固定深度队列的局限性

传统固定深度队列 `queue.Queue(maxsize=N)` 的反压机制：
当队列满时阻塞生产者，当队列空时阻塞消费者。

**面对变长 CoT 的问题**：

```
场景: 8 个推理 Worker，队列深度 = 32

时刻 t0: 32 个任务在队列中
时刻 t1: 30 个短序列 (1s) 完成，2 个长 CoT (60s) 仍在运行
         队列迅速排空 → 30 个 Worker 空闲
时刻 t2: 新任务无法提交（因为之前的 batch 还没全部完成？
         或者队列已重新填满但 Worker 分配不均？）

问题:
1. 队列深度固定，不感知模型版本 → 可能积累过多陈旧样本
2. 无法区分"30 个短任务完成"和"2 个长任务仍在运行"
3. 头部阻塞: wait_results(count=32) 必须等最慢的那个完成
```

### 6.2 动态门控的四大优势

#### 优势 1: 自适应准入控制

固定队列按计数反压。动态门控按**语义容量**反压。

长 CoT 任务长时间占据 `running` 状态，自然消耗容量预算，
无需完成就能减缓新任务提交速率。这是固定队列无法实现的隐式负载感知。

```
动态门控:
  running = 20 (含 2 个 60s 长 CoT)
  concurrency_capacity = 32 - 20 = 12  ← 仍有空间提交新任务
  staleness_capacity = (S+V+1)×B - (accepted+20) ← 版本感知

固定队列:
  queue_size = 32, current = 30
  capacity = 32 - 30 = 2  ← 只看计数，不看运行状态
```

#### 优势 2: 版本感知的全局预算

固定队列没有"模型版本"概念。即使训练卡住（版本不增长），
固定队列仍会继续接受新样本，导致缓冲区充满过时数据。

动态门控在训练卡住时自动收缩容量：

```
训练卡住: version 停留在 V=5
staleness_capacity = (S+5+1)×B - accepted
随着 accepted 增长 → staleness_capacity → 0 → 停止提交
```

#### 优势 3: 流水线重叠（`active_submit_and_wait`）

**源码**: `areal/infra/workflow_executor.py:629-726`

这是动态门控最关键的吞吐量优化。与固定队列的"提交全部 → 等待全部"不同，
`active_submit_and_wait` 实现了**持续提交与结果收集的重叠**：

```
固定队列模式:                   动态门控 + 流水线模式:

提交 32 个任务                   提交一批任务
    ↓                               ↓
等待 32 个全部完成               ┌─→ 收集已完成的结果 (1s 超时)
    ↓                            │   ↓
提交下一批 32 个                 │   短任务完成 → 立即提交替补任务
    ↓                            │   ↓
等待...                          └── 循环直到 batch_size 个结果就绪

                                 长 CoT 60s 期间:
                                 - 短任务持续进出
                                 - Worker 保持忙碌
                                 - 不存在"等最慢的"瓶颈
```

#### 优势 4: 组合约束的弹性

`min(concurrency, staleness)` 的组合意味着系统自动适应当前瓶颈：

| 场景 | 瓶颈 | 系统行为 |
|------|------|---------|
| 短序列为主 | 并发 | 快速完成 → 容量快速恢复 → 高吞吐 |
| 长 CoT 为主 | 陈旧度 | running 占满 → 容量缩减 → 等待完成后恢复 |
| 混合负载 | 动态切换 | 短任务快速轮转填补长任务留下的空闲 |

### 6.3 吞吐量提升的定量分析

#### 理论模型

设：
- $T_{\text{short}}$ = 短序列完成时间（~1s）
- $T_{\text{long}}$ = 长 CoT 完成时间（~60s）
- $p$ = 长 CoT 比例（如 20%）
- $B$ = batch_size = 32
- $W$ = Worker 数量 = 8

**固定队列（串行 batch）:**

$$T_{\text{batch}} = \max(T_1, T_2, ..., T_B) \approx T_{\text{long}}$$

每个 batch 的时间被最慢的任务决定。Worker 平均利用率：

$$\text{utilization}_{\text{fixed}} = \frac{\sum T_i}{W \times T_{\text{batch}}} = \frac{0.8B \times T_{\text{short}} + 0.2B \times T_{\text{long}}}{W \times T_{\text{long}}}$$

代入数值：$\frac{0.8 \times 32 \times 1 + 0.2 \times 32 \times 60}{8 \times 60} = \frac{25.6 + 384}{480} \approx 85\%$

但这是理论值。实际上由于 batch 之间的间隙，利用率更低。

**动态门控（流水线）:**

短任务完成后立即提交新任务，Worker 几乎无空闲：

$$\text{utilization}_{\text{dynamic}} \approx \min\left(1, \frac{\text{capacity} \times \bar{T}_{\text{task}}}{W}\right)$$

在容量充足时接近 100%。关键改进是消除了 batch 边界等待。

**粗略吞吐量比**：

在 20% 长 CoT 场景下：

$$\frac{\text{throughput}_{\text{dynamic}}}{\text{throughput}_{\text{fixed}}} \approx \frac{T_{\text{long}}}{\bar{T}_{\text{task}}} = \frac{60}{0.8 \times 1 + 0.2 \times 60} = \frac{60}{12.8} \approx 4.7\times$$

这个 **~5x 提升**来自消除了长尾等待——固定队列每个 batch 被最慢的 60s 任务拖累，
而动态门控在这 60s 内持续处理短任务。

### 6.4 实际配置与经验数据

**源码**: `docs/en/algorithms/async.md`

| 配置 | 建议范围 | 说明 |
|------|---------|------|
| `max_head_offpolicyness` | 2-8（通常），最高 16-256（M2PO） | η=0 为同步，通常 2x 慢 |
| `consumer_batch_size` | 取决于训练 batch size | 通常 = GPU 数 × 微批次 |
| `max_concurrent_rollouts` | 默认 = consumer_batch_size | 可调大以提升推理 GPU 利用率 |

**AReaL v0.3 论文配置** (`blog/AReaL_v0_3.md:282`):
- `max_head_offpolicyness η = 16`
- Batch Size = 2048

**官方文档经验**:
> "Setting `max_head_offpolicyness=0` reverts AReaL to synchronous RL. The synchronous setting
> is useful for debugging but is typically **2x slower** than asynchronous training."

> 在 8 步陈旧度下，proximal 近似方法实现 **27% 加速**（163 分钟 vs 207 分钟/300 步）。

---

## 7. 代码质量发现

### Critical 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 1 | `workflow_executor.py` | 532-539 | `_input_cv` 和 `_result_cv` 之间的锁顺序：`submit_task_input` 先获取 `_input_cv` 再获取 `_result_cv`，需确保其他路径不会反序获取 |
| 2 | `async_task_runner.py` | 352-365 | `CancelledError` 绕过 `on_rollout_rejected()` 回调，导致 `running` 计数器永久偏移，逐步"泄漏"容量 |

### High 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 3 | `rollout_controller.py` | 1028-1035 | `set_version` 在 `_version_lock` 内执行同步 RPC（60s 超时），阻塞所有 `get_version()` 调用 → 阻塞容量计算 → 阻塞整个 rollout 管线 |
| 4 | `async_task_runner.py` | 385-397 | 输出队列满时 runner 线程死亡，无恢复机制 |
| 5 | `staleness_manager.py` | 97-111 | `get_capacity()` 持锁调用外部 `get_version()`，若其涉及 I/O 则成为瓶颈 |

### Moderate 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 6 | `staleness_manager.py` | 104-107 | `accepted` 计数器假设消费速率与 `consumer_bs/version` 匹配。高拒绝率下容量公式过于宽松 |
| 7 | `async_task_runner.py` | 324-326 | 暂停期间不排空已完成结果，恢复时可能导致输出队列瞬间溢出 |
| 8 | `workflow_executor.py` | 287 | `_pending_inputs` 是无界 deque，理论上可无限增长（虽然受 staleness 门控软限制） |

### 设计观察

| # | 关注点 | 说明 |
|---|--------|------|
| 9 | 恢复后容量突变 | Checkpoint 恢复时 `accepted=0` 但 `version` 可能很大 → 容量瞬间很高 → 并发约束兜底 |
| 10 | 多轮对话放大 | `MultiTurnWorkflow` 的 `max_turns` 放大单任务 `running` 时长，需相应调大 `max_concurrent_rollouts` |
| 11 | Round-robin 负载不均 | `_choose_worker` 轮询分配，不感知 Worker 实际负载。长 CoT 场景下某些 Worker 过载 |

---

## 8. 设计总结

### 核心公式一览

| 公式 | 表达式 | 用途 |
|------|--------|------|
| **陈旧度容量** | $(S + V + 1) \times B - (n_{\text{accepted}} + n_{\text{running}})$ | 控制总样本量不超过版本允许上限 |
| **并发容量** | $C - n_{\text{running}}$ | 限制同时占用推理 GPU 的任务数 |
| **有效容量** | $\min(\text{陈旧度容量}, \text{并发容量})$ | 最终准入决策 |
| **待处理上限** | $(S + 1) \times B$ | 输入 deque 的软限制 |
| **稳态缓冲深度** | $(S + 1) \times B$ | 长期运行时的有效缓冲区大小 |

### 动态门控 vs 固定队列

| 维度 | 固定深度队列 | 动态 Staleness 门控 |
|------|------------|-------------------|
| 反压信号 | 队列计数 | 版本 + 并发 + 运行状态 |
| 版本感知 | 无 | 有（容量随版本线性增长） |
| 变长适应 | 差（等最慢的） | 好（短任务持续轮转） |
| 流水线 | 难（batch 边界） | `active_submit_and_wait` 消除边界 |
| 训练卡住保护 | 无（持续填充过时数据） | 有（版本不增 → 容量归零） |
| 估算吞吐量提升 | 基线 | **~2-5x**（取决于 CoT 长尾比例） |

### 一句话总结

> AReaL 的 Rollout Buffer 通过将**模型版本信息注入容量函数**，
> 实现了一个能自动感知训练进度、动态调整缓冲深度的准入控制系统。
> 配合 `active_submit_and_wait` 的流水线重叠和算法层的多级 off-policy 修正，
> 在保持训练稳定性的同时，显著提升了变长 CoT 场景下的推理 GPU 利用率。
