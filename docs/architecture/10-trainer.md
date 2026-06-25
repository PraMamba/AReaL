# 训练器层

> 源码位置：`areal/trainer/` 文件数：15 个 | 总行数：4590 行

## 1. 模块定位

训练器层是 AReaL 系统中**训练循环的编排中枢**。它不负责底层的模型前向/反向传播（那 是 Engine 层的职责），而是负责：

- 编排多个 Engine（Actor / Critic / Ref / Teacher / Rollout）的协调调用
- 管理训练主循环中各阶段的执行顺序与资源调度
- 处理模型 offload/onload、权重同步、检查点保存与恢复
- 将训练步骤拆解为 micro-batch 并调用对应的损失函数

四种训练器覆盖四种训练范式：PPO/GRPO 强化学习、SFT 监督微调、DPO 直接偏好优化、 RW 奖励模型训练。

## 2. 文件清单

| 文件                        | 行数 | 核心职责                                |
| --------------------------- | ---- | --------------------------------------- |
| `trainer/__init__.py`       | 8    | 导出四种 Trainer                        |
| `trainer/rl_trainer.py`     | 1327 | PPOTrainer: RL 训练主循环编排           |
| `trainer/sft_trainer.py`    | 446  | SFTTrainer: 监督微调训练循环            |
| `trainer/dpo_trainer.py`    | 523  | DPOTrainer: 偏好优化训练循环            |
| `trainer/rw_trainer.py`     | 485  | RWTrainer: 奖励模型训练循环             |
| `trainer/ppo/__init__.py`   | 6    | 导出 PPOActor/PPOCritic                 |
| `trainer/ppo/actor.py`      | 1109 | PPOActor + grpo_loss_fn: 策略损失核心   |
| `trainer/ppo/critic.py`     | 149  | PPOCritic + ppo_loss_fn: 价值损失核心   |
| `trainer/ppo/stats.py`      | 38   | infer_token_denominator: token 统计辅助 |
| `trainer/dpo/__init__.py`   | 5    | 导出 DPOEngine/DPOController            |
| `trainer/dpo/dpo_engine.py` | 207  | DPOEngine + compute_dpo_loss            |
| `trainer/rw/__init__.py`    | 5    | 导出 RWEngine/RWController              |
| `trainer/rw/rw_engine.py`   | 146  | RWEngine + compute_rw_loss              |
| `trainer/sft/__init__.py`   | 5    | 导出 LMEngine/LMController              |
| `trainer/sft/lm_engine.py`  | 131  | LMEngine + compute_packed_sft_loss      |

## 3. 架构总览

```
                          +-------------------+
                          |  PPOTrainer (RL)   | <-- rl_trainer.py:105
                          |  SFTTrainer (SFT)  | <-- sft_trainer.py:54
                          |  DPOTrainer (DPO)  | <-- dpo_trainer.py:84
                          |  RWTrainer  (RW)   | <-- rw_trainer.py:76
                          +---------+---------+
                                    |
              +---------------------+---------------------+
              |                     |                     |
     +--------v--------+  +--------v--------+  +---------v--------+
     |  PPOActor        |  |  PPOCritic       |  | DPOEngine /      |
     |  (ppo/actor.py)  |  |  (ppo/critic.py) |  | RWEngine /       |
     |                  |  |                  |  | LMEngine         |
     +--------+---------+  +--------+---------+  +---------+--------+
              |                     |                      |
              v                     v                      v
     +--------+---------+  +--------+---------+  +---------+--------+
     | grpo_loss_fn      |  | ppo_loss_fn      |  | compute_dpo_loss |
     | ppo_actor_loss_fn |  | ppo_critic_loss_fn|  | compute_rw_loss  |
     | sapo_loss_fn      |  |                  |  | compute_sft_loss |
     +------------------+  +------------------+  +------------------+
              |                     |                      |
              +---------------------+----------------------+
                                    |
                          +---------v---------+
                          |  TrainEngine API   |
                          | (train_batch /     |
                          |  forward / eval)   |
                          +-------------------+
```

## 4. PPOTrainer: RL 训练主循环

### 4.1 初始化流程

PPOTrainer.__init__（第 105-383 行）完成以下初始化序列：

```
PPOTrainer.__init__
  |
  +-- 解析 ModelAllocation (actor / rollout / critic / ref / teacher)
  |     每个角色从 config.xxx.backend 解析 "fsdp" / "megatron" / "archon"
  |
  +-- _validate_cfg()                              -- 第 1194 行
  |     检查 offload / colocation / weight_update_mode 配置一致性
  |
  +-- _create_train_engine() / _create_critic()    -- 第 912 / 939 行
  |     根据 backend 选择 FSDP/Megatron/Archon 引擎类
  |     single_controller 模式: cls.as_controller(config, scheduler)
  |     SPMD 模式:              cls(config=actor_config)
  |
  +-- 创建 DataLoader
  |     online 模式: _EmptyDataLoader (第 78 行, 生成空 dict)
  |     offline 模式: StatefulDataLoader + RDataset
  |
  +-- actor/critic/ref.initialize(ft_spec, role)
  |
  +-- _init_rollout()                              -- 第 966 行
  |     选择 SGLang / vLLM 后端
  |     single_controller: RolloutController / RolloutControllerV2
  |     SPMD: RemoteSGLangEngine / RemotevLLMEngine
  |
  +-- 构建 WeightUpdateMeta (disk / xccl)          -- 第 305-346 行
  +-- actor.connect_engine(rollout, weight_update_meta)
  +-- 初始化 Evaluator / Saver / RecoverHandler / StatsLogger
  +-- recover_handler.load() 恢复检查点
  +-- _apply_initial_offload_policy()
```

### 4.2 训练主循环

PPOTrainer.train()（第 514-824 行）是 RL 训练的核心步骤编排：

```
for global_step in range(start_step, max_steps):
    |
    |  [Phase 1] Rollout
    |  +-- onload_rollout() (如果 colocation)
    |  +-- actor.prepare_batch(dataloader, workflow, ...)    -- 第 573 行
    |  +-- offload_rollout() (如果 colocation)
    |
    |  [Phase 2] Critic Values (可选)
    |  +-- onload_model(critic)
    |  +-- critic.compute_values(rollout_batch)              -- 第 595 行
    |      将 values 写入 traj["values"]
    |
    |  [Phase 3] Ref Log-Probs (可选)
    |  +-- onload_model(ref)
    |  +-- ref.compute_logp(rollout_batch)                   -- 第 612 行
    |      将 ref_logp 写入 traj["ref_logp"]
    |  +-- offload_model(ref)
    |
    |  [Phase 4] Teacher Log-Probs (可选, KDRL)
    |  +-- teacher.compute_logp(rollout_batch)               -- 第 630 行
    |      写入 traj["teacher_logp"] / rl_loss_weight / distill_loss_weight
    |
    |  [Phase 5] Recompute Proximal Log-Probs (如果需要)
    |  +-- onload_model(actor)
    |  +-- actor.compute_logp(rollout_batch)                 -- 第 652 行
    |      写入 traj["prox_logp"]
    |
    |  [Phase 6] Compute Advantages
    |  +-- actor.compute_advantages(rollout_batch)           -- 第 665 行
    |
    |  [Phase 7] PPO Update
    |  +-- saver.maybe_wait_for_staging()
    |  +-- actor.ppo_update(adv_batch)                       -- 第 685 行
    |  +-- actor.step_lr_scheduler()
    |
    |  [Phase 8] Critic Update (可选)
    |  +-- critic.ppo_update(adv_batch)                      -- 第 710 行
    |  +-- critic.step_lr_scheduler()
    |
    |  [Phase 9] Weight Sync
    |  +-- rollout.pause()
    |  +-- actor.update_weights(versioned_meta)              -- 第 733 行
    |  +-- actor/critic/rollout.set_version(new_version)
    |
    |  [Phase 10] Save & Checkpoint
    |  +-- _save_hf()
    |  +-- _save_recover_checkpoint()
    |
    |  [Phase 11] Evaluate
    |  +-- _evaluate(eval_workflow)
    |
    |  [Phase 12] Cleanup & Stats
    |  +-- clear_batches()
    |  +-- _export_and_commit_stats()
    |  +-- rollout.resume()
```

### 4.3 Offload/Onload 资源调度

PPOTrainer 管理五种模型的 offload 策略（第 136-145 行）：

| 角色    | 触发条件                    | 实现                                                    |
| ------- | --------------------------- | ------------------------------------------------------- |
| rollout | actor-rollout colocation    | \_offload_rollout: pause -> pause_generation -> offload |
| actor   | colocation 或 actor.offload | \_offload_model(actor)                                  |
| critic  | critic.offload              | \_offload_model(critic)                                 |
| ref     | ref.offload                 | \_offload_model(ref)                                    |
| teacher | teacher.offload             | \_offload_model(teacher)                                |

Rollout 的 offload 最复杂（第 422-500 行），需三步序列： `pause() -> pause_generation() -> offload()` ；
onload 反向执行：`onload() -> continue_generation() -> resume()` 。

### 4.4 在线模式 vs 离线模式

```
                                PPOTrainer
                               /          \
                    online 模式              offline 模式
                    (agent.mode="online")    (标准 dataset)
                    /                         \
    _EmptyDataLoader                    StatefulDataLoader
    (生成空 dict)                        (RDataset / HF Dataset)
    |                                   |
    不创建 eval_rollout                  创建 eval_rollout
    不支持 valid_dataset                 支持 valid_dataset
    启动 proxy_gateway                   标准 workflow 驱动
```

## 5. PPOActor / PPOCritic: 策略与价值损失核心

### 5.1 PPOActor 三大方法

PPOActor（第 43-366 行）封装策略端的三个核心操作：

**compute_logp**（第 131 行）：调用 engine.forward() 计算当前策略的 log-prob，用 于 proximal policy 的重计算。

**compute_advantages**（第 142-249 行）：完整的优势估计流水线：

```
compute_advantages(data)
  |
  +-- reward_overlong_penalty()          -- 超长序列惩罚
  +-- reward scaling/clipping/norm       -- 奖励预处理
  +-- KL 奖励计算: -kl_ctl * kl_estimator(old_logp, ref_logp)
  +-- 末尾 token 附加 task_reward
  +-- GAE 反向递推:
  |     for t in reversed(range(max_seqlen-1)):
  |       delta = rewards[:,t] + gamma * V(t+1) - V(t)
  |       A(t) = delta + gamma * lambda * A(t+1)
  +-- advantage normalization (可选)
  +-- 写入 data["advantages"] / data["returns"]
```

**ppo_update**（第 253-366 行）：将数据拆为 micro-batch，对每个 mb 调用 engine.train_batch(mb,
loss_fn=grpo_loss_fn) 。

### 5.2 grpo_loss_fn: 统一损失入口

grpo_loss_fn（第 408-596 行）是策略端损失的统一入口，它内部路由到三种损失函数：

```
grpo_loss_fn(logprobs, entropy, input_data, ...)
  |
  +-- _resolve_proximal_logp()              -- 确定 proximal policy log-prob
  |     三种方法:
  |     recompute  : 使用 forward pass 重计算的 prox_logp_gt
  |     loglinear  : log-linear 版本插值近似（无需额外 forward）
  |     metrics    : recompute + 记录近似误差指标
  |
  +-- _apply_m2po_masking() (可选)          -- M2PO 高方差 token 过滤
  |
  +-- 损失计算 (二选一):
  |     use_sapo_loss=True  -> sapo_loss_fn()
  |     use_sapo_loss=False -> ppo_actor_loss_fn()
  |       参数: eps_clip, eps_clip_higher, c_clip (dual clip),
  |              rejection_sampling, importance_sampling_level
  |
  +-- Teacher KD 损失 (可选):
  |     rl_loss_weight=0  -> 纯 reverse KL importance-sampling
  |     rl_loss_weight>0  -> KDRL 联合损失: rl_loss + distill_loss
  |
  +-- 统计日志: importance_weight, approx_kl, entropy, clip_ratio
  +-- _log_proximal_approximation_stats()
  +-- _log_version_staleness_stats()
```

### 5.3 Proximal Policy 近似机制

compute_prox_logp_approximations（第 604-684 行）实现版本感知的策略近似：

```
版本关系:
  v_behave (rollout 时的策略版本)
  v_proximal = v_theta - 1  (上一次广播的策略版本)
  v_theta (当前训练版本)

插值因子:
  alpha = (v_proximal - v_behave) / (v_theta - v_behave)
  alpha in [0, 1], 仅对 generated tokens (version >= 0) 生效

三种近似方法:
  loglinear : log(p_prox) = (1-a)*log(p_behave) + a*log(p_theta)     -- 对数空间插值
  linear    : p_prox = (1-a)*p_behave + a*p_theta, 再取 log           -- 概率空间插值
  rollout   : p_prox = p_behave (无近似, 仅用于对比指标)
```

### 5.4 PPOCritic

PPOCritic（第 25-74 行）较为简洁：

- compute_values: engine.eval() + forward，输出 value 预测
- ppo_update: 拆 micro-batch，调用 ppo_critic_loss_fn（value clipping 损失）

## 6. Controller vs ControllerV2 架构演进

每种 Engine 组件都有两代 Controller 实现：

```
                    +------------------+
                    |  TrainController  | <-- areal.infra
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                                       |
+--------v---------+              +--------------v-----------+
| PPOActorController|              | GatewayTrainController    | <-- areal.experimental
| PPOCriticController|             +-------------+------------+
| DPOController     |                            |
| RWController      |              +-------------+------------+
| LMController      |              | PPOActorControllerV2     |
+------------------+              | PPOCriticControllerV2    |
                                  | DPOControllerV2          |
                                  | RWControllerV2           |
                                  | LMControllerV2           |
                                  +-------------------------+
```

**V1 (TrainController)**：基于 RPC 的 \_custom_function_call 方法，通过 rpc_meta={"broadcast":
True} 将调用广播到所有 worker。

```python
# actor.py:368  PPOActorController
class PPOActorController(TrainController):
    def compute_logp(self, *args, **kwargs):
        return self._custom_function_call(
            "compute_logp", *args, rpc_meta={"broadcast": True}, **kwargs)
```

**V2 (GatewayTrainController)**：基于 HTTP Gateway 的序列化调用，通过 \_gateway_post /
\_gateway_post_result 发送 JSON payload。

```python
# actor.py:385  PPOActorControllerV2
class PPOActorControllerV2(GatewayTrainController):
    def compute_logp(self, *args, **kwargs):
        payload = {"args": serialize_value(list(args)),
                   "kwargs": serialize_value(kwargs)}
        return self._gateway_post_result("/ppo/actor/compute_logp", payload)
```

V2 位于 areal.experimental.training_service，对应 RolloutControllerV2 （config.\_version ==
"v2"）。

### 各 Controller 暴露的 RPC 端点

| 组件      | V1 方法                                      | V2 HTTP 端点                                                              |
| --------- | -------------------------------------------- | ------------------------------------------------------------------------- |
| PPOActor  | compute_logp, compute_advantages, ppo_update | /ppo/actor/compute_logp, /ppo/actor/compute_advantages, /ppo/actor/update |
| PPOCritic | compute_values, ppo_update                   | /ppo/critic/compute_values, /ppo/critic/update                            |
| DPOEngine | train_dpo, evaluate_dpo, compute_logp        | /dpo/train, /dpo/evaluate, /dpo/compute_logp                              |
| RWEngine  | train_rw, evaluate_rw                        | /rw/train, /rw/evaluate                                                   |
| LMEngine  | train_lm, evaluate_lm                        | /sft/train, /sft/evaluate                                                 |

## 7. SFT / DPO / RW 训练器的共同模式

三种非 RL 训练器共享几乎相同的骨架结构：

### 7.1 共同训练循环模板

```
__init__:
  +-- 解析 ModelAllocation
  +-- _create_actor() (backend 分发: FSDP/Megatron/Archon)
  +-- 创建 DataLoader (+ 可选 DataController for RDataset)
  +-- actor.initialize(ft_spec, role="actor")
  +-- 初始化 Evaluator / Saver / RecoverHandler / StatsLogger
  +-- recover_handler.load()

train():
  for global_step in range(start_step, max_steps):
      +-- _load_bcast_from(data_generator)    -- 加载 + 广播
      +-- saver.maybe_wait_for_staging()
      +-- actor.train_xxx(batch)               -- 核心训练步
      +-- actor.step_lr_scheduler()
      +-- actor.set_version(global_step + 1)
      +-- _save_hf()
      +-- _save_recover_checkpoint()
      +-- _evaluate()
      +-- clear_batches()
      +-- _export_and_commit_stats()
```

### 7.2 差异对比

| 维度         | SFT                     | DPO                          | RW                           |
| ------------ | ----------------------- | ---------------------------- | ---------------------------- |
| 配置类       | SFTConfig               | DPOConfig                    | RWConfig                     |
| 引擎类       | LMEngine                | DPOEngine                    | RWEngine                     |
| 核心训练调用 | actor.train_lm(batch)   | actor.train_dpo(batch)       | actor.train_rw(batch)        |
| collate 函数 | collate_samples_to_list | dpo_modeling_collate_fn      | rw_modeling_collate_fn       |
| 损失函数     | compute_packed_sft_loss | compute_dpo_loss             | compute_rw_loss              |
| 是否需要 ref | 否                      | 是 (ref.compute_logp)        | 否                           |
| 数据格式     | input_ids + loss_mask   | chosen_ids/rejected_ids 成对 | chosen_ids/rejected_ids 成对 |
| group_size   | 1                       | 2 (chosen + rejected)        | 2 (chosen + rejected)        |

### 7.3 各损失函数细节

**compute_packed_sft_loss**（lm_engine.py:81）：

- 负对数似然：loss = -logprobs.sum() / n_valid_tokens
- 支持 packed sequence（cu_seqlens 格式）
- 统计 PPL（perplexity）per sequence

**compute_dpo_loss**（dpo_engine.py:147）：

- 支持 sigmoid / hinge / ipo 三种 loss_type
- IPO 模式按 completion 长度归一化 logratios
- 统计 chosen_reward, rejected_reward, reward_accuracy, reward_margin

**compute_rw_loss**（rw_engine.py:115）：

- Bradley-Terry 排序损失：loss = -log_sigmoid(score_chosen - score_rejected)
- 从 terminal token 提取 score
- 处理 padded pair（valid_pairs 过滤）

## 8. 关键设计决策与约束

### 8.1 Trainer 与 Engine 的职责边界

Trainer 不直接操作模型参数或梯度。它通过 engine.train_batch(input\_, loss_fn=...) 将损失函数以闭包形式传入
Engine。Engine 负责前向传播、调用 loss_fn、反向传播和优化器 步进。这种设计使得：

- 同一个 Trainer 可以驱动 FSDP / Megatron / Archon 三种后端
- 损失函数与并行策略完全解耦

### 8.2 Single-Controller vs SPMD 双模式

所有 Trainer 通过 is_single_controller() 分支支持两种部署模式：

- **Single-Controller**：主进程创建 Controller 代理，通过 RPC 调度远程 worker
  - cls.as_controller(config, scheduler) 创建控制器
  - 数据直接从 dataloader 获取
- **SPMD**：每个 rank 运行完整训练逻辑
  - cls(config=config) 直接创建本地引擎
  - 数据需要 broadcast_tensor_container() 跨模型并行组广播

### 8.3 异步 Rollout 与训练的交织

PPOTrainer 的训练循环中，rollout 和训练是**同步交替**执行的： 每步先 rollout 收集数据，再训练更新，最后同步权重到推理端。 异步性体现在：

- rollout 的推理引擎在训练期间被 pause() / resume() 控制
- colocation 模式下通过 offload/onload 在同一组 GPU 上轮流执行
- 权重更新通过 WeightUpdateMeta 支持 disk 和 xccl 两种模式
- 版本号 (set_version) 用于追踪策略的 staleness

### 8.4 M2PO 高方差 token 过滤

M2PO（Second-Momentum PPO）机制（actor.py:769-850）通过计算 old_logp 与 prox_logp 之间差异的二阶矩来识别高方差
token，并将其从损失计算中移除：

```
delta = old_logp - prox_logp
m2 = delta^2
对 m2 排序, 从最大值开始移除, 直到剩余平均 m2 < threshold
```

### 8.5 KDRL 联合损失

当配置了 teacher 模型时，grpo_loss_fn 支持两种知识蒸馏模式（actor.py:482-512）：

- rl_loss_weight=0：纯 reverse KL importance-sampling 蒸馏
- rl_loss_weight>0：KDRL 联合损失 = rl_loss_weight * rl_loss + distill_loss_weight \*
  rkl_penalty

### 8.6 Rejection Sampling

grpo_loss_fn 通过 RejectionSamplingConfig 支持基于 token/sequence 级别的拒绝 采样，在损失计算前过滤不符合条件的样本。
