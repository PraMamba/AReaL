# AReaL 部分 Rollout 处理与软暂停策略深度分析

> 基于源码的详细分析，覆盖软暂停的具体实现机制、长尾序列的等待缓解策略、
> 训练器闲置问题，以及为何未采用"中止+带前缀重试"方案。

---

## 目录

1. [软暂停机制的双层架构](#1-软暂停机制的双层架构)
2. [长尾等待的缓解机制](#2-长尾等待的缓解机制)
3. [训练器闲置时间分析](#3-训练器闲置时间分析)
4. [为什么不选择"中止+带前缀重试"](#4-为什么不选择中止带前缀重试)
5. [Per-Token 版本追踪与部分 Rollout 的正确性](#5-per-token-版本追踪与部分-rollout-的正确性)
6. [代码质量发现](#6-代码质量发现)
7. [设计总结](#7-设计总结)

---

## 1. 软暂停机制的双层架构

AReaL 的"软暂停"实际上由**两层独立的暂停机制**组成，各自服务不同目的：

### 第一层：Dispatcher 级暂停 — `rollout.pause()`

**调用点**: `areal/trainer/rl_trainer.py:445`

```python
# pause inference for updating weights, save, and evaluation
self.rollout.pause()
```

**传播路径**:

```
PPOTrainer.train()
  └─ self.rollout.pause()
      └─ RolloutController.pause()          (rollout_controller.py:1041-1043)
          ├─ self.dispatcher.pause()         ← 本地 BatchTaskDispatcher
          │   └─ self.runner.pause()         ← 设置 AsyncTaskRunner.paused 事件
          │       └─ _commit_loop 检测到 paused → 停止从 _pending_inputs 取任务
          │
          └─ self._collective_rpc("pause")   ← 远程 Worker 的 WorkflowExecutor
```

**效果**: 停止向异步执行引擎提交**新任务**。已在运行的异步任务**继续执行直到完成**。

### 第二层：Server 级暂停 — `pause_generation()`

**调用点**: `areal/engine/fsdp_engine.py:1089-1090`（在 `_update_weights_from_distributed` 内部）

```python
if dist.get_rank() == 0:
    self.rollout_engine.pause_generation()
```

**传播路径**:

```
FSDPEngine._update_weights_from_distributed()
  └─ self.rollout_engine.pause_generation()
      └─ RolloutCallback.pause_generation()  (同步 HTTP POST)
          └─ RolloutController → 集体 RPC → 所有推理 Worker
              └─ RemoteInfEngine.pause_generation()
                  ├─ HTTP → SGLang: /pause_generation
                  │   └─ 停止接受新生成请求
                  │
                  └─ HTTP → vLLM: /areal_pause_generation
                      └─ _generation_run_event.clear()
                      └─ abort_all_reqs()  ← 中止所有运行中请求！
                      └─ reset_prefix_cache()  ← 清除 KV 缓存！
```

**效果**: 告知推理服务器停止接受新生成请求。vLLM 额外执行**强制中止所有在运行请求**。

### 关键差异：vLLM 实际是"硬暂停"

**源码**: `areal/engine/vllm_ext/areal_vllm_server.py:227-234`

```python
@router.post("/areal_pause_generation")
async def pause_generation(raw_request: Request):
    logger.info("API server starts pause_generation and aborts all requests")
    llm = raw_request.app.state.engine_client
    _generation_run_event.clear()                           # 门控新请求
    await llm.engine_core.call_utility_async("abort_all_reqs")  # 中止所有！
    return to_json_response(True, "Generation paused and all requests aborted")
```

`abort_all_reqs` 的实现（同文件 line 274-313）：

```python
def abort_all_reqs(self):
    scheduler = self.scheduler
    abort_lists = list(scheduler.running) + list(scheduler.waiting)
    # ... 逐个中止 ...
    success = scheduler.reset_prefix_cache()  # 清除 KV 缓存防污染
```

**因此**：
- **SGLang**: 停止接受新请求，在运行请求自然完成（真正的"软暂停"）
- **vLLM**: 立即中止所有请求 + 清除 KV 缓存（实际是"硬暂停"）

### 完整时序

```
训练循环:

  T1: ppo_update()                     ← 梯度更新
  T2: self.rollout.pause()             ← Dispatcher 停止提交新任务
      │                                    (但已提交的异步任务继续运行)
      │
  T3: self.actor.update_weights(meta)  ← 进入权重同步
      │
      ├─ pause_generation()            ← 推理服务器停止/中止生成
      │   ├─ vLLM: abort_all_reqs()    ← 强制中止 + 清 KV cache
      │   └─ SGLang: 停止接受新请求     ← 在运行请求继续
      │
      ├─ time.sleep(pause_grace_period)  ← 默认 0.0s
      ├─ dist.barrier()
      ├─ NCCL chunked broadcast        ← 权重传输
      ├─ dist.barrier()
      │
      └─ continue_generation()         ← 推理服务器恢复
  T4: set_version(new_version)
  T5: save_hf()                        ← 保存检查点
  T6: save_recover_checkpoint()
  T7: evaluate()                       ← 评估（推理 GPU 此时已恢复但 dispatcher 仍暂停！）
  T8: clear_batches()
  T9: export_stats()
  T10: self.rollout.resume()           ← Dispatcher 恢复提交
```

---

## 2. 长尾等待的缓解机制

### 核心问题

对于 32K token 的长序列，如果采用纯粹的"等待自然完成"策略：

```
32K tokens / 50 tokens/sec (单序列) ≈ 640 秒 ≈ 10.7 分钟
```

这种等待是否会造成训练器长时间闲置？

### AReaL 的答案：**训练器不等待**

AReaL 的设计中，**训练器实际上不等待任何在运行的 rollout 完成**。时间线分析：

#### 机制 1：vLLM 直接中止，零等待

vLLM 后端的 `pause_generation()` 调用 `abort_all_reqs()`，**立即中止所有在运行请求**。
无论 32K 序列还是 1K 序列，全部被中止。等待时间 = HTTP 请求延迟 ≈ 毫秒级。

被中止的请求通过**客户端重试循环**自动恢复（见机制 3）。

#### 机制 2：SGLang 的 pause 行为

SGLang 使用原生 `/pause_generation` 端点。其具体行为取决于 SGLang 版本，
但典型实现是停止接受新请求并等待在运行请求完成。

然而，`pause_grace_period` 默认为 0.0，意味着系统**不会显式等待 SGLang 排空**。
权重同步在 `dist.barrier()` 后立即开始，可能与 SGLang 的尾部生成重叠。

#### 机制 3：客户端重试循环 — 中止后自动恢复

**源码**: `areal/infra/remote_inf_engine.py:768-798`

```python
# Deal with rollout interruption
stop_reason = None
ori_max_new_tokens = gconfig.max_new_tokens
while (
    stop_reason not in ["stop", "tool_calls", "length"]
    and len(accumulated_output_tokens) < ori_max_new_tokens
):
    # Request is interrupted, wait for some time to avoid interfering
    # with update weights requests
    while self.workflow_executor.is_paused():
        await asyncio.sleep(0.5)           # ← 等待 pause 结束

    # Build request using backend (会带上当前版本号)
    http_req = self.backend.build_generation_request(
        req,
        with_lora=self.config.use_lora,
        version=self.get_version(),        # ← 使用更新后的版本
    )

    result = await arequest_with_retry(...)
    # ... 处理结果，累积 tokens ...
```

**关键设计**：

1. 每次生成请求中断后，客户端进入**暂停等待循环**（`while is_paused(): sleep(0.5)`）
2. 暂停结束后（权重已更新），使用**新版本号**重新构建请求
3. 新的 tokens 使用 `self.get_version()` 打版本戳
4. 最终的 `output_versions` 是混合版本的：前半部分 v_old，后半部分 v_new

#### 机制 4：KV Cache 亲和性路由

**源码**: `areal/infra/remote_inf_engine.py:753-763`

```python
# A single "rid" shares the same server to allow KV cache reuse
if req.rid in self.rid_to_address:
    server_addr = self.rid_to_address[req.rid]
else:
    server_addr = self.choose_server()
    # ... cache the mapping ...
    self.rid_to_address[req.rid] = server_addr
```

中断重试时，**相同的 `rid` 路由到相同的服务器**。
SGLang 的 prefix matching 机制可以复用已生成部分的 KV cache（如果未清除）。
vLLM 则因 `reset_prefix_cache()` 清除了 KV cache，需要重新计算。

#### 机制 5：Staleness 容量自动调整

权重更新后版本号增加，StalenessManager 的容量公式自动扩展：

```
版本 V → V+1:
staleness_capacity 增加 consumer_batch_size 个槽位
```

恢复后的 rollout 立即获得新的提交配额。

### 缓解效果汇总

| 缓解机制 | 效果 | 适用后端 |
|----------|------|---------|
| vLLM abort_all_reqs | 零等待，立即中止 | vLLM |
| 客户端重试循环 | 中断→等待 pause 结束→用新权重继续 | 全部 |
| KV Cache 亲和性 | rid 路由到相同 server，潜在 cache 复用 | SGLang |
| pause_grace_period=0 | 不等待排空 | 全部 |
| Per-token 版本戳 | 混合版本序列的正确 IS 修正 | 全部 |

---

## 3. 训练器闲置时间分析

### 闲置发生在哪里？

```
┌──────────────────────────────────────────────────────────┐
│          pause() → resume() 之间的时间分解                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ① update_weights()        ~1-6s (NCCL chunked)         │
│     ├─ pause_generation()   ~10-100ms                    │
│     ├─ NCCL broadcast       ~0.5-5s (取决于模型/网络)     │
│     └─ continue_generation() ~10ms                       │
│                                                          │
│  ② set_version()           ~100ms (HTTP RPC)             │
│                                                          │
│  ③ save_hf()               ~2-30s (取决于模型大小/磁盘)   │  ← 推理 GPU 已恢复
│                                                          │     但 dispatcher 仍暂停！
│  ④ save_checkpoint()       ~2-10s                        │
│                                                          │
│  ⑤ evaluate()              ~0-300s (取决于评估设置)       │  ← 可能很长！
│                                                          │
│  ⑥ clear_batches()         ~10ms                         │
│  ⑦ export_stats()          ~100ms                        │
│                                                          │
│  Total: ~5-350s                                          │
└──────────────────────────────────────────────────────────┘
```

### 关键发现：推理 GPU 在 ③-⑦ 期间空闲

`continue_generation()` 在 ① 结束时就被调用了（`fsdp_engine.py:1134`），
此时推理服务器已经可以接受新请求。

但 `self.rollout.resume()` 直到 ⑦ 之后才被调用（`rl_trainer.py:527`），
这意味着 BatchTaskDispatcher 在 ②-⑦ 期间**不提交任何新任务**。

**推理 GPU 的无效闲置时间 = ③ + ④ + ⑤ + ⑥ + ⑦ ≈ 4-340 秒**

这是一个**明显的优化机会**：将 `resume()` 提前到 `update_weights()` 完成后，
使 rollout 生成与 save/eval/stats 重叠。

### 训练 GPU 的闲置时间

训练 GPU 在 pause→resume 期间执行的都是有用工作（save、eval、stats），
因此训练 GPU 实际上**不闲置**。只有推理 GPU 因 dispatcher 暂停而闲置。

### 与训练步总时间的比例

| 模型规模 | 权重同步 | save+eval | 总 pause 时长 | 训练步总时长 | 暂停占比 |
|----------|---------|-----------|-------------|------------|---------|
| 7B | ~1s | ~5s | ~6s | ~30-60s | ~10-20% |
| 70B | ~5s | ~30s | ~35s | ~120-300s | ~12-28% |
| 70B+eval | ~5s | ~300s | ~305s | ~420-600s | ~50-72% |

**有评估的训练步中，推理 GPU 闲置比例可达 50%+**。

---

## 4. 为什么不选择"中止+带前缀重试"

### SkyRL 方案的核心思路

SkyRL 的做法：发现权重更新时间到了 → 立即中止所有在运行的生成 → 用新权重从 prefix 位置重新生成。

### AReaL 不采用此方案的原因

#### 原因 1：AReaL 的客户端重试循环已实现等效功能

**源码**: `remote_inf_engine.py:768-798`

AReaL 的 `_agenerate` 方法天然支持中断恢复：

```python
while stop_reason not in ["stop", "tool_calls", "length"]:
    while self.workflow_executor.is_paused():
        await asyncio.sleep(0.5)          # 等 pause 结束

    http_req = self.backend.build_generation_request(
        req, version=self.get_version()   # 用新版本
    )
    result = await arequest_with_retry(...)
    accumulated_output_tokens.extend(gen_result.output_tokens)
    accumulated_versions.extend([self.get_version()] * len(gen_result.output_tokens))
```

这**本质上就是"中止+带前缀重试"**——vLLM 后端通过 `abort_all_reqs()` 中止所有请求，
然后客户端重试循环自动用新权重续生成。区别在于 AReaL 是在客户端侧实现的，
而非在服务器侧。

#### 原因 2：Per-Token 版本追踪消除了"全新权重重生成"的必要性

SkyRL 中止后从 prefix 重试是为了确保所有 token 都使用最新权重生成。
AReaL 的 per-token 版本追踪允许混合版本序列：

```
同一条轨迹的 versions 张量:
[-1, -1, -1, 5, 5, 5, 5, 6, 6, 6, 6, 6]
 ← prompt →  ← v5 生成 →  ← v6 生成 →
```

算法层（decoupled loss + proximal 近似）能正确处理这种混合版本序列，
因此不需要确保所有 token 来自同一版本。

#### 原因 3：KV Cache 污染问题

如果用旧权重的 KV cache 配合新权重续生成，KV cache 中的 key/value 向量
是由旧模型计算的，而 attention 查询是由新模型计算的。这种不匹配可能导致
生成质量下降。

AReaL 的 vLLM 实现通过 `reset_prefix_cache()` 解决了这个问题（清除所有 KV cache）。
SGLang 依赖其 prefix matching 机制来决定是否复用。

#### 原因 4：架构简洁性

"中止+重试"需要服务器侧维护生成进度的状态（已生成的 token 数、KV cache 位置），
并在重试时恢复。AReaL 的客户端重试循环将这一状态维护在客户端
（`accumulated_output_tokens`、`accumulated_versions`），不依赖服务器端状态。

#### 原因 5：权重更新频率已由 `max_head_offpolicyness` 控制

SkyRL 强调"更高的权重更新频率"。AReaL 通过 `max_head_offpolicyness` 参数
在基础设施层控制了更新频率的上界。在典型配置 (η=2-8) 下，权重更新频率已经足够高，
不需要通过中止正在运行的序列来进一步提高频率。

### AReaL 方案 vs SkyRL 方案对比

| 维度 | AReaL 软暂停 | SkyRL 中止+重试 |
|------|------------|----------------|
| 中止时机 | vLLM: 立即中止; SGLang: 自然完成 | 立即中止 |
| 恢复方式 | 客户端重试循环（新权重续生成） | 服务器侧 prefix 重试 |
| KV Cache | vLLM 清除; SGLang 保留 | 保留并复用 |
| 版本一致性 | per-token 版本追踪（允许混合） | 全新权重（保证单一版本） |
| 复杂性 | 客户端侧（简单） | 服务器侧（需要状态管理） |
| 计算浪费 | 已中止的 tokens 需重新生成 | 已中止的 tokens 需重新生成 |
| 适用场景 | 通用（不依赖特定推理框架） | 需要推理框架支持 prefix resume |

**实际区别不大**：在 vLLM 后端下，AReaL 事实上已经实现了"中止+重试"——
只是重试逻辑在客户端而非服务器侧。

---

## 5. Per-Token 版本追踪与部分 Rollout 的正确性

### 混合版本序列的产生

```
时间线:

  T1: 推理引擎版本 = 5
      开始生成 token_1, token_2, ..., token_100
      → versions = [5, 5, 5, ..., 5]  (100 个 token)

  T2: pause_generation() → 请求被中止/暂停

  T3: 权重更新 → 推理引擎版本 = 6

  T4: continue_generation() + dispatcher resume

  T5: 客户端重试循环检测到 is_paused()=False
      用 version=6 续生成 token_101, ..., token_200
      → accumulated_versions = [5]*100 + [6]*100
```

### 算法层的正确处理

**源码**: `areal/trainer/ppo/actor.py:554-603`

```python
# Per-token 插值系数
v_behave = versions.float()     # [5, 5, ..., 5, 6, 6, ..., 6]
v_theta = float(current_version)  # 比如 7
v_proximal = current_version - 1  # 6

# 对于 v_behave=5 的 token:
#   alpha = (6-5)/(7-5) = 0.5 → 50% 插值
# 对于 v_behave=6 的 token:
#   alpha = (6-6)/(7-6) = 0   → 直接用 old_logp
```

每个 token 根据自己的版本戳获得独立的 IS 修正，混合版本序列的处理是数学上正确的。

---

## 6. 代码质量发现

### High 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 1 | `rl_trainer.py` | 445-527 | `resume()` 在 save/eval/stats 之后才调用。推理 GPU 在 `continue_generation()` 后到 `resume()` 之间空闲。建议将 `resume()` 提前到 `update_weights()` 完成后 |

### Medium 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 2 | `remote_inf_engine.py` | 1178-1180 | `pause_grace_period` 默认 0.0s，依赖 HTTP 请求阻塞行为但无验证机制 |
| 3 | `async_task_runner.py` | 324-327 | 暂停期间已完成的 asyncio 任务不被 reap，结果延迟到恢复后才进入输出队列 |
| 4 | `rl_trainer.py` | 445/1090 | 两阶段暂停（dispatcher pause 在 T2，server pause 在 T3）之间的窗口内推理服务器仍在运行 |

### Positive 发现

| # | 文件 | 行号 | 亮点 |
|---|------|------|------|
| 5 | `areal_vllm_server.py` | 274-313 | `abort_all_reqs` 正确重置 prefix cache，防止 KV cache 被旧权重污染 |
| 6 | `rollout_controller.py` | 1041-1047 | pause/resume 顺序不对称设计正确（pause: 本地先→远程后; resume: 远程先→本地后） |
| 7 | `remote_inf_engine.py` | 768-798 | 客户端重试循环优雅处理中断恢复，自动使用新版本号 |

---

## 7. 设计总结

### 核心问题的直接回答

**Q: 32K token 的长尾序列是否导致训练器长时间闲置？**

> **不会**。vLLM 后端通过 `abort_all_reqs()` 立即中止所有在运行请求（~毫秒级），
> SGLang 依赖 `pause_grace_period`（默认 0.0s）。权重同步不等待任何序列完成。
>
> 真正的闲置发生在 `update_weights()` 完成后到 `rollout.resume()` 之间的
> save/eval/stats 阶段（~4-300s），此时**推理 GPU 已恢复但 dispatcher 未提交新任务**。
> 这是目前最大的优化机会。

**Q: 为什么不用 SkyRL 的"中止+带前缀重试"？**

> AReaL 在 vLLM 后端下**事实上已经实现了等效方案**：
> `abort_all_reqs()` 中止所有请求 → 客户端重试循环等待 pause 结束 →
> 用新权重重新发起请求。差异仅在于重试逻辑在客户端而非服务器侧。
>
> AReaL 额外提供了 per-token 版本追踪，使得**即使不中止、让旧序列自然完成**，
> 算法层也能通过 decoupled loss 正确处理混合版本数据。这是比"全部中止"更通用的方案。

### 软暂停的本质

```
                    AReaL 的"软暂停"实际上是三种策略的组合

    ┌──────────────────────────────────────────────────┐
    │  1. Dispatcher 级: 停止提交新任务                  │
    │     → 控制增量（不新增 off-policy 数据）           │
    │                                                  │
    │  2. Server 级: 中止/暂停在运行请求                 │
    │     → vLLM: 硬中止 (abort_all_reqs)              │
    │     → SGLang: 软暂停 (停止新请求，在运行请求完成)   │
    │                                                  │
    │  3. 客户端重试: 中断恢复循环                       │
    │     → 自动等待 pause 结束                         │
    │     → 用新版本号续生成                             │
    │     → Per-token 版本追踪保证算法正确性              │
    └──────────────────────────────────────────────────┘
```

### 最高优先级优化建议

**将 `self.rollout.resume()` 从 line 527 提前到 `update_weights()` 完成后（约 line 466）**：

```python
# 当前实现:
self.rollout.pause()       # line 445
update_weights()           # line 458  ← 权重同步
set_version()              # line 460
# --- 推理 GPU 已恢复，但 dispatcher 仍暂停 ---
save_hf()                  # line 475
save_checkpoint()          # line 485
evaluate()                 # line 497  ← 可能很长！
clear_batches()            # line 515
export_stats()             # line 522
self.rollout.resume()      # line 527  ← 太晚了！

# 建议优化:
self.rollout.pause()       # line 445
update_weights()           # line 458
set_version()              # line 460
self.rollout.resume()      # ← 提前到这里！
save_hf()                  # 与新 rollout 生成重叠
save_checkpoint()          # 与新 rollout 生成重叠
evaluate()                 # 与新 rollout 生成重叠
```

这可以将推理 GPU 闲置时间从 save+eval+stats 的总和（~4-300s）**减少到接近零**。
