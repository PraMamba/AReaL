# 调度循环与 KV-Cache 内存管理深度分析

> 基于源码的底层分析，覆盖软暂停时 vLLM KV-cache 的排空机制、调度循环的四阶段流水线、
> 以及请求中止→缓存清理→权重广播→恢复生成的完整内存管理时序。

---

## 目录

1. [调度循环的四阶段流水线架构](#1-调度循环的四阶段流水线架构)
2. [软暂停时的 KV-Cache 排空机制](#2-软暂停时的-kv-cache-排空机制)
3. [vLLM abort_all_reqs 的完整内存清理流程](#3-vllm-abort_all_reqs-的完整内存清理流程)
4. [权重更新期间的内存状态转换](#4-权重更新期间的内存状态转换)
5. [调度循环的暂停/恢复协议](#5-调度循环的暂停恢复协议)
6. [vLLM vs SGLang 的 KV-Cache 管理差异](#6-vllm-vs-sglang-的-kv-cache-管理差异)
7. [代码质量发现](#7-代码质量发现)
8. [设计总结](#8-设计总结)

---

## 1. 调度循环的四阶段流水线架构

### 1.1 数据结构与线程映射

**源码**: `areal/infra/workflow_executor.py:257-726`, `areal/infra/async_task_runner.py`

```
Main Thread              Producer Thread          AsyncTaskRunner Thread       Consumer Thread
(active_submit_and_wait) (_commit_loop)           (uvloop _run_async_loop)     (_fetch_loop)

submit_task_input()      _get_next_task_for_      _drain_pending_inputs()      runner.wait()
      │                    submission()                  │                          │
      ▼                        │                        ▼                          ▼
┌──────────────┐         ┌─────┴─────┐           ┌──────────────┐          ┌──────────────┐
│_pending_inputs│ ──────→ │ capacity  │ ────────→ │ input_queue  │ ───────→ │ output_queue │
│  (deque)     │         │   gate    │           │ (Queue, 有界) │          │ (Queue, 有界) │
│  无锁保护:   │         │ 三重检查: │           │              │          │              │
│  _input_cv   │         │ ①not paused│          │  ↓ asyncio   │          │              │
└──────────────┘         │ ②capacity>0│          │  ↓ Tasks     │          └──────┬───────┘
                         │ ③has items │          │  ↓ (running) │                 │
                         └───────────┘           └──────────────┘          ┌──────▼───────┐
                                                                          │_pending_results│
                                                                          │  (dict)       │
                                                                          │  _result_cv   │
                                                                          └──────┬───────┘
                                                                                 │
                                                                          wait_results()
                                                                          → Main Thread
```

### 1.2 容量门控的三重检查

**源码**: `workflow_executor.py:431-444`

```python
def _get_next_task_for_submission(self) -> TInput | None:
    with self._input_cv:
        while not self._shutdown_event.is_set():
            self._check_thread_exception()
            if (
                not self.runner.paused.is_set()           # ① 未暂停
                and self.staleness_manager.get_capacity() > 0  # ② 有容量
                and self._pending_inputs                  # ③ 有待处理项
            ):
                return self._pending_inputs.popleft()
            self._input_cv.wait()  # 阻塞直到条件变化
    return None
```

**三重检查的必要性**:
- 检查 ① 在暂停期间立即阻止提交
- 检查 ② 通过 staleness 公式动态限制提交
- 检查 ③ 防止空取

### 1.3 反馈循环闭合

**源码**: `workflow_executor.py:418-420`

```python
# 消费者线程在收集结果后通知生产者
with self._input_cv:
    self._input_cv.notify()  # ← 唤醒 _get_next_task_for_submission
```

当 rollout 完成时，`on_rollout_accepted()/on_rollout_rejected()` 更新 staleness 计数器。
消费者线程收集结果后通知生产者线程重新评估容量——**形成闭合的反压反馈循环**。

### 1.4 避免忙等待的四种机制

| 等待点 | 机制 | 文件:行号 |
|--------|------|----------|
| 生产者线程空闲 | `_input_cv.wait()` (Condition Variable) | `workflow_executor.py:442` |
| 异步循环空闲 | `await _input_event.wait()` (asyncio Event) | `async_task_runner.py:471` |
| 异步循环暂停 | `await asyncio.sleep(0.5)` (轮询) | `async_task_runner.py:326` |
| 消费者线程空闲 | `output_queue.get(timeout=0.05)` | `async_task_runner.py:616` |
| 主线程等待结果 | `_result_cv.wait(timeout=remaining)` | `workflow_executor.py:581` |

---

## 2. 软暂停时的 KV-Cache 排空机制

### 2.1 暂停的两层协议回顾

```
Layer 1: rollout.pause()         → 停止调度循环提交新任务
Layer 2: pause_generation()      → 停止推理服务器接受/执行生成请求
```

**KV-Cache 排空发生在 Layer 2**，具体行为取决于推理后端。

### 2.2 vLLM 的"硬排空"机制

**源码**: `areal/engine/vllm_ext/areal_vllm_server.py:227-234`

```python
@router.post("/areal_pause_generation")
async def pause_generation(raw_request: Request):
    llm = raw_request.app.state.engine_client
    _generation_run_event.clear()                              # ① 门控新请求
    await llm.engine_core.call_utility_async("abort_all_reqs") # ② 中止所有并释放 KV
    return to_json_response(True, "...")                        # ③ 只有 ② 完成后才响应
```

**三步序列**:

1. **`_generation_run_event.clear()`** — 立即关闭新请求入口
2. **`await abort_all_reqs()`** — 中止所有运行中和等待中的请求，释放 KV-cache
3. **HTTP 200 响应** — 只有步骤 2 完全完成后才发送

**步骤 2 的 `await` 语义确保 HTTP 响应是 KV-cache 释放完成的隐式确认**。

### 2.3 vLLM 请求门控

**源码**: `areal_vllm_server.py:244-270`

```python
async def _wait_if_paused():
    """Wait if generation is paused."""
    if not _generation_run_event.is_set():
        await _generation_run_event.wait()

@router.post("/v1/completions", ...)
async def create_completion(request, raw_request):
    await _wait_if_paused()  # ← 新请求在此被阻塞
    response = await original_create_completion(request, raw_request)
    return response
```

**门控时序**:
- `_generation_run_event.clear()` 在 `abort_all_reqs()` 之前（line 232 在 233 之前）
- 这确保**即使 abort 还在进行中，新请求也已被门控**
- 已经通过门控的请求可能在 abort 期间到达 EngineCore 的 ADD 队列——会被 abort 一并清理

---

## 3. vLLM abort_all_reqs 的完整内存清理流程

### 3.1 完整实现

**源码**: `areal_vllm_server.py:274-313`

```python
def abort_all_reqs(self):
    """Abort all running and waiting requests and clean up resources."""
    scheduler = self.scheduler

    # ① 收集所有活跃请求
    abort_lists = list(scheduler.running) + list(scheduler.waiting)

    if not abort_lists:
        # ⑤ 即使无请求也重置 prefix cache
        success = scheduler.reset_prefix_cache()
        if not success:
            raise RuntimeError("Prefix cache must be reset...")
        return

    # ② 为每个请求创建 ABORT 输出
    client_outputs = {}
    for req in abort_lists:
        engine_output = EngineCoreOutput(
            request_id=req.request_id,
            new_token_ids=[],
            finish_reason=FinishReason.ABORT,
            new_logprobs=None,
            new_prompt_logprobs_tensors=None,
            stop_reason=None,
        )
        if req.client_index not in client_outputs:
            client_outputs[req.client_index] = []
        client_outputs[req.client_index].append(engine_output)

    # ③ 通知 scheduler 完成这些请求（释放 KV-cache blocks）
    request_ids = [req.request_id for req in abort_lists]
    scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    # ④ 将 abort 通知发送给客户端
    for client_index, outputs in client_outputs.items():
        engine_core_outputs = EngineCoreOutputs(outputs=outputs)
        self.output_queue.put_nowait((client_index, engine_core_outputs))

    # ⑤ 重置 prefix cache（防止新请求复用旧权重的 KV 条目）
    success = scheduler.reset_prefix_cache()
    if not success:
        raise RuntimeError("Prefix cache must be reset to prevent kv cache pollution!")
```

### 3.2 各步骤的内存影响

```
步骤 ①: scheduler.running + scheduler.waiting
  → 枚举所有占用 KV-cache blocks 的请求
  → 内存状态: [KV blocks 被占用]

步骤 ③: scheduler.finish_requests(ids, FINISHED_ABORTED)
  → 内部调用 _free_request() 对每个请求:
    → kv_cache_manager.free(request)  ← 释放该请求的所有 KV blocks
    → 将 block 返回到 BlockPool 的空闲列表
  → 内存状态: [KV blocks 已释放到空闲池]

步骤 ⑤: scheduler.reset_prefix_cache()
  → 内部调用 BlockPool.reset_prefix_cache()
    → 检查: num_used_blocks == 1 (只有 null block)
    → 如果是: 清除所有 prefix hash 表
    → 如果不是: 返回 False → RuntimeError
  → 内存状态: [KV blocks 空闲 + prefix hash 清除]
```

### 3.3 KV-Cache Block 的释放路径

```
abort_all_reqs()
  └─ scheduler.finish_requests(request_ids, FINISHED_ABORTED)
      └─ for each request_id:
          └─ scheduler._free_request(request)
              ├─ kv_cache_manager.free(request)
              │   └─ block_pool.free_blocks(request.kv_block_hashes)
              │       └─ 将 block indices 返回到 free_blocks 列表
              │       └─ 减少 block 的 ref_count
              │       └─ ref_count == 0 时 block 真正空闲
              │
              ├─ 从 scheduler.running / scheduler.waiting 中移除
              └─ 添加到 scheduler.finished_req_ids
```

### 3.4 Prefix Cache 的清除机制

```
scheduler.reset_prefix_cache()
  └─ block_pool.reset_prefix_cache()
      ├─ 检查: num_gpu_blocks - get_num_free_blocks() == 1 (null block)
      │   → 如果有未释放的 block: 返回 False
      │   → RuntimeError("Prefix cache must be reset!")
      │
      ├─ 清除所有 hash → block 映射
      │   → 旧权重生成的 KV 条目的 hash 全部失效
      │
      └─ 此后新请求无法匹配任何 prefix cache 条目
         → 必须从头计算 KV
         → 确保使用新权重计算所有 KV
```

### 3.5 为什么必须重置 Prefix Cache

```
场景: 不重置 prefix cache 的风险

  权重更新前:
    Prompt "Hello" → KV-cache 条目 (hash=0x1234) → 用旧权重计算的 K/V 向量

  权重更新后:
    新请求 "Hello" → prefix cache 命中 hash=0x1234 → 复用旧权重的 K/V 向量！
    → Attention(Q_new, K_old, V_old) → 错误的注意力分数 → 生成质量劣化

  AReaL 的解决:
    reset_prefix_cache() → 清除所有 hash 映射
    → 新请求 "Hello" → cache miss → 用新权重从头计算 K/V → 正确
```

---

## 4. 权重更新期间的内存状态转换

### 4.1 完整的 GPU 内存时间线

```
时间 →

GPU 内存布局:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  训练前 (正常运行):                                                  │
│  [模型权重 W_old] [KV-cache blocks: ████████░░░░░░] [推理临时缓冲]   │
│                   ▲ 被各请求占用       ▲ 空闲                        │
│                                                                     │
│  ── pause_generation() ──                                           │
│                                                                     │
│  ① _generation_run_event.clear():                                   │
│  [模型权重 W_old] [KV-cache blocks: ████████░░░░░░] [推理临时缓冲]   │
│                   ▲ 仍然被占用（abort 未开始）                        │
│                                                                     │
│  ② abort_all_reqs():                                                │
│  [模型权重 W_old] [KV-cache blocks: ░░░░░░░░░░░░░░] [推理临时缓冲]   │
│                   ▲ 全部释放！                                       │
│                                                                     │
│  ③ reset_prefix_cache():                                            │
│  [模型权重 W_old] [KV-cache blocks: ░░░░░░░░░░░░░░] [prefix hash=∅] │
│                   ▲ 空闲            ▲ hash 表清空                    │
│                                                                     │
│  ── NCCL broadcast 权重 ──                                          │
│                                                                     │
│  ④ dist.broadcast(tensor, src=0):                                   │
│  [recv buffer] → model.load_weights() → [模型权重 W_new]             │
│  [KV-cache blocks: ░░░░░░░░░░░░░░]                                  │
│                                                                     │
│  ── continue_generation() ──                                        │
│                                                                     │
│  ⑤ _generation_run_event.set():                                     │
│  [模型权重 W_new] [KV-cache blocks: ████░░░░░░░░░░] [新请求开始]     │
│                   ▲ 新请求开始占用                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 权重加载的内存模式

**源码**: `vllm_worker_extension.py:131-144`

```python
for name, dtype, shape in zip(names, dtypes, shapes):
    tensor = torch.empty(shape, dtype=target_dtype, device=self.model_runner.device)
    # ↑ 临时接收缓冲区（峰值额外显存 = 单参数大小）
    torch.distributed.broadcast(tensor, src=0, group=group, async_op=False)
    self.model_runner.model.load_weights(weights=[(name, tensor)])
    # ↑ load_weights 内部调用 param.data.copy_(tensor) — 原地更新
    # 临时 tensor 在下次迭代时被 GC 回收
```

**内存特征**:
- **不分配新的模型参数**：`load_weights` 使用 `param.data.copy_()`，原地写入现有参数内存
- **临时缓冲区**：每个参数一个 `torch.empty`，用完即弃
- **峰值额外显存**：~单个最大参数大小（通常 <512 MB）
- **无内存碎片**：因为不涉及参数张量的重新分配

### 4.3 反复 abort/reload 的内存稳定性

```
Cycle 1: abort → free KV → reset prefix → broadcast W_v1 → resume → 分配 KV
Cycle 2: abort → free KV → reset prefix → broadcast W_v2 → resume → 分配 KV
...
Cycle N: abort → free KV → reset prefix → broadcast W_vN → resume → 分配 KV

每个 cycle:
  - 模型权重: 原地 copy_，内存地址不变
  - KV-cache blocks: 释放到空闲池 → 重新分配 → 释放到空闲池
  - Prefix cache hash: 清除 → 重建 → 清除
  - 临时接收缓冲: 分配 → 释放（CUDA caching allocator 复用）

→ 无累积的内存泄漏或碎片
```

---

## 5. 调度循环的暂停/恢复协议

### 5.1 暂停期间各数据结构的状态

| 数据结构 | 位置 | 暂停时状态 | 说明 |
|----------|------|-----------|------|
| `_pending_inputs` | deque | **冻结** | 生产者线程阻塞在 `_input_cv.wait()` |
| `runner.input_queue` | Queue (有界) | **冻结** | 已入队但未转为 asyncio task 的项 |
| `running_tasks` | asyncio dict | **继续完成** | 在运行的异步任务自然结束 |
| `runner.output_queue` | Queue (有界) | **继续累积** | 完成的结果进入输出队列 |
| `_pending_results` | dict | **继续累积** | 消费者线程持续收集结果 |

### 5.2 暂停的传播顺序

**Controller 模式** (`rollout_controller.py:1041-1047`):

```python
def pause(self):
    self.dispatcher.pause()              # ① 先暂停本地 dispatcher
    self._collective_rpc("pause", ...)   # ② 再暂停远程 worker

def resume(self):
    self._collective_rpc("resume", ...)  # ① 先恢复远程 worker
    self.dispatcher.resume()             # ② 再恢复本地 dispatcher
```

**顺序不对称是刻意设计**:
- Pause: 先停调度器 → 再停 worker（防止向已停 worker 提交任务）
- Resume: 先启 worker → 再启调度器（确保 worker 就绪后再开始提交）

### 5.3 恢复的级联唤醒

```python
# BatchTaskDispatcher.resume() (workflow_executor.py:495-502)
def resume(self):
    self.runner.resume()         # ① AsyncTaskRunner.paused.clear()
    with self._input_cv:
        self._input_cv.notify() # ② 唤醒生产者线程

# AsyncTaskRunner.resume() (async_task_runner.py:635-642)
def resume(self):
    self.paused.clear()          # ③ 清除暂停标志
    self._signal_new_input()     # ④ loop.call_soon_threadsafe(event.set)
                                 #    唤醒异步循环的 _wait_for_new_tasks()
```

**级联唤醒链**:
1. `paused.clear()` → 异步循环的 pause check (line 325) 通过
2. `_signal_new_input()` → uvloop 唤醒，开始 drain `input_queue`
3. `_input_cv.notify()` → 生产者线程唤醒，检查容量并开始提交

### 5.4 零任务丢失保证

暂停/恢复期间**不会丢失任何任务**：

```
暂停前:
  _pending_inputs: [T5, T6, T7]         ← 等待提交
  input_queue: [T3, T4]                 ← 等待转为 asyncio task
  running_tasks: {T1: task, T2: task}   ← 运行中

暂停后（一段时间）:
  _pending_inputs: [T5, T6, T7]         ← 不变（生产者阻塞）
  input_queue: [T3, T4]                 ← 不变（drain 被跳过）
  running_tasks: {}                     ← T1, T2 自然完成
  output_queue: [R1, R2]               ← T1, T2 的结果
  _pending_results: {1: R1, 2: R2}     ← 消费者线程持续收集

恢复后:
  → input_queue 中的 T3, T4 被 drain 为 asyncio task
  → _pending_inputs 中的 T5, T6, T7 被生产者取出提交
  → 所有任务最终完成，无丢失
```

---

## 6. vLLM vs SGLang 的 KV-Cache 管理差异

### 6.1 对比表

| 维度 | vLLM | SGLang |
|------|------|--------|
| **暂停端点** | `/areal_pause_generation` (AReaL 自定义) | `/pause_generation` (SGLang 原生) |
| **中止方式** | `abort_all_reqs()` 强制中止所有请求 | 依赖 SGLang 原生实现（版本相关） |
| **KV-Cache 释放** | 显式 `finish_requests(ABORTED)` | SGLang 内部处理 |
| **Prefix Cache** | 显式 `reset_prefix_cache()` + RuntimeError 保护 | SGLang 内部处理 |
| **LoRA XCCL 支持** | 完整 | 不支持（仅 disk） |
| **HTTP 响应语义** | `await abort` 后才响应 → **确认性** | 取决于实现 → **启发性** |
| **权重更新协议** | 两步：`set_weight_meta` + `update_weight_xccl` | 单步：`/update_weights_from_distributed` |
| **abort 标志** | 自定义 monkey-patch | 原生 `abort_all_requests=True` 参数 |

### 6.2 SGLang 的原生 abort 集成

**源码**: `sglang_remote.py:155-181`

```python
# SGLang 分布式权重更新请求
payload = {
    "names": [...],
    "dtypes": [...],
    "shapes": [...],
    "group_name": meta.nccl_group_name,
    "abort_all_requests": True,  # ← SGLang 原生支持
}
return WeightUpdateRequests(
    requests=[HttpRequest(endpoint="/update_weights_from_distributed", payload=payload)]
)
```

SGLang 在权重更新端点中**原生支持** `abort_all_requests` 参数，
将 abort 和权重更新合并为一步操作。AReaL 不需要为 SGLang 定制 abort 逻辑。

### 6.3 为什么 vLLM 需要 Monkey-Patch

vLLM 没有原生的"pause + abort all + reset prefix cache"原子操作。
AReaL 通过 `hook()` 函数（`areal_vllm_server.py:355-386`）将 `abort_all_reqs`
注入到 `EngineCore` 类上：

```python
def hook():
    setattr(EngineCore, "abort_all_reqs", abort_all_reqs)
    setattr(EngineCore, "areal_injected_update_weight", ...)
    setattr(EngineCore, "areal_injected_update_weight_xccl", ...)
    # ... 更多 monkey-patch
```

**版本耦合风险**: 这些 patch 依赖 vLLM v1 内部 API（`EngineCore`, `scheduler.running`,
`scheduler.waiting`, `scheduler.finish_requests`, `scheduler.reset_prefix_cache`）。
vLLM 内部 API 不稳定，升级 vLLM 版本可能需要更新这些 patch。

---

## 7. 代码质量发现

### Medium 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 1 | `areal_vllm_server.py` | 309-313 | `reset_prefix_cache()` 在 KV Connector 延迟释放 blocks 时会失败并抛出 RuntimeError。标准单引擎配置不受影响，但 disaggregated prefill 配置可能触发 |
| 2 | `vllm_worker_extension.py` | 31 | `model_config.model = model_path` 在 disk 更新失败时不回滚，留下不一致状态 |
| 3 | `vllm_worker_extension.py` | 131-144 | 每参数串行 broadcast + load_weights，无流水线重叠。大模型场景下显著慢于批量方式 |

### Low 级别

| # | 文件 | 行号 | 问题 |
|---|------|------|------|
| 4 | `areal_vllm_server.py` | 232-233 | `_generation_run_event.clear()` 与 `abort_all_reqs()` 之间存在竞态窗口——已通过门控的请求可能被 abort 中止。功能安全（客户端重试处理），但浪费计算 |
| 5 | `areal_vllm_server.py` | 38-39 | `asyncio.Event` 非线程安全，但在 uvloop 单循环模型下安全 |
| 6 | `async_task_runner.py` | 325-327 | 暂停期间使用 500ms 轮询而非事件驱动唤醒 |
| 7 | `vllm_worker_extension.py` | 75-87 | `set_weight_meta` 两步协议通过实例属性传递状态，无序列号保护 |

### Positive 发现

| # | 文件 | 行号 | 亮点 |
|---|------|------|------|
| 8 | `areal_vllm_server.py` | 279-285 | 即使无请求也重置 prefix cache——处理了自然完成请求留下的陈旧 prefix 条目 |
| 9 | `areal_vllm_server.py` | 274-313 | `abort_all_reqs` 在 EngineCore 的单线程忙循环中执行——与 scheduler step 天然互斥，无并发风险 |
| 10 | `vllm_worker_extension.py` | 131-144 | `load_weights` 使用 `param.data.copy_()` 原地更新——无内存碎片，无参数重分配 |
| 11 | `workflow_executor.py` | 418-420 | 消费者线程在收集结果后通知生产者——闭合反压反馈循环 |

---

## 8. 设计总结

### KV-Cache 排空机制

> vLLM 后端通过 **`abort_all_reqs()` 三步清理** 实现硬排空:
> 1. `scheduler.finish_requests(ABORTED)` → 释放所有 KV-cache blocks 到空闲池
> 2. `output_queue.put_nowait(ABORT)` → 通知客户端请求已中止
> 3. `scheduler.reset_prefix_cache()` → 清除所有 prefix hash 映射
>
> **HTTP 200 响应只在步骤 1-3 全部完成后才返回**——这是训练侧确认 KV-cache
> 已完全释放的唯一机制，无需轮询或额外的 readiness check。
>
> `reset_prefix_cache()` 是**防止 KV 污染的最后防线**：清除所有旧权重生成的
> prefix hash，确保新请求必须用新权重从头计算 KV 向量。即使无请求被中止，
> 也会执行此步骤。

### 调度循环设计

> 四阶段流水线（pending → capacity gate → asyncio tasks → results）通过
> **Condition Variable + asyncio Event + 有界 Queue** 避免忙等待。
> 暂停只影响生产者线程和 asyncio drain——消费者线程持续收集结果，
> 在运行的异步任务自然完成。**零任务丢失**保证通过冻结各队列状态实现。
>
> 反馈循环由消费者线程的 `_input_cv.notify()` 闭合——rollout 完成后
> staleness 容量恢复，生产者被唤醒继续提交。

### 内存稳定性

> 反复 abort/reload 循环**不会导致 GPU 内存碎片或泄漏**:
> - 模型权重通过 `param.data.copy_()` 原地更新，内存地址不变
> - KV-cache blocks 在 BlockPool 的空闲列表中循环分配/释放
> - CUDA caching allocator 复用临时接收缓冲区
> - Prefix hash 表每次清除后重建
