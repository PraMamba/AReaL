# 通用工具库

> 源码位置：`areal/utils/` 文件数：35 个 | 总行数：10728 行 最后更新：2026-06-13

## 1. 模块职责概述

`areal/utils/` 是 AReaL 框架的横切关注点（cross-cutting concerns）工具库，为上层的
Workflow、Engine、Dataset、Reward 等核心模块提供基础能力支撑。该模块不包含任何业务逻辑，而是提供以下六大类基础服务：

1. **数据处理**：张量批处理、打包/解包、微批次切分、序列对齐（`data.py`、`seqpack.py`、`dataloader.py`）
1. **RL 数学函数**：PPO/SAPO/DPO 损失函数、KL 散度估计、归一化、词表并行 logprobs（`functional/`）
1. **性能追踪**：Chrome Trace 格式事件记录、会话生命周期跟踪、PyTorch Profiler 集成（`perf_tracer.py`）
1. **分布式协调**：名称解析服务、分布式锁、分布式统计聚合（`name_resolve.py`、`lock.py`、`stats_tracker.py`）
1. **训练生命周期**：检查点保存/恢复、评估调度、统计日志、训练恢复（`saver.py`、`recover.py`、`evaluator.py`、`stats_logger.py`）
1. **基础工具**：日志系统、随机种子、时间调度、网络、文件系统、版本检查等

整体设计遵循"零业务耦合"原则——任何文件都不依赖具体的 Workflow 或 Engine 实现，仅通过 `areal.api` 中的抽象接口交互。

## 2. 文件清单

以下行数均来自 `wc -l` 实际输出：

| 文件                           | 行数 | 职责简述                                                          |
| ------------------------------ | ---- | ----------------------------------------------------------------- |
| `__init__.py`                  | 1    | 包标记（空文件）                                                  |
| `async_checkpoint.py`          | 210  | 异步 DCP 检查点管理器（同步/异步双模式）                          |
| `constants.py`                 | 87   | 全局常量：分布式超时、内存对齐、近端 logp 枚举                    |
| `data.py`                      | 1689 | **核心**：张量批处理、打包/解包、微批次切分、归一化、KL 估计      |
| `dataloader.py`                | 113  | StatefulDataLoader 工厂、评估分布式采样器                         |
| `dynamic_import.py`            | 37   | 从字符串路径动态导入 Python 对象                                  |
| `environ.py`                   | 34   | 环境变量布尔解析、CI/SPMD 模式检测                                |
| `errors.py`                    | 21   | 自定义异常类（EngineError、FrameworkError）                       |
| `evaluator.py`                 | 38   | 按频率触发评估回调的调度器                                        |
| `fs.py`                        | 112  | 文件系统工具：网络文件系统检测、共享路径验证                      |
| `functional/__init__.py`       | 33   | functional 子包公开接口汇总                                       |
| `functional/functional.py`     | 804  | **核心**：PPO/SAPO/DPO 损失函数、拒绝采样、Critic 损失            |
| `functional/vocab_parallel.py` | 469  | **核心**：词表并行 log-probabilities 和 entropy 的自定义 autograd |
| `hf_utils.py`                  | 143  | HuggingFace tokenizer/processor 加载、聊天模板、文件下载          |
| `image.py`                     | 42   | 图像 Base64 编码、批量尺寸对齐填充                                |
| `lock.py`                      | 102  | 基于 torch.distributed Store 的分布式互斥锁                       |
| `logging.py`                   | 560  | **核心**：按组件分色的日志系统、文件日志、第三方日志抑制          |
| `math.py`                      | 13   | ceil_div、align 等基础数学函数                                    |
| `name_resolve.py`              | 1263 | **核心**：分布式名称解析（Memory/NFS/etcd3/Ray 四种后端）         |
| `names.py`                     | 34   | 标准化名称解析路径格式                                            |
| `network.py`                   | 206  | IP 地址探测、端口扫描、host:port 解析（IPv4/IPv6）                |
| `offload.py`                   | 51   | torch_memory_saver (TMS) 显存卸载环境配置                         |
| `perf_tracer.py`               | 2125 | **核心**：Chrome Trace 性能追踪器 + 会话生命周期追踪器            |
| `pkg_version.py`               | 67   | Python 包版本比较工具                                             |
| `printing.py`                  | 21   | 统计数据表格化输出                                                |
| `recover.py`                   | 445  | 训练恢复：RecoverInfo 序列化/反序列化 + RecoverHandler            |
| `save_load.py`                 | 97   | HF/safetensors 模型权重加载（多线程并发）                         |
| `saver.py`                     | 191  | 模型检查点定期保存调度器（含异步模式）                            |
| `seeding.py`                   | 61   | 全局随机种子管理、确定性 shuffle                                  |
| `seqpack.py`                   | 615  | **核心**：序列打包算法（FFD + Karmarkar-Karp）                    |
| `stats_logger.py`              | 181  | 训练指标多后端日志（wandb/swanlab/trackio/tensorboard）           |
| `stats_tracker.py`             | 332  | 分布式统计追踪器：分层作用域、多种归约模式                        |
| `testing_utils.py`             | 197  | 测试工具：模型/数据集下载、Archon 模型加载、TestWorkflow          |
| `timeutil.py`                  | 285  | 频率控制器、值调度器（线性/余弦/指数/链式）                       |
| `wrapper.py`                   | 49   | 可组合方法包装器（wrapable/wrap 装饰器模式）                      |

## 3. 核心数据结构与接口

### 3.1 数据处理层 (`data.py`)

```
+----------------------------+
| TrajBatchMeta              |  批次元数据（序列数、分组大小、长度）
+----------------------------+
| MicroBatchItem (NamedTuple)|  单个微批次项（原始/填充/对齐信息）
+----------------------------+
| MicroBatchList (dataclass) |  微批次列表容器（含前向/后向索引映射）
+----------------------------+
| Normalization              |  自适应归一化（batch/group 级别 mean/std）
+----------------------------+
| KLEstimator                |  KL 散度近似估计器（k1/k2/k3 三种方法）
+----------------------------+
```

关键公开接口链：

```
collate_samples_to_list() --> concat_batch() --> split_padded_tensor_dict_into_mb_list()
                                                          |
                                                   pad_mb_list() --> MicroBatchList.__iter__()
                                                          |
                                              (训练前向) --> unpad_logits()
                                                          |
                                              split_batch() / split_and_unpad_tensor()
```

### 3.2 RL 损失函数层 (`functional/`)

```
+-------------------------------+
| functional.py                 |
|   ppo_actor_loss_fn()         |  PPO Actor 损失（含拒绝采样、GSPO）
|   sapo_loss_fn()              |  SAPO 损失（sigmoid 门控替代 clipping）
|   ppo_critic_loss_fn()        |  PPO Critic 价值损失（MSE/Huber）
|   dpo_pair_logratios()        |  DPO 成对 log-ratio 聚合
|   dpo_preference_loss()       |  DPO 偏好损失（sigmoid/IPO）
|   apply_rejection_sampling()  |  拒绝采样（mask/clamp 两种模式）
|   masked_normalization()      |  分布式掩码归一化
+-------------------------------+
| vocab_parallel.py             |
|   gather_logprobs()           |  词表并行 log-probabilities
|   gather_logprobs_entropy()   |  词表并行 logprobs + entropy（共享中间量）
+-------------------------------+
```

### 3.3 性能追踪层 (`perf_tracer.py`)

```
+-----------------------+         +---------------------+
| PerfTracer            |-------->| SessionTracer       |
|   trace_scope()       |  1:0..1 |   register_task()   |
|   atrace_scope()      |         |   register_session()|
|   instant()           |         |   record_event()    |
|   merge_profiler_trace|         |   flush()           |
|   save()              |         +---------------------+
+-----------------------+                  |
       |                           +------------------+
  GLOBAL_TRACER (singleton)        | SessionRecord    |
       |                           |   phases: dict   |
  模块级便利函数:                    |   counters: dict |
  trace_scope(), instant(),        |   to_dict()      |
  save(), configure(), reset()     +------------------+
```

### 3.4 分布式名称解析 (`name_resolve.py`)

```
NameRecordRepository (抽象基类)
    |
    +-- MemoryNameRecordRepository    (内存，测试用)
    |
    +-- NfsNameRecordRepository       (NFS 文件系统)
    |
    +-- Etcd3NameRecordRepository     (etcd3，含 TTL 保活)
    |
    +-- RayNameResolveRepository      (Ray Actor，分布式 KV)
              |
              +-- DistributedKVStore  (Ray remote Actor)

DEFAULT_REPOSITORY ---> 模块级函数: add/get/delete/wait/watch_names
reconfigure(config)  -> 运行时切换后端
```

### 3.5 序列打包算法 (`seqpack.py`)

```
get_allocate_fn(algorithm) --> ffd_allocate() 或 kk_allocate()

ffd_allocate: First Fit Decreasing 贪心算法
kk_allocate:  Karmarkar-Karp 最大差分法（更均衡）

辅助:
  partition_balanced()            Numba JIT 动态规划均衡分区
  reorder_to_balanced_batches()   批次重排序
  balanced_greedy_partition()     贪心等分
```

## 4. 算法与逻辑详解

### 4.1 data.py 的批处理/打包逻辑

该文件（1689 行）实现了 AReaL 训练数据在"轨迹列表"与"单一批次"之间的完整转换流水线。

**核心数据表示**：系统内部存在两种张量布局：

- **Padded 格式**：`[batch, seq_len]`，用 `attention_mask` 标记有效位置。适合理解和调试。
- **Packed 格式**：`[total_tokens]`，用 `cu_seqlens`（累积序列长度）标记边界。适合高效计算。

**关键转换链**（行 369-430）：

1. `concat_batch(data: list[dict]) -> (dict, TrajBatchMeta)`：将多条轨迹拼接为一个批次。调用
   `concat_padded_tensors` 执行异维度填充+dim0 拼接。返回元数据 `TrajBatchMeta`
   记录每条轨迹的分组大小和序列长度，供后续反向拆分使用。

1. `split_batch(result, meta) -> list`：`concat_batch` 的逆操作。根据元数据将批次结果拆回每条轨迹。自动从
   `attention_mask` 推导每组的真实序列长度，去除填充。

1. `batched_call(fn, data, unpack=True)`：便利函数——先合并、调用 `fn`、再拆分。这是 Trainer 调用 Engine
   的标准桥接模式。

**微批次切分**（行 696-813）：

`split_padded_tensor_dict_into_mb_list` 是微批次管理的入口：

- 首先调用 `allocate_balanced_mbs_synced` 使用 FFD/KK 算法将序列分配到微批次（确保跨 rank 同步）
- 然后按分配结果重排序列，构建 `forward_indices` 和 `backward_indices` 映射
- 返回 `MicroBatchList`，其 `__iter__` 方法逐个 yield `MicroBatchItem`

**序列对齐与填充**（行 818-979）：

`pad_packed_tensor_dict` 执行两级对齐：

- **序列级**（`seq_align_to`）：将每条序列对齐到指定倍数（用于上下文并行/序列并行的均匀切分）
- **批次级**（`pad_to_length`）：将总长度对齐到 `N_TOKENS_PER_PAGE`（256）的倍数，减少 GPU 内存碎片

**自适应归一化**（行 1378-1596）：

`Normalization` 类实现了灵活的 advantage 归一化，支持：

- `mean_level`/`std_level` 分别控制均值和标准差的计算粒度（batch/group/none）
- `mean_leave1out`：leave-one-out 均值估计（每个样本排除自身计算均值）
- `group_size`：组内归一化的组大小
- 在分布式场景下通过 `all_reduce` 跨 rank 同步统计量

### 4.2 perf_tracer.py 的追踪系统

该文件（2125 行）实现了双轨追踪系统：

**轨道一：性能事件追踪（PerfTracer）**

核心是 Chrome Trace Format 的事件记录器：

- `trace_scope()` 返回上下文管理器 `_Scope`，记录 "X"（Complete）类型事件，包含名称、类别、时间戳、持续时间
- `instant()` 记录 "i"（Instant）类型事件
- 使用 `time.perf_counter_ns()` 高精度计时，所有时间戳相对于 `_origin_ns` 基准偏移
- 线程/进程元数据自动发射（`thread_name`、`process_name`、`process_sort_index`）
- `merge_profiler_trace()`（行 1451-1603）：将 PyTorch Profiler 的 Chrome Trace JSON
  合并到主事件流中，通过虚拟 TID 映射避免线程 ID 冲突

**轨道二：会话生命周期追踪（SessionTracer）**

- 基于声明式的事件规则系统：`PhaseSpec` 定义阶段配置，`EventBinding` 定义事件到状态更新的映射
- `SessionRecord` 是核心数据结构，通过 `ClassVar`（`PHASE_CONFIGS`、`FIELD_SPECS`）配置阶段跟踪和序列化规则
- 支持 generate/reward/toolcall 三种内置阶段，每种阶段允许多次执行
- 自动计算派生指标：`total_s`、`generate_s`、`reward_s`、`toolcall_s`
- 达到 `flush_threshold` 阈值时自动批量写入 JSONL 文件

**全局单例管理**：

```
configure(config, rank, role) -> GLOBAL_TRACER  (进程唯一)
                                    |
                               session_tracer  (可选启用)
```

通过 `ContextVar` 在异步上下文中传播 `task_id`、`session_id`、`global_step`，支持装饰器
`@trace_perf`、`@trace_session`、`@session_context` 自动注入追踪信息。

### 4.3 name_resolve.py 的分布式名称解析

该文件（1263 行）实现了一个分布式键值存储抽象，语义上等价于 `Dict[str, Set[str]]` 的多映射（multimap）。

**抽象接口**（`NameRecordRepository`，行 42-182）：

- `add(name, value, delete_on_exit, keepalive_ttl, replace)`：添加键值对
- `get(name)` / `get_subtree(name_root)`：点查/前缀查
- `wait(name, timeout, poll_frequency)`：阻塞等待键出现
- `watch_names(names, callback)`：监控键删除事件

**四种后端实现**：

| 后端  | 类                           | 行数范围 | 存储介质    | 适用场景             |
| ----- | ---------------------------- | -------- | ----------- | -------------------- |
| 内存  | `MemoryNameRecordRepository` | 184-282  | 进程内 dict | 单进程测试           |
| NFS   | `NfsNameRecordRepository`    | 284-410  | 文件系统    | 共享存储集群（默认） |
| etcd3 | `Etcd3NameRecordRepository`  | 412-777  | etcd3 服务  | 高可用生产环境       |
| Ray   | `RayNameResolveRepository`   | 883-1208 | Ray Actor   | Ray 集群             |

**NFS 后端的特殊处理**：

- 使用原子性的 `write-to-tmp + rename` 模式避免并发写入冲突
- 处理 NFS Stale file handle (errno 116) 的重试逻辑
- `clear_subtree` 使用 `shutil.rmtree` 并逐级清理空目录

**etcd3/Ray 后端的保活机制**：

- TTL-based lease：创建条目时关联一个有过期时间的 lease
- 后台守护线程以 `keepalive_ttl/3` 的频率刷新 lease
- 进程崩溃时 lease 自动过期，键值被清理

**模块级 facade**（行 1227-1263）： 模块导出 `add`、`get`、`delete`、`wait` 等函数，默认指向
`NfsNameRecordRepository` 实例。调用 `reconfigure(config)` 可在运行时切换到 etcd3 或 Ray 后端。

### 4.4 seqpack.py 的序列打包算法

该文件（615 行）提供两种将不等长序列分配到固定容量组的算法：

**FFD（First Fit Decreasing）**（行 196-255）：

- 按序列长度降序排序
- 对每个序列，在容量充足的组中选择当前总量最小的组放入
- 使用 `bisect` 维护有序组列表，O(n log n) 复杂度
- 循环增加 `min_groups` 直到组数满足 `n_groups_divisor` 整除约束

**KK（Karmarkar-Karp 最大差分法）**（行 260-537）：

核心数据结构：

- `_KKSet`：一组元素及其累积和
- `_KKState`：一个 k 路分区方案，包含 k 个 `_KKSet`

算法流程：

1. 每个元素初始化为一个独立的 `_KKState`
1. 用最大堆按 spread（最大组和 - 最小组和）排序
1. 每次取出 spread 最大的两个 state，将第一个的最大组与第二个的最小组配对合并
1. 重复直到只剩一个 state

KK 在序列长度差异大时（如 RL rollout）产生更均衡的分区，但有容量违约风险时回退到 FFD。

**辅助函数**（行 22-123）：

- `partition_balanced`：基于 Numba JIT 的动态规划精确均衡分区，O(n^2 * k) 复杂度
- `reorder_to_balanced_batches`：按序列数量约束的贪心重排序

### 4.5 functional/ 的 RL 损失函数

**PPO Actor 损失**（`ppo_actor_loss_fn`，行 429-559）：

支持标准 PPO 和多种扩展：

- **重要性采样级别**：token 级（标准 PPO）或 sequence 级（GSPO），后者通过
  `_compute_sequence_level_ratio_and_advantages` 计算序列几何均值比率
- **解耦剪切**：`eps_clip_higher` 允许上下限不对称
- **双重剪切**：`c_clip` 参数实现 Dual Clipping PPO
- **拒绝采样**：通过 `apply_rejection_sampling` 在损失计算前过滤或截断偏离过大的行为策略样本

**SAPO 损失**（`sapo_loss_fn`，行 562-638）： 用不对称 sigmoid 门控替代 PPO 的硬 clipping，提供平滑梯度：

```
gate_pos = sigmoid(tau_pos * (ratio - 1)) * (4 / tau_pos)
gate_neg = sigmoid(tau_neg * (ratio - 1)) * (4 / tau_neg)
```

**词表并行 logprobs**（`vocab_parallel.py`）：

`_VocabParallelLogProbs`（行 84-194）和 `_VocabParallelLogProbsEntropy`（行 197-372）是两个自定义
`torch.autograd.Function`：

- 在前向中跨 TP rank all-reduce max/sum，避免全量 vocab gather
- 反向利用 in-place 操作复用 softmax 张量，减少约 50% 的内存占用
- 使用 `_chunked_apply` 按 1024 token 分块处理，控制峰值显存

## 5. 数据流（输入/输出）

### 5.1 训练数据流

```
Dataset 输出: list[dict[str, Tensor]]
       |
       v
collate_samples_to_list()          每个样本转为 [1, seqlen] 的字典
       |
       v
concat_batch(data)                 合并为单个 batched dict + TrajBatchMeta
       |
       v
pack_tensor_dict(data)             Padded [B,S] --> Packed [total_len] + cu_seqlens
       |
       v
split_padded_tensor_dict_into_mb_list()  按 FFD/KK 切分微批次
       |
       v
pad_mb_list()                      对齐 + 填充 --> MicroBatchList.padded_mbs
       |
       v
for mb_item in mb_list:            迭代每个 MicroBatchItem
    model.forward(mb_item.padded_mb)
    unpad_logits(...)              去填充恢复原始长度
       |
       v
split_batch(result, meta)          拆回每条轨迹的结果
```

### 5.2 性能追踪数据流

```
用户代码:
  with trace_scope("train_step"):
      ...
       |
       v
PerfTracer._record_complete()     生成 Chrome Trace 事件 dict
       |
       v
PerfTracer._events (内存缓冲)
       |
       v (按 save_interval 周期)
PerfTracer.save()                  追加写入 JSONL 文件
       |
       v
traces-r{rank}.jsonl               可用 Perfetto/chrome://tracing 查看

SessionTracer:
  trace_session_event("mark_generate_start")
       |
       v
SessionRecord.apply_phase_event()  更新阶段时间戳
       |
       v (达到 flush_threshold)
SessionTracer.flush()              追加写入 sessions-r{rank}.jsonl
```

### 5.3 名称解析数据流

```
TrainController:
  name_resolve.add("exp/trial/gen_servers", "host:port")
       |
       v
NfsNameRecordRepository:
  write /record_root/exp/trial/gen_servers/ENTRY
       |
       v (其他 rank/进程)
  name_resolve.wait("exp/trial/gen_servers")
       |
       v
  读取 ENTRY 文件内容 -> "host:port"
```

## 6. 关键设计决策与不变量

### 6.1 Padded/Packed 双表示

**决策**：系统内部同时维护 Padded（`attention_mask`）和 Packed（`cu_seqlens`）两种张量布局。

**理由**：Padded 格式适合按 batch 维度的操作（如分布式 all-gather），Packed 格式适合变长序列的高效 GPU 计算（如 Flash
Attention）。`pack_tensor_dict` 和 `unpad_logits` 是两种格式的桥接函数。

**不变量**：

- Packed 张量的 `cu_seqlens` 始终是单调递增的 int32 张量，长度为 batch_size + 1
- `cu_seqlens[0] == 0`，`cu_seqlens[-1] == total_tokens`
- `attention_mask` 为 bool 类型，shape 始终是 `[batch, seq_len]`

### 6.2 微批次切分的跨 rank 同步

**决策**：`allocate_balanced_mbs_synced` 在切分后通过 `all_gather_object` 同步所有 rank
的微批次数量，取最大值后重新切分。

**理由**：FSDP 训练要求所有 rank 执行相同数量的前向/后向步骤。不同 rank 上的序列长度分布可能不同，导致 FFD/KK
产生不同数量的微批次。同步确保一致性。

### 6.3 词表并行的 in-place 梯度优化

**决策**：`_VocabParallelLogProbs.backward` 直接在保存的 softmax 张量上原地修改计算梯度。

**理由**：节省约 2.3GB 显存（典型配置 seq=8192, vocab=152K, tp=2）。代价是不支持 `retain_graph=True` 和高阶梯度，但
RL 训练只需一阶梯度。

**不变量**：

- `_VocabParallelLogProbsEntropy` 不能使用 in-place（需多次读取 softmax），因此额外分配一个 grad_input 张量
- 两个 Function 都不支持 `retain_graph=True`

### 6.4 PerfTracer 全局单例 + ContextVar

**决策**：使用全局 `GLOBAL_TRACER` 单例 + `ContextVar` 传播 session/task ID。

**理由**：

- 全局单例避免将 tracer 实例层层传递到每个函数
- `ContextVar` 支持在异步上下文（asyncio）中隔离 session 状态
- `_NullContext` 零开销回退：未配置时所有 trace 调用返回空上下文管理器

### 6.5 名称解析的模块级 facade

**决策**：`name_resolve.py` 在模块级直接导出 `add`/`get`/`delete` 等函数，绑定到
`DEFAULT_REPOSITORY`。运行时通过 `reconfigure()` 切换后端。

**理由**：简化调用方代码——`name_resolve.add(...)` 比 `name_resolve.DEFAULT_REPOSITORY.add(...)`
更简洁。`reconfigure()` 使用全局变量重绑定，保证切换后所有调用方自动使用新后端。

### 6.6 N_TOKENS_PER_PAGE 对齐

**决策**：微批次填充到 256 token 对齐（`N_TOKENS_PER_PAGE = 256`，行 815）。

**理由**：GPU 页大小为 2MB，以 bf16 dtype 和 hidden_size=4096 为例，一页正好容纳 256 个 token 的一维数据。对齐到页边界减少
CUDA 内存分配器碎片。

## 7. 已知问题与限制

1. **data.py 行 711 TODO**：`split_padded_tensor_dict_into_mb_list`
   中注释指出应先对齐序列再切分，目前顺序相反，可能在某些边界情况下导致次优的填充量。

1. **NFS 后端的 Stale file handle**：`NfsNameRecordRepository.get()` 需要最多 100 次重试来处理 NFS 的
   errno 116（行 356-366），每次等待 5ms。在高并发场景下可能影响延迟。

1. **vocab_parallel in-place 限制**：不支持 `retain_graph=True` 和高阶梯度。如果未来需要在 PPO
   中使用二阶优化方法，需要重写 backward。

1. **PerfTracer 保存策略**：事件先缓存在内存列表中，仅在 `save()` 调用时写入磁盘。如果进程异常退出且 `_save_at_exit` atexit
   回调未触发，事件可能丢失。

1. **stats_tracker 键冲突**：`export_all()` 中多个 tracker 可能产生重复键（行 330-331），仅打印警告而非报错。

1. **etcd3 依赖可选**：etcd3 模块在导入时 try-except 捕获（行 17-19），如果 etcd3 未安装但配置了 etcd3 后端，会在运行时抛出
   `NameError`。

1. **KK 算法容量违约回退**：`kk_allocate` 的分区可能违反容量约束（行 526-536），此时静默回退到
   FFD。这种回退行为可能在性能敏感场景中引入不可预期的分区质量退化。

## 8. 相关测试覆盖

以下测试文件直接针对 `areal/utils/` 中的功能：

| 测试文件                                     | 覆盖目标                                               |
| -------------------------------------------- | ------------------------------------------------------ |
| `tests/test_functional.py`                   | PPO/SAPO/DPO 损失函数、masked_normalization            |
| `tests/test_rejection_sampling.py`           | apply_rejection_sampling（mask/clamp，token/sequence） |
| `tests/test_vocab_parallel.py`               | vocab_parallel logprobs/entropy 正确性                 |
| `tests/torchrun/run_vocab_parallel.py`       | 词表并行分布式正确性                                   |
| `tests/test_perf_tracer.py`                  | PerfTracer 事件记录、保存、session 追踪                |
| `tests/torchrun/run_perf_tracer.py`          | 分布式性能追踪                                         |
| `tests/test_seqpack.py`                      | FFD 打包算法                                           |
| `tests/test_kk_allocate.py`                  | KK 打包算法                                            |
| `tests/torchrun/run_kk_vs_ffd.py`            | KK vs FFD 分布式比较                                   |
| `tests/test_lock.py`                         | DistributedLock 单元测试                               |
| `tests/torchrun/run_lock.py`                 | 分布式锁多进程测试                                     |
| `tests/test_utils.py`                        | data.py 中的批处理/切分/打包函数                       |
| `tests/test_packed_vs_padded_consistency.py` | Packed 与 Padded 格式一致性                            |
| `tests/test_dynamic_import.py`               | import_from_string 正确性                              |
| `tests/test_wrapper.py`                      | wrapable/wrap 装饰器                                   |
| `tests/test_network_ipv6_utils.py`           | IPv6 地址解析、host:port 分割                          |
| `tests/test_offload.py`                      | TMS 显存卸载                                           |
| `tests/test_recover.py`                      | RecoverInfo 序列化/反序列化、RecoverHandler            |
| `tests/test_adv_norm_config.py`              | Normalization 类（batch/group/leave-one-out）          |
| `tests/test_ppo_stats.py`                    | PPO 统计量计算                                         |
| `tests/test_prox_approx.py`                  | 近端 log-probability 近似方法                          |

测试覆盖重点集中在数学正确性（损失函数、归一化、KL 估计）和分布式一致性（跨 rank 同步、vocab 并行）两个方面。`torchrun/` 目录下的测试需要多 GPU
环境执行。
