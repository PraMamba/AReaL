# API 契约层

> 源码位置：`areal/api/` 文件数：8 个 | 总行数：6343 行 最后更新：2026-06-13

## 1. 模块职责概述

`areal/api/` 是 AReaL 项目的**契约层**（Contract Layer），定义了整个分布式 RL 训练框架中所有核心组件之间的接口协议、配置数据结构和
I/O 消息格式。该模块本身不包含任何业务逻辑的具体实现，而是通过抽象基类（ABC）、dataclass
和类型别名，建立起训练引擎、推理引擎、调度器、工作流和奖励函数之间的**统一通信契约**。

在整体架构中，`areal/api/` 处于**中枢位置**：

- **上游消费者**：`examples/` 目录中的训练脚本和 YAML 配置文件通过 Hydra/OmegaConf 实例化 `cli_args.py` 中定义的配置
  dataclass。
- **下游实现者**：`areal/engine/`（FSDPEngine、MegatronEngine、ArchonEngine、SGLang/vLLM
  远程引擎）、`areal/workflow/`（RLVRWorkflow、MultiTurnWorkflow
  等）、`areal/infra/`（调度器、RPC）分别实现该模块定义的抽象接口。
- **横向依赖**：`areal/reward/` 中的奖励函数通过 `AsyncRewardWrapper` 获得异步执行能力；`areal/utils/`
  提供日志、序列打包等底层支持。

核心设计原则：

1. **接口与实现分离** -- 所有引擎、调度器、工作流的公共 API 在此定义，具体实现在各自模块中完成。
1. **配置即文档** -- 通过 dataclass + metadata 的方式，每个配置字段都携带 `help` 说明和 `choices` 约束，可直接生成 CLI
   文档。
1. **惰性导入** -- `__init__.py` 使用 `__getattr__` 懒加载机制，避免在 import 时引入不必要的重量级依赖（如
   PyTorch、Transformers）。

## 2. 文件清单

| 文件               | 行数 | 职责                                                                                  |
| ------------------ | ---- | ------------------------------------------------------------------------------------- |
| `__init__.py`      | 68   | 包入口，定义 `__all__` 导出列表和惰性导入映射                                         |
| `alloc_mode.py`    | 1176 | GPU 资源分配语法解析：并行策略（5D）、分配模式、Lark 语法解析器                       |
| `cli_args.py`      | 3047 | 全局配置 dataclass 体系（50+ 个 dataclass），CLI 参数解析，Hydra 集成                 |
| `engine_api.py`    | 1008 | 训练引擎（`TrainEngine`）和推理引擎（`InferenceEngine`）的抽象基类                    |
| `io_struct.py`     | 417  | I/O 数据结构：请求/响应、权重更新元数据、检查点元数据、运行时信息                     |
| `reward_api.py`    | 183  | 奖励函数接口和 `AsyncRewardWrapper` 异步执行包装器                                    |
| `scheduler_api.py` | 329  | 分布式调度器抽象基类：Worker/Job 管理、引擎创建与 RPC 调用                            |
| `workflow_api.py`  | 115  | 工作流抽象基类：`RolloutWorkflow`、`AgentWorkflow`（已弃用）、`WorkflowLike` 类型别名 |

## 3. 核心数据结构与接口

### 3.1 `__init__.py` -- 惰性导入入口（第 1-68 行）

通过 `__all__` 列出 27 个公共导出符号，并使用 `_LAZY_IMPORTS` 字典 + `__getattr__` 实现按需导入：

```python
__all__ = [
    "RolloutWorkflow", "AsyncRewardWrapper",
    "TrainEngine", "InferenceEngine",
    "Scheduler", "Worker", "Job",
    "AllocationType", "ModelAllocation", "ParallelStrategy",
    "FSDPParallelStrategy", "MegatronParallelStrategy",
    "ModelRequest", "ModelResponse", "WeightUpdateMeta",
    "SaveLoadMeta", "StepInfo", "FinetuneSpec", "ParamSpec",
    "RolloutStat", "LocalInfServerInfo",
    "WorkflowLike", "AgentWorkflow",
]
```

### 3.2 `alloc_mode.py` -- 资源分配与并行策略（第 1-1176 行）

#### 枚举与异常

| 名称                         | 类型        | 位置     | 说明                                                                          |
| ---------------------------- | ----------- | -------- | ----------------------------------------------------------------------------- |
| `AllocationType`             | `enum.Enum` | 第 16 行 | 向后兼容的分配类型：`COLOCATE`(0)、`DECOUPLED_TRAIN`(1)、`LLM_SERVER_ONLY`(2) |
| `AllocationValidationError`  | `Exception` | 第 24 行 | 分配模式验证失败异常                                                          |
| `InvalidAllocationModeError` | `Exception` | 第 28 行 | 遗留异常，向后兼容                                                            |

#### 核心 dataclass

| 类名                       | 基类               | 位置      | 说明                                                                            |
| -------------------------- | ------------------ | --------- | ------------------------------------------------------------------------------- |
| `ParallelStrategy`         | dataclass          | 第 33 行  | 5D 并行策略：TP、PP、DP、CP（上下文并行）、EP（专家并行）                       |
| `FSDPParallelStrategy`     | `ParallelStrategy` | 第 204 行 | FSDP 专用并行策略（继承，无额外字段）                                           |
| `MegatronParallelStrategy` | `ParallelStrategy` | 第 214 行 | Megatron 专用，增加 `virtual_pipeline_parallel_size` 和 `use_sequence_parallel` |
| `ModelAllocation`          | dataclass          | 第 250 行 | 单模型分配：后端类型、组件名、并行策略、调度策略                                |
| `_AllocationMode`          | dataclass          | 第 383 行 | **已弃用**，仅用于 SPMD 启动器的向后兼容                                        |
| `ParallelDimension`        | dataclass          | 第 642 行 | 解析器内部用的单维度并行参数                                                    |
| `InferenceParallelism`     | dataclass          | 第 657 行 | 向后兼容的推理并行配置                                                          |

`ParallelStrategy` 的关键属性（第 109-160 行）：

- `tp_size` / `pp_size` / `dp_size` / `cp_size` / `ep_size` / `etp_size` / `edp_size` --
  各维度缩写属性
- `world_size` -- 计算公式：`dp * cp * tp * pp`
- `parallelism_eq(this, other)` -- 静态方法比较两个策略是否等价（避免 OmegaConf 兼容性问题）

`ModelAllocation` 的关键方法（第 285-350 行）：

- `from_str(spec, name, scheduling_strategy)` -- 从字符串解析单组件分配，如 `"fsdp:d4"` 或
  `"sglang:d4t2"`
- `world_size` -- 属性，colocation 时返回 0

#### Lark 语法解析器

`ALLOCATION_GRAMMAR`（第 594-638 行）定义了分配模式的完整 EBNF 语法，支持：

- **后端前缀**：`sglang`、`vllm`、`fsdp`、`megatron`、`archon`
- **并行维度**：`d`(数据)、`t`(张量)、`p`(流水线)、`c`(上下文)、`e`(专家)
- **操作符**：`+`（分离，GPU 独立）、`|`（共置，GPU 共享），且 `|` 优先级高于 `+`
- **MoE 混合语法**：`(attn:d1p12t4|ffn:d1p12e4)` 注意力与 FFN 层可用不同并行策略

`_ParallelStrategyTransformer`（第 681 行）和 `_LLMParallelParser`（第 1084 行）负责解析树到
`ModelAllocation` 对象的转换。

### 3.3 `cli_args.py` -- 配置 dataclass 体系（第 1-3047 行）

该文件定义了 50+ 个 dataclass，构成一个层次化的配置树。以下按功能分组列出核心结构。

#### 算法与训练参数

| 类名                        | 基类      | 位置       | 说明                                                                         |
| --------------------------- | --------- | ---------- | ---------------------------------------------------------------------------- |
| `NormConfig`                | dataclass | 第 42 行   | 奖励/优势归一化配置（batch/group 级别，leave-one-out）                       |
| `MicroBatchSpec`            | dataclass | 第 99 行   | 微批次拆分规格（数量、粒度、最大 token 数、打包算法）                        |
| `GenerationHyperparameters` | dataclass | 第 163 行  | 文本生成超参数（temperature、top_p、top_k、stop 等），含 OpenAI API 格式转换 |
| `OptimizerConfig`           | dataclass | 第 335 行  | 优化器配置（adam/sgd/adam_bf16、学习率、调度器）                             |
| `RejectionSamplingConfig`   | dataclass | 第 1277 行 | 拒绝采样/截断配置（token/sequence 级别、mask/clamp 动作、ratio/KL 度量）     |

#### 引擎后端配置

| 类名                            | 基类      | 位置      | 说明                                                         |
| ------------------------------- | --------- | --------- | ------------------------------------------------------------ |
| `FSDPWrapPolicy`                | dataclass | 第 404 行 | FSDP 层包裹策略                                              |
| `FSDPEngineConfig`              | dataclass | 第 414 行 | FSDP 后端配置（参数卸载、内存高效加载、逐层优化）            |
| `ArchonFP8Config`               | dataclass | 第 465 行 | Archon FP8 精度配置（blockwise、排除模块、Triton 内核）      |
| `ArchonEngineConfig`            | dataclass | 第 538 行 | Archon 后端配置（注意力类型、编译、激活检查点、流水线调度）  |
| `DistributedDataParallelConfig` | dataclass | 第 713 行 | Megatron DDP 配置（梯度规约、参数聚合、FP8）                 |
| `FP8EngineConfig`               | dataclass | 第 730 行 | 通用 FP8 训练配置（scaling recipe、amax 历史、注意力 FP8）   |
| `MegatronEngineConfig`          | dataclass | 第 835 行 | Megatron-LM 后端配置（DDP、VPP、精度、MoE 分发器、异步保存） |

#### 训练引擎配置层次

```
TrainEngineConfig (第 1059 行)
    |-- PPOActorConfig (第 1442 行) -- PPO actor 专用参数
    |       |-- TeacherConfig (第 2880 行) -- 蒸馏教师模型
    |-- PPOCriticConfig (第 1642 行) -- PPO critic 专用参数
    |-- DPOEngineConfig (第 2834 行) -- DPO 训练引擎
```

`TrainEngineConfig`（第 1059 行）的关键字段：

- `experiment_name`、`trial_name` -- 必填（`MISSING`）
- `backend` -- 后端+并行策略字符串，如 `"fsdp:d4"`（必填）
- `mb_spec: MicroBatchSpec` -- 微批次规格
- `optimizer: OptimizerConfig | None` -- None 表示不训练
- `fsdp`、`archon`、`megatron` -- 各后端专属配置（嵌套 dataclass）
- `scheduling_strategy: SchedulingStrategy` -- 调度策略（分离/共置）

#### 推理引擎配置

| 类名                    | 位置       | 说明                                                        |
| ----------------------- | ---------- | ----------------------------------------------------------- |
| `vLLMConfig`            | 第 1678 行 | vLLM 运行时配置（模型、KV 缓存、LoRA、多节点）              |
| `SGLangConfig`          | 第 1790 行 | SGLang 运行时配置（模型、注意力后端、DP attention、MoE EP） |
| `AgentConfig`           | 第 1946 行 | Agent 工作流配置（代理模式、工具解析、会话超时）            |
| `InferenceEngineConfig` | 第 2068 行 | 推理引擎配置（离策略控制、轨迹 dump、后端字符串）           |

#### 基础设施配置

| 类名                     | 位置       | 说明                                         |
| ------------------------ | ---------- | -------------------------------------------- |
| `SchedulingStrategyType` | 第 944 行  | 枚举：`separation` / `colocation`            |
| `SchedulingStrategy`     | 第 950 行  | 调度策略（类型、目标角色、是否 fork）        |
| `SchedulingSpec`         | 第 969 行  | 资源规格（CPU/GPU/内存/端口/镜像/环境变量）  |
| `NameResolveConfig`      | 第 2556 行 | 分布式名称解析（NFS/etcd3/Ray）              |
| `ClusterSpecConfig`      | 第 2582 行 | 集群规格（名称解析、文件根、节点数、GPU 数） |
| `SchedulerConfig`        | 第 2612 行 | 调度器配置（单控制器模式）                   |

#### 监控与持久化配置

| 类名                   | 位置       | 说明                                              |
| ---------------------- | ---------- | ------------------------------------------------- |
| `_Timer`               | 第 2247 行 | 定时器基类（epoch/step/秒级频率触发）             |
| `EvaluatorConfig`      | 第 2272 行 | 评估调度（继承 `_Timer`）                         |
| `SaverConfig`          | 第 2284 行 | 检查点保存调度（sync/async/auto 模式）            |
| `RecoverConfig`        | 第 2309 行 | 实验恢复与容错（on/off/auto 模式）                |
| `WandBConfig`          | 第 2351 行 | Weights & Biases 实验跟踪                         |
| `SwanlabConfig`        | 第 2383 行 | SwanLab 实验跟踪                                  |
| `TensorBoardConfig`    | 第 2412 行 | TensorBoard 日志                                  |
| `TrackioConfig`        | 第 2419 行 | Trackio 实验跟踪（Hugging Face）                  |
| `StatsLoggerConfig`    | 第 2449 行 | 统一统计日志配置（聚合 WandB/SwanLab/TB/Trackio） |
| `SessionTracerConfig`  | 第 2474 行 | 会话生命周期追踪                                  |
| `MemoryProfilerConfig` | 第 2498 行 | CUDA 内存快照分析                                 |
| `PerfTracerConfig`     | 第 2517 行 | 性能追踪配置                                      |

#### 实验顶层配置层次

```
BaseExperimentConfig (第 2724 行)
    |-- SFTConfig (第 2812 行) -- 监督微调
    |-- RWConfig (第 2819 行) -- 奖励模型训练
    |-- DPOConfig (第 2863 行) -- 直接偏好优化
    |-- PPOConfig (第 2893 行) -- 近端策略优化
            |-- GRPOConfig (第 2937 行) -- GRPO（PPO 的别名）
```

#### 配置解析函数

| 函数                                      | 位置       | 说明                                                                |
| ----------------------------------------- | ---------- | ------------------------------------------------------------------- |
| `parse_cli_args(argv)`                    | 第 2943 行 | 解析命令行参数和 `--config` 指定的 YAML 文件，返回 Hydra DictConfig |
| `to_structured_cfg(cfg, config_cls)`      | 第 3003 行 | 将 DictConfig 与 dataclass 默认值合并，拦截遗留配置键               |
| `load_expr_config(argv, config_cls)`      | 第 3013 行 | 完整的配置加载流程：解析 CLI -> 合并 -> 实例化 -> 环境设置          |
| `conf_as_dict(cfg)`                       | 第 3031 行 | 将 OmegaConf/dataclass 配置转为普通字典                             |
| `save_config(cfg, log_dir)`               | 第 3037 行 | 保存配置到 YAML 文件                                                |
| `_migrate_legacy_rejection_sampling(cfg)` | 第 2985 行 | 拦截已移除的 `behave_imp_weight_*` 遗留键并给出迁移指引             |

### 3.4 `engine_api.py` -- 引擎抽象接口（第 1-1008 行）

#### `TrainEngine`（ABC，第 32 行）

训练引擎的核心抽象，定义了分布式训练的完整生命周期：

| 方法签名                                                             | 类型     | 说明                                |
| -------------------------------------------------------------------- | -------- | ----------------------------------- |
| `create_process_group(parallel_strategy)`                            | 抽象     | 初始化 PyTorch 分布式通信组         |
| `initialize(*args, **kwargs)`                                        | 抽象     | 加载模型和初始化训练环境            |
| `data_parallel_group`                                                | 抽象属性 | 数据并行通信组                      |
| `data_parallel_rank` / `data_parallel_world_size`                    | 抽象属性 | DP rank 和 world size               |
| `context_and_model_parallel_group`                                   | 抽象属性 | 上下文+模型并行通信组               |
| `cpu_group`                                                          | 抽象属性 | CPU 通信组                          |
| `train(mode=True)` / `eval()`                                        | 抽象     | 训练/评估模式切换                   |
| `forward_backward_batch(mb_list, process_output_fn, forward_only)`   | 抽象     | 微批次前向+反向传播                 |
| `train_batch(input_, loss_fn, loss_weight_fn)`                       | 抽象     | 完整训练步骤                        |
| `eval_batch(input_, loss_fn, loss_weight_fn)`                        | 抽象     | 评估步骤（`@torch.no_grad()`）      |
| `forward_batch(input_, output_seqlens, aggregate_fn)`                | 抽象     | 纯前向推理（`@torch.no_grad()`）    |
| `optimizer_zero_grad()` / `optimizer_step()` / `lr_scheduler_step()` | 抽象     | 优化器操作                          |
| `update_weights(meta: WeightUpdateMeta)`                             | 抽象     | 阻塞式权重更新到推理引擎            |
| `connect_engine(engine: InferenceEngine, meta)`                      | 抽象     | 连接推理引擎                        |
| `rollout_batch(data, workflow, ...)`                                 | 抽象     | 批量 rollout（同步）                |
| `prepare_batch(dataloader, workflow, ...)`                           | 抽象     | 异步批量数据准备                    |
| `save(meta: SaveLoadMeta)` / `load(meta)`                            | 抽象     | 检查点保存/加载                     |
| `set_version(v)` / `get_version()`                                   | 抽象     | 权重版本管理                        |
| `export_stats()`                                                     | 抽象     | 导出训练统计（跨 DP 组 all-reduce） |
| `onload()` / `offload()`                                             | 抽象     | GPU/CPU 模型迁移                    |
| `config_perf_tracer(config, rank, role)`                             | 具体     | 配置性能追踪器                      |

#### `InferenceEngine`（ABC，第 547 行）

推理引擎的核心抽象，支持同步/异步、本地/远程两种模式：

| 方法签名                                                       | 类型  | 说明                                          |
| -------------------------------------------------------------- | ----- | --------------------------------------------- |
| `initialize(*args, **kwargs)`                                  | 具体  | 初始化推理环境（本地 GPU 或远程连接）         |
| `launch_server(server_args)`                                   | 具体  | 启动本地推理服务器，返回 `LocalInfServerInfo` |
| `teardown_server()`                                            | 具体  | 关闭本地推理服务器                            |
| `agenerate(req: ModelRequest) -> ModelResponse`                | async | 异步生成响应                                  |
| `submit(data, workflow, ...) -> int`                           | 具体  | 提交请求（非阻塞），返回 task_id              |
| `wait(count, timeout) -> list[dict]`                           | 具体  | 等待指定数量的结果                            |
| `wait_for_task(task_id, timeout)`                              | 具体  | 等待特定任务完成                              |
| `rollout_batch(data, workflow, ...)`                           | 具体  | 批量 rollout（同步）                          |
| `prepare_batch(dataloader, workflow, ...)`                     | 具体  | 异步批量数据准备（首次调用缓存生成器）        |
| `init_weights_update_group(meta, rank_ids) -> Future`          | 具体  | 初始化分布式权重更新通信组                    |
| `update_weights_from_distributed(meta, param_specs) -> Future` | 具体  | 分布式权重更新（非阻塞）                      |
| `update_weights_from_disk(meta) -> Future`                     | 具体  | 磁盘权重更新（非阻塞）                        |
| `pause_generation()` / `continue_generation()`                 | 具体  | 权重更新期间暂停/恢复生成                     |
| `pause()` / `resume()`                                         | 具体  | 评估期间暂停/恢复请求提交                     |
| `offload()` / `onload(tags)`                                   | 具体  | GPU/CPU 模型迁移                              |

### 3.5 `io_struct.py` -- I/O 数据结构（第 1-417 行）

| 类名                   | 位置      | 说明                                                                           |
| ---------------------- | --------- | ------------------------------------------------------------------------------ |
| `ModelRequest`         | 第 28 行  | 模型推理请求（rid、input_ids、gconfig、tokenizer、VLM 图像数据）               |
| `ModelResponse`        | 第 63 行  | 模型推理响应（输入/输出 token、logprob、版本、延迟统计、MoE 路由）             |
| `FinetuneSpec`         | 第 134 行 | 微调训练规格（epoch 数、数据集大小、batch size，计算属性 `total_train_steps`） |
| `ParamSpec`            | 第 150 行 | 参数规格（名称、形状、dtype，计算属性 `size` 返回字节数）                      |
| `WeightUpdateMeta`     | 第 183 行 | 权重更新元数据（类型 disk/xccl/awex、路径、LoRA、NCCL 配置）                   |
| `HttpRequest`          | 第 304 行 | HTTP 请求封装（endpoint、payload、method）                                     |
| `HttpGenerationResult` | 第 313 行 | HTTP 生成结果（output_tokens、logprobs、stop_reason）                          |
| `WeightUpdateRequests` | 第 323 行 | 权重更新 HTTP 请求集合                                                         |
| `SaveLoadMeta`         | 第 330 行 | 检查点保存/加载元数据（路径、格式、是否含优化器状态）                          |
| `RolloutStat`          | 第 341 行 | Rollout 统计（accepted/enqueued/rejected/running）                             |
| `StepInfo`             | 第 349 行 | 训练步信息（epoch、epoch_step、global_step），含 `next()` 方法                 |
| `LocalInfServerInfo`   | 第 369 行 | 本地推理服务器信息（host、port、process）                                      |
| `DeviceRuntimeInfo`    | 第 378 行 | 设备运行时内存信息（allocated/reserved/used/total），含 `get_current()`        |

`WeightUpdateMeta` 的关键工厂方法：

- `from_disk(...)` -- 磁盘权重更新（第 218 行）
- `from_megatron_xccl(...)` / `from_fsdp_xccl(...)` -- XCCL 分布式权重更新（第 247/267 行）
- `from_awex(...)` -- AWEX 权重更新（第 287 行）
- `with_version(version)` -- 返回带版本号路径的副本（第 203 行）

### 3.6 `reward_api.py` -- 奖励函数接口（第 1-183 行）

| 名称                                                                   | 类型         | 位置     | 说明                                                  |
| ---------------------------------------------------------------------- | ------------ | -------- | ----------------------------------------------------- |
| `reward_fn(prompt, completions, prompt_ids, completion_ids, **kwargs)` | 函数（占位） | 第 40 行 | 奖励函数签名约定（返回 `float`）                      |
| `AsyncRewardWrapper`                                                   | 类           | 第 62 行 | 将同步奖励函数包装为异步执行（`ProcessPoolExecutor`） |

`AsyncRewardWrapper` 的关键设计（第 62-183 行）：

- **进程池共享**：通过类变量 `_executors` 按 `max_workers` 键共享 `ProcessPoolExecutor`
- **线程安全**：使用 `threading.Lock` 保护进程池创建和回收
- **超时重试**：`__call__` 最多重试 `max_retries` 次，超时返回 0
- **崩溃恢复**：捕获 `BrokenProcessPool` 异常后自动重建进程池
- **清洁退出**：通过 `atexit.register` 确保进程池在退出前正确关闭
- **自动 worker 数**：默认 `max_workers = max((cpu_count // device_count) // 2, 1)`

### 3.7 `scheduler_api.py` -- 调度器抽象接口（第 1-329 行）

| 类名        | 位置     | 说明                                                   |
| ----------- | -------- | ------------------------------------------------------ |
| `Worker`    | 第 14 行 | 工作进程数据结构（id、ip、worker_ports、engine_ports） |
| `Job`       | 第 36 行 | 作业定义（role、replicas、tasks、scheduling_strategy） |
| `Scheduler` | 第 43 行 | 调度器 ABC                                             |

`Scheduler` 的抽象方法：

| 方法签名                                                        | 说明                                         |
| --------------------------------------------------------------- | -------------------------------------------- |
| `n_gpus_per_node -> int`                                        | 每节点 GPU 数（属性）                        |
| `create_workers(job, ...) -> list[str]`                         | 创建工作进程                                 |
| `get_workers(role, timeout) -> list[Worker]`                    | 获取指定角色的就绪工作进程                   |
| `delete_workers(role, reverse_order)`                           | 删除工作进程（可反序关闭避免 TCPStore 警告） |
| `fork_workers(role, target_role, command) -> list[str]`         | 从现有 worker fork 新进程（共置场景）        |
| `create_engine(worker_id, engine, engine_name, ...) -> Any`     | 在远程 worker 上创建引擎实例（async）        |
| `set_worker_env(worker_id, env)`                                | 设置 worker 环境变量（async）                |
| `call_engine(worker_id, method, engine_name, ...) -> Any`       | 同步调用引擎方法                             |
| `async_call_engine(worker_id, method, engine_name, ...) -> Any` | 异步调用引擎方法                             |

### 3.8 `workflow_api.py` -- 工作流抽象接口（第 1-115 行）

| 名称                           | 类型          | 位置      | 说明                                                           |
| ------------------------------ | ------------- | --------- | -------------------------------------------------------------- |
| `RolloutWorkflow`              | ABC           | 第 14 行  | Rollout 工作流基类，定义 `arun_episode(engine, data)` 抽象方法 |
| `AgentWorkflow`                | ABC（已弃用） | 第 63 行  | Agent 工作流基类，定义 `run(data, **extra_kwargs)` 抽象方法    |
| `_DeprecatedAgentWorkflowMeta` | ABCMeta       | 第 42 行  | 确保 `AgentWorkflow` 子类实例化时触发弃用警告的元类            |
| `WorkflowLike`                 | TypeAlias     | 第 110 行 | 联合类型 \`RolloutWorkflow                                     |

`WorkflowLike` 支持四种传入形式：

1. `RolloutWorkflow` 实例
1. `RolloutWorkflow` 子类（类型对象）
1. 字符串模块路径（如 `"areal.workflow.rlvr.RLVRWorkflow"`）
1. 任何具有兼容 `run()` 方法的对象

## 4. 算法与逻辑详解

### 4.1 AllocationMode 分配语法解析

分配语法的解析是 `alloc_mode.py` 最复杂的逻辑，采用 Lark 的 Earley 解析器处理。

#### 语法规则（第 594-638 行）

核心语法结构如下：

```
expression      = disaggregate_chain | component
disaggregate_chain = component ("+" component)+     # 分离：独立 GPU
component       = colocate_expr | single_allocation
colocate_expr   = single_allocation ("|" single_allocation)+  # 共置：共享 GPU
single_allocation = inf_para | train_para
inf_para        = INFER_BACKEND ("[" NAME "]")? ":" inf_dim+
train_para      = TRAIN_BACKEND ("[" NAME "]")? ":" common_dim+
                | TRAIN_BACKEND ":" hybrid_moe_syntax
```

**操作符优先级**（第 592-593 行注释）：`|`（共置）的绑定优先级高于 `+`（分离）。因此 `"a+b|c"` 解析为 `"a+(b|c)"`，而非
`"(a+b)|c"`。

#### 解析流程

1. `_LLMParallelParser.parse(expression)`（第 1096 行）调用 Lark 解析字符串为 AST。
1. `_ParallelStrategyTransformer`（第 681 行）通过 Lark 的 Transformer 模式自底向上遍历 AST，将各节点转为
   `ModelAllocation` 对象。
1. 核心转换方法：
   - `modern_inf_para`（第 784 行）：处理推理后端解析，如 `sglang[rollout]:d2`
   - `train_backend_with_name`（第 829 行）：处理带名称的训练后端，如 `fsdp[actor]:d4`
   - `colocate_expr`（第 743 行）：处理 `|` 操作符，第一个组件为 anchor（separation），后续组件设为 colocation
   - `hybrid_moe_syntax`（第 1042 行）-> `hybrid_train_para`（第 943 行）：处理 MoE 混合并行

#### 验证规则

- **名称唯一性**（第 688-693 行）：`_validate_name` 确保组件名不重复
- **3+ 组件必须命名**（第 727-733 行）：`disaggregate_chain` 中若有 3 个以上组件则全部必须有名称
- **共置世界大小一致**（第 771-776 行）：`colocate_expr` 中所有组件的 `world_size` 必须相同
- **FSDP 限制**（第 275-283 行）：FSDP 后端不支持 PP 和 EP
- **MoE PP 一致性**（第 986-990 行）：注意力和 FFN 模块的 PP 大小必须相同
- **MoE 世界大小一致**（第 1026-1030 行）：注意力和专家模块的总 world size 必须相同
- **后端必须显式指定**（第 896-910 行）：不再支持自动后端选择

### 4.2 CLI 参数解析流程

配置解析由 `load_expr_config`（第 3013 行）驱动，完整流程如下：

```
argv (命令行参数)
  |
  v
parse_cli_args(argv) ---- 第 2943 行
  |-- argparse 解析 --config 路径
  |-- Hydra 初始化 + compose（支持 override）
  |
  v
DictConfig (原始配置)
  |
  v
to_structured_cfg(cfg, config_cls) ---- 第 3003 行
  |-- _migrate_legacy_rejection_sampling(cfg)  检查遗留键
  |-- OmegaConf.structured(config_cls)  生成默认配置
  |-- OmegaConf.merge(default_cfg, cfg)  合并（YAML 可省略默认值）
  |
  v
OmegaConf.to_object(cfg) -> config_cls 实例 ---- 第 3016 行
  |-- 触发所有 dataclass 的 __post_init__ 验证
  |
  v
环境设置
  |-- name_resolve.reconfigure(...)
  |-- save_config(...)
```

### 4.3 引擎抽象接口设计

`TrainEngine` 和 `InferenceEngine` 的接口设计遵循以下原则：

1. **生命周期分阶段**：`create_process_group` -> `initialize` -> `train/eval` ->
   `destroy`，每个阶段独立可控。

1. **同步/异步双模式**（InferenceEngine）：

   - 同步：`rollout_batch` -- 提交并等待全部结果
   - 异步：`submit` + `wait` / `wait_for_task` -- 细粒度控制

1. **权重更新抽象**（第 175-183 行、第 635-698 行）：

   - TrainEngine 通过 `update_weights(meta)` 阻塞式推送
   - InferenceEngine 通过 `update_weights_from_distributed(meta, param_specs)` 非阻塞拉取（返回
     `Future`）
   - 支持三种模式：磁盘（disk）、NCCL/XCCL 分布式（xccl）、AWEX

1. **性能分析钩子**：`config_perf_tracer` 和 `save_perf_tracer` 在两个引擎中均提供具体空实现，允许子类按需覆盖。

## 5. 数据流（输入/输出）

```
+-------------------+
|   YAML Config     |
|  + CLI Overrides  |
+--------+----------+
         |
         | parse_cli_args() + to_structured_cfg()
         v
+-------------------+
|   cli_args.py     |  PPOConfig / GRPOConfig / SFTConfig ...
|  (配置 dataclass) |
+--------+----------+
         |
         | 实例化时通过字段引用
         v
+-------------------+       +-------------------+
| alloc_mode.py     |       |   io_struct.py    |
| (ParallelStrategy,|       | (ModelRequest,    |
|  ModelAllocation)  |       |  ModelResponse,   |
+--------+----------+       |  WeightUpdateMeta)|
         |                   +--------+----------+
         |                            |
    +----+----+             +---------+----------+
    |         |             |                    |
    v         v             v                    v
+--------+ +--------+  +--------+  +-------------------+
|engine_ | |schedu- |  |work-   |  |  reward_api.py    |
|api.py  | |ler_    |  |flow_   |  | (AsyncRewardWrapper|
|        | |api.py  |  |api.py  |  |  reward_fn 签名)   |
+---+----+ +---+----+  +---+----+  +--------+----------+
    |          |            |                |
    |          |            |                |
    v          v            v                v
+--------------------------------------------------+
|        areal/engine/    areal/workflow/           |
|        areal/infra/     areal/reward/             |
|          (具体实现层)                              |
+--------------------------------------------------+
```

上下游关系说明：

```
+----------------+    导入    +------------------+    导入    +---------------+
| examples/      | --------> | areal/api/       | <-------- | areal/engine/ |
| (训练脚本)      |           | cli_args.py      |           | fsdp_engine   |
| + YAML configs  |           | engine_api.py    |           | megatron_eng  |
+----------------+           | alloc_mode.py    |           | sglang_remote |
                              | io_struct.py     |           | vllm_remote   |
                              | workflow_api.py  |           +---------------+
                              | scheduler_api.py |
                              | reward_api.py    | <-------- +---------------+
                              +------------------+           | areal/infra/  |
                                       ^                     | schedulers    |
                                       |                     | RPC           |
                                       +--------- 导入 ------+---------------+
                                       |
                              +--------+--------+
                              | areal/workflow/  |
                              | rlvr.py          |
                              | multi_turn.py    |
                              +--------+--------+
                                       |
                                       v  使用
                              +-----------------+
                              | areal/reward/   |
                              | (奖励函数实现)   |
                              +-----------------+
```

## 6. 关键设计决策与不变量

### 6.1 惰性导入策略

`__init__.py` 使用 `__getattr__` + `_LAZY_IMPORTS` 字典实现惰性导入（第 56-64 行），而非直接
`from ... import ...`。这一设计确保 `import areal.api` 不会触发 PyTorch、Transformers 等重量级模块的加载，对
CLI 工具和文档生成等场景至关重要。首次访问某个名称时才执行 `importlib.import_module`，并通过 `globals()[name] = val`
缓存结果。

### 6.2 OmegaConf 兼容性约束

所有配置 dataclass 必须兼容 OmegaConf 的结构化配置要求：

- 不能使用 `torch.dtype` 等不可序列化的类型注解（第 856 行注释）
- `__post_init__` 中的验证必须容忍 `MISSING` 占位符
- `parallelism_eq` 使用静态方法而非 `__eq__`，避免 OmegaConf 代理对象的比较问题（第 192 行注释）
- vLLMConfig / SGLangConfig 的 `build_cmd` 等方法必须为 `@staticmethod`，确保 OmegaConf 序列化兼容（第
  1861 行注释）

### 6.3 后端必须显式指定

`alloc_mode.py` 中明确移除了自动后端推断（第 820-825 行、第 896-910 行），所有分配字符串必须以后端名称开头（如
`fsdp:d4`、`sglang:d2t4`）。这一决策消除了隐式推断带来的歧义，使配置更具可读性和可调试性。

### 6.4 AllocationMode 已弃用

`AllocationMode` 类已被移除（第 1158-1176 行），推荐使用 `ModelAllocation` + 每引擎 `backend` 字段。内部
`_AllocationMode` 仅保留用于 SPMD 启动器的向后兼容。尝试导入 `AllocationMode` 会触发明确的 `AttributeError`（第
1168-1176 行）。

### 6.5 WeightUpdateMeta 的三种传输模式

权重更新支持三种类型（`io_struct.py` 第 184 行）：

- `"disk"` -- 通过共享文件系统传输，适用于跨节点无 NCCL 连接的场景
- `"xccl"` -- 通过 NCCL/XCCL 分布式通信传输，最快但需要通信组初始化
- `"awex"` -- AWEX（Archon Weight Exchange）传输，Archon 引擎专用

### 6.6 AgentWorkflow 的弃用策略

`AgentWorkflow` 通过自定义元类 `_DeprecatedAgentWorkflowMeta`（第 42 行）确保即使子类未调用
`super().__init__()`，弃用警告也会在 `__call__`（即 `__init__` 之前）触发。这是因为 `ABCMeta.__call__` 在
`__init__` 之前执行，保证了警告的可靠触发。

### 6.7 不变量

- 所有 `ParallelStrategy` 的 `world_size = dp * cp * tp * pp`
- 共置组件的 `world_size` 必须相同
- `SchedulingSpec` 的 `scheduling_spec` 长度只能为 1 或 2
- `BaseExperimentConfig.total_train_epochs` 必须为正数
- `RejectionSamplingConfig` 的 `action='clamp'` 仅支持 `metric='ratio'`
- `TrainEngineConfig` 的 `optimizer_dtype` 仅允许 `'float32'` 或 `'bfloat16'`
- `FSDPEngineConfig` 不支持 PP 和 EP（仅 DP/TP/CP）
- `MegatronParallelStrategy` 的 VPP 需要 PP > 1

## 7. 已知问题与限制

1. **cli_args.py 文件过大**（3047 行）：所有配置 dataclass
   集中在一个文件中，违反单一职责原则。随着新引擎、新算法的加入，该文件持续膨胀。可考虑拆分为
   `engine_configs.py`、`experiment_configs.py`、`infra_configs.py` 等子模块。

1. **OmegaConf 序列化限制**：不能使用 `torch.dtype` 等类型注解，导致精度类型只能用字符串表示（如
   `"float32"`、`"bfloat16"`），需要在 `__post_init__` 中做别名规范化（第 1254-1263 行）。

1. **遗留向后兼容负担**：`_AllocationMode`、`AllocationType`、`InferenceParallelism` 等遗留类仅为 SPMD
   启动器保留，增加了代码复杂度。

1. **MegatronEngineConfig 中的 TODO**：`use_torch_fsdp2` 和 `use_custom_fsdp` 标记为
   `TODO: pending test`（第 842-843 行），表明这些功能尚未完成测试。

1. **vLLMConfig 的布尔参数反转**：`no_enable_chunked_prefill` 和 `no_enable_prefix_caching`
   使用双重否定，因为 `get_py_cmd()` 会忽略值为 `False` 的参数（第 1696-1709 行注释）。这是一种权宜之计，降低了可读性。

1. **`prepare_batch` 的隐式缓存**：`InferenceEngine.prepare_batch`（第 869
   行）在首次调用时缓存内部数据生成器，后续调用时传入不同参数**不会生效**（第 882-894 行 warning 文档）。这一行为容易导致使用者困惑。

1. **`AsyncRewardWrapper` 的序列化约束**：由于使用 `ProcessPoolExecutor`，奖励函数及其参数必须可 pickle 序列化（第
   69 行注释），排除了使用 lambda、闭包或持有不可序列化状态的奖励函数。

## 8. 相关测试覆盖

| 测试文件                           | 行数 | 覆盖范围                                                                                  |
| ---------------------------------- | ---- | ----------------------------------------------------------------------------------------- |
| `tests/test_allocation_mode.py`    | 484  | `alloc_mode.py` 的解析、验证、向后兼容属性、操作符优先级、`ModelAllocation.from_str`      |
| `tests/test_rejection_sampling.py` | 839  | `RejectionSamplingConfig` 的验证、ratio/KL 度量过滤、token/sequence 级别、mask/clamp 动作 |
| `tests/test_train_engine.py`       | --   | `TrainEngine` 的集成测试（需 GPU）                                                        |
| `tests/test_inference_engines.py`  | --   | `InferenceEngine` 的集成测试（需 GPU + 推理服务器）                                       |
| `tests/test_dynamic_import.py`     | --   | `__init__.py` 惰性导入机制                                                                |
| `tests/test_train_controller.py`   | --   | 训练控制器中引擎 API 和调度器的集成                                                       |
| `tests/test_local_scheduler.py`    | --   | 本地调度器（`Scheduler` ABC 的实现）                                                      |
| `tests/test_ray_scheduler.py`      | --   | Ray 调度器实现                                                                            |
| `tests/test_slurm_scheduler.py`    | --   | Slurm 调度器实现                                                                          |
| `tests/test_rollout_controller.py` | --   | Rollout 控制器中推理引擎的集成                                                            |
| `tests/test_recover.py`            | --   | `RecoverConfig` 恢复功能                                                                  |
| `tests/test_perf_tracer.py`        | --   | `PerfTracerConfig` 性能追踪                                                               |
| `tests/test_trackio_backend.py`    | --   | `TrackioConfig` 实验跟踪                                                                  |

测试覆盖的关键观察：

- `alloc_mode.py` 和 `RejectionSamplingConfig` 拥有独立的、较为完整的单元测试。
- `engine_api.py`、`scheduler_api.py`、`workflow_api.py` 作为纯抽象接口，不直接拥有单元测试，其覆盖来自下游实现的集成测试。
- `cli_args.py` 中大量的 `__post_init__` 验证逻辑没有独立的单元测试，依赖于实验配置加载的端到端测试。
- `io_struct.py` 和 `reward_api.py` 的测试覆盖主要通过工作流和引擎的集成测试间接完成。
