# AReaL (`areal/`) — 源码架构分析

> 最后更新：2026-06-13 一句话定位：面向 LLM 对齐的大规模异步强化学习分布式训练框架

## 1. 系统概述

AReaL（Asynchronous Reinforcement Learning for Language
reasoning）是一个面向大语言模型强化学习对齐的分布式训练框架。其核心使命是将 RL 训练（PPO/GRPO/DPO/SFT）与异步 Rollout 推理高效地调度在
GPU 集群上，最大化训练吞吐。

在上层项目中，AReaL 提供完整的端到端 RL
训练管线：从数据集加载、工作流定义、推理引擎调度、奖励计算、到训练引擎的前向/反向/优化器更新。它同时支持三种并行训练后端（FSDP2、Megatron-Core、Archon），两种推理后端（SGLang、vLLM），以及三种集群部署模式（Local、Ray、Slurm）。

定量规模概述：

| 指标         | 数值                              |
| ------------ | --------------------------------- |
| 生产文件数   | 402 个                            |
| 生产代码行数 | 107,052 行                        |
| 测试文件数   | 233 个                            |
| 测试代码行数 | 79,883 行                         |
| 顶层子模块数 | 11 个                             |
| 训练引擎后端 | 3 种（FSDP2 / Megatron / Archon） |
| 推理引擎后端 | 2 种（SGLang / vLLM）             |
| 部署模式     | 3 种（Local / Ray / Slurm）       |

## 2. 总体流程图

AReaL 的 RL 训练主循环遵循 "异步 Rollout → 训练 → 权重同步" 的迭代模式：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PPOTrainer 主循环                               │
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────────┐  │
│  │  DataLoader   │───▶│  prepare_batch   │───▶│  train_batch (Actor)  │  │
│  │              │    │                  │    │                       │  │
│  │ 输入：数据集   │    │ 输入：原始样本     │    │ 输入：轨迹 + logprobs  │  │
│  │ 输出：原始样本 │    │ 输出：RL 轨迹      │    │ 输出：loss + stats    │  │
│  └──────────────┘    └────────┬─────────┘    └───────────┬───────────┘  │
│                               │                          │              │
│                    ┌──────────▼──────────┐    ┌──────────▼──────────┐  │
│                    │  WorkflowExecutor   │    │  optimizer_step     │  │
│                    │                    │    │                    │  │
│                    │  ┌──────────────┐  │    │  梯度裁剪 → 优化器    │  │
│                    │  │ RolloutWork- │  │    │  更新 → LR 调度      │  │
│                    │  │ flow.arun_   │  │    └──────────┬──────────┘  │
│                    │  │ episode()    │  │               │              │
│                    │  └──────┬───────┘  │    ┌──────────▼──────────┐  │
│                    │         │          │    │  update_weights     │  │
│                    │  ┌──────▼───────┐  │    │                    │  │
│                    │  │ Inference    │  │    │  训练权重 → 推理引擎  │  │
│                    │  │ Engine       │  │    │  (disk / NCCL)      │  │
│                    │  │ .agenerate() │  │    └────────────────────┘  │
│                    │  └──────┬───────┘  │                            │
│                    │         │          │                            │
│                    │  ┌──────▼───────┐  │                            │
│                    │  │ Reward       │  │                            │
│                    │  │ Function     │  │                            │
│                    │  └──────────────┘  │                            │
│                    └────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

分布式部署视角：

```
┌──────────────────────────────────────────────────────────────────┐
│                     Scheduler (Local/Ray/Slurm)                  │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐ │
│  │   Launcher     │  │   Launcher     │  │   Launcher         │ │
│  │                │  │                │  │                    │ │
│  │ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌────────────────┐ │ │
│  │ │ Guard 进程  │ │  │ │ Guard 进程  │ │  │ │ SGLang/vLLM    │ │ │
│  │ │            │ │  │ │            │ │  │ │ Server         │ │ │
│  │ │ TrainEngine│ │  │ │ TrainEngine│ │  │ │                │ │ │
│  │ │ (Actor)    │ │  │ │ (Ref/Criti)│ │  │ │ InferenceEngine│ │ │
│  │ └────────────┘ │  │ └────────────┘ │  │ └────────────────┘ │ │
│  │  Worker: actor │  │  Worker: ref   │  │  Worker: rollout   │ │
│  └────────────────┘  └────────────────┘  └────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              TrainController / RolloutController            │  │
│  │     RPC 调用引擎方法 + 聚合统计 + 数据分发/收集              │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## 3. 子模块导航表

| #   | 模块               | 文档链接                                                       | 源码位置                                            | 行数   | 一句话概述                                         |
| --- | ------------------ | -------------------------------------------------------------- | --------------------------------------------------- | ------ | -------------------------------------------------- |
| 1   | API 契约层         | [01-api.md](01-api.md)                                         | `areal/api/`                                        | 6,343  | 定义引擎、工作流、调度器的抽象接口与配置 dataclass |
| 2   | 通用工具库         | [02-utils.md](02-utils.md)                                     | `areal/utils/`                                      | 10,728 | 日志、数据操作、性能追踪、分布式名称解析等基础设施 |
| 3   | 基础设施平台与工具 | [03-infra-foundation.md](03-infra-foundation.md)               | `areal/infra/platforms/`, `utils/`, `sandbox/`      | 2,590  | 硬件平台抽象、HTTP/并发/进程工具、沙箱集成         |
| 4   | RPC 与序列化       | [04-infra-rpc.md](04-infra-rpc.md)                             | `areal/infra/rpc/`                                  | 3,252  | 分布式远程调用、张量传输、Guard 进程管理           |
| 5   | 数据集服务         | [05-infra-data-service.md](05-infra-data-service.md)           | `areal/infra/data_service/`                         | 2,100  | 数据集微服务（Gateway/Router/Worker 模式）         |
| 6   | 编排控制层         | [06-infra-orchestration.md](06-infra-orchestration.md)         | `areal/infra/controller/` 等                        | 6,392  | Rollout/Train 控制器、工作流执行器、远程推理引擎   |
| 7   | 启动器与调度器     | [07-infra-deployment.md](07-infra-deployment.md)               | `areal/infra/launcher/`, `scheduler/`               | 6,999  | Local/Ray/Slurm 部署、Worker 生命周期管理          |
| 8   | 模型层             | [08-models.md](08-models.md)                                   | `areal/models/`                                     | 8,192  | Ulysses 序列并行、Megatron-Core 桥接、树注意力     |
| 9   | 训练与推理引擎     | [09-engine.md](09-engine.md)                                   | `areal/engine/`                                     | 12,932 | FSDP2/Megatron 训练引擎 + SGLang/vLLM 推理引擎     |
| 10  | 训练器             | [10-trainer.md](10-trainer.md)                                 | `areal/trainer/`                                    | 4,590  | SFT/RL/DPO/RW 训练编排 + PPO Actor/Critic 损失     |
| 11  | 数据集/奖励/工作流 | [11-dataset-reward-workflow.md](11-dataset-reward-workflow.md) | `areal/dataset/`, `reward/`, `workflow/`            | 2,519  | 数据集加载、奖励函数、RLVR/多轮/视觉/Agent 工作流  |
| 12  | Archon 引擎与模型  | [12-experimental-archon.md](12-experimental-archon.md)         | `areal/experimental/engine/`, `models/`             | 13,673 | 第三代训练引擎 + Qwen2/3/3.5 模型 + MoE 专家并行   |
| 13  | 实验性微服务       | [13-experimental-services.md](13-experimental-services.md)     | `areal/experimental/*_service/`, `weight_update/`   | 16,378 | V2 微服务架构（推理/训练/Agent/权重更新服务）      |
| 14  | OpenAI 兼容层      | [14-experimental-openai.md](14-experimental-openai.md)         | `areal/experimental/openai/`, `camel/`, `workflow/` | 5,174  | OpenAI SDK 兼容 + Agent 工作流代理                 |
| 15  | 开发工具           | [15-tools.md](15-tools.md)                                     | `areal/tools/`                                      | 5,003  | 性能分析、安装验证、CI 检查工具                    |

## 4. 组件依赖拓扑

```
areal（顶层包）
├── api（契约层 — 所有其他模块的接口定义）
│   ├── cli_args（配置 dataclass 树）
│   ├── engine_api（TrainEngine / InferenceEngine ABC）
│   ├── workflow_api（RolloutWorkflow / AgentWorkflow ABC）
│   ├── scheduler_api（Scheduler / Worker / Job ABC）
│   ├── reward_api（AsyncRewardWrapper）
│   ├── alloc_mode（资源分配与并行策略）
│   └── io_struct（ModelRequest / ModelResponse / StepInfo 等数据对象）
│
├── utils（基础工具层 — 被所有模块依赖）
│   ├── logging（统一日志 + 颜色方案）
│   ├── data（批处理、微批切分、序列打包、张量容器操作）
│   ├── functional（RL 损失函数：PPO Actor/Critic/DPO/SAPO）
│   ├── perf_tracer（性能追踪 + 会话追踪）
│   ├── name_resolve（分布式名称注册：NFS / etcd / Ray）
│   ├── seqpack（序列打包算法：FFD / KK）
│   └── ...（网络、文件系统、版本、时间等工具）
│
├── infra（基础设施层）
│   ├── platforms（硬件抽象：CUDA / CPU / NPU）
│   │   └── Platform ABC → CudaPlatform / CpuPlatform / NPUPlatform
│   ├── rpc（远程调用）
│   │   ├── rtensor（分布式张量传输：HTTP / Ray 后端）
│   │   ├── serialization（序列化协议：Tensor/NDArray/Dataclass/Tokenizer）
│   │   └── guard（Guard 进程：数据蓝图 + 引擎蓝图）
│   ├── controller
│   │   ├── RolloutController（异步推理调度 + 统计聚合）
│   │   └── TrainController（训练数据分发 + 微批管理）
│   ├── WorkflowExecutor（批量任务调度 + 异步 Rollout 管线）
│   ├── RemoteInfEngine（远程推理引擎代理 + 权重更新协调）
│   ├── launcher
│   │   ├── LocalLauncher / RayLauncher / SlurmLauncher
│   │   └── SGLangServerWrapper / vLLMServerWrapper
│   ├── scheduler
│   │   └── LocalScheduler / RayScheduler / SlurmScheduler
│   └── data_service（数据集微服务 Gateway/Router/Worker）
│
├── models（模型实现层）
│   ├── fsdp/ulysses（Ulysses 序列并行：all-to-all 通信）
│   ├── mcore（Megatron-Core 桥接：HF↔MCore 权重转换、BailingMoE）
│   ├── transformers（HuggingFace 模型补丁：Qwen2VL/Qwen3VL、视觉 SP 分片）
│   └── tree_attn（树注意力：Trie→打包→Triton 内核 / FlexAttention）
│
├── engine（引擎层 — 实现 api 中定义的引擎接口）
│   ├── fsdp_engine（FSDP2 训练引擎 + PPOActor/Critic/LM/RW/DPO 变体）
│   ├── megatron_engine（Megatron 训练引擎 + 同样的变体层次）
│   ├── sglang_remote / vllm_remote（远程推理引擎客户端）
│   ├── fsdp_utils（梯度裁剪、AnyPrecisionAdamW、并行化、检查点）
│   ├── megatron_utils（检查点、FP8 量化、流水线并行、Context Parallel）
│   └── vllm_ext（vLLM 服务器扩展 + Worker 扩展）
│
├── trainer（训练器层 — 编排引擎完成完整训练循环）
│   ├── PPOTrainer（RL 训练主循环：rollout → train → eval → save）
│   ├── SFTTrainer / RWTrainer / DPOTrainer
│   └── ppo/（PPOActor 损失计算、PPOCritic、统计）
│       └── dpo/ / rw/ / sft/（各训练模式的引擎控制器）
│
├── dataset（数据集加载器：gsm8k / clevr / geometry3k / virl39k / hhrlhf / torl）
├── reward（奖励函数：gsm8k / clevr / geometry3k + MathVerify）
├── workflow（Rollout 工作流：RLVR / MultiTurn / Vision + Agent SDK 集成）
│
├── experimental（实验性功能）
│   ├── engine（Archon 训练引擎：第三代并行引擎）
│   ├── models/archon（Archon 模型：Qwen2/3/3.5 + MoE + FP8）
│   ├── inference_service（V2 推理微服务架构）
│   ├── training_service（V2 训练微服务架构）
│   ├── agent_service（Agent 微服务）
│   ├── weight_update（AWEX 权重同步服务）
│   ├── openai（OpenAI 兼容 API 层 + 代理服务器）
│   ├── camel（CAMEL 框架集成）
│   └── workflow（V2 多轮工作流）
│
└── tools（开发工具：性能分析、安装验证、CI 检查）
```

## 5. 关键架构决策

1. **异步 Rollout 与训练解耦** — Rollout（推理生成）和训练在不同 Worker 上独立运行，通过 `WorkflowExecutor` 的
   submit/wait 异步模式衔接。原因：推理和训练的 GPU 利用模式截然不同（推理是 memory-bound，训练是
   compute-bound），解耦后可以独立扩缩容并最大化集群利用率。

1. **三代引擎并存而非替换** — FSDP2、Megatron、Archon 三种训练引擎共存，均实现相同的 `TrainEngine` ABC。原因：FSDP2
   适合中小模型快速迭代，Megatron 适合大模型流水线并行，Archon 是最新一代支持更灵活的多维并行（dp_shard/tp/cp/ep/etp）和 FP8
   训练。三者面向不同的模型规模和并行需求。

1. **Controller 作为 SPMD 协调者而非中心化调度** — `TrainController` 和 `RolloutController` 运行在每个
   Worker 内部，通过 RPC 调用引擎方法并聚合统计。原因：避免中心化调度器成为瓶颈，每个 Worker 自主管理本地引擎，Controller 只负责跨
   Worker 的数据分发和统计聚合。

1. **Lazy Import 模式贯穿全局** — 几乎所有 `__init__.py` 使用 `_LAZY_IMPORTS` + `__getattr__`
   模式延迟导入。原因：框架依赖重型 GPU 库（torch、megatron、sglang、vllm），启动时全量导入会显著拖慢 CLI 响应和非 GPU 场景的使用体验。

1. **Guard 进程隔离引擎生命周期** — 每个 Worker 通过 Guard 进程（FastAPI 服务）管理引擎创建/调用/销毁。原因：引擎崩溃不会导致整个
   Worker 进程退出，Guard 可以安全地 fork 子进程并清理资源。

1. **权重更新双路径：磁盘 vs 分布式** — `update_weights_from_disk`
   通过共享文件系统传输权重，`update_weights_from_distributed` 通过 NCCL
   直接传输。原因：磁盘路径简单但慢，适合低频更新；分布式路径快但需要建立 NCCL 组，适合高频异步更新。V2 架构引入 AWEX（Asynchronous Weight
   Exchange）进一步优化。

1. **V1/V2 架构共存过渡** — V1 使用 SPMD + RPC（Guard 进程），V2 使用微服务 +
   HTTP（Gateway/Router/Worker）。两套架构通过共享的 `TrainEngine`/`InferenceEngine` ABC 保持兼容。原因：V2
   的微服务架构更适合弹性部署和独立扩缩容，但 V1 的 SPMD 模式在单集群场景下开销更低。

1. **序列打包（Sequence Packing）贯穿数据管线** — 从数据加载到引擎前向，使用 `seqpack` 的 FFD/KK 算法将变长序列紧密打包，避免
   padding 浪费。原因：LLM 训练中输入长度差异巨大，传统 padding 在长短序列混合时浪费 50%+ 的计算。

1. **树注意力（Tree Attention）支持分支推理** — 通过 Trie 数据结构构建共享前缀的注意力掩码，支持树形推理（如
   Best-of-N、MCTS）。原因：多个候选回复共享相同前缀时，树注意力避免重复计算前缀部分的 KV cache。

1. **统一的工作流抽象** — `RolloutWorkflow.arun_episode()` 是单一的 rollout
   入口，支持单轮对话、多轮对话、视觉推理、Agent 工作流（OpenAI/Anthropic/LangChain SDK）。原因：不同 RL 场景的 rollout
   逻辑差异巨大，但训练侧（prepare_batch → train_batch）可以保持统一。

1. **平台抽象层屏蔽硬件差异** — `Platform` ABC 抽象了 CUDA/NPU/CPU 的设备管理（内存查询、同步、事件）。原因：支持华为 NPU（昇腾）等非
   NVIDIA 硬件，避免硬编码 `torch.cuda`。

1. **MathVerify 单例进程池** — 数学奖励函数通过 `AsyncRewardWrapper` 在独立进程池中执行，自动恢复崩溃的工作进程。原因：数学验证（如
   sympy）可能触发无限循环或段错误，隔离在子进程中防止影响主训练循环。

## 6. 分层隔离模型

```
┌─────────────────────────────────────────────────────────────┐
│                    用户层（Trainer API）                      │
│  PPOTrainer / SFTTrainer / DPOTrainer / RWTrainer           │
│  输入：ExperimentConfig → 输出：训练统计 + 检查点             │
├─────────────────────────────────────────────────────────────┤
│                    编排层（Controller + Workflow）            │
│  RolloutController / TrainController / WorkflowExecutor     │
│  输入：引擎引用 + 数据 → 输出：轨迹 + 聚合统计               │
├─────────────────────────────────────────────────────────────┤
│                    引擎层（Engine ABC 实现）                  │
│  FSDPEngine / MegatronEngine / ArchonEngine                 │
│  RemoteSGLangEngine / RemotevLLMEngine                      │
│  输入：微批列表 + 损失函数 → 输出：logprobs / loss / 梯度    │
├─────────────────────────────────────────────────────────────┤
│                    模型层（Model + Parallelism）              │
│  HF Transformers / Megatron-Core / Archon 模型               │
│  Ulysses SP / Tree Attention / Expert Parallel               │
│  输入：token ids + attention mask → 输出：logits             │
├─────────────────────────────────────────────────────────────┤
│                    基础设施层（Infra）                        │
│  Scheduler / Launcher / RPC / Platform / DataService         │
│  输入：集群配置 → 输出：Worker 进程 + 通信通道               │
└─────────────────────────────────────────────────────────────┘
```

数据在层间流动时的变换：

| 数据     | 用户层                | 编排层                  | 引擎层                       | 模型层            |
| -------- | --------------------- | ----------------------- | ---------------------------- | ----------------- |
| 训练样本 | `dict[str, Any]` 列表 | 轨迹字典（含 logprobs） | `MicroBatchList`             | PackedInputs 张量 |
| 权重     | 配置路径              | `WeightUpdateMeta`      | FSDP ShardedState / MCore SD | `nn.Parameter`    |
| 统计     | StatsLogger 输出      | `dict[str, float]` 聚合 | `stats_tracker` 原始值       | 损失张量标量      |

## 7. 已知架构注意事项

| #   | 问题描述                                                             | 位置                                                  | 严重性 | 影响                                 |
| --- | -------------------------------------------------------------------- | ----------------------------------------------------- | ------ | ------------------------------------ |
| 1   | V1/V2 架构并存导致部分控制器有两套实现（Controller vs ControllerV2） | `trainer/ppo/actor.py`, `trainer/sft/lm_engine.py` 等 | 中     | 维护成本增加，新贡献者需理解两套路径 |
| 2   | `cli_args.py` 单文件 3047 行，包含 40+ 个 dataclass                  | `areal/api/cli_args.py`                               | 中     | 配置层次难以一目了然，修改需谨慎     |
| 3   | 测试集中在 `tests/` 而非与源码共置                                   | `tests/`                                              | 低     | 测试与源码的映射关系不够直观         |
| 4   | `experimental/` 目录占总代码量 33%（35,226/107,052 行）              | `areal/experimental/`                                 | 低     | 实验性功能规模庞大，API 稳定性未定   |
| 5   | `AgentWorkflow` 已废弃但仍保留                                       | `areal/api/workflow_api.py:63`                        | 低     | 向后兼容保留，新代码不应使用         |
