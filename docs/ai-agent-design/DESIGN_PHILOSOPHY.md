# Project Design Philosophy

> 目的：给后续 Claude Code / Codex / 人类贡献者提供可执行的 AReaL 设计边界说明。本文基于源码、测试、文档、CI、GitHub
> PR/Issue/review comments 与高频 contributor 提交历史；不是通用设计模式清单。

## 1. 摘要

- AReaL 的核心边界是 **API/config contract + trainer orchestration + backend adapters + async
  rollout workflow**，不是单一 monolithic trainer。
- `areal/api` 是稳定契约层：`TrainEngine`、`InferenceEngine`、`Scheduler`、`RolloutWorkflow`、IO
  structs 与 config dataclasses 是跨模块协作点。
- 新功能优先通过 **配置对象、backend adapter、workflow/reward/dataset 插件点** 扩展；不要绕过 public API 直接改
  trainer 内部时序。
- Trainer 负责 orchestration：engine 创建、rollout、logprob/advantage、update、weight
  sync、checkpoint/eval；后端实现留在 `engine/` 或 `experimental/engine/`。
- 异步 rollout 的不变量是：`arun_episode()` 非阻塞、输出标准 tensor dict、prompt token version 为 `-1`、生成
  token 携带行为策略版本。
- Weight update 必须走 `WeightUpdateMeta`，并保持 pause rollout → update weights → set version
  → resume 的时序。
- 配置是架构入口：新增配置必须在 `areal/api/cli_args.py` dataclass 中有默认值、校验和 CLI 文档更新。
- 分布式代码的首要风险是 collective 顺序、process group、DeviceMesh 维度和资源生命周期；不要猜集群配置。
- 维护者偏好小而可验证的 PR、早开 draft、明确 tests/docs/compatibility；对标准 OpenAI/Anthropic
  API、无证明的复杂优化、WIP 大 PR、过度抽象容忍度低。
- `experimental/` 是服务化、Archon、weight update 等新能力孵化区；不要把 experimental 细节当作稳定 public API。

## 2. 项目目标与非目标

### 目标

结论级别：明确事实

说明：AReaL 是面向 LLM agent applications 的大规模异步 RL 系统，核心能力包括 agentic RL、online
RL、异步训练、FSDP/Megatron/Archon 训练后端与 SGLang/vLLM 推理后端。

证据：

- 文档：`README.md:15-39` 描述项目定位和 agentic/online/asynchronous RL 能力。
- 构建：`pyproject.toml:5-41` 描述包名、关键词和 distributed/RL/LLM training 目标。
- 源码：`areal/api/engine_api.py:32`、`areal/api/engine_api.py:547` 分别定义训练与推理后端契约。

对后续开发的要求：

- 新能力必须说明它服务于训练、推理、workflow、调度、数据、reward 或实验服务中的哪条主线。
- 与 agent workflow 或 online RL 相关的变更必须说明是否影响异步 rollout、grouping、version/staleness。

### 非目标

结论级别：强推断

说明：AReaL 不追求在核心路径中重写外部标准 API，也不鼓励为了单个后端污染公共流程。

证据：

- Issue：`#1304` maintainer comment 明确建议 group_size 语义只在 new inference_service controller
  实现，并说 “No, we should avoid modifying these standard APIs.”
- PR review：`#1162` maintainer comment 认为 monkey patches hard to read/maintain，建议
  compositional bridge 而不是对 SGLang 代码打散补丁。
- 源码：`areal/infra/remote_inf_engine.py:125`、`areal/engine/sglang_remote.py:40`、`areal/engine/vllm_remote.py:40`
  将公共 remote inference facade 与 backend-specific adapter 分离。

对后续开发的要求：

- 不要为 AReaL 特有语义修改 OpenAI Responses API / Anthropic Messages API 的标准形状。
- 后端差异放入 adapter / bridge；不要把单后端 hack 放进 `RemoteInfEngine` 主流程。

## 3. 核心设计哲学

### 稳定契约优先于局部重写

结论级别：明确事实

说明：训练、推理、workflow、scheduler 都通过抽象 API 连接，核心协议比单个实现更重要。

证据：

- 源码：`areal/api/engine_api.py:32` 定义 `TrainEngine`；`areal/api/engine_api.py:363-398` 规定
  `train_batch()` 支持 list-first trajectory batch 且保留 dict backward compatibility。
- 源码：`areal/api/engine_api.py:547-633` 定义 `InferenceEngine` 与 `async agenerate()`。
- 源码：`areal/api/scheduler_api.py:43-55` 定义
  `Scheduler`，`areal/api/scheduler_api.py:181-193` 规定 remote worker 上通过 import path 创建
  engine。
- PR：`#1150 refactor(infra): standardize list-first trajectory batch dispatch` 修改 engine
  API 与 FSDP/Megatron/Archon 实现以统一 batch contract。

对后续开发的要求：

- 新后端先实现 API contract，再接 trainer 或 controller。
- 修改 contract 时必须同步所有实现、测试和文档；不能只改单个后端。

### Trainer 是编排层，不是后端实现层

结论级别：强推断

说明：`PPOTrainer` 负责组合 actor/critic/ref/rollout、训练循环、weight sync、checkpoint/eval/stats；具体
FSDP/Megatron/Archon 逻辑留在 engine。

证据：

- 源码：`areal/trainer/rl_trainer.py:102` 定义
  `PPOTrainer`；`areal/trainer/rl_trainer.py:301-339` 构造
  `WeightUpdateMeta`；`areal/trainer/rl_trainer.py:702-721` 编排 pause/update/set version。
- 源码：`areal/engine/fsdp_engine.py:218`、`areal/engine/megatron_engine.py:141`、`areal/experimental/engine/archon_engine.py:147`
  分别实现后端。
- PR：`#1157 feat(infra): allow colocation with offloading and disk weight updates` 同时修改
  trainer/controller/engine/RPC，说明 orchestration 和 backend 能力分层但协议联动。

对后续开发的要求：

- 不要把 backend-specific HTTP payload、FSDP/Megatron tensor 操作、Archon DeviceMesh 逻辑写进
  trainer。
- Trainer 改动必须说明对 rollout、update、save/recover、eval 时序的影响。

### 配置即架构入口

结论级别：明确事实

说明：AReaL 使用 dataclass + Hydra/OmegaConf + YAML/CLI override 组合系统；后端与并行策略通过 per-engine
backend 字符串表达。

证据：

- 源码：`areal/api/cli_args.py:1042` `TrainEngineConfig`；`areal/api/cli_args.py:1967`
  `InferenceEngineConfig`；`areal/api/cli_args.py:2571` `BaseExperimentConfig`。
- 源码：`areal/api/cli_args.py:2790-2868` `load_expr_config()` 加载 YAML 并 merge structured
  config，同时对旧 key 抛迁移错误。
- 测试：`tests/test_allocation_mode.py:156-180` 覆盖 per-engine backend / allocation parsing。
- Issue：`#1044 refactor(api): migrate allocation_mode to per-engine backend fields` 说明从
  monolithic `allocation_mode` 迁移到 explicit per-engine backend，并保留 backward
  compatibility shim。

对后续开发的要求：

- 新配置项必须有默认值、类型、docstring/说明、`__post_init__` 校验；用户可见配置改动同步 `docs/en/cli_reference.md` 与
  `docs/zh/cli_reference.md`。
- 不要用隐式环境变量或硬编码路径替代 `cli_args.py` 的结构化配置。

### 组合优先，少用深继承

结论级别：强推断

说明：后端实现通过 facade/adapter/strategy 组合；算法能力常由 engine subclass 组合 PPO/SFT/RW/DPO
子组件，而不是无限继承层级。

证据：

- 源码：`areal/engine/fsdp_engine.py:1935` `FSDPPPOActor` 组合
  `PPOActor`；`areal/engine/megatron_engine.py:1882` `MegatronPPOActor` 采用相同模式。
- 源码：`areal/infra/remote_inf_engine.py:327` remote facade 组合
  backend；`areal/engine/sglang_remote.py:40` 与 `areal/engine/vllm_remote.py:40` 分别实现
  backend adapter。
- PR review：`#1162` maintainer 建议 compositional bridge 替代难维护 monkey patch。
- 项目规范：`AGENTS.md` 明确“Composition over inheritance -- keep hierarchies \<= 2 levels”。

对后续开发的要求：

- 先找已有 adapter、bridge、utility、registry；不要引入平行抽象层。
- 若必须新增抽象，PR 设计说明应列出被替代的重复逻辑和未选择方案。

### 异步吞吐与策略一致性同等重要

结论级别：明确事实

说明：AReaL 使用异步 rollout 提升吞吐，但通过 versions、staleness、pause/resume 和 `WeightUpdateMeta`
保持训练语义。

证据：

- 源码：`areal/workflow/rlvr.py:164-167` 与 `areal/workflow/multi_turn.py:105-108` 构造 prompt
  `-1`、response `output_versions`。
- 源码：`areal/infra/remote_inf_engine.py:838` 在 generation response 中记录当前 version。
- 源码：`areal/api/io_struct.py:166-270` `WeightUpdateMeta` 支持
  disk/xccl/awex、LoRA、versioned path。
- 源码：`areal/infra/staleness_manager.py:20-79` 根据 version/capacity 控制异步消费。
- 文档：`docs/en/algorithms/async.md:7-55` 说明 async RL 与 off-policy/staleness。

对后续开发的要求：

- Workflow 不得丢失 `versions/logprobs/rewards/loss_mask` 等训练需要字段。
- Weight sync 相关 PR 必须同时验证训练端、推理端、pause/resume、version、LoRA 与 disk/xccl/awex 路径。

### 简单可验证优先于未证明的复杂优化

结论级别：强推断

说明：维护者倾向先合入正确、可测、生命周期清晰的实现；复杂 IPC、LRU、monkey patch、全局环境修改需要证明收益和失败路径。

证据：

- Issue：`#1117` maintainer 对 RTensor optimization 表示 shard_id direct indexing 足够，LRU
  不必要，IPC path 除非成为瓶颈才做。
- PR：`#1294 perf(infra)` 通过 shared background thread 优化 controller 初始化，但 review 关注
  initialization guard / cleanup 一致性。
- PR：`#1157` maintainer comment 拒绝 `update_weight_from_tensor` 方向，理由是内部测试性能太慢，倾向 awex
  CUDA IPC。
- PR：`#1256` closed WIP sandbox review 风险包括全局 env 线程安全、sandbox pool 状态管理。

对后续开发的要求：

- 性能优化 PR 必须给出瓶颈、测试或 benchmark/log；不要只凭“更高级”引入复杂机制。
- 引入并发、IPC、sandbox、cache、shell/SSH 路径时必须写清异常清理和安全边界。

### 兼容性与文档同步是维护者审查重点

结论级别：明确事实

说明：用户 YAML、CLI docs、双 pyproject/lock、双语文档、pre-commit 都是项目交付边界。

证据：

- 文档：`CONTRIBUTING.md:106-115` 要求 pre-commit 与 conventional
  commits；`CONTRIBUTING.md:128-164` 要求测试 marker；`CONTRIBUTING.md:80-86` 要求文档中英文和
  `./docs/build_all.sh`。
- CI：`.github/workflows/pre-commit.yml:30-48` 运行
  pre-commit；`.github/workflows/test-areal.yml:329-353` 按 marker 选择 tests。
- 配置：`.pre-commit-config.yaml:67-101` 包含 pyproject consistency、uv lock、CLI docs
  generation hooks。
- PR：`#1300` 添加 Megatron save option 同时修改
  `areal/api/cli_args.py`、engine、`docs/en/cli_reference.md`、`docs/zh/cli_reference.md`。

对后续开发的要求：

- 用户可见行为必须同步 docs/en + docs/zh。
- 修改依赖必须同步 `pyproject.toml`、必要时 `pyproject.vllm.toml` 与 locks。

## 4. 架构总览

```text
User YAML / CLI
    |
    v
areal/api/cli_args.py  ---->  areal/api/* contracts
    |                         - TrainEngine / InferenceEngine
    |                         - Scheduler / RolloutWorkflow
    |                         - ModelRequest/Response, WeightUpdateMeta
    v
Trainer layer (areal/trainer)
    - PPO/SFT/RW/DPO orchestration
    - rollout -> logp/advantage -> train -> weight sync -> save/eval
    |
    +--> Workflow extension layer (areal/workflow, examples/*)
    |       async arun_episode(engine, data) -> tensor dict / interaction / None
    |
    +--> Dataset / Reward extension layer (areal/dataset, areal/reward)
    |
    +--> Training backend adapters (areal/engine, areal/experimental/engine)
    |       FSDP / Megatron / Archon implement TrainEngine
    |
    +--> Inference backend adapters (areal/engine + areal/infra/remote_inf_engine.py)
    |       SGLang / vLLM implement backend-specific payload + weight update endpoints
    |
    +--> Infra layer (areal/infra)
            launcher / scheduler / controller / RPC / workflow_executor / staleness
```

依赖方向（推荐）：`api` 契约向下被 `trainer`、`engine`、`workflow`、`infra` 使用；`trainer`
编排但不实现后端细节；`workflow/dataset/reward` 是用户扩展层；`experimental` 可依赖核心契约但不应反向污染稳定 public API。

## 5. 模块边界

## 架构边界

| 边界                          | 允许                                                                              | 禁止                                                  | 证据                                                                                                              |
| ----------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `areal/api` 契约层            | 添加向后兼容 config 字段、抽象 API 方法需同步所有实现                             | 删除/重命名 public config/API；把后端私有逻辑放进 API | 源码：`areal/api/engine_api.py:32`, `areal/api/scheduler_api.py:43`; PR：`#1150`, `#1275`                         |
| `areal/trainer` orchestration | 编排 rollout、loss、update、weight sync、save/eval                                | 写 FSDP/Megatron/Archon 内部 tensor/HTTP payload      | 源码：`areal/trainer/rl_trainer.py:702-721`; 后端：`areal/engine/fsdp_engine.py:218`                              |
| `areal/workflow` rollout 扩展 | 实现 async `arun_episode()`，调用 `InferenceEngine.agenerate()`，打包 tensor dict | 直接训练、同步权重、阻塞 I/O、重复初始化大对象        | 源码：`areal/api/workflow_api.py:14`; 文档：`docs/en/reference/rollout_workflow.md:47-57`                         |
| `areal/engine` 生产后端       | FSDP/Megatron/SGLang/vLLM adapters，实现 API contract                             | 定义业务数据集或 reward 语义                          | 源码：`areal/engine/__init__.py:3-50`; PR：`#1301`, `#1281`                                                       |
| `areal/experimental`          | 新服务、Archon、weight update 原型；配 experimental tests                         | 当作稳定 public API 默认依赖；无测试推广到核心        | 文档：`docs/en/tutorial/archon.md:5`; Issue：`#1302` 建议从 `experimental/weight_update` 入手                     |
| Scheduler/launcher            | 管理 worker lifecycle、resource、RPC、remote engine creation                      | 实现算法 loss 或训练 batch                            | 源码：`areal/api/scheduler_api.py:43-55`, `areal/api/scheduler_api.py:181-193`; Issue：`#1302` K8S scheduler 指引 |
| Weight update 协议            | 通过 `WeightUpdateMeta` 扩展 disk/xccl/awex/LoRA/version                          | 私自定义训练/推理两端不一致协议                       | 源码：`areal/api/io_struct.py:166-270`; PR：`#1157`, `#1162`                                                      |
| Public facade                 | 通过 `__all__` / lazy import 暴露稳定入口                                         | 绕过 facade 依赖深层 internal 路径作为用户 API        | 源码：`areal/__init__.py:17-43`, `areal/api/__init__.py:3-68`, `areal/engine/__init__.py:3-50`                    |
| 测试/CI                       | pytest marker、torchrun integration、pre-commit、docs build                       | 慢测不标记、>2 GPU 不 skip、跳过 pre-commit           | 文档：`CONTRIBUTING.md:119-164`; CI：`.github/workflows/test-areal.yml:329-353`                                   |

### Public API / internal API 判定

结论级别：强推断

- Public API：`areal/__init__.py` 导出的 trainer/controller/platform；`areal/api/__init__.py`
  导出的 contracts/IO structs；`areal/engine/__init__.py`
  导出的生产后端；`areal/workflow/__init__.py` 导出的内置 workflow；公开 docs 和 examples 使用的 config
  keys。
- Internal / unstable：`areal/experimental/**`、后端内部 helper、`infra/rpc` 细节、`models/*`
  具体转换逻辑；这些可以演进，但仍需测试与迁移说明。
- 未知：项目没有单独 API stability policy；此处基于 `__all__`、docs/examples 和 maintainer comments 推断。

## 6. 已识别的设计模式

### Facade / Public Surface with Lazy Import

结论级别：明确事实

出现位置：

- `areal/__init__.py`
- `areal/api/__init__.py`
- `areal/engine/__init__.py`
- `areal/workflow/__init__.py`

解决的问题：限制用户可见 API、避免重依赖导入成本、为内部重构保留空间。

为什么这是项目偏好的方式：README/examples 通过 package-level imports 和字符串路径使用公开入口；PR `#918/#919` 在移动
infra/launcher/scheduler 后同步导入路径，说明 facade/backward compatibility 重要。

后续开发如何遵循：

- 新稳定 API 需要明确是否加入 `__all__` / docs / examples。
- 内部 helper 不应被文档推荐为用户入口。

不应该怎么用：不要让用户示例依赖 `experimental` 深层私有路径，除非明确标注 experimental。

证据：

- 源码：`areal/__init__.py:17-43`; `areal/api/__init__.py:3-68`;
  `areal/engine/__init__.py:3-50`; `areal/workflow/__init__.py:3-28`。
- PR：`#918 refactor(infra): move scheduler and rpc modules under areal/infra`;
  `#919 refactor(infra): move launcher modules to infra/launcher subpackage`。

### Abstract Interface + Adapter

结论级别：明确事实

出现位置：

- `areal/api/engine_api.py::TrainEngine`
- `areal/api/engine_api.py::InferenceEngine`
- `areal/api/scheduler_api.py::Scheduler`
- `areal/engine/sglang_remote.py::SGLangBackend`
- `areal/engine/vllm_remote.py::VLLMBackend`

解决的问题：训练后端、推理后端、调度器可替换，同时维持 trainer/controller 调用稳定。

后续开发如何遵循：

- 新 training backend 实现 `TrainEngine`；新 inference backend 优先实现 backend adapter 并复用
  `RemoteInfEngine`。
- 新 scheduler 遵循 `Scheduler` API，镜像 `areal/infra/scheduler`，复用 HTTP guards。

不应该怎么用：不要复制 `RemoteInfEngine` 主流程；不要在 scheduler 中加入算法逻辑。

证据：

- 源码：`areal/api/engine_api.py:32-398`, `areal/api/engine_api.py:547-633`,
  `areal/api/scheduler_api.py:43-193`。
- Issue：`#1302` maintainer 建议 Kubernetes scheduler follow `areal/api/scheduler_api.py`
  and mirror `areal/infra/scheduler`。
- PR review：`#1162` maintainer 建议 compositional bridge 替代 monkey patch。

### Strategy / Configuration Object

结论级别：明确事实

出现位置：

- `areal/api/alloc_mode.py::ParallelStrategy`
- `areal/api/cli_args.py::*Config`
- `examples/math/gsm8k_grpo.yaml`

解决的问题：将 backend 与 5D 并行策略（DP/TP/PP/CP/EP/ETP）显式化，并让 YAML/CLI 驱动实验组合。

后续开发如何遵循：

- 新配置字段放入 dataclass；通过 `__post_init__` fail fast；更新 CLI docs。
- 新并行模式或 backend 字符串必须有 parser/test/docs。

不应该怎么用：不要继续使用旧 `allocation_mode` 思维；不要省略 backend 前缀或隐式推断资源。

证据：

- 源码：`areal/api/alloc_mode.py:32-160`; `areal/api/cli_args.py:1042`, `1967`, `2571`,
  `2790-2868`。
- 测试：`tests/test_allocation_mode.py:156-180`。
- Issue/PR：`#1044 refactor(api): migrate allocation_mode to per-engine backend fields`。

### Pipeline / Orchestrator

结论级别：强推断

出现位置：

- `areal/trainer/rl_trainer.py::PPOTrainer`
- `areal/infra/workflow_executor.py`
- `areal/infra/staleness_manager.py`

解决的问题：将异步 rollout、训练 batch、advantage/loss、weight sync、checkpoint/eval 串成可控流水线。

后续开发如何遵循：

- 修改训练时序前说明输入/输出队列、staleness、version、clear_batches 的变化。
- controller/executor 生命周期改动必须覆盖异常路径。

不应该怎么用：不要在 workflow 或 backend adapter 中偷跑 trainer step；不要打乱 pause/update/set
version/resume。

证据：

- 源码：`areal/trainer/rl_trainer.py:537`, `areal/trainer/rl_trainer.py:702-721`;
  `areal/infra/workflow_executor.py:262`, `359`;
  `areal/infra/staleness_manager.py:20-79`。
- PR：`#1294` pipeline controller initialization；`#1282` RTensor buffer drain across
  consumers。

### Plugin by Import String

结论级别：明确事实

出现位置：

- `areal/utils/dynamic_import.py::import_from_string`
- `examples/math/gsm8k_rl.py`
- `areal/workflow/rlvr.py`

解决的问题：让用户通过字符串路径接入 workflow、reward、agent workflow，避免注册中心复杂度。

后续开发如何遵循：

- 自定义 workflow/reward 保持可 import；错误信息要包含 import path。
- 文档示例给出完整 module path。

不应该怎么用：不要动态导入不可信输入；不要在 import side effect 中启动重资源。

证据：

- 源码：`areal/utils/dynamic_import.py:7-37`; `examples/math/gsm8k_rl.py:24-43`;
  `areal/workflow/rlvr.py:75-79`。

### Registry / Model Spec

结论级别：强推断

出现位置：

- `areal/models/mcore/registry.py`
- `areal/experimental/models/archon/model_spec.py`

解决的问题：模型 architecture 与 config/load/save/state-dict adapter 的扩展映射。

后续开发如何遵循：

- Megatron-Core 新模型优先扩展 `mcore` registry/load/save 路径。
- Archon 新模型优先走 `ModelSpec` / `register_model_spec` / `state_dict_adapter`。

不应该怎么用：不要在 engine 主循环里写模型 architecture 大分支；不要只支持 load 不支持 save/parity tests。

证据：

- 源码：`areal/models/mcore/registry.py:105-156`;
  `areal/experimental/models/archon/model_spec.py:85-137`。
- PR：`#1281` Qwen2.5-VL Megatron support 修改
  `areal/models/mcore/registry.py`、`hf_load.py` 与 tests；`#1301` 增加 Qwen3-VL dense/MoE
  load/save 和 distributed tests。

### Async Wrapper / Callback-like Reward Strategy

结论级别：明确事实

出现位置：

- `areal/api/reward_api.py::AsyncRewardWrapper`
- `areal/api/reward_api.py::RewardFunction`
- `areal/workflow/rlvr.py`

解决的问题：reward 可由用户函数提供，同时允许同步 reward 移出 async hot path。

后续开发如何遵循：

- CPU-heavy reward 用 `AsyncRewardWrapper` 或异步实现。
- Reward 不访问 trainer 状态；只基于 prompt/response/label 等 task data。

不应该怎么用：不要在 `arun_episode()` 中直接执行长时间同步 reward。

证据：

- 源码：`areal/api/reward_api.py:41-60`, `areal/api/reward_api.py:63-140`;
  `areal/workflow/rlvr.py:129-143`。
- 文档：`docs/en/best_practices/workflow.md:10-72`。

### Controller / Worker / RPC Service Layer

结论级别：强推断

出现位置：

- `areal/infra/scheduler/local.py`
- `areal/infra/rpc/guard/*`
- `areal/experimental/inference_service/controller/*`
- `areal/experimental/training_service/controller/*`

解决的问题：跨 Local/Ray/Slurm 和 service controller 的 worker 生命周期、engine creation、guarded
HTTP/RPC 管理。

后续开发如何遵循：

- 新 service/controller 复用 guard、scheduler、controller patterns。
- 生命周期改动必须覆盖 start/ready/error/teardown/cleanup。

不应该怎么用：不要为每个 service 复制一套 RPC server；不要绕过 guards 直接拼 HTTP。

证据：

- PR：`#1126 refactor(infra): decompose rpc_server into shared guard + blueprints`。
- PR：`#1265 refactor(service): rename service controllers and unify service controller configs`。
- Issue：`#1302` K8S scheduler 指引要求复用 HTTP guards。

## 7. gh CLI 变更脉络分析

### 执行的 gh/git 命令

- `gh repo view --json name,owner,description,defaultBranchRef,repositoryTopics,licenseInfo`
- `gh pr list --state merged --limit 100 --json number,title,author,mergedAt,labels,files,additions,deletions`
- `gh pr list --state closed --limit 100 --json number,title,author,closedAt,labels,comments,reviews`
- `gh pr view 1309/1307/1301/1294/1282/1281/1280/1275/1157/1150/1126/970/919/918/1300/1162 --json ...`
- `gh issue list --state all --limit 100 --json number,title,author,createdAt,closedAt,labels,comments`
- `gh issue view 1308/1304/1302/1040/1044/1025/940 --json ...`
- `git shortlog -sn --all`
- `git log --author=<AUTHOR> --stat --oneline --date=short`
- `git log --author=<AUTHOR> --name-only --pretty=format:'%h %ad %s' --date=short`

### 最近 merged PR 观察

结论级别：明确事实

- 最近 100 个 merged PR 中，`areal/`、`tests/`、`examples/`、`docs/` 是最高频改动根目录。
- 最近 PR 类型集中在 engine/infra/experimental service、VLM/Megatron、RTensor/weight
  update、CI/release/governance。
- 样本：`#1301` Qwen3-VL Megatron path 配套 load/save/tests；`#1294` controller init
  performance 配 integration tests；`#1282` RTensor leak fix 跨
  engine/trainer/controller/test；`#1162` SGLang PP 支持提供 unit/distributed tests 和多
  training engine 测试截图。

证据：

- gh CLI：`gh_pr_merged_100.json` 中 top changed roots：`areal` 533、`tests` 129、`examples`
  78、`docs` 62。
- PR：`#1301`, `#1294`, `#1282`, `#1162`。

### Closed / not-merged PR 与拒绝信号

结论级别：强推断

- `#1309` 被关闭：维护者说明 tree attention transition logprob 长度 `n` 是 intentional design，因为
  advantage/loss 假设 logprob 长度为 `n`，最后一项被忽略。
- `#1226` 被拆分：service/example 大 PR 后续拆为更小 PR。
- `#1256` WIP sandbox 暴露全局 env 线程安全、代码抽取、sandbox pool 状态管理风险。
- `#1241` LoRA checkpoint PR review 关注 shell command injection、PEFT type KeyError、state
  dict remap 效率。

对后续开发的要求：

- 不要只凭局部直觉判断 bug；先追踪上下游 contract 和测试。
- 安全/并发/生命周期类 PR 必须先写设计边界和 failure mode。
- 大 PR 应拆分为 API/config、engine/controller、tests、docs/examples。

### Issue / maintainer comments 设计偏好

结论级别：明确事实

- `#1304`：grouped online rollout 应在 client/session 或 new inference_service controller
  实现；不要修改 OpenAI/Anthropic 标准 API。
- `#1302`：K8S scheduler follow `areal/api/scheduler_api.py`、mirror
  `areal/infra/scheduler`、reuse HTTP guards、补文档；weight update 从
  `areal/experimental/weight_update` + `tests/experimental/weight_update` 开始；鼓励早开 draft
  PR。
- `#1117`：RTensor cache 优先简单 shard_id 索引；LRU/IPC 复杂化需要真实瓶颈证明。
- `#1290`：vLLM GRPO collapse 先复现和标注 hypothesis，workaround 是默认 SGLang，并从 algorithmic
  config 调整。

## 8. 高频 Contributor 设计习惯

## 高频 Contributor 设计习惯

| Contributor              | 高频修改模块                                                                               | 稳定设计习惯                                                                                          | 代表 commits / PR                                                     | 对后续开发的启发                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Wei Fu / `garrett4wade`  | `areal/experimental`, `areal/engine`, `areal/api`, `areal/infra`, docs/CI/governance       | Conventional scope 清晰；infra/service/weight update 改动常配 tests/docs；维护 API/CI/治理边界        | PR `#1307`, `#1294`, `#1157`, `#1126`, commits `a1b20b3e`, `e391814d` | 复杂 infra 改动要可拆、可测、说明 lifecycle；API/文档/CI 同步                         |
| Rongzhi Gu / `Adiactive` | Megatron engine, `areal/models/mcore`, VLM load/save, distributed tests                    | 模型支持落在 mcore registry/load/save + engine path；大功能配 CPU/unit/distributed tests              | PR `#1301`, `#1281`, `#1291`, `#1144`                                 | 新模型不要只改 forward；要覆盖 HF load/save/parity 和分布式路径                       |
| Wentai Zhang / `rchardx` | Archon, config validation, checkpoint, utils, docs                                         | 把 validation 前移到 dataclass；Archon 能力以 experimental backend 演进；重构时抽 utility 简化 engine | PR `#970`, commits `c26bea9b`, `cc2bec74`, `4f5a2944`                 | 配置改动 fail fast；Archon 变更要尊重 experimental + parity/reproducibility           |
| `nuzant` / Zhiyu Mei     | inference/agent service, OpenAI proxy, tree training, gateway/controller                   | 维护 service/controller 边界；强调既有 algorithm contract；review 会拒绝误判设计为 bug                | PR `#1136`, `#912`; Review `#1309`                                    | 修 bug 前先确认 loss/logprob/session contract；service 改动走 controller/gateway 边界 |
| `fishcrap` / Xujie Shen  | `areal/utils`, `areal/experimental`, `areal/infra`, `areal/workflow`, RTensor/OpenAI proxy | 精准 bugfix、workflow/proxy edge cases、RTensor 生命周期；倾向简化数据路径                            | commits `e97f2a0c`, `b284fc78`, PR `#1017`                            | 生命周期/empty trajectory/version buffer 要有 regression test                         |
| `guozhihao-224`          | RTensor, train_controller, trainer clear path, CLI/docs                                    | 跨 engine/trainer/controller 修资源泄漏；明确 single-controller 条件和 fan-out 清理                   | PR `#1282`, `#1134`                                                   | 跨模块 fix 要列清 upstream/downstream invariant，并加 targeted tests                  |
| `TaoZex`                 | sequence packing, SGLang PP, algorithm/engine tests                                        | 新算法/并行能力配 docs、examples、unit + torchrun tests                                               | PR `#1151`, `#1162`                                                   | 并行策略变更必须同时给 docs 和 distributed test                                       |
| `gursimar`               | LoRA, Megatron/vLLM extension                                                              | 版本化 LoRA、XCCL update、MoE LoRA；常修 adapter/registry/version routing                             | PR `#1123`, `#1159`, `#1145`                                          | LoRA/adapter 改动必须说明 versioned naming、routing 和 PP/TP shard 语义               |

## 9. 推荐扩展方式

## 推荐扩展方式

| 扩展目标          | 推荐位置                                                                                    | 推荐模式                                                                     | 必须测试                                                   | 禁止做法                                                   |
| ----------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| 新 workflow       | `areal/workflow/` 或 `examples/<task>/workflow.py`；稳定后导出 `areal/workflow/__init__.py` | 实现 `RolloutWorkflow.arun_episode()`；字符串路径插件；异步 I/O              | unit test + example one-step；必要时 `pytest.mark.asyncio` | 在 workflow 内训练、同步权重、阻塞 I/O、重复初始化大对象   |
| 新 reward         | `areal/reward/` 或 example-local reward                                                     | 函数/`AsyncRewardWrapper`；由 workflow 注入                                  | reward 单测 + workflow 集成                                | reward 访问 trainer/engine 内部状态                        |
| 新 dataset        | `areal/dataset/` 或 HuggingFace dataset transform                                           | 输出 SFT/GRPO 所需字段；保持 task data 与 reward 分离                        | dataset loader 单测，字段 schema 检查                      | 在 dataset 中计算 reward/loss 或硬编码路径                 |
| 新配置项          | `areal/api/cli_args.py`                                                                     | dataclass field + default + `__post_init__` + CLI docs                       | config parsing/unit test；`docs/generate_cli_docs.py`      | 无默认破坏旧 YAML；只改 docs/en 不改 docs/zh               |
| 新训练后端        | `areal/engine/`（稳定）或 `areal/experimental/engine/`（实验）                              | 实现 `TrainEngine`；接入 backend selection/strategy；支持 `WeightUpdateMeta` | unit + GPU/distributed/torchrun；weight update tests       | 绕过 `TrainEngine` 或私有 weight sync 协议                 |
| 新推理后端        | backend adapter + `RemoteInfEngine` 组合                                                    | 实现 payload/parse/health/pause/resume/update endpoint                       | inference engine tests + weight update + backend marker    | 复制 `RemoteInfEngine` 主流程；改标准 OpenAI/Anthropic API |
| 新 scheduler      | `areal/infra/scheduler/<backend>.py`                                                        | 实现 `Scheduler`；镜像 Local/Ray/Slurm；复用 guards                          | scheduler unit/integration；资源分配 tests                 | 猜集群配置；把算法逻辑放 scheduler                         |
| 新 Archon model   | `areal/experimental/models/archon/...`                                                      | `ModelSpec`/registry/state_dict_adapter                                      | HF parity + save/load + distributed if relevant            | 只改 model forward，不处理 checkpoint parity               |
| 新 Megatron model | `areal/models/mcore/*`                                                                      | registry + hf_load/hf_save + engine path                                     | CPU/unit + distributed torchrun + VLM/MoE parity           | engine 主循环里写 architecture 特判                        |
| 新文档            | `docs/en` + `docs/zh`                                                                       | 双语同步；`./docs/build_all.sh`                                              | docs build 或至少 mdformat/check                           | 只跑 `jupyter-book build docs/en` 作为最终验证             |
| 新依赖            | `pyproject.toml` / `pyproject.vllm.toml` + locks                                            | 最小必要；区分 sglang/vLLM CUDA variant                                      | pyproject consistency + lock + install test                | 只改一个 pyproject；引入未说明 CUDA/平台依赖               |

## 10. 不应破坏的不变量

1. **Public config compatibility**：旧 YAML key 迁移必须有清晰错误或
   shim；新字段默认向后兼容。证据：`areal/api/cli_args.py:2790-2868`, Issue `#1044`。
1. **List-first trajectory batch contract**：`TrainEngine.train_batch()` preferred format
   是 `list[dict]`，dict 仅为 backward compatibility。证据：`areal/api/engine_api.py:363-398`,
   PR `#1150`。
1. **Prompt/generated version 语义**：prompt token version 为 `-1`，generated token version
   来自 inference engine。证据：`areal/workflow/rlvr.py:164-167`,
   `areal/infra/remote_inf_engine.py:838`。
1. **Weight sync 协议统一**：disk/xccl/awex/LoRA/version 通过 `WeightUpdateMeta`
   传递。证据：`areal/api/io_struct.py:166-270`。
1. **Weight sync 时序**：pause rollout → update weights → set version →
   resume/continue。证据：`areal/trainer/rl_trainer.py:702-721`。
1. **Workflow async contract**：`arun_episode()` 非阻塞，返回 tensor
   dict/interaction/None。证据：`areal/api/workflow_api.py:14-39`,
   `docs/en/reference/rollout_workflow.md:47-57`。
1. **Distributed collective 对齐**：所有相关 rank 同序调用，使用正确 process
   group、src、ReduceOp。证据：`AGENTS.md` distributed rules,
   `areal/engine/fsdp_utils/grad.py:106-118`。
1. **DeviceMesh/parallel dims 正确**：Archon/FSDP/Megatron
   维度不可随意拼。证据：`areal/experimental/models/archon/parallel_dims.py:66-108`,
   `areal/engine/fsdp_utils/parallel.py:41-79`。
1. **外部标准 API 不被 AReaL 私有语义污染**：OpenAI/Anthropic 标准接口保持兼容。证据：Issue `#1304` maintainer
   comment。
1. **CI/test marker 语义**：slow/gpu/multi_gpu/backend-specific 测试正确标记；>2 GPU
   skip。证据：`CONTRIBUTING.md:119-164`, `.github/workflows/test-areal.yml:329-353`。
1. **依赖 variant 一致性**：SGLang/vLLM 不兼容依赖通过双 pyproject/lock
   管理。证据：`pyproject.toml:140-176`, `.pre-commit-config.yaml:67-89`, PR `#1141`。
1. **安全边界**：无硬编码 secret/path/endpoint；非本地 admin key
   必须配置。证据：`areal/api/cli_args.py:1948-1957`,
   `areal/experimental/inference_service/gateway/auth.py:43-50`。

## 11. 常见反模式

### 误判既有 contract 为 bug

表现：只看局部函数，认为 shape/version/logprob 多一项或少一项是 bug，直接改掉。

为什么不符合本项目：训练 loss/advantage/staleness 依赖跨模块 contract；局部“数学直觉”可能破坏整体语义。

维护者或历史 PR 证据：

- PR：`#1309` 关闭；review comment 说明 tree attention logprob 长度 `n` 是 intended
  design，最后一项被忽略。
- Issue：`#1308` maintainer 关闭并指向 `#1309`。

正确做法：先追踪 producer/consumer：workflow → engine eval/train → loss/advantage → tests；补
regression test 后再改。

PR 自查问题：

- 我是否找到了这个字段的所有下游使用？
- 是否存在历史测试故意覆盖当前行为？
- PR 是否说明旧行为为何不是 contract？

### 大而杂的 WIP PR

表现：一个 PR 同时改 API、controller、examples、tests、docs、sandbox/security，且未收敛设计。

为什么不符合本项目：维护者偏好小步快跑、早开 draft 但 scope 可审；大 PR 易和 roadmap 冲突。

维护者或历史 PR 证据：

- PR：`#1226` 被拆分为后续更小 PR。
- PR：`#1256` WIP sandbox 暴露 env/thread/sandbox pool 风险。
- 文档：`CONTRIBUTING.md:93-96` 建议 new features/refactoring 先 issue 或 draft PR 讨论。

正确做法：拆为 API/config、核心实现、tests、docs/examples；高风险设计先 issue/RFC。

PR 自查问题：

- 这个 PR 是否可以拆成可独立验证的 2-4 个 PR？
- 是否写明明确不做什么？

### 绕过 public API / 修改 internal 却不说明影响

表现：用户示例直接依赖 `experimental` 深层路径；新后端不实现 `TrainEngine`/`InferenceEngine`；修改 `cli_args.py`
不同步文档。

为什么不符合本项目：public facade 和 config contract 是用户稳定入口。

证据：

- 源码：`areal/__init__.py:17-43`, `areal/api/__init__.py:3-68`。
- PR：`#1300` 配置新增同步 CLI docs；`#918/#919` 大规模移动仍更新 imports/docs。

正确做法：明确 public/internal；稳定能力经 `__all__`、docs、examples 暴露。

PR 自查问题：

- 是否新增或修改 public import/config key？
- 是否需要 migration note 或 backward compatibility shim？

### 在 async workflow 中阻塞

表现：`arun_episode()` 内使用同步文件/HTTP/SDK、CPU-heavy reward、每 episode 初始化 client/model。

为什么不符合本项目：异步 rollout 是吞吐核心，阻塞会拖垮 producer/consumer pipeline。

证据：

- 源码：`areal/api/workflow_api.py:14-39`; `areal/api/reward_api.py:63-140`。
- 文档：`docs/en/best_practices/workflow.md:10-72`。

正确做法：使用 async client、`workflow_context`、`aiofiles`、`AsyncRewardWrapper`；昂贵初始化放
`__init__`。

PR 自查问题：

- 是否有 `requests`、普通 `open()`、同步 SDK 调用？
- 是否有 per-episode 重初始化？

### 私自定义 weight update / version 协议

表现：训练端和推理端分别拼路径/版本；只测 disk 不测 xccl/LoRA；忽略 pause/resume。

为什么不符合本项目：权重版本是异步 RL correctness 的核心。

证据：

- 源码：`areal/api/io_struct.py:166-270`; `areal/trainer/rl_trainer.py:702-721`。
- PR：`#1157`, `#1162`, `#1282`。

正确做法：通过 `WeightUpdateMeta.with_version()`；同时覆盖 train engine、remote
inference、controller、LoRA/PP/TP paths。

PR 自查问题：

- 是否所有消费者看到同一 version？
- 异常路径是否释放 buffer/group/session？

### 分布式 collective / DeviceMesh 侥幸实现

表现：只在 rank0 调 collective；默认 `dist.get_rank()`；DeviceMesh 维度临时拼；无 torchrun test。

为什么不符合本项目：错误通常表现为 hang/OOM/wrong results，CI 难定位。

证据：

- 项目规范：`AGENTS.md` distributed rules。
- 源码：`areal/experimental/models/archon/parallel_dims.py:66-108`;
  `areal/engine/fsdp_utils/parallel.py:41-79`。
- 测试惯例：`tests/torchrun/*`, `tests/test_megatron_engine_distributed.py`。

正确做法：所有 rank 同序，显式 group/src；加 unit + torchrun integration；硬件不足时 skip 并说明。

PR 自查问题：

- 是否每个 collective 的参与 rank 集合一致？
- 是否验证了多 GPU/多节点或说明了缺口？

### 新增依赖/文档/CLI 不完整

表现：只改 `pyproject.toml`，不改 `pyproject.vllm.toml`/lock；只改英文 docs；不跑 pre-commit。

为什么不符合本项目：SGLang/vLLM 依赖分裂、双语文档、CLI docs 是发布边界。

证据：

- 构建：`pyproject.toml:140-176`; `.pre-commit-config.yaml:67-101`。
- 文档：`CONTRIBUTING.md:80-86`, `CONTRIBUTING.md:106-115`。
- PR：`#1141 feat(ci): separate vllm and sglang pyproject.toml`。

正确做法：同步 variant、locks、CLI docs、docs/en+zh；记录未跑项。

PR 自查问题：

- 是否影响安装 matrix？
- 是否更新双语文档和 CLI reference？

## 12. PR 设计说明模板

# PR Design Explanation

## Problem

这个 PR 解决什么问题？它是 bug、feature、refactor、perf、docs、test 还是 governance？请说明用户或系统层面的失败模式。

## Scope

这个 PR 修改什么？明确不修改什么？如果涉及多个边界，请说明拆分理由。

## Existing Design Followed

遵循了项目中的哪些既有设计模式、模块边界或 contributor 习惯？

证据：

- 源码：`path/to/file.py::Class.method`
- 测试：`tests/test_x.py::test_case`
- 文档：`docs/en/...`
- PR/Issue：`#123 <title>`

## Alternatives Considered

考虑过哪些方案？为什么没有选择？例如：为什么不用 monkey patch？为什么不改标准 API？为什么不新增依赖？

## Final Design

最终设计是什么？为什么符合 AReaL 的 API/config/adapter/workflow/trainer 边界？

## Compatibility

是否影响 public API、配置 key、YAML、数据格式、错误语义、weight version、性能、安全或外部标准 API？如影响，迁移方式是什么？

## Tests

新增或修改了哪些测试？如何运行？

- Unit:
- GPU / multi-GPU:
- torchrun / distributed:
- Docs build:
- Pre-commit:

## Risk

维护者需要重点审查什么？列出未测项、硬件缺口、性能假设和安全边界。

## 13. 证据索引

### 源码

- `areal/api/engine_api.py:32` — `TrainEngine` 抽象契约。
- `areal/api/engine_api.py:363-398` — list-first `train_batch()` contract。
- `areal/api/engine_api.py:547-633` — `InferenceEngine` 与 `async agenerate()`。
- `areal/api/scheduler_api.py:43-193` — scheduler worker/engine lifecycle API。
- `areal/api/workflow_api.py:14-39` — `RolloutWorkflow.arun_episode()`。
- `areal/api/reward_api.py:41-140` — reward function 与 `AsyncRewardWrapper`。
- `areal/api/io_struct.py:166-270` — `WeightUpdateMeta` disk/xccl/awex/LoRA/version
  protocol。
- `areal/api/cli_args.py:1042`, `1967`, `2571`, `2790-2868` — config dataclasses 与
  YAML/CLI loading。
- `areal/api/alloc_mode.py:32-160` — parallel strategy / backend allocation。
- `areal/trainer/rl_trainer.py:102`, `301-339`, `537`, `702-721` — PPO trainer
  orchestration 与 weight sync 时序。
- `areal/workflow/rlvr.py:48`, `129-167` — RLVR async generation、reward、versions。
- `areal/workflow/multi_turn.py:18`, `59-108` — multi-turn workflow 与 versions。
- `areal/infra/remote_inf_engine.py:125`, `327`, `838`, `939` — remote inference
  facade、version、weight update。
- `areal/infra/workflow_executor.py:262`, `359`, `746` — async rollout queue/executor。
- `areal/infra/staleness_manager.py:20-79` — staleness/capacity control。
- `areal/engine/fsdp_engine.py:218`, `1935` — FSDP backend 与 PPO actor composition。
- `areal/engine/megatron_engine.py:141`, `1882` — Megatron backend 与 PPO actor
  composition。
- `areal/experimental/engine/archon_engine.py:147` — experimental Archon backend。
- `areal/models/mcore/registry.py:105-156` — Megatron model registry。
- `areal/experimental/models/archon/model_spec.py:85-137` — Archon model spec registry。
- `areal/utils/dynamic_import.py:7-37` — string import plugin mechanism。
- `areal/utils/logging.py:303-340` — project logger entry。

### 测试

- `tests/test_allocation_mode.py:156-180` — per-engine backend/allocation compatibility。
- `tests/test_megatron_engine_vlm.py` — VLM/Megatron model support tests。
- `tests/test_megatron_engine_vlm_distributed.py` — distributed VLM tests。
- `tests/test_rtensor.py` — RTensor buffer/lifecycle tests。
- `tests/test_sglang_pp_unit.py` 与 `tests/test_sglang_pp_distributed.py` — SGLang PP
  support tests。
- `tests/torchrun/*` — distributed entry scripts convention。

### 文档 / CI / 配置

- `README.md:15-39`, `README.md:248-287` — project goals and docs map。
- `CONTRIBUTING.md:80-96` — docs bilingual build and new feature/refactor discussion。
- `CONTRIBUTING.md:106-115` — pre-commit/conventional commits。
- `CONTRIBUTING.md:119-164` — CI `safe-to-test`, pytest markers, GPU skip。
- `.github/PULL_REQUEST_TEMPLATE.md:23-39` — checklist / breaking change details。
- `GOVERNANCE.md:37-56` — lead maintainer and approval policy。
- `.pre-commit-config.yaml:47-131` — ruff/mdformat/nbstripout/CLI docs/conventional
  commits。
- `.github/workflows/pre-commit.yml:30-48` — CI pre-commit command。
- `.github/workflows/test-areal.yml:329-353` — pytest marker selection。
- `pyproject.toml:140-176` — CUDA/SGLang/vLLM dependency split note。
- `pyproject.toml:283-333` — pytest markers and Ruff config。
- `docs/en/tutorial/archon.md:5-54` — Archon experimental backend description。
- `docs/en/reference/rollout_workflow.md:47-57` — workflow output contract。
- `docs/en/algorithms/async.md:7-55` — async RL/staleness semantics。

### PR / Issue / Review comments

- PR：`#1307 gov: enforce 2-approval merge policy on main` — governance, CODEOWNERS,
  pre-commit policy。
- PR：`#1301 feat(engine): add Qwen3-VL dense and MoE support to Megatron path` — model
  support + tests。
- PR：`#1294 perf(infra): pipeline controller initialization with background threads` —
  performance optimization + lifecycle review focus。
- PR：`#1282 fix(infra): drain RTensor _fetch_buffer on all consumer workers` — resource
  cleanup across engine/trainer/controller。
- PR：`#1281 feat(engine): add Megatron support for Qwen2.5-VL` — model
  registry/load/save/tests pattern。
- PR：`#1280 fix: apply_chat_template compatibility with transformers>=5.0` — centralized
  compatibility wrapper。
- PR：`#1275 feat(infra): add n_gpus_per_node abstract property to Scheduler API` —
  Scheduler API expansion across implementations。
- PR：`#1162 feat: support pp for Sglang` — PP weight update, tests, maintainer
  preference for compositional bridge over monkey patch。
- PR：`#1157 feat(infra): allow colocation with offloading and disk weight updates` —
  weight update/offload orchestration and performance decision against slow tensor
  update path。
- PR：`#1150 refactor(infra): standardize list-first trajectory batch dispatch` — engine
  API contract standardization。
- PR：`#1126 refactor(infra): decompose rpc_server into shared guard + blueprints` —
  shared guard/blueprint composition。
- PR：`#970 refactor(api): move validation into config __post_init__ methods` — fail-fast
  config validation。
- PR：`#919`, `#918` — infra module reorganization with import/docs updates。
- PR：`#1309 fix(tree_attn): skip transition logprob for last node in sequence` — closed
  as intended design; key anti-pattern evidence。
- Issue：`#1308` — tree_attn issue closed due to intended design。
- Issue：`#1304` — online grouping; new controller only; do not modify standard APIs。
- Issue：`#1302` — roadmap; K8S scheduler and weight update contribution guidance。
- Issue：`#1117` — RTensor optimization simplicity preference。
- Issue：`#1290` — backend-specific instability handled with
  reproduction/workaround/hypothesis。
- Issue：`#1044` — allocation mode migration to explicit per-engine backend fields。

### 高频 contributor commits / logs

- `git shortlog -sn --all` — top contributors: lichangye.lcy, Wei Fu, Wentai Zhang, 博惟,
  nuzant, fishcrap, Xujie Shen, hcy, Huawei Vancouver ICI Lab, yulangz。
- Commit：`a1b20b3e refactor(infra): move launcher modules to infra/launcher subpackage (#919)`。
- Commit：`e391814d refactor(infra): move scheduler and rpc modules under areal/infra (#918)`。
- Commit：`84eaef12 refactor(api): move validation into config __post_init__ methods (#970)`。
- Commit：`e97f2a0c refactor(infra): simplify RTensor to single-shard and adopt per-trajectory list pipeline (#1017)`。
- Commit：`b48b0139 refactor: replace string literals with enums and fix logging issues (#1008)`。
- Commit：`c26bea9b refactor(archon): extract utility functions and simplify engine code (#954)`。

### 未知 / 需要维护者确认

- 未知：项目没有独立 API stability policy；本文 public API 判定基于 `__all__`、docs/examples 与历史 PR 推断。
- 弱推断：`experimental/` 全部视为 unstable 是基于目录名、Archon 文档和 issue guidance；个别 experimental
  service 可能正在成为未来主路径。
- 弱推断：依赖方向不是严格 clean architecture，存在工程实用主义横向依赖；重构前应用 import graph 再确认。
