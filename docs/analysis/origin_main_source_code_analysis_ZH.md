# AReaL 源代码分析：origin/main 自 v1.0.3 以来的变更

**分析日期**：2026-04-27 **Commit 范围**：`376ecbb8` (v1.0.3) → `b0ea0dd8` (origin/main HEAD)
**范围**：49 个非合并 commit | 27,743 行新增 | 2,548 行删除 | 321 个文件变更 | 105 个新文件

______________________________________________________________________

## 目录

1. [执行摘要](#1-executive-summary)
1. [定量概览](#2-quantitative-overview)
   - 2.1 [汇总统计](#21-summary-statistics)
   - 2.2 [贡献者分布](#22-contributor-breakdown)
   - 2.3 [目录级变更分布](#23-directory-level-change-distribution)
   - 2.4 [Commit 类型分布](#24-commit-type-distribution)
   - 2.5 [最大的 10 个 Commit](#25-top-10-largest-commits)
   - 2.6 [开发时间线](#26-development-timeline)
1. [完整 Commit 清单](#3-full-commit-inventory)
1. [功能深度分析](#4-feature-deep-dives)
   - 4.1 [Awex（异步权重交换）系统](#41-awex-async-weight-exchange-system)
   - 4.2 [Scaffolding Rollout Workflow](#42-scaffolding-rollout-workflow)
   - 4.3 [DPO（直接偏好优化）](#43-dpo-direct-preference-optimization)
   - 4.4 [拒绝采样配置](#44-rejection-sampling-configuration)
   - 4.5 [Karmarkar-Karp 分区算法](#45-karmarkar-karp-partitioning)
   - 4.6 [MoE 模型的 LoRA 支持](#46-lora-for-moe-models)
   - 4.7 [Terminal Bench 训练示例](#47-terminal-bench-training-example)
   - 4.8 [推理服务重构](#48-inference-service-refactoring)
   - 4.9 [引擎演进](#49-engine-evolution)
   - 4.10 [基础设施可靠性](#410-infrastructure-reliability)
1. [.claude AI 辅助开发工具](#5-claude-ai-assisted-development-tooling)
   - 5.1 [清单](#51-inventory)
   - 5.2 [Commands](#52-commands)
   - 5.3 [Skills](#53-skills)
   - 5.4 [Agents](#54-agents)
   - 5.5 [Rules](#55-rules)
   - 5.6 [Hooks](#56-hooks)
1. [Archon 训练框架](#6-archon-training-framework)
   - 6.1 [架构概览](#61-architecture-overview)
   - 6.2 [自 v1.0.3 以来的关键变更](#62-key-changes-since-v103)
   - 6.3 [ArchonDPOEngine](#63-archondpoengine)
1. [架构评估](#7-architectural-assessment)
   - 7.1 [设计模式](#71-design-patterns)
   - 7.2 [可扩展性评估](#72-scalability-assessment)
   - 7.3 [技术债务](#73-technical-debt)
   - 7.4 [建议](#74-recommendations)
1. [代码质量审查](#8-code-quality-review)
   - 8.1 [规范遵循情况](#81-convention-adherence)
   - 8.2 [测试覆盖率](#82-test-coverage)
   - 8.3 [按严重程度分类的发现](#83-findings-by-severity)
1. [战略方向分析](#9-strategic-direction-analysis)
1. [新文件清单](#10-new-file-inventory)

______________________________________________________________________

## 1. 执行摘要

自 v1.0.3 以来的 49 个 commit 代表了一次为期 19 天（2026 年 4 月 9 日至 27 日）的重大功能冲刺，由 18 位贡献者在 321
个文件中新增了约 25,195 行净代码。此版本以五个变革性功能为主导：

1. **Awex（异步权重交换）**：一个完整的 GPU 到 GPU 权重同步系统，支持在不同并行策略（FSDP、Megatron
   TP/PP/CP/EP、SGLang）的训练引擎和推理引擎之间进行直接 NCCL P2P 传输。

1. **Scaffolding Rollout Workflow**：一个模块化的智能体执行框架（8,286 行），支持在 rollout
   过程中进行多步骤、多工具、多轮次的智能体行为——这是本次发布中最大的单个 commit。

1. **DPO Trainer**：完整的直接偏好优化实现，包含 sigmoid 和 IPO 损失变体，支持所有三种引擎后端（FSDP、Megatron、Archon）。

1. **统一拒绝采样**：新的 `RejectionSamplingConfig` 取代了遗留的字符串类型参数，采用结构化的 dataclass 支持在多种散度指标下进行
   token/序列级别的过滤。

1. **Karmarkar-Karp 分区算法**：一种替代的序列打包算法，可提供比 First Fit Decreasing 更好的微批次平衡效果。

此外，`.claude/` 目录也得到了显著增强，新增了
commands（`/create-pr`、`/gen-commit-msg`）、skills（`commit-conventions`）以及改进的 AI
辅助开发工作流。Archon 引擎获得了 DPO 支持、可重入的 offload 上下文、即时 HCCL 预热和两阶段 teardown 以实现干净关闭。

______________________________________________________________________

## 2. 定量概览

### 2.1 汇总统计

| 指标                           | 值                                |
| ------------------------------ | --------------------------------- |
| 非合并 commit 总数             | 49                                |
| 独立贡献者数                   | 18                                |
| 变更文件数（去重）             | 321                               |
| 总新增行数                     | 27,743                            |
| 总删除行数                     | 2,548                             |
| 净新增行数                     | +25,195                           |
| 新增文件数                     | 105                               |
| 删除文件数                     | 1                                 |
| 日期范围                       | 2026-04-09 至 2026-04-27（19 天） |
| 活跃日均 commit 数             | 4.1                               |
| 每个 commit 平均变更行数       | ~619                              |
| 功能与修复比                   | 1.64:1                            |
| 测试与功能代码比               | 1.18:1                            |
| Conventional Commit 规范遵循率 | 48/49 (98%)                       |

### 2.2 贡献者分布

| 排名 | 贡献者                      | Commit 数 | 占比  |
| ---- | --------------------------- | --------- | ----- |
| 1    | Wei Fu (garrett4wade)       | 20        | 40.8% |
| 2    | xiao (Wangxiaoxiaoa)        | 4         | 8.2%  |
| 3    | Ran Yan                     | 3         | 6.1%  |
| 4    | HT-Yuan                     | 3         | 6.1%  |
| 5    | Gursimran                   | 2         | 4.1%  |
| 6    | TaoZex                      | 2         | 4.1%  |
| 7    | sitabulaixizawaluduo        | 2         | 4.1%  |
| 8    | Pratyush Sharma             | 2         | 4.1%  |
| 9–18 | 其他 10 位贡献者（各 1 个） | 11        | 22.4% |

Wei Fu 以 40.8% 的占比成为主要贡献者。其余 17 位贡献者形成了健康的长尾分布，表明这是一个活跃的开源社区，且有企业参与。

### 2.3 目录级变更分布

| 目录                    | 变更文件数 | 新增行数 | 删除行数 | 净变化  | 新增占比 |
| ----------------------- | ---------- | -------- | -------- | ------- | -------- |
| **examples/**           | 93         | 10,833   | 86       | +10,747 | 39.1%    |
| **tests/**              | 47         | 6,413    | 277      | +6,136  | 23.1%    |
| **areal/experimental/** | 70         | 5,455    | 770      | +4,685  | 19.7%    |
| **docs/**               | 13         | 1,148    | 503      | +645    | 4.1%     |
| **areal/trainer/**      | 9          | 820      | 30       | +790    | 3.0%     |
| **areal/utils/**        | 6          | 733      | 79       | +654    | 2.6%     |
| **areal/infra/**        | 17         | 657      | 171      | +486    | 2.4%     |
| **areal/engine/**       | 11         | 624      | 63       | +561    | 2.3%     |
| **areal/api/**          | 4          | 380      | 56       | +324    | 1.4%     |
| **.claude/**            | 3          | 55       | 27       | +28     | 0.2%     |
| **areal/workflow/**     | 0          | 0        | 0        | 0       | 0.0%     |

关键发现：

- **examples/** 以 39.1% 占据主导地位——由 scaffolding（8,286 行）和 Terminal Bench（1,947 行）驱动。
- **tests/** 占 23.1%，表明测试纪律良好。
- **areal/experimental/** 占 19.7%，反映了 awex 和推理服务的活跃开发。
- **areal/workflow/** 零变更——workflow 逻辑保持稳定。

### 2.4 Commit 类型分布

| 类型         | 数量 | 占比  | 描述            |
| ------------ | ---- | ----- | --------------- |
| **feat**     | 18   | 36.7% | 新功能          |
| **fix**      | 11   | 22.4% | 缺陷修复        |
| **chore**    | 10   | 20.4% | 维护/依赖更新   |
| **refactor** | 3    | 6.1%  | 代码重构        |
| **gov**      | 3    | 6.1%  | 治理/维护者更新 |
| **docs**     | 2    | 4.1%  | 文档            |
| **test**     | 1    | 2.0%  | 测试新增        |
| **perf**     | 1    | 2.0%  | 性能改进        |

**Scope 分布**（最活跃的领域）：

| Scope        | 数量 |
| ------------ | ---- |
| experimental | 6    |
| engine       | 5    |
| infra        | 4    |
| service      | 3    |
| trainer      | 2    |
| deps         | 2    |

### 2.5 最大的 10 个 Commit

| 排名   | Hash       | 新增行数   | 标题                                                       |
| ------ | ---------- | ---------- | ---------------------------------------------------------- |
| **1**  | `d37095ae` | **+8,286** | feat: add scaffolding rollout workflow (#1064)             |
| **2**  | `615d1bae` | **+4,112** | feat: add awex backend for weight update (#1214)           |
| **3**  | `628c389e` | **+2,238** | feat(service): add external model API support (#1183)      |
| **4**  | `70acd22f` | **+2,158** | feat(trainer): add dpo (#1190)                             |
| **5**  | `aeb237bd` | **+1,947** | feat(example): add Terminal Bench training example (#1224) |
| **6**  | `bc9f0098` | **+1,911** | feat(api): add unified RejectionSamplingConfig (#1088)     |
| **7**  | `8c8a8dbd` | **+1,408** | feat(utils): add Karmarkar-Karp partitioning (#1151)       |
| **8**  | `ae8c792f` | **+757**   | feat: add disk-mode weight update flow (#1237)             |
| **9**  | `8cc52ba0` | **+686**   | refactor(experimental): reuse HTTP clients (#1253)         |
| **10** | `6e69226c` | **+656**   | feat(experimental): Megatron awex TP adapter (#1239)       |

前 10 个 commit 贡献了 **24,159 行新增代码**——占总量的 87%。

### 2.6 开发时间线

| 日期    | Commit 数 | 主要活动                                                            |
| ------- | --------- | ------------------------------------------------------------------- |
| 4月9日  | 1         | FSDP 集合类型 wrap 修复                                             |
| 4月16日 | 2         | 社区会议文档、资源重组                                              |
| 4月17日 | 4         | Scaffolding workflow（8,286 行）、外部模型 API、LoRA MoE、IPv6 修复 |
| 4月19日 | 1         | Ray RPC 序列化修复                                                  |
| 4月20日 | 6         | Awex 后端（4,112 行）、RejectionSamplingConfig、KK 分区、DPO        |
| 4月21日 | 3         | 内存分析器、治理、awex 依赖升级                                     |
| 4月22日 | 5         | Megatron awex TP、磁盘模式权重更新、Terminal Bench                  |
| 4月23日 | 7         | Megatron PP/CP awex、Docker 修复、teardown 修复                     |
| 4月24日 | 4         | NPU HCCL 修复、SFT 批次测试、Megatron awex EP                       |
| 4月25日 | 3         | 推理服务重构                                                        |
| 4月26日 | 4         | 多模态张量修复、日志、offload/onload 端点                           |
| 4月27日 | 9         | Engine from_pretrained、CI 修复、LSP 文档                           |

开发速度从第 1 周（1.75 commit/天）到第 2 周及之后（5.1 commit/天）加速了 2.9 倍。

______________________________________________________________________

## 3. 完整 Commit 清单

### 按领域分类

#### (a) .claude / AI Agent 工具（2 个 commit）

| Hash       | 行数 | 标题                                                                                           |
| ---------- | ---- | ---------------------------------------------------------------------------------------------- |
| `434df57b` | +131 | gov: add maintainer (#1227) — 包含 commit-conventions skill、create-pr/gen-commit-msg commands |
| `b0ea0dd8` | +28  | docs(workflow): add LSP-first code navigation guidance                                         |

#### (b) Archon / Experimental Engine（3 个 commit）

| Hash       | 行数 | 标题                                                                            |
| ---------- | ---- | ------------------------------------------------------------------------------- |
| `50f0a0b0` | +166 | fix(engine): eagerly init HCCL subgroups to fix ref compute_logp on NPU (#1254) |
| `db2a193b` | +50  | perf(trainer): reduce redundant offload/onload transitions (#1163)              |
| `3ed3e817` | +438 | fix(infra): add two-phase teardown to prevent TCPStore race at shutdown (#1244) |

#### (c) 推理服务（6 个 commit）

| Hash       | 行数   | 标题                                                                                     |
| ---------- | ------ | ---------------------------------------------------------------------------------------- |
| `628c389e` | +2,238 | feat(service): add external model API support for inference service (#1183)              |
| `2d6ea231` | +392   | refactor(service): extract inference bridge backends into sglang/vllm submodules (#1221) |
| `8cc52ba0` | +686   | refactor(experimental): reuse HTTP clients, add response models (#1253)                  |
| `5c723ffe` | +164   | feat(experimental): add offload/onload endpoints (#1276)                                 |
| `ed06091b` | +88    | chore(experimental): suppress HTTP service logging (#1274)                               |
| `d38268e1` | +33    | feat(infra): add n_gpus_per_node abstract property to Scheduler API (#1275)              |

#### (d) 训练服务 / Awex 权重更新（8 个 commit）

| Hash       | 行数   | 标题                                                                        |
| ---------- | ------ | --------------------------------------------------------------------------- |
| `615d1bae` | +4,112 | feat: add awex backend for weight update (#1214)                            |
| `6e69226c` | +656   | feat(experimental): MegatronEngine awex TP adapter (#1239)                  |
| `ae8c792f` | +757   | feat: add disk-mode weight update flow to gateway (#1237)                   |
| `5bd7a180` | +227   | feat: Implement Megatron PP and CP with Awex (#1246)                        |
| `b7b10278` | +210   | feat(experimental): MegatronEngine awex EP adapter (#1252)                  |
| `01dab41e` | +8     | refactor(experimental): rename WeightUpdate*Adapter to Awex*Adapter (#1269) |
| `e6f3c3cb` | +235   | chore(deps): upgrade awex to 0.7.0 (#1228)                                  |
| `4629c4ef` | +18    | chore(deps): upgrade mbridge from 0.15.1 to 310e8fb (#1258)                 |

#### (e) 核心引擎 — FSDP / Megatron（6 个 commit）

| Hash       | 行数 | 标题                                                                         |
| ---------- | ---- | ---------------------------------------------------------------------------- |
| `073adbf2` | +13  | fix: FSDP initialization for set-valued wrap class names (#1187)             |
| `e5531199` | +325 | feat(engine): lora support for MoE models (#1159)                            |
| `349c6ed6` | +91  | fix(engine): avoid duplicating multimodal tensors (#1272)                    |
| `e47dc676` | +267 | feat(engine): support direct engine construction via from_pretrained (#1140) |
| `d58cca56` | +376 | feat(engine): add built-in memory profiler support (#1223)                   |
| `50f0a0b0` | +166 | fix(engine): eagerly init HCCL subgroups for NPU (#1254)                     |

#### (f) Workflow / 算法（5 个 commit）

| Hash       | 行数   | 标题                                                       |
| ---------- | ------ | ---------------------------------------------------------- |
| `d37095ae` | +8,286 | feat: add scaffolding rollout workflow (#1064)             |
| `70acd22f` | +2,158 | feat(trainer): add dpo (#1190)                             |
| `bc9f0098` | +1,911 | feat(api): add unified RejectionSamplingConfig (#1088)     |
| `8c8a8dbd` | +1,408 | feat(utils): add Karmarkar-Karp partitioning (#1151)       |
| `aeb237bd` | +1,947 | feat(example): add Terminal Bench training example (#1224) |

#### (g) 基础设施（8 个 commit）

| Hash       | 行数 | 标题                                                                         |
| ---------- | ---- | ---------------------------------------------------------------------------- |
| `3ed3e817` | +438 | fix(infra): add two-phase teardown to prevent TCPStore race (#1244)          |
| `ad622efd` | +267 | fix(infra): move data service seed to worker-level config (#1210)            |
| `e70b1934` | +123 | fix(Service): fix data service failures in IPv6-only environments (#1208)    |
| `f3d7e50a` | +51  | fix: serialize ray object refs in rpc payloads (#1198)                       |
| `e8c1e1fd` | +6   | fix: handle integer device ids in ray rpc server (#1199)                     |
| `fe91acc1` | +10  | feat: add configurable setup_timeout for data service gateway (#1263)        |
| `9fb5247d` | +27  | test(infra): regression for single-controller SFT batch partitioning (#1255) |
| `f34468c7` | +36  | feat(eval): support for running eval before training (#1232)                 |

#### (h) 治理 / 文档 / 社区（9 个 commit）

| Hash       | 行数 | 标题                                                       |
| ---------- | ---- | ---------------------------------------------------------- |
| `d4891785` | +51  | chore: move figures into assets/figures/ (#1192)           |
| `8965973b` | +2   | chore: add @CormickKneey as maintainer (#1201)             |
| `65928242` | +15  | chore: add new maintainers (#1220)                         |
| `c3ba6faa` | +17  | docs(community): add first biweekly meeting record (#1215) |
| `259b3430` | +1   | chore: update tencent meeting link (#1219)                 |
| `2aaddaec` | +14  | chore: add new maintainer (#1234)                          |
| `68cd2e5a` | +15  | gov: add maintainer (#1235)                                |
| `b499183f` | +13  | chore: update news for scaffoldings (#1236)                |
| `2851ea71` | +0   | gov: update governance (#1248)                             |

#### (i) CI / 构建 / 依赖（5 个 commit）

| Hash       | 行数 | 标题                                                                 |
| ---------- | ---- | -------------------------------------------------------------------- |
| `7fbd077c` | +388 | fix(docker): move venv out of /AReaL to avoid mount override (#1251) |
| `bcb216d6` | +2   | fix: Update CI test GCP Image (#1277)                                |
| `5d15f659` | +3   | fix: fix pre-commit CI env                                           |
| `15384e7f` | +12  | chore: add uv lock check (#1259)                                     |
| `e6f3c3cb` | +235 | chore(deps): upgrade awex to 0.7.0 (#1228)                           |

______________________________________________________________________

## 4. 功能深度分析

### 4.1 Awex（异步权重交换）系统

**本次发布中最大的跨领域功能新增。** 横跨 8 个 commit，新增约 5,500+ 行代码。

#### 问题陈述

在 AReaL 的异步训练架构中，训练引擎和推理引擎运行在不同的 GPU 集群上，使用不同的并行策略。更新后的模型权重必须从训练侧传播到推理侧，以供下一批 rollout
使用。挑战在于：两个引擎可能使用不同的分片策略（例如，训练使用 FSDP 的 DP 分片，推理使用 SGLang 的 TP 分片）。

#### 架构

Awex 通过三层设计运作：

**1. 协议层** — 两个 `@runtime_checkable Protocol` 类：

- `AwexTrainingAdapter` (`areal/experimental/weight_update/training_adapter.py`) —
  训练侧（发送权重）
- `AwexInferenceAdapter` (`areal/experimental/weight_update/inference_adapter.py`) —
  推理侧（接收权重）

两者共享相同的方法签名（`init_weight_update_group`、`execute_weight_update`、`batch_isend_irecv`、`teardown_weight_update_group`），仅在发送与接收语义上有所不同。

**2. 适配器层** — 每种引擎类型的具体实现：

- `AwexFSDPAdapter` — 提取 DTensor 分片元数据，根据 placements 计算偏移量
- `AwexMegatronAdapter` — 使用 `all_gather_param` + `convert_to_hf` 规范化为 HF 命名格式
- `AwexSGLangAdapter` — 处理 SGLang
  的融合参数命名（qkv_proj、gate_up_proj、w13_weight/w2_weight），通过拆分融合为 HF 兼容的逐专家名称

**3. HTTP 端点层** — Flask
blueprint（训练侧，`areal/experimental/training_service/worker/awex.py`）和 FastAPI
端点（推理侧，`areal/experimental/inference_service/sglang/awex.py`）通过 HTTP
暴露适配器操作。网关负责编排多步骤握手过程。

#### 权重传输协议流程

```
1. 双方 → /awex/report_parallelism     （报告并行策略）
2. 双方 → /awex/report_weight_meta      （将分片元数据发布到 KV 存储）
3. TransferPlanBuilder 计算最优的 P2P 传输计划
4. 双方 → /awex/init_weights_update_group  （建立专用 NCCL 组）
5. 训练侧：nccl_build_send_ops()  →  batch_send_recv()
   推理侧：nccl_build_recv_ops()  →  batch_send_recv()
```

#### 并行支持

| 模式   | 训练侧                                                                             | 推理侧                                                     |
| ------ | ---------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **TP** | FSDP：DTensor 分片 placements；Megatron：`all_gather_param` + `dp_replicated=True` | SGLang：TP 感知的分片偏移量                                |
| **EP** | Megatron：`mpu.get_expert_model_parallel_world_size()`                             | SGLang：每个专家的 `EP_SHARDING` / `EP_TP_SHARDING` 偏移量 |
| **PP** | 每个 PP stage 通过 `get_transformer_layer_offset` 产出其自身的层子集               | 网关按名称合并不相交的 PP stage 元数据                     |
| **CP** | 在 `RankInfo` 中报告，带有 `cp_mode="ring"`                                        | 在 dp_replicated 逻辑中与 TP 类似处理                      |

#### 流式模式与磁盘模式

- **流式模式**（默认）：直接 NCCL P2P 传输。零磁盘 I/O，最低延迟。
- **磁盘模式**（commit `ae8c792f`）：训练 worker 将参数保存到共享文件系统；推理 worker
  从该路径加载。在权重替换期间通过上下文管理器暂停/恢复生成。

#### MoE 处理

SGLang 适配器的 `_unfuse_params` 处理 SGLang 融合专家参数的复杂情况：

- `experts.w13_weight` 形状：`[num_total_experts, 2*ffn_hidden, hidden]`（每个专家的 gate+up 融合）
- `experts.w2_weight` 形状：`[num_total_experts, hidden, ffn_hidden]`（每个专家的 down）

这些参数被拆分为 HuggingFace 风格的逐专家名称，以实现跨引擎互操作性。

#### 关键文件

| 文件                                                        | 行数 | 用途                                             |
| ----------------------------------------------------------- | ---- | ------------------------------------------------ |
| `areal/experimental/weight_update/nccl_group.py`            | 225  | 跨进程通信的 NCCL 组管理                         |
| `areal/experimental/weight_update/awex/fsdp_adapter.py`     | 334  | FSDP 训练侧适配器                                |
| `areal/experimental/weight_update/awex/megatron_adapter.py` | 278  | Megatron 训练侧适配器（TP/PP/CP/EP）             |
| `areal/experimental/weight_update/awex/sglang_adapter.py`   | 404  | SGLang 推理侧适配器                              |
| `areal/experimental/weight_update/gateway/app.py`           | 589  | 网关协调服务                                     |
| `areal/experimental/training_service/worker/awex.py`        | 152  | 训练端点的 Flask blueprint                       |
| `areal/experimental/inference_service/sglang/awex.py`       | 96   | 推理端点的 FastAPI 路由                          |
| `areal/experimental/inference_service/sglang/scheduler.py`  | 293  | 组合到 SGLang Scheduler 上的 AwexSchedulerBridge |
| `tests/experimental/weight_update/test_nccl_integration.py` | 890  | NCCL 集成测试                                    |

______________________________________________________________________

### 4.2 Scaffolding Rollout Workflow

**最大的单个 commit，涵盖 31 个文件共 8,286 行。** 引入了一个模块化的智能体执行框架，支持在 rollout
过程中进行多步骤、多工具、多轮次的智能体行为。

#### "Scaffolding" 的含义

Scaffolding 指的是一个模块化的智能体执行框架，来源于 NVIDIA TensorRT-LLM 的 scaffolding 模块。它在 RL 训练循环和
rollout 生成之间提供了一个抽象层，支持复杂的智能体行为——而不仅仅是单次文本生成。

#### 核心抽象

1. **Controller** (`core/controller.py`) — 将生成 episode 逻辑定义为产出 `Task` 对象的 Python
   生成器。Controller 可以通过管道和并行分支进行组合。

1. **Worker** (`core/worker.py`) — 针对推理引擎执行任务。`SGLangWorker` 封装了 OpenAI 兼容的 completions
   API。

1. **ScaffoldingLlm** (`core/scaffolding_llm.py`) — 编排器，在独立线程中运行专用的 asyncio 事件循环，将
   controller 的任务分派给 worker。可配置并行度（`max_parallel_requests=64`）。

1. **Task** / **TaskCollection** — 用于生成请求和追踪的数据容器。

#### Controller 模式

| Controller                   | 用途                                                    |
| ---------------------------- | ------------------------------------------------------- |
| `NativeGenerationController` | 单次生成（基线）                                        |
| `NativeChatController`       | 聊天格式生成                                            |
| `MajorityVoteController`     | N 个并行生成，多数投票选择                              |
| `BestOfNController`          | N 个并行生成，基于奖励选择                              |
| `PipelineTrajectoryMaker`    | 将生成和奖励组合成管道                                  |
| `MultiTurnChatController`    | 多轮对话，可配置的反思消息                              |
| `SearchAgentController`      | 完整的工具调用智能体循环，带搜索/访问工具和 token 预算  |
| `TraceTrajectoryMaker`       | 使用 `ChatTracer` 封装任意 controller，实现逐轮信用分配 |
| `LLMJudgeController`         | 使用辅助 LLM 判断答案正确性                             |

#### 对 RL 训练的意义

Scaffolding 框架将 AReaL 的能力扩展到单轮 RLHF 之外：

- **智能体 RL**：训练可以使用工具、搜索和迭代求解的模型
- **多轮信用分配**：`ChatTracer` 支持可配置的 `reward_discount`，实现奖励在对话轮次间的反向传播
- **单轮训练**："individual" 导出模式为每一轮创建独立的 `InteractionWithTokenLogpReward` 对象，用于逐轮 PPO 更新

#### 集成方式

`ScaffoldingWorkflow` 类（`examples/scaffolding/workflow.py`）继承 `RolloutWorkflow` 并通过
scaffolding 管道实现 `agenerate()`。使用占位 logprobs（0.0），因为 actor 上的 `recompute_logprob=true`
会在 PPO 更新时重新计算精确的 logprobs。

#### 关键文件

| 文件                                           | 行数  | 用途                     |
| ---------------------------------------------- | ----- | ------------------------ |
| `examples/scaffolding/controllers.py`          | 1,070 | Controller 实现          |
| `examples/scaffolding/workflow.py`             | 217   | ScaffoldingWorkflow 集成 |
| `examples/scaffolding/core/controller.py`      | 207   | 引入的 controller 基类   |
| `examples/scaffolding/core/scaffolding_llm.py` | 230   | 编排器                   |
| `examples/scaffolding/search_scaffolding.py`   | 321   | 搜索智能体 controller    |
| `examples/scaffolding/worker.py`               | 249   | SGLang worker            |
| `tests/test_controllers.py`                    | 935   | Controller 测试          |
| `tests/test_scaffolding_llm_integration.py`    | 1,060 | 集成测试                 |
| `tests/test_self_contained.py`                 | 795   | 端到端测试               |

______________________________________________________________________

### 4.3 DPO（直接偏好优化）

**涵盖所有三种引擎后端的完整端到端实现。** 2,158 行代码横跨 26 个文件。

#### 损失函数

通过 `DPOEngineConfig.loss_type` 配置两种变体：

1. **Sigmoid（默认）** — 原始 DPO 损失（Rafailov et al., 2023）：

   ```
   L = -logsigmoid(β * (log(π_chosen/π_ref_chosen) - log(π_rejected/π_ref_rejected)))
   ```

1. **IPO** — Identity Preference Optimization（Azar et al., 2023）：

   ```
   L = (avg_logratio_chosen - avg_logratio_rejected - 1/(2β))²
   ```

   IPO 在计算平方损失前对每个 token 的 log-ratio 进行归一化，使得 β 在不同长度的序列间具有可比性。

#### 关键实现细节

- **fp64 scatter-add** 用于累加逐序列的 log 概率（`areal/utils/functional/functional.py` 中的
  `dpo_pair_logratios`）。fp32 累加在序列超过约 2K token 时可能导致 log-ratio 符号翻转。
- **交错的 chosen/rejected
  对**：`[chosen_0, rejected_0, chosen_1, rejected_1, ...]`。`DPOController` 在 RPC 分发中强制
  `group_size=2`。
- **损失掩码偏移**：每个序列的最后一个位置被显式置零，以防止跨序列泄漏。

#### 与 PPO/GRPO 的比较

| 方面     | DPO                                        | PPO/GRPO                            |
| -------- | ------------------------------------------ | ----------------------------------- |
| 奖励模型 | 不需要（隐含在偏好数据中）                 | 需要显式的奖励函数                  |
| Rollout  | 离线（chosen/rejected 对的数据集）         | 在线（实时生成）                    |
| 参考模型 | 必需（用于 KL 的冻结副本）                 | 可选（用于 KL 惩罚）                |
| 训练循环 | 简单：加载批次 → 计算 ref logps → 训练步骤 | 复杂：生成 → 奖励 → 优势 → PPO 更新 |
| 数据格式 | 配对偏好数据（HH-RLHF 风格）               | Prompt + 可验证奖励                 |

#### 配置选项

```python
@dataclass
class DPOEngineConfig:
    beta: float = 0.1         # KL 惩罚系数
    loss_type: str = "sigmoid" # "sigmoid" | "ipo"
    # 完整的引擎配置能力（FSDP/Megatron/Archon、微批次、梯度检查点）
```

#### 关键文件

| 文件                                   | 行数 | 用途                                                            |
| -------------------------------------- | ---- | --------------------------------------------------------------- |
| `areal/trainer/dpo_trainer.py`         | 515  | DPO 训练循环                                                    |
| `areal/trainer/dpo/dpo_engine.py`      | 207  | DPO 特定的损失和引擎封装                                        |
| `areal/utils/functional/functional.py` | +64  | `compute_dpo_loss`、`dpo_pair_logratios`、`dpo_preference_loss` |
| `areal/dataset/hhrlhf.py`              | 50   | Anthropic HH-RLHF 数据集加载器                                  |
| `examples/alignment/hhrlhf_dpo.yaml`   | 117  | Qwen2.5-7B 在 HH-RLHF 上的示例配置                              |
| `tests/test_dpo.py`                    | 518  | 全面测试                                                        |

______________________________________________________________________

### 4.4 拒绝采样配置

**统一配置，取代遗留的字符串类型参数。** 1,911 行代码横跨 52 个文件。

#### 问题

在异步 RL 训练中，行为策略（生成 rollout 数据的策略）会偏离近端策略（正在更新的策略）。来自 off-policy
数据的极端重要性权重会使训练不稳定。拒绝采样通过过滤或截断这些样本来解决此问题。

#### 配置维度

| 参数     | 选项                                  | 描述                              |
| -------- | ------------------------------------- | --------------------------------- |
| `level`  | "token" / "sequence"                  | 过滤粒度                          |
| `action` | "mask" / "clamp"                      | 将 loss_mask 置零或限制重要性权重 |
| `metric` | "ratio" / "kl_k1" / "kl_k2" / "kl_k3" | 散度度量                          |
| `agg`    | "sum" / "mean" / "max"                | 序列级聚合方式                    |
| `upper`  | float（默认 5.0）                     | 上限过滤阈值                      |
| `lower`  | float（可选）                         | 可选的下限                        |

#### 度量

- `ratio`：直接重要性比率 π_proximal / π_behave
- `kl_k1`：前向 KL 估计器 log(r)——可以为负
- `kl_k2`：二次近似 0.5 * (log r)²——始终非负
- `kl_k3`：精确前向 KL 估计器 r - log(r) - 1——始终非负

#### 集成方式

`apply_rejection_sampling()` 从 PPO 损失计算中调用。返回 `RejectionSamplingResult`，包含修改后的
`loss_mask`、`behave_imp_weight` 和 `filtered_fraction` 统计信息。包含从已移除的
`behave_imp_weight_mode`/`behave_imp_weight_cap` 参数的自动迁移。

______________________________________________________________________

### 4.5 Karmarkar-Karp 分区算法

**替代序列打包算法。** 1,408 行代码横跨 12 个文件。

#### 问题

RL 训练中变长序列导致微批次不平衡。由于所有 GPU 在梯度 all-reduce 边界处同步，最慢的 GPU 决定了整体吞吐量。减少 token
数量的最大-最小差异可直接提升 GPU 利用率。

#### 算法

`areal/utils/seqpack.py` 中的 Karmarkar-Karp 最大差分法：

1. 按长度升序排列序列
1. 为每个序列创建一个 `_KKState`
1. 使用以差值（max_sum - min_sum）为键的最大堆
1. 迭代弹出差值最大的两个状态，将最重的与最轻的配对合并
1. 持续直到剩余一个包含所有 k 个分区的状态

**复杂度**：O(n log n)——与 FFD 相同，但产生更好的平衡效果。

#### 性能

来自测试套件中的对比基准：

- 在 100 次双峰序列长度的随机试验中，KK 在 70% 以上的情况下获胜或持平
- 对于双峰/均匀分布：差值 \< 平均负载的 15%
- 对于偏斜（Pareto）分布：差值 \< 平均负载的 55%
- **安全回退**：如果 KK 违反容量约束，自动回退到 FFD

#### 配置

```python
MicroBatchSpec.packing_algorithm = "kk"  # 或 "ffd"（默认）
```

`get_allocate_fn()` 注册表使算法选择对训练管道透明。

______________________________________________________________________

### 4.6 MoE 模型的 LoRA 支持

**完整的 Megatron 后端 LoRA 在混合专家架构上的支持。** 325 行代码横跨 7 个文件。

核心挑战是 Megatron 融合格式（`linear_fc1`、`linear_fc2`）与 HuggingFace
逐专家格式（`experts.{i}.gate_proj`、`experts.{i}.up_proj`、`experts.{i}.down_proj`）之间的参数命名转换。

`areal/engine/megatron_utils/megatron_lora.py` 中的两个转换函数：

- **`convert_qwen3_lora_to_hf`**：Dense 模型的 LoRA（attention + MLP 投影）
- **`convert_qwen3_moe_lora_to_hf`**：MoE 特定的分组/逐专家适配器处理

示例配置训练 Qwen3-30B-A3B-Base（总 30B，活跃 3B）：

- 后端：`megatron:(attn:d1p6t1c1|ffn:d1p6t1e1)` — attention/FFN 分别并行
- LoRA rank 32, alpha 32，目标为所有线性层
- vLLM 推理，启用 LoRA 服务

______________________________________________________________________

### 4.7 Terminal Bench 训练示例

**Terminal Bench 基准的完整训练管道。** 1,947 行代码横跨 15 个文件。

Terminal Bench 评估 LLM 作为自主软件工程智能体的能力。智能体接收一个任务，可以访问 Docker 容器化的终端，并必须通过编写代码和执行命令来解决任务。

关键组件：

- **CamelTerminalAgent**：通过工具调用（shell、文件写入）与 Docker 环境交互。可配置 token 预算、迭代限制、超时处理。
- **CamelRLVRWorkflow**：`RolloutWorkflow` 为每个任务运行多条轨迹，支持
  `filter_uniform_reward`（丢弃所有轨迹获得相同奖励的任务）、`encourage_completion_reward`
  奖励塑形和逐轮奖励折扣（0.9）。

这代表了 AReaL 将训练智能体能力扩展到数学推理之外的方向，使用与 RLVR 方法论兼容的可验证奖励（单元测试通过/失败）。

______________________________________________________________________

### 4.8 推理服务重构

**后端抽象和模块化。** 6 个 commit，约 3,601 行。

#### 后端协议

`InfBridgeBackend` 协议（`areal/experimental/inference_service/backend.py`）定义了 8 个方法。两个实现：

| 后端                  | 文件                         | 目标                                             |
| --------------------- | ---------------------------- | ------------------------------------------------ |
| `SGLangBridgeBackend` | `sglang/bridge.py`（148 行） | SGLang `/generate` 端点                          |
| `VLLMBridgeBackend`   | `vllm/bridge.py`（150 行）   | vLLM `/v1/completions` 和 `/v1/chat/completions` |

每个后端约 150 行，无交叉依赖。添加新的推理后端只需实现一个包含 8 个方法的类。

#### Gateway 演进

推理 gateway（`areal/experimental/inference_service/gateway/app.py`）新增了：

- 通过 bearer token 进行认证（admin 与 session keys）
- 基于路由的 worker 选择（轮询、session 固定、基于模型的路由）
- SSE 流式传输支持 chat completions
- Session 生命周期管理
- 外部模型注册，支持非 AReaL 推理端点
- 并行数据代理注册，失败时支持回滚

#### 关键新文件

| 文件                                        | 行数 | 用途                       |
| ------------------------------------------- | ---- | -------------------------- |
| `inference_service/backend.py`              | 94   | `InfBridgeBackend` 协议    |
| `inference_service/sglang/bridge.py`        | 148  | SGLang 后端                |
| `inference_service/sglang/scheduler.py`     | 293  | AwexSchedulerBridge        |
| `inference_service/sglang/launch_server.py` | 164  | 自定义 SGLang 服务器启动器 |
| `inference_service/sglang/rpc_proxy.py`     | 54   | ZMQ RPC 代理               |
| `inference_service/vllm/bridge.py`          | 150  | vLLM 后端                  |
| `inference_service/router/state.py`         | 51   | 外部模型的路由状态         |

______________________________________________________________________

### 4.9 引擎演进

#### `from_pretrained` 工厂方法

`FSDPEngine.from_pretrained`（commit `e47dc676`）提供了遵循 HuggingFace 惯例的简化构造器：

```python
engine = FSDPEngine.from_pretrained(
    model_path="Qwen/Qwen2.5-7B",
    dp_size=2, tp_size=4,
    dtype=torch.bfloat16,
    learning_rate=1e-5,
)
```

新文件：`areal/engine/sglang_remote.py` 和 `areal/engine/vllm_remote.py`（各 44 行），用于远程引擎的
`from_pretrained`。

#### 内置内存分析器

`MemoryProfilerConfig`（commit `d58cca56`）：

- `profile_steps`：CUDA 内存快照的训练步骤列表（默认：\[0, 1\]）
- `max_entries`：环形缓冲区大小（默认：100,000）
- 同时避免收集 Context Parallel logits——防止不必要的 GPU 内存分配和跨 rank 通信。

#### 其他修复

- **多模态张量去重**（`349c6ed6`）：防止 FSDP 微批次拆分时复制视觉张量
- **HCCL 预热**（`50f0a0b0`）：即时初始化 NCCL/HCCL 通信器，防止 NPU 上的竞态条件
- **FSDP wrap 修复**（`073adbf2`）：正确处理集合类型的 `wrap_class_names`

______________________________________________________________________

### 4.10 基础设施可靠性

#### 两阶段 Teardown（commit `3ed3e817`）

解决了一个分布式系统竞态条件：rank-0 拥有 TCPStore 服务器，同时终止会导致 NCCL 心跳监控期间出现 "recvValue failed" 错误。

解决方案：

- **阶段 1**：在所有 worker 上调用 `engine.destroy()`。每个引擎执行 CPU barrier（gloo）然后
  `dist.destroy_process_group()`。
- **阶段 2**：仅在阶段 1 完成后，才终止操作系统进程。`reverse_order` 参数确保 rank-0 最后被终止。

#### 其他修复

| 修复                   | Commit     | 影响                                   |
| ---------------------- | ---------- | -------------------------------------- |
| IPv6 代理故障          | `e70b1934` | 数据服务在仅 IPv6 环境中正常工作       |
| Ray RPC 序列化         | `f3d7e50a` | 处理 payload 中的 ray.ObjectRef        |
| 整数设备 ID            | `e8c1e1fd` | Ray RPC 服务器接受整数设备 ID          |
| 可配置的 setup_timeout | `fe91acc1` | 用于 gateway 初始化时的大模型/慢速网络 |
| 数据服务 seed          | `ad622efd` | 移至 worker 级别配置以确保可复现性     |
| Docker venv 挂载       | `7fbd077c` | 防止容器挂载覆盖 venv                  |

______________________________________________________________________

## 5. .claude AI 辅助开发工具

### 5.1 清单

`.claude/` 目录是开源项目中最全面的 Claude Code 集成之一：

| 类别         | 数量 | 项目                                                                                                                                                        |
| ------------ | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agents**   | 8    | planner, simple-code-reviewer, code-verifier, fsdp-engine-expert, archon-engine-expert, megatron-engine-expert, algorithm-expert, launcher-scheduler-expert |
| **Skills**   | 7    | add-dataset, add-workflow, add-reward, add-archon-model, debug-distributed, add-unit-tests, commit-conventions                                              |
| **Commands** | 7    | create-pr, gen-commit-msg, review-pr, translate-doc-zh, update-docker-image, upgrade-megatron-core, upgrade-vllm                                            |
| **Rules**    | 4    | api-config.md, code-style.md, distributed.md, testing.md                                                                                                    |
| **数据文件** | 2    | review-pr-domains-and-signals.md, review-pr-templates.md                                                                                                    |
| **Hooks**    | 1    | check-expert-update.sh (PostToolUse)                                                                                                                        |

### 5.2 Commands

**`/create-pr`**（796 行）：完整的 PR 创建工作流，包含 7 个步骤：

1. 验证前置条件
1. 检测推送远程（fork 支持）
1. 检查已有 PR
1. 从 origin/main 获取并 rebase
1. 将 commit 压缩为单个 commit
1. 分析变更，生成 PR 标题/描述
1. 通过 `gh` 推送并创建/更新 PR

包含冲突、推送失败和 PR 创建失败的错误处理。具有备份分支和确认提示的安全检查。

**`/gen-commit-msg`**（170 行）：5 步 commit 消息生成器，分析暂存的变更，按类型分类，从文件路径确定 scope，并生成
Conventional Commits 格式。

**`/review-pr`**：复杂的多阶段 PR 审查，包含领域/信号检测、按严重程度动态路由模型（CRITICAL → Opus、MEDIUM → Sonnet、LOW
→ Haiku），以及 9 个 L1 领域和细粒度的 L2 信号。

### 5.3 Skills

**`commit-conventions`**（180 行）：始终加载的 skill，定义 Conventional Commits 格式。包括：

- 类型选择表（feat/fix/docs/gov/style/refactor/perf/test/build/ci/chore/revert）
- 从文件路径模式推断 scope（13 种映射）
- 格式规则（祈使语气、50-72 字符主题、72 字符正文换行）
- 4 个实际示例

**领域特定的
skills**（`add-dataset`、`add-workflow`、`add-reward`、`add-archon-model`、`debug-distributed`、`add-unit-tests`）：常见开发任务的分步指南，编码了项目特定的模式和惯例。

### 5.4 Agents

分阶段工作流：

1. **规划阶段**：`planner` 用于架构设计和实现规划
1. **领域专长**：`fsdp-engine-expert`、`archon-engine-expert`、`megatron-engine-expert`
   提供引擎特定指导；`algorithm-expert` 用于 RL 算法；`launcher-scheduler-expert` 用于集群部署
1. **代码格式化**：`code-verifier` 用于格式化、lint 和测试
1. **代码质量**：`simple-code-reviewer` 用于逻辑问题和代码异味

### 5.5 Rules

四项项目级代码质量标准：

- **`api-config.md`**：配置 dataclass 设计模式、`__post_init__` 验证、字段命名
- **`code-style.md`**：日志、性能模式、命名约定、张量约定
- **`distributed.md`**：分布式训练模式和约束
- **`testing.md`**：测试策略和覆盖率要求

### 5.6 Hooks

**`check-expert-update.sh`**（PostToolUse hook）：当引擎/模型目录中的文件被修改时触发，提醒开发者更新相应的 expert
agent 定义。确保 agent 知识随代码库演进保持最新。

______________________________________________________________________

## 6. Archon 训练框架

### 6.1 架构概览

ArchonEngine（`areal/experimental/engine/archon_engine.py`，1,550 行）是一个重要的 torch 原生训练后端，与
`FSDPEngine` 和 `MegatronEngine` 一起实现了 `TrainEngine` 抽象接口。

**关键能力**：

- 完整的并行支持：PP（流水线并行）、TP（张量并行）、CP（上下文并行）、EP（专家并行）、DP（数据并行）
- `ArchonParallelDims` 抽象，用于基于 mesh 的并行组访问
- 多个引擎子类：`ArchonPPOActor`、`ArchonPPOCritic`、`ArchonLMEngine`、`ArchonRWEngine`、`ArchonDPOEngine`（新增）

### 6.2 自 v1.0.3 以来的关键变更

#### 即时 HCCL/NCCL 通信器预热（commit `50f0a0b0`）

`create_process_group` 中的 `warmup_process_groups()` 调用在训练开始前即时初始化通信器，防止 NPU 硬件上的
HCCL/NCCL 竞态条件。`areal/engine/core/distributed.py` 中的实现处理了：None 组、重复组（通过
`dict.fromkeys`）、仅 CPU 平台和未初始化的 dist。

#### 可重入的 Offload 上下文（commit `db2a193b`）

`_offload_depth` 计数器模式防止嵌套上下文中的冗余 offload/onload 转换。只有最外层调用才会实际执行
offload/onload——内部调用为空操作。这减少了复杂训练循环中的 GPU-CPU 内存传输开销。

```python
self._offload_depth: int = 0  # 可重入计数器

# 在 _offload_aware_context 中：
if self._offload_depth == 0:
    self.onload()  # 只有最外层调用触发 onload
self._offload_depth += 1
# ... yield ...
self._offload_depth -= 1
if self._offload_depth == 0:
    self.offload()  # 只有最外层调用触发 offload
```

#### 两阶段 Teardown（commit `3ed3e817`）

预销毁 CPU barrier 防止 NCCL HeartbeatMonitor 线程产生嘈杂的 stderr 回溯。barrier 周围的 `try/except` 配合
warning 级别日志使 teardown 具有弹性。`destroy()` 通过 `self.own_global_group = False` 实现幂等性。

### 6.3 ArchonDPOEngine

新子类（第 1509-1549 行），扩展 ArchonEngine 以支持 DPO 训练：

```python
class ArchonDPOEngine(ArchonEngine):
    # 确保 mb_spec.granularity == 2 以适配配对的 DPO 数据
    # 根据 config._version 路由到 DPOController 或 DPOControllerV2
    # 提供 as_controller() 用于分布式分发
```

**Controllers**：

- `DPOController`（v1）：基于 RPC 的遗留分发，`group_size=2` 配对
- `DPOControllerV2`（v2）：基于 gateway 的分布式分发

______________________________________________________________________

## 7. 架构评估

### 7.1 设计模式

**在代码库中一致应用的模式：**

1. **`@runtime_checkable Protocol`** 用于接口（引擎 API、bridge 后端、awex
   适配器）。这是项目的主要抽象机制，提供结构化类型而不依赖继承链——与声明的组合优于继承的偏好一致。

1. **延迟初始化**：Awex 适配器在首次请求时创建，scaffolding 组件在首次 episode 时创建。有利于减少分布式环境中的启动开销。

1. **训练 worker 使用 Flask blueprint，推理/gateway 使用 FastAPI**：反映了底层工作负载的同步与异步特性。

1. **`HttpRequest` dataclass 作为中间层**：在推理 bridge 中将"发送什么"与"如何发送"分离。

1. **注册表模式**：`get_allocate_fn()` 用于打包算法，`PACKING_ALGORITHMS` 集合用于验证。

### 7.2 可扩展性评估

- **Awex**：水平扩展——传输计划是逐 rank 的，每个 rank 只发送/接收其本地分片。NCCL P2P 避免了集合 all-to-all 瓶颈。
- **KK 分区**：通过减少微批次不平衡直接提升吞吐量，随着 RL 工作负载中序列长度方差的增加，这一点变得更加重要。
- **两阶段 teardown**：使用 gloo（CPU）barrier 正确扩展，避免 GPU 同步死锁。
- **Scaffolding**：可配置的并行度（`max_parallel_requests=64`）支持高吞吐量的多步骤智能体 rollout。

### 7.3 技术债务

1. **Awex 适配器代码重复**：FSDP 和 Megatron 训练适配器在组初始化/执行/teardown 方法中共享约 80%
   的相同代码。添加第三个训练后端需要复制粘贴这些样板代码。

1. **三个引擎后端共享代码有限**：FSDPEngine、MegatronEngine 和 ArchonEngine
   独立实现进程组创建、检查点管理和训练循环。`areal/engine/core/` 提供了一些共享工具，但利用不足。

1. **Gateway 职责累积**：推理 gateway 已增长到包含路由、认证、session 管理、模型注册、worker 生命周期和权重更新协调。

1. **Scaffolding 放置位置**：尽管是一个带有大量测试的完整 workflow，它位于 `examples/scaffolding/` 而非
   `areal/workflow/` 或 `areal/experimental/`。

1. **Flask/FastAPI 分裂**：训练侧使用 Flask（同步），推理侧使用 FastAPI（异步）。虽然是有意为之，但给维护者带来了认知负担。

### 7.4 建议

| 优先级 | 建议                                                                                     |
| ------ | ---------------------------------------------------------------------------------------- |
| **高** | 提取共享的 awex 适配器逻辑到基类中，以防止新引擎类型添加时的重复                         |
| **中** | 为 ArchonEngine 和 MegatronEngine 添加 `from_pretrained` 工厂方法，保持 API 一致性       |
| **中** | 将推理 gateway 分解为专注的子路由（路由、生命周期、模型管理）                            |
| **中** | 将 scaffolding workflow 从 `examples/` 提升到 `areal/experimental/` 或 `areal/workflow/` |
| **低** | 统一 Flask/FastAPI 分裂的文档，便于维护者入门                                            |
| **低** | 在 bridge 后端中添加 `max_new_tokens` 计算的共享工具                                     |
| **低** | 将 `.claude/` 文件中重复的 scope 推断表提取到共享数据文件                                |

______________________________________________________________________

## 8. 代码质量审查

### 8.1 规范遵循情况

**日志**：所有新代码正确使用 `areal.utils.logging.getLogger()` 并采用 PascalCase
名称：`"DPOEngine"`、`"DPOTrainer"`、`"AwexBlueprint"`、`"AwexInferenceEndpoints"`、`"RLVRControllers"`、`"ScaffoldingWorkflow"`、`"SeqPack"`。唯一例外：引入的
`core/controller.py` 使用标准库 logging。

**命名**：`DPOEngineConfig`、`DPOConfig`、`ArchonDPOEngine`、`DPOTrainer`、`DPOController`
均遵循既定模式（`XxxConfig`、`XxxEngine`、`XxxTrainer`、`XxxController`）。

**许可证头**：所有新文件包含 `# SPDX-License-Identifier: Apache-2.0`。

**导入风格**：无通配符导入。分组遵循 stdlib/第三方/areal 排序。

### 8.2 测试覆盖率

| 领域          | 测试文件                        | 行数   | 质量                                  |
| ------------- | ------------------------------- | ------ | ------------------------------------- |
| DPO 损失      | `test_dpo.py`                   | 518    | 高——正确性、边界情况、错误处理        |
| 拒绝采样      | `test_rejection_sampling.py`    | 839    | 非常高——所有模式/级别/格式组合        |
| KK 算法       | `test_kk_allocate.py`           | 603    | 非常高——内部实现 + 集成 + 对比        |
| KK 端到端     | `test_kk_e2e.py`                | 177    | 良好——分布式比较                      |
| 进程组预热    | `test_warmup_process_groups.py` | 89     | 良好——所有守卫分支                    |
| Scaffolding   | 4 个文件                        | ~2,995 | 对 examples/ 代码而言非常高           |
| Awex 权重更新 | 9 个文件                        | ~2,236 | 全面——NCCL、磁盘、controller、KV 存储 |

**新增测试代码总量：5,000+ 行。** 测试与功能代码比为 1.18:1，表现优秀。

### 8.3 按严重程度分类的发现

#### 中等（3 个）

1. **`ArchonDPOEngine.__init__` 静默修改配置**：通过 `deepcopy` 静默将 `mb_spec.granularity` 修正为
   2，而非抛出 `ValueError`。根据 `api-config.md`，验证应使用 `__post_init__` 并给出清晰的错误信息。

1. **`apply_rejection_sampling` 圈复杂度**：约 200 行，跨 `level × action × ndim × agg × metric`
   存在深层嵌套分支。建议提取为更小的辅助函数。

1. **训练侧 awex.py 错误处理不一致**：与推理侧对应部分相比缺少显式的 try/except 块。完全依赖 `run_endpoint` 进行错误处理。

#### 低（10 个）

1. 3 个 `.claude/` 文件中的重复 scope 推断表
1. `create-pr.md` 备份步骤未在工作流序列中强制执行
1. ArchonDPOEngine 在 `__init__` 中实例化 Logger（应在模块级别）
1. `_version` 私有字段用于 `ArchonDPOEngine.as_controller` 中的控制流路由
1. `DPOTrainer.close()` 中的冗余 `hasattr` 检查
1. `compute_dpo_loss` 中未使用的 `entropy`/`vocab_*_logits` 参数缺少解释性注释
1. 引入的 `core/controller.py` 使用标准库 logging 而非 `areal.utils.logging`
1. `data.py` 中 `packing_algorithm` 的 `getattr` 回退——该字段应正式定义
1. awex 训练侧 `_state` 字典模式存在线程安全间隙（缺少 `threading.Lock`）
1. 推理侧 `debug/randomize_parameters` 端点在训练侧缺失

**总体评估**：自 v1.0.3 以来的代码质量很高。DPO 实现具有精心的数值处理，KK 算法是一个带有优秀测试的干净插入式替换，`.claude/`
工具是开源项目中最全面的之一。未发现关键的安全问题或正确性缺陷。

______________________________________________________________________

## 9. 战略方向分析

自 v1.0.3 以来的变更揭示了 AReaL 发展轨迹中的四个战略转变：

### 从单轮 RLHF 到智能体 RL

Scaffolding、Terminal Bench 和多轮信用分配的组合代表了一个战略转变——从训练 LLM 作为响应生成器转向训练其作为自主智能体。Scaffolding
框架的可组合 controller 架构支持在不改变核心训练基础设施的情况下训练越来越复杂的智能体行为。

### 从同步到异步训练

Awex 系统结合拒绝采样，实现了完全异步的 RL 训练，其中 rollout 和训练在不同硬件上并发进行。拒绝采样为策略散度提供安全网，支持在 token
和序列粒度上的可配置度量和操作。

### 从仅 PPO 到多算法对齐

添加 DPO（sigmoid + IPO 变体）在偏好数据可用时提供了 PPO 的离线替代方案。共享的引擎基础设施（FSDP/Megatron/Archon）意味着 DPO
立即受益于所有工程优化。

### 从均匀到自适应批处理

KK 分区算法通过减少同步屏障处的 GPU 空闲时间直接解决计算效率问题。相比 FFD 超过 70% 的胜率在变长 RL 工作负载中转化为有意义的吞吐量提升。

### 定量健康指标

| 指标                           | 值                          | 信号           |
| ------------------------------ | --------------------------- | -------------- |
| 功能与修复比                   | 1.64:1                      | 健康的平衡     |
| 测试与功能代码比               | 1.18:1                      | 强测试纪律     |
| 新增文件与删除比               | 105:1                       | 增长阶段       |
| 社区广度                       | 18 位贡献者                 | 活跃的开源社区 |
| Conventional Commit 规范遵循率 | 98%                         | 一致的标准     |
| 开发加速                       | 2.9 倍（第 1 周 → 第 2 周） | 增长势头       |

______________________________________________________________________

## 10. 新文件清单

### 新建目录

| 目录                                           | 文件数 | 用途                     |
| ---------------------------------------------- | ------ | ------------------------ |
| `areal/experimental/weight_update/`            | ~15    | Awex 权重更新系统        |
| `areal/experimental/inference_service/sglang/` | 6      | SGLang 推理后端          |
| `areal/experimental/inference_service/vllm/`   | 2      | vLLM 推理后端            |
| `areal/trainer/dpo/`                           | 2      | DPO trainer 模块         |
| `examples/scaffolding/`                        | ~30    | Scaffolding rollout 框架 |
| `examples/terminal_bench/`                     | ~15    | Terminal Bench RL 训练   |
| `tests/experimental/weight_update/`            | ~9     | 权重更新测试             |

### 删除的文件

| 文件                                                                   | 原因                            |
| ---------------------------------------------------------------------- | ------------------------------- |
| `areal/experimental/inference_service/data_proxy/backend.py`（340 行） | 被 sglang/vllm 子模块重构所取代 |

### 移动的文件

| 原路径                                       | 新路径                            | 原因     |
| -------------------------------------------- | --------------------------------- | -------- |
| `inference_service/data_proxy/inf_bridge.py` | `inference_service/inf_bridge.py` | 模块重构 |
| `assets/*.png`（19 个文件）                  | `assets/figures/*.png`            | 资源整理 |

### 最常修改的文件

| 文件                                         | 涉及的 Commit 数                                        |
| -------------------------------------------- | ------------------------------------------------------- |
| `areal/api/cli_args.py`                      | 5（RejectionSampling、DPO、KK、内存分析器、训练前评估） |
| `areal/engine/fsdp_engine.py`                | 5（from_pretrained、DPO、多模态、预热、offload）        |
| `areal/experimental/engine/archon_engine.py` | 4（DPO、预热、teardown、offload）                       |
| `areal/engine/megatron_engine.py`            | 4（DPO、LoRA MoE、预热、offload）                       |
| `docs/en/cli_reference.md`                   | 5（因新 CLI 字段而重新生成）                            |

______________________________________________________________________

*分析于 2026-04-27 生成，使用 5 个并行 Opus 4.6
agent：Explore（代码库清单）、Architect-Reviewer（设计评估）、Code-Reviewer（质量审计）、Data-Analyst（定量指标）、Data-Scientist（ML/RL
功能分析）。*
