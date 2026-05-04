# AReaL Source Code Analysis: origin/main Since v1.0.3

**Analysis Date**: 2026-04-27 **Commit Range**: `376ecbb8` (v1.0.3) → `b0ea0dd8`
(origin/main HEAD) **Scope**: 49 non-merge commits | 27,743 insertions | 2,548 deletions
| 321 files changed | 105 new files

______________________________________________________________________

## Table of Contents

1. [Executive Summary](#1-executive-summary)
1. [Quantitative Overview](#2-quantitative-overview)
   - 2.1 [Summary Statistics](#21-summary-statistics)
   - 2.2 [Contributor Breakdown](#22-contributor-breakdown)
   - 2.3 [Directory-Level Change Distribution](#23-directory-level-change-distribution)
   - 2.4 [Commit Type Distribution](#24-commit-type-distribution)
   - 2.5 [Top 10 Largest Commits](#25-top-10-largest-commits)
   - 2.6 [Development Timeline](#26-development-timeline)
1. [Full Commit Inventory](#3-full-commit-inventory)
1. [Feature Deep Dives](#4-feature-deep-dives)
   - 4.1 [Awex (Async Weight Exchange) System](#41-awex-async-weight-exchange-system)
   - 4.2 [Scaffolding Rollout Workflow](#42-scaffolding-rollout-workflow)
   - 4.3 [DPO (Direct Preference Optimization)](#43-dpo-direct-preference-optimization)
   - 4.4 [Rejection Sampling Configuration](#44-rejection-sampling-configuration)
   - 4.5 [Karmarkar-Karp Partitioning](#45-karmarkar-karp-partitioning)
   - 4.6 [LoRA for MoE Models](#46-lora-for-moe-models)
   - 4.7 [Terminal Bench Training Example](#47-terminal-bench-training-example)
   - 4.8 [Inference Service Refactoring](#48-inference-service-refactoring)
   - 4.9 [Engine Evolution](#49-engine-evolution)
   - 4.10 [Infrastructure Reliability](#410-infrastructure-reliability)
1. [.claude AI-Assisted Development Tooling](#5-claude-ai-assisted-development-tooling)
   - 5.1 [Inventory](#51-inventory)
   - 5.2 [Commands](#52-commands)
   - 5.3 [Skills](#53-skills)
   - 5.4 [Agents](#54-agents)
   - 5.5 [Rules](#55-rules)
   - 5.6 [Hooks](#56-hooks)
1. [Archon Training Framework](#6-archon-training-framework)
   - 6.1 [Architecture Overview](#61-architecture-overview)
   - 6.2 [Key Changes Since v1.0.3](#62-key-changes-since-v103)
   - 6.3 [ArchonDPOEngine](#63-archondpoengine)
1. [Architectural Assessment](#7-architectural-assessment)
   - 7.1 [Design Patterns](#71-design-patterns)
   - 7.2 [Scalability Assessment](#72-scalability-assessment)
   - 7.3 [Technical Debt](#73-technical-debt)
   - 7.4 [Recommendations](#74-recommendations)
1. [Code Quality Review](#8-code-quality-review)
   - 8.1 [Convention Adherence](#81-convention-adherence)
   - 8.2 [Test Coverage](#82-test-coverage)
   - 8.3 [Findings by Severity](#83-findings-by-severity)
1. [Strategic Direction Analysis](#9-strategic-direction-analysis)
1. [New File Inventory](#10-new-file-inventory)

______________________________________________________________________

## 1. Executive Summary

The 49 commits since v1.0.3 represent a major feature sprint over 19 days (April 9–27,
2026), adding approximately 25,195 net lines of code across 321 files by 18
contributors. The release is dominated by five transformative features:

1. **Awex (Async Weight Exchange)**: A complete GPU-to-GPU weight synchronization system
   enabling direct NCCL P2P transfers between training and inference engines across
   different parallelism strategies (FSDP, Megatron TP/PP/CP/EP, SGLang).

1. **Scaffolding Rollout Workflow**: A modular agent execution framework (8,286 lines)
   enabling multi-step, multi-tool, multi-turn agent behaviors during rollout — the
   single largest commit in this release.

1. **DPO Trainer**: Full Direct Preference Optimization implementation with sigmoid and
   IPO loss variants, supporting all three engine backends (FSDP, Megatron, Archon).

1. **Unified Rejection Sampling**: A new `RejectionSamplingConfig` replacing legacy
   string-typed parameters with a structured dataclass supporting token/sequence-level
   filtering at multiple divergence metrics.

1. **Karmarkar-Karp Partitioning**: An alternative sequence packing algorithm providing
   measurably better micro-batch balance than First Fit Decreasing.

Additionally, the `.claude/` directory received significant enhancements with new
commands (`/create-pr`, `/gen-commit-msg`), skills (`commit-conventions`), and improved
AI-assisted development workflows. The Archon engine gained DPO support, reentrant
offload contexts, eager HCCL warmup, and two-phase teardown for clean shutdown.

______________________________________________________________________

## 2. Quantitative Overview

### 2.1 Summary Statistics

| Metric                            | Value                              |
| --------------------------------- | ---------------------------------- |
| Total non-merge commits           | 49                                 |
| Unique contributors               | 18                                 |
| Files changed (unique)            | 321                                |
| Total insertions                  | 27,743                             |
| Total deletions                   | 2,548                              |
| Net lines added                   | +25,195                            |
| New files added                   | 105                                |
| Files deleted                     | 1                                  |
| Date range                        | 2026-04-09 to 2026-04-27 (19 days) |
| Average commits/day (active days) | 4.1                                |
| Average lines changed per commit  | ~619                               |
| Feature-to-fix ratio              | 1.64:1                             |
| Test-to-feature code ratio        | 1.18:1                             |
| Conventional commit compliance    | 48/49 (98%)                        |

### 2.2 Contributor Breakdown

| Rank | Contributor                    | Commits | % of Total |
| ---- | ------------------------------ | ------- | ---------- |
| 1    | Wei Fu (garrett4wade)          | 20      | 40.8%      |
| 2    | xiao (Wangxiaoxiaoa)           | 4       | 8.2%       |
| 3    | Ran Yan                        | 3       | 6.1%       |
| 4    | HT-Yuan                        | 3       | 6.1%       |
| 5    | Gursimran                      | 2       | 4.1%       |
| 6    | TaoZex                         | 2       | 4.1%       |
| 7    | sitabulaixizawaluduo           | 2       | 4.1%       |
| 8    | Pratyush Sharma                | 2       | 4.1%       |
| 9–18 | 10 other contributors (1 each) | 11      | 22.4%      |

Wei Fu is the dominant contributor at 40.8%. The remaining 17 contributors form a
healthy long-tail distribution, indicating an active open-source community with
corporate participation.

### 2.3 Directory-Level Change Distribution

| Directory               | Files Changed | Insertions | Deletions | Net     | % of Insertions |
| ----------------------- | ------------- | ---------- | --------- | ------- | --------------- |
| **examples/**           | 93            | 10,833     | 86        | +10,747 | 39.1%           |
| **tests/**              | 47            | 6,413      | 277       | +6,136  | 23.1%           |
| **areal/experimental/** | 70            | 5,455      | 770       | +4,685  | 19.7%           |
| **docs/**               | 13            | 1,148      | 503       | +645    | 4.1%            |
| **areal/trainer/**      | 9             | 820        | 30        | +790    | 3.0%            |
| **areal/utils/**        | 6             | 733        | 79        | +654    | 2.6%            |
| **areal/infra/**        | 17            | 657        | 171       | +486    | 2.4%            |
| **areal/engine/**       | 11            | 624        | 63        | +561    | 2.3%            |
| **areal/api/**          | 4             | 380        | 56        | +324    | 1.4%            |
| **.claude/**            | 3             | 55         | 27        | +28     | 0.2%            |
| **areal/workflow/**     | 0             | 0          | 0         | 0       | 0.0%            |

Key observations:

- **examples/** dominates at 39.1% — driven by scaffolding (8,286 lines) and Terminal
  Bench (1,947 lines).
- **tests/** at 23.1% indicates excellent testing discipline.
- **areal/experimental/** at 19.7% reflects active awex and inference service
  development.
- **areal/workflow/** had zero changes — workflow logic was stable.

### 2.4 Commit Type Distribution

| Type         | Count | %     | Description                   |
| ------------ | ----- | ----- | ----------------------------- |
| **feat**     | 18    | 36.7% | New features                  |
| **fix**      | 11    | 22.4% | Bug fixes                     |
| **chore**    | 10    | 20.4% | Maintenance/dependencies      |
| **refactor** | 3     | 6.1%  | Code restructuring            |
| **gov**      | 3     | 6.1%  | Governance/maintainer updates |
| **docs**     | 2     | 4.1%  | Documentation                 |
| **test**     | 1     | 2.0%  | Test additions                |
| **perf**     | 1     | 2.0%  | Performance improvements      |

**Scope distribution** (most active):

| Scope        | Count |
| ------------ | ----- |
| experimental | 6     |
| engine       | 5     |
| infra        | 4     |
| service      | 3     |
| trainer      | 2     |
| deps         | 2     |

### 2.5 Top 10 Largest Commits

| Rank   | Hash       | +Insertions | Title                                                      |
| ------ | ---------- | ----------- | ---------------------------------------------------------- |
| **1**  | `d37095ae` | **+8,286**  | feat: add scaffolding rollout workflow (#1064)             |
| **2**  | `615d1bae` | **+4,112**  | feat: add awex backend for weight update (#1214)           |
| **3**  | `628c389e` | **+2,238**  | feat(service): add external model API support (#1183)      |
| **4**  | `70acd22f` | **+2,158**  | feat(trainer): add dpo (#1190)                             |
| **5**  | `aeb237bd` | **+1,947**  | feat(example): add Terminal Bench training example (#1224) |
| **6**  | `bc9f0098` | **+1,911**  | feat(api): add unified RejectionSamplingConfig (#1088)     |
| **7**  | `8c8a8dbd` | **+1,408**  | feat(utils): add Karmarkar-Karp partitioning (#1151)       |
| **8**  | `ae8c792f` | **+757**    | feat: add disk-mode weight update flow (#1237)             |
| **9**  | `8cc52ba0` | **+686**    | refactor(experimental): reuse HTTP clients (#1253)         |
| **10** | `6e69226c` | **+656**    | feat(experimental): Megatron awex TP adapter (#1239)       |

These top 10 commits account for **24,159 insertions** — 87% of the total.

### 2.6 Development Timeline

| Date   | Commits | Notable Activity                                                           |
| ------ | ------- | -------------------------------------------------------------------------- |
| Apr 9  | 1       | FSDP set-valued wrap fix                                                   |
| Apr 16 | 2       | Community meeting docs, asset reorganization                               |
| Apr 17 | 4       | Scaffolding workflow (8,286 lines), external model API, LoRA MoE, IPv6 fix |
| Apr 19 | 1       | Ray RPC serialization fix                                                  |
| Apr 20 | 6       | Awex backend (4,112 lines), RejectionSamplingConfig, KK partitioning, DPO  |
| Apr 21 | 3       | Memory profiler, governance, awex dep upgrade                              |
| Apr 22 | 5       | Megatron awex TP, disk-mode weight update, Terminal Bench                  |
| Apr 23 | 7       | Megatron PP/CP awex, Docker fix, teardown fix                              |
| Apr 24 | 4       | NPU HCCL fix, SFT batch test, Megatron awex EP                             |
| Apr 25 | 3       | Inference service refactoring                                              |
| Apr 26 | 4       | Multimodal tensor fix, logging, offload/onload endpoints                   |
| Apr 27 | 9       | Engine from_pretrained, CI fix, LSP docs                                   |

Development accelerated 2.9x from Week 1 (1.75 commits/day) to Week 2+ (5.1
commits/day).

______________________________________________________________________

## 3. Full Commit Inventory

### Categorized by Domain

#### (a) .claude / AI Agent Tooling (2 commits)

| Hash       | Lines | Title                                                                                              |
| ---------- | ----- | -------------------------------------------------------------------------------------------------- |
| `434df57b` | +131  | gov: add maintainer (#1227) — includes commit-conventions skill, create-pr/gen-commit-msg commands |
| `b0ea0dd8` | +28   | docs(workflow): add LSP-first code navigation guidance                                             |

#### (b) Archon / Experimental Engine (3 commits)

| Hash       | Lines | Title                                                                           |
| ---------- | ----- | ------------------------------------------------------------------------------- |
| `50f0a0b0` | +166  | fix(engine): eagerly init HCCL subgroups to fix ref compute_logp on NPU (#1254) |
| `db2a193b` | +50   | perf(trainer): reduce redundant offload/onload transitions (#1163)              |
| `3ed3e817` | +438  | fix(infra): add two-phase teardown to prevent TCPStore race at shutdown (#1244) |

#### (c) Inference Service (6 commits)

| Hash       | Lines  | Title                                                                                    |
| ---------- | ------ | ---------------------------------------------------------------------------------------- |
| `628c389e` | +2,238 | feat(service): add external model API support for inference service (#1183)              |
| `2d6ea231` | +392   | refactor(service): extract inference bridge backends into sglang/vllm submodules (#1221) |
| `8cc52ba0` | +686   | refactor(experimental): reuse HTTP clients, add response models (#1253)                  |
| `5c723ffe` | +164   | feat(experimental): add offload/onload endpoints (#1276)                                 |
| `ed06091b` | +88    | chore(experimental): suppress HTTP service logging (#1274)                               |
| `d38268e1` | +33    | feat(infra): add n_gpus_per_node abstract property to Scheduler API (#1275)              |

#### (d) Training Service / Awex Weight Update (8 commits)

| Hash       | Lines  | Title                                                                       |
| ---------- | ------ | --------------------------------------------------------------------------- |
| `615d1bae` | +4,112 | feat: add awex backend for weight update (#1214)                            |
| `6e69226c` | +656   | feat(experimental): MegatronEngine awex TP adapter (#1239)                  |
| `ae8c792f` | +757   | feat: add disk-mode weight update flow to gateway (#1237)                   |
| `5bd7a180` | +227   | feat: Implement Megatron PP and CP with Awex (#1246)                        |
| `b7b10278` | +210   | feat(experimental): MegatronEngine awex EP adapter (#1252)                  |
| `01dab41e` | +8     | refactor(experimental): rename WeightUpdate*Adapter to Awex*Adapter (#1269) |
| `e6f3c3cb` | +235   | chore(deps): upgrade awex to 0.7.0 (#1228)                                  |
| `4629c4ef` | +18    | chore(deps): upgrade mbridge from 0.15.1 to 310e8fb (#1258)                 |

#### (e) Core Engine — FSDP / Megatron (6 commits)

| Hash       | Lines | Title                                                                        |
| ---------- | ----- | ---------------------------------------------------------------------------- |
| `073adbf2` | +13   | fix: FSDP initialization for set-valued wrap class names (#1187)             |
| `e5531199` | +325  | feat(engine): lora support for MoE models (#1159)                            |
| `349c6ed6` | +91   | fix(engine): avoid duplicating multimodal tensors (#1272)                    |
| `e47dc676` | +267  | feat(engine): support direct engine construction via from_pretrained (#1140) |
| `d58cca56` | +376  | feat(engine): add built-in memory profiler support (#1223)                   |
| `50f0a0b0` | +166  | fix(engine): eagerly init HCCL subgroups for NPU (#1254)                     |

#### (f) Workflow / Algorithm (5 commits)

| Hash       | Lines  | Title                                                      |
| ---------- | ------ | ---------------------------------------------------------- |
| `d37095ae` | +8,286 | feat: add scaffolding rollout workflow (#1064)             |
| `70acd22f` | +2,158 | feat(trainer): add dpo (#1190)                             |
| `bc9f0098` | +1,911 | feat(api): add unified RejectionSamplingConfig (#1088)     |
| `8c8a8dbd` | +1,408 | feat(utils): add Karmarkar-Karp partitioning (#1151)       |
| `aeb237bd` | +1,947 | feat(example): add Terminal Bench training example (#1224) |

#### (g) Infrastructure (8 commits)

| Hash       | Lines | Title                                                                        |
| ---------- | ----- | ---------------------------------------------------------------------------- |
| `3ed3e817` | +438  | fix(infra): add two-phase teardown to prevent TCPStore race (#1244)          |
| `ad622efd` | +267  | fix(infra): move data service seed to worker-level config (#1210)            |
| `e70b1934` | +123  | fix(Service): fix data service failures in IPv6-only environments (#1208)    |
| `f3d7e50a` | +51   | fix: serialize ray object refs in rpc payloads (#1198)                       |
| `e8c1e1fd` | +6    | fix: handle integer device ids in ray rpc server (#1199)                     |
| `fe91acc1` | +10   | feat: add configurable setup_timeout for data service gateway (#1263)        |
| `9fb5247d` | +27   | test(infra): regression for single-controller SFT batch partitioning (#1255) |
| `f34468c7` | +36   | feat(eval): support for running eval before training (#1232)                 |

#### (h) Governance / Docs / Community (9 commits)

| Hash       | Lines | Title                                                      |
| ---------- | ----- | ---------------------------------------------------------- |
| `d4891785` | +51   | chore: move figures into assets/figures/ (#1192)           |
| `8965973b` | +2    | chore: add @CormickKneey as maintainer (#1201)             |
| `65928242` | +15   | chore: add new maintainers (#1220)                         |
| `c3ba6faa` | +17   | docs(community): add first biweekly meeting record (#1215) |
| `259b3430` | +1    | chore: update tencent meeting link (#1219)                 |
| `2aaddaec` | +14   | chore: add new maintainer (#1234)                          |
| `68cd2e5a` | +15   | gov: add maintainer (#1235)                                |
| `b499183f` | +13   | chore: update news for scaffoldings (#1236)                |
| `2851ea71` | +0    | gov: update governance (#1248)                             |

#### (i) CI / Build / Dependencies (5 commits)

| Hash       | Lines | Title                                                                |
| ---------- | ----- | -------------------------------------------------------------------- |
| `7fbd077c` | +388  | fix(docker): move venv out of /AReaL to avoid mount override (#1251) |
| `bcb216d6` | +2    | fix: Update CI test GCP Image (#1277)                                |
| `5d15f659` | +3    | fix: fix pre-commit CI env                                           |
| `15384e7f` | +12   | chore: add uv lock check (#1259)                                     |
| `e6f3c3cb` | +235  | chore(deps): upgrade awex to 0.7.0 (#1228)                           |

______________________________________________________________________

## 4. Feature Deep Dives

### 4.1 Awex (Async Weight Exchange) System

**The largest cross-cutting feature addition in this release.** Spanning 8 commits and
~5,500+ lines of new code.

#### Problem Statement

In AReaL's async training architecture, the training engine and inference engine run on
separate GPU clusters with different parallelism strategies. Updated model weights must
be propagated from training to inference for the next rollout batch. The challenge: the
two engines may use different sharding strategies (e.g., training uses FSDP with DP
sharding, inference uses TP in SGLang).

#### Architecture

Awex operates through a three-layer design:

**1. Protocol Layer** — Two `@runtime_checkable Protocol` classes:

- `AwexTrainingAdapter` (`areal/experimental/weight_update/training_adapter.py`) —
  training side (sends weights)
- `AwexInferenceAdapter` (`areal/experimental/weight_update/inference_adapter.py`) —
  inference side (receives weights)

Both share the same method signatures (`init_weight_update_group`,
`execute_weight_update`, `batch_isend_irecv`, `teardown_weight_update_group`), differing
only in send vs receive semantics.

**2. Adapter Layer** — Concrete implementations per engine type:

- `AwexFSDPAdapter` — extracts DTensor shard metadata, computes offsets from placements
- `AwexMegatronAdapter` — uses `all_gather_param` + `convert_to_hf` to normalize to HF
  naming
- `AwexSGLangAdapter` — handles SGLang's fused parameter naming (qkv_proj, gate_up_proj,
  w13_weight/w2_weight) by unfusing to HF-compatible per-expert names

**3. HTTP Endpoint Layer** — Flask blueprint (training side,
`areal/experimental/training_service/worker/awex.py`) and FastAPI endpoints (inference
side, `areal/experimental/inference_service/sglang/awex.py`) expose adapter operations
over HTTP. The gateway orchestrates the multi-step handshake.

#### Weight Transfer Protocol Flow

```
1. Both sides → /awex/report_parallelism     (report parallelism strategy)
2. Both sides → /awex/report_weight_meta      (publish shard metadata to KV store)
3. TransferPlanBuilder computes optimal P2P transfer plan
4. Both sides → /awex/init_weights_update_group  (establish dedicated NCCL group)
5. Training:  nccl_build_send_ops()  →  batch_send_recv()
   Inference: nccl_build_recv_ops()  →  batch_send_recv()
```

#### Parallelism Support

| Mode   | Training Side                                                                       | Inference Side                                              |
| ------ | ----------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **TP** | FSDP: DTensor shard placements; Megatron: `all_gather_param` + `dp_replicated=True` | SGLang: TP-aware shard offsets                              |
| **EP** | Megatron: `mpu.get_expert_model_parallel_world_size()`                              | SGLang: `EP_SHARDING` / `EP_TP_SHARDING` offsets per expert |
| **PP** | Each PP stage yields its own layer subset via `get_transformer_layer_offset`        | Gateway merges disjoint PP stage metadata by name           |
| **CP** | Reported in `RankInfo` with `cp_mode="ring"`                                        | Treated similarly to TP for dp_replicated logic             |

#### Streaming vs Disk Mode

- **Streaming mode** (default): Direct NCCL P2P transfers. Zero disk I/O, lowest
  latency.
- **Disk mode** (commit `ae8c792f`): Training workers save parameters to shared
  filesystem; inference workers load from that path. Generation is paused/resumed via
  context manager during weight replacement.

#### MoE Handling

The SGLang adapter's `_unfuse_params` handles the complex case where SGLang fuses
experts:

- `experts.w13_weight` shape: `[num_total_experts, 2*ffn_hidden, hidden]` (gate+up fused
  per-expert)
- `experts.w2_weight` shape: `[num_total_experts, hidden, ffn_hidden]` (down per-expert)

These are unfused into per-expert HuggingFace-style names for cross-engine
interoperability.

#### Key Files

| File                                                        | Lines | Purpose                                               |
| ----------------------------------------------------------- | ----- | ----------------------------------------------------- |
| `areal/experimental/weight_update/nccl_group.py`            | 225   | NCCL group management for cross-process communication |
| `areal/experimental/weight_update/awex/fsdp_adapter.py`     | 334   | FSDP training-side adapter                            |
| `areal/experimental/weight_update/awex/megatron_adapter.py` | 278   | Megatron training-side adapter (TP/PP/CP/EP)          |
| `areal/experimental/weight_update/awex/sglang_adapter.py`   | 404   | SGLang inference-side adapter                         |
| `areal/experimental/weight_update/gateway/app.py`           | 589   | Gateway coordination service                          |
| `areal/experimental/training_service/worker/awex.py`        | 152   | Flask blueprint for training endpoints                |
| `areal/experimental/inference_service/sglang/awex.py`       | 96    | FastAPI endpoints for inference                       |
| `areal/experimental/inference_service/sglang/scheduler.py`  | 293   | AwexSchedulerBridge composing onto SGLang Scheduler   |
| `tests/experimental/weight_update/test_nccl_integration.py` | 890   | NCCL integration tests                                |

______________________________________________________________________

### 4.2 Scaffolding Rollout Workflow

**The single largest commit at 8,286 lines across 31 files.** Introduces a modular agent
execution framework for multi-step, multi-tool, multi-turn agent behaviors during
rollout.

#### What "Scaffolding" Means

Scaffolding refers to a modular agent execution framework vendored from NVIDIA's
TensorRT-LLM scaffolding module. It provides an abstraction layer between the RL
training loop and rollout generation, enabling complex agent behaviors — not just
single-pass text generation.

#### Core Abstractions

1. **Controller** (`core/controller.py`) — Defines generation episode logic as a Python
   generator yielding `Task` objects. Controllers can be composed via pipelines and
   parallel branches.

1. **Worker** (`core/worker.py`) — Executes tasks against an inference engine. The
   `SGLangWorker` wraps OpenAI-compatible completions API.

1. **ScaffoldingLlm** (`core/scaffolding_llm.py`) — Orchestrator running a dedicated
   asyncio event loop in a separate thread, dispatching tasks from controllers to
   workers. Configurable parallelism (`max_parallel_requests=64`).

1. **Task** / **TaskCollection** — Data containers for generation requests and tracing.

#### Controller Patterns

| Controller                   | Purpose                                                               |
| ---------------------------- | --------------------------------------------------------------------- |
| `NativeGenerationController` | Single-pass generation (baseline)                                     |
| `NativeChatController`       | Chat-format generation                                                |
| `MajorityVoteController`     | N parallel generations with majority-vote selection                   |
| `BestOfNController`          | N parallel generations with reward-based selection                    |
| `PipelineTrajectoryMaker`    | Composes generation + reward into a pipeline                          |
| `MultiTurnChatController`    | Multi-turn conversation with configurable reflection messages         |
| `SearchAgentController`      | Full tool-calling agent loop with search/visit tools and token budget |
| `TraceTrajectoryMaker`       | Wraps any controller with `ChatTracer` for per-turn credit assignment |
| `LLMJudgeController`         | Uses secondary LLM to judge answer correctness                        |

#### Significance for RL Training

The scaffolding framework extends AReaL's capabilities beyond single-turn RLHF:

- **Agentic RL**: Training models that can use tools, search, and iterate on solutions
- **Multi-turn credit assignment**: `ChatTracer` with configurable `reward_discount`
  enables backward propagation of rewards across conversation turns
- **Individual turn training**: "individual" export style creates separate
  `InteractionWithTokenLogpReward` objects per turn for per-turn PPO updates

#### Integration

The `ScaffoldingWorkflow` class (`examples/scaffolding/workflow.py`) extends
`RolloutWorkflow` and implements `agenerate()` via the scaffolding pipeline. Uses
placeholder logprobs (0.0) since `recompute_logprob=true` on the actor causes exact
logprobs to be recomputed during PPO update.

#### Key Files

| File                                           | Lines | Purpose                         |
| ---------------------------------------------- | ----- | ------------------------------- |
| `examples/scaffolding/controllers.py`          | 1,070 | Controller implementations      |
| `examples/scaffolding/workflow.py`             | 217   | ScaffoldingWorkflow integration |
| `examples/scaffolding/core/controller.py`      | 207   | Vendored controller base        |
| `examples/scaffolding/core/scaffolding_llm.py` | 230   | Orchestrator                    |
| `examples/scaffolding/search_scaffolding.py`   | 321   | Search agent controller         |
| `examples/scaffolding/worker.py`               | 249   | SGLang worker                   |
| `tests/test_controllers.py`                    | 935   | Controller tests                |
| `tests/test_scaffolding_llm_integration.py`    | 1,060 | Integration tests               |
| `tests/test_self_contained.py`                 | 795   | End-to-end tests                |

______________________________________________________________________

### 4.3 DPO (Direct Preference Optimization)

**Full end-to-end implementation across all three engine backends.** 2,158 lines across
26 files.

#### Loss Functions

Two variants configured via `DPOEngineConfig.loss_type`:

1. **Sigmoid (default)** — Original DPO loss (Rafailov et al., 2023):

   ```
   L = -logsigmoid(β * (log(π_chosen/π_ref_chosen) - log(π_rejected/π_ref_rejected)))
   ```

1. **IPO** — Identity Preference Optimization (Azar et al., 2023):

   ```
   L = (avg_logratio_chosen - avg_logratio_rejected - 1/(2β))²
   ```

   IPO normalizes log-ratios per-token before squared loss, making β comparable across
   variable-length sequences.

#### Key Implementation Details

- **fp64 scatter-add** for accumulating per-sequence log-probabilities
  (`dpo_pair_logratios` in `areal/utils/functional/functional.py`). fp32 accumulation
  can flip the log-ratio sign on sequences exceeding ~2K tokens.
- **Interleaved chosen/rejected pairs**:
  `[chosen_0, rejected_0, chosen_1, rejected_1, ...]`. The `DPOController` enforces
  `group_size=2` in RPC dispatching.
- **Loss mask shift**: Final position of each sequence explicitly zeroed to prevent
  cross-sequence leakage.

#### Comparison with PPO/GRPO

| Aspect          | DPO                                                 | PPO/GRPO                                            |
| --------------- | --------------------------------------------------- | --------------------------------------------------- |
| Reward model    | Not needed (implicit in preferences)                | Explicit reward function required                   |
| Rollout         | Offline (dataset of chosen/rejected pairs)          | Online (live generation)                            |
| Reference model | Required (frozen copy for KL)                       | Optional (for KL penalty)                           |
| Training loop   | Simple: load batch → compute ref logps → train step | Complex: generate → reward → advantage → PPO update |
| Data format     | Paired preference data (HH-RLHF style)              | Prompt + verifiable reward                          |

#### Config Options

```python
@dataclass
class DPOEngineConfig:
    beta: float = 0.1         # KL penalty coefficient
    loss_type: str = "sigmoid" # "sigmoid" | "ipo"
    # Full engine configurability (FSDP/Megatron/Archon, micro-batching, grad checkpointing)
```

#### Key Files

| File                                   | Lines | Purpose                                                         |
| -------------------------------------- | ----- | --------------------------------------------------------------- |
| `areal/trainer/dpo_trainer.py`         | 515   | DPO training loop                                               |
| `areal/trainer/dpo/dpo_engine.py`      | 207   | DPO-specific loss and engine wrapper                            |
| `areal/utils/functional/functional.py` | +64   | `compute_dpo_loss`, `dpo_pair_logratios`, `dpo_preference_loss` |
| `areal/dataset/hhrlhf.py`              | 50    | Anthropic HH-RLHF dataset loader                                |
| `examples/alignment/hhrlhf_dpo.yaml`   | 117   | Example config for Qwen2.5-7B on HH-RLHF                        |
| `tests/test_dpo.py`                    | 518   | Comprehensive tests                                             |

______________________________________________________________________

### 4.4 Rejection Sampling Configuration

**Unified configuration replacing legacy string-typed parameters.** 1,911 lines across
52 files.

#### Problem

In async RL training, the behavior policy (which generated rollout data) diverges from
the proximal policy (being updated). Extreme importance weights from off-policy data
destabilize training. Rejection sampling filters or truncates these samples.

#### Configuration Axes

| Parameter | Options                               | Description                                   |
| --------- | ------------------------------------- | --------------------------------------------- |
| `level`   | "token" / "sequence"                  | Granularity of filtering                      |
| `action`  | "mask" / "clamp"                      | Zero out loss_mask vs bound importance weight |
| `metric`  | "ratio" / "kl_k1" / "kl_k2" / "kl_k3" | Divergence measure                            |
| `agg`     | "sum" / "mean" / "max"                | Aggregation for sequence-level                |
| `upper`   | float (default 5.0)                   | Upper filtering threshold                     |
| `lower`   | float (optional)                      | Optional lower bound                          |

#### Metrics

- `ratio`: Direct importance ratio π_proximal / π_behave
- `kl_k1`: Forward KL estimator log(r) — can be negative
- `kl_k2`: Quadratic approximation 0.5 * (log r)² — always non-negative
- `kl_k3`: Exact forward KL estimator r - log(r) - 1 — always non-negative

#### Integration

`apply_rejection_sampling()` is called from PPO loss computation. Returns
`RejectionSamplingResult` containing modified `loss_mask`, `behave_imp_weight`, and
`filtered_fraction` statistics. Includes automatic migration from removed
`behave_imp_weight_mode`/`behave_imp_weight_cap` parameters.

______________________________________________________________________

### 4.5 Karmarkar-Karp Partitioning

**Alternative sequence packing algorithm.** 1,408 lines across 12 files.

#### Problem

Variable-length sequences in RL training cause micro-batch imbalance. Since all GPUs
synchronize at gradient all-reduce boundaries, the slowest GPU determines throughput.
Reducing max-min spread of token counts directly improves GPU utilization.

#### Algorithm

The Karmarkar-Karp Largest Differencing Method in `areal/utils/seqpack.py`:

1. Sort sequences by length ascending
1. Create one `_KKState` per sequence
1. Use a max-heap keyed by spread (max_sum - min_sum)
1. Iteratively pop two states with largest spreads, merge by pairing heaviest with
   lightest
1. Continue until one state remains containing all k partitions

**Complexity**: O(n log n) — same as FFD but produces measurably better balance.

#### Performance

From comparative benchmarks in the test suite:

- Over 100 random trials with bimodal sequence lengths, KK wins or ties ≥70% of the time
- For bimodal/uniform distributions: spread \< 15% of mean load
- For skewed (Pareto) distributions: spread \< 55% of mean load
- **Safety fallback**: If KK violates capacity constraints, automatically falls back to
  FFD

#### Configuration

```python
MicroBatchSpec.packing_algorithm = "kk"  # or "ffd" (default)
```

The `get_allocate_fn()` registry makes algorithm selection transparent to the training
pipeline.

______________________________________________________________________

### 4.6 LoRA for MoE Models

**Full Megatron-backend LoRA on Mixture-of-Experts architectures.** 325 lines across 7
files.

The core challenge is parameter naming conversion between Megatron's fused format
(`linear_fc1`, `linear_fc2`) and HuggingFace's per-expert format
(`experts.{i}.gate_proj`, `experts.{i}.up_proj`, `experts.{i}.down_proj`).

Two conversion functions in `areal/engine/megatron_utils/megatron_lora.py`:

- **`convert_qwen3_lora_to_hf`**: Dense-model LoRA (attention + MLP projections)
- **`convert_qwen3_moe_lora_to_hf`**: MoE-specific with grouped/per-expert adapter
  handling

Example config trains Qwen3-30B-A3B-Base (30B total, 3B active) with:

- Backend: `megatron:(attn:d1p6t1c1|ffn:d1p6t1e1)` — separate attention/FFN parallelism
- LoRA rank 32, alpha 32, targeting all linear layers
- vLLM inference with LoRA-enabled serving

______________________________________________________________________

### 4.7 Terminal Bench Training Example

**Complete training pipeline for the Terminal Bench benchmark.** 1,947 lines across 15
files.

Terminal Bench evaluates LLMs as autonomous software engineering agents. The agent
receives a task, has access to a Docker-containerized terminal, and must solve it by
writing code and executing commands.

Key components:

- **CamelTerminalAgent**: Interacts with Docker environment through tool calls (shell,
  file writing). Configurable token budgets, iteration limits, timeout handling.
- **CamelRLVRWorkflow**: `RolloutWorkflow` running multiple trajectories per task, with
  `filter_uniform_reward` (discards tasks where all trajectories get the same reward),
  `encourage_completion_reward` shaping, and per-turn reward discount (0.9).

This represents AReaL's expansion into training agentic capabilities beyond mathematical
reasoning, using verifiable rewards (unit test pass/fail) compatible with RLVR
methodology.

______________________________________________________________________

### 4.8 Inference Service Refactoring

**Backend abstraction and modularization.** 6 commits, ~3,601 lines.

#### Backend Protocol

`InfBridgeBackend` protocol (`areal/experimental/inference_service/backend.py`) defines
8 methods. Two implementations:

| Backend               | File                           | Target                                            |
| --------------------- | ------------------------------ | ------------------------------------------------- |
| `SGLangBridgeBackend` | `sglang/bridge.py` (148 lines) | SGLang `/generate` endpoint                       |
| `VLLMBridgeBackend`   | `vllm/bridge.py` (150 lines)   | vLLM `/v1/completions` and `/v1/chat/completions` |

Each backend is ~150 lines with no cross-dependencies. Adding a new inference backend
requires implementing one class with 8 methods.

#### Gateway Evolution

The inference gateway (`areal/experimental/inference_service/gateway/app.py`) gained:

- Authentication via bearer tokens (admin vs session keys)
- Router-based worker selection (round-robin, session pinning, model-based routing)
- SSE streaming support for chat completions
- Session lifecycle management
- External model registration for non-AReaL inference endpoints
- Parallel data proxy registration with rollback on failure

#### Key New Files

| File                                        | Lines | Purpose                          |
| ------------------------------------------- | ----- | -------------------------------- |
| `inference_service/backend.py`              | 94    | `InfBridgeBackend` protocol      |
| `inference_service/sglang/bridge.py`        | 148   | SGLang backend                   |
| `inference_service/sglang/scheduler.py`     | 293   | AwexSchedulerBridge              |
| `inference_service/sglang/launch_server.py` | 164   | Custom SGLang server launcher    |
| `inference_service/sglang/rpc_proxy.py`     | 54    | ZMQ RPC proxy                    |
| `inference_service/vllm/bridge.py`          | 150   | vLLM backend                     |
| `inference_service/router/state.py`         | 51    | Router state for external models |

______________________________________________________________________

### 4.9 Engine Evolution

#### `from_pretrained` Factory Method

`FSDPEngine.from_pretrained` (commit `e47dc676`) provides a simplified constructor
following the HuggingFace convention:

```python
engine = FSDPEngine.from_pretrained(
    model_path="Qwen/Qwen2.5-7B",
    dp_size=2, tp_size=4,
    dtype=torch.bfloat16,
    learning_rate=1e-5,
)
```

New files: `areal/engine/sglang_remote.py` and `areal/engine/vllm_remote.py` (44 lines
each) for `from_pretrained` on remote engines.

#### Built-in Memory Profiler

`MemoryProfilerConfig` (commit `d58cca56`):

- `profile_steps`: List of training steps for CUDA memory snapshots (default: \[0, 1\])
- `max_entries`: Ring buffer size (default: 100,000)
- Also avoids gathering Context Parallel logits — preventing unnecessary GPU memory
  allocation and cross-rank communication.

#### Additional Fixes

- **Multimodal tensor deduplication** (`349c6ed6`): Prevents duplicating vision tensors
  during FSDP micro-batch splitting
- **HCCL warmup** (`50f0a0b0`): Eagerly initializes NCCL/HCCL communicators to prevent
  race conditions on NPU
- **FSDP wrap fix** (`073adbf2`): Handles set-valued `wrap_class_names` correctly

______________________________________________________________________

### 4.10 Infrastructure Reliability

#### Two-Phase Teardown (commit `3ed3e817`)

Addresses a distributed systems race condition: rank-0 owns the TCPStore server, and
simultaneous termination caused "recvValue failed" errors during NCCL's heartbeat
monitoring.

Solution:

- **Phase 1**: Call `engine.destroy()` on all workers. Each engine executes a CPU
  barrier (gloo) then `dist.destroy_process_group()`.
- **Phase 2**: Only after Phase 1 completes, kill OS processes. The `reverse_order`
  parameter kills rank-0 last.

#### Additional Fixes

| Fix                        | Commit     | Impact                                           |
| -------------------------- | ---------- | ------------------------------------------------ |
| IPv6 proxy failures        | `e70b1934` | Data service works in IPv6-only environments     |
| Ray RPC serialization      | `f3d7e50a` | Handles ray.ObjectRef in payloads                |
| Integer device IDs         | `e8c1e1fd` | Ray RPC server accepts int device IDs            |
| Configurable setup_timeout | `fe91acc1` | Large models/slow networks during gateway init   |
| Data service seed          | `ad622efd` | Moved to worker-level config for reproducibility |
| Docker venv mount          | `7fbd077c` | Prevents container mount override of venv        |

______________________________________________________________________

## 5. .claude AI-Assisted Development Tooling

### 5.1 Inventory

The `.claude/` directory is one of the most comprehensive Claude Code integrations in an
open-source project:

| Category       | Count | Items                                                                                                                                                       |
| -------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agents**     | 8     | planner, simple-code-reviewer, code-verifier, fsdp-engine-expert, archon-engine-expert, megatron-engine-expert, algorithm-expert, launcher-scheduler-expert |
| **Skills**     | 7     | add-dataset, add-workflow, add-reward, add-archon-model, debug-distributed, add-unit-tests, commit-conventions                                              |
| **Commands**   | 7     | create-pr, gen-commit-msg, review-pr, translate-doc-zh, update-docker-image, upgrade-megatron-core, upgrade-vllm                                            |
| **Rules**      | 4     | api-config.md, code-style.md, distributed.md, testing.md                                                                                                    |
| **Data files** | 2     | review-pr-domains-and-signals.md, review-pr-templates.md                                                                                                    |
| **Hooks**      | 1     | check-expert-update.sh (PostToolUse)                                                                                                                        |

### 5.2 Commands

**`/create-pr`** (796 lines): Full PR creation workflow with 7 steps:

1. Verify prerequisites
1. Detect push remote (fork support)
1. Check existing PR
1. Fetch & rebase from origin/main
1. Squash commits to single commit
1. Analyze changes, generate PR title/description
1. Push and create/update PR via `gh`

Includes error handling for conflicts, push failures, and PR creation failures. Safety
checks with backup branch and confirmation prompts.

**`/gen-commit-msg`** (170 lines): 5-step commit message generator analyzing staged
changes, categorizing by type, determining scope from file paths, and generating
Conventional Commits format.

**`/review-pr`**: Sophisticated multi-phase PR review with domain/signal detection,
dynamic model routing by severity (CRITICAL → Opus, MEDIUM → Sonnet, LOW → Haiku), and 9
L1 domains with fine-grained L2 signals.

### 5.3 Skills

**`commit-conventions`** (180 lines): Always-loaded skill defining Conventional Commits
format. Includes:

- Type selection table
  (feat/fix/docs/gov/style/refactor/perf/test/build/ci/chore/revert)
- Scope inference from file path patterns (13 mappings)
- Format rules (imperative mood, 50-72 char subject, 72 char body wrap)
- 4 worked examples

**Domain-specific skills** (`add-dataset`, `add-workflow`, `add-reward`,
`add-archon-model`, `debug-distributed`, `add-unit-tests`): Step-by-step guides for
common development tasks, encoding project-specific patterns and conventions.

### 5.4 Agents

Stage-by-stage workflow:

1. **Planning Stage**: `planner` for architecture design and implementation planning
1. **Domain Expertise**: `fsdp-engine-expert`, `archon-engine-expert`,
   `megatron-engine-expert` for engine-specific guidance; `algorithm-expert` for RL
   algorithms; `launcher-scheduler-expert` for cluster deployment
1. **Code Formatting**: `code-verifier` for formatting, linting, and tests
1. **Code Quality**: `simple-code-reviewer` for logic issues and code smells

### 5.5 Rules

Four project-wide code quality standards:

- **`api-config.md`**: Configuration dataclass design patterns, `__post_init__`
  validation, field naming
- **`code-style.md`**: Logging, performance patterns, naming conventions, tensor
  conventions
- **`distributed.md`**: Distributed training patterns and constraints
- **`testing.md`**: Testing strategy and coverage requirements

### 5.6 Hooks

**`check-expert-update.sh`** (PostToolUse hook): Fires when files in engine/model
directories are modified, reminding the developer to update the corresponding expert
agent definition. Ensures agent knowledge stays current as the codebase evolves.

______________________________________________________________________

## 6. Archon Training Framework

### 6.1 Architecture Overview

The ArchonEngine (`areal/experimental/engine/archon_engine.py`, 1,550 lines) is a
substantial torch-native training backend implementing the `TrainEngine` abstract
interface alongside `FSDPEngine` and `MegatronEngine`.

**Key capabilities**:

- Full parallelism support: PP (Pipeline Parallelism), TP (Tensor Parallelism), CP
  (Context Parallelism), EP (Expert Parallelism), DP (Data Parallelism)
- `ArchonParallelDims` abstraction for mesh-based parallel group access
- Multiple engine subclasses: `ArchonPPOActor`, `ArchonPPOCritic`, `ArchonLMEngine`,
  `ArchonRWEngine`, `ArchonDPOEngine` (new)

### 6.2 Key Changes Since v1.0.3

#### Eager HCCL/NCCL Communicator Warmup (commit `50f0a0b0`)

The `warmup_process_groups()` call in `create_process_group` eagerly initializes
communicators before training begins, preventing HCCL/NCCL race conditions on NPU
hardware. Implementation in `areal/engine/core/distributed.py` handles: None groups,
duplicates (via `dict.fromkeys`), CPU-only platforms, and uninitialized dist.

#### Reentrant Offload Context (commit `db2a193b`)

The `_offload_depth` counter pattern prevents redundant offload/onload transitions in
nested contexts. Only the outermost call actually performs offload/onload — inner calls
are no-ops. This reduces GPU-CPU memory transfer overhead in complex training loops.

```python
self._offload_depth: int = 0  # Reentrance counter

# In _offload_aware_context:
if self._offload_depth == 0:
    self.onload()  # Only outermost call triggers onload
self._offload_depth += 1
# ... yield ...
self._offload_depth -= 1
if self._offload_depth == 0:
    self.offload()  # Only outermost call triggers offload
```

#### Two-Phase Teardown (commit `3ed3e817`)

Pre-destroy CPU barrier prevents noisy stderr backtraces from NCCL HeartbeatMonitor
threads. The `try/except` around the barrier with warning-level log makes teardown
resilient. `destroy()` is idempotent via `self.own_global_group = False`.

### 6.3 ArchonDPOEngine

New subclass (lines 1509-1549) extending ArchonEngine for DPO training:

```python
class ArchonDPOEngine(ArchonEngine):
    # Ensures mb_spec.granularity == 2 for paired DPO data
    # Routes to DPOController or DPOControllerV2 based on config._version
    # Provides as_controller() for distributed dispatch
```

**Controllers**:

- `DPOController` (v1): Legacy RPC-based dispatch with `group_size=2` pairing
- `DPOControllerV2` (v2): Gateway-based distributed dispatch

______________________________________________________________________

## 7. Architectural Assessment

### 7.1 Design Patterns

**Patterns consistently applied across the codebase:**

1. **`@runtime_checkable Protocol`** for interfaces (engine APIs, bridge backends, awex
   adapters). This is the project's primary abstraction mechanism, providing structural
   typing without inheritance chains — aligned with the stated preference for
   composition over inheritance.

1. **Lazy initialization**: Awex adapters created on first request, scaffolding
   components created on first episode. Good for reducing startup overhead in
   distributed environments.

1. **Flask blueprints for training workers, FastAPI for inference/gateway**: Reflects
   the sync vs async nature of the underlying workloads.

1. **`HttpRequest` dataclass as intermediary**: Separates "what to send" from "how to
   send it" in the inference bridge.

1. **Registry pattern**: `get_allocate_fn()` for packing algorithms,
   `PACKING_ALGORITHMS` set for validation.

### 7.2 Scalability Assessment

- **Awex**: Scales horizontally — transfer plans are per-rank, each rank only
  sends/receives its local shards. NCCL P2P avoids collective all-to-all bottleneck.
- **KK partitioning**: Directly improves throughput by reducing micro-batch imbalance,
  which matters more as sequence length variance increases in RL workloads.
- **Two-phase teardown**: Scales correctly using gloo (CPU) barriers that avoid GPU
  synchronization deadlocks.
- **Scaffolding**: Configurable parallelism (`max_parallel_requests=64`) enables
  high-throughput multi-step agent rollouts.

### 7.3 Technical Debt

1. **Awex adapter code duplication**: FSDP and Megatron training adapters share ~80%
   identical code in group init/execute/teardown methods. Adding a third training
   backend would require copy-pasting this boilerplate.

1. **Three engine backends with limited shared code**: FSDPEngine, MegatronEngine, and
   ArchonEngine independently implement process group creation, checkpoint management,
   and training loops. `areal/engine/core/` provides some shared utilities but is
   underutilized.

1. **Gateway responsibility accumulation**: The inference gateway has grown to include
   routing, auth, session management, model registration, worker lifecycle, and weight
   update coordination.

1. **Scaffolding placement**: Despite being a full workflow with extensive tests, it
   lives in `examples/scaffolding/` rather than `areal/workflow/` or
   `areal/experimental/`.

1. **Flask/FastAPI split**: Training side uses Flask (sync), inference side uses FastAPI
   (async). Intentional but creates cognitive overhead for maintainers.

### 7.4 Recommendations

| Priority   | Recommendation                                                                                           |
| ---------- | -------------------------------------------------------------------------------------------------------- |
| **High**   | Extract shared awex adapter logic into a base class to prevent duplication as new engine types are added |
| **Medium** | Add `from_pretrained` factory methods to ArchonEngine and MegatronEngine for API consistency             |
| **Medium** | Decompose inference gateway into focused sub-routers (routing, lifecycle, model management)              |
| **Medium** | Promote scaffolding workflow from `examples/` to `areal/experimental/` or `areal/workflow/`              |
| **Low**    | Unify Flask/FastAPI split documentation for maintainer onboarding                                        |
| **Low**    | Add shared utility for `max_new_tokens` computation in bridge backends                                   |
| **Low**    | Extract duplicate scope-inference tables in `.claude/` files to a shared data file                       |

______________________________________________________________________

## 8. Code Quality Review

### 8.1 Convention Adherence

**Logging**: All new code correctly uses `areal.utils.logging.getLogger()` with
PascalCase names: `"DPOEngine"`, `"DPOTrainer"`, `"AwexBlueprint"`,
`"AwexInferenceEndpoints"`, `"RLVRControllers"`, `"ScaffoldingWorkflow"`, `"SeqPack"`.
Only exception: vendored `core/controller.py` using stdlib logging.

**Naming**: `DPOEngineConfig`, `DPOConfig`, `ArchonDPOEngine`, `DPOTrainer`,
`DPOController` all follow established patterns (`XxxConfig`, `XxxEngine`, `XxxTrainer`,
`XxxController`).

**License headers**: All new files include `# SPDX-License-Identifier: Apache-2.0`.

**Import style**: No wildcard imports. Grouping follows stdlib/third-party/areal
ordering.

### 8.2 Test Coverage

| Area                 | Test File                       | Lines  | Quality                                           |
| -------------------- | ------------------------------- | ------ | ------------------------------------------------- |
| DPO Loss             | `test_dpo.py`                   | 518    | High — correctness, edge cases, error handling    |
| Rejection Sampling   | `test_rejection_sampling.py`    | 839    | Very High — all mode/level/format combos          |
| KK Algorithm         | `test_kk_allocate.py`           | 603    | Very High — internals + integration + comparative |
| KK E2E               | `test_kk_e2e.py`                | 177    | Good — distributed comparison                     |
| Process Group Warmup | `test_warmup_process_groups.py` | 89     | Good — all guard branches                         |
| Scaffolding          | 4 files                         | ~2,995 | Very High for examples/ code                      |
| Awex Weight Update   | 9 files                         | ~2,236 | Extensive — NCCL, disk, controller, KV store      |

**Total new test code: 5,000+ lines.** Test-to-feature code ratio of 1.18:1 is
excellent.

### 8.3 Findings by Severity

#### MEDIUM (3)

1. **`ArchonDPOEngine.__init__` silently mutates config**: Silently corrects
   `mb_spec.granularity` to 2 via `deepcopy` rather than raising `ValueError`. Per
   `api-config.md`, validation should use `__post_init__` with clear errors.

1. **`apply_rejection_sampling` cyclomatic complexity**: ~200 lines with deeply nested
   branches across `level × action × ndim × agg × metric`. Would benefit from extraction
   into smaller helper functions.

1. **Training-side awex.py inconsistent error handling**: Lacks explicit try/except
   blocks compared to inference-side counterpart. Relies entirely on `run_endpoint` for
   error handling.

#### LOW (10)

1. Duplicate scope inference tables across 3 `.claude/` files
1. `create-pr.md` backup step not enforced in workflow sequence
1. Logger instantiation inside `__init__` for ArchonDPOEngine (should be module-level)
1. `_version` private field used for control flow routing in
   `ArchonDPOEngine.as_controller`
1. Redundant `hasattr` checks in `DPOTrainer.close()`
1. Unused `entropy`/`vocab_*_logits` params in `compute_dpo_loss` lack explanatory
   comments
1. Vendored `core/controller.py` uses stdlib logging instead of `areal.utils.logging`
1. `getattr` fallback in `data.py` for `packing_algorithm` — field should be formally
   defined
1. Thread-safety gap in awex training-side `_state` dict pattern (missing
   `threading.Lock`)
1. Inference-side `debug/randomize_parameters` endpoint missing from training side

**Overall Assessment**: Code quality since v1.0.3 is strong. The DPO implementation has
careful numerical handling, the KK algorithm is a clean drop-in with excellent tests,
and the `.claude/` tooling is among the most thorough in open-source projects. No
critical security issues or correctness bugs identified.

______________________________________________________________________

## 9. Strategic Direction Analysis

The changes since v1.0.3 reveal four strategic shifts in AReaL's trajectory:

### From Single-Turn RLHF to Agentic RL

The combination of scaffolding, Terminal Bench, and multi-turn credit assignment
represents a strategic shift toward training LLMs as autonomous agents rather than just
response generators. The scaffolding framework's composable controller architecture
enables training on increasingly complex agent behaviors without changing core training
infrastructure.

### From Synchronous to Asynchronous Training

The Awex system, combined with rejection sampling, enables fully asynchronous RL
training where rollout and training happen concurrently on different hardware. Rejection
sampling provides the safety net for policy divergence, with configurable metrics and
actions at both token and sequence granularity.

### From PPO-Only to Multi-Algorithm Alignment

Adding DPO (sigmoid + IPO variants) provides an offline alternative to PPO when
preference data is available. The shared engine infrastructure (FSDP/Megatron/Archon)
means DPO immediately benefits from all engineering optimizations.

### From Uniform to Adaptive Batching

The KK partitioning algorithm directly addresses computational efficiency by reducing
GPU idle time at synchronization barriers. The measured 70%+ win rate over FFD
translates to meaningful throughput improvements for variable-length RL workloads.

### Quantitative Health Indicators

| Indicator                      | Value                  | Signal                       |
| ------------------------------ | ---------------------- | ---------------------------- |
| Feature-to-fix ratio           | 1.64:1                 | Healthy balance              |
| Test-to-feature code ratio     | 1.18:1                 | Strong testing discipline    |
| New files added vs deleted     | 105:1                  | Growth phase                 |
| Community breadth              | 18 contributors        | Active open-source community |
| Conventional commit compliance | 98%                    | Consistent standards         |
| Development acceleration       | 2.9x (Week 1 → Week 2) | Increasing momentum          |

______________________________________________________________________

## 10. New File Inventory

### New Directories Created

| Directory                                      | Files | Purpose                       |
| ---------------------------------------------- | ----- | ----------------------------- |
| `areal/experimental/weight_update/`            | ~15   | Awex weight update system     |
| `areal/experimental/inference_service/sglang/` | 6     | SGLang inference backend      |
| `areal/experimental/inference_service/vllm/`   | 2     | vLLM inference backend        |
| `areal/trainer/dpo/`                           | 2     | DPO trainer module            |
| `examples/scaffolding/`                        | ~30   | Scaffolding rollout framework |
| `examples/terminal_bench/`                     | ~15   | Terminal Bench RL training    |
| `tests/experimental/weight_update/`            | ~9    | Weight update tests           |

### Files Deleted

| File                                                                     | Reason                                        |
| ------------------------------------------------------------------------ | --------------------------------------------- |
| `areal/experimental/inference_service/data_proxy/backend.py` (340 lines) | Subsumed by sglang/vllm submodule refactoring |

### Files Moved

| From                                         | To                                | Reason               |
| -------------------------------------------- | --------------------------------- | -------------------- |
| `inference_service/data_proxy/inf_bridge.py` | `inference_service/inf_bridge.py` | Module restructuring |
| `assets/*.png` (19 files)                    | `assets/figures/*.png`            | Asset organization   |

### Most Frequently Modified Files

| File                                         | Commits Touching                                                   |
| -------------------------------------------- | ------------------------------------------------------------------ |
| `areal/api/cli_args.py`                      | 5 (RejectionSampling, DPO, KK, memory profiler, eval-before-train) |
| `areal/engine/fsdp_engine.py`                | 5 (from_pretrained, DPO, multimodal, warmup, offload)              |
| `areal/experimental/engine/archon_engine.py` | 4 (DPO, warmup, teardown, offload)                                 |
| `areal/engine/megatron_engine.py`            | 4 (DPO, LoRA MoE, warmup, offload)                                 |
| `docs/en/cli_reference.md`                   | 5 (regenerated for new CLI fields)                                 |

______________________________________________________________________

*Analysis generated 2026-04-27 using 5 parallel Opus 4.6 agents: Explore (codebase
inventory), Architect-Reviewer (design assessment), Code-Reviewer (quality audit),
Data-Analyst (quantitative metrics), Data-Scientist (ML/RL feature analysis).*
