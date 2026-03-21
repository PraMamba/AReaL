# AReaL Claude Code 定制化使用教程：Agents、Commands 与 Skills

## 概述

AReaL 项目通过 `.claude/` 目录下的三大定制化机制，构建了一套完整的 AI 辅助开发体系。
这三类机制各司其职，协同工作：

| 机制 | 目录 | 触发方式 | 核心用途 |
|------|------|---------|---------|
| **Agents** | `.claude/agents/` | 自动（PROACTIVE）或按需调用 | 子进程专家，执行特定领域任务 |
| **Commands** | `.claude/commands/` | 用户主动调用（`/command-name`） | 执行具体操作（创建 PR、生成 commit message） |
| **Skills** | `.claude/skills/` | 用户主动调用（`/skill-name`） | 交互式分步开发指南 |

辅助机制：

| 机制 | 目录 | 触发方式 | 核心用途 |
|------|------|---------|---------|
| **Rules** | `.claude/rules/` | 路径匹配自动加载 | 编码标准，编辑对应路径文件时自动注入 |
| **Data** | `.claude/data/` | 被 Commands 引用 | 结构化数据（变更类型表、审查模板库） |
| **Hooks** | `.claude/hooks/` | 工具调用后自动触发 | Shell 脚本，在特定事件后执行 |
| **Settings** | `.claude/settings.json` | 全局配置 | Hook 注册、权限配置 |

```
.claude/
├── agents/                    # 8 个专家 Agent
│   ├── planner.md
│   ├── code-verifier.md
│   ├── simple-code-reviewer.md
│   ├── algorithm-expert.md
│   ├── fsdp-engine-expert.md
│   ├── archon-engine-expert.md
│   ├── megatron-engine-expert.md
│   └── launcher-scheduler-expert.md
├── commands/                  # 3 个用户命令
│   ├── pr-review.md
│   ├── create-pr.md
│   └── gen-commit-msg.md
├── skills/                    # 6 个开发技能
│   ├── add-workflow/SKILL.md
│   ├── add-reward/SKILL.md
│   ├── add-dataset/SKILL.md
│   ├── add-unit-tests/SKILL.md
│   ├── add-archon-model/SKILL.md
│   └── debug-distributed/SKILL.md
├── rules/                     # 4 条路径规则
│   ├── code-style.md
│   ├── api-config.md
│   ├── distributed.md
│   └── testing.md
├── data/                      # 命令数据文件
│   ├── pr-review-change-types.md
│   └── pr-review-templates.md
├── hooks/                     # 自动化钩子
│   └── check-expert-update.sh
└── settings.json              # 全局设置
```

---

## 第一部分：Agents（智能体）

### 1.1 什么是 Agent

Agent 是以子进程运行的专家智能体。每个 Agent 有独立的模型选择、工具权限和激活条件。
Agent 以 Markdown 文件定义，文件头部的 YAML frontmatter 指定元数据：

```yaml
---
name: code-verifier              # Agent 名称
description: Code verification   # 描述（用于自动激活判断）
tools:                           # 可用工具列表
  - Read
  - Grep
  - Glob
  - Bash                         # 注意：只有 code-verifier 有 Bash
model: haiku                     # 使用的模型（haiku/sonnet/opus）
---
```

### 1.2 Agent 的三种激活模式

| 模式 | 关键词 | 含义 | 示例 |
|------|--------|------|------|
| **PROACTIVE** | `Use PROACTIVELY` | 代码变更后自动激活，无需用户请求 | `code-verifier`, `simple-code-reviewer`, `planner` |
| **按需** | `Use when requested` / `Use only when` | 检测到相关话题时激活 | `algorithm-expert`, `fsdp-engine-expert` |
| **手动** | `Use when user modifies` | 用户修改特定路径代码时激活 | `launcher-scheduler-expert` |

### 1.3 AReaL 的 8 个 Agent

#### 第一层：规划（编码前）

**`planner`** -- 实现规划器

| 属性 | 值 |
|------|---|
| 模型 | **Opus**（深度推理） |
| 工具 | Read, Grep, Glob, Task（只读） |
| 激活 | **PROACTIVE** -- 多文件变更、新功能、架构决策前自动激活 |

工作流程：
1. **理解需求** -- 最多问 2-3 个关键问题
2. **研究代码** -- 搜索相似实现、调用方、依赖关系、测试
3. **输出计划** -- 简单任务用 Quick Path，复杂任务用 Full Plan

```markdown
## Summary
[1-2 sentences]

## Changes
| File | Action | Purpose |
|------|--------|---------|

## Steps
1. Step 1
2. Step 2

## Patterns to Follow
## Risks
## Testing
```

**设计要点**：Planner 是**只读**的 -- 它不修改代码，只做研究和规划。这确保了在编码前
有充分的架构思考。

#### 第二层：验证（编码后）

**`code-verifier`** -- 代码验证器

| 属性 | 值 |
|------|---|
| 模型 | **Haiku**（快速执行） |
| 工具 | Read, Grep, Glob, **Bash**（唯一可执行命令的 Agent） |
| 激活 | **PROACTIVE** -- 代码变更后自动激活 |

5 阶段工作流：
1. **识别变更** -- `git status --short`, `git diff --name-only HEAD`
2. **格式化和 Lint** -- `pre-commit run --all-files`
3. **运行测试** -- GPU 感知路由（先检测 `torch.cuda.is_available()`）
4. **文档检查** -- 如果 `cli_args.py` 变更，重新生成 CLI 文档
5. **结构化报告** -- 输出每项检查的 PASS/FAIL/SKIP 状态

**设计要点**：使用 Haiku（最快最便宜的模型），因为任务是机械性的：运行命令、解析输出、
报告结果。可自动修复格式问题并提醒重新 stage。

**`simple-code-reviewer`** -- 轻量代码审查器

| 属性 | 值 |
|------|---|
| 模型 | **Sonnet**（平衡分析） |
| 工具 | Read, Grep, Glob（**只读**，不能执行命令） |
| 激活 | **PROACTIVE** -- 代码变更后自动激活 |

3 个检查方向：
- **AReaL 特定模式** -- 日志使用 `getLogger()`、async 正确性、Tensor 形状约定
- **常见问题** -- 缺少 `await`、阻塞 I/O、资源泄漏
- **分布式代码** -- 同步缺失、设备不匹配、Mesh 维度错误

**设计要点**：与 code-verifier 的分工 -- verifier **运行**东西（验证），reviewer
**读**东西（审查）。reviewer 是只读的，不会"顺手修了"而跳过分析。

#### 第三层：领域专家（按需）

| Agent | 模型 | 领域 | 激活条件 |
|-------|------|------|---------|
| `algorithm-expert` | Opus | GRPO/PPO/DAPO 算法 | RL 算法相关问题 |
| `fsdp-engine-expert` | Opus | FSDP2 分片、参数分布 | FSDP 引擎代码变更 |
| `archon-engine-expert` | Opus | MoE/EP/ETP、Archon 模型 | Archon 引擎代码变更 |
| `megatron-engine-expert` | Opus | 流水线并��、微批调度 | Megatron 引擎代码变更 |
| `launcher-scheduler-expert` | Sonnet | 集群部署（Slurm/Ray/K8s） | Launcher/Scheduler 代码修改 |

每个专家 Agent 都包含：
- **核心概念** -- 该子系统的架构和关键类
- **配置指南** -- 如何正确配置
- **诊断表** -- Symptom → Cause → Fix 对照表
- **常见使用模式** -- 典型场景和配置示例

### 1.4 Agent 的协同工作流

在一个典型的开发周期中：

```
planner (Opus)         →  规划架构和实现步骤
    ↓
[用户/AI 编码]          →  参考领域专家获取指导
    ↓
code-verifier (Haiku)  →  运行 pre-commit + 测试
    ↓
simple-code-reviewer (Sonnet) → 读代码找逻辑问题
    ↓
/pr-review (动态)      →  创建 PR 时的全面审查
```

### 1.5 如何编写一个 Agent

Agent 文件的标准结构：

```markdown
---
name: my-expert                   # 名称（kebab-case）
description: >                    # 描述（用于自动激活判断）
  Expert on X. Use PROACTIVELY    # "PROACTIVELY" = 自动激活
  when Y happens.                 # "Use when requested" = 手动
tools:                            # 可用工具
  - Read
  - Grep
  - Glob
  # - Bash                        # 谨慎授予：只在需要执行命令时
  # - Task                        # 允许��用子 Agent
model: sonnet                     # haiku/sonnet/opus
---

# Agent 标题

角色描述和专业领域。

## When to Activate
触发条件列表。

## Core Concepts
核心概念和关键类/方法。

## Configuration
配置方式和参数说明。

## Common Usage Patterns
常见使用场景。

## Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
诊断对照表。

## Resources
相关文件路径。

---
<!--
================== MAINTAINER GUIDE ==================
维护说明（不会被 AI 展示给用户）
=======================================================
-->
```

**关键决策**：

| 决策 | 选择建议 |
|------|---------|
| 模型选择 | Haiku = 机械任务；Sonnet = 分析任务；Opus = 深度推理 |
| 是否给 Bash | 只有需要执行命令的 Agent 才给（如 code-verifier） |
| 是否 PROACTIVE | 每次代码变更都应触发的用 PROACTIVE，专业领域的用按需 |
| 是否给 Task | 需要调用子 Agent 的给（如 planner 需要探索代码库） |

---

## 第二部分：Commands（命令）

### 2.1 什么是 Command

Command 是用户通过 `/command-name` 主动调用的操作脚本。与 Agent 不同，Command
不是独立进程，而是将指令注入到当前对话上下文中执行。

Command 文件同样使用 YAML frontmatter：

```yaml
---
name: create-pr
description: >
  Rebase from the latest `origin/main`, squash the commits,
  and create a PR on github. Invoke with /create-pr.
---
```

### 2.2 AReaL 的 3 个 Command

#### `/gen-commit-msg` -- 生成 Commit Message

**来源**：`.claude/commands/gen-commit-msg.md`

**用法**：

```
/gen-commit-msg [--amend] [--scope <scope>]
```

**工作流程**：
1. 分析 staged changes（`git diff --cached`）
2. 自动分类变更类型（feat/fix/docs/refactor/test/chore/perf）
3. 从变更文件路径推断 scope（workflow/engine/reward/dataset/api/...）
4. 生成 Conventional Commits 格式消息
5. 预览并确认后提交

**消息格式**：

```
<type>(<scope>): <subject>

<body>

Key changes:
- change 1
- change 2

Refs: #123, #456
```

**Scope 映射**：

| 文件路径 | Scope |
|---------|-------|
| `areal/workflow/` | `workflow` |
| `areal/engine/` | `engine` |
| `areal/reward/` | `reward` |
| `areal/dataset/` | `dataset` |
| `areal/api/` | `api` |
| `areal/utils/` | `utils` |
| `areal/infra/` | `infra` |
| `docs/` | `docs` |

#### `/create-pr` -- 创建 Pull Request

**来源**：`.claude/commands/create-pr.md`

**用法**：

```
/create-pr [--draft] [--base <branch>]
```

**7 步工作流**：

1. **检查前提** -- 确认不在 main 分支，无未提交变更
2. **检查已有 PR** -- 如存在，询问是否强制更新
3. **Fetch + Rebase** -- `git fetch origin main && git rebase origin/main`
4. **Squash Commits** -- `git reset --soft origin/main`，然后用 `/gen-commit-msg`
   逻辑生成 squash commit message
5. **分析变更** -- 分类所有变更、确定类型和 scope
6. **生成 PR 标题和描述** -- 严格遵循 `.github/PULL_REQUEST_TEMPLATE.md` 模板
7. **Push + 创建/更新 PR** -- `git push -f -u origin` + `gh pr create`

**安全检查**（在关键步骤前确认）：
- 未提交变更 → 停止
- 在 main 分支 → 停止
- Force push → 警告并确认
- 已有 PR → 询问权限

**PR 描述模板**：

```markdown
## Description
[变更说明]

## Related Issue
Fixes #(issue)

## Type of Change
- [ ] Bug fix / New feature / Breaking change / ...

## Checklist
- [ ] I have run formatting tools
- [ ] I have run relevant unit tests
- [ ] I have added tests for new functionality
...
```

#### `/pr-review` -- 智能 PR 代码审查

**来源**：`.claude/commands/pr-review.md`

这是最复杂的 Command，实现了一个 4 阶段动态审查流水线。

**用法**：

```
/pr-review [PR-URL-or-number] [--quick] [--economy]
```

**4 阶段流水线**：

```
阶段 1: 深度分析 [Haiku + Sonnet]
    ├─ PR 状态检查
    ├─ 变更摘要
    └─ 变更类型检测（31 种类型，4 个风险等级）
    ↓
阶段 2: 动态 Agent 规划 [Sonnet]
    ├─ 按风险区域生成任务
    ├─ 合并相关变更
    ├─ 选择模型（CRITICAL→Opus, MEDIUM→Sonnet, LOW→Haiku）
    └─ 从 23+ 模板中选择审查检查表
    ↓
阶段 3: 并行执行 [动态模型]
    └─ 所有审查 Agent 并行运行
    ↓
阶段 4: 置信度评分 [Haiku]
    ├─ 每项发现评分 0-100
    ├─ 过滤误报（分数 0：已有问题、设计如此、linter 可捕获）
    └─ 生成分类汇总报告
```

**31 种变更类型**（来自 `.claude/data/pr-review-change-types.md`）：

| 风险等级 | 数量 | 默认模型 | 示例 |
|---------|------|---------|------|
| CRITICAL | 8 | Opus | ARCHON_CORE, FSDP_CORE, MEGATRON_CORE, DCP_CHECKPOINT |
| HIGH | 8 | Opus | DISTRIBUTED_COMM, DTENSOR, MOE_LAYER, TRAINER_CORE |
| MEDIUM | 12 | Sonnet | TENSOR_OPS, WORKFLOW_ENGINE, API_CONFIG, REWARD |
| LOW | 3 | Haiku | TESTS, DOCS, CONFIG_ONLY |

**23+ 审查模板**（来自 `.claude/data/pr-review-templates.md`）：

- 框架专属：Archon EP/ETP、FSDP Core、Pipeline Parallelism、DCP Checkpoint...
- 通用：Logic/Boundary、Concurrency、Tensor Shape、Numerical Stability...
- 轻量：Documentation、Test Coverage、Import Check、Security...

**风险联动规则** -- 当检测到特定变更时，自动触发关联审查：

```
EP 变更       → 自动检查 FSDP 交互 + dp_shard_mod_ep mesh
Megatron 变更 → 自动检查 Pipeline + AC
REWARD 变更   → 自动检查 Workflow 交互 + AsyncRewardWrapper
DCP 变更      → 自动检查 FSDP2 集成 + 分布式一致性
```

**模型选择策略**：

| 模式 | CRITICAL/HIGH | MEDIUM | LOW |
|------|---------------|--------|-----|
| 默认 | Opus | Sonnet | Haiku |
| `--quick` | Sonnet | Sonnet | Sonnet |
| `--economy` | Sonnet | Haiku | Haiku |

### 2.3 Command vs Agent 的区别

| 维度 | Agent | Command |
|------|-------|---------|
| 运行方式 | 子进程（独立上下文） | 注入当前对话 |
| 触发方式 | 自动或按需 | 仅用户主动调用 `/name` |
| 典型用途 | 持续性专家（规划、验证、审查） | 一次性操作（创建 PR、生成消息） |
| 文件位置 | `.claude/agents/*.md` | `.claude/commands/*.md` |
| 可组合性 | Agent 可调用其他 Agent（via Task tool） | Command 可引用其他 Command 逻辑 |

---

## 第三部分：Skills（技能）

### 3.1 什么是 Skill

Skill 是**交互式分步开发指南**。与 Command（执行操作）不同，Skill 提供的是**引导式
工作流** -- 它告�� AI 按什么步骤完成一个开发任务，包含代码模板、参考实现和常见错误
提醒。

Skill 文件位于子目录中：

```
.claude/skills/
├── add-workflow/SKILL.md      # /add-workflow
├── add-reward/SKILL.md        # /add-reward
├── add-dataset/SKILL.md       # /add-dataset
├── add-unit-tests/SKILL.md    # /add-unit-tests
├── add-archon-model/SKILL.md  # /add-archon-model
└── debug-distributed/SKILL.md # /debug-distributed
```

### 3.2 AReaL 的 6 个 Skill

#### `/add-workflow` -- 添加 RolloutWorkflow

**4 步流程**：

1. **创建文件** `areal/workflow/<name>.py`
   - 继承 `RolloutWorkflow`
   - 实现 `async def arun_episode()`
   - 使用 `AsyncRewardWrapper` 包装奖励函数
2. **注册** 到 `areal/workflow/__init__.py`
3. **更新训练脚本** 使用新 workflow
4. **添加测试** `areal/tests/test_<name>_workflow.py`

**关键要求**：
- `arun_episode` 必须是 `async def` 且非阻塞
- 使用 `aiofiles` 替代 `open()` 做文件操作
- 输出 Tensor 形状遵循 `[batch, seq_len, ...]` 约定

**参考实现**：`areal/workflow/multi_turn.py`, `areal/workflow/rlvr.py`

#### `/add-reward` -- 添加奖励函数

**4 步流程**：

1. **创建文件** `areal/reward/<name>.py`
   - 函数签名：`def <name>_reward_fn(prompt, completions, prompt_ids, completion_ids, **kwargs) -> float`
   - 异常时返回 `0.0`，不 raise
2. **注册** 到 `areal/reward/__init__.py`
3. **处理阻塞操作** -- 如需 API 调用，由 workflow 的 `AsyncRewardWrapper` 处理
4. **添加测试** -- 正确答案、错误答案、边界情况

**关键要求**：
- 确定性：相同输入 → 相同输出
- 返回 float，不返回 Tensor
- 使用 `areal.utils.logging.getLogger()`，不用 `print`

**参考实现**：`areal/reward/gsm8k.py`, `areal/reward/geometry3k.py`

#### `/add-dataset` -- 添加数据集

**4 步流程**：

1. **创建文件** `areal/dataset/<name>.py`
   - SFT 数据集：返回包含 `input_ids` 和 `loss_mask` 的 HuggingFace `Dataset`
   - RL 数据集：返回包含 `messages` 和 `answer` 的 HuggingFace `Dataset`
2. **注册** 到 `areal/dataset/__init__.py`
3. **配置**（可选）-- 在 `areal/api/cli_args.py` 添加特殊配置
4. **添加测试** -- 验证加载、字段完整性

**参考实现**：`areal/dataset/gsm8k.py`, `areal/dataset/geometry3k.py`

#### `/add-unit-tests` -- 添加单元测试

**6 步流程**：

1. **理解测试类型** -- Unit（直接 pytest）vs Distributed（torchrun）
2. **创建测试文件** -- 命名 `test_<module>_<feature>.py`
3. **编写测试函数** -- Arrange-Act-Assert ��式
4. **添加 Pytest Markers** -- `slow`/`ci`/`gpu`/`multi_gpu`
5. **Mock 分布式环境** -- `torch.distributed.fake_pg`
6. **处理 GPU 依赖** -- `@pytest.mark.skipif(not CUDA_AVAILABLE, ...)`

**CI 策略**：

```python
# 无标记 → CI 运行（快速测试）
def test_fast(): ...

# @slow → CI 不运行（除非加 @ci）
@pytest.mark.slow
def test_slow(): ...

# @slow + @ci → CI 运行（关键慢测试）
@pytest.mark.slow
@pytest.mark.ci
def test_slow_but_critical(): ...
```

**与其他 Skill 的集成**：
- `/add-dataset` → 运行 `/add-unit-tests` 添加数据集测试
- `/add-workflow` → 运行 `/add-unit-tests` 添加 workflow 测试
- `/add-reward` → 运行 `/add-unit-tests` 添加奖励函数测试

#### `/add-archon-model` -- 添加 Archon 模型架构

**10 步流程**（最复杂的 Skill）：

1. **分析目标模型** -- 读取 HF `config.json` 和 `modeling_*.py`
2. **选择参考实现** -- `qwen2`（纯 Dense）或 `qwen3`（Dense + MoE + QK norm）
3. **实现 `args.py`** -- 映射 HF config 到 Archon ModelArgs
4. **实现 `model.py`** -- Attention、FFN、TransformerBlock、顶层模型
5. **实现 `rope.py`** -- RoPE 变体（标准则复用 qwen2）
6. **实现 `state_dict_adapter.py`** -- HF ↔ Archon 权重键名映射（最易出错）
7. **实现 `parallelize.py`** -- TP → EP → CP → AC → compile → FSDP 顺序
8. **创建 `spec.py`** -- 组装 `ModelSpec` 并注册
9. **注册** 到 `__init__.py`
10. **分阶段测试** -- Args → State Dict → Weight Completeness → Forward Precision

**文件清单**：

```
areal/experimental/models/archon/<model>/
├── __init__.py
├── spec.py                          # ModelSpec 注册
├── model/
│   ├── args.py                      # ModelArgs 数据类
│   ├── model.py                     # 模型架构
│   ├── rope.py                      # RoPE 实现
│   └── state_dict_adapter.py        # 权重映射
└── infra/
    └── parallelize.py               # 并行策略
```

#### `/debug-distributed` -- 调试分布式训练

**4 类问题的调试指南**：

1. **Hang（死锁）** -- `py-spy dump --pid <PID>` 获取调用栈
2. **Wrong Results** -- 检查 DTensor placements、梯度 reduction
3. **OOM** -- 检查 FSDP 覆盖率、内存使用
4. **通信错误** -- NCCL 诊断、Device Mesh 验证

**调试环境变量**：

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL
NCCL_DEBUG=INFO
CUDA_LAUNCH_BLOCKING=1          # 同步 CUDA（慢，仅调试用）
```

### 3.3 Skill 的结构模式

每个 Skill 文件遵循统一结构：

```markdown
---
name: add-<component>
description: Guide for adding ... Use when user wants to ...
---

# 标题

## When to Use
触发条件。

## Prerequisites
前提条件。

## Step-by-Step Guide

### Step 1: 创建文件
代码模板。

### Step 2: 注册
注册步骤。

### Step 3: 配置（可选）
配置说明。

### Step 4: 添加测试
测试模板。

## Reference Implementations
| Name | File | Description |
参考实现表。

## Key Requirements
关键要求列表。

## Common Mistakes
- ❌ 常见错误 1
- ❌ 常见错误 2

---
<!-- MAINTAINER GUIDE -->
```

### 3.4 Command vs Skill 的区别

| 维度 | Command | Skill |
|------|---------|-------|
| 用途 | 执行操作 | 引导开发 |
| 交互性 | 低（执行后输出结果） | 高（分步引导，每步有模板和检查） |
| 产出 | 具体动作完成（PR 创建、commit 提交） | 代码文件、测试、注册 |
| 举例 | `/create-pr` 创建一个 PR | `/add-reward` 引导完成奖励函数开发 |
| 可复用性 | 任何项目均可使用 | 针对 AReaL 特定架构 |

---

## 第四部分：支撑机制

### 4.1 Rules -- 路径自动加载的编码规则

**来源**：`.claude/rules/`

Rules 通过路径匹配自动加载，无需用户操作。当 AI 编辑匹配路径的文件时，对应 Rule
自动注入上下文。

| Rule 文件 | 作用域 | 核心内容 |
|-----------|--------|---------|
| `code-style.md` | 全局 | 设计模式、日志规范、性能模式、命名约定、Tensor 约定 |
| `api-config.md` | `areal/api/**` | 数据类字段排序、验证、向后兼容协议 |
| `distributed.md` | `areal/engine/**`, `areal/experimental/**` | 进程组、DeviceMesh、通信陷阱表 |
| `testing.md` | `**/tests/**`, `test_*.py` | Pytest markers、Arrange-Act-Assert、GPU skip、断言 |

**工作原理**：当编辑 `areal/api/cli_args.py` 时，`code-style.md`（全局）和
`api-config.md`（API 路径）同时激活。AI 在生成代码时自动遵循这些标准。

### 4.2 Data -- 命令的结构化数据

**来源**：`.claude/data/`

Data 文件不直接被用户调用，而是被 Command（特别是 `/pr-review`）引用的结构化数据：

- `pr-review-change-types.md` -- 31 种变更类型定义、文件路径模式、代码模式、风险联动
  规则
- `pr-review-templates.md` -- 23+ 审查任务模板，每个包含适用条件和检查清单

### 4.3 Hooks -- PostToolUse 自动化

**来源**：`.claude/settings.json`, `.claude/hooks/check-expert-update.sh`

配置：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/check-expert-update.sh"
          }
        ]
      }
    ]
  }
}
```

每次使用 Write 或 Edit 工具修改代码时，`check-expert-update.sh` 自动运行，检查修改
的文件路径是否匹配某个领域专家 Agent：

| 代码路径 | 对应专家 Agent |
|---------|---------------|
| `archon/` 或 `archon_engine*` | `archon-engine-expert.md` |
| `fsdp_engine*` 或 `fsdp/` | `fsdp-engine-expert.md` |
| `megatron*` | `megatron-engine-expert.md` |
| `trainer/ppo/` 或 `workflow/` 或 `reward/` | `algorithm-expert.md` |

匹配时输出提醒：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expert Update Reminder
Modified: areal/engine/fsdp_engine.py
Consider updating: .claude/agents/fsdp-engine-expert.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

这创造了一个**自我改进的反馈循环**：代码变更 → Hook 提醒 → 更新专家文档 → 未来审查
更准确。

---

## 第五部分：完整工作流示例

### 场景：添加一个新的 MATH 数据集 + 奖励函数

```
1. /add-dataset math
   → Skill 引导创建 areal/dataset/math.py
   → 注册到 __init__.py

2. /add-reward math
   → Skill 引导创建 areal/reward/math_reward.py
   → 注册到 __init__.py

3. /add-unit-tests
   → Skill 引导创建测试文件

4. [AI 编码完成]

5. code-verifier 自动激活（PROACTIVE）
   → 运行 pre-commit
   → 运行相关测试
   → 报告 PASS/FAIL

6. simple-code-reviewer 自动激活（PROACTIVE）
   → 检查 AReaL 模式、分布式正确性
   → 报告 Critical Issues / Suggestions / Looks Good

7. /gen-commit-msg
   → 分析变更，生成 feat(dataset): add MATH dataset and reward

8. /create-pr
   → Rebase, squash, push, 创建 PR

9. /pr-review
   → 检测变更类型：DATASET + REWARD
   → 生成审查任务：Dataset Loader Correctness + Reward Function Correctness
   → 并行执行 Sonnet 审查
   → 置信度评分，过滤误报
   → 输出审查报告
```

---

## 附录：快速参考

### 调用方式

| 类型 | 调用方式 | 示例 |
|------|---------|------|
| Agent | 自动或被 AI 判断调用 | （无需用户操作） |
| Command | `/command-name [args]` | `/create-pr --draft` |
| Skill | `/skill-name [args]` | `/add-reward math` |

### 所有可用的用户命令

| 命令 | 用途 |
|------|------|
| `/gen-commit-msg` | 生成 commit message |
| `/create-pr` | 创建/更新 Pull Request |
| `/pr-review` | PR 代码审查 |
| `/add-workflow` | 添加 Workflow |
| `/add-reward` | 添加奖励函数 |
| `/add-dataset` | 添加数据集 |
| `/add-unit-tests` | 添加单元测试 |
| `/add-archon-model` | 添加 Archon 模型架构 |
| `/debug-distributed` | 调试分布式训练 |

### 源文件索引

| 文件 | 类型 | 用途 |
|------|------|------|
| `.claude/agents/planner.md` | Agent | 实现规划 |
| `.claude/agents/code-verifier.md` | Agent | 代码验证 |
| `.claude/agents/simple-code-reviewer.md` | Agent | 代码审查 |
| `.claude/agents/algorithm-expert.md` | Agent | RL 算法专家 |
| `.claude/agents/fsdp-engine-expert.md` | Agent | FSDP 引擎专家 |
| `.claude/agents/archon-engine-expert.md` | Agent | Archon 引擎专家 |
| `.claude/agents/megatron-engine-expert.md` | Agent | Megatron 引擎专家 |
| `.claude/agents/launcher-scheduler-expert.md` | Agent | 部署专家 |
| `.claude/commands/pr-review.md` | Command | PR 审查 |
| `.claude/commands/create-pr.md` | Command | 创建 PR |
| `.claude/commands/gen-commit-msg.md` | Command | 生成 commit message |
| `.claude/skills/add-workflow/SKILL.md` | Skill | 添加 Workflow |
| `.claude/skills/add-reward/SKILL.md` | Skill | 添加奖励函数 |
| `.claude/skills/add-dataset/SKILL.md` | Skill | 添加数据集 |
| `.claude/skills/add-unit-tests/SKILL.md` | Skill | 添加测试 |
| `.claude/skills/add-archon-model/SKILL.md` | Skill | 添加 Archon 模型 |
| `.claude/skills/debug-distributed/SKILL.md` | Skill | 调试分布式 |
| `.claude/rules/code-style.md` | Rule | 代码风格 |
| `.claude/rules/api-config.md` | Rule | API 配置 |
| `.claude/rules/distributed.md` | Rule | 分布式模式 |
| `.claude/rules/testing.md` | Rule | 测试标准 |
| `.claude/data/pr-review-change-types.md` | Data | 变更类型定义 |
| `.claude/data/pr-review-templates.md` | Data | 审查模板库 |
| `.claude/hooks/check-expert-update.sh` | Hook | 专家更新提醒 |
| `.claude/settings.json` | Config | Hook 注册 |
