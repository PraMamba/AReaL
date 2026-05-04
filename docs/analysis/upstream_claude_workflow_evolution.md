# .claude/ AI 辅助开发工作规范演进分析

> **分析范围**: upstream/main 最近 250 个 commits（2026-01-25 ~ 2026-04-27） **相关 commits**: 13
> 个（占总量 5.2%），净增 23,636 行（占总净增 28.9%） **时间跨度**: 74 天（2026-01-30 ~ 2026-04-15），PR #866 ~
> #1177

______________________________________________________________________

## 目录

- [一、总览与统计](#%E4%B8%80%E6%80%BB%E8%A7%88%E4%B8%8E%E7%BB%9F%E8%AE%A1)
- [二、Genesis: Claude Code 初始配置](#%E4%BA%8Cgenesis-claude-code-%E5%88%9D%E5%A7%8B%E9%85%8D%E7%BD%AE)
- [三、Agent 智能体架构](#%E4%B8%89agent-%E6%99%BA%E8%83%BD%E4%BD%93%E6%9E%B6%E6%9E%84)
- [四、Skill 技能引导系统](#%E5%9B%9Bskill-%E6%8A%80%E8%83%BD%E5%BC%95%E5%AF%BC%E7%B3%BB%E7%BB%9F)
- [五、Command 命令系统](#%E4%BA%94command-%E5%91%BD%E4%BB%A4%E7%B3%BB%E7%BB%9F)
- [六、Rule 代码规范系统](#%E5%85%ADrule-%E4%BB%A3%E7%A0%81%E8%A7%84%E8%8C%83%E7%B3%BB%E7%BB%9F)
- [七、多平台 Harness 生态](#%E4%B8%83%E5%A4%9A%E5%B9%B3%E5%8F%B0-harness-%E7%94%9F%E6%80%81)
- [八、Claude SDK 与 Agent Service 集成](#%E5%85%ABclaude-sdk-%E4%B8%8E-agent-service-%E9%9B%86%E6%88%90)
- [九、质量评审与发现问题](#%E4%B9%9D%E8%B4%A8%E9%87%8F%E8%AF%84%E5%AE%A1%E4%B8%8E%E5%8F%91%E7%8E%B0%E9%97%AE%E9%A2%98)
- [十、总结与建议](#%E5%8D%81%E6%80%BB%E7%BB%93%E4%B8%8E%E5%BB%BA%E8%AE%AE)
- [附录 A: 完整 Commit 清单](#%E9%99%84%E5%BD%95-a-%E5%AE%8C%E6%95%B4-commit-%E6%B8%85%E5%8D%95)
- [附录 B: 当前 .claude/ 目录结构](#%E9%99%84%E5%BD%95-b-%E5%BD%93%E5%89%8D-claude-%E7%9B%AE%E5%BD%95%E7%BB%93%E6%9E%84)

______________________________________________________________________

## 一、总览与统计

### 1.1 开发阶段划分

.claude/ 工作规范体系在 74 天内经历了 4 个清晰的演进阶段:

| 阶段         | 时间范围        | Commits | 净增行数 | 核心主题                                 |
| ------------ | --------------- | ------- | -------- | ---------------------------------------- |
| Genesis      | Jan 30          | 1       | +3,611   | 从零创建完整 .claude/ 生态               |
| 架构完善     | Feb 2 - Feb 9   | 4       | +9,052   | Agent 重命名、技能扩展、基础设施整合     |
| 多平台扩展   | Feb 25 - Mar 25 | 5       | +9,175   | OpenCode 移植、Codex 适配、双语文档      |
| 服务化与深化 | Mar 27 - Apr 15 | 3       | +1,762   | Agent Service、分类模型重设计、Fork 支持 |

### 1.2 组件清单

| 类别               | 数量 | 位置                     |
| ------------------ | ---- | ------------------------ |
| Agents（智能体）   | 8    | `.claude/agents/`        |
| Skills（技能指南） | 7    | `.claude/skills/`        |
| Commands（命令）   | 7    | `.claude/commands/`      |
| Rules（规范）      | 4    | `.claude/rules/`         |
| Data（参考数据）   | 2    | `.claude/data/`          |
| Hooks（钩子）      | 1    | `.claude/hooks/`         |
| Settings（配置）   | 2    | `.claude/settings*.json` |

### 1.3 贡献者分析

| 贡献者         | Commits | 占比  | 主要贡献                         |
| -------------- | ------- | ----- | -------------------------------- |
| Wentai Zhang   | 6       | 46.2% | Agent 架构、Skills、Harness 扩展 |
| KennyMcCormick | 2       | 15.4% | Agent Service 基础设施           |
| fishcrap       | 2       | 15.4% | Claude SDK 集成、Workflow        |
| ZIYI ZENG      | 1       | 7.7%  | 双语文档                         |
| Wei Fu         | 1       | 7.7%  | 基础设施整合                     |
| 其他           | 1       | 7.7%  | Create-pr Skill                  |

**对比 Archon**: .claude 贡献者更分散（6 人 vs 4 人），bus factor 更健康（前 3 名占 77% vs Archon 单人 89%）。

______________________________________________________________________

## 二、Genesis: Claude Code 初始配置

### 2.1 关键 Commit

| 日期   | Commit     | PR   | 主题                                    | 影响      |
| ------ | ---------- | ---- | --------------------------------------- | --------- |
| Jan 30 | `73170373` | #866 | Add Claude Code configuration for AReaL | +3,613/-2 |

### 2.2 创世内容

PR #866 一次性创建了完整的 `.claude/` 目录，包含 23 个新文件:

```
.claude/
├── agents/          # 7 个智能体定义
│   ├── algorithm-expert.md        (193 行)
│   ├── archon-expert.md           (234 行)
│   ├── code-verifier.md           (195 行)
│   ├── fsdp-expert.md             (175 行)
│   ├── megatron-expert.md         (233 行)
│   ├── planner.md                 (155 行)
│   └── simple-code-reviewer.md    (119 行)
├── commands/        # 2 个命令
│   ├── gen-commit-msg.md          (165 行)
│   └── pr-review.md               (278 行)
├── data/            # 2 个参考数据
│   ├── pr-review-change-types.md  (149 行)
│   └── pr-review-templates.md     (424 行)
├── hooks/           # 1 个钩子
│   └── check-expert-update.sh     (67 行)
├── rules/           # 4 个规范
│   ├── api-config.md              (62 行)
│   ├── code-style.md              (52 行)
│   ├── distributed.md             (43 行)
│   └── testing.md                 (61 行)
├── skills/          # 4 个技能
│   ├── add-dataset/SKILL.md       (230 行)
│   ├── add-reward/SKILL.md        (211 行)
│   ├── add-workflow/SKILL.md      (192 行)
│   └── debug-distributed/SKILL.md (250 行)
└── settings.json                  (15 行)
```

**设计哲学**: 一次到位地建立完整的分类体系（Agent/Skill/Command/Rule/Data/Hook），而非逐步积累。

______________________________________________________________________

## 三、Agent 智能体架构

### 3.1 相关 Commits

| 日期   | Commit     | PR   | 主题                          | 影响        |
| ------ | ---------- | ---- | ----------------------------- | ----------- |
| Jan 30 | `73170373` | #866 | 初始 7 个 Agent 创建          | +3,613/-2   |
| Feb 2  | `3b792657` | #873 | Agent 架构重构，重命名 + 新增 | +1,351/-671 |
| Feb 25 | `3888251e` | #934 | Agent 定义 API 修复           | +4,998/-468 |

### 3.2 三层模型选择策略

Agents 按认知复杂度采用三层模型选择:

| 层级         | 模型   | Agents                                                        | 用途                   |
| ------------ | ------ | ------------------------------------------------------------- | ---------------------- |
| **深度推理** | Opus   | planner, fsdp/archon/megatron-engine-expert, algorithm-expert | 架构设计、领域专家分析 |
| **中等分析** | Sonnet | simple-code-reviewer, launcher-scheduler-expert               | 代码审查、配置分析     |
| **机械执行** | Haiku  | code-verifier                                                 | 格式化、Lint、测试运行 |

**成本优化**: 只有需要深度推理的任务使用高成本模型，机械性任务使用轻量模型。

### 3.3 Agent 重命名演进（PR #873）

| 旧名称               | 新名称                         | 变更原因           |
| -------------------- | ------------------------------ | ------------------ |
| `archon-expert.md`   | `archon-engine-expert.md`      | 明确"引擎"领域范围 |
| `fsdp-expert.md`     | `fsdp-engine-expert.md`        | 统一命名约定       |
| `megatron-expert.md` | `megatron-engine-expert.md`    | 统一命名约定       |
| _(新增)_             | `launcher-scheduler-expert.md` | 覆盖集群调度领域   |

### 3.4 Agent 详细清单

| Agent                       | 模型   | 工具                       | 激活方式 | 行数 | 职责                 |
| --------------------------- | ------ | -------------------------- | -------- | ---- | -------------------- |
| `planner`                   | opus   | Read, Grep, Glob, Task     | 主动     | 179  | 架构设计与实现规划   |
| `code-verifier`             | haiku  | Read, Grep, Glob, **Bash** | 主动     | 202  | 格式化/Lint/测试执行 |
| `simple-code-reviewer`      | sonnet | Read, Grep, Glob           | 主动     | 119  | 快速代码质量检查     |
| `archon-engine-expert`      | opus   | Read, Grep, Glob, Task     | 手动     | 296  | Archon 引擎专家      |
| `fsdp-engine-expert`        | opus   | Read, Grep, Glob, Task     | 手动     | 296  | FSDP 引擎专家        |
| `megatron-engine-expert`    | opus   | Read, Grep, Glob, Task     | 手动     | 296  | Megatron 引擎专家    |
| `algorithm-expert`          | opus   | Read, Grep, Glob, Task     | 手动     | 192  | RL 算法专家          |
| `launcher-scheduler-expert` | sonnet | Read, Grep, Glob, Task     | 手动     | 186  | 集群调度专家         |

**设计亮点**:

- 除 `code-verifier` 外全部为只读 Agent（无 Bash/Write/Edit 权限），符合最小权限原则
- 每个 Agent 内置隐藏的 `<!-- MAINTAINER GUIDE -->` 更新指南
- 主动/手动激活方式清晰区分

### 3.5 开发者工作流中的 Agent 编排

```
规划阶段 ──→ planner (opus)
   │
实现阶段 ──→ engine-expert (opus) + skill 引导
   │
格式化/Lint ──→ code-verifier (haiku)
   │
质量检查 ──→ simple-code-reviewer (sonnet)
   │
提交/PR ──→ gen-commit-msg + create-pr 命令
```

______________________________________________________________________

## 四、Skill 技能引导系统

### 4.1 相关 Commits

| 日期   | Commit     | PR    | 主题                          | 影响             |
| ------ | ---------- | ----- | ----------------------------- | ---------------- |
| Jan 30 | `73170373` | #866  | 初始 4 个 Skill 创建          | (包含在 genesis) |
| Feb 2  | `3b792657` | #873  | 新增 add-unit-tests Skill     | +1,351/-671      |
| Feb 9  | `5f333d26` | #914  | 新增 /add-archon-model Skill  | +551/-5          |
| Mar 25 | `682d5640` | #1082 | 新增 commit-conventions Skill | +8,189/-370      |

### 4.2 Skill 分类与设计

| Skill                | 用途                | 复杂度       | 设计模式                  |
| -------------------- | ------------------- | ------------ | ------------------------- |
| `add-dataset`        | 数据集加载器创建    | 模板型       | 代码模板 + 注册指南       |
| `add-workflow`       | Workflow 实现       | 模板型       | 接口模板 + 集成步骤       |
| `add-reward`         | 奖励函数创建        | 模板型       | 函数模板 + 测试指南       |
| `add-archon-model`   | Archon 模型架构添加 | **半自动化** | 10 步分析-生成流程        |
| `add-unit-tests`     | 测试开发            | 指南型       | 测试策略 + 模式参考       |
| `debug-distributed`  | 分布式调试          | 参考型       | 诊断清单 + 工具指引       |
| `commit-conventions` | Git 提交格式        | 约定型       | Conventional Commits 规范 |

### 4.3 深度解析: /add-archon-model（PR #914）

这是最复杂的 Skill（509 行），展示了 AI 辅助开发的最佳实践:

**10 步流程**:

1. 分析 HuggingFace 源码（自动读取目标模型的 `modeling_*.py`）
1. 选择参考模型（qwen2 或 qwen3）
1. 创建模型目录结构
1. 实现模型参数类（`args.py`）
1. 实现模型主体（`model.py`）
1. 实现 RoPE（`rope.py`）
1. 实现状态字典适配器（`state_dict_adapter.py`）
1. 实现并行化函数（`parallelize.py`）
1. 实现 ModelSpec（`spec.py`）
1. 编写测试（HF parity 测试 + 分布式测试）

**设计特点**:

- 每步包含 "Common Mistakes" 反模式列表
- 引用 Archon 引擎注册表模式和 Protocol 类型
- 支持 MoE 模型的扩展路径

### 4.4 Skill 间交叉引用

```
add-archon-model ──引用──→ Archon Engine Expert agent
add-unit-tests ──引用──→ add-dataset, add-workflow, add-reward skills
commit-conventions ←──被引用── create-pr command
debug-distributed ──引用──→ distributed.md rule
```

______________________________________________________________________

## 五、Command 命令系统

### 5.1 相关 Commits

| 日期   | Commit     | PR    | 主题                                   | 影响          |
| ------ | ---------- | ----- | -------------------------------------- | ------------- |
| Jan 30 | `73170373` | #866  | gen-commit-msg + pr-review             | (genesis)     |
| Feb 2  | `a0ec9cee` | #875  | 新增 create-pr 命令                    | +1,087/-152   |
| Mar 7  | `8805b9ad` | #995  | 新增 translate-doc-zh 命令             | +422/-348     |
| Mar 25 | `682d5640` | #1082 | 新增 upgrade-vllm/megatron-core/docker | +8,189/-370   |
| Mar 27 | `3142b88a` | #1092 | create-pr 增加 Fork 支持               | +105/-7       |
| Apr 1  | `8357f717` | #1124 | review-pr 分类模型重设计               | +1,948/-1,403 |

### 5.2 Command 详细清单

| 命令                     | 用途                          | 行数           | 复杂度 |
| ------------------------ | ----------------------------- | -------------- | ------ |
| `/create-pr`             | PR 创建（rebase/squash/push） | 795            | 极高   |
| `/gen-commit-msg`        | 智能 commit message 生成      | 170            | 低     |
| `/review-pr`             | 动态 PR 代码审查              | 293 + 数据文件 | 极高   |
| `/translate-doc-zh`      | 英中文档翻译                  | 106            | 低     |
| `/update-docker-image`   | Docker 镜像版本更新           | 99             | 中     |
| `/upgrade-megatron-core` | Megatron-Core 版本升级        | 810            | 极高   |
| `/upgrade-vllm`          | vLLM 版本升级                 | 1,023          | 极高   |

### 5.3 深度解析: /create-pr（PR #875, #1092）

**7 步安全工作流**:

1. 检查当前分支状态
1. 获取最新 `origin/main` 并 rebase
1. 交互式确认 rebase 结果
1. 压缩 commits（保留最终 squash message）
1. **检测推送远端**（PR #1092 新增 Fork 检测）
1. 推送到正确的远端
1. 创建 PR（使用 `commit-conventions` Skill 格式）

**Fork 工作流检测**（PR #1092 新增）:

```
检测 origin 是否可写（dry-run push）
├── 可写 → 使用 origin（maintainer 模式）
└── 不可写 → 查找 fork remote → 使用 --repo + --head 参数（contributor 模式）
```

### 5.4 深度解析: /review-pr 分类模型演进

#### 原始设计（PR #866）: Change Type 模型

- 基于文件变更类型（feat/fix/refactor）分类
- 单一维度分类

#### 重设计（PR #1124）: Domain/Signal 模型

- **12 个 L1 域**: model_compute, parallelism_infra, engine_lifecycle, config_api,
  training_loop, data_pipeline, inference_serving, checkpoint_io, infra_deploy,
  observability, ai_workflow, documentation
- **40+ L2 信号**: 每个域下细分的具体信号（如 `archon_core`, `fsdp_utils`, `pp_schedule`）
- **跨域关联规则**: 如 `archon_core` 自动触发 Model Compute 检查
- **风险等级驱动的模型选择**: 高风险信号使用 Opus，低风险使用 Sonnet/Haiku

**4 阶段审查架构**:

```
Phase 1: 变更分析与域/信号检测
Phase 2: 动态 Agent 分配（基于域/信号/风险）
Phase 3: 并行任务执行
Phase 4: 汇总报告生成
```

### 5.5 深度解析: /upgrade-vllm 和 /upgrade-megatron-core

这两个命令是 AI 辅助依赖升级的典范:

**设计模式**: 每个命令包含完整的 API 调用站点审计清单，覆盖:

- 引擎层、推理服务层、代理服务层的所有 API 调用
- 内部属性使用（`model._model_runner` 等）
- 详细的升级步骤、验证清单、回滚计划

**核心价值**: AI Agent 可以精确定位每个需要检查的调用站点，避免遗漏。

______________________________________________________________________

## 六、Rule 代码规范系统

### 6.1 创建 Commit

| 日期   | Commit     | PR   | 主题                |
| ------ | ---------- | ---- | ------------------- |
| Jan 30 | `73170373` | #866 | 初始 4 个 Rule 创建 |

### 6.2 规范清单

| 规范             | 作用域（paths）                            | 行数 | 核心内容                        |
| ---------------- | ------------------------------------------ | ---- | ------------------------------- |
| `api-config.md`  | `areal/api/**`                             | 62   | 配置数据类设计模式              |
| `code-style.md`  | (全局)                                     | 67   | 编码约定（logging、命名、性能） |
| `distributed.md` | `areal/engine/**`, `areal/experimental/**` | 43   | 分布式训练模式与约束            |
| `testing.md`     | `**/tests/**`, `test_*.py`                 | 61   | 测试策略与覆盖率要求            |

### 6.3 设计亮点

**路径作用域激活**: 通过 frontmatter 的 `paths` 字段控制规范的加载时机，最小化上下文开销。只有修改相关文件时才加载对应规范。

**代码风格规范要点**:

| 类别            | 规范                                                                           |
| --------------- | ------------------------------------------------------------------------------ |
| **Logger 命名** | 使用 `getLogger("PascalCaseName")`，禁止 `getLogger(__name__)`                 |
| **性能**        | 避免 GPU-CPU 同步（`.item()`, `.tolist()`），优先批操作，谨慎使用 in-place ops |
| **命名约定**    | Config → `XxxConfig`，Engine → `XxxEngine`，Workflow → `XxxWorkflow`           |
| **颜色方案**    | 蓝色=基础设施，白色=编排，紫色=RL，绿色=数据，青色=计算后端                    |

______________________________________________________________________

## 七、多平台 Harness 生态

### 7.1 相关 Commits

| 日期   | Commit     | PR    | 主题                               | 影响        |
| ------ | ---------- | ----- | ---------------------------------- | ----------- |
| Feb 25 | `3888251e` | #934  | 移植 Agent 基础设施到 OpenCode     | +4,998/-468 |
| Mar 25 | `682d5640` | #1082 | 新增 Codex Harness，对齐 AI 工作流 | +8,189/-370 |

### 7.2 三平台 Harness 架构

```
.agents/                    # 平台无关的规范参考
├── skills/                 # 12+ SKILL.md 文件
└── references/             # 共享参考数据

.claude/                    # Claude Code harness
├── agents/, skills/, commands/, rules/, data/, hooks/
└── settings.json

.opencode/                  # OpenCode harness
├── agent/, skills/, command/, data/
└── config

.codex/                     # Codex harness
├── 8 个 agent (.md + .toml 对)
└── config.toml
```

### 7.3 演进历程

| 阶段                 | PR    | 日期   | 内容                            |
| -------------------- | ----- | ------ | ------------------------------- |
| Claude Code 独占     | #866  | Jan 30 | 仅有 `.claude/` 目录            |
| OpenCode 移植        | #934  | Feb 25 | 创建 `.opencode/`，统一命名     |
| Codex + 平台无关参考 | #1082 | Mar 25 | `.codex/` + `.agents/` 中立参考 |

### 7.4 同步机制

PR #1124 引入了 `sync_review_pr_refs.py` 脚本，用于从 `.agents/skills/review-pr/references/`
重新生成各 harness 的审查数据文件，确保跨平台一致性。

______________________________________________________________________

## 八、Claude SDK 与 Agent Service 集成

### 8.1 相关 Commits

| 日期   | Commit     | PR    | 主题                                                | 影响        |
| ------ | ---------- | ----- | --------------------------------------------------- | ----------- |
| Feb 4  | `3ab79d97` | #885  | Anthropic Claude SDK 集成到 RL 训练                 | +893/-52    |
| Feb 25 | `01ea62b3` | #937  | 修复 Agent API 的 max_turns 参数泄漏                | +3/-0       |
| Mar 19 | `a81bbd84` | #1048 | Agent Service 微服务基础设施                        | +3,052/-0   |
| Apr 15 | `f7e690a4` | #1177 | Agent Service Controller + Guard + Claude Agent SDK | +1,721/-819 |

### 8.2 Claude SDK for RL Training（PR #885）

将 Anthropic Claude SDK 作为 OpenAI 的替代方案，用于 Agentic RL 训练:

- `areal/workflow/anthropic/claude_math_agent.py`（159 行）: Claude 数学 Agent Workflow
- `examples/experimental/proxy/config_claude.yaml`（198 行）: Claude 训练配置
- 复用 OpenAI proxy 客户端，增加缓存和工具调用支持

### 8.3 Agent Service 微服务（PR #1048, #1177）

#### 初始架构（PR #1048）

从零构建 `areal/experimental/agent_service/` 微服务基础设施:

```
agent_service/
├── gateway/       # OpenAI-compatible WebSocket bridge
├── router/        # Admin key auth + 路由
├── data_proxy/    # 数据代理
├── worker/        # 工作节点
└── protocol/      # 协议定义
```

#### 生产化增强（PR #1177）

- 新增 `AgentServiceController`: 服务编排
- 新增 `Guard`: timing-safe WebSocket 认证（`hmac.compare_digest`）
- 替换 Tau2/PydanticAI demo 为 Claude Agent SDK（`ClaudeAgent` + `ClaudeSDKClient`）
- Config 数据类 + `__post_init__` 验证
- Session 生命周期管理（`close_session`）
- ThreadPoolExecutor 健康监控

______________________________________________________________________

## 九、质量评审与发现问题

### 9.1 高优先级问题

#### 问题 1: CLAUDE.md 与实际目录不同步

| 类别     | CLAUDE.md 列出 | 实际存在 | 缺失                                                           |
| -------- | -------------- | -------- | -------------------------------------------------------------- |
| Agents   | 8              | 8        | ✅ 同步                                                        |
| Skills   | 6              | 7        | `commit-conventions`                                           |
| Commands | 4              | 7        | `update-docker-image`, `upgrade-megatron-core`, `upgrade-vllm` |
| Rules    | 4              | 4        | ✅ 同步                                                        |

**影响**: CLAUDE.md 是用户发现能力的主要入口。3 个命令和 1 个技能缺失意味着约 36% 的命令功能对用户不可见。

### 9.2 中优先级问题

| #   | 问题                               | 位置                                                 | 说明                                        |
| --- | ---------------------------------- | ---------------------------------------------------- | ------------------------------------------- |
| 2   | Logger 命名指导不一致              | `simple-code-reviewer.md` vs `code-style.md`         | reviewer 推荐点分路径，rule 要求 PascalCase |
| 3   | create-pr PR 模板不一致            | `create-pr.md` vs `.github/PULL_REQUEST_TEMPLATE.md` | 格式和 checklist 措辞不同                   |
| 4   | review-pr 维护指南引用不存在的路径 | `review-pr.md:283-287`                               | `.agents/skills/review-pr/` 不存在          |
| 5   | 审查域数据引用已删除的目录         | `review-pr-domains-and-signals.md` Domain 11         | `.agents/`, `.opencode/`, `.codex/`         |

### 9.3 低优先级问题

| #   | 问题                                 | 位置                                                     |
| --- | ------------------------------------ | -------------------------------------------------------- |
| 6   | 引擎专家 Agent 重复结构              | 三个 engine-expert 结构几乎相同                          |
| 7   | planner 引用不存在的 "Explore agent" | `planner.md:30`                                          |
| 8   | add-dataset 格式指导不一致           | 模板代码与 "Required Fields" 描述不同                    |
| 9   | code-style.md 无路径作用域           | 全局激活，可限定为 `areal/**`                            |
| 10  | distributed.md 冗余路径              | `areal/engine/fsdp_utils/**` 已被 `areal/engine/**` 覆盖 |
| 11  | Hook 缺少 launcher-scheduler 映射    | `check-expert-update.sh` 未覆盖 `areal/infra/launcher/`  |
| 12  | settings.local.json 包含临时权限     | 特定分析任务的 awk 命令权限不应提交                      |

______________________________________________________________________

## 十、总结与建议

### 10.1 整体评价

.claude/ 工作规范体系是一个**设计良好、分类清晰**的 AI 辅助开发框架:

| 维度         | 评价   | 说明                                                                   |
| ------------ | ------ | ---------------------------------------------------------------------- |
| **分类法**   | 优秀   | Agent(谁)/Skill(怎么做)/Command(执行)/Rule(必须)/Data(知识)/Hook(监控) |
| **成本优化** | 优秀   | 三层模型选择（Opus/Sonnet/Haiku）                                      |
| **安全性**   | 良好   | 只读 Agent、安全门控、最小权限                                         |
| **可维护性** | 良好   | 内置维护指南、Hook 提醒                                                |
| **跨平台**   | 良好   | Claude/OpenCode/Codex 三平台支持                                       |
| **文档同步** | 需改进 | CLAUDE.md 与实际目录存在 36% 的命令缺失                                |

### 10.2 开发生命周期覆盖

```
规划 ──→ planner agent
  │
实现 ──→ engine-expert agents + skills (add-dataset/workflow/reward/archon-model)
  │
测试 ──→ add-unit-tests skill + code-verifier agent
  │
调试 ──→ debug-distributed skill + engine-expert agents
  │
审查 ──→ simple-code-reviewer agent + /review-pr command
  │
提交 ──→ /gen-commit-msg + commit-conventions skill + /create-pr command
  │
翻译 ──→ /translate-doc-zh command
  │
升级 ──→ /upgrade-megatron-core + /upgrade-vllm + /update-docker-image
  │
质量 ──→ rules (api-config, code-style, distributed, testing)
```

**唯一空白**: 缺少专门的 **部署/发布** Skill 或 Command（`update-docker-image` 部分覆盖）。

### 10.3 优先改进建议

1. **同步 CLAUDE.md**: 添加 3 个缺失命令和 1 个缺失 Skill 到文档索引
1. **修复 Logger 指导**: 统一 `simple-code-reviewer.md` 与 `code-style.md` 的 Logger 命名约定
1. **清理陈旧引用**: 更新 `review-pr.md` 和 `review-pr-domains-and-signals.md` 中的无效路径
1. **对齐 PR 模板**: 使 `create-pr.md` 与 `.github/PULL_REQUEST_TEMPLATE.md` 保持一致
1. **扩展 Hook 覆盖**: 将 launcher-scheduler 路径映射添加到 `check-expert-update.sh`
1. **添加 CI 验证**: 自动检查 `.claude/` 文件间的交叉引用有效性

______________________________________________________________________

## 附录 A: 完整 Commit 清单

| #   | 日期   | Commit     | PR    | 类型     | 主题                                                      | +/-           |
| --- | ------ | ---------- | ----- | -------- | --------------------------------------------------------- | ------------- |
| 1   | Jan 30 | `73170373` | #866  | chore    | Add Claude Code configuration for AReaL                   | +3,613/-2     |
| 2   | Feb 2  | `3b792657` | #873  | chore    | Refactor agent architecture and complete ecosystem        | +1,351/-671   |
| 3   | Feb 2  | `a0ec9cee` | #875  | chore    | Consolidate infrastructure and add create-pr command      | +1,087/-152   |
| 4   | Feb 4  | `3ab79d97` | #885  | feat     | Anthropic Claude SDK integration for RL training          | +893/-52      |
| 5   | Feb 9  | `5f333d26` | #914  | feat     | Add /add-archon-model skill for new model support         | +551/-5       |
| 6   | Feb 25 | `3888251e` | #934  | chore    | Port agent infrastructure to OpenCode                     | +4,998/-468   |
| 7   | Feb 25 | `01ea62b3` | #937  | fix      | Filter out max_turns from kwargs before API calls         | +3/-0         |
| 8   | Mar 7  | `8805b9ad` | #995  | feat     | Add bilingual documentation with translate-doc-zh command | +422/-348     |
| 9   | Mar 19 | `a81bbd84` | #1048 | feat     | Add Agent Service microservice infrastructure             | +3,052/-0     |
| 10  | Mar 25 | `682d5640` | #1082 | chore    | Add Codex harness and align AI workflows                  | +8,189/-370   |
| 11  | Mar 27 | `3142b88a` | #1092 | feat     | Add fork workflow support to create-pr skill              | +105/-7       |
| 12  | Apr 1  | `8357f717` | #1124 | refactor | Redesign review-pr taxonomy and sync flow                 | +1,948/-1,403 |
| 13  | Apr 15 | `f7e690a4` | #1177 | feat     | Agent service Controller, Guard, and Claude Agent SDK     | +1,721/-819   |

## 附录 B: 当前 .claude/ 目录结构

```
.claude/
├── settings.json                              # PostToolUse hook 配置
├── settings.local.json                        # 本地权限覆盖
│
├── agents/                                    # 8 个智能体定义
│   ├── algorithm-expert.md                    # RL 算法专家 (opus)
│   ├── archon-engine-expert.md                # Archon 引擎专家 (opus)
│   ├── code-verifier.md                       # 格式/Lint/测试 (haiku)
│   ├── fsdp-engine-expert.md                  # FSDP 引擎专家 (opus)
│   ├── launcher-scheduler-expert.md           # 集群调度专家 (sonnet)
│   ├── megatron-engine-expert.md              # Megatron 引擎专家 (opus)
│   ├── planner.md                             # 实现规划 (opus)
│   └── simple-code-reviewer.md                # 代码审查 (sonnet)
│
├── commands/                                  # 7 个用户命令
│   ├── create-pr.md                           # PR 创建 (795 行)
│   ├── gen-commit-msg.md                      # Commit Message 生成 (170 行)
│   ├── review-pr.md                           # 动态 PR 审查 (293 行)
│   ├── translate-doc-zh.md                    # 英中翻译 (106 行)
│   ├── update-docker-image.md                 # Docker 镜像更新 (99 行)
│   ├── upgrade-megatron-core.md               # Megatron-Core 升级 (810 行)
│   └── upgrade-vllm.md                        # vLLM 升级 (1,023 行)
│
├── data/                                      # 参考数据
│   ├── review-pr-domains-and-signals.md       # 12 域 + 40+ 信号分类 (254 行)
│   └── review-pr-templates.md                 # 审查模板 (320 行)
│
├── hooks/                                     # 自动钩子
│   └── check-expert-update.sh                 # 引擎代码变更提醒 (67 行)
│
├── rules/                                     # 代码规范
│   ├── api-config.md                          # 配置数据类设计 (62 行)
│   ├── code-style.md                          # 编码约定 (67 行)
│   ├── distributed.md                         # 分布式训练模式 (43 行)
│   └── testing.md                             # 测试策略 (61 行)
│
└── skills/                                    # 7 个技能指南
    ├── add-archon-model/SKILL.md              # Archon 模型添加 (509 行)
    ├── add-dataset/SKILL.md                   # 数据集加载器 (230 行)
    ├── add-reward/SKILL.md                    # 奖励函数 (211 行)
    ├── add-unit-tests/SKILL.md                # 单元测试 (228 行)
    ├── add-workflow/SKILL.md                  # Workflow 实现 (192 行)
    ├── commit-conventions/SKILL.md            # 提交约定 (161 行)
    └── debug-distributed/SKILL.md             # 分布式调试 (250 行)
```
