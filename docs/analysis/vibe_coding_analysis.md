# AReaL Vibe Coding Infrastructure: Deep Analysis & Migration Guide

> An in-depth analysis of the AI-assisted development ("Vibe Coding") infrastructure
> used to build AReaL — 21,000 messages, 720,000 lines of code, zero manually typed
> characters. This document maps every key practice from the published article to its
> actual implementation in the codebase, and provides a migration playbook for adapting
> these patterns to any project.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component-by-Component Analysis](#2-component-by-component-analysis)
3. [Key Design Patterns](#3-key-design-patterns)
4. [Migration Playbook](#4-migration-playbook)
5. [Article Claims Mapped to Implementation](#5-article-claims-mapped-to-implementation)

---

## 1. Architecture Overview

### 1.1 The Evolution: From Monolith to Layered System

The article describes a critical lesson: stuffing all project knowledge into a single
`CLAUDE.md` file degrades AI performance. AReaL's solution is a **layered information
architecture** where `CLAUDE.md` serves as a lightweight router, and domain knowledge is
distributed across specialized files loaded on demand.

**Actual implementation** — the `.claude/` directory contains 24 configuration files
organized into 6 functional layers:

```
.claude/
├── settings.json                      # Layer 0: Hook automation
├── rules/          (4 files)          # Layer 1: Global constraints (auto-loaded by path)
├── agents/         (8 files)          # Layer 2: Domain experts (activated by context)
├── skills/         (6 directories)    # Layer 3: Guided workflows (invoked by user/AI)
├── commands/       (3 files)          # Layer 4: User-triggered actions
├── data/           (2 files)          # Layer 5: Reference data for commands
└── hooks/          (1 file)           # Supporting: Shell scripts for hooks
```

### 1.2 CLAUDE.md as Router

`CLAUDE.md` (158 lines) contains only:

- **Project identity**: What AReaL is, its tech stack, directory structure
- **Core commands**: How to set up the environment, run tests, run pre-commit
- **Boundary rules**: "Always Do / Ask First / Never Do" constraints
- **Progressive disclosure table**: Points AI to deeper documentation by task type

The progressive disclosure table is the key routing mechanism:

| Task             | Reference                                           |
| ---------------- | --------------------------------------------------- |
| Add Workflow     | `docs/customization/agent.md`, `areal/workflow/...` |
| Add Dataset      | `docs/customization/`, `areal/dataset/gsm8k.py`     |
| Add Reward       | `areal/api/reward_api.py`, `areal/reward/...`       |
| Algorithm Details | `docs/algorithms/*.md`                              |

**Design principle**: The AI's system prompt stays small. Deep knowledge is pulled in
only when relevant, preventing context dilution.

### 1.3 Information Flow

```
User request
    │
    ▼
CLAUDE.md (always loaded, ~158 lines)
    │
    ├──▶ rules/ (auto-loaded when file path matches)
    │       e.g., editing areal/engine/** triggers distributed.md
    │
    ├──▶ agents/ (activated by context or proactively)
    │       e.g., FSDP code change activates fsdp-engine-expert
    │
    ├──▶ skills/ (invoked via /slash-commands)
    │       e.g., /add-reward triggers step-by-step guide
    │
    └──▶ commands/ (invoked via /slash-commands)
            e.g., /pr-review triggers dynamic review pipeline
```

---

## 2. Component-by-Component Analysis

### 2.1 Rules Layer — Path-Scoped Global Constraints

**Location**: `.claude/rules/`

Rules are automatically loaded when the AI edits files matching their `paths:` frontmatter.
This is Claude Code's built-in mechanism — no custom code needed.

| Rule File        | Scoped To                                     | Content Summary                                                         |
| ---------------- | --------------------------------------------- | ----------------------------------------------------------------------- |
| `api-config.md`  | `areal/api/**`                                | Dataclass conventions, field ordering, validation patterns, CLI rules    |
| `code-style.md`  | Global (no path restriction)                  | Logging conventions, performance patterns, naming, tensor conventions    |
| `distributed.md` | `areal/engine/**`, `areal/experimental/**`, `areal/utils/fsdp/**` | Process group rules, DeviceMesh/DTensor patterns, common pitfall table |
| `testing.md`     | `**/tests/**`, `*_test.py`, `test_*.py`       | Pytest markers, test structure, mocking, GPU skip patterns, assertions   |

**What's domain-specific**: The specific rules content (FSDP sharding, DTensor mesh names,
reward function naming).

**What's universally reusable**: The **pattern** of path-scoped rules. Any project can use
this to enforce:
- API design conventions when editing API code
- Testing standards when editing test files
- Infrastructure constraints when editing infra code

**Migration template** — create a rule file:

```markdown
---
paths:
  - src/api/**
---

# API Design Rules

## Naming Conventions
- Controllers: `XxxController`
- Services: `XxxService`

## Validation
- Use Pydantic validators, not manual checks
- Raise `ValueError` with descriptive messages
```

#### Key Design Detail: The Common Pitfall Table

The `distributed.md` rule includes a diagnostic table that encodes debugging experience:

| Issue         | Cause                       | Fix                            |
| ------------- | --------------------------- | ------------------------------ |
| Hang          | Mismatched collective calls | Ensure all ranks call same op  |
| Wrong results | Incorrect reduction op      | Check `ReduceOp` (SUM vs MEAN)`|
| OOM           | Unsharded tensor on wrong device | Verify DTensor placements |

This is a critical pattern: **encoding past debugging experience as structured lookup
tables** that the AI can reference instantly. For migration, create similar tables for your
project's recurring failure modes.

---

### 2.2 Agents Layer — Domain Expert System

**Location**: `.claude/agents/`

Agents are specialized AI personas with constrained tools and specific model selections.
AReaL uses 8 agents divided into 3 categories:

#### Category 1: Workflow Agents (Proactive, Auto-Activated)

| Agent              | Model  | Tools                | Activation                        |
| ------------------ | ------ | -------------------- | --------------------------------- |
| `planner`          | Opus   | Read, Grep, Glob, Task | Before multi-file changes       |
| `code-verifier`    | Haiku  | Read, Grep, Glob, Bash | After code changes, before commit |
| `simple-code-reviewer` | Sonnet | Read, Grep, Glob   | After code changes               |

**Design insight**: These three form a **pipeline** — plan before coding, verify after
coding, review before committing. The model selection follows a cost-capability curve:
Opus for deep reasoning (planning), Haiku for fast execution (running linters/tests),
Sonnet for balanced analysis (code review).

#### Category 2: Domain Expert Agents (Context-Activated)

| Agent                     | Model  | Domain                    |
| ------------------------- | ------ | ------------------------- |
| `algorithm-expert`        | Opus   | GRPO, PPO, DAPO, rewards  |
| `fsdp-engine-expert`      | Opus   | FSDP2 sharding, parallelism |
| `archon-engine-expert`    | Opus   | Archon/MoE, EP/ETP         |
| `megatron-engine-expert`  | Opus   | Pipeline parallelism       |
| `launcher-scheduler-expert` | Sonnet | Slurm/Ray/K8s deployment |

**Design insight**: Each expert agent's prompt is structured as a **knowledge graph**,
not just a description. It includes:
- Core concepts and their relationships
- Configuration constraints and valid combinations
- Common failure modes and diagnostic procedures
- Critical file paths and their roles
- Integration patterns with other components

For example, `fsdp-engine-expert.md` doesn't just say "knows about FSDP" — it encodes
specific knowledge like which `ParallelStrategy` fields affect FSDP, how weight
synchronization works (XCCL vs disk-based), and which algorithm subclasses exist.

#### Category 3: Model Selection Strategy

The model selection across agents reveals a deliberate cost-optimization strategy:

| Reasoning Depth Required | Model  | Use Case                    |
| ------------------------ | ------ | --------------------------- |
| Deep architectural       | Opus   | Planning, domain expertise  |
| Balanced analysis        | Sonnet | Code review, launcher issues |
| Fast execution           | Haiku  | Linting, formatting, basic checks |

**What's domain-specific**: The specific expert knowledge content.

**What's universally reusable**: The agent architecture pattern — proactive workflow agents
+ domain experts with appropriate model tiers.

**Migration template** — create a domain expert agent:

```markdown
---
name: database-expert
description: Database design and query optimization expert.
  Use when modifying schema, migrations, or query-heavy code.
tools:
  - Read
  - Grep
  - Glob
  - Task
model: opus
---

# Database Expert

## Core Knowledge
- ORM: SQLAlchemy with async session management
- Migration: Alembic with autogenerate
- Query patterns: Prefer joined eager loading over N+1

## Common Pitfalls
| Issue | Cause | Fix |
|-------|-------|-----|
| N+1 queries | Missing eager load | Add `selectinload()` |
| Deadlock | Conflicting row locks | Use `FOR UPDATE SKIP LOCKED` |

## Key Files
- `src/db/models.py` — All ORM models
- `alembic/versions/` — Migrations
```

---

### 2.3 Commands Layer — User-Triggered Automation

**Location**: `.claude/commands/`

Commands are invoked via `/slash-command` syntax and perform multi-step automated workflows.

#### `/create-pr` — Automated PR Creation (693 lines)

The most sophisticated command. Its workflow:

1. **Verify prerequisites** — Not on main, no uncommitted changes, `gh` CLI available
2. **Check existing PR** — Ask permission before force-updating
3. **Fetch and rebase** — Rebase onto `origin/main`
4. **Squash all commits** — Into a single commit with generated message
5. **Generate PR description** — Following the project's GitHub PR template
6. **Force push and create PR** — With comprehensive error handling

**Key design decisions**:
- Always squashes to a single commit (opinionated, suits AI-generated code well)
- Generates commit messages using Conventional Commits format
- Includes safety checks at every critical step
- Handles failure recovery for rebase conflicts, push failures, etc.

#### `/gen-commit-msg` — Intelligent Commit Message Generation (166 lines)

1. Analyzes staged changes
2. Categorizes change type (feat/fix/docs/refactor/test/chore/perf)
3. Infers scope from file paths
4. Generates message in Conventional Commits format
5. Presents for user confirmation before committing

#### `/pr-review` — Dynamic Code Review (279 lines + 575 lines of reference data)

This is the crown jewel of the toolchain. Detailed analysis in
[Section 3.1](#31-dynamic-pr-review-system).

**What's domain-specific**: The specific PR template, commit message scoping rules, and
review change type tables.

**What's universally reusable**: The command patterns — especially `/pr-review`'s
dynamic agent allocation architecture and `/create-pr`'s safety-first workflow design.

---

### 2.4 Skills Layer — Guided Development Workflows

**Location**: `.claude/skills/`

Skills are step-by-step guides for common development tasks. Each skill is a `SKILL.md`
file in its own directory.

| Skill                | Lines | Purpose                                |
| -------------------- | ----- | -------------------------------------- |
| `add-reward`         | 211   | Create a new reward function           |
| `add-dataset`        | ~100  | Create a new dataset loader            |
| `add-workflow`       | ~150  | Create a new RolloutWorkflow           |
| `add-archon-model`   | ~300  | Add HuggingFace model to ArchonEngine  |
| `add-unit-tests`     | ~120  | Add unit or distributed tests          |
| `debug-distributed`  | 250   | Debug hangs, OOM, wrong results        |

**Design insight**: Each skill follows a consistent structure:
1. **When to Use** — Trigger conditions
2. **Step-by-Step Guide** — Numbered steps with code templates
3. **Reference Implementations** — Table of existing examples to follow
4. **Key Requirements** — Hard constraints
5. **Common Mistakes** — Anti-patterns to avoid
6. **Maintainer Guide** — (HTML comment) How to update the skill itself

The `add-reward` skill exemplifies the "pattern replication" approach the article
describes: it provides a complete code template with the exact function signature,
registration steps, async wrapper usage, and test template. The AI fills in
domain-specific logic while the structural scaffolding ensures consistency.

**What's domain-specific**: The code templates, file paths, and registration steps.

**What's universally reusable**: The skill structure pattern — step-by-step guide +
code template + reference implementations + common mistakes.

**Migration template** — create a skill for adding API endpoints:

```markdown
---
name: add-api-endpoint
description: Guide for adding a new API endpoint. Use when user
  wants to create a REST endpoint.
---

# Add API Endpoint

## When to Use
- User asks to create a new endpoint
- User mentions REST API, route, controller

## Step-by-Step Guide

### Step 1: Create Route Handler
Create `src/routes/<resource>.py`:
[code template]

### Step 2: Register in Router
Update `src/routes/__init__.py`:
[registration code]

### Step 3: Add Tests
Create `tests/test_<resource>.py`:
[test template]

## Reference Implementations
| Endpoint | File | Description |
|----------|------|-------------|
| /users | `src/routes/users.py` | CRUD with auth |
| /items | `src/routes/items.py` | CRUD with pagination |

## Common Mistakes
- Missing input validation
- Not using dependency injection for DB sessions
- Forgetting to add OpenAPI docs
```

---

### 2.5 Hooks Layer — Automated Reminders

**Location**: `.claude/hooks/`, `.claude/settings.json`

AReaL uses a single PostToolUse hook that fires whenever the AI writes or edits a file.

**`settings.json`** configuration:

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

**`check-expert-update.sh`** maps code paths to expert agent files:

| Code Path Pattern                        | Expert Agent File          |
| ---------------------------------------- | -------------------------- |
| `areal/experimental/models/archon/`      | `archon-engine-expert.md`  |
| `areal/experimental/engine/archon*`      | `archon-engine-expert.md`  |
| `areal/engine/fsdp_engine*`              | `fsdp-engine-expert.md`    |
| `areal/utils/fsdp/`                      | `fsdp-engine-expert.md`    |
| `areal/engine/megatron*`                 | `megatron-engine-expert.md` |
| `areal/trainer/ppo/`, `areal/workflow/`  | `algorithm-expert.md`      |
| `areal/reward/`                          | `algorithm-expert.md`      |

When the AI modifies FSDP engine code, the hook prints a reminder:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expert Update Reminder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Modified: areal/engine/fsdp_engine.py
Consider updating: .claude/agents/fsdp-engine-expert.md (FSDP)
━━━━━━━━━━━━━━━━━━━━━━━��━━━━━━━━━━━━
```

**Design insight**: This solves a subtle problem — when code changes, the expert agent's
knowledge can become stale. The hook ensures the human (or AI) is reminded to update the
agent's knowledge base. This is a form of **documentation-code co-evolution**.

**What's universally reusable**: The hook pattern itself. Any project can map code paths
to documentation files that need updating.

---

### 2.6 Data Layer — Reference Knowledge for Commands

**Location**: `.claude/data/`

Two files provide structured reference data for `/pr-review`:

**`pr-review-change-types.md`** (150 lines) — Detection tables that map file paths and
code patterns to risk levels:

- **CRITICAL** (8 types): Archon core, FSDP core, Megatron core, DCP checkpoints
- **HIGH** (8 types): Distributed communication, DTensor, MoE, async/concurrent
- **MEDIUM** (12 types): Tensor ops, workflow/engine, API config, compile, reward, dataset
- **LOW** (3 types): Tests, docs, config-only changes

Plus framework-specific risk identification tables and cross-component risk linkage rules.

**`pr-review-templates.md`** (425 lines) — Review task checklists organized by change
type:

- Framework-specific templates (Archon EP/ETP, FSDP Core, Megatron Pipeline, DCP, Trainer)
- General templates (Logic, Concurrency, Tensor, Numerical, TP, Communication, API, etc.)

**Design insight**: Separating reference data from command logic (`@import` mechanism)
keeps the command file focused on workflow while data files can be updated independently.
This is particularly powerful because the data files encode **accumulated institutional
knowledge** — every framework-specific risk and review checklist represents a past
debugging experience codified into reusable form.

---

## 3. Key Design Patterns

### 3.1 Dynamic PR Review System

The `/pr-review` command implements what the article calls "dynamic agent template" —
assembling a review team on-the-fly based on PR content. This is the most architecturally
interesting component.

**4-Phase Pipeline**:

```
Phase 1: Analysis (Haiku + Sonnet)
  ├─ PR status check
  ├─ Change summary
  └─ Change type detection against risk tables
      ↓
Phase 2: Planning (Sonnet)
  └─ Generate review tasks from detected types
      ↓
Phase 3: Execution (Parallel, Dynamic Models)
  └─ Run all review tasks in parallel
     Each task uses model appropriate to its risk level
      ↓
Phase 4: Scoring (Haiku)
  └─ Confidence scoring (0-100), false positive filtering
     Final structured report
```

**Model selection by risk level**:

| Risk Level       | Default Mode | Quick Mode | Economy Mode |
| ---------------- | ------------ | ---------- | ------------ |
| CRITICAL / HIGH  | Opus         | Sonnet     | Sonnet       |
| MEDIUM           | Sonnet       | Sonnet     | Haiku        |
| LOW              | Haiku        | Sonnet     | Haiku        |

**Review depth scales with model capability**:
- **Opus**: Complete context, cross-file traces, verify parallel strategy interactions
- **Sonnet**: Changed code + direct callers/callees, type signature consistency
- **Haiku**: Format and basic correctness only

**Why this matters for migration**: This pattern can be adapted for any project by:
1. Defining your own change type taxonomy and risk levels
2. Creating detection rules (file path patterns + code patterns)
3. Writing review task templates for each change type
4. Mapping risk levels to model tiers

### 3.2 Proactive vs. On-Demand Activation

AReaL agents use two activation modes encoded in the `description` frontmatter:

- **Proactive** (`"Use PROACTIVELY..."`): Auto-activated by context
  - `planner`: Before multi-file changes
  - `code-verifier`: After code changes
  - `simple-code-reviewer`: After code changes

- **On-Demand** (`"Use when..."`): Activated only when needed
  - Domain experts: When their specific domain is relevant
  - `launcher-scheduler-expert`: When requested

**Design principle**: Proactive agents form the **default workflow** (plan → implement →
verify → review). Domain experts are **exception handlers** that activate when specialized
knowledge is needed.

### 3.3 PostToolUse Hooks for Documentation Co-Evolution

The hook system (`settings.json` + `check-expert-update.sh`) ensures that when code
changes, the corresponding expert agent documentation is flagged for update. This creates
a feedback loop:

```
Code changes → Hook fires → Reminder to update agent → Agent stays current
→ Future code changes benefit from up-to-date agent knowledge
```

**Reusable pattern**: Map any `{code path → documentation file}` relationship. Examples
for other projects:
- `src/api/` changes → remind to update API documentation
- `schema/` changes → remind to update data dictionary
- `config/` changes → remind to update deployment guide

### 3.4 Evidence-Driven Debugging

The `debug-distributed` skill codifies the article's core debugging methodology:

1. **Minimal reproduction first** — Always reduce before debugging
2. **External evidence** — Use `py-spy`, `NCCL_DEBUG`, `TORCH_DISTRIBUTED_DEBUG` to
   convert runtime state into text the AI can consume
3. **Structured diagnostic workflow** — Symptom → Environment variables → Debug steps →
   Common causes → Fix patterns

The skill provides ready-to-copy code snippets for each debugging scenario:
- Hang debugging: process group verification, shape checking, timeout adjustment
- Wrong results: DTensor placement inspection, gradient reduction verification
- OOM: Memory usage reporting, FSDP coverage checking
- Communication errors: Error-cause-solution lookup table

### 3.5 Nested Planning with the Planner Agent

The `planner` agent enforces a structured planning process:

1. **Phase 1: Understanding** — Clarify requirements with specific questions (not
   open-ended), identify scope, find existing patterns
2. **Phase 2: Research** — Search for similar implementations, find callers/dependencies,
   check tests, check configuration
3. **Phase 3: Plan Output** — Quick path (2-3 files) or full plan (complex tasks)

**Key constraint**: The planner is read-only (Tools: Read, Grep, Glob, Task). It
researches and designs but never modifies code. This separation of concerns prevents
the common failure mode where AI starts implementing before understanding the full scope.

### 3.6 Maintainer Guide Pattern

Every agent, command, and skill includes an HTML-commented `MAINTAINER GUIDE` at the
bottom:

```html
<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/agents/planner.md
Activation: Automatic (PROACTIVE) when complex tasks detected

## How to Update

### Updating Plan Output Format
1. Add to the markdown template in "Phase 3: Plan Output"
2. Document when the section is required

================================================================================
-->
```

This is invisible to the AI (it's an HTML comment) but provides clear instructions for
human maintainers on how to modify the configuration file. It answers: what is this file,
where is it, and how do I change specific aspects of its behavior.

---

## 4. Migration Playbook

### 4.1 Minimal `.claude/` Skeleton

For any new project, start with this minimal structure:

```
.claude/
├── settings.json          # (Optional) Hook automation
├── rules/
│   └── code-style.md      # Your project's coding conventions
├── agents/
│   ├── planner.md         # Copy from AReaL, adjust domain references
│   ├── code-verifier.md   # Copy from AReaL, adjust test commands
│   └── code-reviewer.md   # Copy from AReaL, adjust review focus
├── commands/
│   ├── create-pr.md       # Copy from AReaL, adjust PR template
│   └── gen-commit-msg.md  # Copy from AReaL, adjust scope inference
└── skills/                # Add as needed
```

### 4.2 Step-by-Step Migration

#### Step 1: Create CLAUDE.md (The Router)

```markdown
# CLAUDE.md - [Your Project]

## Project Overview
[1-2 sentences about what the project does]

**Tech Stack**: [list]

**Core Directories**:
- `src/` - [description]
- `tests/` - [description]
- `docs/` - [description]

## Core Commands
[How to install, test, lint, build]

## Boundaries

### Always Do
- Read relevant files before modifying code
- Run [your linter] before committing
- Follow existing code patterns

### Ask First
- Modifying [critical config files]
- Adding new dependencies
- Changing [sensitive subsystems]

### Never Do
- Hardcode secrets
- Skip [your quality checks]
- Use wildcard imports

## Progressive Disclosure
| Task | Reference |
|------|-----------|
| Add Feature X | `docs/guides/feature-x.md` |
| Add Feature Y | `src/y/example.py` |
```

**Key principle**: Keep CLAUDE.md under 200 lines. Route to deeper docs.

#### Step 2: Create Code Style Rules

Create `.claude/rules/code-style.md` with your project's conventions:
- Naming patterns
- Logging approach
- Performance constraints
- Import style

If your project has distinct subsystems, create path-scoped rules for each.

#### Step 3: Set Up the Workflow Agent Pipeline

Copy and adapt these three agents from AReaL:

1. **`planner.md`** — Change the domain references but keep the 3-phase structure
   (Understanding → Research → Plan Output). This is highly reusable.

2. **`code-verifier.md`** — Change the test commands, linting tools, and file
   categorization to match your project's toolchain. Keep the 5-phase workflow
   (Identify → Format → Test → Docs → Report).

3. **`simple-code-reviewer.md`** — Replace AReaL-specific patterns with your project's
   patterns. Keep the structured output format (Critical Issues → Suggestions →
   Looks Good).

#### Step 4: Add Domain Expert Agents (As Needed)

For each major subsystem in your project that has:
- Specialized knowledge that general coding AI lacks
- Common pitfalls that repeat across sessions
- Configuration constraints that are easy to violate

Create an expert agent with:
- Core concepts and relationships
- Configuration constraints and valid combinations
- Common failure modes with diagnostic steps
- Critical file paths

#### Step 5: Set Up Commands

1. **`/create-pr`** — Highly reusable. Adjust the PR template and commit message
   format for your project.

2. **`/gen-commit-msg`** — Adjust scope inference rules (the mapping from file paths
   to commit scopes).

3. **`/pr-review`** — The most complex to adapt. You'll need to:
   - Define your own change type taxonomy (what are the risk levels in your project?)
   - Create detection rules (which file paths and code patterns indicate each type?)
   - Write review task templates (what should reviewers check for each change type?)

#### Step 6: Add Skills for Repetitive Tasks

Identify tasks your team does repeatedly and create skills for them. Good candidates:
- Adding a new API endpoint
- Creating a new database migration
- Setting up a new microservice
- Adding a new test suite

#### Step 7: (Optional) Set Up Hooks

If your project has documentation that must stay in sync with code, create a
PostToolUse hook similar to AReaL's `check-expert-update.sh`.

### 4.3 Adaptation Checklist by Project Type

#### Web Application (Backend API)

| AReaL Component | Adapt To |
|-----------------|----------|
| `distributed.md` rule | `database.md` — ORM patterns, migration rules, query optimization |
| `api-config.md` rule | `api-design.md` — REST conventions, auth patterns, error handling |
| `fsdp-engine-expert` agent | `database-expert` — Schema design, index strategy, N+1 detection |
| `algorithm-expert` agent | `business-logic-expert` — Domain rules, validation, state machines |
| `add-reward` skill | `add-endpoint` — Route + handler + validation + tests |
| `add-dataset` skill | `add-migration` — Schema change + migration + seed data |
| PR review change types | Map: auth changes=CRITICAL, schema changes=HIGH, UI=MEDIUM, docs=LOW |

#### Frontend Application

| AReaL Component | Adapt To |
|-----------------|----------|
| `code-style.md` rule | `component-patterns.md` — Component structure, state management, styling |
| `testing.md` rule | `testing.md` — Unit tests, integration tests, E2E, visual regression |
| `add-workflow` skill | `add-page` — Route + page component + data fetching + tests |
| `add-reward` skill | `add-feature` — Component + hook + store slice + tests |
| PR review change types | Map: auth/routing=CRITICAL, state management=HIGH, components=MEDIUM |

#### Data Pipeline / ML Project

| AReaL Component | Adapt To |
|-----------------|----------|
| `distributed.md` rule | `pipeline.md` — Data flow patterns, idempotency rules, retry logic |
| `testing.md` rule | `data-testing.md` — Data quality checks, schema validation, fixture patterns |
| `fsdp-engine-expert` agent | `pipeline-expert` — Orchestration (Airflow/Dagster), resource management |
| `algorithm-expert` agent | `model-expert` — Training patterns, evaluation metrics, experiment tracking |
| `debug-distributed` skill | `debug-pipeline` — Data quality issues, scheduling failures, resource errors |

#### Infrastructure / DevOps

| AReaL Component | Adapt To |
|-----------------|----------|
| `distributed.md` rule | `infrastructure.md` — IaC patterns, networking rules, security constraints |
| `launcher-scheduler-expert` agent | `cloud-expert` — AWS/GCP/Azure patterns, cost optimization |
| `debug-distributed` skill | `debug-infra` — Network issues, permission errors, resource limits |
| PR review change types | Map: IAM/networking=CRITICAL, compute config=HIGH, monitoring=MEDIUM |

---

## 5. Article Claims Mapped to Implementation

### Claim 1: "Information layering — from monolith to structured architecture"

**Article says**: "CLAUDE.md slimmed down to a concise entry file... real detail knowledge
was split into four layers."

**Implementation evidence**:
- `CLAUDE.md`: 158 lines, contains project overview + routing table
- `.claude/rules/`: 4 files, path-scoped, auto-loaded
- `.claude/agents/`: 8 files, context-activated domain experts
- `.claude/skills/`: 6 directories, user-invoked guided workflows
- `.claude/commands/`: 3 files, automated actions

The article mentions "four layers" (rules, agents, skills, commands). The actual
implementation has 6 functional layers (adding hooks and data).

---

### Claim 2: "AI is a planning amplifier, not a coding replacement"

**Article says**: "The #1 use by task type is 'feature planning' (74 times); #2 is 'code
review' (43 times). Directly having AI write code from scratch: only 9 times."

**Implementation evidence**: The `planner` agent is configured as **proactive** (auto-
activates before multi-file changes), using the most powerful model (Opus), and is
explicitly read-only (Tools: Read, Grep, Glob, Task — no Write, no Edit, no Bash).

The planner's 3-phase process (Understanding → Research → Plan Output) with structured
question guidelines ("Good vs Bad Questions" table) directly implements the article's
claim that AI should explore possibilities before writing code.

---

### Claim 3: "Multi-file coordinated changes are where AI excels most"

**Article says**: "These tasks were marked as 'most valuable' the most — 37 times."

**Implementation evidence**: The `planner` agent specifically activates for "multi-file
changes (3+ files affected)". The `/create-pr` command squashes all changes into a single
commit, treating multi-file changes as a single atomic unit. The `/pr-review` command's
risk linkage rules (in `pr-review-change-types.md`) explicitly handle cross-component
interactions:

| Detected Change | Auto-Linked Review |
|----------------|-------------------|
| EP changes | FSDP interaction check |
| Megatron changes | Pipeline + AC check |
| COMPILE changes | Performance regression + FSDP/TP interaction check |

---

### Claim 4: "Expert agents are not just knowledge bases, they are diagnostic manuals"

**Article says**: "Like experienced colleagues — first confirm context, then systematically
narrow down the scope."

**Implementation evidence**: Each domain expert agent includes:
- **Core concepts**: Not just definitions but relationships between concepts
- **Configuration constraints**: Valid combinations and invalid states
- **Diagnostic workflows**: Ordered steps to narrow down issues
- **Common pitfall tables**: Symptom → Cause → Fix mappings

For example, `distributed.md` rule provides a structured pitfall table; `debug-distributed`
skill provides step-by-step diagnostic workflows for 4 failure categories (hang, wrong
results, OOM, communication errors), each with specific environment variables, code
snippets, and common causes.

---

### Claim 5: "Evidence-driven development — design verification before writing code"

**Article says**: "Your test quality is the ceiling of AI output quality... Tests are
not a safety net after the fact; they're the 'contract terms' between you and the AI."

**Implementation evidence**:
- `testing.md` rule: Enforces test structure (Arrange-Act-Assert), proper assertions
  (`torch.testing.assert_close` with explicit `rtol`/`atol`), and GPU skip patterns
- `code-verifier` agent: Proactively runs tests after code changes
- `add-unit-tests` skill: Provides a complete guide for adding tests
- Every skill includes a "Step N: Add Tests" as a mandatory step
- `debug-distributed` skill opens with "Minimal Reproduction" as its first principle

---

### Claim 6: "Minimal reproducible demo is an undervalued practice"

**Article says**: "First spend time distilling the problem into a minimal reproduction
script, then ask AI with that script."

**Implementation evidence**: `debug-distributed` skill (`.claude/skills/debug-distributed/
SKILL.md`) opens with:

> **Always follow the minimal demo principle**: Reproduce with the least amount of code
> to narrow down the issue faster.

It then provides a concrete reduction strategy:
1. Remove unrelated model components
2. Use small tensor sizes
3. Reduce world_size to minimum (e.g., 2 GPUs)
4. Remove torch.compile if possible
5. Disable activation checkpointing

---

### Claim 7: "Dynamic code review — assembling expert teams based on PR content"

**Article says**: "The system analyzes which files the PR changed, classifies by risk
level (CRITICAL/HIGH/MEDIUM/LOW), then selects specialized review tasks from a template
library."

**Implementation evidence**: Fully implemented in 3 files:
- `.claude/commands/pr-review.md` (279 lines): 4-phase workflow with model configuration
- `.claude/data/pr-review-change-types.md` (150 lines): 31 change types across 4 risk
  levels, framework-specific risk tables, cross-component risk linkage rules
- `.claude/data/pr-review-templates.md` (425 lines): 23+ review task templates

The confidence scoring system (0-100) with explicit false positive guidelines prevents
over-reporting. The structured output format ensures actionable results.

---

### Claim 8: "Configuration engineering is becoming an independent engineering practice"

**Article says**: "AReaL's .claude/ directory — CLAUDE.md entry, 4 rules files, 8 agent
configs, /pr-review review templates — are engineering artifacts that need ongoing
maintenance."

**Implementation evidence**: Every configuration file includes a `MAINTAINER GUIDE` in
HTML comments explaining:
- File location and invocation method
- How to update specific aspects
- Dependencies on other files

The `check-expert-update.sh` hook automates part of this maintenance by reminding
developers to update agent knowledge when code changes.

Total configuration: 24 files, approximately 3,500 lines of prompt engineering and
workflow definition — a non-trivial engineering investment that requires versioning,
review, and iteration just like application code.

---

### Claim 9: "Encode preferences into rules — AI always defers to your first version"

**Article says**: "AI's first version of code often doesn't match your style preferences
(e.g., it always wants to use inheritance while you prefer composition). The lesson:
encode preferences into rules."

**Implementation evidence**: `code-style.md` rule explicitly states:

> **Prefer composition over inheritance**: Avoid deep class hierarchies
> - Good: `Engine` holds a `Checkpointer` instance
> - Avoid: `CheckpointableEngine(Engine)` → `FSDPCheckpointableEngine(...)`

> Keep inheritance shallow (≤2 levels when possible)

This is a direct encoding of a human preference that the AI would otherwise violate
repeatedly.

---

### Claim 10: "Multi-session parallelism — pass@k sampling and pipeline"

**Article says**: "32 days had 402 multi-session parallel events... multitasking (different
sessions handle different tasks) and pass@k sampling (same problem, different constraints,
pick the best)."

**Implementation evidence**: While multi-session parallelism is a methodology practice
rather than a configuration artifact, the toolchain supports it through:
- `/create-pr` handles rebasing (resolves divergence from parallel work)
- Git worktree support mentioned in the skill system (`superpowers:using-git-worktrees`)
- Write-after-read consistency checks prevent file corruption from concurrent sessions
- The review pipeline (`simple-code-reviewer` → `/pr-review`) provides independent
  verification of work done in any session

---

## Summary: What to Take Away

### The 3 Most Reusable Components (Copy Directly)

1. **Planner Agent** (`planner.md`) — The 3-phase planning process is domain-independent.
   Change the domain references, keep the structure.

2. **Code Verifier Agent** (`code-verifier.md`) — The 5-phase verification workflow
   adapts to any project by changing test/lint commands.

3. **Commit Message Generator** (`gen-commit-msg.md`) — Conventional Commits format
   with scope inference works universally.

### The 3 Most Valuable Patterns (Adapt to Your Domain)

1. **Dynamic PR Review** — Risk-based change type detection → model-appropriate review
   depth → parallel execution → confidence scoring. Requires investment to define your
   own change taxonomy but pays off enormously.

2. **Path-Scoped Rules** — Different coding standards for different parts of the codebase,
   loaded only when relevant. Prevents context dilution.

3. **PostToolUse Hooks for Doc-Code Co-Evolution** — Automated reminders to update
   documentation when related code changes. Simple to implement, high long-term value.

### The Core Methodology (Apply Everywhere)

1. **CLAUDE.md as Router, Not Dump** — Keep it under 200 lines. Route to deeper docs.
2. **Encode Debugging Experience as Lookup Tables** — Symptom → Cause → Fix.
3. **Every Sub-Task Needs Input, Output, and Verification** — Tests are contracts.
4. **Proactive Agents Form the Default Pipeline** — Plan → Implement → Verify → Review.
5. **Domain Experts Carry Knowledge Graphs, Not Just Descriptions** — Include
   constraints, failure modes, diagnostic procedures, and file references.
