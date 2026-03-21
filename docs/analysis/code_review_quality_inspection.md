# AReaL Code Review & Quality Inspection: Complete Implementation Analysis

## 1. Overview: The Multi-Defense Quality System

AReaL implements a **4-layer quality defense system** that ensures every code change
passes through progressively deeper levels of inspection before it can reach the main
branch:

```
Layer 1: Pre-commit Hooks          (automated formatting & linting)
    ↓
Layer 2: Code Verifier Agent       (proactive automated verification)
    ↓
Layer 3: Simple Code Reviewer      (quick quality pattern checks)
    ↓
Layer 4: Dynamic PR Review         (comprehensive AI code review)
```

Each layer catches a different class of defects. Layer 1 catches formatting and syntax
issues. Layer 2 runs the tools and confirms they pass. Layer 3 reads the code for logic
and pattern violations. Layer 4 dynamically generates specialized review tasks based on
what was actually changed, dispatching domain-expert agents in parallel.

The full pipeline follows a **plan-implement-verify-review** workflow:

1. **Plan** (`planner` agent, Opus model) -- designs implementation before code is
   written
2. **Implement** -- code changes are made
3. **Verify** (`code-verifier` agent, Haiku model) -- runs formatting, linting, and
   tests
4. **Review** (`simple-code-reviewer` agent, then `/pr-review` command) -- reads code
   for quality issues

This maps directly to the article's claim that "AI never tires, never rushes to meet
deadlines by lowering standards." The system is structurally incapable of skipping
steps -- each layer is either proactive (auto-activated after code changes) or required
before merge (PR review). There is no "rush to deadline" shortcut that bypasses the
pipeline.

---

## 2. Layer 1: Pre-commit Hooks -- Automated Formatting & Linting

**Source**: `.pre-commit-config.yaml`

Pre-commit hooks are the first defense layer. They run automatically on every `git
commit` attempt and **block non-conforming code** from being committed.

### 6 Tool Integrations

| # | Tool | Version | Purpose | File Types |
|---|------|---------|---------|------------|
| 1 | **clang-format** | v19.1.7 | C/C++/CUDA formatting | `.c`, `.cc`, `.cpp`, `.h`, `.hpp`, `.cu`, `.cuh` |
| 2 | **pre-commit-hooks** | v6.0.0 | Basic hygiene (5 checks) | Multiple |
| 3 | **mdformat** | 0.7.17 | Markdown formatting (wrap=88) | `.md` |
| 4 | **Ruff** | v0.14.9 | Python linting + formatting | `.py`, `.pyi`, `.ipynb` |
| 5 | **nbstripout** | 0.7.1 | Strip Jupyter notebook outputs | `.ipynb` |
| 6 | **generate-cli-docs** | local | CLI doc regeneration | `cli_args.py` only |

### pre-commit-hooks Breakdown (5 sub-checks)

- `check-yaml` -- validates YAML syntax
- `end-of-file-fixer` -- ensures files end with a newline
- `trailing-whitespace` -- removes trailing whitespace
- `check-added-large-files` -- blocks files > 1000KB (excludes `uv.lock`)
- `check-json` -- validates JSON syntax

### Ruff Configuration

**Source**: `pyproject.toml:246-274`

```
line-length = 88
target-version = "py312"

Lint rules enabled:
  E   -- pycodestyle errors
  W   -- pycodestyle warnings
  F   -- pyflakes
  I   -- isort (import sorting)
  UP  -- pyupgrade (modernize Python syntax)

Ignored:
  E501 -- line too long (handled by formatter)

Custom isort section:
  "areal" as dedicated import group after third-party
```

Ruff serves as a unified replacement for Black (formatting), isort (import sorting), and
flake8 (linting) -- a single tool covering three responsibilities.

### The CLI Docs Local Hook

**Source**: `.pre-commit-config.yaml:59-69`

A local hook triggers `python docs/generate_cli_docs.py` whenever `areal/api/cli_args.py`
or `docs/generate_cli_docs.py` is modified. This ensures CLI documentation stays
synchronized with the actual config dataclass definitions -- a doc-code co-evolution
mechanism at the pre-commit level.

### Trigger Behavior

Pre-commit hooks run automatically on every commit attempt. If any hook fails (e.g., Ruff
finds a linting error), the commit is **blocked** and the developer must fix the issue
before retrying. Many hooks (Ruff formatting, trailing whitespace, end-of-file) auto-fix
and require only re-staging.

---

## 3. Layer 2: Code Verifier Agent -- Proactive Automated Verification

**Source**: `.claude/agents/code-verifier.md`

| Property | Value |
|----------|-------|
| Model | **Haiku** (fast execution, no deep reasoning needed) |
| Tools | Read, Grep, Glob, **Bash** (can execute commands) |
| Activation | **PROACTIVE** -- auto-activates after code changes |

The code-verifier is the only agent that can **execute** commands (via Bash tool). It
runs the actual formatting, linting, and test commands rather than just reading code.

### 5-Phase Workflow

**Phase 1: Identify Changed Files**
- Runs `git status --short` and `git diff --name-only HEAD`
- Categorizes changes: Python files -> Ruff/tests, Markdown -> mdformat, Config ->
  syntax validation, API changes -> CLI docs regeneration

**Phase 2: Run Formatting & Linting**
- Executes `pre-commit run --all-files` (or `--files` for specific changes)
- Covers: Ruff lint + format, mdformat, clang-format, nbstripout

**Phase 3: Run Tests (If Applicable)**
- First checks GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Routes to appropriate test category based on changed files

GPU-aware test categorization:

| Category | Command | GPU Required |
|----------|---------|--------------|
| Unit tests | `pytest areal/tests/test_*.py` | No |
| GRPO tests | `pytest areal/tests/grpo/` | Yes |
| FSDP tests | `pytest areal/tests/test_fsdp_*.py` | Yes |
| Distributed | `pytest areal/tests/torchrun/` | Yes, multi-GPU |

**Phase 4: Documentation Checks**
- If `cli_args.py` changed: runs `uv run python docs/generate_cli_docs.py`
- If markdown changed: runs `mdformat --check docs/`

**Phase 5: Report Results**
- Outputs a structured verification report with pass/fail/skip status per check
- Concludes with a clear "Ready to Commit: YES/NO" verdict

### Auto-Fix Behavior

The code-verifier auto-fixes what it can:
- Ruff formatting -- auto-fixed, reports what changed
- Import sorting -- auto-fixed by Ruff
- Trailing whitespace -- auto-fixed
- Markdown formatting -- runs mdformat to fix

After auto-fixing, it reminds: "Files were auto-formatted. Please review changes and
re-stage: `git add -p`"

### Design Choice: Haiku Model

The code-verifier uses **Haiku** (the fastest, cheapest model) because its tasks are
straightforward: run commands, parse output, report results. No deep reasoning is needed.
This makes verification near-instantaneous and cost-effective for frequent invocations.

---

## 4. Layer 3: Simple Code Reviewer Agent -- Quick Quality Checks

**Source**: `.claude/agents/simple-code-reviewer.md`

| Property | Value |
|----------|-------|
| Model | **Sonnet** (balanced analysis capability) |
| Tools | Read, Grep, Glob (**read-only** -- cannot execute) |
| Activation | **PROACTIVE** -- auto-activates after code changes |

The simple-code-reviewer is explicitly **read-only**. It identifies issues but cannot fix
them. This separation of concerns is intentional: the code-verifier *runs things*, the
code-reviewer *reads things*.

### 3 Focus Areas

**1. AReaL-Specific Patterns**

| Pattern | What It Checks |
|---------|----------------|
| Logging | Use `areal.utils.logging.getLogger()` not `print` |
| Async | `arun_episode` must be non-blocking, use `await` |
| Tensor | Follow `[batch, seq_len, ...]` convention |
| Config | Extend dataclasses in `areal/api/cli_args.py` |
| Imports | No `*` imports; heavy deps inside functions |

**2. Common Issues to Catch**
- Missing `await` in `async def` functions
- Blocking I/O in `arun_episode`
- Tensor shape mismatches, missing batch dimensions
- Missing or incorrect type annotations
- Exception swallowing, wrong exception types
- Resource leaks (unclosed files, connections, GPU memory)

**3. Distributed Code Issues**
- Missing `all_reduce`/`all_gather` synchronization
- Device mismatch (tensors on different devices)
- Mesh dimension errors in DTensor operations
- Gradient issues (missing `detach()`, `no_grad` context)

### Structured Output Format

```
## Quick Review Summary
**Files Reviewed**: [list]
**Issues Found**: X (Y critical, Z suggestions)

### Critical Issues
1. **[Issue Title]** - `file.py:123`
   - Problem: [description]
   - Fix: [suggestion]

### Suggestions
1. **[Suggestion Title]** - `file.py:456`

### Looks Good [OK]
- [positive observations]
```

### Distinction from /pr-review

The simple-code-reviewer is **lightweight and fast** -- designed for quick checks during
development. The `/pr-review` command (Layer 4) is **comprehensive and dynamic** --
designed for thorough review before merge. The maintainer guide explicitly states: "For
comprehensive PR reviews, use `/pr-review` command instead. This agent is for quick,
lightweight checks."

---

## 5. Layer 4: Dynamic PR Review (`/pr-review`) -- Comprehensive AI Code Review

**Source**: `.claude/commands/pr-review.md`, `.claude/data/pr-review-change-types.md`,
`.claude/data/pr-review-templates.md`

This is the most sophisticated quality gate in the system. Rather than applying a fixed
checklist to every PR, it **dynamically generates targeted review tasks** based on what
was actually changed.

### 4-Phase Pipeline

```
Phase 1: Deep PR Analysis [Haiku + Sonnet]
    ├─ 1.0 PR Status Check [Haiku]
    ├─ 1.1 Get PR Summary [Haiku]
    └─ 1.2-1.4 Change Type Detection [Sonnet]
    ↓
Phase 2: Dynamic Agent Planning [Sonnet]
    ↓
Phase 3: Execute Review Tasks [Parallel, Dynamic Model Selection]
    ↓
Phase 4: Confidence Scoring & Summary [Haiku]
```

#### Phase 1: Deep Analysis

- **1.0 PR Status Check** (Haiku): Is the PR closed? Draft? Bot-generated? Gates whether
  review should proceed.
- **1.1 Get PR Summary** (Haiku): Collects title, description, modified files, change
  summary.
- **1.2 Change Type Detection** (Sonnet): Analyzes each file change against detection
  tables to classify change types by risk level.
- **1.3 Framework-Specific Risk Identification**: Maps detected types to specific risk
  checklists (Archon risks, FSDP risks, Megatron risks, DCP risks).
- **1.4 Output**: Produces a structured `CHANGE_ANALYSIS_REPORT` with detected types,
  risk level, affected files, identified risks, and related frameworks.

#### Phase 2: Dynamic Agent Planning (Sonnet)

Based on Phase 1 output, this phase:
1. Generates tasks by risk area (each high-risk area gets a dedicated task)
2. Merges related changes (interdependent changes combined)
3. Selects appropriate model per task (CRITICAL/HIGH -> Opus, MEDIUM -> Sonnet, LOW ->
   Haiku)
4. Selects review task templates from the template library
5. Outputs a numbered `GENERATED_REVIEW_TASKS` list with model, reason, checklist, and
   focus files per task

#### Phase 3: Parallel Execution (Dynamic Models)

- All review agents execute **in parallel** for maximum throughput
- Each agent reviews independently using its assigned model
- Each produces a structured `REVIEW_RESULT` with findings, severity, file/line
  references, reasons, and fix suggestions

#### Phase 4: Confidence Scoring & Summary (Haiku)

- Scores each finding 0-100 for confidence
- Filters false positives (score 0: pre-existing issues, intentionally designed code,
  linter-catchable issues, unmodified lines, explicitly disabled)
- Produces a summary report grouped by severity with statistics

### 31 Change Types Across 4 Risk Levels

**Source**: `.claude/data/pr-review-change-types.md`

| Risk Level | Count | Change Types | Default Model |
|------------|-------|-------------|---------------|
| **CRITICAL** | 8 | ARCHON_CORE, ARCHON_PARALLEL, ARCHON_MOE, ARCHON_PARALLELIZE, ARCHON_ENGINE, FSDP_CORE, MEGATRON_CORE, DCP_CHECKPOINT | Opus |
| **HIGH** | 8 | DISTRIBUTED_COMM, DTENSOR, MOE_LAYER, EP_ETP, TENSOR_PARALLEL, SEQUENCE_PARALLEL, ASYNC_CONCURRENT, TRAINER_CORE | Opus |
| **MEDIUM** | 12 | TENSOR_OPS, NUMERICAL, WORKFLOW_ENGINE, API_CONFIG, COMPILE, ACTIVATION_CKPT, CHECKPOINT_RECOVERY, REWARD, DATASET, LAUNCHER_SCHEDULER, ATTENTION | Sonnet |
| **LOW** | 3 | TESTS, DOCS, CONFIG_ONLY | Haiku |

Each change type has specific **file path patterns** and **code patterns** for detection.
For example, `ARCHON_MOE` triggers on files matching `archon/moe/` containing patterns
like `router`, `grouped_experts`, `TokenReorderer`, or `grouped_mm`.

### 23+ Review Task Templates

**Source**: `.claude/data/pr-review-templates.md`

Templates are organized into **framework-specific** and **general** categories:

**Framework-Specific Templates:**
- Archon EP/ETP Strategy Correctness Review [Opus]
- ArchonParallelDims Configuration Validation [Opus]
- MoE Layer Implementation Correctness [Opus]
- Model Parallelization Application Order [Opus]
- FSDP Core Correctness [Opus]
- FSDP Interaction with Other Parallel Strategies [Opus]
- FSDP State Management [Sonnet]
- Pipeline Parallelism Correctness [Opus]
- Megatron Model Sharding [Opus]
- Distributed Checkpoint Correctness [Opus]
- FSDP2 + DCP Integration [Opus]
- Trainer Core Logic [Opus]

**General Templates:**
- Logic and Boundary Conditions [Opus]
- Concurrency and Async [Opus]
- Tensor Shape and Data Type [Opus]
- Numerical Stability [Sonnet]
- Tensor Parallel (TP) Correctness [Opus]
- Communication and Synchronization [Sonnet]
- API Compatibility [Sonnet]
- Configuration and Parameter Validation [Sonnet]
- Workflow and Engine Interaction [Sonnet]
- Activation Checkpointing (AC) [Sonnet]
- Performance Regression Risk [Sonnet]
- Context-Aware Review [Sonnet]
- Sequence Parallel (SP/CP) Correctness [Opus]
- Checkpoint and Recovery [Sonnet]
- Reward Function Correctness [Sonnet]
- Dataset Loader Correctness [Sonnet]
- Launcher and Scheduler Configuration [Sonnet]
- torch.compile Compatibility [Sonnet]
- Documentation Format Check [Haiku]
- Test Coverage Check [Haiku]
- Logging and Metrics [Haiku]
- Import and Dependencies [Haiku]
- Security and Sensitive Information [Haiku]

Each template includes an **applicability condition** (when to use it) and a detailed
**checklist** of specific things to verify.

### Risk Linkage Rules

**Source**: `.claude/data/pr-review-change-types.md:105-121`

When certain change types are detected, the system automatically links additional review
tasks from related components:

| Detected Change | Auto-Linked Review |
|-----------------|-------------------|
| EP changes | FSDP interaction check, dp_shard_mod_ep mesh check |
| ETP changes | TP + EP combination check, mesh dimension check |
| Megatron changes | Pipeline + AC check |
| Distributed comm changes | Process group + sync check |
| SEQUENCE_PARALLEL changes | TP combination + Attention mask check, Ulysses check |
| CHECKPOINT_RECOVERY changes | FSDP state dict check, DCP compatibility check |
| DCP_CHECKPOINT changes | FSDP2 integration check, distributed consistency check |
| COMPILE changes | Performance regression + FSDP/TP interaction check |
| REWARD changes | Workflow interaction check, AsyncRewardWrapper check |
| LAUNCHER_SCHEDULER changes | Resource config + parallel strategy match check |
| TRAINER_CORE changes | Engine lifecycle + workflow integration check |
| ARCHON_ENGINE changes | DCP checkpoint + parallel dims check |

This means a single change to expert parallelism code automatically triggers reviews of
FSDP interaction, mesh configuration, and potentially checkpoint compatibility -- without
the developer needing to request these reviews manually.

### Model Selection Strategy

| Mode | CRITICAL/HIGH | MEDIUM | LOW |
|------|---------------|--------|-----|
| **Default** | Opus | Sonnet | Haiku |
| **Quick** (`--quick`) | Sonnet | Sonnet | Sonnet |
| **Economy** (`--economy`) | Sonnet | Haiku | Haiku |

Review depth scales with model capability:

| Model | Review Depth |
|-------|-------------|
| **Opus** | Complete context, cross-file traces, verify parallel strategy interactions |
| **Sonnet** | Changed code + direct callers/callees, type signature consistency |
| **Haiku** | Format and basic correctness only |

### Dynamic Generation Examples

| PR Type | Detected Types | Generated Tasks |
|---------|---------------|-----------------|
| Docs only | [DOCS] | 1 Haiku |
| Config only | [CONFIG_ONLY] | 1-2 Haiku |
| Single bug fix | [TENSOR_OPS] | 2-4 Sonnet |
| Archon core | [ARCHON_*, EP_ETP, DTENSOR] | 4-8 Opus |
| Cross-domain | [WORKFLOW_ENGINE, FSDP_CORE, TESTS] | 5-10 mixed |

A docs-only PR gets a single lightweight Haiku review. A core Archon change triggers 4-8
Opus-level deep reviews. The system self-scales.

---

## 6. Supporting Infrastructure: Rules as Quality Guardrails

**Source**: `.claude/rules/`

AReaL defines 4 **path-scoped rules** that are automatically loaded when Claude operates
on files matching specific path patterns. These rules encode institutional knowledge
about coding standards, preventing recurring style violations.

### Rule: `code-style.md` (Global)

**Scope**: All files (no path restriction)

| Category | Key Standards |
|----------|--------------|
| **Design Patterns** | Composition over inheritance; shallow hierarchies (<=2 levels); prefer delegation over mixins |
| **Logging** | Use `areal.utils.logging.getLogger("PascalCaseName")`, not `print` or stdlib `logging`; color scheme by category (blue=infrastructure, purple=RL, cyan=engines, green=data) |
| **Performance** | Avoid GPU-CPU sync (`.item()`, `.tolist()`); prefer batch operations; careful with in-place ops and autograd |
| **Naming** | `XxxConfig` for configs, `XxxEngine` for engines, `XxxWorkflow` for workflows, `xxx_reward` for rewards |
| **Tensor** | `[batch, seq_len, hidden]` shape convention; explicit dtype/device |
| **Imports** | Group: stdlib, third-party, areal; no wildcard imports |

### Rule: `api-config.md` (API Path)

**Scope**: `areal/api/**`

Enforces dataclass conventions for configuration:
- Field ordering: required -> common optional -> advanced optional -> internal (`_prefix`)
- Validation via `__post_init__` with clear `ValueError` messages
- Backward compatibility protocol: add with defaults (safe), deprecate before removal,
  add new + keep old for renames
- CLI integration: clear `help` metadata, `Literal` for enum-like choices

### Rule: `distributed.md` (Engine/FSDP Path)

**Scope**: `areal/engine/**`, `areal/experimental/**`, `areal/utils/fsdp/**`

Encodes distributed training pitfalls:
- Never create global process group at module level
- Pass `process_group` explicitly
- Mesh dimension names must match `ArchonParallelDims`
- DTensor requires consistent mesh across all ranks

Common pitfall table:

| Issue | Cause | Fix |
|-------|-------|-----|
| Hang | Mismatched collective calls | Ensure all ranks call same op |
| Wrong results | Incorrect reduction op | Check `ReduceOp` (SUM vs MEAN) |
| OOM | Unsharded tensor on wrong device | Verify DTensor placements |

### Rule: `testing.md` (Test Path)

**Scope**: `**/tests/**`, `*_test.py`, `test_*.py`

Enforces testing standards:
- Pytest markers: `@pytest.mark.slow` (>10s), `@pytest.mark.asyncio`, `@pytest.mark.skipif`
- Test naming: `test_<what>_<condition>_<expected>()`
- Structure: Arrange-Act-Assert pattern
- GPU tests: always skip gracefully with `@pytest.mark.skipif(not CUDA_AVAILABLE, ...)`
- Assertions: `torch.testing.assert_close()` with explicit `rtol`/`atol`
- Fixtures: `tmp_path` over manual temp dirs, `monkeypatch` for env vars

### How Rules Work

Rules are **auto-loaded by path context**. When Claude edits a file in `areal/api/`,
both the global `code-style.md` and the scoped `api-config.md` are active. This means
the AI always has the relevant coding standards in context -- it cannot "forget" them
because they are structurally injected based on what files are being touched.

---

## 7. Supporting Infrastructure: Domain Expert Agents

**Source**: `.claude/agents/`

AReaL maintains 5 domain expert agents, each a deep knowledge repository for its
subsystem. These serve two purposes: (1) direct consultation when working in their
domain, and (2) as specialized reviewers dispatched by `/pr-review` via risk linkage.

### Expert Agent Summary

| Agent | Model | Tools | Domain | Key Content |
|-------|-------|-------|--------|-------------|
| `algorithm-expert` | **Opus** | Read, Grep, Glob, Task | GRPO/PPO/DAPO, reward shaping, advantage normalization, loss computation | Algorithm family table (7 algorithms), core config parameters, workflow patterns, reward function signatures, loss computation formulas, debugging commands |
| `fsdp-engine-expert` | **Opus** | Read, Grep, Glob, Task | FSDP2 sharding, parameter distribution, gradient handling, checkpoint management | Configuration components, initialization flow, algorithm subclasses (PPOActor/Critic, LM, RW), parallel strategy guidelines, weight sync mechanisms (XCCL vs disk), full implementation structure map |
| `archon-engine-expert` | **Opus** | Read, Grep, Glob, Task | MoE/EP/ETP, Archon model architecture, pipeline+expert parallelism | Engine comparison, configuration approach, model architecture addition workflow, MoE training patterns, checkpoint strategies, implementation structure with file paths |
| `megatron-engine-expert` | **Opus** | Read, Grep, Glob, Task | Pipeline parallelism, micro-batch scheduling, weight sharding | Parallel strategy selection table, common configuration patterns, engine selection guide (vs FSDP), diagnostic workflow |
| `launcher-scheduler-expert` | **Sonnet** | Read, Grep, Glob, Task | Cluster deployment (Slurm/Ray/K8s), resource scheduling, process management | Launcher vs Scheduler distinction, config dataclass details, env var propagation chain, diagnostic symptom->cause->fix table, best practices checklist |

### Expert Agents as Diagnostic Manuals

Each expert agent includes **Symptom -> Cause -> Fix** tables that encode debugging
knowledge. For example, from `fsdp-engine-expert.md`:

| Symptom | Likely Cause | First Steps |
|---------|-------------|-------------|
| Initialization failure | Invalid parallel dimensions | Check `dp * sp * tp == world_size` |
| Out of memory | Insufficient GPU memory | Enable `offload_params=True`, reduce batch size |
| Weight sync failure | Network/NCCL issues | Switch to disk-based updates |

This means the AI reviewer doesn't just identify issues -- it can immediately suggest the
correct diagnostic path and fix, drawing on accumulated project-specific debugging
knowledge.

### How `/pr-review` Leverages Experts

When `/pr-review` detects change types in a PR, the risk linkage rules
(`.claude/data/pr-review-change-types.md:105-121`) trigger cross-component reviews. For
example:
- A change to `areal/experimental/engine/archon_engine.py` triggers `ARCHON_ENGINE`
  detection
- Risk linkage auto-links: "DCP checkpoint + parallel dims check"
- Phase 2 planning selects the "Model Parallelization Application Order" and
  "Distributed Checkpoint Correctness" templates
- Phase 3 dispatches Opus-level agents with these templates, effectively deploying the
  archon-engine-expert's knowledge into the review

---

## 8. Supporting Infrastructure: PostToolUse Hooks for Doc-Code Co-Evolution

**Source**: `.claude/settings.json`, `.claude/hooks/check-expert-update.sh`

### Hook Configuration

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

Every time the `Write` or `Edit` tool is used (i.e., every time code is modified), the
`check-expert-update.sh` hook runs automatically.

### Code Path -> Expert Agent Mapping

**Source**: `.claude/hooks/check-expert-update.sh:21-64`

| Code Path Pattern | Expert Agent File | Domain |
|------------------|-------------------|--------|
| `areal/experimental/models/archon/` or `areal/experimental/engine/archon*` | `archon-engine-expert.md` | Archon/MoE |
| `areal/engine/fsdp_engine*` or `areal/utils/fsdp/` | `fsdp-engine-expert.md` | FSDP |
| `areal/engine/megatron*` | `megatron-engine-expert.md` | Megatron/PP |
| `areal/trainer/ppo/` or `areal/workflow/` or `areal/reward/` | `algorithm-expert.md` | Algorithm/Workflow/Reward |

### The Feedback Loop

When code in a mapped path is modified, the hook outputs a reminder:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expert Update Reminder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Modified: areal/engine/fsdp_engine.py
Consider updating: .claude/agents/fsdp-engine-expert.md (FSDP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

This creates a **self-improving feedback loop**:

1. Code changes -> hook fires
2. Hook reminds to update expert agent documentation
3. Updated expert agent provides better reviews in future PRs
4. Better reviews catch more issues in future code changes

The expert agents are living documents that evolve with the codebase, ensuring review
quality improves over time rather than degrading as the codebase grows.

---

## 9. End-to-End Quality Flow: From Code Change to Merge

Here is a complete walkthrough of how a typical code change moves through the quality
system:

### Step 1: Planning (Before Code)

The `planner` agent (Opus model) activates **proactively** for multi-file changes.

- **Phase 1 (Understanding)**: Clarifies requirements, identifies scope
- **Phase 2 (Research)**: Searches codebase for similar implementations, finds
  callers/dependencies, checks existing tests
- **Phase 3 (Plan Output)**: Produces a structured plan with files to change, steps,
  patterns to follow, risks, and testing strategy

### Step 2: Implementation

Code is written. During this phase:
- **Path-scoped rules** are auto-loaded (code-style, api-config, distributed, testing)
  based on which files are being edited
- **Domain expert agents** can be consulted for subsystem-specific guidance
- **PostToolUse hooks** fire on every Write/Edit, reminding about expert doc updates

### Step 3: Code Verification (code-verifier, Haiku)

The code-verifier auto-activates and runs its 5-phase workflow:
1. Identifies changed files
2. Runs pre-commit (formatting + linting) -- auto-fixes what it can
3. Runs relevant tests (GPU-aware categorization)
4. Checks documentation freshness
5. Reports structured results with pass/fail/skip per check

**What it catches**: Formatting violations, linting errors, import ordering, trailing
whitespace, test failures, stale CLI docs.

### Step 4: Code Review (simple-code-reviewer, Sonnet)

The simple-code-reviewer auto-activates and reads the changed code:
- Checks AReaL-specific patterns (logging, async, tensor shapes, configs, imports)
- Looks for common issues (missing await, blocking I/O, resource leaks)
- Checks distributed code patterns (synchronization, device placement, mesh dimensions)

**What it catches**: Logic errors, pattern violations, missing synchronization, incorrect
async usage, tensor shape issues.

### Step 5: PR Review (/pr-review, Dynamic)

When a PR is opened, the dynamic PR review system:
1. Analyzes the PR to detect change types and risk levels
2. Dynamically generates review tasks with appropriate model selection
3. Dispatches parallel review agents with framework-specific checklists
4. Scores findings for confidence, filters false positives
5. Produces a comprehensive review summary

**What it catches**: Cross-component interaction bugs, parallel strategy correctness,
numerical stability issues, API compatibility breaks, performance regressions, security
concerns.

### How Different Layers Catch Different Classes of Issues

| Issue Class | Caught By | Example |
|------------|-----------|---------|
| Formatting | Layer 1 (pre-commit) | Wrong indentation, unsorted imports |
| Syntax errors | Layer 1 (Ruff lint) | Undefined variable, unused import |
| Test failures | Layer 2 (code-verifier) | Broken function after refactor |
| Pattern violations | Layer 3 (code-reviewer) | Using `print` instead of logger |
| Distributed bugs | Layer 4 (PR review) | Missing all-reduce in new TP code |
| Cross-component issues | Layer 4 (PR review) | EP change breaks FSDP interaction |
| Performance regressions | Layer 4 (PR review) | Accidental GPU-CPU sync in hot path |
| API breaks | Layer 4 (PR review) | Changed function signature without updating callers |

---

## 10. Key Design Insights

### Why "Dynamic Agent Template" Outperforms Fixed Reviewers

Traditional code review (both human and AI) applies the same checklist to every PR. The
`/pr-review` system instead **generates the checklist dynamically** based on what was
actually changed. A docs-only PR gets a single Haiku check. A core Archon change triggers
4-8 Opus deep reviews with framework-specific checklists. This means:

- **No wasted effort**: Simple changes get simple reviews
- **No blind spots**: Complex changes get the specific deep checks they need
- **Domain expertise scales**: 31 change types x 23+ templates = hundreds of specific
  review paths, all encoded as structured knowledge rather than hoping the reviewer
  remembers to check

### Cost Optimization via Tiered Model Selection

The 3-tier model strategy (Opus/Sonnet/Haiku) is an explicit cost-quality tradeoff:

- **Opus** (~$15/MTok input, $75/MTok output): Reserved for CRITICAL/HIGH risk --
  distributed parallelism, engine core, checkpoint correctness
- **Sonnet** (~$3/MTok input, $15/MTok output): Used for MEDIUM risk -- tensor ops,
  workflows, API configs
- **Haiku** (~$0.25/MTok input, $1.25/MTok output): Used for LOW risk -- docs, tests,
  config files, and orchestration tasks

A simple config PR might cost ~$0.01 to review. A core engine PR might cost ~$5-10. The
system allocates expensive reasoning exactly where it's needed.

### Confidence Scoring as False Positive Management

The Phase 4 confidence scoring (0-100) serves a critical purpose: **managing the false
positive rate**. AI code reviewers are notorious for flagging non-issues. The scoring
system explicitly defines what gets score 0 (false positive):

- Pre-existing issues not introduced by this PR
- Intentionally designed code that looks like a bug
- Issues linter/compiler would catch
- Issues on lines the user didn't modify
- Explicitly disabled issues (lint ignore comments)

This means findings are filtered before presentation, preventing reviewer fatigue from
false positives.

### Verification vs Review: The Separation Principle

The system cleanly separates two distinct activities:

| | Code Verifier | Code Reviewer |
|--|--------------|---------------|
| **Model** | Haiku | Sonnet |
| **Tools** | Read, Grep, Glob, **Bash** | Read, Grep, Glob |
| **Can execute?** | Yes (runs commands) | No (read-only) |
| **Asks** | "Does this pass?" | "Is this good?" |
| **Judges** | Objective (pass/fail) | Subjective (quality) |
| **Auto-fixes?** | Yes (formatting, imports) | No (reports only) |

Verification is objective: does it pass pre-commit? Do tests pass? Review is subjective:
is the logic correct? Are distributed patterns followed? Separating these prevents the
reviewer from being tempted to "just fix it" and skipping proper analysis.

### Institutional Knowledge Encoded as Review Templates

The 23+ review task templates in `.claude/data/pr-review-templates.md` encode
project-specific institutional knowledge that would normally exist only in senior
engineers' heads. Each template contains:

- **Applicability condition**: When this check matters
- **Detailed checklist**: Specific things to verify (e.g., "AC application order: must
  after TP/CP, before FSDP")
- **Domain context**: Why each check matters

This knowledge survives team changes, doesn't degrade with fatigue, and is consistently
applied. When a new risk is discovered (e.g., a production bug caused by incorrect EP
constraint validation), a new template or checklist item can be added, and all future PRs
touching EP code will be checked against it.

### The Planner Agent as Architectural Gate

The `planner` agent (Opus model) activates **before** code is written for complex tasks.
This is a quality measure often overlooked: preventing bad architecture from being built
in the first place. By researching existing patterns, finding callers/dependencies, and
producing a structured plan, the planner reduces the chance of implementing something
that would be caught and rejected in review.

---

*This analysis is based on direct inspection of the source files listed below. Every
claim references an actual file in the AReaL repository.*

## Appendix: Source File Reference

| File | Purpose | Section(s) |
|------|---------|------------|
| `.pre-commit-config.yaml` | Pre-commit hook configuration | 2 |
| `pyproject.toml` (lines 226-274) | Ruff + pytest configuration | 2 |
| `.claude/agents/code-verifier.md` | Code verification agent | 3 |
| `.claude/agents/simple-code-reviewer.md` | Lightweight code reviewer | 4 |
| `.claude/commands/pr-review.md` | Dynamic PR review command | 5 |
| `.claude/data/pr-review-change-types.md` | 31 change types across 4 risk levels | 5 |
| `.claude/data/pr-review-templates.md` | 23+ review task templates | 5 |
| `.claude/rules/code-style.md` | Global code style rules | 6 |
| `.claude/rules/api-config.md` | API/config design rules | 6 |
| `.claude/rules/distributed.md` | Distributed code rules | 6 |
| `.claude/rules/testing.md` | Testing rules | 6 |
| `.claude/settings.json` | Hook automation config | 8 |
| `.claude/hooks/check-expert-update.sh` | Expert update reminder hook | 8 |
| `.claude/agents/algorithm-expert.md` | RL algorithm expert | 7 |
| `.claude/agents/fsdp-engine-expert.md` | FSDP engine expert | 7 |
| `.claude/agents/archon-engine-expert.md` | Archon engine expert | 7 |
| `.claude/agents/megatron-engine-expert.md` | Megatron engine expert | 7 |
| `.claude/agents/launcher-scheduler-expert.md` | Cluster deployment expert | 7 |
| `.claude/agents/planner.md` | Planning agent | 9, 10 |
