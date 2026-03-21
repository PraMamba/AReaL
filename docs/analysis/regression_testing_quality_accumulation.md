# AReaL Regression Testing & Quality Accumulation: Complete Implementation Analysis

## 1. Overview: The Regression Testing Philosophy

As AI-assisted projects scale, a critical challenge emerges: **new features breaking
existing functionality**. The problem intensifies as module dependencies grow -- a change
to an engine's checkpoint logic might silently break a workflow that depends on it, or a
config refactor might invalidate a training pipeline that was working yesterday.

AReaL addresses this with a regression testing strategy built on three pillars:

1. **Multi-layered testing** -- unit tests, integration tests, distributed tests, and
   end-to-end training validation, each catching different classes of regressions
2. **Guided test creation** -- the `/add-unit-tests` skill and `code-verifier` agent
   ensure every new feature gets tests, and every bug fix adds a permanent guard
3. **CI gates** -- GitHub Actions provisions GPU hardware and blocks merges on test
   failure, making regression impossible to merge silently

The "snowball" metaphor works like this: each bug fix adds a regression test. That test
runs in every future CI pipeline. Over time, the test suite accumulates hundreds of
specific guards, each one a permanent scar from a past failure that can never recur. The
test suite grows monotonically with project complexity, forming an ever-expanding safety
net.

```
Bug discovered → Fix committed → Regression test added → CI runs it forever
                                         ↓
                      Next feature change triggers same test
                                         ↓
                      If test fails → change is blocked before merge
                                         ↓
                      If test passes → confirmed no regression
```

---

## 2. Test Infrastructure at Scale

### Test Inventory

| Metric | Count |
|--------|-------|
| Test files | **71** |
| Test functions (`def test_`) | **~883** |
| Test classes (`class Test`) | **~152** |
| Total test code | **~28,856 lines** |
| Pytest marker usages | **179** |
| Commits touching `areal/tests/` | **175** |

### Directory Structure

```
areal/tests/
├── test_*.py                          (43 root-level test files)
├── experimental/
│   ├── archon/                        (17 test files)
│   │   ├── torchrun/                  (11 distributed test runners)
│   │   └── conftest.py                (PyTorch version gating)
│   └── openai/                        (6 test files)
├── fp8/                               (4 test files)
├── grpo/                              (1 test + 3 configs + entrypoint)
├── sft/                               (1 test + 3 configs + 3 ref_losses files)
└── torchrun/                          (11 distributed test runners)
```

### Test Categories

| Category | Count | GPU Required | Example Files |
|----------|-------|-------------|---------------|
| **Unit tests** | ~60% | No | `test_utils.py`, `test_serialization.py`, `test_adv_norm_config.py` |
| **Integration tests** | ~30% | Yes | `grpo/test_grpo.py`, `sft/test_sft.py`, `test_examples.py` |
| **Distributed tests** | ~10% | Multi-GPU | `torchrun/run_*.py`, `experimental/archon/torchrun/` |
| **End-to-end tests** | ~5% | Yes | `test_grpo.py` (full GRPO training), `test_sft.py` (full SFT training) |

### Pytest Marker Strategy

**Source**: `pyproject.toml:226-240`

```toml
markers = [
    "slow: mark test as slow, expected to cost more than 30 seconds",
    "ci: mark test as must-run in CI (only marked for slow tests).",
    "gpu: mark test that uses a single GPU",
    "multi_gpu: mark test that uses more than one GPU",
]
```

The marker strategy enables **selective execution**:

| Marker Combination | Local Dev | CI Pipeline |
|-------------------|-----------|-------------|
| No markers (fast unit test) | Runs | Runs |
| `@pytest.mark.slow` | Runs | **Excluded** |
| `@pytest.mark.slow` + `@pytest.mark.ci` | Runs | **Runs** (forced) |
| `@pytest.mark.gpu` | Skips if no GPU | Runs (A100) |
| `@pytest.mark.multi_gpu` | Skips if <2 GPUs | Runs (2x A100) |

CI filter expression: `pytest -m "not slow or ci"` -- this excludes slow tests *unless*
they also have the `ci` marker.

### Version Gating

**Source**: `areal/tests/experimental/archon/conftest.py`

Archon tests require PyTorch >= 2.9.1. A conftest.py hook skips the entire test suite on
older versions:

```python
_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
_MIN_TORCH_VERSION = (2, 9, 1)

if _TORCH_VERSION < _MIN_TORCH_VERSION:
    collect_ignore_glob = ["test_*.py"]
```

---

## 3. How Regression Tests Are Written

AReaL uses 5 distinct patterns for regression testing, each suited to a different class
of regression.

### Pattern 1: Explicit Regression Tests

Named `test_regression_issue_NNN()` with docstrings documenting the original bug, the fix,
and the specific error that must never recur.

**Example: Issue #482 -- GRPOConfig AttributeError**

**Source**: `areal/tests/test_model_utils.py:175-194`

```python
def test_regression_issue_482(self):
    """Regression test for issue #482 - GRPOConfig with vLLM training.

    This test verifies that the bug reported in issue #482 is fixed.
    The bug was: AttributeError: 'GRPOConfig' object has no attribute
    'weight_update_mode'
    The fix accesses config.actor.weight_update_mode instead.
    """
    config = create_grpo_config(
        experiment_name="boba_grpo_vllm_16_gpus",
        trial_name="trial_0",
        allocation_mode="vllm:d4p1t2+d8p1t1",
        fileroot="/tmp/areal",
    )

    # This should not raise AttributeError
    result = get_model_update_meta(config)

    assert isinstance(result, WeightUpdateMeta)
    assert result.type == "disk"
```

**Example: TrainController Missing Method Parameter**

**Source**: `areal/tests/test_train_controller.py:282-299`

```python
def test_merge_results_accepts_method_parameter(self, train_controller):
    """Test that _merge_results accepts method parameter.

    This is a regression test for the bug at line 279 where the method
    parameter was missing from the signature.
    """
    results = [torch.tensor([[0.5, 0.5]]), torch.tensor([[0.3, 0.3]])]

    # This should work without TypeError
    try:
        result = train_controller._merge_results(
            results, group_indices=[[0], [1]]
        )
        assert result is not None
    except TypeError as e:
        if "missing" in str(e) and "required positional argument" in str(e):
            pytest.fail(f"_merge_results missing required parameter: {e}")
```

### Pattern 2: Reference Value Comparison

SFT tests compare actual training losses against **frozen reference files** for each
backend. Any code change that shifts training behavior -- even slightly -- triggers a
failure.

**Source**: `areal/tests/sft/test_sft.py:65-75`

```python
with open(os.path.join(tmp_path, "losses.json")) as f:
    losses: list[float] = json.load(f)

with open(ref_losses_path) as f:
    ref_losses: list[float] = json.load(f)

assert all(
    loss == pytest.approx(ref_loss, rel=1.6e-2, abs=1e-5)
    for loss, ref_loss in zip(losses, ref_losses)
)
```

Three reference files anchor the expected behavior:

- `areal/tests/sft/ref_losses_fsdp.json`
- `areal/tests/sft/ref_losses_megatron.json`
- `areal/tests/sft/ref_losses_archon.json`

The tolerance (`rel=1.6e-2, abs=1e-5`) is tight enough to catch meaningful regressions
while accommodating floating-point nondeterminism across GPU runs.

### Pattern 3: End-to-End Behavioral Thresholds

GRPO tests run **actual RL training** and assert that the model achieves a minimum reward,
validating that the entire pipeline (dataset loading, tokenization, inference, reward
computation, loss calculation, optimization) works end-to-end.

**Source**: `areal/tests/grpo/test_grpo.py:66-69`

```python
with open(os.path.join(tmp_path, "rewards.json")) as f:
    rewards: list[float] = json.load(f)

assert rewards[-1] > 0.6
```

This test is parametrized across all three backends:

```python
@pytest.mark.parametrize("backend", ["fsdp", "megatron", "archon"])
def test_grpo(tmp_path: str, backend: str) -> None:
```

A regression in *any* component (workflow, engine, reward, dataset, trainer) that
prevents the model from learning will cause `rewards[-1]` to stay below 0.6, catching
the regression.

### Pattern 4: Cross-Backend Consistency

Both SFT and GRPO tests use `@pytest.mark.parametrize("backend", ["fsdp", "megatron",
"archon"])` to ensure all three training engines produce consistent results. This catches
backend-specific regressions that might be invisible if only one engine were tested.

The effect: a single test function generates 3 separate test cases. If FSDP regresses but
Megatron doesn't, the FSDP case fails independently, precisely identifying the regression
source.

### Pattern 5: Error Message Regression Guards

Some tests document the exact error message they prevent, creating a direct link between
the test and the specific failure mode it guards against.

**Source**: `areal/tests/experimental/archon/test_parallel_dims.py:263-271`

```python
class TestETPMeshDimensions:
    """Test that ep_tp mesh is correctly 2D after flattening.

    This tests the fix for the ValueError:
    `placements` must have the same length as `device_mesh.ndim`!
    Found placements length: 2, and device_mesh.ndim: 3.

    The ep_tp mesh must be 2D [ep, tp] for ExpertTensorParallel to work.
    """
```

**Source**: `areal/tests/experimental/archon/torchrun/run_parallel_dims.py:119-122`

```python
"""
This tests the fix for the ValueError:
`placements` must have the same length as `device_mesh.ndim`!
Found placements length: 2, and device_mesh.ndim: 3.
"""
```

The error message is embedded directly in the docstring so that any developer (or AI)
reading the test immediately understands what failure it prevents and why it exists.

---

## 4. How Critical Behaviors Are Covered

### Module Coverage Map

| Production Module | Test Files | Coverage Tier |
|------------------|------------|---------------|
| `areal/engine/fsdp_engine.py` | `test_fsdp_*.py` (7 files) | **Comprehensive** |
| `areal/engine/megatron_engine.py` | `test_megatron_engine*.py` (2 files) | **Comprehensive** |
| `areal/experimental/engine/archon_engine.py` | `experimental/archon/` (17 files) | **Comprehensive** |
| `areal/infra/scheduler/` | `test_local_scheduler.py`, `test_ray_scheduler.py`, `test_slurm_scheduler.py` | **Comprehensive** |
| `areal/infra/` (controllers) | `test_train_controller.py`, `test_rollout_controller.py` | **Comprehensive** |
| `areal/utils/` | `test_utils.py`, `test_functional.py`, `test_serialization.py`, `test_datapack.py`, `test_rtensor.py` | **Comprehensive** |
| `areal/api/` | `test_model_utils.py`, `test_allocation_mode.py`, `test_adv_norm_config.py` | **Good** |
| `areal/reward/` | `test_math_verify_reward.py` | **Good** |
| `areal/experimental/` (non-archon) | `experimental/openai/` (6 files), `fp8/` (4 files) | **Good** |
| `areal/dataset/` | (via `test_examples.py` integration) | **Indirect** |
| `areal/workflow/` | `test_workflow_detection.py` + integration tests | **Indirect** |
| `areal/trainer/` | (via `test_grpo.py`, `test_sft.py` integration) | **Indirect** |

### Integration Tests as Cross-Module Safety Nets

The integration tests exercise the full dependency chain, catching regressions that unit
tests cannot:

**`test_grpo.py`** exercises:
- `areal.api.cli_args.GRPOConfig` (config parsing)
- `areal.infra.launcher.local` (process launching)
- `areal.engine.{fsdp,megatron}_engine` / `areal.experimental.engine.archon_engine`
  (training)
- `areal.workflow.rlvr` (rollout workflow)
- `areal.reward.*` (reward computation)
- `areal.dataset.*` (data loading)

A regression in *any* of these components will cause the GRPO test to fail.

**`test_sft.py`** exercises a similar chain for supervised fine-tuning, with the added
protection of reference loss comparison.

### The Largest Test Files

| Test File | Lines | What It Protects |
|-----------|-------|-----------------|
| `test_local_scheduler.py` | 2,462 | Worker lifecycle, GPU allocation, port management, error recovery |
| `test_rollout_controller.py` | 1,394 | Controller + scheduler + engine interactions, async callbacks |
| `test_examples.py` | 830 | End-to-end example training runs, success pattern matching |
| `test_datapack.py` | ~800 | Data packing, sequence length management, batch splitting |
| `test_train_controller.py` | 764 | Train controller + worker coordination, weight updates |

These files represent the most heavily tested components -- the ones where regressions
would be most costly.

---

## 5. The Snowball Accumulation Machinery

Six mechanisms ensure that the test suite grows with the project, creating an
ever-expanding safety net.

### Mechanism 1: `/add-unit-tests` Skill -- Guided Test Creation

**Source**: `.claude/skills/add-unit-tests/SKILL.md`

The `/add-unit-tests` skill provides a step-by-step guide for adding tests:

1. **Understand test types** -- unit vs distributed, location patterns
2. **Create test file** -- naming convention `test_<module>_<feature>.py`
3. **Write test functions** -- Arrange-Act-Assert pattern with descriptive names
4. **Add pytest markers** -- CI strategy via `slow`/`ci`/`gpu`/`multi_gpu`
5. **Mock distributed environment** -- `torch.distributed.fake_pg` for unit tests
6. **Handle GPU dependencies** -- graceful skip with `@pytest.mark.skipif`

The skill explicitly integrates with other development skills:

| After This Skill... | ...Run This |
|---------------------|-------------|
| `/add-dataset` | Add dataset loader tests |
| `/add-workflow` | Add workflow behavior tests |
| `/add-reward` | Add reward function tests |

This creates a structural expectation: **every new feature gets tests** because the
development workflow explicitly includes test creation as a step.

### Mechanism 2: `code-verifier` Agent -- Proactive Test Execution

**Source**: `.claude/agents/code-verifier.md`

The code-verifier agent (Haiku model, with Bash execution capability) activates
**proactively** after every code change. Its Phase 3 workflow:

1. Checks GPU availability
2. Maps modified files to relevant test files (e.g., modified
   `areal/workflow/multi_turn.py` → run `areal/tests/test_workflow.py`)
3. Runs appropriate tests based on GPU availability:
   - Unit tests: always run
   - GRPO/FSDP tests: run only if GPU available
   - Distributed tests: run only if multi-GPU available
4. Reports structured results with pass/fail/skip per test category

When tests fail, the agent reports the failure immediately, preventing broken code from
being committed. This creates a **tight feedback loop**: code change → test failure →
immediate fix → re-test → commit.

### Mechanism 3: Testing Rules -- Path-Scoped Standards

**Source**: `.claude/rules/testing.md`

When editing any file matching `**/tests/**`, `*_test.py`, or `test_*.py`, the testing
rules are auto-loaded into the AI's context. These enforce:

- **Test naming**: `test_<what>_<condition>_<expected>()`
- **Structure**: Arrange-Act-Assert pattern
- **GPU handling**: Always skip gracefully with `@pytest.mark.skipif`
- **Assertions**: `torch.testing.assert_close()` with explicit `rtol`/`atol`
- **Mocking**: Use `torch.distributed.fake_pg`, don't mock FSDP/DTensor internals
- **Fixtures**: `tmp_path` over manual temp dirs, `monkeypatch` for env vars

These rules ensure that **every test written follows the same patterns**, making the test
suite consistent and maintainable as it grows.

### Mechanism 4: PR Review "Test Coverage Check"

**Source**: `.claude/data/pr-review-templates.md:383-391`

The `/pr-review` command includes a "Test Coverage Check" template that fires on any PR
containing test changes:

```
### Test Coverage Check [Haiku]

Applicable: TESTS type detected
Checklist:
- Test cases cover main paths
- Boundary condition tests
- Error handling tests
```

This means every PR is reviewed for test quality -- not just whether tests exist, but
whether they cover the right things.

### Mechanism 5: CI Pipeline Gates

**Source**: `.github/workflows/test-areal.yml`

The CI pipeline provisions real GPU hardware and runs tests in three tiers (detailed in
Section 6). The key property: **PRs cannot merge if tests fail**. This means:

- A regression introduced in a PR is caught *before* it reaches main
- The developer must fix the regression (and potentially add a new regression test) before
  the PR can be approved
- Each merge to main is guaranteed to pass all tests

### Mechanism 6: CLAUDE.md Mandate

**Source**: `CLAUDE.md`

The project instructions include an explicit mandate under "Always Do":

> Add tests for new functionality

This is a structural rule that applies to every code change. Combined with the
`code-verifier` agent that proactively runs tests, this creates a **two-layer
enforcement**: the rule says "add tests," and the agent verifies they pass.

---

## 6. The CI Pipeline: Three Test Tiers

### Tier 1: Format Check

**Source**: `.github/workflows/format-check.yml`

| Property | Value |
|----------|-------|
| Trigger | All pull requests |
| Runner | `ubuntu-latest` (no GPU) |
| Tools | Ruff v0.14.9, clang-format v19.1.7 |

```bash
ruff check areal/ examples/
ruff format --check areal/ examples/
find . -type f \( -name '*.c' -o ... \) -exec clang-format --dry-run --Werror {} +
```

Catches formatting and linting regressions. Fast, no GPU needed, runs on every PR.

### Tier 2: Unit + Experimental Tests

**Source**: `.github/workflows/test-areal.yml:265-275`

| Property | Value |
|----------|-------|
| Trigger | PRs with `safe-to-test` label, `workflow_dispatch` |
| Runner | Self-hosted GCP `a2-highgpu-2g` (2x A100 GPUs) |
| Container | `ghcr.io/inclusionai/areal-runtime:dev` |
| Timeout | 120 minutes |
| Environment | `CI=true`, `AREAL_IS_IN_CI=1` |

```bash
pytest -m "not slow or ci" --durations=20 -s -vv \
    areal/tests/test_*.py areal/tests/experimental/
```

This runs all unit and experimental tests, excluding slow tests unless they have the `ci`
marker. The `--durations=20` flag reports the 20 slowest tests for performance monitoring.

### Tier 3: Integration Tests

**Source**: `.github/workflows/test-areal.yml:277-297`

```bash
# SFT integration tests
pytest -s -vv areal/tests/sft/

# GRPO integration tests
pytest -s -vv areal/tests/grpo/
```

These run **full training pipelines** -- SFT with reference loss comparison, GRPO with
reward threshold validation -- across all three backends (FSDP, Megatron, Archon).

### Infrastructure: Dynamic GPU Provisioning

The CI pipeline dynamically provisions and tears down GPU instances:

1. **Provision**: Creates `a2-highgpu-2g` GCP instance with A100 GPUs, tries multiple
   zones
2. **Setup**: Starts Docker container with NVIDIA runtime, GPU passthrough, 58GB shared
   memory
3. **Test**: Runs all three test tiers sequentially
4. **Cleanup**: Deletes instance (runs `always()`, even if tests fail)

Key safeguards:

- `max-run-duration: "2h"` and `instance-termination-action: DELETE` prevent runaway costs
- `concurrency: cancel-in-progress: true` cancels stale test runs when new commits push
- Ephemeral runners ensure clean state for every test run

---

## 7. The Bug Fix → Test → Guard Lifecycle

### Concrete Example: Issue #482

**Timeline**:

1. **Bug discovered**: `AttributeError: 'GRPOConfig' object has no attribute
   'weight_update_mode'` when using GRPOConfig with vLLM training
2. **Root cause**: Code accessed `config.weight_update_mode` directly instead of
   `config.actor.weight_update_mode`
3. **Fix committed** (commit `9d3666d`): 34 lines of production code changed
4. **Regression test added** (same commit): 195 lines of tests including
   `test_regression_issue_482()` with full docstring documenting the bug, the fix, and the
   expected behavior
5. **Permanent guard**: The test runs in every CI pipeline. Any future change that
   reintroduces the `AttributeError` will be caught immediately.

### The Feedback Loop

```
Developer/AI makes code change
         ↓
code-verifier agent auto-activates
         ↓
Runs pre-commit (formatting) + relevant tests
         ↓
┌─── Test passes ──→ Ready to commit ──→ PR opened
│                                             ↓
│                                   CI pipeline runs all 3 tiers
│                                             ↓
│                                   ┌─── Tests pass ──→ Merge allowed
│                                   │
│                                   └─── Tests fail ──→ Fix required
│                                                           ↓
│                                                  New regression test added
│                                                           ↓
│                                                  CI pipeline re-runs
│                                                           ↓
│                                                  Merge (with new guard)
│
└─── Test fails ──→ Fix code immediately
                         ↓
                    (Optional) Add regression test if bug was non-obvious
                         ↓
                    Re-run code-verifier
                         ↓
                    Commit when passing
```

### Why This Creates Snowball Accumulation

Each iteration through this loop can **only add** to the test suite, never subtract. The
process is:

1. A bug is found (either by a human, by the AI, or by a failing test)
2. The bug is fixed
3. A test is added that specifically guards against this bug
4. The test runs in every future CI pipeline

Over time:

- 175 commits have touched `areal/tests/`
- Each commit either adds new tests, updates existing tests to match refactors, or
  consolidates redundant tests
- The test count grows monotonically with project complexity
- Each new test is a **permanent scar** -- a record of a past failure that will never
  recur

The snowball only grows. Even test consolidation (like commit `55ff540` which removed
4,000+ lines of redundant archon tests) preserves coverage while reducing maintenance
burden -- the guards remain, just expressed more efficiently.

---

## 8. Coverage Gaps and Expansion Opportunities

### Currently Under-Tested Modules

| Module | Current Coverage | Gap |
|--------|-----------------|-----|
| `areal/dataset/` (6 loaders) | Indirect via `test_examples.py`, `test_grpo.py` | No unit tests for individual loaders |
| `areal/workflow/` (core logic) | `test_workflow_detection.py` + integration | No unit tests for `arun_episode` logic |
| `areal/trainer/` (PPO actor/critic) | Indirect via `test_grpo.py` | No isolated trainer unit tests |
| `areal/utils/logging.py` | No dedicated tests | Logging configuration untested |

### The Gap-Closing Pattern

When a regression is discovered in an under-tested module, the fix commit adds both:

1. The production code fix
2. Targeted unit tests for the specific behavior that regressed

This pattern naturally prioritizes test coverage for the most regression-prone areas: the
modules that break most often get the most tests, because each break adds a guard.

### Skill Integration for Gap Coverage

The `/add-unit-tests` skill explicitly connects to feature-creation skills:

- **After `/add-dataset`**: Add tests for the new dataset loader (data format validation,
  tokenizer compatibility, max_length truncation, distributed sampling)
- **After `/add-workflow`**: Add tests for the new workflow (async correctness, tensor
  output format, engine interaction)
- **After `/add-reward`**: Add tests for the new reward function (signature match,
  determinism, numerical range, edge cases)

This means every new feature module starts with at least baseline test coverage from day
one.

---

## 9. Key Design Insights

### Reference Value Files as Regression Anchors

The `ref_losses_{fsdp,megatron,archon}.json` files represent a powerful technique:
**freezing expected numerical behavior** as a regression anchor. Any code change that
alters the training trajectory -- even if it doesn't cause a crash -- will produce
different losses that fail the `pytest.approx(ref_loss, rel=1.6e-2, abs=1e-5)` check.

This catches a class of regressions that traditional tests miss: **silent correctness
regressions** where the code runs without errors but produces subtly wrong results. A
change to gradient clipping, a dtype conversion bug, or a normalization error would be
invisible to a test that only checks "did it run?" but would be caught immediately by
loss comparison.

### Parametrized Cross-Backend Testing

A single test function with
`@pytest.mark.parametrize("backend", ["fsdp", "megatron", "archon"])` multiplies
coverage by 3x. This is particularly effective for catching backend-specific regressions:

- A change to FSDP's gradient handling that doesn't affect Megatron
- An Archon-specific MoE routing change that breaks under certain configs
- A Megatron pipeline schedule change that shifts loss values

All three backends produce independently verifiable results from the same test logic.

### GPU-Aware Graceful Degradation

Tests skip instead of fail when GPU hardware is unavailable:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_feature():
    ...
```

But CI **always** has GPUs (2x A100). This means:

- Local development: fast feedback from unit tests, GPU tests skipped gracefully
- CI pipeline: **full coverage** including GPU and distributed tests
- Nothing is actually skipped in the merge gate -- it's impossible to merge code that
  breaks GPU tests

### The "slow + ci" Marker Pattern

Some tests take >10 seconds but are critical for regression prevention. The dual-marker
pattern resolves the tension:

```python
@pytest.mark.slow        # Excluded from default CI run
@pytest.mark.ci          # But forced back in by CI filter
def test_slow_but_critical():
    ...
```

CI filter: `pytest -m "not slow or ci"` -- excludes slow tests, but re-includes any
slow test that also has `ci`. This allows developers to run fast tests locally while
ensuring critical slow tests still run in CI.

### Test Consolidation as Quality Maintenance

Quality accumulation isn't only about adding tests -- it's also about **pruning
redundancy**. Commit `55ff540` ("refactor(archon): consolidate and simplify test suite
#888") removed 4,000+ lines of redundant archon tests while preserving coverage.

This prevents the test suite from becoming unmaintainable as it grows. The principle:
each behavior should be tested exactly once, with the most efficient test that covers it.
Redundant tests are noise that slow CI and obscure signal.

### The "Contract" Metaphor

Tests serve as **contracts** between the code and its users (whether human or AI):

- **Unit tests** are contracts on individual function behavior ("this function returns X
  given Y")
- **Integration tests** are contracts on component interaction ("these components work
  together to produce Z")
- **Regression tests** are contracts on specific bug fixes ("this specific bug will never
  recur")
- **Reference value tests** are contracts on numerical behavior ("training with these
  inputs produces these losses")

When AI implements a new feature, these contracts constrain it: the feature must work
without violating any existing contract. If it does, the test fails, and the AI
immediately knows what it broke and why.

---

*This analysis is based on direct inspection of the source files listed below. Every
claim references an actual file in the AReaL repository.*

## Appendix: Source File Reference

| File | Purpose | Section(s) |
|------|---------|------------|
| `pyproject.toml:226-274` | Pytest markers + Ruff config | 2 |
| `areal/tests/test_model_utils.py:175-194` | Regression test for issue #482 | 3, 7 |
| `areal/tests/test_train_controller.py:282-299` | Regression test for missing method param | 3 |
| `areal/tests/sft/test_sft.py` | Reference loss comparison test | 3 |
| `areal/tests/sft/ref_losses_fsdp.json` | Frozen reference losses (FSDP) | 3 |
| `areal/tests/sft/ref_losses_megatron.json` | Frozen reference losses (Megatron) | 3 |
| `areal/tests/sft/ref_losses_archon.json` | Frozen reference losses (Archon) | 3 |
| `areal/tests/grpo/test_grpo.py` | End-to-end GRPO with reward threshold | 3 |
| `areal/tests/experimental/archon/test_parallel_dims.py:263-271` | Error message regression guard | 3 |
| `areal/tests/experimental/archon/torchrun/run_parallel_dims.py:119-122` | Distributed fix test | 3 |
| `.claude/skills/add-unit-tests/SKILL.md` | Guided test creation workflow | 5 |
| `.claude/agents/code-verifier.md` | Proactive test execution agent | 5 |
| `.claude/rules/testing.md` | Path-scoped testing rules | 5 |
| `.claude/data/pr-review-templates.md:383-391` | Test Coverage Check template | 5 |
| `.github/workflows/test-areal.yml` | GPU CI pipeline on GCP A100 | 6 |
| `.github/workflows/format-check.yml` | Format/lint CI check | 6 |
| `.github/workflows/install-test.yml` | Package installation tests | 6 |
| `areal/tests/experimental/archon/conftest.py` | PyTorch version gating | 2 |
| `areal/tests/test_local_scheduler.py` | Largest test file (2,462 lines) | 4 |
| `CLAUDE.md` | "Always Do: Add tests for new functionality" | 5 |
