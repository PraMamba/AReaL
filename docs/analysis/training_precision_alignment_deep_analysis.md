# AReaL 训练精度对齐体系源码与 Commit History 深度分析

## 0. Executive Summary

**项目名称**: AReaL (Asynchronous Reinforcement Learning for LLM Alignment)

**总体判断**: **较完整** — 在 determinism 控制、mixed precision 对齐、分布式并行 correctness
测试方面具备系统性能力，但缺少 golden loss regression CI、自动化二分定位工具和通用 tensor dump 框架。

**最强能力**:

1. 多引擎数值一致性验证体系（Archon vs FSDP vs HuggingFace 三路对比）
1. FP8/BF16 逐层逐算子精度对比框架（含 cosine similarity、activation hooks、operation categorization）
1. 确定性训练模式（Megatron + Archon 双引擎，覆盖 CUBLAS/NCCL/TE/compile/AC RNG 全链路）
1. FP32 master weights 体系（FSDP optimizer_dtype 解耦、Megatron precision-aware
   optimizer、Kahan summation AdamW）

**最大短板**:

1. 无 RL (PPO/GRPO) golden loss regression（SFT 已有 16 步 golden loss，但 RL 训练尚未覆盖）
1. 无通用 activation/gradient dump-and-compare 工具（FP8 测试有 hooks 但未推广）
1. Nightly CI 尚为占位符（`Dummy test (placeholder)`）
1. 无 TF32 显式控制（依赖 PyTorch 默认值或 deterministic mode 隐式关闭）
1. 无跨硬件（A100/H100/Ascend）精度对齐验证

**最值得借鉴的源码模块**:

- `tests/experimental/archon/` — 完整的多并行策略 correctness matrix
- `tests/fp8/comparison_utils.py` — FP8/BF16 逐层对比框架
- `areal/engine/megatron_utils/deterministic.py` +
  `areal/experimental/engine/archon_utils.py` — 确定性训练配置
- `areal/engine/fsdp_utils/grad.py` — FP32 gradient norm 全链路
- `tests/test_cuda_deterministic.py` — bit-identical 回归测试
- `areal/utils/seeding.py` — 基于 SHA256 的 per-role seed 系统

**最值得研究的 commits / PR**:

1. `237a49f6` — fix(fsdp): maintain fp32 master weights for AdamW (#1292)
1. `5b4ed832` — feat(archon): add deterministic training mode (#943)
1. `89dda13a` — \[Feat\] Add FP8 training support (#758)
1. `5e9fb505` — fix(megatron): compensate pipeline schedule's /num_microbatches in grad
   (#1273)
1. `4f5a2944` — feat(archon): add moe_router_dtype config for FP32 router gate GEMM
   (#1009)
1. `055066a9` — fix(archon): default score_before_experts to False for HF parity (#940)
1. `e7c4a49a` — Fix the dataloader shuffle and random seed issue (PR #4)

**是否适合作为训练精度对齐基础设施参考**: **是** — 特别适合学习多引擎对比验证、FP8 精度对比框架和确定性训练配置，但需自行补齐 golden loss
regression 和通用 tensor dump 能力。

______________________________________________________________________

## 1. 项目训练流程与精度相关架构总览

### 1.1 训练主入口

AReaL 提供三种训练器入口：

- **RL 训练**: `areal/trainer/rl_trainer.py:PPOTrainer` — GRPO/PPO/DPO RL 训练主控
- **SFT 训练**: `areal/trainer/sft_trainer.py:SFTTrainer`
- **DPO 训练**: `areal/trainer/dpo_trainer.py:DPOTrainer`

所有 Trainer 在初始化时调用
`seeding.set_random_seed(config.seed, key=f"trainer{rank}")`（rl_trainer.py:129,
sft_trainer.py:76, dpo_trainer.py:107）。

### 1.2 三大训练引擎

| 引擎           | 文件                                         | 精度相关特性                                                                  |
| -------------- | -------------------------------------------- | ----------------------------------------------------------------------------- |
| FSDPEngine     | `areal/engine/fsdp_engine.py`                | FSDP2 MixedPrecisionPolicy、optimizer_dtype 解耦、FP32 grad norm              |
| MegatronEngine | `areal/engine/megatron_engine.py`            | Megatron-Core optimizer、FP8 TE 集成、loss scaling、precision-aware optimizer |
| ArchonEngine   | `areal/experimental/engine/archon_engine.py` | PyTorch native FSDP2、torch.compile + deterministic、FP8 blockwise            |

### 1.3 配置系统如何传递精度参数

核心配置定义在 `areal/api/cli_args.py`，关键字段：

| 字段                           | 位置(行号) | 默认值       | 作用                          |
| ------------------------------ | ---------- | ------------ | ----------------------------- |
| `dtype`                        | ~1109      | `"bfloat16"` | forward/backward 计算精度     |
| `grad_reduce_dtype`            | ~1113      | `"float32"`  | gradient all-reduce 精度      |
| `optimizer_dtype`              | ~1117      | `"float32"`  | 参数存储精度(master weights)  |
| `use_deterministic_algorithms` | 661, 894   | `False`      | 确定性训练开关                |
| `ac_preserve_rng_state`        | 585-588    | `False`      | AC 保存 RNG 状态              |
| `disable_dropout`              | -          | `True`       | 禁用 dropout                  |
| `moe_router_dtype`             | 672-678    | `"fp32"`     | MoE router gate 精度          |
| `seed` / `random_seed`         | 1796       | `1`          | 全局随机种子                  |
| `initial_loss_scale`           | 386        | -            | Megatron loss scaling 初始值  |
| `grad_reduce_in_fp32`          | 718        | `True`       | Megatron DDP FP32 梯度 reduce |

### 1.4 训练 Step 流程

**FSDP `train_batch`** (fsdp_engine.py:760-798):

1. `optimizer_zero_grad()` — 清零梯度
1. 构建 `MicroBatchList`
1. `compute_total_loss_weight()` — **FP32** all-reduce loss 权重（core/train_engine.py:59
   显式 `.float()`）
1. `forward_backward_batch()` — 逐 micro-batch forward (FSDP cast to param_dtype) +
   backward (reduce in reduce_dtype)
1. `optimizer_step()` — `fsdp2_clip_grad_norm` (**FP32** 全链路) → `optimizer.step()`

**Megatron `train_batch`** (megatron_engine.py:902-960):

1. `optimizer_zero_grad()` — 清零梯度和 gradient buffers
1. 构建 `MicroBatchList`
1. `compute_total_loss_weight()` — FP32 all-reduce over DP+CP group
1. 计算 `loss_multiplier = dp_world_size * loss_scale * num_microbatches`（补偿 Megatron 内部的
   `/num_microbatches`）
1. `forward_backward_batch()` — Megatron pipeline schedule (1F1B / GPipe)，调用
   `finalize_model_grads` 做 DP gradient reduce
1. `optimizer_step()` — Megatron optimizer 内部处理 grad clip 和 loss scaling

______________________________________________________________________

## 2. 精度对齐能力矩阵

| 能力项                                   | 是否具备    | 源码证据                                                                                                   | commit/PR 证据                     | 成熟度 | 备注                                                                                    |
| ---------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------- | ------ | --------------------------------------------------------------------------------------- |
| **配置一致性扫描**                       | 间接存在    | `cli_args.py` `__post_init__` 验证                                                                         | `84eaef12`                         | 2      | 有字段验证但无配置 diff 工具                                                            |
| **随机种子/RNG 控制**                    | ✅ 明确存在 | `areal/utils/seeding.py`, `megatron_utils/deterministic.py`, `archon_utils.py:213-249`                     | `5b4ed832`, `d5b98d6d`, `e7c4a49a` | 3      | SHA256-based per-role seed, CUBLAS/NCCL/TE/compile 全覆盖                               |
| **数据加载顺序确定性**                   | ✅ 明确存在 | `seeding.py:Shuffler`, `set_random_seed` in data workers                                                   | `e7c4a49a`, `22f357b3`, `ad622efd` | 3      | per-epoch per-rank 不同 seed                                                            |
| **初始权重一致性**                       | ✅ 明确存在 | `fsdp_engine.py` memory_efficient_load broadcast, `megatron_engine.py:243` TP-aware seed                   | `128299b2`                         | 3      | rank 0 加载 + broadcast                                                                 |
| **单步 forward loss 对齐**               | ✅ 明确存在 | `test_grpo.py`, `test_forward.py`, `test_hf_parity_*.py`                                                   | `055066a9`, `5b4ed832`             | 3      | Archon vs FSDP vs HF 三路对比                                                           |
| **activation dump/compare**              | 部分存在    | `tests/fp8/model_hooks.py` hooks 系统, `test_hf_parity_qwen3.py` `_capture_hf`/`_capture_archon`           | `89dda13a`                         | 2      | 仅限 FP8 和 HF parity 测试，无通用框架                                                  |
| **gradient dump/compare**                | 部分存在    | `tests/fp8/model_hooks.py:collect_gradients_after_train_batch`, `test_grpo.py:test_logprobs_gradient_flow` | `89dda13a`                         | 2      | FP8 测试有 per-param gradient 对比，含 NaN/Inf 检查                                     |
| **optimizer state 对齐**                 | ✅ 明确存在 | `test_fsdp_optimizer_dtype.py`, `fsdp_utils/optimizer.py:AnyPrecisionAdamW`                                | `237a49f6`                         | 3      | 5 dtype invariant 回归测试                                                              |
| **scheduler/lr curve 对齐**              | 未发现      | -                                                                                                          | -                                  | 0      | 无 LR scheduler 数值一致性测试                                                          |
| **loss curve golden regression**         | ✅ 明确存在 | `tests/sft/test_sft.py`, `ref_losses_fsdp.json` / `ref_losses_megatron.json` / `ref_losses_archon.json`    | -                                  | 3      | 16 步 SFT golden loss regression (rel=1.6%, abs=1e-5)，覆盖 FSDP/Megatron/Archon 三引擎 |
| **mixed precision 对齐**                 | ✅ 明确存在 | `fsdp_utils/parallel.py:MixedPrecisionPolicy`, optimizer_dtype 解耦, FP32 grad reduce                      | `237a49f6`, `7f72f4c5`             | 3      | compute/reduce/storage 三 dtype 解耦                                                    |
| **FP16/BF16/FP8 数值稳定性**             | ✅ 明确存在 | `tests/fp8/`, `functional.py:231` fp32 upcast, `vocab_parallel.py:18` logits.float()                       | `89dda13a`, `f6331e09`, `0b15ead8` | 3      | FP8 逐层 cosine similarity 对比                                                         |
| **TF32 控制**                            | 间接存在    | `torch.use_deterministic_algorithms(True)` 隐式关闭 TF32                                                   | `5b4ed832`                         | 1      | 无显式 `allow_tf32` 设置                                                                |
| **NaN/Inf/overflow 检测**                | ✅ 明确存在 | `actor.py:760`, `functional.py:233`, `test_fp8_linear.py:122-164`, `torchrun/dist_utils.py:183-185`        | -                                  | 2      | PPO actor 硬 error，FP8 测试断言                                                        |
| **checkpoint resume 一致性**             | ✅ 明确存在 | `test_checkpoint_e2e.py`, `run_checkpoint_tests.py:save_load_forward_match`                                | `128299b2`, `f5b7fba7`             | 3      | save→load→forward 输出 allclose 验证                                                    |
| **data parallel correctness**            | ✅ 明确存在 | `test_distributed_dp.py`                                                                                   | -                                  | 3      | torchrun 多 GPU                                                                         |
| **tensor parallel correctness**          | ✅ 明确存在 | `test_distributed_tp.py`, `run_tp_forward.py`                                                              | `82021272`, `a8c8fd67`             | 3      | TP 前向输出与非 TP 对比                                                                 |
| **pipeline parallel correctness**        | ✅ 明确存在 | `test_distributed_pp.py`, `run_pp_gradient_verify.py`, `run_pp_tests.py`                                   | `945bdd52`, `5e9fb505`             | 3      | PP vs non-PP gradient 一致性验证                                                        |
| **sequence parallel correctness**        | ✅ 明确存在 | `test_distributed_cp.py`, `run_cp_forward.py`, `test_ulysses_all_to_all.py`                                | `1388338c`, `7927735c`             | 3      | Ulysses CP forward 验证                                                                 |
| **expert parallel / MoE correctness**    | ✅ 明确存在 | `test_distributed_ep.py`, `test_moe_hf_parity.py`, `test_moe_common.py`                                    | `106037c1`, `055066a9`             | 3      | EP/ETP + HF parity                                                                      |
| **collective communication correctness** | 间接存在    | `warmup_process_groups` dummy all-reduce, deterministic NCCL_ALGO=Ring                                     | `5b4ed832`                         | 2      | 无独立 collective 验证                                                                  |
| **CI 精度回归测试**                      | 部分存在    | `.github/workflows/test-areal.yml` 运行 unit tests on GPU                                                  | `4eb423c6`                         | 2      | PR CI 有 GPU 测试，但 nightly 为占位符                                                  |
| **自动化二分定位能力**                   | ❌ 未发现   | -                                                                                                          | -                                  | 0      | 无 bisect 工具                                                                          |
| **跨硬件/跨后端对齐能力**                | 间接存在    | `areal/infra/platforms/` 抽象层 (CUDA/Ascend)                                                              | -                                  | 1      | 有平台抽象但无跨平台精度验证                                                            |

______________________________________________________________________

## 3. 源码证据地图

### 3.1 随机性控制

| 组件                | 文件:行号                                                      | 机制                                                                                                                        |
| ------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 全局 seed           | `areal/utils/seeding.py:22-36`                                 | `set_random_seed(base_seed, key)`: SHA256(key) 派生 per-role seed，设置 PYTHONHASHSEED/transformers/random/numpy/torch/cuda |
| Megatron TP seed    | `megatron_engine.py:243`                                       | `tensor_parallel.model_parallel_cuda_manual_seed(self.seed)`                                                                |
| Trainer seed        | `rl_trainer.py:129`, `sft_trainer.py:76`, `dpo_trainer.py:107` | 初始化时调用 `seeding.set_random_seed`                                                                                      |
| Data worker seed    | `infra/data_service/worker/app.py:67`                          | `seeding.set_random_seed(config.seed, key=f"data_worker_{config.rank}")`                                                    |
| Inference seed      | `infra/launcher/sglang_server.py:180`                          | `config.random_seed = base_random_seed + server_local_idx`                                                                  |
| AC RNG preservation | `archon_utils.py:222-223`                                      | deterministic mode 下强制 `preserve_rng_state=True`                                                                         |
| MoE expert ordering | `moe/utils.py:54`                                              | `stable=True` 确保 token 分配确定性                                                                                         |
| Shuffle seed        | `seeding.py:46-61`                                             | `Shuffler` 类，基于计数器的确定性 shuffle seed                                                                              |

### 3.2 确定性训练配置

**Megatron 路径** (`megatron_utils/deterministic.py:12-39`):

```
model_config.deterministic_mode = True
model_config.cross_entropy_loss_fusion = False
model_config.bias_dropout_fusion = False
NVTE_ALLOW_NONDETERMINISTIC_ALGO = "0"
NCCL_ALGO = "Ring"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
```

**Archon 路径** (`archon_utils.py:213-249`): 在 Megatron 基础上额外添加：

```
TORCH_COMPILE_DETERMINISTIC = "1"  (当 torch.compile 激活时)
ac_config.preserve_rng_state = True
```

### 3.3 Mixed Precision 架构

**FSDP 三 dtype 解耦** (`fsdp_utils/parallel.py:384-396`):

```python
MixedPrecisionPolicy(
    param_dtype=getattr(torch, config.dtype),       # compute: bfloat16
    reduce_dtype=getattr(torch, config.grad_reduce_dtype),  # reduce: float32
    cast_forward_inputs=True,
)
# 参数存储在 optimizer_dtype (float32) → FSDP 自动 cast 到 param_dtype
```

**FP32 关键路径**:

- Gradient norm: `fsdp_utils/grad.py:89-176` — `get_grad_norm_fp32` 全程 FP32
- Loss weight: `core/train_engine.py:59` — `.float()` 显式转换
- Log-softmax: `functional/vocab_parallel.py:18` — `logits.float()` 防止 BF16 overflow
- PPO log-ratio: `functional/functional.py:231` — `.float()` 显式 upcast

**MoE Router FP32** (`moe/router.py:24,40`): `@torch.amp.custom_fwd/custom_bwd` 确保 gate
GEMM 在 FP32 下执行。

### 3.4 NaN/Inf 防护

| 检查点              | 文件:行号                        | 行为                                 |
| ------------------- | -------------------------------- | ------------------------------------ |
| PPO prox_logp       | `actor.py:760-764`               | 硬 RuntimeError                      |
| PPO log-ratio       | `functional.py:233`              | `torch.where(isfinite, x, 0.0)` 清洗 |
| KL 估计器           | `data.py:1654-1655`              | `clamp(min=-10, max=10)`             |
| Tree attention      | `triton_kernel.py:304`           | 避免 `-inf - (-inf) = NaN`           |
| Stats tracker       | `stats_tracker.py:274,289`       | `torch.isinf(x)` 检查                |
| FP8 linear output   | `test_fp8_linear.py:122-164`     | 测试断言 no NaN/Inf                  |
| Distributed forward | `torchrun/dist_utils.py:183-185` | `validate_no_nan` 辅助函数           |

### 3.5 Checkpoint 精度一致性

**FSDP DCP** (`fsdp_engine.py`, `fsdp_utils/checkpoint.py`):

- PyTorch DCP (Distributed Checkpoint Protocol) save/load
- `_cast_to_compute_dtype` helper — FP32 存储参数 → compute dtype 再导出/广播
- RNG state 保存: `megatron_utils/checkpointer.py:184-226` — snapshot
  random/numpy/torch/CUDA RNG states

**Archon Checkpoint Round-trip** (`run_checkpoint_tests.py:260-314`):

```python
# save → load → forward → allclose 验证
allclose = torch.allclose(output_before, output_after, rtol=1e-4, atol=1e-4)
if max_diff > 1e-3:
    success = False  # 硬失败
```

### 3.6 测试体系组织

```
tests/
├── test_cuda_deterministic.py        # bit-identical 确定性回归 (rtol=0, atol=0)
├── test_packed_vs_padded_consistency.py  # packed vs padded forward 一致性
├── test_fsdp_optimizer_dtype.py      # FP32 master weight 回归 (#1292)
├── test_fsdp_grad.py                 # 梯度 clipping 正确性
├── test_fsdp_dcp.py                  # DCP checkpoint 正确性
├── test_vocab_parallel.py            # vocab-parallel 数值稳定性
├── experimental/archon/
│   ├── test_forward.py               # Archon vs HF forward
│   ├── test_grpo.py                  # GRPO loss Archon vs FSDP 数值一致性
│   ├── test_hf_parity_qwen2.py       # 逐层 Archon vs HF parity
│   ├── test_hf_parity_qwen3.py
│   ├── test_hf_parity_qwen3_5.py
│   ├── test_hf_parity_qwen3_moe.py
│   ├── test_hf_parity_qwen3_5_moe.py
│   ├── test_moe_hf_parity.py         # MoE 深度诊断
│   ├── test_moe_common.py            # MoE router FP32 + numerical parity
│   ├── test_checkpoint_e2e.py        # Checkpoint round-trip
│   ├── test_distributed_dp.py        # DP correctness
│   ├── test_distributed_tp.py        # TP correctness
│   ├── test_distributed_pp.py        # PP correctness (含 gradient verify)
│   ├── test_distributed_cp.py        # CP/Ulysses correctness
│   ├── test_distributed_ep.py        # EP/ETP correctness
│   ├── test_weight_sync.py           # 权重 completeness + value matching
│   ├── test_state_dict_adapter.py    # state dict 转换一致性
│   ├── fp8/                          # FP8 专项
│   │   ├── test_fp8_linear.py        # FP8 linear NaN/Inf + numerical
│   │   ├── test_dequant.py           # FP8 dequant correctness
│   │   ├── test_moe_dispatch.py      # FP8 MoE dispatch
│   │   └── torchrun/                 # FP8 分布式测试
│   └── torchrun/
│       ├── run_checkpoint_tests.py   # 分布式 checkpoint round-trip
│       ├── run_pp_gradient_verify.py # PP vs non-PP gradient 对比
│       ├── run_tp_forward.py         # TP forward 验证
│       ├── run_cp_forward.py         # CP forward 验证
│       ├── run_ep_tests.py           # EP 测试
│       ├── run_vs_fsdp.py            # Archon vs FSDP 对比
│       └── dist_utils.py             # golden model 创建 + verify_outputs_match
├── sft/                                  # SFT golden loss regression
│   ├── test_sft.py                       # 16步SFT训练golden loss对比 (rel=1.6%, abs=1e-5)
│   ├── ref_losses_fsdp.json              # FSDP引擎 reference losses
│   ├── ref_losses_megatron.json          # Megatron引擎 reference losses
│   └── ref_losses_archon.json            # Archon引擎 reference losses
├── fp8/                              # Megatron FP8 对比
│   ├── test_fp8_bf16_comparison.py   # FP8 vs BF16 全链路对比
│   ├── comparison_utils.py           # 对比工具 (cos_sim, grouped by op type)
│   └── model_hooks.py               # activation/gradient hook 系统
└── torchrun/
    ├── run_fsdp_optimizer_dtype.py   # optimizer dtype 回归
    ├── run_fsdp_dcp_distributed.py   # 分布式 DCP
    ├── run_vocab_parallel.py         # vocab-parallel 数值稳定性
    └── run_fsdp_ulysses_*.py         # Ulysses correctness
```

### 3.7 Tolerance Tiers 设计

项目定义了分级 tolerance 体系（`test_qwen3_5.py:97-107`）：

| Tier         | rtol | atol | 用途                                    |
| ------------ | ---- | ---- | --------------------------------------- |
| EXACT        | 1e-6 | 1e-6 | 数学等价操作（如 RoPE 非旋转维度）      |
| TIGHT        | 1e-5 | 1e-5 | 同架构同权重 forward                    |
| KERNEL       | 1e-4 | 1e-4 | 不同 kernel 实现（如 fused vs unfused） |
| RELAXED      | 1e-3 | 1e-3 | BF16 累积误差                           |
| E2E          | 1e-2 | 5e-2 | 端到端多层累积                          |
| KERNEL_BF16  | 1e-2 | 1e-2 | BF16 kernel 对比                        |
| RELAXED_BF16 | 5e-2 | 5e-2 | BF16 宽松对比                           |

FP8 测试使用 cosine similarity threshold: forward 0.99, gradient 0.94。

______________________________________________________________________

## 4. Commit / PR / Issue 历史演进时间线

### 精度对齐能力演进时间线

#### Phase 1: 基础精度基础设施 (2025-02 ~ 2025-05)

**时间**: 2025-02-27 **commit**: `db3e3ded` / `2436ce51` **PR**: PR #1 **涉及文件**:
`realhf/api/quickstart/model.py` **问题背景**: micro-batch 间 loss scale 不归一化，导致不同
micro-batch size 下 loss 不一致 **修改内容**: normalize loss scale by tokens across micro
batches **新增测试**: 无 **对精度对齐体系的启发**: loss 归一化是训练精度对齐的第一步；不同 micro-batch 策略必须产生等价的梯度

______________________________________________________________________

**时间**: 2025-02-28 **commit**: `e7c4a49a` / `22f357b3` **PR**: PR #4 **涉及文件**:
`realhf/api/core/data_api.py`, `realhf/system/model_worker.py`, `realhf/base/seeding.py`
**问题背景**: dataloader 不 shuffle，random seed 不正确，导致数据顺序不确定 **修改内容**: 修复 dataloader shuffle
和 random seed 问题，实现 per-epoch per-rank 不同 seed **新增测试**: 无 **对精度对齐体系的启发**:
数据顺序确定性是训练可复现性的基础；seed 必须同时区分 epoch 和 rank

______________________________________________________________________

**时间**: 2025-03-12 **commit**: `fb23009e` **PR**: PR #27 **涉及文件**:
`realhf/impl/model/backend/megatron.py` **问题背景**: BF16 训练路径不工作 **修改内容**: support bf16
training **新增测试**: 无 **对精度对齐体系的启发**: mixed precision 是精度对齐的核心关注点

______________________________________________________________________

**时间**: 2025-03-17 **commit**: `0b15ead8` **PR**: PR #34 **涉及文件**:
`realhf/impl/model/backend/megatron.py` (1 行) **问题背景**: FP16 训练失败 **修改内容**: fix the fp16
training issue **新增测试**: 无 **对精度对齐体系的启发**: 不同精度路径都需要验证

______________________________________________________________________

#### Phase 2: 并行策略 correctness (2025-08 ~ 2025-11)

**时间**: 2025-09-12 **commit**: `9727753d` **PR**: #320 **涉及文件**:
`areal/engine/fsdp_engine.py`, `areal/utils/fsdp.py`,
`areal/utils/multi_tensor_apply.py` **问题背景**: FSDP engine 中 gradient norm clipping
不正确；TP mesh 下梯度被重复计算 **修改内容**: 修复 grad norm clipping，添加 TP duplicate filtering，添加
`multi_tensor_applier` 回退实现 **新增测试**: 无显式测试，但修复了 `test_fsdp_grad.py` 的基础 **对精度对齐体系的启发**:
**gradient norm 在分布式环境下容易出错** — TP 复制参数的梯度必须去重，否则 clip 阈值偏大

______________________________________________________________________

**时间**: 2025-09-14 **commit**: `6b78aad2` **PR**: #335 **涉及文件**:
`areal/engine/fsdp_engine.py`, `areal/utils/fsdp.py` **问题背景**: Qwen3 TP 下 q/k norm
wrapping 不正确，gradient clipping 和 grad norm scale 有误 **修改内容**: 修复 TP q/k norm wrapping，修复
gradient clipping 和 grad norm 缩放 **新增测试**: 无 **对精度对齐体系的启发**: TP 下 Layer Norm 的 wrapping
方式直接影响数值结果

______________________________________________________________________

**时间**: 2025-09-15 **commit**: `d5b98d6d` **PR**: #340 **涉及文件**:
`areal/experimental/api/cli_args.py`, `areal/experimental/megatron_engine.py`,
`areal/experimental/utils/mcore/determinisitc.py` **问题背景**: MegatronEngine 缺少确定性训练选项
**修改内容**: 添加 `use_deterministic_algorithms` 配置，实现 NVTE/NCCL/CUBLAS/torch 确定性设置 **新增测试**:
无 **对精度对齐体系的启发**: **确定性训练是精度对齐的前提条件** — 必须控制 NCCL 算法、CUBLAS workspace、TE 随机性

______________________________________________________________________

**时间**: 2025-11-05 **commit**: `9fb41107` **PR**: #536 **涉及文件**:
`areal/tests/grpo/config.yaml`, `areal/tests/test_packed_vs_padded_consistency.py`
**问题背景**: CI 测试随机失败 **修改内容**: 修复 randomly failed CI tests，增加 packed_vs_padded 测试的确定性
**新增测试**: 添加 `set_random_seed` 到测试中 **影响范围**: CI 稳定性 **对精度对齐体系的启发**: 测试本身也需要确定性控制

______________________________________________________________________

#### Phase 3: FP8 + 高级精度基础设施 (2025-12 ~ 2026-03)

**时间**: 2025-12-31 **commit**: `89dda13a` **PR**: #758 **涉及文件**: 14 个文件，核心包括
`fp8_utils.py`, `fp8_kernels.py`, `comparison_utils.py`, `model_hooks.py`,
`test_fp8_bf16_comparison.py`, `test_fp8_conversion.py`, `test_fp8_rmsnorm.py` **问题背景**:
需要 FP8 训练支持以提高训练效率 **修改内容**: 完整 FP8 训练支持，包括量化/反量化、CLI 配置、模型加载/保存、Megatron 集成 **新增测试**:
大量 — FP8 转换、BF16 对比、gradient 正确性、RMSNorm 对比 **影响范围**: 全链路 — 从配置到训练到 checkpoint
**对精度对齐体系的启发**: **FP8 精度对比框架** 是最值得借鉴的设计 — cosine similarity + operation categorization
\+ activation hooks + conditional dump

______________________________________________________________________

**时间**: 2026-02-28 **commit**: `5b4ed832` / `f5cb33c4` **PR**: #943 **涉及文件**:
`cli_args.py`, `archon_engine.py`, `test_cuda_deterministic.py`, `rl_trainer.py`,
`sft_trainer.py`, debugging 文档 **问题背景**: Archon Engine 缺少确定性训练模式，MoE `_grouped_mm` 和
`torch.compile` 的确定性未验证 **修改内容**: 添加 `use_deterministic_algorithms`
配置（Archon），实现确定性环境设置，添加 CUDA 回归测试 **新增测试**: ✅ `test_cuda_deterministic.py` — 4 个
bit-identical 回归测试 (rtol=0, atol=0) **影响范围**: Archon Engine 全链路 **对精度对齐体系的启发**: **典型的
"bug → fix → regression test" 闭环** — 确定性问题直接转化为 bit-identical CI 测试

______________________________________________________________________

**时间**: 2026-03-08 **commit**: `4f5a2944` **PR**: #1009 **涉及文件**: `cli_args.py`,
`archon_engine.py`, `moe/args.py`, `moe/router.py`, `test_moe_common.py` **问题背景**: MoE
router gate GEMM 在 BF16 下数值不稳定，大 expert 数量下 routing 不精确 **修改内容**: 添加 `moe_router_dtype`
配置（默认 `"fp32"`），实现 `RouterGatingLinearFunction` (FP32 custom autograd) **新增测试**: ✅
`test_moe_common.py` — MoE router FP32 数值对比 **对精度对齐体系的启发**: **关键算子必须显式控制精度** — router
gate 是 MoE 训练稳定性的瓶颈

______________________________________________________________________

**时间**: 2026-02-28 **commit**: `055066a9` / `1fd9f949` **PR**: #940 **涉及文件**:
`moe/args.py`, `test_moe_args.py`, `test_moe_hf_parity.py` **问题背景**:
MoEArgs.score_before_experts 默认 True，但 HF 模型 (Mixtral, Qwen3-MoE, JetMoE) 都在 expert
computation 之后 apply scores，导致 HF parity 断裂 **修改内容**: 将 `score_before_experts` 默认值从 True
改为 False **新增测试**: ✅ `test_moe_hf_parity.py` — 1068 行 MoE HF parity 测试 **对精度对齐体系的启发**:
**模型行为必须与参考实现对齐** — 一个看似无害的默认值差异可以导致完全不同的训练结果

______________________________________________________________________

#### Phase 4: Master Weight + Grad 补偿 (2026-04 ~ 2026-05)

**时间**: 2026-04-29 **commit**: `5e9fb505` **PR**: #1273 **涉及文件**: `megatron_engine.py`,
`test_megatron_engine.py` **问题背景**: Megatron Core 的 pipeline schedule 内部将 loss 除以
`num_microbatches`，AReaL 已有全局归一化 `w_i / W_total`，导致 **梯度被额外缩小 N 倍**，optimizer 步长偏小
**修改内容**: 将 `loss_multiplier` 乘以 `len(mb_list)` 以抵消 Megatron 的 `/num_microbatches`
**新增测试**: ✅ `test_qwen3_grad_norm_mb_invariance` — 不同 `max_tokens_per_mb` 下 grad_norm
一致性 (within 1e-3) **对精度对齐体系的启发**: **框架间的 loss normalization 语义差异是精度陷阱** — 必须端到端验证
gradient magnitude

______________________________________________________________________

**时间**: 2026-05-27 **commit**: `237a49f6` / `7f72f4c5` **PR**: #1292 → #1369 **涉及文件**:
`cli_args.py`, `fsdp_engine.py`, `test_fsdp_optimizer_dtype.py`,
`run_fsdp_optimizer_dtype.py` **问题背景**: `torch.optim.AdamW` 在 `actor.dtype=bfloat16`
时静默继承 BF16 dtype，导致 optimizer state 精度丢失，SFT 后期 loss 平台化（比 DS-Z3 / Megatron 高 ~3x）
**修改内容**: 解耦 parameter storage dtype 和 compute dtype — 添加 `optimizer_dtype`（默认
`float32`），模型加载时使用 optimizer_dtype，FSDP2 `MixedPrecisionPolicy` 仅管 compute，导出/广播时 cast 回
compute dtype **新增测试**: ✅ 5 个 dtype invariant 回归测试（1/2 GPU） **对精度对齐体系的启发**: **这是最典型的精度对齐
case** — 一个隐式 dtype 继承导致了 3x 的 loss 差异，修复后需要 5 个 invariant 测试固化

______________________________________________________________________

## 5. 典型精度问题案例复盘

### Case 1: FP32 Master Weights 缺失导致 SFT Loss 平台化

**commit**: `237a49f6` (#1292)

**现象**: 使用 FSDP + BF16 训练 SFT 时，后期 loss 收敛到约为 DeepSpeed ZeRO-3 / Megatron 的 3 倍高。

**根因**: `torch.optim.AdamW` 在创建时从 `model.parameters()` 继承了 BF16 dtype，导致 `exp_avg` 和
`exp_avg_sq` 也是 BF16。BF16 的 7-bit mantissa 无法精确表示 Adam 所需的微小更新量，随着训练推进，优化器状态精度损失累积。

**如何定位**: 通过对比 FSDP 与 DS-Z3 / Megatron（两者都有 FP32 master weight 路径）的 loss curve 发现差异。检查
optimizer state dtype 确认根因。

**如何修复**: 引入 `optimizer_dtype` 配置（默认 `float32`），在模型加载阶段使用 FP32 创建参数，FSDP2 的
`MixedPrecisionPolicy` 仅在 forward/backward 时 cast 到 compute dtype。导出和权重广播时通过
`_cast_to_compute_dtype` 转回。

**新增测试**: 5 个 dtype invariant 测试：(1) param storage dtype (2) AdamW state dtype (3) FSDP
forward cast (4) HF export dtype (5) xccl cast。

**借鉴**: optimizer state dtype 必须与 compute dtype 解耦，并且需要端到端的 dtype invariant 测试来固化。

______________________________________________________________________

### Case 2: Pipeline Schedule 额外除以 num_microbatches

**commit**: `5e9fb505` (#1273)

**现象**: MegatronEngine 报告的 grad_norm 比 FSDPEngine 在相同数据上小约 N 倍（N = micro-batch 数量）。

**根因**: Megatron Core 的 `_forward_step_helper` 在 `(loss, {})` 返回路径上将 loss 除以
`num_microbatches`。AReaL 已经通过 `w_i / W_total` 做了全局归一化，所以这个额外的除法使每个梯度（以及 optimizer
step）缩小了 `num_microbatches` 倍。

**如何定位**: 对比不同 `max_tokens_per_mb` 下的 grad_norm，发现两者之比恰好等于 micro-batch 数之比。

**如何修复**: 在 `MegatronEngine.train_batch` 中将 `loss_multiplier` 乘以 `len(mb_list)` 以抵消。

**新增测试**: `test_qwen3_grad_norm_mb_invariance` — 在不同 micro-batch 数下验证 grad_norm 一致性
(within 1e-3)。

**借鉴**: 当集成第三方框架时，必须验证 loss normalization 语义的端到端一致性。

______________________________________________________________________

### Case 3: MoE score_before_experts 默认值与 HF 不一致

**commit**: `055066a9` (#940)

**现象**: Archon MoE 模型加载 HF checkpoint 后，forward 输出与 HF 参考实现不一致。

**根因**: `MoEArgs.score_before_experts` 默认 True，在 expert computation 之前 apply router
scores。但 HF 模型 (Mixtral, Qwen3-MoE, JetMoE, GraniteMoe) 都在 expert 之后 apply scores。对于非线性
expert (SwiGLU)，这两个顺序产生不同结果。

**如何定位**: 编写 1068 行的 `test_moe_hf_parity.py`，逐层对比 Archon 和 HF 的 activation。

**如何修复**: 将默认值从 True 改为 False。

**新增测试**: ✅ `test_moe_hf_parity.py` — 完整的 MoE HF parity 诊断测试。

**借鉴**: 模型行为的"等价性"不能靠假设 — 必须有逐层 parity 测试来验证。

______________________________________________________________________

### Case 4: FSDP TP 下 Gradient Norm 重复计算

**commit**: `9727753d` (#320)

**现象**: 使用 TP 时，gradient norm 偏大，导致 clipping 过于激进。

**根因**: TP 下某些参数被复制到多个 rank（如 Replicate placement），但 gradient norm
计算没有去重，导致这些参数的梯度被重复计算。

**如何修复**: 添加 `is_param_not_tensor_parallel_duplicate` 过滤器（`fsdp_utils/grad.py:62-76`），只有
non-replicate TP placement 的参数才参与 norm 计算。

**借鉴**: 分布式 gradient norm 需要感知所有并行维度的 placement 语义。

______________________________________________________________________

### Case 5: Dataloader Shuffle 和 Seed 不正确

**commit**: `e7c4a49a` (PR #4), `22f357b3`

**现象**: 训练数据顺序不确定，不同 run 产生不同结果。

**根因**: dataloader 没有正确 shuffle，random seed 设置有误，导致 per-epoch 和 per-rank 的数据顺序不可复现。

**如何修复**: 实现 per-epoch per-rank 不同 seed 的 shuffle 机制，创建 `seeding.py` 模块。

**借鉴**: 数据顺序确定性是训练可复现性的基础，但容易被忽视。

______________________________________________________________________

### Case 6: MoE Router 在 BF16 下数值不稳定

**commit**: `4f5a2944` (#1009)

**现象**: 大 expert 数量的 MoE 模型训练时，router 决策不稳定。

**根因**: router gate GEMM 在 BF16 下执行，BF16 的精度不足以区分相近的 expert scores。

**如何修复**: 添加 `moe_router_dtype` 配置（默认 `"fp32"`），实现 `RouterGatingLinearFunction` — 一个
custom `torch.autograd.Function`，在 FP32 下执行 gate GEMM。

**新增测试**: `test_moe_common.py` — 验证 FP32 router 的数值正确性。

**借鉴**: 精度敏感的算子（routing、loss computation、normalization）必须显式控制 dtype。

______________________________________________________________________

### Case 7: FP8 Blockwise 训练在 TP + MoE 下不正确

**commit**: `0ee85625` (#1118)

**现象**: FP8 blockwise 训练在 TP > 1 或 MoE 场景下产生错误结果或崩溃。

**根因**: (1) GroupedExperts 的 w1/w2/w3 shapes 在 post-parallelism check 中被遗漏 (2) dense FP8
linear forward 没有正确处理 DTensor (3) FP8 dequant 对 Shard(1) 的 checkpoint 处理不当。

**如何修复**: 扩展 shard alignment validation，在 FP8 linear forward 中转换 DTensor 到 local
tensor，限制 FP8 dequant 只处理 `float8_e4m3fn`，对 Shard(1) FP8 checkpoint 提前失败。

**借鉴**: FP8 与并行策略的交叉场景是精度 bug 高发区。

______________________________________________________________________

### Case 8: Data Service Seed 在 Request 级别重复初始化

**commit**: `ad622efd` (#1210)

**现象**: 数据服务在多数据集场景下 shuffle 行为不一致。

**根因**: seed 在每个 request 中重新设置，干扰了跨 dataset 的 shuffle 状态。

**如何修复**: 将 seed 移到 worker-level config，在 worker 启动时只设置一次。

**借鉴**: seed 应该在进程级别设置一次，不应在 request 级别重复初始化。

______________________________________________________________________

## 6. 三阶段精度对齐流程映射

### 阶段一：训练前准备与基础对齐

| 检查项              | 状态    | 源码证据                                                                                     |
| ------------------- | ------- | -------------------------------------------------------------------------------------------- |
| 配置一致性          | ⚠️ 部分 | `cli_args.py` `__post_init__` 做字段验证，但无配置 diff/snapshot 工具                        |
| 环境一致性          | ⚠️ 部分 | 有 `validate_installation.py` 和 `validate_docker_installation.py`，但不验证精度相关环境变量 |
| seed/RNG            | ✅ 完整 | `seeding.py` + deterministic mode + per-role/per-rank SHA256 seed                            |
| 数据顺序            | ✅ 完整 | `Shuffler` + per-epoch per-rank seed + `StatefulDataLoader` state 保存                       |
| 模型结构            | ✅ 完整 | HF parity tests 验证 Archon vs HF 模型结构等价                                               |
| 初始化权重          | ✅ 完整 | memory_efficient_load (rank 0 → broadcast)，TP-aware seed                                    |
| dropout/正则        | ✅ 完整 | `disable_dropout_in_model()` 递归关闭所有 Dropout                                            |
| deterministic flags | ✅ 完整 | CUBLAS/NCCL/TE/compile/AC RNG 全覆盖                                                         |

**阶段一评价**: 较完整。缺少配置快照和 diff 工具，但核心要素（seed、数据、权重、确定性标志）都已覆盖。

### 阶段二：单卡/单步对齐

| 检查项                 | 状态      | 源码证据                                                                           |
| ---------------------- | --------- | ---------------------------------------------------------------------------------- |
| forward loss           | ✅ 完整   | `test_grpo.py`, `test_forward.py` — Archon vs FSDP vs HF                           |
| activation             | ✅ 较完整 | `test_hf_parity_*.py` 逐层 hooks 捕获对比, `fp8/model_hooks.py`                    |
| backward gradient      | ✅ 较完整 | `test_grpo.py:test_logprobs_gradient_flow`, `fp8/model_hooks.py:collect_gradients` |
| optimizer update       | ✅ 完整   | `test_fsdp_optimizer_dtype.py` — 5 dtype invariant                                 |
| scheduler              | ❌ 缺失   | 无 LR scheduler 数值一致性测试                                                     |
| loss scaling           | ⚠️ 部分   | Megatron 有内置 loss scaler，FSDP/Archon 无（依赖 BF16）                           |
| tensor dump            | ⚠️ 部分   | FP8 测试有 conditional save，但无通用 dump 框架                                    |
| operator-level compare | ✅ 完整   | FP8 `comparison_utils.py` 按 op type 分组对比                                      |

**阶段二评价**: 较完整。Archon Engine 的 HF parity + FP8 对比体系是亮点。缺少 LR scheduler 验证和通用 tensor
dump。

### 阶段三：多步/分布式/长稳对齐

| 检查项                    | 状态                  | 源码证据                                                                    |
| ------------------------- | --------------------- | --------------------------------------------------------------------------- |
| loss curve                | ✅ SFT 完整 / RL 缺失 | `tests/sft/test_sft.py` + `ref_losses_*.json` (16 步 golden loss, rel=1.6%) |
| checkpoint resume         | ✅ 完整               | `run_checkpoint_tests.py` save→load→forward match                           |
| DP correctness            | ✅ 完整               | `test_distributed_dp.py`                                                    |
| TP correctness            | ✅ 完整               | `test_distributed_tp.py`, `run_tp_forward.py`                               |
| PP correctness            | ✅ 完整               | `test_distributed_pp.py`, `run_pp_gradient_verify.py`                       |
| SP/CP correctness         | ✅ 完整               | `test_distributed_cp.py`, `run_cp_forward.py`                               |
| EP/MoE correctness        | ✅ 完整               | `test_distributed_ep.py`, `test_moe_hf_parity.py`                           |
| gradient accumulation     | ✅ 完整               | `train_batch` 中 sequential micro-batch backward                            |
| communication collectives | ⚠️ 间接               | `warmup_process_groups` + NCCL_ALGO=Ring，但无独立验证                      |
| mixed precision stability | ✅ 完整               | FP8/BF16 对比，FP32 master weights，FP32 grad reduce                        |
| NaN/Inf monitoring        | ✅ 较完整             | actor.py 硬 error，functional.py sanitization                               |
| CI regression             | ⚠️ 部分               | PR CI 有 GPU unit tests，nightly 为占位符                                   |

**阶段三评价**: 分布式 correctness 测试是项目最大亮点（5 种并行策略全覆盖），但缺少 loss curve regression 和 nightly
长稳测试。

______________________________________________________________________

## 7. 可复用设计模式

### 7.1 SHA256-based Per-Role Seed 系统

**设计目标**: 确保不同角色（trainer/data_worker/proxy/inference）和不同 rank 使用不同但确定的 seed

**源码位置**: `areal/utils/seeding.py`

**工作流程**:

```
base_seed = config.seed (全局)
role_key = f"trainer{rank}" | f"data_worker_{rank}" | f"proxy{rank}"
final_seed = base_seed + SHA256(role_key) & 0xFFFFFFFF
→ 设置 PYTHONHASHSEED, transformers, random, numpy, torch, CUDA
```

**优点**: 避免不同角色 seed 碰撞；SHA256 确保均匀分布；key-based 设计方便扩展

**局限**: 不感知 TP/PP/EP 并行维度（这些由 Megatron 的 `model_parallel_cuda_manual_seed` 处理）

**迁移建议**: 直接复用；建议增加 `get_rng_state_dict()` / `set_rng_state_dict()` 用于 checkpoint 中的 RNG
state 保存

### 7.2 Tolerance Tier 体系

**设计目标**: 为不同精度级别的对比提供标准化 tolerance

**源码位置**: `tests/experimental/archon/test_qwen3_5.py:97-107`

**工作流程**: 定义 EXACT/TIGHT/KERNEL/RELAXED/E2E 等级别，每级明确 rtol/atol

**优点**: 避免每个测试随意选择 tolerance；有明确的精度预期文档

**局限**: BF16 tier 可能在不同硬件上需要调整

**迁移建议**: 定义项目级 tolerance tier 标准，与 `torch.testing.assert_close` 配合使用

### 7.3 FP8/BF16 逐层 Cosine Similarity 对比框架

**设计目标**: 系统性对比 FP8 和 BF16 模型在每一层每一类操作上的数值差异

**源码位置**: `tests/fp8/comparison_utils.py`, `tests/fp8/model_hooks.py`

**工作流程**:

1. `model_hooks.py` 注册 forward/backward hooks 到每个 attention/MLP/LayerNorm 模块
1. 收集 `activations` + `gradients` + `output_gradients` 字典
1. `compare_tensors_dict()` 按 operation type 分组，计算 max_diff/mean_diff/cosine_similarity
1. `find_problematic_operations()` 找出 cosine similarity 低于阈值的算子
1. 可选 `save_data=True` dump tensor 到磁盘

**优点**:

- Operation categorization 允许快速定位哪类算子是精度瓶颈
- Cosine similarity 比 allclose 更适合高维 tensor 对比
- Conditional dump 避免常规运行时的 I/O 开销

**局限**: 仅限 FP8 vs BF16 场景，未推广到通用 precision comparison

**迁移建议**: 将 hook 系统和 compare_tensors_dict 提取为通用工具，支持任意两个模型/配置的逐层对比

### 7.4 多引擎 Correctness Matrix

**设计目标**: 验证 Archon vs FSDP vs HuggingFace 在相同权重下的数值一致性

**源码位置**: `tests/experimental/archon/utils.py:DualEngineFixture`, `test_grpo.py`,
`test_forward.py`

**工作流程**:

1. `DualEngineFixture` 同时初始化 Archon 和 FSDP Engine（共享权重）
1. 对相同输入运行 forward，用 `ComparisonMetrics` 对比 logits/logprobs/loss/gradients
1. 对比 importance ratio（PPO 核心指标，应接近 1.0）
1. 检查 gradient NaN/Inf

**优点**: 直接验证不同引擎的训练等价性，不依赖 golden baseline

**局限**: 需要两个引擎同时在 GPU 上，内存需求高

**迁移建议**: 复用 DualEngineFixture 模式，适配到自己的引擎组合

### 7.5 Checkpoint Save-Load-Forward-Match 测试

**设计目标**: 验证 checkpoint round-trip 后 forward 输出不变

**源码位置**: `tests/experimental/archon/torchrun/run_checkpoint_tests.py`

**工作流程**:

```python
# 1. 初始化模型 → forward → 记录 output_before
# 2. save(checkpoint)
# 3. 清理模型状态
# 4. load(checkpoint) → forward → 记录 output_after
# 5. assert allclose(output_before, output_after, rtol=1e-4, atol=1e-4)
# 6. 如果 max_diff > 1e-3 则硬失败
```

**优点**: 简单直接地验证 checkpoint 精度无损

**迁移建议**: 直接复用，扩展到验证 optimizer state、LR scheduler state、RNG state 的 round-trip

### 7.6 确定性训练配置模式

**设计目标**: 一键启用完整的确定性训练环境

**源码位置**: `megatron_utils/deterministic.py` + `archon_utils.py:213-249`

**工作流程**: 单个 `use_deterministic_algorithms=True` 触发：

```
torch.use_deterministic_algorithms(True, warn_only=True)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
NCCL_ALGO = "Ring"
NVTE_ALLOW_NONDETERMINISTIC_ALGO = "0"
TORCH_COMPILE_DETERMINISTIC = "1"
model_config.deterministic_mode = True
model_config.cross_entropy_loss_fusion = False
model_config.bias_dropout_fusion = False
ac_config.preserve_rng_state = True
```

**优点**: 全面覆盖 PyTorch/CUDA/NCCL/TE/compile/AC 六个层面的确定性控制

**局限**: 性能有损（Ring all-reduce 比 NVLS Tree 慢，deterministic matmul 需要额外 workspace）

**迁移建议**: 直接复用这个 checklist，确保你的框架覆盖所有六个层面

### 7.7 SFT Golden Loss JSON Regression 机制

**设计目标**: 端到端验证 SFT 训练的 loss 轨迹与 golden baseline 一致

**源码位置**: `tests/sft/test_sft.py`, `tests/sft/ref_losses_*.json`

**工作流程**:

1. 使用小模型（Qwen3-0.6B）+ 固定配置在 subprocess 中运行 16 步 SFT 训练
1. 收集每步 loss 到 `losses.json`
1. 与预存的 `ref_losses_{backend}.json` 逐步对比
1. 使用 `pytest.approx(ref_loss, rel=1.6e-2, abs=1e-5)` — 1.6% 相对 + 1e-5 绝对 tolerance
1. 三引擎（FSDP/Megatron/Archon）各有独立的 reference losses

**优点**:

- 端到端验证：从配置→数据加载→forward→backward→optimizer step 全链路
- 多引擎覆盖：同一测试框架验证三种训练后端
- Tolerance 明确合理：1.6% 相对容差足以捕获精度回归，同时容忍硬件间的微小差异

**局限**:

- 仅覆盖 SFT，未覆盖 RL (PPO/GRPO)
- `tests/sft/` 不在 `tests/test_*.py` glob 中，需确保 CI 显式包含
- 16 步较短，可能无法捕获后期才出现的精度问题

**迁移建议**: 直接复用此模式，为每种训练模式（SFT/PPO/GRPO/DPO）和每种引擎维护独立的 golden loss JSON

### 7.8 PP vs Non-PP Gradient 对比测试

**设计目标**: 验证 pipeline parallelism 不改变梯度值

**源码位置**: `tests/experimental/archon/torchrun/run_pp_gradient_verify.py`

**工作流程**:

1. 创建简单模型（SimpleBlock 堆叠）
1. 相同 seed 初始化两份：一份 non-PP，一份 PP (2-stage)
1. 相同输入 forward + backward
1. 收集所有参数的 gradient，对比（scatter 到 PP stages 后 gather 回来）

**优点**: 直接验证 PP 不引入数值误差

**迁移建议**: 为每种并行策略编写类似的 "parallel vs non-parallel gradient match" 测试

______________________________________________________________________

## 8. 缺口分析与改造建议

### P0: 必须补齐

#### 8.1 RL (PPO/GRPO) Golden Loss Regression

**问题**: SFT 已有 16 步 golden loss regression (`tests/sft/test_sft.py`, 覆盖
FSDP/Megatron/Archon 三引擎，tolerance rel=1.6%), 但 RL 训练（PPO/GRPO）尚无对应的 golden loss
regression。RL 训练链路更复杂（涉及 rollout、reward、advantage 计算），是精度 bug 高发区。

**为什么重要**: RL 训练的 loss 行为比 SFT 更复杂，更容易受到数值精度问题影响。

**当前部分实现**: `tests/sft/test_sft.py` + `ref_losses_*.json` 已为 SFT 提供了完整模板。

**建议设计**:

1. 复用 SFT golden loss 框架，添加 `tests/grpo/test_grpo_regression.py`
1. 使用小模型在固定 seed + 固定 mock rollout 数据上训练 10-20 步 GRPO
1. 记录每步 loss 为 JSON golden baseline
1. Threshold 建议：GRPO loss 偏差 \< 5%（RL 天然 variance 更大）

**涉及模块**: `tests/grpo/`, 参照 `tests/sft/` 模板

**预期收益**: RL 训练链路的数值行为变更也能被自动捕获

#### 8.2 Nightly CI 实质化

**问题**: `nightly.yml` 中的测试是占位符 (`Dummy test (placeholder)`, `sleep 10`)

**为什么重要**: 长时间训练回归（如 loss 漂移、NaN 出现在第 100 步后）只能在 nightly 测试中发现

**建议设计**: 在 nightly CI 中运行：

1. 小模型 SFT 100 步 golden loss 回归
1. 小模型 GRPO 50 步 golden loss 回归
1. 确定性模式下的 bit-identical 多步训练
1. Checkpoint resume 后 loss 连续性验证

### P1: 强烈建议补齐

#### 8.3 通用 Tensor Dump 框架

**问题**: activation/gradient dump 能力分散在 FP8 测试和 HF parity 测试中，不是通用工具

**建议设计**:

1. 提供 `TensorDumper` context manager，注册 forward/backward hooks
1. 支持按 layer/module/step 选择性 dump
1. 提供 `TensorComparer` 读取两个 dump 进行逐层对比
1. 集成到 training loop 中（如 `dump_steps=[1, 10, 100]`）

**涉及模块**: `areal/utils/debug/`, 扩展 `fp8/model_hooks.py` 的 hook 系统

#### 8.4 TF32 显式控制

**问题**: 无显式 `torch.backends.cuda.matmul.allow_tf32` 和 `cudnn.allow_tf32` 设置。在 Ampere+
GPU 上，PyTorch 默认启用 TF32，这会引入约 1e-3 级别的数值差异。

**建议设计**: 在配置中添加 `allow_tf32: bool` 字段，默认 `True`（性能）；确定性模式下自动设为 `False`

#### 8.5 LR Scheduler 数值一致性测试

**问题**: 无 LR scheduler 的数值一致性测试。不同 scheduler 实现或配置变更可能改变 learning rate 曲线。

**建议设计**: 测试 warmup + decay 曲线的每一步 LR 值与预期值的一致性

#### 8.6 配置快照与 Diff 工具

**问题**: 无法自动检测两次训练之间的配置差异（哪些超参变了）

**建议设计**: 训练开始时 dump 完整配置 JSON，提供 `config_diff(run1, run2)` 工具

### P2: 长期优化项

#### 8.7 自动化二分定位工具

**问题**: 当发现 loss 回归时，无工具自动 bisect 到引入问题的 commit

**建议设计**: 结合 golden loss regression + `git bisect`，自动化定位引入精度回归的 commit

#### 8.8 跨硬件精度对齐验证

**问题**: 无 A100 vs H100 vs Ascend 的精度对齐验证

**建议设计**: 在不同硬件上运行相同 golden test，对比 loss curve 偏差

#### 8.9 Communication Collective 独立验证

**问题**: 无独立的 all-reduce/reduce-scatter/all-gather 精度验证

**建议设计**: 在不同 group size 和 dtype 下验证 collective 操作的数值一致性

______________________________________________________________________

## 9. 推荐学习路线

### 第 1 步：读文档

1. `CLAUDE.md` — 项目概览和命令
1. `docs/tutorial/gsm8k_grpo.md` — 训练流程架构
1. `docs/best_practices/debugging.md` — 调试指南
1. `docs/tutorial/archon.md` — Archon Engine 教程（含 deterministic mode 文档）

### 第 2 步：跑 Examples/Tests

1. `tests/test_cuda_deterministic.py` — 确定性回归测试（bit-identical）
1. `tests/test_packed_vs_padded_consistency.py` — 数值一致性基线
1. `tests/test_fsdp_optimizer_dtype.py` — FP32 master weight 回归
1. `tests/experimental/archon/test_grpo.py` — GRPO Archon vs FSDP 数值对比
1. `tests/experimental/archon/test_hf_parity_qwen3.py` — 逐层 HF parity
1. `tests/experimental/archon/test_forward.py` — Archon vs HF forward
1. `tests/fp8/test_fp8_bf16_comparison.py` — FP8 vs BF16 全链路对比

### 第 3 步：读源码文件

1. `areal/utils/seeding.py` — seed 系统
1. `areal/engine/megatron_utils/deterministic.py` — Megatron 确定性配置
1. `areal/experimental/engine/archon_utils.py:213-350` — Archon 确定性 + 配置准备
1. `areal/engine/fsdp_utils/grad.py` — FP32 gradient norm 全链路
1. `areal/engine/fsdp_utils/parallel.py:380-400` — MixedPrecisionPolicy 构建
1. `areal/engine/core/train_engine.py:30-110` — FP32 loss weight 计算
1. `areal/utils/functional/vocab_parallel.py` — logits FP32 upcast
1. `areal/utils/functional/functional.py:220-240` — PPO log-ratio FP32 upcast + sanitize
1. `areal/trainer/ppo/actor.py:750-766` — NaN/Inf 检查
1. `tests/fp8/comparison_utils.py` — 对比工具
1. `tests/fp8/model_hooks.py` — activation/gradient hook 系统
1. `tests/experimental/archon/utils.py` — ComparisonMetrics + DualEngineFixture
1. `tests/experimental/archon/torchrun/dist_utils.py` — golden model +
   verify_outputs_match

### 第 4 步：复现 Commit/PR 中的问题

1. `237a49f6` (#1292) — 用 BF16 optimizer_dtype 训练 SFT，观察 loss 平台化
1. `5e9fb505` (#1273) — 比较不同 micro-batch 数下的 grad_norm
1. `055066a9` (#940) — 比较 score_before_experts=True/False 的 forward 输出
1. `9727753d` (#320) — 在 TP 下观察 gradient norm 是否因去重而变化

### 第 5 步：抽象设计模式

1. Tolerance Tier 标准化
1. FP8/BF16 逐层对比框架
1. 多引擎 DualEngineFixture 模式
1. Checkpoint Round-trip Forward Match 模式
1. 确定性训练 Checklist
1. PP Gradient Verify 模式

### 第 6 步：迁移到自己的训练系统

1. 移植 seeding.py — per-role seed 系统
1. 移植 deterministic mode — 六层面确定性控制
1. 实现 FP32 关键路径 — grad norm、loss weight、log-softmax
1. 建立 tolerance tier 标准
1. 实现 activation/gradient hook + compare 工具
1. 建立 checkpoint round-trip test
1. 建立 golden loss regression CI（AReaL 自身尚未有，需自行设计）
1. 为每种并行策略建立 parallel vs non-parallel correctness test

______________________________________________________________________

## 10. 对我自研分布式训练系统的迁移建议

### 10.1 核心理念

AReaL 的精度对齐体系体现了三个核心理念：

1. **FP32 关键路径**: gradient norm、loss weight、log-softmax、PPO log-ratio — 这些路径必须在 FP32
   下执行，因为累积误差会直接影响训练收敛
1. **显式 dtype 控制**: compute/reduce/storage 三 dtype 解耦，MoE router 独立 dtype — 不信任任何隐式
   dtype 继承
1. **多引擎对比验证**: 不依赖 golden baseline，而是用多个引擎实现的一致性来交叉验证 — 这在缺乏标准答案时是唯一可靠的验证方式

### 10.2 迁移优先级

| 优先级 | 能力                                                | 迁移难度 | 预期收益                 |
| ------ | --------------------------------------------------- | -------- | ------------------------ |
| P0     | Per-role seed 系统                                  | 低       | 训练可复现性基础         |
| P0     | 确定性训练 Checklist (6 层面)                       | 低       | bit-identical 复现能力   |
| P0     | FP32 关键路径 (grad norm, loss weight, log-softmax) | 中       | 防止 BF16 精度损失       |
| P0     | Checkpoint round-trip test                          | 低       | 防止 checkpoint 精度丢失 |
| P1     | Tolerance Tier 标准                                 | 低       | 测试一致性和可读性       |
| P1     | 多并行策略 correctness matrix                       | 高       | 分布式训练正确性保证     |
| P1     | Activation/gradient hook 对比框架                   | 中       | 精度问题快速定位         |
| P1     | Golden loss regression CI                           | 中       | 自动检测训练效果退化     |
| P2     | FP8 逐层 cosine similarity 对比                     | 高       | FP8 训练精度保证         |
| P2     | 配置快照 + diff 工具                                | 低       | 训练对比分析             |
| P2     | 自动化 bisect 工具                                  | 中       | 快速定位精度回归         |

### 10.3 关键注意事项

1. **不要信任框架默认值**: AReaL 的 `optimizer_dtype=float32` (#1292) 和
   `score_before_experts=False` (#940) 案例表明，框架默认值经常与精度需求不一致
1. **loss normalization 必须端到端验证**: AReaL 的 `loss_multiplier * len(mb_list)` 补偿 (#1273)
   说明，框架间的 loss 语义差异可以导致梯度幅度完全错误
1. **确定性模式是调试前提**: 在 non-deterministic 模式下，精度问题和随机性噪声无法区分
1. **MoE 是精度 bug 高发区**: router dtype、expert ordering、score timing、FP8 + EP 交叉 —
   每个都可能引入数值差异

______________________________________________________________________

## Appendix A. 检索关键词与命令记录

### 源码搜索关键词（实际执行）

```bash
git grep -rn "allclose|assert_close" -- "*.py"
git grep -rn "rtol|atol" -- "*.py"
git grep -rn "deterministic|torch.use_deterministic_algorithms|CUBLAS_WORKSPACE_CONFIG|cudnn.deterministic" -- "*.py"
git grep -rn "set_seed|manual_seed|torch.manual_seed|np.random.seed|random.seed" -- "*.py"
git grep -rn "tf32|allow_tf32|matmul.*precision" -- "*.py"
git grep -rn "isnan|isinf|nan_to_num|check_nan|has_nan|has_inf" -- "*.py"
git grep -rn "grad_scaler|GradScaler|loss_scale|loss_scaling" -- "*.py"
git grep -rn "dump|snapshot|save.*tensor|save.*activation" -- "*.py"
```

### Git history 搜索关键词（实际执行）

```bash
git log --oneline --all --grep="precision"
git log --oneline --all --grep="accuracy"
git log --oneline --all --grep="determin"
git log --oneline --all --grep="golden"
git log --oneline --all --grep="loss"
git log --oneline --all --grep="gradient"
git log --oneline --all --grep="bf16|fp16|fp8"
git log --oneline --all --grep="checkpoint"
git log --oneline --all --grep="distributed"
git log --oneline --all --grep="tensor parallel|pipeline parallel|sequence parallel"
git log --oneline --all --grep="all_reduce|reduce_scatter|all_gather"
git log --oneline --all --grep="seed|rng|random"
git log --oneline --all --grep="nan|inf|overflow"
git log --oneline --all --grep="reproducib|numerical|stability"
git log --oneline --all --grep="optimizer state|master weight|fp32"
git log --oneline --all --grep="resume|recover"
```

### 实际检查的核心目录

- `areal/utils/seeding.py` ✅
- `areal/engine/megatron_utils/deterministic.py` ✅
- `areal/experimental/engine/archon_utils.py` ✅
- `areal/engine/fsdp_utils/grad.py` ✅
- `areal/engine/fsdp_utils/parallel.py` ✅
- `areal/engine/fsdp_utils/optimizer.py` ✅
- `areal/engine/core/train_engine.py` ✅
- `areal/engine/core/distributed.py` ✅
- `areal/engine/fsdp_engine.py` ✅
- `areal/engine/megatron_engine.py` ✅
- `areal/experimental/engine/archon_engine.py` ✅
- `areal/api/cli_args.py` ✅
- `areal/trainer/ppo/actor.py` ✅
- `areal/trainer/rl_trainer.py` ✅
- `areal/utils/functional/vocab_parallel.py` ✅
- `areal/utils/functional/functional.py` ✅
- `areal/utils/stats_tracker.py` ✅
- `areal/utils/recover.py` ✅
- `areal/experimental/models/archon/moe/router.py` ✅
- `areal/experimental/models/archon/moe/utils.py` ✅
- `areal/experimental/models/archon/activation_checkpoint.py` ✅
- `areal/engine/megatron_utils/checkpointer.py` ✅
- `areal/engine/megatron_utils/fp8/` ✅
- `tests/test_cuda_deterministic.py` ✅
- `tests/test_packed_vs_padded_consistency.py` ✅
- `tests/test_fsdp_optimizer_dtype.py` ✅
- `tests/experimental/archon/test_grpo.py` ✅
- `tests/experimental/archon/test_forward.py` ✅
- `tests/experimental/archon/test_hf_parity_qwen3.py` ✅
- `tests/experimental/archon/test_moe_hf_parity.py` ✅
- `tests/experimental/archon/test_checkpoint_e2e.py` ✅
- `tests/experimental/archon/test_distributed_*.py` ✅
- `tests/experimental/archon/test_qwen3_5.py` ✅
- `tests/experimental/archon/test_weight_sync.py` ✅
- `tests/experimental/archon/test_moe_common.py` ✅
- `tests/experimental/archon/utils.py` ✅
- `tests/experimental/archon/torchrun/dist_utils.py` ✅
- `tests/experimental/archon/torchrun/run_checkpoint_tests.py` ✅
- `tests/experimental/archon/torchrun/run_pp_gradient_verify.py` ✅
- `tests/experimental/archon/torchrun/run_tp_forward.py` ✅
- `tests/fp8/comparison_utils.py` ✅
- `tests/fp8/model_hooks.py` ✅
- `tests/fp8/test_fp8_bf16_comparison.py` ✅
- `.github/workflows/test-areal.yml` ✅
- `.github/workflows/nightly.yml` ✅

______________________________________________________________________

## Appendix B. 关键文件清单

### 精度控制核心

| 文件                                             | 功能                          | 精度相关度 |
| ------------------------------------------------ | ----------------------------- | ---------- |
| `areal/utils/seeding.py`                         | 全局 seed 系统                | ★★★★★      |
| `areal/engine/megatron_utils/deterministic.py`   | Megatron 确定性模式           | ★★★★★      |
| `areal/experimental/engine/archon_utils.py`      | Archon 确定性 + 配置          | ★★★★★      |
| `areal/engine/fsdp_utils/grad.py`                | FP32 gradient norm            | ★★★★★      |
| `areal/engine/fsdp_utils/parallel.py`            | MixedPrecisionPolicy          | ★★★★★      |
| `areal/engine/core/train_engine.py`              | FP32 loss weight              | ★★★★☆      |
| `areal/utils/functional/vocab_parallel.py`       | logits FP32 upcast            | ★★★★☆      |
| `areal/utils/functional/functional.py`           | PPO log-ratio FP32 + sanitize | ★★★★☆      |
| `areal/trainer/ppo/actor.py`                     | NaN/Inf 检查                  | ★★★★☆      |
| `areal/engine/fsdp_utils/optimizer.py`           | AnyPrecisionAdamW (Kahan)     | ★★★★☆      |
| `areal/experimental/models/archon/moe/router.py` | FP32 router gate              | ★★★★☆      |

### 测试框架核心

| 文件                                               | 功能                      | 精度相关度 |
| -------------------------------------------------- | ------------------------- | ---------- |
| `tests/test_cuda_deterministic.py`                 | bit-identical 回归        | ★★★★★      |
| `tests/experimental/archon/test_grpo.py`           | GRPO 多引擎对比           | ★★★★★      |
| `tests/fp8/comparison_utils.py`                    | FP8/BF16 逐层对比         | ★★★★★      |
| `tests/fp8/model_hooks.py`                         | activation/gradient hooks | ★★★★★      |
| `tests/experimental/archon/utils.py`               | ComparisonMetrics         | ★★★★☆      |
| `tests/experimental/archon/torchrun/dist_utils.py` | golden model + verify     | ★★★★☆      |
| `tests/test_fsdp_optimizer_dtype.py`               | FP32 master weight 回归   | ★★★★☆      |

______________________________________________________________________

## Appendix C. 关键 commits / PR / issues 清单

| Commit     | PR    | 时间       | 主题                                                   | 新增测试 | 精度类别             |
| ---------- | ----- | ---------- | ------------------------------------------------------ | -------- | -------------------- |
| `db3e3ded` | #1    | 2025-02-27 | Normalize loss scale by tokens                         | ❌       | loss normalization   |
| `e7c4a49a` | #4    | 2025-02-28 | Fix dataloader shuffle and random seed                 | ❌       | data determinism     |
| `fb23009e` | #27   | 2025-03-12 | Support bf16 training                                  | ❌       | mixed precision      |
| `0b15ead8` | #34   | 2025-03-17 | Fix the fp16 training issue                            | ❌       | mixed precision      |
| `9727753d` | #320  | 2025-09-12 | Fix gradient norm clipping for FSDP                    | ❌       | gradient correctness |
| `6b78aad2` | #335  | 2025-09-14 | Fix FSDP TP q/k norm + grad clipping                   | ❌       | TP correctness       |
| `d5b98d6d` | #340  | 2025-09-15 | Add deterministic option for MegatronEngine            | ❌       | determinism          |
| `a8c8fd67` | #426  | 2025-10-14 | Fix FSDP tensor parallelism for PPO                    | ❌       | TP correctness       |
| `9fb41107` | #536  | 2025-11-05 | Fix randomly failed CI tests                           | ✅       | test determinism     |
| `128299b2` | #497  | 2025-10-28 | Support PyTorch DCP for FSDP                           | ✅       | checkpoint           |
| `89dda13a` | #758  | 2025-12-31 | Add FP8 training support                               | ✅✅     | FP8 precision        |
| `be0271ff` | #802  | 2026-01-12 | Direct TE FP8→PyTorch FP8 conversion                   | ❌       | FP8 conversion       |
| `055066a9` | #940  | 2026-02-28 | Default score_before_experts to False for HF parity    | ✅✅     | HF parity            |
| `5b4ed832` | #943  | 2026-02-28 | Add deterministic training mode for Archon             | ✅✅     | determinism          |
| `4f5a2944` | #1009 | 2026-03-08 | Add moe_router_dtype for FP32 router gate              | ✅       | MoE stability        |
| `f6331e09` | #1087 | 2026-03-31 | Add FP8 blockwise training support                     | ✅✅     | FP8 precision        |
| `0ee85625` | #1118 | 2026-03-31 | Harden FP8 blockwise for TP and MoE                    | ✅       | FP8 + parallel       |
| `ad622efd` | #1210 | 2026-04-20 | Move data service seed to worker-level config          | ❌       | data determinism     |
| `5e9fb505` | #1273 | 2026-04-29 | Compensate pipeline schedule /num_microbatches in grad | ✅       | gradient correctness |
| `237a49f6` | #1292 | 2026-05-27 | Maintain fp32 master weights for AdamW                 | ✅✅     | optimizer precision  |

**图例**: ✅ 有测试，✅✅ 有系统性测试套件，❌ 无专门测试
