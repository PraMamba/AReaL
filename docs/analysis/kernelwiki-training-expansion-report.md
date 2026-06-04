# KernelWiki Training Library Expansion Report: AReaL

**Framework**: AReaL (A Large-Scale Asynchronous Reinforcement Learning System) **GitHub
URL**: inclusionAI/AReaL **Analysis Path**:
`/root/AReaL/.worktrees/source_code_analysis` **Analysis Date**: 2026-05-28 **Library
Type**: **Training-Orchestration** (zero CUDA kernel files; 3 native Triton kernels; 14+
upstream kernel providers)

______________________________________________________________________

## Library Type Classification and Dimension Emphasis

AReaL contains **zero** `.cu`/`.cuh`/`.ptx` CUDA kernel files and only **3 native Triton
kernel files** (1,425 lines total). Per the KernelWiki training library analysis skill,
AReaL is classified as a **training-orchestration framework**. Dimension emphasis is
adjusted accordingly:

| Dimension                   | Emphasis                           | Rationale                                                         |
| --------------------------- | ---------------------------------- | ----------------------------------------------------------------- |
| Dim 1: Compute Kernels      | **Light** (dependency graph focus) | Zero CUDA kernels; AReaL orchestrates upstream kernel providers   |
| Dim 2: Communication        | **Deep**                           | Rich NCCL integration with RL-specific cross-engine communication |
| Dim 3: Parallelism          | **Deep** (primary dimension)       | 5D composable parallelism + RL async actor-learner architecture   |
| Dim 4: Memory Management    | **Deep**                           | 17+ memory strategies including RL-specific offload lifecycles    |
| Dim 5: Precision Management | **Deep**                           | 14 precision strategies spanning 4 FP8 scaling recipes            |
| Dim 6: Profiling            | **Moderate**                       | 18 observability features with RL-specific metrics                |

______________________________________________________________________

## Dimension 1: Compute Kernels

### Kernel File Census

Total kernel files: **3** (all Triton; zero CUDA)

| Type                   | Count | Directories                                                                                            |
| ---------------------- | ----- | ------------------------------------------------------------------------------------------------------ |
| CUDA/C++               | 0     | N/A                                                                                                    |
| Triton                 | 3     | `areal/engine/megatron_utils/fp8/`, `areal/experimental/models/archon/moe/`, `areal/models/tree_attn/` |
| Extension entry points | 0     | No PYBIND11_MODULE, TORCH_LIBRARY, or CUDAExtension                                                    |

**Custom PyTorch Op** (pure-Python, no compiled extension):

- `@torch.library.custom_op("areal::_varlen_attn")` at
  `areal/experimental/models/archon/attention/varlen.py:19` — wraps
  `torch.ops.aten._flash_attention_forward`
- `@torch.library.custom_op("areal::_varlen_attn_backward")` at line 97 — backward
  counterpart

### Native Triton Kernels

| Kernel                          | File Path                                            | Lines | Description                                                                                                          |
| ------------------------------- | ---------------------------------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------- |
| `_blockwise_cast_to_fp8_triton` | `areal/engine/megatron_utils/fp8/kernels.py:13`      | 98    | Blockwise FP8 E4M3 quantization (128x128 tiles, per-tile absmax scale). Adapted from Slime project.                  |
| `weight_dequant_kernel`         | `areal/engine/megatron_utils/fp8/kernels.py:115`     | 41    | FP8 weight dequantization (loads FP8 weights + per-block scale, multiplies to restore float). Adapted from DeepSeek. |
| `_fill_indices_kernel`          | `areal/experimental/models/archon/moe/kernels.py:12` | 48    | MoE permutation index generation for expert-grouped token dispatch. Adapted from torchtitan.                         |
| `_tree_attn_fwd_triton`         | `areal/models/tree_attn/triton_kernel.py:180`        | 245   | Tree-masked sparse attention forward (bit-packed ancestor masks, CSR-style block-sparse, GQA, autotuned).            |
| `_tree_attn_bwd_preprocess`     | `areal/models/tree_attn/triton_kernel.py:425`        | 48    | Backward preprocess: `delta = rowsum(O * dO)`.                                                                       |
| `_tree_attn_bwd_dq`             | `areal/models/tree_attn/triton_kernel.py:473`        | 154   | Backward dQ with sparse block iteration guided by `kv_indices`. Autotuned.                                           |
| `_tree_attn_bwd_dkdv`           | `areal/models/tree_attn/triton_kernel.py:627`        | 412   | Backward dK/dV with sparse block iteration guided by `q_indices`. Autotuned.                                         |

### Kernel Dependency Graph (Primary — Orchestration Framework)

Since AReaL writes no CUDA kernels, this dependency graph is the core of Dimension 1.

| Provider Library                               | Dependency Type                 | Kernel Types Provided                                                                                                                | Import/Include Evidence                                                                                 |
| ---------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| **PyTorch** (torch >= 2.9.1)                   | REQUIRED                        | GEMM (cuBLAS), SDPA/FlashAttention, normalization, elementwise, reduction, NCCL collectives, torch.inductor fused kernels            | 210+ files import `torch`; `torch.ops.aten._flash_attention_forward` at `archon/attention/varlen.py:49` |
| **FlashAttention 4** (flash-attn-4 >= 4.0.0b4) | REQUIRED                        | FlashAttention-2/3/4 forward/backward, FlashMLA, variable-length attention                                                           | `pyproject.toml:239`; validated at `tools/validation_base.py:65`                                        |
| **torchao** (== 0.15.0)                        | REQUIRED                        | Blockwise FP8 GEMM (`fp8_blockwise_mm`)                                                                                              | `experimental/models/archon/fp8.py:36`; `archon/moe/grouped_experts.py:132`                             |
| **Megatron-Core** (== 0.17.0)                  | OPTIONAL \[megatron\]           | TP GEMM (ColumnParallelLinear/RowParallelLinear), FusedLayerNorm, FP8 TE layers, pipeline-parallel scheduling, distributed optimizer | `engine/megatron_engine.py:20-31`; `models/mcore/registry.py:8-13`                                      |
| **TransformerEngine**                          | OPTIONAL (runtime)              | FP8 Linear/Norm, `multi_tensor_l2norm`, `multi_tensor_scale`, `FusedAdam`                                                            | `engine/fsdp_utils/grad.py:19-25`; `models/mcore/bailing_moe.py:240`                                    |
| **SGLang** (== 0.5.10.post1)                   | OPTIONAL \[sglang\]             | PagedAttention, continuous batching, FlashAttention-3 (sgl_kernel), fused sampling                                                   | `experimental/inference_service/sglang/launch_server.py:18-25`                                          |
| **vLLM** (== 0.19.1)                           | OPTIONAL \[vllm\]               | PagedAttention, continuous batching, Marlin GEMM, FlashAttention-2/3, fused sampling                                                 | `engine/vllm_ext/areal_vllm_server.py:10-22`                                                            |
| **flash-linear-attention** (== 0.4.2)          | OPTIONAL \[cuda-train\]         | `chunk_simple_gla`, `chunk_gated_delta_rule`, `FusedRMSNormGated`                                                                    | `models/mcore/lightning_attention.py:304`; `experimental/models/archon/qwen3_5/model/model.py:29-30`    |
| **NVIDIA apex**                                | OPTIONAL (runtime fallback)     | `amp_C.multi_tensor_l2norm`, `amp_C.multi_tensor_scale`, `FusedAdam`                                                                 | `engine/fsdp_utils/grad.py:29-33` (fallback after TE)                                                   |
| **nv-grouped-gemm**                            | OPTIONAL (runtime via Megatron) | Batched grouped GEMM for MoE experts                                                                                                 | `models/mcore/bailing_moe.py:206` (`moe_grouped_gemm=True`)                                             |
| **causal-conv1d**                              | OPTIONAL (runtime)              | CUDA causal 1D depthwise convolution (Mamba/SSM layers)                                                                              | `experimental/models/archon/qwen3_5/model/model.py:24`                                                  |
| **torch_memory_saver** (== 0.0.9)              | OPTIONAL \[tms\]                | LD_PRELOAD CUDA memory hook (cudaMalloc/cudaFree interception)                                                                       | `utils/offload.py:13,31-37`                                                                             |
| **HuggingFace kernels** (== 0.12.2)            | OPTIONAL \[kernels\]            | Dynamic dispatch to any HF Hub-published attention kernel                                                                            | `engine/fsdp_utils/attn_impl.py:3,16`                                                                   |
| **TileLang** (>= 0.1.9)                        | OPTIONAL \[cuda-train\]         | Tile-based CUDA kernel DSL (no direct imports found in areal/)                                                                       | `pyproject.toml:173`                                                                                    |
| **nvidia-modelopt**                            | OPTIONAL \[cuda-train\]         | Model quantization/pruning (no direct imports found in areal/)                                                                       | `pyproject.toml:171`                                                                                    |

### Proposed New kernel_types

No new kernel_types needed. AReaL's native Triton kernels map to existing tags:

- Tree attention kernels -> `attention` (existing)
- FP8 cast kernels -> `fp8-cast` (existing)
- MoE permutation kernel -> `moe` (existing)

______________________________________________________________________

## Dimension 2: Communication Kernels and Strategies

### Collective Operations

| Operation         | Algorithm(s)               | SM Usage | File Path                                                                                      | Parallelism Trigger                                                                            |
| ----------------- | -------------------------- | -------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| AllReduce         | NCCL (Ring/Tree/NVLS auto) | Full SM  | `engine/core/train_engine.py:61,98`; `engine/fsdp_utils/grad.py:112-169`                       | DP gradient norm, loss aggregation, advantage normalization                                    |
| ReduceScatter     | NCCL via FSDP2 backward    | Full SM  | `experimental/models/archon/qwen2/infra/parallelize.py:56` (implicit via FSDP2)                | FSDP2 backward gradient sync, CP backward                                                      |
| AllGather         | NCCL via FSDP2 forward     | Full SM  | `engine/fsdp_engine.py:1964-2047`; `engine/megatron_utils/packed_context_parallel.py:176`      | FSDP2 forward param collection, SP logprob gather, CP logprob gather, TP weight reconstruction |
| AllToAll          | NCCL via PyTorch           | Full SM  | `models/fsdp/ulysses.py:61,90`; `experimental/models/archon/expert_parallel.py:131,163,207`    | Ulysses SP (head/seq scatter-gather), EP token dispatch/combine                                |
| Broadcast         | NCCL                       | Full SM  | `engine/fsdp_engine.py:1374,1399`; `engine/megatron_engine.py:1425`; `utils/data.py:1251-1306` | Train->inference weight sync, rollout batch distribution, init state dict                      |
| P2P (isend/irecv) | NCCL batch_isend_irecv     | Full SM  | `experimental/weight_update/nccl_group.py:177-197`                                             | AWEX weight transfer warmup, ring-exchange NCCL group init                                     |
| Barrier           | Gloo (CPU backend)         | None     | `infra/dist_rollout.py:139,148`; `trainer/rl_trainer.py:1106-1191`                             | RL step boundaries (Gloo avoids NCCL deadlock)                                                 |

### Communication-Compute Overlap Patterns

| Pattern                           | Mechanism                                                                                                       | Evidence                                                        |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Megatron DDP gradient overlap     | `overlap_grad_reduce=True`: gradient AllReduce overlapped with backward pass computation                        | `api/cli_args.py:719`; `engine/megatron_engine.py:447`          |
| Param gather + optimizer step     | `overlap_param_gather_with_optimizer_step=True`: distributed optimizer AllGather overlapped with optimizer step | `api/cli_args.py:858`; `engine/megatron_engine.py:1311-1312`    |
| PP P2P overlap with VPP           | `overlap_p2p_comm=True` when VPP > 1: pipeline P2P activations overlap with compute                             | `models/mcore/registry.py:243-251`                              |
| MoE shared expert overlap         | `moe_shared_expert_overlap=True`: shared expert compute overlapped with AllToAll token dispatch                 | `api/cli_args.py:905-908`                                       |
| FSDP per-layer optimizer prefetch | `PerLayerOptimStep` with `prefetch_layers=N`: async H2D optimizer state DMA via pinned memory                   | `engine/fsdp_utils/optimizer.py:340-615`; `api/cli_args.py:444` |
| FSDP forward/backward prefetch    | Explicit `set_modules_to_forward_prefetch()` / `set_modules_to_backward_prefetch()` chains                      | `experimental/models/archon/qwen3/infra/parallelize.py:436-499` |
| Async weight broadcast            | `dist.broadcast(..., async_op=True)` with handle-batched wait for weight sync                                   | `experimental/engine/archon_weight_sync.py:454,489`             |
| DTensor async redistribution      | `redistribute(..., async_op=True)` for TP input/output layout conversion                                        | `models/parallel_styles.py:61,69`                               |
| RL two-batch-in-flight overlap    | `active_submit_and_wait` keeps 2 rollout batches in-flight simultaneously                                       | `infra/workflow_executor.py:1283`                               |

### Advanced Communication Features Checklist

- [ ] Symmetric memory support (NCCL 2.27+): **No** — zero occurrences. NVLS excluded
  via `^NVLS` in deterministic mode (`engine/megatron_utils/deterministic.py:27`)
- [ ] Device API support (NCCL 2.28+): LSA **No**, Multimem **No**, GIN **No**
- [ ] Copy Engine zero-SM collectives: **No** — all communication uses SM-based NCCL
- [ ] NCCL Inspector integration: **No**
- [ ] PyTorch SymmetricMemory: **No**
- [ ] Alternative backend support: **Partial** — DeepEP flag exists
  (`moe_enable_deepep: bool`, `api/cli_args.py:912`) but not fully wired; NVSHMEM
  referenced only in validation (`tools/validate_docker_installation.py:299`)

### RL-Specific Communication Patterns (Unique to AReaL)

| Pattern                          | Mechanism                                                                                                                                                                      | Files                                                                                |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Cross-engine NCCL weight sync    | `init_custom_process_group` creates NCCL groups spanning separate training and inference distributed contexts. Per-PP-rank groups for stage-matched parameter broadcast.       | `experimental/weight_update/nccl_group.py:18-117`; `engine/fsdp_engine.py:1457-1557` |
| RTensor HTTP fetch               | Inference workers store rollout results as `RTensor` with shard IDs; training workers fetch via `aiohttp GET /data/{shard_id}`. Primary actor-learner data transfer mechanism. | `infra/rpc/rtensor.py`                                                               |
| Rollout broadcast + redistribute | DP-head ranks receive rollouts, `all_gather_tensor_container` across DP, load-balance with packing, `broadcast_tensor_container` across model-parallel group.                  | `infra/dist_rollout.py`                                                              |
| AWEX P2P weight transfer         | Experimental `batch_isend_irecv` for direct shard-level P2P parameter transfer from training to inference, bypassing broadcast.                                                | `experimental/weight_update/nccl_group.py:177-197`                                   |
| Gloo CPU barrier pattern         | All RL step boundaries use `dist.barrier(group=cpu_group)` with Gloo backend to avoid NCCL deadlocks.                                                                          | `trainer/rl_trainer.py:1106-1191`; `experimental/engine/archon_checkpoint.py:46-48`  |
| RL advantage normalization       | `dist.all_reduce` across DP group for global PPO advantage mean/variance.                                                                                                      | `utils/data.py:1525-1586`; `utils/functional/functional.py:41-43`                    |

### Proposed New Communication techniques

| Tag                        | Evidence                                      | Description                                                                                |
| -------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `cross-engine-nccl-sync`   | `experimental/weight_update/nccl_group.py:18` | NCCL process groups spanning separate distributed contexts for train-inference weight sync |
| `rl-async-rollout-overlap` | `infra/workflow_executor.py:1283`             | Two-batch-in-flight async rollout overlapping inference generation with training           |

______________________________________________________________________

## Dimension 3: Parallelism Strategies

### Supported Parallelism Dimensions

| Dimension                           | Supported                                         | Implementation File                                                                                         | Communication Pattern Triggered                            |
| ----------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Data Parallel (FSDP2, ZeRO-3)       | **Yes**                                           | `engine/fsdp_utils/__init__.py:62-106`; `engine/fsdp_utils/parallel.py:370-396`                             | AllGather before forward + ReduceScatter after backward    |
| Tensor Parallel                     | **Yes**                                           | `engine/fsdp_utils/parallel.py:219-396` (DTensor); `engine/megatron_engine.py:230` (Mcore)                  | AllReduce/AllGather per MLP and Attention sublayer         |
| Pipeline Parallel                   | **Yes** (Megatron + Archon only; blocked on FSDP) | `engine/megatron_utils/pipeline_parallel.py:17-99`; `experimental/models/archon/pipeline_parallel.py:48-82` | Send/Recv for micro-batch passing                          |
| Context Parallel (Ring)             | **Yes** (Megatron)                                | `engine/megatron_utils/packed_context_parallel.py`                                                          | AllGather across CP group (interleaved-causal split)       |
| Context Parallel (Ulysses AllToAll) | **Yes** (FSDP + Archon)                           | `models/fsdp/ulysses.py:61,90`; `experimental/models/archon/ulysses.py:81`                                  | Two AllToAll per attention layer (head/seq scatter-gather) |
| Expert Parallel                     | **Yes**                                           | `engine/fsdp_utils/parallel.py:54-122`; `experimental/models/archon/expert_parallel.py:70-200`              | AllToAll for token routing + combine                       |
| Sequence Parallel                   | **Yes**                                           | Integrated with TP via DTensor `SequenceParallel` on LayerNorm layers                                       | Fused with TP communication                                |
| RL Async (Actor-Learner)            | **Yes** (unique)                                  | `infra/workflow_executor.py:262-732`; `infra/staleness_manager.py:20-182`                                   | HTTP RPC + NCCL broadcast for weight sync                  |

### DeviceMesh Topology

**FSDP backend (without EP) -- 3D mesh:**

```
init_device_mesh(mesh_shape=(dp, sp, tp), mesh_dim_names=("dp", "sp", "tp"))
Derived: dp_sp = dp x sp (FSDP sharding domain), sp_tp = sp x tp
Constraint: dp * sp * tp == world_size
```

**FSDP backend (with EP, ETP=TP mode) -- 4D mesh:**

```
init_device_mesh(mesh_shape=(dp_mod_ep, dp_in_ep, sp, tp), mesh_dim_names=(...))
Derived: ep = dp_in_ep x sp (EP borrows from DP, not a new world dimension)
```

**Archon backend (without EP) -- 4D mesh:**

```
init_device_mesh(mesh_shape=(pp, dp_shard, cp, tp), mesh_dim_names=("pp","dp_shard","cp","tp"))
Derived: dp_shard_cp = dp_shard x cp (FSDP sharding + loss AllReduce)
Constraint: pp * dp_shard * cp * tp == world_size
```

**Archon backend (with EP) -- 5D mesh:**

```
init_device_mesh(mesh_shape=(pp, dp_shard_mod_ep, dp_shard_in_ep, cp, tp), mesh_dim_names=(...))
Derived: ep_tp 2D mesh for ExpertTensorParallel (2D sharding [Shard(0), Shard(1/2)])
```

**Megatron-Core backend -- 5D internal:**

```
mpu.initialize_model_parallel(tp, pp, vpp, cp, ep)
No PyTorch DeviceMesh -- Megatron maintains internal process group state
```

### Pipeline Scheduling Strategies

| Schedule              | Backend           | Description                                                               | Bubble Rate | Evidence                                                |
| --------------------- | ----------------- | ------------------------------------------------------------------------- | ----------- | ------------------------------------------------------- |
| 1F1B                  | Megatron + Archon | Classic one-forward-one-backward per micro-batch                          | ~P/M        | `experimental/models/archon/pipeline_parallel.py:48-82` |
| Interleaved1F1B       | Megatron + Archon | Looped schedule, 2+ virtual stages per rank (default Archon)              | ~1/(P\*V)   | `api/cli_args.py:601`                                   |
| InterleavedZeroBubble | Archon            | Zero-bubble variant of Interleaved                                        | ~0          | `experimental/models/archon/pipeline_parallel.py:71`    |
| ZBVZeroBubble         | Archon            | V-style assignment (rank 0 owns stages 0 and 2P-1), exactly 2 stages/rank | ~0          | `experimental/models/archon/pipeline_parallel.py:75`    |

### RL Async Parallelism Architecture (Unique Differentiator)

**Worker role separation:**

- **Rollout workers** (inference/generation): vLLM or SGLang servers producing
  trajectories
- **Training workers** (gradient update): FSDP, Megatron, or Archon engines running
  PPO/DPO/SFT
- **Auxiliary workers**: Reference model, critic model, teacher model (each in own
  engine group)

**Resource allocation modes** (`AllocationType`, `api/alloc_mode.py:16-21`):

- `COLOCATE` (0): Actor and rollout share GPUs; memory offload (`offload/onload`)
  alternates between them
- `DECOUPLED_TRAIN` (1): Separate GPU pools for training and inference
- `LLM_SERVER_ONLY` (2): Inference-only, no training

**Staleness control** (`infra/staleness_manager.py:20-182`):

- `max_concurrent_rollouts`: concurrency limit
- `max_staleness`: maximum policy version lag between generation and training
- Version-stamped per-token tracking enables per-token importance weighting

**Weight update flow** (`trainer/rl_trainer.py:716-740`):

1. `actor.prepare_batch()` -- block until accepted trajectories arrive
1. Compute critic values, ref logprobs, teacher logprobs (synchronous)
1. `actor.ppo_update()` -- gradient step
1. `rollout.pause()` -- **sole synchronization barrier** in the entire RL loop
1. `actor.update_weights()` -- NCCL broadcast to inference workers (via `xccl`), or disk
   checkpoint, or AWEX P2P
1. `rollout.resume()` -- re-open inference queue

### Key Parallelism Findings

1. **SP participates in FSDP sharding**: FSDP mesh is `dp_sp` (not just `dp`), so CP/SP
   ranks hold non-overlapping parameter shards. Increasing SP reduces both memory and
   sequence length per device simultaneously.
1. **EP borrows from DP**: EP does not add a new world dimension; it carves from
   `dp_in_ep` sub-ranks. The invariant `dp * sp * tp == world_size` is preserved.
1. **Async RL decouples throughput**: Training can proceed without waiting for
   generation. The staleness window controls maximum policy lag.
1. **Weight update is the only hard sync point**: Between PPO steps, training and
   inference run fully independently. AWEX mode removes even this barrier.
1. **CP differs by backend**: Megatron uses interleaved-chunk ring CP; FSDP/Archon use
   Ulysses AllToAll. Different communication costs and constraints.

### Proposed New Parallelism techniques

| Tag                      | Evidence                                            | Description                                                                           |
| ------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `rl-async-actor-learner` | `infra/workflow_executor.py:262-732`                | Fully asynchronous actor-learner RL training with version-stamped staleness control   |
| `ulysses-sp`             | `models/fsdp/ulysses.py:61,90`                      | DeepSpeed Ulysses-style sequence parallelism via AllToAll head/seq scatter-gather     |
| `colocated-memory-swap`  | `utils/offload.py`; `engine/fsdp_engine.py:901-935` | GPU memory time-multiplexing between colocated inference and training engines via TMS |

______________________________________________________________________

## Dimension 4: Memory Management

### Memory Component Analysis

| Component          | Storage Format                           | Sharding Strategy                                   | Communication Kernel Triggered                  |
| ------------------ | ---------------------------------------- | --------------------------------------------------- | ----------------------------------------------- |
| Parameters         | BF16 (cast from FP32 master by FSDP2 MP) | FSDP2 ZeRO-3 (AllGather/ReduceScatter)              | AllGather before each layer's forward pass      |
| Gradients          | BF16 (reduced in FP32)                   | FSDP2 ReduceScatter                                 | ReduceScatter after each layer's backward pass  |
| Optimizer States   | FP32 (default) or BF16 (adam_bf16)       | FSDP2 ZeRO-3                                        | None (local optimizer step after ReduceScatter) |
| Activations        | BF16                                     | Selective recomputation or full checkpoint          | None (stored or recomputed locally)             |
| RL Rollout Buffers | BF16/FP32 mixed                          | Not sharded (gathered to DP-head, broadcast to all) | AllGather + Broadcast                           |
| Reference Model    | Same as actor                            | Separate engine (offloadable)                       | None when offloaded                             |
| Critic/Value Model | Same as actor                            | Separate engine (offloadable)                       | None when offloaded                             |
| Inference KV-Cache | BF16 (managed by vLLM/SGLang)            | TP-sharded within inference engine                  | None (internal to inference engine)             |

### Activation Checkpointing Strategies

| Strategy                               | Backend           | Description                                                                                                               | Memory-Compute Tradeoff                                        | Evidence                                                    |
| -------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------- |
| HF `gradient_checkpointing_enable`     | FSDP              | Standard non-reentrant checkpointing on all transformer blocks                                                            | Maximum memory savings, ~33% compute overhead                  | `engine/fsdp_engine.py:1076-1079`                           |
| Megatron granular recompute            | Megatron          | `recompute_granularity` (full/selective), `recompute_method` (uniform/block), `recompute_num_layers`, `recompute_modules` | Flexible 30-100% activation reduction                          | `engine/megatron_engine.py:520-526`                         |
| Archon full/selective/op/memory-budget | Archon            | Four modes: `"full"`, `"selective"` (per-N-layer or per-op SAC), `"memory_budget"` (auto ratio)                           | Highly tunable; per-op SAC saves ~50% by recomputing cheap ops | `experimental/models/archon/activation_checkpoint.py:1-270` |
| PPO `recompute_logprob`                | All (RL-specific) | Recomputes proximal log-probs via forward pass instead of caching from inference                                          | Eliminates log-prob transfer buffer; +1 forward pass           | `trainer/ppo/actor.py:82,178-186`                           |

### CPU Offload Strategies

| Strategy                          | Target                     | Mechanism                                                                             | Evidence                                                      |
| --------------------------------- | -------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| FSDP2 `CPUOffloadPolicy`          | Parameters                 | FSDP2 built-in `offload_policy` evicts params to CPU between forward/backward         | `engine/fsdp_engine.py:405-406`; `api/cli_args.py:421-423`    |
| `PerLayerOptimWrapper`            | Optimizer states           | States always on CPU; async H2D/D2H DMA via pinned memory per-layer streaming         | `engine/fsdp_utils/optimizer.py:324-559`                      |
| Megatron adapter `release_memory` | Params + optimizer states  | Moves all CUDA tensors to CPU, replaces with empty placeholders                       | `experimental/weight_update/awex/megatron_adapter.py:411-550` |
| SGLang adapter memory release     | Inference KV-cache + model | Calls SGLang's `release_memory_occupation` RPC                                        | `experimental/weight_update/awex/sglang_adapter.py:581-641`   |
| TorchMemorySaver (TMS)            | All engine GPU memory      | `LD_PRELOAD` CUDA memory hook intercepts cudaMalloc/cudaFree for transparent CPU swap | `utils/offload.py:13,31-37`; `engine/fsdp_engine.py:901-935`  |
| Memory-efficient model loading    | Peak init-phase memory     | Rank 0 loads, others use `meta` tensors, broadcast via `fsdp2_load_full_state_dict`   | `engine/fsdp_engine.py:413-460`                               |

### RL-Specific Memory Lifecycle

AReaL's RL training step alternates GPU memory ownership between inference and training:

```
onload_rollout -> collect_rollout -> offload_rollout -> onload_actor ->
  [onload_critic -> compute_values -> offload_critic] ->
  [onload_ref -> compute_ref_logp -> offload_ref] ->
  ppo_update -> update_weights -> offload_actor -> repeat
```

Each model (actor, critic, ref, teacher) has independent `offload: bool` config.
Colocated mode uses TMS `LD_PRELOAD` hook for transparent GPU memory swapping.

### Proposed New Memory techniques

| Tag                             | Evidence                                 | Description                                                                                           |
| ------------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `rl-model-lifecycle-offload`    | `trainer/rl_trainer.py:422-786`          | RL-specific alternating offload/onload lifecycle for actor, critic, ref, teacher, and rollout engines |
| `per-layer-optimizer-streaming` | `engine/fsdp_utils/optimizer.py:324-559` | Per-layer optimizer state CPU-GPU streaming via async pinned-memory DMA                               |

______________________________________________________________________

## Dimension 5: Precision Management

### FP8 Scaling Strategies Found

| Strategy                   | Class Name                     | Granularity                                 | Data Formats                                   | Scale Factor Type                                                   | GPU Support               | Evidence                                                          |
| -------------------------- | ------------------------------ | ------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------- |
| Delayed Scaling            | TE `DelayedScaling`            | Per-tensor                                  | E4M3 (fwd) + E5M2 (bwd) in HYBRID; or all E4M3 | FP32 from historical amax (`amax_history_len`, `amax_compute_algo`) | Hopper + Blackwell        | `api/cli_args.py:730-831`; `engine/megatron_engine.py:1158-1190`  |
| Current Scaling            | TE `Float8CurrentScaling`      | Per-tensor                                  | HYBRID default                                 | FP32 from current amax (no history)                                 | Hopper + Blackwell        | `api/cli_args.py:750` (`recipe="tensorwise"`)                     |
| Block Scaling              | TE `Float8BlockScaling`        | 2D configurable blocks                      | All E4M3                                       | FP32 power-of-2 per block                                           | Hopper (software)         | `api/cli_args.py:750` (`recipe="blockwise"`)                      |
| MXFP8                      | TE `MXFP8BlockScaling`         | 32 elements                                 | All E4M3                                       | E8M0 (power-of-2 only, 8-bit exponent)                              | Blackwell only (hardware) | `api/cli_args.py:750` (`recipe="mxfp8"`); enforced Blackwell-only |
| Triton blockwise FP8 cast  | `blockwise_cast_to_fp8_triton` | 128x128 blocks                              | E4M3                                           | FP32 (absmax/fp8_max per block)                                     | Hopper+ (Triton)          | `engine/megatron_utils/fp8/kernels.py:13-111`                     |
| UE8M0 blockwise (DeepGEMM) | `quant_weight_ue8m0`           | 128x128 blocks                              | E4M3                                           | UE8M0 packed uint8 (power-of-2 via `ceil_to_ue8m0`)                 | Blackwell SM100+ only     | `engine/megatron_utils/fp8/ue8m0.py:1-183`                        |
| torchao blockwise FP8      | `fp8_blockwise_mm`             | Mixed: 1x128 (activation), 128x128 (weight) | E4M3                                           | Implicit within torchao                                             | Hopper+                   | `experimental/models/archon/fp8.py:18-163`                        |

### Precision per Training Component

| Component               | FSDP Engine             | Megatron (FP8 off) | Megatron (FP8 on)         | Archon (FP8 off) | Archon (FP8 on)          |
| ----------------------- | ----------------------- | ------------------ | ------------------------- | ---------------- | ------------------------ |
| Forward GEMM inputs     | BF16                    | BF16               | FP8 E4M3 (TE recipe)      | BF16             | FP8 E4M3 (torchao)       |
| Forward GEMM weights    | BF16 (from FP32 master) | BF16               | FP8 E4M3 (TE param)       | BF16             | BF16 master, FP8 in GEMM |
| Backward gradients      | BF16                    | BF16               | FP8 E5M2 (HYBRID) or E4M3 | BF16             | BF16                     |
| Gradient reduction      | FP32                    | FP32               | FP32 or FP8 (mxfp8)       | FP32             | FP32                     |
| Optimizer master params | FP32                    | FP32               | FP32                      | FP32             | FP32                     |
| Adam momentum           | FP32 or BF16            | FP32 or BF16       | FP32 or BF16              | FP32             | FP32                     |
| Adam variance           | FP32 or BF16            | FP32 or BF16       | FP32 or BF16              | FP32             | FP32                     |
| MoE router gate         | BF16                    | FP32 (override)    | FP32 (override)           | FP32 (override)  | FP32 (override)          |
| AllGather (FSDP param)  | BF16                    | BF16 or FP8        | FP8 (`fp8_param_gather`)  | BF16             | BF16                     |

### FP8 Communication Integration

- [ ] FP8 AllGather in FSDP2 (parameters communicated in FP8): **Yes** — via
  `fp8_param_gather=True` in Megatron path (`api/cli_args.py:726`;
  `engine/megatron_utils/megatron.py:86-197`)
- [ ] FP8 ReduceScatter (gradients communicated in FP8): **No** — gradient reduction
  always FP32
- [ ] NVLink-SHARP FP8 in-switch reduction: **No** — not referenced
- [ ] Estimated communication volume reduction vs BF16: **~50%** for parameter AllGather
  when `fp8_param_gather=True` (2 bytes -> 1 byte per param + scale overhead)

### Special Precision Features

- **AnyPrecisionAdamW with Kahan summation** (`engine/fsdp_utils/optimizer.py:44-192`):
  BF16 optimizer states with Kahan compensation buffer for numerical stability.
  Activated via `optimizer.type="adam_bf16"` + `optimizer_dtype="bfloat16"`.
- **MoE router FP32 override** (`experimental/models/archon/moe/router.py:29-51`): Gate
  linear GEMM forced to FP32 in both forward and backward for numerical stability,
  regardless of model dtype.
- **FP8 checkpoint detection** (`experimental/models/archon/fp8_checkpoint.py:34-328`):
  Automatic detection and dequantization of FP8 checkpoints during loading. Heuristic:
  presence of `*_scale_inv` keys.
- **FP8BlockwiseTensorHelper** (`engine/megatron_utils/fp8/tensor_helper.py:11-441`):
  PyTorch tensor subclass bridging TE and PyTorch FP8 formats. Supports chunk, split,
  cat, view with automatic scale propagation.

### Proposed New Precision techniques

| Tag                       | Evidence                                   | Description                                                                               |
| ------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `kahan-bf16-optimizer`    | `engine/fsdp_utils/optimizer.py:44-192`    | BF16 optimizer states with Kahan summation compensation for numerical stability           |
| `ue8m0-deepgemm-quantize` | `engine/megatron_utils/fp8/ue8m0.py:1-183` | Blackwell-native UE8M0 blockwise quantization with TMA-aligned scale packing for DeepGEMM |

______________________________________________________________________

## Dimension 6: Profiling and Observability

### Built-in Profiling Capabilities

| Feature                            | Supported        | Integration Method                                                                                         | Evidence                                                           |
| ---------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| PerfTracer Chrome Trace            | Yes (opt-in)     | Built-in `@trace_perf` decorator + `trace_scope` context manager; JSONL output convertible to Chrome Trace | `utils/perf_tracer.py:1271-1912`; `api/cli_args.py:2517-2554`      |
| SessionTracer lifecycle            | Yes (opt-in)     | Built-in per-session tracing with generate/reward/toolcall phase spans and staleness tracking              | `utils/perf_tracer.py:922-1127`                                    |
| Embedded PyTorch Profiler          | Yes (step-gated) | `torch.profiler.profile` embedded within PerfTracer scopes; merged into JSONL stream                       | `utils/perf_tracer.py:1155-1222`                                   |
| NVTX annotations for nsys          | No               | Not used directly (PyTorch Profiler integration serves this role)                                          | N/A                                                                |
| NCCL Inspector plugin              | No               | Not integrated                                                                                             | N/A                                                                |
| Flight Recorder                    | No               | Not used                                                                                                   | N/A                                                                |
| CUDA Memory Snapshot               | Yes (opt-in)     | `torch.cuda.memory._record_memory_history/dump_snapshot` on configurable steps                             | `engine/megatron_engine.py:1217-1228`; `api/cli_args.py:2498-2516` |
| DeviceRuntimeInfo memory telemetry | Yes (always-on)  | `torch.cuda.memory_allocated/reserved/mem_get_info` at lifecycle points                                    | `api/io_struct.py:378-415`                                         |
| MFU metrics                        | No               | Not computed (no `model_flops` calculation found)                                                          | N/A                                                                |

### Experiment Tracking Backends

| Backend               | Config Class        | Default Status       | Evidence                    |
| --------------------- | ------------------- | -------------------- | --------------------------- |
| WandB                 | `WandBConfig`       | Disabled             | `api/cli_args.py:2351-2382` |
| SwanLab               | `SwanlabConfig`     | Disabled             | `api/cli_args.py:2383-2411` |
| TensorBoard           | `TensorBoardConfig` | Disabled (path=None) | `api/cli_args.py:2412-2418` |
| Trackio (HuggingFace) | `TrackioConfig`     | Disabled             | `api/cli_args.py:2419-2444` |

### RL-Specific Metrics (PPO Actor — 30+ metrics)

**Per valid token**: `approx_kl`, `entropy`, `actor_loss`, `clip_ratio`,
`dual_clip_ratio`, `new_logp`, `old_logp`, `importance_weight`, `advantages`,
`kl_rewards`, `final_reward`, `rkl_loss`, `vocab_min_logits`, `vocab_max_logits`

**Per sequence**: `no_eos_ratios`, `task_reward`, `prompt_len`, `seq_len`,
`correct_seq_len`, `incorrect_seq_len`

**KL estimators**: `kl_div_direct`, `kl_div_taylor`, `kl_div_dual`

**Version/staleness**: `sample_staleness_proximal_avg/max/min`,
`sample_staleness_theta_avg/max/min`, `v_theta`, `v_proximal`

**Training phase timers** (`timeperf/*`): `rollout`, `critic_values`, `ref_logp`,
`teacher_logp`, `recompute_logp`, `compute_advantage`, `train_step`,
`critic_train_step`, `update_weights`, `save`, `eval`, `clear_batches`, plus per-model
`{role}_onload`, `{role}_offload`

### Health Monitoring

- **Worker health check poller**: Background async polling of `/health` HTTP endpoints;
  marks workers healthy/unhealthy
  (`experimental/inference_service/router/app.py:191-214`)
- **NCCL hang watchdog**: Daemon thread logging warnings at 30s/60s/120s if
  `init_weights_update_group` blocks
  (`experimental/inference_service/sglang/pp_bridge.py:132-177`)
- **SGLang Prometheus metrics**: `enable_metrics=True` by default for SGLang inference
  workers (`api/cli_args.py:1846-1852`)

### Standalone Profiling Tools

| Tool                            | What It Does                                              | Output                                             |
| ------------------------------- | --------------------------------------------------------- | -------------------------------------------------- |
| `tools/profile_archon.py`       | Archon engine torch.profiler with CUDA op breakdown       | Chrome Trace JSON + top-30 ops table               |
| `tools/profile_fsdp.py`         | FSDP engine torch.profiler with memory tracking           | Chrome Trace JSON + top-30 ops table               |
| `tools/profile_engines.py`      | Side-by-side Archon vs FSDP comparison                    | Speedup ratio + peak memory diff                   |
| `tools/perf_trace_converter.py` | Merges multi-rank PerfTracer JSONL to single Chrome Trace | `traces.json` for `chrome://tracing` or Perfetto   |
| `tools/plot_session_trace.py`   | Generates HTML dashboards from SessionTracer JSONL        | Timeline, latency scatter, distribution histograms |

______________________________________________________________________

## Synthesis: Expansion Decision Summary

### S.1 Library Classification

| Property                        | Value                                                                                                                       |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Library                         | AReaL                                                                                                                       |
| GitHub URL                      | inclusionAI/AReaL                                                                                                           |
| Type                            | **training-orchestration**                                                                                                  |
| Contains CUDA Kernels           | No (3 native Triton kernels only)                                                                                           |
| Primary Knowledge Dimensions    | Dim 3 (Parallelism — 5D + RL async), Dim 4 (Memory — 17+ strategies), Dim 5 (Precision — 7 FP8 recipes)                     |
| Recommended KernelWiki Priority | **P1** (important but secondary — no native CUDA kernels; value is in orchestration patterns that trigger upstream kernels) |

### S.2 Proposed Tags (for controlled vocabulary YAML)

```yaml
kernel_types:
  # No new kernel_types needed -- AReaL's native kernels map to existing tags
  # (attention, fp8-cast, moe)

techniques:
  # New from AReaL
  - rl-async-actor-learner      # Fully asynchronous RL training with version-stamped staleness control
  - ulysses-sp                   # DeepSpeed Ulysses-style sequence parallelism via AllToAll head/seq scatter-gather
  - colocated-memory-swap        # GPU memory time-multiplexing between inference and training engines via TMS
  - cross-engine-nccl-sync       # NCCL process groups spanning separate distributed contexts for train-inference weight sync
  - rl-async-rollout-overlap     # Two-batch-in-flight async rollout overlapping inference generation with training
  - per-layer-optimizer-streaming # Per-layer optimizer state CPU-GPU streaming via async pinned-memory DMA
  - kahan-bf16-optimizer          # BF16 optimizer states with Kahan summation compensation
  - rl-model-lifecycle-offload    # RL-specific alternating offload/onload for actor/critic/ref/teacher/rollout
  - ue8m0-deepgemm-quantize      # Blackwell-native UE8M0 blockwise quantization with TMA-aligned packing

hardware_features:
  # No new hardware_features needed -- existing tags cover AReaL's usage
  # (fp8, block-scale, e8m0-scale-format, symmetric-memory, copy-engine already exist)

source_categories:
  # New
  - rl-training-framework        # Large-scale RL training orchestration framework (distinct from SFT/pretraining frameworks)
```

### S.3 Wiki Page Topics

| #   | Wiki Subdirectory | Proposed Page ID                   | Title                                           | Source Evidence                                                                                       | Related Existing KernelWiki Pages                          |
| --- | ----------------- | ---------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 1   | training/         | training-rl-async-architecture     | Asynchronous RL Training Architecture           | `areal/infra/workflow_executor.py`, `areal/infra/staleness_manager.py`, `areal/trainer/rl_trainer.py` | technique-comm-compute-overlap                             |
| 2   | parallelism/      | parallel-ulysses-sp                | Ulysses Sequence Parallelism (AllToAll)         | `areal/models/fsdp/ulysses.py`, `areal/experimental/models/archon/ulysses.py`                         | technique-comm-compute-overlap, technique-nccl-kernels     |
| 3   | training/         | training-colocated-memory-swap     | Colocated Inference-Training GPU Memory Sharing | `areal/utils/offload.py`, `areal/engine/fsdp_engine.py` (offload/onload), `torch_memory_saver`        | technique-fused-optimizer                                  |
| 4   | communication/    | comm-cross-engine-nccl             | Cross-Engine NCCL Weight Synchronization        | `areal/experimental/weight_update/nccl_group.py`, `areal/engine/fsdp_engine.py:1457-1557`             | technique-nccl-kernels, technique-comm-compute-overlap     |
| 5   | training/         | training-kahan-bf16-optimizer      | BF16 Optimizer States with Kahan Summation      | `areal/engine/fsdp_utils/optimizer.py:44-192`                                                         | technique-fused-optimizer, technique-fp8-training-pipeline |
| 6   | training/         | training-per-layer-optim-streaming | Per-Layer Optimizer CPU-GPU Streaming           | `areal/engine/fsdp_utils/optimizer.py:324-559`                                                        | technique-fused-optimizer                                  |
| 7   | parallelism/      | parallel-5d-composable             | 5D Composable Parallelism (DP+TP+PP+CP+EP)      | `areal/engine/fsdp_utils/parallel.py`, `areal/experimental/models/archon/parallel_dims.py`            | technique-comm-compute-overlap                             |

### S.4 Repository Mappings (slug -> org/repo)

```python
# For the PR candidate search script
"areal": "inclusionAI/AReaL",

# For the PR page generation script
"areal": "inclusionAI/AReaL",
```

### S.5 Keyword-to-Tag Mappings (for automated PR tagger)

```python
# keyword -> kernel_type tag (KW_TO_KT)
"tree_attn": "attention",
"tree_attention": "attention",
"triton_kernel": "attention",
"blockwise_cast": "fp8-cast",
"weight_dequant": "fp8-cast",
"fill_indices": "moe",
"expert_parallel": "moe",
"grouped_experts": "moe",
"ppo_update": "fused-optimizer",
"fsdp_engine": "weight-update",
"megatron_engine": "weight-update",

# keyword -> technique tag (KW_TO_TECH)
"async_rollout": "rl-async-actor-learner",
"staleness_manager": "rl-async-actor-learner",
"workflow_executor": "rl-async-rollout-overlap",
"ulysses": "ulysses-sp",
"all_to_all_single_autograd": "ulysses-sp",
"torch_memory_saver": "colocated-memory-swap",
"offload_rollout": "rl-model-lifecycle-offload",
"onload_rollout": "rl-model-lifecycle-offload",
"weight_update_group": "cross-engine-nccl-sync",
"init_custom_process_group": "cross-engine-nccl-sync",
"PerLayerOptimStep": "per-layer-optimizer-streaming",
"AnyPrecisionAdamW": "kahan-bf16-optimizer",
"kahan_summation": "kahan-bf16-optimizer",
"activation_checkpoint": "activation-checkpointing",
"gradient_checkpointing": "activation-checkpointing",
"fp8_autocast": "mixed-precision-training",
"FP8EngineConfig": "mixed-precision-training",
"blockwise_fp8": "rowwise-dynamic-scaling",
"mxfp8": "mxfp8-block-scaling",
"ue8m0": "ue8m0-deepgemm-quantize",
"overlap_grad_reduce": "compute-comm-overlap",
"overlap_param_gather": "compute-comm-overlap",
"zero_bubble": "zero-bubble-schedule",
"ZBVZeroBubble": "zero-bubble-schedule",
"pipeline_parallel": "pipeline-v-schedule",
"DeviceMesh": "device-mesh-topology",

# keyword -> hardware_feature tag (KW_TO_TAGS)
"fp8": "fp8",
"float8": "fp8",
"e4m3": "fp8",
"e5m2": "fp8",
"block_scaling": "block-scale",
"E8M0": "e8m0-scale-format",
"MXFP8": "block-scale",
```

### S.6 PR Search Keywords (for candidate ledger)

```yaml
keywords_used:
  - fsdp
  - fsdp2
  - fully_shard
  - tensor_parallel
  - pipeline_parallel
  - expert_parallel
  - context_parallel
  - ulysses
  - DeviceMesh
  - fp8
  - mxfp8
  - float8
  - blockwise
  - precision
  - mixed_precision
  - gradient_checkpointing
  - activation_checkpoint
  - ppo
  - grpo
  - rl_trainer
  - rollout
  - weight_update
  - offload
  - onload
  - torch_memory_saver
  - staleness
  - async
  - overlap
  - zero_bubble
  - megatron
  - archon
  - tree_attention
  - triton
  - optimizer
  - adam_bf16
  - kahan
  - per_layer_optim
  - nccl
  - allreduce
  - allgather
  - reduce_scatter
  - alltoall
  - profiler
  - perf_tracer
  - wandb
```

### S.7 Inclusion Policy Lane

```yaml
training-orchestration:
  description: |
    AReaL is an RL training orchestration framework. Capture PRs touching
    parallelism strategies, communication patterns, FP8 precision, memory
    management, or RL-specific training infrastructure. Skip pure docs,
    workflow recipes, agent integrations, and CI-only changes.
  capture_criteria:
    - changed_paths_match:
        - "areal/engine/**"
        - "areal/trainer/**"
        - "areal/models/**"
        - "areal/infra/controller/**"
        - "areal/infra/dist_rollout.py"
        - "areal/infra/workflow_executor.py"
        - "areal/infra/staleness_manager.py"
        - "areal/experimental/engine/**"
        - "areal/experimental/weight_update/**"
        - "areal/experimental/models/archon/**"
        - "areal/utils/perf_tracer.py"
        - "areal/utils/functional/**"
    - title_contains_any:
        - fsdp
        - tensor_parallel
        - pipeline_parallel
        - expert_parallel
        - context_parallel
        - parallelism
        - fp8
        - mxfp8
        - precision
        - mixed_precision
        - gradient_checkpoint
        - activation_checkpoint
        - ppo
        - grpo
        - rl_trainer
        - rollout
        - weight_update
        - offload
        - overlap
        - zero_bubble
        - megatron
        - archon
        - nccl
        - allreduce
        - allgather
        - ulysses
        - tree_attention
        - triton
        - optimizer
        - profiler
        - perf_trace
  skip_criteria:
    - changed_paths_match_only:
        - "docs/**"
        - "blog/**"
        - "examples/**"
        - "tests/**"
        - "assets/**"
        - "*.md"
        - ".github/**"
        - "notebook/**"
        - "benchmark/**"
        - "areal/workflow/**"
        - "areal/reward/**"
        - "areal/dataset/**"
    - pure_config_only: true
```

### S.8 Schema Extensions (if any)

Proposed new optional frontmatter fields for Wiki pages from AReaL and similar RL
training frameworks:

```yaml
# For training/ and parallelism/ pages
scope: training              # "training" | "inference" | "both"
rl_specific: true            # Whether this technique is specific to RL training (not SFT/pretraining)
parallelism_dimensions:      # Which parallelism dimensions are involved
  - dp
  - tp
  - pp
  - cp
  - ep
  - rl-async
communication_pattern: collective  # "collective" | "p2p" | "broadcast" | "alltoall" | "rpc"
engine_backends:             # Which AReaL engine backends support this
  - fsdp
  - megatron
  - archon
```

### S.9 Hardware Features Relevant to This Library's Training Workloads

| Hardware Feature                        | Inference Relevance | Training Relevance | Specific Impact on AReaL                                                                                                                     |
| --------------------------------------- | ------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| NVLink 5 (1.8 TB/s)                     | Partial             | Core               | Doubles gradient AllReduce/ReduceScatter bandwidth; critical for FSDP2 AllGather parameter collection and cross-engine weight sync broadcast |
| NVSwitch 4 (NVL72, 130 TB/s)            | Partial             | Core               | Enables efficient AllToAll for EP token dispatch and Ulysses SP at scale                                                                     |
| NVLink-SHARP FP8                        | No                  | Core               | Would reduce AllReduce bandwidth by 4x for gradient sync (not yet used by AReaL)                                                             |
| Symmetric Memory (9x latency reduction) | Partial             | Core               | Would accelerate small-message AllReduce for RL advantage normalization (not yet used; NVLS excluded in deterministic mode)                  |
| Copy Engine (zero-SM transfer)          | Partial             | Core               | Would enable zero-SM parameter AllGather during forward pass, freeing SMs for compute (not yet used)                                         |
| MXFP8 hardware (Blackwell)              | Yes                 | Core               | AReaL supports MXFP8 via TE recipe and UE8M0 quantization; eliminates software block scaling overhead, 50-60% BF16 speedup                   |
| 192 MB L2 Cache (3.8x vs H100)          | Beneficial          | Beneficial         | Improves activation recomputation performance for gradient checkpointing                                                                     |
| 192 GB HBM3e @ 8 TB/s                   | Core                | Core               | Larger models per GPU; critical for colocated mode where inference KV-cache and training parameters share GPU memory                         |

### S.10 Upstream/Downstream Dependencies to Also Track

| Slug                     | GitHub URL                     | Relationship             | Justification                                                                                                          |
| ------------------------ | ------------------------------ | ------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `megatron-core`          | NVIDIA/Megatron-LM             | kernel-provider          | Provides TP/PP GEMM, FP8 TE layers, distributed optimizer, pipeline scheduling; AReaL's Megatron backend depends on it |
| `torchao`                | pytorch/ao                     | kernel-provider          | Provides blockwise FP8 GEMM via `fp8_blockwise_mm`; used in Archon FP8 path                                            |
| `flash-attn`             | Dao-AILab/flash-attention      | kernel-provider          | Provides FlashAttention-2/3/4 forward/backward; core attention kernel for all AReaL backends                           |
| `sglang`                 | sgl-project/sglang             | runtime-dependency       | Inference backend for RL rollout generation; PagedAttention, continuous batching                                       |
| `vllm`                   | vllm-project/vllm              | runtime-dependency       | Alternative inference backend for RL rollout generation                                                                |
| `flash-linear-attention` | fla-org/flash-linear-attention | kernel-provider          | Provides Triton-based chunked linear attention for Qwen3.5 hybrid models                                               |
| `transformer-engine`     | NVIDIA/TransformerEngine       | kernel-provider          | FP8 Linear/Norm, multi_tensor ops; used transitively through Megatron-Core                                             |
| `torch_memory_saver`     | SagerNet/torch_memory_saver    | runtime-dependency       | LD_PRELOAD CUDA memory hook enabling colocated inference-training GPU sharing                                          |
| `nccl`                   | NVIDIA/nccl                    | communication-backend    | All GPU-to-GPU collective communication; cross-engine weight sync groups                                               |
| `deepep`                 | deepseek-ai/DeepEP             | kernel-provider (future) | MoE expert parallel via NVSHMEM; flag exists (`moe_enable_deepep`) but not fully wired                                 |
