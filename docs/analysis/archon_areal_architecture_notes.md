# Archon 架构设计与实现解析

> TL;DR：Archon 是一个非常具有潜力的框架。  
> AReaL 仓库：[https://github.com/inclusionAI/AReaL](https://github.com/inclusionAI/AReaL)  
> AReaL v1.0.0 Release：[https://github.com/inclusionAI/AReaL/releases/tag/v1.0.0](https://github.com/inclusionAI/AReaL/releases/tag/v1.0.0)

[TOC]

---

# 1. 简介

AReaL 在前几天刚刚放出自己的 `1.0.0` 版本，支持了包括自由 agentic 训练、OpenClaw 接入和大量原生功能。其中一个非常特殊的更新是正式支持了 **Archon**，官方称其为：

> A fully working, PyTorch-native 5D parallel training engine.

本文旨在（也是方便自己学习）展示 Archon 的优雅设计哲学，以及深入浅出地理解 Archon 的工作原理。

Archon 是一个基于 **PyTorch 原生分布式 API** 构建的大规模 Transformer 模型训练框架，专为百亿至千亿参数规模的语言模型训练设计。它继承并扩展了 **TorchTitan**（PyTorch 团队推出的训练框架，设计也极其优雅，推荐源码阅读）的设计哲学，包含以下特点：

- 纯 PyTorch 原生实现
- 深度集成 `torch.distributed`、`DeviceMesh`、`DTensor` 等现代分布式原语
- 针对 **Qwen2** 和 **Qwen3（含 MoE 架构）** 模型进行了深度优化
- 支持多种先进并行策略的组合使用

在本文中，我们将：

1. 在 **Section 2** 中分析 Archon 的架构设计：为什么会以这样的结构设计。
2. 在 **Section 3** 中深入研究各个核心模块的实现细节。
3. 在 **Section 4** 中探讨关键技术创新点。
4. 在 **Section 5** 中总结代码质量与工程实践。

---

# 2. 架构设计哲学

## 2.1 分层解耦的并行策略架构

Archon 的核心设计思想是：**将不同的并行策略解耦为独立的可组合模块**，而非传统框架中的紧耦合实现（没法对齐，超级无敌爆怒）。

```text
┌─────────────────────────────────────────────────────────────┐
│                      模型层 (Model Layer)                   │
│              Qwen2Model / Qwen3Model (含 MoE)              │
├─────────────────────────────────────────────────────────────┤
│                   并行化层 (Parallelization)                │
│   TP (Tensor Parallel) + CP (Ulysses SP) + AC (Checkpoint) │
├─────────────────────────────────────────────────────────────┤
│                  流水线层 (Pipeline Parallel)               │
│  1F1B / Interleaved1F1B / InterleavedZeroBubble / ZBV...   │
├─────────────────────────────────────────────────────────────┤
│                   数据并行层 (Data Parallel)                │
│             FSDP (Fully Sharded Data Parallel)              │
├─────────────────────────────────────────────────────────────┤
│                   专家并行层 (Expert Parallel)              │
│     EP (Expert Parallel) + ETP (Expert Tensor Parallel)    │
└─────────────────────────────────────────────────────────────┘
```

**核心洞察**：每一层都通过标准化接口与上下层交互，使得任意组合成为可能。例如，你可以独立选择是否使用 PP、TP、CP，它们的组合不需要彼此感知。

## 2.2 整体调用分析

具体来说，当我们使用 Archon 的时候，会遵循以下 pipeline（长代码预警）：

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           阶段 0: 包导入与模型注册                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  from areal.experimental.models.archon import ...                           │
│         │                                                                   │
│         ▼                                                                   │
│  archon/__init__.py 执行                                                    │
│         │                                                                   │
│         ├── from .qwen2 import spec ──► qwen2/spec.py                       │
│         │                                 │                                 │
│         │                                 ▼                                 │
│         │                    QWEN2_SPEC = ModelSpec(...)                    │
│         │                    register_model_spec(QWEN2_SPEC)                │
│         │                                 │                                 │
│         │                                 ▼                                 │
│         │                    _MODEL_SPECS["qwen2"] = QWEN2_SPEC             │
│         │                                                                   │
│         └── from .qwen3 import spec ──► qwen3/spec.py (同上)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      阶段 1: 训练脚本初始化 (用户代码)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  # 假设的训练脚本入口                                                        │
│                                                                             │
│  1. 解析配置 & 初始化分布式                                                  │
│     torch.distributed.init_process_group(...)                               │
│     world_size = dist.get_world_size()                                      │
│     rank = dist.get_rank()                                                  │
│                                                                             │
│  2. 创建并行维度配置                                                         │
│     parallel_dims = ArchonParallelDims(                                     │
│         dp_shard=-1,      # 自动计算: world_size/(tp*cp*pp)                 │
│         tp=2,                                                               │
│         cp=2,             # Ulysses SP                                      │
│         pp=4,             # 流水线并行度                                     │
│         ep=2,             # 专家并行度 (MoE 时)                              │
│         etp=2,            # 专家 TP (等于 tp)                                │
│         world_size=32,                                                      │
│         device_type="cuda",                                                 │
│     )                                                                       │
│                                                                             │
│     __post_init__() 执行：                                                  │
│         - 计算 dp_shard = 32/(2*2*4) = 2                                    │
│         - 验证: 2*2*2*4 = 32 ✓                                              │
│         - 验证 EP 约束 (EP%CP==0, dp_shard*cp%EP==0 等)                     │
│         - lazy build mesh (首次访问时构建)                                  │
│                                                                             │
│  3. 构建 Device Mesh                                                        │
│     mesh = parallel_dims.world_mesh  # 触发 build_mesh()                    │
│         │                                                                   │
│         ├── EP>1 ? ── Yes ──► _build_mesh_with_ep()                         │
│         │                      5D mesh: (pp, dp_shard_mod_ep,               │
│         │                               dp_shard_in_ep, cp, tp)             │
│         │                                                                   │
│         └── 创建子 mesh: ep, ep_tp, dp_shard_cp, pp_cp_tp...                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         阶段 2: 模型创建与初始化                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  # 方式 A: 从 HuggingFace Config 创建                                       │
│  from transformers import AutoConfig                                        │
│  hf_config = AutoConfig.from_pretrained("Qwen/Qwen2-7B")                    │
│                                                                             │
│  model_spec = get_model_spec(hf_config.model_type)  # "qwen2"               │
│         │                                                                   │
│         ▼                                                                   │
│  _MODEL_SPECS["qwen2"] ──► QWEN2_SPEC                                       │
│         │                                                                   │
│         ├── model_args_class: Qwen2ModelArgs                                │
│         │       from_hf_config(hf_config)                                   │
│         │             │                                                     │
│         │             ▼                                                     │
│         │       创建 Qwen2ModelArgs 实例                                     │
│         │       (dim, n_heads, n_layers, ...)                               │
│         │                                                                   │
│         └── model_class: Qwen2Model                                         │
│                 __init__(model_args)                                        │
│                       │                                                     │
│                       ▼                                                     │
│                 模块创建流程：                                               │
│                 ┌─────────────────┐                                         │
│                 │ tok_embeddings  │                                         │
│                 │    (Embedding)  │                                         │
│                 ├─────────────────┤                                         │
│                 │ layers          │ ◄── 关键设计！                          │
│                 │ (ModuleDict)    │                                         │
│                 │ ├── "0": TransformerBlock                                │
│                 │ ├── "1": TransformerBlock                                │
│                 │ └── ...                                                   │
│                 ├─────────────────┤                                         │
│                 │ norm (RMSNorm)  │                                         │
│                 ├─────────────────┤                                         │
│                 │ output (Linear) │                                         │
│                 └─────────────────┘                                         │
│                                                                             │
│  # 方式 B: 直接创建 (非 HF)                                                  │
│  model_args = Qwen2ModelArgs(dim=4096, n_heads=32, n_layers=32, ...)        │
│  model = Qwen2Model(model_args)                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      阶段 3: 流水线并行分割 (PP 阶段)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  if parallel_dims.pp_enabled:                                               │
│      stages, model_parts, has_first, has_last = pipeline_llm(               │
│          model,                                                             │
│          device,                                                            │
│          parallel_dims,                                                     │
│          archon_config,                                                     │
│          parallelize_fn=parallelize_qwen2,                                  │
│      )                                                                      │
│         │                                                                   │
│         ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │ pipeline_llm() 内部执行流程:                                       │      │
│  │                                                                    │      │
│  │ 1. 确定虚拟 stage 数                                                │      │
│  │    if pp_schedule == "1F1B":                                      │      │
│  │        stages_per_rank = 1                                         │      │
│  │    else (Interleaved1F1B / ZBV...):                                │      │
│  │        stages_per_rank = 2  # 默认                                 │      │
│  │                                                                    │      │
│  │    num_virtual_stages = pp_degree * stages_per_rank                │      │
│  │    # 例: pp=4, stages_per_rank=2 ──► 8 个虚拟 stages               │      │
│  │                                                                    │      │
│  │ 2. 生成分配方案                                                     │      │
│  │    module_names_per_stage = generate_llm_fqn_per_model_part(       │      │
│  │        num_stages=8, num_layers=32                                 │      │
│  │    )                                                                │      │
│  │    # 例: [['tok_embeddings', 'layers.0', 'layers.1', 'layers.2'],  │      │
│  │    #      ['layers.3', 'layers.4', 'layers.5'], ...]               │      │
│  │                                                                    │      │
│  │ 3. 分割模型                                                         │      │
│  │    stages, model_parts = pipeline_module_split(                    │      │
│  │        whole_model,                                                │      │
│  │        pp_mesh,                                                    │      │
│  │        pp_schedule,                                                │      │
│  │        device,                                                     │      │
│  │        module_names_per_stage,                                     │      │
│  │    )                                                                │      │
│  │                                                                    │      │
│  │    对于每个本 rank 负责的 stage：                                    │      │
│  │    ┌───────────────────────────────────────────────────────────┐    │      │
│  │    │ _build_stage_from_modules():                             │    │      │
│  │    │   1. deep_copy(whole_model) ──► 独立模型实例             │    │      │
│  │    │   2. 删除不属于本 stage 的 layers:                        │    │      │
│  │    │      for key in layer_keys:                               │    │      │
│  │    │          if key not in layers_to_keep:                    │    │      │
│  │    │              del module_value[key]                        │    │      │
│  │    │   3. 将非本 stage 模块设为 None:                          │    │      │
│  │    │      if module_name not in modules_to_keep:               │    │      │
│  │    │          setattr(model, module_name, None)                │    │      │
│  │    │   4. 创建 PipelineStage(model_part, stage_idx, ...)       │    │      │
│  │    └───────────────────────────────────────────────────────────┘    │      │
│  │                                                                    │      │
│  │ 4. 对每个 model_part 应用并行化                                     │      │
│  │    for i, m in enumerate(model_parts):                            │      │
│  │        model_parts[i] = parallelize_qwen2(m, parallel_dims, ...)  │      │
│  └────────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  注意：PP 后，模型被分割成多个 parts，每个 rank 持有 1 个或多个 parts。      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  阶段 4: 并行化应用 (核心: parallelize_qwen2)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  def parallelize_qwen3(model, parallel_dims, ...):                          │
│      # 严格按序应用并行策略                                                 │
│      # 注意: Qwen2 无 MoE，跳过步骤 2；Qwen3 含 MoE，完整执行             │
│                                                                             │
│      步骤 1: Tensor Parallelism (TP, 非 MoE 部分)                           │
│      if parallel_dims.tp_enabled:                                           │
│          apply_non_moe_tp(model, tp_mesh, loss_parallel=True)               │
│          validate_tp_constraints(model_args, tp_size)                       │
│          parallelize_module() 对每层应用：                                   │
│              wq/wk/wv: ColwiseParallel                                      │
│              wo:       RowwiseParallel                                      │
│              feed_forward.w1/w3: ColwiseParallel (仅非 MoE 层)              │
│              feed_forward.w2: RowwiseParallel   (仅非 MoE 层)               │
│              attention_norm/ffn_norm: SequenceParallel                      │
│                                                                             │
│      步骤 2: Expert Parallelism (MoE 模型, EP + MoE TP)                     │
│      # 必须在 FSDP 之前：distribute_module 需要在 FSDP 包裹前注册 hooks    │
│      if tp_mesh is not None or ep_mesh is not None:                         │
│          apply_moe_ep_tp(model, tp_mesh, ep_mesh, etp, ep_tp_mesh)          │
│          # EP: ExpertParallel._apply() → distribute_module()                │
│          #     注册 _token_dispatch (forward_pre) 和 _token_combine (hook)  │
│          # ETP=TP: ExpertTensorParallel 2D sharding [Shard(0), Shard(1/2)]  │
│          # ETP=1:  ExpertParallel Shard(0), TP 维度被 EP 借用               │
│                                                                             │
│      步骤 3: Context Parallelism (CP / Ulysses SP)                          │
│      if parallel_dims.cp_enabled:                                           │
│          apply_cp(model, cp_group, tp_size)                                 │
│          for layer in model.layers.values():                                │
│              layer.attention.set_cp_group(cp_group)                         │
│                                                                             │
│      步骤 4: Activation Checkpointing (AC)                                  │
│      if ac_config and ac_config.mode != "none":                            │
│          apply_ac(model, ac_config, ...)                                    │
│          for layer_id, block in model.layers.items():                       │
│              if ac_config.mode == "full":                                  │
│                  block = _apply_full_ac(block)                              │
│              elif ac_config.mode == "selective":                           │
│                  if selective_ac_option == "op":                           │
│                      block = _apply_op_sac(block, op_sac_save_list)         │
│                  else:                                                      │
│                      block = _apply_layer_sac(block, ac_config)             │
│              model.layers.register_module(layer_id, block)                  │
│                                                                             │
│      步骤 5: torch.compile                                                  │
│      if enable_compile:                                                     │
│          apply_compile(model)                                               │
│          for name, block in model.layers.items():                           │
│              model.layers[name] = torch.compile(                            │
│                  block, backend="inductor", fullgraph=True                 │
│              )                                                              │
│                                                                             │
│      步骤 6: FSDP                                                           │
│      dp_mesh = parallel_dims.get_mesh("dp_shard_cp")                        │
│      if dp_mesh is not None:                                                │
│          apply_fsdp(model, dp_mesh, ...)                                    │
│          fully_shard(model.tok_embeddings, ...)                             │
│          for block in model.layers.values():                                │
│              fully_shard(block, reshard_after_forward=not pp_enabled)       │
│          fully_shard([model.norm, model.output], ...)                       │
│          fully_shard(model, ...)                                            │
│                                                                             │
│          # distribute_module 自动注册 hooks:                                │
│          # - forward_pre: _token_dispatch (All-to-All)                      │
│          # - forward:     _token_combine (All-to-All 反向)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       阶段 5: 权重加载与初始化                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  # 方式 A: 随机初始化                                                        │
│  model.init_weights()                                                       │
│  model.init_buffers(buffer_device)  # 预计算 rope_cache 等 buffer          │
│                                                                             │
│  # 方式 B: 从 HuggingFace Checkpoint 加载                                    │
│  adapter = model_spec.state_dict_adapter_class(hf_config, hf_path)          │
│         │                                                                   │
│         ▼                                                                   │
│  reader = adapter.get_hf_storage_reader(hf_path)                            │
│  adapter.from_hf(hf_state_dict)  # FQN 映射转换                              │
│         │                                                                   │
│         ▼                                                                   │
│  archon_state_dict = {                                                      │
│      "tok_embeddings.weight": ...,                                         │
│      "layers.0.attention.wq.weight": ...,                                  │
│      "layers.0.feed_forward.w1.weight": ...,                               │
│      ...                                                                    │
│  }                                                                          │
│                                                                             │
│  # 如果是 EP/TP 模型，权重已经是 DTensor，需确保正确分布                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         阶段 6: 训练循环启动                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  # 创建 Pipeline Schedule                                                   │
│  schedule = build_pipeline_schedule(                                        │
│      stages,                                                                │
│      pp_schedule,                                                           │
│      n_microbatches,                                                        │
│      loss_fn=loss_fn,                                                       │
│  )                                                                          │
│  schedule.step()  # 由 PyTorch Pipeline 驱动                                │
│                                                                             │
│  # 或者非 PP 场景                                                            │
│  for batch in dataloader:                                                   │
│      inputs, labels = ulysses_slice_inputs(                                 │
│          inputs, labels, cp_rank, cp_size                                  │
│      )                                                                      │
│                                                                             │
│      output = model(tokens, positions, cu_seqlens, max_seqlen)             │
│      output = ulysses_gather_output(output, cp_group)                       │
│                                                                             │
│      loss = loss_fn(output, labels)                                         │
│      loss.backward()                                                        │
│                                                                             │
│      optimizer.step()                                                       │
│      optimizer.zero_grad()                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2.2.1 初始化调用链（Initialization Chain）

```text
用户代码
    │
    ▼
import areal.experimental.models.archon
    │
    ├──► base.py: BaseModelArgs, BaseStateDictAdapter, BaseArchonModel
    │
    ├──► parallel_dims.py: ArchonParallelDims
    │       │
    │       └──► world_size 验证: dp_shard * tp * cp * pp == world_size
    │
    ├──► pipeline_parallel.py: pipeline_llm, pipeline_module_split
    │
    ├──► expert_parallel.py: ExpertParallel, ExpertTensorParallel
    │
    ├──► model_spec.py: ModelSpec, register_model_spec
    │       │
    │       ├──► qwen2/spec.py ──► 注册 QWEN2_SPEC
    │       └──► qwen3/spec.py ──► 注册 QWEN3_SPEC
    │
    └──► (其他模块)
```

## 2.2.2 并行化调用链（Parallelization Chain）

```text
parallelize_qwen3(model, parallel_dims, ...)
    │
    ├──► apply_non_moe_tp(model, tp_mesh)          # 步骤 1
    │       │
    │       ├──► validate_tp_constraints()  # 头数整除验证
    │       │
    │       └──► parallelize_module()  # PyTorch TP
    │               │
    │               ├── ColwiseParallel (wq, wk, wv, w1, w3 非MoE层)
    │               └── RowwiseParallel (wo, w2 非MoE层)
    │
    ├──► apply_moe_ep_tp(model, tp_mesh, ep_mesh)  # 步骤 2 (MoE 模型)
    │       │
    │       ├──► ExpertTensorParallel (etp=tp)
    │       │       ���──► distribute_module() + 2D sharding
    │       └──► ExpertParallel (etp=1)
    │               └──► distribute_module() + Shard(0)
    │
    ├──► apply_cp(model, cp_group)                  # 步骤 3
    │       │
    │       └──► layer.attention.set_cp_group(cp_group)
    │               │
    │               └──► 启用 Ulysses All-to-All
    │
    ├──► apply_ac(model, ac_config)                 # 步骤 4
    │       │
    │       ├──► _apply_full_ac()   # mode=full
    │       └──► _apply_op_sac()    # mode=selective, op-level
    │
    ├──► apply_compile(model)                       # 步骤 5
    │       │
    │       └──► torch.compile(block, fullgraph=True)  # 逐层编译
    │
    └──► apply_fsdp(model, dp_mesh)                 # 步骤 6
            │
            └──► fully_shard()  # FSDP2 逐层包裹
```

## 2.2.3 专家并行调用链（MoE + EP Chain）

```text
# MoE 前向 (单节点)
MoE.forward(x)
    │
    ├──► router(x, expert_bias) ──► top_scores, selected_indices
    │
    ├──► reorderer(top_scores, selected_indices)
    │       │
    │       └──► (top_scores_sorted, token_indices_sorted, num_per_expert)
    │
    ├──► routed_input = x_flat[token_indices_sorted // top_k]
    │
    ├──► experts(routed_input, num_per_expert) ──► routed_output
    │       │
    │       └──► _run_experts_grouped_mm()  # 或 for-loop
    │
    ├──► shared_experts(x_flat)  # 可选
    │
    └──► combine & reshape ──► output

# 多节点 EP 场景
experts.forward()
    │
    ├──► 【Pre-hook】ExpertParallel._token_dispatch()
    │       │
    │       ├──► all_to_all_single(num_tokens_per_expert)
    │       ├──► all_to_all_single_autograd(routed_input)
    │       └──► _permute()  # 对齐 padding
    │
    ├──► 本地 expert 计算
    │
    └──► 【Hook】ExpertParallel._token_combine()
            │
            └──► all_to_all_single_autograd(output)  # 反向 all-to-all
```

## 2.2.4 状态流转示意图

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         模型状态流转图                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [1] 初始状态 (meta device)                                             │
│       ┌─────────────────────┐                                           │
│       │ Qwen2Model on meta  │                                           │
│       │ - layers: ModuleDict│                                           │
│       │ - 参数: 未初始化     │                                           │
│       └─────────────────────┘                                           │
│                  │                                                      │
│                  ▼                                                      │
│  [2] PP 分割后                                                           │
│       ┌─────────────────────┐  ┌─────────────────────┐                  │
│       │ Stage 0 (Rank 0)    │  │ Stage 1 (Rank 1)    │                  │
│       │ - tok_embeddings    │  │ - layers.8-15       │                  │
│       │ - layers.0-7        │  │ - norm, output      │                  │
│       │ (deep copy)         │  │ (deep copy)         │                  │
│       └─────────────────────┘  └─────────────────────┘                  │
│                  │                                                      │
│                  ▼                                                      │
│  [3] TP 应用后                                                           │
│       参数变为 DTensor                                                   │
│       ┌────────────────────────────────────────┐                        │
│       │ wq: DTensor(local_tensor, device_mesh  │                        │
│       │      =tp_mesh, placements=[Shard(0)])  │                        │
│       └────────────────────────────────────────┘                        │
│                  │                                                      │
│                  ▼                                                      │
│  [4] EP 应用后 (MoE)                                                     │
│       ┌────────────────────────────────────────┐                        │
│       │ experts.w1: DTensor(                   │                        │
│       │   local_tensor,                        │                        │
│       │   device_mesh=ep_mesh,                 │                        │
│       │   placements=[Shard(0)]                │  # ETP=1              │
│       │   # 或 [Shard(0), Shard(1)]            │  # ETP=TP             │
│       │ )                                      │                        │
│       └────────────────────────────────────────┘                        │
│                  │                                                      │
│                  ▼                                                      │
│  [5] FSDP 应用后                                                         │
│       所有参数进一步被 FSDP 管理                                          │
│       ┌────────────────────────────────────────┐                        │
│       │ FSDP(FSDP(FSDP(...(Qwen2Model)...)))   │                        │
│       │ 每个 block 独立 FSDP 包裹              │                        │
│       └────────────────────────────────────────┘                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2.2.5 内存与通信图谱

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            内存 & 通信模式图                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    All-to-All (Ulysses)     ┌─────────────┐                │
│  │   CP Rank   │◄───────────────────────────►│   CP Rank   │                │
│  │     0       │   交换: [seq, heads] ↔ [heads, seq]      │     1         │
│  │  (seq_local)│                              │  (seq_local)│                │
│  └─────────────┘                              └─────────────┘                │
│         │                                            │                        │
│         │ All-Gather (TP backward)                   │                        │
│         ▼                                            ▼                        │
│  ┌─────────────┐                              ┌─────────────┐                │
│  │   TP Rank   │                              │   TP Rank   │                │
│  │     0       │                              │     1       │                │
│  │ (hidden/2)  │                              │ (hidden/2)  │                │
│  └─────────────┘                              └─────────────┘                │
│         │                                            │                        │
│         │ Pipeline Send/Recv                         │                        │
│         ▼                                            ▼                        │
│  ┌─────────────┐                              ┌─────────────┐                │
│  │   PP Stage  │◄────────────────────────────►│   PP Stage  │                │
│  │      0      │      activation 传递         │      1      │                │
│  │  (layers    │                              │  (layers    │                │
│  │   0-7)      │                              │   8-15)     │                │
│  └─────────────┘                              └─────────────┘                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      All-to-All (EP)                               │    │
│  │  ┌─────────┐    tokens     ┌─────────┐    tokens     ┌─────────┐   │    │
│  │  │ EP Rank │◄──dispatch──►│ EP Rank │◄──dispatch──►│ EP Rank │   │    │
│  │  │    0    │               │    1    │               │    2    │   │    │
│  │  │ E0, E1  │◄──combine────│ E2, E3  │◄──combine────│ E4, E5  │   │    │
│  │  └─────────┘               └─────────┘               └─────────┘   │    │
│  │       ▲                                                    ▲         │    │
│  │       └───────────────── EP Mesh ─────────────────────────┘         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2.2.6 启动流程关键决策点

| 阶段 | 关键决策 | 影响 |
| --- | --- | --- |
| Mesh 构建 | EP 是否借用 TP | 决定 EP mesh 维度计算方式 |
| PP 分割 | Virtual stages 数量 | 影响显存占用与 bubble 大小 |
| TP 应用 | Loss parallel 模式 | 决定输出层是否保持 sharded |
| AC 策略 | op-level vs layer-level | 精细控制内存 vs 实现简单性 |
| Compile 时机 | 是否在 AC 之后、FSDP 之前 | 决定是否能捕获完整计算图 |
| FSDP 策略 | `reshard_after_forward` | PP 场景下通常禁用以减少通信 |

---

# 3. 关键设计决策

| 设计决策 | 方案选择 | 技术原理 |
| --- | --- | --- |
| 序列并行策略 | Ulysses（All-to-All） | 相较于 Ring Attention，实现更简单，通信模式规整 |
| 专家并行通信 | All-to-All（非 All-Gather） | 直接对齐 token 到目标 expert，避免冗余计算 |
| 流水线调度 | 虚拟阶段（Virtual Stages） | 支持单 rank 多 stage，实现 Interleaved 1F1B |
| 权重切分 | DTensor 原生支持 | 利用 PyTorch 2.0+ 的张量并行原语 |
| 激活检查点 | Op-level 选择性 AC | 精细控制内存与计算的 trade-off |
| 模型层存储 | ModuleDict（非 ModuleList） | 支持流水线阶段的灵活切分与删除 |

---

# 4. 核心模块深度解析

## 4.1 ModelSpec 注册机制（`model_spec.py`）

Archon 采用 **自注册模式** 管理模型定义，这是一个优雅的设计模式：

```python
@dataclass
class ModelSpec:
    name: str
    model_class: type[nn.Module]
    model_args_class: type[BaseModelArgs]
    state_dict_adapter_class: type[...]
    parallelize_fn: ParallelizeFn
    supported_model_types: frozenset[str]
    pipelining_fn: PipeliningFn | None = None
```

**设计深意**：

1. **解耦模型定义与框架核心**：新增模型只需实现接口并注册，无需修改框架。
2. **统一配置入口**：`from_hf_config()` 工厂方法实现 HuggingFace Config 到 Archon Args 的无缝转换。
3. **状态字典适配**：自动处理 HF 与 Archon 权重命名差异，支持 MoE expert 权重拆分/合并。

## 4.1.1 5D 并行维度管理（`parallel_dims.py`）

```python
@dataclass
class ArchonParallelDims:
    dp_shard: int = -1   # FSDP 数据并行
    cp: int = 1          # Ulysses 序列并行
    tp: int = 1          # 张量并行
    pp: int = 1          # 流水线并行
    ep: int = 1          # 专家并行
    etp: int = 1         # 专家张量并行
```

**Device Mesh 构建策略**：

| EP 模式 | Mesh 维度 | 说明 |
| --- | --- | --- |
| `EP=1` | `(pp, dp_shard, cp, tp)` | 4D mesh，标准配置 |
| `EP>1, ETP=1` | `(pp, dp_shard_mod_ep, dp_shard_in_ep, cp, tp)` | 5D mesh，EP 借用 TP |
| `EP>1, ETP=TP` | `(pp, dp_shard_mod_ep, dp_shard_in_ep, cp, tp)` | 5D mesh，EP/TP 独立 |

EP 与 TP 的资源分配是动态协商的：

- 当 `etp=1` 时，TP 维度被 EP “借用” 用于 token dispatch。
- 当 `etp=tp` 时，EP 与 TP 正交，expert 权重使用 2D sharding：`[Shard(0), Shard(1/2)]`。

## 4.1.2 Mesh 构建核心逻辑

```python
def _build_mesh_with_ep(self) -> DeviceMesh:
    """Build mesh when EP is enabled.

    Handles both etp=1 and etp=tp cases:
    - etp=1: EP borrows from dp_shard_in_ep * cp * tp
    - etp=tp: EP borrows from dp_shard_in_ep * cp only (tp independent)
    """
    if self.etp == self.tp:
        dp_shard_mod_ep = self.dp_shard * self.cp // self.ep
        dp_shard_in_ep = self.ep // self.cp
    else:
        dp_shard_mod_ep = self.dp_shard * self.cp * self.tp // self.ep
        dp_shard_in_ep = self.ep // (self.cp * self.tp)
```

## 4.2 流水线并行的实现（`pipeline_parallel.py`）

Archon 的流水线实现体现了 **虚拟阶段（Virtual Stages）** 的设计理念：

```python
def generate_llm_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    first_stage_less_layers: int = 1,
    last_stage_less_layers: int = 1,
    is_critic: bool = False,
) -> list[list[str]]:
```

**负载均衡算法**：

1. 计算有效层数：`effective_layers = num_layers + first_stage_less_layers + last_stage_less_layers`
2. 均匀分配有效层数到各 stage
3. 首 stage 减去 embedding 权重，尾 stage 减去 output 权重

**调度策略矩阵**：

| 调度策略 | Stage / Rank | 特点 |
| --- | --- | --- |
| `1F1B` | 1 | 标准 1F1B，每 rank 一个 stage |
| `Interleaved1F1B` | `num_stages / pp_degree` | 循环分配，rank 0 持有 stage 0,4,8... |
| `InterleavedZeroBubble` | 同上 | 零气泡优化版本 |
| `ZBVZeroBubble` | 2 | V 形分配，rank 0 持有 stage 0 和 N-1 |

**代码实现亮点**：

```python
def _get_stage_indices() -> tuple[int, ...]:
    """Get stage indices for this rank based on schedule style.

    Examples (pp_degree=4, num_stages=8):
        1F1B:                  Rank 0->(0,), Rank 1->(1,), ...
        Interleaved1F1B:       Rank 0->(0,4), Rank 1->(1,5), Rank 2->(2,6), Rank 3->(3,7)
        InterleavedZeroBubble: (same loop-style assignment as Interleaved1F1B)
        ZBVZeroBubble:         Rank 0->(0,7), Rank 1->(1,6), Rank 2->(2,5), Rank 3->(3,4)
    """
    if num_stages % pp_degree != 0:
        raise ValueError(
            f"num_stages ({num_stages}) must be evenly divisible by pp_degree ({pp_degree})"
        )

    stages_per_rank = num_stages // pp_degree
    schedule_class = get_schedule_class(pp_schedule)
    v_style_schedules = (ScheduleZBVZeroBubble, ScheduleDualPipeV)
    style = "v" if schedule_class in v_style_schedules else "loop"

    if style == "v":
        if stages_per_rank != 2:
            raise ValueError(
                f"V-style schedules require exactly 2 stages per rank, got {stages_per_rank}"
            )
        stage_v_pairs = list(
            zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1))
        )
        return stage_v_pairs[pp_rank]
    else:
        return tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
```

## 4.2.1 ModuleDict 用于流水线切分

Archon 使用 `ModuleDict` 而非 `ModuleList` 存储 Transformer layers：

```python
self.layers = nn.ModuleDict()
for layer_id in range(n_layers):
    self.layers[str(layer_id)] = TransformerBlock(...)

# 流水线切分时
for layer_key in list(module_value.keys()):
    if layer_key not in layers_to_keep:
        del module_value[layer_key]
```

**优势**：

- 支持非连续的层分配（如 Interleaved schedule）
- 保留原始 FQN（`layers.0`、`layers.1` ...）便于状态字典转换

## 4.3 Ulysses 序列并行（`ulysses.py`）

与 Ring Attention 不同，Archon 采用 **Ulysses 序列并行**。

**通信模式对比**：

| 特性 | Ring Attention | Ulysses（Archon） |
| --- | --- | --- |
| 切分维度 | 序列维度（seq） | 注意力头维度（heads） |
| 通信模式 | P2P 环形通信 | All-to-All 集体通信 |
| 计算特性 | 块级计算 | 全注意力计算 |
| 实现复杂度 | 高（需处理边界） | 低（规整通信） |

**关键代码**：

```python
# Forward: gather_seq_scatter_heads
xq = gather_seq_scatter_heads(xq, seq_dim=1, head_dim=2, ...)

# Backward: gather_heads_scatter_seq
output = gather_heads_scatter_seq(output, head_dim=2, seq_dim=1, ...)
```

**优势**：

1. 直接使用 PyTorch 原生 `all_to_all_single`，无需自定义 CUDA kernel。
2. 与 TP 自然兼容（TP 切分 hidden dim，Ulysses 切分 heads）。
3. 支持 packed sequences（`cu_seqlens` 语义）。

## 4.3.1 Ulysses 在 Attention 中的实现

```python
def forward(self, x, rope_cache, positions, cu_seqlens, max_seqlen, ...):
    bs, seqlen, _ = x.shape

    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.view(bs, seqlen, -1, self.head_dim)
    xk = xk.view(bs, seqlen, -1, self.head_dim)
    xv = xv.view(bs, seqlen, -1, self.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, rope_cache, positions)

    if self._sp_enabled:
        kv_heads = xk.size(2)
        if kv_heads < self._cp_size:
            repeats = self._cp_size // kv_heads
            xk = repeat_kv(xk, repeats)
            xv = repeat_kv(xv, repeats)

        xq = gather_seq_scatter_heads(xq, seq_dim=1, head_dim=2, ...)
        xk = gather_seq_scatter_heads(xk, seq_dim=1, head_dim=2, ...)
        xv = gather_seq_scatter_heads(xv, seq_dim=1, head_dim=2, ...)
        seqlen = xq.shape[1]

    # Attention computation...

    if self._sp_enabled:
        output = gather_heads_scatter_seq(output, head_dim=2, seq_dim=1, ...)
        seqlen = output.shape[1]
```

## 4.4 MoE 深度实现（`moe/`）

### 4.4.1 路由机制（`router.py`）

```python
class TokenChoiceTopKRouter(nn.Module):
    def forward(self, x, expert_bias):
        # 1. 计算路由分数
        scores = self.gate(x)

        # 2. 应用专家偏置（auxiliary-loss-free load balancing）
        if expert_bias is not None:
            scores_for_choice = scores + expert_bias
```

**Node-limited Routing**：

```python
def _get_node_limited_routing_scores(self, scores):
    # 将专家分组，组内 top-2 分数决定组优先级
    scores_grouped = scores_for_choice.view(-1, num_groups, experts_per_group)
    top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
    group_scores = top2_scores_in_group.sum(dim=-1)

    # 只考虑 top 组内的专家
    _, group_idx = torch.topk(group_scores, k=self.num_limited_groups, ...)
```

**设计意图**：限制通信范围，在大规模 expert 场景下减少跨节点流量。

### 4.4.2 GroupedExperts（`grouped_experts.py`）

3D 权重张量设计：

```python
self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
```

**计算路径**：

```python
# 路径1: torch._grouped_mm (高效)
h = F.silu(torch._grouped_mm(x, w1.T, offs=offsets))
h = h * torch._grouped_mm(x, w3.T, offs=offsets)
out = torch._grouped_mm(h, w2.T, offs=offsets)

# 路径2: for-loop (兼容)
for expert_idx, x_expert in enumerate(x_splits):
    h = F.silu(x_expert @ w1[expert_idx].T)
    ...
```

**Alignment Padding**：由于 `grouped_mm` 需要固定的对齐要求，框架自动处理 padding：

```python
# moe/utils.py
def indices_padding_wrapper(fn):
    # 自动对齐 token 数量到 grouped_mm 要求
```

### 4.4.3 Token Reordering（`token_reorderer.py`）

分离式 token 重排序设计：

- `reorderer` 作为独立模块，可被 `ReordererSequenceParallel` 包装
- 支持 TP 切分下的全局索引计算

### 4.4.4 MoE Layer 整体架构

```python
class MoE(nn.Module):
    def forward(self, x):
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)

        # 1. 路由
        top_scores, selected_indices, num_tokens_per_expert = self.router(
            x_flat, self.expert_bias
        )

        # 2. 重排序
        top_scores_sorted, token_indices_sorted, num_per_expert = self.reorderer(
            top_scores, selected_indices
        )

        # 3. 收集 tokens
        routed_input = x_flat[token_indices_sorted // self.top_k]

        # 4. Expert 计算
        # 如果启用 EP，dispatch/combine 通过 distribute_module 注册的 hooks 自动执行
        routed_output = self.experts(routed_input, num_per_expert)

        # 5. Shared experts（在 unsort 之前计算以重叠通信）
        shared_out = (
            self.shared_experts(x_flat) if self.shared_experts is not None else None
        )

        # 6. Unsort 路由输出回原始位置
        routed_output_unsorted = torch.zeros(
            bs * slen * self.top_k, dim,
            device=routed_output.device, dtype=routed_output.dtype
        )
        routed_output_unsorted[token_indices_sorted] = routed_output
        routed_output_unsorted = routed_output_unsorted.view(bs * slen, self.top_k, dim)

        # 7. 合并 expert 输出（使用原始 top_scores，非 sorted）
        if self.score_before_experts:
            out_experts = routed_output_unsorted.sum(dim=1)
        else:
            out_experts = torch.bmm(
                top_scores.unsqueeze(1).float(),
                routed_output_unsorted.float()
            ).squeeze(1).to(x.dtype)

        # 8. 添加 shared experts 输出
        if shared_out is not None:
            out = shared_out + out_experts
        else:
            out = out_experts

        return out.view(bs, slen, dim)
```

## 4.5 专家并行策略矩阵（`expert_parallel.py`）

| 配置 | 策略类 | 权重切分 | 通信模式 |
| --- | --- | --- | --- |
| `EP=1, TP=1` | None | Replicate | 无 |
| `EP=1, TP>1` | TensorParallel | `Shard(1)` or `Shard(2)` | All-reduce |
| `EP>1, TP=1` | ExpertParallel | `Shard(0)` | All-to-All |
| `EP>1, TP>1, ETP=1` | ExpertParallel | `Shard(0)`（TP 被 EP 借用） | All-to-All |
| `EP>1, TP>1, ETP=TP` | ExpertTensorParallel | `[Shard(0), Shard(1/2)]` | All-to-All + All-reduce |

**关键机制**：

```python
class ExpertParallel(BaseExpertParallel):
    def _token_dispatch(self, module, inputs, device_mesh):
        # 1. 交换 token 数量
        num_tokens_per_expert_received = all_to_all_single(...)

        # 2. 变长 All-to-All dispatch
        routed_input = all_to_all_single_autograd(
            routed_input, output_splits, input_splits, group
        )

        # 3. 对齐 padding 以支持 grouped_mm
        routed_input = _permute(routed_input, ...)
```

## 4.5.1 ExpertTensorParallel 实现

```python
class ExpertTensorParallel(ExpertParallel):
    """EP + TP 组合策略"""

    def _partition_fn(self, name, module, device_mesh):
        # 2D sharding: [Shard(0), Shard(1/2)]
        module.register_parameter(
            "w1",
            nn.Parameter(
                distribute_tensor(module.w1, device_mesh, [Shard(0), Shard(1)])
            )
        )
        module.register_parameter(
            "w2",
            nn.Parameter(
                distribute_tensor(module.w2, device_mesh, [Shard(0), Shard(2)])
            )
        )
        module.register_parameter(
            "w3",
            nn.Parameter(
                distribute_tensor(module.w3, device_mesh, [Shard(0), Shard(1)])
            )
        )
```

## 4.6 激活检查点策略（`activation_checkpoint.py`）

四级 AC 策略：

```python
class ActivationCheckpointConfig:
    mode: str  # "none", "full", "selective", "memory_budget"
    selective_ac_option: str  # "op" 或层数间隔（如 "2"）
```

**Op-level Selective AC**：

```python
op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops.aten.max.default,
    torch.ops.areal._varlen_attn.default,
}
```

**自定义策略逻辑**：

```python
def _get_custom_policy(meta):
    def _custom_policy(ctx, func, *args, **kwargs):
        if func == torch.ops.aten.mm.default and args[1].shape in mm_recompute_shapes:
            return CheckpointPolicy.PREFER_RECOMPUTE

        to_save = func in op_sac_save_list and not (
            func == torch.ops.aten.mm.default and meta[f"{mode}_mm_count"] % 2 == 0
        )
        return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE
```

## 4.6.1 AC 模式详解

| 模式 | 说明 | 适用场景 |
| --- | --- | --- |
| `none` | 不启用 AC | 显存充足 |
| `full` | 每层都 checkpoint | 极致省显存 |
| `selective (op)` | 选择性保存特定算子 | 平衡性能与显存 |
| `selective (layer)` | 每隔 N 层 checkpoint | 简单配置 |
| `memory_budget` | PyTorch 自动规划 | 需要 compile 支持 |

## 4.7 并行化应用顺序（`parallelize.py`）

严格的有序应用策略：

```python
def parallelize_qwen3(model, parallel_dims, ...):
    # 1. Non-MoE TP - 切分非 MoE 权重
    if parallel_dims.tp_enabled:
        apply_non_moe_tp(model, tp_mesh)

    # 2. MoE EP+TP - 必须在 FSDP 之前（distribute_module 需要先注册 hooks）
    if tp_mesh is not None or ep_mesh is not None:
        apply_moe_ep_tp(model, tp_mesh, ep_mesh, etp, ep_tp_mesh)

    # 3. CP - 设置通信组
    if parallel_dims.cp_enabled:
        apply_cp(model, cp_group)

    # 4. AC - 必须在 TP/EP 之后（AC 包裹并行化后的模块）
    if ac_config and ac_config.mode != "none":
        apply_ac(model, ac_config, ...)

    # 5. torch.compile - 必须在 AC 之后、FSDP 之前
    if enable_compile:
        apply_compile(model)

    # 6. FSDP - 最后应用
    dp_mesh = parallel_dims.get_mesh("dp_shard_cp")
    if dp_mesh is not None:
        apply_fsdp(model, dp_mesh, ...)
```

**设计原理**：

- **TP → EP**：非 MoE TP 先完成，EP 使用 `distribute_module` 注册 hooks 和切分专家权重。
- **EP → FSDP**：EP 的 hooks 必须在 FSDP 包裹之前注册，否则会与 FSDP hooks 冲突。
- **TP → AC**：TP 改变模块结构，AC 需要包裹并行化后的模块。
- **AC → Compile**：Compile 需要看到 AC wrapper 以正确生成 backward。
- **Compile → FSDP**：FSDP 的 all-gather 需要与 compiled forward 协调。

## 4.7.1 TP 实现细节

```python
def apply_tp(model, tp_mesh, loss_parallel=True):
    validate_tp_constraints(model.model_args, tp_mesh.size())

    root_plan = {}
    if model.tok_embeddings is not None:
        root_plan["tok_embeddings"] = RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        )

    if model.norm is not None:
        root_plan["norm"] = SequenceParallel()

    if model.output is not None:
        root_plan["output"] = ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=True,
        )

    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention.wq": ColwiseParallel(use_local_output=True),
            "attention.wk": ColwiseParallel(use_local_output=True),
            "attention.wv": ColwiseParallel(use_local_output=True),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }
        parallelize_module(transformer_block, tp_mesh, layer_plan)
```

---

# 5. 关键技术创新点

## 5.1 无辅助损失的负载均衡

```python
# MoE layer 中的 buffer
self.register_buffer("expert_bias", torch.zeros(num_experts))
self.register_buffer("tokens_per_expert", torch.zeros(num_experts))

# 训练过程中更新偏置（基于 tokens_per_expert 统计）
```

**与传统方法的对比**：

| 方法 | 机制 | 优缺点 |
| --- | --- | --- |
| 传统辅助损失 | load balancing loss | 增加梯度复杂度，需调权重系数 |
| Archon 偏置法 | 直接调整路由分数 | 无额外梯度流，动态适应负载 |

## 5.2 自定义 Flash Attention Op

```python
@torch.library.custom_op("areal::_varlen_attn", mutates_args=())
def _varlen_attn(query, key, value, cu_seq_q, cu_seq_k, max_seq_q, max_seq_k):
    return torch.ops.aten._flash_attention_forward(
        query, key, value, cu_seq_q, cu_seq_k,
        max_seq_q, max_seq_k, ...
    )
```

将 Flash Attention 包装为 custom op，使其：

- 可被 `torch.compile` 正确追踪
- 可被选择性 AC 识别和处理

## 5.3 权重绑定与 FSDP 兼容

```python
if getattr(model.model_args, "enable_weight_tying", False):
    if model.output is not None and model.tok_embeddings is not None:
        model.output.weight = model.tok_embeddings.weight  # 绑定权重

# FSDP 需要特殊处理 tied weights
```

## 5.4 灵活的 Pipeline 调度策略

```python
def build_pipeline_schedule(stages, pp_schedule, n_microbatches, ...):
    schedule_class = get_schedule_class(pp_schedule)
    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)

    num_total_stages = len(stages) * pp_degree
    if n_microbatches < num_total_stages:
        logger.warning(
            f"n_microbatches ({n_microbatches}) < num_total_stages ({num_total_stages}), may result in pipeline bubble"
        )

    return schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        scale_grads=False,
    )
```

---

# 6. 代码质量与工程实践

## 6.1 类型安全

- 全程使用 `TYPE_CHECKING` 避免循环导入
- `ParallelizeFn` 和 `PipeliningFn` 使用 `Protocol` 定义接口
- 配置类使用 `dataclass` 配合 `__post_init__` 验证

## 6.2 可观测性

每个模块实现 rank-aware logger：

```python
@functools.cache
def _get_logger() -> logging.Logger:
    rank = dist.get_rank() if dist.is_initialized() else 0
    return logging.getLogger(f"[Archon {ModuleName} Rank {rank}]")
```

## 6.3 防御式编程

Mesh 获取使用安全模式：

```python
def get_mesh(self, name: str) -> DeviceMesh | None:
    _ = self.world_mesh  # 确保 mesh 已构建
    return self._meshes.get(name)  # 不存在返回 None 而非抛异常
```

## 6.4 配置验证

所有约束在 `__post_init__` 中验证：

| 并行类型 | 验证条件 | 说明 |
| --- | --- | --- |
| TP | `n_heads % tp == 0` | 注意力头数可被 TP 度整除 |
| TP | `n_kv_heads % tp == 0` | KV 头数可被 TP 度整除 |
| CP | `q_heads % cp == 0` | 考虑 TP 后查询头数可被 CP 度整除 |
| EP | `num_experts % ep == 0` | 专家数可被 EP 度整除 |
| EP + ETP | EP 与 ETP / TP / CP 的整除关系 | 复杂的多维约束 |

---

# 7. 总结与评价

## 7.1 架构优势

1. **模块化设计**：5 种并行策略独立实现，可灵活组合。
2. **原生 PyTorch**：无需自定义 C++ 扩展，降低维护成本。
3. **HuggingFace 兼容**：无缝加载/保存 HF 格式 checkpoint。
4. **MoE 原生支持**：从路由到 expert 计算的端到端优化。

## 7.2 设计取舍

| 取舍点 | Archon 选择 | 替代方案 | 权衡分析 |
| --- | --- | --- | --- |
| 序列并行 | Ulysses | Ring Attention | 简单性 > 极致性能 |
| EP 通信 | All-to-All | All-Gather | 直接对齐 > 冗余计算 |
| 专家计算 | grouped_mm | for-loop | 性能 > 兼容性 |
| AC 粒度 | Op-level | Layer-level | 灵活性 > 简单性 |

## 7.3 适用场景

- **大规模 MoE 模型训练**（Qwen3-MoE 类架构）
- **超长序列训练**（Ulysses SP + CP 组合）
- **多维度并行探索**（灵活配置 TP / CP / PP / EP 组合）

## 7.4 代码洞察

Archon 的代码体现了 **现代 PyTorch 分布式训练的最佳实践**：

1. **DeviceMesh 抽象**：将硬件拓扑与算法解耦。
2. **DTensor 张量并行**：声明式 sharding 而非手动通信。
3. **distribute_module 钩子**：自动注册通信逻辑。
4. **torch.compile 原生支持**：与现代编译栈深度整合。

这是一个 **工程成熟度很高** 的训练框架，其设计思路和实现细节对构建大规模分布式训练系统具有重要的参考价值。

---

# 附录：核心类关系图

```text
BaseArchonModel (abstract)
    ├── forward(tokens, positions, cu_seqlens, max_seqlen)
    ├── init_weights()
    └── init_buffers()
    │
    ├── Qwen2Model
    │   ├── tok_embeddings (nn.Embedding)
    │   ├── layers (ModuleDict[str, TransformerBlock])
    │   ├── norm (RMSNorm)
    │   ├── output (nn.Linear) or score (for critic)
    │   └── rope_cache (buffer)
    │
    └── Qwen3Model (adds MoE support)
        └── _is_moe_layer() logic for sparse layers

TransformerBlock
    ├── attention_norm (RMSNorm)
    ├── attention (Attention)
    ├── ffn_norm (RMSNorm)
    └── feed_forward (FeedForward or MoE)

Attention
    ├── wq, wk, wv, wo (nn.Linear)
    ├── q_norm, k_norm (RMSNorm, Qwen3 only)
    ├── packed_attn (SDPA / Varlen / Tree wrapper)
    └── _cp_group (Ulysses SP process group)

MoE (replaces FeedForward in MoE layers)
    ├── router (TokenChoiceTopKRouter)
    ├── experts (GroupedExperts)
    ├── reorderer (TokenReorderer)
    └── shared_experts (FeedForward, optional)

GroupedExperts
    ├── w1, w2, w3 (3D Parameters: [num_experts, *, *])
    └── forward() with grouped_mm or for-loop

ParallelStyle implementations (for TP / EP)
    ├── ExpertParallel (EP only)
    ├── ExpertTensorParallel (EP + TP)
    ├── TensorParallel (TP only for experts)
    └── ReordererSequenceParallel (TP split for reorderer)

ModelSpec (registration)
    ├── name: str
    ├── model_class: type[nn.Module]
    ├── model_args_class: type[BaseModelArgs]
    ├── state_dict_adapter_class: type[BaseStateDictAdapter]
    ├── parallelize_fn: ParallelizeFn
    └── pipelining_fn: PipeliningFn
```
