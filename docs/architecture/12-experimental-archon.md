# Archon 引擎与模型架构

> 源码位置：`areal/experimental/engine/`, `areal/experimental/models/archon/` 文件数：50 个 |
> 总行数：13673 行

______________________________________________________________________

## 1. 模块定位

Archon 是 AReaL 的第三种训练后端，与 FSDP 引擎和 Megatron 引擎并列。其设计理念 是**全面拥抱 PyTorch 原生分布式
API**——DeviceMesh + DTensor + torch.distributed.pipelining

- torch.distributed.fsdp (FSDP2)——而非依赖 Megatron-LM 私有并行层。

### 为什么需要第三种引擎

| 维度          | FSDPEngine       | MegatronEngine          | **ArchonEngine**               |
| ------------- | ---------------- | ----------------------- | ------------------------------ |
| 底层并行框架  | FSDP2            | Megatron-LM 内部 API    | DeviceMesh + FSDP2 + DTensor   |
| 张量并行      | 无               | Megatron ColumnParallel | PyTorch TP (ColwiseParallel)   |
| 流水线并行    | 无               | Megatron Pipeline       | torch.distributed.pipelining   |
| 专家并行      | 无               | 有                      | distribute_module + EP hooks   |
| 模型定义方式  | HuggingFace 原生 | Megatron 私有格式       | 自定义 Archon 模型 + ModelSpec |
| torch.compile | 不支持           | 不支持                  | 支持 (per-block Inductor)      |
| FP8 训练      | 不支持           | Megatron FP8            | torchao fp8_blockwise_mm       |
| 代码复杂度    | 低               | 高                      | 中                             |

核心价值：在保持 AReaL RL 训练流水线兼容性的前提下，提供 **TP + PP + CP + EP + ETP** 全五维并行，同时不引入 Megatron-LM
的外部依赖。

______________________________________________________________________

## 2. 目录与文件清单

```
areal/experimental/
  engine/                           # 引擎层 (5 文件, 3312 行)
    archon_engine.py          1571  # ArchonEngine 主类 + PPO/SFT/DPO/RW 子类
    archon_checkpoint.py       517  # HF/DCP 检查点 保存/加载
    archon_runner.py           332  # SequentialRunner / PipelinedRunner
    archon_utils.py            364  # 优化器、LR 调度、AC 配置、确定性模式
    archon_weight_sync.py      528  # 训练-推理权重同步 (XCCL / disk)

  models/archon/                    # 模型架构层 (45 文件, 10361 行)
    __init__.py                 50  # 公开 API + 触发 spec 注册
    base.py                    172  # BaseModelArgs, BaseStateDictAdapter, BaseArchonModel
    model_spec.py              137  # ModelSpec 注册表 (register/get_model_spec)
    parallel_dims.py           420  # ArchonParallelDims (5维并行 mesh)
    pipeline_parallel.py       494  # PP 阶段划分 + 调度构建
    expert_parallel.py         514  # EP / ETP / TP-for-experts 并行风格
    fp8.py                     230  # FP8 blockwise matmul 补丁
    fp8_checkpoint.py          328  # FP8 检查点反量化
    activation_checkpoint.py   313  # 激活检查点 (full/selective/memory_budget)
    compile.py                  47  # torch.compile per-block
    ulysses.py                  82  # Ulysses 序列并行 (slice/gather)
    moe_weight_converter.py    446  # Dense->MoE 权重转换
    utils.py                   168  # TP/CP/EP 约束校验

    attention/                      # 注意力后端
      __init__.py               12
      sdpa.py                  125  # SDPA wrapper
      varlen.py                327  # VarlenAttention (flash_attn) + 自定义 Op

    moe/                            # MoE 子系统
      __init__.py               30
      args.py                  109  # MoEArgs 数据类
      router.py                308  # TokenChoiceTopKRouter
      moe.py                   261  # MoE 层 (路由 + 专家 + 共享专家)
      grouped_experts.py       288  # GroupedExperts (3D权重 + grouped_mm)
      token_reorderer.py        73  # TokenReorderer (EP SP 解耦)
      kernels.py               230  # CUDA 核函数辅助
      utils.py                 298  # permute/unpermute 对齐

    qwen2/                          # Qwen2 (Dense)
      __init__.py               17
      spec.py                   25  # QWEN2_SPEC 注册
      model/args.py             70  # Qwen2ModelArgs
      model/model.py           385  # Qwen2Model (Attention + FFN)
      model/rope.py            161  # RoPE (标准)
      model/state_dict_adapter.py 133  # HF<->Archon 键映射
      infra/parallelize.py     366  # parallelize_qwen2

    qwen3/                          # Qwen3 (Dense + MoE)
      __init__.py               17
      spec.py                   25  # QWEN3_SPEC 注册 (qwen3, qwen3_moe)
      model/args.py            111  # Qwen3ModelArgs + MoEArgs
      model/model.py           499  # Qwen3Model (QK-Norm + MoE层)
      model/rope.py             18  # RoPE (复用 Qwen2)
      model/state_dict_adapter.py 468  # MoE 键转换
      infra/parallelize.py     757  # parallelize_qwen3 (含 EP/ETP)

    qwen3_5/                        # Qwen3.5 (Hybrid: Full + Linear Attention)
      __init__.py               17
      spec.py                   29  # QWEN3_5_SPEC (qwen3_5, qwen3_5_moe, ...)
      model/args.py            182  # Qwen3_5ModelArgs (layer_types, GatedDeltaNet 参数)
      model/model.py           681  # Qwen3_5Model (GatedDeltaNet + GatedAttention)
      model/rope.py            155  # Partial RoPE (partial_rotary_factor)
      model/state_dict_adapter.py 482  # 复合 VLM namespace 映射
      infra/parallelize.py     301  # parallelize_qwen3_5
```

______________________________________________________________________

## 3. 核心数据结构

### 3.1 ModelSpec 注册机制

```
model_spec.py (L85-L137)

    +------------------+
    |    ModelSpec      |
    +------------------+
    | name             |  "Qwen2" / "Qwen3" / "Qwen3_5"
    | model_class      |  Qwen2Model / Qwen3Model / Qwen3_5Model
    | model_args_class |  Qwen2ModelArgs / Qwen3ModelArgs / ...
    | state_dict_adapter_class |  Qwen2StateDictAdapter / ...
    | parallelize_fn   |  parallelize_qwen2 / parallelize_qwen3 / ...
    | pipelining_fn    |  pipeline_llm (所有模型共享)
    | supported_model_types |  frozenset({"qwen2"}) / ...
    +------------------+

    _MODEL_SPECS: dict[str, ModelSpec]    # 全局注册表
    register_model_spec(spec) -> 按 model_type 写入
    get_model_spec(model_type) -> 按 model_type 查询
```

注册时机：每个 `qwen{X}/spec.py` 模块级调用 `register_model_spec()`，由 `__init__.py` 中
`import qwen2.spec` / `qwen3.spec` / `qwen3_5.spec` 触发。

### 3.2 ArchonParallelDims (五维并行)

```
parallel_dims.py (L27-L109)

    +-----------------------------+
    |   ArchonParallelDims        |
    +-----------------------------+
    | dp_shard  (-1=auto)         |  FSDP 数据并行
    | tp                          |  张量并行
    | cp                          |  上下文并行 (Ulysses SP)
    | pp                          |  流水线并行
    | ep                          |  专家并行
    | etp (1 或 tp)               |  专家张量并行
    | world_size                  |
    | device_type ("cuda"/"npu")  |
    +-----------------------------+

    约束: world_size = dp_shard * tp * cp * pp
    FSDP 作用域: fsdp_size = dp_shard * cp  (CP rank 参与权重切分)
```

**DeviceMesh 构建策略**

当 EP 关闭时构建 4-D mesh：

```
    (pp, dp_shard, cp, tp)

    派生 submesh:
      dp         = dp_shard               (数据加载)
      dp_shard_cp = dp_shard * cp          (FSDP 权重切分)
      dp_cp      = dp_shard * cp           (loss all-reduce)
      pp_cp_tp   = pp * cp * tp            (数据广播 / 模型并行组)
```

当 EP 打开时构建 5-D mesh：

```
    (pp, dp_shard_mod_ep, dp_shard_in_ep, cp, tp)

    其中:
      etp=1:  ep = dp_shard_in_ep * cp * tp     (TP 被 EP 借用)
      etp=tp: ep = dp_shard_in_ep * cp           (TP 保持独立)
              ep_tp = [ep, tp]                    (2D expert+tensor 并行)
```

**EP 策略选择表**（直接源自代码 L49-L64）

```
    +-----+-----+-----+-----------------------+----------------------------+
    | EP  | TP  | etp | 策略                  | 专家权重切分方式           |
    +-----+-----+-----+-----------------------+----------------------------+
    | 1   | 1   | -   | 无                    | Replicate                  |
    | 1   | >1  | -   | TensorParallel        | [Shard(1/2)]               |
    | >1  | 1   | -   | ExpertParallel        | [Shard(0)]                 |
    | >1  | >1  | 1   | ExpertParallel        | [Shard(0)] (TP 被 EP 借用) |
    | >1  | >1  | tp  | ExpertTensorParallel  | [Shard(0), Shard(1/2)]     |
    +-----+-----+-----+-----------------------+----------------------------+
```

### 3.3 ArchonTrainContext

```
archon_engine.py (L123-L144)

    @dataclass
    ArchonTrainContext:
        mb_input: dict[str, Any]        # 原始 microbatch
        labels: Tensor | None           # 标签 (tree training 时为 None)
        pad_length: int                 # batch 填充长度
        trie_node: TrieNode | None      # tree attention 元数据
```

______________________________________________________________________

## 4. 核心流程

### 4.1 ArchonEngine 初始化流程

```
ArchonEngine.__init__(config)
  |-- AutoConfig.from_pretrained()      -> model_config
  |-- get_model_spec(model_type)        -> ModelSpec
  |
  v
create_process_group(parallel_strategy)
  |-- dist.init_process_group()
  |-- ArchonParallelDims(dp_shard, tp, cp, pp, ep, etp)
  |-- parallel_dims.world_mesh          -> DeviceMesh
  |-- warmup_process_groups()           -> 预热 NCCL communicator
  |
  v
initialize(addr=None, ft_spec)
  |-- WeightSyncState + DistributedLock
  |
  |-- _create_device_model()
  |     |-- with torch.device("meta"):
  |     |       spec.model_class(model_args)      # meta device 上构建结构
  |     +-- model.to(dtype)
  |
  |-- _create_state_dict_adapter()
  |
  |-- [FP8] enable_fp8_linear() / enable_fp8_experts()    # meta device 补丁
  |
  |-- prepare_training_config()
  |     |-- build_ac_config()
  |     |-- validate_zero_bubble_compatibility()
  |     |-- setup_deterministic_mode()          (可选)
  |     +-- force_pad_to_maximum()              (compile/PP/tree 时强制)
  |
  |-- _setup_parallelism(ac_config, enable_compile)
  |     |-- [PP=1] _apply_parallelism()
  |     |     +-- spec.parallelize_fn(model, parallel_dims, ...)
  |     |-- [PP>1] _apply_pipeline_parallelism()
  |     |     |-- spec.pipelining_fn(model, ..., parallelize_fn)
  |     |     |     |-- generate_llm_fqn_per_model_part()
  |     |     |     |-- pipeline_module_split()    # deepcopy + stage 分割
  |     |     |     +-- parallelize_fn() per model_part
  |     |     +-- 确定 _pp_last_stage_rank
  |
  |-- _materialize_and_load_weights()
  |     |-- model.to_empty(device)             # meta -> 实际设备
  |     |-- load_model_from_hf()               # DCP + HuggingFaceStorageReader
  |     +-- model.init_buffers()               # rope_cache 等
  |
  |-- _create_optimizer(ft_spec)
  |     |-- create_optimizer(params, config)
  |     +-- create_lr_scheduler(optimizer, config, steps)
  |
  +-- create_runner(pp_enabled, ...)
        |-- [PP=0] SequentialRunner
        +-- [PP>0] PipelinedRunner
```

### 4.2 训练步 (train_batch)

```
ArchonEngine.train_batch(input_, loss_fn, loss_weight_fn)
  |
  |-- optimizer_zero_grad()
  |-- _normalize_batch_input()          -> concat_batch or passthrough
  |-- _prepare_mb_list()
  |     |-- amend_position_ids()
  |     |-- split_padded_tensor_dict_into_mb_list()
  |     |-- pack_tensor_dict()          per microbatch
  |     +-- pad_mb_list(batch_align_to=lcm(page_size, seq_len_divisor))
  |
  |-- compute_total_loss_weight()       -> all-reduce over DP group
  |
  |-- forward_backward_batch(mb_list, process_output, forward_only=False)
  |     |
  |     +-- runner.run(mb_list, process_output_fn, forward_only=False)
  |           |
  |           |-- [SequentialRunner]
  |           |     for mb in mb_list:
  |           |       inputs, ctx = prepare_inputs_fn(mb)
  |           |       logits = model(input_ids, position_ids, cu_seqlens, ...)
  |           |       loss = process_output_fn(logits, ctx)
  |           |       loss.backward()
  |           |
  |           +-- [PipelinedRunner]
  |                 batched_args, batched_kwargs = prepare_pipelined_mb_inputs()
  |                 schedule = build_pipeline_schedule(stages, pp_schedule, n_mbs)
  |                 schedule.step(*args, target=target, ...)
  |
  +-- optimizer_step()
        |-- fsdp2_clip_grad_norm(all_params, max_norm,
        |       fsdp_group, tp_group, pp_group)
        |-- [grad_norm 有限] optimizer.step()
        +-- 返回 {update_successful, grad_norm, lr}
```

### 4.3 权重同步 (训练 -> 推理)

```
update_weights(meta: WeightUpdateMeta)
  |
  |-- [type="xccl"] update_weights_from_distributed()
  |     |-- pause_generation()
  |     |-- [gen_pp_size=1] _update_weights_single_group()
  |     |     for name, param in model.named_parameters():
  |     |       full_tensor = _get_full_tensor(param)     # DTensor -> full
  |     |       hf_pairs = adapter.convert_single_to_hf(name, tensor)
  |     |       累积到 bucket -> dist.broadcast(src=0, group=state.group)
  |     |
  |     +-- [gen_pp_size>1] _update_weights_per_stage()
  |           每个 PP-stage head 仅广播自己的参数
  |           通过独立的 update_weight_group_{pp_rank} NCCL 组
  |
  +-- [type="disk"] update_weights_from_disk()
        save_model_to_hf() -> rollout_engine.update_weights_from_disk()
```

______________________________________________________________________

## 5. 关键设计决策

### 5.1 Pipeline Parallel 调度

```
pipeline_parallel.py (L48-L82, L349-L494)

    调度类型:
    +---------------------------+----------+-----------------------------+
    | 调度名称                  | 类别      | 阶段分配方式                |
    +---------------------------+----------+-----------------------------+
    | 1F1B                      | Single   | rank -> (rank,)             |
    | Interleaved1F1B           | Multi    | rank -> (rank, rank+pp, ..) |
    | InterleavedZeroBubble     | Multi    | loop-style (同 Interleaved) |
    | ZBVZeroBubble             | V-style  | rank -> (rank, N-1-rank)    |
    | DualPipeV                 | V-style  | rank -> (rank, N-1-rank)    |
    +---------------------------+----------+-----------------------------+

    V-style 特点: rank 0 持有 (stage 0, stage N-1)
    Loop-style 特点: rank 0 持有 (stage 0, stage pp_degree)
```

Zero-Bubble 兼容性约束（`archon_utils.py` L150-L205）：

- Zero-Bubble 使用 `retain_graph=True` 的 split backward，与以下特性冲突：
  - `torch.compile`（自动禁用）
  - MoE `donated_buffer`（自动禁用）
  - op-level selective AC / memory_budget AC（自动回退到 full AC）

### 5.2 FP8 训练支持

```
fp8.py (L18-L163)

    阶段 1A (训练时 FP8 matmul):
      enable_fp8_linear(model, exclude_fqns={"output","router","score"})
        -> 为每个合格 nn.Linear 的 forward 打补丁:
           x -> pad to 128 -> fp8_blockwise_mm(x, weight) -> unpad

      enable_fp8_experts(model)
        -> 为每个 GroupedExperts 打补丁:
           per-expert for-loop + fp8_blockwise_mm

    阶段 1B (FP8 检查点加载):
      fp8_checkpoint.py:
        _prepare_fp8_state_dict()  -> 将 BF16 placeholder 转为 float8_e4m3fn
        dequant_fp8_state_dict()   -> FP8 权重反量化为 BF16 (GPU Triton / CPU fallback)
        _dequant_dtensor()         -> 处理 FSDP 切片的 FP8 DTensor

    约束:
      - FP8 训练要求 dtype=bfloat16 (master weights)
      - FP8 与 torch.compile 不兼容 (自动禁用)
      - 权重维度必须 128 对齐 (validate_fp8_shard_alignment)
      - FP8 检查点加载仅支持 Shard(0)，不支持 Shard(1) (Phase 2 待实现)
```

### 5.3 激活检查点策略

```
activation_checkpoint.py (L37-L82, L249-L308)

    +-------------------+---------------------------------------------+
    | 模式              | 行为                                        |
    +-------------------+---------------------------------------------+
    | none              | 不做检查点                                  |
    | full              | 每个 TransformerBlock 全量检查点            |
    | selective (op)    | 按算子级别选择性保存 (mm, attention, ...)   |
    | selective (N)     | 每 N 层做一次检查点                         |
    | memory_budget     | 设置内存预算 (0.0~1.0), 需要 torch.compile |
    +-------------------+---------------------------------------------+

    op-level SAC 保存列表 (qwen3/parallelize.py L59-L83):
      aten.mm, sdp_*_attention, reduce_scatter_tensor,
      all_to_all_single, aten.max, flex_attention, _varlen_attn
```

### 5.4 Expert Parallel 架构

```
expert_parallel.py (L29-L514)

    +------------------------------------------------------+
    |         BaseExpertParallel (ABC)                      |
    +------------------------------------------------------+
    | _partition_fn()    切分专家权重                        |
    | _token_dispatch()  分发 token 到持有对应专家的设备     |
    | _token_combine()   汇聚专家输出到原始位置              |
    | _apply()           通过 distribute_module 注册 hooks  |
    +------------------------------------------------------+
              |                        |
    +---------+----------+   +---------+------------------+
    | ExpertParallel     |   | ExpertTensorParallel       |
    | (EP only, etp=1)   |   | (EP + TP, etp=tp)          |
    +---------+----------+   +---------+------------------+
    | w1/w2/w3: Shard(0) |   | w1: [Shard(0), Shard(1)]   |
    | all-to-all dispatch|   | w2: [Shard(0), Shard(2)]   |
    | _permute/_unpermute|   | w3: [Shard(0), Shard(1)]   |
    +--------------------+   | 2D mesh [ep, tp]           |
                             +----------------------------+

    +------------------------------------------------------+
    | TensorParallel (EP 关闭, TP only for experts)        |
    +------------------------------------------------------+
    | w1/w3: Shard(1)  w2: Shard(2)                        |
    | Replicate input, Partial gradient backward            |
    +------------------------------------------------------+

    +------------------------------------------------------+
    | ReordererSequenceParallel (etp=1 时使用)             |
    +------------------------------------------------------+
    | 将 token 按 TP rank 切分                              |
    | 输出的 token_indices 调整为全局索引                   |
    +------------------------------------------------------+
```

### 5.5 检查点 (HF/DCP 双格式)

```
archon_checkpoint.py (L86-L517)

    保存 HF 格式:
      save_model_to_hf(engine, path, tokenizer)
        |-- get_model_state_dict()
        |-- adapter.to_hf(state_dict)
        |-- HuggingFaceStorageWriter (sharded)
        |-- _consolidate_shards_distributed()    # round-robin 合并
        |-- _write_safetensors_index()
        +-- 原子重命名: tmp_path -> path

    加载 HF 格式:
      load_model_from_hf(engine, path)
        |-- adapter.to_hf(state_dict)            # 生成 HF key placeholder
        |-- [FP8] _prepare_fp8_state_dict()      # BF16->FP8 placeholder
        |-- dcp.load(hf_state_dict, HuggingFaceStorageReader)
        |-- [FP8] dequant_fp8_state_dict()
        |-- adapter.from_hf(hf_state_dict)       # HF key -> Archon key
        +-- set_model_state_dict()

    DCP 格式 (原生分布式检查点):
      DCPState(Stateful)
        |-- state_dict(): model + optim (flatten=True for PP)
        +-- load_state_dict(): strict=False for PP (部分 key)

    优化器状态: 按 rank 分片保存
      save_optimizer_state() -> optim_world_size_{N}_rank_{R}.pt
```

______________________________________________________________________

## 6. Qwen2 -> Qwen3 -> Qwen3.5 架构演进

### 6.1 架构对比

```
    +---------------------+-------------------+-------------------+-------------------+
    |                     | Qwen2             | Qwen3             | Qwen3.5           |
    +---------------------+-------------------+-------------------+-------------------+
    | 注意力              | MHA/GQA           | MHA/GQA + QK-Norm | GatedAttention     |
    |                     |                   |                   | + GatedDeltaNet    |
    | FFN                 | SwiGLU            | SwiGLU / MoE      | SwiGLU / MoE      |
    | MoE 支持            | 无                | decoder_sparse_step| 全层 MoE          |
    | Norm                | RMSNorm(weight*x) | RMSNorm           | (1+weight)*norm(x) |
    | RoPE                | 标准              | 标准              | Partial RoPE      |
    | Weight Tying        | 支持              | 支持              | 支持              |
    | 注意力类型          | varlen/sdpa/tree  | varlen/sdpa/tree  | varlen only       |
    | Critic              | score 层          | score 层          | score 层          |
    | 共享专家             | -                 | num_shared_experts| shared_expert_gate|
    | model_types         | qwen2             | qwen3, qwen3_moe | qwen3_5, qwen3_5_moe, |
    |                     |                   |                   | qwen3_5_text, ...   |
    +---------------------+-------------------+-------------------+-------------------+
```

### 6.2 GatedDeltaNet (Qwen3.5 独有)

```
model.py (L141-L308)

    GatedDeltaNet 是 linear_attention 层的实现:

    输入 x [B, T, dim]
      |
      v
    +-- in_proj_qkv(x) --> mixed_qkv [B, T, conv_dim]
    +-- in_proj_z(x)   --> z         [B, T, value_dim]   (gate)
    +-- in_proj_a(x)   --> a         [B, T, num_v_heads]
    +-- in_proj_b(x)   --> b         [B, T, num_v_heads]
      |
      v
    causal_conv1d_fn(mixed_qkv, conv1d.weight, activation="silu", seq_idx)
      |                               ^
      |                               |
      v                     packing 隔离: seq_idx 防止跨序列卷积
    split -> Q [B,T,num_k_heads,head_k_dim]
             K [B,T,num_k_heads,head_k_dim]
             V [B,T,num_v_heads,head_v_dim]
      |
      v
    compute_decay_beta(A_log, dt_bias, a, b)
      -> beta = sigmoid(b)
      -> g = -exp(A_log) * softplus(a + dt_bias)
      |
      v
    chunk_gated_delta_rule(Q, K, V, g, beta, cu_seqlens)
      |                      (fla 库 gated delta rule 核)
      v
    Qwen3_5RMSNormGated(output, gate=z)
      |    output * silu(z) * (weight * norm(output))
      v
    out_proj(output) -> [B, T, dim]
```

GatedDeltaNet 与传统 attention 的区别：

- 使用因果卷积代替位置编码（depthwise Conv1d）
- 使用 gated delta rule 代替 softmax attention（亚二次复杂度）
- 使用 gated RMSNorm（`output * silu(gate)`）

### 6.3 GatedAttention (Qwen3.5 full_attention 层)

```
    与标准 MHA 的区别:
    - wq 输出 2x 宽度: (query, gate) = chunk(wq(x), 2)
    - 使用 Q/K Norm (Qwen3_5RMSNorm with (1+weight) semantics)
    - 使用 Partial RoPE (仅对 head_dim 的前 partial_rotary_factor 比例应用)
    - 输出门控: output = attn_output * sigmoid(gate)
```

### 6.4 Hybrid 层调度

```
    Qwen3.5 通过 layer_types 列表控制每层类型:
    layer_types = ["full_attention", "linear_attention", "full_attention", ...]

    TransformerBlock 根据 layer_type 选择:
      full_attention   -> GatedAttention (标准 attention + gate)
      linear_attention -> GatedDeltaNet  (线性注意力)
```

______________________________________________________________________

## 7. MoE 路由器与专家系统

### 7.1 路由流程

```
moe/moe.py (L124-L219)

    MoE.forward(x: [B, S, dim])
      |
      v
    x_flat = x.view(-1, dim)                    # [B*S, dim]
      |
      v
    router(x_flat, expert_bias)
      |-- gate(x) = RouterGateLinear(x)          # [B*S, num_experts]
      |     (可选 FP32 精度 GEMM via RouterGatingLinearFunction)
      |-- sigmoid(scores) 或 softmax(scores)     # float32 避免 loss explosion
      |-- [节点限制路由] _get_node_limited_routing_scores()
      |     group_scores = sum(top2_per_group)
      |     mask out non-selected groups
      |-- topk(scores_for_choice, k=top_k)
      |-- [route_norm] normalize top_scores
      |-- top_scores *= route_scale
      +-- histc -> num_tokens_per_expert
      |
      v
    reorderer(top_scores, selected_indices)
      |-- argsort(selected_indices)              # 按专家排序
      +-- 返回 sorted scores, sorted indices, per-expert counts
      |
      v
    routed_input = x_flat[token_indices // top_k]
      |
      |-- [score_before_experts] routed_input *= scores
      |
      v
    experts(routed_input, num_tokens_per_expert)
      |-- [grouped_mm] torch._grouped_mm          # BF16 CUTLASS
      |-- [for_loop]   per-expert matmul           # fallback
      |-- [FP8]        fp8_blockwise_mm per expert # FP8 path
      |
      v
    unsort -> weighted sum -> + shared_experts(x_flat)
      |
      v
    output [B, S, dim]
```

### 7.2 负载均衡

```
    MoE 使用 auxiliary-loss-free 负载均衡:
    - expert_bias: 运行时 buffer, 外部更新
    - tokens_per_expert: 每个前向累积统计 (torch.no_grad)
    - load_balance_coeff: 控制 bias 更新幅度

    路由时: scores_for_choice = scores + expert_bias
    选择仍基于 bias-adjusted scores, 但实际权重使用原始 scores
```

### 7.3 GroupedExperts 3D 权重

```
moe/grouped_experts.py (L195-L275)

    w1: [num_experts, hidden_dim, dim]    gate projection
    w2: [num_experts, dim, hidden_dim]    down projection
    w3: [num_experts, hidden_dim, dim]    up projection

    SwiGLU 计算: silu(x @ w1.T) * (x @ w3.T) @ w2.T

    grouped_mm: torch._grouped_mm(x, w.T, offs=cumsum(per_expert_counts))
      - 使用 BF16 CUTLASS kernel
      - 需要 PyTorch 2.4+
      - 仅 CUDA 可用

    非 EP 模式下自动添加 indices_padding_wrapper 进行 grouped_mm 对齐
```

______________________________________________________________________

## 8. 接口契约与扩展点

### 8.1 添加新模型的步骤

```
    1. 创建目录 areal/experimental/models/archon/new_model/
    2. 实现:
       - model/args.py      (继承 BaseModelArgs, 实现 from_hf_config)
       - model/model.py     (继承 BaseArchonModel, 实现 forward/init_weights/init_buffers)
       - model/state_dict_adapter.py  (继承 BaseStateDictAdapter, 实现 from_hf/to_hf)
       - model/rope.py      (如有不同的位置编码)
       - infra/parallelize.py  (实现 parallelize_fn, 遵循 ParallelizeFn Protocol)
    3. 创建 spec.py:
       spec = ModelSpec(
           name="NewModel",
           model_class=NewModel,
           model_args_class=NewModelArgs,
           state_dict_adapter_class=NewStateDictAdapter,
           parallelize_fn=parallelize_new_model,
           supported_model_types=frozenset({"new_model_type"}),
           pipelining_fn=pipeline_llm,   # 共用
       )
       register_model_spec(spec)
    4. 在 __init__.py 中添加 import 触发注册
```

### 8.2 parallelize_fn 的应用顺序

```
    parallelize_fn (以 parallelize_qwen3 为例):

    1. validate_tp_constraints / validate_cp_constraints / validate_ep_constraints
    2. [TP>1]  parallelize_module(attention layers, {wq/wk/wv: ColwiseParallel, wo: RowwiseParallel})
    3. [TP>1]  parallelize_module(ffn layers, {w1/w3: ColwiseParallel, w2: RowwiseParallel})
    4. [TP>1]  parallelize_module(output, ColwiseParallel with loss_parallel)
    5. [CP>1]  set_cp_group() on attention layers (Ulysses SP)
    6. [EP>1]  ExpertParallel/ExpertTensorParallel on experts
              ReordererSequenceParallel on reorderer (etp=1 only)
              TensorParallel on experts (EP disabled, TP>1 only)
              ReplicateParallel on router.gate
    7. [AC]   apply_ac(model, ac_config, op_sac_save_list)
    8. [compile] apply_compile(model)      # per-block torch.compile
    9. [FSDP] fully_shard(layer, ...)      # 每层单独 FSDP
              fully_shard(model, ...)      # 模型级 FSDP
```

### 8.3 ArchonEngine 子类体系

```
    ArchonEngine (TrainEngine)                    # archon_engine.py L147
      |-- train_batch / eval_batch / forward_batch
      |-- save / load / update_weights
      |
      +-- ArchonPPOActor                          # L1400  PPO Actor
      +-- ArchonPPOCritic                         # L1436  PPO Critic
      +-- ArchonLMEngine                          # L1468  SFT
      +-- ArchonRWEngine                          # L1499  Reward Modeling
      +-- ArchonDPOEngine                         # L1533  DPO

    每个子类通过组合模式持有对应 Trainer:
      self.actor = PPOActor(config, self)
      self.lm_engine = LMEngine(self)
      ...
```

### 8.4 ForwardBackwardRunner 策略模式

```
    archon_runner.py (L30-L332)

    ForwardBackwardRunner (ABC)
      |-- run(mb_list, process_output_fn, forward_only)
      |
      +-- SequentialRunner              # PP=1, 逐 microbatch 串行
      |     model(input_ids, position_ids, cu_seqlens, max_seqlen)
      |     result = process_output_fn(logits, ctx)
      |     result.backward()           # forward_only=False
      |
      +-- PipelinedRunner               # PP>1, 使用 pipeline schedule
            _create_schedule() -> build_pipeline_schedule()
            schedule.step() / schedule.eval()
            _patch_skip_output_merge()  # 避免 output 合并 (内存优化)
            _NullOutputChunks          # train 模式立即释放 logits

    工厂: create_runner(pp_enabled, ...) -> SequentialRunner | PipelinedRunner
```

### 8.5 BaseStateDictAdapter 契约

```
    base.py (L57-L141)

    BaseStateDictAdapter (ABC):
      fqn_to_index_mapping: dict[str, int] | None   # 多文件 safetensors 分片映射

      from_hf(hf_state_dict) -> archon_state_dict    # HF key -> Archon key
      to_hf(archon_state_dict) -> hf_state_dict      # Archon key -> HF key
      convert_single_to_hf(name, tensor)             # 单参数转换 (用于权重同步)
      get_hf_storage_reader(path)                    # HuggingFaceStorageReader
      _maybe_composite_hf_key(hf_key)                # VLM namespace 重映射 (Qwen3.5)
```

______________________________________________________________________

## 9. 跨模块依赖

```
    archon_engine.py
      |-- imports from areal.api (TrainEngine, FinetuneSpec, ParallelStrategy, ...)
      |-- imports from areal.engine.core (distributed, train_engine utilities)
      |-- imports from areal.engine.fsdp_utils (grad clipping, lr scheduler)
      |-- imports from areal.utils (logging, data, functional, lock, offload)
      |-- imports from areal.models.tree_attn (tree attention)
      |-- imports from areal.infra (DistRolloutCoordinator, platforms)
      |
    archon models
      |-- imports from areal.models.fsdp.ulysses (gather/scatter heads)
      |-- imports from areal.models.parallel_styles (ReplicateParallel)
      |-- [FP8] imports from areal.engine.megatron_utils.fp8.kernels (weight_dequant)
      |-- [FP8] imports from torchao.prototype.blockwise_fp8_training
      |-- [Qwen3.5] imports from fla (chunk_gated_delta_rule)
      |-- [Qwen3.5] imports from causal_conv1d (causal_conv1d_fn)
      |
    外部依赖:
      torch.distributed.pipelining      (PipelineStage, 各种 Schedule)
      torch.distributed.tensor          (DTensor, distribute_module/tensor)
      torch.distributed.checkpoint      (DCP, HuggingFaceStorageWriter/Reader)
      transformers                      (AutoConfig, PretrainedConfig)
```
