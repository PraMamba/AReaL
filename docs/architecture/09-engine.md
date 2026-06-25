# 训练与推理引擎层

> 源码位置：`areal/engine/` 文件数：33 个 | 总行数：12,932 行

______________________________________________________________________

## 1. 概述

引擎层是 AReaL 的计算核心，负责模型的分布式训练与远程推理。该层实现了两套训练后端 （FSDP2 与 Megatron）和两套推理后端（SGLang 与
vLLM），通过统一的 `TrainEngine` / `InferenceEngine` 抽象接口向上层工作流暴露能力。

```
                      +------------------+
                      |  TrainEngine API |  (areal/api/)
                      +--------+---------+
                               |
              +----------------+----------------+
              |                                 |
    +---------v----------+          +-----------v---------+
    |    FSDPEngine       |          |   MegatronEngine    |
    |  (fsdp_engine.py)   |          | (megatron_engine.py)|
    |  2,331 行           |          |  2,354 行           |
    +-----+-----+--------+          +-----+------+--------+
          |     |                         |      |
    fsdp_utils/ |                  megatron_utils/|
    (6 文件     |                  (13 文件       |
     1,636 行)  |                   3,098 行)     |
                |                                |
        +-------v--------+              +--------v--------+
        | InferenceEngine |              |   core/         |
        +---+--------+---+              | (4 文件, 388 行) |
            |        |                   +-----------------+
  +---------v--+  +--v-----------+
  |SGLang远程  |  |vLLM远程      |
  |564 行      |  |527 行        |
  +------------+  +--+-----------+
                     |
              vllm_ext/ (2 文件, 747 行)
```

______________________________________________________________________

## 2. 目录与文件清单

| 子目录                       | 文件                         | 行数  | 职责                                                                               |
| ---------------------------- | ---------------------------- | ----- | ---------------------------------------------------------------------------------- |
| `engine/`                    | `__init__.py`                | 50    | 延迟导入注册表                                                                     |
| `engine/`                    | `fsdp_engine.py`             | 2,331 | FSDP2 训练引擎 + 5 个算法变体                                                      |
| `engine/`                    | `megatron_engine.py`         | 2,354 | Megatron 训练引擎 + 5 个算法变体                                                   |
| `engine/`                    | `sglang_remote.py`           | 564   | SGLang 远程推理引擎                                                                |
| `engine/`                    | `vllm_remote.py`             | 527   | vLLM 远程推理引擎                                                                  |
| `engine/core/`               | `__init__.py`                | 15    | 导出 3 个共享函数                                                                  |
| `engine/core/`               | `train_engine.py`            | 144   | `compute_total_loss_weight` / `aggregate_eval_losses` / `reorder_and_pad_outputs`  |
| `engine/core/`               | `distributed.py`             | 139   | `init_custom_process_group` / `warmup_process_groups` / `patch_dist_group_timeout` |
| `engine/core/`               | `model.py`                   | 90    | 模型类型检测 (Qwen-VL/Gemma3/MoE) + `disable_dropout_in_model`                     |
| `engine/fsdp_utils/`         | `__init__.py`                | 206   | `apply_fsdp2` / `fsdp2_load_full_state_dict` / cosine scheduler                    |
| `engine/fsdp_utils/`         | `parallel.py`                | 396   | `ParallelHelper` / `parallelize_model` (TP+SP+DP 并行化)                           |
| `engine/fsdp_utils/`         | `grad.py`                    | 281   | `fsdp2_clip_grad_norm` (跨 DP/TP/PP 梯度裁剪)                                      |
| `engine/fsdp_utils/`         | `optimizer.py`               | 633   | `AnyPrecisionAdamW` + `PerLayerOptimWrapper`                                       |
| `engine/fsdp_utils/`         | `checkpoint.py`              | 63    | `DCPState` (DCP 分布式检查点)                                                      |
| `engine/fsdp_utils/`         | `attn_impl.py`               | 27    | 注意力实现验证                                                                     |
| `engine/fsdp_utils/`         | `multi_tensor_apply.py`      | 30    | TE/Apex 回退的本地 L2 norm / scale                                                 |
| `engine/megatron_utils/`     | `__init__.py`                | 1     | 空                                                                                 |
| `engine/megatron_utils/`     | `megatron.py`                | 1,444 | TP all-gather / HF 权重转换 / FP8 参数处理                                         |
| `engine/megatron_utils/`     | `checkpointer.py`            | 532   | `MegatronCheckpointManager` (异步 DCP 存取)                                        |
| `engine/megatron_utils/`     | `packed_context_parallel.py` | 400   | CP 序列拆分 / 重组 / packed forward                                                |
| `engine/megatron_utils/`     | `pipeline_parallel.py`       | 285   | PP 阶段划分 / 负载均衡层分配                                                       |
| `engine/megatron_utils/`     | `megatron_lora.py`           | 398   | LoRA 权重转换 (Megatron \<-> HF)                                                   |
| `engine/megatron_utils/`     | `deterministic.py`           | 39    | MoE 确定性训练配置                                                                 |
| `engine/megatron_utils/fp8/` | `__init__.py`                | 63    | FP8 子包导出                                                                       |
| `engine/megatron_utils/fp8/` | `config.py`                  | 54    | `get_block_size_from_config`                                                       |
| `engine/megatron_utils/fp8/` | `cuda.py`                    | 38    | SM 版本检测 (Hopper/Blackwell)                                                     |
| `engine/megatron_utils/fp8/` | `deepgemm.py`                | 101   | DeepGEMM JIT 检测 / UE8M0 判断                                                     |
| `engine/megatron_utils/fp8/` | `kernels.py`                 | 156   | Triton blockwise FP8 量化 / 反量化内核                                             |
| `engine/megatron_utils/fp8/` | `quantize.py`                | 199   | 高层 FP8 量化 API                                                                  |
| `engine/megatron_utils/fp8/` | `tensor_helper.py`           | 442   | `FP8BlockwiseTensorHelper` 张量包装器                                              |
| `engine/megatron_utils/fp8/` | `ue8m0.py`                   | 183   | UE8M0 格式工具 (Blackwell 2 的幂缩放)                                              |
| `engine/vllm_ext/`           | `areal_vllm_server.py`       | 400   | vLLM 自定义 FastAPI 端点                                                           |
| `engine/vllm_ext/`           | `vllm_worker_extension.py`   | 347   | vLLM worker 权重更新 / NCCL 组管理                                                 |

______________________________________________________________________

## 3. 类继承体系

### 3.1 训练引擎继承树

```
TrainEngine (areal/api/)                    -- 抽象基类
|
+-- FSDPEngine (fsdp_engine.py:219)         -- FSDP2 实现, 2,331 行
|   +-- FSDPPPOActor   (L2160)              -- PPO Actor
|   +-- FSDPPPOCritic  (L2196)              -- PPO Critic
|   +-- FSDPLMEngine   (L2228)              -- SFT 语言模型
|   +-- FSDPRWEngine   (L2259)              -- 奖励模型
|   +-- FSDPDPOEngine  (L2293)              -- DPO 训练
|
+-- MegatronEngine (megatron_engine.py:168) -- Megatron 实现, 2,354 行
    +-- MegatronPPOActor   (L2183)
    +-- MegatronPPOCritic  (L2219)
    +-- MegatronLMEngine   (L2251)
    +-- MegatronRWEngine   (L2282)
    +-- MegatronDPOEngine  (L2316)
```

### 3.2 推理引擎继承树

```
InferenceEngine (areal/api/)                -- 抽象基类
|
+-- RemoteSGLangEngine (sglang_remote.py:315)
|   内部组合: RemoteInfEngine + SGLangBackend
|
+-- RemotevLLMEngine (vllm_remote.py:277)
    内部组合: RemoteInfEngine + VLLMBackend
```

两个推理引擎均采用 **组合模式**：`RemoteSGLangEngine` / `RemotevLLMEngine` 持有内部 `RemoteInfEngine`
实例和对应的 `SGLangBackend` / `VLLMBackend` 策略对象，所有公共方法 委托给 `_engine`。`Backend` 策略类负责构造 HTTP
请求、解析响应、生成/暂停/恢复命令等 后端特定逻辑。

______________________________________________________________________

## 4. 核心数据流

### 4.1 train_batch 微批处理流程

两种引擎的 `train_batch` 均遵循相同的四步流水线：

```
输入 input_                         输出 stats
  |                                    ^
  v                                    |
[Step 1] _prepare_mb_list          [Step 4] optimizer_step
  |  序列打包 -> 拆分微批 ->            |  梯度裁剪 -> 参数更新
  |  对齐填充 -> 设备迁移               |  学习率调度
  v                                    |
[Step 2] compute_total_loss_weight [Step 3] forward_backward_batch
  |  loss_weight_fn 求和 ->            |  微批逐个前向 ->
  |  all_reduce (DP 组)               |  logprobs/entropy ->
  v                                    |  loss_fn -> backward
  +------------------------------------+
```

**FSDP 引擎** 在 `forward_backward_batch` 中直接迭代微批列表，对每个微批执行：

1. `_prepare_mb_inputs` -- Ulysses SP 填充与切片
1. `model(**inputs)` -- HuggingFace 模型前向
1. `process_output_fn(logits, ctx_dict)` -- 计算 loss
1. `loss.backward()` -- 反向传播

**Megatron 引擎** 的 `forward_backward_batch` 额外包装了 Megatron Core 的流水线调度器：

1. 定义 `forward_step` 闭包（处理 CP 拆分 + PP 阶段）
1. 调用 `get_forward_backward_func()` 获取调度函数（1F1B 或 interleaved）
1. 调度器自动处理微批在 PP 阶段间的流水推进

### 4.2 forward_batch 推理流程

```
input_ --> _normalize_batch_input --> _prepare_mb_list --> forward_backward_batch(forward_only=True)
                                                              |
                                                              v
                                                     _compute_forward_result
                                                     (gather_logprobs / values)
                                                              |
                                                              v
                                                     reorder_and_pad_outputs
                                                     (聚合 + 反序 + 填充)
```

Megatron 路径在 PP 末端阶段收集结果后，通过 `broadcast_tensor` 广播到所有 PP 阶段。

______________________________________________________________________

## 5. FSDP vs Megatron 并行策略对比

| 维度              | FSDP2 引擎                                                    | Megatron 引擎                                   |
| ----------------- | ------------------------------------------------------------- | ----------------------------------------------- |
| **并行维度**      | DP + SP (Ulysses) + TP                                        | DP + CP + TP + PP + EP + ETP + VPP              |
| **PP 支持**       | 不支持 (断言 `pp_size==1`)                                    | 原生支持, 含 VPP                                |
| **Mesh 构建**     | `ParallelHelper.build_mesh` (torch DeviceMesh)                | `mpu.initialize_model_parallel` (Megatron 内置) |
| **模型创建**      | HuggingFace `from_pretrained` / `from_config`                 | `make_mcore_model` (mbridge/megatron-bridge)    |
| **模型分片**      | `fully_shard` (FSDP2 API) + `parallelize_module` (DTensor TP) | Megatron DDP + TP + PP 原生分片                 |
| **优化器**        | PyTorch AdamW / AnyPrecisionAdamW / SGD                       | `get_megatron_optimizer` (分布式优化器)         |
| **LR 调度**       | HuggingFace schedulers + 自定义 cosine                        | Megatron `OptimizerParamScheduler`              |
| **梯度裁剪**      | `fsdp2_clip_grad_norm` (自定义, 跨 DP+TP+PP)                  | Megatron 内置 (`clip_grad` 参数)                |
| **FP8**           | 不支持                                                        | 完整支持 (TE + Triton + DeepGEMM)               |
| **检查点**        | HF format + DCP (torch.distributed.checkpoint)                | HF format + Megatron DCP (dist_checkpointing)   |
| **LoRA**          | PEFT `get_peft_model`                                         | `MegatronBridgeLoRA`                            |
| **VLM**           | `AutoModelForImageTextToText`                                 | mbridge/megatron-bridge VLM 支持                |
| **tree training** | Ulysses SP 互斥                                               | CP 互斥                                         |

______________________________________________________________________

## 6. 关键机制详解

### 6.1 权重更新机制

训练引擎将更新后的参数同步到推理引擎（SGLang/vLLM），支持两种模式：

#### from_distributed (XCCL/NCCL)

```
FSDPEngine rank 0                     SGLang/vLLM workers
      |                                      |
      +-- pause_generation() ----HTTP---->   |
      |                                      |
      +-- _get_full_tensor(param)            |
      |   (DTensor.full_tensor 集体操作)      |
      |                                      |
      +-- _cast_to_compute_dtype             |
      |   (fp32 master -> bf16)              |
      |                                      |
      +-- update_weights_from_distributed    |
      |   (发送 ParamSpec 列表)               |
      |                                      |
      +-- dist.broadcast(tensor,             |
      |     src=0, group=update_group)       |
      |                              <--NCCL-+ (接收权重)
      |                                      |
      +-- continue_generation() --HTTP--->   |
```

**分桶流水化**：大模型的参数按 `weight_chunked_mem_mb` 分桶，采用 single-pending-bucket 异步广播流水线：

- 前一桶的 broadcast 通过 `async_op=True` 在独立 CUDA stream 上执行
- 主流程同时准备下一桶的 `_get_full_tensor`
- 通过 `_wait_pending_weight_update_bucket` 在桶切换时同步

**PP 多组**：当推理端 PP > 1 时，为每个 PP 阶段创建独立的 NCCL 组 （`update_weight_group_{pp_rank}`），避免 PP
事件循环死锁。

#### from_disk

```
train engine                          推理引擎
     |                                   |
     +-- pause_generation() ---->        |
     +-- _save_model_to_hf(path)         |
     +-- update_weights_from_disk -->    |
     |                                   +-- 从 path 重新加载
     +-- continue_generation() --->      |
```

### 6.2 梯度裁剪 (fsdp2_clip_grad_norm)

```
fsdp2_clip_grad_norm(parameters, max_norm, fsdp_group, tp_group, pp_group)
    |
    +-- get_main_grads_for_grad_norm
    |   过滤 TP 重复参数 (仅 tp_rank==0 保留复制参数)
    |
    +-- get_grad_norm_fp32
    |   |-- multi_tensor_l2norm (TE/Apex/本地回退)
    |   |-- all_reduce(SUM, dp_group)   -- 跨 DP 汇聚
    |   +-- all_reduce(SUM, tp_group)   -- 跨 TP 汇聚
    |
    +-- all_reduce(pp_group)            -- 跨 PP 汇聚 (L2: SUM+root; inf: MAX)
    |
    +-- clip_grad_by_total_norm_fp32
        |-- clip_coeff = max_norm / (total_norm + 1e-6)
        +-- multi_tensor_scale (就地缩放)
```

TE/Apex 可用时使用 CUDA 多张量操作加速；不可用时回退到 `local_multi_tensor_l2_norm` /
`local_multi_tensor_scale`（纯 PyTorch 实现）。

### 6.3 AnyPrecisionAdamW 优化器

```
AnyPrecisionAdamW (fsdp_utils/optimizer.py:44)

    exp_avg      : bf16 (momentum)
    exp_avg_sq   : bf16 (variance)
    compensation : bf16 (Kahan summation 误差缓冲)

    更新步骤:
    1. weight_decay: p.data *= (1 - lr * wd)
    2. momentum:     exp_avg = beta1 * exp_avg + (1-beta1) * grad
    3. variance:     exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * grad^2
    4. Kahan summation:
       compensation += -step_size * exp_avg / centered_variance
       temp = p.data.clone()
       p.data += compensation
       compensation += (temp - p.data)   <-- 回收舍入误差
```

Kahan summation 使得 bf16 优化器状态可达到接近 fp32 的精度，内存节省约 50%。

### 6.4 PerLayerOptimWrapper -- 逐层流水化优化器

当开启 `per_layer_optim_step` 时，优化器状态常驻 CPU，通过三流流水线加速：

```
H2D stream         Compute stream       D2H stream
    |                    |                    |
[prefetch L0]            |                    |
[prefetch L1]            |                    |
    |              [compute L0]               |
[prefetch L2]            |             [offload L0]
    |              [compute L1]               |
[prefetch L3]            |             [offload L1]
    ...                 ...                  ...
```

- `_prefetch_layer`: 将某层的 `exp_avg`/`exp_avg_sq`/`step` 从 pinned CPU 异步拷贝到 GPU
- `_compute_for_layer`: 调用 `torch.optim.adam.adam` 内核执行更新
- `_offload_layer`: 将更新后的状态和参数异步拷回 CPU
- `_record_streams_for_layer`: 确保分配器安全（防止跨流提前回收）

### 6.5 FP8 量化支持

FP8 量化仅在 Megatron 引擎中支持，分层架构如下：

```
megatron_utils/fp8/
|
+-- cuda.py          -- SM 版本检测 (Hopper >= 90, Blackwell >= 100)
+-- deepgemm.py      -- DeepGEMM JIT 可用性检测, UE8M0 判断
+-- config.py        -- 从 quantization_config 提取 block_size
+-- kernels.py       -- Triton 内核: blockwise_cast_to_fp8_triton / weight_dequant
+-- ue8m0.py         -- UE8M0 格式: 仅存指数位 (2的幂缩放, Blackwell 专用)
|   +-- ceil_to_ue8m0         -- 向上取整到2的幂
|   +-- quant_weight_ue8m0    -- UE8M0 权重量化
|   +-- transform_scale_ue8m0 -- TMA 对齐 + packed int32 布局
+-- quantize.py      -- 高层 API: quantize_params / dequantize_params
+-- tensor_helper.py -- FP8BlockwiseTensorHelper (torch.Tensor 子类)
                        自动管理 data + scale_inv 的联动操作
```

**量化流程**：

```
bf16/fp32 weight
    |
    v
blockwise_cast_to_fp8_triton (Triton 内核)
    |  -- 按 [BLOCK_M, BLOCK_N] 分块 (默认 128x128)
    |  -- 每块计算 absmax, scale = absmax / FP8_MAX
    |  -- 量化: clamp(x / scale, FP8_MIN, FP8_MAX)
    v
(fp8_e4m3fn weight, float32 scale)
    |
    v [Blackwell GPU 且 DeepGEMM 可用]
quant_weight_ue8m0
    |  -- scale 取指数位 -> uint8
    |  -- 4 个 uint8 打包为 1 个 int32
    |  -- TMA 16 字节对齐
    v
(fp8 weight, packed_ue8m0 scale)
```

### 6.6 Context Parallel (Megatron 专用)

```
packed_context_parallel.py

原始序列: [s0 s1 s2 s3 s4 s5 s6 s7]  (8 tokens, CP=2)

CP 拆分 (交错模式, 用于因果掩码负载均衡):
  GPU 0: [s0 s1] + [s6 s7]  (前半 + 后半)
  GPU 1: [s2 s3] + [s4 s5]  (中间两段)

+-- preprocess_packed_seqs_context_parallel
|   将 packed sequences 按交错模式拆分到 CP rank
|   返回 (splitted_ids, PackedSeqParams)
|
+-- split_packed_seqs_for_context_parallel
|   对 labels 做相同拆分 (用于 logprobs 计算)
|
+-- packed_context_parallel_forward
|   调用 model forward, 可选 gather CP 输出
|
+-- reassemble_cp_packed_logprobs
    CP all-gather 后按交错模式重组完整 logprobs
```

### 6.7 Pipeline Parallel 阶段划分

```
pipeline_parallel.py

configure_pipeline_layer_splits(parallel_strategy, hf_config, tf_config)
    |
    +-- estimate_stage_parameter_buckets
    |   估算每层参数量 (含 MoE expert 不均匀分布)
    |
    +-- _compute_stage_layer_lengths
    |   基于参数量负载均衡的层分配算法:
    |   - 考虑 embedding 和 output head 的额外开销
    |   - 分配到 pp_size * vpp_size 个阶段
    |   - 确保每阶段至少 1 层
    |
    +-- PipelineParallelLayerLayout
        生成阶段布局: [["embedding", "decoder"*N],
                       ["decoder"*M],
                       ...,
                       ["decoder"*K, "loss"]]
```

______________________________________________________________________

## 7. 推理引擎对比

### 7.1 SGLang vs vLLM 后端

| 特性                | SGLangBackend                                              | VLLMBackend                                                          |
| ------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------- |
| **生成端点**        | `/generate`                                                | `/v1/completions` 或 `/v1/chat/completions`                          |
| **请求格式**        | `input_ids` + 嵌套 `sampling_params`                       | 扁平 payload (top_p/top_k/max_tokens 在顶层)                         |
| **logprobs 解析**   | `meta_info.output_token_logprobs`                          | `choices[0].logprobs.tokens` (需解析 "token:123" 格式)               |
| **分布式权重更新**  | 单端点 `/update_weights_from_distributed`                  | 两步: `/areal_set_update_weight_meta` + `/areal_update_weights_xccl` |
| **LoRA 分布式更新** | 不支持 (抛出 ValueError)                                   | 支持 (`_lora` 变体端点)                                              |
| **PP 多组支持**     | 完整支持 (Scenario 2, `pp_rank` payload)                   | 基础支持 (`rank_offset` 计算)                                        |
| **暂停/恢复**       | `/pause_generation` + `/continue_generation`               | `/areal_pause_generation` + `/areal_continue_generation`             |
| **模型卸载**        | `/release_memory_occupation` + `/resume_memory_occupation` | `/sleep` + `/wake_up`                                                |
| **routed_experts**  | 支持 (base64 编码 int32 数组)                              | 不支持                                                               |

### 7.2 vLLM 扩展

`vllm_ext/` 包含两个文件用于在 vLLM 服务端注入 AReaL 专用功能：

- **`areal_vllm_server.py`** (400 行)：基于 FastAPI 的自定义路由，添加
  `/areal_update_weights`、`/areal_init_weights_update_group`、
  `/areal_update_weights_xccl`、`/areal_pause_generation` 等端点
- **`vllm_worker_extension.py`** (347 行)：`VLLMWorkerExtension` 混入类，在 vLLM worker
  中实现权重热更新（全量和 LoRA）以及 NCCL 进程组管理

______________________________________________________________________

## 8. 设计决策与关键约束

### 8.1 延迟导入机制

`engine/__init__.py` 采用 `__getattr__` 延迟导入，避免在只需 FSDP 时加载 Megatron
依赖（反之亦然）。`_LAZY_IMPORTS` 字典映射 14 个公共类名到各自模块路径。

### 8.2 进程组管理

- **自定义进程组** (`init_custom_process_group`)：复用 PyTorch 内部 API 创建独立于
  默认全局组的进程组，用于训练-推理间的权重同步
- **通信器预热** (`warmup_process_groups`)：在初始化阶段执行虚拟 all-reduce，强制 NCCL/HCCL
  通信器提前初始化，避免运行时竞态导致的 HCCP 初始化失败 (issue #1099)
- **CPU 组**：每个引擎创建 gloo 后端的 CPU 进程组用于 barrier 同步，避免 NCCL 组 在 teardown 时的竞态问题

### 8.3 内存管理

- **torch_memory_saver**：`offload()` / `onload()` 通过 `torch_memory_saver.pause()` /
  `resume()` 实现 GPU 显存与 CPU 内存间的整体交换，需配置 `enable_offload=True`
- **\_offload_aware_context**：可重入上下文管理器，嵌套调用仅在最外层执行实际的 onload/offload
- **meta device 加载**：FSDP 的 `memory_efficient_load` 模式下，非 rank 0 使用 meta device
  创建模型（零内存），权重由 rank 0 通过 `fsdp2_load_full_state_dict` 广播

### 8.4 精度对齐

- **fp32 master weights**：FSDP 引擎支持 `optimizer_dtype=float32` 存储参数， FSDP2
  `MixedPrecisionPolicy` 在前向/反向中自动转换为 `config.dtype` (通常 bf16)
- **导出时转换**：`_cast_to_compute_dtype` 确保 HF 导出和 XCCL 权重同步使用计算精度
- **adam_bf16 互操作**：Megatron 引擎自动将 `adam_bf16` 转换为 `adam` +
  `use_precision_aware_optimizer`

### 8.5 算法变体的组合模式

5 个算法变体 (PPOActor, PPOCritic, LMEngine, RWEngine, DPOEngine) 通过 **组合而非深层继承** 实现：

```
class FSDPPPOActor(FSDPEngine):          # 继承引擎基础设施
    def __init__(self, config):
        super().__init__(config)
        self.actor = PPOActor(config, self)  # 组合: 持有算法实例

    def ppo_update(self, *args, **kwargs):
        self.actor.ppo_update(*args, **kwargs)  # 委托调用
```

每个变体类仅约 30-40 行，职责清晰：继承引擎基础设施，组合算法逻辑。

### 8.6 VLM 多模态支持

- FSDP 引擎使用 `AutoModelForImageTextToText` 加载 VLM
- 支持 Qwen2-VL、Qwen2.5-VL、Qwen3-VL(含 MoE)、Gemma3
- `_prepare_multimodal_forward_inputs`：将多模态 payload（pixel_values 等） 从原始微批移动到 forward
  微批，避免重复
- VLM 权重同步时需要名称映射 (`_get_model_name_parameters`)， SGLang 和 vLLM 使用不同的命名约定

### 8.7 Tree Training 支持

两个引擎均支持 tree training（前缀共享的树结构训练），通过：

- `build_packed_tree_batch`：将多条序列打包为前缀树
- `build_tree_attn_kwargs`：构建树注意力的 mask 元数据
- `gather_packed_tree_logprobs_entropy`：从树结构中正确提取各序列的 logprobs
- 约束：FSDP 下与 Ulysses SP 互斥，Megatron 下与 CP 互斥
