# AReaL 分布式训练后端与并行策略深度分析

> 基于源码的详细分析，覆盖 Archon 后端如何组合 FSDP2 + TP + SP + PP + EP，
> 以及 MoE 模型下 EP AllGather 同步开销与推理端权重接收机制。

---

## 目录

1. [三种训练后端概览](#1-三种训练后端概览)
2. [Archon 并行维度架构：DeviceMesh 层级](#2-archon-并行维度架构devicemesh-层级)
3. [五维并行的组合顺序与原理](#3-五维并行的组合顺序与原理)
4. [Expert Parallelism 的 All-to-All 通信机制](#4-expert-parallelism-的-all-to-all-通信机制)
5. [Pipeline Parallelism 与 FSDP2 的协作](#5-pipeline-parallelism-与-fsdp2-的协作)
6. [EP 下的 AllGather 同步开销分析](#6-ep-下的-allgather-同步开销分析)
7. [推理端无缝接收跨 EP Rank 权重](#7-推理端无缝接收跨-ep-rank-权重)
8. [设计总结](#8-设计总结)

---

## 1. 三种训练后端概览

| 后端 | 并行能力 | 模型格式 | 适用场景 |
|------|---------|---------|---------|
| **FSDP** | FSDP2 + TP + CP | HuggingFace 原生 | 快速实验、中等规模 |
| **Megatron** | TP + PP + EP + CP | Megatron 自定义 | 大规模生产、MoE |
| **Archon** | FSDP2 + TP + SP + PP + EP + ETP | HuggingFace 原生 (via adapter) | PyTorch 原生全组合 |

Archon 的独特定位：使用 **PyTorch 原生 API（DTensor + DeviceMesh + FSDP2）** 实现
Megatron 级别的并行组合，避免依赖 Megatron 的自定义框架。

---

## 2. Archon 并行维度架构：DeviceMesh 层级

### 2.1 ArchonParallelDims 核心设计

**源码**: `areal/experimental/models/archon/parallel_dims.py:24-419`

```python
@dataclass
class ArchonParallelDims:
    dp_shard: int = -1   # FSDP 分片维度（自动计算）
    cp: int = 1          # Context Parallel (Ulysses SP)
    tp: int = 1          # Tensor Parallel
    pp: int = 1          # Pipeline Parallel
    ep: int = 1          # Expert Parallel
    etp: int = 1         # Expert Tensor Parallel (1 或 tp)
    world_size: int = 1
```

**基本约束**: `world_size = pp × dp_shard × cp × tp`

### 2.2 无 EP 时的 4D Mesh

当 `ep=1` 时，创建 4 维 DeviceMesh：

```
DeviceMesh 维度: [pp, dp_shard, cp, tp]

示例: 8 GPU, pp=2, dp_shard=2, cp=1, tp=2
  ┌──────────────────────────────────────┐
  │  PP Stage 0:                          │
  │    DP0: [GPU 0 (TP0), GPU 1 (TP1)]   │
  │    DP1: [GPU 2 (TP0), GPU 3 (TP1)]   │
  ├──────────────────────────────────────┤
  │  PP Stage 1:                          │
  │    DP0: [GPU 4 (TP0), GPU 5 (TP1)]   │
  │    DP1: [GPU 6 (TP0), GPU 7 (TP1)]   │
  └──────────────────────────────────────┘

派生子 Mesh:
  dp_shard_cp = dp_shard × cp  → FSDP 参数分片
  dp_cp = dp_shard × cp        → 损失 all-reduce
  pp_cp_tp = pp × cp × tp      → 数据广播到所有模型并行 rank
```

### 2.3 有 EP 时的 5D Mesh — EP 从 DP 中"借"维度

当 `ep>1` 时，数据并行维度被拆分以容纳 EP：

```
DeviceMesh 维度: [pp, dp_shard_mod_ep, dp_shard_in_ep, cp, tp]
```

**关键设计**：EP 不独立占用维度，而是从 DP 空间中"借"rank：

| etp 值 | EP 借用范围 | dp_shard_mod_ep | dp_shard_in_ep |
|--------|-----------|-----------------|----------------|
| **etp=1** (TP 被 EP 借用) | `dp_shard × cp × tp` | `dp_shard×cp×tp / ep` | `ep / (cp×tp)` |
| **etp=tp** (TP 保持独立) | `dp_shard × cp` | `dp_shard×cp / ep` | `ep / cp` |

**etp=1 的含义**：TP 维度被 EP "吞并"——每个 TP rank 处理不同的 token 子集
（通过 `ReordererSequenceParallel`），而非处理相同 token 的不同权重分片。

**etp=tp 的含义**：TP 维度保持独立，专家权重在 EP 和 TP 两个维度上进行 2D 分片。
额外创建 `ep_tp` 2D mesh 用于 `ExpertTensorParallel`。

**完整示例** (4 GPU, dp_shard=2, tp=2, ep=2, etp=1):

```
EP 借用范围 = dp_shard × cp × tp = 2 × 1 × 2 = 4
dp_shard_mod_ep = 4 / 2 = 2
dp_shard_in_ep = 2 / (1 × 2) = 1

DeviceMesh [pp=1, dp_shard_mod_ep=2, dp_shard_in_ep=1, cp=1, tp=2]:
  GPU 0: (mod_ep=0, in_ep=0, cp=0, tp=0)
  GPU 1: (mod_ep=0, in_ep=0, cp=0, tp=1)
  GPU 2: (mod_ep=1, in_ep=0, cp=0, tp=0)
  GPU 3: (mod_ep=1, in_ep=0, cp=0, tp=1)

EP mesh (flatten dp_shard_in_ep × cp × tp):
  EP Group 0: [GPU 0, GPU 1]  ← 持有 Expert 0-31 (假设 64 experts)
  EP Group 1: [GPU 2, GPU 3]  ← 持有 Expert 32-63
```

---

## 3. 五维并行的组合顺序与原理

### 3.1 并行化应用顺序

**源码**: `areal/experimental/models/archon/qwen3/infra/parallelize.py:84-131`

```python
def parallelize_qwen3(model, parallel_dims, ...):
    # 严格按顺序应用:
    # 1. TP for dense layers    → parallelize_module() with DTensor plans
    # 2. EP+TP for MoE layers   → distribute_module() with EP/ETP strategies
    # 3. CP (Ulysses SP)        → set_cp_group() on attention modules
    # 4. AC                     → wrap with CheckpointWrapper
    # 5. torch.compile          → compile transformer blocks
    # 6. FSDP2                  → fully_shard() on all components
```

**为什么这个顺序是必要的**:

```
① TP 必须最先: parallelize_module 设置 DTensor layout（Shard/Replicate）
   → 参数变为 DTensor，后续操作必须在 DTensor 之上

② EP 紧随 TP: distribute_module 在 TP DTensor 之上添加 EP 分片
   → 对于 etp=tp: 形成 2D DTensor [Shard(0), Shard(1/2)]
   → 必须在 FSDP 之前，因为 FSDP 会改变参数生命周期

③ CP 在 TP/EP 之后: set_cp_group 只设置 attention 模块的通信组
   → 不改变参数结构，无序依赖

④ AC 在 TP/EP 之后: 包装 CheckpointWrapper 依赖正确的模块结构
   → 但必须在 compile 之前

⑤ compile 在 AC 之后、FSDP 之前: FSDP hooks 干扰 torch.compile
   → 特别是 MoE 的 token dispatch 需要特殊处理

⑥ FSDP2 最后: 作为最外层包装管理参数生命周期
   → 在已经 TP/EP 分片的 DTensor 之上再做 FSDP 分片
```

### 3.2 TP 应用详解

**源码**: `parallelize.py:196-297` (`apply_non_moe_tp`)

```
Embedding:       RowwiseParallel(output=Shard(1))  → 词汇复制，输出沿序列维分片
Attention Q/K/V: ColwiseParallel(use_local_output=True) → 按头数切分
Attention Wo:    RowwiseParallel(output=Shard(1))  → 行切分 + all-reduce
FFN w1/w3:       ColwiseParallel  → 列切分
FFN w2:          RowwiseParallel(output=Shard(1))  → 行切分
Norms:           SequenceParallel  → 序列并行
Output Head:     ColwiseParallel(output=Shard(-1)) → 词汇并行
```

**Sequence Parallel 模式**: Norm 层操作在序列维分片的数据上，Attention/FFN 通过隐式
all-gather 转为完整输入、计算后 scatter 回序列分片。

### 3.3 EP + TP 策略矩阵

**源码**: `parallelize.py:632-746` (`apply_moe_ep_tp`)

```python
# 策略选择（parallel_dims.py:46-55 的文档化版本）:

if ep == 1 and tp == 1:
    strategy = None                    # 完全复制
elif ep == 1 and tp > 1:
    strategy = TensorParallel          # 纯 TP: [Shard(1/2)]
elif ep > 1 and etp == 1:
    strategy = ExpertParallel          # EP 借用 TP: [Shard(0)]
elif ep > 1 and etp == tp:
    strategy = ExpertTensorParallel    # 2D: [Shard(0), Shard(1/2)]
```

### 3.4 FSDP2 对 MoE 的差异化处理

**源码**: `parallelize.py:300-431` (`apply_fsdp`)

```python
# Dense 参数使用完整 DP mesh
dp_mesh = parallel_dims.get_mesh("dp_shard_cp")  # dp_shard × cp

# Expert 参数使用缩小的 mesh（EP 借走了部分 DP rank）
if parallel_dims.ep_enabled:
    dp_mod_ep_mesh = parallel_dims.get_mesh("dp_shard_mod_ep")  # 更小

# 梯度归一化: 确保 dense 和 expert 的梯度缩放一致
for block in model.layers:
    if hasattr(block, "moe"):
        block.moe.experts.set_gradient_divide_factor(
            parallel_dims.fsdp_gradient_divide_factor
        )
```

**为什么需要不同的 mesh**: EP 从 DP 空间借走了 rank，Expert 参数的 FSDP 分片只在剩余的
`dp_shard_mod_ep` rank 上进行。Dense 参数不受 EP 影响，使用完整的 `dp_shard_cp`。

---

## 4. Expert Parallelism 的 All-to-All 通信机制

### 4.1 ExpertParallel (etp=1)

**源码**: `areal/experimental/models/archon/expert_parallel.py:68-237`

**前向传播的 Token Dispatch**:

```
输入: routed_input [total_tokens, dim], num_tokens_per_expert [num_experts]

Step 1: 计算 split sizes
  local_experts_per_rank = num_experts // ep_size
  input_splits = [sum of tokens for local experts on each rank]

  → All-to-All 交换 split sizes (确定每个 rank 发/收多少 token)

Step 2: Variable-size All-to-All dispatch
  dist.all_to_all(output_list, input_list, group=ep_group)

  → 每个 rank 发送"要去其他 rank 的 expert 处理的 token"
  → 每个 rank 接收"来自其他 rank 的、要在本 rank expert 处理的 token"

Step 3: Permute for grouped_mm alignment
  使用 Triton kernel 按 expert 分组排列 token

  → 生成 (dispatched_tokens, dim) 格式供 grouped_mm 计算
```

```
数据流:

Rank 0 (experts 0-3)          Rank 1 (experts 4-7)
  tokens → expert 0-3           tokens → expert 4-7
  tokens → expert 4-7 ──all2all──→ ← 接收
  ← 接收 ──all2all── tokens → expert 0-3

  计算 expert 0-3              计算 expert 4-7

  results ← ──all2all──── ← results (reverse)
```

### 4.2 ExpertTensorParallel (etp=tp)

**源码**: `expert_parallel.py:331-433`

在 2D `[ep, tp]` mesh 上进行双重分片：

```python
# 权重分片策略:
w1/w3 (gate/up proj): [Shard(0), Shard(1)]  → expert 维度 EP 分片 + 列 TP 分片
w2 (down proj):       [Shard(0), Shard(2)]  → expert 维度 EP 分片 + 行 TP 分片
```

**通信模式**: Token dispatch 的 All-to-All 只在 EP 维度上进行（`device_mesh["ep"]`），
TP 维度的 all-reduce 在 expert 计算内部隐式处理（通过 DTensor 的 Partial placement）。

### 4.3 ReordererSequenceParallel (etp=1 去重)

**源码**: `expert_parallel.py:436-503`

当 etp=1 时，TP 被 EP 借用，但 dense 层仍用 TP。这意味着每个 TP rank 在 dense 层
看到相同的 token，但在 MoE 层需要看到不同的 token（否则 EP All-to-All 会发送重复数据）。

`ReordererSequenceParallel` 解决这个问题：

```python
# TokenReorderer 的序列并行包装
# 将 token 沿 TP 维度分割，每个 TP rank 处理不同的 token 子集
# 调整 token 索引从本地坐标到全局坐标
```

---

## 5. Pipeline Parallelism 与 FSDP2 的协作

### 5.1 PP Stage 划分

**源码**: `areal/experimental/models/archon/pipeline_parallel.py:83-195`

```python
def generate_llm_fqn_per_model_part(model, parallel_dims):
    # 将 transformer layers 分配到 PP stages
    # 第一个 stage: tok_embeddings + 较少的 transformer layers
    # 中间 stages: transformer layers
    # 最后一个 stage: transformer layers + norm + output/score
```

**调度策略** (`pipeline_parallel.py:407-449`):

| 调度方式 | 每 rank stage 数 | 分配模式 | 气泡率 |
|----------|----------------|---------|--------|
| 1F1B | 1 | rank i → stage i | 标准 |
| Interleaved1F1B | ≥2 | 循环: rank 0 → stage 0,4; rank 1 → 1,5 | 较低 |
| InterleavedZeroBubble | ≥2 | 同上 | 极低 |
| ZBVZeroBubble | ≥2 | V 形: rank 0 → stage 0, N-1 | 极低 |

### 5.2 PP + FSDP2 + TP 的协作模式

**关键**: PP stage 独立并行化。每个 stage 单独应用 TP + EP + CP + AC + FSDP：

```python
# pipeline_parallel.py:477-481
for model_part, stage in zip(model_parts, stages):
    parallelize_fn(model_part)   # 每个 stage 独立应用 TP+EP+CP+AC+FSDP
    stage.submod = model_part    # 更新 stage 的子模块引用
```

```
PP Stage 0 (rank 0)              PP Stage 1 (rank 1)
┌───────────────────┐            ┌───────────────────┐
│ tok_embeddings    │            │ transformer_layer_4│
│ transformer_layer_0│            │ transformer_layer_5│
│ transformer_layer_1│            │ transformer_layer_6│
│ transformer_layer_2│            │ transformer_layer_7│
│ transformer_layer_3│            │ norm + output      │
│                   │            │                   │
│ ┌── 独立 FSDP ──┐ │            │ ┌── 独立 FSDP ──┐ │
│ │ 独立 TP       │ │   PP通信    │ │ 独立 TP       │ │
│ │ 独立 EP       │ │ ←──────→  │ │ 独立 EP       │ │
│ │ 独立 CP       │ │            │ │ 独立 CP       │ │
│ └───────────────┘ │            │ └───────────────┘ │
└───────────────────┘            └───────────────────┘
```

PP 维度与 TP/CP/EP 正交——PP 是 stage 间的通信，TP/EP/CP 是 stage 内的通信。

### 5.3 FSDP2 在 PP 模式下的 Reshard 策略

```python
# parallelize.py:344-356
if pp_enabled:
    # PP 模式下默认不 reshard，避免每个 microbatch 重复 all-gather
    reshard_after_forward = False  # 保留 unsharded 参数
else:
    reshard_after_forward = True   # 正常 reshard 节省显存
```

---

## 6. EP 下的 AllGather 同步开销分析

### 6.1 Megatron 引擎的两阶段 EP AllGather

**源码**: `areal/engine/megatron_engine.py:1133-1224`

```python
def _update_bucket_expert_weights_from_distributed(self, meta, named_tensors):
    group = mpu.get_expert_model_parallel_group()
    world_size = mpu.get_expert_model_parallel_world_size()

    # Step 1: 交换参数名称（验证一致性）
    all_names = [None] * world_size
    dist.all_gather_object(all_names, names, group=group)

    # Step 2: AllGather 每个参数的张量数据
    for idx, (_, tensor) in enumerate(named_tensors):
        params = [torch.empty_like(tensor.data, device=device) for _ in range(world_size)]
        handle = dist.all_gather(params, tensor.data, group=group, async_op=True)
        handles.append(handle)

    # Step 3: 等待所有 AllGather 完成
    for handle in handles:
        handle.wait()

    # Step 4: 只有 PP head 进行格式转换 + broadcast
    if not self.is_pipeline_parallel_head():
        return  # 其他 rank 丢弃 gathered 数据

    # Step 5: 展平所有 EP rank 的参数 → HF 格式
    gathered_params = sum(gathered_params, [])  # 合并所有 EP rank
    converted_hf_tensors = []
    for name, param in gathered_params:
        converted_hf_tensors.extend(convert_to_hf(..., name, param))
```

### 6.2 Archon 引擎的隐式 AllGather

**源码**: `areal/experimental/engine/archon_weight_sync.py:109-171`

```python
def update_weights_from_distributed(state, meta, engine):
    for name, param in engine._get_model_name_parameters():
        tensor = _get_full_tensor(param)  # DTensor.full_tensor() → 隐式 AllGather
        hf_pairs = engine.state_dict_adapter.convert_single_to_hf(name, tensor)
```

Archon 利用 DTensor 的 `full_tensor()` 方法，**隐式触发所有必要的集合操作**
（FSDP all-gather + TP all-gather + EP all-gather）。无需显式管理 EP 组。

### 6.3 通信量分析

以 **DeepSeek-V3 架构**为例（256 experts, hidden=7168, intermediate=18432, 61 MoE layers）：

**单 Expert 的参数量**:
```
gate_proj: 7168 × 18432 × 2 bytes = 252 MB
up_proj:   7168 × 18432 × 2 bytes = 252 MB
down_proj: 18432 × 7168 × 2 bytes = 252 MB
单 Expert 总计: ~756 MB
```

**EP AllGather 通信量** (假设 EP=8, 每 rank 32 experts):
```
每 rank 发送: 32 × 756 MB = ~23.6 GB
每 rank 接收: 7 × 32 × 756 MB = ~165 GB (ring all-gather)
每 rank 总通信: ~188.6 GB

61 MoE layers 总计: 61 × 188.6 GB = ~11.5 TB 每 rank

在 400 Gbps NIC 下: 11.5 TB / 50 GB/s ≈ 230 秒 ← 非常大！
```

**但实际不会这么慢**，因为：

1. **分 bucket 传输**: `weight_chunked_mem_mb × ep_size` 限制每次 AllGather 的数据量
2. **AllGather 与 broadcast 重叠**: bucket 内的 AllGather 使用 `async_op=True`
3. **NVLink/NVSwitch**: EP group 通常在同一节点内（NVLink ~600 GB/s），跨节点通信远少于理论值
4. **只有 PP head 参与后续 broadcast**: 非 PP head 的 AllGather 仍必须参与但立即丢弃结果

### 6.4 峰值显存消耗

```
EP AllGather 峰值显存 (per bucket):

输入: N 个 local expert 参数, 总大小 S
输出: N × ep_size 个 buffer, 总大小 S × ep_size

bucket 限制: S × ep_size ≤ weight_chunked_mem_mb
→ 实际 S ≤ weight_chunked_mem_mb / ep_size

例: weight_chunked_mem_mb=1024, ep_size=8
→ 每 bucket 最多 128 MB 的本地参数
→ AllGather 后 128 MB × 8 = 1024 MB
→ 峰值额外显存: ~1 GB (可配置)
```

---

## 7. 推理端无缝接收跨 EP Rank 权重

### 7.1 核心设计：推理端完全不感知 EP

推理引擎（SGLang/vLLM）**始终接收完整的 HuggingFace 格式权重**，不知道训练侧使用了 EP。

```
训练侧 (EP=8, 每 rank 32 experts)    推理侧 (无 EP)

  Rank 0: Expert 0-31 ─┐
  Rank 1: Expert 32-63 ─┤
  Rank 2: Expert 64-95 ─┤  AllGather
  ...                   ─┤  → 格式转换
  Rank 7: Expert 224-255─┘  → Broadcast ──→ SGLang/vLLM
                                         接收 256 个完整 Expert
                            HF 格式:
                            experts.0.gate_proj.weight [18432, 7168]
                            experts.0.up_proj.weight   [18432, 7168]
                            experts.0.down_proj.weight [7168, 18432]
                            ...
                            experts.255.down_proj.weight [7168, 18432]
```

### 7.2 Megatron 的显式 Gather-Convert-Broadcast 流程

**源码**: `megatron_engine.py:1133-1224`

```
Phase 1: AllGather (EP group 内)
  → 每个 EP rank 的本地 expert 参数被 gather 到所有 rank

Phase 2: 格式转换 (仅 PP head)
  → Megatron 格式 → HuggingFace 格式
  → 例: mlp.experts.linear_fc1.weight0 → model.layers.X.mlp.experts.0.gate_proj.weight
     + model.layers.X.mlp.experts.0.up_proj.weight (拆分)

Phase 3: Broadcast (PP head → 推理 Worker)
  → 通过 NCCL weight_update_group 广播到所有推理 Worker
```

### 7.3 Archon 的隐式 Gather 流程

**源码**: `archon_weight_sync.py:109-171`

```python
for name, param in engine._get_model_name_parameters():
    tensor = _get_full_tensor(param)   # DTensor.full_tensor()
    # 对于 EP 分片的参数: full_tensor() 隐式执行 EP AllGather
    # 对于 TP 分片的参数: full_tensor() 隐式执行 TP AllGather
    # 对于 FSDP 分片的参数: full_tensor() 隐式执行 FSDP AllGather

    # 转换为 HF 格式（拆分 3D expert 张量为逐 expert 的 2D 张量）
    hf_pairs = engine.state_dict_adapter.convert_single_to_hf(name, tensor)
    # 例: moe.experts.w1 [64, 18432, 7168]
    # → experts.0.gate_proj.weight [18432, 7168]
    # → experts.1.gate_proj.weight [18432, 7168]
    # → ... (通过 torch.unbind 拆分)
```

### 7.4 推理侧接收协议

**SGLang** (`sglang_remote.py:155`):

```python
payload = {
    "names": [pspec.name for pspec in param_specs],   # HF 格式名称
    "dtypes": [pspec.dtype for pspec in param_specs],
    "shapes": [pspec.shape for pspec in param_specs],  # 完整 shape，无 EP 分片
    "group_name": meta.nccl_group_name,
    "abort_all_requests": True,
}
# → HTTP POST /update_weights_from_distributed
# → SGLang 内部执行 NCCL broadcast recv
# → 直接 load_weights 到模型
```

**vLLM** (`vllm_worker_extension.py:119-150`):

```python
for name, dtype, shape in zip(names, dtypes, shapes):
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.broadcast(tensor, src=0, group=group)          # 接收完整参数
    self.model_runner.model.load_weights([(name, tensor)])  # 直接加载
```

**关键**: 推理引擎的 `load_weights` 接口只接受标准 HuggingFace 名称和完整张量，
完全不涉及 EP 分片。训练侧的 AllGather + 格式转换保证了这一点。

### 7.5 SGLang 推理侧的可选 EP

`SGLangConfig.enable_ep_moe`（`cli_args.py:1385`）是 SGLang 服务器自己的内部 EP
配置，用于推理时的 expert 并行。这与训练侧的 EP 完全正交——训练侧总是发送 gathered 的
完整权重，SGLang 在 `load_weights` 后根据自己的 EP 配置重新分片。

---

## 8. 设计总结

### Archon 如何组合 FSDP2 + TP + SP + PP + EP

```
                         ArchonParallelDims
                              │
                              │ build_mesh()
                              ▼
                    ┌─── DeviceMesh ───────────────────────┐
                    │  [pp, dp_shard_mod_ep,               │
                    │   dp_shard_in_ep, cp, tp]            │
                    └──────────┬──────────────────────────┘
                               │
            ┌──────────────────┼──────────────────────┐
            ▼                  ▼                      ▼
    parallelize_fn      pipeline_llm           build_pipeline_schedule
    (per stage)         (split model)          (1F1B/ZeroBubble)
            │
  ┌─────────┼──────────┬──────────────┬──────────┐
  ▼         ▼          ▼              ▼          ▼
apply_    apply_     apply_       apply_     apply_
non_moe   moe_ep    cp           ac          fsdp
_tp       _tp       (Ulysses)    (checkpoint) (fully_shard)
  │         │          │                       │
  │    ┌────┴────┐     │              ┌────────┴────────┐
  │    │ EP策略   │     │              │  Dense: dp_shard_cp
  │    │ 选择矩阵 │     │              │  Expert: dp_mod_ep
  │    └─────────┘     │              └─────────────────┘
  │                    │
  ▼                    ▼
DTensor              All-to-All
Shard/Replicate      head scatter / seq gather
```

### EP AllGather 同步开销的关键结论

| 维度 | Megatron | Archon |
|------|----------|--------|
| AllGather 方式 | 显式 `dist.all_gather` | 隐式 `DTensor.full_tensor()` |
| 缓冲区控制 | `weight_chunked_mem_mb × ep_size` 限制 | 由 DTensor 运行时管理 |
| 非 PP head rank | 参与 AllGather 但丢弃结果 | 同上 |
| 通信量 | 随 `num_experts × (1-1/ep_size)` 线性增长 | 同上 |
| 峰值显存 | ~`weight_chunked_mem_mb` (可配置) | 取决于 DTensor 实现 |

### 推理端接收的保证

> 推理引擎**完全不感知 EP**。训练侧在权重同步时执行 AllGather 将所有 EP rank 的
> expert 参数聚合到 PP head，转换为 HuggingFace 格式后通过 NCCL broadcast 发送到
> 推理 Worker。推理 Worker 通过标准的 `load_weights` 接口加载完整参数。
> 这种设计使得**训练侧的并行策略变更不影响推理代码**。
