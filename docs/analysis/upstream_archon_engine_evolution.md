# Archon 训练引擎演进分析

> **分析范围**: upstream/main 最近 250 个 commits（2026-01-25 ~ 2026-04-27） **Archon 相关
> commits**: 27 个（占总量 10.8%），净增 16,898 行代码（占总净增 20.7%） **时间跨度**: 65 天（2026-01-26 ~
> 2026-03-31），PR #849 ~ #1118

______________________________________________________________________

## 目录

- [一、总览与统计](#%E4%B8%80%E6%80%BB%E8%A7%88%E4%B8%8E%E7%BB%9F%E8%AE%A1)
- [二、DCP 检查点系统](#%E4%BA%8Cdcp-%E6%A3%80%E6%9F%A5%E7%82%B9%E7%B3%BB%E7%BB%9F)
- [三、流水线并行（Pipeline Parallelism）](#%E4%B8%89%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%B9%B6%E8%A1%8Cpipeline-parallelism)
- [四、MoE 混合专家支持](#%E5%9B%9Bmoe-%E6%B7%B7%E5%90%88%E4%B8%93%E5%AE%B6%E6%94%AF%E6%8C%81)
- [五、FP8 低精度训练](#%E4%BA%94fp8-%E4%BD%8E%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83)
- [六、内存优化策略](#%E5%85%AD%E5%86%85%E5%AD%98%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5)
- [七、模型架构扩展](#%E4%B8%83%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E6%89%A9%E5%B1%95)
- [八、代码质量与工程重构](#%E5%85%AB%E4%BB%A3%E7%A0%81%E8%B4%A8%E9%87%8F%E4%B8%8E%E5%B7%A5%E7%A8%8B%E9%87%8D%E6%9E%84)
- [九、树训练与确定性模式](#%E4%B9%9D%E6%A0%91%E8%AE%AD%E7%BB%83%E4%B8%8E%E7%A1%AE%E5%AE%9A%E6%80%A7%E6%A8%A1%E5%BC%8F)
- [十、架构评审与技术债务](#%E5%8D%81%E6%9E%B6%E6%9E%84%E8%AF%84%E5%AE%A1%E4%B8%8E%E6%8A%80%E6%9C%AF%E5%80%BA%E5%8A%A1)
- [附录 A: 完整 Commit 清单](#%E9%99%84%E5%BD%95-a-%E5%AE%8C%E6%95%B4-commit-%E6%B8%85%E5%8D%95)
- [附录 B: 当前文件结构](#%E9%99%84%E5%BD%95-b-%E5%BD%93%E5%89%8D%E6%96%87%E4%BB%B6%E7%BB%93%E6%9E%84)

______________________________________________________________________

## 一、总览与统计

### 1.1 开发阶段划分

Archon 引擎在 65 天内经历了 4 个清晰的开发阶段:

| 阶段         | 时间范围        | Commits | 净增行数 | 核心主题                             |
| ------------ | --------------- | ------- | -------- | ------------------------------------ |
| 基础建设     | Jan 26 - Jan 31 | 3       | +6,929   | DCP 检查点、Meta 设备初始化、PP 基础 |
| PP 扩展      | Feb 3 - Feb 10  | 11      | -1,634   | PP 调度策略、Runner 抽象、测试整合   |
| 稳定与增强   | Feb 15 - Mar 2  | 6       | +3,120   | 异步检查点、确定性模式、内存优化     |
| MoE/FP8 扩展 | Mar 8 - Mar 31  | 7       | +6,483   | Qwen3.5、FP8 训练、Router 修复       |

### 1.2 Commit 类型分布

| 类型     | 数量 | 占比  | 说明       |
| -------- | ---- | ----- | ---------- |
| feat     | 15   | 55.6% | 新功能开发 |
| fix      | 7    | 25.9% | Bug 修复   |
| refactor | 3    | 11.1% | 代码重构   |
| docs     | 1    | 3.7%  | 文档       |
| revert   | 1    | 3.7%  | 回滚       |

### 1.3 贡献者分析

| 贡献者       | Commits | 占比  | 主要贡献                     |
| ------------ | ------- | ----- | ---------------------------- |
| Wentai Zhang | 24      | 88.9% | 全栈架构师，涵盖所有功能领域 |
| Wei Fu       | 1       | 3.7%  | Data proxy 修复              |
| nuzant       | 1       | 3.7%  | 树训练支持                   |
| fishcrap     | 1       | 3.7%  | Router gate 修复             |

> **风险提示**: 88.9% 的代码由单一贡献者完成，bus factor 为 1。

### 1.4 开发节奏分析

**月度分布**:

| 月份    | 全部 Commits | Archon | Archon 占比 |
| ------- | ------------ | ------ | ----------- |
| 2026-01 | 17           | 3      | 17.6%       |
| 2026-02 | 60           | 15     | 25.0%       |
| 2026-03 | 88           | 9      | 10.2%       |

**峰值周**: 2026-02-03 ~ 02-09（W06），8 个 Archon commits，为整个分析周期中开发密度最高的一周。

**工作时间模式**: 51.8% 的 commits 集中在周一和周二，51.9% 在下午（13:00-17:00 UTC+8），表明"上午开发 → 下午集成"的工作节奏。

______________________________________________________________________

## 二、DCP 检查点系统

### 2.1 概述

DCP（Distributed Checkpoint）系统是 Archon 引擎的第一个重大功能，替代了手动 safetensors 方案，解决了大规模 MoE
模型在检查点时的 OOM 问题。

### 2.2 相关 Commits

| 日期   | Commit     | PR   | 主题                                                      | 影响        |
| ------ | ---------- | ---- | --------------------------------------------------------- | ----------- |
| Jan 26 | `38238e3d` | #849 | DCP-based HF checkpoint save/load with MoE expert support | +2,540/-345 |
| Jan 27 | `a05015dd` | #860 | Meta device + DCP for memory-efficient model init         | +210/-116   |
| Feb 15 | `cc2bec74` | #926 | Async checkpoint saving for ArchonEngine                  | +1,164/-114 |

### 2.3 技术要点

#### 2.3.1 DCP 替代 safetensors（PR #849）

**问题**: 原有方案在 `save` 时需要将完整状态字典 gather 到 rank 0，对大规模 MoE 模型造成 OOM。

**方案**: 采用 PyTorch 原生 DCP 基础设施:

- 每个 rank 只读写自己的分片，避免全量聚合
- 新增 `MoEStateDictAdapter` 基类，处理 DTensor 感知的专家权重转换（3D grouped → 2D individual）
- 支持 EP（Expert Parallel）和 ETP（Expert-Tensor Parallel）两种放置策略

**关键文件**:

- `areal/experimental/engine/archon_checkpoint.py`（新建，515 行）
- `areal/experimental/models/archon/moe_weight_converter.py`（新建）

#### 2.3.2 Meta 设备初始化（PR #860）

**优化效果**: 峰值内存从 `N × model_size` 降低到 `~1/N × model_size`（N 为 GPU 数量）。

**实现流程**:

1. 在 `meta` 设备上创建模型结构（无实际内存分配）
1. 应用 FSDP/TP/EP 并行化
1. 调用 `to_empty()` 物化张量
1. 通过 DCP 加载权重（每个 rank 只读自己的分片）

```python
# archon_engine.py 中的初始化流程
model = ModelSpec.model_class(args, device="meta")  # Step 1: meta device
apply_parallelism(model)                              # Step 2: parallelism
model.to_empty(device=device)                         # Step 3: materialize
dcp.load(state_dict, checkpoint_id=path)              # Step 4: load shards
```

#### 2.3.3 异步检查点（PR #926）

**设计**: `AsyncCheckpointManager` 支持 SYNC/ASYNC/AUTO 三种模式:

- **SYNC**: 传统阻塞式保存
- **ASYNC**: GPU→pinned-CPU 暂存（短暂阻塞），DCP 上传和 safetensors 合并在后台线程执行
- **AUTO**: 根据检查点大小自动选择

**架构细节**:

- 后台 Gloo Process Group 用于 barrier 同步
- Safetensors `index.json` 支持多文件 HF 检查点
- 集成到 RL 和 SFT 训练循环

______________________________________________________________________

## 三、流水线并行（Pipeline Parallelism）

### 3.1 概述

PP 支持是 Archon 引擎最大的功能模块，从基础的 Schedule1F1B 演进到 4 种高级调度策略，是开发密度最高的模块（11 commits，1
周内完成核心开发）。

### 3.2 相关 Commits

| 日期   | Commit     | PR   | 主题                              | 影响        |
| ------ | ---------- | ---- | --------------------------------- | ----------- |
| Jan 31 | `945bdd52` | #864 | PP 基础支持（Schedule1F1B）       | +5,026/-386 |
| Feb 3  | `b0da4a36` | #877 | PP > 1 + XCCL 权重同步            | +82/-33     |
| Feb 3  | `809c3982` | #882 | 提取 Runner 和 WeightSync 模块    | +789/-588   |
| Feb 4  | `3ae000a9` | #890 | 跳过 output merge，内存减半       | +17/-0      |
| Feb 5  | `07a80de3` | #895 | Interleaved1F1B 调度              | +419/-143   |
| Feb 10 | `ab411da7` | #916 | ZBVZeroBubble 调度                | +752/-399   |
| Feb 26 | `4da391f8` | #936 | InterleavedZeroBubble (ZB1P) 调度 | +71/-51     |
| Mar 2  | `a7f3735e` | #951 | PP 内存处理改进                   | +302/-28    |

### 3.3 技术要点

#### 3.3.1 基础 PP 架构（PR #864）

**设备网格扩展**: 将 4D mesh（dp × cp × tp）扩展为 5D（pp × dp × cp × tp）。

**核心组件**:

- `PipelineStage` + `Schedule1F1B`（PyTorch 原生）
- 均衡的层分配策略（`pp_layers_per_stage` 配置）
- DCP 和 HF 检查点的 PP 分片适配

**影响范围**: 39 个文件变更，是 Archon 最大的单次提交（+5,026 行）。

#### 3.3.2 Runner 抽象（PR #882）

将前向/反向执行逻辑从 `ArchonEngine` 中提取为独立模块:

```
ForwardBackwardRunner (ABC)
├── SequentialRunner     # 单设备/DP-only 模式
└── PipelinedRunner      # PP 模式，管理 PipelineSchedule
```

- `SequentialRunner`: 直接遍历 microbatch 执行前向/反向
- `PipelinedRunner`: 委托给 PyTorch 的 PipelineSchedule，处理 loss_fn 闭包和输出块管理

**设计优点**: 工厂函数 `create_runner()` 根据配置自动选择，引擎层无需 if/else 分支。

#### 3.3.3 PP 调度策略演进

| 调度策略                  | PR   | 特点                                    | 适用场景                 |
| ------------------------- | ---- | --------------------------------------- | ------------------------ |
| **Schedule1F1B**          | #864 | 基础 1F1B，每 rank 1 个 stage           | 通用 PP                  |
| **Interleaved1F1B**       | #895 | 每 rank 多个虚拟 stage                  | 更高吞吐量               |
| **ZBVZeroBubble**         | #916 | V 型 stage 分配，分离 I-grad/W-grad     | 近零气泡，2× stages/rank |
| **InterleavedZeroBubble** | #936 | ZB1P，循环型 stage 分配 + 分离 backward | 平衡气泡与复杂度         |

**Stage 分配逻辑**（`pipeline_parallel.py:298-329`）:

- **循环型**（1F1B、Interleaved1F1B、IZB）: rank `r` 获得 stages `[r, r+pp, r+2*pp, ...]`
- **V 型**（ZBVZeroBubble）: rank `r` 获得 stages `[r, num_stages-1-r]`

**调度选择机制**: 通过 `get_schedule_class(pp_schedule)` 动态解析，结合
`issubclass(schedule_class, PipelineScheduleMulti)` 判断单/多 stage 模式，完全避免硬编码的条件分支。

#### 3.3.4 PP 内存优化

**Output Merge 跳过**（PR #890）: PyTorch 的 `Schedule1F1B._merge_outputs()` 会 `torch.cat` 所有
microbatch 输出，导致内存翻倍。通过 monkey-patch `_merge_outputs` 返回 `None` 来跳过这一不必要的分配。

**NullOutputChunks**（`archon_runner.py:245-253`）: 训练时用 `_NullOutputChunks` 替代输出列表，使
logits 在 backward 后立即释放。

**Reshard 策略**（PR #951）: 新增 `reshard_after_forward_policy` 配置，控制 FSDP
在前向传播后何时重新分片参数，在内存和计算之间取得平衡。

______________________________________________________________________

## 四、MoE 混合专家支持

### 4.1 概述

MoE 支持贯穿多个阶段，从检查点基础设施到模型实现，再到数值稳定性修复。

### 4.2 相关 Commits

| 日期   | Commit     | PR    | 主题                            | 影响        |
| ------ | ---------- | ----- | ------------------------------- | ----------- |
| Jan 26 | `38238e3d` | #849  | DCP 检查点 + MoE 专家支持       | +2,540/-345 |
| Feb 27 | `1fd9f949` | #940  | score_before_experts 默认值修复 | +1,070/-2   |
| Mar 8  | `4f5a2944` | #1009 | FP32 Router Gate GEMM 配置      | +478/-120   |
| Mar 12 | `1927decc` | #1012 | Qwen3.5 dense 和 MoE 支持       | +4,978/-226 |
| Mar 16 | `978532ea` | #1029 | RouterGateLinear 包装器         | +86/-7      |

### 4.3 技术要点

#### 4.3.1 Router Gate 数值稳定性（PR #1009）

**问题**: 大规模专家数量下，BF16 精度的 Router Gate GEMM 会导致数值不稳定。

**方案**: 采用 Megatron-Core 风格的自定义 `torch.autograd.Function`（`RouterGatingLinearFunction`）:

- 前向：FP32 精度计算 Gate GEMM
- 反向：FP32 精度梯度计算
- 激活保存为 BF16（内存效率），计算使用 FP32（数值稳定）
- `@torch.amp.custom_fwd/bwd` 防止 AMP 干扰

#### 4.3.2 DTensor 兼容性修复（PR #1029）

**问题**: 直接使用 `nn.Linear` 作为 Gate 时，`ReplicateParallel` 的 DTensor hook 不会在
`module.__call__()` 上触发。

**方案**: 新增 `RouterGateLinear(nn.Linear)` 包装器，确保 DTensor hook 通过标准 Module
调用路径触发。`TokenChoiceTopKRouter` 的前向简化为 `scores = self.gate(x)`。

#### 4.3.3 HF 权重兼容性修复（PR #940）

**问题**: HuggingFace 模型（Mixtral、Qwen3-MoE、JetMoE、GraniteMoe）均在专家计算**之后**应用 Router 分数，但
Archon 默认设置 `score_before_experts=True`，导致加载 HF 检查点时行为不一致。

**修复**: 将 `MoEArgs.score_before_experts` 默认值从 `True` 改为 `False`，并增加 1,070 行 HF parity
测试。

#### 4.3.4 MoE 模块组合架构

MoE 模块采用组合模式而非继承:

```
MoE
├── TokenChoiceTopKRouter    # 路由器（Top-K 选择 + 可选 FP32 Gate）
├── GroupedExperts            # 分组专家计算（grouped_mm / for-loop）
└── TokenReorderer           # Token 重排（支持 EP/ETP 分片）
```

`TokenReorderer` 的独立抽象是为了支持 `etp=1` 时的 Sequence Parallel 分片，将本该耦合的 EP/TP 逻辑解耦。

______________________________________________________________________

## 五、FP8 低精度训练

### 5.1 概述

FP8 训练支持是 Archon 引擎最后一个重大功能（2026-03-31），采用 torchao 的 blockwise FP8 matmul，在保持 BF16
主权重的同时实现计算加速。

### 5.2 相关 Commits

| 日期   | Commit     | PR    | 主题                              | 影响       |
| ------ | ---------- | ----- | --------------------------------- | ---------- |
| Mar 31 | `f6331e09` | #1087 | FP8 blockwise 训练支持            | +2,958/-85 |
| Mar 31 | `0ee85625` | #1118 | 强化 FP8 在 TP/MoE 场景下的鲁棒性 | +94/-28    |

### 5.3 技术要点

#### 5.3.1 设计模式: 构建后变换（Construct-then-Transform）

FP8 实现采用两阶段设计:

**On-the-fly FP8 矩阵乘法**（`fp8.py`）:

- `enable_fp8_linear` 修补合格的 `nn.Linear` 模块的 forward 方法
- 使用 torchao 的 `fp8_blockwise_mm`（128×128 分块，FP8 e4m3fn 格式）
- 主权重保持 BF16，运行时按需量化
- 排除数值敏感层: `{"output", "router", "score"}`
- 使用 `types.MethodType` 进行修补，确保 `copy.deepcopy`（PP stage 分裂）的安全性

**FP8 检查点加载**（`fp8_checkpoint.py`）:

1. **检测**: 通过 safetensors index 中的 `_scale_inv` 键判断
1. **准备**: 修改占位符 dtype 为 `float8_e4m3fn`，插入 scale 占位符
1. **DCP 加载**: 使用修改后的状态字典
1. **反量化**: 加载后将 FP8 反量化回 BF16，支持 DTensor 感知的本地分片反量化

#### 5.3.2 MoE 专家 FP8 支持

`enable_fp8_experts` 单独修补 `GroupedExperts` 模块，采用逐专家 for-loop + FP8 matmul 的降级方案（因
`grouped_mm` 尚不支持 FP8）。

#### 5.3.3 后并行化验证

`validate_fp8_shard_alignment`（`fp8.py:166-231`）在并行化应用后检查 TP 分片是否破坏了 128-block 对齐要求，是一个
fail-fast 安全网。

#### 5.3.4 当前限制

| 限制                | 说明                                    | 状态              |
| ------------------- | --------------------------------------- | ----------------- |
| Shard(1) FP8 反量化 | 列分片（TP/ETP）的 FP8 检查点加载不支持 | Phase 2 TODO      |
| grouped_mm FP8      | `torch._grouped_mm` 不支持 FP8          | 等待 PyTorch 升级 |
| 编译兼容性          | FP8 激活时自动禁用 `torch.compile`      | 设计决策          |
| Triton 独占         | cuBLAS 不支持混合 per-operand scaling   | torchao 限制      |

______________________________________________________________________

## 六、内存优化策略

### 6.1 多层次优化矩阵

Archon 引擎实现了从初始化到训练循环的多层次内存优化:

| 优化策略             | 阶段   | PR   | 效果                          |
| -------------------- | ------ | ---- | ----------------------------- |
| Meta 设备初始化      | 初始化 | #860 | 峰值内存 N×model → ~1/N×model |
| PP Output Merge 跳过 | 训练   | #890 | 评估时内存减半                |
| NullOutputChunks     | 训练   | #890 | 训练时即时释放 logits         |
| Reshard 策略配置     | 训练   | #951 | FSDP 内存/计算权衡可调        |
| MoE Donated Buffer   | 训练   | #951 | 对齐填充的内存重用            |
| CPU Offload          | 训练   | 存量 | `torch_memory_saver` 集成     |
| 异步检查点           | 检查点 | #926 | GPU→CPU staging 最小化阻塞    |

### 6.2 Output Merge 跳过技术细节

```python
# archon_runner.py:162-175
# PyTorch Schedule1F1B._merge_outputs() 会 torch.cat 所有 microbatch 输出，
# 导致内存翻倍。Monkey-patch 返回 None 跳过不必要的分配。
def _patch_skip_output_merge(self):
    def _skip_merge(self):
        return None
    self.schedule._merge_outputs = types.MethodType(_skip_merge, self.schedule)
```

> **技术债**: 标记为 `TODO(pytorch-upgrade)`，PyTorch 2.10+ 的 `return_outputs=False` 将提供原生替代方案。

______________________________________________________________________

## 七、模型架构扩展

### 7.1 概述

Archon 引擎通过 `ModelSpec` 注册表模式支持多个模型架构，在分析期间新增了 Qwen3.5。

### 7.2 相关 Commits

| 日期   | Commit     | PR    | 主题                                 | 影响        |
| ------ | ---------- | ----- | ------------------------------------ | ----------- |
| Mar 12 | `1927decc` | #1012 | Qwen3.5 dense 和 MoE 支持（DP-only） | +4,978/-226 |

### 7.3 ModelSpec 注册表模式

```python
# model_spec.py 中的注册表设计
@dataclass
class ModelSpec:
    model_class: type[BaseArchonModel]
    parallelize_fn: ParallelizeFn      # Protocol type
    pipelining_fn: PipeliningFn        # Protocol type
    state_dict_adapter: StateDictAdapter

# 各模型自注册
register_model_spec("qwen2", ModelSpec(...))
register_model_spec("qwen3", ModelSpec(...))
register_model_spec("qwen3_5", ModelSpec(...))
```

**扩展性**: 添加新模型仅需实现 `ModelSpec` 并在 `__init__.py` 中导入。引擎层零修改。

### 7.4 Qwen3.5: 混合注意力架构

Qwen3.5 是一个**架构全新的混合模型**，交替使用两种注意力层:

- **GatedAttention**（`full_attention`）: 带 sigmoid 输出门的全注意力，Q 投影输出 2x 宽度
- **GatedDeltaNet**（`linear_attention`）: 基于 chunk 的 delta rule 线性注意力，依赖 `fla` 库

**实现规模**: 22 个文件变更，+4,978 行，是第二大单次提交。包含:

- `Qwen3_5Model` 模型实现（679 行）
- 两种 RMSNorm 变体（`(1+w)*norm(x)` 和 `w*norm(x)*silu(gate)`）
- 部分 RoPE（`partial_rotary_factor=0.25`，仅 25% head_dim 参与旋转）
- HF 状态字典适配器（支持 per-expert 2D 和 fused 3D 格式）
- MoE `shared_expert_gate` 支持
- FSDP2 并行化（**仅 DP，不支持 TP/CP/EP**）
- 29 个单元测试 + 4 个 HF parity E2E 测试

______________________________________________________________________

## 八、代码质量与工程重构

### 8.1 概述

3 个专门的重构 commit 显示了团队对代码质量的持续关注。

### 8.2 相关 Commits

| 日期  | Commit     | PR   | 主题                           | 影响        |
| ----- | ---------- | ---- | ------------------------------ | ----------- |
| Feb 3 | `809c3982` | #882 | 提取 Runner 和 WeightSync 模块 | +789/-588   |
| Feb 4 | `55ff540e` | #888 | 整合简化测试套件               | +137/-4,107 |
| Mar 2 | `c26bea9b` | #954 | 提取工具函数，简化引擎代码     | +486/-340   |

### 8.3 技术要点

#### 8.3.1 模块提取（PR #882）

从 `ArchonEngine` 提取出两个独立模块:

- `archon_runner.py`（332 行）: `ForwardBackwardRunner` 抽象
- `archon_weight_sync.py`（245 行）: `WeightSyncState` 和权重同步函数

#### 8.3.2 测试套件整合（PR #888）

**规模**: 删除 ~4,000 行测试代码（48%），同时通过更全面的 E2E 测试保持覆盖率。

**具体操作**:

- 删除被 E2E 测试覆盖的冗余单元测试（MoE 组件、Attention、Router 等 7 个文件）
- 合并分散的分布式测试文件（`test_distributed_etp.py` → `test_distributed_ep.py`）
- `test_state_dict_adapter.py` 从 1,134 行精简到 223 行

#### 8.3.3 工具函数提取（PR #954）

将以下逻辑从引擎中提取到 `archon_utils.py`（364 行）:

- 优化器/调度器创建
- 激活检查点配置
- Zero-bubble 验证
- 确定性模式设置
- `pad_to_maximum` 验证
- `DistributedLock` 上下文管理器支持

### 8.4 当前引擎模块结构

| 模块                    | 行数      | 职责                           |
| ----------------------- | --------- | ------------------------------ |
| `archon_engine.py`      | 1,549     | 编排层: 初始化、训练、评估     |
| `archon_checkpoint.py`  | 515       | 检查点 I/O: DCP + 异步保存     |
| `archon_utils.py`       | 364       | 工具函数: 优化器、验证         |
| `archon_runner.py`      | 332       | 执行策略: Sequential/Pipelined |
| `archon_weight_sync.py` | 245       | 权重同步: XCCL + 分桶广播      |
| **合计**                | **3,005** |                                |

______________________________________________________________________

## 九、树训练与确定性模式

### 9.1 相关 Commits

| 日期   | Commit     | PR   | 主题                  | 影响     |
| ------ | ---------- | ---- | --------------------- | -------- |
| Feb 10 | `e03f32f3` | #912 | Archon 引擎树训练支持 | +481/-58 |
| Feb 28 | `f5cb33c4` | #943 | 确定性训练模式        | +588/-26 |

### 9.2 树训练（PR #912）

遵循 FSDP 引擎的已有模式，为 Archon 增加树训练能力:

- `ArchonTrainContext` 增加 `TrieNode` 字段
- CP 验证和 `pad_to_maximum` 强制
- `_prepare_mb_list()` 中增加树训练路径（使用 `build_packed_tree_batch()`）
- Qwen2/Qwen3 模型层增加 `block_mask` 和 `triton_attn_data` 参数
- 新增 `areal/models/tree_attn/module_archon.py` 适配层

### 9.3 确定性训练（PR #943）

新增 `use_deterministic_algorithms` 配置，启用:

- PyTorch 确定性算法
- cuBLAS 确定性工作空间
- NCCL Ring reduction（替代 Tree reduction）
- `torch.compile` 确定性模式
- Activation Checkpoint RNG 状态保存

**测试**: 包含 CUDA 回归测试，覆盖 `grouped_mm`、compile、AC、logprobs 场景。

______________________________________________________________________

## 十、架构评审与技术债务

### 10.1 架构优势

| 维度     | 评级     | 说明                                                  |
| -------- | -------- | ----------------------------------------------------- |
| 设计模式 | **强**   | Registry 模式、Strategy 模式（Runner）、Protocol 类型 |
| 可扩展性 | **强**   | 6D 并行、meta 设备初始化、内存高效 PP                 |
| 技术选型 | **适当** | PyTorch 原生 DCP/DTensor/PipelineSchedule             |
| 集成模式 | **良好** | 清晰委托，少量 monkey-patching 技术债                 |
| 性能架构 | **强**   | 多层内存优化、FP8 支持、对齐批处理                    |
| 技术债务 | **可控** | 有文档的 TODO，清晰的升级路径                         |

### 10.2 已追踪的技术债务

| 位置                        | 描述                                    | 升级路径                             |
| --------------------------- | --------------------------------------- | ------------------------------------ |
| `archon_runner.py:165`      | `_patch_skip_output_merge` monkey-patch | PyTorch 2.10+ `return_outputs=False` |
| `fp8_checkpoint.py:170-173` | Shard(1) FP8 反量化不支持               | Phase 2 实现                         |
| `grouped_experts.py:98-101` | `grouped_mm` 硬编码 BF16                | 等待 PyTorch 支持                    |
| `archon_checkpoint.py:36`   | DCP barrier bug 的 workaround           | 上游修复                             |

### 10.3 未追踪的架构风险

**风险 1: 初始化顺序脆弱性**

`initialize()` 方法有严格的执行顺序依赖（`_create_device_model` → FP8 patching →
`prepare_training_config` → `_setup_parallelism` → `validate_fp8_shard_alignment` →
`_materialize_and_load_weights` → `_create_optimizer` →
`create_runner`），但没有形式化的状态机或阶段间断言。

**风险 2: 优化器检查点不可移植**

`save_optimizer_state` 使用 rank 分片的 `torch.save`（每个 rank 一个 `.pt` 文件），文件命名不支持跨 world size
重分片。

**风险 3: 并行配置组合爆炸**

6 个并行维度 × 4+ PP 调度变体，组合测试负担巨大。

**风险 4: PP Stage 分裂的内存开销**

`copy.deepcopy(whole_model)` 为每个 pipeline stage 深拷贝整个模型。Meta 设备缓解了实际张量分配问题，但任何非 meta
的张量或 buffer 都会被拷贝。

**风险 5: 权重同步转换开销**

`archon_weight_sync.py` 的分桶广播在每次同步时通过 `state_dict_adapter.convert_single_to_hf` 重建 HF
格式张量，转换开销随模型规模线性增长。

______________________________________________________________________

## 附录 A: 完整 Commit 清单

| #   | 日期   | Commit     | PR    | 类型     | 主题                                                      | +/-         |
| --- | ------ | ---------- | ----- | -------- | --------------------------------------------------------- | ----------- |
| 1   | Jan 26 | `38238e3d` | #849  | feat     | DCP-based HF checkpoint save/load with MoE expert support | +2,540/-345 |
| 2   | Jan 27 | `a05015dd` | #860  | feat     | Meta device + DCP for memory-efficient model init         | +210/-116   |
| 3   | Jan 31 | `945bdd52` | #864  | feat     | Pipeline parallelism (PP) support                         | +5,026/-386 |
| 4   | Feb 3  | `b0da4a36` | #877  | feat     | PP > 1 for RL training with XCCL weight sync              | +82/-33     |
| 5   | Feb 3  | `809c3982` | #882  | refactor | Extract runner and weight sync modules                    | +789/-588   |
| 6   | Feb 4  | `6cd95fee` | #886  | fix      | Enable torch.compile for attention_norm/ffn_norm          | +0/-8       |
| 7   | Feb 4  | `434b6e7e` | #887  | revert   | Revert torch.compile for norms                            | +8/-0       |
| 8   | Feb 4  | `55ff540e` | #888  | refactor | Consolidate and simplify test suite                       | +137/-4,107 |
| 9   | Feb 4  | `3ae000a9` | #890  | fix      | Skip output merge in PP schedule (halve memory)           | +17/-0      |
| 10  | Feb 5  | `07a80de3` | #895  | feat     | Interleaved1F1B pipeline schedule                         | +419/-143   |
| 11  | Feb 6  | `64fdccdc` | #900  | docs     | Archon engine tutorial and AI-assisted dev guide          | +469/-1     |
| 12  | Feb 9  | `5f333d26` | #914  | feat     | /add-archon-model skill for new model support             | +551/-5     |
| 13  | Feb 10 | `e03f32f3` | #912  | feat     | Tree training support for Archon engine                   | +481/-58    |
| 14  | Feb 10 | `ab411da7` | #916  | feat     | ZBVZeroBubble pipeline schedule                           | +752/-399   |
| 15  | Feb 15 | `cc2bec74` | #926  | feat     | Async checkpoint saving                                   | +1,164/-114 |
| 16  | Feb 26 | `4da391f8` | #936  | feat     | InterleavedZeroBubble (ZB1P) schedule                     | +71/-51     |
| 17  | Feb 27 | `1fd9f949` | #940  | fix      | Default score_before_experts to False                     | +1,070/-2   |
| 18  | Feb 28 | `f5cb33c4` | #943  | feat     | Deterministic training mode                               | +588/-26    |
| 19  | Mar 2  | `a7f3735e` | #951  | feat     | PP memory handling improvements                           | +302/-28    |
| 20  | Mar 2  | `c26bea9b` | #954  | refactor | Extract utility functions                                 | +486/-340   |
| 21  | Mar 8  | `4f5a2944` | #1009 | feat     | moe_router_dtype config (FP32 router gate)                | +478/-120   |
| 22  | Mar 12 | `1927decc` | #1012 | feat     | Qwen3.5 dense and MoE support                             | +4,978/-226 |
| 23  | Mar 16 | `978532ea` | #1029 | fix      | RouterGateLinear for DTensor hooks                        | +86/-7      |
| 24  | Mar 23 | `9639749e` | #1067 | fix      | Simplify RTensor serialization                            | +36/-102    |
| 25  | Mar 30 | `cbe35f5a` | #1105 | fix      | Add missing POST /data/batch endpoint                     | +409/-1     |
| 26  | Mar 31 | `f6331e09` | #1087 | feat     | FP8 blockwise training support                            | +2,958/-85  |
| 27  | Mar 31 | `0ee85625` | #1118 | fix      | Harden FP8 for TP and MoE                                 | +94/-28     |

## 附录 B: 当前文件结构

### 引擎模块（5 文件，3,005 行）

```
areal/experimental/engine/
├── archon_engine.py          # 1,549 行 - 主引擎编排
├── archon_checkpoint.py      #   515 行 - DCP + 异步检查点
├── archon_utils.py           #   364 行 - 工具函数
├── archon_runner.py          #   332 行 - 前向/反向 Runner
└── archon_weight_sync.py     #   245 行 - 权重同步
```

### 模型模块（45 文件）

```
areal/experimental/models/archon/
├── __init__.py, base.py, model_spec.py        # 核心注册
├── parallel_dims.py, pipeline_parallel.py     # 并行基础设施
├── activation_checkpoint.py, compile.py       # AC + 编译
├── expert_parallel.py, ulysses.py             # EP + CP
├── fp8.py, fp8_checkpoint.py                  # FP8 子系统
├── moe_weight_converter.py, utils.py          # 工具
├── attention/(sdpa.py, varlen.py)             # 注意力
├── moe/(args, grouped_experts, kernels,       # MoE 子系统
│        moe, router, token_reorderer, utils)
├── qwen2/(spec, parallelize, model, rope,     # Qwen2 模型族
│          state_dict_adapter)
├── qwen3/(spec, parallelize, model, rope,     # Qwen3 模型族
│          state_dict_adapter)
└── qwen3_5/(spec, parallelize, model, rope,   # Qwen3.5 混合注意力
             state_dict_adapter)
```

### 测试套件（55 文件）

```
tests/experimental/archon/
├── 19 个单元测试文件（conftest, HF parity, MoE, pipeline, etc.）
├── 5 个分布式测试（DP/TP/CP/EP/PP）
├── fp8/（13 个 FP8 专用测试）
└── torchrun/（13 个多 GPU 测试脚本）
```
