# Archon 训练引擎 v1.0.0 → v1.0.3 未覆盖更新深度分析

> **背景**: 现有分析文档覆盖了两个区间:
>
> - `archon_*.md` 系列深度解析: 基于 v1.0.0 代码的**静态架构分析**
> - `origin_main_source_code_analysis.md`: v1.0.3 → HEAD 的**增量 commit 分析**
>
> **本文覆盖的空白区间**: v1.0.0 (`99ce5342`, 2026-03-02) → v1.0.3 (`376ecbb8`, 2026-04-16) 其中包含
> **114 个 commits**，其中 7 个 Archon 专属 + **18 个并行策略相关**（跨全部三个引擎）， 引入了 3 项 Archon
> 重大新功能和**大量 5D 并行策略更新**。
>
> **量化影响**: Archon 专属 +9,039/-569，并行策略相关 +10,366/-4,020

______________________________________________________________________

## 目录

- [一、覆盖空白概述](#%E4%B8%80%E8%A6%86%E7%9B%96%E7%A9%BA%E7%99%BD%E6%A6%82%E8%BF%B0)
- [二、FP8 Blockwise 低精度训练（重大新增）](#%E4%BA%8Cfp8-blockwise-%E4%BD%8E%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83%E9%87%8D%E5%A4%A7%E6%96%B0%E5%A2%9E)
- [三、Qwen3.5 混合注意力模型（重大新增）](#%E4%B8%89qwen35-%E6%B7%B7%E5%90%88%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%A8%A1%E5%9E%8B%E9%87%8D%E5%A4%A7%E6%96%B0%E5%A2%9E)
- [四、MoE Router 数值稳定性与 DTensor 修复](#%E5%9B%9Bmoe-router-%E6%95%B0%E5%80%BC%E7%A8%B3%E5%AE%9A%E6%80%A7%E4%B8%8E-dtensor-%E4%BF%AE%E5%A4%8D)
- [五、Data Proxy RTensor 修复（次要）](#%E4%BA%94data-proxy-rtensor-%E4%BF%AE%E5%A4%8D%E6%AC%A1%E8%A6%81)
- [六、5D 并行策略更新（跨引擎）](#%E5%85%AD5d-%E5%B9%B6%E8%A1%8C%E7%AD%96%E7%95%A5%E6%9B%B4%E6%96%B0%E8%B7%A8%E5%BC%95%E6%93%8E)
- [七、代码质量评审发现](#%E4%B8%83%E4%BB%A3%E7%A0%81%E8%B4%A8%E9%87%8F%E8%AF%84%E5%AE%A1%E5%8F%91%E7%8E%B0)
- [八、量化影响分析](#%E5%85%AB%E9%87%8F%E5%8C%96%E5%BD%B1%E5%93%8D%E5%88%86%E6%9E%90)
- [九、与现有文档的关联](#%E4%B9%9D%E4%B8%8E%E7%8E%B0%E6%9C%89%E6%96%87%E6%A1%A3%E7%9A%84%E5%85%B3%E8%81%94)
- [附录: 完整 Commit 清单](#%E9%99%84%E5%BD%95-%E5%AE%8C%E6%95%B4-commit-%E6%B8%85%E5%8D%95)

______________________________________________________________________

## 一、覆盖空白概述

### 1.1 问题发现

对比 upstream/main 最近 250 个 commits 与现有分析文档:

| 文档                                        | 覆盖范围        | Archon 内容                 |
| ------------------------------------------- | --------------- | --------------------------- |
| `archon_areal_architecture_notes.md`        | v1.0.0 代码快照 | 引擎架构、并行化、编译      |
| `archon_pipeline_parallel_deep_dive.md`     | v1.0.0 代码快照 | PP 调度、Stage 分裂         |
| `archon_expert_parallel_deep_dive.md`       | v1.0.0 代码快照 | EP/ETP DTensor 集成         |
| `archon_compile_deep_dive.md`               | v1.0.0 代码快照 | torch.compile 配置          |
| `archon_activation_checkpoint_deep_dive.md` | v1.0.0 代码快照 | AC 策略（仅提及 FP8）       |
| `origin_main_source_code_analysis.md`       | v1.0.3 → HEAD   | DPO 引擎、Offload、Teardown |

**空白区间**: v1.0.0 → v1.0.3（2026-03-02 ~ 2026-04-16），7 个 Archon commits **未被任何文档覆盖**。

### 1.2 空白区间 Commits

| #   | 日期   | Commit     | PR    | 类型 | 主题                                       | +/-         |
| --- | ------ | ---------- | ----- | ---- | ------------------------------------------ | ----------- |
| 1   | Mar 8  | `4f5a2944` | #1009 | feat | MoE FP32 Router Gate GEMM 配置             | +478/-120   |
| 2   | Mar 12 | `1927decc` | #1012 | feat | Qwen3.5 dense + MoE 支持（DP-only）        | +4,978/-226 |
| 3   | Mar 16 | `978532ea` | #1029 | fix  | Router Gate nn.Module 包装（DTensor 兼容） | +86/-7      |
| 4   | Mar 23 | `9639749e` | #1067 | fix  | RTensor 序列化简化                         | +36/-102    |
| 5   | Mar 30 | `cbe35f5a` | #1105 | fix  | Data Proxy 缺失的 POST /data/batch 端点    | +409/-1     |
| 6   | Mar 31 | `f6331e09` | #1087 | feat | FP8 Blockwise 训练支持                     | +2,958/-85  |
| 7   | Mar 31 | `0ee85625` | #1118 | fix  | FP8 在 TP/MoE 场景下的鲁棒性强化           | +94/-28     |

### 1.3 功能影响矩阵

| 功能领域           | 生产代码     | 测试代码     | 测试:代码比 | 新文件 | 显著性       |
| ------------------ | ------------ | ------------ | ----------- | ------ | ------------ |
| FP8 Blockwise 训练 | 860 行       | 2,010 行     | **2.3:1**   | 17     | **重大新增** |
| Qwen3.5 模型       | 1,897 行     | 3,081 行     | **1.6:1**   | 13     | **重大新增** |
| MoE Router 改进    | 116 行       | 429 行       | **3.7:1**   | 1      | 重要修复     |
| Data Proxy 修复    | 98 行        | 347 行       | **3.5:1**   | 1      | 次要修复     |
| **合计**           | **2,971 行** | **5,867 行** | **2.0:1**   | **31** |              |

______________________________________________________________________

## 二、FP8 Blockwise 低精度训练（重大新增）

### 2.1 概述

**PR #1087** + **PR #1118**（2026-03-31），+3,052/-113，26+7 个文件。

这是 Archon 引擎的一项**全新训练能力**：通过 torchao 的 128×128 Blockwise FP8（e4m3fn）矩阵乘法实现计算加速，同时保持 BF16
主权重。**现有的所有分析文档中均未覆盖此功能**（`archon_activation_checkpoint_deep_dive.md` 仅在量化与 AC 交互的上下文中提及
FP8 一词，并非对 FP8 训练的分析）。

### 2.2 架构设计

#### 2.2.1 "构建后变换"模式

FP8 采用 monkey-patching 策略，在模型构建后、并行化前注入 FP8 计算路径:

```
[1] 在 meta 设备上构建标准 BF16 模型（已有流程）
[2] ★ FP8 前向方法修补（archon_engine.py:316-338）     ← 新增步骤
[3] 应用 FSDP/TP/EP 并行化（已有流程）
[4] ★ FP8 分片对齐验证（archon_engine.py:362-368）     ← 新增步骤
[5] 权重物化和加载（已有流程，增加 FP8 检查点支持）
```

#### 2.2.2 两个独立修补面

**Linear 层修补**（`fp8.py:enable_fp8_linear`）:

- 遍历模型中所有 `nn.Linear`，排除匹配 FQN 子串的模块（默认排除: `output`, `router`, `score`）
- 使用 `types.MethodType` 替换 `forward` 方法为 FP8 路径
- 运行时对输入和权重执行按需 FP8 量化，通过
  `torchao.prototype.blockwise_fp8_training.linear.fp8_blockwise_mm` 计算
- 使用 `types.MethodType` 而非子类化的原因: 保持 `nn.Linear` 类型不变，确保 FSDP `fully_shard()` 和 DTensor
  `parallelize_module()` 正常工作

**MoE 专家修补**（`fp8.py:enable_fp8_experts`）:

- 修补 `GroupedExperts` 模块的 `forward` 方法
- 因 `torch._grouped_mm` 不支持 FP8，降级为逐专家 for-loop + FP8 matmul
- 已知性能问题: `.tolist()` 在每次前向传播时触发 GPU→CPU 同步

#### 2.2.3 后并行化验证

`validate_fp8_shard_alignment`（`fp8.py:166-231`）在 TP/PP 分片后验证:

- 所有被修补的 `nn.Linear` 的本地权重维度仍为 128 的倍数
- 所有被修补的 `GroupedExperts` 的 3D 权重（w1/w2/w3）的逐专家切片仍为 128 对齐
- **防护价值**: 避免运行时 Triton/cuBLAS 因非对齐维度而崩溃

#### 2.2.4 FP8 检查点加载流水线

`fp8_checkpoint.py`（329 行）实现了完整的 FP8 检查点加载:

```
检测（_detect_fp8_checkpoint）→ 准备（_prepare_fp8_state_dict）→ DCP 加载 → 反量化（dequant_fp8_state_dict）
```

1. **检测**: 通过 safetensors index 中的 `*_scale_inv` 键判断是否为 FP8 检查点
1. **准备**: 将占位符 dtype 从 BF16 改为 `float8_e4m3fn`，插入 scale 占位符
1. **DCP 加载**: 使用修改后的状态字典
1. **反量化**: FP8 → BF16，支持 DTensor Shard(0) 感知的本地分片反量化
   - GPU 路径: Triton 核
   - CPU 回退: 纯 PyTorch 实现

### 2.3 配置接口

新增 `ArchonFP8Config` 数据类（嵌套在 `ArchonEngineConfig.fp8_config`）:

| 字段              | 类型        | 默认值                        | 说明                                 |
| ----------------- | ----------- | ----------------------------- | ------------------------------------ |
| `mode`            | str         | `"disabled"`                  | `"disabled"` / `"blockwise"`         |
| `exclude_modules` | list\[str\] | `["output","router","score"]` | 排除的 FQN 子串                      |
| `include_experts` | bool        | `False`                       | 是否对 MoE 专家启用 FP8              |
| `use_triton`      | bool        | `True`                        | 必须为 True（cuBLAS 不支持混合缩放） |

示例配置: `examples/math/gsm8k_sft_archon_fp8.yaml`（96 行）。

### 2.4 已知限制与技术债务

| 限制                | 严重性 | 说明                                            | 状态              |
| ------------------- | ------ | ----------------------------------------------- | ----------------- |
| Shard(1) FP8 反量化 | **高** | 列分片（TP/ETP）的 FP8 检查点加载不支持         | Phase 2 TODO      |
| grouped_mm 无 FP8   | **高** | MoE 专家降级为 for-loop，有 GPU-CPU 同步开销    | 等待 PyTorch 上游 |
| torch.compile 互斥  | **中** | FP8 启用时自动禁用编译                          | 设计决策          |
| Triton 独占         | **中** | cuBLAS 不支持混合 per-operand scaling           | torchao 限制      |
| 需要 SM90+          | **低** | 仅 Hopper GPU 支持 FP8 e4m3fn                   | 硬件要求          |
| 原型依赖            | **低** | 依赖 `torchao.prototype.blockwise_fp8_training` | 尚未稳定          |

### 2.5 测试覆盖

13 个专用 FP8 测试文件（1,843 行），覆盖:

- 前向正确性（余弦相似度 > 0.9，FP8 量化噪声下的合理阈值）
- 反向梯度正确性（方向 + 幅度）
- 训练收敛（loss 下降）
- 检查点检测/准备/反量化
- 分布式分片反量化
- MoE dispatch（空专家、不均匀分布、非 2 的幂专家数）
- Scale 张量布局兼容性

______________________________________________________________________

## 三、Qwen3.5 混合注意力模型（重大新增）

### 3.1 概述

**PR #1012**（2026-03-12），+4,978/-226，22 个文件。这是**空白区间中最大的单次提交**。

Qwen3.5 **不是** Qwen3 的增量更新。它是一个**架构全新的混合模型**，交替使用两种注意力机制 —— 这在现有的 Qwen2/Qwen3
代码中完全不存在。**现有的所有分析文档均未提及 Qwen3.5**。

### 3.2 混合注意力架构

每一层是 `full_attention` 或 `linear_attention` 之一，由 `layer_types` 列表配置:

```
Qwen3_5Model
├── TransformerBlock (layer_type="full_attention")
│   ├── GatedAttention       ← Q 投影输出 2x 宽度，sigmoid 门控
│   ├── FeedForward / MoE
│   └── Qwen3_5RMSNorm       ← (1+w)*norm(x)，权重初始化为 0
│
├── TransformerBlock (layer_type="linear_attention")
│   ├── GatedDeltaNet        ← 线性注意力，chunk-based delta rule
│   ├── FeedForward / MoE
│   └── Qwen3_5RMSNormGated  ← w*norm(x)*silu(gate)
│
└── ... (交替排列)
```

#### 3.2.1 GatedDeltaNet（线性注意力）

`model.py:141-315`（167 行），Qwen3.5 最核心的新模块:

1. 合并 QKV 因果卷积（`causal_conv1d_fn`，带 `seq_idx` 序列隔离）
1. 门控 delta rule 计算（`chunk_gated_delta_rule`，来自 `fla` 库）
1. 逐头门控 RMSNorm

**外部依赖处理**:

- `causal_conv1d_fn`: 有纯 PyTorch 回退（`nn.Conv1d + SiLU + 逐段处理`）
- `FusedRMSNormGated`: 有纯 PyTorch 回退（`Qwen3_5RMSNormGated`）
- `chunk_gated_delta_rule`: **无回退**，缺失时直接 `assert` 失败 — 这是正确的设计选择，因为纯 PyTorch 实现太慢无法用于训练

#### 3.2.2 GatedAttention（全注意力变体）

`model.py:322-428`（106 行），与标准 Archon 注意力的区别:

- Q 投影输出 2x 宽度，分裂为 `query` 和 `gate`
- 最终输出乘以 `sigmoid(gate)`
- 复用已有的 `VarlenAttentionWrapper` 处理 Flash Attention

#### 3.2.3 Qwen3.5 特有的差异

| 特性         | Qwen2/Qwen3                       | Qwen3.5                                  |
| ------------ | --------------------------------- | ---------------------------------------- |
| 注意力类型   | 单一（全注意力）                  | 混合（全注意力 + 线性注意力）            |
| RMSNorm      | `w * norm(x)`，权重初始化 1.0     | `(1+w) * norm(x)`，权重初始化 0.0        |
| RoPE         | 完整旋转                          | 部分旋转（`partial_rotary_factor=0.25`） |
| head_dim     | 由 `hidden_size / num_heads` 推导 | 显式配置（默认 256）                     |
| MoE 共享专家 | 标准                              | 带 sigmoid 门控（`shared_expert_gate`）  |

### 3.3 与 Archon 架构的集成

**完全遵循 ModelSpec 注册模式**，对引擎层零侵入:

```python
# qwen3_5/spec.py
QWEN3_5_SPEC = ModelSpec(
    name="Qwen3_5",
    model_class=Qwen3_5Model,
    model_args_class=Qwen3_5ModelArgs,
    state_dict_adapter_class=Qwen3_5StateDictAdapter,
    parallelize_fn=parallelize_qwen3_5,
    supported_model_types=frozenset({
        "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"
    }),
    pipelining_fn=pipeline_llm,
)
```

- `archon/__init__.py` 仅新增 1 行导入
- `archon_engine.py` **零修改** — 完全通过 ModelSpec dispatch 接入
- 复用已有的 `MoE`、`GroupedExperts`、`TokenChoiceTopKRouter` 基础设施

### 3.4 并行化现状: DP-Only

**这是 Qwen3.5 当前最大的限制**:

| 并行策略         | Qwen3（完整支持）                  | Qwen3.5（当前）       |
| ---------------- | ---------------------------------- | --------------------- |
| FSDP（数据并行） | ✅                                 | ✅                    |
| TP（张量并行）   | ✅ ColwiseParallel/RowwiseParallel | ❌ 仅日志警告，不生效 |
| CP（上下文并行） | ✅ Ulysses SP                      | ❌ 仅日志警告         |
| EP（专家并行）   | ✅ ExpertParallel/ETP              | ❌ 仅日志警告         |
| PP（流水线并行） | ✅ pipeline_llm                    | ✅ 复用共享实现       |

`parallelize_qwen3_5`（301 行）vs Qwen3 的 `parallelize_qwen3`（757 行）— **缺少 60% 的并行化代码**。

**风险**: 用户配置 `tp_size=4` 时，代码仅打印日志警告但继续运行，**静默降级为 FSDP-only**，可能导致 OOM 或预期外的性能表现。

### 3.5 测试覆盖

- `test_qwen3_5.py`（1,396 行）: 35+ 测试用例
  - 基础（CPU）: 配置解析、Norm 一致性、RoPE 部分旋转
  - 模块（GPU）: GatedDeltaNet 前向/反向一致性、序列打包精确匹配、因果卷积边界隔离
  - 集成（GPU）: 完整模型前向/反向 vs HF 参考，参数梯度余弦相似度 > 0.99
- `test_hf_parity_qwen3_5.py` + `test_hf_parity_qwen3_5_moe.py`: HF parity E2E 测试
- `test_state_dict_adapter.py`（446 行）: 状态字典双向转换

### 3.6 已知限制

| 限制                  | 严重性 | 说明                                                        |
| --------------------- | ------ | ----------------------------------------------------------- |
| 无 TP/CP/EP 支持      | **高** | 无法在大规模集群上高效训练大型 Qwen3.5                      |
| `fla` 硬依赖          | **中** | `chunk_gated_delta_rule` 无回退，未安装时运行时 assert 失败 |
| GatedDeltaNet 仅训练  | **中** | 无 KV-cache/decode 路径，不支持推理                         |
| Conv1d 回退低效       | **低** | 逐段处理有 Python 循环 + `.item()` GPU-CPU 同步             |
| Tree attention 不支持 | **低** | `tree_attn_meta` 参数类型标注为 `None`                      |

______________________________________________________________________

## 四、MoE Router 数值稳定性与 DTensor 修复

### 4.1 概述

**PR #1009**（Mar 8）+ **PR #1029**（Mar 16），+564/-127，14 个文件。

两个关联的改进，解决 MoE 路由器在大规模专家数量下的两个独立问题:

### 4.2 FP32 Router Gate GEMM（PR #1009）

**问题**: BF16 精度的 Router Gate GEMM 在大专家数量下导致数值不稳定，softmax/sigmoid 放大微小分数差异，造成路由振荡。

**方案**: 自定义 `torch.autograd.Function`（`RouterGatingLinearFunction`，`router.py:14-53`）:

```python
class RouterGatingLinearFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, input, weight, router_dtype):
        # 前向: 将 input 和 weight 提升到 router_dtype (FP32) 计算
        output = torch.mm(input.to(router_dtype), weight.to(router_dtype).t())
        ctx.save_for_backward(input, weight)  # 保存 BF16 激活 (省内存)
        ctx.router_dtype = router_dtype
        return output.to(input.dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        # 反向: 在 router_dtype 下计算梯度，转回原始 dtype
        ...
```

**设计亮点**:

- 激活保存为 BF16（内存效率），计算使用 FP32（数值稳定）
- `@torch.amp.custom_fwd/bwd` 防止 AMP 干扰显式 dtype 管理
- 源自 Megatron-Core，经过生产验证
- `router_dtype=None` 时完全回退到标准 `F.linear`，零开销

### 4.3 DTensor 兼容性修复（PR #1029）

**问题**: PR #1009 中 `TokenChoiceTopKRouter.forward()` 直接调用
`router_gating_linear(x, self.gate.weight, ...)` 函数，绕过了 `nn.Module.__call__()`。当使用
`ReplicateParallel` 进行 DTensor 并行时，DTensor 的 hook 不会触发，导致跨 TP rank 的梯度 all-reduce 缺失。

**方案**: 新增 `RouterGateLinear(nn.Linear)`（`router.py:76-106`），将 FP32 GEMM 封装在标准
`nn.Linear` 子类中:

```python
class RouterGateLinear(nn.Linear):
    def __init__(self, in_features, out_features, router_dtype=None, ...):
        super().__init__(in_features, out_features, bias=False)
        self.router_dtype = router_dtype

    def forward(self, input):
        return router_gating_linear(input, self.weight, self.router_dtype)
```

**效果**: `TokenChoiceTopKRouter` 现在使用 `self.gate = RouterGateLinear(...)`，前向简化为
`scores = self.gate(x)`，DTensor hook 通过标准 Module 调用路径正确触发。

### 4.4 成熟度评估

**生产就绪**。设计源自 Megatron-Core，有明确的回退路径，向后兼容（`router_dtype=None` 保持原有行为），并已集成到 Qwen3 MoE 的
HF config 加载路径中。

### 4.5 注意事项

- `Qwen3_5ModelArgs.from_hf_config` **缺少 `router_dtype` 透传**（与 Qwen3 不同），Qwen3.5 MoE
  用户无法通过标准路径配置 FP32 路由
- 缺少专用的数值稳定性测试（验证 FP32 vs BF16 路由在阈值附近的分歧）

______________________________________________________________________

## 五、Data Proxy RTensor 修复（次要）

### 5.1 RTensor 序列化简化（PR #1067）

`9639749e`（Mar 23），+36/-102。将 `proxy/server.py` 中 ~130 行的双路径序列化逻辑（分片元数据路径 vs
遗留列表路径）替换为统一的 `serialize_value`/`deserialize_value`（来自
`areal.infra.rpc.serialization`），精简至 ~20 行。

### 5.2 Data Proxy 批量端点（PR #1105）

`cbe35f5a`（Mar 30），+409/-1。PR #1077 新增了 `POST /data/batch` 批量 RTensor 获取，但仅在 Flask RPC
server 上实现，FastAPI Data Proxy 缺失此端点。修复后对齐了错误处理和 JSON 解析行为，并新增 12 个单元测试。

______________________________________________________________________

## 六、5D 并行策略更新（跨引擎）

### 6.1 勘误说明

> **原始分析仅覆盖了 7 个 Archon 专属 commits，遗漏了同一区间内 18 个直接影响 5D 并行策略的跨引擎 commits。** Archon
> 引擎的内部并行原语（`parallel_dims.py`、`pipeline_parallel.py`、`expert_parallel.py`、`ulysses.py`）确实无功能性逻辑变更（仅
> license header），但 FSDP、Megatron 引擎及跨引擎层面均有重大并行更新。

### 6.2 按并行维度分类的更新

#### 6.2.1 Context Parallelism（CP/Ulysses SP）— 3 commits，2 引擎

| Commit     | PR    | 引擎     | 类型       | 主题                                                 |
| ---------- | ----- | -------- | ---------- | ---------------------------------------------------- |
| `036ab169` | #929  | FSDP     | **新功能** | 视觉编码器跨 Ulysses SP rank 分片                    |
| `412d2241` | #990  | 全部     | 修复       | PPO token 统计在 CP 下不一致                         |
| `483a4e86` | #1079 | Megatron | **新功能** | BailingMoeV2.5: Lightning Attention + MLA + MoE + CP |

**视觉编码器 SP 分片**（`036ab169`，+936/-104）:

- 新增 `areal/models/transformers/vision_sp_shard.py`（424 行）
- 此前 VLM 使用 Ulysses SP 时，每个 SP rank 都冗余运行完整的 ViT 编码器（O(SP) 冗余计算）
- 新方案: 将图像贪心分配到各 SP rank，各 rank 本地运行 ViT，再 `all_gather` 汇聚完整 embedding
- 反向传播通过自定义 autograd Function 的 `all_reduce(SUM)` 保证梯度正确性
- 支持 Qwen2-VL、Qwen2.5-VL、Qwen3-VL

**BailingMoeV2.5 CP 支持**（`483a4e86`，+2,190/-52）:

- 新增 `areal/models/mcore/lightning_attention.py`（670 行）
- Lightning Attention（线性注意力）无法像标准 dot-product attention 那样直接因果分裂
- 采用 "head-parallel redistribution" 模式: CP format `[S/CP, B, H_local, D]` →
  head-parallel format `[S, B, H_local/CP, D]`，在完整序列上运行线性递归，然后反向 all-to-all
- 包含 Zigzag 负载均衡（`_build_zigzag_undo_indices`）和 MLA 的 TP 重复参数修复

#### 6.2.2 Tensor Parallelism（TP）— 4 commits，2 引擎

| Commit     | PR    | 引擎     | 类型       | 主题                                 |
| ---------- | ----- | -------- | ---------- | ------------------------------------ |
| `722e235a` | #1056 | Megatron | **新功能** | Megatron Bridge 适配（TP + PP 测试） |
| `9c70289c` | #1123 | Megatron | **新功能** | Megatron LoRA RL 训练（TP/PP 支持）  |
| `0ee85625` | #1118 | Archon   | 修复       | FP8 在 TP > 1 下的 DTensor 解包      |
| `483a4e86` | #1079 | Megatron | **新功能** | MLA TP all-gather 重复参数修复       |

**Megatron Bridge**（`722e235a`，+1,276/-1,405）:

- 新增 `megatron-bridge` 作为第二个 bridge 后端（alongside 已有的 `mbridge`）
- 显式测试 TP > 1 和 PP > 1 配置
- 升级 `megatron-core 0.13.1 → ~0.15.1`

**Megatron LoRA**（`9c70289c`，+776/-45）:

- 新增 `areal/engine/megatron_utils/megatron_lora.py`（296 行）
- 首个 MegatronEngine 的端到端 LoRA 路径
- PP 感知的层索引提取（正则表达式匹配 Megatron-core 线性层名）
- 禁用分布式优化器（LoRA 参数量小，使用标准 Adam 避免 ZeRO 状态分片复杂性）

#### 6.2.3 Pipeline Parallelism（PP）— 3 commits，2 引擎

| Commit     | PR    | 引擎     | 类型       | 主题                               |
| ---------- | ----- | -------- | ---------- | ---------------------------------- |
| `a4ea7730` | #1145 | vLLM     | **修复**   | PP > 1 时 XCCL LoRA 权重更新被覆盖 |
| `722e235a` | #1056 | Megatron | **新功能** | Megatron Bridge PP > 1 支持        |
| `cca5b865` | #1135 | Megatron | 修复       | Bridge 引入的树注意力缩进 bug      |

**PP > 1 LoRA 权重修复**（`a4ea7730`，+47/-5）:

- PP 下每个 stage 只持有部分层，单次 XCCL 权重更新只携带部分 LoRA 分片
- 原代码在收到第一个分片后立即调用 `update_lora_model()`，覆盖了不完整的权重
- 修复: 引入 `_lora_partial_shards` 缓冲区，收集所有 PP group 的分片后原子合并应用
- **没有此修复，PP > 1 + LoRA 推理会产生静默错误的输出**

#### 6.2.4 Data Parallelism（DP/FSDP）— 5 commits

| Commit     | PR    | 类型       | 主题                                         | +/-      |
| ---------- | ----- | ---------- | -------------------------------------------- | -------- |
| `c1bede50` | #983  | **新功能** | Per-layer 优化器步骤 + H2D/D2H 流水线        | +854/-26 |
| `2ddd9595` | #1074 | **性能**   | Pipeline weight sync + single pending bucket | +100/-31 |
| `595a3c4a` | #1139 | 修复       | LoRA 冻结 rank 的 grad norm 挂起             | +24/-4   |
| `f34bea8b` | #1182 | 修复       | 非 rank-0 使用 meta 设备避免 CPU OOM         | +32/-6   |
| `61281ba8` | #1108 | 重构       | PerLayerOptimWrapper 平台抽象                | +11/-11  |

**Per-layer 优化器流水线**（`c1bede50`，+854 行）:

- 新增 `areal/engine/fsdp_utils/optimizer.py`（451 行）
- CPU offload 优化器步骤从"全量 Adam on CPU"改为**逐层流水线**: H2D 预取第 i+1 层 → 设备上执行 Adam 第 i 层 → D2H
  回存第 i-1 层
- 三个 CUDA stream 并行: `_h2d_stream`、`_d2h_stream`、compute stream
- 新配置: `per_layer_optim_step`、`optim_step_prefetch_layers`

**Pipeline weight sync 优化**（`2ddd9595`，+100 行）:

- 将 FSDP 权重同步从同步逐桶广播改为**异步单桶流水线**
- Bucket N-1 的广播与 Bucket N 的 all-gather 重叠执行
- `_PendingWeightUpdateBucket` 数据类跟踪异步句柄和专用 CUDA stream

#### 6.2.5 Expert Parallelism（EP）— 3 commits

| Commit     | PR    | 引擎     | 类型       | 主题                                                  |
| ---------- | ----- | -------- | ---------- | ----------------------------------------------------- |
| `978532ea` | #1029 | Archon   | 修复       | Router Gate DTensor hook 在 EP 下不触发               |
| `4f5a2944` | #1009 | Archon   | **新功能** | MoE Router FP32 Gate GEMM（改善大规模专家路由稳定性） |
| `483a4e86` | #1079 | Megatron | **新功能** | BailingMoeV2.5（256 专家，top-8 路由）                |

#### 6.2.6 跨维度 / 跨引擎 — 4 commits

| Commit     | PR    | 影响维度    | 类型       | 主题                                           |
| ---------- | ----- | ----------- | ---------- | ---------------------------------------------- |
| `93b572d0` | #1044 | **全部 5D** | 破坏性重构 | `allocation_mode` → 每引擎 `backend` 字段迁移  |
| `9c86e0f7` | #1109 | DP + PP     | 修复       | 跨引擎的 padded 分布式 eval 加固               |
| `aebe13d8` | #1083 | 全部        | 新功能     | 全部三个引擎的 NUMA CPU 亲和性绑定             |
| `03d71153` | #930  | DP + EP     | 新功能     | 完整 MIS/TIS 支持（off-policy MoE 训练稳定性） |

**`allocation_mode` 破坏性 API 重构**（`93b572d0`，153 文件，+2,452/-2,119）:

- 移除全局 `AllocationMode` 字符串（如 `"sglang[rollout]:d2+fsdp[actor]:d4"`）
- 替换为每引擎的显式 `backend` 字段（如 `actor.backend="fsdp:d4"`、`rollout.backend="sglang:d4t2"`）
- `AllocationMode` 重命名为 `_AllocationMode` 并发出 `FutureWarning`
- **正式限定 FSDP 仅支持 DP × TP × CP**（PP 和 EP 在 `ModelAllocation.__post_init__` 中抛
  `AllocationValidationError`）

### 6.3 并行维度影响矩阵

| Commit     | PP  | DP  | CP  | TP  | EP  | 引擎     | 类型   |
| ---------- | :-: | :-: | :-: | :-: | :-: | -------- | ------ |
| `2ddd9595` |     |  ●  |     |     |     | FSDP     | 性能   |
| `c1bede50` |     |  ●  |     |     |     | FSDP     | 新功能 |
| `f34bea8b` |     |  ●  |     |     |     | FSDP     | 修复   |
| `595a3c4a` |     |  ●  |     |     |     | FSDP     | 修复   |
| `61281ba8` |     |  ●  |     |     |     | FSDP     | 重构   |
| `036ab169` |     |     |  ●  |     |     | FSDP     | 新功能 |
| `412d2241` |     |     |  ●  |     |     | 全部     | 修复   |
| `483a4e86` |     |     |  ●  |  ●  |  ●  | Megatron | 新功能 |
| `722e235a` |  ●  |     |     |  ●  |     | Megatron | 新功能 |
| `9c70289c` |  ●  |     |     |  ●  |     | Megatron | 新功能 |
| `a4ea7730` |  ●  |     |     |     |     | vLLM     | 修复   |
| `cca5b865` |  ●  |     |     |     |     | Megatron | 修复   |
| `0ee85625` |     |     |     |  ●  |  ●  | Archon   | 修复   |
| `978532ea` |     |     |     |     |  ●  | Archon   | 修复   |
| `4f5a2944` |     |     |     |     |  ●  | Archon   | 新功能 |
| `93b572d0` |  ●  |  ●  |  ●  |  ●  |  ●  | 全部     | 重构   |
| `9c86e0f7` |  ●  |  ●  |     |     |     | 全部     | 修复   |
| `aebe13d8` |  ●  |  ●  |  ●  |  ●  |  ●  | 全部     | 新功能 |

### 6.4 量化统计

| 维度             | Commits | 新功能 | 修复                | 代码量             |
| ---------------- | ------- | ------ | ------------------- | ------------------ |
| CP/Ulysses       | 3       | 2      | 1                   | +3,306 行          |
| TP               | 4       | 3      | 1                   | +2,087 行          |
| PP               | 4       | 1      | 3                   | +1,323 行          |
| DP/FSDP          | 5       | 2      | 3                   | +1,008 行          |
| EP/MoE           | 3       | 2      | 1                   | +564 行            |
| 跨维度 API       | 4       | 2      | 2                   | +3,489 行          |
| **合计（去重）** | **18**  | **9**  | **6 修复 + 3 重构** | **+10,366/-4,020** |

### 6.5 小结

**Archon 引擎的内部 5D 并行原语无功能变更** — 这一点是准确的。但从整个框架视角看:

- **CP 是最活跃的并行维度**: 3 个 commits，新增 VLM 视觉编码器 SP 分片和 BailingMoeV2.5 的 Lightning Attention
  CP 支持
- **Megatron 引擎获得了全功能 TP + PP**: 通过 `megatron-bridge` 适配层和 LoRA 集成
- **FSDP 引擎获得两个主要性能优化**: per-layer 优化器流水线和权重同步 pending bucket
- **vLLM PP > 1 + LoRA 存在静默数据损坏 bug 被修复**: 此前部分分片覆盖导致推理结果错误
- **并行配置 API 发生了破坏性重构**: `allocation_mode` → 每引擎 `backend` 字段

______________________________________________________________________

## 七、代码质量评审发现

### 6.1 高严重性

| #   | 位置                        | 问题                               | 说明                                                                    |
| --- | --------------------------- | ---------------------------------- | ----------------------------------------------------------------------- |
| H1  | `router.py:290-295`         | `torch.histc` 在整数输入上可能误计 | 建议替换为 `torch.bincount`，更安全且专为整数计数设计                   |
| H2  | `fp8_checkpoint.py:174-181` | Shard(1) 运行时 "地雷"             | TP > 1 + FP8 检查点加载会在 `_dequant_dtensor` 深处才失败，建议提前验证 |

### 6.2 中严重性

| #   | 位置                                  | 问题                                                       |
| --- | ------------------------------------- | ---------------------------------------------------------- |
| M1  | `qwen3_5/model/model.py:645`          | `max_seqlen.item()` 在前向路径触发 GPU-CPU 同步            |
| M2  | `qwen3_5/model/args.py:91-96`         | `layer_types` 值无验证，拼写错误会静默创建错误层           |
| M3  | `qwen3_5/model/model.py:228-231`      | GatedDeltaNet 前向中按需创建张量，非预期路径               |
| M4  | `qwen3_5/model/model.py:255-264`      | Conv1d 回退逐段处理有循环内 `.item()`                      |
| M5  | `fp8_checkpoint.py:47-116`            | `_prepare_fp8_state_dict` 就地修改输入并返回，API 语义模糊 |
| M7  | `qwen3_5/infra/parallelize.py:95-111` | TP/CP/EP 不支持时仅警告不报错，用户可能不知道并行未生效    |

### 6.3 代码质量亮点

1. **文档质量优秀**: 每个公开函数有完整 docstring（Args/Returns/Raises）
1. **模式一致性**: Qwen3.5 严格遵循 Qwen3 的 TransformerBlock/FeedForward/init_weights 结构
1. **FP8 防御性验证**: `validate_fp8_shard_alignment()` 提供了精确的错误信息
1. **安全回退**: GatedDeltaNet 的 `causal_conv1d_fn` 和 Norm 都有测试过的回退路径
1. **Router 自定义 autograd**: 正确实现混合精度 + 内存高效的 saved tensors
1. **代码风格合规**: Logger 使用 PascalCase，无通配符导入，命名遵循 `XxxModel`/`XxxModelArgs`

______________________________________________________________________

## 八、量化影响分析

### 7.1 代码库增长

| 指标            | Gap 前（v1.0.0） | Gap 后（v1.0.3） | 增长       |
| --------------- | ---------------- | ---------------- | ---------- |
| Archon 模型代码 | 7,593 行         | 10,304 行        | **+35.7%** |
| Archon 引擎代码 | 2,698 行         | 3,005 行         | **+11.4%** |
| Archon 测试代码 | 13,504 行        | 18,677 行        | **+38.3%** |
| **合计**        | **23,795 行**    | **31,986 行**    | **+34.4%** |

### 7.2 配置面扩展

| 指标                       | Gap 前 | Gap 后 | 变化         |
| -------------------------- | ------ | ------ | ------------ |
| `cli_args.py` field 声明数 | 257    | 269    | +12 (+4.7%)  |
| `cli_args.py` 总行数       | 2,285  | 2,458  | +173 (+7.6%) |

新增嵌套数据类: `ArchonFP8Config`（4 字段 + 2 属性）。

### 7.3 两个提交主导增长

| Commit     | 功能     | 净增行     | 占 Gap 总量 |
| ---------- | -------- | ---------- | ----------- |
| `1927decc` | Qwen3.5  | +4,752     | 56.1%       |
| `f6331e09` | FP8 训练 | +2,873     | 33.9%       |
| **合计**   |          | **+7,625** | **90.0%**   |

### 7.4 贡献者

| 贡献者       | Commits | 主要贡献                   |
| ------------ | ------- | -------------------------- |
| Wentai Zhang | 5       | FP8、Qwen3.5、Router dtype |
| Wei Fu       | 1       | RTensor 简化               |
| fishcrap     | 1       | Router DTensor 修复        |

______________________________________________________________________

## 九、与现有文档的关联

### 9.1 现有文档需要更新的内容

| 现有文档                                    | 需要补充的内容                                           |
| ------------------------------------------- | -------------------------------------------------------- |
| `archon_areal_architecture_notes.md`        | FP8 训练子系统、Qwen3.5 混合注意力、ModelSpec 新增条目   |
| `archon_expert_parallel_deep_dive.md`       | `RouterGateLinear` DTensor 兼容机制、`router_dtype` 配置 |
| `archon_compile_deep_dive.md`               | FP8 与 torch.compile 互斥约束                            |
| `archon_activation_checkpoint_deep_dive.md` | FP8 scale 张量与 AC 的交互                               |
| `archon_parallel_dims_deep_dive.md`         | Qwen3.5 DP-only 约束对 ParallelDims 的影响               |

### 9.2 特征间交互

| 特性 A      | 特性 B          | 交互关系                                                                |
| ----------- | --------------- | ----------------------------------------------------------------------- |
| FP8 Linear  | MoE FP32 Router | **共存设计**: FP8 默认排除 `router`，Router 保持 BF16/FP32              |
| FP8 Experts | MoE FP32 Router | **独立**: FP8 专家修补 `GroupedExperts`，不触及 Router                  |
| Qwen3.5     | FP8             | **兼容**: Qwen3.5 的 `nn.Linear` 可被 FP8 修补（Conv1d 不受影响）       |
| Qwen3.5     | FP32 Router     | **兼容但有缺口**: 复用共享 MoE 基础设施，但缺少 `router_dtype` 配置透传 |

______________________________________________________________________

## 附录: 完整 Commit 清单

| #   | 日期   | Commit     | PR    | 类型 | 主题                                     | 新文件 | +/-         |
| --- | ------ | ---------- | ----- | ---- | ---------------------------------------- | ------ | ----------- |
| 1   | Mar 8  | `4f5a2944` | #1009 | feat | moe_router_dtype 配置 (FP32 Router Gate) | 1      | +478/-120   |
| 2   | Mar 12 | `1927decc` | #1012 | feat | Qwen3.5 dense + MoE 支持 (DP-only)       | 13     | +4,978/-226 |
| 3   | Mar 16 | `978532ea` | #1029 | fix  | RouterGateLinear DTensor 兼容            | 0      | +86/-7      |
| 4   | Mar 23 | `9639749e` | #1067 | fix  | RTensor 序列化简化                       | 0      | +36/-102    |
| 5   | Mar 30 | `cbe35f5a` | #1105 | fix  | Data Proxy POST /data/batch 端点         | 1      | +409/-1     |
| 6   | Mar 31 | `f6331e09` | #1087 | feat | FP8 Blockwise 训练支持                   | 17     | +2,958/-85  |
| 7   | Mar 31 | `0ee85625` | #1118 | fix  | FP8 TP/MoE 鲁棒性强化                    | 0      | +94/-28     |
