# 模型层

> 源码位置：`areal/models/` 文件数：23 个 | 总行数：8192 行

______________________________________________________________________

## 1. 概述

模型层位于引擎层与 HuggingFace/Megatron-Core 之间，承担三项核心职责：

1. **并行样式与序列并行**：为 FSDP2 和 HuggingFace Transformers 提供 Ulysses 序列并行（All-to-All 通信）以及自定义
   `ParallelStyle` 支持。
1. **Megatron-Core 桥接**：将 HuggingFace 权重格式双向转换至 MCore 格式，包括 QKV 合并/拆分、MoE 专家分片、FP8 量化等。
1. **树注意力**：为共享前缀的批次序列提供基于 Trie 的打包方案，配合 PyTorch Flex Attention 或 Triton
   自定义内核实现非标准因果掩码下的高效训练。

```
                       +-----------------------+
                       |   HuggingFace / MCore  |
                       +-----------+-----------+
                                   |
          +------------------------+------------------------+
          |                        |                        |
+---------v--------+  +-----------v-----------+  +---------v--------+
| fsdp/             |  | mcore/                 |  | transformers/    |
| ulysses.py (284)  |  | registry.py (326)      |  | ulyssess_patch   |
| parallel_styles   |  | hf_load.py (755)       |  |   (249)          |
|   .py (81)        |  | hf_save.py (853)       |  | qwen2_vl (87)    |
+------------------+  | bailing_moe.py (466)   |  | qwen3_vl (143)   |
                       | bailing_moe_bridge     |  | vision_sp_shard  |
                       |   (393)                |  |   (426)          |
                       | lightning_attn (672)   |  +------------------+
                       | common.py (100)        |
                       | qwen3.py (34)          |
                       +-----------------------+
                                   |
                       +-----------v-----------+
                       | tree_attn/             |
                       | tree.py (897)          |
                       | triton_kernel.py(1039) |
                       | functional.py (639)    |
                       | module_fsdp.py (189)   |
                       | module_megatron (218)  |
                       | module_archon (153)    |
                       | module.py (62)         |
                       | visualize.py (108)     |
                       | constants.py (18)      |
                       +-----------------------+
```

______________________________________________________________________

## 2. 文件清单

| #   | 文件路径                                 | 行数 | 职责                                              |
| --- | ---------------------------------------- | ---- | ------------------------------------------------- |
| 1   | `models/parallel_styles.py`              | 81   | 自定义 `ReplicateParallel` 并行样式               |
| 2   | `models/fsdp/ulysses.py`                 | 284  | Ulysses 序列并行核心：All-to-All 通信原语         |
| 3   | `models/mcore/registry.py`               | 326  | MCore 模型注册表：配置转换、模型创建、ValueHead   |
| 4   | `models/mcore/common.py`                 | 100  | HF-to-MCore 基础配置映射工具                      |
| 5   | `models/mcore/qwen3.py`                  | 34   | Qwen3 Dense 专用配置转换                          |
| 6   | `models/mcore/hf_load.py`                | 755  | HF SafeTensors -> MCore 权重加载（含 TP/EP 分片） |
| 7   | `models/mcore/hf_save.py`                | 853  | MCore -> HF SafeTensors 权重保存（含并行归约）    |
| 8   | `models/mcore/bailing_moe.py`            | 466  | BailingMoeV2.5 异构层规格构建（Lightning + MLA）  |
| 9   | `models/mcore/bailing_moe_bridge.py`     | 393  | BailingMoe mbridge Bridge 注册与权重名映射        |
| 10  | `models/mcore/lightning_attention.py`    | 672  | Lightning Attention MCore 模块（fla 内核 + CP）   |
| 11  | `models/transformers/qwen2_vl.py`        | 87   | Qwen2-VL Ulysses 注意力补丁                       |
| 12  | `models/transformers/qwen3_vl.py`        | 143  | Qwen3-VL Ulysses 注意力补丁 + TP deepstack 修复   |
| 13  | `models/transformers/ulyssess_patch.py`  | 249  | 通用 Ulysses Monkey-Patch 入口                    |
| 14  | `models/transformers/vision_sp_shard.py` | 426  | 视觉模型跨 SP 分片编码                            |
| 15  | `models/tree_attn/constants.py`          | 18   | 树注意力常量（BLOCK_SIZE、Triton 开关）           |
| 16  | `models/tree_attn/tree.py`               | 897  | Trie 构建、贪心打包、掩码生成、position_ids       |
| 17  | `models/tree_attn/triton_kernel.py`      | 1039 | Triton 树注意力前向/反向内核                      |
| 18  | `models/tree_attn/functional.py`         | 639  | 树结构 logprob/entropy 计算（节点缓存优化）       |
| 19  | `models/tree_attn/module_fsdp.py`        | 189  | FSDP 引擎树注意力补丁（Flex Attention）           |
| 20  | `models/tree_attn/module_megatron.py`    | 218  | Megatron 引擎树注意力（PytorchFlexAttention）     |
| 21  | `models/tree_attn/module_archon.py`      | 153  | Archon 引擎树注意力封装                           |
| 22  | `models/tree_attn/module.py`             | 62   | 树注意力统一导出模块                              |
| 23  | `models/tree_attn/visualize.py`          | 108  | 注意力掩码 ASCII 可视化工具                       |

______________________________________________________________________

## 3. Ulysses 序列并行

### 3.1 核心通信模式

Ulysses 序列并行的核心思路是：在注意力计算前，用 All-to-All 集合通信将序列维度"换"成头维度，使每个 Rank 持有**完整序列、部分头**，从而在每个
Rank 上独立做标准注意力，再用逆 All-to-All 换回来。

```
输入: [batch, seq/SP, heads, dim]   (每个 Rank 持有部分序列)
         |
         v  All-to-All (scatter heads, gather seq)
         |
      [batch, seq, heads/SP, dim]    (每个 Rank 持有完整序列、部分头)
         |
         v  标准 Flash Attention
         |
      [batch, seq, heads/SP, dim]
         |
         v  All-to-All (scatter seq, gather heads)
         |
输出: [batch, seq/SP, heads, dim]   (恢复原始分布)
```

**关键函数**（`fsdp/ulysses.py`）：

| 函数                           | 行号    | 说明                                                                        |
| ------------------------------ | ------- | --------------------------------------------------------------------------- |
| `all_to_all_tensor`            | 第153行 | 底层通用 All-to-All，使用 `all_to_all_single_autograd` 兼容 `torch.compile` |
| `_gather_seq_scatter_heads`    | 第46行  | `[bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]`                              |
| `_gather_heads_scatter_seq`    | 第71行  | `[bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]`                              |
| `ulysses_pad_and_slice_inputs` | 第233行 | Pad + 按 Rank 切片输入                                                      |
| `ulysses_prepare_inputs`       | 第249行 | 为 loss 计算准备 rolled labels                                              |

**对齐处理**：当序列长度不能被 SP 组大小整除时，先进行零填充（`_pad_tensor`，第118行），All-to-All
后再移除填充（`_unpad_tensor`，第127行）。

### 3.2 HuggingFace Transformers 补丁

补丁通过 Monkey-Patch 方式注入，入口在 `transformers/ulyssess_patch.py` 的 `apply_monkey_patch`
函数（第149行）：

```
apply_monkey_patch(model, ulysses_sp_size, shard_vision_across_sp)
    |
    +-- VL 模型: 替换 Attention.forward + 包装 TextModel.forward
    |     |-- Qwen2-VL   -> qwen2_vl.ulysses_flash_attn_forward
    |     |-- Qwen2.5-VL -> 同上（复用 Qwen2 补丁）
    |     |-- Qwen3-VL   -> qwen3_vl.ulysses_flash_attn_forward
    |     +-- 可选: apply_vision_sp_shard_patch()
    |
    +-- 纯文本模型: 替换 _flash_attention_forward 全局函数
```

**VLM 输入切片**（`ulyssess_patch.py` 第74行 `patch_vlm_for_ulysses_input_slicing`）：

- 在第一次 `forward` 调用时切片 `inputs_embeds`
- 对 Qwen3-VL 的 `visual_pos_masks` 和 `deepstack_visual_embeds` 做对应切片
- 使用 `_needs_initial_slice` 标志确保只在最外层切片一次

### 3.3 Vision SP Shard

`vision_sp_shard.py` 解决的问题：在 Ulysses SP 下，每个 Rank 只持有部分文本序列，但视觉嵌入需要完整。如果每个 Rank 都跑完整
ViT，则有冗余计算。

**策略**：按整图（不是按 patch）分配给各 Rank，各 Rank 独立编码，最后 All-Gather 拼合。

```
SP Rank 0: 编码图片 0, 1     --+
SP Rank 1: 编码图片 2, 3     --+--> All-Gather --> 全部嵌入 (0,1,2,3,4,5)
SP Rank 2: 编码图片 4, 5     --+
```

- `_assign_images_to_dp_ranks`（第56行）：贪心连续装箱分配
- `GatherVisionEmbeddings`（第145行）：自定义 `autograd.Function`，前向 all_gather，反向
  all_reduce(SUM) + 按分配切片
- 支持 Qwen3-VL 的 deepstack 输出（`_unpack_deepstack`，第235行）

______________________________________________________________________

## 4. ReplicateParallel 并行样式

`parallel_styles.py`（81行）提供 `ReplicateParallel` 类，继承自 PyTorch 的 `ParallelStyle`。

**用途**：在张量并行下，某些模块（如 MoE 路由门控、QK Norm、Critic 评分层）需要保持复制计算，但输入/输出可能是
`DTensor`。`ReplicateParallel` 通过前向钩子完成 `Tensor <-> DTensor` 转换：

```
输入 (可能是 Tensor)
    |-- _prepare_input_fn: 转 DTensor(Replicate)，可选 redistribute
    |-- 模块原始 forward
    |-- _prepare_output_fn: redistribute + 可选 to_local()
输出 (可能是 Tensor)
```

______________________________________________________________________

## 5. Megatron-Core 桥接

### 5.1 架构注册表

`registry.py`（326行）是 MCore 模型创建的统一入口：

| 函数                       | 行号    | 说明                                                                        |
| -------------------------- | ------- | --------------------------------------------------------------------------- |
| `make_hf_and_mcore_config` | 第115行 | HF config -> `TransformerConfig`，支持 mbridge / megatron-bridge / 直接转换 |
| `make_mcore_layer_specs`   | 第151行 | 按架构（Qwen3/BailingMoe）生成层规格                                        |
| `make_mcore_model`         | 第168行 | 创建 `GPTModel`，支持 mbridge/megatron-bridge/直接构建                      |
| `ValueHead`                | 第30行  | Critic 模型值头（替换 output_layer，支持序列并行 gather）                   |
| `unwrap_to_gpt_model`      | 第95行  | 从 DDP/VLM 包装中提取底层 `GPTModel`                                        |

**支持的架构分发**（第137行）：

```
architecture == "Qwen3ForCausalLM"
    -> hf_to_mcore_config_qwen3_dense + make_mcore_layer_specs_qwen3_dense

architecture in ("BailingMoeV2_5", "BailingMoeLinear", "BailingHybrid")
    -> hf_to_mcore_config_bailing_moe + make_mcore_layer_specs_bailing_moe
```

### 5.2 HF -> MCore 权重加载

`hf_load.py`（755行）的核心函数 `load_weights_from_hf_with_mbridge_fast`（第641行）：

```
safetensors 文件 --> safe_open 惰性切片
      |
      v  mbridge._weight_name_mapping (local -> global -> hf)
      |
      v  _weight_to_mcore_tp (按类型分发)
      |
      +-- QKV: _merge_qkv_weights (3路合并 + TP 切片)
      |         或 _load_fused_qkv_weight (已融合 -> 交错重排)
      +-- FC1: _merge_gate_up_weights (gate+up 合并)
      +-- MoE Expert: _slice_moe_expert_weight (dim=1 切片)
      |   +-- Stacked: _slice_moe_expert_fc1_stacked_gate_up (3D 展开)
      |   +-- Stacked: _slice_moe_expert_fc2_stacked_down (转置 + 切片)
      +-- Vision QKV: _convert_vision_qkv_hf_to_mcore (分组->交错)
      +-- 通用: _slice_generic_weight (自动推断分割维度)
      |
      v  FP8 处理 (dequantize / FP8BlockwiseTensorHelper)
      |
      v  param.copy_() 或 to_te_fp8_inplace()
```

**并行加载优化**：

- 使用 Union-Find 算法（`make_filename_bins`，第569行）将共享同一 safetensors 文件的权重分组
- 分组后用 `ThreadPoolExecutor` 多线程并行加载（第733行）

### 5.3 MCore -> HF 权重保存

`hf_save.py`（853行）的核心函数 `save_weights_to_hf_with_mbridge_fast`（第443行）：

```
MCore 参数 --> 分为 non-expert 和 expert 两组
      |
      v  Non-expert: all_gather TP -> _weight_merge -> _weight_to_hf
      |       保存: 按 PP rank 分片
      |
      v  Expert:
      |   +-- Per-expert-flat (Qwen3-MoE, BailingMoe, DeepSeek):
      |   |     各 EP rank 独立保存各自专家
      |   +-- Stacked (Qwen3-VL-MoE):
      |         EP all-gather -> 合并 -> 只 ep_rank=0 写入
      |
      v  元数据: config.json 修补 + safetensors.index.json
```

**关键设计**：

- `McoreDistributedWeightSpec`（第232行）：描述每个参数的分布属性（TP/PP/VPP rank、形状、dtype）
- `_bridge_uses_stacked_experts`（第184行）：通过检查 `_MLP_MAPPING` 是否含 `{expert_id}` 模板来判断 flat
  vs stacked
- 多线程分片保存（第686行），每个 GPU 保存自己负责的分片

### 5.4 BailingMoe 异构层

BailingMoeV2.5 使用混合注意力架构：

```
层 0: Lightning Attention (线性) + Dense MLP
层 1: Lightning Attention (线性) + Dense MLP
层 2: Lightning Attention (线性) + Dense MLP
层 3: MLA (标准 Softmax)        + MoE MLP    <-- 每 layer_group_size 层
层 4: Lightning Attention (线性) + MoE MLP
...
```

**层类型判定**（`bailing_moe.py` 第109行 `is_lightning_layer`）：

```python
(layer_number + 1) % layer_group_size != 0  # True = Lightning, False = MLA
```

**4 种层规格组合**（`make_mcore_layer_specs_bailing_moe`，第277行）：

1. Lightning Attention + Dense MLP
1. Lightning Attention + MoE MLP
1. MLA Attention + Dense MLP
1. MLA Attention + MoE MLP

**MLA RoPE 修补**（第41行 `_patch_mla_thd_rope_for_cp`）：修复 CP>1 下 MLA 的 THD 打包序列 RoPE
频率表被截断的问题。当检测到频率表过短时，利用 `freqs[p] = p * freqs[1]` 的线性关系重建完整表。

### 5.5 Lightning Attention

`lightning_attention.py`（672行）在 MCore 框架内实现 Lightning Attention（线性注意力 + 学习衰减）：

**核心组件**：

| 类                       | 行号    | 说明                                          |
| ------------------------ | ------- | --------------------------------------------- |
| `LightningCoreAttention` | 第241行 | 核心计算，调用 fla 的 `chunk_simple_gla` 内核 |
| `GroupRMSNorm`           | 第344行 | 分组 RMSNorm，用于门控归一化                  |
| `LightningSelfAttention` | 第387行 | 完整注意力层，集成 QKV/Gate/Proj 投影         |

**g_gamma 衰减计算**（第276行）：

```
alibi_slopes = ALiBi 几何级数斜率（全局 H 个头）
layer_scale  = 1 - layer_idx/(num_layers-1) + 1e-5
g_gamma      = -alibi_slopes * layer_scale
g_gamma_local = g_gamma[tp_rank * H_local : (tp_rank+1) * H_local]  # TP 切片
```

**CP（上下文并行）支持**（`forward` 第624行起）：

```
                    输入: [S/CP, B, H_local, D]
                        |
                        v  All-to-All CP->HP: [S, B, H_local/CP, D]
                        |
                        v  撤销 zigzag 排序（恢复顺序 token 序）
                        |
                        v  RoPE 应用
                        |
                        v  Lightning Attention (fla kernel, CP 切片 g_gamma)
                        |
                        v  重做 zigzag 排序
                        |
                        v  All-to-All HP->CP: [S/CP, B, H_local, D]
                    输出
```

### 5.6 BailingMoe Bridge

`bailing_moe_bridge.py`（393行）通过 `@register_model` 装饰器将 `BailingMoeBridge` 注册到 mbridge：

```python
@register_model("bailing_moe_v2")
@register_model("bailing_moe_linear")
@register_model("bailing_hybrid")
class BailingMoeBridge(LLMBridge): ...
```

**注意力权重名映射**（按层类型分发，第340行）：

- Lightning 层：`_LIGHTNING_ATTENTION_MAPPING`（query_key_value, g_proj, g_norm 等）
- MLA 层（q_lora_rank=None）：`_MLA_ATTENTION_MAPPING_Q_DIRECT`
- MLA 层（q_lora_rank!=None）：`_MLA_ATTENTION_MAPPING_Q_LORA`（q_a_proj, q_a_layernorm,
  q_b_proj）

**QKV 格式转换**（`_weight_to_mcore_format`，第260行）：

- HF: `[3*H*D, hidden]`（Q_all | K_all | V_all 拼接）
- MCore: `[H*3*D, hidden]`（q0,k0,v0 | q1,k1,v1 | ... 交错）

______________________________________________________________________

## 6. 树注意力

### 6.1 整体架构

树注意力为共享前缀的序列提供计算优化。在 RL 训练中，同一 prompt 生成多个 response，共享前缀部分只需计算一次：

```
普通批处理 (独立序列):         树打包:
  Seq0: [A B C D]               Trie:
  Seq1: [A B E F]                 A - B - C - D  (seq0)
  Seq2: [A B E G]                       +- E - F  (seq1)
                                        +- E - G  (seq2)
  总 token 数: 12                 总 token 数: 8  (节省 33%)
```

### 6.2 Trie 构建与打包

`tree.py`（897行）是树注意力的核心：

**数据结构**（第41行 `TrieNode`）：

- `start_idx / end_idx`：在扁平化树中的位置范围
- `tokens`：本节点存储的 token 列表
- `sequence_ids`：经过本节点的序列 ID
- `ancestors`：祖先节点列表（根到父）
- `nodes`：所有后代节点（仅根节点使用）

**构建流程**（`build_packed_tree_batch`，第273行）：

```
输入数据 (input_ids, attention_mask)
    |
    v  _extract_sequences: 提取有效 token 序列
    |
    v  _greedy_build_tries: 贪心插入
    |     |-- 尝试插入现有树（检查追加节点数不超过 max_tokens_per_tree）
    |     +-- 放不下则新建树
    |
    v  _compress_trie: 链压缩
    |     |-- 沿单子节点链合并为一个 TrieNode
    |     +-- 验证 sequence_ids 和 node_id 连续性
    |
    v  每棵树独立打包:
    |     |-- _pack_input_ids: 按 trie 结构重排 token
    |     |-- _build_attention_mask: 构建 2D 因果掩码（块式处理控制内存）
    |     |-- get_packed_tree_position_ids: 从掩码计算 position_ids
    |     +-- _pack_extra_data: 打包附加张量（loss_mask 等）
    |
    v  DP 同步: all_gather 树数量，追加 dummy 树对齐
    |
    v  输出 MicroBatchList
```

**注意力掩码生成**（`_build_attention_mask`，第572行）：

- 对每个 sequence_id，收集其在 trie 中经过的所有节点位置
- 在这些位置间施加因果掩码（下三角）
- 块式处理（`_apply_causal_mask_blockwise`，第611行）限制峰值内存为 O(BLOCK^2) 而非 O(N^2)

**position_ids**（`get_packed_tree_position_ids`，第708行）：

```python
ancestor_counts = attention_mask.bool().sum(dim=-1)  # 每个 token 能看到多少祖先
position_ids = clamp_min(ancestor_counts - 1, 0)     # 0-indexed
```

### 6.3 Triton 树注意力内核

`triton_kernel.py`（1039行）实现了完整的前向+反向 Triton 内核。

**核心数据结构** `TreeAttentionData`（第19行）：

```
packed_mask: (B, N, ceil(N/64))   -- 位打包祖先掩码，每 64 token 一个 int64
kv_indices:  1D 稀疏索引          -- 每个 Q 块需要访问的 KV 块列表
kv_offsets:  (B, num_q_blocks+1)  -- kv_indices 的偏移量表
q_indices:   1D 稀疏索引          -- 反向映射（用于 dK/dV 反向传播）
q_offsets:   (B, num_kv_blocks+1) -- q_indices 的偏移量表
```

**位打包掩码构建**（`compute_packed_mask`，第27行）：

```
对每个 token i:
    packed[b, i] = packed[b, parent(i)]  |  (1 << i)
    # 继承父节点的祖先位，并设置自身位
```

**前向内核**（`_tree_attn_fwd_triton`，第180行）：

- Grid: `(ceil(N/BLOCK_M), B*H)`
- 使用 online softmax（数值稳定）
- 从 `kv_offsets` 读取本 Q 块的有效 KV 块范围，仅迭代非零块
- 位解包掩码：`((mask_word >> bit_indices) & 1) != 0`
- 支持 GQA（`off_h_kv = off_h // GQA_GROUP_SIZE`）
- 输出 LSE (log-sum-exp) 供反向使用

**反向内核**分为三个阶段：

1. `_tree_attn_bwd_preprocess`（第425行）：计算 Delta = sum(O * dO)
1. `_tree_attn_bwd_dq`（第473行）：使用 KV 稀疏索引计算 dQ
1. `_tree_attn_bwd_dkdv`（第627行）：遍历所有 Q 块计算 dK/dV，内含 GQA 组循环

### 6.4 树结构 logprob 计算

`functional.py`（639行）为树打包序列提供 logprob 和 entropy 计算：

**挑战**：树打包后，标准的"roll input_ids"获取 labels 的方法不再适用——同一位置的 next-token 在不同序列中可能不同。

**解决方案**（`_gather_packed_tree_logprobs`，第206行）：

1. 遍历每个 sequence_id
1. 获取该序列经过的 trie 节点列表 `[(start0,end0), (start1,end1), ...]`
1. 对每个节点内部计算 logprobs（`_compute_internal_node_logprobs`）
1. 计算节点间转移的 logprob（`_compute_transition_logprob`）
1. 拼接得到完整序列的 logprobs

**缓存优化**：

- `node_cache`: 键 `(start_idx, end_idx)` -> 节点内部 logprobs
- `transition_cache`: 键 `(pred_pos, label_pos)` -> 转移 logprob
- 共享前缀的序列会命中缓存，避免重复计算

### 6.5 引擎集成

树注意力适配三种引擎：

| 引擎     | 文件                         | 注意力实现                                | 集成方式                                                     |
| -------- | ---------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| FSDP     | `module_fsdp.py` (189行)     | `torch.compile(flex_attention)` 或 Triton | Monkey-Patch `_flash_attention_forward`                      |
| Megatron | `module_megatron.py` (218行) | `PytorchFlexAttention` 模块               | Context Manager 修补 `LLMBridge._get_transformer_layer_spec` |
| Archon   | `module_archon.py` (153行)   | `TreeAttentionWrapper`                    | `TreeAttentionMeta.from_trie()` 创建元数据                   |

**FSDP 补丁**（`module_fsdp.py` 第154行）：

```python
def patch_fsdp_for_tree_training(enable=True):
    flash_attention._flash_attention_forward = _tree_attn_fwd_func
```

**Megatron 补丁**（`module_megatron.py` 第170行）：

```python
with patch_bridge_for_tree_training(enable=True):
    # 修改 SelfAttention 的 attn_mask_type -> arbitrary
    # 替换 core_attention -> PytorchFlexAttention
    model = create_model()
```

______________________________________________________________________

## 7. 跨模块交互

```
                +----------------+
                |   Workflow     |  (调用 forward_backward_batch)
                +-------+--------+
                        |
          +-------------+-------------+
          |             |             |
   +------v------+ +---v---+ +------v-------+
   | FSDPEngine  | | MCore | | ArchonEngine |
   +------+------+ +---+---+ +------+-------+
          |             |             |
          v             v             v
   +-----------+  +-----------+ +------------+
   | module_   |  | module_   | | module_    |
   | fsdp.py   |  | megatron  | | archon.py  |
   +-----------+  +-----------+ +------------+
          |             |             |
          +------+------+------+------+
                 |             |
          +------v------+ +---v---------+
          | tree.py     | | functional  |
          | (Trie+Pack) | | (logprob)   |
          +------+------+ +-------------+
                 |
          +------v----------+
          | triton_kernel.py |
          | (GPU 内核)       |
          +-----------------+
```

**数据流**：

1. Workflow 构建 `MicroBatchList`（含 `trie_node`）
1. 引擎在 forward 前调用 `build_tree_attn_kwargs` 构建掩码
1. 注意力层使用 BlockMask 或 Triton 内核执行计算
1. Loss 计算通过 `functional.py` 的 `gather_packed_tree_logprobs` 还原序列级结果

**权重加载/保存流**：

1. 引擎初始化时调用 `registry.make_mcore_model` 创建模型
1. `hf_load.load_weights_from_hf_with_mbridge_fast` 多线程加载
1. 训练后 `hf_save.save_weights_to_hf_with_mbridge_fast` 并行保存

______________________________________________________________________

## 8. 设计决策与约束

### 8.1 设计决策

| 决策                                     | 理由                                                                                                        |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| All-to-All 而非 Ring Attention           | Ulysses All-to-All 通信量 O(N/P) 且实现简单，Ring 适合超长序列但实现复杂                                    |
| Monkey-Patch 而非子类化                  | HuggingFace Transformers 模型层级深，补丁方式侵入最小，且支持多模型复用                                     |
| 位打包掩码（int64）                      | 树掩码稀疏，N x N bool 矩阵太大，64 位打包节省 64 倍空间                                                    |
| Union-Find 权重分组                      | safetensors 文件间可能有交叉引用（QKV 融合），Union-Find 保证同组独立                                       |
| 块式掩码构建                             | `_ATTN_MASK_BLOCK_SIZE=2048`，限制 `tril_indices` 峰值内存约 32MB                                           |
| Stacked vs Flat MoE 保存自适应           | Qwen3-VL-MoE 的 stacked 格式需 EP-gather 后集中写入，自动检测避免手动配置                                   |
| Lightning Attention 自带 RotaryEmbedding | MLA 层的 `qk_pos_emb_head_dim` 与 Lightning 层的 `attn_head_dim * partial_rotary_factor` 不同，必须独立创建 |

### 8.2 约束与限制

| 约束                                                                     | 位置                            |
| ------------------------------------------------------------------------ | ------------------------------- |
| Ulysses SP 要求 `num_attention_heads % sp_size == 0`                     | `ulyssess_patch.py` 第165行     |
| 树注意力要求 `max_tokens_per_tree % lcm(BLOCK_SIZE, parallel_size) == 0` | `tree.py` 第341行               |
| PytorchFlexAttention 不支持 Context Parallel                             | `module_megatron.py` 第58行     |
| Triton 树注意力为实验性功能，需 `AREAL_USE_TRITON_TREE_ATTN=1` 环境变量  | `constants.py` 第12行           |
| Lightning Attention + CP 要求 `heads_per_tp % cp_size == 0`              | `bailing_moe.py` 第392行        |
| VPP 不支持 BailingMoe                                                    | `bailing_moe_bridge.py` 第216行 |
| 视觉 QKV 格式转换假设无 GQA（`num_heads == num_kv_heads`）               | `hf_load.py` 第103行            |
