# Archon Compile 深度解析

> 源文件：`areal/experimental/models/archon/compile.py`（46 行）
> 扩展实现：`areal/experimental/models/archon/qwen3/infra/parallelize.py` `_apply_compile()`（531-629 行）
> 核心导出：`Compilable`（Protocol）· `apply_compile()`

---

[TOC]

---

# 1. 白话解释

## 1.1 一句话总结

`torch.compile` 是 PyTorch 的 **JIT 编译器**：把 Python 写的模型翻译成优化后的 GPU 机器码，通过算子融合、内存规划等手段让训练跑得更快——而 `compile.py` 的工作就是 **决定编译的粒度和时机**。

## 1.2 现实类比

```text
想象你开一家面包店，每天要做 1000 个面包：

方案 A：不编译（Eager 模式）
  → 面包师接到每个订单，一步一步手工操作
  → 预热烤箱 → 放面团 → 取出 → 刷蛋液 → 再放进去
  → 每一步都是独立动作，有大量等待时间

方案 B：torch.compile（Inductor 编译）
  ┌──────────────────────────────────────────────────────────┐
  │ 编译器观察面包师的工作流程，发现可以优化：                 │
  │                                                          │
  │ 融合操作：预热烤箱 + 放面团 合并为一步                    │
  │ 内存复用：上一批的烤盘直接用于下一批                      │
  │ 流水线：一批在烤的时候，同时准备下一批面团                │
  │                                                          │
  │ 结果：同样 1000 个面包，时间减少 20-30%                   │
  └────��─────────────────────────────────────────────────────┘

但是有一个限制：编译器需要「看到完整的工序」（fullgraph=True）
  → 如果中间突然有人来问价格（= Python 控制流打断），编译器就没法优化了
  → 这叫「图断裂」（graph break）

所以 AReaL 的策略是：
  → 只编译「做面包」的部分（TransformerBlock）
  → 不编译「接订单」和「收银」（Embedding/Output 层）
  → 这样既能优化又不会被打断
```

## 1.3 这个文件做了什么

```text
compile.py 在整个系统中的角色：

CLI (cli_args.py)                    配置 (archon_utils.py)
   enable_compile=True     →    force_pad_to_maximum()
                                  ↓ 保证静态形状
                              parallelize_qwen3()          parallelize_qwen2()
                                Step 1: TP                   Step 1: TP
                                Step 2: EP+TP                Step 2: CP
                                Step 3: CP                   Step 3: AC
                                Step 4: AC                 → Step 4: compile  ← 本文件
                              → Step 5: compile ← 本文件    Step 5: FSDP
                                Step 6: FSDP

实际做的事：
  Qwen2（Dense 模型）：
    for name, block in model.layers.items():
        model.layers[name] = torch.compile(block, backend="inductor", fullgraph=True)

  Qwen3（MoE 模型）：
    for block in model.layers:
        if block.moe_enabled:
            分别编译各子模块（跳过 experts 和 norms）
        else:
            整块编译

包装后：原始 module 被替换为 OptimizedModule
  layers.0 → OptimizedModule
               ._orig_mod → 原始 TransformerBlock
```

## 1.4 核心不变量

1. **compile 必须在 TP/EP/CP/AC 之后、FSDP 之前应用**
2. **始终使用 `backend="inductor"` + `fullgraph=True`**
3. **Dense 模型整块编译，MoE 模型子模块级编译**
4. **编译要求 `pad_to_maximum=True`（静态形状，避免重编译）**

---

# 2. 前置概念

## 2.1 torch.compile 的两阶段流水线

```text
Python 模型代码
      ↓
┌─── TorchDynamo（第一阶段）───┐
│ 拦截 Python 字节码执行         │
│ 捕获 PyTorch 操作序列          │
│ 构建 FX Graph（DAG 图）       │
│ 设置 Guard（输入约束）         │
└────────────────────────────┘
      ↓ FX Graph
┌─── TorchInductor（第二阶段）──┐
│ 算子融合（matmul + act → 1 kernel）│
│ 内存规划（减少中间分配）         │
│ 代码生成：                       │
│   - Triton 内核（自定义融合）     │
│   - cuBLAS/cuDNN（标准 GEMM）    │
│ 自动调优（block size, warps）    │
└─────────────────────────────┘
      ↓
优化后的 GPU 内核
```

**关键术语**：
- **FX Graph**：将模型前向传播表示为算子节点的有向无环图
- **Guard**：编译器对输入的假设（如 shape、dtype），如果违反则触发重编译
- **算子融合**：将多个小 kernel 合并为一个大 kernel，减少 GPU 启动开销和中间内存
- **Triton**：NVIDIA 提供的 GPU 编程语言，Inductor 用它生成融合 kernel

## 2.2 什么是 Graph Break

```text
Graph break = Dynamo 遇到无法表示为静态图的 Python 结构

常见原因：
  1. 数据依赖的控制流：if tensor.item() > 0:
  2. Python 副作用：print(tensor)
  3. 不支持的操作：某些 C++ 扩展
  4. 动态形状变化：tensor.tolist()

后果：
  ┌─────────────┐     ┌──────────┐     ┌─────────────┐
  │ 编译图 A     │ → │ Python   │ → │ 编译图 B     │
  │ (matmul+silu)│   │ 回退代码 │   │ (mm+add)     │
  └─────────────┘     └──────────┘     └─────────────┘
  → 图 A 和图 B 之间无法融合 → 丢失优化机会
  → 额外的 kernel 启动开销
  → 中间张量必须物化（不能被内存规划消除）

fullgraph=True 的含义：
  → 如果遇到 graph break，直接报错（而非静默降级）
  → 保证编译区域要么完全优化，要么构建失败
```

## 2.3 为什么按 Block 编译（而非整个模型）

```text
整模型编译的问题：
  Embedding 层 → 词表索引操作 → 容易 graph break
  TransformerBlock × N → 纯计算 → 适合编译
  Output 层 → 词表大小变化 → 容易 graph break

AReaL 的策略：
  ┌─────────────────────────────────────────────────┐
  │  model                                          │
  │  ├── tok_embeddings  ← 不编译（词表索引）        │
  │  ├── layers                                      │
  │  │   ├── 0: TransformerBlock  ← torch.compile ✓ │
  │  │   ├── 1: TransformerBlock  ← torch.compile ✓ │
  │  │   ├── ...                                     │
  │  │   └── N: TransformerBlock  ← torch.compile ✓ │
  │  ├── norm             ← 不编译                   │
  │  └── output           ← 不编译（词表投影）       │
  └─────────────────────────────────────────────────┘

额外好处：
  所有 TransformerBlock 结构相同 → Dynamo 可以复用编译结果
  （第 1 个 block 编译后，后续 block 检查 Guard 通过即可复用）
```

## 2.4 compile 在并行化流水线中的位置及原因

```text
Qwen3 的 6 步流水线 (parallelize.py:100-106 注释，实现在 132-187 行):

  Step 1: TP   → 分片权重，注册 DTensor hooks
  Step 2: EP   → 分片专家权重，注册 dispatch/combine hooks
  Step 3: CP   → 配置 Ulysses 注意力组
  Step 4: AC   → 用 CheckpointWrapper 包装
  ─── compile 必须在这里 ───
  Step 5: compile → torch.compile 编译 TransformerBlock
  ─── FSDP 必须在最外层 ───
  Step 6: FSDP  → fully_shard 分片参数

Qwen2 的 5 步流水线 (qwen2 parallelize.py:85-90):
  Step 1: TP → Step 2: CP → Step 3: AC → Step 4: compile → Step 5: FSDP
```

**为什么 compile 在 AC 之后？**
- AC 用 `CheckpointWrapper` 包装了 TransformerBlock
- 如果先编译再包装 AC，编译图中不包含 checkpoint 边界
- Dynamo 能识别 `CheckpointWrapper` 的高阶操作语义，保留 checkpoint 行为
- `memory_budget` AC 模式依赖 Inductor 的内存规划来决定保留/重算策略

**为什么 compile 在 FSDP 之前？**
- FSDP 的 `fully_shard` 安装 pre-forward/post-forward Python hooks
- 这些 hooks 包含分布式通信、进程组状态、条件判断
- 如果在 FSDP 之后编译，Dynamo 会在 tracing 时遇到这些 hooks → graph break
- 先编译再 FSDP：编译图只捕获纯计算，FSDP hooks 在编译区域外层 eager 执行

## 2.5 compile 对 state_dict 的影响

```python
# 编译前：
model.layers["0"]  →  TransformerBlock

# 编译后：
model.layers["0"]  →  OptimizedModule
                        ._orig_mod → TransformerBlock

# state_dict key 变化：
# 编译前: layers.0.attention.wq.weight
# 编译后: layers.0._orig_mod.attention.wq.weight

# 如果同时有 AC + compile（compile 在外层，AC 在内层）：
# layers.0._orig_mod._checkpoint_wrapped_module.attention.wq.weight

# state_dict_adapter.py 的修复：
name = name.replace("._checkpoint_wrapped_module", "")  # 去掉 AC 前缀
name = name.replace("._orig_mod", "")                   # 去掉 compile 前缀
# → layers.0.attention.wq.weight  （恢复为 HuggingFace 格式）
```

Qwen2 的处理在 `qwen2/model/state_dict_adapter.py:91-95`，Qwen3 在 `qwen3/model/state_dict_adapter.py:217-221`。

## 2.6 compile 与 pad_to_maximum 的关系

```text
问题：torch.compile 生成的 kernel 绑定了特定的 tensor shape
     如果每个 batch 的 seq_len 不同，会触发重编译（非常慢）

解决：compile 开启时，强制 pad_to_maximum=True (archon_utils.py:265-271)
     所有 batch 填充到最大长度 → shape 固定 → 无需重编译

代价：处理填充 token 的额外计算
收益：避免重编译的巨大开销

force_pad_to_maximum() 函数 (archon_utils.py:255-271):
  if enable_compile and not config.pad_to_maximum:
      config.pad_to_maximum = True
```

## 2.7 compile 与 zero-bubble PP 的兼容性

```text
Zero-bubble 流水线调度使用 split backward + retain_graph=True
→ 与 Inductor 的 donated_buffer 优化冲突
  （donated_buffer 假设反向后 buffer 不会被重用，但 retain_graph=True 需要重用）

自动降级策略 (archon_utils.py:155-178):
  if 使用 zero-bubble 调度:
      logger.warning("... incompatible with torch.compile. Disabling.")
      enable_compile = False
```

## 2.8 compile 与确定性训练

```text
setup_deterministic_mode() (archon_utils.py:211-247):
  当确定性模式开启 且 compile 活跃时：
    设置 TORCH_COMPILE_DETERMINISTIC=1
    → 强制 Inductor 选择确定性 kernel
    → 避免非确定性 cuDNN 算法和 atomics 归约
```

---

# 3. 源码逐行地图

## 3.1 文件结构总览

```text
compile.py (46 行) — 基础实现（Dense 模型）
├── import 区域 (1-8)
├── _get_logger() (11-15)        ← rank-aware 日志
├── Compilable (18-21)           ← Protocol 类型约束
└── apply_compile() (24-45)      ← 主入口 ★

qwen3/infra/parallelize.py _apply_compile() (531-629) — 扩展实现（MoE 模型）
├── Dynamo 配置 (544-549)         ← capture_scalar_outputs, LRU cache
├── 遍历 layers (551-599)         ← MoE/非MoE 分支编译
├── GroupedExperts mm 编译 (601-625) ← 模块级函数编译 + EP 动态形状
└── 日志 (627-629)
```

## 3.2 `compile.py` 完整源码解析

### 3.2.1 `_get_logger()`（11-15 行）

```python
@functools.cache  # 同一进程只创建一次
def _get_logger() -> logging.Logger:
    """Get rank-aware logger for this module."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    return logging.getLogger(f"[Archon Compile Rank {rank}]")
```

与其他 Archon 模块（activation_checkpoint, expert_parallel）使用相同的 rank-aware 日志模式。

### 3.2.2 `Compilable` Protocol（18-21 行）

```python
class Compilable(Protocol):
    """Protocol for models that can be compiled with apply_compile."""
    layers: nn.ModuleDict
```

**设计理念**：
- 使用 `Protocol`（结构化类型）而非继承，遵循鸭子类型原则
- 任何具有 `layers: nn.ModuleDict` 属性的模型都可以被编译
- 不要求模型继承特定基类，更灵活
- `apply_ac()` 中使用 `hasattr(model, "layers")` 做运行时检查，而 `apply_compile()` 使用 `Protocol` 做静态类型检查——两种方式达到相同目的

### 3.2.3 `apply_compile()`（24-45 行）★

```python
def apply_compile(model: Compilable) -> None:
    """Apply torch.compile to each TransformerBlock.

    Compiling per-block is more efficient than whole model due to:
    1. Repeated structure allows compilation reuse
    2. Avoids graph breaks from embedding/output layers

    Must be called AFTER TP and AC, BEFORE FSDP.

    Args:
        model: The model to compile. Must have a `layers` attribute (ModuleDict).
    """
    for name, block in model.layers.items():
        model.layers[name] = torch.compile(
            block,
            backend="inductor",    # TorchInductor 后端
            fullgraph=True,        # 不允许 graph break
        )

    _get_logger().info(
        f"Compiled {len(model.layers)} TransformerBlocks with torch.compile"
    )
```

**逐行解读**：
1. 遍历 `model.layers`（ModuleDict），`name` 是层 ID（如 `"0"`, `"1"` ...）
2. `torch.compile(block, ...)` 返回 `OptimizedModule`，包装了原始 block
3. 直接替换 `model.layers[name]`，原始 block 存储在 `OptimizedModule._orig_mod` 中
4. `backend="inductor"`：使用 TorchInductor 编译后端
5. `fullgraph=True`：要求整个 block 的 forward 必须被完整捕获为一个图，否则报错

**调用位置**：
- Qwen2：`qwen2/infra/parallelize.py:133-134`
  ```python
  if enable_compile:
      apply_compile(model)
  ```

## 3.3 Qwen3 `_apply_compile()` 扩展实现（531-629 行）★★★

Qwen3 没有使用 `compile.py` 的 `apply_compile`，而是导入 `Compilable` Protocol 后自己实现了 MoE 感知的编译逻辑。

### 3.3.1 函数签名（531-543 行）

```python
def _apply_compile(model: Compilable, ep_enabled: bool = False) -> None:
    """Apply torch.compile to Qwen3 model (MoE-aware).

    For MoE layers, compile submodules separately to avoid graph breaks
    from FSDP(GroupedExperts). For non-MoE layers, compile the whole block.

    Must be called AFTER TP and AC, BEFORE FSDP.

    Args:
        model: The model to compile.
        ep_enabled: Whether Expert Parallelism is enabled. If True, marks
            dynamic shapes for varying token counts per expert.
    """
```

### 3.3.2 Dynamo 全局配置（544-549 行）

```python
# token-choice MoE 中动态形状需要此标志
torch._dynamo.config.capture_scalar_outputs = True

# PyTorch issue #166926 的临时修复
if hasattr(torch._C._dynamo.eval_frame, "_set_lru_cache"):
    torch._C._dynamo.eval_frame._set_lru_cache(False)
```

**`capture_scalar_outputs = True`**：
- MoE 的 token-choice 路由中，分派给每个专家的 token 数量是数据依赖的标量
- 提取标量值（如 `tensor.tolist()` 用于 All-to-All split sizes）默认会 graph break
- 此标志告诉 Dynamo 将标量提取也纳入图中

**`_set_lru_cache(False)`**：
- Dynamo 的帧评估有 LRU 缓存机制
- 存在 guard 失效 bug（PyTorch #166926），禁用缓存规避问题

### 3.3.3 遍历 layers：MoE vs 非 MoE 分支（551-599 行）

```python
for name, block in model.layers.items():
    if getattr(block, "moe_enabled", False):
        # ─── MoE 层：子模块级编译 ───
        # 原因：FSDP(GroupedExperts) 内部使用 torch._dynamo.disable
        #       导致整块编译时产生 graph break

        # 1. 如果有 CheckpointWrapper，解包获取内层 block
        if isinstance(block, CheckpointWrapper):
            inner_block = block._checkpoint_wrapped_module
        else:
            inner_block = block

        # 2. 遍历内层 block 的子模块
        for attr_name, submod in inner_block.named_children():
            # 验证通过 wrapper 和直接访问得到同一对象
            assert getattr(block, attr_name) == getattr(inner_block, attr_name)

            if isinstance(submod, moe_module.MoE):
                # MoE 子模块：再进一步拆分
                for moe_attr, moe_submod in submod.named_children():
                    if moe_attr == "experts":
                        # 跳过 experts（B200 硬件问题）
                        # https://github.com/pytorch/torchtitan/issues/1940
                        continue
                    setattr(
                        submod, moe_attr,
                        torch.compile(moe_submod, backend="inductor", fullgraph=True),
                    )

            elif attr_name in ("attention_norm", "ffn_norm"):
                # 跳过 norms：SequenceParallel 的 async redistribute
                # 在 forward 中产生 AsyncCollectiveTensor，
                # backward 期望 local tensor → graph break
                continue

            else:
                # 其他子模块（attention, feed_forward 等）：正常编译
                setattr(
                    inner_block, attr_name,
                    torch.compile(submod, backend="inductor", fullgraph=True),
                )

    else:
        # ─── 非 MoE 层：整块编译 ───
        model.layers[name] = torch.compile(
            block, backend="inductor", fullgraph=True,
        )
```

**MoE 层编译决策树**：

```text
TransformerBlock (moe_enabled=True)
├── attention           → torch.compile ✓
├── attention_norm      → 跳过 ✗ (SequenceParallel async)
├── feed_forward (MoE)
│   ├── router          → torch.compile ✓
│   ├── reorderer       → torch.compile ✓
│   ├── experts         → 跳过 ✗ (B200 硬件问题)
│   └── shared_experts  → torch.compile ✓
├── ffn_norm            → 跳过 ✗ (SequenceParallel async)
└── ...其他子模块       → torch.compile ✓
```

### 3.3.4 GroupedExperts mm 函数编译（601-625 行）

```python
# 检查是否已经被 patch 过（避免重复）
already_patched = (
    "_run_experts_grouped_mm_dynamic"
    in grouped_experts._run_experts_grouped_mm.__qualname__
)

if not already_patched:
    # 编译 _run_experts_grouped_mm 模块级函数
    grouped_experts._run_experts_grouped_mm = torch.compile(
        grouped_experts._run_experts_grouped_mm,
        backend="inductor",
        fullgraph=True,
    )

    # 如果 EP 开启，需要处理动态形状
    if ep_enabled:
        compiled_fn = grouped_experts._run_experts_grouped_mm

        def _run_experts_grouped_mm_dynamic(
            w1: torch.Tensor,
            w2: torch.Tensor,
            w3: torch.Tensor,
            x: torch.Tensor,
            num_tokens_per_expert: torch.Tensor,
        ) -> torch.Tensor:
            torch._dynamo.mark_dynamic(x, 0)  # 标记 dim 0 为动态
            return compiled_fn(w1, w2, w3, x, num_tokens_per_expert)

        grouped_experts._run_experts_grouped_mm = _run_experts_grouped_mm_dynamic
```

**为什么需要 `mark_dynamic(x, 0)`？**
- Expert Parallelism 通过 All-to-All 重分布 token
- 每个 rank 收到的 token 数量随 batch 变化（数据依赖）
- `x` 的 dim 0 是 token 数量 → 动态维度
- `mark_dynamic` 告诉 Dynamo 对此维度生成灵活的 guard
- 没有这个标记 → 每次 token 数量变化都触发重编译

**为什么要检查 `already_patched`？**
- `_run_experts_grouped_mm` 是模块级函数（不是实例方法）
- 如果 Pipeline Parallelism 多次调用 `_apply_compile`（对不同 stage），这个函数只应该被编译一次

### 3.3.5 调用位置

```python
# qwen3/infra/parallelize.py:164-167
if enable_compile:
    ep_enabled = parallel_dims.ep > 1
    _apply_compile(model, ep_enabled=ep_enabled)
```

## 3.4 配置入口

### 3.4.1 CLI 参数（cli_args.py:438-442）

```python
# ArchonEngineConfig 中的字段
enable_compile: bool = field(
    default=True,
    metadata={"help": "Enable torch.compile for TransformerBlocks."},
)
```

默认开启。

### 3.4.2 配置流转

```text
cli_args.py                     archon_utils.py                    archon_engine.py
  enable_compile: bool    →   prepare_training_config()        →  ac_config, enable_compile
                              ├── validate_zero_bubble_compat()     ↓
                              │   → 可能禁用 compile            _setup_parallelism()
                              ├── force_pad_to_maximum()             ↓
                              │   → 强制静态形状              parallelize_qwen3()
                              └── setup_deterministic_mode()         ↓
                                  → 设置 TORCH_COMPILE_DETERMINISTIC  Step 5: compile
```

---

# 4. 验证方法

## 4.1 理解检验题

### 题 1：编译粒度
> 为什么 AReaL 按 TransformerBlock 编译，而不是编译整个模型？

**答案**：两个原因：
1. Embedding 层和 Output 层包含词表索引操作，容易导致 graph break
2. 所有 TransformerBlock 结构相同，Dynamo 可以复用第一个 block 的编译结果（Guard 检查通过即可），减少总编译时间

### 题 2：MoE 层为什么不能整块编译
> Qwen3 的 MoE 层为什么需要子模块级编译？跳过了哪些子模块？

**答案**：
- `FSDP(GroupedExperts)` 内部使用 `torch._dynamo.disable`，整块编译会 graph break
- 跳过的子模块：
  1. `experts`（GroupedExperts）：B200 硬件问题（torchtitan #1940）
  2. `attention_norm` / `ffn_norm`：SequenceParallel 的 async redistribute 在 forward 产生 AsyncCollectiveTensor，backward 期望 local tensor，Dynamo 无法静态捕获

### 题 3：顺序依赖
> 如果把 compile 移到 FSDP 之后会怎样？

**答案**：FSDP 的 `fully_shard` 安装了 Python hooks（管理 all-gather/reshard），Dynamo tracing 时会遇到这些 hooks 包含的分布式通信和条件判断 → graph break。`fullgraph=True` 下会直接报错。

### 题 4：动态形状
> `torch._dynamo.mark_dynamic(x, 0)` 在 EP 场景中解决了什么问题？

**答案**：EP 的 All-to-All 通信使每个 rank 收到的 token 数量随 batch 变化（数据依赖）。没有 `mark_dynamic`，每次 x 的 dim 0 变化都会触发 Dynamo 重编译（极慢）。标记后，Dynamo 生成灵活的 guard 允许该维度变化而不重编译。

### 题 5：pad_to_maximum
> 为什么 compile 开启时必须 `pad_to_maximum=True`？

**答案**：torch.compile 生成的 kernel 绑定特定 tensor shape。如果 seq_len 随 batch 变化，Dynamo 的 shape guard 会失败 → 触发重编译。`pad_to_maximum=True` 保证所有 batch 填充到相同长度 → shape 固定 → 无需重编译。代价是处理填充 token 的额外计算。

## 4.2 运行测试

```bash
# 需要 GPU 的测试
# 先检查 GPU
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# torch.compile 确定性测试
uv run pytest tests/test_cuda_deterministic.py -v -k "compile"

# Qwen3 并行化测试（包含 compile 验证）
# 注意：需要多 GPU 环境
uv run pytest tests/experimental/archon/test_qwen3_parallelize.py -v
```

## 4.3 交互式验证

### 验证 1：观察 OptimizedModule 包装

```python
import torch
import torch.nn as nn
from areal.experimental.models.archon.compile import Compilable, apply_compile

class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
    def forward(self, x):
        return self.linear(x)

class DummyModel:
    def __init__(self):
        self.layers = nn.ModuleDict({str(i): DummyBlock() for i in range(3)})

model = DummyModel()
print("编译前:", type(model.layers["0"]))
# → <class '__main__.DummyBlock'>

apply_compile(model)
print("编译后:", type(model.layers["0"]))
# → <class 'torch._dynamo.eval_frame.OptimizedModule'>

# 原始 module 存在 _orig_mod 中：
print("原始 module:", type(model.layers["0"]._orig_mod))
# → <class '__main__.DummyBlock'>
```

### 验证 2：观察 state_dict key 变化

```python
# 编译后的 state_dict keys 包含 _orig_mod
for k in model.layers.state_dict().keys():
    print(k)
# 0._orig_mod.linear.weight
# 0._orig_mod.linear.bias
# 1._orig_mod.linear.weight
# ...
```

### 验证 3：fullgraph=True 的 graph break 检测

```python
class BadBlock(nn.Module):
    def forward(self, x):
        if x.sum().item() > 0:  # 数据依赖的控制流 → graph break
            return x * 2
        return x

# fullgraph=True 会报错：
try:
    compiled = torch.compile(BadBlock(), backend="inductor", fullgraph=True)
    compiled(torch.randn(4))
except Exception as e:
    print(f"预期的 graph break 错误: {e}")
```

## 4.4 常见误区

| 误区 | 正解 |
|-----|------|
| "torch.compile 改变了模型的输出" | compile 是语义保持的优化，输出应完全一致（确定性模式下） |
| "整个模型都被编译了" | 只编译 TransformerBlock，Embedding/Output 层不编译 |
| "compile 和 FSDP 不兼容" | 兼容，但必须先 compile 再 FSDP |
| "MoE 层的 experts 被编译了" | Qwen3 中 experts 被跳过（B200 硬件问题） |
| "compile 很慢" | 首次编译慢，之后 Guard 通过则复用（静态形状下不会重编译） |
| "不用 compile 也能 memory_budget AC" | memory_budget AC 依赖 Inductor，必须开启 compile |

---

# 5. 附录

## 5.1 Dense vs MoE 编译策略对比

| 维度 | Dense（Qwen2） | MoE（Qwen3） |
|------|----------------|---------------|
| 实现位置 | `compile.py:24-45` | `qwen3/parallelize.py:531-629` |
| 编译粒度 | 整块 TransformerBlock | 子模块级分别编译 |
| graph break 处理 | 无（Dense 层不会 break） | 跳过 experts/norms |
| Dynamo 配置 | 无额外配置 | `capture_scalar_outputs=True`, `_set_lru_cache(False)` |
| EP 动态形状 | 不涉及 | `mark_dynamic(x, 0)` |
| GroupedExperts mm | 不涉及 | 模块级函数单独编译 |
| 导入依赖 | `from .compile import apply_compile` | `from .compile import Compilable`（只用 Protocol） |

## 5.2 文件依赖关系

```text
compile.py
  导入:
  ├── torch (torch.compile)
  ├── torch.distributed (dist.get_rank)
  ├── torch.nn (nn.ModuleDict)
  └── areal.utils.logging → getLogger

  被导入:
  ├── qwen2/infra/parallelize.py → apply_compile (函数)
  ├── qwen3/infra/parallelize.py → Compilable (Protocol)
  └── (qwen3 自己实现了 _apply_compile，不用 apply_compile)

相关文件（不直接导入但紧密关联）:
  ├── archon_utils.py
  │   ├── validate_zero_bubble_compatibility() → 可能禁用 compile
  │   ├── force_pad_to_maximum() → compile 需要静态形状
  │   └── setup_deterministic_mode() → 设置 TORCH_COMPILE_DETERMINISTIC
  ├── archon_engine.py
  │   └── _setup_parallelism() → 传递 enable_compile
  ├── qwen2/model/state_dict_adapter.py:91-95 → 去掉 _orig_mod 前缀
  └── qwen3/model/state_dict_adapter.py:217-221 → 去掉 _orig_mod 前缀
```

## 5.3 compile 相关环境变量和 torch 配置汇总

| 配置项 | 设置位置 | 作用 |
|-------|---------|------|
| `torch._dynamo.config.capture_scalar_outputs` | `_apply_compile()` L545 | 允许 Dynamo 捕获标量输出，避免 MoE 路由中 graph break |
| `torch._C._dynamo.eval_frame._set_lru_cache(False)` | `_apply_compile()` L548-549 | 修复 guard 失效 bug (#166926) |
| `torch._dynamo.mark_dynamic(x, 0)` | `_apply_compile()` L622 | EP 场景标记 token 维度为动态 |
| `TORCH_COMPILE_DETERMINISTIC=1` | `setup_deterministic_mode()` L242-247 | 强制使用确定性 kernel |
| `torch._functorch.config.activation_memory_budget` | `apply_ac()` L277 | memory_budget AC（依赖 compile） |
| `pad_to_maximum=True` | `force_pad_to_maximum()` L265-271 | 保证静态形状避免重编译 |

## 5.4 已知限制和临时修复追踪

| 问题 | 临时修复 | 追踪 |
|-----|---------|------|
| B200 硬件上 experts 编译失败 | 跳过 experts 编译 | torchtitan #1940 |
| Dynamo LRU cache guard 失效 | `_set_lru_cache(False)` | PyTorch #166926 |
| SequenceParallel async + Inductor 不兼容 | 跳过 norms 编译 | 待 PyTorch 升级 |
| SAC + compile 区域交互 | `inductor_compiled_code` HOP 被注释掉 | 待 PyTorch 升级后启用 |

## 5.5 验证清单

- [ ] 理解 torch.compile 的两阶段流水线（Dynamo + Inductor）
- [ ] 理解 graph break 的概念及其对性能的影响
- [ ] 理解为什么按 TransformerBlock 编译而非整模型
- [ ] 理解 Dense vs MoE 模型的编译策略差异
- [ ] 理解 MoE 层中哪些子模块被跳过及原因
- [ ] 理解 compile 在并行化流水线 Step 5 的位置约束
- [ ] 理解 `fullgraph=True` 的含义和保证
- [ ] 理解 `pad_to_maximum` 与 compile 的关系
- [ ] 理解 `mark_dynamic` 在 EP 场景的作用
- [ ] 理解 OptimizedModule 对 state_dict key 的影响
- [ ] 理解 zero-bubble PP 与 compile 的不兼容性
