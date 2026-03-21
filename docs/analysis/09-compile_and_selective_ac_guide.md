# torch.compile + Selective Activation Checkpointing

---

## 一、torch.compile 在训练中做什么

### 1.1 核心流程

```
你的 Python 模型代码
        │
        ▼  TorchDynamo (tracing)
FX Graph (算子级计算图)
        │
        ▼  AOTAutograd (forward + backward 联合图)
Joint Forward-Backward Graph
        │
        ▼  Min-Cut Partitioner (决定 save vs recompute)
Partitioned Graph (forward 部分 / backward 部分)
        │
        ▼  TorchInductor (代码生成)
Triton Kernels (GPU) / C++ (CPU)
        │
        ▼  可选: CUDA Graphs (消除 kernel launch 开销)
高效执行
```

### 1.2 三层优化

| 层 | 做什么 | 效果 |
|----|-------|------|
| **Operator Fusion** | 把多个小算子合成一个 kernel | 减少 GPU kernel launch 和显存读写 |
| **Min-Cut Recomputation** | 自动决定哪些激活存、哪些重算 | 同时改善速度和内存 |
| **CUDA Graphs** | 把整个计算图打包成一次 GPU 提交 | 消除 CPU-GPU 同步开销 |

### 1.3 使用方式

```python
# 最简单
model = torch.compile(model)

# 训练推荐
model = torch.compile(model, fullgraph=True)

# 极致性能
model = torch.compile(model, fullgraph=True, mode="max-autotune")
```

---

## 二、Graph Break：compile 的头号敌人

### 2.1 什么是 Graph Break

TorchDynamo 在 trace 时如果遇到无法编译的 Python 代码，就会**断开**当前图，产生多个小图。

```python
def forward(self, x):
    x = self.layer1(x)        # ─┐ graph 1
    print(x.shape)             # ─┘ graph break! (print 是 Python 副作用)
    x = self.layer2(x)        # ─┐ graph 2
    if x.sum() > 0:           # ─┘ graph break! (数据依赖的分支)
        x = self.layer3(x)    # ─┐ graph 3
    return x
```

### 2.2 Graph Break 为什么有害

每个 graph break：
- 打断 operator fusion（不能跨 break 融合 kernel）
- 打断 min-cut（不能跨 break 优化 save/recompute 决策）
- 打断 CUDA Graphs（每个子图单独提交）
- 增加 CPU-GPU 同步点

### 2.3 常见 Graph Break 原因及修复

| 原因 | 示例 | 修复 |
|------|------|------|
| `print` / logging | `print(x.shape)` | 删除或用 `torch.compiler.disable` 包裹 |
| 数据依赖分支 | `if x.sum() > 0` | 用 `torch.where` 或 `torch.cond` |
| 不支持的 Python 操作 | `next(iterator)` | 重写为静态索引 |
| 自定义 C++ 算子 | `my_custom_op(x)` | 用 `torch.library.custom_op` 注册 |
| 分布式通信 | `dist.all_reduce` | compile 区域排除通信（或用 compiled autograd） |

### 2.4 fullgraph=True 的意义

```python
model = torch.compile(model, fullgraph=True)  # 有 graph break 就报错
```

强制暴露所有 graph break → 逼你修好。torchtitan 的推荐做法：

```python
# torchtitan 中的用法
for layer in model.layers:
    layer = torch.compile(layer, fullgraph=True)
# 分布式通信在 layer 外面，不进入编译区域
```

---

## 三、Activation Checkpointing 回顾

### 3.1 三种策略对比

```
速度 ↑
  │
  │  ★ compile (min-cut)     ← 速度最好，内存也比 eager 好
  │
  │  ● eager (不 AC)         ← 基线：什么都存
  │
  │     ★ compile + SAC      ← 速度好 + 内存大幅节省
  │
  │  ▲ full AC               ← 内存最省，但最慢（全部重算）
  │
  └──────────────────────── 内存 →（越右越省）
```

### 3.2 Full AC (传统)

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        # 整个 block 的激活都不存，backward 时全部重算
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

**问题**：matmul 很贵（占 Transformer 计算量 >90%），全部重算太浪费。

---

## 四、Selective Activation Checkpointing (SAC)

### 4.1 核心思想

不是"全存"或"全重算"，而是**按算子类型选择**：

```
matmul (贵，计算密集)  →  SAVE (存住，不重算)
pointwise (便宜，内存带宽受限)  →  RECOMPUTE (不存，重算)
```

SAC 让你只重算**便宜的**算子（如 activation function、加法、norm），保留**昂贵的** matmul 结果。

### 4.2 策略函数 (policy_fn)

```python
import functools
from torch.utils.checkpoint import (
    checkpoint,
    create_selective_checkpoint_contexts,
    CheckpointPolicy,
)

def selective_policy(ctx, op, *args, **kwargs):
    # matmul 系列算子：保存结果，不重算
    if op in (torch.ops.aten.mm.default,
              torch.ops.aten.bmm.default,
              torch.ops.aten._scaled_dot_product_flash_attention.default):
        return CheckpointPolicy.MUST_SAVE
    # 其他算子：重算（便宜的 pointwise、norm 等）
    return CheckpointPolicy.PREFER_RECOMPUTE

context_fn = functools.partial(
    create_selective_checkpoint_contexts, selective_policy
)

# 在 checkpoint 中使用
output = checkpoint(
    block._forward, x,
    use_reentrant=False,
    context_fn=context_fn,
)
```

### 4.3 也可以用 op 列表（更简洁）

```python
# 直接列出要保存的算子
ops_to_save = [
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
]

context_fn = functools.partial(
    create_selective_checkpoint_contexts, ops_to_save
)
```

### 4.4 Memory Budget API（更新的自动方式）

```python
# 设置激活内存预算（0~1 之间的比例）
torch._functorch.config.activation_memory_budget = 0.5

# compile 会自动用 min-cut 在预算内找最优 save/recompute 组合
model = torch.compile(model, fullgraph=True)
```

---

## 五、torch.compile 的 Min-Cut 与 SAC 的关系

### 5.1 compile 自带的 min-cut

torch.compile 在 AOTAutograd 中自动运行 min-cut/max-flow 算法：

```
Forward-Backward 联合图
        │
        ▼  min-cut partitioner
把图切成 forward 子图 + backward 子图
切面上的 tensor = 必须保存的激活
切面以外的 = backward 时从保存的激活重算
```

**默认行为**：只重算便宜的（pointwise、fusible）ops，不重算 matmul → **同时改善速度和内存**。

### 5.2 compile + AC + SAC = 最佳组合

```python
# torchtitan 中的实际做法：

# 1. 对每个 TransformerBlock 应用 selective AC
for layer in model.layers:
    # SAC: 保存 matmul 结果，重算 norm/activation
    layer.setup_selective_ac(selective_policy)

# 2. torch.compile 每个 layer (fullgraph)
for layer in model.layers:
    layer = torch.compile(layer, fullgraph=True)

# 3. 分布式并行包裹在 compile 外面
for layer in model.layers:
    fully_shard(layer)  # FSDP hooks 不进入 compile 区域
```

**效果叠加**：
- SAC：宏观决策——哪些"区域"做 checkpoint
- compile min-cut：微观优化——checkpoint 区域内进一步优化 save/recompute
- Inductor：kernel fusion，把多个 pointwise 合成一个 kernel

---

## 六、对 MFU 的实际影响

torchtitan 在 Llama 3 8B/70B 上的实测（A100 80GB）：

| 配置 | MFU | 说明 |
|------|-----|------|
| eager + full AC | ~38% | 基线 |
| compile + full AC | ~48% | kernel fusion + CUDA graphs |
| **compile + selective AC** | **~54%** | 避免重算 matmul，最优 |
| compile + no AC | OOM | 激活放不下 |

Selective AC 比 full AC 快 ~15%，因为省去了大量不必要的 matmul 重算。

---

## 七、实践清单

### 7.1 让 compile 工作的前提

```
1. 消除所有 graph break (fullgraph=True)
   - 删除 print/assert
   - 数据依赖分支 → torch.where / torch.cond
   - 自定义 op → torch.library.custom_op

2. 保持 tensor shape 稳定
   - 固定 batch_size × seq_len（padding 到固定长度）
   - 否则每次 shape 变化都会触发重编译

3. 分布式通信放在 compile 区域外
   - FSDP hooks 自然在 module 边界
   - TP 的 all-reduce 通过 DTensor 自动处理
```

### 7.2 SAC 的选择标准

```
                  计算密度高？
                  /        \
                是          否
                /            \
         MUST_SAVE      PREFER_RECOMPUTE
         (matmul,       (pointwise, norm,
          FlashAttn)     dropout, activation)
```

**经验法则**：保存 matmul 类（`aten.mm`, `aten.bmm`, `scaled_dot_product_attention`），重算其余一切。

### 7.3 torchtitan 完整配置示例

```toml
# torchtitan config
[training]
compile = true

[activation_checkpoint]
mode = "selective"  # "full" | "selective" | "none"
selective_ac_option = "op"  # 按算子类型选择
```

---

## 八、compile 模式对比

| 模式 | 编译时间 | 运行速度 | 用途 |
|------|---------|---------|------|
| `default` | 中等 | 好 | 一般用途 |
| `reduce-overhead` | 较长 | 更好 | 启用 CUDA Graphs |
| `max-autotune` | 最长 | **最好** | 搜索最优 kernel 配置 |

训练推荐：`mode="max-autotune"` + `fullgraph=True`（第一个 step 慢，之后快）。

---

## 九、推荐资源

1. **PyTorch AC Techniques Blog** — SAC + Memory Budget 的官方详解
   https://pytorch.org/blog/activation-checkpointing-techniques/

2. **torch.compile Tutorial** — graph break 诊断与修复
   https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

3. **fullgraph=True Programming Model** — 消除 graph break 的完整指南
   https://docs.pytorch.org/docs/stable/compile/programming_model.fullgraph_true.html

4. **torch.utils.checkpoint API** — SAC policy_fn + CheckpointPolicy
   https://docs.pytorch.org/docs/stable/checkpoint.html

5. **Min-Cut Recomputation Discussion** — AOTAutograd min-cut 设计动机
   https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467

6. **ezyang: Ways to Use torch.compile** — 实用建议
   https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/

7. **torchtitan** — compile + SAC 的生产实践
   https://github.com/pytorch/torchtitan
