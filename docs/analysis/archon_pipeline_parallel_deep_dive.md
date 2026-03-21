# Archon Pipeline Parallel 深度解析

> 源文件：`areal/experimental/models/archon/pipeline_parallel.py`（493 行）
> 核心函数：`pipeline_llm` · `generate_llm_fqn_per_model_part` · `pipeline_module_split` · `build_pipeline_schedule`

---

[TOC]

---

# 1. 白话解释

## 1.1 一句话总结

这个文件把一个完整的 Transformer 模型**按层切成多段**，分配到不同 GPU 上，让它们像工厂流水线一样依次加工数据——前一个 GPU 算完把结果传给后一个 GPU，多个"微批次"同时在不同阶段流动，从而突破单 GPU 显存限制。

## 1.2 流水线并行的现实类比

```text
想象一条汽车组装线，32 道工序（= 32 层 Transformer）：

方案 A：一个工人从头做到尾
  → 太慢，一个人记不住所有工序（= 一张 GPU 放不下所有权重）

方案 B：4 个工人各负责 8 道工序（= Pipeline Parallel, pp=4）
  → 工人 0 做完 1-8 道，把半成品传给工人 1 做 9-16 道…
  → 问题：工人 3 要等工人 0/1/2 都做完才能开始 → 流水线气泡

方案 C：每人负责两段不相邻的工序（= Interleaved 1F1B, 虚拟 stage）
  → 工人 0 做 1-4 道 和 17-20 道
  → 每段更短，半成品更快到达下一个工人 → 气泡减半

方案 D：每人负责"首 + 尾"工序（= ZBV 零气泡）
  → 工人 0 做 1-4 道 和 29-32 道
  → 一个工人可以同时做"第一段的正向"和"最后一段的反向" → 气泡趋近于零
```

## 1.3 这个文件做了什么

```text
pipeline_llm()                     ← 总入口
  │
  ├── generate_llm_fqn_per_model_part()   ← 决定每个 stage 分到哪些层
  │     "32 层，8 个 stage → stage 0 拿 [tok_embeddings, layers.0-3]"
  │
  ├── pipeline_module_split()             ← 把模型真正切开
  │     ├── _get_stage_indices()          ← 决定当前 Rank 负责哪些 stage
  │     └── _build_stage_from_modules()   ← deep copy → 删除不属于本 stage 的层
  │
  ├── parallelize_fn()                    ← 对每个 stage 再应用 TP/FSDP
  │
  └── return (stages, model_parts, has_first, has_last)

build_pipeline_schedule()          ← 创建调度器，控制微批次的执行顺序
```

## 1.4 核心不变量

```
所有 transformer 层 [0, 1, ..., num_layers-1] 恰好出现在一个 stage 中，不重复、不遗漏。
```

---

# 2. 前置概念

## 2.1 流水线气泡（Pipeline Bubble）

在基本的 1F1B（一次正向 + 一次反向）调度中，假设有 `p` 个 stage 和 `m` 个微批次：

```text
                     时间 →
Stage 0:  F₀  F₁  F₂  F₃  B₃  B₂  B₁  B₀
Stage 1:      F₀  F₁  F₂  F₃  B₃  B₂  B₁  B₀
Stage 2:          F₀  F₁  F₂  F₃  B₃  B₂  B₁  B₀
Stage 3:              F₀  F₁  F₂  F₃  B₃  B₂  B₁  B₀
                  ↑
              这些空白区域就是"气泡"——GPU 在等待上游数据

气泡比例 ≈ (p - 1) / m
```

**减少气泡的方法**：增加 `m`（更多微批次）或减少每个 stage 的工作量（虚拟 stage）。

## 2.2 虚拟 Stage（Virtual Stages）

**物理 Stage**：每个 GPU 只负责一段模型 → `num_stages = pp_degree`
**虚拟 Stage**：每个 GPU 负责多段不连续的模型 → `num_stages = pp_degree × stages_per_rank`

```text
4 个 GPU, 8 个虚拟 stage:

Loop 风格 (Interleaved1F1B):
  Rank 0 → stages (0, 4)    模型前 1/8 + 中间 1/8
  Rank 1 → stages (1, 5)
  Rank 2 → stages (2, 6)
  Rank 3 → stages (3, 7)

V 风格 (ZBVZeroBubble):
  Rank 0 → stages (0, 7)    模型最前 + 最后
  Rank 1 → stages (1, 6)
  Rank 2 → stages (2, 5)
  Rank 3 → stages (3, 4)
```

**为什么 V 风格能实现零气泡？**

V 风格把首尾 stage 放在同一个 GPU 上。数据正向流过 0→1→2→3→4→5→6→7，当 Rank 0 完成 stage 7 的反向传播后，可以立刻开始下一个微批次在 stage 0 的正向传播——两个 stage 在流水线的两端，计算依赖最大限度错开，GPU 几乎没有空闲时间。

## 2.3 负载均衡算法

模型不只有 Transformer 层——还有 Embedding 和 Output Head。如果不考虑它们的计算成本，首尾 stage 会过载：

```text
不做均衡:
  Stage 0: [tok_embeddings + 8 层]  ← 过载！比中间多 embedding 的开销
  Stage 3: [8 层 + norm + output]   ← 过载！

做均衡 (first/last_stage_less_layers=1):
  把 embedding 计算量等价于 1 层，output+norm 等价于 1 层
  有效总层数 = 32 + 1 + 1 = 34
  均匀分配后，首 stage 少分 1 层、尾 stage 少分 1 层
  → 各 stage 实际计算量大致相等
```

## 2.4 ModuleDict vs ModuleList

Archon 用 `ModuleDict` 而非 `ModuleList` 存 Transformer 层：

```python
self.layers = nn.ModuleDict()
for i in range(32):
    self.layers[str(i)] = TransformerBlock(i, args)
```

**关键优势**：流水线切分时只需 `del module_dict["5"]`，剩余层的 key 不变（仍是 `"0"`, `"1"`, `"4"`, ...）。如果用 `ModuleList`，删除后自动重新编号（`"0"`, `"1"`, `"2"`, ...），破坏与 checkpoint 的 FQN 映射。

## 2.5 Meta Device 与 Deep Copy

```python
# 模型在 meta device 上创建（只有 shape 信息，没有实际数据）
with torch.device("meta"):
    model = Qwen2Model(args)

# deep copy meta 模型 ≈ 复制结构信息，几乎不占显存
model_copy = copy.deepcopy(model)  # 快速且免费
```

**为什么要求 meta device？** 对于 70B 参数模型，每次 deep copy 的实际张量会消耗 ~140GB。但 meta 张量只有 shape/dtype 元数据，deep copy 近乎零成本。代码在第 232 行 assert 了这个前提条件。

## 2.6 PipelineStage 与 Schedule

```text
PipelineStage = "做什么"          Schedule = "什么时候做"
┌──────────────────────┐        ┌──────────────────────────────┐
│ 包裹一个 nn.Module    │        │ 控制微批次执行顺序            │
│ 知道自己是第几个 stage │  ←───  │ 决定何时 forward / backward   │
│ 管理 Send/Recv 通信   │        │ 管理跨 stage 激活值传递       │
└──────────────────────┘        └──────────────────────────────┘

Schedule 类型:
  PipelineScheduleSingle (1F1B)
    → 接收单个 stage: schedule_class(stages[0], ...)
  PipelineScheduleMulti (Interleaved/ZeroBubble)
    → 接收 stage 列表: schedule_class(stages, ...)
```

## 2.7 调度策略矩阵

| 调度策略 | Stage/Rank | 分配风格 | 气泡 | 通信开销 |
| --- | --- | --- | --- | --- |
| `1F1B` | 1 | — | `(p-1)/m` | 低（仅相邻通信）|
| `Interleaved1F1B` | ≥2 | Loop | `(p-1)/(v·m)` | 中（跨 rank 通信）|
| `InterleavedZeroBubble` | ≥2 | Loop | 接近零 | 中 |
| `ZBVZeroBubble` | 2 | V | 接近零 | 中 |
| `ScheduleDualPipeV` | 2 | V | 接近零 | 中（代码内部支持，非 CLI 可选项）|

## 2.8 scale_grads=False 的含义

```python
schedule_class(..., scale_grads=False)
```

当 `scale_grads=True` 时，schedule 自动将每个微批次的梯度除以 `n_microbatches`。Archon 设为 `False` 的原因：

1. **RL 训练的 loss 自有归一化逻辑**——GRPO/PPO/DAPO 的 loss 已处理 batch 大小，PP 再除一次会双重归一化。
2. **与 FSDP 解耦**——FSDP 有自己的梯度规约语义，PP 不应干涉。
3. **灵活性**——不同微批次可能有不同有效 batch size（变长序列 packing），均匀除法语义上不正确。

---

# 3. 源码逐行地图

## 3.1 导入与日志（1-43 行）

```python
# 第 1 行: 源自 TorchTitan 项目
# Adapted from torchtitan: torchtitan/distributed/pipeline_parallel.py

# 第 3-6 行: 标准库
import copy           # deep copy 用于模型分割
import functools      # cache 装饰器用于日志单例
import math           # ceil 用于虚拟 stage 数计算
from collections.abc import Callable  # loss_fn 类型注解

# 第 9-20 行: PyTorch 分布式 PP API
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,       # 多 stage 调度基类
    PipelineScheduleSingle,      # 单 stage 调度基类 (1F1B)
    ScheduleDualPipeV,           # V 风格双管道
    ScheduleZBVZeroBubble,       # V 风格零气泡
    get_schedule_class,          # 字符串 → 调度类的工厂函数
)

# 第 24-28 行: TYPE_CHECKING 避免循环导入
if TYPE_CHECKING:
    from torch.distributed.pipelining.schedules import _PipelineSchedule

    from areal.api.cli_args import ArchonEngineConfig
    from areal.experimental.models.archon import ArchonParallelDims

# 第 31-35 行: Rank 感知日志
@functools.cache
def _get_logger() -> logging.Logger:
    rank = dist.get_rank() if dist.is_initialized() else 0
    return logging.getLogger(f"[Archon PipelineParallel Rank {rank}]")

# 第 38-43 行: 公开 API
__all__ = [
    "generate_llm_fqn_per_model_part",
    "pipeline_module_split",
    "pipeline_llm",
    "build_pipeline_schedule",
]
```

## 3.2 `build_pipeline_schedule`（46-80 行）

```python
def build_pipeline_schedule(
    stages: list[PipelineStage],     # 本 rank 的 stage 列表
    pp_schedule: str,                 # 调度名称（"1F1B", "Interleaved1F1B" 等）
    n_microbatches: int,              # 微批次数量
    pp_degree: int = 1,              # PP 并行度（用于气泡警告）
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                                      # Loss 函数（接收 logits 和 target，返回标量 loss；eval 模式传 None）
) -> "_PipelineSchedule":

    # 第 65 行: 通过名称获取调度类
    schedule_class = get_schedule_class(pp_schedule)

    # 第 66 行: 判断是单 stage 还是多 stage 调度
    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)

    # 第 68-73 行: 气泡警告
    # num_total_stages = 本 rank 的 stage 数 × pp_degree = 全局虚拟 stage 总数
    num_total_stages = len(stages) * pp_degree
    if n_microbatches < num_total_stages:
        _get_logger().warning(...)
    # 微批次数 < 虚拟 stage 数时，流水线无法充分填充

    # 第 75-80 行: 创建调度实例
    return schedule_class(
        stages if looped_schedule else stages[0],
        # ↑ 多 stage 调度传列表，单 stage 调度传单个
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        scale_grads=False,   # 不自动缩放梯度（见 2.8 节）
    )
```

**关键设计**：单 stage 调度（如 1F1B）只接受一个 `PipelineStage` 对象，多 stage 调度（如 Interleaved1F1B）接受列表。第 76 行的条件分支正是处理这个 API 差异。

## 3.3 `generate_llm_fqn_per_model_part`（83-195 行）

这是负载均衡的核心算法。

```python
def generate_llm_fqn_per_model_part(
    num_stages: int,                  # 虚拟 stage 总数
    num_layers: int,                  # Transformer 层数
    first_stage_less_layers: int = 1, # Embedding 的等效层数权重
    last_stage_less_layers: int = 1,  # norm+output 的等效层数权重
    is_critic: bool = False,          # Critic 模型用 'score' 替代 'output'
) -> list[list[str]]:
```

### 算法流程

```text
┌─────────────────────────────────────────────────────────────┐
│ 输入: num_stages=4, num_layers=8, first_less=1, last_less=1 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  步骤 1: 计算有效层数                                        │
│    effective = 8 + 1 + 1 = 10                                │
│                                                             │
│  步骤 2: 均匀分配                                            │
│    layers_per_stage = 10 // 4 = 2                            │
│    extra_layers = 10 % 4 = 2  (前 2 个 stage 各 +1)          │
│                                                             │
│  步骤 3: 逐 stage 分配                                       │
│    Stage 0: effective=3, 减去 first_less=1 → 2 层            │
│      → ['tok_embeddings', 'layers.0', 'layers.1']            │
│                                                             │
│    Stage 1: effective=3, 中间 stage → 3 层                    │
│      → ['layers.2', 'layers.3', 'layers.4']                  │
│                                                             │
│    Stage 2: effective=2, 中间 stage → 2 层                    │
│      → ['layers.5', 'layers.6']                              │
│                                                             │
│    Stage 3: effective=2, 减去 last_less=1 → 1 层              │
│      → ['layers.7', 'norm', 'output']                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 逐行注解

```python
    # 第 117 行: Critic 模型用 "score" 替代 "output"（value head vs lm_head）
    output_module = "score" if is_critic else "output"

    # 第 120-126 行: 边界情况
    if num_stages < 1: raise ValueError(...)    # 至少 1 个 stage
    if num_stages == 1:                          # 单 stage：包含全部模块
        return [["tok_embeddings"] + [f"layers.{i}" for i in range(num_layers)] + ["norm", output_module]]

    # 第 128-129 行: 有效层数 = 实际层数 + 首尾虚拟权重
    num_effective_layers = num_layers + first_stage_less_layers + last_stage_less_layers

    # 第 131-153 行: 多重验证
    # - num_stages 不能超过有效层数
    # - layers_per_stage 不能为 0
    # - first/last_less 不能超过单 stage 的层数

    # 第 137-138 行: 均分 + 余数
    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages
    # 余数层分配给前 extra_layers 个 stage（每个 +1）

    # 第 158-193 行: 逐 stage 填充
    for stage_idx in range(num_stages):
        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:         # 前面的 stage 多拿 1 层
            effective_layers_for_stage += 1

        if stage_idx == 0:
            # 首 stage: tok_embeddings + (effective - first_less) 层
            stage_modules.append("tok_embeddings")
            num_transformer_layers = effective_layers_for_stage - first_stage_less_layers
            for _ in range(num_transformer_layers):
                stage_modules.append(f"layers.{current_layer}")
                current_layer += 1

        elif stage_idx == num_stages - 1:
            # 尾 stage: (effective - last_less) 层 + norm + output
            num_transformer_layers = effective_layers_for_stage - last_stage_less_layers
            ...
            stage_modules.extend(["norm", output_module])

        else:
            # 中间 stage: 全部是 transformer 层
            for _ in range(effective_layers_for_stage):
                stage_modules.append(f"layers.{current_layer}")
                current_layer += 1
```

### 更多示例

| 参数 | 结果 |
| --- | --- |
| `(2, 4)` | `[['tok_embeddings', 'layers.0', 'layers.1'], ['layers.2', 'layers.3', 'norm', 'output']]` |
| `(1, 4)` | `[['tok_embeddings', 'layers.0', 'layers.1', 'layers.2', 'layers.3', 'norm', 'output']]` |
| `(4, 8, is_critic=True)` | 最后一个 stage 用 `'score'` 替代 `'output'` |
| `(8, 32)` | Stage 0: 4 层, Stage 1: 5 层, Stage 2-6: 4 层, Stage 7: 3 层 + norm + output |

## 3.4 `pipeline_module_split`（198-344 行）

这是模型切割的执行层。

```python
def pipeline_module_split(
    whole_model: nn.Module,              # 完整模型（meta device 上）
    pp_mesh: DeviceMesh,                 # PP 维度的 device mesh
    pp_schedule: str,                    # 调度策略名称
    device: torch.device,               # 目标设备
    module_names_per_stage: list[list[str]],  # 各 stage 的模块名列表
) -> tuple[list[PipelineStage], list[nn.Module]]:
```

### 3.4.1 `_get_stage_indices`（296-327 行）

决定当前 Rank 负责哪些虚拟 stage。

```python
def _get_stage_indices() -> tuple[int, ...]:
    stages_per_rank = num_stages // pp_degree

    # 第 312-314 行: 判断调度风格
    schedule_class = get_schedule_class(pp_schedule)
    v_style_schedules = (ScheduleZBVZeroBubble, ScheduleDualPipeV)
    style = "v" if schedule_class in v_style_schedules else "loop"

    if style == "v":
        # V 风格: Rank 0 → (0, num_stages-1), Rank 1 → (1, num_stages-2)
        # 第 322-323 行: 构建 V 形配对
        stage_v_pairs = list(
            zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1))
        )
        # pp_degree=4, num_stages=8 时:
        # zip([0,1,2,3], [7,6,5,4]) → [(0,7), (1,6), (2,5), (3,4)]
        return stage_v_pairs[pp_rank]
    else:
        # Loop 风格: Rank i → (i, i+pp, i+2pp, ...)
        # 第 327 行
        return tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
        # pp_degree=4, pp_rank=0, stages_per_rank=2 → (0, 4)
```

**V 配对图解（pp=4, 8 stages）：**

```text
Stage:    0   1   2   3   4   5   6   7
Rank:     0   1   2   3   3   2   1   0
                              ↑
                        V 形折返点

正向数据流: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
           R0  R1  R2  R3  R3  R2  R1  R0
```

### 3.4.2 `_build_stage_from_modules`（226-294 行）

把完整模型裁剪成单个 stage。

```python
def _build_stage_from_modules(stage_idx, module_names, num_stages):
    # 第 232 行: 前置断言——模型必须在 meta device 上
    assert next(whole_model.parameters()).device.type == "meta"

    # 第 236 行: Deep copy（meta device 上几乎零成本）
    model = copy.deepcopy(whole_model)
    modules_to_keep = set(module_names)

    # 第 239-283 行: 遍历顶层子模块，删除不属于本 stage 的部分
    for module_name, module_value in list(model.named_children()):

        if isinstance(module_value, nn.ModuleDict):
            # —— 处理 layers（ModuleDict） ——
            # 提取要保留的 key: "layers.0" → "0"
            layers_to_keep = {
                name.split(".", 1)[1] for name in modules_to_keep
                if name.startswith(f"{module_name}.")
            }

            if layers_to_keep:
                # 删除不需要的 key
                for layer_key in list(module_value.keys()):
                    if layer_key not in layers_to_keep:
                        del module_value[layer_key]
                # 结果: ModuleDict 只保留本 stage 的 key，FQN 不变
            else:
                setattr(model, module_name, nn.ModuleDict())

        elif isinstance(module_value, nn.ModuleList):
            # —— 处理 ModuleList（兼容路径，Archon 实际用 ModuleDict）——
            ...

        elif module_name not in modules_to_keep:
            # —— 简单模块（如 tok_embeddings, norm, output）——
            setattr(model, module_name, None)
            # 设为 None，模型 forward() 需处理 None 的情况

    # 第 286-292 行: 创建 PipelineStage 包装
    stage = PipelineStage(
        model,
        stage_idx,       # 在全局虚拟 stage 序列中的位置
        num_stages,      # 虚拟 stage 总数
        device,
        group=pp_mesh.get_group(),  # PP 进程组，用于 Send/Recv
    )
    return stage, model
```

**具体裁剪示例（32 层 Qwen2Model）：**

| Stage | 保留 | 删除/置 None |
| --- | --- | --- |
| Stage 0 | `tok_embeddings`, `layers["0"-"3"]` | `norm=None`, `output=None`, `layers["4"-"31"]` 被 del |
| 中间 Stage | `layers["8"-"11"]` | `tok_embeddings=None`, `norm=None`, `output=None`, 其他 layers 被 del |
| 最后 Stage | `layers["28"-"31"]`, `norm`, `output` | `tok_embeddings=None`, 其他 layers 被 del |

### 3.4.3 主循环（329-344 行）

```python
    stages, model_parts = [], []

    for stage_idx in _get_stage_indices():
        stage, model_part = _build_stage_from_modules(
            stage_idx, module_names_per_stage[stage_idx], num_stages
        )
        stages.append(stage)
        model_parts.append(model_part)

    return stages, model_parts
```

## 3.5 `pipeline_llm`（347-492 行）

总入口，串联所有步骤。

```python
def pipeline_llm(
    model: nn.Module,                         # 完整模型（meta device）
    device: torch.device,                     # 目标设备
    parallel_dims: "ArchonParallelDims",      # 并行维度配置
    archon_config: "ArchonEngineConfig",      # PP 相关配置
    parallelize_fn: Callable,                 # TP/FSDP 并行化函数
    **parallelize_kwargs,
) -> tuple[list[PipelineStage], list[nn.Module], bool, bool]:
```

### 关键流程

```python
    # 第 377-379 行: 获取 PP mesh
    pp_mesh = parallel_dims.get_mesh("pp")

    # 第 381-384 行: 从 config 读取 PP 参数
    pp_schedule = archon_config.pp_schedule        # 默认 "Interleaved1F1B"
    layers_per_stage = archon_config.pp_layers_per_stage  # 默认 None（自动）
    first_stage_less_layers = archon_config.pp_first_stage_less_layers  # 默认 1
    last_stage_less_layers = archon_config.pp_last_stage_less_layers    # 默认 1

    # 第 388-400 行: 从模型获取层数（兼容多种接口）
    # 优先级: model_args.num_hidden_layers > model_args.n_layers
    #         > config.num_hidden_layers > config.n_layers
    # 覆盖 Archon 模型和 HuggingFace 模型两种约定

    # 第 405 行: 检测是否 Critic 模型（用 'score' 替代 'output'）
    is_critic = getattr(getattr(model, "model_args", None), "is_critic", False)

    # 第 407-449 行: 计算虚拟 stage 数
    if layers_per_stage is not None:
        # 用户指定每 stage 的层数 → 计算虚拟 stage 数
        num_virtual_stages = math.ceil(
            (num_layers + first_stage_less_layers + last_stage_less_layers) / layers_per_stage
        )
        # 多重验证: 整除性、调度类型兼容性、V 风格约束
    else:
        # 自动: 1F1B → 1 stage/rank, 其他 → 2 stages/rank
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = pp_degree * stages_per_rank

    # 第 457-463 行: 生成层分配方案
    module_names_per_stage = generate_llm_fqn_per_model_part(
        num_stages=num_virtual_stages, num_layers=num_layers, ...
    )

    # 第 468-474 行: 切割模型
    stages, model_parts = pipeline_module_split(model, pp_mesh, ...)

    # 第 477-481 行: 对每个 model_part 应用 TP/FSDP
    for i, m in enumerate(model_parts):
        m = parallelize_fn(m, parallel_dims, **parallelize_kwargs)
        model_parts[i] = m
        stages[i].submod = m  # 更新 PipelineStage 的模型引用

    # 第 484-485 行: 判断本 rank 是否持有首/尾 stage
    has_first_stage = any(s.is_first for s in stages)
    has_last_stage = any(s.is_last for s in stages)

    return stages, model_parts, has_first_stage, has_last_stage
```

**`has_first_stage` / `has_last_stage` 的用途**：
- `has_first_stage`：决定是否接收输入数据（只有首 stage 需要 token 输入）
- `has_last_stage`：决定是否计算 loss（只有尾 stage 有 logits 输出）

---

# 4. 我该怎么验证自己真的懂了

## 4.1 纸面练习

### 练习 1：手算层分配

给定 `num_stages=4, num_layers=12, first_less=1, last_less=1`，手动计算每个 stage 的模块列表。

```text
预期答案:
  effective = 12 + 1 + 1 = 14
  layers_per_stage = 14 // 4 = 3
  extra_layers = 14 % 4 = 2  (stage 0 和 1 各 +1)

  Stage 0: effective=4, transformer=4-1=3
    → ['tok_embeddings', 'layers.0', 'layers.1', 'layers.2']
  Stage 1: effective=4, transformer=4
    → ['layers.3', 'layers.4', 'layers.5', 'layers.6']
  Stage 2: effective=3, transformer=3
    → ['layers.7', 'layers.8', 'layers.9']
  Stage 3: effective=3, transformer=3-1=2
    → ['layers.10', 'layers.11', 'norm', 'output']

  验证: 所有层 0-11 恰好出现一次 ✓
```

### 练习 2：画 V 风格分配

给定 `pp_degree=4, num_stages=8`，画出 V 风格和 Loop 风格的分配差异。

```text
V 风格:
  Rank 0: (0, 7)  → 首+尾
  Rank 1: (1, 6)
  Rank 2: (2, 5)
  Rank 3: (3, 4)  → 中间两段

Loop 风格:
  Rank 0: (0, 4)  → 前半两段
  Rank 1: (1, 5)
  Rank 2: (2, 6)
  Rank 3: (3, 7)  → 后半两段
```

### 练习 3：理解 ModuleDict 裁剪

Stage 2 持有 `['layers.8', 'layers.9', 'layers.10', 'layers.11']`。对 Qwen2Model 执行 `_build_stage_from_modules` 后，模型结构变成什么？

```text
model.tok_embeddings = None
model.layers = ModuleDict({"8": ..., "9": ..., "10": ..., "11": ...})
model.norm = None
model.output = None
```

## 4.2 运行测试

```bash
# 单元测试: generate_llm_fqn_per_model_part 的各种边界情况
uv run pytest tests/experimental/archon/test_pipeline_parallel.py -v

# 分布式测试（需要 GPU）
# PP forward/backward 正确性
torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_pp_tests.py

# PP + TP/DP/EP 组合
torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_pp_combinations.py

# PP checkpoint save/load
torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_checkpoint_tests.py
```

## 4.3 交互式验证

```python
from areal.experimental.models.archon.pipeline_parallel import generate_llm_fqn_per_model_part

# 验证 1: 所有层恰好分配一次
def verify_complete_assignment(num_stages, num_layers, **kwargs):
    result = generate_llm_fqn_per_model_part(num_stages, num_layers, **kwargs)
    all_layers = []
    for stage in result:
        all_layers.extend(name for name in stage if name.startswith("layers."))
    indices = sorted(int(name.split(".")[1]) for name in all_layers)
    assert indices == list(range(num_layers)), f"Missing or duplicate layers: {indices}"
    assert "tok_embeddings" in result[0], "First stage must have tok_embeddings"
    assert "norm" in result[-1], "Last stage must have norm"
    print(f"✓ {num_stages} stages, {num_layers} layers: all assigned correctly")

verify_complete_assignment(4, 8)
verify_complete_assignment(8, 32)
verify_complete_assignment(2, 3)
verify_complete_assignment(1, 4)

# 验证 2: 边界错误检测
import pytest
with pytest.raises(ValueError): generate_llm_fqn_per_model_part(0, 4)   # < 1 stage
with pytest.raises(ValueError): generate_llm_fqn_per_model_part(20, 4)  # 太多 stage
print("✓ Validation errors caught correctly")

# 验证 3: Critic 模型使用 'score' 而非 'output'
result = generate_llm_fqn_per_model_part(2, 4, is_critic=True)
assert "score" in result[-1] and "output" not in result[-1]
print("✓ Critic model uses 'score' instead of 'output'")
```

## 4.4 常见理解误区

| 误区 | 正确理解 |
| --- | --- |
| "PP 把每层复制到所有 GPU" | 每层只存在于一个 stage/GPU，不复制 |
| "虚拟 stage 需要更多 GPU" | 虚拟 stage 数 > pp_degree，但 GPU 数不变；每个 GPU 持有多段 |
| "deep copy 很贵" | 模型在 meta device 上，deep copy 只复制结构元数据 |
| "删掉的层会泄漏内存" | meta tensor 没有实际存储，del/None 几乎零成本 |
| "Loop 和 V 风格效果一样" | V 风格把首尾 stage 放同一 GPU，才能实现零气泡 |
| "scale_grads=False 意味着不做梯度平均" | 只是 PP schedule 不做；RL 训练的 loss 和 FSDP 各自处理归一化 |
| "has_last_stage 在所有 GPU 上都是 True" | 只有持有最后一个 stage 的 rank 为 True（通常只有 1 或 2 个 rank）|

---

# 5. 附录

## 5.1 系统集成调用链

```text
ArchonEngine._setup_parallelism()
  │
  └─ _apply_pipeline_parallelism()
       │
       ├─ if pp_enabled:
       │    self.spec.pipelining_fn(model, device, parallel_dims, archon_config, parallelize_fn)
       │    # pipelining_fn 实际指向 pipeline_llm()
       │    ├── generate_llm_fqn_per_model_part(...)  → 层分配方案
       │    ├── pipeline_module_split(...)             → 模型裁剪
       │    └── parallelize_fn(model_part, ...)        → 对每段应用 TP/FSDP
       │
       │    结果存储:
       │    self.pp_stages = [PipelineStage, ...]
       │    self.model_parts = [nn.Module, ...]
       │    self.pp_has_first_stage = True/False
       │    self.pp_has_last_stage = True/False
       │
       └─ else:
            parallelize_fn(model, ...)  → 直接并行化完整模型

PipelinedRunner (archon_runner.py)
  │
  ├── _run_train()
  │    ├── schedule = build_pipeline_schedule(stages, pp_schedule, n_microbatches, loss_fn=...)
  │    └── schedule.step(*args, target=batched_target, **batched_kwargs)
  │
  └── _run_eval()
       ├── schedule = build_pipeline_schedule(stages, pp_schedule, n_microbatches, loss_fn=None)
       └── schedule.eval(*args, **batched_kwargs)
```

## 5.2 ArchonEngineConfig 中 PP 相关字段

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `pp_schedule` | `"Interleaved1F1B"` | 调度策略名称 |
| `pp_layers_per_stage` | `None` | 每个虚拟 stage 的层数（None=自动） |
| `pp_first_stage_less_layers` | `1` | Embedding 等效层数权重 |
| `pp_last_stage_less_layers` | `1` | norm+output 等效层数权重 |

## 5.3 complete 验证清单

- [ ] 理解 `generate_llm_fqn_per_model_part` 的均衡算法（给定参数能手算结果）
- [ ] 理解 V 风格 vs Loop 风格的 stage 分配逻辑（能画出分配图）
- [ ] 理解 ModuleDict 裁剪 vs ModuleList 裁剪的区别（知道为什么用 ModuleDict）
- [ ] 理解 meta device 的作用（知道为什么 deep copy 是免费的）
- [ ] 理解 `PipelineScheduleSingle` vs `PipelineScheduleMulti` 的 API 差异
- [ ] 理解 `scale_grads=False` 的原因（知道梯度归一化由谁负责）
- [ ] 理解 `has_first_stage` / `has_last_stage` 的下游影响
