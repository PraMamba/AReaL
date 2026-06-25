# 开发工具层

> 源码位置：`areal/tools/` 文件数：13 个 | 总行数：5003 行

## 1. 模块定位

`areal/tools/` 是 AReaL 项目的**开发辅助工具集**，不参与运行时训练流水线，而是为 开发者提供四类独立能力：

1. **性能分析**（Profiling）-- 定位 GPU 算子瓶颈，对比引擎实现差异
1. **追踪可视化**（Tracing）-- 将训练过程中产生的 JSONL 追踪数据转换为可交互图表
1. **安装验证**（Validation）-- 检查运行环境依赖完整性和 CUDA 扩展可用性
1. **代码质量门禁**（CI Guards）-- pre-commit 钩子，守护许可证头、项目配置一致性、 CODEOWNERS 格式

## 2. 目录结构与文件清单

```
areal/tools/
|-- check_license_header.py          69 行   CI 钩子：SPDX 许可证头检查/自动修复
|-- check_pyproject_consistency.py  421 行   CI 钩子：双 pyproject.toml 一致性校验
|-- format_codeowners.py            133 行   CI 钩子：CODEOWNERS 格式化与校验
|-- perf_trace_converter.py         553 行   追踪转换：JSONL -> Chrome Trace JSON
|-- plot_session_trace.py          1431 行   追踪可视化：Session JSONL -> HTML 报告
|-- profile_archon.py               387 行   性能分析：Archon 引擎 forward/backward
|-- profile_engines.py              535 行   性能分析：Archon vs FSDP 对比
|-- profile_fsdp.py                 429 行   性能分析：FSDP/HF 引擎 forward/backward
|-- validate_docker_installation.py 367 行   安装验证：Docker 环境专用验证器
|-- validate_installation.py         42 行   安装验证：动态安装验证入口
|-- validation_base.py              589 行   安装验证：BaseInstallationValidator 基类
|-- profiling_utils/
|   |-- __init__.py                   9 行   公开 generate_random_seq_lens
|   |-- utils.py                     38 行   共享工具：随机变长序列生成
```

## 3. 四大功能域架构

### 3.1 性能分析工具套件

```
                      profile_engines.py (对比编排器)
                     /                    \
          profile_archon.py          profile_fsdp.py
          (Archon 引擎分析)          (FSDP/HF 引擎分析)
                \                        /
                 profiling_utils/utils.py
                 (generate_random_seq_lens)
```

**三个分析工具的维度对比**：

| 维度     | profile_archon           | profile_fsdp            | profile_engines       |
| -------- | ------------------------ | ----------------------- | --------------------- |
| 行数     | 387 行                   | 429 行                  | 535 行                |
| 目标引擎 | Archon (自研模型)        | FSDP/HuggingFace        | 两者对比              |
| 输入格式 | packed (cu_seqlens)      | padded (attention_mask) | 两种都构造            |
| 模型加载 | load_archon_model        | AutoModelForCausalLM    | 先后各加载一次        |
| 模式     | forward/backward/both    | forward/backward/both   | forward/backward/both |
| 输出     | 单个 Chrome Trace        | 单个 Chrome Trace       | 两个 Trace + 对比报告 |
| 特有参数 | --gradient-checkpointing | --attn-impl             | --top-ops             |

**关键数据流**：

```
用户 CLI 参数
    |
    v
parse_args() --> 解析 --mode, --batch-size, --seq-lens, --dtype 等
    |
    v
create_packed_input() / create_padded_input()
    |  调用 profiling_utils.generate_random_seq_lens() 生成变长序列
    v
warmup (N 次空跑, torch.cuda.synchronize)
    |
    v
torch.profiler.profile(CPU + CUDA)
    |  记录 record_shapes, profile_memory
    v
key_averages().table(sort_by="cuda_time_total")
    |
    v
prof.export_chrome_trace() --> 输出 .json 供 Perfetto 查看
```

**profile_engines.py 的编排逻辑** (第 417-525 行)：

1. 固定随机种子 (seed=42) 确保两引擎输入一致
1. 依次调用 `profile_archon()` 和 `profile_fsdp()`，每次调用后
   `del model; torch.cuda.empty_cache()`
1. 通过 `EngineResult` 数据类 (第 40-46 行) 收集 CUDA 总耗时、峰值显存、top-N 算子
1. `print_comparison()` (第 349-414 行) 输出并列表格和定性判定(快/慢/等效，阈值 5%)

### 3.2 追踪数据流水线

运行时 PerfTracer (定义于 `areal/api/cli_args.py:2517` PerfTracerConfig) 在各引擎中 埋点输出 JSONL
文件，工具层负责后处理：

```
运行时引擎 (FSDPEngine / ArchonEngine / SGLang / vLLM / Megatron)
    | 输出 JSONL 追踪文件 (每 rank 一个)
    v
perf_trace_converter.py -----> Chrome Trace JSON (.json)
    |                           供 chrome://tracing 或 Perfetto 查看
    |
    | 同一 JSONL 也可输入
    v
plot_session_trace.py -------> 交互式 HTML 报告
                                (Plotly 生成 4 类图表)
```

**perf_trace_converter.py 核心处理** (553 行)：

- `_load_events()` (第 14 行)：逐行解析 JSONL
- `_extract_rank()` / `_extract_role()` (第 30/57 行)：从 event.args 提取分布式标识
- `_remap_process_and_thread_ids()` (第 121-284 行，164 行)：最核心的函数，将多 rank 的 pid/tid 映射为全局唯一
  ID，生成 process_name / thread_name 元数据事件
- `convert_jsonl_to_chrome_trace()` (第 297 行)：公开 API，支持文件/目录/glob 输入， 合并、重映射、排序后输出
  `{"traceEvents": [...], "displayTimeUnit": "ms"}`
- 流式 ID (flow events s/t/f) 和 correlation 也在第 355-401 行做全局重映射

**plot_session_trace.py 核心处理** (1431 行，本模块最大文件)：

- 输入：Session JSONL（含 submit_ts, finalized_ts, phases, rank, task_id 等字段）
- 数据预处理管线 (第 1297-1332 行 main 函数)： `_ensure_numeric` -> `_maybe_compute_durations` ->
  `_extract_phase_timestamps` -> `_compute_offsets` -> `_determine_step_timepoints` ->
  `_apply_step_assignments`
- 输出 4 类 HTML 图表：

| 图表               | 构建函数                          | 说明                                   |
| ------------------ | --------------------------------- | -------------------------------------- |
| 整体分布直方图     | build_overall_distribution_figure | total_s/generate_s/reward_s/toolcall_s |
| 按 step 分布直方图 | build_step_distribution_figure    | Dropdown 切换 step                     |
| 生命周期时间线     | build_timeline_figures            | 每 rank 一个图，显示 phases 色段       |
| 延迟散点图         | build_latency_figure              | 总耗时随时间变化，p50/p95 参考线       |

Session 时间线中的阶段色段定义 (第 37-42 行)：

- idle (灰色 #d1d5db)、generate (蓝色 #2563eb)、reward (红色 #dc2626)、 toolcall (橙色 #f59e0b)

### 3.3 安装验证体系

```
                validation_base.py (589 行)
                BaseInstallationValidator
                /                       \
validate_installation.py           validate_docker_installation.py
DynamicInstallationValidator       DockerInstallationValidator
(42 行，最小子类)                  (367 行，扩展大量 Docker 包)
```

**BaseInstallationValidator 类结构** (第 29 行)：

```
BaseInstallationValidator
|-- PACKAGE_IMPORT_MAP       dict  包名 -> import 名映射 (第 39-57 行)
|-- CUDA_SUBMODULES          dict  包名 -> CUDA 子模块列表 (第 60-70 行)
|-- CRITICAL_PACKAGES        set   关键包集合 (第 74-90 行)
|-- __init__()                     接受 pyproject_path
|-- parse_pyproject()              解析 pyproject.toml 的 [project].dependencies
|-- _get_optional_dep_versions()   解析 optional-dependencies 版本说明符
|-- add_additional_package()       添加 pyproject 之外的包 (Docker 特有)
|-- normalize_package_name()       PEP 503 名称归一化
|-- get_installed_version()        尝试多种名称变体获取已安装版本
|-- check_version()                使用 packaging.specifiers 做版本匹配
|-- _test_import_direct()          在当前进程执行 importlib.import_module
|-- _run_import_tests()            顺序执行所有导入测试
|-- test_cuda_submodules()         深度验证 CUDA 扩展 (torch.cuda, flash_attn 等)
|-- validate_all_dependencies()    主验证入口，分"关键包"和"其他包"两组报告
|-- test_cuda_functionality()      基础 CUDA 运算 + flash_attn 功能测试
|-- print_summary()                汇总统计，区分 CRITICAL_FAILURES 和 WARNINGS
|-- run()                          编排入口：parse -> validate -> cuda_test -> summary
```

**DockerInstallationValidator** (第 24 行) 在基类之上：

- 扩展 CUDA_SUBMODULES：增加 grouped_gemm, apex, transformer_engine, flash_attn_3,
  megatron-core, DeepSeek-V3 系列 (flash_mla, deep_gemm, deep_ep, fla, causal_conv1d)
- 扩展 CRITICAL_PACKAGES：增加 Docker 特有的必须包
- 重写 parse_pyproject()：自动检测 sglang/vllm 互斥后端 (第 96-138 行)
- 重写 test_cuda_functionality()：增加 Transformer Engine FP8、Apex FusedAdam、
  FlashMLA、DeepGEMM、DeepEP、fla MultiScaleRetention 等专项测试
- `_detect_pyproject()` (第 335 行)：根据是否安装 vllm 自动选择 pyproject.vllm.toml

**DynamicInstallationValidator** (第 17 行) 几乎为空壳子类，仅重写 `get_validation_title()`
返回标题字符串，所有验证逻辑完全继承自基类。

### 3.4 CI 代码质量门禁

三个工具均注册为 pre-commit 钩子 (`.pre-commit-config.yaml` 第 72/108/118 行)。

**check_license_header.py** (69 行)：

- 检查 areal/ 下所有 `.py` 文件是否包含 `# SPDX-License-Identifier: Apache-2.0`
- `needs_header()` (第 21 行)：读取文件全文，检查 HEADER 是否存在
- `fix_file()` (第 30 行)：自动插入 SPDX 头，尊重 shebang 行 (`#!` 开头则插入第 2 行)
- 退出码语义：0=全部已有头；1=已自动修复（pre-commit 会 re-stage）

**check_pyproject_consistency.py** (421 行)：

- 比较 pyproject.toml 与 pyproject.vllm.toml 的一致性
- `ESCAPABLE_PACKAGES` (第 32-46 行)：允许不一致的包（torch, sglang, vllm 等后端特有包）
- `_Checker` 类 (第 103 行)：累积错误的比较引擎
  - `check_dependencies()`：比较 \[project\].dependencies，跳过 escapable 包
  - `check_optional_deps()`：比较 optional-dependencies，跳过 BACKEND_EXTRAS
  - `check_override_deps()`：比较 \[tool.uv\].override-dependencies
  - `check_uv_sources()`：比较 \[tool.uv.sources\]
  - `run()` (第 301 行)：按 7 步编排全面比较 (build-system -> project metadata -> dependencies ->
    optional-dependencies -> dependency-groups -> tool.uv -> other tool sections)

**format_codeowners.py** (133 行)：

- 解析 `.github/CODEOWNERS`，进行三类处理：
  1. 对齐 owner 列到统一缩进位 (MIN_COLUMN=32，第 34 行)
  1. 校验 owner token 格式 (`@user` 或 `@org/team`，正则第 30 行)
  1. 检测重复路径模式、警告单人 owner
- 退出码：0=无变更；1=已重写文件；2=格式错误

## 4. 关键依赖关系

```
                     外部依赖
                    +--------+
                    | torch  |-- profiler, cuda, nn
                    +--------+
                    | plotly |-- go, make_subplots (plot_session_trace)
                    +--------+
                    | pandas |-- DataFrame 处理 (plot_session_trace)
                    +--------+
                    |packaging|-- Requirement, SpecifierSet, Version
                    +--------+
                    |tomllib |-- TOML 解析 (validation_base, check_pyproject)
                    +--------+

                    项目内依赖
            +-----------------------------+
            | areal.infra.current_platform|-- set_device, device_type
            +-----------------------------+
            | areal.utils.testing_utils   |-- MODEL_PATHS, load_archon_model
            +-----------------------------+
            | areal.engine.fsdp_utils     |-- attn_impl 校验
            +-----------------------------+
            | areal.experimental.models   |-- ActivationCheckpointConfig
            +-----------------------------+
            | areal.api.cli_args          |-- PerfTracerConfig (运行时追踪配置)
            +-----------------------------+
```

模块间内部依赖关系：

```
profile_engines -----import----> profile_archon.{create_packed_input,
                                                  run_forward,
                                                  run_forward_backward}
profile_engines -----import----> profile_fsdp.{create_padded_input,
                                               load_hf_model,
                                               run_forward,
                                               run_forward_backward}
profile_archon  -----import----> profiling_utils.generate_random_seq_lens
profile_fsdp    -----import----> profiling_utils.generate_random_seq_lens
profile_engines -----import----> profiling_utils.generate_random_seq_lens

validate_docker -----extends---> validation_base.BaseInstallationValidator
validate_install-----extends---> validation_base.BaseInstallationValidator
```

## 5. 数据格式与协议

### 5.1 PerfTracer JSONL 事件格式

```json
{
  "ph": "X",
  "name": "forward",
  "ts": 1718000000.123,
  "dur": 45.6,
  "pid": 12345,
  "tid": 1,
  "args": {
    "rank": 0,
    "role": "actor"
  }
}
```

- `ph`：Chrome Trace 事件类型 (X=完成事件, M=元数据, s/t/f=流事件)
- `pid`/`tid`：进程/线程 ID，converter 会做全局重映射
- `args.rank` 和 `args.role`：分布式训练标识，用于分组显示

### 5.2 Session JSONL 记录格式

```json
{
  "rank": 0,
  "task_id": 1,
  "session_id": 0,
  "status": "accepted",
  "submit_ts": 1718000000.0,
  "finalized_ts": 1718000045.6,
  "total_s": 45.6,
  "generate_s": 30.2,
  "reward_s": 10.1,
  "toolcall_s": 5.3,
  "phases": {
    "generate": [{"start_ts": 1718000001.0, "end_ts": 1718000031.2}],
    "reward": [{"start_ts": 1718000032.0, "end_ts": 1718000042.1}],
    "toolcall": [{"start_ts": 1718000042.5, "end_ts": 1718000045.0}]
  }
}
```

### 5.3 Chrome Trace JSON 输出格式

```json
{
  "traceEvents": [
    {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "[actor] Rank 0, Process 12345"}},
    {"name": "thread_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "[Thread 1]"}},
    {"ph": "X", "name": "forward", "ts": 0.123, "dur": 45.6, "pid": 1, "tid": 1}
  ],
  "displayTimeUnit": "ms"
}
```

## 6. 设计模式与实现亮点

### 6.1 模板方法模式 -- 安装验证

`BaseInstallationValidator.run()` (第 583 行) 定义验证流程骨架：

```python
def run(self):
    self.parse_pyproject()           # 可重写：Docker 版扩展 CUDA 包
    self.validate_all_dependencies() # 继承：统一的导入+版本检查
    self.test_cuda_functionality()   # 可重写：Docker 版增加 FP8/Apex 测试
    success = self.print_summary()   # 继承：统一的汇总报告
    return success
```

子类仅需重写 `parse_pyproject()` 添加额外包、`test_cuda_functionality()` 增加专项测试、
`get_validation_title()` 改标题。DynamicInstallationValidator 以 42 行完成最小继承。

### 6.2 策略模式 -- 性能分析的输入构造

Archon 和 FSDP 使用不同的输入张量格式：

- **Archon**：packed 格式，`input_ids` shape `[1, total_len]`，附带 `cu_seqlens` 累计 序列长度索引（第
  147-189 行 `create_packed_input`）
- **FSDP/HF**：padded 格式，`input_ids` shape `[batch, max_seqlen]`，附带 `attention_mask`（第
  198-241 行 `create_padded_input`）

`profile_engines.py` 通过 `functools.partial` 将不同的 run 函数绑定为统一的 `run_fn()` 可调用对象（第 189-192
行、第 289-293 行），实现统一的 warmup/profile 循环。

### 6.3 Escapable 白名单 -- 配置一致性检查

`check_pyproject_consistency.py` 的核心设计是 `ESCAPABLE_PACKAGES` (第 32 行) 和 `BACKEND_EXTRAS`
(第 50 行)。SGLang 和 vLLM 后端对 torch/torchao/transformers 等
有互斥的版本约束，通过白名单机制跳过这些包的版本比较，只确保其余所有包完全一致。

### 6.4 全局 ID 重映射 -- 追踪合并

`perf_trace_converter.py` 的 `_remap_process_and_thread_ids()` (第 121-284 行)
是最复杂的函数。它解决的问题是：多 rank 各自输出的 JSONL 中 pid/tid 会冲突。 重映射策略为：

1. 收集所有 `(rank, role, original_pid)` 三元组为 pid key
1. 按 `(rank_sort_key, role_sort_key, value_sort_key)` 排序
1. 分配连续的新 pid (从 1 开始)
1. 对 tid 在每个新 pid 内部分配连续编号
1. 生成 process_name / thread_name / process_sort_index / thread_sort_index 四类 元数据事件

## 7. 运行入口与 CLI 接口

| 工具                  | 运行方式                                             | 关键参数                              |
| --------------------- | ---------------------------------------------------- | ------------------------------------- |
| profile_archon        | `python -m areal.tools.profile_archon`               | --mode, --seq-lens, --dtype, --output |
| profile_fsdp          | `python -m areal.tools.profile_fsdp`                 | --mode, --seq-lens, --attn-impl       |
| profile_engines       | `python -m areal.tools.profile_engines`              | --mode, --top-ops, --output-dir       |
| perf_trace_converter  | `python -m areal.tools.perf_trace_converter`         | input \[output\], --display-time-unit |
| plot_session_trace    | `python -m areal.tools.plot_session_trace`           | input, -B batch_size, -L, -N limit    |
| validate_installation | `python areal/tools/validate_installation.py`        | (无参数)                              |
| validate_docker       | `python areal/tools/validate_docker_installation.py` | (无参数)                              |
| check_license_header  | pre-commit 自动调用                                  | file1.py file2.py ...                 |
| check_pyproject       | pre-commit 自动调用                                  | \[file_a file_b\]                     |
| format_codeowners     | pre-commit 自动调用                                  | (固定 .github/CODEOWNERS)             |

## 8. 设计约束与扩展指南

### 当前约束

1. **性能分析工具依赖 GPU**：`profile_*` 系列必须在有 CUDA 设备的环境运行，通过 `current_platform.set_device(0)`
   初始化设备
1. **追踪工具假设 JSONL 格式**：`perf_trace_converter` 和 `plot_session_trace` 要求 输入严格为每行一个 JSON
   对象的 JSONL 格式
1. **Docker 验证器使用相对导入**：`validate_docker_installation.py` 第 19 行使用
   `from validation_base import BaseInstallationValidator`（非包内导入），需要在 `areal/tools/`
   目录下直接运行
1. **pyproject 一致性检查硬编码双文件**：默认比较 `pyproject.toml` 和 `pyproject.vllm.toml`，不支持第三后端扩展

### 扩展指南

- **新增引擎分析器**：仿照 `profile_archon.py` 创建 `profile_<engine>.py`，实现
  `create_*_input()`、`run_forward()`、`run_forward_backward()`，然后在 `profile_engines.py`
  中添加对比分支
- **新增追踪可视化**：在 `plot_session_trace.py` 中添加新的 `build_*_figure()` 函数， 在 `main()` 末尾调用并写出
  HTML
- **新增环境验证器**：继承 `BaseInstallationValidator`，重写 `parse_pyproject()` 添加 环境特有包，重写
  `test_cuda_functionality()` 增加专项测试
- **新增 CI 钩子**：创建新脚本，在 `.pre-commit-config.yaml` 中注册 entry
