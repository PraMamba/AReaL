# upstream/main feature 与精度对齐 debug 实现分析

> 分析日期：2026-06-02 分析对象：`upstream/main` at `0cfcd04a`
> (`feat:enable v2 training pipeline with controller parity (#1363)`) 工作方式：主分析 + Explore
> / architect / code-reviewer / analyst 四个 subagent 并行只读调查，等待全部完成后综合。

## 结论概览

AReaL 的上游实现 feature 或修复精度对齐问题时，最稳定的工程模式不是“改一个函数”，而是：

1. 先把误差定义成一个可验证不变量，例如 dtype invariant、microbatch invariant、CP invariant、mask
   invariant、version/staleness invariant。
1. 在真实训练路径上做局部修复，尽量保持 public API、部署 artifact、权重同步协议不变。
1. 用最小反例、端到端 torchrun、或大规模实跑指标证明修复有效。
1. 补齐 docs/examples/config/tests，让 feature 入口、debug 入口和用户迁移路径明确。

源码层面已经存在较完整的
debug/对齐基础设施：`stats_tracker`、`stats_logger`、`perf_tracer`、`WorkflowExecutor` 的
trajectory dump、`StalenessManager`、`prox_logp_method`、reward
parser、checkpoint/recover、profiling tools、以及分布式 torchrun 测试。缺口是这些能力还分散，没有统一的 precision
alignment harness 能同时比较 HF / SGLang / vLLM / FSDP / Megatron / Archon 的
token、logprob、version、reward、advantage 和权重同步状态。

## 使用的验证命令

本次调查使用了以下命令族验证，而不是只读 PR 标题：

```bash
git fetch upstream main --prune
git log --oneline --decorate --max-count=40 upstream/main
gh pr view <number> --repo areal-project/AReaL --json number,title,mergedAt,mergeCommit,files,body,url
rg -n "<keyword>" areal docs examples tests
nl -ba <file> | sed -n '<range>p'
```

注意：`inclusionAI/AReaL` 已经迁移/重定向到 `areal-project/AReaL`，所以部分 `gh` 命令需要使用
`--repo areal-project/AReaL`。

## upstream/main 的典型实现模式

### 1. FSDP fp32 master weights：优化器精度不变量

- PR：[#1369](https://github.com/areal-project/AReaL/pull/1369)
- Commit：`237a49f6`
- 文件：`areal/api/cli_args.py`、`areal/engine/fsdp_engine.py`、`tests/test_fsdp_optimizer_dtype.py`、`tests/torchrun/run_fsdp_optimizer_dtype.py`、`docs/*/best_practices/handling_oom.md`
- 问题：`torch.optim.AdamW` 会继承 bf16 参数 dtype，导致长 SFT 训练后期 loss plateau，和 DS-Z3 / Megatron
  precision-aware optimizer path 不对齐。
- 修复模式：
  - 新增 `TrainEngineConfig.optimizer_dtype`，默认 `float32`。
  - FSDP 参数 storage 使用 optimizer dtype，forward/backward 仍通过 mixed precision policy 使用
    compute dtype。
  - HF export 与 XCCL weight sync 前显式 cast 回 compute dtype，避免部署 artifact 和 rollout
    broadcast 行为变化。
  - 测试覆盖 storage dtype、Adam state dtype、forward dtype、HF export dtype、XCCL cast dtype
    五类不变量。

这是典型的“内部训练精度增强，但外部协议不变”的修复。上游 PR 还用 466-step SFT loss 曲线证明 fp32 master weights
的收益，而不是只给单元测试。

### 2. Megatron grad_norm 与 FSDP 对齐：microbatch 不变量

- PR：[#1273](https://github.com/areal-project/AReaL/pull/1273)
- Commit：`5e9fb505`
- 文件：`areal/engine/megatron_engine.py`、`tests/test_megatron_engine_distributed.py`、`tests/torchrun/run_megatron_engine_distributed.py`
- 问题：Megatron Core pipeline schedule 在特定返回路径上对 loss 做 `/num_microbatches`，而 AReaL 已经按全局
  loss weight 做过归一化，导致 MegatronEngine 的 `grad_norm` 和实际 update 比 FSDPEngine 小 N 倍。
- 修复模式：
  - 在 `MegatronEngine.train_batch` 里让 `loss_multiplier` 包含 `len(mb_list)`，抵消 Megatron
    Core 的额外除法。
  - 回归测试 `test_qwen3_grad_norm_mb_invariance` 用同一输入、不同 `max_tokens_per_mb` 验证
    `grad_norm` 在 `1e-3` 内一致。

这是 backend parity 的典型修复：不变量不是“loss 数值看起来正常”，而是“同一 batch 换 microbatch 切分后梯度尺度不变”。

### 3. SFT CP-invariant stats：指标口径不变量

- PR：[#1249](https://github.com/areal-project/AReaL/pull/1249)
- Commit：`95e2379d`
- 文件：`areal/engine/megatron_engine.py`、`areal/engine/megatron_utils/packed_context_parallel.py`、`areal/trainer/sft/lm_engine.py`、`tests/test_reassemble_cp_logprobs.py`
- 问题：CP-local loss 路径中，`sft/n_tokens`、`sft/n_valid_tokens`、`prompt_tokens` 被低估为 CP-local
  计数；ratio metrics 没暴露问题，因为 numerator/denominator 同步缩放。
- 修复模式：
  - 在 Megatron forward path 保留 pre-split `_global_loss_mask`。
  - raw token counters 使用 global mask，CP-local tensors 继续使用 local denominator 以满足 shape
    matching。
  - 测试与实跑验证 CP=2 和 CP=4 下 global token count 一致，local count 分别按 2/4 缩放。

这类修复说明：精度对齐不只看 loss，也要看 metrics 的统计口径是否跨并行维度一致。

### 4. 2D padded sequence advantage mask：mask 不变量

- PR：[#1346](https://github.com/areal-project/AReaL/pull/1346)
- Commit：`b2fb234d`
- 文件：`areal/utils/functional/functional.py`、`tests/test_functional.py`
- 问题：sequence-level PPO/GSPO 在 2D padded batch 中平均 advantage 时把 `loss_mask=False` 的
  padding values 加进 numerator，导致只改 padding advantage 就能改变 valid-token loss、gradient 和
  update。
- 修复模式：
  - reduce 前先 `torch.where(loss_mask, advantages, 0.0)`。
  - broadcast 回 token shape 后再次把 masked positions 清零。
  - 测试 `test_sequence_level_2d_advantage_average_ignores_masked_values` 构造 clean vs
    contaminated padding advantage，断言 valid loss 不变且 masked loss 为 0。

这个 PR 的描述质量很高：包含最小反例、未修/已修数值、边界 hook、验证脚本和 local pytest blocker 说明。它是以后写精度 bug PR 的参考模板。

### 5. staleness recovery：异步训练版本不变量

- PR：[#1345](https://github.com/areal-project/AReaL/pull/1345)
- Commit：`6a84d4ed`
- 文件：`areal/infra/staleness_manager.py`、`areal/trainer/rl_trainer.py`、`tests/test_staleness_manager.py`
- 问题：checkpoint recovery 后 model version 跳到高值，但 `accepted` counter 仍为 0，使 capacity 公式膨胀为
  `(max_staleness + recovered_version + 1) * batch_size`，异步 rollout 可能大量提交并迅速 stale。
- 修复模式：
  - `StalenessManager.on_version_recovered(version)` 将 `accepted` 调整为
    `version * consumer_batch_size`。
  - `PPOTrainer` recover 后调用该方法。
  - 测试验证 recovered version 不改变 `(max_staleness + 1) * batch_size` 上界。

这是异步 RL 系统里必须重视的系统级正确性：version、accepted/running counters、capacity formula 三者需要同时对齐。

### 6. controller v2 parity：feature 上线的兼容模式

- PR：[#1363](https://github.com/areal-project/AReaL/pull/1363)
- Commit：`0cfcd04a`
- 文件：`areal/trainer/rl_trainer.py`、`areal/engine/sglang_remote.py`、`areal/engine/vllm_remote.py`、`areal/experimental/training_service/controller/controller.py`、weight
  update controller/gateway、`examples/math/*.yaml`
- 目标：启用 v2 training pipeline，并让 controller v2 与现有入口保持 parity。
- 实现模式：
  - `rl_trainer`、`sglang_remote`、`vllm_remote` 根据 `config._version == "v2"` 路由到
    `RolloutControllerV2`。
  - `GatewayTrainController` 增加 version management、`connect_engine`、`clear_batches` 等接口。
  - examples 增加 `agent` config section，默认 workflow 切到 `MathAgent`。
  - 测试覆盖 example config 和 weight update controller connect path。

这类 feature 实现的重点是“新路径可切换、旧路径保留、examples 全量更新、controller/weight update 版本管理补齐”。

### 7. 其他可引用上游线索

- #1289 / `b4387877`：Megatron lr schedule 修复，传 absolute `total_train_steps` 给
  `lr_decay_steps`，对齐 HF cosine schedule。
- #1310 / `9c2ec43f`：AWEX colocated CUDA IPC weight transfer，新增
  gateway/adapter/protocol/worker endpoints 和 2/4/8 GPU integration tests。
- #930 / `03d71153`：MIS/TIS 和 rollout-training mismatch 稳定化，扩展 PPO functional 与
  examples/tests。
- #940 / `1fd9f949`：Archon MoE HF parity，修 router score 语义并新增 HF parity 测试。
- #943 / `f5cb33c4`：Archon deterministic training + debug docs，覆盖 deterministic
  algorithms、cuBLAS、NCCL Ring 和 activation checkpoint RNG。
- #928 / `23f058ab`、#796 / `de4c6825`：LoRA weight update versioning / XCCL regression，说明
  weight update version 是 rollout 精度对齐的一等对象。

## 源码中的 feature/debug 对齐模块地图

### 配置与文档生成

- `areal/api/cli_args.py`
  - 集中定义 rollout、actor、engine、perf tracer、memory profiler、saver/recover、stats logger
    等配置。
  - 关键 debug 配置包括：
    - `rollout.max_head_offpolicyness`
    - `rollout.enable_rollout_tracing`
    - `rollout.check_trajectory_format`
    - `rollout.dump_to_file`
    - `actor.recompute_logprob`
    - `actor.prox_logp_method`
    - `perf_tracer.profile_steps`
    - `memory_profiler.profile_steps`
    - `weight_update_mode`
- `docs/generate_cli_docs.py`
  - 自动发现 config dataclass，生成双语 CLI reference。
- `docs/build_all.sh`
  - canonical docs build entrypoint；不要直接用 `jupyter-book build docs/en|docs/zh` 作为
    release preview。

建议：新增 feature/debug 配置时优先从 `cli_args.py` 出发，并同步 `docs/generate_cli_docs.py` 生成
reference。

### Workflow 与 rollout debug

- `areal/api/workflow_api.py`
  - 定义 `RolloutWorkflow.arun_episode()` 合约。
- `areal/workflow/rlvr.py`、`vision_rlvr.py`、`multi_turn.py`
  - 将 dataset sample 转换为 prompt、generation、reward、trajectory tensor。
  - `RLVRWorkflow` 已在 generation/reward 阶段接入 session tracing 和 reward metrics。
- `areal/infra/workflow_executor.py`
  - 统一执行 workflow、处理 async tasks、accept/reject、staleness、dump trajectory。
  - 最适合新增 trajectory semantic validator 和 fail-fast debug mode。
- `examples/docs/debug/cmp_rollout.py`
  - 用 Transformers vs SGLang 比较 rollout accuracy。
  - 当前较专用，适合抽象成通用 CLI：输入模型/backend/dataset/batch，输出 token/logprob/reward diff report。

可插入的精度 debug 点：

1. `engine.agenerate()` 返回后 dump output tokens、backend logprobs、output versions、stop
   reason、server address。
1. reward 计算前后 dump raw text、parsed answer、ground truth、timeout/parse failure。
1. `WorkflowExecutor._execute_workflow()` 返回 trajectory 后检查：
   - prompt token `versions == -1`
   - generated token version 非负
   - `loss_mask` prompt/completion 边界正确
   - `logprobs` 与 generated token 对齐
   - multimodal fields shape 一致

### Trainer / PPO 对齐路径

- `areal/trainer/rl_trainer.py`
  - 主训练循环：rollout -> critic/ref/teacher/prox/current logprob -> advantage -> PPO update
    -> weight update。
  - recover 后同步 `StalenessManager.on_version_recovered()`。
  - weight update 时设置 actor/critic/rollout/eval_rollout version。
- `areal/trainer/ppo/actor.py`
  - `recompute_logprob`、`prox_logp_method`、decoupled loss、KL reward、advantage、importance
    weight、staleness metrics 的核心位置。
  - `_log_proximal_approximation_stats()` 和 `_log_version_staleness_stats()`
    是现有最接近“精度对齐实验模式”的统计入口。
- `areal/utils/functional/functional.py`
  - PPO/SAPO loss、sequence-level ratio/advantage、rejection sampling、masking 等数值逻辑。

建议：做精度对齐时，把同一 batch 上的 rollout logprob、train recompute logprob、ref logprob、prox
logprob、teacher logprob、current logprob、versions 和 loss mask 放在同一份 debug artifact 里。

### Engine 与 weight sync

- `areal/engine/fsdp_engine.py`
  - FSDP forward/train/update weights；fp32 master weights、HF export cast、XCCL cast 都在这里。
- `areal/engine/megatron_engine.py`
  - Megatron train/forward/scheduler/CP reassemble；microbatch loss multiplier 和 absolute
    lr decay steps 修复都在这里。
- `areal/engine/sglang_remote.py`、`areal/engine/vllm_remote.py`
  - rollout controller 入口，v2 path 路由。
- `areal/infra/remote_inf_engine.py`
  - 远程 generation 的第一现场，适合记录 request version、response
    tokens/logprobs、abort/resubmit、server latency。
- `areal/experimental/weight_update/*`
  - AWEX / gateway / adapters / colocated CUDA IPC transfer。

建议：在 debug-only 模式下给 weight update 增加 checksum / dtype / shape / version / chunk-size
报告，尤其是 XCCL 和 colocated CUDA IPC path。

### Metrics / tracing / profiling

- `areal/utils/stats_tracker.py`
  - 支持 scoped scalar、denominator/stat、分布式 reduce。
  - 对 tensor metrics 使用 denominator 是精度统计的关键。
- `areal/utils/stats_logger.py`
  - 将 config、commit/branch/dirty version info、metrics 提交到
    wandb/swanlab/tensorboard/trackio。
- `areal/utils/perf_tracer.py`
  - 写 `traces-r{rank}.jsonl` 和 `sessions-r{rank}.jsonl`，支持 sync/async scope、session
    lifecycle、scheduled profiler。
- `areal/tools/perf_trace_converter.py`
  - 合并 JSONL 为 Chrome Trace JSON。
- `areal/tools/profile_fsdp.py`、`profile_archon.py`、`profile_engines.py`
  - engine 级 profile 工具。

文档入口：

- `docs/en/best_practices/debugging.md`
  - workflow 单样本/并发调试、Transformers vs inference backend rollout consistency、分布式
    hang/deadlock 诊断。
- `docs/en/best_practices/algo_perf.md`
  - reward、importance weight、sequence length 等训练稳定性指标解释。
- `docs/en/best_practices/perf_profiling.md`
  - perf tracer 启用、trace 文件路径、Perfetto 查看和 scheduled profiler。

### Reward/parser

- `areal/api/reward_api.py`
  - reward function contract 与 async wrapper。
- `areal/reward/__init__.py`
  - `MathVerifyWorker` 封装 `math_verify.parse()` 和 `verify()`。
- `areal/reward/gsm8k.py`、`geometry3k.py`、`clevr_count_70k.py`
  - 内置 reward functions。
- `tests/test_math_verify_reward.py`、`tests/test_async_reward_wrapper.py`
  - parser/reward timeout/retry 相关测试。

reward 是精度对齐的重要组成部分：如果 parser timeout、解析失败或格式 anchor 不一致被静默为 0 reward，训练表现会被误判为模型质量问题。

## 当前缺口与风险

### 缺少统一 precision alignment harness

当前能力分散在：

- `examples/docs/debug/cmp_rollout.py`
- `actor.prox_logp_method: metrics`
- `rollout.dump_to_file`
- `perf_tracer`
- `tests/torchrun/*`

建议新增统一工具或 debug mode，输出同一 batch 的：

- prompt / completion tokens
- HF / SGLang / vLLM generation diff
- rollout backend logprobs vs train recompute logprobs
- ref / prox / teacher / current logprobs
- output versions 与 current/proximal version
- reward parse details
- advantage / KL / importance weight
- weight sync version/checksum/dtype

### trajectory dump 信息不足

当前 `dump_to_file` 更偏 prompt/completion/reward/version，复现 logprob 精度问题还不够。建议 debug mode
下扩展 raw tensor、backend metadata、request id、server id、generation params、token-level
logprobs。

### `check_trajectory_format` 只做结构检查

建议新增 semantic checks：

- prompt/generated version 规则
- next-token logprob 与 `loss_mask` 对齐
- completion tokens 与 `output_logprobs` 长度一致
- multimodal fields shape 一致
- packed/2D padded path mask semantics 一致

### code-reviewer 发现的潜在实现风险

以下风险来自 code-reviewer subagent 的只读评审，并经主分析抽样查看关键代码确认，适合后续开专项修复或测试：

1. `versions` 可能没有随 next-token logprob 一起左移。

   - `areal/trainer/ppo/actor.py` 中 `loss_mask` 与 `logprobs` 会 `torch.roll(..., -1)`，但
     `versions` 传入 `_resolve_proximal_logp()` 时仍来自 `input_data.get("versions")`。
   - 风险：staleness/prox-logp approximation 可能按错位 version 计算。
   - 建议：构造 prompt+2 output tokens 的端到端测试，确认首个 loss token 的 version 等于首个 generated token
     version。

1. generation request version 与写入 `output_versions` 的 version 可能发生 race。

   - `remote_inf_engine.py` 构造请求时使用 `self.get_version()`，响应后又使用当前 `self.get_version()`
     填充 `accumulated_versions`。
   - 风险：HTTP await 期间 weight update 推进版本后，旧权重生成的 token 被标为新 version。
   - 建议：每个 generation segment 捕获 `segment_version`，同时用于 request payload 和 response token
     version。

1. accepted/rejected scalar 指标可能隐藏真实拒绝率。

   - `WorkflowExecutor` accepted 时只记录 `accepted=1`，rejected 时只记录
     `rejected=1`；`stats_tracker.scalar()` 导出均值。
   - 风险：看主指标时无法直接得到 acceptance/rejection rate，只能从 `__count` 间接推。
   - 建议：每个 rollout 同时记录 `accepted={0,1}`、`rejected={0,1}`，或改为 SUM/count 语义。

1. staleness max/min/avg 先按 microbatch 汇总再用 scalar 平均。

   - 风险：变长样本下 token-weighted avg、global max/min 会被 microbatch equal-weight 平均掩盖。
   - 建议：直接记录 per-token staleness tensor，并用 generated mask 作为 denominator。

1. `MathVerifyWorker.verify()` timeout 可能不彻底，异常会静默变成 0 reward。

   - `ThreadPoolExecutor` context manager 在 timeout 后默认等待 shutdown，可能降低 timeout 效果。
   - 所有异常返回 0 reward，会把 parser 系统性失败伪装成模型失败。
   - 建议：增加 parse_failure/timeout metrics，或依赖外层进程级 timeout。

## 推荐后续 feature/debug 落地路径

如果要在 AReaL 里实现一个用于精度对齐的 feature，建议按以下顺序做：

1. 定义不变量。

   - 例如：backend token 一致、logprob diff bounded、version 不错位、CP token count
     invariant、microbatch grad_norm invariant、masked padding 不影响 valid loss。

1. 选择统一入口。

   - workflow 输出语义：`WorkflowExecutor._execute_workflow()`
   - generation 第一现场：`RemoteInfEngine.agenerate()`
   - 训练侧 logprob：`FSDPEngine.forward_batch()` / Megatron forward path
   - PPO 中间量：`PPOTrainer.train()` rollout 后 / update 前，`PPOActor` stats
   - weight sync：FSDP/Megatron update weights path

1. 加 debug-only artifact。

   - JSONL 比较报告优先于分散日志。
   - 每条记录带 `global_step`、`task_id`、`session_id`、`version`、backend、rank。

1. 加 focused tests。

   - 单函数最小反例：mask、ratio、reward parser。
   - torchrun invariant：FSDP/Megatron/CP/weight sync。
   - race test：version 在 await 期间变化。

1. 更新 docs/examples。

   - example YAML 中打开
     `dump_to_file`、`recompute_logprob`、`prox_logp_method: metrics`、`perf_tracer.enabled`。
   - docs 中给出如何判断 diff 来自 generation backend、train recompute、reward parser、weight sync 或
     staleness。

## 直接可用的入口清单

| 目标                          | 首选入口                                                                            | 辅助入口                                                                        |
| ----------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 实现新 rollout feature        | `areal/workflow/*`, `areal/api/workflow_api.py`                                     | `examples/*/*.yaml`, `tests/test_workflow_detection.py`                         |
| debug generation/backend 差异 | `areal/infra/remote_inf_engine.py`                                                  | `examples/docs/debug/cmp_rollout.py`                                            |
| debug token/logprob 对齐      | `PPOTrainer.train()` rollout 后、`FSDPEngine.forward_batch()`                       | `actor.prox_logp_method: metrics`                                               |
| debug async staleness         | `areal/infra/staleness_manager.py`                                                  | `tests/test_staleness_manager.py`, `rollout.max_head_offpolicyness`             |
| debug reward/parser           | `areal/reward/*`, `areal/api/reward_api.py`                                         | `tests/test_math_verify_reward.py`, `tests/test_async_reward_wrapper.py`        |
| debug distributed metrics     | `areal/utils/stats_tracker.py`                                                      | `docs/en/reference/metrics_tracking.md`                                         |
| debug performance/hang        | `areal/utils/perf_tracer.py`, `areal/tools/perf_trace_converter.py`                 | `docs/en/best_practices/debugging.md`, `perf_profiling.md`                      |
| debug checkpoint/recover      | `areal/utils/saver.py`, `areal/utils/recover.py`                                    | `docs/en/reference/checkpointing.md`, `tests/test_recover.py`                   |
| debug weight sync             | `FSDPEngine.update_weights()`, Megatron update path, `experimental/weight_update/*` | XCCL / AWEX tests under `tests/torchrun` and `tests/experimental/weight_update` |

## 总结

`upstream/main` 已经展示了成熟的 feature/debug 修复范式：用不变量描述问题，用真实路径修复，用 focused tests 和实跑指标证明，用
docs/examples 让行为可复现。AReaL 本身也具备多数必要组件，但这些组件仍是分散的。后续最有价值的工作是把现有 rollout dump、prox
metrics、stats tracker、perf tracer、reward parser diagnostics、weight version/checksum
串成一个统一的 precision alignment report，让用户能快速判断偏差来自 backend generation、训练侧 recompute、reward
解析、异步 staleness 还是权重同步。
