# AReaL 源码走读：SGLang Pipeline Parallel / PP 支持实现解析

在 RLHF / RLVR 训练系统里，推理侧常常不是“附属模块”，而是吞吐、显存和权重同步链路里的第一等公民。AReaL 的训练引擎可以是 FSDP、Megatron 或
Archon，而 rollout 侧又可以交给 SGLang/vLLM 这样的推理服务。于是一个看似简单的问题出现了：如果 SGLang 自己也用 Pipeline
Parallelism，把模型层切到多个 PP stage 上，训练侧该如何把最新权重正确、及时地同步过去？

本文不展开 SGLang PP 的通用原理，也不讲 Megatron/Archon pipeline schedule 的数学细节。我们只沿着 AReaL
源码看一个具体问题：`rollout.backend=sglang:d1p2t2` 这种配置如何变成真实的 SGLang PP server、rank
拓扑和权重更新通信；它解决了什么死锁问题，又留下了哪些边界与维护风险。

# 前言

## 工程背景：PP 出现在 rollout/inference，而不是只出现在训练

AReaL 的配置把 actor、critic、rollout 当成独立 engine 来分配资源。`PPOTrainer.__init__` 会分别解析
`config.actor.backend` 和 `config.rollout.backend`：

- `areal/trainer/rl_trainer.py:131-135`：`actor_alloc = ModelAllocation.from_str(config.actor.backend)`，`rollout_alloc = ModelAllocation.from_str(config.rollout.backend)`。
- `areal/api/alloc_mode.py:32-45`：`ParallelStrategy` 把 TP、PP、DP、CP、EP 都作为并行维度描述。
- `areal/api/alloc_mode.py:153-160`：`world_size = data_parallel_size * context_parallel_size * tensor_parallel_size * pipeline_parallel_size`。

这意味着 SGLang 的 PP 不是训练模型内部的一个 schedule 细节，而是 rollout server 的资源拓扑：一个 SGLang server
instance 内部使用 `TP × PP` 张 GPU；`d` 维才表示独立 server replica 数量。文档也在 inference backend 表中写到
SGLang 支持 `d/t/p`，并说明每个 inference instance 使用 `t × p`
GPUs（`docs/en/reference/alloc_mode.md:106-112`）。

## 核心矛盾：传 `pp_size` 容易，权重同步不容易

SGLang PP 支持的核心矛盾可以浓缩成三句话：

1. rollout server 可以按 PP stage 拆层，但训练侧每轮 PPO update 后仍要把 actor 最新权重同步到 rollout server。
1. 如果仍用一个覆盖所有 SGLang TP×PP worker 的全局 NCCL group，SGLang PP scheduler 的事件循环可能在 PP rank 0
   的初始化请求上阻塞，导致后续 PP rank 永远收不到请求。
1. 因此 AReaL 真正要实现的不是“加一个 `--pp-size` 参数”，而是“按 PP stage 建立独立权重更新组，并让 SGLang 只在匹配的 PP rank
   上 join / recv / destroy”。

这点在 FSDP 代码注释里说得最直接：当 inference side `gen_pp_size > 1` 时，单个 group 会因为 SGLang PP
scheduler 先处理 PP rank 0、再转发给 PP rank 1 的顺序而 deadlock；per-PP-rank groups
可以避免这个问题（`areal/engine/fsdp_engine.py:1385-1398`）。

## 本文主线

本文按机制而不是按文件走读：

1. 配置如何从 `sglang:d1p2t2` 进入 SGLang 启动参数。
1. 为什么 PP 权重同步必须从单 group 变成 per-PP group。
1. `PPSchedulerBridge` 如何在 SGLang 进程内做 stage-aware 路由。
1. FSDP、Megatron、Archon 三类训练引擎如何分别适配 SGLang PP。
1. rank / group / tensor shape / 状态流如何变化。
1. 性能、显存、测试覆盖与设计风险在哪里。

## 不展开的内容

本文不讲 Pipeline Parallel 的理论调度，不讲 SGLang 内部每层如何切分，也不讲 Megatron/Archon 的完整 PP schedule。它只分析
AReaL 如何把 SGLang PP 接到 rollout、权重同步、初始化、保存/加载和测试链路里。

## 核心文件表

| 文件                                                                                  | 职责                                                                |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `areal/api/alloc_mode.py`                                                             | 解析 `sglang:d...p...t...`，定义 PP 维度和 world size 计算          |
| `areal/api/cli_args.py`                                                               | `SGLangConfig.build_args/build_cmd` 把 `pp_size` 传给 SGLang server |
| `areal/trainer/rl_trainer.py`                                                         | 从用户配置进入 rollout engine/controller 的主入口                   |
| `areal/engine/sglang_remote.py`                                                       | 构造 SGLang 生成请求、权重更新请求、PP-aware init group payload     |
| `areal/infra/remote_inf_engine.py`                                                    | 统一远程 inference engine 的 HTTP 请求、init/update 异步调度        |
| `areal/experimental/inference_service/sglang/launch_server.py`                        | AReaL 版 SGLang server 启动入口                                     |
| `areal/experimental/inference_service/sglang/scheduler.py`                            | 复制/适配 SGLang scheduler 进程，并绑定 AReaL bridge                |
| `areal/experimental/inference_service/sglang/pp_bridge.py`                            | PP stage-aware 权重更新路由核心                                     |
| `areal/engine/fsdp_engine.py`                                                         | FSDP actor 同步到 SGLang PP 的特殊路径                              |
| `areal/engine/megatron_engine.py` / `areal/experimental/engine/archon_weight_sync.py` | 真 PP 训练引擎与 SGLang PP 的 stage-to-stage 同步                   |

# 一、配置入口：`sglang:d1p2t2` 解决的是“用户意图如何变成 server 拓扑”

## 1.1 设计哲学与核心问题

SGLang PP 的第一层问题不是通信，而是配置语义：用户不会直接写 SGLang 的内部 rank map，而是写 AReaL 的 backend 字符串。AReaL
必须把 `p` 解释成 rollout server 内部的 pipeline stage 数，并把它一路传到 SGLang 启动命令。

如果这一层缺失，会出现一个很隐蔽的问题：配置里写了 `sglang:d1p2t2`，资源调度也可能按 `tp*pp=4` 张 GPU 申请，但 SGLang 进程本身没收到
`pp_size=2`，最后仍按 PP=1 运行。测试里专门把这类问题称为 “silently fall back to
PP=1”（`tests/test_sglang_pp_unit.py:363-367`）。

## 1.2 源码入口与关键对象

```text
areal/api/alloc_mode.py
  - ParallelStrategy：定义 pipeline_parallel_size / pp_size
  - ModelAllocation.from_str：解析 sglang:d1p2t2
  - _LLMParallelParser.modern_inf_para：把 inf_dim p 写入 strategy_kwargs

areal/api/cli_args.py
  - SGLangConfig.build_args：把 pp_size 转成 SGLang server args
  - SGLangConfig.build_cmd：构造 areal.experimental.inference_service.sglang.launch_server 命令

areal/trainer/rl_trainer.py
  - PPOTrainer：解析 rollout_alloc，并将 pp_size 传给 SGLangConfig
```

## 1.3 主流程拆解

用户配置通常长这样：

```yaml
rollout:
  backend: "sglang:d1p2t2"
actor:
  backend: "megatron:d1p2t2"   # 或 fsdp/archon
```

主路径是：

```text
PPOTrainer.__init__
  -> ModelAllocation.from_str(config.rollout.backend)
    -> _LLMParallelParser.modern_inf_para
      -> ParallelStrategy(data_parallel_size=1,
                          pipeline_parallel_size=2,
                          tensor_parallel_size=2)

PPOTrainer._make_rollout_engine / _make_inference_engine
  -> SGLangConfig.build_args(
       tp_size=self.rollout_alloc.parallel.tp_size,
       pp_size=self.rollout_alloc.parallel.pp_size,
       base_gpu_id=0,
     )
  -> RemoteSGLangEngine / RolloutController.initialize(server_args=...)
```

源码证据：

- inference grammar 允许
  `INF_DIM_TYPE: "d" | "t" | "p"`（`areal/api/alloc_mode.py:622-628`）。
- `modern_inf_para` 遇到 `dim.type_ == "p"` 就写入
  `pipeline_parallel_size`（`areal/api/alloc_mode.py:796-806`）。
- `PPOTrainer` 的 SGLang 分支调用
  `SGLangConfig.build_args(... pp_size=self.rollout_alloc.parallel.pp_size)`（`areal/trainer/rl_trainer.py:981-994`）。
- `SGLangConfig.build_args` 只在 `pp_size > 1` 时注入
  `args["pp_size"]`（`areal/api/cli_args.py:1842-1882`）。

一个 rank/资源示意图：

```text
rollout.backend = sglang:d2p2t4

DP replicas: 2 个 SGLang server instance
每个 instance: PP=2 × TP=4 = 8 个 worker/GPU
总 inference world_size = 2 × 2 × 4 = 16

server replica 0: pp0/tp0..3, pp1/tp0..3
server replica 1: pp0/tp0..3, pp1/tp0..3
```

注意这里的 `d=2` 是两个 server replica，不是 SGLang 内部所有 worker 都直接暴露给 AReaL 的 rollout dispatcher。

## 1.4 关键细节与误区澄清

第一个容易误解的点：`SGLangConfig` dataclass 本身没有用户可填的 `pp_size` 字段。PP 来自 `rollout.backend`
解析结果，而不是 `sglang:` 配置块。`SGLangConfig.build_args` 接收外部传入的 `pp_size`，再决定是否写入 server
args（`areal/api/cli_args.py:1842-1882`）。

第二个容易误解的点：文档和 README 对 SGLang PP 的描述并不完全一致。`docs/en/reference/alloc_mode.md:106-112` 说
SGLang inference backend 支持 `d/t/p`，但 README 的 inference backend 表仍标记 SGLang Pipeline
Parallel 为 ❌（`README.md:239-242`）。本文以源码和测试为准：源码路径和测试都已经覆盖 `sglang:...p2...`，README
这里更像滞后的能力表。

第三个容易误解的点：旧 SPMD launcher 里仍有阻断逻辑。`areal/infra/utils/launcher.py:263-268` 对
`allocation_mode.gen_backend == "sglang"` 且 `gen.pp_size > 1` 抛
`NotImplementedError`。而本文主线是 per-engine backend + single-controller / controller 路径；测试的
E2E 也是 `scheduler.type=local`（`tests/test_sglang_pp_distributed.py:145-195`）。

## 1.5 本章小结

💡 小结

- `p` 是从 `rollout.backend` 解析出来的资源维度，不是 `sglang` 配置块里的字段。
- SGLang server args 只在 `pp_size > 1` 时显式带上 `pp_size`，保持 PP=1 的兼容行为。
- SGLang PP 的主测试路径是 single-controller/local scheduler；旧 SPMD launcher 仍显式阻断 PP
  generation。

# 二、启动与注入：`PPSchedulerBridge` 解决的是“如何在不改 SGLang 源码模块的情况下接管权重更新”

## 2.1 设计哲学与核心问题

SGLang PP 支持必须进入 SGLang scheduler/model runner 内部，因为是否 join 某个 NCCL group 取决于 SGLang
worker 的 `pp_rank`。AReaL 有两种选择：直接改 SGLang 源码，或在启动时替换 scheduler 入口并对实例方法做组合式包装。

源码选择了第二种：AReaL 复制/适配 SGLang 的 launch 与 scheduler process，把 AReaL bridge 绑定到已经构造好的
scheduler 实例上。它不是“完全零 patch”——因为 launch/scheduler 文件确实复制了上游内部流程；但它避免了 module-level
monkey patch，不会全局替换 SGLang 类定义。

## 2.2 源码入口与关键对象

```text
areal/experimental/inference_service/sglang/launch_server.py
  - areal_launch_server：传入 areal_run_scheduler_process 替代默认 scheduler process

areal/experimental/inference_service/sglang/scheduler.py
  - areal_run_scheduler_process：创建 SGLang Scheduler 后绑定 bridge
  - AwexSchedulerBridge：绑定 awex_* 管理接口
  - PPSchedulerBridge：绑定 PP-aware 权重更新逻辑

areal/experimental/inference_service/sglang/pp_bridge.py
  - PPSchedulerBridge.bind：pp_size<=1 时 no-op；PP>1 时包装 tp_worker/model_runner
```

## 2.3 主流程拆解

SGLang server 的启动命令不是直接 `python -m sglang.launch_server`，而是：

```text
SGLangConfig.build_cmd_from_args
  -> get_py_cmd("areal.experimental.inference_service.sglang.launch_server", args)
```

对应源码在 `areal/api/cli_args.py:1835-1839`。

进入 AReaL 的 SGLang launcher 后：

```text
areal_launch_server(server_args)
  -> Engine._launch_subprocesses(
       run_scheduler_process_func=areal_run_scheduler_process,
     )
  -> register_awex_endpoints(app, rpc_proxy)
```

证据在 `areal/experimental/inference_service/sglang/launch_server.py:41-64`。

scheduler 子进程里：

```text
areal_run_scheduler_process(... pp_rank, dp_rank, ...)
  -> configure_scheduler(... pp_rank, dp_rank)
  -> Scheduler(..., pp_rank, ..., dp_rank)
  -> AwexSchedulerBridge(scheduler).bind()
  -> PPSchedulerBridge(scheduler, server_args).bind()
  -> scheduler.run_event_loop()
```

证据在 `areal/experimental/inference_service/sglang/scheduler.py:132-152` 和 `:204-224`。

`PPSchedulerBridge.bind()` 做的事情很克制：

```python
if self._pp_size <= 1:
    return

tp_worker = scheduler.tp_worker
model_runner = tp_worker.model_runner
self._bind_tp_worker(tp_worker, model_runner)
self._bind_model_runner(model_runner)
```

对应 `areal/experimental/inference_service/sglang/pp_bridge.py:63-90`。

## 2.4 关键细节与误区澄清

这里有一个容易误解的点：`PPSchedulerBridge` 文档说 “No inheritance. No monkey-patch on module-level
classes”（`areal/experimental/inference_service/sglang/pp_bridge.py:14-23`），这并不等于没有维护风险。`launch_server.py`
明确写着它 adapted from SGLang upstream launch
flow（`areal/experimental/inference_service/sglang/launch_server.py:3-8`），`scheduler.py`
也复制了 `sglang.srt.managers.scheduler.run_scheduler_process`
的流程（`areal/experimental/inference_service/sglang/scheduler.py:121-152`）。

因此正确结论是：AReaL 没有全局 monkey patch SGLang 模块命名空间，但它依赖 SGLang
私有启动/调度结构和若干实例方法签名。`SGLangConfig.build_args` 只要求
`sglang>=0.5.10.post1`（`areal/api/cli_args.py:1887-1888`），而 `pyproject.toml` 实际 pinned
`sglang[tracing]==0.5.10.post1`。如果运行环境装了更高版本但内部字段变化，源码中未看到更细的签名检查。

另一个误区是：`AwexSchedulerBridge` 和 `PPSchedulerBridge` 都在 scheduler 创建后绑定，但 classic
`/update_weights_from_distributed` 主路径真正和 PP routing 强相关的是 `PPSchedulerBridge`。AWEX
是另一个实验性权重更新服务面，文章后面会单独讲它的风险。

## 2.5 本章小结

💡 小结

- AReaL 通过自定义 SGLang launcher/scheduler process 把 PP routing 插入 SGLang 进程。
- `PPSchedulerBridge` 是实例级 method wrapping，不是 module-level 全局 monkey patch。
- 这种方式侵入小，但依赖 SGLang 内部结构；上游版本变化仍是维护风险。

# 三、per-PP 权重更新组：它解决的是“SGLang PP event loop 下的 rendezvous 死锁”

## 3.1 设计哲学与核心问题

权重同步是 SGLang PP 支持中最核心的机制。每次训练 actor 更新后，PPOTrainer 会调用 actor 的 `update_weights`，再把
rollout engine 的版本号推进到新版本（`areal/trainer/rl_trainer.py:719-729`）。如果权重更新模式是
XCCL/NCCL，训练侧会把参数 tensor broadcast 到 SGLang。

PP=1 时，一个 group 覆盖所有 inference worker 没问题：trainer rank0 + 所有 SGLang TP/DP worker 一起
rendezvous。

PP>1 时就不成立。SGLang 的 PP scheduler 不是所有 PP rank 同时处理同一个 HTTP init 请求；PP rank 0 先处理，后续 PP
rank 依赖转发。如果 rank0 在等待一个包含 rank1 的 NCCL rendezvous，它会阻塞；rank1 又收不到 init 请求，于是死锁。这就是
per-PP-rank group 存在的根本原因。

## 3.2 源码入口与关键对象

```text
areal/engine/sglang_remote.py
  - SGLangBackend.build_init_weights_group_request：根据 group_name 后缀构造 per-PP payload
  - SGLangBackend.build_distributed_weight_update_requests：构造 update_weights_from_distributed 请求

areal/infra/remote_inf_engine.py
  - _init_weights_update_group_remote：向每个 SGLang server 发 init group 请求
  - _update_weights_from_distributed：向每个 SGLang server 发 update 请求

areal/experimental/inference_service/sglang/pp_bridge.py
  - _extract_pp_rank_from_group_name：从 update_weight_group_k 提取 k
  - _bind_tp_worker：非目标 pp_rank 写入 None sentinel 并跳过 group creation
  - _bind_model_runner：非目标 pp_rank 跳过 update/destroy
```

## 3.3 主流程拆解

训练侧初始化权重更新组时，最终会走到 rollout engine：

```text
training_engine.connect_engine(...)
  -> rollout_engine.init_weights_update_group(meta)
    -> RemoteInfEngine.init_weights_update_group(meta)
      -> _init_weights_update_group_remote(...)
        -> backend.build_init_weights_group_request(addr, server_idx, meta)
        -> POST /init_weights_update_group
```

`RemoteInfEngine` 的实现把 init 请求异步发给所有 server
address：`areal/infra/remote_inf_engine.py:886-943` 和 `:1359-1418`。

SGLangBackend 里分两条路径：

```text
if gen_parallel.pp_size > 1 and group_name endswith _{digit}:
    pp_rank = suffix
    n_servers = gen_world_size // (tp_size * pp_size)
    rank_offset = 1 + server_idx * tp_size
    world_size = n_servers * tp_size + 1
    payload includes pp_rank
else:
    instance_size = tp_size * pp_size
    rank_offset = 1 + server_idx * instance_size
    world_size = gen_world_size + 1
```

证据在 `areal/engine/sglang_remote.py:214-269`。

这里最重要的 shape/rank 不是 tensor shape，而是 group shape：

```text
例：rollout = sglang:d2p2t4

全体 inference worker = 2(DP server) × 2(PP) × 4(TP) = 16

PP=0 权重更新组：trainer + server0(pp0,tp0..3) + server1(pp0,tp0..3)
world_size = 1 + 2 × 4 = 9
rank_offset(server0)=1
rank_offset(server1)=1 + 1×4 = 5

PP=1 权重更新组：trainer + server0(pp1,tp0..3) + server1(pp1,tp0..3)
world_size = 9
rank_offset 同上
```

这就是为什么 per-PP 模式下 `rank_offset` 只乘 `tp_size`，而不是 `tp_size * pp_size`。测试也覆盖了
DP=2、PP=2、TP=2 时 `world_size=5`、server1 `rank_offset=3`
的情况（`tests/test_sglang_pp_unit.py:135-162`）。

SGLang 侧收到 init 请求后，`PPSchedulerBridge` 决定是否 join：

```text
recv_req.group_name = "update_weight_group_1"
pp_rank_from_name = 1

if model_runner.pp_rank != 1:
    model_runner._model_update_group[group_name] = None
    return success(skip)
else:
    call original tp_worker.init_weights_update_group(recv_req)
```

证据在 `areal/experimental/inference_service/sglang/pp_bridge.py:102-180`。

后续 update/destroy 也依赖同一个 sentinel：

- `update_weights_from_distributed` 如果发现 `_model_update_group[group_name] is None`，直接返回
  skip（`areal/experimental/inference_service/sglang/pp_bridge.py:188-209`）。
- `destroy_weights_update_group` 对 sentinel 也只 pop，不调用原
  destroy（`areal/experimental/inference_service/sglang/pp_bridge.py:215-234`）。

## 3.4 关键细节与误区澄清

最重要的误区：PP>1 时 fallback 到单 group
不是安全优化，而是潜在死锁路径。`SGLangBackend.build_init_weights_group_request` 只有在 group name
后缀能解析成数字时才走 per-PP path；否则仍走 `world_size = gen_parallel.world_size + 1`
的原始路径（`areal/engine/sglang_remote.py:214-269`）。`pp_bridge.py` 文档也说 group name 没有数字后缀时
fall through to original
method（`areal/experimental/inference_service/sglang/pp_bridge.py:19-23`）。

基于 FSDP 注释对单 group 死锁的解释（`areal/engine/fsdp_engine.py:1385-1393`），这意味着未来如果某个训练引擎使用自定义
group name 且 `pp_size>1`，可能静默进入已知风险路径。源码目前没有对 `pp_rank` 做范围校验；测试甚至构造了 `pp=2` 但后缀为 5、10
的场景并断言能解析（`tests/test_sglang_pp_unit.py:190-198`）。这不是功能缺陷的直接证明，但确实是维护风险：`update_weight_group_10`
在 `pp_size=2` 下没有匹配 stage。

## 3.5 本章小结

💡 小结

- SGLang PP 权重同步的核心协议是 `update_weight_group_{pp_rank}`。
- per-PP group 的 world size 是 `1 + dp_server_count × tp_size`，不是 `1 + dp × pp × tp`。
- `PPSchedulerBridge` 通过 sentinel 让非目标 PP rank 跳过 init/update/destroy，避免 PP event-loop
  rendezvous 死锁。
- 当前后缀解析偏宽松，非标准 group name 在 PP>1 下可能回落到风险路径。

# 四、训练引擎适配：同一个 SGLang PP，FSDP、Megatron、Archon 三条路并不相同

## 4.1 设计哲学与核心问题

SGLang 只是接收权重的一侧，真正的发送方是训练引擎。问题在于三类训练引擎对 PP 的理解不同：

- Megatron/Archon 可以真的按 pipeline stage 持有不同层。
- FSDP 在 AReaL 中不支持训练侧 PP；它只能从 rank0 组织完整权重，再广播给 SGLang 的每个 inference PP stage。

因此 AReaL 不能写一套完全统一的“训练 PP rank -> inference PP rank”逻辑，而是在各训练引擎里适配。

## 4.2 源码入口与关键对象

```text
areal/engine/fsdp_engine.py
  - _init_weight_update_from_distributed：PP>1 时由 rank0 枚举所有 inference PP groups
  - _update_bucket_weights_from_distributed_async：多 group 时顺序广播每个 bucket

areal/engine/megatron_engine.py
  - weight_update_group_name：按 mpu.get_pipeline_model_parallel_rank 命名
  - _init_weight_update_from_distributed：要求 train_pp_size == gen_pp_size

areal/experimental/engine/archon_weight_sync.py
  - WeightSyncState：每个 PP rank 的 group state
  - _init_per_pp_weight_update_groups：每个 PP-stage head 只创建自己的 group
  - _update_weights_per_stage：每个 PP-stage head 只发本 stage 参数
```

## 4.3 主流程拆解

### FSDP：训练侧无 PP，rank0 对 inference PP stage 逐个广播

FSDP allocation 明确拒绝
`pipeline_parallel_size > 1`（`areal/api/alloc_mode.py:274-283`）。测试里如果要保持 4 GPU 规模，会把
FSDP 的 PP 折进 DP：`fsdp:d{dp * pp}t{tp}`（`tests/test_sglang_pp_distributed.py:203-212`）。

但 FSDP 可以同步到 SGLang PP：

```text
FSDPEngine._init_weight_update_from_distributed(meta)
  -> gen_pp_size = meta.gen_allocation.parallel.pp_size
  -> if gen_pp_size > 1:
       _init_per_pp_weight_update_groups(meta, gen_pp_size)

_init_per_pp_weight_update_groups
  -> dist.get_rank()==0:
       for pp_rank in range(gen_pp_size):
         group_name = f"update_weight_group_{pp_rank}"
         rollout_engine.init_weights_update_group(pp_meta)
         init_custom_process_group(world_size=per_pp_world_size+1, rank=0)
```

证据在 `areal/engine/fsdp_engine.py:1377-1499`。

更新时，如果 `weight_update_groups` 数量大于 1：

```text
for group_name, group in zip(weight_update_group_names, weight_update_groups):
    pp_meta.nccl_group_name = group_name
    fut = rollout_engine.update_weights_from_distributed(pp_meta, param_specs)
    for tensor in named_tensors:
        dist.broadcast(tensor, src=0, group=group)
    fut.result()
```

证据在 `areal/engine/fsdp_engine.py:1301-1317`。

这说明 FSDP 路径是“rank0 对每个 SGLang PP group 重复发送 bucket”。它简单、兼容，但通信量会乘以 `gen_pp_size`。

### Megatron：训练 PP stage 与 SGLang PP stage 一一对应

Megatron 初始化时就设置：

```python
self.is_pp_head = (
    mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    and mpu.get_tensor_model_parallel_rank() == 0
)
self.weight_update_group_name = f"update_weight_group_{mpu.get_pipeline_model_parallel_rank()}"
```

对应 `areal/engine/megatron_engine.py:316-322`。

PP>1 时，它会先校验
`train_pp_size == gen_pp_size`（`areal/engine/megatron_engine.py:1627-1647`）。如果匹配，只有
pipeline parallel head 创建自己 stage 的 group：

```text
if self.is_pipeline_parallel_head():
    meta.nccl_group_name = self.weight_update_group_name
    fut = rollout_engine.init_weights_update_group(meta)
    init_custom_process_group(world_size=per_pp_world_size+1, rank=0)
    fut.result()
```

证据在 `areal/engine/megatron_engine.py:1648-1688`。

### Archon：类似 Megatron，但更明确地封装成 WeightSyncState

Archon 的 `WeightSyncState` 初始 group name 就是
`update_weight_group_{pp_rank}`（`areal/experimental/engine/archon_weight_sync.py:49-59`）。

PP>1 时：

- `_init_per_pp_weight_update_groups` 要求
  `train_pp_size == gen_pp_size`（`areal/experimental/engine/archon_weight_sync.py:155-168`）。
- 非 PP-head rank 不创建 group，只记录 group names 以便 cleanup
  逻辑统一（`areal/experimental/engine/archon_weight_sync.py:170-182`）。
- PP-stage head 只创建自己的
  `update_weight_group_{pp_rank}`（`areal/experimental/engine/archon_weight_sync.py:188-227`）。
- 更新时 `_update_weights_per_stage` 只在本 stage group 上发送本 stage
  参数（`areal/experimental/engine/archon_weight_sync.py:349-418`）。

一个重要细节是：`_update_weights_per_stage` 会先对每个参数调用 `_get_full_tensor(param)`，然后才判断非 head 是否
continue（`areal/experimental/engine/archon_weight_sync.py:382-386`）。对应测试明确说明，非 PP-head
也必须参与 DTensor all-gather，否则 head 会挂住（`tests/test_sglang_pp_unit.py:605-610`、`:697-728`）。

## 4.4 关键细节与误区澄清

误区一：SGLang PP 开了，不代表 FSDP actor 也支持 PP。`ModelAllocation.__post_init__` 对 FSDP backend
直接拒绝 `pipeline_parallel_size > 1`（`areal/api/alloc_mode.py:274-283`）。FSDP 能支持的是“同步到一个
PP>1 的 SGLang rollout”。

误区二：Megatron/Archon 的训练 PP size 可以和 SGLang PP size 不一致。源码明确 fail fast：Megatron 在
`areal/engine/megatron_engine.py:1627-1647` 抛 `ValueError`；Archon 在
`areal/experimental/engine/archon_weight_sync.py:155-168` 抛 `ValueError`。原因也很直接：如果
train_pp_size \< gen_pp_size，后面的 SGLang stage 没有训练源；如果 train_pp_size >
gen_pp_size，多出来的训练 stage 会创建 SGLang 永远不 join 的 group。

误区三：非 head rank 可以不碰参数。对 DTensor/FSDP2 来说，`full_tensor()` 可能触发 collective；非 head 跳过会导致其他
rank hang。Archon 的测试专门把这个行为锁住（`tests/test_sglang_pp_unit.py:605-728`）。

## 4.5 本章小结

💡 小结

- FSDP 是“无训练 PP，但可同步到 SGLang PP”：rank0 枚举所有 inference PP groups。
- Megatron/Archon 是“训练 PP 与 inference PP 一一对应”：每个 PP-stage head 只服务自己的 stage。
- `train_pp_size == gen_pp_size` 是 Megatron/Archon 的硬约束，不是性能建议。
- 非发送 rank 也可能必须参与参数 materialization，以满足 DTensor collective 语义。

# 五、完整主路径串联：一次 PPO 更新如何穿过 SGLang PP

## 5.1 完整调用栈

下面把前面机制串起来，按一次真实训练流程看：

```text
User:
  python examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
    rollout.backend=sglang:d1p2t2 actor.backend=megatron:d1p2t2 \
    scheduler.type=local

  │
  ├─ Step 1: 配置解析
  │     └─ PPOTrainer.__init__
  │        -> ModelAllocation.from_str(actor.backend / rollout.backend)
  │
  ├─ Step 2: SGLang server args 构造
  │     └─ SGLangConfig.build_args(tp_size=2, pp_size=2, ...)
  │        -> args["pp_size"] = 2
  │
  ├─ Step 3: rollout server 启动
  │     └─ RolloutController.initialize(server_args)
  │        -> worker replicas = rollout dp_size
  │        -> each worker gpu = tp_size * pp_size
  │        -> areal.experimental.inference_service.sglang.launch_server
  │
  ├─ Step 4: SGLang scheduler bridge 绑定
  │     └─ areal_run_scheduler_process
  │        -> Scheduler(... pp_rank ...)
  │        -> PPSchedulerBridge(...).bind()
  │
  ├─ Step 5: rollout 生成
  │     └─ RemoteInfEngine.agenerate
  │        -> POST /generate 到某个 server replica
  │        -> SGLang 内部自行完成 TP/PP 推理
  │
  ├─ Step 6: actor 更新后同步权重
  │     └─ PPOTrainer train loop
  │        -> actor.update_weights(versioned_meta)
  │        -> training engine 创建 update_weight_group_{pp_rank}
  │        -> rollout_engine.init_weights_update_group(pp_meta)
  │        -> rollout_engine.update_weights_from_distributed(pp_meta, param_specs)
  │        -> dist.broadcast(tensor, group=per_pp_group)
  │
  └─ Step 7: 保存 / checkpoint
        └─ trainer._save_hf / _save_recover_checkpoint
           与 SGLang PP group 无关，仍由 actor backend 决定格式
```

## 5.2 每一层做了什么

| 层                | 输入                    | 输出 / 状态变化                  | 通信                        | 显存影响                                    | 执行频率             |
| ----------------- | ----------------------- | -------------------------------- | --------------------------- | ------------------------------------------- | -------------------- |
| 配置解析          | backend string          | `rollout_alloc.parallel.pp_size` | 无                          | 无                                          | 初始化一次           |
| server args       | SGLangConfig + pp_size  | `args["pp_size"]`                | 无                          | 无                                          | 启动时一次           |
| RolloutController | alloc + scheduling spec | `replicas=dp_size`, `gpu=tp*pp`  | scheduler RPC               | 决定 server 占用 GPU 数                     | 初始化一次           |
| SGLang launch     | args                    | SGLang server process            | SGLang 内部分布式 init      | 参数/KV cache 分散到 TP/PP                  | server 启动一次      |
| Bridge bind       | Scheduler instance      | instance method wrappers         | 无                          | 无                                          | scheduler 初始化一次 |
| Generate          | ModelRequest            | ModelResponse/logprobs/version   | HTTP + SGLang 内部 PP 通信  | 推理激活/KV 按 SGLang PP/TP 分布            | 每个 rollout 请求    |
| Weight init       | WeightUpdateMeta        | per-PP process groups            | TCP store + NCCL init       | 少量 group state                            | 连接/初始化时        |
| Weight update     | ParamSpec + tensors     | SGLang 权重刷新                  | HTTP + NCCL broadcast       | bucket buffer + full tensor materialization | 每个训练 update      |
| Save/checkpoint   | actor/critic            | HF/DCP/recover checkpoint        | backend-specific barrier/io | checkpoint 峰值由训练 backend 决定          | checkpoint 周期      |

## 5.3 哪些逻辑不在主路径

- `areal/infra/utils/launcher.py` 中旧 SPMD generation PP 检查不是本文主路径；它对 `gen.pp_size > 1`
  直接抛 `NotImplementedError`（`areal/infra/utils/launcher.py:263-268`）。
- AWEX `/awex/*` endpoint 是实验性权重更新服务面，不是 classic
  `RemoteSGLangEngine.update_weights_from_distributed` 主路径。它在 `launch_server.py`
  注册（`areal/experimental/inference_service/sglang/launch_server.py:61-64`），但 classic 路径走
  `/init_weights_update_group` 和 `/update_weights_from_distributed`。
- save/load checkpoint 不因为 SGLang PP 改格式。PPOTrainer 在 actor 更新后调用 `_save_hf` 和
  `_save_recover_checkpoint`（`areal/trainer/rl_trainer.py:731-751`、`:1073-1120`）；SGLang
  只是被更新的 rollout server。
- `SGLangConfig.enable_ep_moe`、`dp_size` 等字段属于 SGLang 自己的内部并行/attention 配置，不等同于 AReaL 的
  `rollout.backend` 资源解析。文档也提示 internal backend configs 不影响 GPU
  allocation（`docs/en/reference/alloc_mode.md:114-119`）。

## 5.4 本章小结

💡 小结

- 一次 PPO 更新里，SGLang PP 影响 server 启动、generate 内部执行和分布式权重更新；不改变 checkpoint 格式。
- AReaL rollout dispatcher 仍按 DP replica 分发请求，不会把一个用户请求拆给所有 PP rank。
- classic weight update 和 AWEX 是两套服务面，不能混为同一条主路径。

# 六、关键数据流 / 状态流 / shape 流程

## 6.1 Tensor shape 变化：权重同步传的是参数 shape，不是训练 batch shape

SGLang PP 支持的关键 tensor 流不是训练 forward 的 `[batch, seq]`，而是权重更新 bucket：

```text
训练侧参数:
  name: str
  tensor: torch.Tensor, shape = param.shape, dtype = param.dtype

构造 ParamSpec:
  ParamSpec(name=name, shape=tuple(tensor.shape), dtype="bf16/fp32/...")

HTTP 请求:
  /update_weights_from_distributed
  payload.names  = [name0, name1, ...]
  payload.dtypes = [dtype0, dtype1, ...]
  payload.shapes = [shape0, shape1, ...]
  payload.group_name = "update_weight_group_k"

NCCL broadcast:
  dist.broadcast(tensor, src=0, group=update_weight_group_k)
```

`SGLangBackend.build_distributed_weight_update_requests` 明确把 `ParamSpec` 转成
`names/dtypes/shapes/group_name`（`areal/engine/sglang_remote.py:161-187`）。FSDP bucket 构造
`ParamSpec` 的位置在 `areal/engine/fsdp_engine.py:1277-1284`；Archon multi-group bucket 在
`areal/experimental/engine/archon_weight_sync.py:475-485`。

显存上真正要关注的是：

- DTensor/FSDP 参数需要 `full_tensor()` materialize 完整权重，FSDP 在 `_get_full_tensor` 里处理
  DTensor 和 CPU offload（`areal/engine/fsdp_engine.py:1247-1265`）。
- bucket 大小由 `meta.weight_chunked_mem_mb` 控制，FSDP 在 update 时换算成
  bytes（`areal/engine/fsdp_engine.py:1525-1526`），Archon 同样在 `_update_weights_per_stage`
  里使用（`areal/experimental/engine/archon_weight_sync.py:363-418`）。

## 6.2 Rank / Mesh / Process Group 变化

以 `sglang:d2p2t4` 为例：

```text
rollout world_size = 16
server replicas = d = 2
per server workers = p × t = 2 × 4 = 8

Per-PP weight update groups:

update_weight_group_0:
  rank0: training stage/head
  rank1-4:  server0, pp0, tp0-3
  rank5-8:  server1, pp0, tp0-3

update_weight_group_1:
  rank0: training stage/head
  rank1-4:  server0, pp1, tp0-3
  rank5-8:  server1, pp1, tp0-3
```

对应公式来自 `SGLangBackend.build_init_weights_group_request`：

```text
n_servers = gen_parallel.world_size // (tp_size * pp_size)
rank_offset = 1 + server_idx * tp_size
world_size = n_servers * tp_size + 1
```

源码在 `areal/engine/sglang_remote.py:231-258`。

训练侧的 rank 语义随 backend 不同：

- FSDP：只有 global rank0 创建所有 per-PP groups，非 rank0 不参与这些 SGLang update
  groups（`areal/engine/fsdp_engine.py:1455-1504`）。
- Megatron：每个 pipeline stage 的 PP head 创建自己的
  group（`areal/engine/megatron_engine.py:1648-1688`）。
- Archon：`pipeline_parallel_rank` 对应 `update_weight_group_{rank}`，每个 PP-stage head 只创建一个
  group（`areal/experimental/engine/archon_weight_sync.py:188-227`）。

## 6.3 状态切换：`WeightUpdateMeta` 与 `_model_update_group` 是关键状态

状态流可以分成训练侧和 SGLang 侧。

训练侧：

```text
进入 init:
  meta.gen_allocation 已包含 rollout backend / parallel strategy
  training engine 写入:
    meta.nccl_master_address
    meta.nccl_master_port
    meta.nccl_group_name = update_weight_group_k

执行 update:
  meta.nccl_group_name 决定这次 HTTP update 请求路由到哪个 PP stage group
  weight_update_group_names / groups 保存本进程拥有的 group
```

FSDP 在 `areal/engine/fsdp_engine.py:1510-1517` 更新 meta 的 master address、port、group
name；Archon 在 `WeightSyncState` 中持有 group aliases 和多 group
lists（`areal/experimental/engine/archon_weight_sync.py:49-59`）。

SGLang 侧：

```text
PPSchedulerBridge.bind:
  保存原始方法引用 _orig_tp_init / _orig_update / _orig_destroy
  替换当前 tp_worker/model_runner 实例方法

init group:
  if local pp_rank != group suffix:
      model_runner._model_update_group[group_name] = None

update/destroy:
  if group_name in _model_update_group and value is None:
      skip
  else:
      call original method
```

源码在 `areal/experimental/inference_service/sglang/pp_bridge.py:98-234`。

这个状态是进程内、实例级的，不是线程安全的全局 registry；它依赖 SGLang scheduler/model_runner 的生命周期。bridge
没有恢复原始方法的逻辑，因为它绑定后随 SGLang server 生命周期存在。

## 6.4 本章小结

💡 小结

- SGLang PP 的关键 shape 是权重参数 shape 和 per-PP group shape，而不是训练 batch shape。
- per-PP group 把通信范围从 `dp×pp×tp + trainer` 缩到 `dp×tp + trainer`。
- `WeightUpdateMeta.nccl_group_name` 是跨训练侧和 SGLang 侧的路由键。
- `_model_update_group[group_name] = None` 是 SGLang 侧跳过非目标 PP stage 的 sentinel 状态。

# 七、核心机制深挖：patch、通信语义与配置归一化

## 7.1 Bridge 注入：零侵入接入还是维护风险？

它解决的问题是：AReaL 需要在 SGLang
`TpWorker.init_weights_update_group`、`ModelRunner.update_weights_from_distributed`、`destroy_weights_update_group`
前加 PP rank 判断，但不想改 SGLang 类定义。

实现方式是实例方法替换：

- `_bind_tp_worker` 保存 `_orig_tp_init = tp_worker.init_weights_update_group`，再赋值
  `tp_worker.init_weights_update_group = _pp_init_weights_update_group`（`areal/experimental/inference_service/sglang/pp_bridge.py:98-180`）。
- `_bind_model_runner` 同样替换 `model_runner.update_weights_from_distributed` 和
  `destroy_weights_update_group`（`areal/experimental/inference_service/sglang/pp_bridge.py:182-234`）。

它的好处是影响范围局限在当前 SGLang scheduler 进程内的实例。坏处是方法签名必须和当前 SGLang 版本匹配。源码中只看到
`sglang>=0.5.10.post1` 的版本下限检查（`areal/api/cli_args.py:1887-1888`），没有对
`TpWorker`/`ModelRunner` 方法签名做 runtime assertion。

## 7.2 通信原语：前向和反向是否对称？

这里要区分两种“通信”：

1. SGLang 推理内部的 PP 通信：由 SGLang 自己处理，AReaL 不在源码里显式参与每层 send/recv。
1. AReaL 权重同步通信：AReaL 显式调用 HTTP + NCCL broadcast。

权重同步不是 autograd 通信，没有 backward 对称性，也没有 gradient scaling。它的语义是：训练侧作为 `src=0`，向 SGLang 指定
group broadcast 参数 tensor。Archon bucket 里使用 async broadcast handles 后
wait（`areal/experimental/engine/archon_weight_sync.py:487-493`）；FSDP 单 group path 也构造
handles 并等待，multi-group path为了 SGLang PP 事件循环顺序而逐 group
同步完成（`areal/engine/fsdp_engine.py:1301-1317`）。

`RemoteInfEngine._update_weights_from_distributed` 先发 HTTP update 请求，再让训练侧 broadcast
tensor。SGLang server 收到请求后会在对应 update group 里接收。HTTP 请求本身对所有 server address
并发发出（`areal/infra/remote_inf_engine.py:1421-1456`）。

## 7.3 配置归一化：用户配置如何变成真实行为？

配置路径可以概括为：

```text
rollout.backend string
  -> ModelAllocation.parallel.pp_size
  -> PPOTrainer / RolloutController / launcher
  -> SGLangConfig.build_args(pp_size=...)
  -> areal SGLang launch_server argv
  -> SGLang server_args.pp_size
  -> areal_run_scheduler_process(... pp_rank ...)
  -> PPSchedulerBridge(server_args.pp_size)
```

影响行为的关键条件：

- `pp_size == 1`：`SGLangConfig.build_args` 不写 `pp_size`；`PPSchedulerBridge.bind` no-op。
- `pp_size > 1`：server args 包含 `pp_size`；bridge 包装实例方法；weight update group 必须使用数字后缀。
- `group_name` 没有数字后缀：即使 `pp_size > 1`，`SGLangBackend` 也走单 group fallback。
- `meta.use_lora == True` 且 distributed update：SGLangBackend 直接抛错，要求 disk
  mode（`areal/engine/sglang_remote.py:161-173`）。

## 7.4 本章小结

💡 小结

- PP bridge 是实例级方法替换：影响小，但依赖上游对象结构。
- AReaL 显式管理的是权重同步通信，不是 SGLang forward PP 通信。
- 配置真正改变行为的开关不只是 `pp_size>1`，还包括 `group_name` 是否能映射到 `pp_rank`。
- LoRA + SGLang distributed weight update 是明确不支持的组合。

# 八、显存、性能与通信分析

## 8.1 显存收益范围

| 内容                 | 是否因 SGLang PP 节省 | 原因                                                                     |
| -------------------- | --------------------- | ------------------------------------------------------------------------ |
| SGLang 推理参数      | ✅                    | 模型层按 PP stage 分散在 SGLang worker 上，具体切分由 SGLang 实现        |
| SGLang 推理激活/KV   | 部分 ✅               | 推理执行由 SGLang 管理，PP/TP 可降低单卡压力；AReaL 源码未展开内部 shape |
| 训练侧参数           | ❌                    | actor backend 自己决定；SGLang PP 不改变 FSDP/Megatron/Archon 参数存储   |
| optimizer state      | ❌                    | checkpoint/save 仍由 actor backend 管理                                  |
| 权重同步 full tensor | ❌/可能更高           | FSDP/Archon/Megatron 仍要 materialize / convert / bucket 后 broadcast    |
| 通信 buffer          | ⚠️                    | bucket 限制峰值，但 PP>1 会有多个 group / 多次 broadcast                 |
| 输入 rollout batch   | ❌                    | AReaL dispatcher 按 DP server replica 分发，不按 PP stage 切 batch       |

真正的显存收益主要在 SGLang 推理服务内部：一个大模型可通过 PP 分层放到更多 GPU 上。AReaL 自己的源码层面，权重同步反而需要额外小心 full tensor
和 bucket 峰值。FSDP 的 `_get_full_tensor` 可能把 DTensor materialize 成完整
tensor（`areal/engine/fsdp_engine.py:1247-1265`），再按 `weight_chunked_mem_mb` 分
bucket（`areal/engine/fsdp_engine.py:1525-1586`）。

## 8.2 通信开销

PP 支持带来的新增/变化通信主要在权重同步：

| 阶段              | 通信类型                  | group                   | 频率                   | 说明                                           |
| ----------------- | ------------------------- | ----------------------- | ---------------------- | ---------------------------------------------- |
| server launch     | SGLang 内部 init          | SGLang 自己             | server 启动            | AReaL 只传 `pp_size/tp_size/dist_init_addr`    |
| rollout generate  | HTTP + SGLang 内部 PP/TP  | SGLang 自己             | 每个请求               | AReaL 只向某个 server replica POST `/generate` |
| init weight group | HTTP + TCPStore/NCCL init | per-PP group            | 初始化/连接            | PP>1 时每个 stage 一个 group                   |
| update weight     | HTTP                      | 所有 server address     | 每个 bucket/每个 group | 触发 SGLang 进入 recv                          |
| update weight     | NCCL broadcast            | `update_weight_group_k` | 每个 bucket            | 训练侧 src=0，SGLang matching PP stage 接收    |
| pause/continue    | HTTP/RPC                  | rollout engine          | 每次 update 前后       | FSDP/Megatron/Archon update 前暂停生成，后恢复 |
| checkpoint        | barrier/io                | actor cpu group         | checkpoint 周期        | 与 SGLang PP 无直接关系                        |

FSDP multi-group path 是最容易产生串行瓶颈的地方：对每个 bucket，它遍历所有 per-PP groups，发 update 请求、broadcast
所有 tensor，然后等待 future（`areal/engine/fsdp_engine.py:1301-1317`）。这避免了 SGLang PP event loop
死锁，但也牺牲了并发空间。

Megatron/Archon 更自然：每个训练 PP-stage head 只负责自己的 stage group，因此通信更接近 stage-local；但要求 train
PP size 与 gen PP size 完全一致。

## 8.3 性能取舍

这个实现的核心取舍是：

- 用更多 NCCL group 和更复杂的 group routing，换掉单 group 死锁风险。
- FSDP 用“重复广播到每个 inference PP stage”的简单兼容方案，换取不要求 FSDP 自己支持训练 PP。
- Megatron/Archon 用“stage-to-stage”减少无关参数发送，但换来 `train_pp_size == gen_pp_size` 的硬约束。
- bridge 用实例级 wrapping 避免改 SGLang 源码模块，但换来上游内部 API 变动风险。
- distributed update 比 disk update 更快，但不支持 SGLang LoRA；LoRA 必须走 disk
  mode（`areal/engine/sglang_remote.py:161-173`）。

## 8.4 本章小结

💡 小结

- SGLang PP 的主要显存收益在推理服务内部，而不是训练侧 optimizer/参数状态。
- 权重同步是通信开销核心：PP>1 后从一个大 group 变成多个 stage-local groups。
- FSDP 路径简单但可能放大通信；Megatron/Archon 路径高效但要求 PP 对齐。
- bucket 化降低峰值显存，但不能消除 full tensor materialization 和多 group broadcast 成本。

# 九、配置项、边界条件与坑点

配置不能只看表，要看它改变了哪条源码路径。

| 配置 / 条件                                    | 影响源码路径                                                    | 行为变化                                                  | 风险 / 坑点                                    |
| ---------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------- |
| `rollout.backend=sglang:dXpYtZ`                | `alloc_mode.py` -> `rl_trainer.py` -> `SGLangConfig.build_args` | Y 写入 rollout `pp_size`；Y>1 时 server args 带 `pp_size` | 旧 SPMD launcher 仍阻断 SGLang PP              |
| `pp_size == 1`                                 | `cli_args.py:1881-1882`, `pp_bridge.py:68-70`                   | 不注入 `pp_size`，bridge no-op                            | group name 即使有 `_0` 也不进入 per-PP payload |
| `pp_size > 1` + `update_weight_group_k`        | `sglang_remote.py:214-258`                                      | per-PP group payload，携带 `pp_rank=k`                    | `k` 当前未做范围校验                           |
| `pp_size > 1` + 非数字后缀 group               | `sglang_remote.py:259-269`                                      | fallback 单 group                                         | 可能回到已知 deadlock-prone 路径               |
| `actor.backend=fsdp:...p2...`                  | `alloc_mode.py:274-283`                                         | 直接报错                                                  | FSDP 不支持训练 PP；只能 rollout SGLang PP     |
| Megatron/Archon `train_pp_size != gen_pp_size` | `megatron_engine.py:1627-1647`, `archon_weight_sync.py:155-168` | fail fast                                                 | 必须让 actor PP 与 rollout PP 对齐             |
| `use_lora=True` + SGLang distributed update    | `sglang_remote.py:161-173`                                      | 抛错，要求 disk mode                                      | 不能用 XCCL/NCCL 更新 LoRA                     |
| `sglang.enable_multithread_load`               | `cli_args.py:1853-1862`                                         | 写入 `model_loader_extra_config` JSON                     | 和 PP 本身无直接关系，只是 loader 额外参数     |
| `sglang.dp_size/enable_dp_attention`           | `SGLangConfig` 字段                                             | 传给 SGLang 自己内部                                      | 不等于 AReaL `rollout.backend` 的 `d` 资源分配 |
| `scheduler.type=local`                         | `rl_trainer.py:1024-1039`                                       | 走 RolloutController / V2                                 | 测试覆盖的主 E2E 路径                          |

几个硬边界值得单独强调：

- SGLang 版本：源码要求 `sglang>=0.5.10.post1`（`areal/api/cli_args.py:1887-1888`），依赖文件实际 pin 到
  `0.5.10.post1`。如果手动装更高版本，bridge 可能遇到内部结构变化。
- 多机：`RolloutControllerV2` 会根据 `tp_size * pp_size` 和 `n_gpus_per_node` 推导
  `nnodes_per_instance`，并要求可整除（`areal/experimental/inference_service/controller/controller.py:104-125`）。
- standalone eval：`docs/zh/tutorial/eval.md` 路径传
  `pp_size=rollout_alloc.parallel.pp_size`，但 `docs/en/tutorial/eval.md` 和
  `examples/math/gsm8k_eval.py` 附近存在写法差异；若要用 SGLang PP eval，建议以源码实际版本复核。

💡 小结

- 最小开启方式是 `rollout.backend` 中带 `p>1`，不是改 `sglang` 配置块。
- 真正进入 PP 权重同步还要求 group name 使用 `update_weight_group_{pp_rank}` 约定。
- FSDP、Megatron、Archon 的约束不同，不能把 rollout PP 等同于 actor PP。
- LoRA、旧 launcher、自定义 group name 是最容易踩坑的组合。

# 十、测试、示例与覆盖缺口

## 10.1 已覆盖路径：测试证明了什么

测试是理解这个特性意图的好材料。

| 测试 / 示例                                           | 覆盖行为                         | 说明                                                                    |
| ----------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------- |
| `tests/test_sglang_pp_unit.py:43-101`                 | PP=1 backward compatible         | 单 group world_size/rank_offset 不带 `pp_rank`                          |
| `tests/test_sglang_pp_unit.py:107-180`                | PP>1 per-PP-rank payload         | 校验 `world_size=n_servers*tp+1`、`rank_offset`、`pp_rank`              |
| `tests/test_sglang_pp_unit.py:214-250`                | allocation parsing               | `sglang:d2p2t2` 正确解析                                                |
| `tests/test_sglang_pp_unit.py:357-430`                | `pp_size` threading              | `build_args/build_cmd` 与 trainer/controller 源码窗口必须包含 `pp_size` |
| `tests/test_sglang_pp_unit.py:478-560`                | Archon per-stage init            | stage head 只创建自己的 group                                           |
| `tests/test_sglang_pp_unit.py:605-728`                | 非 head 仍参与 materialization   | 防止 DTensor all-gather hang                                            |
| `tests/test_sglang_pp_unit.py:731-838`                | train/gen PP mismatch            | Archon mismatch 必须 fail fast                                          |
| `tests/test_sglang_pp_distributed.py:145-195`         | 4-GPU E2E                        | Megatron/FSDP/Archon 与 `sglang:d1p2t2` 组合跑训练 smoke                |
| `tests/torchrun/run_sglang_pp_weight_sync.py:176-260` | group init 算术                  | head detection、group name、payload                                     |
| `tests/torchrun/run_sglang_pp_weight_sync.py:287-437` | 模拟 layer partition + broadcast | 不实例化真实引擎，验证通信基本语义                                      |

## 10.2 未覆盖风险：测试没有证明什么

| 风险点                                      | 当前是否有测试                       | 可能后果                                           |
| ------------------------------------------- | ------------------------------------ | -------------------------------------------------- |
| PP>1 非数字后缀 group name 应 fail fast     | 未看到                               | 可能 fallback 到单 group 并 hang                   |
| `pp_rank` 后缀越界                          | 未看到；现有测试还允许 5/10          | rendezvous 等不到对应 SGLang stage                 |
| AWEX 在 PP>1 下 result aggregation          | 未覆盖                               | 多个 PP rank push，proxy 只读一个 result，可能错乱 |
| `/awex/debug/*` 鉴权与路径限制              | 未覆盖                               | 可变更模型权重或写任意可访问路径                   |
| 真 SGLang 内部方法签名漂移                  | torchrun helper 明确避免实例化重依赖 | 上游版本变动时 bridge 运行期失败                   |
| FSDP 向每个 PP stage 广播全量参数的规模性能 | 未看到性能测试                       | 大模型 update latency 随 `pp_size` 放大            |
| runnable example YAML 直接展示 `sglang:p2`  | 基本缺失                             | 用户需要从测试命令/override 推断配置               |
| 多机 SGLang PP E2E                          | 未在当前测试中确认                   | cross-node rendezvous、端口、rank offset 风险      |

另外，code-review 子代理尝试运行 `pytest tests/test_sglang_pp_unit.py -q` 时，当前环境在 collection 阶段因缺少
`colorlog.formatter` 失败；因此本文没有声称本地完整测试通过，只把源码与测试文件本身作为证据。

💡 小结

- 单元测试很好地锁住了 rank math、配置传递和部分 deadlock 防护。
- E2E 覆盖了 4 GPU 的 Megatron/FSDP/Archon smoke path。
- 测试缺口集中在异常 group name、越界 pp_rank、AWEX PP 行为、多机与性能规模。

# 十一、局限性与已知优化点

## 11.1 硬约束

- FSDP backend 不支持训练 PP：`ModelAllocation.__post_init__`
  直接拒绝（`areal/api/alloc_mode.py:274-283`）。
- Megatron/Archon 要求 `train_pp_size == gen_pp_size`：否则 fail
  fast（`areal/engine/megatron_engine.py:1627-1647`，`areal/experimental/engine/archon_weight_sync.py:155-168`）。
- SGLang distributed weight update 不支持 LoRA：必须用 disk
  update（`areal/engine/sglang_remote.py:161-173`）。
- per-PP routing 依赖 group name 数字后缀；源码未看到范围校验。
- classic PP 权重同步是 runtime sync 机制，不改变 actor checkpoint 格式。

## 11.2 维护成本

- `launch_server.py` 和 `scheduler.py` 复制/适配 SGLang 内部流程，依赖私有 helper 和对象结构。
- `PPSchedulerBridge` 替换实例方法，依赖 `init_weights_update_group` /
  `update_weights_from_distributed` / `destroy_weights_update_group` 的签名。
- README 与 alloc_mode 文档能力表不一致，容易误导用户。
- AWEX debug endpoints 暴露在 SGLang FastAPI app 上；`awex.py` 注册
  `/awex/debug/get_parameters`、`/awex/debug/randomize_parameters`（`areal/experimental/inference_service/sglang/awex.py:79-93`），而
  `scheduler.py:110-118` 允许保存参数或随机化参数。若服务端口不是严格内网控制面，这是安全/运维风险。
- `AwexSchedulerBridge` 只用 `tp_rank==0` 和 `dp_rank==0` 选择 result
  pusher（`areal/experimental/inference_service/sglang/scheduler.py:39-49`），未限制
  `pp_rank`；而 `RpcProxy.collective_rpc_with_result` 只读一个
  result（`areal/experimental/inference_service/sglang/rpc_proxy.py:44-46`）。PP>1 下可能有多个
  PP stage 推结果，属于独立服务面的风险。

## 11.3 性能瓶颈

- FSDP 对所有 per-PP groups 顺序广播 bucket（`areal/engine/fsdp_engine.py:1301-1317`），通信量随
  `pp_size` 增长。
- `full_tensor()` materialization 仍可能造成训练侧峰值显存或 CPU/GPU 内存压力。
- init group 需要每个 PP stage 单独 TCPStore/NCCL init，初始化成本随 `pp_size` 增加。
- group init 里有 watchdog 日志检测 30/60/120
  秒阻塞（`areal/experimental/inference_service/sglang/pp_bridge.py:132-151`），这说明作者已经把 NCCL
  init hang 当成现实风险处理，但它只报警不自动恢复。

## 11.4 已知优化点

- 对 `group_name` 增加严格校验：`pp_size>1` 时要求 `update_weight_group_{k}` 且 `0 <= k < pp_size`。
- FSDP 如果能获取 SGLang PP stage 参数归属，可按 stage 过滤 `ParamSpec/tensor`，避免向每个 stage 广播全量参数。
- 对 AWEX result aggregation 加 `pp_rank` 聚合，或者仅允许一个全局 elected rank push。
- 对 bridge 绑定增加启动时 signature/attribute assertion，版本不匹配时 fail fast。
- 增加多机 SGLang PP E2E 和大模型 weight update latency/memory benchmark。
- 补一个用户可直接运行的 `sglang:p2` example YAML，减少只靠测试命令传播配置的成本。

💡 小结

- 当前实现优先解决“正确同步且不 deadlock”，而不是把通信和维护成本压到最低。
- 风险主要在异常配置 fail-fast、FSDP 重复广播、AWEX PP 聚合和 SGLang 上游内部 API 漂移。
- 下一步优化应该围绕严格校验、stage-aware 过滤、PP-aware 结果聚合和多机性能测试展开。

# 小结与展望

AReaL 的 SGLang Pipeline Parallel 支持，可以用四个关键词概括。

**关键词一：配置归一化。** 用户只写 `rollout.backend=sglang:dXpYtZ`，AReaL 通过 `ModelAllocation` 把 `p` 变成
`rollout_alloc.parallel.pp_size`，再由 `SGLangConfig.build_args` 写入 SGLang server
args。它把用户意图统一到 AReaL 的 engine allocation 模型里。

**关键词二：per-PP-rank group。** 这是整个实现的核心。PP>1 时不再使用覆盖所有 inference workers 的单一 NCCL
group，而是用 `update_weight_group_{pp_rank}` 拆成每个 PP stage 一个 group。这个设计直接针对 SGLang PP
scheduler event loop 的 rendezvous 死锁问题。

**关键词三：实例级 bridge。** `PPSchedulerBridge` 没有全局替换 SGLang 类，而是在 scheduler 创建后包装当前
`tp_worker` 和 `model_runner` 实例方法。它实现了较低侵入的 stage-aware routing，但仍要承担上游内部结构变化的维护成本。

**关键词四：按训练 backend 分化。** Megatron/Archon 与 SGLang PP 是 stage-to-stage 映射，要求 train/gen PP
size 一致；FSDP 没有训练 PP，所以由 rank0 枚举所有 inference PP groups。这种分化让同一个 rollout PP
能接入不同训练后端，但也带来不同通信成本和边界条件。

这个实现适合的场景，是 rollout 模型太大、单纯 TP/DP 难以放下，需要 SGLang
在推理侧按层切分，同时训练侧能够接受更复杂的权重同步拓扑。它不适合对更新延迟极端敏感、需要 LoRA distributed update、或者无法严格控制 SGLang
版本与服务端口暴露范围的场景。

与“只走 disk update”相比，XCCL/NCCL update 更快，但拓扑和 rank 语义复杂；与“单全局 group”相比，per-PP group
复杂，但避免了源码中明确指出的死锁路径。后续最值得继续走读的方向，是 AWEX 新权重更新服务如何与 PP/DP/TP 汇合、SGLang 上游 PP load/update
对缺失参数的真实处理、以及多机大模型下 per-PP group 的初始化与更新性能。
