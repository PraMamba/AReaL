# 数据集、奖励函数与工作流层

> 源码位置：`areal/dataset/`, `areal/reward/`, `areal/workflow/` 文件数：22 个 | 总行数：2519 行

______________________________________________________________________

## 1. 模块定位

本文档覆盖 AReaL 训练流水线中从"数据加载"到"奖励计算"再到"轨迹采集"的三个紧密协作子模块。它们构成了 RL 训练的核心数据路径：

```
+------------------+     +-----------------+     +------------------+
|  areal/dataset/  | --> | areal/workflow/ | --> |  areal/reward/   |
|  数据加载 & 预处理 |     | 轨迹采集 & 编排   |     |  奖励函数计算     |
|  (7 文件, 1046行) |     | (11文件, 1080行)  |     |  (4 文件, 243行)  |
+------------------+     +-----------------+     +------------------+
        |                        |                        |
        v                        v                        v
   HuggingFace              InferenceEngine          math_verify /
   datasets +               (SGLang/vLLM) +          正则匹配
   Processor                 SDK (OpenAI/           AsyncRewardWrapper
                             Anthropic/              (进程池并发)
                             LangChain)
```

三个子模块的职责边界清晰：

- **dataset**: 将原始数据集转化为训练所需格式（SFT 的 `input_ids + loss_mask`、RL 的 `messages`）
- **reward**: 判断模型生成结果是否正确（返回 `float`）
- **workflow**: 编排"tokenize -> generate -> reward"循环，输出可用于策略梯度更新的张量字典

______________________________________________________________________

## 2. 文件清单与行数

### 2.1 areal/dataset/ (7 文件, 1046 行)

| 文件                 | 行数 | 职责                                     |
| -------------------- | ---- | ---------------------------------------- |
| `__init__.py`        | 212  | 注册表 + 路由分发 + RDataset 适配        |
| `clevr_count_70k.py` | 240  | CLEVR 视觉计数数据集（SFT + RL）         |
| `geometry3k.py`      | 207  | 几何题数据集（SFT + RL），含图像处理     |
| `virl39k.py`         | 147  | 视觉推理数据集（仅 RL）                  |
| `torl_data.py`       | 100  | ToRL 工具使用数据集（仅 RL），含下载逻辑 |
| `hhrlhf.py`          | 78   | HH-RLHF 偏好数据集（RW + DPO）           |
| `gsm8k.py`           | 62   | GSM8K 数学数据集（SFT + RL）             |

### 2.2 areal/reward/ (4 文件, 243 行)

| 文件                 | 行数 | 职责                                 |
| -------------------- | ---- | ------------------------------------ |
| `__init__.py`        | 152  | MathVerifyWorker 单例 + 懒加载注册表 |
| `geometry3k.py`      | 41   | 几何题奖励（正则提取 + MathVerify）  |
| `clevr_count_70k.py` | 32   | CLEVR 计数奖励（正则精确匹配）       |
| `gsm8k.py`           | 18   | GSM8K 奖励（直接调用 MathVerify）    |

### 2.3 areal/workflow/ (11 文件, 1230 行)

| 文件                             | 行数 | 职责                                     |
| -------------------------------- | ---- | ---------------------------------------- |
| `__init__.py`                    | 28   | 懒加载注册三大工作流                     |
| `rlvr.py`                        | 178  | RLVRWorkflow：单轮 RL 核心工作流         |
| `vision_rlvr.py`                 | 167  | VisionRLVRWorkflow：视觉 RL 工作流       |
| `multi_turn.py`                  | 135  | MultiTurnWorkflow：多轮重试工作流        |
| `anthropic/__init__.py`          | 6    | Anthropic agent 导出                     |
| `anthropic/math_agent.py`        | 80   | Anthropic Messages API 单轮 agent        |
| `anthropic/claude_math_agent.py` | 162  | Claude Agent SDK + MCP 工具 agent        |
| `langchain/__init__.py`          | 13   | LangChain agent 导出                     |
| `langchain/math_agent.py`        | 183  | LangChain ChatOpenAI agent (单轮 + 工具) |
| `openai/math_agent.py`           | 171  | OpenAI Agents SDK (单轮/多轮/工具)       |
| `openai_agent/math_agent.py`     | 107  | OpenAI multi-agent handoff 编排          |

______________________________________________________________________

## 3. 核心数据结构与接口

### 3.1 数据集注册与路由

```
VALID_DATASETS (L15, dataset/__init__.py)
  = ["gsm8k", "clevr_count_70k", "geometry3k", "virl39k", "hh-rlhf", "torl_data"]

调用链:
  get_custom_dataset(dataset_config)          # L165, 公开 API
    |
    +-- is_single_controller()?
    |     Yes --> RDataset(path, type, ...)    # 远程数据服务
    |     No  --> _get_custom_dataset(...)     # L27, 本地加载
    |               |
    |               +-- path 中包含 "gsm8k"?  --> from .gsm8k import ...
    |               +-- path 中包含 "clevr"?  --> from .clevr_count_70k import ...
    |               +-- ...（逐条 if-elif）
    |               +-- 都不匹配? --> load_from_disk(path) 回退
```

**设计要点**：

- 使用 **lazy import**（条件分支内 `from .xxx import`），避免加载无关依赖
- 路径匹配采用 `"xxx" in path` 子串检测，而非精确匹配
- 兜底路径支持任意 `save_to_disk()` 格式的 HuggingFace Dataset

### 3.2 数据集输出格式

不同训练类型产出不同字段：

```
SFT 模式:
  {
    "input_ids":  [int, ...],       # 完整序列（prompt + answer）
    "loss_mask":  [0, 0, ..., 1, 1] # prompt 部分为 0，answer 部分为 1
  }

RL 模式:
  {
    "messages": [                    # OpenAI 格式消息列表
      {"role": "user", "content": "..."},
    ]
  }

视觉 RL 模式 (clevr/geometry3k/virl39k):
  {
    "messages":      str,            # apply_chat_template 后的完整文本
    "messages_chat": [{...}],        # OpenAI 多模态格式 (image_url + text)
    "images":        [bytes, ...],   # JPEG 压缩后的图像字节流
    "answer":        str             # 标准答案
  }

RW/DPO 模式 (hh-rlhf):
  {
    "chosen_ids":         [int, ...],
    "rejected_ids":       [int, ...],
    "chosen_loss_mask":   [0, ..., 1, ...],  # 仅 DPO
    "rejected_loss_mask": [0, ..., 1, ...]   # 仅 DPO
  }
```

### 3.3 奖励函数签名

所有 RLVR 奖励函数遵循统一签名（定义于 `areal/api/reward_api.py` L40）：

```python
def reward_fn(
    prompt: str,                  # 解码后的提示文本
    completions: str,             # 解码后的模型生成文本
    prompt_ids: list[int],        # prompt token IDs
    completion_ids: list[int],    # 生成 token IDs
    **kwargs,                     # 数据集的其余字段（answer 等）
) -> float                        # 返回 0.0 或 1.0
```

Agent 工作流的奖励函数签名更简洁：

```python
def math_reward_fn(completions: str, answer: str) -> float
```

### 3.4 工作流基类

```python
# areal/api/workflow_api.py L14
class RolloutWorkflow(ABC):
    @abstractmethod
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None
    # 返回 None 表示该轨迹被丢弃，不参与训练
```

工作流返回的张量字典：

```
{
  "input_ids":      [1, seq_len]     # int32, 完整序列
  "loss_mask":      [1, seq_len]     # int32, 0=prompt, 1=response
  "logprobs":       [1, seq_len]     # float32, prompt部分=0.0
  "versions":       [1, seq_len]     # int32, prompt部分=-1
  "attention_mask": [1, seq_len]     # bool, 全 1
  "rewards":        [1]              # float32, 标量奖励
}
```

______________________________________________________________________

## 4. 核心流程

### 4.1 RLVRWorkflow.arun_episode 流程 (L139, rlvr.py)

```
arun_episode(engine, data)
  |
  |-- [1] 动态加载奖励函数
  |       if isinstance(self.reward_fn, str):
  |           self.reward_fn = import_from_string(self.reward_fn)
  |           self.async_reward_fn = AsyncRewardWrapper(reward_fn)
  |
  |-- [2] Tokenize
  |       input_ids = self.get_input_ids_fn(
  |           self.data_extract_prompt_fn(data),  # 提取 messages
  |           self.tokenizer,
  |           self.enable_thinking                 # 是否启用思考标签
  |       )
  |
  |-- [3] 构造请求
  |       req = ModelRequest(
  |           rid=uuid.uuid4().hex,
  |           input_ids=input_ids,
  |           gconfig=self.gconfig.new(n_samples=1),
  |       )
  |
  |-- [4] _collect_samples(engine, req, prompt_str, data)
  |       |-- @session_context() 注册新会话
  |       |-- engine.agenerate(req)        # 推理引擎生成
  |       |-- _compute_rewards(resp, ...)  # 异步奖励计算
  |       |-- stats_tracker 记录指标
  |
  |-- [5] 组装输出张量
  |       seq       = input_tokens + output_tokens
  |       logprobs  = [0.0]*input_len + output_logprobs
  |       loss_mask = [0]*input_len + [1]*output_len
  |       return {k: v.unsqueeze(0) for k, v in res.items()}
```

### 4.2 MultiTurnWorkflow 多轮重试循环 (L58, multi_turn.py)

```
arun_episode(engine, data)
  |
  |-- 初始化: t=0, reward=0.0, discount=1.0
  |
  |-- while reward == 0.0 and t < max_turns:
  |       |
  |       |-- engine.agenerate(req)
  |       |-- async_reward_fn(prompt, completion, ...)
  |       |
  |       |-- if reward == 0.0 and t < max_turns:
  |       |       input_ids += output_tokens
  |       |       input_ids += [eos_token_id]        # 补 EOS
  |       |       input_ids += multi_turn_prompt_ids  # 追加重试提示
  |       |       discount *= turn_discount           # 衰减奖励
  |       |
  |       |-- t += 1
  |
  |-- reward = reward * discount  # 越晚答对，奖励越低
  |
  |-- 拼接所有轮次的 seq/logprobs/loss_mask/versions
```

**多轮提示构造**（`__init__` 中预计算，L42-56）：

```
预计算方式：
  s1 = tokenize("assistant: some random message.")
  s2 = tokenize(s1 + "user: Your answer is wrong... try again.")
  multi_turn_prompt_ids = s2[len(s1):]

这样避免了 encode-decode 不一致问题
```

### 4.3 VisionRLVRWorkflow 视觉处理 (L103, vision_rlvr.py)

继承 `RLVRWorkflow`，覆写 `arun_episode` 增加视觉处理：

```
arun_episode(engine, data)
  |
  |-- [1] Processor 处理多模态输入
  |       processed_input = processor(
  |           images=data["images"],
  |           text=data["messages"],
  |       )
  |       input_ids = processed_input["input_ids"]
  |       mm_token_type_ids = processed_input["mm_token_type_ids"]
  |
  |-- [2] 图像编码
  |       byte_images = image2base64(data["images"])
  |
  |-- [3] 构造带视觉数据的请求
  |       req = ModelRequest(
  |           input_ids=input_ids,
  |           image_data=byte_images,
  |           vision_msg_vllm=data.get("messages_chat"),
  |           processor=self.processor,
  |       )
  |
  |-- [4] generate + reward（复用父类逻辑）
  |
  |-- [5] 额外输出字段
  |       "mm_token_type_ids": [1, seq_len]
  |       "multi_modal_input": [{
  |           "pixel_values": tensor,
  |           "image_grid_thw": tensor  # 仅 Qwen 系列
  |       }]
```

### 4.4 Agent 工作流的代理模式

Agent 工作流不继承 `RolloutWorkflow`，而是实现独立的 `run()` 接口：

```
async def run(self, data: dict, **extra_kwargs) -> float | dict[str, float]
  |
  |-- extra_kwargs 来自 AReaL proxy:
  |       base_url:    代理服务器地址
  |       api_key:     会话级 API 密钥
  |       http_client: httpx.AsyncClient
  |
  |-- 构造 SDK 客户端（OpenAI/Anthropic/LangChain）
  |       --> 指向 AReaL 代理而非真实 API
  |
  |-- 调用 SDK 完成推理
  |
  |-- math_reward_fn(completion, answer) 计算奖励
  |       --> 使用 math_verify.parse() + verify()
```

______________________________________________________________________

## 5. 关键设计模式

### 5.1 MathVerifyWorker 单例模式 (L31-119, reward/__init__.py)

```
                    get_math_verify_worker()
                           |
                    _MATH_VERIFY_WORKER 为 None?
                      |           |
                     Yes          No
                      |           |
               创建 MathVerifyWorker()    返回缓存实例
                      |
                      v
         +---------------------------+
         |  MathVerifyWorker         |
         |  - gold_extraction_target |  ExprExtractionConfig + LatexExtractionConfig
         |  - pred_extraction_target |  同上
         |  - precision = 6          |  有效数字精度
         |  - timeout = 5.0s         |  线程安全超时
         +---------------------------+
                      |
               verify(response, ground_truth)
                      |
         +---------------------------+
         | ThreadPoolExecutor(1)     |  避免 signal.alarm()
         | _verify_impl:             |  主线程不安全问题
         |   parse(ground_truth)     |
         |   parse(response)         |
         |   math_verify.verify()    |
         +---------------------------+
                      |
              TimeoutError? --> return 0.0
              Exception?   --> return 0.0
```

**为何不直接用 `math_metric()`**：

- `math_metric()` 内部使用 `signal.alarm()` 做超时控制
- `signal.alarm()` 只能在主线程中调用
- RLVR 工作流在 `asyncio` 事件循环中运行，奖励函数通过 `ProcessPoolExecutor` 分发
- 因此改用 `concurrent.futures.ThreadPoolExecutor` + `future.result(timeout=)` 实现线程安全超时

### 5.2 AsyncRewardWrapper 进程池包装 (L62-183, api/reward_api.py)

```
AsyncRewardWrapper
  |
  |-- 类变量:
  |     _executors: dict[int, ProcessPoolExecutor]  # 按 max_workers 共享
  |     _lock: threading.Lock                        # 线程安全
  |
  |-- __init__(reward_fn, timeout=15s, max_workers=auto, max_retries=3)
  |     max_workers = (cpu_count // device_count) // 2
  |     共享 executor: _executors[max_workers]
  |
  |-- __call__(*args, **kwargs) -> float:
  |     for attempt in range(max_retries + 1):
  |       |-- run_in_executor(executor, partial(reward_fn, ...))
  |       |-- asyncio.wait_for(future, timeout=timeout_seconds)
  |       |-- TimeoutError?    --> retry 或 return 0
  |       |-- BrokenProcessPool? --> _recreate_executor() + retry
  |       |-- Exception?       --> retry 或 raise
  |
  |-- atexit 注册: 进程退出时 shutdown 所有 executor
```

**关键特性**：

- 奖励函数在**子进程**中执行，隔离 CUDA 上下文
- `max_workers` 自动按 `(cpu / gpu数) / 2` 计算
- 支持 `BrokenProcessPool` 自动重建
- 超时和崩溃均返回 `0.0`，不阻塞训练

### 5.3 懒加载注册表模式

`dataset/`, `reward/`, `workflow/` 三个包的 `__init__.py` 均采用相同的模块级 `__getattr__` 懒加载：

```python
# 以 workflow/__init__.py 为例 (L9-24)

_LAZY_IMPORTS = {
    "RLVRWorkflow":       "areal.workflow.rlvr",
    "MultiTurnWorkflow":  "areal.workflow.multi_turn",
    "VisionRLVRWorkflow": "areal.workflow.vision_rlvr",
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val    # 缓存到模块全局变量
        return val
    raise AttributeError(...)
```

**好处**：

- `from areal.workflow import RLVRWorkflow` 首次访问时才加载 `rlvr.py`
- 避免导入 `areal.workflow` 时连带加载 torch / transformers 等重量级依赖
- `globals()[name] = val` 确保后续访问不再触发 `__getattr__`

### 5.4 视觉数据集的分布式预处理 (clevr_count_70k.py, L135-150)

```
get_clevr_count_70k_sft_dataset(path, split, processor, max_length)
  |
  |-- dist.is_initialized()?
  |     |
  |    Yes:
  |     |-- num_proc = min(cpu_count, 16)
  |     |-- if RANK == 0:
  |     |       _do_preprocess(...)    # Rank 0 先处理，写入 HF 缓存
  |     |-- dist.barrier()              # 等待 Rank 0 完成
  |     |
  |    No:
  |     |-- num_proc = None            # 单进程模式（慢）
  |
  |-- _do_preprocess(...)              # 所有 rank 从 HF 缓存加载
```

**设计意图**：Rank 0 先执行数据预处理写入 HuggingFace 缓存，其余 rank 等待后直接从缓存读取，避免重复计算。

### 5.5 Agent 工作流的 SDK 适配层

四种 Agent SDK 共享同一架构模式，但各自独立实现：

```
+-------------------+    +-------------------+    +-------------------+    +-------------------+
| OpenAI SDK        |    | Anthropic SDK     |    | LangChain         |    | Claude Agent SDK  |
| (直接 HTTP)       |    | (Messages API)    |    | (ChatOpenAI)      |    | (MCP 工具)        |
+-------------------+    +-------------------+    +-------------------+    +-------------------+
| MathAgent         |    | MathAgent         |    | MathAgent         |    | MathToolAgent     |
|   单轮生成        |    |   单轮生成        |    |   单轮生成        |    |   MCP Calculator  |
| MultiTurnMathAgent|    |                   |    | MathToolAgent     |    |   多轮工具调用    |
|   多轮重试        |    |                   |    |   create_agent()  |    |                   |
| MathToolAgent     |    |                   |    |   工具调用        |    |                   |
|   OpenAI Agents   |    |                   |    |                   |    |                   |
+-------------------+    +-------------------+    +-------------------+    +-------------------+
        |                        |                        |                        |
        v                        v                        v                        v
   AsyncOpenAI            AsyncAnthropic            ChatOpenAI            ClaudeSDKClient
   client.chat            client.messages           llm.ainvoke           client.query
   .completions           .create(...)              (data["messages"])    (content)
   .create(...)                                                          + receive_response()
        |                        |                        |                        |
        +------------------------+------------------------+------------------------+
                                         |
                                AsyncRewardWrapper(math_reward_fn)
                                         |
                                  math_verify: parse() + verify()
```

______________________________________________________________________

## 6. 类继承与依赖关系

### 6.1 工作流继承树

```
RolloutWorkflow (ABC)                       # areal/api/workflow_api.py L14
  |
  +-- RLVRWorkflow                          # areal/workflow/rlvr.py L49
  |     |
  |     +-- VisionRLVRWorkflow              # areal/workflow/vision_rlvr.py L26
  |           (覆写: arun_episode,
  |            _compute_rewards,
  |            _collect_samples)
  |
  +-- MultiTurnWorkflow                     # areal/workflow/multi_turn.py L19

AgentWorkflow (ABC, deprecated)             # areal/api/workflow_api.py L63
  (Agent 工作流不再需要继承此类，
   任何实现 run() 方法的类即可)
```

### 6.2 奖励函数依赖图

```
areal/reward/__init__.py
  |
  +-- MathVerifyWorker (L31)
  |     |
  |     +-- math_verify.parser.parse()
  |     +-- math_verify.grader.verify()
  |
  +-- get_math_verify_worker() (L115) -- 单例工厂
  |
  +-- get_custom_reward_fn(path) (L15) -- 按名路由
  |     |
  |     +-- "clevr_count_70k" --> clevr_count_70k_reward_fn
  |     +-- "geometry3k"      --> geometry3k_reward_fn
  |
  +-- _LAZY_IMPORTS (L133)
        +-- "gsm8k_reward_fn"         --> areal.reward.gsm8k
        +-- "geometry3k_reward_fn"    --> areal.reward.geometry3k
        +-- "clevr_count_70k_reward_fn" --> areal.reward.clevr_count_70k
```

### 6.3 数据集与奖励的配对关系

| 数据集          | 训练模式 | 奖励函数                       | 验证策略                         |
| --------------- | -------- | ------------------------------ | -------------------------------- |
| gsm8k           | SFT, RL  | `gsm8k_reward_fn`              | MathVerifyWorker 直接调用        |
| clevr_count_70k | SFT, RL  | `clevr_count_70k_reward_fn`    | 正则 `\[([0-9\.]+)\]` 精确匹配   |
| geometry3k      | SFT, RL  | `geometry3k_reward_fn`         | 正则提取 + MathVerifyWorker 回退 |
| virl39k         | RL       | 无内置（使用 geometry3k 兼容） | 同 geometry3k                    |
| hh-rlhf         | RW, DPO  | 无（偏好对比，不需要标量奖励） | N/A                              |
| torl_data       | RL       | 无内置（使用 gsm8k 兼容）      | 同 gsm8k                         |

______________________________________________________________________

## 7. 关键实现细节

### 7.1 图像处理流水线

三个视觉数据集使用不同的图像预处理策略：

```
clevr_count_70k:
  convert_image(image, max_pixels=336*336)
    --> 按面积比缩放
    --> 转 RGB
    --> JPEG 编码为 bytes

geometry3k:
  convert_image(image, fixed_width=512, fixed_height=512)  # SFT
  convert_image(image, fixed_width=448, fixed_height=448)  # RL
    --> CenterCrop 到固定尺寸
    --> 转 RGB
    --> JPEG 编码为 bytes

virl39k:
  convert_image(image, min_size=28, max_pixels=320*320)
    --> 小图放大到 min_size
    --> 大图按面积比缩小
    --> 转 RGB
    --> 返回 PIL Image（不编码为 bytes）
```

### 7.2 多模型 image_token 适配 (grep -n "image_token", clevr_count_70k.py)

```python
# L69-76, clevr_count_70k.py（同一模式在 geometry3k, virl39k 中重复出现）
image_processor_type = processor.image_processor.image_processor_type.lower()
if "qwen" in image_processor_type:
    image_token = "<|vision_start|><|image_pad|><|vision_end|>"
elif "gemma3" in image_processor_type:
    image_token = processor.boi_token
else:
    image_token = processor.image_token if processor is not None else "<image>"
```

支持的视觉模型系列：

- **Qwen-VL**: 使用 vision_start / image_pad / vision_end 三段标记
- **Gemma3**: 使用 `boi_token`（Beginning of Image）
- **其他** (LLaVA 等): 使用 `processor.image_token`

### 7.3 ToRL 数据集的下载与同步 (L43-59, torl_data.py)

```
prepare_torl_data(rank)
  |
  |-- rank == 0 且无 _SUCCESS 标记?
  |     Yes --> 下载 train.parquet + test.parquet
  |             写入 /tmp/areal/torl_data/_SUCCESS
  |
  |-- 所有 rank 轮询等待 _SUCCESS 文件
  |     超时 120 秒 --> TimeoutError
  |
  |-- 数据格式特殊:
  |     prompt = sample["prompt"]             # 已是 messages 列表
  |     answer = sample["reward_model"]["ground_truth"]
  |     answer = f"\\boxed{{{answer}}}"       # 包装为 boxed 格式
```

### 7.4 HH-RLHF DPO loss_mask 计算 (L51-68, hhrlhf.py)

```python
# 找到 chosen 和 rejected 的公共前缀长度
prompt_len = 0
for c, r in zip(chosen_ids, rejected_ids):
    if c == r:
        prompt_len += 1
    else:
        break

# loss_mask: prompt 部分为 0，response 部分为 1
chosen_loss_mask   = [0]*prompt_len + [1]*(len(chosen_ids) - prompt_len)
rejected_loss_mask = [0]*prompt_len + [1]*(len(rejected_ids) - prompt_len)
```

通过 token ID 逐位比对找出公共前缀（即 prompt），自动区分 prompt 和 response 部分。

### 7.5 OpenAI multi-agent handoff 模式 (openai_agent/math_agent.py)

```
build_math_agent()
  |
  |-- Problem Analyzer   -- 分析题目结构
  |-- Solution Specialist -- 分步求解
  |-- Refinement Agent    -- 修正错误
  |-- Verification Agent  -- 验证答案
  |
  |-- Main Agent (Math Problem Solver)
  |     handoffs = [
  |       analyze_problem   --> Problem Analyzer
  |       solve_problem     --> Solution Specialist
  |       refine_solution   --> Refinement Agent
  |       verify_solution   --> Verification Agent
  |     ]
  |
  |-- 执行流程:
  |     Main --> analyze_problem --> solve_problem
  |          --> refine_solution (如需) --> verify_solution
```

这是唯一使用 OpenAI Agents SDK `handoff()` 机制的实现，将数学问题拆分为分析/求解/修正/验证四个阶段，由不同的专家 agent 处理。

### 7.6 Claude Agent SDK + MCP 工具 (anthropic/claude_math_agent.py)

```python
# 使用 @tool 装饰器定义 MCP 工具 (L26-69)
@tool("add", "Add two numbers", {"a": float, "b": float})
async def add(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]}

# 创建 MCP server (L73-77)
calculator_server = create_sdk_mcp_server(
    name="calc", version="1.0.0",
    tools=[add, subtract, multiply, divide, power, sqrt],
)

# 运行时使用 ClaudeSDKClient (L151-158)
async with ClaudeSDKClient(options=options) as client:
    await client.query(content)
    async for message in client.receive_response():
        # 收集 TextBlock 文本
```

______________________________________________________________________

## 8. 扩展指南

### 8.1 添加新数据集

1. 在 `areal/dataset/` 下创建 `my_dataset.py`，实现 `get_my_dataset_rl_dataset()` 和/或
   `get_my_dataset_sft_dataset()`
1. 在 `areal/dataset/__init__.py` 的 `VALID_DATASETS` 列表中添加 `"my_dataset"`（L15）
1. 在 `_get_custom_dataset()` 中添加 `elif "my_dataset" in path:` 分支（L27-135）
1. RL 数据集必须输出 `{"messages": [...]}` 格式
1. SFT 数据集必须输出 `{"input_ids": [...], "loss_mask": [...]}` 格式

### 8.2 添加新奖励函数

1. 在 `areal/reward/` 下创建 `my_reward.py`，实现签名兼容的奖励函数
1. 在 `areal/reward/__init__.py` 中：
   - 添加到 `VALID_REWARD_FN` 列表（L12）
   - 在 `get_custom_reward_fn()` 中添加路由分支（L15-28）
   - 在 `_LAZY_IMPORTS` 中注册懒加载（L133-137）
   - 在 `__all__` 中导出函数名（L122-130）
1. 如需数学验证能力，使用 `get_math_verify_worker().verify(response, ground_truth)`

### 8.3 添加新工作流

**RLVR 类工作流**（继承 `RolloutWorkflow`）：

1. 在 `areal/workflow/` 下创建文件
1. 继承 `RolloutWorkflow`，实现 `async def arun_episode()`
1. 返回包含 `input_ids`, `loss_mask`, `logprobs`, `rewards` 等张量的字典
1. 在 `areal/workflow/__init__.py` 的 `_LAZY_IMPORTS` 中注册

**Agent 类工作流**（不需要继承任何基类）：

1. 在 `areal/workflow/<sdk_name>/` 下创建文件
1. 实现 `async def run(self, data: dict, **extra_kwargs) -> float`
1. 通过 `extra_kwargs` 获取 `base_url` / `api_key` / `http_client`
1. 使用 `AsyncRewardWrapper` 包装同步奖励函数

### 8.4 单轮 vs 多轮 vs Agent 工作流选型

```
需求判断:
  |
  |-- 纯文本单轮 RL?           --> RLVRWorkflow
  |-- 视觉单轮 RL?             --> VisionRLVRWorkflow
  |-- 需要多次重试直到答对?     --> MultiTurnWorkflow
  |-- 需要使用外部 SDK/工具?    --> Agent 工作流
  |     |
  |     +-- OpenAI 兼容 API?    --> openai/MathAgent
  |     +-- 需要工具调用?        --> openai/MathToolAgent 或 langchain/MathToolAgent
  |     +-- 需要多 agent 协作?   --> openai_agent/build_math_agent
  |     +-- Anthropic API?       --> anthropic/MathAgent
  |     +-- Claude + MCP 工具?   --> anthropic/MathToolAgent
```
