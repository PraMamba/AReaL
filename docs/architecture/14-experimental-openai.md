# OpenAI 兼容层与 Agent 接口

> 源码位置：`areal/experimental/openai/`, `areal/experimental/camel/`,
> `areal/experimental/workflow/` 文件数：15 个 | 总行数：5174 行

______________________________________________________________________

## 1. 模块定位与设计意图

本模块为 AReaL 提供了完整的 **OpenAI API 兼容层**，使得任何基于 OpenAI SDK / Anthropic SDK 构建的 Agent 框架（如
OpenAI Agents SDK、CAMEL、Claude Code CLI 等）能够 **无感知地** 将 AReaL 推理引擎作为后端，同时在交互过程中透明地收集 RL
训练所需的 token 级 logprob 和 reward 信号。

核心设计思想：

- **零改造接入**：外部 Agent 代码只需将 `base_url` 指向 AReaL 代理服务器即可，无需修改 任何 API 调用方式。
- **RL 信号透明采集**：每次推理请求的 `input_tokens`、`output_tokens`、`logprobs`、`versions` 均被自动记录到
  `InteractionCache`，供训练阶段直接使用。
- **会话级隔离**：通过 API Key 分发机制，实现多会话并行且互不干扰。
- **多协议支持**：同时兼容 OpenAI Chat Completions、Responses、Anthropic Messages 三种 API 格式。

______________________________________________________________________

## 2. 目录结构与文件清单

```
areal/experimental/
  openai/
    __init__.py               (  15 行)  公共导出: ArealOpenAI, InteractionWithTokenLogpReward, ...
    client.py                 (1359 行)  核心: ArealOpenAI 客户端、AsyncCompletions/ResponsesWithReward
    types.py                  ( 231 行)  数据模型: InteractionWithTokenLogpReward
    cache.py                  ( 268 行)  InteractionCache (OrderedDict + 父子关系 + 奖励折扣)
    tool_call_parser.py       ( 298 行)  工具调用解析 (SGLang / vLLM 双后端)
    proxy/
      __init__.py             (   9 行)  导出: OpenAIProxyClient, OpenAIProxyWorkflow
      server.py               ( 205 行)  路径常量、Pydantic 模型、SessionData、序列化
      proxy_rollout_server.py (1044 行)  FastAPI 代理 Rollout 服务器 (核心服务端)
      proxy_gateway.py        ( 750 行)  无状态代理网关 (路由 + 就绪队列 + 会话刷新)
      client_session.py       ( 337 行)  OpenAIProxyClient (HTTP 会话管理)
      workflow.py             ( 257 行)  OpenAIProxyWorkflow (RolloutWorkflow 实现)
      online_agent.py         (  88 行)  _OnlineAgent (在线训练模式等待器)
  camel/
    openai_model.py           ( 217 行)  CAMEL 框架 AReaL 后端适配器
  workflow/
    multi_turn_v2.py          (  96 行)  V2 多轮对话 Workflow (直接使用 ArealOpenAI)
```

______________________________________________________________________

## 3. 架构总览

```
                                     +-- 外部 Agent / SDK ---+
                                     |  OpenAI SDK           |
                                     |  Anthropic SDK        |
                                     |  CAMEL 框架           |
                                     +---------+-------------+
                                               |
                                   HTTP (Bearer Token 认证)
                                               |
                      +------------------------v--------------------------+
                      |            proxy_gateway.py (无状态)               |
                      |  - 路由表 routes[api_key] -> _SessionRoute        |
                      |  - 就绪队列 ready_workers (在线模式)               |
                      |  - LRU key pool (bounded, 驱逐淘汰)               |
                      |  - 会话刷新 (end old -> wait ready -> start new)   |
                      +----+-------------------+-------------------+------+
                           |                   |                   |
                      round-robin          readiness           forward
                           |              queue match              |
              +------------v-----------+   +-------v---------------v---------+
              | proxy_rollout_server   |   | proxy_rollout_server            |
              | (Worker A)             |   | (Worker B)                      |
              |                        |   |                                 |
              | FastAPI endpoints:     |   | FastAPI endpoints:              |
              | /chat/completions      |   | /chat/completions               |
              | /responses             |   | /responses                      |
              | /v1/messages           |   | /v1/messages                    |
              | /rl/start_session      |   | /rl/start_session               |
              | /rl/end_session        |   | /rl/end_session                 |
              | /rl/set_reward         |   | /rl/set_reward                  |
              | /export_trajectories   |   | /export_trajectories            |
              +--------+-------+------+   +----------+------+---------------+
                       |       |                      |      |
                       v       v                      v      v
              +--------+-------+------+   +-----------+------+--------------+
              |  ArealOpenAI 客户端   |   |  ArealOpenAI 客户端             |
              |                       |   |                                 |
              | chat.completions =    |   |  AsyncCompletionsWithReward     |
              |   AsyncCompletions    |   |  AsyncResponsesWithReward      |
              |   WithReward          |   |                                 |
              | responses =           |   |  InteractionCache              |
              |   AsyncResponses      |   |    (OrderedDict)               |
              |   WithReward          |   |    [id -> Interaction]          |
              |                       |   |                                 |
              | InteractionCache      |   |  tool_call_parser              |
              +--------+--------------+   +----------+---------------------+
                       |                              |
                       v                              v
              +--------+--------------+   +-----------+--------------------+
              |  AReaL InferenceEngine |   |  AReaL InferenceEngine        |
              |  (SGLang / vLLM)       |   |  (SGLang / vLLM)              |
              +------------------------+   +-------------------------------+
```

______________________________________________________________________

## 4. 核心组件详解

### 4.1 ArealOpenAI 客户端 (`client.py`, 1359 行)

ArealOpenAI (第 1212 行) 继承 `AsyncOpenAI`，是整个兼容层的入口点。它用 AReaL 自定义实现 替换了 OpenAI SDK
的两个核心资源：

```
ArealOpenAI(AsyncOpenAI)
  |
  +-- self.chat.completions = AsyncCompletionsWithReward (第 1250 行)
  |     覆盖 chat.completions.create()
  |
  +-- self.responses = AsyncResponsesWithReward (第 1238 行)
  |     覆盖 responses.create()
  |
  +-- self._cache = InteractionCache (第 1235 行)
  |     有序字典，存储所有交互记录
  |
  +-- 奖励管理方法:
        set_reward(id, reward)           按 ID 设置奖励 (第 1265 行)
        set_last_reward(reward)          对最近交互设奖励 (第 1271 行)
        apply_reward_discount(discount)  反向折扣传播 (第 1277 行)
        export_interactions(style)       导出交互 (第 1307 行)
```

**API 调用转换流程** (`AsyncCompletionsWithReward.create`, 第 538 行):

```
OpenAI SDK 调用
  |
  v
1. 消息归一化 (_ensure_message_dict_list, 第 80 行)
   - BaseModel -> dict (model_dump)
   - 递归归一化嵌套对象
  |
  v
2. 图像提取 (_extract_images_from_messages, 第 151 行)
   - 从 data URI 提取 base64
   - 为 tokenizer 生成 {"type":"image"} 占位
   - 为 vLLM 生成 {"image_url": {"url":"placeholder"}}
  |
  v
3. Prompt 构建 (两种模式)
   - "hf" 模式: apply_chat_template 全量编码 (第 612 行)
   - "concat" 模式: 与父交互拼接 (concat_prompt_token_ids_with_parent, 第 346 行)
     - 复用父交互的 token，只编码新增部分
     - 通过 EOS token 计数对齐父子边界
  |
  v
4. GenerationHyperparameters 构建 (第 717 行)
   - temperature, top_p, max_new_tokens, stop, frequency_penalty
   - stop_token_ids = {eos_token_id, pad_token_id}
  |
  v
5. ModelRequest 构建 + engine.agenerate() 调用 (第 730-741 行)
  |
  v
6. 输出处理
   - decode output_tokens_without_stop (第 742 行)
   - process_tool_calls 解析工具调用 (第 746-755 行)
   - 构建 ChatCompletion / ChatCompletionChunk 对象 (第 792 行)
  |
  v
7. 缓存更新
   - cache[id].completion = chat_completion
   - cache[id].model_response = response
   - cache[id].output_message_list = [message.model_dump()]
```

**concat 模式的 Prompt 拼接算法** (`concat_prompt_token_ids_with_parent`, 第 346 行):

该算法解决多轮对话中避免重复编码完整历史的问题。关键步骤：

1. 取父交互的 `input_tokens + output_tokens_without_stop + [eos_token_id]`
1. 将完整消息列表（父消息 + 父输出 + 当前消息）通过 `apply_chat_template` 编码
1. 统计父 token 中 EOS 出现次数 `parent_eos_num`
1. 在子 token 序列中定位第 `parent_eos_num` 个 EOS 的位置
1. 拼接结果 = `parent_tokens + all_tokens[child_tokens_truncate_idx + 1:]`

此方案确保跨不同 tokenizer 模型的兼容性。

**流式响应** (`_create_stream`, 第 808 行):

由于 AReaL 推理引擎不支持真正的流式输出，该方法通过模拟流式来兼容 streaming=True 的请求：

```
chunk 1: role="assistant", content=""          (角色声明)
chunk 2: content=output_text                   (完整文本，一次发出)
chunk 3..N: tool_calls[i] (如有)               (工具调用逐个发出)
chunk N+1: finish_reason + usage               (结束标志)
```

缓存在流式 generator 创建 **之前** 更新（第 770-782 行），确保即使客户端提前断开， 交互记录也已持久化。

### 4.2 InteractionWithTokenLogpReward (`types.py`, 231 行)

这是 RL 信号的核心数据结构（第 36 行），将 OpenAI API 交互与 token 级训练数据关联：

```
InteractionWithTokenLogpReward
  |
  +-- model_response: ModelResponse     引擎返回（含 input_tokens, output_tokens, logprobs）
  +-- reward: float                     该轮奖励值
  +-- parent: InteractionWithTokenLogpReward  父交互引用（多轮树结构）
  +-- messages: list[dict]              输入消息列表
  +-- output_message_list: list[dict]   输出消息列表
  +-- completion: ChatCompletion        Completions API 响应对象
  +-- response: Response                Responses API 响应对象
  +-- chat_template_type: str           "hf" 或 "concat"
  +-- _cache: dict[str,Tensor]          计算缓存
```

**to_tensor_dict()** 方法（第 143 行）将交互转换为训练可用的张量：

```
{
  "input_ids":      [batch=1, seq_len]    完整 token 序列
  "loss_mask":      [batch=1, seq_len]    0=输入(不计算loss), 1=输出(计算loss)
  "logprobs":       [batch=1, seq_len]    token 级 log 概率
  "versions":       [batch=1, seq_len]    模型版本标记
  "attention_mask": [batch=1, seq_len]    全 1 (无 padding)
  "rewards":        [batch=1]             标量奖励
}
```

在 concat 模式下（第 149 行），`to_tensor_dict()` 会递归合并父交互的张量数据：

```
父 logprobs + [0.0] * (input_len - parent_len) + 子 output_logprobs
父 loss_mask + [  0] * (input_len - parent_len) + [1] * output_len
```

这确保了多轮对话中，每一轮的 prompt 部分被 mask 掉，只有实际生成的 token 参与 loss 计算。

### 4.3 InteractionCache (`cache.py`, 268 行)

继承 `OrderedDict[str, InteractionWithTokenLogpReward]`（第 13 行），提供：

**自动父子关系建立**（`__setitem__`, 第 107 行）：

每次插入新交互时，自动遍历已有交互，按 **最长前缀匹配** 规则确定父节点：

```
已有交互按 messages 长度降序排序
for parent in sorted_interactions:
    parent_data = parent.messages + parent.output_message_list
    if is_prefix(parent_data, new_interaction.messages):
        new_interaction.parent = parent
        break
```

**奖励折扣传播**（`apply_reward_discount`, 第 55 行）：

按插入顺序的逆序遍历，执行几何折扣：

```
reversed_interactions = list(reversed(cache.values()))
for i, interaction in enumerate(reversed_interactions):
    current_reward = current_reward * turn_discount + interaction.reward
    interaction.reward = current_reward
```

**导出策略**（`export_interactions`, 第 178 行）：

- `concat` 模式：构建对话树，只返回叶子节点（无子节点的交互）
- `individual` 模式：返回所有已完成的交互

导出时会过滤未完成的交互（`output_message_list is None`，第 223 行）， 这是为了处理 Anthropic Agent SDK
可能发送的内部请求（如 Claude Code CLI 的 git 历史分析）。

### 4.4 工具调用解析 (`tool_call_parser.py`, 298 行)

支持 SGLang 和 vLLM 两种推理后端的工具调用解析，通过 fallback 策略自动选择：

```
process_tool_calls() (第 256 行)
  |
  +-- 尝试 1: _process_tool_calls_sglang() (第 61 行)
  |     使用 sglang.srt.function_call.FunctionCallParser
  |     使用 sglang.srt.parser.reasoning_parser.ReasoningParser
  |
  +-- 尝试 2: _process_tool_calls_vllm() (第 144 行)
  |     使用 vllm.tool_parsers.ToolParserManager
  |     使用 vllm.reasoning.ReasoningParserManager
  |
  +-- 都未安装: 跳过解析，返回原文
```

**SGLang 到 vLLM 解析器名称映射**（`_SGLANG_TO_VLLM_TOOL_PARSER`, 第 18 行）：

```
qwen/qwen25/qwen3/qwen3_xml  ->  qwen3_xml
qwen3_coder                  ->  qwen3_coder
hermes                        ->  hermes
llama3/llama3_json            ->  llama3_json
llama4_json                   ->  llama4_json
mistral                       ->  mistral
deepseek_v3                   ->  deepseek_v3
```

**推理内容分离**（`_detect_think_and_return_ori_think`, 第 34 行）：

在解析工具调用之前，先将 `<think>...</think>` 标签内的推理内容与正文分离：

```
输入: "<think>我需要计算...</think>\n<tool_call>..."
输出: (reasoning_text="<think>我需要计算...</think>",
       content_text="\n<tool_call>...")
```

分离后仅对 `content_text` 执行工具调用解析，最终拼接回 `reasoning_text + content_text`。

**Responses API 兼容**（`use_responses=True`）：

当来自 Responses API 时，工具调用返回 `ResponseFunctionToolCall` 而非
`ChatCompletionMessageFunctionToolCall`，格式差异：

```
Completions: {"type":"function", "id":"call_xxx", "function":{"name":..., "arguments":...}}
Responses:   {"type":"function_call", "id":"fc-xxx", "call_id":"call_xxx", "name":..., "arguments":..., "status":"completed"}
```

______________________________________________________________________

## 5. 代理服务器架构

### 5.1 proxy_rollout_server (`proxy_rollout_server.py`, 1044 行)

这是部署在每个推理 Worker 上的 FastAPI 服务器，是外部 SDK 请求到达 AReaL 引擎的最终桥梁。

**全局状态**（第 96-129 行）：

```
_engine: InferenceEngine          推理引擎实例 (通过 /create_engine 创建)
_openai_client: ArealOpenAI       OpenAI 兼容客户端 (引擎初始化后创建)
_session_cache: dict[str, SessionData]  活跃会话
_capacity: int                    可用容量 (通过 /grant_capacity 增加)
_admin_api_key: str               管理密钥 (初始为随机值，防止未配置时被访问)
_api_key_to_session: dict         会话 key -> session_id 映射
_session_to_api_key: dict         session_id -> 会话 key 反向映射
```

**端点清单**：

```
管理端点 (Admin Key 认证):
  POST /grant_capacity              增加可用会话容量
  POST /rl/start_session            启动新 RL 会话，分配会话 API Key
  POST /export_trajectories         导出已完成会话的轨迹数据

会话端点 (Session Key 认证):
  POST /chat/completions            OpenAI Chat Completions API
  POST /responses                   OpenAI Responses API
  POST /v1/messages                 Anthropic Messages API (通过 LiteLLM 适配)
  POST /rl/set_reward               设置某交互的奖励值
  POST /rl/end_session              结束 RL 会话

引擎管理端点 (无认证):
  GET  /health                      健康检查
  POST /configure                   配置随机种子
  POST /set_env                     设置环境变量
  POST /create_engine               创建推理引擎实例
  POST /call                        调用引擎方法
  POST /alloc_ports                 分配空闲端口
```

**认证模型**（双层 API Key）：

```
+-- Admin API Key (管理员密钥)
|     用于: start_session, grant_capacity, export_trajectories
|     来源: AgentConfig.admin_api_key
|     校验: HMAC 常量时间比较 (_require_admin_key, 第 177 行)
|     安全: 如果绑定 0.0.0.0 且使用默认 key，validate_admin_api_key 会拒绝
|
+-- Session API Key (会话密钥)
      用于: chat/completions, responses, v1/messages, set_reward, end_session
      来源: start_session 时随机生成 (secrets.token_urlsafe(32))
      校验: _require_session_key (第 188 行) 查 _api_key_to_session 映射
      生命周期: start_session -> end_session (或超时清理)
```

**Anthropic Messages 兼容**（第 757 行）：

通过 LiteLLM 的 `AnthropicAdapter` 实现双向转换：

```
Anthropic 请求 -> _translate_anthropic_to_openai_request() -> OpenAI 格式
                    |
                    v
            _call_client_create() (ArealOpenAI)
                    |
                    v
OpenAI 响应 -> _adapter.translate_completion_output_params() -> Anthropic 格式
```

对于流式请求，使用 `_adapter.translate_completion_output_params_streaming()` 将 OpenAI SSE 块转换为
Anthropic SSE 事件（message_start, content_block_delta 等）。

**会话生命周期**：

```
grant_capacity (容量+1)
       |
       v
start_session
  - 检查容量 > 0 (否则 429)
  - 定期清理过期会话 (_cleanup_stale_sessions, 第 364 行)
  - 生成 session_id = "{task_id}-{idx}"
  - 生成 session_api_key = secrets.token_urlsafe(32)
  - 创建 SessionData, 容量-1
       |
       v
chat/completions | responses | v1/messages (多次)
  - 通过 session_key 认证
  - 调用 ArealOpenAI -> InteractionCache 自动记录
       |
       v
set_reward (可选，多次)
  - 为指定交互设置奖励值
       |
       v
end_session
  - SessionData.finish() 标记完成
  - 返回 interaction_count
       |
       v
export_trajectories (Admin Key)
  - await SessionData.wait_for_finish()
  - apply_reward_discount + export_interactions
  - 移除会话，清理 API Key 映射
  - 序列化返回 (tensor_dict 或 string 格式)
```

### 5.2 proxy_gateway (`proxy_gateway.py`, 750 行)

无状态网关层，位于外部用户与多个 proxy_rollout_server Worker 之间：

**核心数据结构**：

```
routes: dict[api_key, _SessionRoute]       会话路由表
  - _SessionRoute: {worker_addr, session_id, pending_future}

ready_workers: asyncio.Queue[_ReadyWorkerEntry]  就绪 Worker 队列 (在线模式)
  - _ReadyWorkerEntry: {worker_addr, future}

known_keys: OrderedDict[api_key, None]     LRU Key 池 (bounded, 默认 4096)

rr_index: [int]                            round-robin 计数器
```

**会话分配策略** (`start_session`, 第 353 行)：

```
            +-- 请求携带已知 key 且有活跃路由? --+
            |                                    |
           YES                                  NO
            |                                    |
     REFRESH PATH                        REUSE / NEW PATH
     (第 383 行)                          (第 479 行)
            |                                    |
    1. 结束旧会话                         1. 尝试就绪队列
    2. 等待 ready worker                     (在线模式 pre-granted)
       (带超时)                           2. Round-robin fallback
    3. 在新 worker 上启动新会话               遍历所有 worker
    4. 保留同一 api_key                      直到一个返回 200
```

**在线模式工作流**（`wait_for_session`, 第 686 行）：

```
_OnlineAgent -> POST /internal/wait_for_session
                 {worker_addr: "http://worker-0:8000"}
                      |
                      v
               创建 Future, 放入 ready_workers 队列
                      |
                      v (阻塞等待，最长 1 小时)
                      |
         外部用户 -> POST /rl/start_session
                      |
                      v
               从 ready_workers 取出 entry
               在 entry.worker_addr 上 start_session
               记录路由: routes[api_key] = _SessionRoute(future=entry.future)
                      |
                      v
         外部用户完成交互 -> POST /rl/end_session
                      |
                      v
               future.set_result(CompletedSessionInfo)
                      |
                      v
         wait_for_session 解除阻塞，返回 session 凭证
                      |
                      v
         OpenAIProxyWorkflow 用凭证调用 export_trajectories
```

**LRU Key 驱逐**（`_track_key`, 第 253 行）：

```
known_keys 超过 key_pool_size (默认 4096) 时:
  1. 弹出最早的 key
  2. 如果有活跃路由，异步结束后端会话
  3. reject 该路由的 pending_future (防止 _OnlineAgent 挂起)
```

### 5.3 会话管理 (`server.py`, 205 行)

定义了代理架构的基础设施：

**路径常量**（第 179-188 行）：

| 常量                                 | 路径                        |
| ------------------------------------ | --------------------------- |
| `RL_START_SESSION_PATHNAME`          | `rl/start_session`          |
| `RL_END_SESSION_PATHNAME`            | `rl/end_session`            |
| `RL_SET_REWARD_PATHNAME`             | `rl/set_reward`             |
| `CHAT_COMPLETIONS_PATHNAME`          | `chat/completions`          |
| `RESPONSES_PATHNAME`                 | `responses`                 |
| `ANTHROPIC_MESSAGES_PATHNAME`        | `v1/messages`               |
| `GRANT_CAPACITY_PATHNAME`            | `grant_capacity`            |
| `EXPORT_TRAJECTORIES_PATHNAME`       | `export_trajectories`       |
| `INTERNAL_WAIT_FOR_SESSION_PATHNAME` | `internal/wait_for_session` |

**SessionData**（第 66 行）：

```
SessionData
  +-- session_id: str
  +-- _completions: InteractionCache     该会话的交互缓存
  +-- _completed: bool                   是否已完成
  +-- _completed_event: threading.Event  完成信号（用于异步等待）
  +-- _start_time / _last_access_time / _end_time  时间戳
  |
  +-- is_stale(timeout) -> bool          超时检测 (默认 3600s)
  +-- finish()                           标记完成，触发 Event
  +-- wait_for_finish() -> bool          异步轮询等待 (1s 间隔)
  +-- export_interactions()              apply_reward_discount + export
```

**序列化**（`serialize_interactions`, 第 129 行）：

- 有 tensor 数据时：序列化 `tensor_dict` (input_ids, logprobs, loss_mask 等)
- 无 tensor 数据时：序列化原始 `messages` + `output_message_list` + `reward`

两种格式都通过 `areal.infra.rpc.serialization` 进行序列化，支持 HTTP 传输。

### 5.4 OpenAIProxyClient (`client_session.py`, 337 行)

异步上下文管理器，封装与 proxy_rollout_server 的 HTTP 交互：

```python
async with OpenAIProxyClient(session, base_url, task_id, admin_api_key) as client:
    # __aenter__: POST /rl/start_session (Admin Key)
    #   -> 获取 session_id + session_api_key
    api_key = client.session_api_key  # Agent 用这个 key 调用 chat/completions

    await client.set_reward(id, 1.0)       # POST /rl/set_reward (Session Key)
    await client.set_last_reward(0.5)      # POST /rl/set_reward (Session Key, id=None)
    # __aexit__: POST /rl/end_session (Session Key)

interactions = await client.export_interactions()  # POST /export_trajectories (Admin Key)
```

重试策略（第 266 行）：指数退避，对 502/503/504/429/408 和连接错误自动重试， `start_session`
无限重试（`stop_never`），其他操作最多重试 10 次。

### 5.5 OpenAIProxyWorkflow (`workflow.py`, 257 行)

继承 `RolloutWorkflow`，是连接 Agent 执行与 RL 训练数据流的桥梁。 支持三种运行模式（第 84 行）：

```
+-------+----------+------------------------------------------+
| 模式   | Agent 运 | 描述                                      |
|       | 行方式    |                                          |
+-------+----------+------------------------------------------+
| inline | 同进程   | Agent.run() 在当前 event loop 内执行       |
|        |          | 传入 base_url + api_key + http_client     |
+-------+----------+------------------------------------------+
| subproc| 子进程   | Agent.run() 在 ProcessPoolExecutor 中执行  |
|        |          | 通过环境变量传递 OPENAI_BASE_URL/KEY       |
|        |          | + ANTHROPIC_BASE_URL/KEY                  |
+-------+----------+------------------------------------------+
| online | 外部用户 | _OnlineAgent 等待外部用户完成会话           |
|        |          | 不主动调用 Agent，只被动等待               |
+-------+----------+------------------------------------------+
```

**arun_episode 流程**（第 164 行）：

```
1. grant_capacity() -- 授权一个会话容量
2. 根据模式运行:
   - inline/subproc: start_session -> run_agent -> set_reward -> end_session -> export
   - online: _OnlineAgent.run() 等待外部用户 -> export
3. 返回 dict[str, InteractionWithTokenLogpReward]
```

______________________________________________________________________

## 6. 外部框架集成

### 6.1 CAMEL 框架适配 (`camel/openai_model.py`, 217 行)

`AReaLOpenAICompatibleModel`（第 71 行）继承 CAMEL 的 `BaseModelBackend`， 将 CAMEL 的 Agent 框架接入
AReaL 推理引擎：

```
AReaLOpenAICompatibleModel(BaseModelBackend)
  +-- _client: AsyncOpenAI (实际为 ArealOpenAI)
  +-- tokenizer: PreTrainedTokenizerFast
  +-- _run(): raise NotImplementedError (不支持同步)
  +-- _arun(): await self._client.chat.completions.create()
  +-- token_counter: AReaLTokenCounter (使用 HF tokenizer 计数)
```

`AReaLTokenCounter`（第 43 行）替代 OpenAI 的 tiktoken，使用 HuggingFace tokenizer 进行 token
计数，每条消息额外加 `tokens_per_message` 个 token，回复前加 3 个 token。

### 6.2 V2 多轮工作流 (`workflow/multi_turn_v2.py`, 96 行)

`MultiTurnWorkflow`（第 17 行）是直接使用 ArealOpenAI 客户端的多轮对话工作流：

```
arun_episode 流程:
  1. 创建 ArealOpenAI(engine, tokenizer)
  2. 循环 (最多 max_turns 轮):
     a. client.chat.completions.create(messages) -> ChatCompletion
     b. client.get_interaction(comp.id) -> 获取 token 数据
     c. reward_fn(prompt, response, input_tokens, output_tokens, **data)
     d. 如果 reward == 0 且未到最大轮数:
        - 追加 assistant 回复到 messages
        - 追加反思提示: "Your answer is either wrong or not parsable..."
        - discount *= turn_discount
  3. client.set_reward(comp.id, reward * discount)
  4. client.export_interactions()
```

这个工作流直接在同一进程中使用 ArealOpenAI 客户端，不经过 HTTP 代理， 适用于自包含的多轮评估场景。

______________________________________________________________________

## 7. 数据流与生命周期

### 7.1 完整的 RL 训练数据采集流

```
+---------+     +----------+     +-------+     +--------+     +----------+
| Dataset | --> | Workflow  | --> | Proxy | --> | ArealOA| --> | Engine   |
| Loader  |     | arun_     |     | Client|     | I 客户 |     | agenerate|
|         |     | episode   |     |       |     | 端     |     |          |
+---------+     +-----+----+     +---+---+     +----+---+     +-----+----+
                      |              |              |               |
                      |              |              v               |
                      |              |     InteractionCache         |
                      |              |     (自动建立父子关系)        |
                      |              |              |               |
                      v              v              v               v
                +-----+----+   +----+----+   +-----+-----+   +-----+----+
                | set_reward|   |end_     |   |export_    |   |ModelResp |
                | (per turn)|   |session  |   |interactions|   |input_ids |
                +-----+----+   +----+----+   +-----+-----+   |output_ids|
                      |              |              |         |logprobs  |
                      v              v              v         +----------+
                +-----+--------------+--------------+----+
                |     apply_reward_discount              |
                |     (反向几何折扣)                       |
                +-----+----------------------------------+
                      |
                      v
                +-----+----------------------------------+
                | to_tensor_dict()                       |
                | {input_ids, loss_mask, logprobs,       |
                |  versions, attention_mask, rewards}    |
                +----------------------------------------+
                      |
                      v
                   RL 训练 (GRPO / PPO / DAPO)
```

### 7.2 在线模式时序图

```
TrainController      ProxyWorkflow      ProxyGateway     Worker-0        外部用户
     |                    |                  |              |                |
     |  arun_episode()    |                  |              |                |
     +-------------------->                  |              |                |
     |                    | grant_capacity   |              |                |
     |                    +------------------------------------>             |
     |                    |                  |              |                |
     |                    | _OnlineAgent.run |              |                |
     |                    +----> wait_for_session           |                |
     |                    |      (阻塞)      +<--worker_addr|                |
     |                    |                  |              |                |
     |                    |                  |  放入 ready_workers 队列       |
     |                    |                  |              |                |
     |                    |                  |              |    start_session
     |                    |                  |<-----------------------------+
     |                    |                  |              |                |
     |                    |                  |  从队列取出   |                |
     |                    |                  +--start_session-->             |
     |                    |                  |              |                |
     |                    |                  |  返回 api_key |                |
     |                    |                  +------------------------------>
     |                    |                  |              |                |
     |                    |                  |              |  chat/completions
     |                    |                  |              |<---------------+
     |                    |                  |              |  (多轮交互)     |
     |                    |                  |              +--------------->
     |                    |                  |              |                |
     |                    |                  |              |  set_reward    |
     |                    |                  |              |<---------------+
     |                    |                  |              |                |
     |                    |                  |   end_session|                |
     |                    |                  |<-----------------------------+
     |                    |                  |              |                |
     |                    |     future.set_result(session_info)              |
     |                    |<----- 返回 CompletedSessionInfo |                |
     |                    |                  |              |                |
     |                    | export_trajectories             |                |
     |                    +------------------------------------>             |
     |                    |                  |              |                |
     |<---interactions----+                  |              |                |
     |                    |                  |              |                |
```

______________________________________________________________________

## 8. 关键设计决策与工程约束

### 8.1 API Key 双层认证

采用 Admin Key + Session Key 分离设计，而非 Session ID 路径嵌入，原因：

1. **SDK 兼容性**：OpenAI/Anthropic SDK 只支持修改 `api_key` 和 `base_url`，不支持在 URL 路径中嵌入会话标识。
1. **安全性**：Admin Key 使用 `hmac.compare_digest` 常量时间比较（第 183 行），防止计时攻击。 启动时验证是否使用默认 key（第
   284 行），在 0.0.0.0 绑定时拒绝默认 key。
1. **灵活性**：Gateway 的刷新路径可以复用同一个 api_key（`request.api_key`）， 外部用户无需感知后端 Worker 切换。

### 8.2 容量控制机制

每个 Worker 的 `_capacity` 变量（第 102 行）由 `grant_capacity` 端点显式增加， `start_session`
消耗容量。这个设计使得 Rollout Controller 能够精确控制并发度：

- 在线模式下，只有当 RL 训练管线 "准备好处理新数据" 时才授予容量
- 避免外部用户请求淹没系统，因为无容量时 start_session 返回 429

### 8.3 concat vs hf 模板模式

| 特性        | hf 模式                        | concat 模式                         |
| ----------- | ------------------------------ | ----------------------------------- |
| Prompt 构建 | 每轮完整 `apply_chat_template` | 复用父 token + 追加新 token         |
| 父子关系    | 无                             | 自动最长前缀匹配                    |
| 导出格式    | `individual` 仅                | `concat` (叶子节点) 或 `individual` |
| stop token  | 支持                           | 不支持（第 704 行警告）             |
| 适用场景    | 单轮或每轮独立                 | 多轮连续对话                        |
| 性能        | 每轮重新编码                   | 增量编码，节省 tokenization 开销    |

### 8.4 流式响应的缓存时序

缓存更新在 async generator 创建 **之前** 完成（第 770-782 行），这是一个关键设计决策：

> LiteLLM 的流式适配器在迭代原始 generator **之前** 就会生成初始 chunk（如 message_start），
> 如果客户端在此阶段断开，generator 内部的缓存更新代码永远不会执行。

### 8.5 Gateway 无状态设计

Gateway 只持有路由表和就绪队列，不存储任何会话数据或交互记录。 会话数据完全存储在各 Worker 的 SessionData 中。这使得：

- Gateway 可以无损重启（路由状态丢失后，已有的 session key 失效，但不会丢失数据）
- Worker 可以独立扩缩容
- 轨迹导出直接从 Worker 获取，不经过 Gateway

### 8.6 过期会话清理

`_cleanup_stale_sessions`（第 364 行）采用惰性清理策略：

- 仅在 `start_session` 调用时触发
- 最多每 60 秒清理一次
- 同时清理孤立的 API Key 映射（客户端崩溃后遗留）
- 默认超时 3600 秒（可通过 `AgentConfig.session_timeout_seconds` 配置）

______________________________________________________________________

> 文档生成信息：基于 `areal/experimental/openai/`, `areal/experimental/camel/`,
> `areal/experimental/workflow/` 共 15 个源文件、5174 行代码分析生成。 行号引用均来自源码的 `grep -n` 结果。
