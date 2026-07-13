# AIBridge Python v1 → v2 迁移指南

> 本指南帮助现有 `agn-sdk`（v1，Python）用户平滑迁移到 `aibridge`（v2，Rust 核心 + 五语言绑定）。
> v2 是破坏性升级（semver v2.0.0），API 风格有变化，但能力、provider、方法名基本保持一致。
> 适配对象：已有项目用 `agn-sdk` 的 Python 代码迁移到 `aibridge`。

---

## 1. 为什么要迁移

| 维度 | v1（agn-sdk） | v2（aibridge） |
|---|---|---|
| 实现语言 | Python（~19700 行） | Rust 核心 + 五语言原生绑定 |
| 支持语言 | 仅 Python | Python / JS-TS / Go / JVM / .NET |
| 性能 | Python 原生 | Rust（无 GIL、原生 async、零成本抽象） |
| 类型安全 | Pydantic + `**kwargs` | serde struct + Builder（编译期保证） |
| provider 数 | 38 个 | 38 个（全量迁移，零丢失） |
| 品牌 | agn-sdk | aibridge |

**迁移收益**：一套 API 五语言通用；Rust 核心更快更稳；显式 struct 替代 `**kwargs`，IDE 补全与编译期检查更好。

**迁移成本**：主要是包名、错误类名、参数传递方式的机械替换。方法名与能力基本不变，业务逻辑无需重写。

---

## 2. 速查表（一页纸变更总览）

| 变更点 | v1（agn-sdk） | v2（aibridge） | 影响 |
|---|---|---|---|
| 包名 | `agn-sdk` | `aibridge` | `pip install aibridge` |
| 导入 | `from agn import Client` | `from aibridge import Client` | 改 import |
| 错误基类 | `AGNError` | `AibridgeError` | 改异常类名 |
| 错误子类名 | `RateLimitError` 等 | `RateLimitError` 等 | **不变** |
| 错误 code | `RATE_LIMIT_ERROR`（大写） | `rate_limit_error`（snake_case） | 若解析 code 需改 |
| 参数传递 | `**kwargs` + `ChatOptions` | `Request` struct + Builder | 改调用方式 |
| Options 中间层 | `ChatOptions/ImageOptions/...` | 去除，直接 builder | 删掉 Options |
| 流式入口 | `chat(stream=True)` | `chat_stream(req)` 独立方法 | 拆成两个方法 |
| 翻译 | `client.translate(...)` | `transcribe(req, translate=true)` | 合并进 transcribe |
| 方法名 | `chat/image_generate/...` | **不变** | — |
| provider 名 | `openai/agnes/...` | **不变**（别名也保留） | — |
| 环境变量 | `AGN_API_KEY` | `AIBRIDGE_API_KEY`（兼容老 `AGN_*`） | 可不改 |

---

## 3. 包名与导入

### v1

```python
# 安装：pip install agn-sdk
from agn import Client, Router
from agn import AGNError, RateLimitError, ValidationError
from agn import ChatOptions, ImageOptions, SpeechOptions
```

### v2

```python
# 安装：pip install aibridge
from aibridge import Client
from aibridge import AibridgeError, RateLimitError, ValidationError
```

v2 去除了 `Options` 中间层，不再导出 `ChatOptions/ImageOptions/...`。`Router` 在 Rust 核心已实现，Python 绑定将随版本迭代暴露。

---

## 4. 错误类对照

错误分类完全一致，仅基类改名、code 格式调整。子类名保持不变，便于 `except` 子句平滑迁移。

### 4.1 类名对照

| v1 | v2 | 说明 |
|---|---|---|
| `AGNError` | `AibridgeError` | 基类（改名） |
| `AuthenticationError` | `AuthenticationError` | 认证失败 |
| `RateLimitError` | `RateLimitError` | 限流 |
| `ValidationError` | `ValidationError` | 参数校验 |
| `ModelNotFoundError` | `ModelNotFoundError` | 模型不存在 |
| `APIError` | `APIError` | Provider API 错误 |
| `NetworkError` | `NetworkError` | 网络错误 |
| `TimeoutError` | `TimeoutError` | 超时 |
| `UnsupportedCapabilityError` | `UnsupportedCapabilityError` | 能力不支持 |
| `ProviderNotFoundError` | `ProviderNotFoundError` | provider 不存在 |
| `VoiceNotAvailableError` | `VoiceNotAvailableError` | 音色不可用 |
| `ServiceUnavailableError` | `ServiceUnavailableError` | 服务暂不可用 |

### 4.2 code 字段格式变化

v1 的 `code` 是大写常量，v2 改为 snake_case（与 Rust 核心对齐）：

```python
# v1
except RateLimitError as e:
    assert e.code == "RATE_LIMIT_ERROR"

# v2
except RateLimitError as e:
    assert e.code == "rate_limit_error"  # snake_case
```

若代码里硬编码了大写 code 字符串，迁移时改成 snake_case。异常消息格式 v2 为 `[code] message`。

### 4.3 捕获写法迁移

```python
# v1
from agn import AGNError, RateLimitError

try:
    resp = await client.chat(...)
except RateLimitError as e:
    print(f"限流，{e.retry_after} 秒后重试")
except AGNError as e:
    print(f"其他 SDK 错误: {e}")

# v2（仅基类名改）
from aibridge import AibridgeError, RateLimitError

try:
    resp = await client.chat(...)
except RateLimitError as e:
    print(f"限流，{e.retry_after} 秒后重试")
except AibridgeError as e:
    print(f"其他 SDK 错误: {e}")
```

---

## 5. Client 构造对照

### v1

```python
client = Client(
    provider="agnes",
    api_key="your-key",
    base_url="https://api.agnes.ai/v1",
    timeout=300,
    max_retries=3,
    retry_delay=2.0,
)
await client.start()
```

### v2（Python 绑定）

```python
client = Client(
    provider="agnes",
    api_key="your-key",       # 关键字参数
    base_url="https://api.agnes.ai/v1",
)
await client.start()
```

v2 Python 绑定目前暴露 `api_key` 与 `base_url` 两个关键字参数；`timeout/max_retries/retry_delay` 走环境变量或后续版本暴露。Rust 核心的 `ClientOptions` 完整支持全部连接参数（见下文 Rust 示例）。

### v2（Rust 核心，完整参数）

```rust
use aibridge_core::client::Client;
use aibridge_core::config::ClientOptions;

let client = Client::new(
    "agnes",
    ClientOptions::builder()
        .api_key("your-key")
        .base_url("https://api.agnes.ai/v1")
        .timeout(300)
        .max_retries(3)
        .retry_delay(2.0)
        .build(),
)?;
client.start().await?;
```

### 免认证 provider

edge-tts 在 v1/v2 均免认证，构造时不传 `api_key`：

```python
# v1
client = Client(provider="edge-tts")

# v2
client = Client(provider="edge-tts")
```

v2 额外的 `echo` 是 mock 适配器（免认证，用于管线验证与单元测试）。

---

## 6. 参数传递范式（核心变化）

这是 v1→v2 最大的变化：**`**kwargs` + `Options` 中间层 → 显式 `Request` struct + Builder 链式调用**。

### 6.1 范式对照

**v1：三种传参方式混用**

```python
# 方式 A：独立参数
resp = await client.chat(model="gpt-4o", messages=[...], temperature=0.7, max_tokens=1000)

# 方式 B：Options 中间层（options 优先级高于独立参数）
opts = ChatOptions(temperature=0.7, max_tokens=1000, top_p=0.9)
resp = await client.chat(model="gpt-4o", messages=[...], options=opts)

# 方式 C：**kwargs 透传厂商特有参数
resp = await client.chat(model="gpt-4o", messages=[...], reasoning_effort="high")
```

**v2：统一用 Request builder（Rust 核心）**

```rust
let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("Hello!")])
    .temperature(0.7)
    .max_tokens(1000)
    .top_p(0.9)
    .extra("reasoning_effort", "high")  // 厂商特有参数走 extra
    .build();
let resp = client.chat(req).await?;
```

**v2：Python 绑定（已暴露的方法用关键字参数）**

```python
# Python 绑定对外仍是关键字参数风格，但去掉了 Options 中间层
resp = await client.chat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=1000,
)
```

### 6.2 Options 类全部去除

| v1 Options 类 | v2 替代 |
|---|---|
| `ChatOptions` | `ChatRequest::builder(model, messages)` |
| `ImageOptions` | `ImageRequest::builder(model, prompt)` |
| `VideoOptions` | `VideoRequest::builder(model, prompt)` |
| `EmbedOptions` | `EmbedRequest::builder(model, input)` |
| `TranscribeOptions` | `TranscribeRequest::builder(model, file)` |
| `SpeechOptions` | `SpeechRequest::builder(model, input, voice)` |

`ParameterMapping` 及预置映射常量（`OPENAI_COMPATIBLE_MAPPING` 等）在 v2 是 Rust 适配器内部实现细节，用户不再接触，无需迁移。

### 6.3 厂商特有参数透传

v1 用 `**kwargs`，v2 用 `extra` 字段（`HashMap<String, serde_json::Value>`）：

```rust
// v2 Rust：extra 透传
let req = ChatRequest::builder("gpt-4o", messages)
    .extra("reasoning_effort", "high")
    .extra("custom_flag", true)
    .build();
```

---

## 7. 各能力对照（v1 vs v2 示例）

下表给出六大能力的 v1 与 v2 代码对照。v2 侧同时给出 Rust 核心 API（完整能力）与 Python 绑定 API（已暴露的方法）。echo 适配器的示例可免认证直接运行；真实 provider 示例需替换为有效 API key。

### 7.1 文本对话 chat

**v1（Python）**

```python
from agn import Client

client = Client(provider="agnes", api_key="your-key")
await client.start()

resp = await client.chat(
    model="claude-3-opus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    temperature=0.7,
    max_tokens=1000,
)
print(resp.choices[0].message.content)
await client.close()
```

**v2（Python 绑定，echo 可直接运行）**

```python
from aibridge import Client

client = Client(provider="echo")  # 真实用例改 "agnes" + api_key
await client.start()

resp = await client.chat(
    model="echo-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    temperature=0.7,
    max_tokens=1000,
)
print(resp.choices[0].message.content)
await client.close()
```

**v2（Rust 核心）**

```rust
use aibridge_core::client::Client;
use aibridge_core::config::ClientOptions;
use aibridge_core::model::chat::{ChatMessage, ChatRequest};

let mut client = Client::new("agnes", ClientOptions::builder().api_key("your-key").build())?;
client.start().await?;

let req = ChatRequest::builder("claude-3-opus", vec![
    ChatMessage::system("You are a helpful assistant."),
    ChatMessage::user("Hello!"),
])
.temperature(0.7)
.max_tokens(1000)
.build();

let resp = client.chat(req).await?;
println!("{}", resp.choices[0].message.content.as_deref().unwrap_or(""));
client.close().await?;
```

### 7.2 图像生成 image_generate

**v1（Python）**

```python
result = await client.image_generate(
    model="dall-e-3",
    prompt="A beautiful sunset over the ocean",
    size="1024x1024",
    n=1,
)
print(result.data[0].url)
```

**v2（Rust 核心，Python 绑定逐步暴露中）**

```rust
use aibridge_core::model::image::ImageRequest;

let req = ImageRequest::builder("dall-e-3", "A beautiful sunset over the ocean")
    .size("1024x1024")
    .n(1)
    .build();

let result = client.image_generate(req).await?;
println!("{}", result.data[0].url.as_deref().unwrap_or(""));
```

v2 用 `ImageRequest::builder(model, prompt)` 链式构造，替代 v1 的独立参数 + `ImageOptions`。`reference_images` 在 v2 用 `FileInput` 枚举（`Path/Url/Bytes/Base64`）统一表达。

### 7.3 视频生成 video_create + video_poll

**v1（Python）**

```python
task = await client.video_create(
    model="video-gen-1",
    prompt="A cat walking through a forest",
    width=1280,
    height=720,
)
print(task.task_id)

# 轮询
status = await client.video_poll(task_id=task.task_id, model="video-gen-1")
print(status.status)
```

**v2（Rust 核心）**

```rust
use aibridge_core::model::video::VideoRequest;

let req = VideoRequest::builder("video-gen-1", "A cat walking through a forest")
    .width(1280)
    .height(720)
    .build();

let task = client.video_create(req).await?;
println!("{}", task.task_id);

// 轮询
let status = client.video_poll(&task.task_id, "video-gen-1").await?;
println!("{:?}", status.status);
```

`video_poll` 在 v1/v2 签名一致：`(task_id, model)`。`VideoRequest` 的 `mode` 字段用 `VideoMode` 枚举（`text2video/image2video/keyframes/multiimage`），替代 v1 的字符串字面量。

### 7.4 文字转语音 speech（TTS）

**v1（Python）**

```python
result = await client.speech(
    model="tts-1",
    input="你好，欢迎使用语音合成",
    voice="alloy",
    response_format="mp3",
    speed=1.0,
)
result.save_to_file("output.mp3")
```

**v2（Python 绑定，echo 可直接运行）**

```python
result = await client.speech(
    model="echo-tts",   # 真实用例改 "tts-1"
    input="hello",
    voice="alloy",
    response_format="mp3",
    speed=1.0,
)
with open("output.mp3", "wb") as f:
    f.write(result.audio_data)  # bytes
```

**v2（Rust 核心）**

```rust
use aibridge_core::model::audio::SpeechRequest;

let req = SpeechRequest::builder("tts-1", "你好，欢迎使用语音合成", "alloy")
    .response_format("mp3")
    .speed(1.0)
    .build();

let result = client.speech(req).await?;
let audio: Vec<u8> = result.audio_data.unwrap_or_default();
std::fs::write("output.mp3", &audio)?;
```

v2 的 `SpeechResult.save_to_file()` 辅助方法在 Python 绑定层尚未暴露，可直接写 `result.audio_data`（bytes）到文件。`voice` 在 v2 Rust 核心用 `VoiceSpec`（支持候选列表自动降级），Python 绑定接受字符串。

### 7.5 语音转文字 transcribe（ASR）

**v1（Python）**

```python
result = await client.transcribe(
    model="whisper-1",
    file="/path/to/audio.mp3",
    language="zh",
    prompt="这是一段关于人工智能的对话",
)
print(result.text)
```

**v2（Rust 核心，Python 绑定逐步暴露中）**

```rust
use aibridge_core::model::audio::TranscribeRequest;
use aibridge_core::model::image::FileInput;

let req = TranscribeRequest::builder("whisper-1", FileInput::path("/path/to/audio.mp3"))
    .language("zh")
    .prompt("这是一段关于人工智能的对话")
    .build();

let result = client.transcribe(req).await?;
println!("{}", result.text);
```

v2 用 `FileInput` 枚举统一表达音频输入（`Path/Url/Bytes/Base64`），替代 v1 的 `file` 参数接受多种类型。

**翻译（translate）变化**：v1 的 `client.translate(...)` 在 v2 合并进 `transcribe`，用 `translate(true)` 开关：

```rust
// v2 翻译为英文
let req = TranscribeRequest::builder("whisper-1", FileInput::path("/path/to/chinese.mp3"))
    .translate(true)  // 启用翻译模式
    .build();
let result = client.transcribe(req).await?;  // result.task == "translate"
```

### 7.6 文本嵌入 embed

**v1（Python）**

```python
result = await client.embed(
    model="text-embedding-3-small",
    input="hello world",
)
print(result.get_embeddings()[0][:5])
```

**v2（Rust 核心，Python 绑定逐步暴露中）**

```rust
use aibridge_core::model::common::EmbedRequest;

let req = EmbedRequest::builder("text-embedding-3-small", "hello world").build();
let result = client.embed(req).await?;
// result.data[0].embedding 为向量
```

---

## 8. 流式对话对照

### v1：`chat(stream=True)` 一个方法两种返回

```python
# v1：stream=False 返 ChatCompletion，stream=True 返 AsyncGenerator
resp = await client.chat(model="gpt-4o", messages=[...], stream=True)
async for chunk in resp:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### v2：独立的 `chat_stream` 方法

v2 把流式拆成独立方法 `chat_stream`，返回异步迭代器。语义与 v1 一致（逐块产出 `ChatCompletionChunk`）。

**v2（Python 绑定，echo 可直接运行）**

```python
stream = await client.chat_stream(
    model="echo-chat",
    messages=[{"role": "user", "content": "hello"}],
)
async for chunk in stream:
    # chunk.choices[0].delta.content 为增量文本
    print(chunk.choices[0].content, end="", flush=True)
```

**v2（Rust 核心）**

```rust
let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("Hello!")])
    .stream(true)
    .build();

use futures::StreamExt;
let mut stream = client.chat_stream(req).await?;
while let Some(chunk_result) = stream.next().await {
    let chunk = chunk_result?;
    if let Some(delta) = chunk.choices.first() {
        if let Some(content) = &delta.delta.content {
            print!("{}", content);
        }
    }
}
```

取消机制：Python 用 `asyncio` 取消协程即可（Rust 侧自动 drop stream → tokio task abort）。

---

## 9. Provider 列表对照

v2 全量迁移 v1 的 38 个 provider，名称与别名完全保留，**迁移零成本**。下表按类别列出。

### 9.1 完整 provider 列表

| 类别 | provider（主名） | 别名 |
|---|---|---|
| MVP | `openai` `agnes` `volcengine_cv` `gemini` | — |
| 兼容族 | `azure` | — |
| 聚合平台 | `siliconflow` `togetherai` `fireworksai` `cloudflareai` | `sf` / `together` / `fireworks` / `cloudflare`、`workersai` |
| 扩展模型 | `grok` `yi` `sensenova` `hunyuan` `groq` | `xaigrok` / `lingyiwanwu` / `shangtang` / `tencent_hunyuan` |
| 更多模型 | `deepseek` `stepfun` `mistral` `cohere` `perplexity` | `step` |
| 新兴模型 | `ideogram` `luma` `llama` | `ideo` / `dream-machine`、`lumalabs` / `meta-llama`、`meta` |
| 中文模型 | `qwen` `zhipu` `doubao` `ernie` `kimi` `minimax` | — |
| 独立协议 | `anthropic` `stability` `runway` `pika` `kling` | — |
| 音频 TTS | `edge-tts` `elevenlabs` `cartesia` | `edge_tts`、`edge` / `eleven`、`11labs` / `sonic` |
| 音频 ASR | `deepgram` `assemblyai` | `dg` / `assembly`、`aai` |
| Mock | `echo` | — |

### 9.2 v1 → v2 provider 兼容性

- **全部 38 个 provider 保留**：v1 用到的 provider 名在 v2 全部可用，无需改名。
- **别名保留**：v1 的别名（如 `sf` → `siliconflow`）在 v2 完全一致。
- **免认证 provider**：`edge-tts` 在 v1/v2 均免认证；v2 新增 `echo` mock 适配器（免认证，测试用）。
- **v1.3.0 起的"免费 Provider 免认证"特性保留**。
- **v1.1.0 起的"实时拉取模型列表"（`list_models` 调 provider `/models` 端点）保留**。
- **v1.3.3 起的"TTS 音色健康检查/推荐/自动降级"保留**。

### 9.3 新增 provider

v2 相对 v1 设计阶段新增 `echo` mock 适配器（阶段 0.6 管线验证用，常驻可用，不调网络）。真实 provider 与 v1.3.3 持平。

---

## 10. 环境变量对照

v2 引入新前缀 `AIBRIDGE_`，同时**兼容老 `AGN_` 前缀**（迁移期并存，平滑过渡）。

| 用途 | v1 | v2（新） | v2（兼容老） |
|---|---|---|---|
| 全局 API Key | `AGN_API_KEY` | `AIBRIDGE_API_KEY` | `AGN_API_KEY` |
| Provider 专属 Key | `AGN_OPENAI_API_KEY` | `AIBRIDGE_OPENAI_API_KEY` | `AGN_OPENAI_API_KEY` |
| 全局 Base URL | `AGN_BASE_URL` | `AIBRIDGE_BASE_URL` | `AGN_BASE_URL` |
| Provider 专属 URL | `AGN_OPENAI_BASE_URL` | `AIBRIDGE_OPENAI_BASE_URL` | `AGN_OPENAI_BASE_URL` |
| 轮询 URL（视频） | `AGN_{PROVIDER}_POLL_URL` | `AIBRIDGE_{PROVIDER}_POLL_URL` | `AGN_{PROVIDER}_POLL_URL` |

**优先级**：代码显式传入 > `AIBRIDGE_{PROVIDER}_*` > `AGN_{PROVIDER}_*` > `AIBRIDGE_API_KEY` > `AGN_API_KEY`。

迁移期可继续用老 `AGN_*` 环境变量，无需立即改。建议新项目用 `AIBRIDGE_*` 前缀。

---

## 11. 破坏性变更清单

以下是 v1 用法在 v2 必须修改的项，逐条核对：

| # | v1 用法 | v2 要求 | 必改 |
|---|---|---|---|
| 1 | `from agn import ...` | `from aibridge import ...` | 是 |
| 2 | `pip install agn-sdk` | `pip install aibridge` | 是 |
| 3 | `except AGNError` | `except AibridgeError` | 是 |
| 4 | `e.code == "RATE_LIMIT_ERROR"` | `e.code == "rate_limit_error"` | 是（若解析 code） |
| 5 | `chat(stream=True)` | `chat_stream(...)` 独立方法 | 是 |
| 6 | `client.translate(...)` | `transcribe(req, translate=true)` | 是 |
| 7 | `ChatOptions(...)` + `options=` | 删除，参数直接传 | 是 |
| 8 | `ImageOptions/VideoOptions/...` | 删除，用 builder | 是 |
| 9 | `**kwargs` 透传厂商参数 | `extra("key", value)` | 是（Rust）/ 关键字参数（Python） |
| 10 | `result.save_to_file()` | 手动写 `audio_data` 到文件 | 是（TTS，Python 绑定暂未暴露辅助方法） |
| 11 | `Client(provider, api_key, base_url, timeout, ...)` 全部位置/关键字参数 | `Client(provider, *, api_key, base_url)`（Python 绑定） | 是（timeout 等走环境变量） |
| 12 | `from agn import ChatOptions, ImageOptions, ...` | 删除（v2 无 Options 类） | 是 |

**不变项**（确认无需改）：
- 方法名：`chat/image_generate/video_create/video_poll/embed/transcribe/speech/list_models/list_voices/recommend_voices` 全部不变
- provider 名与别名：全部不变
- 错误子类名：全部不变
- 异步上下文管理器：`async with client:` 语义不变
- 响应模型字段名：`ChatCompletion.choices[0].message.content` 等基本对齐

---

## 12. 迁移检查清单

按顺序逐项检查，确保迁移完整：

- [ ] 依赖：`pip uninstall agn-sdk && pip install aibridge`
- [ ] 全局替换 import：`from agn` → `from aibridge`
- [ ] 错误基类：`AGNError` → `AibridgeError`（子类名不动）
- [ ] 错误 code 字符串：大写 → snake_case（若有硬编码）
- [ ] 删除所有 `ChatOptions/ImageOptions/VideoOptions/EmbedOptions/TranscribeOptions/SpeechOptions` 用法
- [ ] 流式调用：`chat(stream=True)` → `chat_stream(...)`
- [ ] 翻译调用：`translate(...)` → `transcribe(req, translate=true)`
- [ ] TTS 保存文件：`result.save_to_file(path)` → 手动写 `result.audio_data`
- [ ] Client 构造：`timeout/max_retries/retry_delay` 改走环境变量（或等后续版本暴露）
- [ ] 环境变量：可继续用 `AGN_*`，建议新代码用 `AIBRIDGE_*`
- [ ] provider 名与别名：无需改（全量保留）
- [ ] 运行测试：用 `echo` 适配器做免认证冒烟测试（`Client(provider="echo")`）

---

## 13. 迁移示例：完整脚本对照

下面是一个完整脚本的 v1→v2 迁移对照，覆盖 chat + 流式 + speech。

### v1 完整脚本

```python
import asyncio
from agn import Client, ChatOptions, AGNError, RateLimitError

async def main():
    client = Client(provider="agnes", api_key="your-key", base_url="https://api.agnes.ai/v1")
    async with client:
        # 对话
        opts = ChatOptions(temperature=0.7, max_tokens=1000)
        resp = await client.chat(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Hello!"}],
            options=opts,
        )
        print(resp.choices[0].message.content)

        # 流式
        stream = await client.chat(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "讲个笑话"}],
            stream=True,
        )
        async for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
        print()

        # TTS
        audio = await client.speech(model="tts-1", input="你好", voice="alloy")
        audio.save_to_file("hello.mp3")

asyncio.run(main())
```

### v2 完整脚本（Python 绑定）

```python
import asyncio
from aibridge import Client, AibridgeError, RateLimitError

async def main():
    client = Client(provider="agnes", api_key="your-key", base_url="https://api.agnes.ai/v1")
    async with client:
        # 对话（去掉 Options，直接传关键字参数）
        resp = await client.chat(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0.7,
            max_tokens=1000,
        )
        print(resp.choices[0].message.content)

        # 流式（独立方法 chat_stream）
        stream = await client.chat_stream(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "讲个笑话"}],
        )
        async for chunk in stream:
            print(chunk.choices[0].content or "", end="", flush=True)
        print()

        # TTS（手动写文件）
        audio = await client.speech(model="tts-1", input="你好", voice="alloy")
        with open("hello.mp3", "wb") as f:
            f.write(audio.audio_data)

asyncio.run(main())
```

---

## 14. 常见问题

**Q1：v2 Python 绑定为什么部分能力还没暴露？**
A：v2 是 Rust 核心 + 五语言绑定架构。Rust 核心已全部实现 38 provider + 六大能力；Python 绑定（PyO3）目前暴露 `chat/chat_stream/speech`，其余能力（`image_generate/video_*/embed/transcribe/list_models/list_voices`）随版本迭代暴露。急需完整能力的场景可直接用 Rust 核心，或等绑定层补全。

**Q2：迁移后性能会有提升吗？**
A：是。Rust 核心无 GIL、原生 async、零成本抽象，IO 密集场景吞吐与延迟显著优于 Python。Python 绑定通过 PyO3 直连 Rust 核心（无 JSON 序列化边界），真实 IO 在 tokio worker 线程执行，不阻塞 asyncio 事件循环。

**Q3：v1 和 v2 能并存吗？**
A：能。两者包名不同（`agn-sdk` vs `aibridge`）、import 路径不同（`agn` vs `aibridge`），可在同一环境并存。但建议迁移完成后卸载 v1 避免混淆。

**Q4：环境变量必须改吗？**
A：不必。v2 兼容老 `AGN_*` 前缀，迁移期可继续用。新项目建议用 `AIBRIDGE_*`。

**Q5：旧版 v1 会归档吗？**
A：会。v2 正式发版后，v1 仓库归档保留，PyPI 上的 `agn-sdk` 不再更新。建议迁移到 `aibridge`。

---

## 15. 相关文档

- [设计文档](design.md)：架构、数据模型、FFI 边界、错误处理
- [进度文档](PROGRESS.md)：当前实施进度与接手指南
- [README（v2）](index.md)：五语言快速开始 + provider 列表
- [原 README（v1）](https://github.com/WingkySky/AGN-SDK/blob/main/README.md)：Python v1 文档（归档参考）
