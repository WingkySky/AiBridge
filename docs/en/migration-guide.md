# AIBridge Python v1 → v2 Migration Guide

> This guide helps existing `agn-sdk` (v1, Python) users migrate smoothly to `aibridge` (v2, Rust core + five-language bindings).
> v2 is a breaking upgrade (semver v2.0.0). The API style changes, but capabilities, providers, and method names remain largely consistent.
> Target audience: existing projects migrating Python code from `agn-sdk` to `aibridge`.

---

## 1. Why Migrate

| Dimension | v1 (agn-sdk) | v2 (aibridge) |
|---|---|---|
| Implementation language | Python (~19,700 lines) | Rust core + five-language native bindings |
| Supported languages | Python only | Python / JS-TS / Go / JVM / .NET |
| Performance | Native Python | Rust (no GIL, native async, zero-cost abstractions) |
| Type safety | Pydantic + `**kwargs` | serde struct + Builder (compile-time guarantees) |
| Number of providers | 38 | 38 (full migration, zero loss) |
| Brand | agn-sdk | aibridge |

**Migration benefits**: one API shared across five languages; a faster and more stable Rust core; explicit structs replacing `**kwargs`, with better IDE completion and compile-time checks.

**Migration cost**: mainly mechanical replacement of package names, error class names, and parameter-passing styles. Method names and capabilities are largely unchanged, so business logic needs no rewrite.

---

## 2. Quick Reference (One-Page Change Overview)

| Change point | v1 (agn-sdk) | v2 (aibridge) | Impact |
|---|---|---|---|
| Package name | `agn-sdk` | `aibridge` | `pip install aibridge` |
| Import | `from agn import Client` | `from aibridge import Client` | Change import |
| Error base class | `AGNError` | `AibridgeError` | Change exception class name |
| Error subclass names | `RateLimitError`, etc. | `RateLimitError`, etc. | **Unchanged** |
| Error code | `RATE_LIMIT_ERROR` (uppercase) | `rate_limit_error` (snake_case) | Change needed if parsing code |
| Parameter passing | `**kwargs` + `ChatOptions` | `Request` struct + Builder | Change call style |
| Options intermediate layer | `ChatOptions/ImageOptions/...` | Removed, use builder directly | Remove Options |
| Streaming entry point | `chat(stream=True)` | `chat_stream(req)` as a separate method | Split into two methods |
| Translation | `client.translate(...)` | `transcribe(req, translate=true)` | Merged into transcribe |
| Method names | `chat/image_generate/...` | **Unchanged** | — |
| Provider names | `openai/agnes/...` | **Unchanged** (aliases also preserved) | — |
| Environment variables | `AGN_API_KEY` | `AIBRIDGE_API_KEY` (compatible with legacy `AGN_*`) | Optional |

---

## 3. Package Name and Import

### v1

```python
# Install: pip install agn-sdk
from agn import Client, Router
from agn import AGNError, RateLimitError, ValidationError
from agn import ChatOptions, ImageOptions, SpeechOptions
```

### v2

```python
# Install: pip install aibridge
from aibridge import Client
from aibridge import AibridgeError, RateLimitError, ValidationError
```

v2 removes the `Options` intermediate layer and no longer exports `ChatOptions/ImageOptions/...`. `Router` is already implemented in the Rust core; the Python binding will expose it in future releases.

---

## 4. Error Class Mapping

The error taxonomy is completely identical; only the base class is renamed and the code format is adjusted. Subclass names remain unchanged, allowing smooth migration of `except` clauses.

### 4.1 Class Name Mapping

| v1 | v2 | Description |
|---|---|---|
| `AGNError` | `AibridgeError` | Base class (renamed) |
| `AuthenticationError` | `AuthenticationError` | Authentication failure |
| `RateLimitError` | `RateLimitError` | Rate limiting |
| `ValidationError` | `ValidationError` | Parameter validation |
| `ModelNotFoundError` | `ModelNotFoundError` | Model not found |
| `APIError` | `APIError` | Provider API error |
| `NetworkError` | `NetworkError` | Network error |
| `TimeoutError` | `TimeoutError` | Timeout |
| `UnsupportedCapabilityError` | `UnsupportedCapabilityError` | Capability not supported |
| `ProviderNotFoundError` | `ProviderNotFoundError` | provider not found |
| `VoiceNotAvailableError` | `VoiceNotAvailableError` | Voice not available |
| `ServiceUnavailableError` | `ServiceUnavailableError` | Service temporarily unavailable |

### 4.2 code Field Format Change

In v1 the `code` was an uppercase constant; in v2 it changes to snake_case (aligned with the Rust core):

```python
# v1
except RateLimitError as e:
    assert e.code == "RATE_LIMIT_ERROR"

# v2
except RateLimitError as e:
    assert e.code == "rate_limit_error"  # snake_case
```

If your code hardcodes uppercase code strings, change them to snake_case when migrating. In v2 the exception message format is `[code] message`.

### 4.3 Migrating the Catch Syntax

```python
# v1
from agn import AGNError, RateLimitError

try:
    resp = await client.chat(...)
except RateLimitError as e:
    print(f"限流，{e.retry_after} 秒后重试")
except AGNError as e:
    print(f"其他 SDK 错误: {e}")

# v2 (only the base class name changes)
from aibridge import AibridgeError, RateLimitError

try:
    resp = await client.chat(...)
except RateLimitError as e:
    print(f"限流，{e.retry_after} 秒后重试")
except AibridgeError as e:
    print(f"其他 SDK 错误: {e}")
```

---

## 5. Client Construction Mapping

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

### v2 (Python binding)

```python
client = Client(
    provider="agnes",
    api_key="your-key",       # keyword argument
    base_url="https://api.agnes.ai/v1",
)
await client.start()
```

The v2 Python binding currently exposes two keyword arguments, `api_key` and `base_url`; `timeout/max_retries/retry_delay` go through environment variables or will be exposed in future releases. The Rust core's `ClientOptions` fully supports all connection parameters (see the Rust example below).

### v2 (Rust core, full parameters)

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

### Auth-free providers

edge-tts requires no authentication in both v1 and v2; do not pass `api_key` when constructing:

```python
# v1
client = Client(provider="edge-tts")

# v2
client = Client(provider="edge-tts")
```

The additional `echo` in v2 is a mock adapter (auth-free, used for pipeline validation and unit tests).

---

## 6. Parameter-Passing Paradigm (Core Change)

This is the biggest change from v1 to v2: **`**kwargs` + `Options` intermediate layer → explicit `Request` struct + Builder chained calls**.

### 6.1 Paradigm Comparison

**v1: three parameter-passing styles mixed**

```python
# Style A: standalone parameters
resp = await client.chat(model="gpt-4o", messages=[...], temperature=0.7, max_tokens=1000)

# Style B: Options intermediate layer (options take precedence over standalone parameters)
opts = ChatOptions(temperature=0.7, max_tokens=1000, top_p=0.9)
resp = await client.chat(model="gpt-4o", messages=[...], options=opts)

# Style C: **kwargs to pass through vendor-specific parameters
resp = await client.chat(model="gpt-4o", messages=[...], reasoning_effort="high")
```

**v2: unified use of the Request builder (Rust core)**

```rust
let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("Hello!")])
    .temperature(0.7)
    .max_tokens(1000)
    .top_p(0.9)
    .extra("reasoning_effort", "high")  // vendor-specific parameters go through extra
    .build();
let resp = client.chat(req).await?;
```

**v2: Python binding (exposed methods use keyword arguments)**

```python
# The Python binding remains keyword-argument style externally, but the Options intermediate layer is removed
resp = await client.chat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=1000,
)
```

### 6.2 All Options Classes Removed

| v1 Options class | v2 replacement |
|---|---|
| `ChatOptions` | `ChatRequest::builder(model, messages)` |
| `ImageOptions` | `ImageRequest::builder(model, prompt)` |
| `VideoOptions` | `VideoRequest::builder(model, prompt)` |
| `EmbedOptions` | `EmbedRequest::builder(model, input)` |
| `TranscribeOptions` | `TranscribeRequest::builder(model, file)` |
| `SpeechOptions` | `SpeechRequest::builder(model, input, voice)` |

`ParameterMapping` and the preset mapping constants (`OPENAI_COMPATIBLE_MAPPING`, etc.) are internal implementation details of the Rust adapters in v2. Users no longer interact with them, so no migration is needed.

### 6.3 Passing Through Vendor-Specific Parameters

v1 uses `**kwargs`; v2 uses the `extra` field (`HashMap<String, serde_json::Value>`):

```rust
// v2 Rust: pass-through via extra
let req = ChatRequest::builder("gpt-4o", messages)
    .extra("reasoning_effort", "high")
    .extra("custom_flag", true)
    .build();
```

---

## 7. Capability-by-Capability Mapping (v1 vs v2 Examples)

The tables below give v1 and v2 code comparisons for the six major capabilities. On the v2 side, both the Rust core API (full capability) and the Python binding API (exposed methods) are shown. The echo adapter examples can run directly without authentication; real provider examples must be replaced with a valid API key.

### 7.1 Text Chat: chat

**v1 (Python)**

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

**v2 (Python binding, echo runs directly)**

```python
from aibridge import Client

client = Client(provider="echo")  # for a real use case change to "agnes" + api_key
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

**v2 (Rust core)**

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

### 7.2 Image Generation: image_generate

**v1 (Python)**

```python
result = await client.image_generate(
    model="dall-e-3",
    prompt="A beautiful sunset over the ocean",
    size="1024x1024",
    n=1,
)
print(result.data[0].url)
```

**v2 (Rust core; Python binding being exposed incrementally)**

```rust
use aibridge_core::model::image::ImageRequest;

let req = ImageRequest::builder("dall-e-3", "A beautiful sunset over the ocean")
    .size("1024x1024")
    .n(1)
    .build();

let result = client.image_generate(req).await?;
println!("{}", result.data[0].url.as_deref().unwrap_or(""));
```

v2 uses `ImageRequest::builder(model, prompt)` for chained construction, replacing v1's standalone parameters + `ImageOptions`. In v2, `reference_images` is expressed uniformly with the `FileInput` enum (`Path/Url/Bytes/Base64`).

### 7.3 Video Generation: video_create + video_poll

**v1 (Python)**

```python
task = await client.video_create(
    model="video-gen-1",
    prompt="A cat walking through a forest",
    width=1280,
    height=720,
)
print(task.task_id)

# Polling
status = await client.video_poll(task_id=task.task_id, model="video-gen-1")
print(status.status)
```

**v2 (Rust core)**

```rust
use aibridge_core::model::video::VideoRequest;

let req = VideoRequest::builder("video-gen-1", "A cat walking through a forest")
    .width(1280)
    .height(720)
    .build();

let task = client.video_create(req).await?;
println!("{}", task.task_id);

// Polling
let status = client.video_poll(&task.task_id, "video-gen-1").await?;
println!("{:?}", status.status);
```

`video_poll` has an identical signature in v1 and v2: `(task_id, model)`. The `VideoRequest`'s `mode` field uses the `VideoMode` enum (`text2video/image2video/keyframes/multiimage`), replacing v1's string literals.

### 7.4 Text-to-Speech: speech (TTS)

**v1 (Python)**

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

**v2 (Python binding, echo runs directly)**

```python
result = await client.speech(
    model="echo-tts",   # for a real use case change to "tts-1"
    input="hello",
    voice="alloy",
    response_format="mp3",
    speed=1.0,
)
with open("output.mp3", "wb") as f:
    f.write(result.audio_data)  # bytes
```

**v2 (Rust core)**

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

The v2 `SpeechResult.save_to_file()` helper method is not yet exposed in the Python binding layer; you can write `result.audio_data` (bytes) to a file directly. In the v2 Rust core, `voice` uses `VoiceSpec` (supporting a candidate list with automatic fallback); the Python binding accepts a string.

### 7.5 Speech-to-Text: transcribe (ASR)

**v1 (Python)**

```python
result = await client.transcribe(
    model="whisper-1",
    file="/path/to/audio.mp3",
    language="zh",
    prompt="这是一段关于人工智能的对话",
)
print(result.text)
```

**v2 (Rust core; Python binding being exposed incrementally)**

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

v2 uses the `FileInput` enum to express audio input uniformly (`Path/Url/Bytes/Base64`), replacing v1's `file` parameter that accepted multiple types.

**Translation (translate) change**: v1's `client.translate(...)` is merged into `transcribe` in v2, using the `translate(true)` switch:

```rust
// v2 translate to English
let req = TranscribeRequest::builder("whisper-1", FileInput::path("/path/to/chinese.mp3"))
    .translate(true)  // enable translation mode
    .build();
let result = client.transcribe(req).await?;  // result.task == "translate"
```

### 7.6 Text Embedding: embed

**v1 (Python)**

```python
result = await client.embed(
    model="text-embedding-3-small",
    input="hello world",
)
print(result.get_embeddings()[0][:5])
```

**v2 (Rust core; Python binding being exposed incrementally)**

```rust
use aibridge_core::model::common::EmbedRequest;

let req = EmbedRequest::builder("text-embedding-3-small", "hello world").build();
let result = client.embed(req).await?;
// result.data[0].embedding is the vector
```

---

## 8. Streaming Chat Mapping

### v1: `chat(stream=True)` — one method, two return types

```python
# v1: stream=False returns ChatCompletion, stream=True returns AsyncGenerator
resp = await client.chat(model="gpt-4o", messages=[...], stream=True)
async for chunk in resp:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### v2: the separate `chat_stream` method

v2 splits streaming into a separate method `chat_stream`, which returns an async iterator. The semantics are identical to v1 (yielding `ChatCompletionChunk` chunk by chunk).

**v2 (Python binding, echo runs directly)**

```python
stream = await client.chat_stream(
    model="echo-chat",
    messages=[{"role": "user", "content": "hello"}],
)
async for chunk in stream:
    # chunk.choices[0].delta.content is the incremental text
    print(chunk.choices[0].content, end="", flush=True)
```

**v2 (Rust core)**

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

Cancellation mechanism: in Python, simply cancel the coroutine with `asyncio` (on the Rust side, the stream is automatically dropped → tokio task abort).

---

## 9. Provider List Mapping

v2 fully migrates all 38 providers from v1, preserving names and aliases completely — **zero migration cost**. The table below lists them by category.

### 9.1 Complete Provider List

| Category | provider (main name) | Aliases |
|---|---|---|
| MVP | `openai` `agnes` `volcengine_cv` `gemini` | — |
| Compatible family | `azure` | — |
| Aggregation platforms | `siliconflow` `togetherai` `fireworksai` `cloudflareai` | `sf` / `together` / `fireworks` / `cloudflare`, `workersai` |
| Extended models | `grok` `yi` `sensenova` `hunyuan` `groq` | `xaigrok` / `lingyiwanwu` / `shangtang` / `tencent_hunyuan` |
| More models | `deepseek` `stepfun` `mistral` `cohere` `perplexity` | `step` |
| Emerging models | `ideogram` `luma` `llama` | `ideo` / `dream-machine`, `lumalabs` / `meta-llama`, `meta` |
| Chinese models | `qwen` `zhipu` `doubao` `ernie` `kimi` `minimax` | — |
| Standalone protocols | `anthropic` `stability` `runway` `pika` `kling` | — |
| Audio TTS | `edge-tts` `elevenlabs` `cartesia` | `edge_tts`, `edge` / `eleven`, `11labs` / `sonic` |
| Audio ASR | `deepgram` `assemblyai` | `dg` / `assembly`, `aai` |
| Mock | `echo` | — |

### 9.2 v1 → v2 Provider Compatibility

- **All 38 providers preserved**: every provider name used in v1 is available in v2, with no renaming needed.
- **Aliases preserved**: v1's aliases (e.g., `sf` → `siliconflow`) are exactly the same in v2.
- **Auth-free providers**: `edge-tts` requires no authentication in both v1 and v2; v2 adds the `echo` mock adapter (auth-free, for testing).
- **The "free providers require no authentication" feature since v1.3.0 is preserved.**
- **The "real-time model list fetching" (`list_models` calls the provider's `/models` endpoint) since v1.1.0 is preserved.**
- **The "TTS voice health check / recommendation / automatic fallback" since v1.3.3 is preserved.**

### 9.3 New Providers

Relative to v1, v2 adds the `echo` mock adapter during the design phase (used for pipeline validation in phase 0.6, always available, makes no network calls). Real providers are on par with v1.3.3.

---

## 10. Environment Variable Mapping

v2 introduces a new prefix `AIBRIDGE_`, while **remaining compatible with the legacy `AGN_` prefix** (they coexist during the migration period for a smooth transition).

| Purpose | v1 | v2 (new) | v2 (legacy-compatible) |
|---|---|---|---|
| Global API Key | `AGN_API_KEY` | `AIBRIDGE_API_KEY` | `AGN_API_KEY` |
| Provider-specific Key | `AGN_OPENAI_API_KEY` | `AIBRIDGE_OPENAI_API_KEY` | `AGN_OPENAI_API_KEY` |
| Global Base URL | `AGN_BASE_URL` | `AIBRIDGE_BASE_URL` | `AGN_BASE_URL` |
| Provider-specific URL | `AGN_OPENAI_BASE_URL` | `AIBRIDGE_OPENAI_BASE_URL` | `AGN_OPENAI_BASE_URL` |
| Poll URL (video) | `AGN_{PROVIDER}_POLL_URL` | `AIBRIDGE_{PROVIDER}_POLL_URL` | `AGN_{PROVIDER}_POLL_URL` |

**Priority**: explicit values passed in code > `AIBRIDGE_{PROVIDER}_*` > `AGN_{PROVIDER}_*` > `AIBRIDGE_API_KEY` > `AGN_API_KEY`.

During the migration period you can keep using the legacy `AGN_*` environment variables, with no immediate change required. New projects are recommended to use the `AIBRIDGE_*` prefix.

---

## 11. Breaking Changes Checklist

The following are the items where v1 usage must be modified for v2 — verify each one:

| # | v1 usage | v2 requirement | Must change |
|---|---|---|---|
| 1 | `from agn import ...` | `from aibridge import ...` | Yes |
| 2 | `pip install agn-sdk` | `pip install aibridge` | Yes |
| 3 | `except AGNError` | `except AibridgeError` | Yes |
| 4 | `e.code == "RATE_LIMIT_ERROR"` | `e.code == "rate_limit_error"` | Yes (if parsing code) |
| 5 | `chat(stream=True)` | `chat_stream(...)` as a separate method | Yes |
| 6 | `client.translate(...)` | `transcribe(req, translate=true)` | Yes |
| 7 | `ChatOptions(...)` + `options=` | Removed, pass parameters directly | Yes |
| 8 | `ImageOptions/VideoOptions/...` | Removed, use builder | Yes |
| 9 | `**kwargs` pass-through of vendor parameters | `extra("key", value)` | Yes (Rust) / keyword arguments (Python) |
| 10 | `result.save_to_file()` | Manually write `audio_data` to a file | Yes (TTS; the Python binding does not yet expose the helper method) |
| 11 | `Client(provider, api_key, base_url, timeout, ...)` with all positional/keyword arguments | `Client(provider, *, api_key, base_url)` (Python binding) | Yes (timeout, etc. go through environment variables) |
| 12 | `from agn import ChatOptions, ImageOptions, ...` | Removed (v2 has no Options classes) | Yes |

**Unchanged items** (confirmed no change needed):
- Method names: `chat/image_generate/video_create/video_poll/embed/transcribe/speech/list_models/list_voices/recommend_voices` all unchanged
- provider names and aliases: all unchanged
- Error subclass names: all unchanged
- Async context manager: `async with client:` semantics unchanged
- Response model field names: `ChatCompletion.choices[0].message.content`, etc. are largely aligned

---

## 12. Migration Checklist

Check each item in order to ensure a complete migration:

- [ ] Dependency: `pip uninstall agn-sdk && pip install aibridge`
- [ ] Global replace imports: `from agn` → `from aibridge`
- [ ] Error base class: `AGNError` → `AibridgeError` (subclass names untouched)
- [ ] Error code strings: uppercase → snake_case (if hardcoded)
- [ ] Remove all `ChatOptions/ImageOptions/VideoOptions/EmbedOptions/TranscribeOptions/SpeechOptions` usage
- [ ] Streaming calls: `chat(stream=True)` → `chat_stream(...)`
- [ ] Translation calls: `translate(...)` → `transcribe(req, translate=true)`
- [ ] TTS file saving: `result.save_to_file(path)` → manually write `result.audio_data`
- [ ] Client construction: move `timeout/max_retries/retry_delay` to environment variables (or wait for a future release to expose them)
- [ ] Environment variables: you can keep using `AGN_*`; `AIBRIDGE_*` is recommended for new code
- [ ] provider names and aliases: no change needed (fully preserved)
- [ ] Run tests: use the `echo` adapter for an auth-free smoke test (`Client(provider="echo")`)

---

## 13. Migration Example: Full Script Comparison

Below is a full-script v1→v2 migration comparison, covering chat + streaming + speech.

### v1 Full Script

```python
import asyncio
from agn import Client, ChatOptions, AGNError, RateLimitError

async def main():
    client = Client(provider="agnes", api_key="your-key", base_url="https://api.agnes.ai/v1")
    async with client:
        # Chat
        opts = ChatOptions(temperature=0.7, max_tokens=1000)
        resp = await client.chat(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Hello!"}],
            options=opts,
        )
        print(resp.choices[0].message.content)

        # Streaming
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

### v2 Full Script (Python binding)

```python
import asyncio
from aibridge import Client, AibridgeError, RateLimitError

async def main():
    client = Client(provider="agnes", api_key="your-key", base_url="https://api.agnes.ai/v1")
    async with client:
        # Chat (drop Options, pass keyword arguments directly)
        resp = await client.chat(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0.7,
            max_tokens=1000,
        )
        print(resp.choices[0].message.content)

        # Streaming (separate method chat_stream)
        stream = await client.chat_stream(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "讲个笑话"}],
        )
        async for chunk in stream:
            print(chunk.choices[0].content or "", end="", flush=True)
        print()

        # TTS (manually write file)
        audio = await client.speech(model="tts-1", input="你好", voice="alloy")
        with open("hello.mp3", "wb") as f:
            f.write(audio.audio_data)

asyncio.run(main())
```

---

## 14. FAQ

**Q1: Why are some capabilities not yet exposed in the v2 Python binding?**
A: v2 is a Rust core + five-language bindings architecture. The Rust core already fully implements 38 providers + six major capabilities; the Python binding (PyO3) currently exposes `chat/chat_stream/speech`, with the remaining capabilities (`image_generate/video_*/embed/transcribe/list_models/list_voices`) exposed across future releases. If you urgently need full capabilities, you can use the Rust core directly or wait for the binding layer to catch up.

**Q2: Will performance improve after migration?**
A: Yes. The Rust core has no GIL, native async, and zero-cost abstractions; in IO-intensive scenarios its throughput and latency are significantly better than Python. The Python binding connects directly to the Rust core via PyO3 (no JSON serialization boundary), and real IO executes on tokio worker threads without blocking the asyncio event loop.

**Q3: Can v1 and v2 coexist?**
A: Yes. They have different package names (`agn-sdk` vs `aibridge`) and different import paths (`agn` vs `aibridge`), so they can coexist in the same environment. However, it is recommended to uninstall v1 after migration to avoid confusion.

**Q4: Must environment variables be changed?**
A: No. v2 is compatible with the legacy `AGN_*` prefix, so you can keep using it during migration. `AIBRIDGE_*` is recommended for new projects.

**Q5: Will the old v1 be archived?**
A: Yes. After v2 is officially released, the v1 repository will be archived and preserved, and `agn-sdk` on PyPI will no longer be updated. Migrating to `aibridge` is recommended.

---

## 15. Related Documentation

- [Design](design.md): architecture, data models, FFI boundary, error handling
- [Progress](PROGRESS.md): current implementation progress and handover guide
- [README (v2)](index.md): five-language quick start + provider list
- [Original README (v1)](https://github.com/WingkySky/AiBridge/blob/main/README_v1.md): Python v1 documentation (archived reference)
