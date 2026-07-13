# AIBridge

> 跨语言 AI 统一接口 SDK —— 一套 API 调用所有 AI 模型。
> Rust 核心 + Python / JS-TS / Go / JVM / .NET 五语言原生绑定。

[![状态](https://img.shields.io/badge/状态-阶段3发布收尾中-yellow)](PROGRESS.md)
[![provider](https://img.shields.io/badge/provider-38+-blue)](#支持的-provider)
[![核心测试](https://img.shields.io/badge/core%20单测-1448-brightgreen)](PROGRESS.md)

AIBridge（原名 agn-sdk）是多模态 AI 统一接口 SDK：文本对话（chat）、图像生成（image）、视频生成（video）、文字转语音（TTS）、语音转文字（ASR）、文本嵌入（embed），一套 API 调用 38+ 个 AI provider。

- **五语言原生 import**：Python / JS-TS / Go / JVM(Java,Kotlin) / .NET(C#) 直接调用同一套能力
- **Rust 核心**：无 GIL、原生 async、零成本抽象，IO 密集场景高性能
- **38+ provider**：OpenAI / Claude / Gemini / 通义千问 / 智谱 / 文心一言 / DeepSeek / 火山引擎 / Stability / Runway / Pika / 可灵 / Edge-TTS / ElevenLabs / Deepgram …
- **六大能力**：chat（含流式）/ image / video / TTS / ASR / embed
- **免认证 provider**：edge-tts 免费 TTS，无需 API key

> **v2 是破坏性升级**。若你已在用 Python v1（`agn-sdk`），请参阅 [迁移指南](migration-guide.md)。

---

## 目录

- [架构](#架构)
- [支持的 provider](#支持的-provider)
- [五语言快速开始](#五语言快速开始)
  - [Python](#python)
  - [Node.js / TypeScript](#nodejs--typescript)
  - [Go](#go)
  - [Java / Kotlin (JVM)](#java--kotlin-jvm)
  - [C# / .NET](#c--net)
- [安装](#安装)
- [能力一览](#能力一览)
- [相关文档](#相关文档)

---

## 架构

```
                    aibridge-core  (Rust, 纯 async 逻辑)
                  ┌──────────┴──────────┐
        直连(原生async)              C ABI (aibridge-ffi cdylib)
        ┌─────┴─────┐            ┌─────┬─────┬─────┐
   aibridge-    aibridge-     aibridge- aibridge- aibridge-
   python       node          go       jvm       dotnet
   (PyO3)       (napi-rs)     (CGO)    (JNA)     (P/Invoke)
   asyncio      Promise/      goroutine CompletableFuture Task/
   AsyncIter    AsyncIter     +channel  /Flow      IAsyncEnum
```

- **Python / JS-TS 直连 Rust 核心**：PyO3 / napi-rs 直连 `aibridge-core`，无 JSON 序列化边界，享真正原生 async。
- **Go / JVM / .NET 走 C ABI**：通过 `aibridge-ffi` 的 C ABI（句柄 + JSON 边界 + 全局 tokio runtime），各语言用原生异步原语包装。
- **五种语言共享同一个 Rust 核心**，绑定层都薄，行为一致。

详见 [设计文档](design.md)。

---

## 支持的 provider

共 **38 个真实 provider + 1 个 mock**（echo，测试用），按类别：

| 类别 | provider |
|---|---|
| **MVP** | `openai` `agnes` `volcengine_cv`（火山引擎） `gemini` |
| **OpenAI 兼容族** | `azure` `siliconflow`（别名 `sf`） `togetherai`（`together`） `fireworksai`（`fireworks`） `cloudflareai`（`cloudflare` / `workersai`） |
| **扩展模型** | `grok`（`xaigrok`） `yi`（`lingyiwanwu`） `sensenova`（`shangtang`） `hunyuan`（`tencent_hunyuan`） `groq` |
| **更多模型** | `deepseek` `stepfun`（`step`） `mistral` `cohere` `perplexity` |
| **新兴模型** | `ideogram`（`ideo`） `luma`（`dream-machine` / `lumalabs`） `llama`（`meta-llama` / `meta`） |
| **中文模型** | `qwen`（通义千问） `zhipu`（智谱） `doubao`（豆包） `ernie`（文心一言） `kimi` `minimax` |
| **独立协议** | `anthropic`（Claude） `stability`（Stability AI） `runway` `pika` `kling`（可灵） |
| **音频 TTS** | `edge-tts`（免费，免认证，别名 `edge_tts` / `edge`） `elevenlabs`（`eleven` / `11labs`） `cartesia`（`sonic`） |
| **音频 ASR** | `deepgram`（`dg`） `assemblyai`（`assembly` / `aai`） |
| **Mock** | `echo`（免认证，测试用，不调网络） |

---

## 五语言快速开始

下面每个示例都用 **`echo` 适配器**（免认证、不调网络），可直接运行验证管线。真实使用时把 `provider="echo"` 换成 `provider="openai"` 等并传入 `api_key`。

### Python

```python
import asyncio
from aibridge import Client

async def main():
    # echo 免认证；真实用例：Client(provider="openai", api_key="sk-xxx")
    client = Client(provider="echo")
    await client.start()

    # 文本对话
    resp = await client.chat(
        model="echo-chat",
        messages=[{"role": "user", "content": "你好"}],
    )
    print(resp.choices[0].message.content)  # "你好 [echo]"

    # 流式对话
    stream = await client.chat_stream(
        model="echo-chat",
        messages=[{"role": "user", "content": "你好"}],
    )
    async for chunk in stream:
        print(chunk.choices[0].content, end="", flush=True)
    print()

    # 文字转语音
    audio = await client.speech(model="echo-tts", input="你好", voice="alloy")
    print(f"音频 {len(audio.audio_data)} 字节")

    await client.close()

asyncio.run(main())
```

运行：

```bash
pip install maturin
maturin develop -m crates/aibridge-python/Cargo.toml
python examples/hello_python.py
```

### Node.js / TypeScript

```javascript
const { Client } = require('./crates/aibridge-node');

async function main() {
  // echo 免认证；真实用例：new Client('openai', { apiKey: 'sk-xxx' })
  const client = new Client('echo', {});
  await client.start();

  // 文本对话
  const resp = await client.chat({
    model: 'echo-chat',
    messages: [{ role: 'user', content: '你好' }],
  });
  console.log(resp.choices[0].message.content); // "你好 [echo]"

  // 流式对话
  const stream = await client.chatStream({
    model: 'echo-chat',
    messages: [{ role: 'user', content: '你好' }],
  });
  for await (const chunk of stream) {
    const delta = chunk.choices[0];
    if (delta.content) process.stdout.write(delta.content);
  }
  console.log();

  // 文字转语音
  const audio = await client.speech({
    model: 'echo-tts',
    input: '你好',
    voice: 'alloy',
  });
  console.log(`音频 ${audio.audioData.length} 字节`);

  await client.close();
}

main();
```

运行：

```bash
cd crates/aibridge-node && npm install && napi build && cd ../..
node examples/hello_node.js
```

### Go

```go
package main

import (
	"fmt"

	aibridge "github.com/aibridge/aibridge-go"
)

func main() {
	// echo 免认证；真实用例：aibridge.NewClient("openai", &aibridge.ClientOpts{ApiKey: "sk-xxx"})
	client, err := aibridge.NewClient("echo", nil)
	if err != nil {
		panic(err)
	}
	defer client.Close()
	client.Start()

	// 文本对话
	chatReq := &aibridge.ChatRequest{
		Model:    "echo-chat",
		Messages: []aibridge.ChatMessage{aibridge.NewUserTextMessage("你好")},
	}
	resp, err := client.Chat(chatReq)
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Choices[0].Message.Content) // "你好 [echo]"

	// 流式对话
	stream, err := client.ChatStream(chatReq)
	if err != nil {
		panic(err)
	}
	for chunk := range stream.Ch() {
		if len(chunk.Choices) > 0 {
			fmt.Print(chunk.Choices[0].Delta.Content)
		}
	}
	fmt.Println()

	// 文字转语音
	speechReq := &aibridge.SpeechRequest{
		Model: "echo-tts",
		Input: "你好",
		Voice: aibridge.SingleVoice("alloy"),
	}
	speech, err := client.Speech(speechReq)
	if err != nil {
		panic(err)
	}
	fmt.Printf("音频 %d 字节\n", len(speech.AudioData))
}
```

运行：

```bash
cargo build -p aibridge-ffi
cd bindings/go
CGO_ENABLED=1 DYLD_LIBRARY_PATH=../../target/debug go run ./example
```

### Java / Kotlin (JVM)

```java
package io.aibridge;

import java.util.List;

public class Hello {
    public static void main(String[] args) {
        // echo 免认证；真实用例：new Client("openai", "sk-xxx")
        try (Client client = new Client("echo")) {
            client.start();

            // 文本对话
            ChatRequest req = ChatRequest.builder(
                    "echo-chat",
                    List.of(ChatMessage.user("你好")))
                .build();
            ChatCompletion resp = client.chat(req);
            System.out.println(resp.choices.get(0).message.content); // "你好 [echo]"

            // 流式对话
            ChatRequest streamReq = ChatRequest.builder(
                    "echo-chat",
                    List.of(ChatMessage.user("你好")))
                .stream(true)
                .build();
            try (ChatStream stream = client.chatStream(streamReq)) {
                while (stream.hasNext()) {
                    ChatCompletionChunk chunk = stream.next();
                    String c = chunk.firstDeltaContent();
                    if (c != null) System.out.print(c);
                }
            }
            System.out.println();

            // 文字转语音
            SpeechRequest speechReq = SpeechRequest.builder("echo-tts", "你好", "alloy").build();
            SpeechResultFull audio = client.speech(speechReq);
            System.out.println("音频 " + audio.audioLength() + " 字节");
        }
    }
}
```

运行：

```bash
cd bindings/jvm && ./gradlew run
```

### C# / .NET

```csharp
using AIBridge;

// echo 免认证；真实用例：new Client("openai", "sk-xxx")
using var client = new Client("echo");
client.Start();

// 文本对话
var chatReq = new ChatRequest("echo-chat", new[]
{
    ChatMessage.User("你好"),
});
ChatCompletion resp = client.Chat(chatReq);
Console.WriteLine(resp.Choices[0].Message.Content); // "你好 [echo]"

// 流式对话
int chunkCount = 0;
var assembled = new System.Text.StringBuilder();
await foreach (ChatCompletionChunk chunk in client.ChatStreamAsync(chatReq))
{
    chunkCount++;
    if (chunk.Choices.Count > 0 && chunk.Choices[0].Delta.Content != null)
        assembled.Append(chunk.Choices[0].Delta.Content);
}
Console.WriteLine(assembled.ToString());

// 文字转语音
var speechReq = new SpeechRequest("echo-tts", "你好", "alloy");
SpeechResult audio = client.Speech(speechReq);
Console.WriteLine($"音频 {audio.AudioData.Length} 字节");
```

运行：

```bash
cargo build -p aibridge-ffi
cd bindings/dotnet && dotnet run
```

---

## 安装

> 阶段 3 发布进行中，以下为各语言的目标安装方式。发布前可从源码构建。

| 语言 | 安装命令 | 包名 |
|---|---|---|
| Python | `pip install aibridge` | PyPI `aibridge` |
| Node.js | `npm install aibridge` | npm `aibridge` |
| Go | `go get github.com/aibridge/aibridge-go`（需单独装 libaibridge） | Go module `aibridge-go` |
| JVM | Maven `io.aibridge:aibridge` | Maven Central |
| .NET | `dotnet add package AIBridge` | NuGet `AIBridge` |

从源码构建（开发/发布前）：

```bash
# Rust 核心 + ffi
cargo build --workspace

# Python 绑定
pip install maturin
maturin develop -m crates/aibridge-python/Cargo.toml

# Node 绑定
cd crates/aibridge-node && npm install && napi build

# Go / JVM / .NET 绑定需先 cargo build -p aibridge-ffi 产 libaibridge 动态库
```

---

## 能力一览

| 能力 | 方法 | 说明 |
|---|---|---|
| 文本对话 | `chat` | 支持多轮、system/user/assistant/tool、多模态、工具调用 |
| 流式对话 | `chat_stream` | 原生异步迭代器，逐块产出 |
| 图像生成 | `image_generate` | 文生图、图生图（reference_images）、局部重绘（mask） |
| 视频生成 | `video_create` + `video_poll` | 文生视频、图生视频，任务轮询 |
| 文字转语音 | `speech` | TTS，支持音色候选列表自动降级 |
| 语音转文字 | `transcribe` | ASR，支持文件路径/URL/bytes/base64 输入 |
| 文本嵌入 | `embed` | 文本向量化 |
| 模型列表 | `list_models` | 实时拉取 provider 可用模型 |
| 音色列表 | `list_voices` / `recommend_voices` | 音色健康检查/推荐/自动降级 |

### 错误处理

统一错误基类 `AibridgeError`，子类按错误性质分类（各语言异常名一致）：

- `AuthenticationError` 认证失败
- `RateLimitError` 限流（含 `retry_after`）
- `ValidationError` 参数校验
- `ModelNotFoundError` 模型不存在
- `APIError` Provider API 错误
- `NetworkError` 网络错误
- `TimeoutError` 超时
- `UnsupportedCapabilityError` 能力不支持
- `ProviderNotFoundError` provider 不存在
- `VoiceNotAvailableError` 音色不可用
- `ServiceUnavailableError` 服务暂不可用（可重试）

错误带稳定 `code`（snake_case，如 `rate_limit_error`）与 `retryable` 标识，便于业务层重试决策。

---

## 相关文档

- [设计文档](design.md) — 架构、数据模型、FFI 边界、异步桥接、错误处理、适配器迁移策略
- [迁移指南](migration-guide.md) — Python v1（agn-sdk）→ v2（aibridge）破坏性升级对照与示例
- [进度文档](PROGRESS.md) — 当前实施进度与接手指南
- [原 README（v1）](https://github.com/WingkySky/AGN-SDK/blob/main/README_v1.md) — Python v1 文档（归档参考）

---

## License

同仓库主 LICENSE。
