# AIBridge

> Cross-language unified AI interface SDK - one API to call every AI model.
> Rust core + native bindings for five languages: Python / JS-TS / Go / JVM / .NET.

[![status](https://img.shields.io/badge/status-Phase%203%20release%20in%20progress-yellow)](PROGRESS.md)
[![provider](https://img.shields.io/badge/provider-38+-blue)](#supported-providers)
[![core tests](https://img.shields.io/badge/core%20tests-1448-brightgreen)](PROGRESS.md)

AIBridge (formerly agn-sdk) is a multimodal unified AI interface SDK: text chat, image generation (image), video generation (video), text-to-speech (TTS), speech-to-text (ASR), and text embedding (embed) - one API to call 38+ AI providers.

- **Native imports in five languages**: Python / JS-TS / Go / JVM (Java, Kotlin) / .NET (C#) all call the same set of capabilities directly.
- **Rust core**: no GIL, native async, zero-cost abstractions, high performance for IO-bound workloads.
- **38+ providers**: OpenAI / Claude / Gemini / Qwen / Zhipu / ERNIE / DeepSeek / Volcengine / Stability / Runway / Pika / Kling / Edge-TTS / ElevenLabs / Deepgram …
- **Six capabilities**: chat (including streaming) / image / video / TTS / ASR / embed
- **Auth-free provider**: edge-tts offers free TTS with no API key required.

> **v2 is a breaking upgrade**. If you are already using Python v1 (`agn-sdk`), see the [Migration Guide](migration-guide.md).

---

## Table of Contents

- [Architecture](#architecture)
- [Supported Providers](#supported-providers)
- [Quick Start in Five Languages](#quick-start-in-five-languages)
  - [Python](#python)
  - [Node.js / TypeScript](#nodejs--typescript)
  - [Go](#go)
  - [Java / Kotlin (JVM)](#java--kotlin-jvm)
  - [C# / .NET](#c--net)
- [Installation](#installation)
- [Capabilities Overview](#capabilities-overview)
- [Documentation](#documentation)

---

## Architecture

```
                    aibridge-core  (Rust, pure async logic)
                  ┌──────────┴──────────┐
      direct (native async)          C ABI (aibridge-ffi cdylib)
        ┌─────┴─────┐            ┌─────┬─────┬─────┐
   aibridge-    aibridge-     aibridge- aibridge- aibridge-
   python       node          go       jvm       dotnet
   (PyO3)       (napi-rs)     (CGO)    (JNA)     (P/Invoke)
   asyncio      Promise/      goroutine CompletableFuture Task/
   AsyncIter    AsyncIter     +channel  /Flow      IAsyncEnum
```

- **Python / JS-TS connect directly to the Rust core**: PyO3 / napi-rs link straight to `aibridge-core` with no JSON serialization boundary, giving you true native async.
- **Go / JVM / .NET go through the C ABI**: via the C ABI of `aibridge-ffi` (handle + JSON boundary + global tokio runtime), each language wraps it with its own native async primitives.
- **All five languages share the same Rust core**, keeping the binding layers thin and behavior consistent.

See the [Design](design.md) doc for details.

---

## Supported Providers

A total of **38 real providers + 1 mock** (echo, for testing), grouped by category:

| Category | provider |
|---|---|
| **MVP** | `openai` `agnes` `volcengine_cv` (Volcengine) `gemini` |
| **OpenAI-compatible family** | `azure` `siliconflow` (alias `sf`) `togetherai` (`together`) `fireworksai` (`fireworks`) `cloudflareai` (`cloudflare` / `workersai`) |
| **Extended models** | `grok` (`xaigrok`) `yi` (`lingyiwanwu`) `sensenova` (`shangtang`) `hunyuan` (`tencent_hunyuan`) `groq` |
| **More models** | `deepseek` `stepfun` (`step`) `mistral` `cohere` `perplexity` |
| **Emerging models** | `ideogram` (`ideo`) `luma` (`dream-machine` / `lumalabs`) `llama` (`meta-llama` / `meta`) |
| **Chinese models** | `qwen` (Qwen) `zhipu` (Zhipu) `doubao` (Doubao) `ernie` (ERNIE) `kimi` `minimax` |
| **Standalone protocols** | `anthropic` (Claude) `stability` (Stability AI) `runway` `pika` `kling` (Kling) |
| **Audio TTS** | `edge-tts` (free, no auth required, aliases `edge_tts` / `edge`) `elevenlabs` (`eleven` / `11labs`) `cartesia` (`sonic`) |
| **Audio ASR** | `deepgram` (`dg`) `assemblyai` (`assembly` / `aai`) |
| **Mock** | `echo` (no auth required, for testing, makes no network calls) |

---

## Quick Start in Five Languages

Each example below uses the **`echo` adapter** (no auth, no network calls), so you can run it directly to verify the pipeline. For real usage, replace `provider="echo"` with `provider="openai"` (etc.) and pass an `api_key`.

### Python

```python
import asyncio
from aibridge import Client

async def main():
    # echo requires no auth; real use: Client(provider="openai", api_key="sk-xxx")
    client = Client(provider="echo")
    await client.start()

    # Text chat
    resp = await client.chat(
        model="echo-chat",
        messages=[{"role": "user", "content": "你好"}],
    )
    print(resp.choices[0].message.content)  # "你好 [echo]"

    # Streaming chat
    stream = await client.chat_stream(
        model="echo-chat",
        messages=[{"role": "user", "content": "你好"}],
    )
    async for chunk in stream:
        print(chunk.choices[0].content, end="", flush=True)
    print()

    # Text-to-speech
    audio = await client.speech(model="echo-tts", input="你好", voice="alloy")
    print(f"audio {len(audio.audio_data)} bytes")

    await client.close()

asyncio.run(main())
```

Run:

```bash
pip install maturin
maturin develop -m crates/aibridge-python/Cargo.toml
python examples/hello_python.py
```

### Node.js / TypeScript

```javascript
const { Client } = require('./crates/aibridge-node');

async function main() {
  // echo requires no auth; real use: new Client('openai', { apiKey: 'sk-xxx' })
  const client = new Client('echo', {});
  await client.start();

  // Text chat
  const resp = await client.chat({
    model: 'echo-chat',
    messages: [{ role: 'user', content: '你好' }],
  });
  console.log(resp.choices[0].message.content); // "你好 [echo]"

  // Streaming chat
  const stream = await client.chatStream({
    model: 'echo-chat',
    messages: [{ role: 'user', content: '你好' }],
  });
  for await (const chunk of stream) {
    const delta = chunk.choices[0];
    if (delta.content) process.stdout.write(delta.content);
  }
  console.log();

  // Text-to-speech
  const audio = await client.speech({
    model: 'echo-tts',
    input: '你好',
    voice: 'alloy',
  });
  console.log(`audio ${audio.audioData.length} bytes`);

  await client.close();
}

main();
```

Run:

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
	// echo requires no auth; real use: aibridge.NewClient("openai", &aibridge.ClientOpts{ApiKey: "sk-xxx"})
	client, err := aibridge.NewClient("echo", nil)
	if err != nil {
		panic(err)
	}
	defer client.Close()
	client.Start()

	// Text chat
	chatReq := &aibridge.ChatRequest{
		Model:    "echo-chat",
		Messages: []aibridge.ChatMessage{aibridge.NewUserTextMessage("你好")},
	}
	resp, err := client.Chat(chatReq)
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Choices[0].Message.Content) // "你好 [echo]"

	// Streaming chat
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

	// Text-to-speech
	speechReq := &aibridge.SpeechRequest{
		Model: "echo-tts",
		Input: "你好",
		Voice: aibridge.SingleVoice("alloy"),
	}
	speech, err := client.Speech(speechReq)
	if err != nil {
		panic(err)
	}
	fmt.Printf("audio %d bytes\n", len(speech.AudioData))
}
```

Run:

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
        // echo requires no auth; real use: new Client("openai", "sk-xxx")
        try (Client client = new Client("echo")) {
            client.start();

            // Text chat
            ChatRequest req = ChatRequest.builder(
                    "echo-chat",
                    List.of(ChatMessage.user("你好")))
                .build();
            ChatCompletion resp = client.chat(req);
            System.out.println(resp.choices.get(0).message.content); // "你好 [echo]"

            // Streaming chat
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

            // Text-to-speech
            SpeechRequest speechReq = SpeechRequest.builder("echo-tts", "你好", "alloy").build();
            SpeechResultFull audio = client.speech(speechReq);
            System.out.println("audio " + audio.audioLength() + " bytes");
        }
    }
}
```

Run:

```bash
cd bindings/jvm && ./gradlew run
```

### C# / .NET

```csharp
using AIBridge;

// echo requires no auth; real use: new Client("openai", "sk-xxx")
using var client = new Client("echo");
client.Start();

// Text chat
var chatReq = new ChatRequest("echo-chat", new[]
{
    ChatMessage.User("你好"),
});
ChatCompletion resp = client.Chat(chatReq);
Console.WriteLine(resp.Choices[0].Message.Content); // "你好 [echo]"

// Streaming chat
int chunkCount = 0;
var assembled = new System.Text.StringBuilder();
await foreach (ChatCompletionChunk chunk in client.ChatStreamAsync(chatReq))
{
    chunkCount++;
    if (chunk.Choices.Count > 0 && chunk.Choices[0].Delta.Content != null)
        assembled.Append(chunk.Choices[0].Delta.Content);
}
Console.WriteLine(assembled.ToString());

// Text-to-speech
var speechReq = new SpeechRequest("echo-tts", "你好", "alloy");
SpeechResult audio = client.Speech(speechReq);
Console.WriteLine($"audio {audio.AudioData.Length} bytes");
```

Run:

```bash
cargo build -p aibridge-ffi
cd bindings/dotnet && dotnet run
```

---

## Installation

> Phase 3 release is in progress. The commands below are the target installation methods for each language. Before release, you can build from source.

| Language | Install command | Package |
|---|---|---|
| Python | `pip install aibridge` | PyPI `aibridge` |
| Node.js | `npm install aibridge` | npm `aibridge` |
| Go | `go get github.com/aibridge/aibridge-go` (requires installing libaibridge separately) | Go module `aibridge-go` |
| JVM | Maven `io.aibridge:aibridge` | Maven Central |
| .NET | `dotnet add package AIBridge` | NuGet `AIBridge` |

Build from source (development / pre-release):

```bash
# Rust core + ffi
cargo build --workspace

# Python bindings
pip install maturin
maturin develop -m crates/aibridge-python/Cargo.toml

# Node bindings
cd crates/aibridge-node && npm install && napi build

# Go / JVM / .NET bindings require building the libaibridge dynamic library first via cargo build -p aibridge-ffi
```

---

## Capabilities Overview

| Capability | Method | Description |
|---|---|---|
| Text chat | `chat` | Supports multi-turn, system/user/assistant/tool, multimodal, and tool calls |
| Streaming chat | `chat_stream` | Native async iterator, yielding chunk by chunk |
| Image generation | `image_generate` | Text-to-image, image-to-image (reference_images), inpainting (mask) |
| Video generation | `video_create` + `video_poll` | Text-to-video, image-to-video, task polling |
| Text-to-speech | `speech` | TTS, with automatic fallback across a list of candidate voices |
| Speech-to-text | `transcribe` | ASR, supporting file path / URL / bytes / base64 input |
| Text embedding | `embed` | Text vectorization |
| Model listing | `list_models` | Fetches a provider's available models in real time |
| Voice listing | `list_voices` / `recommend_voices` | Voice health check / recommendation / automatic fallback |

### Error Handling

A unified error base class `AibridgeError`, with subclasses categorized by error nature (exception names are consistent across languages):

- `AuthenticationError` - authentication failed
- `RateLimitError` - rate limited (includes `retry_after`)
- `ValidationError` - parameter validation
- `ModelNotFoundError` - model does not exist
- `APIError` - provider API error
- `NetworkError` - network error
- `TimeoutError` - timeout
- `UnsupportedCapabilityError` - capability not supported
- `ProviderNotFoundError` - provider does not exist
- `VoiceNotAvailableError` - voice not available
- `ServiceUnavailableError` - service temporarily unavailable (retryable)

Errors carry a stable `code` (snake_case, e.g. `rate_limit_error`) and a `retryable` flag, making retry decisions easy at the business layer.

---

## Documentation

- [Design](design.md) - architecture, data models, FFI boundary, async bridging, error handling, adapter migration strategy
- [Migration Guide](migration-guide.md) - breaking-change comparison and examples for Python v1 (agn-sdk) -> v2 (aibridge)
- [Progress](PROGRESS.md) - current implementation progress and handoff guide
- [Original README (v1)](https://github.com/WingkySky/AiBridge/blob/main/README_v1.md) - Python v1 docs (archived for reference)

---

## License

Same as the repository's main LICENSE.
