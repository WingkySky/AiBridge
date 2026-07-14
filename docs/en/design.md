# AIBridge Rust Refactor Design Document

> Date: 2026-07-07
> Status: Draft, pending user review
> Scope: Refactor the existing Python `agn-sdk` (v1.3.3, ~19,700 lines) into a cross-language SDK `aibridge` using Rust

---

## 1. Background and Goals

AIBridge (formerly agn-sdk, named after its early development based on Agnes) is a multimodal AI unified interface SDK: one set of APIs to call all AI models (chat / image / video / TTS / ASR / embed). The existing Python implementation is already published on PyPI and in use by real projects.

Core goals of this refactor:

1. **Five-language native import**: Python / JS-TS / Go / JVM (Java, Kotlin) / .NET (C#) directly call the same set of capabilities
2. **Full migration**: All 14 adapters + 6 major capabilities rewritten in Rust
3. **MVP first**: The four providers agnes / Volcengine (volcengine_cv) / gemini / openai first run end-to-end across five languages (a heavy dependency of the user's other project)
4. **v2 breaking upgrade**: Redesign an FFI-friendly unified API, provide a Python v1→v2 migration guide
5. **Rename and archive**: `agn-sdk` → `aibridge`, archive the old v1, the new version takes over

Non-goals:

- Not maintaining 1:1 compatibility with the v1 API (use a migration guide as a transition)
- Not doing performance benchmark comparisons against the old Python version
- Not introducing new providers in the first release (only migrating the existing 14)

---

## 2. Key Decisions Summary

| # | Decision Point | Choice | Rationale |
|---|---|---|---|
| 1 | Target languages | Python + JS/TS + Go + JVM + .NET | User requires full coverage of five languages |
| 2 | Python compatibility | v2 breaking upgrade + migration guide | 1:1 compatibility would hijack the API style of all languages, cost too high |
| 3 | First release scope | Full migration, MVP-first four providers | Full migration is the ultimate goal; four providers run first to validate the pipeline |
| 4 | Streaming/async | Fully native async | Best experience; PyO3/napi direct connection + C ABI wrapping for static languages |
| 5 | Repo/release | Monorepo + take over and rename | Unified versioning/CI, semver v2.0.0 expresses the breaking change |
| 6 | Architecture approach | Rust core + dual-entry bindings | The only one that simultaneously satisfies fully native async + consistent five-language experience |
| 7 | Brand name | aibridge | "AI bridge", straightforward and memorable, available on both PyPI/npm |

---

## 3. Overall Architecture

```
                    aibridge-core  (Rust, pure async logic)
                  ┌──────────┴──────────┐
    direct(native async)             C ABI (aibridge-ffi cdylib)
        ┌─────┴─────┐            ┌─────┬─────┬─────┐
   aibridge-    aibridge-     aibridge- aibridge- aibridge-
   python       node          go       jvm       dotnet
   (PyO3)       (napi-rs)     (CGO)    (JNA)     (P/Invoke)
   asyncio      Promise/      goroutine CompletableFuture Task/
   AsyncIter    AsyncIter     +channel  /Flow      IAsyncEnum
```

**Dual-entry principle**: Python/JS use PyO3/napi-rs to directly connect to `aibridge-core`, enjoying truly native async (no serialization boundary); Go/JVM/.NET have no cross-language async FFI concept, so they go through the C ABI of `aibridge-ffi` (blocking calls + wrapping with each language's native async primitives). All five languages share the same Rust core, and the binding layers are all thin.

**Preserving the original five-layer mental model**: API layer (Client) → routing layer (Router) → adapter layer (Adapter) → core layer (Core) → model layer (Model), only switching the language to Rust, `**kwargs` to explicit struct, Pydantic to serde, dynamic registration to compile-time match.

---

## 4. Monorepo Layout

```
aibridge/                         # Refactor the existing agn-sdk repo and rename it
├── Cargo.toml                    # workspace root
├── crates/
│   ├── aibridge-core/            # Rust core (pure async logic, no FFI pollution)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── client.rs         # unified Client
│   │   │   ├── router.rs         # Router routing/fallback
│   │   │   ├── adapter/{mod,base,factory}.rs   # Adapter trait + factory
│   │   │   ├── adapters/         # 14 adapters
│   │   │   │   ├── openai_compat.rs             # OpenAI-compatible foundation
│   │   │   │   ├── openai.rs agnes.rs azure.rs
│   │   │   │   ├── anthropic.rs gemini.rs volcengine_cv.rs
│   │   │   │   ├── runway.rs pika.rs kling.rs stability.rs
│   │   │   │   ├── chinese.rs aggregation_platforms.rs
│   │   │   │   ├── additional_models.rs more_models.rs emerging_models.rs
│   │   │   │   └── audio_adapters.rs            # edge-tts/elevenlabs/cartesia/deepgram/assemblyai
│   │   │   ├── http.rs           # reqwest+h2 wrapper
│   │   │   ├── retry.rs          # retry mechanism
│   │   │   ├── error.rs          # thiserror error enum
│   │   │   ├── config.rs         # config + environment variables
│   │   │   ├── util.rs
│   │   │   └── model/{chat,image,video,audio,common,options}.rs  # serde struct
│   │   └── tests/
│   ├── aibridge-ffi/             # C ABI cdylib (for Go/JVM/.NET)
│   │   ├── src/{lib,handle,error,stream,runtime}.rs
│   │   ├── include/aibridge.h    # generated by cbindgen
│   │   └── cbindgen.toml
│   ├── aibridge-python/          # PyO3 binding (directly connects to aibridge-core)
│   │   ├── src/lib.rs            # #[pymodule] aibridge
│   │   └── pyproject.toml        # maturin
│   └── aibridge-node/            # napi-rs binding (directly connects to aibridge-core)
│       ├── src/lib.rs
│       ├── package.json
│       └── build.rs
├── bindings/
│   ├── go/                       # CGO calls aibridge-ffi (goroutine+channel)
│   │   ├── aibridge.go aibridge.go.h stream.go
│   │   └── go.mod (github.com/aibridge/aibridge-go)
│   ├── jvm/                      # JNA calls aibridge-ffi (CompletableFuture/Flow, pure Java, no native Rust)
│   │   ├── src/main/{java,kotlin}/io/aibridge/...
│   │   └── build.gradle.kts
│   └── dotnet/                   # C# P/Invoke calls aibridge-ffi (Task/IAsyncEnumerable)
│       ├── AIBridge/ (C# project)
│       └── AIBridge.csproj
├── tests/                        # cross-language consistency test suite + shared fixtures
├── docs/                         # design document + per-language migration guides
└── .github/workflows/            # CI matrix (5 languages × multiple platforms)
```

**Key decision**: Go/JVM/.NET all go through the single C ABI entry of `aibridge-ffi`. JVM uses JNA (pure Java, no need to write native Rust), Go uses CGO, .NET uses P/Invoke. This avoids writing a separate set of Rust native glue for each of the three static languages.

---

## 5. aibridge-core Module Design

### 5.1 Python → Rust Module Mapping

| Existing Python | Rust (aibridge-core) | Notes |
|---|---|---|
| `agn/client.py` | `client::Client` | Unified entry, method signatures changed to explicit Request struct |
| `agn/router.py` | `router::Router` | Multi-provider routing/load balancing/fallback |
| `adapters/base.py` BaseAdapter | `adapter::Adapter` trait | `#[async_trait]`, unsupported capabilities default to returning UnsupportedCapability |
| `adapters/factory.py` AdapterFactory | `adapter::create_adapter()` explicit match | Replaces runtime registration; adding an adapter = adding a match branch |
| `adapters/*.py` (14) | `adapters::*` (14 modules) | trait implementations |
| `core/http_client.py` | `http` (reqwest+h2) | Replaces httpx |
| `core/retry.py` | `retry` | Replaces tenacity |
| `core/errors.py` | `error` (thiserror) | Standard error enum |
| `core/config.py` | `config` | Config + environment variables |
| `models/*.py` (Pydantic) | `model::*` (serde struct) | Replaces Pydantic |

### 5.2 Adapter trait

```rust
#[async_trait]
pub trait Adapter: Send + Sync {
    fn provider_type(&self) -> &str;
    fn provider_name(&self) -> &str;
    fn capabilities(&self) -> Capabilities;
    fn requires_api_key(&self) -> bool { true }

    async fn start(&mut self) -> Result<()>;
    async fn close(&mut self) -> Result<()>;
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion>;
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream>;  // impl Stream
    async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult>;
    async fn video_create(&self, req: VideoRequest) -> Result<VideoTask>;
    async fn video_poll(&self, task_id: &str, model: &str) -> Result<VideoStatus>;
    async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult>;
    async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscriptionResult>;
    async fn speech(&self, req: SpeechRequest) -> Result<SpeechResult>;
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>>;
    async fn list_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>>;
    async fn recommend_voices(&self, language: Option<&str>, gender: Option<&str>, limit: usize) -> Result<Vec<VoiceInfo>>;

    // Unsupported methods default to returning UnsupportedCapabilityError; the trait provides default implementations
}
```

### 5.3 Client Shape

```rust
pub struct Client { adapter: Box<dyn Adapter> }
impl Client {
    pub fn new(provider: &str, opts: ClientOptions) -> Result<Self>;
    pub async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion>;
    pub async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream>;
    pub async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult>;
    pub async fn video_create(&self, req: VideoRequest) -> Result<VideoTask>;
    pub async fn video_poll(&self, task_id: &str, model: &str) -> Result<VideoStatus>;
    pub async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult>;
    pub async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscriptionResult>;
    pub async fn speech(&self, req: SpeechRequest) -> Result<SpeechResult>;
    pub async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>>;
    pub async fn list_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>>;
    pub async fn recommend_voices(&self, lang: Option<&str>, gender: Option<&str>, limit: usize) -> Result<Vec<VoiceInfo>>;
}
```

- **Optional parameters**: Request struct + Builder (`..Default::default()`), replacing `**kwargs`. Provider-specific parameters pass through via `extra: HashMap<String, serde_json::Value>`.
- **Streaming**: `ChatStream: impl Stream<Item = Result<ChatCompletionChunk>>` (`async-stream` crate), native async stream inside the core.
- **Factory**: compile-time explicit `match`, replacing Python's runtime `AdapterFactory.register`. More static, more safe.

---

## 6. Unified Data Model (serde struct replacing Pydantic + kwargs)

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub stop: Option<StopSeq>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub response_format: Option<ResponseFormat>,
    pub extra: HashMap<String, serde_json::Value>,  // provider-specific parameter pass-through
}
// ChatRequest::builder("gpt-4o", messages).temperature(0.7).max_tokens(1000).build()

#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    System { content: String },
    User { content: UserContent },            // String or multimodal Vec<ContentPart>
    Assistant { content: String, tool_calls: Option<Vec<ToolCall>> },
    Tool { tool_call_id: String, content: String },
}

pub enum FileInput { Path(String), Url(String), Bytes(Vec<u8>), Base64(String) }
```

- **Removing the Options middle layer**: Python's `ChatOptions/ImageOptions/...` are replaced in Rust directly by `Request::builder()` chained calls, with a 1:1 comparison in the migration guide.
- **Response models**: `ChatCompletion`/`ChatCompletionChunk`/`ImageResult`/`VideoTask`/`VideoStatus`/`EmbeddingResult`/`TranscriptionResult`/`SpeechResult`/`ModelInfo`/`VoiceInfo` are all serde structs, with field names aligned to Python Pydantic (the migration guide can compare field by field).

---

## 7. aibridge-ffi C ABI Boundary

**Core principle**: handle-based lifecycle + JSON strings as the boundary for complex structs + native binary passing + global tokio runtime.

```c
/* handles (opaque) */
typedef struct aibridge_client_s aibridge_client_t;
typedef struct aibridge_stream_s aibridge_stream_t;
typedef struct { const uint8_t* ptr; size_t len; } aibridge_bytes_t;

/* lifecycle */
aibridge_client_t* aibridge_client_new(const char* provider, const char* config_json);
aibridge_status_t   aibridge_client_start(aibridge_client_t*);
void                aibridge_client_destroy(aibridge_client_t*);

/* blocking calls: internal runtime.block_on, complex structures go through the JSON boundary */
aibridge_status_t aibridge_client_chat(aibridge_client_t*, const char* request_json,
                                       char** out_response_json);          /* caller calls aibridge_string_free */
/* binary payloads do not go through JSON (avoiding base64 bloat) */
aibridge_status_t aibridge_client_speech(aibridge_client_t*, const char* request_json,
                                         aibridge_bytes_t** out_audio, char** out_meta_json);

/* streaming: stream handle + blocking next() */
aibridge_status_t aibridge_client_chat_stream(aibridge_client_t*, const char* request_json,
                                              aibridge_stream_t** out_stream);
aibridge_status_t aibridge_stream_next(aibridge_stream_t*, char** out_chunk_json);  /* 0=chunk, 1=end, negative=error */
void              aibridge_stream_destroy(aibridge_stream_t*);   /* triggers Rust drop → tokio task abort */

/* error: return code + thread-local last_error slot */
const char* aibridge_last_error(void);   /* thread-local, no need to free */

/* free */
void aibridge_string_free(char*);
void aibridge_bytes_free(aibridge_bytes_t*);
```

**Key decisions and trade-offs**:

1. **JSON boundary vs C struct mirrorgen**: chose JSON. Using mirrorgen for 5 languages × a large number of structs has high maintenance cost; the JSON boundary keeps the C ABI to only ~20 functions, and each language binding only does JSON (de)serialization, which is type-safe and consistent across languages. The SDK is IO-intensive, so JSON serialization overhead is negligible.
2. **Global tokio runtime**: `Lazy<TokioRuntime>` (multi-threaded), each FFI call does `handle().block_on(async{...})`. The C side can call different clients concurrently; the same stream is serial (each language binding is responsible for synchronization).
3. **Error model**: `aibridge_status_t` return code (0 success / negative error category) + `aibridge_last_error()` thread-local detailed message (JSON, containing `code/message/details/retryable`). Each language binding maps it to a native exception.
4. **String/binary ownership**: `char*`/`aibridge_bytes_t` allocated by Rust must be freed by the caller calling `aibridge_*_free`. Each language binding wraps this with RAII (Go finalizer / JVM Cleaner / .NET Dispose).

---

## 8. Async and Streaming Cross-Language Bridging

| Language | Binding path | Async primitive | Streaming type | Cancel mechanism |
|---|---|---|---|---|
| Python | PyO3 **direct** to aibridge-core | asyncio coroutine | `AsyncIterator` | asyncio cancel → drop stream |
| JS/TS | napi-rs **direct** to aibridge-core | `Promise` | `AsyncIterable` | `AbortSignal` → drop stream |
| Go | CGO → aibridge-ffi | goroutine | `chan Chunk` | `ctx.Done()` → `aibridge_stream_destroy` |
| JVM | JNA → aibridge-ffi | `CompletableFuture` / Kotlin `suspend` | `Flow.Publisher` | cancel → destroy |
| .NET | P/Invoke → aibridge-ffi | `Task<T>` | `IAsyncEnumerable<T>` | `CancellationToken` → destroy |

**Direct connection vs C ABI**:

- **Python/JS direct connection**: no JSON boundary, PyO3 `#[pyo3] async fn` / napi `#[napi] async fn` automatically bridge native async; Rust structs are annotated with `#[pyclass]`/napi and then directly mapped to host objects. Streaming internally uses `tokio::spawn` + channel to bridge into a native async iterator. Best experience, zero serialization.
- **Go/JVM/.NET via C ABI**: the FFI call itself is blocking; each language schedules it on its own async executor (Go goroutine / JVM ForkJoinPool / .NET ThreadPool), wrapping the blocking FFI into native async primitives. Streaming uses a "stream handle + blocking `next()`" loop on a background thread, pushing results to channel/Flow/IAsyncEnumerable.
- **Unified cancel**: cancellation in all languages ultimately lands on `aibridge_stream_destroy` (Rust drop → tokio task abort), with consistent semantics. The direct-connection bindings for Python/JS bridge asyncio cancel / AbortSignal to a Rust-side drop stream.

---

## 9. Error Handling and Type Mapping

### 9.1 Rust Error Enum (thiserror)

```rust
#[derive(Debug, thiserror::Error)]
pub enum AibridgeError {
    #[error("认证失败: {message}")] Authentication { message: String },
    #[error("限流: {message}")] RateLimit { message: String, retry_after: Option<f64> },
    #[error("参数校验错误: {message}")] Validation { message: String, details: serde_json::Value },
    #[error("模型不存在: {model}")] ModelNotFound { model: String },
    #[error("API 调用错误: {message}")] Api { status: u16, message: String },
    #[error("网络错误: {0}")] Network(#[from] reqwest::Error),
    #[error("超时")] Timeout,
    #[error("不支持的能力: {capability}")] UnsupportedCapability { capability: String },
    #[error("Provider 不存在: {provider}")] ProviderNotFound { provider: String },
}
```

This corresponds to Python v1's `AGNError` hierarchy (`AuthenticationError`/`RateLimitError`/`ValidationError`/`ModelNotFoundError`/`APIError`/`TimeoutError`/`UnsupportedCapabilityError`/`ProviderNotFoundError`), with consistent categorization, and the migration guide provides a name mapping (`AGNError` → `AibridgeError`).

### 9.2 FFI Error Passing

The return code `aibridge_status_t` (i32) maps error categories; `aibridge_last_error()` returns thread-local JSON: `{"code":"rate_limit","message":"...","retryable":true,"details":{...}}`.

### 9.3 Per-Language Exception Mapping

- **Python**: `AibridgeError` base class + subclasses (`RateLimitError`, etc.), sharing names with v1 for easy migration
- **JS/TS**: `AibridgeError` + subclasses
- **Go**: `error` + type assertion interface `type AibridgeError interface{ Code() string }`
- **JVM**: `AibridgeException` + subclasses
- **.NET**: `AibridgeException` + subclasses

---

## 10. Adapter Migration Strategy

### 10.1 Capturing Commonality: OpenAI-Compatible Foundation

Of the 14 adapters, 80% use the OpenAI-compatible protocol (agnes/openai/azure/aggregation platforms/the compatible portion of Chinese platforms). Rust extracts an `OpenAiCompatAdapter` base implementation (HTTP request construction + response parsing + parameter mapping), and sub-adapters only override the differences (base_url, model list, parameter mapping, special endpoints).

### 10.2 Rust-ifying the Parameter Mapping Mechanism

Preserve Python's `ParameterMapping` concept, Rust-ified as a `param_mapping` table + `apply_mapping()`. Preset constants: `OPENAI_COMPATIBLE_MAPPING`, `ANTHROPIC_MAPPING`, `GEMINI_MAPPING`, `COHERE_MAPPING`, reused by OpenAI-compatible adapters.

### 10.3 Batch Migration Order (MVP First)

| Batch | Adapters | Protocol | Priority |
|---|---|---|---|
| **Phase 1 · MVP** | openai, agnes, volcengine_cv, gemini | compatible foundation + standalone protocols | **Highest** |
| Phase 2a | azure, aggregation_platforms, additional_models, more_models, emerging_models, chinese (compatible portion) | OpenAI-compatible family | High |
| Phase 2b | anthropic, stability, runway, pika, kling | standalone protocols | Medium |
| Phase 2c | edge-tts, elevenlabs, cartesia, deepgram, assemblyai (audio_adapters) | binary payloads | Medium |

> Effort estimate: ~14,000 lines of Python; after sharing the foundation, the actual new Rust to write is about 6,000–8,000 lines.

### 10.4 Preserved Features

- v1.1.0's "real-time model list fetching" (list_models calling the provider's /models endpoint) is preserved
- v1.3.0's "auth-free free Providers" (edge-tts requires no api_key, `requires_api_key() -> false`) is preserved
- v1.3.3's "TTS voice health check/recommendation/automatic fallback" is preserved

---

## 11. Testing Strategy

Three layers of testing:

1. **Rust core unit tests** (within aibridge-core): each adapter mocks HTTP, covering normal + abnormal + boundary cases. Coverage ≥80%.
2. **Per-language binding tests**: each language tests its own glue layer (handle lifecycle, async bridging, error mapping, streaming).
3. **Cross-language consistency tests**: the same set of inputs + mocks run once through each of the five languages, asserting consistent output. This prevents a single language's binding from missing an implementation or drifting in behavior.

**Shared fixtures**: the HTTP request/response mock data from the Python v1 tests is extracted into JSON files (`tests/fixtures/`), shared by the Rust tests + the five-language binding tests. This guarantees "consistent behavior across five languages + consistency with the old Python version." **This is the core of quality assurance.**

Routine testing does not call the real AI API (high cost, unstable). A separate "real interface smoke test" toggle is kept (environment variable configures the key, off by default in CI).

---

## 12. Build and Release

### 12.1 Per-Language Packaging Approach

| Language | Tool | Publish to | User install experience |
|---|---|---|---|
| Rust core + ffi | cargo | — (internal) | Produces libaibridge.{so,dylib,dll} |
| Python | maturin | PyPI `aibridge` v2.0.0 | `pip install aibridge`, prebuilt wheel, no Rust needed |
| JS/TS | napi-rs | npm `aibridge` | `npm install aibridge`, prebuilt .node, no Rust needed |
| Go | cgo | Go module `aibridge-go` | Requires installing libaibridge separately (install script provided, Go ecosystem convention) |
| JVM | JNA + Gradle | Maven `io.aibridge:aibridge` | Dynamic library packed into the jar (by platform classifier), transparent to users |
| .NET | P/Invoke | NuGet `AIBridge` | Dynamic library packed into the package (runtimes/{rid}/native/), transparent to users |

### 12.2 Binary Distribution Challenges

Go/JVM/.NET depend on the `libaibridge` dynamic library. JVM and .NET pack the dynamic library into their respective packages (split by OS/arch), transparent to users; due to the cgo mechanism, Go requires users to install the dynamic library separately (install script provided), which is standard practice in the Go ecosystem.

### 12.3 CI Matrix

GitHub Actions: platform (linux/macos/windows × amd64/arm64) × language. The dynamic library from aibridge-ffi serves as a build artifact for Go/JVM/.NET packaging to consume. The Rust core + 5 bindings each have one workflow.

---

## 13. Phased Implementation Plan

> Time is a rough estimate for one full-time person; the implementation phase can be accelerated with multiple agents in parallel.

### Phase 0 · Laying the Foundation (2–3 weeks)

- Monorepo setup, Cargo workspace
- aibridge-core skeleton: error/config/http/retry/model/Adapter trait/Client/Router
- aibridge-ffi C ABI skeleton + global runtime + cbindgen
- Cross-language pipeline established: use openai's chat stub to run a five-language hello world end-to-end (including async + streaming + errors), validating the biggest technical risk

### Phase 1 · MVP Four Providers (3–4 weeks) ⭐

openai → agnes → volcengine_cv → gemini, implemented one by one in Rust and run end-to-end across five languages + tests. **Once this step is done, the user's other project can begin integrating.**

### Phase 2 · Remaining Adapters (4–6 weeks)

Migrate the remaining 10 in three batches: 2a (compatible family) → 2b (standalone protocols) → 2c (audio).

### Phase 3 · Release Wrap-up (2–3 weeks)

Full-platform CI builds, official release of the five-language packages, Python v1→v2 migration guide, documentation website, archiving of the old v1, tag v2.0.0.

**Total cycle roughly 3–4 months** (one full-time person).

### Multi-Agent Parallel Orchestration Strategy (Implementation Phase)

- **Adapter migration**: one agent per provider in parallel (sharing the OpenAiCompatAdapter base), with shared fixtures guaranteeing consistency
- **Language bindings**: once aibridge-core and aibridge-ffi are stable, one agent per language binding in parallel
- **Testing**: a separate agent aggregates the cross-language consistency tests
- **Dependency order**: Phase 0 must be serial (bindings cannot parallelize before the core is stable); from Phase 1 onward, adapters and bindings can run in parallel

---

## 14. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Fully native async across FFI is complex | High | PyO3/napi direct connection avoids C ABI async; static languages use blocking + native async wrapping, already fixed in the design |
| Long cycle for full migration of 14 adapters | Medium | OpenAI-compatible foundation reuses 80%; MVP four providers prioritized to ensure early availability |
| Behavior drift across five languages | Medium | Shared fixtures + cross-language consistency tests |
| Go/JVM/.NET dynamic library distribution | Medium | JVM/.NET pack it into the package; Go provides an install script |
| crates.io/package name taken | Low | PyPI/npm confirmed aibridge available; re-confirm crates.io/Maven/NuGet before release, use a suffix to work around if needed |
| Breaking upgrades for old Python users | Medium | v2.0.0 semver + migration guide + archived old v1 retained |

---

## 15. Python v1→v2 Migration Guide Highlights

- Package name: `agn-sdk` → `aibridge`, `from agn import Client` → `from aibridge import Client`
- Error classes: `AGNError` → `AibridgeError` (subclass names unchanged)
- Parameters: `**kwargs` pass-through → `Request` struct + Builder chained calls (`ChatOptions` → `ChatRequest::builder()`)
- Options classes: the `ChatOptions/ImageOptions/...` middle layer is removed, use the Request builder directly
- All other method names, capabilities, and provider names remain consistent

---

## Appendix A: Capability Comparison with Python v1

| Capability | v1 (agn-sdk) | v2 (aibridge) | Status |
|---|---|---|---|
| chat (including streaming) | ✅ | ✅ | Migrated |
| image_generate | ✅ | ✅ | Migrated |
| video_create + poll | ✅ | ✅ | Migrated |
| transcribe (ASR) | ✅ | ✅ | Migrated |
| speech (TTS) | ✅ | ✅ | Migrated |
| embed | ✅ | ✅ | Migrated |
| list_models (real-time fetching) | ✅ | ✅ | Migrated |
| list_voices / recommend_voices | ✅ | ✅ | Migrated |
| Router (multi-provider routing) | ✅ | ✅ | Migrated |
| Voice automatic fallback | ✅ | ✅ | Migrated |
