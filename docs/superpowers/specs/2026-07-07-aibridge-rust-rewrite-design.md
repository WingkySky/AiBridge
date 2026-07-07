# AIBridge Rust 重构设计文档

> 日期：2026-07-07
> 状态：草案，待用户审查
> 范围：将现有 Python `agn-sdk` (v1.3.3, ~19700 行) 用 Rust 重构为跨语言 SDK `aibridge`

---

## 1. 背景与目标

AIBridge（原名 agn-sdk，因早期基于 Agnes 开发而得名）是多模态 AI 统一接口 SDK，一套 API 调用所有 AI 模型（chat / image / video / TTS / ASR / embed）。现有 Python 实现已发布 PyPI，已有项目在用。

本次重构的核心目标：

1. **五语言原生 import**：Python / JS-TS / Go / JVM(Java,Kotlin) / .NET(C#) 直接调用同一套能力
2. **全量迁移**：14 个适配器 + 6 大能力全部用 Rust 重写
3. **MVP 优先**：agnes / 火山引擎(volcengine_cv) / gemini / openai 四个 provider 先在五语言跑通（用户另一项目主力依赖）
4. **v2 破坏性升级**：重新设计 FFI 友好的统一 API，提供 Python v1→v2 迁移指南
5. **更名与归档**：`agn-sdk` → `aibridge`，旧版 v1 归档，新版接管

非目标：

- 不保留 v1 API 1:1 兼容（用迁移指南过渡）
- 不做 Python 老版性能基准对照
- 不在首版引入新 provider（仅迁移现有 14 个）

---

## 2. 关键决策摘要

| # | 决策点 | 选择 | 理由 |
|---|---|---|---|
| 1 | 目标语言 | Python + JS/TS + Go + JVM + .NET | 用户要求五语言全覆盖 |
| 2 | Python 兼容性 | v2 破坏性升级 + 迁移指南 | 1:1 兼容会绑架所有语言 API 风格，成本过高 |
| 3 | 首版范围 | 全量迁移，MVP 优先四 provider | 全量是最终目标，四 provider 先跑通验证管线 |
| 4 | 流式/异步 | 全原生 async | 体验最佳；PyO3/napi 直连 + C ABI 静态语言包装 |
| 5 | 仓库/发布 | Monorepo + 接管更名 | 统一版本/CI，semver v2.0.0 表达破坏性 |
| 6 | 架构方案 | Rust 核心 + 双入口绑定 | 唯一同时满足全原生 async + 五语言体验一致 |
| 7 | 品牌名 | aibridge | "AI 桥"，直白好记，PyPI/npm 均可用 |

---

## 3. 整体架构

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

**双入口原理**：Python/JS 用 PyO3/napi-rs 直连 `aibridge-core`，享真正原生 async（无序列化边界）；Go/JVM/.NET 没有跨语言 async FFI 概念，走 `aibridge-ffi` 的 C ABI（阻塞调用 + 各语言原生异步原语包装）。五种语言共享同一个 Rust 核心，绑定层都薄。

**保持原五层心智**：API 层(Client) → 路由层(Router) → 适配器层(Adapter) → 核心层(Core) → 模型层(Model)，仅语言换 Rust、`**kwargs` 换显式 struct、Pydantic 换 serde、动态注册换编译期 match。

---

## 4. Monorepo 布局

```
aibridge/                         # 重构现有 agn-sdk 仓库并更名
├── Cargo.toml                    # workspace 根
├── crates/
│   ├── aibridge-core/            # Rust 核心（纯 async 逻辑，无 FFI 污染）
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── client.rs         # 统一 Client
│   │   │   ├── router.rs         # Router 路由/fallback
│   │   │   ├── adapter/{mod,base,factory}.rs   # Adapter trait + 工厂
│   │   │   ├── adapters/         # 14 个适配器
│   │   │   │   ├── openai_compat.rs             # OpenAI 兼容地基
│   │   │   │   ├── openai.rs agnes.rs azure.rs
│   │   │   │   ├── anthropic.rs gemini.rs volcengine_cv.rs
│   │   │   │   ├── runway.rs pika.rs kling.rs stability.rs
│   │   │   │   ├── chinese.rs aggregation_platforms.rs
│   │   │   │   ├── additional_models.rs more_models.rs emerging_models.rs
│   │   │   │   └── audio_adapters.rs            # edge-tts/elevenlabs/cartesia/deepgram/assemblyai
│   │   │   ├── http.rs           # reqwest+h2 封装
│   │   │   ├── retry.rs          # 重试机制
│   │   │   ├── error.rs          # thiserror 错误枚举
│   │   │   ├── config.rs         # 配置 + 环境变量
│   │   │   ├── util.rs
│   │   │   └── model/{chat,image,video,audio,common,options}.rs  # serde struct
│   │   └── tests/
│   ├── aibridge-ffi/             # C ABI cdylib（给 Go/JVM/.NET）
│   │   ├── src/{lib,handle,error,stream,runtime}.rs
│   │   ├── include/aibridge.h    # cbindgen 生成
│   │   └── cbindgen.toml
│   ├── aibridge-python/          # PyO3 绑定（直连 aibridge-core）
│   │   ├── src/lib.rs            # #[pymodule] aibridge
│   │   └── pyproject.toml        # maturin
│   └── aibridge-node/            # napi-rs 绑定（直连 aibridge-core）
│       ├── src/lib.rs
│       ├── package.json
│       └── build.rs
├── bindings/
│   ├── go/                       # CGO 调 aibridge-ffi（goroutine+channel）
│   │   ├── aibridge.go aibridge.go.h stream.go
│   │   └── go.mod (github.com/aibridge/aibridge-go)
│   ├── jvm/                      # JNA 调 aibridge-ffi（CompletableFuture/Flow，纯 Java 无 native Rust）
│   │   ├── src/main/{java,kotlin}/io/aibridge/...
│   │   └── build.gradle.kts
│   └── dotnet/                   # C# P/Invoke 调 aibridge-ffi（Task/IAsyncEnumerable）
│       ├── AIBridge/ (C# 项目)
│       └── AIBridge.csproj
├── tests/                        # 跨语言一致性测试套件 + 共享 fixture
├── docs/                         # 设计文档 + 各语言迁移指南
└── .github/workflows/            # CI 矩阵（5 语言 × 多平台）
```

**关键决策**：Go/JVM/.NET 统一走 `aibridge-ffi` 这一个 C ABI 入口。JVM 用 JNA（纯 Java，无需写 native Rust），Go 用 CGO，.NET 用 P/Invoke。避免三种静态语言各写一套 Rust native 胶水。

---

## 5. aibridge-core 模块设计

### 5.1 Python → Rust 模块映射

| Python 现有 | Rust (aibridge-core) | 说明 |
|---|---|---|
| `agn/client.py` | `client::Client` | 统一入口，方法签名改显式 Request struct |
| `agn/router.py` | `router::Router` | 多 provider 路由/负载均衡/fallback |
| `adapters/base.py` BaseAdapter | `adapter::Adapter` trait | `#[async_trait]`，不支持能力默认返 UnsupportedCapability |
| `adapters/factory.py` AdapterFactory | `adapter::create_adapter()` 显式 match | 替代运行时注册，新增适配器=加 match 分支 |
| `adapters/*.py`(14个) | `adapters::*`(14个模块) | trait 实现 |
| `core/http_client.py` | `http` (reqwest+h2) | 替代 httpx |
| `core/retry.py` | `retry` | 替代 tenacity |
| `core/errors.py` | `error` (thiserror) | 标准错误枚举 |
| `core/config.py` | `config` | 配置 + 环境变量 |
| `models/*.py` (Pydantic) | `model::*` (serde struct) | 替代 Pydantic |

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

    // 不支持的方法默认返 UnsupportedCapabilityError，trait 提供默认实现
}
```

### 5.3 Client 形态

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

- **可选参数**：Request struct + Builder（`..Default::default()`），替代 `**kwargs`。Provider 特有参数走 `extra: HashMap<String, serde_json::Value>` 透传。
- **流式**：`ChatStream: impl Stream<Item = Result<ChatCompletionChunk>>`（`async-stream` crate），核心内部原生 async stream。
- **工厂**：编译期显式 `match`，替代 Python 运行时 `AdapterFactory.register`。更静态、更安全。

---

## 6. 统一数据模型（serde struct 替代 Pydantic + kwargs）

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
    pub extra: HashMap<String, serde_json::Value>,  // provider 特有参数透传
}
// ChatRequest::builder("gpt-4o", messages).temperature(0.7).max_tokens(1000).build()

#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    System { content: String },
    User { content: UserContent },            // String 或多模态 Vec<ContentPart>
    Assistant { content: String, tool_calls: Option<Vec<ToolCall>> },
    Tool { tool_call_id: String, content: String },
}

pub enum FileInput { Path(String), Url(String), Bytes(Vec<u8>), Base64(String) }
```

- **去掉 Options 中间层**：Python 的 `ChatOptions/ImageOptions/...` 在 Rust 直接用 `Request::builder()` 链式调用替代，迁移指南做 1:1 对照。
- **响应模型**：`ChatCompletion`/`ChatCompletionChunk`/`ImageResult`/`VideoTask`/`VideoStatus`/`EmbeddingResult`/`TranscriptionResult`/`SpeechResult`/`ModelInfo`/`VoiceInfo` 全部 serde struct，字段名与 Python Pydantic 对齐（迁移指南可逐字段对照）。

---

## 7. aibridge-ffi C ABI 边界

**核心原则**：句柄式生命周期 + JSON 字符串作复杂 struct 边界 + 二进制原生传递 + 全局 tokio runtime。

```c
/* 句柄（opaque） */
typedef struct aibridge_client_s aibridge_client_t;
typedef struct aibridge_stream_s aibridge_stream_t;
typedef struct { const uint8_t* ptr; size_t len; } aibridge_bytes_t;

/* 生命周期 */
aibridge_client_t* aibridge_client_new(const char* provider, const char* config_json);
aibridge_status_t   aibridge_client_start(aibridge_client_t*);
void                aibridge_client_destroy(aibridge_client_t*);

/* 阻塞式调用：内部 runtime.block_on，复杂结构走 JSON 边界 */
aibridge_status_t aibridge_client_chat(aibridge_client_t*, const char* request_json,
                                       char** out_response_json);          /* 调用方 aibridge_string_free */
/* 二进制载荷不走 JSON（避免 base64 膨胀） */
aibridge_status_t aibridge_client_speech(aibridge_client_t*, const char* request_json,
                                         aibridge_bytes_t** out_audio, char** out_meta_json);

/* 流式：stream 句柄 + 阻塞 next() */
aibridge_status_t aibridge_client_chat_stream(aibridge_client_t*, const char* request_json,
                                              aibridge_stream_t** out_stream);
aibridge_status_t aibridge_stream_next(aibridge_stream_t*, char** out_chunk_json);  /* 0=chunk, 1=结束, 负=错 */
void              aibridge_stream_destroy(aibridge_stream_t*);   /* 触发 Rust drop → tokio task abort */

/* 错误：返回码 + 线程局部 last_error 槽 */
const char* aibridge_last_error(void);   /* 线程局部，无需释放 */

/* 释放 */
void aibridge_string_free(char*);
void aibridge_bytes_free(aibridge_bytes_t*);
```

**关键决策与权衡**：

1. **JSON 边界 vs C struct mirrorgen**：选 JSON。5 语言 × 大量 struct 用 mirrorgen 维护成本高；JSON 边界让 C ABI 只需 ~20 个函数，各语言绑定只做 JSON (de)serialize，类型安全且跨语言一致。SDK 是 IO 密集，JSON 序列化开销可忽略。
2. **全局 tokio runtime**：`Lazy<TokioRuntime>`（多线程），每个 FFI 调用 `handle().block_on(async{...})`。C 侧可并发调用不同 client；同一 stream 串行（各语言绑定负责同步）。
3. **错误模型**：`aibridge_status_t` 返回码（0 成功 / 负数错误类别）+ `aibridge_last_error()` 线程局部详细消息（JSON，含 `code/message/details/retryable`）。各语言绑定映射成本地异常。
4. **字符串/二进制所有权**：Rust 分配的 `char*`/`aibridge_bytes_t` 必须由调用方调 `aibridge_*_free` 释放。各语言绑定用 RAII 封装（Go finalizer / JVM Cleaner / .NET Dispose）。

---

## 8. 异步与流式跨语言桥接

| 语言 | 绑定路径 | 异步原语 | 流式类型 | cancel 机制 |
|---|---|---|---|---|
| Python | PyO3 **直连** aibridge-core | asyncio 协程 | `AsyncIterator` | asyncio cancel → drop stream |
| JS/TS | napi-rs **直连** aibridge-core | `Promise` | `AsyncIterable` | `AbortSignal` → drop stream |
| Go | CGO → aibridge-ffi | goroutine | `chan Chunk` | `ctx.Done()` → `aibridge_stream_destroy` |
| JVM | JNA → aibridge-ffi | `CompletableFuture` / Kotlin `suspend` | `Flow.Publisher` | cancel → destroy |
| .NET | P/Invoke → aibridge-ffi | `Task<T>` | `IAsyncEnumerable<T>` | `CancellationToken` → destroy |

**直连 vs C ABI**：

- **Python/JS 直连**：无 JSON 边界，PyO3 `#[pyo3] async fn` / napi `#[napi] async fn` 自动桥接原生 async；Rust struct 用 `#[pyclass]`/napi 标注后直接映射为宿主对象。流式内部 `tokio::spawn` + channel 桥接成原生异步迭代器。体验最佳、零序列化。
- **Go/JVM/.NET 走 C ABI**：FFI 调用本身阻塞，各语言在自己的异步执行器上调度（Go goroutine / JVM ForkJoinPool / .NET ThreadPool），把阻塞 FFI 包成原生异步原语。流式用"stream 句柄 + 阻塞 `next()`"在后台线程循环，结果 push 到 channel/Flow/IAsyncEnumerable。
- **统一 cancel**：所有语言的取消最终都落到 `aibridge_stream_destroy`（Rust drop → tokio task abort），语义一致。Python/JS 的直连绑定把 asyncio cancel / AbortSignal 桥接为 Rust 侧 drop stream。

---

## 9. 错误处理与类型映射

### 9.1 Rust 错误枚举（thiserror）

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

对应 Python v1 的 `AGNError` 体系（`AuthenticationError`/`RateLimitError`/`ValidationError`/`ModelNotFoundError`/`APIError`/`TimeoutError`/`UnsupportedCapabilityError`/`ProviderNotFoundError`），分类一致，迁移指南做名称对照（`AGNError` → `AibridgeError`）。

### 9.2 FFI 错误传递

返回码 `aibridge_status_t`（i32）映射错误类别；`aibridge_last_error()` 返回线程局部 JSON：`{"code":"rate_limit","message":"...","retryable":true,"details":{...}}`。

### 9.3 各语言异常映射

- **Python**：`AibridgeError` 基类 + 子类（`RateLimitError` 等），与 v1 同名便于迁移
- **JS/TS**：`AibridgeError` + 子类
- **Go**：`error` + 类型断言接口 `type AibridgeError interface{ Code() string }`
- **JVM**：`AibridgeException` + 子类
- **.NET**：`AibridgeException` + 子类

---

## 10. 适配器迁移策略

### 10.1 抓共性：OpenAI 兼容地基

14 个适配器中 80% 是 OpenAI 兼容协议（agnes/openai/azure/聚合平台/中文平台兼容部分）。Rust 抽出 `OpenAiCompatAdapter` 基础实现（HTTP 请求构造 + 响应解析 + 参数映射），子适配器只 override 差异（base_url、model 列表、参数 mapping、特殊端点）。

### 10.2 参数映射机制 Rust 化

保留 Python `ParameterMapping` 概念，Rust 化为 `param_mapping` 表 + `apply_mapping()`。预置常量：`OPENAI_COMPATIBLE_MAPPING`、`ANTHROPIC_MAPPING`、`GEMINI_MAPPING`、`COHERE_MAPPING`，OpenAI 兼容适配器复用。

### 10.3 分批迁移顺序（MVP 优先）

| 批次 | 适配器 | 协议 | 优先级 |
|---|---|---|---|
| **阶段 1 · MVP** | openai, agnes, volcengine_cv, gemini | 兼容地基 + 独立协议 | **最高** |
| 阶段 2a | azure, aggregation_platforms, additional_models, more_models, emerging_models, chinese(兼容部分) | OpenAI 兼容族 | 高 |
| 阶段 2b | anthropic, stability, runway, pika, kling | 独立协议 | 中 |
| 阶段 2c | edge-tts, elevenlabs, cartesia, deepgram, assemblyai (audio_adapters) | 二进制载荷 | 中 |

> 工作量估算：~14000 行 Python，共享地基后实际新写 Rust 约 6000–8000 行。

### 10.4 保留特性

- v1.1.0 的"实时拉取模型列表"（list_models 调 provider /models 端点）保留
- v1.3.0 的"免费 Provider 免认证"（edge-tts 无需 api_key，`requires_api_key() -> false`）保留
- v1.3.3 的"TTS 音色健康检查/推荐/自动降级"保留

---

## 11. 测试策略

三层测试：

1. **Rust 核心单测**（aibridge-core 内）：每个适配器 mock HTTP，覆盖正常 + 异常 + 边界。覆盖率 ≥80%。
2. **各语言绑定测试**：每种语言测自己的胶水层（句柄生命周期、异步桥接、错误映射、流式）。
3. **跨语言一致性测试**：同一组输入 + mock，五种语言跑一遍，断言输出一致。防止单种语言绑定漏实现或行为漂移。

**共享 fixture**：Python v1 测试里的 HTTP 请求/响应 mock 数据抽成 JSON 文件（`tests/fixtures/`），Rust 测试 + 五语言绑定测试共用。保证"五语言行为一致 + 与 Python 老版一致"。**这是质量保险的核心。**

平时测试不调真实 AI API（成本高、不稳定）。另留"真实接口冒烟测试"开关（环境变量配 key，CI 默认关）。

---

## 12. 构建与发布

### 12.1 各语言打包方式

| 语言 | 工具 | 发布到 | 用户安装体验 |
|---|---|---|---|
| Rust 核心 + ffi | cargo | —（内部） | 产出 libaibridge.{so,dylib,dll} |
| Python | maturin | PyPI `aibridge` v2.0.0 | `pip install aibridge`，预编译 wheel，无需 Rust |
| JS/TS | napi-rs | npm `aibridge` | `npm install aibridge`，预编译 .node，无需 Rust |
| Go | cgo | Go module `aibridge-go` | 需单独装 libaibridge（提供安装脚本，Go 生态惯例） |
| JVM | JNA + Gradle | Maven `io.aibridge:aibridge` | 动态库打进 jar（按平台 classifier），用户无感 |
| .NET | P/Invoke | NuGet `AIBridge` | 动态库打进包（runtimes/{rid}/native/），用户无感 |

### 12.2 二进制分发难点

Go/JVM/.NET 依赖 `libaibridge` 动态库。JVM 和 .NET 把动态库打进各自包里（按 OS/arch 分包），用户无感；Go 因 cgo 机制，用户单独装动态库（提供安装脚本），属 Go 生态常规做法。

### 12.3 CI 矩阵

GitHub Actions：平台（linux/macos/windows × amd64/arm64）× 语言。aibridge-ffi 的动态库作为构建 artifact 供 Go/JVM/.NET 打包消费。Rust 核心 + 5 绑定各一个 workflow。

---

## 13. 分阶段实施计划

> 时间为全职单人粗估；实施阶段用多 agent 并行可加速。

### 阶段 0 · 打地基（2–3 周）

- Monorepo 搭建，Cargo workspace
- aibridge-core 骨架：error/config/http/retry/model/Adapter trait/Client/Router
- aibridge-ffi C ABI 骨架 + 全局 runtime + cbindgen
- 跨语言管线打通：用 openai 的 chat stub 跑通五语言 hello world（含 async + 流式 + 错误），验证最大技术风险

### 阶段 1 · MVP 四 provider（3–4 周）⭐

openai → agnes → volcengine_cv → gemini，逐个 Rust 实现并跑通五语言 + 测试。**做完这步，用户另一项目即可开始接入。**

### 阶段 2 · 剩余适配器（4–6 周）

按 2a(兼容族) → 2b(独立协议) → 2c(音频) 三批搬完剩下 10 个。

### 阶段 3 · 发布收尾（2–3 周）

全平台 CI 构建、五语言包正式发版、Python v1→v2 迁移指南、文档网站、老版 v1 归档、打 v2.0.0 tag。

**总周期约 3–4 个月**（全职单人）。

### 多 agent 并行编排策略（实施阶段）

- **适配器迁移**：每个 provider 一个 agent 并行（共享 OpenAiCompatAdapter 基础），fixture 共享保证一致
- **语言绑定**：aibridge-core 与 aibridge-ffi 稳定后，五种语言绑定各一个 agent 并行
- **测试**：跨语言一致性测试单独 agent 汇总
- **依赖顺序**：阶段 0 必须串行（核心未稳定前绑定无法并行）；阶段 1+ 适配器与绑定可并行

---

## 14. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 全原生 async 跨 FFI 复杂 | 高 | PyO3/napi 直连避开 C ABI async；静态语言用阻塞+原生异步包装，已在设计中固化 |
| 14 适配器全量迁移周期长 | 中 | OpenAI 兼容地基复用 80%；MVP 四 provider 优先保证早期可用 |
| 五语言行为漂移 | 中 | 共享 fixture + 跨语言一致性测试 |
| Go/JVM/.NET 动态库分发 | 中 | JVM/.NET 打进包；Go 提供安装脚本 |
| crates.io/包名占用 | 低 | PyPI/npm 已确认 aibridge 可用；crates.io/Maven/NuGet 发布前再确认，可用后缀规避 |
| Python 老用户升级破坏 | 中 | v2.0.0 semver + 迁移指南 + 旧版 v1 归档保留 |

---

## 15. Python v1→v2 迁移指南要点

- 包名：`agn-sdk` → `aibridge`，`from agn import Client` → `from aibridge import Client`
- 错误类：`AGNError` → `AibridgeError`（子类名不变）
- 参数：`**kwargs` 透传 → `Request` struct + Builder 链式调用（`ChatOptions` → `ChatRequest::builder()`）
- Options 类：`ChatOptions/ImageOptions/...` 中间层去除，直接用 Request builder
- 其余方法名、能力、provider 名保持一致

---

## 附录 A：与 Python v1 能力对照

| 能力 | v1 (agn-sdk) | v2 (aibridge) | 状态 |
|---|---|---|---|
| chat (含流式) | ✅ | ✅ | 迁移 |
| image_generate | ✅ | ✅ | 迁移 |
| video_create + poll | ✅ | ✅ | 迁移 |
| transcribe (ASR) | ✅ | ✅ | 迁移 |
| speech (TTS) | ✅ | ✅ | 迁移 |
| embed | ✅ | ✅ | 迁移 |
| list_models (实时拉取) | ✅ | ✅ | 迁移 |
| list_voices / recommend_voices | ✅ | ✅ | 迁移 |
| Router (多 provider 路由) | ✅ | ✅ | 迁移 |
| 音色自动降级 | ✅ | ✅ | 迁移 |
