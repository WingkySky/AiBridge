//! AIBridge Python 绑定（PyO3）
//!
//! 直连 aibridge-core，原生 asyncio 协程与 AsyncIterator 流式。
//! 由 maturin 构建为 PyPI 包 `aibridge`。
//!
//! 阶段 0.6 实现：Client / chat / speech / chat_stream / 错误映射 / 数据模型。
//!
//! 架构要点：
//! - 全局多线程 tokio runtime：core 的 async future（含 reqwest 等真实 IO）spawn 到
//!   tokio 上执行，PyO3 协程通过 await JoinHandle 拿结果。echo adapter 无网络也走
//!   同一路径，保持一致。
//! - `PyClient` 持有 `Arc<tokio::sync::Mutex<Client>>`，支持 `start`/`close` 可变操作。
//! - 流式：`chat_stream` 把 core `ChatStream`（`BoxStream`）通过 `tokio::sync::Mutex`
//!   封装进 `ChatStreamIterator`。`__anext__` 返回 PyO3 内置的
//!   [`pyo3::coroutine::Coroutine`]，其包裹的 Rust future 在 tokio runtime 上
//!   `spawn` 消费 stream（真实 IO 在 tokio worker 线程，不阻塞 asyncio 事件循环）。
//!   `Coroutine` 的 waker 通过 `asyncio.Future` + `call_soon_threadsafe` 把
//!   "chunk 就绪"通知回 asyncio 事件循环，await 期间让出线程给其他协程。

// PyO3 0.28 的 `#[pymethods]` 宏展开会用到 Rust 1.77+ 稳定的语法（如 let 链），
// 与 workspace MSRV 1.75 冲突。该 lint 针对宏生成代码，非手写代码，故整体允许。
#![allow(clippy::incompatible_msrv)]

use std::sync::Arc;

use futures::StreamExt;
use pyo3::coroutine::Coroutine;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use tokio::sync::Mutex;

use aibridge_core::adapter::ChatStream as CoreChatStream;
use aibridge_core::client::Client as CoreClient;
use aibridge_core::config::ClientOptions as CoreClientOptions;
use aibridge_core::error::AibridgeError as CoreAibridgeError;
use aibridge_core::model::audio::{
    SpeechRequest as CoreSpeechRequest, SpeechResult as CoreSpeechResult,
};
use aibridge_core::model::chat::{
    ChatCompletion as CoreChatCompletion, ChatCompletionChunk as CoreChatCompletionChunk,
    ChatMessage as CoreChatMessage, ChatRequest as CoreChatRequest,
};

// ===========================================================================
// 全局 tokio runtime
// ===========================================================================

/// 全局多线程 tokio runtime
///
/// 用 `once_cell::sync::Lazy` 在首次访问时初始化。core 的 async future（含
/// reqwest 等真实 IO）spawn 到此 runtime 上执行，PyO3 协程通过 await JoinHandle
/// 取回结果。多线程 runtime 保证真实 adapter 的并发能力。
static RUNTIME: once_cell::sync::Lazy<tokio::runtime::Runtime> = once_cell::sync::Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("初始化 tokio runtime 失败")
});

// ===========================================================================
// 错误映射
// ===========================================================================

// 异常类层级（设计文档 9.3 节）：
//   AibridgeError (基类，继承 PyException)
//     ├── AuthenticationError
//     ├── RateLimitError
//     ├── ValidationError
//     ├── ModelNotFoundError
//     ├── APIError
//     ├── NetworkError
//     ├── TimeoutError
//     ├── UnsupportedCapabilityError
//     ├── ProviderNotFoundError
//     ├── VoiceNotAvailableError
//     └── ServiceUnavailableError
//
// 子类名与 Python v1 (agn-sdk) 保持一致，便于迁移（v1 `AGNError` → v2 `AibridgeError`）。
// 用 `create_exception!` 宏生成，构造方式 `XxxError::new_err(message)`。
use pyo3::create_exception;

create_exception!(
    aibridge,
    AibridgeError,
    pyo3::exceptions::PyException,
    "AIBridge SDK 错误基类"
);
create_exception!(
    aibridge,
    AuthenticationError,
    AibridgeError,
    "认证失败（API Key 无效/过期/无权限）"
);
create_exception!(aibridge, RateLimitError, AibridgeError, "请求频率超过限制");
create_exception!(aibridge, ValidationError, AibridgeError, "请求参数校验错误");
create_exception!(
    aibridge,
    ModelNotFoundError,
    AibridgeError,
    "请求的模型不存在"
);
create_exception!(aibridge, APIError, AibridgeError, "Provider API 调用错误");
create_exception!(aibridge, NetworkError, AibridgeError, "网络错误");
create_exception!(aibridge, TimeoutError, AibridgeError, "请求超时");
create_exception!(
    aibridge,
    UnsupportedCapabilityError,
    AibridgeError,
    "Provider 不支持该能力"
);
create_exception!(
    aibridge,
    ProviderNotFoundError,
    AibridgeError,
    "Provider 不存在"
);
create_exception!(
    aibridge,
    VoiceNotAvailableError,
    AibridgeError,
    "音色不可用"
);
create_exception!(
    aibridge,
    ServiceUnavailableError,
    AibridgeError,
    "服务暂时不可用（可重试）"
);

/// 将 core `AibridgeError` 映射为对应的 Python 异常 `PyErr`
///
/// 对齐设计文档 9.3 节错误映射表。消息格式为 `[code] message`，便于调用方
/// 获取稳定标识码（core `AibridgeError::code()`）。
fn map_error(err: CoreAibridgeError) -> PyErr {
    let code = err.code();
    let message = format!("[{code}] {err}");
    match err {
        CoreAibridgeError::Authentication { .. } => AuthenticationError::new_err(message),
        CoreAibridgeError::RateLimit { .. } => RateLimitError::new_err(message),
        CoreAibridgeError::Validation { .. } => ValidationError::new_err(message),
        CoreAibridgeError::ModelNotFound { .. } => ModelNotFoundError::new_err(message),
        CoreAibridgeError::Api { .. } => APIError::new_err(message),
        CoreAibridgeError::Network(_) => NetworkError::new_err(message),
        CoreAibridgeError::Timeout => TimeoutError::new_err(message),
        CoreAibridgeError::UnsupportedCapability { .. } => {
            UnsupportedCapabilityError::new_err(message)
        }
        CoreAibridgeError::ProviderNotFound { .. } => ProviderNotFoundError::new_err(message),
        CoreAibridgeError::VoiceNotAvailable { .. } => VoiceNotAvailableError::new_err(message),
        CoreAibridgeError::ServiceUnavailable { .. } => ServiceUnavailableError::new_err(message),
    }
}

// ===========================================================================
// 数据模型
// ===========================================================================

/// 对话消息
///
/// 对应 core `ChatMessage`。Python 侧用 dict 构造（`{"role": "user", "content": "..."}`），
/// 仅支持 chat hello world 所需的 system/user/assistant 文本消息。
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
struct ChatMessage {
    /// 角色（system / user / assistant / tool）
    role: String,
    /// 文本内容
    content: String,
}

#[pymethods]
impl ChatMessage {
    #[new]
    #[pyo3(signature = (role, content))]
    fn new(role: String, content: String) -> Self {
        Self { role, content }
    }

    #[getter]
    fn role(&self) -> String {
        self.role.clone()
    }

    #[getter]
    fn content(&self) -> String {
        self.content.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChatMessage(role={:?}, content={:?})",
            self.role, self.content
        )
    }
}

impl ChatMessage {
    /// 将 Python 侧消息（dict 或 ChatMessage）转换为 core `ChatMessage`
    ///
    /// 支持 dict 形式 `{"role": "user", "content": "..."}` 与 `ChatMessage` 实例。
    /// 仅处理文本内容；多模态内容后续阶段支持。
    fn to_core(obj: &Bound<'_, PyAny>) -> PyResult<CoreChatMessage> {
        if let Ok(msg) = obj.extract::<PyRef<'_, ChatMessage>>() {
            return Self::from_role_content(&msg.role, &msg.content);
        }
        let dict: std::collections::HashMap<String, String> = obj.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "消息必须是 ChatMessage 或含 role/content 的 dict",
            )
        })?;
        let role = dict
            .get("role")
            .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("消息缺少 role 字段"))?;
        let content = dict
            .get("content")
            .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("消息缺少 content 字段"))?;
        Self::from_role_content(role, content)
    }

    /// 按角色构造 core `ChatMessage`（仅文本）
    fn from_role_content(role: &str, content: &str) -> PyResult<CoreChatMessage> {
        match role {
            "system" => Ok(CoreChatMessage::system(content)),
            "user" => Ok(CoreChatMessage::user(content)),
            "assistant" => Ok(CoreChatMessage::assistant(content)),
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "不支持的消息角色: {other}（阶段 0.6 仅支持 system/user/assistant）"
            ))),
        }
    }
}

/// 对话完成结果
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct ChatCompletion {
    id: String,
    model: String,
    choices: Vec<ChatChoice>,
}

#[pymethods]
impl ChatCompletion {
    #[getter]
    fn id(&self) -> String {
        self.id.clone()
    }

    #[getter]
    fn model(&self) -> String {
        self.model.clone()
    }

    #[getter]
    fn choices(&self) -> Vec<ChatChoice> {
        self.choices.clone()
    }

    fn __repr__(&self) -> String {
        format!("ChatCompletion(id={:?}, model={:?})", self.id, self.model)
    }
}

impl ChatCompletion {
    /// 从 core `ChatCompletion` 构造 Python 包装
    fn from_core(c: CoreChatCompletion) -> Self {
        Self {
            id: c.id,
            model: c.model,
            choices: c
                .choices
                .into_iter()
                .map(|ch| ChatChoice {
                    index: ch.index,
                    message: ChoiceMessage {
                        role: ch.message.role,
                        content: ch.message.content.unwrap_or_default(),
                        finish_reason: ch.finish_reason.unwrap_or_default(),
                    },
                })
                .collect(),
        }
    }
}

/// 对话选项（choices 元素）
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct ChatChoice {
    index: u32,
    message: ChoiceMessage,
}

#[pymethods]
impl ChatChoice {
    #[getter]
    fn index(&self) -> u32 {
        self.index
    }

    #[getter]
    fn message(&self) -> ChoiceMessage {
        self.message.clone()
    }

    fn __repr__(&self) -> String {
        format!("ChatChoice(index={})", self.index)
    }
}

/// 选项中的消息
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct ChoiceMessage {
    role: String,
    content: String,
    finish_reason: String,
}

#[pymethods]
impl ChoiceMessage {
    #[getter]
    fn role(&self) -> String {
        self.role.clone()
    }

    #[getter]
    fn content(&self) -> String {
        self.content.clone()
    }

    #[getter]
    fn finish_reason(&self) -> String {
        self.finish_reason.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChoiceMessage(role={:?}, content={:?})",
            self.role, self.content
        )
    }
}

/// 流式对话块
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct ChatCompletionChunk {
    id: String,
    model: String,
    choices: Vec<ChatChunkDelta>,
}

#[pymethods]
impl ChatCompletionChunk {
    #[getter]
    fn id(&self) -> String {
        self.id.clone()
    }

    #[getter]
    fn model(&self) -> String {
        self.model.clone()
    }

    #[getter]
    fn choices(&self) -> Vec<ChatChunkDelta> {
        self.choices.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChatCompletionChunk(id={:?}, model={:?})",
            self.id, self.model
        )
    }
}

impl ChatCompletionChunk {
    fn from_core(c: CoreChatCompletionChunk) -> Self {
        Self {
            id: c.id,
            model: c.model,
            choices: c
                .choices
                .into_iter()
                .map(|d| ChatChunkDelta {
                    index: d.index,
                    role: d.delta.role.unwrap_or_default(),
                    content: d.delta.content.unwrap_or_default(),
                    finish_reason: d.finish_reason.unwrap_or_default(),
                })
                .collect(),
        }
    }
}

/// 流式块中的增量
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct ChatChunkDelta {
    index: u32,
    role: String,
    content: String,
    finish_reason: String,
}

#[pymethods]
impl ChatChunkDelta {
    #[getter]
    fn index(&self) -> u32 {
        self.index
    }

    #[getter]
    fn role(&self) -> String {
        self.role.clone()
    }

    #[getter]
    fn content(&self) -> String {
        self.content.clone()
    }

    #[getter]
    fn finish_reason(&self) -> String {
        self.finish_reason.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChatChunkDelta(index={}, content={:?})",
            self.index, self.content
        )
    }
}

/// 文字转语音结果
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct SpeechResult {
    /// 音频二进制数据（bytes）
    audio_data: Vec<u8>,
    /// 音频 MIME 类型
    content_type: String,
    /// 音频格式（mp3/wav 等）
    format: String,
    /// 估计音频时长（秒）
    duration: Option<f64>,
    /// 使用的模型 ID
    model: Option<String>,
}

#[pymethods]
impl SpeechResult {
    /// 音频二进制数据（Python `bytes`）
    #[getter]
    fn audio_data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.audio_data)
    }

    /// 音频数据长度（便捷访问）
    #[getter]
    fn size(&self) -> usize {
        self.audio_data.len()
    }

    #[getter]
    fn content_type(&self) -> String {
        self.content_type.clone()
    }

    #[getter]
    fn format(&self) -> String {
        self.format.clone()
    }

    #[getter]
    fn duration(&self) -> Option<f64> {
        self.duration
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.model.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SpeechResult(size={}, format={:?})",
            self.audio_data.len(),
            self.format
        )
    }
}

impl SpeechResult {
    fn from_core(r: CoreSpeechResult) -> Self {
        Self {
            audio_data: r.audio_data.unwrap_or_default(),
            content_type: r.content_type,
            format: r.format,
            duration: r.duration,
            model: r.model,
        }
    }
}

// ===========================================================================
// 流式迭代器
// ===========================================================================

/// 流式对话迭代器
///
/// 由 `Client.chat_stream` 返回，实现 `__aiter__`/`__anext__` 协议。
/// `async for chunk in stream:` 每次取一个 `ChatCompletionChunk`，流结束抛
/// `StopAsyncIteration`。
///
/// 实现说明（不阻塞 asyncio 事件循环）：
/// `__anext__` 同步返回一个 PyO3 内置的 [`Coroutine`]（可 `await` 的 Python 对象），
/// 其包裹的 Rust future 在全局 tokio runtime 上 `spawn` 消费 core `ChatStream`：
/// - 真实 adapter 的 reqwest IO 在 tokio worker 线程执行，asyncio 线程仅 await
///   `JoinHandle`（Pending 时让出，不阻塞事件循环，其他协程可运行）。
/// - chunk 就绪后，`Coroutine` 的 `AsyncioWaker` 通过 `asyncio.Future` +
///   `call_soon_threadsafe` 把就绪通知调度回 asyncio 事件循环（PyO3 内置实现，
///   无需手写 loop 引用），`await` 返回 chunk。
/// - 流结束：future 返回 `Err(StopAsyncIteration)`；取 chunk 出错：返回对应
///   `AibridgeError` 子类；正常 chunk：`Ok(chunk)` → `StopIteration(chunk)`。
///
/// GIL 处理：future 在 tokio 上 await stream 期间不持 GIL（`spawn` 的 task 在
/// tokio worker 跑），仅在拿到 chunk 后 `Python::with_gil` 构造 Python 对象。
#[pyclass]
struct ChatStreamIterator {
    /// core 流（None 表示已耗尽）
    inner: Arc<Mutex<Option<CoreChatStream>>>,
}

#[pymethods]
impl ChatStreamIterator {
    /// 返回自身（异步迭代器协议：`__aiter__` 返回 self）
    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// 取下一个 chunk（同步返回 `Coroutine`，可 `await`）
    ///
    /// 返回的 `Coroutine` `await` 后得到 `ChatCompletionChunk`，或抛
    /// `StopAsyncIteration`（流结束）/ 对应 `AibridgeError` 子类（取 chunk 出错）。
    ///
    /// 不阻塞事件循环：实际取 chunk 的 future 在 tokio runtime 上推进，
    /// `Coroutine` 的 waker 负责把就绪通知桥接回 asyncio 事件循环。
    fn __anext__(&self, py: Python<'_>) -> PyResult<Py<Coroutine>> {
        let inner = self.inner.clone();

        // 构造包裹"取下一个 chunk"逻辑的 future。该 future 在 Coroutine 被
        // poll 时推进（poll 发生在 asyncio 线程，持 GIL），但其内部把 stream
        // 消费 spawn 到 tokio runtime，await JoinHandle 期间 Pending 让出线程。
        let fut = async move {
            // 在 tokio runtime 上消费 stream。spawn 后 await JoinHandle：
            // - stream.next()（含真实 reqwest IO）在 tokio worker 线程执行
            // - asyncio 线程仅 poll JoinHandle，Pending 时注册 waker 让出
            let join_result = RUNTIME
                .spawn(async move {
                    let mut guard = inner.lock().await;
                    match guard.as_mut() {
                        None => None,
                        Some(stream) => stream.next().await,
                    }
                })
                .await;

            // JoinError（task panic/取消）→ RuntimeError
            let item: Option<Result<CoreChatCompletionChunk, CoreAibridgeError>> = join_result
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "chat_stream 消费任务失败: {e}"
                    ))
                })?;

            // 在 tokio worker 线程拿到 item，需重新进入 GIL 上下文构造 Python 对象。
            // Coroutine future 被 poll 时所在线程（asyncio 线程）已 attached GIL，
            // `Python::attach` 在已 attached 线程上直接复用（返回 R），安全构造 pyclass。
            Python::attach(|py: Python<'_>| -> PyResult<Py<PyAny>> {
                match item {
                    // 流结束 → 抛 StopAsyncIteration（await 时终止 async for）
                    None => Err(pyo3::exceptions::PyStopAsyncIteration::new_err(())),
                    // 正常 chunk → 返回 chunk 对象（Coroutine 抛 StopIteration(chunk)）
                    Some(Ok(c)) => {
                        let chunk = Py::new(py, ChatCompletionChunk::from_core(c))?;
                        Ok(chunk.into_any())
                    }
                    // 取 chunk 出错 → 抛对应 AibridgeError 子类
                    Some(Err(e)) => Err(map_error(e)),
                }
            })
        };

        // 用 PyO3 内置 Coroutine 包装 future。Coroutine 实现 __await__/__next__/send，
        // 可直接被 `await`。其 waker 自动桥接 tokio 唤醒 → asyncio.Future.set_result
        // （通过 call_soon_threadsafe），无需手写 asyncio loop 引用。
        let name = PyString::new(py, "ChatStreamIterator.__anext__");
        let coroutine =
            pyo3::impl_::coroutine::new_coroutine(&name, Some("ChatStreamIterator"), None, fut);
        Py::new(py, coroutine)
    }
}

// ===========================================================================
// 客户端
// ===========================================================================

/// AIBridge 统一客户端
///
/// 对应 Python v1 `Client`，是用户使用 SDK 的唯一入口。
///
/// 示例：
/// ```python
/// import asyncio
/// from aibridge import Client
///
/// async def main():
///     client = Client(provider="echo")
///     await client.start()
///     resp = await client.chat(model="echo-chat",
///                              messages=[{"role": "user", "content": "hello"}])
///     print(resp.choices[0].message.content)
///     await client.close()
///
/// asyncio.run(main())
/// ```
#[pyclass]
struct Client {
    /// core 客户端（用 tokio Mutex 保护以支持 start/close 可变操作）
    inner: Arc<Mutex<CoreClient>>,
    /// Provider 类型（构造后不变，缓存以避免同步 getter 中 block_on）
    provider_type: String,
}

#[pymethods]
impl Client {
    /// 创建客户端
    ///
    /// 参数：
    /// - `provider`: Provider 类型（如 "echo"、"openai"）
    /// - `api_key`: 可选 API Key（免认证 provider 可省略）
    /// - `base_url`: 可选 API Base URL
    #[new]
    #[pyo3(signature = (provider, *, api_key=None, base_url=None))]
    fn new(provider: &str, api_key: Option<String>, base_url: Option<String>) -> PyResult<Self> {
        let mut opts_builder = CoreClientOptions::builder();
        if let Some(key) = api_key {
            opts_builder = opts_builder.api_key(key);
        }
        if let Some(url) = base_url {
            opts_builder = opts_builder.base_url(url);
        }
        let opts = opts_builder.build();
        let core_client = CoreClient::new(provider, opts).map_err(map_error)?;
        let provider_type = core_client.provider_type().to_string();
        Ok(Self {
            inner: Arc::new(Mutex::new(core_client)),
            provider_type,
        })
    }

    /// Provider 类型
    #[getter]
    fn provider_type(&self) -> String {
        self.provider_type.clone()
    }

    /// 启动客户端（初始化适配器）
    async fn start(&self) -> PyResult<()> {
        let inner = self.inner.clone();
        let result = RUNTIME
            .spawn(async move {
                let mut client = inner.lock().await;
                client.start().await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("start 任务失败: {e}"))
            })?;
        result.map_err(map_error)
    }

    /// 关闭客户端（释放资源）
    async fn close(&self) -> PyResult<()> {
        let inner = self.inner.clone();
        let result = RUNTIME
            .spawn(async move {
                let mut client = inner.lock().await;
                client.close().await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("close 任务失败: {e}"))
            })?;
        result.map_err(map_error)
    }

    /// 文本对话
    ///
    /// 参数：
    /// - `model`: 模型名称
    /// - `messages`: 消息列表（`ChatMessage` 或 `{"role":..., "content":...}` dict）
    /// - `temperature`: 可选温度系数
    /// - `max_tokens`: 可选最大 token 数
    #[pyo3(signature = (model, messages, *, temperature=None, max_tokens=None))]
    async fn chat(
        &self,
        model: String,
        messages: Vec<Py<PyAny>>,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> PyResult<ChatCompletion> {
        // 在持有 GIL 时把 Python 消息转换为 core 消息
        let core_messages = Python::attach(|py| {
            messages
                .iter()
                .map(|m| ChatMessage::to_core(m.bind(py)))
                .collect::<PyResult<Vec<CoreChatMessage>>>()
        })?;

        let mut builder = CoreChatRequest::builder(model, core_messages);
        if let Some(t) = temperature {
            builder = builder.temperature(t);
        }
        if let Some(m) = max_tokens {
            builder = builder.max_tokens(m);
        }
        let req = builder.build();

        let inner = self.inner.clone();
        let result = RUNTIME
            .spawn(async move {
                let client = inner.lock().await;
                client.chat(req).await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("chat 任务失败: {e}"))
            })?;

        let completion = result.map_err(map_error)?;
        Ok(ChatCompletion::from_core(completion))
    }

    /// 流式文本对话
    ///
    /// 返回 `ChatStreamIterator`（异步迭代器），`async for chunk in ...` 逐块消费。
    ///
    /// 参数同 `chat`。
    #[pyo3(signature = (model, messages, *, temperature=None, max_tokens=None))]
    async fn chat_stream(
        &self,
        model: String,
        messages: Vec<Py<PyAny>>,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> PyResult<ChatStreamIterator> {
        let core_messages = Python::attach(|py| {
            messages
                .iter()
                .map(|m| ChatMessage::to_core(m.bind(py)))
                .collect::<PyResult<Vec<CoreChatMessage>>>()
        })?;

        let mut builder = CoreChatRequest::builder(model, core_messages).stream(true);
        if let Some(t) = temperature {
            builder = builder.temperature(t);
        }
        if let Some(m) = max_tokens {
            builder = builder.max_tokens(m);
        }
        let req = builder.build();

        let inner = self.inner.clone();
        let stream_result = RUNTIME
            .spawn(async move {
                let client = inner.lock().await;
                client.chat_stream(req).await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("chat_stream 任务失败: {e}"))
            })?;

        let stream = stream_result.map_err(map_error)?;
        Ok(ChatStreamIterator {
            inner: Arc::new(Mutex::new(Some(stream))),
        })
    }

    /// 文字转语音
    ///
    /// 参数：
    /// - `model`: TTS 模型名称
    /// - `input`: 要合成的文本
    /// - `voice`: 音色（字符串）
    /// - `response_format`: 可选音频格式（默认 mp3）
    /// - `speed`: 可选语速（0.25-4.0）
    #[pyo3(signature = (model, input, voice, *, response_format=None, speed=None))]
    async fn speech(
        &self,
        model: String,
        input: String,
        voice: String,
        response_format: Option<String>,
        speed: Option<f64>,
    ) -> PyResult<SpeechResult> {
        let mut builder = CoreSpeechRequest::builder(model, input, voice);
        if let Some(f) = response_format {
            builder = builder.response_format(f);
        }
        if let Some(s) = speed {
            builder = builder.speed(s);
        }
        let req = builder.build();

        let inner = self.inner.clone();
        let result = RUNTIME
            .spawn(async move {
                let client = inner.lock().await;
                client.speech(req).await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("speech 任务失败: {e}"))
            })?;

        let speech = result.map_err(map_error)?;
        Ok(SpeechResult::from_core(speech))
    }
}

// ===========================================================================
// 模块入口
// ===========================================================================

/// Python 模块入口：`import aibridge`
///
/// 函数名为 `_aibridge`（与 `[lib] name = "_aibridge"` 一致），生成 `PyInit__aibridge`
/// 符号。`#[pyo3(name = "aibridge")]` 把模块的 Python 名重写为 `aibridge`，配合
/// pyproject.toml 的 `module-name = "aibridge"`，使 `import aibridge` 正常工作。
/// lib name 用下划线前缀以避免与 aibridge-ffi 的 `libaibridge.dylib` 产物冲突。
#[pymodule]
#[pyo3(name = "aibridge")]
fn _aibridge(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    // 触发全局 runtime 初始化（首次访问 Lazy 即建）
    let _ = &*RUNTIME;

    // 错误类（基类 + 子类）
    m.add("AibridgeError", py.get_type::<AibridgeError>())?;
    m.add("AuthenticationError", py.get_type::<AuthenticationError>())?;
    m.add("RateLimitError", py.get_type::<RateLimitError>())?;
    m.add("ValidationError", py.get_type::<ValidationError>())?;
    m.add("ModelNotFoundError", py.get_type::<ModelNotFoundError>())?;
    m.add("APIError", py.get_type::<APIError>())?;
    m.add("NetworkError", py.get_type::<NetworkError>())?;
    m.add("TimeoutError", py.get_type::<TimeoutError>())?;
    m.add(
        "UnsupportedCapabilityError",
        py.get_type::<UnsupportedCapabilityError>(),
    )?;
    m.add(
        "ProviderNotFoundError",
        py.get_type::<ProviderNotFoundError>(),
    )?;
    m.add(
        "VoiceNotAvailableError",
        py.get_type::<VoiceNotAvailableError>(),
    )?;
    m.add(
        "ServiceUnavailableError",
        py.get_type::<ServiceUnavailableError>(),
    )?;

    // 数据模型
    m.add_class::<ChatMessage>()?;
    m.add_class::<ChatCompletion>()?;
    m.add_class::<ChatChoice>()?;
    m.add_class::<ChoiceMessage>()?;
    m.add_class::<ChatCompletionChunk>()?;
    m.add_class::<ChatChunkDelta>()?;
    m.add_class::<SpeechResult>()?;

    // 客户端与流式
    m.add_class::<Client>()?;
    m.add_class::<ChatStreamIterator>()?;

    // 模块版本
    m.add("__version__", aibridge_core::VERSION)?;

    Ok(())
}
