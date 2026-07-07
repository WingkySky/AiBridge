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
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString, PyTuple};
use tokio::sync::Mutex;

use aibridge_core::adapter::ChatStream as CoreChatStream;
use aibridge_core::client::Client as CoreClient;
use aibridge_core::config::ClientOptions as CoreClientOptions;
use aibridge_core::error::AibridgeError as CoreAibridgeError;
use aibridge_core::model::audio::{
    SpeechRequest as CoreSpeechRequest, SpeechResult as CoreSpeechResult,
    TranscribeRequest as CoreTranscribeRequest, TranscriptionResult as CoreTranscriptionResult,
    TranscriptionSegment as CoreTranscriptionSegment, TranscriptionWord as CoreTranscriptionWord,
};
use aibridge_core::model::chat::{
    ChatCompletion as CoreChatCompletion, ChatCompletionChunk as CoreChatCompletionChunk,
    ChatMessage as CoreChatMessage, ChatRequest as CoreChatRequest,
};
use aibridge_core::model::common::{
    ModelInfo as CoreModelInfo, ModelType as CoreModelType, TaskStatus as CoreTaskStatus,
    VideoMode as CoreVideoMode, VoiceInfo as CoreVoiceInfo,
};
use aibridge_core::model::image::{
    FileInput as CoreFileInput, ImageData as CoreImageData, ImageRequest as CoreImageRequest,
    ImageResult as CoreImageResult,
};
use aibridge_core::model::options::{
    EmbedInput as CoreEmbedInput, EmbedRequest as CoreEmbedRequest,
    EmbeddingItem as CoreEmbeddingItem, EmbeddingResult as CoreEmbeddingResult,
    EmbeddingVector as CoreEmbeddingVector,
};
use aibridge_core::model::video::{
    VideoRequest as CoreVideoRequest, VideoStatus as CoreVideoStatus, VideoTask as CoreVideoTask,
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
// Python ↔ core 转换辅助
// ===========================================================================

/// 将 Python `**kwargs` 字典转换为 core `extra` 透传参数表
///
/// 用于 image_generate / video_create / transcribe / embed 等方法的厂商特有参数透传。
/// 每个值经 [`py_to_json`] 转为 `serde_json::Value`。
fn kwargs_to_extra(
    d: &Bound<'_, PyDict>,
) -> PyResult<std::collections::HashMap<String, serde_json::Value>> {
    let mut map = std::collections::HashMap::new();
    for (k, v) in d.iter() {
        let key = k.extract::<String>()?;
        map.insert(key, py_to_json(&v)?);
    }
    Ok(map)
}

/// 将任意 Python 对象转换为 `serde_json::Value`
///
/// 支持 None/bool/int/float/str/list/tuple/dict，其余类型兜底为字符串表示。
/// bool 必须先于 int 判断（Python 中 bool 是 int 子类，`extract::<i64>` 对 True 返 Ok）。
fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    // bool 先于 int 判断
    if obj.cast::<PyBool>().is_ok() {
        let b: bool = obj.extract()?;
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(serde_json::json!(i));
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(serde_json::json!(u));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::json!(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in d.iter() {
            map.insert(k.extract::<String>()?, py_to_json(&v)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    if let Ok(l) = obj.cast::<PyList>() {
        let mut arr = Vec::new();
        for item in l.iter() {
            arr.push(py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Array(arr));
    }
    if let Ok(t) = obj.cast::<PyTuple>() {
        let mut arr = Vec::new();
        for item in t.iter() {
            arr.push(py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Array(arr));
    }
    // 兜底：字符串表示
    Ok(serde_json::Value::String(obj.str()?.to_string()))
}

/// 将 Python 文件参数（str/bytes）转换为 core `FileInput`
///
/// - str 以 http(s):// 开头 → `FileInput::Url`
/// - 其他 str → `FileInput::Path`
/// - bytes → `FileInput::Bytes`
fn py_to_file_input(obj: &Bound<'_, PyAny>) -> PyResult<CoreFileInput> {
    if let Ok(s) = obj.extract::<String>() {
        if s.starts_with("http://") || s.starts_with("https://") {
            Ok(CoreFileInput::url(s))
        } else {
            Ok(CoreFileInput::path(s))
        }
    } else if let Ok(b) = obj.extract::<Vec<u8>>() {
        Ok(CoreFileInput::bytes(b))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "文件参数必须是 str（路径/URL）或 bytes",
        ))
    }
}

/// 将 Python embed 输入（str 或 list[str]）转换为 core `EmbedInput`
fn py_to_embed_input(obj: &Bound<'_, PyAny>) -> PyResult<CoreEmbedInput> {
    if let Ok(s) = obj.extract::<String>() {
        Ok(CoreEmbedInput::Single(s))
    } else if let Ok(v) = obj.extract::<Vec<String>>() {
        Ok(CoreEmbedInput::Multiple(v))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "input 必须是 str 或 list[str]",
        ))
    }
}

/// 将视频生成模式字符串转换为 core `VideoMode`
fn parse_video_mode(s: &str) -> CoreVideoMode {
    match s.to_lowercase().as_str() {
        "image2video" => CoreVideoMode::Image2Video,
        "keyframes" => CoreVideoMode::Keyframes,
        "multiimage" => CoreVideoMode::Multiimage,
        _ => CoreVideoMode::Text2Video,
    }
}

/// 将 core `TaskStatus` 转为字符串（Python 侧用字符串表示任务状态）
fn task_status_to_str(s: CoreTaskStatus) -> &'static str {
    match s {
        CoreTaskStatus::Pending => "pending",
        CoreTaskStatus::Processing => "processing",
        CoreTaskStatus::Success => "success",
        CoreTaskStatus::Failed => "failed",
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

/// 图像数据（`ImageResult.data` 元素）
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct ImageData {
    url: Option<String>,
    b64_json: Option<String>,
    revised_prompt: Option<String>,
}

#[pymethods]
impl ImageData {
    #[getter]
    fn url(&self) -> Option<String> {
        self.url.clone()
    }

    #[getter]
    fn b64_json(&self) -> Option<String> {
        self.b64_json.clone()
    }

    #[getter]
    fn revised_prompt(&self) -> Option<String> {
        self.revised_prompt.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageData(url={:?}, has_b64={})",
            self.url,
            self.b64_json.is_some()
        )
    }
}

impl ImageData {
    fn from_core(d: CoreImageData) -> Self {
        Self {
            url: d.url,
            b64_json: d.b64_json,
            revised_prompt: d.revised_prompt,
        }
    }
}

/// 图像生成结果
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct ImageResult {
    id: String,
    created: u64,
    model: String,
    data: Vec<ImageData>,
}

#[pymethods]
impl ImageResult {
    #[getter]
    fn id(&self) -> String {
        self.id.clone()
    }

    #[getter]
    fn created(&self) -> u64 {
        self.created
    }

    #[getter]
    fn model(&self) -> String {
        self.model.clone()
    }

    #[getter]
    fn data(&self) -> Vec<ImageData> {
        self.data.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageResult(id={:?}, model={:?}, data_len={})",
            self.id,
            self.model,
            self.data.len()
        )
    }
}

impl ImageResult {
    fn from_core(r: CoreImageResult) -> Self {
        Self {
            id: r.id,
            created: r.created,
            model: r.model,
            data: r.data.into_iter().map(ImageData::from_core).collect(),
        }
    }
}

/// 视频任务（创建后返回）
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct VideoTask {
    task_id: String,
    model: String,
    status: String,
    created_at: u64,
}

#[pymethods]
impl VideoTask {
    #[getter]
    fn task_id(&self) -> String {
        self.task_id.clone()
    }

    #[getter]
    fn model(&self) -> String {
        self.model.clone()
    }

    #[getter]
    fn status(&self) -> String {
        self.status.clone()
    }

    #[getter]
    fn created_at(&self) -> u64 {
        self.created_at
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoTask(task_id={:?}, status={:?})",
            self.task_id, self.status
        )
    }
}

impl VideoTask {
    fn from_core(t: CoreVideoTask) -> Self {
        Self {
            task_id: t.task_id,
            model: t.model,
            status: task_status_to_str(t.status).to_string(),
            created_at: t.created_at,
        }
    }
}

/// 视频任务状态（轮询返回）
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct VideoStatus {
    task_id: String,
    status: String,
    video_url: Option<String>,
    progress: Option<u32>,
    error: Option<String>,
    created_at: Option<u64>,
    updated_at: Option<u64>,
}

#[pymethods]
impl VideoStatus {
    #[getter]
    fn task_id(&self) -> String {
        self.task_id.clone()
    }

    #[getter]
    fn status(&self) -> String {
        self.status.clone()
    }

    #[getter]
    fn video_url(&self) -> Option<String> {
        self.video_url.clone()
    }

    #[getter]
    fn progress(&self) -> Option<u32> {
        self.progress
    }

    #[getter]
    fn error(&self) -> Option<String> {
        self.error.clone()
    }

    #[getter]
    fn created_at(&self) -> Option<u64> {
        self.created_at
    }

    #[getter]
    fn updated_at(&self) -> Option<u64> {
        self.updated_at
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoStatus(task_id={:?}, status={:?}, progress={:?})",
            self.task_id, self.status, self.progress
        )
    }
}

impl VideoStatus {
    fn from_core(s: CoreVideoStatus) -> Self {
        Self {
            task_id: s.task_id,
            status: task_status_to_str(s.status).to_string(),
            video_url: s.video_url,
            progress: s.progress,
            error: s.error,
            created_at: s.created_at,
            updated_at: s.updated_at,
        }
    }
}

/// 转写词级时间戳
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct TranscriptionWord {
    word: String,
    start: f64,
    end: f64,
    confidence: Option<f64>,
}

#[pymethods]
impl TranscriptionWord {
    #[getter]
    fn word(&self) -> String {
        self.word.clone()
    }

    #[getter]
    fn start(&self) -> f64 {
        self.start
    }

    #[getter]
    fn end(&self) -> f64 {
        self.end
    }

    #[getter]
    fn confidence(&self) -> Option<f64> {
        self.confidence
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscriptionWord(word={:?}, start={}, end={})",
            self.word, self.start, self.end
        )
    }
}

impl TranscriptionWord {
    fn from_core(w: CoreTranscriptionWord) -> Self {
        Self {
            word: w.word,
            start: w.start,
            end: w.end,
            confidence: w.confidence,
        }
    }
}

/// 转写分段（带时间戳）
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct TranscriptionSegment {
    id: u32,
    start: f64,
    end: f64,
    text: String,
    confidence: Option<f64>,
    speaker: Option<String>,
}

#[pymethods]
impl TranscriptionSegment {
    #[getter]
    fn id(&self) -> u32 {
        self.id
    }

    #[getter]
    fn start(&self) -> f64 {
        self.start
    }

    #[getter]
    fn end(&self) -> f64 {
        self.end
    }

    #[getter]
    fn text(&self) -> String {
        self.text.clone()
    }

    #[getter]
    fn confidence(&self) -> Option<f64> {
        self.confidence
    }

    #[getter]
    fn speaker(&self) -> Option<String> {
        self.speaker.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscriptionSegment(id={}, text={:?})",
            self.id, self.text
        )
    }
}

impl TranscriptionSegment {
    fn from_core(s: CoreTranscriptionSegment) -> Self {
        Self {
            id: s.id,
            start: s.start,
            end: s.end,
            text: s.text,
            confidence: s.confidence,
            speaker: s.speaker,
        }
    }
}

/// 语音转文字结果
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct TranscriptionResult {
    text: String,
    language: Option<String>,
    duration: Option<f64>,
    task: String,
    model: Option<String>,
    segments: Option<Vec<TranscriptionSegment>>,
    words: Option<Vec<TranscriptionWord>>,
}

#[pymethods]
impl TranscriptionResult {
    #[getter]
    fn text(&self) -> String {
        self.text.clone()
    }

    #[getter]
    fn language(&self) -> Option<String> {
        self.language.clone()
    }

    #[getter]
    fn duration(&self) -> Option<f64> {
        self.duration
    }

    #[getter]
    fn task(&self) -> String {
        self.task.clone()
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.model.clone()
    }

    #[getter]
    fn segments(&self) -> Option<Vec<TranscriptionSegment>> {
        self.segments.clone()
    }

    #[getter]
    fn words(&self) -> Option<Vec<TranscriptionWord>> {
        self.words.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscriptionResult(text={:?}, language={:?})",
            self.text, self.language
        )
    }
}

impl TranscriptionResult {
    fn from_core(r: CoreTranscriptionResult) -> Self {
        Self {
            text: r.text,
            language: r.language,
            duration: r.duration,
            task: r.task,
            model: r.model,
            segments: r
                .segments
                .map(|s| s.into_iter().map(TranscriptionSegment::from_core).collect()),
            words: r
                .words
                .map(|w| w.into_iter().map(TranscriptionWord::from_core).collect()),
        }
    }
}

/// 单个嵌入项
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct EmbeddingItem {
    index: u32,
    embedding: Vec<f64>,
}

#[pymethods]
impl EmbeddingItem {
    #[getter]
    fn index(&self) -> u32 {
        self.index
    }

    #[getter]
    fn embedding(&self) -> Vec<f64> {
        self.embedding.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingItem(index={}, dim={})",
            self.index,
            self.embedding.len()
        )
    }
}

impl EmbeddingItem {
    fn from_core(i: CoreEmbeddingItem) -> Self {
        let embedding = match i.embedding {
            CoreEmbeddingVector::Float(v) => v,
            // base64 编码向量未解码，返空（echo 及多数 provider 走 Float）
            CoreEmbeddingVector::Base64(_) => Vec::new(),
        };
        Self {
            index: i.index,
            embedding,
        }
    }
}

/// 文本嵌入结果
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct EmbeddingResult {
    model: String,
    data: Vec<EmbeddingItem>,
    prompt_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

#[pymethods]
impl EmbeddingResult {
    #[getter]
    fn model(&self) -> String {
        self.model.clone()
    }

    #[getter]
    fn data(&self) -> Vec<EmbeddingItem> {
        self.data.clone()
    }

    #[getter]
    fn prompt_tokens(&self) -> Option<u64> {
        self.prompt_tokens
    }

    #[getter]
    fn total_tokens(&self) -> Option<u64> {
        self.total_tokens
    }

    /// 提取所有嵌入向量（便捷访问，等价于 `[item.embedding for item in data]`）
    fn get_embeddings(&self) -> Vec<Vec<f64>> {
        self.data.iter().map(|i| i.embedding.clone()).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingResult(model={:?}, count={})",
            self.model,
            self.data.len()
        )
    }
}

impl EmbeddingResult {
    fn from_core(r: CoreEmbeddingResult) -> Self {
        let (prompt_tokens, total_tokens) = r
            .usage
            .map(|u| (u.prompt_tokens, u.total_tokens))
            .unzip();
        Self {
            model: r.model,
            data: r.data.into_iter().map(EmbeddingItem::from_core).collect(),
            prompt_tokens,
            total_tokens,
        }
    }
}

/// 模型信息
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct ModelInfo {
    id: String,
    name: String,
    model_type: String,
    provider: String,
    capabilities: Vec<String>,
    max_tokens: Option<u32>,
    supports_streaming: bool,
    description: Option<String>,
    created: Option<u64>,
}

#[pymethods]
impl ModelInfo {
    #[getter]
    fn id(&self) -> String {
        self.id.clone()
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// 模型类型（"chat"/"image"/"video"/"audio"）
    #[getter(r#type)]
    fn model_type(&self) -> String {
        self.model_type.clone()
    }

    #[getter]
    fn provider(&self) -> String {
        self.provider.clone()
    }

    #[getter]
    fn capabilities(&self) -> Vec<String> {
        self.capabilities.clone()
    }

    #[getter]
    fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    #[getter]
    fn supports_streaming(&self) -> bool {
        self.supports_streaming
    }

    #[getter]
    fn description(&self) -> Option<String> {
        self.description.clone()
    }

    #[getter]
    fn created(&self) -> Option<u64> {
        self.created
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelInfo(id={:?}, type={:?}, provider={:?})",
            self.id, self.model_type, self.provider
        )
    }
}

impl ModelInfo {
    fn from_core(m: CoreModelInfo) -> Self {
        Self {
            id: m.id,
            name: m.name,
            model_type: m.model_type.as_str().to_string(),
            provider: m.provider,
            capabilities: m.capabilities,
            max_tokens: m.max_tokens,
            supports_streaming: m.supports_streaming,
            description: m.description,
            created: m.created,
        }
    }
}

/// 音色信息
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
struct VoiceInfo {
    short_name: Option<String>,
    name: Option<String>,
    locale: Option<String>,
    gender: Option<String>,
    voice_id: Option<String>,
}

#[pymethods]
impl VoiceInfo {
    #[getter]
    fn short_name(&self) -> Option<String> {
        self.short_name.clone()
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.name.clone()
    }

    #[getter]
    fn locale(&self) -> Option<String> {
        self.locale.clone()
    }

    #[getter]
    fn gender(&self) -> Option<String> {
        self.gender.clone()
    }

    #[getter]
    fn voice_id(&self) -> Option<String> {
        self.voice_id.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "VoiceInfo(short_name={:?}, locale={:?}, gender={:?})",
            self.short_name, self.locale, self.gender
        )
    }
}

impl VoiceInfo {
    fn from_core(v: CoreVoiceInfo) -> Self {
        Self {
            short_name: v.short_name,
            name: v.name,
            locale: v.locale,
            gender: v.gender,
            voice_id: v.voice_id,
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

    /// 图像生成
    ///
    /// 参数：
    /// - `model`: 图像模型名称
    /// - `prompt`: 提示词
    /// - `size`: 图像尺寸（默认 "1024x1024"）
    /// - `n`: 生成数量（默认 1）
    /// - `negative_prompt`: 负面提示词
    /// - `reference_images`: 参考图列表（图生图），元素为 str（路径/URL）或 bytes
    /// - `mask`: 遮罩图（局部重绘），str 或 bytes
    /// - `response_format`: 响应格式（"url"/"b64_json"，默认 "url"）
    /// - `**kwargs`: 厂商特有参数透传
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (model, prompt, size="1024x1024", n=1, negative_prompt=None, reference_images=None, mask=None, response_format="url", **kwargs))]
    async fn image_generate(
        &self,
        model: String,
        prompt: String,
        size: &str,
        n: u32,
        negative_prompt: Option<String>,
        reference_images: Option<Vec<Py<PyAny>>>,
        mask: Option<Py<PyAny>>,
        response_format: &str,
        kwargs: Option<Py<PyDict>>,
    ) -> PyResult<ImageResult> {
        // 在持 GIL 时转换 Python 对象为 core 类型（**kwargs / reference_images / mask）
        let (extra, ref_imgs, mask_input) = Python::attach(|py| -> PyResult<(
            std::collections::HashMap<String, serde_json::Value>,
            Vec<CoreFileInput>,
            Option<CoreFileInput>,
        )> {
            let extra = match &kwargs {
                Some(d) => kwargs_to_extra(d.bind(py))?,
                None => std::collections::HashMap::new(),
            };
            let ref_imgs = match &reference_images {
                Some(imgs) => imgs
                    .iter()
                    .map(|i| py_to_file_input(i.bind(py)))
                    .collect::<PyResult<Vec<_>>>()?,
                None => Vec::new(),
            };
            let mask_input = match &mask {
                Some(m) => Some(py_to_file_input(m.bind(py))?),
                None => None,
            };
            Ok((extra, ref_imgs, mask_input))
        })?;

        let mut builder = CoreImageRequest::builder(model, prompt)
            .size(size)
            .n(n)
            .response_format(response_format);
        if let Some(np) = negative_prompt {
            builder = builder.negative_prompt(np);
        }
        if !ref_imgs.is_empty() {
            builder = builder.reference_images(ref_imgs);
        }
        if let Some(m) = mask_input {
            builder = builder.mask(m);
        }
        for (k, v) in extra {
            builder = builder.extra(k, v);
        }
        let req = builder.build();

        let inner = self.inner.clone();
        let result = RUNTIME
            .spawn(async move {
                let client = inner.lock().await;
                client.image_generate(req).await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("image_generate 任务失败: {e}"))
            })?;

        let r = result.map_err(map_error)?;
        Ok(ImageResult::from_core(r))
    }

    /// 创建视频生成任务
    ///
    /// 参数：
    /// - `model`: 视频模型名称
    /// - `prompt`: 提示词
    /// - `width`/`height`: 视频分辨率（默认 1280x720）
    /// - `num_frames`: 帧数（部分模型需要）
    /// - `frame_rate`: 帧率（默认 24）
    /// - `mode`: 生成模式（"text2video"/"image2video"/"keyframes"/"multiimage"，默认 "text2video"）
    /// - `reference_images`: 参考图列表（图生视频）
    /// - `negative_prompt`: 负面提示词
    /// - `seed`: 随机种子
    /// - `**kwargs`: 厂商特有参数透传
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model, prompt, width=1280, height=720, num_frames=None, frame_rate=24, mode="text2video", reference_images=None, negative_prompt=None, seed=None, **kwargs))]
    async fn video_create(
        &self,
        model: String,
        prompt: String,
        width: u32,
        height: u32,
        num_frames: Option<u32>,
        frame_rate: u32,
        mode: &str,
        reference_images: Option<Vec<Py<PyAny>>>,
        negative_prompt: Option<String>,
        seed: Option<u64>,
        kwargs: Option<Py<PyDict>>,
    ) -> PyResult<VideoTask> {
        let (extra, ref_imgs) = Python::attach(|py| -> PyResult<(
            std::collections::HashMap<String, serde_json::Value>,
            Vec<CoreFileInput>,
        )> {
            let extra = match &kwargs {
                Some(d) => kwargs_to_extra(d.bind(py))?,
                None => std::collections::HashMap::new(),
            };
            let ref_imgs = match &reference_images {
                Some(imgs) => imgs
                    .iter()
                    .map(|i| py_to_file_input(i.bind(py)))
                    .collect::<PyResult<Vec<_>>>()?,
                None => Vec::new(),
            };
            Ok((extra, ref_imgs))
        })?;

        let mut builder = CoreVideoRequest::builder(model, prompt)
            .width(width)
            .height(height)
            .frame_rate(frame_rate)
            .mode(parse_video_mode(mode));
        if let Some(nf) = num_frames {
            builder = builder.num_frames(nf);
        }
        if !ref_imgs.is_empty() {
            builder = builder.reference_images(ref_imgs);
        }
        if let Some(np) = negative_prompt {
            builder = builder.negative_prompt(np);
        }
        if let Some(s) = seed {
            builder = builder.seed(s);
        }
        for (k, v) in extra {
            builder = builder.extra(k, v);
        }
        let req = builder.build();

        let inner = self.inner.clone();
        let result = RUNTIME
            .spawn(async move {
                let client = inner.lock().await;
                client.video_create(req).await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("video_create 任务失败: {e}"))
            })?;

        let t = result.map_err(map_error)?;
        Ok(VideoTask::from_core(t))
    }

    /// 查询视频任务状态
    ///
    /// 参数：
    /// - `task_id`: 任务 ID（由 `video_create` 返回）
    /// - `model`: 模型名称（部分 Provider 需要，默认空串）
    #[pyo3(signature = (task_id, model=""))]
    async fn video_poll(&self, task_id: String, model: &str) -> PyResult<VideoStatus> {
        // model 为 &str（借用），spawn 需 'static，故先转 owned
        let model = model.to_string();
        let inner = self.inner.clone();
        let result = RUNTIME
            .spawn(async move {
                let client = inner.lock().await;
                client.video_poll(&task_id, &model).await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("video_poll 任务失败: {e}"))
            })?;

        let s = result.map_err(map_error)?;
        Ok(VideoStatus::from_core(s))
    }

    /// 语音转文字
    ///
    /// 参数：
    /// - `model`: ASR 模型名称
    /// - `file`: 音频文件（str 路径/URL 或 bytes）
    /// - `language`: 语言代码（如 "zh"、"en"）
    /// - `prompt`: 提示词（改善专有名词识别）
    /// - `response_format`: 响应格式（默认 "json"）
    /// - `temperature`: 温度系数（0-1）
    /// - `**kwargs`: 厂商特有参数透传
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model, file, language=None, prompt=None, response_format="json", temperature=None, **kwargs))]
    async fn transcribe(
        &self,
        model: String,
        file: Py<PyAny>,
        language: Option<String>,
        prompt: Option<String>,
        response_format: &str,
        temperature: Option<f64>,
        kwargs: Option<Py<PyDict>>,
    ) -> PyResult<TranscriptionResult> {
        let (file_input, extra) = Python::attach(|py| -> PyResult<(
            CoreFileInput,
            std::collections::HashMap<String, serde_json::Value>,
        )> {
            let file_input = py_to_file_input(file.bind(py))?;
            let extra = match &kwargs {
                Some(d) => kwargs_to_extra(d.bind(py))?,
                None => std::collections::HashMap::new(),
            };
            Ok((file_input, extra))
        })?;

        let mut builder = CoreTranscribeRequest::builder(model, file_input)
            .response_format(response_format);
        if let Some(l) = language {
            builder = builder.language(l);
        }
        if let Some(p) = prompt {
            builder = builder.prompt(p);
        }
        if let Some(t) = temperature {
            builder = builder.temperature(t);
        }
        for (k, v) in extra {
            builder = builder.extra(k, v);
        }
        let req = builder.build();

        let inner = self.inner.clone();
        let result = RUNTIME
            .spawn(async move {
                let client = inner.lock().await;
                client.transcribe(req).await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("transcribe 任务失败: {e}"))
            })?;

        let r = result.map_err(map_error)?;
        Ok(TranscriptionResult::from_core(r))
    }

    /// 文本嵌入
    ///
    /// 参数：
    /// - `model`: 嵌入模型名称
    /// - `input`: 文本（str 或 list[str]）
    /// - `**kwargs`: 厂商特有参数透传
    #[pyo3(signature = (model, input, **kwargs))]
    async fn embed(
        &self,
        model: String,
        input: Py<PyAny>,
        kwargs: Option<Py<PyDict>>,
    ) -> PyResult<EmbeddingResult> {
        let (embed_input, extra) = Python::attach(|py| -> PyResult<(
            CoreEmbedInput,
            std::collections::HashMap<String, serde_json::Value>,
        )> {
            let embed_input = py_to_embed_input(input.bind(py))?;
            let extra = match &kwargs {
                Some(d) => kwargs_to_extra(d.bind(py))?,
                None => std::collections::HashMap::new(),
            };
            Ok((embed_input, extra))
        })?;

        let req = CoreEmbedRequest {
            model,
            input: embed_input,
            dimensions: None,
            encoding_format: None,
            user: None,
            extra,
        };

        let inner = self.inner.clone();
        let result = RUNTIME
            .spawn(async move {
                let client = inner.lock().await;
                client.embed(req).await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("embed 任务失败: {e}"))
            })?;

        let r = result.map_err(map_error)?;
        Ok(EmbeddingResult::from_core(r))
    }

    /// 获取可用模型列表
    ///
    /// 参数：
    /// - `model_type`: 可选模型类型过滤（"chat"/"image"/"video"/"audio"）
    #[pyo3(signature = (model_type=None))]
    async fn list_models(&self, model_type: Option<String>) -> PyResult<Vec<ModelInfo>> {
        let filter = model_type.map(|s| CoreModelType::from(s.as_str()));
        let inner = self.inner.clone();
        let result = RUNTIME
            .spawn(async move {
                let client = inner.lock().await;
                client.list_models(filter).await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("list_models 任务失败: {e}"))
            })?;

        let models = result.map_err(map_error)?;
        Ok(models.into_iter().map(ModelInfo::from_core).collect())
    }

    /// 获取 Provider 可用音色列表
    ///
    /// 参数：
    /// - `language`: 可选语言过滤（如 "zh-CN"）
    #[pyo3(signature = (language=None))]
    async fn list_voices(&self, language: Option<String>) -> PyResult<Vec<VoiceInfo>> {
        let inner = self.inner.clone();
        let result = RUNTIME
            .spawn(async move {
                let client = inner.lock().await;
                client.list_voices(language.as_deref()).await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("list_voices 任务失败: {e}"))
            })?;

        let voices = result.map_err(map_error)?;
        Ok(voices.into_iter().map(VoiceInfo::from_core).collect())
    }

    /// 推荐可用音色（按语言/性别过滤）
    ///
    /// 参数：
    /// - `language`: 可选语言过滤（如 "zh-CN"）
    /// - `gender`: 可选性别过滤（"Female"/"Male"）
    /// - `limit`: 返回数量上限（默认 10）
    #[pyo3(signature = (language=None, gender=None, limit=10))]
    async fn recommend_voices(
        &self,
        language: Option<String>,
        gender: Option<String>,
        limit: usize,
    ) -> PyResult<Vec<VoiceInfo>> {
        let inner = self.inner.clone();
        let result = RUNTIME
            .spawn(async move {
                let client = inner.lock().await;
                client
                    .recommend_voices(language.as_deref(), gender.as_deref(), limit)
                    .await
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("recommend_voices 任务失败: {e}"))
            })?;

        let voices = result.map_err(map_error)?;
        Ok(voices.into_iter().map(VoiceInfo::from_core).collect())
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
    m.add_class::<ImageData>()?;
    m.add_class::<ImageResult>()?;
    m.add_class::<VideoTask>()?;
    m.add_class::<VideoStatus>()?;
    m.add_class::<TranscriptionWord>()?;
    m.add_class::<TranscriptionSegment>()?;
    m.add_class::<TranscriptionResult>()?;
    m.add_class::<EmbeddingItem>()?;
    m.add_class::<EmbeddingResult>()?;
    m.add_class::<ModelInfo>()?;
    m.add_class::<VoiceInfo>()?;

    // 客户端与流式
    m.add_class::<Client>()?;
    m.add_class::<ChatStreamIterator>()?;

    // 模块版本
    m.add("__version__", aibridge_core::VERSION)?;

    Ok(())
}
