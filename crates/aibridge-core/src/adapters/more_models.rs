//! 更多主流模型适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/more_models.py`。
//!
//! 支持五个主流模型 Provider（文档见各适配器注释）：
//! - **DeepSeek**：OpenAI 兼容协议，支持思考模式（reasoning_effort + 自动注入 thinking）
//! - **阶跃星辰 StepFun**：OpenAI 兼容协议（base_url 含 `/v1` 前缀）
//! - **Mistral AI**：OpenAI 兼容协议（base_url 含 `/v1` 前缀）
//! - **Cohere**：非标准协议（`POST /chat`，message/chat_history 结构，响应 `text` 字段）
//! - **Perplexity AI**：OpenAI 兼容协议，AI 搜索
//!
//! ## 结构
//!
//! Python v1 是 5 个独立 adapter 类，统一注册到工厂（StepFun 额外注册 `step` 别名）。
//! Rust 同样实现 5 个独立 struct：
//! - `DeepSeekAdapter` / `StepFunAdapter` / `MistralAdapter` / `PerplexityAdapter`：
//!   组合 `OpenAiCompatAdapter` 地基，chat/chat_stream/image/embed/list_models 全部委托
//!   （DeepSeek 额外在 chat 前注入 `thinking`，对齐 Python 自动思考模式行为）
//! - `CohereAdapter`：组合 `OpenAiCompatAdapter` 仅复用 embed 能力校验等公共逻辑，
//!   chat/chat_stream/list_models 独立实现（Cohere 特有 `/chat` 端点 + message/chat_history
//!   请求体 + `text` 响应 + `event_type` 流式事件 + `/models` 响应 `models[].name` 字段）
//!
//! ## provider_type 标识（对齐 Python v1）
//!
//! | Provider | provider_type | 别名 |
//! |---|---|---|
//! | DeepSeek | `deepseek` | — |
//! | 阶跃星辰 StepFun | `stepfun` | `step` |
//! | Mistral AI | `mistral` | — |
//! | Cohere | `cohere` | — |
//! | Perplexity AI | `perplexity` | — |
//!
//! 注册到工厂由 `adapter::factory` 完成（阶段 2a 收尾批次），本模块仅声明适配器类型。

use async_trait::async_trait;
use futures::stream::{StreamExt, TryStreamExt};
use serde_json::{json, Value};

use crate::adapter::{Adapter, Capabilities, CapabilitySet, ChatStream};
use crate::adapters::openai_compat::OpenAiCompatAdapter;
use crate::config::ProviderConfig;
use crate::error::{AibridgeError, Result};
use crate::model::chat::{
    ChatChoice, ChatCompletion, ChatCompletionChunk, ChatCompletionDelta, ChatMessage, ChatRequest,
    ChoiceMessage, DeltaMessage, UserContent,
};
use crate::model::common::{infer_model_type, ModelInfo, ModelType};
use crate::util;

// ==================== 默认 Base URL ====================

/// DeepSeek 默认 Base URL
///
/// 对应 Python v1 `DeepSeekAdapter.DEFAULT_BASE_URL`。
/// DeepSeek base_url 不含 `/v1` 前缀，chat 端点为 `POST /chat/completions`，
/// models 端点为 `GET /models`（与 OpenAI 兼容地基的相对路径一致）。
pub const DEFAULT_DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";

/// 阶跃星辰 StepFun 默认 Base URL
///
/// 对应 Python v1 `StepFunAdapter.DEFAULT_BASE_URL = "https://api.stepfun.com"`，
/// 但 Python `start()` 中 httpx base_url 设为 `base_url + "/v1"`，故实际请求 base
/// 为 `https://api.stepfun.com/v1`。此处直接采用含 `/v1` 的值，行为等价。
pub const DEFAULT_STEPFUN_BASE_URL: &str = "https://api.stepfun.com/v1";

/// Mistral AI 默认 Base URL
///
/// 对应 Python v1 `MistralAdapter.DEFAULT_BASE_URL`（已含 `/v1` 前缀）。
pub const DEFAULT_MISTRAL_BASE_URL: &str = "https://api.mistral.ai/v1";

/// Cohere 默认 Base URL
///
/// 对应 Python v1 `CohereAdapter.DEFAULT_BASE_URL`（已含 `/v1` 前缀）。
/// Cohere 的 chat 端点为 `POST /chat`（非标准 OpenAI），models 端点为 `GET /models`。
pub const DEFAULT_COHERE_BASE_URL: &str = "https://api.cohere.ai/v1";

/// Perplexity AI 默认 Base URL
///
/// 对应 Python v1 `PerplexityAdapter.DEFAULT_BASE_URL`。
/// Perplexity base_url 不含 `/v1` 前缀，chat 端点为 `POST /chat/completions`，
/// models 端点为 `GET /models`（与 OpenAI 兼容地基的相对路径一致）。
pub const DEFAULT_PERPLEXITY_BASE_URL: &str = "https://api.perplexity.ai";

// ==================== 能力集合构造 ====================

/// DeepSeek 支持的能力集合
///
/// 对齐 Python v1 `DeepSeekAdapter.supported_capabilities = ["chat", "vision"]`。
/// Rust 额外声明 ChatStream（OpenAI 兼容地基支持流式，Python 老版也实现了 chat_stream）。
fn deepseek_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// StepFun 支持的能力集合
///
/// 对齐 Python v1 `StepFunAdapter.supported_capabilities = ["chat", "vision"]`。
fn stepfun_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// Mistral 支持的能力集合
///
/// 对齐 Python v1 `MistralAdapter.supported_capabilities = ["chat", "vision"]`。
fn mistral_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// Cohere 支持的能力集合
///
/// 对齐 Python v1 `CohereAdapter.supported_capabilities = ["chat", "vision"]`。
fn cohere_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// Perplexity 支持的能力集合
///
/// 对齐 Python v1 `PerplexityAdapter.supported_capabilities = ["chat", "vision"]`。
/// Perplexity 的 Sonar 模型具备联网搜索能力，额外声明 WebSearch。
fn perplexity_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps.insert(Capabilities::WebSearch);
    caps
}

// ==================== DeepSeek 适配器 ====================

/// DeepSeek 适配器
///
/// OpenAI 兼容协议，chat/chat_stream/image(不支持)/embed(不支持)/list_models 委托给
/// `OpenAiCompatAdapter` 地基。DeepSeek 特有行为：当请求体含 `reasoning_effort` 时
/// 自动注入 `thinking: {"type": "enabled"}`（对齐 Python v1 `_build_request_body`）。
///
/// - Base URL: `https://api.deepseek.com`
/// - Chat: `POST /chat/completions`
/// - Models: `GET /models`
/// - 认证: Bearer Token
/// - 文档: <https://api-docs.deepseek.com/>
pub struct DeepSeekAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl DeepSeekAdapter {
    /// 创建 DeepSeek 适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_DEEPSEEK_BASE_URL,
            deepseek_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 compat 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "deepseek";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "DeepSeek";

    /// 构造 DeepSeek chat 请求体（含自动 thinking 注入）
    ///
    /// 对齐 Python v1 `DeepSeekAdapter._build_request_body`：
    /// 1. 先用兼容地基构造标准 OpenAI 请求体（model/messages/temperature/max_tokens/top_p 等）
    /// 2. 若 body 含 `reasoning_effort` 字段（来自 ChatRequest.reasoning_effort 或 extra 透传），
    ///    且 body 尚无 `thinking` 字段，则自动注入 `thinking: {"type": "enabled"}`
    fn build_chat_body(&self, req: &ChatRequest, stream: bool) -> Value {
        let mut body = self.compat.build_chat_body(req, stream);
        // 自动思考模式：有 reasoning_effort 且无 thinking 时注入
        if body.get("reasoning_effort").is_some() && body.get("thinking").is_none() {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("thinking".to_string(), json!({ "type": "enabled" }));
            }
        }
        body
    }
}

#[async_trait]
impl Adapter for DeepSeekAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
    }

    fn capabilities(&self) -> CapabilitySet {
        self.compat.capabilities_set().clone()
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    async fn start(&mut self) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    /// 文本对话：构造 DeepSeek 请求体（含自动 thinking），委托地基发送 + 解析
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.compat.ensure_capability_pub(Capabilities::Chat)?;
        let body = self.build_chat_body(&req, false);
        let value = self.compat.post_chat(&body).await?;
        self.compat.parse_chat_completion(&value, &req.model)
    }

    /// 流式文本对话：委托地基 `POST /chat/completions` (stream=true)
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.compat.chat_stream(req).await
    }

    /// 模型列表（实时拉取）：委托地基 `GET /models`
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }

    // image_generate / video_create / video_poll / embed / transcribe / speech / list_voices
    // 走 trait 默认实现，返 UnsupportedCapability，与 Python 老版行为一致。
}

// ==================== StepFun 适配器 ====================

/// 阶跃星辰 StepFun 适配器
///
/// OpenAI 兼容协议，全部能力委托给 `OpenAiCompatAdapter` 地基。
///
/// - Base URL: `https://api.stepfun.com/v1`（Python default 不含 `/v1`，但 `start()` 拼接，
///   此处直接采用含 `/v1` 的值，行为等价）
/// - Chat: `POST /chat/completions`
/// - Models: `GET /models`
/// - 认证: Bearer Token
/// - 文档: <https://platform.stepfun.com/docs/>
pub struct StepFunAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl StepFunAdapter {
    /// 创建 StepFun 适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_STEPFUN_BASE_URL,
            stepfun_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 compat 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "stepfun";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "阶跃星辰 StepFun";
}

#[async_trait]
impl Adapter for StepFunAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
    }

    fn capabilities(&self) -> CapabilitySet {
        self.compat.capabilities_set().clone()
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    async fn start(&mut self) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.compat.chat(req).await
    }

    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.compat.chat_stream(req).await
    }

    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }
}

// ==================== Mistral 适配器 ====================

/// Mistral AI 适配器
///
/// OpenAI 兼容协议，全部能力委托给 `OpenAiCompatAdapter` 地基。
///
/// - Base URL: `https://api.mistral.ai/v1`
/// - Chat: `POST /chat/completions`
/// - Models: `GET /models`
/// - 认证: Bearer Token
/// - 文档: <https://docs.mistral.ai/>
pub struct MistralAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl MistralAdapter {
    /// 创建 Mistral 适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_MISTRAL_BASE_URL,
            mistral_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 compat 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "mistral";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "Mistral AI";
}

#[async_trait]
impl Adapter for MistralAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
    }

    fn capabilities(&self) -> CapabilitySet {
        self.compat.capabilities_set().clone()
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    async fn start(&mut self) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.compat.chat(req).await
    }

    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.compat.chat_stream(req).await
    }

    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }
}

// ==================== Cohere 适配器 ====================

/// Cohere 适配器
///
/// 非标准 OpenAI 协议，chat/chat_stream/list_models 独立实现，不委托兼容地基。
///
/// - Base URL: `https://api.cohere.ai/v1`
/// - Chat: `POST /chat`（请求体含 `message` + `chat_history` + `system_prompt`，非标准 OpenAI）
/// - 响应: `{"text": "...", "usage": {"tokens": {"input_tokens":..., "output_tokens":...}}}`
/// - 流式: SSE，每行 `data: {"event_type": "text-generation"|"stream-end", "text": "..."}`
/// - Models: `GET /models`（响应 `{"models": [{"name": "...", ...}]}`，`name` 即模型 ID）
/// - 认证: Bearer Token
/// - 文档: <https://docs.cohere.com/>
///
/// embed 走 trait 默认实现返 UnsupportedCapability（Python v1 未实现 embed）。
pub struct CohereAdapter {
    /// OpenAI 兼容地基（仅复用其 HttpClient 与错误映射，不委托 chat 等方法）
    compat: OpenAiCompatAdapter,
}

impl CohereAdapter {
    /// 创建 Cohere 适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_COHERE_BASE_URL,
            cohere_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 compat 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "cohere";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "Cohere";

    /// API key
    fn api_key(&self) -> Option<&str> {
        self.compat.api_key()
    }

    /// base_url
    fn base_url(&self) -> &str {
        self.compat.base_url()
    }

    /// 拼接完整 URL（base_url + 相对路径）
    fn url(&self, path: &str) -> String {
        let base = self.base_url().trim_end_matches('/');
        let path = path.trim_start_matches('/');
        format!("{base}/{path}")
    }

    /// 校验能力是否被支持
    fn ensure_capability(&self, cap: Capabilities) -> Result<()> {
        if self.compat.capabilities_set().contains(&cap) {
            Ok(())
        } else {
            Err(AibridgeError::UnsupportedCapability {
                capability: format!("{} (provider: {})", cap.as_str(), Self::PROVIDER_TYPE),
            })
        }
    }

    /// 发送带认证的 POST JSON 请求，并用 OpenAI 错误映射处理响应
    ///
    /// 复用兼容地基的 `map_api_error`，错误分类与 OpenAI 兼容族一致。
    async fn post_authed_json(&self, path: &str, body: &Value) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .compat
            .http_inner()
            .post(&url)
            .bearer_auth(self.api_key().unwrap_or(""))
            .header("accept", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    AibridgeError::Timeout
                } else {
                    AibridgeError::Network(e)
                }
            })?;
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(OpenAiCompatAdapter::map_api_error(status_code, &body_text));
        }
        resp.json::<Value>().await.map_err(AibridgeError::from)
    }

    /// 发送带认证的 GET 请求，并用 OpenAI 错误映射处理响应
    async fn get_authed_json(&self, path: &str) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .compat
            .http_inner()
            .get(&url)
            .bearer_auth(self.api_key().unwrap_or(""))
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    AibridgeError::Timeout
                } else {
                    AibridgeError::Network(e)
                }
            })?;
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(OpenAiCompatAdapter::map_api_error(status_code, &body_text));
        }
        resp.json::<Value>().await.map_err(AibridgeError::from)
    }

    /// 转换统一消息为 Cohere 格式，并提取 system prompt
    ///
    /// 对齐 Python v1 `CohereAdapter._convert_messages`：
    /// - system 消息 → 提取为 `system_prompt`（最后一条 system 覆盖前面的）
    /// - user 消息 → `{"role": "USER", "content": ...}`（多模态取文本拼接，纯文本直接取）
    /// - 其他（assistant 等）→ `{"role": "CHATBOT", "content": ...}`
    ///
    /// 返回 `(chat_history, system_prompt)`，`chat_history` 不含最后一条 user 消息
    /// （最后一条 user 消息作为 Cohere `message` 字段单独发送）。
    fn convert_messages(messages: &[ChatMessage]) -> (Vec<Value>, Option<String>) {
        let mut converted: Vec<Value> = Vec::new();
        let mut system_prompt: Option<String> = None;

        for msg in messages {
            match msg {
                ChatMessage::System { content, .. } => {
                    system_prompt = Some(content.clone());
                }
                ChatMessage::User { content, .. } => {
                    let text = match content {
                        UserContent::Text(s) => s.clone(),
                        // 多模态：拼接所有 Text 部件，忽略图片
                        UserContent::Parts(parts) => parts
                            .iter()
                            .filter_map(|p| match p {
                                crate::model::chat::ContentPart::Text { text } => {
                                    Some(text.clone())
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    };
                    converted.push(json!({ "role": "USER", "content": text }));
                }
                ChatMessage::Assistant { content, .. } => {
                    let text = content.clone().unwrap_or_default();
                    converted.push(json!({ "role": "CHATBOT", "content": text }));
                }
                ChatMessage::Tool { content, .. } => {
                    // 工具结果消息按 CHATBOT 角色处理（Cohere 无原生 tool 角色）
                    converted.push(json!({ "role": "CHATBOT", "content": content }));
                }
            }
        }

        (converted, system_prompt)
    }

    /// 构造 Cohere chat 请求体
    ///
    /// 对齐 Python v1 `CohereAdapter.chat` 的 body 构造：
    /// - `message`：最后一条 user 消息的 content（无则空串）
    /// - `chat_history`：除最后一条 user 消息外的所有消息
    /// - `system_prompt`：若有 system 消息则填入
    /// - `temperature` / `max_tokens`：从统一请求透传
    /// - `extra` 透传到顶层
    fn build_chat_body(req: &ChatRequest, stream: bool) -> Value {
        let (converted, system_prompt) = Self::convert_messages(&req.messages);

        // 最后一条 user 消息的 content 作为 Cohere message 字段
        let message = converted
            .last()
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        // chat_history 不含最后一条消息
        let chat_history = if converted.len() > 1 {
            converted[..converted.len() - 1].to_vec()
        } else {
            Vec::new()
        };

        let mut body = json!({
            "model": req.model,
            "message": message,
            "chat_history": chat_history,
        });
        if stream {
            body["stream"] = json!(true);
        }
        if let Some(sp) = system_prompt {
            body["system_prompt"] = json!(sp);
        }
        if let Some(t) = req.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(mt) = req.max_tokens {
            body["max_tokens"] = json!(mt);
        }
        // extra 透传到顶层
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        body
    }

    /// 解析 Cohere chat 响应 → ChatCompletion
    ///
    /// Cohere 响应格式：`{"text": "...", "usage": {"tokens": {"input_tokens":..., "output_tokens":...}}}`
    /// （非标准 OpenAI choices 结构）。对应 Python v1 `CohereAdapter._parse_response`。
    fn parse_chat_completion(value: &Value, fallback_model: &str) -> Result<ChatCompletion> {
        let text = value
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // usage 解析：Cohere 用 tokens.input_tokens / tokens.output_tokens
        let usage = value.get("usage").and_then(parse_cohere_usage);

        Ok(ChatCompletion {
            id: value
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| util::generate_id("chatcmpl")),
            object: "chat.completion".to_string(),
            created: value
                .get("created_at")
                .and_then(|v| v.as_u64())
                .or_else(|| value.get("created").and_then(|v| v.as_u64()))
                .unwrap_or_else(util::current_timestamp),
            model: fallback_model.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChoiceMessage {
                    role: "assistant".to_string(),
                    content: Some(text),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage,
            service_tier: None,
            system_fingerprint: None,
        })
    }

    /// 解析单个 Cohere 流式 chunk
    ///
    /// Cohere 流式事件格式：`{"event_type": "text-generation"|"stream-end", "text": "...", ...}`
    /// - `text-generation`：增量文本，delta.content = text，finish_reason = None
    /// - `stream-end`：结束事件，delta.content = ""，finish_reason = "stop"
    /// - 其他 event_type：返回 None（跳过）
    ///
    /// 对应 Python v1 `CohereAdapter._parse_chunk`。
    fn parse_chunk(value: &Value, fallback_model: &str) -> Option<ChatCompletionChunk> {
        let event_type = value
            .get("event_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let generation_id = value
            .get("generation_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| util::generate_id("chatcmpl"));

        match event_type {
            "text-generation" => {
                let text = value.get("text").and_then(|v| v.as_str()).unwrap_or("");
                Some(ChatCompletionChunk {
                    id: generation_id,
                    object: "chat.completion.chunk".to_string(),
                    created: util::current_timestamp(),
                    model: fallback_model.to_string(),
                    choices: vec![ChatCompletionDelta {
                        index: 0,
                        delta: DeltaMessage {
                            role: Some("assistant".to_string()),
                            content: Some(text.to_string()),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                })
            }
            "stream-end" => Some(ChatCompletionChunk {
                id: generation_id,
                object: "chat.completion.chunk".to_string(),
                created: util::current_timestamp(),
                model: fallback_model.to_string(),
                choices: vec![ChatCompletionDelta {
                    index: 0,
                    delta: DeltaMessage {
                        role: Some("assistant".to_string()),
                        content: Some(String::new()),
                        tool_calls: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: None,
            }),
            _ => None,
        }
    }

    /// 解析 Cohere /models 响应 → Vec<ModelInfo>
    ///
    /// Cohere 响应：`{"models": [{"name": "...", ...}]}`，模型 ID 字段名为 `name`
    /// （非标准 OpenAI `id`），需转换为统一 `id` 字段。
    /// 对应 Python v1 `CohereAdapter.list_models` 的预处理逻辑。
    fn parse_models(value: &Value, provider: &str) -> Vec<ModelInfo> {
        let arr = value.get("models").and_then(|v| v.as_array());
        match arr {
            Some(arr) => arr
                .iter()
                .map(|m| {
                    // Cohere 用 name 作为模型 ID，统一映射到 id
                    let id = m
                        .get("id")
                        .and_then(|v| v.as_str())
                        .or_else(|| m.get("name").and_then(|v| v.as_str()))
                        .unwrap_or("")
                        .to_string();
                    let model_type = infer_model_type(&id);
                    ModelInfo {
                        name: id.clone(),
                        id,
                        model_type,
                        provider: provider.to_string(),
                        capabilities: Vec::new(),
                        max_tokens: None,
                        supports_streaming: matches!(model_type, ModelType::Chat),
                        description: m
                            .get("description")
                            .and_then(|v| v.as_str())
                            .map(str::to_owned),
                        created: None,
                    }
                })
                .collect(),
            None => Vec::new(),
        }
    }
}

#[async_trait]
impl Adapter for CohereAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
    }

    fn capabilities(&self) -> CapabilitySet {
        self.compat.capabilities_set().clone()
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    async fn start(&mut self) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    /// 文本对话（Cohere 特有协议：POST /chat）
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.ensure_capability(Capabilities::Chat)?;
        let body = Self::build_chat_body(&req, false);
        let value = self.post_authed_json("chat", &body).await?;
        Self::parse_chat_completion(&value, &req.model)
    }

    /// 流式文本对话（Cohere 特有协议：POST /chat stream=true）
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.ensure_capability(Capabilities::ChatStream)?;
        let body = Self::build_chat_body(&req, true);
        let url = self.url("chat");

        let resp = self
            .compat
            .http_inner()
            .post(&url)
            .bearer_auth(self.api_key().unwrap_or(""))
            .header("accept", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    AibridgeError::Timeout
                } else {
                    AibridgeError::Network(e)
                }
            })?;
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(OpenAiCompatAdapter::map_api_error(status_code, &body_text));
        }

        let model = req.model.clone();
        // 按字节流读取，按行切分解析 SSE（Cohere 流式同样是 SSE data: 格式）
        let byte_stream = resp
            .bytes_stream()
            .map_err(|e| e.to_string())
            .map(|r| r.map(|b| b.to_vec()));
        let lines_stream = CohereLinesStream::new(byte_stream);

        let stream = async_stream::stream! {
            let mut s = lines_stream;
            while let Some(line_result) = s.next().await {
                let line = match line_result {
                    Ok(l) => l,
                    Err(msg) => {
                        yield Err(AibridgeError::Api {
                            status: 0,
                            message: format!("流式读取错误: {msg}"),
                        });
                        return;
                    }
                };
                let line = line.trim();
                // 空行或注释行（心跳）跳过
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }
                // 去除 "data: " 前缀
                let data = if let Some(rest) = line.strip_prefix("data: ") {
                    rest
                } else if let Some(rest) = line.strip_prefix("data:") {
                    rest
                } else {
                    continue;
                };
                // 结束标记
                if data.trim() == "[DONE]" {
                    return;
                }
                match serde_json::from_str::<Value>(data) {
                    Ok(v) => match Self::parse_chunk(&v, &model) {
                        Some(chunk) => yield Ok(chunk),
                        None => continue,
                    },
                    // 单行 JSON 解析失败不致命，跳过（与 Python 老版一致）
                    Err(_) => continue,
                }
            }
        };

        Ok(stream.boxed())
    }

    /// 模型列表（Cohere 特有：GET /models，响应 models[].name 即模型 ID）
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let value = self.get_authed_json("models").await?;
        let models = Self::parse_models(&value, self.provider_type());
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }

    // image_generate / video_create / video_poll / embed / transcribe / speech / list_voices
    // 走 trait 默认实现，返 UnsupportedCapability，与 Python 老版行为一致。
}

/// 解析 Cohere usage 统计
///
/// Cohere usage 格式：`{"tokens": {"input_tokens": N, "output_tokens": M}}`，
/// 需转换为统一 ChatUsage（prompt/completion/total）。
fn parse_cohere_usage(v: &Value) -> Option<crate::model::chat::ChatUsage> {
    let tokens = v.get("tokens")?;
    let prompt = tokens.get("input_tokens").and_then(|x| x.as_u64())?;
    let completion = tokens
        .get("output_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(0);
    Some(crate::model::chat::ChatUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    })
}

// ==================== Cohere SSE 行流适配器 ====================

/// 将字节流按行切分的适配器（Cohere 流式用）
///
/// 与 `openai_compat::LinesStream` 等价，独立实现避免引用其私有结构。
struct CohereLinesStream<S> {
    inner: S,
    buffer: Vec<u8>,
}

impl<S> CohereLinesStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
        }
    }
}

impl<S> futures::Stream for CohereLinesStream<S>
where
    S: futures::Stream<Item = std::result::Result<Vec<u8>, String>> + Unpin,
{
    type Item = std::result::Result<String, String>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;
        loop {
            // 先看缓冲区是否已有完整行
            if let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
                let mut line: Vec<u8> = self.buffer.drain(..=pos).collect();
                // 去掉末尾 \n 与可能的 \r
                if line.last() == Some(&b'\n') {
                    line.pop();
                }
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                let s = String::from_utf8_lossy(&line).into_owned();
                return Poll::Ready(Some(Ok(s)));
            }
            // 缓冲区无完整行，拉取下一 chunk
            match std::pin::Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Err(msg))) => return Poll::Ready(Some(Err(msg))),
                Poll::Ready(Some(Ok(chunk))) => {
                    self.buffer.extend_from_slice(&chunk);
                    // 继续循环，尝试从缓冲区切出行
                }
                Poll::Ready(None) => {
                    // 流结束，把缓冲区剩余内容作为最后一行返回
                    if !self.buffer.is_empty() {
                        let mut line = std::mem::take(&mut self.buffer);
                        if line.last() == Some(&b'\n') {
                            line.pop();
                        }
                        if line.last() == Some(&b'\r') {
                            line.pop();
                        }
                        let s = String::from_utf8_lossy(&line).into_owned();
                        return Poll::Ready(Some(Ok(s)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

// ==================== Perplexity 适配器 ====================

/// Perplexity AI 适配器
///
/// OpenAI 兼容协议，全部能力委托给 `OpenAiCompatAdapter` 地基。
/// Perplexity 特有参数 `extra_body` 经统一请求的 `extra` 透传到请求体顶层。
///
/// - Base URL: `https://api.perplexity.ai`
/// - Chat: `POST /chat/completions`
/// - Models: `GET /models`
/// - 认证: Bearer Token
/// - 文档: <https://docs.perplexity.ai/>
pub struct PerplexityAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl PerplexityAdapter {
    /// 创建 Perplexity 适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_PERPLEXITY_BASE_URL,
            perplexity_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 compat 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "perplexity";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "Perplexity AI";
}

#[async_trait]
impl Adapter for PerplexityAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
    }

    fn capabilities(&self) -> CapabilitySet {
        self.compat.capabilities_set().clone()
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    async fn start(&mut self) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.compat.chat(req).await
    }

    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.compat.chat_stream(req).await
    }

    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientOptions;
    use crate::http::HttpClient;
    use crate::model::chat::ChatMessage;
    use crate::model::image::ImageRequest;
    use crate::model::options::{EmbedInput, EmbedRequest, ReasoningEffort};
    use crate::model::video::VideoRequest;
    use futures::stream::StreamExt;
    use mockito::Server;
    use std::collections::HashMap;

    // ==================== 通用测试辅助 ====================

    /// 构造测试用 OpenAiCompatAdapter（指向 mockito server，给定 provider 信息与能力）
    fn make_compat(
        server: &Server,
        provider_type: &str,
        provider_name: &str,
        caps: CapabilitySet,
    ) -> OpenAiCompatAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options(provider_type, opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        OpenAiCompatAdapter::with_http(http, config, provider_type, provider_name, caps)
    }

    /// 构造测试用 DeepSeekAdapter（指向 mockito server）
    fn make_deepseek(server: &Server) -> DeepSeekAdapter {
        let compat = make_compat(server, "deepseek", "DeepSeek", deepseek_capabilities());
        DeepSeekAdapter::with_compat(compat)
    }

    /// 构造测试用 StepFunAdapter（指向 mockito server）
    fn make_stepfun(server: &Server) -> StepFunAdapter {
        let compat = make_compat(
            server,
            "stepfun",
            "阶跃星辰 StepFun",
            stepfun_capabilities(),
        );
        StepFunAdapter::with_compat(compat)
    }

    /// 构造测试用 MistralAdapter（指向 mockito server）
    fn make_mistral(server: &Server) -> MistralAdapter {
        let compat = make_compat(server, "mistral", "Mistral AI", mistral_capabilities());
        MistralAdapter::with_compat(compat)
    }

    /// 构造测试用 CohereAdapter（指向 mockito server）
    fn make_cohere(server: &Server) -> CohereAdapter {
        let compat = make_compat(server, "cohere", "Cohere", cohere_capabilities());
        CohereAdapter::with_compat(compat)
    }

    /// 构造测试用 PerplexityAdapter（指向 mockito server）
    fn make_perplexity(server: &Server) -> PerplexityAdapter {
        let compat = make_compat(
            server,
            "perplexity",
            "Perplexity AI",
            perplexity_capabilities(),
        );
        PerplexityAdapter::with_compat(compat)
    }

    /// 标准 OpenAI chat 成功响应体
    fn openai_chat_body() -> Value {
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        })
    }

    // ==================== 默认 Base URL 常量校验 ====================

    #[test]
    fn default_base_urls_match_python() {
        assert_eq!(DEFAULT_DEEPSEEK_BASE_URL, "https://api.deepseek.com");
        assert_eq!(DEFAULT_STEPFUN_BASE_URL, "https://api.stepfun.com/v1");
        assert_eq!(DEFAULT_MISTRAL_BASE_URL, "https://api.mistral.ai/v1");
        assert_eq!(DEFAULT_COHERE_BASE_URL, "https://api.cohere.ai/v1");
        assert_eq!(DEFAULT_PERPLEXITY_BASE_URL, "https://api.perplexity.ai");
    }

    // ==================== DeepSeek 测试 ====================

    #[tokio::test]
    async fn deepseek_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_deepseek(&server);
        assert_eq!(adapter.provider_type(), "deepseek");
        assert_eq!(adapter.provider_name(), "DeepSeek");
    }

    #[tokio::test]
    async fn deepseek_requires_api_key() {
        let server = Server::new_async().await;
        let adapter = make_deepseek(&server);
        assert!(adapter.requires_api_key());
    }

    #[tokio::test]
    async fn deepseek_capabilities_include_chat_and_vision() {
        let server = Server::new_async().await;
        let adapter = make_deepseek(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::Vision));
        // 不支持 image / embed
        assert!(!caps.contains(&Capabilities::ImageGenerate));
        assert!(!caps.contains(&Capabilities::Embedding));
    }

    #[tokio::test]
    async fn deepseek_chat_success() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_deepseek(&server);
        let req = ChatRequest::builder("deepseek-chat", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .max_tokens(100)
            .build();
        let resp = adapter.chat(req).await.expect("chat 应成功");
        assert_eq!(resp.id, "chatcmpl-1");
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn deepseek_chat_auto_injects_thinking_when_reasoning_effort_present() {
        // 对齐 Python：传 reasoning_effort 时自动注入 thinking={"type":"enabled"}
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "deepseek-reasoner",
                "reasoning_effort": "high",
                "thinking": {"type": "enabled"}
            })))
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_deepseek(&server);
        let req = ChatRequest::builder("deepseek-reasoner", vec![ChatMessage::user("hi")])
            .reasoning_effort(ReasoningEffort::High)
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn deepseek_chat_no_thinking_without_reasoning_effort() {
        // 不传 reasoning_effort 时不应注入 thinking。
        // 用 PartialJson 验证请求体不含 thinking 字段：mockito 无原生"不含字段"断言，
        // 故用完整 JsonString 严格匹配预期 body（model + messages + temperature，无 thinking）。
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_deepseek(&server);
        let req = ChatRequest::builder("deepseek-chat", vec![ChatMessage::user("hi")]).build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn deepseek_chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(json!({"error": {"message": "invalid key"}}).to_string())
            .create_async()
            .await;
        let adapter = make_deepseek(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn deepseek_chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;
        let adapter = make_deepseek(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn deepseek_chat_stream_parses_sse() {
        let mut server = Server::new_async().await;
        let sse = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let adapter = make_deepseek(&server);
        let req = ChatRequest::builder("deepseek-chat", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.expect("stream 应建立");
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(chunks.len(), 3);
        let mut content = String::new();
        content.push_str(chunks[1].choices[0].delta.content.as_deref().unwrap_or(""));
        content.push_str(chunks[2].choices[0].delta.content.as_deref().unwrap_or(""));
        assert_eq!(content, "Hello world");
    }

    #[tokio::test]
    async fn deepseek_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "deepseek-chat", "object": "model"},
                        {"id": "deepseek-reasoner", "object": "model"}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_deepseek(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "deepseek-chat");
        assert_eq!(models[0].provider, "deepseek");
    }

    #[tokio::test]
    async fn deepseek_list_models_filter_by_type() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "deepseek-chat", "object": "model"},
                        {"id": "dall-e-3", "object": "model"}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_deepseek(&server);
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "dall-e-3");
    }

    #[tokio::test]
    async fn deepseek_list_models_error_429() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;
        let adapter = make_deepseek(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn deepseek_image_generate_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_deepseek(&server);
        let req = ImageRequest::builder("m", "prompt").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn deepseek_video_create_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_deepseek(&server);
        let req = VideoRequest::builder("m", "p").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn deepseek_embed_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_deepseek(&server);
        let req = EmbedRequest {
            model: "m".into(),
            input: EmbedInput::Single("hi".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let err = adapter.embed(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ==================== StepFun 测试 ====================

    #[tokio::test]
    async fn stepfun_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_stepfun(&server);
        assert_eq!(adapter.provider_type(), "stepfun");
        assert_eq!(adapter.provider_name(), "阶跃星辰 StepFun");
    }

    #[tokio::test]
    async fn stepfun_chat_success() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "step-1-8k",
                "temperature": 0.5,
                "max_tokens": 50
            })))
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_stepfun(&server);
        let req = ChatRequest::builder("step-1-8k", vec![ChatMessage::user("hi")])
            .temperature(0.5)
            .max_tokens(50)
            .build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn stepfun_chat_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(404)
            .with_body(json!({"error": {"message": "model not found"}}).to_string())
            .create_async()
            .await;
        let adapter = make_stepfun(&server);
        let req = ChatRequest::builder("bad-model", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn stepfun_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "step-1-8k", "object": "model"},
                        {"id": "step-1v-8k", "object": "model"}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_stepfun(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].provider, "stepfun");
    }

    #[tokio::test]
    async fn stepfun_chat_stream_sends_stream_true() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "stream": true,
                "stream_options": {"include_usage": true}
            })))
            .with_status(200)
            .with_body("data: [DONE]\n")
            .create_async()
            .await;
        let adapter = make_stepfun(&server);
        let req = ChatRequest::builder("step-1-8k", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        while stream.next().await.is_some() {}
        mock.assert_async().await;
    }

    // ==================== Mistral 测试 ====================

    #[tokio::test]
    async fn mistral_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_mistral(&server);
        assert_eq!(adapter.provider_type(), "mistral");
        assert_eq!(adapter.provider_name(), "Mistral AI");
    }

    #[tokio::test]
    async fn mistral_chat_success() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "mistral-large-latest",
                "temperature": 0.7,
                "top_p": 0.9
            })))
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_mistral(&server);
        let req = ChatRequest::builder("mistral-large-latest", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .top_p(0.9)
            .build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn mistral_chat_error_429() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(429)
            .with_body(json!({"error": {"message": "rate limit"}}).to_string())
            .create_async()
            .await;
        let adapter = make_mistral(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn mistral_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "mistral-large-latest", "object": "model"},
                        {"id": "mistral-embed", "object": "model"}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_mistral(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].provider, "mistral");
    }

    #[tokio::test]
    async fn mistral_chat_stream_parses_sse() {
        let mut server = Server::new_async().await;
        let sse = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mistral-large-latest\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mistral-large-latest\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(sse)
            .create_async()
            .await;
        let adapter = make_mistral(&server);
        let req =
            ChatRequest::builder("mistral-large-latest", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        let mut content = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            if let Some(c) = chunk.choices[0].delta.content.as_deref() {
                content.push_str(c);
            }
        }
        assert_eq!(content, "Hi!");
    }

    // ==================== Cohere 测试 ====================

    #[tokio::test]
    async fn cohere_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_cohere(&server);
        assert_eq!(adapter.provider_type(), "cohere");
        assert_eq!(adapter.provider_name(), "Cohere");
    }

    #[tokio::test]
    async fn cohere_requires_api_key() {
        let server = Server::new_async().await;
        let adapter = make_cohere(&server);
        assert!(adapter.requires_api_key());
    }

    #[tokio::test]
    async fn cohere_chat_success() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat")
            .match_header("authorization", "Bearer test-key")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "command-r-plus",
                "message": "hi",
                "chat_history": [],
                "temperature": 0.7
            })))
            .with_status(200)
            .with_body(
                json!({
                    "id": "gen-1",
                    "text": "Hello from Cohere!",
                    "usage": {"tokens": {"input_tokens": 3, "output_tokens": 4}}
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_cohere(&server);
        let req = ChatRequest::builder("command-r-plus", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .build();
        let resp = adapter.chat(req).await.expect("chat 应成功");
        assert_eq!(resp.id, "gen-1");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("Hello from Cohere!")
        );
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        // usage 解析：Cohere tokens.input/output → prompt/completion/total
        let usage = resp.usage.as_ref().expect("应有 usage");
        assert_eq!(usage.prompt_tokens, 3);
        assert_eq!(usage.completion_tokens, 4);
        assert_eq!(usage.total_tokens, 7);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn cohere_chat_extracts_system_prompt_and_history() {
        // 多消息场景：system 提取为 system_prompt，历史消息进 chat_history，
        // 最后一条 user 作为 message
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "command-r-plus",
                "message": "second question",
                "chat_history": [
                    {"role": "USER", "content": "first question"},
                    {"role": "CHATBOT", "content": "first answer"}
                ],
                "system_prompt": "you are helpful"
            })))
            .with_status(200)
            .with_body(json!({"text": "ok"}).to_string())
            .create_async()
            .await;
        let adapter = make_cohere(&server);
        let req = ChatRequest::builder(
            "command-r-plus",
            vec![
                ChatMessage::system("you are helpful"),
                ChatMessage::user("first question"),
                ChatMessage::assistant("first answer"),
                ChatMessage::user("second question"),
            ],
        )
        .build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("ok"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn cohere_chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat")
            .with_status(401)
            .with_body(json!({"error": {"message": "invalid token"}}).to_string())
            .create_async()
            .await;
        let adapter = make_cohere(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn cohere_chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;
        let adapter = make_cohere(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn cohere_chat_stream_parses_events() {
        let mut server = Server::new_async().await;
        // Cohere 流式：text-generation 事件 + stream-end 事件 + [DONE]
        let sse = "data: {\"event_type\":\"text-generation\",\"text\":\"Hello\",\"generation_id\":\"g1\"}\n\
                   data: {\"event_type\":\"text-generation\",\"text\":\" world\",\"generation_id\":\"g1\"}\n\
                   data: {\"event_type\":\"stream-end\",\"generation_id\":\"g1\"}\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let adapter = make_cohere(&server);
        let req = ChatRequest::builder("command-r-plus", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.expect("stream 应建立");
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        // 3 个有效 chunk：2 个 text-generation + 1 个 stream-end
        assert_eq!(chunks.len(), 3);
        // 拼接前两块内容
        let mut content = String::new();
        content.push_str(chunks[0].choices[0].delta.content.as_deref().unwrap_or(""));
        content.push_str(chunks[1].choices[0].delta.content.as_deref().unwrap_or(""));
        assert_eq!(content, "Hello world");
        // 第三块是 stream-end，finish_reason = stop，content 为空
        assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunks[2].choices[0].delta.content.as_deref(), Some(""));
    }

    #[tokio::test]
    async fn cohere_chat_stream_skips_unknown_event_types() {
        let mut server = Server::new_async().await;
        // 含未知 event_type（stream-start）应被跳过
        let sse = "data: {\"event_type\":\"stream-start\"}\n\
                   data: {\"event_type\":\"text-generation\",\"text\":\"Hi\"}\n\
                   data: {\"event_type\":\"stream-end\"}\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat")
            .with_status(200)
            .with_body(sse)
            .create_async()
            .await;
        let adapter = make_cohere(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        // stream-start 被跳过，仅 text-generation + stream-end
        assert_eq!(chunks.len(), 2);
    }

    #[tokio::test]
    async fn cohere_chat_stream_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad token"}}).to_string())
            .create_async()
            .await;
        let adapter = make_cohere(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let result = adapter.chat_stream(req).await;
        match result {
            Err(e) => assert!(matches!(e, AibridgeError::Authentication { .. })),
            Ok(_) => panic!("chat_stream 应返回错误而非 stream"),
        }
    }

    #[tokio::test]
    async fn cohere_list_models_success_converts_name_to_id() {
        // Cohere /models 响应用 name 字段作为模型 ID，需转换为统一 id
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "models": [
                        {"name": "command-r-plus", "description": "Command R+"},
                        {"name": "command-r", "description": "Command R"}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_cohere(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        // name 字段被映射到 id
        assert_eq!(models[0].id, "command-r-plus");
        assert_eq!(models[0].name, "command-r-plus");
        assert_eq!(models[0].provider, "cohere");
        assert_eq!(models[0].description.as_deref(), Some("Command R+"));
    }

    #[tokio::test]
    async fn cohere_list_models_filter_by_type() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "models": [
                        {"name": "command-r-plus"},
                        {"name": "dall-e-3"}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_cohere(&server);
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "dall-e-3");
    }

    #[tokio::test]
    async fn cohere_list_models_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad token"}}).to_string())
            .create_async()
            .await;
        let adapter = make_cohere(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn cohere_image_generate_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_cohere(&server);
        let req = ImageRequest::builder("m", "prompt").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn cohere_embed_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_cohere(&server);
        let req = EmbedRequest {
            model: "m".into(),
            input: EmbedInput::Single("hi".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let err = adapter.embed(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ==================== Perplexity 测试 ====================

    #[tokio::test]
    async fn perplexity_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_perplexity(&server);
        assert_eq!(adapter.provider_type(), "perplexity");
        assert_eq!(adapter.provider_name(), "Perplexity AI");
    }

    #[tokio::test]
    async fn perplexity_capabilities_include_web_search() {
        let server = Server::new_async().await;
        let adapter = make_perplexity(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::WebSearch));
    }

    #[tokio::test]
    async fn perplexity_chat_success() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "sonar",
                "temperature": 0.5
            })))
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_perplexity(&server);
        let req = ChatRequest::builder("sonar", vec![ChatMessage::user("hi")])
            .temperature(0.5)
            .build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn perplexity_chat_passes_extra_body_through() {
        // Perplexity 特有参数 extra_body 经统一请求的 extra 透传到请求体顶层
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "sonar",
                "extra_body": {"search_recency_filter": "month"}
            })))
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_perplexity(&server);
        let req = ChatRequest::builder("sonar", vec![ChatMessage::user("hi")])
            .extra("extra_body", json!({"search_recency_filter": "month"}))
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn perplexity_chat_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(404)
            .with_body(json!({"error": {"message": "model not found"}}).to_string())
            .create_async()
            .await;
        let adapter = make_perplexity(&server);
        let req = ChatRequest::builder("bad-model", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn perplexity_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "sonar", "object": "model"},
                        {"id": "sonar-pro", "object": "model"}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_perplexity(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "sonar");
        assert_eq!(models[0].provider, "perplexity");
    }

    #[tokio::test]
    async fn perplexity_chat_stream_parses_sse() {
        let mut server = Server::new_async().await;
        let sse = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"sonar\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Answer\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"sonar\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(sse)
            .create_async()
            .await;
        let adapter = make_perplexity(&server);
        let req = ChatRequest::builder("sonar", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        let mut content = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            if let Some(c) = chunk.choices[0].delta.content.as_deref() {
                content.push_str(c);
            }
        }
        assert_eq!(content, "Answer!");
    }

    #[tokio::test]
    async fn perplexity_image_generate_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_perplexity(&server);
        let req = ImageRequest::builder("m", "prompt").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ==================== start/close 无副作用 ====================

    #[tokio::test]
    async fn deepseek_start_close_are_noops() {
        let server = Server::new_async().await;
        let mut adapter = make_deepseek(&server);
        adapter.start().await.unwrap();
        adapter.close().await.unwrap();
    }

    #[tokio::test]
    async fn cohere_start_close_are_noops() {
        let server = Server::new_async().await;
        let mut adapter = make_cohere(&server);
        adapter.start().await.unwrap();
        adapter.close().await.unwrap();
    }

    // ==================== Cohere 辅助函数单元测试 ====================

    #[test]
    fn cohere_parse_chunk_text_generation_returns_content() {
        let v = json!({"event_type": "text-generation", "text": "hello"});
        let chunk = CohereAdapter::parse_chunk(&v, "command-r").expect("应返回 Some");
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn cohere_parse_chunk_stream_end_returns_stop() {
        let v = json!({"event_type": "stream-end", "generation_id": "g1"});
        let chunk = CohereAdapter::parse_chunk(&v, "command-r").expect("应返回 Some");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some(""));
        assert_eq!(chunk.id, "g1");
    }

    #[test]
    fn cohere_parse_chunk_unknown_event_returns_none() {
        let v = json!({"event_type": "stream-start"});
        assert!(CohereAdapter::parse_chunk(&v, "command-r").is_none());
    }

    #[test]
    fn cohere_parse_models_handles_name_field() {
        let v = json!({
            "models": [
                {"name": "command-r-plus", "description": "top"},
                {"id": "embed-english-v3.0"}
            ]
        });
        let models = CohereAdapter::parse_models(&v, "cohere");
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "command-r-plus");
        assert_eq!(models[0].description.as_deref(), Some("top"));
        // 第二个用 id 字段
        assert_eq!(models[1].id, "embed-english-v3.0");
    }

    #[test]
    fn cohere_parse_models_empty_returns_vec() {
        let v = json!({"models": []});
        let models = CohereAdapter::parse_models(&v, "cohere");
        assert!(models.is_empty());
        // 无 models 字段
        let v2 = json!({});
        assert!(CohereAdapter::parse_models(&v2, "cohere").is_empty());
    }

    #[test]
    fn cohere_parse_chat_completion_extracts_text_and_usage() {
        let v = json!({
            "id": "gen-9",
            "text": "response text",
            "usage": {"tokens": {"input_tokens": 10, "output_tokens": 20}}
        });
        let cc = CohereAdapter::parse_chat_completion(&v, "command-r").unwrap();
        assert_eq!(cc.id, "gen-9");
        assert_eq!(
            cc.choices[0].message.content.as_deref(),
            Some("response text")
        );
        let usage = cc.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn cohere_parse_chat_completion_without_usage() {
        let v = json!({"text": "no usage here"});
        let cc = CohereAdapter::parse_chat_completion(&v, "command-r").unwrap();
        assert_eq!(
            cc.choices[0].message.content.as_deref(),
            Some("no usage here")
        );
        assert!(cc.usage.is_none());
    }

    #[test]
    fn cohere_convert_messages_extracts_system_and_history() {
        let msgs = vec![
            ChatMessage::system("be helpful"),
            ChatMessage::user("q1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("q2"),
        ];
        let (history, system) = CohereAdapter::convert_messages(&msgs);
        assert_eq!(system.as_deref(), Some("be helpful"));
        // chat_history 应含前 3 条（system 已提取，user q1 + assistant a1）
        assert_eq!(history.len(), 3);
        assert_eq!(history[0]["role"], "USER");
        assert_eq!(history[0]["content"], "q1");
        assert_eq!(history[1]["role"], "CHATBOT");
        assert_eq!(history[1]["content"], "a1");
        // 最后一条 user 作为 message，在 build_chat_body 中单独处理
        assert_eq!(history[2]["role"], "USER");
        assert_eq!(history[2]["content"], "q2");
    }
}
