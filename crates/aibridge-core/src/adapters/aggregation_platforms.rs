//! 聚合平台模型适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/aggregation_platforms.py`。
//!
//! 支持四个聚合平台（均为 OpenAI 兼容协议族，差异在 base_url / 端点路径 / 响应结构）：
//! - **SiliconFlow (硅基流动)**：标准 OpenAI 兼容，支持 chat/reasoning/embedding
//! - **Together AI**：标准 OpenAI 兼容，聚合开源模型
//! - **Fireworks AI**：标准 OpenAI 兼容，高性能推理
//! - **Cloudflare Workers AI**：边缘推理，使用 `/v1/run/{model}` 特殊端点 + `result` 响应结构
//!
//! ## 结构
//!
//! Python v1 是 4 个独立 adapter 类，统一注册到工厂。Rust 同样实现 4 个独立 struct：
//! - `SiliconFlowAdapter` / `TogetherAIAdapter` / `FireworksAIAdapter`：组合 `OpenAiCompatAdapter`
//!   地基，chat/chat_stream/image/embed/list_models 全部委托（仅 base_url/provider_type/capabilities
//!   差异）
//! - `CloudflareAIAdapter`：组合 `OpenAiCompatAdapter` 仅复用 embed（标准 /embeddings 端点），
//!   chat/chat_stream/list_models 独立实现（Cloudflare 特有的 `/v1/run/{model}` 端点 +
//!   `result` 响应结构 + `/models/search` 端点）
//!
//! ## provider_type 标识（对齐 Python v1）
//!
//! | 平台 | provider_type | 别名 |
//! |---|---|---|
//! | SiliconFlow | `siliconflow` | `sf` |
//! | Together AI | `togetherai` | `together` |
//! | Fireworks AI | `fireworksai` | `fireworks` |
//! | Cloudflare Workers AI | `cloudflareai` | `cloudflare` / `workersai` |

use async_trait::async_trait;
use futures::stream::{StreamExt, TryStreamExt};
use serde_json::{json, Value};

use crate::adapter::{Adapter, Capabilities, CapabilitySet, ChatStream};
use crate::adapters::openai_compat::OpenAiCompatAdapter;
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::chat::{
    ChatChoice, ChatCompletion, ChatCompletionChunk, ChatCompletionDelta, ChatRequest,
    ChoiceMessage, DeltaMessage,
};
use crate::model::common::{infer_model_type, ModelInfo, ModelType};
use crate::model::options::{EmbedRequest, EmbeddingResult};
use crate::util;

// ==================== 默认 Base URL ====================

/// SiliconFlow 默认 Base URL
///
/// 对应 Python v1 `SiliconFlowAdapter.DEFAULT_BASE_URL`。
pub const DEFAULT_SILICONFLOW_BASE_URL: &str = "https://api.siliconflow.cn/v1";

/// Together AI 默认 Base URL
///
/// 对应 Python v1 `TogetherAIAdapter.DEFAULT_BASE_URL`。
pub const DEFAULT_TOGETHERAI_BASE_URL: &str = "https://api.together.xyz/v1";

/// Fireworks AI 默认 Base URL
///
/// 对应 Python v1 `FireworksAIAdapter.DEFAULT_BASE_URL`。
pub const DEFAULT_FIREWORKSAI_BASE_URL: &str = "https://api.fireworks.ai/inference/v1";

/// Cloudflare Workers AI 默认 Base URL（API 根，需拼接 account_id）
///
/// 对应 Python v1 `CloudflareAIAdapter.DEFAULT_BASE_URL`。
/// 实际请求 base 为 `{DEFAULT_CLOUDFLAREAI_BASE_URL}/accounts/{account_id}/ai`。
pub const DEFAULT_CLOUDFLAREAI_BASE_URL: &str = "https://api.cloudflare.com/client/v4";

// ==================== 能力集合构造 ====================

/// SiliconFlow 支持的能力集合
///
/// 对齐 Python v1 `SiliconFlowAdapter.supported_capabilities`。
fn siliconflow_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps.insert(Capabilities::ToolCall);
    caps.insert(Capabilities::Reasoning);
    caps.insert(Capabilities::JsonMode);
    caps.insert(Capabilities::Embedding);
    caps.insert(Capabilities::AudioTranscribe);
    caps.insert(Capabilities::AudioSpeech);
    caps
}

/// Together AI 支持的能力集合
///
/// 对齐 Python v1 `TogetherAIAdapter.supported_capabilities`。
fn togetherai_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps.insert(Capabilities::JsonMode);
    caps.insert(Capabilities::Embedding);
    caps.insert(Capabilities::AudioTranscribe);
    caps.insert(Capabilities::AudioSpeech);
    caps
}

/// Fireworks AI 支持的能力集合
///
/// 对齐 Python v1 `FireworksAIAdapter.supported_capabilities`。
fn fireworksai_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps.insert(Capabilities::ToolCall);
    caps.insert(Capabilities::JsonMode);
    caps.insert(Capabilities::Embedding);
    caps.insert(Capabilities::AudioTranscribe);
    caps
}

/// Cloudflare Workers AI 支持的能力集合
///
/// 对齐 Python v1 `CloudflareAIAdapter.supported_capabilities`。
fn cloudflareai_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps.insert(Capabilities::Embedding);
    caps
}

// ==================== SiliconFlow 适配器 ====================

/// SiliconFlow (硅基流动) 适配器
///
/// OpenAI 兼容协议，全部能力委托给 `OpenAiCompatAdapter` 地基。
///
/// - Base URL: `https://api.siliconflow.cn/v1`
/// - Chat: `POST /chat/completions`（支持 reasoning 等特有参数，经 `extra` 透传）
/// - Embed: `POST /embeddings`
/// - Models: `GET /models`
/// - 认证: Bearer Token
pub struct SiliconFlowAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/image/embed/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl SiliconFlowAdapter {
    /// 创建 SiliconFlow 适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            "siliconflow",
            "SiliconFlow 硅基流动",
            DEFAULT_SILICONFLOW_BASE_URL,
            siliconflow_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 HttpClient 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }
}

#[async_trait]
impl Adapter for SiliconFlowAdapter {
    fn provider_type(&self) -> &str {
        "siliconflow"
    }

    fn provider_name(&self) -> &str {
        "SiliconFlow 硅基流动"
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

    async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        self.compat.embed(req).await
    }

    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }
}

// ==================== Together AI 适配器 ====================

/// Together AI 适配器
///
/// OpenAI 兼容协议，全部能力委托给 `OpenAiCompatAdapter` 地基。
///
/// - Base URL: `https://api.together.xyz/v1`
/// - Chat: `POST /chat/completions`
/// - Embed: `POST /embeddings`
/// - Models: `GET /models`
/// - 认证: Bearer Token
pub struct TogetherAIAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/image/embed/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl TogetherAIAdapter {
    /// 创建 Together AI 适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            "togetherai",
            "Together AI",
            DEFAULT_TOGETHERAI_BASE_URL,
            togetherai_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 HttpClient 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }
}

#[async_trait]
impl Adapter for TogetherAIAdapter {
    fn provider_type(&self) -> &str {
        "togetherai"
    }

    fn provider_name(&self) -> &str {
        "Together AI"
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

    async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        self.compat.embed(req).await
    }

    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }
}

// ==================== Fireworks AI 适配器 ====================

/// Fireworks AI 适配器
///
/// OpenAI 兼容协议，全部能力委托给 `OpenAiCompatAdapter` 地基。
///
/// - Base URL: `https://api.fireworks.ai/inference/v1`
/// - Chat: `POST /chat/completions`
/// - Embed: `POST /embeddings`
/// - Models: `GET /models`
/// - 认证: Bearer Token
pub struct FireworksAIAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/image/embed/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl FireworksAIAdapter {
    /// 创建 Fireworks AI 适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            "fireworksai",
            "Fireworks AI",
            DEFAULT_FIREWORKSAI_BASE_URL,
            fireworksai_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 HttpClient 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }
}

#[async_trait]
impl Adapter for FireworksAIAdapter {
    fn provider_type(&self) -> &str {
        "fireworksai"
    }

    fn provider_name(&self) -> &str {
        "Fireworks AI"
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

    async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        self.compat.embed(req).await
    }

    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }
}

// ==================== Cloudflare Workers AI 适配器 ====================

/// Cloudflare Workers AI 适配器
///
/// 部分兼容 OpenAI 协议，但 chat 端点与响应结构与标准 OpenAI 不同：
/// - Base URL: `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai`
/// - Chat: `POST /v1/run/{model}`（模型用 `@cf/` 前缀）
/// - 响应: `{"result": {"response": "..."}}`（非标准 OpenAI choices 结构）
/// - Models: `GET /models/search`（响应 `{"result": {"models": [...]}}`）
/// - Embed: `POST /embeddings`（标准 OpenAI 兼容，委托 compat）
/// - 认证: Bearer Token + Account ID（account_id 从 `config.extra["account_id"]` 取）
///
/// `account_id` 为必填，构造时缺失则 `new()` 返回 `Validation` 错误。
pub struct CloudflareAIAdapter {
    /// OpenAI 兼容地基（仅 embed 委托给它，标准 /embeddings 端点）
    compat: OpenAiCompatAdapter,
    /// chat/list_models 端点专用 HTTP 客户端（独立于 compat，避免暴露 compat 私有字段）
    http: HttpClient,
    /// Cloudflare account_id（构造时校验非空，用于构造 ai_base；保留供调试/扩展）
    #[allow(dead_code)]
    account_id: String,
    /// Provider 配置（保留引用以便取 api_key）
    config: ProviderConfig,
}

impl CloudflareAIAdapter {
    /// 创建 Cloudflare 适配器
    ///
    /// - `config.extra["account_id"]` 必须存在且非空，否则返回 `Validation` 错误
    /// - `config.base_url` 为 None 时用 `DEFAULT_CLOUDFLAREAI_BASE_URL` 兜底
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let account_id = config
            .extra
            .get("account_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| {
                AibridgeError::validation(
                    "Cloudflare account_id 不能为空（请配置 extra.account_id）",
                )
            })?;

        let api_root = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_CLOUDFLAREAI_BASE_URL.to_string());

        // Cloudflare 实际请求 base：{api_root}/accounts/{account_id}/ai
        let ai_base = format!(
            "{}/accounts/{}/ai",
            api_root.trim_end_matches('/'),
            account_id
        );

        // compat 用于 embed（标准 /embeddings 端点），需让其 base_url 指向 ai_base。
        // OpenAiCompatAdapter::new 会优先用 config.base_url，故这里把 config 副本的
        // base_url 改写为 ai_base 后再传入，确保 embed 端点走 /accounts/{id}/ai/embeddings。
        let mut compat_config = config.clone();
        compat_config.base_url = Some(ai_base.clone());
        let compat = OpenAiCompatAdapter::new(
            compat_config,
            "cloudflareai",
            "Cloudflare Workers AI",
            &ai_base,
            cloudflareai_capabilities(),
        )?;

        // chat/list_models 端点专用 HttpClient，base_url 同 ai_base
        let http = HttpClient::new(
            &ClientOptions::builder()
                .api_key(config.api_key.clone().unwrap_or_default())
                .base_url(ai_base)
                .timeout(config.timeout)
                .max_retries(config.max_retries)
                .retry_delay(config.retry_delay)
                .build(),
        )?;

        Ok(Self {
            compat,
            http,
            account_id,
            config,
        })
    }

    /// 用显式 HttpClient + compat 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(
        compat: OpenAiCompatAdapter,
        http: HttpClient,
        account_id: impl Into<String>,
        config: ProviderConfig,
    ) -> Self {
        Self {
            compat,
            http,
            account_id: account_id.into(),
            config,
        }
    }

    /// account_id（测试可见）
    #[cfg(test)]
    pub fn account_id(&self) -> &str {
        &self.account_id
    }

    /// API key
    fn api_key(&self) -> Option<&str> {
        self.config.api_key.as_deref()
    }

    /// base_url（compat 持有的 ai_base）
    fn base_url(&self) -> &str {
        self.compat.base_url()
    }

    /// 拼接完整 URL（base_url + 相对路径）
    fn url(&self, path: &str) -> String {
        let base = self.base_url().trim_end_matches('/');
        let path = path.trim_start_matches('/');
        format!("{base}/{path}")
    }

    /// 校验能力是否被支持（Cloudflare 独立实现，因 ensure_capability 在 compat 内为私有）
    fn ensure_capability(&self, cap: Capabilities) -> Result<()> {
        if self.compat.capabilities_set().contains(&cap) {
            Ok(())
        } else {
            Err(AibridgeError::UnsupportedCapability {
                capability: format!("{} (provider: cloudflareai)", cap.as_str()),
            })
        }
    }

    /// 发送带认证的 POST JSON 请求，并用 OpenAI 错误映射处理响应
    async fn post_authed_json(&self, path: &str, body: &Value) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .http
            .inner()
            .post(&url)
            .bearer_auth(self.api_key().unwrap_or(""))
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
            .http
            .inner()
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

    /// 格式化 Cloudflare 模型名（补充 `@cf/` 前缀）
    ///
    /// 对应 Python v1 `CloudflareAIAdapter._format_model`。
    fn format_model(model: &str) -> String {
        if model.starts_with("@cf/") {
            model.to_string()
        } else {
            format!("@cf/{model}")
        }
    }

    /// 构造 Cloudflare chat 请求体
    ///
    /// 对应 Python v1 `CloudflareAIAdapter.chat` 的 body 构造：
    /// - 不含 `model`（模型在 URL 路径里）
    /// - 含 `messages` / `temperature` / `max_tokens` / `stream`
    /// - `extra` 透传
    fn build_chat_body(req: &ChatRequest, stream: bool) -> Value {
        let mut body = serde_json::to_value(req).unwrap_or_else(|_| json!({}));
        if let Some(obj) = body.as_object_mut() {
            // 移除 model（Cloudflare 模型在 URL 路径，不在 body）
            obj.remove("model");
            if stream {
                obj.insert("stream".to_string(), json!(true));
            } else if obj.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
                obj.insert("stream".to_string(), json!(false));
            }
            // extra 透传到顶层
            if let Some(extra) = obj.remove("extra") {
                if let Some(extra_map) = extra.as_object() {
                    for (k, v) in extra_map {
                        obj.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        body
    }

    /// 解析 Cloudflare chat 响应 → ChatCompletion
    ///
    /// Cloudflare 响应格式：`{"result": {"response": "..."}}`（非标准 OpenAI choices）。
    /// 对应 Python v1 `CloudflareAIAdapter._parse_response`。
    fn parse_chat_completion(value: &Value, fallback_model: &str) -> Result<ChatCompletion> {
        let result = value.get("result").unwrap_or(&Value::Null);
        let content = if result.is_object() {
            result
                .get("response")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        } else {
            // result 不是对象时，转为字符串作为 content
            result.as_str().unwrap_or("").to_string()
        };

        let usage = value.get("usage").and_then(parse_usage);

        Ok(ChatCompletion {
            id: value
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| util::generate_id("chatcmpl")),
            object: "chat.completion".to_string(),
            created: value
                .get("timestamp")
                .and_then(|v| v.as_u64())
                .or_else(|| value.get("created").and_then(|v| v.as_u64()))
                .unwrap_or_else(util::current_timestamp),
            model: fallback_model.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChoiceMessage {
                    role: "assistant".to_string(),
                    content: Some(content),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage,
            service_tier: None,
            system_fingerprint: None,
        })
    }

    /// 解析单个 Cloudflare 流式 chunk
    ///
    /// Cloudflare 流式响应：每个 SSE data 是 `{"result": {"response": "增量文本"}}`。
    /// 对应 Python v1 `CloudflareAIAdapter._parse_chunk`。
    fn parse_chunk(value: &Value, fallback_model: &str) -> Option<ChatCompletionChunk> {
        let result = value.get("result").unwrap_or(&Value::Null);
        let content = if result.is_object() {
            result
                .get("response")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        } else {
            String::new()
        };

        // 空内容块跳过（与 Python 老版一致：无有效内容返回 None）
        if content.is_empty() {
            return None;
        }

        Some(ChatCompletionChunk {
            id: value
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| util::generate_id("chatcmpl")),
            object: "chat.completion.chunk".to_string(),
            created: util::current_timestamp(),
            model: fallback_model.to_string(),
            choices: vec![ChatCompletionDelta {
                index: 0,
                delta: DeltaMessage {
                    role: Some("assistant".to_string()),
                    content: Some(content),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        })
    }

    /// 解析 Cloudflare /models/search 响应 → Vec<ModelInfo>
    ///
    /// Cloudflare 响应：`{"result": {"models": [...]}}`，兼容 result 为 list 的情况。
    /// 对应 Python v1 `CloudflareAIAdapter.list_models`。
    fn parse_models(value: &Value, provider: &str) -> Vec<ModelInfo> {
        let result = value.get("result").unwrap_or(&Value::Null);
        let arr: Option<&Vec<Value>> = if result.is_object() {
            result.get("models").and_then(|v| v.as_array())
        } else {
            // result 直接是 list 的情况
            result.as_array()
        };

        match arr {
            Some(arr) => arr
                .iter()
                .map(|m| {
                    // Cloudflare 模型条目可能是字符串或对象 {"id": "...", "name": "..."}
                    let id = if let Some(s) = m.as_str() {
                        s.to_string()
                    } else {
                        m.get("id")
                            .or_else(|| m.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string()
                    };
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
impl Adapter for CloudflareAIAdapter {
    fn provider_type(&self) -> &str {
        "cloudflareai"
    }

    fn provider_name(&self) -> &str {
        "Cloudflare Workers AI"
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

    /// 文本对话（Cloudflare 特有协议：POST /v1/run/{model}）
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.ensure_capability(Capabilities::Chat)?;
        let cf_model = Self::format_model(&req.model);
        let body = Self::build_chat_body(&req, false);
        let value = self
            .post_authed_json(&format!("v1/run/{cf_model}"), &body)
            .await?;
        Self::parse_chat_completion(&value, &req.model)
    }

    /// 流式文本对话（Cloudflare 特有协议：POST /v1/run/{model} stream=true）
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.ensure_capability(Capabilities::ChatStream)?;
        let cf_model = Self::format_model(&req.model);
        let body = Self::build_chat_body(&req, true);
        let url = self.url(&format!("v1/run/{cf_model}"));

        let resp = self
            .http
            .inner()
            .post(&url)
            .bearer_auth(self.api_key().unwrap_or(""))
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
        let byte_stream = resp
            .bytes_stream()
            .map_err(|e| e.to_string())
            .map(|r| r.map(|b| b.to_vec()));
        let lines_stream = CloudflareLinesStream::new(byte_stream);

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
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }
                let data = if let Some(rest) = line.strip_prefix("data: ") {
                    rest
                } else if let Some(rest) = line.strip_prefix("data:") {
                    rest
                } else {
                    continue;
                };
                if data.trim() == "[DONE]" {
                    return;
                }
                match serde_json::from_str::<Value>(data) {
                    Ok(v) => match Self::parse_chunk(&v, &model) {
                        Some(chunk) => yield Ok(chunk),
                        None => continue,
                    },
                    Err(_) => continue,
                }
            }
        };

        Ok(stream.boxed())
    }

    /// 文本嵌入（委托给 OpenAiCompatAdapter，标准 /embeddings 端点）
    ///
    /// Cloudflare Workers AI 同样提供标准 OpenAI 兼容 embeddings 端点，
    /// 复用地基实现。Python v1 声明了 EMBEDDING 能力但未实现 embed（默认返不支持），
    /// 此处补齐为标准 OpenAI 兼容实现。
    async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        self.compat.embed(req).await
    }

    /// 模型列表（Cloudflare 特有：GET /models/search）
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let value = self.get_authed_json("models/search").await?;
        let models = Self::parse_models(&value, self.provider_type());
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }
}

/// 解析 usage 统计
///
/// 与 `openai_compat::parse_usage` 同结构，本模块独立持有副本（原函数为模块私有）。
fn parse_usage(v: &Value) -> Option<crate::model::chat::ChatUsage> {
    let prompt = v.get("prompt_tokens").and_then(|x| x.as_u64())?;
    let completion = v
        .get("completion_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(0);
    let total = v
        .get("total_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(prompt + completion);
    Some(crate::model::chat::ChatUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: total,
    })
}

// ==================== Cloudflare SSE 行流适配器 ====================

/// 将字节流按行切分的适配器（Cloudflare 流式用）
///
/// 与 `openai_compat::LinesStream` 等价，独立实现避免引用其私有结构。
struct CloudflareLinesStream<S> {
    inner: S,
    buffer: Vec<u8>,
}

impl<S> CloudflareLinesStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
        }
    }
}

impl<S> futures::Stream for CloudflareLinesStream<S>
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
            if let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
                let mut line: Vec<u8> = self.buffer.drain(..=pos).collect();
                if line.last() == Some(&b'\n') {
                    line.pop();
                }
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                let s = String::from_utf8_lossy(&line).into_owned();
                return Poll::Ready(Some(Ok(s)));
            }
            match std::pin::Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Err(msg))) => return Poll::Ready(Some(Err(msg))),
                Poll::Ready(Some(Ok(chunk))) => {
                    self.buffer.extend_from_slice(&chunk);
                }
                Poll::Ready(None) => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientOptions;
    use crate::http::HttpClient;
    use crate::model::chat::{ChatMessage, ChatRequest};
    use crate::model::image::ImageRequest;
    use crate::model::options::{EmbedInput, EmbeddingVector};
    use crate::model::video::VideoRequest;
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

    /// 构造测试用 SiliconFlowAdapter（指向 mockito server）
    fn make_siliconflow(server: &Server) -> SiliconFlowAdapter {
        let compat = make_compat(
            server,
            "siliconflow",
            "SiliconFlow 硅基流动",
            siliconflow_capabilities(),
        );
        SiliconFlowAdapter::with_compat(compat)
    }

    /// 构造测试用 TogetherAIAdapter（指向 mockito server）
    fn make_togetherai(server: &Server) -> TogetherAIAdapter {
        let compat = make_compat(
            server,
            "togetherai",
            "Together AI",
            togetherai_capabilities(),
        );
        TogetherAIAdapter::with_compat(compat)
    }

    /// 构造测试用 FireworksAIAdapter（指向 mockito server）
    fn make_fireworksai(server: &Server) -> FireworksAIAdapter {
        let compat = make_compat(
            server,
            "fireworksai",
            "Fireworks AI",
            fireworksai_capabilities(),
        );
        FireworksAIAdapter::with_compat(compat)
    }

    /// 构造测试用 CloudflareAIAdapter（指向 mockito server）
    ///
    /// 需注入 account_id 到 config.extra，并让 compat/chat http 的 base_url 都指向
    /// `{server}/accounts/test-account/ai`（模拟 Cloudflare ai_base）。
    /// 注意：config.base_url 必须设为 ai_base，因为 OpenAiCompatAdapter::with_http
    /// 用 config.base_url 作为其 base_url（而非传入 http 的 base_url）。
    fn make_cloudflare(server: &Server) -> CloudflareAIAdapter {
        let ai_base = format!(
            "{}/accounts/test-account/ai",
            server.url().trim_end_matches('/')
        );
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(ai_base.clone())
            .timeout(5)
            .extra("account_id", json!("test-account"))
            .build();
        let config = ProviderConfig::from_options("cloudflareai", opts);
        let compat_http =
            HttpClient::new(&ClientOptions::builder().base_url(&ai_base).build()).unwrap();
        let compat = OpenAiCompatAdapter::with_http(
            compat_http,
            config.clone(),
            "cloudflareai",
            "Cloudflare Workers AI",
            cloudflareai_capabilities(),
        );
        let chat_http =
            HttpClient::new(&ClientOptions::builder().base_url(&ai_base).build()).unwrap();
        CloudflareAIAdapter::with_compat(compat, chat_http, "test-account", config)
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

    // ==================== SiliconFlow 测试 ====================

    #[tokio::test]
    async fn siliconflow_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_siliconflow(&server);
        assert_eq!(adapter.provider_type(), "siliconflow");
        assert_eq!(adapter.provider_name(), "SiliconFlow 硅基流动");
    }

    #[tokio::test]
    async fn siliconflow_requires_api_key() {
        let server = Server::new_async().await;
        let adapter = make_siliconflow(&server);
        assert!(adapter.requires_api_key());
    }

    #[tokio::test]
    async fn siliconflow_capabilities_include_chat_embed_reasoning() {
        let server = Server::new_async().await;
        let adapter = make_siliconflow(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::Reasoning));
        assert!(caps.contains(&Capabilities::Embedding));
        assert!(caps.contains(&Capabilities::JsonMode));
    }

    #[tokio::test]
    async fn siliconflow_chat_success() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_siliconflow(&server);
        let req = ChatRequest::builder("Qwen/Qwen2.5-7B-Instruct", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .build();
        let resp = adapter.chat(req).await.expect("chat 应成功");
        assert_eq!(resp.id, "chatcmpl-1");
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn siliconflow_chat_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(json!({"error": {"message": "invalid key"}}).to_string())
            .create_async()
            .await;
        let adapter = make_siliconflow(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn siliconflow_embed_success() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_body(
                json!({
                    "object": "list",
                    "data": [{"object":"embedding","index":0,"embedding":[0.1,0.2]}],
                    "model": "BAAI/bge-large-zh-v1.5",
                    "usage": {"prompt_tokens": 2, "total_tokens": 2}
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_siliconflow(&server);
        let req = EmbedRequest {
            model: "BAAI/bge-large-zh-v1.5".into(),
            input: EmbedInput::Single("hello".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let resp = adapter.embed(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
        if let EmbeddingVector::Float(v) = &resp.data[0].embedding {
            assert_eq!(v, &vec![0.1, 0.2]);
        } else {
            panic!("应为 Float 向量");
        }
    }

    #[tokio::test]
    async fn siliconflow_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "Qwen/Qwen2.5-7B-Instruct", "object": "model"},
                        {"id": "BAAI/bge-large-zh-v1.5", "object": "model"}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_siliconflow(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(models[0].provider, "siliconflow");
    }

    #[tokio::test]
    async fn siliconflow_list_models_filter_by_type() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "Qwen/Qwen2.5-7B-Instruct", "object": "model"},
                        {"id": "dall-e-3", "object": "model"}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_siliconflow(&server);
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "dall-e-3");
    }

    #[tokio::test]
    async fn siliconflow_list_models_error_429() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;
        let adapter = make_siliconflow(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    // ==================== Together AI 测试 ====================

    #[tokio::test]
    async fn togetherai_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_togetherai(&server);
        assert_eq!(adapter.provider_type(), "togetherai");
        assert_eq!(adapter.provider_name(), "Together AI");
    }

    #[tokio::test]
    async fn togetherai_chat_success() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "meta-llama/Llama-3-70B-chat-hf",
                "temperature": 0.5
            })))
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_togetherai(&server);
        let req = ChatRequest::builder(
            "meta-llama/Llama-3-70B-chat-hf",
            vec![ChatMessage::user("hi")],
        )
        .temperature(0.5)
        .max_tokens(100)
        .build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn togetherai_chat_error_429() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(429)
            .with_body(json!({"error": {"message": "rate limit"}}).to_string())
            .create_async()
            .await;
        let adapter = make_togetherai(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn togetherai_embed_success() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_body(
                json!({
                    "data": [{"object":"embedding","index":0,"embedding":[0.9]}],
                    "model": "BAAI/bge-base-en-v1.5"
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_togetherai(&server);
        let req = EmbedRequest {
            model: "BAAI/bge-base-en-v1.5".into(),
            input: EmbedInput::Single("text".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let resp = adapter.embed(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
    }

    #[tokio::test]
    async fn togetherai_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "meta-llama/Llama-3-70B-chat-hf"},
                        {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1"}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_togetherai(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].provider, "togetherai");
    }

    // ==================== Fireworks AI 测试 ====================

    #[tokio::test]
    async fn fireworksai_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_fireworksai(&server);
        assert_eq!(adapter.provider_type(), "fireworksai");
        assert_eq!(adapter.provider_name(), "Fireworks AI");
    }

    #[tokio::test]
    async fn fireworksai_capabilities_include_tool_call() {
        let server = Server::new_async().await;
        let adapter = make_fireworksai(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::ToolCall));
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::Embedding));
        assert!(caps.contains(&Capabilities::AudioTranscribe));
    }

    #[tokio::test]
    async fn fireworksai_chat_success() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "accounts/fireworks/models/llama-v3-70b-instruct"
            })))
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_fireworksai(&server);
        let req = ChatRequest::builder(
            "accounts/fireworks/models/llama-v3-70b-instruct",
            vec![ChatMessage::user("hi")],
        )
        .build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn fireworksai_chat_error_404() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(404)
            .with_body(json!({"error": {"message": "model not found"}}).to_string())
            .create_async()
            .await;
        let adapter = make_fireworksai(&server);
        let req = ChatRequest::builder("bad-model", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn fireworksai_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [{"id": "accounts/fireworks/models/llama-v3-70b-instruct"}]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_fireworksai(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].provider, "fireworksai");
    }

    #[tokio::test]
    async fn fireworksai_embed_success() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_body(
                json!({
                    "data": [{"object":"embedding","index":0,"embedding":[1.0,2.0]}],
                    "model": "nomic-ai/nomic-embed-text-v1.5"
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_fireworksai(&server);
        let req = EmbedRequest {
            model: "nomic-ai/nomic-embed-text-v1.5".into(),
            input: EmbedInput::Single("text".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let resp = adapter.embed(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
    }

    // ==================== Cloudflare 测试 ====================

    #[tokio::test]
    async fn cloudflare_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_cloudflare(&server);
        assert_eq!(adapter.provider_type(), "cloudflareai");
        assert_eq!(adapter.provider_name(), "Cloudflare Workers AI");
    }

    #[tokio::test]
    async fn cloudflare_requires_api_key() {
        let server = Server::new_async().await;
        let adapter = make_cloudflare(&server);
        assert!(adapter.requires_api_key());
    }

    #[tokio::test]
    async fn cloudflare_account_id_injected() {
        let server = Server::new_async().await;
        let adapter = make_cloudflare(&server);
        assert_eq!(adapter.account_id(), "test-account");
    }

    #[tokio::test]
    async fn cloudflare_new_without_account_id_returns_validation_error() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://example.com")
            .build();
        let config = ProviderConfig::from_options("cloudflareai", opts);
        let result = CloudflareAIAdapter::new(config);
        assert!(matches!(result, Err(AibridgeError::Validation { .. })));
    }

    #[tokio::test]
    async fn cloudflare_format_model_adds_cf_prefix() {
        assert_eq!(
            CloudflareAIAdapter::format_model("llama-3-8b"),
            "@cf/llama-3-8b"
        );
        assert_eq!(
            CloudflareAIAdapter::format_model("@cf/llama-3-8b"),
            "@cf/llama-3-8b"
        );
    }

    #[tokio::test]
    async fn cloudflare_chat_success() {
        let mut server = Server::new_async().await;
        // Cloudflare 端点：/accounts/test-account/ai/v1/run/@cf/meta/llama-3-8b-instruct
        let mock = server
            .mock(
                "POST",
                "/accounts/test-account/ai/v1/run/@cf/meta/llama-3-8b-instruct",
            )
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(
                json!({
                    "result": {"response": "Hi from Cloudflare!"},
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_cloudflare(&server);
        let req = ChatRequest::builder(
            "@cf/meta/llama-3-8b-instruct",
            vec![ChatMessage::user("hi")],
        )
        .temperature(0.5)
        .build();
        let resp = adapter.chat(req).await.expect("chat 应成功");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("Hi from Cloudflare!")
        );
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 7);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn cloudflare_chat_auto_prefixes_cf() {
        // 不带 @cf/ 前缀的模型名应被自动补全
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/accounts/test-account/ai/v1/run/@cf/llama-3-8b-instruct",
            )
            .with_status(200)
            .with_body(json!({"result": {"response": "ok"}}).to_string())
            .create_async()
            .await;
        let adapter = make_cloudflare(&server);
        let req =
            ChatRequest::builder("llama-3-8b-instruct", vec![ChatMessage::user("hi")]).build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("ok"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn cloudflare_chat_body_excludes_model_and_keeps_params() {
        // 验证请求体不含 model 字段，但保留 messages / temperature
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/accounts/test-account/ai/v1/run/@cf/test-model")
            .match_body(mockito::Matcher::PartialJson(json!({
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7
            })))
            .with_status(200)
            .with_body(json!({"result": {"response": "ok"}}).to_string())
            .create_async()
            .await;
        let adapter = make_cloudflare(&server);
        let req = ChatRequest::builder("test-model", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("ok"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn cloudflare_chat_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/accounts/test-account/ai/v1/run/@cf/m")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad token"}}).to_string())
            .create_async()
            .await;
        let adapter = make_cloudflare(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn cloudflare_chat_error_429() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/accounts/test-account/ai/v1/run/@cf/m")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow"}}).to_string())
            .create_async()
            .await;
        let adapter = make_cloudflare(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn cloudflare_chat_stream_success() {
        let mut server = Server::new_async().await;
        // SSE 流：两个增量块 + [DONE]
        let sse = "data: {\"result\":{\"response\":\"Hello\"}}\n\
                   data: {\"result\":{\"response\":\" world\"}}\n\
                   data: [DONE]\n";
        let mock = server
            .mock("POST", "/accounts/test-account/ai/v1/run/@cf/m")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let adapter = make_cloudflare(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let stream = adapter.chat_stream(req).await.expect("流应建立");
        let chunks: Vec<_> = stream.collect().await;
        let contents: Vec<String> = chunks
            .into_iter()
            .filter_map(|c| c.ok())
            .filter_map(|c| c.choices.into_iter().next())
            .filter_map(|d| d.delta.content)
            .collect();
        assert_eq!(contents, vec!["Hello", " world"]);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn cloudflare_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/accounts/test-account/ai/models/search")
            .with_status(200)
            .with_body(
                json!({
                    "result": {
                        "models": [
                            {"id": "@cf/meta/llama-3-8b-instruct", "description": "Llama 3 8B"},
                            {"id": "@cf/baai/bge-base-en-v1.5", "description": "Embedding"}
                        ]
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_cloudflare(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "@cf/meta/llama-3-8b-instruct");
        assert_eq!(models[0].provider, "cloudflareai");
        assert_eq!(models[0].description.as_deref(), Some("Llama 3 8B"));
    }

    #[tokio::test]
    async fn cloudflare_list_models_result_as_list() {
        // 兼容 result 直接为 list 的情况
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/accounts/test-account/ai/models/search")
            .with_status(200)
            .with_body(
                json!({
                    "result": [{"id": "@cf/m1"}, "@cf/m2"]
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_cloudflare(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "@cf/m1");
        assert_eq!(models[1].id, "@cf/m2");
    }

    #[tokio::test]
    async fn cloudflare_list_models_filter_by_type() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/accounts/test-account/ai/models/search")
            .with_status(200)
            .with_body(
                json!({
                    "result": {
                        "models": [
                            {"id": "@cf/meta/llama-3-8b-instruct"},
                            {"id": "@cf/stabilityai/stable-diffusion-xl-base-1.0"}
                        ]
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_cloudflare(&server);
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "@cf/stabilityai/stable-diffusion-xl-base-1.0");
    }

    #[tokio::test]
    async fn cloudflare_list_models_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/accounts/test-account/ai/models/search")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad token"}}).to_string())
            .create_async()
            .await;
        let adapter = make_cloudflare(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn cloudflare_embed_success() {
        // embed 委托 compat（标准 /embeddings 端点）
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/accounts/test-account/ai/embeddings")
            .with_status(200)
            .with_body(
                json!({
                    "data": [{"object":"embedding","index":0,"embedding":[0.5,0.6]}],
                    "model": "@cf/baai/bge-base-en-v1.5"
                })
                .to_string(),
            )
            .create_async()
            .await;
        let adapter = make_cloudflare(&server);
        let req = EmbedRequest {
            model: "@cf/baai/bge-base-en-v1.5".into(),
            input: EmbedInput::Single("text".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let resp = adapter.embed(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
        if let EmbeddingVector::Float(v) = &resp.data[0].embedding {
            assert_eq!(v, &vec![0.5, 0.6]);
        } else {
            panic!("应为 Float 向量");
        }
    }

    #[tokio::test]
    async fn cloudflare_embed_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/accounts/test-account/ai/embeddings")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad token"}}).to_string())
            .create_async()
            .await;
        let adapter = make_cloudflare(&server);
        let req = EmbedRequest {
            model: "@cf/baai/bge-base-en-v1.5".into(),
            input: EmbedInput::Single("text".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let err = adapter.embed(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    // ==================== 能力校验（不支持的能力返回错误） ====================

    #[tokio::test]
    async fn cloudflare_image_generate_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_cloudflare(&server);
        let req = ImageRequest::builder("m", "prompt").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn siliconflow_video_create_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_siliconflow(&server);
        let req = VideoRequest::builder("m", "p").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ==================== 默认 Base URL 常量校验 ====================

    #[test]
    fn default_base_urls_match_python() {
        assert_eq!(
            DEFAULT_SILICONFLOW_BASE_URL,
            "https://api.siliconflow.cn/v1"
        );
        assert_eq!(DEFAULT_TOGETHERAI_BASE_URL, "https://api.together.xyz/v1");
        assert_eq!(
            DEFAULT_FIREWORKSAI_BASE_URL,
            "https://api.fireworks.ai/inference/v1"
        );
        assert_eq!(
            DEFAULT_CLOUDFLAREAI_BASE_URL,
            "https://api.cloudflare.com/client/v4"
        );
    }

    // ==================== start/close 无副作用 ====================

    #[tokio::test]
    async fn siliconflow_start_close_are_noops() {
        let server = Server::new_async().await;
        let mut adapter = make_siliconflow(&server);
        adapter.start().await.unwrap();
        adapter.close().await.unwrap();
    }

    #[tokio::test]
    async fn cloudflare_start_close_are_noops() {
        let server = Server::new_async().await;
        let mut adapter = make_cloudflare(&server);
        adapter.start().await.unwrap();
        adapter.close().await.unwrap();
    }
}
