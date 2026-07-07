//! OpenAI 兼容协议适配器地基
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/openai.py` 与各 OpenAI 兼容适配器
//! 的公共部分。14 个适配器中约 80% 是 OpenAI 兼容协议（agnes/openai/azure/
//! 聚合平台/中文平台兼容部分），本模块抽出通用 HTTP 请求构造 + 响应解析 +
//! 参数映射 + 错误映射逻辑，子适配器（openai/agnes）只需 override base_url、
//! provider_type、特有参数等差异，复用本模块的 chat/image/embed/list_models 实现。
//!
//! 设计要点（与设计文档第 10 节一致）：
//! - `OpenAiCompatAdapter` 持有 `HttpClient`、`ProviderConfig`、provider_type 等
//! - 实现 `Adapter` trait 的核心方法：chat / chat_stream / image_generate / embed / list_models
//! - 预置 `OPENAI_COMPATIBLE_MAPPING` 通用参数映射（用 `ParameterMapping` 机制）
//! - 错误映射：OpenAI API 错误（HTTP status + error.message）→ AibridgeError
//!   （401→Authentication，429→RateLimit，404→ModelNotFound，5xx→Api 等）
//!
//! 注意：本模块为地基，不注册到工厂；具体子适配器（openai/agnes）在阶段 1.1/1.2 实现。

use futures::stream::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::adapter::{Capabilities, CapabilitySet, ChatStream};
use crate::config::ProviderConfig;
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::chat::{
    ChatChoice, ChatCompletion, ChatCompletionChunk, ChatCompletionDelta, ChatRequest,
    ChoiceMessage, DeltaMessage,
};
use crate::model::common::{infer_model_type, ModelInfo, ModelType};
use crate::model::image::{ImageData, ImageRequest, ImageResult};
use crate::model::options::{
    EmbedRequest, EmbeddingItem, EmbeddingResult, EmbeddingUsage, EmbeddingVector, ParameterMapping,
};
use crate::util;

/// OpenAI 官方默认 Base URL
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI 兼容协议的通用参数映射
///
/// 对应 Python v1 `OPENAI_COMPATIBLE_MAPPING`。
/// OpenAI 协议本身即"通用参数名"，因此映射表为空（保持原名透传）。
/// 子适配器可在此基础上叠加自己的 rename_map。
///
/// 保留为常量是为了与设计文档 10.2 节"预置常量 OPENAI_COMPATIBLE_MAPPING"对齐，
/// 后续若发现部分兼容平台需要重命名（如 max_tokens → maxOutputTokens），可在此扩展。
pub fn openai_compatible_mapping() -> ParameterMapping {
    ParameterMapping::new()
}

/// OpenAI 兼容适配器地基
///
/// 持有 HTTP 客户端与 Provider 配置，实现 OpenAI 兼容协议的核心能力。
/// 子适配器（openai/agnes）通过组合（而非继承）复用本结构的方法：
/// 子适配器内部持有一个 `OpenAiCompatAdapter`，并把自身的 base_url、provider_type
/// 等差异传入；本结构的方法均为 `pub`，子适配器可直接调用。
///
/// Rust 无继承，故采用"地基结构 + trait 方法转发"模式：
/// 子适配器实现 `Adapter` trait 时，将 chat/image/embed/list_models 等委托给
/// 内部的 `OpenAiCompatAdapter` 实例。
pub struct OpenAiCompatAdapter {
    /// HTTP 客户端（封装 reqwest，含连接池与超时）
    http: HttpClient,
    /// Provider 配置（api_key / base_url / timeout 等）
    config: ProviderConfig,
    /// Provider 类型标识（如 "openai"、"agnes"），用于能力错误信息
    provider_type: String,
    /// Provider 显示名称（如 "OpenAI"）
    provider_name: String,
    /// 实际 base_url（已合并 config.base_url 与默认值）
    base_url: String,
    /// 支持的能力集合
    capabilities: CapabilitySet,
}

impl OpenAiCompatAdapter {
    /// 创建 OpenAI 兼容适配器
    ///
    /// - `config`：Provider 配置，base_url 为 None 时用 `default_base_url` 兜底
    /// - `provider_type`：Provider 类型标识
    /// - `provider_name`：Provider 显示名称
    /// - `default_base_url`：config.base_url 为空时的兜底 base_url
    /// - `capabilities`：支持的能力集合
    pub fn new(
        config: ProviderConfig,
        provider_type: impl Into<String>,
        provider_name: impl Into<String>,
        default_base_url: &str,
        capabilities: CapabilitySet,
    ) -> Result<Self> {
        let provider_type = provider_type.into();
        let provider_name = provider_name.into();
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| default_base_url.to_string());

        // 构造 HttpClient：把 base_url 透传，便于 post_json 等方法自动拼接相对路径
        let opts = crate::config::ClientOptions::builder()
            .api_key(config.api_key.clone().unwrap_or_default())
            .base_url(base_url.clone())
            .timeout(config.timeout)
            .max_retries(config.max_retries)
            .retry_delay(config.retry_delay)
            .build();
        let http = HttpClient::new(&opts)?;

        Ok(Self {
            http,
            config,
            provider_type,
            provider_name,
            base_url,
            capabilities,
        })
    }

    /// 用显式的 HttpClient 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_http(
        http: HttpClient,
        config: ProviderConfig,
        provider_type: impl Into<String>,
        provider_name: impl Into<String>,
        capabilities: CapabilitySet,
    ) -> Self {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_string());
        Self {
            http,
            config,
            provider_type: provider_type.into(),
            provider_name: provider_name.into(),
            base_url,
            capabilities,
        }
    }

    /// Provider 类型标识
    pub fn provider_type_str(&self) -> &str {
        &self.provider_type
    }

    /// Provider 显示名称
    pub fn provider_name_str(&self) -> &str {
        &self.provider_name
    }

    /// base_url
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// API key（可能为空，免费 provider 场景）
    pub fn api_key(&self) -> Option<&str> {
        self.config.api_key.as_deref()
    }

    /// 支持的能力集合
    pub fn capabilities_set(&self) -> &CapabilitySet {
        &self.capabilities
    }

    /// 拼接完整 URL（base_url + 相对路径）
    fn url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        let path = path.trim_start_matches('/');
        format!("{base}/{path}")
    }

    /// 校验请求的能力是否被支持（不支持则返 UnsupportedCapability）
    fn ensure_capability(&self, cap: Capabilities) -> Result<()> {
        if self.capabilities.contains(&cap) {
            Ok(())
        } else {
            Err(AibridgeError::UnsupportedCapability {
                capability: format!("{} (provider: {})", cap.as_str(), self.provider_type),
            })
        }
    }

    /// 发送带认证的 POST JSON 请求，并用 OpenAI 错误映射处理响应
    ///
    /// 不依赖 HttpClient 的自动错误处理（其 from_http_status 不提取 OpenAI
    /// error.message / retry_after），而是用 map_api_error 统一映射。
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
            return Err(Self::map_api_error(status_code, &body_text));
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
            return Err(Self::map_api_error(status_code, &body_text));
        }
        resp.json::<Value>().await.map_err(AibridgeError::from)
    }

    /// 构造 OpenAI chat/completions 请求体
    ///
    /// 将统一 `ChatRequest` 转为 OpenAI 协议的 JSON 请求体。
    /// 通用参数走 `OPENAI_COMPATIBLE_MAPPING`（当前为透传），
    /// provider 特有参数走 `extra` 透传。
    ///
    /// 对子适配器开放（pub）：Azure 等子适配器复用请求体构造，
    /// 仅 override HTTP path/header（见设计文档 10.1 节）。
    pub fn build_chat_body(&self, req: &ChatRequest, stream: bool) -> Value {
        // 序列化统一请求，得到基础字段
        let mut body = serde_json::to_value(req).unwrap_or_else(|_| json!({}));
        // 强制覆盖 stream 标志（统一请求的 stream 字段默认 false，流式调用时需置 true）
        if stream {
            body["stream"] = json!(true);
            // 流式场景附带 stream_options.include_usage，便于末尾拿到 usage 统计
            body["stream_options"] = json!({ "include_usage": true });
        } else if body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            // 非流式调用但请求体带了 stream=true，移除避免歧义
            body["stream"] = json!(false);
        }
        // extra 透传：合并到顶层（extra 字段本身不发送）
        if let Some(obj) = body.as_object_mut() {
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

    /// 文本对话（非流式）
    ///
    /// POST /chat/completions，构造 OpenAI 请求体，解析响应 → ChatCompletion。
    pub async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.ensure_capability(Capabilities::Chat)?;
        let body = self.build_chat_body(&req, false);
        let value = self.post_authed_json("chat/completions", &body).await?;
        self.parse_chat_completion(&value, &req.model)
    }

    /// 流式文本对话
    ///
    /// POST /chat/completions stream=true，解析 SSE 流 → ChatStream。
    /// SSE 格式：每行 `data: <json>`，`data: [DONE]` 标记结束。
    pub async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.ensure_capability(Capabilities::ChatStream)?;
        let body = self.build_chat_body(&req, true);
        let url = self.url("chat/completions");

        // 流式请求需直接用 reqwest::Client（HttpClient 未暴露流式接口）
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
            return Err(Self::map_api_error(status_code, &body_text));
        }

        let model = req.model.clone();
        // 按字节流读取，按行切分解析 SSE
        // 把 reqwest::Error 统一转成 String，便于 LinesStream 跨错误类型复用
        let byte_stream = resp
            .bytes_stream()
            .map_err(|e| e.to_string())
            .map(|r| r.map(|b| b.to_vec()));
        let lines_stream = LinesStream::new(byte_stream);

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
                // 空行或注释行（以 ":" 开头的心跳）跳过
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }
                // 去除 "data: " 前缀
                let data = if let Some(rest) = line.strip_prefix("data: ") {
                    rest
                } else if let Some(rest) = line.strip_prefix("data:") {
                    rest
                } else {
                    // 非 data 行，跳过
                    continue;
                };
                // 结束标记
                if data.trim() == "[DONE]" {
                    return;
                }
                // 解析 JSON
                match serde_json::from_str::<Value>(data) {
                    Ok(v) => {
                        match Self::parse_chunk(&v, &model) {
                            Ok(Some(chunk)) => yield Ok(chunk),
                            Ok(None) => continue,
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        }
                    }
                    Err(_) => {
                        // 单行 JSON 解析失败不致命，跳过（与 Python 老版一致）
                        continue;
                    }
                }
            }
            // 流自然结束（未收到 [DONE]）也视为正常
        };

        Ok(stream.boxed())
    }

    /// 图像生成
    ///
    /// POST /images/generations → ImageResult
    pub async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
        self.ensure_capability(Capabilities::ImageGenerate)?;
        let body = self.build_image_body(&req);
        let value = self.post_authed_json("images/generations", &body).await?;
        self.parse_image_result(&value, &req.model)
    }

    /// 文本嵌入
    ///
    /// POST /embeddings → EmbeddingResult
    pub async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        self.ensure_capability(Capabilities::Embedding)?;
        let body = self.build_embed_body(&req);
        let value = self.post_authed_json("embeddings", &body).await?;
        self.parse_embedding_result(&value, &req.model)
    }

    /// 模型列表（实时拉取）
    ///
    /// GET /models → Vec<ModelInfo>，按 `filter` 过滤模型类型。
    pub async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let value = self.get_authed_json("models").await?;
        let models = self.parse_models(&value);
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }

    // ==================== 内部：请求体构造 ====================

    /// 构造 OpenAI images/generations 请求体
    ///
    /// 对子适配器开放（pub）：Azure 等子适配器复用请求体构造。
    pub fn build_image_body(&self, req: &ImageRequest) -> Value {
        let mut body = serde_json::to_value(req).unwrap_or_else(|_| json!({}));
        // extra 透传
        if let Some(obj) = body.as_object_mut() {
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

    /// 构造 OpenAI embeddings 请求体
    ///
    /// 对子适配器开放（pub）：Azure 等子适配器复用请求体构造。
    pub fn build_embed_body(&self, req: &EmbedRequest) -> Value {
        let mut body = serde_json::to_value(req).unwrap_or_else(|_| json!({}));
        // extra 透传
        if let Some(obj) = body.as_object_mut() {
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

    // ==================== 内部：响应解析 ====================

    /// 解析 OpenAI chat/completions 响应 → ChatCompletion
    ///
    /// 对子适配器开放（pub）：Azure 等子适配器复用响应解析。
    pub fn parse_chat_completion(
        &self,
        value: &Value,
        fallback_model: &str,
    ) -> Result<ChatCompletion> {
        let id = value
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| util::generate_id("chatcmpl"));
        let created = value
            .get("created")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(util::current_timestamp);
        let model = value
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| fallback_model.to_string());
        let object = value
            .get("object")
            .and_then(|v| v.as_str())
            .unwrap_or("chat.completion")
            .to_string();
        let service_tier = value
            .get("service_tier")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let system_fingerprint = value
            .get("system_fingerprint")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let usage = value.get("usage").and_then(parse_usage);

        let choices = value
            .get("choices")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .enumerate()
                    .map(|(i, c)| parse_choice(c, i))
                    .collect()
            })
            .unwrap_or_default();

        Ok(ChatCompletion {
            id,
            object,
            created,
            model,
            choices,
            usage,
            service_tier,
            system_fingerprint,
        })
    }

    /// 解析单个 SSE chunk（OpenAI 流式格式）→ Option<ChatCompletionChunk>
    ///
    /// 返回 None 表示该 chunk 无有效 choices（如纯 usage 块），调用方跳过。
    ///
    /// 对子适配器开放（pub）：Azure 等子适配器复用流式 chunk 解析。
    pub fn parse_chunk(value: &Value, fallback_model: &str) -> Result<Option<ChatCompletionChunk>> {
        let id = value
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| util::generate_id("chatcmpl"));
        let created = value
            .get("created")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(util::current_timestamp);
        let model = value
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| fallback_model.to_string());
        let object = value
            .get("object")
            .and_then(|v| v.as_str())
            .unwrap_or("chat.completion.chunk")
            .to_string();
        let usage = value.get("usage").and_then(parse_usage);

        let choices: Vec<ChatCompletionDelta> = value
            .get("choices")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .enumerate()
                    .map(|(i, c)| parse_delta(c, i))
                    .collect()
            })
            .unwrap_or_default();

        // 无 choices 且无 usage 的空块跳过
        if choices.is_empty() && usage.is_none() {
            return Ok(None);
        }
        Ok(Some(ChatCompletionChunk {
            id,
            object,
            created,
            model,
            choices,
            usage,
        }))
    }

    /// 解析 OpenAI images/generations 响应 → ImageResult
    ///
    /// 对子适配器开放（pub）：Azure 等子适配器复用响应解析。
    pub fn parse_image_result(&self, value: &Value, fallback_model: &str) -> Result<ImageResult> {
        let id = value
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| util::generate_id("img"));
        let created = value
            .get("created")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(util::current_timestamp);
        let model = value
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| fallback_model.to_string());
        let object = value
            .get("object")
            .and_then(|v| v.as_str())
            .unwrap_or("image.generation")
            .to_string();
        let data = value
            .get("data")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(parse_image_data).collect())
            .unwrap_or_default();
        Ok(ImageResult {
            id,
            object,
            created,
            model,
            data,
        })
    }

    /// 解析 OpenAI embeddings 响应 → EmbeddingResult
    ///
    /// 对子适配器开放（pub）：Azure 等子适配器复用响应解析。
    pub fn parse_embedding_result(
        &self,
        value: &Value,
        fallback_model: &str,
    ) -> Result<EmbeddingResult> {
        let object = value
            .get("object")
            .and_then(|v| v.as_str())
            .unwrap_or("list")
            .to_string();
        let model = value
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| fallback_model.to_string());
        let data = value
            .get("data")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(parse_embedding_item).collect())
            .unwrap_or_default();
        let usage = value.get("usage").and_then(parse_embed_usage);
        Ok(EmbeddingResult {
            object,
            data,
            model,
            usage,
        })
    }

    /// 解析 OpenAI /models 响应 → Vec<ModelInfo>
    ///
    /// OpenAI 返回 `{"data": [{"id": "...", "created": ..., "owned_by": "..."}]}`
    fn parse_models(&self, value: &Value) -> Vec<ModelInfo> {
        let arr = value.get("data").and_then(|v| v.as_array());
        match arr {
            Some(arr) => arr
                .iter()
                .map(|m| {
                    let id = m
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let model_type = infer_model_type(&id);
                    ModelInfo {
                        name: id.clone(),
                        id,
                        model_type,
                        provider: self.provider_type.clone(),
                        capabilities: Vec::new(),
                        max_tokens: None,
                        supports_streaming: matches!(model_type, ModelType::Chat),
                        description: None,
                        created: m.get("created").and_then(|v| v.as_u64()),
                    }
                })
                .collect(),
            None => Vec::new(),
        }
    }

    // ==================== 内部：错误映射 ====================

    /// 将 OpenAI API 错误响应映射为 AibridgeError
    ///
    /// 优先提取 OpenAI 错误体 `{"error": {"message": "..."}}` 中的 message，
    /// 再按 HTTP 状态码分类：
    /// - 401/403 → Authentication
    /// - 429 → RateLimit（尝试从 Retry-After / 错误体提取 retry_after）
    /// - 404 → ModelNotFound
    /// - 400 → Validation
    /// - 4xx（其他）→ Api
    /// - 5xx → Api
    pub fn map_api_error(status: u16, body: &str) -> AibridgeError {
        // 尝试解析 OpenAI 错误结构 {"error": {"message": "...", "type": "..."}}
        let message = parse_error_message(body, status);
        match status {
            401 | 403 => AibridgeError::Authentication { message },
            429 => {
                let retry_after = parse_retry_after(body);
                AibridgeError::RateLimit {
                    message,
                    retry_after,
                }
            }
            400 => AibridgeError::Validation {
                message,
                details: serde_json::json!({ "status_code": status, "response": body }),
            },
            404 => AibridgeError::ModelNotFound { model: message },
            s if (400..500).contains(&s) => AibridgeError::Api { status: s, message },
            s if (500..600).contains(&s) => AibridgeError::Api { status: s, message },
            s => AibridgeError::Api { status: s, message },
        }
    }
}

/// 解析 OpenAI 错误体中的 message 字段
///
/// OpenAI 错误格式：`{"error": {"message": "...", "type": "...", "code": "..."}}`
/// 解析失败时回退到 `HTTP {status}` 字符串。
fn parse_error_message(body: &str, status: u16) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        if let Some(msg) = v
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return msg.to_string();
        }
        // 部分兼容平台直接用顶层 message
        if let Some(msg) = v.get("message").and_then(|m| m.as_str()) {
            return msg.to_string();
        }
    }
    if body.trim().is_empty() {
        format!("HTTP {status}")
    } else {
        format!("HTTP {status}: {body}")
    }
}

/// 从错误体尝试解析 retry_after（秒）
///
/// OpenAI 限流响应偶尔携带 `error.retry_after` 或顶层 `retry_after`。
fn parse_retry_after(body: &str) -> Option<f64> {
    let v = serde_json::from_str::<Value>(body).ok()?;
    v.get("error")
        .and_then(|e| e.get("retry_after"))
        .and_then(|r| r.as_f64())
        .or_else(|| v.get("retry_after").and_then(|r| r.as_f64()))
}

/// 解析 usage 统计
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

/// 解析单个 choice → ChatChoice
fn parse_choice(c: &Value, index: usize) -> ChatChoice {
    let idx = c
        .get("index")
        .and_then(|v| v.as_u64())
        .map(|i| i as u32)
        .unwrap_or(index as u32);
    let finish_reason = c
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let message = c.get("message").cloned().unwrap_or(Value::Null);
    let role = message
        .get("role")
        .and_then(|v| v.as_str())
        .unwrap_or("assistant")
        .to_string();
    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let tool_calls = message
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().map(parse_tool_call).collect());
    ChatChoice {
        index: idx,
        message: ChoiceMessage {
            role,
            content,
            tool_calls,
        },
        finish_reason,
    }
}

/// 解析单个流式 delta → ChatCompletionDelta
fn parse_delta(c: &Value, index: usize) -> ChatCompletionDelta {
    let idx = c
        .get("index")
        .and_then(|v| v.as_u64())
        .map(|i| i as u32)
        .unwrap_or(index as u32);
    let finish_reason = c
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let delta = c.get("delta").cloned().unwrap_or(Value::Null);
    let role = delta
        .get("role")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let content = delta
        .get("content")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let tool_calls = delta
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().map(parse_tool_call).collect());
    ChatCompletionDelta {
        index: idx,
        delta: DeltaMessage {
            role,
            content,
            tool_calls,
        },
        finish_reason,
    }
}

/// 解析单个 ToolCall（OpenAI 格式）
fn parse_tool_call(v: &Value) -> crate::model::options::ToolCall {
    let id = v
        .get("id")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let tool_type = v
        .get("type")
        .and_then(|x| x.as_str())
        .unwrap_or("function")
        .to_string();
    let function = v.get("function").cloned().unwrap_or(Value::Null);
    let name = function
        .get("name")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let arguments = function
        .get("arguments")
        .and_then(|x| x.as_str())
        .unwrap_or("{}")
        .to_string();
    crate::model::options::ToolCall {
        id,
        tool_type,
        function: crate::model::options::ToolCallFunction { name, arguments },
    }
}

/// 解析单个 ImageData
fn parse_image_data(v: &Value) -> ImageData {
    ImageData {
        url: v.get("url").and_then(|x| x.as_str()).map(str::to_owned),
        b64_json: v
            .get("b64_json")
            .and_then(|x| x.as_str())
            .map(str::to_owned),
        revised_prompt: v
            .get("revised_prompt")
            .and_then(|x| x.as_str())
            .map(str::to_owned),
    }
}

/// 解析单个 EmbeddingItem
fn parse_embedding_item(v: &Value) -> EmbeddingItem {
    let index = v
        .get("index")
        .and_then(|x| x.as_u64())
        .map(|i| i as u32)
        .unwrap_or(0);
    let embedding = if let Some(arr) = v.get("embedding").and_then(|x| x.as_array()) {
        EmbeddingVector::Float(arr.iter().filter_map(|x| x.as_f64()).collect())
    } else if let Some(s) = v.get("embedding").and_then(|x| x.as_str()) {
        EmbeddingVector::Base64(s.to_string())
    } else {
        EmbeddingVector::Float(Vec::new())
    };
    EmbeddingItem {
        object: "embedding".to_string(),
        index,
        embedding,
    }
}

/// 解析嵌入 usage
fn parse_embed_usage(v: &Value) -> Option<EmbeddingUsage> {
    let prompt = v.get("prompt_tokens").and_then(|x| x.as_u64())?;
    let total = v
        .get("total_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(prompt);
    Some(EmbeddingUsage {
        prompt_tokens: prompt,
        total_tokens: total,
    })
}

// ==================== SSE 行流适配器 ====================

/// 将字节流按行切分的适配器
///
/// reqwest 的 `bytes_stream` 返回字节 chunk，需自行按 `\n` 切分。
/// 本结构维护一个未完成行的缓冲区，逐 chunk 拼接出完整行。
/// 泛型 `S` 须为 `Stream<Item = Result<Vec<u8>, String>>`（错误统一转 String 便于复用）。
struct LinesStream<S> {
    inner: S,
    buffer: Vec<u8>,
}

impl<S> LinesStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
        }
    }
}

impl<S> futures::Stream for LinesStream<S>
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

// ==================== 测试辅助序列化结构 ====================

/// OpenAI /models 响应的最小解析结构（测试用，验证反序列化对齐协议）
#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize)]
struct OpenAiModelsResponse {
    object: String,
    data: Vec<OpenAiModelEntry>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize)]
struct OpenAiModelEntry {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientOptions;
    use crate::model::chat::{ChatMessage, ChatRequest};
    use crate::model::image::ImageRequest;
    use crate::model::options::EmbedInput;
    use mockito::Server;
    use std::collections::HashMap;

    /// 构造测试用 OpenAiCompatAdapter（指向 mockito server）
    fn make_adapter(server: &Server, caps: CapabilitySet) -> OpenAiCompatAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("openai", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        OpenAiCompatAdapter::with_http(http, config, "openai", "OpenAI", caps)
    }

    /// 全能力集合（chat + chat_stream + image_generate + embedding）
    fn full_caps() -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::Chat);
        caps.insert(Capabilities::ChatStream);
        caps.insert(Capabilities::ImageGenerate);
        caps.insert(Capabilities::Embedding);
        caps.insert(Capabilities::Vision);
        caps.insert(Capabilities::ToolCall);
        caps
    }

    // ============ chat 正常路径 ============

    #[tokio::test]
    async fn chat_success_parses_completion() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        });
        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .max_tokens(100)
            .build();
        let resp = adapter.chat(req).await.expect("chat 应成功");

        assert_eq!(resp.id, "chatcmpl-1");
        assert_eq!(resp.model, "gpt-4o");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 7);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_sends_temperature_and_max_tokens() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "gpt-4o",
                "temperature": 0.5,
                "max_tokens": 50
            })))
            .with_status(200)
            .with_body(json!({
                "id": "x", "object": "chat.completion", "created": 1, "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role":"assistant","content":"ok"}, "finish_reason": "stop"}]
            }).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")])
            .temperature(0.5)
            .max_tokens(50)
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_passes_extra_params_through() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "gpt-4o",
                "custom_param": "custom_value"
            })))
            .with_status(200)
            .with_body(json!({
                "id": "x", "object": "chat.completion", "created": 1, "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role":"assistant","content":"ok"}, "finish_reason": "stop"}]
            }).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")])
            .extra("custom_param", "custom_value")
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_with_tool_calls_parses() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("weather?")]).build();
        let resp = adapter.chat(req).await.unwrap();
        let tool_calls = resp.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("应有 tool_calls");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, "{\"city\":\"Beijing\"}");
    }

    // ============ chat 错误路径 ============

    #[tokio::test]
    async fn chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(json!({"error": {"message": "Invalid API key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(429)
            .with_body(
                json!({"error": {"message": "Rate limit exceeded", "retry_after": 1.5}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::RateLimit { retry_after, .. } => {
                assert_eq!(retry_after, Some(1.5));
            }
            _ => panic!("应为 RateLimit"),
        }
    }

    #[tokio::test]
    async fn chat_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(404)
            .with_body(
                json!({"error": {"message": "The model 'gpt-x' does not exist"}}).to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-x", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn chat_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(500)
            .with_body(json!({"error": {"message": "Internal server error"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 500),
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn chat_error_400_returns_validation() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(400)
            .with_body(json!({"error": {"message": "max_tokens is invalid"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Validation { message, .. } => {
                assert!(message.contains("max_tokens"));
            }
            _ => panic!("应为 Validation"),
        }
    }

    #[tokio::test]
    async fn chat_unsupported_capability_returns_error() {
        // 不支持 Chat 能力
        let server = Server::new_async().await;
        let adapter = make_adapter(&server, CapabilitySet::new());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ chat_stream 正常 + 错误路径 ============

    #[tokio::test]
    async fn chat_stream_parses_sse_chunks() {
        let mut server = Server::new_async().await;
        // 构造 SSE 响应：3 个 data 行 + [DONE]
        let sse = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat/completions")
            .match_query(mockito::Matcher::Missing)
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.expect("stream 应建立");
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(chunks.len(), 3);
        // 第 1 块 role
        assert_eq!(
            chunks[0].choices[0].delta.role.as_deref(),
            Some("assistant")
        );
        // 拼接内容
        let mut content = String::new();
        content.push_str(chunks[1].choices[0].delta.content.as_deref().unwrap_or(""));
        content.push_str(chunks[2].choices[0].delta.content.as_deref().unwrap_or(""));
        assert_eq!(content, "Hello world");
        // 第 3 块 finish_reason
        assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn chat_stream_handles_heartbeat_and_empty_lines() {
        let mut server = Server::new_async().await;
        let sse = ": heartbeat\n\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(sse)
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("hi"));
    }

    #[tokio::test]
    async fn chat_stream_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(json!({"error": {"message": "Unauthorized"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let result = adapter.chat_stream(req).await;
        match result {
            Err(e) => assert!(matches!(e, AibridgeError::Authentication { .. })),
            Ok(_) => panic!("chat_stream 应返回错误而非 stream"),
        }
    }

    #[tokio::test]
    async fn chat_stream_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let result = adapter.chat_stream(req).await;
        match result {
            Err(e) => assert!(matches!(e, AibridgeError::RateLimit { .. })),
            Ok(_) => panic!("chat_stream 应返回错误而非 stream"),
        }
    }

    #[tokio::test]
    async fn chat_stream_sends_stream_true() {
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

        let adapter = make_adapter(&server, full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        while stream.next().await.is_some() {}
        mock.assert_async().await;
    }

    // ============ image_generate 正常 + 错误路径 ============

    #[tokio::test]
    async fn image_generate_success_parses_url() {
        let mut server = Server::new_async().await;
        let body = json!({
            "created": 1700000000,
            "data": [{
                "url": "https://example.com/img.png",
                "revised_prompt": "a cute cat"
            }]
        });
        server
            .mock("POST", "/images/generations")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ImageRequest::builder("dall-e-3", "a cat")
            .size("1024x1024")
            .build();
        let resp = adapter.image_generate(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(
            resp.data[0].url.as_deref(),
            Some("https://example.com/img.png")
        );
        assert_eq!(resp.data[0].revised_prompt.as_deref(), Some("a cute cat"));
    }

    #[tokio::test]
    async fn image_generate_success_parses_b64() {
        let mut server = Server::new_async().await;
        let body = json!({
            "created": 1700000000,
            "data": [{"b64_json": "aGVsbG8="}]
        });
        server
            .mock("POST", "/images/generations")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = ImageRequest::builder("dall-e-3", "a cat").build();
        let resp = adapter.image_generate(req).await.unwrap();
        assert_eq!(resp.data[0].b64_json.as_deref(), Some("aGVsbG8="));
    }

    #[tokio::test]
    async fn image_generate_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/images/generations")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;
        let adapter = make_adapter(&server, full_caps());
        let req = ImageRequest::builder("dall-e-3", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn image_generate_error_429() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/images/generations")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;
        let adapter = make_adapter(&server, full_caps());
        let req = ImageRequest::builder("dall-e-3", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn image_generate_unsupported_capability() {
        let server = Server::new_async().await;
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::Chat);
        let adapter = make_adapter(&server, caps);
        let req = ImageRequest::builder("dall-e-3", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ embed 正常 + 错误路径 ============

    #[tokio::test]
    async fn embed_success_parses_vectors() {
        let mut server = Server::new_async().await;
        let body = json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]}
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        });
        server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = EmbedRequest {
            model: "text-embedding-3-small".into(),
            input: EmbedInput::Multiple(vec!["a".into(), "b".into()]),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let resp = adapter.embed(req).await.unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].index, 0);
        if let EmbeddingVector::Float(v) = &resp.data[0].embedding {
            assert_eq!(v, &vec![0.1, 0.2, 0.3]);
        } else {
            panic!("应为 Float 向量");
        }
        assert_eq!(resp.usage.as_ref().unwrap().prompt_tokens, 4);
    }

    #[tokio::test]
    async fn embed_single_input() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "text-embedding-3-small",
                "input": "hello"
            })))
            .with_status(200)
            .with_body(
                json!({
                    "object": "list",
                    "data": [{"object":"embedding","index":0,"embedding":[0.1]}],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let req = EmbedRequest {
            model: "text-embedding-3-small".into(),
            input: EmbedInput::Single("hello".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let resp = adapter.embed(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn embed_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;
        let adapter = make_adapter(&server, full_caps());
        let req = EmbedRequest {
            model: "text-embedding-3-small".into(),
            input: EmbedInput::Single("hi".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let err = adapter.embed(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn embed_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(404)
            .with_body(json!({"error": {"message": "model not found"}}).to_string())
            .create_async()
            .await;
        let adapter = make_adapter(&server, full_caps());
        let req = EmbedRequest {
            model: "embed-x".into(),
            input: EmbedInput::Single("hi".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let err = adapter.embed(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn embed_unsupported_capability() {
        let server = Server::new_async().await;
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::Chat);
        let adapter = make_adapter(&server, caps);
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

    // ============ list_models 正常 + 错误路径 ============

    #[tokio::test]
    async fn list_models_success() {
        let mut server = Server::new_async().await;
        let body = json!({
            "object": "list",
            "data": [
                {"id": "gpt-4o", "object": "model", "created": 1700000000, "owned_by": "openai"},
                {"id": "dall-e-3", "object": "model", "created": 1700000000, "owned_by": "openai"},
                {"id": "whisper-1", "object": "model", "created": 1700000000, "owned_by": "openai"}
            ]
        });
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 3);
        // 类型推断
        assert_eq!(models[0].id, "gpt-4o");
        assert_eq!(models[0].model_type, ModelType::Chat);
        assert_eq!(models[1].model_type, ModelType::Image);
        assert_eq!(models[2].model_type, ModelType::Audio);
        // provider 字段填充
        assert_eq!(models[0].provider, "openai");
    }

    #[tokio::test]
    async fn list_models_filter_by_type() {
        let mut server = Server::new_async().await;
        let body = json!({
            "data": [
                {"id": "gpt-4o", "object": "model", "created": 1, "owned_by": "openai"},
                {"id": "dall-e-3", "object": "model", "created": 1, "owned_by": "openai"}
            ]
        });
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, full_caps());
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "dall-e-3");
    }

    #[tokio::test]
    async fn list_models_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;
        let adapter = make_adapter(&server, full_caps());
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn list_models_error_429() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow"}}).to_string())
            .create_async()
            .await;
        let adapter = make_adapter(&server, full_caps());
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn list_models_error_500() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(500)
            .with_body(json!({"error": {"message": "internal"}}).to_string())
            .create_async()
            .await;
        let adapter = make_adapter(&server, full_caps());
        let err = adapter.list_models(None).await.unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 500),
            _ => panic!("应为 Api"),
        }
    }

    // ============ 错误映射单元测试 ============

    #[test]
    fn map_api_error_401_with_message() {
        let body = json!({"error": {"message": "Invalid API key"}}).to_string();
        let err = OpenAiCompatAdapter::map_api_error(401, &body);
        match err {
            AibridgeError::Authentication { message } => {
                assert_eq!(message, "Invalid API key");
            }
            _ => panic!("应为 Authentication"),
        }
    }

    #[test]
    fn map_api_error_429_extracts_retry_after() {
        let body = json!({"error": {"message": "slow down", "retry_after": 2.0}}).to_string();
        let err = OpenAiCompatAdapter::map_api_error(429, &body);
        match err {
            AibridgeError::RateLimit { retry_after, .. } => {
                assert_eq!(retry_after, Some(2.0));
            }
            _ => panic!("应为 RateLimit"),
        }
    }

    #[test]
    fn map_api_error_404_uses_message_as_model() {
        let body = json!({"error": {"message": "model gpt-x not found"}}).to_string();
        let err = OpenAiCompatAdapter::map_api_error(404, &body);
        match err {
            AibridgeError::ModelNotFound { model } => {
                assert!(model.contains("gpt-x"));
            }
            _ => panic!("应为 ModelNotFound"),
        }
    }

    #[test]
    fn map_api_error_500_is_api() {
        let err = OpenAiCompatAdapter::map_api_error(503, "service unavailable");
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 503),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_no_json_body_falls_back_to_http_status() {
        let err = OpenAiCompatAdapter::map_api_error(502, "Bad Gateway");
        match err {
            AibridgeError::Api { message, .. } => {
                assert!(message.contains("502"));
            }
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_400_is_validation() {
        let body = json!({"error": {"message": "bad param"}}).to_string();
        let err = OpenAiCompatAdapter::map_api_error(400, &body);
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    // ============ 参数映射 ============

    #[test]
    fn openai_compatible_mapping_is_passthrough() {
        let pm = openai_compatible_mapping();
        let mut params = HashMap::new();
        params.insert("max_tokens".to_string(), json!(1000));
        params.insert("temperature".to_string(), json!(0.7));
        let result = pm.apply(&params);
        // OpenAI 兼容映射为空表，参数原名透传
        assert_eq!(
            result.get("max_tokens").and_then(|v| v.as_i64()),
            Some(1000)
        );
        assert_eq!(
            result.get("temperature").and_then(|v| v.as_f64()),
            Some(0.7)
        );
    }

    // ============ 辅助方法 ============

    #[tokio::test]
    async fn base_url_uses_config_when_provided() {
        // new() 应使用 config.base_url，而非默认值
        let config = ProviderConfig::from_options(
            "openai",
            ClientOptions::builder()
                .api_key("k")
                .base_url("https://custom.example.com/v1")
                .build(),
        );
        let adapter = OpenAiCompatAdapter::new(
            config,
            "openai",
            "OpenAI",
            DEFAULT_OPENAI_BASE_URL,
            full_caps(),
        )
        .unwrap();
        assert_eq!(adapter.base_url(), "https://custom.example.com/v1");
    }

    #[tokio::test]
    async fn base_url_falls_back_to_default_when_missing() {
        let config =
            ProviderConfig::from_options("openai", ClientOptions::builder().api_key("k").build());
        let adapter = OpenAiCompatAdapter::new(
            config,
            "openai",
            "OpenAI",
            DEFAULT_OPENAI_BASE_URL,
            full_caps(),
        )
        .unwrap();
        assert_eq!(adapter.base_url(), DEFAULT_OPENAI_BASE_URL);
    }

    #[test]
    fn ensure_capability_blocks_unsupported() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("openai", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter =
            OpenAiCompatAdapter::with_http(http, config, "openai", "OpenAI", CapabilitySet::new());
        let err = adapter.ensure_capability(Capabilities::Chat).unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[test]
    fn ensure_capability_allows_supported() {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::Chat);
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("openai", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = OpenAiCompatAdapter::with_http(http, config, "openai", "OpenAI", caps);
        assert!(adapter.ensure_capability(Capabilities::Chat).is_ok());
    }

    #[test]
    fn build_chat_body_includes_messages_and_model() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("openai", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = OpenAiCompatAdapter::with_http(http, config, "openai", "OpenAI", full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .build();
        let body = adapter.build_chat_body(&req, false);
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["temperature"], 0.7);
        assert!(body.get("messages").is_some());
        // 非 stream 模式不应注入 stream_options
        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn build_chat_body_stream_adds_stream_options() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("openai", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = OpenAiCompatAdapter::with_http(http, config, "openai", "OpenAI", full_caps());
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let body = adapter.build_chat_body(&req, true);
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn parse_chunk_skips_empty_choices_without_usage() {
        let value =
            json!({"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[]});
        let result = OpenAiCompatAdapter::parse_chunk(&value, "m").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_chunk_with_usage_returns_some() {
        let value = json!({
            "id":"x","object":"chat.completion.chunk","created":1,"model":"m",
            "choices":[],
            "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
        });
        let result = OpenAiCompatAdapter::parse_chunk(&value, "m").unwrap();
        assert!(result.is_some());
        let chunk = result.unwrap();
        assert_eq!(chunk.usage.as_ref().unwrap().total_tokens, 7);
    }
}
