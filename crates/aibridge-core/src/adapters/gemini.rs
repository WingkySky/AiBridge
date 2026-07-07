//! Google Gemini 适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/gemini.py`。
//!
//! Gemini 是独立协议（非 OpenAI 兼容），核心端点：
//! - 对话：`POST /models/{model}:generateContent`
//! - 流式对话：`POST /models/{model}:streamGenerateContent?alt=sse`
//! - 嵌入：`POST /models/{model}:embedContent`（单条）/ `:batchEmbedContents`（多条）
//! - 模型列表：`GET /models`
//! - 认证：API Key 通过 `x-goog-api-key` header 传递（也可作 `key` query param）
//!
//! 设计要点（与设计文档第 10 节一致）：
//! - 独立协议，不复用 `OpenAiCompatAdapter`
//! - 请求体用 Gemini 的 `contents`/`parts` 结构，system 消息转 `systemInstruction`
//! - assistant 角色 → Gemini 的 `model` 角色
//! - 多模态 `image_url` (data URI) → Gemini 的 `inline_data` (mime_type + data)
//! - 参数映射 `GEMINI_MAPPING`：max_tokens→maxOutputTokens、top_p→topP、top_k→topK、stop→stopSequences
//! - 错误映射：Gemini 错误体 `{"error":{"code","message","status"}}` → AibridgeError
//! - 图像生成：通过 `:generateContent` + `generationConfig.responseModalities:["IMAGE","TEXT"]`
//!   调用 Gemini 图像生成模型（如 `gemini-2.0-flash-exp-image-generation`），
//!   响应 `parts` 中的 `inlineData.data` (base64) 转为统一 `ImageData.b64_json`

use std::collections::HashMap;

use async_trait::async_trait;
use futures::stream::{StreamExt, TryStreamExt};
use serde_json::{json, Value};

use crate::adapter::{Adapter, Capabilities, CapabilitySet, ChatStream};
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::chat::{
    ChatChoice, ChatCompletion, ChatCompletionChunk, ChatCompletionDelta, ChatMessage, ChatRequest,
    ChoiceMessage, DeltaMessage, UserContent,
};
use crate::model::common::{infer_model_type, ModelInfo, ModelType};
use crate::model::image::{ImageData, ImageRequest, ImageResult};
use crate::model::options::{
    EmbedInput, EmbedRequest, EmbeddingItem, EmbeddingResult, EmbeddingVector, ParameterMapping,
};
use crate::util;

/// Gemini 官方默认 Base URL
pub const DEFAULT_GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Gemini API key 请求头名
const API_KEY_HEADER: &str = "x-goog-api-key";

/// Gemini 通用参数映射
///
/// 对应 Python v1 `agn/models/options.py` 的 `GEMINI_MAPPING`。
/// Gemini 的 generationConfig 字段命名与统一参数不同，需重命名：
/// - `max_tokens` → `maxOutputTokens`
/// - `top_p` → `topP`
/// - `top_k` → `topK`
/// - `stop` → `stopSequences`
/// - `response_format` → `response_mime_type`（值需额外转换，此处仅重命名键）
///
/// 注：Python 老版的 `value_map`（reasoning/web_search）在 Rust 统一请求中
/// 走 `extra` 透传更合适，故此处仅保留 rename_map。
pub fn gemini_mapping() -> ParameterMapping {
    let mut rename_map = HashMap::new();
    rename_map.insert(
        "max_tokens".to_string(),
        Some("maxOutputTokens".to_string()),
    );
    rename_map.insert("top_p".to_string(), Some("topP".to_string()));
    rename_map.insert("top_k".to_string(), Some("topK".to_string()));
    rename_map.insert("stop".to_string(), Some("stopSequences".to_string()));
    rename_map.insert(
        "response_format".to_string(),
        Some("response_mime_type".to_string()),
    );
    ParameterMapping { rename_map }
}

/// Google Gemini 适配器
///
/// 持有 HTTP 客户端与 Provider 配置，实现 Gemini 独立协议。
/// 不复用 `OpenAiCompatAdapter`（Gemini 请求/响应结构、认证方式、错误格式均不同）。
pub struct GeminiAdapter {
    /// HTTP 客户端（封装 reqwest，含连接池与超时）
    http: HttpClient,
    /// Provider 配置（api_key / base_url / timeout 等）
    config: ProviderConfig,
    /// 实际 base_url（已合并 config.base_url 与默认值）
    base_url: String,
    /// 支持的能力集合
    capabilities: CapabilitySet,
}

impl GeminiAdapter {
    /// 创建 Gemini 适配器
    ///
    /// `config.base_url` 为空时用 `DEFAULT_GEMINI_BASE_URL` 兜底。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_GEMINI_BASE_URL.to_string());

        let opts = ClientOptions::builder()
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
            base_url,
            capabilities: Self::default_capabilities(),
        })
    }

    /// 用显式 HttpClient 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_http(http: HttpClient, config: ProviderConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_GEMINI_BASE_URL.to_string());
        Self {
            http,
            config,
            base_url,
            capabilities: Self::default_capabilities(),
        }
    }

    /// 默认能力集合：CHAT / IMAGE_GENERATE / EMBEDDING
    fn default_capabilities() -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::Chat);
        caps.insert(Capabilities::ChatStream);
        caps.insert(Capabilities::ImageGenerate);
        caps.insert(Capabilities::Embedding);
        caps.insert(Capabilities::Vision);
        caps
    }

    /// API key（可能为空）
    fn api_key(&self) -> Option<&str> {
        self.config.api_key.as_deref().filter(|s| !s.is_empty())
    }

    /// base_url
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// 拼接完整 URL
    fn url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        let path = path.trim_start_matches('/');
        format!("{base}/{path}")
    }

    /// 校验能力是否被支持
    fn ensure_capability(&self, cap: Capabilities) -> Result<()> {
        if self.capabilities.contains(&cap) {
            Ok(())
        } else {
            Err(AibridgeError::UnsupportedCapability {
                capability: format!("{} (provider: gemini)", cap.as_str()),
            })
        }
    }

    /// 校验 API key 是否存在（requires_api_key=true）
    fn ensure_api_key(&self) -> Result<&str> {
        match self.api_key() {
            Some(k) => Ok(k),
            None => Err(AibridgeError::Validation {
                message: "Gemini 适配器需要 API key（x-goog-api-key）".into(),
                details: serde_json::json!({ "provider": "gemini" }),
            }),
        }
    }

    // ==================== 消息格式转换 ====================

    /// 将统一 `ChatRequest` 转为 Gemini 请求体
    ///
    /// - system 消息 → `systemInstruction`（单独字段，不在 contents 中）
    /// - user 消息 → `contents` 中 role=user
    /// - assistant 消息 → `contents` 中 role=model（Gemini 用 model 表示助手）
    /// - tool 消息 → 作为 user 内容追加（Gemini 无独立 tool role）
    /// - 多模态 image_url (data URI) → `inline_data` (mime_type + data)
    /// - 通用参数（temperature/top_p/top_k/max_tokens/stop）→ `generationConfig`，经 GEMINI_MAPPING 重命名
    /// - extra 透传到顶层
    fn build_chat_body(&self, req: &ChatRequest) -> Value {
        let mut contents: Vec<Value> = Vec::new();
        let mut system_instruction: Option<String> = None;

        for msg in &req.messages {
            match msg {
                ChatMessage::System { content, .. } => {
                    if system_instruction.is_none() {
                        system_instruction = Some(content.clone());
                    }
                }
                ChatMessage::User { content, .. } => {
                    let parts = Self::convert_user_content(content);
                    contents.push(json!({ "role": "user", "parts": parts }));
                }
                ChatMessage::Assistant { content, .. } => {
                    let text = content.clone().unwrap_or_default();
                    contents.push(json!({ "role": "model", "parts": [{ "text": text }] }));
                }
                ChatMessage::Tool { content, .. } => {
                    // 工具结果作为 user 消息内容追加（Gemini 无 tool role）
                    contents.push(json!({ "role": "user", "parts": [{ "text": content }] }));
                }
            }
        }

        let mut body = json!({ "contents": contents });

        if let Some(sys) = system_instruction {
            body["systemInstruction"] = json!({ "parts": [{ "text": sys }] });
        }

        // generationConfig：经 GEMINI_MAPPING 重命名通用参数
        let gen_config = self.build_generation_config(req);
        if !gen_config.is_null() {
            body["generationConfig"] = gen_config;
        }

        // extra 透传到顶层
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }

        body
    }

    /// 构造 generationConfig（应用 GEMINI_MAPPING 重命名）
    ///
    /// 只收集非空的通用生成参数。返回 `Value::Null` 表示无参数。
    fn build_generation_config(&self, req: &ChatRequest) -> Value {
        let mut params: HashMap<String, Value> = HashMap::new();
        if let Some(t) = req.temperature {
            params.insert("temperature".into(), json!(t));
        }
        if let Some(t) = req.top_p {
            params.insert("top_p".into(), json!(t));
        }
        if let Some(t) = req.top_k {
            params.insert("top_k".into(), json!(t));
        }
        if let Some(t) = req.max_tokens {
            params.insert("max_tokens".into(), json!(t));
        }
        if let Some(stop) = &req.stop {
            // StopSeq 可能是 Single 或 Multiple，统一转数组
            let arr: Vec<String> = match stop {
                crate::model::options::StopSeq::Single(s) => vec![s.clone()],
                crate::model::options::StopSeq::Multiple(v) => v.clone(),
            };
            params.insert("stop".into(), json!(arr));
        }

        let mapped = gemini_mapping().apply(&params);
        if mapped.is_empty() {
            return Value::Null;
        }
        let mut obj = serde_json::Map::new();
        for (k, v) in mapped {
            obj.insert(k, v);
        }
        Value::Object(obj)
    }

    /// 将统一 UserContent 转为 Gemini parts 数组
    ///
    /// - 纯文本 → `[{"text": "..."}]`
    /// - 多模态 → 文本部件 + 图像 inline_data（data URI 解析出 mime_type + base64 data）
    fn convert_user_content(content: &UserContent) -> Vec<Value> {
        match content {
            UserContent::Text(s) => vec![json!({ "text": s })],
            UserContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    crate::model::chat::ContentPart::Text { text } => Some(json!({ "text": text })),
                    crate::model::chat::ContentPart::ImageUrl { image_url } => {
                        Self::convert_image_url(&image_url.url)
                    }
                })
                .collect(),
        }
    }

    /// 将 OpenAI 风格 image_url（URL 或 data URI）转为 Gemini part
    ///
    /// - data URI (`data:image/png;base64,xxx`) → `inline_data` (mime_type + data)
    /// - 普通 URL → `file_data` (file_uri)（Gemini 用 file_data 引用远程文件）
    fn convert_image_url(url: &str) -> Option<Value> {
        if let Some(rest) = url.strip_prefix("data:") {
            // data URI 格式: data:<mime>;base64,<data>
            if let Some((meta, data)) = rest.split_once(',') {
                let mime_type = meta.split(';').next().unwrap_or("image/png");
                return Some(json!({
                    "inline_data": { "mime_type": mime_type, "data": data }
                }));
            }
        }
        // 普通远程 URL：用 file_data 引用
        Some(json!({
            "file_data": { "file_uri": url, "mime_type": "image/png" }
        }))
    }

    // ==================== HTTP 请求 ====================

    /// 发送带认证的 POST JSON 请求，用 Gemini 错误映射处理响应
    async fn post_authed_json(&self, url: &str, body: &Value) -> Result<Value> {
        let api_key = self.ensure_api_key()?;
        let resp = self
            .http
            .inner()
            .post(url)
            .header(API_KEY_HEADER, api_key)
            .json(body)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(Self::map_api_error(status_code, &body_text));
        }
        resp.json::<Value>().await.map_err(AibridgeError::from)
    }

    /// 发送带认证的 GET 请求，用 Gemini 错误映射处理响应
    async fn get_authed_json(&self, url: &str) -> Result<Value> {
        let api_key = self.ensure_api_key()?;
        let resp = self
            .http
            .inner()
            .get(url)
            .header(API_KEY_HEADER, api_key)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(Self::map_api_error(status_code, &body_text));
        }
        resp.json::<Value>().await.map_err(AibridgeError::from)
    }

    // ==================== 响应解析 ====================

    /// 解析 Gemini generateContent 响应 → ChatCompletion
    ///
    /// Gemini 响应格式：
    /// ```json
    /// {
    ///   "candidates": [{
    ///     "content": { "parts": [{"text":"..."}], "role": "model" },
    ///     "finishReason": "STOP"
    ///   }],
    ///   "usageMetadata": {
    ///     "promptTokenCount": 5,
    ///     "candidatesTokenCount": 10,
    ///     "totalTokenCount": 15
    ///   }
    /// }
    /// ```
    fn parse_chat_completion(&self, value: &Value, fallback_model: &str) -> Result<ChatCompletion> {
        let candidates = value.get("candidates").and_then(|v| v.as_array());
        let (content, finish_reason) = match candidates.and_then(|arr| arr.first()) {
            Some(cand) => {
                let parts = cand
                    .get("content")
                    .and_then(|c| c.get("parts"))
                    .and_then(|p| p.as_array());
                let text = parts
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                            .collect::<Vec<_>>()
                            .join("")
                    })
                    .unwrap_or_default();
                let raw_reason = cand
                    .get("finishReason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("STOP");
                (text, map_finish_reason(raw_reason))
            }
            None => (String::new(), "stop".to_string()),
        };

        let usage = value.get("usageMetadata").and_then(parse_usage_metadata);

        Ok(ChatCompletion {
            id: util::generate_id("chatcmpl"),
            object: "chat.completion".into(),
            created: util::current_timestamp(),
            model: fallback_model.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChoiceMessage {
                    role: "assistant".into(),
                    content: Some(content),
                    tool_calls: None,
                },
                finish_reason: Some(finish_reason),
            }],
            usage,
            service_tier: None,
            system_fingerprint: None,
        })
    }

    /// 解析单个 SSE chunk（Gemini streamGenerateContent 格式）→ Option<ChatCompletionChunk>
    ///
    /// Gemini 每个 SSE data 行是一个完整的 generateContent 响应（含 candidates）。
    /// 文本增量取 candidates[0].content.parts 的 text 拼接；finishReason 出现时单独发一个结束块。
    /// 返回 None 表示该 chunk 无有效内容（如空 candidates），调用方跳过。
    fn parse_stream_chunk(value: &Value, model: &str) -> Result<Vec<ChatCompletionChunk>> {
        let id = util::generate_id("chatcmpl");
        let created = util::current_timestamp();
        let candidates = match value.get("candidates").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => return Ok(Vec::new()),
        };

        let mut chunks = Vec::new();
        if let Some(cand) = candidates.first() {
            let parts = cand
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array());
            let text = parts
                .map(|arr| {
                    arr.iter()
                        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join("")
                })
                .unwrap_or_default();

            if !text.is_empty() {
                chunks.push(ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model.to_string(),
                    choices: vec![ChatCompletionDelta {
                        index: 0,
                        delta: DeltaMessage {
                            role: None,
                            content: Some(text),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                });
            }

            // finishReason 出现时发结束块
            if let Some(raw) = cand.get("finishReason").and_then(|v| v.as_str()) {
                chunks.push(ChatCompletionChunk {
                    id,
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model.to_string(),
                    choices: vec![ChatCompletionDelta {
                        index: 0,
                        delta: DeltaMessage {
                            role: None,
                            content: None,
                            tool_calls: None,
                        },
                        finish_reason: Some(map_finish_reason(raw)),
                    }],
                    usage: None,
                });
            }
        }

        // 末尾可能携带 usageMetadata
        if let Some(usage) = value.get("usageMetadata").and_then(parse_usage_metadata) {
            chunks.push(ChatCompletionChunk {
                id: util::generate_id("chatcmpl"),
                object: "chat.completion.chunk".into(),
                created,
                model: model.to_string(),
                choices: Vec::new(),
                usage: Some(usage),
            });
        }

        Ok(chunks)
    }

    /// 解析 Gemini /models 响应 → Vec<ModelInfo>
    ///
    /// Gemini 响应特殊：模型列表在 `models` 键下（非 OpenAI 的 `data`），
    /// 模型 ID 在 `name` 字段且带 `models/` 前缀（如 `models/gemini-1.5-pro`），
    /// 显示名在 `displayName` 字段。需去掉前缀作为统一 id。
    fn parse_models(&self, value: &Value) -> Vec<ModelInfo> {
        let arr = value.get("models").and_then(|v| v.as_array());
        match arr {
            Some(arr) => arr
                .iter()
                .filter_map(|m| {
                    let name = m.get("name").and_then(|v| v.as_str())?;
                    // 去掉 "models/" 前缀作为模型 ID
                    let id = name.strip_prefix("models/").unwrap_or(name).to_string();
                    if id.is_empty() {
                        return None;
                    }
                    let display_name = m
                        .get("displayName")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&id)
                        .to_string();
                    let model_type = infer_model_type(&id);
                    let supports_streaming = matches!(model_type, ModelType::Chat);
                    Some(ModelInfo {
                        name: display_name,
                        id,
                        model_type,
                        provider: "gemini".into(),
                        capabilities: Vec::new(),
                        max_tokens: m
                            .get("inputTokenLimit")
                            .and_then(|v| v.as_u64())
                            .map(|x| x as u32),
                        supports_streaming,
                        description: m
                            .get("description")
                            .and_then(|v| v.as_str())
                            .map(str::to_owned),
                        created: None,
                    })
                })
                .collect(),
            None => Vec::new(),
        }
    }

    // ==================== 错误映射 ====================

    /// 将 Gemini API 错误响应映射为 AibridgeError
    ///
    /// Gemini 错误格式：
    /// ```json
    /// { "error": { "code": 400, "message": "...", "status": "INVALID_ARGUMENT" } }
    /// ```
    /// 状态码分类：
    /// - 401/403 → Authentication
    /// - 429 → RateLimit
    /// - 404 → ModelNotFound
    /// - 400 → Validation
    /// - 4xx（其他）→ Api
    /// - 5xx → Api
    pub fn map_api_error(status: u16, body: &str) -> AibridgeError {
        let message = parse_gemini_error_message(body, status);
        match status {
            401 | 403 => AibridgeError::Authentication { message },
            429 => AibridgeError::RateLimit {
                message,
                retry_after: None,
            },
            400 => AibridgeError::Validation {
                message,
                details: serde_json::json!({ "status_code": status, "response": body }),
            },
            404 => AibridgeError::ModelNotFound { model: message },
            s => AibridgeError::Api { status: s, message },
        }
    }
}

#[async_trait]
impl Adapter for GeminiAdapter {
    fn provider_type(&self) -> &str {
        "gemini"
    }

    fn provider_name(&self) -> &str {
        "Google Gemini"
    }

    fn capabilities(&self) -> CapabilitySet {
        self.capabilities.clone()
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    async fn start(&mut self) -> Result<()> {
        // HttpClient 已在 new() 中初始化，无需额外启动
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        // reqwest::Client 通过 Drop 自动释放连接池，无需显式关闭
        Ok(())
    }

    /// 文本对话
    ///
    /// `POST /models/{model}:generateContent`，构造 Gemini 请求体，解析响应 → ChatCompletion。
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.ensure_capability(Capabilities::Chat)?;
        let model = req.model.clone();
        let body = self.build_chat_body(&req);
        let url = self.url(&format!("models/{model}:generateContent"));
        let value = self.post_authed_json(&url, &body).await?;
        self.parse_chat_completion(&value, &model)
    }

    /// 流式文本对话
    ///
    /// `POST /models/{model}:streamGenerateContent?alt=sse`，SSE 流 → ChatStream。
    /// Gemini 流式格式：每个 `data:` 行是一个完整 generateContent 响应（含 candidates）。
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.ensure_capability(Capabilities::ChatStream)?;
        let model = req.model.clone();
        let body = self.build_chat_body(&req);
        let url = self.url(&format!("models/{model}:streamGenerateContent?alt=sse"));

        let api_key = self.ensure_api_key()?;
        let resp = self
            .http
            .inner()
            .post(&url)
            .header(API_KEY_HEADER, api_key)
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(Self::map_api_error(status_code, &body_text));
        }

        // 按字节流读取，按行切分解析 SSE（与 openai_compat 一致的 LinesStream 模式）
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
                // Gemini 流无 [DONE] 标记，自然结束即结束
                match serde_json::from_str::<Value>(data) {
                    Ok(v) => match Self::parse_stream_chunk(&v, &model) {
                        Ok(chunks) => {
                            for chunk in chunks {
                                yield Ok(chunk);
                            }
                        }
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    },
                    Err(_) => {
                        // 单行 JSON 解析失败不致命，跳过（与 Python 老版一致）
                        continue;
                    }
                }
            }
        };

        Ok(stream.boxed())
    }

    /// 图像生成
    ///
    /// 通过 `:generateContent` 端点调用 Gemini 图像生成模型
    /// （如 `gemini-2.0-flash-exp-image-generation`）。
    /// 请求体 `generationConfig.responseModalities: ["IMAGE","TEXT"]`，
    /// 响应 `candidates[0].content.parts` 中的 `inlineData.data` (base64) 转为 `ImageData.b64_json`。
    async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
        self.ensure_capability(Capabilities::ImageGenerate)?;
        let model = req.model.clone();
        let prompt = req.prompt.clone();

        let mut body = json!({
            "contents": [{ "role": "user", "parts": [{ "text": prompt }] }],
            "generationConfig": { "responseModalities": ["IMAGE", "TEXT"] }
        });

        // 透传 extra（如 negative_prompt 等 provider 特有参数）
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }

        let url = self.url(&format!("models/{model}:generateContent"));
        let value = self.post_authed_json(&url, &body).await?;

        // 提取响应中的图像（inlineData → b64_json）
        let mut data: Vec<ImageData> = Vec::new();
        if let Some(parts) = value
            .get("candidates")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|cand| cand.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
        {
            for part in parts {
                if let Some(inline) = part.get("inlineData").or_else(|| part.get("inline_data")) {
                    let b64 = inline
                        .get("data")
                        .and_then(|d| d.as_str())
                        .map(str::to_owned);
                    let mime = inline
                        .get("mimeType")
                        .or_else(|| inline.get("mime_type"))
                        .and_then(|m| m.as_str())
                        .unwrap_or("image/png");
                    if let Some(b64) = b64 {
                        data.push(ImageData {
                            url: None,
                            b64_json: Some(b64),
                            revised_prompt: Some(format!("Generated image ({mime})")),
                        });
                    }
                }
            }
        }

        Ok(ImageResult {
            id: util::generate_id("img"),
            object: "image.generation".into(),
            created: util::current_timestamp(),
            model,
            data,
        })
    }

    /// 文本嵌入
    ///
    /// - 单条输入：`POST /models/{model}:embedContent`
    /// - 多条输入：`POST /models/{model}:batchEmbedContents`
    ///
    /// Gemini 嵌入响应无 usage 统计，故 `usage` 为 None。
    async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        self.ensure_capability(Capabilities::Embedding)?;
        let model = req.model.clone();

        let result = match &req.input {
            EmbedInput::Single(text) => {
                let mut body = json!({
                    "model": format!("models/{model}"),
                    "content": { "parts": [{ "text": text }] }
                });
                if let Some(dim) = req.dimensions {
                    body["outputDimensionality"] = json!(dim);
                }
                let url = self.url(&format!("models/{model}:embedContent"));
                let value = self.post_authed_json(&url, &body).await?;

                let values = value
                    .get("embedding")
                    .and_then(|e| e.get("values"))
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect::<Vec<f64>>())
                    .unwrap_or_default();
                vec![EmbeddingItem {
                    object: "embedding".into(),
                    index: 0,
                    embedding: EmbeddingVector::Float(values),
                }]
            }
            EmbedInput::Multiple(texts) => {
                let requests: Vec<Value> = texts
                    .iter()
                    .map(|text| {
                        let mut r = json!({
                            "model": format!("models/{model}"),
                            "content": { "parts": [{ "text": text }] }
                        });
                        if let Some(dim) = req.dimensions {
                            r["outputDimensionality"] = json!(dim);
                        }
                        r
                    })
                    .collect();
                let body = json!({ "requests": requests });
                let url = self.url(&format!("models/{model}:batchEmbedContents"));
                let value = self.post_authed_json(&url, &body).await?;

                value
                    .get("embeddings")
                    .and_then(|e| e.as_array())
                    .map(|arr| {
                        arr.iter()
                            .enumerate()
                            .map(|(i, emb)| {
                                let values = emb
                                    .get("values")
                                    .and_then(|v| v.as_array())
                                    .map(|arr| {
                                        arr.iter().filter_map(|x| x.as_f64()).collect::<Vec<f64>>()
                                    })
                                    .unwrap_or_default();
                                EmbeddingItem {
                                    object: "embedding".into(),
                                    index: i as u32,
                                    embedding: EmbeddingVector::Float(values),
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            }
        };

        Ok(EmbeddingResult {
            object: "list".into(),
            data: result,
            model,
            usage: None,
        })
    }

    /// 模型列表（实时拉取）
    ///
    /// `GET /models` → Vec<ModelInfo>，按 `filter` 过滤模型类型。
    /// Gemini 响应的 `models` 数组中 `name` 带 `models/` 前缀，需去掉。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let url = self.url("models");
        let value = self.get_authed_json(&url).await?;
        let models = self.parse_models(&value);
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }
}

// ==================== 辅助函数 ====================

/// 将 reqwest::Error 映射为 AibridgeError（超时 → Timeout，其余 → Network）
fn map_reqwest_error(err: reqwest::Error) -> AibridgeError {
    if err.is_timeout() {
        AibridgeError::Timeout
    } else {
        AibridgeError::Network(err)
    }
}

/// 解析 Gemini 错误体中的 message 字段
///
/// Gemini 错误格式：`{"error": {"code": 400, "message": "...", "status": "INVALID_ARGUMENT"}}`
/// 解析失败时回退到 `HTTP {status}` 字符串。
fn parse_gemini_error_message(body: &str, status: u16) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        if let Some(msg) = v
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return msg.to_string();
        }
        // 部分错误体直接用顶层 message
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

/// 将 Gemini finishReason 映射为统一 finish_reason
///
/// - STOP → stop
/// - MAX_TOKENS → length
/// - SAFETY → content_filter
/// - RECITATION → stop（Python 老版映射为 stop）
/// - 其他 → stop（兜底）
fn map_finish_reason(raw: &str) -> String {
    match raw {
        "STOP" => "stop".into(),
        "MAX_TOKENS" => "length".into(),
        "SAFETY" => "content_filter".into(),
        "RECITATION" => "stop".into(),
        _ => "stop".into(),
    }
}

/// 解析 Gemini usageMetadata → ChatUsage
///
/// Gemini 字段：promptTokenCount / candidatesTokenCount / totalTokenCount
fn parse_usage_metadata(v: &Value) -> Option<crate::model::chat::ChatUsage> {
    let prompt = v.get("promptTokenCount").and_then(|x| x.as_u64())?;
    let completion = v
        .get("candidatesTokenCount")
        .and_then(|x| x.as_u64())
        .unwrap_or(0);
    let total = v
        .get("totalTokenCount")
        .and_then(|x| x.as_u64())
        .unwrap_or(prompt + completion);
    Some(crate::model::chat::ChatUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: total,
    })
}

// ==================== SSE 行流适配器（与 openai_compat 一致） ====================

/// 将字节流按行切分的适配器
///
/// 维护一个未完成行的缓冲区，逐 chunk 拼接出完整行。
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
    use crate::model::chat::ContentPart;
    use crate::model::image::FileInput;
    use mockito::Server;

    /// 构造测试用 GeminiAdapter（指向 mockito server）
    fn make_adapter(server: &Server) -> GeminiAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("gemini", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        GeminiAdapter::with_http(http, config)
    }

    /// 构造无 API key 的适配器（测试 ensure_api_key）
    fn make_adapter_no_key(server: &Server) -> GeminiAdapter {
        let opts = ClientOptions::builder()
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("gemini", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        GeminiAdapter::with_http(http, config)
    }

    // ============ chat 正常路径 ============

    #[tokio::test]
    async fn chat_success_parses_completion() {
        let mut server = Server::new_async().await;
        let body = json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello from Gemini!"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8
            }
        });
        let mock = server
            .mock("POST", "/models/gemini-2.5-pro:generateContent")
            .match_header("x-goog-api-key", "test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gemini-2.5-pro", vec![ChatMessage::user("hi")]).build();
        let resp = adapter.chat(req).await.expect("chat 应成功");

        assert_eq!(resp.model, "gemini-2.5-pro");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("Hello from Gemini!")
        );
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 8);
        assert_eq!(resp.usage.as_ref().unwrap().prompt_tokens, 5);
        assert_eq!(resp.usage.as_ref().unwrap().completion_tokens, 3);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_converts_messages_to_gemini_format() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/models/gemini-2.5-pro:generateContent")
            .match_body(mockito::Matcher::PartialJson(json!({
                "contents": [
                    { "role": "user", "parts": [{ "text": "hello" }] },
                    { "role": "model", "parts": [{ "text": "hi there" }] },
                    { "role": "user", "parts": [{ "text": "how are you?" }] }
                ],
                "systemInstruction": { "parts": [{ "text": "You are helpful." }] }
            })))
            .with_status(200)
            .with_body(
                json!({
                    "candidates": [{
                        "content": { "parts": [{"text": "ok"}], "role": "model" },
                        "finishReason": "STOP"
                    }]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder(
            "gemini-2.5-pro",
            vec![
                ChatMessage::system("You are helpful."),
                ChatMessage::user("hello"),
                ChatMessage::assistant("hi there"),
                ChatMessage::user("how are you?"),
            ],
        )
        .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_applies_generation_config_mapping() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/models/gemini-2.5-pro:generateContent")
            .match_body(mockito::Matcher::PartialJson(json!({
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.9,
                    "topK": 40,
                    "maxOutputTokens": 1000,
                    "stopSequences": ["END"]
                }
            })))
            .with_status(200)
            .with_body(
                json!({
                    "candidates": [{
                        "content": { "parts": [{"text": "ok"}], "role": "model" },
                        "finishReason": "STOP"
                    }]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gemini-2.5-pro", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .top_p(0.9)
            .top_k(40)
            .max_tokens(1000)
            .stop(crate::model::options::StopSeq::Single("END".into()))
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_passes_extra_to_top_level() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/models/gemini-2.5-pro:generateContent")
            .match_body(mockito::Matcher::PartialJson(json!({
                "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
                "thinkingConfig": { "thinkingBudget": -1 }
            })))
            .with_status(200)
            .with_body(
                json!({
                    "candidates": [{
                        "content": { "parts": [{"text": "ok"}], "role": "model" },
                        "finishReason": "STOP"
                    }]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gemini-2.5-pro", vec![ChatMessage::user("hi")])
            .extra("thinkingConfig", json!({ "thinkingBudget": -1 }))
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_finish_reason_mappings() {
        let cases = [
            ("STOP", "stop"),
            ("MAX_TOKENS", "length"),
            ("SAFETY", "content_filter"),
            ("RECITATION", "stop"),
            ("OTHER", "stop"),
        ];
        for (raw, expected) in cases {
            let mut server = Server::new_async().await;
            server
                .mock("POST", "/models/m:generateContent")
                .with_status(200)
                .with_body(
                    json!({
                        "candidates": [{
                            "content": { "parts": [{"text": "x"}], "role": "model" },
                            "finishReason": raw
                        }]
                    })
                    .to_string(),
                )
                .create_async()
                .await;
            let adapter = make_adapter(&server);
            let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
            let resp = adapter.chat(req).await.unwrap();
            assert_eq!(
                resp.choices[0].finish_reason.as_deref(),
                Some(expected),
                "finishReason={raw} 应映射为 {expected}"
            );
        }
    }

    #[tokio::test]
    async fn chat_multimodal_converts_data_uri_to_inline_data() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/models/m:generateContent")
            .match_body(mockito::Matcher::PartialJson(json!({
                "contents": [{
                    "role": "user",
                    "parts": [
                        { "text": "describe this" },
                        { "inline_data": { "mime_type": "image/png", "data": "aGVsbG8=" } }
                    ]
                }]
            })))
            .with_status(200)
            .with_body(
                json!({
                    "candidates": [{
                        "content": { "parts": [{"text": "a cat"}], "role": "model" },
                        "finishReason": "STOP"
                    }]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder(
            "m",
            vec![ChatMessage::user_multimodal(vec![
                ContentPart::Text {
                    text: "describe this".into(),
                },
                ContentPart::ImageUrl {
                    image_url: crate::model::chat::ImageUrl::new("data:image/png;base64,aGVsbG8="),
                },
            ])],
        )
        .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_no_candidates_returns_empty_content() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/models/m:generateContent")
            .with_status(200)
            .with_body(json!({}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some(""));
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    // ============ chat 错误路径 ============

    #[tokio::test]
    async fn chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/models/m:generateContent")
            .with_status(401)
            .with_body(
                json!({"error": {"code": 401, "message": "API key not valid", "status": "UNAUTHENTICATED"}}).to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Authentication { message } => {
                assert!(message.contains("API key not valid"));
            }
            _ => panic!("应为 Authentication，实际: {err:?}"),
        }
    }

    #[tokio::test]
    async fn chat_error_403_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/models/m:generateContent")
            .with_status(403)
            .with_body(json!({"error": {"message": "permission denied"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/models/m:generateContent")
            .with_status(429)
            .with_body(
                json!({"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}}).to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::RateLimit {
                message,
                retry_after,
            } => {
                assert!(message.contains("Resource exhausted"));
                assert_eq!(retry_after, None);
            }
            _ => panic!("应为 RateLimit"),
        }
    }

    #[tokio::test]
    async fn chat_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/models/gemini-x:generateContent")
            .with_status(404)
            .with_body(
                json!({"error": {"code": 404, "message": "models/gemini-x not found"}}).to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gemini-x", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::ModelNotFound { model } => {
                assert!(model.contains("gemini-x"));
            }
            _ => panic!("应为 ModelNotFound"),
        }
    }

    #[tokio::test]
    async fn chat_error_400_returns_validation() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/models/m:generateContent")
            .with_status(400)
            .with_body(
                json!({"error": {"code": 400, "message": "Invalid argument", "status": "INVALID_ARGUMENT"}}).to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Validation { message, .. } => {
                assert!(message.contains("Invalid argument"));
            }
            _ => panic!("应为 Validation"),
        }
    }

    #[tokio::test]
    async fn chat_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/models/m:generateContent")
            .with_status(500)
            .with_body(json!({"error": {"message": "Internal error"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 500),
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn chat_missing_api_key_returns_validation() {
        let server = Server::new_async().await;
        let adapter = make_adapter_no_key(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[tokio::test]
    async fn chat_unsupported_capability_returns_error() {
        // GeminiAdapter 默认支持 Chat，这里通过空能力集合的 adapter 测试
        let server = Server::new_async().await;
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .build();
        let config = ProviderConfig::from_options("gemini", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        let mut adapter = GeminiAdapter::with_http(http, config);
        adapter.capabilities = CapabilitySet::new();
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ chat_stream 正常 + 错误路径 ============

    #[tokio::test]
    async fn chat_stream_parses_sse_chunks() {
        let mut server = Server::new_async().await;
        // Gemini SSE：每个 data 行是一个完整 generateContent 响应
        let sse = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"}}]}\n\
                   data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" world\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n";
        server
            .mock("POST", "/models/gemini-2.5-pro:streamGenerateContent")
            .match_query(mockito::Matcher::UrlEncoded("alt".into(), "sse".into()))
            .match_header("x-goog-api-key", "test-key")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gemini-2.5-pro", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.expect("stream 应建立");
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        // 第 1 块文本 "Hello"，第 2 块文本 " world" + 结束块
        assert!(chunks.len() >= 2);
        let mut content = String::new();
        let mut finish: Option<String> = None;
        for chunk in &chunks {
            if let Some(c) = chunk
                .choices
                .first()
                .and_then(|d| d.delta.content.as_deref())
            {
                content.push_str(c);
            }
            if let Some(f) = chunk
                .choices
                .first()
                .and_then(|d| d.finish_reason.as_deref())
            {
                finish = Some(f.to_string());
            }
        }
        assert_eq!(content, "Hello world");
        assert_eq!(finish.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn chat_stream_handles_heartbeat_and_empty_lines() {
        let mut server = Server::new_async().await;
        let sse = ": heartbeat\n\n\
                   data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}],\"role\":\"model\"}}]}\n\n";
        server
            .mock("POST", "/models/m:streamGenerateContent")
            .match_query(mockito::Matcher::UrlEncoded("alt".into(), "sse".into()))
            .with_status(200)
            .with_body(sse)
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
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
            .mock("POST", "/models/m:streamGenerateContent")
            .match_query(mockito::Matcher::UrlEncoded("alt".into(), "sse".into()))
            .with_status(401)
            .with_body(json!({"error": {"message": "Unauthorized"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        match adapter.chat_stream(req).await {
            Err(e) => assert!(matches!(e, AibridgeError::Authentication { .. })),
            Ok(_) => panic!("chat_stream 应返回错误而非 stream"),
        }
    }

    #[tokio::test]
    async fn chat_stream_sends_alt_sse_query() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/models/m:streamGenerateContent")
            .match_query(mockito::Matcher::UrlEncoded("alt".into(), "sse".into()))
            .with_status(200)
            .with_body("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"x\"}]}}]}\n\n")
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        while stream.next().await.is_some() {}
        mock.assert_async().await;
    }

    // ============ image_generate 正常 + 错误路径 ============

    #[tokio::test]
    async fn image_generate_success_parses_inline_data() {
        let mut server = Server::new_async().await;
        let body = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": "aGVsbG8="
                        }
                    }],
                    "role": "model"
                },
                "finishReason": "STOP"
            }]
        });
        let mock = server
            .mock(
                "POST",
                "/models/gemini-2.0-flash-exp-image-generation:generateContent",
            )
            .match_body(mockito::Matcher::PartialJson(json!({
                "contents": [{ "role": "user", "parts": [{ "text": "a cat" }] }],
                "generationConfig": { "responseModalities": ["IMAGE", "TEXT"] }
            })))
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("gemini-2.0-flash-exp-image-generation", "a cat").build();
        let resp = adapter.image_generate(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].b64_json.as_deref(), Some("aGVsbG8="));
        assert!(resp.data[0].revised_prompt.is_some());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_accepts_snake_case_inline_data() {
        // 部分 Gemini 兼容端点返回 snake_case 的 inline_data
        let mut server = Server::new_async().await;
        let body = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": "AAAA"
                        }
                    }]
                }
            }]
        });
        server
            .mock("POST", "/models/m:generateContent")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("m", "test").build();
        let resp = adapter.image_generate(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].b64_json.as_deref(), Some("AAAA"));
    }

    #[tokio::test]
    async fn image_generate_no_image_returns_empty_data() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/models/m:generateContent")
            .with_status(200)
            .with_body(
                json!({
                    "candidates": [{
                        "content": { "parts": [{ "text": "no image generated" }] }
                    }]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("m", "test").build();
        let resp = adapter.image_generate(req).await.unwrap();
        assert_eq!(resp.data.len(), 0);
    }

    #[tokio::test]
    async fn image_generate_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/models/m:generateContent")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("m", "test").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn image_generate_passes_extra_params() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/models/m:generateContent")
            .match_body(mockito::Matcher::PartialJson(json!({
                "negativePrompt": "blurry"
            })))
            .with_status(200)
            .with_body(json!({
                "candidates": [{
                    "content": { "parts": [{ "inlineData": { "mimeType": "image/png", "data": "x" } }] }
                }]
            }).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("m", "test")
            .extra("negativePrompt", "blurry")
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_unsupported_capability() {
        let server = Server::new_async().await;
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .build();
        let config = ProviderConfig::from_options("gemini", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        let mut adapter = GeminiAdapter::with_http(http, config);
        adapter.capabilities = CapabilitySet::new();
        let req = ImageRequest::builder("m", "test").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ embed 正常 + 错误路径 ============

    #[tokio::test]
    async fn embed_single_input_success() {
        let mut server = Server::new_async().await;
        let body = json!({
            "embedding": { "values": [0.1, 0.2, 0.3] }
        });
        let mock = server
            .mock("POST", "/models/text-embedding-004:embedContent")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "models/text-embedding-004",
                "content": { "parts": [{ "text": "hello" }] }
            })))
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = EmbedRequest {
            model: "text-embedding-004".into(),
            input: EmbedInput::Single("hello".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let resp = adapter.embed(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].index, 0);
        if let EmbeddingVector::Float(v) = &resp.data[0].embedding {
            assert_eq!(v, &vec![0.1, 0.2, 0.3]);
        } else {
            panic!("应为 Float 向量");
        }
        assert!(resp.usage.is_none(), "Gemini 嵌入无 usage 统计");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn embed_single_input_with_dimensions() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/models/m:embedContent")
            .match_body(mockito::Matcher::PartialJson(json!({
                "outputDimensionality": 256
            })))
            .with_status(200)
            .with_body(json!({"embedding": {"values": [0.1]}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = EmbedRequest {
            model: "m".into(),
            input: EmbedInput::Single("hi".into()),
            dimensions: Some(256),
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let _ = adapter.embed(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn embed_multiple_input_uses_batch_endpoint() {
        let mut server = Server::new_async().await;
        let body = json!({
            "embeddings": [
                { "values": [0.1, 0.2] },
                { "values": [0.3, 0.4] }
            ]
        });
        let mock = server
            .mock("POST", "/models/text-embedding-004:batchEmbedContents")
            .match_body(mockito::Matcher::PartialJson(json!({
                "requests": [
                    { "model": "models/text-embedding-004", "content": { "parts": [{ "text": "a" }] } },
                    { "model": "models/text-embedding-004", "content": { "parts": [{ "text": "b" }] } }
                ]
            })))
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = EmbedRequest {
            model: "text-embedding-004".into(),
            input: EmbedInput::Multiple(vec!["a".into(), "b".into()]),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let resp = adapter.embed(req).await.unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].index, 0);
        assert_eq!(resp.data[1].index, 1);
        if let EmbeddingVector::Float(v) = &resp.data[0].embedding {
            assert_eq!(v, &vec![0.1, 0.2]);
        } else {
            panic!("应为 Float 向量");
        }
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn embed_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/models/m:embedContent")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = EmbedRequest {
            model: "m".into(),
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
            .mock("POST", "/models/embed-x:embedContent")
            .with_status(404)
            .with_body(json!({"error": {"message": "model not found"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
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
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .build();
        let config = ProviderConfig::from_options("gemini", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        let mut adapter = GeminiAdapter::with_http(http, config);
        adapter.capabilities = CapabilitySet::new();
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
    async fn list_models_success_strips_models_prefix() {
        let mut server = Server::new_async().await;
        let body = json!({
            "models": [
                {
                    "name": "models/gemini-1.5-pro",
                    "displayName": "Gemini 1.5 Pro",
                    "description": "Gemini 1.5 Pro model",
                    "inputTokenLimit": 2000000
                },
                {
                    "name": "models/text-embedding-004",
                    "displayName": "Text Embedding 004"
                },
                {
                    "name": "models/imagen-3.0",
                    "displayName": "Imagen 3"
                }
            ]
        });
        server
            .mock("GET", "/models")
            .match_header("x-goog-api-key", "test-key")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].id, "gemini-1.5-pro");
        assert_eq!(models[0].name, "Gemini 1.5 Pro");
        assert_eq!(models[0].provider, "gemini");
        assert_eq!(models[0].model_type, ModelType::Chat);
        assert_eq!(models[0].max_tokens, Some(2000000));
        assert_eq!(models[1].id, "text-embedding-004");
        // text-embedding-004 不含图像/视频/音频关键字，infer 为 Chat
        assert_eq!(models[2].id, "imagen-3.0");
        assert_eq!(models[2].model_type, ModelType::Image);
    }

    #[tokio::test]
    async fn list_models_filter_by_type() {
        let mut server = Server::new_async().await;
        let body = json!({
            "models": [
                { "name": "models/gemini-1.5-pro", "displayName": "Gemini 1.5 Pro" },
                { "name": "models/imagen-3.0", "displayName": "Imagen 3" }
            ]
        });
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "imagen-3.0");
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

        let adapter = make_adapter(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn list_models_error_429() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    // ============ 错误映射单元测试 ============

    #[test]
    fn map_api_error_401_with_message() {
        let body = json!({"error": {"code": 401, "message": "API key invalid"}}).to_string();
        let err = GeminiAdapter::map_api_error(401, &body);
        match err {
            AibridgeError::Authentication { message } => {
                assert_eq!(message, "API key invalid");
            }
            _ => panic!("应为 Authentication"),
        }
    }

    #[test]
    fn map_api_error_429_is_rate_limit_no_retry_after() {
        let body = json!({"error": {"message": "slow down"}}).to_string();
        let err = GeminiAdapter::map_api_error(429, &body);
        match err {
            AibridgeError::RateLimit { retry_after, .. } => {
                assert_eq!(retry_after, None);
            }
            _ => panic!("应为 RateLimit"),
        }
    }

    #[test]
    fn map_api_error_404_uses_message_as_model() {
        let body = json!({"error": {"message": "models/x not found"}}).to_string();
        let err = GeminiAdapter::map_api_error(404, &body);
        match err {
            AibridgeError::ModelNotFound { model } => {
                assert!(model.contains("models/x not found"));
            }
            _ => panic!("应为 ModelNotFound"),
        }
    }

    #[test]
    fn map_api_error_400_is_validation() {
        let body =
            json!({"error": {"code": 400, "message": "bad arg", "status": "INVALID_ARGUMENT"}})
                .to_string();
        let err = GeminiAdapter::map_api_error(400, &body);
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[test]
    fn map_api_error_500_is_api() {
        let err = GeminiAdapter::map_api_error(503, "service unavailable");
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 503),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_no_json_falls_back_to_http_status() {
        let err = GeminiAdapter::map_api_error(502, "Bad Gateway");
        match err {
            AibridgeError::Api { message, .. } => {
                assert!(message.contains("502"));
            }
            _ => panic!("应为 Api"),
        }
    }

    // ============ 参数映射 ============

    #[test]
    fn gemini_mapping_renames_keys() {
        let pm = gemini_mapping();
        let mut params = HashMap::new();
        params.insert("max_tokens".to_string(), json!(1000));
        params.insert("top_p".to_string(), json!(0.9));
        params.insert("top_k".to_string(), json!(40));
        params.insert("stop".to_string(), json!(["END"]));
        params.insert("temperature".to_string(), json!(0.7));
        let result = pm.apply(&params);
        assert_eq!(
            result.get("maxOutputTokens").and_then(|v| v.as_i64()),
            Some(1000)
        );
        assert_eq!(result.get("topP").and_then(|v| v.as_f64()), Some(0.9));
        assert_eq!(result.get("topK").and_then(|v| v.as_i64()), Some(40));
        assert!(result.contains_key("stopSequences"));
        // temperature 不在 rename_map，原名透传
        assert_eq!(
            result.get("temperature").and_then(|v| v.as_f64()),
            Some(0.7)
        );
    }

    // ============ 辅助方法测试 ============

    #[test]
    fn convert_image_url_data_uri_to_inline_data() {
        let part = GeminiAdapter::convert_image_url("data:image/png;base64,aGVsbG8=").unwrap();
        assert!(part.get("inline_data").is_some());
        assert_eq!(part["inline_data"]["mime_type"].as_str(), Some("image/png"));
        assert_eq!(part["inline_data"]["data"].as_str(), Some("aGVsbG8="));
    }

    #[test]
    fn convert_image_url_data_uri_jpeg() {
        let part = GeminiAdapter::convert_image_url("data:image/jpeg;base64,AAAA").unwrap();
        assert_eq!(
            part["inline_data"]["mime_type"].as_str(),
            Some("image/jpeg")
        );
    }

    #[test]
    fn convert_image_url_remote_url_to_file_data() {
        let part = GeminiAdapter::convert_image_url("https://example.com/img.png").unwrap();
        assert!(part.get("file_data").is_some());
        assert_eq!(
            part["file_data"]["file_uri"].as_str(),
            Some("https://example.com/img.png")
        );
    }

    #[test]
    fn parse_gemini_error_message_extracts_error_message() {
        let body = json!({"error": {"code": 400, "message": "bad arg"}}).to_string();
        let msg = parse_gemini_error_message(&body, 400);
        assert_eq!(msg, "bad arg");
    }

    #[test]
    fn parse_gemini_error_message_falls_back_to_http_status() {
        let msg = parse_gemini_error_message("", 500);
        assert_eq!(msg, "HTTP 500");
    }

    #[test]
    fn map_finish_reason_stop_variants() {
        assert_eq!(map_finish_reason("STOP"), "stop");
        assert_eq!(map_finish_reason("MAX_TOKENS"), "length");
        assert_eq!(map_finish_reason("SAFETY"), "content_filter");
        assert_eq!(map_finish_reason("RECITATION"), "stop");
        assert_eq!(map_finish_reason("UNKNOWN"), "stop");
    }

    #[test]
    fn parse_usage_metadata_extracts_tokens() {
        let v = json!({
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        });
        let usage = parse_usage_metadata(&v).unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn parse_usage_metadata_falls_back_to_sum() {
        let v = json!({ "promptTokenCount": 10, "candidatesTokenCount": 5 });
        let usage = parse_usage_metadata(&v).unwrap();
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn parse_usage_metadata_none_without_prompt() {
        let v = json!({ "candidatesTokenCount": 5 });
        assert!(parse_usage_metadata(&v).is_none());
    }

    #[tokio::test]
    async fn base_url_uses_config_when_provided() {
        let config = ProviderConfig::from_options(
            "gemini",
            ClientOptions::builder()
                .api_key("k")
                .base_url("https://custom.example.com/v1beta")
                .build(),
        );
        let adapter = GeminiAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), "https://custom.example.com/v1beta");
    }

    #[tokio::test]
    async fn base_url_falls_back_to_default_when_missing() {
        let config =
            ProviderConfig::from_options("gemini", ClientOptions::builder().api_key("k").build());
        let adapter = GeminiAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), DEFAULT_GEMINI_BASE_URL);
    }

    #[tokio::test]
    async fn requires_api_key_is_true() {
        let server = Server::new_async().await;
        let adapter = make_adapter(&server);
        assert!(adapter.requires_api_key());
    }

    #[tokio::test]
    async fn capabilities_contains_chat_image_embed() {
        let server = Server::new_async().await;
        let adapter = make_adapter(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::ImageGenerate));
        assert!(caps.contains(&Capabilities::Embedding));
    }

    #[tokio::test]
    async fn provider_metadata_correct() {
        let server = Server::new_async().await;
        let adapter = make_adapter(&server);
        assert_eq!(adapter.provider_type(), "gemini");
        assert_eq!(adapter.provider_name(), "Google Gemini");
    }

    #[tokio::test]
    async fn start_and_close_are_noops() {
        let mut adapter = GeminiAdapter::new(ProviderConfig::from_options(
            "gemini",
            ClientOptions::builder().api_key("k").build(),
        ))
        .unwrap();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    #[tokio::test]
    async fn unsupported_video_returns_unsupported_capability() {
        let server = Server::new_async().await;
        let adapter = make_adapter(&server);
        let req = crate::model::video::VideoRequest::builder("m", "prompt").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn unsupported_speech_returns_unsupported_capability() {
        let server = Server::new_async().await;
        let adapter = make_adapter(&server);
        let req = crate::model::audio::SpeechRequest::builder("m", "hi", "v1").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn unsupported_transcribe_returns_unsupported_capability() {
        let server = Server::new_async().await;
        let adapter = make_adapter(&server);
        let req =
            crate::model::audio::TranscribeRequest::builder("m", FileInput::path("/tmp/a.mp3"))
                .build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }
}
