//! Stability AI 适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/stability.py`。
//!
//! Stability AI 为**独立协议**（非 OpenAI 兼容），不复用 `OpenAiCompatAdapter`：
//! - 图像生成：`POST /v1/generation/{engine_id}/text-to-image`（同步，返回 base64 图像）
//! - 模型列表：`GET /v1/engines/list`（返回顶层数组）
//! - 认证：`Authorization: Bearer <api_key>`
//!
//! ## 协议要点
//! - 请求体用 `text_prompts` 数组（每项含 `text` + `weight`，正向 weight=1，负向 weight=-1）
//! - 尺寸用 `width` / `height` 数值字段（非 size 字符串）
//! - 采样参数：`steps` / `seed` / `cfg_scale` / `samples`（生成数量，上限 10）/ `style_preset`
//! - 响应：`{"artifacts": [{"base64": "..."}]}`（注意是 `base64` 而非 `b64_json`）
//!
//! ## 能力范围
//! 仅 `ImageGenerate`。chat / video / embed / audio 走 `Adapter` trait 默认实现
//! 返 `UnsupportedCapability`，与 Python v1 抛 `UnsupportedCapabilityError` 行为一致。
//!
//! ## 错误映射（对齐任务要求，与 Python v1 略有差异）
//! - 401 → Authentication
//! - 429 → RateLimit
//! - 400 → Validation（Python v1 走 Api，此处按统一错误规范映射为 Validation）
//! - 5xx 及其余 ≥400 → Api（提取 message）

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::adapter::{Adapter, Capabilities, CapabilitySet};
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::common::{infer_model_type, ModelInfo, ModelType};
use crate::model::image::{ImageData, ImageRequest, ImageResult};
use crate::util;

// ==================== 默认配置 ====================

/// Stability AI 默认 Base URL
///
/// 对应 Python v1 `DEFAULT_BASE_URL`。
pub const DEFAULT_STABILITY_BASE_URL: &str = "https://api.stability.ai";

/// Stability AI 默认引擎（SDXL 1.1）
///
/// 对应 Python v1 `DEFAULT_ENGINE`。
pub const DEFAULT_ENGINE: &str = "stable-diffusion-xl-1024-v1-1";

/// 默认图像尺寸（与 Python v1 一致）
const DEFAULT_WIDTH: u32 = 1024;
const DEFAULT_HEIGHT: u32 = 1024;

/// samples（生成数量）上限（与 Python v1 `min(samples, 10)` 一致）
const MAX_SAMPLES: u32 = 10;

// ==================== 能力集合 ====================

/// Stability 支持的能力集合
///
/// 对齐 Python v1 `StabilityAdapter.supported_capabilities = ["image"]`。
/// Rust 用 `ImageGenerate` 表达图像生成能力。
fn stability_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::ImageGenerate);
    caps
}

// ==================== StabilityAdapter ====================

/// Stability AI 适配器
///
/// 持有 HTTP 客户端与 Provider 配置，实现 Stability 独立图像生成协议。
/// 不复用 `OpenAiCompatAdapter`（请求体结构、响应字段与 OpenAI 不一致）。
///
/// ## 阶段范围
/// 阶段 2b 实现 `image_generate`（文生图）+ `list_models`（实时拉取 engines）。
/// 图像编辑（image-to-image，multipart）待后续阶段补齐 `image_edit` trait 方法后再实现。
pub struct StabilityAdapter {
    /// HTTP 客户端（封装 reqwest，含连接池与超时）
    http: HttpClient,
    /// Provider 配置（api_key / base_url / timeout 等）
    config: ProviderConfig,
    /// 实际 base_url（已合并 config.base_url 与默认值）
    base_url: String,
    /// 支持的能力集合
    capabilities: CapabilitySet,
}

impl StabilityAdapter {
    /// 创建 Stability AI 适配器
    ///
    /// `config.base_url` 为空时用 [`DEFAULT_STABILITY_BASE_URL`] 兜底。
    /// `config.api_key` 为空时不在此处报错（由上层按 `requires_api_key` 校验）。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_STABILITY_BASE_URL.to_string());

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
            capabilities: stability_capabilities(),
        })
    }

    /// 用显式 HttpClient 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_http(http: HttpClient, config: ProviderConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_STABILITY_BASE_URL.to_string());
        Self {
            http,
            config,
            base_url,
            capabilities: stability_capabilities(),
        }
    }

    /// API key（可能为空）
    fn api_key(&self) -> &str {
        self.config.api_key.as_deref().unwrap_or("")
    }

    /// base_url
    pub fn base_url(&self) -> &str {
        &self.base_url
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
                capability: format!("{} (provider: stability)", cap.as_str()),
            })
        }
    }

    /// 发送带 Bearer 认证的 POST JSON 请求，并用 Stability 错误映射处理响应
    async fn post_authed_json(&self, path: &str, body: &Value) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .http
            .inner()
            .post(&url)
            .bearer_auth(self.api_key())
            .header("Accept", "application/json")
            .json(body)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        Self::parse_json_response(resp).await
    }

    /// 发送带 Bearer 认证的 GET 请求，并用 Stability 错误映射处理响应
    async fn get_authed_json(&self, path: &str) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .http
            .inner()
            .get(&url)
            .bearer_auth(self.api_key())
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(map_reqwest_error)?;
        Self::parse_json_response(resp).await
    }

    /// 统一解析响应：状态码非 2xx 走错误映射，成功则解析为 JSON
    async fn parse_json_response(resp: reqwest::Response) -> Result<Value> {
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(Self::map_api_error(status_code, &body_text));
        }
        resp.json::<Value>().await.map_err(AibridgeError::from)
    }

    // ==================== 内部：请求体构造 ====================

    /// 构造 Stability text-to-image 请求体
    ///
    /// 移植自 Python v1 `image_generate`：
    /// - `text_prompts`：正向 weight=1，负向 weight=-1
    /// - `width` / `height`：从 width/height/size 解析，默认 1024x1024
    /// - `samples`：生成数量，上限 10
    /// - `steps` / `seed` / `cfg_scale` / `style_preset`：透传
    /// - `extra`：合并到顶层（透传厂商特有参数）
    fn build_generate_body(&self, req: &ImageRequest) -> Value {
        // text_prompts：正向提示词
        let mut text_prompts = vec![json!({ "text": req.prompt, "weight": 1 })];
        // 负面提示词 weight=-1
        if let Some(np) = &req.negative_prompt {
            text_prompts.push(json!({ "text": np, "weight": -1 }));
        }

        // 尺寸解析
        let (width, height) = resolve_dimensions(req.width, req.height, req.size.as_deref());

        let mut body = json!({
            "text_prompts": text_prompts,
            "width": width,
            "height": height,
        });

        // 采样参数（可选）
        if let Some(steps) = req.steps {
            body["steps"] = json!(steps);
        }
        if let Some(seed) = req.seed {
            body["seed"] = json!(seed);
        }
        if let Some(cfg) = req.cfg_scale {
            body["cfg_scale"] = json!(cfg);
        }
        // samples 上限 10
        body["samples"] = json!(req.n.min(MAX_SAMPLES));
        // 风格预设
        if let Some(style) = &req.style {
            body["style_preset"] = json!(style);
        }

        // extra 透传（合并到顶层）
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        body
    }

    // ==================== 内部：响应解析 ====================

    /// 解析 Stability text-to-image 响应 → ImageResult
    ///
    /// Stability 响应结构：
    /// ```json
    /// { "artifacts": [{"base64": "...", "finishReason": "SUCCESS", "seed": 123}] }
    /// ```
    /// 注意：Stability 用 `base64` 而非 `b64_json`，无 `url` 字段。
    /// artifacts 为空时返 Api 错误（与 Python v1 `raise APIError("No image generated")` 一致）。
    fn parse_image_result(value: &Value, fallback_model: &str) -> Result<ImageResult> {
        let data: Vec<ImageData> = value
            .get("artifacts")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(parse_artifact).collect())
            .unwrap_or_default();

        if data.is_empty() {
            return Err(AibridgeError::Api {
                status: 0,
                message: "Stability 未返回图像数据 (artifacts 为空)".to_string(),
            });
        }

        Ok(ImageResult {
            id: util::generate_id("img"),
            object: "image.generation".to_string(),
            created: util::current_timestamp(),
            model: fallback_model.to_string(),
            data,
        })
    }

    /// 解析 Stability `/v1/engines/list` 响应 → Vec<ModelInfo>
    ///
    /// Stability engines/list 返回**顶层数组**（非 `{"data": [...]}`），格式：
    /// ```json
    /// [{"id": "stable-diffusion-xl-1024-v1-1", "name": "...", "description": "...", "create_time": 123}]
    /// ```
    /// 兼容 `{"data": [...]}` 包装格式。模型类型由 `infer_model_type` 推断
    /// （`stable-diffusion` 关键字 → Image）。
    fn parse_engines(value: &Value) -> Vec<ModelInfo> {
        let arr: Vec<&Value> = match value {
            Value::Array(a) => a.iter().collect(),
            Value::Object(_) => value
                .get("data")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().collect())
                .unwrap_or_default(),
            _ => Vec::new(),
        };

        arr.iter()
            .map(|m| {
                let id = m
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = m
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
                    .unwrap_or_else(|| id.clone());
                ModelInfo {
                    model_type: infer_model_type(&id),
                    provider: "stability".to_string(),
                    description: m
                        .get("description")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned),
                    created: m.get("create_time").and_then(|v| v.as_u64()),
                    id,
                    name,
                    capabilities: Vec::new(),
                    max_tokens: None,
                    supports_streaming: false,
                }
            })
            .collect()
    }

    // ==================== 内部：错误映射 ====================

    /// 将 Stability API 错误响应映射为 AibridgeError
    ///
    /// 对齐任务要求（与 Python v1 `_handle_stability_error` 略有差异）：
    /// - 401 → Authentication（"Invalid Stability API key"）
    /// - 429 → RateLimit（"Stability rate limit exceeded"）
    /// - 400 → Validation（携带 details，Python v1 走 Api）
    /// - 其余 ≥400 → Api（提取 message，回退 `HTTP {status}`）
    pub fn map_api_error(status: u16, body: &str) -> AibridgeError {
        match status {
            401 => AibridgeError::Authentication {
                message: "Invalid Stability API key".to_string(),
            },
            429 => AibridgeError::RateLimit {
                message: "Stability rate limit exceeded".to_string(),
                retry_after: None,
            },
            400 => {
                let message = parse_error_message(body, status);
                let details =
                    serde_json::from_str::<Value>(body).unwrap_or(serde_json::Value::Null);
                AibridgeError::Validation { message, details }
            }
            _ => {
                let message = parse_error_message(body, status);
                AibridgeError::Api { status, message }
            }
        }
    }
}

#[async_trait]
impl Adapter for StabilityAdapter {
    fn provider_type(&self) -> &str {
        "stability"
    }

    fn provider_name(&self) -> &str {
        "Stability AI"
    }

    fn capabilities(&self) -> CapabilitySet {
        self.capabilities.clone()
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    async fn start(&mut self) -> Result<()> {
        // HttpClient 在 new() 时已构造，无额外资源需初始化
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        // HttpClient 由 Drop 自动释放，无额外资源
        Ok(())
    }

    /// 图像生成：`POST /v1/generation/{engine_id}/text-to-image`
    ///
    /// `engine_id` 取 `req.model`，为空时用 [`DEFAULT_ENGINE`] 兜底。
    /// 响应解析 base64 图像到 `ImageResult.data`。
    async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
        self.ensure_capability(Capabilities::ImageGenerate)?;
        let engine = if req.model.is_empty() {
            DEFAULT_ENGINE.to_string()
        } else {
            req.model.clone()
        };
        let body = self.build_generate_body(&req);
        let path = format!("v1/generation/{engine}/text-to-image");
        let value = self.post_authed_json(&path, &body).await?;
        Self::parse_image_result(&value, &engine)
    }

    /// 模型列表（实时拉取 Stability engines）
    ///
    /// `GET /v1/engines/list`，返回顶层数组，按 `filter` 过滤模型类型。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let value = self.get_authed_json("v1/engines/list").await?;
        let models = Self::parse_engines(&value);
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }

    // chat / chat_stream / video / embed / audio 走 trait 默认实现返 UnsupportedCapability，
    // 与 Python v1 抛 UnsupportedCapabilityError 行为一致。
}

// ==================== 辅助函数 ====================

/// 从 ImageRequest 的 width/height/size 解析图像尺寸
///
/// 优先级：显式 width+height > size 字符串 > 默认 1024x1024。
/// 仅提供 width 或 height 之一时，另一维度用默认值。
/// size 字符串解析失败时回退到默认值。
///
/// 与 Python v1 `kwargs.get("width", 1024)` / `kwargs.get("height", 1024)` 行为一致。
fn resolve_dimensions(width: Option<u32>, height: Option<u32>, size: Option<&str>) -> (u32, u32) {
    match (width, height) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => (w, DEFAULT_HEIGHT),
        (None, Some(h)) => (DEFAULT_WIDTH, h),
        (None, None) => match size.and_then(|s| util::parse_size(s).ok()) {
            Some((w, h)) => (w, h),
            None => (DEFAULT_WIDTH, DEFAULT_HEIGHT),
        },
    }
}

/// 解析单个 artifact（Stability 图像响应项）
///
/// Stability artifact 字段：`base64`（图像数据）/ `seed` / `finishReason`。
/// 仅提取 base64，映射到 `ImageData.b64_json`。base64 为空的项被过滤。
fn parse_artifact(v: &Value) -> Option<ImageData> {
    let b64 = v.get("base64").and_then(|x| x.as_str())?;
    if b64.is_empty() {
        return None;
    }
    Some(ImageData {
        b64_json: Some(b64.to_owned()),
        url: None,
        revised_prompt: None,
    })
}

/// 将 reqwest::Error 映射为 AibridgeError
///
/// 超时 → Timeout；其余 → Network。
fn map_reqwest_error(err: reqwest::Error) -> AibridgeError {
    if err.is_timeout() {
        AibridgeError::Timeout
    } else {
        AibridgeError::Network(err)
    }
}

/// 解析错误体中的 message 字段
///
/// 兼容多种 Stability 错误体结构：
/// - `{"error": {"message": "..."}}`（OpenAI 风格）
/// - `{"message": "..."}`（Stability 常见顶层 message）
/// - `{"errors": [{"message": "..."}]}`（Stability 校验错误数组）
/// - `{"error": "..."}`（顶层 error 字符串）
///
/// 解析失败时回退到 `HTTP {status}` 字符串。
fn parse_error_message(body: &str, status: u16) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        // errors 数组（Stability validation 错误）
        if let Some(arr) = v.get("errors").and_then(|m| m.as_array()) {
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|x| x.get("message").and_then(|m| m.as_str()).map(str::to_owned))
                .collect();
            if !parts.is_empty() {
                return parts.join("; ");
            }
        }
        // error.message（嵌套）
        if let Some(msg) = v
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return msg.to_string();
        }
        // 顶层 error 字符串
        if let Some(msg) = v.get("error").and_then(|m| m.as_str()) {
            return msg.to_string();
        }
        // 顶层 message
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::chat::{ChatMessage, ChatRequest};
    use crate::model::video::VideoRequest;
    use mockito::Server;
    use serde_json::json;

    // ==================== 测试辅助 ====================

    /// 构造测试用 StabilityAdapter（指向 mockito server）
    fn make_adapter(server: &Server) -> StabilityAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("stability", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        StabilityAdapter::with_http(http, config)
    }

    /// 构造不指向任何 server 的 StabilityAdapter（用于不发请求的元信息/能力测试）
    fn make_adapter_no_server() -> StabilityAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(DEFAULT_STABILITY_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("stability", opts);
        StabilityAdapter::new(config).expect("StabilityAdapter 构造应成功")
    }

    // ============ 元信息 ============

    #[test]
    fn provider_type_and_name_match_python() {
        let adapter = make_adapter_no_server();
        assert_eq!(adapter.provider_type(), "stability");
        assert_eq!(adapter.provider_name(), "Stability AI");
    }

    #[test]
    fn requires_api_key_is_true() {
        let adapter = make_adapter_no_server();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn capabilities_contains_only_image() {
        let adapter = make_adapter_no_server();
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::ImageGenerate));
        // chat / video 不声明
        assert!(!caps.contains(&Capabilities::Chat));
        assert!(!caps.contains(&Capabilities::VideoGenerate));
    }

    #[test]
    fn base_url_defaults_when_missing() {
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("stability", opts);
        let adapter = StabilityAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), DEFAULT_STABILITY_BASE_URL);
    }

    #[test]
    fn base_url_uses_config_when_provided() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://custom.stability-proxy.com")
            .build();
        let config = ProviderConfig::from_options("stability", opts);
        let adapter = StabilityAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), "https://custom.stability-proxy.com");
    }

    // ============ image_generate 正常路径 ============

    #[tokio::test]
    async fn image_generate_success_parses_base64() {
        let mut server = Server::new_async().await;
        let body = json!({
            "artifacts": [
                {"base64": "aGVsbG8=", "finishReason": "SUCCESS", "seed": 123},
                {"base64": "d29ybGQ=", "finishReason": "SUCCESS", "seed": 456}
            ]
        });
        let mock = server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .match_header("authorization", "Bearer test-key")
            .match_header("accept", "application/json")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat").build();
        let resp = adapter
            .image_generate(req)
            .await
            .expect("image_generate 应成功");

        assert_eq!(resp.model, "stable-diffusion-xl-1024-v1-1");
        assert_eq!(resp.object, "image.generation");
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].b64_json.as_deref(), Some("aGVsbG8="));
        assert_eq!(resp.data[1].b64_json.as_deref(), Some("d29ybGQ="));
        // Stability 只返回 base64，无 url
        assert!(resp.data[0].url.is_none());
        // id 与 created 应已填充
        assert!(resp.id.starts_with("img_"));
        assert!(resp.created > 0);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_uses_default_engine_when_model_empty() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .with_status(200)
            .with_body(json!({"artifacts": [{"base64": "aGk="}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("", "a cat").build();
        let resp = adapter.image_generate(req).await.unwrap();
        // 回退到默认引擎
        assert_eq!(resp.model, DEFAULT_ENGINE);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_uses_custom_engine_in_path() {
        // 验证自定义 model 作为 engine_id 出现在 URL 路径
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-3-medium/text-to-image",
            )
            .with_status(200)
            .with_body(json!({"artifacts": [{"base64": "aGk="}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-3-medium", "a cat").build();
        let resp = adapter.image_generate(req).await.unwrap();
        assert_eq!(resp.model, "stable-diffusion-3-medium");
        mock.assert_async().await;
    }

    // ============ image_generate 参数映射 ============

    #[tokio::test]
    async fn image_generate_sends_text_prompts_with_weight() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .match_body(mockito::Matcher::PartialJson(json!({
                "text_prompts": [
                    {"text": "a cat", "weight": 1},
                    {"text": "blurry", "weight": -1}
                ]
            })))
            .with_status(200)
            .with_body(json!({"artifacts": [{"base64": "aGk="}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat")
            .negative_prompt("blurry")
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_default_size_1024x1024() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .match_body(mockito::Matcher::PartialJson(json!({
                "width": 1024,
                "height": 1024
            })))
            .with_status(200)
            .with_body(json!({"artifacts": [{"base64": "aGk="}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat").build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_parses_size_string_to_width_height() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .match_body(mockito::Matcher::PartialJson(json!({
                "width": 1344,
                "height": 768
            })))
            .with_status(200)
            .with_body(json!({"artifacts": [{"base64": "aGk="}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat")
            .size("1344x768")
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_width_height_take_priority_over_size() {
        // 同时提供 width/height 和 size，应优先用 width/height
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .match_body(mockito::Matcher::PartialJson(json!({
                "width": 1536,
                "height": 640
            })))
            .with_status(200)
            .with_body(json!({"artifacts": [{"base64": "aGk="}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat")
            .width(1536)
            .height(640)
            .size("1024x1024")
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_passes_sampling_params() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .match_body(mockito::Matcher::PartialJson(json!({
                "steps": 40,
                "seed": 42,
                "cfg_scale": 7.5
            })))
            .with_status(200)
            .with_body(json!({"artifacts": [{"base64": "aGk="}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat")
            .steps(40)
            .seed(42)
            .cfg_scale(7.5)
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_caps_samples_at_10() {
        // 请求 n=20，应被截断为 samples=10
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .match_body(mockito::Matcher::PartialJson(json!({
                "samples": 10
            })))
            .with_status(200)
            .with_body(json!({"artifacts": [{"base64": "aGk="}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat")
            .n(20)
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_passes_style_preset() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .match_body(mockito::Matcher::PartialJson(json!({
                "style_preset": "anime"
            })))
            .with_status(200)
            .with_body(json!({"artifacts": [{"base64": "aGk="}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat")
            .style("anime")
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_passes_extra_params() {
        // extra 字段透传到顶层（如 sampler、sampler_seed 等厂商特有参数）
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .match_body(mockito::Matcher::PartialJson(json!({
                "sampler": "K_DPMPP_2M",
                "custom_param": "value"
            })))
            .with_status(200)
            .with_body(json!({"artifacts": [{"base64": "aGk="}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat")
            .extra("sampler", "K_DPMPP_2M")
            .extra("custom_param", "value")
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_no_negative_prompt_omits_second_entry() {
        // 无 negative_prompt 时 text_prompts 仅含正向一项
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .match_body(mockito::Matcher::PartialJson(json!({
                "text_prompts": [{"text": "a cat", "weight": 1}]
            })))
            .with_status(200)
            .with_body(json!({"artifacts": [{"base64": "aGk="}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat").build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    // ============ image_generate 错误路径 ============

    #[tokio::test]
    async fn image_generate_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .with_status(401)
            .with_body(json!({"message": "invalid api key"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn image_generate_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .with_status(429)
            .with_body(json!({"message": "slow down"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn image_generate_error_400_returns_validation() {
        // 任务要求 400 → Validation（携带 details）
        let mut server = Server::new_async().await;
        server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .with_status(400)
            .with_body(json!({"message": "invalid width"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        match err {
            AibridgeError::Validation { message, details } => {
                assert_eq!(message, "invalid width");
                // details 携带原始错误体
                assert_eq!(details["message"], "invalid width");
            }
            _ => panic!("应为 Validation"),
        }
    }

    #[tokio::test]
    async fn image_generate_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .with_status(500)
            .with_body(json!({"message": "internal error"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 500);
                assert_eq!(message, "internal error");
            }
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn image_generate_error_403_returns_api() {
        // 403（引擎不可用/配额）走 Api（与 Python v1 一致）
        let mut server = Server::new_async().await;
        server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .with_status(403)
            .with_body(json!({"message": "engine not accessible"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 403);
                assert!(message.contains("engine not accessible"));
            }
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn image_generate_error_no_json_body_falls_back() {
        // 非 JSON 错误体回退到 HTTP {status}
        let mut server = Server::new_async().await;
        server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .with_status(502)
            .with_body("Bad Gateway")
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("502")),
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn image_generate_no_artifacts_returns_api_error() {
        // artifacts 为空时返 Api 错误（与 Python v1 `raise APIError("No image generated")` 一致）
        let mut server = Server::new_async().await;
        server
            .mock(
                "POST",
                "/v1/generation/stable-diffusion-xl-1024-v1-1/text-to-image",
            )
            .with_status(200)
            .with_body(json!({"artifacts": []}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("stable-diffusion-xl-1024-v1-1", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("artifacts")),
            _ => panic!("应为 Api"),
        }
    }

    // ============ list_models ============

    #[tokio::test]
    async fn list_models_success_parses_top_level_array() {
        // Stability engines/list 返回顶层数组
        let mut server = Server::new_async().await;
        let mock = server
            .mock("GET", "/v1/engines/list")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!([
                    {"id": "stable-diffusion-xl-1024-v1-1", "name": "Stable Diffusion XL 1.1", "description": "SDXL", "create_time": 1700000000},
                    {"id": "stable-diffusion-3-medium", "name": "Stable Diffusion 3 Medium", "create_time": 1700000100}
                ])
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "stable-diffusion-xl-1024-v1-1");
        assert_eq!(models[0].name, "Stable Diffusion XL 1.1");
        assert_eq!(models[0].provider, "stability");
        // stable-diffusion 关键字 → Image
        assert_eq!(models[0].model_type, ModelType::Image);
        assert_eq!(models[0].description.as_deref(), Some("SDXL"));
        assert_eq!(models[0].created, Some(1700000000));
        assert!(!models[0].supports_streaming);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn list_models_filter_by_image_type() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/v1/engines/list")
            .with_status(200)
            .with_body(
                json!([
                    {"id": "stable-diffusion-xl-1024-v1-1", "name": "SDXL"},
                    {"id": "esrgan-v1-x2plus", "name": "ESRGAN"}
                ])
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        // 仅 stable-diffusion 命中 Image 关键字
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "stable-diffusion-xl-1024-v1-1");
    }

    #[tokio::test]
    async fn list_models_accepts_data_wrapper() {
        // 兼容 {"data": [...]} 包装格式
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/v1/engines/list")
            .with_status(200)
            .with_body(
                json!({
                    "data": [{"id": "stable-diffusion-xl-1024-v1-1", "name": "SDXL"}]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "stable-diffusion-xl-1024-v1-1");
    }

    #[tokio::test]
    async fn list_models_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/v1/engines/list")
            .with_status(401)
            .with_body(json!({"message": "bad key"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn list_models_empty_array() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/v1/engines/list")
            .with_status(200)
            .with_body(json!([]).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert!(models.is_empty());
    }

    // ============ 不支持的能力 ============

    #[tokio::test]
    async fn chat_returns_unsupported() {
        let adapter = make_adapter_no_server();
        let req = ChatRequest::builder("m", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn video_create_returns_unsupported() {
        let adapter = make_adapter_no_server();
        let req = VideoRequest::builder("m", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn video_poll_returns_unsupported() {
        let adapter = make_adapter_no_server();
        let err = adapter.video_poll("task-1", "m").await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ start / close ============

    #[tokio::test]
    async fn start_and_close_are_noops() {
        let mut adapter = make_adapter_no_server();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    // ============ map_api_error 单元测试 ============

    #[test]
    fn map_api_error_401_is_authentication() {
        let err = StabilityAdapter::map_api_error(401, "");
        match err {
            AibridgeError::Authentication { message } => assert!(message.contains("Stability")),
            _ => panic!("应为 Authentication"),
        }
    }

    #[test]
    fn map_api_error_429_is_rate_limit() {
        let err = StabilityAdapter::map_api_error(429, "");
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[test]
    fn map_api_error_400_is_validation_with_details() {
        let body = json!({"message": "invalid width"}).to_string();
        let err = StabilityAdapter::map_api_error(400, &body);
        match err {
            AibridgeError::Validation { message, details } => {
                assert_eq!(message, "invalid width");
                assert_eq!(details["message"], "invalid width");
            }
            _ => panic!("应为 Validation"),
        }
    }

    #[test]
    fn map_api_error_400_extracts_errors_array() {
        // Stability 校验错误数组格式
        let body =
            json!({"errors": [{"message": "field1 invalid"}, {"message": "field2 invalid"}]})
                .to_string();
        let err = StabilityAdapter::map_api_error(400, &body);
        match err {
            AibridgeError::Validation { message, .. } => {
                assert!(message.contains("field1 invalid"));
                assert!(message.contains("field2 invalid"));
            }
            _ => panic!("应为 Validation"),
        }
    }

    #[test]
    fn map_api_error_500_is_api() {
        let err = StabilityAdapter::map_api_error(500, "{\"message\":\"internal\"}");
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 500);
                assert_eq!(message, "internal");
            }
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_403_is_api() {
        let err = StabilityAdapter::map_api_error(403, "{\"message\":\"forbidden\"}");
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 403);
                assert_eq!(message, "forbidden");
            }
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_no_json_falls_back_to_http_status() {
        let err = StabilityAdapter::map_api_error(502, "Bad Gateway");
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("502")),
            _ => panic!("应为 Api"),
        }
    }

    // ============ resolve_dimensions 单元测试 ============

    #[test]
    fn resolve_dimensions_uses_width_height_when_both_present() {
        assert_eq!(resolve_dimensions(Some(512), Some(512), None), (512, 512));
        assert_eq!(resolve_dimensions(Some(1344), Some(768), None), (1344, 768));
    }

    #[test]
    fn resolve_dimensions_uses_size_string_when_no_width_height() {
        assert_eq!(
            resolve_dimensions(None, None, Some("1024x1024")),
            (1024, 1024)
        );
        assert_eq!(
            resolve_dimensions(None, None, Some("1344x768")),
            (1344, 768)
        );
    }

    #[test]
    fn resolve_dimensions_defaults_when_nothing_provided() {
        assert_eq!(resolve_dimensions(None, None, None), (1024, 1024));
    }

    #[test]
    fn resolve_dimensions_falls_back_when_size_invalid() {
        // size 字符串非法时回退到默认 1024x1024
        assert_eq!(resolve_dimensions(None, None, Some("abc")), (1024, 1024));
        assert_eq!(resolve_dimensions(None, None, Some("1024")), (1024, 1024));
    }

    #[test]
    fn resolve_dimensions_partial_width_uses_default_height() {
        assert_eq!(resolve_dimensions(Some(512), None, None), (512, 1024));
    }

    #[test]
    fn resolve_dimensions_partial_height_uses_default_width() {
        assert_eq!(resolve_dimensions(None, Some(512), None), (1024, 512));
    }

    // ============ parse_engines / parse_image_result 单元测试 ============

    #[test]
    fn parse_engines_top_level_array() {
        let value = json!([
            {"id": "stable-diffusion-xl-1024-v1-1", "name": "SDXL"},
            {"id": "stable-diffusion-3-medium", "name": "SD3"}
        ]);
        let models = StabilityAdapter::parse_engines(&value);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "stable-diffusion-xl-1024-v1-1");
        assert_eq!(models[0].model_type, ModelType::Image);
    }

    #[test]
    fn parse_engines_data_wrapper() {
        let value = json!({"data": [{"id": "stable-diffusion-xl-1024-v1-1"}]});
        let models = StabilityAdapter::parse_engines(&value);
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn parse_engines_empty_array() {
        let value = json!([]);
        let models = StabilityAdapter::parse_engines(&value);
        assert!(models.is_empty());
    }

    #[test]
    fn parse_engines_uses_id_as_name_when_missing() {
        let value = json!([{"id": "stable-diffusion-xl-1024-v1-1"}]);
        let models = StabilityAdapter::parse_engines(&value);
        assert_eq!(models[0].name, "stable-diffusion-xl-1024-v1-1");
    }

    #[test]
    fn parse_image_result_extracts_base64() {
        let value = json!({
            "artifacts": [{"base64": "aGk="}, {"base64": "eQ=="}]
        });
        let result = StabilityAdapter::parse_image_result(&value, "sdxl").unwrap();
        assert_eq!(result.data.len(), 2);
        assert_eq!(result.data[0].b64_json.as_deref(), Some("aGk="));
        assert_eq!(result.model, "sdxl");
    }

    #[test]
    fn parse_image_result_empty_artifacts_returns_error() {
        let value = json!({"artifacts": []});
        let result = StabilityAdapter::parse_image_result(&value, "sdxl");
        assert!(result.is_err());
    }

    #[test]
    fn parse_image_result_missing_artifacts_returns_error() {
        let value = json!({});
        let result = StabilityAdapter::parse_image_result(&value, "sdxl");
        assert!(result.is_err());
    }

    #[test]
    fn parse_image_result_filters_empty_base64() {
        // base64 为空的 artifact 被过滤；全部为空则返错
        let value = json!({"artifacts": [{"base64": ""}, {"base64": "aGk="}]});
        let result = StabilityAdapter::parse_image_result(&value, "sdxl").unwrap();
        assert_eq!(result.data.len(), 1);
        assert_eq!(result.data[0].b64_json.as_deref(), Some("aGk="));
    }

    // ============ build_generate_body 单元测试 ============

    #[test]
    fn build_generate_body_includes_required_fields() {
        let adapter = make_adapter_no_server();
        let req = ImageRequest::builder("sdxl", "a cat")
            .negative_prompt("blurry")
            .build();
        let body = adapter.build_generate_body(&req);
        // text_prompts 含正向 + 负向
        let prompts = body.get("text_prompts").and_then(|v| v.as_array()).unwrap();
        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[0]["text"], "a cat");
        assert_eq!(prompts[0]["weight"], 1);
        assert_eq!(prompts[1]["text"], "blurry");
        assert_eq!(prompts[1]["weight"], -1);
        // 默认尺寸
        assert_eq!(body["width"], 1024);
        assert_eq!(body["height"], 1024);
        // samples 默认 1
        assert_eq!(body["samples"], 1);
    }

    #[test]
    fn build_generate_body_merges_extra() {
        let adapter = make_adapter_no_server();
        let req = ImageRequest::builder("sdxl", "a cat")
            .extra("sampler", "K_DPMPP_2M")
            .build();
        let body = adapter.build_generate_body(&req);
        assert_eq!(body["sampler"], "K_DPMPP_2M");
    }
}
