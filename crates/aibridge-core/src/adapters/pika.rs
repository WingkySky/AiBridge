//! Pika 适配器（视频生成，独立协议）
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/pika.py`。
//!
//! Pika API 为**独立协议**（非 OpenAI 兼容），不复用 `OpenAiCompatAdapter`：
//! - 创建视频任务：`POST /generations`（请求体用 `prompt_text` 字段，图生视频用 `prompt_image`）
//! - 查询任务状态：`GET /generations/{task_id}`
//! - 认证：`Bearer <api_key>`
//! - 默认 Base URL：`https://api.pika.art/v1`
//!
//! ## 阶段范围
//! 阶段 2b 实现 video_create + video_poll + list_models（硬编码）。
//! 不支持的能力（chat / image / embed / audio）走 `Adapter` trait 默认实现返
//! `UnsupportedCapability`，与 Python v1 抛 `UnsupportedCapabilityError` 行为一致。

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::adapter::{Adapter, Capabilities, CapabilitySet};
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::common::{ModelInfo, ModelType, TaskStatus, VideoMode};
use crate::model::image::FileInput;
use crate::model::video::{VideoRequest, VideoStatus, VideoTask};
use crate::util;

/// Pika 默认 Base URL
///
/// 对应 Python v1 `DEFAULT_BASE_URL`（已含 `/v1` 前缀）。
pub const DEFAULT_PIKA_BASE_URL: &str = "https://api.pika.art/v1";

/// 默认视频宽度（与 `VideoRequest` 默认值一致，用于判断是否需要发送 width 字段）
const DEFAULT_WIDTH: u32 = 1280;
/// 默认视频高度（与 `VideoRequest` 默认值一致，用于判断是否需要发送 height 字段）
const DEFAULT_HEIGHT: u32 = 720;

// ==================== 能力集合构造 ====================

/// Pika 支持的能力集合
///
/// 对齐 Python v1 `PikaAdapter.supported_capabilities = ["video"]`。
/// Rust 用 `VideoGenerate` 表达视频生成能力（含 video_create + video_poll），
/// 并声明 text2video / image2video 两个子能力。
fn pika_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::VideoGenerate);
    caps.insert(Capabilities::VideoText2Video);
    caps.insert(Capabilities::VideoImage2Video);
    caps
}

// ==================== Pika 视频生成适配器 ====================

/// Pika 适配器
///
/// Pika 视频生成平台，支持文生视频与图生视频（Pika 1.0 / 2.0）。
/// 官方 API 文档：https://pika.art/
///
/// ## API 规范
/// - Base URL: `https://api.pika.art/v1`
/// - 创建生成: `POST /generations`
/// - 查询状态: `GET /generations/{id}`
/// - 认证: `Authorization: Bearer {api_key}`
///
/// ## 阶段范围
/// 阶段 2b 实现 video_create + video_poll + list_models（硬编码）。
pub struct PikaAdapter {
    /// HTTP 客户端（封装 reqwest，含连接池与超时）
    http: HttpClient,
    /// Provider 配置（api_key / base_url / timeout 等）
    config: ProviderConfig,
    /// 实际 base_url（已合并 config.base_url 与默认值）
    base_url: String,
    /// 支持的能力集合
    capabilities: CapabilitySet,
}

impl PikaAdapter {
    /// 创建 Pika 适配器
    ///
    /// `config.base_url` 为空时用 [`DEFAULT_PIKA_BASE_URL`] 兜底。
    /// `config.api_key` 为空时不在此处报错（由上层按 `requires_api_key` 校验）。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_PIKA_BASE_URL.to_string());

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
            capabilities: pika_capabilities(),
        })
    }

    /// 用显式 HttpClient 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_http(http: HttpClient, config: ProviderConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_PIKA_BASE_URL.to_string());
        Self {
            http,
            config,
            base_url,
            capabilities: pika_capabilities(),
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
                capability: format!("{} (provider: pika)", cap.as_str()),
            })
        }
    }

    /// 发送带 Bearer 认证的 POST JSON 请求，并用 Pika 错误映射处理响应
    async fn post_authed_json(&self, path: &str, body: &Value) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .http
            .inner()
            .post(&url)
            .bearer_auth(self.api_key())
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
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

    /// 发送带 Bearer 认证的 GET 请求，并用 Pika 错误映射处理响应
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

    /// 构造 Pika `/generations` 请求体
    ///
    /// 移植自 Python v1 `video_create`：
    /// - `model` / `prompt_text` 必选
    /// - image2video 模式：`reference_images[0]` → `prompt_image`（优先），`first_frame` 次之
    /// - `width` / `height` 仅在非默认值时发送（对齐 Python「显式传才发送」语义）
    /// - `seed` / `aspect_ratio` / `duration` / `negative_prompt_text` 可选透传
    /// - `extra` 字段合并到顶层（透传厂商特有参数）
    fn build_generations_body(&self, req: &VideoRequest) -> Value {
        let mut body = json!({
            "model": req.model,
            "prompt_text": req.prompt,
        });

        // image2video 模式：reference_images[0] 优先作为 prompt_image，first_frame 次之
        let is_image2video = matches!(req.mode, VideoMode::Image2Video);
        if is_image2video {
            if let Some(url) = req.reference_images.first().and_then(file_input_to_url) {
                body["prompt_image"] = json!(url);
            } else if let Some(ff) = req.first_frame.as_ref().and_then(file_input_to_url) {
                body["prompt_image"] = json!(ff);
            }
        }

        // width / height 仅在非默认值时发送（默认 1280x720 不发送，让 Pika 用自身默认或 aspect_ratio）
        if req.width != DEFAULT_WIDTH {
            body["width"] = json!(req.width);
        }
        if req.height != DEFAULT_HEIGHT {
            body["height"] = json!(req.height);
        }
        if let Some(seed) = req.seed {
            body["seed"] = json!(seed);
        }
        if let Some(ar) = &req.aspect_ratio {
            body["aspect_ratio"] = json!(ar);
        }
        if let Some(d) = req.duration {
            body["duration"] = json!(d);
        }
        if let Some(np) = &req.negative_prompt {
            body["negative_prompt_text"] = json!(np);
        }

        // extra 透传（合并到顶层）
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        body
    }

    /// 解析 Pika `/generations` 创建响应 → VideoTask
    ///
    /// Pika 创建响应任务 ID 可能在 `id` / `generation_id` / `taskId` 字段，
    /// 均缺失时回退到生成的 `vid_` 前缀 ID（与 Python v1 一致）。
    fn parse_video_task(value: &Value, model: &str) -> Result<VideoTask> {
        let task_id = value
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| {
                value
                    .get("generation_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            })
            .or_else(|| {
                value
                    .get("taskId")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| util::generate_id("vid"));
        let raw_status = value
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("pending");
        let created_at = value
            .get("created_at")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(util::current_timestamp);
        Ok(VideoTask {
            task_id,
            model: model.to_string(),
            status: map_pika_status(raw_status),
            created_at,
        })
    }

    /// 解析 Pika `/generations/{id}` 查询响应 → VideoStatus
    ///
    /// 移植自 Python v1 `video_poll`：
    /// - 视频 URL 兼容多路径：`video_url` / `url` / `output.video_url` / `output.url` / `results[0].url`
    /// - 错误信息兼容：`error` / `error_message` / `failure_reason`
    /// - `progress` / `created_at` / `updated_at` 透传（数字形式）
    fn parse_video_status(value: &Value, task_id: &str) -> VideoStatus {
        let raw_status = value
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("pending");
        let status = map_pika_status(raw_status);

        // 视频 URL：多路径兼容（无条件提取，与 Python v1 一致）
        let video_url = value
            .get("video_url")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| value.get("url").and_then(|v| v.as_str()).map(str::to_owned))
            .or_else(|| {
                value
                    .get("output")
                    .and_then(|o| o.get("video_url"))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            })
            .or_else(|| {
                value
                    .get("output")
                    .and_then(|o| o.get("url"))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            })
            .or_else(|| {
                value
                    .get("results")
                    .and_then(|r| r.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|f| f.get("url"))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            });

        // 错误信息：多字段兼容（无条件提取，与 Python v1 一致）
        let error = value
            .get("error")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| {
                value
                    .get("error_message")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            })
            .or_else(|| {
                value
                    .get("failure_reason")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            });

        let progress = value
            .get("progress")
            .and_then(|v| v.as_u64())
            .map(|p| p as u32);
        let created_at = value.get("created_at").and_then(|v| v.as_u64());
        let updated_at = value.get("updated_at").and_then(|v| v.as_u64());

        VideoStatus {
            task_id: task_id.to_string(),
            status,
            video_url,
            progress,
            error,
            created_at,
            updated_at,
        }
    }

    /// 将 Pika API 错误响应映射为 AibridgeError
    ///
    /// 移植自 Python v1 `_handle_pika_error` 并按阶段 2b 统一错误映射要求调整：
    /// - 401 → Authentication（"Invalid Pika API key"）
    /// - 429 → RateLimit（"Pika rate limit exceeded"）
    /// - 404 → ModelNotFound（Pika 资源/任务不存在）
    /// - 400 → Validation（请求参数校验错误）
    /// - 其余 ≥400 → Api（提取 error.message / message / error / detail，回退 `HTTP {status}`）
    pub fn map_api_error(status: u16, body: &str) -> AibridgeError {
        match status {
            401 => AibridgeError::Authentication {
                message: "Invalid Pika API key".to_string(),
            },
            429 => AibridgeError::RateLimit {
                message: "Pika rate limit exceeded".to_string(),
                retry_after: None,
            },
            404 => AibridgeError::ModelNotFound {
                model: "Pika generation or resource not found".to_string(),
            },
            400 => AibridgeError::validation_with_details(
                parse_error_message(body, status),
                serde_json::from_str(body).unwrap_or(serde_json::Value::Null),
            ),
            _ => {
                let message = parse_error_message(body, status);
                AibridgeError::Api { status, message }
            }
        }
    }
}

#[async_trait]
impl Adapter for PikaAdapter {
    fn provider_type(&self) -> &str {
        "pika"
    }

    fn provider_name(&self) -> &str {
        "Pika"
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

    /// 创建视频生成任务：`POST /generations`
    async fn video_create(&self, req: VideoRequest) -> Result<VideoTask> {
        self.ensure_capability(Capabilities::VideoGenerate)?;
        let body = self.build_generations_body(&req);
        let value = self.post_authed_json("generations", &body).await?;
        Self::parse_video_task(&value, &req.model)
    }

    /// 查询视频任务状态：`GET /generations/{task_id}`
    async fn video_poll(&self, task_id: &str, _model: &str) -> Result<VideoStatus> {
        self.ensure_capability(Capabilities::VideoGenerate)?;
        let path = format!("generations/{task_id}");
        let value = self.get_authed_json(&path).await?;
        Ok(Self::parse_video_status(&value, task_id))
    }

    /// 模型列表（硬编码）
    ///
    /// Pika 无标准 `/models` 端点，暂保留硬编码列表（与 Python v1 一致）。
    /// 含 pika-1.0 / pika-2 两个模型。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = pika_hardcoded_models();
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }

    // chat / chat_stream / image / embed / audio 走 trait 默认实现返 UnsupportedCapability，
    // 与 Python v1 抛 UnsupportedCapabilityError 行为一致。
}

// ==================== 内部：辅助函数 ====================

/// 映射 Pika 状态字符串到统一 TaskStatus
///
/// 移植自 Python v1 `_map_pika_status`：
/// - pending / queued / in_queue → Pending
/// - processing / in_progress / generating → Processing
/// - completed / finished / succeeded / success → Success
/// - failed / error / failure / cancelled → Failed
/// - 未知 → Pending（与 Python 默认值一致）
fn map_pika_status(raw: &str) -> TaskStatus {
    match raw.to_lowercase().as_str() {
        "pending" | "queued" | "in_queue" => TaskStatus::Pending,
        "processing" | "in_progress" | "generating" => TaskStatus::Processing,
        "completed" | "finished" | "succeeded" | "success" => TaskStatus::Success,
        "failed" | "error" | "failure" | "cancelled" => TaskStatus::Failed,
        _ => TaskStatus::Pending,
    }
}

/// 从 FileInput 提取 URL 字符串
///
/// Pika 的 `prompt_image` 字段接受 URL 或 base64 字符串：
/// - `Url(s)` / `Base64(s)` → `Some(s)`
/// - `Path(_)` / `Bytes(_)` → `None`（需调用方先上传为可访问 URL）
fn file_input_to_url(input: &FileInput) -> Option<String> {
    match input {
        FileInput::Url(s) | FileInput::Base64(s) => Some(s.clone()),
        FileInput::Path(_) | FileInput::Bytes(_) => None,
    }
}

/// 解析错误体中的 message 字段
///
/// 通用错误消息提取，兼容多种错误体结构：
/// - `{"error": {"message": "..."}}`（OpenAI 风格）
/// - `{"error": "..."}`（顶层 error 字符串）
/// - `{"message": "..."}`（顶层 message）
/// - `{"detail": "..."}`（顶层 detail）
///
/// 解析失败时回退到 `HTTP {status}` 字符串。
fn parse_error_message(body: &str, status: u16) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        // error.message（OpenAI 风格）
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
        // 顶层 detail
        if let Some(msg) = v.get("detail").and_then(|m| m.as_str()) {
            return msg.to_string();
        }
    }
    if body.trim().is_empty() {
        format!("HTTP {status}")
    } else {
        format!("HTTP {status}: {body}")
    }
}

/// Pika 硬编码模型列表
///
/// 对应 Python v1 `PikaAdapter.list_models`。
/// 注意：该 Provider 无标准 `/models` 端点，暂保留硬编码列表。
fn pika_hardcoded_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "pika-1.0".into(),
            name: "Pika 1.0".into(),
            model_type: ModelType::Video,
            provider: "pika".into(),
            capabilities: vec!["text2video".into(), "image2video".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Pika 1.0 视频生成模型".into()),
            created: None,
        },
        ModelInfo {
            id: "pika-2".into(),
            name: "Pika 2.0".into(),
            model_type: ModelType::Video,
            provider: "pika".into(),
            capabilities: vec!["text2video".into(), "image2video".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Pika 2.0 最新视频生成模型".into()),
            created: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::chat::ChatRequest;
    use crate::model::image::ImageRequest;
    use mockito::Server;
    use serde_json::json;

    // ==================== 通用测试辅助 ====================

    /// 构造测试用 PikaAdapter（指向 mockito server）
    fn make_pika(server: &Server) -> PikaAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("pika", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        PikaAdapter::with_http(http, config)
    }

    /// 构造不指向任何 server 的 PikaAdapter（用于不发请求的元信息/能力测试）
    fn make_pika_no_server() -> PikaAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(DEFAULT_PIKA_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("pika", opts);
        PikaAdapter::new(config).expect("PikaAdapter 构造应成功")
    }

    // ============ 元信息 ============

    #[test]
    fn pika_provider_type_and_name_match_python() {
        let adapter = make_pika_no_server();
        assert_eq!(adapter.provider_type(), "pika");
        assert_eq!(adapter.provider_name(), "Pika");
    }

    #[test]
    fn pika_requires_api_key_is_true() {
        let adapter = make_pika_no_server();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn pika_capabilities_contains_only_video() {
        let adapter = make_pika_no_server();
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::VideoGenerate));
        assert!(caps.contains(&Capabilities::VideoText2Video));
        assert!(caps.contains(&Capabilities::VideoImage2Video));
        // chat / image 不声明
        assert!(!caps.contains(&Capabilities::Chat));
        assert!(!caps.contains(&Capabilities::ImageGenerate));
    }

    #[test]
    fn pika_base_url_defaults_when_missing() {
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("pika", opts);
        let adapter = PikaAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), DEFAULT_PIKA_BASE_URL);
    }

    #[test]
    fn pika_base_url_uses_config_when_provided() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://custom.pika-proxy.com/v1")
            .build();
        let config = ProviderConfig::from_options("pika", opts);
        let adapter = PikaAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), "https://custom.pika-proxy.com/v1");
    }

    #[test]
    fn pika_base_url_ignores_empty_string() {
        // 空白 base_url 应回退到默认值
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("   ")
            .build();
        let config = ProviderConfig::from_options("pika", opts);
        let adapter = PikaAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), DEFAULT_PIKA_BASE_URL);
    }

    // ============ video_create 正常路径 ============

    #[tokio::test]
    async fn pika_video_create_success_returns_task() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "gen-abc",
            "status": "pending",
            "created_at": 1700000000
        });
        let mock = server
            .mock("POST", "/generations")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat running")
            .aspect_ratio("16:9")
            .duration(5)
            .build();
        let task = adapter
            .video_create(req)
            .await
            .expect("video_create 应成功");

        assert_eq!(task.task_id, "gen-abc");
        assert_eq!(task.model, "pika-1.0");
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.created_at, 1700000000);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn pika_video_create_sends_model_and_prompt_text() {
        // 验证请求体用 prompt_text 字段（非 prompt），且默认不发送 width/height
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "pika-1.0",
                "prompt_text": "a cat"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat").build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn pika_video_create_passes_optional_params() {
        // aspect_ratio / duration / seed / negative_prompt 透传
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "aspect_ratio": "16:9",
                "duration": 5,
                "seed": 42,
                "negative_prompt_text": "blurry"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat")
            .aspect_ratio("16:9")
            .duration(5)
            .seed(42)
            .negative_prompt("blurry")
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn pika_video_create_passes_width_height_when_non_default() {
        // 非默认 width/height 才发送
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "width": 1920,
                "height": 1080
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat")
            .width(1920)
            .height(1080)
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn pika_video_create_default_width_height_not_sent() {
        // 默认 1280x720 不应出现在请求体
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generations")
            .match_body(mockito::Matcher::Json(json!({
                "model": "pika-1.0",
                "prompt_text": "a cat"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat").build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn pika_video_create_image2video_with_reference_image() {
        // image2video 模式：reference_images[0] → prompt_image
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "prompt_image": "https://example.com/a.png"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "animate this")
            .mode(VideoMode::Image2Video)
            .reference_images(vec![FileInput::url("https://example.com/a.png")])
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn pika_video_create_image2video_with_first_frame() {
        // reference_images 为空时，first_frame 作为 prompt_image 后备
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "prompt_image": "https://example.com/first.png"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "animate this")
            .mode(VideoMode::Image2Video)
            .first_frame(FileInput::url("https://example.com/first.png"))
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn pika_video_create_text2video_does_not_send_prompt_image() {
        // text2video 模式即使有 reference_images 也不发送 prompt_image
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generations")
            .match_body(mockito::Matcher::Json(json!({
                "model": "pika-1.0",
                "prompt_text": "a cat"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat")
            .mode(VideoMode::Text2Video)
            .reference_images(vec![FileInput::url("https://example.com/a.png")])
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn pika_video_create_passes_extra_params() {
        // extra 字段透传到顶层
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "custom_param": "custom_value",
                "options": {"frame_rate": 30}
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat")
            .extra("custom_param", "custom_value")
            .extra("options", json!({"frame_rate": 30}))
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn pika_video_create_accepts_generation_id_field() {
        // 任务 ID 在 generation_id 字段
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(200)
            .with_body(json!({"generation_id": "gen-xyz", "status": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat").build();
        let task = adapter.video_create(req).await.unwrap();
        assert_eq!(task.task_id, "gen-xyz");
        assert_eq!(task.status, TaskStatus::Pending);
    }

    #[tokio::test]
    async fn pika_video_create_accepts_task_id_field() {
        // 任务 ID 在 taskId 字段
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(200)
            .with_body(json!({"taskId": "task-001", "status": "processing"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat").build();
        let task = adapter.video_create(req).await.unwrap();
        assert_eq!(task.task_id, "task-001");
        assert_eq!(task.status, TaskStatus::Processing);
    }

    #[tokio::test]
    async fn pika_video_create_uses_generated_id_when_missing() {
        // 响应缺任务 ID 时，回退到生成的 vid_ ID
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(200)
            .with_body(json!({"status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat").build();
        let task = adapter.video_create(req).await.unwrap();
        assert!(task.task_id.starts_with("vid_"));
    }

    // ============ video_create 错误路径 ============

    #[tokio::test]
    async fn pika_video_create_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(401)
            .with_body(json!({"error": {"message": "invalid key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn pika_video_create_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(429)
            .with_body(json!({"message": "slow down"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn pika_video_create_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(404)
            .with_body(json!({"error": "not found"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn pika_video_create_error_400_returns_validation() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(400)
            .with_body(json!({"error": {"message": "invalid prompt"}}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Validation { message, .. } => {
                assert!(message.contains("invalid prompt"));
            }
            _ => panic!("应为 Validation"),
        }
    }

    #[tokio::test]
    async fn pika_video_create_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(500)
            .with_body(json!({"error": {"message": "internal"}}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 500);
                assert!(message.contains("internal"));
            }
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn pika_video_create_error_no_json_body_falls_back() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(502)
            .with_body("Bad Gateway")
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let req = VideoRequest::builder("pika-1.0", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("502")),
            _ => panic!("应为 Api"),
        }
    }

    // ============ video_poll 正常路径 ============

    #[tokio::test]
    async fn pika_video_poll_success_returns_video_url() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "gen-abc",
            "status": "completed",
            "video_url": "https://example.com/video.mp4",
            "progress": 100,
            "created_at": 1700000000,
            "updated_at": 1700000100
        });
        let mock = server
            .mock("GET", "/generations/gen-abc")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let status = adapter
            .video_poll("gen-abc", "pika-1.0")
            .await
            .expect("video_poll 应成功");

        assert_eq!(status.task_id, "gen-abc");
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://example.com/video.mp4")
        );
        assert_eq!(status.progress, Some(100));
        assert_eq!(status.created_at, Some(1700000000));
        assert_eq!(status.updated_at, Some(1700000100));
        assert!(status.error.is_none());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn pika_video_poll_processing_status() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/generations/gen-1")
            .with_status(200)
            .with_body(
                json!({
                    "id": "gen-1",
                    "status": "processing",
                    "progress": 45
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let status = adapter.video_poll("gen-1", "pika-1.0").await.unwrap();
        assert_eq!(status.status, TaskStatus::Processing);
        assert_eq!(status.progress, Some(45));
    }

    #[tokio::test]
    async fn pika_video_poll_failed_returns_error() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/generations/gen-2")
            .with_status(200)
            .with_body(
                json!({
                    "id": "gen-2",
                    "status": "failed",
                    "failure_reason": "content policy violation"
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let status = adapter.video_poll("gen-2", "pika-1.0").await.unwrap();
        assert_eq!(status.status, TaskStatus::Failed);
        assert_eq!(status.error.as_deref(), Some("content policy violation"));
    }

    #[tokio::test]
    async fn pika_video_poll_failed_extracts_error_message_field() {
        // error_message 字段
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/generations/gen-3")
            .with_status(200)
            .with_body(
                json!({
                    "id": "gen-3",
                    "status": "error",
                    "error_message": "internal error"
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let status = adapter.video_poll("gen-3", "pika-1.0").await.unwrap();
        assert_eq!(status.status, TaskStatus::Failed);
        assert_eq!(status.error.as_deref(), Some("internal error"));
    }

    #[tokio::test]
    async fn pika_video_poll_compatible_url_paths() {
        // 兼容 output.video_url / output.url / results[0].url / 顶层 url
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/generations/gen-4")
            .with_status(200)
            .with_body(
                json!({
                    "id": "gen-4",
                    "status": "finished",
                    "output": {"video_url": "https://legacy.example.com/v.mp4"}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let status = adapter.video_poll("gen-4", "pika-1.0").await.unwrap();
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://legacy.example.com/v.mp4")
        );
    }

    #[tokio::test]
    async fn pika_video_poll_url_from_results_array() {
        // results[0].url 路径
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/generations/gen-5")
            .with_status(200)
            .with_body(
                json!({
                    "id": "gen-5",
                    "status": "succeeded",
                    "results": [{"url": "https://results.example.com/v.mp4"}]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let status = adapter.video_poll("gen-5", "pika-1.0").await.unwrap();
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://results.example.com/v.mp4")
        );
    }

    #[tokio::test]
    async fn pika_video_poll_queued_status_maps_to_pending() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/generations/gen-6")
            .with_status(200)
            .with_body(json!({"id": "gen-6", "status": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let status = adapter.video_poll("gen-6", "pika-1.0").await.unwrap();
        assert_eq!(status.status, TaskStatus::Pending);
    }

    #[tokio::test]
    async fn pika_video_poll_cancelled_maps_to_failed() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/generations/gen-7")
            .with_status(200)
            .with_body(
                json!({"id": "gen-7", "status": "cancelled", "error": "user cancelled"})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let status = adapter.video_poll("gen-7", "pika-1.0").await.unwrap();
        assert_eq!(status.status, TaskStatus::Failed);
        assert_eq!(status.error.as_deref(), Some("user cancelled"));
    }

    // ============ video_poll 错误路径 ============

    #[tokio::test]
    async fn pika_video_poll_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/generations/gen-x")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let err = adapter.video_poll("gen-x", "pika-1.0").await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn pika_video_poll_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/generations/nonexistent")
            .with_status(404)
            .with_body(json!({"error": "not found"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let err = adapter
            .video_poll("nonexistent", "pika-1.0")
            .await
            .unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn pika_video_poll_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/generations/gen-y")
            .with_status(429)
            .with_body(json!({"message": "slow down"}).to_string())
            .create_async()
            .await;

        let adapter = make_pika(&server);
        let err = adapter.video_poll("gen-y", "pika-1.0").await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    // ============ 不支持的能力 ============

    #[tokio::test]
    async fn pika_chat_returns_unsupported() {
        let adapter = make_pika_no_server();
        let req = ChatRequest::builder("pika-1.0", vec![]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn pika_image_generate_returns_unsupported() {
        let adapter = make_pika_no_server();
        let req = ImageRequest::builder("pika-1.0", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn pika_embed_returns_unsupported() {
        let adapter = make_pika_no_server();
        let req = crate::model::options::EmbedRequest {
            model: "pika-1.0".into(),
            input: crate::model::options::EmbedInput::Single("hi".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: std::collections::HashMap::new(),
        };
        let err = adapter.embed(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ list_models ============

    #[tokio::test]
    async fn pika_list_models_returns_hardcoded() {
        let adapter = make_pika_no_server();
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "pika-1.0");
        assert_eq!(models[0].provider, "pika");
        assert_eq!(models[0].model_type, ModelType::Video);
        assert_eq!(models[1].id, "pika-2");
    }

    #[tokio::test]
    async fn pika_list_models_filter_by_video_type() {
        let adapter = make_pika_no_server();
        let videos = adapter.list_models(Some(ModelType::Video)).await.unwrap();
        assert_eq!(videos.len(), 2);
        assert!(videos.iter().all(|m| m.model_type == ModelType::Video));
    }

    #[tokio::test]
    async fn pika_list_models_filter_by_image_returns_empty() {
        let adapter = make_pika_no_server();
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert!(images.is_empty());
    }

    // ============ start / close ============

    #[tokio::test]
    async fn pika_start_and_close_are_noops() {
        let mut adapter = make_pika_no_server();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    // ============ 错误映射单元测试 ============

    #[test]
    fn map_api_error_401_is_authentication() {
        let err = PikaAdapter::map_api_error(401, "{\"error\":{\"message\":\"bad\"}}");
        match err {
            AibridgeError::Authentication { message } => assert!(message.contains("Pika")),
            _ => panic!("应为 Authentication"),
        }
    }

    #[test]
    fn map_api_error_429_is_rate_limit() {
        let err = PikaAdapter::map_api_error(429, "{}");
        match err {
            AibridgeError::RateLimit {
                message,
                retry_after,
            } => {
                assert!(message.contains("Pika"));
                assert!(retry_after.is_none());
            }
            _ => panic!("应为 RateLimit"),
        }
    }

    #[test]
    fn map_api_error_404_is_model_not_found() {
        let err = PikaAdapter::map_api_error(404, "{}");
        match err {
            AibridgeError::ModelNotFound { model } => assert!(model.contains("Pika")),
            _ => panic!("应为 ModelNotFound"),
        }
    }

    #[test]
    fn map_api_error_400_is_validation() {
        let err = PikaAdapter::map_api_error(400, "{\"error\":{\"message\":\"bad param\"}}");
        match err {
            AibridgeError::Validation { message, .. } => assert_eq!(message, "bad param"),
            _ => panic!("应为 Validation"),
        }
    }

    #[test]
    fn map_api_error_500_extracts_message() {
        let err = PikaAdapter::map_api_error(500, "{\"error\":{\"message\":\"internal\"}}");
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 500);
                assert_eq!(message, "internal");
            }
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_extracts_top_level_message() {
        let err = PikaAdapter::map_api_error(503, "{\"message\":\"unavailable\"}");
        match err {
            AibridgeError::Api { message, .. } => assert_eq!(message, "unavailable"),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_extracts_detail_field() {
        let err = PikaAdapter::map_api_error(422, "{\"detail\":\"validation failed\"}");
        match err {
            AibridgeError::Api { message, .. } => assert_eq!(message, "validation failed"),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_no_json_falls_back_to_http_status() {
        let err = PikaAdapter::map_api_error(502, "Bad Gateway");
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("502")),
            _ => panic!("应为 Api"),
        }
    }

    // ============ map_pika_status 单元测试 ============

    #[test]
    fn map_pika_status_pending_variants() {
        assert_eq!(map_pika_status("pending"), TaskStatus::Pending);
        assert_eq!(map_pika_status("queued"), TaskStatus::Pending);
        assert_eq!(map_pika_status("in_queue"), TaskStatus::Pending);
    }

    #[test]
    fn map_pika_status_processing_variants() {
        assert_eq!(map_pika_status("processing"), TaskStatus::Processing);
        assert_eq!(map_pika_status("in_progress"), TaskStatus::Processing);
        assert_eq!(map_pika_status("generating"), TaskStatus::Processing);
    }

    #[test]
    fn map_pika_status_success_variants() {
        assert_eq!(map_pika_status("completed"), TaskStatus::Success);
        assert_eq!(map_pika_status("finished"), TaskStatus::Success);
        assert_eq!(map_pika_status("succeeded"), TaskStatus::Success);
        assert_eq!(map_pika_status("success"), TaskStatus::Success);
    }

    #[test]
    fn map_pika_status_failed_variants() {
        assert_eq!(map_pika_status("failed"), TaskStatus::Failed);
        assert_eq!(map_pika_status("error"), TaskStatus::Failed);
        assert_eq!(map_pika_status("failure"), TaskStatus::Failed);
        assert_eq!(map_pika_status("cancelled"), TaskStatus::Failed);
    }

    #[test]
    fn map_pika_status_case_insensitive() {
        assert_eq!(map_pika_status("PENDING"), TaskStatus::Pending);
        assert_eq!(map_pika_status("Completed"), TaskStatus::Success);
        assert_eq!(map_pika_status("FAILED"), TaskStatus::Failed);
    }

    #[test]
    fn map_pika_status_unknown_defaults_to_pending() {
        assert_eq!(map_pika_status("unknown_state"), TaskStatus::Pending);
        assert_eq!(map_pika_status(""), TaskStatus::Pending);
    }

    // ============ file_input_to_url 单元测试 ============

    #[test]
    fn file_input_to_url_returns_url_for_url_variant() {
        let f = FileInput::url("https://example.com/x.png");
        assert_eq!(
            file_input_to_url(&f),
            Some("https://example.com/x.png".to_string())
        );
    }

    #[test]
    fn file_input_to_url_returns_base64_for_base64_variant() {
        let f = FileInput::base64("aGVsbG8=");
        assert_eq!(file_input_to_url(&f), Some("aGVsbG8=".to_string()));
    }

    #[test]
    fn file_input_to_url_returns_none_for_path_and_bytes() {
        assert_eq!(file_input_to_url(&FileInput::path("/tmp/x")), None);
        assert_eq!(file_input_to_url(&FileInput::bytes(vec![1, 2])), None);
    }

    // ============ build_generations_body 单元测试 ============

    #[test]
    fn build_generations_body_includes_model_and_prompt_text() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("pika", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = PikaAdapter::with_http(http, config);
        let req = VideoRequest::builder("pika-1.0", "a cat").build();
        let body = adapter.build_generations_body(&req);
        assert_eq!(body["model"], "pika-1.0");
        assert_eq!(body["prompt_text"], "a cat");
        // 默认 width/height 不发送
        assert!(body.get("width").is_none());
        assert!(body.get("height").is_none());
        // 无 prompt_image
        assert!(body.get("prompt_image").is_none());
    }

    #[test]
    fn build_generations_body_image2video_sets_prompt_image() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("pika", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = PikaAdapter::with_http(http, config);
        let req = VideoRequest::builder("pika-1.0", "animate")
            .mode(VideoMode::Image2Video)
            .reference_images(vec![FileInput::url("https://example.com/a.png")])
            .build();
        let body = adapter.build_generations_body(&req);
        assert_eq!(body["prompt_image"], "https://example.com/a.png");
    }

    #[test]
    fn build_generations_body_passes_optional_fields() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("pika", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = PikaAdapter::with_http(http, config);
        let req = VideoRequest::builder("pika-2", "a cat")
            .width(1920)
            .height(1080)
            .aspect_ratio("16:9")
            .duration(5)
            .seed(42)
            .negative_prompt("blurry")
            .build();
        let body = adapter.build_generations_body(&req);
        assert_eq!(body["width"], 1920);
        assert_eq!(body["height"], 1080);
        assert_eq!(body["aspect_ratio"], "16:9");
        assert_eq!(body["duration"], 5);
        assert_eq!(body["seed"], 42);
        assert_eq!(body["negative_prompt_text"], "blurry");
    }

    #[test]
    fn build_generations_body_extra_passthrough() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("pika", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = PikaAdapter::with_http(http, config);
        let req = VideoRequest::builder("pika-1.0", "a cat")
            .extra("custom_param", "custom_value")
            .build();
        let body = adapter.build_generations_body(&req);
        assert_eq!(body["custom_param"], "custom_value");
    }
}
