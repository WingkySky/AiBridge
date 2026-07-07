//! Runway 视频生成适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/runway.py`。
//!
//! Runway 为**独立协议**（非 OpenAI 兼容），不复用 `OpenAiCompatAdapter`：
//! - 文生视频：`POST /text_to_video`
//! - 图生视频：`POST /image_to_video`（需 `promptImage`）
//! - 查询任务：`GET /assets/{id}`
//! - 认证：`Authorization: Bearer {api_key}`
//! - 默认 Base URL：`https://api.runwayml.com/v1`
//!
//! ## 阶段范围
//! 阶段 2b 实现 video_create + video_poll + list_models（硬编码）。
//! 不支持的能力（chat / image / embed / transcribe / speech）走 `Adapter` trait
//! 默认实现返 `UnsupportedCapability`，与 Python v1 抛 `UnsupportedCapabilityError` 行为一致。

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

/// Runway 默认 Base URL
///
/// 对应 Python v1 `DEFAULT_BASE_URL`（已含 `/v1` 前缀）。
pub const DEFAULT_RUNWAY_BASE_URL: &str = "https://api.runwayml.com/v1";

/// Runway 支持的能力集合
///
/// 对齐 Python v1 `RunwayAdapter.supported_capabilities = ["video"]`。
/// Rust 用 `VideoGenerate` 表达视频生成能力（含 video_create + video_poll），
/// 并声明 text2video / image2video 两个子能力。
fn runway_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::VideoGenerate);
    caps.insert(Capabilities::VideoText2Video);
    caps.insert(Capabilities::VideoImage2Video);
    caps
}

/// Runway 适配器
///
/// Runway Gen-3 Alpha / Gen-3 Turbo 视频生成平台，支持文生视频与图生视频。
/// 官方 API 文档：https://docs.dev.runwayml.ai/
///
/// ## API 规范
/// - Base URL: `https://api.runwayml.com/v1`
/// - 文生视频: `POST /text_to_video`（body 含 `prompt`）
/// - 图生视频: `POST /image_to_video`（body 含 `promptImage` + `promptText`）
/// - 查询任务: `GET /assets/{id}`
/// - 认证: Bearer Token
pub struct RunwayAdapter {
    /// HTTP 客户端（封装 reqwest，含连接池与超时）
    http: HttpClient,
    /// Provider 配置（api_key / base_url / timeout 等）
    config: ProviderConfig,
    /// 实际 base_url（已合并 config.base_url 与默认值）
    base_url: String,
    /// 支持的能力集合
    capabilities: CapabilitySet,
}

impl RunwayAdapter {
    /// 创建 Runway 适配器
    ///
    /// `config.base_url` 为空时用 [`DEFAULT_RUNWAY_BASE_URL`] 兜底。
    /// `config.api_key` 为空时不在此处报错（由上层按 `requires_api_key` 校验）。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_RUNWAY_BASE_URL.to_string());

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
            capabilities: runway_capabilities(),
        })
    }

    /// 用显式 HttpClient 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_http(http: HttpClient, config: ProviderConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_RUNWAY_BASE_URL.to_string());
        Self {
            http,
            config,
            base_url,
            capabilities: runway_capabilities(),
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
                capability: format!("{} (provider: runway)", cap.as_str()),
            })
        }
    }

    /// 发送带 Bearer 认证的 POST JSON 请求，并用 Runway 错误映射处理响应
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

    /// 发送带 Bearer 认证的 GET 请求，并用 Runway 错误映射处理响应
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

    /// 构造 Runway 视频创建请求体，并返回对应端点
    ///
    /// 移植自 Python v1 `video_create`：
    /// - `image2video` 模式且有参考图 → `POST /image_to_video`，body 含 `promptImage` + `promptText`
    /// - 其余 → `POST /text_to_video`，body 含 `prompt`
    /// - 参考图优先 `reference_images[0]`，回退 `first_frame`（Runway 仅支持单张首帧）
    /// - 可选参数：width / height / seed / motion / cameraMotion / aspectRatio
    /// - extra 字段合并到顶层（透传厂商特有参数）
    fn build_create_body(&self, req: &VideoRequest) -> (&'static str, Value) {
        // 默认模型 gen-3（与 list_models 一致）
        let model = if req.model.is_empty() {
            "gen-3".to_string()
        } else {
            req.model.clone()
        };

        // 图生视频：优先 reference_images[0]，回退 first_frame
        let prompt_image = req
            .reference_images
            .first()
            .map(file_input_to_url)
            .filter(|s| !s.is_empty())
            .or_else(|| req.first_frame.as_ref().map(file_input_to_url))
            .filter(|s| !s.is_empty());

        let is_image2video = matches!(req.mode, VideoMode::Image2Video);

        let (endpoint, mut body) = match (is_image2video, prompt_image) {
            // 图生视频：Runway 用 promptImage + promptText
            (true, Some(img)) => {
                let b = json!({
                    "model": model,
                    "promptImage": img,
                    "promptText": req.prompt,
                });
                ("image_to_video", b)
            }
            // 文生视频：Runway 用 prompt
            _ => {
                let b = json!({
                    "model": model,
                    "prompt": req.prompt,
                });
                ("text_to_video", b)
            }
        };

        // 可选参数（移植自 Python v1：kwargs 有则加）
        body["width"] = json!(req.width);
        body["height"] = json!(req.height);
        if let Some(seed) = req.seed {
            body["seed"] = json!(seed);
        }
        if let Some(motion) = req.motion_strength {
            body["motion"] = json!(motion);
        }
        if let Some(cm) = &req.camera_motion {
            body["cameraMotion"] = json!(cm);
        }
        if let Some(ar) = &req.aspect_ratio {
            body["aspectRatio"] = json!(ar);
        }

        // extra 透传（合并到顶层，覆盖同名字段）
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }

        (endpoint, body)
    }

    /// 解析 Runway 创建任务响应 → VideoTask
    ///
    /// Runway 创建响应：`{"id": "...", "status": "pending", "createdAt": "..."}`
    /// task_id 兼容 `id` / `assetId` / `taskId` 字段，缺失时生成 `vid` 前缀 ID（与 Python 一致）。
    fn parse_video_task(value: &Value, model: &str) -> Result<VideoTask> {
        let task_id = value
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| {
                value
                    .get("assetId")
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
        Ok(VideoTask {
            task_id,
            model: model.to_string(),
            status: map_runway_status(raw_status),
            created_at: util::current_timestamp(),
        })
    }

    /// 解析 Runway 任务查询响应 → VideoStatus
    ///
    /// 移植自 Python v1 `video_poll`：
    /// - 视频 URL 兼容 `url` / `videoUrl` / `output.url` / `output.videoUrl` / `assets[0].url`
    /// - 错误信息兼容 `error` / `errorMessage`
    /// - progress / createdAt / updatedAt 直接透传（数字形式）
    fn parse_video_status(value: &Value, task_id: &str) -> VideoStatus {
        let raw_status = value.get("status").and_then(|v| v.as_str()).unwrap_or("");
        let status = map_runway_status(raw_status);

        // 视频 URL：多路径兼容（与 Python v1 的提取顺序一致）
        let video_url = value
            .get("url")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| {
                value
                    .get("videoUrl")
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
                    .get("output")
                    .and_then(|o| o.get("videoUrl"))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            })
            .or_else(|| {
                value
                    .get("assets")
                    .and_then(|a| a.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|item| item.get("url"))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            });

        // 错误信息：error / errorMessage
        let error = value
            .get("error")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| {
                value
                    .get("errorMessage")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            });

        let progress = value
            .get("progress")
            .and_then(|v| v.as_u64())
            .map(|p| p as u32);

        // createdAt / updatedAt：Runway 通常返回 ISO 字符串，此处仅在为数字时解析
        // （与方舟 volcengine_cv 行为一致；ISO 字符串场景留待后续统一重构）
        let created_at = value.get("createdAt").and_then(|v| v.as_u64());
        let updated_at = value.get("updatedAt").and_then(|v| v.as_u64());

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

    /// 将 Runway API 错误响应映射为 AibridgeError
    ///
    /// 映射规则（与 [`AibridgeError::from_http_status`] 一致）：
    /// - 401/403 → Authentication（"Invalid Runway API key"）
    /// - 429 → RateLimit（"Runway rate limit exceeded"）
    /// - 404 → ModelNotFound（提取错误体消息）
    /// - 400 → Validation（提取错误体消息，details 携带原始 body）
    /// - 其余 ≥400（含 5xx）→ Api
    pub fn map_api_error(status: u16, body: &str) -> AibridgeError {
        match status {
            401 | 403 => AibridgeError::Authentication {
                message: "Invalid Runway API key".to_string(),
            },
            429 => AibridgeError::RateLimit {
                message: "Runway rate limit exceeded".to_string(),
                retry_after: None,
            },
            404 => AibridgeError::ModelNotFound {
                model: parse_error_message(body, status),
            },
            400 => AibridgeError::Validation {
                message: parse_error_message(body, status),
                details: serde_json::from_str::<Value>(body).unwrap_or(Value::Null),
            },
            _ => {
                let message = parse_error_message(body, status);
                AibridgeError::Api { status, message }
            }
        }
    }
}

#[async_trait]
impl Adapter for RunwayAdapter {
    fn provider_type(&self) -> &str {
        "runway"
    }

    fn provider_name(&self) -> &str {
        "Runway"
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

    /// 创建视频生成任务
    ///
    /// 根据模式与参考图选择端点：
    /// - `image2video` 且有参考图 → `POST /image_to_video`
    /// - 其余 → `POST /text_to_video`
    async fn video_create(&self, req: VideoRequest) -> Result<VideoTask> {
        self.ensure_capability(Capabilities::VideoGenerate)?;
        let model = if req.model.is_empty() {
            "gen-3".to_string()
        } else {
            req.model.clone()
        };
        let (endpoint, body) = self.build_create_body(&req);
        let value = self.post_authed_json(endpoint, &body).await?;
        Self::parse_video_task(&value, &model)
    }

    /// 查询视频任务状态：`GET /assets/{task_id}`
    async fn video_poll(&self, task_id: &str, _model: &str) -> Result<VideoStatus> {
        self.ensure_capability(Capabilities::VideoGenerate)?;
        let path = format!("assets/{task_id}");
        let value = self.get_authed_json(&path).await?;
        Ok(Self::parse_video_status(&value, task_id))
    }

    /// 模型列表（硬编码）
    ///
    /// Runway 无标准 `/models` 端点，暂保留硬编码列表（与 Python v1 一致）。
    /// 含 gen-3 / gen-3-turbo 两个视频模型。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = runway_hardcoded_models();
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }

    // chat / chat_stream / image_generate / embed / transcribe / speech 走 trait 默认实现
    // 返 UnsupportedCapability，与 Python v1 抛 UnsupportedCapabilityError 行为一致。
}

// ==================== 内部：硬编码模型列表 ====================

/// Runway 硬编码模型列表
///
/// 对应 Python v1 `RunwayAdapter.list_models`。
/// 注意：该 Provider 无标准 `/models` 端点，暂保留硬编码列表。
fn runway_hardcoded_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "gen-3".into(),
            name: "Gen-3 Alpha".into(),
            model_type: ModelType::Video,
            provider: "runway".into(),
            capabilities: vec!["text2video".into(), "image2video".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Runway Gen-3 Alpha 视频生成模型".into()),
            created: None,
        },
        ModelInfo {
            id: "gen-3-turbo".into(),
            name: "Gen-3 Turbo".into(),
            model_type: ModelType::Video,
            provider: "runway".into(),
            capabilities: vec!["text2video".into(), "image2video".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Runway Gen-3 Turbo 快速视频生成模型".into()),
            created: None,
        },
    ]
}

// ==================== 辅助函数 ====================

/// 映射 Runway 状态字符串到统一 TaskStatus
///
/// 移植自 Python v1 `_map_runway_status`：
/// - pending / queued → Pending
/// - processing / running / in_progress → Processing
/// - completed / succeeded / success → Success
/// - failed / error / cancelled → Failed
/// - 未知 → Pending（与 Python 默认值一致）
fn map_runway_status(raw: &str) -> TaskStatus {
    match raw.to_lowercase().as_str() {
        "pending" | "queued" => TaskStatus::Pending,
        "processing" | "running" | "in_progress" => TaskStatus::Processing,
        "completed" | "succeeded" | "success" => TaskStatus::Success,
        "failed" | "error" | "cancelled" => TaskStatus::Failed,
        _ => TaskStatus::Pending,
    }
}

/// 从 FileInput 提取 URL/base64 字符串
///
/// Runway `promptImage` 需要 URL 或 base64 字符串：
/// - `Url(s)` / `Base64(s)` → `s`
/// - `Path(_)` / `Bytes(_)` → 空字符串（需上层预转换为 URL/base64，此处不处理）
fn file_input_to_url(input: &FileInput) -> String {
    match input {
        FileInput::Url(s) | FileInput::Base64(s) => s.clone(),
        FileInput::Path(_) | FileInput::Bytes(_) => String::new(),
    }
}

/// 解析错误体中的 message 字段
///
/// 兼容多种错误体结构（Runway 无统一规范，多种均可能出现）：
/// - `{"error": "..."}`（顶层 error 字符串）
/// - `{"error": {"message": "..."}}`（OpenAI 风格）
/// - `{"message": "..."}`
/// - `{"detail": "..."}`
///
/// 解析失败时回退到 `HTTP {status}` 字符串。
fn parse_error_message(body: &str, status: u16) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        // 顶层 error 字符串（Runway 常见）
        if let Some(msg) = v.get("error").and_then(|m| m.as_str()) {
            return msg.to_string();
        }
        // OpenAI 风格 error.message
        if let Some(msg) = v
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::HttpClient;
    use crate::model::chat::{ChatMessage, ChatRequest};
    use crate::model::image::ImageRequest;
    use mockito::Server;
    use serde_json::json;

    // ==================== 通用测试辅助 ====================

    /// 构造测试用 RunwayAdapter（指向 mockito server）
    fn make_runway(server: &Server) -> RunwayAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("runway", opts);
        // http 客户端的 base_url 指向 mockito（与 emerging_models 测试结构一致）
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        RunwayAdapter::with_http(http, config)
    }

    /// 构造不指向任何 server 的 RunwayAdapter（用于不发请求的元信息/能力测试）
    fn make_runway_no_server() -> RunwayAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(DEFAULT_RUNWAY_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("runway", opts);
        RunwayAdapter::new(config).expect("RunwayAdapter 构造应成功")
    }

    // ============ 元信息 ============

    #[test]
    fn runway_provider_type_and_name_match_python() {
        let adapter = make_runway_no_server();
        assert_eq!(adapter.provider_type(), "runway");
        assert_eq!(adapter.provider_name(), "Runway");
    }

    #[test]
    fn runway_requires_api_key_is_true() {
        let adapter = make_runway_no_server();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn runway_capabilities_contains_only_video() {
        let adapter = make_runway_no_server();
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::VideoGenerate));
        assert!(caps.contains(&Capabilities::VideoText2Video));
        assert!(caps.contains(&Capabilities::VideoImage2Video));
        // chat / image 不声明
        assert!(!caps.contains(&Capabilities::Chat));
        assert!(!caps.contains(&Capabilities::ImageGenerate));
    }

    #[test]
    fn runway_base_url_defaults_when_missing() {
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("runway", opts);
        let adapter = RunwayAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), DEFAULT_RUNWAY_BASE_URL);
    }

    #[test]
    fn runway_base_url_uses_config_when_provided() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://custom.runway-proxy.com/v1")
            .build();
        let config = ProviderConfig::from_options("runway", opts);
        let adapter = RunwayAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), "https://custom.runway-proxy.com/v1");
    }

    // ============ video_create 正常路径 ============

    #[tokio::test]
    async fn runway_video_create_text2video_success_returns_task() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "asset-abc",
            "status": "pending",
            "createdAt": "2024-01-01T00:00:00Z"
        });
        let mock = server
            .mock("POST", "/text_to_video")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "a cat walking")
            .aspect_ratio("16:9")
            .seed(42)
            .build();
        let task = adapter
            .video_create(req)
            .await
            .expect("video_create 应成功");

        assert_eq!(task.task_id, "asset-abc");
        assert_eq!(task.model, "gen-3");
        assert_eq!(task.status, TaskStatus::Pending);
        assert!(task.created_at > 0);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn runway_video_create_text2video_sends_model_and_prompt() {
        // 验证 text_to_video 端点的请求体结构
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/text_to_video")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "gen-3",
                "prompt": "a cat",
                "width": 1280,
                "height": 720,
                "seed": 42,
                "aspectRatio": "16:9"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "a cat")
            .aspect_ratio("16:9")
            .seed(42)
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn runway_video_create_uses_default_model_when_empty() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/text_to_video")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "gen-3"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("", "a cat").build();
        let task = adapter.video_create(req).await.unwrap();
        assert_eq!(task.model, "gen-3");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn runway_video_create_image2video_with_reference_image() {
        // image2video 模式：走 /image_to_video，body 含 promptImage + promptText
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/image_to_video")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "gen-3",
                "promptImage": "https://example.com/start.png",
                "promptText": "animate this"
            })))
            .with_status(200)
            .with_body(json!({"id": "img-task", "status": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "animate this")
            .mode(VideoMode::Image2Video)
            .reference_images(vec![FileInput::url("https://example.com/start.png")])
            .build();
        let task = adapter.video_create(req).await.unwrap();
        assert_eq!(task.task_id, "img-task");
        assert_eq!(task.status, TaskStatus::Pending);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn runway_video_create_image2video_falls_back_to_first_frame() {
        // image2video 但无 reference_images，用 first_frame 作为 promptImage
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/image_to_video")
            .match_body(mockito::Matcher::PartialJson(json!({
                "promptImage": "https://example.com/first.png"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "animate")
            .mode(VideoMode::Image2Video)
            .first_frame(FileInput::url("https://example.com/first.png"))
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn runway_video_create_image2video_without_image_falls_back_to_text() {
        // image2video 但既无 reference_images 也无 first_frame → 回退 text_to_video
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/text_to_video")
            .match_body(mockito::Matcher::PartialJson(json!({
                "prompt": "a cat"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "a cat")
            .mode(VideoMode::Image2Video)
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn runway_video_create_passes_motion_and_camera_motion() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/text_to_video")
            .match_body(mockito::Matcher::PartialJson(json!({
                "motion": 7.5,
                "cameraMotion": "zoom_in"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "a cat")
            .motion_strength(7.5)
            .camera_motion("zoom_in")
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn runway_video_create_passes_extra_params() {
        // extra 透传到顶层
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/text_to_video")
            .match_body(mockito::Matcher::PartialJson(json!({
                "custom_param": "custom_value",
                "watermark": false
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "a cat")
            .extra("custom_param", "custom_value")
            .extra("watermark", false)
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn runway_video_create_uses_assetid_field_when_id_missing() {
        // 响应缺 id 但有 assetId
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/text_to_video")
            .with_status(200)
            .with_body(json!({"assetId": "asset-xyz", "status": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "a cat").build();
        let task = adapter.video_create(req).await.unwrap();
        assert_eq!(task.task_id, "asset-xyz");
    }

    #[tokio::test]
    async fn runway_video_create_generates_id_when_all_id_fields_missing() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/text_to_video")
            .with_status(200)
            .with_body(json!({"status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "a cat").build();
        let task = adapter.video_create(req).await.unwrap();
        assert!(task.task_id.starts_with("vid_"));
    }

    // ============ video_create 错误路径 ============

    #[tokio::test]
    async fn runway_video_create_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/text_to_video")
            .with_status(401)
            .with_body(json!({"error": "Invalid API key"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn runway_video_create_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/text_to_video")
            .with_status(429)
            .with_body(json!({"error": "slow down"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn runway_video_create_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/text_to_video")
            .with_status(404)
            .with_body(json!({"error": "model not found"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-unknown", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn runway_video_create_error_400_returns_validation() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/text_to_video")
            .with_status(400)
            .with_body(json!({"error": "invalid prompt"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Validation { message, .. } => {
                assert!(message.contains("invalid prompt"));
            }
            _ => panic!("应为 Validation"),
        }
    }

    #[tokio::test]
    async fn runway_video_create_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/text_to_video")
            .with_status(500)
            .with_body(json!({"error": "internal"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "a cat").build();
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
    async fn runway_video_create_error_no_json_body_falls_back() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/text_to_video")
            .with_status(502)
            .with_body("Bad Gateway")
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let req = VideoRequest::builder("gen-3", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("502")),
            _ => panic!("应为 Api"),
        }
    }

    // ============ video_poll 正常路径 ============

    #[tokio::test]
    async fn runway_video_poll_success_returns_video_url() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "asset-abc",
            "status": "completed",
            "url": "https://example.com/video.mp4",
            "progress": 100,
            "createdAt": 1700000000,
            "updatedAt": 1700000100
        });
        let mock = server
            .mock("GET", "/assets/asset-abc")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let status = adapter
            .video_poll("asset-abc", "gen-3")
            .await
            .expect("video_poll 应成功");

        assert_eq!(status.task_id, "asset-abc");
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
    async fn runway_video_poll_processing_returns_progress() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/assets/asset-1")
            .with_status(200)
            .with_body(json!({"id": "asset-1", "status": "processing", "progress": 45}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let status = adapter.video_poll("asset-1", "gen-3").await.unwrap();
        assert_eq!(status.status, TaskStatus::Processing);
        assert_eq!(status.progress, Some(45));
        assert!(status.video_url.is_none());
    }

    #[tokio::test]
    async fn runway_video_poll_failed_returns_error() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/assets/asset-2")
            .with_status(200)
            .with_body(
                json!({"id": "asset-2", "status": "failed", "error": "content policy violation"})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let status = adapter.video_poll("asset-2", "gen-3").await.unwrap();
        assert_eq!(status.status, TaskStatus::Failed);
        assert_eq!(status.error.as_deref(), Some("content policy violation"));
    }

    #[tokio::test]
    async fn runway_video_poll_failed_returns_error_message_field() {
        // errorMessage 字段兜底
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/assets/asset-3")
            .with_status(200)
            .with_body(
                json!({"id": "asset-3", "status": "error", "errorMessage": "internal failure"})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let status = adapter.video_poll("asset-3", "gen-3").await.unwrap();
        assert_eq!(status.status, TaskStatus::Failed);
        assert_eq!(status.error.as_deref(), Some("internal failure"));
    }

    #[tokio::test]
    async fn runway_video_poll_queued_maps_to_pending() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/assets/asset-4")
            .with_status(200)
            .with_body(json!({"id": "asset-4", "status": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let status = adapter.video_poll("asset-4", "gen-3").await.unwrap();
        assert_eq!(status.status, TaskStatus::Pending);
    }

    #[tokio::test]
    async fn runway_video_poll_video_url_output_path() {
        // output.url 兜底路径
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/assets/asset-5")
            .with_status(200)
            .with_body(
                json!({
                    "id": "asset-5",
                    "status": "succeeded",
                    "output": {"url": "https://example.com/out.mp4"}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let status = adapter.video_poll("asset-5", "gen-3").await.unwrap();
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://example.com/out.mp4")
        );
    }

    #[tokio::test]
    async fn runway_video_poll_video_url_assets_array_path() {
        // assets[0].url 兜底路径
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/assets/asset-6")
            .with_status(200)
            .with_body(
                json!({
                    "id": "asset-6",
                    "status": "completed",
                    "assets": [{"url": "https://example.com/arr.mp4"}]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let status = adapter.video_poll("asset-6", "gen-3").await.unwrap();
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://example.com/arr.mp4")
        );
    }

    #[tokio::test]
    async fn runway_video_poll_video_url_videourl_field() {
        // videoUrl 字段兜底
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/assets/asset-7")
            .with_status(200)
            .with_body(
                json!({
                    "id": "asset-7",
                    "status": "completed",
                    "videoUrl": "https://example.com/vu.mp4"
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let status = adapter.video_poll("asset-7", "gen-3").await.unwrap();
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://example.com/vu.mp4")
        );
    }

    // ============ video_poll 错误路径 ============

    #[tokio::test]
    async fn runway_video_poll_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/assets/asset-x")
            .with_status(401)
            .with_body(json!({"error": "bad key"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let err = adapter.video_poll("asset-x", "gen-3").await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn runway_video_poll_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/assets/nonexistent")
            .with_status(404)
            .with_body(json!({"error": "asset not found"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let err = adapter
            .video_poll("nonexistent", "gen-3")
            .await
            .unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn runway_video_poll_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/assets/asset-y")
            .with_status(429)
            .with_body(json!({"error": "slow down"}).to_string())
            .create_async()
            .await;

        let adapter = make_runway(&server);
        let err = adapter.video_poll("asset-y", "gen-3").await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    // ============ 不支持的能力 ============

    #[tokio::test]
    async fn runway_chat_returns_unsupported() {
        let adapter = make_runway_no_server();
        let req = ChatRequest::builder("gen-3", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn runway_image_generate_returns_unsupported() {
        let adapter = make_runway_no_server();
        let req = ImageRequest::builder("gen-3", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn runway_chat_stream_returns_unsupported() {
        let adapter = make_runway_no_server();
        let req = ChatRequest::builder("gen-3", vec![ChatMessage::user("hi")]).build();
        let result = adapter.chat_stream(req).await;
        assert!(matches!(
            result,
            Err(AibridgeError::UnsupportedCapability { .. })
        ));
    }

    // ============ list_models ============

    #[tokio::test]
    async fn runway_list_models_returns_hardcoded() {
        let adapter = make_runway_no_server();
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "gen-3");
        assert_eq!(models[0].name, "Gen-3 Alpha");
        assert_eq!(models[0].provider, "runway");
        assert_eq!(models[0].model_type, ModelType::Video);
        assert_eq!(models[1].id, "gen-3-turbo");
    }

    #[tokio::test]
    async fn runway_list_models_filter_by_video_type() {
        let adapter = make_runway_no_server();
        let videos = adapter.list_models(Some(ModelType::Video)).await.unwrap();
        assert_eq!(videos.len(), 2);
        assert!(videos.iter().all(|m| m.model_type == ModelType::Video));
    }

    #[tokio::test]
    async fn runway_list_models_filter_by_image_returns_empty() {
        let adapter = make_runway_no_server();
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert!(images.is_empty());
    }

    // ============ start / close ============

    #[tokio::test]
    async fn runway_start_and_close_are_noops() {
        let mut adapter = make_runway_no_server();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    // ============ 错误映射单元测试 ============

    #[test]
    fn runway_map_api_error_401_is_authentication() {
        let err = RunwayAdapter::map_api_error(401, "");
        match err {
            AibridgeError::Authentication { message } => assert!(message.contains("Runway")),
            _ => panic!("应为 Authentication"),
        }
    }

    #[test]
    fn runway_map_api_error_403_is_authentication() {
        let err = RunwayAdapter::map_api_error(403, "");
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[test]
    fn runway_map_api_error_429_is_rate_limit() {
        let err = RunwayAdapter::map_api_error(429, "");
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[test]
    fn runway_map_api_error_404_is_model_not_found() {
        let err = RunwayAdapter::map_api_error(404, "");
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[test]
    fn runway_map_api_error_400_is_validation() {
        let err = RunwayAdapter::map_api_error(400, "");
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[test]
    fn runway_map_api_error_400_carries_details() {
        let body = json!({"error": "bad param"}).to_string();
        let err = RunwayAdapter::map_api_error(400, &body);
        match err {
            AibridgeError::Validation { message, details } => {
                assert_eq!(message, "bad param");
                assert_eq!(details["error"], "bad param");
            }
            _ => panic!("应为 Validation"),
        }
    }

    #[test]
    fn runway_map_api_error_500_is_api() {
        let err = RunwayAdapter::map_api_error(500, "");
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 500),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn runway_map_api_error_extracts_error_message() {
        let body = json!({"error": "something went wrong"}).to_string();
        let err = RunwayAdapter::map_api_error(500, &body);
        match err {
            AibridgeError::Api { message, .. } => assert_eq!(message, "something went wrong"),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn runway_map_api_error_extracts_openai_style_message() {
        let body = json!({"error": {"message": "rate limited"}}).to_string();
        let err = RunwayAdapter::map_api_error(429, &body);
        match err {
            AibridgeError::RateLimit { message, .. } => {
                // 429 固定消息，不读 body
                assert!(message.contains("rate limit"));
            }
            _ => panic!("应为 RateLimit"),
        }
    }

    #[test]
    fn runway_map_api_error_no_json_falls_back_to_http_status() {
        let err = RunwayAdapter::map_api_error(502, "Bad Gateway");
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("502")),
            _ => panic!("应为 Api"),
        }
    }

    // ============ map_runway_status 单元测试 ============

    #[test]
    fn map_runway_status_pending_is_pending() {
        assert_eq!(map_runway_status("pending"), TaskStatus::Pending);
    }

    #[test]
    fn map_runway_status_queued_is_pending() {
        assert_eq!(map_runway_status("queued"), TaskStatus::Pending);
    }

    #[test]
    fn map_runway_status_processing_is_processing() {
        assert_eq!(map_runway_status("processing"), TaskStatus::Processing);
        assert_eq!(map_runway_status("running"), TaskStatus::Processing);
        assert_eq!(map_runway_status("in_progress"), TaskStatus::Processing);
    }

    #[test]
    fn map_runway_status_success_variants() {
        assert_eq!(map_runway_status("completed"), TaskStatus::Success);
        assert_eq!(map_runway_status("succeeded"), TaskStatus::Success);
        assert_eq!(map_runway_status("success"), TaskStatus::Success);
    }

    #[test]
    fn map_runway_status_failed_variants() {
        assert_eq!(map_runway_status("failed"), TaskStatus::Failed);
        assert_eq!(map_runway_status("error"), TaskStatus::Failed);
        assert_eq!(map_runway_status("cancelled"), TaskStatus::Failed);
    }

    #[test]
    fn map_runway_status_case_insensitive() {
        assert_eq!(map_runway_status("PENDING"), TaskStatus::Pending);
        assert_eq!(map_runway_status("Completed"), TaskStatus::Success);
        assert_eq!(map_runway_status("FAILED"), TaskStatus::Failed);
    }

    #[test]
    fn map_runway_status_unknown_defaults_to_pending() {
        assert_eq!(map_runway_status("unknown_state"), TaskStatus::Pending);
        assert_eq!(map_runway_status(""), TaskStatus::Pending);
    }

    // ============ file_input_to_url 单元测试 ============

    #[test]
    fn file_input_to_url_returns_url_for_url_variant() {
        let f = FileInput::url("https://example.com/x.png");
        assert_eq!(file_input_to_url(&f), "https://example.com/x.png");
    }

    #[test]
    fn file_input_to_url_returns_base64_for_base64_variant() {
        let f = FileInput::base64("aGVsbG8=");
        assert_eq!(file_input_to_url(&f), "aGVsbG8=");
    }

    #[test]
    fn file_input_to_url_returns_empty_for_path_variant() {
        let f = FileInput::path("/tmp/x.png");
        assert_eq!(file_input_to_url(&f), "");
    }

    #[test]
    fn file_input_to_url_returns_empty_for_bytes_variant() {
        let f = FileInput::bytes(vec![1, 2, 3]);
        assert_eq!(file_input_to_url(&f), "");
    }

    // ============ build_create_body 单元测试 ============

    #[test]
    fn build_create_body_text2video_includes_fields() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("runway", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = RunwayAdapter::with_http(http, config);
        let req = VideoRequest::builder("gen-3", "a cat")
            .aspect_ratio("16:9")
            .seed(42)
            .build();
        let (endpoint, body) = adapter.build_create_body(&req);
        assert_eq!(endpoint, "text_to_video");
        assert_eq!(body["model"], "gen-3");
        assert_eq!(body["prompt"], "a cat");
        assert_eq!(body["aspectRatio"], "16:9");
        assert_eq!(body["seed"], 42);
        assert_eq!(body["width"], 1280);
        assert_eq!(body["height"], 720);
    }

    #[test]
    fn build_create_body_image2video_uses_prompt_image() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("runway", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = RunwayAdapter::with_http(http, config);
        let req = VideoRequest::builder("gen-3", "animate")
            .mode(VideoMode::Image2Video)
            .reference_images(vec![FileInput::url("https://example.com/a.png")])
            .build();
        let (endpoint, body) = adapter.build_create_body(&req);
        assert_eq!(endpoint, "image_to_video");
        assert_eq!(body["promptImage"], "https://example.com/a.png");
        assert_eq!(body["promptText"], "animate");
        // text_to_video 的 prompt 字段不应出现
        assert!(body.get("prompt").is_none());
    }

    #[test]
    fn build_create_body_merges_extra() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("runway", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = RunwayAdapter::with_http(http, config);
        let req = VideoRequest::builder("gen-3", "a cat")
            .extra("custom", "value")
            .build();
        let (_, body) = adapter.build_create_body(&req);
        assert_eq!(body["custom"], "value");
    }

    #[test]
    fn build_create_body_uses_default_model_when_empty() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("runway", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = RunwayAdapter::with_http(http, config);
        let req = VideoRequest::builder("", "a cat").build();
        let (_, body) = adapter.build_create_body(&req);
        assert_eq!(body["model"], "gen-3");
    }
}
