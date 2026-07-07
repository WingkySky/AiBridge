//! Kling 适配器（可灵，视频生成，独立协议）
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/kling.py`。
//!
//! Kling（快手可灵）API 为**独立协议**（非 OpenAI 兼容），不复用 `OpenAiCompatAdapter`：
//! - 文生视频：`POST /videos/generations`
//! - 图生视频：`POST /videos/image2video`
//! - 查询任务状态：`GET /videos/generations/{task_id}`
//! - 认证：`Authorization: Bearer {api_key}`
//! - 默认 Base URL：`https://api.klingai.com/v1`
//!
//! ## 请求体规范
//! - `model_name` / `prompt` 必选
//! - 图生视频：`reference_images[0]` → `image` 字段
//! - `negative_prompt` / `cfg_scale` / `duration` / `aspect_ratio` 可选透传
//! - `camera_control`（镜头控制对象）/ `mode`（std/pro 档位）等厂商特有参数走 `extra` 透传
//!
//! ## 响应解析
//! - 创建/查询响应的任务信息包裹在 `data` 字段内（缺失时回退到顶层，更健壮）
//! - 任务 ID：`data.task_id`（缺失回退顶层，再缺失生成 `vid_` 前缀 ID）
//! - 状态：`data.task_status`，映射 submitted/queued → pending、processing、succeed/success → success、failed/error → failed
//! - 视频 URL：`data.task_result.videos[0].url`
//! - 错误信息：`data.task_status_msg` / `data.error`
//! - `progress`：success=100、pending=0、其余=50（与 Python v1 一致，Kling 响应无 progress 字段）
//!
//! ## 阶段范围
//! 阶段 2b 实现 video_create + video_poll + list_models（硬编码）。
//! 不支持的能力（chat / image / embed / audio）走 `Adapter` trait 默认实现返
//! `UnsupportedCapability`，与 Python v1 抛 `UnsupportedCapabilityError` 行为一致。
//! Python v1 明确不支持 image_generate（Kolors 是单独模型）。

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

/// Kling 默认 Base URL
///
/// 对应 Python v1 `DEFAULT_BASE_URL`（已含 `/v1` 前缀）。
pub const DEFAULT_KLING_BASE_URL: &str = "https://api.klingai.com/v1";

// ==================== 能力集合构造 ====================

/// Kling 支持的能力集合
///
/// 对齐 Python v1 `KlingAdapter.supported_capabilities = ["video"]`。
/// Rust 用 `VideoGenerate` 表达视频生成能力（含 video_create + video_poll），
/// 并声明 text2video / image2video 两个子能力。
fn kling_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::VideoGenerate);
    caps.insert(Capabilities::VideoText2Video);
    caps.insert(Capabilities::VideoImage2Video);
    caps
}

// ==================== Kling 视频生成适配器 ====================

/// Kling 适配器
///
/// 快手可灵视频生成平台，支持文生视频与图生视频（kling-v1 / v1-5 / v2）。
/// 官方 API 文档：https://app.klingai.com/docs/api
///
/// ## API 规范
/// - Base URL: `https://api.klingai.com/v1`
/// - 文生视频: `POST /videos/generations`
/// - 图生视频: `POST /videos/image2video`
/// - 查询状态: `GET /videos/generations/{task_id}`
/// - 认证: `Authorization: Bearer {api_key}`
///
/// ## 阶段范围
/// 阶段 2b 实现 video_create + video_poll + list_models（硬编码）。
pub struct KlingAdapter {
    /// HTTP 客户端（封装 reqwest，含连接池与超时）
    http: HttpClient,
    /// Provider 配置（api_key / base_url / timeout 等）
    config: ProviderConfig,
    /// 实际 base_url（已合并 config.base_url 与默认值）
    base_url: String,
    /// 支持的能力集合
    capabilities: CapabilitySet,
}

impl KlingAdapter {
    /// 创建 Kling 适配器
    ///
    /// `config.base_url` 为空时用 [`DEFAULT_KLING_BASE_URL`] 兜底。
    /// `config.api_key` 为空时不在此处报错（由上层按 `requires_api_key` 校验）。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_KLING_BASE_URL.to_string());

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
            capabilities: kling_capabilities(),
        })
    }

    /// 用显式 HttpClient 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_http(http: HttpClient, config: ProviderConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_KLING_BASE_URL.to_string());
        Self {
            http,
            config,
            base_url,
            capabilities: kling_capabilities(),
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
                capability: format!("{} (provider: kling)", cap.as_str()),
            })
        }
    }

    /// 发送带 Bearer 认证的 POST JSON 请求，并用 Kling 错误映射处理响应
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

    /// 发送带 Bearer 认证的 GET 请求，并用 Kling 错误映射处理响应
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

    /// 构造 Kling 视频生成请求体
    ///
    /// 移植自 Python v1 `video_create`：
    /// - `model_name` / `prompt` 必选（Kling 用 `model_name` 而非 `model`）
    /// - image2video 模式且 `reference_images[0]` 可转 URL 时，写入 `image` 字段
    /// - `negative_prompt` / `cfg_scale` / `duration` / `aspect_ratio` 可选透传
    /// - `camera_control`（镜头控制）/ `mode`（std/pro）等厂商特有参数走 `extra` 透传到顶层
    fn build_video_body(&self, req: &VideoRequest) -> Value {
        let mut body = json!({
            "model_name": req.model,
            "prompt": req.prompt,
        });

        // image2video 模式：reference_images[0] → image 字段
        if is_image2video_request(req) {
            if let Some(url) = req.reference_images.first().and_then(file_input_to_url) {
                body["image"] = json!(url);
            }
        }

        if let Some(np) = &req.negative_prompt {
            body["negative_prompt"] = json!(np);
        }
        if let Some(cfg) = req.cfg_scale {
            body["cfg_scale"] = json!(cfg);
        }
        if let Some(d) = req.duration {
            body["duration"] = json!(d);
        }
        if let Some(ar) = &req.aspect_ratio {
            body["aspect_ratio"] = json!(ar);
        }

        // extra 透传（合并到顶层：camera_control / mode 等厂商特有参数）
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        body
    }

    /// 解析 Kling 创建响应 → VideoTask
    ///
    /// 移植自 Python v1 `video_create` 响应解析：
    /// - 任务信息包裹在 `data` 字段内（缺失时回退顶层，更健壮）
    /// - 任务 ID：`data.task_id`（回退顶层 task_id，再缺失生成 `vid_` 前缀 ID）
    /// - 状态：`data.task_status`（缺失默认 pending）
    /// - `created_at`：优先取响应值，缺失用当前时间戳（对齐 Python v1）
    fn parse_video_task(value: &Value, model: &str) -> Result<VideoTask> {
        let data = value.get("data").unwrap_or(value);
        let task_id = data
            .get("task_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| {
                value
                    .get("task_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| util::generate_id("vid"));
        let raw_status = data
            .get("task_status")
            .and_then(|v| v.as_str())
            .or_else(|| value.get("task_status").and_then(|v| v.as_str()))
            .unwrap_or("pending");
        let created_at = data
            .get("created_at")
            .and_then(|v| v.as_u64())
            .or_else(|| value.get("created_at").and_then(|v| v.as_u64()))
            .unwrap_or_else(util::current_timestamp);
        Ok(VideoTask {
            task_id,
            model: model.to_string(),
            status: map_kling_status(raw_status),
            created_at,
        })
    }

    /// 解析 Kling 查询响应 → VideoStatus
    ///
    /// 移植自 Python v1 `video_poll`：
    /// - 任务信息包裹在 `data` 字段内（缺失时回退顶层）
    /// - 视频 URL：`data.task_result.videos[0].url`
    /// - 错误信息：`data.task_status_msg` / `data.error`
    /// - `progress`：success=100、pending=0、其余=50（Kling 响应无 progress 字段，按状态推算）
    /// - `updated_at` 缺失时回退当前时间戳（对齐 Python v1）
    fn parse_video_status(value: &Value, task_id: &str) -> VideoStatus {
        let data = value.get("data").unwrap_or(value);
        let raw_status = data
            .get("task_status")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let status = map_kling_status(raw_status);

        // 视频 URL：data.task_result.videos[0].url
        let video_url = data
            .get("task_result")
            .and_then(|tr| tr.get("videos"))
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|f| f.get("url"))
            .and_then(|v| v.as_str())
            .map(str::to_owned);

        // 错误信息：task_status_msg 优先，回退 error
        let error = data
            .get("task_status_msg")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| {
                data.get("error")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            });

        // progress：按状态推算（Kling 响应无 progress 字段）
        let progress = match status {
            TaskStatus::Success => Some(100),
            TaskStatus::Pending => Some(0),
            _ => Some(50),
        };

        let created_at = data.get("created_at").and_then(|v| v.as_u64());
        let updated_at = data
            .get("updated_at")
            .and_then(|v| v.as_u64())
            .or_else(|| Some(util::current_timestamp()));

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

    /// 将 Kling API 错误响应映射为 AibridgeError
    ///
    /// 移植自 Python v1 `_handle_kling_error` 并按阶段 2b 统一错误映射要求调整：
    /// - 401 → Authentication（"Invalid Kling API key"）
    /// - 429 → RateLimit（"Kling rate limit exceeded or quota exhausted"）
    /// - 404 → ModelNotFound（Kling 任务/资源不存在）
    /// - 400 → Validation（请求参数校验错误）
    /// - 其余 ≥400 → Api（提取 error.message / message / error / detail，回退 `HTTP {status}`）
    pub fn map_api_error(status: u16, body: &str) -> AibridgeError {
        match status {
            401 => AibridgeError::Authentication {
                message: "Invalid Kling API key".to_string(),
            },
            429 => AibridgeError::RateLimit {
                message: "Kling rate limit exceeded or quota exhausted".to_string(),
                retry_after: None,
            },
            404 => AibridgeError::ModelNotFound {
                model: "Kling task or resource not found".to_string(),
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
impl Adapter for KlingAdapter {
    fn provider_type(&self) -> &str {
        "kling"
    }

    fn provider_name(&self) -> &str {
        "可灵 Kling"
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
    /// - image2video 模式且有可转 URL 的 reference_images[0] → `POST /videos/image2video`
    /// - 其余 → `POST /videos/generations`
    async fn video_create(&self, req: VideoRequest) -> Result<VideoTask> {
        self.ensure_capability(Capabilities::VideoGenerate)?;
        let body = self.build_video_body(&req);
        let endpoint = if is_image2video_request(&req) {
            "videos/image2video"
        } else {
            "videos/generations"
        };
        let value = self.post_authed_json(endpoint, &body).await?;
        Self::parse_video_task(&value, &req.model)
    }

    /// 查询视频任务状态：`GET /videos/generations/{task_id}`
    async fn video_poll(&self, task_id: &str, _model: &str) -> Result<VideoStatus> {
        self.ensure_capability(Capabilities::VideoGenerate)?;
        let path = format!("videos/generations/{task_id}");
        let value = self.get_authed_json(&path).await?;
        Ok(Self::parse_video_status(&value, task_id))
    }

    /// 模型列表（硬编码）
    ///
    /// Kling 无标准 `/models` 端点，暂保留硬编码列表（与 Python v1 一致）。
    /// 含 kling-v1 / kling-v1-5 / kling-v2 三个模型。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = kling_hardcoded_models();
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }

    // chat / chat_stream / image / embed / audio 走 trait 默认实现返 UnsupportedCapability，
    // 与 Python v1 抛 UnsupportedCapabilityError 行为一致。
}

// ==================== 内部：辅助函数 ====================

/// 判断是否为图生视频请求
///
/// 同时满足：mode 为 Image2Video，且 reference_images[0] 可转 URL（Url/Base64）。
/// 对齐 Python v1 `mode == "image2video" and reference_images` 的端点选择逻辑。
fn is_image2video_request(req: &VideoRequest) -> bool {
    matches!(req.mode, VideoMode::Image2Video)
        && req
            .reference_images
            .first()
            .and_then(file_input_to_url)
            .is_some()
}

/// 映射 Kling 状态字符串到统一 TaskStatus
///
/// 移植自 Python v1 `_map_kling_status`（大小写不敏感，比 Python 更健壮）：
/// - submitted / queued → Pending
/// - processing → Processing
/// - succeed / success → Success
/// - failed / error → Failed
/// - 未知 → Pending（与 Python 默认值一致）
fn map_kling_status(raw: &str) -> TaskStatus {
    match raw.to_lowercase().as_str() {
        "submitted" | "queued" => TaskStatus::Pending,
        "processing" => TaskStatus::Processing,
        "succeed" | "success" => TaskStatus::Success,
        "failed" | "error" => TaskStatus::Failed,
        _ => TaskStatus::Pending,
    }
}

/// 从 FileInput 提取 URL 字符串
///
/// Kling 的 `image` 字段接受 URL 或 base64 字符串：
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
/// - `{"message": "..."}`（顶层 message，Kling 常用）
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
        // 顶层 message（Kling 常用）
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

/// Kling 硬编码模型列表
///
/// 对应 Python v1 `KlingAdapter.list_models`。
/// 注意：该 Provider 无标准 `/models` 端点，暂保留硬编码列表。
fn kling_hardcoded_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "kling-v1".into(),
            name: "Kling 1.0".into(),
            model_type: ModelType::Video,
            provider: "kling".into(),
            capabilities: vec!["text2video".into(), "image2video".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Kling 1.0 标准版本".into()),
            created: None,
        },
        ModelInfo {
            id: "kling-v1-5".into(),
            name: "Kling 1.5".into(),
            model_type: ModelType::Video,
            provider: "kling".into(),
            capabilities: vec!["text2video".into(), "image2video".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Kling 1.5 改进版本".into()),
            created: None,
        },
        ModelInfo {
            id: "kling-v2".into(),
            name: "Kling 2.0".into(),
            model_type: ModelType::Video,
            provider: "kling".into(),
            capabilities: vec!["text2video".into(), "image2video".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Kling 2.0 最新版本，质量更好".into()),
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

    /// 构造测试用 KlingAdapter（指向 mockito server）
    fn make_kling(server: &Server) -> KlingAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("kling", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        KlingAdapter::with_http(http, config)
    }

    /// 构造不指向任何 server 的 KlingAdapter（用于不发请求的元信息/能力测试）
    fn make_kling_no_server() -> KlingAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(DEFAULT_KLING_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("kling", opts);
        KlingAdapter::new(config).expect("KlingAdapter 构造应成功")
    }

    // ============ 元信息 ============

    #[test]
    fn kling_provider_type_and_name_match_python() {
        let adapter = make_kling_no_server();
        assert_eq!(adapter.provider_type(), "kling");
        assert_eq!(adapter.provider_name(), "可灵 Kling");
    }

    #[test]
    fn kling_requires_api_key_is_true() {
        let adapter = make_kling_no_server();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn kling_capabilities_contains_only_video() {
        let adapter = make_kling_no_server();
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::VideoGenerate));
        assert!(caps.contains(&Capabilities::VideoText2Video));
        assert!(caps.contains(&Capabilities::VideoImage2Video));
        // chat / image 不声明
        assert!(!caps.contains(&Capabilities::Chat));
        assert!(!caps.contains(&Capabilities::ImageGenerate));
    }

    #[test]
    fn kling_base_url_defaults_when_missing() {
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("kling", opts);
        let adapter = KlingAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), DEFAULT_KLING_BASE_URL);
    }

    #[test]
    fn kling_base_url_uses_config_when_provided() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://custom.kling-proxy.com/v1")
            .build();
        let config = ProviderConfig::from_options("kling", opts);
        let adapter = KlingAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), "https://custom.kling-proxy.com/v1");
    }

    #[test]
    fn kling_base_url_ignores_empty_string() {
        // 空白 base_url 应回退到默认值
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("   ")
            .build();
        let config = ProviderConfig::from_options("kling", opts);
        let adapter = KlingAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), DEFAULT_KLING_BASE_URL);
    }

    // ============ video_create 正常路径 ============

    #[tokio::test]
    async fn kling_video_create_success_returns_task() {
        let mut server = Server::new_async().await;
        let body = json!({
            "code": 0,
            "message": "success",
            "data": {
                "task_id": "vid-abc123",
                "task_status": "submitted"
            }
        });
        let mock = server
            .mock("POST", "/videos/generations")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "A cat running in the park").build();
        let task = adapter
            .video_create(req)
            .await
            .expect("video_create 应成功");

        assert_eq!(task.task_id, "vid-abc123");
        assert_eq!(task.model, "kling-v1");
        // submitted → pending
        assert_eq!(task.status, TaskStatus::Pending);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn kling_video_create_sends_model_name_and_prompt() {
        // 验证请求体用 model_name 字段（非 model），且包含 prompt
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/videos/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model_name": "kling-v1",
                "prompt": "A cat running"
            })))
            .with_status(200)
            .with_body(
                json!({"code": 0, "data": {"task_id": "x", "task_status": "submitted"}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "A cat running").build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn kling_video_create_passes_optional_params() {
        // negative_prompt / cfg_scale / duration / aspect_ratio 透传
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/videos/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "negative_prompt": "blurry, low quality",
                "cfg_scale": 0.5,
                "duration": 10,
                "aspect_ratio": "16:9"
            })))
            .with_status(200)
            .with_body(
                json!({"code": 0, "data": {"task_id": "x", "task_status": "submitted"}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v2", "a beautiful sunset")
            .negative_prompt("blurry, low quality")
            .cfg_scale(0.5)
            .duration(10)
            .aspect_ratio("16:9")
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn kling_video_create_image2video_uses_image2video_endpoint() {
        // image2video 模式：走 /videos/image2video 端点，body 含 image 字段
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/videos/image2video")
            .match_body(mockito::Matcher::PartialJson(json!({
                "image": "https://example.com/input.jpg"
            })))
            .with_status(200)
            .with_body(
                json!({"code": 0, "data": {"task_id": "vid-img2vid-001", "task_status": "submitted"}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1-5", "Make this image move")
            .mode(VideoMode::Image2Video)
            .reference_images(vec![FileInput::url("https://example.com/input.jpg")])
            .build();
        let task = adapter.video_create(req).await.unwrap();
        assert_eq!(task.task_id, "vid-img2vid-001");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn kling_video_create_text2video_does_not_send_image_field() {
        // text2video 模式即使有 reference_images 也不发送 image 字段，且走 /videos/generations
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/videos/generations")
            .match_body(mockito::Matcher::Json(json!({
                "model_name": "kling-v1",
                "prompt": "a cat"
            })))
            .with_status(200)
            .with_body(
                json!({"code": 0, "data": {"task_id": "x", "task_status": "submitted"}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat")
            .mode(VideoMode::Text2Video)
            .reference_images(vec![FileInput::url("https://example.com/a.png")])
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn kling_video_create_image2video_without_reference_falls_back_to_generations() {
        // image2video 模式但无 reference_images → 回退到 /videos/generations 端点
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/videos/generations")
            .with_status(200)
            .with_body(
                json!({"code": 0, "data": {"task_id": "x", "task_status": "submitted"}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat")
            .mode(VideoMode::Image2Video)
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn kling_video_create_passes_extra_params() {
        // extra 字段透传到顶层（camera_control / mode 等厂商特有参数）
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/videos/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "camera_control": {"pan": "left"},
                "mode": "std"
            })))
            .with_status(200)
            .with_body(
                json!({"code": 0, "data": {"task_id": "x", "task_status": "submitted"}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat")
            .extra("camera_control", json!({"pan": "left"}))
            .extra("mode", "std")
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn kling_video_create_accepts_top_level_task_id() {
        // 任务 ID 在顶层 task_id（无 data 包裹）
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos/generations")
            .with_status(200)
            .with_body(json!({"task_id": "top-level-id", "task_status": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat").build();
        let task = adapter.video_create(req).await.unwrap();
        assert_eq!(task.task_id, "top-level-id");
        // queued → pending
        assert_eq!(task.status, TaskStatus::Pending);
    }

    #[tokio::test]
    async fn kling_video_create_uses_generated_id_when_missing() {
        // 响应缺任务 ID 时，回退到生成的 vid_ ID
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos/generations")
            .with_status(200)
            .with_body(json!({"code": 0, "data": {"task_status": "submitted"}}).to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat").build();
        let task = adapter.video_create(req).await.unwrap();
        assert!(task.task_id.starts_with("vid_"));
    }

    #[tokio::test]
    async fn kling_video_create_processing_status() {
        // processing 状态映射
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos/generations")
            .with_status(200)
            .with_body(
                json!({"code": 0, "data": {"task_id": "p-1", "task_status": "processing"}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat").build();
        let task = adapter.video_create(req).await.unwrap();
        assert_eq!(task.status, TaskStatus::Processing);
    }

    // ============ video_create 错误路径 ============

    #[tokio::test]
    async fn kling_video_create_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos/generations")
            .with_status(401)
            .with_body(json!({"message": "invalid key"}).to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Authentication { message } => assert!(message.contains("Kling")),
            _ => panic!("应为 Authentication"),
        }
    }

    #[tokio::test]
    async fn kling_video_create_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos/generations")
            .with_status(429)
            .with_body(json!({"message": "slow down"}).to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::RateLimit {
                message,
                retry_after,
            } => {
                assert!(message.contains("Kling"));
                assert!(retry_after.is_none());
            }
            _ => panic!("应为 RateLimit"),
        }
    }

    #[tokio::test]
    async fn kling_video_create_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos/generations")
            .with_status(404)
            .with_body(json!({"message": "not found"}).to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn kling_video_create_error_400_returns_validation() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos/generations")
            .with_status(400)
            .with_body(json!({"error": {"message": "invalid prompt"}}).to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Validation { message, .. } => {
                assert!(message.contains("invalid prompt"));
            }
            _ => panic!("应为 Validation"),
        }
    }

    #[tokio::test]
    async fn kling_video_create_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos/generations")
            .with_status(500)
            .with_body(json!({"error": {"message": "internal"}}).to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat").build();
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
    async fn kling_video_create_error_no_json_body_falls_back() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos/generations")
            .with_status(502)
            .with_body("Bad Gateway")
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("502")),
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn kling_video_create_error_extracts_top_level_message() {
        // Kling 错误体常用顶层 message 字段
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos/generations")
            .with_status(503)
            .with_body(json!({"message": "service unavailable"}).to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let req = VideoRequest::builder("kling-v1", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Api { message, .. } => assert_eq!(message, "service unavailable"),
            _ => panic!("应为 Api"),
        }
    }

    // ============ video_poll 正常路径 ============

    #[tokio::test]
    async fn kling_video_poll_success_returns_video_url() {
        let mut server = Server::new_async().await;
        let body = json!({
            "code": 0,
            "data": {
                "task_id": "vid-abc123",
                "task_status": "succeed",
                "task_result": {
                    "videos": [{"url": "https://cdn.example.com/video.mp4"}]
                },
                "created_at": 1700000000,
                "updated_at": 1700000300
            }
        });
        let mock = server
            .mock("GET", "/videos/generations/vid-abc123")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let status = adapter
            .video_poll("vid-abc123", "kling-v1")
            .await
            .expect("video_poll 应成功");

        assert_eq!(status.task_id, "vid-abc123");
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://cdn.example.com/video.mp4")
        );
        assert_eq!(status.progress, Some(100));
        assert_eq!(status.created_at, Some(1700000000));
        assert_eq!(status.updated_at, Some(1700000300));
        assert!(status.error.is_none());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn kling_video_poll_pending_status() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/generations/vid-1")
            .with_status(200)
            .with_body(
                json!({
                    "code": 0,
                    "data": {
                        "task_id": "vid-1",
                        "task_status": "submitted",
                        "created_at": 1700000000,
                        "updated_at": 1700000100
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let status = adapter.video_poll("vid-1", "kling-v1").await.unwrap();
        assert_eq!(status.status, TaskStatus::Pending);
        assert_eq!(status.progress, Some(0));
        assert!(status.video_url.is_none());
    }

    #[tokio::test]
    async fn kling_video_poll_processing_status() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/generations/vid-2")
            .with_status(200)
            .with_body(
                json!({
                    "code": 0,
                    "data": {
                        "task_id": "vid-2",
                        "task_status": "processing",
                        "created_at": 1700000000,
                        "updated_at": 1700000200
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let status = adapter.video_poll("vid-2", "kling-v1").await.unwrap();
        assert_eq!(status.status, TaskStatus::Processing);
        assert_eq!(status.progress, Some(50));
    }

    #[tokio::test]
    async fn kling_video_poll_failed_returns_error_msg() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/generations/vid-3")
            .with_status(200)
            .with_body(
                json!({
                    "code": 0,
                    "data": {
                        "task_id": "vid-3",
                        "task_status": "failed",
                        "task_status_msg": "Insufficient credits",
                        "created_at": 1700000000,
                        "updated_at": 1700000300
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let status = adapter.video_poll("vid-3", "kling-v1").await.unwrap();
        assert_eq!(status.status, TaskStatus::Failed);
        assert_eq!(status.error.as_deref(), Some("Insufficient credits"));
        // failed → progress 50
        assert_eq!(status.progress, Some(50));
    }

    #[tokio::test]
    async fn kling_video_poll_failed_extracts_error_field() {
        // task_status_msg 缺失时回退 error 字段
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/generations/vid-4")
            .with_status(200)
            .with_body(
                json!({
                    "code": 0,
                    "data": {
                        "task_id": "vid-4",
                        "task_status": "error",
                        "error": "internal failure"
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let status = adapter.video_poll("vid-4", "kling-v1").await.unwrap();
        assert_eq!(status.status, TaskStatus::Failed);
        assert_eq!(status.error.as_deref(), Some("internal failure"));
    }

    #[tokio::test]
    async fn kling_video_poll_queued_maps_to_pending() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/generations/vid-5")
            .with_status(200)
            .with_body(
                json!({"code": 0, "data": {"task_id": "vid-5", "task_status": "queued"}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let status = adapter.video_poll("vid-5", "kling-v1").await.unwrap();
        assert_eq!(status.status, TaskStatus::Pending);
    }

    #[tokio::test]
    async fn kling_video_poll_success_status_keyword() {
        // "success" 关键字也映射到 Success
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/generations/vid-6")
            .with_status(200)
            .with_body(
                json!({
                    "code": 0,
                    "data": {
                        "task_id": "vid-6",
                        "task_status": "success",
                        "task_result": {"videos": [{"url": "https://example.com/v.mp4"}]}
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let status = adapter.video_poll("vid-6", "kling-v1").await.unwrap();
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://example.com/v.mp4")
        );
    }

    #[tokio::test]
    async fn kling_video_poll_accepts_top_level_data() {
        // 无 data 包裹时，回退到顶层解析
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/generations/vid-7")
            .with_status(200)
            .with_body(
                json!({
                    "task_id": "vid-7",
                    "task_status": "succeed",
                    "task_result": {"videos": [{"url": "https://top.example.com/v.mp4"}]}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let status = adapter.video_poll("vid-7", "kling-v1").await.unwrap();
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://top.example.com/v.mp4")
        );
    }

    #[tokio::test]
    async fn kling_video_poll_updated_at_falls_back_to_current_timestamp() {
        // updated_at 缺失时回退当前时间戳（对齐 Python v1）
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/generations/vid-8")
            .with_status(200)
            .with_body(
                json!({
                    "code": 0,
                    "data": {
                        "task_id": "vid-8",
                        "task_status": "processing",
                        "created_at": 1700000000
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let status = adapter.video_poll("vid-8", "kling-v1").await.unwrap();
        assert_eq!(status.created_at, Some(1700000000));
        assert!(status.updated_at.is_some());
    }

    // ============ video_poll 错误路径 ============

    #[tokio::test]
    async fn kling_video_poll_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/generations/vid-x")
            .with_status(401)
            .with_body(json!({"message": "bad key"}).to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let err = adapter.video_poll("vid-x", "kling-v1").await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn kling_video_poll_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/generations/nonexistent")
            .with_status(404)
            .with_body(json!({"message": "task not found"}).to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let err = adapter
            .video_poll("nonexistent", "kling-v1")
            .await
            .unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn kling_video_poll_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/generations/vid-y")
            .with_status(429)
            .with_body(json!({"message": "slow down"}).to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let err = adapter.video_poll("vid-y", "kling-v1").await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn kling_video_poll_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/generations/vid-z")
            .with_status(500)
            .with_body(json!({"message": "internal"}).to_string())
            .create_async()
            .await;

        let adapter = make_kling(&server);
        let err = adapter.video_poll("vid-z", "kling-v1").await.unwrap_err();
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 500);
                assert!(message.contains("internal"));
            }
            _ => panic!("应为 Api"),
        }
    }

    // ============ 不支持的能力 ============

    #[tokio::test]
    async fn kling_chat_returns_unsupported() {
        let adapter = make_kling_no_server();
        let req = ChatRequest::builder("kling-v1", vec![]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn kling_image_generate_returns_unsupported() {
        let adapter = make_kling_no_server();
        let req = ImageRequest::builder("kling-v1", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn kling_embed_returns_unsupported() {
        let adapter = make_kling_no_server();
        let req = crate::model::options::EmbedRequest {
            model: "kling-v1".into(),
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
    async fn kling_list_models_returns_hardcoded() {
        let adapter = make_kling_no_server();
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].id, "kling-v1");
        assert_eq!(models[0].provider, "kling");
        assert_eq!(models[0].model_type, ModelType::Video);
        assert_eq!(models[1].id, "kling-v1-5");
        assert_eq!(models[2].id, "kling-v2");
    }

    #[tokio::test]
    async fn kling_list_models_filter_by_video_type() {
        let adapter = make_kling_no_server();
        let videos = adapter.list_models(Some(ModelType::Video)).await.unwrap();
        assert_eq!(videos.len(), 3);
        assert!(videos.iter().all(|m| m.model_type == ModelType::Video));
    }

    #[tokio::test]
    async fn kling_list_models_filter_by_image_returns_empty() {
        let adapter = make_kling_no_server();
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert!(images.is_empty());
    }

    #[tokio::test]
    async fn kling_list_models_filter_by_chat_returns_empty() {
        let adapter = make_kling_no_server();
        let chats = adapter.list_models(Some(ModelType::Chat)).await.unwrap();
        assert!(chats.is_empty());
    }

    // ============ start / close ============

    #[tokio::test]
    async fn kling_start_and_close_are_noops() {
        let mut adapter = make_kling_no_server();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    // ============ 错误映射单元测试 ============

    #[test]
    fn map_api_error_401_is_authentication() {
        let err = KlingAdapter::map_api_error(401, "{\"message\":\"bad\"}");
        match err {
            AibridgeError::Authentication { message } => assert!(message.contains("Kling")),
            _ => panic!("应为 Authentication"),
        }
    }

    #[test]
    fn map_api_error_429_is_rate_limit() {
        let err = KlingAdapter::map_api_error(429, "{}");
        match err {
            AibridgeError::RateLimit {
                message,
                retry_after,
            } => {
                assert!(message.contains("Kling"));
                assert!(retry_after.is_none());
            }
            _ => panic!("应为 RateLimit"),
        }
    }

    #[test]
    fn map_api_error_404_is_model_not_found() {
        let err = KlingAdapter::map_api_error(404, "{}");
        match err {
            AibridgeError::ModelNotFound { model } => assert!(model.contains("Kling")),
            _ => panic!("应为 ModelNotFound"),
        }
    }

    #[test]
    fn map_api_error_400_is_validation() {
        let err = KlingAdapter::map_api_error(400, "{\"error\":{\"message\":\"bad param\"}}");
        match err {
            AibridgeError::Validation { message, .. } => assert_eq!(message, "bad param"),
            _ => panic!("应为 Validation"),
        }
    }

    #[test]
    fn map_api_error_500_extracts_message() {
        let err = KlingAdapter::map_api_error(500, "{\"error\":{\"message\":\"internal\"}}");
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
        // Kling 错误体常用顶层 message
        let err = KlingAdapter::map_api_error(503, "{\"message\":\"unavailable\"}");
        match err {
            AibridgeError::Api { message, .. } => assert_eq!(message, "unavailable"),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_extracts_detail_field() {
        let err = KlingAdapter::map_api_error(422, "{\"detail\":\"validation failed\"}");
        match err {
            AibridgeError::Api { message, .. } => assert_eq!(message, "validation failed"),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_extracts_top_level_error_string() {
        let err = KlingAdapter::map_api_error(409, "{\"error\":\"conflict\"}");
        match err {
            AibridgeError::Api { message, .. } => assert_eq!(message, "conflict"),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_no_json_falls_back_to_http_status() {
        let err = KlingAdapter::map_api_error(502, "Bad Gateway");
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("502")),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_empty_body_falls_back_to_http_status() {
        let err = KlingAdapter::map_api_error(500, "");
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("500")),
            _ => panic!("应为 Api"),
        }
    }

    // ============ map_kling_status 单元测试 ============

    #[test]
    fn map_kling_status_pending_variants() {
        assert_eq!(map_kling_status("submitted"), TaskStatus::Pending);
        assert_eq!(map_kling_status("queued"), TaskStatus::Pending);
    }

    #[test]
    fn map_kling_status_processing() {
        assert_eq!(map_kling_status("processing"), TaskStatus::Processing);
    }

    #[test]
    fn map_kling_status_success_variants() {
        assert_eq!(map_kling_status("succeed"), TaskStatus::Success);
        assert_eq!(map_kling_status("success"), TaskStatus::Success);
    }

    #[test]
    fn map_kling_status_failed_variants() {
        assert_eq!(map_kling_status("failed"), TaskStatus::Failed);
        assert_eq!(map_kling_status("error"), TaskStatus::Failed);
    }

    #[test]
    fn map_kling_status_case_insensitive() {
        assert_eq!(map_kling_status("SUBMITTED"), TaskStatus::Pending);
        assert_eq!(map_kling_status("Processing"), TaskStatus::Processing);
        assert_eq!(map_kling_status("SUCCEED"), TaskStatus::Success);
        assert_eq!(map_kling_status("FAILED"), TaskStatus::Failed);
    }

    #[test]
    fn map_kling_status_unknown_defaults_to_pending() {
        assert_eq!(map_kling_status("unknown_state"), TaskStatus::Pending);
        assert_eq!(map_kling_status(""), TaskStatus::Pending);
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

    // ============ is_image2video_request 单元测试 ============

    #[test]
    fn is_image2video_request_true_for_image2video_with_url() {
        let req = VideoRequest::builder("kling-v1", "animate")
            .mode(VideoMode::Image2Video)
            .reference_images(vec![FileInput::url("https://example.com/a.png")])
            .build();
        assert!(is_image2video_request(&req));
    }

    #[test]
    fn is_image2video_request_false_for_text2video() {
        let req = VideoRequest::builder("kling-v1", "a cat")
            .mode(VideoMode::Text2Video)
            .reference_images(vec![FileInput::url("https://example.com/a.png")])
            .build();
        assert!(!is_image2video_request(&req));
    }

    #[test]
    fn is_image2video_request_false_for_image2video_without_reference() {
        let req = VideoRequest::builder("kling-v1", "a cat")
            .mode(VideoMode::Image2Video)
            .build();
        assert!(!is_image2video_request(&req));
    }

    #[test]
    fn is_image2video_request_false_for_image2video_with_path_input() {
        // Path 类型无法转 URL，回退到 text2video 端点
        let req = VideoRequest::builder("kling-v1", "a cat")
            .mode(VideoMode::Image2Video)
            .reference_images(vec![FileInput::path("/tmp/x.png")])
            .build();
        assert!(!is_image2video_request(&req));
    }

    // ============ build_video_body 单元测试 ============

    #[test]
    fn build_video_body_includes_model_name_and_prompt() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("kling", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = KlingAdapter::with_http(http, config);
        let req = VideoRequest::builder("kling-v1", "a cat").build();
        let body = adapter.build_video_body(&req);
        assert_eq!(body["model_name"], "kling-v1");
        assert_eq!(body["prompt"], "a cat");
        // text2video 不发 image 字段
        assert!(body.get("image").is_none());
    }

    #[test]
    fn build_video_body_image2video_sets_image_field() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("kling", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = KlingAdapter::with_http(http, config);
        let req = VideoRequest::builder("kling-v1", "animate")
            .mode(VideoMode::Image2Video)
            .reference_images(vec![FileInput::url("https://example.com/a.png")])
            .build();
        let body = adapter.build_video_body(&req);
        assert_eq!(body["image"], "https://example.com/a.png");
    }

    #[test]
    fn build_video_body_passes_optional_fields() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("kling", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = KlingAdapter::with_http(http, config);
        let req = VideoRequest::builder("kling-v2", "a cat")
            .negative_prompt("blurry")
            .cfg_scale(0.5)
            .duration(10)
            .aspect_ratio("16:9")
            .build();
        let body = adapter.build_video_body(&req);
        assert_eq!(body["negative_prompt"], "blurry");
        assert_eq!(body["cfg_scale"], 0.5);
        assert_eq!(body["duration"], 10);
        assert_eq!(body["aspect_ratio"], "16:9");
    }

    #[test]
    fn build_video_body_extra_passthrough() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("kling", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = KlingAdapter::with_http(http, config);
        let req = VideoRequest::builder("kling-v1", "a cat")
            .extra("camera_control", json!({"pan": "left"}))
            .extra("mode", "std")
            .build();
        let body = adapter.build_video_body(&req);
        assert_eq!(body["camera_control"]["pan"], "left");
        assert_eq!(body["mode"], "std");
    }
}
