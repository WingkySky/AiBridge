//! 火山引擎方舟（Seedream/Seedance）CV 适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/volcengine_cv.py`。
//!
//! 火山引擎方舟 CV 为**独立协议**（非 OpenAI 兼容），不复用 `OpenAiCompatAdapter`：
//! - 图像生成 (Seedream)：`POST /images/generations`（同步）
//! - 视频生成 (Seedance)：`POST /contents/generations/tasks`（异步任务，body 直传）
//! - 查询视频任务：`GET /contents/generations/tasks/{task_id}`
//! - 模型列表：`GET /models`（OpenAI 兼容端点，返回已开通模型）
//! - 认证：`Bearer <ARK_API_KEY>`
//!
//! 最近 3 个 Python commit 的关键对齐点（本实现已严格遵循）：
//! 1. **模型 ID + 视频端点 + 请求体结构**：使用方舟 Model ID（如 `doubao-seedream-4-0-250828`），
//!    视频走 `/contents/generations/tasks`，请求体为 `{"model", "content": [...]}` 结构。
//! 2. **图像 size 归一化到 Seedream 规范**：方舟 Seedream 对 size 有平台特定规范
//!    （最小 3686400 像素），小尺寸（如 1024x1024）会触发 503，需归一化到合法档。
//! 3. **视频创建改用方舟官方推荐的 body 直传方式**：参数（duration/ratio/resolution/seed/
//!    watermark/camera_fixed 等）直接放 request body 顶层（强校验），不再塞进 `extra`。

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::adapter::{Adapter, Capabilities, CapabilitySet};
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::common::{infer_model_type, ModelInfo, ModelType, TaskStatus};
use crate::model::image::{FileInput, ImageRequest, ImageResult};
use crate::model::video::{VideoRequest, VideoStatus, VideoTask};
use crate::util;

/// 方舟默认 Base URL
///
/// 对应 Python v1 `DEFAULT_BASE_URL`。
pub const DEFAULT_VOLCENGINE_BASE_URL: &str = "https://ark.cn-beijing.volces.com/api/v3";

// ==================== 方舟 Seedream 图像 size 规范 ====================
//
// 官方文档：https://www.volcengine.com/docs/82379/1541523
// - 方式 1（枚举）："2K" / "3K" / "4K"
// - 方式 2（像素值 WIDTHxHEIGHT）：需同时满足
//   * 总像素 ∈ [MIN_PIXELS, MAX_PIXELS]
//   * 宽高比 ∈ [MIN_RATIO, MAX_RATIO]

/// 方舟 Seedream 最小总像素（2560x1440 = 3686400）
const MIN_PIXELS: u64 = 3_686_400;
/// 方舟 Seedream 最大总像素（4096x4096 = 16777216）
const MAX_PIXELS: u64 = 16_777_216;
/// 方舟 Seedream 最小宽高比
const MIN_RATIO: f64 = 1.0 / 16.0;
/// 方舟 Seedream 最大宽高比
const MAX_RATIO: f64 = 16.0;

/// 方舟 Seedream 合法枚举值（大写匹配）
const SIZE_ENUMS: &[&str] = &["2K", "3K", "4K"];

/// 2K 推荐宽高像素值表（官方推荐表，最小合法档），按宽高比升序排列
///
/// 格式：`(宽高比 ratio, 宽, 高)`。参考方舟推荐宽高像素值表。
/// 不合法尺寸按最接近的宽高比映射到此表的某个档。
static PRESETS_2K: &[(f64, u32, u32)] = &[
    (9.0 / 16.0, 1600, 2848), // 9:16
    (2.0 / 3.0, 1664, 2496),  // 2:3
    (3.0 / 4.0, 1728, 2304),  // 3:4
    (1.0, 2048, 2048),        // 1:1
    (4.0 / 3.0, 2304, 1728),  // 4:3
    (3.0 / 2.0, 2496, 1664),  // 3:2
    (16.0 / 9.0, 2848, 1600), // 16:9
    (21.0 / 9.0, 3136, 1344), // 21:9
];

/// 火山引擎方舟 CV 适配器（Seedream 图像 / Seedance 视频）
///
/// 持有 HTTP 客户端与 Provider 配置，实现方舟独立协议。
/// 不复用 `OpenAiCompatAdapter`（方舟图像/视频端点与请求体结构与 OpenAI 不一致）。
pub struct VolcengineCvAdapter {
    /// HTTP 客户端（封装 reqwest，含连接池与超时）
    http: HttpClient,
    /// Provider 配置
    config: ProviderConfig,
    /// 实际 base_url（已合并 config.base_url 与默认值）
    base_url: String,
    /// 支持的能力集合
    capabilities: CapabilitySet,
}

impl VolcengineCvAdapter {
    /// 创建火山引擎方舟 CV 适配器
    ///
    /// `config.base_url` 为空时用 `DEFAULT_VOLCENGINE_BASE_URL` 兜底。
    /// `config.api_key` 为空时不在此处报错（由上层按 `requires_api_key` 校验）。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_VOLCENGINE_BASE_URL.to_string());

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
            .unwrap_or_else(|| DEFAULT_VOLCENGINE_BASE_URL.to_string());
        Self {
            http,
            config,
            base_url,
            capabilities: Self::default_capabilities(),
        }
    }

    /// 默认能力集合：图像生成 + 视频生成
    fn default_capabilities() -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::ImageGenerate);
        caps.insert(Capabilities::VideoGenerate);
        caps.insert(Capabilities::VideoText2Video);
        caps.insert(Capabilities::VideoImage2Video);
        caps
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

    /// 归一化图像 size 到方舟 Seedream 规范
    ///
    /// 移植自 Python v1 `_normalize_image_size`。
    ///
    /// 规则：
    /// - 空/非法 → 默认 `2048x2048`（1:1 的 2K 推荐档）
    /// - 枚举值（2K/3K/4K）原样透传
    /// - `WIDTHxHEIGHT`（兼容 `x`/`X`/`*` 分隔符）：已合法（总像素与宽高比均在范围内）原样透传；
    ///   不合法则按最接近的宽高比映射到 2K 推荐档
    pub fn normalize_image_size(size: &str) -> String {
        let s = size.trim();
        if s.is_empty() {
            return "2048x2048".to_string();
        }
        let upper = s.to_uppercase();

        // 方式 1：枚举值原样透传
        if SIZE_ENUMS.contains(&upper.as_str()) {
            return s.to_string();
        }

        // 方式 2：解析 WIDTHxHEIGHT（兼容 x / X / * 分隔符）
        let normalized = upper.replace(['X', '*'], "x");
        let parts: Vec<&str> = normalized.split('x').collect();
        if parts.len() != 2 {
            return "2048x2048".to_string();
        }
        let (w, h) = match (parts[0].parse::<u64>(), parts[1].parse::<u64>()) {
            (Ok(w), Ok(h)) => (w, h),
            _ => return "2048x2048".to_string(),
        };
        if w == 0 || h == 0 {
            return "2048x2048".to_string();
        }

        let total = w * h;
        let ratio = w as f64 / h as f64;

        // 已合法：原样透传
        if (MIN_PIXELS..=MAX_PIXELS).contains(&total) && (MIN_RATIO..=MAX_RATIO).contains(&ratio) {
            return format!("{w}x{h}");
        }

        // 不合法：按最接近的宽高比映射到 2K 推荐档
        let (best_w, best_h) = PRESETS_2K
            .iter()
            .min_by(|a, b| {
                let da = (ratio - a.0).abs();
                let db = (ratio - b.0).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, w, h)| (*w, *h))
            .unwrap_or((2048, 2048));
        format!("{best_w}x{best_h}")
    }

    /// 校验请求的能力是否被支持（不支持则返 UnsupportedCapability）
    fn ensure_capability(&self, cap: Capabilities) -> Result<()> {
        if self.capabilities.contains(&cap) {
            Ok(())
        } else {
            Err(AibridgeError::UnsupportedCapability {
                capability: format!("{} (provider: volcengine_cv)", cap.as_str()),
            })
        }
    }

    /// 发送带 Bearer 认证的 POST JSON 请求，并用方舟错误映射处理响应
    async fn post_authed_json(&self, path: &str, body: &Value) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .http
            .inner()
            .post(&url)
            .bearer_auth(self.api_key())
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

    /// 发送带 Bearer 认证的 GET 请求，并用方舟错误映射处理响应
    async fn get_authed_json(&self, path: &str) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .http
            .inner()
            .get(&url)
            .bearer_auth(self.api_key())
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

    // ==================== 内部：请求体构造 ====================

    /// 构造方舟图像生成请求体
    ///
    /// 移植自 Python v1 `image_generate`：
    /// - size 归一化到 Seedream 规范（默认 1024x1024 → 归一化到 2K 档）
    /// - n / response_format / negative_prompt / seed 透传
    /// - extra 字段合并到顶层
    fn build_image_body(&self, req: &ImageRequest) -> Value {
        // 用户未指定 size 时默认 1024x1024（与 Python 一致），再归一化
        let raw_size = req.size.clone().unwrap_or_else(|| "1024x1024".to_string());
        let size = Self::normalize_image_size(&raw_size);

        let mut body = json!({
            "model": req.model,
            "prompt": req.prompt,
            "size": size,
            "n": req.n,
            "response_format": req.response_format,
        });
        if let Some(np) = &req.negative_prompt {
            body["negative_prompt"] = json!(np);
        }
        if let Some(seed) = req.seed {
            body["seed"] = json!(seed);
        }
        // extra 透传（合并到顶层）
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        body
    }

    /// 构造方舟视频生成请求体（body 直传方式）
    ///
    /// 移植自 Python v1 `video_create`（对齐方舟官方推荐 body 直传）：
    /// - `content` 数组：文本必选，image2video 模式附加首帧 `image_url`
    /// - 视频格式参数直接放 body 顶层（强校验）：
    ///   duration / ratio(来自 aspect_ratio) / resolution / seed /
    ///   watermark / camera_fixed(来自 first_frame 之外的字段，见下)
    /// - 高级参数：generate_audio / service_tier / priority / draft
    /// - extra 透传到顶层（覆盖同名字段）
    ///
    /// 注意：方舟用 `ratio` 字段表示宽高比，对应统一 `VideoRequest.aspect_ratio`。
    /// `camera_fixed` 来自统一字段 `camera_motion` 为 "fixed" 时为 true，或来自 extra.camerafixed。
    fn build_video_body(&self, req: &VideoRequest) -> Value {
        // content 数组：文本必选
        let mut content = vec![json!({ "type": "text", "text": req.prompt })];

        // image2video 模式：附加首帧 image_url
        // 方舟仅支持首帧（reference_images[0]），与 Python 老版一致
        let is_image2video = matches!(req.mode, crate::model::common::VideoMode::Image2Video);
        if is_image2video {
            if let Some(url) = req.reference_images.first().and_then(file_input_url) {
                content.push(json!({
                    "type": "image_url",
                    "image_url": { "url": url }
                }));
            } else if let Some(ff) = req.first_frame.as_ref().and_then(file_input_url) {
                content.push(json!({
                    "type": "image_url",
                    "image_url": { "url": ff }
                }));
            }
        }

        let mut body = json!({
            "model": req.model,
            "content": content,
        });

        // 视频格式参数（body 直传）
        if let Some(d) = req.duration {
            body["duration"] = json!(d);
        }
        if let Some(ar) = &req.aspect_ratio {
            // 方舟用 ratio 字段表示宽高比
            body["ratio"] = json!(ar);
        }
        if let Some(r) = &req.resolution {
            body["resolution"] = json!(r);
        }
        if let Some(seed) = req.seed {
            body["seed"] = json!(seed);
        }
        if let Some(wm) = req.watermark {
            body["watermark"] = json!(wm);
        }
        // camera_fixed：统一字段无直接映射，从 camera_motion == "fixed" 推断，或来自 extra
        if req.camera_motion.as_deref() == Some("fixed") {
            body["camera_fixed"] = json!(true);
        }

        // 高级参数（仅部分模型支持，统一模型无直接字段，走 extra 透传）
        // extra 透传到顶层（覆盖同名字段）
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        body
    }

    // ==================== 内部：响应解析 ====================

    /// 解析方舟图像生成响应 → ImageResult
    ///
    /// 方舟图像响应结构与 OpenAI 兼容（`{data: [{url/b64_json/revised_prompt}]}`）。
    fn parse_image_result(value: &Value, fallback_model: &str) -> Result<ImageResult> {
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
            .map(|arr| arr.iter().map(parse_image_data_item).collect())
            .unwrap_or_default();
        Ok(ImageResult {
            id,
            object,
            created,
            model,
            data,
        })
    }

    /// 解析方舟视频任务创建响应 → VideoTask
    ///
    /// 方舟创建任务响应：`{"id": "cgt-xxx", "status": "queued", ...}`
    fn parse_video_task(value: &Value, model: &str) -> Result<VideoTask> {
        let task_id = value
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| util::generate_id("vtask"));
        let raw_status = value
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("queued");
        Ok(VideoTask {
            task_id,
            model: model.to_string(),
            status: map_video_status(raw_status),
            created_at: util::current_timestamp(),
        })
    }

    /// 解析方舟视频任务查询响应 → VideoStatus
    ///
    /// 移植自 Python v1 `_parse_video_status`：
    /// - 视频成功时 URL 在 `content.video_url`（方舟新协议），兼容 `video_url` /
    ///   `output.video_url` / `url` 旧路径
    /// - 失败时错误信息在 `error.message` / `message` / `error`
    fn parse_video_status(value: &Value, task_id: &str) -> VideoStatus {
        let raw_status = value.get("status").and_then(|v| v.as_str()).unwrap_or("");
        let status = map_video_status(raw_status);

        // 视频 URL：方舟 content.video_url 优先，兼容其他路径
        let video_url = if status == TaskStatus::Success {
            value
                .get("content")
                .and_then(|c| c.get("video_url"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .or_else(|| {
                    value
                        .get("video_url")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned)
                })
                .or_else(|| {
                    value
                        .get("output")
                        .and_then(|o| o.get("video_url"))
                        .and_then(|v| v.as_str())
                        .map(str::to_owned)
                })
                .or_else(|| value.get("url").and_then(|v| v.as_str()).map(str::to_owned))
        } else {
            None
        };

        // 错误信息：error.message / message / error
        let error = if status == TaskStatus::Failed {
            value
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .or_else(|| {
                    value
                        .get("message")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned)
                })
                .or_else(|| {
                    value
                        .get("error")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned)
                })
        } else {
            None
        };

        let progress = value
            .get("progress")
            .and_then(|v| v.as_u64())
            .map(|p| p as u32);
        let created_at = value.get("created").and_then(|v| v.as_u64());
        let updated_at = value
            .get("updated")
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

    /// 解析方舟 /models 响应 → Vec<ModelInfo>
    ///
    /// 方舟 /models 为 OpenAI 兼容端点：`{"data": [{"id": "...", "created": ..., "owned_by": "..."}]}`。
    /// 返回的模型 ID 为方舟规范格式（如 `doubao-seedream-4-0-250828`）。
    fn parse_models(value: &Value) -> Vec<ModelInfo> {
        match value.get("data").and_then(|v| v.as_array()) {
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
                        provider: "volcengine_cv".to_string(),
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

    /// 将方舟 API 错误响应映射为 AibridgeError
    ///
    /// 移植自 Python v1 `_handle_error`：
    /// - 401 → Authentication（"Invalid Volcengine API key or access denied"）
    /// - 429 → RateLimit
    /// - 404 → Api（模型端点未开通，提示检查方舟控制台 endpoint ID）
    /// - 其余 ≥400 → Api（提取 error.message / message / error，回退 `HTTP {status}`）
    pub fn map_api_error(status: u16, body: &str) -> AibridgeError {
        match status {
            401 => AibridgeError::Authentication {
                message: "Invalid Volcengine API key or access denied".to_string(),
            },
            403 => AibridgeError::Authentication {
                message: "Volcengine access denied".to_string(),
            },
            429 => AibridgeError::RateLimit {
                message: "Volcengine rate limit exceeded".to_string(),
                retry_after: None,
            },
            404 => AibridgeError::Api {
                status,
                message:
                    "Model endpoint not found. Check your endpoint ID in Volcengine Ark console."
                        .to_string(),
            },
            _ => {
                let message = parse_error_message(body, status);
                AibridgeError::Api { status, message }
            }
        }
    }
}

#[async_trait]
impl Adapter for VolcengineCvAdapter {
    fn provider_type(&self) -> &str {
        "volcengine_cv"
    }

    fn provider_name(&self) -> &str {
        "火山引擎 CV"
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

    /// 图像生成（Seedream 文生图）
    ///
    /// `POST /images/generations`，size 归一化到 Seedream 规范。
    async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
        self.ensure_capability(Capabilities::ImageGenerate)?;
        let body = self.build_image_body(&req);
        let value = self.post_authed_json("images/generations", &body).await?;
        Self::parse_image_result(&value, &req.model)
    }

    /// 创建视频生成任务（Seedance，body 直传方式）
    ///
    /// `POST /contents/generations/tasks`，参数直接放 request body 顶层（强校验）。
    async fn video_create(&self, req: VideoRequest) -> Result<VideoTask> {
        self.ensure_capability(Capabilities::VideoGenerate)?;
        let body = self.build_video_body(&req);
        let value = self
            .post_authed_json("contents/generations/tasks", &body)
            .await?;
        Self::parse_video_task(&value, &req.model)
    }

    /// 查询视频任务状态
    ///
    /// `GET /contents/generations/tasks/{task_id}`。
    async fn video_poll(&self, task_id: &str, _model: &str) -> Result<VideoStatus> {
        self.ensure_capability(Capabilities::VideoGenerate)?;
        let path = format!("contents/generations/tasks/{task_id}");
        let value = self.get_authed_json(&path).await?;
        Ok(Self::parse_video_status(&value, task_id))
    }

    /// 模型列表（实时拉取方舟已开通模型）
    ///
    /// `GET /models`，按 `filter` 过滤模型类型。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let value = self.get_authed_json("models").await?;
        let models = Self::parse_models(&value);
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }
}

// ==================== 辅助函数 ====================

/// 从 FileInput 提取 URL 字符串
///
/// - `Url(s)` → `Some(s)`
/// - `Base64(s)` → `Some(s)`（方舟接受 base64 字符串作为 image_url）
/// - `Path`/`Bytes` → `None`（方舟仅接受 URL/base64，本地路径与字节需调用方先上传）
fn file_input_url(f: &FileInput) -> Option<String> {
    match f {
        FileInput::Url(s) | FileInput::Base64(s) => Some(s.clone()),
        FileInput::Path(_) | FileInput::Bytes(_) => None,
    }
}

/// 映射方舟视频状态字符串到统一 TaskStatus
///
/// 移植自 Python v1 `_map_video_status`。
fn map_video_status(raw: &str) -> TaskStatus {
    match raw.to_lowercase().as_str() {
        "queued" | "pending" | "submitted" => TaskStatus::Pending,
        "processing" | "running" | "in_progress" => TaskStatus::Processing,
        "succeeded" | "success" | "completed" => TaskStatus::Success,
        "failed" | "error" | "cancelled" => TaskStatus::Failed,
        _ => TaskStatus::Pending,
    }
}

/// 解析单个 ImageData（方舟图像响应项）
///
/// 字段与 OpenAI 兼容：url / b64_json / revised_prompt。
fn parse_image_data_item(v: &Value) -> crate::model::image::ImageData {
    crate::model::image::ImageData {
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

/// 解析方舟错误体中的 message 字段
///
/// 方舟错误格式：`{"error": {"message": "..."}}` 或顶层 `{"message": "..."}`。
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
        if let Some(msg) = v.get("message").and_then(|m| m.as_str()) {
            return msg.to_string();
        }
        if let Some(msg) = v.get("error").and_then(|m| m.as_str()) {
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
    use crate::config::ClientOptions;
    use crate::model::common::VideoMode;
    use mockito::Server;
    use serde_json::json;

    /// 构造测试用 VolcengineCvAdapter（指向 mockito server）
    fn make_adapter(server: &Server) -> VolcengineCvAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-ark-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("volcengine_cv", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        VolcengineCvAdapter::with_http(http, config)
    }

    /// 全能力集合（image + video）
    fn full_caps() -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::ImageGenerate);
        caps.insert(Capabilities::VideoGenerate);
        caps.insert(Capabilities::VideoText2Video);
        caps.insert(Capabilities::VideoImage2Video);
        caps
    }

    // ============ normalize_image_size 单元测试 ============

    #[test]
    fn normalize_size_empty_returns_default() {
        assert_eq!(VolcengineCvAdapter::normalize_image_size(""), "2048x2048");
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("   "),
            "2048x2048"
        );
    }

    #[test]
    fn normalize_size_enum_passthrough() {
        assert_eq!(VolcengineCvAdapter::normalize_image_size("2K"), "2K");
        assert_eq!(VolcengineCvAdapter::normalize_image_size("3K"), "3K");
        assert_eq!(VolcengineCvAdapter::normalize_image_size("4K"), "4K");
        // 小写也接受（匹配时大写化）
        assert_eq!(VolcengineCvAdapter::normalize_image_size("2k"), "2k");
    }

    #[test]
    fn normalize_size_valid_pixels_passthrough() {
        // 2560x1440 = 3686400，正好最小像素，1:16/16:1 范围内
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("2560x1440"),
            "2560x1440"
        );
        // 2048x2048 = 4194304，合法
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("2048x2048"),
            "2048x2048"
        );
    }

    #[test]
    fn normalize_size_small_1024_maps_to_2k_preset() {
        // 1024x1024 = 1048576 < MIN_PIXELS，不合法，按最接近宽高比映射
        // 1:1 对应 2048x2048
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("1024x1024"),
            "2048x2048"
        );
    }

    #[test]
    fn normalize_size_small_16x9_maps_to_2848x1600() {
        // 1280x720 = 921600 < MIN_PIXELS，16:9 对应 2848x1600
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("1280x720"),
            "2848x1600"
        );
    }

    #[test]
    fn normalize_size_small_9x16_maps_to_1600x2848() {
        // 720x1280，9:16 对应 1600x2848
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("720x1280"),
            "1600x2848"
        );
    }

    #[test]
    fn normalize_size_accepts_uppercase_x_and_star() {
        // 1024X1024 与 1024*1024 等价于 1024x1024
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("2048X2048"),
            "2048x2048"
        );
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("2048*2048"),
            "2048x2048"
        );
    }

    #[test]
    fn normalize_size_invalid_returns_default() {
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("abc"),
            "2048x2048"
        );
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("1024"),
            "2048x2048"
        );
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("1024x"),
            "2048x2048"
        );
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("0x1024"),
            "2048x2048"
        );
    }

    #[test]
    fn normalize_size_too_large_maps_to_preset() {
        // 8192x8192 = 67108864 > MAX_PIXELS，不合法；1:1 → 2048x2048
        assert_eq!(
            VolcengineCvAdapter::normalize_image_size("8192x8192"),
            "2048x2048"
        );
    }

    // ============ image_generate 正常路径 ============

    #[tokio::test]
    async fn image_generate_success_parses_url() {
        let mut server = Server::new_async().await;
        let body = json!({
            "created": 1700000000,
            "data": [{
                "url": "https://ark.example.com/img.png",
                "revised_prompt": "a cute cat"
            }]
        });
        let mock = server
            .mock("POST", "/images/generations")
            .match_header("authorization", "Bearer test-ark-key")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "doubao-seedream-4-0-250828",
                "prompt": "a cat",
                "size": "2048x2048",
                "n": 1,
                "response_format": "url"
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("doubao-seedream-4-0-250828", "a cat")
            .size("1024x1024") // 小尺寸，应被归一化到 2048x2048
            .build();
        let resp = adapter.image_generate(req).await.expect("image 应成功");

        assert_eq!(resp.data.len(), 1);
        assert_eq!(
            resp.data[0].url.as_deref(),
            Some("https://ark.example.com/img.png")
        );
        assert_eq!(resp.data[0].revised_prompt.as_deref(), Some("a cute cat"));
        assert_eq!(resp.model, "doubao-seedream-4-0-250828");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_size_normalization_applied() {
        // 验证：用户传 1024x1024，实际发送 2048x2048（归一化生效）
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/images/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "size": "2048x2048"
            })))
            .with_status(200)
            .with_body(
                json!({
                    "created": 1,
                    "data": [{"url": "https://x/img.png"}]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("doubao-seedream-4-0-250828", "cat")
            .size("1024x1024")
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_enum_size_passthrough() {
        // 2K 枚举值原样透传
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/images/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "size": "2K"
            })))
            .with_status(200)
            .with_body(json!({"created": 1, "data": [{"url": "https://x"}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("doubao-seedream-4-0-250828", "cat")
            .size("2K")
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_b64_response() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/images/generations")
            .with_status(200)
            .with_body(
                json!({
                    "created": 1700000000,
                    "data": [{"b64_json": "aGVsbG8="}]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("doubao-seedream-4-0-250828", "cat").build();
        let resp = adapter.image_generate(req).await.unwrap();
        assert_eq!(resp.data[0].b64_json.as_deref(), Some("aGVsbG8="));
    }

    #[tokio::test]
    async fn image_generate_passes_negative_prompt_and_seed() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/images/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "negative_prompt": "blurry",
                "seed": 42
            })))
            .with_status(200)
            .with_body(json!({"created": 1, "data": [{"url": "https://x"}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("doubao-seedream-4-0-250828", "cat")
            .negative_prompt("blurry")
            .seed(42)
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn image_generate_passes_extra_params() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/images/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "guidance_scale": 7.5
            })))
            .with_status(200)
            .with_body(json!({"created": 1, "data": [{"url": "https://x"}]}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("doubao-seedream-4-0-250828", "cat")
            .extra("guidance_scale", 7.5)
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    // ============ image_generate 错误路径 ============

    #[tokio::test]
    async fn image_generate_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/images/generations")
            .with_status(401)
            .with_body(json!({"error": {"message": "invalid key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("doubao-seedream-4-0-250828", "cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn image_generate_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/images/generations")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("doubao-seedream-4-0-250828", "cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn image_generate_error_404_returns_api_with_hint() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/images/generations")
            .with_status(404)
            .with_body(json!({"error": {"message": "not found"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("doubao-seedream-4-0-250828", "cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 404);
                assert!(message.contains("Ark console"));
            }
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn image_generate_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/images/generations")
            .with_status(500)
            .with_body(json!({"error": {"message": "internal"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("doubao-seedream-4-0-250828", "cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 500);
                assert!(message.contains("internal"));
            }
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn image_generate_error_no_json_body_falls_back() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/images/generations")
            .with_status(502)
            .with_body("Bad Gateway")
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("doubao-seedream-4-0-250828", "cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("502")),
            _ => panic!("应为 Api"),
        }
    }

    // ============ video_create body 直传 正常路径 ============

    #[tokio::test]
    async fn video_create_text2video_body_direct() {
        // 验证 body 直传结构：model + content[text]，参数在顶层
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/contents/generations/tasks")
            .match_header("authorization", "Bearer test-ark-key")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "doubao-seedance-1-0-pro-250528",
                "content": [{"type": "text", "text": "a cat walking"}],
                "duration": 5,
                "ratio": "16:9",
                "resolution": "1080p",
                "seed": 123
            })))
            .with_status(200)
            .with_body(
                json!({
                    "id": "cgt-abc123",
                    "status": "queued"
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = VideoRequest::builder("doubao-seedance-1-0-pro-250528", "a cat walking")
            .duration(5)
            .aspect_ratio("16:9")
            .resolution("1080p")
            .seed(123)
            .build();
        let task = adapter
            .video_create(req)
            .await
            .expect("video_create 应成功");

        assert_eq!(task.task_id, "cgt-abc123");
        assert_eq!(task.model, "doubao-seedance-1-0-pro-250528");
        assert_eq!(task.status, TaskStatus::Pending);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_create_image2video_with_first_frame_url() {
        // image2video 模式：content 含 text + image_url
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/contents/generations/tasks")
            .match_body(mockito::Matcher::PartialJson(json!({
                "content": [
                    {"type": "text", "text": "animate this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
                ]
            })))
            .with_status(200)
            .with_body(json!({"id": "cgt-1", "status": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = VideoRequest::builder("doubao-seedance-1-0-pro-250528", "animate this")
            .mode(VideoMode::Image2Video)
            .reference_images(vec![FileInput::url("https://example.com/a.png")])
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_create_image2video_with_base64_first_frame() {
        // base64 形式的首帧也能作为 image_url
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/contents/generations/tasks")
            .match_body(mockito::Matcher::PartialJson(json!({
                "content": [
                    {"type": "text", "text": "go"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}
                ]
            })))
            .with_status(200)
            .with_body(json!({"id": "cgt-2", "status": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = VideoRequest::builder("doubao-seedance-1-0-pro-250528", "go")
            .mode(VideoMode::Image2Video)
            .first_frame(FileInput::base64("data:image/png;base64,xxx"))
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_create_watermark_and_camera_fixed() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/contents/generations/tasks")
            .match_body(mockito::Matcher::PartialJson(json!({
                "watermark": false,
                "camera_fixed": true
            })))
            .with_status(200)
            .with_body(json!({"id": "cgt-3", "status": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = VideoRequest::builder("doubao-seedance-1-0-pro-250528", "cat")
            .watermark(false)
            .camera_motion("fixed")
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_create_extra_params_passthrough() {
        // generate_audio / service_tier / priority / draft 走 extra 透传
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/contents/generations/tasks")
            .match_body(mockito::Matcher::PartialJson(json!({
                "generate_audio": true,
                "service_tier": "flex",
                "priority": 5,
                "draft": false
            })))
            .with_status(200)
            .with_body(json!({"id": "cgt-4", "status": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = VideoRequest::builder("doubao-seedance-1-0-pro-250528", "cat")
            .extra("generate_audio", true)
            .extra("service_tier", "flex")
            .extra("priority", 5)
            .extra("draft", false)
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_create_uses_default_id_when_missing() {
        // 方舟返回体缺 id 时，回退到生成的 vtask ID
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/contents/generations/tasks")
            .with_status(200)
            .with_body(json!({"status": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = VideoRequest::builder("doubao-seedance-1-0-pro-250528", "cat").build();
        let task = adapter.video_create(req).await.unwrap();
        assert!(task.task_id.starts_with("vtask_"));
    }

    // ============ video_create 错误路径 ============

    #[tokio::test]
    async fn video_create_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/contents/generations/tasks")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = VideoRequest::builder("doubao-seedance-1-0-pro-250528", "cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn video_create_error_429() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/contents/generations/tasks")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = VideoRequest::builder("doubao-seedance-1-0-pro-250528", "cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn video_create_error_404_hint() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/contents/generations/tasks")
            .with_status(404)
            .with_body(json!({"error": {"message": "endpoint not found"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = VideoRequest::builder("doubao-seedance-x-250528", "cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("Ark console")),
            _ => panic!("应为 Api"),
        }
    }

    // ============ video_poll 正常路径 ============

    #[tokio::test]
    async fn video_poll_success_extracts_content_video_url() {
        // 方舟新协议：视频 URL 在 content.video_url
        let mut server = Server::new_async().await;
        let mock = server
            .mock("GET", "/contents/generations/tasks/cgt-abc")
            .match_header("authorization", "Bearer test-ark-key")
            .with_status(200)
            .with_body(
                json!({
                    "id": "cgt-abc",
                    "status": "succeeded",
                    "content": {"video_url": "https://ark.example.com/v.mp4"},
                    "progress": 100,
                    "created": 1700000000,
                    "updated": 1700000100
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let status = adapter
            .video_poll("cgt-abc", "doubao-seedance-1-0-pro-250528")
            .await
            .expect("video_poll 应成功");

        assert_eq!(status.task_id, "cgt-abc");
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://ark.example.com/v.mp4")
        );
        assert_eq!(status.progress, Some(100));
        assert_eq!(status.created_at, Some(1700000000));
        assert_eq!(status.updated_at, Some(1700000100));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_poll_processing_status() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/contents/generations/tasks/cgt-1")
            .with_status(200)
            .with_body(
                json!({
                    "id": "cgt-1",
                    "status": "running",
                    "progress": 45
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let status = adapter.video_poll("cgt-1", "m").await.unwrap();
        assert_eq!(status.status, TaskStatus::Processing);
        assert_eq!(status.progress, Some(45));
        assert!(status.video_url.is_none());
    }

    #[tokio::test]
    async fn video_poll_failed_extracts_error_message() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/contents/generations/tasks/cgt-2")
            .with_status(200)
            .with_body(
                json!({
                    "id": "cgt-2",
                    "status": "failed",
                    "error": {"message": "content policy violation"}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let status = adapter.video_poll("cgt-2", "m").await.unwrap();
        assert_eq!(status.status, TaskStatus::Failed);
        assert_eq!(status.error.as_deref(), Some("content policy violation"));
    }

    #[tokio::test]
    async fn video_poll_failed_extracts_top_level_message() {
        // 兼容顶层 message 字段
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/contents/generations/tasks/cgt-3")
            .with_status(200)
            .with_body(
                json!({
                    "id": "cgt-3",
                    "status": "error",
                    "message": "internal error"
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let status = adapter.video_poll("cgt-3", "m").await.unwrap();
        assert_eq!(status.status, TaskStatus::Failed);
        assert_eq!(status.error.as_deref(), Some("internal error"));
    }

    #[tokio::test]
    async fn video_poll_compatible_video_url_paths() {
        // 兼容旧版 video_url / output.video_url / url 路径
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/contents/generations/tasks/cgt-4")
            .with_status(200)
            .with_body(
                json!({
                    "id": "cgt-4",
                    "status": "completed",
                    "output": {"video_url": "https://legacy.example.com/v.mp4"}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let status = adapter.video_poll("cgt-4", "m").await.unwrap();
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://legacy.example.com/v.mp4")
        );
    }

    #[tokio::test]
    async fn video_poll_queued_status_maps_to_pending() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/contents/generations/tasks/cgt-5")
            .with_status(200)
            .with_body(json!({"id": "cgt-5", "status": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let status = adapter.video_poll("cgt-5", "m").await.unwrap();
        assert_eq!(status.status, TaskStatus::Pending);
    }

    #[tokio::test]
    async fn video_poll_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/contents/generations/tasks/cgt-x")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let err = adapter.video_poll("cgt-x", "m").await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    // ============ list_models ============

    #[tokio::test]
    async fn list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .match_header("authorization", "Bearer test-ark-key")
            .with_status(200)
            .with_body(json!({
                "object": "list",
                "data": [
                    {"id": "doubao-seedream-4-0-250828", "object": "model", "created": 1, "owned_by": "volcengine"},
                    {"id": "doubao-seedance-1-0-pro-250528", "object": "model", "created": 1, "owned_by": "volcengine"},
                    {"id": "doubao-pro-32k", "object": "model", "created": 1, "owned_by": "volcengine"}
                ]
            }).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 3);
        // 类型推断：seedream→image，seedance→video，doubao-pro→chat
        assert_eq!(models[0].model_type, ModelType::Image);
        assert_eq!(models[1].model_type, ModelType::Video);
        assert_eq!(models[2].model_type, ModelType::Chat);
        // provider 字段填充
        assert_eq!(models[0].provider, "volcengine_cv");
    }

    #[tokio::test]
    async fn list_models_filter_image() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "doubao-seedream-4-0-250828", "created": 1},
                        {"id": "doubao-seedance-1-0-pro-250528", "created": 1}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "doubao-seedream-4-0-250828");
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
    async fn list_models_empty_data() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(json!({"data": []}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert!(models.is_empty());
    }

    // ============ Adapter trait 元信息 ============

    #[tokio::test]
    async fn provider_metadata() {
        let server = Server::new_async().await;
        let adapter = make_adapter(&server);
        assert_eq!(adapter.provider_type(), "volcengine_cv");
        assert_eq!(adapter.provider_name(), "火山引擎 CV");
        assert!(adapter.requires_api_key());
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::ImageGenerate));
        assert!(caps.contains(&Capabilities::VideoGenerate));
    }

    #[tokio::test]
    async fn unsupported_chat_returns_error() {
        let server = Server::new_async().await;
        let adapter = make_adapter(&server);
        let req = crate::model::chat::ChatRequest::builder("m", vec![]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn unsupported_embed_returns_error() {
        let server = Server::new_async().await;
        let adapter = make_adapter(&server);
        let req = crate::model::options::EmbedRequest {
            model: "m".into(),
            input: crate::model::options::EmbedInput::Single("hi".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: std::collections::HashMap::new(),
        };
        let err = adapter.embed(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn start_and_close_are_noops() {
        let mut server = Server::new_async().await;
        let mut adapter = make_adapter(&server);
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
        // 抑制未使用 mut 警告
        let _ = &mut server;
    }

    // ============ base_url 选择 ============

    #[test]
    fn base_url_uses_config_when_provided() {
        let config = ProviderConfig::from_options(
            "volcengine_cv",
            ClientOptions::builder()
                .api_key("k")
                .base_url("https://custom.ark.example.com/api/v3")
                .build(),
        );
        let adapter = VolcengineCvAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), "https://custom.ark.example.com/api/v3");
    }

    #[test]
    fn base_url_falls_back_to_default_when_missing() {
        let config = ProviderConfig::from_options(
            "volcengine_cv",
            ClientOptions::builder().api_key("k").build(),
        );
        let adapter = VolcengineCvAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), DEFAULT_VOLCENGINE_BASE_URL);
    }

    // ============ 错误映射单元测试 ============

    #[test]
    fn map_api_error_401_is_authentication() {
        let err = VolcengineCvAdapter::map_api_error(401, "{\"error\":{\"message\":\"bad\"}}");
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[test]
    fn map_api_error_429_is_rate_limit() {
        let err = VolcengineCvAdapter::map_api_error(429, "{}");
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[test]
    fn map_api_error_404_has_console_hint() {
        let err = VolcengineCvAdapter::map_api_error(404, "{}");
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("Ark console")),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_500_extracts_message() {
        let err = VolcengineCvAdapter::map_api_error(500, "{\"error\":{\"message\":\"internal\"}}");
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 500);
                assert_eq!(message, "internal");
            }
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_top_level_message() {
        // 顶层 message 字段
        let err = VolcengineCvAdapter::map_api_error(400, "{\"message\":\"bad param\"}");
        match err {
            AibridgeError::Api { message, .. } => assert_eq!(message, "bad param"),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_api_error_no_json_falls_back_to_http_status() {
        let err = VolcengineCvAdapter::map_api_error(502, "Bad Gateway");
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("502")),
            _ => panic!("应为 Api"),
        }
    }

    // ============ 状态映射单元测试 ============

    #[test]
    fn map_video_status_variants() {
        assert_eq!(map_video_status("queued"), TaskStatus::Pending);
        assert_eq!(map_video_status("PENDING"), TaskStatus::Pending);
        assert_eq!(map_video_status("submitted"), TaskStatus::Pending);
        assert_eq!(map_video_status("processing"), TaskStatus::Processing);
        assert_eq!(map_video_status("running"), TaskStatus::Processing);
        assert_eq!(map_video_status("in_progress"), TaskStatus::Processing);
        assert_eq!(map_video_status("succeeded"), TaskStatus::Success);
        assert_eq!(map_video_status("success"), TaskStatus::Success);
        assert_eq!(map_video_status("completed"), TaskStatus::Success);
        assert_eq!(map_video_status("failed"), TaskStatus::Failed);
        assert_eq!(map_video_status("error"), TaskStatus::Failed);
        assert_eq!(map_video_status("cancelled"), TaskStatus::Failed);
        // 未知状态默认 Pending
        assert_eq!(map_video_status("unknown"), TaskStatus::Pending);
        assert_eq!(map_video_status(""), TaskStatus::Pending);
    }

    // ============ file_input_url 辅助 ============

    #[test]
    fn file_input_url_extracts_url() {
        assert_eq!(
            file_input_url(&FileInput::url("https://x")),
            Some("https://x".to_string())
        );
    }

    #[test]
    fn file_input_url_extracts_base64() {
        assert_eq!(
            file_input_url(&FileInput::base64("aGk=")),
            Some("aGk=".to_string())
        );
    }

    #[test]
    fn file_input_url_rejects_path_and_bytes() {
        assert_eq!(file_input_url(&FileInput::path("/tmp/x")), None);
        assert_eq!(file_input_url(&FileInput::bytes(vec![1, 2])), None);
    }

    // ============ full_caps 占位（保持与 openai_compat 测试结构一致）============

    #[test]
    fn full_caps_contains_image_and_video() {
        let caps = full_caps();
        assert!(caps.contains(&Capabilities::ImageGenerate));
        assert!(caps.contains(&Capabilities::VideoGenerate));
    }
}
