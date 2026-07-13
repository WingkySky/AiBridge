//! 新兴模型适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/emerging_models.py`。
//!
//! 支持三个 Provider（按协议分两类）：
//! - **Ideogram**：独立协议（`Api-Key` header + `/generate` 端点 + `image_request` 包裹），
//!   文字渲染最强的图像生成平台。仅 image 能力。
//! - **Luma Dream Machine**：独立协议（Bearer + `/generations` 端点），高质量视频生成。
//!   仅 video 能力（video_create + video_poll）。
//! - **Meta Llama**：OpenAI 兼容协议（`POST /chat/completions` + `GET /models`），
//!   Meta 官方 Llama API。chat + chat_stream + vision 能力，全部委托 [`OpenAiCompatAdapter`] 地基。
//!
//! ## 结构
//!
//! Python v1 是 3 个独立 adapter 类，统一注册到工厂。Rust 同样实现 3 个独立 struct：
//! - [`IdeogramAdapter`]：独立协议，自带 HttpClient，实现 image_generate + list_models（硬编码）
//! - [`LumaAdapter`]：独立协议，自带 HttpClient，实现 video_create + video_poll + list_models（硬编码）
//! - [`LlamaAdapter`]：组合 [`OpenAiCompatAdapter`] 地基，chat/chat_stream/list_models 全部委托
//!
//! ## provider_type 标识（对齐 Python v1）
//!
//! | Provider | provider_type | 别名 |
//! |---|---|---|
//! | Ideogram | `ideogram` | `ideo` |
//! | Luma Dream Machine | `luma` | `dream-machine` / `lumalabs` |
//! | Meta Llama | `llama` | `meta-llama` / `meta` |
//!
//! ## 阶段范围
//!
//! 阶段 2a 实现各 Provider 的核心能力（Ideogram 图像、Luma 视频、Llama 对话）+ list_models。
//! 不支持的能力（如 Ideogram 的 chat/video、Luma 的 chat/image、Llama 的 image/video）
//! 走 `Adapter` trait 默认实现返 `UnsupportedCapability`，与 Python v1 抛
//! `UnsupportedCapabilityError` 行为一致。

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::adapter::{Adapter, Capabilities, CapabilitySet, ChatStream};
use crate::adapters::openai_compat::OpenAiCompatAdapter;
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::chat::{ChatCompletion, ChatRequest};
use crate::model::common::{ModelInfo, ModelType, TaskStatus};
use crate::model::image::{FileInput, ImageData, ImageRequest, ImageResult};
use crate::model::video::{VideoRequest, VideoStatus, VideoTask};
use crate::util;

// ==================== 默认 Base URL ====================

/// Ideogram 默认 Base URL
///
/// 对应 Python v1 `IdeogramAdapter.DEFAULT_BASE_URL`。
pub const DEFAULT_IDEOGRAM_BASE_URL: &str = "https://api.ideogram.ai";

/// Luma Dream Machine 默认 Base URL
///
/// 对应 Python v1 `LumaAdapter.DEFAULT_BASE_URL`（已含 `/dream-machine/v1` 前缀）。
pub const DEFAULT_LUMA_BASE_URL: &str = "https://api.lumalabs.ai/dream-machine/v1";

/// Meta Llama 默认 Base URL
///
/// 对应 Python v1 `LlamaAdapter.DEFAULT_BASE_URL`（已含 `/v1` 前缀）。
pub const DEFAULT_LLAMA_BASE_URL: &str = "https://api.llama.com/v1";

// ==================== 能力集合构造 ====================

/// Ideogram 支持的能力集合
///
/// 对齐 Python v1 `IdeogramAdapter.supported_capabilities = ["image"]`。
/// Rust 用 `ImageGenerate` 表达图像生成能力。
fn ideogram_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::ImageGenerate);
    caps
}

/// Luma 支持的能力集合
///
/// 对齐 Python v1 `LumaAdapter.supported_capabilities = ["video"]`。
/// Rust 用 `VideoGenerate` 表达视频生成能力（含 video_create + video_poll）。
fn luma_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::VideoGenerate);
    caps.insert(Capabilities::VideoText2Video);
    caps.insert(Capabilities::VideoImage2Video);
    caps
}

/// Meta Llama 支持的能力集合
///
/// 对齐 Python v1 `LlamaAdapter.supported_capabilities = ["chat", "vision"]`。
/// chat_stream 虽未在 Python 显式声明，但 Python 实现了该方法且 OpenAI 兼容协议天然支持流式，
/// 故 Rust 一并声明 ChatStream（与 openai/azure 等兼容适配器保持一致）。
fn llama_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

// ==================== Ideogram 图像生成适配器 ====================

/// Ideogram 适配器
///
/// 文字渲染最强的图像生成平台，支持 V2/V2A/V1 等模型。
/// 官方 API 文档：https://developers.ideogram.com/
///
/// ## API 规范
/// - Base URL: `https://api.ideogram.ai`
/// - 文生图: `POST /generate`（body 用 `image_request` 包裹）
/// - 图生图/Remix: `POST /remix`（body 用 `image_request` 包裹）
/// - 局部重绘: `POST /inpaint`
/// - 扩图: `POST /outpaint`
/// - 认证: `Api-Key` header（注意不是 Bearer Token）
///
/// ## 阶段范围
/// 阶段 2a 仅实现文生图（`/generate`）+ list_models（硬编码）。
/// Remix/Inpaint/Outpaint 等图像编辑能力走 trait 默认实现返 UnsupportedCapability，
/// 待后续阶段补齐 `image_edit` trait 方法后再实现。
pub struct IdeogramAdapter {
    /// HTTP 客户端（封装 reqwest，含连接池与超时）
    http: HttpClient,
    /// Provider 配置（api_key / base_url / timeout 等）
    config: ProviderConfig,
    /// 实际 base_url（已合并 config.base_url 与默认值）
    base_url: String,
    /// 支持的能力集合
    capabilities: CapabilitySet,
}

impl IdeogramAdapter {
    /// 创建 Ideogram 适配器
    ///
    /// `config.base_url` 为空时用 [`DEFAULT_IDEOGRAM_BASE_URL`] 兜底。
    /// `config.api_key` 为空时不在此处报错（由上层按 `requires_api_key` 校验）。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_IDEOGRAM_BASE_URL.to_string());

        // 构造 HttpClient：把 base_url 透传，便于 post_authed_json 等方法自动拼接相对路径
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
            capabilities: ideogram_capabilities(),
        })
    }

    /// 用显式 HttpClient 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_http(http: HttpClient, config: ProviderConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_IDEOGRAM_BASE_URL.to_string());
        Self {
            http,
            config,
            base_url,
            capabilities: ideogram_capabilities(),
        }
    }

    /// API key（可能为空，免费 provider 场景）
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
                capability: format!("{} (provider: ideogram)", cap.as_str()),
            })
        }
    }

    /// 发送带 `Api-Key` 认证的 POST JSON 请求，并用 Ideogram 错误映射处理响应
    ///
    /// 注意：Ideogram 用 `Api-Key` header 而非 Bearer Token，与 OpenAI 兼容族不同。
    async fn post_authed_json(&self, path: &str, body: &Value) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .http
            .inner()
            .post(&url)
            .header("Api-Key", self.api_key())
            .header("Content-Type", "application/json")
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

    /// 构造 Ideogram `/generate` 请求体
    ///
    /// 移植自 Python v1 `image_generate`：
    /// - 内层 `image_request` 包含 prompt / model 及可选参数
    /// - aspect_ratio 归一化为大写（与 Python `aspect_ratio.upper()` 一致）
    /// - num_images 上限 8（与 Python `min(int(num_images), 8)` 一致）
    /// - magic_prompt_level 映射到 `magic_prompt_option` 字段（与 Python 一致）
    /// - extra 字段合并到内层 image_request（透传厂商特有参数）
    fn build_generate_body(&self, req: &ImageRequest) -> Value {
        // 默认模型 V_2A_TURBO（与 Python 一致）
        let model = if req.model.is_empty() {
            "V_2A_TURBO".to_string()
        } else {
            req.model.clone()
        };

        let mut image_request = json!({
            "prompt": req.prompt,
            "model": model,
        });

        // 负面提示词
        if let Some(np) = &req.negative_prompt {
            image_request["negative_prompt"] = json!(np);
        }
        // 宽高比（归一化为大写）
        if let Some(ar) = &req.aspect_ratio {
            image_request["aspect_ratio"] = json!(ar.to_uppercase());
        }
        // 分辨率（Ideogram 用 resolution 字段表示分辨率，如 "1536x1536"）
        if let Some(size) = &req.size {
            image_request["resolution"] = json!(size);
        }
        // 风格类型（style 字段映射到 style_type）
        if let Some(style) = &req.style {
            image_request["style_type"] = json!(style);
        }
        // 魔法提示词增强：Python 用 kwargs.magic_prompt_level 映射到 magic_prompt_option
        if let Some(mp) = req.extra.get("magic_prompt_level") {
            image_request["magic_prompt_option"] = mp.clone();
        }
        // 生成数量（上限 8）
        let n = req.n.min(8);
        image_request["num_images"] = json!(n);
        // 随机种子
        if let Some(seed) = req.seed {
            image_request["seed"] = json!(seed);
        }

        // extra 透传（合并到内层 image_request，跳过已处理的 magic_prompt_level）
        if let Some(obj) = image_request.as_object_mut() {
            for (k, v) in &req.extra {
                if k != "magic_prompt_level" {
                    obj.insert(k.clone(), v.clone());
                }
            }
        }

        // Ideogram 用 image_request 包裹
        json!({ "image_request": image_request })
    }

    /// 解析 Ideogram `/generate` 响应 → ImageResult
    ///
    /// Ideogram 响应结构：
    /// ```json
    /// {
    ///   "request_id": "...",
    ///   "data": [{"url": "...", "base64": "...", "prompt": "..."}]
    /// }
    /// ```
    /// 注意：Ideogram 用 `base64` 而非 `b64_json`，`prompt` 而非 `revised_prompt`。
    fn parse_image_result(value: &Value, fallback_model: &str) -> Result<ImageResult> {
        let id = value
            .get("request_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| util::generate_id("img"));
        let created = value
            .get("created")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(util::current_timestamp);
        let model = fallback_model.to_string();
        let data = value
            .get("data")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(parse_ideogram_image_data).collect())
            .unwrap_or_default();
        Ok(ImageResult {
            id,
            object: "image.generation".to_string(),
            created,
            model,
            data,
        })
    }

    /// 将 Ideogram API 错误响应映射为 AibridgeError
    ///
    /// 移植自 Python v1 `_handle_ideogram_error`：
    /// - 401 → Authentication（"Invalid Ideogram API key"）
    /// - 402 → Api（"Ideogram payment required or credits exhausted"，额度耗尽）
    /// - 429 → RateLimit（"Ideogram rate limit exceeded or credits exhausted"）
    /// - 其余 ≥400 → Api（提取 message / error / detail，回退 `HTTP {status}`）
    pub fn map_api_error(status: u16, body: &str) -> AibridgeError {
        match status {
            401 => AibridgeError::Authentication {
                message: "Invalid Ideogram API key".to_string(),
            },
            402 => AibridgeError::Api {
                status,
                message: "Ideogram payment required or credits exhausted".to_string(),
            },
            429 => AibridgeError::RateLimit {
                message: "Ideogram rate limit exceeded or credits exhausted".to_string(),
                retry_after: None,
            },
            _ => {
                let message = parse_error_message(body, status);
                AibridgeError::Api { status, message }
            }
        }
    }
}

/// 解析单个 Ideogram 图像数据项
///
/// Ideogram 字段映射（与 OpenAI 不同）：
/// - `url` → `url`
/// - `base64` → `b64_json`
/// - `prompt` → `revised_prompt`（模型优化后的提示词）
fn parse_ideogram_image_data(v: &Value) -> ImageData {
    ImageData {
        url: v.get("url").and_then(|x| x.as_str()).map(str::to_owned),
        b64_json: v.get("base64").and_then(|x| x.as_str()).map(str::to_owned),
        revised_prompt: v.get("prompt").and_then(|x| x.as_str()).map(str::to_owned),
    }
}

#[async_trait]
impl Adapter for IdeogramAdapter {
    fn provider_type(&self) -> &str {
        "ideogram"
    }

    fn provider_name(&self) -> &str {
        "Ideogram"
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

    /// 图像生成：`POST /generate`（body 用 `image_request` 包裹）
    async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
        self.ensure_capability(Capabilities::ImageGenerate)?;
        let body = self.build_generate_body(&req);
        let model = if req.model.is_empty() {
            "V_2A_TURBO".to_string()
        } else {
            req.model.clone()
        };
        let value = self.post_authed_json("generate", &body).await?;
        Self::parse_image_result(&value, &model)
    }

    /// 模型列表（硬编码）
    ///
    /// Ideogram 无标准 `/models` 端点，暂保留硬编码列表（与 Python v1 一致）。
    /// 含 V_2A / V_2A_TURBO / V_2 / V_1 / V_1_TURBO 五个模型。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = ideogram_hardcoded_models();
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }

    // chat / chat_stream / video / embed / audio 走 trait 默认实现返 UnsupportedCapability，
    // 与 Python v1 抛 UnsupportedCapabilityError 行为一致。
}

// ==================== Luma Dream Machine 视频生成适配器 ====================

/// Luma Dream Machine 适配器
///
/// 高质量视频生成平台，支持文生视频和图生视频。
/// 官方 API 文档：https://docs.lumalabs.ai/
///
/// ## API 规范
/// - Base URL: `https://api.lumalabs.ai/dream-machine/v1`
/// - 创建生成: `POST /generations`
/// - 查询状态: `GET /generations/{id}`
/// - 认证: `Authorization: Bearer {api_key}`
///
/// ## 阶段范围
/// 阶段 2a 实现 video_create + video_poll + list_models（硬编码）。
pub struct LumaAdapter {
    /// HTTP 客户端（封装 reqwest，含连接池与超时）
    http: HttpClient,
    /// Provider 配置（api_key / base_url / timeout 等）
    config: ProviderConfig,
    /// 实际 base_url（已合并 config.base_url 与默认值）
    base_url: String,
    /// 支持的能力集合
    capabilities: CapabilitySet,
}

impl LumaAdapter {
    /// 创建 Luma 适配器
    ///
    /// `config.base_url` 为空时用 [`DEFAULT_LUMA_BASE_URL`] 兜底。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_LUMA_BASE_URL.to_string());

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
            capabilities: luma_capabilities(),
        })
    }

    /// 用显式 HttpClient 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_http(http: HttpClient, config: ProviderConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_LUMA_BASE_URL.to_string());
        Self {
            http,
            config,
            base_url,
            capabilities: luma_capabilities(),
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

    /// 校验请求的能力是否被支持
    fn ensure_capability(&self, cap: Capabilities) -> Result<()> {
        if self.capabilities.contains(&cap) {
            Ok(())
        } else {
            Err(AibridgeError::UnsupportedCapability {
                capability: format!("{} (provider: luma)", cap.as_str()),
            })
        }
    }

    /// 发送带 Bearer 认证的 POST JSON 请求，并用 Luma 错误映射处理响应
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

    /// 发送带 Bearer 认证的 GET 请求，并用 Luma 错误映射处理响应
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

    /// 构造 Luma `/generations` 请求体
    ///
    /// 移植自 Python v1 `video_create`：
    /// - prompt / model 必选
    /// - aspect_ratio / duration / resolution / loop / negative_prompt / camera_motion 透传
    /// - reference_images[0] → keyframes.frame0（图生视频起始帧）
    /// - first_frame → keyframes.frame0（优先于 reference_images）
    /// - last_frame → keyframes.frame1（结束帧）
    /// - extra 字段合并到顶层（透传厂商特有参数）
    fn build_generations_body(&self, req: &VideoRequest) -> Value {
        // 默认模型 ray-2（与 Python 一致）
        let model = if req.model.is_empty() {
            "ray-2".to_string()
        } else {
            req.model.clone()
        };

        let mut body = json!({
            "prompt": req.prompt,
            "model": model,
        });

        // 宽高比
        if let Some(ar) = &req.aspect_ratio {
            body["aspect_ratio"] = json!(ar);
        }
        // 时长（VideoRequest.duration 是 u32 秒数，Luma 接受 "5s"/"9s" 字符串）
        if let Some(d) = req.duration {
            body["duration"] = json!(format!("{d}s"));
        }
        // 分辨率
        if let Some(r) = &req.resolution {
            body["resolution"] = json!(r);
        }
        // 循环（with_audio 字段不语义对应 loop，用 extra 里的 loop 优先）
        if let Some(lp) = req.extra.get("loop") {
            body["loop"] = lp.clone();
        }
        // 负面提示词
        if let Some(np) = &req.negative_prompt {
            body["negative_prompt"] = json!(np);
        }
        // 相机运动
        if let Some(cm) = &req.camera_motion {
            body["camera_motion"] = json!(cm);
        }

        // 关键帧：first_frame / last_frame / reference_images[0] → keyframes
        let mut keyframes = json!({});
        // first_frame 优先作为 frame0
        if let Some(ff) = &req.first_frame {
            keyframes["frame0"] = json!({ "type": "image", "url": file_input_to_url(ff) });
        } else if !req.reference_images.is_empty() {
            // reference_images[0] 作为 frame0（图生视频起始帧）
            keyframes["frame0"] =
                json!({ "type": "image", "url": file_input_to_url(&req.reference_images[0]) });
        }
        // last_frame 作为 frame1（结束帧）
        if let Some(lf) = &req.last_frame {
            keyframes["frame1"] = json!({ "type": "image", "url": file_input_to_url(lf) });
        }
        // extra 里的 keyframes 直接覆盖（更细粒度控制）
        if let Some(custom_kf) = req.extra.get("keyframes") {
            if let Some(custom_obj) = custom_kf.as_object() {
                if let Some(obj) = keyframes.as_object_mut() {
                    for (k, v) in custom_obj {
                        obj.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        if !keyframes.as_object().map(|o| o.is_empty()).unwrap_or(true) {
            body["keyframes"] = keyframes;
        }

        // extra 透传（合并到顶层，跳过已处理的 loop / keyframes）
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                if k != "loop" && k != "keyframes" {
                    obj.insert(k.clone(), v.clone());
                }
            }
        }

        body
    }

    /// 解析 Luma `/generations` 创建响应 → VideoTask
    ///
    /// Luma 创建响应：`{"id": "...", "state": "queued", ...}`
    fn parse_video_task(value: &Value, model: &str) -> Result<VideoTask> {
        let task_id = value
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| util::generate_id("vid"));
        let state = value
            .get("state")
            .and_then(|v| v.as_str())
            .unwrap_or("queued");
        Ok(VideoTask {
            task_id,
            model: model.to_string(),
            status: map_luma_status(state),
            created_at: util::current_timestamp(),
        })
    }

    /// 解析 Luma `/generations/{id}` 查询响应 → VideoStatus
    ///
    /// 移植自 Python v1 `video_poll`：
    /// - 视频 URL 在 `assets.video` / `assets.mp4` / `video`
    /// - 错误信息在 `failure_reason` / `error`
    /// - 进度估算：success=100, dreaming=30, processing=70, queued/pending=5
    /// - created_at / updated_at 为 ISO 时间字符串，转时间戳
    fn parse_video_status(value: &Value, task_id: &str) -> VideoStatus {
        let state = value.get("state").and_then(|v| v.as_str()).unwrap_or("");
        let status = map_luma_status(state);

        // 视频 URL：assets.video 优先，兼容 assets.mp4 / video
        let video_url = if status == TaskStatus::Success {
            value
                .get("assets")
                .and_then(|a| a.get("video"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .or_else(|| {
                    value
                        .get("assets")
                        .and_then(|a| a.get("mp4"))
                        .and_then(|v| v.as_str())
                        .map(str::to_owned)
                })
                .or_else(|| {
                    value
                        .get("video")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned)
                })
        } else {
            None
        };

        // 错误信息
        let error = if status == TaskStatus::Failed {
            value
                .get("failure_reason")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .or_else(|| {
                    value
                        .get("error")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned)
                })
                .or_else(|| Some("Generation failed".to_string()))
        } else {
            None
        };

        // 进度估算（先判断 state，因为 dreaming/processing 都映射到 processing 但进度不同）
        let state_lower = state.to_lowercase();
        let progress = if status == TaskStatus::Success {
            100
        } else if state_lower == "dreaming" {
            30
        } else if status == TaskStatus::Processing {
            70
        } else if state_lower == "queued" || state_lower == "pending" {
            5
        } else {
            0
        };

        // Luma 返回 ISO 时间字符串，转为时间戳
        let created_at = value
            .get("created_at")
            .and_then(|v| v.as_str())
            .and_then(parse_iso_to_timestamp);
        let updated_at = value
            .get("updated_at")
            .and_then(|v| v.as_str())
            .and_then(parse_iso_to_timestamp)
            .or_else(|| Some(util::current_timestamp()));

        VideoStatus {
            task_id: task_id.to_string(),
            status,
            video_url,
            progress: Some(progress),
            error,
            created_at,
            updated_at,
        }
    }

    /// 将 Luma API 错误响应映射为 AibridgeError
    ///
    /// 移植自 Python v1 `_handle_luma_error`：
    /// - 401 → Authentication（"Invalid Luma API key"）
    /// - 402 → Api（"Luma credits exhausted or payment required"）
    /// - 429 → RateLimit（"Luma rate limit exceeded or credits exhausted"）
    /// - 其余 ≥400 → Api（提取 detail / error.message / message，回退 `HTTP {status}`）
    pub fn map_api_error(status: u16, body: &str) -> AibridgeError {
        match status {
            401 => AibridgeError::Authentication {
                message: "Invalid Luma API key".to_string(),
            },
            402 => AibridgeError::Api {
                status,
                message: "Luma credits exhausted or payment required".to_string(),
            },
            429 => AibridgeError::RateLimit {
                message: "Luma rate limit exceeded or credits exhausted".to_string(),
                retry_after: None,
            },
            _ => {
                let message = parse_error_message(body, status);
                AibridgeError::Api { status, message }
            }
        }
    }
}

#[async_trait]
impl Adapter for LumaAdapter {
    fn provider_type(&self) -> &str {
        "luma"
    }

    fn provider_name(&self) -> &str {
        "Luma Dream Machine"
    }

    fn capabilities(&self) -> CapabilitySet {
        self.capabilities.clone()
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

    /// 创建视频生成任务：`POST /generations`
    async fn video_create(&self, req: VideoRequest) -> Result<VideoTask> {
        self.ensure_capability(Capabilities::VideoGenerate)?;
        let body = self.build_generations_body(&req);
        let model = if req.model.is_empty() {
            "ray-2".to_string()
        } else {
            req.model.clone()
        };
        let value = self.post_authed_json("generations", &body).await?;
        Self::parse_video_task(&value, &model)
    }

    /// 查询视频任务状态：`GET /generations/{id}`
    async fn video_poll(&self, task_id: &str, _model: &str) -> Result<VideoStatus> {
        self.ensure_capability(Capabilities::VideoGenerate)?;
        let path = format!("generations/{task_id}");
        let value = self.get_authed_json(&path).await?;
        Ok(Self::parse_video_status(&value, task_id))
    }

    /// 模型列表（硬编码）
    ///
    /// Luma 无标准 `/models` 端点，暂保留硬编码列表（与 Python v1 一致）。
    /// 含 ray-2 / ray-2-flash / dream-machine 三个模型。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = luma_hardcoded_models();
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }

    // chat / chat_stream / image / embed / audio 走 trait 默认实现返 UnsupportedCapability，
    // 与 Python v1 抛 UnsupportedCapabilityError 行为一致。
}

// ==================== Meta Llama 适配器（OpenAI 兼容）====================

/// Meta Llama 适配器
///
/// Meta 官方 Llama API，完全兼容 OpenAI 接口规范。
/// 官方 API 文档：https://docs.llama.com/
///
/// ## API 规范
/// - Base URL: `https://api.llama.com/v1`
/// - Chat: `POST /chat/completions`（OpenAI 兼容）
/// - Models: `GET /models`（OpenAI 兼容）
/// - 认证: Bearer Token
/// - 支持流式输出
///
/// 全部能力（chat / chat_stream / list_models）委托给 [`OpenAiCompatAdapter`] 地基，
/// 仅 base_url / provider_type / capabilities 差异。
pub struct LlamaAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl LlamaAdapter {
    /// 创建 Meta Llama 适配器
    ///
    /// `config.base_url` 为空时回退到 [`DEFAULT_LLAMA_BASE_URL`]。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            "llama",
            "Meta Llama",
            DEFAULT_LLAMA_BASE_URL,
            llama_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 compat 地基构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }
}

#[async_trait]
impl Adapter for LlamaAdapter {
    fn provider_type(&self) -> &str {
        "llama"
    }

    fn provider_name(&self) -> &str {
        "Meta Llama"
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

    /// 模型列表（实时拉取）
    ///
    /// 调用 `GET /models`（OpenAI 兼容端点），实时拉取模型列表。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }

    // image_generate / video / embed / audio 走 trait 默认实现返 UnsupportedCapability，
    // 与 Python v1 抛 UnsupportedCapabilityError 行为一致。
}

// ==================== 内部：硬编码模型列表 ====================

/// Ideogram 硬编码模型列表
///
/// 对应 Python v1 `IdeogramAdapter.list_models`。
/// 注意：该 Provider 无标准 `/models` 端点，暂保留硬编码列表。
fn ideogram_hardcoded_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "V_2A".into(),
            name: "Ideogram V2A".into(),
            model_type: ModelType::Image,
            provider: "ideogram".into(),
            capabilities: vec!["text2image".into(), "image2image".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Ideogram V2A 文生图模型，文字渲染强".into()),
            created: None,
        },
        ModelInfo {
            id: "V_2A_TURBO".into(),
            name: "Ideogram V2A Turbo".into(),
            model_type: ModelType::Image,
            provider: "ideogram".into(),
            capabilities: vec!["text2image".into(), "image2image".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Ideogram V2A Turbo 快速版本，文字渲染强".into()),
            created: None,
        },
        ModelInfo {
            id: "V_2".into(),
            name: "Ideogram V2".into(),
            model_type: ModelType::Image,
            provider: "ideogram".into(),
            capabilities: vec!["text2image".into(), "image2image".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Ideogram V2 高质量模型".into()),
            created: None,
        },
        ModelInfo {
            id: "V_1".into(),
            name: "Ideogram V1".into(),
            model_type: ModelType::Image,
            provider: "ideogram".into(),
            capabilities: vec!["text2image".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Ideogram V1 标准模型".into()),
            created: None,
        },
        ModelInfo {
            id: "V_1_TURBO".into(),
            name: "Ideogram V1 Turbo".into(),
            model_type: ModelType::Image,
            provider: "ideogram".into(),
            capabilities: vec!["text2image".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Ideogram V1 Turbo 快速模型".into()),
            created: None,
        },
    ]
}

/// Luma 硬编码模型列表
///
/// 对应 Python v1 `LumaAdapter.list_models`。
/// 注意：该 Provider 无标准 `/models` 端点，暂保留硬编码列表。
fn luma_hardcoded_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "ray-2".into(),
            name: "Luma Ray 2".into(),
            model_type: ModelType::Video,
            provider: "luma".into(),
            capabilities: vec!["text2video".into(), "image2video".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Luma Ray 2 高质量视频生成模型".into()),
            created: None,
        },
        ModelInfo {
            id: "ray-2-flash".into(),
            name: "Luma Ray 2 Flash".into(),
            model_type: ModelType::Video,
            provider: "luma".into(),
            capabilities: vec!["text2video".into(), "image2video".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Luma Ray 2 Flash 快速视频生成模型".into()),
            created: None,
        },
        ModelInfo {
            id: "dream-machine".into(),
            name: "Dream Machine".into(),
            model_type: ModelType::Video,
            provider: "luma".into(),
            capabilities: vec!["text2video".into(), "image2video".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some("Luma Dream Machine 初代视频模型".into()),
            created: None,
        },
    ]
}

// ==================== 内部：辅助函数 ====================

/// 映射 Luma 状态到标准 TaskStatus
///
/// 移植自 Python v1 `_map_luma_status`：
/// - queued / pending → Pending
/// - dreaming / processing → Processing
/// - completed / succeeded / success → Success
/// - failed / error → Failed
/// - 未知 → Pending（与 Python 默认值一致）
fn map_luma_status(raw_state: &str) -> TaskStatus {
    match raw_state.to_lowercase().as_str() {
        "queued" | "pending" => TaskStatus::Pending,
        "dreaming" | "processing" => TaskStatus::Processing,
        "completed" | "succeeded" | "success" => TaskStatus::Success,
        "failed" | "error" => TaskStatus::Failed,
        _ => TaskStatus::Pending,
    }
}

/// 将 FileInput 转为 URL 字符串
///
/// Luma keyframes 需要 URL 形式的图像引用：
/// - `Url(s)` → 直接返回 s
/// - `Base64(s)` → 返回 s（Luma 接受 base64 字符串作为 url）
/// - `Path(_)` / `Bytes(_)` → 空字符串（这些形式需上层预转换为 URL/base64，此处不处理）
///
/// 与 Python v1 行为一致：Python 仅处理 data: 前缀和 http URL，其余原样透传。
fn file_input_to_url(input: &FileInput) -> String {
    match input {
        FileInput::Url(s) | FileInput::Base64(s) => s.clone(),
        FileInput::Path(_) | FileInput::Bytes(_) => String::new(),
    }
}

/// 解析 ISO 8601 时间字符串为 Unix 时间戳
///
/// Luma 返回 ISO 时间字符串（如 `2024-01-01T00:00:00Z`），转为时间戳。
/// 解析失败返回 None（与 Python try/except 行为一致）。
fn parse_iso_to_timestamp(s: &str) -> Option<u64> {
    // 处理 Z 后缀（替换为 +00:00 便于统一解析）
    let normalized = if let Some(stripped) = s.strip_suffix('Z') {
        format!("{stripped}+00:00")
    } else {
        s.to_string()
    };
    parse_rfc3339(&normalized)
}

/// 手动解析 RFC3339 时间字符串为 Unix 时间戳
///
/// 支持格式：`YYYY-MM-DDTHH:MM:SS[.fff][+HH:MM | -HH:MM | Z]`
/// 解析失败返回 None。不引入 chrono 依赖，用 Howard Hinnant 的 civil_from_days 算法。
fn parse_rfc3339(s: &str) -> Option<u64> {
    // 格式：YYYY-MM-DDTHH:MM:SS
    if s.len() < 19 {
        return None;
    }
    let bytes = s.as_bytes();
    // 解析日期时间部分 YYYY-MM-DDTHH:MM:SS
    let year: u32 = std::str::from_utf8(&bytes[0..4]).ok()?.parse().ok()?;
    if bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return None;
    }
    let month: u32 = std::str::from_utf8(&bytes[5..7]).ok()?.parse().ok()?;
    let day: u32 = std::str::from_utf8(&bytes[8..10]).ok()?.parse().ok()?;
    let hour: u32 = std::str::from_utf8(&bytes[11..13]).ok()?.parse().ok()?;
    let minute: u32 = std::str::from_utf8(&bytes[14..16]).ok()?.parse().ok()?;
    let second: u32 = std::str::from_utf8(&bytes[17..19]).ok()?.parse().ok()?;

    // 解析时区偏移：[+|-]HH:MM 或 Z（已规范化为 +00:00）
    let tz_offset_seconds: i64 = if s.len() >= 25 {
        let sign = match bytes[19] {
            b'+' => 1i64,
            b'-' => -1i64,
            _ => return None,
        };
        let tz_hour: i64 = std::str::from_utf8(&bytes[20..22]).ok()?.parse().ok()?;
        let tz_minute: i64 = std::str::from_utf8(&bytes[23..25]).ok()?.parse().ok()?;
        sign * (tz_hour * 3600 + tz_minute * 60)
    } else {
        // 无时区信息，按 UTC 处理
        0
    };

    // 转为 Unix 时间戳（UTC）
    let utc_seconds = civil_to_unix(year, month, day, hour, minute, second)?;
    // 减去时区偏移得到 UTC 时间戳
    Some((utc_seconds as i64 - tz_offset_seconds) as u64)
}

/// 公历日期时间转 Unix 时间戳（UTC）
///
/// 算法：Howard Hinnant 的 civil_from_days，从 1970-01-01 起累加天数。
/// 返回 None 表示日期非法或超出支持范围。
fn civil_to_unix(
    year: u32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> Option<i64> {
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }
    let y = year as i64;
    let m = month as i64;
    let d = day as i64;
    // 调整：3月为年初（避免闰年判断的边界问题）
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = (y_adj - era * 400) as u64; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy as u64; // [0, 146096]
    let days = era * 146097 + doe as i64 - 719468;
    let seconds = days * 86400 + (hour as i64) * 3600 + (minute as i64) * 60 + second as i64;
    Some(seconds)
}

/// 解析错误体中的 message 字段
///
/// 通用错误消息提取，兼容多种错误体结构：
/// - `{"error": {"message": "..."}}`（OpenAI 风格）
/// - `{"message": "..."}`（部分平台顶层 message）
/// - `{"detail": "..."}`（Luma 用 detail 字段）
/// - `{"detail": ["err1", "err2"]}`（Luma validation 错误数组）
/// - `{"error": "..."}`（Ideogram 用顶层 error 字符串）
///
/// 解析失败时回退到 `HTTP {status}` 字符串。
fn parse_error_message(body: &str, status: u16) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        // Luma 优先用 detail 字段
        if let Some(msg) = v.get("detail").and_then(|m| m.as_str()) {
            return msg.to_string();
        }
        // detail 可能是数组（Luma 的 validation 错误）
        if let Some(arr) = v.get("detail").and_then(|m| m.as_array()) {
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|x| x.as_str().map(str::to_owned))
                .collect();
            if !parts.is_empty() {
                return parts.join("; ");
            }
        }
        // OpenAI 风格 error.message
        if let Some(msg) = v
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return msg.to_string();
        }
        // 顶层 error 字符串（Ideogram 风格）
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
    use crate::error::AibridgeError;
    use crate::http::HttpClient;
    use crate::model::chat::ChatMessage;
    use crate::model::image::FileInput;
    use futures::stream::StreamExt;
    use mockito::Server;
    use serde_json::json;

    // ==================== 通用测试辅助 ====================

    /// 构造测试用 IdeogramAdapter（指向 mockito server）
    fn make_ideogram(server: &Server) -> IdeogramAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("ideogram", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        IdeogramAdapter::with_http(http, config)
    }

    /// 构造测试用 LumaAdapter（指向 mockito server）
    fn make_luma(server: &Server) -> LumaAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("luma", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        LumaAdapter::with_http(http, config)
    }

    /// 构造测试用 LlamaAdapter（指向 mockito server）
    fn make_llama(server: &Server) -> LlamaAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("llama", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        let compat = OpenAiCompatAdapter::with_http(
            http,
            config,
            "llama",
            "Meta Llama",
            llama_capabilities(),
        );
        LlamaAdapter::with_compat(compat)
    }

    /// 构造不指向任何 server 的 IdeogramAdapter（用于不发请求的元信息/能力测试）
    fn make_ideogram_no_server() -> IdeogramAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(DEFAULT_IDEOGRAM_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("ideogram", opts);
        IdeogramAdapter::new(config).expect("IdeogramAdapter 构造应成功")
    }

    /// 构造不指向任何 server 的 LumaAdapter
    fn make_luma_no_server() -> LumaAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(DEFAULT_LUMA_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("luma", opts);
        LumaAdapter::new(config).expect("LumaAdapter 构造应成功")
    }

    /// 构造不指向任何 server 的 LlamaAdapter
    fn make_llama_no_server() -> LlamaAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(DEFAULT_LLAMA_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("llama", opts);
        LlamaAdapter::new(config).expect("LlamaAdapter 构造应成功")
    }

    // ============ Ideogram 元信息 ============

    #[test]
    fn ideogram_provider_type_and_name_match_python() {
        let adapter = make_ideogram_no_server();
        assert_eq!(adapter.provider_type(), "ideogram");
        assert_eq!(adapter.provider_name(), "Ideogram");
    }

    #[test]
    fn ideogram_requires_api_key_is_true() {
        let adapter = make_ideogram_no_server();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn ideogram_capabilities_contains_only_image() {
        let adapter = make_ideogram_no_server();
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::ImageGenerate));
        // chat / video 不声明
        assert!(!caps.contains(&Capabilities::Chat));
        assert!(!caps.contains(&Capabilities::VideoGenerate));
    }

    #[test]
    fn ideogram_base_url_defaults_when_missing() {
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("ideogram", opts);
        let adapter = IdeogramAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), DEFAULT_IDEOGRAM_BASE_URL);
    }

    #[test]
    fn ideogram_base_url_uses_config_when_provided() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://custom.ideogram-proxy.com")
            .build();
        let config = ProviderConfig::from_options("ideogram", opts);
        let adapter = IdeogramAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), "https://custom.ideogram-proxy.com");
    }

    // ============ Ideogram image_generate ============

    #[tokio::test]
    async fn ideogram_image_generate_success_parses_url() {
        let mut server = Server::new_async().await;
        let body = json!({
            "request_id": "req-123",
            "data": [{
                "url": "https://example.com/img.png",
                "prompt": "a cute cat"
            }]
        });
        let mock = server
            .mock("POST", "/generate")
            .match_header("Api-Key", "test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_ideogram(&server);
        let req = ImageRequest::builder("V_2A", "a cat")
            .aspect_ratio("16:9")
            .n(2)
            .seed(42)
            .build();
        let resp = adapter
            .image_generate(req)
            .await
            .expect("image_generate 应成功");

        assert_eq!(resp.id, "req-123");
        assert_eq!(resp.model, "V_2A");
        assert_eq!(resp.data.len(), 1);
        assert_eq!(
            resp.data[0].url.as_deref(),
            Some("https://example.com/img.png")
        );
        assert_eq!(resp.data[0].revised_prompt.as_deref(), Some("a cute cat"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn ideogram_image_generate_parses_base64() {
        let mut server = Server::new_async().await;
        let body = json!({
            "request_id": "req-456",
            "data": [{"base64": "aGVsbG8="}]
        });
        server
            .mock("POST", "/generate")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_ideogram(&server);
        let req = ImageRequest::builder("V_2A_TURBO", "a cat").build();
        let resp = adapter.image_generate(req).await.unwrap();
        assert_eq!(resp.data[0].b64_json.as_deref(), Some("aGVsbG8="));
    }

    #[tokio::test]
    async fn ideogram_image_generate_wraps_in_image_request() {
        // 验证请求体用 image_request 包裹，且 aspect_ratio 归一化为大写
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generate")
            .match_body(mockito::Matcher::PartialJson(json!({
                "image_request": {
                    "model": "V_2A",
                    "prompt": "a cat",
                    "aspect_ratio": "16:9",
                    "num_images": 3
                }
            })))
            .with_status(200)
            .with_body(json!({"request_id": "x", "data": []}).to_string())
            .create_async()
            .await;

        let adapter = make_ideogram(&server);
        let req = ImageRequest::builder("V_2A", "a cat")
            .aspect_ratio("16:9")
            .n(3)
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn ideogram_image_generate_uses_default_model_when_empty() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generate")
            .match_body(mockito::Matcher::PartialJson(json!({
                "image_request": {"model": "V_2A_TURBO"}
            })))
            .with_status(200)
            .with_body(json!({"request_id": "x", "data": []}).to_string())
            .create_async()
            .await;

        let adapter = make_ideogram(&server);
        let req = ImageRequest::builder("", "a cat").build();
        let resp = adapter.image_generate(req).await.unwrap();
        assert_eq!(resp.model, "V_2A_TURBO");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn ideogram_image_generate_caps_num_images_at_8() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generate")
            .match_body(mockito::Matcher::PartialJson(json!({
                "image_request": {"num_images": 8}
            })))
            .with_status(200)
            .with_body(json!({"data": []}).to_string())
            .create_async()
            .await;

        let adapter = make_ideogram(&server);
        // 请求 20 张，应被截断为 8
        let req = ImageRequest::builder("V_2A", "a cat").n(20).build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn ideogram_image_generate_passes_extra_params() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generate")
            .match_body(mockito::Matcher::PartialJson(json!({
                "image_request": {
                    "magic_prompt_option": "HIGH",
                    "style_type": "REALISTIC",
                    "custom_param": "custom_value"
                }
            })))
            .with_status(200)
            .with_body(json!({"data": []}).to_string())
            .create_async()
            .await;

        let adapter = make_ideogram(&server);
        let req = ImageRequest::builder("V_2A", "a cat")
            .style("REALISTIC")
            .extra("magic_prompt_level", "HIGH")
            .extra("custom_param", "custom_value")
            .build();
        let _ = adapter.image_generate(req).await.unwrap();
        mock.assert_async().await;
    }

    // ============ Ideogram image_generate 错误路径 ============

    #[tokio::test]
    async fn ideogram_image_generate_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generate")
            .with_status(401)
            .with_body(json!({"error": "Invalid API key"}).to_string())
            .create_async()
            .await;

        let adapter = make_ideogram(&server);
        let req = ImageRequest::builder("V_2A", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn ideogram_image_generate_error_402_returns_api_payment() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generate")
            .with_status(402)
            .with_body(json!({"message": "credits exhausted"}).to_string())
            .create_async()
            .await;

        let adapter = make_ideogram(&server);
        let req = ImageRequest::builder("V_2A", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 402);
                assert!(message.contains("credits exhausted"));
            }
            _ => panic!("应为 Api (402)"),
        }
    }

    #[tokio::test]
    async fn ideogram_image_generate_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generate")
            .with_status(429)
            .with_body(json!({"error": "slow down"}).to_string())
            .create_async()
            .await;

        let adapter = make_ideogram(&server);
        let req = ImageRequest::builder("V_2A", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn ideogram_image_generate_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generate")
            .with_status(500)
            .with_body(json!({"error": "internal"}).to_string())
            .create_async()
            .await;

        let adapter = make_ideogram(&server);
        let req = ImageRequest::builder("V_2A", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 500),
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn ideogram_chat_returns_unsupported() {
        let adapter = make_ideogram_no_server();
        let req = ChatRequest::builder("V_2A", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn ideogram_video_create_returns_unsupported() {
        let adapter = make_ideogram_no_server();
        let req = VideoRequest::builder("V_2A", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ Ideogram list_models ============

    #[tokio::test]
    async fn ideogram_list_models_returns_hardcoded() {
        let adapter = make_ideogram_no_server();
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 5);
        assert_eq!(models[0].id, "V_2A");
        assert_eq!(models[0].provider, "ideogram");
        assert_eq!(models[0].model_type, ModelType::Image);
    }

    #[tokio::test]
    async fn ideogram_list_models_filter_by_image_type() {
        let adapter = make_ideogram_no_server();
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert_eq!(images.len(), 5);
        assert!(images.iter().all(|m| m.model_type == ModelType::Image));
    }

    #[tokio::test]
    async fn ideogram_list_models_filter_by_video_returns_empty() {
        let adapter = make_ideogram_no_server();
        let videos = adapter.list_models(Some(ModelType::Video)).await.unwrap();
        assert!(videos.is_empty());
    }

    // ============ Luma 元信息 ============

    #[test]
    fn luma_provider_type_and_name_match_python() {
        let adapter = make_luma_no_server();
        assert_eq!(adapter.provider_type(), "luma");
        assert_eq!(adapter.provider_name(), "Luma Dream Machine");
    }

    #[test]
    fn luma_requires_api_key_is_true() {
        let adapter = make_luma_no_server();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn luma_capabilities_contains_only_video() {
        let adapter = make_luma_no_server();
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::VideoGenerate));
        assert!(caps.contains(&Capabilities::VideoText2Video));
        assert!(caps.contains(&Capabilities::VideoImage2Video));
        // chat / image 不声明
        assert!(!caps.contains(&Capabilities::Chat));
        assert!(!caps.contains(&Capabilities::ImageGenerate));
    }

    #[test]
    fn luma_base_url_defaults_when_missing() {
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("luma", opts);
        let adapter = LumaAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), DEFAULT_LUMA_BASE_URL);
    }

    #[test]
    fn luma_base_url_uses_config_when_provided() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://custom.luma-proxy.com/v1")
            .build();
        let config = ProviderConfig::from_options("luma", opts);
        let adapter = LumaAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), "https://custom.luma-proxy.com/v1");
    }

    // ============ Luma video_create ============

    #[tokio::test]
    async fn luma_video_create_success_returns_task() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "gen-abc",
            "state": "queued"
        });
        let mock = server
            .mock("POST", "/generations")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let req = VideoRequest::builder("ray-2", "a cat running")
            .aspect_ratio("16:9")
            .duration(5)
            .resolution("720p")
            .build();
        let task = adapter
            .video_create(req)
            .await
            .expect("video_create 应成功");

        assert_eq!(task.task_id, "gen-abc");
        assert_eq!(task.model, "ray-2");
        assert_eq!(task.status, TaskStatus::Pending);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn luma_video_create_sends_model_and_prompt() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "ray-2",
                "prompt": "a cat",
                "aspect_ratio": "16:9",
                "duration": "5s",
                "resolution": "720p"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "state": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let req = VideoRequest::builder("ray-2", "a cat")
            .aspect_ratio("16:9")
            .duration(5)
            .resolution("720p")
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn luma_video_create_uses_default_model_when_empty() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "ray-2"
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "state": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let req = VideoRequest::builder("", "a cat").build();
        let task = adapter.video_create(req).await.unwrap();
        assert_eq!(task.model, "ray-2");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn luma_video_create_with_first_frame_builds_keyframes() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "keyframes": {
                    "frame0": {"type": "image", "url": "https://example.com/start.png"},
                    "frame1": {"type": "image", "url": "https://example.com/end.png"}
                }
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "state": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let req = VideoRequest::builder("ray-2", "a cat")
            .first_frame(FileInput::url("https://example.com/start.png"))
            .last_frame(FileInput::url("https://example.com/end.png"))
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn luma_video_create_with_reference_images_builds_frame0() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/generations")
            .match_body(mockito::Matcher::PartialJson(json!({
                "keyframes": {
                    "frame0": {"type": "image", "url": "https://example.com/ref.png"}
                }
            })))
            .with_status(200)
            .with_body(json!({"id": "x", "state": "queued"}).to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let req = VideoRequest::builder("ray-2", "a cat")
            .reference_images(vec![FileInput::url("https://example.com/ref.png")])
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn luma_video_create_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(401)
            .with_body(json!({"detail": "Invalid API key"}).to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let req = VideoRequest::builder("ray-2", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn luma_video_create_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(429)
            .with_body(json!({"detail": "slow down"}).to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let req = VideoRequest::builder("ray-2", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn luma_video_create_error_402_returns_api_payment() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(402)
            .with_body(json!({"detail": "credits exhausted"}).to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let req = VideoRequest::builder("ray-2", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 402),
            _ => panic!("应为 Api (402)"),
        }
    }

    #[tokio::test]
    async fn luma_video_create_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/generations")
            .with_status(500)
            .with_body(json!({"detail": "internal"}).to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let req = VideoRequest::builder("ray-2", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 500),
            _ => panic!("应为 Api"),
        }
    }

    // ============ Luma video_poll ============

    #[tokio::test]
    async fn luma_video_poll_success_returns_video_url() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "gen-abc",
            "state": "completed",
            "assets": {"video": "https://example.com/video.mp4"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:05Z"
        });
        let mock = server
            .mock("GET", "/generations/gen-abc")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let status = adapter
            .video_poll("gen-abc", "ray-2")
            .await
            .expect("video_poll 应成功");

        assert_eq!(status.task_id, "gen-abc");
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://example.com/video.mp4")
        );
        assert_eq!(status.progress, Some(100));
        assert!(status.error.is_none());
        assert!(status.created_at.is_some());
        assert!(status.updated_at.is_some());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn luma_video_poll_processing_returns_progress() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "gen-abc",
            "state": "dreaming"
        });
        server
            .mock("GET", "/generations/gen-abc")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let status = adapter.video_poll("gen-abc", "ray-2").await.unwrap();
        assert_eq!(status.status, TaskStatus::Processing);
        assert_eq!(status.progress, Some(30));
        assert!(status.video_url.is_none());
    }

    #[tokio::test]
    async fn luma_video_poll_failed_returns_error() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "gen-abc",
            "state": "failed",
            "failure_reason": "content policy violation"
        });
        server
            .mock("GET", "/generations/gen-abc")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let status = adapter.video_poll("gen-abc", "ray-2").await.unwrap();
        assert_eq!(status.status, TaskStatus::Failed);
        assert_eq!(status.error.as_deref(), Some("content policy violation"));
    }

    #[tokio::test]
    async fn luma_video_poll_processing_state_returns_70_progress() {
        let mut server = Server::new_async().await;
        let body = json!({"id": "x", "state": "processing"});
        server
            .mock("GET", "/generations/x")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let status = adapter.video_poll("x", "ray-2").await.unwrap();
        assert_eq!(status.status, TaskStatus::Processing);
        assert_eq!(status.progress, Some(70));
    }

    #[tokio::test]
    async fn luma_video_poll_queued_returns_5_progress() {
        let mut server = Server::new_async().await;
        let body = json!({"id": "x", "state": "queued"});
        server
            .mock("GET", "/generations/x")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let status = adapter.video_poll("x", "ray-2").await.unwrap();
        assert_eq!(status.status, TaskStatus::Pending);
        assert_eq!(status.progress, Some(5));
    }

    #[tokio::test]
    async fn luma_video_poll_assets_mp4_fallback() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "x",
            "state": "completed",
            "assets": {"mp4": "https://example.com/video.mp4"}
        });
        server
            .mock("GET", "/generations/x")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let status = adapter.video_poll("x", "ray-2").await.unwrap();
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://example.com/video.mp4")
        );
    }

    #[tokio::test]
    async fn luma_video_poll_error_404_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/generations/nonexistent")
            .with_status(404)
            .with_body(json!({"detail": "not found"}).to_string())
            .create_async()
            .await;

        let adapter = make_luma(&server);
        let err = adapter
            .video_poll("nonexistent", "ray-2")
            .await
            .unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 404),
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn luma_chat_returns_unsupported() {
        let adapter = make_luma_no_server();
        let req = ChatRequest::builder("ray-2", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn luma_image_generate_returns_unsupported() {
        let adapter = make_luma_no_server();
        let req = ImageRequest::builder("ray-2", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ Luma list_models ============

    #[tokio::test]
    async fn luma_list_models_returns_hardcoded() {
        let adapter = make_luma_no_server();
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].id, "ray-2");
        assert_eq!(models[0].provider, "luma");
        assert_eq!(models[0].model_type, ModelType::Video);
    }

    #[tokio::test]
    async fn luma_list_models_filter_by_video_type() {
        let adapter = make_luma_no_server();
        let videos = adapter.list_models(Some(ModelType::Video)).await.unwrap();
        assert_eq!(videos.len(), 3);
        assert!(videos.iter().all(|m| m.model_type == ModelType::Video));
    }

    #[tokio::test]
    async fn luma_list_models_filter_by_image_returns_empty() {
        let adapter = make_luma_no_server();
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert!(images.is_empty());
    }

    // ============ Llama 元信息 ============

    #[test]
    fn llama_provider_type_and_name_match_python() {
        let adapter = make_llama_no_server();
        assert_eq!(adapter.provider_type(), "llama");
        assert_eq!(adapter.provider_name(), "Meta Llama");
    }

    #[test]
    fn llama_requires_api_key_is_true() {
        let adapter = make_llama_no_server();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn llama_capabilities_contains_chat_and_vision() {
        let adapter = make_llama_no_server();
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::Vision));
        // image / video 不声明
        assert!(!caps.contains(&Capabilities::ImageGenerate));
        assert!(!caps.contains(&Capabilities::VideoGenerate));
    }

    #[test]
    fn llama_base_url_defaults_when_missing() {
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("llama", opts);
        let adapter = LlamaAdapter::new(config).unwrap();
        assert_eq!(adapter.compat.base_url(), DEFAULT_LLAMA_BASE_URL);
    }

    #[test]
    fn llama_base_url_uses_config_when_provided() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://custom.llama-proxy.com/v1")
            .build();
        let config = ProviderConfig::from_options("llama", opts);
        let adapter = LlamaAdapter::new(config).unwrap();
        assert_eq!(
            adapter.compat.base_url(),
            "https://custom.llama-proxy.com/v1"
        );
    }

    // ============ Llama chat ============

    #[tokio::test]
    async fn llama_chat_success_returns_completion() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "llama-4-maverick",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_llama(&server);
        let req = ChatRequest::builder("llama-4-maverick", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .max_tokens(100)
            .build();
        let resp = adapter.chat(req).await.expect("chat 应成功");

        assert_eq!(resp.id, "chatcmpl-1");
        assert_eq!(resp.model, "llama-4-maverick");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 7);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn llama_chat_sends_temperature_and_max_tokens() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "llama-4-maverick",
                "temperature": 0.5,
                "max_tokens": 50
            })))
            .with_status(200)
            .with_body(
                json!({
                    "id": "x", "object": "chat.completion", "created": 1, "model": "llama-4-maverick",
                    "choices": [{"index": 0, "message": {"role":"assistant","content":"ok"}, "finish_reason": "stop"}]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_llama(&server);
        let req = ChatRequest::builder("llama-4-maverick", vec![ChatMessage::user("hi")])
            .temperature(0.5)
            .max_tokens(50)
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn llama_chat_passes_extra_params_through() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "llama-4-maverick",
                "custom_param": "custom_value"
            })))
            .with_status(200)
            .with_body(
                json!({
                    "id": "x", "object": "chat.completion", "created": 1, "model": "llama-4-maverick",
                    "choices": [{"index": 0, "message": {"role":"assistant","content":"ok"}, "finish_reason": "stop"}]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_llama(&server);
        let req = ChatRequest::builder("llama-4-maverick", vec![ChatMessage::user("hi")])
            .extra("custom_param", "custom_value")
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn llama_chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(json!({"error": {"message": "Invalid Llama API key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_llama(&server);
        let req = ChatRequest::builder("llama-4-maverick", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn llama_chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;

        let adapter = make_llama(&server);
        let req = ChatRequest::builder("llama-4-maverick", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn llama_chat_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(500)
            .with_body(json!({"error": {"message": "internal"}}).to_string())
            .create_async()
            .await;

        let adapter = make_llama(&server);
        let req = ChatRequest::builder("llama-4-maverick", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 500),
            _ => panic!("应为 Api"),
        }
    }

    // ============ Llama chat_stream ============

    #[tokio::test]
    async fn llama_chat_stream_parses_sse_chunks() {
        let mut server = Server::new_async().await;
        let sse = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"llama-4-maverick\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"llama-4-maverick\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"llama-4-maverick\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;

        let adapter = make_llama(&server);
        let req = ChatRequest::builder("llama-4-maverick", vec![ChatMessage::user("hi")]).build();
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
        assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn llama_chat_stream_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(json!({"error": {"message": "Unauthorized"}}).to_string())
            .create_async()
            .await;

        let adapter = make_llama(&server);
        let req = ChatRequest::builder("llama-4-maverick", vec![ChatMessage::user("hi")]).build();
        let result = adapter.chat_stream(req).await;
        match result {
            Err(e) => assert!(matches!(e, AibridgeError::Authentication { .. })),
            Ok(_) => panic!("chat_stream 应返回错误而非 stream"),
        }
    }

    // ============ Llama list_models ============

    #[tokio::test]
    async fn llama_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(
                json!({
                    "object": "list",
                    "data": [
                        {"id": "llama-4-maverick", "object": "model", "created": 1700000000, "owned_by": "meta"},
                        {"id": "llama-4-scout", "object": "model", "created": 1700000000, "owned_by": "meta"}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_llama(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "llama-4-maverick");
        assert_eq!(models[0].provider, "llama");
        assert_eq!(models[1].id, "llama-4-scout");
    }

    #[tokio::test]
    async fn llama_list_models_filter_by_chat_type() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "llama-4-maverick", "object": "model", "created": 1},
                        {"id": "dall-e-3", "object": "model", "created": 1}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_llama(&server);
        let chats = adapter.list_models(Some(ModelType::Chat)).await.unwrap();
        assert_eq!(chats.len(), 1);
        assert_eq!(chats[0].id, "llama-4-maverick");
    }

    #[tokio::test]
    async fn llama_list_models_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_llama(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn llama_list_models_error_429() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow"}}).to_string())
            .create_async()
            .await;

        let adapter = make_llama(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    // ============ Llama 不支持的能力 ============

    #[tokio::test]
    async fn llama_image_generate_returns_unsupported() {
        let adapter = make_llama_no_server();
        let req = ImageRequest::builder("llama-4-maverick", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn llama_video_create_returns_unsupported() {
        let adapter = make_llama_no_server();
        let req = VideoRequest::builder("llama-4-maverick", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ start / close ============

    #[tokio::test]
    async fn ideogram_start_and_close_are_noops() {
        let mut adapter = make_ideogram_no_server();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    #[tokio::test]
    async fn luma_start_and_close_are_noops() {
        let mut adapter = make_luma_no_server();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    #[tokio::test]
    async fn llama_start_and_close_are_noops() {
        let mut adapter = make_llama_no_server();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    // ============ 错误映射单元测试 ============

    #[test]
    fn ideogram_map_api_error_401_is_authentication() {
        let err = IdeogramAdapter::map_api_error(401, "");
        match err {
            AibridgeError::Authentication { message } => {
                assert!(message.contains("Ideogram"));
            }
            _ => panic!("应为 Authentication"),
        }
    }

    #[test]
    fn ideogram_map_api_error_402_is_api_payment() {
        let err = IdeogramAdapter::map_api_error(402, "");
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 402);
                assert!(message.contains("credits"));
            }
            _ => panic!("应为 Api (402)"),
        }
    }

    #[test]
    fn ideogram_map_api_error_429_is_rate_limit() {
        let err = IdeogramAdapter::map_api_error(429, "");
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[test]
    fn ideogram_map_api_error_500_extracts_message() {
        let body = json!({"error": "internal error"}).to_string();
        let err = IdeogramAdapter::map_api_error(500, &body);
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 500);
                assert_eq!(message, "internal error");
            }
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn luma_map_api_error_401_is_authentication() {
        let err = LumaAdapter::map_api_error(401, "");
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[test]
    fn luma_map_api_error_402_is_api_payment() {
        let err = LumaAdapter::map_api_error(402, "");
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 402),
            _ => panic!("应为 Api (402)"),
        }
    }

    #[test]
    fn luma_map_api_error_429_is_rate_limit() {
        let err = LumaAdapter::map_api_error(429, "");
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[test]
    fn luma_map_api_error_extracts_detail_field() {
        let body = json!({"detail": "validation failed"}).to_string();
        let err = LumaAdapter::map_api_error(400, &body);
        match err {
            AibridgeError::Api { message, .. } => {
                assert_eq!(message, "validation failed");
            }
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn luma_map_api_error_extracts_detail_array() {
        let body = json!({"detail": ["error1", "error2"]}).to_string();
        let err = LumaAdapter::map_api_error(422, &body);
        match err {
            AibridgeError::Api { message, .. } => {
                assert_eq!(message, "error1; error2");
            }
            _ => panic!("应为 Api"),
        }
    }

    // ============ map_luma_status 单元测试 ============

    #[test]
    fn map_luma_status_queued_is_pending() {
        assert_eq!(map_luma_status("queued"), TaskStatus::Pending);
    }

    #[test]
    fn map_luma_status_dreaming_is_processing() {
        assert_eq!(map_luma_status("dreaming"), TaskStatus::Processing);
    }

    #[test]
    fn map_luma_status_completed_is_success() {
        assert_eq!(map_luma_status("completed"), TaskStatus::Success);
    }

    #[test]
    fn map_luma_status_failed_is_failed() {
        assert_eq!(map_luma_status("failed"), TaskStatus::Failed);
    }

    #[test]
    fn map_luma_status_case_insensitive() {
        assert_eq!(map_luma_status("QUEUED"), TaskStatus::Pending);
        assert_eq!(map_luma_status("Completed"), TaskStatus::Success);
    }

    #[test]
    fn map_luma_status_unknown_defaults_to_pending() {
        assert_eq!(map_luma_status("unknown_state"), TaskStatus::Pending);
    }

    // ============ parse_iso_to_timestamp / civil_to_unix 单元测试 ============

    #[test]
    fn parse_iso_to_timestamp_z_suffix() {
        // 2024-01-01T00:00:00Z = 1704067200
        let ts = parse_iso_to_timestamp("2024-01-01T00:00:00Z");
        assert_eq!(ts, Some(1704067200));
    }

    #[test]
    fn parse_iso_to_timestamp_with_offset() {
        // 2024-01-01T00:00:00+08:00 = 1704067200 - 8*3600 = 1704038400
        let ts = parse_iso_to_timestamp("2024-01-01T00:00:00+08:00");
        assert_eq!(ts, Some(1704038400));
    }

    #[test]
    fn parse_iso_to_timestamp_invalid_returns_none() {
        assert_eq!(parse_iso_to_timestamp("not a date"), None);
        assert_eq!(parse_iso_to_timestamp("2024"), None);
    }

    #[test]
    fn civil_to_unix_epoch() {
        // 1970-01-01T00:00:00 = 0
        assert_eq!(civil_to_unix(1970, 1, 1, 0, 0, 0), Some(0));
    }

    #[test]
    fn civil_to_unix_known_date() {
        // 2024-01-01T00:00:00 = 1704067200
        assert_eq!(civil_to_unix(2024, 1, 1, 0, 0, 0), Some(1704067200));
    }

    #[test]
    fn civil_to_unix_invalid_date_returns_none() {
        assert_eq!(civil_to_unix(2024, 13, 1, 0, 0, 0), None);
        assert_eq!(civil_to_unix(2024, 1, 32, 0, 0, 0), None);
        assert_eq!(civil_to_unix(2024, 1, 1, 24, 0, 0), None);
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

    // ============ build_generate_body / build_generations_body 单元测试 ============

    #[test]
    fn ideogram_build_generate_body_wraps_in_image_request() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("ideogram", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = IdeogramAdapter::with_http(http, config);
        let req = ImageRequest::builder("V_2A", "a cat")
            .aspect_ratio("16:9")
            .n(2)
            .build();
        let body = adapter.build_generate_body(&req);
        // 应被 image_request 包裹
        assert!(body.get("image_request").is_some());
        let inner = body.get("image_request").unwrap();
        assert_eq!(inner["model"], "V_2A");
        assert_eq!(inner["prompt"], "a cat");
        // aspect_ratio 归一化为大写
        assert_eq!(inner["aspect_ratio"], "16:9");
        assert_eq!(inner["num_images"], 2);
    }

    #[test]
    fn luma_build_generations_body_includes_fields() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("luma", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = LumaAdapter::with_http(http, config);
        let req = VideoRequest::builder("ray-2", "a cat")
            .aspect_ratio("16:9")
            .duration(5)
            .resolution("720p")
            .build();
        let body = adapter.build_generations_body(&req);
        assert_eq!(body["model"], "ray-2");
        assert_eq!(body["prompt"], "a cat");
        assert_eq!(body["aspect_ratio"], "16:9");
        // duration 转为 "5s" 字符串
        assert_eq!(body["duration"], "5s");
        assert_eq!(body["resolution"], "720p");
    }

    #[test]
    fn luma_build_generations_body_no_keyframes_when_empty() {
        let opts = ClientOptions::builder().base_url("https://x").build();
        let config = ProviderConfig::from_options("luma", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url("https://x").build()).unwrap();
        let adapter = LumaAdapter::with_http(http, config);
        let req = VideoRequest::builder("ray-2", "a cat").build();
        let body = adapter.build_generations_body(&req);
        // 无 keyframes 字段
        assert!(body.get("keyframes").is_none());
    }
}
