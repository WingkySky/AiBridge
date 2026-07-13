//! 更多主流模型适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/additional_models.py`。
//!
//! 支持五个 OpenAI 兼容协议族的 Provider（差异主要在 base_url / provider_type / 能力声明）：
//! - **xAI Grok**：`https://api.x.ai/v1`，chat + vision
//! - **零一万物 Yi**：`https://api.lingyiwanwu.com/v1`，chat + vision
//! - **商汤日日新 SenseNova**：`https://api.sensenova.cn/v1/cc-switch`，chat + vision，
//!   list_models 端点为 `../llm/models`（base_url 末段是 `/cc-switch`，需相对路径上跳一级）
//! - **腾讯混元 Hunyuan**：`https://hunyuan.tencentcloudapi.com/v1`，chat + vision
//! - **Groq**：`https://api.groq.com/openai/v1`，chat + chat_stream + vision（Python 还声明
//!   AUDIO_TRANSCRIBE/TRANSLATE，但阶段 2a 范围仅核心对话能力，audio 走 trait 默认实现）
//!
//! ## 结构
//!
//! Python v1 是 5 个独立 adapter 类，统一注册到工厂。Rust 同样实现 5 个独立 struct：
//! - `GrokAdapter` / `YiAdapter` / `HunyuanAdapter` / `GroqAdapter`：组合 `OpenAiCompatAdapter`
//!   地基，chat/chat_stream/list_models 全部委托（仅 base_url/provider_type/capabilities 差异）
//! - `SenseNovaAdapter`：chat/chat_stream 委托地基的请求体构造与响应解析，但 HTTP 层与
//!   list_models 端点独立实现（因 SenseNova 的 base_url 末段为 `/cc-switch`，list_models
//!   需用相对路径 `../llm/models` 上跳一级，与地基硬编码的 `models` 路径不同）
//!
//! ## provider_type 标识（对齐 Python v1）
//!
//! | Provider | provider_type | 别名 |
//! |---|---|---|
//! | xAI Grok | `grok` | `xaigrok` |
//! | 零一万物 Yi | `yi` | `lingyiwanwu` |
//! | 商汤日日新 SenseNova | `sensenova` | `shangtang` |
//! | 腾讯混元 Hunyuan | `hunyuan` | `tencent_hunyuan` |
//! | Groq | `groq` | — |
//!
//! ## 阶段范围
//!
//! 阶段 2a 仅实现核心对话能力（chat / chat_stream）+ list_models 实时拉取。
//! Python 老版中 Groq 声明的 audio transcribe/translate 与各 Provider 的 image/video
//! 不支持能力，均走 `Adapter` trait 默认实现返 `UnsupportedCapability`，
//! 待阶段 2c audio_adapters 统一补齐 audio 部分。

use async_trait::async_trait;

use crate::adapter::{Adapter, Capabilities, CapabilitySet, ChatStream};
use crate::adapters::openai_compat::OpenAiCompatAdapter;
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::chat::{ChatCompletion, ChatRequest};
use crate::model::common::{infer_model_type, ModelInfo, ModelType};

// ==================== 默认 Base URL ====================

/// xAI Grok 默认 Base URL
///
/// 对应 Python v1 `GrokAdapter.DEFAULT_BASE_URL`。
pub const DEFAULT_GROK_BASE_URL: &str = "https://api.x.ai/v1";

/// 零一万物 Yi 默认 Base URL
///
/// 对应 Python v1 `YiAdapter.DEFAULT_BASE_URL`。
pub const DEFAULT_YI_BASE_URL: &str = "https://api.lingyiwanwu.com/v1";

/// 商汤日日新 SenseNova 默认 Base URL
///
/// 对应 Python v1 `SenseNovaAdapter.DEFAULT_BASE_URL`。
/// 末段 `/cc-switch` 是 chat 端点的 base，list_models 需上跳到 `/v1/llm/models`。
pub const DEFAULT_SENSENOVA_BASE_URL: &str = "https://api.sensenova.cn/v1/cc-switch";

/// 腾讯混元 Hunyuan 默认 Base URL
///
/// 对应 Python v1 `HunyuanAdapter.DEFAULT_BASE_URL`。
pub const DEFAULT_HUNYUAN_BASE_URL: &str = "https://hunyuan.tencentcloudapi.com/v1";

/// Groq 默认 Base URL
///
/// 对应 Python v1 `GroqAdapter.DEFAULT_BASE_URL`。
pub const DEFAULT_GROQ_BASE_URL: &str = "https://api.groq.com/openai/v1";

// ==================== 能力集合构造 ====================

/// xAI Grok 支持的能力集合
///
/// 对齐 Python v1 `GrokAdapter.supported_capabilities = ["chat", "vision"]`。
/// chat_stream 虽未在 Python 显式声明，但 Python 实现了该方法且 OpenAI 兼容协议天然支持流式，
/// 故 Rust 一并声明 ChatStream（与 openai/azure 等兼容适配器保持一致）。
fn grok_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// 零一万物 Yi 支持的能力集合
///
/// 对齐 Python v1 `YiAdapter.supported_capabilities = ["chat", "vision"]`。
fn yi_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// 商汤日日新 SenseNova 支持的能力集合
///
/// 对齐 Python v1 `SenseNovaAdapter.supported_capabilities = ["chat", "vision"]`。
fn sensenova_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// 腾讯混元 Hunyuan 支持的能力集合
///
/// 对齐 Python v1 `HunyuanAdapter.supported_capabilities = ["chat", "vision"]`。
fn hunyuan_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// Groq 支持的能力集合
///
/// 对齐 Python v1 `GroqAdapter.supported_capabilities`（CHAT / CHAT_STREAM / VISION +
/// AUDIO_TRANSCRIBE / AUDIO_TRANSLATE）。阶段 2a 范围仅核心对话能力，
/// audio 走 trait 默认实现返 UnsupportedCapability，待阶段 2c 补齐。
fn groq_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

// ==================== xAI Grok 适配器 ====================

/// xAI Grok 适配器
///
/// OpenAI 兼容协议，全部能力委托给 `OpenAiCompatAdapter` 地基。
///
/// - Base URL: `https://api.x.ai/v1`
/// - Chat: `POST /chat/completions`
/// - Models: `GET /models`
/// - 认证: Bearer Token
pub struct GrokAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl GrokAdapter {
    /// 创建 xAI Grok 适配器
    ///
    /// `config.base_url` 为空时回退到 [`DEFAULT_GROK_BASE_URL`]。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            "grok",
            "xAI Grok",
            DEFAULT_GROK_BASE_URL,
            grok_capabilities(),
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
impl Adapter for GrokAdapter {
    fn provider_type(&self) -> &str {
        "grok"
    }

    fn provider_name(&self) -> &str {
        "xAI Grok"
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

    // image_generate / embed / video / audio 走 trait 默认实现返 UnsupportedCapability，
    // 与 Python v1 抛 UnsupportedCapabilityError 行为一致。
}

// ==================== 零一万物 Yi 适配器 ====================

/// 零一万物 Yi 适配器
///
/// OpenAI 兼容协议，全部能力委托给 `OpenAiCompatAdapter` 地基。
///
/// - Base URL: `https://api.lingyiwanwu.com/v1`
/// - Chat: `POST /chat/completions`
/// - Models: `GET /models`
/// - 认证: Bearer Token
pub struct YiAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl YiAdapter {
    /// 创建零一万物 Yi 适配器
    ///
    /// `config.base_url` 为空时回退到 [`DEFAULT_YI_BASE_URL`]。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            "yi",
            "零一万物 Yi",
            DEFAULT_YI_BASE_URL,
            yi_capabilities(),
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
impl Adapter for YiAdapter {
    fn provider_type(&self) -> &str {
        "yi"
    }

    fn provider_name(&self) -> &str {
        "零一万物 Yi"
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

// ==================== 商汤日日新 SenseNova 适配器 ====================

/// 商汤日日新 SenseNova 适配器
///
/// OpenAI 兼容协议，chat/chat_stream 复用地基的请求体构造与响应解析，
/// 但 list_models 端点独立实现：
/// - Base URL: `https://api.sensenova.cn/v1/cc-switch`（末段 `/cc-switch` 是 chat 端点 base）
/// - Chat: `POST /chat/completions`（即 `.../v1/cc-switch/chat/completions`）
/// - Models: `GET ../llm/models`（相对路径上跳一级到 `.../v1/llm/models`）
/// - 认证: Bearer Token
///
/// 因地基地 `list_models` 硬编码相对路径 `models`（会拼成 `.../cc-switch/models`，错误），
/// 故本适配器持有一个独立 HttpClient，自行实现 list_models；
/// chat/chat_stream 仍委托地基（地基的 `chat/completions` 相对路径与 SenseNova 一致）。
pub struct SenseNovaAdapter {
    /// OpenAI 兼容地基（chat/chat_stream 委托给它，复用请求体构造与响应解析）
    compat: OpenAiCompatAdapter,
    /// list_models 端点专用 HTTP 客户端（独立于 compat，避免暴露 compat 私有字段）
    http: HttpClient,
    /// list_models 端点的完整 URL（已上跳一级，形如 `.../v1/llm/models`）
    ///
    /// 由 base_url 末段去掉 `/cc-switch` 后拼 `llm/models` 得到。
    /// 用 String 而非每次计算，避免重复字符串处理。
    models_url: String,
    /// API key（Bearer 认证用，可能为空）
    api_key: Option<String>,
}

impl SenseNovaAdapter {
    /// 创建商汤日日新 SenseNova 适配器
    ///
    /// 解析顺序（与 Python v1 `__init__` 一致）：
    /// 1. `config.base_url` 为空时用 [`DEFAULT_SENSENOVA_BASE_URL`] 兜底
    /// 2. 由 base_url 计算 models_url：取 base_url 去掉末段（`/cc-switch`）后拼 `llm/models`
    ///
    /// # models_url 计算示例
    /// - base_url = `https://api.sensenova.cn/v1/cc-switch`
    /// - models_url = `https://api.sensenova.cn/v1/llm/models`
    ///
    /// # 错误
    /// - HTTP 客户端构造失败（罕见，reqwest 配置错误）
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let caps = sensenova_capabilities();

        // 解析 base_url：config 优先，否则默认值
        let base_url = config
            .base_url
            .clone()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_SENSENOVA_BASE_URL.to_string());

        // 由 base_url 计算 models_url：去掉末段路径，拼 `llm/models`
        // base_url 形如 `.../v1/cc-switch`，去掉最后一段得到 `.../v1`，再拼 `llm/models`
        let models_url = build_sensenova_models_url(&base_url);

        // 构造 chat 用的 compat 地基（base_url 保持原值，chat/completions 相对路径匹配）
        let compat = OpenAiCompatAdapter::new(
            config.clone(),
            "sensenova",
            "商汤日日新 SenseNova",
            &base_url,
            caps,
        )?;

        // 构造 list_models 用的 HttpClient（不设 base_url，请求时用完整 models_url）
        let http_opts = ClientOptions::builder()
            .api_key(config.api_key.clone().unwrap_or_default())
            .timeout(config.timeout)
            .max_retries(config.max_retries)
            .retry_delay(config.retry_delay)
            .build();
        let http = HttpClient::new(&http_opts)?;

        Ok(Self {
            compat,
            http,
            models_url,
            api_key: config.api_key.clone(),
        })
    }

    /// 用显式 compat + http 构造（测试用，可注入 mockito 后端）
    ///
    /// `models_url` 为 list_models 的完整 URL；`api_key` 用于 Bearer 认证。
    #[cfg(test)]
    pub fn with_compat(
        compat: OpenAiCompatAdapter,
        http: HttpClient,
        models_url: String,
        api_key: Option<String>,
    ) -> Self {
        Self {
            compat,
            http,
            models_url,
            api_key,
        }
    }

    /// list_models 端点完整 URL（已上跳一级，形如 `.../v1/llm/models`）
    pub fn models_url(&self) -> &str {
        &self.models_url
    }

    /// 发送带 Bearer 认证的 GET 请求，并用 OpenAI 错误映射处理响应
    ///
    /// SenseNova 错误体结构与 OpenAI 一致（`{"error": {"message": "..."}}`），
    /// 故复用 [`OpenAiCompatAdapter::map_api_error`]。
    async fn get_authed_json(&self, url: &str) -> Result<serde_json::Value> {
        let resp = self
            .http
            .inner()
            .get(url)
            .bearer_auth(self.api_key.as_deref().unwrap_or(""))
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
        resp.json::<serde_json::Value>()
            .await
            .map_err(AibridgeError::from)
    }
}

/// 由 SenseNova chat base_url 计算 list_models 端点 URL
///
/// base_url 末段为 `/cc-switch`（chat 端点 base），需上跳一级拼 `llm/models`。
/// 例：`https://api.sensenova.cn/v1/cc-switch` → `https://api.sensenova.cn/v1/llm/models`。
///
/// 实现方式：找到最后一个 `/`，截到其前一位；若末段已是空（以 `/` 结尾），则直接拼 `llm/models`。
/// 保留 scheme + host 段不变（只处理 path 部分）。
fn build_sensenova_models_url(base_url: &str) -> String {
    // 去掉末尾斜杠，统一处理
    let trimmed = base_url.trim_end_matches('/');
    // 截到倒数第二个 `/`（即去掉最后一段 path）
    // 例：`https://host/v1/cc-switch` → `https://host/v1`
    let parent = match trimmed.rfind('/') {
        Some(idx) => &trimmed[..idx],
        None => trimmed,
    };
    format!("{parent}/llm/models")
}

#[async_trait]
impl Adapter for SenseNovaAdapter {
    fn provider_type(&self) -> &str {
        "sensenova"
    }

    fn provider_name(&self) -> &str {
        "商汤日日新 SenseNova"
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

    /// 文本对话：委托地基 `POST /chat/completions`
    ///
    /// SenseNova 的 base_url 末段 `/cc-switch` + 地基相对路径 `chat/completions`
    /// 正好拼出 `.../v1/cc-switch/chat/completions`，与官方端点一致。
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.compat.chat(req).await
    }

    /// 流式文本对话：委托地基 `POST /chat/completions` (stream=true)
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.compat.chat_stream(req).await
    }

    /// 模型列表（实时拉取）
    ///
    /// 调用 `GET {models_url}`（即 `.../v1/llm/models`），解析响应。
    /// 响应结构与 OpenAI `/models` 一致：`{"data": [{"id": "...", ...}]}`。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let value = self.get_authed_json(&self.models_url).await?;
        let models = parse_models_response(&value, "sensenova");
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }
}

// ==================== 腾讯混元 Hunyuan 适配器 ====================

/// 腾讯混元 Hunyuan 适配器
///
/// OpenAI 兼容协议，全部能力委托给 `OpenAiCompatAdapter` 地基。
///
/// - Base URL: `https://hunyuan.tencentcloudapi.com/v1`
/// - Chat: `POST /chat/completions`
/// - Models: `GET /models`
/// - 认证: Bearer Token
pub struct HunyuanAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl HunyuanAdapter {
    /// 创建腾讯混元 Hunyuan 适配器
    ///
    /// `config.base_url` 为空时回退到 [`DEFAULT_HUNYUAN_BASE_URL`]。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            "hunyuan",
            "腾讯混元 Hunyuan",
            DEFAULT_HUNYUAN_BASE_URL,
            hunyuan_capabilities(),
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
impl Adapter for HunyuanAdapter {
    fn provider_type(&self) -> &str {
        "hunyuan"
    }

    fn provider_name(&self) -> &str {
        "腾讯混元 Hunyuan"
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

// ==================== Groq 适配器 ====================

/// Groq 适配器
///
/// OpenAI 兼容协议，全部核心能力委托给 `OpenAiCompatAdapter` 地基。
///
/// - Base URL: `https://api.groq.com/openai/v1`
/// - Chat: `POST /chat/completions`
/// - Models: `GET /models`
/// - 认证: Bearer Token
///
/// Python v1 还声明了 AUDIO_TRANSCRIBE / AUDIO_TRANSLATE（Whisper 语音识别），
/// 阶段 2a 范围仅核心对话能力，audio 走 trait 默认实现返 UnsupportedCapability，
/// 待阶段 2c audio_adapters 补齐。
pub struct GroqAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl GroqAdapter {
    /// 创建 Groq 适配器
    ///
    /// `config.base_url` 为空时回退到 [`DEFAULT_GROQ_BASE_URL`]。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            "groq",
            "Groq",
            DEFAULT_GROQ_BASE_URL,
            groq_capabilities(),
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
impl Adapter for GroqAdapter {
    fn provider_type(&self) -> &str {
        "groq"
    }

    fn provider_name(&self) -> &str {
        "Groq"
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

    // image_generate / video / audio 走 trait 默认实现返 UnsupportedCapability，
    // 与 Python v1 抛 UnsupportedCapabilityError 行为一致（speech 显式抛错也走默认实现）。
}

// ==================== 内部：模型列表响应解析 ====================

/// 解析 OpenAI 兼容 `/models`（或 SenseNova `/llm/models`）响应 → Vec<ModelInfo>
///
/// 响应格式：`{"data": [{"id": "...", "created": ..., "owned_by": "..."}]}`
/// 与 [`OpenAiCompatAdapter`] 内部的 `parse_models` 等价，但因该方法是私有，本模块独立实现一份。
///
/// `provider` 填入对应 provider_type（如 "sensenova"），用于 ModelInfo.provider 字段。
fn parse_models_response(value: &serde_json::Value, provider: &str) -> Vec<ModelInfo> {
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
                    provider: provider.to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AibridgeError;
    use crate::http::HttpClient;
    use crate::model::chat::ChatMessage;
    use crate::model::image::ImageRequest;
    use crate::model::video::VideoRequest;
    use futures::stream::StreamExt;
    use mockito::Server;
    use serde_json::json;

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

    /// 构造测试用 GrokAdapter（指向 mockito server）
    fn make_grok(server: &Server) -> GrokAdapter {
        let compat = make_compat(server, "grok", "xAI Grok", grok_capabilities());
        GrokAdapter::with_compat(compat)
    }

    /// 构造测试用 YiAdapter（指向 mockito server）
    fn make_yi(server: &Server) -> YiAdapter {
        let compat = make_compat(server, "yi", "零一万物 Yi", yi_capabilities());
        YiAdapter::with_compat(compat)
    }

    /// 构造测试用 SenseNovaAdapter（指向 mockito server）
    ///
    /// mockito server 的 URL 作为 chat base_url（形如 `http://127.0.0.1:port`），
    /// chat 端点走 compat 地基，请求 `{server}/chat/completions`；
    /// list_models 走独立 http，请求 `{server}/llm/models`（直接覆盖 models_url 以隔离计算逻辑）。
    fn make_sensenova(server: &Server) -> SenseNovaAdapter {
        let compat = make_compat(
            server,
            "sensenova",
            "商汤日日新 SenseNova",
            sensenova_capabilities(),
        );
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        let models_url = format!("{}/llm/models", server.url());
        SenseNovaAdapter::with_compat(compat, http, models_url, Some("test-key".to_string()))
    }

    /// 构造测试用 HunyuanAdapter（指向 mockito server）
    fn make_hunyuan(server: &Server) -> HunyuanAdapter {
        let compat = make_compat(
            server,
            "hunyuan",
            "腾讯混元 Hunyuan",
            hunyuan_capabilities(),
        );
        HunyuanAdapter::with_compat(compat)
    }

    /// 构造测试用 GroqAdapter（指向 mockito server）
    fn make_groq(server: &Server) -> GroqAdapter {
        let compat = make_compat(server, "groq", "Groq", groq_capabilities());
        GroqAdapter::with_compat(compat)
    }

    /// 构造不指向任何 server 的 GrokAdapter（用于不发请求的元信息/能力测试）
    fn make_grok_no_server() -> GrokAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(DEFAULT_GROK_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("grok", opts);
        GrokAdapter::new(config).expect("GrokAdapter 构造应成功")
    }

    /// 构造不指向任何 server 的 YiAdapter
    fn make_yi_no_server() -> YiAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(DEFAULT_YI_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("yi", opts);
        YiAdapter::new(config).expect("YiAdapter 构造应成功")
    }

    /// 构造不指向任何 server 的 HunyuanAdapter
    fn make_hunyuan_no_server() -> HunyuanAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(DEFAULT_HUNYUAN_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("hunyuan", opts);
        HunyuanAdapter::new(config).expect("HunyuanAdapter 构造应成功")
    }

    /// 构造不指向任何 server 的 GroqAdapter
    fn make_groq_no_server() -> GroqAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(DEFAULT_GROQ_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("groq", opts);
        GroqAdapter::new(config).expect("GroqAdapter 构造应成功")
    }

    // ============ Grok 元信息 ============

    #[test]
    fn grok_provider_type_and_name_match_python() {
        let adapter = make_grok_no_server();
        assert_eq!(adapter.provider_type(), "grok");
        assert_eq!(adapter.provider_name(), "xAI Grok");
    }

    #[test]
    fn grok_requires_api_key_is_true() {
        let adapter = make_grok_no_server();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn grok_capabilities_contains_chat_and_vision() {
        let adapter = make_grok_no_server();
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::Vision));
        // image / video / audio 不声明（阶段 2a 范围）
        assert!(!caps.contains(&Capabilities::ImageGenerate));
        assert!(!caps.contains(&Capabilities::VideoGenerate));
    }

    #[test]
    fn grok_base_url_defaults_when_missing() {
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("grok", opts);
        let adapter = GrokAdapter::new(config).unwrap();
        assert_eq!(adapter.compat.base_url(), DEFAULT_GROK_BASE_URL);
    }

    #[test]
    fn grok_base_url_uses_config_when_provided() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://custom.grok-proxy.com/v1")
            .build();
        let config = ProviderConfig::from_options("grok", opts);
        let adapter = GrokAdapter::new(config).unwrap();
        assert_eq!(
            adapter.compat.base_url(),
            "https://custom.grok-proxy.com/v1"
        );
    }

    // ============ Grok chat ============

    #[tokio::test]
    async fn grok_chat_success_returns_completion() {
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
                    "model": "grok-3",
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

        let adapter = make_grok(&server);
        let req = ChatRequest::builder("grok-3", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .max_tokens(100)
            .build();
        let resp = adapter.chat(req).await.expect("chat 应成功");

        assert_eq!(resp.id, "chatcmpl-1");
        assert_eq!(resp.model, "grok-3");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 7);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn grok_chat_sends_temperature_and_max_tokens() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "grok-3",
                "temperature": 0.5,
                "max_tokens": 50
            })))
            .with_status(200)
            .with_body(json!({
                "id": "x", "object": "chat.completion", "created": 1, "model": "grok-3",
                "choices": [{"index": 0, "message": {"role":"assistant","content":"ok"}, "finish_reason": "stop"}]
            }).to_string())
            .create_async()
            .await;

        let adapter = make_grok(&server);
        let req = ChatRequest::builder("grok-3", vec![ChatMessage::user("hi")])
            .temperature(0.5)
            .max_tokens(50)
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn grok_chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(json!({"error": {"message": "Invalid xAI API key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_grok(&server);
        let req = ChatRequest::builder("grok-3", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn grok_chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;

        let adapter = make_grok(&server);
        let req = ChatRequest::builder("grok-3", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn grok_chat_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(500)
            .with_body(json!({"error": {"message": "internal"}}).to_string())
            .create_async()
            .await;

        let adapter = make_grok(&server);
        let req = ChatRequest::builder("grok-3", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 500),
            _ => panic!("应为 Api"),
        }
    }

    // ============ Grok chat_stream ============

    #[tokio::test]
    async fn grok_chat_stream_parses_sse_chunks() {
        let mut server = Server::new_async().await;
        let sse = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"grok-3\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"grok-3\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"grok-3\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;

        let adapter = make_grok(&server);
        let req = ChatRequest::builder("grok-3", vec![ChatMessage::user("hi")]).build();
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
    async fn grok_chat_stream_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(json!({"error": {"message": "Unauthorized"}}).to_string())
            .create_async()
            .await;

        let adapter = make_grok(&server);
        let req = ChatRequest::builder("grok-3", vec![ChatMessage::user("hi")]).build();
        let result = adapter.chat_stream(req).await;
        match result {
            Err(e) => assert!(matches!(e, AibridgeError::Authentication { .. })),
            Ok(_) => panic!("chat_stream 应返回错误而非 stream"),
        }
    }

    // ============ Grok list_models ============

    #[tokio::test]
    async fn grok_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(json!({
                "object": "list",
                "data": [
                    {"id": "grok-3", "object": "model", "created": 1700000000, "owned_by": "xai"},
                    {"id": "grok-2-vision", "object": "model", "created": 1700000000, "owned_by": "xai"}
                ]
            }).to_string())
            .create_async()
            .await;

        let adapter = make_grok(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "grok-3");
        assert_eq!(models[0].provider, "grok");
        assert_eq!(models[1].id, "grok-2-vision");
    }

    #[tokio::test]
    async fn grok_list_models_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_grok(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    // ============ Grok 不支持的能力 ============

    #[tokio::test]
    async fn grok_image_generate_returns_unsupported() {
        let adapter = make_grok_no_server();
        let req = ImageRequest::builder("grok-2-image", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn grok_video_create_returns_unsupported() {
        let adapter = make_grok_no_server();
        let req = VideoRequest::builder("grok-video", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ Yi 元信息 + chat + list_models ============

    #[test]
    fn yi_provider_type_and_name_match_python() {
        let adapter = make_yi_no_server();
        assert_eq!(adapter.provider_type(), "yi");
        assert_eq!(adapter.provider_name(), "零一万物 Yi");
    }

    #[test]
    fn yi_requires_api_key_is_true() {
        let adapter = make_yi_no_server();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn yi_base_url_defaults_when_missing() {
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("yi", opts);
        let adapter = YiAdapter::new(config).unwrap();
        assert_eq!(adapter.compat.base_url(), DEFAULT_YI_BASE_URL);
    }

    #[tokio::test]
    async fn yi_chat_success_returns_completion() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(
                json!({
                    "id": "chatcmpl-yi",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "yi-large",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "你好！"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_yi(&server);
        let req = ChatRequest::builder("yi-large", vec![ChatMessage::user("你好")]).build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.model, "yi-large");
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("你好！"));
    }

    #[tokio::test]
    async fn yi_chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(json!({"error": {"message": "Invalid Yi API key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_yi(&server);
        let req = ChatRequest::builder("yi-large", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn yi_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "yi-large", "object": "model", "created": 1, "owned_by": "01ai"},
                        {"id": "yi-vision", "object": "model", "created": 1, "owned_by": "01ai"}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_yi(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "yi-large");
        assert_eq!(models[0].provider, "yi");
    }

    #[tokio::test]
    async fn yi_list_models_filter_by_chat_type() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "yi-large", "object": "model", "created": 1},
                        {"id": "yi-vision", "object": "model", "created": 1}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_yi(&server);
        // yi-large 和 yi-vision 都推断为 Chat 类型（infer_model_type 按名称关键字推断）
        let chats = adapter.list_models(Some(ModelType::Chat)).await.unwrap();
        assert_eq!(chats.len(), 2);
    }

    // ============ SenseNova 元信息 + chat + list_models ============

    #[test]
    fn sensenova_provider_type_and_name_match_python() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url(DEFAULT_SENSENOVA_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("sensenova", opts);
        let adapter = SenseNovaAdapter::new(config).expect("SenseNovaAdapter 构造应成功");
        assert_eq!(adapter.provider_type(), "sensenova");
        assert_eq!(adapter.provider_name(), "商汤日日新 SenseNova");
    }

    #[test]
    fn sensenova_requires_api_key_is_true() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url(DEFAULT_SENSENOVA_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("sensenova", opts);
        let adapter = SenseNovaAdapter::new(config).unwrap();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn sensenova_base_url_defaults_when_missing() {
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("sensenova", opts);
        let adapter = SenseNovaAdapter::new(config).unwrap();
        assert_eq!(adapter.compat.base_url(), DEFAULT_SENSENOVA_BASE_URL);
    }

    #[test]
    fn sensenova_models_url_builds_from_default_base() {
        // 默认 base_url = https://api.sensenova.cn/v1/cc-switch
        // models_url 应为 https://api.sensenova.cn/v1/llm/models
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url(DEFAULT_SENSENOVA_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("sensenova", opts);
        let adapter = SenseNovaAdapter::new(config).unwrap();
        assert_eq!(
            adapter.models_url(),
            "https://api.sensenova.cn/v1/llm/models"
        );
    }

    #[test]
    fn sensenova_models_url_builds_from_custom_base_with_trailing_slash() {
        // 自定义 base_url（含末尾斜杠也应正确处理）
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://custom.sensenova.cn/v2/cc-switch/")
            .build();
        let config = ProviderConfig::from_options("sensenova", opts);
        let adapter = SenseNovaAdapter::new(config).unwrap();
        assert_eq!(
            adapter.models_url(),
            "https://custom.sensenova.cn/v2/llm/models"
        );
    }

    #[tokio::test]
    async fn sensenova_chat_success_returns_completion() {
        let mut server = Server::new_async().await;
        // chat 端点走 compat 地基，相对路径 chat/completions 拼到 server 根
        server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(
                json!({
                    "id": "chatcmpl-sn",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "SenseChat-5",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "你好"},
                        "finish_reason": "stop"
                    }]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_sensenova(&server);
        let req = ChatRequest::builder("SenseChat-5", vec![ChatMessage::user("hi")]).build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.model, "SenseChat-5");
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("你好"));
    }

    #[tokio::test]
    async fn sensenova_chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;

        let adapter = make_sensenova(&server);
        let req = ChatRequest::builder("SenseChat-5", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn sensenova_chat_stream_parses_sse_chunks() {
        let mut server = Server::new_async().await;
        let sse = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"SenseChat-5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"你好\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"SenseChat-5\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;

        let adapter = make_sensenova(&server);
        let req = ChatRequest::builder("SenseChat-5", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("你好"));
    }

    #[tokio::test]
    async fn sensenova_list_models_success() {
        let mut server = Server::new_async().await;
        // list_models 走独立 http，请求 {server}/llm/models
        server
            .mock("GET", "/llm/models")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "SenseChat-5", "object": "model", "created": 1},
                        {"id": "SenseChat-Vision", "object": "model", "created": 1}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_sensenova(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "SenseChat-5");
        assert_eq!(models[0].provider, "sensenova");
    }

    #[tokio::test]
    async fn sensenova_list_models_filter_by_image_type() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/llm/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "SenseChat-5", "object": "model", "created": 1},
                        {"id": "dall-e-3", "object": "model", "created": 1}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_sensenova(&server);
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "dall-e-3");
    }

    #[tokio::test]
    async fn sensenova_list_models_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/llm/models")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_sensenova(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn sensenova_list_models_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/llm/models")
            .with_status(500)
            .with_body(json!({"error": {"message": "internal"}}).to_string())
            .create_async()
            .await;

        let adapter = make_sensenova(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 500),
            _ => panic!("应为 Api"),
        }
    }

    // ============ Hunyuan 元信息 + chat + list_models ============

    #[test]
    fn hunyuan_provider_type_and_name_match_python() {
        let adapter = make_hunyuan_no_server();
        assert_eq!(adapter.provider_type(), "hunyuan");
        assert_eq!(adapter.provider_name(), "腾讯混元 Hunyuan");
    }

    #[test]
    fn hunyuan_requires_api_key_is_true() {
        let adapter = make_hunyuan_no_server();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn hunyuan_base_url_defaults_when_missing() {
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("hunyuan", opts);
        let adapter = HunyuanAdapter::new(config).unwrap();
        assert_eq!(adapter.compat.base_url(), DEFAULT_HUNYUAN_BASE_URL);
    }

    #[tokio::test]
    async fn hunyuan_chat_success_returns_completion() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(
                json!({
                    "id": "chatcmpl-hy",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "hunyuan-pro",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "你好！"},
                        "finish_reason": "stop"
                    }]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_hunyuan(&server);
        let req = ChatRequest::builder("hunyuan-pro", vec![ChatMessage::user("hi")]).build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.model, "hunyuan-pro");
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("你好！"));
    }

    #[tokio::test]
    async fn hunyuan_chat_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(404)
            .with_body(json!({"error": {"message": "model hunyuan-x not found"}}).to_string())
            .create_async()
            .await;

        let adapter = make_hunyuan(&server);
        let req = ChatRequest::builder("hunyuan-x", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn hunyuan_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "hunyuan-pro", "object": "model", "created": 1},
                        {"id": "hunyuan-standard", "object": "model", "created": 1}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_hunyuan(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "hunyuan-pro");
        assert_eq!(models[0].provider, "hunyuan");
    }

    #[tokio::test]
    async fn hunyuan_list_models_error_429() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow"}}).to_string())
            .create_async()
            .await;

        let adapter = make_hunyuan(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    // ============ Groq 元信息 + chat + list_models ============

    #[test]
    fn groq_provider_type_and_name_match_python() {
        let adapter = make_groq_no_server();
        assert_eq!(adapter.provider_type(), "groq");
        assert_eq!(adapter.provider_name(), "Groq");
    }

    #[test]
    fn groq_requires_api_key_is_true() {
        let adapter = make_groq_no_server();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn groq_capabilities_contains_chat_stream() {
        let adapter = make_groq_no_server();
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::Vision));
        // audio 阶段 2a 不声明（走默认实现）
        assert!(!caps.contains(&Capabilities::AudioTranscribe));
        assert!(!caps.contains(&Capabilities::AudioSpeech));
    }

    #[test]
    fn groq_base_url_defaults_when_missing() {
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("groq", opts);
        let adapter = GroqAdapter::new(config).unwrap();
        assert_eq!(adapter.compat.base_url(), DEFAULT_GROQ_BASE_URL);
    }

    #[tokio::test]
    async fn groq_chat_success_returns_completion() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(
                json!({
                    "id": "chatcmpl-groq",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "llama-3.1-70b-versatile",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi!"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_groq(&server);
        let req =
            ChatRequest::builder("llama-3.1-70b-versatile", vec![ChatMessage::user("hi")]).build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.model, "llama-3.1-70b-versatile");
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hi!"));
        assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 4);
    }

    #[tokio::test]
    async fn groq_chat_passes_extra_params_through() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "llama-3.1-70b-versatile",
                "custom_param": "custom_value"
            })))
            .with_status(200)
            .with_body(json!({
                "id": "x", "object": "chat.completion", "created": 1, "model": "llama-3.1-70b-versatile",
                "choices": [{"index": 0, "message": {"role":"assistant","content":"ok"}, "finish_reason": "stop"}]
            }).to_string())
            .create_async()
            .await;

        let adapter = make_groq(&server);
        let req = ChatRequest::builder("llama-3.1-70b-versatile", vec![ChatMessage::user("hi")])
            .extra("custom_param", "custom_value")
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn groq_chat_error_400_returns_validation() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(400)
            .with_body(json!({"error": {"message": "max_tokens is invalid"}}).to_string())
            .create_async()
            .await;

        let adapter = make_groq(&server);
        let req =
            ChatRequest::builder("llama-3.1-70b-versatile", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Validation { message, .. } => {
                assert!(message.contains("max_tokens"));
            }
            _ => panic!("应为 Validation"),
        }
    }

    #[tokio::test]
    async fn groq_chat_stream_sends_stream_true() {
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

        let adapter = make_groq(&server);
        let req =
            ChatRequest::builder("llama-3.1-70b-versatile", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        while stream.next().await.is_some() {}
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn groq_list_models_success() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(json!({
                "data": [
                    {"id": "llama-3.1-70b-versatile", "object": "model", "created": 1, "owned_by": "groq"},
                    {"id": "whisper-large-v3", "object": "model", "created": 1, "owned_by": "groq"}
                ]
            }).to_string())
            .create_async()
            .await;

        let adapter = make_groq(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "llama-3.1-70b-versatile");
        assert_eq!(models[0].provider, "groq");
        // whisper 推断为 Audio 类型
        assert_eq!(models[1].model_type, ModelType::Audio);
    }

    #[tokio::test]
    async fn groq_list_models_filter_audio() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(
                json!({
                    "data": [
                        {"id": "llama-3.1-70b-versatile", "object": "model", "created": 1},
                        {"id": "whisper-large-v3", "object": "model", "created": 1}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_groq(&server);
        let audio = adapter.list_models(Some(ModelType::Audio)).await.unwrap();
        assert_eq!(audio.len(), 1);
        assert_eq!(audio[0].id, "whisper-large-v3");
    }

    // ============ Groq 不支持的能力（阶段 2a 范围）============

    #[tokio::test]
    async fn groq_speech_returns_unsupported() {
        // Python v1 Groq.speech 显式抛 UnsupportedCapabilityError（仅支持 Whisper ASR，无 TTS）
        let adapter = make_groq_no_server();
        let req = crate::model::audio::SpeechRequest::builder("tts-1", "hi", "alloy").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn groq_image_generate_returns_unsupported() {
        let adapter = make_groq_no_server();
        let req = ImageRequest::builder("groq-image", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ start / close ============

    #[tokio::test]
    async fn grok_start_and_close_are_noops() {
        let mut adapter = make_grok_no_server();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    #[tokio::test]
    async fn sensenova_start_and_close_are_noops() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url(DEFAULT_SENSENOVA_BASE_URL)
            .build();
        let config = ProviderConfig::from_options("sensenova", opts);
        let mut adapter = SenseNovaAdapter::new(config).unwrap();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    #[tokio::test]
    async fn groq_start_and_close_are_noops() {
        let mut adapter = make_groq_no_server();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    // ============ build_sensenova_models_url 单元测试 ============

    #[test]
    fn build_sensenova_models_url_default() {
        let url = build_sensenova_models_url("https://api.sensenova.cn/v1/cc-switch");
        assert_eq!(url, "https://api.sensenova.cn/v1/llm/models");
    }

    #[test]
    fn build_sensenova_models_url_trailing_slash() {
        let url = build_sensenova_models_url("https://api.sensenova.cn/v1/cc-switch/");
        assert_eq!(url, "https://api.sensenova.cn/v1/llm/models");
    }

    #[test]
    fn build_sensenova_models_url_custom_version() {
        let url = build_sensenova_models_url("https://custom.sensenova.cn/v2/cc-switch");
        assert_eq!(url, "https://custom.sensenova.cn/v2/llm/models");
    }
}
