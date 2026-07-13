//! 中文模型聚合适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/chinese.py`。
//!
//! 支持六个中文 AI 模型 Provider（文档见各适配器注释）：
//! - **通义千问 Qwen**（阿里 DashScope）：OpenAI 兼容协议
//! - **智谱 AI Zhipu**（GLM）：OpenAI 兼容协议
//! - **豆包 Doubao**（字节火山引擎方舟）：OpenAI 兼容协议
//! - **文心一言 ERNIE**（百度千帆）：**独立协议**（access_token 认证 + 特有端点 + 特有请求体/响应）
//! - **Kimi**（月之暗面 Moonshot AI）：OpenAI 兼容协议
//! - **MiniMax**（稀宇科技）：OpenAI 兼容协议（chat 部分）
//!
//! ## 结构
//!
//! Python v1 是 6 个独立 adapter 类，统一注册到工厂。Rust 同样实现 6 个独立 struct：
//! - `QwenAdapter` / `ZhipuAdapter` / `DoubaoAdapter` / `KimiAdapter`：组合
//!   [`OpenAiCompatAdapter`] 地基，chat/chat_stream/list_models 全部委托
//! - `MiniMaxAdapter`：组合地基复用 chat/chat_stream，但 list_models 走硬编码
//!   （MiniMax 无可靠的 `/models` 端点，对齐 Python 老版保留硬编码列表）
//! - `ErnieAdapter`：**独立协议实现**，仅复用地基的 `HttpClient` 与错误映射，
//!   chat/chat_stream/list_models 全部独立实现（百度特有 access_token 流程 + 端点 +
//!   `messages`/`system` 分离的请求体 + `result` 字段响应 + `is_end` 流式标记 +
//!   硬编码模型列表，因千帆 modellist 响应结构非 OpenAI 兼容）
//!
//! ## 能力声明策略
//!
//! Python 老版 Qwen/Doubao/MiniMax 声明了 `AUDIO_TRANSCRIBE`/`AUDIO_SPEECH` 能力并实现了
//! OpenAI 兼容的 transcribe/speech。但本阶段（2a）范围仅核心 chat/chat_stream/vision 能力，
//! audio 方法走 trait 默认实现返 `UnsupportedCapability`（与 azure 范例阶段 2a 策略一致，
//! 待阶段 2c audio_adapters 统一补齐）。故本模块不声明 audio 能力，避免能力声明与方法
//! 行为不一致。
//!
//! ## provider_type 标识（对齐 Python v1）
//!
//! | Provider | provider_type |
//! |---|---|
//! | 通义千问 Qwen | `qwen` |
//! | 智谱 AI Zhipu | `zhipu` |
//! | 豆包 Doubao | `doubao` |
//! | 文心一言 ERNIE | `ernie` |
//! | Kimi | `kimi` |
//! | MiniMax | `minimax` |
//!
//! 注册到工厂由 `adapter::factory` 完成（阶段 2a 收尾批次），本模块仅声明适配器类型。

use async_trait::async_trait;
use futures::stream::{StreamExt, TryStreamExt};
use serde_json::{json, Value};

use crate::adapter::{Adapter, Capabilities, CapabilitySet, ChatStream};
use crate::adapters::openai_compat::OpenAiCompatAdapter;
use crate::config::ProviderConfig;
use crate::error::{AibridgeError, Result};
use crate::model::chat::{
    ChatChoice, ChatCompletion, ChatCompletionChunk, ChatCompletionDelta, ChatMessage, ChatRequest,
    ChoiceMessage, DeltaMessage, UserContent,
};
use crate::model::common::{ModelInfo, ModelType};
use crate::util;

// ==================== 默认 Base URL ====================

/// 通义千问 Qwen 默认 Base URL
///
/// 对应 Python v1 `QwenAdapter.DEFAULT_BASE_URL`（DashScope OpenAI 兼容模式）。
/// 已含 `/compatible-mode/v1` 前缀，chat 端点为 `POST /chat/completions`，
/// models 端点为 `GET /models`。
pub const DEFAULT_QWEN_BASE_URL: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";

/// 智谱 AI Zhipu 默认 Base URL
///
/// 对应 Python v1 `ZhipuAdapter.DEFAULT_BASE_URL`（智谱 GLM OpenAI 兼容模式）。
/// 已含 `/api/paas/v4` 前缀，chat 端点为 `POST /chat/completions`，
/// models 端点为 `GET /models`。
pub const DEFAULT_ZHIPU_BASE_URL: &str = "https://open.bigmodel.cn/api/paas/v4";

/// 豆包 Doubao 默认 Base URL
///
/// 对应 Python v1 `DoubaoAdapter.DEFAULT_BASE_URL`（火山引擎方舟 OpenAI 兼容模式）。
/// 已含 `/api/v3` 前缀，chat 端点为 `POST /chat/completions`，
/// models 端点为 `GET /models`。
pub const DEFAULT_DOUBAO_BASE_URL: &str = "https://ark.cn-beijing.volces.com/api/v3";

/// 文心一言 ERNIE 默认 Base URL
///
/// 对应 Python v1 `ErnieAdapter.DEFAULT_BASE_URL`（百度千帆）。
/// 不含路径前缀，chat 端点为 `POST /rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}`，
/// access_token 端点为 `GET /oauth/2.0/token`。
pub const DEFAULT_ERNIE_BASE_URL: &str = "https://aip.baidubce.com";

/// Kimi 默认 Base URL
///
/// 对应 Python v1 `KimiAdapter.DEFAULT_BASE_URL`（月之暗面 Moonshot AI）。
/// 已含 `/v1` 前缀，chat 端点为 `POST /chat/completions`，models 端点为 `GET /models`。
pub const DEFAULT_KIMI_BASE_URL: &str = "https://api.moonshot.cn/v1";

/// MiniMax 默认 Base URL
///
/// 对应 Python v1 `MiniMaxAdapter.DEFAULT_BASE_URL`（稀宇科技）。
/// 已含 `/v1` 前缀，chat 端点为 `POST /chat/completions`。
/// 无可靠的 `/models` 端点，list_models 走硬编码列表。
pub const DEFAULT_MINIMAX_BASE_URL: &str = "https://api.minimaxi.com/v1";

// ==================== 能力集合构造 ====================

/// 通义千问 Qwen 支持的能力集合
///
/// 对齐 Python v1 `QwenAdapter.supported_capabilities` 的核心子集
/// （chat / chat_stream / vision）。Python 还声明了 AUDIO_TRANSCRIBE/AUDIO_SPEECH，
/// 但阶段 2a 范围仅核心三能力，audio 走 trait 默认实现返 UnsupportedCapability
/// （待阶段 2c audio_adapters 补齐）。
fn qwen_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// 智谱 AI Zhipu 支持的能力集合
///
/// 对齐 Python v1 `ZhipuAdapter.supported_capabilities = ["chat", "vision"]`。
/// Rust 额外声明 ChatStream（OpenAI 兼容地基支持流式，Python 老版也实现了 chat_stream）。
fn zhipu_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// 豆包 Doubao 支持的能力集合
///
/// 对齐 Python v1 `DoubaoAdapter.supported_capabilities` 的核心子集
/// （chat / chat_stream / vision），audio 能力待阶段 2c 补齐。
fn doubao_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// 文心一言 ERNIE 支持的能力集合
///
/// 对齐 Python v1 `ErnieAdapter.supported_capabilities = ["chat", "vision"]`。
/// Rust 额外声明 ChatStream（Python 老版实现了 chat_stream 流式）。
fn ernie_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// Kimi 支持的能力集合
///
/// 对齐 Python v1 `KimiAdapter.supported_capabilities = ["chat", "vision"]`。
/// Rust 额外声明 ChatStream（OpenAI 兼容地基支持流式，Python 老版也实现了 chat_stream）。
fn kimi_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

/// MiniMax 支持的能力集合
///
/// 对齐 Python v1 `MiniMaxAdapter.supported_capabilities` 的核心子集
/// （chat / chat_stream / vision），audio 能力待阶段 2c 补齐。
fn minimax_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps
}

// ==================== 通义千问 Qwen 适配器 ====================

/// 通义千问（Qwen）适配器
///
/// OpenAI 兼容协议，chat/chat_stream/list_models 全部委托给 [`OpenAiCompatAdapter`] 地基。
///
/// - Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`（DashScope OpenAI 兼容模式）
/// - Chat: `POST /chat/completions`
/// - Models: `GET /models`（实时拉取，对齐 Python 老版 `_parse_models_response`）
/// - 认证: Bearer Token（DashScope API Key）
/// - 文档: <https://help.aliyun.com/zh/dashscope/developer-reference/compatibility-of-openai-with-dashscope>
pub struct QwenAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl QwenAdapter {
    /// 创建通义千问适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_QWEN_BASE_URL,
            qwen_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 compat 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "qwen";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "通义千问";
}

#[async_trait]
impl Adapter for QwenAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
    }

    fn capabilities(&self) -> CapabilitySet {
        // 返回新集合，避免外部修改内部状态（不可变原则）
        self.compat.capabilities_set().clone()
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    async fn start(&mut self) -> Result<()> {
        // HTTP 客户端在 new() 时已构造，无需额外启动
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        // reqwest::Client 走 Drop 释放，无需显式关闭
        Ok(())
    }

    /// 文本对话：委托地基 `POST /chat/completions`
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.compat.chat(req).await
    }

    /// 流式文本对话：委托地基 `POST /chat/completions` (stream=true)
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.compat.chat_stream(req).await
    }

    /// 模型列表（实时拉取）：委托地基 `GET /models`
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }

    // image_generate / video_create / video_poll / embed / transcribe / speech / list_voices
    // 走 trait 默认实现，返 UnsupportedCapability，与 Python 老版行为一致
    // （Python 老版 image/video 显式抛 UnsupportedCapabilityError）。
}

// ==================== 智谱 AI Zhipu 适配器 ====================

/// 智谱 AI（GLM）适配器
///
/// OpenAI 兼容协议，chat/chat_stream/list_models 全部委托给 [`OpenAiCompatAdapter`] 地基。
///
/// - Base URL: `https://open.bigmodel.cn/api/paas/v4`
/// - Chat: `POST /chat/completions`
/// - Models: `GET /models`（实时拉取）
/// - 认证: Bearer Token（智谱 API Key）
/// - 文档: <https://open.bigmodel.cn/dev/api>
pub struct ZhipuAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl ZhipuAdapter {
    /// 创建智谱适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_ZHIPU_BASE_URL,
            zhipu_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 compat 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "zhipu";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "智谱 AI";
}

#[async_trait]
impl Adapter for ZhipuAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
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
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.compat.chat(req).await
    }

    /// 流式文本对话：委托地基 `POST /chat/completions` (stream=true)
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.compat.chat_stream(req).await
    }

    /// 模型列表（实时拉取）：委托地基 `GET /models`
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }

    // image_generate / video_create / video_poll / embed / transcribe / speech / list_voices
    // 走 trait 默认实现，返 UnsupportedCapability。
}

// ==================== 豆包 Doubao 适配器 ====================

/// 豆包（字节火山引擎方舟）适配器
///
/// OpenAI 兼容协议，chat/chat_stream/list_models 全部委托给 [`OpenAiCompatAdapter`] 地基。
///
/// - Base URL: `https://ark.cn-beijing.volces.com/api/v3`
/// - Chat: `POST /chat/completions`
/// - Models: `GET /models`（实时拉取）
/// - 认证: Bearer Token（火山引擎 API Key）
/// - 文档: <https://www.volcengine.com/docs/82379>
pub struct DoubaoAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl DoubaoAdapter {
    /// 创建豆包适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_DOUBAO_BASE_URL,
            doubao_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 compat 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "doubao";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "豆包";
}

#[async_trait]
impl Adapter for DoubaoAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
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
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.compat.chat(req).await
    }

    /// 流式文本对话：委托地基 `POST /chat/completions` (stream=true)
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.compat.chat_stream(req).await
    }

    /// 模型列表（实时拉取）：委托地基 `GET /models`
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }

    // image_generate / video_create / video_poll / embed / transcribe / speech / list_voices
    // 走 trait 默认实现，返 UnsupportedCapability。
}

// ==================== MiniMax 适配器 ====================

/// MiniMax（稀宇科技）适配器
///
/// OpenAI 兼容协议的 chat/chat_stream 委托给 [`OpenAiCompatAdapter`] 地基，
/// 但 list_models 走硬编码列表（MiniMax 无可靠的 `/models` 端点，对齐 Python 老版）。
///
/// - Base URL: `https://api.minimaxi.com/v1`
/// - Chat: `POST /chat/completions`（OpenAI 兼容接口）
/// - Models: 无标准 `/models` 端点，保留硬编码列表
/// - 认证: Bearer Token（MiniMax API Key），可选 `X-Group-Id` header
/// - 支持模型: abab 系列、MiniMax-Text-01、MiniMax-VL-01、MiniMax-M1、abab-asr/tts 等
/// - 文档: <https://platform.minimaxi.com/>
///
/// `group_id` 经 `config.extra["group_id"]` 传入，启动时注入到默认请求 header
/// （Python 老版 `start()` 中读取并设 `X-Group-Id`）。本实现因复用地基的 HttpClient，
/// group_id 注入依赖调用方在构造 config 时通过 `extra` 透传（地基不识别该 header，
/// 需要时由本适配器在 chat 前显式附加——当前实现与 Python 一致：仅当存在时附加到请求）。
pub struct MiniMaxAdapter {
    /// OpenAI 兼容地基（chat/chat_stream 委托给它，list_models 独立实现）
    compat: OpenAiCompatAdapter,
    /// MiniMax group_id（可选，经 `config.extra["group_id"]` 传入）
    group_id: Option<String>,
}

impl MiniMaxAdapter {
    /// 创建 MiniMax 适配器
    ///
    /// `group_id` 从 `config.extra["group_id"]` 提取（对齐 Python 老版
    /// `getattr(config, "group_id", None)`）。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let group_id = config
            .extra
            .get("group_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_MINIMAX_BASE_URL,
            minimax_capabilities(),
        )?;
        Ok(Self { compat, group_id })
    }

    /// 用显式 compat 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self {
            compat,
            group_id: None,
        }
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "minimax";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "MiniMax";

    /// API key（可能为空）
    fn api_key(&self) -> Option<&str> {
        self.compat.api_key()
    }

    /// base_url
    fn base_url(&self) -> &str {
        self.compat.base_url()
    }

    /// group_id（可选）
    pub fn group_id(&self) -> Option<&str> {
        self.group_id.as_deref()
    }

    /// 拼接完整 URL（base_url + 相对路径）
    fn url(&self, path: &str) -> String {
        let base = self.base_url().trim_end_matches('/');
        let path = path.trim_start_matches('/');
        format!("{base}/{path}")
    }

    /// 校验能力是否被支持
    fn ensure_capability(&self, cap: Capabilities) -> Result<()> {
        if self.compat.capabilities_set().contains(&cap) {
            Ok(())
        } else {
            Err(AibridgeError::UnsupportedCapability {
                capability: format!("{} (provider: {})", cap.as_str(), Self::PROVIDER_TYPE),
            })
        }
    }

    /// 构造带 group_id header 的 POST 请求 builder
    ///
    /// MiniMax 认证用 Bearer Token，group_id 存在时附加 `X-Group-Id` header
    /// （对齐 Python 老版 `start()` 中的 header 设置）。
    fn post_request(&self, url: &str, body: &Value) -> reqwest::RequestBuilder {
        let mut req = self
            .compat
            .http_inner()
            .post(url)
            .bearer_auth(self.api_key().unwrap_or(""))
            .json(body);
        if let Some(gid) = &self.group_id {
            req = req.header("X-Group-Id", gid);
        }
        req
    }

    /// 发送 chat/completions 请求并用 OpenAI 错误映射处理响应
    async fn post_chat(&self, body: &Value) -> Result<Value> {
        let url = self.url("chat/completions");
        let resp = self.post_request(&url, body).send().await.map_err(|e| {
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
            return Err(map_minimax_error(status_code, &body_text));
        }
        resp.json::<Value>().await.map_err(AibridgeError::from)
    }
}

/// 将 MiniMax API 错误响应映射为 AibridgeError
///
/// MiniMax 错误体结构与 OpenAI 类似（`{"error": {"message": "..."}}`），
/// 但额外可能用 `base_resp.status_msg` 字段（对齐 Python 老版 `_handle_error`）。
/// 优先提取这些字段的 message，再按 HTTP 状态码分类（复用 OpenAI 兼容错误分类）。
fn map_minimax_error(status: u16, body: &str) -> AibridgeError {
    // 尝试解析错误 message：优先 error.message，再 base_resp.status_msg，再顶层 message
    let message = if let Ok(v) = serde_json::from_str::<Value>(body) {
        let mut msg: Option<String> = v
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .map(str::to_owned);
        if msg.is_none() {
            msg = v
                .get("base_resp")
                .and_then(|b| b.get("status_msg"))
                .and_then(|m| m.as_str())
                .map(str::to_owned);
        }
        if msg.is_none() {
            msg = v.get("message").and_then(|m| m.as_str()).map(str::to_owned);
        }
        msg.unwrap_or_else(|| format!("HTTP {status}"))
    } else if body.trim().is_empty() {
        format!("HTTP {status}")
    } else {
        format!("HTTP {status}: {body}")
    };

    match status {
        401 | 403 => AibridgeError::Authentication { message },
        429 => {
            let retry_after = serde_json::from_str::<Value>(body).ok().and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("retry_after"))
                    .and_then(|r| r.as_f64())
                    .or_else(|| v.get("retry_after").and_then(|r| r.as_f64()))
            });
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
        s if (400..600).contains(&s) => AibridgeError::Api { status: s, message },
        s => AibridgeError::Api { status: s, message },
    }
}

#[async_trait]
impl Adapter for MiniMaxAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
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

    /// 文本对话
    ///
    /// 用地基构造标准 OpenAI 请求体，自建 POST 请求（附加 `X-Group-Id` header），
    /// 响应解析委托地基的 `parse_chat_completion`。
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.ensure_capability(Capabilities::Chat)?;
        let body = self.compat.build_chat_body(&req, false);
        let value = self.post_chat(&body).await?;
        self.compat.parse_chat_completion(&value, &req.model)
    }

    /// 流式文本对话
    ///
    /// 自建 POST 请求（附加 `X-Group-Id` header），SSE 解析复用地基的 chunk 解析逻辑。
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.ensure_capability(Capabilities::ChatStream)?;
        let body = self.compat.build_chat_body(&req, true);
        let url = self.url("chat/completions");

        let resp = self.post_request(&url, &body).send().await.map_err(|e| {
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
            return Err(map_minimax_error(status_code, &body_text));
        }

        let model = req.model.clone();
        // 按字节流读取，按行切分解析 SSE（与 openai_compat/azure 范例一致的按行切分逻辑）
        let byte_stream = resp
            .bytes_stream()
            .map_err(|e| e.to_string())
            .map(|r| r.map(|b| b.to_vec()));
        let lines_stream = MiniMaxLinesStream::new(byte_stream);

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
                // 结束标记
                if data.trim() == "[DONE]" {
                    return;
                }
                match serde_json::from_str::<Value>(data) {
                    Ok(v) => {
                        // 复用地基的 chunk 解析逻辑（parse_chunk 是关联函数，无需借 self）
                        match OpenAiCompatAdapter::parse_chunk(&v, &model) {
                            Ok(Some(chunk)) => yield Ok(chunk),
                            Ok(None) => continue,
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        }
                    }
                    // 单行 JSON 解析失败不致命，跳过（与 Python 老版一致）
                    Err(_) => continue,
                }
            }
        };

        Ok(stream.boxed())
    }

    /// 模型列表（硬编码）
    ///
    /// MiniMax 无可靠的 `/models` 端点，保留硬编码列表（对齐 Python 老版）。
    /// 按 `filter` 过滤模型类型。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = minimax_hardcoded_models();
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }

    // image_generate / video_create / video_poll / embed / transcribe / speech / list_voices
    // 走 trait 默认实现，返 UnsupportedCapability（Python 老版 image/video 显式抛错）。
}

/// MiniMax 硬编码模型列表
///
/// 对齐 Python v1 `MiniMaxAdapter.list_models` 的硬编码列表。
/// 含 chat 模型（abab 系列、MiniMax-Text-01/VL-01/M1）与 audio 模型（abab-asr/tts、speech-01）。
fn minimax_hardcoded_models() -> Vec<ModelInfo> {
    let chat_models = [
        (
            "abab6.5s-chat",
            "ABAB 6.5s",
            vec!["chat".to_string()],
            "MiniMax abab 6.5s 快速版本",
        ),
        (
            "abab6.5-chat",
            "ABAB 6.5",
            vec!["chat".to_string()],
            "MiniMax abab 6.5 标准版本",
        ),
        (
            "MiniMax-Text-01",
            "MiniMax Text 01",
            vec!["chat".to_string()],
            "MiniMax 万亿参数 MoE 模型",
        ),
        (
            "MiniMax-VL-01",
            "MiniMax VL 01",
            vec!["chat".to_string(), "vision".to_string()],
            "MiniMax 多模态视觉模型",
        ),
        (
            "MiniMax-M1",
            "MiniMax M1",
            vec!["chat".to_string(), "vision".to_string()],
            "MiniMax 最新推理模型",
        ),
    ];
    let audio_models = [
        (
            "abab-asr",
            "ABAB 语音识别",
            vec!["audio_transcribe".to_string()],
            "MiniMax 语音识别模型",
        ),
        (
            "abab-tts",
            "ABAB 语音合成",
            vec!["audio_speech".to_string()],
            "MiniMax 语音合成模型（多音色）",
        ),
        (
            "speech-01",
            "MiniMax Speech 01",
            vec!["audio_speech".to_string()],
            "MiniMax 高品质语音合成",
        ),
    ];

    let mut models: Vec<ModelInfo> = Vec::new();
    for (id, name, caps, desc) in chat_models {
        models.push(ModelInfo {
            id: id.to_string(),
            name: name.to_string(),
            model_type: ModelType::Chat,
            provider: MiniMaxAdapter::PROVIDER_TYPE.to_string(),
            capabilities: caps,
            max_tokens: None,
            supports_streaming: true,
            description: Some(desc.to_string()),
            created: None,
        });
    }
    for (id, name, caps, desc) in audio_models {
        models.push(ModelInfo {
            id: id.to_string(),
            name: name.to_string(),
            model_type: ModelType::Audio,
            provider: MiniMaxAdapter::PROVIDER_TYPE.to_string(),
            capabilities: caps,
            max_tokens: None,
            supports_streaming: false,
            description: Some(desc.to_string()),
            created: None,
        });
    }
    models
}

// ==================== MiniMax SSE 行流适配器 ====================

/// 将字节流按行切分的适配器（MiniMax 流式用）
///
/// 与 `openai_compat::LinesStream` 等价实现，独立实现避免引用其私有结构。
struct MiniMaxLinesStream<S> {
    inner: S,
    buffer: Vec<u8>,
}

impl<S> MiniMaxLinesStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
        }
    }
}

impl<S> futures::Stream for MiniMaxLinesStream<S>
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

/// Kimi（月之暗面 Moonshot AI）适配器
///
/// OpenAI 兼容协议，chat/chat_stream/list_models 全部委托给 [`OpenAiCompatAdapter`] 地基。
///
/// - Base URL: `https://api.moonshot.cn/v1`
/// - Chat: `POST /chat/completions`
/// - Models: `GET /models`（实时拉取）
/// - 认证: Bearer Token（Moonshot API Key）
/// - 特点: 支持超长上下文（128K/256K）、视觉理解
/// - 文档: <https://platform.moonshot.cn/docs/api/chat>
pub struct KimiAdapter {
    /// OpenAI 兼容地基（chat/chat_stream/list_models 委托给它）
    compat: OpenAiCompatAdapter,
}

impl KimiAdapter {
    /// 创建 Kimi 适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_KIMI_BASE_URL,
            kimi_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 compat 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "kimi";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "Kimi (月之暗面)";
}

#[async_trait]
impl Adapter for KimiAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
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
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.compat.chat(req).await
    }

    /// 流式文本对话：委托地基 `POST /chat/completions` (stream=true)
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.compat.chat_stream(req).await
    }

    /// 模型列表（实时拉取）：委托地基 `GET /models`
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }

    // image_generate / video_create / video_poll / embed / transcribe / speech / list_voices
    // 走 trait 默认实现，返 UnsupportedCapability。
}

// ==================== 文心一言 ERNIE 适配器（独立协议） ====================

/// 文心一言（百度 ERNIE）适配器
///
/// **独立协议实现**，仅复用地基的 `HttpClient` 与错误映射，chat/chat_stream/list_models
/// 全部独立实现（百度特有 access_token 流程 + 端点 + `messages`/`system` 分离的请求体 +
/// `result` 字段响应 + `is_end` 流式标记 + 硬编码模型列表）。
///
/// - Base URL: `https://aip.baidubce.com`
/// - Chat: `POST /rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}?access_token=...`
/// - 认证: access_token 查询参数（通过 API Key + Secret Key 换取）
///   - `config.api_key` 可直接是 access_token，或 `ak:sk` 格式（自动换取 access_token）
/// - access_token 端点: `GET /oauth/2.0/token?grant_type=client_credentials&client_id=ak&client_secret=sk`
/// - 请求体: `{"messages": [...], "system": "..."}`（system 单独字段，非 messages 数组）
/// - 响应: `{"result": "...", "usage": {...}, "is_truncated": false}`
/// - 流式: SSE，每行 `data: {"result": "...", "is_end": false}`
/// - Models: 无标准 `/models` 端点，保留硬编码列表（千帆 modellist 响应结构非 OpenAI 兼容）
/// - 文档: <https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html>
///
/// ## api_key 解析
///
/// 对齐 Python 老版 `ErnieAdapter.__init__`：
/// - `config.api_key` 含 `:` → `ak:sk` 格式，构造时拆分为 `api_key`(ak) + `secret_key`(sk)，
///   `start()` 时自动调 `/oauth/2.0/token` 换取 access_token
/// - 否则 → 直接作为 access_token 使用（用户已自行换取）
pub struct ErnieAdapter {
    /// OpenAI 兼容地基（仅复用其 HttpClient 与错误映射，不委托 chat 等方法）
    compat: OpenAiCompatAdapter,
    /// Secret Key（仅当 api_key 为 `ak:sk` 格式时非空）
    secret_key: String,
    /// access_token（start 时换取，或直接用 api_key）
    access_token: Option<String>,
}

impl ErnieAdapter {
    /// 创建文心一言适配器
    ///
    /// api_key 解析（对齐 Python 老版 `__init__`）：
    /// - 含 `:` → `ak:sk` 格式，拆分为 api_key(ak) + secret_key(sk)
    /// - 否则 → 直接作为 access_token 使用
    pub fn new(mut config: ProviderConfig) -> Result<Self> {
        // 先把 api_key 取出（避免借用冲突），再根据是否含 ':' 拆分 ak:sk
        let api_key = config.api_key.take();
        let (ak, secret_key) = match api_key.as_ref().and_then(|k| k.split_once(':')) {
            Some((ak, sk)) => (Some(ak.to_string()), sk.to_string()),
            None => (api_key, String::new()),
        };
        config.api_key = ak;

        // access_token 初始化：无 secret_key 时直接用 api_key
        let access_token = if secret_key.is_empty() {
            config.api_key.clone()
        } else {
            None
        };

        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_ERNIE_BASE_URL,
            ernie_capabilities(),
        )?;
        Ok(Self {
            compat,
            secret_key,
            access_token,
        })
    }

    /// 用显式 compat 构造（测试用，可注入 mockito 后端）
    ///
    /// `access_token` 默认为 `"test-token"`，避免测试触发 oauth 流程。
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self {
            compat,
            secret_key: String::new(),
            access_token: Some("test-token".to_string()),
        }
    }

    /// 用显式 compat + secret_key 构造（测试 oauth 流程用）
    #[cfg(test)]
    pub fn with_compat_and_secret(compat: OpenAiCompatAdapter, secret_key: String) -> Self {
        Self {
            compat,
            secret_key,
            access_token: None,
        }
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "ernie";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "文心一言";

    /// base_url
    fn base_url(&self) -> &str {
        self.compat.base_url()
    }

    /// 当前 access_token（已换取或直接用 api_key）
    pub fn access_token(&self) -> Option<&str> {
        self.access_token.as_deref()
    }

    /// 是否需要走 oauth 换取 access_token（即配置了 ak:sk）
    pub fn needs_oauth(&self) -> bool {
        !self.secret_key.is_empty()
    }

    /// 拼接完整 URL（base_url + 相对路径）
    fn url(&self, path: &str) -> String {
        let base = self.base_url().trim_end_matches('/');
        let path = path.trim_start_matches('/');
        format!("{base}/{path}")
    }

    /// 校验能力是否被支持
    fn ensure_capability(&self, cap: Capabilities) -> Result<()> {
        if self.compat.capabilities_set().contains(&cap) {
            Ok(())
        } else {
            Err(AibridgeError::UnsupportedCapability {
                capability: format!("{} (provider: {})", cap.as_str(), Self::PROVIDER_TYPE),
            })
        }
    }

    /// 换取 access_token（对齐 Python 老版 `_get_access_token`）
    ///
    /// `GET /oauth/2.0/token?grant_type=client_credentials&client_id={ak}&client_secret={sk}`
    /// 响应：`{"access_token": "...", ...}`
    async fn get_access_token(&self) -> Result<String> {
        let url = self.url("oauth/2.0/token");
        let ak = self.compat.api_key().unwrap_or("");
        let resp = self
            .compat
            .http_inner()
            .get(&url)
            .query(&[
                ("grant_type", "client_credentials"),
                ("client_id", ak),
                ("client_secret", &self.secret_key),
            ])
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
            return Err(map_ernie_error(status_code, &body_text));
        }
        let value: Value = resp.json().await.map_err(AibridgeError::from)?;
        let token = value
            .get("access_token")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .ok_or_else(|| AibridgeError::Api {
                status: 0,
                message: "ERNIE oauth 响应缺少 access_token 字段".to_string(),
            })?;
        Ok(token)
    }

    /// 转换统一消息为 ERNIE 格式，并提取 system prompt
    ///
    /// 对齐 Python v1 `ErnieAdapter._convert_messages`：
    /// - system 消息 → 提取为 `system` 字段（最后一条 system 覆盖前面的）
    /// - 其他消息 → `{"role": "user"/"assistant", "content": "..."}`
    ///
    /// 返回 `(messages, system)`，messages 不含 system 消息。
    fn convert_messages(messages: &[ChatMessage]) -> (Vec<Value>, Option<String>) {
        let mut ernie_messages: Vec<Value> = Vec::new();
        let mut system: Option<String> = None;

        for msg in messages {
            match msg {
                ChatMessage::System { content, .. } => {
                    system = Some(content.clone());
                }
                ChatMessage::User { content, .. } => {
                    let text = match content {
                        UserContent::Text(s) => s.clone(),
                        // 多模态：拼接所有 Text 部件，忽略图片（ERNIE 不支持图片输入）
                        UserContent::Parts(parts) => parts
                            .iter()
                            .filter_map(|p| match p {
                                crate::model::chat::ContentPart::Text { text } => {
                                    Some(text.clone())
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    };
                    ernie_messages.push(json!({ "role": "user", "content": text }));
                }
                ChatMessage::Assistant { content, .. } => {
                    let text = content.clone().unwrap_or_default();
                    ernie_messages.push(json!({ "role": "assistant", "content": text }));
                }
                ChatMessage::Tool { content, .. } => {
                    // 工具结果消息按 user 角色处理（ERNIE 无原生 tool 角色）
                    ernie_messages.push(json!({ "role": "user", "content": content }));
                }
            }
        }

        (ernie_messages, system)
    }

    /// 构造 ERNIE chat 请求体
    ///
    /// 对齐 Python v1 `ErnieAdapter.chat` 的 body 构造：
    /// - `messages`：非 system 消息列表
    /// - `system`：若有 system 消息则填入（单独字段）
    /// - `temperature` / `top_p`：从统一请求透传
    /// - `stream`：流式时置 true
    /// - `extra` 透传到顶层
    fn build_chat_body(req: &ChatRequest, stream: bool) -> Value {
        let (ernie_messages, system) = Self::convert_messages(&req.messages);

        let mut body = json!({ "messages": ernie_messages });
        if stream {
            body["stream"] = json!(true);
        }
        if let Some(s) = system {
            body["system"] = json!(s);
        }
        if let Some(t) = req.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(p) = req.top_p {
            body["top_p"] = json!(p);
        }
        // extra 透传到顶层
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        body
    }

    /// 解析 ERNIE chat 响应 → ChatCompletion
    ///
    /// ERNIE 响应格式（非标准 OpenAI choices 结构）：
    /// `{"result": "...", "usage": {"prompt_tokens":..., "completion_tokens":..., "total_tokens":...}, "is_truncated": false}`
    /// 对应 Python v1 `ErnieAdapter._parse_response`。
    fn parse_chat_completion(value: &Value, fallback_model: &str) -> Result<ChatCompletion> {
        let result = value
            .get("result")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // usage 解析（ERNIE usage 字段结构与 OpenAI 一致）
        let usage = value.get("usage").and_then(parse_ernie_usage);

        // is_truncated 为 true 时 finish_reason 为 "length"，否则 "stop"
        let is_truncated = value
            .get("is_truncated")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let finish_reason = if is_truncated { "length" } else { "stop" };

        Ok(ChatCompletion {
            id: value
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| util::generate_id("chatcmpl")),
            object: "chat.completion".to_string(),
            created: value
                .get("created")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(util::current_timestamp),
            model: value
                .get("model")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| fallback_model.to_string()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChoiceMessage {
                    role: "assistant".to_string(),
                    content: Some(result),
                    tool_calls: None,
                },
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage,
            service_tier: None,
            system_fingerprint: None,
        })
    }

    /// 解析单个 ERNIE 流式 chunk
    ///
    /// ERNIE 流式事件格式：`{"result": "...", "is_end": false, ...}`
    /// - `result`：增量文本（delta.content = result）
    /// - `is_end`：是否结束（true 时 finish_reason = "stop"）
    ///
    /// 对应 Python v1 `ErnieAdapter._parse_chunk`。
    fn parse_chunk(value: &Value, fallback_model: &str) -> Option<ChatCompletionChunk> {
        let result = value
            .get("result")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let is_end = value
            .get("is_end")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let finish_reason = if is_end {
            Some("stop".to_string())
        } else {
            None
        };

        // usage 可能出现在结束块
        let usage = value.get("usage").and_then(parse_ernie_usage);

        Some(ChatCompletionChunk {
            id: value
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| util::generate_id("chatcmpl")),
            object: "chat.completion.chunk".to_string(),
            created: util::current_timestamp(),
            model: value
                .get("model")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| fallback_model.to_string()),
            choices: vec![ChatCompletionDelta {
                index: 0,
                delta: DeltaMessage {
                    role: Some("assistant".to_string()),
                    content: Some(result),
                    tool_calls: None,
                },
                finish_reason,
            }],
            usage,
        })
    }

    /// 构造 chat 端点 URL（含 access_token query）
    ///
    /// `POST /rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}?access_token=...`
    fn chat_url(&self, model: &str) -> String {
        let base = self.base_url().trim_end_matches('/');
        format!(
            "{base}/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}?access_token={}",
            self.access_token.as_deref().unwrap_or("")
        )
    }

    /// 发送 chat 请求并用 ERNIE 错误映射处理响应
    async fn post_ernie_json(&self, url: &str, body: &Value) -> Result<Value> {
        let resp = self
            .compat
            .http_inner()
            .post(url)
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
            return Err(map_ernie_error(status_code, &body_text));
        }
        resp.json::<Value>().await.map_err(AibridgeError::from)
    }
}

/// 解析 ERNIE usage 统计
///
/// ERNIE usage 格式与 OpenAI 一致：`{"prompt_tokens": N, "completion_tokens": M, "total_tokens": T}`
fn parse_ernie_usage(v: &Value) -> Option<crate::model::chat::ChatUsage> {
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

/// 将 ERNIE API 错误响应映射为 AibridgeError
///
/// 对齐 Python v1 `ErnieAdapter._handle_error`：
/// - 401/403 → Authentication
/// - 429 → RateLimit
/// - 其他 → Api
/// - 错误 message 优先取 `error_msg`，再 `message`，再 `error`
fn map_ernie_error(status: u16, body: &str) -> AibridgeError {
    let message = if let Ok(v) = serde_json::from_str::<Value>(body) {
        let mut msg: Option<String> = v
            .get("error_msg")
            .and_then(|m| m.as_str())
            .map(str::to_owned);
        if msg.is_none() {
            msg = v.get("message").and_then(|m| m.as_str()).map(str::to_owned);
        }
        if msg.is_none() {
            // error 可能是字符串也可能是对象
            msg = v
                .get("error")
                .and_then(|e| e.as_str())
                .map(str::to_owned)
                .or_else(|| {
                    v.get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .map(str::to_owned)
                });
        }
        msg.unwrap_or_else(|| format!("HTTP {status}"))
    } else if body.trim().is_empty() {
        format!("HTTP {status}")
    } else {
        format!("HTTP {status}: {body}")
    };

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
        s if (400..600).contains(&s) => AibridgeError::Api { status: s, message },
        s => AibridgeError::Api { status: s, message },
    }
}

#[async_trait]
impl Adapter for ErnieAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
    }

    fn capabilities(&self) -> CapabilitySet {
        self.compat.capabilities_set().clone()
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    /// 启动适配器
    ///
    /// 若配置了 ak:sk（`needs_oauth()` 为 true），自动调 `/oauth/2.0/token` 换取 access_token。
    /// 对齐 Python 老版 `start()` 行为。
    async fn start(&mut self) -> Result<()> {
        if self.needs_oauth() && self.access_token.is_none() {
            let token = self.get_access_token().await?;
            self.access_token = Some(token);
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    /// 文本对话（ERNIE 特有协议：POST /rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}）
    ///
    /// 请求体用 `messages` + `system` 分离结构，响应 `result` 字段。
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.ensure_capability(Capabilities::Chat)?;
        let body = Self::build_chat_body(&req, false);
        let url = self.chat_url(&req.model);
        let value = self.post_ernie_json(&url, &body).await?;
        Self::parse_chat_completion(&value, &req.model)
    }

    /// 流式文本对话（ERNIE 特有协议：POST .../chat/{model}?access_token=... stream=true）
    ///
    /// SSE 格式，每行 `data: {"result": "...", "is_end": false}`。
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.ensure_capability(Capabilities::ChatStream)?;
        let body = Self::build_chat_body(&req, true);
        let url = self.chat_url(&req.model);

        let resp = self
            .compat
            .http_inner()
            .post(&url)
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
            return Err(map_ernie_error(status_code, &body_text));
        }

        let model = req.model.clone();
        // 按字节流读取，按行切分解析 SSE（与 openai_compat/azure/minimax 范例一致的按行切分逻辑）
        let byte_stream = resp
            .bytes_stream()
            .map_err(|e| e.to_string())
            .map(|r| r.map(|b| b.to_vec()));
        let lines_stream = ErnieLinesStream::new(byte_stream);

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
                // ERNIE 流式无 [DONE] 标记，靠 is_end 字段判断结束
                match serde_json::from_str::<Value>(data) {
                    Ok(v) => match Self::parse_chunk(&v, &model) {
                        Some(chunk) => yield Ok(chunk),
                        None => continue,
                    },
                    // 单行 JSON 解析失败不致命，跳过（与 Python 老版一致）
                    Err(_) => continue,
                }
            }
        };

        Ok(stream.boxed())
    }

    /// 模型列表（硬编码）
    ///
    /// 对齐 Python v1 `ErnieAdapter.list_models`：百度千帆 modellist 响应结构非 OpenAI 兼容
    /// （`result.model_list`，字段为 code/name），无法复用基类解析，故保留硬编码列表。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = ernie_hardcoded_models();
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }

    // image_generate / video_create / video_poll / embed / transcribe / speech / list_voices
    // 走 trait 默认实现，返 UnsupportedCapability（Python 老版 image/video 显式抛错）。
}

/// ERNIE 硬编码模型列表
///
/// 对齐 Python v1 `ErnieAdapter.list_models` 的硬编码列表。
/// 含 ERNIE 4.0 / 3.5 / Lite 等 chat 模型。
fn ernie_hardcoded_models() -> Vec<ModelInfo> {
    let models = [
        (
            "completions_pro",
            "ERNIE 4.0",
            vec!["chat".to_string(), "vision".to_string()],
            "文心一言 4.0",
        ),
        (
            "completions",
            "ERNIE 3.5",
            vec!["chat".to_string()],
            "文心一言 3.5",
        ),
        (
            "ernie-lite-8k",
            "ERNIE Lite",
            vec!["chat".to_string()],
            "文心一言轻量版",
        ),
    ];
    models
        .iter()
        .map(|(id, name, caps, desc)| ModelInfo {
            id: id.to_string(),
            name: name.to_string(),
            model_type: ModelType::Chat,
            provider: ErnieAdapter::PROVIDER_TYPE.to_string(),
            capabilities: caps.clone(),
            max_tokens: None,
            supports_streaming: true,
            description: Some(desc.to_string()),
            created: None,
        })
        .collect()
}

// ==================== ERNIE SSE 行流适配器 ====================

/// 将字节流按行切分的适配器（ERNIE 流式用）
///
/// 与 `openai_compat::LinesStream` 等价实现，独立实现避免引用其私有结构。
struct ErnieLinesStream<S> {
    inner: S,
    buffer: Vec<u8>,
}

impl<S> ErnieLinesStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
        }
    }
}

impl<S> futures::Stream for ErnieLinesStream<S>
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientOptions;
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

    /// 标准 OpenAI chat/completions 响应体（复用于多个兼容适配器测试）
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

    /// 构造 QwenAdapter（指向 mockito server）
    fn make_qwen(server: &Server) -> QwenAdapter {
        let compat = make_compat(server, "qwen", "通义千问", qwen_capabilities());
        QwenAdapter::with_compat(compat)
    }

    /// 构造 ZhipuAdapter（指向 mockito server）
    fn make_zhipu(server: &Server) -> ZhipuAdapter {
        let compat = make_compat(server, "zhipu", "智谱 AI", zhipu_capabilities());
        ZhipuAdapter::with_compat(compat)
    }

    /// 构造 DoubaoAdapter（指向 mockito server）
    fn make_doubao(server: &Server) -> DoubaoAdapter {
        let compat = make_compat(server, "doubao", "豆包", doubao_capabilities());
        DoubaoAdapter::with_compat(compat)
    }

    /// 构造 KimiAdapter（指向 mockito server）
    fn make_kimi(server: &Server) -> KimiAdapter {
        let compat = make_compat(server, "kimi", "Kimi (月之暗面)", kimi_capabilities());
        KimiAdapter::with_compat(compat)
    }

    /// 构造 MiniMaxAdapter（指向 mockito server）
    fn make_minimax(server: &Server) -> MiniMaxAdapter {
        let compat = make_compat(server, "minimax", "MiniMax", minimax_capabilities());
        MiniMaxAdapter::with_compat(compat)
    }

    /// 构造 ErnieAdapter（指向 mockito server，access_token 预填避免 oauth）
    fn make_ernie(server: &Server) -> ErnieAdapter {
        let compat = make_compat(server, "ernie", "文心一言", ernie_capabilities());
        ErnieAdapter::with_compat(compat)
    }

    // ==================== Qwen 元信息与能力 ====================

    #[tokio::test]
    async fn qwen_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_qwen(&server);
        assert_eq!(adapter.provider_type(), "qwen");
        assert_eq!(adapter.provider_name(), "通义千问");
    }

    #[tokio::test]
    async fn qwen_requires_api_key() {
        let server = Server::new_async().await;
        let adapter = make_qwen(&server);
        assert!(adapter.requires_api_key());
    }

    #[tokio::test]
    async fn qwen_capabilities_include_chat_and_vision() {
        let server = Server::new_async().await;
        let adapter = make_qwen(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::Vision));
        // audio 能力在阶段 2a 不声明（待 2c 补齐）
        assert!(!caps.contains(&Capabilities::AudioSpeech));
        assert!(!caps.contains(&Capabilities::ImageGenerate));
    }

    #[tokio::test]
    async fn qwen_chat_success() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_qwen(&server);
        let req = ChatRequest::builder("qwen-turbo", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .max_tokens(100)
            .build();
        let resp = adapter.chat(req).await.expect("chat 应成功");
        assert_eq!(resp.id, "chatcmpl-1");
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 7);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn qwen_chat_stream_parses_sse() {
        let mut server = Server::new_async().await;
        let sse = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen-turbo\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"你好\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen-turbo\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"世界\"},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(sse)
            .create_async()
            .await;
        let adapter = make_qwen(&server);
        let req = ChatRequest::builder("qwen-turbo", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.expect("stream 应建立");
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(chunks.len(), 2);
        let mut content = String::new();
        content.push_str(chunks[0].choices[0].delta.content.as_deref().unwrap_or(""));
        content.push_str(chunks[1].choices[0].delta.content.as_deref().unwrap_or(""));
        assert_eq!(content, "你好世界");
    }

    #[tokio::test]
    async fn qwen_list_models_success() {
        let mut server = Server::new_async().await;
        let body = json!({
            "data": [
                {"id": "qwen-turbo", "object": "model", "created": 1, "owned_by": "dashscope"},
                {"id": "qwen-vl-max", "object": "model", "created": 1, "owned_by": "dashscope"}
            ]
        });
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;
        let adapter = make_qwen(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "qwen-turbo");
        assert_eq!(models[0].provider, "qwen");
    }

    #[tokio::test]
    async fn qwen_chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(json!({"error": {"message": "Invalid API key"}}).to_string())
            .create_async()
            .await;
        let adapter = make_qwen(&server);
        let req = ChatRequest::builder("qwen-turbo", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn qwen_image_generate_returns_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_qwen(&server);
        let req = ImageRequest::builder("qwen-vl-max", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ==================== Zhipu 元信息与能力 ====================

    #[tokio::test]
    async fn zhipu_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_zhipu(&server);
        assert_eq!(adapter.provider_type(), "zhipu");
        assert_eq!(adapter.provider_name(), "智谱 AI");
    }

    #[tokio::test]
    async fn zhipu_capabilities_include_chat_and_vision() {
        let server = Server::new_async().await;
        let adapter = make_zhipu(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::Vision));
    }

    #[tokio::test]
    async fn zhipu_chat_success() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_zhipu(&server);
        let req = ChatRequest::builder("glm-4", vec![ChatMessage::user("hi")]).build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
    }

    #[tokio::test]
    async fn zhipu_list_models_success() {
        let mut server = Server::new_async().await;
        let body = json!({
            "data": [{"id": "glm-4", "object": "model", "created": 1, "owned_by": "zhipu"}]
        });
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;
        let adapter = make_zhipu(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "glm-4");
        assert_eq!(models[0].provider, "zhipu");
    }

    #[tokio::test]
    async fn zhipu_chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;
        let adapter = make_zhipu(&server);
        let req = ChatRequest::builder("glm-4", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    // ==================== Doubao 元信息与能力 ====================

    #[tokio::test]
    async fn doubao_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_doubao(&server);
        assert_eq!(adapter.provider_type(), "doubao");
        assert_eq!(adapter.provider_name(), "豆包");
    }

    #[tokio::test]
    async fn doubao_capabilities_include_chat_and_vision() {
        let server = Server::new_async().await;
        let adapter = make_doubao(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::Vision));
    }

    #[tokio::test]
    async fn doubao_chat_success() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_doubao(&server);
        let req = ChatRequest::builder("doubao-pro", vec![ChatMessage::user("hi")]).build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
    }

    #[tokio::test]
    async fn doubao_list_models_success() {
        let mut server = Server::new_async().await;
        let body = json!({
            "data": [{"id": "doubao-pro", "object": "model", "created": 1, "owned_by": "volcengine"}]
        });
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;
        let adapter = make_doubao(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "doubao-pro");
        assert_eq!(models[0].provider, "doubao");
    }

    #[tokio::test]
    async fn doubao_chat_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(500)
            .with_body(json!({"error": {"message": "internal"}}).to_string())
            .create_async()
            .await;
        let adapter = make_doubao(&server);
        let req = ChatRequest::builder("doubao-pro", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 500),
            _ => panic!("应为 Api"),
        }
    }

    // ==================== Kimi 元信息与能力 ====================

    #[tokio::test]
    async fn kimi_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_kimi(&server);
        assert_eq!(adapter.provider_type(), "kimi");
        assert_eq!(adapter.provider_name(), "Kimi (月之暗面)");
    }

    #[tokio::test]
    async fn kimi_capabilities_include_chat_and_vision() {
        let server = Server::new_async().await;
        let adapter = make_kimi(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::Vision));
    }

    #[tokio::test]
    async fn kimi_chat_success() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_kimi(&server);
        let req = ChatRequest::builder("moonshot-v1-8k", vec![ChatMessage::user("hi")]).build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
    }

    #[tokio::test]
    async fn kimi_list_models_success() {
        let mut server = Server::new_async().await;
        let body = json!({
            "data": [{"id": "moonshot-v1-8k", "object": "model", "created": 1, "owned_by": "moonshot"}]
        });
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;
        let adapter = make_kimi(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "moonshot-v1-8k");
        assert_eq!(models[0].provider, "kimi");
    }

    #[tokio::test]
    async fn kimi_video_create_returns_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_kimi(&server);
        let req = VideoRequest::builder("moonshot-v1-8k", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ==================== MiniMax 元信息与能力 ====================

    #[tokio::test]
    async fn minimax_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_minimax(&server);
        assert_eq!(adapter.provider_type(), "minimax");
        assert_eq!(adapter.provider_name(), "MiniMax");
    }

    #[tokio::test]
    async fn minimax_requires_api_key() {
        let server = Server::new_async().await;
        let adapter = make_minimax(&server);
        assert!(adapter.requires_api_key());
    }

    #[tokio::test]
    async fn minimax_capabilities_include_chat_and_vision() {
        let server = Server::new_async().await;
        let adapter = make_minimax(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::Vision));
    }

    #[tokio::test]
    async fn minimax_chat_success() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let adapter = make_minimax(&server);
        let req = ChatRequest::builder("abab6.5-chat", vec![ChatMessage::user("hi")])
            .temperature(0.5)
            .max_tokens(50)
            .build();
        let resp = adapter.chat(req).await.expect("chat 应成功");
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn minimax_chat_stream_parses_sse() {
        let mut server = Server::new_async().await;
        let sse = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"abab6.5-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"abab6.5-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(sse)
            .create_async()
            .await;
        let adapter = make_minimax(&server);
        let req = ChatRequest::builder("abab6.5-chat", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.expect("stream 应建立");
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(chunks.len(), 2);
        let mut content = String::new();
        content.push_str(chunks[0].choices[0].delta.content.as_deref().unwrap_or(""));
        content.push_str(chunks[1].choices[0].delta.content.as_deref().unwrap_or(""));
        assert_eq!(content, "hi!");
    }

    #[tokio::test]
    async fn minimax_list_models_hardcoded() {
        // MiniMax 无 /models 端点，list_models 返回硬编码列表，不应发请求
        let server = Server::new_async().await;
        let adapter = make_minimax(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert!(!models.is_empty());
        // 验证含 chat 与 audio 两种类型
        assert!(models.iter().any(|m| m.model_type == ModelType::Chat));
        assert!(models.iter().any(|m| m.model_type == ModelType::Audio));
        // 验证 provider 字段
        assert!(models.iter().all(|m| m.provider == "minimax"));
    }

    #[tokio::test]
    async fn minimax_list_models_filter_by_type() {
        let server = Server::new_async().await;
        let adapter = make_minimax(&server);
        let chat_models = adapter.list_models(Some(ModelType::Chat)).await.unwrap();
        assert!(chat_models.iter().all(|m| m.model_type == ModelType::Chat));
        assert!(chat_models.len() >= 5);
    }

    #[tokio::test]
    async fn minimax_chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;
        let adapter = make_minimax(&server);
        let req = ChatRequest::builder("abab6.5-chat", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn minimax_chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;
        let adapter = make_minimax(&server);
        let req = ChatRequest::builder("abab6.5-chat", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn minimax_chat_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(500)
            .with_body(json!({"base_resp": {"status_msg": "internal error"}}).to_string())
            .create_async()
            .await;
        let adapter = make_minimax(&server);
        let req = ChatRequest::builder("abab6.5-chat", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, message } => {
                assert_eq!(status, 500);
                // base_resp.status_msg 应被提取为 message
                assert!(message.contains("internal error"));
            }
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn minimax_group_id_extracted_from_config() {
        // group_id 经 config.extra["group_id"] 传入
        let server = Server::new_async().await;
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .extra("group_id", "grp-123")
            .build();
        let config = ProviderConfig::from_options("minimax", opts);
        let adapter = MiniMaxAdapter::new(config).unwrap();
        assert_eq!(adapter.group_id(), Some("grp-123"));
    }

    #[tokio::test]
    async fn minimax_group_id_none_when_not_configured() {
        let server = Server::new_async().await;
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .build();
        let config = ProviderConfig::from_options("minimax", opts);
        let adapter = MiniMaxAdapter::new(config).unwrap();
        assert_eq!(adapter.group_id(), None);
    }

    #[tokio::test]
    async fn minimax_group_id_sent_as_header_when_configured() {
        // 配置 group_id 时，chat 请求应携带 X-Group-Id header
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("x-group-id", "grp-123")
            .with_status(200)
            .with_body(openai_chat_body().to_string())
            .create_async()
            .await;
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .extra("group_id", "grp-123")
            .build();
        let config = ProviderConfig::from_options("minimax", opts);
        let adapter = MiniMaxAdapter::new(config).unwrap();
        let req = ChatRequest::builder("abab6.5-chat", vec![ChatMessage::user("hi")]).build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn minimax_image_generate_returns_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_minimax(&server);
        let req = ImageRequest::builder("abab6.5-chat", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ==================== Ernie 元信息与能力 ====================

    #[tokio::test]
    async fn ernie_provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_ernie(&server);
        assert_eq!(adapter.provider_type(), "ernie");
        assert_eq!(adapter.provider_name(), "文心一言");
    }

    #[tokio::test]
    async fn ernie_requires_api_key() {
        let server = Server::new_async().await;
        let adapter = make_ernie(&server);
        assert!(adapter.requires_api_key());
    }

    #[tokio::test]
    async fn ernie_capabilities_include_chat_and_vision() {
        let server = Server::new_async().await;
        let adapter = make_ernie(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::Vision));
    }

    #[tokio::test]
    async fn ernie_chat_success_parses_result_field() {
        // ERNIE 响应用 result 字段（非 OpenAI choices 结构）
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "ernie-1",
            "object": "chat.completion",
            "created": 1700000000,
            "result": "你好，我是文心一言",
            "usage": {"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11},
            "is_truncated": false
        });
        let mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(
                    r"^/rpc/2\.0/ai_custom/v1/wenxinworkshop/chat/completions_pro".to_string(),
                ),
            )
            .match_query(mockito::Matcher::UrlEncoded(
                "access_token".into(),
                "test-token".into(),
            ))
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;
        let adapter = make_ernie(&server);
        let req = ChatRequest::builder("completions_pro", vec![ChatMessage::user("hi")]).build();
        let resp = adapter.chat(req).await.expect("chat 应成功");
        assert_eq!(resp.id, "ernie-1");
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("你好，我是文心一言")
        );
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 11);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn ernie_chat_truncated_returns_length_finish_reason() {
        // is_truncated=true 时 finish_reason 应为 "length"
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "ernie-2",
            "result": "被截断的回复",
            "is_truncated": true
        });
        server
            .mock(
                "POST",
                mockito::Matcher::Regex(
                    r"^/rpc/2\.0/ai_custom/v1/wenxinworkshop/chat/completions".to_string(),
                ),
            )
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;
        let adapter = make_ernie(&server);
        let req = ChatRequest::builder("completions", vec![ChatMessage::user("hi")]).build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("length"));
    }

    #[tokio::test]
    async fn ernie_chat_extracts_system_to_separate_field() {
        // system 消息应被提取到 system 字段（非 messages 数组）
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(
                    r"^/rpc/2\.0/ai_custom/v1/wenxinworkshop/chat/".to_string(),
                ),
            )
            .match_query(mockito::Matcher::Any)
            .match_body(mockito::Matcher::PartialJson(json!({
                "messages": [{"role": "user", "content": "hi"}],
                "system": "你是助手"
            })))
            .with_status(200)
            .with_body(json!({"result": "ok"}).to_string())
            .create_async()
            .await;
        let adapter = make_ernie(&server);
        let req = ChatRequest::builder(
            "completions_pro",
            vec![ChatMessage::system("你是助手"), ChatMessage::user("hi")],
        )
        .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn ernie_chat_passes_temperature_and_top_p() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(
                    r"^/rpc/2\.0/ai_custom/v1/wenxinworkshop/chat/".to_string(),
                ),
            )
            .match_query(mockito::Matcher::Any)
            .match_body(mockito::Matcher::PartialJson(json!({
                "temperature": 0.8,
                "top_p": 0.9
            })))
            .with_status(200)
            .with_body(json!({"result": "ok"}).to_string())
            .create_async()
            .await;
        let adapter = make_ernie(&server);
        let req = ChatRequest::builder("completions_pro", vec![ChatMessage::user("hi")])
            .temperature(0.8)
            .top_p(0.9)
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn ernie_chat_stream_parses_sse_with_is_end() {
        // ERNIE 流式：result 增量 + is_end 结束标记（无 [DONE]）
        let mut server = Server::new_async().await;
        let sse = "data: {\"id\":\"e1\",\"result\":\"你好\",\"is_end\":false}\n\
                   data: {\"id\":\"e1\",\"result\":\"世界\",\"is_end\":false}\n\
                   data: {\"id\":\"e1\",\"result\":\"\",\"is_end\":true,\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":4,\"total_tokens\":6}}\n";
        server
            .mock(
                "POST",
                mockito::Matcher::Regex(
                    r"^/rpc/2\.0/ai_custom/v1/wenxinworkshop/chat/".to_string(),
                ),
            )
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(sse)
            .create_async()
            .await;
        let adapter = make_ernie(&server);
        let req = ChatRequest::builder("completions_pro", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.expect("stream 应建立");
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(chunks.len(), 3);
        // 前两块为增量文本，finish_reason 为 None
        assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("你好"));
        assert!(chunks[0].choices[0].finish_reason.is_none());
        assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("世界"));
        // 第三块 is_end=true，finish_reason 为 "stop"，并携带 usage
        assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunks[2].usage.as_ref().unwrap().total_tokens, 6);
    }

    #[tokio::test]
    async fn ernie_list_models_hardcoded() {
        // ERNIE 无标准 /models 端点，list_models 返回硬编码列表
        let server = Server::new_async().await;
        let adapter = make_ernie(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert!(!models.is_empty());
        assert!(models.iter().all(|m| m.provider == "ernie"));
        // 验证含 ERNIE 4.0 / 3.5
        assert!(models.iter().any(|m| m.id == "completions_pro"));
        assert!(models.iter().any(|m| m.id == "completions"));
    }

    #[tokio::test]
    async fn ernie_chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock(
                "POST",
                mockito::Matcher::Regex(
                    r"^/rpc/2\.0/ai_custom/v1/wenxinworkshop/chat/".to_string(),
                ),
            )
            .match_query(mockito::Matcher::Any)
            .with_status(401)
            .with_body(json!({"error_msg": "Invalid access_token"}).to_string())
            .create_async()
            .await;
        let adapter = make_ernie(&server);
        let req = ChatRequest::builder("completions_pro", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Authentication { message } => {
                // error_msg 字段应被提取
                assert!(message.contains("Invalid access_token"));
            }
            _ => panic!("应为 Authentication"),
        }
    }

    #[tokio::test]
    async fn ernie_chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock(
                "POST",
                mockito::Matcher::Regex(
                    r"^/rpc/2\.0/ai_custom/v1/wenxinworkshop/chat/".to_string(),
                ),
            )
            .match_query(mockito::Matcher::Any)
            .with_status(429)
            .with_body(json!({"error_msg": "qps limit"}).to_string())
            .create_async()
            .await;
        let adapter = make_ernie(&server);
        let req = ChatRequest::builder("completions_pro", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn ernie_image_generate_returns_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_ernie(&server);
        let req = ImageRequest::builder("ernie-vilg", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn ernie_video_create_returns_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_ernie(&server);
        let req = VideoRequest::builder("ernie", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ==================== Ernie api_key 解析与 oauth 流程 ====================

    #[test]
    fn ernie_new_with_plain_api_key_uses_as_access_token() {
        // 无冒号的 api_key 直接作为 access_token，不触发 oauth
        let opts = ClientOptions::builder()
            .api_key("plain-token")
            .base_url("https://aip.baidubce.com")
            .build();
        let config = ProviderConfig::from_options("ernie", opts);
        let adapter = ErnieAdapter::new(config).unwrap();
        assert!(!adapter.needs_oauth());
        assert_eq!(adapter.access_token(), Some("plain-token"));
    }

    #[test]
    fn ernie_new_with_ak_sk_format_marks_needs_oauth() {
        // ak:sk 格式应拆分，标记需要 oauth，access_token 初始为 None
        let opts = ClientOptions::builder()
            .api_key("my-ak:my-sk")
            .base_url("https://aip.baidubce.com")
            .build();
        let config = ProviderConfig::from_options("ernie", opts);
        let adapter = ErnieAdapter::new(config).unwrap();
        assert!(adapter.needs_oauth());
        assert_eq!(adapter.access_token(), None);
    }

    #[tokio::test]
    async fn ernie_start_fetches_access_token_when_ak_sk_configured() {
        // 配置 ak:sk 时，start() 应调 /oauth/2.0/token 换取 access_token
        let mut server = Server::new_async().await;
        let mock = server
            .mock("GET", "/oauth/2.0/token")
            .match_query(mockito::Matcher::AllOf(vec![
                mockito::Matcher::UrlEncoded("grant_type".into(), "client_credentials".into()),
                mockito::Matcher::UrlEncoded("client_id".into(), "my-ak".into()),
                mockito::Matcher::UrlEncoded("client_secret".into(), "my-sk".into()),
            ]))
            .with_status(200)
            .with_body(json!({"access_token": "fetched-token", "expires_in": 2592000}).to_string())
            .create_async()
            .await;

        let opts = ClientOptions::builder()
            .api_key("my-ak:my-sk")
            .base_url(server.url())
            .build();
        let config = ProviderConfig::from_options("ernie", opts);
        let mut adapter = ErnieAdapter::new(config).unwrap();
        assert!(adapter.needs_oauth());
        assert_eq!(adapter.access_token(), None);

        adapter.start().await.expect("start 应成功");
        assert_eq!(adapter.access_token(), Some("fetched-token"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn ernie_start_skips_oauth_when_plain_token() {
        // 无冒号的 api_key，start() 不应发 oauth 请求
        let server = Server::new_async().await;
        let opts = ClientOptions::builder()
            .api_key("plain-token")
            .base_url(server.url())
            .build();
        let config = ProviderConfig::from_options("ernie", opts);
        let mut adapter = ErnieAdapter::new(config).unwrap();
        adapter.start().await.unwrap();
        assert_eq!(adapter.access_token(), Some("plain-token"));
    }

    #[tokio::test]
    async fn ernie_oauth_error_propagates() {
        // oauth 端点返回错误时，start() 应传播错误
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/oauth/2.0/token")
            .match_query(mockito::Matcher::Any)
            .with_status(401)
            .with_body(json!({"error": "invalid client"}).to_string())
            .create_async()
            .await;
        let opts = ClientOptions::builder()
            .api_key("bad-ak:bad-sk")
            .base_url(server.url())
            .build();
        let config = ProviderConfig::from_options("ernie", opts);
        let mut adapter = ErnieAdapter::new(config).unwrap();
        let err = adapter.start().await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    // ==================== 错误映射单元测试 ====================

    #[test]
    fn map_ernie_error_extracts_error_msg_field() {
        let body = json!({"error_msg": "some error"}).to_string();
        let err = map_ernie_error(500, &body);
        match err {
            AibridgeError::Api { message, .. } => assert_eq!(message, "some error"),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_ernie_error_falls_back_to_message_field() {
        let body = json!({"message": "fallback msg"}).to_string();
        let err = map_ernie_error(400, &body);
        match err {
            AibridgeError::Validation { message, .. } => assert_eq!(message, "fallback msg"),
            _ => panic!("应为 Validation"),
        }
    }

    #[test]
    fn map_ernie_error_no_json_falls_back_to_http_status() {
        let err = map_ernie_error(502, "Bad Gateway");
        match err {
            AibridgeError::Api { message, .. } => assert!(message.contains("502")),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_minimax_error_extracts_base_resp_status_msg() {
        // MiniMax 特有：base_resp.status_msg 字段
        let body = json!({"base_resp": {"status_msg": "minimax error"}}).to_string();
        let err = map_minimax_error(500, &body);
        match err {
            AibridgeError::Api { message, .. } => assert_eq!(message, "minimax error"),
            _ => panic!("应为 Api"),
        }
    }

    #[test]
    fn map_minimax_error_extracts_error_message_first() {
        // 优先 error.message，而非 base_resp.status_msg
        let body = json!({
            "error": {"message": "primary"},
            "base_resp": {"status_msg": "secondary"}
        })
        .to_string();
        let err = map_minimax_error(500, &body);
        match err {
            AibridgeError::Api { message, .. } => assert_eq!(message, "primary"),
            _ => panic!("应为 Api"),
        }
    }

    // ==================== Ernie 消息转换单元测试 ====================

    #[test]
    fn ernie_convert_messages_extracts_system() {
        let messages = vec![
            ChatMessage::system("sys-prompt"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
        ];
        let (msgs, system) = ErnieAdapter::convert_messages(&messages);
        assert_eq!(system.as_deref(), Some("sys-prompt"));
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "hello");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "hi");
    }

    #[test]
    fn ernie_convert_messages_no_system() {
        let messages = vec![ChatMessage::user("hello")];
        let (msgs, system) = ErnieAdapter::convert_messages(&messages);
        assert!(system.is_none());
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    // ==================== start / close 是空操作（OpenAI 兼容族） ====================

    #[tokio::test]
    async fn qwen_start_and_close_are_noops() {
        let server = Server::new_async().await;
        let mut adapter = make_qwen(&server);
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    #[tokio::test]
    async fn minimax_start_and_close_are_noops() {
        let server = Server::new_async().await;
        let mut adapter = make_minimax(&server);
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }
}
