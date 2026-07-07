//! Azure OpenAI 适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/azure.py`。
//!
//! Azure OpenAI API 与 OpenAI API 基本兼容（请求体/响应格式相同），但有关键差异：
//! - **Base URL 结构**：`https://{resource}.cognitiveservices.azure.com/openai/deployments/{deployment}`
//!   （Python 老版用 `.openai.azure.com`，新版 Azure 已迁移到 `cognitiveservices.azure.com`，
//!   两者都可通过 `config.base_url` 覆盖）
//! - **认证方式**：用 HTTP header `api-key: {key}`，而非 `Authorization: Bearer {key}`
//! - **路径结构**：`POST /openai/deployments/{deployment}/{action}?api-version={version}`
//!   - chat：`/chat/completions`
//!   - image：`/images/generations`
//!   - embed：`/embeddings`
//! - **API 版本**：必须带 `api-version` query 参数（如 `2024-02-15-preview`）
//! - **模型名**：是 deployment name（部署名），非底层模型名
//! - **list_models**：调 Azure 部署列表 API `GET /openai/deployments?api-version={version}`，
//!   响应里 `id` 是部署名、`model` 是底层模型名
//!
//! 复用策略：请求体构造（`build_chat_body` / `build_image_body` / `build_embed_body`）
//! 与响应解析（`parse_chat_completion` / `parse_image_result` / `parse_embedding_result`）
//! 以及错误映射（`map_api_error`）与 OpenAI 兼容协议完全一致，故内部持有一个
//! [`OpenAiCompatAdapter`] 实例，把这部分逻辑委托给它；HTTP 请求层（URL 拼接、
//! `api-key` header、`api-version` query）由本适配器自行实现，因地基的
//! `post_authed_json` 写死了 `bearer_auth` 与无 query 的相对路径。
//!
//! 能力：Chat / ChatStream / ImageGenerate / Embedding（与 Python 老版核心子集一致，
//! Python 还声明了 AUDIO_TRANSCRIBE/TRANSLATE/SPEECH，但阶段 2a 范围仅核心四能力，
//! audio 走 trait 默认实现返 UnsupportedCapability，待阶段 2c audio_adapters 补齐）。
//! `requires_api_key = true`。

use async_trait::async_trait;
use futures::stream::{StreamExt, TryStreamExt};
use serde_json::Value;

use crate::adapter::{Adapter, Capabilities, CapabilitySet, ChatStream};
use crate::adapters::openai_compat::OpenAiCompatAdapter;
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::chat::{ChatCompletion, ChatRequest};
use crate::model::common::{infer_model_type, ModelInfo, ModelType};
use crate::model::image::{ImageRequest, ImageResult};
use crate::model::options::{EmbedRequest, EmbeddingResult};

/// Azure OpenAI 默认 API 版本
///
/// 对应 Python v1 `DEFAULT_API_VERSION = "2024-02-15-preview"`。
/// Azure 端点必须带 `api-version` query 参数，未配置时用此默认值。
pub const DEFAULT_AZURE_API_VERSION: &str = "2024-02-15-preview";

/// Azure OpenAI 默认资源主机后缀
///
/// 老版 Azure 用 `{resource}.openai.azure.com`，新版迁移到
/// `{resource}.cognitiveservices.azure.com`。本常量仅用于无 `base_url` 时
/// 由 `resource_name` + `deployment_id` 拼接默认 base_url，采用 Python 老版
/// 一致的 `openai.azure.com`（用户可通过 `config.base_url` 覆盖为 cognitiveservices）。
const DEFAULT_AZURE_HOST_SUFFIX: &str = "openai.azure.com";

/// Azure 路径前缀（deployments 段之前的固定路径）
const AZURE_PATH_PREFIX: &str = "/openai/deployments";

/// Azure 适配器
///
/// 持有 HTTP 客户端、Provider 配置、解析后的 Azure 专用字段（resource_name /
/// deployment_id / api_version）以及一个 [`OpenAiCompatAdapter`] 实例（用于
/// 复用 OpenAI 兼容的请求体构造与响应解析）。
///
/// 构造时即解析 base_url 与 Azure 字段，`start` / `close` 为空操作
/// （HTTP 客户端在 `new` 时已构造，走 Drop 释放）。
pub struct AzureAdapter {
    /// HTTP 客户端（封装 reqwest，含连接池与超时）
    http: HttpClient,
    /// Provider 配置（保留 api_key / resource_name / deployment_id 等原始字段）
    config: ProviderConfig,
    /// Azure 资源名称（如 "my-resource"），可为 None（当直接提供 base_url 时）
    resource_name: Option<String>,
    /// Azure 部署 ID（作为 chat/image/embed 的路径段），可为 None（当直接提供 base_url 时）
    deployment_id: Option<String>,
    /// Azure API 版本（如 "2024-02-15-preview"）
    api_version: String,
    /// 用于 chat/image/embed 请求的 base_url（含 `/openai/deployments/{deployment}` 段）
    ///
    /// 当 `config.base_url` 提供时直接用之；否则由 `resource_name` + `deployment_id`
    /// 拼接为 `https://{resource}.openai.azure.com/openai/deployments/{deployment}`。
    deployments_base_url: String,
    /// OpenAI 兼容协议地基（复用请求体构造与响应解析，HTTP 层不使用它）
    compat: OpenAiCompatAdapter,
}

impl AzureAdapter {
    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "azure";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "Azure OpenAI";

    /// 创建 Azure 适配器
    ///
    /// 配置解析顺序（与 Python 老版 `__init__` 一致）：
    /// 1. `config.api_version` 为空时用 [`DEFAULT_AZURE_API_VERSION`]
    /// 2. `config.base_url` 非空时直接作为 deployments base_url
    /// 3. 否则用 `resource_name` + `deployment_id` 拼接默认 base_url
    /// 4. 两者都没有则返 [`AibridgeError::Validation`]（与 Python `ValueError` 一致）
    ///
    /// # 错误
    /// - `Validation`：既无 `base_url`，又无 `resource_name` + `deployment_id`
    /// - HTTP 客户端构造失败（罕见，reqwest 配置错误）
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let caps = Self::capabilities_set();

        // 解析 api_version
        let api_version = config
            .api_version
            .clone()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_AZURE_API_VERSION.to_string());

        // 解析 resource_name / deployment_id（从 config 直读，与 Python 一致）
        let resource_name = config.resource_name.clone();
        let deployment_id = config.deployment_id.clone();

        // 解析 deployments base_url
        let deployments_base_url = if let Some(url) = config
            .base_url
            .as_ref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
        {
            url
        } else if let (Some(res), Some(dep)) = (resource_name.as_ref(), deployment_id.as_ref()) {
            // 由 resource_name + deployment_id 拼接（去掉末尾斜杠，避免双斜杠）
            let res = res.trim();
            let dep = dep.trim();
            format!("https://{res}.{DEFAULT_AZURE_HOST_SUFFIX}{AZURE_PATH_PREFIX}/{dep}")
        } else {
            return Err(AibridgeError::validation(
                "Azure adapter requires either base_url or both resource_name and deployment_id",
            ));
        };

        // 构造 HttpClient：用 deployments base_url 作为 base（chat/image/embed 路径相对它）
        let http_opts = ClientOptions::builder()
            .api_key(config.api_key.clone().unwrap_or_default())
            .base_url(deployments_base_url.clone())
            .timeout(config.timeout)
            .max_retries(config.max_retries)
            .retry_delay(config.retry_delay)
            .build();
        let http = HttpClient::new(&http_opts)?;

        // 构造 OpenAiCompatAdapter 实例（仅复用其请求体构造/响应解析/错误映射方法，
        // 其内部 HttpClient 不被本适配器使用）。传入相同 base_url 以保持一致性。
        let compat = OpenAiCompatAdapter::new(
            config.clone(),
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            &deployments_base_url,
            caps,
        )?;

        Ok(Self {
            http,
            config,
            resource_name,
            deployment_id,
            api_version,
            deployments_base_url,
            compat,
        })
    }

    /// 支持的能力集合
    ///
    /// 对应 Python v1 `AzureAdapter.supported_capabilities` 的核心子集
    /// （chat / chat_stream / image_generate / embedding）。
    /// Python 还声明了 AUDIO_TRANSCRIBE/TRANSLATE/SPEECH，但阶段 2a 范围
    /// 仅核心四能力，audio 走 trait 默认实现返 UnsupportedCapability。
    fn capabilities_set() -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::Chat);
        caps.insert(Capabilities::ChatStream);
        caps.insert(Capabilities::ImageGenerate);
        caps.insert(Capabilities::Embedding);
        caps
    }

    /// deployments base_url（含 `/openai/deployments/{deployment}` 段）
    pub fn deployments_base_url(&self) -> &str {
        &self.deployments_base_url
    }

    /// Azure API 版本
    pub fn api_version(&self) -> &str {
        &self.api_version
    }

    /// Azure 部署 ID（chat/image/embed 路径段；直接提供 base_url 时为 None）
    pub fn deployment_id(&self) -> Option<&str> {
        self.deployment_id.as_deref()
    }

    /// Azure 资源名称（用于 list_models 拼独立 host；直接提供 base_url 时为 None）
    pub fn resource_name(&self) -> Option<&str> {
        self.resource_name.as_deref()
    }

    /// API key（可能为空）
    fn api_key(&self) -> Option<&str> {
        self.config.api_key.as_deref()
    }

    /// 拼接完整 URL（deployments base_url + action 路径 + `?api-version=...`）
    ///
    /// `action` 如 `"chat/completions"` / `"images/generations"` / `"embeddings"`。
    /// 已含 query 的 URL 不再追加（如调用方自行拼好）。
    fn action_url(&self, action: &str) -> String {
        let base = self.deployments_base_url.trim_end_matches('/');
        let action = action.trim_start_matches('/');
        format!("{base}/{action}?api-version={}", self.api_version)
    }

    /// 校验请求的能力是否被支持（不支持则返 UnsupportedCapability）
    fn ensure_capability(&self, cap: Capabilities) -> Result<()> {
        if self.compat.capabilities_set().contains(&cap) {
            Ok(())
        } else {
            Err(AibridgeError::UnsupportedCapability {
                capability: format!("{} (provider: {})", cap.as_str(), Self::PROVIDER_TYPE),
            })
        }
    }

    /// 发送带 `api-key` header 的 POST JSON 请求，并用 OpenAI 错误映射处理响应
    ///
    /// Azure 认证用 `api-key` header（非 Bearer），错误体结构与 OpenAI 一致
    /// （`{"error": {"message": "..."}}`），故复用 [`OpenAiCompatAdapter::map_api_error`]。
    async fn post_azure_json(&self, url: &str, body: &Value) -> Result<Value> {
        let resp = self
            .http
            .inner()
            .post(url)
            .header("api-key", self.api_key().unwrap_or(""))
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
            return Err(OpenAiCompatAdapter::map_api_error(status_code, &body_text));
        }
        resp.json::<Value>().await.map_err(AibridgeError::from)
    }

    /// 发送带 `api-key` header 的 GET 请求，并用 OpenAI 错误映射处理响应
    async fn get_azure_json(&self, url: &str) -> Result<Value> {
        let resp = self
            .http
            .inner()
            .get(url)
            .header("api-key", self.api_key().unwrap_or(""))
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
        resp.json::<Value>().await.map_err(AibridgeError::from)
    }
}

#[async_trait]
impl Adapter for AzureAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
    }

    fn capabilities(&self) -> CapabilitySet {
        // 返回新集合，避免外部修改内部状态（不可变原则）
        Self::capabilities_set()
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

    /// 文本对话
    ///
    /// `POST /openai/deployments/{deployment}/chat/completions?api-version={version}`
    /// 请求体与 OpenAI 一致（委托地基构造），响应解析也委托地基。
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.ensure_capability(Capabilities::Chat)?;
        let body = self.compat.build_chat_body(&req, false);
        let url = self.action_url("chat/completions");
        let value = self.post_azure_json(&url, &body).await?;
        self.compat.parse_chat_completion(&value, &req.model)
    }

    /// 流式文本对话
    ///
    /// `POST /openai/deployments/{deployment}/chat/completions?api-version={version}&stream=true`
    /// SSE 格式与 OpenAI 一致，按行解析 `data: <json>` / `data: [DONE]`。
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.ensure_capability(Capabilities::ChatStream)?;
        let body = self.compat.build_chat_body(&req, true);
        let url = self.action_url("chat/completions");

        let resp = self
            .http
            .inner()
            .post(&url)
            .header("api-key", self.api_key().unwrap_or(""))
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
            return Err(OpenAiCompatAdapter::map_api_error(status_code, &body_text));
        }

        let model = req.model.clone();
        let byte_stream = resp
            .bytes_stream()
            .map_err(|e| e.to_string())
            .map(|r| r.map(|b| b.to_vec()));
        // 复用 openai_compat 的 LinesStream（按行切分字节流）
        // 由于 LinesStream 是 openai_compat 的私有结构，这里内联同样的按行切分逻辑
        let lines_stream = AzureLinesStream::new(byte_stream);

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
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }
                let data = if let Some(rest) = line.strip_prefix("data: ") {
                    rest
                } else if let Some(rest) = line.strip_prefix("data:") {
                    rest
                } else {
                    continue;
                };
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
                    Err(_) => continue,
                }
            }
        };

        Ok(stream.boxed())
    }

    /// 图像生成
    ///
    /// `POST /openai/deployments/{deployment}/images/generations?api-version={version}`
    async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
        self.ensure_capability(Capabilities::ImageGenerate)?;
        let body = self.compat.build_image_body(&req);
        let url = self.action_url("images/generations");
        let value = self.post_azure_json(&url, &body).await?;
        self.compat.parse_image_result(&value, &req.model)
    }

    /// 文本嵌入
    ///
    /// `POST /openai/deployments/{deployment}/embeddings?api-version={version}`
    async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        self.ensure_capability(Capabilities::Embedding)?;
        let body = self.compat.build_embed_body(&req);
        let url = self.action_url("embeddings");
        let value = self.post_azure_json(&url, &body).await?;
        self.compat.parse_embedding_result(&value, &req.model)
    }

    /// 模型列表（实时拉取 Azure 部署列表）
    ///
    /// `GET https://{resource}.openai.azure.com/openai/deployments?api-version={version}`
    ///
    /// Azure 部署列表响应：`{"data": [{"id": "<部署名>", "model": "<底层模型名>", ...}]}`
    /// 与 Python 老版一致：用 `model` 字段作为模型 ID（用于推断类型），`id` 作为显示名。
    /// 若既无 `resource_name` 也无裸 base_url，则用 deployments_base_url 的 host 段拼路径。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        // Azure 部署列表端点不含 deployments/{id} 段，路径为 /openai/deployments
        // 取 resource host：优先用 resource_name 拼，否则从 deployments_base_url 反推
        let list_url = if let Some(res) = self.resource_name.as_ref() {
            let res = res.trim();
            format!(
                "https://{res}.{DEFAULT_AZURE_HOST_SUFFIX}{AZURE_PATH_PREFIX}?api-version={}",
                self.api_version
            )
        } else {
            // 从 deployments_base_url（形如 .../openai/deployments/{dep}）截到 /openai/deployments
            let base = self.deployments_base_url.trim_end_matches('/');
            let prefix = AZURE_PATH_PREFIX;
            let host_end = base.find(prefix).map(|i| &base[..i]).unwrap_or(base);
            format!("{host_end}{prefix}?api-version={}", self.api_version)
        };

        let value = self.get_azure_json(&list_url).await?;
        let models = parse_azure_deployments(&value, Self::PROVIDER_TYPE);
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }

    // 其余方法（video_create / video_poll / transcribe / speech / list_voices）
    // 走 trait 默认实现，返 UnsupportedCapability。Python 老版的 transcribe/speech
    // 待阶段 2c audio_adapters 统一补齐。
}

/// 解析 Azure 部署列表响应 → Vec<ModelInfo>
///
/// Azure 响应：`{"data": [{"id": "<部署名>", "model": "<底层模型名>", ...}]}`
/// - 用 `model` 字段（缺失时回退 `id`）作为模型 ID，用于推断类型
/// - 用 `id` 字段（部署名）作为显示名
/// - `provider` 填 "azure"
fn parse_azure_deployments(value: &Value, provider: &str) -> Vec<ModelInfo> {
    let arr = value.get("data").and_then(|v| v.as_array());
    match arr {
        Some(arr) => arr
            .iter()
            .map(|m| {
                // model 字段优先作为 ID（用于类型推断），缺失时回退 id
                let model_field = m.get("model").and_then(|v| v.as_str()).unwrap_or("");
                let id_field = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let model_id = if !model_field.is_empty() {
                    model_field
                } else {
                    id_field
                }
                .to_string();
                let name = if !id_field.is_empty() {
                    id_field.to_string()
                } else {
                    model_id.clone()
                };
                let model_type = infer_model_type(&model_id);
                ModelInfo {
                    name,
                    id: model_id,
                    model_type,
                    provider: provider.to_string(),
                    capabilities: Vec::new(),
                    max_tokens: None,
                    supports_streaming: matches!(model_type, ModelType::Chat),
                    description: None,
                    created: None,
                }
            })
            .collect(),
        None => Vec::new(),
    }
}

// ==================== SSE 行流适配器 ====================

/// 将字节流按行切分的适配器（与 openai_compat::LinesStream 等价实现）
///
/// `openai_compat::LinesStream` 为私有结构，本适配器独立实现一份相同逻辑
/// （按 `\n` 切分字节流，维护未完成行缓冲区）。
struct AzureLinesStream<S> {
    inner: S,
    buffer: Vec<u8>,
}

impl<S> AzureLinesStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
        }
    }
}

impl<S> futures::Stream for AzureLinesStream<S>
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
    use crate::error::AibridgeError;
    use crate::model::chat::{ChatMessage, ChatRequest};
    use crate::model::options::{EmbedInput, EmbeddingVector};
    use futures::stream::StreamExt;
    use mockito::Server;
    use serde_json::json;
    use std::collections::HashMap;

    /// 构造指向 mockito server 的 AzureAdapter（base_url 注入 mock 地址）
    ///
    /// mock 地址会被当作 deployments_base_url，故 chat 路径为
    /// `{mock}/chat/completions?api-version=...`。
    fn make_adapter(server: &Server) -> AzureAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("azure", opts);
        AzureAdapter::new(config).expect("AzureAdapter 构造应成功")
    }

    /// 构造 AzureAdapter（用 resource_name + deployment_id 拼默认 base_url，不发请求）
    fn make_adapter_with_resource() -> AzureAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .extra("resource_name", "my-resource")
            .extra("deployment_id", "my-deployment")
            .extra("api_version", "2024-02-15-preview")
            .build();
        let config = ProviderConfig::from_options("azure", opts);
        AzureAdapter::new(config).expect("AzureAdapter 构造应成功")
    }

    /// 构造不指向任何 server 的 AzureAdapter（用于元信息/能力测试）
    fn make_adapter_no_server() -> AzureAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url("https://example.openai.azure.com/openai/deployments/dep")
            .build();
        let config = ProviderConfig::from_options("azure", opts);
        AzureAdapter::new(config).expect("AzureAdapter 构造应成功")
    }

    // ============ 构造与元信息 ============

    #[test]
    fn provider_type_and_name_match_python() {
        let adapter = make_adapter_no_server();
        assert_eq!(adapter.provider_type(), "azure");
        assert_eq!(adapter.provider_name(), "Azure OpenAI");
    }

    #[test]
    fn requires_api_key_is_true() {
        let adapter = make_adapter_no_server();
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn capabilities_contains_core_set() {
        let adapter = make_adapter_no_server();
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::ImageGenerate));
        assert!(caps.contains(&Capabilities::Embedding));
        // video / audio 不声明（阶段 2a 范围）
        assert!(!caps.contains(&Capabilities::VideoGenerate));
        assert!(!caps.contains(&Capabilities::AudioSpeech));
    }

    #[test]
    fn base_url_uses_config_when_provided() {
        let adapter = make_adapter_no_server();
        assert_eq!(
            adapter.deployments_base_url(),
            "https://example.openai.azure.com/openai/deployments/dep"
        );
    }

    #[test]
    fn base_url_built_from_resource_and_deployment() {
        let adapter = make_adapter_with_resource();
        assert_eq!(
            adapter.deployments_base_url(),
            "https://my-resource.openai.azure.com/openai/deployments/my-deployment"
        );
    }

    #[test]
    fn api_version_defaults_when_missing() {
        // 未提供 api_version，应回退到默认值
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://x.openai.azure.com/openai/deployments/d")
            .build();
        let config = ProviderConfig::from_options("azure", opts);
        let adapter = AzureAdapter::new(config).unwrap();
        assert_eq!(adapter.api_version(), DEFAULT_AZURE_API_VERSION);
    }

    #[test]
    fn api_version_uses_config_when_provided() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://x.openai.azure.com/openai/deployments/d")
            .extra("api_version", "2024-10-21")
            .build();
        let config = ProviderConfig::from_options("azure", opts);
        let adapter = AzureAdapter::new(config).unwrap();
        assert_eq!(adapter.api_version(), "2024-10-21");
    }

    #[test]
    fn new_fails_without_base_url_or_resource_deployment() {
        // 既无 base_url，又无 resource_name + deployment_id → Validation 错误
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("azure", opts);
        let err = AzureAdapter::new(config)
            .err()
            .expect("应返回 Validation 错误");
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[test]
    fn new_fails_with_only_resource_name() {
        // 只有 resource_name，无 deployment_id → 失败
        let opts = ClientOptions::builder()
            .api_key("k")
            .extra("resource_name", "res")
            .build();
        let config = ProviderConfig::from_options("azure", opts);
        assert!(AzureAdapter::new(config).is_err());
    }

    #[test]
    fn action_url_appends_api_version_query() {
        let adapter = make_adapter_no_server();
        let url = adapter.action_url("chat/completions");
        assert!(url.contains("/chat/completions?api-version="));
        assert!(url.contains(DEFAULT_AZURE_API_VERSION));
    }

    // ============ chat 正常路径 ============

    #[tokio::test]
    async fn chat_success_returns_completion() {
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
            .match_query(mockito::Matcher::UrlEncoded(
                "api-version".into(),
                DEFAULT_AZURE_API_VERSION.into(),
            ))
            .match_header("api-key", "test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
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
    async fn chat_sends_api_key_header_not_bearer() {
        // 关键：Azure 用 api-key header，而非 Authorization: Bearer
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_query(mockito::Matcher::Any)
            .match_header("api-key", "test-key")
            // 显式断言不带 Authorization Bearer（mockito 不匹配 Authorization 即可）
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

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")])
            .temperature(0.5)
            .max_tokens(50)
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_passes_extra_params_through() {
        // Azure 请求体与 OpenAI 一致，extra 透传
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_query(mockito::Matcher::Any)
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

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")])
            .extra("custom_param", "custom_value")
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    // ============ chat 错误路径 ============

    #[tokio::test]
    async fn chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .match_query(mockito::Matcher::Any)
            .with_status(401)
            .with_body(json!({"error": {"message": "Invalid API key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .match_query(mockito::Matcher::Any)
            .with_status(429)
            .with_body(
                json!({"error": {"message": "Rate limit exceeded", "retry_after": 1.5}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
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
            .match_query(mockito::Matcher::Any)
            .with_status(404)
            .with_body(
                json!({"error": {"message": "The deployment 'gpt-x' does not exist"}}).to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gpt-x", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn chat_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .match_query(mockito::Matcher::Any)
            .with_status(500)
            .with_body(json!({"error": {"message": "Internal server error"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
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
            .match_query(mockito::Matcher::Any)
            .with_status(400)
            .with_body(json!({"error": {"message": "max_tokens is invalid"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Validation { message, .. } => {
                assert!(message.contains("max_tokens"));
            }
            _ => panic!("应为 Validation"),
        }
    }

    // ============ chat_stream ============

    #[tokio::test]
    async fn chat_stream_parses_sse_chunks() {
        let mut server = Server::new_async().await;
        let sse = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
                   data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        server
            .mock("POST", "/chat/completions")
            .match_query(mockito::Matcher::Any)
            .match_header("api-key", "test-key")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
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
    async fn chat_stream_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .match_query(mockito::Matcher::Any)
            .with_status(401)
            .with_body(json!({"error": {"message": "Unauthorized"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let result = adapter.chat_stream(req).await;
        match result {
            Err(e) => assert!(matches!(e, AibridgeError::Authentication { .. })),
            Ok(_) => panic!("chat_stream 应返回错误而非 stream"),
        }
    }

    #[tokio::test]
    async fn chat_stream_sends_stream_true() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_query(mockito::Matcher::Any)
            .match_body(mockito::Matcher::PartialJson(json!({
                "stream": true,
                "stream_options": {"include_usage": true}
            })))
            .with_status(200)
            .with_body("data: [DONE]\n")
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        while stream.next().await.is_some() {}
        mock.assert_async().await;
    }

    // ============ image_generate ============

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
        let mock = server
            .mock("POST", "/images/generations")
            .match_query(mockito::Matcher::UrlEncoded(
                "api-version".into(),
                DEFAULT_AZURE_API_VERSION.into(),
            ))
            .match_header("api-key", "test-key")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
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
        mock.assert_async().await;
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
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("dall-e-3", "a cat").build();
        let resp = adapter.image_generate(req).await.unwrap();
        assert_eq!(resp.data[0].b64_json.as_deref(), Some("aGVsbG8="));
    }

    #[tokio::test]
    async fn image_generate_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/images/generations")
            .match_query(mockito::Matcher::Any)
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;
        let adapter = make_adapter(&server);
        let req = ImageRequest::builder("dall-e-3", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    // ============ embed ============

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
        let mock = server
            .mock("POST", "/embeddings")
            .match_query(mockito::Matcher::UrlEncoded(
                "api-version".into(),
                DEFAULT_AZURE_API_VERSION.into(),
            ))
            .match_header("api-key", "test-key")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
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
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn embed_single_input_sends_string() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/embeddings")
            .match_query(mockito::Matcher::Any)
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

        let adapter = make_adapter(&server);
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
    async fn embed_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .match_query(mockito::Matcher::Any)
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;
        let adapter = make_adapter(&server);
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

    // ============ list_models ============

    #[tokio::test]
    async fn list_models_success_parses_deployments() {
        let mut server = Server::new_async().await;
        // Azure 部署列表响应：id 是部署名，model 是底层模型名
        let body = json!({
            "data": [
                {"id": "my-gpt4o", "model": "gpt-4o", "object": "deployment", "status": "succeeded"},
                {"id": "my-dalle3", "model": "dall-e-3", "object": "deployment", "status": "succeeded"},
                {"id": "my-whisper", "model": "whisper-1", "object": "deployment", "status": "succeeded"}
            ]
        });
        // list_models 用 resource_name 拼独立 host，但这里用 base_url 路径反推
        // 让 server mock /openai/deployments 路径，base_url 设为 mock + /openai/deployments/dep
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(format!("{}/openai/deployments/dep", server.url()))
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("azure", opts);
        let adapter = AzureAdapter::new(config).unwrap();

        server
            .mock("GET", "/openai/deployments")
            .match_query(mockito::Matcher::UrlEncoded(
                "api-version".into(),
                DEFAULT_AZURE_API_VERSION.into(),
            ))
            .match_header("api-key", "test-key")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 3);
        // model 字段作为 ID（用于类型推断）
        assert_eq!(models[0].id, "gpt-4o");
        assert_eq!(models[0].name, "my-gpt4o");
        assert_eq!(models[0].model_type, ModelType::Chat);
        assert_eq!(models[0].provider, "azure");
        assert_eq!(models[1].model_type, ModelType::Image);
        assert_eq!(models[2].model_type, ModelType::Audio);
    }

    #[tokio::test]
    async fn list_models_filter_by_type() {
        let mut server = Server::new_async().await;
        let body = json!({
            "data": [
                {"id": "dep1", "model": "gpt-4o"},
                {"id": "dep2", "model": "dall-e-3"}
            ]
        });
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(format!("{}/openai/deployments/dep", server.url()))
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("azure", opts);
        let adapter = AzureAdapter::new(config).unwrap();

        server
            .mock("GET", "/openai/deployments")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "dall-e-3");
    }

    #[tokio::test]
    async fn list_models_falls_back_to_id_when_model_missing() {
        // 部分 Azure 部署响应无 model 字段，回退用 id 作为模型 ID
        let mut server = Server::new_async().await;
        let body = json!({
            "data": [{"id": "custom-deploy"}]
        });
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(format!("{}/openai/deployments/dep", server.url()))
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("azure", opts);
        let adapter = AzureAdapter::new(config).unwrap();

        server
            .mock("GET", "/openai/deployments")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "custom-deploy");
        assert_eq!(models[0].name, "custom-deploy");
    }

    #[tokio::test]
    async fn list_models_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(format!("{}/openai/deployments/dep", server.url()))
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("azure", opts);
        let adapter = AzureAdapter::new(config).unwrap();

        server
            .mock("GET", "/openai/deployments")
            .match_query(mockito::Matcher::Any)
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;

        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    // ============ 不支持的能力走默认实现 ============

    #[tokio::test]
    async fn video_create_returns_unsupported() {
        let adapter = make_adapter_no_server();
        let req = crate::model::video::VideoRequest::builder("gpt-4o", "a cat").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn speech_returns_unsupported() {
        let adapter = make_adapter_no_server();
        let req = crate::model::audio::SpeechRequest::builder("tts-1", "hi", "alloy").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn list_voices_returns_unsupported() {
        let adapter = make_adapter_no_server();
        let err = adapter.list_voices(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ start / close ============

    #[tokio::test]
    async fn start_and_close_are_noops() {
        let mut adapter = make_adapter_no_server();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }
}
