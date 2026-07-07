//! Agnes AI 适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/agnes.py`。
//!
//! Agnes AI 是 OpenAI 兼容协议：chat / chat_stream / image_generate / embed /
//! list_models 全部复用 `OpenAiCompatAdapter` 地基实现；视频生成（Video V2.0）
//! 是 Agnes 特有协议（POST /videos 创建任务 + GET /videos/{task_id} 轮询），
//! 本模块独立实现 `video_create` / `video_poll`。
//!
//! 设计要点（与设计文档第 10 节一致）：
//! - 组合（而非继承）`OpenAiCompatAdapter`：AgnesAdapter 内部持有一个 compat 实例，
//!   实现 `Adapter` trait 时把 chat/image/embed/list_models 委托给 compat
//! - 视频协议独立实现：参考 Python `agnes.py` 的 `video_create` / `video_poll`
//! - `list_models` 实时拉取（v1.1.0 特性）：直接复用 compat 的实现，无需 override
//! - `requires_api_key = true`：Agnes 需要 API Key 认证
//!
//! 能力声明（对齐 Python `agnes.py` 的 `supported_capabilities`）：
//! CHAT / CHAT_STREAM / IMAGE_GENERATE / VIDEO（VIDEO_TEXT2VIDEO + VIDEO_IMAGE2VIDEO）
//! / EMBEDDING，外加 VISION / TOOL_CALL / JSON_MODE / REASONING 等 chat 子能力。

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::adapter::{Adapter, Capabilities, CapabilitySet, ChatStream};
use crate::adapters::openai_compat::OpenAiCompatAdapter;
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::chat::{ChatCompletion, ChatRequest};
use crate::model::common::{ModelInfo, ModelType, TaskStatus, VoiceInfo};
use crate::model::image::{ImageRequest, ImageResult};
use crate::model::options::{EmbedRequest, EmbeddingResult};
use crate::model::video::{VideoRequest, VideoStatus, VideoTask};
use crate::util;

/// Agnes AI 默认 Base URL
///
/// 对应 Python v1 `DEFAULT_BASE_URL`。
pub const DEFAULT_AGNES_BASE_URL: &str = "https://api.agnes.ai/v1";

/// Agnes AI 适配器
///
/// 组合 `OpenAiCompatAdapter` 复用 OpenAI 兼容能力，独立实现 Agnes 视频协议。
///
/// 构造时传入 `ProviderConfig`，内部据此创建 `OpenAiCompatAdapter` 实例（用于
/// chat / image / embed / list_models 委托）与独立的 `HttpClient`（用于 Agnes
/// 特有的视频端点）。视频端点不委托 compat，是因为 Agnes 视频协议（POST /videos
/// + GET /videos/{task_id}）与 OpenAI 兼容协议差异较大，独立实现更清晰。
pub struct AgnesAdapter {
    /// OpenAI 兼容地基（chat / image / embed / list_models 委托给它）
    compat: OpenAiCompatAdapter,
    /// 视频端点专用 HTTP 客户端（独立于 compat，避免暴露 compat 私有字段）
    http: HttpClient,
    /// 视频轮询 URL（Agnes 特有的 /agnesapi 轮询通道，可选）
    ///
    /// 配置时设置则 `video_poll` 优先走此通道（带 query 参数），失败再回退到
    /// /videos/{task_id}。对应 Python v1 `poll_url` 双路径策略。
    poll_url: Option<String>,
    /// Provider 配置（保留引用以便视频方法取 api_key / base_url）
    config: ProviderConfig,
}

impl AgnesAdapter {
    /// 创建 Agnes 适配器
    ///
    /// - `config`：Provider 配置，`base_url` 为 None 时用 `DEFAULT_AGNES_BASE_URL` 兜底
    /// - 内部构造 `OpenAiCompatAdapter`（委托用）与 `HttpClient`（视频端点用）
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let caps = agnes_capabilities();
        let poll_url = config.poll_url.clone();
        let base_url = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_AGNES_BASE_URL.to_string());

        // OpenAiCompatAdapter 内部会从 config.base_url 兜底到传入的默认值
        let compat = OpenAiCompatAdapter::new(
            config.clone(),
            "agnes",
            "Agnes AI",
            DEFAULT_AGNES_BASE_URL,
            caps,
        )?;

        // 视频端点专用 HttpClient：base_url 与 compat 一致，保证 URL 拼接正确
        let http = HttpClient::new(
            &ClientOptions::builder()
                .api_key(config.api_key.clone().unwrap_or_default())
                .base_url(base_url)
                .timeout(config.timeout)
                .max_retries(config.max_retries)
                .retry_delay(config.retry_delay)
                .build(),
        )?;

        Ok(Self {
            compat,
            http,
            poll_url,
            config,
        })
    }

    /// 用显式的 HttpClient 构造（测试用，可注入 mockito 后端）
    ///
    /// 接受一个已构造的 `OpenAiCompatAdapter` 与 `HttpClient`，便于测试时
    /// 复用 mockito 注入逻辑。两者应指向同一 base_url。
    #[cfg(test)]
    pub fn with_compat(
        compat: OpenAiCompatAdapter,
        http: HttpClient,
        config: ProviderConfig,
        poll_url: Option<String>,
    ) -> Self {
        Self {
            compat,
            http,
            poll_url,
            config,
        }
    }

    /// Agnes 支持的能力集合
    ///
    /// 对齐 Python v1 `agnes.py` 的 `supported_capabilities`。
    fn capabilities_set(&self) -> &CapabilitySet {
        self.compat.capabilities_set()
    }

    /// API key（可能为空，但 Agnes 要求非空）
    fn api_key(&self) -> Option<&str> {
        self.config.api_key.as_deref()
    }

    /// base_url（已合并 config 与默认值）
    fn base_url(&self) -> &str {
        self.compat.base_url()
    }

    /// 拼接完整 URL（base_url + 相对路径）
    fn url(&self, path: &str) -> String {
        let base = self.base_url().trim_end_matches('/');
        let path = path.trim_start_matches('/');
        format!("{base}/{path}")
    }

    /// 发送带认证的 POST JSON 请求，并用 OpenAI 错误映射处理响应
    ///
    /// 视频端点同样走 OpenAI 兼容的错误体格式 `{"error": {"message": "..."}}`，
    /// 复用 `OpenAiCompatAdapter::map_api_error` 统一映射。
    async fn post_authed_json(&self, path: &str, body: &Value) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .http
            .inner()
            .post(&url)
            .bearer_auth(self.api_key().unwrap_or(""))
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

    /// 发送带认证的 GET 请求（带可选 query 参数），并用 OpenAI 错误映射处理响应
    async fn get_authed_json(&self, path: &str, query: &[(&str, &str)]) -> Result<Value> {
        let url = self.url(path);
        let mut req = self
            .http
            .inner()
            .get(&url)
            .bearer_auth(self.api_key().unwrap_or(""));
        for (k, v) in query {
            req = req.query(&[(*k, *v)]);
        }
        let resp = req.send().await.map_err(|e| {
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

    // ==================== 视频协议（Agnes 特有） ====================

    /// 构造视频创建请求体
    ///
    /// 对应 Python v1 `agnes.py:video_create` 的 body 构造逻辑：
    /// - 基础字段：model / prompt
    /// - 可选参数：width / height / num_frames / frame_rate / mode / seed / negative_prompt
    /// - 参考图像：按 mode 分流到 extra_body
    ///   - keyframes 模式 + ≥2 图：extra_body.keyframes = { start, end }
    ///   - multiimage 模式 + ≥2 图：extra_body.image = [urls]
    ///   - 其余 ≥1 图：extra_body.image = url
    fn build_video_body(req: &VideoRequest) -> Value {
        let mut body = json!({
            "model": req.model,
            "prompt": req.prompt,
        });

        if req.width != 0 {
            body["width"] = json!(req.width);
        }
        if req.height != 0 {
            body["height"] = json!(req.height);
        }
        if let Some(n) = req.num_frames {
            body["num_frames"] = json!(n);
        }
        if req.frame_rate != 0 {
            body["frame_rate"] = json!(req.frame_rate);
        }
        // mode 序列化为小写字符串（text2video / image2video / keyframes / multiimage）
        body["mode"] = json!(serde_json::to_value(req.mode).unwrap_or(json!("text2video")));
        if let Some(seed) = req.seed {
            body["seed"] = json!(seed);
        }
        if let Some(np) = &req.negative_prompt {
            body["negative_prompt"] = json!(np);
        }

        // 参考图像分流到 extra_body
        if !req.reference_images.is_empty() {
            // 提取参考图像的 URL 字符串（FileInput::Url 取其 URL，其余类型序列化为字符串）
            let urls: Vec<String> = req
                .reference_images
                .iter()
                .map(file_input_to_string)
                .collect();
            let extra_body: Value = match req.mode {
                crate::model::common::VideoMode::Keyframes if urls.len() >= 2 => json!({
                    "keyframes": { "start": urls[0], "end": urls[urls.len() - 1] }
                }),
                crate::model::common::VideoMode::Multiimage if urls.len() >= 2 => json!({
                    "image": urls
                }),
                _ => json!({ "image": urls[0] }),
            };
            body["extra_body"] = extra_body;
        }

        // extra 透传：合并到顶层
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        body
    }

    /// 解析视频创建响应 → VideoTask
    ///
    /// 对应 Python v1 `agnes.py:video_create` 的响应解析：
    /// - task_id：优先 `id`，回退 `video_id`，再回退生成 ID
    /// - status：默认 "pending"
    /// - created_at：默认当前时间戳
    fn parse_video_task(value: &Value, model: &str) -> VideoTask {
        let task_id = value
            .get("id")
            .and_then(|v| v.as_str())
            .or_else(|| value.get("video_id").and_then(|v| v.as_str()))
            .map(str::to_owned)
            .unwrap_or_else(|| util::generate_id("vid"));
        let status = value
            .get("status")
            .and_then(|v| v.as_str())
            .map(parse_task_status)
            .unwrap_or(TaskStatus::Pending);
        let created_at = value
            .get("created")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(util::current_timestamp);
        VideoTask {
            task_id,
            model: model.to_string(),
            status,
            created_at,
        }
    }

    /// 解析视频轮询响应 → VideoStatus
    ///
    /// 对应 Python v1 `agnes.py:video_poll` 的响应解析：
    /// - status：默认 "pending"
    /// - video_url：优先顶层 `video_url`，回退 `output.video_url`
    /// - progress / error / created / updated 透传
    fn parse_video_status(value: &Value, task_id: &str) -> VideoStatus {
        let status = value
            .get("status")
            .and_then(|v| v.as_str())
            .map(parse_task_status)
            .unwrap_or(TaskStatus::Pending);
        let video_url = value
            .get("video_url")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| {
                value
                    .get("output")
                    .and_then(|o| o.get("video_url"))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            });
        let progress = value
            .get("progress")
            .and_then(|v| v.as_u64())
            .map(|p| p as u32);
        let error = value
            .get("error")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let created_at = value.get("created").and_then(|v| v.as_u64());
        let updated_at = value.get("updated").and_then(|v| v.as_u64());
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
}

/// 将 FileInput 转为可放入请求体的字符串表示
///
/// Agnes 视频协议的参考图像字段期望 URL 字符串。
/// - `Url(s)` → s
/// - `Base64(s)` → s（直接透传 base64 字符串）
/// - `Path(s)` → s（透传路径，由服务端解析）
/// - `Bytes(_)` → 序列化为 JSON 字符串（极少用，保留兜底）
fn file_input_to_string(f: &crate::model::image::FileInput) -> String {
    use crate::model::image::FileInput;
    match f {
        FileInput::Url(s) | FileInput::Path(s) | FileInput::Base64(s) => s.clone(),
        FileInput::Bytes(b) => serde_json::to_string(b).unwrap_or_default(),
    }
}

/// 解析任务状态字符串 → TaskStatus
///
/// 容忍服务端返回的多种写法（success/succeeded/failed/failure 等）。
fn parse_task_status(s: &str) -> TaskStatus {
    let lower = s.to_lowercase();
    match lower.as_str() {
        "success" | "succeeded" | "completed" | "done" => TaskStatus::Success,
        "failed" | "failure" | "error" => TaskStatus::Failed,
        "processing" | "running" | "generating" => TaskStatus::Processing,
        _ => TaskStatus::Pending,
    }
}

/// 构造 Agnes 支持的能力集合
fn agnes_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    // 对话能力
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps.insert(Capabilities::ToolCall);
    caps.insert(Capabilities::JsonMode);
    caps.insert(Capabilities::Reasoning);
    // 图像能力
    caps.insert(Capabilities::ImageGenerate);
    // 视频能力
    caps.insert(Capabilities::VideoGenerate);
    caps.insert(Capabilities::VideoText2Video);
    caps.insert(Capabilities::VideoImage2Video);
    // 嵌入能力
    caps.insert(Capabilities::Embedding);
    caps
}

#[async_trait]
impl Adapter for AgnesAdapter {
    fn provider_type(&self) -> &str {
        "agnes"
    }

    fn provider_name(&self) -> &str {
        "Agnes AI"
    }

    fn capabilities(&self) -> CapabilitySet {
        self.capabilities_set().clone()
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    async fn start(&mut self) -> Result<()> {
        // HttpClient 在构造时已创建，无需额外启动
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        // reqwest::Client 走 Drop 释放，无需显式关闭
        Ok(())
    }

    /// 文本对话（委托给 OpenAiCompatAdapter）
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.compat.chat(req).await
    }

    /// 流式文本对话（委托给 OpenAiCompatAdapter）
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.compat.chat_stream(req).await
    }

    /// 图像生成（委托给 OpenAiCompatAdapter）
    async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
        self.compat.image_generate(req).await
    }

    /// 创建视频生成任务（Agnes 特有协议）
    ///
    /// POST /videos，body 直传参数（model/prompt/width/height/.../extra_body）。
    /// 参考 Python v1 `agnes.py:video_create`。
    async fn video_create(&self, req: VideoRequest) -> Result<VideoTask> {
        // 能力校验：必须支持视频生成
        if !self
            .capabilities_set()
            .contains(&Capabilities::VideoGenerate)
        {
            return Err(AibridgeError::UnsupportedCapability {
                capability: format!("{} (provider: agnes)", Capabilities::VideoGenerate.as_str()),
            });
        }
        let body = Self::build_video_body(&req);
        let value = self.post_authed_json("videos", &body).await?;
        Ok(Self::parse_video_task(&value, &req.model))
    }

    /// 查询视频任务状态（Agnes 特有协议）
    ///
    /// 优先走 poll_url（Agnes 特有的 /agnesapi 轮询通道，带 video_id/model_name query），
    /// 网络错误时回退到 GET /videos/{task_id}。对应 Python v1 `agnes.py:video_poll`
    /// 的双路径策略。
    async fn video_poll(&self, task_id: &str, model: &str) -> Result<VideoStatus> {
        if !self
            .capabilities_set()
            .contains(&Capabilities::VideoGenerate)
        {
            return Err(AibridgeError::UnsupportedCapability {
                capability: format!("{} (provider: agnes)", Capabilities::VideoGenerate.as_str()),
            });
        }

        // 优先走 poll_url 通道（若配置）
        if let Some(poll_url) = &self.poll_url {
            let query: [(&str, &str); 2] = [("video_id", task_id), ("model_name", model)];
            match self.get_authed_json(poll_url, &query).await {
                Ok(value) => return Ok(Self::parse_video_status(&value, task_id)),
                Err(AibridgeError::Network(_) | AibridgeError::Timeout) => {
                    // 网络错误时回退到 /videos/{task_id}（旧版兼容路径）
                    let value = self
                        .get_authed_json(&format!("videos/{task_id}"), &[])
                        .await?;
                    return Ok(Self::parse_video_status(&value, task_id));
                }
                Err(e) => return Err(e), // 4xx 等 API 错误直接抛出，不回退
            }
        }

        // 无 poll_url：直接走 /videos/{task_id}
        let value = self
            .get_authed_json(&format!("videos/{task_id}"), &[])
            .await?;
        Ok(Self::parse_video_status(&value, task_id))
    }

    /// 文本嵌入（委托给 OpenAiCompatAdapter）
    async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        self.compat.embed(req).await
    }

    /// 模型列表（实时拉取，委托给 OpenAiCompatAdapter）
    ///
    /// v1.1.0 特性：GET /models 实时拉取，不再使用硬编码列表。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }

    /// 列出可用音色（Agnes 不支持音频能力）
    async fn list_voices(&self, _language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        Err(AibridgeError::UnsupportedCapability {
            capability: format!("{} (provider: agnes)", Capabilities::ListVoices.as_str()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::openai_compat::DEFAULT_OPENAI_BASE_URL;
    use crate::config::ClientOptions;
    use crate::http::HttpClient;
    use crate::model::chat::{ChatMessage, ChatRequest};
    use crate::model::common::VideoMode;
    use crate::model::image::{FileInput, ImageRequest};
    use crate::model::options::{EmbedInput, EmbedRequest};
    use crate::model::video::VideoRequest;
    use mockito::Server;
    use std::collections::HashMap;

    /// 构造测试用 AgnesAdapter（指向 mockito server）
    ///
    /// 为 compat（chat/image/embed/list_models）与视频端点各创建一个 HttpClient，
    /// 两者指向同一 mockito server，保证 URL 拼接一致。
    fn make_adapter(server: &Server, poll_url: Option<String>) -> AgnesAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("agnes", opts);
        let compat_http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        let video_http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        let caps = agnes_capabilities();
        let compat =
            OpenAiCompatAdapter::with_http(compat_http, config.clone(), "agnes", "Agnes AI", caps);
        AgnesAdapter::with_compat(compat, video_http, config, poll_url)
    }

    /// 不带 poll_url 的便捷构造
    fn make_adapter_no_poll(server: &Server) -> AgnesAdapter {
        make_adapter(server, None)
    }

    // ============ Adapter trait 基本属性 ============

    #[tokio::test]
    async fn provider_type_is_agnes() {
        let server = Server::new_async().await;
        let adapter = make_adapter_no_poll(&server);
        assert_eq!(adapter.provider_type(), "agnes");
    }

    #[tokio::test]
    async fn provider_name_is_agnes_ai() {
        let server = Server::new_async().await;
        let adapter = make_adapter_no_poll(&server);
        assert_eq!(adapter.provider_name(), "Agnes AI");
    }

    #[tokio::test]
    async fn requires_api_key_is_true() {
        let server = Server::new_async().await;
        let adapter = make_adapter_no_poll(&server);
        assert!(adapter.requires_api_key());
    }

    #[tokio::test]
    async fn capabilities_includes_chat_image_video_embed() {
        let server = Server::new_async().await;
        let adapter = make_adapter_no_poll(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::ImageGenerate));
        assert!(caps.contains(&Capabilities::VideoGenerate));
        assert!(caps.contains(&Capabilities::VideoText2Video));
        assert!(caps.contains(&Capabilities::VideoImage2Video));
        assert!(caps.contains(&Capabilities::Embedding));
    }

    #[tokio::test]
    async fn start_and_close_are_noops() {
        let server = Server::new_async().await;
        let mut adapter = make_adapter_no_poll(&server);
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    // ============ chat 委托（正常 + 错误路径） ============

    #[tokio::test]
    async fn chat_delegates_to_compat() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "chatcmpl-agnes-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "agnes-chat",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Agnes!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
        });
        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = ChatRequest::builder("agnes-chat", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .max_tokens(100)
            .build();
        let resp = adapter.chat(req).await.expect("chat 应成功");
        assert_eq!(resp.id, "chatcmpl-agnes-1");
        assert_eq!(resp.model, "agnes-chat");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("Hello from Agnes!")
        );
        assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 9);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(json!({"error": {"message": "Invalid API key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = ChatRequest::builder("agnes-chat", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(429)
            .with_body(
                json!({"error": {"message": "Rate limit exceeded", "retry_after": 2.0}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = ChatRequest::builder("agnes-chat", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::RateLimit { retry_after, .. } => {
                assert_eq!(retry_after, Some(2.0));
            }
            _ => panic!("应为 RateLimit"),
        }
    }

    // ============ image 委托 ============

    #[tokio::test]
    async fn image_generate_delegates_to_compat() {
        let mut server = Server::new_async().await;
        let body = json!({
            "created": 1700000000,
            "data": [{
                "url": "https://example.com/agnes-img.png",
                "revised_prompt": "a cute cat"
            }]
        });
        server
            .mock("POST", "/images/generations")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = ImageRequest::builder("agnes-image", "a cat")
            .size("1024x1024")
            .build();
        let resp = adapter.image_generate(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(
            resp.data[0].url.as_deref(),
            Some("https://example.com/agnes-img.png")
        );
    }

    #[tokio::test]
    async fn image_generate_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/images/generations")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;
        let adapter = make_adapter_no_poll(&server);
        let req = ImageRequest::builder("agnes-image", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    // ============ video_create 正常 + 错误路径 ============

    #[tokio::test]
    async fn video_create_success_returns_task() {
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "vid-task-1",
            "status": "pending",
            "created": 1700000000
        });
        let mock = server
            .mock("POST", "/videos")
            .match_header("authorization", "Bearer test-key")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "seedance-2.0",
                "prompt": "a cat walking",
                "mode": "text2video"
            })))
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = VideoRequest::builder("seedance-2.0", "a cat walking").build();
        let task = adapter
            .video_create(req)
            .await
            .expect("video_create 应成功");
        assert_eq!(task.task_id, "vid-task-1");
        assert_eq!(task.model, "seedance-2.0");
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.created_at, 1700000000);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_create_falls_back_to_video_id_field() {
        let mut server = Server::new_async().await;
        // 服务端返回 video_id 而非 id
        let body = json!({"video_id": "vid-alt-1", "status": "processing"});
        server
            .mock("POST", "/videos")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = VideoRequest::builder("seedance-2.0", "prompt").build();
        let task = adapter.video_create(req).await.unwrap();
        assert_eq!(task.task_id, "vid-alt-1");
        assert_eq!(task.status, TaskStatus::Processing);
    }

    #[tokio::test]
    async fn video_create_sends_width_height_and_mode() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/videos")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "seedance-2.0",
                "prompt": "a cat",
                "width": 1920,
                "height": 1080,
                "mode": "image2video",
                "seed": 42
            })))
            .with_status(200)
            .with_body(json!({"id": "t-1", "status": "pending"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = VideoRequest::builder("seedance-2.0", "a cat")
            .width(1920)
            .height(1080)
            .mode(VideoMode::Image2Video)
            .seed(42)
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_create_image2video_single_image_in_extra_body() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/videos")
            .match_body(mockito::Matcher::PartialJson(json!({
                "mode": "image2video",
                "extra_body": {"image": "https://example.com/ref.png"}
            })))
            .with_status(200)
            .with_body(json!({"id": "t-1"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = VideoRequest::builder("seedance-2.0", "animate")
            .mode(VideoMode::Image2Video)
            .reference_images(vec![FileInput::url("https://example.com/ref.png")])
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_create_keyframes_mode_uses_start_end() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/videos")
            .match_body(mockito::Matcher::PartialJson(json!({
                "mode": "keyframes",
                "extra_body": {
                    "keyframes": {
                        "start": "https://example.com/start.png",
                        "end": "https://example.com/end.png"
                    }
                }
            })))
            .with_status(200)
            .with_body(json!({"id": "t-1"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = VideoRequest::builder("seedance-2.0", "animate")
            .mode(VideoMode::Keyframes)
            .reference_images(vec![
                FileInput::url("https://example.com/start.png"),
                FileInput::url("https://example.com/end.png"),
            ])
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_create_multiimage_mode_uses_image_array() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/videos")
            .match_body(mockito::Matcher::PartialJson(json!({
                "mode": "multiimage",
                "extra_body": {
                    "image": ["https://example.com/a.png", "https://example.com/b.png"]
                }
            })))
            .with_status(200)
            .with_body(json!({"id": "t-1"}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = VideoRequest::builder("seedance-2.0", "animate")
            .mode(VideoMode::Multiimage)
            .reference_images(vec![
                FileInput::url("https://example.com/a.png"),
                FileInput::url("https://example.com/b.png"),
            ])
            .build();
        let _ = adapter.video_create(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_create_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos")
            .with_status(401)
            .with_body(json!({"error": {"message": "Invalid API key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = VideoRequest::builder("seedance-2.0", "prompt").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn video_create_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos")
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = VideoRequest::builder("seedance-2.0", "prompt").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn video_create_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/videos")
            .with_status(500)
            .with_body(json!({"error": {"message": "internal"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = VideoRequest::builder("seedance-2.0", "prompt").build();
        let err = adapter.video_create(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 500),
            _ => panic!("应为 Api"),
        }
    }

    // ============ video_poll 正常 + 错误路径 ============

    #[tokio::test]
    async fn video_poll_success_via_videos_endpoint() {
        let mut server = Server::new_async().await;
        let body = json!({
            "status": "success",
            "video_url": "https://example.com/video.mp4",
            "progress": 100,
            "created": 1700000000,
            "updated": 1700000100
        });
        let mock = server
            .mock("GET", "/videos/vid-task-1")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let status = adapter
            .video_poll("vid-task-1", "seedance-2.0")
            .await
            .expect("video_poll 应成功");
        assert_eq!(status.task_id, "vid-task-1");
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://example.com/video.mp4")
        );
        assert_eq!(status.progress, Some(100));
        assert_eq!(status.created_at, Some(1700000000));
        assert_eq!(status.updated_at, Some(1700000100));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_poll_reads_video_url_from_output_field() {
        let mut server = Server::new_async().await;
        // video_url 嵌套在 output 对象中
        let body = json!({
            "status": "success",
            "output": {"video_url": "https://example.com/nested.mp4"}
        });
        server
            .mock("GET", "/videos/t-2")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let status = adapter.video_poll("t-2", "seedance-2.0").await.unwrap();
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://example.com/nested.mp4")
        );
    }

    #[tokio::test]
    async fn video_poll_processing_status() {
        let mut server = Server::new_async().await;
        let body = json!({"status": "processing", "progress": 45});
        server
            .mock("GET", "/videos/t-3")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let status = adapter.video_poll("t-3", "seedance-2.0").await.unwrap();
        assert_eq!(status.status, TaskStatus::Processing);
        assert_eq!(status.progress, Some(45));
        assert!(status.video_url.is_none());
    }

    #[tokio::test]
    async fn video_poll_failed_status_with_error() {
        let mut server = Server::new_async().await;
        let body = json!({"status": "failed", "error": "content policy violation"});
        server
            .mock("GET", "/videos/t-4")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let status = adapter.video_poll("t-4", "seedance-2.0").await.unwrap();
        assert_eq!(status.status, TaskStatus::Failed);
        assert_eq!(status.error.as_deref(), Some("content policy violation"));
    }

    #[tokio::test]
    async fn video_poll_prefers_poll_url_when_configured() {
        let mut server = Server::new_async().await;
        // poll_url 走 /agnesapi/video/query 路径，带 query 参数
        let body = json!({"status": "success", "video_url": "https://example.com/v.mp4"});
        let mock = server
            .mock("GET", "/agnesapi/video/query")
            .match_query(mockito::Matcher::AllOf(vec![
                mockito::Matcher::UrlEncoded("video_id".into(), "vid-x".into()),
                mockito::Matcher::UrlEncoded("model_name".into(), "seedance-2.0".into()),
            ]))
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server, Some("agnesapi/video/query".to_string()));
        let status = adapter.video_poll("vid-x", "seedance-2.0").await.unwrap();
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://example.com/v.mp4")
        );
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn video_poll_does_not_fallback_on_api_error() {
        let mut server = Server::new_async().await;
        // poll_url 返回 500（API 错误，非 Network/Timeout），不应回退，直接抛 Api 错误
        server
            .mock("GET", "/agnesapi/video/query")
            .with_status(500)
            .with_body(json!({"error": {"message": "internal"}}).to_string())
            .create_async()
            .await;
        // /videos/{id} 不应被调用：mockito 用 expect(0) 断言 0 次命中
        server
            .mock("GET", "/videos/vid-y")
            .expect(0)
            .create_async()
            .await;

        let adapter = make_adapter(&server, Some("agnesapi/video/query".to_string()));
        // 500 是 API 错误（非 Network/Timeout），直接抛出，不回退
        let err = adapter
            .video_poll("vid-y", "seedance-2.0")
            .await
            .unwrap_err();
        assert!(matches!(err, AibridgeError::Api { .. }));
    }

    #[tokio::test]
    async fn video_poll_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/t-5")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let err = adapter.video_poll("t-5", "seedance-2.0").await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn video_poll_error_404_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/videos/t-missing")
            .with_status(404)
            .with_body(json!({"error": {"message": "task not found"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let err = adapter
            .video_poll("t-missing", "seedance-2.0")
            .await
            .unwrap_err();
        // 404 在 OpenAI 错误映射里走 ModelNotFound
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    // ============ embed 委托 ============

    #[tokio::test]
    async fn embed_delegates_to_compat() {
        let mut server = Server::new_async().await;
        let body = json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "model": "agnes-embed",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        });
        server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let req = EmbedRequest {
            model: "agnes-embed".into(),
            input: EmbedInput::Single("hello".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let resp = adapter.embed(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.model, "agnes-embed");
    }

    #[tokio::test]
    async fn embed_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(401)
            .with_body(json!({"error": {"message": "bad key"}}).to_string())
            .create_async()
            .await;
        let adapter = make_adapter_no_poll(&server);
        let req = EmbedRequest {
            model: "agnes-embed".into(),
            input: EmbedInput::Single("hi".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let err = adapter.embed(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    // ============ list_models 委托（实时拉取） ============

    #[tokio::test]
    async fn list_models_pulls_from_models_endpoint() {
        let mut server = Server::new_async().await;
        let body = json!({
            "object": "list",
            "data": [
                {"id": "agnes-chat", "object": "model", "created": 1700000000, "owned_by": "agnes"},
                {"id": "seedream-4.0", "object": "model", "created": 1700000000, "owned_by": "agnes"},
                {"id": "seedance-2.0", "object": "model", "created": 1700000000, "owned_by": "agnes"}
            ]
        });
        let mock = server
            .mock("GET", "/models")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 3);
        // 类型推断
        assert_eq!(models[0].id, "agnes-chat");
        assert_eq!(models[0].model_type, ModelType::Chat);
        assert_eq!(models[1].model_type, ModelType::Image);
        assert_eq!(models[2].model_type, ModelType::Video);
        // provider 字段填充为 agnes
        assert_eq!(models[0].provider, "agnes");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn list_models_filter_by_type() {
        let mut server = Server::new_async().await;
        let body = json!({
            "data": [
                {"id": "agnes-chat", "object": "model", "created": 1, "owned_by": "agnes"},
                {"id": "seedance-2.0", "object": "model", "created": 1, "owned_by": "agnes"}
            ]
        });
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter_no_poll(&server);
        let videos = adapter.list_models(Some(ModelType::Video)).await.unwrap();
        assert_eq!(videos.len(), 1);
        assert_eq!(videos[0].id, "seedance-2.0");
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
        let adapter = make_adapter_no_poll(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    // ============ list_voices 不支持 ============

    #[tokio::test]
    async fn list_voices_returns_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_adapter_no_poll(&server);
        let err = adapter.list_voices(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ============ 构造与 base_url 兜底 ============

    #[tokio::test]
    async fn new_uses_default_base_url_when_missing() {
        let config =
            ProviderConfig::from_options("agnes", ClientOptions::builder().api_key("k").build());
        let adapter = AgnesAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), DEFAULT_AGNES_BASE_URL);
    }

    #[tokio::test]
    async fn new_uses_config_base_url_when_provided() {
        let config = ProviderConfig::from_options(
            "agnes",
            ClientOptions::builder()
                .api_key("k")
                .base_url("https://custom.agnes.example.com/v1")
                .build(),
        );
        let adapter = AgnesAdapter::new(config).unwrap();
        assert_eq!(adapter.base_url(), "https://custom.agnes.example.com/v1");
    }

    #[tokio::test]
    async fn new_preserves_poll_url() {
        let config = ProviderConfig::from_options(
            "agnes",
            ClientOptions::builder()
                .api_key("k")
                .poll_url("agnesapi/video/query")
                .build(),
        );
        let adapter = AgnesAdapter::new(config).unwrap();
        assert_eq!(adapter.poll_url.as_deref(), Some("agnesapi/video/query"));
    }

    // ============ 内部解析函数单元测试 ============

    #[test]
    fn parse_task_status_recognizes_variants() {
        assert_eq!(parse_task_status("success"), TaskStatus::Success);
        assert_eq!(parse_task_status("SUCCEEDED"), TaskStatus::Success);
        assert_eq!(parse_task_status("completed"), TaskStatus::Success);
        assert_eq!(parse_task_status("failed"), TaskStatus::Failed);
        assert_eq!(parse_task_status("FAILURE"), TaskStatus::Failed);
        assert_eq!(parse_task_status("error"), TaskStatus::Failed);
        assert_eq!(parse_task_status("processing"), TaskStatus::Processing);
        assert_eq!(parse_task_status("running"), TaskStatus::Processing);
        assert_eq!(parse_task_status("queued"), TaskStatus::Pending);
        assert_eq!(parse_task_status("unknown"), TaskStatus::Pending);
    }

    #[test]
    fn parse_video_task_uses_id_first() {
        let value = json!({"id": "t-id", "video_id": "t-vid", "status": "pending"});
        let task = AgnesAdapter::parse_video_task(&value, "m");
        assert_eq!(task.task_id, "t-id");
        assert_eq!(task.model, "m");
        assert_eq!(task.status, TaskStatus::Pending);
    }

    #[test]
    fn parse_video_task_falls_back_to_video_id() {
        let value = json!({"video_id": "t-vid", "status": "success"});
        let task = AgnesAdapter::parse_video_task(&value, "m");
        assert_eq!(task.task_id, "t-vid");
        assert_eq!(task.status, TaskStatus::Success);
    }

    #[test]
    fn parse_video_task_generates_id_when_missing() {
        let value = json!({"status": "pending"});
        let task = AgnesAdapter::parse_video_task(&value, "m");
        assert!(task.task_id.starts_with("vid_"), "应生成 vid_ 前缀 ID");
    }

    #[test]
    fn parse_video_status_extracts_top_level_video_url() {
        let value = json!({"status": "success", "video_url": "https://x.com/v.mp4"});
        let s = AgnesAdapter::parse_video_status(&value, "t-1");
        assert_eq!(s.video_url.as_deref(), Some("https://x.com/v.mp4"));
    }

    #[test]
    fn parse_video_status_extracts_nested_output_video_url() {
        let value = json!({
            "status": "success",
            "output": {"video_url": "https://x.com/nested.mp4"}
        });
        let s = AgnesAdapter::parse_video_status(&value, "t-1");
        assert_eq!(s.video_url.as_deref(), Some("https://x.com/nested.mp4"));
    }

    #[test]
    fn build_video_body_text2video_minimal() {
        let req = VideoRequest::builder("seedance-2.0", "a cat").build();
        let body = AgnesAdapter::build_video_body(&req);
        assert_eq!(body["model"], "seedance-2.0");
        assert_eq!(body["prompt"], "a cat");
        assert_eq!(body["mode"], "text2video");
        // 无参考图像时不应有 extra_body
        assert!(body.get("extra_body").is_none());
    }

    #[test]
    fn build_video_body_includes_all_optional_params() {
        let req = VideoRequest::builder("seedance-2.0", "a cat")
            .width(1920)
            .height(1080)
            .num_frames(120)
            .frame_rate(30)
            .seed(42)
            .negative_prompt("blurry")
            .build();
        let body = AgnesAdapter::build_video_body(&req);
        assert_eq!(body["width"], 1920);
        assert_eq!(body["height"], 1080);
        assert_eq!(body["num_frames"], 120);
        assert_eq!(body["frame_rate"], 30);
        assert_eq!(body["seed"], 42);
        assert_eq!(body["negative_prompt"], "blurry");
    }

    #[test]
    fn build_video_body_extra_passthrough() {
        let req = VideoRequest::builder("seedance-2.0", "a cat")
            .extra("custom_param", "custom_value")
            .build();
        let body = AgnesAdapter::build_video_body(&req);
        assert_eq!(body["custom_param"], "custom_value");
    }

    #[test]
    fn build_video_body_keyframes_with_two_images() {
        let req = VideoRequest::builder("seedance-2.0", "animate")
            .mode(VideoMode::Keyframes)
            .reference_images(vec![
                FileInput::url("https://x.com/start.png"),
                FileInput::url("https://x.com/end.png"),
            ])
            .build();
        let body = AgnesAdapter::build_video_body(&req);
        assert_eq!(
            body["extra_body"]["keyframes"]["start"],
            "https://x.com/start.png"
        );
        assert_eq!(
            body["extra_body"]["keyframes"]["end"],
            "https://x.com/end.png"
        );
    }

    #[test]
    fn build_video_body_multiimage_with_two_images() {
        let req = VideoRequest::builder("seedance-2.0", "animate")
            .mode(VideoMode::Multiimage)
            .reference_images(vec![
                FileInput::url("https://x.com/a.png"),
                FileInput::url("https://x.com/b.png"),
            ])
            .build();
        let body = AgnesAdapter::build_video_body(&req);
        let images = body["extra_body"]["image"].as_array().unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(images[0], "https://x.com/a.png");
    }

    #[test]
    fn build_video_body_image2video_single_image() {
        let req = VideoRequest::builder("seedance-2.0", "animate")
            .mode(VideoMode::Image2Video)
            .reference_images(vec![FileInput::url("https://x.com/ref.png")])
            .build();
        let body = AgnesAdapter::build_video_body(&req);
        assert_eq!(body["extra_body"]["image"], "https://x.com/ref.png");
    }

    #[test]
    fn file_input_to_string_handles_variants() {
        assert_eq!(
            file_input_to_string(&FileInput::url("https://x.com/a.png")),
            "https://x.com/a.png"
        );
        assert_eq!(
            file_input_to_string(&FileInput::path("/tmp/a.png")),
            "/tmp/a.png"
        );
        assert_eq!(
            file_input_to_string(&FileInput::base64("aGVsbG8=")),
            "aGVsbG8="
        );
    }

    #[test]
    fn agnes_capabilities_contains_expected_set() {
        let caps = agnes_capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::ImageGenerate));
        assert!(caps.contains(&Capabilities::VideoGenerate));
        assert!(caps.contains(&Capabilities::VideoText2Video));
        assert!(caps.contains(&Capabilities::VideoImage2Video));
        assert!(caps.contains(&Capabilities::Embedding));
        // Agnes 不声明音频能力
        assert!(!caps.contains(&Capabilities::AudioSpeech));
        assert!(!caps.contains(&Capabilities::AudioTranscribe));
    }

    /// 编译期断言：DEFAULT_AGNES_BASE_URL 与 openai_compat 默认值不同
    #[test]
    fn agnes_default_base_url_differs_from_openai() {
        assert_ne!(DEFAULT_AGNES_BASE_URL, DEFAULT_OPENAI_BASE_URL);
        assert_eq!(DEFAULT_AGNES_BASE_URL, "https://api.agnes.ai/v1");
    }
}
