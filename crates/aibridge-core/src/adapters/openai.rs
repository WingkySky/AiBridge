//! OpenAI 适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/openai.py`。
//!
//! OpenAI 是 OpenAI 兼容协议的本源，本适配器不重写任何 HTTP/解析逻辑，
//! 而是组合（委托）[`crate::adapters::openai_compat::OpenAiCompatAdapter`] 地基：
//! Rust 无继承，用"持有地基 + trait 方法转发"模式复用 80% 通用代码。
//!
//! 与 Python 老版的行为对照：
//! - `provider_type = "openai"`、`provider_name = "OpenAI"`
//! - 默认 `base_url = https://api.openai.com/v1`（可被 config.base_url 覆盖）
//! - 能力：Chat / ChatStream / ImageGenerate / Embedding
//! - `list_models` 实时拉取 `GET /models`
//! - 不支持的能力（video / transcribe / speech / list_voices）走 trait 默认实现，
//!   返 `UnsupportedCapability`（与 Python `video_create` 抛错一致）
//! - `requires_api_key = true`

use async_trait::async_trait;

use crate::adapter::{Adapter, Capabilities, CapabilitySet, ChatStream};
use crate::adapters::openai_compat::{OpenAiCompatAdapter, DEFAULT_OPENAI_BASE_URL};
use crate::config::ProviderConfig;
use crate::error::Result;
use crate::model::chat::{ChatCompletion, ChatRequest};
use crate::model::common::{ModelInfo, ModelType};
use crate::model::image::{ImageRequest, ImageResult};
use crate::model::options::{EmbedRequest, EmbeddingResult};

/// OpenAI 适配器
///
/// 组合持有 [`OpenAiCompatAdapter`] 地基，把 `Adapter` trait 的核心方法
/// （chat / chat_stream / image_generate / embed / list_models）委托给地基实现。
/// 本结构仅负责：provider 元信息、能力集合声明、构造地基。
///
/// 构造时即建立 HTTP 客户端（地基 `new` 内部完成），`start` / `close` 为空操作。
pub struct OpenAiAdapter {
    /// OpenAI 兼容协议地基，复用 chat/image/embed/list_models 实现
    compat: OpenAiCompatAdapter,
}

impl OpenAiAdapter {
    /// 创建 OpenAI 适配器
    ///
    /// - `config.base_url` 为空时回退到 `https://api.openai.com/v1`
    /// - 内部构造 `OpenAiCompatAdapter`（含 HTTP 客户端、连接池、错误映射）
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let caps = Self::capabilities_set();
        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_OPENAI_BASE_URL,
            caps,
        )?;
        Ok(Self { compat })
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "openai";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "OpenAI";

    /// 支持的能力集合
    ///
    /// 对应 Python v1 `OpenAIAdapter.supported_capabilities` 的核心子集
    /// （chat / chat_stream / image_generate / embedding）。
    /// video / audio 不在 OpenAI 本适配器能力内，走 trait 默认实现返 UnsupportedCapability。
    fn capabilities_set() -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::Chat);
        caps.insert(Capabilities::ChatStream);
        caps.insert(Capabilities::ImageGenerate);
        caps.insert(Capabilities::Embedding);
        caps
    }
}

#[async_trait]
impl Adapter for OpenAiAdapter {
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

    /// 文本对话：委托地基 `POST /chat/completions`
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.compat.chat(req).await
    }

    /// 流式文本对话：委托地基 `POST /chat/completions` (stream=true)
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.compat.chat_stream(req).await
    }

    /// 图像生成：委托地基 `POST /images/generations`
    async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
        self.compat.image_generate(req).await
    }

    /// 文本嵌入：委托地基 `POST /embeddings`
    async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        self.compat.embed(req).await
    }

    /// 模型列表（实时拉取）：委托地基 `GET /models`
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.compat.list_models(filter).await
    }

    // 其余方法（video_create / video_poll / transcribe / speech / list_voices）
    // 走 trait 默认实现，返 UnsupportedCapability，与 Python 老版行为一致。
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientOptions;
    use crate::error::AibridgeError;
    use crate::model::chat::ChatMessage;
    use crate::model::options::{EmbedInput, EmbeddingVector};
    use futures::stream::StreamExt;
    use mockito::Server;
    use serde_json::json;
    use std::collections::HashMap;

    /// 构造指向 mockito server 的 OpenAiAdapter（base_url 注入 mock 地址）
    fn make_adapter(server: &Server) -> OpenAiAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("openai", opts);
        OpenAiAdapter::new(config).expect("OpenAiAdapter 构造应成功")
    }

    /// 构造不指向任何 server 的 OpenAiAdapter（用于不发请求的元信息/能力测试）
    fn make_adapter_no_server() -> OpenAiAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url("https://api.openai.com/v1")
            .build();
        let config = ProviderConfig::from_options("openai", opts);
        OpenAiAdapter::new(config).expect("OpenAiAdapter 构造应成功")
    }

    // ============ 元信息 ============

    #[test]
    fn provider_type_and_name_match_python() {
        let adapter = make_adapter_no_server();
        assert_eq!(adapter.provider_type(), "openai");
        assert_eq!(adapter.provider_name(), "OpenAI");
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
        // video / audio 不应声明
        assert!(!caps.contains(&Capabilities::VideoGenerate));
        assert!(!caps.contains(&Capabilities::AudioSpeech));
    }

    #[test]
    fn base_url_defaults_to_openai_when_missing() {
        // 不提供 base_url，应回退到 OpenAI 官方默认值
        let opts = ClientOptions::builder().api_key("k").build();
        let config = ProviderConfig::from_options("openai", opts);
        let adapter = OpenAiAdapter::new(config).unwrap();
        // 地基的 base_url 暴露为 pub，可直接读取
        assert_eq!(adapter.compat.base_url(), DEFAULT_OPENAI_BASE_URL);
    }

    #[test]
    fn base_url_uses_config_when_provided() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://custom.openai-proxy.com/v1")
            .build();
        let config = ProviderConfig::from_options("openai", opts);
        let adapter = OpenAiAdapter::new(config).unwrap();
        assert_eq!(
            adapter.compat.base_url(),
            "https://custom.openai-proxy.com/v1"
        );
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
            .match_header("authorization", "Bearer test-key")
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
    async fn chat_sends_bearer_auth_and_model() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
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

    // ============ chat 错误路径 ============

    #[tokio::test]
    async fn chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
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
            .with_status(429)
            .with_body(json!({"error": {"message": "slow down"}}).to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn chat_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/chat/completions")
            .with_status(404)
            .with_body(
                json!({"error": {"message": "The model 'gpt-x' does not exist"}}).to_string(),
            )
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let req = ChatRequest::builder("gpt-x", vec![ChatMessage::user("hi")]).build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
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
        server
            .mock("POST", "/images/generations")
            .match_header("authorization", "Bearer test-key")
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
        server
            .mock("POST", "/embeddings")
            .match_header("authorization", "Bearer test-key")
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
    }

    #[tokio::test]
    async fn embed_single_input_sends_string() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/embeddings")
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
    async fn list_models_success() {
        let mut server = Server::new_async().await;
        let body = json!({
            "object": "list",
            "data": [
                {"id": "gpt-4o", "object": "model", "created": 1700000000, "owned_by": "openai"},
                {"id": "dall-e-3", "object": "model", "created": 1700000000, "owned_by": "openai"},
                {"id": "whisper-1", "object": "model", "created": 1700000000, "owned_by": "openai"}
            ]
        });
        server
            .mock("GET", "/models")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].id, "gpt-4o");
        assert_eq!(models[0].model_type, ModelType::Chat);
        assert_eq!(models[1].model_type, ModelType::Image);
        assert_eq!(models[2].model_type, ModelType::Audio);
        // provider 字段填充为 "openai"
        assert_eq!(models[0].provider, "openai");
    }

    #[tokio::test]
    async fn list_models_filter_by_type() {
        let mut server = Server::new_async().await;
        let body = json!({
            "data": [
                {"id": "gpt-4o", "object": "model", "created": 1, "owned_by": "openai"},
                {"id": "dall-e-3", "object": "model", "created": 1, "owned_by": "openai"}
            ]
        });
        server
            .mock("GET", "/models")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server);
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "dall-e-3");
    }

    #[tokio::test]
    async fn list_models_error_401_returns_authentication() {
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

    // ============ 不支持的能力走默认实现 ============

    #[tokio::test]
    async fn video_create_returns_unsupported() {
        // 与 Python 老版 `video_create` 抛 UnsupportedCapabilityError 行为一致
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
