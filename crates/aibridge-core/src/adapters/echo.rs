//! Echo（Mock）适配器
//!
//! 用于阶段 0.6 五语言管线验证的回显适配器。
//! 不调用任何真实网络，所有方法返回固定/回显响应，便于在没有 API Key 与
//! 外部依赖的情况下端到端验证 chat / chat_stream / image / video / embed /
//! transcribe / speech / list_models / list_voices 等能力的数据流。
//!
//! 设计要点：
//! - `provider_type = "echo"`，`requires_api_key = false`（免认证，便于管线验证）
//! - `capabilities` 声明全部能力，使 Router / Client 的能力检查均通过
//! - `chat` 回显最后一条 user 消息内容并追加 " [echo]"
//! - `chat_stream` 用 `async_stream` 产生 3 个 chunk，逐步拼接出完整回显
//! - 二进制能力（speech）返回固定字节，验证跨语言二进制传递

use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::stream::StreamExt;

use crate::adapter::{Adapter, Capabilities, CapabilitySet, ChatStream};
use crate::error::Result;
use crate::model::audio::{SpeechResult, TranscriptionResult};
use crate::model::chat::{
    ChatChoice, ChatCompletion, ChatCompletionChunk, ChatCompletionDelta, ChatMessage, ChatRequest,
    ChoiceMessage, DeltaMessage, UserContent,
};
use crate::model::common::{ModelInfo, ModelType, TaskStatus, VoiceInfo};
use crate::model::image::{ImageData, ImageRequest, ImageResult};
use crate::model::options::{
    EmbedInput, EmbedRequest, EmbeddingItem, EmbeddingResult, EmbeddingUsage, EmbeddingVector,
};
use crate::model::video::{VideoRequest, VideoStatus, VideoTask};

/// Echo（Mock）适配器
///
/// 不持有任何资源（无 HTTP 客户端），构造轻量。
/// 所有方法返回固定/回显响应，不调用网络。
#[derive(Debug, Default, Clone)]
pub struct EchoAdapter;

impl EchoAdapter {
    /// 创建 Echo 适配器
    ///
    /// 无需任何配置，构造不进行网络初始化。
    pub fn new() -> Self {
        Self
    }

    /// 构造全部能力集合（echo 声明支持所有能力，便于管线验证）
    fn all_capabilities() -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::Chat);
        caps.insert(Capabilities::ChatStream);
        caps.insert(Capabilities::Vision);
        caps.insert(Capabilities::ToolCall);
        caps.insert(Capabilities::Reasoning);
        caps.insert(Capabilities::JsonMode);
        caps.insert(Capabilities::WebSearch);
        caps.insert(Capabilities::ImageGenerate);
        caps.insert(Capabilities::ImageEdit);
        caps.insert(Capabilities::VideoGenerate);
        caps.insert(Capabilities::VideoText2Video);
        caps.insert(Capabilities::VideoImage2Video);
        caps.insert(Capabilities::Embedding);
        caps.insert(Capabilities::AudioTranscribe);
        caps.insert(Capabilities::AudioSpeech);
        caps.insert(Capabilities::ListVoices);
        caps
    }

    /// 提取请求中最后一条 user 消息的文本内容
    ///
    /// 多模态消息取首个文本部件；无 user 消息时返回空串。
    fn last_user_text(req: &ChatRequest) -> String {
        for msg in req.messages.iter().rev() {
            if let ChatMessage::User { content, .. } = msg {
                return match content {
                    UserContent::Text(s) => s.clone(),
                    UserContent::Parts(parts) => parts
                        .iter()
                        .find_map(|p| match p {
                            crate::model::chat::ContentPart::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                        .unwrap_or_default(),
                };
            }
        }
        String::new()
    }

    /// 当前 Unix 时间戳（秒）
    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// 构造固定模型列表
    fn fixed_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "echo-chat".into(),
                name: "Echo Chat".into(),
                model_type: ModelType::Chat,
                provider: "echo".into(),
                capabilities: vec!["chat".into(), "chat_stream".into()],
                max_tokens: Some(4096),
                supports_streaming: true,
                description: Some("Echo mock chat model".into()),
                created: None,
            },
            ModelInfo {
                id: "echo-image".into(),
                name: "Echo Image".into(),
                model_type: ModelType::Image,
                provider: "echo".into(),
                capabilities: vec!["text2image".into()],
                max_tokens: None,
                supports_streaming: false,
                description: Some("Echo mock image model".into()),
                created: None,
            },
            ModelInfo {
                id: "echo-video".into(),
                name: "Echo Video".into(),
                model_type: ModelType::Video,
                provider: "echo".into(),
                capabilities: vec!["text2video".into()],
                max_tokens: None,
                supports_streaming: false,
                description: Some("Echo mock video model".into()),
                created: None,
            },
            ModelInfo {
                id: "echo-tts".into(),
                name: "Echo TTS".into(),
                model_type: ModelType::Audio,
                provider: "echo".into(),
                capabilities: vec!["audio_speech".into()],
                max_tokens: None,
                supports_streaming: false,
                description: Some("Echo mock TTS model".into()),
                created: None,
            },
            ModelInfo {
                id: "echo-asr".into(),
                name: "Echo ASR".into(),
                model_type: ModelType::Audio,
                provider: "echo".into(),
                capabilities: vec!["audio_transcribe".into()],
                max_tokens: None,
                supports_streaming: false,
                description: Some("Echo mock ASR model".into()),
                created: None,
            },
            ModelInfo {
                id: "echo-embed".into(),
                name: "Echo Embed".into(),
                model_type: ModelType::Chat,
                provider: "echo".into(),
                capabilities: vec!["embedding".into()],
                max_tokens: None,
                supports_streaming: false,
                description: Some("Echo mock embedding model".into()),
                created: None,
            },
        ]
    }

    /// 构造固定音色列表
    fn fixed_voices() -> Vec<VoiceInfo> {
        vec![
            VoiceInfo::builder()
                .short_name("echo-voice-1")
                .name("Echo Voice 1")
                .locale("zh-CN")
                .gender("Female")
                .voice_id("echo-voice-1")
                .build(),
            VoiceInfo::builder()
                .short_name("echo-voice-2")
                .name("Echo Voice 2")
                .locale("en-US")
                .gender("Male")
                .voice_id("echo-voice-2")
                .build(),
        ]
    }

    /// 1x1 透明 PNG 的 Base64 编码
    ///
    /// 用于 image_generate 返回的占位图，避免引入图像库。
    const PLACEHOLDER_PNG_B64: &'static str =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

    /// 构造完整回显文本
    fn echo_text(req: &ChatRequest) -> String {
        format!("{} [echo]", Self::last_user_text(req))
    }
}

#[async_trait]
impl Adapter for EchoAdapter {
    fn provider_type(&self) -> &str {
        "echo"
    }

    fn provider_name(&self) -> &str {
        "Echo (Mock)"
    }

    fn capabilities(&self) -> CapabilitySet {
        Self::all_capabilities()
    }

    /// 免认证：echo 不需要 API Key，便于管线验证
    fn requires_api_key(&self) -> bool {
        false
    }

    async fn start(&mut self) -> Result<()> {
        // 无资源需初始化
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        // 无资源需释放
        Ok(())
    }

    /// 文本对话：回显最后一条 user 消息内容 + " [echo]"
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        let content = Self::echo_text(&req);
        Ok(ChatCompletion {
            id: "echo-chat-1".into(),
            object: "chat.completion".into(),
            created: Self::now_secs(),
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChoiceMessage {
                    role: "assistant".into(),
                    content: Some(content),
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(crate::model::chat::ChatUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            }),
            service_tier: None,
            system_fingerprint: None,
        })
    }

    /// 流式文本对话：用 async_stream 产生 3 个 chunk
    ///
    /// chunk 内容逐步拼接：第 1 块含 role，第 2 块含前半段内容，
    /// 第 3 块含后半段内容并标记 finish_reason。
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        let model = req.model.clone();
        let full = Self::echo_text(&req);
        let created = Self::now_secs();

        // 将完整回显拆为两段，模拟流式增量
        let mid = full.len() / 2;
        let (first_half, second_half) = if full.is_empty() {
            (String::new(), String::new())
        } else {
            (full[..mid].to_string(), full[mid..].to_string())
        };

        let stream = async_stream::stream! {
            // chunk 1：角色声明
            yield Ok(ChatCompletionChunk {
                id: "echo-stream-1".into(),
                object: "chat.completion.chunk".into(),
                created,
                model: model.clone(),
                choices: vec![ChatCompletionDelta {
                    index: 0,
                    delta: DeltaMessage {
                        role: Some("assistant".into()),
                        content: None,
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
            });
            // chunk 2：前半段内容
            yield Ok(ChatCompletionChunk {
                id: "echo-stream-1".into(),
                object: "chat.completion.chunk".into(),
                created,
                model: model.clone(),
                choices: vec![ChatCompletionDelta {
                    index: 0,
                    delta: DeltaMessage {
                        role: None,
                        content: Some(first_half),
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
            });
            // chunk 3：后半段内容 + 结束
            yield Ok(ChatCompletionChunk {
                id: "echo-stream-1".into(),
                object: "chat.completion.chunk".into(),
                created,
                model,
                choices: vec![ChatCompletionDelta {
                    index: 0,
                    delta: DeltaMessage {
                        role: None,
                        content: Some(second_half),
                        tool_calls: None,
                    },
                    finish_reason: Some("stop".into()),
                }],
                usage: None,
            });
        };

        Ok(stream.boxed())
    }

    /// 图像生成：返固定 1 张占位图（1x1 PNG Base64）
    async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
        Ok(ImageResult {
            id: "echo-image-1".into(),
            object: "image.generation".into(),
            created: Self::now_secs(),
            model: req.model,
            data: vec![ImageData {
                url: None,
                b64_json: Some(Self::PLACEHOLDER_PNG_B64.into()),
                revised_prompt: Some(req.prompt),
            }],
        })
    }

    /// 创建视频任务：返固定已完成任务
    async fn video_create(&self, req: VideoRequest) -> Result<VideoTask> {
        Ok(VideoTask {
            task_id: "echo-task-1".into(),
            model: req.model,
            status: TaskStatus::Success,
            created_at: Self::now_secs(),
        })
    }

    /// 查询视频任务状态：返固定完成状态
    async fn video_poll(&self, task_id: &str, _model: &str) -> Result<VideoStatus> {
        Ok(VideoStatus {
            task_id: task_id.to_string(),
            status: TaskStatus::Success,
            video_url: Some("https://example.com/echo.mp4".into()),
            progress: Some(100),
            error: None,
            created_at: Some(Self::now_secs()),
            updated_at: Some(Self::now_secs()),
        })
    }

    /// 文本嵌入：输入 N 条文本返 N 个 3 维向量
    async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        let n = match &req.input {
            EmbedInput::Single(_) => 1,
            EmbedInput::Multiple(texts) => texts.len(),
        };
        // 每个向量用 3 维占位值，索引区分以验证数量对齐
        let data: Vec<EmbeddingItem> = (0..n)
            .map(|i| EmbeddingItem {
                object: "embedding".into(),
                index: i as u32,
                embedding: EmbeddingVector::Float(vec![
                    i as f64,
                    (i as f64) + 0.1,
                    (i as f64) + 0.2,
                ]),
            })
            .collect();
        Ok(EmbeddingResult {
            object: "list".into(),
            data,
            model: req.model,
            usage: Some(EmbeddingUsage {
                prompt_tokens: n as u64,
                total_tokens: n as u64,
            }),
        })
    }

    /// 语音转文字：返固定转写文本
    async fn transcribe(
        &self,
        _req: crate::model::audio::TranscribeRequest,
    ) -> Result<TranscriptionResult> {
        Ok(TranscriptionResult {
            text: "echo transcription".into(),
            language: Some("zh".into()),
            duration: Some(1.0),
            segments: None,
            words: None,
            task: "transcribe".into(),
            usage: None,
            model: Some("echo-asr".into()),
        })
    }

    /// 文字转语音：返固定二进制音频（验证跨语言二进制传递）
    async fn speech(&self, _req: crate::model::audio::SpeechRequest) -> Result<SpeechResult> {
        Ok(SpeechResult {
            audio_data: Some(b"mock-audio-data".to_vec()),
            audio_url: None,
            audio_base64: None,
            content_type: "audio/mpeg".into(),
            format: "mp3".into(),
            duration: Some(1.0),
            model: Some("echo-tts".into()),
        })
    }

    /// 模型列表：返固定 6 个 echo 模型
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = Self::fixed_models();
        match filter {
            Some(t) => Ok(models.into_iter().filter(|m| m.model_type == t).collect()),
            None => Ok(models),
        }
    }

    /// 音色列表：返固定 2 个 echo 音色
    async fn list_voices(&self, _language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        Ok(Self::fixed_voices())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::factory::create_adapter;
    use crate::config::{ClientOptions, ProviderConfig};
    use crate::model::audio::{SpeechRequest, TranscribeRequest};
    use crate::model::image::FileInput;

    /// 构造测试用 ProviderConfig（echo 免认证，但 config 仍可携带任意字段）
    fn echo_config() -> ProviderConfig {
        ProviderConfig::from_options("echo", ClientOptions::default())
    }

    #[tokio::test]
    async fn chat_echoes_last_user_message() {
        let adapter = EchoAdapter::new();
        let req = ChatRequest::builder(
            "echo-chat",
            vec![
                ChatMessage::system("you are helpful"),
                ChatMessage::user("hello world"),
            ],
        )
        .build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(resp.id, "echo-chat-1");
        assert_eq!(resp.model, "echo-chat");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("hello world [echo]")
        );
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn chat_echoes_multimodal_user_text() {
        let adapter = EchoAdapter::new();
        let req = ChatRequest::builder(
            "echo-chat",
            vec![ChatMessage::user_multimodal(vec![
                crate::model::chat::ContentPart::Text {
                    text: "describe this".into(),
                },
                crate::model::chat::ContentPart::ImageUrl {
                    image_url: crate::model::chat::ImageUrl::new("https://example.com/x.png"),
                },
            ])],
        )
        .build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("describe this [echo]")
        );
    }

    #[tokio::test]
    async fn chat_stream_produces_three_chunks() {
        let adapter = EchoAdapter::new();
        let req = ChatRequest::builder("echo-chat", vec![ChatMessage::user("hi")]).build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(chunks.len(), 3, "应产生 3 个 chunk");
        // 第 1 块含 role
        assert_eq!(
            chunks[0].choices[0].delta.role.as_deref(),
            Some("assistant")
        );
        // 拼接第 2、3 块内容应等于完整回显
        let mut assembled = String::new();
        assembled.push_str(chunks[1].choices[0].delta.content.as_deref().unwrap_or(""));
        assembled.push_str(chunks[2].choices[0].delta.content.as_deref().unwrap_or(""));
        assert_eq!(assembled, "hi [echo]");
        // 第 3 块标记结束
        assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn speech_returns_bytes() {
        let adapter = EchoAdapter::new();
        let req = SpeechRequest::builder("echo-tts", "hello", "echo-voice-1").build();
        let resp = adapter.speech(req).await.unwrap();
        assert_eq!(
            resp.audio_data.as_deref(),
            Some(b"mock-audio-data".as_slice())
        );
        assert_eq!(resp.content_type, "audio/mpeg");
        assert_eq!(resp.format, "mp3");
        assert_eq!(resp.model.as_deref(), Some("echo-tts"));
    }

    #[tokio::test]
    async fn list_models_returns_six_echo_models() {
        let adapter = EchoAdapter::new();
        let models = adapter.list_models(None).await.unwrap();
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "echo-chat",
                "echo-image",
                "echo-video",
                "echo-tts",
                "echo-asr",
                "echo-embed"
            ]
        );
    }

    #[tokio::test]
    async fn list_models_filter_by_type() {
        let adapter = EchoAdapter::new();
        let images = adapter.list_models(Some(ModelType::Image)).await.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "echo-image");

        let audios = adapter.list_models(Some(ModelType::Audio)).await.unwrap();
        assert_eq!(audios.len(), 2); // echo-tts + echo-asr
    }

    #[tokio::test]
    async fn image_generate_returns_placeholder_png() {
        let adapter = EchoAdapter::new();
        let req = ImageRequest::builder("echo-image", "a cat").build();
        let resp = adapter.image_generate(req).await.unwrap();
        assert_eq!(resp.data.len(), 1);
        assert!(resp.data[0].b64_json.is_some());
        // 应为合法 base64（可解码）
        assert!(crate::util::decode_base64(resp.data[0].b64_json.as_ref().unwrap()).is_ok());
    }

    #[tokio::test]
    async fn video_create_and_poll_roundtrip() {
        let adapter = EchoAdapter::new();
        let req = VideoRequest::builder("echo-video", "a cat walking").build();
        let task = adapter.video_create(req).await.unwrap();
        assert_eq!(task.task_id, "echo-task-1");
        assert_eq!(task.status, TaskStatus::Success);

        let status = adapter
            .video_poll("echo-task-1", "echo-video")
            .await
            .unwrap();
        assert_eq!(status.status, TaskStatus::Success);
        assert_eq!(
            status.video_url.as_deref(),
            Some("https://example.com/echo.mp4")
        );
        assert_eq!(status.progress, Some(100));
    }

    #[tokio::test]
    async fn embed_returns_n_vectors() {
        let adapter = EchoAdapter::new();
        let req = EmbedRequest {
            model: "echo-embed".into(),
            input: EmbedInput::Multiple(vec!["a".into(), "b".into(), "c".into()]),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: std::collections::HashMap::new(),
        };
        let resp = adapter.embed(req).await.unwrap();
        assert_eq!(resp.data.len(), 3);
        for (i, item) in resp.data.iter().enumerate() {
            assert_eq!(item.index, i as u32);
            if let EmbeddingVector::Float(v) = &item.embedding {
                assert_eq!(v.len(), 3, "每个向量应为 3 维");
            } else {
                panic!("应为 Float 向量");
            }
        }
    }

    #[tokio::test]
    async fn transcribe_returns_fixed_text() {
        let adapter = EchoAdapter::new();
        let req = TranscribeRequest::builder("echo-asr", FileInput::path("/tmp/a.mp3")).build();
        let resp = adapter.transcribe(req).await.unwrap();
        assert_eq!(resp.text, "echo transcription");
    }

    #[tokio::test]
    async fn list_voices_returns_two() {
        let adapter = EchoAdapter::new();
        let voices = adapter.list_voices(None).await.unwrap();
        assert_eq!(voices.len(), 2);
        assert_eq!(voices[0].short_name.as_deref(), Some("echo-voice-1"));
    }

    #[tokio::test]
    async fn recommend_voices_filters_by_gender() {
        let adapter = EchoAdapter::new();
        let female = adapter
            .recommend_voices(None, Some("female"), 10)
            .await
            .unwrap();
        assert_eq!(female.len(), 1);
        assert_eq!(female[0].gender.as_deref(), Some("Female"));
    }

    #[tokio::test]
    async fn requires_api_key_is_false() {
        let adapter = EchoAdapter::new();
        assert!(!adapter.requires_api_key());
    }

    #[tokio::test]
    async fn capabilities_contains_all() {
        let adapter = EchoAdapter::new();
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::ImageGenerate));
        assert!(caps.contains(&Capabilities::VideoGenerate));
        assert!(caps.contains(&Capabilities::Embedding));
        assert!(caps.contains(&Capabilities::AudioSpeech));
        assert!(caps.contains(&Capabilities::AudioTranscribe));
        assert!(caps.contains(&Capabilities::ListVoices));
    }

    #[test]
    fn factory_create_echo_succeeds() {
        let result = create_adapter(echo_config());
        assert!(result.is_ok(), "工厂应能创建 echo 适配器");
        let adapter = result.unwrap();
        assert_eq!(adapter.provider_type(), "echo");
        assert_eq!(adapter.provider_name(), "Echo (Mock)");
        assert!(!adapter.requires_api_key());
    }

    #[tokio::test]
    async fn start_and_close_are_noops() {
        let mut adapter = EchoAdapter::new();
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }
}
