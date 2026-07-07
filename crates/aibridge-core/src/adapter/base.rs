//! Adapter trait 与能力定义
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/base.py`。
//!
//! 设计要点（与设计文档 5.2 节一致）：
//! - `Adapter` trait 用 `#[async_trait]`
//! - 不支持的方法默认返 `UnsupportedCapability`，trait 提供默认实现
//! - `Capabilities` 为枚举（替代 Python 的字符串常量），更类型安全

use async_trait::async_trait;
use futures::stream::BoxStream;
use std::collections::HashSet;

use crate::config::ProviderConfig;
use crate::error::{AibridgeError, Result};
use crate::model::common::{ModelInfo, ModelType, VoiceInfo};
use crate::model::{
    ChatCompletion, ChatCompletionChunk, ChatRequest, EmbedRequest, EmbeddingResult, ImageRequest,
    ImageResult, SpeechRequest, SpeechResult, TranscribeRequest, TranscriptionResult, VideoRequest,
    VideoStatus, VideoTask,
};

/// 流式对话的类型别名
///
/// 用 `BoxStream` 封装 `impl Stream<Item = Result<ChatCompletionChunk>>`，
/// 便于跨 `dyn Adapter` 使用（无法直接返回 `impl Stream`）。
pub type ChatStream = BoxStream<'static, Result<ChatCompletionChunk>>;

/// 能力枚举
///
/// 对应 Python v1 `Capabilities` 字符串常量类。
/// 用枚举替代字符串，编译期保证能力名拼写正确。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capabilities {
    // 对话能力
    /// 文本对话
    Chat,
    /// 流式文本对话
    ChatStream,
    /// 视觉理解
    Vision,
    /// 工具调用
    ToolCall,
    /// 推理/思考模式
    Reasoning,
    /// JSON 模式
    JsonMode,
    /// 联网搜索
    WebSearch,

    // 图像能力
    /// 图像生成
    ImageGenerate,
    /// 图像编辑（图生图）
    ImageEdit,

    // 视频能力
    /// 视频生成
    VideoGenerate,
    /// 文生视频
    VideoText2Video,
    /// 图生视频
    VideoImage2Video,

    // 嵌入能力
    /// 文本嵌入
    Embedding,

    // 音频能力
    /// 语音转文字
    AudioTranscribe,
    /// 文字转语音
    AudioSpeech,
    /// 音色查询
    ListVoices,
}

impl Capabilities {
    /// 转为字符串标识（用于序列化/对照 Python v1）
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::ChatStream => "chat_stream",
            Self::Vision => "vision",
            Self::ToolCall => "tool_call",
            Self::Reasoning => "reasoning",
            Self::JsonMode => "json_mode",
            Self::WebSearch => "web_search",
            Self::ImageGenerate => "image",
            Self::ImageEdit => "image_edit",
            Self::VideoGenerate => "video",
            Self::VideoText2Video => "text2video",
            Self::VideoImage2Video => "image2video",
            Self::Embedding => "embedding",
            Self::AudioTranscribe => "audio_transcribe",
            Self::AudioSpeech => "audio_speech",
            Self::ListVoices => "list_voices",
        }
    }
}

/// 能力集合（用 HashSet 便于快速查找）
pub type CapabilitySet = HashSet<Capabilities>;

/// Adapter trait
///
/// 所有 Provider 适配器必须实现此 trait。
/// 适配器负责将统一接口转换为各 Provider 的特定 API 调用格式，
/// 并将响应归一化为统一的数据结构。
///
/// 不支持的方法默认返 `UnsupportedCapability`，子类按需 override。
#[async_trait]
pub trait Adapter: Send + Sync {
    /// Provider 类型标识（如 "agnes"、"openai"）
    fn provider_type(&self) -> &str;

    /// Provider 显示名称
    fn provider_name(&self) -> &str;

    /// 支持的能力集合
    fn capabilities(&self) -> CapabilitySet;

    /// 是否需要 API Key（免费 Provider 如 Edge TTS 返回 false）
    fn requires_api_key(&self) -> bool {
        true
    }

    /// 启动适配器（初始化 HTTP 客户端、连接池等资源）
    async fn start(&mut self) -> Result<()>;

    /// 关闭适配器（释放所有资源）
    async fn close(&mut self) -> Result<()>;

    /// 文本对话
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        let _ = req;
        Err(unsupported(self.provider_type(), Capabilities::Chat))
    }

    /// 流式文本对话
    ///
    /// 返回 `BoxStream<Item = Result<ChatCompletionChunk>>`。
    /// 默认实现返 `UnsupportedCapability`。
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        let _ = req;
        Err(unsupported(self.provider_type(), Capabilities::ChatStream))
    }

    /// 图像生成
    async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
        let _ = req;
        Err(unsupported(
            self.provider_type(),
            Capabilities::ImageGenerate,
        ))
    }

    /// 创建视频生成任务
    async fn video_create(&self, req: VideoRequest) -> Result<VideoTask> {
        let _ = req;
        Err(unsupported(
            self.provider_type(),
            Capabilities::VideoGenerate,
        ))
    }

    /// 查询视频任务状态
    async fn video_poll(&self, task_id: &str, model: &str) -> Result<VideoStatus> {
        let _ = task_id;
        let _ = model;
        Err(unsupported(
            self.provider_type(),
            Capabilities::VideoGenerate,
        ))
    }

    /// 文本嵌入
    async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        let _ = req;
        Err(unsupported(self.provider_type(), Capabilities::Embedding))
    }

    /// 语音转文字
    async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscriptionResult> {
        let _ = req;
        Err(unsupported(
            self.provider_type(),
            Capabilities::AudioTranscribe,
        ))
    }

    /// 文字转语音
    async fn speech(&self, req: SpeechRequest) -> Result<SpeechResult> {
        let _ = req;
        Err(unsupported(self.provider_type(), Capabilities::AudioSpeech))
    }

    /// 获取可用模型列表（实时拉取）
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>>;

    /// 列出可用音色
    async fn list_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        let _ = language;
        Err(unsupported(self.provider_type(), Capabilities::ListVoices))
    }

    /// 推荐可用音色（按语言/性别过滤）
    async fn recommend_voices(
        &self,
        language: Option<&str>,
        gender: Option<&str>,
        limit: usize,
    ) -> Result<Vec<VoiceInfo>> {
        let voices = self.list_voices(language).await?;
        let filtered: Vec<VoiceInfo> = match gender {
            Some(g) => {
                let g_lower = g.to_lowercase();
                voices
                    .into_iter()
                    .filter(|v| {
                        v.gender
                            .as_deref()
                            .map(|x| x.to_lowercase() == g_lower)
                            .unwrap_or(false)
                    })
                    .collect()
            }
            None => voices,
        };
        Ok(filtered.into_iter().take(limit).collect())
    }

    /// 检查是否支持指定能力
    fn supports_capability(&self, cap: Capabilities) -> bool {
        self.capabilities().contains(&cap)
    }

    /// 检查是否支持指定模型类型
    fn supports_model_type(&self, model_type: ModelType) -> bool {
        match model_type {
            ModelType::Chat => self.supports_capability(Capabilities::Chat),
            ModelType::Image => self.supports_capability(Capabilities::ImageGenerate),
            ModelType::Video => self.supports_capability(Capabilities::VideoGenerate),
            ModelType::Audio => {
                self.supports_capability(Capabilities::AudioTranscribe)
                    || self.supports_capability(Capabilities::AudioSpeech)
            }
        }
    }
}

/// 构造"不支持的能力"错误
fn unsupported(provider: &str, cap: Capabilities) -> AibridgeError {
    AibridgeError::UnsupportedCapability {
        capability: format!("{} (provider: {provider})", cap.as_str()),
    }
}

/// 从 `ProviderConfig` 构造适配器的工厂函数 trait
///
/// 具体适配器的构造逻辑各异，工厂返回 `Box<dyn Adapter>`。
/// 对应 Python v1 `AdapterFactory.create`。
pub type AdapterConstructor = fn(ProviderConfig) -> Result<Box<dyn Adapter>>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// 测试用的空适配器（仅实现必需方法，其余走默认实现）
    #[allow(dead_code)]
    struct DummyAdapter {
        config: ProviderConfig,
        started: bool,
        caps: CapabilitySet,
    }

    #[async_trait]
    impl Adapter for DummyAdapter {
        fn provider_type(&self) -> &str {
            "dummy"
        }
        fn provider_name(&self) -> &str {
            "Dummy"
        }
        fn capabilities(&self) -> CapabilitySet {
            self.caps.clone()
        }
        async fn start(&mut self) -> Result<()> {
            self.started = true;
            Ok(())
        }
        async fn close(&mut self) -> Result<()> {
            self.started = false;
            Ok(())
        }
        async fn list_models(&self, _filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
            Ok(vec![])
        }
    }

    fn dummy_config() -> ProviderConfig {
        ProviderConfig {
            provider_type: "dummy".into(),
            api_key: Some("k".into()),
            base_url: None,
            poll_url: None,
            timeout: 300,
            max_retries: 3,
            retry_delay: 2.0,
            enabled: true,
            resource_name: None,
            deployment_id: None,
            api_version: None,
            extra: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn default_chat_returns_unsupported_when_no_capability() {
        let mut adapter = DummyAdapter {
            config: dummy_config(),
            started: false,
            caps: CapabilitySet::new(),
        };
        adapter.start().await.unwrap();
        let req = ChatRequest::builder("m", vec![]).build();
        let result = adapter.chat(req).await;
        assert!(matches!(
            result,
            Err(AibridgeError::UnsupportedCapability { .. })
        ));
    }

    #[tokio::test]
    async fn default_speech_returns_unsupported() {
        let adapter = DummyAdapter {
            config: dummy_config(),
            started: true,
            caps: CapabilitySet::new(),
        };
        let req = SpeechRequest::builder("tts-1", "hi", "alloy").build();
        let result = adapter.speech(req).await;
        assert!(matches!(
            result,
            Err(AibridgeError::UnsupportedCapability { .. })
        ));
    }

    #[tokio::test]
    async fn default_video_poll_returns_unsupported() {
        let adapter = DummyAdapter {
            config: dummy_config(),
            started: true,
            caps: CapabilitySet::new(),
        };
        let result = adapter.video_poll("t-1", "m").await;
        assert!(matches!(
            result,
            Err(AibridgeError::UnsupportedCapability { .. })
        ));
    }

    #[tokio::test]
    async fn supports_capability_checks_set() {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::Chat);
        let adapter = DummyAdapter {
            config: dummy_config(),
            started: true,
            caps,
        };
        assert!(adapter.supports_capability(Capabilities::Chat));
        assert!(!adapter.supports_capability(Capabilities::Embedding));
    }

    #[tokio::test]
    async fn supports_model_type_maps_correctly() {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::ImageGenerate);
        let adapter = DummyAdapter {
            config: dummy_config(),
            started: true,
            caps,
        };
        assert!(adapter.supports_model_type(ModelType::Image));
        assert!(!adapter.supports_model_type(ModelType::Chat));
    }

    #[tokio::test]
    async fn recommend_voices_filters_by_gender() {
        #[allow(dead_code)]
        struct VoiceAdapter {
            config: ProviderConfig,
            caps: CapabilitySet,
        }
        #[async_trait]
        impl Adapter for VoiceAdapter {
            fn provider_type(&self) -> &str {
                "voice"
            }
            fn provider_name(&self) -> &str {
                "Voice"
            }
            fn capabilities(&self) -> CapabilitySet {
                self.caps.clone()
            }
            async fn start(&mut self) -> Result<()> {
                Ok(())
            }
            async fn close(&mut self) -> Result<()> {
                Ok(())
            }
            async fn list_models(&self, _: Option<ModelType>) -> Result<Vec<ModelInfo>> {
                Ok(vec![])
            }
            async fn list_voices(&self, _lang: Option<&str>) -> Result<Vec<VoiceInfo>> {
                Ok(vec![
                    VoiceInfo::builder()
                        .short_name("v1")
                        .gender("Female")
                        .build(),
                    VoiceInfo::builder().short_name("v2").gender("Male").build(),
                ])
            }
        }
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::ListVoices);
        let adapter = VoiceAdapter {
            config: dummy_config(),
            caps,
        };
        let female = adapter
            .recommend_voices(None, Some("female"), 10)
            .await
            .unwrap();
        assert_eq!(female.len(), 1);
        assert_eq!(female[0].short_name.as_deref(), Some("v1"));

        let limited = adapter.recommend_voices(None, None, 1).await.unwrap();
        assert_eq!(limited.len(), 1);
    }

    #[test]
    fn capabilities_as_str_matches_python_constants() {
        assert_eq!(Capabilities::Chat.as_str(), "chat");
        assert_eq!(Capabilities::ChatStream.as_str(), "chat_stream");
        assert_eq!(Capabilities::ImageGenerate.as_str(), "image");
        assert_eq!(Capabilities::VideoGenerate.as_str(), "video");
        assert_eq!(Capabilities::Embedding.as_str(), "embedding");
        assert_eq!(Capabilities::AudioTranscribe.as_str(), "audio_transcribe");
        assert_eq!(Capabilities::AudioSpeech.as_str(), "audio_speech");
        assert_eq!(Capabilities::ListVoices.as_str(), "list_voices");
    }

    #[tokio::test]
    async fn default_requires_api_key_true() {
        let adapter = DummyAdapter {
            config: dummy_config(),
            started: true,
            caps: CapabilitySet::new(),
        };
        assert!(adapter.requires_api_key());
    }
}
