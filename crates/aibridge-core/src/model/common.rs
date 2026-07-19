//! 通用数据模型
//!
//! 定义模型类型常量、模型信息、Provider 信息等通用数据结构。
//! 对应 Python v1 (agn-sdk) 的 `agn/models/common.py`。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// 模型类型
///
/// 对应 Python v1 `ModelType` 常量类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// 文本对话
    Chat,
    /// 图像生成
    Image,
    /// 视频生成
    Video,
    /// 音频（ASR/TTS）
    Audio,
}

impl ModelType {
    /// 转为字符串
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }
}

/// 从字符串解析模型类型
impl From<&str> for ModelType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "image" => Self::Image,
            "video" => Self::Video,
            "audio" => Self::Audio,
            _ => Self::Chat,
        }
    }
}

/// 视频生成模式
///
/// 对应 Python v1 `VideoMode` 常量类。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoMode {
    /// 文生视频
    #[default]
    Text2Video,
    /// 图生视频
    Image2Video,
    /// 视频生视频
    Video2Video,
    /// 关键帧模式
    Keyframes,
    /// 多图模式
    Multiimage,
}

/// 任务状态
///
/// 对应 Python v1 `TaskStatus` 常量类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    /// 排队中
    Pending,
    /// 处理中
    Processing,
    /// 成功
    Success,
    /// 失败
    Failed,
}

impl TaskStatus {
    /// 是否终态
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Success | Self::Failed)
    }
}

/// 模型信息
///
/// 描述单个 AI 模型的基本信息和能力。
/// 对应 Python v1 `ModelInfo`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// 模型标识符
    pub id: String,
    /// 模型显示名称
    pub name: String,
    /// 模型类型
    #[serde(rename = "type")]
    pub model_type: ModelType,
    /// 提供商名称
    pub provider: String,
    /// 支持的能力列表（如 "text2image"、"image2image"）
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// 最大 token 数（仅 chat 模型）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// 是否支持流式输出
    #[serde(default)]
    pub supports_streaming: bool,
    /// 模型描述
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 模型创建时间戳
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
}

/// Provider 信息
///
/// 描述单个 AI 模型提供商的元信息。
/// 对应 Python v1 `ProviderInfo`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// Provider 类型标识
    #[serde(rename = "type")]
    pub provider_type: String,
    /// Provider 显示名称
    pub name: String,
    /// Provider 描述
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Provider 官网
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub website: Option<String>,
    /// 支持的能力列表
    #[serde(default)]
    pub supported_capabilities: Vec<String>,
    /// 支持的模型类型
    #[serde(default)]
    pub supported_model_types: Vec<ModelType>,
}

/// 语音信息
///
/// 用于 TTS Provider 的音色描述（edge-tts / elevenlabs 等返回）。
/// 字段较为宽松，以适配不同 Provider 的音色元数据格式。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VoiceInfo {
    /// 音色短名（edge-tts 的 ShortName / elevenlabs 的 voice_id）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub short_name: Option<String>,
    /// 音色显示名
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// 语言区域（如 "zh-CN" / "en-US"）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locale: Option<String>,
    /// 性别（"Female" / "Male"）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gender: Option<String>,
    /// 音色 ID（部分 Provider 用）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<String>,
    /// 额外元数据
    #[serde(default, flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl VoiceInfo {
    /// 创建一个新的 Builder
    pub fn builder() -> VoiceInfoBuilder {
        VoiceInfoBuilder::default()
    }
}

/// `VoiceInfo` 的 Builder
#[derive(Debug, Default, Clone)]
pub struct VoiceInfoBuilder {
    inner: VoiceInfo,
}

impl VoiceInfoBuilder {
    pub fn short_name(mut self, v: impl Into<String>) -> Self {
        self.inner.short_name = Some(v.into());
        self
    }
    pub fn name(mut self, v: impl Into<String>) -> Self {
        self.inner.name = Some(v.into());
        self
    }
    pub fn locale(mut self, v: impl Into<String>) -> Self {
        self.inner.locale = Some(v.into());
        self
    }
    pub fn gender(mut self, v: impl Into<String>) -> Self {
        self.inner.gender = Some(v.into());
        self
    }
    pub fn voice_id(mut self, v: impl Into<String>) -> Self {
        self.inner.voice_id = Some(v.into());
        self
    }
    pub fn extra(mut self, k: impl Into<String>, v: impl Into<serde_json::Value>) -> Self {
        self.inner.extra.insert(k.into(), v.into());
        self
    }
    pub fn build(self) -> VoiceInfo {
        self.inner
    }
}

/// 模型类型推断关键字（用于从 /models 端点拉取的模型 ID 推断类型）
///
/// 对应 Python v1 `BaseAdapter._infer_type` 的关键字表。
pub fn infer_model_type(model_id: &str) -> ModelType {
    let lower = model_id.to_lowercase();
    let image_keywords = [
        "image",
        "flux",
        "sd3",
        "sdxl",
        "dall",
        "seedream",
        "wanx",
        "ideogram",
        "midjourney",
        "stable-diffusion",
        "imagen",
    ];
    let video_keywords = [
        "video",
        "veo",
        "seedance",
        "cogvideox",
        "wan",
        "kling",
        "runway",
        "pika",
        "luma",
        "sora",
        "vidu",
    ];
    let audio_keywords = [
        "whisper",
        "tts",
        "speech",
        "transcribe",
        "nova",
        "sonic",
        "edge-tts",
        "eleven",
        "cosyvoice",
        "sensevoice",
    ];
    if image_keywords.iter().any(|kw| lower.contains(kw)) {
        ModelType::Image
    } else if video_keywords.iter().any(|kw| lower.contains(kw)) {
        ModelType::Video
    } else if audio_keywords.iter().any(|kw| lower.contains(kw)) {
        ModelType::Audio
    } else {
        ModelType::Chat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_type_serde_lowercase() {
        let json = serde_json::to_string(&ModelType::Image).unwrap();
        assert_eq!(json, "\"image\"");
        let t: ModelType = serde_json::from_str("\"video\"").unwrap();
        assert_eq!(t, ModelType::Video);
    }

    #[test]
    fn model_type_from_str() {
        assert_eq!(ModelType::from("chat"), ModelType::Chat);
        assert_eq!(ModelType::from("IMAGE"), ModelType::Image);
        assert_eq!(ModelType::from("unknown"), ModelType::Chat);
    }

    #[test]
    fn model_type_as_str() {
        assert_eq!(ModelType::Audio.as_str(), "audio");
    }

    #[test]
    fn video_mode_default() {
        assert_eq!(VideoMode::default(), VideoMode::Text2Video);
    }

    #[test]
    fn task_status_is_terminal() {
        assert!(TaskStatus::Success.is_terminal());
        assert!(TaskStatus::Failed.is_terminal());
        assert!(!TaskStatus::Pending.is_terminal());
        assert!(!TaskStatus::Processing.is_terminal());
    }

    #[test]
    fn infer_model_type_image() {
        assert_eq!(infer_model_type("dall-e-3"), ModelType::Image);
        assert_eq!(infer_model_type("seedream-4.0"), ModelType::Image);
        assert_eq!(infer_model_type("FLUX-schnell"), ModelType::Image);
    }

    #[test]
    fn infer_model_type_video() {
        assert_eq!(infer_model_type("seedance-2.0"), ModelType::Video);
        assert_eq!(infer_model_type("kling-v1"), ModelType::Video);
        assert_eq!(infer_model_type("veo-3"), ModelType::Video);
    }

    #[test]
    fn infer_model_type_audio() {
        assert_eq!(infer_model_type("whisper-1"), ModelType::Audio);
        assert_eq!(infer_model_type("tts-1-hd"), ModelType::Audio);
        assert_eq!(infer_model_type("edge-tts"), ModelType::Audio);
    }

    #[test]
    fn infer_model_type_chat_default() {
        assert_eq!(infer_model_type("gpt-4o"), ModelType::Chat);
        assert_eq!(infer_model_type("claude-3-opus"), ModelType::Chat);
    }

    #[test]
    fn model_info_serialize() {
        let m = ModelInfo {
            id: "gpt-4o".into(),
            name: "GPT-4o".into(),
            model_type: ModelType::Chat,
            provider: "openai".into(),
            capabilities: vec!["chat".into()],
            max_tokens: Some(128000),
            supports_streaming: true,
            description: None,
            created: None,
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("\"type\":\"chat\""));
        assert!(json.contains("\"max_tokens\":128000"));
        // skip_serializing_if 生效：description/created 不出现
        assert!(!json.contains("description"));
        assert!(!json.contains("created"));
    }

    #[test]
    fn voice_info_builder() {
        let v = VoiceInfo::builder()
            .short_name("zh-CN-XiaoxiaoNeural")
            .name("Xiaoxiao")
            .locale("zh-CN")
            .gender("Female")
            .extra("wheels", 4)
            .build();
        assert_eq!(v.short_name.as_deref(), Some("zh-CN-XiaoxiaoNeural"));
        assert_eq!(v.gender.as_deref(), Some("Female"));
        assert_eq!(v.extra.get("wheels").and_then(|x| x.as_i64()), Some(4));
    }

    #[test]
    fn voice_info_flatten_extra() {
        let v = VoiceInfo {
            short_name: Some("v1".into()),
            ..Default::default()
        };
        let json = serde_json::json!({"short_name": "v1", "custom": "x"});
        let parsed: VoiceInfo = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.short_name.as_deref(), Some("v1"));
        assert_eq!(
            parsed.extra.get("custom").and_then(|x| x.as_str()),
            Some("x")
        );
        // 序列化回带 extra
        let _ = v;
    }
}
