//! 语音数据模型
//!
//! 定义语音转文字（ASR）与文字转语音（TTS）相关的 serde struct。
//! 对应 Python v1 (agn-sdk) 的 `agn/models/audio.py`。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::model::image::FileInput;

/// 语音转文字请求
///
/// 对应 Python v1 `TranscribeOptions` + 请求参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeRequest {
    /// 模型名称（如 "whisper-1"）
    pub model: String,
    /// 音频文件（路径、URL、base64 或二进制）
    pub file: FileInput,
    /// 语言代码（如 "zh"、"en"）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// 提示词（改善专有名词识别、纠正错别字）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// 响应格式（"json" / "text" / "srt" / "vtt" / "verbose_json"）
    #[serde(
        default = "default_transcribe_format",
        skip_serializing_if = "is_default_transcribe_format"
    )]
    pub response_format: String,
    /// 温度系数（0-1）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// 时间戳精度（"word" / "segment"）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub timestamp_granularities: Vec<String>,
    /// 是否翻译为英文（部分模型支持）
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub translate: bool,
    /// 厂商特有参数透传
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

impl TranscribeRequest {
    /// 创建 Builder
    pub fn builder(model: impl Into<String>, file: FileInput) -> TranscribeRequestBuilder {
        TranscribeRequestBuilder {
            inner: TranscribeRequest {
                model: model.into(),
                file,
                language: None,
                prompt: None,
                response_format: default_transcribe_format(),
                temperature: None,
                timestamp_granularities: Vec::new(),
                translate: false,
                extra: HashMap::new(),
            },
        }
    }
}

/// `TranscribeRequest` 的 Builder
#[derive(Debug, Clone)]
pub struct TranscribeRequestBuilder {
    inner: TranscribeRequest,
}

impl TranscribeRequestBuilder {
    pub fn language(mut self, l: impl Into<String>) -> Self {
        self.inner.language = Some(l.into());
        self
    }
    pub fn prompt(mut self, p: impl Into<String>) -> Self {
        self.inner.prompt = Some(p.into());
        self
    }
    pub fn response_format(mut self, r: impl Into<String>) -> Self {
        self.inner.response_format = r.into();
        self
    }
    pub fn temperature(mut self, t: f64) -> Self {
        self.inner.temperature = Some(t);
        self
    }
    pub fn timestamp_granularities(mut self, g: Vec<String>) -> Self {
        self.inner.timestamp_granularities = g;
        self
    }
    pub fn translate(mut self, t: bool) -> Self {
        self.inner.translate = t;
        self
    }
    pub fn extra(mut self, k: impl Into<String>, v: impl Into<serde_json::Value>) -> Self {
        self.inner.extra.insert(k.into(), v.into());
        self
    }
    pub fn build(self) -> TranscribeRequest {
        self.inner
    }
}

/// 转写结果
///
/// 对应 Python v1 `TranscriptionResult`。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TranscriptionResult {
    /// 完整转写文本
    pub text: String,
    /// 检测到的语言
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// 音频时长（秒）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    /// 分段信息
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<TranscriptionSegment>>,
    /// 词级时间戳
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<TranscriptionWord>>,
    /// 任务类型（transcribe / translate）
    #[serde(default = "default_task", skip_serializing_if = "is_default_task")]
    pub task: String,
    /// 使用统计
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
    /// 使用的模型 ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// 转写分段信息（带时间戳）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// 分段 ID
    pub id: u32,
    /// 开始时间（秒）
    pub start: f64,
    /// 结束时间（秒）
    pub end: f64,
    /// 分段文本
    pub text: String,
    /// 分段置信度（0-1）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// 说话人标识（说话人分离时使用）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

/// 转写词级时间戳信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionWord {
    /// 词文本
    pub word: String,
    /// 开始时间（秒）
    pub start: f64,
    /// 结束时间（秒）
    pub end: f64,
    /// 置信度（0-1）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

/// 文字转语音请求
///
/// 对应 Python v1 `SpeechOptions` + 请求参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRequest {
    /// 模型名称（如 "tts-1"、"tts-1-hd"）
    pub model: String,
    /// 要合成的文本
    pub input: String,
    /// 音色（单个或候选列表用于自动降级）
    pub voice: VoiceSpec,
    /// 音频输出格式（"mp3" / "opus" / "aac" / "flac" / "wav" / "pcm"）
    #[serde(
        default = "default_speech_format",
        skip_serializing_if = "is_default_speech_format"
    )]
    pub response_format: String,
    /// 语速（0.25-4.0，默认 1.0）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    /// 音量（0-2，默认 1.0）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
    /// 音调（-1 到 1，默认 0）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch: Option<f64>,
    /// 情感风格（如 "happy"、"sad"、"neutral"）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
    /// 说话风格
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    /// 厂商特有参数透传
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

impl SpeechRequest {
    /// 创建 Builder（单个音色）
    pub fn builder(
        model: impl Into<String>,
        input: impl Into<String>,
        voice: impl Into<String>,
    ) -> SpeechRequestBuilder {
        Self::builder_with_voices(model, input, vec![voice.into()])
    }

    /// 创建 Builder（候选音色列表，用于自动降级）
    pub fn builder_with_voices(
        model: impl Into<String>,
        input: impl Into<String>,
        voices: Vec<String>,
    ) -> SpeechRequestBuilder {
        SpeechRequestBuilder {
            inner: SpeechRequest {
                model: model.into(),
                input: input.into(),
                voice: VoiceSpec { voices },
                response_format: default_speech_format(),
                speed: None,
                volume: None,
                pitch: None,
                emotion: None,
                style: None,
                extra: HashMap::new(),
            },
        }
    }
}

/// 音色规格（支持候选列表用于自动降级）
///
/// 对应 Python v1 `speech` 的 `voice: str | list[str]` 参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSpec {
    /// 音色列表（至少 1 个；多个时启用 fallback 降级）
    pub voices: Vec<String>,
}

impl VoiceSpec {
    /// 单个音色
    pub fn single(v: impl Into<String>) -> Self {
        Self {
            voices: vec![v.into()],
        }
    }

    /// 候选音色列表
    pub fn multiple(voices: Vec<String>) -> Self {
        Self { voices }
    }

    /// 主音色（列表第一个）
    pub fn primary(&self) -> Option<&str> {
        self.voices.first().map(String::as_str)
    }
}

/// `SpeechRequest` 的 Builder
#[derive(Debug, Clone)]
pub struct SpeechRequestBuilder {
    inner: SpeechRequest,
}

impl SpeechRequestBuilder {
    pub fn response_format(mut self, r: impl Into<String>) -> Self {
        self.inner.response_format = r.into();
        self
    }
    pub fn speed(mut self, s: f64) -> Self {
        self.inner.speed = Some(s);
        self
    }
    pub fn volume(mut self, v: f64) -> Self {
        self.inner.volume = Some(v);
        self
    }
    pub fn pitch(mut self, p: f64) -> Self {
        self.inner.pitch = Some(p);
        self
    }
    pub fn emotion(mut self, e: impl Into<String>) -> Self {
        self.inner.emotion = Some(e.into());
        self
    }
    pub fn style(mut self, s: impl Into<String>) -> Self {
        self.inner.style = Some(s.into());
        self
    }
    pub fn extra(mut self, k: impl Into<String>, v: impl Into<serde_json::Value>) -> Self {
        self.inner.extra.insert(k.into(), v.into());
        self
    }
    pub fn build(self) -> SpeechRequest {
        self.inner
    }
}

/// 文字转语音结果
///
/// 对应 Python v1 `SpeechResult`。注意：`audio_data` 不参与 serde
/// （二进制数据通过 FFI 的 `aibridge_bytes_t` 单独传递），仅用于 core 内部。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpeechResult {
    /// 音频二进制数据（不序列化，FFI 单独传递）
    #[serde(skip)]
    pub audio_data: Option<Vec<u8>>,
    /// 音频 URL（部分 Provider 返回）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_url: Option<String>,
    /// 音频 Base64 编码
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_base64: Option<String>,
    /// 音频 MIME 类型
    #[serde(default = "default_content_type")]
    pub content_type: String,
    /// 音频格式（mp3/wav/opus 等）
    #[serde(default = "default_format")]
    pub format: String,
    /// 估计音频时长（秒）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    /// 使用的模型 ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl SpeechResult {
    /// 获取音频二进制数据（优先 audio_data，其次解码 audio_base64）
    pub fn get_audio_bytes(&self) -> Option<Vec<u8>> {
        if let Some(data) = &self.audio_data {
            return Some(data.clone());
        }
        if let Some(b64) = &self.audio_base64 {
            return crate::util::decode_base64(b64).ok();
        }
        None
    }
}

fn default_transcribe_format() -> String {
    "json".into()
}

fn is_default_transcribe_format(s: &str) -> bool {
    s == "json"
}

fn default_speech_format() -> String {
    "mp3".into()
}

fn is_default_speech_format(s: &str) -> bool {
    s == "mp3"
}

fn default_task() -> String {
    "transcribe".into()
}

fn is_default_task(s: &str) -> bool {
    s.is_empty() || s == "transcribe"
}

fn default_content_type() -> String {
    "audio/mpeg".into()
}

fn default_format() -> String {
    "mp3".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcribe_request_builder() {
        let req = TranscribeRequest::builder("whisper-1", FileInput::path("/tmp/a.mp3"))
            .language("zh")
            .response_format("verbose_json")
            .temperature(0.2)
            .build();
        assert_eq!(req.model, "whisper-1");
        assert_eq!(req.language.as_deref(), Some("zh"));
        assert_eq!(req.response_format, "verbose_json");
    }

    #[test]
    fn transcribe_request_skip_defaults() {
        let req = TranscribeRequest::builder("whisper-1", FileInput::path("/tmp/a.mp3")).build();
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("language"));
        assert!(!json.contains("prompt"));
        assert!(!json.contains("response_format")); // "json" 被跳过
        assert!(!json.contains("translate")); // false 被跳过
    }

    #[test]
    fn transcribe_request_translate_flag() {
        let req = TranscribeRequest::builder("whisper-1", FileInput::path("/tmp/a.mp3"))
            .translate(true)
            .build();
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"translate\":true"));
    }

    #[test]
    fn transcription_result_deserialize() {
        let json = serde_json::json!({
            "text": "hello world",
            "language": "en",
            "duration": 5.5,
            "task": "transcribe"
        });
        let r: TranscriptionResult = serde_json::from_value(json).unwrap();
        assert_eq!(r.text, "hello world");
        assert_eq!(r.language.as_deref(), Some("en"));
        assert!((r.duration.unwrap() - 5.5).abs() < f64::EPSILON);
    }

    #[test]
    fn transcription_result_default_task() {
        let r = TranscriptionResult {
            text: "hi".into(),
            ..Default::default()
        };
        let json = serde_json::to_string(&r).unwrap();
        // 默认 "transcribe" 被跳过
        assert!(!json.contains("task"));
    }

    #[test]
    fn speech_request_builder_single_voice() {
        let req = SpeechRequest::builder("tts-1", "hello", "alloy")
            .speed(1.5)
            .build();
        assert_eq!(req.model, "tts-1");
        assert_eq!(req.voice.primary(), Some("alloy"));
        assert_eq!(req.speed, Some(1.5));
    }

    #[test]
    fn speech_request_builder_voice_fallback() {
        let req = SpeechRequest::builder_with_voices(
            "edge-tts",
            "hello",
            vec!["zh-CN-XiaoxiaoNeural".into(), "zh-CN-YunxiNeural".into()],
        )
        .build();
        assert_eq!(req.voice.voices.len(), 2);
        assert_eq!(req.voice.primary(), Some("zh-CN-XiaoxiaoNeural"));
    }

    #[test]
    fn speech_request_skip_defaults() {
        let req = SpeechRequest::builder("tts-1", "hi", "alloy").build();
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("response_format")); // "mp3" 被跳过
        assert!(!json.contains("speed"));
    }

    #[test]
    fn speech_result_get_audio_bytes_from_data() {
        let r = SpeechResult {
            audio_data: Some(vec![1, 2, 3]),
            ..Default::default()
        };
        assert_eq!(r.get_audio_bytes(), Some(vec![1, 2, 3]));
    }

    #[test]
    fn speech_result_get_audio_bytes_from_base64() {
        let encoded = crate::util::encode_base64(b"hello");
        let r = SpeechResult {
            audio_base64: Some(encoded),
            ..Default::default()
        };
        assert_eq!(r.get_audio_bytes(), Some(b"hello".to_vec()));
    }

    #[test]
    fn speech_result_get_audio_bytes_none() {
        let r = SpeechResult::default();
        assert!(r.get_audio_bytes().is_none());
    }

    #[test]
    fn speech_result_audio_data_not_serialized() {
        let r = SpeechResult {
            audio_data: Some(vec![1, 2, 3]),
            ..Default::default()
        };
        let json = serde_json::to_string(&r).unwrap();
        // audio_data 被 skip，不应出现
        assert!(!json.contains("audio_data"));
    }

    #[test]
    fn voice_spec_single_and_multiple() {
        let s = VoiceSpec::single("alloy");
        assert_eq!(s.voices.len(), 1);
        let m = VoiceSpec::multiple(vec!["a".into(), "b".into()]);
        assert_eq!(m.voices.len(), 2);
    }
}
