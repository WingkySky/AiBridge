//! Deepgram 适配器（超低延迟语音识别 ASR）
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/audio_adapters.py` 的 `DeepgramAdapter`。
//!
//! Deepgram 是全球最快的语音识别(ASR)服务之一，Nova 系列模型以超低延迟和高准确率著称。
//!
//! ## 协议（独立协议，非 OpenAI 兼容）
//!
//! - Base URL: `https://api.deepgram.com/v1`
//! - 认证: `Authorization: Token <API_KEY>`（非 Bearer）
//! - transcribe: `POST /listen`
//!   - 音频以二进制 raw body 上传（`Content-Type` 为音频 MIME），同时通过 query param 传
//!     `model` / `smart_format` / `punctuate` / `language` / `diarize` / `profanity_filter` /
//!     `utterances` / `keywords` / `detect_language` 等参数
//!   - 远程 URL 输入：不传 body，改用 `url` query param（Deepgram 原生远程音频端点）
//!   - 响应: JSON，结构为 `{ results: { channels: [{ alternatives: [{ transcript, confidence,
//!     words, paragraphs }] }] }, metadata: { duration, model_info: { name, language } } }`
//! - list_models: 无标准 /models 端点，保留硬编码 ASR 模型列表
//! - 文档: https://developers.deepgram.com/reference/listen-file
//!
//! ## 错误映射
//!
//! - 401 → Authentication（API Key/Token 无效）
//! - 403 → Authentication（权限不足或账户停用）
//! - 429 → RateLimit（限流或配额耗尽）
//! - 400/422 → Validation（请求参数错误）
//! - 5xx → Api（服务端错误）
//! - 其余 4xx → Api
//!
//! ## 特性
//!
//! - `requires_api_key = true`（需 API Key）
//! - `capabilities` 仅 `AudioTranscribe`（transcribe + translate 均走 `transcribe()` 方法）
//! - Deepgram 不支持翻译为英文（`translate=true` 时返回 Validation 错误，与 Python v1 一致：
//!   Python 版 `DeepgramAdapter` 也未实现 translate）

use async_trait::async_trait;
use serde_json::Value;

use crate::adapter::{Adapter, Capabilities, CapabilitySet};
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::audio::{
    TranscribeRequest, TranscriptionResult, TranscriptionSegment, TranscriptionWord,
};
use crate::model::common::{ModelInfo, ModelType};
use crate::model::image::FileInput;

// ==================== 常量 ====================

/// Provider 类型标识
const PROVIDER_TYPE: &str = "deepgram";
/// Provider 显示名称
const PROVIDER_NAME: &str = "Deepgram";
/// 默认 API 基地址
const DEFAULT_API_BASE: &str = "https://api.deepgram.com/v1";
/// 预录音频转写端点路径
const LISTEN_PATH: &str = "/listen";
/// 默认模型（Nova-2 通用，Deepgram 推荐默认）
const DEFAULT_MODEL: &str = "nova-2";

// ==================== 音频输入解析 ====================

/// 解析后的音频输入形式
///
/// Deepgram 支持两种音频输入方式：
/// - 直接上传二进制 body（Bytes/Base64/Path）
/// - 远程 URL（通过 `url` query param，不传 body）
#[derive(Debug)]
enum AudioInput {
    /// 二进制音频 + Content-Type（直接作为请求 body 上传）
    Bytes(Vec<u8>, String),
    /// 远程 URL（用 `url` query param，不传 body）
    RemoteUrl(String),
}

// ==================== DeepgramAdapter ====================

/// Deepgram 适配器
///
/// 持有 HTTP 客户端与 API Key。所有请求按需发起（无长连接）。
pub struct DeepgramAdapter {
    /// Provider 配置
    #[allow(dead_code)]
    config: ProviderConfig,
    /// HTTP 客户端
    http: HttpClient,
    /// API Key（`Authorization: Token <key>` 用）
    api_key: Option<String>,
    /// API 基地址（默认 `https://api.deepgram.com/v1`，可由 config.base_url 覆盖）
    api_base: String,
}

impl DeepgramAdapter {
    /// 创建 Deepgram 适配器
    ///
    /// `config.base_url` 可覆盖 API 基地址（主要用于测试指向 mock server），
    /// 为空时用 `DEFAULT_API_BASE`。API Key 从 `config.api_key` 取。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let api_base = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_API_BASE.to_string());
        let api_key = config.api_key.clone();
        let opts = ClientOptions::builder().timeout(config.timeout).build();
        let http = HttpClient::new(&opts)?;
        Ok(Self {
            config,
            http,
            api_key,
            api_base,
        })
    }

    // ==================== 纯函数（协议构造/解析，可单测） ====================

    /// 构造 /listen 端点完整 URL（`{api_base}/listen`）
    fn build_listen_url(api_base: &str) -> String {
        format!(
            "{base}{path}",
            base = api_base.trim_end_matches('/'),
            path = LISTEN_PATH,
        )
    }

    /// 从文件扩展名推断音频 MIME 类型
    ///
    /// 对应 Python v1 `get_audio_bytes` 的 `mime_map`。未知扩展名回退 `audio/wav`。
    fn mime_from_extension(ext: &str) -> &'static str {
        match ext.to_lowercase().as_str() {
            "mp3" => "audio/mpeg",
            "wav" => "audio/wav",
            "ogg" => "audio/ogg",
            "flac" => "audio/flac",
            "m4a" | "mp4" => "audio/mp4",
            "webm" => "audio/webm",
            "aac" => "audio/aac",
            "opus" => "audio/opus",
            _ => "audio/wav",
        }
    }

    /// 从 `extra` 读取布尔参数（支持 bool 或字符串 "true"/"false"），未设置返回 `default`
    ///
    /// Deepgram 特有参数（smart_format/punctuate/diarize 等）通过 `extra` 透传，
    /// 此函数统一处理 bool 与字符串两种形式。
    fn get_bool_extra(
        extra: &std::collections::HashMap<String, Value>,
        key: &str,
        default: bool,
    ) -> bool {
        match extra.get(key) {
            Some(Value::Bool(b)) => *b,
            Some(Value::String(s)) => matches!(s.to_lowercase().as_str(), "true" | "1"),
            _ => default,
        }
    }

    /// 构造 Deepgram 查询参数列表
    ///
    /// 从 `TranscribeRequest` 提取 Deepgram 特有参数：
    /// - `model`（必传，使用已解析的 model，空则由调用方填默认值）
    /// - `smart_format`（默认 true）
    /// - `punctuate`（默认 true）
    /// - `language`（请求显式设置时传）
    /// - `diarize`（默认 false，true 时传 "true"）
    /// - `profanity_filter`（默认 false）
    /// - `utterances`（默认 false）
    /// - `detect_language`（默认 false）
    /// - `keywords`（列表，每个元素作为一个重复的 `keywords` 参数）
    ///
    /// 与 Python v1 一致：布尔参数仅在为 true 时加入 query（false 省略，依赖服务端默认）。
    fn build_query_params(req: &TranscribeRequest, model: &str) -> Vec<(String, String)> {
        let mut params: Vec<(String, String)> = Vec::new();
        params.push(("model".to_string(), model.to_string()));

        if Self::get_bool_extra(&req.extra, "smart_format", true) {
            params.push(("smart_format".to_string(), "true".to_string()));
        }
        if Self::get_bool_extra(&req.extra, "punctuate", true) {
            params.push(("punctuate".to_string(), "true".to_string()));
        }
        if let Some(lang) = &req.language {
            if !lang.is_empty() {
                params.push(("language".to_string(), lang.clone()));
            }
        }
        if Self::get_bool_extra(&req.extra, "diarize", false) {
            params.push(("diarize".to_string(), "true".to_string()));
        }
        if Self::get_bool_extra(&req.extra, "profanity_filter", false) {
            params.push(("profanity_filter".to_string(), "true".to_string()));
        }
        if Self::get_bool_extra(&req.extra, "utterances", false) {
            params.push(("utterances".to_string(), "true".to_string()));
        }
        if Self::get_bool_extra(&req.extra, "detect_language", false) {
            params.push(("detect_language".to_string(), "true".to_string()));
        }
        // keywords：列表，每个关键词作为一个重复的 query 参数
        if let Some(keywords) = req.extra.get("keywords").and_then(|v| v.as_array()) {
            for kw in keywords {
                if let Some(s) = kw.as_str() {
                    params.push(("keywords".to_string(), s.to_string()));
                }
            }
        }

        params
    }

    /// 解析 `FileInput` 为 Deepgram 要求的音频输入形式
    ///
    /// - `Bytes` → 直接 body（Content-Type 默认 `audio/wav`）
    /// - `Base64` → 解码后 body（Content-Type 默认 `audio/wav`）
    /// - `Path` → 读取文件后 body（按扩展名推断 Content-Type）；文件不存在 → Validation 错误
    /// - `Url` → 用 `url` query param（不传 body）
    async fn resolve_audio_input(file: &FileInput) -> Result<AudioInput> {
        match file {
            FileInput::Bytes(b) => Ok(AudioInput::Bytes(b.clone(), "audio/wav".to_string())),
            FileInput::Base64(s) => {
                let bytes = crate::util::decode_base64(s)
                    .map_err(|e| AibridgeError::validation(format!("Base64 音频解码失败: {e}")))?;
                Ok(AudioInput::Bytes(bytes, "audio/wav".to_string()))
            }
            FileInput::Path(p) => {
                let path = std::path::Path::new(p);
                if !path.exists() {
                    return Err(AibridgeError::validation(format!("音频文件不存在: {p}")));
                }
                let bytes = tokio::fs::read(path)
                    .await
                    .map_err(|e| AibridgeError::validation(format!("读取音频文件失败: {e}")))?;
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                let mime = Self::mime_from_extension(ext).to_string();
                Ok(AudioInput::Bytes(bytes, mime))
            }
            FileInput::Url(u) => Ok(AudioInput::RemoteUrl(u.clone())),
        }
    }

    /// 解析 Deepgram 响应为统一 `TranscriptionResult`
    ///
    /// Deepgram 响应结构：
    /// ```json
    /// {
    ///   "results": {
    ///     "channels": [{
    ///       "alternatives": [{
    ///         "transcript": "...",
    ///         "confidence": 0.99,
    ///         "words": [{ "word", "start", "end", "confidence" }],
    ///         "paragraphs": { "paragraphs": [{ "speaker", "sentences": [{ "text", "start", "end" }] }] }
    ///       }]
    ///     }]
    ///   },
    ///   "metadata": { "duration": 3.5, "model_info": { "name": "nova-2-general", "language": "en" } }
    /// }
    /// ```
    ///
    /// - `text`：拼接所有 channel 的 `alternatives[0].transcript`（空格分隔）
    /// - `language`：取自 `metadata.model_info.language`
    /// - `duration`：取自 `metadata.duration`
    /// - `words`：合并所有 channel 的 `alternatives[0].words`
    /// - `segments`：从 `alternatives[0].paragraphs.paragraphs[].sentences[]` 提取，
    ///   带说话人信息（`paragraph.speaker`）
    fn parse_deepgram_response(result: &Value, model: &str) -> TranscriptionResult {
        let results = result.get("results").unwrap_or(&Value::Null);
        let channels = results.get("channels").and_then(|c| c.as_array());

        let mut full_text_parts: Vec<String> = Vec::new();
        let mut all_words: Vec<TranscriptionWord> = Vec::new();
        let mut all_segments: Vec<TranscriptionSegment> = Vec::new();

        let mut language: Option<String> = None;
        let mut duration: Option<f64> = None;

        // metadata：duration + model_info.language
        if let Some(metadata) = result.get("metadata") {
            duration = metadata.get("duration").and_then(|d| d.as_f64());
            if let Some(model_info) = metadata.get("model_info") {
                language = model_info
                    .get("language")
                    .and_then(|l| l.as_str())
                    .map(str::to_owned);
            }
        }

        if let Some(channels) = channels {
            for channel in channels {
                let Some(alts) = channel.get("alternatives").and_then(|a| a.as_array()) else {
                    continue;
                };
                let Some(alt) = alts.first() else {
                    continue;
                };

                // transcript
                if let Some(transcript) = alt.get("transcript").and_then(|t| t.as_str()) {
                    if !transcript.is_empty() {
                        full_text_parts.push(transcript.to_string());
                    }
                }

                // words
                if let Some(words) = alt.get("words").and_then(|w| w.as_array()) {
                    for w in words {
                        let word = w
                            .get("word")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string();
                        let start = w.get("start").and_then(|x| x.as_f64()).unwrap_or(0.0);
                        let end = w.get("end").and_then(|x| x.as_f64()).unwrap_or(0.0);
                        let confidence = w.get("confidence").and_then(|x| x.as_f64());
                        all_words.push(TranscriptionWord {
                            word,
                            start,
                            end,
                            confidence,
                        });
                    }
                }

                // paragraphs → sentences → segments
                if let Some(paras) = alt
                    .get("paragraphs")
                    .and_then(|p| p.get("paragraphs"))
                    .and_then(|p| p.as_array())
                {
                    for para in paras {
                        let speaker = para
                            .get("speaker")
                            .and_then(|s| s.as_i64())
                            .map(|s| s.to_string());
                        if let Some(sentences) = para.get("sentences").and_then(|s| s.as_array()) {
                            for sent in sentences {
                                let id = all_segments.len() as u32;
                                let start =
                                    sent.get("start").and_then(|x| x.as_f64()).unwrap_or(0.0);
                                let end = sent.get("end").and_then(|x| x.as_f64()).unwrap_or(0.0);
                                let text = sent
                                    .get("text")
                                    .and_then(|x| x.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                all_segments.push(TranscriptionSegment {
                                    id,
                                    start,
                                    end,
                                    text,
                                    confidence: None,
                                    speaker: speaker.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        let full_text = full_text_parts.join(" ");
        TranscriptionResult {
            text: full_text,
            language,
            duration,
            segments: if all_segments.is_empty() {
                None
            } else {
                Some(all_segments)
            },
            words: if all_words.is_empty() {
                None
            } else {
                Some(all_words)
            },
            task: "transcribe".to_string(),
            usage: None,
            model: Some(model.to_string()),
        }
    }

    /// 将 Deepgram HTTP 错误响应映射为 `AibridgeError`
    ///
    /// 映射规则见模块文档"错误映射"小节。
    fn map_deepgram_error(status: u16, body: &str) -> AibridgeError {
        match status {
            401 => AibridgeError::authentication(format!("Deepgram API key (Token) 无效: {body}")),
            403 => AibridgeError::authentication(format!(
                "Deepgram API key 权限不足或账户已停用: {body}"
            )),
            429 => AibridgeError::rate_limit(format!("Deepgram 限流或配额耗尽: {body}")),
            400 | 422 => AibridgeError::validation(format!("Deepgram 请求参数错误: {body}")),
            s if s >= 500 => AibridgeError::api(s, format!("Deepgram 服务错误 ({s}): {body}")),
            s => AibridgeError::api(s, format!("Deepgram HTTP {s}: {body}")),
        }
    }

    /// 获取 API Key，为空时返 Validation 错误
    fn require_api_key(&self) -> Result<&str> {
        self.api_key
            .as_deref()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| AibridgeError::validation("Deepgram 需要 API key（Token）"))
    }
}

#[async_trait]
impl Adapter for DeepgramAdapter {
    fn provider_type(&self) -> &str {
        PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    fn capabilities(&self) -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::AudioTranscribe);
        caps
    }

    /// 需 API Key 认证
    fn requires_api_key(&self) -> bool {
        true
    }

    async fn start(&mut self) -> Result<()> {
        // HTTP 客户端在 new 时已建好，保持 no-op
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        // 无长连接资源需释放
        Ok(())
    }

    /// 语音转文字（Deepgram ASR）
    ///
    /// - `translate=true` 时返回 Validation 错误（Deepgram 不支持翻译）
    /// - 二进制音频直接作为 body 上传；URL 输入用 `url` query param
    async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscriptionResult> {
        // Deepgram 不支持翻译为英文（Python v1 也未实现 translate）
        if req.translate {
            return Err(AibridgeError::validation(
                "Deepgram 不支持翻译为英文（translate=true），仅支持语音转文字（transcribe）",
            ));
        }

        let api_key = self.require_api_key()?;

        // 模型缺省回退到默认模型
        let model = if req.model.is_empty() {
            DEFAULT_MODEL.to_string()
        } else {
            req.model.clone()
        };

        // 解析音频输入（可能读文件/解码 base64，异步）
        let audio_input = Self::resolve_audio_input(&req.file).await?;

        // 构造查询参数（model 用已解析的非空值）
        let mut params = Self::build_query_params(&req, &model);
        // URL 输入：追加 url query param，不传 body
        if let AudioInput::RemoteUrl(u) = &audio_input {
            params.push(("url".to_string(), u.clone()));
        }

        let url = Self::build_listen_url(&self.api_base);

        // 构造请求：Authorization: Token <key>，二进制 body（若有）
        let mut request = self
            .http
            .inner()
            .post(&url)
            .header("Authorization", format!("Token {api_key}"));

        if let AudioInput::Bytes(bytes, content_type) = &audio_input {
            request = request
                .header("Content-Type", content_type.as_str())
                .body(bytes.clone());
        }
        request = request.query(&params);

        let resp = request.send().await.map_err(map_reqwest_err)?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(Self::map_deepgram_error(status.as_u16(), &body));
        }

        let body_bytes = resp.bytes().await.map_err(map_reqwest_err)?;
        let result: Value = serde_json::from_slice(&body_bytes)
            .map_err(|e| AibridgeError::validation(format!("Deepgram 响应解析失败: {e}")))?;

        Ok(Self::parse_deepgram_response(&result, &model))
    }

    /// 列出 Deepgram ASR 模型（硬编码列表，无 /models 拉取需求）
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = deepgram_models();
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }
}

// ==================== 辅助函数 ====================

/// 将 reqwest::Error 映射为 AibridgeError（超时 → Timeout，其余 → Network）
fn map_reqwest_err(e: reqwest::Error) -> AibridgeError {
    if e.is_timeout() {
        AibridgeError::Timeout
    } else {
        AibridgeError::Network(e)
    }
}

/// Deepgram ASR 模型硬编码列表
///
/// 对应 Python v1 `DeepgramAdapter.list_models`。含 Nova-3/Nova-2 全系列 +
/// Whisper 托管版 + 旧版 Enhanced/Base。
fn deepgram_models() -> Vec<ModelInfo> {
    fn m(id: &str, name: &str, desc: &str) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            name: name.into(),
            model_type: ModelType::Audio,
            provider: PROVIDER_TYPE.into(),
            capabilities: vec!["audio_transcribe".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some(desc.into()),
            created: None,
        }
    }
    vec![
        m("nova-3", "Nova 3", "Nova 3 最新通用模型，最高准确率"),
        m("nova-2", "Nova 2", "Nova 2 通用模型，推荐默认使用"),
        m("nova-2-general", "Nova 2 General", "Nova 2 通用场景"),
        m(
            "nova-2-meeting",
            "Nova 2 Meeting",
            "Nova 2 会议场景优化（多人对话）",
        ),
        m(
            "nova-2-phonecall",
            "Nova 2 Phone Call",
            "Nova 2 电话通话优化（8kHz 音频）",
        ),
        m(
            "nova-2-conversationalai",
            "Nova 2 Conversational AI",
            "Nova 2 对话 AI/语音助手优化",
        ),
        m(
            "nova-2-video",
            "Nova 2 Video",
            "Nova 2 视频/播客/多说话人场景",
        ),
        m("nova-2-medical", "Nova 2 Medical", "Nova 2 医疗领域优化"),
        m("nova-2-finance", "Nova 2 Finance", "Nova 2 金融领域优化"),
        m(
            "nova-2-drivethru",
            "Nova 2 Drive-Thru",
            "Nova 2 餐厅免下车窗口优化",
        ),
        m(
            "whisper-large",
            "Whisper Large (Deepgram)",
            "OpenAI Whisper Large 托管版",
        ),
        m(
            "whisper-medium",
            "Whisper Medium (Deepgram)",
            "OpenAI Whisper Medium 托管版",
        ),
        m(
            "whisper-small",
            "Whisper Small (Deepgram)",
            "OpenAI Whisper Small 托管版",
        ),
        m(
            "enhanced",
            "Enhanced",
            "Enhanced 增强模型（旧版，兼容使用）",
        ),
        m("base", "Base", "Base 基础模型（最快、成本最低）"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientOptions;
    use crate::model::audio::SpeechRequest;
    use crate::model::chat::ChatRequest;
    use crate::model::image::{FileInput, ImageRequest};
    use crate::model::options::{EmbedInput, EmbedRequest};
    use crate::model::video::VideoRequest;
    use std::collections::HashMap;

    /// 构造测试用适配器（base_url 指向 mockito server，api_key 可选）
    fn make_adapter(base_url: Option<String>, api_key: Option<&str>) -> DeepgramAdapter {
        let mut opts = ClientOptions::builder();
        if let Some(u) = base_url {
            opts = opts.base_url(u);
        }
        if let Some(k) = api_key {
            opts = opts.api_key(k);
        }
        let config = ProviderConfig::from_options(PROVIDER_TYPE, opts.build());
        DeepgramAdapter::new(config).expect("构造 DeepgramAdapter 失败")
    }

    /// 构造一个最小 Deepgram 成功响应 JSON
    fn sample_response() -> Value {
        serde_json::json!({
            "results": {
                "channels": [{
                    "alternatives": [{
                        "transcript": "hello world",
                        "confidence": 0.99,
                        "words": [
                            {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.98},
                            {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.97}
                        ]
                    }]
                }]
            },
            "metadata": {
                "duration": 1.0,
                "model_info": {"name": "nova-2-general", "language": "en"}
            }
        })
    }

    // ============ 基本属性 ============

    #[test]
    fn requires_api_key_is_true() {
        let adapter = make_adapter(None, Some("test-key"));
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn capabilities_contains_transcribe_only() {
        let adapter = make_adapter(None, Some("test-key"));
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::AudioTranscribe));
        assert!(!caps.contains(&Capabilities::Chat));
        assert!(!caps.contains(&Capabilities::ImageGenerate));
        assert!(!caps.contains(&Capabilities::AudioSpeech));
        assert!(!caps.contains(&Capabilities::ListVoices));
    }

    #[test]
    fn provider_type_and_name() {
        let adapter = make_adapter(None, Some("test-key"));
        assert_eq!(adapter.provider_type(), "deepgram");
        assert_eq!(adapter.provider_name(), "Deepgram");
    }

    #[tokio::test]
    async fn start_and_close_are_noops() {
        let mut adapter = make_adapter(None, Some("test-key"));
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    // ============ list_models ============

    #[tokio::test]
    async fn list_models_returns_deepgram_models() {
        let adapter = make_adapter(None, Some("test-key"));
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 15);
        assert_eq!(models[0].id, "nova-3");
        assert_eq!(models[1].id, "nova-2");
        assert_eq!(models[0].model_type, ModelType::Audio);
        assert_eq!(models[0].provider, "deepgram");
    }

    #[tokio::test]
    async fn list_models_filter_by_audio() {
        let adapter = make_adapter(None, Some("test-key"));
        let audio = adapter.list_models(Some(ModelType::Audio)).await.unwrap();
        assert_eq!(audio.len(), 15);
        let chat = adapter.list_models(Some(ModelType::Chat)).await.unwrap();
        assert!(chat.is_empty());
    }

    // ============ 不支持能力（默认实现） ============

    #[tokio::test]
    async fn chat_returns_unsupported() {
        let adapter = make_adapter(None, Some("test-key"));
        let req = ChatRequest::builder("m", vec![]).build();
        assert!(matches!(
            adapter.chat(req).await.unwrap_err(),
            AibridgeError::UnsupportedCapability { .. }
        ));
    }

    #[tokio::test]
    async fn image_generate_returns_unsupported() {
        let adapter = make_adapter(None, Some("test-key"));
        let req = ImageRequest::builder("m", "p").build();
        assert!(matches!(
            adapter.image_generate(req).await.unwrap_err(),
            AibridgeError::UnsupportedCapability { .. }
        ));
    }

    #[tokio::test]
    async fn video_create_returns_unsupported() {
        let adapter = make_adapter(None, Some("test-key"));
        let req = VideoRequest::builder("m", "p").build();
        assert!(matches!(
            adapter.video_create(req).await.unwrap_err(),
            AibridgeError::UnsupportedCapability { .. }
        ));
    }

    #[tokio::test]
    async fn embed_returns_unsupported() {
        let adapter = make_adapter(None, Some("test-key"));
        let req = EmbedRequest {
            model: "m".into(),
            input: EmbedInput::Single("hi".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        assert!(matches!(
            adapter.embed(req).await.unwrap_err(),
            AibridgeError::UnsupportedCapability { .. }
        ));
    }

    #[tokio::test]
    async fn speech_returns_unsupported() {
        let adapter = make_adapter(None, Some("test-key"));
        let req = SpeechRequest::builder("aura", "hi", "voice").build();
        assert!(matches!(
            adapter.speech(req).await.unwrap_err(),
            AibridgeError::UnsupportedCapability { .. }
        ));
    }

    // ============ build_listen_url ============

    #[test]
    fn build_listen_url_is_correct() {
        let url = DeepgramAdapter::build_listen_url("https://api.deepgram.com/v1");
        assert_eq!(url, "https://api.deepgram.com/v1/listen");
    }

    #[test]
    fn build_listen_url_strips_trailing_slash() {
        let url = DeepgramAdapter::build_listen_url("https://api.deepgram.com/v1/");
        assert_eq!(url, "https://api.deepgram.com/v1/listen");
    }

    // ============ mime_from_extension ============

    #[test]
    fn mime_from_extension_known() {
        assert_eq!(DeepgramAdapter::mime_from_extension("mp3"), "audio/mpeg");
        assert_eq!(DeepgramAdapter::mime_from_extension("wav"), "audio/wav");
        assert_eq!(DeepgramAdapter::mime_from_extension("ogg"), "audio/ogg");
        assert_eq!(DeepgramAdapter::mime_from_extension("flac"), "audio/flac");
        assert_eq!(DeepgramAdapter::mime_from_extension("m4a"), "audio/mp4");
        assert_eq!(DeepgramAdapter::mime_from_extension("mp4"), "audio/mp4");
        assert_eq!(DeepgramAdapter::mime_from_extension("webm"), "audio/webm");
        assert_eq!(DeepgramAdapter::mime_from_extension("aac"), "audio/aac");
        assert_eq!(DeepgramAdapter::mime_from_extension("opus"), "audio/opus");
    }

    #[test]
    fn mime_from_extension_unknown_defaults_wav() {
        assert_eq!(DeepgramAdapter::mime_from_extension("xyz"), "audio/wav");
        assert_eq!(DeepgramAdapter::mime_from_extension(""), "audio/wav");
    }

    #[test]
    fn mime_from_extension_case_insensitive() {
        assert_eq!(DeepgramAdapter::mime_from_extension("MP3"), "audio/mpeg");
        assert_eq!(DeepgramAdapter::mime_from_extension("WAV"), "audio/wav");
    }

    // ============ get_bool_extra ============

    #[test]
    fn get_bool_extra_default_when_absent() {
        let extra = HashMap::new();
        assert!(DeepgramAdapter::get_bool_extra(
            &extra,
            "smart_format",
            true
        ));
        assert!(!DeepgramAdapter::get_bool_extra(&extra, "diarize", false));
    }

    #[test]
    fn get_bool_extra_reads_bool() {
        let mut extra = HashMap::new();
        extra.insert("smart_format".to_string(), serde_json::json!(false));
        assert!(!DeepgramAdapter::get_bool_extra(
            &extra,
            "smart_format",
            true
        ));
    }

    #[test]
    fn get_bool_extra_reads_string_true() {
        let mut extra = HashMap::new();
        extra.insert("diarize".to_string(), serde_json::json!("true"));
        assert!(DeepgramAdapter::get_bool_extra(&extra, "diarize", false));
        extra.insert("diarize".to_string(), serde_json::json!("TRUE"));
        assert!(DeepgramAdapter::get_bool_extra(&extra, "diarize", false));
    }

    #[test]
    fn get_bool_extra_reads_string_false() {
        let mut extra = HashMap::new();
        extra.insert("smart_format".to_string(), serde_json::json!("false"));
        assert!(!DeepgramAdapter::get_bool_extra(
            &extra,
            "smart_format",
            true
        ));
    }

    // ============ build_query_params ============

    #[test]
    fn build_query_params_defaults_smart_format_and_punctuate() {
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1, 2, 3])).build();
        let params = DeepgramAdapter::build_query_params(&req, "nova-2");
        // model 总是存在
        assert!(params.contains(&("model".to_string(), "nova-2".to_string())));
        // smart_format / punctuate 默认 true
        assert!(params.contains(&("smart_format".to_string(), "true".to_string())));
        assert!(params.contains(&("punctuate".to_string(), "true".to_string())));
        // 默认 false 的不出现
        assert!(!params.iter().any(|(k, _)| k == "diarize"));
        assert!(!params.iter().any(|(k, _)| k == "profanity_filter"));
        assert!(!params.iter().any(|(k, _)| k == "utterances"));
        assert!(!params.iter().any(|(k, _)| k == "detect_language"));
        assert!(!params.iter().any(|(k, _)| k == "language"));
    }

    #[test]
    fn build_query_params_disables_smart_format() {
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1]))
            .extra("smart_format", false)
            .extra("punctuate", false)
            .build();
        let params = DeepgramAdapter::build_query_params(&req, "nova-2");
        assert!(!params.iter().any(|(k, _)| k == "smart_format"));
        assert!(!params.iter().any(|(k, _)| k == "punctuate"));
    }

    #[test]
    fn build_query_params_includes_language() {
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1]))
            .language("zh")
            .build();
        let params = DeepgramAdapter::build_query_params(&req, "nova-2");
        assert!(params.contains(&("language".to_string(), "zh".to_string())));
    }

    #[test]
    fn build_query_params_enables_optional_flags() {
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1]))
            .extra("diarize", true)
            .extra("profanity_filter", true)
            .extra("utterances", true)
            .extra("detect_language", true)
            .build();
        let params = DeepgramAdapter::build_query_params(&req, "nova-2");
        assert!(params.contains(&("diarize".to_string(), "true".to_string())));
        assert!(params.contains(&("profanity_filter".to_string(), "true".to_string())));
        assert!(params.contains(&("utterances".to_string(), "true".to_string())));
        assert!(params.contains(&("detect_language".to_string(), "true".to_string())));
    }

    #[test]
    fn build_query_params_keywords_repeated() {
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1]))
            .extra("keywords", serde_json::json!(["foo:2", "bar:1"]))
            .build();
        let params = DeepgramAdapter::build_query_params(&req, "nova-2");
        // keywords 出现两次（重复 query 参数）
        let kw_values: Vec<String> = params
            .iter()
            .filter(|(k, _)| k == "keywords")
            .map(|(_, v)| v.clone())
            .collect();
        assert_eq!(kw_values.len(), 2);
        assert!(kw_values.contains(&"foo:2".to_string()));
        assert!(kw_values.contains(&"bar:1".to_string()));
    }

    #[test]
    fn build_query_params_uses_resolved_model() {
        // 空 model 的请求，调用方传入解析后的默认模型
        let req = TranscribeRequest::builder("", FileInput::bytes(vec![1])).build();
        let params = DeepgramAdapter::build_query_params(&req, DEFAULT_MODEL);
        assert!(params.contains(&("model".to_string(), "nova-2".to_string())));
    }

    // ============ resolve_audio_input ============

    #[tokio::test]
    async fn resolve_audio_input_bytes() {
        let input = DeepgramAdapter::resolve_audio_input(&FileInput::bytes(vec![1, 2, 3]))
            .await
            .unwrap();
        match input {
            AudioInput::Bytes(b, ct) => {
                assert_eq!(b, vec![1, 2, 3]);
                assert_eq!(ct, "audio/wav");
            }
            _ => panic!("应为 Bytes 变体"),
        }
    }

    #[tokio::test]
    async fn resolve_audio_input_base64() {
        let encoded = crate::util::encode_base64(b"audio-bytes");
        let input = DeepgramAdapter::resolve_audio_input(&FileInput::base64(encoded))
            .await
            .unwrap();
        match input {
            AudioInput::Bytes(b, _) => assert_eq!(b, b"audio-bytes".to_vec()),
            _ => panic!("应为 Bytes 变体"),
        }
    }

    #[tokio::test]
    async fn resolve_audio_input_base64_invalid_returns_validation() {
        let err = DeepgramAdapter::resolve_audio_input(&FileInput::base64("!!!not-base64!!!"))
            .await
            .unwrap_err();
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[tokio::test]
    async fn resolve_audio_input_url() {
        let input =
            DeepgramAdapter::resolve_audio_input(&FileInput::url("https://example.com/audio.mp3"))
                .await
                .unwrap();
        match input {
            AudioInput::RemoteUrl(u) => assert_eq!(u, "https://example.com/audio.mp3"),
            _ => panic!("应为 RemoteUrl 变体"),
        }
    }

    #[tokio::test]
    async fn resolve_audio_input_path_reads_file() {
        // 写一个临时文件再读取
        let path = std::env::temp_dir().join("aibridge_deepgram_test_read.wav");
        std::fs::write(&path, b"wav-bytes").unwrap();
        let input = DeepgramAdapter::resolve_audio_input(&FileInput::path(path.to_str().unwrap()))
            .await
            .unwrap();
        match input {
            AudioInput::Bytes(b, ct) => {
                assert_eq!(b, b"wav-bytes".to_vec());
                assert_eq!(ct, "audio/wav");
            }
            _ => panic!("应为 Bytes 变体"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn resolve_audio_input_path_mp3_mime() {
        let path = std::env::temp_dir().join("aibridge_deepgram_test_read.mp3");
        std::fs::write(&path, b"mp3-bytes").unwrap();
        let input = DeepgramAdapter::resolve_audio_input(&FileInput::path(path.to_str().unwrap()))
            .await
            .unwrap();
        match input {
            AudioInput::Bytes(_, ct) => assert_eq!(ct, "audio/mpeg"),
            _ => panic!("应为 Bytes 变体"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn resolve_audio_input_path_nonexistent_returns_validation() {
        let err = DeepgramAdapter::resolve_audio_input(&FileInput::path(
            "/tmp/aibridge_nonexistent_file_xyz.wav",
        ))
        .await
        .unwrap_err();
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    // ============ parse_deepgram_response ============

    #[test]
    fn parse_response_basic_with_words() {
        let result = DeepgramAdapter::parse_deepgram_response(&sample_response(), "nova-2");
        assert_eq!(result.text, "hello world");
        assert_eq!(result.language.as_deref(), Some("en"));
        assert!((result.duration.unwrap() - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.task, "transcribe");
        assert_eq!(result.model.as_deref(), Some("nova-2"));
        let words = result.words.expect("应有 words");
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert!((words[0].start - 0.0).abs() < f64::EPSILON);
        assert!((words[0].end - 0.5).abs() < f64::EPSILON);
        assert!((words[0].confidence.unwrap() - 0.98).abs() < f64::EPSILON);
        assert!(result.segments.is_none());
    }

    #[test]
    fn parse_response_with_segments_and_speaker() {
        let json = serde_json::json!({
            "results": {
                "channels": [{
                    "alternatives": [{
                        "transcript": "hello world",
                        "paragraphs": {
                            "paragraphs": [{
                                "speaker": 0,
                                "sentences": [
                                    {"text": "hello", "start": 0.0, "end": 0.5},
                                    {"text": "world", "start": 0.5, "end": 1.0}
                                ]
                            }]
                        }
                    }]
                }]
            },
            "metadata": {"duration": 1.0, "model_info": {"language": "en"}}
        });
        let result = DeepgramAdapter::parse_deepgram_response(&json, "nova-3");
        assert_eq!(result.text, "hello world");
        let segments = result.segments.expect("应有 segments");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].id, 0);
        assert_eq!(segments[0].text, "hello");
        assert!((segments[0].start - 0.0).abs() < f64::EPSILON);
        assert!((segments[0].end - 0.5).abs() < f64::EPSILON);
        assert_eq!(segments[1].id, 1);
        assert_eq!(segments[1].text, "world");
        assert_eq!(segments[0].speaker.as_deref(), Some("0"));
        assert_eq!(segments[1].speaker.as_deref(), Some("0"));
    }

    #[test]
    fn parse_response_empty_channels() {
        let json = serde_json::json!({
            "results": {"channels": []},
            "metadata": {"duration": 0.0}
        });
        let result = DeepgramAdapter::parse_deepgram_response(&json, "nova-2");
        assert!(result.text.is_empty());
        assert!(result.words.is_none());
        assert!(result.segments.is_none());
        assert!(result.language.is_none());
        assert!((result.duration.unwrap() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_response_multiple_channels_concatenates() {
        let json = serde_json::json!({
            "results": {
                "channels": [
                    {"alternatives": [{"transcript": "hello"}]},
                    {"alternatives": [{"transcript": "world"}]}
                ]
            },
            "metadata": {}
        });
        let result = DeepgramAdapter::parse_deepgram_response(&json, "nova-2");
        assert_eq!(result.text, "hello world");
    }

    #[test]
    fn parse_response_missing_metadata() {
        let json = serde_json::json!({
            "results": {
                "channels": [{"alternatives": [{"transcript": "hi"}]}]
            }
        });
        let result = DeepgramAdapter::parse_deepgram_response(&json, "nova-2");
        assert_eq!(result.text, "hi");
        assert!(result.duration.is_none());
        assert!(result.language.is_none());
    }

    #[test]
    fn parse_response_empty_transcript_skipped() {
        let json = serde_json::json!({
            "results": {
                "channels": [{"alternatives": [{"transcript": ""}]}]
            },
            "metadata": {}
        });
        let result = DeepgramAdapter::parse_deepgram_response(&json, "nova-2");
        assert!(result.text.is_empty());
    }

    // ============ map_deepgram_error ============

    #[test]
    fn map_error_401_is_authentication() {
        let err = DeepgramAdapter::map_deepgram_error(401, "invalid token");
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[test]
    fn map_error_403_is_authentication() {
        let err = DeepgramAdapter::map_deepgram_error(403, "forbidden");
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[test]
    fn map_error_429_is_rate_limit() {
        let err = DeepgramAdapter::map_deepgram_error(429, "slow down");
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[test]
    fn map_error_400_is_validation() {
        let err = DeepgramAdapter::map_deepgram_error(400, "bad request");
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[test]
    fn map_error_422_is_validation() {
        let err = DeepgramAdapter::map_deepgram_error(422, "unprocessable");
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[test]
    fn map_error_500_is_api() {
        let err = DeepgramAdapter::map_deepgram_error(500, "server error");
        assert!(matches!(err, AibridgeError::Api { status: 500, .. }));
    }

    #[test]
    fn map_error_503_is_api() {
        let err = DeepgramAdapter::map_deepgram_error(503, "unavailable");
        assert!(matches!(err, AibridgeError::Api { status: 503, .. }));
    }

    #[test]
    fn map_error_other_4xx_is_api() {
        let err = DeepgramAdapter::map_deepgram_error(418, "teapot");
        assert!(matches!(err, AibridgeError::Api { status: 418, .. }));
    }

    // ============ transcribe（mockito） ============

    #[tokio::test]
    async fn transcribe_bytes_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", LISTEN_PATH)
            .match_header("authorization", "Token test-key")
            .match_header("content-type", "audio/wav")
            .match_query(mockito::Matcher::AllOf(vec![
                mockito::Matcher::UrlEncoded("model".into(), "nova-2".into()),
                mockito::Matcher::UrlEncoded("smart_format".into(), "true".into()),
            ]))
            .match_body(mockito::Matcher::from(b"audio-bytes".to_vec()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(sample_response().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req =
            TranscribeRequest::builder("nova-2", FileInput::bytes(b"audio-bytes".to_vec())).build();
        let result = adapter.transcribe(req).await.unwrap();
        assert_eq!(result.text, "hello world");
        assert_eq!(result.language.as_deref(), Some("en"));
        assert_eq!(result.model.as_deref(), Some("nova-2"));
        assert_eq!(result.task, "transcribe");
        assert!(result.words.is_some());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn transcribe_url_uses_url_query_param_no_body() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", LISTEN_PATH)
            .match_header("authorization", "Token test-key")
            .match_query(mockito::Matcher::AllOf(vec![
                mockito::Matcher::UrlEncoded("url".into(), "https://example.com/audio.mp3".into()),
                mockito::Matcher::UrlEncoded("model".into(), "nova-2".into()),
            ]))
            .match_body(mockito::Matcher::Missing)
            .with_status(200)
            .with_body(sample_response().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req =
            TranscribeRequest::builder("nova-2", FileInput::url("https://example.com/audio.mp3"))
                .build();
        let result = adapter.transcribe(req).await.unwrap();
        assert_eq!(result.text, "hello world");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn transcribe_with_language_query_param() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", LISTEN_PATH)
            .match_query(mockito::Matcher::UrlEncoded("language".into(), "zh".into()))
            .with_status(200)
            .with_body(sample_response().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1, 2, 3]))
            .language("zh")
            .build();
        let _ = adapter.transcribe(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn transcribe_smart_format_disabled_not_in_query() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", LISTEN_PATH)
            .match_query(mockito::Matcher::UrlEncoded(
                "model".into(),
                "nova-2".into(),
            ))
            // smart_format 不应出现（mock 默认允许其他 query，这里仅断言 model 存在）
            .with_status(200)
            .with_body(sample_response().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1, 2, 3]))
            .extra("smart_format", false)
            .build();
        let params = DeepgramAdapter::build_query_params(&req, "nova-2");
        assert!(!params.iter().any(|(k, _)| k == "smart_format"));
        let _ = adapter.transcribe(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn transcribe_uses_default_model_when_empty() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", LISTEN_PATH)
            .match_query(mockito::Matcher::UrlEncoded(
                "model".into(),
                DEFAULT_MODEL.into(),
            ))
            .with_status(200)
            .with_body(sample_response().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = TranscribeRequest::builder("", FileInput::bytes(vec![1, 2, 3])).build();
        let result = adapter.transcribe(req).await.unwrap();
        assert_eq!(result.model.as_deref(), Some(DEFAULT_MODEL));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn transcribe_path_input_uploads_file_bytes() {
        let mut server = mockito::Server::new_async().await;
        // 写临时 wav 文件
        let path = std::env::temp_dir().join("aibridge_deepgram_transcribe_test.wav");
        std::fs::write(&path, b"wav-file-content").unwrap();
        let mock = server
            .mock("POST", LISTEN_PATH)
            .match_query(mockito::Matcher::Any)
            .match_header("content-type", "audio/wav")
            .match_body(mockito::Matcher::from(b"wav-file-content".to_vec()))
            .with_status(200)
            .with_body(sample_response().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req =
            TranscribeRequest::builder("nova-2", FileInput::path(path.to_str().unwrap())).build();
        let _ = adapter.transcribe(req).await.unwrap();
        mock.assert_async().await;
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn transcribe_error_401_returns_authentication() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", LISTEN_PATH)
            .match_query(mockito::Matcher::Any)
            .with_status(401)
            .with_body("{\"err_msg\":\"invalid api key\"}")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("bad-key"));
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1, 2, 3])).build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn transcribe_error_429_returns_rate_limit() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", LISTEN_PATH)
            .match_query(mockito::Matcher::Any)
            .with_status(429)
            .with_body("{\"err_msg\":\"quota exceeded\"}")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1, 2, 3])).build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn transcribe_error_400_returns_validation() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", LISTEN_PATH)
            .match_query(mockito::Matcher::Any)
            .with_status(400)
            .with_body("{\"err_msg\":\"invalid model\"}")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = TranscribeRequest::builder("bad-model", FileInput::bytes(vec![1, 2, 3])).build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[tokio::test]
    async fn transcribe_error_500_returns_api() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", LISTEN_PATH)
            .match_query(mockito::Matcher::Any)
            .with_status(500)
            .with_body("internal error")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1, 2, 3])).build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Api { status: 500, .. }));
    }

    #[tokio::test]
    async fn transcribe_error_403_returns_authentication() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", LISTEN_PATH)
            .match_query(mockito::Matcher::Any)
            .with_status(403)
            .with_body("forbidden")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1, 2, 3])).build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    // ============ translate（Deepgram 不支持） ============

    #[tokio::test]
    async fn transcribe_translate_true_returns_validation() {
        let adapter = make_adapter(None, Some("test-key"));
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1, 2, 3]))
            .translate(true)
            .build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[tokio::test]
    async fn transcribe_translate_false_proceeds() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", LISTEN_PATH)
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(sample_response().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1, 2, 3]))
            .translate(false)
            .build();
        let result = adapter.transcribe(req).await.unwrap();
        assert_eq!(result.text, "hello world");
    }

    // ============ API key 缺失 ============

    #[tokio::test]
    async fn transcribe_without_api_key_returns_validation() {
        let adapter = make_adapter(Some("https://example.com".into()), None);
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1, 2, 3])).build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[tokio::test]
    async fn transcribe_with_empty_api_key_returns_validation() {
        let adapter = make_adapter(Some("https://example.com".into()), Some(""));
        let req = TranscribeRequest::builder("nova-2", FileInput::bytes(vec![1, 2, 3])).build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }
}
