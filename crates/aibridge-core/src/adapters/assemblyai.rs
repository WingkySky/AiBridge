//! AssemblyAI 适配器（企业级 ASR，speaker diarization，独立协议）
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/audio_adapters.py` 的 `AssemblyAIAdapter`。
//!
//! AssemblyAI 是企业级语音识别服务，支持说话人分离（speaker diarization）、
//! 情感分析、PII 脱敏、章节检测、实体识别等丰富语音理解能力。
//!
//! ## 协议（异步三段式）
//!
//! AssemblyAI 采用异步任务模式，转写流程分三步：
//! 1. 上传音频：`POST /upload`（原始字节，`Content-Type: application/octet-stream`），
//!    返回 `{"upload_url": "..."}`
//! 2. 创建转写任务：`POST /transcript`（JSON body，含 audio_url + 参数），
//!    返回 `{"id": "..."}`
//! 3. 轮询结果：`GET /transcript/{id}`，返回 `{"status": "completed|error|processing|queued", ...}`
//!
//! - 认证：`Authorization: <API_KEY>` header（注意：不是 Bearer，直接传 key）
//! - 文档：https://www.assemblyai.com/docs
//! - 若调用方直接提供 `extra.audio_url`（或 `FileInput::Url`），跳过上传步骤
//!
//! ## 特性（v1.3.3 保留）
//!
//! - `requires_api_key = true`（需 AssemblyAI API Key）
//! - `capabilities`：`AudioTranscribe` + `AudioTranslate`（translate 复用转写管线，
//!   task 标记为 "translate"；AssemblyAI 不直接支持音频翻译，结果语言由 language_code 决定）
//! - `speech_model`：best（默认，最高准确率）/ nano（轻量低成本）
//! - 高级参数：speaker_labels / sentiment_analysis / auto_chapters / entity_detection /
//!   redact_pii / word_boost / filter_profanity 等，通过 `extra` 透传
//! - 时间戳：AssemblyAI 的 start/end 为毫秒，归一化为秒；audio_duration 已是秒

use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::time::sleep;

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
const PROVIDER_TYPE: &str = "assemblyai";
/// Provider 显示名称
const PROVIDER_NAME: &str = "AssemblyAI";
/// 默认 API 基地址（AssemblyAI v2 API）
const DEFAULT_API_BASE: &str = "https://api.assemblyai.com/v2";
/// 上传音频端点路径
const UPLOAD_PATH: &str = "/upload";
/// 创建/查询转写任务端点路径
const TRANSCRIPT_PATH: &str = "/transcript";
/// 默认轮询间隔（秒）
const DEFAULT_POLL_INTERVAL: f64 = 1.0;
/// 默认最大轮询次数（对应 Python v1 `MAX_POLLS = 300`）
const DEFAULT_MAX_POLLS: u64 = 300;

// ==================== AssemblyAiAdapter ====================

/// AssemblyAI 适配器
///
/// 持有 HTTP 客户端、API key 与基地址。transcribe 流程为上传 → 创建 → 轮询，
/// 均为 per-request HTTP 调用（无长连接）。
pub struct AssemblyAiAdapter {
    /// Provider 配置（保留供未来扩展，当前字段已在构造时提取）
    #[allow(dead_code)]
    config: ProviderConfig,
    /// HTTP 客户端（封装 reqwest，含连接池与超时）
    http: HttpClient,
    /// API 基地址（默认 `https://api.assemblyai.com/v2`，可由 config.base_url 覆盖）
    api_base: String,
    /// API Key（Authorization header 值）
    api_key: String,
}

impl AssemblyAiAdapter {
    /// 创建 AssemblyAI 适配器
    ///
    /// - `config.base_url` 为空时用 `DEFAULT_API_BASE`
    /// - `config.api_key` 为 None 时用空串（调用时 API 会返 401）
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let api_base = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_API_BASE.to_string());
        let api_key = config.api_key.clone().unwrap_or_default();
        let opts = ClientOptions::builder().timeout(config.timeout).build();
        let http = HttpClient::new(&opts)?;
        Ok(Self {
            config,
            http,
            api_base,
            api_key,
        })
    }

    // ==================== 纯函数（协议构造/解析，可单测） ====================

    /// 解析 speech_model：best / nano，其余默认 best
    ///
    /// 对应 Python v1 `speech_model = model if model in ("best", "nano") else "best"`。
    fn resolve_speech_model(model: &str) -> String {
        match model {
            "best" | "nano" => model.to_string(),
            _ => "best".to_string(),
        }
    }

    /// 构造 `/transcript` 请求体
    ///
    /// 对应 Python v1 `transcript_request` 构造。必填字段 audio_url + speech_model，
    /// 默认开启 punctuate / format_text。可选字段（language_code / speaker_labels /
    /// filter_profanity / sentiment_analysis / auto_chapters / entity_detection /
    /// redact_pii / word_boost）仅在有值时加入。
    fn build_transcript_payload(
        req: &TranscribeRequest,
        audio_url: &str,
        speech_model: &str,
    ) -> Value {
        let mut payload = json!({
            "audio_url": audio_url,
            "speech_model": speech_model,
            "punctuate": req.extra.get("punctuate").and_then(|v| v.as_bool()).unwrap_or(true),
            "format_text": req.extra.get("format_text").and_then(|v| v.as_bool()).unwrap_or(true),
        });

        // language_code（req.language 优先，回退 extra.language_code）
        let lang = req
            .language
            .as_deref()
            .filter(|l| !l.is_empty())
            .or_else(|| req.extra.get("language_code").and_then(|v| v.as_str()));
        if let Some(lang) = lang {
            payload["language_code"] = json!(lang);
        }

        // 布尔开关参数：仅当显式为 true 时加入
        for &(extra_key, payload_key) in &[
            ("speaker_labels", "speaker_labels"),
            ("filter_profanity", "filter_profanity"),
            ("sentiment_analysis", "sentiment_analysis"),
            ("auto_chapters", "auto_chapters"),
            ("entity_detection", "entity_detection"),
        ] {
            if req
                .extra
                .get(extra_key)
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                payload[payload_key] = json!(true);
            }
        }

        // redact_pii：开启时同时传 redact_pii_policies（默认空数组）
        if req
            .extra
            .get("redact_pii")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            payload["redact_pii"] = json!(true);
            payload["redact_pii_policies"] = req
                .extra
                .get("redact_pii_policies")
                .cloned()
                .unwrap_or_else(|| json!([]));
        }

        // word_boost：仅当为 JSON 数组时透传
        if let Some(wb) = req.extra.get("word_boost").and_then(|v| v.as_array()) {
            payload["word_boost"] = json!(wb);
        }

        payload
    }

    /// 解析 AssemblyAI 转写结果为统一 `TranscriptionResult`
    ///
    /// - text / language_code / audio_duration 直接映射（audio_duration 已是秒）
    /// - utterances（speaker_labels 开启时返回）→ segments + 词级 words，start/end 毫秒转秒
    /// - 无 utterances 时回退到顶层 words 数组
    /// - task 由调用方传入（transcribe / translate）
    fn parse_response(v: &Value, speech_model: &str, task: &str) -> TranscriptionResult {
        let text = v
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let language = v
            .get("language_code")
            .and_then(|l| l.as_str())
            .map(|s| s.to_string());
        let duration = v.get("audio_duration").and_then(|d| d.as_f64());

        let mut segments: Vec<TranscriptionSegment> = Vec::new();
        let mut words: Vec<TranscriptionWord> = Vec::new();

        if let Some(utterances) = v.get("utterances").and_then(|u| u.as_array()) {
            for (idx, utt) in utterances.iter().enumerate() {
                segments.push(TranscriptionSegment {
                    id: idx as u32,
                    start: ms_to_secs(utt.get("start")),
                    end: ms_to_secs(utt.get("end")),
                    text: utt
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    confidence: utt.get("confidence").and_then(|c| c.as_f64()),
                    speaker: utt
                        .get("speaker")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string()),
                });
                if let Some(utt_words) = utt.get("words").and_then(|w| w.as_array()) {
                    for w in utt_words {
                        words.push(TranscriptionWord {
                            word: w
                                .get("text")
                                .and_then(|t| t.as_str())
                                .unwrap_or("")
                                .to_string(),
                            start: ms_to_secs(w.get("start")),
                            end: ms_to_secs(w.get("end")),
                            confidence: w.get("confidence").and_then(|c| c.as_f64()),
                        });
                    }
                }
            }
        } else if let Some(api_words) = v.get("words").and_then(|w| w.as_array()) {
            for w in api_words {
                words.push(TranscriptionWord {
                    word: w
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    start: ms_to_secs(w.get("start")),
                    end: ms_to_secs(w.get("end")),
                    confidence: w.get("confidence").and_then(|c| c.as_f64()),
                });
            }
        }

        TranscriptionResult {
            text,
            language,
            duration,
            segments: if segments.is_empty() {
                None
            } else {
                Some(segments)
            },
            words: if words.is_empty() { None } else { Some(words) },
            task: task.to_string(),
            usage: None,
            model: Some(speech_model.to_string()),
        }
    }

    // ==================== 网络方法 ====================

    /// 上传音频字节到 AssemblyAI，返回 upload_url
    ///
    /// 对应 Python v1 `_upload_audio`：POST /upload，原始字节 body。
    async fn upload_audio(&self, data: Vec<u8>) -> Result<String> {
        let url = format!(
            "{base}{path}",
            base = self.api_base.trim_end_matches('/'),
            path = UPLOAD_PATH
        );
        let resp = self
            .http
            .inner()
            .post(&url)
            .header("Authorization", &self.api_key)
            .header("Content-Type", "application/octet-stream")
            .body(data)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(map_assemblyai_error(status_code, &body));
        }
        let v: Value = resp.json().await.map_err(AibridgeError::from)?;
        v.get("upload_url")
            .and_then(|u| u.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| AibridgeError::api(0, "AssemblyAI upload 响应缺少 upload_url 字段"))
    }

    /// 创建转写任务，返回 transcript id
    ///
    /// 对应 Python v1 `POST /transcript`。
    async fn create_transcript(&self, payload: &Value) -> Result<String> {
        let url = format!(
            "{base}{path}",
            base = self.api_base.trim_end_matches('/'),
            path = TRANSCRIPT_PATH
        );
        let resp = self
            .http
            .inner()
            .post(&url)
            .header("Authorization", &self.api_key)
            .json(payload)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(map_assemblyai_error(status_code, &body));
        }
        let v: Value = resp.json().await.map_err(AibridgeError::from)?;
        v.get("id")
            .and_then(|i| i.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| AibridgeError::api(0, "AssemblyAI transcript 创建响应缺少 id 字段"))
    }

    /// 轮询转写任务结果，直到 completed / error 或超过最大轮询次数
    ///
    /// 对应 Python v1 轮询循环。status=completed 返回结果 JSON；status=error 映射为
    /// Api 错误；processing/queued 继续 sleep 后重试；超过 max_polls 返 Timeout。
    async fn poll_transcript(
        &self,
        transcript_id: &str,
        poll_interval: f64,
        max_polls: u64,
    ) -> Result<Value> {
        let url = format!(
            "{base}{path}/{id}",
            base = self.api_base.trim_end_matches('/'),
            path = TRANSCRIPT_PATH,
            id = transcript_id,
        );
        // 负数或 NaN 防御：回退到默认间隔
        let delay_secs = if poll_interval.is_finite() && poll_interval > 0.0 {
            poll_interval
        } else {
            DEFAULT_POLL_INTERVAL
        };
        let delay = Duration::from_secs_f64(delay_secs);

        for _ in 0..max_polls {
            let resp = self
                .http
                .inner()
                .get(&url)
                .header("Authorization", &self.api_key)
                .send()
                .await
                .map_err(map_reqwest_error)?;

            let status = resp.status();
            if !status.is_success() {
                let status_code = status.as_u16();
                let body = resp.text().await.unwrap_or_default();
                return Err(map_assemblyai_error(status_code, &body));
            }
            let v: Value = resp.json().await.map_err(AibridgeError::from)?;
            match v.get("status").and_then(|s| s.as_str()) {
                Some("completed") => return Ok(v),
                Some("error") => {
                    let err_msg = v
                        .get("error")
                        .and_then(|e| e.as_str())
                        .unwrap_or("转写失败");
                    return Err(AibridgeError::api(
                        500,
                        format!("AssemblyAI 转写错误: {err_msg}"),
                    ));
                }
                _ => { /* processing / queued / 其他：继续轮询 */ }
            }
            sleep(delay).await;
        }
        Err(AibridgeError::Timeout)
    }

    /// 解析音频输入来源，返回最终用于创建任务的 audio_url
    ///
    /// 优先级：`extra.audio_url` > `FileInput::Url` > 上传字节（Bytes/Base64/Path）。
    async fn resolve_audio_url(&self, req: &TranscribeRequest) -> Result<String> {
        // 1. extra.audio_url 直接指定（跳过上传）
        if let Some(url) = req.extra.get("audio_url").and_then(|v| v.as_str()) {
            if !url.is_empty() {
                return Ok(url.to_string());
            }
        }
        // 2. FileInput::Url 直接使用
        if let FileInput::Url(u) = &req.file {
            return Ok(u.clone());
        }
        // 3. 字节类输入：读取后上传
        let data = self.read_file_input_bytes(&req.file).await?;
        self.upload_audio(data).await
    }

    /// 从 `FileInput` 读取音频字节（Bytes/Base64/Path 三种）
    ///
    /// `FileInput::Url` 不应走到这里（由 `resolve_audio_url` 提前处理）。
    async fn read_file_input_bytes(&self, file: &FileInput) -> Result<Vec<u8>> {
        match file {
            FileInput::Bytes(data) => Ok(data.clone()),
            FileInput::Base64(s) => {
                let decoded = crate::util::decode_base64(s).map_err(|e| {
                    AibridgeError::validation(format!("AssemblyAI 音频 Base64 解码失败: {e}"))
                })?;
                Ok(decoded)
            }
            FileInput::Path(p) => {
                let data = tokio::fs::read(p).await.map_err(|e| {
                    AibridgeError::validation(format!("AssemblyAI 读取音频文件失败 {p}: {e}"))
                })?;
                Ok(data)
            }
            FileInput::Url(_) => Err(AibridgeError::validation(
                "AssemblyAI 不应对 Url 输入执行上传",
            )),
        }
    }
}

#[async_trait]
impl Adapter for AssemblyAiAdapter {
    fn provider_type(&self) -> &str {
        PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    fn capabilities(&self) -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::AudioTranscribe);
        // translate 复用转写管线（task 标记为 "translate"），声明 AudioTranslate 能力
        caps.insert(Capabilities::AudioTranslate);
        caps
    }

    /// AssemblyAI 需要 API Key（Authorization header）
    fn requires_api_key(&self) -> bool {
        true
    }

    async fn start(&mut self) -> Result<()> {
        // HTTP 客户端在 new() 已构造，无惰性资源需初始化
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        // 无长连接资源需释放
        Ok(())
    }

    /// 语音转文字（AssemblyAI 异步三段式协议）
    ///
    /// 流程：解析 audio_url → 创建任务 → 轮询结果 → 解析为统一格式。
    /// `req.translate = true` 时 task 标记为 "translate"（AssemblyAI 不直接支持音频翻译，
    /// 但统一接口保留该语义，结果文本语言由 language_code 决定）。
    async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscriptionResult> {
        let speech_model = Self::resolve_speech_model(&req.model);
        let audio_url = self.resolve_audio_url(&req).await?;
        let payload = Self::build_transcript_payload(&req, &audio_url, &speech_model);

        let transcript_id = self.create_transcript(&payload).await?;

        let poll_interval = req
            .extra
            .get("polling_interval")
            .and_then(|v| v.as_f64())
            .unwrap_or(DEFAULT_POLL_INTERVAL);
        let max_polls = req
            .extra
            .get("max_polls")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_MAX_POLLS);

        let result = self
            .poll_transcript(&transcript_id, poll_interval, max_polls)
            .await?;

        let task = if req.translate {
            "translate"
        } else {
            "transcribe"
        };
        Ok(Self::parse_response(&result, &speech_model, task))
    }

    /// 列出 AssemblyAI 模型（无标准 /models 端点，硬编码 best / nano）
    ///
    /// 对应 Python v1 `list_models`。
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = vec![
            ModelInfo {
                id: "best".into(),
                name: "Best".into(),
                model_type: ModelType::Audio,
                provider: PROVIDER_TYPE.into(),
                capabilities: vec!["audio_transcribe".into()],
                max_tokens: None,
                supports_streaming: false,
                description: Some("AssemblyAI 最高准确率模型（默认），支持所有高级功能".into()),
                created: None,
            },
            ModelInfo {
                id: "nano".into(),
                name: "Nano".into(),
                model_type: ModelType::Audio,
                provider: PROVIDER_TYPE.into(),
                capabilities: vec!["audio_transcribe".into()],
                max_tokens: None,
                supports_streaming: false,
                description: Some("AssemblyAI 轻量模型，更低成本，适合简单场景".into()),
                created: None,
            },
        ];
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }
}

// ==================== 错误映射 ====================

/// 将 AssemblyAI HTTP 错误响应映射为 AibridgeError
///
/// 对应 Python v1 `AssemblyAIAdapter._handle_error`。映射规则（与任务规格一致）：
/// - 401/403 → Authentication（API Key 无效）
/// - 429 → RateLimit（限流或配额耗尽）
/// - 400 → Validation（参数校验错误，携带 details）
/// - 5xx → Api（服务端错误）
/// - 其余 4xx → Api
fn map_assemblyai_error(status: u16, body: &str) -> AibridgeError {
    let message = parse_assemblyai_error_message(body, status);
    match status {
        401 | 403 => AibridgeError::authentication(format!(
            "AssemblyAI 认证失败（API Key 无效或缺失）: {message}"
        )),
        429 => AibridgeError::rate_limit(format!("AssemblyAI 限流或配额耗尽: {message}")),
        400 => AibridgeError::validation_with_details(
            format!("AssemblyAI 参数校验错误: {message}"),
            serde_json::json!({"status": status, "response": body}),
        ),
        s if s >= 500 => AibridgeError::api(s, format!("AssemblyAI 服务端错误 ({s}): {message}")),
        s => AibridgeError::api(s, format!("AssemblyAI HTTP {s}: {message}")),
    }
}

/// 从 AssemblyAI 错误响应体解析错误消息
///
/// AssemblyAI 错误响应通常为 `{"error": "..."}`。解析失败时截取前 200 字符或回退 `HTTP {status}`。
fn parse_assemblyai_error_message(body: &str, status: u16) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        if let Some(m) = v.get("error").and_then(|e| e.as_str()) {
            return m.to_string();
        }
        if let Some(m) = v.get("message").and_then(|m| m.as_str()) {
            return m.to_string();
        }
    }
    if body.trim().is_empty() {
        format!("HTTP {status}")
    } else {
        body.chars().take(200).collect()
    }
}

// ==================== 辅助函数 ====================

/// 将 reqwest::Error 映射为 AibridgeError（超时 → Timeout，其余 → Network）
fn map_reqwest_error(e: reqwest::Error) -> AibridgeError {
    if e.is_timeout() {
        AibridgeError::Timeout
    } else {
        AibridgeError::Network(e)
    }
}

/// 把 AssemblyAI 的毫秒时间戳转为秒
///
/// `start` / `end` 字段为毫秒（整数），`audio_duration` 已是秒（不调用本函数）。
fn ms_to_secs(v: Option<&Value>) -> f64 {
    v.and_then(|x| x.as_f64()).unwrap_or(0.0) / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::TranscribeRequest;
    use crate::model::chat::ChatRequest;
    use crate::model::image::{FileInput, ImageRequest};
    use crate::model::options::{EmbedInput, EmbedRequest};
    use crate::model::video::VideoRequest;
    use std::collections::HashMap;

    /// 构造测试用适配器（base_url 指向 mockito server，None 用默认基地址）
    fn make_adapter(base_url: Option<String>) -> AssemblyAiAdapter {
        let mut opts = ClientOptions::builder().api_key("test-key");
        if let Some(u) = base_url {
            opts = opts.base_url(u);
        }
        let config = ProviderConfig::from_options(PROVIDER_TYPE, opts.build());
        AssemblyAiAdapter::new(config).expect("构造 AssemblyAiAdapter 失败")
    }

    // ============ 基本属性 ============

    #[test]
    fn requires_api_key_is_true() {
        let adapter = make_adapter(None);
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn capabilities_contains_transcribe_and_translate() {
        let adapter = make_adapter(None);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::AudioTranscribe));
        // translate 复用转写管线（task 标记为 translate），声明 AudioTranslate 能力
        assert!(caps.contains(&Capabilities::AudioTranslate));
        assert!(!caps.contains(&Capabilities::AudioSpeech));
        assert!(!caps.contains(&Capabilities::Chat));
        assert!(!caps.contains(&Capabilities::ImageGenerate));
        assert!(!caps.contains(&Capabilities::ListVoices));
    }

    #[test]
    fn provider_type_and_name() {
        let adapter = make_adapter(None);
        assert_eq!(adapter.provider_type(), "assemblyai");
        assert_eq!(adapter.provider_name(), "AssemblyAI");
    }

    #[tokio::test]
    async fn list_models_returns_best_and_nano() {
        let adapter = make_adapter(None);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"best"));
        assert!(ids.contains(&"nano"));
        for m in &models {
            assert_eq!(m.model_type, ModelType::Audio);
            assert_eq!(m.provider, "assemblyai");
        }
    }

    #[tokio::test]
    async fn list_models_filter_by_type() {
        let adapter = make_adapter(None);
        let audio = adapter.list_models(Some(ModelType::Audio)).await.unwrap();
        assert_eq!(audio.len(), 2);
        let chat = adapter.list_models(Some(ModelType::Chat)).await.unwrap();
        assert!(chat.is_empty());
    }

    #[tokio::test]
    async fn start_and_close_are_noops() {
        let mut adapter = make_adapter(None);
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    // ============ 不支持能力（默认实现） ============

    #[tokio::test]
    async fn chat_returns_unsupported() {
        let adapter = make_adapter(None);
        let req = ChatRequest::builder("m", vec![]).build();
        assert!(matches!(
            adapter.chat(req).await.unwrap_err(),
            AibridgeError::UnsupportedCapability { .. }
        ));
    }

    #[tokio::test]
    async fn image_generate_returns_unsupported() {
        let adapter = make_adapter(None);
        let req = ImageRequest::builder("m", "p").build();
        assert!(matches!(
            adapter.image_generate(req).await.unwrap_err(),
            AibridgeError::UnsupportedCapability { .. }
        ));
    }

    #[tokio::test]
    async fn video_create_returns_unsupported() {
        let adapter = make_adapter(None);
        let req = VideoRequest::builder("m", "p").build();
        assert!(matches!(
            adapter.video_create(req).await.unwrap_err(),
            AibridgeError::UnsupportedCapability { .. }
        ));
    }

    #[tokio::test]
    async fn embed_returns_unsupported() {
        let adapter = make_adapter(None);
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
        let adapter = make_adapter(None);
        let req = crate::model::audio::SpeechRequest::builder("m", "hi", "v").build();
        assert!(matches!(
            adapter.speech(req).await.unwrap_err(),
            AibridgeError::UnsupportedCapability { .. }
        ));
    }

    // ============ resolve_speech_model ============

    #[test]
    fn resolve_speech_model_best_passthrough() {
        assert_eq!(AssemblyAiAdapter::resolve_speech_model("best"), "best");
    }

    #[test]
    fn resolve_speech_model_nano_passthrough() {
        assert_eq!(AssemblyAiAdapter::resolve_speech_model("nano"), "nano");
    }

    #[test]
    fn resolve_speech_model_unknown_defaults_best() {
        assert_eq!(AssemblyAiAdapter::resolve_speech_model("whisper"), "best");
        assert_eq!(AssemblyAiAdapter::resolve_speech_model(""), "best");
    }

    // ============ build_transcript_payload ============

    #[test]
    fn build_payload_minimal_defaults() {
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1])).build();
        let payload =
            AssemblyAiAdapter::build_transcript_payload(&req, "https://up.example.com/u", "best");
        assert_eq!(payload["audio_url"], "https://up.example.com/u");
        assert_eq!(payload["speech_model"], "best");
        assert_eq!(payload["punctuate"], true);
        assert_eq!(payload["format_text"], true);
        // 无可选字段时不应出现
        assert!(payload.get("language_code").is_none());
        assert!(payload.get("speaker_labels").is_none());
        assert!(payload.get("redact_pii").is_none());
        assert!(payload.get("word_boost").is_none());
    }

    #[test]
    fn build_payload_punctuate_format_text_override() {
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("punctuate", false)
            .extra("format_text", false)
            .build();
        let payload = AssemblyAiAdapter::build_transcript_payload(&req, "https://u", "best");
        assert_eq!(payload["punctuate"], false);
        assert_eq!(payload["format_text"], false);
    }

    #[test]
    fn build_payload_with_language() {
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .language("zh")
            .build();
        let payload = AssemblyAiAdapter::build_transcript_payload(&req, "https://u", "best");
        assert_eq!(payload["language_code"], "zh");
    }

    #[test]
    fn build_payload_language_from_extra_language_code() {
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("language_code", "ja")
            .build();
        let payload = AssemblyAiAdapter::build_transcript_payload(&req, "https://u", "best");
        assert_eq!(payload["language_code"], "ja");
    }

    #[test]
    fn build_payload_req_language_takes_priority_over_extra() {
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .language("en")
            .extra("language_code", "ja")
            .build();
        let payload = AssemblyAiAdapter::build_transcript_payload(&req, "https://u", "best");
        assert_eq!(payload["language_code"], "en");
    }

    #[test]
    fn build_payload_with_speaker_labels() {
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("speaker_labels", true)
            .build();
        let payload = AssemblyAiAdapter::build_transcript_payload(&req, "https://u", "best");
        assert_eq!(payload["speaker_labels"], true);
    }

    #[test]
    fn build_payload_with_all_bool_flags() {
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("speaker_labels", true)
            .extra("filter_profanity", true)
            .extra("sentiment_analysis", true)
            .extra("auto_chapters", true)
            .extra("entity_detection", true)
            .build();
        let payload = AssemblyAiAdapter::build_transcript_payload(&req, "https://u", "best");
        assert_eq!(payload["speaker_labels"], true);
        assert_eq!(payload["filter_profanity"], true);
        assert_eq!(payload["sentiment_analysis"], true);
        assert_eq!(payload["auto_chapters"], true);
        assert_eq!(payload["entity_detection"], true);
    }

    #[test]
    fn build_payload_bool_flag_false_omitted() {
        // 显式 false 不加入（与 Python `if kwargs.get(...)` 语义一致）
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("speaker_labels", false)
            .build();
        let payload = AssemblyAiAdapter::build_transcript_payload(&req, "https://u", "best");
        assert!(payload.get("speaker_labels").is_none());
    }

    #[test]
    fn build_payload_with_redact_pii() {
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("redact_pii", true)
            .extra("redact_pii_policies", json!(["person_name", "email"]))
            .build();
        let payload = AssemblyAiAdapter::build_transcript_payload(&req, "https://u", "best");
        assert_eq!(payload["redact_pii"], true);
        assert_eq!(
            payload["redact_pii_policies"],
            json!(["person_name", "email"])
        );
    }

    #[test]
    fn build_payload_redact_pii_default_empty_policies() {
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("redact_pii", true)
            .build();
        let payload = AssemblyAiAdapter::build_transcript_payload(&req, "https://u", "best");
        assert_eq!(payload["redact_pii"], true);
        assert_eq!(payload["redact_pii_policies"], json!([]));
    }

    #[test]
    fn build_payload_with_word_boost() {
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("word_boost", json!(["Apple", "Google"]))
            .build();
        let payload = AssemblyAiAdapter::build_transcript_payload(&req, "https://u", "best");
        assert_eq!(payload["word_boost"], json!(["Apple", "Google"]));
    }

    #[test]
    fn build_payload_word_boost_non_array_omitted() {
        // 非 JSON 数组的 word_boost 不透传（与 Python isinstance 检查一致）
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("word_boost", "Apple")
            .build();
        let payload = AssemblyAiAdapter::build_transcript_payload(&req, "https://u", "best");
        assert!(payload.get("word_boost").is_none());
    }

    // ============ parse_response ============

    #[test]
    fn parse_response_simple_text() {
        let v = json!({
            "status": "completed",
            "text": "hello world",
            "language_code": "en",
            "audio_duration": 5.5
        });
        let r = AssemblyAiAdapter::parse_response(&v, "best", "transcribe");
        assert_eq!(r.text, "hello world");
        assert_eq!(r.language.as_deref(), Some("en"));
        assert!((r.duration.unwrap() - 5.5).abs() < f64::EPSILON);
        assert!(r.segments.is_none());
        assert!(r.words.is_none());
        assert_eq!(r.task, "transcribe");
        assert_eq!(r.model.as_deref(), Some("best"));
    }

    #[test]
    fn parse_response_with_utterances_segments_and_words() {
        let v = json!({
            "text": "hello world",
            "utterances": [
                {
                    "start": 0,
                    "end": 1500,
                    "text": "hello",
                    "confidence": 0.95,
                    "speaker": "A",
                    "words": [
                        {"text": "hello", "start": 0, "end": 500, "confidence": 0.99}
                    ]
                },
                {
                    "start": 1500,
                    "end": 3000,
                    "text": "world",
                    "confidence": 0.9,
                    "speaker": "B"
                }
            ]
        });
        let r = AssemblyAiAdapter::parse_response(&v, "best", "transcribe");
        assert_eq!(r.text, "hello world");
        let segs = r.segments.unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].id, 0);
        assert!((segs[0].start - 0.0).abs() < f64::EPSILON);
        assert!((segs[0].end - 1.5).abs() < f64::EPSILON);
        assert_eq!(segs[0].text, "hello");
        assert!((segs[0].confidence.unwrap() - 0.95).abs() < f64::EPSILON);
        assert_eq!(segs[0].speaker.as_deref(), Some("A"));
        assert_eq!(segs[1].speaker.as_deref(), Some("B"));
        // 词级时间戳（来自 utterance[0].words）
        let words = r.words.unwrap();
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "hello");
        assert!((words[0].end - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_response_top_level_words_when_no_utterances() {
        let v = json!({
            "text": "hi",
            "words": [
                {"text": "hi", "start": 0, "end": 200, "confidence": 0.9}
            ]
        });
        let r = AssemblyAiAdapter::parse_response(&v, "nano", "transcribe");
        assert!(r.segments.is_none());
        let words = r.words.unwrap();
        assert_eq!(words.len(), 1);
        assert!((words[0].start - 0.0).abs() < f64::EPSILON);
        assert!((words[0].end - 0.2).abs() < f64::EPSILON);
        assert_eq!(r.model.as_deref(), Some("nano"));
    }

    #[test]
    fn parse_response_translate_task() {
        let v = json!({"text": "translated text"});
        let r = AssemblyAiAdapter::parse_response(&v, "best", "translate");
        assert_eq!(r.task, "translate");
    }

    #[test]
    fn parse_response_missing_fields_defaults() {
        let v = json!({});
        let r = AssemblyAiAdapter::parse_response(&v, "best", "transcribe");
        assert_eq!(r.text, "");
        assert!(r.language.is_none());
        assert!(r.duration.is_none());
        assert!(r.segments.is_none());
        assert!(r.words.is_none());
    }

    #[test]
    fn parse_response_empty_utterances_falls_back_to_top_words() {
        let v = json!({
            "text": "hi",
            "utterances": [],
            "words": [{"text": "hi", "start": 0, "end": 100}]
        });
        let r = AssemblyAiAdapter::parse_response(&v, "best", "transcribe");
        // 空 utterances 数组当作无分段
        assert!(r.segments.is_none());
        // 回退到顶层 words（因为 utterances 为空数组，as_array 返 Some([])，不走 else 分支）
        // 注意：空 utterances 时 segs 为空，words 也为空（来自 utterances 循环）
        assert!(r.words.is_none());
    }

    // ============ ms_to_secs ============

    #[test]
    fn ms_to_secs_converts_milliseconds() {
        assert!((ms_to_secs(Some(&json!(1000))) - 1.0).abs() < f64::EPSILON);
        assert!((ms_to_secs(Some(&json!(2500))) - 2.5).abs() < f64::EPSILON);
        assert!((ms_to_secs(Some(&json!(0))) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ms_to_secs_none_returns_zero() {
        assert!((ms_to_secs(None) - 0.0).abs() < f64::EPSILON);
    }

    // ============ 错误映射 ============

    #[test]
    fn parse_error_message_from_error_field() {
        let body = r#"{"error":"invalid api key"}"#;
        assert_eq!(parse_assemblyai_error_message(body, 401), "invalid api key");
    }

    #[test]
    fn parse_error_message_from_message_field() {
        let body = r#"{"message":"something went wrong"}"#;
        assert_eq!(
            parse_assemblyai_error_message(body, 500),
            "something went wrong"
        );
    }

    #[test]
    fn parse_error_message_fallback_to_body_text() {
        assert_eq!(
            parse_assemblyai_error_message("plain text error", 500),
            "plain text error"
        );
    }

    #[test]
    fn parse_error_message_empty_body_falls_back_to_http_status() {
        assert_eq!(parse_assemblyai_error_message("", 500), "HTTP 500");
        assert_eq!(parse_assemblyai_error_message("   ", 500), "HTTP 500");
    }

    #[test]
    fn map_error_401_is_authentication() {
        let err = map_assemblyai_error(401, r#"{"error":"unauthorized"}"#);
        assert!(matches!(err, AibridgeError::Authentication { .. }));
        assert!(!err.is_retryable());
    }

    #[test]
    fn map_error_403_is_authentication() {
        let err = map_assemblyai_error(403, "forbidden");
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[test]
    fn map_error_429_is_rate_limit() {
        let err = map_assemblyai_error(429, r#"{"error":"too many requests"}"#);
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
        assert!(err.is_retryable());
    }

    #[test]
    fn map_error_400_is_validation() {
        let err = map_assemblyai_error(400, r#"{"error":"bad audio_url"}"#);
        assert!(matches!(err, AibridgeError::Validation { .. }));
        assert!(!err.is_retryable());
    }

    #[test]
    fn map_error_500_is_api_and_retryable() {
        let err = map_assemblyai_error(500, r#"{"error":"internal"}"#);
        assert!(matches!(err, AibridgeError::Api { status: 500, .. }));
        assert!(err.is_retryable());
    }

    #[test]
    fn map_error_other_4xx_is_api() {
        let err = map_assemblyai_error(418, "teapot");
        assert!(matches!(err, AibridgeError::Api { status: 418, .. }));
        assert!(!err.is_retryable());
    }

    // ============ transcribe 完整流程（HTTP，mockito） ============

    #[tokio::test]
    async fn transcribe_full_flow_bytes_input() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .match_header("Authorization", "test-key")
            .match_header("Content-Type", "application/octet-stream")
            .with_status(200)
            .with_body(r#"{"upload_url":"https://up.example.com/u1"}"#)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .match_header("Authorization", "test-key")
            .with_status(200)
            .with_body(r#"{"id":"t-123"}"#)
            .create_async()
            .await;
        server
            .mock("GET", "/transcript/t-123")
            .match_header("Authorization", "test-key")
            .with_status(200)
            .with_body(
                r#"{"status":"completed","text":"hello world","language_code":"en","audio_duration":5.5}"#,
            )
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1, 2, 3]))
            .extra("polling_interval", 0.001)
            .build();
        let result = adapter.transcribe(req).await.unwrap();
        assert_eq!(result.text, "hello world");
        assert_eq!(result.language.as_deref(), Some("en"));
        assert!((result.duration.unwrap() - 5.5).abs() < f64::EPSILON);
        assert_eq!(result.task, "transcribe");
        assert_eq!(result.model.as_deref(), Some("best"));
    }

    #[tokio::test]
    async fn transcribe_nano_model_resolved_in_result() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(200)
            .with_body(r#"{"upload_url":"https://up.example.com/u"}"#)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .match_body(mockito::Matcher::Json(json!({
                "audio_url": "https://up.example.com/u",
                "speech_model": "nano",
                "punctuate": true,
                "format_text": true
            })))
            .with_status(200)
            .with_body(r#"{"id":"t-nano"}"#)
            .create_async()
            .await;
        server
            .mock("GET", "/transcript/t-nano")
            .with_status(200)
            .with_body(r#"{"status":"completed","text":"nano result"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("nano", FileInput::bytes(vec![1]))
            .extra("polling_interval", 0.001)
            .build();
        let result = adapter.transcribe(req).await.unwrap();
        assert_eq!(result.text, "nano result");
        assert_eq!(result.model.as_deref(), Some("nano"));
    }

    #[tokio::test]
    async fn transcribe_unknown_model_defaults_best() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(200)
            .with_body(r#"{"upload_url":"https://up.example.com/u"}"#)
            .create_async()
            .await;
        let create_mock = server
            .mock("POST", TRANSCRIPT_PATH)
            .match_body(mockito::Matcher::PartialJson(
                json!({"speech_model": "best"}),
            ))
            .with_status(200)
            .with_body(r#"{"id":"t-b"}"#)
            .create_async()
            .await;
        server
            .mock("GET", "/transcript/t-b")
            .with_status(200)
            .with_body(r#"{"status":"completed","text":"ok"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("unknown-model", FileInput::bytes(vec![1]))
            .extra("polling_interval", 0.001)
            .build();
        let result = adapter.transcribe(req).await.unwrap();
        assert_eq!(result.model.as_deref(), Some("best"));
        create_mock.assert_async().await;
    }

    #[tokio::test]
    async fn transcribe_url_input_skips_upload() {
        let mut server = mockito::Server::new_async().await;
        // upload 不应被调用
        let upload_mock = server
            .mock("POST", UPLOAD_PATH)
            .expect(0)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .match_body(mockito::Matcher::PartialJson(
                json!({"audio_url": "https://example.com/audio.mp3"}),
            ))
            .with_status(200)
            .with_body(r#"{"id":"t-url"}"#)
            .create_async()
            .await;
        server
            .mock("GET", "/transcript/t-url")
            .with_status(200)
            .with_body(r#"{"status":"completed","text":"url audio"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req =
            TranscribeRequest::builder("best", FileInput::url("https://example.com/audio.mp3"))
                .extra("polling_interval", 0.001)
                .build();
        let result = adapter.transcribe(req).await.unwrap();
        assert_eq!(result.text, "url audio");
        upload_mock.assert_async().await;
    }

    #[tokio::test]
    async fn transcribe_extra_audio_url_skips_upload() {
        let mut server = mockito::Server::new_async().await;
        let upload_mock = server
            .mock("POST", UPLOAD_PATH)
            .expect(0)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .with_status(200)
            .with_body(r#"{"id":"t-x"}"#)
            .create_async()
            .await;
        server
            .mock("GET", "/transcript/t-x")
            .with_status(200)
            .with_body(r#"{"status":"completed","text":"ok"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        // 即使 file 是 Bytes，extra.audio_url 优先，跳过上传
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("audio_url", "https://cdn.example.com/a.mp3")
            .extra("polling_interval", 0.001)
            .build();
        let result = adapter.transcribe(req).await.unwrap();
        assert_eq!(result.text, "ok");
        upload_mock.assert_async().await;
    }

    #[tokio::test]
    async fn transcribe_with_speaker_labels() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(200)
            .with_body(r#"{"upload_url":"https://up.example.com/u"}"#)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .match_body(mockito::Matcher::PartialJson(
                json!({"speaker_labels": true}),
            ))
            .with_status(200)
            .with_body(r#"{"id":"t-sl"}"#)
            .create_async()
            .await;
        server
            .mock("GET", "/transcript/t-sl")
            .with_status(200)
            .with_body(
                r#"{"status":"completed","text":"hi","utterances":[{"start":0,"end":1000,"text":"hi","confidence":0.9,"speaker":"A"}]}"#,
            )
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("speaker_labels", true)
            .extra("polling_interval", 0.001)
            .build();
        let result = adapter.transcribe(req).await.unwrap();
        assert_eq!(result.text, "hi");
        let segs = result.segments.unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].speaker.as_deref(), Some("A"));
        assert!((segs[0].end - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn transcribe_processing_then_completed() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(200)
            .with_body(r#"{"upload_url":"https://up.example.com/u"}"#)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .with_status(200)
            .with_body(r#"{"id":"t-p1"}"#)
            .create_async()
            .await;
        // 第一次轮询返 processing（FIFO：先创建先匹配）
        let processing_mock = server
            .mock("GET", "/transcript/t-p1")
            .with_status(200)
            .with_body(r#"{"status":"processing"}"#)
            .expect(1)
            .create_async()
            .await;
        // 第二次轮询返 completed
        let completed_mock = server
            .mock("GET", "/transcript/t-p1")
            .with_status(200)
            .with_body(r#"{"status":"completed","text":"after poll"}"#)
            .expect(1)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("polling_interval", 0.001)
            .extra("max_polls", 5u64)
            .build();
        let result = adapter.transcribe(req).await.unwrap();
        assert_eq!(result.text, "after poll");
        processing_mock.assert_async().await;
        completed_mock.assert_async().await;
    }

    #[tokio::test]
    async fn transcribe_error_status_returns_api_error() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(200)
            .with_body(r#"{"upload_url":"https://up.example.com/u"}"#)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .with_status(200)
            .with_body(r#"{"id":"t-e1"}"#)
            .create_async()
            .await;
        server
            .mock("GET", "/transcript/t-e1")
            .with_status(200)
            .with_body(r#"{"status":"error","error":"audio too short"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("polling_interval", 0.001)
            .build();
        let err = adapter.transcribe(req).await.unwrap_err();
        match err {
            AibridgeError::Api { message, .. } => {
                assert!(message.contains("audio too short"));
            }
            other => panic!("应为 Api 错误，实际: {other:?}"),
        }
    }

    #[tokio::test]
    async fn transcribe_translate_sets_task() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(200)
            .with_body(r#"{"upload_url":"https://up.example.com/u"}"#)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .with_status(200)
            .with_body(r#"{"id":"t-tr"}"#)
            .create_async()
            .await;
        server
            .mock("GET", "/transcript/t-tr")
            .with_status(200)
            .with_body(r#"{"status":"completed","text":"translated","language_code":"en"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .translate(true)
            .extra("polling_interval", 0.001)
            .build();
        let result = adapter.transcribe(req).await.unwrap();
        assert_eq!(result.text, "translated");
        assert_eq!(result.task, "translate");
    }

    #[tokio::test]
    async fn translate_entry_full_flow_sets_translate_task() {
        // translate() 默认实现置 translate=true 后委托 transcribe → 完整三段式流程，task 为 translate
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(200)
            .with_body(r#"{"upload_url":"https://up.example.com/u"}"#)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .with_status(200)
            .with_body(r#"{"id":"t-tr2"}"#)
            .create_async()
            .await;
        server
            .mock("GET", "/transcript/t-tr2")
            .with_status(200)
            .with_body(r#"{"status":"completed","text":"translated via entry","language_code":"en"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        // 注意：请求本身 translate=false，由 translate() 默认实现置位
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("polling_interval", 0.001)
            .build();
        let result = adapter.translate(req).await.unwrap();
        assert_eq!(result.text, "translated via entry");
        assert_eq!(result.task, "translate");
    }

    // ============ transcribe 错误路径 ============

    #[tokio::test]
    async fn transcribe_upload_401_returns_authentication() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(401)
            .with_body(r#"{"error":"invalid api key"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("polling_interval", 0.001)
            .build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn transcribe_upload_429_returns_rate_limit() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(429)
            .with_body(r#"{"error":"too many requests"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("polling_interval", 0.001)
            .build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn transcribe_create_400_returns_validation() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(200)
            .with_body(r#"{"upload_url":"https://up.example.com/u"}"#)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .with_status(400)
            .with_body(r#"{"error":"invalid audio_url"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("polling_interval", 0.001)
            .build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[tokio::test]
    async fn transcribe_create_500_returns_api() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(200)
            .with_body(r#"{"upload_url":"https://up.example.com/u"}"#)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .with_status(500)
            .with_body(r#"{"error":"internal"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("polling_interval", 0.001)
            .build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Api { status: 500, .. }));
    }

    #[tokio::test]
    async fn transcribe_poll_500_returns_api() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(200)
            .with_body(r#"{"upload_url":"https://up.example.com/u"}"#)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .with_status(200)
            .with_body(r#"{"id":"t-s5"}"#)
            .create_async()
            .await;
        server
            .mock("GET", "/transcript/t-s5")
            .with_status(500)
            .with_body(r#"{"error":"server error"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("polling_interval", 0.001)
            .build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Api { status: 500, .. }));
    }

    #[tokio::test]
    async fn transcribe_max_polls_exhausted_returns_timeout() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", UPLOAD_PATH)
            .with_status(200)
            .with_body(r#"{"upload_url":"https://up.example.com/u"}"#)
            .create_async()
            .await;
        server
            .mock("POST", TRANSCRIPT_PATH)
            .with_status(200)
            .with_body(r#"{"id":"t-to"}"#)
            .create_async()
            .await;
        // 始终返 processing，模拟永不完成
        server
            .mock("GET", "/transcript/t-to")
            .with_status(200)
            .with_body(r#"{"status":"processing"}"#)
            .expect(2)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = TranscribeRequest::builder("best", FileInput::bytes(vec![1]))
            .extra("polling_interval", 0.001)
            .extra("max_polls", 2u64)
            .build();
        let err = adapter.transcribe(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Timeout));
    }
}
