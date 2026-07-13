//! Cartesia 适配器（Sonic 超低延迟 TTS，独立协议）
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/audio_adapters.py` 的 `CartesiaAdapter`。
//!
//! Cartesia 是新一代超低延迟语音合成平台，Sonic 模型以实时性和自然度著称
//!（首包延迟 <200ms），支持多语言、情感控制与声音克隆。
//!
//! ## 协议
//!
//! - 认证：`X-API-Key` header + `Cartesia-Version: 2024-06-10`（非 Bearer）
//! - TTS 合成：`POST /v1/tts/bytes`，请求体 JSON，响应为二进制音频
//! - 音色列表：`GET /tts/voices`，响应 JSON 数组（Python v1 未实现，v2 新增）
//! - 模型列表：硬编码（sonic-2 / sonic-english / sonic-multilingual）
//!
//! ## 特性
//!
//! - `requires_api_key = true`（需 Cartesia API Key）
//! - 音色支持名称（如 "Chinese Woman"）或 voice_id（UUID）；名称经内置
//!   `DEFAULT_VOICES` 表解析为 ID，`extra.voice_id` 优先级最高
//! - 输出格式：mp3（默认）/ wav / ogg / opus / raw / flac，通过 `response_format`
//!   或 `extra.output_format`（完整 dict）控制
//! - `extra` 透传：language / speed / emotion / voice_embedding / continue / add_timestamps
//! - 音色列表带缓存，避免重复网络拉取

use std::collections::HashMap;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::adapter::{Adapter, Capabilities, CapabilitySet};
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::audio::{SpeechRequest, SpeechResult};
use crate::model::common::{ModelInfo, ModelType, VoiceInfo};

// ==================== 常量 ====================

/// Provider 类型标识
const PROVIDER_TYPE: &str = "cartesia";
/// Provider 显示名称
const PROVIDER_NAME: &str = "Cartesia";
/// 默认 API 基地址
const DEFAULT_API_BASE: &str = "https://api.cartesia.ai";
/// 默认 API 版本（Cartesia-Version header 值）
const DEFAULT_API_VERSION: &str = "2024-06-10";
/// TTS 合成端点路径（对应 Python v1 `/v1/tts/bytes`）
const TTS_PATH: &str = "/v1/tts/bytes";
/// 音色列表端点路径（Cartesia 官方 `/tts/voices`）
const VOICES_PATH: &str = "/tts/voices";
/// 默认音色（voice 为空时兜底，对应 Python v1 默认值）
const DEFAULT_VOICE: &str = "Generic Woman";

/// 内置音色名称 → voice_id 映射表
///
/// 对应 Python v1 `CartesiaAdapter.DEFAULT_VOICES`。Cartesia 内置音色的名称
/// 可直接作为 `speech` 的 `voice` 参数，本表将其解析为 UUID 形式的 voice_id。
/// 传入已是 UUID 的 voice_id 时原样透传。
fn lookup_default_voice(name: &str) -> Option<&'static str> {
    let v = match name {
        "Barbershop Man" => "a0e99841-438c-4a64-b679-ae501e7d6091",
        "Miles" => "b27adc08-7b68-4a70-8bd3-4d7f4a4f91e8",
        "Cali Bot" => "a3d1d9d3-5de0-4c8b-8325-04f0e08d0e8a",
        "Customer Support Lady" => "297b6c1f-3135-403e-a9c7-28a7a4dcd690",
        "Doc Brown" => "f114a467-c40a-4db8-964d-8aca59832a2f",
        "Generic Man" => "c8e59920-61aa-494c-8cfe-9730e0ce4c65",
        "Generic Woman" => "b2b21a4e-7f85-4905-a50e-70c04fb7cd9e",
        "Helpdesk Woman" => "70ca091e-a631-4b0c-a08b-9c930ffb0a8e",
        "Japanese Woman" => "8f2d130a-3a73-4d4e-bab7-d0c999371c71",
        "Classy British Lady" => "56553793-917c-41a4-a057-750be38b4b61",
        "Merchant" => "50d0be50-c6b1-4b10-a4dd-e10588f01b06",
        "Movie Guy" => "248be419-c632-4f23-adf1-5324ed7dbf1d",
        "New York Guy" => "01d31f54-303d-480a-9c27-22b9c9f3c85e",
        "News Lady" => "bf9923ea-7a3f-4f62-817d-c38d59f82f33",
        "Nurse" => "660569f7-267b-4240-9b74-0a4bb4dce2e5",
        "Polite Man" => "156fb8d2-335b-4950-9cb3-a2d33bef8830",
        "Salesman" => "87748186-23bb-4158-a1eb-332911b0b706",
        "Southern Woman" => "5d1b15b6-65c6-4f3c-a349-2f2fcc5c13d1",
        "The Don" => "820a3788-2b37-4d21-847a-b65d8a68c99a",
        "The Laughing Guy" => "0a7e6a33-0006-447c-a723-133a0a5b09c8",
        "Chemistry Professor" => "694f9389-aac1-45b6-b726-9d9369183238",
        "Chinese Woman" => "e90c66b3-51bd-4906-a51a-469152d083b1",
        "Sharon" => "e00d0e5a-87e7-469a-9023-d510c6f5f970",
        "Competitive Podcaster" => "8c4a4d43-d33d-4a80-a9a6-a79ce1e39a69",
        _ => return None,
    };
    Some(v)
}

// ==================== Cartesia 原始音色反序列化结构 ====================

/// Cartesia `/tts/voices` 返回的单个音色项（原始 JSON 结构）
///
/// 字段名与 Cartesia 服务端返回的 JSON 一致。`gender` 字段 Cartesia 官方
/// 当前不返回，保留用于兼容；缺失时由 `infer_gender` 按名称启发式推断。
/// 其余字段（description / metadata 等）由 serde 忽略。
#[derive(Debug, Deserialize)]
struct CartesiaVoiceRaw {
    /// 音色 ID（UUID）
    id: String,
    /// 音色名称
    name: String,
    /// 语言代码（如 "zh" / "en"，部分音色为 null）
    #[serde(default)]
    language: Option<String>,
    /// 性别（Cartesia 官方未返回，保留兼容）
    #[serde(default)]
    gender: Option<String>,
}

// ==================== CartesiaAdapter ====================

/// Cartesia 适配器
///
/// 持有 HTTP 客户端、API key / 版本 / 基地址，以及音色列表缓存。
/// speech 合成与 list_voices 均为 per-request HTTP 调用（无长连接）。
pub struct CartesiaAdapter {
    /// Provider 配置（保留供未来扩展，当前字段已在构造时提取）
    #[allow(dead_code)]
    config: ProviderConfig,
    /// HTTP 客户端（封装 reqwest，含连接池与超时）
    http: HttpClient,
    /// API 基地址（默认 `https://api.cartesia.ai`，可由 config.base_url 覆盖）
    api_base: String,
    /// API 版本（Cartesia-Version header 值，默认 `2024-06-10`）
    api_version: String,
    /// API Key（X-API-Key header 值）
    api_key: String,
    /// 音色列表缓存（避免重复网络拉取）
    voices_cache: Mutex<Option<Vec<VoiceInfo>>>,
}

impl CartesiaAdapter {
    /// 创建 Cartesia 适配器
    ///
    /// - `config.base_url` 为空时用 `DEFAULT_API_BASE`
    /// - `config.api_version` 为空时用 `DEFAULT_API_VERSION`
    /// - `config.api_key` 为 None 时用空串（调用时 API 会返 401）
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let api_base = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_API_BASE.to_string());
        let api_version = config
            .api_version
            .clone()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_API_VERSION.to_string());
        let api_key = config.api_key.clone().unwrap_or_default();
        let opts = ClientOptions::builder().timeout(config.timeout).build();
        let http = HttpClient::new(&opts)?;
        Ok(Self {
            config,
            http,
            api_base,
            api_version,
            api_key,
            voices_cache: Mutex::new(None),
        })
    }

    // ==================== 纯函数（协议构造/解析，可单测） ====================

    /// 解析 voice_id：`extra.voice_id` > `DEFAULT_VOICES` 名称查找 > 原样透传
    ///
    /// 对应 Python v1 `CartesiaAdapter._get_voice_id` + `kwargs.get("voice_id")`。
    /// voice 为空时兜底为 `DEFAULT_VOICE`（"Generic Woman"）。
    fn resolve_voice_id(req: &SpeechRequest) -> String {
        if let Some(vid) = req.extra.get("voice_id").and_then(|v| v.as_str()) {
            if !vid.is_empty() {
                return vid.to_string();
            }
        }
        let voice = req.voice.primary().unwrap_or("");
        let voice = if voice.is_empty() {
            DEFAULT_VOICE
        } else {
            voice
        };
        lookup_default_voice(voice).unwrap_or(voice).to_string()
    }

    /// 构造 `voice` 字段：有 `extra.voice_embedding` 时走克隆模式，否则走 id 模式
    ///
    /// 对应 Python v1：默认 `{"mode":"id","id":voice_id}`，有 embedding 时覆盖为
    /// `{"mode":"embedding","embedding":[...]}`。
    fn build_voice(voice_id: &str, extra: &HashMap<String, Value>) -> Value {
        if let Some(embedding) = extra.get("voice_embedding") {
            return json!({"mode": "embedding", "embedding": embedding});
        }
        json!({"mode": "id", "id": voice_id})
    }

    /// 构造 `output_format` 字段
    ///
    /// 优先用 `extra.output_format`（完整 dict）；否则由 `response_format` 构建：
    /// mp3 → `{container, bit_rate:128000, sample_rate:44100}`，
    /// wav → `{container, encoding:pcm_f32le, sample_rate:24000}`，
    /// 其余 → `{container}`。
    fn build_output_format(response_format: &str, extra: &HashMap<String, Value>) -> Value {
        if let Some(of) = extra.get("output_format") {
            if of.is_object() {
                return of.clone();
            }
        }
        let container = if response_format.is_empty() {
            "mp3"
        } else {
            response_format
        }
        .to_lowercase();
        match container.as_str() {
            "mp3" => json!({"container": "mp3", "bit_rate": 128000, "sample_rate": 44100}),
            "wav" => json!({"container": "wav", "encoding": "pcm_f32le", "sample_rate": 24000}),
            _ => json!({"container": container}),
        }
    }

    /// 从 output_format 提取 container（默认 mp3）
    fn container_of(output_format: &Value) -> String {
        output_format
            .get("container")
            .and_then(|v| v.as_str())
            .unwrap_or("mp3")
            .to_string()
    }

    /// container → Accept header 值（MIME 类型）
    ///
    /// 对应 Python v1 `accept_map`。
    fn accept_type(container: &str) -> &'static str {
        match container {
            "mp3" => "audio/mpeg",
            "wav" => "audio/wav",
            "ogg" => "audio/ogg",
            "opus" => "audio/ogg;codec=opus",
            "raw" | "pcm" => "audio/pcm",
            "flac" => "audio/flac",
            _ => "audio/mpeg",
        }
    }

    /// 构造 speech 请求体
    ///
    /// 对应 Python v1 `CartesiaAdapter.speech` 的 payload 构造。可选字段
    ///（language / speed / emotions / continue / add_timestamps）仅在有值时加入。
    fn build_speech_payload(req: &SpeechRequest, voice_id: &str) -> Value {
        let output_format = Self::build_output_format(&req.response_format, &req.extra);
        let voice = Self::build_voice(voice_id, &req.extra);
        let mut payload = json!({
            "model_id": req.model,
            "transcript": req.input,
            "voice": voice,
            "output_format": output_format,
        });

        // language（extra.language）
        if let Some(lang) = req.extra.get("language").and_then(|v| v.as_str()) {
            payload["language"] = json!(lang);
        }

        // speed（req.speed 优先，回退 extra.speed）
        if let Some(speed) = req.speed {
            payload["speed"] = json!(speed);
        } else if let Some(speed) = req.extra.get("speed") {
            payload["speed"] = speed.clone();
        }

        // emotions（req.emotion → [emotion]；或 extra.emotions 列表；或 extra.emotion）
        if let Some(emotion) = req.emotion.as_ref() {
            payload["emotions"] = json!([emotion]);
        } else if let Some(emotions) = req.extra.get("emotions") {
            payload["emotions"] = emotions.clone();
        } else if let Some(emotion) = req.extra.get("emotion") {
            payload["emotions"] = json!([emotion]);
        }

        // continue / continuation
        let cont = req
            .extra
            .get("continue")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            || req
                .extra
                .get("continuation")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
        if cont {
            payload["continue"] = json!(true);
        }

        // add_timestamps
        if req
            .extra
            .get("add_timestamps")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            payload["add_timestamps"] = json!(true);
        }

        payload
    }

    /// 把 Cartesia 原始音色转为统一 `VoiceInfo`
    ///
    /// gender 优先用服务端返回值，缺失时按名称启发式推断。
    fn convert_voice(raw: CartesiaVoiceRaw) -> VoiceInfo {
        let gender = raw.gender.or_else(|| infer_gender(&raw.name));
        VoiceInfo {
            voice_id: Some(raw.id),
            short_name: Some(raw.name.clone()),
            name: Some(raw.name),
            locale: raw.language,
            gender,
            extra: HashMap::new(),
        }
    }

    /// 拉取全量音色列表（带缓存）
    ///
    /// 缓存未命中时 HTTP GET `/tts/voices`，缓存后直接返回（language 过滤在
    /// `list_voices` 做）。
    async fn list_voices_raw(&self) -> Result<Vec<VoiceInfo>> {
        // 先检查缓存
        {
            let cache = self.voices_cache.lock().await;
            if let Some(ref voices) = *cache {
                return Ok(voices.clone());
            }
        }
        let url = format!(
            "{base}{path}",
            base = self.api_base.trim_end_matches('/'),
            path = VOICES_PATH
        );
        let resp = self
            .http
            .inner()
            .get(&url)
            .header("X-API-Key", &self.api_key)
            .header("Cartesia-Version", &self.api_version)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(map_cartesia_error(status_code, &body));
        }
        let raw_voices: Vec<CartesiaVoiceRaw> = resp.json().await.map_err(AibridgeError::from)?;
        let voices: Vec<VoiceInfo> = raw_voices.into_iter().map(Self::convert_voice).collect();

        // 写缓存
        let mut cache = self.voices_cache.lock().await;
        *cache = Some(voices.clone());
        Ok(voices)
    }
}

#[async_trait]
impl Adapter for CartesiaAdapter {
    fn provider_type(&self) -> &str {
        PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    fn capabilities(&self) -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert(Capabilities::AudioSpeech);
        caps.insert(Capabilities::ListVoices);
        caps
    }

    /// Cartesia 需要 API Key（X-API-Key header）
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

    /// 文字转语音（Cartesia Sonic TTS）
    ///
    /// POST `/v1/tts/bytes`，请求体 JSON，响应二进制音频。voice 列表取首个
    ///（Cartesia 不支持音色降级，与 Edge TTS 不同）。
    async fn speech(&self, req: SpeechRequest) -> Result<SpeechResult> {
        let voice_id = Self::resolve_voice_id(&req);
        let payload = Self::build_speech_payload(&req, &voice_id);
        // 从 payload 取 output_format 派生 container/Accept，保证与请求体一致
        let output_format = payload
            .get("output_format")
            .cloned()
            .unwrap_or_else(|| json!({"container": "mp3"}));
        let container = Self::container_of(&output_format);
        let accept = Self::accept_type(&container);
        let url = format!(
            "{base}{path}",
            base = self.api_base.trim_end_matches('/'),
            path = TTS_PATH
        );

        let resp = self
            .http
            .inner()
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .header("Cartesia-Version", &self.api_version)
            .header("Accept", accept)
            .json(&payload)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(map_cartesia_error(status_code, &body));
        }

        let audio_data = resp.bytes().await.map_err(map_reqwest_error)?.to_vec();

        // 空音频检测：TTS 不应返回空音频，判为服务端临时问题（可重试）
        if audio_data.is_empty() {
            return Err(AibridgeError::service_unavailable(
                "Cartesia 返回空音频（可能是限流或服务端抖动），可重试",
            ));
        }

        Ok(SpeechResult {
            audio_data: Some(audio_data),
            audio_url: None,
            audio_base64: None,
            content_type: accept.to_string(),
            format: container,
            duration: None,
            model: Some(if req.model.is_empty() {
                PROVIDER_TYPE.to_string()
            } else {
                req.model.clone()
            }),
        })
    }

    /// 列出可用音色（带缓存，按 language 前缀过滤）
    async fn list_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        let voices = self.list_voices_raw().await?;
        match language {
            Some(lang) if !lang.is_empty() => Ok(voices
                .into_iter()
                .filter(|v| {
                    v.locale
                        .as_deref()
                        .map(|l| l.starts_with(lang))
                        .unwrap_or(false)
                })
                .collect()),
            _ => Ok(voices),
        }
    }

    /// 列出 Cartesia 模型（硬编码，不实时拉取）
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = vec![
            ModelInfo {
                id: "sonic-2".into(),
                name: "Cartesia Sonic 2".into(),
                model_type: ModelType::Audio,
                provider: PROVIDER_TYPE.into(),
                capabilities: vec!["audio_speech".into()],
                max_tokens: None,
                supports_streaming: false,
                description: Some("Cartesia Sonic 2 超低延迟多语言 TTS 模型（首包 <200ms）".into()),
                created: None,
            },
            ModelInfo {
                id: "sonic-english".into(),
                name: "Cartesia Sonic English".into(),
                model_type: ModelType::Audio,
                provider: PROVIDER_TYPE.into(),
                capabilities: vec!["audio_speech".into()],
                max_tokens: None,
                supports_streaming: false,
                description: Some("Cartesia Sonic 英文专用 TTS 模型".into()),
                created: None,
            },
            ModelInfo {
                id: "sonic-multilingual".into(),
                name: "Cartesia Sonic Multilingual".into(),
                model_type: ModelType::Audio,
                provider: PROVIDER_TYPE.into(),
                capabilities: vec!["audio_speech".into()],
                max_tokens: None,
                supports_streaming: false,
                description: Some("Cartesia Sonic 多语言 TTS 模型".into()),
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

/// 将 Cartesia HTTP 错误响应映射为 AibridgeError
///
/// 对应 Python v1 `CartesiaAdapter._handle_error`。映射规则（与任务规格一致）：
/// - 401/403 → Authentication（X-API-Key 无效）
/// - 429 → RateLimit（限流或配额耗尽）
/// - 400/422 → Validation（参数校验错误，携带 details）
/// - 404 → VoiceNotAvailable（voice_id 或模型不存在）
/// - 5xx → Api（服务端错误）
/// - 其余 4xx → Api
fn map_cartesia_error(status: u16, body: &str) -> AibridgeError {
    let message = parse_cartesia_error_message(body, status);
    match status {
        401 | 403 => AibridgeError::authentication(format!(
            "Cartesia 认证失败（X-API-Key 无效或缺失）: {message}"
        )),
        429 => AibridgeError::rate_limit(format!("Cartesia 限流或配额耗尽: {message}")),
        400 | 422 => AibridgeError::validation_with_details(
            format!("Cartesia 参数校验错误: {message}"),
            serde_json::json!({"status": status, "response": body}),
        ),
        404 => {
            AibridgeError::voice_not_available(format!("Cartesia voice_id 或模型不存在: {message}"))
        }
        s if s >= 500 => AibridgeError::api(s, format!("Cartesia 服务端错误 ({s}): {message}")),
        s => AibridgeError::api(s, format!("Cartesia HTTP {s}: {message}")),
    }
}

/// 从 Cartesia 错误响应体解析错误消息
///
/// 尝试解析 JSON 的 `message` 或 `error` 字段（error 可为字符串或
/// `{"message": "..."}` 对象）；解析失败时截取前 200 字符或回退 `HTTP {status}`。
fn parse_cartesia_error_message(body: &str, status: u16) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        if let Some(m) = v.get("message").and_then(|m| m.as_str()) {
            return m.to_string();
        }
        if let Some(error) = v.get("error") {
            if let Some(s) = error.as_str() {
                return s.to_string();
            }
            if let Some(m) = error.get("message").and_then(|m| m.as_str()) {
                return m.to_string();
            }
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

/// 按音色名称启发式推断性别
///
/// Cartesia `/tts/voices` 不返回 gender 字段，本函数按名称关键词推断：
/// 含 "woman"/"lady"/"girl" → Female；含 "man"/"guy"/"boy" → Male；
/// 其余 → None（未知）。注意先判 "woman"（含 "man" 子串）再判 "man"。
fn infer_gender(name: &str) -> Option<String> {
    let lower = name.to_lowercase();
    if lower.contains("woman") || lower.contains("lady") || lower.contains("girl") {
        Some("Female".to_string())
    } else if lower.contains("man") || lower.contains("guy") || lower.contains("boy") {
        Some("Male".to_string())
    } else {
        None
    }
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
    fn make_adapter(base_url: Option<String>) -> CartesiaAdapter {
        let mut opts = ClientOptions::builder().api_key("test-key");
        if let Some(u) = base_url {
            opts = opts.base_url(u);
        }
        let config = ProviderConfig::from_options(PROVIDER_TYPE, opts.build());
        CartesiaAdapter::new(config).expect("构造 CartesiaAdapter 失败")
    }

    // ============ 基本属性 ============

    #[test]
    fn requires_api_key_is_true() {
        let adapter = make_adapter(None);
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn capabilities_contains_speech_and_list_voices() {
        let adapter = make_adapter(None);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::AudioSpeech));
        assert!(caps.contains(&Capabilities::ListVoices));
        assert!(!caps.contains(&Capabilities::Chat));
        assert!(!caps.contains(&Capabilities::ImageGenerate));
    }

    #[test]
    fn provider_type_and_name() {
        let adapter = make_adapter(None);
        assert_eq!(adapter.provider_type(), "cartesia");
        assert_eq!(adapter.provider_name(), "Cartesia");
    }

    #[tokio::test]
    async fn list_models_returns_sonic_models() {
        let adapter = make_adapter(None);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 3);
        assert!(models.iter().any(|m| m.id == "sonic-2"));
        assert!(models.iter().any(|m| m.id == "sonic-english"));
        assert!(models.iter().any(|m| m.id == "sonic-multilingual"));
        for m in &models {
            assert_eq!(m.model_type, ModelType::Audio);
            assert_eq!(m.provider, "cartesia");
        }
    }

    #[tokio::test]
    async fn list_models_filter_by_type() {
        let adapter = make_adapter(None);
        let audio = adapter.list_models(Some(ModelType::Audio)).await.unwrap();
        assert_eq!(audio.len(), 3);
        let chat = adapter.list_models(Some(ModelType::Chat)).await.unwrap();
        assert!(chat.is_empty());
    }

    #[tokio::test]
    async fn start_and_close_are_noops() {
        let mut adapter = make_adapter(None);
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    #[test]
    fn api_version_defaults_when_unset() {
        let adapter = make_adapter(None);
        assert_eq!(adapter.api_version, DEFAULT_API_VERSION);
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
    async fn transcribe_returns_unsupported() {
        let adapter = make_adapter(None);
        let req = TranscribeRequest::builder("m", FileInput::path("/tmp/a.mp3")).build();
        assert!(matches!(
            adapter.transcribe(req).await.unwrap_err(),
            AibridgeError::UnsupportedCapability { .. }
        ));
    }

    // ============ resolve_voice_id ============

    #[test]
    fn resolve_voice_id_name_lookup() {
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman").build();
        assert_eq!(
            CartesiaAdapter::resolve_voice_id(&req),
            "e90c66b3-51bd-4906-a51a-469152d083b1"
        );
        let req = SpeechRequest::builder("sonic-2", "hi", "Barbershop Man").build();
        assert_eq!(
            CartesiaAdapter::resolve_voice_id(&req),
            "a0e99841-438c-4a64-b679-ae501e7d6091"
        );
    }

    #[test]
    fn resolve_voice_id_passthrough_uuid() {
        // 非 known 名称（UUID 形式）原样透传
        let req = SpeechRequest::builder("sonic-2", "hi", "custom-voice-uuid").build();
        assert_eq!(CartesiaAdapter::resolve_voice_id(&req), "custom-voice-uuid");
    }

    #[test]
    fn resolve_voice_id_empty_falls_back_to_default() {
        let req = SpeechRequest::builder("sonic-2", "hi", "").build();
        assert_eq!(
            CartesiaAdapter::resolve_voice_id(&req),
            "b2b21a4e-7f85-4905-a50e-70c04fb7cd9e" // Generic Woman
        );
    }

    #[test]
    fn resolve_voice_id_extra_voice_id_takes_priority() {
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman")
            .extra("voice_id", "override-uuid")
            .build();
        assert_eq!(CartesiaAdapter::resolve_voice_id(&req), "override-uuid");
    }

    #[test]
    fn resolve_voice_id_empty_extra_voice_id_ignored() {
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman")
            .extra("voice_id", "")
            .build();
        assert_eq!(
            CartesiaAdapter::resolve_voice_id(&req),
            "e90c66b3-51bd-4906-a51a-469152d083b1"
        );
    }

    // ============ build_output_format / accept_type / container_of ============

    #[test]
    fn build_output_format_default_mp3() {
        let of = CartesiaAdapter::build_output_format("mp3", &HashMap::new());
        assert_eq!(of["container"], "mp3");
        assert_eq!(of["bit_rate"], 128000);
        assert_eq!(of["sample_rate"], 44100);
    }

    #[test]
    fn build_output_format_wav() {
        let of = CartesiaAdapter::build_output_format("wav", &HashMap::new());
        assert_eq!(of["container"], "wav");
        assert_eq!(of["encoding"], "pcm_f32le");
        assert_eq!(of["sample_rate"], 24000);
    }

    #[test]
    fn build_output_format_empty_defaults_mp3() {
        let of = CartesiaAdapter::build_output_format("", &HashMap::new());
        assert_eq!(of["container"], "mp3");
    }

    #[test]
    fn build_output_format_extra_overrides() {
        let mut extra = HashMap::new();
        extra.insert(
            "output_format".to_string(),
            json!({"container": "raw", "sample_rate": 16000}),
        );
        let of = CartesiaAdapter::build_output_format("mp3", &extra);
        assert_eq!(of["container"], "raw");
        assert_eq!(of["sample_rate"], 16000);
    }

    #[test]
    fn accept_type_mapping() {
        assert_eq!(CartesiaAdapter::accept_type("mp3"), "audio/mpeg");
        assert_eq!(CartesiaAdapter::accept_type("wav"), "audio/wav");
        assert_eq!(CartesiaAdapter::accept_type("ogg"), "audio/ogg");
        assert_eq!(CartesiaAdapter::accept_type("opus"), "audio/ogg;codec=opus");
        assert_eq!(CartesiaAdapter::accept_type("raw"), "audio/pcm");
        assert_eq!(CartesiaAdapter::accept_type("pcm"), "audio/pcm");
        assert_eq!(CartesiaAdapter::accept_type("flac"), "audio/flac");
        assert_eq!(CartesiaAdapter::accept_type("unknown"), "audio/mpeg");
    }

    #[test]
    fn container_of_extracts_or_defaults() {
        assert_eq!(
            CartesiaAdapter::container_of(&json!({"container": "wav"})),
            "wav"
        );
        assert_eq!(CartesiaAdapter::container_of(&json!({})), "mp3");
    }

    // ============ build_voice / build_speech_payload ============

    #[test]
    fn build_voice_id_mode() {
        let v = CartesiaAdapter::build_voice("vid-123", &HashMap::new());
        assert_eq!(v["mode"], "id");
        assert_eq!(v["id"], "vid-123");
    }

    #[test]
    fn build_voice_embedding_mode() {
        let mut extra = HashMap::new();
        extra.insert("voice_embedding".to_string(), json!([0.1, 0.2, 0.3]));
        let v = CartesiaAdapter::build_voice("vid-123", &extra);
        assert_eq!(v["mode"], "embedding");
        assert_eq!(v["embedding"][2], 0.3);
        assert!(v.get("id").is_none());
    }

    #[test]
    fn build_speech_payload_minimal() {
        let req = SpeechRequest::builder("sonic-2", "你好", "Chinese Woman").build();
        let payload = CartesiaAdapter::build_speech_payload(&req, "voice-uuid");
        assert_eq!(payload["model_id"], "sonic-2");
        assert_eq!(payload["transcript"], "你好");
        assert_eq!(payload["voice"]["mode"], "id");
        assert_eq!(payload["voice"]["id"], "voice-uuid");
        assert_eq!(payload["output_format"]["container"], "mp3");
        // 无可选字段时不应出现
        assert!(payload.get("language").is_none());
        assert!(payload.get("speed").is_none());
        assert!(payload.get("emotions").is_none());
        assert!(payload.get("continue").is_none());
        assert!(payload.get("add_timestamps").is_none());
    }

    #[test]
    fn build_speech_payload_with_all_optional() {
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman")
            .speed(1.5)
            .emotion("positivity")
            .extra("language", "zh")
            .extra("continue", true)
            .extra("add_timestamps", true)
            .build();
        let payload = CartesiaAdapter::build_speech_payload(&req, "voice-uuid");
        assert_eq!(payload["language"], "zh");
        assert_eq!(payload["speed"], 1.5);
        assert_eq!(payload["emotions"], json!(["positivity"]));
        assert_eq!(payload["continue"], true);
        assert_eq!(payload["add_timestamps"], true);
    }

    #[test]
    fn build_speech_payload_embedding_overrides_voice() {
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman")
            .extra("voice_embedding", json!([0.1, 0.2]))
            .build();
        let payload = CartesiaAdapter::build_speech_payload(&req, "voice-uuid");
        assert_eq!(payload["voice"]["mode"], "embedding");
        assert!(payload["voice"].get("id").is_none());
    }

    #[test]
    fn build_speech_payload_emotions_from_extra_list() {
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman")
            .extra("emotions", json!(["curiosity", "surprise"]))
            .build();
        let payload = CartesiaAdapter::build_speech_payload(&req, "voice-uuid");
        assert_eq!(payload["emotions"], json!(["curiosity", "surprise"]));
    }

    #[test]
    fn build_speech_payload_speed_from_extra_when_req_speed_none() {
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman")
            .extra("speed", 2.0)
            .build();
        let payload = CartesiaAdapter::build_speech_payload(&req, "voice-uuid");
        assert_eq!(payload["speed"], 2.0);
    }

    #[test]
    fn build_speech_payload_continuation_alias() {
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman")
            .extra("continuation", true)
            .build();
        let payload = CartesiaAdapter::build_speech_payload(&req, "voice-uuid");
        assert_eq!(payload["continue"], true);
    }

    // ============ infer_gender ============

    #[test]
    fn infer_gender_female_keywords() {
        assert_eq!(infer_gender("Chinese Woman").as_deref(), Some("Female"));
        assert_eq!(infer_gender("News Lady").as_deref(), Some("Female"));
        assert_eq!(infer_gender("Business Girl").as_deref(), Some("Female"));
    }

    #[test]
    fn infer_gender_male_keywords() {
        assert_eq!(infer_gender("Barbershop Man").as_deref(), Some("Male"));
        assert_eq!(infer_gender("Movie Guy").as_deref(), Some("Male"));
        assert_eq!(infer_gender("News Boy").as_deref(), Some("Male"));
    }

    #[test]
    fn infer_gender_woman_checked_before_man() {
        // "Woman" 含 "man" 子串，必须先判 woman 返 Female
        assert_eq!(infer_gender("Woman").as_deref(), Some("Female"));
        assert_eq!(infer_gender("Salesman").as_deref(), Some("Male"));
    }

    #[test]
    fn infer_gender_none_for_neutral() {
        assert!(infer_gender("Sharon").is_none());
        assert!(infer_gender("Competitive Podcaster").is_none());
    }

    // ============ convert_voice ============

    #[test]
    fn convert_voice_maps_all_fields() {
        let raw = CartesiaVoiceRaw {
            id: "uuid-1".into(),
            name: "Chinese Woman".into(),
            language: Some("zh".into()),
            gender: None,
        };
        let v = CartesiaAdapter::convert_voice(raw);
        assert_eq!(v.voice_id.as_deref(), Some("uuid-1"));
        assert_eq!(v.short_name.as_deref(), Some("Chinese Woman"));
        assert_eq!(v.name.as_deref(), Some("Chinese Woman"));
        assert_eq!(v.locale.as_deref(), Some("zh"));
        assert_eq!(v.gender.as_deref(), Some("Female")); // 名称推断
    }

    #[test]
    fn convert_voice_uses_explicit_gender_when_present() {
        let raw = CartesiaVoiceRaw {
            id: "uuid-2".into(),
            name: "Some Voice".into(),
            language: None,
            gender: Some("Male".into()),
        };
        let v = CartesiaAdapter::convert_voice(raw);
        assert_eq!(v.gender.as_deref(), Some("Male")); // 服务端值优先，不推断
    }

    // ============ 错误解析 ============

    #[test]
    fn parse_error_message_from_message_field() {
        let body = r#"{"message":"voice not found"}"#;
        assert_eq!(parse_cartesia_error_message(body, 404), "voice not found");
    }

    #[test]
    fn parse_error_message_from_error_string() {
        let body = r#"{"error":"bad request"}"#;
        assert_eq!(parse_cartesia_error_message(body, 400), "bad request");
    }

    #[test]
    fn parse_error_message_from_error_object() {
        let body = r#"{"error":{"message":"invalid model_id"}}"#;
        assert_eq!(parse_cartesia_error_message(body, 422), "invalid model_id");
    }

    #[test]
    fn parse_error_message_fallback_to_body_text() {
        assert_eq!(
            parse_cartesia_error_message("plain text error", 500),
            "plain text error"
        );
    }

    #[test]
    fn parse_error_message_empty_body_falls_back_to_http_status() {
        assert_eq!(parse_cartesia_error_message("", 500), "HTTP 500");
        assert_eq!(parse_cartesia_error_message("   ", 500), "HTTP 500");
    }

    #[test]
    fn map_cartesia_error_401_is_authentication() {
        let err = map_cartesia_error(401, r#"{"message":"invalid api key"}"#);
        assert!(matches!(err, AibridgeError::Authentication { .. }));
        assert!(!err.is_retryable());
    }

    #[test]
    fn map_cartesia_error_403_is_authentication() {
        let err = map_cartesia_error(403, "forbidden");
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[test]
    fn map_cartesia_error_429_is_rate_limit() {
        let err = map_cartesia_error(429, r#"{"message":"slow down"}"#);
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
        assert!(err.is_retryable());
    }

    #[test]
    fn map_cartesia_error_400_is_validation() {
        let err = map_cartesia_error(400, r#"{"message":"bad param"}"#);
        assert!(matches!(err, AibridgeError::Validation { .. }));
        assert!(!err.is_retryable());
    }

    #[test]
    fn map_cartesia_error_422_is_validation() {
        let err = map_cartesia_error(422, r#"{"error":"unprocessable"}"#);
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[test]
    fn map_cartesia_error_404_is_voice_not_available() {
        let err = map_cartesia_error(404, r#"{"message":"voice not found"}"#);
        assert!(matches!(err, AibridgeError::VoiceNotAvailable { .. }));
    }

    #[test]
    fn map_cartesia_error_500_is_api_and_retryable() {
        let err = map_cartesia_error(500, r#"{"message":"server error"}"#);
        assert!(matches!(err, AibridgeError::Api { status: 500, .. }));
        assert!(err.is_retryable());
    }

    #[test]
    fn map_cartesia_error_other_4xx_is_api() {
        let err = map_cartesia_error(418, "teapot");
        assert!(matches!(err, AibridgeError::Api { status: 418, .. }));
        assert!(!err.is_retryable());
    }

    // ============ speech（HTTP，mockito） ============

    #[tokio::test]
    async fn speech_returns_binary_audio() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", TTS_PATH)
            .match_header("X-API-Key", "test-key")
            .match_header("Cartesia-Version", "2024-06-10")
            .match_header("Accept", "audio/mpeg")
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .with_body(b"\xFF\xFB\x90\x00\x01\x02\x03")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = SpeechRequest::builder("sonic-2", "你好世界", "Chinese Woman").build();
        let result = adapter.speech(req).await.unwrap();
        assert_eq!(
            result.audio_data,
            Some(vec![0xFF, 0xFB, 0x90, 0x00, 0x01, 0x02, 0x03])
        );
        assert_eq!(result.content_type, "audio/mpeg");
        assert_eq!(result.format, "mp3");
        assert_eq!(result.model.as_deref(), Some("sonic-2"));
    }

    #[tokio::test]
    async fn speech_sends_correct_payload() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", TTS_PATH)
            .match_header("X-API-Key", "test-key")
            .match_body(mockito::Matcher::Json(json!({
                "model_id": "sonic-2",
                "transcript": "hello",
                "voice": {"mode": "id", "id": "e90c66b3-51bd-4906-a51a-469152d083b1"},
                "output_format": {"container": "mp3", "bit_rate": 128000, "sample_rate": 44100}
            })))
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .with_body(b"\x00\x01\x02")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = SpeechRequest::builder("sonic-2", "hello", "Chinese Woman").build();
        let result = adapter.speech(req).await.unwrap();
        assert_eq!(result.audio_data, Some(vec![0x00, 0x01, 0x02]));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn speech_wav_format_sets_accept_header() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", TTS_PATH)
            .match_header("Accept", "audio/wav")
            .with_status(200)
            .with_header("content-type", "audio/wav")
            .with_body(b"RIFFxxxxWAVE")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman")
            .response_format("wav")
            .build();
        let result = adapter.speech(req).await.unwrap();
        assert_eq!(result.format, "wav");
        assert_eq!(result.content_type, "audio/wav");
    }

    #[tokio::test]
    async fn speech_401_returns_authentication() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", TTS_PATH)
            .with_status(401)
            .with_body(r#"{"message":"Invalid API key"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn speech_429_returns_rate_limit() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", TTS_PATH)
            .with_status(429)
            .with_body(r#"{"message":"rate limit"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn speech_400_returns_validation() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", TTS_PATH)
            .with_status(400)
            .with_body(r#"{"error":"transcript too long"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[tokio::test]
    async fn speech_500_returns_api() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", TTS_PATH)
            .with_status(500)
            .with_body(r#"{"message":"internal error"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Api { status: 500, .. }));
    }

    #[tokio::test]
    async fn speech_404_returns_voice_not_available() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", TTS_PATH)
            .with_status(404)
            .with_body(r#"{"message":"voice not found"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::VoiceNotAvailable { .. }));
    }

    #[tokio::test]
    async fn speech_empty_audio_returns_service_unavailable() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", TTS_PATH)
            .with_status(200)
            .with_body(b"")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let req = SpeechRequest::builder("sonic-2", "hi", "Chinese Woman").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ServiceUnavailable { .. }));
    }

    // ============ list_voices（HTTP，mockito） ============

    #[tokio::test]
    async fn list_voices_fetches_and_parses() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!([
            {"id": "e90c66b3-51bd-4906-a51a-469152d083b1", "name": "Chinese Woman", "language": "zh"},
            {"id": "a0e99841-438c-4a64-b679-ae501e7d6091", "name": "Barbershop Man", "language": "en"}
        ]);
        server
            .mock("GET", VOICES_PATH)
            .match_header("X-API-Key", "test-key")
            .match_header("Cartesia-Version", "2024-06-10")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let voices = adapter.list_voices(None).await.unwrap();
        assert_eq!(voices.len(), 2);
        assert_eq!(
            voices[0].voice_id.as_deref(),
            Some("e90c66b3-51bd-4906-a51a-469152d083b1")
        );
        assert_eq!(voices[0].name.as_deref(), Some("Chinese Woman"));
        assert_eq!(voices[0].locale.as_deref(), Some("zh"));
        assert_eq!(voices[0].gender.as_deref(), Some("Female")); // 名称推断
        assert_eq!(voices[1].gender.as_deref(), Some("Male")); // 名称推断
    }

    #[tokio::test]
    async fn list_voices_filters_by_language() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!([
            {"id": "v1", "name": "Chinese Woman", "language": "zh"},
            {"id": "v2", "name": "Japanese Woman", "language": "ja"},
            {"id": "v3", "name": "Barbershop Man", "language": "en"}
        ]);
        server
            .mock("GET", VOICES_PATH)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let zh = adapter.list_voices(Some("zh")).await.unwrap();
        assert_eq!(zh.len(), 1);
        assert_eq!(zh[0].voice_id.as_deref(), Some("v1"));
        let none = adapter.list_voices(Some("fr")).await.unwrap();
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn list_voices_caches_second_call() {
        let mut server = mockito::Server::new_async().await;
        let m = server
            .mock("GET", VOICES_PATH)
            .with_status(200)
            .with_body(
                serde_json::json!([{"id": "v1", "name": "Chinese Woman", "language": "zh"}])
                    .to_string(),
            )
            .expect(1) // 第二次走缓存，不命中 HTTP
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let _ = adapter.list_voices(None).await.unwrap();
        let _ = adapter.list_voices(None).await.unwrap(); // 走缓存
        m.assert_async().await;
    }

    #[tokio::test]
    async fn list_voices_401_returns_authentication() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("GET", VOICES_PATH)
            .with_status(401)
            .with_body(r#"{"message":"invalid key"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let err = adapter.list_voices(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn list_voices_500_returns_api() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("GET", VOICES_PATH)
            .with_status(503)
            .with_body(r#"{"message":"unavailable"}"#)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let err = adapter.list_voices(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Api { .. }));
    }

    // ============ recommend_voices ============

    #[tokio::test]
    async fn recommend_voices_filters_by_gender_and_limit() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!([
            {"id": "v1", "name": "Chinese Woman", "language": "zh"},
            {"id": "v2", "name": "Barbershop Man", "language": "zh"},
            {"id": "v3", "name": "Japanese Woman", "language": "ja"}
        ]);
        server
            .mock("GET", VOICES_PATH)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        // 中文 + Female → 仅 v1
        let female_zh = adapter
            .recommend_voices(Some("zh"), Some("Female"), 10)
            .await
            .unwrap();
        assert_eq!(female_zh.len(), 1);
        assert_eq!(female_zh[0].voice_id.as_deref(), Some("v1"));

        // 中文不限性别 → v1 + v2，limit=1 → 仅 v1
        let limited = adapter.recommend_voices(Some("zh"), None, 1).await.unwrap();
        assert_eq!(limited.len(), 1);
    }

    #[tokio::test]
    async fn recommend_voices_male_filter() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!([
            {"id": "v1", "name": "Chinese Woman", "language": "zh"},
            {"id": "v2", "name": "Barbershop Man", "language": "zh"},
            {"id": "v3", "name": "Movie Guy", "language": "en"}
        ]);
        server
            .mock("GET", VOICES_PATH)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let males = adapter
            .recommend_voices(None, Some("male"), 10)
            .await
            .unwrap();
        assert_eq!(males.len(), 2); // v2 + v3
    }
}
