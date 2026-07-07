//! ElevenLabs 适配器（高质量多语言 TTS）
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/audio_adapters.py` 的 `ElevenLabsAdapter`。
//!
//! ElevenLabs 是全球流行的高质量语音合成平台，支持超逼真音色、多语言、声音克隆。
//!
//! ## 协议（独立协议，非 OpenAI 兼容）
//!
//! - Base URL: `https://api.elevenlabs.io/v1`
//! - 认证: `xi-api-key` 请求头（非 Bearer）
//! - speech: `POST /text-to-speech/{voice_id}`
//!   - 请求体: `{ "text", "model_id", "voice_settings"?: { stability, similarity_boost, style, use_speaker_boost } }`
//!   - 查询参数: `output_format`（非 mp3 时携带，如 wav/ogg/opus/ulaw/pcm）
//!   - 响应: 二进制音频流（mp3 默认）
//! - list_voices: `GET /voices`，返回 `{ "voices": [{ voice_id, name, labels, category }] }`
//! - list_models: 无标准 /models 拉取需求，保留硬编码 TTS 模型列表
//!
//! ## 错误映射
//!
//! - 401/403 → Authentication（xi-api-key 无效）
//! - 429 → RateLimit（限流或配额耗尽）
//! - 400/422 → Validation（请求参数错误）
//! - 404 → VoiceNotAvailable（voice_id 或模型不存在）
//! - 5xx → Api（服务端错误）
//! - 其余 4xx → Api
//!
//! ## 特性（v1.3.3 保留）
//!
//! - `requires_api_key = true`（需 API Key）
//! - 内置音色别名表（"Rachel" → voice_id），未知名称视为直接传 voice_id
//! - 收到候选音色列表时取第一个（不实现多音色降级，与 Python v1 一致）

use std::collections::HashMap;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::adapter::{Adapter, Capabilities, CapabilitySet};
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::audio::{SpeechRequest, SpeechResult};
use crate::model::common::{ModelInfo, ModelType, VoiceInfo};

// ==================== 常量 ====================

/// Provider 类型标识
const PROVIDER_TYPE: &str = "elevenlabs";
/// Provider 显示名称
const PROVIDER_NAME: &str = "ElevenLabs";
/// 默认 API 基地址
const DEFAULT_API_BASE: &str = "https://api.elevenlabs.io/v1";
/// 默认模型（多语言 v2，ElevenLabs 推荐）
const DEFAULT_MODEL: &str = "eleven_multilingual_v2";
/// 默认音色 voice_id（Rachel，ElevenLabs 最常用女声）
const DEFAULT_VOICE_ID: &str = "21m00Tcm4TlvDq8ikWAM";
/// TTS 合成端点路径前缀
const TTS_PATH: &str = "/text-to-speech/";
/// 音色列表端点路径
const VOICES_PATH: &str = "/voices";

// ==================== ElevenLabs 原始音色反序列化结构 ====================

/// ElevenLabs /voices 返回的单个音色项（原始 JSON 结构）
///
/// 字段名与 ElevenLabs 服务端返回的 JSON 一致（snake_case）。
#[derive(Debug, Deserialize)]
struct ElevenVoiceRaw {
    /// 音色 ID（如 "21m00Tcm4TlvDq8ikWAM"）
    voice_id: String,
    /// 音色显示名（如 "Rachel"）
    name: String,
    /// 标签集合（含 gender / language / accent 等元数据）
    #[serde(default)]
    labels: HashMap<String, String>,
    /// 音色类别（如 "premade" / "cloned"）
    #[serde(default)]
    category: Option<String>,
}

/// ElevenLabs /voices 响应外壳
#[derive(Debug, Deserialize)]
struct ElevenVoicesResponse {
    /// 音色列表
    voices: Vec<ElevenVoiceRaw>,
}

// ==================== ElevenLabsAdapter ====================

/// ElevenLabs 适配器
///
/// 持有 HTTP 客户端与 API Key。所有请求按需发起（无长连接）。
pub struct ElevenLabsAdapter {
    /// Provider 配置
    #[allow(dead_code)]
    config: ProviderConfig,
    /// HTTP 客户端
    http: HttpClient,
    /// API Key（xi-api-key 请求头用）
    api_key: Option<String>,
    /// API 基地址（默认 `https://api.elevenlabs.io/v1`，可由 config.base_url 覆盖）
    api_base: String,
}

impl ElevenLabsAdapter {
    /// 创建 ElevenLabs 适配器
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

    /// 解析音色：别名（如 "Rachel"）→ voice_id，未知则原样返回（视为直接传了 voice_id）
    ///
    /// 对应 Python v1 `ElevenLabsAdapter._get_voice_id`。
    /// - 空值 → 默认音色 Rachel 的 voice_id
    /// - 已知别名（大小写不敏感）→ 对应 voice_id
    /// - 其余 → 原样返回
    fn resolve_voice_id(voice: &str) -> String {
        if voice.is_empty() {
            return DEFAULT_VOICE_ID.to_string();
        }
        if let Some(id) = lookup_default_voice(voice) {
            return id.to_string();
        }
        voice.to_string()
    }

    /// 构造 speech 请求体
    ///
    /// 对应 Python v1 `ElevenLabsAdapter.speech` 的 payload 构造。
    /// 从 `extra` 提取 ElevenLabs 特有参数（stability/similarity_boost/style/use_speaker_boost）
    /// 组装到 `voice_settings`。
    fn build_speech_payload(input: &str, model: &str, extra: &HashMap<String, Value>) -> Value {
        let mut payload = serde_json::json!({
            "text": input,
            "model_id": model,
        });
        let mut voice_settings = serde_json::Map::new();
        for key in [
            "stability",
            "similarity_boost",
            "style",
            "use_speaker_boost",
        ] {
            if let Some(v) = extra.get(key) {
                voice_settings.insert(key.to_string(), v.clone());
            }
        }
        if !voice_settings.is_empty() {
            payload["voice_settings"] = Value::Object(voice_settings);
        }
        payload
    }

    /// 根据输出格式返回对应 Content-Type
    ///
    /// 对应 Python v1 `ElevenLabsAdapter.speech` 的 `content_type_map`。
    fn content_type_for_format(fmt: &str) -> &'static str {
        match fmt.to_lowercase().as_str() {
            "mp3" => "audio/mpeg",
            "wav" => "audio/wav",
            "ogg" => "audio/ogg",
            "opus" => "audio/opus",
            "ulaw" => "audio/basic",
            "pcm" => "audio/pcm",
            _ => "audio/mpeg",
        }
    }

    /// 把 ElevenLabs 原始音色转为统一 `VoiceInfo`
    ///
    /// - voice_id → voice_id + short_name（ElevenLabs 用 voice_id 作唯一标识）
    /// - name → name
    /// - labels.gender → gender（标准化为 "Female"/"Male"）
    /// - labels.language → locale（标准化为语言代码如 "en"/"zh"）
    /// - labels 其余字段 + category → extra
    fn convert_voice(raw: ElevenVoiceRaw) -> VoiceInfo {
        let gender = raw.labels.get("gender").map(|g| capitalize_gender(g));
        let locale = raw
            .labels
            .get("language")
            .map(|l| normalize_language_code(l));
        let voice_id = raw.voice_id.clone();
        let mut builder = VoiceInfo::builder()
            .voice_id(voice_id.clone())
            .short_name(voice_id)
            .name(raw.name);
        if let Some(g) = gender {
            builder = builder.gender(g);
        }
        if let Some(l) = locale {
            builder = builder.locale(l);
        }
        let mut info = builder.build();
        for (k, val) in raw.labels {
            info.extra.insert(k, Value::String(val));
        }
        if let Some(cat) = raw.category {
            info.extra
                .insert("category".to_string(), Value::String(cat));
        }
        info
    }

    /// 将 ElevenLabs HTTP 错误响应映射为 AibridgeError
    ///
    /// 映射规则见模块文档"错误映射"小节。
    fn map_elevenlabs_error(status: u16, body: &str) -> AibridgeError {
        match status {
            401 | 403 => AibridgeError::authentication(format!(
                "ElevenLabs 认证失败（xi-api-key 无效）: {body}"
            )),
            429 => AibridgeError::rate_limit(format!("ElevenLabs 限流或配额耗尽: {body}")),
            400 | 422 => AibridgeError::validation(format!("ElevenLabs 请求参数错误: {body}")),
            404 => AibridgeError::voice_not_available(format!(
                "ElevenLabs voice_id 或模型不存在: {body}"
            )),
            s if s >= 500 => AibridgeError::api(s, format!("ElevenLabs 服务错误 ({s}): {body}")),
            s => AibridgeError::api(s, format!("ElevenLabs HTTP {s}: {body}")),
        }
    }

    /// 构造 speech 端点完整 URL（`{api_base}/text-to-speech/{voice_id}`）
    fn build_tts_url(api_base: &str, voice_id: &str) -> String {
        format!(
            "{base}{path}{voice_id}",
            base = api_base.trim_end_matches('/'),
            path = TTS_PATH,
            voice_id = voice_id,
        )
    }

    /// 构造 list_voices 端点完整 URL（`{api_base}/voices`）
    fn build_voices_url(api_base: &str) -> String {
        format!(
            "{base}{path}",
            base = api_base.trim_end_matches('/'),
            path = VOICES_PATH,
        )
    }

    /// 获取 API Key，为空时返 Validation 错误
    fn require_api_key(&self) -> Result<&str> {
        self.api_key
            .as_deref()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| AibridgeError::validation("ElevenLabs 需要 API key（xi-api-key）"))
    }
}

#[async_trait]
impl Adapter for ElevenLabsAdapter {
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

    /// 文字转语音（ElevenLabs TTS）
    ///
    /// 收到候选音色列表时取第一个（不实现多音色降级，与 Python v1 一致）。
    async fn speech(&self, req: SpeechRequest) -> Result<SpeechResult> {
        let api_key = self.require_api_key()?;

        // 取主音色并解析为 voice_id
        let voice_raw = req.voice.primary().unwrap_or_default();
        let voice_id = Self::resolve_voice_id(voice_raw);

        // 模型缺省回退到默认模型
        let model = if req.model.is_empty() {
            DEFAULT_MODEL.to_string()
        } else {
            req.model.clone()
        };

        // 输出格式（默认 mp3）
        let fmt = if req.response_format.is_empty() {
            "mp3"
        } else {
            req.response_format.as_str()
        };
        let content_type = Self::content_type_for_format(fmt);

        let payload = Self::build_speech_payload(&req.input, &model, &req.extra);
        let url = Self::build_tts_url(&self.api_base, &voice_id);

        // 构造请求：xi-api-key + Accept + JSON body；非 mp3 时附加 output_format 查询参数
        let mut request = self
            .http
            .inner()
            .post(&url)
            .header("xi-api-key", api_key)
            .header("Accept", content_type)
            .json(&payload);
        if fmt != "mp3" {
            request = request.query(&[("output_format", fmt)]);
        }

        let resp = request.send().await.map_err(map_reqwest_err)?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(Self::map_elevenlabs_error(status.as_u16(), &body));
        }

        let audio_bytes = resp.bytes().await.map_err(map_reqwest_err)?;

        Ok(SpeechResult {
            audio_data: Some(audio_bytes.to_vec()),
            audio_url: None,
            audio_base64: None,
            content_type: content_type.to_string(),
            format: fmt.to_string(),
            duration: None,
            model: Some(model),
        })
    }

    /// 列出可用音色（按 language 过滤）
    ///
    /// language 过滤对 labels.language 做标准化后前缀匹配（如 "en" 匹配 "english"）。
    async fn list_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        let api_key = self.require_api_key()?;
        let url = Self::build_voices_url(&self.api_base);

        let resp = self
            .http
            .inner()
            .get(&url)
            .header("xi-api-key", api_key)
            .send()
            .await
            .map_err(map_reqwest_err)?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(Self::map_elevenlabs_error(status.as_u16(), &body));
        }

        let bytes = resp.bytes().await.map_err(map_reqwest_err)?;
        let resp_json: ElevenVoicesResponse = serde_json::from_slice(&bytes).map_err(|e| {
            AibridgeError::validation(format!("ElevenLabs /voices 响应解析失败: {e}"))
        })?;
        let voices: Vec<VoiceInfo> = resp_json
            .voices
            .into_iter()
            .map(Self::convert_voice)
            .collect();

        match language {
            Some(lang) if !lang.is_empty() => Ok(voices
                .into_iter()
                .filter(|v| voice_matches_language(lang, v))
                .collect()),
            _ => Ok(voices),
        }
    }

    /// 列出 ElevenLabs TTS 模型（硬编码列表，无 /models 拉取需求）
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = elevenlabs_models();
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }
}

// ==================== 辅助函数 ====================

/// 查询内置音色别名 → voice_id
///
/// 对应 Python v1 `ElevenLabsAdapter.DEFAULT_VOICES`。返回 None 表示非已知别名。
fn lookup_default_voice(voice: &str) -> Option<&'static str> {
    let lower = voice.to_lowercase();
    let id = match lower.as_str() {
        "rachel" => "21m00Tcm4TlvDq8ikWAM",
        "drew" => "29vD33N1CtxCmqQRPOHJ",
        "clyde" => "2EiwWnXFnvU5JabPnv8n",
        "paul" => "5Q0t7uMcjvnagumLfvZi",
        "domi" => "AZnzlk1XvdvUeBnXmlld",
        "dave" => "CYw3kZ02Hs0563khs1Fj",
        "fin" => "D38z5RcWu1voky8WS1ja",
        "sarah" => "EXAVITQu4vr4xnSDxMaL",
        "antoni" => "ErXwobaYiN019PkySvjV",
        "thomas" => "GBv7mTt0atIp3Br8iCZE",
        "charlie" => "IKne3meq5aSn9XLyUdCD",
        "george" => "JBFqnCBsd32t6Ie9FZ2Q",
        "emily" => "LcfcDJNUP1GQjkzn1xUU",
        "elli" => "MF3mGyEYCl7XYWbV9V6O",
        "callum" => "N2lVS1w4EtoT3dr4eOWO",
        "patrick" => "ODq5zmih8GrVes37DekR",
        "harry" => "SOYHLrjzK2X1ezoPC6cr",
        "liam" => "TX3LPaxmHKxFdv7VOQHJ",
        "dorothy" => "ThT5KcBeYPX3keUQqHPh",
        "josh" => "TxGEqnHWrfWFTfGW9XjX",
        "arnold" => "VR6AewLTigWG4xSOukaG",
        "charlotte" => "XB0fDUnXU5powFXDhCwa",
        "alice" => "Xb7hH8MSUJpSbSDYk0k2",
        "matilda" => "XrExE9yKIg1WjnnlVkGX",
        "matthew" => "Yko7PKHZNXotIFUBG7I9",
        "james" => "ZQe5CZNOzWyzPSCn5a3c",
        "joseph" => "Zlb1dXrM653N07WRPnSh",
        "jeremy" => "bVMeCyTHy58xNoL34h3p",
        "michael" => "flq6f7yk4E4fJM5XTYuZ",
        "ethan" => "g5CIjZEefAph4nQFvHAz",
        "chris" => "iP95p4xoKVk53GoZ742B",
        "gigi" => "jBpfuIE2acCO8z3wKNLl",
        "freya" => "jsCqWAovK2LkecY7zXl4",
        "brian" => "nPczCjzI2devxB14oouP",
        "grace" => "oWAxZDx7w5VEj9dCyTzz",
        "daniel" => "onwK4e9ZLuTAKqWW03F9",
        "lily" => "pFZP5JQG7iQjIQuC4Bku",
        "serena" => "pMsXgVXv3BLzUgSXRplE",
        "adam" => "pNInz6obpgDQGcFmaJgB",
        "nicole" => "piTKgcLEGmPE4e6mEKli",
        "bill" => "pqHfZKP75CvOlQylNhV4",
        "jessie" => "t0jbNlBVZ17f02VDIeMI",
        "sam" => "yoZ06aMxZJJ28mfd3POQ",
        "glinda" => "z9fAnlkpzviVjnFOo0Tc",
        "giovanni" => "zcAOhNBS3c14rBihAFp1",
        "mimi" => "zrHiDhphv9ZnVXBqCLjz",
        _ => return None,
    };
    Some(id)
}

/// 性别字符串标准化（female/f → Female，male/m → Male）
fn capitalize_gender(g: &str) -> String {
    match g.to_lowercase().as_str() {
        "female" | "f" | "woman" => "Female".to_string(),
        "male" | "m" | "man" => "Male".to_string(),
        _ => g.to_string(),
    }
}

/// 语言名称/代码标准化为 2 字母代码（便于前缀匹配）
///
/// ElevenLabs labels.language 可能是自然语言名（"english"/"chinese"）或代码（"en"/"zh-CN"），
/// 统一映射到 2 字母代码做过滤匹配。
fn normalize_language_code(s: &str) -> String {
    let lower = s.to_lowercase();
    let code = match lower.as_str() {
        "english" | "en" | "en-us" | "en-gb" | "en-au" | "en-in" => "en",
        "chinese" | "zh" | "zh-cn" | "zh-tw" | "mandarin" | "cantonese" => "zh",
        "japanese" | "ja" | "ja-jp" => "ja",
        "korean" | "ko" | "ko-kr" => "ko",
        "spanish" | "es" | "es-es" | "es-mx" => "es",
        "french" | "fr" | "fr-fr" | "fr-ca" => "fr",
        "german" | "de" | "de-de" => "de",
        "italian" | "it" | "it-it" => "it",
        "portuguese" | "pt" | "pt-br" | "pt-pt" => "pt",
        "russian" | "ru" | "ru-ru" => "ru",
        "hindi" | "hi" => "hi",
        "arabic" | "ar" => "ar",
        "dutch" | "nl" => "nl",
        "polish" | "pl" => "pl",
        "turkish" | "tr" => "tr",
        "indonesian" | "id" => "id",
        "vietnamese" | "vi" => "vi",
        "thai" | "th" => "th",
        _ => &lower,
    };
    code.to_string()
}

/// 判断音色是否匹配指定语言（标准化后前缀互匹配）
///
/// 用于 list_voices 的 language 过滤。filter 与 voice 的 locale 双向前缀匹配，
/// 兼容 "en" ↔ "english"（均标准化为 "en"）。
fn voice_matches_language(filter: &str, voice: &VoiceInfo) -> bool {
    let f = normalize_language_code(filter);
    if f.is_empty() {
        return true;
    }
    voice
        .locale
        .as_deref()
        .map(|l| {
            let norm = normalize_language_code(l);
            !norm.is_empty() && (norm.starts_with(&f) || f.starts_with(&norm))
        })
        .unwrap_or(false)
}

/// 将 reqwest::Error 映射为 AibridgeError（超时 → Timeout，其余 → Network）
fn map_reqwest_err(e: reqwest::Error) -> AibridgeError {
    if e.is_timeout() {
        AibridgeError::Timeout
    } else {
        AibridgeError::Network(e)
    }
}

/// ElevenLabs TTS 模型硬编码列表
fn elevenlabs_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "eleven_multilingual_v2".into(),
            name: "Eleven Multilingual v2".into(),
            model_type: ModelType::Audio,
            provider: PROVIDER_TYPE.into(),
            capabilities: vec!["audio_speech".into()],
            max_tokens: None,
            supports_streaming: true,
            description: Some("ElevenLabs 推荐的多语言 TTS 模型，支持 29 种语言，质量最高".into()),
            created: None,
        },
        ModelInfo {
            id: "eleven_turbo_v2".into(),
            name: "Eleven Turbo v2".into(),
            model_type: ModelType::Audio,
            provider: PROVIDER_TYPE.into(),
            capabilities: vec!["audio_speech".into()],
            max_tokens: None,
            supports_streaming: true,
            description: Some("低延迟 TTS 模型，兼顾质量与速度，适合实时场景".into()),
            created: None,
        },
        ModelInfo {
            id: "eleven_flash_v2_5".into(),
            name: "Eleven Flash v2.5".into(),
            model_type: ModelType::Audio,
            provider: PROVIDER_TYPE.into(),
            capabilities: vec!["audio_speech".into()],
            max_tokens: None,
            supports_streaming: true,
            description: Some("最快 TTS 模型，极低延迟，适合高并发场景".into()),
            created: None,
        },
        ModelInfo {
            id: "eleven_multilingual_v1".into(),
            name: "Eleven Multilingual v1".into(),
            model_type: ModelType::Audio,
            provider: PROVIDER_TYPE.into(),
            capabilities: vec!["audio_speech".into()],
            max_tokens: None,
            supports_streaming: true,
            description: Some("多语言 TTS 模型 v1（已弃用，建议用 v2）".into()),
            created: None,
        },
        ModelInfo {
            id: "eleven_monolingual_v1".into(),
            name: "Eleven Monolingual v1".into(),
            model_type: ModelType::Audio,
            provider: PROVIDER_TYPE.into(),
            capabilities: vec!["audio_speech".into()],
            max_tokens: None,
            supports_streaming: true,
            description: Some("单语种（英文）TTS 模型 v1".into()),
            created: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientOptions;
    use crate::model::audio::TranscribeRequest;
    use crate::model::chat::ChatRequest;
    use crate::model::image::{FileInput, ImageRequest};
    use crate::model::options::{EmbedInput, EmbedRequest};
    use crate::model::video::VideoRequest;

    /// 构造测试用适配器（base_url 指向 mockito server，api_key 可选）
    fn make_adapter(base_url: Option<String>, api_key: Option<&str>) -> ElevenLabsAdapter {
        let mut opts = ClientOptions::builder();
        if let Some(u) = base_url {
            opts = opts.base_url(u);
        }
        if let Some(k) = api_key {
            opts = opts.api_key(k);
        }
        let config = ProviderConfig::from_options(PROVIDER_TYPE, opts.build());
        ElevenLabsAdapter::new(config).expect("构造 ElevenLabsAdapter 失败")
    }

    // ============ 基本属性 ============

    #[test]
    fn requires_api_key_is_true() {
        let adapter = make_adapter(None, Some("test-key"));
        assert!(adapter.requires_api_key());
    }

    #[test]
    fn capabilities_contains_speech_and_list_voices() {
        let adapter = make_adapter(None, Some("test-key"));
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::AudioSpeech));
        assert!(caps.contains(&Capabilities::ListVoices));
        assert!(!caps.contains(&Capabilities::Chat));
        assert!(!caps.contains(&Capabilities::ImageGenerate));
        assert!(!caps.contains(&Capabilities::AudioTranscribe));
    }

    #[test]
    fn provider_type_and_name() {
        let adapter = make_adapter(None, Some("test-key"));
        assert_eq!(adapter.provider_type(), "elevenlabs");
        assert_eq!(adapter.provider_name(), "ElevenLabs");
    }

    #[tokio::test]
    async fn start_and_close_are_noops() {
        let mut adapter = make_adapter(None, Some("test-key"));
        assert!(adapter.start().await.is_ok());
        assert!(adapter.close().await.is_ok());
    }

    // ============ list_models ============

    #[tokio::test]
    async fn list_models_returns_elevenlabs_models() {
        let adapter = make_adapter(None, Some("test-key"));
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 5);
        assert_eq!(models[0].id, "eleven_multilingual_v2");
        assert_eq!(models[0].model_type, ModelType::Audio);
        assert_eq!(models[0].provider, "elevenlabs");
    }

    #[tokio::test]
    async fn list_models_filter_by_audio() {
        let adapter = make_adapter(None, Some("test-key"));
        let audio = adapter.list_models(Some(ModelType::Audio)).await.unwrap();
        assert_eq!(audio.len(), 5);
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
    async fn transcribe_returns_unsupported() {
        let adapter = make_adapter(None, Some("test-key"));
        let req = TranscribeRequest::builder("m", FileInput::path("/tmp/a.mp3")).build();
        assert!(matches!(
            adapter.transcribe(req).await.unwrap_err(),
            AibridgeError::UnsupportedCapability { .. }
        ));
    }

    // ============ resolve_voice_id ============

    #[test]
    fn resolve_voice_id_empty_returns_default() {
        assert_eq!(ElevenLabsAdapter::resolve_voice_id(""), DEFAULT_VOICE_ID);
    }

    #[test]
    fn resolve_voice_id_known_alias() {
        assert_eq!(
            ElevenLabsAdapter::resolve_voice_id("Rachel"),
            "21m00Tcm4TlvDq8ikWAM"
        );
        assert_eq!(
            ElevenLabsAdapter::resolve_voice_id("Antoni"),
            "ErXwobaYiN019PkySvjV"
        );
        assert_eq!(
            ElevenLabsAdapter::resolve_voice_id("Adam"),
            "pNInz6obpgDQGcFmaJgB"
        );
    }

    #[test]
    fn resolve_voice_id_case_insensitive() {
        assert_eq!(
            ElevenLabsAdapter::resolve_voice_id("rachel"),
            "21m00Tcm4TlvDq8ikWAM"
        );
        assert_eq!(
            ElevenLabsAdapter::resolve_voice_id("RACHEL"),
            "21m00Tcm4TlvDq8ikWAM"
        );
    }

    #[test]
    fn resolve_voice_id_unknown_passes_through() {
        // 未知别名视为直接传了 voice_id，原样返回
        assert_eq!(
            ElevenLabsAdapter::resolve_voice_id("abc123XYZ"),
            "abc123XYZ"
        );
    }

    // ============ content_type_for_format ============

    #[test]
    fn content_type_for_format_mp3() {
        assert_eq!(
            ElevenLabsAdapter::content_type_for_format("mp3"),
            "audio/mpeg"
        );
        assert_eq!(
            ElevenLabsAdapter::content_type_for_format("MP3"),
            "audio/mpeg"
        );
    }

    #[test]
    fn content_type_for_format_various() {
        assert_eq!(
            ElevenLabsAdapter::content_type_for_format("wav"),
            "audio/wav"
        );
        assert_eq!(
            ElevenLabsAdapter::content_type_for_format("ogg"),
            "audio/ogg"
        );
        assert_eq!(
            ElevenLabsAdapter::content_type_for_format("opus"),
            "audio/opus"
        );
        assert_eq!(
            ElevenLabsAdapter::content_type_for_format("ulaw"),
            "audio/basic"
        );
        assert_eq!(
            ElevenLabsAdapter::content_type_for_format("pcm"),
            "audio/pcm"
        );
    }

    #[test]
    fn content_type_for_format_unknown_defaults_mp3() {
        assert_eq!(
            ElevenLabsAdapter::content_type_for_format("xyz"),
            "audio/mpeg"
        );
    }

    // ============ build_speech_payload ============

    #[test]
    fn build_speech_payload_minimal() {
        let payload = ElevenLabsAdapter::build_speech_payload(
            "hello",
            "eleven_multilingual_v2",
            &HashMap::new(),
        );
        assert_eq!(payload["text"], "hello");
        assert_eq!(payload["model_id"], "eleven_multilingual_v2");
        assert!(payload.get("voice_settings").is_none());
    }

    #[test]
    fn build_speech_payload_with_voice_settings() {
        let mut extra = HashMap::new();
        extra.insert("stability".to_string(), serde_json::json!(0.5));
        extra.insert("similarity_boost".to_string(), serde_json::json!(0.75));
        extra.insert("style".to_string(), serde_json::json!(0.0));
        extra.insert("use_speaker_boost".to_string(), serde_json::json!(true));

        let payload = ElevenLabsAdapter::build_speech_payload("hi", "eleven_turbo_v2", &extra);
        let vs = payload.get("voice_settings").expect("应有 voice_settings");
        assert_eq!(vs["stability"], 0.5);
        assert_eq!(vs["similarity_boost"], 0.75);
        assert_eq!(vs["style"], 0.0);
        assert_eq!(vs["use_speaker_boost"], true);
    }

    #[test]
    fn build_speech_payload_ignores_unrelated_extra() {
        let mut extra = HashMap::new();
        extra.insert("stability".to_string(), serde_json::json!(0.3));
        extra.insert("unrelated_key".to_string(), serde_json::json!("ignored"));

        let payload = ElevenLabsAdapter::build_speech_payload("hi", "m", &extra);
        let vs = payload.get("voice_settings").unwrap();
        assert_eq!(vs["stability"], 0.3);
        assert!(vs.get("unrelated_key").is_none());
    }

    // ============ map_elevenlabs_error ============

    #[test]
    fn map_error_401_is_authentication() {
        let err = ElevenLabsAdapter::map_elevenlabs_error(401, "invalid key");
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[test]
    fn map_error_403_is_authentication() {
        let err = ElevenLabsAdapter::map_elevenlabs_error(403, "forbidden");
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[test]
    fn map_error_429_is_rate_limit() {
        let err = ElevenLabsAdapter::map_elevenlabs_error(429, "slow down");
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[test]
    fn map_error_400_is_validation() {
        let err = ElevenLabsAdapter::map_elevenlabs_error(400, "bad request");
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[test]
    fn map_error_422_is_validation() {
        let err = ElevenLabsAdapter::map_elevenlabs_error(422, "unprocessable");
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[test]
    fn map_error_404_is_voice_not_available() {
        let err = ElevenLabsAdapter::map_elevenlabs_error(404, "voice not found");
        assert!(matches!(err, AibridgeError::VoiceNotAvailable { .. }));
    }

    #[test]
    fn map_error_500_is_api() {
        let err = ElevenLabsAdapter::map_elevenlabs_error(500, "server error");
        assert!(matches!(err, AibridgeError::Api { status: 500, .. }));
    }

    #[test]
    fn map_error_503_is_api() {
        let err = ElevenLabsAdapter::map_elevenlabs_error(503, "unavailable");
        assert!(matches!(err, AibridgeError::Api { status: 503, .. }));
    }

    #[test]
    fn map_error_other_4xx_is_api() {
        let err = ElevenLabsAdapter::map_elevenlabs_error(418, "teapot");
        assert!(matches!(err, AibridgeError::Api { status: 418, .. }));
    }

    // ============ URL 构造 ============

    #[test]
    fn build_tts_url_contains_voice_id() {
        let url = ElevenLabsAdapter::build_tts_url(
            "https://api.elevenlabs.io/v1",
            "21m00Tcm4TlvDq8ikWAM",
        );
        assert_eq!(
            url,
            "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
        );
    }

    #[test]
    fn build_tts_url_strips_trailing_slash() {
        let url = ElevenLabsAdapter::build_tts_url("https://api.elevenlabs.io/v1/", "vid");
        assert_eq!(url, "https://api.elevenlabs.io/v1/text-to-speech/vid");
    }

    #[test]
    fn build_voices_url_is_correct() {
        let url = ElevenLabsAdapter::build_voices_url("https://api.elevenlabs.io/v1");
        assert_eq!(url, "https://api.elevenlabs.io/v1/voices");
    }

    // ============ convert_voice ============

    #[test]
    fn convert_voice_maps_all_fields() {
        let mut labels = HashMap::new();
        labels.insert("gender".to_string(), "female".to_string());
        labels.insert("language".to_string(), "english".to_string());
        labels.insert("accent".to_string(), "american".to_string());
        let raw = ElevenVoiceRaw {
            voice_id: "21m00Tcm4TlvDq8ikWAM".into(),
            name: "Rachel".into(),
            labels,
            category: Some("premade".into()),
        };
        let v = ElevenLabsAdapter::convert_voice(raw);
        assert_eq!(v.voice_id.as_deref(), Some("21m00Tcm4TlvDq8ikWAM"));
        assert_eq!(v.short_name.as_deref(), Some("21m00Tcm4TlvDq8ikWAM"));
        assert_eq!(v.name.as_deref(), Some("Rachel"));
        assert_eq!(v.gender.as_deref(), Some("Female"));
        assert_eq!(v.locale.as_deref(), Some("en"));
        assert_eq!(
            v.extra.get("accent").and_then(|x| x.as_str()),
            Some("american")
        );
        assert_eq!(
            v.extra.get("category").and_then(|x| x.as_str()),
            Some("premade")
        );
    }

    #[test]
    fn convert_voice_without_labels() {
        let raw = ElevenVoiceRaw {
            voice_id: "v1".into(),
            name: "Voice".into(),
            labels: HashMap::new(),
            category: None,
        };
        let v = ElevenLabsAdapter::convert_voice(raw);
        assert_eq!(v.voice_id.as_deref(), Some("v1"));
        assert_eq!(v.name.as_deref(), Some("Voice"));
        assert!(v.gender.is_none());
        assert!(v.locale.is_none());
    }

    #[test]
    fn convert_voice_male_gender() {
        let mut labels = HashMap::new();
        labels.insert("gender".to_string(), "male".to_string());
        labels.insert("language".to_string(), "chinese".to_string());
        let raw = ElevenVoiceRaw {
            voice_id: "v2".into(),
            name: "MaleVoice".into(),
            labels,
            category: None,
        };
        let v = ElevenLabsAdapter::convert_voice(raw);
        assert_eq!(v.gender.as_deref(), Some("Male"));
        assert_eq!(v.locale.as_deref(), Some("zh"));
    }

    // ============ 辅助函数 ============

    #[test]
    fn capitalize_gender_variants() {
        assert_eq!(capitalize_gender("female"), "Female");
        assert_eq!(capitalize_gender("FEMALE"), "Female");
        assert_eq!(capitalize_gender("male"), "Male");
        assert_eq!(capitalize_gender("m"), "Male");
        assert_eq!(capitalize_gender("unknown"), "unknown");
    }

    #[test]
    fn normalize_language_code_known_names() {
        assert_eq!(normalize_language_code("english"), "en");
        assert_eq!(normalize_language_code("chinese"), "zh");
        assert_eq!(normalize_language_code("japanese"), "ja");
        assert_eq!(normalize_language_code("korean"), "ko");
        assert_eq!(normalize_language_code("spanish"), "es");
        assert_eq!(normalize_language_code("french"), "fr");
        assert_eq!(normalize_language_code("german"), "de");
    }

    #[test]
    fn normalize_language_code_codes() {
        assert_eq!(normalize_language_code("en"), "en");
        assert_eq!(normalize_language_code("en-US"), "en");
        assert_eq!(normalize_language_code("zh-CN"), "zh");
        assert_eq!(normalize_language_code("ja-JP"), "ja");
    }

    #[test]
    fn normalize_language_code_unknown_passthrough() {
        assert_eq!(normalize_language_code("klingon"), "klingon");
    }

    #[test]
    fn voice_matches_language_en_matches_english() {
        let v = VoiceInfo::builder().locale("en").build();
        assert!(voice_matches_language("en", &v));
        assert!(voice_matches_language("english", &v));
        assert!(voice_matches_language("EN", &v));
    }

    #[test]
    fn voice_matches_language_zh_matches_chinese() {
        let v = VoiceInfo::builder().locale("zh").build();
        assert!(voice_matches_language("zh", &v));
        assert!(voice_matches_language("chinese", &v));
    }

    #[test]
    fn voice_matches_language_mismatch() {
        let v = VoiceInfo::builder().locale("en").build();
        assert!(!voice_matches_language("zh", &v));
        assert!(!voice_matches_language("ja", &v));
    }

    #[test]
    fn voice_matches_language_no_locale() {
        let v = VoiceInfo::builder().build();
        assert!(!voice_matches_language("en", &v));
    }

    #[test]
    fn lookup_default_voice_known_and_unknown() {
        assert_eq!(lookup_default_voice("Rachel"), Some("21m00Tcm4TlvDq8ikWAM"));
        assert_eq!(lookup_default_voice("Adam"), Some("pNInz6obpgDQGcFmaJgB"));
        assert!(lookup_default_voice("Nonexistent").is_none());
    }

    // ============ speech（mockito） ============

    #[tokio::test]
    async fn speech_returns_binary_audio() {
        let mut server = mockito::Server::new_async().await;
        let audio = b"\x49\x44\x33\x03\x00\x00\x00\x00\x00\x00"; // 伪 mp3 头
        server
            .mock("POST", "/text-to-speech/21m00Tcm4TlvDq8ikWAM")
            .match_header("xi-api-key", "test-key")
            .match_header("accept", "audio/mpeg")
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .with_body(audio)
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = SpeechRequest::builder("eleven_multilingual_v2", "你好世界", "Rachel").build();
        let result = adapter.speech(req).await.unwrap();
        assert_eq!(result.audio_data.as_deref(), Some(audio.as_ref()));
        assert_eq!(result.content_type, "audio/mpeg");
        assert_eq!(result.format, "mp3");
        assert_eq!(result.model.as_deref(), Some("eleven_multilingual_v2"));
    }

    #[tokio::test]
    async fn speech_with_voice_id_directly() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/text-to-speech/CustomVoiceId123")
            .with_status(200)
            .with_body(b"audio-bytes")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        // 直接传 voice_id（未知别名 → 原样）
        let req = SpeechRequest::builder("eleven_turbo_v2", "hi", "CustomVoiceId123").build();
        let result = adapter.speech(req).await.unwrap();
        assert_eq!(result.audio_data.as_deref(), Some(b"audio-bytes".as_ref()));
    }

    #[tokio::test]
    async fn speech_appends_output_format_query_for_non_mp3() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/text-to-speech/21m00Tcm4TlvDq8ikWAM")
            .match_query(mockito::Matcher::UrlEncoded(
                "output_format".into(),
                "wav".into(),
            ))
            .match_header("accept", "audio/wav")
            .with_status(200)
            .with_body(b"wav-bytes")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = SpeechRequest::builder("eleven_multilingual_v2", "hi", "Rachel")
            .response_format("wav")
            .build();
        let result = adapter.speech(req).await.unwrap();
        assert_eq!(result.format, "wav");
        assert_eq!(result.content_type, "audio/wav");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn speech_no_output_format_query_for_mp3() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/text-to-speech/21m00Tcm4TlvDq8ikWAM")
            .match_query(mockito::Matcher::Missing)
            .with_status(200)
            .with_body(b"mp3-bytes")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = SpeechRequest::builder("eleven_multilingual_v2", "hi", "Rachel").build();
        let _ = adapter.speech(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn speech_sends_voice_settings_from_extra() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/text-to-speech/21m00Tcm4TlvDq8ikWAM")
            .match_body(mockito::Matcher::JsonString(
                serde_json::json!({
                    "text": "hi",
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "use_speaker_boost": true
                    }
                })
                .to_string(),
            ))
            .with_status(200)
            .with_body(b"audio")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = SpeechRequest::builder("eleven_multilingual_v2", "hi", "Rachel")
            .extra("stability", 0.5)
            .extra("similarity_boost", 0.75)
            .extra("use_speaker_boost", true)
            .build();
        let _ = adapter.speech(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn speech_uses_default_model_when_empty() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/text-to-speech/21m00Tcm4TlvDq8ikWAM")
            .match_body(mockito::Matcher::JsonString(
                serde_json::json!({
                    "text": "hi",
                    "model_id": DEFAULT_MODEL
                })
                .to_string(),
            ))
            .with_status(200)
            .with_body(b"audio")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = SpeechRequest::builder("", "hi", "Rachel").build();
        let result = adapter.speech(req).await.unwrap();
        assert_eq!(result.model.as_deref(), Some(DEFAULT_MODEL));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn speech_error_401_returns_authentication() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/text-to-speech/21m00Tcm4TlvDq8ikWAM")
            .with_status(401)
            .with_body("{\"detail\":\"invalid api key\"}")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("bad-key"));
        let req = SpeechRequest::builder("eleven_multilingual_v2", "hi", "Rachel").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn speech_error_429_returns_rate_limit() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/text-to-speech/21m00Tcm4TlvDq8ikWAM")
            .with_status(429)
            .with_body("{\"detail\":\"quota exceeded\"}")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = SpeechRequest::builder("eleven_multilingual_v2", "hi", "Rachel").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn speech_error_400_returns_validation() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/text-to-speech/21m00Tcm4TlvDq8ikWAM")
            .with_status(400)
            .with_body("{\"detail\":\"text too long\"}")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = SpeechRequest::builder("eleven_multilingual_v2", "hi", "Rachel").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[tokio::test]
    async fn speech_error_404_returns_voice_not_available() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/text-to-speech/NonexistentVoice")
            .with_status(404)
            .with_body("{\"detail\":\"voice not found\"}")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req =
            SpeechRequest::builder("eleven_multilingual_v2", "hi", "NonexistentVoice").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::VoiceNotAvailable { .. }));
    }

    #[tokio::test]
    async fn speech_error_500_returns_api() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/text-to-speech/21m00Tcm4TlvDq8ikWAM")
            .with_status(500)
            .with_body("internal error")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let req = SpeechRequest::builder("eleven_multilingual_v2", "hi", "Rachel").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Api { status: 500, .. }));
    }

    #[tokio::test]
    async fn speech_without_api_key_returns_validation() {
        let adapter = make_adapter(Some("https://example.com".into()), None);
        let req = SpeechRequest::builder("eleven_multilingual_v2", "hi", "Rachel").build();
        let err = adapter.speech(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    // ============ list_voices（mockito） ============

    #[tokio::test]
    async fn list_voices_fetches_and_parses() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!({
            "voices": [
                {
                    "voice_id": "21m00Tcm4TlvDq8ikWAM",
                    "name": "Rachel",
                    "labels": {"gender": "female", "language": "english", "accent": "american"},
                    "category": "premade"
                },
                {
                    "voice_id": "pNInz6obpgDQGcFmaJgB",
                    "name": "Adam",
                    "labels": {"gender": "male", "language": "english"},
                    "category": "premade"
                }
            ]
        });
        server
            .mock("GET", VOICES_PATH)
            .match_header("xi-api-key", "test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let voices = adapter.list_voices(None).await.unwrap();
        assert_eq!(voices.len(), 2);
        assert_eq!(voices[0].voice_id.as_deref(), Some("21m00Tcm4TlvDq8ikWAM"));
        assert_eq!(voices[0].name.as_deref(), Some("Rachel"));
        assert_eq!(voices[0].gender.as_deref(), Some("Female"));
        assert_eq!(voices[0].locale.as_deref(), Some("en"));
        assert_eq!(voices[1].gender.as_deref(), Some("Male"));
    }

    #[tokio::test]
    async fn list_voices_filters_by_language() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!({
            "voices": [
                {"voice_id": "v1", "name": "EnglishVoice", "labels": {"gender": "female", "language": "english"}},
                {"voice_id": "v2", "name": "ChineseVoice", "labels": {"gender": "male", "language": "chinese"}},
                {"voice_id": "v3", "name": "JapaneseVoice", "labels": {"gender": "female", "language": "japanese"}}
            ]
        });
        server
            .mock("GET", VOICES_PATH)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let en = adapter.list_voices(Some("en")).await.unwrap();
        assert_eq!(en.len(), 1);
        assert_eq!(en[0].voice_id.as_deref(), Some("v1"));
        let zh = adapter.list_voices(Some("zh")).await.unwrap();
        assert_eq!(zh.len(), 1);
        assert_eq!(zh[0].voice_id.as_deref(), Some("v2"));
        let all = adapter.list_voices(None).await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn list_voices_error_401_returns_authentication() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("GET", VOICES_PATH)
            .with_status(401)
            .with_body("invalid key")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("bad-key"));
        let err = adapter.list_voices(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn list_voices_error_429_returns_rate_limit() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("GET", VOICES_PATH)
            .with_status(429)
            .with_body("slow down")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let err = adapter.list_voices(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn list_voices_without_api_key_returns_validation() {
        let adapter = make_adapter(Some("https://example.com".into()), None);
        let err = adapter.list_voices(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    // ============ recommend_voices（默认实现，基于 list_voices） ============

    #[tokio::test]
    async fn recommend_voices_filters_by_gender_and_limit() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!({
            "voices": [
                {"voice_id": "v1", "name": "FemaleEn1", "labels": {"gender": "female", "language": "english"}},
                {"voice_id": "v2", "name": "MaleEn", "labels": {"gender": "male", "language": "english"}},
                {"voice_id": "v3", "name": "FemaleEn2", "labels": {"gender": "female", "language": "english"}}
            ]
        });
        server
            .mock("GET", VOICES_PATH)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        // 语言 + 性别过滤
        let female_en = adapter
            .recommend_voices(Some("en"), Some("Female"), 10)
            .await
            .unwrap();
        assert_eq!(female_en.len(), 2);
        for v in &female_en {
            assert_eq!(v.gender.as_deref(), Some("Female"));
        }
        // limit 截断
        let limited = adapter.recommend_voices(Some("en"), None, 1).await.unwrap();
        assert_eq!(limited.len(), 1);
    }

    #[tokio::test]
    async fn recommend_voices_no_gender_returns_all_up_to_limit() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!({
            "voices": [
                {"voice_id": "v1", "name": "a", "labels": {"gender": "female", "language": "english"}},
                {"voice_id": "v2", "name": "b", "labels": {"gender": "male", "language": "english"}}
            ]
        });
        server
            .mock("GET", VOICES_PATH)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()), Some("test-key"));
        let all = adapter.recommend_voices(None, None, 10).await.unwrap();
        assert_eq!(all.len(), 2);
    }
}
