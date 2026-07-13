//! Edge TTS 适配器（免费神经语音合成，免认证）
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/audio_adapters.py` 的 `EdgeTTSAdapter`。
//!
//! Edge TTS 基于微软 Edge 浏览器的免费神经语音合成服务，底层是 Azure 神经语音引擎，
//! 但**不需要 API Key**，完全免费。支持 100+ 种语音，覆盖 50+ 语言，中文支持优秀。
//!
//! ## 协议
//!
//! Edge TTS 的 `list_voices` 是普通 HTTP GET（返回 JSON 音色列表），
//! 而 `speech` 合成走 WebSocket（SSML 输入，二进制音频输出）。
//! 两者都需要 `Sec-MS-GEC` token（基于固定 `TrustedClientToken` + 当前时间戳的 HMAC-SHA256）。
//!
//! - list_voices: `GET https://speech.platform.bing.com/.../voices/list?TrustedClientToken=...&Sec-MS-GEC=...`
//! - speech: `wss://speech.platform.bing.com/.../edge/v1?TrustedClientToken=...&Sec-MS-GEC=...&ConnectionId=...`
//!
//! ## 特性（v1.3.3 保留）
//!
//! - `requires_api_key = false`（免认证）
//! - 音色自动降级：`voice` 传候选列表时，某音色失败（VoiceNotAvailable / ServiceUnavailable）
//!   自动切换下一个，直到成功或全部失败
//! - 空音频检测：合成返回空音频时查 `list_voices` 区分语义——voice 在线则判服务端临时不可用
//!   （可重试），voice 不在线则判已下线（应换音色）
//! - 音色列表缓存（避免每次空音频都网络查询）

use std::collections::HashMap;
use std::future::Future;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use hmac::{Hmac, Mac};
use rand::Rng;
use serde::Deserialize;
use serde_json::Value;
use sha2::Sha256;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};
use tokio_tungstenite::tungstenite::Message;

use crate::adapter::{Adapter, Capabilities, CapabilitySet};
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::http::HttpClient;
use crate::model::audio::{SpeechRequest, SpeechResult};
use crate::model::common::{ModelInfo, ModelType, VoiceInfo};

// ==================== 常量 ====================

/// Provider 类型标识
const PROVIDER_TYPE: &str = "edge-tts";
/// Provider 显示名称
const PROVIDER_NAME: &str = "Edge TTS";
/// Edge TTS 固定的可信客户端 token（社区逆向所得，所有客户端共用）
const TRUSTED_CLIENT_TOKEN: &str = "6A5AA1D4EAFF4E9FB37E23D68491D6F4";
/// Sec-MS-GEC 版本号（对应 Edge 浏览器版本）
const SEC_MS_GEC_VERSION: &str = "1-130.0.2849.68";
/// 默认音色（中文女声晓晓）
const DEFAULT_VOICE: &str = "zh-CN-XiaoxiaoNeural";
/// 默认 API 基地址
const DEFAULT_API_BASE: &str = "https://speech.platform.bing.com";
/// 音色列表端点路径
const VOICES_PATH: &str = "/consumer/speech/synthesize/readaloud/voices/list";
/// WebSocket 合成端点路径
const WS_PATH: &str = "/consumer/speech/synthesize/readaloud/edge/v1";
/// WS 握手所需 Origin header
const WS_ORIGIN: &str = "https://speech.platform.bing.com";
/// WS 握手 User-Agent
const WS_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)";
/// Windows epoch 偏移（1601-01-01 到 1970-01-01 的 100ns 单位）
const WIN_EPOCH: i64 = 116_444_736_000_000_000;
/// 5 分钟对应的 100ns 单位（GEC token 向下取整到 5 分钟边界）
const GEC_WINDOW_TICKS: i64 = 3_000_000_000;

/// HMAC-SHA256 类型别名（计算 Sec-MS-GEC token）
type HmacSha256 = Hmac<Sha256>;

// ==================== 音色别名查询 ====================

/// 查询常见音色别名（简称/中文名）→ 完整 voice ID
///
/// 对应 Python v1 `EdgeTTSAdapter.COMMON_VOICES`。返回 None 表示非已知别名。
fn lookup_common_voice(voice: &str) -> Option<&'static str> {
    let v = match voice {
        // 中文女声
        "xiaoxiao" | "晓晓" => "zh-CN-XiaoxiaoNeural",
        "xiaoyi" | "晓伊" => "zh-CN-XiaoyiNeural",
        "xiaochen" | "晓辰" => "zh-CN-XiaochenNeural",
        "xiaohan" | "晓涵" => "zh-CN-XiaohanNeural",
        "xiaomeng" | "晓梦" => "zh-CN-XiaomengNeural",
        "xiaomo" | "晓墨" => "zh-CN-XiaomoNeural",
        "xiaoqiu" | "晓秋" => "zh-CN-XiaoqiuNeural",
        "xiaorui" | "晓睿" => "zh-CN-XiaoruiNeural",
        "xiaoshuang" | "晓双" => "zh-CN-XiaoshuangNeural",
        "xiaoxuan" | "晓萱" => "zh-CN-XiaoxuanNeural",
        "xiaoyan" | "晓颜" => "zh-CN-XiaoyanNeural",
        "xiaoyou" | "晓悠" => "zh-CN-XiaoyouNeural",
        // 中文男声
        "yunjian" | "云健" => "zh-CN-YunjianNeural",
        "yunxi" | "云希" => "zh-CN-YunxiNeural",
        "yunxia" | "云夏" => "zh-CN-YunxiaNeural",
        "yunyang" | "云扬" => "zh-CN-YunyangNeural",
        "yunque" | "云泽" => "zh-CN-YunzeNeural",
        // 英文女声
        "jenny" => "en-US-JennyNeural",
        "jenny-multilingual" => "en-US-JennyMultilingualNeural",
        "aria" => "en-US-AriaNeural",
        // 英文男声
        "guy" => "en-US-GuyNeural",
        "roger" => "en-US-RogerNeural",
        "davis" => "en-US-DavisNeural",
        "tony" => "en-US-TonyNeural",
        "jason" => "en-US-JasonNeural",
        // 日文
        "nanami" => "ja-JP-NanamiNeural",
        "keita" => "ja-JP-KeitaNeural",
        // 韩文
        "sun-hi" => "ko-KR-SunHiNeural",
        "in-jun" => "ko-KR-InJoonNeural",
        // 法文
        "denise" => "fr-FR-DeniseNeural",
        "henri" => "fr-FR-HenriNeural",
        // 德文
        "katja" => "de-DE-KatjaNeural",
        "conrad" => "de-DE-ConradNeural",
        // 西班牙文
        "elvira" => "es-ES-ElviraNeural",
        "alvaro" => "es-ES-AlvaroNeural",
        _ => return None,
    };
    Some(v)
}

// ==================== Edge 原始音色反序列化结构 ====================

/// Edge TTS list_voices 返回的单个音色项（原始 JSON 结构）
///
/// 字段名与 Edge 服务端返回的 JSON 一致（PascalCase）。
#[derive(Debug, Deserialize)]
struct EdgeVoiceRaw {
    #[serde(rename = "ShortName")]
    short_name: String,
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Locale")]
    locale: String,
    #[serde(rename = "Gender")]
    gender: String,
    #[serde(rename = "FriendlyName", default)]
    friendly_name: Option<String>,
}

// ==================== EdgeTtsAdapter ====================

/// Edge TTS 适配器
///
/// 持有 HTTP 客户端（list_voices 用）与音色列表缓存。
/// speech 合成按需建立 WebSocket 连接（per-request，不持有长连接）。
pub struct EdgeTtsAdapter {
    /// Provider 配置
    #[allow(dead_code)]
    config: ProviderConfig,
    /// HTTP 客户端（list_voices 的 HTTP GET）
    http: HttpClient,
    /// 音色列表缓存（避免重复网络拉取）
    voices_cache: Mutex<Option<Vec<VoiceInfo>>>,
    /// API 基地址（默认 `https://speech.platform.bing.com`，可由 config.base_url 覆盖）
    api_base: String,
}

impl EdgeTtsAdapter {
    /// 创建 Edge TTS 适配器
    ///
    /// `config.base_url` 可覆盖 API 基地址（主要用于测试指向 mock server），
    /// 为空时用 `DEFAULT_API_BASE`。Edge TTS 免认证，不强制 api_key。
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let api_base = config
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_API_BASE.to_string());
        let opts = ClientOptions::builder().timeout(config.timeout).build();
        let http = HttpClient::new(&opts)?;
        Ok(Self {
            config,
            http,
            voices_cache: Mutex::new(None),
            api_base,
        })
    }

    // ==================== 纯函数（协议构造/解析，可单测） ====================

    /// 解析音色名：别名/中文名/完整 ID → 完整 voice ID
    ///
    /// 对应 Python v1 `EdgeTTSAdapter._resolve_voice`。
    /// - 空值 → 默认音色
    /// - 已知别名（精确或大小写不敏感）→ 对应完整 ID
    /// - 含 ≥2 个 `-`（如 `zh-CN-XiaoxiaoNeural`）→ 视为完整 ID 原样返回
    /// - 其余 → 默认音色
    fn resolve_voice(voice: &str) -> String {
        if voice.is_empty() {
            return DEFAULT_VOICE.to_string();
        }
        if let Some(v) = lookup_common_voice(voice) {
            return v.to_string();
        }
        // 大小写不敏感匹配（中文别名 lowercase 无影响，英文如 Xiaoxiao → xiaoxiao）
        if let Some(v) = lookup_common_voice(&voice.to_lowercase()) {
            return v.to_string();
        }
        // 完整 voice ID：含至少 2 个 `-`（语言-区域-名称）
        if voice.contains('-') && voice.matches('-').count() >= 2 {
            return voice.to_string();
        }
        DEFAULT_VOICE.to_string()
    }

    /// 获取输出格式：返回 (edge-tts format string, content_type, 扩展名)
    ///
    /// 对应 Python v1 `EdgeTTSAdapter._get_output_format`。
    fn get_output_format(fmt: Option<&str>) -> (&'static str, &'static str, &'static str) {
        let key = fmt.unwrap_or("mp3").to_lowercase();
        let edge_fmt: &str = match key.as_str() {
            "mp3" => "audio-24khz-48kbit-mp3-mono",
            "mp3-96k" => "audio-24khz-96kbit-mp3-mono",
            "mp3-128k" => "audio-24khz-128kbit-mp3-mono",
            "mp3-160k" => "audio-24khz-160kbit-mp3-mono",
            "webm" => "audio-24khz-48kbit-opus-mono",
            "webm-24khz-16bit-mono-opus" => "audio-24khz-16bit-mono-opus",
            "ogg" => "audio-24khz-48kbit-opus-mono",
            "wav" | "pcm" => "audio-24khz-16bit-mono-pcm",
            _ => "audio-24khz-48kbit-mp3-mono",
        };
        let (content_type, ext) = if edge_fmt.contains("mp3") {
            ("audio/mpeg", "mp3")
        } else if edge_fmt.contains("opus") || edge_fmt.contains("webm") || edge_fmt.contains("ogg")
        {
            ("audio/ogg", "ogg")
        } else if edge_fmt.contains("pcm") || edge_fmt.contains("wav") {
            ("audio/wav", "wav")
        } else {
            ("audio/mpeg", "mp3")
        };
        (edge_fmt, content_type, ext)
    }

    /// 计算 Sec-MS-GEC token（HMAC-SHA256）
    ///
    /// 算法（社区逆向）：把当前 Unix 秒转为 Windows ticks（100ns 单位），
    /// 向下取整到 5 分钟边界，用 `TRUSTED_CLIENT_TOKEN` 作 key 做 HMAC-SHA256，
    /// 输出大写十六进制。服务端用此 token 鉴权（无需 API Key）。
    fn compute_sec_ms_gec(unix_secs: i64) -> String {
        let mut ticks = unix_secs * 10_000_000 + WIN_EPOCH;
        ticks -= ticks.rem_euclid(GEC_WINDOW_TICKS);
        let mut mac = HmacSha256::new_from_slice(TRUSTED_CLIENT_TOKEN.as_bytes())
            .expect("HMAC 接受任意长度 key");
        mac.update(&ticks.to_be_bytes());
        let result = mac.finalize().into_bytes();
        let mut hex = String::with_capacity(64);
        for b in result.iter() {
            hex.push_str(&format!("{:02X}", b));
        }
        hex
    }

    /// 构造 WebSocket 合成端点 URL
    ///
    /// `api_base` 的 `https://` 替换为 `wss://`（`http://` → `ws://`）。
    fn build_ws_url(gec: &str, connection_id: &str, api_base: &str) -> String {
        let ws_base = api_base
            .replacen("https://", "wss://", 1)
            .replacen("http://", "ws://", 1);
        format!(
            "{base}{path}?TrustedClientToken={token}&Sec-MS-GEC={gec}&Sec-MS-GEC-Version={ver}&ConnectionId={cid}",
            base = ws_base.trim_end_matches('/'),
            path = WS_PATH,
            token = TRUSTED_CLIENT_TOKEN,
            gec = gec,
            ver = SEC_MS_GEC_VERSION,
            cid = connection_id,
        )
    }

    /// 构造 list_voices HTTP 端点 URL
    fn build_voices_url(gec: &str, api_base: &str) -> String {
        format!(
            "{base}{path}?TrustedClientToken={token}&Sec-MS-GEC={gec}&Sec-MS-GEC-Version={ver}",
            base = api_base.trim_end_matches('/'),
            path = VOICES_PATH,
            token = TRUSTED_CLIENT_TOKEN,
            gec = gec,
            ver = SEC_MS_GEC_VERSION,
        )
    }

    /// 构造 WebSocket 握手请求（设置 Origin / User-Agent header）
    ///
    /// Edge TTS 服务端要求 Origin header，否则拒绝握手。
    fn build_ws_request(url: &str) -> Result<tokio_tungstenite::tungstenite::http::Request<()>> {
        let mut request = url
            .into_client_request()
            .map_err(|e| AibridgeError::validation(format!("Edge TTS WebSocket URL 无效: {e}")))?;
        let headers = request.headers_mut();
        headers.insert(
            HeaderName::from_static("origin"),
            HeaderValue::from_static(WS_ORIGIN),
        );
        headers.insert(
            HeaderName::from_static("user-agent"),
            HeaderValue::from_static(WS_USER_AGENT),
        );
        Ok(request)
    }

    /// 从 voice ID 提取 locale（如 `zh-CN-XiaoxiaoNeural` → `zh-CN`）
    fn locale_from_voice(voice: &str) -> String {
        let parts: Vec<&str> = voice.splitn(3, '-').collect();
        if parts.len() >= 2 {
            format!("{}-{}", parts[0], parts[1])
        } else {
            "en-US".to_string()
        }
    }

    /// 构造 SSML（语音合成标记语言）
    ///
    /// 对应 Python v1 构造的 `<speak><voice><prosody>...` 结构。
    fn build_ssml(text: &str, voice: &str, rate: &str, pitch: &str, volume: &str) -> String {
        let locale = Self::locale_from_voice(voice);
        let escaped = escape_xml(text);
        format!(
            "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{locale}'>\
             <voice name='{voice}'>\
             <prosody pitch='{pitch}' rate='{rate}' volume='{volume}'>\
             {escaped}\
             </prosody></voice></speak>"
        )
    }

    /// 构造 WS 配置消息（speech.config，声明输出格式）
    fn build_config_message(output_format: &str, timestamp: &str) -> String {
        let body = serde_json::json!({
            "context": {
                "synthesis": {
                    "audio": {
                        "metadataoptions": {
                            "sentenceBoundaryEnabled": "false",
                            "wordBoundaryEnabled": "false"
                        },
                        "outputFormat": output_format
                    }
                }
            }
        });
        format!(
            "X-Timestamp:{ts}\r\n\
             Content-Type:application/json; charset=utf-8\r\n\
             Path:speech.config\r\n\r\n\
             {body}",
            ts = timestamp,
            body = body,
        )
    }

    /// 构造 WS 合成消息（ssml，携带待合成文本）
    fn build_synth_message(ssml: &str, request_id: &str, timestamp: &str) -> String {
        format!(
            "X-RequestId:{rid}\r\n\
             Content-Type:application/ssml+xml\r\n\
             X-Timestamp:{ts}\r\n\
             Path:ssml\r\n\r\n\
             {ssml}",
            rid = request_id,
            ts = timestamp,
            ssml = ssml,
        )
    }

    /// 解析 WS 二进制消息，提取音频数据
    ///
    /// Edge TTS binary 消息格式：2 字节大端 header 长度 + header 文本 + audio 数据。
    /// 仅当 header 含 `Path:audio` 时返回音频切片，其余返回 None。
    fn parse_audio_binary(msg: &[u8]) -> Option<&[u8]> {
        if msg.len() < 2 {
            return None;
        }
        let header_len = u16::from_be_bytes([msg[0], msg[1]]) as usize;
        if header_len == 0 || msg.len() < 2 + header_len {
            return None;
        }
        let header = &msg[2..2 + header_len];
        let header_str = std::str::from_utf8(header).ok()?;
        if header_str.contains("Path:audio") {
            Some(&msg[2 + header_len..])
        } else {
            None
        }
    }

    /// 解析 WS 文本消息，提取 `Path:` 字段值（如 `turn.end` / `response`）
    fn parse_ws_text_path(msg: &str) -> Option<&str> {
        for line in msg.lines() {
            if let Some(rest) = line.strip_prefix("Path:") {
                return Some(rest.trim());
            }
        }
        None
    }

    /// 从 WS 文本消息中提取错误（`Path:response` 且 body `"Type":"error"`）
    ///
    /// 音色相关错误（Code/Message 含 "voice"）→ VoiceNotAvailable；其余 → Api。
    fn extract_ws_error(msg: &str) -> Option<AibridgeError> {
        let body = msg
            .split("\r\n\r\n")
            .nth(1)
            .or_else(|| msg.split("\n\n").nth(1))?;
        let v: Value = serde_json::from_str(body.trim()).ok()?;
        if v.get("Type").and_then(|t| t.as_str()) != Some("error") {
            return None;
        }
        let code = v.get("Code").and_then(|c| c.as_str()).unwrap_or("UNKNOWN");
        let message = v
            .get("Message")
            .and_then(|m| m.as_str())
            .unwrap_or("Edge TTS 合成错误");
        let lower = format!("{code} {message}").to_lowercase();
        if lower.contains("voice") {
            Some(AibridgeError::voice_not_available(format!(
                "{code}: {message}"
            )))
        } else {
            Some(AibridgeError::api(0, format!("{code}: {message}")))
        }
    }

    /// 空音频错误分类（纯函数，便于单测）
    ///
    /// voice 在线但返空 → ServiceUnavailable（可重试）；
    /// voice 不在线 → VoiceNotAvailable（应换音色）。
    fn empty_audio_error(voice_id: &str, voice_available: bool) -> AibridgeError {
        if voice_available {
            AibridgeError::service_unavailable(format!(
                "Edge TTS 服务端返回空音频（voice={} 仍在线），可能是限流或网络抖动，可重试",
                voice_id
            ))
        } else {
            AibridgeError::voice_not_available(format!(
                "Edge TTS 音色 {} 已下线或不存在，请更换音色",
                voice_id
            ))
        }
    }

    /// 构造语速字符串（edge-tts 格式 `+0%` / `-50%` / `+100%`）
    ///
    /// 优先用 `extra.rate` 字符串（与 Python 兼容），否则从 `speed` 数值换算
    /// （speed=1.0 → `+0%`，speed=0.5 → `-50%`，speed=2.0 → `+100%`）。
    fn build_rate(speed: Option<f64>, extra: &HashMap<String, Value>) -> String {
        if let Some(r) = extra.get("rate").and_then(|v| v.as_str()) {
            return r.to_string();
        }
        let speed = speed.unwrap_or(1.0);
        let pct = ((speed - 1.0) * 100.0).round() as i32;
        format!("{pct:+}%")
    }

    /// 构造音调字符串（edge-tts 格式 `+0Hz` / `+100Hz` / `-50Hz`）
    ///
    /// 优先用 `extra.pitch` 字符串，否则从 `pitch` 数值换算（pitch∈[-1,1] → ±100Hz）。
    fn build_pitch(pitch: Option<f64>, extra: &HashMap<String, Value>) -> String {
        if let Some(p) = extra.get("pitch").and_then(|v| v.as_str()) {
            return p.to_string();
        }
        let pitch = pitch.unwrap_or(0.0);
        let hz = (pitch * 100.0).round() as i32;
        format!("{hz:+}Hz")
    }

    /// 构造音量字符串（edge-tts 格式 `+0%` / `-50%` / `+100%`）
    ///
    /// 优先用 `extra.volume` 字符串，否则从 `volume` 数值换算
    /// （volume=1.0 → `+0%`，volume=0.5 → `-50%`，volume=2.0 → `+100%`）。
    fn build_volume(volume: Option<f64>, extra: &HashMap<String, Value>) -> String {
        if let Some(v) = extra.get("volume").and_then(|v| v.as_str()) {
            return v.to_string();
        }
        let volume = volume.unwrap_or(1.0);
        let pct = ((volume - 1.0) * 100.0).round() as i32;
        format!("{pct:+}%")
    }

    /// 把 Unix 秒格式化为 ISO 8601 UTC 时间戳（`2023-11-14T22:13:20.000Z`）
    ///
    /// 用 Howard Hinnant 的 civil_from_days 算法，避免引入 chrono 依赖。
    fn utc_timestamp_iso(unix_secs: i64) -> String {
        let days = unix_secs.div_euclid(86400);
        let secs_of_day = unix_secs.rem_euclid(86400);
        let hour = secs_of_day / 3600;
        let min = (secs_of_day % 3600) / 60;
        let sec = secs_of_day % 60;

        // civil_from_days（Howard Hinnant）
        let z = days + 719468;
        let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
        let doe = z - era * 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = if m <= 2 { y + 1 } else { y };

        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.000Z",
            year, m, d, hour, min, sec
        )
    }

    /// 判断错误是否可触发音色降级（VoiceNotAvailable / ServiceUnavailable）
    fn is_fallback_error(e: &AibridgeError) -> bool {
        matches!(
            e,
            AibridgeError::VoiceNotAvailable { .. } | AibridgeError::ServiceUnavailable { .. }
        )
    }

    /// 把 Edge 原始音色转为统一 `VoiceInfo`
    fn convert_voice(raw: EdgeVoiceRaw) -> VoiceInfo {
        let display_name = raw.friendly_name.unwrap_or(raw.name);
        VoiceInfo::builder()
            .short_name(raw.short_name.clone())
            .name(display_name)
            .locale(raw.locale)
            .gender(raw.gender)
            .voice_id(raw.short_name)
            .build()
    }

    // ==================== 音色降级（可单测） ====================

    /// 音色降级合成：按候选列表逐个尝试，失败（VoiceNotAvailable / ServiceUnavailable）
    /// 切换下一个，直到成功或全部失败。其余错误不降级直接抛出。
    ///
    /// 抽成独立函数便于单测降级逻辑（`single` 闭包模拟单音色合成结果）。
    async fn speech_with_fallback<F, Fut>(
        candidates: Vec<String>,
        mut single: F,
    ) -> Result<SpeechResult>
    where
        F: FnMut(&str) -> Fut + Send,
        Fut: Future<Output = Result<SpeechResult>> + Send,
    {
        let total = candidates.len();
        let mut last_error: Option<AibridgeError> = None;
        for (idx, voice_raw) in candidates.iter().enumerate() {
            match single(voice_raw).await {
                Ok(result) => return Ok(result),
                Err(e) if Self::is_fallback_error(&e) => {
                    // 单 voice 模式：直接抛当前异常，无需存 last_error
                    if total <= 1 {
                        return Err(e);
                    }
                    // 多 voice 列表模式：记录最后一个失败异常，尝试下一个
                    last_error = Some(e);
                    tracing::warn!(
                        "Edge TTS voice {} 合成失败，尝试下一个候选 ({}/{})",
                        voice_raw,
                        idx + 1,
                        total
                    );
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_error
            .unwrap_or_else(|| AibridgeError::api(0, "Edge TTS 语音合成失败，无可用 voice")))
    }

    // ==================== 网络方法 ====================

    /// 单音色合成（WebSocket 协议）
    ///
    /// 建连 → 发配置消息 → 发 SSML 合成消息 → 收集 audio binary → 检测空音频。
    /// 不含降级逻辑（由 `speech` 包装降级）。
    #[allow(clippy::too_many_arguments)] // 合成参数均为必需，无法再精简
    async fn speech_single(
        &self,
        model: &str,
        input: &str,
        voice_id: &str,
        output_format: &str,
        rate: &str,
        pitch: &str,
        volume: &str,
    ) -> Result<SpeechResult> {
        let (edge_fmt, content_type, ext) = Self::get_output_format(Some(output_format));

        let unix_secs = current_unix_secs();
        let gec = Self::compute_sec_ms_gec(unix_secs);
        let conn_id = generate_connection_id();
        let url = Self::build_ws_url(&gec, &conn_id, &self.api_base);
        let request = Self::build_ws_request(&url)?;

        let (ws_stream, _) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(map_ws_error)?;
        let mut ws = ws_stream;

        let ts = Self::utc_timestamp_iso(unix_secs);
        // 发送配置消息（声明输出格式）
        let config_msg = Self::build_config_message(edge_fmt, &ts);
        ws.send(Message::Text(config_msg))
            .await
            .map_err(map_ws_error)?;

        // 发送合成消息（SSML）
        let request_id = generate_connection_id();
        let ssml = Self::build_ssml(input, voice_id, rate, pitch, volume);
        let synth_msg = Self::build_synth_message(&ssml, &request_id, &ts);
        ws.send(Message::Text(synth_msg))
            .await
            .map_err(map_ws_error)?;

        // 接收消息，收集音频二进制
        let mut audio_chunks: Vec<Vec<u8>> = Vec::new();
        while let Some(msg_result) = ws.next().await {
            let msg = msg_result.map_err(map_ws_error)?;
            match msg {
                Message::Binary(data) => {
                    if let Some(audio) = Self::parse_audio_binary(&data) {
                        audio_chunks.push(audio.to_vec());
                    }
                }
                Message::Text(text) => {
                    if let Some(path) = Self::parse_ws_text_path(&text) {
                        match path {
                            "turn.end" => break,
                            "response" => {
                                if let Some(err) = Self::extract_ws_error(&text) {
                                    return Err(err);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }

        let audio_data: Vec<u8> = audio_chunks.into_iter().flatten().collect();

        // 空音频检测：区分 voice 在线/下线
        if audio_data.is_empty() {
            let available = self.check_voice_available(voice_id).await;
            return Err(Self::empty_audio_error(voice_id, available));
        }

        Ok(SpeechResult {
            audio_data: Some(audio_data),
            audio_url: None,
            audio_base64: None,
            content_type: content_type.to_string(),
            format: ext.to_string(),
            duration: None,
            model: Some(if model.is_empty() {
                PROVIDER_TYPE.to_string()
            } else {
                model.to_string()
            }),
        })
    }

    /// 拉取全量音色列表（带缓存）
    ///
    /// 缓存未命中时 HTTP GET 服务端，缓存后直接返回（language 过滤在 `list_voices` 做）。
    async fn list_voices_raw(&self) -> Result<Vec<VoiceInfo>> {
        // 先检查缓存
        {
            let cache = self.voices_cache.lock().await;
            if let Some(ref voices) = *cache {
                return Ok(voices.clone());
            }
        }
        let unix_secs = current_unix_secs();
        let gec = Self::compute_sec_ms_gec(unix_secs);
        let url = Self::build_voices_url(&gec, &self.api_base);

        let resp = self.http.inner().get(&url).send().await.map_err(|e| {
            if e.is_timeout() {
                AibridgeError::Timeout
            } else {
                AibridgeError::Network(e)
            }
        })?;
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(map_edge_http_error(status_code, &body));
        }
        let raw_voices: Vec<EdgeVoiceRaw> = resp.json().await.map_err(AibridgeError::from)?;
        let voices: Vec<VoiceInfo> = raw_voices.into_iter().map(Self::convert_voice).collect();

        // 写缓存
        let mut cache = self.voices_cache.lock().await;
        *cache = Some(voices.clone());
        Ok(voices)
    }

    /// 检查指定 voice 是否在可用音色列表中（空音频时区分异常语义）
    ///
    /// list_voices 查询本身失败时保守返回 true（按服务端临时问题处理）。
    async fn check_voice_available(&self, voice_id: &str) -> bool {
        match self.list_voices(None).await {
            Ok(voices) => voices
                .iter()
                .any(|v| v.short_name.as_deref() == Some(voice_id)),
            Err(_) => true,
        }
    }
}

#[async_trait]
impl Adapter for EdgeTtsAdapter {
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

    /// 免认证：Edge TTS 无需 API Key
    fn requires_api_key(&self) -> bool {
        false
    }

    async fn start(&mut self) -> Result<()> {
        // 无需惰性初始化（依赖编译期链接），保持 no-op
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        // WS 连接是 per-request 的，无需释放长连接资源
        Ok(())
    }

    /// 文字转语音（含音色自动降级）
    async fn speech(&self, req: SpeechRequest) -> Result<SpeechResult> {
        // 规范化为候选列表：空列表兜底为 [""]（走默认音色）
        let candidates: Vec<String> = if req.voice.voices.is_empty() {
            vec![String::new()]
        } else {
            req.voice.voices.clone()
        };
        let model = req.model.clone();
        let input = req.input.clone();
        let fmt = req.response_format.clone();
        let rate = Self::build_rate(req.speed, &req.extra);
        let pitch = Self::build_pitch(req.pitch, &req.extra);
        let volume = Self::build_volume(req.volume, &req.extra);

        let single = move |voice_raw: &str| {
            let voice_id = Self::resolve_voice(voice_raw);
            let model = model.clone();
            let input = input.clone();
            let fmt = fmt.clone();
            let rate = rate.clone();
            let pitch = pitch.clone();
            let volume = volume.clone();
            async move {
                self.speech_single(&model, &input, &voice_id, &fmt, &rate, &pitch, &volume)
                    .await
            }
        };
        Self::speech_with_fallback(candidates, single).await
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

    /// 列出 Edge TTS 模型（无标准 /models 端点，保留硬编码列表）
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let models = vec![ModelInfo {
            id: PROVIDER_TYPE.into(),
            name: "Edge TTS Neural".into(),
            model_type: ModelType::Audio,
            provider: PROVIDER_TYPE.into(),
            capabilities: vec!["audio_speech".into()],
            max_tokens: None,
            supports_streaming: false,
            description: Some(
                "微软 Edge 浏览器免费神经语音合成，支持 100+ 种语音，50+ 语言，中文支持优秀".into(),
            ),
            created: None,
        }];
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }
}

// ==================== 错误映射 ====================

/// 将 WebSocket 错误映射为 AibridgeError
///
/// 注意：`AibridgeError::Network` 仅接受 `reqwest::Error`，WS 错误无法转换，
/// 故 WS 网络错误映射到 `ServiceUnavailable`（语义：服务暂时不可达，可重试，
/// 与 Network 的 is_retryable 一致）；WS 握手 HTTP 错误映射到 `Api`。
fn map_ws_error(e: tokio_tungstenite::tungstenite::Error) -> AibridgeError {
    use tokio_tungstenite::tungstenite::Error as WsErr;
    match e {
        WsErr::Http(resp) => {
            let status = resp.status().as_u16();
            AibridgeError::api(
                status,
                format!("Edge TTS WebSocket 握手失败 (HTTP {status})"),
            )
        }
        _ => AibridgeError::service_unavailable(format!("Edge TTS WebSocket 网络错误: {e}")),
    }
}

/// 将 Edge TTS list_voices HTTP 错误映射为 AibridgeError
fn map_edge_http_error(status: u16, body: &str) -> AibridgeError {
    match status {
        401 | 403 => AibridgeError::authentication(format!("Edge TTS 认证失败: {body}")),
        429 => AibridgeError::rate_limit(format!("Edge TTS 限流: {body}")),
        404 => AibridgeError::voice_not_available(format!("Edge TTS 音色端点不存在: {body}")),
        s if s >= 500 => {
            AibridgeError::service_unavailable(format!("Edge TTS 服务不可用 ({s}): {body}"))
        }
        s => AibridgeError::api(s, format!("Edge TTS HTTP {s}: {body}")),
    }
}

// ==================== 辅助函数 ====================

/// XML 特殊字符转义（SSML 文本内容用）
fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            _ => out.push(c),
        }
    }
    out
}

/// 当前 Unix 时间戳（秒）
fn current_unix_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// 生成 32 字符十六进制连接 ID（模拟无连字符 UUID，Edge TTS 要求 hex 格式）
fn generate_connection_id() -> String {
    let id: u128 = rand::thread_rng().gen();
    format!("{:032x}", id)
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
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    /// 构造测试用适配器（base_url 指向 mockito server，None 用默认基地址）
    fn make_adapter(base_url: Option<String>) -> EdgeTtsAdapter {
        let mut opts = ClientOptions::builder();
        if let Some(u) = base_url {
            opts = opts.base_url(u);
        }
        let config = ProviderConfig::from_options(PROVIDER_TYPE, opts.build());
        EdgeTtsAdapter::new(config).expect("构造 EdgeTtsAdapter 失败")
    }

    // ============ 基本属性 ============

    #[test]
    fn requires_api_key_is_false() {
        let adapter = make_adapter(None);
        assert!(!adapter.requires_api_key());
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
        assert_eq!(adapter.provider_type(), "edge-tts");
        assert_eq!(adapter.provider_name(), "Edge TTS");
    }

    #[tokio::test]
    async fn list_models_returns_single_edge_tts() {
        let adapter = make_adapter(None);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "edge-tts");
        assert_eq!(models[0].model_type, ModelType::Audio);
        assert_eq!(models[0].provider, "edge-tts");
    }

    #[tokio::test]
    async fn list_models_filter_by_type() {
        let adapter = make_adapter(None);
        let audio = adapter.list_models(Some(ModelType::Audio)).await.unwrap();
        assert_eq!(audio.len(), 1);
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
    async fn transcribe_returns_unsupported() {
        let adapter = make_adapter(None);
        let req = TranscribeRequest::builder("m", FileInput::path("/tmp/a.mp3")).build();
        assert!(matches!(
            adapter.transcribe(req).await.unwrap_err(),
            AibridgeError::UnsupportedCapability { .. }
        ));
    }

    // ============ resolve_voice ============

    #[test]
    fn resolve_voice_empty_returns_default() {
        assert_eq!(EdgeTtsAdapter::resolve_voice(""), DEFAULT_VOICE);
    }

    #[test]
    fn resolve_voice_alias_lookup() {
        assert_eq!(
            EdgeTtsAdapter::resolve_voice("xiaoxiao"),
            "zh-CN-XiaoxiaoNeural"
        );
        assert_eq!(
            EdgeTtsAdapter::resolve_voice("晓晓"),
            "zh-CN-XiaoxiaoNeural"
        );
        assert_eq!(EdgeTtsAdapter::resolve_voice("yunxi"), "zh-CN-YunxiNeural");
        assert_eq!(EdgeTtsAdapter::resolve_voice("jenny"), "en-US-JennyNeural");
        assert_eq!(
            EdgeTtsAdapter::resolve_voice("nanami"),
            "ja-JP-NanamiNeural"
        );
    }

    #[test]
    fn resolve_voice_case_insensitive() {
        assert_eq!(
            EdgeTtsAdapter::resolve_voice("Xiaoxiao"),
            "zh-CN-XiaoxiaoNeural"
        );
        assert_eq!(EdgeTtsAdapter::resolve_voice("JENNY"), "en-US-JennyNeural");
    }

    #[test]
    fn resolve_voice_passes_through_full_id() {
        assert_eq!(
            EdgeTtsAdapter::resolve_voice("zh-CN-XiaoxiaoNeural"),
            "zh-CN-XiaoxiaoNeural"
        );
        assert_eq!(
            EdgeTtsAdapter::resolve_voice("en-US-JennyMultilingualNeural"),
            "en-US-JennyMultilingualNeural"
        );
    }

    #[test]
    fn resolve_voice_unknown_short_returns_default() {
        assert_eq!(EdgeTtsAdapter::resolve_voice("unknown"), DEFAULT_VOICE);
    }

    // ============ get_output_format ============

    #[test]
    fn get_output_format_default_mp3() {
        let (fmt, ct, ext) = EdgeTtsAdapter::get_output_format(None);
        assert_eq!(fmt, "audio-24khz-48kbit-mp3-mono");
        assert_eq!(ct, "audio/mpeg");
        assert_eq!(ext, "mp3");
    }

    #[test]
    fn get_output_format_wav() {
        let (fmt, ct, ext) = EdgeTtsAdapter::get_output_format(Some("wav"));
        assert_eq!(fmt, "audio-24khz-16bit-mono-pcm");
        assert_eq!(ct, "audio/wav");
        assert_eq!(ext, "wav");
    }

    #[test]
    fn get_output_format_webm_is_ogg() {
        let (_, ct, ext) = EdgeTtsAdapter::get_output_format(Some("webm"));
        assert_eq!(ct, "audio/ogg");
        assert_eq!(ext, "ogg");
    }

    #[test]
    fn get_output_format_unknown_defaults_mp3() {
        let (fmt, _, _) = EdgeTtsAdapter::get_output_format(Some("xyz"));
        assert_eq!(fmt, "audio-24khz-48kbit-mp3-mono");
    }

    // ============ compute_sec_ms_gec ============

    #[test]
    fn compute_sec_ms_gec_is_uppercase_hex_64() {
        let gec = EdgeTtsAdapter::compute_sec_ms_gec(1_700_000_000);
        assert_eq!(gec.len(), 64);
        assert!(gec
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()));
    }

    #[test]
    fn compute_sec_ms_gec_deterministic() {
        let a = EdgeTtsAdapter::compute_sec_ms_gec(1_700_000_000);
        let b = EdgeTtsAdapter::compute_sec_ms_gec(1_700_000_000);
        assert_eq!(a, b);
    }

    #[test]
    fn compute_sec_ms_gec_same_within_5min_window() {
        // 1700000000 在窗口 [1699999800, 1700000100) 内
        let a = EdgeTtsAdapter::compute_sec_ms_gec(1_700_000_000);
        let b = EdgeTtsAdapter::compute_sec_ms_gec(1_700_000_099);
        assert_eq!(a, b);
    }

    #[test]
    fn compute_sec_ms_gec_changes_across_boundary() {
        let a = EdgeTtsAdapter::compute_sec_ms_gec(1_700_000_000);
        let b = EdgeTtsAdapter::compute_sec_ms_gec(1_700_000_100);
        assert_ne!(a, b);
    }

    // ============ URL 构造 ============

    #[test]
    fn build_ws_url_contains_all_params() {
        let url =
            EdgeTtsAdapter::build_ws_url("GEC123", "conn456", "https://speech.platform.bing.com");
        assert!(url.starts_with("wss://speech.platform.bing.com"));
        assert!(url.contains(WS_PATH));
        assert!(url.contains("TrustedClientToken=6A5AA1D4EAFF4E9FB37E23D68491D6F4"));
        assert!(url.contains("Sec-MS-GEC=GEC123"));
        assert!(url.contains("Sec-MS-GEC-Version=1-130.0.2849.68"));
        assert!(url.contains("ConnectionId=conn456"));
    }

    #[test]
    fn build_ws_url_converts_http_to_ws() {
        let url = EdgeTtsAdapter::build_ws_url("g", "c", "http://localhost:1234");
        assert!(url.starts_with("ws://localhost:1234"));
    }

    #[test]
    fn build_voices_url_contains_all_params() {
        let url = EdgeTtsAdapter::build_voices_url("GEC123", "https://speech.platform.bing.com");
        assert!(url.starts_with("https://speech.platform.bing.com"));
        assert!(url.contains(VOICES_PATH));
        assert!(url.contains("TrustedClientToken="));
        assert!(url.contains("Sec-MS-GEC=GEC123"));
    }

    #[test]
    fn build_ws_request_sets_origin_and_user_agent() {
        let req = EdgeTtsAdapter::build_ws_request("wss://example.com/path").unwrap();
        assert_eq!(
            req.headers().get("origin").unwrap(),
            "https://speech.platform.bing.com"
        );
        assert!(req.headers().get("user-agent").is_some());
    }

    // ============ SSML / 消息构造 ============

    #[test]
    fn build_ssml_contains_voice_and_text() {
        let ssml =
            EdgeTtsAdapter::build_ssml("hello", "zh-CN-XiaoxiaoNeural", "+0%", "+0Hz", "+0%");
        assert!(ssml.contains("zh-CN-XiaoxiaoNeural"));
        assert!(ssml.contains("hello"));
        assert!(ssml.contains("rate='+0%'"));
        assert!(ssml.contains("pitch='+0Hz'"));
        assert!(ssml.contains("volume='+0%'"));
        assert!(ssml.contains("xml:lang='zh-CN'"));
    }

    #[test]
    fn build_ssml_escapes_xml_special_chars() {
        let ssml =
            EdgeTtsAdapter::build_ssml("a<b&c>d", "zh-CN-XiaoxiaoNeural", "+0%", "+0Hz", "+0%");
        assert!(ssml.contains("&lt;"));
        assert!(ssml.contains("&gt;"));
        assert!(ssml.contains("&amp;"));
        assert!(!ssml.contains("a<b"));
    }

    #[test]
    fn build_config_message_contains_path_and_format() {
        let msg = EdgeTtsAdapter::build_config_message(
            "audio-24khz-48kbit-mp3-mono",
            "2023-01-01T00:00:00.000Z",
        );
        assert!(msg.contains("Path:speech.config"));
        assert!(msg.contains("audio-24khz-48kbit-mp3-mono"));
        assert!(msg.contains("X-Timestamp:2023-01-01T00:00:00.000Z"));
    }

    #[test]
    fn build_synth_message_contains_path_ssml_and_request_id() {
        let ssml = "<speak>hi</speak>";
        let msg = EdgeTtsAdapter::build_synth_message(ssml, "req-123", "2023-01-01T00:00:00.000Z");
        assert!(msg.contains("Path:ssml"));
        assert!(msg.contains("X-RequestId:req-123"));
        assert!(msg.contains(ssml));
    }

    // ============ WS 消息解析 ============

    #[test]
    fn parse_audio_binary_extracts_audio() {
        let header = b"Path:audio\r\n";
        let audio = [0xFFu8, 0xFB, 0x90, 0x00];
        let mut msg = Vec::new();
        msg.extend_from_slice(&(header.len() as u16).to_be_bytes());
        msg.extend_from_slice(header);
        msg.extend_from_slice(&audio);
        let result = EdgeTtsAdapter::parse_audio_binary(&msg).unwrap();
        assert_eq!(result, &audio);
    }

    #[test]
    fn parse_audio_binary_returns_none_for_non_audio() {
        let header = b"Path:turn.end\r\n";
        let mut msg = Vec::new();
        msg.extend_from_slice(&(header.len() as u16).to_be_bytes());
        msg.extend_from_slice(header);
        msg.extend_from_slice(b"body");
        assert!(EdgeTtsAdapter::parse_audio_binary(&msg).is_none());
    }

    #[test]
    fn parse_audio_binary_returns_none_for_short_msg() {
        assert!(EdgeTtsAdapter::parse_audio_binary(&[]).is_none());
        assert!(EdgeTtsAdapter::parse_audio_binary(&[0u8]).is_none());
    }

    #[test]
    fn parse_ws_text_path_extracts_value() {
        let msg = "X-RequestId:abc\r\nContent-Type:application/json\r\nPath:turn.end\r\n\r\n{}";
        assert_eq!(EdgeTtsAdapter::parse_ws_text_path(msg), Some("turn.end"));
    }

    #[test]
    fn parse_ws_text_path_returns_none_when_missing() {
        assert!(EdgeTtsAdapter::parse_ws_text_path("no path here").is_none());
    }

    #[test]
    fn extract_ws_error_detects_voice_error() {
        let msg = "X-RequestId:abc\r\nPath:response\r\n\r\n\
                   {\"Type\":\"error\",\"Code\":\"VoiceNotFound\",\"Message\":\"voice not available\"}";
        let err = EdgeTtsAdapter::extract_ws_error(msg).unwrap();
        assert!(matches!(err, AibridgeError::VoiceNotAvailable { .. }));
    }

    #[test]
    fn extract_ws_error_maps_generic_error_to_api() {
        let msg = "Path:response\r\n\r\n\
                   {\"Type\":\"error\",\"Code\":\"AudioDeliveryFailed\",\"Message\":\"delivery failed\"}";
        let err = EdgeTtsAdapter::extract_ws_error(msg).unwrap();
        assert!(matches!(err, AibridgeError::Api { .. }));
    }

    #[test]
    fn extract_ws_error_returns_none_for_non_error() {
        let msg = "Path:response\r\n\r\n{\"Type\":\"response\"}";
        assert!(EdgeTtsAdapter::extract_ws_error(msg).is_none());
    }

    // ============ 参数映射 ============

    #[test]
    fn build_rate_from_speed() {
        assert_eq!(EdgeTtsAdapter::build_rate(None, &HashMap::new()), "+0%");
        assert_eq!(
            EdgeTtsAdapter::build_rate(Some(1.0), &HashMap::new()),
            "+0%"
        );
        assert_eq!(
            EdgeTtsAdapter::build_rate(Some(2.0), &HashMap::new()),
            "+100%"
        );
        assert_eq!(
            EdgeTtsAdapter::build_rate(Some(0.5), &HashMap::new()),
            "-50%"
        );
    }

    #[test]
    fn build_rate_prefers_extra_string() {
        let mut extra = HashMap::new();
        extra.insert("rate".to_string(), serde_json::json!("-20%"));
        assert_eq!(EdgeTtsAdapter::build_rate(Some(1.0), &extra), "-20%");
    }

    #[test]
    fn build_pitch_from_value() {
        assert_eq!(EdgeTtsAdapter::build_pitch(None, &HashMap::new()), "+0Hz");
        assert_eq!(
            EdgeTtsAdapter::build_pitch(Some(1.0), &HashMap::new()),
            "+100Hz"
        );
        assert_eq!(
            EdgeTtsAdapter::build_pitch(Some(-0.5), &HashMap::new()),
            "-50Hz"
        );
    }

    #[test]
    fn build_volume_from_value() {
        assert_eq!(EdgeTtsAdapter::build_volume(None, &HashMap::new()), "+0%");
        assert_eq!(
            EdgeTtsAdapter::build_volume(Some(2.0), &HashMap::new()),
            "+100%"
        );
        assert_eq!(
            EdgeTtsAdapter::build_volume(Some(0.5), &HashMap::new()),
            "-50%"
        );
    }

    // ============ locale / 时间戳 ============

    #[test]
    fn locale_from_voice_extracts_language() {
        assert_eq!(
            EdgeTtsAdapter::locale_from_voice("zh-CN-XiaoxiaoNeural"),
            "zh-CN"
        );
        assert_eq!(
            EdgeTtsAdapter::locale_from_voice("en-US-JennyNeural"),
            "en-US"
        );
        assert_eq!(
            EdgeTtsAdapter::locale_from_voice("ja-JP-NanamiNeural"),
            "ja-JP"
        );
    }

    #[test]
    fn locale_from_voice_defaults_for_no_dash() {
        assert_eq!(EdgeTtsAdapter::locale_from_voice("voice"), "en-US");
    }

    #[test]
    fn utc_timestamp_iso_known_value() {
        // 1700000000 = 2023-11-14T22:13:20 UTC
        assert_eq!(
            EdgeTtsAdapter::utc_timestamp_iso(1_700_000_000),
            "2023-11-14T22:13:20.000Z"
        );
    }

    // ============ 空音频错误分类 ============

    #[test]
    fn empty_audio_error_service_unavailable_when_online() {
        let err = EdgeTtsAdapter::empty_audio_error("v1", true);
        assert!(matches!(err, AibridgeError::ServiceUnavailable { .. }));
    }

    #[test]
    fn empty_audio_error_voice_not_available_when_offline() {
        let err = EdgeTtsAdapter::empty_audio_error("v1", false);
        assert!(matches!(err, AibridgeError::VoiceNotAvailable { .. }));
    }

    // ============ convert_voice ============

    #[test]
    fn convert_voice_maps_all_fields() {
        let raw = EdgeVoiceRaw {
            short_name: "zh-CN-XiaoxiaoNeural".into(),
            name: "Microsoft Server Speech Text to Speech Voice (zh-CN, XiaoxiaoNeural)".into(),
            locale: "zh-CN".into(),
            gender: "Female".into(),
            friendly_name: Some("Xiaoxiao".into()),
        };
        let v = EdgeTtsAdapter::convert_voice(raw);
        assert_eq!(v.short_name.as_deref(), Some("zh-CN-XiaoxiaoNeural"));
        assert_eq!(v.name.as_deref(), Some("Xiaoxiao")); // friendly_name 优先
        assert_eq!(v.locale.as_deref(), Some("zh-CN"));
        assert_eq!(v.gender.as_deref(), Some("Female"));
        assert_eq!(v.voice_id.as_deref(), Some("zh-CN-XiaoxiaoNeural"));
    }

    #[test]
    fn convert_voice_falls_back_to_name_without_friendly() {
        let raw = EdgeVoiceRaw {
            short_name: "v1".into(),
            name: "FullName".into(),
            locale: "en-US".into(),
            gender: "Male".into(),
            friendly_name: None,
        };
        let v = EdgeTtsAdapter::convert_voice(raw);
        assert_eq!(v.name.as_deref(), Some("FullName"));
    }

    // ============ 辅助函数 ============

    #[test]
    fn escape_xml_replaces_special_chars() {
        assert_eq!(escape_xml("a<b&c>d"), "a&lt;b&amp;c&gt;d");
        assert_eq!(escape_xml("plain"), "plain");
    }

    #[test]
    fn generate_connection_id_is_32_hex() {
        let id = generate_connection_id();
        assert_eq!(id.len(), 32);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // ============ list_voices（HTTP，mockito） ============

    #[tokio::test]
    async fn list_voices_fetches_and_parses() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!([
            {"ShortName": "zh-CN-XiaoxiaoNeural", "Name": "Microsoft Xiaoxiao", "Locale": "zh-CN", "Gender": "Female", "FriendlyName": "Xiaoxiao"},
            {"ShortName": "en-US-JennyNeural", "Name": "Microsoft Jenny", "Locale": "en-US", "Gender": "Female"}
        ]);
        server
            .mock("GET", VOICES_PATH)
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let voices = adapter.list_voices(None).await.unwrap();
        assert_eq!(voices.len(), 2);
        assert_eq!(
            voices[0].short_name.as_deref(),
            Some("zh-CN-XiaoxiaoNeural")
        );
        assert_eq!(voices[0].gender.as_deref(), Some("Female"));
        assert_eq!(voices[0].locale.as_deref(), Some("zh-CN"));
        assert_eq!(voices[1].name.as_deref(), Some("Microsoft Jenny")); // 无 FriendlyName 回退 Name
    }

    #[tokio::test]
    async fn list_voices_filters_by_language() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!([
            {"ShortName": "zh-CN-XiaoxiaoNeural", "Name": "x", "Locale": "zh-CN", "Gender": "Female"},
            {"ShortName": "zh-CN-YunxiNeural", "Name": "y", "Locale": "zh-CN", "Gender": "Male"},
            {"ShortName": "en-US-JennyNeural", "Name": "j", "Locale": "en-US", "Gender": "Female"}
        ]);
        server
            .mock("GET", VOICES_PATH)
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let zh = adapter.list_voices(Some("zh-CN")).await.unwrap();
        assert_eq!(zh.len(), 2);
        let en = adapter.list_voices(Some("en-US")).await.unwrap();
        assert_eq!(en.len(), 1);
        let none = adapter.list_voices(Some("fr-FR")).await.unwrap();
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn list_voices_caches_second_call() {
        let mut server = mockito::Server::new_async().await;
        let m = server
            .mock("GET", VOICES_PATH)
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(
                serde_json::json!([{"ShortName":"v1","Name":"n","Locale":"zh-CN","Gender":"Female"}])
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
    async fn list_voices_error_500_returns_service_unavailable() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("GET", VOICES_PATH)
            .match_query(mockito::Matcher::Any)
            .with_status(503)
            .with_body("unavailable")
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let err = adapter.list_voices(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ServiceUnavailable { .. }));
    }

    // ============ recommend_voices ============

    #[tokio::test]
    async fn recommend_voices_filters_by_gender_and_limit() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!([
            {"ShortName": "zh-CN-XiaoxiaoNeural", "Name": "x", "Locale": "zh-CN", "Gender": "Female"},
            {"ShortName": "zh-CN-YunxiNeural", "Name": "y", "Locale": "zh-CN", "Gender": "Male"},
            {"ShortName": "zh-CN-XiaoyiNeural", "Name": "z", "Locale": "zh-CN", "Gender": "Female"}
        ]);
        server
            .mock("GET", VOICES_PATH)
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        let female = adapter
            .recommend_voices(Some("zh-CN"), Some("Female"), 10)
            .await
            .unwrap();
        assert_eq!(female.len(), 2);
        let limited = adapter
            .recommend_voices(Some("zh-CN"), None, 1)
            .await
            .unwrap();
        assert_eq!(limited.len(), 1);
    }

    // ============ check_voice_available ============

    #[tokio::test]
    async fn check_voice_available_true_for_existing() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!([
            {"ShortName": "zh-CN-XiaoxiaoNeural", "Name": "x", "Locale": "zh-CN", "Gender": "Female"}
        ]);
        server
            .mock("GET", VOICES_PATH)
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_adapter(Some(server.url()));
        assert!(adapter.check_voice_available("zh-CN-XiaoxiaoNeural").await);
        assert!(!adapter.check_voice_available("zh-CN-Nonexistent").await);
    }

    // ============ 音色降级（speech_with_fallback） ============

    #[tokio::test]
    async fn fallback_tries_next_on_voice_not_available() {
        let count = Arc::new(AtomicU32::new(0));
        let count2 = count.clone();
        let single = move |v: &str| {
            let v = v.to_string();
            let count3 = count2.clone();
            async move {
                count3.fetch_add(1, Ordering::SeqCst);
                if v == "v3" {
                    Ok(SpeechResult {
                        audio_data: Some(vec![1]),
                        ..Default::default()
                    })
                } else {
                    Err(AibridgeError::voice_not_available(&v))
                }
            }
        };
        let candidates = vec!["v1".to_string(), "v2".to_string(), "v3".to_string()];
        let result = EdgeTtsAdapter::speech_with_fallback(candidates, single).await;
        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn fallback_returns_last_error_when_all_fail() {
        let single = |v: &str| {
            let v = v.to_string();
            async move { Err(AibridgeError::voice_not_available(&v)) }
        };
        let candidates = vec!["v1".to_string(), "v2".to_string()];
        let result = EdgeTtsAdapter::speech_with_fallback(candidates, single).await;
        assert!(matches!(
            result,
            Err(AibridgeError::VoiceNotAvailable { .. })
        ));
    }

    #[tokio::test]
    async fn fallback_single_voice_raises_directly() {
        let single = |v: &str| {
            let v = v.to_string();
            async move { Err(AibridgeError::voice_not_available(&v)) }
        };
        let candidates = vec!["v1".to_string()];
        let result = EdgeTtsAdapter::speech_with_fallback(candidates, single).await;
        assert!(matches!(
            result,
            Err(AibridgeError::VoiceNotAvailable { .. })
        ));
    }

    #[tokio::test]
    async fn fallback_non_fallback_error_raises_immediately() {
        let count = Arc::new(AtomicU32::new(0));
        let count2 = count.clone();
        let single = move |_v: &str| {
            let count3 = count2.clone();
            async move {
                count3.fetch_add(1, Ordering::SeqCst);
                Err(AibridgeError::authentication("auth fail")) // 非降级错误
            }
        };
        let candidates = vec!["v1".to_string(), "v2".to_string()];
        let result = EdgeTtsAdapter::speech_with_fallback(candidates, single).await;
        assert!(matches!(result, Err(AibridgeError::Authentication { .. })));
        assert_eq!(count.load(Ordering::SeqCst), 1); // 只调一次
    }

    #[tokio::test]
    async fn fallback_first_success_skips_rest() {
        let count = Arc::new(AtomicU32::new(0));
        let count2 = count.clone();
        let single = move |_v: &str| {
            let count3 = count2.clone();
            async move {
                count3.fetch_add(1, Ordering::SeqCst);
                Ok(SpeechResult {
                    audio_data: Some(vec![1]),
                    ..Default::default()
                })
            }
        };
        let candidates = vec!["v1".to_string(), "v2".to_string(), "v3".to_string()];
        let result = EdgeTtsAdapter::speech_with_fallback(candidates, single).await;
        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 1); // 第一个成功，不试后续
    }
}
