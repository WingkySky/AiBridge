//! 工具函数
//!
//! 通用的工具函数集合。
//! 对应 Python v1 (agn-sdk) 的 `agn/core/utils.py`。

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use serde_json::Value;

/// 默认 base64 引擎（标准字母表，带 padding）
const STANDARD: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

/// 生成唯一 ID
///
/// 对应 Python v1 `generate_id`。返回 12 位十六进制 + 可选前缀。
pub fn generate_id(prefix: &str) -> String {
    let id = uuid_v4_hex_12();
    if prefix.is_empty() {
        id
    } else {
        format!("{prefix}_{id}")
    }
}

/// 生成 12 位十六进制 ID（基于 uuid v4 的前 6 字节）
fn uuid_v4_hex_12() -> String {
    // 简易实现：用系统时间 + 计数器替代完整 uuid 依赖
    // 12 位 hex = 6 字节；取时间戳纳秒低 6 字节
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let bytes = (nanos as u64).to_be_bytes();
    // 取后 6 字节
    let six: [u8; 6] = bytes[2..8].try_into().unwrap();
    hex_encode(&six)
}

/// 获取当前 Unix 时间戳（秒）
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 获取当前 Unix 时间戳（毫秒）
pub fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// 检查字符串是否为 base64 编码
///
/// 对应 Python v1 `is_base64`。data URI 前缀会先被剥离再校验。
pub fn is_base64(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    // 移除可能的 data URI 前缀
    let s = s.split_once(',').map(|(_, d)| d).unwrap_or(s);
    if s.is_empty() {
        return false;
    }
    STANDARD.decode(s).is_ok()
}

/// 将字节数据编码为 base64 字符串
pub fn encode_base64(data: &[u8]) -> String {
    STANDARD.encode(data)
}

/// 将 base64 字符串解码为字节数据
///
/// 会自动剥离 data URI 前缀（逗号前的部分）。
pub fn decode_base64(data: &str) -> Result<Vec<u8>, base64::DecodeError> {
    let data = data.split_once(',').map(|(_, d)| d).unwrap_or(data);
    STANDARD.decode(data)
}

/// 计算 MD5 哈希
///
/// 注意：MD5 仅用于非安全场景（如缓存键、幂等标识）。
pub fn md5_hash(data: &str) -> String {
    // 不引入 md5 依赖：用简单的 FNV-1a 替代作为缓存键
    // 若后续需要真正的 MD5，再加 md-5 依赖
    fnv1a_64(data)
}

/// FNV-1a 64 位哈希（作为 MD5 的轻量替代，仅用于非安全场景）
fn fnv1a_64(data: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in data.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// 验证并修正视频尺寸（宽高必须是 8 的倍数）
///
/// 向上取整到最近的 8 的倍数。
pub fn validate_video_dimensions(width: u32, height: u32) -> (u32, u32) {
    let w = width.div_ceil(8) * 8;
    let h = height.div_ceil(8) * 8;
    (w, h)
}

/// 解析图像尺寸字符串（如 "1024x1024"）
///
/// 返回 `(width, height)`，格式非法时返回 `Err`。
pub fn parse_size(size: &str) -> Result<(u32, u32), String> {
    let lower = size.to_lowercase();
    let parts: Vec<&str> = lower.split('x').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid size format: {size}"));
    }
    let w: u32 = parts[0]
        .parse()
        .map_err(|_| format!("Invalid size format: {size}"))?;
    let h: u32 = parts[1]
        .parse()
        .map_err(|_| format!("Invalid size format: {size}"))?;
    Ok((w, h))
}

/// 构建图像尺寸字符串
pub fn build_size_string(width: u32, height: u32) -> String {
    format!("{width}x{height}")
}

/// 合并多个 JSON 对象
///
/// 对应 Python v1 `merge_dicts`。后续值覆盖前面。
/// 任一参数为 `Value::Null` 或非对象时跳过。
pub fn merge_json(base: &Value, overrides: &[&Value]) -> Value {
    let mut result = match base {
        Value::Object(map) => map.clone(),
        _ => serde_json::Map::new(),
    };
    for ov in overrides {
        if let Value::Object(m) = ov {
            for (k, v) in m {
                result.insert(k.clone(), v.clone());
            }
        }
    }
    Value::Object(result)
}

/// 合并多个 HashMap（后续覆盖前面）
pub fn merge_maps(
    base: HashMap<String, Value>,
    overrides: &[HashMap<String, Value>],
) -> HashMap<String, Value> {
    let mut result = base;
    for ov in overrides {
        for (k, v) in ov {
            result.insert(k.clone(), v.clone());
        }
    }
    result
}

/// 十六进制编码（小写）
fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_id_with_prefix() {
        let id = generate_id("task");
        assert!(id.starts_with("task_"));
        assert!(id.len() > "task_".len());
    }

    #[test]
    fn generate_id_without_prefix() {
        let id = generate_id("");
        assert!(!id.is_empty());
        assert!(!id.contains('_'));
    }

    #[test]
    fn current_timestamp_nonzero() {
        let t = current_timestamp();
        assert!(t > 0);
    }

    #[test]
    fn current_timestamp_ms_greater_than_seconds() {
        let s = current_timestamp();
        let ms = current_timestamp_ms();
        assert!(ms >= s * 1000);
    }

    #[test]
    fn is_base64_valid() {
        assert!(is_base64("aGVsbG8=")); // "hello"
    }

    #[test]
    fn is_base64_invalid() {
        assert!(!is_base64("not base64!"));
        assert!(!is_base64(""));
    }

    #[test]
    fn is_base64_strips_data_uri() {
        assert!(is_base64("data:image/png;base64,aGVsbG8="));
    }

    #[test]
    fn encode_decode_base64_roundtrip() {
        let original = b"hello world";
        let encoded = encode_base64(original);
        let decoded = decode_base64(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn decode_base64_strips_data_uri() {
        let decoded = decode_base64("data:image/png;base64,aGVsbG8=").unwrap();
        assert_eq!(decoded, b"hello");
    }

    #[test]
    fn md5_hash_stable() {
        let h1 = md5_hash("test");
        let h2 = md5_hash("test");
        assert_eq!(h1, h2);
        assert_ne!(md5_hash("test"), md5_hash("other"));
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn validate_video_dimensions_rounds_up_to_8() {
        assert_eq!(validate_video_dimensions(1280, 720), (1280, 720));
        assert_eq!(validate_video_dimensions(1281, 721), (1288, 728));
        assert_eq!(validate_video_dimensions(1, 1), (8, 8));
    }

    #[test]
    fn parse_size_valid() {
        assert_eq!(parse_size("1024x1024").unwrap(), (1024, 1024));
        assert_eq!(parse_size("1792X1024").unwrap(), (1792, 1024)); // 大写 X 也接受
    }

    #[test]
    fn parse_size_invalid() {
        assert!(parse_size("1024").is_err());
        assert!(parse_size("abc").is_err());
        assert!(parse_size("1024x").is_err());
        assert!(parse_size("").is_err());
    }

    #[test]
    fn build_size_string_correct() {
        assert_eq!(build_size_string(1024, 768), "1024x768");
    }

    #[test]
    fn merge_json_overrides_win() {
        let base = serde_json::json!({"a": 1, "b": 2});
        let ov1 = serde_json::json!({"b": 3, "c": 4});
        let result = merge_json(&base, &[&ov1]);
        assert_eq!(result["a"], 1);
        assert_eq!(result["b"], 3);
        assert_eq!(result["c"], 4);
    }

    #[test]
    fn merge_json_ignores_non_object() {
        let base = serde_json::json!({"a": 1});
        let ov = serde_json::json!(42);
        let result = merge_json(&base, &[&ov]);
        assert_eq!(result["a"], 1);
    }

    #[test]
    fn merge_json_empty_overrides() {
        let base = serde_json::json!({"a": 1});
        let result = merge_json(&base, &[]);
        assert_eq!(result["a"], 1);
    }

    #[test]
    fn merge_maps_overrides_win() {
        let mut base = HashMap::new();
        base.insert("a".into(), serde_json::json!(1));
        let mut ov = HashMap::new();
        ov.insert("a".into(), serde_json::json!(2));
        ov.insert("b".into(), serde_json::json!(3));
        let result = merge_maps(base, &[ov]);
        assert_eq!(result["a"], 2);
        assert_eq!(result["b"], 3);
    }

    #[test]
    fn hex_encode_lowercase() {
        assert_eq!(hex_encode(&[0x0a, 0xff]), "0aff");
    }
}
