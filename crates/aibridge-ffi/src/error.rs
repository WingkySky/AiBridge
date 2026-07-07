//! FFI 错误处理
//!
//! 对应设计文档 7 节错误模型：
//! - [`AibridgeStatus`]（i32 返回码）：0 成功，负数为错误类别
//! - 线程局部 `last_error` 槽：存 JSON 字符串 `{code,message,details,retryable}`
//! - [`aibridge_last_error`] 暴露给 C 侧读取
//!
//! 错误码映射 `aibridge_core::AibridgeError` 的各变体，便于各语言绑定
//! 做异常映射时无需解析 JSON 即可快速分类。

#[cfg(test)]
use crate::string;
use aibridge_core::error::AibridgeError;
use std::cell::RefCell;
use std::ffi::CString;
use std::ptr;

/// FFI 返回码类型
///
/// 0 表示成功；负数表示各类错误（与 [`AibridgeError`] 变体一一对应）。
pub type AibridgeStatus = i32;

/// 成功
pub const AIBRIDGE_OK: AibridgeStatus = 0;

/// 流式：正常拉取到一个 chunk（仅 `aibridge_stream_next` 使用）
pub const AIBRIDGE_STREAM_CHUNK: AibridgeStatus = 0;

/// 流式：流已结束（仅 `aibridge_stream_next` 使用，1 表示 EOF）
pub const AIBRIDGE_STREAM_END: AibridgeStatus = 1;

// —— 错误类别（负数）——
/// 认证错误
pub const AIBRIDGE_ERR_AUTHENTICATION: AibridgeStatus = -1;
/// 限流错误
pub const AIBRIDGE_ERR_RATE_LIMIT: AibridgeStatus = -2;
/// 参数校验错误
pub const AIBRIDGE_ERR_VALIDATION: AibridgeStatus = -3;
/// 模型不存在
pub const AIBRIDGE_ERR_MODEL_NOT_FOUND: AibridgeStatus = -4;
/// API 调用错误（HTTP 4xx/5xx）
pub const AIBRIDGE_ERR_API: AibridgeStatus = -5;
/// 网络错误
pub const AIBRIDGE_ERR_NETWORK: AibridgeStatus = -6;
/// 超时
pub const AIBRIDGE_ERR_TIMEOUT: AibridgeStatus = -7;
/// 不支持的能力
pub const AIBRIDGE_ERR_UNSUPPORTED_CAPABILITY: AibridgeStatus = -8;
/// Provider 不存在
pub const AIBRIDGE_ERR_PROVIDER_NOT_FOUND: AibridgeStatus = -9;
/// 音色不可用
pub const AIBRIDGE_ERR_VOICE_NOT_AVAILABLE: AibridgeStatus = -10;
/// 服务暂时不可用
pub const AIBRIDGE_ERR_SERVICE_UNAVAILABLE: AibridgeStatus = -11;
/// FFI 层通用错误（参数为空、JSON 解析失败、内部 panic 等）
pub const AIBRIDGE_ERR_FFI: AibridgeStatus = -100;

// 线程局部 last_error 槽
//
// 存储 JSON 字符串 `{code,message,details,retryable}`，C 侧通过
// [`aibridge_last_error`] 读取。每个线程独立，无需加锁。
// 用 `Option<CString>` 便于返回裸指针（指针指向 CString 内部缓冲）。
thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// 将核心层错误写入线程局部 last_error 槽，并返回对应的 FFI 错误码
///
/// 内部把错误序列化为 JSON：`{"code":"...","message":"...","details":...,"retryable":bool}`。
/// `details` 字段：Validation 错误带原始 details，其余为 null。
pub fn set_last_error(err: &AibridgeError) -> AibridgeStatus {
    let code = err.code();
    let message = err.to_string();
    let retryable = err.is_retryable();
    let details = match err {
        AibridgeError::Validation { details, .. } => details.clone(),
        AibridgeError::RateLimit { retry_after, .. } if retry_after.is_some() => {
            serde_json::json!({ "retry_after": retry_after })
        }
        _ => serde_json::Value::Null,
    };

    let payload = serde_json::json!({
        "code": code,
        "message": message,
        "details": details,
        "retryable": retryable,
    });
    let json_str = payload.to_string();

    // 写入线程局部槽（失败则清空，避免残留旧错误）
    match CString::new(json_str) {
        Ok(cstr) => {
            LAST_ERROR.with(|slot| {
                *slot.borrow_mut() = Some(cstr);
            });
        }
        Err(_) => {
            LAST_ERROR.with(|slot| {
                *slot.borrow_mut() = None;
            });
        }
    }

    status_from_error(err)
}

/// 写入 FFI 层自有的错误信息（如参数为空、JSON 解析失败、panic 等）
///
/// 与 [`set_last_error`] 不同，此处不依赖 `AibridgeError`，统一用
/// `AIBRIDGE_ERR_FFI` 错误码。`details` 可为任意 JSON 值。
pub fn set_ffi_error(message: &str, details: serde_json::Value) -> AibridgeStatus {
    let payload = serde_json::json!({
        "code": "ffi_error",
        "message": message,
        "details": details,
        "retryable": false,
    });
    let json_str = payload.to_string();
    if let Ok(cstr) = CString::new(json_str) {
        LAST_ERROR.with(|slot| {
            *slot.borrow_mut() = Some(cstr);
        });
    }
    AIBRIDGE_ERR_FFI
}

/// 写入简单的 FFI 错误（无 details）
pub fn set_ffi_error_simple(message: &str) -> AibridgeStatus {
    set_ffi_error(message, serde_json::Value::Null)
}

/// 清空当前线程的 last_error 槽（成功路径调用）
pub fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

/// 读取当前线程 last_error 的 C 字符串指针
///
/// 返回的指针指向线程局部 `CString` 内部缓冲，调用方**不应释放**。
/// 若当前线程无错误，返回空指针。
///
/// # 线程安全
/// 仅返回当前线程的错误；各语言绑定的异步包装应保证读取与触发错误的调用
/// 在同一线程，或在 FFI 边界立即读取后转存。
pub fn last_error_ptr() -> *const std::os::raw::c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|cstr| cstr.as_ptr())
            .unwrap_or(ptr::null())
    })
}

/// 将核心层错误映射到 FFI 错误码
pub fn status_from_error(err: &AibridgeError) -> AibridgeStatus {
    match err {
        AibridgeError::Authentication { .. } => AIBRIDGE_ERR_AUTHENTICATION,
        AibridgeError::RateLimit { .. } => AIBRIDGE_ERR_RATE_LIMIT,
        AibridgeError::Validation { .. } => AIBRIDGE_ERR_VALIDATION,
        AibridgeError::ModelNotFound { .. } => AIBRIDGE_ERR_MODEL_NOT_FOUND,
        AibridgeError::Api { .. } => AIBRIDGE_ERR_API,
        AibridgeError::Network(_) => AIBRIDGE_ERR_NETWORK,
        AibridgeError::Timeout => AIBRIDGE_ERR_TIMEOUT,
        AibridgeError::UnsupportedCapability { .. } => AIBRIDGE_ERR_UNSUPPORTED_CAPABILITY,
        AibridgeError::ProviderNotFound { .. } => AIBRIDGE_ERR_PROVIDER_NOT_FOUND,
        AibridgeError::VoiceNotAvailable { .. } => AIBRIDGE_ERR_VOICE_NOT_AVAILABLE,
        AibridgeError::ServiceUnavailable { .. } => AIBRIDGE_ERR_SERVICE_UNAVAILABLE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_mapping_is_complete() {
        // 覆盖每个变体，确保映射不漏
        assert_eq!(
            status_from_error(&AibridgeError::authentication("x")),
            AIBRIDGE_ERR_AUTHENTICATION
        );
        assert_eq!(
            status_from_error(&AibridgeError::rate_limit("x")),
            AIBRIDGE_ERR_RATE_LIMIT
        );
        assert_eq!(
            status_from_error(&AibridgeError::validation("x")),
            AIBRIDGE_ERR_VALIDATION
        );
        assert_eq!(
            status_from_error(&AibridgeError::model_not_found("m")),
            AIBRIDGE_ERR_MODEL_NOT_FOUND
        );
        assert_eq!(
            status_from_error(&AibridgeError::api(500, "x")),
            AIBRIDGE_ERR_API
        );
        // Network 变体需构造 reqwest::Error，跳过；用 UnsupportedCapability 等
        assert_eq!(
            status_from_error(&AibridgeError::Timeout),
            AIBRIDGE_ERR_TIMEOUT
        );
        assert_eq!(
            status_from_error(&AibridgeError::unsupported_capability("c")),
            AIBRIDGE_ERR_UNSUPPORTED_CAPABILITY
        );
        assert_eq!(
            status_from_error(&AibridgeError::provider_not_found("p")),
            AIBRIDGE_ERR_PROVIDER_NOT_FOUND
        );
        assert_eq!(
            status_from_error(&AibridgeError::voice_not_available("v")),
            AIBRIDGE_ERR_VOICE_NOT_AVAILABLE
        );
        assert_eq!(
            status_from_error(&AibridgeError::service_unavailable("s")),
            AIBRIDGE_ERR_SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn set_last_error_writes_valid_json() {
        let status = set_last_error(&AibridgeError::rate_limit("慢一点"));
        assert_eq!(status, AIBRIDGE_ERR_RATE_LIMIT);
        let ptr = last_error_ptr();
        assert!(!ptr.is_null());
        let json = unsafe { string::cstr_to_string_unchecked(ptr) };
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["code"], "rate_limit_error");
        assert_eq!(v["retryable"], true);
        assert!(v["message"].as_str().unwrap().contains("慢一点"));
        clear_last_error();
    }

    #[test]
    fn validation_error_carries_details() {
        let err =
            AibridgeError::validation_with_details("坏", serde_json::json!({"field": "model"}));
        set_last_error(&err);
        let ptr = last_error_ptr();
        let json = unsafe { string::cstr_to_string_unchecked(ptr) };
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["details"]["field"], "model");
        clear_last_error();
    }

    #[test]
    fn set_ffi_error_writes_ffi_code() {
        let status = set_ffi_error("参数为空", serde_json::Value::Null);
        assert_eq!(status, AIBRIDGE_ERR_FFI);
        let ptr = last_error_ptr();
        assert!(!ptr.is_null());
        let json = unsafe { string::cstr_to_string_unchecked(ptr) };
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["code"], "ffi_error");
        assert_eq!(v["retryable"], false);
        clear_last_error();
    }

    #[test]
    fn clear_last_error_empties_slot() {
        set_ffi_error_simple("x");
        clear_last_error();
        assert!(last_error_ptr().is_null());
    }

    #[test]
    fn last_error_is_thread_local() {
        // 主线程设置后，子线程应读不到
        set_ffi_error_simple("主线程错误");
        let child = std::thread::spawn(|| last_error_ptr().is_null());
        assert!(child.join().unwrap(), "子线程不应看到主线程的 last_error");
        clear_last_error();
    }
}
