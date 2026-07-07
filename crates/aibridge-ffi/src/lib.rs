//! AIBridge C ABI - FFI 层
//!
//! 暴露 C ABI 供 Go/JVM/.NET 调用。Python/JS 通过 aibridge-python /
//! aibridge-node 直连 aibridge-core，不走本层。
//!
//! 设计要点（设计文档 7 节）：
//! - 全局 tokio runtime（`once_cell::Lazy`），每个 FFI 调用 `block_on`
//! - 句柄式：`aibridge_client_t` / `aibridge_stream_t`（opaque）
//! - 复杂 struct 走 JSON 字符串边界，二进制走 `aibridge_bytes_t`
//! - 错误：`aibridge_status_t` 返回码 + `aibridge_last_error()` 线程局部槽
//! - cbindgen 生成 `include/aibridge.h`
//!
//! # 安全保证
//! 所有 `extern "C"` 函数绝不 panic：内部用 `std::panic::catch_unwind`
//! 捕获 panic，转为 `AIBRIDGE_ERR_FFI` 错误码并写入 last_error。

mod error;
mod handle;
mod runtime;
mod stream;
mod string;

use crate::error::{AibridgeStatus, AIBRIDGE_OK};
use crate::handle::{aibridge_bytes_t, aibridge_client_t, aibridge_stream_t, AibridgeClient};
use crate::runtime::block_on;
use crate::string::{alloc_cstring, cstr_to_string};
use aibridge_core::client::Client;
use aibridge_core::config::ClientOptions;
use aibridge_core::model::{ChatRequest, SpeechRequest};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

// —— 重新导出供 cbindgen 生成头文件所需的符号 ——
pub use crate::string::{aibridge_bytes_free, aibridge_string_free};

/// 错误码类型别名（供 cbindgen 导出为 typedef）
#[allow(non_camel_case_types)]
pub type aibridge_status_t = AibridgeStatus;

// =========================================================================
// 生命周期：client new / start / destroy
// =========================================================================

/// 创建客户端
///
/// `provider` 为 Provider 类型（如 "openai"、"agnes"），UTF-8 C 字符串。
/// `config_json` 为 `ClientOptions` 的 JSON 序列化字符串（可为 `nullptr`，
/// 等价于默认配置）。
///
/// 成功返回 client 指针；失败返回 `nullptr`（错误写入 `aibridge_last_error`）。
///
/// # Safety
/// - `provider` 必须是合法的 NUL 结尾 UTF-8 C 字符串
/// - `config_json` 可为 `nullptr` 或合法 JSON C 字符串
/// - 返回的指针需通过 [`aibridge_client_destroy`] 释放
#[no_mangle]
pub unsafe extern "C" fn aibridge_client_new(
    provider: *const c_char,
    config_json: *const c_char,
) -> *mut aibridge_client_t {
    let result = catch_unwind(AssertUnwindSafe(|| client_new_impl(provider, config_json)));
    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            error::set_ffi_error_simple("aibridge_client_new 内部 panic");
            ptr::null_mut()
        }
    }
}

/// `aibridge_client_new` 的实现
fn client_new_impl(provider: *const c_char, config_json: *const c_char) -> *mut aibridge_client_t {
    let provider_str = match cstr_to_string(provider) {
        Some(Ok(s)) => s,
        Some(Err(_)) => {
            error::set_ffi_error_simple("provider 不是合法 UTF-8");
            return ptr::null_mut();
        }
        None => {
            error::set_ffi_error_simple("provider 为空指针");
            return ptr::null_mut();
        }
    };

    // config_json 可为空（用默认 ClientOptions）
    let opts: ClientOptions = if config_json.is_null() {
        ClientOptions::default()
    } else {
        match cstr_to_string(config_json) {
            Some(Ok(s)) => match serde_json::from_str::<ClientOptions>(&s) {
                Ok(o) => o,
                Err(e) => {
                    error::set_ffi_error(
                        "config_json 反序列化 ClientOptions 失败",
                        serde_json::json!({ "error": e.to_string() }),
                    );
                    return ptr::null_mut();
                }
            },
            Some(Err(_)) => {
                error::set_ffi_error_simple("config_json 不是合法 UTF-8");
                return ptr::null_mut();
            }
            None => ClientOptions::default(),
        }
    };

    match Client::new(&provider_str, opts) {
        Ok(client) => {
            error::clear_last_error();
            let handle = AibridgeClient::new(client);
            Box::into_raw(Box::new(handle))
        }
        Err(e) => {
            error::set_last_error(&e);
            ptr::null_mut()
        }
    }
}

/// 启动客户端（初始化适配器）
///
/// 返回 0 成功；负数为错误码（详见 `aibridge_last_error`）。
///
/// # Safety
/// `client` 必须是 [`aibridge_client_new`] 返回的有效指针。
#[no_mangle]
pub unsafe extern "C" fn aibridge_client_start(
    client: *mut aibridge_client_t,
) -> aibridge_status_t {
    catch_unwind_status(AssertUnwindSafe(|| {
        if client.is_null() {
            return error::set_ffi_error_simple("client 句柄为空");
        }
        // SAFETY: 调用方保证 client 来自 aibridge_client_new
        let handle: &AibridgeClient = &*client;
        let inner = handle.arc();
        let result = block_on(async {
            let mut guard = inner.lock().await;
            guard.start().await
        });
        match result {
            Ok(()) => {
                error::clear_last_error();
                AIBRIDGE_OK
            }
            Err(e) => error::set_last_error(&e),
        }
    }))
}

/// 释放客户端句柄
///
/// # Safety
/// `client` 必须是 [`aibridge_client_new`] 返回的指针，且只能释放一次。
/// 传 `nullptr` 是安全的 no-op。
#[no_mangle]
pub unsafe extern "C" fn aibridge_client_destroy(client: *mut aibridge_client_t) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        handle::destroy_client_impl(client);
    }));
}

// =========================================================================
// 阻塞式调用：chat / speech
// =========================================================================

/// 文本对话（阻塞）
///
/// `request_json` 为 `ChatRequest` 的 JSON 序列化字符串。
/// 成功时 `*out_response_json` 写入 `ChatCompletion` 的 JSON，调用方需通过
/// [`aibridge_string_free`] 释放。
///
/// # Safety
/// - `client` 必须来自 [`aibridge_client_new`]
/// - `request_json` 必须是合法 JSON C 字符串
/// - `out_response_json` 必须指向可写的 `*mut c_char` 槽位
#[no_mangle]
pub unsafe extern "C" fn aibridge_client_chat(
    client: *mut aibridge_client_t,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
) -> aibridge_status_t {
    catch_unwind_status(AssertUnwindSafe(|| {
        chat_impl(client, request_json, out_response_json)
    }))
}

/// `aibridge_client_chat` 的实现
///
/// # Safety
/// `client` 必须来自 [`aibridge_client_new`]（可为 null，内部校验）。
unsafe fn chat_impl(
    client: *mut aibridge_client_t,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
) -> aibridge_status_t {
    if client.is_null() {
        return error::set_ffi_error_simple("client 句柄为空");
    }
    if out_response_json.is_null() {
        return error::set_ffi_error_simple("out_response_json 输出指针为空");
    }
    let req: ChatRequest = match parse_request_json(request_json, "ChatRequest") {
        Ok(r) => r,
        Err(status) => return status,
    };

    // SAFETY: 调用方保证 client 来自 aibridge_client_new
    let handle: &AibridgeClient = &*client;
    let inner = handle.arc();
    let result = block_on(async {
        let guard = inner.lock().await;
        guard.chat(req).await
    });

    match result {
        Ok(completion) => match serde_json::to_string(&completion) {
            Ok(json) => {
                let ptr = alloc_cstring(json);
                if ptr.is_null() {
                    return error::set_ffi_error_simple("响应 JSON 分配失败");
                }
                *out_response_json = ptr;
                error::clear_last_error();
                AIBRIDGE_OK
            }
            Err(e) => error::set_ffi_error(
                "序列化 ChatCompletion 失败",
                serde_json::json!({ "error": e.to_string() }),
            ),
        },
        Err(e) => error::set_last_error(&e),
    }
}

/// 文字转语音（阻塞，二进制载荷走 `aibridge_bytes_t`）
///
/// `request_json` 为 `SpeechRequest` 的 JSON 序列化字符串。
/// 成功时 `*out_audio` 写入二进制音频缓冲（`aibridge_bytes_t`），
/// `*out_meta_json` 写入 `SpeechResult`（不含 audio_data）的 JSON。
/// 两者均由调用方分别通过 [`aibridge_bytes_free`] / [`aibridge_string_free`] 释放。
///
/// 若 Provider 仅返回 `audio_base64`/`audio_url` 而无二进制数据，
/// `*out_audio` 将为 `nullptr`（meta_json 仍写入）。
///
/// # Safety
/// - `client` / `request_json` / `out_audio` / `out_meta_json` 均需合法非空
#[no_mangle]
pub unsafe extern "C" fn aibridge_client_speech(
    client: *mut aibridge_client_t,
    request_json: *const c_char,
    out_audio: *mut *mut aibridge_bytes_t,
    out_meta_json: *mut *mut c_char,
) -> aibridge_status_t {
    catch_unwind_status(AssertUnwindSafe(|| {
        speech_impl(client, request_json, out_audio, out_meta_json)
    }))
}

/// `aibridge_client_speech` 的实现
///
/// # Safety
/// `client` 必须来自 [`aibridge_client_new`]（可为 null，内部校验）。
unsafe fn speech_impl(
    client: *mut aibridge_client_t,
    request_json: *const c_char,
    out_audio: *mut *mut aibridge_bytes_t,
    out_meta_json: *mut *mut c_char,
) -> aibridge_status_t {
    if client.is_null() {
        return error::set_ffi_error_simple("client 句柄为空");
    }
    if out_audio.is_null() {
        return error::set_ffi_error_simple("out_audio 输出指针为空");
    }
    if out_meta_json.is_null() {
        return error::set_ffi_error_simple("out_meta_json 输出指针为空");
    }
    let req: SpeechRequest = match parse_request_json(request_json, "SpeechRequest") {
        Ok(r) => r,
        Err(status) => return status,
    };

    // SAFETY: 调用方保证 client 来自 aibridge_client_new
    let handle: &AibridgeClient = &*client;
    let inner = handle.arc();
    let result = block_on(async {
        let guard = inner.lock().await;
        guard.speech(req).await
    });

    match result {
        Ok(speech) => {
            // 写入二进制音频（若有）
            let audio_bytes = speech.get_audio_bytes();
            let audio_ptr = match audio_bytes {
                Some(data) if !data.is_empty() => {
                    Box::into_raw(Box::new(aibridge_bytes_t::from_vec(data)))
                }
                _ => ptr::null_mut(),
            };
            *out_audio = audio_ptr;

            // 写入 meta JSON（SpeechResult，audio_data 被 serde skip）
            match serde_json::to_string(&speech) {
                Ok(json) => {
                    let ptr = alloc_cstring(json);
                    if ptr.is_null() {
                        // audio 已分配，需回滚释放避免泄漏
                        if !audio_ptr.is_null() {
                            handle::destroy_bytes_ptr(audio_ptr);
                        }
                        return error::set_ffi_error_simple("meta JSON 分配失败");
                    }
                    *out_meta_json = ptr;
                    error::clear_last_error();
                    AIBRIDGE_OK
                }
                Err(e) => {
                    if !audio_ptr.is_null() {
                        handle::destroy_bytes_ptr(audio_ptr);
                    }
                    error::set_ffi_error(
                        "序列化 SpeechResult 失败",
                        serde_json::json!({ "error": e.to_string() }),
                    )
                }
            }
        }
        Err(e) => error::set_last_error(&e),
    }
}

// =========================================================================
// 流式：chat_stream / stream_next / stream_destroy
// =========================================================================

/// 流式文本对话（创建 stream 句柄）
///
/// `request_json` 为 `ChatRequest` 的 JSON 序列化字符串。
/// 成功时 `*out_stream` 写入 stream 句柄，调用方通过
/// [`aibridge_stream_next`] 拉取 chunk，最后用 [`aibridge_stream_destroy`] 释放。
///
/// # Safety
/// - `client` / `request_json` / `out_stream` 均需合法非空
#[no_mangle]
pub unsafe extern "C" fn aibridge_client_chat_stream(
    client: *mut aibridge_client_t,
    request_json: *const c_char,
    out_stream: *mut *mut aibridge_stream_t,
) -> aibridge_status_t {
    catch_unwind_status(AssertUnwindSafe(|| {
        chat_stream_impl(client, request_json, out_stream)
    }))
}

/// `aibridge_client_chat_stream` 的实现
///
/// # Safety
/// `client` 必须来自 [`aibridge_client_new`]（可为 null，内部校验）。
unsafe fn chat_stream_impl(
    client: *mut aibridge_client_t,
    request_json: *const c_char,
    out_stream: *mut *mut aibridge_stream_t,
) -> aibridge_status_t {
    if client.is_null() {
        return error::set_ffi_error_simple("client 句柄为空");
    }
    if out_stream.is_null() {
        return error::set_ffi_error_simple("out_stream 输出指针为空");
    }
    let req: ChatRequest = match parse_request_json(request_json, "ChatRequest") {
        Ok(r) => r,
        Err(status) => return status,
    };

    // SAFETY: 调用方保证 client 来自 aibridge_client_new
    let handle: &AibridgeClient = &*client;
    let inner = handle.arc();
    let result = block_on(async {
        let guard = inner.lock().await;
        guard.chat_stream(req).await
    });

    match result {
        Ok(stream) => {
            let stream_handle = crate::handle::AibridgeStream::new(stream);
            *out_stream = Box::into_raw(Box::new(stream_handle));
            error::clear_last_error();
            AIBRIDGE_OK
        }
        Err(e) => error::set_last_error(&e),
    }
}

/// 拉取下一个流式 chunk（阻塞）
///
/// 返回：
/// - `0`（`AIBRIDGE_STREAM_CHUNK`）：拉到一个 chunk，`*out_chunk_json` 写入 JSON
/// - `1`（`AIBRIDGE_STREAM_END`）：流正常结束
/// - 负数：错误（`aibridge_last_error` 已写入）
///
/// # Safety
/// - `stream` 必须来自 [`aibridge_client_chat_stream`]
/// - `out_chunk_json` 必须指向可写的 `*mut c_char` 槽位
#[no_mangle]
pub unsafe extern "C" fn aibridge_stream_next(
    stream: *mut aibridge_stream_t,
    out_chunk_json: *mut *mut c_char,
) -> aibridge_status_t {
    catch_unwind_status(AssertUnwindSafe(|| {
        stream::stream_next_impl(stream, out_chunk_json)
    }))
}

/// 释放 stream 句柄（触发 Rust drop → tokio task abort）
///
/// # Safety
/// `stream` 必须来自 [`aibridge_client_chat_stream`]，且只能释放一次。
/// 传 `nullptr` 是安全的 no-op。
#[no_mangle]
pub unsafe extern "C" fn aibridge_stream_destroy(stream: *mut aibridge_stream_t) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        handle::destroy_stream_impl(stream);
    }));
}

// =========================================================================
// 错误查询
// =========================================================================

/// 读取当前线程的 last_error（JSON 字符串）
///
/// 返回指向线程局部 `CString` 内部缓冲的指针，**调用方不应释放**。
/// 若当前线程无错误，返回 `nullptr`。
///
/// 输出格式：`{"code":"...","message":"...","details":...,"retryable":bool}`
///
/// # Safety
/// 本函数本身无内存不安全操作（仅读取线程局部变量并返回裸指针），
/// 标 `unsafe extern "C"` 仅为符合 FFI 调用约定。调用方**不得**释放返回的指针，
/// 且需注意返回的指针仅在当前线程的下一次 FFI 调用前保证有效（thread_local 语义）。
#[no_mangle]
pub unsafe extern "C" fn aibridge_last_error() -> *const c_char {
    // 该函数本身不 panic（仅读 thread_local + 返回指针），但仍兜底
    let result = catch_unwind(AssertUnwindSafe(error::last_error_ptr));
    match result {
        Ok(ptr) => ptr,
        Err(_) => ptr::null(),
    }
}

// =========================================================================
// 内部辅助
// =========================================================================

/// 解析请求 JSON 字符串为目标类型
///
/// 返回 `Ok(parsed)` 或 `Err(ffi_status)`（last_error 已写入）。
fn parse_request_json<T: serde::de::DeserializeOwned>(
    request_json: *const c_char,
    type_name: &str,
) -> Result<T, AibridgeStatus> {
    let json_str = match cstr_to_string(request_json) {
        Some(Ok(s)) => s,
        Some(Err(_)) => {
            return Err(error::set_ffi_error(
                &format!("{type_name} 请求 JSON 不是合法 UTF-8"),
                serde_json::Value::Null,
            ));
        }
        None => {
            return Err(error::set_ffi_error(
                &format!("{type_name} 请求 JSON 为空指针"),
                serde_json::Value::Null,
            ));
        }
    };
    match serde_json::from_str::<T>(&json_str) {
        Ok(v) => Ok(v),
        Err(e) => Err(error::set_ffi_error(
            &format!("{type_name} 请求 JSON 反序列化失败"),
            serde_json::json!({ "error": e.to_string() }),
        )),
    }
}

/// 包裹 catch_unwind：把 panic 转为 `AIBRIDGE_ERR_FFI` 并写入 last_error
fn catch_unwind_status<F>(f: F) -> aibridge_status_t
where
    F: FnOnce() -> aibridge_status_t,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(status) => status,
        Err(_) => error::set_ffi_error_simple("FFI 调用内部 panic"),
    }
}

// =========================================================================
// 单元测试
// =========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AIBRIDGE_ERR_FFI;
    use std::ffi::CString;

    /// 构造 C 字符串指针的辅助
    fn cstr(s: &str) -> *const c_char {
        CString::new(s).unwrap().into_raw()
    }

    /// 释放 `cstr` 分配的指针
    unsafe fn free_cstr(p: *const c_char) {
        if !p.is_null() {
            drop(CString::from_raw(p as *mut c_char));
        }
    }

    #[test]
    fn client_new_with_null_provider_returns_null() {
        unsafe {
            let ptr = aibridge_client_new(std::ptr::null(), std::ptr::null());
            assert!(ptr.is_null());
            assert!(!aibridge_last_error().is_null());
            error::clear_last_error();
        }
    }

    #[test]
    fn client_new_with_invalid_config_json_returns_null() {
        unsafe {
            let provider = cstr("openai");
            let bad = cstr("{not json}");
            let ptr = aibridge_client_new(provider, bad);
            assert!(ptr.is_null());
            let err_ptr = aibridge_last_error();
            assert!(!err_ptr.is_null());
            let json = crate::string::cstr_to_string_unchecked(err_ptr);
            assert!(json.contains("config_json"));
            error::clear_last_error();
            free_cstr(provider);
            free_cstr(bad);
        }
    }

    #[test]
    fn client_new_with_unknown_provider_returns_null_and_provider_not_found() {
        // 阶段 0.4：工厂占位，任何 provider 都返 ProviderNotFound
        unsafe {
            let provider = cstr("nonexistent_provider");
            let config = cstr(r#"{"api_key":"sk-test","timeout":60}"#);
            let ptr = aibridge_client_new(provider, config);
            assert!(ptr.is_null());
            let err_ptr = aibridge_last_error();
            assert!(!err_ptr.is_null());
            let json = crate::string::cstr_to_string_unchecked(err_ptr);
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(v["code"], "provider_not_found");
            assert_eq!(v["retryable"], false);
            error::clear_last_error();
            free_cstr(provider);
            free_cstr(config);
        }
    }

    #[test]
    fn client_new_missing_api_key_returns_validation_error() {
        // openai 缺 api_key：core 会返 ValidationError（validate 在 create_adapter 前）
        unsafe {
            let provider = cstr("openai");
            let ptr = aibridge_client_new(provider, std::ptr::null());
            assert!(ptr.is_null());
            let err_ptr = aibridge_last_error();
            assert!(!err_ptr.is_null());
            let json = crate::string::cstr_to_string_unchecked(err_ptr);
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(v["code"], "validation_error");
            error::clear_last_error();
            free_cstr(provider);
        }
    }

    #[test]
    fn chat_with_null_client_returns_ffi_error() {
        unsafe {
            let req = cstr(r#"{"model":"gpt-4o","messages":[]}"#);
            let mut out: *mut c_char = std::ptr::null_mut();
            let status = aibridge_client_chat(std::ptr::null_mut(), req, &mut out);
            assert_eq!(status, AIBRIDGE_ERR_FFI);
            assert!(out.is_null());
            error::clear_last_error();
            free_cstr(req);
        }
    }

    #[test]
    fn chat_with_null_out_returns_ffi_error() {
        unsafe {
            let req = cstr(r#"{"model":"gpt-4o","messages":[]}"#);
            // 构造非空但无效的 client 指针（仅用于触发 out 校验，不实际解引用）
            let fake_client = 0x1usize as *mut aibridge_client_t;
            let status = aibridge_client_chat(fake_client, req, std::ptr::null_mut());
            assert_eq!(status, AIBRIDGE_ERR_FFI);
            error::clear_last_error();
            free_cstr(req);
        }
    }

    #[test]
    fn chat_with_invalid_request_json_returns_ffi_error() {
        unsafe {
            // 用非空 client 指针，但 request_json 非法
            let fake_client = 0x1usize as *mut aibridge_client_t;
            let bad = cstr("{bad json}");
            let mut out: *mut c_char = std::ptr::null_mut();
            let status = aibridge_client_chat(fake_client, bad, &mut out);
            assert_eq!(status, AIBRIDGE_ERR_FFI);
            let err_ptr = aibridge_last_error();
            let json = crate::string::cstr_to_string_unchecked(err_ptr);
            assert!(json.contains("ChatRequest"));
            error::clear_last_error();
            free_cstr(bad);
        }
    }

    #[test]
    fn speech_with_null_client_returns_ffi_error() {
        unsafe {
            let req = cstr(r#"{"model":"tts-1","input":"hi","voice":"alloy"}"#);
            let mut out_audio: *mut aibridge_bytes_t = std::ptr::null_mut();
            let mut out_meta: *mut c_char = std::ptr::null_mut();
            let status =
                aibridge_client_speech(std::ptr::null_mut(), req, &mut out_audio, &mut out_meta);
            assert_eq!(status, AIBRIDGE_ERR_FFI);
            assert!(out_audio.is_null());
            assert!(out_meta.is_null());
            error::clear_last_error();
            free_cstr(req);
        }
    }

    #[test]
    fn speech_with_null_out_pointers_returns_ffi_error() {
        unsafe {
            let req = cstr(r#"{"model":"tts-1","input":"hi","voice":"alloy"}"#);
            let fake_client = 0x1usize as *mut aibridge_client_t;
            // out_audio 为空
            let mut out_meta: *mut c_char = std::ptr::null_mut();
            let status =
                aibridge_client_speech(fake_client, req, std::ptr::null_mut(), &mut out_meta);
            assert_eq!(status, AIBRIDGE_ERR_FFI);
            error::clear_last_error();
            free_cstr(req);
        }
    }

    #[test]
    fn chat_stream_with_null_client_returns_ffi_error() {
        unsafe {
            let req = cstr(r#"{"model":"gpt-4o","messages":[]}"#);
            let mut out_stream: *mut aibridge_stream_t = std::ptr::null_mut();
            let status = aibridge_client_chat_stream(std::ptr::null_mut(), req, &mut out_stream);
            assert_eq!(status, AIBRIDGE_ERR_FFI);
            assert!(out_stream.is_null());
            error::clear_last_error();
            free_cstr(req);
        }
    }

    #[test]
    fn stream_next_null_stream_returns_ffi_error() {
        unsafe {
            let mut out: *mut c_char = std::ptr::null_mut();
            let status = aibridge_stream_next(std::ptr::null_mut(), &mut out);
            assert_eq!(status, AIBRIDGE_ERR_FFI);
            error::clear_last_error();
        }
    }

    #[test]
    fn stream_destroy_null_is_noop() {
        unsafe {
            aibridge_stream_destroy(std::ptr::null_mut());
        }
    }

    #[test]
    fn client_destroy_null_is_noop() {
        unsafe {
            aibridge_client_destroy(std::ptr::null_mut());
        }
    }

    #[test]
    fn last_error_returns_null_after_clear() {
        error::clear_last_error();
        unsafe {
            assert!(aibridge_last_error().is_null());
        }
    }

    #[test]
    fn parse_request_json_handles_null_pointer() {
        let status = parse_request_json::<ChatRequest>(std::ptr::null(), "ChatRequest");
        assert!(status.is_err());
        assert_eq!(status.unwrap_err(), AIBRIDGE_ERR_FFI);
        error::clear_last_error();
    }

    #[test]
    fn parse_request_json_handles_valid_json() {
        let json = r#"{"model":"gpt-4o","messages":[]}"#;
        let c = CString::new(json).unwrap();
        let result = parse_request_json::<ChatRequest>(c.as_ptr(), "ChatRequest");
        assert!(result.is_ok());
        let req = result.unwrap();
        assert_eq!(req.model, "gpt-4o");
    }

    #[test]
    fn alloc_cstring_roundtrip_via_string_free() {
        let ptr = alloc_cstring("test".to_string());
        assert!(!ptr.is_null());
        unsafe {
            aibridge_string_free(ptr);
        }
    }

    #[test]
    fn catch_unwind_status_converts_panic_to_ffi_error() {
        let status = catch_unwind_status(|| panic!("boom"));
        assert_eq!(status, AIBRIDGE_ERR_FFI);
        error::clear_last_error();
    }
}
