//! 流式 chunk 拉取
//!
//! 对应设计文档 7 节的流式接口：
//! - [`stream_next_impl`]：阻塞拉取下一个 chunk
//!   - 返回 `AIBRIDGE_STREAM_CHUNK`(0) = 拉到一个 chunk（写入 out_chunk_json）
//!   - 返回 `AIBRIDGE_STREAM_END`(1) = 流正常结束
//!   - 返回负数 = 流错误（last_error 已写入）
//!
//! 实现要点：在全局 runtime 上 `block_on` 拉取 stream 的 next()，
//! 将 `ChatCompletionChunk` 序列化为 JSON 写入 out 指针。

use crate::error::{self, AIBRIDGE_STREAM_CHUNK, AIBRIDGE_STREAM_END};
use crate::handle::{aibridge_stream_t, AibridgeStream};
use crate::string::alloc_cstring;
use futures::stream::StreamExt;
use std::os::raw::c_char;

/// 阻塞拉取下一个流式 chunk
///
/// # 返回
/// - `AIBRIDGE_STREAM_CHUNK`(0)：成功拉取一个 chunk，`*out_chunk_json` 写入 JSON
/// - `AIBRIDGE_STREAM_END`(1)：流正常结束
/// - 负数：流错误（`aibridge_last_error` 已写入线程局部槽）
///
/// # Safety
/// - `stream` 必须是 [`crate::aibridge_client_chat_stream`] 返回的有效指针
/// - `out_chunk_json` 必须指向可写的 `*mut c_char` 槽位
/// - 输出的字符串由调用方通过 [`crate::aibridge_string_free`] 释放
pub(crate) unsafe fn stream_next_impl(
    stream: *mut aibridge_stream_t,
    out_chunk_json: *mut *mut c_char,
) -> i32 {
    // —— 参数校验 ——
    if stream.is_null() {
        return error::set_ffi_error_simple("stream 句柄为空");
    }
    if out_chunk_json.is_null() {
        return error::set_ffi_error_simple("out_chunk_json 输出指针为空");
    }

    // SAFETY: 调用方保证 stream 来自 chat_stream，且未被释放
    let stream_handle: &mut AibridgeStream = &mut *stream;

    // 已结束的流直接返回 EOF
    if stream_handle.ended {
        return AIBRIDGE_STREAM_END;
    }

    // 取出 stream 的 next（需要 &mut，stream 字段为 Option<ChatStream>）
    let mut stream_opt = stream_handle.stream.take();
    let next_result = match stream_opt.as_mut() {
        Some(s) => crate::runtime::block_on(s.next()),
        None => {
            // stream 已被取走（不应发生，但防御性处理）
            stream_handle.stream = stream_opt;
            stream_handle.ended = true;
            return AIBRIDGE_STREAM_END;
        }
    };

    match next_result {
        None => {
            // 流正常结束
            stream_handle.ended = true;
            stream_handle.stream = stream_opt;
            AIBRIDGE_STREAM_END
        }
        Some(Ok(chunk)) => {
            // 序列化 chunk 为 JSON
            match serde_json::to_string(&chunk) {
                Ok(json) => {
                    let ptr = alloc_cstring(json);
                    if ptr.is_null() {
                        stream_handle.stream = stream_opt;
                        return error::set_ffi_error_simple("chunk JSON 分配失败");
                    }
                    *out_chunk_json = ptr;
                    stream_handle.stream = stream_opt;
                    AIBRIDGE_STREAM_CHUNK
                }
                Err(e) => {
                    stream_handle.stream = stream_opt;
                    error::set_ffi_error(
                        "序列化 ChatCompletionChunk 失败",
                        serde_json::json!({ "error": e.to_string() }),
                    )
                }
            }
        }
        Some(Err(err)) => {
            // 流中产生错误：写入 last_error 并返回对应错误码
            stream_handle.ended = true;
            stream_handle.stream = stream_opt;
            error::set_last_error(&err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AIBRIDGE_ERR_FFI;
    use crate::handle::AibridgeStream;
    use aibridge_core::error::AibridgeError;
    use aibridge_core::model::ChatCompletionChunk;
    use futures::stream;
    use std::ffi::CString;

    /// 构造一个测试 stream 句柄：从 chunk JSON 数组生成
    fn make_test_stream(chunks: Vec<&str>) -> *mut aibridge_stream_t {
        let parsed: Vec<aibridge_core::error::Result<ChatCompletionChunk>> = chunks
            .into_iter()
            .map(|s| {
                serde_json::from_str::<ChatCompletionChunk>(s)
                    .map_err(|e| AibridgeError::validation(format!("测试 fixture 解析失败: {e}")))
            })
            .collect();
        let s = stream::iter(parsed).boxed();
        let handle = AibridgeStream::new(s);
        Box::into_raw(Box::new(handle))
    }

    fn free_stream(ptr: *mut aibridge_stream_t) {
        unsafe {
            crate::handle::destroy_stream_impl(ptr);
        }
    }

    #[test]
    fn next_returns_chunks_then_end() {
        // 用一个合法的 chunk JSON
        let chunk_json = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":0,"model":"m","choices":[]}"#;
        let ptr = make_test_stream(vec![chunk_json, chunk_json]);
        unsafe {
            let mut out: *mut c_char = std::ptr::null_mut();
            let s1 = stream_next_impl(ptr, &mut out);
            assert_eq!(s1, AIBRIDGE_STREAM_CHUNK);
            assert!(!out.is_null());
            let json = cstr_to_string_unchecked_internal(out);
            assert!(json.contains("chatcmpl-1"));
            crate::aibridge_string_free(out);

            let mut out2: *mut c_char = std::ptr::null_mut();
            let s2 = stream_next_impl(ptr, &mut out2);
            assert_eq!(s2, AIBRIDGE_STREAM_CHUNK);
            crate::aibridge_string_free(out2);

            // 第三次应返回 END
            let mut out3: *mut c_char = std::ptr::null_mut();
            let s3 = stream_next_impl(ptr, &mut out3);
            assert_eq!(s3, AIBRIDGE_STREAM_END);
            assert!(out3.is_null());
        }
        free_stream(ptr);
    }

    #[test]
    fn next_with_null_stream_returns_ffi_error() {
        unsafe {
            let mut out: *mut c_char = std::ptr::null_mut();
            let s = stream_next_impl(std::ptr::null_mut(), &mut out);
            assert_eq!(s, AIBRIDGE_ERR_FFI);
            // last_error 应已写入
            assert!(!error::last_error_ptr().is_null());
            error::clear_last_error();
        }
    }

    #[test]
    fn next_with_null_out_returns_ffi_error() {
        let ptr = make_test_stream(vec![]);
        unsafe {
            let s = stream_next_impl(ptr, std::ptr::null_mut());
            assert_eq!(s, AIBRIDGE_ERR_FFI);
            error::clear_last_error();
        }
        free_stream(ptr);
    }

    #[test]
    fn ended_stream_returns_eof_repeatedly() {
        let ptr = make_test_stream(vec![]);
        unsafe {
            let mut out: *mut c_char = std::ptr::null_mut();
            assert_eq!(stream_next_impl(ptr, &mut out), AIBRIDGE_STREAM_END);
            // 再次拉取仍返回 END
            assert_eq!(stream_next_impl(ptr, &mut out), AIBRIDGE_STREAM_END);
        }
        free_stream(ptr);
    }

    // 内部辅助：复用 string 模块的 unchecked 转换
    fn cstr_to_string_unchecked_internal(ptr: *mut c_char) -> String {
        unsafe { crate::string::cstr_to_string_unchecked(ptr) }
    }

    // 抑制未使用警告（CString 在某些测试路径用到）
    #[allow(dead_code)]
    fn _suppress(_c: CString) {}
}
