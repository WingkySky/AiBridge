//! C 字符串与二进制辅助函数
//!
//! 提供跨 FFI 边界的字符串/字节传递与释放：
//! - [`aibridge_string_free`]：释放 Rust 分配的 `char*`
//! - [`aibridge_bytes_free`]：释放 Rust 分配的 [`aibridge_bytes_t`]
//! - [`cstr_to_string`] / [`cstr_to_string_unchecked`]：从 C 指针构造 Rust String
//! - [`alloc_cstring`]：把 Rust String 装进 CString 并返回裸指针（调用方负责释放）
//!
//! 设计要点（设计文档 7 节）：Rust 分配的 `char*` / `aibridge_bytes_t*`
//! 必须由调用方通过对应的 `_free` 函数释放。各语言绑定用 RAII 封装。

use crate::handle::aibridge_bytes_t;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// 把 C `*const c_char` 转为 Rust `String`
///
/// 空指针返回 `None`；非 UTF-8 返回 `Err`。
pub fn cstr_to_string(ptr: *const c_char) -> Option<Result<String, std::str::Utf8Error>> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: 调用方保证 ptr 指向合法的 NUL 结尾 C 字符串
    let cstr = unsafe { CStr::from_ptr(ptr) };
    Some(cstr.to_str().map(|s| s.to_string()))
}

/// 把 C `*const c_char` 转为 Rust `String`（不检查 UTF-8，仅供内部测试用）
///
/// # Safety
/// `ptr` 必须指向合法的 NUL 结尾 C 字符串且为有效 UTF-8。
#[allow(dead_code)]
pub unsafe fn cstr_to_string_unchecked(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

/// 把 Rust `String` 装进 `CString` 并返回裸指针
///
/// 调用方需通过 [`aibridge_string_free`] 释放。返回 `nullptr` 表示分配失败
/// 或字符串含内嵌 NUL。
pub fn alloc_cstring(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// 释放 Rust 分配的 C 字符串
///
/// # Safety
/// `ptr` 必须是由 [`alloc_cstring`]（或 `CString::into_raw`）分配的指针，
/// 且只能释放一次。传 `nullptr` 是安全的 no-op。
///
/// 对应设计文档的 `aibridge_string_free`。
#[no_mangle]
pub unsafe extern "C" fn aibridge_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        // SAFETY: 调用方保证 ptr 来自 CString::into_raw，且未释放过
        drop(CString::from_raw(ptr));
    }
}

/// 释放 Rust 分配的二进制缓冲
///
/// # Safety
/// `ptr` 必须是指向由 FFI 函数（如 `aibridge_client_speech`）输出的
/// `aibridge_bytes_t*` 指针，且只能释放一次。传 `nullptr` 是安全的 no-op。
///
/// 对应设计文档的 `aibridge_bytes_free`。
#[no_mangle]
pub unsafe extern "C" fn aibridge_bytes_free(ptr: *mut aibridge_bytes_t) {
    if !ptr.is_null() {
        // SAFETY: 调用方保证 ptr 来自 Box::into_raw<aibridge_bytes_t>，且未释放过
        let boxed = Box::from_raw(ptr);
        // 还原并释放内部 [u8] 缓冲（from_vec 时用 Box<[u8]> 装箱）
        if !boxed.ptr.is_null() && boxed.len > 0 {
            let slice = std::slice::from_raw_parts_mut(boxed.ptr as *mut u8, boxed.len);
            drop(Box::from_raw(slice));
        }
        // Box<aibridge_bytes_t> 随作用域结束 drop
        drop(boxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_free_cstring_roundtrip() {
        let ptr = alloc_cstring("你好, aibridge".to_string());
        assert!(!ptr.is_null());
        let s = unsafe { cstr_to_string_unchecked(ptr) };
        assert_eq!(s, "你好, aibridge");
        unsafe { aibridge_string_free(ptr) };
    }

    #[test]
    fn alloc_cstring_with_embedded_nul_returns_null() {
        // 含内嵌 NUL 的字符串无法构造 CString
        let ptr = alloc_cstring("a\0b".to_string());
        assert!(ptr.is_null());
    }

    #[test]
    fn string_free_null_is_noop() {
        unsafe { aibridge_string_free(std::ptr::null_mut()) };
    }

    #[test]
    fn cstr_to_string_null_returns_none() {
        assert!(cstr_to_string(std::ptr::null()).is_none());
    }

    #[test]
    fn bytes_free_null_is_noop() {
        unsafe { aibridge_bytes_free(std::ptr::null_mut()) };
    }

    #[test]
    fn bytes_alloc_and_free_roundtrip() {
        // 构造一个 aibridge_bytes_t，释放应不泄漏/不崩溃
        let data = vec![1u8, 2, 3, 4, 5];
        let b = Box::new(aibridge_bytes_t::from_vec(data));
        let ptr = Box::into_raw(b);
        unsafe { aibridge_bytes_free(ptr) };
    }
}
