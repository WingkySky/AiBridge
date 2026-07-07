//! AIBridge C ABI - FFI 层
//!
//! 暴露 C ABI 供 Go/JVM/.NET 调用。
//! Python/JS 通过 aibridge-python / aibridge-node 直连 aibridge-core，不走本层。
//!
//! 设计要点（阶段 0.5 实现）：
//! - 全局 tokio runtime（once_cell::Lazy），每个 FFI 调用 block_on
//! - 句柄式：aibridge_client_t / aibridge_stream_t（opaque）
//! - 复杂 struct 走 JSON 字符串边界，二进制走 aibridge_bytes_t
//! - 错误：aibridge_status_t 返回码 + aibridge_last_error() 线程局部槽
//! - cbindgen 生成 include/aibridge.h
