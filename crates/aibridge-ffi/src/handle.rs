//! opaque 句柄定义
//!
//! 对应设计文档 7 节的句柄式生命周期：
//! - [`AibridgeClient`]：包裹 `aibridge_core::Client`，用 `Arc<Mutex<...>>` 处理
//!   core 的 `start`/`close` 是 `&mut self`
//! - [`AibridgeStream`]：持有流式 `ChatStream` + tokio task handle，`drop` 触发 abort
//! - [`aibridge_bytes_t`]：二进制缓冲（`ptr` + `len`），跨 FFI 传递音频等载荷
//!
//! C 侧只看到 opaque 指针 `aibridge_client_t*` / `aibridge_stream_t*`，
//! 具体布局对调用方不可见。

use aibridge_core::adapter::ChatStream;
use aibridge_core::client::Client;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// 二进制缓冲结构（`#[repr(C)]` 保证 C 侧布局）
///
/// 对应设计文档的 `aibridge_bytes_t`。`ptr` 指向 Rust 分配的字节缓冲，
/// `len` 为长度。调用方通过 [`crate::string::aibridge_bytes_free`] 释放。
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct aibridge_bytes_t {
    /// 指向字节数据的指针（Rust 分配）
    pub ptr: *const u8,
    /// 字节数据长度
    pub len: usize,
}

impl aibridge_bytes_t {
    /// 从 `Vec<u8>` 构造（消耗 vec，将其底层缓冲转为裸指针）
    pub fn from_vec(data: Vec<u8>) -> Self {
        // 用 Box<[u8]> 持有，drop 时释放底层缓冲
        // 注意：into_raw 时必须能还原出同样布局；这里用 Box::into_raw([u8]) 风格
        let len = data.len();
        let boxed: Box<[u8]> = data.into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *const u8;
        Self { ptr, len }
    }
}

// aibridge_bytes_t 持有裸指针，但实际所有权通过 Box<[u8]> 管理，
// aibridge_bytes_free 时用 Box::from_raw 还原。Send/Sync 不必要（FFI 单线程使用）。

/// Client 句柄
///
/// 内部用 `Arc<tokio::sync::Mutex<Client>>`：
/// - `Arc` 允许 stream task 持有 client 引用
/// - `tokio::sync::Mutex` 处理 core 的 `start`/`close` 是 `&mut self`，
///   且其 guard 可跨 `.await`（避免 `std::sync::Mutex` 跨 await 的死锁/panic）
pub struct AibridgeClient {
    /// 内部 core Client（互斥访问）
    pub(crate) inner: Arc<Mutex<Client>>,
}

impl AibridgeClient {
    /// 从 core Client 构造句柄
    pub fn new(client: Client) -> Self {
        Self {
            inner: Arc::new(Mutex::new(client)),
        }
    }

    /// 获取内部 Arc 引用（供 stream task 复用）
    pub fn arc(&self) -> Arc<Mutex<Client>> {
        Arc::clone(&self.inner)
    }
}

/// Stream 句柄
///
/// 持有流式 `ChatStream` 与一个可选的 tokio task handle。
/// `drop` 时若 task 存在则 abort，保证流式任务不会泄漏。
///
/// 注意：`aibridge_stream_next` 在调用方线程串行拉取（各语言绑定负责同步）。
pub struct AibridgeStream {
    /// 流式 chunk 迭代器（ChatStream = BoxStream<Result<Chunk>>）
    pub(crate) stream: Option<ChatStream>,
    /// 可选的后台 task handle（drop 时 abort）
    pub(crate) task: Option<JoinHandle<()>>,
    /// 已结束标志（避免重复拉取）
    pub(crate) ended: bool,
}

impl AibridgeStream {
    /// 构造 stream 句柄
    pub fn new(stream: ChatStream) -> Self {
        Self {
            stream: Some(stream),
            task: None,
            ended: false,
        }
    }

    /// 构造带后台 task 的 stream 句柄
    pub fn with_task(stream: ChatStream, task: JoinHandle<()>) -> Self {
        Self {
            stream: Some(stream),
            task: Some(task),
            ended: false,
        }
    }
}

impl Drop for AibridgeStream {
    fn drop(&mut self) {
        // 先 abort task（若存在），再 drop stream
        if let Some(task) = self.task.take() {
            task.abort();
        }
        if let Some(stream) = self.stream.take() {
            // 主动 drop 流，释放底层资源（BoxStream 无 close 方法，直接 drop 即可）
            drop(stream);
        }
    }
}

// FFI 导出的 opaque 类型别名（C 侧只见指针）
/// `aibridge_client_t` opaque 类型
#[allow(non_camel_case_types)]
pub type aibridge_client_t = AibridgeClient;
/// `aibridge_stream_t` opaque 类型
#[allow(non_camel_case_types)]
pub type aibridge_stream_t = AibridgeStream;

/// 释放 client 句柄
///
/// # Safety
/// `ptr` 必须是 [`aibridge_client_new`] 返回的指针，且只能释放一次。
/// 传 `nullptr` 是安全的 no-op。
unsafe fn drop_client(ptr: *mut aibridge_client_t) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// C 侧调用的 client 释放入口（在 lib.rs 重新导出为 #[no_mangle]）
///
/// 这里单独定义便于内部测试直接调用。
pub(crate) unsafe fn destroy_client_impl(ptr: *mut aibridge_client_t) {
    drop_client(ptr);
}

/// C 侧调用的 stream 释放入口
pub(crate) unsafe fn destroy_stream_impl(ptr: *mut aibridge_stream_t) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// 释放 `aibridge_bytes_t` 指针（内部辅助，供错误回滚用）
///
/// # Safety
/// `ptr` 必须是 `Box::into_raw(Box::new(aibridge_bytes_t::from_vec(...)))` 产生的指针。
pub(crate) unsafe fn destroy_bytes_ptr(ptr: *mut aibridge_bytes_t) {
    if !ptr.is_null() {
        let boxed = Box::from_raw(ptr);
        // 还原底层 [u8] 并释放
        if !boxed.ptr.is_null() && boxed.len > 0 {
            let slice = std::slice::from_raw_parts_mut(boxed.ptr as *mut u8, boxed.len);
            drop(Box::from_raw(slice));
        }
        drop(boxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream::StreamExt;

    #[test]
    fn aibridge_bytes_from_vec_roundtrip() {
        let b = aibridge_bytes_t::from_vec(vec![10, 20, 30]);
        assert_eq!(b.len, 3);
        unsafe {
            assert_eq!(*b.ptr, 10);
            assert_eq!(*b.ptr.add(2), 30);
        }
        // 通过 aibridge_bytes_free 释放（验证完整生命周期）
        unsafe { crate::string::aibridge_bytes_free(Box::into_raw(Box::new(b))) };
    }

    #[test]
    fn destroy_null_pointers_are_noop() {
        unsafe {
            destroy_client_impl(std::ptr::null_mut());
            destroy_stream_impl(std::ptr::null_mut());
        }
    }

    #[test]
    #[allow(clippy::async_yields_async)]
    fn stream_drop_aborts_task() {
        // 构造一个空 stream + 一个会一直挂起的 task，drop 后 task 应被 abort
        use futures::stream;
        let s = stream::iter(vec![]).boxed();
        let handle = crate::runtime::block_on(async {
            tokio::spawn(async {
                // 永久挂起
                std::future::pending::<()>().await;
            })
        });
        let mut stream_handle = AibridgeStream::with_task(s, handle);
        // 手动取走 stream 后 drop（触发 task abort）
        stream_handle.stream.take();
        drop(stream_handle);
        // task 已 abort，无法 join 成功（这里仅验证不 panic）
    }
}
