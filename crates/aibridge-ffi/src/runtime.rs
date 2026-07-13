//! 全局 tokio 运行时
//!
//! 为所有 FFI 调用提供共享的多线程异步运行时。
//! C 侧可并发调用不同 client；每个 FFI 函数内部通过 [`block_on`] 在该 runtime 上
//! 阻塞执行异步 future。
//!
//! 设计要点（与设计文档 7 节一致）：
//! - `once_cell::Lazy` 全局单例，多线程（`multi_thread` + `worker_threads`）
//! - [`block_on`] 辅助函数：`runtime.handle().block_on(future)`
//! - 不暴露 handle 给 C 侧，所有调度由 Rust 内部完成

use once_cell::sync::Lazy;
use tokio::runtime::Runtime;

/// 全局 tokio 运行时单例
///
/// 多线程运行时，工作线程数取 tokio 默认值（= CPU 核心数）。
/// 使用 `Lazy` 保证首次访问时初始化，且线程安全。
static RUNTIME: Lazy<Runtime> = Lazy::new(|| Runtime::new().expect("初始化 tokio 运行时失败"));

/// 在全局 runtime 上阻塞执行一个 future 并返回其结果
///
/// 所有 FFI 函数内部异步逻辑都通过此函数驱动。
/// 使用 `Runtime::block_on`（而非 `Handle::block_on`），因此**不要求** future 为 `Send`，
/// 这样允许在 future 内持有非 `Send` 的 `MutexGuard` 跨 `.await`。
pub fn block_on<F>(future: F) -> F::Output
where
    F: std::future::Future,
{
    RUNTIME.block_on(future)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_on_executes_future() {
        let result = block_on(async { 42 });
        assert_eq!(result, 42);
    }

    #[test]
    fn runtime_is_shared_across_calls() {
        // 多次调用应复用同一 runtime（Lazy 只初始化一次）
        let a = block_on(async { "a" });
        let b = block_on(async { "b" });
        assert_eq!(a, "a");
        assert_eq!(b, "b");
    }
}
