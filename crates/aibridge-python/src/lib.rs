//! AIBridge Python 绑定（PyO3）
//!
//! 直连 aibridge-core，原生 asyncio 协程与 AsyncIterator 流式。
//! 由 maturin 构建为 PyPI 包 `aibridge`。
//! 阶段 0.6 填充 Client/chat/流式/错误映射。

use pyo3::prelude::*;

/// Python 模块入口：import aibridge
#[pymodule]
fn aibridge(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> {
    // 阶段 0.6 填充：注册 Client / Router / 错误类 / 数据模型
    Ok(())
}
