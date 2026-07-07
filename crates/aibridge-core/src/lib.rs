//! AIBridge 核心 - 多模态 AI 统一接口
//!
//! 一套 API 调用所有 AI 模型（chat / image / video / TTS / ASR / embed）。
//! 本 crate 是纯 Rust 逻辑核心，不含 FFI 污染。
//! Python/JS 通过 aibridge-python / aibridge-node 直连本 crate；
//! Go/JVM/.NET 通过 aibridge-ffi 的 C ABI 间接调用。
//!
//! 对应 Python v1 (agn-sdk) 的 agn/ 目录，五层架构保持一致：
//! API 层(client) → 路由层(router) → 适配器层(adapter) → 核心层(http/retry/error/config) → 模型层(model)

// 模块声明（阶段 0.2–0.4）
pub mod adapter;
pub mod adapters;
pub mod client;
pub mod config;
pub mod error;
pub mod http;
pub mod model;
pub mod retry;
pub mod router;
pub mod util;

/// crate 版本号
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
