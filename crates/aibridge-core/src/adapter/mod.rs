//! 适配器层
//!
//! 定义 `Adapter` trait、`Capabilities` 枚举与适配器工厂。
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/base.py` + `agn/adapters/factory.py`。
//!
//! 设计要点（与设计文档 5.2 节一致）：
//! - `Adapter` trait 用 `#[async_trait]`，不支持的方法默认返 `UnsupportedCapability`
//! - 工厂用编译期显式 `match`（替代 Python 运行时注册），新增适配器=加 match 分支
//! - 具体适配器在阶段 1 起填充 `adapters/`

pub mod base;
pub mod factory;

pub use base::{Adapter, Capabilities, CapabilitySet, ChatStream};
pub use factory::create_adapter;
