//! 具体 Provider 适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/*.py`（14 个适配器）。
//!
//! **阶段 0.4**：本模块仅占位，具体适配器在阶段 1 起填充：
//! - 阶段 1（MVP）：openai / agnes / volcengine_cv / gemini
//! - 阶段 2a（兼容族）：azure / aggregation_platforms / additional_models / more_models / emerging_models / chinese
//! - 阶段 2b（独立协议）：anthropic / stability / runway / pika / kling
//! - 阶段 2c（音频）：edge-tts / elevenlabs / cartesia / deepgram / assemblyai
//!
//! 各适配器实现 `adapter::Adapter` trait，由 `adapter::create_adapter` 工厂分发。

/// Echo（Mock）适配器：阶段 0.6 五语言管线验证用，不调网络返固定/回显响应
pub mod echo;

/// OpenAI 兼容协议适配器地基：阶段 1.0 实现，为 openai/agnes 等子适配器提供共享基础
pub mod openai_compat;
