//! 适配器工厂
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/factory.py`。
//!
//! 设计要点（与设计文档 5.2 节一致）：
//! - 用编译期显式 `match` 替代 Python 运行时注册（更静态、更安全）
//! - 新增适配器=加 match 分支
//! - 阶段 0.4 暂只占位分支（返 ProviderNotFound），具体适配器阶段 1 起填充

use crate::adapter::Adapter;
use crate::config::ProviderConfig;
use crate::error::{AibridgeError, Result};

/// 已支持的 provider 列表（用于错误信息与测试）
///
/// 阶段 1 起逐步填充实际支持的 provider 名。
pub const KNOWN_PROVIDERS: &[&str] = &[
    "openai",
    "agnes",
    "volcengine_cv",
    "gemini",
    // 阶段 2 起补充：
    "azure",
    "anthropic",
    "runway",
    "pika",
    "kling",
    "stability",
    "chinese",
    "aggregation_platforms",
    "edge-tts",
    "elevenlabs",
    "cartesia",
    "deepgram",
    "assemblyai",
];

/// 根据配置创建适配器实例
///
/// 对应 Python v1 `AdapterFactory.create`。
/// 阶段 0.4：所有分支均为占位（返 ProviderNotFound），具体适配器在阶段 1 起填充。
pub fn create_adapter(config: ProviderConfig) -> Result<Box<dyn Adapter>> {
    let provider = config.provider_type.as_str();
    match provider {
        // 阶段 1 MVP 适配器（阶段 1.0 起填充实际构造逻辑）
        "openai" | "agnes" | "volcengine_cv" | "gemini" => {
            // TODO(阶段 1): 引入 adapters::openai::OpenAiAdapter 等具体实现
            Err(AibridgeError::ProviderNotFound {
                provider: format!("{provider}（阶段 1 待实现）"),
            })
        }
        // 阶段 2 适配器占位
        "azure"
        | "anthropic"
        | "runway"
        | "pika"
        | "kling"
        | "stability"
        | "chinese"
        | "aggregation_platforms"
        | "edge-tts"
        | "elevenlabs"
        | "cartesia"
        | "deepgram"
        | "assemblyai" => Err(AibridgeError::ProviderNotFound {
            provider: format!("{provider}（阶段 2 待实现）"),
        }),
        // 未知 provider
        _ => Err(AibridgeError::provider_not_found(format!(
            "{provider}（未知 provider，支持：{}）",
            KNOWN_PROVIDERS.join(", ")
        ))),
    }
}

/// 检查 provider 是否已被工厂识别（不一定已实现）
pub fn is_known_provider(provider: &str) -> bool {
    KNOWN_PROVIDERS.contains(&provider)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientOptions;

    fn config_for(provider: &str) -> ProviderConfig {
        ProviderConfig::from_options(provider, ClientOptions::builder().api_key("k").build())
    }

    #[test]
    fn create_unknown_provider_returns_error() {
        let result = create_adapter(config_for("nonexistent"));
        assert!(matches!(
            result,
            Err(AibridgeError::ProviderNotFound { .. })
        ));
    }

    #[test]
    fn create_openai_returns_pending_for_phase0() {
        let result = create_adapter(config_for("openai"));
        // 阶段 0.4：占位返 ProviderNotFound（待阶段 1 实现）
        assert!(matches!(
            result,
            Err(AibridgeError::ProviderNotFound { .. })
        ));
        if let Err(AibridgeError::ProviderNotFound { provider }) = result {
            assert!(provider.contains("阶段 1"));
        }
    }

    #[test]
    fn create_agnes_returns_pending() {
        let result = create_adapter(config_for("agnes"));
        assert!(matches!(
            result,
            Err(AibridgeError::ProviderNotFound { .. })
        ));
    }

    #[test]
    fn create_volcengine_cv_returns_pending() {
        let result = create_adapter(config_for("volcengine_cv"));
        assert!(matches!(
            result,
            Err(AibridgeError::ProviderNotFound { .. })
        ));
    }

    #[test]
    fn create_gemini_returns_pending() {
        let result = create_adapter(config_for("gemini"));
        assert!(matches!(
            result,
            Err(AibridgeError::ProviderNotFound { .. })
        ));
    }

    #[test]
    fn create_phase2_adapter_returns_phase2_message() {
        let result = create_adapter(config_for("anthropic"));
        if let Err(AibridgeError::ProviderNotFound { provider }) = result {
            assert!(provider.contains("阶段 2"));
        } else {
            panic!("应为 ProviderNotFound");
        }
    }

    #[test]
    fn is_known_provider_recognizes_known() {
        assert!(is_known_provider("openai"));
        assert!(is_known_provider("edge-tts"));
        assert!(is_known_provider("assemblyai"));
    }

    #[test]
    fn is_known_provider_rejects_unknown() {
        assert!(!is_known_provider("nonexistent"));
    }

    #[test]
    fn error_for_unknown_mentions_known_providers() {
        let result = create_adapter(config_for("xxx"));
        if let Err(AibridgeError::ProviderNotFound { provider }) = result {
            assert!(provider.contains("openai"));
        } else {
            panic!("应为 ProviderNotFound");
        }
    }
}
