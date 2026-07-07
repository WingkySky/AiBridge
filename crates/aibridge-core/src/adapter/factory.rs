//! 适配器工厂
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/factory.py`。
//!
//! 设计要点（与设计文档 5.2 节一致）：
//! - 用编译期显式 `match` 替代 Python 运行时注册（更静态、更安全）
//! - 新增适配器=加 match 分支
//! - 阶段 0.4 暂只占位分支（返 ProviderNotFound），具体适配器阶段 1 起填充

use crate::adapter::Adapter;
use crate::adapters::agnes::AgnesAdapter;
use crate::adapters::echo::EchoAdapter;
use crate::adapters::gemini::GeminiAdapter;
use crate::adapters::openai::OpenAiAdapter;
use crate::adapters::volcengine_cv::VolcengineCvAdapter;
use crate::config::ProviderConfig;
use crate::error::{AibridgeError, Result};

/// 已支持的 provider 列表（用于错误信息与测试）
///
/// 阶段 1 起逐步填充实际支持的 provider 名。
/// `echo` 为阶段 0.6 管线验证用的 mock 适配器，常驻可用。
pub const KNOWN_PROVIDERS: &[&str] = &[
    "echo",
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
/// `echo` 为阶段 0.6 管线验证用 mock 适配器，已实现；
/// 其余 provider 阶段 0.4 占位（返 ProviderNotFound），阶段 1 起填充。
pub fn create_adapter(config: ProviderConfig) -> Result<Box<dyn Adapter>> {
    let provider = config.provider_type.clone();
    match provider.as_str() {
        // Echo（Mock）适配器：阶段 0.6 管线验证用，已实现
        "echo" => Ok(Box::new(EchoAdapter::new())),
        // 阶段 1 MVP 适配器：阶段 1.0 已实现具体构造逻辑
        "openai" => Ok(Box::new(OpenAiAdapter::new(config)?)),
        "agnes" => Ok(Box::new(AgnesAdapter::new(config)?)),
        "volcengine_cv" => Ok(Box::new(VolcengineCvAdapter::new(config)?)),
        "gemini" => Ok(Box::new(GeminiAdapter::new(config)?)),
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
    fn create_openai_returns_adapter() {
        // 阶段 1：工厂已能构造真实 OpenAiAdapter（仅校验构造成功，不触发 HTTP）
        let adapter = create_adapter(config_for("openai")).expect("工厂应能创建 openai 适配器");
        assert_eq!(adapter.provider_type(), "openai");
    }

    #[test]
    fn create_agnes_returns_adapter() {
        let adapter = create_adapter(config_for("agnes")).expect("工厂应能创建 agnes 适配器");
        assert_eq!(adapter.provider_type(), "agnes");
    }

    #[test]
    fn create_volcengine_cv_returns_adapter() {
        let adapter =
            create_adapter(config_for("volcengine_cv")).expect("工厂应能创建 volcengine_cv 适配器");
        assert_eq!(adapter.provider_type(), "volcengine_cv");
    }

    #[test]
    fn create_gemini_returns_adapter() {
        let adapter = create_adapter(config_for("gemini")).expect("工厂应能创建 gemini 适配器");
        assert_eq!(adapter.provider_type(), "gemini");
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
        assert!(is_known_provider("echo"));
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

    #[test]
    fn create_echo_returns_adapter() {
        // echo 免认证，无需 api_key
        let config = ProviderConfig::from_options("echo", ClientOptions::default());
        let result = create_adapter(config);
        assert!(result.is_ok(), "工厂应能创建 echo 适配器");
        let adapter = result.unwrap();
        assert_eq!(adapter.provider_type(), "echo");
        assert_eq!(adapter.provider_name(), "Echo (Mock)");
        assert!(!adapter.requires_api_key());
    }
}
