//! 错误类型定义
//!
//! 定义统一的错误枚举 `AibridgeError`，用于映射各 Provider 的特定错误。
//! 对应 Python v1 (agn-sdk) 的 `agn/core/errors.py`，分类保持一致。
//!
//! 设计文档 9.1 节规定的错误类别：
//! Authentication / RateLimit / Validation / ModelNotFound / Api / Network / Timeout
//! / UnsupportedCapability / ProviderNotFound。
//!
//! 迁移对照：Python v1 的 `AGNError` → Rust 的 `AibridgeError`（子类名不变）。

use std::time::Duration;

/// SDK 统一错误类型
///
/// 所有 SDK 操作返回的错误枚举，按错误性质分类。
/// 各子类对应 Python v1 的同名错误类。
#[derive(Debug, thiserror::Error)]
pub enum AibridgeError {
    /// 认证错误 - API Key 无效、过期或没有权限
    ///
    /// 对应 Python v1 `AuthenticationError`。
    #[error("认证失败: {message}")]
    Authentication { message: String },

    /// 限流错误 - 请求频率超过限制
    ///
    /// 对应 Python v1 `RateLimitError`。`retry_after` 为服务端建议的等待时间（秒）。
    #[error("限流: {message}")]
    RateLimit {
        message: String,
        /// 服务端建议的等待时间（秒），None 表示未提供
        retry_after: Option<f64>,
    },

    /// 参数校验错误 - 请求参数不合法
    ///
    /// 对应 Python v1 `ValidationError`。`details` 携带结构化错误细节。
    #[error("参数校验错误: {message}")]
    Validation {
        message: String,
        /// 结构化错误细节（字段级错误、约束违反等）
        details: serde_json::Value,
    },

    /// 模型不存在错误 - 请求的模型不可用
    ///
    /// 对应 Python v1 `ModelNotFoundError`。
    #[error("模型不存在: {model}")]
    ModelNotFound { model: String },

    /// API 调用错误 - 来自 Provider API 的错误响应
    ///
    /// 对应 Python v1 `APIError`。`status` 为 HTTP 状态码。
    #[error("API 调用错误: {message}")]
    Api {
        /// HTTP 状态码
        status: u16,
        message: String,
    },

    /// 网络错误 - 网络连接问题
    ///
    /// 对应 Python v1 `NetworkError`。由 `reqwest::Error` 自动转换而来。
    #[error("网络错误: {0}")]
    Network(#[from] reqwest::Error),

    /// 超时错误 - 请求超时
    ///
    /// 对应 Python v1 `TimeoutError`。
    #[error("超时")]
    Timeout,

    /// 不支持的能力错误 - Provider 不支持请求的能力
    ///
    /// 对应 Python v1 `UnsupportedCapabilityError`。
    /// `capability` 为不支持的能力标识（如 "chat_stream"、"embedding"）。
    #[error("不支持的能力: {capability}")]
    UnsupportedCapability { capability: String },

    /// Provider 不存在错误 - 请求的 Provider 类型不可用
    ///
    /// 对应 Python v1 `ProviderNotFoundError`。
    #[error("Provider 不存在: {provider}")]
    ProviderNotFound { provider: String },

    /// 音色不可用错误 - 请求的 voice 已下线或不存在
    ///
    /// 对应 Python v1 `VoiceNotAvailableError`。重试无意义，应换音色。
    #[error("音色不可用: {message}")]
    VoiceNotAvailable { message: String },

    /// 服务不可用错误 - Provider 服务端临时不可用（限流/抖动），可重试
    ///
    /// 对应 Python v1 `ServiceUnavailableError`。
    #[error("服务暂时不可用: {message}")]
    ServiceUnavailable { message: String },
}

impl AibridgeError {
    /// 创建认证错误
    pub fn authentication(message: impl Into<String>) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }

    /// 创建限流错误（无 retry_after）
    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self::RateLimit {
            message: message.into(),
            retry_after: None,
        }
    }

    /// 创建限流错误（带 retry_after）
    pub fn rate_limit_with_retry(message: impl Into<String>, retry_after: f64) -> Self {
        Self::RateLimit {
            message: message.into(),
            retry_after: Some(retry_after),
        }
    }

    /// 创建参数校验错误
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
            details: serde_json::Value::Null,
        }
    }

    /// 创建带结构化细节的参数校验错误
    pub fn validation_with_details(message: impl Into<String>, details: serde_json::Value) -> Self {
        Self::Validation {
            message: message.into(),
            details,
        }
    }

    /// 创建模型不存在错误
    pub fn model_not_found(model: impl Into<String>) -> Self {
        Self::ModelNotFound {
            model: model.into(),
        }
    }

    /// 创建 API 调用错误
    pub fn api(status: u16, message: impl Into<String>) -> Self {
        Self::Api {
            status,
            message: message.into(),
        }
    }

    /// 创建不支持的能力错误
    pub fn unsupported_capability(capability: impl Into<String>) -> Self {
        Self::UnsupportedCapability {
            capability: capability.into(),
        }
    }

    /// 创建 Provider 不存在错误
    pub fn provider_not_found(provider: impl Into<String>) -> Self {
        Self::ProviderNotFound {
            provider: provider.into(),
        }
    }

    /// 创建音色不可用错误
    pub fn voice_not_available(message: impl Into<String>) -> Self {
        Self::VoiceNotAvailable {
            message: message.into(),
        }
    }

    /// 创建服务不可用错误
    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::ServiceUnavailable {
            message: message.into(),
        }
    }

    /// 将 HTTP 状态码映射到对应的错误类型
    ///
    /// 对应 Python v1 `map_http_status_to_error`。
    ///
    /// 映射规则：
    /// - 401/403 → Authentication
    /// - 429 → RateLimit
    /// - 400 → Validation
    /// - 404 → ModelNotFound
    /// - 4xx（其他）→ Api
    /// - 5xx → Api
    pub fn from_http_status(status: u16, body: &str) -> Self {
        let details = serde_json::json!({
            "status_code": status,
            "response": body,
        });
        match status {
            401 | 403 => Self::Authentication {
                message: "Authentication failed. Please check your API key.".into(),
            },
            429 => Self::RateLimit {
                message: "Rate limit exceeded. Please slow down your requests.".into(),
                retry_after: None,
            },
            400 => Self::Validation {
                message: "Invalid request. Please check your parameters.".into(),
                details,
            },
            404 => Self::ModelNotFound {
                model: "The requested model was not found.".into(),
            },
            s if (400..500).contains(&s) => Self::Api {
                status,
                message: format!("Client error: {status}"),
            },
            s if (500..600).contains(&s) => Self::Api {
                status,
                message: format!("Server error: {status}. Please try again later."),
            },
            _ => Self::Api {
                status,
                message: format!("Unexpected status code: {status}"),
            },
        }
    }

    /// 判断该错误是否可重试
    ///
    /// 可重试的错误：RateLimit / Network / Timeout / ServiceUnavailable / Api(5xx)。
    /// 不可重试的错误：Authentication / Validation / ModelNotFound /
    /// UnsupportedCapability / ProviderNotFound / VoiceNotAvailable / Api(4xx)。
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::RateLimit { .. } => true,
            Self::Network(_) => true,
            Self::Timeout => true,
            Self::ServiceUnavailable { .. } => true,
            // 5xx 服务端错误可重试
            Self::Api { status, .. } => *status >= 500,
            // 其余错误重试无意义
            Self::Authentication { .. }
            | Self::Validation { .. }
            | Self::ModelNotFound { .. }
            | Self::UnsupportedCapability { .. }
            | Self::ProviderNotFound { .. }
            | Self::VoiceNotAvailable { .. } => false,
        }
    }

    /// 错误的稳定标识码（snake_case），用于序列化到 FFI 的 last_error JSON
    ///
    /// 对应 Python v1 各错误类的 `code` 字段。
    pub fn code(&self) -> &'static str {
        match self {
            Self::Authentication { .. } => "authentication_error",
            Self::RateLimit { .. } => "rate_limit_error",
            Self::Validation { .. } => "validation_error",
            Self::ModelNotFound { .. } => "model_not_found",
            Self::Api { .. } => "api_error",
            Self::Network(_) => "network_error",
            Self::Timeout => "timeout_error",
            Self::UnsupportedCapability { .. } => "unsupported_capability",
            Self::ProviderNotFound { .. } => "provider_not_found",
            Self::VoiceNotAvailable { .. } => "voice_not_available",
            Self::ServiceUnavailable { .. } => "service_unavailable",
        }
    }

    /// 若为限流错误，返回 retry_after 对应的等待时长
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimit { retry_after, .. } => retry_after.map(Duration::from_secs_f64),
            _ => None,
        }
    }
}

/// SDK Result 类型别名，统一用 `AibridgeError` 作错误
pub type Result<T> = std::result::Result<T, AibridgeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_http_status_401_is_authentication() {
        let err = AibridgeError::from_http_status(401, "unauthorized");
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[test]
    fn from_http_status_429_is_rate_limit() {
        let err = AibridgeError::from_http_status(429, "slow down");
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[test]
    fn from_http_status_400_is_validation() {
        let err = AibridgeError::from_http_status(400, "bad request");
        assert!(matches!(err, AibridgeError::Validation { .. }));
    }

    #[test]
    fn from_http_status_404_is_model_not_found() {
        let err = AibridgeError::from_http_status(404, "not found");
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[test]
    fn from_http_status_500_is_api_and_retryable() {
        let err = AibridgeError::from_http_status(500, "server error");
        assert!(matches!(err, AibridgeError::Api { status: 500, .. }));
        assert!(err.is_retryable());
    }

    #[test]
    fn from_http_status_422_is_api_and_not_retryable() {
        let err = AibridgeError::from_http_status(422, "unprocessable");
        assert!(matches!(err, AibridgeError::Api { status: 422, .. }));
        assert!(!err.is_retryable());
    }

    #[test]
    fn rate_limit_is_retryable() {
        assert!(AibridgeError::rate_limit("slow").is_retryable());
        assert!(AibridgeError::rate_limit_with_retry("slow", 2.5).is_retryable());
    }

    #[test]
    fn timeout_is_retryable() {
        assert!(AibridgeError::Timeout.is_retryable());
    }

    #[test]
    fn authentication_is_not_retryable() {
        assert!(!AibridgeError::authentication("bad key").is_retryable());
    }

    #[test]
    fn validation_is_not_retryable() {
        assert!(!AibridgeError::validation("bad param").is_retryable());
    }

    #[test]
    fn model_not_found_is_not_retryable() {
        assert!(!AibridgeError::model_not_found("gpt-x").is_retryable());
    }

    #[test]
    fn unsupported_capability_is_not_retryable() {
        assert!(!AibridgeError::unsupported_capability("video").is_retryable());
    }

    #[test]
    fn provider_not_found_is_not_retryable() {
        assert!(!AibridgeError::provider_not_found("foo").is_retryable());
    }

    #[test]
    fn voice_not_available_is_not_retryable() {
        assert!(!AibridgeError::voice_not_available("offline").is_retryable());
    }

    #[test]
    fn service_unavailable_is_retryable() {
        assert!(AibridgeError::service_unavailable("temp").is_retryable());
    }

    #[test]
    fn retry_after_returns_duration() {
        let err = AibridgeError::rate_limit_with_retry("slow", 1.5);
        assert_eq!(err.retry_after(), Some(Duration::from_millis(1500)));
    }

    #[test]
    fn retry_after_none_for_non_rate_limit() {
        assert_eq!(AibridgeError::Timeout.retry_after(), None);
    }

    #[test]
    fn code_is_stable() {
        assert_eq!(AibridgeError::Timeout.code(), "timeout_error");
        assert_eq!(AibridgeError::rate_limit("").code(), "rate_limit_error");
        assert_eq!(
            AibridgeError::authentication("").code(),
            "authentication_error"
        );
    }

    #[test]
    fn display_includes_message() {
        let err = AibridgeError::authentication("bad key");
        assert!(err.to_string().contains("bad key"));
    }

    #[test]
    fn validation_with_details_carries_json() {
        let err =
            AibridgeError::validation_with_details("bad", serde_json::json!({"field": "model"}));
        match err {
            AibridgeError::Validation { details, .. } => {
                assert_eq!(details["field"], "model");
            }
            _ => panic!("应为 Validation"),
        }
    }
}
