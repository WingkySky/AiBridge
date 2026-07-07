//! 配置管理
//!
//! 定义 `ClientOptions` 与 `ProviderConfig`，并支持从环境变量加载配置。
//! 对应 Python v1 (agn-sdk) 的 `agn/core/config.py`。
//!
//! 环境变量兼容性：保留对 `AGN_API_KEY` / `AGN_BASE_URL` 等老前缀的读取，
//! 同时新增 `AIBRIDGE_API_KEY` / `AIBRIDGE_BASE_URL` 前缀，迁移期两者并存。

use std::collections::HashMap;
use std::env;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{AibridgeError, Result};

/// 客户端全局选项
///
/// 用于配置单个 Client 的连接与重试参数。
/// 对应 Python v1 `Config` 类与 `Client.__init__` 的连接参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientOptions {
    /// API Key（免费 Provider 如 Edge TTS 可为 None）
    #[serde(default)]
    pub api_key: Option<String>,

    /// API Base URL（可选，部分 Provider 有默认值）
    #[serde(default)]
    pub base_url: Option<String>,

    /// 轮询 URL（视频生成任务状态用，部分 Provider 需要）
    #[serde(default)]
    pub poll_url: Option<String>,

    /// 请求超时时间（秒）
    #[serde(default = "default_timeout")]
    pub timeout: u64,

    /// 最大重试次数
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// 重试初始延迟（秒）
    #[serde(default = "default_retry_delay")]
    pub retry_delay: f64,

    /// 默认 Provider（Router 路由失败时兜底）
    #[serde(default)]
    pub default_provider: Option<String>,

    /// 额外配置（厂商特定配置，如 embed_model / speech_model 等）
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: None,
            poll_url: None,
            timeout: default_timeout(),
            max_retries: default_max_retries(),
            retry_delay: default_retry_delay(),
            default_provider: None,
            extra: HashMap::new(),
        }
    }
}

impl ClientOptions {
    /// 创建一个新的 Builder
    pub fn builder() -> ClientOptionsBuilder {
        ClientOptionsBuilder::default()
    }

    /// 从环境变量加载配置，再与 `self` 合并（`self` 优先级高于环境变量）
    ///
    /// 读取顺序：`AIBRIDGE_{PROVIDER}_*` > `AGN_{PROVIDER}_*` > 通用 `AIBRIDGE_API_KEY` / `AGN_API_KEY`。
    /// `provider` 用于构造 provider 专属环境变量名（大写）。
    pub fn merge_env(mut self, provider: &str) -> Self {
        let upper = provider.to_uppercase();

        // api_key：provider 专属优先，再回退到通用
        if self.api_key.is_none() {
            self.api_key = env_var(&format!("AIBRIDGE_{upper}_API_KEY"))
                .or_else(|| env_var(&format!("AGN_{upper}_API_KEY")))
                .or_else(|| env_var("AIBRIDGE_API_KEY"))
                .or_else(|| env_var("AGN_API_KEY"));
        }

        // base_url
        if self.base_url.is_none() {
            self.base_url = env_var(&format!("AIBRIDGE_{upper}_BASE_URL"))
                .or_else(|| env_var(&format!("AGN_{upper}_BASE_URL")))
                .or_else(|| env_var("AIBRIDGE_BASE_URL"))
                .or_else(|| env_var("AGN_BASE_URL"));
        }

        // poll_url（视频轮询地址，仅部分 Provider 用）
        if self.poll_url.is_none() {
            self.poll_url = env_var(&format!("AIBRIDGE_{upper}_POLL_URL"))
                .or_else(|| env_var(&format!("AGN_{upper}_POLL_URL")));
        }

        self
    }

    /// 获取超时时间对应的 `Duration`
    pub fn timeout_duration(&self) -> Duration {
        Duration::from_secs(self.timeout)
    }
}

/// `ClientOptions` 的 Builder
///
/// 用法：
/// ```ignore
/// let opts = ClientOptions::builder()
///     .api_key("sk-xxx")
///     .base_url("https://api.example.com")
///     .timeout(120)
///     .build();
/// ```
#[derive(Debug, Default, Clone)]
pub struct ClientOptionsBuilder {
    inner: ClientOptions,
}

impl ClientOptionsBuilder {
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.inner.api_key = Some(api_key.into());
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = Some(base_url.into());
        self
    }

    pub fn poll_url(mut self, poll_url: impl Into<String>) -> Self {
        self.inner.poll_url = Some(poll_url.into());
        self
    }

    pub fn timeout(mut self, timeout: u64) -> Self {
        self.inner.timeout = timeout;
        self
    }

    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.inner.max_retries = max_retries;
        self
    }

    pub fn retry_delay(mut self, retry_delay: f64) -> Self {
        self.inner.retry_delay = retry_delay;
        self
    }

    pub fn default_provider(mut self, provider: impl Into<String>) -> Self {
        self.inner.default_provider = Some(provider.into());
        self
    }

    /// 插入一个额外配置项（厂商特定参数）
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.inner.extra.insert(key.into(), value.into());
        self
    }

    pub fn build(self) -> ClientOptions {
        self.inner
    }
}

/// Provider 配置
///
/// 描述单个 AI 模型提供商的连接参数。
/// 对应 Python v1 `agn/models/common.py` 的 `ProviderConfig`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider 类型标识，如 "agnes"、"openai"
    pub provider_type: String,

    /// API Key（免费 Provider 可为 None）
    #[serde(default)]
    pub api_key: Option<String>,

    /// API Base URL（可选，部分 Provider 有默认值）
    #[serde(default)]
    pub base_url: Option<String>,

    /// 轮询 URL（视频生成任务状态用）
    #[serde(default)]
    pub poll_url: Option<String>,

    /// 请求超时时间（秒）
    #[serde(default = "default_timeout")]
    pub timeout: u64,

    /// 最大重试次数
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// 重试延迟（秒）
    #[serde(default = "default_retry_delay")]
    pub retry_delay: f64,

    /// 是否启用该 Provider（Router 用）
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Azure 专用字段：资源名称
    #[serde(default)]
    pub resource_name: Option<String>,

    /// Azure 专用字段：部署 ID
    #[serde(default)]
    pub deployment_id: Option<String>,

    /// API 版本（Azure 等用）
    #[serde(default)]
    pub api_version: Option<String>,

    /// 额外配置（厂商特定配置）
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl ProviderConfig {
    /// 从 `ClientOptions` 与 provider 类型构造配置
    ///
    /// `opts` 的字段优先；opts.api_key 为 None 时保留 None（由调用方按
    /// `requires_api_key` 决定是否报错）。
    pub fn from_options(provider_type: impl Into<String>, opts: ClientOptions) -> Self {
        Self {
            provider_type: provider_type.into(),
            api_key: opts.api_key,
            base_url: opts.base_url,
            poll_url: opts.poll_url,
            timeout: opts.timeout,
            max_retries: opts.max_retries,
            retry_delay: opts.retry_delay,
            enabled: true,
            resource_name: opts
                .extra
                .get("resource_name")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
            deployment_id: opts
                .extra
                .get("deployment_id")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
            api_version: opts
                .extra
                .get("api_version")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
            extra: opts.extra,
        }
    }

    /// 校验配置：必填字段检查等
    ///
    /// `requires_api_key` 为 true 时，api_key 必须非空。
    pub fn validate(&self, requires_api_key: bool) -> Result<()> {
        if self.provider_type.trim().is_empty() {
            return Err(AibridgeError::validation("provider_type 不能为空"));
        }
        if requires_api_key {
            match &self.api_key {
                Some(k) if !k.trim().is_empty() => Ok(()),
                _ => Err(AibridgeError::validation(format!(
                    "Provider '{}' 需要 API key",
                    self.provider_type
                ))),
            }
        } else {
            Ok(())
        }
    }
}

/// 读取环境变量，返回非空字符串（空串视为未设置）
fn env_var(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn default_timeout() -> u64 {
    300
}

fn default_max_retries() -> u32 {
    3
}

fn default_retry_delay() -> f64 {
    2.0
}

fn default_enabled() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// 串行化所有 env 测试（env 变量是进程级共享，并行执行会互相污染）
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn default_options() {
        let opts = ClientOptions::default();
        assert_eq!(opts.timeout, 300);
        assert_eq!(opts.max_retries, 3);
        assert!((opts.retry_delay - 2.0).abs() < f64::EPSILON);
        assert!(opts.api_key.is_none());
    }

    #[test]
    fn builder_sets_fields() {
        let opts = ClientOptions::builder()
            .api_key("sk-xxx")
            .base_url("https://api.example.com")
            .timeout(120)
            .max_retries(5)
            .retry_delay(1.0)
            .default_provider("openai")
            .extra("k", "v")
            .build();
        assert_eq!(opts.api_key.as_deref(), Some("sk-xxx"));
        assert_eq!(opts.base_url.as_deref(), Some("https://api.example.com"));
        assert_eq!(opts.timeout, 120);
        assert_eq!(opts.max_retries, 5);
        assert!((opts.retry_delay - 1.0).abs() < f64::EPSILON);
        assert_eq!(opts.default_provider.as_deref(), Some("openai"));
        assert_eq!(opts.extra.get("k").and_then(|v| v.as_str()), Some("v"));
    }

    #[test]
    fn timeout_duration() {
        let opts = ClientOptions::builder().timeout(42).build();
        assert_eq!(opts.timeout_duration(), Duration::from_secs(42));
    }

    #[test]
    fn merge_env_uses_existing_first() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::set_var("AIBRIDGE_TEST_MERGE_API_KEY", "env-key");
        env::set_var("AIBRIDGE_API_KEY", "global-key");
        let opts = ClientOptions::builder()
            .api_key("explicit-key")
            .build()
            .merge_env("test_merge");
        assert_eq!(opts.api_key.as_deref(), Some("explicit-key"));
        env::remove_var("AIBRIDGE_TEST_MERGE_API_KEY");
        env::remove_var("AIBRIDGE_API_KEY");
    }

    #[test]
    fn merge_env_falls_back_to_provider_specific() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::set_var("AIBRIDGE_FALLBACK_API_KEY", "provider-key");
        let opts = ClientOptions::default().merge_env("fallback");
        assert_eq!(opts.api_key.as_deref(), Some("provider-key"));
        env::remove_var("AIBRIDGE_FALLBACK_API_KEY");
    }

    #[test]
    fn merge_env_falls_back_to_global_agn_prefix() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::set_var("AGN_API_KEY", "agn-global");
        let opts = ClientOptions::default().merge_env("nobody_has_this");
        assert_eq!(opts.api_key.as_deref(), Some("agn-global"));
        env::remove_var("AGN_API_KEY");
    }

    #[test]
    fn merge_env_base_url_provider_specific() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::set_var("AIBRIDGE_URLPROV_BASE_URL", "https://provider.example.com");
        let opts = ClientOptions::default().merge_env("urlprov");
        assert_eq!(
            opts.base_url.as_deref(),
            Some("https://provider.example.com")
        );
        env::remove_var("AIBRIDGE_URLPROV_BASE_URL");
    }

    #[test]
    fn merge_env_empty_string_treated_as_unset() {
        let _guard = ENV_LOCK.lock().unwrap();
        // 确保全局回退变量未设置（隔离其他测试的污染）
        env::remove_var("AIBRIDGE_API_KEY");
        env::remove_var("AGN_API_KEY");
        env::set_var("AIBRIDGE_EMPTY_API_KEY", "");
        let opts = ClientOptions::default().merge_env("empty");
        assert!(opts.api_key.is_none());
        env::remove_var("AIBRIDGE_EMPTY_API_KEY");
    }

    #[test]
    fn provider_config_from_options() {
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("u")
            .timeout(99)
            .build();
        let cfg = ProviderConfig::from_options("openai", opts);
        assert_eq!(cfg.provider_type, "openai");
        assert_eq!(cfg.api_key.as_deref(), Some("k"));
        assert_eq!(cfg.base_url.as_deref(), Some("u"));
        assert_eq!(cfg.timeout, 99);
        assert!(cfg.enabled);
    }

    #[test]
    fn provider_config_validate_requires_key_when_needed() {
        let cfg = ProviderConfig {
            provider_type: "openai".into(),
            api_key: None,
            ..Default::default()
        };
        assert!(cfg.validate(true).is_err());

        let cfg2 = ProviderConfig {
            provider_type: "openai".into(),
            api_key: Some("k".into()),
            ..Default::default()
        };
        assert!(cfg2.validate(true).is_ok());
    }

    #[test]
    fn provider_config_validate_skips_key_when_not_required() {
        let cfg = ProviderConfig {
            provider_type: "edge-tts".into(),
            api_key: None,
            ..Default::default()
        };
        assert!(cfg.validate(false).is_ok());
    }

    #[test]
    fn provider_config_validate_rejects_empty_provider_type() {
        let cfg = ProviderConfig {
            provider_type: "  ".into(),
            api_key: Some("k".into()),
            ..Default::default()
        };
        assert!(cfg.validate(false).is_err());
    }

    impl Default for ProviderConfig {
        fn default() -> Self {
            Self {
                provider_type: String::new(),
                api_key: None,
                base_url: None,
                poll_url: None,
                timeout: default_timeout(),
                max_retries: default_max_retries(),
                retry_delay: default_retry_delay(),
                enabled: true,
                resource_name: None,
                deployment_id: None,
                api_version: None,
                extra: HashMap::new(),
            }
        }
    }

    #[test]
    fn provider_config_serialize_roundtrip() {
        let cfg =
            ProviderConfig::from_options("agnes", ClientOptions::builder().api_key("k").build());
        let json = serde_json::to_string(&cfg).unwrap();
        let back: ProviderConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.provider_type, "agnes");
        assert_eq!(back.api_key.as_deref(), Some("k"));
    }
}
