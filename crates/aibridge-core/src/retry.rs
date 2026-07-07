//! 重试机制
//!
//! 提供异步重试工具，支持指数退避策略。
//! 对应 Python v1 (agn-sdk) 的 `agn/core/retry.py`（替代 tenacity）。
//!
//! 设计要点：
//! - 仅对可重试错误重试（`AibridgeError::is_retryable`）
//! - 指数退避：`delay * multiplier^attempt`，封顶 `max_delay`
//! - 限流错误优先采用服务端 `retry_after`

use std::future::Future;
use std::time::Duration;

use tokio::time::sleep;

use crate::error::{AibridgeError, Result};

/// 重试策略配置
///
/// 对应 Python v1 `retry_on_error` 的参数。
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// 最大尝试次数（含首次）
    pub max_attempts: u32,

    /// 初始延迟（秒）
    pub delay: f64,

    /// 退避倍数
    pub multiplier: f64,

    /// 最大延迟（秒）
    pub max_delay: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            delay: 2.0,
            multiplier: 2.0,
            max_delay: 60.0,
        }
    }
}

impl RetryPolicy {
    /// 创建一个新的 Builder
    pub fn builder() -> RetryPolicyBuilder {
        RetryPolicyBuilder::default()
    }

    /// 计算第 `attempt` 次失败后的等待时长（attempt 从 1 开始）
    ///
    /// 公式：`min(delay * multiplier^(attempt-1), max_delay)`。
    /// 若错误携带 `retry_after`，优先使用 `retry_after`。
    pub fn backoff(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }
        let exp = self.multiplier.powi((attempt - 1) as i32);
        let secs = (self.delay * exp).min(self.max_delay).max(0.0);
        Duration::from_secs_f64(secs)
    }

    /// 计算针对特定错误的等待时长（限流错误优先用 retry_after）
    pub fn backoff_for(&self, attempt: u32, err: &AibridgeError) -> Duration {
        if let Some(retry_after) = err.retry_after() {
            // 服务端建议值封顶到 max_delay，避免过长等待
            let capped = if retry_after > Duration::from_secs_f64(self.max_delay) {
                Duration::from_secs_f64(self.max_delay)
            } else {
                retry_after
            };
            // 至少为指数退避值（取较大者，尊重服务端的下限建议）
            capped.max(self.backoff(attempt))
        } else {
            self.backoff(attempt)
        }
    }
}

/// `RetryPolicy` 的 Builder
#[derive(Debug, Default, Clone, Copy)]
pub struct RetryPolicyBuilder {
    inner: RetryPolicy,
}

impl RetryPolicyBuilder {
    pub fn max_attempts(mut self, n: u32) -> Self {
        self.inner.max_attempts = n;
        self
    }

    pub fn delay(mut self, delay: f64) -> Self {
        self.inner.delay = delay;
        self
    }

    pub fn multiplier(mut self, multiplier: f64) -> Self {
        self.inner.multiplier = multiplier;
        self
    }

    pub fn max_delay(mut self, max_delay: f64) -> Self {
        self.inner.max_delay = max_delay;
        self
    }

    pub fn build(self) -> RetryPolicy {
        self.inner
    }
}

/// 对异步操作执行重试
///
/// 对应 Python v1 `retry_async`。仅当返回的错误可重试时重试。
///
/// # 参数
/// - `policy`: 重试策略
/// - `operation`: 异步闭包，返回 `Result<T>`
///
/// # 示例
/// ```ignore
/// let result: Result<i32> = retry_with(&policy, || async {
///     // 可能失败的异步操作
///     Ok(42)
/// }).await;
/// ```
pub async fn retry_with<T, F, Fut>(policy: &RetryPolicy, operation: F) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut last_err: Option<AibridgeError> = None;
    for attempt in 1..=policy.max_attempts {
        match operation().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                let retryable = e.is_retryable();
                last_err = Some(e);
                if !retryable || attempt == policy.max_attempts {
                    break;
                }
                let wait = match last_err {
                    Some(ref err) => policy.backoff_for(attempt, err),
                    None => policy.backoff(attempt),
                };
                if !wait.is_zero() {
                    sleep(wait).await;
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| AibridgeError::Api {
        status: 0,
        message: "retry_with: 未产生错误但流程异常".into(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn backoff_grows_exponentially() {
        let p = RetryPolicy {
            max_attempts: 5,
            delay: 1.0,
            multiplier: 2.0,
            max_delay: 100.0,
        };
        assert_eq!(p.backoff(1), Duration::from_secs(1));
        assert_eq!(p.backoff(2), Duration::from_secs(2));
        assert_eq!(p.backoff(3), Duration::from_secs(4));
        assert_eq!(p.backoff(4), Duration::from_secs(8));
    }

    #[test]
    fn backoff_capped_at_max_delay() {
        let p = RetryPolicy {
            max_attempts: 10,
            delay: 1.0,
            multiplier: 2.0,
            max_delay: 5.0,
        };
        // 第 4 次：1*2^3 = 8，封顶到 5
        assert_eq!(p.backoff(4), Duration::from_secs(5));
    }

    #[test]
    fn backoff_zero_for_attempt_zero() {
        let p = RetryPolicy::default();
        assert_eq!(p.backoff(0), Duration::ZERO);
    }

    #[test]
    fn backoff_for_rate_limit_uses_retry_after() {
        let p = RetryPolicy {
            max_attempts: 5,
            delay: 1.0,
            multiplier: 2.0,
            max_delay: 60.0,
        };
        let err = AibridgeError::rate_limit_with_retry("slow", 3.0);
        assert_eq!(p.backoff_for(1, &err), Duration::from_secs(3));
    }

    #[test]
    fn backoff_for_rate_limit_capped_by_max_delay() {
        let p = RetryPolicy {
            max_attempts: 5,
            delay: 1.0,
            multiplier: 2.0,
            max_delay: 5.0,
        };
        let err = AibridgeError::rate_limit_with_retry("slow", 100.0);
        assert_eq!(p.backoff_for(1, &err), Duration::from_secs(5));
    }

    #[test]
    fn backoff_for_non_rate_limit_uses_exponential() {
        let p = RetryPolicy {
            max_attempts: 5,
            delay: 1.0,
            multiplier: 2.0,
            max_delay: 60.0,
        };
        let err = AibridgeError::Timeout;
        assert_eq!(p.backoff_for(2, &err), Duration::from_secs(2));
    }

    #[tokio::test]
    async fn retry_succeeds_on_first_attempt() {
        let policy = RetryPolicy::default();
        let result: Result<i32> = retry_with(&policy, || async { Ok(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn retry_succeeds_after_transient_failure() {
        let policy = RetryPolicy::builder()
            .max_attempts(3)
            .delay(0.001)
            .multiplier(1.0)
            .max_delay(0.01)
            .build();
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        let result: Result<i32> = retry_with(&policy, || {
            let c = c.clone();
            async move {
                let n = c.fetch_add(1, Ordering::SeqCst);
                if n < 1 {
                    Err(AibridgeError::Timeout)
                } else {
                    Ok(7)
                }
            }
        })
        .await;
        assert_eq!(result.unwrap(), 7);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_gives_up_after_max_attempts() {
        let policy = RetryPolicy::builder()
            .max_attempts(2)
            .delay(0.001)
            .max_delay(0.01)
            .build();
        let result: Result<i32> =
            retry_with(&policy, || async { Err(AibridgeError::Timeout) }).await;
        assert!(matches!(result, Err(AibridgeError::Timeout)));
    }

    #[tokio::test]
    async fn retry_does_not_retry_non_retryable_error() {
        let policy = RetryPolicy::builder().max_attempts(5).delay(0.001).build();
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        let result: Result<i32> = retry_with(&policy, || {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err(AibridgeError::authentication("bad key"))
            }
        })
        .await;
        // 非可重试错误应立即返回，不重试
        assert!(matches!(result, Err(AibridgeError::Authentication { .. })));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn builder_defaults() {
        let p = RetryPolicy::builder().build();
        assert_eq!(p.max_attempts, 3);
        assert!((p.delay - 2.0).abs() < f64::EPSILON);
        assert!((p.multiplier - 2.0).abs() < f64::EPSILON);
        assert!((p.max_delay - 60.0).abs() < f64::EPSILON);
    }
}
