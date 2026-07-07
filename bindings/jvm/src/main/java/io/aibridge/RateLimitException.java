package io.aibridge;

/**
 * 限流错误（对应 code = "rate_limit_error"）。
 *
 * <p>{@code details} 可能包含 {@code retry_after}（建议等待秒数）。
 */
public class RateLimitException extends AibridgeException {
    public RateLimitException(String message, String details, boolean retryable) {
        super(CODE_RATE_LIMIT, message, details, retryable);
    }
}
