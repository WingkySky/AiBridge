package io.aibridge;

/** 超时错误（对应 code = "timeout_error"） */
public class TimeoutException extends AibridgeException {
    public TimeoutException(String message, String details, boolean retryable) {
        super(CODE_TIMEOUT, message, details, retryable);
    }
}
