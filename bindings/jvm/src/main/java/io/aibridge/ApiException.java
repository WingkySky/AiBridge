package io.aibridge;

/** API 调用错误（HTTP 4xx/5xx，对应 code = "api_error"） */
public class ApiException extends AibridgeException {
    public ApiException(String message, String details, boolean retryable) {
        super(CODE_API, message, details, retryable);
    }
}
