package io.aibridge;

/** 认证错误（API Key 无效/缺失等，对应 code = "authentication_error"） */
public class AuthenticationException extends AibridgeException {
    public AuthenticationException(String message, String details, boolean retryable) {
        super(CODE_AUTHENTICATION, message, details, retryable);
    }
}
