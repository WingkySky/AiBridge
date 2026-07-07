package io.aibridge;

/** 参数校验错误（对应 code = "validation_error"） */
public class ValidationException extends AibridgeException {
    public ValidationException(String message, String details, boolean retryable) {
        super(CODE_VALIDATION, message, details, retryable);
    }
}
