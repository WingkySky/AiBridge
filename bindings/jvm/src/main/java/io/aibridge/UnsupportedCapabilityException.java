package io.aibridge;

/** 不支持的能力（对应 code = "unsupported_capability"） */
public class UnsupportedCapabilityException extends AibridgeException {
    public UnsupportedCapabilityException(String message, String details, boolean retryable) {
        super(CODE_UNSUPPORTED_CAPABILITY, message, details, retryable);
    }
}
