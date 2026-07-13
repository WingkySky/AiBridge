package io.aibridge;

/** Provider 不存在（对应 code = "provider_not_found"） */
public class ProviderNotFoundException extends AibridgeException {
    public ProviderNotFoundException(String message, String details, boolean retryable) {
        super(CODE_PROVIDER_NOT_FOUND, message, details, retryable);
    }
}
