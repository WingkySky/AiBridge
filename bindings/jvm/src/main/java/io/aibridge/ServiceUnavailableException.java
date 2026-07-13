package io.aibridge;

/** 服务暂时不可用（对应 code = "service_unavailable"） */
public class ServiceUnavailableException extends AibridgeException {
    public ServiceUnavailableException(String message, String details, boolean retryable) {
        super(CODE_SERVICE_UNAVAILABLE, message, details, retryable);
    }
}
