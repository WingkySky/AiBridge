package io.aibridge;

/** 网络错误（对应 code = "network_error"） */
public class NetworkException extends AibridgeException {
    public NetworkException(String message, String details, boolean retryable) {
        super(CODE_NETWORK, message, details, retryable);
    }
}
