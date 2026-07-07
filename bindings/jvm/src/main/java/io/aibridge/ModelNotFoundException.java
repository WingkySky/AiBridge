package io.aibridge;

/** 模型不存在（对应 code = "model_not_found"） */
public class ModelNotFoundException extends AibridgeException {
    public ModelNotFoundException(String message, String details, boolean retryable) {
        super(CODE_MODEL_NOT_FOUND, message, details, retryable);
    }
}
