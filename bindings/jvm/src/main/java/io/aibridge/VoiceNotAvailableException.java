package io.aibridge;

/** 音色不可用（对应 code = "voice_not_available"） */
public class VoiceNotAvailableException extends AibridgeException {
    public VoiceNotAvailableException(String message, String details, boolean retryable) {
        super(CODE_VOICE_NOT_AVAILABLE, message, details, retryable);
    }
}
