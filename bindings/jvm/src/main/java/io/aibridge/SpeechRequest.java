package io.aibridge;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * 文字转语音请求（对应 Rust {@code SpeechRequest}）。
 *
 * <p>注意 {@code voice} 在 Rust 侧是 {@code VoiceSpec}（含 {@code voices} 数组），
 * 支持候选列表自动降级。这里用 {@link VoiceSpec} 嵌套表示。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class SpeechRequest {

    /** 模型名称（如 "tts-1"、"echo-tts"） */
    public String model;
    /** 要合成的文本 */
    public String input;
    /** 音色规格（候选列表） */
    public VoiceSpec voice;
    /** 音频输出格式（"mp3" / "opus" / "wav" 等） */
    @JsonProperty("response_format")
    public String responseFormat;
    /** 语速（0.25-4.0） */
    public Double speed;

    public SpeechRequest() {
    }

    public SpeechRequest(String model, String input, String voice) {
        this.model = model;
        this.input = input;
        this.voice = VoiceSpec.single(voice);
    }

    /** 创建 Builder（单音色） */
    public static Builder builder(String model, String input, String voice) {
        return new Builder(model, input, voice);
    }

    /** 链式构造器 */
    public static class Builder {
        private final SpeechRequest req;

        public Builder(String model, String input, String voice) {
            this.req = new SpeechRequest(model, input, voice);
        }

        public Builder responseFormat(String fmt) {
            req.responseFormat = fmt;
            return this;
        }

        public Builder speed(double s) {
            req.speed = s;
            return this;
        }

        public SpeechRequest build() {
            return req;
        }
    }
}
