package io.aibridge;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * 流式增量（对应 Rust {@code ChatCompletionDelta}）。
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ChatCompletionDelta {

    public int index;
    public DeltaMessage delta;
    @JsonProperty("finish_reason")
    public String finishReason;

    public ChatCompletionDelta() {
    }
}
