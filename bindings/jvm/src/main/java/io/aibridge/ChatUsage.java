package io.aibridge;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Token 使用统计（对应 Rust {@code ChatUsage}）。
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ChatUsage {

    @JsonProperty("prompt_tokens")
    public long promptTokens;
    @JsonProperty("completion_tokens")
    public long completionTokens;
    @JsonProperty("total_tokens")
    public long totalTokens;

    public ChatUsage() {
    }
}
