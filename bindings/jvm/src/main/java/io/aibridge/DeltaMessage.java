package io.aibridge;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * 流式增量消息（对应 Rust {@code DeltaMessage}）。
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class DeltaMessage {

    /** 角色（首个块通常为 "assistant"） */
    public String role;
    /** 增量内容 */
    public String content;
    /** 工具调用增量（可选） */
    @JsonProperty("tool_calls")
    public List<Object> toolCalls;

    public DeltaMessage() {
    }
}
