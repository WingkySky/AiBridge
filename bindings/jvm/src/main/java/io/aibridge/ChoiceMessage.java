package io.aibridge;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * 完成结果中的消息（对应 Rust {@code ChoiceMessage}）。
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ChoiceMessage {

    /** 角色（通常为 "assistant"） */
    public String role;
    /** 消息内容 */
    public String content;
    /** 工具调用列表（可选） */
    @JsonProperty("tool_calls")
    public List<Object> toolCalls;

    public ChoiceMessage() {
    }
}
