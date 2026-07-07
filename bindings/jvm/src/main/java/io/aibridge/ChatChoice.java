package io.aibridge;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * 对话选项（对应 Rust {@code ChatChoice}）。
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ChatChoice {

    /** 选项索引 */
    public int index;
    /** 生成的回复消息 */
    public ChoiceMessage message;
    /** 结束原因（stop / length / content_filter / tool_calls） */
    @JsonProperty("finish_reason")
    public String finishReason;

    public ChatChoice() {
    }
}
