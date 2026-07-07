package io.aibridge;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * 流式对话块（对应 Rust {@code ChatCompletionChunk}）。
 *
 * <p>由 {@code aibridge_stream_next} 的 {@code out_chunk_json} 反序列化得到。
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ChatCompletionChunk {

    public String id;
    public String object;
    public long created;
    public String model;
    public List<ChatCompletionDelta> choices;
    public ChatUsage usage;

    public ChatCompletionChunk() {
    }

    /** 取首个 delta 的增量内容（便捷方法，可能为 null） */
    public String firstDeltaContent() {
        if (choices == null || choices.isEmpty()) {
            return null;
        }
        DeltaMessage delta = choices.get(0).delta;
        return delta == null ? null : delta.content;
    }
}
