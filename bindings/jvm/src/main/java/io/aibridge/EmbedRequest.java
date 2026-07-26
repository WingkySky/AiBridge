package io.aibridge;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * 文本嵌入请求（对应 Rust EmbedRequest）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class EmbedRequest {
    public String model;
    /** EmbedInput: 单个字符串或字符串列表 */
    public Object input;
    public Integer dimensions;
    @JsonProperty("encoding_format")
    public String encodingFormat;
    public String user;
    public Map<String, Object> extra;

    public EmbedRequest() {}

    public EmbedRequest(String model, Object input) {
        this.model = model;
        this.input = input;
    }

    public static Builder builder(String model, Object input) {
        return new Builder(model, input);
    }

    public static class Builder {
        private final EmbedRequest req;

        public Builder(String model, Object input) {
            this.req = new EmbedRequest(model, input);
        }

        public Builder dimensions(int d) { req.dimensions = d; return this; }
        public Builder encodingFormat(String e) { req.encodingFormat = e; return this; }
        public Builder user(String u) { req.user = u; return this; }
        public Builder extra(Map<String, Object> e) { req.extra = e; return this; }

        public EmbedRequest build() { return req; }
    }
}

/**
 * 嵌入结果（对应 Rust EmbeddingResult）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class EmbeddingResult {
    public String object;
    public List<EmbeddingItem> data;
    public String model;
    public EmbeddingUsage usage;
}

/**
 * 单个嵌入项（对应 Rust EmbeddingItem）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class EmbeddingItem {
    public String object;
    public int index;
    public EmbeddingVector embedding;
}

/**
 * 嵌入向量（浮点列表或 base64 字符串，对应 Rust EmbeddingVector untagged enum）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class EmbeddingVector {
    public List<Double> floats;
    public String base64;

    public List<Double> getFloats() { return floats; }
    public String getBase64() { return base64; }
}

/**
 * 嵌入使用统计（对应 Rust EmbeddingUsage）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class EmbeddingUsage {
    @JsonProperty("prompt_tokens")
    public long promptTokens;
    @JsonProperty("total_tokens")
    public long totalTokens;
}
