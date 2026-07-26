package io.aibridge;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * 语音转文字请求（对应 Rust TranscribeRequest）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class TranscribeRequest {
    public String model;
    /** FileInput: 路径字符串、URL 字符串、字节数组或 base64 字符串 */
    public Object file;
    public String language;
    public String prompt;
    @JsonProperty("response_format")
    public String responseFormat;
    public Double temperature;
    @JsonProperty("timestamp_granularities")
    public List<String> timestampGranularities;
    public Boolean translate;
    public Map<String, Object> extra;

    public TranscribeRequest() {}

    public TranscribeRequest(String model, Object file) {
        this.model = model;
        this.file = file;
    }

    public static Builder builder(String model, Object file) {
        return new Builder(model, file);
    }

    public static class Builder {
        private final TranscribeRequest req;

        public Builder(String model, Object file) {
            this.req = new TranscribeRequest(model, file);
        }

        public Builder language(String l) { req.language = l; return this; }
        public Builder prompt(String p) { req.prompt = p; return this; }
        public Builder responseFormat(String r) { req.responseFormat = r; return this; }
        public Builder temperature(double t) { req.temperature = t; return this; }
        public Builder timestampGranularities(List<String> g) { req.timestampGranularities = g; return this; }
        public Builder translate(boolean t) { req.translate = t; return this; }
        public Builder extra(Map<String, Object> e) { req.extra = e; return this; }

        public TranscribeRequest build() { return req; }
    }
}

/**
 * 转写结果（对应 Rust TranscriptionResult）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class TranscriptionResult {
    public String text;
    public String language;
    public Double duration;
    public List<TranscriptionSegment> segments;
    public List<TranscriptionWord> words;
    public String task;
    public Object usage;
    public String model;
}

/**
 * 转写分段信息（对应 Rust TranscriptionSegment）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class TranscriptionSegment {
    public int id;
    public double start;
    public double end;
    public String text;
    public Double confidence;
    public String speaker;
}

/**
 * 转写词级时间戳（对应 Rust TranscriptionWord）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class TranscriptionWord {
    public String word;
    public double start;
    public double end;
    public Double confidence;
}
