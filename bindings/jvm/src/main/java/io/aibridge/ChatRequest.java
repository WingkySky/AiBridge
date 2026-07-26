package io.aibridge;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * 对话请求（对应 Rust ChatRequest）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ChatRequest {

    /** 模型名称 */
    public String model;
    /** 消息列表 */
    public List<ChatMessage> messages;
    /** 温度系数 */
    public Double temperature;
    /** 最大生成 token 数 */
    @JsonProperty("max_tokens")
    public Integer maxTokens;
    /** 生成数量 */
    public Integer n;
    /** 存在惩罚 */
    @JsonProperty("presence_penalty")
    public Double presencePenalty;
    /** 频率惩罚 */
    @JsonProperty("frequency_penalty")
    public Double frequencyPenalty;
    /** 随机种子 */
    public Long seed;
    /** 是否流式输出 */
    public Boolean stream;
    /** 用户标识 */
    public String user;
    /** 响应格式 */
    public ResponseFormat responseFormat;
    /** 停止词 */
    public StopSeq stop;
    /** 工具定义列表 */
    public List<ToolDefinition> tools;
    /** 工具选择策略 */
    @JsonProperty("tool_choice")
    public ToolChoice toolChoice;
    /** 厂商特有参数透传 */
    public Map<String, Object> extra;

    public ChatRequest() {}

    public ChatRequest(String model, List<ChatMessage> messages) {
        this.model = model;
        this.messages = messages;
    }

    public static Builder builder(String model, List<ChatMessage> messages) {
        return new Builder(model, messages);
    }

    public static class Builder {
        private final ChatRequest req;

        public Builder(String model, List<ChatMessage> messages) {
            this.req = new ChatRequest(model, messages);
        }

        public Builder temperature(double t) { req.temperature = t; return this; }
        public Builder maxTokens(int n) { req.maxTokens = n; return this; }
        public Builder n(int n) { req.n = n; return this; }
        public Builder presencePenalty(double p) { req.presencePenalty = p; return this; }
        public Builder frequencyPenalty(double f) { req.frequencyPenalty = f; return this; }
        public Builder seed(long s) { req.seed = s; return this; }
        public Builder stream(boolean s) { req.stream = s; return this; }
        public Builder user(String u) { req.user = u; return this; }
        public Builder responseFormat(ResponseFormat rf) { req.responseFormat = rf; return this; }
        public Builder stop(StopSeq s) { req.stop = s; return this; }
        public Builder tools(List<ToolDefinition> ts) { req.tools = ts; return this; }
        public Builder toolChoice(ToolChoice tc) { req.toolChoice = tc; return this; }
        public Builder extra(Map<String, Object> e) { req.extra = e; return this; }

        public ChatRequest build() { return req; }
    }
}

/**
 * 响应格式（对应 Rust ResponseFormat）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class ResponseFormat {
    @JsonProperty("type")
    public String type;

    public ResponseFormat(String type) { this.type = type; }
}

/**
 * 停止词（单个或多个，对应 Rust StopSeq untagged enum）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class StopSeq {
    public String single;
    public List<String> multiple;

    public StopSeq(String single) { this.single = single; }
    public StopSeq(List<String> multiple) { this.multiple = multiple; }
}

/**
 * 工具定义（对应 Rust ToolDefinition）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class ToolDefinition {
    @JsonProperty("type")
    public String toolType;
    public FunctionDefinition function;

    public static ToolDefinition function(FunctionDefinition func) {
        ToolDefinition td = new ToolDefinition();
        td.toolType = "function";
        td.function = func;
        return td;
    }

    public static ToolDefinition webSearch() {
        ToolDefinition td = new ToolDefinition();
        td.toolType = "web_search";
        return td;
    }
}

/**
 * 函数定义（对应 Rust FunctionDefinition）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class FunctionDefinition {
    public String name;
    public String description;
    public Object parameters; // JSON Schema object

    public FunctionDefinition(String name, String description, Object parameters) {
        this.name = name;
        this.description = description;
        this.parameters = parameters;
    }
}

/**
 * 工具调用（模型生成，对应 Rust ToolCall）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class ToolCall {
    public String id;
    @JsonProperty("type")
    public String toolType;
    public ToolCallFunction function;
}

/**
 * 工具调用的函数部分（对应 Rust ToolCallFunction）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class ToolCallFunction {
    public String name;
    public String arguments;
}

/**
 * 工具选择策略（对应 Rust ToolChoice）。
 */
class ToolChoice {
    public static final ToolChoice NONE = new ToolChoice("none");
    public static final ToolChoice AUTO = new ToolChoice("auto");
    public static final ToolChoice REQUIRED = new ToolChoice("required");

    public final String value;

    public ToolChoice(String value) { this.value = value; }
}
