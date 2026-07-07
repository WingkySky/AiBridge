package io.aibridge;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * 对话请求（对应 Rust {@code ChatRequest}）。
 *
 * <p>用 {@link Builder} 链式构造。仅声明 hello world 所需的核心字段，
 * 其余可选参数（temperature/top_p/tools 等）按需扩展。
 *
 * <p>序列化为 JSON 后通过 FFI 边界传给 {@code aibridge_client_chat} /
 * {@code aibridge_client_chat_stream}。
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
    /** 是否流式输出（chat_stream 自动设置） */
    public Boolean stream;

    /** 默认构造（Jackson 反序列化需要） */
    public ChatRequest() {
    }

    public ChatRequest(String model, List<ChatMessage> messages) {
        this.model = model;
        this.messages = messages;
    }

    /** 创建 Builder */
    public static Builder builder(String model, List<ChatMessage> messages) {
        return new Builder(model, messages);
    }

    /** 链式构造器 */
    public static class Builder {
        private final ChatRequest req;

        public Builder(String model, List<ChatMessage> messages) {
            this.req = new ChatRequest(model, messages);
        }

        public Builder temperature(double t) {
            req.temperature = t;
            return this;
        }

        public Builder maxTokens(int n) {
            req.maxTokens = n;
            return this;
        }

        public Builder stream(boolean s) {
            req.stream = s;
            return this;
        }

        public ChatRequest build() {
            return req;
        }
    }
}
