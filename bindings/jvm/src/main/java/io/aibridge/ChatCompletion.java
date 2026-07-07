package io.aibridge;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * 对话完成结果（对应 Rust {@code ChatCompletion}）。
 *
 * <p>由 {@code aibridge_client_chat} 的 {@code out_response_json} 反序列化得到。
 * {@link JsonIgnoreProperties} 忽略未知字段，保证前向兼容。
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ChatCompletion {

    /** 响应 ID */
    public String id;
    /** 对象类型 */
    public String object;
    /** 创建时间戳 */
    public long created;
    /** 使用的模型 */
    public String model;
    /** 回复选项列表 */
    public List<ChatChoice> choices;
    /** Token 使用统计（可选） */
    public ChatUsage usage;
    /** 服务层级（可选） */
    @JsonProperty("service_tier")
    public String serviceTier;
    /** 系统指纹（可选） */
    @JsonProperty("system_fingerprint")
    public String systemFingerprint;

    public ChatCompletion() {
    }
}
