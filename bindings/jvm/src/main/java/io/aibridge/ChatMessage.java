package io.aibridge;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * 对话消息（对应 Rust {@code ChatMessage} tagged enum，{@code role} 作为 tag）。
 *
 * <p>Rust 侧用 {@code #[serde(tag = "role", rename_all = "lowercase")]}，
 * 故序列化形如 {@code {"role":"user","content":"..."}}。
 *
 * <p>Java 侧用单一 POJO + {@code role} 字段简化（避免 tagged enum 的复杂映射）。
 * 当前覆盖 system/user/assistant 三种角色，足够 hello world 使用。
 */
public class ChatMessage {

    /** 角色：system / user / assistant / tool */
    public String role;
    /** 消息内容（user 的纯文本；assistant 的回复） */
    public String content;
    /** 发送者名称（可选） */
    public String name;

    /** 默认构造（Jackson 反序列化需要） */
    public ChatMessage() {
    }

    @JsonCreator
    public ChatMessage(
            @JsonProperty("role") String role,
            @JsonProperty("content") String content) {
        this.role = role;
        this.content = content;
    }

    /** 创建系统消息 */
    public static ChatMessage system(String content) {
        return new ChatMessage("system", content);
    }

    /** 创建用户消息（纯文本） */
    public static ChatMessage user(String content) {
        return new ChatMessage("user", content);
    }

    /** 创建助手消息 */
    public static ChatMessage assistant(String content) {
        return new ChatMessage("assistant", content);
    }
}
