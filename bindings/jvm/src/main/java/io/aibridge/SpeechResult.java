package io.aibridge;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * 文字转语音结果（对应 Rust {@code SpeechResult}）。
 *
 * <p>注意：Rust 侧 {@code audio_data} 被 {@code #[serde(skip)]}，不参与 JSON 序列化，
 * 二进制音频通过 FFI 的 {@code aibridge_bytes_t} 单独传递。本 POJO 仅承载 meta JSON
 * 字段，二进制由 {@link Client#speech} 单独返回并封装到 {@link SpeechResultFull}。
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class SpeechResult {

    /** 音频 URL（部分 Provider 返回） */
    @JsonProperty("audio_url")
    public String audioUrl;
    /** 音频 Base64 编码 */
    @JsonProperty("audio_base64")
    public String audioBase64;
    /** 音频 MIME 类型 */
    @JsonProperty("content_type")
    public String contentType;
    /** 音频格式（mp3/wav/opus 等） */
    public String format;
    /** 估计音频时长（秒） */
    public Double duration;
    /** 使用的模型 ID */
    public String model;

    public SpeechResult() {
    }
}
