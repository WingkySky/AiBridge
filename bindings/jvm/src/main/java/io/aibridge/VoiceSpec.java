package io.aibridge;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.List;

/**
 * 音色规格（对应 Rust {@code VoiceSpec}，候选列表用于自动降级）。
 *
 * <p>序列化形如 {@code {"voices":["alloy"]}}。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class VoiceSpec {

    /** 音色列表（至少 1 个；多个时启用 fallback 降级） */
    public List<String> voices;

    public VoiceSpec() {
    }

    public VoiceSpec(List<String> voices) {
        this.voices = voices;
    }

    /** 单个音色 */
    public static VoiceSpec single(String voice) {
        return new VoiceSpec(Arrays.asList(voice));
    }

    /** 候选音色列表 */
    public static VoiceSpec multiple(String... voices) {
        return new VoiceSpec(Arrays.asList(voices));
    }

    /** 主音色（列表第一个） */
    @JsonProperty("primary")
    public String primary() {
        return voices == null || voices.isEmpty() ? null : voices.get(0);
    }
}
