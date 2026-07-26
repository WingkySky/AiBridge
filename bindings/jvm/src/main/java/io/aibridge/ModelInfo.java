package io.aibridge;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * 模型信息（对应 Rust ModelInfo）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class ModelInfo {
    public String id;
    public String name;
    @JsonProperty("description")
    public String description;
    public String provider;
    public String type; // chat/image/video/audio
    @JsonProperty("capabilities")
    public List<String> capabilities;
}

/**
 * 供应商信息（对应 Rust ProviderInfo）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class ProviderInfo {
    public String name;
    @JsonProperty("description")
    public String description;
    public List<ModelInfo> models;
}

/**
 * 音色信息（对应 Rust VoiceInfo）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class VoiceInfo {
    public String id;
    public String name;
    public String gender;
    public List<String> languages;
    @JsonProperty("sample_url")
    public String sampleUrl;
    public String description;
}
