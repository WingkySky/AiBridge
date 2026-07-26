package io.aibridge;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * 图像生成请求（对应 Rust ImageRequest）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ImageRequest {
    public String model;
    public String prompt;
    public String size;
    public Integer width;
    public Integer height;
    @JsonProperty("aspect_ratio")
    public String aspectRatio;
    public Integer n;
    public String quality;
    public String style;
    @JsonProperty("negative_prompt")
    public String negativePrompt;
    @JsonProperty("negative_prompts")
    public List<String> negativePrompts;
    public Long seed;
    public Integer steps;
    @JsonProperty("cfg_scale")
    public Double cfgScale;
    public String sampler;
    public String scheduler;
    @JsonProperty("response_format")
    public String responseFormat;
    @JsonProperty("output_format")
    public String outputFormat;
    @JsonProperty("reference_images")
    public List<Object> referenceImages; // FileInput (untagged: String or Map)
    @JsonProperty("reference_strength")
    public Double referenceStrength;
    public Object mask; // FileInput
    @JsonProperty("edit_mode")
    public String editMode;
    public Map<String, Object> extra;

    public ImageRequest() {}

    public ImageRequest(String model, String prompt) {
        this.model = model;
        this.prompt = prompt;
    }

    public static Builder builder(String model, String prompt) {
        return new Builder(model, prompt);
    }

    public static class Builder {
        private final ImageRequest req;

        public Builder(String model, String prompt) {
            this.req = new ImageRequest(model, prompt);
        }

        public Builder size(String s) { req.size = s; return this; }
        public Builder width(int w) { req.width = w; return this; }
        public Builder height(int h) { req.height = h; return this; }
        public Builder aspectRatio(String a) { req.aspectRatio = a; return this; }
        public Builder n(int n) { req.n = n; return this; }
        public Builder quality(String q) { req.quality = q; return this; }
        public Builder style(String s) { req.style = s; return this; }
        public Builder negativePrompt(String n) { req.negativePrompt = n; return this; }
        public Builder negativePrompts(List<String> n) { req.negativePrompts = n; return this; }
        public Builder seed(long s) { req.seed = s; return this; }
        public Builder steps(int s) { req.steps = s; return this; }
        public Builder cfgScale(double c) { req.cfgScale = c; return this; }
        public Builder sampler(String s) { req.sampler = s; return this; }
        public Builder scheduler(String s) { req.scheduler = s; return this; }
        public Builder responseFormat(String r) { req.responseFormat = r; return this; }
        public Builder outputFormat(String o) { req.outputFormat = o; return this; }
        public Builder referenceImages(List<Object> imgs) { req.referenceImages = imgs; return this; }
        public Builder referenceStrength(double r) { req.referenceStrength = r; return this; }
        public Builder mask(Object m) { req.mask = m; return this; }
        public Builder editMode(String e) { req.editMode = e; return this; }
        public Builder extra(Map<String, Object> e) { req.extra = e; return this; }

        public ImageRequest build() { return req; }
    }
}

/**
 * 图像数据（对应 Rust ImageData）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class ImageData {
    @JsonProperty("url")
    public String url;
    @JsonProperty("b64_json")
    public String b64Json;
    @JsonProperty("revised_prompt")
    public String revisedPrompt;
}

/**
 * 图像生成结果（对应 Rust ImageResult）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class ImageResult {
    public String id;
    public String object;
    public long created;
    public String model;
    public List<ImageData> data;
}
