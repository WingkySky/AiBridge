package io.aibridge;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * 视频生成模式。
 */
class VideoMode {
    public static final String TEXT2VIDEO = "text2video";
    public static final String IMAGE2VIDEO = "image2video";
    public static final String VIDEO2VIDEO = "video2video";
}

/**
 * 视频生成请求（对应 Rust VideoRequest）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class VideoRequest {
    public String model;
    public String prompt;
    public Integer width;
    public Integer height;
    @JsonProperty("num_frames")
    public Integer numFrames;
    @JsonProperty("frame_rate")
    public Integer frameRate;
    public String mode;
    public Integer duration;
    @JsonProperty("aspect_ratio")
    public String aspectRatio;
    public String resolution;
    @JsonProperty("reference_images")
    public List<Object> referenceImages;
    @JsonProperty("reference_videos")
    public List<Object> referenceVideos;
    @JsonProperty("first_frame")
    public Object firstFrame;
    @JsonProperty("last_frame")
    public Object lastFrame;
    public List<Map<String, Object>> keyframes;
    public String style;
    @JsonProperty("camera_motion")
    public String cameraMotion;
    @JsonProperty("motion_strength")
    public Double motionStrength;
    @JsonProperty("negative_prompt")
    public String negativePrompt;
    public Long seed;
    public Integer steps;
    @JsonProperty("cfg_scale")
    public Double cfgScale;
    @JsonProperty("with_audio")
    public Boolean withAudio;
    public Boolean watermark;
    public Map<String, Object> extra;

    public VideoRequest() {}

    public VideoRequest(String model, String prompt) {
        this.model = model;
        this.prompt = prompt;
    }

    public static Builder builder(String model, String prompt) {
        return new Builder(model, prompt);
    }

    public static class Builder {
        private final VideoRequest req;

        public Builder(String model, String prompt) {
            this.req = new VideoRequest(model, prompt);
        }

        public Builder width(int w) { req.width = w; return this; }
        public Builder height(int h) { req.height = h; return this; }
        public Builder numFrames(int n) { req.numFrames = n; return this; }
        public Builder frameRate(int f) { req.frameRate = f; return this; }
        public Builder mode(String m) { req.mode = m; return this; }
        public Builder duration(int d) { req.duration = d; return this; }
        public Builder aspectRatio(String a) { req.aspectRatio = a; return this; }
        public Builder resolution(String r) { req.resolution = r; return this; }
        public Builder referenceImages(List<Object> imgs) { req.referenceImages = imgs; return this; }
        public Builder referenceVideos(List<Object> vids) { req.referenceVideos = vids; return this; }
        public Builder firstFrame(Object f) { req.firstFrame = f; return this; }
        public Builder lastFrame(Object l) { req.lastFrame = l; return this; }
        public Builder keyframes(List<Map<String, Object>> k) { req.keyframes = k; return this; }
        public Builder style(String s) { req.style = s; return this; }
        public Builder cameraMotion(String c) { req.cameraMotion = c; return this; }
        public Builder motionStrength(double m) { req.motionStrength = m; return this; }
        public Builder negativePrompt(String n) { req.negativePrompt = n; return this; }
        public Builder seed(long s) { req.seed = s; return this; }
        public Builder steps(int s) { req.steps = s; return this; }
        public Builder cfgScale(double c) { req.cfgScale = c; return this; }
        public Builder withAudio(boolean w) { req.withAudio = w; return this; }
        public Builder watermark(boolean w) { req.watermark = w; return this; }
        public Builder extra(Map<String, Object> e) { req.extra = e; return this; }

        public VideoRequest build() { return req; }
    }
}

/**
 * 任务状态。
 */
class TaskStatus {
    public static final String PENDING = "pending";
    public static final String PROCESSING = "processing";
    public static final String SUCCESS = "success";
    public static final String FAILED = "failed";
}

/**
 * 视频任务信息（对应 Rust VideoTask）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class VideoTask {
    @JsonProperty("task_id")
    public String taskId;
    public String model;
    public String status;
    @JsonProperty("created_at")
    public long createdAt;
}

/**
 * 视频任务状态（对应 Rust VideoStatus）。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
class VideoStatus {
    @JsonProperty("task_id")
    public String taskId;
    public String status;
    @JsonProperty("video_url")
    public String videoUrl;
    public Integer progress;
    public String error;
    @JsonProperty("created_at")
    public Long createdAt;
    @JsonProperty("updated_at")
    public Long updatedAt;
}
