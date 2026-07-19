//! 视频生成数据模型
//!
//! 定义视频生成相关的 serde struct。
//! 对应 Python v1 (agn-sdk) 的 `agn/models/video.py`。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::model::common::{TaskStatus, VideoMode};
use crate::model::image::FileInput;

/// 视频生成请求
///
/// 对应设计文档第 6 节，合并 Python v1 `VideoGenerationOptions` + `VideoTaskCreate`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoRequest {
    /// 模型名称
    pub model: String,
    /// 提示词
    pub prompt: String,
    /// 视频宽度（必须是 8 的倍数）
    #[serde(default = "default_width", skip_serializing_if = "is_default_width")]
    pub width: u32,
    /// 视频高度（必须是 8 的倍数）
    #[serde(default = "default_height", skip_serializing_if = "is_default_height")]
    pub height: u32,
    /// 帧数（部分模型需要）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_frames: Option<u32>,
    /// 帧率
    #[serde(
        default = "default_frame_rate",
        skip_serializing_if = "is_default_frame_rate"
    )]
    pub frame_rate: u32,
    /// 生成模式
    #[serde(default)]
    pub mode: VideoMode,
    /// 视频时长（秒），部分模型用此字段替代 num_frames
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<u32>,
    /// 宽高比，如 "16:9" / "9:16" / "1:1"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    /// 分辨率档位，如 "720p" / "1080p"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    /// 参考图像列表（图生视频）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reference_images: Vec<FileInput>,
    /// 参考视频列表（视频生视频）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reference_videos: Vec<FileInput>,
    /// 首帧图片
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_frame: Option<FileInput>,
    /// 尾帧图片
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_frame: Option<FileInput>,
    /// 关键帧列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keyframes: Vec<HashMap<String, serde_json::Value>>,
    /// 视频风格
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    /// 镜头运动
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_motion: Option<String>,
    /// 运动强度（0-10）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motion_strength: Option<f64>,
    /// 负面提示词
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// 随机种子
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// 推理步数
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// CFG Scale
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f64>,
    /// 是否生成音频
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub with_audio: Option<bool>,
    /// 是否添加水印
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub watermark: Option<bool>,
    /// 厂商特有参数透传
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

impl VideoRequest {
    /// 创建 Builder
    pub fn builder(model: impl Into<String>, prompt: impl Into<String>) -> VideoRequestBuilder {
        VideoRequestBuilder {
            inner: VideoRequest {
                model: model.into(),
                prompt: prompt.into(),
                width: default_width(),
                height: default_height(),
                num_frames: None,
                frame_rate: default_frame_rate(),
                mode: VideoMode::default(),
                duration: None,
                aspect_ratio: None,
                resolution: None,
                reference_images: Vec::new(),
                reference_videos: Vec::new(),
                first_frame: None,
                last_frame: None,
                keyframes: Vec::new(),
                style: None,
                camera_motion: None,
                motion_strength: None,
                negative_prompt: None,
                seed: None,
                steps: None,
                cfg_scale: None,
                with_audio: None,
                watermark: None,
                extra: HashMap::new(),
            },
        }
    }
}

/// `VideoRequest` 的 Builder
#[derive(Debug, Clone)]
pub struct VideoRequestBuilder {
    inner: VideoRequest,
}

impl VideoRequestBuilder {
    pub fn width(mut self, w: u32) -> Self {
        self.inner.width = w;
        self
    }
    pub fn height(mut self, h: u32) -> Self {
        self.inner.height = h;
        self
    }
    pub fn num_frames(mut self, n: u32) -> Self {
        self.inner.num_frames = Some(n);
        self
    }
    pub fn frame_rate(mut self, f: u32) -> Self {
        self.inner.frame_rate = f;
        self
    }
    pub fn mode(mut self, m: VideoMode) -> Self {
        self.inner.mode = m;
        self
    }
    pub fn duration(mut self, d: u32) -> Self {
        self.inner.duration = Some(d);
        self
    }
    pub fn aspect_ratio(mut self, a: impl Into<String>) -> Self {
        self.inner.aspect_ratio = Some(a.into());
        self
    }
    pub fn resolution(mut self, r: impl Into<String>) -> Self {
        self.inner.resolution = Some(r.into());
        self
    }
    pub fn reference_images(mut self, imgs: Vec<FileInput>) -> Self {
        self.inner.reference_images = imgs;
        self
    }
    pub fn reference_videos(mut self, vids: Vec<FileInput>) -> Self {
        self.inner.reference_videos = vids;
        self
    }
    pub fn first_frame(mut self, f: FileInput) -> Self {
        self.inner.first_frame = Some(f);
        self
    }
    pub fn last_frame(mut self, f: FileInput) -> Self {
        self.inner.last_frame = Some(f);
        self
    }
    pub fn keyframes(mut self, kfs: Vec<HashMap<String, serde_json::Value>>) -> Self {
        self.inner.keyframes = kfs;
        self
    }
    pub fn style(mut self, s: impl Into<String>) -> Self {
        self.inner.style = Some(s.into());
        self
    }
    pub fn camera_motion(mut self, c: impl Into<String>) -> Self {
        self.inner.camera_motion = Some(c.into());
        self
    }
    pub fn motion_strength(mut self, m: f64) -> Self {
        self.inner.motion_strength = Some(m);
        self
    }
    pub fn negative_prompt(mut self, n: impl Into<String>) -> Self {
        self.inner.negative_prompt = Some(n.into());
        self
    }
    pub fn seed(mut self, s: u64) -> Self {
        self.inner.seed = Some(s);
        self
    }
    pub fn steps(mut self, s: u32) -> Self {
        self.inner.steps = Some(s);
        self
    }
    pub fn cfg_scale(mut self, c: f64) -> Self {
        self.inner.cfg_scale = Some(c);
        self
    }
    pub fn with_audio(mut self, w: bool) -> Self {
        self.inner.with_audio = Some(w);
        self
    }
    pub fn watermark(mut self, w: bool) -> Self {
        self.inner.watermark = Some(w);
        self
    }
    pub fn extra(mut self, k: impl Into<String>, v: impl Into<serde_json::Value>) -> Self {
        self.inner.extra.insert(k.into(), v.into());
        self
    }
    pub fn build(self) -> VideoRequest {
        self.inner
    }
}

/// 视频任务信息（创建任务后的返回）
///
/// 对应 Python v1 `VideoTask`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoTask {
    /// 任务 ID（用于轮询状态）
    pub task_id: String,
    /// 使用的模型
    pub model: String,
    /// 任务状态
    #[serde(default = "default_status")]
    pub status: TaskStatus,
    /// 创建时间戳
    pub created_at: u64,
}

/// 视频任务状态（轮询返回）
///
/// 对应 Python v1 `VideoStatus`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoStatus {
    /// 任务 ID
    pub task_id: String,
    /// 任务状态
    pub status: TaskStatus,
    /// 视频 URL（成功时）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_url: Option<String>,
    /// 进度 0-100
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<u32>,
    /// 错误信息（失败时）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// 创建时间戳
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<u64>,
    /// 更新时间戳
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<u64>,
}

fn default_width() -> u32 {
    1280
}

fn is_default_width(w: &u32) -> bool {
    *w == 1280
}

fn default_height() -> u32 {
    720
}

fn is_default_height(h: &u32) -> bool {
    *h == 720
}

fn default_frame_rate() -> u32 {
    24
}

fn is_default_frame_rate(f: &u32) -> bool {
    *f == 24
}

fn default_status() -> TaskStatus {
    TaskStatus::Pending
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_request_builder_defaults() {
        let req = VideoRequest::builder("seedance-2.0", "a cat walking").build();
        assert_eq!(req.model, "seedance-2.0");
        assert_eq!(req.width, 1280);
        assert_eq!(req.height, 720);
        assert_eq!(req.frame_rate, 24);
        assert_eq!(req.mode, VideoMode::Text2Video);
    }

    #[test]
    fn video_request_builder_chained() {
        let req = VideoRequest::builder("seedance-2.0", "a cat")
            .width(1920)
            .height(1080)
            .duration(5)
            .aspect_ratio("16:9")
            .seed(123)
            .with_audio(true)
            .build();
        assert_eq!(req.width, 1920);
        assert_eq!(req.height, 1080);
        assert_eq!(req.duration, Some(5));
        assert_eq!(req.with_audio, Some(true));
    }

    #[test]
    fn video_request_skip_defaults() {
        let req = VideoRequest::builder("seedance-2.0", "a cat").build();
        let json = serde_json::to_string(&req).unwrap();
        // 默认值被跳过
        assert!(!json.contains("\"width\""));
        assert!(!json.contains("\"height\""));
        assert!(!json.contains("\"frame_rate\""));
        assert!(!json.contains("negative_prompt"));
        assert!(!json.contains("extra"));
    }

    #[test]
    fn video_request_image2video_mode() {
        let req = VideoRequest::builder("kling-v1", "animate this")
            .mode(VideoMode::Image2Video)
            .reference_images(vec![FileInput::url("https://example.com/a.png")])
            .build();
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"mode\":\"image2video\""));
        assert!(json.contains("\"reference_images\""));
    }

    #[test]
    fn video_task_deserialize() {
        let json = serde_json::json!({
            "task_id": "t-1",
            "model": "seedance-2.0",
            "status": "pending",
            "created_at": 1700000000
        });
        let t: VideoTask = serde_json::from_value(json).unwrap();
        assert_eq!(t.task_id, "t-1");
        assert_eq!(t.model, "seedance-2.0");
        assert_eq!(t.status, TaskStatus::Pending);
    }

    #[test]
    fn video_status_success() {
        let json = serde_json::json!({
            "task_id": "t-1",
            "status": "success",
            "video_url": "https://example.com/v.mp4",
            "progress": 100
        });
        let s: VideoStatus = serde_json::from_value(json).unwrap();
        assert_eq!(s.status, TaskStatus::Success);
        assert_eq!(s.video_url.as_deref(), Some("https://example.com/v.mp4"));
        assert_eq!(s.progress, Some(100));
    }

    #[test]
    fn video_status_failed_with_error() {
        let json = serde_json::json!({
            "task_id": "t-1",
            "status": "failed",
            "error": "content policy violation"
        });
        let s: VideoStatus = serde_json::from_value(json).unwrap();
        assert_eq!(s.status, TaskStatus::Failed);
        assert_eq!(s.error.as_deref(), Some("content policy violation"));
    }

    #[test]
    fn video_status_processing_with_progress() {
        let json = serde_json::json!({
            "task_id": "t-1",
            "status": "processing",
            "progress": 45
        });
        let s: VideoStatus = serde_json::from_value(json).unwrap();
        assert_eq!(s.status, TaskStatus::Processing);
        assert_eq!(s.progress, Some(45));
    }

    #[test]
    fn video_request_new_fields_builder() {
        let req = VideoRequest::builder("seedance-2.0", "restyle this")
            .mode(VideoMode::Video2Video)
            .reference_videos(vec![FileInput::url("https://example.com/v.mp4")])
            .keyframes(vec![HashMap::from([(
                "frame".to_string(),
                serde_json::json!("https://example.com/kf.png"),
            )])])
            .style("anime")
            .build();
        assert_eq!(req.mode, VideoMode::Video2Video);
        assert_eq!(req.reference_videos.len(), 1);
        assert_eq!(req.keyframes.len(), 1);
        assert_eq!(req.style.as_deref(), Some("anime"));
    }

    #[test]
    fn video_mode_video2video_serde() {
        // 对齐 Python v1 `VideoOptions.mode` Literal 中的 "video2video" 取值
        let json = serde_json::to_string(&VideoMode::Video2Video).unwrap();
        assert_eq!(json, "\"video2video\"");
        let back: VideoMode = serde_json::from_str("\"video2video\"").unwrap();
        assert_eq!(back, VideoMode::Video2Video);
    }

    #[test]
    fn video_request_new_fields_serde() {
        let req = VideoRequest::builder("seedance-2.0", "restyle this")
            .reference_videos(vec![FileInput::url("https://example.com/v.mp4")])
            .keyframes(vec![HashMap::from([("t".to_string(), serde_json::json!(0))])])
            .style("anime")
            .build();
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"reference_videos\":[\"https://example.com/v.mp4\"]"));
        assert!(json.contains("\"keyframes\":[{\"t\":0}]"));
        assert!(json.contains("\"style\":\"anime\""));
        // 反序列化回读
        let back: VideoRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.reference_videos.len(), 1);
        assert_eq!(back.keyframes.len(), 1);
        assert_eq!(back.style.as_deref(), Some("anime"));
    }

    #[test]
    fn video_request_new_fields_skip_when_empty() {
        let req = VideoRequest::builder("seedance-2.0", "a cat").build();
        let json = serde_json::to_string(&req).unwrap();
        // 新字段为空/None 时不出现在序列化结果中
        assert!(!json.contains("reference_videos"));
        assert!(!json.contains("keyframes"));
        assert!(!json.contains("style"));
    }
}
