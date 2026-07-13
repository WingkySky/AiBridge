//! 图像生成数据模型
//!
//! 定义图像生成相关的 serde struct。
//! 对应 Python v1 (agn-sdk) 的 `agn/models/image.py`。
//!
//! 设计要点（与设计文档第 6 节一致）：
//! - 去掉 Python 的 `ImageOptions` 中间层，改为 `ImageRequest::builder()`
//! - `FileInput` 抽象图像输入（Path/Url/Bytes/Base64），适配多 provider

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// 图像生成请求
///
/// 对应设计文档第 6 节，合并 Python v1 `ImageGenerationOptions` + 请求参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRequest {
    /// 模型名称
    pub model: String,
    /// 提示词
    pub prompt: String,
    /// 图像尺寸（如 "1024x1024"），或使用 width/height
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// 宽度（像素）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// 高度（像素）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    /// 画面比例（如 "16:9"）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    /// 生成数量（1-10）
    #[serde(default = "default_n", skip_serializing_if = "is_default_n")]
    pub n: u32,
    /// 生成质量（"standard" / "hd" / "ultra"）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    /// 图像风格
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    /// 负面提示词
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// 随机种子
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// 推理步数
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// CFG Scale（提示词相关性）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f64>,
    /// 响应格式（"url" / "b64_json"）
    #[serde(
        default = "default_response_format",
        skip_serializing_if = "is_default_response_format"
    )]
    pub response_format: String,
    /// 输出图片格式（"png" / "jpeg" / "webp"）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    /// 参考图（图生图/IP-Adapter），FileInput 列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reference_images: Vec<FileInput>,
    /// 遮罩图片（局部重绘）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<FileInput>,
    /// 编辑模式（inpaint / outpaint / variation）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit_mode: Option<String>,
    /// 厂商特有参数透传
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

impl ImageRequest {
    /// 创建 Builder
    pub fn builder(model: impl Into<String>, prompt: impl Into<String>) -> ImageRequestBuilder {
        ImageRequestBuilder {
            inner: ImageRequest {
                model: model.into(),
                prompt: prompt.into(),
                size: None,
                width: None,
                height: None,
                aspect_ratio: None,
                n: default_n(),
                quality: None,
                style: None,
                negative_prompt: None,
                seed: None,
                steps: None,
                cfg_scale: None,
                response_format: default_response_format(),
                output_format: None,
                reference_images: Vec::new(),
                mask: None,
                edit_mode: None,
                extra: HashMap::new(),
            },
        }
    }
}

/// `ImageRequest` 的 Builder
#[derive(Debug, Clone)]
pub struct ImageRequestBuilder {
    inner: ImageRequest,
}

impl ImageRequestBuilder {
    pub fn size(mut self, s: impl Into<String>) -> Self {
        self.inner.size = Some(s.into());
        self
    }
    pub fn width(mut self, w: u32) -> Self {
        self.inner.width = Some(w);
        self
    }
    pub fn height(mut self, h: u32) -> Self {
        self.inner.height = Some(h);
        self
    }
    pub fn aspect_ratio(mut self, a: impl Into<String>) -> Self {
        self.inner.aspect_ratio = Some(a.into());
        self
    }
    pub fn n(mut self, n: u32) -> Self {
        self.inner.n = n;
        self
    }
    pub fn quality(mut self, q: impl Into<String>) -> Self {
        self.inner.quality = Some(q.into());
        self
    }
    pub fn style(mut self, s: impl Into<String>) -> Self {
        self.inner.style = Some(s.into());
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
    pub fn response_format(mut self, r: impl Into<String>) -> Self {
        self.inner.response_format = r.into();
        self
    }
    pub fn output_format(mut self, o: impl Into<String>) -> Self {
        self.inner.output_format = Some(o.into());
        self
    }
    pub fn reference_images(mut self, imgs: Vec<FileInput>) -> Self {
        self.inner.reference_images = imgs;
        self
    }
    pub fn mask(mut self, m: FileInput) -> Self {
        self.inner.mask = Some(m);
        self
    }
    pub fn edit_mode(mut self, e: impl Into<String>) -> Self {
        self.inner.edit_mode = Some(e.into());
        self
    }
    pub fn extra(mut self, k: impl Into<String>, v: impl Into<serde_json::Value>) -> Self {
        self.inner.extra.insert(k.into(), v.into());
        self
    }
    pub fn build(self) -> ImageRequest {
        self.inner
    }
}

/// 文件输入
///
/// 对应设计文档第 6 节 `FileInput`。
/// 抽象各种图像输入形式（路径、URL、字节、Base64）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FileInput {
    /// 本地文件路径
    Path(String),
    /// 远程 URL
    Url(String),
    /// 原始字节
    Bytes(Vec<u8>),
    /// Base64 编码字符串
    Base64(String),
}

impl FileInput {
    /// 从 URL 创建
    pub fn url(u: impl Into<String>) -> Self {
        Self::Url(u.into())
    }

    /// 从路径创建
    pub fn path(p: impl Into<String>) -> Self {
        Self::Path(p.into())
    }

    /// 从 Base64 创建
    pub fn base64(b: impl Into<String>) -> Self {
        Self::Base64(b.into())
    }

    /// 从字节创建
    pub fn bytes(b: Vec<u8>) -> Self {
        Self::Bytes(b)
    }
}

/// 图像数据
///
/// 对应 Python v1 `ImageData`。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImageData {
    /// 图像 URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Base64 编码的图像
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    /// 修改后的提示词（如模型优化过）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

/// 图像生成结果
///
/// 对应 Python v1 `ImageGenerationResult`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageResult {
    /// 响应 ID
    pub id: String,
    /// 对象类型
    #[serde(default = "default_object_image")]
    pub object: String,
    /// 创建时间戳
    pub created: u64,
    /// 使用的模型
    pub model: String,
    /// 生成的图像列表
    pub data: Vec<ImageData>,
}

fn default_object_image() -> String {
    "image.generation".into()
}

fn default_n() -> u32 {
    1
}

fn is_default_n(n: &u32) -> bool {
    *n == 1
}

fn default_response_format() -> String {
    "url".into()
}

fn is_default_response_format(s: &str) -> bool {
    s == "url"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_request_builder_defaults() {
        let req = ImageRequest::builder("dall-e-3", "a cat").build();
        assert_eq!(req.model, "dall-e-3");
        assert_eq!(req.prompt, "a cat");
        assert_eq!(req.n, 1);
        assert_eq!(req.response_format, "url");
    }

    #[test]
    fn image_request_builder_chained() {
        let req = ImageRequest::builder("dall-e-3", "a cat")
            .size("1024x1024")
            .n(2)
            .quality("hd")
            .style("vivid")
            .seed(42)
            .build();
        assert_eq!(req.size.as_deref(), Some("1024x1024"));
        assert_eq!(req.n, 2);
        assert_eq!(req.quality.as_deref(), Some("hd"));
        assert_eq!(req.seed, Some(42));
    }

    #[test]
    fn image_request_skip_defaults() {
        let req = ImageRequest::builder("dall-e-3", "a cat").build();
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("\"n\"")); // n=1 被跳过
        assert!(!json.contains("\"response_format\"")); // "url" 被跳过
        assert!(!json.contains("negative_prompt"));
        assert!(!json.contains("extra"));
    }

    #[test]
    fn image_request_with_reference_images() {
        let req = ImageRequest::builder("flux", "edit this")
            .reference_images(vec![FileInput::url("https://example.com/a.png")])
            .mask(FileInput::base64("aGVsbG8="))
            .edit_mode("inpaint")
            .build();
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"reference_images\""));
        assert!(json.contains("\"mask\""));
        assert!(json.contains("\"edit_mode\":\"inpaint\""));
    }

    #[test]
    fn file_input_url_serde() {
        let f = FileInput::url("https://example.com/x.png");
        let json = serde_json::to_string(&f).unwrap();
        assert_eq!(json, "\"https://example.com/x.png\"");
    }

    #[test]
    fn file_input_base64_serde() {
        let f = FileInput::base64("aGVsbG8=");
        let json = serde_json::to_string(&f).unwrap();
        assert_eq!(json, "\"aGVsbG8=\"");
    }

    #[test]
    fn image_result_deserialize() {
        let json = serde_json::json!({
            "id": "img-1",
            "object": "image.generation",
            "created": 1700000000,
            "model": "dall-e-3",
            "data": [{
                "url": "https://example.com/result.png",
                "revised_prompt": "a cute cat"
            }]
        });
        let r: ImageResult = serde_json::from_value(json).unwrap();
        assert_eq!(r.id, "img-1");
        assert_eq!(r.model, "dall-e-3");
        assert_eq!(r.data.len(), 1);
        assert_eq!(
            r.data[0].url.as_deref(),
            Some("https://example.com/result.png")
        );
    }

    #[test]
    fn image_data_default_empty() {
        let d = ImageData::default();
        assert!(d.url.is_none());
        assert!(d.b64_json.is_none());
    }

    #[test]
    fn file_input_constructors() {
        let _ = FileInput::path("/tmp/x.png");
        let _ = FileInput::bytes(vec![1, 2, 3]);
        let _ = FileInput::url("https://x");
        let _ = FileInput::base64("aGk=");
    }
}
