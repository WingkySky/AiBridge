//! AIBridge Node.js 绑定（napi-rs）
//!
//! 直连 aibridge-core，原生 Promise / AsyncIterable 流式。
//! 由 napi-rs 构建为 npm 包 `aibridge`。
//!
//! 设计要点（对应设计文档第 8 节 JS 桥接）：
//! - `#[napi] async fn` 自动桥接为 JS `Promise`（基于 napi 全局 tokio runtime）
//! - 复杂请求参数通过 `serde_json::Value` 中转（JS Object → Value → core serde struct）
//! - 返回值用 `#[napi(object)]` struct，JS 拿到原生对象，类型清晰
//! - 流式：`chatStream` 返回 `ChatStreamIterator`，内部 spawn tokio task 消费 core 的
//!   `BoxStream`，通过 `tokio::sync::mpsc` channel 推送 chunk；JS 侧通过 `[Symbol.asyncIterator]`
//!   支持 `for await...of`（见 index.js 的包装）
//! - 错误映射：`AibridgeError` → `napi::Error`，reason 编码为 `[code] message`，
//!   JS 侧 index.js 解析出 `.code` 属性（napi 的 Error code 字段无法承载自定义业务 code）

use std::sync::Arc;

use futures::stream::StreamExt;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;
use tokio::sync::{mpsc, Mutex};

use aibridge_core::client::Client as CoreClient;
use aibridge_core::config::ClientOptions;
use aibridge_core::error::AibridgeError;
use aibridge_core::model::audio::{SpeechRequest, TranscribeRequest, TranscriptionResult, TranscriptionSegment, TranscriptionWord};
use aibridge_core::model::chat::{ChatCompletion, ChatCompletionChunk, ChatRequest};
use aibridge_core::model::common::{ModelInfo, ModelType, VoiceInfo, TaskStatus};
use aibridge_core::model::image::{ImageData, ImageRequest, ImageResult};
use aibridge_core::model::options::{EmbedRequest, EmbeddingItem, EmbeddingResult, EmbeddingVector};
use aibridge_core::model::video::{VideoRequest, VideoTask, VideoStatus};

// ──────────────────────────────────────────────────────────────────────────
// 错误映射
// ──────────────────────────────────────────────────────────────────────────

/// 将 `AibridgeError` 映射为 `napi::Error`
///
/// reason 编码为 `[code] message` 形式，便于 JS 侧解析 `.code` 属性。
/// status 统一用 `GenericFailure`（napi 的 Status 枚举无法承载业务 code）。
fn map_error(err: AibridgeError) -> Error {
    let code = err.code();
    let message = err.to_string();
    Error::new(Status::GenericFailure, format!("[{code}] {message}"))
}

// ──────────────────────────────────────────────────────────────────────────
// JS 友好的返回数据模型（#[napi(object)]）
// ──────────────────────────────────────────────────────────────────────────

/// 对话选项中的消息（JS 返回值形态）
#[napi(object)]
pub struct ChatChoiceJs {
    /// 选项索引
    pub index: u32,
    /// 生成的回复消息
    pub message: ChoiceMessageJs,
    /// 结束原因（stop / length / content_filter / tool_calls）
    pub finish_reason: Option<String>,
}

/// 完成结果中的消息
#[napi(object)]
pub struct ChoiceMessageJs {
    /// 角色（通常为 "assistant"）
    pub role: String,
    /// 消息内容
    pub content: Option<String>,
}

/// Token 使用统计
#[napi(object)]
pub struct ChatUsageJs {
    /// 提示词 token 数
    pub prompt_tokens: i64,
    /// 完成回复 token 数
    pub completion_tokens: i64,
    /// 总 token 数
    pub total_tokens: i64,
}

/// 对话完成结果
#[napi(object)]
pub struct ChatCompletionJs {
    /// 响应 ID
    pub id: String,
    /// 对象类型
    pub object: String,
    /// 创建时间戳（秒）
    pub created: i64,
    /// 使用的模型
    pub model: String,
    /// 回复选项列表
    pub choices: Vec<ChatChoiceJs>,
    /// Token 使用统计
    pub usage: Option<ChatUsageJs>,
    /// 服务层级
    pub service_tier: Option<String>,
    /// 系统指纹
    pub system_fingerprint: Option<String>,
}

/// 流式增量消息
#[napi(object)]
pub struct DeltaMessageJs {
    /// 角色（首个块通常为 "assistant"）
    pub role: Option<String>,
    /// 增量内容
    pub content: Option<String>,
}

/// 流式增量
#[napi(object)]
pub struct ChatCompletionDeltaJs {
    /// 增量索引
    pub index: u32,
    /// 增量消息内容
    pub delta: DeltaMessageJs,
    /// 结束原因
    pub finish_reason: Option<String>,
}

/// 流式对话块
#[napi(object)]
pub struct ChatCompletionChunkJs {
    /// 响应 ID
    pub id: String,
    /// 对象类型
    pub object: String,
    /// 创建时间戳（秒）
    pub created: i64,
    /// 使用的模型
    pub model: String,
    /// 增量选项列表
    pub choices: Vec<ChatCompletionDeltaJs>,
    /// Token 使用统计（仅末尾块可能出现）
    pub usage: Option<ChatUsageJs>,
}

/// 文字转语音结果
#[napi(object)]
pub struct SpeechResultJs {
    /// 音频二进制数据（JS Buffer）
    pub audio_data: Buffer,
    /// 音频 MIME 类型
    pub content_type: String,
    /// 音频格式（mp3/wav/opus 等）
    pub format: String,
    /// 估计音频时长（秒）
    pub duration: Option<f64>,
    /// 使用的模型 ID
    pub model: Option<String>,
    /// 音频 URL（部分 Provider 返回）
    pub audio_url: Option<String>,
}

// ──────────────────────────────────────────────────────────────────────────
// 图像生成 JS 数据模型
// ──────────────────────────────────────────────────────────────────────────

/// 图像数据（JS 返回值形态）
#[napi(object)]
pub struct ImageDataJs {
    /// 图像 URL
    pub url: Option<String>,
    /// Base64 编码的图像
    pub b64_json: Option<String>,
    /// 修改后的提示词（如模型优化过）
    pub revised_prompt: Option<String>,
}

/// 图像生成结果（JS 返回值形态）
#[napi(object)]
pub struct ImageResultJs {
    /// 响应 ID
    pub id: String,
    /// 对象类型
    pub object: String,
    /// 创建时间戳
    pub created: i64,
    /// 使用的模型
    pub model: String,
    /// 生成的图像列表
    pub data: Vec<ImageDataJs>,
}

// ──────────────────────────────────────────────────────────────────────────
// 视频生成 JS 数据模型
// ──────────────────────────────────────────────────────────────────────────

/// 视频任务信息（JS 返回值形态）
#[napi(object)]
pub struct VideoTaskJs {
    /// 任务 ID（用于轮询状态）
    pub task_id: String,
    /// 使用的模型
    pub model: String,
    /// 任务状态（pending / processing / success / failed）
    pub status: String,
    /// 创建时间戳
    pub created_at: i64,
}

/// 视频任务状态（JS 返回值形态）
#[napi(object)]
pub struct VideoStatusJs {
    /// 任务 ID
    pub task_id: String,
    /// 任务状态
    pub status: String,
    /// 视频 URL（成功时）
    pub video_url: Option<String>,
    /// 进度 0-100
    pub progress: Option<u32>,
    /// 错误信息（失败时）
    pub error: Option<String>,
    /// 创建时间戳
    pub created_at: Option<i64>,
    /// 更新时间戳
    pub updated_at: Option<i64>,
}

// ──────────────────────────────────────────────────────────────────────────
// 文本嵌入 JS 数据模型
// ──────────────────────────────────────────────────────────────────────────

/// 嵌入项（JS 返回值形态）
#[napi(object)]
pub struct EmbeddingItemJs {
    /// 对象类型（固定 "embedding"）
    pub object: String,
    /// 索引
    pub index: u32,
    /// 嵌入向量（浮点列表）
    pub embedding: Vec<f64>,
}

/// 嵌入使用统计
#[napi(object)]
pub struct EmbeddingUsageJs {
    /// 提示词 token 数
    pub prompt_tokens: i64,
    /// 总 token 数
    pub total_tokens: i64,
}

/// 嵌入结果（JS 返回值形态）
#[napi(object)]
pub struct EmbeddingResultJs {
    /// 对象类型（固定 "list"）
    pub object: String,
    /// 嵌入向量列表
    pub data: Vec<EmbeddingItemJs>,
    /// 使用的模型
    pub model: String,
    /// 使用统计
    pub usage: Option<EmbeddingUsageJs>,
}

// ──────────────────────────────────────────────────────────────────────────
// 语音转写 JS 数据模型
// ──────────────────────────────────────────────────────────────────────────

/// 转写分段信息（JS 返回值形态）
#[napi(object)]
pub struct TranscriptionSegmentJs {
    /// 分段 ID
    pub id: u32,
    /// 开始时间（秒）
    pub start: f64,
    /// 结束时间（秒）
    pub end: f64,
    /// 分段文本
    pub text: String,
    /// 分段置信度（0-1）
    pub confidence: Option<f64>,
    /// 说话人标识（说话人分离时使用）
    pub speaker: Option<String>,
}

/// 转写词级时间戳信息（JS 返回值形态）
#[napi(object)]
pub struct TranscriptionWordJs {
    /// 词文本
    pub word: String,
    /// 开始时间（秒）
    pub start: f64,
    /// 结束时间（秒）
    pub end: f64,
    /// 置信度（0-1）
    pub confidence: Option<f64>,
}

/// 转写结果（JS 返回值形态）
#[napi(object)]
pub struct TranscriptionResultJs {
    /// 完整转写文本
    pub text: String,
    /// 检测到的语言
    pub language: Option<String>,
    /// 音频时长（秒）
    pub duration: Option<f64>,
    /// 分段信息
    pub segments: Option<Vec<TranscriptionSegmentJs>>,
    /// 词级时间戳
    pub words: Option<Vec<TranscriptionWordJs>>,
    /// 任务类型（transcribe / translate）
    pub task: String,
    /// 使用统计
    pub usage: Option<serde_json::Value>,
    /// 使用的模型 ID
    pub model: Option<String>,
}

// ──────────────────────────────────────────────────────────────────────────
// 模型/音色列表 JS 数据模型
// ──────────────────────────────────────────────────────────────────────────

/// 模型信息（JS 返回值形态）
#[napi(object)]
pub struct ModelInfoJs {
    /// 模型标识符
    pub id: String,
    /// 模型显示名称
    pub name: String,
    /// 模型类型（chat / image / video / audio）
    pub model_type: String,
    /// 提供商名称
    pub provider: String,
    /// 支持的能力列表
    pub capabilities: Vec<String>,
    /// 最大 token 数（仅 chat 模型）
    pub max_tokens: Option<u32>,
    /// 是否支持流式输出
    pub supports_streaming: bool,
    /// 模型描述
    pub description: Option<String>,
    /// 模型创建时间戳
    pub created: Option<i64>,
}

/// 语音信息（JS 返回值形态）
#[napi(object)]
pub struct VoiceInfoJs {
    /// 音色短名
    pub short_name: Option<String>,
    /// 音色显示名
    pub name: Option<String>,
    /// 语言区域
    pub locale: Option<String>,
    /// 性别
    pub gender: Option<String>,
    /// 音色 ID
    pub voice_id: Option<String>,
}

// ──────────────────────────────────────────────────────────────────────────
// core → JS 数据模型转换
// ──────────────────────────────────────────────────────────────────────────

/// 将 core 的 `ChatCompletion` 转为 JS 友好结构
fn to_chat_completion_js(c: ChatCompletion) -> ChatCompletionJs {
    ChatCompletionJs {
        id: c.id,
        object: c.object,
        created: c.created as i64,
        model: c.model,
        choices: c
            .choices
            .into_iter()
            .map(|ch| ChatChoiceJs {
                index: ch.index,
                message: ChoiceMessageJs {
                    role: ch.message.role,
                    content: ch.message.content,
                },
                finish_reason: ch.finish_reason,
            })
            .collect(),
        usage: c.usage.map(|u| ChatUsageJs {
            prompt_tokens: u.prompt_tokens as i64,
            completion_tokens: u.completion_tokens as i64,
            total_tokens: u.total_tokens as i64,
        }),
        service_tier: c.service_tier,
        system_fingerprint: c.system_fingerprint,
    }
}

/// 将 core 的 `ChatCompletionChunk` 转为 JS 友好结构
fn to_chat_chunk_js(c: ChatCompletionChunk) -> ChatCompletionChunkJs {
    ChatCompletionChunkJs {
        id: c.id,
        object: c.object,
        created: c.created as i64,
        model: c.model,
        choices: c
            .choices
            .into_iter()
            .map(|d| ChatCompletionDeltaJs {
                index: d.index,
                delta: DeltaMessageJs {
                    role: d.delta.role,
                    content: d.delta.content,
                },
                finish_reason: d.finish_reason,
            })
            .collect(),
        usage: c.usage.map(|u| ChatUsageJs {
            prompt_tokens: u.prompt_tokens as i64,
            completion_tokens: u.completion_tokens as i64,
            total_tokens: u.total_tokens as i64,
        }),
    }
}

/// 将 core 的 `ImageData` 转为 JS 友好结构
fn to_image_data_js(i: ImageData) -> ImageDataJs {
    ImageDataJs {
        url: i.url,
        b64_json: i.b64_json,
        revised_prompt: i.revised_prompt,
    }
}

/// 将 core 的 `ImageResult` 转为 JS 友好结构
fn to_image_result_js(r: ImageResult) -> ImageResultJs {
    ImageResultJs {
        id: r.id,
        object: r.object,
        created: r.created as i64,
        model: r.model,
        data: r.data.into_iter().map(to_image_data_js).collect(),
    }
}

/// 将 core 的 `VideoTask` 转为 JS 友好结构
fn to_video_task_js(t: VideoTask) -> VideoTaskJs {
    VideoTaskJs {
        task_id: t.task_id,
        model: t.model,
        status: task_status_to_js(t.status),
        created_at: t.created_at as i64,
    }
}

/// 将 core 的 `VideoStatus` 转为 JS 友好结构
fn to_video_status_js(s: VideoStatus) -> VideoStatusJs {
    VideoStatusJs {
        task_id: s.task_id,
        status: task_status_to_js(s.status),
        video_url: s.video_url,
        progress: s.progress,
        error: s.error,
        created_at: s.created_at.map(|v| v as i64),
        updated_at: s.updated_at.map(|v| v as i64),
    }
}

/// 将 core 的 `EmbeddingItem` 转为 JS 友好结构
fn to_embedding_item_js(item: EmbeddingItem) -> EmbeddingItemJs {
    let embedding = match item.embedding {
        EmbeddingVector::Float(vec) => vec,
        EmbeddingVector::Base64(_) => Vec::new(), // base64 不直接暴露给 JS
    };
    EmbeddingItemJs {
        object: item.object,
        index: item.index,
        embedding,
    }
}

/// 将 core 的 `EmbeddingResult` 转为 JS 友好结构
fn to_embedding_result_js(r: EmbeddingResult) -> EmbeddingResultJs {
    EmbeddingResultJs {
        object: r.object,
        model: r.model,
        data: r.data.into_iter().map(to_embedding_item_js).collect(),
        usage: r.usage.map(|u| EmbeddingUsageJs {
            prompt_tokens: u.prompt_tokens as i64,
            total_tokens: u.total_tokens as i64,
        }),
    }
}

/// 将 core 的 `TranscriptionSegment` 转为 JS 友好结构
fn to_transcription_segment_js(s: TranscriptionSegment) -> TranscriptionSegmentJs {
    TranscriptionSegmentJs {
        id: s.id,
        start: s.start,
        end: s.end,
        text: s.text,
        confidence: s.confidence,
        speaker: s.speaker,
    }
}

/// 将 core 的 `TranscriptionWord` 转为 JS 友好结构
fn to_transcription_word_js(w: TranscriptionWord) -> TranscriptionWordJs {
    TranscriptionWordJs {
        word: w.word,
        start: w.start,
        end: w.end,
        confidence: w.confidence,
    }
}

/// 将 core 的 `TranscriptionResult` 转为 JS 友好结构
fn to_transcription_result_js(r: TranscriptionResult) -> TranscriptionResultJs {
    TranscriptionResultJs {
        text: r.text,
        language: r.language,
        duration: r.duration,
        segments: r.segments.map(|s| s.into_iter().map(to_transcription_segment_js).collect()),
        words: r.words.map(|w| w.into_iter().map(to_transcription_word_js).collect()),
        task: r.task,
        usage: r.usage,
        model: r.model,
    }
}

/// 将 `ModelType` 转为字符串
fn model_type_to_js(mt: ModelType) -> String {
    mt.as_str().to_owned()
}

/// 将 core 的 `ModelInfo` 转为 JS 友好结构
fn to_model_info_js(m: ModelInfo) -> ModelInfoJs {
    ModelInfoJs {
        id: m.id,
        name: m.name,
        model_type: model_type_to_js(m.model_type),
        provider: m.provider,
        capabilities: m.capabilities,
        max_tokens: m.max_tokens,
        supports_streaming: m.supports_streaming,
        description: m.description,
        created: m.created.map(|v| v as i64),
    }
}

/// 将 core 的 `VoiceInfo` 转为 JS 友好结构
fn to_voice_info_js(v: VoiceInfo) -> VoiceInfoJs {
    VoiceInfoJs {
        short_name: v.short_name,
        name: v.name,
        locale: v.locale,
        gender: v.gender,
        voice_id: v.voice_id,
    }
}

/// 将 core 的 `TaskStatus` 转为字符串
fn task_status_to_js(s: TaskStatus) -> String {
    match s {
        TaskStatus::Pending => "pending".to_owned(),
        TaskStatus::Processing => "processing".to_owned(),
        TaskStatus::Success => "success".to_owned(),
        TaskStatus::Failed => "failed".to_owned(),
    }
}

// ──────────────────────────────────────────────────────────────────────────
// 统一客户端（napi 类）
// ──────────────────────────────────────────────────────────────────────────

/// AIBridge 统一客户端
///
/// 直连 aibridge-core 的 `Client`。所有异步方法返回 JS `Promise`。
///
/// 示例（JS）：
/// ```js
/// const { Client } = require('aibridge');
/// const client = new Client('echo', {});
/// await client.start();
/// const resp = await client.chat({ model: 'echo-chat', messages: [{ role: 'user', content: 'hello' }] });
/// await client.close();
/// ```
#[napi]
pub struct Client {
    /// core 客户端，用 `Arc<Mutex>` 包裹以支持 `&self` async 方法
    /// （napi async fn 不允许 `&mut self`，需内部可变性）
    inner: Arc<Mutex<CoreClient>>,
}

#[napi]
impl Client {
    /// 创建客户端
    ///
    /// `provider` 为 Provider 类型（如 "echo"、"openai"、"agnes"）。
    /// `options` 为连接选项（api_key / base_url / timeout 等，可选字段）。
    #[napi(constructor)]
    pub fn new(provider: String, options: Option<Value>) -> Result<Self> {
        // 将 JS options 对象转为 ClientOptions（经 serde_json 中转）
        let opts: ClientOptions = match options {
            Some(v) => serde_json::from_value(v)
                .map_err(|e| Error::new(Status::InvalidArg, format!("options 解析失败: {e}")))?,
            None => ClientOptions::default(),
        };

        let core = CoreClient::new(&provider, opts).map_err(map_error)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(core)),
        })
    }

    /// 启动客户端（初始化适配器）
    #[napi]
    pub async fn start(&self) -> Result<()> {
        let mut guard = self.inner.lock().await;
        guard.start().await.map_err(map_error)
    }

    /// 关闭客户端（释放资源）
    #[napi]
    pub async fn close(&self) -> Result<()> {
        let mut guard = self.inner.lock().await;
        guard.close().await.map_err(map_error)
    }

    /// 文本对话
    ///
    /// `request` 形如 `{ model, messages: [{ role, content }], temperature?, max_tokens? }`。
    /// 透传不识别的字段到 core 的 `extra`（厂商特有参数）。
    #[napi]
    pub async fn chat(&self, request: Value) -> Result<ChatCompletionJs> {
        let req: ChatRequest = serde_json::from_value(request)
            .map_err(|e| Error::new(Status::InvalidArg, format!("request 解析失败: {e}")))?;
        let guard = self.inner.lock().await;
        let resp = guard.chat(req).await.map_err(map_error)?;
        Ok(to_chat_completion_js(resp))
    }

    /// 流式文本对话
    ///
    /// 返回 `ChatStreamIterator`，支持 JS `for await...of` 迭代 `ChatCompletionChunk`。
    #[napi]
    pub async fn chat_stream(&self, request: Value) -> Result<ChatStreamIterator> {
        let req: ChatRequest = serde_json::from_value(request)
            .map_err(|e| Error::new(Status::InvalidArg, format!("request 解析失败: {e}")))?;

        // 在 napi 全局 tokio runtime 上获取流（需要在持锁期间拿到 stream）
        let stream = {
            let guard = self.inner.lock().await;
            guard.chat_stream(req).await.map_err(map_error)?
        };

        // 用 mpsc channel 桥接 BoxStream：spawn 一个 task 消费 stream 并推送 chunk
        let (tx, rx) = mpsc::channel::<std::result::Result<ChatCompletionChunkJs, Error>>(16);
        // spawn 到 napi 全局 tokio runtime（fire-and-forget，task 自行消费完毕后 drop tx）
        spawn(async move {
            let mut stream = stream;
            while let Some(item) = stream.next().await {
                let chunk_result = item.map(to_chat_chunk_js).map_err(map_error);
                // 接收方关闭则结束
                if tx.send(chunk_result).await.is_err() {
                    break;
                }
            }
            // stream 结束（正常或错误已推送），drop tx 让 rx 收到 None 表示完成
            drop(tx);
        });

        Ok(ChatStreamIterator {
            rx: Arc::new(Mutex::new(rx)),
        })
    }

    /// 文字转语音
    ///
    /// `request` 形如 `{ model, input, voice, response_format?, speed? }`。
    /// `voice` 可为字符串（单个音色）或字符串数组（候选列表，用于自动降级）。
    /// 返回 `SpeechResultJs`，`audio_data` 为 JS `Buffer`。
    #[napi]
    pub async fn speech(&self, request: Value) -> Result<SpeechResultJs> {
        // 将 voice 字段归一化为 VoiceSpec 结构（core 期望 { voices: [...] }）
        let mut req_value = request;
        if let Some(obj) = req_value.as_object_mut() {
            if let Some(voice) = obj.remove("voice") {
                let voices = match voice {
                    // 字符串 → [voice]
                    Value::String(s) => vec![s],
                    // 字符串数组 → 原样
                    Value::Array(arr) => arr
                        .into_iter()
                        .filter_map(|v| v.as_str().map(str::to_owned))
                        .collect(),
                    // 已是 { voices: [...] } 对象 → 直接用
                    Value::Object(map) if map.contains_key("voices") => {
                        match map.get("voices").cloned() {
                            Some(Value::Array(arr)) => arr
                                .into_iter()
                                .filter_map(|v| v.as_str().map(str::to_owned))
                                .collect(),
                            _ => {
                                return Err(Error::new(
                                    Status::InvalidArg,
                                    "voice.voices 必须为字符串数组",
                                ))
                            }
                        }
                    }
                    _ => {
                        return Err(Error::new(
                            Status::InvalidArg,
                            "voice 必须为字符串、字符串数组或 { voices: [...] } 对象",
                        ))
                    }
                };
                obj.insert("voice".into(), serde_json::json!({ "voices": voices }));
            }
        }

        let req: SpeechRequest = serde_json::from_value(req_value)
            .map_err(|e| Error::new(Status::InvalidArg, format!("request 解析失败: {e}")))?;
        let guard = self.inner.lock().await;
        let resp = guard.speech(req).await.map_err(map_error)?;

        // 优先 audio_data，其次解码 audio_base64，最后空 Buffer
        let bytes = resp.get_audio_bytes().unwrap_or_default();
        Ok(SpeechResultJs {
            audio_data: Buffer::from(bytes),
            content_type: resp.content_type,
            format: resp.format,
            duration: resp.duration,
            model: resp.model,
            audio_url: resp.audio_url,
        })
    }

    /// 图像生成
    ///
    /// `request` 形如 `{ model, prompt, size?, n?, quality?, style? }`。
    /// 返回 `ImageResultJs`，包含生成的图像列表。
    #[napi]
    pub async fn image_generate(&self, request: Value) -> Result<ImageResultJs> {
        let req: ImageRequest = serde_json::from_value(request)
            .map_err(|e| Error::new(Status::InvalidArg, format!("request 解析失败: {e}")))?;
        let guard = self.inner.lock().await;
        let resp = guard.image_generate(req).await.map_err(map_error)?;
        Ok(to_image_result_js(resp))
    }

    /// 视频生成（创建异步任务）
    ///
    /// `request` 形如 `{ model, prompt, width?, height?, duration?, aspect_ratio?, mode? }`。
    /// 返回 `VideoTaskJs`，包含 `task_id` 用于后续轮询。
    #[napi]
    pub async fn video_create(&self, request: Value) -> Result<VideoTaskJs> {
        let req: VideoRequest = serde_json::from_value(request)
            .map_err(|e| Error::new(Status::InvalidArg, format!("request 解析失败: {e}")))?;
        let guard = self.inner.lock().await;
        let resp = guard.video_create(req).await.map_err(map_error)?;
        Ok(to_video_task_js(resp))
    }

    /// 视频生成状态轮询
    ///
    /// `task_id` 为 `video_create` 返回的任务 ID，`model` 为使用的模型。
    /// 返回 `VideoStatusJs`，包含进度、结果 URL 或错误信息。
    #[napi]
    pub async fn video_poll(&self, task_id: String, model: String) -> Result<VideoStatusJs> {
        let guard = self.inner.lock().await;
        let resp = guard.video_poll(&task_id, &model).await.map_err(map_error)?;
        Ok(to_video_status_js(resp))
    }

    /// 文本嵌入
    ///
    /// `request` 形如 `{ model, input: "text" | ["text1", "text2"] }`。
    /// 返回 `EmbeddingResultJs`，包含向量列表。
    #[napi]
    pub async fn embed(&self, request: Value) -> Result<EmbeddingResultJs> {
        let req: EmbedRequest = serde_json::from_value(request)
            .map_err(|e| Error::new(Status::InvalidArg, format!("request 解析失败: {e}")))?;
        let guard = self.inner.lock().await;
        let resp = guard.embed(req).await.map_err(map_error)?;
        Ok(to_embedding_result_js(resp))
    }

    /// 语音转文字（ASR 转写）
    ///
    /// `request` 形如 `{ model, file, language?, response_format? }`。
    /// `file` 为音频路径、URL、Base64 或二进制数据。
    /// 返回 `TranscriptionResultJs`，包含文本、分段和时间戳。
    #[napi]
    pub async fn transcribe(&self, request: Value) -> Result<TranscriptionResultJs> {
        let req: TranscribeRequest = serde_json::from_value(request)
            .map_err(|e| Error::new(Status::InvalidArg, format!("request 解析失败: {e}")))?;
        let guard = self.inner.lock().await;
        let resp = guard.transcribe(req).await.map_err(map_error)?;
        Ok(to_transcription_result_js(resp))
    }

    /// 语音翻译（将非英文音频翻译为英文）
    ///
    /// 参数与 `transcribe` 相同，内部设置 `translate=true`。
    /// 返回 `TranscriptionResultJs`，文本为英文。
    #[napi]
    pub async fn translate(&self, request: Value) -> Result<TranscriptionResultJs> {
        let mut req: TranscribeRequest = serde_json::from_value(request)
            .map_err(|e| Error::new(Status::InvalidArg, format!("request 解析失败: {e}")))?;
        req.translate = true;
        let guard = self.inner.lock().await;
        let resp = guard.translate(req).await.map_err(map_error)?;
        Ok(to_transcription_result_js(resp))
    }

    /// 列出可用模型
    ///
    /// `filter` 可选，为 "chat" / "image" / "video" / "audio" 之一，
    /// 用于按模型类型过滤；省略则返回全部。
    /// 返回 `ModelInfo` 列表。
    #[napi]
    pub async fn list_models(&self, filter: Option<String>) -> Result<Vec<ModelInfoJs>> {
        let guard = self.inner.lock().await;
        let model_type = filter.map(|s| ModelType::from(s.as_str()));
        let resp = guard.list_models(model_type).await.map_err(map_error)?;
        Ok(resp.into_iter().map(to_model_info_js).collect())
    }

    /// 列出可用音色
    ///
    /// `language` 可选，为语言区域代码（如 "zh-CN"），用于过滤音色。
    /// 返回 `VoiceInfo` 列表。
    #[napi]
    pub async fn list_voices(&self, language: Option<String>) -> Result<Vec<VoiceInfoJs>> {
        let guard = self.inner.lock().await;
        let lang = language.as_ref().map(|s| s.as_str());
        let resp = guard.list_voices(lang).await.map_err(map_error)?;
        Ok(resp.into_iter().map(to_voice_info_js).collect())
    }

    /// 推荐可用音色
    ///
    /// `language` 为语言区域代码（如 "zh-CN"），`gender` 为 "Male" / "Female"，
    /// `limit` 为推荐数量上限。
    /// 返回 `VoiceInfo` 列表。
    #[napi]
    pub async fn recommend_voices(
        &self,
        language: Option<String>,
        gender: Option<String>,
        limit: u32,
    ) -> Result<Vec<VoiceInfoJs>> {
        let guard = self.inner.lock().await;
        let lang = language.as_ref().map(|s| s.as_str());
        let gen = gender.as_ref().map(|s| s.as_str());
        let resp = guard.recommend_voices(lang, gen, limit as usize).await.map_err(map_error)?;
        Ok(resp.into_iter().map(to_voice_info_js).collect())
    }
}

// ──────────────────────────────────────────────────────────────────────────
// 流式迭代器（napi 类）
// ──────────────────────────────────────────────────────────────────────────

/// 流式对话迭代器
///
/// 由 `Client.chatStream()` 返回。通过 `[Symbol.asyncIterator]`（index.js 中安装）
/// 支持 JS `for await...of` 语法迭代 `ChatCompletionChunk`。
///
/// 也可直接调用 `.next()` 方法手动迭代（返回 `null` 表示结束）。
#[napi]
pub struct ChatStreamIterator {
    /// chunk 接收端，用 `Arc<Mutex>` 包裹以支持 `&self` async 方法
    rx: Arc<Mutex<mpsc::Receiver<std::result::Result<ChatCompletionChunkJs, Error>>>>,
}

#[napi]
impl ChatStreamIterator {
    /// 拉取下一个 chunk
    ///
    /// 返回 `ChatCompletionChunk`，流结束时返回 `null`。
    /// 流内部错误以 `napi::Error`（reject）形式抛出。
    #[napi]
    pub async fn next(&self) -> Result<Option<ChatCompletionChunkJs>> {
        let mut guard = self.rx.lock().await;
        match guard.recv().await {
            // 收到正常 chunk
            Some(Ok(chunk)) => Ok(Some(chunk)),
            // 接收端收到错误 chunk
            Some(Err(e)) => Err(e),
            // channel 关闭，流结束
            None => Ok(None),
        }
    }
}
