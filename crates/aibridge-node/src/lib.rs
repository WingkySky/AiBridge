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
use aibridge_core::model::audio::SpeechRequest;
use aibridge_core::model::chat::{ChatCompletion, ChatCompletionChunk, ChatRequest};

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
    Error::new(
        Status::GenericFailure,
        format!("[{code}] {message}"),
    )
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
                obj.insert(
                    "voice".into(),
                    serde_json::json!({ "voices": voices }),
                );
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
