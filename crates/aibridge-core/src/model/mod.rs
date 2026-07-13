//! 数据模型层
//!
//! 定义所有 AI 能力的 serde struct，替代 Python v1 的 Pydantic 模型。
//! 对应 Python v1 (agn-sdk) 的 `agn/models/` 目录。
//!
//! 子模块：
//! - [`common`]：通用类型（ModelType / ModelInfo / VoiceInfo / TaskStatus 等）
//! - [`options`]：工具定义、嵌入类型、参数映射等公共类型
//! - [`chat`]：文本对话（ChatRequest / ChatMessage / ChatCompletion / ChatCompletionChunk）
//! - [`image`]：图像生成（ImageRequest / ImageResult / FileInput）
//! - [`video`]：视频生成（VideoRequest / VideoTask / VideoStatus）
//! - [`audio`]：语音（TranscribeRequest / TranscriptionResult / SpeechRequest / SpeechResult）

pub mod audio;
pub mod chat;
pub mod common;
pub mod image;
pub mod options;
pub mod video;

// 重新导出常用类型，方便上层使用
pub use audio::{
    SpeechRequest, SpeechRequestBuilder, SpeechResult, TranscribeRequest, TranscribeRequestBuilder,
    TranscriptionResult, TranscriptionSegment, TranscriptionWord, VoiceSpec,
};
pub use chat::{
    ChatChoice, ChatCompletion, ChatCompletionChunk, ChatCompletionDelta, ChatMessage, ChatRequest,
    ChatRequestBuilder, ChatUsage, ChoiceMessage, ContentPart, DeltaMessage, ImageUrl, UserContent,
};
pub use common::{
    infer_model_type, ModelInfo, ModelType, ProviderInfo, TaskStatus, VideoMode, VoiceInfo,
    VoiceInfoBuilder,
};
pub use image::{FileInput, ImageData, ImageRequest, ImageRequestBuilder, ImageResult};
pub use options::{
    EmbedInput, EmbedRequest, EmbeddingItem, EmbeddingResult, EmbeddingUsage, EmbeddingVector,
    FunctionDefinition, ParameterMapping, ReasoningEffort, ResponseFormat, StopSeq, ToolCall,
    ToolCallFunction, ToolChoice, ToolDefinition,
};
pub use video::{VideoRequest, VideoRequestBuilder, VideoStatus, VideoTask};
