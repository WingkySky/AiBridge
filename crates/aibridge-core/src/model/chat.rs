//! 文本对话数据模型
//!
//! 定义文本对话相关的 serde struct：请求、消息、完成结果、流式块。
//! 对应 Python v1 (agn-sdk) 的 `agn/models/chat.py`。
//!
//! 设计要点（与设计文档第 6 节一致）：
//! - `ChatMessage` 为 tagged enum（`role` 作为 tag），支持 system/user/assistant/tool
//! - `UserContent` 支持 String 或多模态 Vec<ContentPart>
//! - `ChatRequest` 用 Builder 模式，替代 Python 的 `**kwargs` + `ChatOptions`

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::model::options::{ReasoningEffort, ResponseFormat, StopSeq, ToolChoice, ToolDefinition};

/// 对话请求
///
/// 对应设计文档第 6 节 `ChatRequest`。
/// 用 `ChatRequest::builder(model, messages)` 链式构造。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// 模型名称
    pub model: String,
    /// 消息列表
    pub messages: Vec<ChatMessage>,
    /// 温度系数（0-2，越高越随机）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// 核采样（0-1）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-K 采样
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// 最大生成 token 数
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// 停止词
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSeq>,
    /// 生成回复数量
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// 存在惩罚（-2 到 2）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// 频率惩罚（-2 到 2）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// 随机种子
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// 可用工具列表
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    /// 工具选择策略
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// 是否允许并行工具调用
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// 推理努力程度
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// 响应格式
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// 用户标识（用于风控/限流）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// 是否流式输出
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// 厂商特有参数透传（不会被映射，直接加入请求体）
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

impl ChatRequest {
    /// 创建 Builder
    ///
    /// # 示例
    /// ```ignore
    /// let req = ChatRequest::builder("gpt-4o", vec![
    ///     ChatMessage::user("Hello!"),
    /// ]).temperature(0.7).max_tokens(1000).build();
    /// ```
    pub fn builder(model: impl Into<String>, messages: Vec<ChatMessage>) -> ChatRequestBuilder {
        ChatRequestBuilder {
            inner: ChatRequest {
                model: model.into(),
                messages,
                temperature: None,
                top_p: None,
                top_k: None,
                max_tokens: None,
                stop: None,
                n: None,
                presence_penalty: None,
                frequency_penalty: None,
                seed: None,
                tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                reasoning_effort: None,
                response_format: None,
                user: None,
                stream: false,
                extra: HashMap::new(),
            },
        }
    }
}

/// `ChatRequest` 的 Builder
#[derive(Debug, Clone)]
pub struct ChatRequestBuilder {
    inner: ChatRequest,
}

impl ChatRequestBuilder {
    pub fn temperature(mut self, t: f64) -> Self {
        self.inner.temperature = Some(t);
        self
    }
    pub fn top_p(mut self, t: f64) -> Self {
        self.inner.top_p = Some(t);
        self
    }
    pub fn top_k(mut self, t: u32) -> Self {
        self.inner.top_k = Some(t);
        self
    }
    pub fn max_tokens(mut self, t: u32) -> Self {
        self.inner.max_tokens = Some(t);
        self
    }
    pub fn stop(mut self, s: StopSeq) -> Self {
        self.inner.stop = Some(s);
        self
    }
    pub fn n(mut self, n: u32) -> Self {
        self.inner.n = Some(n);
        self
    }
    pub fn presence_penalty(mut self, p: f64) -> Self {
        self.inner.presence_penalty = Some(p);
        self
    }
    pub fn frequency_penalty(mut self, p: f64) -> Self {
        self.inner.frequency_penalty = Some(p);
        self
    }
    pub fn seed(mut self, s: u64) -> Self {
        self.inner.seed = Some(s);
        self
    }
    pub fn tools(mut self, t: Vec<ToolDefinition>) -> Self {
        self.inner.tools = Some(t);
        self
    }
    pub fn tool_choice(mut self, c: ToolChoice) -> Self {
        self.inner.tool_choice = Some(c);
        self
    }
    pub fn parallel_tool_calls(mut self, p: bool) -> Self {
        self.inner.parallel_tool_calls = Some(p);
        self
    }
    pub fn reasoning_effort(mut self, e: ReasoningEffort) -> Self {
        self.inner.reasoning_effort = Some(e);
        self
    }
    pub fn response_format(mut self, f: ResponseFormat) -> Self {
        self.inner.response_format = Some(f);
        self
    }
    pub fn user(mut self, u: impl Into<String>) -> Self {
        self.inner.user = Some(u.into());
        self
    }
    pub fn stream(mut self, s: bool) -> Self {
        self.inner.stream = s;
        self
    }
    pub fn extra(mut self, k: impl Into<String>, v: impl Into<serde_json::Value>) -> Self {
        self.inner.extra.insert(k.into(), v.into());
        self
    }
    pub fn build(self) -> ChatRequest {
        self.inner
    }
}

/// 对话消息（tagged enum，`role` 作为 tag）
///
/// 对应设计文档第 6 节 `ChatMessage`。
/// 对应 Python v1 `ChatMessage`。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    /// 系统消息
    System {
        /// 消息内容
        content: String,
        /// 发送者名称（可选）
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// 用户消息
    User {
        /// 消息内容（字符串或多模态）
        content: UserContent,
        /// 发送者名称（可选）
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// 助手消息
    Assistant {
        /// 消息内容
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        /// 工具调用列表
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<crate::model::options::ToolCall>>,
    },
    /// 工具结果消息
    Tool {
        /// 工具调用 ID
        tool_call_id: String,
        /// 工具返回内容
        content: String,
    },
}

impl ChatMessage {
    /// 创建系统消息
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
            name: None,
        }
    }

    /// 创建用户消息（纯文本）
    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: UserContent::Text(content.into()),
            name: None,
        }
    }

    /// 创建用户消息（多模态）
    pub fn user_multimodal(parts: Vec<ContentPart>) -> Self {
        Self::User {
            content: UserContent::Parts(parts),
            name: None,
        }
    }

    /// 创建助手消息
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: Some(content.into()),
            tool_calls: None,
        }
    }

    /// 创建工具结果消息
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }
}

/// 用户消息内容（字符串或多模态部件列表）
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    /// 纯文本
    Text(String),
    /// 多模态部件列表
    Parts(Vec<ContentPart>),
}

/// 多模态内容部件
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// 文本部件
    Text {
        /// 文本内容
        text: String,
    },
    /// 图片 URL 部件
    ImageUrl {
        /// 图片 URL 信息
        image_url: ImageUrl,
    },
}

/// 图片 URL 信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// 图片 URL 或 data URI
    pub url: String,
    /// 图片细节级别（"low" / "high" / "auto"）
    #[serde(default = "default_detail", skip_serializing_if = "is_default_detail")]
    pub detail: String,
}

impl ImageUrl {
    /// 创建图片 URL
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            detail: "auto".into(),
        }
    }

    /// 指定细节级别
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }
}

fn default_detail() -> String {
    "auto".into()
}

fn is_default_detail(d: &str) -> bool {
    d == "auto"
}

/// 对话完成结果
///
/// 对应 Python v1 `ChatCompletion`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletion {
    /// 响应 ID
    pub id: String,
    /// 对象类型
    #[serde(default = "default_object_completion")]
    pub object: String,
    /// 创建时间戳
    pub created: u64,
    /// 使用的模型
    pub model: String,
    /// 回复选项列表
    pub choices: Vec<ChatChoice>,
    /// Token 使用统计
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatUsage>,
    /// 服务层级
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// 系统指纹
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

fn default_object_completion() -> String {
    "chat.completion".into()
}

/// 对话选项
///
/// 对应 Python v1 `ChatChoice`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// 选项索引
    pub index: u32,
    /// 生成的回复消息
    pub message: ChoiceMessage,
    /// 结束原因（stop / length / content_filter / tool_calls）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// 完成结果中的消息（比 ChatMessage 更宽松，便于解析 provider 响应）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceMessage {
    /// 角色（通常为 "assistant"）
    pub role: String,
    /// 消息内容
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// 工具调用列表
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::model::options::ToolCall>>,
}

/// Token 使用统计
///
/// 对应 Python v1 `ChatUsage`。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatUsage {
    /// 提示词 token 数
    pub prompt_tokens: u64,
    /// 完成回复 token 数
    pub completion_tokens: u64,
    /// 总 token 数
    pub total_tokens: u64,
}

/// 流式对话块
///
/// 对应 Python v1 `ChatCompletionChunk`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// 响应 ID
    pub id: String,
    /// 对象类型
    #[serde(default = "default_object_chunk")]
    pub object: String,
    /// 创建时间戳
    pub created: u64,
    /// 使用的模型
    pub model: String,
    /// 增量选项列表
    pub choices: Vec<ChatCompletionDelta>,
    /// Token 使用统计（仅 stream_options.include_usage=true 时在末尾块出现）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatUsage>,
}

fn default_object_chunk() -> String {
    "chat.completion.chunk".into()
}

/// 流式增量
///
/// 对应 Python v1 `ChatCompletionDelta`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionDelta {
    /// 增量索引
    pub index: u32,
    /// 增量消息内容
    pub delta: DeltaMessage,
    /// 结束原因
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// 流式增量消息
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeltaMessage {
    /// 角色（首个块通常为 "assistant"）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// 增量内容
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// 工具调用增量
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::model::options::ToolCall>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::options::{FunctionDefinition, ToolCallFunction};

    #[test]
    fn chat_message_system_serde() {
        let msg = ChatMessage::system("You are helpful.");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"system\""));
        assert!(json.contains("\"content\":\"You are helpful.\""));
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        match back {
            ChatMessage::System { content, .. } => assert_eq!(content, "You are helpful."),
            _ => panic!("应为 System"),
        }
    }

    #[test]
    fn chat_message_user_text_serde() {
        let msg = ChatMessage::user("Hi");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        // UserContent::Text 序列化为字符串
        assert!(json.contains("\"content\":\"Hi\""));
    }

    #[test]
    fn chat_message_user_multimodal_serde() {
        let msg = ChatMessage::user_multimodal(vec![
            ContentPart::Text {
                text: "What's this?".into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl::new("https://example.com/img.png"),
            },
        ]);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"type\":\"image_url\""));
        assert!(json.contains("https://example.com/img.png"));
    }

    #[test]
    fn chat_message_assistant_with_tool_calls() {
        let msg = ChatMessage::Assistant {
            content: None,
            tool_calls: Some(vec![crate::model::options::ToolCall {
                id: "call_1".into(),
                tool_type: "function".into(),
                function: ToolCallFunction {
                    name: "get_weather".into(),
                    arguments: "{}".into(),
                },
            }]),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"assistant\""));
        assert!(json.contains("\"tool_calls\""));
    }

    #[test]
    fn chat_message_tool_serde() {
        let msg = ChatMessage::tool("call_1", "sunny");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"tool\""));
        assert!(json.contains("\"tool_call_id\":\"call_1\""));
        assert!(json.contains("\"content\":\"sunny\""));
    }

    #[test]
    fn chat_request_builder() {
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")])
            .temperature(0.7)
            .max_tokens(1000)
            .top_p(0.9)
            .stream(true)
            .extra("custom", "value")
            .build();
        assert_eq!(req.model, "gpt-4o");
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(1000));
        assert!(req.stream);
        assert_eq!(
            req.extra.get("custom").and_then(|v| v.as_str()),
            Some("value")
        );
    }

    #[test]
    fn chat_request_skip_none_fields() {
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("temperature"));
        assert!(!json.contains("max_tokens"));
        assert!(!json.contains("stream")); // false 且 skip_serializing_if
        assert!(!json.contains("extra")); // 空且 skip_serializing_if
    }

    #[test]
    fn chat_request_with_tools() {
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("weather?")])
            .tools(vec![ToolDefinition::function(FunctionDefinition {
                name: "get_weather".into(),
                description: None,
                parameters: None,
            })])
            .tool_choice(ToolChoice::Auto)
            .build();
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tools\""));
        assert!(json.contains("\"tool_choice\":\"auto\""));
    }

    #[test]
    fn chat_completion_deserialize_openai_format() {
        let json = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7
            }
        });
        let comp: ChatCompletion = serde_json::from_value(json).unwrap();
        assert_eq!(comp.id, "chatcmpl-1");
        assert_eq!(comp.model, "gpt-4o");
        assert_eq!(comp.choices.len(), 1);
        assert_eq!(comp.choices[0].message.content.as_deref(), Some("Hello!"));
        assert_eq!(comp.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(comp.usage.as_ref().unwrap().total_tokens, 7);
    }

    #[test]
    fn chat_completion_chunk_deserialize() {
        let json = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": null
            }]
        });
        let chunk: ChatCompletionChunk = serde_json::from_value(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn image_url_default_detail_auto() {
        let iu = ImageUrl::new("https://example.com/x.png");
        assert_eq!(iu.detail, "auto");
        let json = serde_json::to_string(&iu).unwrap();
        // detail == "auto" 时被 skip
        assert!(!json.contains("detail"));
    }

    #[test]
    fn image_url_custom_detail_kept() {
        let iu = ImageUrl::new("https://example.com/x.png").with_detail("high");
        let json = serde_json::to_string(&iu).unwrap();
        assert!(json.contains("\"detail\":\"high\""));
    }

    #[test]
    fn user_content_text_roundtrip() {
        let c = UserContent::Text("hello".into());
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, "\"hello\"");
        let back: UserContent = serde_json::from_str(&json).unwrap();
        match back {
            UserContent::Text(s) => assert_eq!(s, "hello"),
            _ => panic!("应为 Text"),
        }
    }

    #[test]
    fn delta_message_default_empty() {
        let d = DeltaMessage::default();
        assert!(d.role.is_none());
        assert!(d.content.is_none());
    }
}
