//! Anthropic Claude 适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/anthropic.py`。
//!
//! Anthropic Claude 采用独立协议（非 OpenAI 兼容），本模块独立实现 chat / chat_stream /
//! list_models，不复用 `OpenAiCompatAdapter` 的请求体构造，但组合它以复用 HttpClient、
//! 能力集合与错误映射（与 Cohere 适配器的组合模式一致）。
//!
//! ## 协议要点
//!
//! - Base URL: `https://api.anthropic.com/v1`
//! - Chat: `POST /messages`（流式同端点，body 中 `stream: true`）
//! - 认证: `x-api-key: {key}` header（非 Bearer）
//! - 版本: `anthropic-version: 2023-06-01` header（固定值）
//! - 请求体: `{model, messages, system, max_tokens, ...}`
//!   - `system` 为顶层独立字段（从统一消息的 system 消息提取，不在 messages 中）
//!   - `max_tokens` 必填（默认 1024）
//!   - `stop` → `stop_sequences`（数组）
//!   - `tools` → Anthropic 原生格式（name/description/input_schema）
//!   - `tool_choice` → `{"type":"none"|"auto"|"any"}`；`parallel_tool_calls=false`
//!     → `disable_parallel_tool_use: true`（语义取反，仅 auto/any 支持）
//!   - `reasoning_effort` → `thinking`（对齐 Python ANTHROPIC_MAPPING 的 value_map）
//! - 工具调用消息: assistant 的 tool_calls → tool_use 内容块（arguments JSON 字符串
//!   parse 为 input 对象）；tool 消息 → user 消息包 tool_result 内容块（连续多条合并）
//! - 响应: `content[].text` 拼接为回复文本；`content[]` 中 tool_use 块解析为统一
//!   ToolCall；`stop_reason` 映射为 finish_reason（tool_use → tool_calls）；
//!   `usage.input_tokens` / `usage.output_tokens` 映射为 prompt / completion tokens
//! - 流式: SSE，每行 `data: <json>`，JSON 含 `type` 字段：
//!   - `content_block_delta`（delta.type==text_delta）→ 增量文本
//!   - `message_delta`（delta.stop_reason）→ 结束原因
//!   - `message_stop` → 流结束
//!   - 注意：流式 tool_use 相关块（content_block_start 的 tool_use、input_json_delta）
//!     暂不产出增量（Python v1 流式未支持工具调用，对齐其行为）；工具调用请使用
//!     非流式 chat。流式结束时的 stop_reason=tool_use 仍会映射为 finish_reason=tool_calls
//! - Models: `GET /models`，响应 `{"data":[{"id":...,"display_name":...}]}`
//!
//! 官方文档: <https://docs.anthropic.com/en/api/messages>

use std::collections::HashMap;

use async_trait::async_trait;
use futures::stream::{StreamExt, TryStreamExt};
use serde_json::{json, Value};

use crate::adapter::{Adapter, Capabilities, CapabilitySet, ChatStream};
use crate::adapters::openai_compat::OpenAiCompatAdapter;
use crate::config::ProviderConfig;
use crate::error::{AibridgeError, Result};
use crate::model::chat::{
    ChatChoice, ChatCompletion, ChatCompletionChunk, ChatCompletionDelta, ChatMessage, ChatRequest,
    ChoiceMessage, ContentPart, DeltaMessage, UserContent,
};
use crate::model::common::{infer_model_type, ModelInfo, ModelType};
use crate::model::options::{
    ParameterMapping, ReasoningEffort, StopSeq, ToolCall, ToolCallFunction, ToolChoice,
    ToolDefinition,
};
use crate::util;

// ==================== 默认配置 ====================

/// Anthropic 默认 Base URL
///
/// 对应 Python v1 `DEFAULT_BASE_URL = "https://api.anthropic.com"`，
/// Rust 直接采用含 `/v1` 前缀的值（Python 在 httpx base_url 上拼接 `/v1/messages`，
/// 此处 base_url 已含 `/v1`，chat 端点为 `messages`，行为等价）。
pub const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";

/// Anthropic API 版本（固定值，对应 Python v1 `DEFAULT_API_VERSION`）
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";

/// 默认 max_tokens（Anthropic 必填字段，未指定时兜底，对齐 Python v1 默认 1024）
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// Anthropic 参数映射
///
/// 对应 Python v1 `ANTHROPIC_MAPPING`。
/// `reasoning_effort → thinking` 的值映射在 `build_chat_body` 中特殊处理
/// （参照 DeepSeek 自动注入 thinking 的模式；value_map 语义见 `ParameterMapping`）。
///
/// rename_map：
/// - `stop` → `stop_sequences`（其余参数 max_tokens / top_p / top_k / temperature 原名透传）
pub fn anthropic_mapping() -> ParameterMapping {
    let mut rename: HashMap<String, Option<String>> = HashMap::new();
    rename.insert("stop".into(), Some("stop_sequences".into()));
    ParameterMapping {
        rename_map: rename,
        ..Default::default()
    }
}

/// Anthropic 支持的能力集合
///
/// 对齐 Python v1 `supported_capabilities`（CHAT / CHAT_STREAM / VISION / TOOL_CALL）。
/// ToolCall：tools/tool_choice 请求体构造、tool_use/tool_result 消息转换与
/// 响应 tool_use 块解析均已实现（流式 tool_use 增量未支持，对齐 Python v1，
/// 详见模块文档）。
fn anthropic_capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::new();
    caps.insert(Capabilities::Chat);
    caps.insert(Capabilities::ChatStream);
    caps.insert(Capabilities::Vision);
    caps.insert(Capabilities::ToolCall);
    caps
}

// ==================== 适配器结构 ====================

/// Anthropic Claude 适配器
///
/// 组合 `OpenAiCompatAdapter` 仅复用其 HttpClient、能力集合与错误映射，
/// chat / chat_stream / list_models 独立实现（Anthropic 特有 `/messages` 端点 +
/// `x-api-key` 认证 + `system` 顶层字段 + `content[].text` 响应 + SSE 事件）。
///
/// - Base URL: `https://api.anthropic.com/v1`
/// - Chat: `POST /messages`
/// - Models: `GET /models`
/// - 认证: `x-api-key` header + `anthropic-version` header
/// - 文档: <https://docs.anthropic.com/en/api/messages>
pub struct AnthropicAdapter {
    /// OpenAI 兼容地基（仅复用 HttpClient + 错误映射 + 能力集合，不委托 chat 等方法）
    compat: OpenAiCompatAdapter,
}

impl AnthropicAdapter {
    /// 创建 Anthropic 适配器
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let compat = OpenAiCompatAdapter::new(
            config,
            Self::PROVIDER_TYPE,
            Self::PROVIDER_NAME,
            DEFAULT_ANTHROPIC_BASE_URL,
            anthropic_capabilities(),
        )?;
        Ok(Self { compat })
    }

    /// 用显式 compat 构造（测试用，可注入 mockito 后端）
    #[cfg(test)]
    pub fn with_compat(compat: OpenAiCompatAdapter) -> Self {
        Self { compat }
    }

    /// Provider 类型标识
    const PROVIDER_TYPE: &'static str = "anthropic";

    /// Provider 显示名称
    const PROVIDER_NAME: &'static str = "Anthropic Claude";

    /// API key（从兼容地基配置提取）
    fn api_key(&self) -> Option<&str> {
        self.compat.api_key()
    }

    /// base_url
    fn base_url(&self) -> &str {
        self.compat.base_url()
    }

    /// 拼接完整 URL（base_url + 相对路径）
    fn url(&self, path: &str) -> String {
        let base = self.base_url().trim_end_matches('/');
        let path = path.trim_start_matches('/');
        format!("{base}/{path}")
    }

    /// 校验能力是否被支持
    fn ensure_capability(&self, cap: Capabilities) -> Result<()> {
        if self.compat.capabilities_set().contains(&cap) {
            Ok(())
        } else {
            Err(AibridgeError::UnsupportedCapability {
                capability: format!("{} (provider: {})", cap.as_str(), Self::PROVIDER_TYPE),
            })
        }
    }

    /// 发送带认证的 POST JSON 请求，并用错误映射处理响应
    ///
    /// Anthropic 用 `x-api-key` + `anthropic-version` header 认证（非 Bearer）。
    /// 错误映射复用 `OpenAiCompatAdapter::map_api_error`：其 `parse_error_message`
    /// 会提取 Anthropic 错误体 `{"type":"error","error":{"message":"..."}}` 中的
    /// `error.message`，分类规则（401→Auth、429→RateLimit、404→ModelNotFound、
    /// 400→Validation、5xx→Api）与任务规格一致。
    async fn post_authed_json(&self, path: &str, body: &Value) -> Result<Value> {
        let url = self.url(path);
        let resp = self.send_post(&url, body).await.map_err(|e| {
            if e.is_timeout() {
                AibridgeError::Timeout
            } else {
                AibridgeError::Network(e)
            }
        })?;
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(Self::map_api_error(status_code, &body_text));
        }
        resp.json::<Value>().await.map_err(AibridgeError::from)
    }

    /// 构造并发送带 Anthropic 认证 header 的 POST 请求
    async fn send_post(
        &self,
        url: &str,
        body: &Value,
    ) -> std::result::Result<reqwest::Response, reqwest::Error> {
        self.compat
            .http_inner()
            .post(url)
            .header("x-api-key", self.api_key().unwrap_or(""))
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
    }

    /// 发送带认证的 GET 请求，并用错误映射处理响应
    async fn get_authed_json(&self, path: &str) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .compat
            .http_inner()
            .get(&url)
            .header("x-api-key", self.api_key().unwrap_or(""))
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    AibridgeError::Timeout
                } else {
                    AibridgeError::Network(e)
                }
            })?;
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(Self::map_api_error(status_code, &body_text));
        }
        resp.json::<Value>().await.map_err(AibridgeError::from)
    }

    /// 将 API 错误响应映射为 AibridgeError
    ///
    /// 复用 `OpenAiCompatAdapter::map_api_error`：其错误消息提取兼容 Anthropic 的
    /// `error.message` 结构，状态码分类符合任务规格。
    fn map_api_error(status: u16, body: &str) -> AibridgeError {
        OpenAiCompatAdapter::map_api_error(status, body)
    }

    // ==================== 消息转换 ====================

    /// 转换统一消息为 Anthropic 格式，并提取 system prompt
    ///
    /// 对齐 Python v1 `AnthropicAdapter._convert_messages`：
    /// - system 消息 → 提取为顶层 `system` 字段（多条用 `\n` 拼接，Python 取最后一条；
    ///   Rust 改为拼接以保留全部 system 上下文）
    /// - user 消息 → `{"role":"user","content":...}`（纯文本用字符串，多模态用 content blocks）
    /// - assistant 消息 → `{"role":"assistant","content":...}`；带 tool_calls 时 content 为
    ///   blocks 数组（文本块在前，每个 ToolCall 转一个 tool_use 块）
    /// - tool 消息 → `{"role":"user","content":[{"type":"tool_result",...}]}`（Anthropic 原生格式）；
    ///   连续多条 tool 消息合并进同一条 user 消息（Anthropic 要求角色交替，且同一轮
    ///   tool_use 的所有 tool_result 须放在紧随其后的同一条 user 消息中）
    ///
    /// 多模态图片：
    /// - `data:` URI → `{"type":"image","source":{"type":"base64","media_type":...,"data":...}}`
    /// - `http(s):` URL → `{"type":"image","source":{"type":"url","url":...}}`（Anthropic 新版支持）
    fn convert_messages(messages: &[ChatMessage]) -> (Vec<Value>, Option<String>) {
        let mut converted: Vec<Value> = Vec::new();
        let mut system_parts: Vec<String> = Vec::new();

        for msg in messages {
            match msg {
                ChatMessage::System { content, .. } => {
                    system_parts.push(content.clone());
                }
                ChatMessage::User { content, .. } => {
                    let c = match content {
                        UserContent::Text(s) => Value::String(s.clone()),
                        UserContent::Parts(parts) => {
                            let blocks: Vec<Value> =
                                parts.iter().filter_map(convert_content_part).collect();
                            Value::Array(blocks)
                        }
                    };
                    converted.push(json!({"role": "user", "content": c}));
                }
                ChatMessage::Assistant {
                    content,
                    tool_calls,
                } => {
                    converted.push(convert_assistant_message(content, tool_calls));
                }
                ChatMessage::Tool {
                    tool_call_id,
                    content,
                } => {
                    // Anthropic 工具结果用 user 角色 + tool_result content block
                    let block = json!({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content
                    });
                    // 连续 tool 消息合并进上一条 tool_result user 消息
                    match converted.last_mut() {
                        Some(last) if is_tool_result_message(last) => {
                            if let Some(arr) = last["content"].as_array_mut() {
                                arr.push(block);
                            }
                        }
                        _ => converted.push(json!({"role": "user", "content": [block]})),
                    }
                }
            }
        }

        let system = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n"))
        };

        (converted, system)
    }

    // ==================== 请求体构造 ====================

    /// 构造 Anthropic chat 请求体
    ///
    /// 对齐 Python v1 `AnthropicAdapter.chat` 的 body 构造：
    /// - `model` / `messages` / `max_tokens`（必填，默认 1024）
    /// - `system`：从 system 消息提取（若存在）
    /// - `temperature` / `top_p` / `top_k`：透传
    /// - `stop` → `stop_sequences`（统一为数组）
    /// - `tools` → Anthropic 原生格式（name/description/input_schema）
    /// - `tool_choice` → `{"type":"none"|"auto"|"any"}`；`parallel_tool_calls`
    ///   语义取反为 `disable_parallel_tool_use`
    /// - `reasoning_effort` → `thinking`（对齐 Python ANTHROPIC_MAPPING value_map）
    /// - `extra` 透传到顶层
    fn build_chat_body(req: &ChatRequest, stream: bool) -> Value {
        let (messages, system) = Self::convert_messages(&req.messages);
        let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);

        let mut body = json!({
            "model": req.model,
            "messages": messages,
            "max_tokens": max_tokens,
        });

        if stream {
            body["stream"] = json!(true);
        }
        if let Some(sys) = system {
            body["system"] = json!(sys);
        }
        if let Some(t) = req.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(top_p) = req.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(top_k) = req.top_k {
            body["top_k"] = json!(top_k);
        }
        if let Some(stop) = &req.stop {
            let seqs: Vec<String> = match stop {
                StopSeq::Single(s) => vec![s.clone()],
                StopSeq::Multiple(v) => v.clone(),
            };
            body["stop_sequences"] = json!(seqs);
        }

        // tools → Anthropic 原生 tools 参数（展开 OpenAI 风格 type/function 包装）
        if let Some(tools) = &req.tools {
            let converted: Vec<Value> = tools.iter().filter_map(convert_tool_definition).collect();
            if !converted.is_empty() {
                body["tools"] = json!(converted);
            }
        }
        // tool_choice → {"type":...}（含 parallel_tool_calls → disable_parallel_tool_use 取反）
        if let Some(tool_choice) = build_tool_choice(&req.tool_choice, req.parallel_tool_calls) {
            body["tool_choice"] = tool_choice;
        }

        // reasoning_effort → thinking（对齐 Python ANTHROPIC_MAPPING.value_map）
        // 仅在未通过 extra 显式设置 thinking 时注入。
        // 注：本函数手工逐字段构建 body，架构上不经 ParameterMapping::apply
        // （anthropic_mapping() 仅保留 rename 规则作参考与测试），故该 value_map
        // 不迁移进 anthropic_mapping()，保持此处硬编码注入。
        if let Some(effort) = req.reasoning_effort {
            if body.get("thinking").is_none() {
                if let Some(thinking) = thinking_for_effort(effort) {
                    body["thinking"] = thinking;
                }
            }
        }

        // extra 透传到顶层
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }

        body
    }

    // ==================== 响应解析 ====================

    /// 解析 Anthropic chat 响应 → ChatCompletion
    ///
    /// Anthropic 响应格式：
    /// ```json
    /// {"id":"msg_1","model":"claude-...","stop_reason":"end_turn",
    ///  "content":[{"type":"text","text":"Hello"}],
    ///  "usage":{"input_tokens":3,"output_tokens":2}}
    /// ```
    /// 拼接所有 type==text 的 content block 文本；type==tool_use 的块解析为统一
    /// ToolCall（input 对象序列化为 JSON 字符串存入 arguments）；stop_reason 映射为
    /// finish_reason（tool_use → tool_calls）。
    /// 对应 Python v1 `AnthropicAdapter._parse_response`（Python 未解析 tool_use，此处补齐）。
    fn parse_chat_completion(value: &Value, fallback_model: &str) -> Result<ChatCompletion> {
        let blocks = value.get("content").and_then(|v| v.as_array());

        // 拼接所有 text 内容块
        let content: String = blocks
            .map(|arr| {
                arr.iter()
                    .filter_map(|block| {
                        if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                            block
                                .get("text")
                                .and_then(|t| t.as_str())
                                .map(str::to_owned)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();

        // 提取 tool_use 内容块 → 统一 ToolCall
        let tool_calls: Vec<ToolCall> = blocks
            .map(|arr| arr.iter().filter_map(parse_tool_use_block).collect())
            .unwrap_or_default();

        let stop_reason = value.get("stop_reason").and_then(|v| v.as_str());
        let finish_reason = stop_reason.map(map_stop_reason);

        let usage = value.get("usage").and_then(parse_anthropic_usage);

        Ok(ChatCompletion {
            id: value
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| util::generate_id("chatcmpl")),
            object: "chat.completion".to_string(),
            created: value
                .get("created")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(util::current_timestamp),
            model: value
                .get("model")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| fallback_model.to_string()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChoiceMessage {
                    role: "assistant".to_string(),
                    // 纯工具调用响应（无文本）时 content 为 None，对齐 OpenAI 语义
                    content: if content.is_empty() && !tool_calls.is_empty() {
                        None
                    } else {
                        Some(content)
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                },
                finish_reason,
            }],
            usage,
            service_tier: None,
            system_fingerprint: None,
        })
    }

    /// 解析单个 Anthropic 流式事件 → Option<ChatCompletionChunk>
    ///
    /// Anthropic 流式事件（JSON 的 `type` 字段）：
    /// - `content_block_delta`（delta.type==text_delta）→ 增量文本 chunk
    /// - `message_delta`（delta.stop_reason）→ 结束原因 chunk（finish_reason）
    /// - `message_stop` → 返回 None 并由调用方结束流
    /// - 其他（message_start / content_block_start / content_block_stop / ping）→ None（跳过）
    ///
    /// 对应 Python v1 `AnthropicAdapter._parse_stream_chunk`。
    /// 注意：tool_use 相关事件（content_block_start 的 tool_use 块、input_json_delta）
    /// 同样跳过不产出增量——Python v1 流式未支持工具调用，对齐其行为；
    /// message_delta 的 stop_reason=tool_use 仍会映射为 finish_reason=tool_calls。
    fn parse_chunk(value: &Value, fallback_model: &str) -> Option<ChatCompletionChunk> {
        let event_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let chunk_id = value
            .get("message")
            .and_then(|m| m.get("id"))
            .and_then(|v| v.as_str())
            .or_else(|| value.get("id").and_then(|v| v.as_str()))
            .map(str::to_owned)
            .unwrap_or_else(|| util::generate_id("chatcmpl"));

        match event_type {
            "content_block_delta" => {
                let delta = value.get("delta").cloned().unwrap_or(Value::Null);
                if delta.get("type").and_then(|t| t.as_str()) == Some("text_delta") {
                    let text = delta
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string();
                    // 空文本不产生 chunk（与 Python 一致：text 为空时不 append）
                    if text.is_empty() {
                        return None;
                    }
                    Some(make_chunk(chunk_id, fallback_model, Some(text), None))
                } else {
                    None
                }
            }
            "message_delta" => {
                let delta = value.get("delta").cloned().unwrap_or(Value::Null);
                let stop_reason = delta.get("stop_reason").and_then(|v| v.as_str());
                stop_reason.map(|r| {
                    make_chunk(
                        chunk_id,
                        fallback_model,
                        Some(String::new()),
                        Some(map_stop_reason(r)),
                    )
                })
            }
            // message_stop / message_start / content_block_start / content_block_stop / ping 等
            // 不产生有效 chunk
            _ => None,
        }
    }

    /// 解析 Anthropic /models 响应 → Vec<ModelInfo>
    ///
    /// Anthropic 响应：`{"data":[{"id":"...","display_name":"...","type":"model",...}]}`。
    /// `display_name` 作为模型显示名与描述。
    fn parse_models(value: &Value, provider: &str) -> Vec<ModelInfo> {
        let arr = value.get("data").and_then(|v| v.as_array());
        match arr {
            Some(arr) => arr
                .iter()
                .map(|m| {
                    let id = m
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let display_name = m
                        .get("display_name")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned);
                    let model_type = infer_model_type(&id);
                    ModelInfo {
                        name: display_name.clone().unwrap_or_else(|| id.clone()),
                        id,
                        model_type,
                        provider: provider.to_string(),
                        capabilities: Vec::new(),
                        max_tokens: None,
                        supports_streaming: matches!(model_type, ModelType::Chat),
                        description: display_name,
                        created: None,
                    }
                })
                .collect(),
            None => Vec::new(),
        }
    }
}

#[async_trait]
impl Adapter for AnthropicAdapter {
    fn provider_type(&self) -> &str {
        Self::PROVIDER_TYPE
    }

    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
    }

    fn capabilities(&self) -> CapabilitySet {
        self.compat.capabilities_set().clone()
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    async fn start(&mut self) -> Result<()> {
        // HttpClient 已在构造时初始化，无需额外启动
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        // reqwest::Client 自身管理连接池生命周期，无需显式关闭
        Ok(())
    }

    /// 文本对话（Anthropic 特有协议：POST /messages）
    async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.ensure_capability(Capabilities::Chat)?;
        let body = Self::build_chat_body(&req, false);
        let value = self.post_authed_json("messages", &body).await?;
        Self::parse_chat_completion(&value, &req.model)
    }

    /// 流式文本对话（Anthropic 特有协议：POST /messages stream=true）
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.ensure_capability(Capabilities::ChatStream)?;
        let body = Self::build_chat_body(&req, true);
        let url = self.url("messages");

        let resp = self.send_post(&url, &body).await.map_err(|e| {
            if e.is_timeout() {
                AibridgeError::Timeout
            } else {
                AibridgeError::Network(e)
            }
        })?;
        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(Self::map_api_error(status_code, &body_text));
        }

        let model = req.model.clone();
        // 按字节流读取，按行切分解析 SSE（Anthropic 流式同为 SSE data: 格式）
        let byte_stream = resp
            .bytes_stream()
            .map_err(|e| e.to_string())
            .map(|r| r.map(|b| b.to_vec()));
        let lines_stream = AnthropicLinesStream::new(byte_stream);

        let stream = async_stream::stream! {
            let mut s = lines_stream;
            while let Some(line_result) = s.next().await {
                let line = match line_result {
                    Ok(l) => l,
                    Err(msg) => {
                        yield Err(AibridgeError::Api {
                            status: 0,
                            message: format!("流式读取错误: {msg}"),
                        });
                        return;
                    }
                };
                let line = line.trim();
                // 空行或注释行（心跳）跳过
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }
                // 仅处理 data: 行（event: 行由 JSON 内 type 字段判断，跳过）
                let data = if let Some(rest) = line.strip_prefix("data: ") {
                    rest
                } else if let Some(rest) = line.strip_prefix("data:") {
                    rest
                } else {
                    continue;
                };
                // 结束标记（Anthropic 通常不发 [DONE]，但兼容处理）
                if data.trim() == "[DONE]" {
                    return;
                }
                match serde_json::from_str::<Value>(data) {
                    Ok(v) => {
                        // message_stop 事件：结束流
                        if v.get("type").and_then(|t| t.as_str()) == Some("message_stop") {
                            return;
                        }
                        match Self::parse_chunk(&v, &model) {
                            Some(chunk) => yield Ok(chunk),
                            None => continue,
                        }
                    }
                    // 单行 JSON 解析失败不致命，跳过（与 Python 老版一致）
                    Err(_) => continue,
                }
            }
        };

        Ok(stream.boxed())
    }

    /// 模型列表（实时拉取）：GET /models
    async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let value = self.get_authed_json("models").await?;
        let models = Self::parse_models(&value, self.provider_type());
        Ok(match filter {
            Some(t) => models.into_iter().filter(|m| m.model_type == t).collect(),
            None => models,
        })
    }

    // image_generate / video_create / video_poll / embed / transcribe / speech / list_voices
    // 走 trait 默认实现，返 UnsupportedCapability，与 Python 老版行为一致。
}

// ==================== 辅助函数 ====================

/// 转换 assistant 消息为 Anthropic 格式
///
/// - 无 tool_calls：保持字符串 content（与纯文本对话一致）
/// - 有 tool_calls：content 为 blocks 数组，文本块（若有且非空）在前，
///   每个 ToolCall 转为 `{"type":"tool_use","id","name","input"}` 块
fn convert_assistant_message(
    content: &Option<String>,
    tool_calls: &Option<Vec<ToolCall>>,
) -> Value {
    let calls: &[ToolCall] = tool_calls.as_deref().unwrap_or(&[]);
    if calls.is_empty() {
        let text = content.clone().unwrap_or_default();
        return json!({"role": "assistant", "content": text});
    }

    let mut blocks: Vec<Value> = Vec::new();
    if let Some(text) = content {
        if !text.is_empty() {
            blocks.push(json!({"type": "text", "text": text}));
        }
    }
    for call in calls {
        blocks.push(json!({
            "type": "tool_use",
            "id": call.id,
            "name": call.function.name,
            "input": parse_tool_arguments(&call.function.arguments),
        }));
    }
    json!({"role": "assistant", "content": blocks})
}

/// 判断一条已转换消息是否为 tool_result user 消息（用于连续 tool 消息合并）
///
/// 判定条件：role==user 且 content 为非空数组、所有元素均为 tool_result 块
/// （排除普通多模态 user 消息与空数组的误判）。
fn is_tool_result_message(v: &Value) -> bool {
    if v.get("role").and_then(|r| r.as_str()) != Some("user") {
        return false;
    }
    v.get("content")
        .and_then(|c| c.as_array())
        .map(|arr| {
            !arr.is_empty()
                && arr
                    .iter()
                    .all(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_result"))
        })
        .unwrap_or(false)
}

/// 将 ToolCall 的 arguments（JSON 字符串）解析为 Anthropic tool_use 的 input 对象
///
/// Anthropic 要求 input 必须是 JSON 对象；非法 JSON 或非对象值兜底为空对象
/// （保留调用本身，不丢 id/name 信息）。
fn parse_tool_arguments(arguments: &str) -> Value {
    match serde_json::from_str::<Value>(arguments) {
        Ok(v) if v.is_object() => v,
        _ => json!({}),
    }
}

/// 转换统一 ToolDefinition 为 Anthropic tools 参数格式
///
/// OpenAI 风格 `{"type":"function","function":{name,description,parameters}}`
/// → Anthropic 原生 `{"name","description","input_schema"}`。
/// parameters 缺省时兜底最小合法 schema（Anthropic 要求 input_schema 必填）；
/// 无 function 定义的工具（如 web_search）无法映射为 Anthropic 自定义工具，跳过。
fn convert_tool_definition(tool: &ToolDefinition) -> Option<Value> {
    let func = tool.function.as_ref()?;
    let mut t = json!({"name": func.name});
    if let Some(desc) = &func.description {
        t["description"] = json!(desc);
    }
    t["input_schema"] = func
        .parameters
        .clone()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
    Some(t)
}

/// 构建 Anthropic tool_choice 参数
///
/// 映射规则：
/// - ToolChoice::None → `{"type":"none"}`
/// - ToolChoice::Auto → `{"type":"auto"}`
/// - ToolChoice::Required → `{"type":"any"}`（强制调用至少一个工具）
/// - parallel_tool_calls=false → `disable_parallel_tool_use: true`（语义取反；
///   仅 auto/any 支持该字段，none 不支持）
/// - 未设置 tool_choice 但 parallel_tool_calls=false → 注入 `{"type":"auto",...}`
///   （Anthropic 提供 tools 时默认即 auto）
/// - parallel_tool_calls=true 为 Anthropic 默认行为，不显式输出该字段
fn build_tool_choice(choice: &Option<ToolChoice>, parallel: Option<bool>) -> Option<Value> {
    // 仅 false 需要显式取反；true/None 均为默认行为
    let disable_parallel = matches!(parallel, Some(false));

    let with_disable = |type_str: &str| {
        let mut v = json!({"type": type_str});
        if disable_parallel {
            v["disable_parallel_tool_use"] = json!(true);
        }
        v
    };

    match choice {
        Some(ToolChoice::None) => Some(json!({"type": "none"})),
        Some(ToolChoice::Auto) => Some(with_disable("auto")),
        Some(ToolChoice::Required) => Some(with_disable("any")),
        None if disable_parallel => Some(with_disable("auto")),
        None => None,
    }
}

/// 解析 Anthropic 响应 content 中的 tool_use 块 → 统一 ToolCall
///
/// `{"type":"tool_use","id","name","input":{...}}` → ToolCall（input 对象序列化为
/// JSON 字符串存入 arguments，对齐 OpenAI 统一格式；input 缺省兜底 `{}`）。
/// 非 tool_use 块返回 None。
fn parse_tool_use_block(block: &Value) -> Option<ToolCall> {
    if block.get("type").and_then(|t| t.as_str()) != Some("tool_use") {
        return None;
    }
    let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
    Some(ToolCall {
        id: block
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        tool_type: "function".to_string(),
        function: ToolCallFunction {
            name: block
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            arguments: input.to_string(),
        },
    })
}

/// 将统一多模态内容部件转换为 Anthropic content block
///
/// - Text → `{"type":"text","text":...}`
/// - ImageUrl（data: URI）→ `{"type":"image","source":{"type":"base64",...}}`
/// - ImageUrl（http(s) URL）→ `{"type":"image","source":{"type":"url","url":...}}`
fn convert_content_part(part: &ContentPart) -> Option<Value> {
    match part {
        ContentPart::Text { text } => Some(json!({"type": "text", "text": text})),
        ContentPart::ImageUrl { image_url } => {
            let url = &image_url.url;
            if let Some(rest) = url.strip_prefix("data:") {
                // data:<media_type>;<encoding>,<data>
                let (media_type, data) = match rest.split_once(',') {
                    Some((meta, d)) => {
                        let mt = meta.split(';').next().unwrap_or("image/png");
                        (mt, d)
                    }
                    None => return None,
                };
                Some(json!({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data}
                }))
            } else {
                // http(s) URL：Anthropic 新版支持 url source
                Some(json!({
                    "type": "image",
                    "source": {"type": "url", "url": url}
                }))
            }
        }
    }
}

/// 将 Anthropic stop_reason 映射为统一 finish_reason
///
/// 对应 Python v1 `stop_reason_map`：
/// - end_turn → stop
/// - max_tokens → length
/// - stop_sequence → stop
/// - tool_use → tool_calls
/// - 其他 → 原值透传
fn map_stop_reason(reason: &str) -> String {
    match reason {
        "end_turn" => "stop".into(),
        "max_tokens" => "length".into(),
        "stop_sequence" => "stop".into(),
        "tool_use" => "tool_calls".into(),
        other => other.into(),
    }
}

/// 将 reasoning_effort 映射为 Anthropic thinking 配置
///
/// 对齐 Python v1 `ANTHROPIC_MAPPING.value_map["reasoning_effort"]`：
/// - Low → `{"type":"enabled","budget_tokens":1024}`
/// - Medium → `{"type":"enabled","budget_tokens":4096}`
/// - High → `{"type":"enabled","budget_tokens":16384}`
/// - None / Auto → 不注入（返回 None）
///
/// 注意：Anthropic 要求 `budget_tokens < max_tokens`，调用方需确保 max_tokens 足够大。
fn thinking_for_effort(effort: ReasoningEffort) -> Option<Value> {
    match effort {
        ReasoningEffort::Low => Some(json!({"type": "enabled", "budget_tokens": 1024})),
        ReasoningEffort::Medium => Some(json!({"type": "enabled", "budget_tokens": 4096})),
        ReasoningEffort::High => Some(json!({"type": "enabled", "budget_tokens": 16384})),
        ReasoningEffort::None | ReasoningEffort::Auto => None,
    }
}

/// 构造单个流式 chunk（简化重复代码）
fn make_chunk(
    id: String,
    model: &str,
    content: Option<String>,
    finish_reason: Option<String>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id,
        object: "chat.completion.chunk".to_string(),
        created: util::current_timestamp(),
        model: model.to_string(),
        choices: vec![ChatCompletionDelta {
            index: 0,
            delta: DeltaMessage {
                role: Some("assistant".to_string()),
                content,
                tool_calls: None,
            },
            finish_reason,
        }],
        usage: None,
    }
}

/// 解析 Anthropic usage 统计
///
/// Anthropic usage 格式：`{"input_tokens": N, "output_tokens": M}`，
/// 需转换为统一 ChatUsage（prompt / completion / total）。
fn parse_anthropic_usage(v: &Value) -> Option<crate::model::chat::ChatUsage> {
    let prompt = v.get("input_tokens").and_then(|x| x.as_u64())?;
    let completion = v.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
    Some(crate::model::chat::ChatUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    })
}

// ==================== SSE 行流适配器 ====================

/// 将字节流按行切分的适配器（Anthropic 流式用）
///
/// 与 `openai_compat::LinesStream` 等价，独立实现避免引用其私有结构。
struct AnthropicLinesStream<S> {
    inner: S,
    buffer: Vec<u8>,
}

impl<S> AnthropicLinesStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
        }
    }
}

impl<S> futures::Stream for AnthropicLinesStream<S>
where
    S: futures::Stream<Item = std::result::Result<Vec<u8>, String>> + Unpin,
{
    type Item = std::result::Result<String, String>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;
        loop {
            // 先看缓冲区是否已有完整行
            if let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
                let mut line: Vec<u8> = self.buffer.drain(..=pos).collect();
                // 去掉末尾 \n 与可能的 \r
                if line.last() == Some(&b'\n') {
                    line.pop();
                }
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                let s = String::from_utf8_lossy(&line).into_owned();
                return Poll::Ready(Some(Ok(s)));
            }
            // 缓冲区无完整行，拉取下一 chunk
            match std::pin::Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Err(msg))) => return Poll::Ready(Some(Err(msg))),
                Poll::Ready(Some(Ok(chunk))) => {
                    self.buffer.extend_from_slice(&chunk);
                    // 继续循环，尝试从缓冲区切出行
                }
                Poll::Ready(None) => {
                    // 流结束，把缓冲区剩余内容作为最后一行返回
                    if !self.buffer.is_empty() {
                        let mut line = std::mem::take(&mut self.buffer);
                        if line.last() == Some(&b'\n') {
                            line.pop();
                        }
                        if line.last() == Some(&b'\r') {
                            line.pop();
                        }
                        let s = String::from_utf8_lossy(&line).into_owned();
                        return Poll::Ready(Some(Ok(s)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientOptions;
    use crate::http::HttpClient;
    use crate::model::image::ImageRequest;
    use crate::model::options::{
        EmbedInput, EmbedRequest, FunctionDefinition, ReasoningEffort, ToolCall, ToolCallFunction,
        ToolChoice, ToolDefinition,
    };
    use crate::model::video::VideoRequest;
    use mockito::Server;
    use std::collections::HashMap;

    // ==================== 通用测试辅助 ====================

    /// 构造测试用 AnthropicAdapter（指向 mockito server）
    fn make_anthropic(server: &Server) -> AnthropicAdapter {
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .timeout(5)
            .build();
        let config = ProviderConfig::from_options("anthropic", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        let compat = OpenAiCompatAdapter::with_http(
            http,
            config,
            "anthropic",
            "Anthropic Claude",
            anthropic_capabilities(),
        );
        AnthropicAdapter::with_compat(compat)
    }

    /// 标准 Anthropic chat 成功响应体
    fn anthropic_chat_body() -> Value {
        json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })
    }

    // ==================== 元信息测试 ====================

    #[tokio::test]
    async fn provider_type_and_name() {
        let server = Server::new_async().await;
        let adapter = make_anthropic(&server);
        assert_eq!(adapter.provider_type(), "anthropic");
        assert_eq!(adapter.provider_name(), "Anthropic Claude");
    }

    #[tokio::test]
    async fn requires_api_key() {
        let server = Server::new_async().await;
        let adapter = make_anthropic(&server);
        assert!(adapter.requires_api_key());
    }

    #[tokio::test]
    async fn capabilities_include_chat_and_stream() {
        let server = Server::new_async().await;
        let adapter = make_anthropic(&server);
        let caps = adapter.capabilities();
        assert!(caps.contains(&Capabilities::Chat));
        assert!(caps.contains(&Capabilities::ChatStream));
        assert!(caps.contains(&Capabilities::Vision));
        // 不支持 image / embed / video
        assert!(!caps.contains(&Capabilities::ImageGenerate));
        assert!(!caps.contains(&Capabilities::Embedding));
        assert!(!caps.contains(&Capabilities::VideoGenerate));
    }

    #[tokio::test]
    async fn start_close_are_noops() {
        let server = Server::new_async().await;
        let mut adapter = make_anthropic(&server);
        adapter.start().await.unwrap();
        adapter.close().await.unwrap();
    }

    #[test]
    fn default_base_url_matches_spec() {
        assert_eq!(DEFAULT_ANTHROPIC_BASE_URL, "https://api.anthropic.com/v1");
        assert_eq!(ANTHROPIC_API_VERSION, "2023-06-01");
    }

    // ==================== chat 正常路径 ====================

    #[tokio::test]
    async fn chat_success_parses_completion() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_header("x-api-key", "test-key")
            .match_header("anthropic-version", "2023-06-01")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(anthropic_chat_body().to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let resp = adapter.chat(req).await.expect("chat 应成功");

        assert_eq!(resp.id, "msg_1");
        assert_eq!(resp.model, "claude-3-5-sonnet-20241022");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        // usage 解析：input/output → prompt/completion/total
        let usage = resp.usage.as_ref().expect("应有 usage");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_sends_x_api_key_and_anthropic_version_headers() {
        // 验证 Anthropic 特有认证 header（x-api-key + anthropic-version），非 Bearer
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_header("x-api-key", "test-key")
            .match_header("anthropic-version", "2023-06-01")
            .match_header("authorization", mockito::Matcher::Missing)
            .with_status(200)
            .with_body(anthropic_chat_body().to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_extracts_system_to_top_level_field() {
        // system 消息应提取到顶层 system 字段，不在 messages 中
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "claude-3-5-sonnet-20241022",
                "system": "you are helpful",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100
            })))
            .with_status(200)
            .with_body(anthropic_chat_body().to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder(
            "claude-3-5-sonnet-20241022",
            vec![
                ChatMessage::system("you are helpful"),
                ChatMessage::user("hi"),
            ],
        )
        .max_tokens(100)
        .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_converts_user_and_assistant_messages() {
        // 多轮对话：user/assistant 消息转为 Anthropic messages（system 提取到顶层）
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::PartialJson(json!({
                "messages": [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "second question"}
                ]
            })))
            .with_status(200)
            .with_body(anthropic_chat_body().to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder(
            "claude-3-5-sonnet-20241022",
            vec![
                ChatMessage::user("first question"),
                ChatMessage::assistant("first answer"),
                ChatMessage::user("second question"),
            ],
        )
        .max_tokens(100)
        .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_default_max_tokens_when_not_provided() {
        // max_tokens 必填，未指定时兜底 1024
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::PartialJson(json!({
                "max_tokens": 1024
            })))
            .with_status(200)
            .with_body(anthropic_chat_body().to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_maps_stop_to_stop_sequences() {
        // stop → stop_sequences（统一为数组）
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::PartialJson(json!({
                "stop_sequences": ["END", "STOP"]
            })))
            .with_status(200)
            .with_body(anthropic_chat_body().to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .stop(StopSeq::Multiple(vec!["END".into(), "STOP".into()]))
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_passes_temperature_top_p_top_k() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::PartialJson(json!({
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            })))
            .with_status(200)
            .with_body(anthropic_chat_body().to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .temperature(0.7)
            .top_p(0.9)
            .top_k(40)
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_passes_extra_params_through() {
        // extra 透传到顶层
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::PartialJson(json!({
                "custom_param": "custom_value"
            })))
            .with_status(200)
            .with_body(anthropic_chat_body().to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .extra("custom_param", "custom_value")
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_injects_thinking_for_reasoning_effort() {
        // reasoning_effort → thinking（对齐 Python ANTHROPIC_MAPPING.value_map）
        // Anthropic 协议不接受 reasoning_effort 字段，仅注入 thinking
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::PartialJson(json!({
                "thinking": {"type": "enabled", "budget_tokens": 16384},
                "max_tokens": 20000
            })))
            .with_status(200)
            .with_body(anthropic_chat_body().to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(20000)
            .reasoning_effort(ReasoningEffort::High)
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_no_thinking_without_reasoning_effort() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .with_status(200)
            .with_body(anthropic_chat_body().to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let _ = adapter.chat(req).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_concatenates_multiple_text_blocks() {
        // content 数组含多个 text 块时，应拼接为完整文本
        let mut server = Server::new_async().await;
        let body = json!({
            "id": "msg_2",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " world"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2}
        });
        server
            .mock("POST", "/messages")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let resp = adapter.chat(req).await.unwrap();
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("Hello world")
        );
    }

    #[tokio::test]
    async fn chat_maps_various_stop_reasons() {
        let cases = [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("stop_sequence", "stop"),
            ("tool_use", "tool_calls"),
        ];
        for (stop_reason, expected_finish) in cases {
            let mut server = Server::new_async().await;
            let body = json!({
                "id": "msg_x",
                "model": "claude-3-5-sonnet-20241022",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": stop_reason,
                "usage": {"input_tokens": 1, "output_tokens": 1}
            });
            server
                .mock("POST", "/messages")
                .with_status(200)
                .with_body(body.to_string())
                .create_async()
                .await;
            let adapter = make_anthropic(&server);
            let req =
                ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
                    .max_tokens(100)
                    .build();
            let resp = adapter.chat(req).await.unwrap();
            assert_eq!(
                resp.choices[0].finish_reason.as_deref(),
                Some(expected_finish),
                "stop_reason={stop_reason} 应映射为 {expected_finish}"
            );
        }
    }

    // ==================== chat 错误路径 ====================

    #[tokio::test]
    async fn chat_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/messages")
            .with_status(401)
            .with_body(
                json!({
                    "type": "error",
                    "error": {"type": "authentication_error", "message": "invalid x-api-key"}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[tokio::test]
    async fn chat_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/messages")
            .with_status(429)
            .with_body(
                json!({
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "rate limit exceeded"}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::RateLimit { .. }));
    }

    #[tokio::test]
    async fn chat_error_404_returns_model_not_found() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/messages")
            .with_status(404)
            .with_body(
                json!({
                    "type": "error",
                    "error": {"type": "not_found_error", "message": "model not found"}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("bad-model", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn chat_error_400_returns_validation() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/messages")
            .with_status(400)
            .with_body(
                json!({
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "max_tokens is invalid"}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Validation { message, .. } => {
                assert!(message.contains("max_tokens"));
            }
            _ => panic!("应为 Validation"),
        }
    }

    #[tokio::test]
    async fn chat_error_500_returns_api() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/messages")
            .with_status(500)
            .with_body(
                json!({
                    "type": "error",
                    "error": {"type": "api_error", "message": "internal server error"}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let err = adapter.chat(req).await.unwrap_err();
        match err {
            AibridgeError::Api { status, .. } => assert_eq!(status, 500),
            _ => panic!("应为 Api"),
        }
    }

    #[tokio::test]
    async fn chat_unsupported_capability_returns_error() {
        // 不支持 Chat 能力时应返 UnsupportedCapability
        let server = Server::new_async().await;
        let opts = ClientOptions::builder()
            .api_key("test-key")
            .base_url(server.url())
            .build();
        let config = ProviderConfig::from_options("anthropic", opts);
        let http =
            HttpClient::new(&ClientOptions::builder().base_url(server.url()).build()).unwrap();
        let compat = OpenAiCompatAdapter::with_http(
            http,
            config,
            "anthropic",
            "Anthropic Claude",
            CapabilitySet::new(),
        );
        let adapter = AnthropicAdapter::with_compat(compat);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let err = adapter.chat(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ==================== chat_stream 正常 + 错误路径 ====================

    #[tokio::test]
    async fn chat_stream_parses_sse_events() {
        // Anthropic 流式：content_block_delta + message_delta + message_stop
        let mut server = Server::new_async().await;
        let sse = "event: message_start\n\
                   data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-5-sonnet-20241022\"}}\n\n\
                   event: content_block_delta\n\
                   data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n\
                   event: content_block_delta\n\
                   data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\n\
                   event: message_delta\n\
                   data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n\
                   event: message_stop\n\
                   data: {\"type\":\"message_stop\"}\n\n";
        server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let mut stream = adapter.chat_stream(req).await.expect("stream 应建立");
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        // 2 个文本 delta + 1 个 message_delta（finish_reason）
        assert_eq!(chunks.len(), 3);
        // 拼接前两块内容
        let mut content = String::new();
        content.push_str(chunks[0].choices[0].delta.content.as_deref().unwrap_or(""));
        content.push_str(chunks[1].choices[0].delta.content.as_deref().unwrap_or(""));
        assert_eq!(content, "Hello world");
        // 第三块是 message_delta，finish_reason = stop
        assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn chat_stream_skips_non_text_delta_events() {
        // content_block_start / ping 等事件应被跳过
        let mut server = Server::new_async().await;
        let sse = "event: message_start\n\
                   data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n\
                   event: content_block_start\n\
                   data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
                   event: content_block_delta\n\
                   data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n\
                   event: ping\n\
                   data: {\"type\":\"ping\"}\n\n\
                   event: message_stop\n\
                   data: {\"type\":\"message_stop\"}\n\n";
        server
            .mock("POST", "/messages")
            .with_status(200)
            .with_body(sse)
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        // 仅 1 个有效 chunk（content_block_delta）
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("Hi"));
    }

    #[tokio::test]
    async fn chat_stream_sends_stream_true_and_headers() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_header("x-api-key", "test-key")
            .match_header("anthropic-version", "2023-06-01")
            .match_body(mockito::Matcher::PartialJson(json!({
                "stream": true,
                "max_tokens": 100
            })))
            .with_status(200)
            .with_body("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let mut stream = adapter.chat_stream(req).await.unwrap();
        while stream.next().await.is_some() {}
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_stream_error_401_returns_authentication() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/messages")
            .with_status(401)
            .with_body(
                json!({"type":"error","error":{"type":"authentication_error","message":"bad key"}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let result = adapter.chat_stream(req).await;
        match result {
            Err(e) => assert!(matches!(e, AibridgeError::Authentication { .. })),
            Ok(_) => panic!("chat_stream 应返回错误而非 stream"),
        }
    }

    #[tokio::test]
    async fn chat_stream_error_429_returns_rate_limit() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/messages")
            .with_status(429)
            .with_body(
                json!({"type":"error","error":{"type":"rate_limit_error","message":"slow down"}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let result = adapter.chat_stream(req).await;
        match result {
            Err(e) => assert!(matches!(e, AibridgeError::RateLimit { .. })),
            Ok(_) => panic!("chat_stream 应返回错误而非 stream"),
        }
    }

    // ==================== list_models ====================

    #[tokio::test]
    async fn list_models_success() {
        let mut server = Server::new_async().await;
        let body = json!({
            "data": [
                {"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet", "type": "model"},
                {"id": "claude-3-opus-20240229", "display_name": "Claude 3 Opus", "type": "model"}
            ]
        });
        server
            .mock("GET", "/models")
            .match_header("x-api-key", "test-key")
            .match_header("anthropic-version", "2023-06-01")
            .with_status(200)
            .with_body(body.to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let models = adapter.list_models(None).await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "claude-3-5-sonnet-20241022");
        assert_eq!(models[0].name, "Claude 3.5 Sonnet");
        assert_eq!(models[0].provider, "anthropic");
        assert_eq!(models[0].description.as_deref(), Some("Claude 3.5 Sonnet"));
    }

    #[tokio::test]
    async fn list_models_error_401() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/models")
            .with_status(401)
            .with_body(
                json!({"type":"error","error":{"type":"authentication_error","message":"bad key"}})
                    .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let err = adapter.list_models(None).await.unwrap_err();
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    // ==================== 不支持能力 ====================

    #[tokio::test]
    async fn image_generate_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_anthropic(&server);
        let req = ImageRequest::builder("claude-3", "a cat").build();
        let err = adapter.image_generate(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn video_create_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_anthropic(&server);
        let req = VideoRequest::builder("claude-3", "a cat walking").build();
        let err = adapter.video_create(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    #[tokio::test]
    async fn embed_unsupported() {
        let server = Server::new_async().await;
        let adapter = make_anthropic(&server);
        let req = EmbedRequest {
            model: "claude-3".into(),
            input: EmbedInput::Single("hi".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let err = adapter.embed(req).await.unwrap_err();
        assert!(matches!(err, AibridgeError::UnsupportedCapability { .. }));
    }

    // ==================== 工具调用（ToolCall）====================

    /// 构造测试用天气工具定义（OpenAI 风格 function 包装）
    fn weather_tool() -> ToolDefinition {
        ToolDefinition::function(FunctionDefinition {
            name: "get_weather".into(),
            description: Some("获取指定城市的天气".into()),
            parameters: Some(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            })),
        })
    }

    /// 构造测试用 ToolCall
    fn make_tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            tool_type: "function".into(),
            function: ToolCallFunction {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    #[tokio::test]
    async fn capabilities_include_tool_call() {
        let server = Server::new_async().await;
        let adapter = make_anthropic(&server);
        assert!(adapter.capabilities().contains(&Capabilities::ToolCall));
    }

    #[test]
    fn build_chat_body_converts_tools_to_anthropic_format() {
        // OpenAI 风格 {"type":"function","function":{...}} → Anthropic 原生
        // {"name","description","input_schema"}（无 type/function 包装）
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .tools(vec![weather_tool()])
            .build();
        let body = AnthropicAdapter::build_chat_body(&req, false);
        let tools = body["tools"].as_array().expect("tools 应为数组");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "get_weather");
        assert_eq!(tools[0]["description"], "获取指定城市的天气");
        assert_eq!(tools[0]["input_schema"]["type"], "object");
        assert_eq!(
            tools[0]["input_schema"]["properties"]["city"]["type"],
            "string"
        );
        assert!(tools[0].get("type").is_none());
        assert!(tools[0].get("function").is_none());
    }

    #[test]
    fn build_chat_body_tool_without_parameters_gets_minimal_schema() {
        // parameters 缺省时兜底最小合法 schema（Anthropic 要求 input_schema 必填）
        let tool = ToolDefinition::function(FunctionDefinition {
            name: "ping".into(),
            description: None,
            parameters: None,
        });
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .tools(vec![tool])
            .build();
        let body = AnthropicAdapter::build_chat_body(&req, false);
        let tools = body["tools"].as_array().expect("tools 应为数组");
        assert_eq!(tools[0]["name"], "ping");
        assert!(tools[0].get("description").is_none());
        assert_eq!(
            tools[0]["input_schema"],
            json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn build_chat_body_tool_choice_variants() {
        // None → none、Auto → auto、Required → any（强制调用至少一个工具）
        for (choice, expected_type) in [
            (ToolChoice::None, "none"),
            (ToolChoice::Auto, "auto"),
            (ToolChoice::Required, "any"),
        ] {
            let req =
                ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
                    .max_tokens(100)
                    .tools(vec![weather_tool()])
                    .tool_choice(choice)
                    .build();
            let body = AnthropicAdapter::build_chat_body(&req, false);
            assert_eq!(
                body["tool_choice"]["type"],
                json!(expected_type),
                "tool_choice 应映射为 type={expected_type}"
            );
        }
    }

    #[test]
    fn build_chat_body_parallel_tool_calls_false_disables_parallel() {
        // parallel_tool_calls=false → disable_parallel_tool_use=true（语义取反）
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .tools(vec![weather_tool()])
            .tool_choice(ToolChoice::Auto)
            .parallel_tool_calls(false)
            .build();
        let body = AnthropicAdapter::build_chat_body(&req, false);
        assert_eq!(body["tool_choice"]["type"], "auto");
        assert_eq!(body["tool_choice"]["disable_parallel_tool_use"], true);
    }

    #[test]
    fn build_chat_body_parallel_tool_calls_false_without_tool_choice_injects_auto() {
        // 未设置 tool_choice 时，parallel_tool_calls=false 注入 auto + disable
        //（Anthropic 提供 tools 时默认即 auto）
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .tools(vec![weather_tool()])
            .parallel_tool_calls(false)
            .build();
        let body = AnthropicAdapter::build_chat_body(&req, false);
        assert_eq!(body["tool_choice"]["type"], "auto");
        assert_eq!(body["tool_choice"]["disable_parallel_tool_use"], true);
    }

    #[test]
    fn build_chat_body_parallel_tool_calls_true_omits_disable() {
        // parallel_tool_calls=true 为 Anthropic 默认行为，不显式输出该字段
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .tools(vec![weather_tool()])
            .tool_choice(ToolChoice::Auto)
            .parallel_tool_calls(true)
            .build();
        let body = AnthropicAdapter::build_chat_body(&req, false);
        assert_eq!(body["tool_choice"]["type"], "auto");
        assert!(body["tool_choice"].get("disable_parallel_tool_use").is_none());
    }

    #[test]
    fn build_chat_body_tool_choice_none_ignores_parallel() {
        // tool_choice=none 不支持 disable_parallel_tool_use（Anthropic 协议限制）
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .tools(vec![weather_tool()])
            .tool_choice(ToolChoice::None)
            .parallel_tool_calls(false)
            .build();
        let body = AnthropicAdapter::build_chat_body(&req, false);
        assert_eq!(body["tool_choice"]["type"], "none");
        assert!(body["tool_choice"].get("disable_parallel_tool_use").is_none());
    }

    #[test]
    fn build_chat_body_no_tools_omits_tools_and_tool_choice() {
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(100)
            .build();
        let body = AnthropicAdapter::build_chat_body(&req, false);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn convert_messages_assistant_tool_calls_to_tool_use_blocks() {
        // assistant 带 tool_calls → content blocks：文本块在前，tool_use 块在后；
        // arguments（JSON 字符串）parse 为 input 对象
        let msgs = vec![
            ChatMessage::user("北京天气?"),
            ChatMessage::Assistant {
                content: Some("我来查一下".into()),
                tool_calls: Some(vec![make_tool_call(
                    "toolu_1",
                    "get_weather",
                    "{\"city\":\"Beijing\"}",
                )]),
            },
        ];
        let (converted, _) = AnthropicAdapter::convert_messages(&msgs);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[1]["role"], "assistant");
        let content = converted[1]["content"]
            .as_array()
            .expect("assistant 带 tool_calls 时 content 应为 blocks 数组");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "我来查一下");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["id"], "toolu_1");
        assert_eq!(content[1]["name"], "get_weather");
        assert_eq!(content[1]["input"]["city"], "Beijing");
    }

    #[test]
    fn convert_messages_assistant_tool_calls_without_text() {
        // content 为 None：只有 tool_use 块，不产生空文本块
        let msgs = vec![ChatMessage::Assistant {
            content: None,
            tool_calls: Some(vec![make_tool_call("toolu_1", "ping", "{}")]),
        }];
        let (converted, _) = AnthropicAdapter::convert_messages(&msgs);
        let content = converted[0]["content"].as_array().expect("应为 blocks 数组");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["input"], json!({}));
    }

    #[test]
    fn convert_messages_assistant_tool_calls_invalid_arguments_fallback_empty_object() {
        // arguments 非法 JSON → input 兜底空对象（Anthropic 要求 input 必须为对象）
        let msgs = vec![ChatMessage::Assistant {
            content: None,
            tool_calls: Some(vec![make_tool_call("toolu_1", "ping", "not-json")]),
        }];
        let (converted, _) = AnthropicAdapter::convert_messages(&msgs);
        let content = converted[0]["content"].as_array().expect("应为 blocks 数组");
        assert_eq!(content[0]["input"], json!({}));
    }

    #[test]
    fn convert_messages_tool_result_becomes_user_message() {
        // tool 消息 → user 消息包 tool_result 内容块（Anthropic 原生格式）
        let msgs = vec![
            ChatMessage::assistant("我来查一下"),
            ChatMessage::tool("toolu_1", "晴天 25°C"),
        ];
        let (converted, _) = AnthropicAdapter::convert_messages(&msgs);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[1]["role"], "user");
        let content = converted[1]["content"].as_array().expect("应为数组");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "toolu_1");
        assert_eq!(content[0]["content"], "晴天 25°C");
    }

    #[test]
    fn convert_messages_consecutive_tool_results_merged_into_one_user_message() {
        // 连续多个 tool 消息合并为一条 user 消息：Anthropic 要求角色交替，
        // 且同一轮 tool_use 的所有 tool_result 须放在紧随其后的同一条 user 消息中
        let msgs = vec![
            ChatMessage::Assistant {
                content: None,
                tool_calls: Some(vec![
                    make_tool_call("toolu_1", "get_weather", "{\"city\":\"Beijing\"}"),
                    make_tool_call("toolu_2", "get_weather", "{\"city\":\"Shanghai\"}"),
                ]),
            },
            ChatMessage::tool("toolu_1", "晴天"),
            ChatMessage::tool("toolu_2", "雨天"),
            ChatMessage::user("谢谢"),
        ];
        let (converted, _) = AnthropicAdapter::convert_messages(&msgs);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[1]["role"], "user");
        let content = converted[1]["content"].as_array().expect("应为数组");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["tool_use_id"], "toolu_1");
        assert_eq!(content[0]["content"], "晴天");
        assert_eq!(content[1]["tool_use_id"], "toolu_2");
        assert_eq!(content[1]["content"], "雨天");
        // 后续普通 user 消息不受影响
        assert_eq!(converted[2]["role"], "user");
        assert_eq!(converted[2]["content"], "谢谢");
    }

    #[test]
    fn parse_chat_completion_parses_tool_use_blocks() {
        // 响应 content 中的 tool_use 块 → 统一 ToolCall（input 对象序列化为 JSON 字符串）；
        // stop_reason "tool_use" → finish_reason "tool_calls"
        let v = json!({
            "id": "msg_tool",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {"type": "text", "text": "我来查一下天气。"},
                {"type": "tool_use", "id": "toolu_01A", "name": "get_weather", "input": {"city": "Beijing"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 10}
        });
        let cc = AnthropicAdapter::parse_chat_completion(&v, "fallback").unwrap();
        assert_eq!(
            cc.choices[0].message.content.as_deref(),
            Some("我来查一下天气。")
        );
        let tool_calls = cc.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("应有 tool_calls");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "toolu_01A");
        assert_eq!(tool_calls[0].tool_type, "function");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, "{\"city\":\"Beijing\"}");
        assert_eq!(cc.choices[0].finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn parse_chat_completion_tool_use_only_content_is_none() {
        // 纯 tool_use 响应（无文本块）：content 为 None（对齐 OpenAI 纯工具调用语义）
        let v = json!({
            "id": "msg_tool2",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "ping", "input": {}}
            ],
            "stop_reason": "tool_use"
        });
        let cc = AnthropicAdapter::parse_chat_completion(&v, "m").unwrap();
        assert!(cc.choices[0].message.content.is_none());
        let tool_calls = cc.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("应有 tool_calls");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.arguments, "{}");
    }

    #[tokio::test]
    async fn chat_with_tools_sends_anthropic_format_and_parses_tool_use() {
        // 端到端：请求体 tools/tool_choice 为 Anthropic 原生格式，响应 tool_use 正确解析
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::PartialJson(json!({
                "tools": [{
                    "name": "get_weather",
                    "description": "获取指定城市的天气",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }],
                "tool_choice": {"type": "auto"}
            })))
            .with_status(200)
            .with_body(
                json!({
                    "id": "msg_tc",
                    "model": "claude-3-5-sonnet-20241022",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Beijing"}}
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 30, "output_tokens": 12}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder(
            "claude-3-5-sonnet-20241022",
            vec![ChatMessage::user("北京天气?")],
        )
        .max_tokens(200)
        .tools(vec![weather_tool()])
        .tool_choice(ToolChoice::Auto)
        .build();
        let resp = adapter.chat(req).await.expect("chat 应成功");
        let tool_calls = resp.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("应有 tool_calls");
        assert_eq!(tool_calls[0].id, "toolu_1");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, "{\"city\":\"Beijing\"}");
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_tool_conversation_sends_tool_use_and_tool_result() {
        // 完整工具对话回路：assistant(tool_use) + tool 结果回传 → Anthropic 消息格式
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::PartialJson(json!({
                "messages": [
                    {"role": "user", "content": "北京天气?"},
                    {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Beijing"}}
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "晴天 25°C"}
                    ]}
                ]
            })))
            .with_status(200)
            .with_body(anthropic_chat_body().to_string())
            .create_async()
            .await;

        let adapter = make_anthropic(&server);
        let req = ChatRequest::builder(
            "claude-3-5-sonnet-20241022",
            vec![
                ChatMessage::user("北京天气?"),
                ChatMessage::Assistant {
                    content: None,
                    tool_calls: Some(vec![make_tool_call(
                        "toolu_1",
                        "get_weather",
                        "{\"city\":\"Beijing\"}",
                    )]),
                },
                ChatMessage::tool("toolu_1", "晴天 25°C"),
            ],
        )
        .max_tokens(200)
        .build();
        let _ = adapter.chat(req).await.expect("chat 应成功");
        mock.assert_async().await;
    }

    // ==================== 辅助函数单元测试 ====================

    #[test]
    fn anthropic_mapping_renames_stop_to_stop_sequences() {
        let pm = anthropic_mapping();
        let mut params = HashMap::new();
        params.insert("stop".to_string(), json!("END"));
        params.insert("max_tokens".to_string(), json!(1000));
        params.insert("temperature".to_string(), json!(0.7));
        let result = pm.apply(&params);
        // stop → stop_sequences
        assert_eq!(
            result.get("stop_sequences").and_then(|v| v.as_str()),
            Some("END")
        );
        assert!(!result.contains_key("stop"));
        // 其余原名透传
        assert_eq!(
            result.get("max_tokens").and_then(|v| v.as_i64()),
            Some(1000)
        );
        assert_eq!(
            result.get("temperature").and_then(|v| v.as_f64()),
            Some(0.7)
        );
    }

    #[test]
    fn map_stop_reason_matches_python() {
        assert_eq!(map_stop_reason("end_turn"), "stop");
        assert_eq!(map_stop_reason("max_tokens"), "length");
        assert_eq!(map_stop_reason("stop_sequence"), "stop");
        assert_eq!(map_stop_reason("tool_use"), "tool_calls");
        // 未知值原样透传
        assert_eq!(map_stop_reason("other"), "other");
    }

    #[test]
    fn thinking_for_effort_maps_levels() {
        assert_eq!(
            thinking_for_effort(ReasoningEffort::Low),
            Some(json!({"type": "enabled", "budget_tokens": 1024}))
        );
        assert_eq!(
            thinking_for_effort(ReasoningEffort::Medium),
            Some(json!({"type": "enabled", "budget_tokens": 4096}))
        );
        assert_eq!(
            thinking_for_effort(ReasoningEffort::High),
            Some(json!({"type": "enabled", "budget_tokens": 16384}))
        );
        assert_eq!(thinking_for_effort(ReasoningEffort::None), None);
        assert_eq!(thinking_for_effort(ReasoningEffort::Auto), None);
    }

    #[test]
    fn convert_messages_extracts_system_and_converts_roles() {
        let msgs = vec![
            ChatMessage::system("be helpful"),
            ChatMessage::user("q1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("q2"),
        ];
        let (converted, system) = AnthropicAdapter::convert_messages(&msgs);
        assert_eq!(system.as_deref(), Some("be helpful"));
        // system 不在 messages 中
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0]["role"], "user");
        assert_eq!(converted[0]["content"], "q1");
        assert_eq!(converted[1]["role"], "assistant");
        assert_eq!(converted[1]["content"], "a1");
        assert_eq!(converted[2]["role"], "user");
        assert_eq!(converted[2]["content"], "q2");
    }

    #[test]
    fn convert_messages_joins_multiple_system_messages() {
        let msgs = vec![
            ChatMessage::system("rule 1"),
            ChatMessage::system("rule 2"),
            ChatMessage::user("hi"),
        ];
        let (converted, system) = AnthropicAdapter::convert_messages(&msgs);
        // 多条 system 用 \n 拼接
        assert_eq!(system.as_deref(), Some("rule 1\nrule 2"));
        assert_eq!(converted.len(), 1);
    }

    #[test]
    fn convert_messages_multimodal_user_to_content_blocks() {
        let msgs = vec![ChatMessage::user_multimodal(vec![
            ContentPart::Text {
                text: "describe this".into(),
            },
            ContentPart::ImageUrl {
                image_url: crate::model::chat::ImageUrl::new("data:image/png;base64,aGVsbG8="),
            },
        ])];
        let (converted, system) = AnthropicAdapter::convert_messages(&msgs);
        assert!(system.is_none());
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "user");
        let content = converted[0]["content"].as_array().expect("应为数组");
        assert_eq!(content.len(), 2);
        // 文本块
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "describe this");
        // 图片块（data: URI → base64 source）
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
        assert_eq!(content[1]["source"]["data"], "aGVsbG8=");
    }

    #[test]
    fn convert_content_part_url_image_uses_url_source() {
        let part = ContentPart::ImageUrl {
            image_url: crate::model::chat::ImageUrl::new("https://example.com/x.png"),
        };
        let block = convert_content_part(&part).expect("应返回 Some");
        assert_eq!(block["type"], "image");
        assert_eq!(block["source"]["type"], "url");
        assert_eq!(block["source"]["url"], "https://example.com/x.png");
    }

    #[test]
    fn parse_chat_completion_extracts_text_and_usage() {
        let v = json!({
            "id": "msg_9",
            "model": "claude-3-opus-20240229",
            "content": [{"type": "text", "text": "response text"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        });
        let cc = AnthropicAdapter::parse_chat_completion(&v, "fallback").unwrap();
        assert_eq!(cc.id, "msg_9");
        assert_eq!(cc.model, "claude-3-opus-20240229");
        assert_eq!(
            cc.choices[0].message.content.as_deref(),
            Some("response text")
        );
        assert_eq!(cc.choices[0].finish_reason.as_deref(), Some("stop"));
        let usage = cc.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn parse_chat_completion_without_usage() {
        let v = json!({
            "id": "msg_x",
            "content": [{"type": "text", "text": "no usage"}],
            "stop_reason": "max_tokens"
        });
        let cc = AnthropicAdapter::parse_chat_completion(&v, "fallback").unwrap();
        assert_eq!(cc.choices[0].message.content.as_deref(), Some("no usage"));
        assert_eq!(cc.choices[0].finish_reason.as_deref(), Some("length"));
        assert!(cc.usage.is_none());
        // model 回退到 fallback
        assert_eq!(cc.model, "fallback");
    }

    #[test]
    fn parse_chunk_content_block_delta_returns_text() {
        let v = json!({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hello"}
        });
        let chunk = AnthropicAdapter::parse_chunk(&v, "claude-3").expect("应返回 Some");
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn parse_chunk_empty_text_returns_none() {
        let v = json!({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": ""}
        });
        assert!(AnthropicAdapter::parse_chunk(&v, "claude-3").is_none());
    }

    #[test]
    fn parse_chunk_message_delta_returns_finish_reason() {
        let v = json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"}
        });
        let chunk = AnthropicAdapter::parse_chunk(&v, "claude-3").expect("应返回 Some");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some(""));
    }

    #[test]
    fn parse_chunk_skips_non_text_delta_types() {
        // 非 text_delta 的 delta（如 input_json_delta）应跳过
        let v = json!({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": "{}"}
        });
        assert!(AnthropicAdapter::parse_chunk(&v, "claude-3").is_none());
    }

    #[test]
    fn parse_chunk_skips_message_start_and_ping() {
        assert!(
            AnthropicAdapter::parse_chunk(&json!({"type":"message_start"}), "claude-3").is_none()
        );
        assert!(AnthropicAdapter::parse_chunk(&json!({"type":"ping"}), "claude-3").is_none());
        assert!(
            AnthropicAdapter::parse_chunk(&json!({"type":"content_block_start"}), "claude-3")
                .is_none()
        );
    }

    #[test]
    fn parse_models_uses_id_and_display_name() {
        let v = json!({
            "data": [
                {"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet"},
                {"id": "claude-3-haiku-20240307", "display_name": "Claude 3 Haiku"}
            ]
        });
        let models = AnthropicAdapter::parse_models(&v, "anthropic");
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "claude-3-5-sonnet-20241022");
        assert_eq!(models[0].name, "Claude 3.5 Sonnet");
        assert_eq!(models[0].provider, "anthropic");
        assert_eq!(models[1].id, "claude-3-haiku-20240307");
    }

    #[test]
    fn parse_models_empty_returns_vec() {
        assert!(AnthropicAdapter::parse_models(&json!({"data": []}), "anthropic").is_empty());
        assert!(AnthropicAdapter::parse_models(&json!({}), "anthropic").is_empty());
    }

    #[test]
    fn parse_anthropic_usage_maps_tokens() {
        let v = json!({"input_tokens": 7, "output_tokens": 3});
        let usage = parse_anthropic_usage(&v).expect("应返回 Some");
        assert_eq!(usage.prompt_tokens, 7);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 10);
    }

    #[test]
    fn parse_anthropic_usage_missing_input_returns_none() {
        assert!(parse_anthropic_usage(&json!({"output_tokens": 3})).is_none());
    }

    #[test]
    fn build_chat_body_includes_required_fields() {
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(50)
            .temperature(0.5)
            .build();
        let body = AnthropicAdapter::build_chat_body(&req, false);
        assert_eq!(body["model"], "claude-3-5-sonnet-20241022");
        assert_eq!(body["max_tokens"], 50);
        assert_eq!(body["temperature"], 0.5);
        assert!(body.get("messages").is_some());
        // 非 stream 模式不应有 stream 字段
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn build_chat_body_stream_adds_stream_true() {
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(50)
            .build();
        let body = AnthropicAdapter::build_chat_body(&req, true);
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn build_chat_body_single_stop_becomes_array() {
        let req = ChatRequest::builder("claude-3-5-sonnet-20241022", vec![ChatMessage::user("hi")])
            .max_tokens(50)
            .stop(StopSeq::Single("END".into()))
            .build();
        let body = AnthropicAdapter::build_chat_body(&req, false);
        assert_eq!(body["stop_sequences"], json!(["END"]));
    }
}
