// Package aibridge - 数据模型
//
// 本文件定义 AIBridge Go 绑定的数据模型（请求/响应 struct），
// 与 Rust 端 aibridge-core 的 serde struct 字段一一对应（JSON 边界）。
// 对应：crates/aibridge-core/src/model/{chat,audio}.rs
package aibridge

import "encoding/json"

// ChatMessage 表示一条对话消息（与 Rust ChatMessage 对齐，role 作为 tag）
//
// JSON 序列化形如：{"role":"user","content":"hello"}
type ChatMessage struct {
	Role       string          `json:"role"`                 // 角色：system / user / assistant / tool
	Content    json.RawMessage `json:"content"`              // 内容：字符串或多模态部件列表（用 RawMessage 兼容两种形态）
	Name       string          `json:"name,omitempty"`       // 发送者名称（可选）
	ToolCallID string          `json:"tool_call_id,omitempty"` // 工具调用 ID（仅 role=tool）
}

// NewUserTextMessage 构造一条纯文本用户消息
func NewUserTextMessage(content string) ChatMessage {
	// content 序列化为 JSON 字符串（带引号），与 Rust UserContent::Text 对齐
	b, _ := json.Marshal(content)
	return ChatMessage{
		Role:    "user",
		Content: b,
	}
}

// NewSystemMessage 构造一条系统消息
func NewSystemMessage(content string) ChatMessage {
	return ChatMessage{
		Role:    "system",
		Content: json.RawMessage(`"` + jsonEscapeString(content) + `"`),
	}
}

// jsonEscapeString 对字符串做 JSON 转义（用于手工拼装带引号的 content）
func jsonEscapeString(s string) string {
	b, _ := json.Marshal(s)
	// 去掉首尾引号
	return string(b[1 : len(b)-1])
}

// ChatRequest 文本对话请求（对应 Rust ChatRequest）
type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`

	Temperature      float64 `json:"temperature,omitempty"`
	TopP             float64 `json:"top_p,omitempty"`
	MaxTokens        uint32  `json:"max_tokens,omitempty"`
	N                uint32  `json:"n,omitempty"`
	PresencePenalty  float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64 `json:"frequency_penalty,omitempty"`
	Seed             uint64  `json:"seed,omitempty"`
	Stream           bool    `json:"stream,omitempty"`
	User             string  `json:"user,omitempty"`

	// 厂商特有参数透传
	Extra map[string]json.RawMessage `json:"extra,omitempty"`
}

// ChatCompletion 文本对话完成结果（对应 Rust ChatCompletion）
type ChatCompletion struct {
	ID                string       `json:"id"`
	Object            string       `json:"object"`
	Created           uint64       `json:"created"`
	Model             string       `json:"model"`
	Choices           []ChatChoice `json:"choices"`
	Usage             *ChatUsage   `json:"usage,omitempty"`
	ServiceTier       string       `json:"service_tier,omitempty"`
	SystemFingerprint string       `json:"system_fingerprint,omitempty"`
}

// ChatChoice 对话选项
type ChatChoice struct {
	Index        int            `json:"index"`
	Message      ChoiceMessage  `json:"message"`
	FinishReason string         `json:"finish_reason,omitempty"`
}

// ChoiceMessage 完成结果中的消息
type ChoiceMessage struct {
	Role      string `json:"role"`
	Content   string `json:"content,omitempty"`
}

// ChatUsage Token 使用统计
type ChatUsage struct {
	PromptTokens     uint64 `json:"prompt_tokens"`
	CompletionTokens uint64 `json:"completion_tokens"`
	TotalTokens      uint64 `json:"total_tokens"`
}

// ChatCompletionChunk 流式对话块（对应 Rust ChatCompletionChunk）
type ChatCompletionChunk struct {
	ID      string                  `json:"id"`
	Object  string                  `json:"object"`
	Created uint64                  `json:"created"`
	Model   string                  `json:"model"`
	Choices []ChatCompletionDelta   `json:"choices"`
	Usage   *ChatUsage              `json:"usage,omitempty"`
}

// ChatCompletionDelta 流式增量
type ChatCompletionDelta struct {
	Index        int          `json:"index"`
	Delta        DeltaMessage `json:"delta"`
	FinishReason string       `json:"finish_reason,omitempty"`
}

// DeltaMessage 流式增量消息
type DeltaMessage struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// SpeechRequest 文字转语音请求（对应 Rust SpeechRequest）
//
// 注意：Voice 字段是 VoiceSpec 对象 {"voices":[...]}，不是纯字符串。
type SpeechRequest struct {
	Model          string    `json:"model"`
	Input          string    `json:"input"`
	Voice          VoiceSpec `json:"voice"`
	ResponseFormat string    `json:"response_format,omitempty"`
	Speed          float64   `json:"speed,omitempty"`
	Volume         float64   `json:"volume,omitempty"`
	Pitch          float64   `json:"pitch,omitempty"`
	Emotion        string    `json:"emotion,omitempty"`
	Style          string    `json:"style,omitempty"`

	Extra map[string]json.RawMessage `json:"extra,omitempty"`
}

// VoiceSpec 音色规格（支持候选列表用于自动降级）
type VoiceSpec struct {
	Voices []string `json:"voices"`
}

// SingleVoice 构造单个音色的 VoiceSpec
func SingleVoice(v string) VoiceSpec {
	return VoiceSpec{Voices: []string{v}}
}

// SpeechResult 文字转语音结果（对应 Rust SpeechResult，audio_data 不参与序列化）
//
// 二进制音频数据通过 FFI 的 aibridge_bytes_t 单独传递，本结构体由 Speech() 填充。
type SpeechResult struct {
	AudioData   []byte  // 二进制音频数据（FFI 单独传递，不来自 JSON）
	AudioURL    string  `json:"audio_url,omitempty"`
	AudioBase64 string  `json:"audio_base64,omitempty"`
	ContentType string  `json:"content_type"`
	Format      string  `json:"format"`
	Duration    float64 `json:"duration,omitempty"`
	Model       string  `json:"model,omitempty"`
}
