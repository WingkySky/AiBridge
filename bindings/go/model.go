// Package aibridge - 数据模型
//
// 本文件定义 AIBridge Go 绑定的数据模型（请求/响应 struct），
// 与 Rust 端 aibridge-core 的 serde struct 字段一一对应（JSON 边界）。
// 对应：crates/aibridge-core/src/model/{chat,audio,image,video,options}.rs
package aibridge

import "encoding/json"

// ───────────────────────────────────────────────────────────────────────
// Chat 文本对话
// ───────────────────────────────────────────────────────────────────────

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

	Temperature      float64         `json:"temperature,omitempty"`
	TopP             float64         `json:"top_p,omitempty"`
	MaxTokens        uint32          `json:"max_tokens,omitempty"`
	N                uint32          `json:"n,omitempty"`
	PresencePenalty  float64         `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64         `json:"frequency_penalty,omitempty"`
	Seed             uint64          `json:"seed,omitempty"`
	Stream           bool            `json:"stream,omitempty"`
	User             string          `json:"user,omitempty"`
	ResponseFormat   *ResponseFormat `json:"response_format,omitempty"`
	Stop             *StopSeq        `json:"stop,omitempty"`
	Tools            []ToolDefinition `json:"tools,omitempty"`
	ToolChoice       *ToolChoice     `json:"tool_choice,omitempty"`

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
	Role      string     `json:"role"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
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
	Role    string     `json:"role,omitempty"`
	Content string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ───────────────────────────────────────────────────────────────────────
// Tool Call / Response Format / Stop Seq
// ───────────────────────────────────────────────────────────────────────

// ResponseFormat 响应格式
type ResponseFormat struct {
	Type string `json:"type"`
}

// StopSeq 停止词（单个或多个）
type StopSeq struct {
	Single  string
	Multiple []string
}

// MarshalJSON 实现自定义序列化（对应 Rust untagged enum）
func (s StopSeq) MarshalJSON() ([]byte, error) {
	if s.Single != "" {
		return json.Marshal(s.Single)
	}
	return json.Marshal(s.Multiple)
}

// UnmarshalJSON 实现自定义反序列化
func (s *StopSeq) UnmarshalJSON(data []byte) error {
	var single string
	if err := json.Unmarshal(data, &single); err == nil {
		s.Single = single
		return nil
	}
	var multiple []string
	if err := json.Unmarshal(data, &multiple); err == nil {
		s.Multiple = multiple
		return nil
	}
	return json.Unmarshal(data, &s.Single)
}

// ToolChoice 工具选择策略（对应 Rust ToolChoice）
type ToolChoice struct {
	Value string `json:"-"`
}

func ToolChoiceNone() *ToolChoice     { return &ToolChoice{Value: "none"} }
func ToolChoiceAuto() *ToolChoice     { return &ToolChoice{Value: "auto"} }
func ToolChoiceRequired() *ToolChoice { return &ToolChoice{Value: "required"} }

// MarshalJSON 实现自定义序列化
func (tc *ToolChoice) MarshalJSON() ([]byte, error) {
	return json.Marshal(tc.Value)
}

// UnmarshalJSON 实现自定义反序列化
func (tc *ToolChoice) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, &tc.Value)
}

// ToolDefinition 工具定义（对应 Rust ToolDefinition）
type ToolDefinition struct {
	ToolType string           `json:"type"`
	Function *FunctionDefinition `json:"function,omitempty"`
}

// FunctionDefinition 函数定义（对应 Rust FunctionDefinition）
type FunctionDefinition struct {
	Name        string                  `json:"name"`
	Description string                  `json:"description,omitempty"`
	Parameters  json.RawMessage         `json:"parameters,omitempty"`
}

// ToolCall 工具调用（模型生成）
type ToolCall struct {
	ID       string            `json:"id"`
	ToolType string            `json:"type"`
	Function ToolCallFunction  `json:"function"`
}

// ToolCallFunction 工具调用的函数部分
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ───────────────────────────────────────────────────────────────────────
// Speech 文字转语音
// ───────────────────────────────────────────────────────────────────────

// SpeechRequest 文字转语音请求（对应 Rust SpeechRequest）
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
type SpeechResult struct {
	AudioData   []byte  // 二进制音频数据（FFI 单独传递，不来自 JSON）
	AudioURL    string  `json:"audio_url,omitempty"`
	AudioBase64 string  `json:"audio_base64,omitempty"`
	ContentType string  `json:"content_type"`
	Format      string  `json:"format"`
	Duration    float64 `json:"duration,omitempty"`
	Model       string  `json:"model,omitempty"`
}

// ───────────────────────────────────────────────────────────────────────
// Transcribe / Translate 语音转文字
// ───────────────────────────────────────────────────────────────────────

// FileInput 文件输入（路径/URL/字节/Base64，对应 Rust FileInput untagged enum）
type FileInput struct {
	// 内部用 json.RawMessage 兼容多种类型（字符串路径/URL 或字节数组）
	raw json.RawMessage
}

// FileInputPath 从路径创建 FileInput
func FileInputPath(p string) FileInput {
	fi, _ := json.Marshal(p)
	return FileInput{raw: fi}
}

// FileInputURL 从 URL 创建 FileInput
func FileInputURL(u string) FileInput {
	fi, _ := json.Marshal(u)
	return FileInput{raw: fi}
}

// FileInputBytes 从字节创建 FileInput
func FileInputBytes(b []byte) FileInput {
	fi, _ := json.Marshal(b)
	return FileInput{raw: fi}
}

// FileInputBase64 从 Base64 创建 FileInput
func FileInputBase64(s string) FileInput {
	fi, _ := json.Marshal(s)
	return FileInput{raw: fi}
}

// MarshalJSON 实现自定义序列化
func (fi FileInput) MarshalJSON() ([]byte, error) {
	return fi.raw, nil
}

// TranscribeRequest 语音转文字请求（对应 Rust TranscribeRequest）
type TranscribeRequest struct {
	Model                   string                  `json:"model"`
	File                    FileInput               `json:"file"`
	Language                string                  `json:"language,omitempty"`
	Prompt                  string                  `json:"prompt,omitempty"`
	ResponseFormat          string                  `json:"response_format,omitempty"`
	Temperature             float64                 `json:"temperature,omitempty"`
	TimestampGranularities  []string                `json:"timestamp_granularities,omitempty"`
	Translate               bool                    `json:"translate,omitempty"`

	Extra map[string]json.RawMessage `json:"extra,omitempty"`
}

// TranscriptionResult 转写结果（对应 Rust TranscriptionResult）
type TranscriptionResult struct {
	Text      string                      `json:"text"`
	Language  string                      `json:"language,omitempty"`
	Duration  float64                     `json:"duration,omitempty"`
	Segments  []TranscriptionSegment      `json:"segments,omitempty"`
	Words     []TranscriptionWord         `json:"words,omitempty"`
	Task      string                      `json:"task,omitempty"`
	Usage     json.RawMessage             `json:"usage,omitempty"`
	Model     string                      `json:"model,omitempty"`
}

// TranscriptionSegment 转写分段信息（对应 Rust TranscriptionSegment）
type TranscriptionSegment struct {
	ID         int     `json:"id"`
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
	Text       string  `json:"text"`
	Confidence *float64 `json:"confidence,omitempty"`
	Speaker    string  `json:"speaker,omitempty"`
}

// TranscriptionWord 转写词级时间戳（对应 Rust TranscriptionWord）
type TranscriptionWord struct {
	Word       string  `json:"word"`
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
	Confidence *float64 `json:"confidence,omitempty"`
}

// ───────────────────────────────────────────────────────────────────────
// Image 图像生成
// ───────────────────────────────────────────────────────────────────────

// ImageRequest 图像生成请求（对应 Rust ImageRequest）
type ImageRequest struct {
	Model           string        `json:"model"`
	Prompt          string        `json:"prompt"`
	Size            string        `json:"size,omitempty"`
	Width           uint32        `json:"width,omitempty"`
	Height          uint32        `json:"height,omitempty"`
	AspectRatio     string        `json:"aspect_ratio,omitempty"`
	N               uint32        `json:"n,omitempty"`
	Quality         string        `json:"quality,omitempty"`
	Style           string        `json:"style,omitempty"`
	NegativePrompt  string        `json:"negative_prompt,omitempty"`
	NegativePrompts []string      `json:"negative_prompts,omitempty"`
	Seed            uint64        `json:"seed,omitempty"`
	Steps           uint32        `json:"steps,omitempty"`
	CfgScale        float64       `json:"cfg_scale,omitempty"`
	Sampler         string        `json:"sampler,omitempty"`
	Scheduler       string        `json:"scheduler,omitempty"`
	ResponseFormat  string        `json:"response_format,omitempty"`
	OutputFormat    string        `json:"output_format,omitempty"`
	ReferenceImages []FileInput   `json:"reference_images,omitempty"`
	ReferenceStrength *float64     `json:"reference_strength,omitempty"`
	Mask            FileInput     `json:"mask,omitempty"`
	EditMode        string        `json:"edit_mode,omitempty"`

	Extra map[string]json.RawMessage `json:"extra,omitempty"`
}

// ImageData 图像数据（对应 Rust ImageData）
type ImageData struct {
	URL         string `json:"url,omitempty"`
	B64JSON     string `json:"b64_json,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

// ImageResult 图像生成结果（对应 Rust ImageResult）
type ImageResult struct {
	ID      string     `json:"id"`
	Object  string     `json:"object"`
	Created uint64     `json:"created"`
	Model   string     `json:"model"`
	Data    []ImageData `json:"data"`
}

// ───────────────────────────────────────────────────────────────────────
// Video 视频生成
// ───────────────────────────────────────────────────────────────────────

// VideoMode 视频生成模式
type VideoMode string

const (
	Text2Video  VideoMode = "text2video"
	Image2Video VideoMode = "image2video"
	Video2Video VideoMode = "video2video"
)

// VideoRequest 视频生成请求（对应 Rust VideoRequest）
type VideoRequest struct {
	Model             string             `json:"model"`
	Prompt            string             `json:"prompt"`
	Width             uint32             `json:"width,omitempty"`
	Height            uint32             `json:"height,omitempty"`
	NumFrames         uint32             `json:"num_frames,omitempty"`
	FrameRate         uint32             `json:"frame_rate,omitempty"`
	Mode              VideoMode          `json:"mode,omitempty"`
	Duration          uint32             `json:"duration,omitempty"`
	AspectRatio       string             `json:"aspect_ratio,omitempty"`
	Resolution        string             `json:"resolution,omitempty"`
	ReferenceImages   []FileInput        `json:"reference_images,omitempty"`
	ReferenceVideos   []FileInput        `json:"reference_videos,omitempty"`
	FirstFrame        FileInput          `json:"first_frame,omitempty"`
	LastFrame         FileInput          `json:"last_frame,omitempty"`
	Keyframes         []map[string]interface{} `json:"keyframes,omitempty"`
	Style             string             `json:"style,omitempty"`
	CameraMotion      string             `json:"camera_motion,omitempty"`
	MotionStrength    float64            `json:"motion_strength,omitempty"`
	NegativePrompt    string             `json:"negative_prompt,omitempty"`
	Seed              uint64             `json:"seed,omitempty"`
	Steps             uint32             `json:"steps,omitempty"`
	CfgScale          float64            `json:"cfg_scale,omitempty"`
	WithAudio         bool               `json:"with_audio,omitempty"`
	Watermark         bool               `json:"watermark,omitempty"`

	Extra map[string]json.RawMessage `json:"extra,omitempty"`
}

// VideoTask 视频任务信息（对应 Rust VideoTask）
type VideoTask struct {
	TaskID    string    `json:"task_id"`
	Model     string    `json:"model"`
	Status    TaskStatus `json:"status"`
	CreatedAt uint64    `json:"created_at"`
}

// VideoStatus 视频任务状态（对应 Rust VideoStatus）
type VideoStatus struct {
	TaskID    string     `json:"task_id"`
	Status    TaskStatus `json:"status"`
	VideoURL  string     `json:"video_url,omitempty"`
	Progress  uint32     `json:"progress,omitempty"`
	Error     string     `json:"error,omitempty"`
	CreatedAt uint64     `json:"created_at,omitempty"`
	UpdatedAt uint64     `json:"updated_at,omitempty"`
}

// ───────────────────────────────────────────────────────────────────────
// Embed 文本嵌入
// ───────────────────────────────────────────────────────────────────────

// EmbedRequest 文本嵌入请求（对应 Rust EmbedRequest）
type EmbedRequest struct {
	Model        string      `json:"model"`
	Input        EmbedInput  `json:"input"`
	Dimensions   uint32      `json:"dimensions,omitempty"`
	EncodingFormat string    `json:"encoding_format,omitempty"`
	User         string      `json:"user,omitempty"`

	Extra map[string]json.RawMessage `json:"extra,omitempty"`
}

// EmbedInput 嵌入输入（单个字符串或字符串列表，对应 Rust EmbedInput untagged enum）
type EmbedInput interface{}

// EmbeddingResult 嵌入结果（对应 Rust EmbeddingResult）
type EmbeddingResult struct {
	Object  string          `json:"object"`
	Data    []EmbeddingItem `json:"data"`
	Model   string          `json:"model"`
	Usage   *EmbeddingUsage `json:"usage,omitempty"`
}

// EmbeddingItem 单个嵌入项（对应 Rust EmbeddingItem）
type EmbeddingItem struct {
	Object    string    `json:"object"`
	Index     uint32    `json:"index"`
	Embedding EmbeddingVector `json:"embedding"`
}

// EmbeddingVector 嵌入向量（浮点列表或 base64 字符串）
type EmbeddingVector struct {
	Floats  []float64
	Base64  string
}

// MarshalJSON 实现自定义序列化
func (v EmbeddingVector) MarshalJSON() ([]byte, error) {
	if v.Floats != nil {
		return json.Marshal(v.Floats)
	}
	return json.Marshal(v.Base64)
}

// UnmarshalJSON 实现自定义反序列化
func (v *EmbeddingVector) UnmarshalJSON(data []byte) error {
	var floats []float64
	if err := json.Unmarshal(data, &floats); err == nil {
		v.Floats = floats
		return nil
	}
	var base64 string
	if err := json.Unmarshal(data, &base64); err == nil {
		v.Base64 = base64
		return nil
	}
	return json.Unmarshal(data, &v.Floats)
}

// EmbeddingUsage 嵌入使用统计（对应 Rust EmbeddingUsage）
type EmbeddingUsage struct {
	PromptTokens uint64 `json:"prompt_tokens"`
	TotalTokens  uint64 `json:"total_tokens"`
}

// ───────────────────────────────────────────────────────────────────────
// Common 通用类型
// ───────────────────────────────────────────────────────────────────────

// ModelType 模型类型过滤器
type ModelType string

const (
	ModelTypeChat   ModelType = "chat"
	ModelTypeImage  ModelType = "image"
	ModelTypeVideo  ModelType = "video"
	ModelTypeAudio  ModelType = "audio"
)

// ModelInfo 模型信息（对应 Rust ModelInfo）
type ModelInfo struct {
	ID          string     `json:"id"`
	Name        string     `json:"name"`
	Description string     `json:"description,omitempty"`
	Provider    string     `json:"provider"`
	Type        ModelType  `json:"type"`
	Capabilities []string  `json:"capabilities,omitempty"`
}

// ProviderInfo 供应商信息（对应 Rust ProviderInfo）
type ProviderInfo struct {
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
	Models      []ModelInfo `json:"models"`
}

// TaskStatus 任务状态
type TaskStatus string

const (
	TaskPending   TaskStatus = "pending"
	TaskProcessing TaskStatus = "processing"
	TaskSuccess   TaskStatus = "success"
	TaskFailed    TaskStatus = "failed"
)

// VoiceInfo 音色信息（对应 Rust VoiceInfo）
type VoiceInfo struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Gender      string   `json:"gender,omitempty"`
	Languages   []string `json:"languages,omitempty"`
	SampleURL   string   `json:"sample_url,omitempty"`
	Description string   `json:"description,omitempty"`
}
