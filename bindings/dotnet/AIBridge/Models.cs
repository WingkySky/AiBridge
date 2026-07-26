using System.Text.Json;
using System.Text.Json.Serialization;

namespace AIBridge;

// ============================================================================
// 数据模型层
//
// 对应 crates/aibridge-core/src/model/{chat,audio,image,video,options}.rs。
// ============================================================================

// ───────────────────────────────────────────────────────────────────────
// Chat 文本对话
// ───────────────────────────────────────────────────────────────────────

public sealed class ChatRequest
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("messages")]
    public List<ChatMessage> Messages { get; set; } = new();

    [JsonPropertyName("temperature")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Temperature { get; set; }

    [JsonPropertyName("max_tokens")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? MaxTokens { get; set; }

    [JsonPropertyName("n")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? N { get; set; }

    [JsonProperty("presence_penalty")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? PresencePenalty { get; set; }

    [JsonProperty("frequency_penalty")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? FrequencyPenalty { get; set; }

    [JsonPropertyName("seed")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ulong? Seed { get; set; }

    [JsonPropertyName("stream")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool Stream { get; set; }

    [JsonPropertyName("user")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? User { get; set; }

    [JsonPropertyName("response_format")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ResponseFormat? ResponseFormat { get; set; }

    [JsonPropertyName("stop")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public StopSeq? Stop { get; set; }

    [JsonPropertyName("tools")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<ToolDefinition>? Tools { get; set; }

    [JsonPropertyName("tool_choice")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ToolChoice? ToolChoice { get; set; }

    [JsonPropertyName("extra")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, JsonElement>? Extra { get; set; }

    public ChatRequest(string model, IEnumerable<ChatMessage> messages)
    {
        Model = model;
        Messages = messages.ToList();
    }

    public ChatRequest() { }
}

public sealed class ChatMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = "user";

    [JsonPropertyName("content")]
    public object Content { get; set; } = string.Empty; // string or list

    [JsonPropertyName("name")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Name { get; set; }

    [JsonPropertyName("tool_call_id")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ToolCallId { get; set; }

    public static ChatMessage User(string content) => new() { Role = "user", Content = content };
    public static ChatMessage System(string content) => new() { Role = "system", Content = content };
    public static ChatMessage Assistant(string content) => new() { Role = "assistant", Content = content };
}

public sealed class ChatCompletion
{
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    [JsonPropertyName("object")] public string Object { get; set; } = "chat.completion";
    [JsonPropertyName("created")] public ulong Created { get; set; }
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("choices")] public List<ChatChoice> Choices { get; set; } = new();
    [JsonPropertyName("usage")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ChatUsage? Usage { get; set; }
    [JsonPropertyName("service_tier")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ServiceTier { get; set; }
    [JsonPropertyName("system_fingerprint")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? SystemFingerprint { get; set; }
}

public sealed class ChatChoice
{
    [JsonPropertyName("index")] public uint Index { get; set; }
    [JsonPropertyName("message")] public ChoiceMessage Message { get; set; } = new();
    [JsonPropertyName("finish_reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? FinishReason { get; set; }
}

public sealed class ChoiceMessage
{
    [JsonPropertyName("role")] public string Role { get; set; } = "assistant";
    [JsonPropertyName("content")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Content { get; set; }
    [JsonPropertyName("tool_calls")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<ToolCall>? ToolCalls { get; set; }
}

public sealed class ChatUsage
{
    [JsonPropertyName("prompt_tokens")] public ulong PromptTokens { get; set; }
    [JsonPropertyName("completion_tokens")] public ulong CompletionTokens { get; set; }
    [JsonPropertyName("total_tokens")] public ulong TotalTokens { get; set; }
}

public sealed class ChatCompletionChunk
{
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    [JsonPropertyName("object")] public string Object { get; set; } = "chat.completion.chunk";
    [JsonPropertyName("created")] public ulong Created { get; set; }
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("choices")] public List<ChatCompletionDelta> Choices { get; set; } = new();
    [JsonPropertyName("usage")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ChatUsage? Usage { get; set; }
}

public sealed class ChatCompletionDelta
{
    [JsonPropertyName("index")] public uint Index { get; set; }
    [JsonPropertyName("delta")] public DeltaMessage Delta { get; set; } = new();
    [JsonPropertyName("finish_reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? FinishReason { get; set; }
}

public sealed class DeltaMessage
{
    [JsonPropertyName("role")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Role { get; set; }
    [JsonPropertyName("content")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Content { get; set; }
    [JsonPropertyName("tool_calls")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<ToolCall>? ToolCalls { get; set; }
}

// ───────────────────────────────────────────────────────────────────────
// Tool Call / Response Format / Stop Seq
// ───────────────────────────────────────────────────────────────────────

public sealed class ResponseFormat
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "text";

    public ResponseFormat(string type) { Type = type; }
}

public sealed class StopSeq
{
    // 单值或数组，由 JsonSerializer 自动判断
}

public sealed class ToolDefinition
{
    [JsonPropertyName("type")]
    public string ToolType { get; set; } = "function";

    [JsonPropertyName("function")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public FunctionDefinition? Function { get; set; }
}

public sealed class FunctionDefinition
{
    [JsonPropertyName("name")] public string Name { get; set; } = string.Empty;
    [JsonPropertyName("description")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Description { get; set; }
    [JsonPropertyName("parameters")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? Parameters { get; set; }
}

public sealed class ToolCall
{
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    [JsonPropertyName("type")] public string ToolType { get; set; } = "function";
    [JsonPropertyName("function")] public ToolCallFunction Function { get; set; } = new();
}

public sealed class ToolCallFunction
{
    [JsonPropertyName("name")] public string Name { get; set; } = string.Empty;
    [JsonPropertyName("arguments")] public string Arguments { get; set; } = string.Empty;
}

public sealed class ToolChoice
{
    public string Value { get; set; }

    private ToolChoice(string value) { Value = value; }
    public static ToolChoice None => new("none");
    public static ToolChoice Auto => new("auto");
    public static ToolChoice Required => new("required");
}

// ───────────────────────────────────────────────────────────────────────
// Speech 文字转语音
// ───────────────────────────────────────────────────────────────────────

public sealed class SpeechRequest
{
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("input")] public string Input { get; set; } = string.Empty;
    [JsonPropertyName("voice")] public VoiceSpec Voice { get; set; } = new();
    [JsonPropertyName("response_format")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ResponseFormat { get; set; }
    [JsonPropertyName("speed")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Speed { get; set; }
    [JsonPropertyName("volume")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Volume { get; set; }
    [JsonPropertyName("pitch")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Pitch { get; set; }
    [JsonPropertyName("emotion")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Emotion { get; set; }
    [JsonPropertyName("style")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Style { get; set; }
    [JsonPropertyName("extra")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, JsonElement>? Extra { get; set; }

    public SpeechRequest(string model, string input, string voice)
    {
        Model = model;
        Input = input;
        Voice = VoiceSpec.Single(voice);
    }

    public SpeechRequest() { }
}

public sealed class VoiceSpec
{
    [JsonPropertyName("voices")]
    public List<string> Voices { get; set; } = new();

    public static VoiceSpec Single(string voice) => new() { Voices = new List<string> { voice } };
}

public sealed class SpeechResult
{
    [JsonIgnore]
    public byte[] AudioData { get; set; } = Array.Empty<byte>();

    [JsonPropertyName("audio_url")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? AudioUrl { get; set; }

    [JsonPropertyName("audio_base64")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? AudioBase64 { get; set; }

    [JsonPropertyName("content_type")]
    public string ContentType { get; set; } = "audio/mpeg";

    [JsonPropertyName("format")]
    public string Format { get; set; } = "mp3";

    [JsonPropertyName("duration")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Duration { get; set; }

    [JsonPropertyName("model")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Model { get; set; }
}

// ───────────────────────────────────────────────────────────────────────
// Transcribe / Translate 语音转文字
// ───────────────────────────────────────────────────────────────────────

/// <summary>文件输入（路径/URL/字节/Base64，对应 Rust FileInput untagged enum）。</summary>
public sealed class FileInput
{
    // 内部用 JsonElement 兼容多种类型
    internal JsonElement Element { get; }

    private FileInput(JsonElement element) { Element = element; }

    public static FileInput Path(string path)
    {
        var doc = JsonDocument.Parse(JsonSerializer.Serialize(path));
        return new FileInput(doc.RootElement);
    }

    public static FileInput Url(string url)
    {
        var doc = JsonDocument.Parse(JsonSerializer.Serialize(url));
        return new FileInput(doc.RootElement);
    }
}

public sealed class TranscribeRequest
{
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("file")] public FileInput File { get; set; } = null!;
    [JsonPropertyName("language")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Language { get; set; }
    [JsonPropertyName("prompt")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Prompt { get; set; }
    [JsonPropertyName("response_format")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ResponseFormat { get; set; }
    [JsonPropertyName("temperature")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Temperature { get; set; }
    [JsonPropertyName("timestamp_granularities")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<string>? TimestampGranularities { get; set; }
    [JsonPropertyName("translate")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool Translate { get; set; }
    [JsonPropertyName("extra")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, JsonElement>? Extra { get; set; }

    public TranscribeRequest() { }
}

public sealed class TranscriptionResult
{
    [JsonPropertyName("text")] public string Text { get; set; } = string.Empty;
    [JsonPropertyName("language")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Language { get; set; }
    [JsonPropertyName("duration")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Duration { get; set; }
    [JsonPropertyName("segments")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<TranscriptionSegment>? Segments { get; set; }
    [JsonPropertyName("words")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<TranscriptionWord>? Words { get; set; }
    [JsonPropertyName("task")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Task { get; set; }
    [JsonPropertyName("usage")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? Usage { get; set; }
    [JsonPropertyName("model")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Model { get; set; }
}

public sealed class TranscriptionSegment
{
    [JsonPropertyName("id")] public int Id { get; set; }
    [JsonPropertyName("start")] public double Start { get; set; }
    [JsonPropertyName("end")] public double End { get; set; }
    [JsonPropertyName("text")] public string Text { get; set; } = string.Empty;
    [JsonPropertyName("confidence")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Confidence { get; set; }
    [JsonPropertyName("speaker")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Speaker { get; set; }
}

public sealed class TranscriptionWord
{
    [JsonPropertyName("word")] public string Word { get; set; } = string.Empty;
    [JsonPropertyName("start")] public double Start { get; set; }
    [JsonPropertyName("end")] public double End { get; set; }
    [JsonPropertyName("confidence")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Confidence { get; set; }
}

// ───────────────────────────────────────────────────────────────────────
// Image 图像生成
// ───────────────────────────────────────────────────────────────────────

public sealed class ImageRequest
{
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("prompt")] public string Prompt { get; set; } = string.Empty;
    [JsonPropertyName("size")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Size { get; set; }
    [JsonPropertyName("width")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? Width { get; set; }
    [JsonPropertyName("height")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? Height { get; set; }
    [JsonPropertyName("aspect_ratio")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? AspectRatio { get; set; }
    [JsonPropertyName("n")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? N { get; set; }
    [JsonPropertyName("quality")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Quality { get; set; }
    [JsonPropertyName("style")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Style { get; set; }
    [JsonPropertyName("negative_prompt")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? NegativePrompt { get; set; }
    [JsonPropertyName("negative_prompts")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<string>? NegativePrompts { get; set; }
    [JsonPropertyName("seed")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ulong? Seed { get; set; }
    [JsonPropertyName("steps")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? Steps { get; set; }
    [JsonPropertyName("cfg_scale")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? CfgScale { get; set; }
    [JsonPropertyName("sampler")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Sampler { get; set; }
    [JsonPropertyName("scheduler")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Scheduler { get; set; }
    [JsonPropertyName("response_format")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ResponseFormat { get; set; }
    [JsonPropertyName("output_format")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? OutputFormat { get; set; }
    [JsonPropertyName("reference_images")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<object>? ReferenceImages { get; set; }
    [JsonPropertyName("reference_strength")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? ReferenceStrength { get; set; }
    [JsonPropertyName("mask")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public object? Mask { get; set; }
    [JsonPropertyName("edit_mode")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? EditMode { get; set; }
    [JsonPropertyName("extra")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, JsonElement>? Extra { get; set; }

    public ImageRequest(string model, string prompt) { Model = model; Prompt = prompt; }
    public ImageRequest() { }
}

public sealed class ImageData
{
    [JsonPropertyName("url")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Url { get; set; }
    [JsonPropertyName("b64_json")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? B64Json { get; set; }
    [JsonPropertyName("revised_prompt")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? RevisedPrompt { get; set; }
}

public sealed class ImageResult
{
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    [JsonPropertyName("object")] public string Object { get; set; } = "image.generation";
    [JsonPropertyName("created")] public ulong Created { get; set; }
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("data")] public List<ImageData> Data { get; set; } = new();
}

// ───────────────────────────────────────────────────────────────────────
// Video 视频生成
// ───────────────────────────────────────────────────────────────────────

public sealed class VideoRequest
{
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("prompt")] public string Prompt { get; set; } = string.Empty;
    [JsonPropertyName("width")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? Width { get; set; }
    [JsonPropertyName("height")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? Height { get; set; }
    [JsonPropertyName("num_frames")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? NumFrames { get; set; }
    [JsonPropertyName("frame_rate")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? FrameRate { get; set; }
    [JsonPropertyName("mode")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Mode { get; set; }
    [JsonPropertyName("duration")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? Duration { get; set; }
    [JsonPropertyName("aspect_ratio")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? AspectRatio { get; set; }
    [JsonPropertyName("resolution")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Resolution { get; set; }
    [JsonPropertyName("reference_images")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<object>? ReferenceImages { get; set; }
    [JsonPropertyName("reference_videos")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<object>? ReferenceVideos { get; set; }
    [JsonPropertyName("first_frame")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public object? FirstFrame { get; set; }
    [JsonPropertyName("last_frame")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public object? LastFrame { get; set; }
    [JsonPropertyName("keyframes")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<Dictionary<string, JsonElement>>? Keyframes { get; set; }
    [JsonPropertyName("style")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Style { get; set; }
    [JsonPropertyName("camera_motion")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? CameraMotion { get; set; }
    [JsonPropertyName("motion_strength")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? MotionStrength { get; set; }
    [JsonPropertyName("negative_prompt")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? NegativePrompt { get; set; }
    [JsonPropertyName("seed")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ulong? Seed { get; set; }
    [JsonPropertyName("steps")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? Steps { get; set; }
    [JsonPropertyName("cfg_scale")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? CfgScale { get; set; }
    [JsonPropertyName("with_audio")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public bool? WithAudio { get; set; }
    [JsonPropertyName("watermark")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public bool? Watermark { get; set; }
    [JsonPropertyName("extra")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, JsonElement>? Extra { get; set; }

    public VideoRequest(string model, string prompt) { Model = model; Prompt = prompt; }
    public VideoRequest() { }
}

public sealed class VideoTask
{
    [JsonPropertyName("task_id")] public string TaskId { get; set; } = string.Empty;
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("status")] public string Status { get; set; } = string.Empty;
    [JsonPropertyName("created_at")] public ulong CreatedAt { get; set; }
}

public sealed class VideoStatus
{
    [JsonPropertyName("task_id")] public string TaskId { get; set; } = string.Empty;
    [JsonPropertyName("status")] public string Status { get; set; } = string.Empty;
    [JsonPropertyName("video_url")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? VideoUrl { get; set; }
    [JsonPropertyName("progress")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? Progress { get; set; }
    [JsonPropertyName("error")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Error { get; set; }
    [JsonPropertyName("created_at")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ulong? CreatedAt { get; set; }
    [JsonPropertyName("updated_at")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ulong? UpdatedAt { get; set; }
}

// ───────────────────────────────────────────────────────────────────────
// Embed 文本嵌入
// ───────────────────────────────────────────────────────────────────────

public sealed class EmbedRequest
{
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    /// <summary>单个字符串或字符串列表</summary>
    [JsonPropertyName("input")]
    public object Input { get; set; } = string.Empty;
    [JsonPropertyName("dimensions")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? Dimensions { get; set; }
    [JsonPropertyName("encoding_format")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? EncodingFormat { get; set; }
    [JsonPropertyName("user")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? User { get; set; }
    [JsonPropertyName("extra")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, JsonElement>? Extra { get; set; }

    public EmbedRequest() { }
}

public sealed class EmbeddingResult
{
    [JsonPropertyName("object")] public string Object { get; set; } = "list";
    [JsonPropertyName("data")] public List<EmbeddingItem> Data { get; set; } = new();
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("usage")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public EmbeddingUsage? Usage { get; set; }
}

public sealed class EmbeddingItem
{
    [JsonPropertyName("object")] public string Object { get; set; } = "embedding";
    [JsonPropertyName("index")] public uint Index { get; set; }
    [JsonPropertyName("embedding")] public EmbeddingVector Embedding { get; set; } = new();
}

public sealed class EmbeddingVector
{
    // 浮点列表或 base64 字符串
    public List<double>? Floats { get; set; }
    public string? Base64 { get; set; }
}

public sealed class EmbeddingUsage
{
    [JsonPropertyName("prompt_tokens")] public ulong PromptTokens { get; set; }
    [JsonPropertyName("total_tokens")] public ulong TotalTokens { get; set; }
}

// ───────────────────────────────────────────────────────────────────────
// Common 通用类型
// ───────────────────────────────────────────────────────────────────────

public sealed class ModelInfo
{
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    [JsonPropertyName("name")] public string Name { get; set; } = string.Empty;
    [JsonPropertyName("description")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Description { get; set; }
    [JsonPropertyName("provider")] public string Provider { get; set; } = string.Empty;
    [JsonPropertyName("type")] public string Type { get; set; } = string.Empty;
    [JsonPropertyName("capabilities")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<string>? Capabilities { get; set; }
}

public sealed class ProviderInfo
{
    [JsonPropertyName("name")] public string Name { get; set; } = string.Empty;
    [JsonPropertyName("description")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Description { get; set; }
    [JsonPropertyName("models")] public List<ModelInfo> Models { get; set; } = new();
}

public sealed class VoiceInfo
{
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    [JsonPropertyName("name")] public string Name { get; set; } = string.Empty;
    [JsonPropertyName("gender")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Gender { get; set; }
    [JsonPropertyName("languages")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<string>? Languages { get; set; }
    [JsonPropertyName("sample_url")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? SampleUrl { get; set; }
    [JsonPropertyName("description")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Description { get; set; }
}
