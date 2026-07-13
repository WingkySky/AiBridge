using System.Text.Json;
using System.Text.Json.Serialization;

namespace AIBridge;

// ============================================================================
// 数据模型层
//
// 对应 crates/aibridge-core/src/model/{chat,audio}.rs 的 serde struct。
// 用 System.Text.Json 反序列化 FFI 返回的 JSON。字段名与 Rust serde 输出对齐
// （Rust 默认 snake_case，C# 这里显式 JsonPropertyName 对齐）。
//
// 只覆盖 hello world 涉及的能力：chat / chat_stream / speech。
// 其余能力（image/video/embed/transcribe/list_models/list_voices）留待后续阶段。
// ============================================================================

/// <summary>对话请求（对应 core ChatRequest）。</summary>
public sealed class ChatRequest
{
    /// <summary>模型名称（如 "echo-chat"、"gpt-4o"）。</summary>
    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    /// <summary>消息列表。</summary>
    [JsonPropertyName("messages")]
    public List<ChatMessage> Messages { get; set; } = new();

    /// <summary>温度系数（可选）。</summary>
    [JsonPropertyName("temperature")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Temperature { get; set; }

    /// <summary>最大生成 token 数（可选）。</summary>
    [JsonPropertyName("max_tokens")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public uint? MaxTokens { get; set; }

    /// <summary>是否流式（chat_stream 不依赖此字段，由调用方法决定）。</summary>
    [JsonPropertyName("stream")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool Stream { get; set; }

    /// <summary>构造请求。</summary>
    public ChatRequest(string model, IEnumerable<ChatMessage> messages)
    {
        Model = model;
        Messages = messages.ToList();
    }

    /// <summary>供 JsonSerializer 用，外部应使用带参构造。</summary>
    public ChatRequest() { }
}

/// <summary>
/// 对话消息（对应 core ChatMessage 的 tagged enum）。
///
/// Rust 用 #[serde(tag="role", rename_all="lowercase")]：
/// 序列化为 {"role":"user","content":"..."} 等。C# 这里用扁平结构 + role 字段，
/// 仅覆盖 system/user 纯文本场景（hello world 足够），其余变体后续扩展。
/// </summary>
public sealed class ChatMessage
{
    /// <summary>角色：system / user / assistant / tool。</summary>
    [JsonPropertyName("role")]
    public string Role { get; set; } = "user";

    /// <summary>消息内容（纯文本场景；多模态见 core UserContent::Parts，暂不支持）。</summary>
    [JsonPropertyName("content")]
    public string Content { get; set; } = string.Empty;

    /// <summary>创建 user 消息。</summary>
    public static ChatMessage User(string content) => new()
    {
        Role = "user",
        Content = content,
    };

    /// <summary>创建 system 消息。</summary>
    public static ChatMessage System(string content) => new()
    {
        Role = "system",
        Content = content,
    };

    /// <summary>创建 assistant 消息。</summary>
    public static ChatMessage Assistant(string content) => new()
    {
        Role = "assistant",
        Content = content,
    };
}

/// <summary>对话完成结果（对应 core ChatCompletion）。</summary>
public sealed class ChatCompletion
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("object")]
    public string Object { get; set; } = "chat.completion";

    [JsonPropertyName("created")]
    public ulong Created { get; set; }

    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("choices")]
    public List<ChatChoice> Choices { get; set; } = new();

    [JsonPropertyName("usage")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ChatUsage? Usage { get; set; }
}

/// <summary>对话选项（对应 core ChatChoice）。</summary>
public sealed class ChatChoice
{
    [JsonPropertyName("index")]
    public uint Index { get; set; }

    [JsonPropertyName("message")]
    public ChoiceMessage Message { get; set; } = new();

    [JsonPropertyName("finish_reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? FinishReason { get; set; }
}

/// <summary>完成结果中的消息（对应 core ChoiceMessage）。</summary>
public sealed class ChoiceMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = "assistant";

    [JsonPropertyName("content")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Content { get; set; }
}

/// <summary>Token 使用统计（对应 core ChatUsage）。</summary>
public sealed class ChatUsage
{
    [JsonPropertyName("prompt_tokens")]
    public ulong PromptTokens { get; set; }

    [JsonPropertyName("completion_tokens")]
    public ulong CompletionTokens { get; set; }

    [JsonPropertyName("total_tokens")]
    public ulong TotalTokens { get; set; }
}

/// <summary>流式增量块（对应 core ChatCompletionChunk）。</summary>
public sealed class ChatCompletionChunk
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("object")]
    public string Object { get; set; } = "chat.completion.chunk";

    [JsonPropertyName("created")]
    public ulong Created { get; set; }

    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("choices")]
    public List<ChatCompletionDelta> Choices { get; set; } = new();

    [JsonPropertyName("usage")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ChatUsage? Usage { get; set; }
}

/// <summary>流式增量（对应 core ChatCompletionDelta）。</summary>
public sealed class ChatCompletionDelta
{
    [JsonPropertyName("index")]
    public uint Index { get; set; }

    [JsonPropertyName("delta")]
    public DeltaMessage Delta { get; set; } = new();

    [JsonPropertyName("finish_reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? FinishReason { get; set; }
}

/// <summary>流式增量消息（对应 core DeltaMessage）。</summary>
public sealed class DeltaMessage
{
    [JsonPropertyName("role")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Role { get; set; }

    [JsonPropertyName("content")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Content { get; set; }
}

/// <summary>文字转语音请求（对应 core SpeechRequest）。</summary>
public sealed class SpeechRequest
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("input")]
    public string Input { get; set; } = string.Empty;

    /// <summary>音色规格（core VoiceSpec，对象含 voices 数组）。</summary>
    [JsonPropertyName("voice")]
    public VoiceSpec Voice { get; set; } = new();

    /// <summary>
    /// 音频输出格式（"mp3" / "opus" / "aac" / "flac" / "wav" / "pcm"）。
    /// 可空：为 null 时 Rust 端用默认 "mp3"（skip_serializing_if 语义）。
    /// </summary>
    [JsonPropertyName("response_format")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ResponseFormat { get; set; }

    /// <summary>构造请求。</summary>
    public SpeechRequest(string model, string input, string voice)
    {
        Model = model;
        Input = input;
        Voice = VoiceSpec.Single(voice);
    }

    /// <summary>供 JsonSerializer 用。</summary>
    public SpeechRequest() { }
}

/// <summary>音色规格（对应 core VoiceSpec）。</summary>
public sealed class VoiceSpec
{
    [JsonPropertyName("voices")]
    public List<string> Voices { get; set; } = new();

    /// <summary>单个音色。</summary>
    public static VoiceSpec Single(string voice) => new() { Voices = new List<string> { voice } };
}

/// <summary>
/// 文字转语音结果元数据（对应 core SpeechResult，audio_data 不序列化）。
/// 二进制音频通过 FFI 的 aibridge_bytes_t 单独返回，见 SpeechResult.AudioData。
/// </summary>
public sealed class SpeechResult
{
    /// <summary>音频二进制数据（来自 FFI aibridge_bytes_t，非 JSON 字段）。</summary>
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
