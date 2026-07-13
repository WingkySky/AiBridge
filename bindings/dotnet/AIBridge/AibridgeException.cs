using System.Text.Json;

namespace AIBridge;

// ============================================================================
// 异常体系
//
// 对应设计文档第 9 节 .NET 异常映射：AibridgeException + 子类。
// 子类与 core AibridgeError 枚举变体一一对应（rate_limit_error/authentication_error/...）。
//
// FFI 错误模型（设计文档 7.4）：aibridge_status_t 返回码 + aibridge_last_error()
// 线程局部 JSON：{"code":"...","message":"...","details":...,"retryable":bool}。
// Client 在 FFI 失败后同线程读取 last_error 转存，按 code 映射为子类异常。
//
// 注意：code 字符串必须与 aibridge-core error.rs 的 AibridgeError::code() 实际返回值
// 完全一致（authentication_error / rate_limit_error / validation_error / model_not_found /
// api_error / network_error / timeout_error / unsupported_capability / provider_not_found /
// voice_not_available / service_unavailable），以保证五语言跨绑定错误码统一。
// ============================================================================

/// <summary>AIBridge 异常基类。</summary>
public class AibridgeException : Exception
{
    /// <summary>错误码字符串（如 "rate_limit_error"、"authentication_error"）。</summary>
    public string Code { get; }

    /// <summary>是否可重试。</summary>
    public bool Retryable { get; }

    /// <summary>原始 details（可能为 null）。</summary>
    public JsonElement? Details { get; }

    public AibridgeException(string message, string code = "unknown", bool retryable = false,
        JsonElement? details = null, Exception? inner = null)
        : base(message, inner)
    {
        Code = code;
        Retryable = retryable;
        Details = details;
    }

    /// <summary>内部构造：仅状态码已知、无 last_error JSON 时，按状态码映射默认子类。</summary>
    internal static AibridgeException FromStatus(int status, string? lastErrorJson)
    {
        // 解析 last_error JSON（同线程立即读，符合 FFI 线程局部语义）
        string code = "ffi_error";
        string message = $"FFI 调用失败，状态码={status}";
        bool retryable = false;
        JsonElement? details = null;

        if (!string.IsNullOrEmpty(lastErrorJson))
        {
            try
            {
                using JsonDocument doc = JsonDocument.Parse(lastErrorJson);
                JsonElement root = doc.RootElement;
                if (root.TryGetProperty("code", out JsonElement c)) code = c.GetString() ?? code;
                if (root.TryGetProperty("message", out JsonElement m))
                    message = m.GetString() ?? message;
                if (root.TryGetProperty("retryable", out JsonElement r)) retryable = r.GetBoolean();
                if (root.TryGetProperty("details", out JsonElement d)) details = d.Clone();
            }
            catch (JsonException)
            {
                // last_error 不是合法 JSON，回退用状态码 + 原始字符串
                message = lastErrorJson;
            }
        }

        // 先按 code（来自 last_error）映射，更精确
        AibridgeException ex = MapByCode(code, message, retryable, details);
        if (ex != null) return ex;

        // code 未识别时按状态码兜底
        return status switch
        {
            AibridgeStatus.Authentication => new AuthenticationException(message, retryable, details),
            AibridgeStatus.RateLimit => new RateLimitException(message, retryable, details),
            AibridgeStatus.Validation => new ValidationException(message, retryable, details),
            AibridgeStatus.ModelNotFound => new ModelNotFoundException(message, retryable, details),
            AibridgeStatus.Api => new ApiException(message, retryable, details),
            AibridgeStatus.Network => new NetworkException(message, retryable, details),
            AibridgeStatus.Timeout => new TimeoutException_(message, retryable, details),
            AibridgeStatus.UnsupportedCapability => new UnsupportedCapabilityException(message, retryable, details),
            AibridgeStatus.ProviderNotFound => new ProviderNotFoundException(message, retryable, details),
            AibridgeStatus.VoiceNotAvailable => new VoiceNotAvailableException(message, retryable, details),
            AibridgeStatus.ServiceUnavailable => new ServiceUnavailableException(message, retryable, details),
            _ => new AibridgeException(message, code, retryable, details),
        };
    }

    /// <summary>按 last_error 的 code 字段映射子类（与 core AibridgeError::code() 对齐）。</summary>
    private static AibridgeException? MapByCode(string code, string message, bool retryable, JsonElement? details)
    {
        return code switch
        {
            "authentication_error" => new AuthenticationException(message, retryable, details),
            "rate_limit_error" => new RateLimitException(message, retryable, details),
            "validation_error" => new ValidationException(message, retryable, details),
            "model_not_found" => new ModelNotFoundException(message, retryable, details),
            "api_error" => new ApiException(message, retryable, details),
            "network_error" => new NetworkException(message, retryable, details),
            "timeout_error" => new TimeoutException_(message, retryable, details),
            "unsupported_capability" => new UnsupportedCapabilityException(message, retryable, details),
            "provider_not_found" => new ProviderNotFoundException(message, retryable, details),
            "voice_not_available" => new VoiceNotAvailableException(message, retryable, details),
            "service_unavailable" => new ServiceUnavailableException(message, retryable, details),
            _ => null, // 未识别 code，交由状态码兜底
        };
    }
}

// —— 子类（与 core AibridgeError 变体一一对应）——————————————

public class AuthenticationException : AibridgeException
{
    public AuthenticationException(string msg, bool retryable = false, JsonElement? details = null)
        : base(msg, "authentication_error", retryable, details) { }
}

public class RateLimitException : AibridgeException
{
    /// <summary>建议等待秒数（若 provider 返回）。</summary>
    public double? RetryAfter { get; }

    public RateLimitException(string msg, bool retryable = true, JsonElement? details = null)
        : base(msg, "rate_limit_error", retryable, details)
    {
        // 尝试从 details.retry_after 读取
        if (details.HasValue && details.Value.TryGetProperty("retry_after", out JsonElement r)
            && r.ValueKind == JsonValueKind.Number)
        {
            RetryAfter = r.GetDouble();
        }
    }
}

public class ValidationException : AibridgeException
{
    public ValidationException(string msg, bool retryable = false, JsonElement? details = null)
        : base(msg, "validation_error", retryable, details) { }
}

public class ModelNotFoundException : AibridgeException
{
    public ModelNotFoundException(string msg, bool retryable = false, JsonElement? details = null)
        : base(msg, "model_not_found", retryable, details) { }
}

public class ApiException : AibridgeException
{
    public ApiException(string msg, bool retryable = false, JsonElement? details = null)
        : base(msg, "api_error", retryable, details) { }
}

public class NetworkException : AibridgeException
{
    public NetworkException(string msg, bool retryable = true, JsonElement? details = null)
        : base(msg, "network_error", retryable, details) { }
}

/// <summary>避免与 System.TimeoutException 重名，加下划线后缀。</summary>
public class TimeoutException_ : AibridgeException
{
    public TimeoutException_(string msg, bool retryable = true, JsonElement? details = null)
        : base(msg, "timeout_error", retryable, details) { }
}

public class UnsupportedCapabilityException : AibridgeException
{
    public UnsupportedCapabilityException(string msg, bool retryable = false, JsonElement? details = null)
        : base(msg, "unsupported_capability", retryable, details) { }
}

public class ProviderNotFoundException : AibridgeException
{
    public ProviderNotFoundException(string msg, bool retryable = false, JsonElement? details = null)
        : base(msg, "provider_not_found", retryable, details) { }
}

public class VoiceNotAvailableException : AibridgeException
{
    public VoiceNotAvailableException(string msg, bool retryable = false, JsonElement? details = null)
        : base(msg, "voice_not_available", retryable, details) { }
}

public class ServiceUnavailableException : AibridgeException
{
    public ServiceUnavailableException(string msg, bool retryable = true, JsonElement? details = null)
        : base(msg, "service_unavailable", retryable, details) { }
}
