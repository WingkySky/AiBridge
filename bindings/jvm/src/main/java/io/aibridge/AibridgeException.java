package io.aibridge;

/**
 * AIBridge 异常基类（对应设计文档 9.3 节 JVM 异常映射）。
 *
 * <p>FFI 错误模型：{@code aibridge_status_t} 返回码（0 成功 / 负数错误类别）+
 * {@code aibridge_last_error()} 线程局部 JSON（含 {@code code/message/details/retryable}）。
 * 本类承载 last_error 的解析结果，子类按 {@code code} 分类。
 *
 * <p>典型用法：
 * <pre>{@code
 * try {
 *     client.chat(req);
 * } catch (RateLimitException e) {
 *     if (e.isRetryable()) { /* 退避重试 *\/ }
 * }
 * }</pre>
 */
public class AibridgeException extends RuntimeException {

    /** 错误码（如 "rate_limit_error"、"validation_error"、"ffi_error"） */
    private final String code;
    /** 详细信息（JSON 值，可能为 null） */
    private final String details;
    /** 是否可重试 */
    private final boolean retryable;

    public AibridgeException(String code, String message, String details, boolean retryable) {
        super(message);
        this.code = code;
        this.details = details;
        this.retryable = retryable;
    }

    public AibridgeException(String code, String message, String details, boolean retryable, Throwable cause) {
        super(message, cause);
        this.code = code;
        this.details = details;
        this.retryable = retryable;
    }

    /** 错误码（如 "rate_limit_error"） */
    public String getCode() {
        return code;
    }

    /** 详细信息（原始 JSON 字符串，可能为 null 或 "null"） */
    public String getDetails() {
        return details;
    }

    /** 是否可重试 */
    public boolean isRetryable() {
        return retryable;
    }

    @Override
    public String toString() {
        String retry = retryable ? " (retryable)" : "";
        return getClass().getSimpleName() + "[" + code + "]" + retry + ": " + getMessage();
    }

    // —— 错误码常量（与 aibridge-core error.rs code() 对齐）—— //
    /** 认证错误 */
    public static final String CODE_AUTHENTICATION = "authentication_error";
    /** 限流错误 */
    public static final String CODE_RATE_LIMIT = "rate_limit_error";
    /** 参数校验错误 */
    public static final String CODE_VALIDATION = "validation_error";
    /** 模型不存在 */
    public static final String CODE_MODEL_NOT_FOUND = "model_not_found";
    /** API 调用错误（HTTP 4xx/5xx） */
    public static final String CODE_API = "api_error";
    /** 网络错误 */
    public static final String CODE_NETWORK = "network_error";
    /** 超时 */
    public static final String CODE_TIMEOUT = "timeout_error";
    /** 不支持的能力 */
    public static final String CODE_UNSUPPORTED_CAPABILITY = "unsupported_capability";
    /** Provider 不存在 */
    public static final String CODE_PROVIDER_NOT_FOUND = "provider_not_found";
    /** 音色不可用 */
    public static final String CODE_VOICE_NOT_AVAILABLE = "voice_not_available";
    /** 服务暂时不可用 */
    public static final String CODE_SERVICE_UNAVAILABLE = "service_unavailable";
    /** FFI 层通用错误（参数为空、JSON 解析失败、内部 panic 等） */
    public static final String CODE_FFI = "ffi_error";
}
