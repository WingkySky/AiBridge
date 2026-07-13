package io.aibridge;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import java.lang.ref.Cleaner;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * AIBridge 客户端（封装 native client 句柄）。
 *
 * <p>通过 JNA 调用 aibridge-ffi 的 {@code aibridge_client_*} 函数，提供：
 * <ul>
 *   <li>{@link #chat}：阻塞文本对话（异步版 {@link #chatAsync}）</li>
 *   <li>{@link #chatStream}：流式文本对话（返回 {@link ChatStream}）</li>
 *   <li>{@link #speech}：文字转语音（异步版 {@link #speechAsync}）</li>
 * </ul>
 *
 * <h3>生命周期与内存管理</h3>
 * <p>句柄用 {@link Cleaner} 兜底释放：即使忘记 {@link #close}，GC 回收时也会调
 * {@code aibridge_client_destroy}。但建议显式 close 以尽早释放资源。
 *
 * <h3>错误处理（FFI 遗留：last_error 线程局部）</h3>
 * <p>每个 FFI 调用失败后，<b>在同一线程立即</b>读取 {@code aibridge_last_error()} 转存
 * 为字符串，再映射为对应子类异常抛出。避免跨线程读取失效指针。
 *
 * <h3>异步</h3>
 * <p>{@code *Async} 方法用 {@link CompletableFuture#supplyAsync} 在固定线程池上执行
 * 阻塞 FFI 调用（FFI 内部 {@code block_on} 会阻塞调用线程）。
 */
public class Client implements AutoCloseable {

    /** 共享异步执行器（虚拟线程，适合阻塞 IO 密集的 FFI 调用） */
    private static final Executor ASYNC_EXECUTOR =
            Executors.newThreadPerTaskExecutor(Thread.ofVirtual().name("aibridge-ffi-", 0).factory());

    private static final Cleaner CLEANER = Cleaner.create();
    private static final ObjectMapper MAPPER = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    /** native client 句柄（null 表示已关闭） */
    private volatile Pointer handle;
    /** 防止重复 close */
    private final AtomicBoolean closed = new AtomicBoolean(false);
    /** Cleaner 注册的清理动作（兜底释放句柄） */
    private final Cleaner.Cleanable cleanable;

    /**
     * 创建客户端。
     *
     * @param provider   Provider 类型（如 "echo"、"openai"）
     * @param configJson ClientOptions 的 JSON（可为 null，等价默认配置）
     * @throws AibridgeException 创建失败（如 provider 不存在、config 非法）
     */
    public Client(String provider, String configJson) {
        Pointer ptr = AibridgeNative.INSTANCE.aibridge_client_new(provider, configJson);
        if (ptr == null) {
            // client_new 失败时 last_error 已写入，立即同线程读取转存
            throw readLastError();
        }
        this.handle = ptr;
        // Cleaner 兜底：GC 回收时调 destroy（传 null 安全 no-op）
        this.cleanable = CLEANER.register(this, () -> AibridgeNative.INSTANCE.aibridge_client_destroy(ptr));
    }

    /** 创建客户端（默认配置） */
    public Client(String provider) {
        this(provider, null);
    }

    /**
     * 启动客户端（初始化适配器）。
     *
     * @throws AibridgeException 启动失败
     */
    public void start() {
        Pointer ptr = requireHandle();
        int status = AibridgeNative.INSTANCE.aibridge_client_start(ptr);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
    }

    /**
     * 文本对话（阻塞）。
     *
     * @param request 对话请求
     * @return 对话完成结果
     * @throws AibridgeException 调用失败
     */
    public ChatCompletion chat(ChatRequest request) {
        Pointer ptr = requireHandle();
        String requestJson = writeJson(request);

        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_chat(ptr, requestJson, outRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
        // out_response_json 由 Rust 分配，必须 string_free
        Pointer jsonPtr = outRef.getValue();
        try {
            String json = jsonPtr.getString(0, "UTF-8");
            return parseJson(json, ChatCompletion.class);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(jsonPtr);
        }
    }

    /**
     * 文本对话（异步）。
     *
     * @return CompletableFuture，正常完成时携带 ChatCompletion
     */
    public CompletableFuture<ChatCompletion> chatAsync(ChatRequest request) {
        return CompletableFuture.supplyAsync(() -> chat(request), ASYNC_EXECUTOR);
    }

    /**
     * 流式文本对话（创建 stream 句柄并启动后台拉取）。
     *
     * <p>返回的 {@link ChatStream} 实现了 {@link java.util.Iterator}，可阻塞遍历 chunk。
     * 使用完毕务必 {@link ChatStream#close()} 释放 stream 句柄（try-with-resources 推荐）。
     *
     * @param request 对话请求（{@code stream} 字段会被强制设为 true）
     * @return 流式迭代器
     * @throws AibridgeException 创建流失败
     */
    public ChatStream chatStream(ChatRequest request) {
        Pointer ptr = requireHandle();
        // 强制 stream=true（语义清晰，避免调用方遗漏）
        if (request.stream == null) {
            request.stream = true;
        }
        String requestJson = writeJson(request);

        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_chat_stream(ptr, requestJson, outRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
        Pointer streamPtr = outRef.getValue();
        if (streamPtr == null) {
            throw new AibridgeException(AibridgeException.CODE_FFI,
                    "chat_stream 返回成功但 stream 句柄为空", null, false);
        }
        return new ChatStream(streamPtr);
    }

    /**
     * 文字转语音（阻塞）。
     *
     * @param request 语音请求
     * @return 完整结果（meta + 二进制音频）
     * @throws AibridgeException 调用失败
     */
    public SpeechResultFull speech(SpeechRequest request) {
        Pointer ptr = requireHandle();
        String requestJson = writeJson(request);

        PointerByReference outAudioRef = new PointerByReference();
        PointerByReference outMetaRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_speech(ptr, requestJson, outAudioRef, outMetaRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }

        // 二进制音频（可为 null：Provider 仅返回 base64/url 时）
        Pointer audioPtr = outAudioRef.getValue();
        byte[] audioData;
        if (audioPtr == null) {
            audioData = new byte[0];
        } else {
            AibridgeNative.AibridgeBytes bytes = new AibridgeNative.AibridgeBytes(audioPtr);
            bytes.read();
            audioData = bytes.toByteArray();
            AibridgeNative.INSTANCE.aibridge_bytes_free(bytes);
        }

        // meta JSON（SpeechResult，audio_data 被 skip）
        Pointer metaPtr = outMetaRef.getValue();
        try {
            String json = metaPtr.getString(0, "UTF-8");
            SpeechResult meta = parseJson(json, SpeechResult.class);
            return new SpeechResultFull(meta, audioData);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(metaPtr);
        }
    }

    /**
     * 文字转语音（异步）。
     *
     * @return CompletableFuture，正常完成时携带 SpeechResultFull
     */
    public CompletableFuture<SpeechResultFull> speechAsync(SpeechRequest request) {
        return CompletableFuture.supplyAsync(() -> speech(request), ASYNC_EXECUTOR);
    }

    /** 关闭客户端，释放 native 句柄。多次调用安全。 */
    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        // 取消 Cleaner 兜底（避免重复 destroy，destroy(null) 亦安全）
        cleanable.clean();
        handle = null;
    }

    // —— 内部辅助 —— //

    /** 校验句柄有效，否则抛 ffi_error */
    private Pointer requireHandle() {
        Pointer ptr = handle;
        if (ptr == null) {
            throw new AibridgeException(AibridgeException.CODE_FFI,
                    "client 句柄为空（已 close 或未初始化）", null, false);
        }
        return ptr;
    }

    /**
     * 读取当前线程的 last_error 并映射为对应子类异常。
     *
     * <p>FFI 遗留：last_error 是线程局部，必须在与触发错误的 FFI 调用相同的线程立即读取。
     * 本方法在 FFI 调用失败后立即被调用，故线程一致。
     */
    private static AibridgeException readLastError() {
        Pointer errPtr = AibridgeNative.INSTANCE.aibridge_last_error();
        if (errPtr == null) {
            return new AibridgeException(AibridgeException.CODE_FFI,
                    "未知错误（last_error 为空）", null, false);
        }
        // 立即转存（指针仅在下次 FFI 调用前有效）
        String json = errPtr.getString(0, "UTF-8");
        return parseError(json);
    }

    /** 解析 last_error JSON 并映射为对应子类异常 */
    private static AibridgeException parseError(String json) {
        try {
            ErrorPayload payload = MAPPER.readValue(json, ErrorPayload.class);
            String code = payload.code != null ? payload.code : AibridgeException.CODE_FFI;
            String details = payload.details != null ? payload.details : "null";
            boolean retryable = Boolean.TRUE.equals(payload.retryable);
            String message = payload.message != null ? payload.message : "(无错误消息)";
            return mapToException(code, message, details, retryable);
        } catch (Exception e) {
            // JSON 解析失败，回退 ffi_error
            return new AibridgeException(AibridgeException.CODE_FFI,
                    "last_error JSON 解析失败: " + e.getMessage() + " (原始: " + json + ")",
                    null, false, e);
        }
    }

    /** 按 code 映射到具体子类（与 aibridge-core error.rs code() 对齐） */
    private static AibridgeException mapToException(String code, String message, String details, boolean retryable) {
        switch (code) {
            case AibridgeException.CODE_AUTHENTICATION:
                return new AuthenticationException(message, details, retryable);
            case AibridgeException.CODE_RATE_LIMIT:
                return new RateLimitException(message, details, retryable);
            case AibridgeException.CODE_VALIDATION:
                return new ValidationException(message, details, retryable);
            case AibridgeException.CODE_MODEL_NOT_FOUND:
                return new ModelNotFoundException(message, details, retryable);
            case AibridgeException.CODE_API:
                return new ApiException(message, details, retryable);
            case AibridgeException.CODE_NETWORK:
                return new NetworkException(message, details, retryable);
            case AibridgeException.CODE_TIMEOUT:
                return new TimeoutException(message, details, retryable);
            case AibridgeException.CODE_UNSUPPORTED_CAPABILITY:
                return new UnsupportedCapabilityException(message, details, retryable);
            case AibridgeException.CODE_PROVIDER_NOT_FOUND:
                return new ProviderNotFoundException(message, details, retryable);
            case AibridgeException.CODE_VOICE_NOT_AVAILABLE:
                return new VoiceNotAvailableException(message, details, retryable);
            case AibridgeException.CODE_SERVICE_UNAVAILABLE:
                return new ServiceUnavailableException(message, details, retryable);
            default:
                return new AibridgeException(code, message, details, retryable);
        }
    }

    /** 序列化为 JSON 字符串 */
    private static String writeJson(Object obj) {
        try {
            return MAPPER.writeValueAsString(obj);
        } catch (Exception e) {
            throw new AibridgeException(AibridgeException.CODE_FFI,
                    "JSON 序列化失败: " + e.getMessage(), null, false, e);
        }
    }

    /** 反序列化 JSON */
    private static <T> T parseJson(String json, Class<T> type) {
        try {
            return MAPPER.readValue(json, type);
        } catch (Exception e) {
            throw new AibridgeException(AibridgeException.CODE_FFI,
                    type.getSimpleName() + " JSON 反序列化失败: " + e.getMessage()
                            + " (原始: " + json + ")", null, false, e);
        }
    }

    /** last_error JSON 载荷（内部解析用） */
    private static class ErrorPayload {
        public String code;
        public String message;
        public String details;
        public Boolean retryable;
    }

    /** 便捷构造：user 消息单条 */
    public static List<ChatMessage> userMessages(String... texts) {
        java.util.ArrayList<ChatMessage> list = new java.util.ArrayList<>();
        for (String t : texts) {
            list.add(ChatMessage.user(t));
        }
        return list;
    }
}
