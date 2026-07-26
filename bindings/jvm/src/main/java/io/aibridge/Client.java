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
 * <p>通过 JNA 调用 aibridge-ffi 的 {@code aibridge_client_*} 函数，提供全部能力：
 * <ul>
 *   <li>{@link #chat} / {@link #chatAsync}：阻塞文本对话</li>
 *   <li>{@link #chatStream}：流式文本对话</li>
 *   <li>{@link #speech} / {@link #speechAsync}：文字转语音</li>
 *   <li>{@link #imageGenerate} / {@link #imageGenerateAsync}：图像生成</li>
 *   <li>{@link #videoCreate} / {@link #videoCreateAsync}：创建视频任务</li>
 *   <li>{@link #videoPoll} / {@link #videoPollAsync}：查询视频状态</li>
 *   <li>{@link #embed} / {@link #embedAsync}：文本嵌入</li>
 *   <li>{@link #transcribe} / {@link #transcribeAsync}：语音转文字</li>
 *   <li>{@link #translate} / {@link #translateAsync}：语音翻译</li>
 *   <li>{@link #listModels} / {@link #listModelsAsync}：获取模型列表</li>
 *   <li>{@link #listVoices} / {@link #listVoicesAsync}：获取音色列表</li>
 *   <li>{@link #recommendVoices} / {@link #recommendVoicesAsync}：推荐音色</li>
 * </ul>
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

    // ────────────────────────────────────────────────────────────────────
    // 生命周期
    // ────────────────────────────────────────────────────────────────────

    public Client(String provider, String configJson) {
        Pointer ptr = AibridgeNative.INSTANCE.aibridge_client_new(provider, configJson);
        if (ptr == null) {
            throw readLastError();
        }
        this.handle = ptr;
        this.cleanable = CLEANER.register(this, () -> AibridgeNative.INSTANCE.aibridge_client_destroy(ptr));
    }

    public Client(String provider) {
        this(provider, null);
    }

    public void start() {
        Pointer ptr = requireHandle();
        int status = AibridgeNative.INSTANCE.aibridge_client_start(ptr);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        cleanable.clean();
        handle = null;
    }

    // ────────────────────────────────────────────────────────────────────
    // 文本对话
    // ────────────────────────────────────────────────────────────────────

    public ChatCompletion chat(ChatRequest request) {
        Pointer ptr = requireHandle();
        String requestJson = writeJson(request);

        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_chat(ptr, requestJson, outRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
        Pointer jsonPtr = outRef.getValue();
        try {
            String json = jsonPtr.getString(0, "UTF-8");
            return parseJson(json, ChatCompletion.class);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(jsonPtr);
        }
    }

    public CompletableFuture<ChatCompletion> chatAsync(ChatRequest request) {
        return CompletableFuture.supplyAsync(() -> chat(request), ASYNC_EXECUTOR);
    }

    public ChatStream chatStream(ChatRequest request) {
        Pointer ptr = requireHandle();
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

    // ────────────────────────────────────────────────────────────────────
    // 文字转语音
    // ────────────────────────────────────────────────────────────────────

    public SpeechResultFull speech(SpeechRequest request) {
        Pointer ptr = requireHandle();
        String requestJson = writeJson(request);

        PointerByReference outAudioRef = new PointerByReference();
        PointerByReference outMetaRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_speech(ptr, requestJson, outAudioRef, outMetaRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }

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

        Pointer metaPtr = outMetaRef.getValue();
        try {
            String json = metaPtr.getString(0, "UTF-8");
            SpeechResult meta = parseJson(json, SpeechResult.class);
            return new SpeechResultFull(meta, audioData);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(metaPtr);
        }
    }

    public CompletableFuture<SpeechResultFull> speechAsync(SpeechRequest request) {
        return CompletableFuture.supplyAsync(() -> speech(request), ASYNC_EXECUTOR);
    }

    // ────────────────────────────────────────────────────────────────────
    // 图像生成
    // ────────────────────────────────────────────────────────────────────

    public ImageResult imageGenerate(ImageRequest request) {
        Pointer ptr = requireHandle();
        String requestJson = writeJson(request);

        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_image_generate(ptr, requestJson, outRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
        Pointer jsonPtr = outRef.getValue();
        try {
            String json = jsonPtr.getString(0, "UTF-8");
            return parseJson(json, ImageResult.class);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(jsonPtr);
        }
    }

    public CompletableFuture<ImageResult> imageGenerateAsync(ImageRequest request) {
        return CompletableFuture.supplyAsync(() -> imageGenerate(request), ASYNC_EXECUTOR);
    }

    // ────────────────────────────────────────────────────────────────────
    // 视频生成
    // ────────────────────────────────────────────────────────────────────

    public VideoTask videoCreate(VideoRequest request) {
        Pointer ptr = requireHandle();
        String requestJson = writeJson(request);

        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_video_create(ptr, requestJson, outRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
        Pointer jsonPtr = outRef.getValue();
        try {
            String json = jsonPtr.getString(0, "UTF-8");
            return parseJson(json, VideoTask.class);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(jsonPtr);
        }
    }

    public CompletableFuture<VideoTask> videoCreateAsync(VideoRequest request) {
        return CompletableFuture.supplyAsync(() -> videoCreate(request), ASYNC_EXECUTOR);
    }

    public VideoStatus videoPoll(String taskId, String model) {
        Pointer ptr = requireHandle();
        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_video_poll(ptr, taskId, model, outRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
        Pointer jsonPtr = outRef.getValue();
        try {
            String json = jsonPtr.getString(0, "UTF-8");
            return parseJson(json, VideoStatus.class);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(jsonPtr);
        }
    }

    public CompletableFuture<VideoStatus> videoPollAsync(String taskId, String model) {
        return CompletableFuture.supplyAsync(() -> videoPoll(taskId, model), ASYNC_EXECUTOR);
    }

    // ────────────────────────────────────────────────────────────────────
    // 文本嵌入
    // ────────────────────────────────────────────────────────────────────

    public EmbeddingResult embed(EmbedRequest request) {
        Pointer ptr = requireHandle();
        String requestJson = writeJson(request);

        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_embed(ptr, requestJson, outRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
        Pointer jsonPtr = outRef.getValue();
        try {
            String json = jsonPtr.getString(0, "UTF-8");
            return parseJson(json, EmbeddingResult.class);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(jsonPtr);
        }
    }

    public CompletableFuture<EmbeddingResult> embedAsync(EmbedRequest request) {
        return CompletableFuture.supplyAsync(() -> embed(request), ASYNC_EXECUTOR);
    }

    // ────────────────────────────────────────────────────────────────────
    // 语音转文字 / 翻译
    // ────────────────────────────────────────────────────────────────────

    public TranscriptionResult transcribe(TranscribeRequest request) {
        Pointer ptr = requireHandle();
        String requestJson = writeJson(request);

        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_transcribe(ptr, requestJson, outRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
        Pointer jsonPtr = outRef.getValue();
        try {
            String json = jsonPtr.getString(0, "UTF-8");
            return parseJson(json, TranscriptionResult.class);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(jsonPtr);
        }
    }

    public CompletableFuture<TranscriptionResult> transcribeAsync(TranscribeRequest request) {
        return CompletableFuture.supplyAsync(() -> transcribe(request), ASYNC_EXECUTOR);
    }

    public TranscriptionResult translate(TranscribeRequest request) {
        Pointer ptr = requireHandle();
        String requestJson = writeJson(request);

        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_translate(ptr, requestJson, outRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
        Pointer jsonPtr = outRef.getValue();
        try {
            String json = jsonPtr.getString(0, "UTF-8");
            return parseJson(json, TranscriptionResult.class);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(jsonPtr);
        }
    }

    public CompletableFuture<TranscriptionResult> translateAsync(TranscribeRequest request) {
        return CompletableFuture.supplyAsync(() -> translate(request), ASYNC_EXECUTOR);
    }

    // ────────────────────────────────────────────────────────────────────
    // 模型列表 / 音色列表
    // ────────────────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    public List<ModelInfo> listModels(String filter) {
        Pointer ptr = requireHandle();
        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_list_models(ptr, filter, outRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
        Pointer jsonPtr = outRef.getValue();
        try {
            String json = jsonPtr.getString(0, "UTF-8");
            return parseJsonList(json, ModelInfo.class);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(jsonPtr);
        }
    }

    public CompletableFuture<List<ModelInfo>> listModelsAsync(String filter) {
        return CompletableFuture.supplyAsync(() -> listModels(filter), ASYNC_EXECUTOR);
    }

    @SuppressWarnings("unchecked")
    public List<VoiceInfo> listVoices(String language) {
        Pointer ptr = requireHandle();
        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_list_voices(ptr, language, outRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
        Pointer jsonPtr = outRef.getValue();
        try {
            String json = jsonPtr.getString(0, "UTF-8");
            return parseJsonList(json, VoiceInfo.class);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(jsonPtr);
        }
    }

    public CompletableFuture<List<VoiceInfo>> listVoicesAsync(String language) {
        return CompletableFuture.supplyAsync(() -> listVoices(language), ASYNC_EXECUTOR);
    }

    @SuppressWarnings("unchecked")
    public List<VoiceInfo> recommendVoices(String language, String gender, int limit) {
        Pointer ptr = requireHandle();
        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_client_recommend_voices(
                ptr, language, gender, limit, outRef);
        if (status != AibridgeNative.AIBRIDGE_OK) {
            throw readLastError();
        }
        Pointer jsonPtr = outRef.getValue();
        try {
            String json = jsonPtr.getString(0, "UTF-8");
            return parseJsonList(json, VoiceInfo.class);
        } finally {
            AibridgeNative.INSTANCE.aibridge_string_free(jsonPtr);
        }
    }

    public CompletableFuture<List<VoiceInfo>> recommendVoicesAsync(String language, String gender, int limit) {
        return CompletableFuture.supplyAsync(() -> recommendVoices(language, gender, limit), ASYNC_EXECUTOR);
    }

    // ────────────────────────────────────────────────────────────────────
    // 内部辅助
    // ────────────────────────────────────────────────────────────────────

    private Pointer requireHandle() {
        Pointer ptr = handle;
        if (ptr == null) {
            throw new AibridgeException(AibridgeException.CODE_FFI,
                    "client 句柄为空（已 close 或未初始化）", null, false);
        }
        return ptr;
    }

    private static AibridgeException readLastError() {
        Pointer errPtr = AibridgeNative.INSTANCE.aibridge_last_error();
        if (errPtr == null) {
            return new AibridgeException(AibridgeException.CODE_FFI,
                    "未知错误（last_error 为空）", null, false);
        }
        String json = errPtr.getString(0, "UTF-8");
        return parseError(json);
    }

    private static AibridgeException parseError(String json) {
        try {
            ErrorPayload payload = MAPPER.readValue(json, ErrorPayload.class);
            String code = payload.code != null ? payload.code : AibridgeException.CODE_FFI;
            String details = payload.details != null ? payload.details : "null";
            boolean retryable = Boolean.TRUE.equals(payload.retryable);
            String message = payload.message != null ? payload.message : "(无错误消息)";
            return mapToException(code, message, details, retryable);
        } catch (Exception e) {
            return new AibridgeException(AibridgeException.CODE_FFI,
                    "last_error JSON 解析失败: " + e.getMessage() + " (原始: " + json + ")",
                    null, false, e);
        }
    }

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

    private static String writeJson(Object obj) {
        try {
            return MAPPER.writeValueAsString(obj);
        } catch (Exception e) {
            throw new AibridgeException(AibridgeException.CODE_FFI,
                    "JSON 序列化失败: " + e.getMessage(), null, false, e);
        }
    }

    private static <T> T parseJson(String json, Class<T> type) {
        try {
            return MAPPER.readValue(json, type);
        } catch (Exception e) {
            throw new AibridgeException(AibridgeException.CODE_FFI,
                    type.getSimpleName() + " JSON 反序列化失败: " + e.getMessage()
                            + " (原始: " + json + ")", null, false, e);
        }
    }

    @SuppressWarnings("unchecked")
    private static <T> List<T> parseJsonList(String json, Class<T> itemType) {
        try {
            return MAPPER.readValue(json,
                    MAPPER.getTypeFactory().constructCollectionType(List.class, itemType));
        } catch (Exception e) {
            throw new AibridgeException(AibridgeException.CODE_FFI,
                    "JSON 数组反序列化失败: " + e.getMessage()
                            + " (原始: " + json + ")", null, false, e);
        }
    }

    private static AibridgeException mapCodeToException(String code, String message, String details, boolean retryable) {
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

    private static class ErrorPayload {
        public String code;
        public String message;
        public String details;
        public Boolean retryable;
    }

    public static List<ChatMessage> userMessages(String... texts) {
        java.util.ArrayList<ChatMessage> list = new java.util.ArrayList<>();
        for (String t : texts) {
            list.add(ChatMessage.user(t));
        }
        return list;
    }
}
