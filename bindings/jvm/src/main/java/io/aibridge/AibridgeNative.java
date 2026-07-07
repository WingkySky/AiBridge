package io.aibridge;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;

/**
 * JNA Library 接口：声明所有 aibridge-ffi 的 C 函数。
 *
 * <p>命名说明：本接口命名为 {@code AibridgeNative} 以避免与 JNA 的
 * {@link com.sun.jna.Native} 工具类同名冲突（后者提供 {@code Native.load}）。
 *
 * <p>JNA 通过 {@code Native.load} 加载 libaibridge（dylib/so/dll），并按 C ABI
 * 调用。句柄（client/stream）用 {@link Pointer} 表示，复杂结构走 JSON 字符串边界，
 * 二进制走 {@link AibridgeBytes}。
 *
 * <p>库搜索路径：
 * <ol>
 *   <li>{@code jna.library.path} 系统属性</li>
 *   <li>{@code java.library.path} 系统属性</li>
 *   <li>{@code DYLD_LIBRARY_PATH} / {@code LD_LIBRARY_PATH} 环境变量</li>
 * </ol>
 *
 * <p>FFI 遗留约束（由 {@link Client} / {@link ChatStream} 负责）：
 * <ul>
 *   <li>{@link #aibridge_last_error()} 线程局部：调 FFI 同线程立即读，转存后抛异常</li>
 *   <li>{@link #aibridge_stream_next} 串行：同一 stream 不可并发 next</li>
 *   <li>{@link #aibridge_bytes_free} / {@link #aibridge_string_free} 必须调（RAII 封装）</li>
 *   <li>client/stream 句柄必须 destroy（RAII 封装）</li>
 * </ul>
 */
public interface AibridgeNative extends Library {

    /** 库名（JNA 自动加平台前缀/后缀：libaibridge.dylib / libaibridge.so / aibridge.dll） */
    String LIBRARY_NAME = "aibridge";

    /** 全局单例（JNA 内部线程安全） */
    AibridgeNative INSTANCE = Native.load(LIBRARY_NAME, AibridgeNative.class);

    // —— FFI 返回码常量（与 aibridge.h 的 AIBRIDGE_* 宏对齐）——
    int AIBRIDGE_OK = 0;
    int AIBRIDGE_STREAM_CHUNK = 0;
    int AIBRIDGE_STREAM_END = 1;
    int AIBRIDGE_ERR_AUTHENTICATION = -1;
    int AIBRIDGE_ERR_RATE_LIMIT = -2;
    int AIBRIDGE_ERR_VALIDATION = -3;
    int AIBRIDGE_ERR_MODEL_NOT_FOUND = -4;
    int AIBRIDGE_ERR_API = -5;
    int AIBRIDGE_ERR_NETWORK = -6;
    int AIBRIDGE_ERR_TIMEOUT = -7;
    int AIBRIDGE_ERR_UNSUPPORTED_CAPABILITY = -8;
    int AIBRIDGE_ERR_PROVIDER_NOT_FOUND = -9;
    int AIBRIDGE_ERR_VOICE_NOT_AVAILABLE = -10;
    int AIBRIDGE_ERR_SERVICE_UNAVAILABLE = -11;
    int AIBRIDGE_ERR_FFI = -100;

    /**
     * 二进制缓冲结构（对应 C 的 {@code aibridge_bytes_t}）。
     *
     * <p>{@code ptr} 指向 Rust 分配的字节缓冲，{@code len} 为长度。
     * 调用方需通过 {@link #aibridge_bytes_free} 释放。
     *
     * <p>JNA {@link Structure.ByReference} 用于 FFI 中 {@code aibridge_bytes_t**}
     * 出参（{@link #aibridge_client_speech} 的 {@code out_audio}）。
     */
    @Structure.FieldOrder({"ptr", "len"})
    class AibridgeBytes extends Structure implements Structure.ByReference {
        /** 指向字节数据的指针（Rust 分配） */
        public Pointer ptr;
        /** 字节数据长度 */
        public long len;

        public AibridgeBytes() {
        }

        /** 从结构体指针构造（用于解引用 {@code aibridge_bytes_t**} 出参） */
        public AibridgeBytes(Pointer p) {
            super(p);
            read();
        }

        /** 从结构体指针读取并拷贝为 Java 字节数组 */
        public byte[] toByteArray() {
            if (ptr == null || len <= 0) {
                return new byte[0];
            }
            return ptr.getByteArray(0, (int) len);
        }
    }

    // —— 生命周期 —— //

    /**
     * 创建客户端。
     *
     * @param provider    Provider 类型（如 "echo"、"openai"），UTF-8 C 字符串
     * @param configJson  ClientOptions 的 JSON（可为 null，等价默认配置）
     * @return client 指针；失败返回 null（错误写入 {@link #aibridge_last_error()}）
     */
    Pointer aibridge_client_new(String provider, String configJson);

    /**
     * 启动客户端（初始化适配器）。
     *
     * @return 0 成功；负数为错误码
     */
    int aibridge_client_start(Pointer client);

    /** 释放客户端句柄（传 null 安全 no-op） */
    void aibridge_client_destroy(Pointer client);

    // —— 阻塞式调用 —— //

    /**
     * 文本对话（阻塞）。
     *
     * @param outResponseJson 写入 ChatCompletion 的 JSON（调用方需 {@link #aibridge_string_free}）
     * @return 0 成功；负数错误码
     */
    int aibridge_client_chat(Pointer client, String requestJson, PointerByReference outResponseJson);

    /**
     * 文字转语音（阻塞，二进制载荷走 {@link AibridgeBytes}）。
     *
     * @param outAudio    写入二进制音频缓冲（可为 null，调用方 {@link #aibridge_bytes_free}）
     * @param outMetaJson 写入 SpeechResult（不含 audio_data）的 JSON（{@link #aibridge_string_free}）
     * @return 0 成功；负数错误码
     */
    int aibridge_client_speech(
            Pointer client,
            String requestJson,
            PointerByReference outAudio,
            PointerByReference outMetaJson);

    // —— 流式 —— //

    /**
     * 创建流式对话（阻塞创建 stream 句柄）。
     *
     * @param outStream 写入 stream 句柄
     * @return 0 成功；负数错误码
     */
    int aibridge_client_chat_stream(
            Pointer client,
            String requestJson,
            PointerByReference outStream);

    /**
     * 拉取下一个流式 chunk（阻塞，串行）。
     *
     * @return 0=chunk（{@code outChunkJson} 写入 JSON）；1=EOF；负数=错误
     */
    int aibridge_stream_next(Pointer stream, PointerByReference outChunkJson);

    /** 释放 stream 句柄（传 null 安全 no-op，触发 Rust drop → tokio task abort） */
    void aibridge_stream_destroy(Pointer stream);

    // —— 错误查询 —— //

    /**
     * 读取当前线程的 last_error（JSON 字符串）。
     *
     * <p>返回指向线程局部缓冲的指针，<b>调用方不应释放</b>。仅在当前线程的下一次
     * FFI 调用前保证有效（thread_local 语义），故调用方需立即读取转存。
     *
     * @return 错误 JSON 指针；当前线程无错误返回 null
     */
    Pointer aibridge_last_error();

    // —— 释放 —— //

    /** 释放 Rust 分配的 C 字符串（传 null 安全 no-op） */
    void aibridge_string_free(Pointer ptr);

    /** 释放 Rust 分配的二进制缓冲（传 null 安全 no-op） */
    void aibridge_bytes_free(AibridgeBytes ptr);
}
