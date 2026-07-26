package io.aibridge;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;

/**
 * JNA Library 接口：声明所有 aibridge-ffi 的 C 函数。
 */
public interface AibridgeNative extends Library {

    String LIBRARY_NAME = "aibridge";
    AibridgeNative INSTANCE = Native.load(LIBRARY_NAME, AibridgeNative.class);

    // —— FFI 返回码常量 ——
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
     * 二进制缓冲结构（对应 C 的 aibridge_bytes_t）。
     */
    @Structure.FieldOrder({"ptr", "len"})
    class AibridgeBytes extends Structure implements Structure.ByReference {
        public Pointer ptr;
        public long len;

        public AibridgeBytes() {}
        public AibridgeBytes(Pointer p) { super(p); read(); }

        public byte[] toByteArray() {
            if (ptr == null || len <= 0) return new byte[0];
            return ptr.getByteArray(0, (int) len);
        }
    }

    // —— 生命周期 ——

    Pointer aibridge_client_new(String provider, String configJson);
    int aibridge_client_start(Pointer client);
    void aibridge_client_destroy(Pointer client);

    // —— 阻塞式调用 ——

    int aibridge_client_chat(Pointer client, String requestJson, PointerByReference outResponseJson);
    int aibridge_client_speech(Pointer client, String requestJson,
                               PointerByReference outAudio, PointerByReference outMetaJson);
    int aibridge_client_image_generate(Pointer client, String requestJson, PointerByReference outResponseJson);
    int aibridge_client_video_create(Pointer client, String requestJson, PointerByReference outResponseJson);
    int aibridge_client_video_poll(Pointer client, String taskId, String model, PointerByReference outResponseJson);
    int aibridge_client_embed(Pointer client, String requestJson, PointerByReference outResponseJson);
    int aibridge_client_transcribe(Pointer client, String requestJson, PointerByReference outResponseJson);
    int aibridge_client_translate(Pointer client, String requestJson, PointerByReference outResponseJson);
    int aibridge_client_list_models(Pointer client, String filter, PointerByReference outResponseJson);
    int aibridge_client_list_voices(Pointer client, String language, PointerByReference outResponseJson);
    int aibridge_client_recommend_voices(Pointer client, String language, String gender,
                                          int limit, PointerByReference outResponseJson);

    // —— 流式 ——

    int aibridge_client_chat_stream(Pointer client, String requestJson, PointerByReference outStream);
    int aibridge_stream_next(Pointer stream, PointerByReference outChunkJson);
    void aibridge_stream_destroy(Pointer stream);

    // —— 错误查询 ——

    Pointer aibridge_last_error();

    // —— 释放 ——

    void aibridge_string_free(Pointer ptr);
    void aibridge_bytes_free(AibridgeBytes ptr);
}
