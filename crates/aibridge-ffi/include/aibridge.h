/*
 * AIBridge C ABI 头文件 - 由 cbindgen 自动生成，请勿手动修改
 *
 * 供 Go (CGO) / JVM (JNA) / .NET (P/Invoke) 调用 aibridge-ffi cdylib。
 * Python / JS 通过 aibridge-python / aibridge-node 直连 aibridge-core，不走本头文件。
 *
 * 详见设计文档 docs/superpowers/specs/2026-07-07-aibridge-rust-rewrite-design.md 第 7 节。
 */


#ifndef AIBRIDGE_H
#define AIBRIDGE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Client 句柄
 *
 * 内部用 `Arc<tokio::sync::Mutex<Client>>`：
 * - `Arc` 允许 stream task 持有 client 引用
 * - `tokio::sync::Mutex` 处理 core 的 `start`/`close` 是 `&mut self`，
 *   且其 guard 可跨 `.await`（避免 `std::sync::Mutex` 跨 await 的死锁/panic）
 */
typedef struct AibridgeClient AibridgeClient;

/**
 * Stream 句柄
 *
 * 持有流式 `ChatStream` 与一个可选的 tokio task handle。
 * `drop` 时若 task 存在则 abort，保证流式任务不会泄漏。
 *
 * 注意：`aibridge_stream_next` 在调用方线程串行拉取（各语言绑定负责同步）。
 */
typedef struct AibridgeStream AibridgeStream;

/**
 * 二进制缓冲结构（`#[repr(C)]` 保证 C 侧布局）
 *
 * 对应设计文档的 `aibridge_bytes_t`。`ptr` 指向 Rust 分配的字节缓冲，
 * `len` 为长度。调用方通过 [`crate::string::aibridge_bytes_free`] 释放。
 */
typedef struct {
  /**
   * 指向字节数据的指针（Rust 分配）
   */
  const uint8_t *ptr;
  /**
   * 字节数据长度
   */
  uintptr_t len;
} aibridge_bytes_t;

/**
 * FFI 返回码类型
 *
 * 0 表示成功；负数表示各类错误（与 [`AibridgeError`] 变体一一对应）。
 */
typedef int32_t AibridgeStatus;

/**
 * 错误码类型别名（供 cbindgen 导出为 typedef）
 */
typedef AibridgeStatus aibridge_status_t;

/**
 * `aibridge_client_t` opaque 类型
 */
typedef AibridgeClient aibridge_client_t;

/**
 * `aibridge_stream_t` opaque 类型
 */
typedef AibridgeStream aibridge_stream_t;

/**
 * API 调用错误（HTTP 4xx/5xx）
 */
#define AIBRIDGE_ERR_API -5

/**
 * 认证错误
 */
#define AIBRIDGE_ERR_AUTHENTICATION -1

/**
 * FFI 层通用错误（参数为空、JSON 解析失败、内部 panic 等）
 */
#define AIBRIDGE_ERR_FFI -100

/**
 * 模型不存在
 */
#define AIBRIDGE_ERR_MODEL_NOT_FOUND -4

/**
 * 网络错误
 */
#define AIBRIDGE_ERR_NETWORK -6

/**
 * Provider 不存在
 */
#define AIBRIDGE_ERR_PROVIDER_NOT_FOUND -9

/**
 * 限流错误
 */
#define AIBRIDGE_ERR_RATE_LIMIT -2

/**
 * 服务暂时不可用
 */
#define AIBRIDGE_ERR_SERVICE_UNAVAILABLE -11

/**
 * 超时
 */
#define AIBRIDGE_ERR_TIMEOUT -7

/**
 * 不支持的能力
 */
#define AIBRIDGE_ERR_UNSUPPORTED_CAPABILITY -8

/**
 * 参数校验错误
 */
#define AIBRIDGE_ERR_VALIDATION -3

/**
 * 音色不可用
 */
#define AIBRIDGE_ERR_VOICE_NOT_AVAILABLE -10

/**
 * 成功
 */
#define AIBRIDGE_OK 0

/**
 * 流式：正常拉取到一个 chunk（仅 `aibridge_stream_next` 使用）
 */
#define AIBRIDGE_STREAM_CHUNK 0

/**
 * 流式：流已结束（仅 `aibridge_stream_next` 使用，1 表示 EOF）
 */
#define AIBRIDGE_STREAM_END 1

/**
 * 释放 Rust 分配的二进制缓冲
 *
 * # Safety
 * `ptr` 必须是指向由 FFI 函数（如 `aibridge_client_speech`）输出的
 * `aibridge_bytes_t*` 指针，且只能释放一次。传 `nullptr` 是安全的 no-op。
 *
 * 对应设计文档的 `aibridge_bytes_free`。
 */
void aibridge_bytes_free(aibridge_bytes_t *ptr);

/**
 * 文本对话（阻塞）
 *
 * `request_json` 为 `ChatRequest` 的 JSON 序列化字符串。
 * 成功时 `*out_response_json` 写入 `ChatCompletion` 的 JSON，调用方需通过
 * [`aibridge_string_free`] 释放。
 *
 * # Safety
 * - `client` 必须来自 [`aibridge_client_new`]
 * - `request_json` 必须是合法 JSON C 字符串
 * - `out_response_json` 必须指向可写的 `*mut c_char` 槽位
 */
aibridge_status_t aibridge_client_chat(aibridge_client_t *client,
                                       const char *request_json,
                                       char **out_response_json);

/**
 * 流式文本对话（创建 stream 句柄）
 *
 * `request_json` 为 `ChatRequest` 的 JSON 序列化字符串。
 * 成功时 `*out_stream` 写入 stream 句柄，调用方通过
 * [`aibridge_stream_next`] 拉取 chunk，最后用 [`aibridge_stream_destroy`] 释放。
 *
 * # Safety
 * - `client` / `request_json` / `out_stream` 均需合法非空
 */
aibridge_status_t aibridge_client_chat_stream(aibridge_client_t *client,
                                              const char *request_json,
                                              aibridge_stream_t **out_stream);

/**
 * 释放客户端句柄
 *
 * # Safety
 * `client` 必须是 [`aibridge_client_new`] 返回的指针，且只能释放一次。
 * 传 `nullptr` 是安全的 no-op。
 */
void aibridge_client_destroy(aibridge_client_t *client);

/**
 * 文本嵌入（阻塞）
 *
 * `request_json` 为 `EmbedRequest` 的 JSON 序列化字符串。
 * 成功时 `*out_response_json` 写入 `EmbeddingResult` 的 JSON。
 *
 * # Safety
 * - `client` 必须来自 [`aibridge_client_new`]
 * - `request_json` 必须是合法 JSON C 字符串
 * - `out_response_json` 必须指向可写的 `*mut c_char` 槽位
 */
aibridge_status_t aibridge_client_embed(aibridge_client_t *client,
                                        const char *request_json,
                                        char **out_response_json);

/**
 * 图像生成（阻塞）
 *
 * `request_json` 为 `ImageRequest` 的 JSON 序列化字符串。
 * 成功时 `*out_response_json` 写入 `ImageResult` 的 JSON。
 *
 * # Safety
 * - `client` 必须来自 [`aibridge_client_new`]
 * - `request_json` 必须是合法 JSON C 字符串
 * - `out_response_json` 必须指向可写的 `*mut c_char` 槽位
 */
aibridge_status_t aibridge_client_image_generate(aibridge_client_t *client,
                                                 const char *request_json,
                                                 char **out_response_json);

/**
 * 获取可用模型列表（阻塞）
 *
 * `filter` 为可选的模型类型过滤器（"chat" / "image" / "video" / "audio"），
 * 可为 `nullptr` 表示不过滤。
 * 成功时 `*out_response_json` 写入 `Vec<ModelInfo>` 的 JSON 数组。
 *
 * # Safety
 * - `client` 必须来自 [`aibridge_client_new`]
 * - `filter` 可为 `nullptr` 或合法 UTF-8 C 字符串
 * - `out_response_json` 必须指向可写的 `*mut c_char` 槽位
 */
aibridge_status_t aibridge_client_list_models(aibridge_client_t *client,
                                              const char *filter,
                                              char **out_response_json);

/**
 * 获取 Provider 可用音色列表（阻塞）
 *
 * `language` 为可选的语言区域过滤（如 "zh-CN"），可为 `nullptr`。
 * 成功时 `*out_response_json` 写入 `Vec<VoiceInfo>` 的 JSON 数组。
 *
 * # Safety
 * - `client` 必须来自 [`aibridge_client_new`]
 * - `language` 可为 `nullptr` 或合法 UTF-8 C 字符串
 * - `out_response_json` 必须指向可写的 `*mut c_char` 槽位
 */
aibridge_status_t aibridge_client_list_voices(aibridge_client_t *client,
                                              const char *language,
                                              char **out_response_json);

/**
 * 创建客户端
 *
 * `provider` 为 Provider 类型（如 "openai"、"agnes"），UTF-8 C 字符串。
 * `config_json` 为 `ClientOptions` 的 JSON 序列化字符串（可为 `nullptr`，
 * 等价于默认配置）。
 *
 * 成功返回 client 指针；失败返回 `nullptr`（错误写入 `aibridge_last_error`）。
 *
 * # Safety
 * - `provider` 必须是合法的 NUL 结尾 UTF-8 C 字符串
 * - `config_json` 可为 `nullptr` 或合法 JSON C 字符串
 * - 返回的指针需通过 [`aibridge_client_destroy`] 释放
 */
aibridge_client_t *aibridge_client_new(const char *provider, const char *config_json);

/**
 * 推荐可用音色（阻塞）
 *
 * `language` 为可选的语言区域（如 "zh-CN"），`gender` 为可选的性别（"Male"/"Female"），
 * `limit` 为返回数量上限。
 * 成功时 `*out_response_json` 写入 `Vec<VoiceInfo>` 的 JSON 数组。
 *
 * # Safety
 * - `client` 必须来自 [`aibridge_client_new`]
 * - `language` / `gender` 可为 `nullptr` 或合法 UTF-8 C 字符串
 * - `out_response_json` 必须指向可写的 `*mut c_char` 槽位
 */
aibridge_status_t aibridge_client_recommend_voices(aibridge_client_t *client,
                                                   const char *language,
                                                   const char *gender,
                                                   uintptr_t limit,
                                                   char **out_response_json);

/**
 * 文字转语音（阻塞，二进制载荷走 `aibridge_bytes_t`）
 *
 * `request_json` 为 `SpeechRequest` 的 JSON 序列化字符串。
 * 成功时 `*out_audio` 写入二进制音频缓冲（`aibridge_bytes_t`），
 * `*out_meta_json` 写入 `SpeechResult`（不含 audio_data）的 JSON。
 * 两者均由调用方分别通过 [`aibridge_bytes_free`] / [`aibridge_string_free`] 释放。
 *
 * 若 Provider 仅返回 `audio_base64`/`audio_url` 而无二进制数据，
 * `*out_audio` 将为 `nullptr`（meta_json 仍写入）。
 *
 * # Safety
 * - `client` / `request_json` / `out_audio` / `out_meta_json` 均需合法非空
 */
aibridge_status_t aibridge_client_speech(aibridge_client_t *client,
                                         const char *request_json,
                                         aibridge_bytes_t **out_audio,
                                         char **out_meta_json);

/**
 * 启动客户端（初始化适配器）
 *
 * 返回 0 成功；负数为错误码（详见 `aibridge_last_error`）。
 *
 * # Safety
 * `client` 必须是 [`aibridge_client_new`] 返回的有效指针。
 */
aibridge_status_t aibridge_client_start(aibridge_client_t *client);

/**
 * 语音转文字（阻塞）
 *
 * `request_json` 为 `TranscribeRequest` 的 JSON 序列化字符串。
 * 成功时 `*out_response_json` 写入 `TranscriptionResult` 的 JSON。
 *
 * # Safety
 * - `client` 必须来自 [`aibridge_client_new`]
 * - `request_json` 必须是合法 JSON C 字符串
 * - `out_response_json` 必须指向可写的 `*mut c_char` 槽位
 */
aibridge_status_t aibridge_client_transcribe(aibridge_client_t *client,
                                             const char *request_json,
                                             char **out_response_json);

/**
 * 语音翻译（阻塞）
 *
 * `request_json` 为 `TranscribeRequest` 的 JSON 序列化字符串。
 * 成功时 `*out_response_json` 写入 `TranscriptionResult` 的 JSON。
 *
 * # Safety
 * - `client` 必须来自 [`aibridge_client_new`]
 * - `request_json` 必须是合法 JSON C 字符串
 * - `out_response_json` 必须指向可写的 `*mut c_char` 槽位
 */
aibridge_status_t aibridge_client_translate(aibridge_client_t *client,
                                            const char *request_json,
                                            char **out_response_json);

/**
 * 创建视频生成任务（阻塞）
 *
 * `request_json` 为 `VideoRequest` 的 JSON 序列化字符串。
 * 成功时 `*out_response_json` 写入 `VideoTask` 的 JSON。
 *
 * # Safety
 * - `client` 必须来自 [`aibridge_client_new`]
 * - `request_json` 必须是合法 JSON C 字符串
 * - `out_response_json` 必须指向可写的 `*mut c_char` 槽位
 */
aibridge_status_t aibridge_client_video_create(aibridge_client_t *client,
                                               const char *request_json,
                                               char **out_response_json);

/**
 * 查询视频任务状态（阻塞）
 *
 * `task_id` 为视频任务 ID（C 字符串），`model` 为模型标识（C 字符串）。
 * 成功时 `*out_response_json` 写入 `VideoStatus` 的 JSON。
 *
 * # Safety
 * - `client` 必须来自 [`aibridge_client_new`]
 * - `task_id` / `model` 必须为合法 NUL 结尾 UTF-8 C 字符串
 * - `out_response_json` 必须指向可写的 `*mut c_char` 槽位
 */
aibridge_status_t aibridge_client_video_poll(aibridge_client_t *client,
                                             const char *task_id,
                                             const char *model,
                                             char **out_response_json);

/**
 * 读取当前线程的 last_error（JSON 字符串）
 *
 * 返回指向线程局部 `CString` 内部缓冲的指针，**调用方不应释放**。
 * 若当前线程无错误，返回 `nullptr`。
 *
 * 输出格式：`{"code":"...","message":"...","details":...,"retryable":bool}`
 *
 * # Safety
 * 本函数本身无内存不安全操作（仅读取线程局部变量并返回裸指针），
 * 标 `unsafe extern "C"` 仅为符合 FFI 调用约定。调用方**不得**释放返回的指针，
 * 且需注意返回的指针仅在当前线程的下一次 FFI 调用前保证有效（thread_local 语义）。
 */
const char *aibridge_last_error(void);

/**
 * 释放 stream 句柄（触发 Rust drop → tokio task abort）
 *
 * # Safety
 * `stream` 必须来自 [`aibridge_client_chat_stream`]，且只能释放一次。
 * 传 `nullptr` 是安全的 no-op。
 */
void aibridge_stream_destroy(aibridge_stream_t *stream);

/**
 * 拉取下一个流式 chunk（阻塞）
 *
 * 返回：
 * - `0`（`AIBRIDGE_STREAM_CHUNK`）：拉到一个 chunk，`*out_chunk_json` 写入 JSON
 * - `1`（`AIBRIDGE_STREAM_END`）：流正常结束
 * - 负数：错误（`aibridge_last_error` 已写入）
 *
 * # Safety
 * - `stream` 必须来自 [`aibridge_client_chat_stream`]
 * - `out_chunk_json` 必须指向可写的 `*mut c_char` 槽位
 */
aibridge_status_t aibridge_stream_next(aibridge_stream_t *stream, char **out_chunk_json);

/**
 * 释放 Rust 分配的 C 字符串
 *
 * # Safety
 * `ptr` 必须是由 [`alloc_cstring`]（或 `CString::into_raw`）分配的指针，
 * 且只能释放一次。传 `nullptr` 是安全的 no-op。
 *
 * 对应设计文档的 `aibridge_string_free`。
 */
void aibridge_string_free(char *ptr);

#endif  /* AIBRIDGE_H */
