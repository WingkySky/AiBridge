// Package aibridge 提供 AIBridge 的 Go 绑定（CGO 调 aibridge-ffi cdylib）。
//
// 设计要点（详见 docs/superpowers/specs/2026-07-07-aibridge-rust-rewrite-design.md 第 7、8 节）：
//   - 句柄式生命周期：Client 持 *C.aibridge_client_t，Close() 调 aibridge_client_destroy
//   - 复杂 struct 走 JSON 边界：Go struct <-> JSON <-> Rust serde
//   - 二进制载荷走 aibridge_bytes_t：speech 返回 []byte，必须 aibridge_bytes_free
//   - 错误：FFI 返回码 + aibridge_last_error() 线程局部 JSON（同线程立即读取）
//   - 流式：stream 句柄 + 阻塞 stream_next()，goroutine 循环拉取 push 到 channel
//
// FFI 遗留问题处理：
//  1. last_error 线程局部：每个 FFI 调用失败后，同一线程立即读 aibridge_last_error() 转存为 Go error
//  2. stream_next 串行：同一 stream 不可并发 next（goroutine 内串行循环保证）
//  3. Rust 分配的内存必须调对应的 free：string_free / bytes_free / stream_destroy / client_destroy
//  4. client/stream 句柄必须 destroy（RAII：Client.Close + stream 用完 destroy）
package aibridge

/*
#cgo CFLAGS: -I${SRCDIR}/../../crates/aibridge-ffi/include
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -laibridge -lm

#include <stdlib.h>
#include "aibridge.h"

// 错误码常量（与 aibridge.h 宏对齐，cgo 无法直接引用宏，此处通过包装函数取值）
static int aibridge_status_ok(void)              { return AIBRIDGE_OK; }
static int aibridge_status_stream_chunk(void)    { return AIBRIDGE_STREAM_CHUNK; }
static int aibridge_status_stream_end(void)      { return AIBRIDGE_STREAM_END; }

// 包装：aibridge_last_error 返回 *const c_char，转为 Go 可读的字符串前需在 C 侧拷贝长度。
// 这里直接用 Go 侧 C.GoStringPtr/GoString 读取，无需 C 包装。
*/
import "C"

import (
	"encoding/json"
	"runtime"
	"unsafe"
)

// statusOK 成功
const statusOK = 0

// statusStreamChunk 流式：拉到一个 chunk
const statusStreamChunk = 0

// statusStreamEnd 流式：流已结束
const statusStreamEnd = 1

// Client 是 AIBridge 客户端，持有一个 FFI client 句柄。
//
// 生命周期：NewClient 创建 -> Start 初始化适配器 -> Chat/ChatStream/Speech 调用 -> Close 释放。
// 必须调用 Close() 释放底层 Rust 句柄，否则内存泄漏。
type Client struct {
	ptr *C.aibridge_client_t
}

// NewClient 创建客户端（对应 aibridge_client_new）
//
// provider 为 Provider 类型（如 "echo"、"openai"）。
// optsJSON 为 ClientOptions 的 JSON 字符串，传 nil/空串表示默认配置。
//
// 失败返回 AibridgeError（如 provider_not_found、validation_error）。
func NewClient(provider string, optsJSON *string) (*Client, error) {
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))

	var cOpts *C.char
	if optsJSON != nil && *optsJSON != "" {
		cOpts = C.CString(*optsJSON)
		defer C.free(unsafe.Pointer(cOpts))
	}

	// aibridge_client_new 失败时返回 nullptr，错误写入 last_error（同线程）
	ptr := C.aibridge_client_new(cProvider, cOpts)
	if ptr == nil {
		return nil, readLastError()
	}

	client := &Client{ptr: ptr}
	// RAII：GC 时若用户忘记 Close，兜底释放（best-effort，不应依赖）
	runtime.SetFinalizer(client, func(c *Client) {
		if c.ptr != nil {
			C.aibridge_client_destroy(c.ptr)
			c.ptr = nil
		}
	})
	return client, nil
}

// Start 启动客户端（初始化适配器，对应 aibridge_client_start）
//
// 返回 0 成功；负数为错误码。
func (c *Client) Start() error {
	if c.ptr == nil {
		return newFfiError("client 句柄为空（已 Close 或未初始化）")
	}
	// 同一 goroutine 调用 + 读取 last_error，保证线程局部语义
	status := C.aibridge_client_start(c.ptr)
	if int32(status) != statusOK {
		return readLastError()
	}
	return nil
}

// Close 释放客户端句柄（对应 aibridge_client_destroy）。
//
// 必须调用，否则内存泄漏。可多次调用（第二次 no-op）。
func (c *Client) Close() {
	if c.ptr != nil {
		C.aibridge_client_destroy(c.ptr)
		c.ptr = nil
		runtime.SetFinalizer(c, nil) // 取消 finalizer，避免重复释放
	}
}

// Chat 文本对话（阻塞，对应 aibridge_client_chat）
//
// 把 req 序列化为 JSON 传入 FFI，FFI 返回 ChatCompletion 的 JSON，反序列化为 Go struct。
func (c *Client) Chat(req *ChatRequest) (*ChatCompletion, error) {
	if c.ptr == nil {
		return nil, newFfiError("client 句柄为空（已 Close 或未初始化）")
	}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, newFfiError("ChatRequest JSON 序列化失败: " + err.Error())
	}

	cReq := C.CString(string(reqJSON))
	defer C.free(unsafe.Pointer(cReq))

	var outResp *C.char
	// aibridge_client_chat 内部 block_on(async)，复杂 struct 走 JSON 边界
	status := C.aibridge_client_chat(c.ptr, cReq, &outResp)
	if int32(status) != statusOK {
		// 失败时 outResp 应为 nil，但稳妥起见仍检查释放
		if outResp != nil {
			C.aibridge_string_free(outResp)
		}
		return nil, readLastError()
	}
	defer C.aibridge_string_free(outResp) // Rust 分配的 char* 必须释放

	// C.GoString 拷贝到 Go 堆后即可安全使用
	respStr := C.GoString(outResp)
	var completion ChatCompletion
	if err := json.Unmarshal([]byte(respStr), &completion); err != nil {
		return nil, newFfiError("ChatCompletion JSON 反序列化失败: " + err.Error())
	}
	return &completion, nil
}

// Speech 文字转语音（阻塞，对应 aibridge_client_speech）
//
// 二进制音频走 aibridge_bytes_t（避免 base64 膨胀），
// meta（SpeechResult 不含 audio_data）走 JSON。
// 两者均为 Rust 分配，必须分别调 aibridge_bytes_free / aibridge_string_free。
func (c *Client) Speech(req *SpeechRequest) (*SpeechResult, error) {
	if c.ptr == nil {
		return nil, newFfiError("client 句柄为空（已 Close 或未初始化）")
	}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, newFfiError("SpeechRequest JSON 序列化失败: " + err.Error())
	}

	cReq := C.CString(string(reqJSON))
	defer C.free(unsafe.Pointer(cReq))

	var outAudio *C.aibridge_bytes_t
	var outMeta *C.char

	status := C.aibridge_client_speech(c.ptr, cReq, &outAudio, &outMeta)
	if int32(status) != statusOK {
		// 失败时仍可能分配了 audio，稳妥释放
		if outAudio != nil {
			C.aibridge_bytes_free(outAudio)
		}
		if outMeta != nil {
			C.aibridge_string_free(outMeta)
		}
		return nil, readLastError()
	}

	// meta JSON 必释放
	if outMeta != nil {
		defer C.aibridge_string_free(outMeta)
	}
	// audio bytes 必释放
	if outAudio != nil {
		defer C.aibridge_bytes_free(outAudio)
	}

	// 解析 meta JSON
	result := &SpeechResult{}
	if outMeta != nil {
		metaStr := C.GoString(outMeta)
		if err := json.Unmarshal([]byte(metaStr), result); err != nil {
			return nil, newFfiError("SpeechResult meta JSON 反序列化失败: " + err.Error())
		}
	}

	// 拷贝二进制音频数据到 Go 切片（必须在 bytes_free 之前完成拷贝）
	if outAudio != nil {
		// aibridge_bytes_t 布局：{ const uint8_t* ptr; uintptr_t len; }
		// 用 unsafe 读取 ptr 和 len
		audioBytes := (*C.aibridge_bytes_t)(unsafe.Pointer(outAudio))
		if audioBytes.ptr != nil && audioBytes.len > 0 {
			// C.GoBytes 会拷贝数据，拷贝完成后即可释放 Rust 侧内存
			result.AudioData = C.GoBytes(unsafe.Pointer(audioBytes.ptr), C.int(int64(audioBytes.len)))
		}
	}

	return result, nil
}

// readLastError 读取当前线程的 last_error 并转为 Go error
//
// 必须在 FFI 调用失败后立即在同一 goroutine 调用（线程局部语义）。
// 返回的指针仅在当前线程的下一次 FFI 调用前有效，故此函数立即拷贝。
func readLastError() error {
	errPtr := C.aibridge_last_error()
	if errPtr == nil {
		return newFfiError("FFI 调用失败但 last_error 为空")
	}
	errJSON := C.GoString(errPtr) // 立即拷贝到 Go 堆
	return parseErrorJSON(errJSON)
}
