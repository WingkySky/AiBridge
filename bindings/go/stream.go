// Package aibridge - 流式文本对话
//
// 流式桥接策略（设计文档第 8 节）：
//   - aibridge_client_chat_stream 创建 stream 句柄
//   - goroutine 内串行循环 aibridge_stream_next（0=chunk/1=EOF/负=错）
//   - chunk 反序列化为 ChatCompletionChunk，push 到 channel
//   - EOF 或错误时关闭 channel，并 aibridge_stream_destroy 释放句柄
//
// FFI 遗留：stream_next 串行（同一 stream 不可并发 next），goroutine 内单线程循环保证。
// cancel：关闭 chan 或调用方放弃读取后，stream_destroy 仍会执行（defer），触发 Rust drop → tokio task abort。
package aibridge

/*
#cgo CFLAGS: -I${SRCDIR}/../../crates/aibridge-ffi/include

#include <stdlib.h>
#include "aibridge.h"
*/
import "C"

import (
	"encoding/json"
	"unsafe"
)

// ChatStream 表示一个流式对话句柄，封装 stream 句柄与读取 channel。
//
// 用法：
//
//	ch, err := client.ChatStream(req)
//	if err != nil { ... }
//	for chunk := range ch {
//	    fmt.Println(chunk.Choices[0].Delta.Content)
//	}
//	// channel 关闭即表示流结束（正常 EOF 或错误）
//
// 错误获取：若流以错误结束，channel 关闭后可调 ch.Err() 获取错误。
type ChatStream struct {
	ptr    *C.aibridge_stream_t
	ch     chan ChatCompletionChunk
	errCh  chan error
	closed bool
}

// ChatStream 创建流式对话（对应 aibridge_client_chat_stream）
//
// 返回的 ChatStream 已在后台 goroutine 中开始拉取 chunk，
// 调用方 for range chan 即可消费。流结束后 channel 自动关闭。
//
// 注意：流式过程中调用方放弃读取不会泄漏——goroutine 会在下次 stream_next
// 返回后检测到 channel 阻塞并最终销毁 stream；但若调用方完全不读，
// goroutine 会阻塞在 channel 发送上。建议用 context 控制或显式消费。
func (c *Client) ChatStream(req *ChatRequest) (*ChatStream, error) {
	if c.ptr == nil {
		return nil, newFfiError("client 句柄为空（已 Close 或未初始化）")
	}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, newFfiError("ChatRequest JSON 序列化失败: " + err.Error())
	}

	cReq := C.CString(string(reqJSON))
	defer C.free(unsafe.Pointer(cReq))

	var outStream *C.aibridge_stream_t
	status := C.aibridge_client_chat_stream(c.ptr, cReq, &outStream)
	if int32(status) != statusOK {
		if outStream != nil {
			C.aibridge_stream_destroy(outStream)
		}
		return nil, readLastError()
	}

	cs := &ChatStream{
		ptr:   outStream,
		ch:    make(chan ChatCompletionChunk),
		errCh: make(chan error, 1), // 缓冲 1，避免 goroutine 因无人读取错误而泄漏
	}

	// 启动后台 goroutine 串行拉取 chunk
	go cs.pullLoop()

	return cs, nil
}

// pullLoop 后台 goroutine：串行调用 aibridge_stream_next
//
// - 0 (STREAM_CHUNK)：拉到 chunk，反序列化后发送到 channel
// - 1 (STREAM_END)：流正常结束，关闭 channel
// - 负数：错误，记录到 errCh，关闭 channel
// 无论何种结束，最后都 aibridge_stream_destroy 释放句柄。
func (cs *ChatStream) pullLoop() {
	defer close(cs.ch)
	// 无论正常结束还是错误，最后都 aibridge_stream_destroy 释放句柄。
	// 注意：destroy 后不能再访问 cs.ptr，故用临时变量保存原始指针。
	streamPtr := cs.ptr
	defer func() {
		C.aibridge_stream_destroy(streamPtr)
		cs.ptr = nil
	}()

	for {
		var outChunk *C.char
		// 串行拉取（同一 stream 不可并发 next，本 goroutine 独占）
		status := C.aibridge_stream_next(streamPtr, &outChunk)

		switch int32(status) {
		case statusStreamChunk:
			// 拉到一个 chunk
			if outChunk == nil {
				cs.errCh <- newFfiError("stream_next 返回 chunk 但 out_chunk_json 为空")
				return
			}
			chunkStr := C.GoString(outChunk)
			C.aibridge_string_free(outChunk) // chunk JSON 由 Rust 分配，必须释放

			var chunk ChatCompletionChunk
			if err := json.Unmarshal([]byte(chunkStr), &chunk); err != nil {
				cs.errCh <- newFfiError("ChatCompletionChunk JSON 反序列化失败: " + err.Error())
				return
			}
			cs.ch <- chunk

		case statusStreamEnd:
			// 流正常结束
			return

		default:
			// 负数：错误
			cs.errCh <- readLastError()
			return
		}
	}
}

// Ch 返回流式 chunk 的只读 channel
func (cs *ChatStream) Ch() <-chan ChatCompletionChunk {
	return cs.ch
}

// Err 返回流式过程中的错误（若以错误结束）
//
// 必须在 channel 关闭后（for range 退出后）调用。
// 若流正常结束，返回 nil。
func (cs *ChatStream) Err() error {
	select {
	case err := <-cs.errCh:
		return err
	default:
		return nil
	}
}
