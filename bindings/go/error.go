// Package aibridge - 错误处理
//
// FFI 错误模型：aibridge_status_t 返回码（0 成功 / 负数错误类别）+
// aibridge_last_error() 线程局部 JSON（含 code/message/details/retryable）。
//
// 本文件把 FFI 错误映射为 Go error + 类型断言接口：
//
//	var err error = ...
//	if ae, ok := err.(AibridgeError); ok {
//	    fmt.Println(ae.Code()) // "rate_limit_error" 等
//	}
package aibridge

import (
	"encoding/json"
	"fmt"
)

// AibridgeError 是 AIBridge 错误的类型断言接口
//
// 对应设计文档 9.3 节：Go 用 error + 类型断言接口 type AibridgeError interface{ Code() string }
type AibridgeError interface {
	error
	Code() string        // 错误码（如 "rate_limit_error"、"validation_error"）
	Retryable() bool     // 是否可重试
	Message() string     // 原始错误消息
}

// aibridgeErrorImpl 是 AibridgeError 的内部实现
type aibridgeErrorImpl struct {
	CodeStr   string          `json:"code"`
	Msg       string          `json:"message"`
	Details   json.RawMessage `json:"details,omitempty"`
	Retry     bool            `json:"retryable"`
}

// Error 实现 error 接口
func (e *aibridgeErrorImpl) Error() string {
	if e.Retry {
		return fmt.Sprintf("aibridge[%s] (retryable): %s", e.CodeStr, e.Msg)
	}
	return fmt.Sprintf("aibridge[%s]: %s", e.CodeStr, e.Msg)
}

// Code 返回错误码
func (e *aibridgeErrorImpl) Code() string { return e.CodeStr }

// Retryable 返回是否可重试
func (e *aibridgeErrorImpl) Retryable() bool { return e.Retry }

// Message 返回原始错误消息
func (e *aibridgeErrorImpl) Message() string { return e.Msg }

// parseErrorJSON 把 last_error 的 JSON 字符串解析为 AibridgeError
//
// 输入格式：{"code":"...","message":"...","details":...,"retryable":bool}
// 若 JSON 解析失败，回退为通用 ffi_error。
func parseErrorJSON(jsonStr string) AibridgeError {
	if jsonStr == "" {
		return &aibridgeErrorImpl{
			CodeStr: "ffi_error",
			Msg:     "未知错误（last_error 为空）",
		}
	}
	var e aibridgeErrorImpl
	if err := json.Unmarshal([]byte(jsonStr), &e); err != nil {
		// JSON 解析失败，回退
		return &aibridgeErrorImpl{
			CodeStr: "ffi_error",
			Msg:     fmt.Sprintf("last_error JSON 解析失败: %s (原始: %s)", err, jsonStr),
		}
	}
	if e.CodeStr == "" {
		e.CodeStr = "ffi_error"
	}
	return &e
}

// newFfiError 构造一个非业务类别的 FFI 层错误（如句柄为空、JSON 解析失败等）
func newFfiError(msg string) AibridgeError {
	return &aibridgeErrorImpl{
		CodeStr: "ffi_error",
		Msg:     msg,
	}
}

// 常见错误码常量（与 aibridge-core error.rs 的 AibridgeError::code() 实际返回值对齐）
const (
	errCodeAuthentication        = "authentication_error"
	errCodeRateLimit             = "rate_limit_error"
	errCodeValidation            = "validation_error"
	errCodeModelNotFound         = "model_not_found"
	errCodeAPI                   = "api_error"
	errCodeNetwork               = "network_error"
	errCodeTimeout               = "timeout_error"
	errCodeUnsupportedCapability = "unsupported_capability"
	errCodeProviderNotFound      = "provider_not_found"
	errCodeVoiceNotAvailable     = "voice_not_available"
	errCodeServiceUnavailable    = "service_unavailable"
	errCodeFFI                   = "ffi_error"
)
