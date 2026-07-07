// 跨语言错误一致性探针（Go）：未知 provider 必须返回 code == "provider_not_found"。
//
// Go 绑定从 FFI last_error JSON 解析出 AibridgeError.Code()（parseErrorJSON），
// code 字段直接来自 core AibridgeError::code()。
//
// 运行：
//
//	cd bindings/go
//	DYLD_LIBRARY_PATH=../../target/debug CGO_ENABLED=1 go run ../../tests/consistency/error_probe_go.go
//
// 退出码 0 表示通过，1 表示失败。
package main

import (
	"fmt"
	"os"

	aibridge "github.com/aibridge/aibridge-go"
)

const expected = "provider_not_found"

func main() {
	// 未知 provider + 假 key（跳过 key 校验，触发 ProviderNotFound）
	// ClientOptions JSON 字段为 snake_case（core serde 默认）
	opts := `{"api_key":"dummy-key"}`
	_, err := aibridge.NewClient("nonexistent", &opts)
	if err == nil {
		fmt.Println("[go] FAIL：未抛出任何错误")
		os.Exit(1)
	}

	// 类型断言取 Code()
	ae, ok := err.(aibridge.AibridgeError)
	if !ok {
		fmt.Printf("[go] FAIL：错误不是 AibridgeError 接口，实际 %T: %v\n", err, err)
		os.Exit(1)
	}

	if ae.Code() == expected {
		fmt.Printf("[go] OK：err.Code()=%q\n", expected)
		fmt.Printf("[go]   message=%q\n", ae.Message())
		os.Exit(0)
	}
	fmt.Printf("[go] FAIL：期望 code=%q，实际 code=%q\n", expected, ae.Code())
	fmt.Printf("[go]   message=%q\n", ae.Message())
	os.Exit(1)
}
