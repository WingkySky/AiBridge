// AIBridge Go 绑定 - Hello World 示例
//
// 使用 echo adapter（provider="echo"，免认证）验证跨语言管线：
//   - Chat：回显最后一条 user 消息 + " [echo]"
//   - ChatStream：3 个 chunk（role / 前半段 / 后半段+finish）
//   - Speech：返回 15 字节 mock 音频（b"mock-audio-data"）
//
// 运行：
//
//	cd bindings/go
//	CGO_ENABLED=1 go run ./example
//
// 需先 cargo build -p aibridge-ffi 产出 target/debug/libaibridge.dylib。
// macOS 若报 dylib 找不到，设置：
//
//	DYLD_LIBRARY_PATH=/Users/skywing/agn-sdk/target/debug
package main

import (
	"fmt"
	"os"

	aibridge "github.com/aibridge/aibridge-go"
)

func main() {
	fmt.Println("=== AIBridge Go 绑定 Hello World ===")
	fmt.Println()

	// 1. 创建并启动 echo 客户端（免认证）
	client, err := aibridge.NewClient("echo", nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "NewClient 失败: %v\n", err)
		os.Exit(1)
	}
	defer client.Close() // RAII：确保释放句柄

	if err := client.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "Start 失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("[OK] echo 客户端已创建并启动")
	fmt.Println()

	// 2. Chat：回显 hello + " [echo]"
	fmt.Println("--- Chat ---")
	chatReq := &aibridge.ChatRequest{
		Model: "echo-chat",
		Messages: []aibridge.ChatMessage{
			aibridge.NewUserTextMessage("hello"),
		},
	}
	completion, err := client.Chat(chatReq)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Chat 失败: %v\n", err)
		os.Exit(1)
	}
	content := ""
	if len(completion.Choices) > 0 {
		content = completion.Choices[0].Message.Content
	}
	fmt.Printf("[OK] Chat 返回: id=%s, model=%s\n", completion.ID, completion.Model)
	fmt.Printf("     choices[0].message.content = %q (期望 %q)\n", content, "hello [echo]")
	if content != "hello [echo]" {
		fmt.Fprintf(os.Stderr, "     [FAIL] 内容不匹配\n")
		os.Exit(1)
	}
	fmt.Println()

	// 3. ChatStream：3 个 chunk
	fmt.Println("--- ChatStream ---")
	streamReq := &aibridge.ChatRequest{
		Model: "echo-chat",
		Messages: []aibridge.ChatMessage{
			aibridge.NewUserTextMessage("hello"),
		},
	}
	stream, err := client.ChatStream(streamReq)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ChatStream 失败: %v\n", err)
		os.Exit(1)
	}

	chunkCount := 0
	var assembledContent string
	var finishReason string
	for chunk := range stream.Ch() {
		chunkCount++
		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta
			if delta.Role != "" {
				fmt.Printf("     chunk %d: role=%q\n", chunkCount, delta.Role)
			}
			if delta.Content != "" {
				fmt.Printf("     chunk %d: content=%q\n", chunkCount, delta.Content)
				assembledContent += delta.Content
			}
			if chunk.Choices[0].FinishReason != "" {
				finishReason = chunk.Choices[0].FinishReason
				fmt.Printf("     chunk %d: finish_reason=%q\n", chunkCount, finishReason)
			}
		}
	}
	if err := stream.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "     [FAIL] 流式错误: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("[OK] ChatStream 收到 %d 个 chunk (期望 3)\n", chunkCount)
	fmt.Printf("     拼接内容 = %q (期望 %q)\n", assembledContent, "hello [echo]")
	fmt.Printf("     finish_reason = %q\n", finishReason)
	if chunkCount != 3 {
		fmt.Fprintf(os.Stderr, "     [FAIL] chunk 数量不匹配\n")
		os.Exit(1)
	}
	fmt.Println()

	// 4. Speech：15 字节 mock 音频
	fmt.Println("--- Speech ---")
	speechReq := &aibridge.SpeechRequest{
		Model: "echo-tts",
		Input: "hello",
		Voice: aibridge.SingleVoice("alloy"),
	}
	speech, err := client.Speech(speechReq)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Speech 失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("[OK] Speech 返回: model=%s, format=%s, content_type=%s\n",
		speech.Model, speech.Format, speech.ContentType)
	fmt.Printf("     len(AudioData) = %d (期望 15)\n", len(speech.AudioData))
	fmt.Printf("     AudioData = %q\n", string(speech.AudioData))
	if len(speech.AudioData) != 15 {
		fmt.Fprintf(os.Stderr, "     [FAIL] 音频字节数不匹配\n")
		os.Exit(1)
	}
	fmt.Println()

	fmt.Println("=== 全部通过 ===")
}
