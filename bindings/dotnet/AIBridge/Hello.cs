namespace AIBridge;

// ============================================================================
// Hello world 示例（Program Main）
//
// 验证 .NET 绑定全链路：echo 适配器免认证可用
//   1. new Client("echo") + Start
//   2. Chat(echo-chat, [user:hello]) → 打印 choices[0].message.content（期望 "hello [echo]"）
//   3. ChatStreamAsync 同上 → await foreach 打印 chunk（期望 3 个）
//   4. Speech(echo-tts, input:hello, voice:alloy) → 打印 AudioData.Length（期望 15）
//   5. Dispose
//
// 运行方式：dotnet run（需先 cargo build -p aibridge-ffi 产 libaibridge.dylib）
// ============================================================================

public static class Hello
{
    public static async Task<int> Main(string[] args)
    {
        Console.WriteLine("=== AIBridge .NET Hello World (echo adapter) ===\n");

        try
        {
            // 1. 创建 + 启动客户端（echo 免认证）
            using var client = new Client("echo");
            client.Start();
            Console.WriteLine("[OK] Client(echo) created and started.\n");

            // 2. Chat：echo 回显 +" [echo]"
            Console.WriteLine("--- Chat ---");
            var chatReq = new ChatRequest("echo-chat", new[]
            {
                ChatMessage.User("hello"),
            });
            ChatCompletion completion = client.Chat(chatReq);
            string? content = completion.Choices.Count > 0
                ? completion.Choices[0].Message.Content
                : null;
            Console.WriteLine($"  choices[0].message.content = {content ?? "(空)"}");
            Console.WriteLine($"  期望: \"hello [echo]\"，实际: \"{content}\"");
            Console.WriteLine($"  匹配: {content == "hello [echo]"}\n");

            // 3. ChatStream：echo 返 3 个 chunk
            Console.WriteLine("--- ChatStream ---");
            var streamReq = new ChatRequest("echo-chat", new[]
            {
                ChatMessage.User("hello"),
            });
            int chunkCount = 0;
            var assembled = new System.Text.StringBuilder();
            await foreach (ChatCompletionChunk chunk in client.ChatStreamAsync(streamReq))
            {
                chunkCount++;
                string? delta = chunk.Choices.Count > 0
                    ? chunk.Choices[0].Delta.Content
                    : null;
                Console.WriteLine($"  chunk #{chunkCount}: delta.content={delta ?? "(空)"}");
                if (delta != null) assembled.Append(delta);
            }
            Console.WriteLine($"  共 {chunkCount} 个 chunk，拼接结果: \"{assembled}\"");
            Console.WriteLine($"  期望 3 个 chunk，实际: {chunkCount}\n");

            // 4. Speech：echo 返 15 字节固定音频
            Console.WriteLine("--- Speech ---");
            var speechReq = new SpeechRequest("echo-tts", "hello", "alloy");
            SpeechResult speech = client.Speech(speechReq);
            Console.WriteLine($"  AudioData.Length = {speech.AudioData.Length}");
            Console.WriteLine($"  format = {speech.Format}, model = {speech.Model}");
            Console.WriteLine($"  期望 15 字节，实际: {speech.AudioData.Length}\n");

            // 5. Dispose（using 自动调）
            Console.WriteLine("[OK] Client disposed.");

            // 汇总
            Console.WriteLine("\n=== 汇总 ===");
            Console.WriteLine($"  Chat 回显匹配: {content == "hello [echo]"}");
            Console.WriteLine($"  Stream chunk 数匹配(==3): {chunkCount == 3}");
            Console.WriteLine($"  Speech 字节数匹配(==15): {speech.AudioData.Length == 15}");
            return 0;
        }
        catch (AibridgeException ex)
        {
            Console.Error.WriteLine($"[FAIL] AibridgeException: code={ex.Code}, retryable={ex.Retryable}");
            Console.Error.WriteLine($"  message: {ex.Message}");
            return 1;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[FAIL] {ex.GetType().Name}: {ex.Message}");
            Console.Error.WriteLine(ex);
            return 2;
        }
    }
}
