package io.aibridge;

import java.util.List;

/**
 * AIBridge JVM 绑定 hello world（阶段 0.6 管线验证）。
 *
 * <p>使用 echo 适配器（免认证）端到端验证：
 * <ol>
 *   <li>{@code chat}：echo-chat 模型，回显最后一条 user 消息 + " [echo]"（期望 "hello [echo]"）</li>
 *   <li>{@code chatStream}：echo-chat 模型，3 个 chunk（role / 前半段 / 后半段+finish）</li>
 *   <li>{@code speech}：echo-tts 模型，返固定 15 字节音频</li>
 * </ol>
 *
 * <p>运行：
 * <pre>{@code
 * ./gradlew run
 * # 或：./gradlew build && java -Djava.library.path=../../target/debug -jar build/libs/aibridge-jvm-*.jar
 * }</pre>
 * 库搜索路径默认指向 {@code ../../target/debug}（见 build.gradle.kts）。
 */
public class Hello {

    public static void main(String[] args) {
        System.out.println("=== AIBridge JVM 绑定 Hello World ===");
        System.out.println("JNA 库路径: " + System.getProperty("jna.library.path", "(默认)"));
        System.out.println();

        // try-with-resources 保证 close 释放 client 句柄
        try (Client client = new Client("echo")) {
            client.start();
            System.out.println("[OK] client 创建并启动成功");

            // 1. 阻塞 chat
            testChat(client);

            // 2. 流式 chatStream
            testChatStream(client);

            // 3. 文字转语音 speech
            testSpeech(client);

            System.out.println();
            System.out.println("=== 全部测试通过 ===");
        } catch (AibridgeException e) {
            System.err.println("[FAIL] AIBridge 错误: " + e);
            System.err.println("  code=" + e.getCode() + " retryable=" + e.isRetryable());
            System.err.println("  details=" + e.getDetails());
            System.exit(1);
        } catch (Exception e) {
            System.err.println("[FAIL] 意外错误: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }

    /** 测试阻塞 chat：期望回显 "hello [echo]" */
    private static void testChat(Client client) {
        System.out.println("--- 测试 1：阻塞 chat ---");
        ChatRequest req = ChatRequest.builder(
                        "echo-chat",
                        List.of(ChatMessage.user("hello")))
                .build();
        ChatCompletion resp = client.chat(req);
        String content = resp.choices.get(0).message.content;
        System.out.println("  chat 响应 id=" + resp.id);
        System.out.println("  choices[0].message.content = \"" + content + "\"");
        if (!"hello [echo]".equals(content)) {
            throw new IllegalStateException("chat 回显不符：期望 \"hello [echo]\"，实际 \"" + content + "\"");
        }
        System.out.println("[OK] chat 回显正确");
        System.out.println();
    }

    /** 测试流式 chatStream：期望 3 个 chunk */
    private static void testChatStream(Client client) {
        System.out.println("--- 测试 2：流式 chatStream ---");
        ChatRequest req = ChatRequest.builder(
                        "echo-chat",
                        List.of(ChatMessage.user("hello")))
                .stream(true)
                .build();

        int chunkCount = 0;
        StringBuilder assembled = new StringBuilder();
        // try-with-resources 保证 close 释放 stream 句柄
        try (ChatStream stream = client.chatStream(req)) {
            while (stream.hasNext()) {
                ChatCompletionChunk chunk = stream.next();
                chunkCount++;
                String deltaContent = chunk.firstDeltaContent();
                String role = chunk.choices.isEmpty() ? null : chunk.choices.get(0).delta.role;
                String finish = chunk.choices.isEmpty() ? null : chunk.choices.get(0).finishReason;
                System.out.println("  chunk[" + chunkCount + "] role=" + role
                        + " content=" + (deltaContent == null ? "(null)" : "\"" + deltaContent + "\"")
                        + " finish=" + finish);
                if (deltaContent != null) {
                    assembled.append(deltaContent);
                }
            }
            if (stream.getError() != null) {
                throw stream.getError();
            }
        }

        System.out.println("  流式拼接内容 = \"" + assembled + "\"");
        System.out.println("  chunk 总数 = " + chunkCount);
        if (chunkCount != 3) {
            throw new IllegalStateException("chunk 数不符：期望 3，实际 " + chunkCount);
        }
        if (!"hello [echo]".equals(assembled.toString())) {
            throw new IllegalStateException("流式拼接不符：期望 \"hello [echo]\"，实际 \"" + assembled + "\"");
        }
        System.out.println("[OK] chatStream 3 个 chunk + 拼接正确");
        System.out.println();
    }

    /** 测试 speech：期望 15 字节音频 */
    private static void testSpeech(Client client) {
        System.out.println("--- 测试 3：文字转语音 speech ---");
        SpeechRequest req = SpeechRequest.builder("echo-tts", "hello", "alloy").build();
        SpeechResultFull result = client.speech(req);
        int len = result.audioLength();
        System.out.println("  audioData.length = " + len);
        System.out.println("  content_type = " + result.getMeta().contentType);
        System.out.println("  format = " + result.getMeta().format);
        if (len != 15) {
            throw new IllegalStateException("音频长度不符：期望 15，实际 " + len);
        }
        System.out.println("[OK] speech 返回 15 字节音频");
        System.out.println();
    }
}
