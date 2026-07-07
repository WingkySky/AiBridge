'use strict';

// AIBridge Node.js 绑定 hello world
//
// 使用 echo mock 适配器（免认证）端到端验证：
// 1. chat：回显用户消息，期望 choices[0].message.content === "hello [echo]"
// 2. chatStream：流式回显，期望 3 个 chunk
// 3. speech：返回固定音频，期望 audioData.length === 15
//
// 运行：node examples/hello_node.js
// 前置：在 crates/aibridge-node 下执行 npm install && npm run build（或 cargo build -p aibridge-node）

const { Client } = require('../crates/aibridge-node');

async function main() {
  // 1. 创建客户端（echo 免认证，无需 api_key）
  const client = new Client('echo', {});
  await client.start();
  console.log('[hello_node] 客户端已启动');

  try {
    // 2. chat：非流式对话
    const resp = await client.chat({
      model: 'echo-chat',
      messages: [{ role: 'user', content: 'hello' }],
    });
    const content = resp.choices[0].message.content;
    console.log('[hello_node] chat 结果: %j', content);
    if (content !== 'hello [echo]') {
      throw new Error(`chat 断言失败：期望 "hello [echo]"，实际 "${content}"`);
    }

    // 3. chatStream：流式对话，for await 迭代 chunk
    const stream = await client.chatStream({
      model: 'echo-chat',
      messages: [{ role: 'user', content: 'hello' }],
    });
    let chunkCount = 0;
    let assembled = '';
    for await (const chunk of stream) {
      chunkCount += 1;
      const delta = chunk.choices[0]?.delta;
      if (delta?.role) {
        console.log('[hello_node] chunk %d role=%j', chunkCount, delta.role);
      }
      if (delta?.content) {
        assembled += delta.content;
        console.log('[hello_node] chunk %d content=%j', chunkCount, delta.content);
      }
    }
    console.log('[hello_node] chatStream 共 %d 个 chunk，拼接内容=%j', chunkCount, assembled);
    if (chunkCount !== 3) {
      throw new Error(`chatStream 断言失败：期望 3 个 chunk，实际 ${chunkCount}`);
    }
    if (assembled !== 'hello [echo]') {
      throw new Error(`chatStream 拼接断言失败：期望 "hello [echo]"，实际 "${assembled}"`);
    }

    // 4. speech：文字转语音，返回 Buffer
    const speech = await client.speech({
      model: 'echo-tts',
      input: 'hello',
      voice: 'alloy',
    });
    console.log(
      '[hello_node] speech 结果: audioData.length=%d, format=%j, contentType=%j',
      speech.audioData.length,
      speech.format,
      speech.contentType
    );
    if (speech.audioData.length !== 15) {
      throw new Error(
        `speech 断言失败：期望 audioData.length === 15，实际 ${speech.audioData.length}`
      );
    }
  } finally {
    // 5. 关闭客户端
    await client.close();
    console.log('[hello_node] 客户端已关闭');
  }

  console.log('[hello_node] 全部断言通过 ✓');
}

main().catch((err) => {
  console.error('[hello_node] 失败:', err);
  process.exit(1);
});
