"""AIBridge Python 绑定 hello world

验证 PyO3 直连 aibridge-core 的端到端管线：
- chat：回显最后 user 消息 + " [echo]"（期望 "hello [echo]"）
- chat_stream：流式产 3 个 chunk
- speech：返回 15 字节固定音频

使用 echo mock 适配器，免认证，无网络依赖。
"""

import asyncio

import aibridge
from aibridge import Client


async def main() -> None:
    print(f"aibridge 版本: {aibridge.__version__}")
    print("=" * 50)

    # 创建 echo 客户端（免认证）
    client = Client(provider="echo")
    await client.start()
    print(f"已创建客户端，provider_type={client.provider_type}")

    # --- chat ---
    print("-" * 50)
    print("[chat] 调用 chat(model='echo-chat', messages=[{'role':'user','content':'hello'}])")
    resp = await client.chat(
        model="echo-chat",
        messages=[{"role": "user", "content": "hello"}],
    )
    content = resp.choices[0].message.content
    print(f"[chat] choices[0].message.content = {content!r}")
    assert content == "hello [echo]", f"期望 'hello [echo]'，实际 {content!r}"
    print("[chat] 通过：回显正确")

    # --- chat_stream ---
    print("-" * 50)
    print("[stream] 调用 chat_stream(...)，逐块消费")
    stream = await client.chat_stream(
        model="echo-chat",
        messages=[{"role": "user", "content": "hello"}],
    )
    chunks = []
    async for chunk in stream:
        delta = chunk.choices[0]
        print(
            f"[stream] chunk: index={delta.index} role={delta.role!r} "
            f"content={delta.content!r} finish_reason={delta.finish_reason!r}"
        )
        chunks.append(delta)
    assert len(chunks) == 3, f"期望 3 个 chunk，实际 {len(chunks)}"
    # 拼接第 2、3 块内容应等于完整回显
    assembled = chunks[1].content + chunks[2].content
    assert assembled == "hello [echo]", f"拼接内容 {assembled!r} 不等于 'hello [echo]'"
    assert chunks[0].role == "assistant", "第 1 块应含 role='assistant'"
    assert chunks[2].finish_reason == "stop", "第 3 块应标记 finish_reason='stop'"
    print(f"[stream] 通过：3 个 chunk，拼接 = {assembled!r}")

    # --- speech ---
    print("-" * 50)
    print("[speech] 调用 speech(model='echo-tts', input='hello', voice='alloy')")
    result = await client.speech(
        model="echo-tts",
        input="hello",
        voice="alloy",
    )
    audio = result.audio_data
    print(f"[speech] len(audio_data) = {len(audio)}")
    print(f"[speech] content_type={result.content_type} format={result.format}")
    assert len(audio) == 15, f"期望 15 字节，实际 {len(audio)}"
    print(f"[speech] 通过：音频 {len(audio)} 字节")

    # --- 关闭 ---
    print("-" * 50)
    await client.close()
    print("[close] 客户端已关闭")

    print("=" * 50)
    print("全部通过：chat 回显 + stream 3 chunk + speech 15 字节")


if __name__ == "__main__":
    asyncio.run(main())
