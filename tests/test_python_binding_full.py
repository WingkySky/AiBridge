"""AIBridge Python 绑定 · 全能力端到端验证（list_voices / recommend_voices / Router）

验证此前文档标记为"未暴露"的能力实际可用（echo mock 免认证、无网络依赖）：
- list_voices：返回 echo 固定音色列表，字段完整
- recommend_voices：按性别过滤 + 数量上限
- Router：first 策略路由到 echo provider 并成功完成 chat
"""

import pytest
from aibridge import Client, Router, VoiceInfo


# ---------------------------------------------------------------------------
# list_voices：echo 固定音色列表（2 个：zh-CN Female / en-US Male）
# ---------------------------------------------------------------------------


async def test_list_voices_returns_two():
    """echo adapter 的 list_voices 返回 2 个固定音色，字段映射正确。"""
    client = Client(provider="echo")
    await client.start()
    try:
        voices = await client.list_voices()
        assert len(voices) == 2
        assert all(isinstance(v, VoiceInfo) for v in voices)

        first = voices[0]
        assert first.short_name == "echo-voice-1"
        assert first.locale == "zh-CN"
        assert first.gender == "Female"

        second = voices[1]
        assert second.short_name == "echo-voice-2"
        assert second.locale == "en-US"
        assert second.gender == "Male"
    finally:
        await client.close()


async def test_list_voices_repr():
    """VoiceInfo.__repr__ 含关键信息。"""
    client = Client(provider="echo")
    await client.start()
    try:
        voices = await client.list_voices()
        assert "echo-voice-1" in repr(voices[0])
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# recommend_voices：按性别过滤 + limit（core 默认实现）
# ---------------------------------------------------------------------------


async def test_recommend_voices_filter_gender():
    """按性别过滤：Female 只剩 echo-voice-1，Male 只剩 echo-voice-2。"""
    client = Client(provider="echo")
    await client.start()
    try:
        females = await client.recommend_voices(None, "Female", 10)
        assert len(females) == 1
        assert females[0].short_name == "echo-voice-1"

        males = await client.recommend_voices(None, "Male", 10)
        assert len(males) == 1
        assert males[0].short_name == "echo-voice-2"
    finally:
        await client.close()


async def test_recommend_voices_limit():
    """limit=1 时只返回 1 条。"""
    client = Client(provider="echo")
    await client.start()
    try:
        voices = await client.recommend_voices(None, None, 1)
        assert len(voices) == 1
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Router：first 策略路由到 echo provider 并完成 chat
# ---------------------------------------------------------------------------


async def test_router_chat_with_echo():
    """Router(first 策略) 注册 echo 后可正常路由 chat。"""
    router = Router(strategy="first", enable_fallback=True, max_retries=2)
    router.add_provider("echo")
    await router.start()
    try:
        resp = await router.chat(
            "echo-chat",
            [{"role": "user", "content": "hello router"}],
        )
        assert resp.model == "echo-chat"  # echo 回显请求的 model 名
        assert len(resp.choices) == 1
        assert "hello router" in resp.choices[0].message.content
    finally:
        await router.close()
