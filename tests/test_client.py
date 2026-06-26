"""
AGN-SDK 客户端测试
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from agn import Client
from agn.core.errors import UnsupportedCapabilityError, ValidationError


class TestClient:
    """Client 类测试"""

    def test_client_init_without_api_key(self) -> None:
        """测试不提供 API Key 时抛出错误"""
        with pytest.raises(ValidationError) as exc_info:
            Client(provider="agnes", api_key="")

        assert "API key is required" in str(exc_info.value)

    def test_client_init_with_api_key(self, mock_api_key: str) -> None:
        """测试提供 API Key 时正常初始化"""
        client = Client(provider="agnes", api_key=mock_api_key)
        assert client.provider_type == "agnes"
        assert client.config.api_key == mock_api_key

    def test_client_init_free_provider_without_api_key(self) -> None:
        """测试免费 Provider（如 Edge TTS）不传 API Key 也能初始化"""
        # Edge TTS 标记 requires_api_key=False，应跳过校验
        client = Client(provider="edge-tts")
        assert client.provider_type == "edge-tts"
        # api_key 可能为 None 或空串（取决于环境变量），关键是不报错且标记为免费
        assert client._adapter.requires_api_key is False

    def test_client_init_free_provider_with_empty_api_key(self) -> None:
        """测试免费 Provider 传空 API Key 也能初始化"""
        client = Client(provider="edge-tts", api_key="")
        assert client.provider_type == "edge-tts"
        assert client._adapter.requires_api_key is False

    def test_client_init_with_base_url(self, mock_api_key: str) -> None:
        """测试自定义 Base URL"""
        custom_url = "https://custom.api.com/v1"
        client = Client(
            provider="agnes",
            api_key=mock_api_key,
            base_url=custom_url,
        )
        assert client.config.base_url == custom_url

    def test_client_init_with_timeout(self, mock_api_key: str) -> None:
        """测试自定义超时时间"""
        client = Client(
            provider="agnes",
            api_key=mock_api_key,
            timeout=600,
        )
        assert client.config.timeout == 600

    @pytest.mark.asyncio
    async def test_client_context_manager(self, mock_api_key: str) -> None:
        """测试异步上下文管理器"""
        async with Client(provider="agnes", api_key=mock_api_key) as client:
            assert client._adapter is not None


class TestClientMessages:
    """消息处理测试"""

    def test_normalize_dict_messages(self, mock_api_key: str) -> None:
        """测试字典消息转换为 ChatMessage"""
        client = Client(provider="agnes", api_key=mock_api_key)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        normalized = client._normalize_messages(messages)
        assert len(normalized) == 2
        assert normalized[0].role == "user"
        assert normalized[0].content == "Hello"
        assert normalized[1].role == "assistant"
        assert normalized[1].content == "Hi!"

    def test_normalize_empty_messages(self, mock_api_key: str) -> None:
        """测试空消息列表"""
        client = Client(provider="agnes", api_key=mock_api_key)
        normalized = client._normalize_messages([])
        assert normalized == []


class TestClientVoices:
    """Client.list_voices / recommend_voices 统一入口测试"""

    @pytest.mark.asyncio
    async def test_list_voices_passes_through_to_adapter(self) -> None:
        """测试 list_voices 透传给 adapter 并原样返回结果"""
        client = Client(provider="edge-tts")
        # 替换 adapter.list_voices 为 AsyncMock
        client._adapter.list_voices = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"ShortName": "zh-CN-XiaoxiaoNeural"}]
        )

        result = await client.list_voices(language="zh-CN")

        client._adapter.list_voices.assert_awaited_once_with(language="zh-CN")
        assert result == [{"ShortName": "zh-CN-XiaoxiaoNeural"}]

    @pytest.mark.asyncio
    async def test_recommend_voices_passes_through_to_adapter(self) -> None:
        """测试 recommend_voices 透传给 adapter 并原样返回结果"""
        client = Client(provider="edge-tts")
        client._adapter.recommend_voices = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"ShortName": "zh-CN-XiaoxiaoNeural", "Gender": "Female"}]
        )

        result = await client.recommend_voices(
            language="zh-CN", gender="female", limit=5
        )

        client._adapter.recommend_voices.assert_awaited_once_with(
            language="zh-CN", gender="female", limit=5
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_voices_propagates_unsupported(self) -> None:
        """测试 Provider 不支持 list_voices 时异常透传到 Client 层"""
        # ElevenLabs 未覆盖 list_voices，应抛 UnsupportedCapabilityError
        client = Client(provider="elevenlabs", api_key="test-key")
        with pytest.raises(UnsupportedCapabilityError):
            await client.list_voices()

    @pytest.mark.asyncio
    async def test_speech_accepts_voice_list(self) -> None:
        """测试 Client.speech 接受 voice 列表并透传给 adapter"""
        client = Client(provider="edge-tts")
        client._adapter.speech = AsyncMock(  # type: ignore[method-assign]
            return_value="speech-result"  # 简化返回值，仅验证透传
        )

        await client.speech(
            model="edge-tts",
            input="测试",
            voice=["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural"],
        )

        client._adapter.speech.assert_awaited_once()
        call_kwargs = client._adapter.speech.call_args.kwargs
        assert call_kwargs["voice"] == ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural"]

    @pytest.mark.asyncio
    async def test_speech_accepts_single_voice(self) -> None:
        """测试 Client.speech 仍兼容单 voice 字符串"""
        client = Client(provider="edge-tts")
        client._adapter.speech = AsyncMock(  # type: ignore[method-assign]
            return_value=Any  # 仅验证透传，不关心返回值
        )

        await client.speech(
            model="edge-tts",
            input="测试",
            voice="zh-CN-XiaoxiaoNeural",
        )

        client._adapter.speech.assert_awaited_once()
        call_kwargs = client._adapter.speech.call_args.kwargs
        assert call_kwargs["voice"] == "zh-CN-XiaoxiaoNeural"
