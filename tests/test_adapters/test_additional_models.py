"""
AGN-SDK 额外模型适配器测试 (Grok/Yi/SenseNova/Hunyuan/Groq)
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agn.adapters.additional_models import (
    GrokAdapter,
    GroqAdapter,
    HunyuanAdapter,
    SenseNovaAdapter,
    YiAdapter,
)
from agn.models.common import ProviderConfig


def _mock_models_response(json_data: dict[str, Any]) -> MagicMock:
    """创建模拟 /models HTTP 响应"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value=json_data)
    return mock_resp


class TestGrokAdapter:
    """GrokAdapter (xAI) 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> GrokAdapter:
        config = ProviderConfig(provider_type="grok", api_key=mock_api_key)
        return GrokAdapter(config=config)

    def test_adapter_init(self, adapter: GrokAdapter) -> None:
        assert adapter.provider_type == "grok"
        assert adapter.provider_name == "xAI Grok"
        assert "chat" in adapter.supported_capabilities
        assert "vision" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: GrokAdapter) -> None:
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: GrokAdapter) -> None:
        async with adapter as a:
            assert a._http_client is not None
            assert "api.x.ai" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: GrokAdapter) -> None:
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: GrokAdapter) -> None:
        """测试实时拉取模型列表（GET /models）"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "grok-3", "name": "Grok 3"},
                {"id": "grok-3-latest"},
                {"id": "grok-3-mini"},
                {"id": "grok-2"},
                {"id": "grok-2-latest"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)

            models = await adapter.list_models()

            mock_get.assert_called_once_with(url="/models")
            assert len(models) >= 5
            model_ids = {m.id for m in models}
            assert "grok-3" in model_ids
            assert "grok-3-latest" in model_ids
            assert "grok-2" in model_ids

        await adapter.close()


class TestYiAdapter:
    """YiAdapter (零一万物) 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> YiAdapter:
        config = ProviderConfig(provider_type="yi", api_key=mock_api_key)
        return YiAdapter(config=config)

    def test_adapter_init(self, adapter: YiAdapter) -> None:
        assert adapter.provider_type == "yi"
        assert adapter.provider_name == "零一万物 Yi"
        assert "chat" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: YiAdapter) -> None:
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: YiAdapter) -> None:
        async with adapter as a:
            assert a._http_client is not None
            assert "lingyiwanwu.com" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: YiAdapter) -> None:
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: YiAdapter) -> None:
        """测试实时拉取模型列表（GET /models）"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "yi-lightning"},
                {"id": "yi-medium"},
                {"id": "yi-34b-chat-0205"},
                {"id": "yi-34b-chat-200k"},
                {"id": "yi-vl-plus"},
                {"id": "yi-large"},
                {"id": "yi-large-turbo"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)

            models = await adapter.list_models()

            mock_get.assert_called_once_with(url="/models")
            assert len(models) >= 7
            model_ids = {m.id for m in models}
            assert "yi-large" in model_ids
            assert "yi-34b-chat-200k" in model_ids
            assert "yi-vl-plus" in model_ids

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: YiAdapter) -> None:
        """测试按类型过滤模型列表"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "yi-lightning"},
                {"id": "yi-medium"},
                {"id": "yi-34b-chat-0205"},
                {"id": "yi-34b-chat-200k"},
                {"id": "yi-vl-plus"},
                {"id": "yi-large"},
                {"id": "yi-large-turbo"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)

            models = await adapter.list_models(model_type="chat")

            mock_get.assert_called_once_with(url="/models")
            assert len(models) >= 7
            for model in models:
                assert model.type == "chat"

        await adapter.close()


class TestSenseNovaAdapter:
    """SenseNovaAdapter (商汤日日新) 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> SenseNovaAdapter:
        config = ProviderConfig(provider_type="sensenova", api_key=mock_api_key)
        return SenseNovaAdapter(config=config)

    def test_adapter_init(self, adapter: SenseNovaAdapter) -> None:
        assert adapter.provider_type == "sensenova"
        assert "日日新" in adapter.provider_name
        assert "chat" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: SenseNovaAdapter) -> None:
        assert adapter.supports_capability("chat")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: SenseNovaAdapter) -> None:
        async with adapter as a:
            assert a._http_client is not None
            assert "sensenova.cn" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: SenseNovaAdapter) -> None:
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: SenseNovaAdapter) -> None:
        """测试实时拉取模型列表（GET /v1/llm/models，相对路径 ../llm/models）"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "sensenova-codex-plus"},
                {"id": "sensenova-llm-v1"},
                {"id": "sensenova-llm-v2"},
                {"id": "sensenova-llm-v3"},
                {"id": "sensechat"},
                {"id": "sensechat-4.0"},
                {"id": "sensechat-5"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)

            models = await adapter.list_models()

            mock_get.assert_called_once_with(url="../llm/models")
            assert len(models) >= 7
            model_ids = {m.id for m in models}
            assert "sensechat" in model_ids
            assert "sensechat-5" in model_ids
            assert "sensenova-llm-v3" in model_ids

        await adapter.close()


class TestHunyuanAdapter:
    """HunyuanAdapter (腾讯混元) 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> HunyuanAdapter:
        config = ProviderConfig(provider_type="hunyuan", api_key=mock_api_key)
        return HunyuanAdapter(config=config)

    def test_adapter_init(self, adapter: HunyuanAdapter) -> None:
        assert adapter.provider_type == "hunyuan"
        assert "混元" in adapter.provider_name
        assert "chat" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: HunyuanAdapter) -> None:
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: HunyuanAdapter) -> None:
        async with adapter as a:
            assert a._http_client is not None
            assert "tencentcloudapi.com" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: HunyuanAdapter) -> None:
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: HunyuanAdapter) -> None:
        """测试实时拉取模型列表（GET /models）"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "hunyuan-turbo"},
                {"id": "hunyuan-latest"},
                {"id": "hunyuan-pro"},
                {"id": "hunyuan-lite"},
                {"id": "hunyuan-standard"},
                {"id": "hunyuan-vision"},
                {"id": "hunyuan-code"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)

            models = await adapter.list_models()

            mock_get.assert_called_once_with(url="/models")
            assert len(models) >= 7
            model_ids = {m.id for m in models}
            assert "hunyuan-turbo" in model_ids
            assert "hunyuan-lite" in model_ids
            assert "hunyuan-vision" in model_ids

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: HunyuanAdapter) -> None:
        """测试按类型过滤模型列表"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "hunyuan-turbo"},
                {"id": "hunyuan-latest"},
                {"id": "hunyuan-pro"},
                {"id": "hunyuan-lite"},
                {"id": "hunyuan-standard"},
                {"id": "hunyuan-vision"},
                {"id": "hunyuan-code"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)

            models = await adapter.list_models(model_type="chat")

            mock_get.assert_called_once_with(url="/models")
            assert len(models) >= 7
            for model in models:
                assert model.type == "chat"

        await adapter.close()


class TestGroqAdapter:
    """GroqAdapter 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> GroqAdapter:
        config = ProviderConfig(provider_type="groq", api_key=mock_api_key)
        return GroqAdapter(config=config)

    def test_adapter_init(self, adapter: GroqAdapter) -> None:
        assert adapter.provider_type == "groq"
        assert adapter.provider_name == "Groq"
        assert "chat" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: GroqAdapter) -> None:
        assert adapter.supports_capability("chat")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: GroqAdapter) -> None:
        async with adapter as a:
            assert a._http_client is not None
            assert "api.groq.com" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: GroqAdapter) -> None:
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: GroqAdapter) -> None:
        """测试实时拉取模型列表（GET /models，含 chat 与 audio 类型）"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "llama-3.3-70b-versatile"},
                {"id": "llama-3.1-70b-versatile"},
                {"id": "llama-3.1-8b-instant"},
                {"id": "llama3-70b-8192"},
                {"id": "llama3-8b-8192"},
                {"id": "mixtral-8x7b-32768"},
                {"id": "gemma2-9b-it"},
                {"id": "gemma-7b-it"},
                {"id": "whisper-large-v3"},
                {"id": "whisper-large-v3-turbo"},
                {"id": "distil-whisper-large-v3-en"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)

            models = await adapter.list_models()

            mock_get.assert_called_once_with(url="/models")
            assert len(models) >= 8
            model_ids = {m.id for m in models}
            assert "llama-3.3-70b-versatile" in model_ids
            assert "llama3-70b-8192" in model_ids
            assert "mixtral-8x7b-32768" in model_ids
            assert "gemma2-9b-it" in model_ids

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: GroqAdapter) -> None:
        """测试按类型过滤模型列表（whisper 等应被过滤掉）"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "llama-3.3-70b-versatile"},
                {"id": "llama-3.1-70b-versatile"},
                {"id": "llama-3.1-8b-instant"},
                {"id": "llama3-70b-8192"},
                {"id": "llama3-8b-8192"},
                {"id": "mixtral-8x7b-32768"},
                {"id": "gemma2-9b-it"},
                {"id": "gemma-7b-it"},
                {"id": "whisper-large-v3"},
                {"id": "whisper-large-v3-turbo"},
                {"id": "distil-whisper-large-v3-en"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)

            models = await adapter.list_models(model_type="chat")

            mock_get.assert_called_once_with(url="/models")
            assert len(models) >= 8
            for model in models:
                assert model.type == "chat"

        await adapter.close()
