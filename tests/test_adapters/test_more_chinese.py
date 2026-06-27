"""
AGN-SDK Kimi/MiniMax/VolcengineCV 适配器测试
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agn.adapters.chinese import KimiAdapter, MiniMaxAdapter
from agn.adapters.volcengine_cv import VolcengineCVAdapter
from agn.models.common import ProviderConfig


class TestKimiAdapter:
    """KimiAdapter 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> KimiAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="kimi", api_key=mock_api_key)
        return KimiAdapter(config=config)

    def test_adapter_init(self, adapter: KimiAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "kimi"
        assert adapter.provider_name == "Kimi (月之暗面)"
        assert "chat" in adapter.supported_capabilities
        assert "vision" in adapter.supported_capabilities
        assert "video" not in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: KimiAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("image")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: KimiAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None
            assert "Authorization" in a._http_client.headers
            assert "api.moonshot.cn" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: KimiAdapter) -> None:
        """测试视频生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    def test_message_conversion_system(self, adapter: KimiAdapter) -> None:
        """测试消息格式转换 - system 提取"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        converted, system = adapter._convert_messages(messages)
        assert system == "You are a helpful assistant."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def _mock_response(self, json_data: dict) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=json_data)
        return mock_resp

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: KimiAdapter) -> None:
        """测试获取所有模型"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "moonshot-v1-8k", "name": "Moonshot v1 8K"},
                {"id": "moonshot-v1-32k", "name": "Moonshot v1 32K"},
                {"id": "moonshot-v1-128k", "name": "Moonshot v1 128K"},
                {"id": "kimi-k2.5", "name": "Kimi K2.5"},
                {"id": "kimi-k2.6", "name": "Kimi K2.6"},
                {"id": "kimi-k2.7-code", "name": "Kimi K2.7 Code"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models()

            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 6
            model_ids = {m.id for m in models}
            assert "moonshot-v1-8k" in model_ids
            assert "moonshot-v1-128k" in model_ids
            assert "kimi-k2.5" in model_ids
            assert "kimi-k2.7-code" in model_ids
            for model in models:
                assert model.provider == "kimi"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: KimiAdapter) -> None:
        """测试获取对话模型（按类型过滤）"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "moonshot-v1-8k", "name": "Moonshot v1 8K"},
                {"id": "moonshot-v1-128k", "name": "Moonshot v1 128K"},
                {"id": "kimi-k2.5", "name": "Kimi K2.5"},
                {"id": "kimi-k2.7-code", "name": "Kimi K2.7 Code"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models(model_type="chat")

            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 4
            for model in models:
                assert model.type == "chat"

        await adapter.close()


class TestMiniMaxAdapter:
    """MiniMaxAdapter 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> MiniMaxAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="minimax", api_key=mock_api_key)
        return MiniMaxAdapter(config=config)

    def test_adapter_init(self, adapter: MiniMaxAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "minimax"
        assert adapter.provider_name == "MiniMax"
        assert "chat" in adapter.supported_capabilities
        assert "vision" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: MiniMaxAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: MiniMaxAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None
            assert "api.minimaxi.com" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: MiniMaxAdapter) -> None:
        """测试视频生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: MiniMaxAdapter) -> None:
        """测试获取所有模型"""
        models = await adapter.list_models()
        assert len(models) >= 5
        model_ids = {m.id for m in models}
        assert "abab6.5s-chat" in model_ids
        assert "MiniMax-Text-01" in model_ids
        assert "MiniMax-M1" in model_ids
        assert "MiniMax-VL-01" in model_ids

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: MiniMaxAdapter) -> None:
        """测试获取对话模型"""
        models = await adapter.list_models(model_type="chat")
        assert len(models) >= 5
        for model in models:
            assert model.type == "chat"


class TestVolcengineCVAdapter:
    """VolcengineCVAdapter (Seedream/Seedance) 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> VolcengineCVAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="volcengine_cv", api_key=mock_api_key)
        return VolcengineCVAdapter(config=config)

    def test_adapter_init(self, adapter: VolcengineCVAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "volcengine_cv"
        assert "Seedream" in adapter.provider_name
        assert "image" in adapter.supported_capabilities
        assert "video" in adapter.supported_capabilities
        assert "chat" not in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: VolcengineCVAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("image")
        assert adapter.supports_capability("video")
        assert not adapter.supports_capability("chat")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: VolcengineCVAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None
            assert "ark.cn-beijing.volces.com" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_chat_not_supported(self, adapter: VolcengineCVAdapter) -> None:
        """测试文本对话不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.chat(
                model="test",
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_video_status_mapping(self, adapter: VolcengineCVAdapter) -> None:
        """测试视频状态映射"""
        assert adapter._map_video_status("queued") == "pending"
        assert adapter._map_video_status("pending") == "pending"
        assert adapter._map_video_status("submitted") == "pending"
        assert adapter._map_video_status("processing") == "processing"
        assert adapter._map_video_status("running") == "processing"
        assert adapter._map_video_status("succeeded") == "success"
        assert adapter._map_video_status("success") == "success"
        assert adapter._map_video_status("completed") == "success"
        assert adapter._map_video_status("failed") == "failed"
        assert adapter._map_video_status("error") == "failed"
        assert adapter._map_video_status("cancelled") == "failed"
        assert adapter._map_video_status("unknown_status") == "pending"

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: VolcengineCVAdapter) -> None:
        """测试获取所有模型"""
        models = await adapter.list_models()
        assert len(models) >= 6
        model_ids = {m.id for m in models}
        assert "seedream-5.0" in model_ids
        assert "seedream-4.0" in model_ids
        assert "seedance-2.0" in model_ids
        assert "seedance-2.0-mini" in model_ids

    @pytest.mark.asyncio
    async def test_list_image_models(self, adapter: VolcengineCVAdapter) -> None:
        """测试获取图像模型"""
        models = await adapter.list_models(model_type="image")
        assert len(models) >= 3
        for model in models:
            assert model.type == "image"
            assert "text2image" in model.capabilities

    @pytest.mark.asyncio
    async def test_list_video_models(self, adapter: VolcengineCVAdapter) -> None:
        """测试获取视频模型"""
        models = await adapter.list_models(model_type="video")
        assert len(models) >= 3
        for model in models:
            assert model.type == "video"
            assert "text2video" in model.capabilities
