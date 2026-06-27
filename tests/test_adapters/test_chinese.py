"""
AGN-SDK 中文模型适配器测试
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agn.adapters.chinese import DoubaoAdapter, ErnieAdapter, QwenAdapter, ZhipuAdapter
from agn.models.common import ProviderConfig


class TestQwenAdapter:
    """QwenAdapter 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> QwenAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="qwen", api_key=mock_api_key)
        return QwenAdapter(config=config)

    def test_adapter_init(self, adapter: QwenAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "qwen"
        assert adapter.provider_name == "通义千问"
        assert "chat" in adapter.supported_capabilities
        assert "vision" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: QwenAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("image")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: QwenAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None

    @pytest.mark.asyncio
    async def test_image_not_supported(self, adapter: QwenAdapter) -> None:
        """测试图像生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.image_generate(
                model="test",
                prompt="A cat",
            )

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: QwenAdapter) -> None:
        """测试视频生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(
                model="test",
                prompt="A cat",
            )


class TestQwenAdapterListModels:
    """QwenAdapter 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> QwenAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="qwen", api_key=mock_api_key)
        return QwenAdapter(config=config)

    def _mock_response(self, json_data: dict) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=json_data)
        return mock_resp

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: QwenAdapter) -> None:
        """测试获取所有模型"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "qwen-turbo", "name": "Qwen Turbo"},
                {"id": "qwen-plus", "name": "Qwen Plus"},
                {"id": "qwen-max", "name": "Qwen Max"},
                {"id": "qwen-vl-max", "name": "Qwen VL Max"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models()

            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 4
            model_ids = {m.id for m in models}
            assert "qwen-turbo" in model_ids
            assert "qwen-plus" in model_ids
            assert "qwen-max" in model_ids
            assert "qwen-vl-max" in model_ids
            for model in models:
                assert model.provider == "qwen"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: QwenAdapter) -> None:
        """测试获取对话模型（按类型过滤）"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "qwen-turbo", "name": "Qwen Turbo"},
                {"id": "qwen-plus", "name": "Qwen Plus"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models(model_type="chat")

            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 2
            for model in models:
                assert model.type == "chat"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_model_capabilities_passthrough(self, adapter: QwenAdapter) -> None:
        """测试模型能力字段透传（/models 返回 capabilities 时应保留）"""
        await adapter.start()

        mock_result = {
            "data": [
                {
                    "id": "qwen-vl-max",
                    "name": "Qwen VL Max",
                    "capabilities": ["chat", "vision"],
                },
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models()
            assert len(models) == 1
            assert "chat" in models[0].capabilities
            assert "vision" in models[0].capabilities

        await adapter.close()


class TestZhipuAdapter:
    """ZhipuAdapter 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> ZhipuAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="zhipu", api_key=mock_api_key)
        return ZhipuAdapter(config=config)

    def test_adapter_init(self, adapter: ZhipuAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "zhipu"
        assert adapter.provider_name == "智谱 AI"
        assert "chat" in adapter.supported_capabilities
        assert "vision" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: ZhipuAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("image")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: ZhipuAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None

    @pytest.mark.asyncio
    async def test_image_not_supported(self, adapter: ZhipuAdapter) -> None:
        """测试图像生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.image_generate(
                model="test",
                prompt="A cat",
            )

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: ZhipuAdapter) -> None:
        """测试视频生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(
                model="test",
                prompt="A cat",
            )


class TestZhipuAdapterListModels:
    """ZhipuAdapter 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> ZhipuAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="zhipu", api_key=mock_api_key)
        return ZhipuAdapter(config=config)

    def _mock_response(self, json_data: dict) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=json_data)
        return mock_resp

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: ZhipuAdapter) -> None:
        """测试获取所有模型"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "glm-4", "name": "GLM-4"},
                {"id": "glm-4v", "name": "GLM-4V"},
                {"id": "glm-3-turbo", "name": "GLM-3 Turbo"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models()

            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 3
            model_ids = {m.id for m in models}
            assert "glm-4" in model_ids
            assert "glm-4v" in model_ids
            assert "glm-3-turbo" in model_ids
            for model in models:
                assert model.provider == "zhipu"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: ZhipuAdapter) -> None:
        """测试获取对话模型（按类型过滤）"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "glm-4", "name": "GLM-4"},
                {"id": "glm-3-turbo", "name": "GLM-3 Turbo"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models(model_type="chat")

            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 2
            for model in models:
                assert model.type == "chat"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_vision_model_capabilities(self, adapter: ZhipuAdapter) -> None:
        """测试视觉模型能力字段透传"""
        await adapter.start()

        mock_result = {
            "data": [
                {
                    "id": "glm-4v",
                    "name": "GLM-4V",
                    "capabilities": ["chat", "vision"],
                },
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models()
            glm4v = next((m for m in models if m.id == "glm-4v"), None)
            assert glm4v is not None
            assert "vision" in glm4v.capabilities

        await adapter.close()


class TestDoubaoAdapter:
    """DoubaoAdapter (豆包) 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> DoubaoAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="doubao", api_key=mock_api_key)
        return DoubaoAdapter(config=config)

    def test_adapter_init(self, adapter: DoubaoAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "doubao"
        assert adapter.provider_name == "豆包"
        assert "chat" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: DoubaoAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: DoubaoAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: DoubaoAdapter) -> None:
        """测试视频生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    def _mock_response(self, json_data: dict) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=json_data)
        return mock_resp

    @pytest.mark.asyncio
    async def test_list_models(self, adapter: DoubaoAdapter) -> None:
        """测试获取所有模型"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "doubao-pro-128k", "name": "Doubao Pro 128K"},
                {"id": "doubao-lite-4k", "name": "Doubao Lite 4K"},
                {"id": "doubao-seed-2-0-pro-260215", "name": "Doubao Seed 2.0 Pro"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models()

            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 3
            model_ids = {m.id for m in models}
            assert "doubao-pro-128k" in model_ids
            assert "doubao-lite-4k" in model_ids
            for model in models:
                assert model.provider == "doubao"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: DoubaoAdapter) -> None:
        """测试获取对话模型（按类型过滤）"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "doubao-pro-128k", "name": "Doubao Pro 128K"},
                {"id": "doubao-lite-4k", "name": "Doubao Lite 4K"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models(model_type="chat")

            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 2
            for model in models:
                assert model.type == "chat"

        await adapter.close()


class TestErnieAdapter:
    """ErnieAdapter (文心一言) 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> ErnieAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="ernie", api_key=mock_api_key)
        return ErnieAdapter(config=config)

    def test_adapter_init(self, adapter: ErnieAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "ernie"
        assert adapter.provider_name == "文心一言"
        assert "chat" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: ErnieAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: ErnieAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None

    def test_access_token_parsing(self) -> None:
        """测试 ak:sk 格式解析"""
        config = ProviderConfig(provider_type="ernie", api_key="ak123:sk456")
        adapter = ErnieAdapter(config=config)
        assert adapter.api_key == "ak123"
        assert adapter.secret_key == "sk456"

    @pytest.mark.asyncio
    async def test_list_models(self, adapter: ErnieAdapter) -> None:
        """测试获取所有模型"""
        models = await adapter.list_models()
        assert len(models) > 0
        model_ids = {m.id for m in models}
        assert "completions_pro" in model_ids
        assert "completions" in model_ids

    def test_message_conversion(self, adapter: ErnieAdapter) -> None:
        """测试消息格式转换"""
        messages = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
        ]
        converted, system = adapter._convert_messages(messages)
        assert system == "你是助手"
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
