"""
AGN-SDK Anthropic/Gemini/Kling 适配器测试
"""

import pytest

from agn.adapters.anthropic import AnthropicAdapter
from agn.adapters.gemini import GeminiAdapter
from agn.adapters.kling import KlingAdapter
from agn.models.common import ProviderConfig


class TestAnthropicAdapter:
    """AnthropicAdapter 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> AnthropicAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="anthropic", api_key=mock_api_key)
        return AnthropicAdapter(config=config)

    def test_adapter_init(self, adapter: AnthropicAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "anthropic"
        assert adapter.provider_name == "Anthropic Claude"
        assert "chat" in adapter.supported_capabilities
        assert "vision" in adapter.supported_capabilities
        assert "image" not in adapter.supported_capabilities
        assert "video" not in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: AnthropicAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("image")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: AnthropicAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None
            assert "x-api-key" in a._http_client.headers
            assert "anthropic-version" in a._http_client.headers

    @pytest.mark.asyncio
    async def test_image_not_supported(self, adapter: AnthropicAdapter) -> None:
        """测试图像生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.image_generate(model="test", prompt="A cat")

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: AnthropicAdapter) -> None:
        """测试视频生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    def test_message_conversion_system(self, adapter: AnthropicAdapter) -> None:
        """测试消息格式转换 - system 提取"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        converted, system = adapter._convert_messages(messages)
        assert system == "You are a helpful assistant."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"


class TestAnthropicListModels:
    """Anthropic 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> AnthropicAdapter:
        config = ProviderConfig(provider_type="anthropic", api_key=mock_api_key)
        return AnthropicAdapter(config=config)

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: AnthropicAdapter) -> None:
        """测试获取所有模型"""
        models = await adapter.list_models()
        assert len(models) > 0
        model_ids = {m.id for m in models}
        assert "claude-3-5-sonnet-20241022" in model_ids
        assert "claude-3-opus-20240229" in model_ids

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: AnthropicAdapter) -> None:
        """测试获取对话模型"""
        models = await adapter.list_models(model_type="chat")
        assert len(models) > 0
        for model in models:
            assert model.type == "chat"
            assert "vision" in model.capabilities


class TestGeminiAdapter:
    """GeminiAdapter 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> GeminiAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="gemini", api_key=mock_api_key)
        return GeminiAdapter(config=config)

    def test_adapter_init(self, adapter: GeminiAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "gemini"
        assert adapter.provider_name == "Google Gemini"
        assert "chat" in adapter.supported_capabilities
        assert "vision" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: GeminiAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("image")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: GeminiAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None
            assert "x-goog-api-key" in a._http_client.headers

    @pytest.mark.asyncio
    async def test_image_not_supported(self, adapter: GeminiAdapter) -> None:
        """测试图像生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.image_generate(model="test", prompt="A cat")

    def test_generation_config(self, adapter: GeminiAdapter) -> None:
        """测试生成配置转换"""
        config = adapter._convert_generation_config(
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000,
            }
        )
        assert config["temperature"] == 0.7
        assert config["topP"] == 0.9
        assert config["maxOutputTokens"] == 1000


class TestGeminiListModels:
    """Gemini 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> GeminiAdapter:
        config = ProviderConfig(provider_type="gemini", api_key=mock_api_key)
        return GeminiAdapter(config=config)

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: GeminiAdapter) -> None:
        """测试获取所有模型"""
        models = await adapter.list_models()
        assert len(models) > 0
        model_ids = {m.id for m in models}
        assert "gemini-2.5-pro" in model_ids
        assert "gemini-1.5-flash" in model_ids

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: GeminiAdapter) -> None:
        """测试获取对话模型"""
        models = await adapter.list_models(model_type="chat")
        assert len(models) > 0
        for model in models:
            assert model.type == "chat"


class TestKlingAdapter:
    """KlingAdapter 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> KlingAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="kling", api_key=mock_api_key)
        return KlingAdapter(config=config)

    def test_adapter_init(self, adapter: KlingAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "kling"
        assert adapter.provider_name == "可灵 Kling"
        assert "video" in adapter.supported_capabilities
        assert "chat" not in adapter.supported_capabilities
        assert "image" not in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: KlingAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("video")
        assert not adapter.supports_capability("chat")
        assert not adapter.supports_capability("image")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: KlingAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None

    @pytest.mark.asyncio
    async def test_chat_not_supported(self, adapter: KlingAdapter) -> None:
        """测试文本对话不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.chat(
                model="test",
                messages=[{"role": "user", "content": "Hello"}],
            )


class TestKlingStatusMapping:
    """Kling 状态映射测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> KlingAdapter:
        config = ProviderConfig(provider_type="kling", api_key=mock_api_key)
        return KlingAdapter(config=config)

    def test_map_pending_status(self, adapter: KlingAdapter) -> None:
        """测试 pending 状态映射"""
        assert adapter._map_kling_status("submitted") == "pending"
        assert adapter._map_kling_status("queued") == "pending"

    def test_map_processing_status(self, adapter: KlingAdapter) -> None:
        """测试 processing 状态映射"""
        assert adapter._map_kling_status("processing") == "processing"

    def test_map_success_status(self, adapter: KlingAdapter) -> None:
        """测试 success 状态映射"""
        assert adapter._map_kling_status("succeed") == "success"
        assert adapter._map_kling_status("success") == "success"

    def test_map_failed_status(self, adapter: KlingAdapter) -> None:
        """测试 failed 状态映射"""
        assert adapter._map_kling_status("failed") == "failed"
        assert adapter._map_kling_status("error") == "failed"

    def test_map_unknown_status(self, adapter: KlingAdapter) -> None:
        """测试未知状态映射"""
        assert adapter._map_kling_status("unknown") == "pending"


class TestKlingListModels:
    """Kling 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> KlingAdapter:
        config = ProviderConfig(provider_type="kling", api_key=mock_api_key)
        return KlingAdapter(config=config)

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: KlingAdapter) -> None:
        """测试获取所有模型"""
        models = await adapter.list_models()
        assert len(models) == 3
        model_ids = {m.id for m in models}
        assert "kling-v1" in model_ids
        assert "kling-v1-5" in model_ids
        assert "kling-v2" in model_ids

    @pytest.mark.asyncio
    async def test_list_video_models(self, adapter: KlingAdapter) -> None:
        """测试获取视频模型"""
        models = await adapter.list_models(model_type="video")
        assert len(models) == 3
        for model in models:
            assert model.type == "video"
            assert "text2video" in model.capabilities
            assert "image2video" in model.capabilities
