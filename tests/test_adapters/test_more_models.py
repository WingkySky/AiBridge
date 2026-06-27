"""
AGN-SDK 更多模型适配器测试 (DeepSeek/StepFun/Mistral/Cohere/Perplexity)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agn.adapters.more_models import (
    CohereAdapter,
    DeepSeekAdapter,
    MistralAdapter,
    PerplexityAdapter,
    StepFunAdapter,
)
from agn.models.common import ProviderConfig


def _mock_models_response(json_data: dict) -> MagicMock:
    """创建模拟 HTTP 响应（用于 list_models 的 /models 端点 Mock）"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value=json_data)
    return mock_resp


class TestDeepSeekAdapter:
    """DeepSeekAdapter 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> DeepSeekAdapter:
        config = ProviderConfig(provider_type="deepseek", api_key=mock_api_key)
        return DeepSeekAdapter(config=config)

    def test_adapter_init(self, adapter: DeepSeekAdapter) -> None:
        assert adapter.provider_type == "deepseek"
        assert adapter.provider_name == "DeepSeek"
        assert "chat" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: DeepSeekAdapter) -> None:
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: DeepSeekAdapter) -> None:
        async with adapter as a:
            assert a._http_client is not None
            assert "api.deepseek.com" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: DeepSeekAdapter) -> None:
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: DeepSeekAdapter) -> None:
        await adapter.start()
        mock_result = {
            "data": [
                {"id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro"},
                {"id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash"},
                {"id": "deepseek-chat", "name": "DeepSeek Chat"},
                {"id": "deepseek-reasoner", "name": "DeepSeek Reasoner"},
                {"id": "deepseek-coder", "name": "DeepSeek Coder"},
            ]
        }
        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)
            models = await adapter.list_models()
            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 5
            model_ids = {m.id for m in models}
            assert "deepseek-v4-pro" in model_ids
            assert "deepseek-v4-flash" in model_ids
            assert "deepseek-coder" in model_ids
            for model in models:
                assert model.provider == "deepseek"
        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: DeepSeekAdapter) -> None:
        await adapter.start()
        mock_result = {
            "data": [
                {"id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro"},
                {"id": "deepseek-coder", "name": "DeepSeek Coder"},
            ]
        }
        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)
            models = await adapter.list_models(model_type="chat")
            assert len(models) == 2
            for model in models:
                assert model.type == "chat"
        await adapter.close()


class TestStepFunAdapter:
    """StepFunAdapter (阶跃星辰) 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> StepFunAdapter:
        config = ProviderConfig(provider_type="stepfun", api_key=mock_api_key)
        return StepFunAdapter(config=config)

    def test_adapter_init(self, adapter: StepFunAdapter) -> None:
        assert adapter.provider_type == "stepfun"
        assert adapter.provider_name == "阶跃星辰 StepFun"
        assert "chat" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: StepFunAdapter) -> None:
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: StepFunAdapter) -> None:
        async with adapter as a:
            assert a._http_client is not None
            assert "stepfun.com" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: StepFunAdapter) -> None:
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: StepFunAdapter) -> None:
        await adapter.start()
        mock_result = {
            "data": [
                {"id": "step-3-flash", "name": "Step 3 Flash"},
                {"id": "step-3-8k", "name": "Step 3 8K"},
                {"id": "step-3-32k", "name": "Step 3 32K"},
                {"id": "step-3-128k", "name": "Step 3 128K"},
                {"id": "step-2-mini", "name": "Step 2 Mini"},
                {"id": "step-1o-turbo", "name": "Step 1o Turbo"},
                {"id": "step-1o-mini", "name": "Step 1o Mini"},
            ]
        }
        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)
            models = await adapter.list_models()
            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 7
            model_ids = {m.id for m in models}
            assert "step-3-flash" in model_ids
            assert "step-3-128k" in model_ids
            assert "step-1o-turbo" in model_ids
            for model in models:
                assert model.provider == "stepfun"
        await adapter.close()


class TestMistralAdapter:
    """MistralAdapter 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> MistralAdapter:
        config = ProviderConfig(provider_type="mistral", api_key=mock_api_key)
        return MistralAdapter(config=config)

    def test_adapter_init(self, adapter: MistralAdapter) -> None:
        assert adapter.provider_type == "mistral"
        assert adapter.provider_name == "Mistral AI"
        assert "chat" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: MistralAdapter) -> None:
        assert adapter.supports_capability("chat")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: MistralAdapter) -> None:
        async with adapter as a:
            assert a._http_client is not None
            assert "api.mistral.ai" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: MistralAdapter) -> None:
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: MistralAdapter) -> None:
        await adapter.start()
        mock_result = {
            "data": [
                {"id": "mistral-sonnet-4-2505", "name": "Mistral Sonnet 4"},
                {"id": "mistral-nemo-2407", "name": "Mistral Nemo"},
                {"id": "mistral-small-2407", "name": "Mistral Small"},
                {"id": "mixtral-8x22b-2404", "name": "Mixtral 8x22B"},
                {"id": "mixtral-8x7b-2407", "name": "Mixtral 8x7B"},
                {"id": "codestral-2405", "name": "Codestral"},
                {"id": "mathstral-2407", "name": "Mathstral"},
            ]
        }
        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)
            models = await adapter.list_models()
            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 7
            model_ids = {m.id for m in models}
            assert "mistral-sonnet-4-2505" in model_ids
            assert "mixtral-8x22b-2404" in model_ids
            assert "codestral-2405" in model_ids
            for model in models:
                assert model.provider == "mistral"
        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: MistralAdapter) -> None:
        await adapter.start()
        mock_result = {
            "data": [
                {"id": "mistral-sonnet-4-2505", "name": "Mistral Sonnet 4"},
                {"id": "codestral-2405", "name": "Codestral"},
            ]
        }
        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)
            models = await adapter.list_models(model_type="chat")
            assert len(models) == 2
            for model in models:
                assert model.type == "chat"
        await adapter.close()


class TestCohereAdapter:
    """CohereAdapter 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> CohereAdapter:
        config = ProviderConfig(provider_type="cohere", api_key=mock_api_key)
        return CohereAdapter(config=config)

    def test_adapter_init(self, adapter: CohereAdapter) -> None:
        assert adapter.provider_type == "cohere"
        assert adapter.provider_name == "Cohere"
        assert "chat" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: CohereAdapter) -> None:
        assert adapter.supports_capability("chat")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: CohereAdapter) -> None:
        async with adapter as a:
            assert a._http_client is not None
            assert "api.cohere.ai" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: CohereAdapter) -> None:
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    def test_message_conversion_system(self, adapter: CohereAdapter) -> None:
        """测试消息格式转换 - system 消息被提取到 system_prompt"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        converted, system = adapter._convert_messages(messages)
        assert system == "You are a helpful assistant."
        # system 消息被提取到 system_prompt，不计入 converted
        assert len(converted) == 2
        # role 被转换为 Cohere 格式
        assert converted[0]["role"] == "USER"
        assert converted[0]["content"] == "Hello"
        assert converted[1]["role"] == "CHATBOT"
        assert converted[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: CohereAdapter) -> None:
        await adapter.start()
        # Cohere v1 /models 响应：模型列表在 "models" 键下，模型 ID 字段为 "name"
        mock_result = {
            "models": [
                {"name": "command-r-plus-08-2024"},
                {"name": "command-r-08-2024"},
                {"name": "command-plus"},
                {"name": "command"},
                {"name": "c4ai-aya-23-8b"},
                {"name": "c4ai-aya-23-35b"},
            ]
        }
        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)
            models = await adapter.list_models()
            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 6
            model_ids = {m.id for m in models}
            assert "command-r-plus-08-2024" in model_ids
            assert "command-r-08-2024" in model_ids
            assert "c4ai-aya-23-35b" in model_ids
            for model in models:
                assert model.provider == "cohere"
        await adapter.close()


class TestPerplexityAdapter:
    """PerplexityAdapter 类测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> PerplexityAdapter:
        config = ProviderConfig(provider_type="perplexity", api_key=mock_api_key)
        return PerplexityAdapter(config=config)

    def test_adapter_init(self, adapter: PerplexityAdapter) -> None:
        assert adapter.provider_type == "perplexity"
        assert adapter.provider_name == "Perplexity AI"
        assert "chat" in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: PerplexityAdapter) -> None:
        assert adapter.supports_capability("chat")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: PerplexityAdapter) -> None:
        async with adapter as a:
            assert a._http_client is not None
            assert "api.perplexity.ai" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: PerplexityAdapter) -> None:
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="A cat")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: PerplexityAdapter) -> None:
        await adapter.start()
        mock_result = {
            "data": [
                {"id": "sonar-pro", "name": "Sonar Pro"},
                {"id": "sonar", "name": "Sonar"},
                {"id": "sonar-pro-realtime", "name": "Sonar Pro Realtime"},
                {"id": "sonar-reasoning-pro", "name": "Sonar Reasoning Pro"},
                {"id": "sonar-reasoning", "name": "Sonar Reasoning"},
                {"id": "llama-3.1-sonar-small-128k-online", "name": "Llama Small"},
                {"id": "llama-3.1-sonar-large-128k-online", "name": "Llama Large"},
                {"id": "llama-3.1-sonar-huge-128k-online", "name": "Llama Huge"},
            ]
        }
        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _mock_models_response(mock_result)
            models = await adapter.list_models()
            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 8
            model_ids = {m.id for m in models}
            assert "sonar-pro" in model_ids
            assert "sonar" in model_ids
            assert "sonar-reasoning" in model_ids
            assert "llama-3.1-sonar-huge-128k-online" in model_ids
            for model in models:
                assert model.provider == "perplexity"
        await adapter.close()
