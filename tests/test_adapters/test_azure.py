"""
AGN-SDK Azure 适配器测试

测试 AzureAdapter 的各项功能，包括 chat、image_generate 等。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agn.adapters.azure import AzureAdapter
from agn.models.chat import ChatCompletion, ChatMessage
from agn.models.common import ProviderConfig
from agn.models.image import ImageGenerationResult


class TestAzureAdapter:
    """AzureAdapter 类测试"""

    @pytest.fixture
    def adapter_config(self, mock_api_key: str) -> ProviderConfig:
        """创建适配器配置"""
        return ProviderConfig(
            provider_type="azure",
            api_key=mock_api_key,
            resource_name="test-resource",
            deployment_id="test-deployment",
        )

    @pytest.fixture
    def adapter(self, adapter_config: ProviderConfig) -> AzureAdapter:
        """创建适配器实例"""
        return AzureAdapter(config=adapter_config)

    def test_adapter_init(self, adapter: AzureAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "azure"
        assert adapter.provider_name == "Azure OpenAI"
        assert adapter.resource_name == "test-resource"
        assert adapter.deployment_id == "test-deployment"

    def test_adapter_base_url(self, adapter: AzureAdapter) -> None:
        """测试 Base URL 构建"""
        expected_url = (
            "https://test-resource.openai.azure.com/openai/deployments/test-deployment"
        )
        assert adapter.base_url == expected_url

    def test_adapter_init_with_base_url(self, mock_api_key: str) -> None:
        """测试使用自定义 Base URL"""
        config = ProviderConfig(
            provider_type="azure",
            api_key=mock_api_key,
            base_url="https://custom.azure.com/openai/deployments/custom",
        )
        adapter = AzureAdapter(config=config)
        assert adapter.base_url == "https://custom.azure.com/openai/deployments/custom"

    def test_adapter_init_missing_config(self, mock_api_key: str) -> None:
        """测试缺少必要配置时抛出错误"""
        config = ProviderConfig(
            provider_type="azure",
            api_key=mock_api_key,
            resource_name="test-resource",
        )
        with pytest.raises(ValueError):
            AzureAdapter(config=config)

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: AzureAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None


class TestAzureAdapterListModels:
    """AzureAdapter 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> AzureAdapter:
        """创建适配器实例"""
        config = ProviderConfig(
            provider_type="azure",
            api_key=mock_api_key,
            resource_name="test-resource",
            deployment_id="test-deployment",
        )
        return AzureAdapter(config=config)

    def _mock_response(self, json_data: dict) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=json_data)
        return mock_resp

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: AzureAdapter) -> None:
        """测试获取所有部署模型"""
        await adapter.start()

        # Azure 部署列表响应：id 是部署名，model 是底层模型名
        mock_result = {
            "data": [
                {
                    "id": "my-deployment",
                    "model": "gpt-4o",
                    "status": "succeeded",
                },
                {
                    "id": "image-deployment",
                    "model": "dall-e-3",
                    "status": "succeeded",
                },
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models()

            # 验证调用了部署列表端点并带 api-version 参数
            mock_get.assert_called_once_with(
                url="/openai/deployments",
                params={"api-version": adapter.api_version},
            )
            assert len(models) == 2
            types = {m.type for m in models}
            assert "chat" in types
            assert "image" in types

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: AzureAdapter) -> None:
        """测试按类型过滤对话模型"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "my-deployment", "model": "gpt-4o", "status": "succeeded"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models(model_type="chat")
            for model in models:
                assert model.type == "chat"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_uses_model_field_for_id(self, adapter: AzureAdapter) -> None:
        """测试使用 model 字段作为模型 ID，id 字段作为显示名"""
        await adapter.start()

        mock_result = {
            "data": [
                {"id": "my-deployment", "model": "gpt-4o", "status": "succeeded"},
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models()
            assert len(models) == 1
            # model 字段作为 id（用于推断类型）
            assert models[0].id == "gpt-4o"
            # id 字段作为显示名
            assert models[0].name == "my-deployment"

        await adapter.close()


class TestAzureAdapterChatMockHTTP:
    """AzureAdapter 文本对话 Mock HTTP 测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> AzureAdapter:
        """创建适配器实例"""
        config = ProviderConfig(
            provider_type="azure",
            api_key=mock_api_key,
            resource_name="test-resource",
            deployment_id="gpt-4o",
        )
        return AzureAdapter(config=config)

    def _mock_response(self, data: dict, status_code: int = 200) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=data)
        mock_resp.headers = {}
        return mock_resp

    @pytest.mark.asyncio
    async def test_chat_basic(
        self, adapter: AzureAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试基础文本对话请求和响应解析"""
        await adapter.start()

        mock_result = {
            "id": "chatcmpl-abc123",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            result = await adapter.chat(
                model="gpt-4o",
                messages=messages,
            )

            assert isinstance(result, ChatCompletion)
            assert result.id == "chatcmpl-abc123"
            assert result.usage.total_tokens == 18

            # 验证请求 URL
            call_args = mock_post.call_args
            assert "chat" in str(call_args).lower()

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_with_params(
        self, adapter: AzureAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试带参数的文本对话"""
        await adapter.start()

        mock_result = {
            "id": "chatcmpl-xyz789",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response with params"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            result = await adapter.chat(
                model="gpt-4o",
                messages=messages,
                temperature=0.8,
                max_tokens=100,
                top_p=0.9,
            )

            assert isinstance(result, ChatCompletion)

            # 验证请求参数
            call_args = mock_post.call_args
            body = call_args.kwargs.get("json") or call_args.args[0]
            assert body["temperature"] == 0.8
            assert body["max_tokens"] == 100
            assert body["top_p"] == 0.9

        await adapter.close()


class TestAzureAdapterImageMockHTTP:
    """AzureAdapter 图像生成 Mock HTTP 测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> AzureAdapter:
        """创建适配器实例"""
        config = ProviderConfig(
            provider_type="azure",
            api_key=mock_api_key,
            resource_name="test-resource",
            deployment_id="dall-e-3",
        )
        return AzureAdapter(config=config)

    def _mock_response(self, data: dict, status_code: int = 200) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=data)
        mock_resp.headers = {}
        return mock_resp

    @pytest.mark.asyncio
    async def test_image_generate_basic(
        self, adapter: AzureAdapter, sample_image_prompt: str
    ) -> None:
        """测试基础图像生成请求和响应解析"""
        await adapter.start()

        mock_result = {
            "created": 1700000000,
            "data": [
                {
                    "url": "https://cdn.example.com/image.png",
                    "revised_prompt": "A beautiful sunset over the ocean",
                }
            ],
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.image_generate(
                model="dall-e-3",
                prompt=sample_image_prompt,
            )

            assert isinstance(result, ImageGenerationResult)
            assert len(result.data) == 1
            assert result.data[0].url == "https://cdn.example.com/image.png"

            # 验证请求 URL
            call_args = mock_post.call_args
            assert (
                "image" in str(call_args).lower()
                or "generation" in str(call_args).lower()
            )

        await adapter.close()

    @pytest.mark.asyncio
    async def test_image_generate_with_params(
        self, adapter: AzureAdapter, sample_image_prompt: str
    ) -> None:
        """测试带参数的图像生成"""
        await adapter.start()

        mock_result = {
            "created": 1700000000,
            "data": [{"url": "https://cdn.example.com/image.png"}],
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.image_generate(
                model="dall-e-3",
                prompt=sample_image_prompt,
                size="1024x1024",
                n=1,
                quality="standard",
                style="vivid",
            )

            assert isinstance(result, ImageGenerationResult)

            # 验证请求参数
            call_args = mock_post.call_args
            body = call_args.kwargs.get("json") or call_args.args[0]
            assert body["size"] == "1024x1024"
            assert body["n"] == 1
            assert body["quality"] == "standard"
            assert body["style"] == "vivid"

        await adapter.close()
