"""
AGN-SDK OpenAI 适配器测试
"""

import pytest

from agn.adapters.openai import OpenAIAdapter
from agn.models.common import ProviderConfig


class TestOpenAIAdapter:
    """OpenAIAdapter 类测试"""

    @pytest.fixture
    def adapter_config(self, mock_api_key: str) -> ProviderConfig:
        """创建适配器配置"""
        return ProviderConfig(
            provider_type="openai",
            api_key=mock_api_key,
        )

    @pytest.fixture
    def adapter(self, adapter_config: ProviderConfig) -> OpenAIAdapter:
        """创建适配器实例"""
        return OpenAIAdapter(config=adapter_config)

    def test_adapter_init(self, adapter: OpenAIAdapter, mock_api_key: str) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "openai"
        assert adapter.provider_name == "OpenAI"
        assert "chat" in adapter.supported_capabilities
        assert "image" in adapter.supported_capabilities
        assert "video" not in adapter.supported_capabilities
        assert adapter.api_key == mock_api_key

    def test_adapter_supports_capability(self, adapter: OpenAIAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("image")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: OpenAIAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None


class TestOpenAIAdapterListModels:
    """OpenAIAdapter 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> OpenAIAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="openai", api_key=mock_api_key)
        return OpenAIAdapter(config=config)

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: OpenAIAdapter) -> None:
        """测试获取所有模型"""
        models = await adapter.list_models()
        assert len(models) > 0

        types = {m.type for m in models}
        assert "chat" in types
        assert "image" in types

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: OpenAIAdapter) -> None:
        """测试获取对话模型"""
        models = await adapter.list_models(model_type="chat")
        for model in models:
            assert model.type == "chat"

    @pytest.mark.asyncio
    async def test_list_image_models(self, adapter: OpenAIAdapter) -> None:
        """测试获取图像模型"""
        models = await adapter.list_models(model_type="image")
        for model in models:
            assert model.type == "image"

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: OpenAIAdapter) -> None:
        """测试视频生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(
                model="test",
                prompt="A video",
            )


class TestOpenAIAdapterChatMockHTTP:
    """OpenAIAdapter 文本对话 Mock HTTP 测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> OpenAIAdapter:
        """创建适配器实例并启动"""
        config = ProviderConfig(provider_type="openai", api_key=mock_api_key)
        return OpenAIAdapter(config=config)

    @pytest.mark.asyncio
    async def test_chat_basic(
        self, adapter: OpenAIAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试基础文本对话请求和响应解析"""
        from unittest.mock import AsyncMock, MagicMock, patch

        from agn.models.chat import ChatMessage

        await adapter.start()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1699999999,
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
                "prompt_tokens": 19,
                "completion_tokens": 10,
                "total_tokens": 29,
            },
        }

        with patch.object(
            adapter._http_client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            messages = [ChatMessage(**m) for m in sample_chat_messages]
            result = await adapter.chat(model="gpt-4o", messages=messages)

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/chat/completions"
            assert call_args[1]["json"]["model"] == "gpt-4o"
            assert len(call_args[1]["json"]["messages"]) == 2
            assert call_args[1]["json"]["messages"][0]["role"] == "system"
            assert call_args[1]["json"]["messages"][1]["role"] == "user"

            assert result.id == "chatcmpl-123"
            assert result.model == "gpt-4o"
            assert len(result.choices) == 1
            assert result.choices[0].message.role == "assistant"
            assert result.choices[0].message.content == "Hello! How can I help you?"
            assert result.choices[0].finish_reason == "stop"
            assert result.usage is not None
            assert result.usage.total_tokens == 29
            assert result.usage.prompt_tokens == 19
            assert result.usage.completion_tokens == 10

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_with_parameters(
        self, adapter: OpenAIAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试带 temperature、max_tokens、stop 参数的对话请求"""
        from unittest.mock import AsyncMock, MagicMock, patch

        from agn.models.chat import ChatMessage

        await adapter.start()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1699999999,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hi there!",
                    },
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
            adapter._http_client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            messages = [ChatMessage(**m) for m in sample_chat_messages]
            result = await adapter.chat(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=100,
                stop=["END"],
            )

            mock_post.assert_called_once()
            body = mock_post.call_args[1]["json"]
            assert body["temperature"] == 0.7
            assert body["max_tokens"] == 100
            assert body["stop"] == ["END"]

            assert result.choices[0].message.content == "Hi there!"
            assert result.usage.total_tokens == 15

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_multiple_choices(
        self, adapter: OpenAIAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试多候选响应（n>1）的解析"""
        from unittest.mock import AsyncMock, MagicMock, patch

        from agn.models.chat import ChatMessage

        await adapter.start()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "id": "chatcmpl-789",
            "object": "chat.completion",
            "created": 1699999999,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response 1"},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Response 2"},
                    "finish_reason": "stop",
                },
                {
                    "index": 2,
                    "message": {"role": "assistant", "content": "Response 3"},
                    "finish_reason": "length",
                },
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 30,
                "total_tokens": 40,
            },
        }

        with patch.object(
            adapter._http_client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            messages = [ChatMessage(**m) for m in sample_chat_messages]
            result = await adapter.chat(model="gpt-4o", messages=messages, n=3)

            mock_post.assert_called_once()
            body = mock_post.call_args[1]["json"]
            assert body["n"] == 3

            assert len(result.choices) == 3
            assert result.choices[0].index == 0
            assert result.choices[0].message.content == "Response 1"
            assert result.choices[0].finish_reason == "stop"
            assert result.choices[1].index == 1
            assert result.choices[1].message.content == "Response 2"
            assert result.choices[2].index == 2
            assert result.choices[2].message.content == "Response 3"
            assert result.choices[2].finish_reason == "length"
            assert result.usage.total_tokens == 40

        await adapter.close()


class TestOpenAIAdapterImageGenerateMockHTTP:
    """OpenAIAdapter 图像生成 Mock HTTP 测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> OpenAIAdapter:
        """创建适配器实例并启动"""
        config = ProviderConfig(provider_type="openai", api_key=mock_api_key)
        return OpenAIAdapter(config=config)

    @pytest.mark.asyncio
    async def test_image_generate_url_format(
        self, adapter: OpenAIAdapter, sample_image_prompt: str
    ) -> None:
        """测试图像生成 URL 格式响应解析"""
        from unittest.mock import AsyncMock, MagicMock, patch

        await adapter.start()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "created": 1699999999,
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "revised_prompt": "A beautiful sunset over the ocean with vivid colors",
                }
            ],
        }

        with patch.object(
            adapter._http_client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            result = await adapter.image_generate(
                model="dall-e-3", prompt=sample_image_prompt
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/images/generations"
            assert call_args[1]["json"]["model"] == "dall-e-3"
            assert call_args[1]["json"]["prompt"] == sample_image_prompt

            assert len(result.data) == 1
            assert result.data[0].url == "https://example.com/image1.png"
            assert result.data[0].b64_json is None
            assert (
                result.data[0].revised_prompt
                == "A beautiful sunset over the ocean with vivid colors"
            )
            assert result.model == "dall-e-3"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_image_generate_b64_format(
        self, adapter: OpenAIAdapter, sample_image_prompt: str
    ) -> None:
        """测试图像生成 base64 格式响应解析"""
        from unittest.mock import AsyncMock, MagicMock, patch

        await adapter.start()

        mock_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "created": 1699999999,
            "data": [{"b64_json": mock_b64}],
        }

        with patch.object(
            adapter._http_client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            result = await adapter.image_generate(
                model="dall-e-3",
                prompt=sample_image_prompt,
                response_format="b64_json",
            )

            mock_post.assert_called_once()
            body = mock_post.call_args[1]["json"]
            assert body["response_format"] == "b64_json"

            assert len(result.data) == 1
            assert result.data[0].b64_json == mock_b64
            assert result.data[0].url is None

        await adapter.close()

    @pytest.mark.asyncio
    async def test_image_generate_with_parameters(
        self, adapter: OpenAIAdapter, sample_image_prompt: str
    ) -> None:
        """测试带 size、n、quality 参数的图像生成请求"""
        from unittest.mock import AsyncMock, MagicMock, patch

        await adapter.start()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "created": 1699999999,
            "data": [
                {"url": "https://example.com/image1.png"},
                {"url": "https://example.com/image2.png"},
            ],
        }

        with patch.object(
            adapter._http_client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            result = await adapter.image_generate(
                model="dall-e-3",
                prompt=sample_image_prompt,
                size="1024x1024",
                n=2,
                quality="hd",
                style="vivid",
            )

            mock_post.assert_called_once()
            body = mock_post.call_args[1]["json"]
            assert body["size"] == "1024x1024"
            assert body["n"] == 2
            assert body["quality"] == "hd"
            assert body["style"] == "vivid"

            assert len(result.data) == 2
            assert result.data[0].url == "https://example.com/image1.png"
            assert result.data[1].url == "https://example.com/image2.png"

        await adapter.close()


class TestOpenAIAdapterEmbedMockHTTP:
    """OpenAIAdapter 文本嵌入 Mock HTTP 测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> OpenAIAdapter:
        """创建适配器实例并启动"""
        config = ProviderConfig(provider_type="openai", api_key=mock_api_key)
        return OpenAIAdapter(config=config)

    @pytest.mark.asyncio
    async def test_embed_single_text(self, adapter: OpenAIAdapter) -> None:
        """测试单文本嵌入请求和响应解析"""
        from unittest.mock import AsyncMock, MagicMock, patch

        await adapter.start()

        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": mock_embedding,
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5,
            },
        }

        with patch.object(
            adapter._http_client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            result = await adapter.embed(
                model="text-embedding-3-small", input="Hello world"
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/embeddings"
            assert call_args[1]["json"]["model"] == "text-embedding-3-small"
            assert call_args[1]["json"]["input"] == "Hello world"

            assert result.object == "list"
            assert result.model == "text-embedding-3-small"
            assert len(result.data) == 1
            assert result.data[0]["embedding"] == mock_embedding
            assert result.data[0]["index"] == 0
            assert result.usage is not None
            assert result.usage["total_tokens"] == 5

        await adapter.close()

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, adapter: OpenAIAdapter) -> None:
        """测试多文本嵌入请求和响应解析"""
        from unittest.mock import AsyncMock, MagicMock, patch

        await adapter.start()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                {"object": "embedding", "index": 2, "embedding": [0.5, 0.6]},
            ],
            "model": "text-embedding-3-small",
            "usage": {
                "prompt_tokens": 15,
                "total_tokens": 15,
            },
        }

        with patch.object(
            adapter._http_client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            texts = ["Hello", "World", "Test"]
            result = await adapter.embed(model="text-embedding-3-small", input=texts)

            mock_post.assert_called_once()
            body = mock_post.call_args[1]["json"]
            assert body["input"] == texts

            assert len(result.data) == 3
            assert result.data[0]["embedding"] == [0.1, 0.2]
            assert result.data[1]["embedding"] == [0.3, 0.4]
            assert result.data[2]["embedding"] == [0.5, 0.6]
            assert result.usage["total_tokens"] == 15

        await adapter.close()

    @pytest.mark.asyncio
    async def test_embed_with_parameters(self, adapter: OpenAIAdapter) -> None:
        """测试带 dimensions、encoding_format、user 参数的嵌入请求"""
        from unittest.mock import AsyncMock, MagicMock, patch

        await adapter.start()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {
                "prompt_tokens": 3,
                "total_tokens": 3,
            },
        }

        with patch.object(
            adapter._http_client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            result = await adapter.embed(
                model="text-embedding-3-small",
                input="Hello",
                dimensions=256,
                encoding_format="float",
                user="test-user",
            )

            mock_post.assert_called_once()
            body = mock_post.call_args[1]["json"]
            assert body["dimensions"] == 256
            assert body["encoding_format"] == "float"
            assert body["user"] == "test-user"

            assert len(result.data) == 1
            assert result.usage["total_tokens"] == 3

        await adapter.close()
