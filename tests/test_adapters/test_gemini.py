"""
AGN-SDK Google Gemini 适配器测试
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agn.adapters.gemini import GeminiAdapter
from agn.models.chat import ChatMessage
from agn.models.common import ProviderConfig


class TestGeminiAdapter:
    """GeminiAdapter 类测试"""

    @pytest.fixture
    def adapter_config(self, mock_api_key: str) -> ProviderConfig:
        """创建适配器配置"""
        return ProviderConfig(
            provider_type="gemini",
            api_key=mock_api_key,
        )

    @pytest.fixture
    def adapter(self, adapter_config: ProviderConfig) -> GeminiAdapter:
        """创建适配器实例"""
        return GeminiAdapter(config=adapter_config)

    def test_adapter_init(self, adapter: GeminiAdapter, mock_api_key: str) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "gemini"
        assert adapter.provider_name == "Google Gemini"
        assert "chat" in adapter.supported_capabilities
        assert "vision" in adapter.supported_capabilities
        assert "embedding" in adapter.supported_capabilities
        assert "image" not in adapter.supported_capabilities
        assert "video" not in adapter.supported_capabilities
        assert adapter.api_key == mock_api_key

    def test_adapter_supports_capability(self, adapter: GeminiAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert adapter.supports_capability("embedding")
        assert not adapter.supports_capability("image")
        assert not adapter.supports_capability("video")

    def test_adapter_supports_model_type(self, adapter: GeminiAdapter) -> None:
        """测试模型类型检查"""
        assert adapter.supports_model_type("chat")
        assert adapter.supports_model_type("embedding")

    @pytest.mark.asyncio
    async def test_adapter_start(self, adapter: GeminiAdapter) -> None:
        """测试适配器启动"""
        await adapter.start()
        assert adapter._http_client is not None
        await adapter.close()

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: GeminiAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None


class TestGeminiAdapterListModels:
    """GeminiAdapter 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> GeminiAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="gemini", api_key=mock_api_key)
        return GeminiAdapter(config=config)

    def _mock_response(self, json_data: dict) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=json_data)
        return mock_resp

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: GeminiAdapter) -> None:
        """测试获取所有模型"""
        await adapter.start()

        # Gemini /models 端点响应：模型列表在 "models" 键下，
        # id 在 "name" 字段且带 "models/" 前缀，显示名在 "displayName"
        mock_result = {
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/gemini-2.5-flash",
                    "displayName": "Gemini 2.5 Flash",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/imagen-3.0-generate-001",
                    "displayName": "Imagen 3",
                    "supportedGenerationMethods": ["predict"],
                },
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models()

            # 验证调用了 /models 端点
            mock_get.assert_called_once_with(url="/models")
            assert len(models) == 3

            # 验证 "models/" 前缀已去掉
            ids = {m.id for m in models}
            assert "gemini-2.5-pro" in ids
            assert "gemini-2.5-flash" in ids
            assert "imagen-3.0-generate-001" in ids
            # 确保前缀未被保留
            assert all(not m.id.startswith("models/") for m in models)

            # 验证 displayName 被用作 name
            names = {m.id: m.name for m in models}
            assert names["gemini-2.5-pro"] == "Gemini 2.5 Pro"
            assert names["imagen-3.0-generate-001"] == "Imagen 3"

            # 验证 provider 标识与类型推断
            assert all(m.provider == "gemini" for m in models)
            types = {m.id: m.type for m in models}
            assert types["gemini-2.5-pro"] == "chat"
            assert types["imagen-3.0-generate-001"] == "image"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: GeminiAdapter) -> None:
        """测试按类型过滤对话模型"""
        await adapter.start()

        # 同一批响应中包含 chat 与 image 类型，过滤后应仅保留 chat
        mock_result = {
            "models": [
                {"name": "models/gemini-2.5-pro", "displayName": "Gemini 2.5 Pro"},
                {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash"},
                {
                    "name": "models/imagen-3.0-generate-001",
                    "displayName": "Imagen 3",
                },
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models(model_type="chat")

            # 过滤后只剩两个对话模型，imagen 被排除
            assert len(models) == 2
            for model in models:
                assert model.type == "chat"
                assert "imagen" not in model.id
            ids = {m.id for m in models}
            assert ids == {"gemini-2.5-pro", "gemini-2.5-flash"}

        await adapter.close()

    @pytest.mark.asyncio
    async def test_image_not_supported(self, adapter: GeminiAdapter) -> None:
        """测试图像生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.image_generate(
                model="test",
                prompt="An image",
            )

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: GeminiAdapter) -> None:
        """测试视频生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(
                model="test",
                prompt="A video",
            )


class TestGeminiAdapterChat:
    """GeminiAdapter 文本对话测试（Mock HTTP）"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> GeminiAdapter:
        """创建已启动的适配器"""
        config = ProviderConfig(
            provider_type="gemini",
            api_key=mock_api_key,
        )
        return GeminiAdapter(config=config)

    def _mock_response(self, json_data: dict, status_code: int = 200) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    @pytest.mark.asyncio
    async def test_chat_basic_completion(
        self, adapter: GeminiAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试基础文本对话响应解析"""
        await adapter.start()

        mock_result = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello! How can I help you today?"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 10,
                "totalTokenCount": 25,
            },
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            result = await adapter.chat(
                model="gemini-2.5-pro",
                messages=messages,
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/models/gemini-2.5-pro:generateContent"

            body = call_args.kwargs["json"]
            assert "contents" in body
            assert len(body["contents"]) == 1
            assert body["contents"][0]["role"] == "user"
            assert body["contents"][0]["parts"][0]["text"] == "Hello!"
            assert "systemInstruction" in body
            assert (
                body["systemInstruction"]["parts"][0]["text"]
                == "You are a helpful assistant."
            )

            from agn.models.chat import ChatCompletion

            assert isinstance(result, ChatCompletion)
            assert result.model == "gemini-2.5-pro"
            assert len(result.choices) == 1
            assert result.choices[0].message.role == "assistant"
            assert (
                result.choices[0].message.content == "Hello! How can I help you today?"
            )
            assert result.choices[0].finish_reason == "stop"
            assert result.usage is not None
            assert result.usage.prompt_tokens == 15
            assert result.usage.completion_tokens == 10
            assert result.usage.total_tokens == 25

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_with_generation_config(
        self, adapter: GeminiAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试带生成配置参数的对话"""
        await adapter.start()

        mock_result = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Creative response"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8,
            },
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            await adapter.chat(
                model="gemini-2.5-pro",
                messages=messages,
                temperature=0.8,
                top_p=0.9,
                top_k=40,
                max_tokens=100,
                stop=["END", "STOP"],
            )

            call_args = mock_post.call_args
            body = call_args.kwargs["json"]
            assert "generationConfig" in body
            config = body["generationConfig"]
            assert config["temperature"] == 0.8
            assert config["topP"] == 0.9
            assert config["topK"] == 40
            assert config["maxOutputTokens"] == 100
            assert config["stopSequences"] == ["END", "STOP"]

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_finish_reason_max_tokens(
        self, adapter: GeminiAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试 finishReason 为 MAX_TOKENS 的响应解析"""
        await adapter.start()

        mock_result = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Partial response"}],
                        "role": "model",
                    },
                    "finishReason": "MAX_TOKENS",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 100,
                "totalTokenCount": 105,
            },
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(role="user", content="Hi")]
            result = await adapter.chat(
                model="gemini-2.5-pro",
                messages=messages,
            )

            assert result.choices[0].finish_reason == "length"
            assert result.choices[0].message.content == "Partial response"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_finish_reason_safety(
        self, adapter: GeminiAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试 finishReason 为 SAFETY 的响应解析"""
        await adapter.start()

        mock_result = {
            "candidates": [
                {
                    "content": {"parts": [], "role": "model"},
                    "finishReason": "SAFETY",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 0,
                "totalTokenCount": 5,
            },
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(role="user", content="Hi")]
            result = await adapter.chat(
                model="gemini-2.5-pro",
                messages=messages,
            )

            assert result.choices[0].finish_reason == "content_filter"
            assert result.choices[0].message.content == ""

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_multiple_parts(self, adapter: GeminiAdapter) -> None:
        """测试多 part 响应的拼接解析"""
        await adapter.start()

        mock_result = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Hello, "},
                            {"text": "world!"},
                            {"text": " How are you?"},
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 8,
                "totalTokenCount": 11,
            },
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(role="user", content="Hi")]
            result = await adapter.chat(
                model="gemini-2.5-pro",
                messages=messages,
            )

            assert result.choices[0].message.content == "Hello, world! How are you?"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_empty_candidates(self, adapter: GeminiAdapter) -> None:
        """测试空 candidates 响应的处理"""
        await adapter.start()

        mock_result = {
            "candidates": [],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 0,
                "totalTokenCount": 3,
            },
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(role="user", content="Hi")]
            result = await adapter.chat(
                model="gemini-2.5-pro",
                messages=messages,
            )

            assert result.choices[0].message.content == ""
            assert result.choices[0].finish_reason == "stop"

        await adapter.close()


class TestGeminiAdapterEmbed:
    """GeminiAdapter 文本嵌入测试（Mock HTTP）"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> GeminiAdapter:
        """创建已启动的适配器"""
        config = ProviderConfig(
            provider_type="gemini",
            api_key=mock_api_key,
        )
        return GeminiAdapter(config=config)

    def _mock_response(self, json_data: dict, status_code: int = 200) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    @pytest.mark.asyncio
    async def test_embed_single_text(self, adapter: GeminiAdapter) -> None:
        """测试单文本嵌入响应解析"""
        await adapter.start()

        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        mock_result = {
            "embedding": {
                "values": mock_embedding,
            }
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            from agn.models.options import EmbeddingResult

            result = await adapter.embed(
                model="text-embedding-004",
                input="Hello world",
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/models/text-embedding-004:embedContent"

            body = call_args.kwargs["json"]
            assert body["model"] == "models/text-embedding-004"
            assert body["content"]["parts"][0]["text"] == "Hello world"

            assert isinstance(result, EmbeddingResult)
            assert result.object == "list"
            assert result.model == "text-embedding-004"
            assert len(result.data) == 1
            assert result.data[0]["object"] == "embedding"
            assert result.data[0]["index"] == 0
            assert result.data[0]["embedding"] == mock_embedding

        await adapter.close()

    @pytest.mark.asyncio
    async def test_embed_single_text_with_output_dimensionality(
        self, adapter: GeminiAdapter
    ) -> None:
        """测试带 output_dimensionality 参数的单文本嵌入"""
        await adapter.start()

        mock_result = {
            "embedding": {
                "values": [0.1, 0.2, 0.3],
            }
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.embed(
                model="text-embedding-004",
                input="Hello world",
                output_dimensionality=256,
            )

            call_args = mock_post.call_args
            body = call_args.kwargs["json"]
            assert body["outputDimensionality"] == 256
            assert len(result.data[0]["embedding"]) == 3

        await adapter.close()

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, adapter: GeminiAdapter) -> None:
        """测试多文本批量嵌入响应解析"""
        await adapter.start()

        mock_result = {
            "embeddings": [
                {"values": [0.1, 0.2, 0.3]},
                {"values": [0.4, 0.5, 0.6]},
            ]
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            from agn.models.options import EmbeddingResult

            result = await adapter.embed(
                model="text-embedding-004",
                input=["text one", "text two"],
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/models/text-embedding-004:batchEmbedContents"

            body = call_args.kwargs["json"]
            assert "requests" in body
            assert len(body["requests"]) == 2
            assert body["requests"][0]["content"]["parts"][0]["text"] == "text one"
            assert body["requests"][1]["content"]["parts"][0]["text"] == "text two"

            assert isinstance(result, EmbeddingResult)
            assert len(result.data) == 2
            assert result.data[0]["index"] == 0
            assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
            assert result.data[1]["index"] == 1
            assert result.data[1]["embedding"] == [0.4, 0.5, 0.6]

        await adapter.close()

    @pytest.mark.asyncio
    async def test_embed_multiple_texts_with_output_dimensionality(
        self, adapter: GeminiAdapter
    ) -> None:
        """测试带 output_dimensionality 参数的批量嵌入"""
        await adapter.start()

        mock_result = {
            "embeddings": [
                {"values": [0.1, 0.2]},
                {"values": [0.3, 0.4]},
            ]
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.embed(
                model="text-embedding-004",
                input=["text one", "text two"],
                output_dimensionality=128,
            )

            call_args = mock_post.call_args
            body = call_args.kwargs["json"]
            assert body["requests"][0]["outputDimensionality"] == 128
            assert body["requests"][1]["outputDimensionality"] == 128
            assert len(result.data) == 2

        await adapter.close()
