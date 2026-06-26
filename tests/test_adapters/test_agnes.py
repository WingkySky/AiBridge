"""
AGN-SDK Agnes 适配器测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agn.adapters.agnes import AgnesAdapter
from agn.models.chat import ChatMessage
from agn.models.common import ProviderConfig


class TestAgnesAdapter:
    """AgnesAdapter 类测试"""

    @pytest.fixture
    def adapter_config(self, mock_api_key: str) -> ProviderConfig:
        """创建适配器配置"""
        return ProviderConfig(
            provider_type="agnes",
            api_key=mock_api_key,
            base_url="https://api.test.agnes.ai/v1",
        )

    @pytest.fixture
    def adapter(self, adapter_config: ProviderConfig) -> AgnesAdapter:
        """创建适配器实例"""
        return AgnesAdapter(config=adapter_config)

    def test_adapter_init(self, adapter: AgnesAdapter, mock_api_key: str) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "agnes"
        assert adapter.provider_name == "Agnes AI"
        assert "chat" in adapter.supported_capabilities
        assert "image" in adapter.supported_capabilities
        assert "video" in adapter.supported_capabilities
        assert adapter.api_key == mock_api_key

    def test_adapter_supports_capability(self, adapter: AgnesAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("image")
        assert adapter.supports_capability("video")
        assert not adapter.supports_capability("audio")

    def test_adapter_supports_model_type(self, adapter: AgnesAdapter) -> None:
        """测试模型类型检查"""
        assert adapter.supports_model_type("chat")
        assert adapter.supports_model_type("image")
        assert adapter.supports_model_type("video")

    @pytest.mark.asyncio
    async def test_adapter_start(self, adapter: AgnesAdapter) -> None:
        """测试适配器启动"""
        await adapter.start()
        assert adapter._http_client is not None
        await adapter.close()

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: AgnesAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None


class TestAgnesAdapterListModels:
    """AgnesAdapter 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> AgnesAdapter:
        """创建适配器实例"""
        config = ProviderConfig(
            provider_type="agnes",
            api_key=mock_api_key,
        )
        return AgnesAdapter(config=config)

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: AgnesAdapter) -> None:
        """测试获取所有模型"""
        models = await adapter.list_models()
        assert len(models) > 0

        # 检查模型类型覆盖
        types = {m.type for m in models}
        assert "chat" in types
        assert "image" in types
        assert "video" in types

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: AgnesAdapter) -> None:
        """测试获取对话模型"""
        models = await adapter.list_models(model_type="chat")
        for model in models:
            assert model.type == "chat"

    @pytest.mark.asyncio
    async def test_list_image_models(self, adapter: AgnesAdapter) -> None:
        """测试获取图像模型"""
        models = await adapter.list_models(model_type="image")
        for model in models:
            assert model.type == "image"

    @pytest.mark.asyncio
    async def test_list_video_models(self, adapter: AgnesAdapter) -> None:
        """测试获取视频模型"""
        models = await adapter.list_models(model_type="video")
        for model in models:
            assert model.type == "video"


class TestAgnesAdapterChat:
    """AgnesAdapter 文本对话测试（Mock HTTP）"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> AgnesAdapter:
        """创建已启动的适配器"""
        config = ProviderConfig(
            provider_type="agnes",
            api_key=mock_api_key,
            base_url="https://api.test.agnes.ai/v1",
        )
        return AgnesAdapter(config=config)

    def _mock_response(self, json_data: dict, status_code: int = 200) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    @pytest.mark.asyncio
    async def test_chat_basic_completion(
        self, adapter: AgnesAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试基础文本对话"""
        await adapter.start()

        mock_result = {
            "id": "chatcmpl-abc123",
            "created": 1700000000,
            "model": "claude-3-sonnet",
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
                model="claude-3-sonnet",
                messages=messages,
            )

            # 验证请求
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs.kwargs["url"] == "/chat/completions"
            assert call_kwargs.kwargs["json"]["model"] == "claude-3-sonnet"
            assert len(call_kwargs.kwargs["json"]["messages"]) == 2

            # 验证响应
            from agn.models.chat import ChatCompletion

            assert isinstance(result, ChatCompletion)
            assert result.id == "chatcmpl-abc123"
            assert result.model == "claude-3-sonnet"
            assert len(result.choices) == 1
            assert result.choices[0].message.role == "assistant"
            assert result.choices[0].message.content == "Hello! How can I help you?"
            assert result.choices[0].finish_reason == "stop"
            assert result.usage is not None
            assert result.usage.total_tokens == 18

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_with_temperature_and_max_tokens(
        self, adapter: AgnesAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试带 temperature 和 max_tokens 的对话"""
        await adapter.start()

        mock_result = {
            "id": "chatcmpl-temp",
            "created": 1700000000,
            "model": "claude-3-sonnet",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Creative response"},
                    "finish_reason": "stop",
                }
            ],
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            await adapter.chat(
                model="claude-3-sonnet",
                messages=messages,
                temperature=0.8,
                max_tokens=100,
                stop=["END"],
            )

            # 验证参数正确传递
            call_kwargs = mock_post.call_args
            body = call_kwargs.kwargs["json"]
            assert body["temperature"] == 0.8
            assert body["max_tokens"] == 100
            assert body["stop"] == ["END"]

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_multiple_choices(
        self, adapter: AgnesAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试多候选响应"""
        await adapter.start()

        mock_result = {
            "id": "chatcmpl-multi",
            "created": 1700000000,
            "model": "claude-3-sonnet",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response 1"},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Response 2"},
                    "finish_reason": "length",
                },
            ],
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            result = await adapter.chat(model="claude-3-sonnet", messages=messages)

            assert len(result.choices) == 2
            assert result.choices[0].index == 0
            assert result.choices[0].message.content == "Response 1"
            assert result.choices[1].index == 1
            assert result.choices[1].message.content == "Response 2"
            assert result.choices[1].finish_reason == "length"

        await adapter.close()


class TestAgnesAdapterImage:
    """AgnesAdapter 图像生成测试（Mock HTTP）"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> AgnesAdapter:
        """创建已启动的适配器"""
        config = ProviderConfig(
            provider_type="agnes",
            api_key=mock_api_key,
            base_url="https://api.test.agnes.ai/v1",
        )
        return AgnesAdapter(config=config)

    def _mock_response(self, json_data: dict, status_code: int = 200) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    @pytest.mark.asyncio
    async def test_image_generate_url(
        self, adapter: AgnesAdapter, sample_image_prompt: str
    ) -> None:
        """测试图像生成（URL 格式）"""
        await adapter.start()

        mock_result = {
            "id": "img-abc123",
            "created": 1700000000,
            "model": "image-model-v1",
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "revised_prompt": "A beautiful sunset over the ocean, vibrant colors",
                },
                {
                    "url": "https://example.com/image2.png",
                    "revised_prompt": "A beautiful sunset over the ocean, warm tones",
                },
            ],
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            from agn.models.image import ImageGenerationResult

            result = await adapter.image_generate(
                model="image-model-v1",
                prompt=sample_image_prompt,
                size="1024x1024",
                n=2,
            )

            # 验证请求
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs.kwargs["url"] == "/images/generations"
            assert call_kwargs.kwargs["json"]["model"] == "image-model-v1"
            assert call_kwargs.kwargs["json"]["prompt"] == sample_image_prompt
            assert call_kwargs.kwargs["json"]["size"] == "1024x1024"
            assert call_kwargs.kwargs["json"]["n"] == 2

            # 验证响应
            assert isinstance(result, ImageGenerationResult)
            assert result.id == "img-abc123"
            assert result.model == "image-model-v1"
            assert len(result.data) == 2
            assert result.data[0].url == "https://example.com/image1.png"
            assert "sunset" in result.data[0].revised_prompt
            assert result.data[1].url == "https://example.com/image2.png"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_image_generate_b64(
        self, adapter: AgnesAdapter, sample_image_prompt: str
    ) -> None:
        """测试图像生成（base64 格式）"""
        await adapter.start()

        mock_result = {
            "id": "img-b64",
            "created": 1700000000,
            "model": "image-model-v1",
            "data": [
                {
                    "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                }
            ],
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.image_generate(
                model="image-model-v1",
                prompt=sample_image_prompt,
                response_format="b64_json",
            )

            assert len(result.data) == 1
            assert result.data[0].b64_json is not None
            assert result.data[0].url is None

        await adapter.close()

    @pytest.mark.asyncio
    async def test_image_generate_with_negative_prompt(
        self, adapter: AgnesAdapter, sample_image_prompt: str
    ) -> None:
        """测试带负面提示词的图像生成"""
        await adapter.start()

        mock_result = {
            "id": "img-neg",
            "created": 1700000000,
            "model": "image-model-v1",
            "data": [{"url": "https://example.com/image.png"}],
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            await adapter.image_generate(
                model="image-model-v1",
                prompt=sample_image_prompt,
                negative_prompt="blurry, low quality",
            )

            call_kwargs = mock_post.call_args
            body = call_kwargs.kwargs["json"]
            assert body["negative_prompt"] == "blurry, low quality"

        await adapter.close()


class TestAgnesAdapterVideo:
    """AgnesAdapter 视频生成测试（Mock HTTP）"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> AgnesAdapter:
        """创建已启动的适配器"""
        config = ProviderConfig(
            provider_type="agnes",
            api_key=mock_api_key,
            base_url="https://api.test.agnes.ai/v1",
        )
        return AgnesAdapter(config=config)

    def _mock_response(self, json_data: dict, status_code: int = 200) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    @pytest.mark.asyncio
    async def test_video_create(
        self, adapter: AgnesAdapter, sample_video_prompt: str
    ) -> None:
        """测试创建视频生成任务"""
        await adapter.start()

        mock_result = {
            "id": "vid-abc123",
            "model": "video-model-v1",
            "status": "pending",
            "created": 1700000000,
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            from agn.models.video import VideoTask

            result = await adapter.video_create(
                model="video-model-v1",
                prompt=sample_video_prompt,
                width=768,
                height=512,
            )

            # 验证请求
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs.kwargs["url"] == "/videos/generations"
            assert call_kwargs.kwargs["json"]["model"] == "video-model-v1"
            assert call_kwargs.kwargs["json"]["prompt"] == sample_video_prompt
            assert call_kwargs.kwargs["json"]["width"] == 768
            assert call_kwargs.kwargs["json"]["height"] == 512

            # 验证响应
            assert isinstance(result, VideoTask)
            assert result.task_id == "vid-abc123"
            assert result.model == "video-model-v1"
            assert result.status == "pending"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_poll_pending(self, adapter: AgnesAdapter) -> None:
        """测试查询视频状态（处理中）"""
        await adapter.start()

        mock_result = {
            "status": "processing",
            "progress": 50,
            "created": 1700000000,
            "updated": 1700000010,
        }

        # video_poll 内部创建新的 AsyncHttpClient，需要 mock 整个类
        mock_client_instance = MagicMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.close = AsyncMock()
        mock_client_instance.get = AsyncMock(
            return_value=self._mock_response(mock_result)
        )

        with patch(
            "agn.adapters.agnes.AsyncHttpClient", return_value=mock_client_instance
        ):
            from agn.models.video import VideoStatus

            result = await adapter.video_poll(
                task_id="vid-abc123",
                model="video-model-v1",
            )

            # 验证请求
            mock_client_instance.get.assert_called_once()
            call_kwargs = mock_client_instance.get.call_args
            assert "/videos/generations/vid-abc123" in call_kwargs.kwargs["url"]

            # 验证响应
            assert isinstance(result, VideoStatus)
            assert result.task_id == "vid-abc123"
            assert result.status == "processing"
            assert result.progress == 50
            assert result.video_url is None
            assert result.error is None

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_poll_completed(self, adapter: AgnesAdapter) -> None:
        """测试查询视频状态（已完成）"""
        await adapter.start()

        mock_result = {
            "status": "success",
            "progress": 100,
            "video_url": "https://example.com/output.mp4",
            "created": 1700000000,
            "updated": 1700000100,
        }

        mock_client_instance = MagicMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.close = AsyncMock()
        mock_client_instance.get = AsyncMock(
            return_value=self._mock_response(mock_result)
        )

        with patch(
            "agn.adapters.agnes.AsyncHttpClient", return_value=mock_client_instance
        ):
            result = await adapter.video_poll(task_id="vid-abc123")

            assert result.status == "success"
            assert result.progress == 100
            assert result.video_url == "https://example.com/output.mp4"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_poll_failed(self, adapter: AgnesAdapter) -> None:
        """测试查询视频状态（失败）"""
        await adapter.start()

        mock_result = {
            "status": "failed",
            "error": "Content policy violation",
            "created": 1700000000,
            "updated": 1700000050,
        }

        mock_client_instance = MagicMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.close = AsyncMock()
        mock_client_instance.get = AsyncMock(
            return_value=self._mock_response(mock_result)
        )

        with patch(
            "agn.adapters.agnes.AsyncHttpClient", return_value=mock_client_instance
        ):
            result = await adapter.video_poll(task_id="vid-failed")

            assert result.status == "failed"
            assert result.error == "Content policy violation"
            assert result.video_url is None

        await adapter.close()


class TestAgnesAdapterEmbed:
    """AgnesAdapter 文本嵌入测试（Mock HTTP）"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> AgnesAdapter:
        """创建已启动的适配器"""
        config = ProviderConfig(
            provider_type="agnes",
            api_key=mock_api_key,
            base_url="https://api.test.agnes.ai/v1",
        )
        return AgnesAdapter(config=config)

    def _mock_response(self, json_data: dict, status_code: int = 200) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    @pytest.mark.asyncio
    async def test_embed_single_text(self, adapter: AgnesAdapter) -> None:
        """测试单文本嵌入"""
        await adapter.start()

        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_result = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": mock_embedding}],
            "model": "embedding-v1",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            from agn.models.options import EmbeddingResult

            result = await adapter.embed(
                model="embedding-v1",
                input="Hello world",
            )

            # 验证请求
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs.kwargs["url"] == "/embeddings"
            assert call_kwargs.kwargs["json"]["model"] == "embedding-v1"
            assert call_kwargs.kwargs["json"]["input"] == "Hello world"

            # 验证响应
            assert isinstance(result, EmbeddingResult)
            assert result.object == "list"
            assert result.model == "embedding-v1"
            assert len(result.data) == 1
            assert result.data[0]["embedding"] == mock_embedding
            assert result.usage["total_tokens"] == 5

        await adapter.close()

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, adapter: AgnesAdapter) -> None:
        """测试多文本嵌入"""
        await adapter.start()

        mock_result = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
            ],
            "model": "embedding-v1",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.embed(
                model="embedding-v1",
                input=["text one", "text two"],
                dimensions=256,
            )

            call_kwargs = mock_post.call_args
            body = call_kwargs.kwargs["json"]
            assert len(body["input"]) == 2
            assert body["dimensions"] == 256

            assert len(result.data) == 2
            assert result.data[0]["index"] == 0
            assert result.data[1]["index"] == 1

        await adapter.close()

    @pytest.mark.asyncio
    async def test_embed_with_encoding_format(self, adapter: AgnesAdapter) -> None:
        """测试带编码格式的嵌入"""
        await adapter.start()

        mock_result = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "embedding-v1",
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            await adapter.embed(
                model="embedding-v1",
                input="test",
                encoding_format="float",
                user="test-user",
            )

            call_kwargs = mock_post.call_args
            body = call_kwargs.kwargs["json"]
            assert body["encoding_format"] == "float"
            assert body["user"] == "test-user"

        await adapter.close()
