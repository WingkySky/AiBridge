"""
AGN-SDK Anthropic 适配器测试
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agn.adapters.anthropic import AnthropicAdapter
from agn.models.chat import ChatMessage
from agn.models.common import ProviderConfig


class TestAnthropicAdapter:
    """AnthropicAdapter 类测试"""

    @pytest.fixture
    def adapter_config(self, mock_api_key: str) -> ProviderConfig:
        """创建适配器配置"""
        return ProviderConfig(
            provider_type="anthropic",
            api_key=mock_api_key,
        )

    @pytest.fixture
    def adapter(self, adapter_config: ProviderConfig) -> AnthropicAdapter:
        """创建适配器实例"""
        return AnthropicAdapter(config=adapter_config)

    def test_adapter_init(self, adapter: AnthropicAdapter, mock_api_key: str) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "anthropic"
        assert adapter.provider_name == "Anthropic Claude"
        assert "chat" in adapter.supported_capabilities
        assert "chat_stream" in adapter.supported_capabilities
        assert "vision" in adapter.supported_capabilities
        assert "thinking" in adapter.supported_capabilities
        assert adapter.api_key == mock_api_key
        assert adapter.base_url == "https://api.anthropic.com"
        assert adapter.api_version == "2023-06-01"

    def test_adapter_supports_capability(self, adapter: AnthropicAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("chat_stream")
        assert adapter.supports_capability("vision")
        assert adapter.supports_capability("tool_call")
        assert adapter.supports_capability("function_call")
        assert adapter.supports_capability("reasoning")
        assert adapter.supports_capability("thinking")
        assert not adapter.supports_capability("image")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_start(self, adapter: AnthropicAdapter) -> None:
        """测试适配器启动"""
        await adapter.start()
        assert adapter._http_client is not None
        await adapter.close()

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: AnthropicAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None


class TestAnthropicAdapterListModels:
    """AnthropicAdapter 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> AnthropicAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="anthropic", api_key=mock_api_key)
        return AnthropicAdapter(config=config)

    def _mock_response(self, json_data: dict) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=json_data)
        return mock_resp

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: AnthropicAdapter) -> None:
        """测试获取所有模型（实时拉取 /v1/models）"""
        await adapter.start()

        # Anthropic /v1/models 响应使用 display_name 字段
        mock_result = {
            "data": [
                {
                    "id": "claude-3-opus-20240229",
                    "display_name": "Claude 3 Opus",
                    "type": "model",
                },
                {
                    "id": "claude-3-5-sonnet-20241022",
                    "display_name": "Claude 3.5 Sonnet",
                    "type": "model",
                },
                {
                    "id": "claude-3-5-haiku-20241022",
                    "display_name": "Claude 3.5 Haiku",
                    "type": "model",
                },
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models()

            # 验证调用了 /v1/models 端点
            mock_get.assert_called_once_with(url="/v1/models")
            assert len(models) == 3
            # 验证 provider 字段被正确设置
            for model in models:
                assert model.provider == "anthropic"
            # 验证 display_name 被解析为 name
            ids = {m.id for m in models}
            assert "claude-3-opus-20240229" in ids
            assert "claude-3-5-sonnet-20241022" in ids
            names = {m.name for m in models}
            assert "Claude 3 Opus" in names
            # Anthropic 模型 id 均被推断为 chat 类型
            types = {m.type for m in models}
            assert "chat" in types

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models(self, adapter: AnthropicAdapter) -> None:
        """测试获取对话模型（按 model_type 过滤）"""
        await adapter.start()

        mock_result = {
            "data": [
                {
                    "id": "claude-3-opus-20240229",
                    "display_name": "Claude 3 Opus",
                    "type": "model",
                },
                {
                    "id": "claude-3-5-sonnet-20241022",
                    "display_name": "Claude 3.5 Sonnet",
                    "type": "model",
                },
            ]
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            models = await adapter.list_models(model_type="chat")

            mock_get.assert_called_once_with(url="/v1/models")
            # Anthropic 所有 claude-* 模型均会被推断为 chat 类型
            for model in models:
                assert model.type == "chat"
            assert len(models) == 2

        await adapter.close()

    @pytest.mark.asyncio
    async def test_image_not_supported(self, adapter: AnthropicAdapter) -> None:
        """测试图像生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.image_generate(
                model="test",
                prompt="An image",
            )

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: AnthropicAdapter) -> None:
        """测试视频生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(
                model="test",
                prompt="A video",
            )


class TestAnthropicAdapterChatMock:
    """AnthropicAdapter 文本对话测试（Mock HTTP）"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> AnthropicAdapter:
        """创建已启动的适配器"""
        config = ProviderConfig(
            provider_type="anthropic",
            api_key=mock_api_key,
        )
        return AnthropicAdapter(config=config)

    def _mock_response(self, json_data: dict, status_code: int = 200) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    @pytest.mark.asyncio
    async def test_chat_basic_completion(
        self, adapter: AnthropicAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试基础文本对话响应解析"""
        await adapter.start()

        mock_result = {
            "id": "msg_abc123",
            "type": "message",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {
                    "type": "text",
                    "text": "Hello! How can I help you today?",
                }
            ],
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 20,
                "output_tokens": 15,
            },
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            result = await adapter.chat(
                model="claude-3-5-sonnet-20241022",
                messages=messages,
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/v1/messages"

            from agn.models.chat import ChatCompletion

            assert isinstance(result, ChatCompletion)
            assert result.id == "msg_abc123"
            assert result.model == "claude-3-5-sonnet-20241022"
            assert len(result.choices) == 1
            assert result.choices[0].message.role == "assistant"
            assert (
                result.choices[0].message.content == "Hello! How can I help you today?"
            )
            assert result.choices[0].finish_reason == "stop"
            assert result.usage is not None
            assert result.usage.prompt_tokens == 20
            assert result.usage.completion_tokens == 15
            assert result.usage.total_tokens == 35

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_message_conversion(
        self, adapter: AnthropicAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试消息格式转换是否正确"""
        await adapter.start()

        mock_result = {
            "id": "msg_conv",
            "type": "message",
            "model": "claude-3-haiku-20240307",
            "content": [{"type": "text", "text": "Hi there!"}],
            "role": "assistant",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            await adapter.chat(
                model="claude-3-haiku-20240307",
                messages=messages,
            )

            call_args = mock_post.call_args
            body = call_args[1]["json"]

            assert body["model"] == "claude-3-haiku-20240307"
            assert "messages" in body
            assert len(body["messages"]) == 1
            assert body["messages"][0]["role"] == "user"
            assert body["messages"][0]["content"] == "Hello!"
            assert body["system"] == "You are a helpful assistant."
            assert body["max_tokens"] == 1024

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_with_temperature_and_top_p(
        self, adapter: AnthropicAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试带 temperature 和 top_p 参数传递"""
        await adapter.start()

        mock_result = {
            "id": "msg_temp",
            "type": "message",
            "model": "claude-3-sonnet-20240229",
            "content": [{"type": "text", "text": "Creative response"}],
            "role": "assistant",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            await adapter.chat(
                model="claude-3-sonnet-20240229",
                messages=messages,
                temperature=0.8,
                top_p=0.9,
                max_tokens=500,
                stop_sequences=["END"],
            )

            call_args = mock_post.call_args
            body = call_args[1]["json"]
            assert body["temperature"] == 0.8
            assert body["top_p"] == 0.9
            assert body["max_tokens"] == 500
            assert body["stop_sequences"] == ["END"]

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_stop_reason_max_tokens(
        self, adapter: AnthropicAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试 max_tokens 结束原因映射"""
        await adapter.start()

        mock_result = {
            "id": "msg_max",
            "type": "message",
            "model": "claude-3-opus-20240229",
            "content": [{"type": "text", "text": "Partial response"}],
            "role": "assistant",
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 100},
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            result = await adapter.chat(
                model="claude-3-opus-20240229",
                messages=messages,
            )

            assert result.choices[0].finish_reason == "length"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_stop_reason_stop_sequence(
        self, adapter: AnthropicAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试 stop_sequence 结束原因映射"""
        await adapter.start()

        mock_result = {
            "id": "msg_stop",
            "type": "message",
            "model": "claude-3-5-haiku-20241022",
            "content": [{"type": "text", "text": "Response with stop"}],
            "role": "assistant",
            "stop_reason": "stop_sequence",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            result = await adapter.chat(
                model="claude-3-5-haiku-20241022",
                messages=messages,
            )

            assert result.choices[0].finish_reason == "stop"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_multiple_content_blocks(
        self, adapter: AnthropicAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试多个 content blocks 响应解析"""
        await adapter.start()

        mock_result = {
            "id": "msg_multi_block",
            "type": "message",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {"type": "text", "text": "First part. "},
                {"type": "text", "text": "Second part."},
            ],
            "role": "assistant",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 30},
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            result = await adapter.chat(
                model="claude-3-5-sonnet-20241022",
                messages=messages,
            )

            assert result.choices[0].message.content == "First part. Second part."

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_user_assistant_messages_only(
        self, adapter: AnthropicAdapter
    ) -> None:
        """测试只有 user 和 assistant 消息（无 system）"""
        await adapter.start()

        mock_result = {
            "id": "msg_no_sys",
            "type": "message",
            "model": "claude-3-haiku-20240307",
            "content": [{"type": "text", "text": "Reply without system"}],
            "role": "assistant",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 8},
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="Hello!"),
                ChatMessage(role="user", content="How are you?"),
            ]
            await adapter.chat(
                model="claude-3-haiku-20240307",
                messages=messages,
            )

            call_args = mock_post.call_args
            body = call_args[1]["json"]

            assert "system" not in body
            assert len(body["messages"]) == 3
            assert body["messages"][0]["role"] == "user"
            assert body["messages"][1]["role"] == "assistant"
            assert body["messages"][2]["role"] == "user"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_system_param_override(
        self, adapter: AnthropicAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试 system 参数覆盖 messages 中的 system 消息"""
        await adapter.start()

        mock_result = {
            "id": "msg_sys_override",
            "type": "message",
            "model": "claude-3-sonnet-20240229",
            "content": [{"type": "text", "text": "Override system prompt"}],
            "role": "assistant",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            await adapter.chat(
                model="claude-3-sonnet-20240229",
                messages=messages,
                system="Custom system prompt",
            )

            call_args = mock_post.call_args
            body = call_args[1]["json"]
            assert body["system"] == "Custom system prompt"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_with_thinking(
        self, adapter: AnthropicAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试带 thinking 参数的对话"""
        await adapter.start()

        mock_result = {
            "id": "msg_thinking",
            "type": "message",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Final answer"},
            ],
            "role": "assistant",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 50},
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            await adapter.chat(
                model="claude-3-5-sonnet-20241022",
                messages=messages,
                thinking={"type": "enabled", "budget_tokens": 1024},
            )

            call_args = mock_post.call_args
            body = call_args[1]["json"]
            assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_extra_body(
        self, adapter: AnthropicAdapter, sample_chat_messages: list[dict]
    ) -> None:
        """测试 extra_body 透传参数"""
        await adapter.start()

        mock_result = {
            "id": "msg_extra",
            "type": "message",
            "model": "claude-3-opus-20240229",
            "content": [{"type": "text", "text": "Extra body test"}],
            "role": "assistant",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            messages = [ChatMessage(**m) for m in sample_chat_messages]
            await adapter.chat(
                model="claude-3-opus-20240229",
                messages=messages,
                extra_body={"top_k": 50},
            )

            call_args = mock_post.call_args
            body = call_args[1]["json"]
            assert body["top_k"] == 50

        await adapter.close()
