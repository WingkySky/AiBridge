"""
AGN-SDK 统一接口和参数映射测试

测试统一 ChatOptions/ImageOptions/VideoOptions/EmbedOptions 的使用方式，
验证"同一接口应对不同模型"的核心设计目标。
"""

import pytest

from agn import (
    AspectRatio,
    Capabilities,
    ChatOptions,
    EmbedOptions,
    ImageOptions,
    ImageStyle,
    ParameterMapping,
    ReasoningEffort,
    ToolChoice,
    ToolDefinition,
    VideoDuration,
    VideoOptions,
)
from agn.adapters.base import BaseAdapter
from agn.adapters.openai import OpenAIAdapter
from agn.models.chat import ChatMessage
from agn.models.common import ProviderConfig
from agn.models.options import (
    ANTHROPIC_MAPPING,
    OPENAI_COMPATIBLE_MAPPING,
)


class TestUnifiedOptions:
    """统一请求选项测试"""

    def test_chat_options_basic(self) -> None:
        """测试 ChatOptions 基础参数"""
        options = ChatOptions(
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
            top_k=40,
        )
        kwargs = options.to_kwargs()
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 2048
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 40

    def test_chat_options_reasoning(self) -> None:
        """测试 ChatOptions 推理/思考模式参数"""
        options = ChatOptions(
            reasoning=True,
            reasoning_effort=ReasoningEffort.HIGH,
            thinking_budget=16384,
        )
        kwargs = options.to_kwargs()
        assert kwargs["reasoning"] is True
        assert kwargs["reasoning_effort"] == ReasoningEffort.HIGH
        assert kwargs["thinking_budget"] == 16384

    def test_chat_options_tools(self) -> None:
        """测试 ChatOptions 工具调用参数"""
        weather_tool = ToolDefinition(
            type="function",
            function={
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        )
        options = ChatOptions(
            tools=[weather_tool],
            tool_choice=ToolChoice.AUTO,
        )
        kwargs = options.to_kwargs()
        assert len(kwargs["tools"]) == 1
        assert kwargs["tool_choice"] == ToolChoice.AUTO

    def test_chat_options_web_search(self) -> None:
        """测试 ChatOptions 联网搜索参数"""
        options = ChatOptions(
            web_search=True,
            search_recency_filter="week",
        )
        kwargs = options.to_kwargs()
        assert kwargs["web_search"] is True
        assert kwargs["search_recency_filter"] == "week"

    def test_chat_options_vision(self) -> None:
        """测试 ChatOptions 视觉参数"""
        options = ChatOptions(
            images=["https://example.com/image.jpg"],
            detail="high",
        )
        kwargs = options.to_kwargs()
        assert kwargs["images"] == ["https://example.com/image.jpg"]
        assert kwargs["detail"] == "high"

    def test_chat_options_extra_params(self) -> None:
        """测试厂商特有参数透传"""
        options = ChatOptions(
            temperature=0.5,
            extra_params={
                "top_k": 50,
                "repetition_penalty": 1.1,
                "custom_field": "value",
            },
        )
        kwargs = options.to_kwargs()
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_k"] == 50
        assert kwargs["repetition_penalty"] == 1.1
        assert kwargs["custom_field"] == "value"

    def test_image_options_basic(self) -> None:
        """测试 ImageOptions 基础参数"""
        options = ImageOptions(
            n=2,
            size="1024x1024",
            style=ImageStyle.VIVID,
            quality="hd",
            negative_prompt="blurry, low quality",
        )
        kwargs = options.to_kwargs()
        assert kwargs["n"] == 2
        assert kwargs["size"] == "1024x1024"
        assert kwargs["style"] == ImageStyle.VIVID
        assert kwargs["quality"] == "hd"
        assert kwargs["negative_prompt"] == "blurry, low quality"

    def test_image_options_aspect_ratio(self) -> None:
        """测试 ImageOptions 画面比例"""
        options = ImageOptions(
            aspect_ratio=AspectRatio.LANDSCAPE_16_9,
            width=1920,
            height=1080,
        )
        kwargs = options.to_kwargs()
        assert kwargs["aspect_ratio"] == AspectRatio.LANDSCAPE_16_9
        assert kwargs["width"] == 1920
        assert kwargs["height"] == 1080

    def test_image_options_reference_images(self) -> None:
        """测试 ImageOptions 参考图参数"""
        options = ImageOptions(
            reference_images=["https://example.com/ref.jpg"],
            reference_strength=0.7,
            mask="data:image/png;base64,...",
            edit_mode="inpaint",
        )
        kwargs = options.to_kwargs()
        assert len(kwargs["reference_images"]) == 1
        assert kwargs["reference_strength"] == 0.7
        assert kwargs["edit_mode"] == "inpaint"

    def test_video_options_basic(self) -> None:
        """测试 VideoOptions 基础参数"""
        options = VideoOptions(
            duration=VideoDuration.DURATION_5S,
            aspect_ratio=AspectRatio.LANDSCAPE_16_9,
            resolution="720p",
            negative_prompt="ugly, distorted",
            seed=42,
        )
        kwargs = options.to_kwargs()
        assert kwargs["duration"] == VideoDuration.DURATION_5S
        assert kwargs["aspect_ratio"] == AspectRatio.LANDSCAPE_16_9
        assert kwargs["resolution"] == "720p"

    def test_video_options_modes(self) -> None:
        """测试 VideoOptions 生成模式"""
        # 图生视频
        options = VideoOptions(
            mode="image2video",
            reference_images=["https://example.com/first_frame.jpg"],
            first_frame="https://example.com/first_frame.jpg",
            motion_strength=5.0,
        )
        kwargs = options.to_kwargs()
        assert kwargs["mode"] == "image2video"
        assert kwargs["motion_strength"] == 5.0

    def test_embed_options_basic(self) -> None:
        """测试 EmbedOptions 基础参数"""
        options = EmbedOptions(
            dimensions=1536,
            encoding_format="float",
        )
        kwargs = options.to_kwargs()
        assert kwargs["dimensions"] == 1536
        assert kwargs["encoding_format"] == "float"


class TestParameterMapping:
    """参数映射测试（通用参数 → 厂商特定参数）"""

    def test_simple_rename(self) -> None:
        """测试简单参数重命名映射"""
        mapping = ParameterMapping(
            rename_map={
                "max_tokens": "max_output_tokens",
                "stop": "stop_sequences",
                "top_p": "topP",
                "top_k": "topK",
            }
        )
        params = mapping.apply(
            {
                "max_tokens": 2048,
                "temperature": 0.7,
                "stop": ["END"],
            }
        )
        assert params["max_output_tokens"] == 2048
        assert params["temperature"] == 0.7
        assert params["stop_sequences"] == ["END"]
        # 原键名不应存在
        assert "max_tokens" not in params

    def test_value_mapping_dict_expand(self) -> None:
        """测试值映射为 dict 时展开"""
        mapping = ParameterMapping(
            value_map={
                "reasoning_effort": {
                    ReasoningEffort.HIGH: {
                        "thinking": {"type": "enabled", "budget_tokens": 16384}
                    },
                    ReasoningEffort.LOW: {
                        "thinking": {"type": "enabled", "budget_tokens": 1024}
                    },
                }
            }
        )
        params = mapping.apply({"reasoning_effort": ReasoningEffort.HIGH})
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 16384

    def test_value_mapping_simple(self) -> None:
        """测试简单值映射"""
        mapping = ParameterMapping(
            value_map={
                "response_format": {
                    "json_object": "json",
                    "text": "text",
                }
            }
        )
        params = mapping.apply({"response_format": "json_object"})
        assert params["response_format"] == "json"

    def test_remove_none_values(self) -> None:
        """测试 None 值移除"""
        mapping = ParameterMapping(remove_when_none=True)
        params = mapping.apply(
            {
                "temperature": 0.7,
                "max_tokens": None,
                "top_p": None,
            }
        )
        assert "temperature" in params
        assert "max_tokens" not in params
        assert "top_p" not in params

    def test_openai_compatible_mapping(self) -> None:
        """测试 OpenAI 兼容映射"""
        params = OPENAI_COMPATIBLE_MAPPING.apply(
            {
                "temperature": 0.7,
                "max_tokens": 2048,
                "reasoning": True,
                "web_search": False,
            }
        )
        # reasoning=True 应该映射为 reasoning_effort=medium
        assert params["reasoning_effort"] == "medium"
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 2048


class TestCapabilities:
    """细粒度能力声明测试"""

    def test_capability_constants(self) -> None:
        """测试能力常量定义"""
        assert Capabilities.CHAT == "chat"
        assert Capabilities.CHAT_STREAM == "chat_stream"
        assert Capabilities.VISION == "vision"
        assert Capabilities.TOOL_CALL == "tool_call"
        assert Capabilities.FUNCTION_CALL == "function_call"
        assert Capabilities.REASONING == "reasoning"
        assert Capabilities.JSON_MODE == "json_mode"
        assert Capabilities.WEB_SEARCH == "web_search"
        assert Capabilities.IMAGE_GENERATE == "image"
        assert Capabilities.IMAGE_EDIT == "image_edit"
        assert Capabilities.VIDEO_GENERATE == "video"
        assert Capabilities.EMBEDDING == "embedding"

    def test_openai_adapter_capabilities(self, mock_api_key: str) -> None:
        """测试 OpenAI 适配器的细粒度能力"""
        config = ProviderConfig(provider_type="openai", api_key=mock_api_key)
        adapter = OpenAIAdapter(config=config)
        assert adapter.supports_capability(Capabilities.CHAT)
        assert adapter.supports_capability(Capabilities.CHAT_STREAM)
        assert adapter.supports_capability(Capabilities.VISION)
        assert adapter.supports_capability(Capabilities.TOOL_CALL)
        assert adapter.supports_capability(Capabilities.FUNCTION_CALL)
        assert adapter.supports_capability(Capabilities.JSON_MODE)
        assert adapter.supports_capability(Capabilities.REASONING)
        assert adapter.supports_capability(Capabilities.IMAGE_GENERATE)
        assert adapter.supports_capability(Capabilities.EMBEDDING)
        assert not adapter.supports_capability(Capabilities.VIDEO_GENERATE)
        assert not adapter.supports_capability(Capabilities.WEB_SEARCH)

    @pytest.mark.asyncio
    async def test_check_capability_raises(self, mock_api_key: str) -> None:
        """测试 check_capability 不支持时抛出异常"""
        from agn.core.errors import UnsupportedCapabilityError

        config = ProviderConfig(provider_type="openai", api_key=mock_api_key)
        adapter = OpenAIAdapter(config=config)

        # 不支持的能力应该抛异常
        with pytest.raises(UnsupportedCapabilityError) as exc_info:
            adapter.check_capability(Capabilities.VIDEO_GENERATE)
        assert "video" in str(exc_info.value)


class TestBaseAdapterHelpers:
    """BaseAdapter 辅助方法测试"""

    def test_merge_options_with_kwargs(self) -> None:
        """测试 _merge_options 合并 Options 和 kwargs"""
        config = ProviderConfig(provider_type="openai", api_key="test-key")
        from agn.adapters.openai import OpenAIAdapter

        adapter = OpenAIAdapter(config=config)

        options = ChatOptions(temperature=0.7, max_tokens=1024)
        kwargs = {"max_tokens": 2048, "top_p": 0.9}  # kwargs 覆盖 options

        params = adapter._merge_options(options, kwargs)
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 2048  # kwargs 优先
        assert params["top_p"] == 0.9

    def test_merge_options_none_options(self) -> None:
        """测试 options=None 时只使用 kwargs"""
        config = ProviderConfig(provider_type="openai", api_key="test-key")
        from agn.adapters.openai import OpenAIAdapter

        adapter = OpenAIAdapter(config=config)

        params = adapter._merge_options(None, {"temperature": 0.5, "max_tokens": 512})
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 512

    def test_extract_system_prompt(self) -> None:
        """测试 _extract_system_prompt"""
        config = ProviderConfig(provider_type="openai", api_key="test-key")
        from agn.adapters.openai import OpenAIAdapter

        adapter = OpenAIAdapter(config=config)

        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello!"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]
        filtered, system = adapter._extract_system_prompt(messages)
        assert system == "You are a helpful assistant."
        assert len(filtered) == 2
        assert filtered[0]["role"] == "user"
        assert filtered[1]["role"] == "assistant"

    def test_build_messages_with_images(self) -> None:
        """测试 _build_messages_with_images 多模态消息构建"""
        config = ProviderConfig(provider_type="openai", api_key="test-key")
        from agn.adapters.openai import OpenAIAdapter

        adapter = OpenAIAdapter(config=config)

        messages = [
            ChatMessage(role="user", content="What's in this image?"),
        ]
        images = ["https://example.com/photo.jpg"]

        result = adapter._build_messages_with_images(messages, images, detail="high")
        assert len(result) == 1
        user_msg = result[0]
        assert isinstance(user_msg["content"], list)
        assert len(user_msg["content"]) == 2  # text + image
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][1]["type"] == "image_url"
        assert (
            user_msg["content"][1]["image_url"]["url"]
            == "https://example.com/photo.jpg"
        )
        assert user_msg["content"][1]["image_url"]["detail"] == "high"


class TestUnifiedUsageExample:
    """统一接口使用示例测试（演示如何同一接口应对不同模型）"""

    def test_same_options_different_providers(self) -> None:
        """
        演示：同一套 ChatOptions 可以用于不同厂商，适配器内部处理映射差异

        用户只需要写一次代码，切换 provider 不需要改参数。
        """
        # 用户只需要定义一次选项
        common_options = ChatOptions(
            temperature=0.7,
            max_tokens=2048,
            reasoning=True,
            reasoning_effort=ReasoningEffort.MEDIUM,
            response_format="json_object",
        )

        # OpenAI 使用这些参数
        openai_kwargs = common_options.to_kwargs()
        assert openai_kwargs["temperature"] == 0.7
        assert openai_kwargs["max_tokens"] == 2048
        assert openai_kwargs["reasoning_effort"] == ReasoningEffort.MEDIUM

        # Anthropic 通过映射后参数名会变化
        anthropic_kwargs = ANTHROPIC_MAPPING.apply(common_options.to_kwargs())
        assert (
            anthropic_kwargs["thinking"]["type"] == "enabled"
        )  # 自动映射为 Anthropic 的 thinking 格式
        assert anthropic_kwargs["temperature"] == 0.7
        assert anthropic_kwargs["max_tokens"] == 2048

    def test_same_image_options_different_providers(self) -> None:
        """
        演示：同一套 ImageOptions 用于不同图像模型
        """
        common_options = ImageOptions(
            n=1,
            aspect_ratio=AspectRatio.LANDSCAPE_16_9,
            style=ImageStyle.PHOTOREALISTIC,
            negative_prompt="blurry, distorted",
        )
        kwargs = common_options.to_kwargs()
        # 所有厂商都接收这些统一参数，适配器内部做映射
        assert kwargs["aspect_ratio"] == AspectRatio.LANDSCAPE_16_9
        assert kwargs["style"] == ImageStyle.PHOTOREALISTIC
        assert kwargs["n"] == 1

    def test_extra_params_for_provider_specific_features(self) -> None:
        """
        演示：厂商特有功能通过 extra_params 透传
        """
        # 例如使用 DeepSeek 的 reasoning_effort，或者 Claude 的 thinking budget
        options = ChatOptions(
            temperature=0.3,
            reasoning=True,
            extra_params={
                # DeepSeek 特有
                "deepseek_reasoning": True,
                # Claude 特有
                "thinking_budget_tokens": 8192,
                # Gemini 特有
                "google_search": True,
            },
        )
        kwargs = options.to_kwargs()
        assert kwargs["temperature"] == 0.3
        assert kwargs["deepseek_reasoning"] is True
        assert kwargs["thinking_budget_tokens"] == 8192
        assert kwargs["google_search"] is True
