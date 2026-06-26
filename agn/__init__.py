"""
AGN-SDK - 多模型统一接口 SDK

一套 API 调用所有 AI 模型，无论文本对话、图像生成、视频生成还是语音处理。

快速开始:

    from agn import Client

    client = Client(provider="agnes", api_key="your-key")

    # 文本对话
    response = await client.chat(
        model="claude-3-opus",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # 图像生成
    result = await client.image_generate(
        model="dall-e-3",
        prompt="A beautiful sunset"
    )

    # 视频生成
    task = await client.video_create(
        model="video-gen-1",
        prompt="A cat walking"
    )

    # 语音转文字
    result = await client.transcribe(
        model="whisper-1",
        file="audio.mp3"
    )

    # 文字转语音
    result = await client.speech(
        model="tts-1",
        input="Hello world",
        voice="alloy"
    )
"""

# 版本信息
__version__ = "1.1.1"

# 核心类
# 导入所有适配器（触发自动注册）
import agn.adapters  # noqa: F401

# 适配器
from agn.adapters.base import BaseAdapter, Capabilities
from agn.adapters.factory import AdapterFactory
from agn.client import Client

# 配置
from agn.core.config import Config, load_env

# 错误类型
from agn.core.errors import (
    AGNError,
    APIError,
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderNotFoundError,
    RateLimitError,
    TimeoutError,
    UnsupportedCapabilityError,
    ValidationError,
)
from agn.models.audio import (
    AudioResponseFormat,
    SpeechResult,
    SpeechVoice,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)
from agn.models.chat import (
    ChatChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatFunction,
    ChatMessage,
    ChatUsage,
)

# 数据模型
from agn.models.common import (
    ImageSize,
    ModelInfo,
    ModelType,
    ProviderConfig,
    ProviderInfo,
    TaskStatus,
    VideoMode,
)
from agn.models.image import (
    ImageData,
    ImageEditRequest,
    ImageGenerationOptions,
    ImageGenerationResult,
    ImageVariationRequest,
)
from agn.models.options import (
    ANTHROPIC_MAPPING,
    COHERE_MAPPING,
    GEMINI_MAPPING,
    OPENAI_COMPATIBLE_MAPPING,
    AspectRatio,
    ChatOptions,
    EmbeddingResult,
    EmbedOptions,
    FunctionDefinition,
    FunctionParameter,
    ImageOptions,
    ImageStyle,
    ParameterMapping,
    ReasoningEffort,
    ResponseFormat,
    SpeechOptions,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    TranscribeOptions,
    VideoDuration,
    VideoOptions,
)
from agn.models.video import (
    VideoGenerationOptions,
    VideoStatus,
    VideoTask,
    VideoTaskCreate,
)
from agn.router import Router

__all__ = [
    # 版本
    "__version__",
    # 核心类
    "Client",
    "Router",
    # 通用模型
    "ProviderConfig",
    "ModelInfo",
    "ProviderInfo",
    "ModelType",
    "VideoMode",
    "ImageSize",
    "TaskStatus",
    # 文本对话模型
    "ChatMessage",
    "ChatFunction",
    "ChatChoice",
    "ChatUsage",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionDelta",
    "ChatCompletionRequest",
    # 图像生成模型
    "ImageData",
    "ImageGenerationResult",
    "ImageEditRequest",
    "ImageVariationRequest",
    "ImageGenerationOptions",
    # 视频生成模型
    "VideoTask",
    "VideoStatus",
    "VideoGenerationOptions",
    "VideoTaskCreate",
    # 语音模型
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionWord",
    "SpeechResult",
    "AudioResponseFormat",
    "SpeechVoice",
    # 统一请求选项
    "ChatOptions",
    "ImageOptions",
    "VideoOptions",
    "EmbedOptions",
    "EmbeddingResult",
    "TranscribeOptions",
    "SpeechOptions",
    "ToolDefinition",
    "ToolCall",
    "ToolChoice",
    "FunctionDefinition",
    "FunctionParameter",
    "ReasoningEffort",
    "ImageStyle",
    "VideoDuration",
    "AspectRatio",
    "ResponseFormat",
    "ParameterMapping",
    "Capabilities",
    "OPENAI_COMPATIBLE_MAPPING",
    "ANTHROPIC_MAPPING",
    "GEMINI_MAPPING",
    "COHERE_MAPPING",
    # 错误类型
    "AGNError",
    "APIError",
    "AuthenticationError",
    "ModelNotFoundError",
    "NetworkError",
    "ProviderNotFoundError",
    "RateLimitError",
    "TimeoutError",
    "UnsupportedCapabilityError",
    "ValidationError",
    # 配置
    "Config",
    "load_env",
    # 适配器
    "BaseAdapter",
    "AdapterFactory",
]
