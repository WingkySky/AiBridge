"""
AGN-SDK 适配器基类

定义所有 Provider 适配器必须实现的抽象接口，以及统一参数映射机制。

设计原则：
1. 用户使用统一参数对象（ChatOptions/ImageOptions/VideoOptions/EmbedOptions/TranscribeOptions/SpeechOptions）
2. 适配器内部通过 ParameterMapping 自动将通用参数映射到厂商特定字段
3. 不支持的能力抛出明确的 UnsupportedCapabilityError
4. 所有响应统一归一化为标准数据结构
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, ClassVar

from agn.models.options import (
    OPENAI_COMPATIBLE_MAPPING,
    ChatOptions,
    EmbeddingResult,
    EmbedOptions,
    ImageOptions,
    ParameterMapping,
    SpeechOptions,
    TranscribeOptions,
    VideoOptions,
)

if TYPE_CHECKING:
    from agn.models.audio import SpeechResult, TranscriptionResult
    from agn.models.chat import ChatCompletion, ChatCompletionChunk, ChatMessage
    from agn.models.common import ModelInfo, ProviderConfig
    from agn.models.image import ImageGenerationResult
    from agn.models.video import VideoStatus, VideoTask


# ==================== 能力常量 ====================


class Capabilities:
    """细粒度能力常量（用于 supports_capability 检查）"""

    # 对话能力
    CHAT = "chat"
    CHAT_STREAM = "chat_stream"
    VISION = "vision"
    TOOL_CALL = "tool_call"
    FUNCTION_CALL = "function_call"
    REASONING = "reasoning"
    THINKING = "thinking"
    JSON_MODE = "json_mode"
    WEB_SEARCH = "web_search"

    # 图像能力
    IMAGE_GENERATE = "image"
    IMAGE_EDIT = "image_edit"
    IMAGE_VARIATION = "image_variation"
    IMAGE_TO_IMAGE = "image_to_image"
    IMAGE_INPAINT = "image_inpaint"

    # 视频能力
    VIDEO_GENERATE = "video"
    VIDEO_TEXT2VIDEO = "text2video"
    VIDEO_IMAGE2VIDEO = "image2video"
    VIDEO_EXTEND = "video_extend"

    # 嵌入能力
    EMBEDDING = "embedding"

    # 其他
    AUDIO_TRANSCRIBE = "audio_transcribe"
    AUDIO_TRANSLATE = "audio_translate"
    AUDIO_SPEECH = "audio_speech"


# 模型类型推断关键字：用于从 /models 端点拉取的模型 ID 推断类型
# 参考 agnes-platform provider_registry._detect_type()
_MODEL_TYPE_KEYWORDS: dict[str, list[str]] = {
    "image": [
        "image",
        "flux",
        "sd3",
        "sdxl",
        "dall",
        "seedream",
        "wanx",
        "ideogram",
        "midjourney",
        "stable-diffusion",
        "Imagen",
    ],
    "video": [
        "video",
        "veo",
        "seedance",
        "cogvideox",
        "wan",
        "kling",
        "runway",
        "pika",
        "luma",
        "Sora",
        "Vidu",
    ],
    "audio": [
        "whisper",
        "tts",
        "speech",
        "transcribe",
        "nova",
        "sonic",
        "edge-tts",
        "eleven",
        "cosyvoice",
        "sensevoice",
    ],
}


class BaseAdapter(ABC):
    """
    适配器基类

    所有 Provider 适配器必须继承此类并实现所有抽象方法。
    适配器负责将统一接口转换为各 Provider 的特定 API 调用格式，
    并将响应归一化为统一的数据结构。

    属性:
        provider_type: Provider 类型标识（如 'agnes'、'openai'）
        provider_name: Provider 显示名称
        supported_capabilities: 支持的能力列表（使用 Capabilities 常量）
        param_mapping: 参数映射规则（默认 OpenAI 兼容）
        requires_api_key: 是否需要 API Key（免费 Provider 如 Edge TTS 设为 False）
    """

    # 类变量：由子类覆盖
    provider_type: ClassVar[str] = ""
    provider_name: ClassVar[str] = ""
    supported_capabilities: ClassVar[list[str]] = []
    param_mapping: ClassVar[ParameterMapping] = OPENAI_COMPATIBLE_MAPPING
    # 是否需要 API Key：免费 Provider（如 Edge TTS）设为 False 即可免认证使用
    requires_api_key: ClassVar[bool] = True

    def __init__(self, config: "ProviderConfig") -> None:
        """
        初始化适配器

        Args:
            config: Provider 配置对象
        """
        self.config = config
        # api_key 兜底为 str：requires_api_key=True 时 Client 已保证非 None，
        # requires_api_key=False 的适配器（如 Edge TTS）不使用此字段，空串不影响
        self.api_key: str = config.api_key or ""
        self._client: Any | None = None  # HTTP 客户端，由子类初始化

    @abstractmethod
    async def start(self) -> None:
        """
        启动适配器

        初始化 HTTP 客户端、连接池等资源。
        在首次使用适配器前必须调用此方法。
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        关闭适配器

        释放所有资源（HTTP 客户端、连接池等）。
        在不再使用适配器时应调用此方法。
        """
        pass

    # ==================== 参数处理辅助方法 ====================

    def _merge_options(
        self,
        options: (
            ChatOptions
            | ImageOptions
            | VideoOptions
            | EmbedOptions
            | TranscribeOptions
            | SpeechOptions
            | None
        ),
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        合并 Options 对象和 kwargs，返回统一参数字典

        Args:
            options: Options 对象（优先）
            kwargs: 额外关键字参数

        Returns:
            合并后的参数字典
        """
        if options is not None:
            params = options.to_kwargs()
            params.update(kwargs)
        else:
            params = dict(kwargs)
        return params

    def _map_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        应用参数映射，将通用参数转换为厂商特定参数

        Args:
            params: 通用参数字典

        Returns:
            映射后的厂商特定参数字典
        """
        return self.param_mapping.apply(params)

    def _build_messages_with_images(
        self,
        messages: list["ChatMessage"],
        images: list[str] | None = None,
        detail: str = "auto",
    ) -> list[dict[str, Any]]:
        """
        构建包含图片的多模态消息列表（OpenAI 格式）

        Args:
            messages: 原始消息列表
            images: 图片 URL/data URI 列表（追加到最后一条 user 消息）
            detail: 图片细节级别

        Returns:
            OpenAI 格式的消息列表
        """
        result: list[dict[str, Any]] = []

        for msg in messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                msg_dict = dict(msg)
            result.append(msg_dict)

        if images:
            # 找到最后一条 user 消息
            for i in range(len(result) - 1, -1, -1):
                if result[i].get("role") == "user":
                    content = result[i].get("content", "")
                    content_blocks: list[dict[str, Any]] = []

                    if isinstance(content, str) and content:
                        content_blocks.append({"type": "text", "text": content})

                    for img_url in images:
                        content_blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": img_url,
                                    "detail": detail,
                                },
                            }
                        )

                    result[i]["content"] = content_blocks
                    break

        return result

    def _extract_system_prompt(
        self, messages: list["ChatMessage"]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        从消息列表中提取 system prompt

        Args:
            messages: 消息列表

        Returns:
            (过滤后的消息列表, system prompt)
        """
        system_prompt: str | None = None
        result: list[dict[str, Any]] = []

        for msg in messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                msg_dict = dict(msg)

            role = msg_dict.get("role", "")
            content = msg_dict.get("content", "")

            if role == "system":
                system_prompt = content if isinstance(content, str) else str(content)
            else:
                result.append(msg_dict)

        return result, system_prompt

    # ==================== 文本对话（统一接口）====================

    async def chat_with_options(
        self,
        model: str,
        messages: list["ChatMessage"],
        options: ChatOptions | None = None,
        **kwargs: Any,
    ) -> "ChatCompletion":
        """
        统一接口：文本对话（接受 ChatOptions 对象）

        Args:
            model: 模型名称
            messages: 消息列表
            options: 统一对话选项
            **kwargs: 额外参数（覆盖 options）

        Returns:
            对话完成结果
        """
        params = self._merge_options(options, kwargs)
        return await self.chat(model, messages, **params)

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list["ChatMessage"],
        **kwargs: Any,
    ) -> "ChatCompletion":
        """
        文本对话（底层接口）

        Args:
            model: 模型名称
            messages: 消息列表
            **kwargs: 所有参数（经过映射后的厂商特定参数）

        Returns:
            对话完成结果
        """
        pass

    async def chat_stream_with_options(
        self,
        model: str,
        messages: list["ChatMessage"],
        options: ChatOptions | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator["ChatCompletionChunk", None]:
        """
        统一接口：流式文本对话（接受 ChatOptions 对象）
        """
        params = self._merge_options(options, kwargs)
        params["stream"] = True
        async for chunk in self.chat_stream(model, messages, **params):
            yield chunk

    async def chat_stream(
        self,
        model: str,
        messages: list["ChatMessage"],
        **kwargs: Any,
    ) -> AsyncGenerator["ChatCompletionChunk", None]:
        """
        流式文本对话（底层接口）

        Args:
            model: 模型名称
            messages: 消息列表
            **kwargs: 其他参数

        Yields:
            逐个返回对话块

        Raises:
            UnsupportedCapabilityError: 如果 Provider 不支持流式输出
        """
        from agn.core.errors import UnsupportedCapabilityError

        if False:
            yield  # 让函数成为异步生成器
        raise UnsupportedCapabilityError(
            message=f"Provider '{self.provider_type}' does not support streaming",
            details={"capability": "chat_stream", "provider": self.provider_type},
        )

    # ==================== 文本嵌入（新增统一接口）====================

    async def embed_with_options(
        self,
        input: str | list[str],
        options: EmbedOptions | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        统一接口：文本嵌入（接受 EmbedOptions 对象）

        Args:
            input: 输入文本或文本列表
            options: 嵌入选项
            **kwargs: 额外参数

        Returns:
            嵌入结果
        """
        params = self._merge_options(options, kwargs)
        model = params.pop("model", None) or self.config.extra.get("embed_model", "")
        return await self.embed(model=model, input=input, **params)

    async def embed(
        self,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        文本嵌入（底层接口）

        Args:
            model: 模型名称
            input: 输入文本或文本列表
            **kwargs: 其他参数

        Returns:
            嵌入结果

        Raises:
            UnsupportedCapabilityError: 如果 Provider 不支持嵌入
        """
        from agn.core.errors import UnsupportedCapabilityError

        raise UnsupportedCapabilityError(
            message=f"Provider '{self.provider_type}' does not support embeddings",
            details={"capability": "embedding", "provider": self.provider_type},
        )

    # ==================== 语音转文字 ====================

    async def transcribe_with_options(
        self,
        options: TranscribeOptions | None = None,
        **kwargs: Any,
    ) -> "TranscriptionResult":
        """
        统一接口：语音转文字（接受 TranscribeOptions 对象）

        Args:
            options: 转写选项
            **kwargs: 额外参数

        Returns:
            转写结果
        """
        params = self._merge_options(options, kwargs)
        file = params.pop("file", None)
        model = params.pop("model", None) or self.config.extra.get(
            "transcribe_model", "whisper-1"
        )
        return await self.transcribe(model=model, file=file, **params)

    async def transcribe(
        self,
        model: str,
        file: Any,
        **kwargs: Any,
    ) -> "TranscriptionResult":
        """
        语音转文字（底层接口）

        Args:
            model: 模型名称
            file: 音频文件（路径、URL、base64 或二进制数据）
            **kwargs: 其他参数（language, prompt, temperature, response_format 等）

        Returns:
            转写结果

        Raises:
            UnsupportedCapabilityError: 如果 Provider 不支持语音转文字
        """
        from agn.core.errors import UnsupportedCapabilityError

        raise UnsupportedCapabilityError(
            message=f"Provider '{self.provider_type}' does not support audio transcription",
            details={"capability": "audio_transcribe", "provider": self.provider_type},
        )

    # ==================== 语音翻译（转写为英文）====================

    async def translate_with_options(
        self,
        options: TranscribeOptions | None = None,
        **kwargs: Any,
    ) -> "TranscriptionResult":
        """
        统一接口：语音翻译（接受 TranscribeOptions 对象）

        Args:
            options: 转写选项
            **kwargs: 额外参数

        Returns:
            翻译结果（英文）
        """
        params = self._merge_options(options, kwargs)
        params["translate"] = True
        file = params.pop("file", None)
        model = params.pop("model", None) or self.config.extra.get(
            "transcribe_model", "whisper-1"
        )
        return await self.translate(model=model, file=file, **params)

    async def translate(
        self,
        model: str,
        file: Any,
        **kwargs: Any,
    ) -> "TranscriptionResult":
        """
        语音翻译（底层接口，翻译为英文）

        Args:
            model: 模型名称
            file: 音频文件
            **kwargs: 其他参数

        Returns:
            翻译结果

        Raises:
            UnsupportedCapabilityError: 如果 Provider 不支持语音翻译
        """
        from agn.core.errors import UnsupportedCapabilityError

        raise UnsupportedCapabilityError(
            message=f"Provider '{self.provider_type}' does not support audio translation",
            details={"capability": "audio_translate", "provider": self.provider_type},
        )

    # ==================== 文字转语音 ====================

    async def speech_with_options(
        self,
        options: SpeechOptions | None = None,
        **kwargs: Any,
    ) -> "SpeechResult":
        """
        统一接口：文字转语音（接受 SpeechOptions 对象）

        Args:
            options: 语音合成选项
            **kwargs: 额外参数

        Returns:
            语音合成结果
        """
        params = self._merge_options(options, kwargs)
        input_text = params.pop("input", None)
        model = params.pop("model", None) or self.config.extra.get(
            "speech_model", "tts-1"
        )
        voice = params.pop("voice", None) or "alloy"
        return await self.speech(model=model, input=input_text, voice=voice, **params)

    async def speech(
        self,
        model: str,
        input: str,
        voice: str | list[str],
        **kwargs: Any,
    ) -> "SpeechResult":
        """
        文字转语音（底层接口）

        Args:
            model: 模型名称
            input: 要合成的文本
            voice: 音色，支持单个音色字符串或音色列表（列表时启用 fallback 降级）
            **kwargs: 其他参数（response_format, speed 等）

        Returns:
            语音合成结果（包含音频数据）

        Raises:
            UnsupportedCapabilityError: 如果 Provider 不支持文字转语音
        """
        from agn.core.errors import UnsupportedCapabilityError

        raise UnsupportedCapabilityError(
            message=f"Provider '{self.provider_type}' does not support text-to-speech",
            details={"capability": "audio_speech", "provider": self.provider_type},
        )

    # ==================== 图像生成（统一接口）====================

    async def image_generate_with_options(
        self,
        model: str,
        prompt: str,
        options: ImageOptions | None = None,
        **kwargs: Any,
    ) -> "ImageGenerationResult":
        """
        统一接口：图像生成（接受 ImageOptions 对象）
        """
        params = self._merge_options(options, kwargs)
        return await self.image_generate(model, prompt, **params)

    @abstractmethod
    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> "ImageGenerationResult":
        """
        图像生成（底层接口）

        Args:
            model: 模型名称
            prompt: 提示词
            **kwargs: 其他参数

        Returns:
            图像生成结果
        """
        pass

    # ==================== 视频生成（统一接口）====================

    async def video_create_with_options(
        self,
        model: str,
        prompt: str,
        options: VideoOptions | None = None,
        **kwargs: Any,
    ) -> "VideoTask":
        """
        统一接口：创建视频生成任务（接受 VideoOptions 对象）
        """
        params = self._merge_options(options, kwargs)
        return await self.video_create(model, prompt, **params)

    @abstractmethod
    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> "VideoTask":
        """
        创建视频生成任务（底层接口）

        Args:
            model: 模型名称
            prompt: 提示词
            **kwargs: 其他参数

        Returns:
            视频任务信息
        """
        pass

    @abstractmethod
    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> "VideoStatus":
        """
        查询视频任务状态

        Args:
            task_id: 任务 ID
            model: 模型名称（部分 Provider 需要）

        Returns:
            视频任务状态
        """
        pass

    # ==================== 模型信息 ====================

    @abstractmethod
    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list["ModelInfo"]:
        """
        获取可用模型列表

        Args:
            model_type: 模型类型过滤（'chat' / 'image' / 'video'）

        Returns:
            模型信息列表
        """
        pass

    @staticmethod
    def _infer_type(model_id: str) -> str:
        """
        根据模型 ID 推断模型类型

        用于从 /models 端点拉取的模型列表（通常只返回 id 字符串）推断
        chat / image / video / audio 类型。

        Args:
            model_id: 模型标识符

        Returns:
            模型类型（chat / image / video / audio）
        """
        lower = model_id.lower()
        for type_name, keywords in _MODEL_TYPE_KEYWORDS.items():
            if any(kw.lower() in lower for kw in keywords):
                return type_name
        return "chat"

    @staticmethod
    def _parse_models_response(
        data: dict[str, Any],
        provider: str,
        model_type: str | None = None,
        items_key: str = "data",
    ) -> list["ModelInfo"]:
        """
        解析 OpenAI 兼容的 /models 响应为 ModelInfo 列表

        适用于返回 {"data": [{"id": ..., ...}]} 结构的端点。
        各适配器调用 /models 端点后，传入响应数据由本方法统一解析。

        Args:
            data: /models 端点返回的 JSON 数据
            provider: Provider 类型标识（如 'openai'、'anthropic'）
            model_type: 模型类型过滤（可选）
            items_key: 响应中模型列表的键名，默认 'data'

        Returns:
            模型信息列表
        """
        from agn.models.common import ModelInfo

        models: list[ModelInfo] = []
        for item in data.get(items_key, []):
            # 兼容 OpenAI 的 "id"、Anthropic 的 "id"、Gemini 的 "name"（需调用方预处理）
            model_id = item.get("id", "")
            if not model_id:
                continue

            inferred_type = BaseAdapter._infer_type(model_id)

            models.append(
                ModelInfo(
                    id=model_id,
                    # 兼容 OpenAI 无 name、Anthropic 的 display_name
                    name=item.get("name") or item.get("display_name") or model_id,
                    type=inferred_type,
                    provider=provider,
                    capabilities=item.get("capabilities", []),
                    description=item.get("description"),
                    created=item.get("created"),
                )
            )

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    # ==================== 音色查询（TTS Provider 可选实现）====================

    async def list_voices(
        self,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        列出可用音色

        TTS Provider 可覆盖此方法返回真实音色列表。默认抛出不支持异常。

        Args:
            language: 按语言过滤，如 "zh-CN" / "en-US"

        Returns:
            音色列表，每项为 dict（字段因 Provider 而异）

        Raises:
            UnsupportedCapabilityError: Provider 不支持音色查询
        """
        from agn.core.errors import UnsupportedCapabilityError

        raise UnsupportedCapabilityError(
            message=f"Provider '{self.provider_type}' does not support list_voices",
            details={"capability": "list_voices", "provider": self.provider_type},
        )

    async def recommend_voices(
        self,
        language: str | None = None,
        gender: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        推荐可用音色

        在 list_voices() 基础上按语言/性别过滤并返回推荐列表。
        TTS Provider 可覆盖此方法提供更智能的推荐。默认实现基于 list_voices 过滤。

        Args:
            language: 按语言过滤，如 "zh-CN" / "en-US"
            gender: 按性别过滤，如 "female" / "male"（不区分大小写）
            limit: 最多返回条数

        Returns:
            推荐音色列表

        Raises:
            UnsupportedCapabilityError: Provider 不支持音色查询
        """
        voices = await self.list_voices(language=language)
        if gender:
            gender_lower = gender.lower()
            voices = [
                v for v in voices if str(v.get("Gender", "")).lower() == gender_lower
            ]
        return voices[:limit]

    # ==================== 能力检查 ====================

    def supports_capability(self, capability: str) -> bool:
        """
        检查是否支持指定能力

        Args:
            capability: 能力名称（使用 Capabilities 常量）

        Returns:
            是否支持
        """
        return capability in self.supported_capabilities

    def supports_model_type(self, model_type: str) -> bool:
        """
        检查是否支持指定模型类型

        Args:
            model_type: 模型类型

        Returns:
            是否支持
        """
        return model_type in self.supported_capabilities

    def check_capability(self, capability: str) -> None:
        """
        检查能力，不支持则抛出异常

        Args:
            capability: 能力名称

        Raises:
            UnsupportedCapabilityError: 如果不支持
        """
        from agn.core.errors import UnsupportedCapabilityError

        if not self.supports_capability(capability):
            raise UnsupportedCapabilityError(
                message=f"Provider '{self.provider_type}' does not support '{capability}'",
                details={"capability": capability, "provider": self.provider_type},
            )

    # ==================== 错误处理 ====================

    def _handle_http_error(
        self, response: Any, provider_name: str | None = None
    ) -> None:
        """
        统一 HTTP 错误处理

        Args:
            response: httpx Response 对象
            provider_name: Provider 名称（可选）
        """

        from agn.core.errors import APIError, AuthenticationError, RateLimitError

        name = provider_name or self.provider_name

        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message=f"Invalid {name} API key")
        if response.status_code == 403:
            raise AuthenticationError(message=f"{name} API key forbidden")
        if response.status_code == 429:
            raise RateLimitError(message=f"{name} rate limit exceeded")
        if response.status_code == 500:
            raise APIError(message=f"{name} internal server error", status_code=500)
        if response.status_code == 502:
            raise APIError(message=f"{name} bad gateway", status_code=502)
        if response.status_code == 503:
            raise APIError(message=f"{name} service unavailable", status_code=503)

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("error", {}).get("message")
                if isinstance(error_data.get("error"), dict)
                else None
            )
            if error_msg is None:
                error_msg = (
                    error_data.get("message")
                    or error_data.get("error")
                    or error_data.get("msg")
                    or f"HTTP {response.status_code}"
                )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=f"{name}: {error_msg}", status_code=response.status_code)

    # ==================== 上下文管理 ====================

    async def __aenter__(self) -> "BaseAdapter":
        """异步上下文管理器入口"""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.close()
