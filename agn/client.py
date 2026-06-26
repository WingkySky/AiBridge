"""
AGN-SDK 统一客户端

提供统一的 API 接口，是用户使用 SDK 的唯一入口。
"""

from collections.abc import AsyncGenerator
from typing import Any, Literal

from agn.adapters.base import BaseAdapter
from agn.adapters.factory import AdapterFactory
from agn.core.config import get_provider_config
from agn.core.errors import ValidationError
from agn.models.audio import SpeechResult, TranscriptionResult
from agn.models.chat import ChatCompletion, ChatCompletionChunk, ChatMessage
from agn.models.common import ModelInfo, ProviderConfig
from agn.models.image import ImageGenerationResult
from agn.models.options import (
    ChatOptions,
    EmbeddingResult,
    EmbedOptions,
    ImageOptions,
    SpeechOptions,
    TranscribeOptions,
    VideoOptions,
)
from agn.models.video import VideoStatus, VideoTask


class Client:
    """
    统一客户端

    提供统一的 API 接口，支持文本对话、图像生成、视频生成等能力。

    使用方式:

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
    """

    def __init__(
        self,
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
        poll_url: str | None = None,
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        **kwargs: Any,
    ) -> None:
        """
        初始化客户端

        Args:
            provider: Provider 类型（如 'agnes'、'openai'、'runway'）
            api_key: API Key（可直接传入或从环境变量读取）
            base_url: API Base URL（可选，部分 Provider 有默认值）
            poll_url: 轮询 URL（视频生成任务状态用）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            **kwargs: 其他 Provider 特定配置
        """
        self.provider_type = provider

        # 构建配置
        config_overrides: dict[str, Any] = {
            "timeout": timeout,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
        }
        if api_key:
            config_overrides["api_key"] = api_key
        if base_url:
            config_overrides["base_url"] = base_url
        if poll_url:
            config_overrides["poll_url"] = poll_url
        config_overrides.update(kwargs)

        # 获取配置（合并环境变量）
        full_config = get_provider_config(provider, config_overrides)

        # 检查是否需要 API Key：免费 Provider（如 Edge TTS）无需认证
        adapter_class = AdapterFactory.get_adapter_class(provider)
        requires_key = (
            adapter_class.requires_api_key if adapter_class is not None else True
        )
        if requires_key and not full_config.get("api_key"):
            raise ValidationError(
                message=f"API key is required for provider '{provider}'",
                details={"provider": provider},
            )

        self.config = ProviderConfig(**full_config)

        # 创建适配器
        self._adapter: BaseAdapter = AdapterFactory.create(self.config)

    async def start(self) -> None:
        """启动客户端（初始化适配器）"""
        await self._adapter.start()

    async def close(self) -> None:
        """关闭客户端（释放资源）"""
        await self._adapter.close()

    async def __aenter__(self) -> "Client":
        """异步上下文管理器入口"""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.close()

    # ==================== 文本对话 ====================

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage] | list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        stop: str | list[str] | None = None,
        options: ChatOptions | None = None,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]:
        """
        文本对话

        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度系数（0.0-2.0）
            max_tokens: 最大输出 token 数
            stream: 是否流式输出
            stop: 停止词
            options: 统一对话选项（优先级高于独立参数）
            **kwargs: 其他参数

        Returns:
            stream=False: ChatCompletion 对象
            stream=True: 异步生成器，逐个返回 ChatCompletionChunk
        """
        chat_messages = self._normalize_messages(messages)

        if options is not None:
            option_kwargs = options.to_kwargs()
            option_kwargs.update(kwargs)
            kwargs = option_kwargs

        if stream:
            return self._adapter.chat_stream(
                model=model,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs,
            )

        return await self._adapter.chat(
            model=model,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        )

    # ==================== 图像生成 ====================

    async def image_generate(
        self,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        negative_prompt: str | None = None,
        reference_images: list[str] | None = None,
        mask: str | None = None,
        response_format: Literal["url", "b64_json"] = "url",
        options: ImageOptions | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        图像生成

        Args:
            model: 模型名称
            prompt: 提示词
            size: 图像尺寸（如 "1024x1024"）
            n: 生成数量（1-10）
            negative_prompt: 负面提示词
            reference_images: 参考图像列表（URL 或 base64）
            mask: 蒙版图像（局部编辑用）
            response_format: 响应格式（"url" 或 "b64_json"）
            options: 统一图像选项（优先级高于独立参数）
            **kwargs: 其他参数

        Returns:
            图像生成结果
        """
        if options is not None:
            option_kwargs = options.to_kwargs()
            option_kwargs.update(kwargs)
            kwargs = option_kwargs

        return await self._adapter.image_generate(
            model=model,
            prompt=prompt,
            size=size,
            n=n,
            negative_prompt=negative_prompt,
            reference_images=reference_images,
            mask=mask,
            response_format=response_format,
            **kwargs,
        )

    # ==================== 视频生成 ====================

    async def video_create(
        self,
        model: str,
        prompt: str,
        width: int = 1280,
        height: int = 720,
        num_frames: int | None = None,
        frame_rate: int = 24,
        mode: Literal[
            "text2video", "image2video", "keyframes", "multiimage"
        ] = "text2video",
        reference_images: list[str] | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
        options: VideoOptions | None = None,
        **kwargs: Any,
    ) -> VideoTask:
        """
        创建视频生成任务

        Args:
            model: 模型名称
            prompt: 提示词
            width: 视频宽度（必须是 8 的倍数）
            height: 视频高度（必须是 8 的倍数）
            num_frames: 帧数（部分模型需要）
            frame_rate: 帧率
            mode: 生成模式
            reference_images: 参考图像列表
            negative_prompt: 负面提示词
            seed: 随机种子
            options: 统一视频选项（优先级高于独立参数）
            **kwargs: 其他参数

        Returns:
            视频任务信息
        """
        if options is not None:
            option_kwargs = options.to_kwargs()
            option_kwargs.update(kwargs)
            kwargs = option_kwargs

        return await self._adapter.video_create(
            model=model,
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            mode=mode,
            reference_images=reference_images,
            negative_prompt=negative_prompt,
            seed=seed,
            **kwargs,
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        """
        查询视频任务状态

        Args:
            task_id: 任务 ID
            model: 模型名称（部分 Provider 需要）

        Returns:
            视频任务状态
        """
        return await self._adapter.video_poll(task_id=task_id, model=model)

    # ==================== 文本嵌入 ====================

    async def embed(
        self,
        model: str,
        input: str | list[str],
        options: EmbedOptions | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        文本嵌入

        Args:
            model: 嵌入模型名称
            input: 输入文本或文本列表
            options: 统一嵌入选项
            **kwargs: 其他参数（dimensions, encoding_format 等）

        Returns:
            嵌入结果
        """
        if options is not None:
            option_kwargs = options.to_kwargs()
            option_kwargs.update(kwargs)
            kwargs = option_kwargs

        return await self._adapter.embed(
            model=model,
            input=input,
            **kwargs,
        )

    # ==================== 语音转文字 ====================

    async def transcribe(
        self,
        model: str,
        file: Any,
        language: str | None = None,
        prompt: str | None = None,
        response_format: Literal["json", "text", "srt", "vtt", "verbose_json"] = "json",
        temperature: float | None = None,
        options: TranscribeOptions | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        语音转文字（ASR）

        Args:
            model: 模型名称（如 'whisper-1'）
            file: 音频文件（文件路径、URL、base64 或二进制数据）
            language: 语言代码（如 'zh'、'en'）
            prompt: 提示词（改善专有名词识别）
            response_format: 响应格式
            temperature: 温度系数
            options: 统一转写选项（优先级高于独立参数）
            **kwargs: 其他参数

        Returns:
            转写结果
        """
        if options is not None:
            option_kwargs = options.to_kwargs()
            option_kwargs.update(kwargs)
            kwargs = option_kwargs
            file = kwargs.pop("file", file)
            model = kwargs.pop("model", model)

        return await self._adapter.transcribe(
            model=model,
            file=file,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            **kwargs,
        )

    async def translate(
        self,
        model: str,
        file: Any,
        prompt: str | None = None,
        response_format: Literal["json", "text", "srt", "vtt", "verbose_json"] = "json",
        temperature: float | None = None,
        options: TranscribeOptions | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        语音翻译（翻译为英文）

        Args:
            model: 模型名称
            file: 音频文件
            prompt: 提示词
            response_format: 响应格式
            temperature: 温度系数
            options: 统一转写选项
            **kwargs: 其他参数

        Returns:
            翻译结果（英文）
        """
        if options is not None:
            option_kwargs = options.to_kwargs()
            option_kwargs.update(kwargs)
            kwargs = option_kwargs
            file = kwargs.pop("file", file)
            model = kwargs.pop("model", model)

        return await self._adapter.translate(
            model=model,
            file=file,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            **kwargs,
        )

    # ==================== 文字转语音 ====================

    async def speech(
        self,
        model: str,
        input: str,
        voice: str | list[str],
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3",
        speed: float | None = None,
        options: SpeechOptions | None = None,
        **kwargs: Any,
    ) -> SpeechResult:
        """
        文字转语音（TTS）

        支持 voice 候选列表自动降级（当适配器实现支持时，如 Edge TTS）：
        传入 list[str] 后，第一个音色失败会自动切换到下一个。

        Args:
            model: 模型名称（如 'tts-1'、'tts-1-hd'）
            input: 要合成的文本
            voice: 音色（如 'alloy'、'echo'、'nova'），或候选音色列表用于自动降级
            response_format: 音频输出格式
            speed: 语速（0.25-4.0）
            options: 统一语音合成选项（优先级高于独立参数）
            **kwargs: 其他参数

        Returns:
            语音合成结果（包含音频数据）
        """
        if options is not None:
            option_kwargs = options.to_kwargs()
            option_kwargs.update(kwargs)
            kwargs = option_kwargs
            input = kwargs.pop("input", input)
            model = kwargs.pop("model", model)
            voice = kwargs.pop("voice", voice)

        return await self._adapter.speech(
            model=model,
            input=input,
            voice=voice,
            response_format=response_format,
            speed=speed,
            **kwargs,
        )

    # ==================== 模型信息 ====================

    async def list_models(
        self,
        model_type: Literal["chat", "image", "video", "audio"] | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        Args:
            model_type: 模型类型过滤

        Returns:
            模型信息列表
        """
        return await self._adapter.list_models(model_type=model_type)

    # ==================== 音色信息 ====================

    async def list_voices(
        self,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        获取 Provider 可用音色列表

        用于业务层维护"可用声音池"，避免使用已被 Provider 下线的音色。
        Provider 不支持时抛 UnsupportedCapabilityError。

        Args:
            language: 按语言过滤，如 "zh-CN" / "en-US"（Provider 实现决定匹配规则）

        Returns:
            音色信息列表，每项为 dict，常见字段：ShortName / Name / Locale / Gender
            （具体字段因 Provider 而异）

        Raises:
            UnsupportedCapabilityError: 当前 Provider 不支持 list_voices
        """
        return await self._adapter.list_voices(language=language)

    async def recommend_voices(
        self,
        language: str | None = None,
        gender: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        推荐可用音色（按语言/性别过滤）

        业务层可直接把返回结果中的 ShortName/voice_id 传给 speech() 的 voice 参数，
        无需自己维护可用性逻辑。Provider 不支持时抛 UnsupportedCapabilityError。

        Args:
            language: 按语言过滤，如 "zh-CN" / "en-US"
            gender: 按性别过滤，如 "female" / "male"（不区分大小写）
            limit: 最多返回条数

        Returns:
            推荐音色列表

        Raises:
            UnsupportedCapabilityError: 当前 Provider 不支持 list_voices
        """
        return await self._adapter.recommend_voices(
            language=language,
            gender=gender,
            limit=limit,
        )

    # ==================== 辅助方法 ====================

    def _normalize_messages(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
    ) -> list[ChatMessage]:
        """
        标准化消息格式

        将字典格式的消息转换为 ChatMessage 对象。

        Args:
            messages: 消息列表（可能包含字典）

        Returns:
            ChatMessage 对象列表
        """
        if not messages:
            return []

        # 如果已经是 ChatMessage 对象，直接返回
        if isinstance(messages[0], ChatMessage):
            return messages

        # 转换字典为 ChatMessage
        result: list[ChatMessage] = []
        for msg in messages:
            if isinstance(msg, dict):
                result.append(ChatMessage(**msg))
            else:
                raise ValidationError(
                    message="Invalid message format",
                    details={"message": msg},
                )
        return result
