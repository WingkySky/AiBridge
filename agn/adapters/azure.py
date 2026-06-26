"""
AGN-SDK Azure OpenAI 适配器

实现 Azure OpenAI API 的统一调用，支持对话、图像、语音转文字、文字转语音。
"""

import logging
from typing import Any

import httpx

from agn.adapters.base import Capabilities
from agn.adapters.factory import AdapterFactory
from agn.adapters.openai import OpenAIAdapter
from agn.models.audio import SpeechResult, TranscriptionResult
from agn.models.chat import ChatCompletion, ChatMessage
from agn.models.common import ModelInfo, ProviderConfig
from agn.models.image import ImageGenerationResult

logger = logging.getLogger(__name__)

# Azure OpenAI 默认 API 版本
DEFAULT_API_VERSION = "2024-02-15-preview"


class AzureAdapter(OpenAIAdapter):
    """
    Azure OpenAI 适配器

    基于 OpenAI 适配器扩展，实现 Azure OpenAI API 的统一调用。

    Azure OpenAI API 与 OpenAI API 基本兼容，但有以下差异：
    - Base URL 格式不同: https://{resource}.openai.azure.com/openai/deployments/{deployment}
    - 需要指定 API 版本（通过 query 参数）
    - 模型 ID 通常是部署名称
    - 认证使用 api-key header 而非 Bearer token
    """

    provider_type = "azure"
    provider_name = "Azure OpenAI"
    supported_capabilities = [
        Capabilities.CHAT,
        Capabilities.CHAT_STREAM,
        Capabilities.IMAGE_GENERATE,
        Capabilities.AUDIO_TRANSCRIBE,
        Capabilities.AUDIO_TRANSLATE,
        Capabilities.AUDIO_SPEECH,
    ]

    def __init__(self, config: ProviderConfig) -> None:
        """
        初始化适配器

        Args:
            config: Provider 配置

        配置项说明:
            - api_key: Azure API Key
            - resource_name: Azure 资源名称（必需）
            - deployment_id: 部署 ID（必需）
            - api_version: API 版本（默认 2024-02-15-preview）
            - base_url: 完整的 Base URL（如果提供则忽略 resource_name/deployment_id）
        """
        super().__init__(config)

        # Azure 专用配置
        self.resource_name = config.resource_name
        self.deployment_id = config.deployment_id
        self.api_version = config.api_version or DEFAULT_API_VERSION

        # 构建 Azure 的 Base URL
        if config.base_url:
            self.base_url = config.base_url
        elif self.resource_name and self.deployment_id:
            self.base_url = f"https://{self.resource_name}.openai.azure.com/openai/deployments/{self.deployment_id}"
        else:
            raise ValueError(
                "Azure adapter requires either base_url or both resource_name and deployment_id"
            )

        logger.debug(
            f"Azure adapter initialized: resource={self.resource_name}, deployment={self.deployment_id}"
        )

    async def start(self) -> None:
        """启动适配器"""
        import httpx

        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "api-key": self.api_key,
                "Content-Type": "application/json",
            },
        )

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatCompletion:
        """
        文本对话

        Args:
            model: 模型名称（实际是部署名称）
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            对话完成结果
        """
        # Azure 使用 deployment_id 作为模型名，这里直接使用传入的 model 参数
        # 如果用户没有传入 deployment_id，使用配置中的 deployment_id
        azure_model = model or self.deployment_id
        if not azure_model:
            raise ValueError("Model name is required for Azure")

        logger.debug(f"Azure chat request: model={azure_model}")

        return await super().chat(azure_model, messages, **kwargs)

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        图像生成

        Args:
            model: 模型名称（实际是部署名称）
            prompt: 提示词
            **kwargs: 其他参数

        Returns:
            图像生成结果
        """
        azure_model = model or self.deployment_id
        if not azure_model:
            raise ValueError("Model name is required for Azure")

        logger.debug(f"Azure image request: model={azure_model}")

        return await super().image_generate(azure_model, prompt, **kwargs)

    # ==================== 语音转文字 ====================

    def _get_audio_base_url(self) -> str:
        """
        获取语音 API 的 Base URL

        Azure 语音 API 不使用 deployments 路径，格式为：
        https://{resource}.openai.azure.com/openai
        """
        if self.resource_name:
            return f"https://{self.resource_name}.openai.azure.com/openai"
        return self.base_url.replace(
            f"/openai/deployments/{self.deployment_id}", "/openai"
        )

    async def transcribe(
        self,
        model: str,
        file: Any,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        语音转文字（Azure OpenAI Whisper）

        Args:
            model: 模型名称（whisper-1 或部署名）
            file: 音频文件
            **kwargs: 其他参数

        Returns:
            转写结果
        """
        client = self._get_client()

        audio_file, filename = self._prepare_audio_file(file)

        data: dict[str, Any] = {"model": model}
        for key, value in kwargs.items():
            if value is not None:
                data[key] = str(value) if isinstance(value, bool) else value

        files: Any
        if isinstance(audio_file, str):
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout)
            ) as http:
                resp = await http.get(audio_file)
                resp.raise_for_status()
                files = {"file": (filename, resp.content, "application/octet-stream")}
        elif hasattr(audio_file, "read"):
            files = {"file": (filename, audio_file, "application/octet-stream")}
        else:
            files = {"file": (filename, audio_file, "application/octet-stream")}

        try:
            audio_base_url = self._get_audio_base_url()
            response = await client.post(
                f"{audio_base_url}/audio/transcriptions",
                data=data,
                files=files,
                params={"api-version": self.api_version},
            )
            self._handle_error(response)
            result = response.json()
        finally:
            if hasattr(audio_file, "close"):
                audio_file.close()

        from agn.models.audio import TranscriptionSegment, TranscriptionWord

        segments = []
        words = []

        if result.get("segments"):
            for seg in result["segments"]:
                segments.append(
                    TranscriptionSegment(
                        id=seg.get("id", 0),
                        start=seg.get("start", 0),
                        end=seg.get("end", 0),
                        text=seg.get("text", ""),
                        avg_logprob=seg.get("avg_logprob"),
                        compression_ratio=seg.get("compression_ratio"),
                        no_speech_prob=seg.get("no_speech_prob"),
                        temperature=seg.get("temperature"),
                        tokens=seg.get("tokens"),
                        seek=seg.get("seek"),
                    )
                )

        if result.get("words"):
            for w in result["words"]:
                words.append(
                    TranscriptionWord(
                        word=w.get("word", ""),
                        start=w.get("start", 0),
                        end=w.get("end", 0),
                    )
                )

        return TranscriptionResult(
            text=result.get("text", ""),
            language=result.get("language"),
            duration=result.get("duration"),
            segments=segments if segments else None,
            words=words if words else None,
            task=result.get("task", "transcribe"),
            usage=result.get("usage"),
            model=result.get("model", model),
        )

    async def translate(
        self,
        model: str,
        file: Any,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        语音翻译（Azure OpenAI Whisper）

        Args:
            model: 模型名称
            file: 音频文件
            **kwargs: 其他参数

        Returns:
            翻译结果（英文）
        """
        client = self._get_client()

        audio_file, filename = self._prepare_audio_file(file)

        data: dict[str, Any] = {"model": model}
        for key, value in kwargs.items():
            if value is not None:
                data[key] = str(value) if isinstance(value, bool) else value

        files: Any
        if isinstance(audio_file, str):
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout)
            ) as http:
                resp = await http.get(audio_file)
                resp.raise_for_status()
                files = {"file": (filename, resp.content, "application/octet-stream")}
        elif hasattr(audio_file, "read"):
            files = {"file": (filename, audio_file, "application/octet-stream")}
        else:
            files = {"file": (filename, audio_file, "application/octet-stream")}

        try:
            audio_base_url = self._get_audio_base_url()
            response = await client.post(
                f"{audio_base_url}/audio/translations",
                data=data,
                files=files,
                params={"api-version": self.api_version},
            )
            self._handle_error(response)
            result = response.json()
        finally:
            if hasattr(audio_file, "close"):
                audio_file.close()

        return TranscriptionResult(
            text=result.get("text", ""),
            language="en",
            duration=result.get("duration"),
            segments=None,
            words=None,
            task="translate",
            usage=result.get("usage"),
            model=result.get("model", model),
        )

    async def speech(
        self,
        model: str,
        input: str,
        voice: str | list[str],
        **kwargs: Any,
    ) -> SpeechResult:
        """
        文字转语音（Azure OpenAI TTS）

        Args:
            model: 模型名称（tts-1, tts-1-hd 或部署名）
            input: 要合成的文本
            voice: 音色
            **kwargs: 其他参数

        Returns:
            语音合成结果
        """
        # 非 EdgeTTS 适配器不实现 voice list fallback，收到列表时取第一个元素
        if isinstance(voice, list):
            voice = voice[0] if voice else ""
        client = self._get_client()

        payload: dict[str, Any] = {
            "model": model,
            "input": input,
            "voice": voice,
        }

        response_format = kwargs.pop("response_format", "mp3")
        payload["response_format"] = response_format

        speed = kwargs.pop("speed", None)
        if speed is not None:
            payload["speed"] = speed

        for key, value in kwargs.items():
            if value is not None and key not in ("volume", "pitch", "emotion", "style"):
                payload[key] = value

        audio_base_url = self._get_audio_base_url()
        response = await client.post(
            f"{audio_base_url}/audio/speech",
            json=payload,
            params={"api-version": self.api_version},
        )
        self._handle_error(response)

        content_type_map = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }

        return SpeechResult(
            audio_data=response.content,
            content_type=content_type_map.get(response_format, "audio/mpeg"),
            format=response_format,
            model=model,
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        Azure 的模型列表取决于部署，这里返回常见的 Azure 部署模型。

        Args:
            model_type: 模型类型过滤

        Returns:
            模型信息列表
        """
        models = [
            # 文本对话模型
            ModelInfo(
                id="gpt-4o",
                name="GPT-4o",
                type="chat",
                provider="azure",
                capabilities=["chat"],
                max_tokens=128000,
                supports_streaming=True,
            ),
            ModelInfo(
                id="gpt-4",
                name="GPT-4",
                type="chat",
                provider="azure",
                capabilities=["chat"],
                max_tokens=8192,
                supports_streaming=True,
            ),
            ModelInfo(
                id="gpt-35-turbo",
                name="GPT-3.5 Turbo",
                type="chat",
                provider="azure",
                capabilities=["chat"],
                max_tokens=16384,
                supports_streaming=True,
            ),
            # 图像生成模型
            ModelInfo(
                id="dall-e-3",
                name="DALL-E 3",
                type="image",
                provider="azure",
                capabilities=["text2image", "image2image"],
            ),
            # 语音转文字模型
            ModelInfo(
                id="whisper",
                name="Azure Whisper",
                type="audio",
                provider="azure",
                capabilities=["audio_transcribe", "audio_translate"],
            ),
            # 文字转语音模型
            ModelInfo(
                id="tts",
                name="Azure TTS",
                type="audio",
                provider="azure",
                capabilities=["audio_speech"],
            ),
            ModelInfo(
                id="tts-hd",
                name="Azure TTS HD",
                type="audio",
                provider="azure",
                capabilities=["audio_speech"],
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    # ==================== 辅助方法 ====================

    def _get_client(self) -> Any:
        """获取 HTTP 客户端"""
        if self._http_client is None:
            raise RuntimeError("Adapter not started. Call start() first.")
        return self._http_client


# 注册适配器
AdapterFactory.register("azure", AzureAdapter)
