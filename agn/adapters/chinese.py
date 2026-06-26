"""
AGN-SDK 中文模型适配器

实现中文 AI 模型 API 的统一调用。
支持：通义千问（Qwen）、智谱（GLM）、文心一言（ERNIE）、豆包（Doubao）、MiniMax、Kimi 等。

这些模型的特点：
- 原生中文优化
- 支持更长的上下文窗口
- 统一的 OpenAI-compatible API 或各自 API
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx

from agn.adapters.base import BaseAdapter, Capabilities
from agn.adapters.factory import AdapterFactory
from agn.core.errors import (
    APIError,
    AuthenticationError,
    RateLimitError,
    UnsupportedCapabilityError,
)
from agn.core.utils import current_timestamp, generate_id
from agn.models.audio import (
    SpeechResult,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)
from agn.models.chat import (
    ChatChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChatMessage,
    ChatUsage,
)
from agn.models.common import ModelInfo, ProviderConfig
from agn.models.image import ImageGenerationResult
from agn.models.video import VideoStatus, VideoTask

logger = logging.getLogger(__name__)


# ==================== 通用 OpenAI 兼容语音支持 Mixin ====================


class OpenAICompatibleAudioMixin(ABC):
    """
    OpenAI 兼容语音能力 Mixin

    为使用标准 OpenAI 音频 API 路径的适配器提供 transcribe 和 speech 方法。
    需要适配器实现：
    - _get_client() -> httpx.AsyncClient
    - _handle_error(response) -> 错误处理
    """

    config: ProviderConfig

    @abstractmethod
    def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端"""
        ...

    @abstractmethod
    def _handle_error(self, response: httpx.Response) -> None:
        """处理错误响应"""
        ...

    def _prepare_audio_file(self, file: Any) -> tuple[Any, str]:
        """
        准备音频文件用于上传

        Args:
            file: 文件路径、URL、base64 或二进制数据

        Returns:
            (文件对象或数据, 文件名)

        Raises:
            ValueError: 不支持的文件类型
        """
        import base64
        from pathlib import Path

        if isinstance(file, (str, Path)):
            if isinstance(file, str) and (
                file.startswith("http://") or file.startswith("https://")
            ):
                return file, "audio.mp3"
            path = Path(file)
            if path.exists():
                return open(path, "rb"), path.name
            if isinstance(file, str) and file.startswith("data:"):
                header, encoded = file.split(",", 1)
                ext = "wav" if "wav" in header else "mp3"
                return base64.b64decode(encoded), f"audio.{ext}"
            if isinstance(file, str):
                try:
                    decoded = base64.b64decode(file, validate=True)
                    if len(decoded) > 10:
                        return decoded, "audio.wav"
                except Exception:
                    pass
            raise ValueError(f"File path does not exist or invalid input: {file!r}")
        elif isinstance(file, bytes):
            return file, "audio.wav"
        elif hasattr(file, "read"):
            name = getattr(file, "name", "audio.mp3")
            basename = os.path.basename(name) if isinstance(name, str) else "audio.mp3"
            return file, basename
        else:
            raise ValueError(f"Unsupported file type: {type(file)}")

    async def get_audio_bytes(self, file: Any) -> tuple[bytes, str]:
        """
        将各种音频输入统一转换为 bytes 和文件名

        Args:
            file: 文件路径、URL、base64、bytes 或类文件对象

        Returns:
            (音频二进制数据, 文件名)

        Raises:
            ValueError: 不支持的文件类型
        """
        import base64
        from pathlib import Path

        if isinstance(file, bytes):
            return file, "audio.wav"

        if isinstance(file, (str, Path)):
            if isinstance(file, str) and (
                file.startswith("http://") or file.startswith("https://")
            ):
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.config.timeout)
                ) as http:
                    resp = await http.get(file)
                    resp.raise_for_status()
                    return resp.content, "audio.mp3"

            path = Path(file)
            if path.exists():
                return path.read_bytes(), path.name

            if isinstance(file, str) and file.startswith("data:"):
                header, encoded = file.split(",", 1)
                ext = "wav" if "wav" in header else "mp3"
                return base64.b64decode(encoded), f"audio.{ext}"

            if isinstance(file, str):
                try:
                    decoded = base64.b64decode(file, validate=True)
                    if len(decoded) > 10:
                        return decoded, "audio.wav"
                except Exception:
                    pass

            raise ValueError(f"Unsupported file input: {file!r}")

        if hasattr(file, "read"):
            pos = file.tell() if hasattr(file, "tell") else 0
            data = file.read()
            if hasattr(file, "seek"):
                file.seek(pos)
            if isinstance(data, str):
                data = data.encode("utf-8")
            name = getattr(file, "name", "audio.mp3")
            basename = os.path.basename(name) if isinstance(name, str) else "audio.mp3"
            return data, basename

        raise ValueError(f"Unsupported file type: {type(file)}")

    async def transcribe(
        self,
        model: str,
        file: Any,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        语音转文字（OpenAI 兼容端点）

        Args:
            model: 模型名称
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
            response = await client.post(
                "/audio/transcriptions",
                data=data,
                files=files,
            )
            self._handle_error(response)
            result = response.json()
        finally:
            if hasattr(audio_file, "close"):
                audio_file.close()

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
        语音翻译（翻译为英文，OpenAI 兼容端点 /audio/translations）

        Args:
            model: 模型名称
            file: 音频文件
            **kwargs: 其他参数

        Returns:
            翻译结果（目标语言始终为英文）
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
            response = await client.post(
                "/audio/translations",
                data=data,
                files=files,
            )
            self._handle_error(response)
            result = response.json()
        finally:
            if hasattr(audio_file, "close"):
                audio_file.close()

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
            language="en",
            duration=result.get("duration"),
            segments=segments if segments else None,
            words=words if words else None,
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
        文字转语音（OpenAI 兼容端点）

        Args:
            model: 模型名称
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

        response = await client.post("/audio/speech", json=payload)
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


class QwenAdapter(OpenAICompatibleAudioMixin, BaseAdapter):
    """
    通义千问（Qwen）适配器

    通义千问 API 参考（DashScope OpenAI 兼容模式）：
    - Chat: POST /compatible-mode/v1/chat/completions
    - 语音: POST /compatible-mode/v1/audio/transcriptions, /compatible-mode/v1/audio/speech (SenseVoice, CosyVoice)
    - 认证: DashScope API Key
    """

    provider_type = "qwen"
    provider_name = "通义千问"
    supported_capabilities = [
        Capabilities.CHAT,
        Capabilities.CHAT_STREAM,
        Capabilities.VISION,
        Capabilities.AUDIO_TRANSCRIBE,
        Capabilities.AUDIO_SPEECH,
    ]

    # 通义千问默认 Base URL
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, config: ProviderConfig) -> None:
        """
        初始化适配器

        Args:
            config: Provider 配置
        """
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """启动适配器"""
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        """关闭适配器"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端"""
        if self._http_client is None:
            raise RuntimeError("Adapter not started. Call start() first.")
        return self._http_client

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatCompletion:
        """
        文本对话

        Args:
            model: 模型名称（qwen-turbo, qwen-plus, qwen-max, qwen-vl-max 等）
            messages: 消息列表
            **kwargs: 其他参数
                - temperature: 温度
                - max_tokens: 最大 token 数
                - top_p: top_p 采样
                - stream: 是否流式输出
                - seed: 随机种子
                - stop: 停止词列表

        Returns:
            对话完成结果
        """
        client = self._get_client()

        # 构建请求体
        body: dict[str, Any] = {
            "model": model,
            "messages": [msg.model_dump() for msg in messages],
        }

        # 可选参数
        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens
        if top_p := kwargs.get("top_p"):
            body["top_p"] = top_p
        if seed := kwargs.get("seed"):
            body["seed"] = seed
        if stop := kwargs.get("stop"):
            body["stop"] = stop

        # 透传额外参数
        extra_body = kwargs.get("extra_body")
        if extra_body and isinstance(extra_body, dict):
            body.update(extra_body)

        logger.debug(f"Sending chat request to Qwen: model={model}")

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"Qwen chat failed: {e}")
            raise

        self._handle_error(response)
        return self._parse_response(response.json(), model)

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        """
        流式文本对话

        Args:
            model: 模型名称
            messages: 消息列表
            **kwargs: 其他参数

        Yields:
            ChatCompletionChunk: 流式响应块
        """
        client = self._get_client()

        body: dict[str, Any] = {
            "model": model,
            "messages": [msg.model_dump() for msg in messages],
            "stream": True,
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens

        try:
            async with client.stream(
                "POST", "/chat/completions", json=body
            ) as response:
                self._handle_error(response)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        chunk = self._parse_chunk(data, model)
                        if chunk:
                            yield chunk
        except Exception as e:
            logger.error(f"Qwen stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        图像生成

        通义千问 VL 模型支持图像输入，但不支持图像生成。
        此方法抛出错误。
        """
        raise UnsupportedCapabilityError(
            message="Qwen does not support direct image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> VideoTask:
        """视频生成（不支持）"""
        raise UnsupportedCapabilityError(
            message="Qwen does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        """视频状态（不支持）"""
        raise UnsupportedCapabilityError(
            message="Qwen does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """获取可用模型列表"""
        models = [
            ModelInfo(
                id="qwen-turbo",
                name="Qwen Turbo",
                type="chat",
                provider="qwen",
                capabilities=["chat", "vision"],
                description="快速响应版本",
            ),
            ModelInfo(
                id="qwen-plus",
                name="Qwen Plus",
                type="chat",
                provider="qwen",
                capabilities=["chat", "vision"],
                description="高性能版本",
            ),
            ModelInfo(
                id="qwen-max",
                name="Qwen Max",
                type="chat",
                provider="qwen",
                capabilities=["chat", "vision"],
                description="最强性能版本",
            ),
            ModelInfo(
                id="qwen-vl-max",
                name="Qwen VL Max",
                type="chat",
                provider="qwen",
                capabilities=["chat", "vision"],
                description="视觉理解最强版本",
            ),
            # 语音转文字模型（SenseVoice）
            ModelInfo(
                id="sensevoice-v1",
                name="SenseVoice",
                type="audio",
                provider="qwen",
                capabilities=["audio_transcribe"],
                description="阿里通义 SenseVoice 多语言语音识别",
            ),
            # 文字转语音模型（CosyVoice）
            ModelInfo(
                id="cosyvoice-v1",
                name="CosyVoice",
                type="audio",
                provider="qwen",
                capabilities=["audio_speech"],
                description="阿里通义 CosyVoice 语音合成",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    def _handle_error(self, response: httpx.Response) -> None:
        """处理错误响应"""
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Qwen API key")
        if response.status_code == 429:
            raise RateLimitError(message="Qwen rate limit exceeded")

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("error", {}).get("message")
                or error_data.get("message")
                or error_data.get("error")
                or f"HTTP {response.status_code}"
            )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=error_msg, status_code=response.status_code)

    def _parse_response(self, data: dict[str, Any], model: str) -> ChatCompletion:
        """解析对话响应"""
        choices = data.get("choices", [])
        if not choices:
            raise APIError(message="No completion choices in response")

        choice = choices[0]
        message_data = choice.get("message", {})
        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=[
                ChatChoice(
                    index=choice.get("index", 0),
                    message=ChatMessage(
                        role=message_data.get("role", "assistant"),
                        content=message_data.get("content", ""),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            ],
            usage=usage,
        )

    def _parse_chunk(self, data_str: str, model: str) -> ChatCompletionChunk | None:
        """解析流式响应块"""
        import json

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return None

        choices = data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta_data = choice.get("delta", {})
        delta_message = ChatMessage(
            role=delta_data.get("role", "assistant"),
            content=delta_data.get("content", ""),
        )
        return ChatCompletionChunk(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=[
                ChatCompletionDelta(
                    index=choice.get("index", 0),
                    delta=delta_message,
                    finish_reason=choice.get("finish_reason"),
                )
            ],
        )


class ZhipuAdapter(BaseAdapter):
    """
    智谱 AI（GLM）适配器

    智谱 API 参考：
    - Chat: POST /v4/chat/completions
    - 认证: Zhipu API Key
    """

    provider_type = "zhipu"
    provider_name = "智谱 AI"
    supported_capabilities = ["chat", "vision"]

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError("Adapter not started. Call start() first.")
        return self._http_client

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatCompletion:
        """文本对话"""
        client = self._get_client()

        body: dict[str, Any] = {
            "model": model,
            "messages": [msg.model_dump() for msg in messages],
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"Zhipu chat failed: {e}")
            raise

        self._handle_error(response)
        return self._parse_response(response.json(), model)

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        """流式文本对话"""
        client = self._get_client()

        body: dict[str, Any] = {
            "model": model,
            "messages": [msg.model_dump() for msg in messages],
            "stream": True,
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature

        try:
            async with client.stream(
                "POST", "/chat/completions", json=body
            ) as response:
                self._handle_error(response)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        chunk = self._parse_chunk(data, model)
                        if chunk:
                            yield chunk
        except Exception as e:
            logger.error(f"Zhipu stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        raise UnsupportedCapabilityError(
            message="Zhipu does not support direct image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> VideoTask:
        raise UnsupportedCapabilityError(
            message="Zhipu does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        raise UnsupportedCapabilityError(
            message="Zhipu does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        models = [
            ModelInfo(
                id="glm-4",
                name="GLM-4",
                type="chat",
                provider="zhipu",
                capabilities=["chat", "vision"],
                description="智谱 GLM-4 对话模型",
            ),
            ModelInfo(
                id="glm-4v",
                name="GLM-4V",
                type="chat",
                provider="zhipu",
                capabilities=["chat", "vision"],
                description="智谱 GLM-4V 视觉模型",
            ),
            ModelInfo(
                id="glm-3-turbo",
                name="GLM-3 Turbo",
                type="chat",
                provider="zhipu",
                capabilities=["chat"],
                description="智谱 GLM-3 Turbo 快速版本",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Zhipu API key")
        if response.status_code == 429:
            raise RateLimitError(message="Zhipu rate limit exceeded")

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("error", {}).get("message")
                or error_data.get("message")
                or error_data.get("error")
                or f"HTTP {response.status_code}"
            )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=error_msg, status_code=response.status_code)

    def _parse_response(self, data: dict[str, Any], model: str) -> ChatCompletion:
        choices = data.get("choices", [])
        if not choices:
            raise APIError(message="No completion choices in response")

        choice = choices[0]
        message_data = choice.get("message", {})
        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=[
                ChatChoice(
                    index=choice.get("index", 0),
                    message=ChatMessage(
                        role=message_data.get("role", "assistant"),
                        content=message_data.get("content", ""),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            ],
            usage=usage,
        )

    def _parse_chunk(self, data_str: str, model: str) -> ChatCompletionChunk | None:
        """解析流式响应块"""
        import json

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return None

        choices = data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta_data = choice.get("delta", {})
        delta_message = ChatMessage(
            role=delta_data.get("role", "assistant"),
            content=delta_data.get("content", ""),
        )
        return ChatCompletionChunk(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=[
                ChatCompletionDelta(
                    index=choice.get("index", 0),
                    delta=delta_message,
                    finish_reason=choice.get("finish_reason"),
                )
            ],
        )


class DoubaoAdapter(OpenAICompatibleAudioMixin, BaseAdapter):
    """
    豆包（字节跳动火山引擎）适配器

    豆包 API 参考（OpenAI 兼容）：
    - Chat: POST /api/v3/chat/completions
    - 语音: POST /api/v3/audio/transcriptions, /api/v3/audio/speech
    - 认证: Bearer Token (火山引擎 API Key)
    - 文档: https://www.volcengine.com/docs/82379
    """

    provider_type = "doubao"
    provider_name = "豆包"
    supported_capabilities = [
        Capabilities.CHAT,
        Capabilities.CHAT_STREAM,
        Capabilities.VISION,
        Capabilities.AUDIO_TRANSCRIBE,
        Capabilities.AUDIO_SPEECH,
    ]

    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError("Adapter not started. Call start() first.")
        return self._http_client

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatCompletion:
        client = self._get_client()

        body: dict[str, Any] = {
            "model": model,
            "messages": [msg.model_dump() for msg in messages],
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"Doubao chat failed: {e}")
            raise

        self._handle_error(response)
        return self._parse_response(response.json(), model)

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        client = self._get_client()

        body: dict[str, Any] = {
            "model": model,
            "messages": [msg.model_dump() for msg in messages],
            "stream": True,
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature

        try:
            async with client.stream(
                "POST", "/chat/completions", json=body
            ) as response:
                self._handle_error(response)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        chunk = self._parse_chunk(data, model)
                        if chunk:
                            yield chunk
        except Exception as e:
            logger.error(f"Doubao stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        raise UnsupportedCapabilityError(
            message="Doubao does not support direct image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> VideoTask:
        raise UnsupportedCapabilityError(
            message="Doubao does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        raise UnsupportedCapabilityError(
            message="Doubao does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        models = [
            # 豆包 Pro 系列
            ModelInfo(
                id="doubao-pro-4k",
                name="Doubao Pro 4K",
                type="chat",
                provider="doubao",
                capabilities=["chat", "vision"],
                description="豆包 Pro 4K 上下文",
            ),
            ModelInfo(
                id="doubao-pro-32k",
                name="Doubao Pro 32K",
                type="chat",
                provider="doubao",
                capabilities=["chat", "vision"],
                description="豆包 Pro 32K 上下文",
            ),
            ModelInfo(
                id="doubao-pro-128k",
                name="Doubao Pro 128K",
                type="chat",
                provider="doubao",
                capabilities=["chat", "vision"],
                description="豆包 Pro 128K 上下文",
            ),
            ModelInfo(
                id="doubao-pro-256k",
                name="Doubao Pro 256K",
                type="chat",
                provider="doubao",
                capabilities=["chat", "vision"],
                description="豆包 Pro 256K 上下文",
            ),
            # 豆包 Lite 系列
            ModelInfo(
                id="doubao-lite-4k",
                name="Doubao Lite 4K",
                type="chat",
                provider="doubao",
                capabilities=["chat"],
                description="豆包 Lite 快速版本 4K",
            ),
            ModelInfo(
                id="doubao-lite-32k",
                name="Doubao Lite 32K",
                type="chat",
                provider="doubao",
                capabilities=["chat"],
                description="豆包 Lite 快速版本 32K",
            ),
            ModelInfo(
                id="doubao-lite-128k",
                name="Doubao Lite 128K",
                type="chat",
                provider="doubao",
                capabilities=["chat"],
                description="豆包 Lite 快速版本 128K",
            ),
            # Doubao Seed 系列 (旗舰 Agent 模型)
            ModelInfo(
                id="doubao-seed-2-0-lite-260215",
                name="Doubao Seed 2.0 Lite",
                type="chat",
                provider="doubao",
                capabilities=["chat", "vision"],
                description="豆包 Seed 2.0 Lite，支持深度思考",
            ),
            ModelInfo(
                id="doubao-seed-2-0-pro-260215",
                name="Doubao Seed 2.0 Pro",
                type="chat",
                provider="doubao",
                capabilities=["chat", "vision"],
                description="豆包 Seed 2.0 Pro，旗舰 Agent 模型",
            ),
            ModelInfo(
                id="doubao-seed-2-0-mini-260428",
                name="Doubao Seed 2.0 Mini",
                type="chat",
                provider="doubao",
                capabilities=["chat", "vision"],
                description="豆包 Seed 2.0 Mini",
            ),
            ModelInfo(
                id="doubao-2-1-pro",
                name="Doubao 2.1 Pro",
                type="chat",
                provider="doubao",
                capabilities=["chat", "vision"],
                description="豆包 2.1 Pro 最新模型",
            ),
            # 语音转文字模型
            ModelInfo(
                id="doubao-asr",
                name="豆包语音识别",
                type="audio",
                provider="doubao",
                capabilities=["audio_transcribe"],
                description="豆包语音识别模型（中文优化）",
            ),
            # 文字转语音模型
            ModelInfo(
                id="doubao-tts",
                name="豆包语音合成",
                type="audio",
                provider="doubao",
                capabilities=["audio_speech"],
                description="豆包语音合成模型",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Doubao API key")
        if response.status_code == 429:
            raise RateLimitError(message="Doubao rate limit exceeded")

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("error", {}).get("message")
                or error_data.get("message")
                or error_data.get("error")
                or f"HTTP {response.status_code}"
            )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=error_msg, status_code=response.status_code)

    def _parse_response(self, data: dict[str, Any], model: str) -> ChatCompletion:
        choices = data.get("choices", [])
        if not choices:
            raise APIError(message="No completion choices in response")

        choice = choices[0]
        message_data = choice.get("message", {})
        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=[
                ChatChoice(
                    index=choice.get("index", 0),
                    message=ChatMessage(
                        role=message_data.get("role", "assistant"),
                        content=message_data.get("content", ""),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            ],
            usage=usage,
        )

    def _parse_chunk(self, data_str: str, model: str) -> ChatCompletionChunk | None:
        """解析流式响应块"""
        import json

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return None

        choices = data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta_data = choice.get("delta", {})
        delta_message = ChatMessage(
            role=delta_data.get("role", "assistant"),
            content=delta_data.get("content", ""),
        )
        return ChatCompletionChunk(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=[
                ChatCompletionDelta(
                    index=choice.get("index", 0),
                    delta=delta_message,
                    finish_reason=choice.get("finish_reason"),
                )
            ],
        )


class ErnieAdapter(BaseAdapter):
    """
    文心一言（百度 ERNIE）适配器

    文心一言 API 参考：
    - Chat: POST /rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}
    - 认证: access_token 查询参数（通过 API Key + Secret Key 获取）
    - 文档: https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html
    """

    provider_type = "ernie"
    provider_name = "文心一言"
    supported_capabilities = ["chat", "vision"]

    DEFAULT_BASE_URL = "https://aip.baidubce.com"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        # api_key 可以直接是 access_token，也可以是 ak:sk 格式
        self.secret_key = ""
        if ":" in self.api_key:
            ak, sk = self.api_key.split(":", 1)
            self.api_key = ak
            self.secret_key = sk
        self._access_token: str | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers={"Content-Type": "application/json"},
        )
        # 如果配置了 ak:sk，自动获取 access_token
        if self.secret_key:
            await self._get_access_token()

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError("Adapter not started. Call start() first.")
        return self._http_client

    async def _get_access_token(self) -> str:
        """获取 access_token"""
        client = self._get_client()
        response = await client.get(
            "/oauth/2.0/token",
            params={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            },
        )
        data = response.json()
        self._access_token = data.get("access_token")
        return self._access_token or ""

    def _convert_messages(
        self, messages: list[ChatMessage] | list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        转换消息格式到 ERNIE 格式
        ERNIE 使用 messages 数组，role 为 user/assistant，system 单独字段
        """
        ernie_messages: list[dict[str, Any]] = []
        system: str | None = None

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = msg.role
                content = msg.content

            if role == "system":
                system = content if isinstance(content, str) else str(content)
            else:
                ernie_messages.append({"role": role, "content": str(content)})

        return ernie_messages, system

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatCompletion:
        client = self._get_client()

        ernie_messages, system = self._convert_messages(messages)

        body: dict[str, Any] = {"messages": ernie_messages}
        if system:
            body["system"] = system
        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if top_p := kwargs.get("top_p"):
            body["top_p"] = top_p

        # 获取 access_token
        access_token = self._access_token or self.api_key

        try:
            response = await client.post(
                f"/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}",
                params={"access_token": access_token},
                json=body,
            )
        except Exception as e:
            logger.error(f"ERNIE chat failed: {e}")
            raise

        self._handle_error(response)
        return self._parse_response(response.json(), model)

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        client = self._get_client()

        ernie_messages, system = self._convert_messages(messages)

        body: dict[str, Any] = {"messages": ernie_messages, "stream": True}
        if system:
            body["system"] = system

        access_token = self._access_token or self.api_key

        try:
            async with client.stream(
                "POST",
                f"/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}",
                params={"access_token": access_token},
                json=body,
            ) as response:
                self._handle_error(response)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        chunk = self._parse_chunk(data, model)
                        if chunk:
                            yield chunk
        except Exception as e:
            logger.error(f"ERNIE stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        raise UnsupportedCapabilityError(
            message="ERNIE does not support direct image generation (use ERNIE-ViLG separately)",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> VideoTask:
        raise UnsupportedCapabilityError(
            message="ERNIE does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        raise UnsupportedCapabilityError(
            message="ERNIE does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        models = [
            ModelInfo(
                id="completions_pro",
                name="ERNIE 4.0",
                type="chat",
                provider="ernie",
                capabilities=["chat", "vision"],
                description="文心一言 4.0",
            ),
            ModelInfo(
                id="completions",
                name="ERNIE 3.5",
                type="chat",
                provider="ernie",
                capabilities=["chat"],
                description="文心一言 3.5",
            ),
            ModelInfo(
                id="ernie-lite-8k",
                name="ERNIE Lite",
                type="chat",
                provider="ernie",
                capabilities=["chat"],
                description="文心一言轻量版",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401 or response.status_code == 403:
            raise AuthenticationError(message="Invalid ERNIE API key or access_token")
        if response.status_code == 429:
            raise RateLimitError(message="ERNIE rate limit exceeded")

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("error_msg")
                or error_data.get("message")
                or error_data.get("error")
                or f"HTTP {response.status_code}"
            )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=error_msg, status_code=response.status_code)

    def _parse_response(self, data: dict[str, Any], model: str) -> ChatCompletion:
        result = data.get("result", "")
        usage_data = data.get("usage", {})
        usage = ChatUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=current_timestamp(),
            model=model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=result),
                    finish_reason="stop" if not data.get("is_truncated") else "length",
                )
            ],
            usage=usage,
        )

    def _parse_chunk(self, data_str: str, model: str) -> ChatCompletionChunk | None:
        """解析流式响应块"""
        import json

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return None

        result = data.get("result", "")
        is_end = data.get("is_end", False)
        delta_message = ChatMessage(role="assistant", content=result)

        return ChatCompletionChunk(
            id=data.get("id", generate_id("chatcmpl")),
            created=current_timestamp(),
            model=model,
            choices=[
                ChatCompletionDelta(
                    index=0,
                    delta=delta_message,
                    finish_reason="stop" if is_end else None,
                )
            ],
        )


class KimiAdapter(BaseAdapter):
    """
    Kimi (月之暗面 Moonshot AI) 适配器

    Kimi API 参考（OpenAI 兼容）：
    - Chat: POST /v1/chat/completions
    - 认证: Bearer Token (Moonshot API Key)
    - Base URL: https://api.moonshot.cn/v1
    - 文档: https://platform.moonshot.cn/docs/api/chat
    - 特点: 支持超长上下文（128K/256K）、视觉理解
    """

    provider_type = "kimi"
    provider_name = "Kimi (月之暗面)"
    supported_capabilities = ["chat", "vision"]

    DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError("Adapter not started. Call start() first.")
        return self._http_client

    def _convert_messages(
        self, messages: list[ChatMessage] | list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """转换消息格式，提取 system prompt"""
        converted: list[dict[str, Any]] = []
        system_prompt: str | None = None

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = msg.role
                content = msg.content

            if role == "system":
                system_prompt = content if isinstance(content, str) else str(content)
            else:
                converted.append({"role": role, "content": content})

        return converted, system_prompt

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatCompletion:
        client = self._get_client()
        converted_messages, system_prompt = self._convert_messages(messages)

        body: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
        }

        if system_prompt:
            body["messages"].insert(0, {"role": "system", "content": system_prompt})
        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens
        if top_p := kwargs.get("top_p"):
            body["top_p"] = top_p

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"Kimi chat failed: {e}")
            raise

        self._handle_error(response)
        return self._parse_response(response.json(), model)

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        client = self._get_client()
        converted_messages, system_prompt = self._convert_messages(messages)

        body: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "stream": True,
        }

        if system_prompt:
            body["messages"].insert(0, {"role": "system", "content": system_prompt})
        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature

        try:
            async with client.stream(
                "POST", "/chat/completions", json=body
            ) as response:
                self._handle_error(response)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        chunk = self._parse_chunk(data, model)
                        if chunk:
                            yield chunk
        except Exception as e:
            logger.error(f"Kimi stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        raise UnsupportedCapabilityError(
            message="Kimi does not support direct image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> VideoTask:
        raise UnsupportedCapabilityError(
            message="Kimi does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        raise UnsupportedCapabilityError(
            message="Kimi does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        models = [
            ModelInfo(
                id="moonshot-v1-8k",
                name="Moonshot v1 8K",
                type="chat",
                provider="kimi",
                capabilities=["chat"],
                description="Kimi 8K 上下文版本",
            ),
            ModelInfo(
                id="moonshot-v1-32k",
                name="Moonshot v1 32K",
                type="chat",
                provider="kimi",
                capabilities=["chat"],
                description="Kimi 32K 上下文版本",
            ),
            ModelInfo(
                id="moonshot-v1-128k",
                name="Moonshot v1 128K",
                type="chat",
                provider="kimi",
                capabilities=["chat"],
                description="Kimi 128K 长上下文版本",
            ),
            ModelInfo(
                id="kimi-k2.5",
                name="Kimi K2.5",
                type="chat",
                provider="kimi",
                capabilities=["chat", "vision"],
                description="Kimi K2.5 最新旗舰模型，支持视觉",
            ),
            ModelInfo(
                id="kimi-k2.6",
                name="Kimi K2.6",
                type="chat",
                provider="kimi",
                capabilities=["chat", "vision"],
                description="Kimi K2.6 最新模型，256K上下文",
            ),
            ModelInfo(
                id="kimi-k2.7-code",
                name="Kimi K2.7 Code",
                type="chat",
                provider="kimi",
                capabilities=["chat", "vision"],
                description="Kimi K2.7 Code 代码专用模型",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Kimi API key")
        if response.status_code == 429:
            raise RateLimitError(message="Kimi rate limit exceeded")

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("error", {}).get("message")
                or error_data.get("message")
                or error_data.get("error")
                or f"HTTP {response.status_code}"
            )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=error_msg, status_code=response.status_code)

    def _parse_response(self, data: dict[str, Any], model: str) -> ChatCompletion:
        choices = data.get("choices", [])
        if not choices:
            raise APIError(message="No completion choices in response")

        choice = choices[0]
        message_data = choice.get("message", {})
        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=[
                ChatChoice(
                    index=choice.get("index", 0),
                    message=ChatMessage(
                        role=message_data.get("role", "assistant"),
                        content=message_data.get("content", ""),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            ],
            usage=usage,
        )

    def _parse_chunk(self, data_str: str, model: str) -> ChatCompletionChunk | None:
        """解析流式响应块"""
        import json

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return None

        choices = data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta_data = choice.get("delta", {})
        delta_message = ChatMessage(
            role=delta_data.get("role", "assistant"),
            content=delta_data.get("content", ""),
        )
        return ChatCompletionChunk(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=[
                ChatCompletionDelta(
                    index=choice.get("index", 0),
                    delta=delta_message,
                    finish_reason=choice.get("finish_reason"),
                )
            ],
        )


class MiniMaxAdapter(OpenAICompatibleAudioMixin, BaseAdapter):
    """
    MiniMax (稀宇科技) 适配器

    MiniMax API 参考（OpenAI 兼容）：
    - Chat: POST /v1/text/chatcompletion_v2（OpenAI兼容接口）
    - 语音: POST /v1/audio/transcriptions, /v1/audio/speech
    - 认证: Bearer Token (MiniMax API Key)
    - Base URL: https://api.minimaxi.com/v1
    - 文档: https://platform.minimaxi.com/
    - 支持模型: abab 系列、MiniMax-Text-01、MiniMax-VL-01、MiniMax-M1、abab-speech 系列
    """

    provider_type = "minimax"
    provider_name = "MiniMax"
    supported_capabilities = [
        Capabilities.CHAT,
        Capabilities.CHAT_STREAM,
        Capabilities.VISION,
        Capabilities.AUDIO_TRANSCRIBE,
        Capabilities.AUDIO_SPEECH,
    ]

    DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self.group_id: str | None = getattr(config, "group_id", None)
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.group_id:
            headers["X-Group-Id"] = self.group_id
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers=headers,
        )

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError("Adapter not started. Call start() first.")
        return self._http_client

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatCompletion:
        client = self._get_client()

        body: dict[str, Any] = {
            "model": model,
            "messages": [
                msg.model_dump() if not isinstance(msg, dict) else msg
                for msg in messages
            ],
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"MiniMax chat failed: {e}")
            raise

        self._handle_error(response)
        return self._parse_response(response.json(), model)

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        client = self._get_client()

        body: dict[str, Any] = {
            "model": model,
            "messages": [
                msg.model_dump() if not isinstance(msg, dict) else msg
                for msg in messages
            ],
            "stream": True,
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature

        try:
            async with client.stream(
                "POST", "/chat/completions", json=body
            ) as response:
                self._handle_error(response)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        chunk = self._parse_chunk(data, model)
                        if chunk:
                            yield chunk
        except Exception as e:
            logger.error(f"MiniMax stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        raise UnsupportedCapabilityError(
            message="MiniMax image generation not implemented in this adapter",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> VideoTask:
        raise UnsupportedCapabilityError(
            message="MiniMax video generation not implemented in this adapter",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        raise UnsupportedCapabilityError(
            message="MiniMax video generation not implemented in this adapter",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        models = [
            ModelInfo(
                id="abab6.5s-chat",
                name="ABAB 6.5s",
                type="chat",
                provider="minimax",
                capabilities=["chat"],
                description="MiniMax abab 6.5s 快速版本",
            ),
            ModelInfo(
                id="abab6.5-chat",
                name="ABAB 6.5",
                type="chat",
                provider="minimax",
                capabilities=["chat"],
                description="MiniMax abab 6.5 标准版本",
            ),
            ModelInfo(
                id="MiniMax-Text-01",
                name="MiniMax Text 01",
                type="chat",
                provider="minimax",
                capabilities=["chat"],
                description="MiniMax 万亿参数 MoE 模型",
            ),
            ModelInfo(
                id="MiniMax-VL-01",
                name="MiniMax VL 01",
                type="chat",
                provider="minimax",
                capabilities=["chat", "vision"],
                description="MiniMax 多模态视觉模型",
            ),
            ModelInfo(
                id="MiniMax-M1",
                name="MiniMax M1",
                type="chat",
                provider="minimax",
                capabilities=["chat", "vision"],
                description="MiniMax 最新推理模型",
            ),
            # 语音转文字模型
            ModelInfo(
                id="abab-asr",
                name="ABAB 语音识别",
                type="audio",
                provider="minimax",
                capabilities=["audio_transcribe"],
                description="MiniMax 语音识别模型",
            ),
            # 文字转语音模型
            ModelInfo(
                id="abab-tts",
                name="ABAB 语音合成",
                type="audio",
                provider="minimax",
                capabilities=["audio_speech"],
                description="MiniMax 语音合成模型（多音色）",
            ),
            ModelInfo(
                id="speech-01",
                name="MiniMax Speech 01",
                type="audio",
                provider="minimax",
                capabilities=["audio_speech"],
                description="MiniMax 高品质语音合成",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid MiniMax API key or group ID")
        if response.status_code == 429:
            raise RateLimitError(message="MiniMax rate limit exceeded")

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("error", {}).get("message")
                or error_data.get("message")
                or error_data.get("base_resp", {}).get("status_msg")
                or error_data.get("error")
                or f"HTTP {response.status_code}"
            )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=error_msg, status_code=response.status_code)

    def _parse_response(self, data: dict[str, Any], model: str) -> ChatCompletion:
        choices = data.get("choices", [])
        if not choices:
            raise APIError(message="No completion choices in response")

        choice = choices[0]
        message_data = choice.get("message", {})
        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=[
                ChatChoice(
                    index=choice.get("index", 0),
                    message=ChatMessage(
                        role=message_data.get("role", "assistant"),
                        content=message_data.get("content", ""),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            ],
            usage=usage,
        )

    def _parse_chunk(self, data_str: str, model: str) -> ChatCompletionChunk | None:
        """解析流式响应块"""
        import json

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return None

        choices = data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta_data = choice.get("delta", {})
        delta_message = ChatMessage(
            role=delta_data.get("role", "assistant"),
            content=delta_data.get("content", ""),
        )
        return ChatCompletionChunk(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=[
                ChatCompletionDelta(
                    index=choice.get("index", 0),
                    delta=delta_message,
                    finish_reason=choice.get("finish_reason"),
                )
            ],
        )


# 注册适配器
AdapterFactory.register("qwen", QwenAdapter)
AdapterFactory.register("zhipu", ZhipuAdapter)
AdapterFactory.register("doubao", DoubaoAdapter)
AdapterFactory.register("ernie", ErnieAdapter)
AdapterFactory.register("kimi", KimiAdapter)
AdapterFactory.register("minimax", MiniMaxAdapter)
