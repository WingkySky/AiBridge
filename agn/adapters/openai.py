"""
AGN-SDK OpenAI 适配器

实现 OpenAI API 的统一调用，包括对话、图像、嵌入、语音转文字、文字转语音等能力。
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from agn.adapters.base import BaseAdapter, Capabilities
from agn.adapters.chinese import OpenAICompatibleAudioMixin
from agn.adapters.factory import AdapterFactory
from agn.core.errors import (
    APIError,
    AuthenticationError,
    NetworkError,
    RateLimitError,
    TimeoutError,
)
from agn.core.utils import current_timestamp, generate_id
from agn.models.audio import (
    TranscriptionResult,
)
from agn.models.chat import (
    ChatChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChatMessage,
)
from agn.models.common import ModelInfo, ProviderConfig
from agn.models.image import ImageData, ImageGenerationResult
from agn.models.options import EmbeddingResult
from agn.models.video import VideoStatus, VideoTask

logger = logging.getLogger(__name__)

# OpenAI 默认 Base URL
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIAdapter(OpenAICompatibleAudioMixin, BaseAdapter):
    """
    OpenAI 适配器

    实现对 OpenAI API 的统一调用。
    """

    provider_type = "openai"
    provider_name = "OpenAI"
    supported_capabilities = [
        Capabilities.CHAT,
        Capabilities.CHAT_STREAM,
        Capabilities.VISION,
        Capabilities.TOOL_CALL,
        Capabilities.FUNCTION_CALL,
        Capabilities.JSON_MODE,
        Capabilities.REASONING,
        Capabilities.THINKING,
        Capabilities.IMAGE_GENERATE,
        Capabilities.EMBEDDING,
        Capabilities.AUDIO_TRANSCRIBE,
        Capabilities.AUDIO_TRANSLATE,
        Capabilities.AUDIO_SPEECH,
    ]

    def __init__(self, config: ProviderConfig) -> None:
        """
        初始化适配器

        Args:
            config: Provider 配置
        """
        super().__init__(config)
        self.base_url = config.base_url or DEFAULT_BASE_URL
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

    # ==================== 文本对话 ====================

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatCompletion:
        """
        文本对话

        Args:
            model: 模型名称
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            对话完成结果
        """
        stream = kwargs.get("stream", False)

        if stream:
            chunks = []
            async for chunk in self.chat_stream(model, messages, **kwargs):
                chunks.append(chunk)

            if not chunks:
                raise APIError(message="No response from server")

            return self._merge_chunks(chunks)

        client = self._get_client()

        body: dict[str, Any] = {
            "model": model,
            "messages": [msg.model_dump() for msg in messages],
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens
        if top_p := kwargs.get("top_p"):
            body["top_p"] = top_p
        if n := kwargs.get("n"):
            body["n"] = n
        if stop := kwargs.get("stop"):
            body["stop"] = stop
        if presence_penalty := kwargs.get("presence_penalty"):
            body["presence_penalty"] = presence_penalty
        if frequency_penalty := kwargs.get("frequency_penalty"):
            body["frequency_penalty"] = frequency_penalty

        logger.debug(f"Sending chat request to OpenAI: model={model}")

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"OpenAI chat request failed: {e}")
            raise

        self._handle_error(response)
        data = response.json()

        choices: list[ChatChoice] = []
        for i, choice_data in enumerate(data.get("choices", [])):
            message_data = choice_data.get("message", {})
            choices.append(
                ChatChoice(
                    index=i,
                    message=ChatMessage(**message_data),
                    finish_reason=choice_data.get("finish_reason"),
                )
            )

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=choices,
            usage=data.get("usage"),
        )

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        流式文本对话

        Args:
            model: 模型名称
            messages: 消息列表
            **kwargs: 其他参数

        Yields:
            逐个返回对话块
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": [msg.model_dump() for msg in messages],
            "stream": True,
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens
        if stop := kwargs.get("stop"):
            body["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers=headers,
        ) as client:
            try:
                async with client.stream(
                    "POST", "/chat/completions", json=body
                ) as response:
                    self._handle_error(response)

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue

                        if line.startswith("data: "):
                            line = line[6:]

                        if line == "[DONE]":
                            break

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        choices: list[ChatCompletionDelta] = []
                        for i, choice_data in enumerate(data.get("choices", [])):
                            delta_data = choice_data.get("delta", {})
                            delta_message = ChatMessage(
                                role=delta_data.get("role", "assistant"),
                                content=delta_data.get("content", ""),
                            )
                            choices.append(
                                ChatCompletionDelta(
                                    index=i,
                                    delta=delta_message,
                                    finish_reason=choice_data.get("finish_reason"),
                                )
                            )

                        chunk = ChatCompletionChunk(
                            id=data.get("id", generate_id("chatcmpl")),
                            created=data.get("created", current_timestamp()),
                            model=data.get("model", model),
                            choices=choices,
                        )

                        yield chunk

            except httpx.TimeoutException as e:
                raise TimeoutError(
                    message="Streaming request timeout", original_error=e
                ) from e
            except httpx.ConnectError as e:
                raise NetworkError(message="Connection error", original_error=e) from e

    def _merge_chunks(self, chunks: list[ChatCompletionChunk]) -> ChatCompletion:
        """合并流式 chunk"""
        if not chunks:
            raise APIError(message="No chunks to merge")

        first_chunk = chunks[0]
        merged_content = ""
        finish_reason = None

        for chunk in chunks:
            if chunk.choices:
                delta = chunk.choices[0]
                if delta.delta.content:
                    merged_content += delta.delta.content
                if delta.finish_reason:
                    finish_reason = delta.finish_reason

        merged_message = ChatMessage(role="assistant", content=merged_content)
        merged_choice = ChatChoice(
            index=0, message=merged_message, finish_reason=finish_reason
        )

        return ChatCompletion(
            id=first_chunk.id,
            created=first_chunk.created,
            model=first_chunk.model,
            choices=[merged_choice],
            usage=None,
        )

    # ==================== 图像生成 ====================

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        图像生成

        Args:
            model: 模型名称
            prompt: 提示词
            **kwargs: 其他参数

        Returns:
            图像生成结果
        """
        client = self._get_client()

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
        }

        if size := kwargs.get("size"):
            body["size"] = size
        if n := kwargs.get("n"):
            body["n"] = n
        if quality := kwargs.get("quality"):
            body["quality"] = quality
        if style := kwargs.get("style"):
            body["style"] = style
        if response_format := kwargs.get("response_format"):
            body["response_format"] = response_format

        logger.debug(f"Sending image request to OpenAI: model={model}")

        try:
            response = await client.post("/images/generations", json=body)
        except Exception as e:
            logger.error(f"OpenAI image request failed: {e}")
            raise

        self._handle_error(response)
        data = response.json()

        image_data_list: list[ImageData] = []
        for item in data.get("data", []):
            image_data_list.append(
                ImageData(
                    url=item.get("url"),
                    b64_json=item.get("b64_json"),
                    revised_prompt=item.get("revised_prompt"),
                )
            )

        return ImageGenerationResult(
            id=data.get("id", generate_id("img")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            data=image_data_list,
        )

    # ==================== 视频生成（不支持）====================

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> VideoTask:
        """
        创建视频生成任务

        OpenAI 暂不支持视频生成，此方法抛出错误。
        """
        from agn.core.errors import UnsupportedCapabilityError

        raise UnsupportedCapabilityError(
            message="OpenAI does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        """查询视频任务状态"""
        from agn.core.errors import UnsupportedCapabilityError

        raise UnsupportedCapabilityError(
            message="OpenAI does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    # ==================== 文本嵌入 ====================

    async def embed(
        self,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        文本嵌入

        Args:
            model: 嵌入模型名称（如 text-embedding-3-small）
            input: 输入文本或文本列表
            **kwargs: 其他参数（dimensions, encoding_format, user）

        Returns:
            嵌入结果
        """
        client = self._get_client()

        embed_model = model or "text-embedding-3-small"

        body: dict[str, Any] = {
            "model": embed_model,
            "input": input,
        }

        if dimensions := kwargs.get("dimensions"):
            body["dimensions"] = dimensions
        if encoding_format := kwargs.get("encoding_format"):
            body["encoding_format"] = encoding_format
        if user := kwargs.get("user"):
            body["user"] = user

        logger.debug(f"Sending embedding request to OpenAI: model={embed_model}")

        try:
            response = await client.post("/embeddings", json=body)
        except Exception as e:
            logger.error(f"OpenAI embedding request failed: {e}")
            raise

        self._handle_error(response)
        data = response.json()

        return EmbeddingResult(
            object=data.get("object", "list"),
            data=data.get("data", []),
            model=data.get("model", embed_model),
            usage=data.get("usage"),
        )

    # ==================== 模型信息 ====================

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        调用 GET /models 实时拉取，不再使用硬编码示例。

        Args:
            model_type: 模型类型过滤（chat/image/audio）

        Returns:
            模型信息列表
        """
        client = self._get_client()
        response = await client.get(url="/models")
        return self._parse_models_response(
            data=response.json(),
            provider="openai",
            model_type=model_type,
        )

    # ==================== 语音翻译（OpenAI 特有能力）====================

    async def translate(
        self,
        model: str,
        file: Any,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        语音翻译（翻译为英文）

        Args:
            model: 模型名称
            file: 音频文件
            **kwargs: 其他参数

        Returns:
            翻译结果
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

    # ==================== 辅助方法 ====================

    def _handle_error(self, response: httpx.Response) -> None:
        """处理 OpenAI 错误响应"""
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid API key")
        if response.status_code == 429:
            raise RateLimitError(message="Rate limit exceeded")
        if response.status_code == 404:
            raise APIError(message="Model not found", status_code=404)

        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get(
                "message", f"HTTP {response.status_code}"
            )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=error_msg, status_code=response.status_code)


# 注册适配器
AdapterFactory.register("openai", OpenAIAdapter)
