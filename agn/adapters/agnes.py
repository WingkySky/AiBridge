"""
AGN-SDK Agnes AI 适配器

实现 Agnes AI 的 API 调用适配。
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from agn.adapters.base import BaseAdapter, Capabilities
from agn.adapters.factory import AdapterFactory
from agn.core.errors import (
    APIError,
    AuthenticationError,
    NetworkError,
    RateLimitError,
    TimeoutError,
)
from agn.core.http_client import AsyncHttpClient
from agn.core.utils import current_timestamp, generate_id
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

# Agnes AI 默认 Base URL
DEFAULT_BASE_URL = "https://api.agnes.ai/v1"


class AgnesAdapter(BaseAdapter):
    """
    Agnes AI 适配器

    实现对 Agnes AI API 的统一调用。
    """

    provider_type = "agnes"
    provider_name = "Agnes AI"
    supported_capabilities = [
        # 对话能力
        Capabilities.CHAT,
        Capabilities.CHAT_STREAM,
        Capabilities.VISION,
        Capabilities.TOOL_CALL,
        Capabilities.FUNCTION_CALL,
        Capabilities.JSON_MODE,
        Capabilities.REASONING,
        Capabilities.THINKING,
        # 图像能力
        Capabilities.IMAGE_GENERATE,
        Capabilities.IMAGE_EDIT,
        Capabilities.IMAGE_TO_IMAGE,
        Capabilities.IMAGE_INPAINT,
        # 视频能力
        Capabilities.VIDEO_GENERATE,
        Capabilities.VIDEO_TEXT2VIDEO,
        Capabilities.VIDEO_IMAGE2VIDEO,
        # 嵌入能力
        Capabilities.EMBEDDING,
    ]

    def __init__(self, config: ProviderConfig) -> None:
        """
        初始化适配器

        Args:
            config: Provider 配置
        """
        super().__init__(config)
        self.base_url = config.base_url or DEFAULT_BASE_URL
        self.poll_url = config.poll_url
        self.api_key = config.api_key or ""
        self._http_client: AsyncHttpClient | None = None

        # 预定义的模型列表（实际应该从 API 获取）
        self._models_cache: list[ModelInfo] | None = None

    async def start(self) -> None:
        """启动适配器"""
        self._http_client = AsyncHttpClient(
            base_url=self.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        await self._http_client.start()

    async def close(self) -> None:
        """关闭适配器"""
        if self._http_client:
            await self._http_client.close()
            self._http_client = None

    def _get_client(self) -> AsyncHttpClient:
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
            **kwargs: 其他参数（temperature, max_tokens, stop 等）

        Returns:
            对话完成结果

        Raises:
            AuthenticationError: 认证失败
            RateLimitError: 请求被限流
            APIError: API 调用失败
        """
        stream = kwargs.get("stream", False)

        if stream:
            # 如果请求流式，收集所有 chunk 后返回完整结果
            chunks = []
            async for chunk in self.chat_stream(model, messages, **kwargs):
                chunks.append(chunk)

            if not chunks:
                raise APIError(message="No response from server")

            # 合并所有 chunk 为完整响应
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
        if stop := kwargs.get("stop"):
            body["stop"] = stop

        logger.debug(
            f"Sending chat request to Agnes AI: model={model}, messages={len(messages)}"
        )

        try:
            response = await client.post(
                url="/chat/completions",
                json=body,
            )
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            raise

        data = response.json()
        logger.debug(f"Received chat response: id={data.get('id')}")

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

        Raises:
            AuthenticationError: 认证失败
            NetworkError: 网络错误
            TimeoutError: 请求超时
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

        logger.debug(f"Sending streaming chat request: model={model}")

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers=headers,
        ) as client:
            try:
                async with client.stream(
                    "POST",
                    "/chat/completions",
                    json=body,
                ) as response:
                    if response.status_code == 401:
                        raise AuthenticationError(message="Invalid API key")
                    if response.status_code == 429:
                        raise RateLimitError(message="Rate limit exceeded")
                    if response.status_code >= 400:
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get("error", {}).get(
                                "message", f"HTTP {response.status_code}"
                            )
                        except Exception:
                            error_msg = f"HTTP {response.status_code}"
                        raise APIError(
                            message=error_msg, status_code=response.status_code
                        )

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
            except httpx.HTTPError as e:
                raise NetworkError(message=f"HTTP error: {e}", original_error=e) from e

    def _merge_chunks(self, chunks: list[ChatCompletionChunk]) -> ChatCompletion:
        """
        将多个流式 chunk 合并为完整的 ChatCompletion

        Args:
            chunks: chunk 列表

        Returns:
            合并后的完整响应
        """
        if not chunks:
            raise APIError(message="No chunks to merge")

        first_chunk = chunks[0]

        # 合并 content
        merged_content = ""
        finish_reason = None

        for chunk in chunks:
            if chunk.choices:
                delta = chunk.choices[0]
                if delta.delta.content:
                    merged_content += delta.delta.content
                if delta.finish_reason:
                    finish_reason = delta.finish_reason

        merged_message = ChatMessage(
            role="assistant",
            content=merged_content,
        )

        merged_choice = ChatChoice(
            index=0,
            message=merged_message,
            finish_reason=finish_reason,
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
            **kwargs: 其他参数（size, n, negative_prompt 等）

        Returns:
            图像生成结果

        Raises:
            APIError: API 调用失败
            RateLimitError: 请求被限流
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
        if negative_prompt := kwargs.get("negative_prompt"):
            body["negative_prompt"] = negative_prompt
        if response_format := kwargs.get("response_format"):
            body["response_format"] = response_format

        if reference_images := kwargs.get("reference_images"):
            extra_body: dict[str, Any] = {}
            if isinstance(reference_images, list) and reference_images:
                extra_body["image"] = reference_images[0]
            body["extra_body"] = extra_body

        logger.debug(
            f"Sending image generation request: model={model}, prompt_length={len(prompt)}"
        )

        try:
            response = await client.post(
                url="/images/generations",
                json=body,
            )
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise

        data = response.json()
        logger.debug(
            f"Received image response: id={data.get('id')}, count={len(data.get('data', []))}"
        )

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
            model: 嵌入模型名称
            input: 输入文本或文本列表
            **kwargs: 其他参数（dimensions, encoding_format, user）

        Returns:
            嵌入结果
        """
        client = self._get_client()

        body: dict[str, Any] = {
            "model": model,
            "input": input,
        }

        if dimensions := kwargs.get("dimensions"):
            body["dimensions"] = dimensions
        if encoding_format := kwargs.get("encoding_format"):
            body["encoding_format"] = encoding_format
        if user := kwargs.get("user"):
            body["user"] = user

        logger.debug(f"Sending embedding request to Agnes AI: model={model}")

        try:
            response = await client.post(
                url="/embeddings",
                json=body,
            )
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            raise

        data = response.json()

        return EmbeddingResult(
            object=data.get("object", "list"),
            data=data.get("data", []),
            model=data.get("model", model),
            usage=data.get("usage"),
        )

    # ==================== 视频生成 ====================

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> VideoTask:
        """
        创建视频生成任务

        Args:
            model: 模型名称
            prompt: 提示词
            **kwargs: 其他参数

        Returns:
            视频任务信息
        """
        client = self._get_client()

        # 构建请求体
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
        }

        # 添加可选参数
        if width := kwargs.get("width"):
            body["width"] = width
        if height := kwargs.get("height"):
            body["height"] = height
        if num_frames := kwargs.get("num_frames"):
            body["num_frames"] = num_frames
        if frame_rate := kwargs.get("frame_rate"):
            body["frame_rate"] = frame_rate
        if mode := kwargs.get("mode"):
            body["mode"] = mode
        if seed := kwargs.get("seed"):
            body["seed"] = seed
        if negative_prompt := kwargs.get("negative_prompt"):
            body["negative_prompt"] = negative_prompt

        # 处理参考图像
        if reference_images := kwargs.get("reference_images"):
            extra_body: dict[str, Any] = {}
            if mode == "keyframes" and len(reference_images) >= 2:
                extra_body["keyframes"] = {
                    "start": reference_images[0],
                    "end": reference_images[-1],
                }
            elif mode == "multiimage" and len(reference_images) >= 2:
                extra_body["image"] = reference_images
            elif len(reference_images) >= 1:
                extra_body["image"] = reference_images[0]
            body["extra_body"] = extra_body

        # 发送请求
        response = await client.post(
            url="/videos/generations",
            json=body,
        )

        data = response.json()

        return VideoTask(
            task_id=data.get("id", data.get("video_id", generate_id("vid"))),
            model=model,
            status=data.get("status", "pending"),
            created_at=data.get("created", current_timestamp()),
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
            model: 模型名称

        Returns:
            视频任务状态
        """
        client = self._get_client()

        # 使用指定的 poll_url 或默认 base_url
        base_url = self.poll_url or self.base_url
        client = AsyncHttpClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        await client.start()

        try:
            response = await client.get(
                url=f"/videos/generations/{task_id}",
            )
            data = response.json()

            return VideoStatus(
                task_id=task_id,
                status=data.get("status", "pending"),
                video_url=data.get("video_url")
                or data.get("output", {}).get("video_url"),
                progress=data.get("progress"),
                error=data.get("error"),
                created_at=data.get("created"),
                updated_at=data.get("updated"),
            )
        finally:
            await client.close()

    # ==================== 模型信息 ====================

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        Args:
            model_type: 模型类型过滤

        Returns:
            模型信息列表
        """
        # 预定义的模型列表（实际应该从 API 获取）
        models = [
            # 文本对话模型
            ModelInfo(
                id="claude-3-opus",
                name="Claude 3 Opus",
                type="chat",
                provider="agnes",
                capabilities=[
                    "chat",
                    "vision",
                    "tool_call",
                    "function_call",
                    "streaming",
                ],
                max_tokens=200000,
                supports_streaming=True,
            ),
            ModelInfo(
                id="claude-3-sonnet",
                name="Claude 3 Sonnet",
                type="chat",
                provider="agnes",
                capabilities=[
                    "chat",
                    "vision",
                    "tool_call",
                    "function_call",
                    "streaming",
                ],
                max_tokens=200000,
                supports_streaming=True,
            ),
            ModelInfo(
                id="gpt-4o",
                name="GPT-4o",
                type="chat",
                provider="agnes",
                capabilities=[
                    "chat",
                    "vision",
                    "tool_call",
                    "function_call",
                    "json_mode",
                    "streaming",
                ],
                max_tokens=128000,
                supports_streaming=True,
            ),
            # 图像生成模型
            ModelInfo(
                id="dall-e-3",
                name="DALL-E 3",
                type="image",
                provider="agnes",
                capabilities=["text2image", "image2image", "inpaint"],
            ),
            ModelInfo(
                id="seedream-3",
                name="Seedream 3.0",
                type="image",
                provider="agnes",
                capabilities=["text2image", "image2image", "inpaint"],
            ),
            # 视频生成模型
            ModelInfo(
                id="video-gen-1",
                name="Video Gen 1",
                type="video",
                provider="agnes",
                capabilities=["text2video"],
            ),
            ModelInfo(
                id="video-gen-2",
                name="Video Gen 2",
                type="video",
                provider="agnes",
                capabilities=["text2video", "image2video"],
            ),
            # 嵌入模型
            ModelInfo(
                id="text-embedding-3-small",
                name="Text Embedding 3 Small",
                type="chat",
                provider="agnes",
                capabilities=["embedding"],
                max_tokens=8191,
            ),
            ModelInfo(
                id="bge-m3",
                name="BGE M3",
                type="chat",
                provider="agnes",
                capabilities=["embedding"],
                max_tokens=8192,
            ),
        ]

        # 按类型过滤
        if model_type:
            models = [m for m in models if m.type == model_type]

        return models


# 注册适配器
AdapterFactory.register("agnes", AgnesAdapter)
