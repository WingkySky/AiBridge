"""
AGN-SDK Anthropic Claude 适配器

实现 Anthropic Claude API 的统一调用。

官方 API 文档：https://docs.anthropic.com/claude/reference/messages_post
- Base URL: https://api.anthropic.com
- Endpoint: POST /v1/messages
- 认证: x-api-key header
- 版本: anthropic-version header（固定值 2023-06-01）
- 支持流式: 是（SSE）
- 支持视觉: 是
"""

import json
import logging
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
from agn.models.chat import (
    ChatChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChatMessage,
)
from agn.models.common import ModelInfo, ProviderConfig
from agn.models.image import ImageGenerationResult
from agn.models.options import ANTHROPIC_MAPPING
from agn.models.video import VideoStatus, VideoTask

logger = logging.getLogger(__name__)

# Anthropic 默认配置
DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_API_VERSION = "2023-06-01"


class AnthropicAdapter(BaseAdapter):
    """
    Anthropic Claude 适配器

    实现对 Anthropic Claude API 的统一调用。
    支持 Claude 3/3.5/4 系列模型，包括：
    - claude-3-opus-20240229
    - claude-3-sonnet-20240229
    - claude-3-haiku-20240307
    - claude-3-5-sonnet-20241022
    - claude-3-5-haiku-20241022
    """

    provider_type = "anthropic"
    provider_name = "Anthropic Claude"
    supported_capabilities = [
        Capabilities.CHAT,
        Capabilities.CHAT_STREAM,
        Capabilities.VISION,
        Capabilities.TOOL_CALL,
        Capabilities.FUNCTION_CALL,
        Capabilities.REASONING,
        Capabilities.THINKING,
    ]
    param_mapping = ANTHROPIC_MAPPING

    def __init__(self, config: ProviderConfig) -> None:
        """
        初始化适配器

        Args:
            config: Provider 配置
        """
        super().__init__(config)
        self.base_url = config.base_url or DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self.api_version = DEFAULT_API_VERSION
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """启动适配器"""
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version,
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

    # ==================== 消息格式转换 ====================

    def _convert_messages(
        self, messages: list[ChatMessage] | list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        将统一消息格式转换为 Anthropic 格式

        Anthropic 要求：
        - system 消息单独放在 system 参数中
        - user/assistant 消息交替排列
        - 内容可以是字符串或 content blocks（支持图片）

        Args:
            messages: 消息列表

        Returns:
            (anthropic_messages, system_prompt)
        """
        anthropic_messages: list[dict[str, Any]] = []
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
            elif role in ("user", "assistant"):
                # 处理多模态内容
                if isinstance(content, list):
                    # Content blocks 格式
                    blocks = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "image_url":
                                url = block.get("image_url", {}).get("url", "")
                                if url.startswith("data:"):
                                    media_type, data = url.split(";", 1)
                                    media_type = media_type.replace("data:", "")
                                    encoding, data = data.split(",", 1)
                                    blocks.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": data,
                                            },
                                        }
                                    )
                            else:
                                blocks.append(block)
                        else:
                            blocks.append({"type": "text", "text": str(block)})
                    anthropic_messages.append({"role": role, "content": blocks})
                else:
                    anthropic_messages.append({"role": role, "content": str(content)})

        return anthropic_messages, system_prompt

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
            model: 模型名称（如 claude-3-5-sonnet-20241022）
            messages: 消息列表
            **kwargs: 其他参数
                - max_tokens: 最大生成 token 数（必填，默认 1024）
                - temperature: 温度
                - top_p: top_p 采样
                - stop_sequences: 停止词列表
                - system: 系统提示词（也可以从 messages 中提取）
                - thinking: 思考模式配置

        Returns:
            对话完成结果
        """
        client = self._get_client()

        anthropic_messages, system_from_messages = self._convert_messages(messages)
        system_prompt = kwargs.get("system", system_from_messages)

        # 构建请求体
        body: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
        }

        if system_prompt:
            body["system"] = system_prompt
        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if top_p := kwargs.get("top_p"):
            body["top_p"] = top_p
        if stop_sequences := kwargs.get("stop_sequences") or kwargs.get("stop"):
            body["stop_sequences"] = stop_sequences
        if thinking := kwargs.get("thinking"):
            body["thinking"] = thinking

        # 透传额外参数
        extra_body = kwargs.get("extra_body")
        if extra_body and isinstance(extra_body, dict):
            body.update(extra_body)

        logger.debug(f"Sending chat request to Anthropic: model={model}")

        try:
            response = await client.post("/v1/messages", json=body)
        except Exception as e:
            logger.error(f"Anthropic chat failed: {e}")
            raise

        self._handle_anthropic_error(response)
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

        anthropic_messages, system_from_messages = self._convert_messages(messages)
        system_prompt = kwargs.get("system", system_from_messages)

        body: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "stream": True,
        }

        if system_prompt:
            body["system"] = system_prompt
        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature

        try:
            async with client.stream("POST", "/v1/messages", json=body) as response:
                self._handle_anthropic_error(response)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        chunks = self._parse_stream_chunk(data, model)
                        for chunk in chunks:
                            yield chunk
        except Exception as e:
            logger.error(f"Anthropic stream chat failed: {e}")
            raise

    # ==================== 图像生成（不支持）====================

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """图像生成（不支持）"""
        raise UnsupportedCapabilityError(
            message="Anthropic does not support image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    # ==================== 视频生成（不支持）====================

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> VideoTask:
        """视频生成（不支持）"""
        raise UnsupportedCapabilityError(
            message="Anthropic does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        """视频状态（不支持）"""
        raise UnsupportedCapabilityError(
            message="Anthropic does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

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
        models = [
            ModelInfo(
                id="claude-3-opus-20240229",
                name="Claude 3 Opus",
                type="chat",
                provider="anthropic",
                capabilities=[
                    "chat",
                    "vision",
                    "tool_call",
                    "function_call",
                    "streaming",
                    "thinking",
                ],
                description="Claude 3 Opus - 最强大的模型",
            ),
            ModelInfo(
                id="claude-3-sonnet-20240229",
                name="Claude 3 Sonnet",
                type="chat",
                provider="anthropic",
                capabilities=[
                    "chat",
                    "vision",
                    "tool_call",
                    "function_call",
                    "streaming",
                    "thinking",
                ],
                description="Claude 3 Sonnet - 平衡性能和速度",
            ),
            ModelInfo(
                id="claude-3-haiku-20240307",
                name="Claude 3 Haiku",
                type="chat",
                provider="anthropic",
                capabilities=[
                    "chat",
                    "vision",
                    "tool_call",
                    "function_call",
                    "streaming",
                ],
                description="Claude 3 Haiku - 最快的模型",
            ),
            ModelInfo(
                id="claude-3-5-sonnet-20241022",
                name="Claude 3.5 Sonnet",
                type="chat",
                provider="anthropic",
                capabilities=[
                    "chat",
                    "vision",
                    "tool_call",
                    "function_call",
                    "streaming",
                    "thinking",
                ],
                description="Claude 3.5 Sonnet - 最新高性能模型",
            ),
            ModelInfo(
                id="claude-3-5-haiku-20241022",
                name="Claude 3.5 Haiku",
                type="chat",
                provider="anthropic",
                capabilities=[
                    "chat",
                    "vision",
                    "tool_call",
                    "function_call",
                    "streaming",
                ],
                description="Claude 3.5 Haiku - 最新快速模型",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    # ==================== 响应解析 ====================

    def _parse_response(self, data: dict[str, Any], model: str) -> ChatCompletion:
        """
        解析对话响应

        Args:
            data: API 响应数据
            model: 模型名称

        Returns:
            对话完成结果
        """
        content = ""
        # 提取 text 类型的内容块
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        # 提取 stop_reason
        finish_reason: str | None = data.get("stop_reason")
        stop_reason_map: dict[str, str] = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        mapped_reason: str | None = (
            stop_reason_map.get(finish_reason, finish_reason)
            if finish_reason is not None
            else None
        )

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=current_timestamp(),
            model=model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=mapped_reason,
                )
            ],
            usage={
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    data.get("usage", {}).get("input_tokens", 0)
                    + data.get("usage", {}).get("output_tokens", 0)
                ),
            },
        )

    def _parse_stream_chunk(
        self, data_str: str, model: str
    ) -> list[ChatCompletionChunk]:
        """
        解析流式响应块

        Anthropic 流式事件类型：
        - message_start: 消息开始
        - content_block_start: 内容块开始
        - content_block_delta: 内容增量
        - content_block_stop: 内容块结束
        - message_delta: 消息增量（包含 stop_reason）
        - message_stop: 消息结束

        Args:
            data_str: SSE 数据字符串
            model: 模型名称

        Returns:
            ChatCompletionChunk 列表
        """
        chunks: list[ChatCompletionChunk] = []

        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            return chunks

        event_type = event.get("type", "")
        chunk_id = generate_id("chatcmpl")
        created = current_timestamp()

        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    delta_message = ChatMessage(role="assistant", content=text)
                    chunks.append(
                        ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=model,
                            choices=[
                                ChatCompletionDelta(
                                    index=0,
                                    delta=delta_message,
                                    finish_reason=None,
                                )
                            ],
                        )
                    )

        elif event_type == "message_delta":
            delta = event.get("delta", {})
            stop_reason = delta.get("stop_reason")
            if stop_reason:
                stop_reason_map = {
                    "end_turn": "stop",
                    "max_tokens": "length",
                    "stop_sequence": "stop",
                }
                delta_message = ChatMessage(role="assistant", content="")
                chunks.append(
                    ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model,
                        choices=[
                            ChatCompletionDelta(
                                index=0,
                                delta=delta_message,
                                finish_reason=stop_reason_map.get(
                                    stop_reason, stop_reason
                                ),
                            )
                        ],
                    )
                )

        return chunks

    # ==================== 错误处理 ====================

    def _handle_anthropic_error(self, response: httpx.Response) -> None:
        """
        处理 Anthropic 错误响应

        错误格式：
        {
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "..."
            }
        }
        """
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Anthropic API key")
        if response.status_code == 429:
            raise RateLimitError(message="Anthropic rate limit exceeded")
        if response.status_code == 400:
            try:
                error_data = response.json()
                error_type = error_data.get("error", {}).get("type", "")
                error_msg = error_data.get("error", {}).get(
                    "message", f"HTTP {response.status_code}"
                )
                if error_type in ("authentication_error", "invalid_request_error"):
                    if "api key" in error_msg.lower():
                        raise AuthenticationError(message=error_msg)
            except Exception:
                pass

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("error", {}).get("message")
                or error_data.get("message")
                or f"HTTP {response.status_code}"
            )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=error_msg, status_code=response.status_code)


# 注册适配器
AdapterFactory.register("anthropic", AnthropicAdapter)
