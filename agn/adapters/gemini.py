"""
AGN-SDK Google Gemini 适配器

实现 Google Gemini API 的统一调用。

官方 API 文档：https://ai.google.dev/gemini-api/docs/text-generation
- Base URL: https://generativelanguage.googleapis.com/v1beta
- Endpoint: POST /models/{model}:generateContent
- 流式: POST /models/{model}:streamGenerateContent
- 认证: key 查询参数或 x-goog-api-key header
- 支持多模态: 是（图片、视频、音频）
"""

import json
import logging
from typing import Any

import httpx

from agn.adapters.base import BaseAdapter
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
from agn.models.options import EmbeddingResult
from agn.models.video import VideoStatus, VideoTask

logger = logging.getLogger(__name__)

# Gemini 默认配置
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiAdapter(BaseAdapter):
    """
    Google Gemini 适配器

    实现对 Google Gemini API 的统一调用。
    支持 Gemini 1.5/2.0/2.5 系列模型。
    """

    provider_type = "gemini"
    provider_name = "Google Gemini"
    supported_capabilities = ["chat", "vision", "embedding"]

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
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
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
        将统一消息格式转换为 Gemini 格式

        Gemini 格式：
        - system: 放在 systemInstruction 字段
        - messages: contents 数组，role 为 user/model
        - parts: 内容数组，可以包含 text 或 inline_data

        Args:
            messages: 消息列表

        Returns:
            (gemini_contents, system_instruction)
        """
        contents: list[dict[str, Any]] = []
        system_instruction: str | None = None

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = msg.role
                content = msg.content

            if role == "system":
                system_instruction = (
                    content if isinstance(content, str) else str(content)
                )
            elif role in ("user", "assistant"):
                # Gemini 使用 "model" 作为 assistant 的 role
                gemini_role = "user" if role == "user" else "model"

                # 处理多模态内容
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append({"text": block.get("text", "")})
                            elif block.get("type") == "image_url":
                                url = block.get("image_url", {}).get("url", "")
                                if url.startswith("data:"):
                                    media_type, data = url.split(";", 1)
                                    media_type = media_type.replace("data:", "")
                                    encoding, data = data.split(",", 1)
                                    parts.append(
                                        {
                                            "inline_data": {
                                                "mime_type": media_type,
                                                "data": data,
                                            },
                                        }
                                    )
                            else:
                                parts.append({"text": str(block)})
                        else:
                            parts.append({"text": str(block)})
                    contents.append({"role": gemini_role, "parts": parts})
                else:
                    contents.append(
                        {
                            "role": gemini_role,
                            "parts": [{"text": str(content)}],
                        }
                    )

        return contents, system_instruction

    def _convert_generation_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        转换生成配置参数

        Args:
            kwargs: 关键字参数

        Returns:
            Gemini generationConfig
        """
        config: dict[str, Any] = {}

        if temperature := kwargs.get("temperature"):
            config["temperature"] = temperature
        if top_p := kwargs.get("top_p"):
            config["topP"] = top_p
        if top_k := kwargs.get("top_k"):
            config["topK"] = top_k
        if max_tokens := kwargs.get("max_tokens"):
            config["maxOutputTokens"] = max_tokens
        if stop := kwargs.get("stop"):
            config["stopSequences"] = stop if isinstance(stop, list) else [stop]

        return config

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
            model: 模型名称（如 gemini-2.5-pro, gemini-1.5-flash）
            messages: 消息列表
            **kwargs: 其他参数
                - temperature: 温度
                - top_p: top_p 采样
                - top_k: top_k 采样
                - max_tokens: 最大生成 token 数
                - stop: 停止词

        Returns:
            对话完成结果
        """
        client = self._get_client()

        contents, system_instruction = self._convert_messages(messages)
        generation_config = self._convert_generation_config(kwargs)

        # 构建请求体
        body: dict[str, Any] = {"contents": contents}

        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if generation_config:
            body["generationConfig"] = generation_config

        # 透传额外参数
        extra_body = kwargs.get("extra_body")
        if extra_body and isinstance(extra_body, dict):
            body.update(extra_body)

        logger.debug(f"Sending chat request to Gemini: model={model}")

        try:
            response = await client.post(f"/models/{model}:generateContent", json=body)
        except Exception as e:
            logger.error(f"Gemini chat failed: {e}")
            raise

        self._handle_gemini_error(response)
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

        contents, system_instruction = self._convert_messages(messages)
        generation_config = self._convert_generation_config(kwargs)

        body: dict[str, Any] = {"contents": contents}

        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if generation_config:
            body["generationConfig"] = generation_config

        try:
            async with client.stream(
                "POST",
                f"/models/{model}:streamGenerateContent",
                json=body,
                params={"alt": "sse"},
            ) as response:
                self._handle_gemini_error(response)
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    # Gemini SSE format: data: {...}\n\n
                    while "\n\n" in buffer:
                        event, buffer = buffer.split("\n\n", 1)
                        if event.startswith("data: "):
                            data_str = event[6:]
                            chunks = self._parse_stream_chunk(data_str, model)
                            for c in chunks:
                                yield c
        except Exception as e:
            logger.error(f"Gemini stream chat failed: {e}")
            raise

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
            model: 嵌入模型名称（如 text-embedding-004）
            input: 输入文本或文本列表
            **kwargs: 其他参数
                - output_dimensionality: 输出维度

        Returns:
            嵌入结果
        """
        client = self._get_client()

        embed_model = model or "text-embedding-004"

        if isinstance(input, str):
            texts = [input]
        else:
            texts = input

        if len(texts) == 1:
            body: dict[str, Any] = {
                "model": f"models/{embed_model}",
                "content": {"parts": [{"text": texts[0]}]},
            }

            if output_dimensionality := kwargs.get("output_dimensionality"):
                body["outputDimensionality"] = output_dimensionality

            logger.debug(f"Sending embedding request to Gemini: model={embed_model}")

            try:
                response = await client.post(
                    f"/models/{embed_model}:embedContent", json=body
                )
            except Exception as e:
                logger.error(f"Gemini embedding request failed: {e}")
                raise

            self._handle_gemini_error(response)
            data = response.json()

            embedding_values = data.get("embedding", {}).get("values", [])

            return EmbeddingResult(
                object="list",
                data=[
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": embedding_values,
                    }
                ],
                model=embed_model,
                usage=None,
            )
        else:
            requests = []
            for text in texts:
                req = {
                    "model": f"models/{embed_model}",
                    "content": {"parts": [{"text": text}]},
                }
                if output_dimensionality := kwargs.get("output_dimensionality"):
                    req["outputDimensionality"] = output_dimensionality
                requests.append(req)

            body = {"requests": requests}

            try:
                response = await client.post(
                    f"/models/{embed_model}:batchEmbedContents", json=body
                )
            except Exception as e:
                logger.error(f"Gemini batch embedding request failed: {e}")
                raise

            self._handle_gemini_error(response)
            data = response.json()

            embeddings = data.get("embeddings", [])
            result_data = []
            for i, emb in enumerate(embeddings):
                result_data.append(
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": emb.get("values", []),
                    }
                )

            return EmbeddingResult(
                object="list",
                data=result_data,
                model=embed_model,
                usage=None,
            )

    # ==================== 图像生成（不支持）====================

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """图像生成（不支持，Gemini 支持图像理解不支持直接生成）"""
        raise UnsupportedCapabilityError(
            message="Gemini does not support direct image generation (Imagen is separate)",
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
            message="Gemini does not support video generation (Veo is separate)",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        """视频状态（不支持）"""
        raise UnsupportedCapabilityError(
            message="Gemini does not support video generation",
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
                id="gemini-2.5-pro",
                name="Gemini 2.5 Pro",
                type="chat",
                provider="gemini",
                capabilities=["chat", "vision"],
                description="Gemini 2.5 Pro - 最新最强模型",
            ),
            ModelInfo(
                id="gemini-2.5-flash",
                name="Gemini 2.5 Flash",
                type="chat",
                provider="gemini",
                capabilities=["chat", "vision"],
                description="Gemini 2.5 Flash - 快速高效模型",
            ),
            ModelInfo(
                id="gemini-1.5-pro",
                name="Gemini 1.5 Pro",
                type="chat",
                provider="gemini",
                capabilities=["chat", "vision"],
                description="Gemini 1.5 Pro",
            ),
            ModelInfo(
                id="gemini-1.5-flash",
                name="Gemini 1.5 Flash",
                type="chat",
                provider="gemini",
                capabilities=["chat", "vision"],
                description="Gemini 1.5 Flash",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    # ==================== 响应解析 ====================

    def _parse_response(self, data: dict[str, Any], model: str) -> ChatCompletion:
        """
        解析对话响应

        Gemini 响应格式：
        {
            "candidates": [{
                "content": {
                    "parts": [{"text": "..."}],
                    "role": "model"
                },
                "finishReason": "STOP",
                ...
            }],
            "usageMetadata": {
                "promptTokenCount": ...,
                "candidatesTokenCount": ...,
                ...
            }
        }
        """
        content = ""
        finish_reason = "stop"

        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            content_parts = candidate.get("content", {}).get("parts", [])
            content = "".join(part.get("text", "") for part in content_parts)

            reason_map = {
                "STOP": "stop",
                "MAX_TOKENS": "length",
                "SAFETY": "content_filter",
                "RECITATION": "stop",
                "OTHER": "stop",
            }
            raw_reason = candidate.get("finishReason", "STOP")
            finish_reason = reason_map.get(raw_reason, "stop")

        usage = data.get("usageMetadata", {})

        return ChatCompletion(
            id=generate_id("chatcmpl"),
            created=current_timestamp(),
            model=model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage={
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
        )

    def _parse_stream_chunk(
        self, data_str: str, model: str
    ) -> list[ChatCompletionChunk]:
        """
        解析流式响应块

        Args:
            data_str: 数据字符串
            model: 模型名称

        Returns:
            ChatCompletionChunk 列表
        """
        chunks: list[ChatCompletionChunk] = []

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return chunks

        chunk_id = generate_id("chatcmpl")
        created = current_timestamp()
        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            content_parts = candidate.get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in content_parts)

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

            # 检查 finishReason
            raw_reason = candidate.get("finishReason")
            if raw_reason:
                reason_map = {
                    "STOP": "stop",
                    "MAX_TOKENS": "length",
                    "SAFETY": "content_filter",
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
                                finish_reason=reason_map.get(raw_reason, raw_reason),
                            )
                        ],
                    )
                )

        return chunks

    # ==================== 错误处理 ====================

    def _handle_gemini_error(self, response: httpx.Response) -> None:
        """
        处理 Gemini 错误响应

        错误格式：
        {
            "error": {
                "code": 400,
                "message": "...",
                "status": "..."
            }
        }
        """
        if response.status_code < 400:
            return

        if response.status_code == 401 or response.status_code == 403:
            raise AuthenticationError(message="Invalid Gemini API key")
        if response.status_code == 429:
            raise RateLimitError(message="Gemini rate limit exceeded")

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
AdapterFactory.register("gemini", GeminiAdapter)
