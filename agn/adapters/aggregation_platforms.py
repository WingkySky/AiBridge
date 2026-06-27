"""
AGN-SDK 聚合平台模型适配器

支持：SiliconFlow (硅基流动)、Together AI、Fireworks AI、Cloudflare Workers AI

官方文档：
- SiliconFlow: https://docs.siliconflow.cn/
- Together AI: https://docs.together.ai/
- Fireworks AI: https://docs.fireworks.ai/
- Cloudflare Workers AI: https://developers.cloudflare.com/workers-ai/
"""

import logging
from typing import Any

import httpx

from agn.adapters.base import BaseAdapter, Capabilities
from agn.adapters.chinese import OpenAICompatibleAudioMixin
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
    ChatUsage,
)
from agn.models.common import ModelInfo, ProviderConfig

logger = logging.getLogger(__name__)


# ==================== SiliconFlow 硅基流动 适配器 ====================


class SiliconFlowAdapter(OpenAICompatibleAudioMixin, BaseAdapter):
    """
    SiliconFlow (硅基流动) 适配器

    官方 API 规范（OpenAI 兼容）：
    - Base URL: https://api.siliconflow.cn/v1
    - Chat: POST /chat/completions
    - 认证: Bearer Token
    - 文档: https://docs.siliconflow.cn/
    - 特点: 聚合多模型、支持深度思考(reasoning)、华为云合作
    """

    provider_type = "siliconflow"
    provider_name = "SiliconFlow 硅基流动"
    supported_capabilities = [
        Capabilities.CHAT,
        Capabilities.CHAT_STREAM,
        Capabilities.VISION,
        Capabilities.TOOL_CALL,
        Capabilities.FUNCTION_CALL,
        Capabilities.REASONING,
        Capabilities.JSON_MODE,
        Capabilities.EMBEDDING,
        Capabilities.AUDIO_TRANSCRIBE,
        Capabilities.AUDIO_SPEECH,
    ]

    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

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
            "messages": [
                msg.model_dump() if hasattr(msg, "model_dump") else msg
                for msg in messages
            ],
        }

        # SiliconFlow 特有参数
        if enable_thinking := kwargs.get("enable_thinking"):
            body["enable_thinking"] = enable_thinking
        if thinking_budget := kwargs.get("thinking_budget"):
            body["thinking_budget"] = thinking_budget
        if min_p := kwargs.get("min_p"):
            body["min_p"] = min_p

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens
        if top_p := kwargs.get("top_p"):
            body["top_p"] = top_p

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"SiliconFlow chat failed: {e}")
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
                msg.model_dump() if hasattr(msg, "model_dump") else msg
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
            logger.error(f"SiliconFlow stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="SiliconFlow image generation not implemented",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="SiliconFlow video generation not implemented",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="SiliconFlow video generation not implemented",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        调用 GET /models 实时拉取，不再使用硬编码示例。

        Args:
            model_type: 模型类型过滤（chat/image/video）

        Returns:
            模型信息列表
        """
        client = self._get_client()
        response = await client.get(url="/models")
        return self._parse_models_response(
            data=response.json(),
            provider="siliconflow",
            model_type=model_type,
        )

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid SiliconFlow API key")
        if response.status_code == 429:
            raise RateLimitError(message="SiliconFlow rate limit exceeded")

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
        choices_data = data.get("choices", [])
        if not choices_data:
            raise APIError(message="No completion choices in response")

        chat_choices: list[ChatChoice] = []
        for idx, choice in enumerate(choices_data):
            message_data = choice.get("message", {})
            chat_message = ChatMessage(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content", ""),
            )
            chat_choices.append(
                ChatChoice(
                    index=choice.get("index", idx),
                    message=chat_message,
                    finish_reason=choice.get("finish_reason"),
                )
            )

        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=chat_choices,
            usage=usage,
        )

    def _parse_chunk(self, data_str: str, model: str) -> ChatCompletionChunk | None:
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


# ==================== Together AI 适配器 ====================


class TogetherAIAdapter(OpenAICompatibleAudioMixin, BaseAdapter):
    """
    Together AI 适配器

    官方 API 规范（OpenAI 兼容）：
    - Base URL: https://api.together.xyz/v1
    - Chat: POST /chat/completions
    - 语音: POST /audio/transcriptions, /audio/speech
    - 认证: Bearer Token
    - 文档: https://docs.together.ai/
    - 特点: 聚合多种开源模型、支持 Llama/Mixtral/Qwen/Whisper 等
    """

    provider_type = "togetherai"
    provider_name = "Together AI"
    supported_capabilities = [
        Capabilities.CHAT,
        Capabilities.CHAT_STREAM,
        Capabilities.VISION,
        Capabilities.JSON_MODE,
        Capabilities.EMBEDDING,
        Capabilities.AUDIO_TRANSCRIBE,
        Capabilities.AUDIO_SPEECH,
    ]

    DEFAULT_BASE_URL = "https://api.together.xyz/v1"

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
            "messages": [
                msg.model_dump() if hasattr(msg, "model_dump") else msg
                for msg in messages
            ],
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens
        if top_p := kwargs.get("top_p"):
            body["top_p"] = top_p

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"Together AI chat failed: {e}")
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
                msg.model_dump() if hasattr(msg, "model_dump") else msg
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
            logger.error(f"Together AI stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Together AI image generation not implemented",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Together AI video generation not implemented",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Together AI video generation not implemented",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        调用 GET /models 实时拉取，不再使用硬编码示例。

        Args:
            model_type: 模型类型过滤（chat/image/video）

        Returns:
            模型信息列表
        """
        client = self._get_client()
        response = await client.get(url="/models")
        return self._parse_models_response(
            data=response.json(),
            provider="togetherai",
            model_type=model_type,
        )

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Together AI API key")
        if response.status_code == 429:
            raise RateLimitError(message="Together AI rate limit exceeded")

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
        choices_data = data.get("choices", [])
        if not choices_data:
            raise APIError(message="No completion choices in response")

        chat_choices: list[ChatChoice] = []
        for idx, choice in enumerate(choices_data):
            message_data = choice.get("message", {})
            chat_message = ChatMessage(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content", ""),
            )
            chat_choices.append(
                ChatChoice(
                    index=choice.get("index", idx),
                    message=chat_message,
                    finish_reason=choice.get("finish_reason"),
                )
            )

        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=chat_choices,
            usage=usage,
        )

    def _parse_chunk(self, data_str: str, model: str) -> ChatCompletionChunk | None:
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


# ==================== Fireworks AI 适配器 ====================


class FireworksAIAdapter(OpenAICompatibleAudioMixin, BaseAdapter):
    """
    Fireworks AI 适配器

    官方 API 规范（OpenAI 兼容）：
    - Base URL: https://api.fireworks.ai/inference/v1
    - Chat: POST /chat/completions
    - 语音: POST /audio/transcriptions, /audio/speech
    - 认证: Bearer Token
    - 文档: https://docs.fireworks.ai/
    - 特点: 高性能推理、极低延迟、支持多种开源模型、Whisper
    """

    provider_type = "fireworksai"
    provider_name = "Fireworks AI"
    supported_capabilities = [
        Capabilities.CHAT,
        Capabilities.CHAT_STREAM,
        Capabilities.VISION,
        Capabilities.TOOL_CALL,
        Capabilities.FUNCTION_CALL,
        Capabilities.JSON_MODE,
        Capabilities.EMBEDDING,
        Capabilities.AUDIO_TRANSCRIBE,
    ]

    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

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
            "messages": [
                msg.model_dump() if hasattr(msg, "model_dump") else msg
                for msg in messages
            ],
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens
        if top_p := kwargs.get("top_p"):
            body["top_p"] = top_p

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"Fireworks AI chat failed: {e}")
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
                msg.model_dump() if hasattr(msg, "model_dump") else msg
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
            logger.error(f"Fireworks AI stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Fireworks AI image generation not implemented",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Fireworks AI video generation not implemented",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Fireworks AI video generation not implemented",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        调用 GET /models 实时拉取，不再使用硬编码示例。

        Args:
            model_type: 模型类型过滤（chat/image/video）

        Returns:
            模型信息列表
        """
        client = self._get_client()
        response = await client.get(url="/models")
        return self._parse_models_response(
            data=response.json(),
            provider="fireworksai",
            model_type=model_type,
        )

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Fireworks AI API key")
        if response.status_code == 429:
            raise RateLimitError(message="Fireworks AI rate limit exceeded")

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
        choices_data = data.get("choices", [])
        if not choices_data:
            raise APIError(message="No completion choices in response")

        chat_choices: list[ChatChoice] = []
        for idx, choice in enumerate(choices_data):
            message_data = choice.get("message", {})
            chat_message = ChatMessage(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content", ""),
            )
            chat_choices.append(
                ChatChoice(
                    index=choice.get("index", idx),
                    message=chat_message,
                    finish_reason=choice.get("finish_reason"),
                )
            )

        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created", current_timestamp()),
            model=data.get("model", model),
            choices=chat_choices,
            usage=usage,
        )

    def _parse_chunk(self, data_str: str, model: str) -> ChatCompletionChunk | None:
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


# ==================== Cloudflare Workers AI 适配器 ====================


class CloudflareAIAdapter(BaseAdapter):
    """
    Cloudflare Workers AI 适配器

    官方 API 规范（OpenAI 兼容）：
    - Base URL: https://api.cloudflare.com/client/v4/accounts/{account_id}/ai
    - Chat: POST /v1/run/@cf/... (使用 Workers AI 模型路径)
    - 认证: Bearer Token (Cloudflare API Token) + Account ID
    - 文档: https://developers.cloudflare.com/workers-ai/
    - 特点: 边缘推理、全球低延迟、支持开源模型
    """

    provider_type = "cloudflareai"
    provider_name = "Cloudflare Workers AI"
    supported_capabilities = [
        Capabilities.CHAT,
        Capabilities.CHAT_STREAM,
        Capabilities.VISION,
        Capabilities.EMBEDDING,
    ]

    DEFAULT_BASE_URL = "https://api.cloudflare.com/client/v4"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self.account_id = config.extra.get("account_id") if config.extra else None
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if not self.account_id:
            raise ValueError("Cloudflare account_id is required")

        self._http_client = httpx.AsyncClient(
            base_url=f"{self.base_url}/accounts/{self.account_id}/ai",
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

    def _format_model(self, model: str) -> str:
        """Cloudflare 使用 @cf/ 前缀"""
        if model.startswith("@cf/"):
            return model
        return f"@cf/{model}"

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatCompletion:
        client = self._get_client()

        body: dict[str, Any] = {
            "messages": [
                msg.model_dump() if hasattr(msg, "model_dump") else msg
                for msg in messages
            ],
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens

        cf_model = self._format_model(model)

        try:
            response = await client.post(f"/v1/run/{cf_model}", json=body)
        except Exception as e:
            logger.error(f"Cloudflare AI chat failed: {e}")
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
            "messages": [
                msg.model_dump() if hasattr(msg, "model_dump") else msg
                for msg in messages
            ],
            "stream": True,
        }

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature

        cf_model = self._format_model(model)

        try:
            async with client.stream(
                "POST", f"/v1/run/{cf_model}", json=body
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
            logger.error(f"Cloudflare AI stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Cloudflare AI image generation not implemented",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Cloudflare AI video generation not implemented",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Cloudflare AI video generation not implemented",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        调用 GET /models/search 实时拉取，不再使用硬编码示例。
        Cloudflare 端点需要 account_id（在 start() 中已校验并拼入 base_url），
        响应结构与 OpenAI 兼容端点不同：{"result": {"models": [...]}}。

        Args:
            model_type: 模型类型过滤（chat/image/video）

        Returns:
            模型信息列表
        """
        client = self._get_client()
        response = await client.get(url="/models/search")
        data = response.json()
        # Cloudflare 响应: {"result": {"models": [...]}}，兼容 result 为 list 的情况
        result = data.get("result", {})
        if isinstance(result, dict):
            items_data = result
            items_key = "models"
        else:
            items_data = {"data": result}
            items_key = "data"
        return self._parse_models_response(
            data=items_data,
            provider="cloudflareai",
            model_type=model_type,
            items_key=items_key,
        )

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Cloudflare API token")
        if response.status_code == 429:
            raise RateLimitError(message="Cloudflare AI rate limit exceeded")

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
        # Cloudflare 响应格式: { "result": { "response": "..." } }
        result = data.get("result", {})
        content = (
            result.get("response", "") if isinstance(result, dict) else str(result)
        )

        chat_message = ChatMessage(role="assistant", content=content)
        chat_choice = ChatChoice(
            index=0,
            message=chat_message,
            finish_reason="stop",
        )

        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("timestamp", current_timestamp()),
            model=model,
            choices=[chat_choice],
            usage=usage,
        )

    def _parse_chunk(self, data_str: str, model: str) -> ChatCompletionChunk | None:
        import json

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return None

        result = data.get("result", {})
        content = result.get("response", "") if isinstance(result, dict) else ""

        delta_message = ChatMessage(role="assistant", content=content)
        return ChatCompletionChunk(
            id=data.get("id", generate_id("chatcmpl")),
            created=current_timestamp(),
            model=model,
            choices=[
                ChatCompletionDelta(
                    index=0,
                    delta=delta_message,
                    finish_reason=None,
                )
            ],
        )


# ==================== 注册适配器 ====================


AdapterFactory.register("siliconflow", SiliconFlowAdapter)
AdapterFactory.register("sf", SiliconFlowAdapter)  # 别名
AdapterFactory.register("togetherai", TogetherAIAdapter)
AdapterFactory.register("together", TogetherAIAdapter)  # 别名
AdapterFactory.register("fireworksai", FireworksAIAdapter)
AdapterFactory.register("fireworks", FireworksAIAdapter)  # 别名
AdapterFactory.register("cloudflareai", CloudflareAIAdapter)
AdapterFactory.register("cloudflare", CloudflareAIAdapter)  # 别名
AdapterFactory.register("workersai", CloudflareAIAdapter)  # 别名
