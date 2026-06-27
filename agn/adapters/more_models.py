"""
AGN-SDK 更多主流模型适配器

支持：DeepSeek、阶跃星辰(StepFun)、Mistral AI、Cohere、Perplexity

官方文档：
- DeepSeek: https://api-docs.deepseek.com/
- StepFun: https://platform.stepfun.com/docs/
- Mistral: https://docs.mistral.ai/
- Cohere: https://docs.cohere.com/
- Perplexity: https://docs.perplexity.ai/
"""

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
    ChatUsage,
)
from agn.models.common import ModelInfo, ProviderConfig

logger = logging.getLogger(__name__)


# ==================== DeepSeek 适配器 ====================


class DeepSeekAdapter(BaseAdapter):
    """
    DeepSeek 适配器

    官方 API 规范（OpenAI 兼容）：
    - Base URL: https://api.deepseek.com
    - Chat: POST /chat/completions
    - 认证: Bearer Token (DeepSeek API Key)
    - 文档: https://api-docs.deepseek.com/
    - 特点: 支持思考模式(reasoning_effort)、深度推理
    """

    provider_type = "deepseek"
    provider_name = "DeepSeek"
    supported_capabilities = ["chat", "vision"]

    DEFAULT_BASE_URL = "https://api.deepseek.com"

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
        body = self._build_request_body(model, messages, kwargs)

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"DeepSeek chat failed: {e}")
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
        body = self._build_request_body(model, messages, kwargs)
        body["stream"] = True

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
            logger.error(f"DeepSeek stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="DeepSeek does not support direct image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="DeepSeek does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="DeepSeek does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        调用 GET /models 实时拉取（OpenAI 兼容端点），不再使用硬编码示例。

        Args:
            model_type: 模型类型过滤（chat/image/video）

        Returns:
            模型信息列表
        """
        client = self._get_client()
        response = await client.get(url="/models")
        return self._parse_models_response(
            data=response.json(),
            provider="deepseek",
            model_type=model_type,
        )

    def _build_request_body(
        self,
        model: str,
        messages: list[ChatMessage],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """构建请求体"""
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                msg.model_dump() if hasattr(msg, "model_dump") else msg
                for msg in messages
            ],
        }

        # DeepSeek 特有参数
        if reasoning_effort := kwargs.get("reasoning_effort"):
            body["reasoning_effort"] = reasoning_effort
            # 启用思考模式
            if "thinking" not in body:
                body["thinking"] = {"type": "enabled"}

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens
        if top_p := kwargs.get("top_p"):
            body["top_p"] = top_p

        return body

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid DeepSeek API key")
        if response.status_code == 429:
            raise RateLimitError(message="DeepSeek rate limit exceeded")

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

        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        chat_choices: list[ChatChoice] = []
        for i, c in enumerate(choices):
            msg_data = c.get("message", {})
            chat_choices.append(
                ChatChoice(
                    index=c.get("index", i),
                    message=ChatMessage(
                        role=msg_data.get("role", "assistant"),
                        content=msg_data.get("content", ""),
                    ),
                    finish_reason=c.get("finish_reason"),
                )
            )

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


# ==================== 阶跃星辰 StepFun 适配器 ====================


class StepFunAdapter(BaseAdapter):
    """
    阶跃星辰 StepFun 适配器

    官方 API 规范（OpenAI 兼容）：
    - Base URL: https://api.stepfun.com 或 https://api.stepfun.ai
    - Chat: POST /v1/chat/completions
    - 认证: Bearer Token
    - 文档: https://platform.stepfun.com/docs/
    - 特点: 支持 Step 系列模型、视觉理解、函数调用
    """

    provider_type = "stepfun"
    provider_name = "阶跃星辰 StepFun"
    supported_capabilities = ["chat", "vision"]

    DEFAULT_BASE_URL = "https://api.stepfun.com"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url + "/v1",
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

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"StepFun chat failed: {e}")
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
            logger.error(f"StepFun stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="StepFun image generation not implemented",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="StepFun video generation not implemented",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="StepFun video generation not implemented",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        调用 GET /models 实时拉取（OpenAI 兼容端点），不再使用硬编码示例。
        base_url 已包含 /v1 前缀，故请求路径为 /models。

        Args:
            model_type: 模型类型过滤（chat/image/video）

        Returns:
            模型信息列表
        """
        client = self._get_client()
        response = await client.get(url="/models")
        return self._parse_models_response(
            data=response.json(),
            provider="stepfun",
            model_type=model_type,
        )

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid StepFun API key")
        if response.status_code == 429:
            raise RateLimitError(message="StepFun rate limit exceeded")

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

        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        chat_choices: list[ChatChoice] = []
        for i, c in enumerate(choices):
            msg_data = c.get("message", {})
            chat_choices.append(
                ChatChoice(
                    index=c.get("index", i),
                    message=ChatMessage(
                        role=msg_data.get("role", "assistant"),
                        content=msg_data.get("content", ""),
                    ),
                    finish_reason=c.get("finish_reason"),
                )
            )

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


# ==================== Mistral AI 适配器 ====================


class MistralAdapter(BaseAdapter):
    """
    Mistral AI 适配器

    官方 API 规范（OpenAI 兼容）：
    - Base URL: https://api.mistral.ai/v1
    - Chat: POST /chat/completions
    - 认证: Bearer Token
    - 文档: https://docs.mistral.ai/
    - 特点: 欧洲顶级开源模型，支持 Mistral、Nemotron、Mixtral 系列
    """

    provider_type = "mistral"
    provider_name = "Mistral AI"
    supported_capabilities = ["chat", "vision"]

    DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

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
            logger.error(f"Mistral chat failed: {e}")
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
            logger.error(f"Mistral stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Mistral image generation not implemented",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Mistral video generation not implemented",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Mistral video generation not implemented",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        调用 GET /models 实时拉取（OpenAI 兼容端点），不再使用硬编码示例。
        base_url 已包含 /v1 前缀，故请求路径为 /models。

        Args:
            model_type: 模型类型过滤（chat/image/video）

        Returns:
            模型信息列表
        """
        client = self._get_client()
        response = await client.get(url="/models")
        return self._parse_models_response(
            data=response.json(),
            provider="mistral",
            model_type=model_type,
        )

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Mistral API key")
        if response.status_code == 429:
            raise RateLimitError(message="Mistral rate limit exceeded")

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

        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        chat_choices: list[ChatChoice] = []
        for i, c in enumerate(choices):
            msg_data = c.get("message", {})
            chat_choices.append(
                ChatChoice(
                    index=c.get("index", i),
                    message=ChatMessage(
                        role=msg_data.get("role", "assistant"),
                        content=msg_data.get("content", ""),
                    ),
                    finish_reason=c.get("finish_reason"),
                )
            )

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


# ==================== Cohere 适配器 ====================


class CohereAdapter(BaseAdapter):
    """
    Cohere 适配器

    官方 API 规范：
    - Base URL: https://api.cohere.ai/v1
    - Chat: POST /chat (非标准 OpenAI 格式)
    - 认证: Bearer Token
    - 文档: https://docs.cohere.com/
    - 特点: Command R+ 企业级 RAG 模型、多语言支持
    """

    provider_type = "cohere"
    provider_name = "Cohere"
    supported_capabilities = ["chat", "vision"]

    DEFAULT_BASE_URL = "https://api.cohere.ai/v1"

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
                "Accept": "application/json",
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
        self, messages: list[ChatMessage]
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
            elif role == "user":
                converted.append({"role": "USER", "content": content})
            else:
                converted.append({"role": "CHATBOT", "content": content})

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
            "message": (
                converted_messages[-1].get("content", "") if converted_messages else ""
            ),
            "chat_history": (
                converted_messages[:-1] if len(converted_messages) > 1 else []
            ),
        }

        if system_prompt:
            body["system_prompt"] = system_prompt

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens

        try:
            response = await client.post("/chat", json=body)
        except Exception as e:
            logger.error(f"Cohere chat failed: {e}")
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
            "message": (
                converted_messages[-1].get("content", "") if converted_messages else ""
            ),
            "chat_history": (
                converted_messages[:-1] if len(converted_messages) > 1 else []
            ),
            "stream": True,
        }

        if system_prompt:
            body["system_prompt"] = system_prompt

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature

        try:
            async with client.stream("POST", "/chat", json=body) as response:
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
            logger.error(f"Cohere stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Cohere does not support direct image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Cohere does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Cohere does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        调用 GET /models 实时拉取。Cohere v1 响应中模型列表位于 "models" 键下，
        且模型 ID 字段名为 "name"（非标准 OpenAI "id"），需预处理转换为统一 "id"
        字段以复用基类 _parse_models_response 解析逻辑。

        Args:
            model_type: 模型类型过滤（chat/image/video）

        Returns:
            模型信息列表
        """
        client = self._get_client()
        response = await client.get(url="/models")
        data = response.json()
        # Cohere v1 用 "name" 作为模型 ID，统一转换为 "id" 字段
        items = data.get("models", [])
        for item in items:
            if "id" not in item and "name" in item:
                item["id"] = item["name"]
        return self._parse_models_response(
            data={"data": items},
            provider="cohere",
            model_type=model_type,
        )

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Cohere API key")
        if response.status_code == 429:
            raise RateLimitError(message="Cohere rate limit exceeded")

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("error", {}).get("message")
                or error_data.get("message")
                or error_data.get("error", {}).get("type")
                or error_data.get("error")
                or f"HTTP {response.status_code}"
            )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=error_msg, status_code=response.status_code)

    def _parse_response(self, data: dict[str, Any], model: str) -> ChatCompletion:
        text = data.get("text", "")

        usage_data = data.get("usage")
        usage = None
        if usage_data:
            prompt_tokens = usage_data.get("tokens", {}).get("input_tokens", 0)
            completion_tokens = usage_data.get("tokens", {}).get("output_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens
            usage = ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        chat_choice = ChatChoice(
            index=0,
            message=ChatMessage(role="assistant", content=text),
            finish_reason="stop",
        )

        return ChatCompletion(
            id=data.get("id", generate_id("chatcmpl")),
            created=data.get("created_at", current_timestamp()),
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

        event_type = data.get("event_type", "")
        if event_type == "text-generation":
            text = data.get("text", "")
            delta_message = ChatMessage(role="assistant", content=text)
            return ChatCompletionChunk(
                id=data.get("generation_id", generate_id("chatcmpl")),
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
        elif event_type == "stream-end":
            delta_message = ChatMessage(role="assistant", content="")
            return ChatCompletionChunk(
                id=data.get("generation_id", generate_id("chatcmpl")),
                created=current_timestamp(),
                model=model,
                choices=[
                    ChatCompletionDelta(
                        index=0,
                        delta=delta_message,
                        finish_reason="stop",
                    )
                ],
            )

        return None


# ==================== Perplexity 适配器 ====================


class PerplexityAdapter(BaseAdapter):
    """
    Perplexity AI 适配器

    官方 API 规范（OpenAI 兼容）：
    - Base URL: https://api.perplexity.ai
    - Chat: POST /chat/completions
    - 认证: Bearer Token
    - 文档: https://docs.perplexity.ai/
    - 特点: AI 搜索、Sonar 模型、联网搜索能力
    """

    provider_type = "perplexity"
    provider_name = "Perplexity AI"
    supported_capabilities = ["chat", "vision"]

    DEFAULT_BASE_URL = "https://api.perplexity.ai"

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

        # Perplexity 特有参数
        if extra_body := kwargs.get("extra_body"):
            body["extra_body"] = extra_body

        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"Perplexity chat failed: {e}")
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
            logger.error(f"Perplexity stream chat failed: {e}")
            raise

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Perplexity does not support direct image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Perplexity does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Perplexity does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        调用 GET /models 实时拉取（OpenAI 兼容端点），不再使用硬编码示例。

        Args:
            model_type: 模型类型过滤（chat/image/video）

        Returns:
            模型信息列表
        """
        client = self._get_client()
        response = await client.get(url="/models")
        return self._parse_models_response(
            data=response.json(),
            provider="perplexity",
            model_type=model_type,
        )

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Perplexity API key")
        if response.status_code == 429:
            raise RateLimitError(message="Perplexity rate limit exceeded")

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

        usage_data = data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        chat_choices: list[ChatChoice] = []
        for i, c in enumerate(choices):
            msg_data = c.get("message", {})
            chat_choices.append(
                ChatChoice(
                    index=c.get("index", i),
                    message=ChatMessage(
                        role=msg_data.get("role", "assistant"),
                        content=msg_data.get("content", ""),
                    ),
                    finish_reason=c.get("finish_reason"),
                )
            )

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


# ==================== 注册适配器 ====================


AdapterFactory.register("deepseek", DeepSeekAdapter)
AdapterFactory.register("stepfun", StepFunAdapter)
AdapterFactory.register("step", StepFunAdapter)  # 别名
AdapterFactory.register("mistral", MistralAdapter)
AdapterFactory.register("cohere", CohereAdapter)
AdapterFactory.register("perplexity", PerplexityAdapter)
