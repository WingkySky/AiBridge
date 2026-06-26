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
        models = [
            # 智谱 GLM 系列
            ModelInfo(
                id="Pro/zai-org/GLM-4.7",
                name="GLM-4.7 Pro",
                type="chat",
                provider="siliconflow",
                capabilities=["chat", "vision"],
                description="智谱 GLM-4.7 旗舰模型",
            ),
            ModelInfo(
                id="Pro/zai-org/GLM-5",
                name="GLM-5 Pro",
                type="chat",
                provider="siliconflow",
                capabilities=["chat", "vision"],
                description="智谱 GLM-5 最新旗舰",
            ),
            # DeepSeek 系列
            ModelInfo(
                id="deepseek-ai/DeepSeek-V3.2",
                name="DeepSeek V3.2",
                type="chat",
                provider="siliconflow",
                capabilities=["chat"],
                description="DeepSeek V3.2 模型",
            ),
            ModelInfo(
                id="Pro/deepseek-ai/DeepSeek-V3.2",
                name="DeepSeek V3.2 Pro",
                type="chat",
                provider="siliconflow",
                capabilities=["chat"],
                description="DeepSeek V3.2 Pro 版本",
            ),
            # Qwen 系列
            ModelInfo(
                id="Qwen/Qwen3-8B",
                name="Qwen 3 8B",
                type="chat",
                provider="siliconflow",
                capabilities=["chat"],
                description="Qwen 3 8B 模型",
            ),
            ModelInfo(
                id="Qwen/Qwen3-14B",
                name="Qwen 3 14B",
                type="chat",
                provider="siliconflow",
                capabilities=["chat"],
                description="Qwen 3 14B 模型",
            ),
            ModelInfo(
                id="Qwen/Qwen3-32B",
                name="Qwen 3 32B",
                type="chat",
                provider="siliconflow",
                capabilities=["chat"],
                description="Qwen 3 32B 模型",
            ),
            ModelInfo(
                id="Qwen/Qwen3.5-397B-A17B",
                name="Qwen 3.5 397B",
                type="chat",
                provider="siliconflow",
                capabilities=["chat"],
                description="Qwen 3.5 超大模型",
            ),
            # 混元系列
            ModelInfo(
                id="tencent/Hunyuan-A13B-Instruct",
                name="Hunyuan A13B",
                type="chat",
                provider="siliconflow",
                capabilities=["chat"],
                description="腾讯混元 A13B 指令模型",
            ),
            # 语音转文字模型
            ModelInfo(
                id="FunAudioLLM/SenseVoiceSmall",
                name="SenseVoice Small",
                type="audio",
                provider="siliconflow",
                capabilities=["audio_transcribe"],
                description="阿里通义 SenseVoice 多语言语音识别",
            ),
            ModelInfo(
                id="iic/SenseVoiceSmall",
                name="SenseVoice Small (iic)",
                type="audio",
                provider="siliconflow",
                capabilities=["audio_transcribe"],
                description="阿里通义 SenseVoice 语音识别",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

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
        models = [
            # Llama 系列
            ModelInfo(
                id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                name="Llama 4 Maverick",
                type="chat",
                provider="togetherai",
                capabilities=["chat"],
                description="Meta Llama 4 Maverick 128E",
            ),
            ModelInfo(
                id="meta-llama/Llama-4-Scout-17B-16E-Instruct-FP8",
                name="Llama 4 Scout",
                type="chat",
                provider="togetherai",
                capabilities=["chat"],
                description="Meta Llama 4 Scout 16E",
            ),
            ModelInfo(
                id="meta-llama/Llama-3-8b-chat-hf",
                name="Llama 3 8B",
                type="chat",
                provider="togetherai",
                capabilities=["chat"],
                description="Meta Llama 3 8B",
            ),
            ModelInfo(
                id="meta-llama/Llama-3-70b-chat-hf",
                name="Llama 3 70B",
                type="chat",
                provider="togetherai",
                capabilities=["chat"],
                description="Meta Llama 3 70B",
            ),
            ModelInfo(
                id="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                name="Llama 3.1 405B",
                type="chat",
                provider="togetherai",
                capabilities=["chat"],
                description="Meta Llama 3.1 405B",
            ),
            # Qwen 系列
            ModelInfo(
                id="Qwen/Qwen2.5-72B-Instruct-Turbo",
                name="Qwen 2.5 72B",
                type="chat",
                provider="togetherai",
                capabilities=["chat"],
                description="Qwen 2.5 72B 指令模型",
            ),
            ModelInfo(
                id="Qwen/Qwen2.5-7B-Instruct-Turbo",
                name="Qwen 2.5 7B",
                type="chat",
                provider="togetherai",
                capabilities=["chat"],
                description="Qwen 2.5 7B 指令模型",
            ),
            # DeepSeek 系列
            ModelInfo(
                id="deepseek-ai/DeepSeek-V3",
                name="DeepSeek V3",
                type="chat",
                provider="togetherai",
                capabilities=["chat"],
                description="DeepSeek V3 模型",
            ),
            ModelInfo(
                id="deepseek-ai/DeepSeek-Coder-V2-Instruct",
                name="DeepSeek Coder V2",
                type="chat",
                provider="togetherai",
                capabilities=["chat"],
                description="DeepSeek Coder V2 指令模型",
            ),
            # Mistral 系列
            ModelInfo(
                id="mistralai/Mixtral-8x22B-Instruct-v0.1",
                name="Mixtral 8x22B",
                type="chat",
                provider="togetherai",
                capabilities=["chat"],
                description="Mistral Mixtral 8x22B",
            ),
            # 语音转文字模型（Whisper）
            ModelInfo(
                id="openai/whisper-large-v3",
                name="Whisper Large v3",
                type="audio",
                provider="togetherai",
                capabilities=["audio_transcribe"],
                description="OpenAI Whisper Large v3 语音识别",
            ),
            ModelInfo(
                id="openai/whisper-large-v3-turbo",
                name="Whisper Large v3 Turbo",
                type="audio",
                provider="togetherai",
                capabilities=["audio_transcribe"],
                description="OpenAI Whisper Large v3 Turbo 快速语音识别",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

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
        models = [
            # Llama 系列
            ModelInfo(
                id="accounts/fireworks/models/llama-v3p1-405b-instruct",
                name="Llama 3.1 405B",
                type="chat",
                provider="fireworksai",
                capabilities=["chat"],
                description="Llama 3.1 405B Instruct",
            ),
            ModelInfo(
                id="accounts/fireworks/models/llama-v3p1-70b-instruct",
                name="Llama 3.1 70B",
                type="chat",
                provider="fireworksai",
                capabilities=["chat"],
                description="Llama 3.1 70B Instruct",
            ),
            ModelInfo(
                id="accounts/fireworks/models/llama-v3p1-8b-instruct",
                name="Llama 3.1 8B",
                type="chat",
                provider="fireworksai",
                capabilities=["chat"],
                description="Llama 3.1 8B Instruct",
            ),
            # Mixtral 系列
            ModelInfo(
                id="accounts/fireworks/models/mixtral-8x22b-instruct",
                name="Mixtral 8x22B",
                type="chat",
                provider="fireworksai",
                capabilities=["chat"],
                description="Mixtral 8x22B Instruct",
            ),
            ModelInfo(
                id="accounts/fireworks/models/mixtral-8x7b-instruct",
                name="Mixtral 8x7B",
                type="chat",
                provider="fireworksai",
                capabilities=["chat"],
                description="Mixtral 8x7B Instruct",
            ),
            # DeepSeek 系列
            ModelInfo(
                id="accounts/fireworks/models/deepseek-v3",
                name="DeepSeek V3",
                type="chat",
                provider="fireworksai",
                capabilities=["chat"],
                description="DeepSeek V3 模型",
            ),
            ModelInfo(
                id="accounts/fireworks/models/deepseek-r1",
                name="DeepSeek R1",
                type="chat",
                provider="fireworksai",
                capabilities=["chat"],
                description="DeepSeek R1 推理模型",
            ),
            # Qwen 系列
            ModelInfo(
                id="accounts/fireworks/models/qwen2p5-72b-instruct",
                name="Qwen 2.5 72B",
                type="chat",
                provider="fireworksai",
                capabilities=["chat"],
                description="Qwen 2.5 72B Instruct",
            ),
            ModelInfo(
                id="accounts/fireworks/models/qwen2p5-7b-instruct",
                name="Qwen 2.5 7B",
                type="chat",
                provider="fireworksai",
                capabilities=["chat"],
                description="Qwen 2.5 7B Instruct",
            ),
            # 语音转文字模型（Whisper）
            ModelInfo(
                id="accounts/fireworks/models/whisper-v3",
                name="Whisper v3",
                type="audio",
                provider="fireworksai",
                capabilities=["audio_transcribe"],
                description="OpenAI Whisper v3 语音识别",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

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
        models = [
            # Llama 系列
            ModelInfo(
                id="@cf/meta/llama-3.1-8b-instruct",
                name="Llama 3.1 8B",
                type="chat",
                provider="cloudflareai",
                capabilities=["chat"],
                description="Meta Llama 3.1 8B Instruct",
            ),
            ModelInfo(
                id="@cf/meta/llama-3.1-70b-instruct",
                name="Llama 3.1 70B",
                type="chat",
                provider="cloudflareai",
                capabilities=["chat"],
                description="Meta Llama 3.1 70B Instruct",
            ),
            ModelInfo(
                id="@cf/meta/llama-3-8b-instruct-lora",
                name="Llama 3 8B LoRA",
                type="chat",
                provider="cloudflareai",
                capabilities=["chat"],
                description="Meta Llama 3 8B LoRA",
            ),
            # Mistral 系列
            ModelInfo(
                id="@cf/mistral/mistral-7b-instruct-v0.2",
                name="Mistral 7B v0.2",
                type="chat",
                provider="cloudflareai",
                capabilities=["chat"],
                description="Mistral 7B Instruct v0.2",
            ),
            # DeepSeek
            ModelInfo(
                id="@cf/deepseek-ai/DeepSeek-V3-base",
                name="DeepSeek V3 Base",
                type="chat",
                provider="cloudflareai",
                capabilities=["chat"],
                description="DeepSeek V3 Base 模型",
            ),
            # Qwen 系列
            ModelInfo(
                id="@cf/qwen/qwen2.5-7b-instruct",
                name="Qwen 2.5 7B",
                type="chat",
                provider="cloudflareai",
                capabilities=["chat"],
                description="Qwen 2.5 7B Instruct",
            ),
            ModelInfo(
                id="@cf/qwen/qwen2.5-72b-instruct",
                name="Qwen 2.5 72B",
                type="chat",
                provider="cloudflareai",
                capabilities=["chat"],
                description="Qwen 2.5 72B Instruct",
            ),
            # Gemma 系列
            ModelInfo(
                id="@cf/google/gemma-2-2b-it",
                name="Gemma 2 2B IT",
                type="chat",
                provider="cloudflareai",
                capabilities=["chat"],
                description="Google Gemma 2 2B Instruct",
            ),
            ModelInfo(
                id="@cf/google/gemma-2-7b-it",
                name="Gemma 2 7B IT",
                type="chat",
                provider="cloudflareai",
                capabilities=["chat"],
                description="Google Gemma 2 7B Instruct",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

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
