"""
AGN-SDK Pika 适配器

实现 Pika API 的统一调用。

Pika API 参考（基于公开文档）：
- 创建视频任务: POST /v1/generations
- 查询任务状态: GET /v1/generations/{id}
- 认证: Bearer Token
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
from agn.models.chat import ChatCompletion, ChatMessage
from agn.models.common import ModelInfo, ProviderConfig
from agn.models.image import ImageGenerationResult
from agn.models.video import VideoStatus, VideoTask

logger = logging.getLogger(__name__)

# Pika 默认 Base URL
DEFAULT_BASE_URL = "https://api.pika.art/v1"


class PikaAdapter(BaseAdapter):
    """
    Pika 适配器

    实现对 Pika API 的统一调用。
    主要支持视频生成（Pika 1.0）。
    """

    provider_type = "pika"
    provider_name = "Pika"
    supported_capabilities = ["video"]

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

    # ==================== 文本对话（不支持）====================

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatCompletion:
        """
        文本对话

        Pika 不支持文本对话，此方法抛出错误。
        """
        raise UnsupportedCapabilityError(
            message="Pika does not support chat",
            details={"provider": self.provider_type, "capability": "chat"},
        )

    # ==================== 图像生成（不支持）====================

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        图像生成

        Pika 不支持直接图像生成，此方法抛出错误。
        """
        raise UnsupportedCapabilityError(
            message="Pika does not support direct image generation",
            details={"provider": self.provider_type, "capability": "image"},
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
            model: 模型名称（pika-1.0）
            prompt: 提示词
            **kwargs: 其他参数
                - mode: "text2video" 或 "image2video"
                - width: 视频宽度
                - height: 视频高度
                - reference_images: 参考图像列表（image2video 模式）
                - seed: 随机种子
                - aspect_ratio: 宽高比（如 "16:9"、"9:16"）
                - duration: 视频时长（秒）
                - negative_prompt: 负面提示词

        Returns:
            视频任务信息
        """
        client = self._get_client()

        mode = kwargs.get("mode", "text2video")
        reference_images = kwargs.get("reference_images")

        # 构建请求体
        body: dict[str, Any] = {
            "model": model,
            "prompt_text": prompt,
        }

        # 图生视频模式
        if mode == "image2video" and reference_images:
            body["prompt_image"] = reference_images[0]

        # 添加可选参数
        if width := kwargs.get("width"):
            body["width"] = width
        if height := kwargs.get("height"):
            body["height"] = height
        if seed := kwargs.get("seed"):
            body["seed"] = seed
        if aspect_ratio := kwargs.get("aspect_ratio"):
            body["aspect_ratio"] = aspect_ratio
        if duration := kwargs.get("duration"):
            body["duration"] = duration
        if negative_prompt := kwargs.get("negative_prompt"):
            body["negative_prompt_text"] = negative_prompt

        # 透传额外参数
        extra_body = kwargs.get("extra_body")
        if extra_body and isinstance(extra_body, dict):
            body.update(extra_body)

        logger.debug(f"Sending video request to Pika: model={model}, mode={mode}")

        try:
            response = await client.post("/generations", json=body)
        except Exception as e:
            logger.error(f"Pika video create failed: {e}")
            raise

        self._handle_pika_error(response)
        data = response.json()

        # 提取任务 ID
        task_id = (
            data.get("id")
            or data.get("generation_id")
            or data.get("taskId")
            or generate_id("vid")
        )

        return VideoTask(
            task_id=task_id,
            model=model,
            status=data.get("status", "pending"),
            created_at=data.get("created_at") or current_timestamp(),
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
            model: 模型名称（可选）

        Returns:
            视频任务状态
        """
        client = self._get_client()

        logger.debug(f"Polling Pika task: {task_id}")

        try:
            response = await client.get(f"/generations/{task_id}")
        except Exception as e:
            logger.error(f"Pika video poll failed: {e}")
            raise

        self._handle_pika_error(response)
        data = response.json()

        # 映射状态
        status_raw = data.get("status", "pending").lower()
        status = self._map_pika_status(status_raw)

        # 提取视频 URL
        video_url = (
            data.get("video_url")
            or data.get("url")
            or (data.get("output") or {}).get("video_url")
            or (data.get("output") or {}).get("url")
            or (
                ((data.get("results") or [{}])[0]).get("url")
                if data.get("results")
                else None
            )
        )

        # 提取错误信息
        error = (
            data.get("error") or data.get("error_message") or data.get("failure_reason")
        )

        return VideoStatus(
            task_id=task_id,
            status=status,
            video_url=video_url,
            progress=data.get("progress"),
            error=error,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    def _map_pika_status(self, raw_status: str) -> str:
        """
        映射 Pika 状态到标准状态

        Args:
            raw_status: Pika 原始状态

        Returns:
            标准状态值
        """
        status_map = {
            "pending": "pending",
            "queued": "pending",
            "in_queue": "pending",
            "processing": "processing",
            "in_progress": "processing",
            "generating": "processing",
            "completed": "success",
            "finished": "success",
            "succeeded": "success",
            "success": "success",
            "failed": "failed",
            "error": "failed",
            "failure": "failed",
            "cancelled": "failed",
        }
        return status_map.get(raw_status, "pending")

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
                id="pika-1.0",
                name="Pika 1.0",
                type="video",
                provider="pika",
                capabilities=["text2video", "image2video"],
                description="Pika 1.0 video generation model",
            ),
            ModelInfo(
                id="pika-2",
                name="Pika 2.0",
                type="video",
                provider="pika",
                capabilities=["text2video", "image2video"],
                description="Pika 2.0 latest video generation model",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    # ==================== 辅助方法 ====================

    def _handle_pika_error(self, response: httpx.Response) -> None:
        """处理 Pika 错误响应"""
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Pika API key")
        if response.status_code == 429:
            raise RateLimitError(message="Pika rate limit exceeded")
        if response.status_code == 404:
            raise APIError(message="Generation not found", status_code=404)

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("error")
                or error_data.get("message")
                or error_data.get("detail")
                or f"HTTP {response.status_code}"
            )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=error_msg, status_code=response.status_code)


# 注册适配器
AdapterFactory.register("pika", PikaAdapter)
