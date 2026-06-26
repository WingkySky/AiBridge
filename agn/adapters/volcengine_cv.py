"""
AGN-SDK 火山引擎方舟（Seedream/Seedance）适配器

实现火山引擎方舟平台图像和视频生成 API 的统一调用。

官方 API 文档：https://www.volcengine.com/docs/82379
- Base URL: https://ark.cn-beijing.volces.com/api/v3
- 图像生成 (Seedream): POST /images/generations (同步)
- 视频生成 (Seedance): POST /videos/generations (异步任务)
- 查询视频任务: GET /videos/generations/{task_id}
- 认证: Bearer Token (火山引擎 API Key)
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
from agn.models.image import ImageData, ImageGenerationResult
from agn.models.video import VideoStatus, VideoTask

logger = logging.getLogger(__name__)


class VolcengineCVAdapter(BaseAdapter):
    """
    火山引擎方舟 CV 适配器（Seedream 图像 / Seedance 视频）

    支持：
    - Seedream 系列：文生图
    - Seedance 系列：文生视频、图生视频
    """

    provider_type = "volcengine_cv"
    provider_name = "火山引擎 Seedream/Seedance"
    supported_capabilities = ["image", "video"]

    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

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

    # ==================== 文本对话（不支持）====================

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatCompletion:
        """文本对话（不支持，请使用 DoubaoAdapter）"""
        raise UnsupportedCapabilityError(
            message="VolcengineCV adapter does not support chat. Use DoubaoAdapter for Doubao chat models.",
            details={"provider": self.provider_type, "capability": "chat"},
        )

    # ==================== 图像生成 (Seedream) ====================

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        生成图像（Seedream 文生图）

        Args:
            model: 模型端点 ID (用户在火山方舟创建的接入点 ID)
            prompt: 提示词
            **kwargs:
                - size: 图像尺寸，默认 "1024x1024"
                - n: 生成数量，默认 1
                - response_format: "url" 或 "b64_json"
                - negative_prompt: 负面提示词
                - seed: 随机种子

        Returns:
            图像生成结果
        """
        client = self._get_client()

        size = kwargs.get("size", "1024x1024")
        n = kwargs.get("n", 1)
        response_format = kwargs.get("response_format", "url")

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": n,
            "response_format": response_format,
        }

        if negative_prompt := kwargs.get("negative_prompt"):
            body["negative_prompt"] = negative_prompt
        if seed := kwargs.get("seed"):
            body["seed"] = seed

        try:
            response = await client.post("/images/generations", json=body)
        except Exception as e:
            logger.error(f"Seedream image generation failed: {e}")
            raise

        self._handle_error(response)
        data = response.json()
        return self._parse_image_response(data, model)

    # ==================== 视频生成 (Seedance) ====================

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> VideoTask:
        """
        创建视频生成任务（Seedance）

        Args:
            model: 模型端点 ID
            prompt: 提示词
            **kwargs:
                - mode: "text2video" (默认) 或 "image2video"
                - reference_images: 参考图像 URL 列表 (image2video)
                - negative_prompt: 负面提示词
                - duration: 视频时长（秒）
                - aspect_ratio: 宽高比，"16:9", "9:16", "1:1"
                - resolution: 分辨率，"720p", "1080p"
                - seed: 随机种子

        Returns:
            视频任务信息
        """
        client = self._get_client()

        mode = kwargs.get("mode", "text2video")
        reference_images = kwargs.get("reference_images", [])

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
        }

        if negative_prompt := kwargs.get("negative_prompt"):
            body["negative_prompt"] = negative_prompt
        if duration := kwargs.get("duration"):
            body["duration"] = duration
        if aspect_ratio := kwargs.get("aspect_ratio"):
            body["aspect_ratio"] = aspect_ratio
        if resolution := kwargs.get("resolution"):
            body["resolution"] = resolution
        if seed := kwargs.get("seed"):
            body["seed"] = seed

        if mode == "image2video" and reference_images:
            body["image_url"] = reference_images[0] if reference_images else None

        try:
            response = await client.post("/videos/generations", json=body)
        except Exception as e:
            logger.error(f"Seedance video create failed: {e}")
            raise

        self._handle_error(response)
        data = response.json()

        task_id = data.get("id", generate_id("vtask"))
        raw_status = data.get("status", "queued")

        return VideoTask(
            task_id=task_id,
            model=model,
            status=self._map_video_status(raw_status),
            created_at=current_timestamp(),
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        """
        查询视频生成任务状态

        Args:
            task_id: 任务 ID
            model: 模型名称（可选，用于返回信息）

        Returns:
            视频任务状态
        """
        client = self._get_client()

        try:
            response = await client.get(f"/videos/generations/{task_id}")
        except Exception as e:
            logger.error(f"Seedance video poll failed: {e}")
            raise

        self._handle_error(response)
        data = response.json()
        return self._parse_video_status(data, task_id)

    # ==================== 模型列表 ====================

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        列出支持的模型

        注意：火山方舟使用"接入点（Endpoint）"模式，模型 ID 为用户在控制台创建的接入点 ID。
        此处返回模型系列名称供参考。
        """
        models = [
            ModelInfo(
                id="seedream-5.0",
                name="Seedream 5.0",
                type="image",
                provider="volcengine_cv",
                capabilities=["image", "text2image"],
                description="豆包 Seedream 5.0 最新图像生成模型",
            ),
            ModelInfo(
                id="seedream-4.0",
                name="Seedream 4.0",
                type="image",
                provider="volcengine_cv",
                capabilities=["image", "text2image"],
                description="豆包 Seedream 4.0 图像生成模型",
            ),
            ModelInfo(
                id="seedream-3.0",
                name="Seedream 3.0",
                type="image",
                provider="volcengine_cv",
                capabilities=["image", "text2image"],
                description="豆包 Seedream 3.0 图像生成模型",
            ),
            ModelInfo(
                id="seedance-2.0",
                name="Seedance 2.0",
                type="video",
                provider="volcengine_cv",
                capabilities=["video", "text2video", "image2video"],
                description="豆包 Seedance 2.0 视频生成模型",
            ),
            ModelInfo(
                id="seedance-2.0-mini",
                name="Seedance 2.0 Mini",
                type="video",
                provider="volcengine_cv",
                capabilities=["video", "text2video", "image2video"],
                description="豆包 Seedance 2.0 Mini 快速版视频生成",
            ),
            ModelInfo(
                id="seedance-1.0",
                name="Seedance 1.0",
                type="video",
                provider="volcengine_cv",
                capabilities=["video", "text2video"],
                description="豆包 Seedance 1.0 视频生成模型",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    # ==================== 响应解析 ====================

    def _parse_image_response(
        self, data: dict[str, Any], model: str
    ) -> ImageGenerationResult:
        """解析图像生成响应"""
        images: list[ImageData] = []
        for item in data.get("data", []):
            images.append(
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
            data=images,
        )

    def _parse_video_status(self, data: dict[str, Any], task_id: str) -> VideoStatus:
        """解析视频任务状态响应"""
        raw_status = data.get("status", "")
        status = self._map_video_status(raw_status)

        video_url: str | None = None
        if status == "success":
            video_url = (
                data.get("video_url")
                or data.get("output", {}).get("video_url")
                or data.get("url")
            )

        error_msg: str | None = None
        if status == "failed":
            error_msg = (
                data.get("error", {}).get("message")
                or data.get("message")
                or data.get("error")
            )

        return VideoStatus(
            task_id=task_id,
            status=status,
            video_url=video_url,
            progress=data.get("progress"),
            error=error_msg,
            created_at=data.get("created"),
            updated_at=data.get("updated", current_timestamp()),
        )

    # ==================== 状态映射 ====================

    def _map_video_status(self, raw_status: str) -> str:
        """映射火山引擎视频状态到统一状态"""
        status_map = {
            "queued": "pending",
            "pending": "pending",
            "submitted": "pending",
            "processing": "processing",
            "running": "processing",
            "in_progress": "processing",
            "succeeded": "success",
            "success": "success",
            "completed": "success",
            "failed": "failed",
            "error": "failed",
            "cancelled": "failed",
        }
        return status_map.get(raw_status.lower(), "pending")

    # ==================== 错误处理 ====================

    def _handle_error(self, response: httpx.Response) -> None:
        """处理错误响应"""
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(
                message="Invalid Volcengine API key or access denied"
            )
        if response.status_code == 429:
            raise RateLimitError(message="Volcengine rate limit exceeded")
        if response.status_code == 404:
            raise APIError(
                message="Model endpoint not found. Check your endpoint ID in Volcengine Ark console.",
                status_code=404,
            )

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


# 注册适配器
AdapterFactory.register("volcengine_cv", VolcengineCVAdapter)
AdapterFactory.register("seedream", VolcengineCVAdapter)
AdapterFactory.register("seedance", VolcengineCVAdapter)
