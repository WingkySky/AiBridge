"""
AGN-SDK Kling (可灵) 适配器

实现快手 Kling AI 视频生成 API 的统一调用。

官方 API 文档：https://app.klingai.com/docs/api
- Base URL: https://api.klingai.com/v1
- 文生视频: POST /videos/generations
- 图生视频: POST /videos/image2video
- 查询任务: GET /videos/generations/{task_id}
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

# Kling 默认配置
DEFAULT_BASE_URL = "https://api.klingai.com/v1"


class KlingAdapter(BaseAdapter):
    """
    Kling (可灵) 适配器

    实现对快手 Kling AI 视频生成 API 的统一调用。
    支持：
    - kling-v1（1.0 标准版）
    - kling-v1-5（1.5 标准版）
    - kling-v2（2.0 最新版）
    - 文生视频、图生视频
    """

    provider_type = "kling"
    provider_name = "可灵 Kling"
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
        """文本对话（不支持）"""
        raise UnsupportedCapabilityError(
            message="Kling does not support chat",
            details={"provider": self.provider_type, "capability": "chat"},
        )

    # ==================== 图像生成（不支持）====================

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """图像生成（不支持，Kolors 是单独模型）"""
        raise UnsupportedCapabilityError(
            message="Kling does not support direct image generation (Kolors is separate)",
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
            model: 模型名称（kling-v1, kling-v1-5, kling-v2）
            prompt: 提示词
            **kwargs: 其他参数
                - mode: "text2video"（默认）或 "image2video"
                - reference_images: 参考图像列表（image2video 模式）
                - negative_prompt: 负面提示词
                - cfg_scale: CFG 强度（0-1）
                - duration: 视频时长（5 或 10 秒，v2 支持更长）
                - aspect_ratio: 宽高比（"16:9", "9:16", "1:1"）
                - camera_control: 相机控制配置

        Returns:
            视频任务信息
        """
        client = self._get_client()

        mode = kwargs.get("mode", "text2video")
        reference_images = kwargs.get("reference_images", [])

        # 构建请求体
        body: dict[str, Any] = {
            "model_name": model,
            "prompt": prompt,
        }

        # 负面提示词
        if negative_prompt := kwargs.get("negative_prompt"):
            body["negative_prompt"] = negative_prompt

        # 配置参数
        params: dict[str, Any] = {}
        if cfg_scale := kwargs.get("cfg_scale"):
            params["cfg_scale"] = cfg_scale
        if duration := kwargs.get("duration"):
            params["duration"] = duration
        if aspect_ratio := kwargs.get("aspect_ratio"):
            params["aspect_ratio"] = aspect_ratio
        if camera_control := kwargs.get("camera_control"):
            params["camera_control"] = camera_control
        if mode == "image2video" and reference_images:
            if reference_images:
                body["image"] = reference_images[0]
                params["mode"] = (
                    "std" if not kwargs.get("mode") else kwargs.get("mode", "std")
                )

        if params:
            body.update(params)

        # 选择端点
        if mode == "image2video" and reference_images:
            endpoint = "/videos/image2video"
        else:
            endpoint = "/videos/generations"

        # 透传额外参数
        extra_body = kwargs.get("extra_body")
        if extra_body and isinstance(extra_body, dict):
            body.update(extra_body)

        logger.debug(f"Sending video request to Kling: model={model}, mode={mode}")

        try:
            response = await client.post(endpoint, json=body)
        except Exception as e:
            logger.error(f"Kling video create failed: {e}")
            raise

        self._handle_kling_error(response)
        data = response.json()

        task_data = data.get("data", {})
        task_id = task_data.get("task_id") or data.get("task_id") or generate_id("vid")
        raw_status = task_data.get("task_status", "pending")
        # 使用状态映射确保返回标准状态值
        status = self._map_kling_status(raw_status) if raw_status else "pending"

        return VideoTask(
            task_id=task_id,
            model=model,
            status=status,
            created_at=current_timestamp(),
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

        logger.debug(f"Polling Kling task: {task_id}")

        try:
            response = await client.get(f"/videos/generations/{task_id}")
        except Exception as e:
            logger.error(f"Kling video poll failed: {e}")
            raise

        self._handle_kling_error(response)
        data = response.json()

        task_data = data.get("data", data)
        task_status = task_data.get("task_status", "").lower()
        status = self._map_kling_status(task_status)

        # 提取视频 URL
        video_url = None
        task_result = task_data.get("task_result", {})
        if isinstance(task_result, dict):
            videos = task_result.get("videos", [])
            if videos and isinstance(videos, list):
                video_url = videos[0].get("url")

        # 错误信息
        error = task_data.get("task_status_msg") or task_data.get("error")

        return VideoStatus(
            task_id=task_id,
            status=status,
            video_url=video_url,
            progress=100 if status == "success" else (0 if status == "pending" else 50),
            error=error,
            created_at=task_data.get("created_at"),
            updated_at=task_data.get("updated_at") or current_timestamp(),
        )

    def _map_kling_status(self, raw_status: str) -> str:
        """
        映射 Kling 状态到标准状态

        Kling 状态：
        - submitted / queued → pending
        - processing → processing
        - succeed → success
        - failed → failed
        """
        status_map = {
            "submitted": "pending",
            "queued": "pending",
            "processing": "processing",
            "succeed": "success",
            "success": "success",
            "failed": "failed",
            "error": "failed",
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
                id="kling-v1",
                name="Kling 1.0",
                type="video",
                provider="kling",
                capabilities=["text2video", "image2video"],
                description="Kling 1.0 标准版本",
            ),
            ModelInfo(
                id="kling-v1-5",
                name="Kling 1.5",
                type="video",
                provider="kling",
                capabilities=["text2video", "image2video"],
                description="Kling 1.5 改进版本",
            ),
            ModelInfo(
                id="kling-v2",
                name="Kling 2.0",
                type="video",
                provider="kling",
                capabilities=["text2video", "image2video"],
                description="Kling 2.0 最新版本，质量更好",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    # ==================== 辅助方法 ====================

    def _handle_kling_error(self, response: httpx.Response) -> None:
        """处理 Kling 错误响应"""
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Kling API key")
        if response.status_code == 429:
            raise RateLimitError(message="Kling rate limit exceeded or quota exhausted")

        try:
            error_data = response.json()
            error_code = error_data.get("code")
            error_msg = (
                error_data.get("message")
                or error_data.get("error")
                or error_data.get("msg")
                or f"HTTP {response.status_code}"
            )
            if error_code:
                error_msg = f"[{error_code}] {error_msg}"
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=error_msg, status_code=response.status_code)


# 注册适配器
AdapterFactory.register("kling", KlingAdapter)
