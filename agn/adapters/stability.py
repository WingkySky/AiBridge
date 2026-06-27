"""
AGN-SDK Stability AI 适配器

实现 Stability AI API 的统一调用。

Stability AI API 参考：
- 图像生成: POST /v1/generation/{engine_id}/text-to-image
- 图像编辑: POST /v1/generation/{engine_id}/image-to-image
- 认证: Bearer Token
"""

import base64
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

# Stability AI 默认 Base URL 和引擎
DEFAULT_BASE_URL = "https://api.stability.ai"
DEFAULT_ENGINE = "stable-diffusion-xl-1024-v1-1"


class StabilityAdapter(BaseAdapter):
    """
    Stability AI 适配器

    实现对 Stability AI API 的统一调用。
    主要支持图像生成（SDXL、SD3）。
    """

    provider_type = "stability"
    provider_name = "Stability AI"
    supported_capabilities = ["image"]

    def __init__(
        self, config: ProviderConfig, default_engine: str | None = None
    ) -> None:
        """
        初始化适配器

        Args:
            config: Provider 配置
            default_engine: 默认引擎（可选，默认使用 SDXL 1.1）
        """
        super().__init__(config)
        self.base_url = config.base_url or DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self.default_engine = default_engine or DEFAULT_ENGINE
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """启动适配器"""
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
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

        Stability 不支持文本对话，此方法抛出错误。
        """
        raise UnsupportedCapabilityError(
            message="Stability AI does not support chat",
            details={"provider": self.provider_type, "capability": "chat"},
        )

    # ==================== 图像生成 ====================

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        创建图像生成任务

        Args:
            model: 模型名称（引擎 ID）
                - stable-diffusion-xl-1024-v1-1 (SDXL)
                - stable-diffusion-3-medium (SD3)
            prompt: 提示词
            **kwargs: 其他参数
                - negative_prompt: 负面提示词
                - width: 图像宽度
                - height: 图像高度
                - steps: 采样步数
                - seed: 随机种子
                - cfg_scale: CFG 强度
                - samples: 生成数量（1-10）
                - style_preset: 风格预设

        Returns:
            图像生成结果
        """
        client = self._get_client()

        engine = model or self.default_engine

        # 构建请求体
        body: dict[str, Any] = {
            "text_prompts": [{"text": prompt, "weight": 1}],
        }

        # 负面提示词
        if negative_prompt := kwargs.get("negative_prompt"):
            body["text_prompts"].append(
                {
                    "text": negative_prompt,
                    "weight": -1,
                }
            )

        # 尺寸
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        body["width"] = width
        body["height"] = height

        # 采样参数
        if steps := kwargs.get("steps"):
            body["steps"] = steps
        if seed := kwargs.get("seed"):
            body["seed"] = seed
        if cfg_scale := kwargs.get("cfg_scale"):
            body["cfg_scale"] = cfg_scale
        if samples := kwargs.get("samples"):
            body["samples"] = min(samples, 10)
        if style_preset := kwargs.get("style_preset"):
            body["style_preset"] = style_preset

        # 透传额外参数
        extra_body = kwargs.get("extra_body")
        if extra_body and isinstance(extra_body, dict):
            body.update(extra_body)

        logger.debug(f"Sending image request to Stability: engine={engine}")

        try:
            response = await client.post(
                f"/v1/generation/{engine}/text-to-image",
                json=body,
            )
        except Exception as e:
            logger.error(f"Stability image generate failed: {e}")
            raise

        self._handle_stability_error(response)
        data = response.json()

        # 提取图像
        artifacts = data.get("artifacts", [])
        if not artifacts:
            raise APIError(message="No image generated")

        images: list[ImageData] = []
        for artifact in artifacts:
            img_data = artifact.get("base64", "")
            if img_data:
                images.append(ImageData(b64_json=img_data))

        return ImageGenerationResult(
            id=generate_id("img"),
            created=current_timestamp(),
            model=engine,
            data=images,
        )

    async def image_edit(
        self,
        model: str,
        prompt: str,
        image: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        图像编辑（图生图）

        Args:
            model: 模型名称（引擎 ID）
            prompt: 提示词
            image: 源图像 URL 或 Base64
            **kwargs: 其他参数
                - mask: 蒙版图像 URL 或 Base64
                - negative_prompt: 负面提示词
                - width: 图像宽度
                - height: 图像高度
                - steps: 采样步数
                - seed: 随机种子
                - cfg_scale: CFG 强度
                - samples: 生成数量

        Returns:
            图像生成结果
        """
        client = self._get_client()

        engine = model or self.default_engine

        # 处理图像数据
        if image.startswith("data:"):
            img_data = image.split(",", 1)[1]
        elif image.startswith("http"):
            img_response = await client.get(image)
            img_data = base64.b64encode(img_response.content).decode()
        else:
            img_data = image

        # 构建 multipart 请求
        files = {
            "init_image": (
                "image.png",
                base64.b64decode(img_data),
                "image/png",
            ),
        }

        # 添加蒙版
        if mask := kwargs.get("mask"):
            if mask.startswith("data:"):
                mask_data = mask.split(",", 1)[1]
            elif mask.startswith("http"):
                mask_response = await client.get(mask)
                mask_data = base64.b64encode(mask_response.content).decode()
            else:
                mask_data = mask
            files["mask_image"] = (
                "mask.png",
                base64.b64decode(mask_data),
                "image/png",
            )

        # 构建请求数据
        data: dict[str, Any] = {
            "text_prompts": [{"text": prompt, "weight": 1}],
        }

        if negative_prompt := kwargs.get("negative_prompt"):
            data["text_prompts"].append(
                {
                    "text": negative_prompt,
                    "weight": -1,
                }
            )

        if width := kwargs.get("width"):
            data["width"] = width
        if height := kwargs.get("height"):
            data["height"] = height
        if steps := kwargs.get("steps"):
            data["steps"] = steps
        if seed := kwargs.get("seed"):
            data["seed"] = seed
        if cfg_scale := kwargs.get("cfg_scale"):
            data["cfg_scale"] = cfg_scale
        if samples := kwargs.get("samples"):
            data["samples"] = min(samples, 10)

        logger.debug(f"Sending image edit request to Stability: engine={engine}")

        try:
            response = await client.post(
                f"/v1/generation/{engine}/image-to-image",
                data=data,
                files=files,
            )
        except Exception as e:
            logger.error(f"Stability image edit failed: {e}")
            raise

        self._handle_stability_error(response)
        result_data = response.json()

        # 提取图像
        artifacts = result_data.get("artifacts", [])
        if not artifacts:
            raise APIError(message="No image generated")

        images: list[ImageData] = []
        for artifact in artifacts:
            img_base64 = artifact.get("base64", "")
            if img_base64:
                images.append(ImageData(b64_json=img_base64))

        return ImageGenerationResult(
            id=generate_id("img"),
            created=current_timestamp(),
            model=engine,
            data=images,
        )

    # ==================== 视频生成（不支持）====================

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> VideoTask:
        """
        视频生成

        Stability 不支持视频生成，此方法抛出错误。
        """
        raise UnsupportedCapabilityError(
            message="Stability AI does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        """
        视频任务状态

        Stability 不支持视频生成，此方法抛出错误。
        """
        raise UnsupportedCapabilityError(
            message="Stability AI does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    # ==================== 模型信息 ====================

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """
        获取可用模型列表

        调用 GET /v1/engines/list 实时拉取，不再使用硬编码示例。

        Args:
            model_type: 模型类型过滤

        Returns:
            模型信息列表
        """
        client = self._get_client()

        logger.debug("Fetching engines list from Stability AI")

        response = await client.get("/v1/engines/list")
        self._handle_stability_error(response)

        # Stability 返回顶层数组，包装成 {"data": [...]} 交给基类统一解析
        engines = response.json()
        if isinstance(engines, list):
            data: dict[str, Any] = {"data": engines}
        else:
            data = engines

        return self._parse_models_response(
            data=data,
            provider="stability",
            model_type=model_type,
        )

    # ==================== 辅助方法 ====================

    def _handle_stability_error(self, response: httpx.Response) -> None:
        """处理 Stability AI 错误响应"""
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Stability API key")
        if response.status_code == 429:
            raise RateLimitError(message="Stability rate limit exceeded")
        if response.status_code == 403:
            raise APIError(
                message="Engine not accessible or quota exceeded", status_code=403
            )

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
AdapterFactory.register("stability", StabilityAdapter)
