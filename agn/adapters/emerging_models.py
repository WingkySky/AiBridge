"""
AGN-SDK 主流模型适配器（新增）

支持：
- Ideogram: 文字渲染最强的图像生成平台
- Luma Dream Machine: 高质量视频生成
- Meta Llama: Meta 官方 Llama API（OpenAI 兼容）

官方文档：
- Ideogram: https://developers.ideogram.com/
- Luma: https://docs.lumalabs.ai/
- Meta Llama: https://docs.llama.com/
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
from agn.models.chat import (
    ChatChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChatMessage,
)
from agn.models.common import ModelInfo, ProviderConfig
from agn.models.image import ImageData, ImageGenerationResult
from agn.models.video import VideoStatus, VideoTask

logger = logging.getLogger(__name__)


# ==================== Ideogram 图像生成适配器 ====================


class IdeogramAdapter(BaseAdapter):
    """
    Ideogram 适配器

    文字渲染最强的图像生成平台，支持 V2/V2A/V1 等模型。
    官方 API 文档：https://developers.ideogram.com/

    API 规范：
    - Base URL: https://api.ideogram.ai
    - 文生图: POST /generate (image_request body)
    - 图生图/Remix: POST /remix
    - 局部重绘: POST /inpaint
    - 扩图: POST /outpaint
    - 认证: Api-Key header
    """

    provider_type = "ideogram"
    provider_name = "Ideogram"
    supported_capabilities = ["image"]

    DEFAULT_BASE_URL = "https://api.ideogram.ai"

    def __init__(self, config: ProviderConfig) -> None:
        """初始化 Ideogram 适配器"""
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
                "Api-Key": self.api_key,
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
            message="Ideogram does not support chat",
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
            model: 模型名称（V_2, V_2A, V_2A_TURBO, V_1, V_1_TURBO）
            prompt: 提示词
            **kwargs: 其他参数
                - negative_prompt: 负面提示词
                - aspect_ratio: 宽高比（"16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"）
                - resolution: 分辨率（"1536x1536", "1024x1024", "1536x1024", "1024x1536"）
                - style_type: 风格类型（"AUTO", "GENERAL", "REALISTIC", "DESIGN", "RENDER_3D", "ANIME"）
                - magic_prompt_level: 魔法提示词增强（"AUTO", "LOW", "MEDIUM", "HIGH", "OFF"）
                - num_images: 生成数量（1-8）
                - seed: 随机种子
                - color_palette: 颜色调色板（hex 颜色列表）
                - weight: 提示词权重（0-1）

        Returns:
            图像生成结果
        """
        client = self._get_client()

        model_id = model or "V_2A_TURBO"

        # 构建请求体
        body: dict[str, Any] = {
            "prompt": prompt,
            "model": model_id,
        }

        # 负面提示词
        if negative_prompt := kwargs.get("negative_prompt"):
            body["negative_prompt"] = negative_prompt

        # 宽高比
        if aspect_ratio := kwargs.get("aspect_ratio"):
            body["aspect_ratio"] = aspect_ratio.upper()

        # 分辨率
        if resolution := kwargs.get("resolution"):
            body["resolution"] = resolution

        # 风格类型
        if style_type := kwargs.get("style_type"):
            body["style_type"] = style_type

        # 魔法提示词增强
        if magic_prompt_level := kwargs.get("magic_prompt_level"):
            body["magic_prompt_option"] = magic_prompt_level

        # 生成数量
        if num_images := kwargs.get("num_images"):
            body["num_images"] = min(int(num_images), 8)

        # 随机种子
        if seed := kwargs.get("seed"):
            body["seed"] = int(seed)

        # 透传额外参数
        extra_body = kwargs.get("extra_body")
        if extra_body and isinstance(extra_body, dict):
            body.update(extra_body)

        # Ideogram 使用 image_request 包裹
        request_body = {"image_request": body}

        logger.debug(f"Sending image request to Ideogram: model={model_id}")

        try:
            response = await client.post("/generate", json=request_body)
        except Exception as e:
            logger.error(f"Ideogram image generate failed: {e}")
            raise

        self._handle_ideogram_error(response)
        data = response.json()

        # 提取图像
        image_data_list = []
        for item in data.get("data", []):
            img = ImageData(
                url=item.get("url"),
                b64_json=item.get("base64"),
                revised_prompt=item.get("prompt"),
            )
            image_data_list.append(img)

        return ImageGenerationResult(
            id=data.get("request_id", generate_id("img")),
            created=current_timestamp(),
            model=model_id,
            data=image_data_list,
        )

    async def image_edit(
        self,
        model: str,
        prompt: str,
        image: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        图像编辑（Remix 图生图）

        Args:
            model: 模型名称
            prompt: 提示词
            image: 源图像 URL 或 Base64
            **kwargs: 其他参数
                - mask: 蒙版（inpaint 模式）
                - mode: "remix"（默认）或 "inpaint" 或 "outpaint"
                - aspect_ratio: 宽高比
                - negative_prompt: 负面提示词
                - num_images: 生成数量
                - seed: 随机种子

        Returns:
            图像生成结果
        """
        client = self._get_client()

        model_id = model or "V_2A_TURBO"
        mode = kwargs.get("mode", "remix")

        # 处理图像数据
        if image.startswith("data:"):
            img_data = image.split(",", 1)[1]
        elif image.startswith("http"):
            img_response = await client.get(image)
            img_data = base64.b64encode(img_response.content).decode()
        else:
            img_data = image

        # 构建请求体
        image_request: dict[str, Any] = {
            "prompt": prompt,
            "model": model_id,
            "image": img_data,
        }

        if negative_prompt := kwargs.get("negative_prompt"):
            image_request["negative_prompt"] = negative_prompt
        if aspect_ratio := kwargs.get("aspect_ratio"):
            image_request["aspect_ratio"] = aspect_ratio.upper()
        if num_images := kwargs.get("num_images"):
            image_request["num_images"] = min(int(num_images), 8)
        if seed := kwargs.get("seed"):
            image_request["seed"] = int(seed)

        # 蒙版（inpaint 模式）
        mask = kwargs.get("mask")
        if mask and mode == "inpaint":
            if mask.startswith("data:"):
                mask_data = mask.split(",", 1)[1]
            elif mask.startswith("http"):
                mask_response = await client.get(mask)
                mask_data = base64.b64encode(mask_response.content).decode()
            else:
                mask_data = mask
            image_request["mask"] = mask_data

        # 选择端点
        if mode == "inpaint":
            endpoint = "/inpaint"
        elif mode == "outpaint":
            endpoint = "/outpaint"
            if mask is None:
                image_request["mask"] = ""
        else:
            endpoint = "/remix"

        request_body = {"image_request": image_request}

        logger.debug(f"Sending image edit to Ideogram: mode={mode}, model={model_id}")

        try:
            response = await client.post(endpoint, json=request_body)
        except Exception as e:
            logger.error(f"Ideogram image edit failed: {e}")
            raise

        self._handle_ideogram_error(response)
        data = response.json()

        image_data_list = []
        for item in data.get("data", []):
            img = ImageData(
                url=item.get("url"),
                b64_json=item.get("base64"),
                revised_prompt=item.get("prompt"),
            )
            image_data_list.append(img)

        return ImageGenerationResult(
            id=generate_id("img"),
            created=current_timestamp(),
            model=model_id,
            data=image_data_list,
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
            message="Ideogram does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        """视频任务状态（不支持）"""
        raise UnsupportedCapabilityError(
            message="Ideogram does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    # ==================== 模型信息 ====================

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """获取可用模型列表"""
        models = [
            ModelInfo(
                id="V_2A",
                name="Ideogram V2A",
                type="image",
                provider="ideogram",
                capabilities=["text2image", "image2image"],
                description="Ideogram V2A 文生图模型，文字渲染强",
            ),
            ModelInfo(
                id="V_2A_TURBO",
                name="Ideogram V2A Turbo",
                type="image",
                provider="ideogram",
                capabilities=["text2image", "image2image"],
                description="Ideogram V2A Turbo 快速版本，文字渲染强",
            ),
            ModelInfo(
                id="V_2",
                name="Ideogram V2",
                type="image",
                provider="ideogram",
                capabilities=["text2image", "image2image"],
                description="Ideogram V2 高质量模型",
            ),
            ModelInfo(
                id="V_1",
                name="Ideogram V1",
                type="image",
                provider="ideogram",
                capabilities=["text2image"],
                description="Ideogram V1 标准模型",
            ),
            ModelInfo(
                id="V_1_TURBO",
                name="Ideogram V1 Turbo",
                type="image",
                provider="ideogram",
                capabilities=["text2image"],
                description="Ideogram V1 Turbo 快速模型",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    # ==================== 辅助方法 ====================

    def _handle_ideogram_error(self, response: httpx.Response) -> None:
        """处理 Ideogram 错误响应"""
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Ideogram API key")
        if response.status_code == 429:
            raise RateLimitError(
                message="Ideogram rate limit exceeded or credits exhausted"
            )
        if response.status_code == 402:
            raise APIError(
                message="Ideogram payment required or credits exhausted",
                status_code=402,
            )

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("message")
                or error_data.get("error")
                or error_data.get("detail")
                or f"HTTP {response.status_code}"
            )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=error_msg, status_code=response.status_code)


# ==================== Luma Dream Machine 视频生成适配器 ====================


class LumaAdapter(BaseAdapter):
    """
    Luma Dream Machine 适配器

    高质量视频生成平台，支持文生视频和图生视频。
    官方 API 文档：https://docs.lumalabs.ai/

    API 规范：
    - Base URL: https://api.lumalabs.ai/dream-machine/v1
    - 创建生成: POST /generations
    - 查询状态: GET /generations/{id}
    - 认证: Authorization: Bearer {api_key}
    """

    provider_type = "luma"
    provider_name = "Luma Dream Machine"
    supported_capabilities = ["video"]

    DEFAULT_BASE_URL = "https://api.lumalabs.ai/dream-machine/v1"

    def __init__(self, config: ProviderConfig) -> None:
        """初始化 Luma 适配器"""
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
        """文本对话（不支持）"""
        raise UnsupportedCapabilityError(
            message="Luma does not support chat",
            details={"provider": self.provider_type, "capability": "chat"},
        )

    # ==================== 图像生成（不支持）====================

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """图像生成（不支持）"""
        raise UnsupportedCapabilityError(
            message="Luma does not support direct image generation",
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
            model: 模型名称（dream-machine, ray-2, ray-2-flash）
            prompt: 提示词
            **kwargs: 其他参数
                - aspect_ratio: 宽高比（"16:9", "9:16", "1:1", "4:3", "3:4", "21:9"）
                - duration: 视频时长（"5s", "9s"）
                - resolution: 分辨率（"720p", "1080p"）
                - loop: 是否循环（布尔值）
                - reference_images: 参考图像列表（第一张作为起始帧）
                - end_image: 结束帧图像（可选）
                - keyframes: 关键帧配置（更细粒度控制）
                - camera_motion: 相机运动（"crane_up", "zoom_in", "pan_left" 等）
                - negative_prompt: 负面提示词

        Returns:
            视频任务信息
        """
        client = self._get_client()

        model_id = model or "ray-2"

        # 构建请求体
        body: dict[str, Any] = {
            "prompt": prompt,
            "model": model_id,
        }

        # 宽高比
        if aspect_ratio := kwargs.get("aspect_ratio"):
            body["aspect_ratio"] = aspect_ratio

        # 时长
        if duration := kwargs.get("duration"):
            body["duration"] = duration

        # 分辨率
        if resolution := kwargs.get("resolution"):
            body["resolution"] = resolution

        # 循环
        if loop := kwargs.get("loop"):
            body["loop"] = bool(loop)

        # 负面提示词
        if negative_prompt := kwargs.get("negative_prompt"):
            body["negative_prompt"] = negative_prompt

        # 关键帧
        keyframes: dict[str, Any] = {}
        reference_images = kwargs.get("reference_images", [])
        if reference_images:
            keyframes["frame0"] = {
                "type": "image",
                "url": reference_images[0],
            }
        end_image = kwargs.get("end_image")
        if end_image:
            keyframes["frame1"] = {
                "type": "image",
                "url": end_image,
            }

        # 直接传 keyframes 参数覆盖
        custom_keyframes = kwargs.get("keyframes")
        if custom_keyframes and isinstance(custom_keyframes, dict):
            keyframes.update(custom_keyframes)

        if keyframes:
            body["keyframes"] = keyframes

        # 相机运动
        if camera_motion := kwargs.get("camera_motion"):
            body["camera_motion"] = camera_motion

        # 透传额外参数
        extra_body = kwargs.get("extra_body")
        if extra_body and isinstance(extra_body, dict):
            body.update(extra_body)

        logger.debug(f"Sending video request to Luma: model={model_id}")

        try:
            response = await client.post("/generations", json=body)
        except Exception as e:
            logger.error(f"Luma video create failed: {e}")
            raise

        self._handle_luma_error(response)
        data = response.json()

        task_id = data.get("id") or generate_id("vid")
        state = data.get("state", "queued")

        return VideoTask(
            task_id=task_id,
            model=model_id,
            status=self._map_luma_status(state),
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

        logger.debug(f"Polling Luma task: {task_id}")

        try:
            response = await client.get(f"/generations/{task_id}")
        except Exception as e:
            logger.error(f"Luma video poll failed: {e}")
            raise

        self._handle_luma_error(response)
        data = response.json()

        state = data.get("state", "")
        status = self._map_luma_status(state)

        # 提取视频 URL
        video_url = None
        if status == "success":
            assets = data.get("assets", {})
            video_url = assets.get("video") or assets.get("mp4")
            if not video_url:
                # 尝试从 video 字段直接获取
                video_url = data.get("video", "")

        # 错误信息
        error = None
        if status == "failed":
            failure_reason = data.get("failure_reason", "")
            error = failure_reason or data.get("error", "Generation failed")

        # 进度估算（先判断 state，因为 dreaming/processing 都映射到 processing 但进度不同）
        progress = 0
        state_lower = state.lower()
        if status == "success":
            progress = 100
        elif state_lower == "dreaming":
            progress = 30
        elif status == "processing":
            progress = 70
        elif state_lower in ("queued", "pending"):
            progress = 5

        # Luma 返回 ISO 时间字符串，转为时间戳
        created_ts = data.get("created_at")
        updated_ts = data.get("updated_at") or current_timestamp()
        if isinstance(created_ts, str):
            from datetime import datetime

            try:
                created_ts = int(
                    datetime.fromisoformat(
                        created_ts.replace("Z", "+00:00")
                    ).timestamp()
                )
            except Exception:
                created_ts = None

        return VideoStatus(
            task_id=task_id,
            status=status,
            video_url=video_url,
            progress=progress,
            error=error,
            created_at=created_ts,
            updated_at=updated_ts,
        )

    def _map_luma_status(self, raw_state: str) -> str:
        """
        映射 Luma 状态到标准状态

        Luma 状态：
        - queued → pending
        - dreaming → processing
        - completed → success
        - failed → failed
        """
        state_lower = raw_state.lower()
        status_map = {
            "queued": "pending",
            "pending": "pending",
            "dreaming": "processing",
            "processing": "processing",
            "completed": "success",
            "succeeded": "success",
            "success": "success",
            "failed": "failed",
            "error": "failed",
        }
        return status_map.get(state_lower, "pending")

    # ==================== 模型信息 ====================

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """获取可用模型列表"""
        models = [
            ModelInfo(
                id="ray-2",
                name="Luma Ray 2",
                type="video",
                provider="luma",
                capabilities=["text2video", "image2video"],
                description="Luma Ray 2 高质量视频生成模型",
            ),
            ModelInfo(
                id="ray-2-flash",
                name="Luma Ray 2 Flash",
                type="video",
                provider="luma",
                capabilities=["text2video", "image2video"],
                description="Luma Ray 2 Flash 快速视频生成模型",
            ),
            ModelInfo(
                id="dream-machine",
                name="Dream Machine",
                type="video",
                provider="luma",
                capabilities=["text2video", "image2video"],
                description="Luma Dream Machine 初代视频模型",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    # ==================== 辅助方法 ====================

    def _handle_luma_error(self, response: httpx.Response) -> None:
        """处理 Luma 错误响应"""
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Luma API key")
        if response.status_code == 429:
            raise RateLimitError(
                message="Luma rate limit exceeded or credits exhausted"
            )
        if response.status_code == 402:
            raise APIError(
                message="Luma credits exhausted or payment required", status_code=402
            )

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("detail")
                or error_data.get("error", {}).get("message")
                or error_data.get("message")
                or f"HTTP {response.status_code}"
            )
            if isinstance(error_msg, list):
                error_msg = "; ".join(str(e) for e in error_msg)
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=str(error_msg), status_code=response.status_code)


# ==================== Meta Llama 适配器（OpenAI 兼容）====================


class LlamaAdapter(BaseAdapter):
    """
    Meta Llama 适配器

    Meta 官方 Llama API，完全兼容 OpenAI 接口规范。
    官方 API 文档：https://docs.llama.com/

    API 规范：
    - Base URL: https://api.llama.com
    - Chat: POST /chat/completions (OpenAI 兼容)
    - 认证: Bearer Token
    - 支持流式输出
    """

    provider_type = "llama"
    provider_name = "Meta Llama"
    supported_capabilities = ["chat", "vision"]

    DEFAULT_BASE_URL = "https://api.llama.com/v1"

    def __init__(self, config: ProviderConfig) -> None:
        """初始化 Meta Llama 适配器"""
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

    def _build_request_body(
        self,
        model: str,
        messages: list[ChatMessage],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """构建 OpenAI 兼容的请求体"""
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
                for m in messages
            ],
        }

        # 通用参数
        if temperature := kwargs.get("temperature"):
            body["temperature"] = temperature
        if max_tokens := kwargs.get("max_tokens"):
            body["max_tokens"] = max_tokens
        if top_p := kwargs.get("top_p"):
            body["top_p"] = top_p
        if frequency_penalty := kwargs.get("frequency_penalty"):
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty := kwargs.get("presence_penalty"):
            body["presence_penalty"] = presence_penalty
        if stop := kwargs.get("stop"):
            body["stop"] = stop
        if seed := kwargs.get("seed"):
            body["seed"] = seed
        if response_format := kwargs.get("response_format"):
            body["response_format"] = response_format

        # 工具调用
        if tools := kwargs.get("tools"):
            body["tools"] = tools
        if tool_choice := kwargs.get("tool_choice"):
            body["tool_choice"] = tool_choice

        # 透传额外参数
        extra_body = kwargs.get("extra_body")
        if extra_body and isinstance(extra_body, dict):
            body.update(extra_body)

        return body

    # ==================== 文本对话 ====================

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatCompletion:
        """
        文本对话（非流式）

        Args:
            model: 模型名称（llama-4-maverick, llama-4-scout, llama-3.3-70b-instruct 等）
            messages: 消息列表
            **kwargs: 其他参数（temperature, max_tokens, top_p, tools 等）

        Returns:
            ChatCompletion 响应
        """
        client = self._get_client()
        body = self._build_request_body(model, messages, kwargs)

        try:
            response = await client.post("/chat/completions", json=body)
        except Exception as e:
            logger.error(f"Llama chat failed: {e}")
            raise

        self._handle_error(response)
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
            ChatCompletionChunk
        """
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
            logger.error(f"Llama stream chat failed: {e}")
            raise

    def _parse_response(self, data: dict[str, Any], model: str) -> ChatCompletion:
        """解析非流式响应"""
        from agn.models.chat import ChatUsage

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        # 解析工具调用
        tool_calls = message.get("tool_calls")

        # 解析 usage
        usage_data = data.get("usage", {})
        usage = None
        if usage_data:
            usage = ChatUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        # 构建 choices 列表
        chat_message = ChatMessage(
            role=message.get("role", "assistant"),
            content=message.get("content", "") or "",
            tool_calls=tool_calls,
        )
        choices = [
            ChatChoice(
                index=choice.get("index", 0),
                message=chat_message,
                finish_reason=choice.get("finish_reason"),
            )
        ]

        return ChatCompletion(
            id=data.get("id", generate_id("chat")),
            model=model,
            created=data.get("created", current_timestamp()),
            choices=choices,
            usage=usage,
        )

    def _parse_chunk(self, data_str: str, model: str) -> ChatCompletionChunk | None:
        """解析流式 chunk"""
        import json

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return None

        choices = data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})

        delta_message = ChatMessage(
            role=delta.get("role", "assistant"),
            content=delta.get("content", "") or "",
            tool_calls=delta.get("tool_calls"),
        )

        chunk_delta = ChatCompletionDelta(
            index=choice.get("index", 0),
            delta=delta_message,
            finish_reason=choice.get("finish_reason"),
        )

        return ChatCompletionChunk(
            id=data.get("id", generate_id("chat")),
            model=model,
            created=data.get("created", current_timestamp()),
            choices=[chunk_delta],
        )

    # ==================== 图像生成（不支持直接生成）====================

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """图像生成（不支持直接生成，多模态模型通过 chat 接口理解图片）"""
        raise UnsupportedCapabilityError(
            message="Meta Llama does not support direct image generation",
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
            message="Meta Llama does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> VideoStatus:
        """视频任务状态（不支持）"""
        raise UnsupportedCapabilityError(
            message="Meta Llama does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    # ==================== 模型信息 ====================

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """获取可用模型列表"""
        models = [
            ModelInfo(
                id="llama-4-maverick",
                name="Llama 4 Maverick",
                type="chat",
                provider="llama",
                capabilities=["chat", "vision"],
                description="Llama 4 Maverick 旗舰多模态模型",
            ),
            ModelInfo(
                id="llama-4-scout",
                name="Llama 4 Scout",
                type="chat",
                provider="llama",
                capabilities=["chat", "vision"],
                description="Llama 4 Scout 轻量多模态模型",
            ),
            ModelInfo(
                id="llama-3.3-70b-instruct",
                name="Llama 3.3 70B Instruct",
                type="chat",
                provider="llama",
                capabilities=["chat"],
                description="Llama 3.3 70B 指令微调版",
            ),
            ModelInfo(
                id="llama-3.1-405b-instruct",
                name="Llama 3.1 405B Instruct",
                type="chat",
                provider="llama",
                capabilities=["chat"],
                description="Llama 3.1 405B 超大参数模型",
            ),
            ModelInfo(
                id="llama-3.1-70b-instruct",
                name="Llama 3.1 70B Instruct",
                type="chat",
                provider="llama",
                capabilities=["chat"],
                description="Llama 3.1 70B 通用模型",
            ),
            ModelInfo(
                id="llama-3.1-8b-instruct",
                name="Llama 3.1 8B Instruct",
                type="chat",
                provider="llama",
                capabilities=["chat"],
                description="Llama 3.1 8B 轻量模型",
            ),
            ModelInfo(
                id="llama-guard-4",
                name="Llama Guard 4",
                type="chat",
                provider="llama",
                capabilities=["chat"],
                description="Llama Guard 4 安全审查模型",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    # ==================== 辅助方法 ====================

    def _handle_error(self, response: httpx.Response) -> None:
        """处理 Llama API 错误响应"""
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Meta Llama API key")
        if response.status_code == 429:
            raise RateLimitError(message="Meta Llama rate limit exceeded")

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
AdapterFactory.register("ideogram", IdeogramAdapter)
AdapterFactory.register("ideo", IdeogramAdapter)  # 别名
AdapterFactory.register("luma", LumaAdapter)
AdapterFactory.register("dream-machine", LumaAdapter)
AdapterFactory.register("lumalabs", LumaAdapter)
AdapterFactory.register("llama", LlamaAdapter)
AdapterFactory.register("meta-llama", LlamaAdapter)
AdapterFactory.register("meta", LlamaAdapter)
