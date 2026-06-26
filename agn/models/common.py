"""
AGN-SDK 通用数据模型

定义 Provider 配置、模型信息等通用数据结构。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """
    Provider 配置

    用于配置单个 AI 模型提供商的连接参数。
    """

    provider_type: str = Field(
        ..., description="Provider 类型标识，如 'agnes'、'openai'"
    )
    # API Key 改为可选：部分 Provider（如 Edge TTS）免费且无需认证
    api_key: str | None = Field(None, description="API Key（免费 Provider 可不传）")
    base_url: str | None = Field(
        None, description="API Base URL（可选，部分 Provider 有默认值）"
    )
    poll_url: str | None = Field(None, description="轮询 URL（视频生成任务状态用）")
    timeout: int = Field(300, ge=1, le=600, description="请求超时时间（秒）")
    max_retries: int = Field(3, ge=0, le=10, description="最大重试次数")
    retry_delay: float = Field(2.0, ge=0.1, le=60, description="重试延迟（秒）")
    enabled: bool = Field(True, description="是否启用该 Provider")

    # Azure 专用字段
    resource_name: str | None = Field(None, description="Azure 资源名称")
    deployment_id: str | None = Field(None, description="Azure 部署 ID")
    api_version: str | None = Field(None, description="API 版本")

    # 额外配置（用于厂商特定配置）
    extra: dict[str, Any] = Field(
        default_factory=dict, description="额外配置项（厂商特定配置）"
    )

    model_config = {"extra": "allow"}


class ModelInfo(BaseModel):
    """
    模型信息

    描述单个 AI 模型的基本信息和能力。
    """

    id: str = Field(..., description="模型标识符")
    name: str = Field(..., description="模型显示名称")
    type: Literal["chat", "image", "video", "audio"] = Field(
        ..., description="模型类型"
    )
    provider: str = Field(..., description="提供商名称")
    capabilities: list[str] = Field(
        default_factory=list,
        description="支持的能力列表，如 ['text2image', 'image2image']",
    )
    max_tokens: int | None = Field(None, description="最大 token 数（仅 chat 模型）")
    supports_streaming: bool = Field(False, description="是否支持流式输出")
    description: str | None = Field(None, description="模型描述")
    created: int | None = Field(None, description="模型创建时间戳")

    model_config = {"extra": "allow"}


class ProviderInfo(BaseModel):
    """
    Provider 信息

    描述单个 AI 模型提供商的元信息。
    """

    type: str = Field(..., description="Provider 类型标识")
    name: str = Field(..., description="Provider 显示名称")
    description: str | None = Field(None, description="Provider 描述")
    website: str | None = Field(None, description="Provider 官网")
    supported_capabilities: list[str] = Field(
        default_factory=list,
        description="支持的能力列表",
    )
    supported_model_types: list[Literal["chat", "image", "video", "audio"]] = Field(
        default_factory=list,
        description="支持的模型类型",
    )

    model_config = {"extra": "allow"}


# 预定义的模型类型常量
class ModelType:
    """模型类型常量"""

    CHAT = "chat"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


# 预定义的视频生成模式常量
class VideoMode:
    """视频生成模式常量"""

    TEXT2VIDEO = "text2video"
    IMAGE2VIDEO = "image2video"
    KEYFRAMES = "keyframes"
    MULTIIMAGE = "multiimage"


# 预定义的图像尺寸常量
class ImageSize:
    """图像尺寸常量"""

    SIZE_256X256 = "256x256"
    SIZE_512X512 = "512x512"
    SIZE_1024X1024 = "1024x1024"
    SIZE_1792X1024 = "1792x1024"  # 16:9
    SIZE_1024X1792 = "1024x1792"  # 9:16

    # 标准尺寸映射
    STANDARD_SIZES = [
        SIZE_256X256,
        SIZE_512X512,
        SIZE_1024X1024,
        SIZE_1792X1024,
        SIZE_1024X1792,
    ]


# 预定义的状态常量
class TaskStatus:
    """任务状态常量"""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
