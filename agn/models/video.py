"""
AGN-SDK 视频生成数据模型

定义视频生成相关的 Pydantic 模型。
"""

from typing import Literal

from pydantic import BaseModel, Field


class VideoTask(BaseModel):
    """
    视频任务信息

    表示创建视频生成任务后的返回信息。
    """

    task_id: str = Field(..., description="任务 ID（用于轮询状态）")
    model: str = Field(..., description="使用的模型")
    status: Literal["pending", "processing", "success", "failed"] = Field(
        "pending",
        description="任务状态",
    )
    created_at: int = Field(..., description="创建时间戳")

    model_config = {"extra": "allow"}


class VideoStatus(BaseModel):
    """
    视频任务状态

    表示视频生成任务的当前状态。
    """

    task_id: str = Field(..., description="任务 ID")
    status: Literal["pending", "processing", "success", "failed"] = Field(
        ...,
        description="任务状态",
    )
    video_url: str | None = Field(None, description="视频 URL（成功时）")
    progress: int | None = Field(None, ge=0, le=100, description="进度 0-100")
    error: str | None = Field(None, description="错误信息（失败时）")
    created_at: int | None = Field(None, description="创建时间戳")
    updated_at: int | None = Field(None, description="更新时间戳")

    model_config = {"extra": "allow"}


class VideoGenerationOptions(BaseModel):
    """
    视频生成选项

    通用的视频生成配置选项。
    """

    width: int = Field(1280, ge=128, le=3840, description="视频宽度（必须是 8 的倍数）")
    height: int = Field(720, ge=128, le=2160, description="视频高度（必须是 8 的倍数）")
    num_frames: int | None = Field(None, description="帧数（部分模型需要）")
    frame_rate: int = Field(24, ge=1, le=120, description="帧率")
    mode: Literal["text2video", "image2video", "keyframes", "multiimage"] = Field(
        "text2video",
        description="生成模式",
    )
    seed: int | None = Field(None, description="随机种子（可选）")
    negative_prompt: str | None = Field(None, description="负面提示词")
    # 以下为部分模型（如火山引擎 Seedance、Agnes 等）使用的参数
    duration: int | None = Field(
        None, description="视频时长（秒），部分模型用此字段替代 num_frames"
    )
    aspect_ratio: str | None = Field(
        None, description="宽高比，如 '16:9' / '9:16' / '1:1'"
    )
    resolution: str | None = Field(None, description="分辨率档位，如 '720p' / '1080p'")

    model_config = {"extra": "allow"}


class VideoTaskCreate(BaseModel):
    """
    视频任务创建请求

    用于创建视频生成任务的请求参数。
    """

    model: str = Field(..., description="模型名称")
    prompt: str = Field(..., description="提示词")
    width: int = Field(1280, ge=128, le=3840, description="视频宽度")
    height: int = Field(720, ge=128, le=2160, description="视频高度")
    num_frames: int | None = Field(None, description="帧数")
    frame_rate: int = Field(24, ge=1, le=120, description="帧率")
    mode: Literal["text2video", "image2video", "keyframes", "multiimage"] = Field(
        "text2video",
        description="生成模式",
    )
    reference_images: list[str] | None = Field(None, description="参考图像列表")
    negative_prompt: str | None = Field(None, description="负面提示词")
    seed: int | None = Field(None, description="随机种子")

    model_config = {"extra": "allow"}
