"""
AGN-SDK Runway 适配器测试

测试 RunwayAdapter 的各项功能，包括视频创建、视频轮询、错误处理等。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agn.adapters.runway import RunwayAdapter
from agn.core.errors import UnsupportedCapabilityError
from agn.models.common import ProviderConfig
from agn.models.video import VideoStatus, VideoTask


class TestRunwayAdapter:
    """RunwayAdapter 类测试"""

    @pytest.fixture
    def adapter_config(self, mock_api_key: str) -> ProviderConfig:
        """创建适配器配置"""
        return ProviderConfig(
            provider_type="runway",
            api_key=mock_api_key,
        )

    @pytest.fixture
    def adapter(self, adapter_config: ProviderConfig) -> RunwayAdapter:
        """创建适配器实例"""
        return RunwayAdapter(config=adapter_config)

    def test_adapter_init(self, adapter: RunwayAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "runway"
        assert adapter.provider_name == "Runway"
        assert "video" in adapter.supported_capabilities
        assert "chat" not in adapter.supported_capabilities
        assert "image" not in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: RunwayAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("video")
        assert not adapter.supports_capability("chat")
        assert not adapter.supports_capability("image")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: RunwayAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None

    @pytest.mark.asyncio
    async def test_chat_not_supported(self, adapter: RunwayAdapter) -> None:
        """测试文本对话不支持"""

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.chat(
                model="test",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_image_generate_not_supported(self, adapter: RunwayAdapter) -> None:
        """测试图像生成不支持"""

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.image_generate(
                model="test",
                prompt="A cat",
            )


class TestRunwayAdapterListModels:
    """RunwayAdapter 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> RunwayAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="runway", api_key=mock_api_key)
        return RunwayAdapter(config=config)

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: RunwayAdapter) -> None:
        """测试获取所有模型"""
        models = await adapter.list_models()
        assert len(models) > 0

        types = {m.type for m in models}
        assert "video" in types
        assert "chat" not in types
        assert "image" not in types

    @pytest.mark.asyncio
    async def test_list_video_models(self, adapter: RunwayAdapter) -> None:
        """测试获取视频模型"""
        models = await adapter.list_models(model_type="video")
        assert len(models) > 0
        for model in models:
            assert model.type == "video"

    @pytest.mark.asyncio
    async def test_list_chat_models_empty(self, adapter: RunwayAdapter) -> None:
        """测试获取对话模型（空列表）"""
        models = await adapter.list_models(model_type="chat")
        assert len(models) == 0


class TestRunwayAdapterStatusMapping:
    """RunwayAdapter 状态映射测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> RunwayAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="runway", api_key=mock_api_key)
        return RunwayAdapter(config=config)

    def test_map_pending_status(self, adapter: RunwayAdapter) -> None:
        """测试 pending 状态映射"""
        assert adapter._map_runway_status("pending") == "pending"
        assert adapter._map_runway_status("queued") == "pending"

    def test_map_processing_status(self, adapter: RunwayAdapter) -> None:
        """测试 processing 状态映射"""
        assert adapter._map_runway_status("processing") == "processing"
        assert adapter._map_runway_status("running") == "processing"
        assert adapter._map_runway_status("in_progress") == "processing"

    def test_map_success_status(self, adapter: RunwayAdapter) -> None:
        """测试 success 状态映射"""
        assert adapter._map_runway_status("completed") == "success"
        assert adapter._map_runway_status("succeeded") == "success"
        assert adapter._map_runway_status("success") == "success"

    def test_map_failed_status(self, adapter: RunwayAdapter) -> None:
        """测试 failed 状态映射"""
        assert adapter._map_runway_status("failed") == "failed"
        assert adapter._map_runway_status("error") == "failed"
        assert adapter._map_runway_status("cancelled") == "failed"

    def test_map_unknown_status(self, adapter: RunwayAdapter) -> None:
        """测试未知状态映射"""
        assert adapter._map_runway_status("unknown_status") == "pending"


class TestRunwayAdapterVideoMockHTTP:
    """RunwayAdapter 视频生成 Mock HTTP 测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> RunwayAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="runway", api_key=mock_api_key)
        return RunwayAdapter(config=config)

    def _mock_response(self, data: dict, status_code: int = 200) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=data)
        mock_resp.headers = {}
        return mock_resp

    @pytest.mark.asyncio
    async def test_video_create_text2video(self, adapter: RunwayAdapter) -> None:
        """测试文本生视频请求"""
        await adapter.start()

        mock_result = {
            "id": "vid-abc123",
            "status": "pending",
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.video_create(
                model="gen-3",
                prompt="A cat running in the park",
            )

            assert isinstance(result, VideoTask)
            assert result.task_id == "vid-abc123"
            assert result.model == "gen-3"

            # 验证请求 URL 和 body
            call_args = mock_post.call_args
            assert "/text_to_video" in str(call_args)
            body = call_args.kwargs.get("json") or call_args.args[0]
            assert body["model"] == "gen-3"
            assert body["prompt"] == "A cat running in the park"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_create_image2video(self, adapter: RunwayAdapter) -> None:
        """测试图生视频请求"""
        await adapter.start()

        mock_result = {
            "id": "vid-img2vid-001",
            "status": "pending",
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.video_create(
                model="gen-3-turbo",
                prompt="Make this image move",
                mode="image2video",
                reference_images=["https://example.com/input.jpg"],
            )

            assert isinstance(result, VideoTask)

            # 验证请求 URL（应该是 image_to_video 端点）
            call_args = mock_post.call_args
            assert "/image_to_video" in str(call_args)
            body = call_args.kwargs.get("json") or call_args.args[0]
            assert body["promptImage"] == "https://example.com/input.jpg"
            assert body["promptText"] == "Make this image move"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_create_with_params(self, adapter: RunwayAdapter) -> None:
        """测试带参数的短视频创建"""
        await adapter.start()

        mock_result = {
            "id": "vid-xyz789",
            "status": "pending",
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.video_create(
                model="gen-3",
                prompt="A beautiful sunset",
                width=1920,
                height=1080,
                seed=42,
                motion="strong",
                camera_motion="zoom in",
                aspect_ratio="16:9",
            )

            assert isinstance(result, VideoTask)

            # 验证请求参数
            call_args = mock_post.call_args
            body = call_args.kwargs.get("json") or call_args.args[0]
            assert body["width"] == 1920
            assert body["height"] == 1080
            assert body["seed"] == 42
            assert body["motion"] == "strong"
            assert body["cameraMotion"] == "zoom in"
            assert body["aspectRatio"] == "16:9"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_poll_pending(self, adapter: RunwayAdapter) -> None:
        """测试查询视频状态（pending）"""
        await adapter.start()

        mock_result = {
            "id": "vid-abc123",
            "status": "pending",
            "createdAt": 1700000000,
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            result = await adapter.video_poll(task_id="vid-abc123")

            assert isinstance(result, VideoStatus)
            assert result.task_id == "vid-abc123"
            assert result.status == "pending"

            # 验证请求 URL
            call_args = mock_get.call_args
            assert "/assets/vid-abc123" in str(call_args)

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_poll_processing(self, adapter: RunwayAdapter) -> None:
        """测试查询视频状态（processing）"""
        await adapter.start()

        mock_result = {
            "id": "vid-abc123",
            "status": "processing",
            "progress": 50,
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            result = await adapter.video_poll(task_id="vid-abc123")

            assert result.status == "processing"
            assert result.progress == 50

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_poll_success(self, adapter: RunwayAdapter) -> None:
        """测试查询视频状态（success）"""
        await adapter.start()

        mock_result = {
            "id": "vid-abc123",
            "status": "completed",
            "url": "https://cdn.example.com/video.mp4",
            "progress": 100,
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            result = await adapter.video_poll(task_id="vid-abc123")

            assert result.status == "success"
            assert result.progress == 100
            assert result.video_url == "https://cdn.example.com/video.mp4"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_poll_failed(self, adapter: RunwayAdapter) -> None:
        """测试查询视频状态（failed）"""
        await adapter.start()

        mock_result = {
            "id": "vid-abc123",
            "status": "failed",
            "error": "Insufficient credits",
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            result = await adapter.video_poll(task_id="vid-abc123")

            assert result.status == "failed"
            assert result.error == "Insufficient credits"

        await adapter.close()
