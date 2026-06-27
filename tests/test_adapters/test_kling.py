"""
AGN-SDK Kling (可灵) 适配器测试

测试 KlingAdapter 的各项功能，包括视频创建、视频轮询、错误处理等。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agn.adapters.kling import KlingAdapter
from agn.core.errors import UnsupportedCapabilityError
from agn.models.common import ProviderConfig
from agn.models.video import VideoStatus, VideoTask


class TestKlingAdapter:
    """KlingAdapter 类基础测试"""

    @pytest.fixture
    def adapter_config(self, mock_api_key: str) -> ProviderConfig:
        """创建适配器配置"""
        return ProviderConfig(
            provider_type="kling",
            api_key=mock_api_key,
        )

    @pytest.fixture
    def adapter(self, adapter_config: ProviderConfig) -> KlingAdapter:
        """创建适配器实例"""
        return KlingAdapter(config=adapter_config)

    def test_adapter_init(self, adapter: KlingAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "kling"
        assert adapter.provider_name == "可灵 Kling"
        assert "video" in adapter.supported_capabilities
        assert "chat" not in adapter.supported_capabilities
        assert "image" not in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: KlingAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("video")
        assert not adapter.supports_capability("chat")
        assert not adapter.supports_capability("image")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: KlingAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None

    @pytest.mark.asyncio
    async def test_chat_not_supported(self, adapter: KlingAdapter) -> None:
        """测试文本对话不支持"""
        with pytest.raises(UnsupportedCapabilityError):
            await adapter.chat(
                model="test",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_image_not_supported(self, adapter: KlingAdapter) -> None:
        """测试图像生成不支持"""
        with pytest.raises(UnsupportedCapabilityError):
            await adapter.image_generate(
                model="test",
                prompt="A cat",
            )


class TestKlingAdapterListModels:
    """KlingAdapter 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> KlingAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="kling", api_key=mock_api_key)
        return KlingAdapter(config=config)

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: KlingAdapter) -> None:
        """测试获取所有模型"""
        models = await adapter.list_models()
        assert len(models) > 0

        types = {m.type for m in models}
        assert "video" in types

    @pytest.mark.asyncio
    async def test_list_video_models(self, adapter: KlingAdapter) -> None:
        """测试获取视频模型"""
        models = await adapter.list_models(model_type="video")
        assert len(models) > 0
        for model in models:
            assert model.type == "video"


class TestKlingAdapterStatusMapping:
    """KlingAdapter 状态映射测试"""

    def test_map_pending_status(self) -> None:
        """测试 pending 状态映射"""
        adapter = KlingAdapter(
            config=ProviderConfig(provider_type="kling", api_key="test")
        )
        assert adapter._map_kling_status("submitted") == "pending"
        assert adapter._map_kling_status("queued") == "pending"

    def test_map_processing_status(self) -> None:
        """测试 processing 状态映射"""
        adapter = KlingAdapter(
            config=ProviderConfig(provider_type="kling", api_key="test")
        )
        assert adapter._map_kling_status("processing") == "processing"

    def test_map_success_status(self) -> None:
        """测试 success 状态映射"""
        adapter = KlingAdapter(
            config=ProviderConfig(provider_type="kling", api_key="test")
        )
        assert adapter._map_kling_status("succeed") == "success"
        assert adapter._map_kling_status("success") == "success"

    def test_map_failed_status(self) -> None:
        """测试 failed 状态映射"""
        adapter = KlingAdapter(
            config=ProviderConfig(provider_type="kling", api_key="test")
        )
        assert adapter._map_kling_status("failed") == "failed"
        assert adapter._map_kling_status("error") == "failed"


class TestKlingAdapterVideoMockHTTP:
    """KlingAdapter 视频生成 Mock HTTP 测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> KlingAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="kling", api_key=mock_api_key)
        return KlingAdapter(config=config)

    def _mock_response(self, data: dict, status_code: int = 200) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=data)
        mock_resp.headers = {}
        return mock_resp

    @pytest.mark.asyncio
    async def test_video_create_basic(self, adapter: KlingAdapter) -> None:
        """测试基础视频创建请求和响应解析"""
        await adapter.start()

        mock_result = {
            "code": 0,
            "message": "success",
            "data": {
                "task_id": "vid-abc123",
                "task_status": "submitted",
            },
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.video_create(
                model="kling-v1",
                prompt="A cat running in the park",
            )

            assert isinstance(result, VideoTask)
            assert result.task_id == "vid-abc123"
            assert result.model == "kling-v1"
            # submitted 会被映射为 pending
            assert result.status == "pending"

            # 验证请求 URL 和 body
            call_args = mock_post.call_args
            assert "/videos/generations" in str(call_args)
            body = call_args.kwargs.get("json") or call_args.args[0]
            assert body["model_name"] == "kling-v1"
            assert body["prompt"] == "A cat running in the park"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_create_with_params(self, adapter: KlingAdapter) -> None:
        """测试带参数的短视频创建"""
        await adapter.start()

        mock_result = {
            "code": 0,
            "data": {
                "task_id": "vid-xyz789",
                "task_status": "submitted",
            },
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.video_create(
                model="kling-v2",
                prompt="A beautiful sunset over the ocean",
                negative_prompt="blurry, low quality",
                duration=10,
                aspect_ratio="16:9",
                cfg_scale=0.5,
            )

            assert isinstance(result, VideoTask)
            # submitted 会被映射为 pending
            assert result.status == "pending"

            # 验证请求参数
            call_args = mock_post.call_args
            body = call_args.kwargs.get("json") or call_args.args[0]
            assert body["negative_prompt"] == "blurry, low quality"
            assert body["duration"] == 10
            assert body["aspect_ratio"] == "16:9"
            assert body["cfg_scale"] == 0.5

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_create_image2video(self, adapter: KlingAdapter) -> None:
        """测试图生视频模式"""
        await adapter.start()

        mock_result = {
            "code": 0,
            "data": {
                "task_id": "vid-img2vid-001",
                "task_status": "submitted",
            },
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.video_create(
                model="kling-v1-5",
                prompt="Make this image move",
                mode="image2video",
                reference_images=["https://example.com/input.jpg"],
            )

            assert isinstance(result, VideoTask)
            # submitted 会被映射为 pending
            assert result.status == "pending"

            # 验证请求 URL（应该是 image2video 端点）
            call_args = mock_post.call_args
            assert "/videos/image2video" in str(call_args)
            body = call_args.kwargs.get("json") or call_args.args[0]
            assert body["image"] == "https://example.com/input.jpg"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_poll_pending(self, adapter: KlingAdapter) -> None:
        """测试查询视频状态（pending）"""
        await adapter.start()

        mock_result = {
            "code": 0,
            "data": {
                "task_id": "vid-abc123",
                "task_status": "submitted",
                "created_at": 1700000000,
                "updated_at": 1700000100,
            },
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            result = await adapter.video_poll(task_id="vid-abc123")

            assert isinstance(result, VideoStatus)
            assert result.task_id == "vid-abc123"
            assert result.status == "pending"
            assert result.progress == 0
            assert result.video_url is None

        await adapter.close()

    @pytest.mark.asyncio
    async def test_video_poll_processing(self, adapter: KlingAdapter) -> None:
        """测试查询视频状态（processing）"""
        await adapter.start()

        mock_result = {
            "code": 0,
            "data": {
                "task_id": "vid-abc123",
                "task_status": "processing",
                "created_at": 1700000000,
                "updated_at": 1700000200,
            },
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
    async def test_video_poll_success(self, adapter: KlingAdapter) -> None:
        """测试查询视频状态（success）"""
        await adapter.start()

        mock_result = {
            "code": 0,
            "data": {
                "task_id": "vid-abc123",
                "task_status": "succeed",
                "task_result": {
                    "videos": [{"url": "https://cdn.example.com/video.mp4"}]
                },
                "created_at": 1700000000,
                "updated_at": 1700000300,
            },
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
    async def test_video_poll_failed(self, adapter: KlingAdapter) -> None:
        """测试查询视频状态（failed）"""
        await adapter.start()

        mock_result = {
            "code": 0,
            "data": {
                "task_id": "vid-abc123",
                "task_status": "failed",
                "task_status_msg": "Insufficient credits",
                "created_at": 1700000000,
                "updated_at": 1700000300,
            },
        }

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_result)

            result = await adapter.video_poll(task_id="vid-abc123")

            assert result.status == "failed"
            assert result.error == "Insufficient credits"

        await adapter.close()
