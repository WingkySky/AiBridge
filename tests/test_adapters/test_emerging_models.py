"""
AGN-SDK 新兴主流模型适配器测试

覆盖：
- Ideogram: 文字渲染最强的图像生成平台（文生图/图生图/Inpaint/Outpaint）
- Luma Dream Machine: 高质量视频生成（文生视频/图生视频）
- Meta Llama: Meta 官方 Llama API（聊天/多模态/流式）
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agn.adapters.emerging_models import (
    IdeogramAdapter,
    LlamaAdapter,
    LumaAdapter,
)
from agn.models.common import ProviderConfig

# ==================== Ideogram 测试 ====================


class TestIdeogramAdapter:
    """IdeogramAdapter 基本能力测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> IdeogramAdapter:
        config = ProviderConfig(provider_type="ideogram", api_key=mock_api_key)
        return IdeogramAdapter(config=config)

    def test_adapter_init(self, adapter: IdeogramAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "ideogram"
        assert adapter.provider_name == "Ideogram"
        assert "image" in adapter.supported_capabilities
        assert "chat" not in adapter.supported_capabilities
        assert "video" not in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: IdeogramAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("image")
        assert not adapter.supports_capability("chat")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: IdeogramAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None
            assert "api.ideogram.ai" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_chat_not_supported(self, adapter: IdeogramAdapter) -> None:
        """测试文本对话不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.chat(
                model="test", messages=[{"role": "user", "content": "hi"}]
            )

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: IdeogramAdapter) -> None:
        """测试视频生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="cat")
        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_poll(task_id="test")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: IdeogramAdapter) -> None:
        """测试获取所有模型"""
        models = await adapter.list_models()
        assert len(models) >= 5
        model_ids = {m.id for m in models}
        assert "V_2A" in model_ids
        assert "V_2A_TURBO" in model_ids
        assert "V_2" in model_ids
        assert "V_1" in model_ids
        assert "V_1_TURBO" in model_ids
        for m in models:
            assert m.type == "image"

    @pytest.mark.asyncio
    async def test_list_image_models_filter(self, adapter: IdeogramAdapter) -> None:
        """测试按类型过滤模型"""
        image_models = await adapter.list_models(model_type="image")
        assert len(image_models) == 5
        chat_models = await adapter.list_models(model_type="chat")
        assert len(chat_models) == 0


class TestIdeogramImageGenerate:
    """Ideogram 图像生成 API 测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> IdeogramAdapter:
        config = ProviderConfig(provider_type="ideogram", api_key=mock_api_key)
        return IdeogramAdapter(config=config)

    @pytest.mark.asyncio
    async def test_image_generate_basic(self, adapter: IdeogramAdapter) -> None:
        """测试基本文生图"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"url": "https://ideogram.ai/api/images/test1.png", "prompt": "A cat"},
            ]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        result = await adapter.image_generate(model="V_2A", prompt="A cat")
        assert len(result.data) == 1
        assert result.data[0].url == "https://ideogram.ai/api/images/test1.png"
        assert result.model == "V_2A"
        assert result.object == "image.generation"

        # 验证请求构造
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/generate"
        body = call_args[1]["json"]
        assert "image_request" in body
        assert body["image_request"]["prompt"] == "A cat"
        assert body["image_request"]["model"] == "V_2A"

    @pytest.mark.asyncio
    async def test_image_generate_default_model(self, adapter: IdeogramAdapter) -> None:
        """测试默认模型使用 V_2A_TURBO"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"url": "https://example.com/img.png"}]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        result = await adapter.image_generate(model="", prompt="Hello")
        body = mock_client.post.call_args[1]["json"]
        assert body["image_request"]["model"] == "V_2A_TURBO"

    @pytest.mark.asyncio
    async def test_image_generate_with_params(self, adapter: IdeogramAdapter) -> None:
        """测试带参数的图像生成"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"url": "https://example.com/1.png"},
                {"url": "https://example.com/2.png"},
            ]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        await adapter.image_generate(
            model="V_2A",
            prompt="A logo",
            negative_prompt="blurry",
            aspect_ratio="16:9",
            resolution="1536x1024",
            style_type="DESIGN",
            magic_prompt_level="HIGH",
            num_images=2,
            seed=42,
        )

        body = mock_client.post.call_args[1]["json"]["image_request"]
        assert body["negative_prompt"] == "blurry"
        assert body["aspect_ratio"] == "16:9"
        assert body["resolution"] == "1536x1024"
        assert body["style_type"] == "DESIGN"
        assert body["magic_prompt_option"] == "HIGH"
        assert body["num_images"] == 2
        assert body["seed"] == 42

    @pytest.mark.asyncio
    async def test_image_generate_base64_response(
        self, adapter: IdeogramAdapter
    ) -> None:
        """测试返回 base64 格式图片"""
        import base64 as b64

        fake_b64 = b64.b64encode(b"fake_png_data").decode()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"base64": fake_b64}]}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        result = await adapter.image_generate(model="V_2A", prompt="test")
        assert result.data[0].b64_json == fake_b64

    @pytest.mark.asyncio
    async def test_image_edit_remix(self, adapter: IdeogramAdapter) -> None:
        """测试 Remix 图生图"""
        import base64 as b64

        fake_img_b64 = b64.b64encode(b"fake_img").decode()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"url": "https://example.com/remixed.png"}]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        result = await adapter.image_edit(
            model="V_2A",
            prompt="Make it red",
            image=f"data:image/png;base64,{fake_img_b64}",
        )
        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/remixed.png"
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/remix"
        body = call_args[1]["json"]["image_request"]
        assert body["prompt"] == "Make it red"
        assert body["image"] == fake_img_b64

    @pytest.mark.asyncio
    async def test_authentication_error(self, adapter: IdeogramAdapter) -> None:
        """测试认证错误"""
        from agn.core.errors import AuthenticationError

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"message": "Unauthorized"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        with pytest.raises(AuthenticationError):
            await adapter.image_generate(model="V_2A", prompt="test")

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, adapter: IdeogramAdapter) -> None:
        """测试限流错误"""
        from agn.core.errors import RateLimitError

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"message": "Rate limit exceeded"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        with pytest.raises(RateLimitError):
            await adapter.image_generate(model="V_2A", prompt="test")

    @pytest.mark.asyncio
    async def test_credits_exhausted_error(self, adapter: IdeogramAdapter) -> None:
        """测试积分耗尽错误"""
        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_response.json.return_value = {"detail": "Payment required"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        with pytest.raises(Exception):
            await adapter.image_generate(model="V_2A", prompt="test")


# ==================== Luma Dream Machine 测试 ====================


class TestLumaAdapter:
    """LumaAdapter 基本能力测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> LumaAdapter:
        config = ProviderConfig(provider_type="luma", api_key=mock_api_key)
        return LumaAdapter(config=config)

    def test_adapter_init(self, adapter: LumaAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "luma"
        assert adapter.provider_name == "Luma Dream Machine"
        assert "video" in adapter.supported_capabilities
        assert "chat" not in adapter.supported_capabilities
        assert "image" not in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: LumaAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("video")
        assert not adapter.supports_capability("chat")
        assert not adapter.supports_capability("image")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: LumaAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None
            assert "lumalabs.ai" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_chat_not_supported(self, adapter: LumaAdapter) -> None:
        """测试文本对话不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.chat(
                model="test", messages=[{"role": "user", "content": "hi"}]
            )

    @pytest.mark.asyncio
    async def test_image_not_supported(self, adapter: LumaAdapter) -> None:
        """测试图像生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.image_generate(model="test", prompt="cat")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: LumaAdapter) -> None:
        """测试获取所有模型"""
        models = await adapter.list_models()
        assert len(models) >= 3
        model_ids = {m.id for m in models}
        assert "ray-2" in model_ids
        assert "ray-2-flash" in model_ids
        assert "dream-machine" in model_ids
        for m in models:
            assert m.type == "video"
            assert "text2video" in m.capabilities

    @pytest.mark.asyncio
    async def test_list_video_models_filter(self, adapter: LumaAdapter) -> None:
        """测试按类型过滤模型"""
        video_models = await adapter.list_models(model_type="video")
        assert len(video_models) == 3
        chat_models = await adapter.list_models(model_type="chat")
        assert len(chat_models) == 0


class TestLumaVideoCreate:
    """Luma 视频生成 API 测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> LumaAdapter:
        config = ProviderConfig(provider_type="luma", api_key=mock_api_key)
        return LumaAdapter(config=config)

    @pytest.mark.asyncio
    async def test_video_create_text2video(self, adapter: LumaAdapter) -> None:
        """测试文生视频创建"""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "gen-abc123",
            "state": "queued",
            "prompt": "A cat walking",
            "model": "ray-2",
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        task = await adapter.video_create(model="ray-2", prompt="A cat walking")
        assert task.task_id == "gen-abc123"
        assert task.model == "ray-2"
        assert task.status == "pending"

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/generations"
        body = call_args[1]["json"]
        assert body["prompt"] == "A cat walking"
        assert body["model"] == "ray-2"

    @pytest.mark.asyncio
    async def test_video_create_default_model(self, adapter: LumaAdapter) -> None:
        """测试默认模型"""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "gen-1", "state": "queued"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        await adapter.video_create(model="", prompt="test")
        body = mock_client.post.call_args[1]["json"]
        assert body["model"] == "ray-2"

    @pytest.mark.asyncio
    async def test_video_create_with_params(self, adapter: LumaAdapter) -> None:
        """测试带参数的视频创建"""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "gen-2", "state": "queued"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        await adapter.video_create(
            model="ray-2",
            prompt="A dog running",
            aspect_ratio="16:9",
            duration="5s",
            resolution="720p",
            loop=True,
            negative_prompt="blurry",
            camera_motion="zoom_in",
        )

        body = mock_client.post.call_args[1]["json"]
        assert body["aspect_ratio"] == "16:9"
        assert body["duration"] == "5s"
        assert body["resolution"] == "720p"
        assert body["loop"] is True
        assert body["negative_prompt"] == "blurry"
        assert body["camera_motion"] == "zoom_in"

    @pytest.mark.asyncio
    async def test_video_create_with_reference_image(
        self, adapter: LumaAdapter
    ) -> None:
        """测试图生视频（起始帧）"""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "gen-3", "state": "queued"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        await adapter.video_create(
            model="ray-2",
            prompt="Make it come alive",
            reference_images=["https://example.com/start.png"],
        )

        body = mock_client.post.call_args[1]["json"]
        assert "keyframes" in body
        assert body["keyframes"]["frame0"]["type"] == "image"
        assert body["keyframes"]["frame0"]["url"] == "https://example.com/start.png"

    @pytest.mark.asyncio
    async def test_video_create_with_start_and_end(self, adapter: LumaAdapter) -> None:
        """测试首尾帧控制"""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "gen-4", "state": "queued"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        await adapter.video_create(
            model="ray-2",
            prompt="Smooth transition",
            reference_images=["https://example.com/start.png"],
            end_image="https://example.com/end.png",
        )

        body = mock_client.post.call_args[1]["json"]
        assert "frame0" in body["keyframes"]
        assert "frame1" in body["keyframes"]
        assert body["keyframes"]["frame1"]["url"] == "https://example.com/end.png"


class TestLumaVideoPoll:
    """Luma 视频轮询测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> LumaAdapter:
        config = ProviderConfig(provider_type="luma", api_key=mock_api_key)
        return LumaAdapter(config=config)

    @pytest.mark.asyncio
    async def test_video_poll_completed(self, adapter: LumaAdapter) -> None:
        """测试视频生成完成"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "gen-abc123",
            "state": "completed",
            "assets": {"video": "https://cdn.lumalabs.ai/gen-abc123.mp4"},
            "created_at": 1735689600,
            "updated_at": 1735690800,
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        adapter._http_client = mock_client

        status = await adapter.video_poll(task_id="gen-abc123")
        assert status.task_id == "gen-abc123"
        assert status.status == "success"
        assert status.video_url == "https://cdn.lumalabs.ai/gen-abc123.mp4"
        assert status.progress == 100

    @pytest.mark.asyncio
    async def test_video_poll_dreaming(self, adapter: LumaAdapter) -> None:
        """测试视频生成中"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "gen-1",
            "state": "dreaming",
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        adapter._http_client = mock_client

        status = await adapter.video_poll(task_id="gen-1")
        assert status.status == "processing"
        assert status.progress == 30
        assert status.video_url is None

    @pytest.mark.asyncio
    async def test_video_poll_queued(self, adapter: LumaAdapter) -> None:
        """测试视频排队中"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "gen-1",
            "state": "queued",
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        adapter._http_client = mock_client

        status = await adapter.video_poll(task_id="gen-1")
        assert status.status == "pending"
        assert status.progress == 5

    @pytest.mark.asyncio
    async def test_video_poll_failed(self, adapter: LumaAdapter) -> None:
        """测试视频生成失败"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "gen-fail",
            "state": "failed",
            "failure_reason": "Content policy violation",
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        adapter._http_client = mock_client

        status = await adapter.video_poll(task_id="gen-fail")
        assert status.status == "failed"
        assert status.error == "Content policy violation"

    def test_status_mapping(self, adapter: LumaAdapter) -> None:
        """测试状态映射"""
        assert adapter._map_luma_status("queued") == "pending"
        assert adapter._map_luma_status("dreaming") == "processing"
        assert adapter._map_luma_status("completed") == "success"
        assert adapter._map_luma_status("failed") == "failed"
        assert adapter._map_luma_status("unknown") == "pending"


# ==================== Meta Llama 测试 ====================


class TestLlamaAdapter:
    """LlamaAdapter 基本能力测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> LlamaAdapter:
        config = ProviderConfig(provider_type="llama", api_key=mock_api_key)
        return LlamaAdapter(config=config)

    def test_adapter_init(self, adapter: LlamaAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "llama"
        assert adapter.provider_name == "Meta Llama"
        assert "chat" in adapter.supported_capabilities
        assert "vision" in adapter.supported_capabilities
        assert "image" not in adapter.supported_capabilities
        assert "video" not in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: LlamaAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("chat")
        assert adapter.supports_capability("vision")
        assert not adapter.supports_capability("image")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: LlamaAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None
            assert "api.llama.com" in str(a._http_client.base_url)

    @pytest.mark.asyncio
    async def test_image_not_supported(self, adapter: LlamaAdapter) -> None:
        """测试图像生成不支持（多模态通过 chat 理解图片，不生成）"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.image_generate(model="test", prompt="cat")

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: LlamaAdapter) -> None:
        """测试视频生成不支持"""
        from agn.core.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(model="test", prompt="cat")
        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_poll(task_id="test")

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: LlamaAdapter) -> None:
        """测试获取所有模型（实时拉取 /models 端点）"""
        await adapter.start()
        try:
            mock_result = {
                "data": [
                    {"id": "llama-4-maverick", "name": "Llama 4 Maverick"},
                    {"id": "llama-4-scout", "name": "Llama 4 Scout"},
                    {"id": "llama-3.3-70b-instruct", "name": "Llama 3.3 70B"},
                    {"id": "llama-3.1-405b-instruct", "name": "Llama 3.1 405B"},
                    {"id": "llama-3.1-70b-instruct", "name": "Llama 3.1 70B"},
                    {"id": "llama-3.1-8b-instruct", "name": "Llama 3.1 8B"},
                    {"id": "llama-guard-4", "name": "Llama Guard 4"},
                ]
            }
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json = MagicMock(return_value=mock_result)

            with patch.object(
                adapter._http_client, "get", new_callable=AsyncMock
            ) as mock_get:
                mock_get.return_value = mock_resp
                models = await adapter.list_models()

                mock_get.assert_called_once_with(url="/models")
                assert len(models) == 7
                model_ids = {m.id for m in models}
                assert "llama-4-maverick" in model_ids
                assert "llama-4-scout" in model_ids
                assert "llama-3.3-70b-instruct" in model_ids
                assert "llama-3.1-405b-instruct" in model_ids
                assert "llama-3.1-70b-instruct" in model_ids
                assert "llama-3.1-8b-instruct" in model_ids
                for m in models:
                    assert m.provider == "llama"
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_list_chat_models_filter(self, adapter: LlamaAdapter) -> None:
        """测试按类型过滤模型（chat 类型，过滤掉 image/video）"""
        await adapter.start()
        try:
            # 混合类型响应：含 chat、image、video 模型
            mock_result = {
                "data": [
                    {"id": "llama-4-maverick", "name": "Llama 4 Maverick"},
                    {"id": "llama-3.3-70b-instruct", "name": "Llama 3.3 70B"},
                    {"id": "flux-pro", "name": "Flux Pro"},
                    {"id": "agnes-video-1", "name": "Agnes Video 1"},
                ]
            }
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json = MagicMock(return_value=mock_result)

            with patch.object(
                adapter._http_client, "get", new_callable=AsyncMock
            ) as mock_get:
                mock_get.return_value = mock_resp

                # 全量列表应包含多种类型
                all_models = await adapter.list_models()
                types = {m.type for m in all_models}
                assert "chat" in types
                assert "image" in types
                assert "video" in types

                # 按 chat 过滤只返回 chat 类型
                chat_models = await adapter.list_models(model_type="chat")
                assert len(chat_models) >= 1
                for m in chat_models:
                    assert m.type == "chat"

                # 按 image 过滤返回空（Llama 无原生 image 模型时）
                image_models = await adapter.list_models(model_type="image")
                # mock 中 flux-pro 会被推断为 image 类型
                assert all(m.type == "image" for m in image_models)
        finally:
            await adapter.close()


class TestLlamaChat:
    """Llama 聊天 API 测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> LlamaAdapter:
        config = ProviderConfig(provider_type="llama", api_key=mock_api_key)
        return LlamaAdapter(config=config)

    @pytest.mark.asyncio
    async def test_chat_basic(self, adapter: LlamaAdapter) -> None:
        """测试基本对话"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama-4-maverick",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        result = await adapter.chat(
            model="llama-4-maverick",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert result.choices[0].message.content == "Hello! How can I help you?"
        assert result.choices[0].message.role == "assistant"
        assert result.model == "llama-4-maverick"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage is not None
        assert result.usage.total_tokens == 30

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/chat/completions"
        body = call_args[1]["json"]
        assert body["model"] == "llama-4-maverick"
        assert len(body["messages"]) == 1

    @pytest.mark.asyncio
    async def test_chat_with_params(self, adapter: LlamaAdapter) -> None:
        """测试带参数的对话"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        await adapter.chat(
            model="llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.1,
            stop=["\n\n"],
            seed=42,
        )

        body = mock_client.post.call_args[1]["json"]
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 100
        assert body["top_p"] == 0.9
        assert body["frequency_penalty"] == 0.5
        assert body["presence_penalty"] == 0.1
        assert body["stop"] == ["\n\n"]
        assert body["seed"] == 42

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self, adapter: LlamaAdapter) -> None:
        """测试工具调用"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-tool",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Beijing"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]

        result = await adapter.chat(
            model="llama-4-maverick",
            messages=[{"role": "user", "content": "Weather in Beijing?"}],
            tools=tools,
        )
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        assert (
            result.choices[0].message.tool_calls[0]["function"]["name"] == "get_weather"
        )

    @pytest.mark.asyncio
    async def test_authentication_error(self, adapter: LlamaAdapter) -> None:
        """测试认证错误"""
        from agn.core.errors import AuthenticationError

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        with pytest.raises(AuthenticationError):
            await adapter.chat(
                model="llama-4-maverick", messages=[{"role": "user", "content": "hi"}]
            )

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, adapter: LlamaAdapter) -> None:
        """测试限流错误"""
        from agn.core.errors import RateLimitError

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        adapter._http_client = mock_client

        with pytest.raises(RateLimitError):
            await adapter.chat(
                model="llama-4-maverick", messages=[{"role": "user", "content": "hi"}]
            )


class TestLlamaStream:
    """Llama 流式输出测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> LlamaAdapter:
        config = ProviderConfig(provider_type="llama", api_key=mock_api_key)
        return LlamaAdapter(config=config)

    @pytest.mark.asyncio
    async def test_chat_stream(self, adapter: LlamaAdapter) -> None:
        """测试流式对话（验证 _parse_chunk 解析正确）"""
        # 直接测试 _parse_chunk 方法，避免复杂的 async context manager mock
        chunk1 = adapter._parse_chunk(
            '{"id":"1","choices":[{"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}',
            "llama-4-maverick",
        )
        assert chunk1 is not None
        assert chunk1.choices[0].delta.content == "Hello"

        chunk2 = adapter._parse_chunk(
            '{"id":"1","choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
            "llama-4-maverick",
        )
        assert chunk2 is not None
        assert chunk2.choices[0].delta.content == " world"

        chunk3 = adapter._parse_chunk(
            '{"id":"1","choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}',
            "llama-4-maverick",
        )
        assert chunk3 is not None
        assert chunk3.choices[0].delta.content == "!"
        assert chunk3.choices[0].finish_reason == "stop"

        # 测试无效 JSON 返回 None
        assert adapter._parse_chunk("invalid json", "model") is None

    @pytest.mark.asyncio
    async def test_parse_response_basic(self, adapter: LlamaAdapter) -> None:
        """测试非流式响应解析"""
        data = {
            "id": "chatcmpl-test",
            "created": 1700000000,
            "model": "llama-4-maverick",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        result = adapter._parse_response(data, "llama-4-maverick")
        assert result.id == "chatcmpl-test"
        assert result.choices[0].message.content == "Hi there!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.total_tokens == 8


# ==================== 路由映射测试 ====================


class TestRouterEmergingModelsMapping:
    """路由器新模型映射测试"""

    def test_router_has_ideogram_models(self) -> None:
        """测试路由表包含 Ideogram 模型"""
        from agn.router import Router

        ideogram_models = [
            "V_2A",
            "V_2A_TURBO",
            "V_2",
            "V_1",
            "V_1_TURBO",
        ]
        for model in ideogram_models:
            assert (
                Router.MODEL_PROVIDER_MAP.get(model) == "ideogram"
            ), f"{model} should map to ideogram"

    def test_router_has_luma_models(self) -> None:
        """测试路由表包含 Luma 模型"""
        from agn.router import Router

        luma_models = ["ray-2", "ray-2-flash", "dream-machine"]
        for model in luma_models:
            assert (
                Router.MODEL_PROVIDER_MAP.get(model) == "luma"
            ), f"{model} should map to luma"

    def test_router_has_llama_models(self) -> None:
        """测试路由表包含 Llama 模型"""
        from agn.router import Router

        llama_models = [
            "llama-4-maverick",
            "llama-4-scout",
            "llama-3.3-70b-instruct",
            "llama-3.1-405b-instruct",
            "llama-3.1-70b-instruct",
            "llama-3.1-8b-instruct",
        ]
        for model in llama_models:
            assert (
                Router.MODEL_PROVIDER_MAP.get(model) == "llama"
            ), f"{model} should map to llama"

    def test_adapter_factory_registration(self) -> None:
        """测试适配器工厂注册"""
        from agn.adapters import (
            IdeogramAdapter,
            LlamaAdapter,
            LumaAdapter,
        )
        from agn.adapters.factory import AdapterFactory

        assert AdapterFactory.get_adapter_class("ideogram") is IdeogramAdapter
        assert AdapterFactory.get_adapter_class("ideo") is IdeogramAdapter
        assert AdapterFactory.get_adapter_class("luma") is LumaAdapter
        assert AdapterFactory.get_adapter_class("dream-machine") is LumaAdapter
        assert AdapterFactory.get_adapter_class("lumalabs") is LumaAdapter
        assert AdapterFactory.get_adapter_class("llama") is LlamaAdapter
        assert AdapterFactory.get_adapter_class("meta-llama") is LlamaAdapter
        assert AdapterFactory.get_adapter_class("meta") is LlamaAdapter
