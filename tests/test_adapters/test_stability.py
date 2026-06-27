"""
AGN-SDK Stability AI 适配器测试

测试 StabilityAdapter 的各项功能，包括图像生成、错误处理等。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agn.adapters.stability import StabilityAdapter
from agn.core.errors import UnsupportedCapabilityError
from agn.models.common import ProviderConfig
from agn.models.image import ImageGenerationResult


class TestStabilityAdapter:
    """StabilityAdapter 类测试"""

    @pytest.fixture
    def adapter_config(self, mock_api_key: str) -> ProviderConfig:
        """创建适配器配置"""
        return ProviderConfig(
            provider_type="stability",
            api_key=mock_api_key,
        )

    @pytest.fixture
    def adapter(self, adapter_config: ProviderConfig) -> StabilityAdapter:
        """创建适配器实例"""
        return StabilityAdapter(config=adapter_config)

    def test_adapter_init(self, adapter: StabilityAdapter) -> None:
        """测试适配器初始化"""
        assert adapter.provider_type == "stability"
        assert adapter.provider_name == "Stability AI"
        assert "image" in adapter.supported_capabilities
        assert "chat" not in adapter.supported_capabilities
        assert "video" not in adapter.supported_capabilities

    def test_adapter_supports_capability(self, adapter: StabilityAdapter) -> None:
        """测试能力检查"""
        assert adapter.supports_capability("image")
        assert not adapter.supports_capability("chat")
        assert not adapter.supports_capability("video")

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, adapter: StabilityAdapter) -> None:
        """测试异步上下文管理器"""
        async with adapter as a:
            assert a._http_client is not None

    @pytest.mark.asyncio
    async def test_chat_not_supported(self, adapter: StabilityAdapter) -> None:
        """测试文本对话不支持"""

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.chat(
                model="test",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_video_not_supported(self, adapter: StabilityAdapter) -> None:
        """测试视频生成不支持"""

        with pytest.raises(UnsupportedCapabilityError):
            await adapter.video_create(
                model="test",
                prompt="A cat",
            )


class TestStabilityAdapterListModels:
    """StabilityAdapter 模型列表测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> StabilityAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="stability", api_key=mock_api_key)
        return StabilityAdapter(config=config)

    def _mock_response(self, json_data: list | dict) -> MagicMock:
        """创建模拟 HTTP 响应（Stability 返回顶层数组）"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.headers = {}
        return mock_resp

    @pytest.mark.asyncio
    async def test_list_all_models(self, adapter: StabilityAdapter) -> None:
        """测试获取所有模型（实时拉取 /v1/engines/list）"""
        await adapter.start()

        # Stability 返回顶层数组
        mock_engines = [
            {
                "id": "stable-diffusion-xl-1024-v1-1",
                "name": "Stable Diffusion XL 1.1",
                "type": "picture",
            },
            {
                "id": "stable-diffusion-3-medium",
                "name": "Stable Diffusion 3 Medium",
                "type": "picture",
            },
        ]

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_engines)

            models = await adapter.list_models()

            # 验证调用了 /v1/engines/list 端点
            mock_get.assert_called_once_with("/v1/engines/list")
            assert len(models) == 2
            for model in models:
                assert model.provider == "stability"
            ids = {m.id for m in models}
            assert "stable-diffusion-xl-1024-v1-1" in ids
            assert "stable-diffusion-3-medium" in ids

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_image_models(self, adapter: StabilityAdapter) -> None:
        """测试获取图像模型（按类型过滤）"""
        await adapter.start()

        # 模型 ID 中包含 "stable-diffusion"，基类 _infer_type 会推断为 image
        mock_engines = [
            {
                "id": "stable-diffusion-xl-1024-v1-1",
                "name": "Stable Diffusion XL 1.1",
            },
            {
                "id": "stable-diffusion-3-medium",
                "name": "Stable Diffusion 3 Medium",
            },
        ]

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_engines)

            models = await adapter.list_models(model_type="image")
            assert len(models) > 0
            for model in models:
                assert model.type == "image"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_models_wraps_top_level_array(
        self, adapter: StabilityAdapter
    ) -> None:
        """测试 Stability 顶层数组响应被正确包装为 {"data": [...]}"""
        await adapter.start()

        mock_engines = [
            {
                "id": "stable-diffusion-xl-1024-v1-1",
                "name": "Stable Diffusion XL 1.1",
            }
        ]

        with patch.object(
            adapter._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = self._mock_response(mock_engines)

            models = await adapter.list_models()

            assert len(models) == 1
            assert models[0].id == "stable-diffusion-xl-1024-v1-1"
            assert models[0].name == "Stable Diffusion XL 1.1"
            assert models[0].provider == "stability"

        await adapter.close()


class TestStabilityAdapterStatusMapping:
    """StabilityAdapter 状态映射测试（图像生成无异步状态）"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> StabilityAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="stability", api_key=mock_api_key)
        return StabilityAdapter(config=config)

    def test_adapter_has_default_engine(self, mock_api_key: str) -> None:
        """测试默认引擎配置"""
        config = ProviderConfig(provider_type="stability", api_key=mock_api_key)
        adapter = StabilityAdapter(config=config)
        assert adapter.default_engine == "stable-diffusion-xl-1024-v1-1"

    def test_adapter_custom_engine(self, mock_api_key: str) -> None:
        """测试自定义引擎配置"""
        config = ProviderConfig(provider_type="stability", api_key=mock_api_key)
        adapter = StabilityAdapter(
            config=config, default_engine="stable-diffusion-3-medium"
        )
        assert adapter.default_engine == "stable-diffusion-3-medium"


class TestStabilityAdapterImageGenerateMockHTTP:
    """StabilityAdapter 图像生成 Mock HTTP 测试"""

    @pytest.fixture
    def adapter(self, mock_api_key: str) -> StabilityAdapter:
        """创建适配器实例"""
        config = ProviderConfig(provider_type="stability", api_key=mock_api_key)
        return StabilityAdapter(config=config)

    def _mock_response(self, data: dict, status_code: int = 200) -> MagicMock:
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=data)
        mock_resp.headers = {}
        return mock_resp

    @pytest.mark.asyncio
    async def test_image_generate_basic(
        self, adapter: StabilityAdapter, sample_image_prompt: str
    ) -> None:
        """测试基础图像生成请求和响应解析"""
        await adapter.start()

        mock_result = {
            "artifacts": [
                {
                    "base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                }
            ]
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.image_generate(
                model="stable-diffusion-xl-1024-v1-1",
                prompt=sample_image_prompt,
            )

            assert isinstance(result, ImageGenerationResult)
            assert result.model == "stable-diffusion-xl-1024-v1-1"
            assert len(result.data) == 1
            assert result.data[0].b64_json is not None
            assert result.data[0].url is None

        await adapter.close()

    @pytest.mark.asyncio
    async def test_image_generate_with_params(
        self, adapter: StabilityAdapter, sample_image_prompt: str
    ) -> None:
        """测试带参数的图像生成请求"""
        await adapter.start()

        mock_result = {
            "artifacts": [
                {
                    "base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                }
            ]
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.image_generate(
                model="stable-diffusion-3-medium",
                prompt=sample_image_prompt,
                width=512,
                height=512,
                steps=30,
                seed=42,
                cfg_scale=7.5,
                samples=2,
                negative_prompt="blurry, low quality",
                style_preset="photograph",
            )

            assert isinstance(result, ImageGenerationResult)
            # 验证请求参数被正确传递
            call_args = mock_post.call_args
            assert "/v1/generation/stable-diffusion-3-medium/text-to-image" in str(
                call_args
            )
            body = call_args.kwargs.get("json") or call_args.args[0]
            assert body["width"] == 512
            assert body["height"] == 512
            assert body["steps"] == 30
            assert body["seed"] == 42
            assert body["cfg_scale"] == 7.5
            assert body["samples"] == 2
            assert body["style_preset"] == "photograph"
            # 验证负面提示词
            text_prompts = body["text_prompts"]
            assert len(text_prompts) == 2
            assert text_prompts[0]["text"] == sample_image_prompt
            assert text_prompts[0]["weight"] == 1
            assert text_prompts[1]["text"] == "blurry, low quality"
            assert text_prompts[1]["weight"] == -1

        await adapter.close()

    @pytest.mark.asyncio
    async def test_image_generate_multiple_images(
        self, adapter: StabilityAdapter, sample_image_prompt: str
    ) -> None:
        """测试生成多张图像"""
        await adapter.start()

        mock_result = {
            "artifacts": [
                {"base64": "image_data_1"},
                {"base64": "image_data_2"},
                {"base64": "image_data_3"},
            ]
        }

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.image_generate(
                model="stable-diffusion-xl-1024-v1-1",
                prompt=sample_image_prompt,
                samples=3,
            )

            assert isinstance(result, ImageGenerationResult)
            assert len(result.data) == 3

        await adapter.close()

    @pytest.mark.asyncio
    async def test_image_generate_no_artifacts(
        self, adapter: StabilityAdapter, sample_image_prompt: str
    ) -> None:
        """测试无图像生成时的错误处理"""
        await adapter.start()

        mock_result = {"artifacts": []}

        with patch.object(
            adapter._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            from agn.core.errors import APIError

            with pytest.raises(APIError, match="No image generated"):
                await adapter.image_generate(
                    model="stable-diffusion-xl-1024-v1-1",
                    prompt=sample_image_prompt,
                )

        await adapter.close()
