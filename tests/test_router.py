"""
AGN-SDK 路由器测试
"""

import pytest

from agn import Router
from agn.core.errors import ModelNotFoundError


class TestRouter:
    """Router 类测试"""

    def test_router_init(self, mock_provider_config: dict) -> None:
        """测试路由器初始化"""
        router = Router(providers=[mock_provider_config])
        assert "agnes" in router.providers
        assert router.strategy == "first"
        assert router.enable_fallback is True

    def test_router_multiple_providers(self, mock_provider_config: dict) -> None:
        """测试多 Provider 配置"""
        providers = [
            mock_provider_config,
            {**mock_provider_config, "provider_type": "openai", "api_key": "key2"},
        ]
        router = Router(providers=providers)
        assert "agnes" in router.providers
        assert "openai" in router.providers
        assert len(router._provider_order) == 2

    def test_select_provider_by_model(self, mock_provider_config: dict) -> None:
        """测试根据模型名选择 Provider"""
        router = Router(providers=[mock_provider_config])
        provider = router._select_provider("claude-3-opus")
        assert provider == "agnes"

    def test_select_provider_default(self, mock_provider_config: dict) -> None:
        """测试默认 Provider 选择"""
        router = Router(
            providers=[mock_provider_config],
            default_provider="agnes",
        )
        provider = router._select_provider("unknown-model")
        assert provider == "agnes"

    def test_select_provider_not_found(self) -> None:
        """测试 Provider 不存在时抛出错误"""
        router = Router(providers=[])
        with pytest.raises(ModelNotFoundError):
            router._select_provider("claude-3-opus")

    def test_router_with_weight(self, mock_provider_config: dict) -> None:
        """测试带权重的 Provider 配置"""
        providers = [
            {**mock_provider_config, "weight": 3},
            {
                **mock_provider_config,
                "provider_type": "openai",
                "api_key": "key2",
                "weight": 2,
            },
        ]
        router = Router(providers=providers, strategy="weighted")
        assert router._weights["agnes"] == 3
        assert router._weights["openai"] == 2

    def test_router_custom_strategy(self, mock_provider_config: dict) -> None:
        """测试自定义路由策略"""
        router = Router(providers=[mock_provider_config], strategy="round_robin")
        assert router.strategy == "round_robin"

    def test_router_fallback_config(self, mock_provider_config: dict) -> None:
        """测试 Fallback 配置"""
        router = Router(
            providers=[mock_provider_config],
            enable_fallback=False,
            max_retries=0,
        )
        assert router.enable_fallback is False
        assert router.max_retries == 0


class TestRouterStrategy:
    """路由策略测试"""

    @pytest.fixture
    def multi_provider_router(self, mock_provider_config: dict) -> Router:
        """创建多 Provider 路由器"""
        providers = [
            {
                **mock_provider_config,
                "provider_type": "agnes",
                "api_key": "key1",
                "weight": 5,
            },
            {
                **mock_provider_config,
                "provider_type": "openai",
                "api_key": "key2",
                "weight": 3,
            },
            {
                **mock_provider_config,
                "provider_type": "qwen",
                "api_key": "key3",
                "weight": 2,
            },
        ]
        return Router(providers=providers, strategy="first")

    def test_strategy_first(self, multi_provider_router: Router) -> None:
        """测试 first 策略"""
        multi_provider_router.strategy = "first"
        # 没有适配器的情况下，_get_capable_providers 会返回空
        # 所以这里直接测试 _pick_by_strategy
        candidates = ["agnes", "openai", "qwen"]
        result = multi_provider_router._pick_by_strategy(candidates)
        assert result == "agnes"

    def test_strategy_round_robin(self, multi_provider_router: Router) -> None:
        """测试 round_robin 策略"""
        multi_provider_router.strategy = "round_robin"
        candidates = ["agnes", "openai", "qwen"]

        results = []
        for _ in range(6):
            results.append(multi_provider_router._pick_by_strategy(candidates))

        # 应该循环出现
        assert results[0] == "agnes"
        assert results[1] == "openai"
        assert results[2] == "qwen"
        assert results[3] == "agnes"
        assert results[4] == "openai"
        assert results[5] == "qwen"

    def test_strategy_random(self, multi_provider_router: Router) -> None:
        """测试 random 策略"""
        multi_provider_router.strategy = "random"
        candidates = ["agnes", "openai", "qwen"]

        # 多次调用，结果应该在候选中
        for _ in range(10):
            result = multi_provider_router._pick_by_strategy(candidates)
            assert result in candidates

    def test_strategy_weighted(self, multi_provider_router: Router) -> None:
        """测试 weighted 策略"""
        multi_provider_router.strategy = "weighted"
        candidates = ["agnes", "openai", "qwen"]

        # 多次调用，结果应该在候选中
        for _ in range(20):
            result = multi_provider_router._pick_by_strategy(candidates)
            assert result in candidates

    def test_strategy_latency(self, multi_provider_router: Router) -> None:
        """测试 latency 策略"""
        multi_provider_router.strategy = "latency"
        candidates = ["agnes", "openai", "qwen"]

        # 设置延迟数据
        multi_provider_router._provider_latency["agnes"] = 0.5
        multi_provider_router._provider_latency["openai"] = 0.3
        multi_provider_router._provider_latency["qwen"] = 0.8

        result = multi_provider_router._pick_by_strategy(candidates)
        # 应该选延迟最低的 openai
        assert result == "openai"

    def test_strategy_single_candidate(self, multi_provider_router: Router) -> None:
        """测试只有一个候选时的策略"""
        candidates = ["agnes"]
        for strategy in ["first", "round_robin", "random", "weighted", "latency"]:
            multi_provider_router.strategy = strategy
            result = multi_provider_router._pick_by_strategy(candidates)
            assert result == "agnes"

    def test_strategy_empty_candidates(self, multi_provider_router: Router) -> None:
        """测试空候选列表抛出错误"""
        with pytest.raises(ModelNotFoundError):
            multi_provider_router._pick_by_strategy([])


class TestRouterHealth:
    """健康状态测试"""

    @pytest.fixture
    def multi_provider_router(self, mock_provider_config: dict) -> Router:
        """创建多 Provider 路由器"""
        providers = [
            {**mock_provider_config, "provider_type": "agnes", "api_key": "key1"},
            {**mock_provider_config, "provider_type": "openai", "api_key": "key2"},
        ]
        return Router(providers=providers)

    def test_initial_health_status(self, multi_provider_router: Router) -> None:
        """测试初始健康状态"""
        health = multi_provider_router.get_health_status()
        assert health["agnes"] is True
        assert health["openai"] is True

    def test_mark_unhealthy(self, multi_provider_router: Router) -> None:
        """测试标记 Provider 不健康"""
        multi_provider_router._provider_health["agnes"] = False
        health = multi_provider_router.get_health_status()
        assert health["agnes"] is False
        assert health["openai"] is True

    def test_latency_stats_initial(self, multi_provider_router: Router) -> None:
        """测试初始延迟统计"""
        latency = multi_provider_router.get_latency_stats()
        assert latency["agnes"] == 0.0
        assert latency["openai"] == 0.0

    def test_latency_stats_update(self, multi_provider_router: Router) -> None:
        """测试延迟统计更新"""
        multi_provider_router._provider_latency["agnes"] = 0.1
        latency = multi_provider_router.get_latency_stats()
        assert latency["agnes"] == 0.1


class TestRouterFallback:
    """Fallback 测试"""

    @pytest.fixture
    def multi_provider_router(self, mock_provider_config: dict) -> Router:
        """创建多 Provider 路由器"""
        providers = [
            {**mock_provider_config, "provider_type": "agnes", "api_key": "key1"},
            {**mock_provider_config, "provider_type": "openai", "api_key": "key2"},
        ]
        return Router(providers=providers, enable_fallback=True, max_retries=2)

    def test_fallback_providers_list(self, multi_provider_router: Router) -> None:
        """测试 Fallback Provider 列表"""
        fallbacks = multi_provider_router._get_fallback_providers("agnes", "chat")
        # 由于还没启动适配器，_get_capable_providers 会返回空
        # 这里只测试逻辑
        assert isinstance(fallbacks, list)

    def test_fallback_disabled(self, mock_provider_config: dict) -> None:
        """测试禁用 Fallback"""
        router = Router(
            providers=[mock_provider_config],
            enable_fallback=False,
        )
        assert router.enable_fallback is False


class TestRouterModelMapping:
    """模型映射测试"""

    def test_register_model_mapping(self, mock_provider_config: dict) -> None:
        """测试注册自定义模型映射"""
        router = Router(providers=[mock_provider_config])
        router.register_model_mapping("my-custom-model", "agnes")
        assert router.MODEL_PROVIDER_MAP["my-custom-model"] == "agnes"


class TestModelProviderMapping:
    """模型-Provider 映射测试"""

    def test_known_mappings(self) -> None:
        """测试已知模型映射"""
        router = Router(providers=[])

        # Agnes AI
        assert Router.MODEL_PROVIDER_MAP.get("claude-3-opus") == "agnes"
        assert Router.MODEL_PROVIDER_MAP.get("claude-3-sonnet") == "agnes"
        assert Router.MODEL_PROVIDER_MAP.get("dall-e-3") == "agnes"

        # OpenAI
        assert Router.MODEL_PROVIDER_MAP.get("gpt-4o") == "openai"
        assert Router.MODEL_PROVIDER_MAP.get("gpt-3.5-turbo") == "openai"

        # Runway
        assert Router.MODEL_PROVIDER_MAP.get("gen-3") == "runway"
        assert Router.MODEL_PROVIDER_MAP.get("gen-3-turbo") == "runway"

        # Pika
        assert Router.MODEL_PROVIDER_MAP.get("pika-1.0") == "pika"
        assert Router.MODEL_PROVIDER_MAP.get("pika-2") == "pika"

        # Stability
        assert (
            Router.MODEL_PROVIDER_MAP.get("stable-diffusion-xl-1024-v1-1")
            == "stability"
        )
        assert Router.MODEL_PROVIDER_MAP.get("sdxl") == "stability"

        # Qwen
        assert Router.MODEL_PROVIDER_MAP.get("qwen-turbo") == "qwen"
        assert Router.MODEL_PROVIDER_MAP.get("qwen-plus") == "qwen"

        # Zhipu
        assert Router.MODEL_PROVIDER_MAP.get("glm-4") == "zhipu"
        assert Router.MODEL_PROVIDER_MAP.get("glm-3-turbo") == "zhipu"

        # Anthropic
        assert (
            Router.MODEL_PROVIDER_MAP.get("claude-3-5-sonnet-20241022") == "anthropic"
        )
        assert Router.MODEL_PROVIDER_MAP.get("claude-3-opus-20240229") == "anthropic"

        # Gemini
        assert Router.MODEL_PROVIDER_MAP.get("gemini-2.5-pro") == "gemini"
        assert Router.MODEL_PROVIDER_MAP.get("gemini-1.5-flash") == "gemini"

        # Doubao
        assert Router.MODEL_PROVIDER_MAP.get("doubao-pro-128k") == "doubao"
        assert Router.MODEL_PROVIDER_MAP.get("doubao-lite-4k") == "doubao"

        # ERNIE
        assert Router.MODEL_PROVIDER_MAP.get("completions_pro") == "ernie"

        # Kling
        assert Router.MODEL_PROVIDER_MAP.get("kling-v2") == "kling"

        # Kimi
        assert Router.MODEL_PROVIDER_MAP.get("moonshot-v1-128k") == "kimi"
        assert Router.MODEL_PROVIDER_MAP.get("kimi-k2.5") == "kimi"
        assert Router.MODEL_PROVIDER_MAP.get("kimi-k2.7-code") == "kimi"

        # MiniMax
        assert Router.MODEL_PROVIDER_MAP.get("abab6.5s-chat") == "minimax"
        assert Router.MODEL_PROVIDER_MAP.get("MiniMax-M1") == "minimax"
        assert Router.MODEL_PROVIDER_MAP.get("MiniMax-VL-01") == "minimax"

        # Seedream/Seedance
        assert Router.MODEL_PROVIDER_MAP.get("seedream-5.0") == "seedream"
        assert Router.MODEL_PROVIDER_MAP.get("seedance-2.0") == "seedance"

        # Doubao Seed
        assert Router.MODEL_PROVIDER_MAP.get("doubao-seed-2-0-pro-260215") == "doubao"
        assert Router.MODEL_PROVIDER_MAP.get("doubao-2-1-pro") == "doubao"

        # DeepSeek
        assert Router.MODEL_PROVIDER_MAP.get("deepseek-v4-pro") == "deepseek"
        assert Router.MODEL_PROVIDER_MAP.get("deepseek-v4-flash") == "deepseek"
        assert Router.MODEL_PROVIDER_MAP.get("deepseek-coder") == "deepseek"

        # StepFun
        assert Router.MODEL_PROVIDER_MAP.get("step-3-flash") == "stepfun"
        assert Router.MODEL_PROVIDER_MAP.get("step-3-128k") == "stepfun"
        assert Router.MODEL_PROVIDER_MAP.get("step-1o-turbo") == "stepfun"

        # Mistral
        assert Router.MODEL_PROVIDER_MAP.get("mistral-sonnet-4-2505") == "mistral"
        assert Router.MODEL_PROVIDER_MAP.get("mixtral-8x22b-2404") == "mistral"
        assert Router.MODEL_PROVIDER_MAP.get("codestral-2405") == "mistral"

        # Cohere
        assert Router.MODEL_PROVIDER_MAP.get("command-r-plus-08-2024") == "cohere"
        assert Router.MODEL_PROVIDER_MAP.get("command-r-08-2024") == "cohere"

        # Perplexity
        assert Router.MODEL_PROVIDER_MAP.get("sonar-pro") == "perplexity"
        assert Router.MODEL_PROVIDER_MAP.get("sonar-reasoning") == "perplexity"

        # xAI Grok
        assert Router.MODEL_PROVIDER_MAP.get("grok-3") == "grok"
        assert Router.MODEL_PROVIDER_MAP.get("grok-3-latest") == "grok"

        # 零一万物 Yi
        assert Router.MODEL_PROVIDER_MAP.get("yi-large") == "yi"
        assert Router.MODEL_PROVIDER_MAP.get("yi-34b-chat-200k") == "yi"

        # 商汤日日新
        assert Router.MODEL_PROVIDER_MAP.get("sensechat-5") == "sensenova"

        # 腾讯混元
        assert Router.MODEL_PROVIDER_MAP.get("hunyuan-turbo") == "hunyuan"
        assert Router.MODEL_PROVIDER_MAP.get("hunyuan-lite") == "hunyuan"

        # Groq
        assert Router.MODEL_PROVIDER_MAP.get("llama-3.3-70b-versatile") == "groq"
        assert Router.MODEL_PROVIDER_MAP.get("mixtral-8x7b-32768") == "groq"

        # SiliconFlow
        assert Router.MODEL_PROVIDER_MAP.get("GLM-4.7") == "siliconflow"
        assert (
            Router.MODEL_PROVIDER_MAP.get("deepseek-ai/DeepSeek-V3.2") == "siliconflow"
        )

        # Together AI
        assert (
            Router.MODEL_PROVIDER_MAP.get("meta-llama/Llama-3-70b-chat-hf")
            == "togetherai"
        )

        # Fireworks AI
        assert (
            Router.MODEL_PROVIDER_MAP.get(
                "accounts/fireworks/models/llama-v3p1-405b-instruct"
            )
            == "fireworksai"
        )

        # Cloudflare Workers AI
        assert (
            Router.MODEL_PROVIDER_MAP.get("@cf/meta/llama-3.1-8b-instruct")
            == "cloudflareai"
        )
        assert (
            Router.MODEL_PROVIDER_MAP.get("@cf/mistral/mistral-7b-instruct-v0.2")
            == "cloudflareai"
        )

    def test_mapping_count(self) -> None:
        """测试映射数量"""
        # 至少应该有 120+ 个预定义映射
        assert len(Router.MODEL_PROVIDER_MAP) >= 120
