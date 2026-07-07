//! 适配器工厂
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/factory.py`。
//!
//! 设计要点（与设计文档 5.2 节一致）：
//! - 用编译期显式 `match` 替代 Python 运行时注册（更静态、更安全）
//! - 新增适配器=加 match 分支
//! - 阶段 0.4 暂只占位分支（返 ProviderNotFound），具体适配器阶段 1 起填充

use crate::adapter::Adapter;
use crate::adapters::additional_models::{
    GrokAdapter, GroqAdapter, HunyuanAdapter, SenseNovaAdapter, YiAdapter,
};
use crate::adapters::aggregation_platforms::{
    CloudflareAIAdapter, FireworksAIAdapter, SiliconFlowAdapter, TogetherAIAdapter,
};
use crate::adapters::agnes::AgnesAdapter;
use crate::adapters::azure::AzureAdapter;
use crate::adapters::chinese::{
    DoubaoAdapter, ErnieAdapter, KimiAdapter, MiniMaxAdapter, QwenAdapter, ZhipuAdapter,
};
use crate::adapters::echo::EchoAdapter;
use crate::adapters::emerging_models::{IdeogramAdapter, LlamaAdapter, LumaAdapter};
use crate::adapters::gemini::GeminiAdapter;
use crate::adapters::more_models::{
    CohereAdapter, DeepSeekAdapter, MistralAdapter, PerplexityAdapter, StepFunAdapter,
};
use crate::adapters::openai::OpenAiAdapter;
use crate::adapters::volcengine_cv::VolcengineCvAdapter;
use crate::config::ProviderConfig;
use crate::error::{AibridgeError, Result};

/// 已支持的 provider 列表（用于错误信息与测试）
///
/// 阶段 1 起逐步填充实际支持的 provider 名。
/// `echo` 为阶段 0.6 管线验证用的 mock 适配器，常驻可用。
pub const KNOWN_PROVIDERS: &[&str] = &[
    "echo",
    "openai",
    "agnes",
    "volcengine_cv",
    "gemini",
    // 阶段 2a 已实现（兼容族）：
    "azure",
    "siliconflow",
    "togetherai",
    "fireworksai",
    "cloudflareai",
    "grok",
    "yi",
    "sensenova",
    "hunyuan",
    "groq",
    "deepseek",
    "stepfun",
    "mistral",
    "cohere",
    "perplexity",
    // 阶段 2a 第三批 emerging_models：
    "ideogram",
    "luma",
    "llama",
    // 阶段 2a 第三批 chinese：
    "qwen",
    "zhipu",
    "doubao",
    "ernie",
    "kimi",
    "minimax",
    // 阶段 2b/2c 待实现：
    "anthropic",
    "runway",
    "pika",
    "kling",
    "stability",
    "edge-tts",
    "elevenlabs",
    "cartesia",
    "deepgram",
    "assemblyai",
];

/// 根据配置创建适配器实例
///
/// 对应 Python v1 `AdapterFactory.create`。
/// `echo` 为阶段 0.6 管线验证用 mock 适配器，已实现；
/// 其余 provider 阶段 0.4 占位（返 ProviderNotFound），阶段 1 起填充。
pub fn create_adapter(config: ProviderConfig) -> Result<Box<dyn Adapter>> {
    let provider = config.provider_type.clone();
    match provider.as_str() {
        // Echo（Mock）适配器：阶段 0.6 管线验证用，已实现
        "echo" => Ok(Box::new(EchoAdapter::new())),
        // 阶段 1 MVP 适配器：阶段 1.0 已实现具体构造逻辑
        "openai" => Ok(Box::new(OpenAiAdapter::new(config)?)),
        "agnes" => Ok(Box::new(AgnesAdapter::new(config)?)),
        "volcengine_cv" => Ok(Box::new(VolcengineCvAdapter::new(config)?)),
        "gemini" => Ok(Box::new(GeminiAdapter::new(config)?)),
        // 阶段 2a 适配器：OpenAI 兼容族
        "azure" => Ok(Box::new(AzureAdapter::new(config)?)),
        // 聚合平台：别名对齐 Python agn/adapters/aggregation_platforms.py 末尾 register 调用
        "siliconflow" | "sf" => Ok(Box::new(SiliconFlowAdapter::new(config)?)),
        "togetherai" | "together" => Ok(Box::new(TogetherAIAdapter::new(config)?)),
        "fireworksai" | "fireworks" => Ok(Box::new(FireworksAIAdapter::new(config)?)),
        "cloudflareai" | "cloudflare" | "workersai" => {
            Ok(Box::new(CloudflareAIAdapter::new(config)?))
        }
        // 扩展模型：别名对齐 Python agn/adapters/additional_models.py 末尾 register 调用
        "grok" | "xaigrok" => Ok(Box::new(GrokAdapter::new(config)?)),
        "yi" | "lingyiwanwu" => Ok(Box::new(YiAdapter::new(config)?)),
        "sensenova" | "shangtang" => Ok(Box::new(SenseNovaAdapter::new(config)?)),
        "hunyuan" | "tencent_hunyuan" => Ok(Box::new(HunyuanAdapter::new(config)?)),
        "groq" => Ok(Box::new(GroqAdapter::new(config)?)),
        // 更多模型：别名对齐 Python agn/adapters/more_models.py 末尾 register 调用
        "deepseek" => Ok(Box::new(DeepSeekAdapter::new(config)?)),
        "stepfun" | "step" => Ok(Box::new(StepFunAdapter::new(config)?)),
        "mistral" => Ok(Box::new(MistralAdapter::new(config)?)),
        "cohere" => Ok(Box::new(CohereAdapter::new(config)?)),
        "perplexity" => Ok(Box::new(PerplexityAdapter::new(config)?)),
        // 新兴模型：别名对齐 Python agn/adapters/emerging_models.py 末尾 register 调用
        "ideogram" | "ideo" => Ok(Box::new(IdeogramAdapter::new(config)?)),
        "luma" | "dream-machine" | "lumalabs" => Ok(Box::new(LumaAdapter::new(config)?)),
        "llama" | "meta-llama" | "meta" => Ok(Box::new(LlamaAdapter::new(config)?)),
        // 中文模型：别名对齐 Python agn/adapters/chinese.py 末尾 register 调用
        // （qwen/zhipu/doubao/ernie/kimi/minimax 均无别名，直接注册主名）
        "qwen" => Ok(Box::new(QwenAdapter::new(config)?)),
        "zhipu" => Ok(Box::new(ZhipuAdapter::new(config)?)),
        "doubao" => Ok(Box::new(DoubaoAdapter::new(config)?)),
        "ernie" => Ok(Box::new(ErnieAdapter::new(config)?)),
        "kimi" => Ok(Box::new(KimiAdapter::new(config)?)),
        "minimax" => Ok(Box::new(MiniMaxAdapter::new(config)?)),
        // 阶段 2 适配器占位
        "anthropic" | "runway" | "pika" | "kling" | "stability" | "edge-tts" | "elevenlabs"
        | "cartesia" | "deepgram" | "assemblyai" => Err(AibridgeError::ProviderNotFound {
            provider: format!("{provider}（阶段 2 待实现）"),
        }),
        // 未知 provider
        _ => Err(AibridgeError::provider_not_found(format!(
            "{provider}（未知 provider，支持：{}）",
            KNOWN_PROVIDERS.join(", ")
        ))),
    }
}

/// 检查 provider 是否已被工厂识别（不一定已实现）
pub fn is_known_provider(provider: &str) -> bool {
    KNOWN_PROVIDERS.contains(&provider)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientOptions;

    fn config_for(provider: &str) -> ProviderConfig {
        ProviderConfig::from_options(provider, ClientOptions::builder().api_key("k").build())
    }

    #[test]
    fn create_unknown_provider_returns_error() {
        let result = create_adapter(config_for("nonexistent"));
        assert!(matches!(
            result,
            Err(AibridgeError::ProviderNotFound { .. })
        ));
    }

    #[test]
    fn create_openai_returns_adapter() {
        // 阶段 1：工厂已能构造真实 OpenAiAdapter（仅校验构造成功，不触发 HTTP）
        let adapter = create_adapter(config_for("openai")).expect("工厂应能创建 openai 适配器");
        assert_eq!(adapter.provider_type(), "openai");
    }

    #[test]
    fn create_agnes_returns_adapter() {
        let adapter = create_adapter(config_for("agnes")).expect("工厂应能创建 agnes 适配器");
        assert_eq!(adapter.provider_type(), "agnes");
    }

    #[test]
    fn create_volcengine_cv_returns_adapter() {
        let adapter =
            create_adapter(config_for("volcengine_cv")).expect("工厂应能创建 volcengine_cv 适配器");
        assert_eq!(adapter.provider_type(), "volcengine_cv");
    }

    #[test]
    fn create_gemini_returns_adapter() {
        let adapter = create_adapter(config_for("gemini")).expect("工厂应能创建 gemini 适配器");
        assert_eq!(adapter.provider_type(), "gemini");
    }

    #[test]
    fn create_azure_returns_adapter() {
        // 阶段 2a：工厂已能构造真实 AzureAdapter（仅校验构造成功，不触发 HTTP）
        // Azure 构造需 base_url 或 resource_name + deployment_id，此处用 base_url
        let opts = ClientOptions::builder()
            .api_key("k")
            .base_url("https://example.openai.azure.com/openai/deployments/gpt-4")
            .build();
        let config = ProviderConfig::from_options("azure", opts);
        let adapter = create_adapter(config).expect("工厂应能创建 azure 适配器");
        assert_eq!(adapter.provider_type(), "azure");
    }

    #[test]
    fn create_siliconflow_returns_adapter() {
        let adapter =
            create_adapter(config_for("siliconflow")).expect("工厂应能创建 siliconflow 适配器");
        assert_eq!(adapter.provider_type(), "siliconflow");
    }

    #[test]
    fn create_togetherai_returns_adapter() {
        let adapter =
            create_adapter(config_for("togetherai")).expect("工厂应能创建 togetherai 适配器");
        assert_eq!(adapter.provider_type(), "togetherai");
    }

    #[test]
    fn create_fireworksai_returns_adapter() {
        let adapter =
            create_adapter(config_for("fireworksai")).expect("工厂应能创建 fireworksai 适配器");
        assert_eq!(adapter.provider_type(), "fireworksai");
    }

    #[test]
    fn create_cloudflareai_returns_adapter() {
        // Cloudflare 需 extra.account_id 才能通过构造校验
        let opts = ClientOptions::builder()
            .api_key("k")
            .extra("account_id", "test-account")
            .build();
        let config = ProviderConfig::from_options("cloudflareai", opts);
        let adapter = create_adapter(config).expect("工厂应能创建 cloudflareai 适配器");
        assert_eq!(adapter.provider_type(), "cloudflareai");
    }

    #[test]
    fn create_aggregation_platform_aliases_map_to_main_provider_type() {
        // 别名对齐 Python agn/adapters/aggregation_platforms.py 末尾 register 调用：
        // sf -> siliconflow / together -> togetherai / fireworks -> fireworksai
        // cloudflare & workersai -> cloudflareai
        let sf = create_adapter(config_for("sf")).expect("别名 sf 应映射到 siliconflow");
        assert_eq!(sf.provider_type(), "siliconflow");
        let together =
            create_adapter(config_for("together")).expect("别名 together 应映射到 togetherai");
        assert_eq!(together.provider_type(), "togetherai");
        let fireworks =
            create_adapter(config_for("fireworks")).expect("别名 fireworks 应映射到 fireworksai");
        assert_eq!(fireworks.provider_type(), "fireworksai");

        let opts = ClientOptions::builder()
            .api_key("k")
            .extra("account_id", "test-account")
            .build();
        let cloudflare = create_adapter(ProviderConfig::from_options("cloudflare", opts.clone()))
            .expect("别名 cloudflare 应映射到 cloudflareai");
        assert_eq!(cloudflare.provider_type(), "cloudflareai");
        let workersai = create_adapter(ProviderConfig::from_options("workersai", opts))
            .expect("别名 workersai 应映射到 cloudflareai");
        assert_eq!(workersai.provider_type(), "cloudflareai");
    }

    #[test]
    fn create_grok_returns_adapter() {
        // 阶段 2a additional_models：GrokAdapter 自带 DEFAULT_GROK_BASE_URL 回退，仅需 api_key
        let adapter = create_adapter(config_for("grok")).expect("工厂应能创建 grok 适配器");
        assert_eq!(adapter.provider_type(), "grok");
    }

    #[test]
    fn create_yi_returns_adapter() {
        let adapter = create_adapter(config_for("yi")).expect("工厂应能创建 yi 适配器");
        assert_eq!(adapter.provider_type(), "yi");
    }

    #[test]
    fn create_sensenova_returns_adapter() {
        let adapter =
            create_adapter(config_for("sensenova")).expect("工厂应能创建 sensenova 适配器");
        assert_eq!(adapter.provider_type(), "sensenova");
    }

    #[test]
    fn create_hunyuan_returns_adapter() {
        let adapter = create_adapter(config_for("hunyuan")).expect("工厂应能创建 hunyuan 适配器");
        assert_eq!(adapter.provider_type(), "hunyuan");
    }

    #[test]
    fn create_groq_returns_adapter() {
        let adapter = create_adapter(config_for("groq")).expect("工厂应能创建 groq 适配器");
        assert_eq!(adapter.provider_type(), "groq");
    }

    #[test]
    fn create_deepseek_returns_adapter() {
        let adapter = create_adapter(config_for("deepseek")).expect("工厂应能创建 deepseek 适配器");
        assert_eq!(adapter.provider_type(), "deepseek");
    }

    #[test]
    fn create_stepfun_returns_adapter() {
        let adapter = create_adapter(config_for("stepfun")).expect("工厂应能创建 stepfun 适配器");
        assert_eq!(adapter.provider_type(), "stepfun");
    }

    #[test]
    fn create_mistral_returns_adapter() {
        let adapter = create_adapter(config_for("mistral")).expect("工厂应能创建 mistral 适配器");
        assert_eq!(adapter.provider_type(), "mistral");
    }

    #[test]
    fn create_cohere_returns_adapter() {
        let adapter = create_adapter(config_for("cohere")).expect("工厂应能创建 cohere 适配器");
        assert_eq!(adapter.provider_type(), "cohere");
    }

    #[test]
    fn create_perplexity_returns_adapter() {
        let adapter =
            create_adapter(config_for("perplexity")).expect("工厂应能创建 perplexity 适配器");
        assert_eq!(adapter.provider_type(), "perplexity");
    }

    #[test]
    fn create_ideogram_returns_adapter() {
        // 阶段 2a emerging_models：IdeogramAdapter 自带 DEFAULT_IDEOGRAM_BASE_URL 兜底，仅需 api_key
        let adapter = create_adapter(config_for("ideogram")).expect("工厂应能创建 ideogram 适配器");
        assert_eq!(adapter.provider_type(), "ideogram");
    }

    #[test]
    fn create_luma_returns_adapter() {
        let adapter = create_adapter(config_for("luma")).expect("工厂应能创建 luma 适配器");
        assert_eq!(adapter.provider_type(), "luma");
    }

    #[test]
    fn create_llama_returns_adapter() {
        let adapter = create_adapter(config_for("llama")).expect("工厂应能创建 llama 适配器");
        assert_eq!(adapter.provider_type(), "llama");
    }

    #[test]
    fn create_emerging_models_aliases_map_to_main_provider_type() {
        // 别名对齐 Python agn/adapters/emerging_models.py 末尾 register 调用：
        // ideo -> ideogram / dream-machine & lumalabs -> luma / meta-llama & meta -> llama
        let ideo = create_adapter(config_for("ideo")).expect("别名 ideo 应映射到 ideogram");
        assert_eq!(ideo.provider_type(), "ideogram");
        let dream_machine =
            create_adapter(config_for("dream-machine")).expect("别名 dream-machine 应映射到 luma");
        assert_eq!(dream_machine.provider_type(), "luma");
        let lumalabs = create_adapter(config_for("lumalabs")).expect("别名 lumalabs 应映射到 luma");
        assert_eq!(lumalabs.provider_type(), "luma");
        let meta_llama =
            create_adapter(config_for("meta-llama")).expect("别名 meta-llama 应映射到 llama");
        assert_eq!(meta_llama.provider_type(), "llama");
        let meta = create_adapter(config_for("meta")).expect("别名 meta 应映射到 llama");
        assert_eq!(meta.provider_type(), "llama");
    }

    #[test]
    fn create_qwen_returns_adapter() {
        // 阶段 2a chinese：QwenAdapter 自带 DEFAULT_QWEN_BASE_URL 兜底，仅需 api_key
        let adapter = create_adapter(config_for("qwen")).expect("工厂应能创建 qwen 适配器");
        assert_eq!(adapter.provider_type(), "qwen");
    }

    #[test]
    fn create_zhipu_returns_adapter() {
        let adapter = create_adapter(config_for("zhipu")).expect("工厂应能创建 zhipu 适配器");
        assert_eq!(adapter.provider_type(), "zhipu");
    }

    #[test]
    fn create_doubao_returns_adapter() {
        let adapter = create_adapter(config_for("doubao")).expect("工厂应能创建 doubao 适配器");
        assert_eq!(adapter.provider_type(), "doubao");
    }

    #[test]
    fn create_ernie_returns_adapter() {
        // ErnieAdapter: api_key 不含 ':' 时直接作 access_token，构造无特殊要求
        let adapter = create_adapter(config_for("ernie")).expect("工厂应能创建 ernie 适配器");
        assert_eq!(adapter.provider_type(), "ernie");
    }

    #[test]
    fn create_kimi_returns_adapter() {
        let adapter = create_adapter(config_for("kimi")).expect("工厂应能创建 kimi 适配器");
        assert_eq!(adapter.provider_type(), "kimi");
    }

    #[test]
    fn create_minimax_returns_adapter() {
        let adapter = create_adapter(config_for("minimax")).expect("工厂应能创建 minimax 适配器");
        assert_eq!(adapter.provider_type(), "minimax");
    }

    #[test]
    fn create_additional_models_aliases_map_to_main_provider_type() {
        // 别名对齐 Python agn/adapters/additional_models.py 末尾 register 调用：
        // xaigrok -> grok / lingyiwanwu -> yi / shangtang -> sensenova / tencent_hunyuan -> hunyuan
        let xaigrok = create_adapter(config_for("xaigrok")).expect("别名 xaigrok 应映射到 grok");
        assert_eq!(xaigrok.provider_type(), "grok");
        let lingyiwanwu =
            create_adapter(config_for("lingyiwanwu")).expect("别名 lingyiwanwu 应映射到 yi");
        assert_eq!(lingyiwanwu.provider_type(), "yi");
        let shangtang =
            create_adapter(config_for("shangtang")).expect("别名 shangtang 应映射到 sensenova");
        assert_eq!(shangtang.provider_type(), "sensenova");
        let tencent_hunyuan = create_adapter(config_for("tencent_hunyuan"))
            .expect("别名 tencent_hunyuan 应映射到 hunyuan");
        assert_eq!(tencent_hunyuan.provider_type(), "hunyuan");
    }

    #[test]
    fn create_more_models_aliases_map_to_main_provider_type() {
        // 别名对齐 Python agn/adapters/more_models.py 末尾 register 调用：
        // step -> stepfun（deepseek/mistral/cohere/perplexity 无别名）
        let step = create_adapter(config_for("step")).expect("别名 step 应映射到 stepfun");
        assert_eq!(step.provider_type(), "stepfun");
    }

    #[test]
    fn create_phase2_adapter_returns_phase2_message() {
        let result = create_adapter(config_for("anthropic"));
        if let Err(AibridgeError::ProviderNotFound { provider }) = result {
            assert!(provider.contains("阶段 2"));
        } else {
            panic!("应为 ProviderNotFound");
        }
    }

    #[test]
    fn is_known_provider_recognizes_known() {
        assert!(is_known_provider("echo"));
        assert!(is_known_provider("openai"));
        assert!(is_known_provider("edge-tts"));
        assert!(is_known_provider("assemblyai"));
        // 阶段 2a 已实现 provider 应被识别
        assert!(is_known_provider("azure"));
        assert!(is_known_provider("siliconflow"));
        assert!(is_known_provider("cloudflareai"));
        // 阶段 2a 第二批 additional_models + more_models
        assert!(is_known_provider("grok"));
        assert!(is_known_provider("yi"));
        assert!(is_known_provider("sensenova"));
        assert!(is_known_provider("hunyuan"));
        assert!(is_known_provider("groq"));
        assert!(is_known_provider("deepseek"));
        assert!(is_known_provider("stepfun"));
        assert!(is_known_provider("mistral"));
        assert!(is_known_provider("cohere"));
        assert!(is_known_provider("perplexity"));
        // 阶段 2a 第三批 emerging_models + chinese
        assert!(is_known_provider("ideogram"));
        assert!(is_known_provider("luma"));
        assert!(is_known_provider("llama"));
        assert!(is_known_provider("qwen"));
        assert!(is_known_provider("zhipu"));
        assert!(is_known_provider("doubao"));
        assert!(is_known_provider("ernie"));
        assert!(is_known_provider("kimi"));
        assert!(is_known_provider("minimax"));
    }

    #[test]
    fn is_known_provider_rejects_unknown() {
        assert!(!is_known_provider("nonexistent"));
    }

    #[test]
    fn error_for_unknown_mentions_known_providers() {
        let result = create_adapter(config_for("xxx"));
        if let Err(AibridgeError::ProviderNotFound { provider }) = result {
            assert!(provider.contains("openai"));
        } else {
            panic!("应为 ProviderNotFound");
        }
    }

    #[test]
    fn create_echo_returns_adapter() {
        // echo 免认证，无需 api_key
        let config = ProviderConfig::from_options("echo", ClientOptions::default());
        let result = create_adapter(config);
        assert!(result.is_ok(), "工厂应能创建 echo 适配器");
        let adapter = result.unwrap();
        assert_eq!(adapter.provider_type(), "echo");
        assert_eq!(adapter.provider_name(), "Echo (Mock)");
        assert!(!adapter.requires_api_key());
    }
}
