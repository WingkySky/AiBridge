//! 具体 Provider 适配器
//!
//! 对应 Python v1 (agn-sdk) 的 `agn/adapters/*.py`（14 个适配器）。
//!
//! **阶段 0.4**：本模块仅占位，具体适配器在阶段 1 起填充：
//! - 阶段 1（MVP）：openai / agnes / volcengine_cv / gemini
//! - 阶段 2a（兼容族）：azure / aggregation_platforms / additional_models / more_models / emerging_models / chinese
//! - 阶段 2b（独立协议）：anthropic / stability / runway / pika / kling
//! - 阶段 2c（音频）：edge-tts / elevenlabs / cartesia / deepgram / assemblyai
//!
//! 各适配器实现 `adapter::Adapter` trait，由 `adapter::create_adapter` 工厂分发。

/// Echo（Mock）适配器：阶段 0.6 五语言管线验证用，不调网络返固定/回显响应
pub mod echo;

/// OpenAI 兼容协议适配器地基：阶段 1.0 实现，为 openai/agnes 等子适配器提供共享基础
pub mod openai_compat;

/// Agnes 适配器：阶段 1 MVP，OpenAI 兼容协议的子适配器
pub mod agnes;

/// Gemini 适配器：阶段 1 MVP，Google Gemini 独立协议
pub mod gemini;

/// OpenAI 适配器：阶段 1 MVP，OpenAI 官方协议
pub mod openai;

/// 火山引擎 CV 适配器：阶段 1 MVP，火山引擎视觉/视频生成协议
pub mod volcengine_cv;

/// Azure OpenAI 适配器：阶段 2a，OpenAI 兼容协议的子适配器（Azure 部署）
pub mod azure;

/// 聚合平台适配器：阶段 2a，含 SiliconFlow/TogetherAI/FireworksAI/CloudflareAI 四个 OpenAI 兼容子适配器
pub mod aggregation_platforms;

/// 扩展模型适配器：阶段 2a，含 Grok/Yi/SenseNova/Hunyuan/Groq 五个 OpenAI 兼容子适配器
pub mod additional_models;

/// 更多模型适配器：阶段 2a，含 DeepSeek/StepFun/Mistral/Cohere/Perplexity 五个 OpenAI 兼容子适配器
pub mod more_models;

/// 新兴模型适配器：阶段 2a，含 Ideogram/Luma/Llama 三个子适配器
pub mod emerging_models;

/// 中文模型适配器：阶段 2a，含 Qwen/Zhipu/Doubao/Ernie/Kimi/MiniMax 六个 OpenAI 兼容子适配器
pub mod chinese;

/// Anthropic Claude 适配器：阶段 2b 独立协议，文本对话/流式/多模态
pub mod anthropic;

/// Stability AI 适配器：阶段 2b 独立协议，文生图/图生图
pub mod stability;

/// Runway 适配器：阶段 2b 独立协议，视频生成（文生视频/图生视频/任务轮询）
pub mod runway;

/// Pika 适配器：阶段 2b 独立协议，视频生成（文生视频/图生视频/任务轮询）
pub mod pika;

/// Kling 适配器：阶段 2b 独立协议，可灵视频生成（文生视频/图生视频/任务轮询）
pub mod kling;

/// Edge TTS 适配器：阶段 2c 音频，免费文字转语音（免认证，WebSocket 协议）
pub mod edge_tts;

/// ElevenLabs 适配器：阶段 2c 音频，ElevenLabs TTS 文字转语音（高质量音色/多语种/克隆）
pub mod elevenlabs;

/// Cartesia 适配器：阶段 2c 音频，Cartesia Sonic TTS 文字转语音（低延迟流式）
pub mod cartesia;
