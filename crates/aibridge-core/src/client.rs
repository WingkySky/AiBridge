//! 统一客户端
//!
//! 提供统一的 API 接口，是用户使用 SDK 的唯一入口。
//! 对应 Python v1 (agn-sdk) 的 `agn/client.py`。
//!
//! 设计要点（与设计文档 5.3 节一致）：
//! - `Client` 持有 `Box<dyn Adapter>`，方法签名改显式 Request struct
//! - 可选参数通过 Request Builder（替代 `**kwargs`）
//! - 不保留 Python 的 `Options` 中间层

use crate::adapter::{create_adapter, Adapter, ChatStream};
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::model::common::{ModelInfo, ModelType, VoiceInfo};
use crate::model::{
    ChatCompletion, ChatRequest, EmbedRequest, EmbeddingResult, ImageRequest, ImageResult,
    SpeechRequest, SpeechResult, TranscribeRequest, TranscriptionResult, VideoRequest, VideoStatus,
    VideoTask,
};

/// 统一客户端
///
/// 对应 Python v1 `Client`。是用户使用 SDK 的唯一入口。
///
/// # 示例
/// ```ignore
/// let client = Client::new("openai", ClientOptions::builder().api_key("sk-xxx").build())?;
/// client.start().await?;
/// let resp = client.chat(
///     ChatRequest::builder("gpt-4o", vec![ChatMessage::user("Hello!")]).build()
/// ).await?;
/// ```
pub struct Client {
    provider_type: String,
    adapter: Box<dyn Adapter>,
}

impl Client {
    /// 创建客户端
    ///
    /// `provider` 为 Provider 类型（如 "agnes"、"openai"）。
    /// `opts` 为连接选项（api_key / base_url / timeout 等）。
    ///
    /// 会自动合并环境变量（`AIBRIDGE_{PROVIDER}_*` / `AGN_{PROVIDER}_*`）。
    pub fn new(provider: &str, opts: ClientOptions) -> Result<Self> {
        let opts = opts.merge_env(provider);
        let config = ProviderConfig::from_options(provider, opts);

        // 校验：requires_api_key 时必须有 api_key
        // 阶段 0.4 无法预知适配器的 requires_api_key（适配器未实现），
        // 暂按"有则需 key"的保守策略：未知 provider 也要求 key。
        // 例外：免认证 provider（如 echo mock 适配器）跳过 key 校验，便于管线验证
        let requires_api_key = !Self::is_free_provider(provider);
        config.validate(requires_api_key)?;

        let adapter = create_adapter(config.clone()).map_err(|e| match e {
            AibridgeError::ProviderNotFound { provider: p } => {
                AibridgeError::ProviderNotFound { provider: p }
            }
            other => other,
        })?;

        Ok(Self {
            provider_type: provider.to_string(),
            adapter,
        })
    }

    /// 判断 provider 是否为免认证（不需要 API Key）
    ///
    /// 阶段 0.6：`echo` 为 mock 适配器，免认证便于五语言管线验证。
    /// 阶段 2c 起将补充 edge-tts 等真实免认证 provider。
    fn is_free_provider(provider: &str) -> bool {
        matches!(provider, "echo")
    }

    /// 启动客户端（初始化适配器）
    pub async fn start(&mut self) -> Result<()> {
        self.adapter.start().await
    }

    /// 关闭客户端（释放资源）
    pub async fn close(&mut self) -> Result<()> {
        self.adapter.close().await
    }

    /// Provider 类型
    pub fn provider_type(&self) -> &str {
        &self.provider_type
    }

    /// 文本对话
    pub async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        self.adapter.chat(req).await
    }

    /// 流式文本对话
    pub async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        self.adapter.chat_stream(req).await
    }

    /// 图像生成
    pub async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
        self.adapter.image_generate(req).await
    }

    /// 创建视频生成任务
    pub async fn video_create(&self, req: VideoRequest) -> Result<VideoTask> {
        self.adapter.video_create(req).await
    }

    /// 查询视频任务状态
    pub async fn video_poll(&self, task_id: &str, model: &str) -> Result<VideoStatus> {
        self.adapter.video_poll(task_id, model).await
    }

    /// 文本嵌入
    pub async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        self.adapter.embed(req).await
    }

    /// 语音转文字
    pub async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscriptionResult> {
        self.adapter.transcribe(req).await
    }

    /// 文字转语音
    pub async fn speech(&self, req: SpeechRequest) -> Result<SpeechResult> {
        self.adapter.speech(req).await
    }

    /// 获取可用模型列表
    pub async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        self.adapter.list_models(filter).await
    }

    /// 获取 Provider 可用音色列表
    pub async fn list_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        self.adapter.list_voices(language).await
    }

    /// 推荐可用音色（按语言/性别过滤）
    pub async fn recommend_voices(
        &self,
        language: Option<&str>,
        gender: Option<&str>,
        limit: usize,
    ) -> Result<Vec<VoiceInfo>> {
        self.adapter.recommend_voices(language, gender, limit).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::chat::{ChatMessage, ChatRequest};
    use std::env;
    use std::sync::Mutex;

    /// 串行化 env 测试（env 变量是进程级共享，并行执行会互相污染）
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn new_rejects_missing_api_key() {
        let result = Client::new("openai", ClientOptions::default());
        // 缺 api_key + 阶段 0 占位：ValidationError 优先（validate 在 create_adapter 前）
        assert!(result.is_err());
    }

    #[test]
    fn new_with_key_reaches_factory_and_returns_provider_not_found() {
        // 阶段 0.4：工厂占位返 ProviderNotFound
        let result = Client::new("openai", ClientOptions::builder().api_key("sk-xxx").build());
        assert!(matches!(
            result,
            Err(AibridgeError::ProviderNotFound { .. })
        ));
    }

    #[test]
    fn new_unknown_provider_returns_provider_not_found() {
        let result = Client::new("nonexistent", ClientOptions::builder().api_key("k").build());
        // validate(true) 通过（key 存在），然后 create_adapter 返 ProviderNotFound
        assert!(matches!(
            result,
            Err(AibridgeError::ProviderNotFound { .. })
        ));
    }

    #[test]
    fn new_with_env_var_api_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::set_var("AIBRIDGE_ENVTEST_API_KEY", "env-key");
        let result = Client::new(
            "envtest",
            ClientOptions::default(), // 无 api_key，从环境变量补
        );
        // envtest 不是已知 provider，validate 通过后工厂返 ProviderNotFound
        assert!(matches!(
            result,
            Err(AibridgeError::ProviderNotFound { .. })
        ));
        env::remove_var("AIBRIDGE_ENVTEST_API_KEY");
    }

    #[test]
    fn new_echo_without_api_key_succeeds() {
        // echo 免认证：无 api_key 也能创建客户端
        let result = Client::new("echo", ClientOptions::default());
        assert!(result.is_ok(), "echo 应免认证创建客户端");
    }

    #[tokio::test]
    async fn echo_client_chat_roundtrip() {
        // 端到端验证：Client → EchoAdapter → chat 回显
        let mut client = Client::new("echo", ClientOptions::default()).unwrap();
        client.start().await.unwrap();
        let req = ChatRequest::builder("echo-chat", vec![ChatMessage::user("hello")]).build();
        let resp = client.chat(req).await.unwrap();
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("hello [echo]")
        );
        client.close().await.unwrap();
    }
}
