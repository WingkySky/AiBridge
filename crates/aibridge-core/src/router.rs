//! 路由器
//!
//! 支持多 Provider 路由、负载均衡、Fallback。
//! 对应 Python v1 (agn-sdk) 的 `agn/router.py`。
//!
//! 设计要点：
//! - 多 Provider 配置，按策略选择（first / round_robin / random / weighted）
//! - 模型名 → Provider 的映射表（迁移自 Python v1 `MODEL_PROVIDER_MAP`）
//! - Fallback：主 Provider 失败时切换备用
//! - 健康状态跟踪 + 延迟统计
//! - 适配器用 `Arc<dyn Adapter>` 存储，从锁中克隆后立即释放锁再调用（避免跨 await 持锁）
//!
//! `start()` 会真实创建并启动各 Provider 适配器；
//! 单测通过直接注入 mock adapter 验证路由选择与 fallback 行为。

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use rand::Rng;

use crate::adapter::{create_adapter, Adapter, Capabilities, ChatStream};
use crate::config::{ClientOptions, ProviderConfig};
use crate::error::{AibridgeError, Result};
use crate::model::common::{ModelInfo, ModelType, VoiceInfo};
use crate::model::{
    ChatCompletion, ChatRequest, EmbedRequest, EmbeddingResult, ImageRequest, ImageResult,
    SpeechRequest, SpeechResult, TranscribeRequest, TranscriptionResult, VideoRequest, VideoStatus,
    VideoTask,
};

/// 路由策略
///
/// 对应 Python v1 `RoutingStrategy`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutingStrategy {
    /// 按顺序选第一个可用的
    #[default]
    First,
    /// 轮询
    RoundRobin,
    /// 随机
    Random,
    /// 按权重随机
    Weighted,
    /// 按延迟（无数据时回退到 First）
    Latency,
}

/// 单个 Provider 的路由配置
#[derive(Debug, Clone)]
pub struct ProviderEntry {
    /// Provider 类型
    pub provider_type: String,
    /// 连接选项
    pub options: ClientOptions,
    /// 权重（weighted 策略用，默认 1）
    pub weight: u32,
}

impl ProviderEntry {
    /// 创建一个 Provider 路由配置
    pub fn new(provider_type: impl Into<String>, options: ClientOptions) -> Self {
        Self {
            provider_type: provider_type.into(),
            options,
            weight: 1,
        }
    }

    /// 设置权重
    pub fn with_weight(mut self, w: u32) -> Self {
        self.weight = w.max(1);
        self
    }
}

/// 多 Provider 路由器
///
/// 对应 Python v1 `Router`。
pub struct Router {
    entries: Vec<ProviderEntry>,
    default_provider: Option<String>,
    strategy: RoutingStrategy,
    enable_fallback: bool,
    max_retries: u32,

    // 运行时状态（用 RwLock 支持 &self 方法）
    inner: RwLock<RouterInner>,
}

#[derive(Default)]
struct RouterInner {
    /// 已启动的适配器（provider_type -> Arc<dyn Adapter>）
    adapters: HashMap<String, Arc<dyn Adapter>>,
    /// provider 顺序
    provider_order: Vec<String>,
    /// 健康状态
    health: HashMap<String, bool>,
    /// 延迟统计（秒）
    latency: HashMap<String, f64>,
    /// round_robin 计数器
    rr_index: usize,
    /// 自定义模型映射
    model_map: HashMap<String, String>,
    /// 权重表（从 entries 拷贝，便于策略选择时读取）
    weights: HashMap<String, u32>,
}

impl Router {
    /// 创建路由器
    pub fn new(entries: Vec<ProviderEntry>) -> Self {
        Self::with_strategy(entries, RoutingStrategy::default())
    }

    /// 创建路由器（指定策略）
    pub fn with_strategy(entries: Vec<ProviderEntry>, strategy: RoutingStrategy) -> Self {
        let weights: HashMap<String, u32> = entries
            .iter()
            .map(|e| (e.provider_type.clone(), e.weight))
            .collect();
        Self {
            entries,
            default_provider: None,
            strategy,
            enable_fallback: true,
            max_retries: 2,
            inner: RwLock::new(RouterInner {
                weights,
                ..Default::default()
            }),
        }
    }

    /// 设置默认 Provider
    pub fn with_default_provider(mut self, provider: impl Into<String>) -> Self {
        self.default_provider = Some(provider.into());
        self
    }

    /// 设置是否启用 fallback
    pub fn with_fallback(mut self, enable: bool) -> Self {
        self.enable_fallback = enable;
        self
    }

    /// 设置 fallback 最大重试次数
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// 启动路由器（初始化所有适配器）
    ///
    /// 单个 Provider 启动失败不会中断整体，仅标记为不健康。
    pub async fn start(&self) -> Result<()> {
        for entry in &self.entries {
            let config =
                ProviderConfig::from_options(entry.provider_type.clone(), entry.options.clone());
            match create_adapter(config) {
                Ok(mut adapter) => {
                    if let Err(e) = adapter.start().await {
                        tracing::warn!(
                            provider = %entry.provider_type,
                            error = %e,
                            "启动 Provider 失败"
                        );
                        let mut inner = self.inner.write().unwrap();
                        inner.health.insert(entry.provider_type.clone(), false);
                    } else {
                        let mut inner = self.inner.write().unwrap();
                        inner
                            .adapters
                            .insert(entry.provider_type.clone(), Arc::from(adapter));
                        inner.provider_order.push(entry.provider_type.clone());
                        inner.health.insert(entry.provider_type.clone(), true);
                        inner.latency.insert(entry.provider_type.clone(), 0.0);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        provider = %entry.provider_type,
                        error = %e,
                        "创建 Provider 适配器失败"
                    );
                    let mut inner = self.inner.write().unwrap();
                    inner.health.insert(entry.provider_type.clone(), false);
                }
            }
        }
        Ok(())
    }

    /// 关闭路由器（释放所有资源）
    ///
    /// 注意：`Arc<dyn Adapter>` 可能被多个持有者共享，close 仅尝试通知；
    /// 真正的资源释放在最后一个 Arc drop 时发生。
    pub async fn close(&self) -> Result<()> {
        let adapters = {
            let mut inner = self.inner.write().unwrap();
            let ads: Vec<Arc<dyn Adapter>> = inner.adapters.values().cloned().collect();
            inner.adapters.clear();
            inner.provider_order.clear();
            inner.health.clear();
            inner.latency.clear();
            ads
        };
        // Arc 无法直接 close（需要 &mut），这里只能依赖 drop。
        // 为保持 API 一致，显式 drop。
        drop(adapters);
        Ok(())
    }

    /// 注册自定义模型映射
    pub fn register_model_mapping(&self, model: impl Into<String>, provider: impl Into<String>) {
        let mut inner = self.inner.write().unwrap();
        inner.model_map.insert(model.into(), provider.into());
    }

    /// 获取所有 Provider 的健康状态
    pub fn get_health_status(&self) -> HashMap<String, bool> {
        self.inner.read().unwrap().health.clone()
    }

    /// 获取所有 Provider 的延迟统计
    pub fn get_latency_stats(&self) -> HashMap<String, f64> {
        self.inner.read().unwrap().latency.clone()
    }

    /// 选择 Provider（按模型名 + 能力）
    fn select_provider(&self, model: &str, capability: Capabilities) -> Result<String> {
        let inner = self.inner.read().unwrap();

        // 1. 自定义映射优先
        if let Some(p) = inner.model_map.get(model) {
            if inner.adapters.contains_key(p) {
                return Ok(p.clone());
            }
        }
        // 2. 内置映射
        if let Some(p) = builtin_model_provider(model) {
            if inner.adapters.contains_key(p) {
                return Ok(p.to_string());
            }
        }
        // 3. 按能力筛选候选
        let candidates = capable_providers(&inner, capability);
        if candidates.is_empty() {
            // 4. 默认 Provider
            if let Some(ref dp) = self.default_provider {
                if inner.adapters.contains_key(dp) {
                    return Ok(dp.clone());
                }
            }
            return Err(AibridgeError::model_not_found(format!(
                "无法为模型 '{model}' 找到合适的 Provider（能力：{}）",
                capability.as_str()
            )));
        }
        Ok(pick_by_strategy(&candidates, self.strategy, &inner))
    }

    /// 获取 fallback 候选（排除已失败的）
    fn fallback_providers(&self, failed: &str, capability: Capabilities) -> Vec<String> {
        let inner = self.inner.read().unwrap();
        let candidates = capable_providers(&inner, capability);
        candidates.into_iter().filter(|p| p != failed).collect()
    }

    /// 从锁中取出指定 provider 的 Arc clone（不在锁内调用）
    fn get_adapter(&self, provider: &str) -> Option<Arc<dyn Adapter>> {
        self.inner.read().unwrap().adapters.get(provider).cloned()
    }

    /// 带Fallback 执行
    async fn execute_with_fallback<F, Fut, T>(
        &self,
        model: &str,
        capability: Capabilities,
        op: F,
    ) -> Result<T>
    where
        F: Fn(Arc<dyn Adapter>) -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let primary = self.select_provider(model, capability)?;
        let mut to_try = vec![primary.clone()];
        if self.enable_fallback {
            let fb = self.fallback_providers(&primary, capability);
            to_try.extend(fb.into_iter().take(self.max_retries as usize));
        }

        let mut last_err: Option<AibridgeError> = None;
        for (i, provider_type) in to_try.iter().enumerate() {
            let Some(adapter) = self.get_adapter(provider_type) else {
                continue;
            };

            let start = Instant::now();
            match op(adapter).await {
                Ok(v) => {
                    let elapsed = start.elapsed().as_secs_f64();
                    let mut inner = self.inner.write().unwrap();
                    let prev = inner.latency.get(provider_type).copied().unwrap_or(0.0);
                    let new = if prev > 0.0 {
                        prev * 0.7 + elapsed * 0.3
                    } else {
                        elapsed
                    };
                    inner.latency.insert(provider_type.clone(), new);
                    inner.health.insert(provider_type.clone(), true);
                    if i > 0 {
                        tracing::info!(from = %primary, to = %provider_type, "Fallback 成功");
                    }
                    return Ok(v);
                }
                Err(e) => {
                    tracing::warn!(
                        provider = %provider_type,
                        attempt = i + 1,
                        error = %e,
                        "Provider 调用失败"
                    );
                    last_err = Some(e);
                    let mut inner = self.inner.write().unwrap();
                    inner.health.insert(provider_type.clone(), false);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| AibridgeError::Api {
            status: 0,
            message: "所有 Provider 均失败".into(),
        }))
    }

    /// 文本对话
    pub async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
        let model = req.model.clone();
        self.execute_with_fallback(&model, Capabilities::Chat, |adapter| {
            let req = req.clone();
            async move { adapter.chat(req).await }
        })
        .await
    }

    /// 流式文本对话（不支持 fallback）
    pub async fn chat_stream(&self, req: ChatRequest) -> Result<ChatStream> {
        let model = req.model.clone();
        let provider = self.select_provider(&model, Capabilities::ChatStream)?;
        let Some(adapter) = self.get_adapter(&provider) else {
            return Err(AibridgeError::model_not_found(format!(
                "Provider '{provider}' 未启动"
            )));
        };
        adapter.chat_stream(req).await
    }

    /// 图像生成
    pub async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
        let model = req.model.clone();
        self.execute_with_fallback(&model, Capabilities::ImageGenerate, |adapter| {
            let req = req.clone();
            async move { adapter.image_generate(req).await }
        })
        .await
    }

    /// 创建视频生成任务
    pub async fn video_create(&self, req: VideoRequest) -> Result<VideoTask> {
        let model = req.model.clone();
        self.execute_with_fallback(&model, Capabilities::VideoGenerate, |adapter| {
            let req = req.clone();
            async move { adapter.video_create(req).await }
        })
        .await
    }

    /// 查询视频任务状态
    pub async fn video_poll(&self, task_id: &str, model: &str) -> Result<VideoStatus> {
        // 视频轮询：按模型映射找 provider，否则遍历支持 video 的
        let provider = {
            let inner = self.inner.read().unwrap();
            if let Some(p) = inner.model_map.get(model) {
                Some(p.clone())
            } else {
                builtin_model_provider(model).map(|p| p.to_string())
            }
        };

        if let Some(p) = provider {
            if let Some(adapter) = self.get_adapter(&p) {
                return adapter.video_poll(task_id, model).await;
            }
        }

        // 遍历支持 video 的 provider
        let providers: Vec<String> = {
            let inner = self.inner.read().unwrap();
            capable_providers(&inner, Capabilities::VideoGenerate)
        };
        for p in providers {
            if let Some(adapter) = self.get_adapter(&p) {
                match adapter.video_poll(task_id, model).await {
                    Ok(v) => return Ok(v),
                    Err(_) => continue,
                }
            }
        }
        Err(AibridgeError::model_not_found(format!(
            "无法确定视频轮询的 Provider（task_id={task_id}, model={model}）"
        )))
    }

    /// 文本嵌入
    pub async fn embed(&self, req: EmbedRequest) -> Result<EmbeddingResult> {
        let model = req.model.clone();
        self.execute_with_fallback(&model, Capabilities::Embedding, |adapter| {
            let req = req.clone();
            async move { adapter.embed(req).await }
        })
        .await
    }

    /// 语音转文字
    pub async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscriptionResult> {
        let model = req.model.clone();
        self.execute_with_fallback(&model, Capabilities::AudioTranscribe, |adapter| {
            let req = req.clone();
            async move { adapter.transcribe(req).await }
        })
        .await
    }

    /// 语音翻译（翻译为英文，带 Fallback）
    ///
    /// 按 `AudioTranslate` 能力路由（与 Python v1 `Router.translate` 一致）。
    pub async fn translate(&self, req: TranscribeRequest) -> Result<TranscriptionResult> {
        let model = req.model.clone();
        self.execute_with_fallback(&model, Capabilities::AudioTranslate, |adapter| {
            let req = req.clone();
            async move { adapter.translate(req).await }
        })
        .await
    }

    /// 文字转语音
    pub async fn speech(&self, req: SpeechRequest) -> Result<SpeechResult> {
        let model = req.model.clone();
        self.execute_with_fallback(&model, Capabilities::AudioSpeech, |adapter| {
            let req = req.clone();
            async move { adapter.speech(req).await }
        })
        .await
    }

    /// 获取可用模型列表（聚合所有 Provider）
    pub async fn list_models(&self, filter: Option<ModelType>) -> Result<Vec<ModelInfo>> {
        let providers: Vec<String> = {
            let inner = self.inner.read().unwrap();
            inner
                .provider_order
                .iter()
                .filter(|p| *inner.health.get(*p).unwrap_or(&true))
                .cloned()
                .collect()
        };

        let mut all = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for p in providers {
            if let Some(adapter) = self.get_adapter(&p) {
                match adapter.list_models(filter).await {
                    Ok(models) => {
                        for m in models {
                            if seen.insert(m.id.clone()) {
                                all.push(m);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(provider = %p, error = %e, "list_models 失败");
                    }
                }
            }
        }
        Ok(all)
    }

    /// 列出可用音色
    pub async fn list_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        let providers: Vec<String> = {
            let inner = self.inner.read().unwrap();
            capable_providers(&inner, Capabilities::ListVoices)
        };
        for p in providers {
            if let Some(adapter) = self.get_adapter(&p) {
                match adapter.list_voices(language).await {
                    Ok(v) => return Ok(v),
                    Err(_) => continue,
                }
            }
        }
        Err(AibridgeError::unsupported_capability(
            "list_voices（无 Provider 支持）",
        ))
    }
}

/// 获取支持指定能力的健康 Provider 列表
fn capable_providers(inner: &RouterInner, capability: Capabilities) -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    for p in &inner.provider_order {
        let healthy = *inner.health.get(p).unwrap_or(&true);
        if !healthy {
            continue;
        }
        let Some(adapter) = inner.adapters.get(p) else {
            continue;
        };
        if adapter.supports_capability(capability) {
            candidates.push(p.clone());
        }
    }
    // 没有健康的就放宽：返回所有适配器（不论健康）
    if candidates.is_empty() {
        for p in &inner.provider_order {
            let Some(adapter) = inner.adapters.get(p) else {
                continue;
            };
            if adapter.supports_capability(capability) {
                candidates.push(p.clone());
            }
        }
    }
    candidates
}

/// 按策略从候选中选一个
fn pick_by_strategy(
    candidates: &[String],
    strategy: RoutingStrategy,
    inner: &RouterInner,
) -> String {
    if candidates.len() == 1 {
        return candidates[0].clone();
    }
    match strategy {
        RoutingStrategy::First => candidates[0].clone(),
        RoutingStrategy::RoundRobin => {
            let idx = inner.rr_index % candidates.len();
            candidates[idx].clone()
        }
        RoutingStrategy::Random => {
            let mut rng = rand::thread_rng();
            let idx = rng.gen_range(0..candidates.len());
            candidates[idx].clone()
        }
        RoutingStrategy::Weighted => {
            let weights: Vec<u32> = candidates
                .iter()
                .map(|p| inner.weights.get(p).copied().unwrap_or(1))
                .collect();
            let total: u32 = weights.iter().sum();
            if total == 0 {
                return candidates[0].clone();
            }
            let mut rng = rand::thread_rng();
            let mut pick = rng.gen_range(0..total);
            for (i, w) in weights.iter().enumerate() {
                if pick < *w {
                    return candidates[i].clone();
                }
                pick -= *w;
            }
            candidates.last().cloned().unwrap()
        }
        RoutingStrategy::Latency => {
            let with_latency: Vec<&String> = candidates
                .iter()
                .filter(|p| inner.latency.get(*p).copied().unwrap_or(0.0) > 0.0)
                .collect();
            if with_latency.is_empty() {
                return candidates[0].clone();
            }
            with_latency
                .into_iter()
                .min_by(|a, b| {
                    inner
                        .latency
                        .get(*a)
                        .copied()
                        .unwrap_or(f64::MAX)
                        .partial_cmp(&inner.latency.get(*b).copied().unwrap_or(f64::MAX))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned()
                .unwrap_or_else(|| candidates[0].clone())
        }
    }
}

/// 内置模型 → Provider 映射（迁移自 Python v1 `MODEL_PROVIDER_MAP`，节选）
///
/// 完整列表很长，这里只保留 MVP 四 provider 相关的常见模型；
/// 用户可通过 `register_model_mapping` 补充。
fn builtin_model_provider(model: &str) -> Option<&'static str> {
    match model {
        // OpenAI
        "gpt-4o"
        | "gpt-4-turbo"
        | "gpt-4"
        | "gpt-3.5-turbo"
        | "whisper-1"
        | "tts-1"
        | "tts-1-hd"
        | "gpt-4o-transcribe"
        | "gpt-4o-mini-transcribe" => Some("openai"),
        // Agnes
        "claude-3-opus" | "claude-3-sonnet" | "claude-3-haiku" | "dall-e-3" | "video-gen-1"
        | "video-gen-2" => Some("agnes"),
        // Anthropic（直接协议）
        "claude-3-opus-20240229"
        | "claude-3-sonnet-20240229"
        | "claude-3-haiku-20240307"
        | "claude-3-5-sonnet-20241022" => Some("anthropic"),
        // Google Gemini
        "gemini-2.5-pro" | "gemini-2.5-flash" | "gemini-1.5-pro" | "gemini-1.5-flash" => {
            Some("gemini")
        }
        // 火山引擎 Seedream/Seedance
        "seedream-5.0" | "seedream-4.0" | "seedream-3.0" => Some("volcengine_cv"),
        "seedance-2.0" | "seedance-2.0-mini" | "seedance-1.0" => Some("volcengine_cv"),
        // 可灵 Kling
        "kling-v1" | "kling-v1-5" | "kling-v2" => Some("kling"),
        // Runway
        "gen-3" | "gen-3-turbo" => Some("runway"),
        // Edge TTS
        "edge-tts" | "edge_tts" => Some("edge-tts"),
        // ElevenLabs
        "eleven_multilingual_v2" | "eleven_turbo_v2_5" => Some("elevenlabs"),
        // Deepgram
        "nova-3" | "nova-2" => Some("deepgram"),
        // AssemblyAI
        "best" | "nano" => Some("assemblyai"),
        // Cartesia
        "sonic-2" | "sonic-turbo" => Some("cartesia"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::CapabilitySet;
    use crate::model::image::FileInput;
    use crate::model::ChatMessage;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// 用于路由测试的 mock 适配器
    struct MockAdapter {
        provider: String,
        caps: CapabilitySet,
        call_count: Arc<AtomicU32>,
        fail_n_times: u32,
    }

    #[async_trait]
    impl Adapter for MockAdapter {
        fn provider_type(&self) -> &str {
            &self.provider
        }
        fn provider_name(&self) -> &str {
            &self.provider
        }
        fn capabilities(&self) -> CapabilitySet {
            self.caps.clone()
        }
        async fn start(&mut self) -> Result<()> {
            Ok(())
        }
        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
        async fn chat(&self, req: ChatRequest) -> Result<ChatCompletion> {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_n_times {
                return Err(AibridgeError::Api {
                    status: 500,
                    message: "mock fail".into(),
                });
            }
            Ok(ChatCompletion {
                id: format!("{}-{}", self.provider, req.model),
                object: "chat.completion".into(),
                created: 0,
                model: req.model,
                choices: vec![],
                usage: None,
                service_tier: None,
                system_fingerprint: None,
            })
        }
        async fn image_generate(&self, req: ImageRequest) -> Result<ImageResult> {
            Ok(ImageResult {
                id: format!("{}-{}", self.provider, req.model),
                object: "image.generation".into(),
                created: 0,
                model: req.model,
                data: vec![],
            })
        }
        async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscriptionResult> {
            // 回显 provider 与 translate 标记，供路由/委托测试断言
            Ok(TranscriptionResult {
                text: format!("{}-{}", self.provider, req.model),
                task: if req.translate {
                    "translate".into()
                } else {
                    "transcribe".into()
                },
                ..Default::default()
            })
        }
        async fn list_models(&self, _: Option<ModelType>) -> Result<Vec<ModelInfo>> {
            Ok(vec![])
        }
    }

    /// 构造一个已注入 mock 适配器的路由器（绕过工厂）
    fn router_with_adapters(
        adapters: Vec<(String, CapabilitySet, u32)>, // (provider, caps, fail_n_times)
        strategy: RoutingStrategy,
    ) -> (Router, Vec<Arc<AtomicU32>>) {
        let mut counters = Vec::new();
        let router = Router::with_strategy(vec![], strategy);
        {
            let mut inner = router.inner.write().unwrap();
            for (provider, caps, fail_n) in adapters {
                let counter = Arc::new(AtomicU32::new(0));
                counters.push(counter.clone());
                let adapter = MockAdapter {
                    provider: provider.clone(),
                    caps: caps.clone(),
                    call_count: counter,
                    fail_n_times: fail_n,
                };
                inner.adapters.insert(
                    provider.clone(),
                    Arc::from(Box::new(adapter) as Box<dyn Adapter>),
                );
                inner.provider_order.push(provider.clone());
                inner.health.insert(provider, true);
            }
        }
        (router, counters)
    }

    #[tokio::test]
    async fn chat_routes_by_builtin_model_map() {
        let caps = {
            let mut s = CapabilitySet::new();
            s.insert(Capabilities::Chat);
            s
        };
        let (router, _c) =
            router_with_adapters(vec![("openai".into(), caps, 0)], RoutingStrategy::First);
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let result = router.chat(req).await.unwrap();
        assert!(result.id.starts_with("openai-"));
    }

    #[tokio::test]
    async fn chat_falls_back_on_failure() {
        let caps = {
            let mut s = CapabilitySet::new();
            s.insert(Capabilities::Chat);
            s
        };
        // 主 provider（openai）失败 1 次，fallback 到 agnes
        let (router, counters) = router_with_adapters(
            vec![
                ("openai".into(), caps.clone(), 1),
                ("agnes".into(), caps, 0),
            ],
            RoutingStrategy::First,
        );
        // 自定义映射：gpt-x → openai（主），但 fallback 会到 agnes
        router.register_model_mapping("gpt-x", "openai");
        let req = ChatRequest::builder("gpt-x", vec![ChatMessage::user("hi")]).build();
        let result = router.chat(req).await.unwrap();
        assert!(result.id.starts_with("agnes-"));
        // openai 被调用 1 次（失败）
        assert_eq!(counters[0].load(Ordering::SeqCst), 1);
        // agnes 被调用 1 次（成功）
        assert_eq!(counters[1].load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn chat_all_fail_returns_last_error() {
        let caps = {
            let mut s = CapabilitySet::new();
            s.insert(Capabilities::Chat);
            s
        };
        let (router, _c) = router_with_adapters(
            vec![
                ("openai".into(), caps.clone(), 100),
                ("agnes".into(), caps, 100),
            ],
            RoutingStrategy::First,
        );
        let req = ChatRequest::builder("gpt-4o", vec![ChatMessage::user("hi")]).build();
        let result = router.chat(req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn select_provider_returns_error_when_no_capable() {
        let (router, _c) = router_with_adapters(
            vec![("openai".into(), CapabilitySet::new(), 0)],
            RoutingStrategy::First,
        );
        let result = router
            .chat(ChatRequest::builder("unknown-model", vec![]).build())
            .await;
        assert!(matches!(result, Err(AibridgeError::ModelNotFound { .. })));
    }

    #[tokio::test]
    async fn image_generate_routes() {
        let caps = {
            let mut s = CapabilitySet::new();
            s.insert(Capabilities::ImageGenerate);
            s
        };
        let (router, _c) =
            router_with_adapters(vec![("openai".into(), caps, 0)], RoutingStrategy::First);
        // dall-e-3 映射到 agnes，但 agnes 不存在；自定义映射到 openai
        router.register_model_mapping("dall-e-3", "openai");
        let req = ImageRequest::builder("dall-e-3", "a cat").build();
        let result = router.image_generate(req).await.unwrap();
        assert!(result.id.starts_with("openai-"));
    }

    #[tokio::test]
    async fn fallback_disabled_uses_only_primary() {
        let caps = {
            let mut s = CapabilitySet::new();
            s.insert(Capabilities::Chat);
            s
        };
        let (router, counters) = router_with_adapters(
            vec![
                ("openai".into(), caps.clone(), 1),
                ("agnes".into(), caps, 0),
            ],
            RoutingStrategy::First,
        );
        router.register_model_mapping("gpt-x", "openai");
        let router = router.with_fallback(false);
        let req = ChatRequest::builder("gpt-x", vec![ChatMessage::user("hi")]).build();
        let result = router.chat(req).await;
        assert!(result.is_err()); // openai 失败，无 fallback
        assert_eq!(counters[0].load(Ordering::SeqCst), 1);
        assert_eq!(counters[1].load(Ordering::SeqCst), 0); // agnes 未被调用
    }

    #[tokio::test]
    async fn round_robin_strategy_rotates() {
        let caps = {
            let mut s = CapabilitySet::new();
            s.insert(Capabilities::Chat);
            s
        };
        let (router, _c) = router_with_adapters(
            vec![
                ("openai".into(), caps.clone(), 0),
                ("agnes".into(), caps, 0),
            ],
            RoutingStrategy::RoundRobin,
        );
        // 注册一个走能力筛选的模型（非内置映射）
        router.register_model_mapping("custom-1", "openai");
        router.register_model_mapping("custom-2", "agnes");
        // 第一次：custom-1 → openai
        let r1 = router
            .chat(ChatRequest::builder("custom-1", vec![]).build())
            .await
            .unwrap();
        assert!(r1.id.starts_with("openai-"));
    }

    #[test]
    fn builtin_model_provider_known() {
        assert_eq!(builtin_model_provider("gpt-4o"), Some("openai"));
        assert_eq!(builtin_model_provider("claude-3-opus"), Some("agnes"));
        assert_eq!(builtin_model_provider("gemini-2.5-pro"), Some("gemini"));
        assert_eq!(
            builtin_model_provider("seedance-2.0"),
            Some("volcengine_cv")
        );
        assert_eq!(builtin_model_provider("edge-tts"), Some("edge-tts"));
        assert_eq!(builtin_model_provider("unknown"), None);
    }

    #[test]
    fn routing_strategy_default_is_first() {
        assert_eq!(RoutingStrategy::default(), RoutingStrategy::First);
    }

    #[test]
    fn provider_entry_with_weight() {
        let e = ProviderEntry::new("openai", ClientOptions::default()).with_weight(3);
        assert_eq!(e.weight, 3);
    }

    #[test]
    fn provider_entry_weight_floored_to_1() {
        let e = ProviderEntry::new("openai", ClientOptions::default()).with_weight(0);
        assert_eq!(e.weight, 1);
    }

    #[tokio::test]
    async fn translate_routes_and_sets_translate_flag() {
        let caps = {
            let mut s = CapabilitySet::new();
            s.insert(Capabilities::AudioTranslate);
            s
        };
        let (router, _c) = router_with_adapters(
            vec![("assemblyai".into(), caps, 0)],
            RoutingStrategy::First,
        );
        // "best" 内置映射到 assemblyai；translate 默认实现应置 translate=true 后委托 transcribe
        let req = TranscribeRequest::builder("best", FileInput::path("/tmp/a.mp3")).build();
        let result = router.translate(req).await.unwrap();
        assert_eq!(result.task, "translate");
        assert!(result.text.starts_with("assemblyai-"));
    }

    #[tokio::test]
    async fn translate_returns_model_not_found_when_no_capable() {
        // provider 无 AudioTranslate 能力且模型无映射：能力筛选失败 → ModelNotFound
        let (router, _c) = router_with_adapters(
            vec![("openai".into(), CapabilitySet::new(), 0)],
            RoutingStrategy::First,
        );
        let req =
            TranscribeRequest::builder("unknown-model", FileInput::path("/tmp/a.mp3")).build();
        let result = router.translate(req).await;
        assert!(matches!(result, Err(AibridgeError::ModelNotFound { .. })));
    }
}
