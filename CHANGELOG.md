# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - 2026-08-27

### Added

**Agnes Video 2.5 / 2.5-Flash 双契约接入：**
- 新模型接入：`agnes-video-2.5` / `agnes-video-2.5-flash`（OpenAI Videos 兼容新契约），同批注册 `agnes-video-v2.0`（旧契约）。新旧契约在 `AgnesAdapter` 内部按模型名分流，对外统一 `VideoRequest` 接口不变
- `VideoRequest` 新增 `reference_audios` 字段（承载 2.5 音频参考能力）+ 对应 builder 方法
- Python 绑定 `video_create` 签名补齐 7 个统一参数：`duration` / `resolution` / `aspect_ratio` / `reference_videos` / `reference_audios` / `first_frame` / `last_frame`（全部带默认值，向后兼容）
- Python 绑定 `mode` 参数新增 `"video2video"` 取值（此前会静默回退 text2video）
- 参数前置校验：Flash 不支持视频参考、参考图 ≤ 5 张、seconds 4-12、aspect_ratio 白名单（不符直接返回 `Validation` 错误，不发请求）

### Fixed

- `DEFAULT_AGNES_BASE_URL` 修正为官方统一域名 `https://apihub.agnes-ai.com/v1`（旧域名 `api.agnes.ai` 已废弃）
- Agnes 视频轮询对 2.5 家族默认走 `/agnesapi?video_id=&model_name=` 通道，成功地址解析增加 `metadata.url` 回退
- Agnes 视频任务 ID 解析改为 `video_id` 优先（2.5 轮询以 `video_id` 为查询键）

### Tests

- `aibridge-core`：agnes 适配器新增 20+ 个 2.5 单元/集成测试（双契约分流、请求体构建、参数校验、agnesapi 轮询通道、metadata.url 解析），全量 1530 个测试通过
- Python：新增 `tests/test_video_unified_params.py`（4 个端到端用例，本地 mock 服务器实测统一参数翻译、轮询通道、Flash 校验短路、V2.0 回归）

## [2.0.0] - 2026-07-26

### 🎉 重大变更 — Rust 重写（agn-sdk v1 → aibridge v2）

AIBridge v2 是用 Rust 从零重写的多模态 AI 统一接口 SDK，替代 Python v1。
五语言绑定（Python / Node.js / Go / JVM / .NET），38+ provider，全能力覆盖。

### Added

**Core 层：**
- 统一 `Client` + `Adapter` trait 架构，38 个真实 provider + echo mock
- 完整能力：chat / chat_stream / image_generate / video_create / video_poll / embed / transcribe / translate / speech / list_models / list_voices / recommend_voices
- Router：五种路由策略（first/round_robin/random/weighted/latency）+ Fallback + EMA 延迟统计 + 健康跟踪 + 自定义模型映射
- Anthropic native 协议 ToolCall（含 tool_choice 映射、tool_use 响应解析）
- 指数退避重试（RetryPolicy + retry_with），rate-limit-aware retry_after 处理
- SSE 流式解析（OpenAI 兼容 + Anthropic 独立实现）
- HTTP/2 + 连接池（reqwest）
- 统一错误码（11 种 AibridgeError 变体）
- Voice 音色候选列表自动 fallback 降级

**模型字段补齐：**
- ChatRequest：repetition_penalty / min_p / thinking_budget / stream_options / web_search / search_recency_filter / search_domain_filter
- VideoRequest：reference_videos / keyframes / style
- ImageRequest：sampler / scheduler / reference_strength / negative_prompts
- EmbedRequest：dimensions / encoding_format / user
- ParameterMapping：value_map 支持值映射与对象展开（Anthropic thinking 映射）

**Provider 迁移（38 个真实 + 1 mock）：**
- MVP：openai / agnes / volcengine_cv / gemini
- OpenAI 兼容族：azure / siliconflow / togetherai / fireworksai / cloudflareai / grok / yi / sensenova / hunyuan / groq / deepseek / stepfun / mistral / cohere / perplexity / ideogram / luma / llama / qwen / zhipu / doubao / ernie / kimi / minimax
- 独立协议：anthropic / stability / runway / pika / kling
- 音频 TTS/ASR：edge-tts(免认证) / elevenlabs / cartesia / deepgram / assemblyai
- Mock：echo（全能力回显，管线验证用）

**Python 绑定（PyO3）：**
- 全能力封装：chat / chat_stream / speech / image_generate / video_create / video_poll / embed / transcribe / translate / list_models / list_voices / recommend_voices
- Router pyclass（完整实现，含 Fallback）
- 异步上下文管理器（`__aenter__` / `__aexit__`）
- 原生 asyncio 协程 + AsyncIterator 流式（不阻塞事件循环）

**Node.js 绑定（napi-rs）：**
- 全能力封装（直连 core，绕过 FFI）
- mpsc channel 桥接流式迭代器

**FFI 层（C ABI）：**
- 21 个导出函数：client_new/start/destroy、chat/chat_stream/speech/image/video/embed/transcribe/translate/list_models/list_voices/recommend_voices、stream_next/stream_destroy、last_error/string_free/bytes_free
- panic 安全包装（catch_unwind）
- 线程局部错误 JSON

**五语言绑定：**
- Go：CGO wrapper + goroutine streaming
- JVM：JNA wrapper + Iterator + Flow.Publisher (reactive streams)
- .NET：P/Invoke + SafeHandle RAII + IAsyncEnumerable streaming
- 所有绑定均通过 echo adapter 全能力验证

**文档：**
- 中英文 README
- mkdocs 文档网站 + GitHub Pages 自动部署
- Python v1 → v2 迁移指南
- 设计文档（系统架构、API 规范、五语言绑定方案）

### Changed

- 架构从 Python monolith 拆分为 Rust workspace（core + ffi + python + node）
- 请求参数从 Python `**kwargs` 改为 Request struct + Builder pattern
- 错误体系：AGNError → AibridgeError（11 种子类，统一错误码）
- 版本从 v1.3.3 跳至 v2.0.0

### Fixed

- .NET 绑定：[JsonProperty] → [JsonPropertyName]（presence_penalty / frequency_penalty）
- .NET 绑定：元组字段名 lastError → lastErr
- .NET 绑定：TargetFramework net8.0 → net10.0

### Removed

- Python v1 代码归档至 `agn/` 目录，不再维护

## [1.3.3] - 2026-06-27

### Changed

- VolcengineCVAdapter 视频创建参数传递方式改为官方推荐的 body 直传（强校验）
  - 旧方式：参数拼入 text 末尾的 `--flag value`（弱校验，参数错误会被静默忽略）
  - 新方式：参数直接放 request body（强校验，参数错误会明确报错）
  - 参考官方文档 https://www.volcengine.com/docs/82379/1520757 新方式说明

### Added

- VolcengineCVAdapter 视频创建新增高级参数支持（kwargs）：
  - `generate_audio`：是否生成音频（仅 Seedance 2.0/1.5 Pro 支持）
  - `service_tier`：服务等级 "default"(在线) / "flex"(离线，更便宜)
  - `priority`：请求优先级 0-9（仅 Seedance 2.0）
  - `draft`：是否开启样片模式（仅 Seedance 1.5 Pro）
- 参数名对齐方舟规范：`aspect_ratio` → body 的 `ratio` 字段，`camerafixed` → body 的 `camera_fixed` 字段

## [1.3.2] - 2026-06-27

### Fixed

- VolcengineCVAdapter 图像生成 size 参数不符合方舟规范：方舟 Seedream 要求总像素 ∈ [3686400, 16777216]、宽高比 ∈ [1/16, 16]，OpenAI 风格的小尺寸（如 `1024x1024` / `1280x720`）会触发方舟 503 错误。新增 `_normalize_image_size()` 静态方法，将不合法尺寸按最接近宽高比映射到 2K 推荐档（官方推荐尺寸表），已合法尺寸和枚举值（`2K`/`3K`/`4K`）原样透传

### Changed

- 同步更新模块 docstring：视频端点 `/videos/generations` → `/contents/generations/tasks`（1.3.1 改动遗漏的文档同步）

## [1.3.1] - 2026-06-27

### Fixed

- VolcengineCVAdapter 模型 ID 不符合方舟规范：`list_models` 原硬编码 `seedream-5.0` / `seedance-2.0` 等模型系列名（方舟 API 不识别），改为实时拉取 `GET /models`，返回 `doubao-seedream-4-0-250828` / `doubao-seedance-1-0-pro-250528` 等方舟规范格式 ID
- VolcengineCVAdapter 视频创建端点错误：`POST /videos/generations` → `POST /contents/generations/tasks`（方舟 Video Generation API 官方端点），原端点无法调通
- VolcengineCVAdapter 视频查询端点错误：`GET /videos/generations/{task_id}` → `GET /contents/generations/tasks/{task_id}`
- VolcengineCVAdapter 视频创建请求体结构错误：`{"model","prompt"}` → `{"model","content":[{"type":"text","text":"提示词 --flag value"}]}`，对齐方舟 content 数组结构
- VolcengineCVAdapter 视频参数传递方式：独立字段（duration/aspect_ratio/resolution/seed）转换为方舟 text flag（`--dur`/`--rt`/`--rs`/`--seed`/`--wm`/`--cf`），bool 值统一转小写
- VolcengineCVAdapter `_parse_video_status` 增加 `content.video_url` 响应路径解析（方舟查询任务响应结构）

## [1.3.0] - 2026-06-27

### Fixed

- AgnesAdapter 视频创建端点：`/videos/generations` → `/videos`，对齐 Agnes Video V2.0 官方协议
- AgnesAdapter 视频轮询端点：优先使用 `poll_url`（Agnes 特有的 /agnesapi 轮询通道）+ `/videos/{task_id}` 双路径回退；4xx 直接抛出、网络错误回退
- AgnesAdapter 视频轮询连接池 Bug：原实现 `client = self._get_client()` 后立即被新 `AsyncHttpClient` 覆盖，且每次轮询都新建/关闭客户端，违背连接池复用原则。现统一复用 `self._http_client`
- AgnesAdapter `list_models` 硬编码 9 个假模型 ID（如 `video-gen-1`、`dall-e-3` 等）改为实时调用 `GET /models` 拉取

### Added

- `VideoGenerationOptions` 新增 `duration` / `aspect_ratio` / `resolution` 三个通用字段，便于火山引擎 Seedance、Agnes 等以时长/宽高比/分辨率档位为参数的模型通过统一 `options` 调用
- `BaseAdapter._infer_type(model_id)` 静态方法：根据模型 ID 推断 chat / image / video / audio 类型
- `BaseAdapter._parse_models_response(data, provider, model_type, items_key)` 静态方法：统一解析 OpenAI 兼容的 `{"data":[{"id":...}]}` 响应为 `ModelInfo` 列表，兼容 `display_name` 字段
- 全量改造硬编码模型列表为实时拉取：OpenAI / Anthropic / Gemini / Azure / Stability / Qwen / Zhipu / Doubao / Kimi / DeepSeek / StepFun / Mistral / Cohere / Perplexity / Grok / Yi / SenseNova / Hunyuan / Groq / SiliconFlow / TogetherAI / FireworksAI / CloudflareAI / Llama / ElevenLabs / Cartesia 共 26 个适配器

### Changed

- 无标准 `/models` 端点的 Provider（Runway / Pika / Kling / Volcengine CV / Ideogram / Luma / Ernie / MiniMax / Deepgram / AssemblyAI / EdgeTTS）保留硬编码列表，并在源码加 `# NOTE` 注释说明原因

## [1.2.0] - 2026-06-26

### Added

- TTS 音色健康检查 / 推荐：`Client.list_voices()` 和 `Client.recommend_voices(language, gender, limit)` 统一入口，业务层无需自己维护"可用声音池"
- `BaseAdapter.list_voices` / `recommend_voices` 默认实现（不支持的 Provider 抛 `UnsupportedCapabilityError`）
- `EdgeTTSAdapter.list_voices` 带类级缓存（`_voices_cache`），避免每次空音频都网络查询
- `EdgeTTSAdapter.recommend_voices` 覆盖实现，按语言/性别过滤可用音色
- TTS 音色自动降级：`speech(voice=["XiaoxiaoNeural", "XiaoyiNeural"])` 支持候选列表，第一个失败自动切换到下一个
- 空音频语义化异常：`EdgeTTSAdapter` 空音频时主动查询 `list_voices` 区分语义
  - voice 仍在线 → 抛 `ServiceUnavailableError`（服务端临时问题，可重试）
  - voice 已下线 → 抛 `VoiceNotAvailableError`（重试无意义，应换音色）
- 新增 `VoiceNotAvailableError` / `ServiceUnavailableError` 标准错误类型

### Changed

- `BaseAdapter.speech` 及所有子类 `speech` 方法的 `voice` 参数从 `str` 改为 `str | list[str]`，支持候选列表降级
- `Client.speech` 的 `voice` 参数同步改为 `str | list[str]`
- 非 EdgeTTS 适配器（ElevenLabs / Cartesia / Azure / OpenAI 兼容）收到 voice 列表时取第一个元素（不实现 fallback，但签名兼容）

### Fixed

- 解决 edge-tts 声音被微软下线时上层只能靠文件大小事后发现的空缺：现在 SDK 主动判别 voice 可用性并给出语义化异常，上层可区分"该换声音"还是"该等一下重试"

## [1.1.1] - 2026-06-26

### Fixed

- EdgeTTSAdapter 空音频检测：edge-tts 服务端未返回音频时不再静默返回空 `SpeechResult`，改为抛出 `APIError`（code=`NO_AUDIO_RECEIVED`），调用方可直接捕获异常而非靠文件大小事后发现

## [1.1.0] - 2026-06-26

### Added

- 支持免费 Provider 免认证使用：`BaseAdapter` 新增 `requires_api_key` 类变量，免费 Provider（如 Edge TTS）设为 `False` 即可不传 API Key
- 新增免费 Provider 场景测试（`test_client_init_free_provider_without_api_key` 等）

### Changed

- `ProviderConfig.api_key` 从 `str` 改为 `str | None`，免费 Provider 可不传
- `Client` API Key 校验逻辑改为条件式：仅 `requires_api_key=True` 时检查
- 所有适配器 `__init__` 的 `self.api_key = config.api_key` 改为 `or ""` 兜底为 `str`（36 处）

### Fixed

- 修复设计缺陷：原先所有 Provider 都被强制要求 `api_key`，导致 Edge TTS 等免费模型无法正常使用

## [1.0.0] - 2026-06-25

### Added

- 多模型统一接口 SDK 首个正式版本
- 统一 API：chat / image_generate / video_create / transcribe / speech / embed
- 分层架构：API 层 / 路由器层 / 适配器层 / 核心层 / 数据模型层
- 支持 Provider：Agnes / OpenAI / Azure / Gemini / Anthropic / Runway / Pika / Kling / Stability / 中文模型聚合平台 / Edge TTS / ElevenLabs / Cartesia / Deepgram / AssemblyAI / Volcengine 等
- 生产级特性：异步优先、重试机制、错误映射、参数归一化、负载均衡、Fallback
- 643 单元测试，mypy strict 0 错误
