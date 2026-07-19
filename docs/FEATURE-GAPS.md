# AiBridge v2 功能差距与下一步开发计划

> 生成日期：2026-07-18
> 对比基准：旧版 Python agn-sdk (`agn/` 目录) vs 新版 Rust AiBridge (`crates/aibridge-core/`)
> 当前状态：阶段 0-2 完成（38 个真实 provider + echo mock，五语言绑定管线打通）
> 测试：aibridge-core 1449 单测 + aibridge-ffi 39 单测全通过

---

## 一、已完成迁移功能清单

### Provider（38 个真实 + 1 个 mock echo）

| 类别 | Provider |
|------|----------|
| MVP 核心 | openai, agnes, volcengine_cv, gemini |
| OpenAI 兼容族 | azure, siliconflow/sf, togetherai/together, fireworksai/fireworks, cloudflareai/cloudflare/workersai, grok/xaigrok, yi/lingyiwanwu, sensenova/shangtang, hunyuan/tencent_hunyuan, groq, deepseek, stepfun/step, mistral, cohere, perplexity, ideogram/ideo, luma/dream-machine/lumalabs, llama/meta-llama/meta, qwen, zhipu, doubao, ernie, kimi, minimax |
| 独立协议 | anthropic, stability, runway, pika, kling |
| 音频 TTS/ASR | edge-tts/edge_tts/edge（免认证 WebSocket）, elevenlabs/eleven/11labs, cartesia/sonic（流式 Sonic）, deepgram/dg, assemblyai/assembly/aai |
| Mock | echo（免认证，全能力回显，管线验证用） |

### 核心能力完成度

| 能力 | Core | Python 绑定 | Node 绑定 | FFI（Go/JVM/.NET）|
|------|------|------------|-----------|-------------------|
| chat / chat_stream | ✅ | ✅ | ✅ | ✅ chat / ✅ chat_stream / ❌ speech 以外能力 |
| image_generate | ✅ | ✅ | ❌ | ❌ |
| video_create / video_poll | ✅ | ✅ | ❌ | ❌ |
| embed | ✅ | ✅ | ❌ | ❌ |
| transcribe | ✅ | ✅ | ❌ | ❌ |
| speech | ✅ | ✅ | ✅ | ✅ |
| list_models | ✅ | ✅ | ❌ | ❌ |
| list_voices / recommend_voices | ✅（core 层） | ❌ **未暴露** | ❌ | ❌ |
| translate（语音翻译） | ⚠️ 只有 flag | ❌ | ❌ | ❌ |
| Router（多 Provider 路由） | ✅（core 层完整） | ❌ **未暴露** | ❌ | ❌ |

### Core 层已实现的高级特性
- 工具调用（ToolDefinition / ToolCall / ToolChoice / parallel_tool_calls）
- 多模态 Vision（ContentPart::ImageUrl + detail 级别）
- 推理/思考模式（reasoning_effort，Anthropic thinking 特殊处理）
- 结构化输出（ResponseFormat::Text/JsonObject/JsonSchema）
- 参数透传（extra: HashMap<String, Value>）
- Voice 音色候选列表自动 fallback 降级
- 免认证 Provider（echo / edge-tts 系列别名）
- 统一错误码（11 种 AibridgeError 变体，含 retry_after 提取）
- 指数退避重试（RetryPolicy，max_attempts=3，封顶 60s）
- Router：五种路由策略（first/round_robin/random/weighted/latency）+ Fallback + EMA 延迟统计 + 健康跟踪 + 自定义模型映射
- HTTP/2 + 连接池（reqwest）
- SSE 流式解析（OpenAI 兼容 + Anthropic 独立实现）

---

## 二、功能差距清单（按优先级排序）

### P0 — 缺失/不完整（影响功能完整性）

#### 1. Client/Adapter 缺少独立 `translate()` 方法
- **现状**：`TranscribeRequest` 有 `translate: bool` 字段，但：
  - Adapter trait 上没有独立的 `translate()` 方法
  - Client 上没有 `translate()` 入口
  - Deepgram adapter 注释说明"translate 未实现（Python v1 也未实现）"
- **旧版**：`client.translate(model, file, ...)` 是独立方法，底层和 transcribe 共用接口
- **影响**：语音翻译功能不可用；用户无法通过统一 API 翻译音频为英文
- **涉及文件**：
  - `crates/aibridge-core/src/client.rs` — 加 `translate(req)` 方法
  - `crates/aibridge-core/src/adapter/base.rs` — Adapter trait 加 `translate()`（默认实现调 transcribe 并设 translate flag，或返 UnsupportedCapability）
  - `crates/aibridge-python/src/lib.rs` — 加 `translate` pyfunction
  - 支持 translate 的 adapter（deepgram/assemblyai）需覆盖实现

#### 2. ChatRequest 缺少字段
| 缺失字段 | 类型 | 说明 |
|----------|------|------|
| `repetition_penalty` | `Option<f64>` | 重复惩罚（开源模型常用，范围 >=0） |
| `min_p` | `Option<f64>` | Min-P 采样（0-1） |
| `thinking_budget` | `Option<u32>` | 独立思考 token 预算（和 reasoning_effort 互补，Anthropic 直接支持） |
| `stream_options` | `Option<HashMap>` | 流式选项（如 `include_usage` 在末尾块返回 token 用量统计） |
| `web_search` | `Option<bool>` | 是否启用联网搜索（旧版 ChatOptions 有布尔开关，Rust 只有 `ToolDefinition::web_search()` 工具方式） |
| `search_recency_filter` | `Option<String>` | 搜索时间过滤（day/week/month/year） |
| `search_domain_filter` | `Option<Vec<String>>` | 搜索域名白名单 |

- **涉及文件**：`crates/aibridge-core/src/model/chat.rs`
- **注意**：`repetition_penalty` 在 OPENAI_COMPATIBLE_MAPPING 中已处理（options.rs:343 rename_map 中有），但 ChatRequest 本身缺字段

#### 3. ParameterMapping 层不完整
- **现状**：`options.rs` 有基础 `ParamMapping`，仅支持 `rename_map`（键名重命名），**不支持 value_map**
- **旧版**：`ParameterMapping` 支持 rename_map + value_map（值映射，如 `reasoning: true → {"thinking": {"type": "enabled"}}`）+ extra_headers
- **影响**：Anthropic 的 `reasoning_effort → thinking.budget_tokens` 值映射在 `build_chat_body` 中硬编码特殊处理，无法扩展到其他 provider
- **涉及文件**：`crates/aibridge-core/src/model/options.rs`

#### 4. Anthropic ToolCall 未实现
- **位置**：`crates/aibridge-core/src/adapters/anthropic.rs:80`
- **现状**：注释明确说明"工具调用的完整请求体/响应解析未在本阶段实现，tool 消息按 user/tool_result 简化转换"，Anthropic adapter 的 capabilities 中**不包含 Capabilities::ToolCall**
- **影响**：通过 Anthropic 原生协议调用时无法使用 function calling（走 agnes 代理不受影响）

---

### P1 — Python 绑定缺口（用户主用语言）

#### 5. Python 绑定缺少 `list_voices` / `recommend_voices`
- **现状**：Core 层已实现（edge-tts/elevenlabs/cartesia），但 Python 绑定未暴露这两个方法
- **旧版**：`client.list_voices()` 和 `client.recommend_voices()` 可用
- **影响**：Python 用户无法查询可用音色，业务层"声音池"维护功能缺失
- **涉及文件**：`crates/aibridge-python/src/lib.rs`

#### 6. Python 绑定缺少 Router
- **现状**：`crates/aibridge-core/src/router.rs` 完整实现，但 Python 绑定完全没有 Router 类
- **旧版**：`agn/router.py` 有 Router 类
- **影响**：Python 用户无法使用多 Provider 负载均衡/Fallback
- **涉及文件**：`crates/aibridge-python/src/lib.rs`（需新增 Router pyclass）

#### 7. Python 绑定缺少异步上下文管理器
- **旧版**：支持 `async with Client(...) as client:`
- **现状**：需确认 PyO3 绑定是否实现了 `__aenter__` / `__aexit__`
- **涉及文件**：`crates/aibridge-python/src/lib.rs`

---

### P2 — 其他语言绑定缺口

#### 8. Node 绑定只有 3 个能力
- **现状**：417 行，只暴露 `chat` / `chat_stream` / `speech`
- **缺**：image_generate / video_create / video_poll / embed / transcribe / list_models / list_voices
- **涉及文件**：`crates/aibridge-node/src/`

#### 9. FFI 层（C ABI）只有 3 个能力
- **现状**：只导出 `aibridge_client_chat` / `aibridge_client_chat_stream` / `aibridge_client_speech`
- **缺**：image/video/embed/transcribe/list_models/list_voices/recommend_voices/translate + Router
- **影响**：Go/JVM/.NET 绑定只能用这 3 个能力
- **涉及文件**：`crates/aibridge-ffi/src/lib.rs`

#### 10. Go/JVM/.NET 绑定只有 hello world
- **现状**：管线验证通过，但未封装完整能力（依赖 FFI 补全）
- **涉及文件**：`bindings/go/`、`bindings/jvm/`、`bindings/dotnet/`
- **注意**：.NET 需先 `brew install --cask dotnet-sdk`

---

### P3 — 模型字段补齐

#### 11. VideoRequest 字段确认
VideoRequest **已包含**：width/height/num_frames/frame_rate/mode/duration/aspect_ratio/resolution/reference_images/first_frame/last_frame/camera_motion/motion_strength/negative_prompt/seed/steps/cfg_scale/with_audio/watermark/extra

**真正缺失的字段**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `reference_videos` | `Vec<FileInput>` | 参考视频（video2video 模式） |
| `keyframes` | `Vec<HashMap>` | 关键帧列表 |
| `fps` | `Option<u32>` | 帧率别名（已有 frame_rate，确认是否需要） |
| `style` | `Option<String>` | 视频风格 |

- **还需确认**：`VideoMode` 枚举是否包含 `Video2Video` 变体（目前有 Text2Video/Image2Video/Keyframes/Multiimage）

#### 12. ImageRequest 字段确认
已包含：size/width/height/aspect_ratio/n/style/quality/negative_prompt/seed/steps/cfg_scale/response_format/output_format/reference_images/mask/edit_mode/extra

**真正缺失的字段**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `sampler` | `Option<String>` | 采样器名称 |
| `scheduler` | `Option<String>` | 调度器 |
| `reference_strength` | `Option<f64>` | 参考图强度（0-1） |
| `negative_prompts` | `Vec<String>` | 负面提示词列表（目前只有单个 negative_prompt: Option<String>） |

#### 13. EmbedRequest 字段
| 缺失字段 | 类型 | 说明 |
|----------|------|------|
| `dimensions` | `Option<u32>` | 输出向量维度 |
| `encoding_format` | `Option<String>` | float/base64 编码 |
| `user` | `Option<String>` | 用户标识 |

- **涉及文件**：`crates/aibridge-core/src/model/options.rs`（EmbedRequest 定义位置）

---

### P4 — 工程化 / 集成

#### 14. retry_with 未自动集成到 HTTP 调用
- **现状**：`retry.rs` 实现了 `retry_with()`，但各适配器的 HTTP 调用直接 `.await`，**未包裹 retry_with**
- **影响**：自动重试实际上未生效（只有 ClientConfig 里有 max_retries 配置，但没有被使用）
- **涉及文件**：`crates/aibridge-core/src/adapters/openai_compat.rs` + 各独立协议适配器

#### 15. 阶段 0.4 过时注释
- `client.rs:50`、`router.rs:13` 中有"阶段 0.4 适配器未实现"的注释，阶段 2 已全部实现，应清理

#### 16. 部分 provider 能力未完整声明
- Cohere/Perplexity：embed 走默认 UnsupportedCapability
- 中文适配器（chinese.rs）：audio 能力待 2c 补齐注释
- 这些取决于实际 API 支持情况，优先级低

#### 15. SpeechResult 缺少便利方法
- **旧版**：`SpeechResult.save_to_file(path)` 和 `get_audio_bytes()` 便利方法
- **现状**：Rust SpeechResult 有 audio_data/audio_url/audio_base64，但绑定层可能没暴露 save_to_file
- **涉及文件**：`crates/aibridge-core/src/model/audio.rs`、`crates/aibridge-python/src/lib.rs`

#### 16. 其他低优先级差异
- EdgeTTS proxy 参数（`speech(..., proxy="http://...")`）— 旧版支持，Rust 版待确认
- 中文 provider 的 audio 能力（chinese.rs 注释说待 2c 补齐）— qwen/doubao/minimax 的 ASR/TTS
- SpeechResult.content_type / duration 字段（Rust 已有，确认 Python 绑定是否暴露）
- AssemblyAI 说话人分离字段透传（speaker/channel 等）

---

### P5 — 发布准备

- [ ] CHANGELOG 更新为 v2.0.0
- [ ] v2.0.0 git tag
- [ ] 一致性测试（`tests/consistency/`）纳入 CI
- [ ] 真实 API key 冒烟测试（需用户配置）
- [ ] PyPI 发布（需用户 token）
- [ ] npm 发布（需用户 token）
- [ ] Maven Central / NuGet 发布（后续）
- [ ] 文档网站 v2 API 参考更新（mkdocs 已配置）
- [ ] .NET hello world 验证（`brew install --cask dotnet-sdk`）

---

## 三、推荐开发顺序

### 阶段 3a：Core 层补齐（P0+P3，预估 1-2 天）
1. Client/Adapter trait 加 `translate()` 方法 + Deepgram/AssemblyAI 实现
2. ChatRequest 补字段：repetition_penalty / min_p / thinking_budget / stream_options / web_search / 搜索过滤
3. ParamMapping 加 value_map 支持，迁移 Anthropic thinking 硬编码到映射表
4. Anthropic ToolCall 实现（native 协议 function calling）
5. VideoRequest 补：reference_videos / keyframes / style；VideoMode 加 Video2Video
6. ImageRequest 补：sampler / scheduler / reference_strength / negative_prompts 列表
7. EmbedRequest 补：dimensions / encoding_format / user
8. 清理过时注释
9. `cargo test` 全量通过

### 阶段 3b：Python 绑定补全（P1，预估 1 天）
1. Python 绑定加 `list_voices` / `recommend_voices`
2. Python 绑定加 `Router` pyclass（多 Provider 路由/Fallback）
3. Python 绑定加 `translate`
4. Python `__aenter__` / `__aexit__` 异步上下文管理器
5. 跑 Python 一致性测试验证

### 阶段 3c：重试机制 + FFI/Node 补全（P2+P4，预估 2 天）
1. 把 `retry_with` 集成到 openai_compat 地基和独立适配器的 HTTP 调用
2. FFI 层补全：image/video/embed/transcribe/list_models/list_voices/translate
3. Node 绑定补全剩余能力（参考 Python 模式，napi async fn）
4. Go/JVM/.NET 绑定通过 FFI 封装

### 阶段 3d：发布（P5，预估 0.5 天）
1. CHANGELOG + tag
2. 文档更新
3. 等用户提供 PyPI/npm token 后发布

---

## 四、关键文件索引

| 模块 | 路径 |
|------|------|
| Rust core 入口 | `crates/aibridge-core/src/lib.rs` |
| Client | `crates/aibridge-core/src/client.rs` |
| Adapter trait | `crates/aibridge-core/src/adapter/base.rs` |
| Adapter 工厂 | `crates/aibridge-core/src/adapter/factory.rs` |
| OpenAI 兼容地基（1962 行） | `crates/aibridge-core/src/adapters/openai_compat.rs` |
| Chat 模型 | `crates/aibridge-core/src/model/chat.rs` |
| Image 模型 | `crates/aibridge-core/src/model/image.rs` |
| Video 模型 | `crates/aibridge-core/src/model/video.rs` |
| Audio 模型 | `crates/aibridge-core/src/model/audio.rs` |
| Options/工具/Embed 模型 | `crates/aibridge-core/src/model/options.rs` |
| Common 模型 | `crates/aibridge-core/src/model/common.rs` |
| Router | `crates/aibridge-core/src/router.rs` |
| 错误处理 | `crates/aibridge-core/src/error.rs` |
| 重试策略 | `crates/aibridge-core/src/retry.rs` |
| HTTP 客户端 | `crates/aibridge-core/src/http.rs` |
| 参数映射 | `crates/aibridge-core/src/model/options.rs`（ParamMapping） |
| FFI 层 | `crates/aibridge-ffi/src/lib.rs`（1610 行） |
| Python 绑定 | `crates/aibridge-python/src/lib.rs`（2090 行） |
| Node 绑定 | `crates/aibridge-node/src/lib.rs`（417 行） |
| Go 绑定 | `bindings/go/` |
| JVM 绑定 | `bindings/jvm/` |
| .NET 绑定 | `bindings/dotnet/` |
| 旧版 Python 参考 | `agn/` |
| 设计文档 | `docs/superpowers/specs/2026-07-07-aibridge-rust-rewrite-design.md` |
| 进度文档 | `docs/PROGRESS.md` |
| 迁移指南 | `docs/migration-guide.md`（Python v1→v2） |
| 一致性测试 | `tests/consistency/` |

---

## 五、修正说明

本文档基于两次代码审查生成：
1. 初次对比（人工）发现主要结构差异
2. Rust SDK 深度探索 agent 发现了更精确的细节（1449 单测、VideoRequest 实际已有很多字段、retry_with 未接入 HTTP、Anthropic ToolCall 未实现、Python list_voices 未暴露等）
