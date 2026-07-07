# AIBridge Rust 重构 · 进度与接手文档

> 本文档供任何 agent 接手 AIBridge Rust 重构工作使用。自包含，不依赖 Claude memory。
> 最后更新：2026-07-08

---

## 1. 项目概述

将 Python `agn-sdk`（多模态 AI 统一接口 SDK，v1.3.3，~19700 行）用 Rust 重构为跨语言 SDK `aibridge`，支持五种语言直接 import。

- **分支**：`feat/aibridge-rust-rewrite`（基于 `main`，`main` 仍是 Python 旧版）
- **品牌名**：aibridge（原 agn-sdk，PyPI/npm 均可用）
- **五语言**：Python（PyO3 直连 core）/ JS-TS（napi-rs 直连 core）/ Go（CGO 调 ffi）/ JVM（JNA 调 ffi）/ .NET（P/Invoke 调 ffi）

## 2. 关键文档（必读）

| 文档 | 路径 | 内容 |
|---|---|---|
| 设计文档 | [docs/superpowers/specs/2026-07-07-aibridge-rust-rewrite-design.md](superpowers/specs/2026-07-07-aibridge-rust-rewrite-design.md) | 架构、数据模型、FFI 边界、异步桥接、错误处理、适配器迁移策略 |
| 实现计划 | [docs/superpowers/plans/2026-07-07-aibridge-implementation-plan.md](superpowers/plans/2026-07-07-aibridge-implementation-plan.md) | 阶段 0-3 任务分解、多 agent 编排策略、里程碑 |
| 迁移指南 | [docs/migration-guide.md](migration-guide.md) | Python v1（agn-sdk）→ v2（aibridge）破坏性升级对照与示例 |
| v2 README | [README_aibridge.md](../README_aibridge.md) | 五语言快速开始 + provider 列表 |
| 本进度文档 | docs/PROGRESS.md | 当前进度 + 接手指南（本文档） |

## 3. 架构概览

```
                    aibridge-core  (Rust, 纯 async 逻辑)
                  ┌──────────┴──────────┐
        直连(原生async)              C ABI (aibridge-ffi cdylib)
        ┌─────┴─────┐            ┌─────┬─────┬─────┐
   aibridge-    aibridge-     aibridge- aibridge- aibridge-
   python       node          go       jvm       dotnet
   (PyO3)       (napi-rs)     (CGO)    (JNA)     (P/Invoke)
```

### Monorepo 布局
```
aibridge/
├── crates/
│   ├── aibridge-core/      # Rust 核心（纯逻辑，无 FFI）
│   ├── aibridge-ffi/       # C ABI cdylib（给 Go/JVM/.NET）
│   ├── aibridge-python/    # PyO3 绑定（直连 core）
│   └── aibridge-node/      # napi-rs 绑定（直连 core）
├── bindings/
│   ├── go/                 # CGO 调 ffi
│   ├── jvm/                # JNA 调 ffi（Java）
│   └── dotnet/             # P/Invoke 调 ffi（C#）
├── docs/                   # 设计文档 + 计划 + 迁移指南 + 本进度文档
└── examples/               # 五语言 hello world（echo adapter）
```

## 4. 当前进度（截至 2026-07-08）

### ✅ 阶段 0：地基（完成）
- Cargo workspace + 4 crate 骨架
- aibridge-core：error/config/http/retry/model/adapter/client/router
- aibridge-ffi：C ABI（全局 tokio runtime + 句柄 + JSON 边界 + cbindgen 生成 include/aibridge.h）
- 五语言绑定 hello world 跑通（Python/Node/Go/JVM，.NET 代码就绪待 dotnet sdk）

### ✅ 阶段 1：MVP 四 provider（完成）
- openai / agnes / 火山(volcengine_cv) / gemini
- 五语言可用（含流式 chat_stream）
- Python/Node 流式已重构（PyO3 Coroutine / napi async fn，真实 IO 不阻塞）

### ✅ 阶段 1.5：质量收尾（完成）
- 跨语言一致性测试（四语言 chat/stream/speech/错误全一致）
- 错误 code 统一（对齐 Rust error.rs 的 code() 值）
- dylib 产物名冲突修复（aibridge-python lib 改 `_aibridge`）

### ✅ 阶段 2：全量适配器迁移（完成，38 真实 provider + echo mock）

阶段 2 分三批搬完 v1 全部适配器，编译期 match 工厂已注册全部 provider。工厂占位分支已清空。

#### 阶段 2a：OpenAI 兼容族（完成，24 provider）
| 文件 | provider（含别名） |
|---|---|
| openai.rs | openai |
| agnes.rs | agnes |
| volcengine_cv.rs | volcengine_cv（火山引擎） |
| gemini.rs | gemini |
| azure.rs | azure |
| aggregation_platforms.rs | siliconflow\|sf, togetherai\|together, fireworksai\|fireworks, cloudflareai\|cloudflare\|workersai |
| additional_models.rs | grok\|xaigrok, yi\|lingyiwanwu, sensenova\|shangtang, hunyuan\|tencent_hunyuan, groq |
| more_models.rs | deepseek, stepfun\|step, mistral, cohere, perplexity |
| emerging_models.rs | ideogram\|ideo, luma\|dream-machine\|lumalabs, llama\|meta-llama\|meta |
| chinese.rs | qwen, zhipu, doubao, ernie, kimi, minimax |

#### 阶段 2b：独立协议 5 个（完成）
| 文件 | provider | 能力 |
|---|---|---|
| anthropic.rs | anthropic | Claude messages API，流式 SSE，文本对话/多模态 |
| stability.rs | stability | Stability AI 文生图/图生图 |
| runway.rs | runway | 视频生成（文生视频/图生视频/任务轮询） |
| pika.rs | pika | 视频生成（文生视频/图生视频/任务轮询） |
| kling.rs | kling | 可灵视频生成（文生视频/图生视频/任务轮询） |

#### 阶段 2c：音频 5 个（完成，二进制载荷）
| 文件 | provider | 能力 | 备注 |
|---|---|---|---|
| edge_tts.rs | edge-tts（别名 edge_tts / edge） | 免费 TTS | 免认证（`requires_api_key=false`） |
| elevenlabs.rs | elevenlabs（别名 eleven / 11labs） | TTS | 高质量音色/多语种/克隆 |
| cartesia.rs | cartesia（别名 sonic） | TTS | Sonic 低延迟流式 |
| deepgram.rs | deepgram（别名 dg） | ASR | Token 鉴权 / REST |
| assemblyai.rs | assemblyai（别名 assembly / aai） | ASR | Key 鉴权 / REST |

#### 已迁移 provider 完整列表（38 真实 + 1 mock）

**MVP（4）**：openai、agnes、volcengine_cv、gemini
**兼容族（24）**：azure、siliconflow、togetherai、fireworksai、cloudflareai、grok、yi、sensenova、hunyuan、groq、deepseek、stepfun、mistral、cohere、perplexity、ideogram、luma、llama、qwen、zhipu、doubao、ernie、kimi、minimax
**独立协议（5）**：anthropic、stability、runway、pika、kling
**音频（5）**：edge-tts、elevenlabs、cartesia、deepgram、assemblyai
**Mock（1）**：echo（阶段 0.6 管线验证用，常驻）

### ⏳ 阶段 3：发布收尾（进行中）
- [x] Python v1→v2 迁移指南（[docs/migration-guide.md](migration-guide.md)）
- [x] v2 README（[README_aibridge.md](../README_aibridge.md)）
- [x] 进度文档更新（本文档）
- [ ] CI 矩阵（平台 × 语言，交叉编译）
- [ ] 五语言包发布（PyPI `aibridge` / npm `aibridge` / Maven `io.aibridge:aibridge` / NuGet `AIBridge` / Go module `aibridge-go`）
- [ ] 文档网站
- [ ] 旧版 v1 归档 + 打 v2.0.0 tag
- [ ] .NET hello world 验证（待 dotnet sdk）
- [ ] 一致性测试纳入 CI（当前手动跑）
- [ ] 真实 API key 冒烟测试（用户验证）

## 5. 测试状态

- **aibridge-core**：1448 单测全通过（含 38 provider mock HTTP 测试 + 数据模型 + 错误 + 路由）
- **aibridge-ffi**：39 单测全通过
- **五语言 hello world**（echo adapter）：Python/Node/Go/JVM 跑通，.NET 代码就绪待 dotnet
- **跨语言一致性测试**：tests/consistency/（四语言 chat/stream/speech/错误全一致）

## 6. 遗留问题（非阻塞）

1. **.NET hello world**：待装 dotnet sdk（`brew install --cask dotnet-sdk`，需 sudo），代码 + P/Invoke 已就绪
2. **一致性测试纳入 CI**：当前手动跑，待接入 CI matrix
3. **真实 IO 流式验证**：Python/Node 流式重构已完成（不阻塞推理），但真实 API key 验证待用户
4. **dylib 分发**：Go/JVM/.NET 依赖 libaibridge，JVM/.NET 打进包，Go 提供安装脚本（阶段 3 处理）
5. **Python 绑定能力暴露**：Rust 核心已全部实现 38 provider + 六大能力；Python 绑定（PyO3）目前暴露 `chat/chat_stream/speech`，`image_generate/video_*/embed/transcribe/list_models/list_voices` 待后续版本暴露（见迁移指南 Q1）

## 7. 新 agent 接手指南

### 7.1 环境
- Rust 1.94 + cargo（crates.io 镜像已配 `~/.cargo/config.toml` 走 rsproxy，国内网络必须）
- Python 3.14 + maturin（`pip install maturin`）
- Node 25 + npm
- Go 1.26
- Java 21 (Temurin)
- dotnet 未装（.NET 绑定代码就绪，hello world 待验证）

### 7.2 验证当前状态
```bash
git checkout feat/aibridge-rust-rewrite
cargo test -p aibridge-core        # 期望 1448 passed
cargo test -p aibridge-ffi         # 期望 39 passed
cargo build --workspace            # 0 warning
```

### 7.3 怎么继续阶段 3

阶段 3 是发布收尾，主要工作：

1. **CI 矩阵**：GitHub Actions 平台（linux/macos/windows × amd64/arm64）× 语言。aibridge-ffi 动态库作为构建 artifact 供 Go/JVM/.NET 打包消费。Rust 核心 + 5 绑定各一个 workflow。
2. **五语言包发布**：
   - Python：`maturin build --release` 产 wheel，发布到 PyPI `aibridge`
   - Node：`napi build --release` 产 .node，发布到 npm `aibridge`
   - Go：提供 libaibridge 安装脚本，Go module `aibridge-go`
   - JVM：动态库打进 jar（按平台 classifier），Maven `io.aibridge:aibridge`
   - .NET：动态库打进包（runtimes/{rid}/native/），NuGet `AIBridge`
3. **Python 绑定补全**：在 `crates/aibridge-python/src/lib.rs` 的 `#[pymethods] impl Client` 补 `image_generate/video_create/video_poll/embed/transcribe/list_models/list_voices/recommend_voices`，参照已有 `chat`/`speech` 的模式（builder 构造 + RUNTIME.spawn + map_error）。
4. **文档网站**：mkdocs 或 similar，整合设计文档 + 迁移指南 + 五语言 API。
5. **v1 归档**：`main` 分支（Python v1）打 `v1.3.3` tag 后归档，README 指向 v2。

### 7.4 关键约束（必须遵守）
- **中文注释**（项目强制规则，文件模块文档字符串 + 公开项文档注释）
- **错误 code 对齐** Rust `aibridge-core/src/error.rs` 的 `code()` 实际值（带 `_error` 后缀）
- **mockito 1.x 用法**：`let mut server = mockito::Server::new_async().await;`（不是 `async_try_start`）
- **aibridge-python lib name = `_aibridge`**（避免与 ffi 的 libaibridge dylib 冲突，不要改回 `aibridge`）
- **Python/Node 流式**已重构（PyO3 Coroutine / napi async fn），不要再改回 block_on
- **Adapter trait** 在 `crates/aibridge-core/src/adapter/base.rs`；工厂注册在 `adapter/factory.rs`（编译期 match，非运行时注册）
- **openai_compat.rs** 的 9 个方法已 pub，子适配器组合委托复用
- **环境变量双前缀兼容**：`AIBRIDGE_*` 新前缀 + `AGN_*` 老前缀并存（见 `config.rs merge_env`）

### 7.5 关键命令
```bash
# Rust 核心
cargo test -p aibridge-core
cargo build --workspace
cargo clippy -p aibridge-core -- -D warnings

# Python 绑定
pip install maturin
maturin develop -m crates/aibridge-python/Cargo.toml
python examples/hello_python.py

# Node 绑定
cd crates/aibridge-node && npm install && napi build && cd ../..
node examples/hello_node.js

# Go 绑定
cargo build -p aibridge-ffi
cd bindings/go && CGO_ENABLED=1 DYLD_LIBRARY_PATH=../../target/debug go run ./example

# JVM 绑定
cd bindings/jvm && ./gradlew run

# .NET 绑定
cargo build -p aibridge-ffi
cd bindings/dotnet && dotnet run
```

## 8. 提交历史（阶段 0-3）

```
（阶段 3）
<待提交> docs: 阶段3 Python v1→v2 迁移指南 + README + 进度更新

（阶段 2c 第三批，阶段 2 全部完成）
12925ad feat(aibridge-core): 阶段2c 第三批收尾 注册 deepgram + assemblyai 到工厂（阶段2 全部完成）
3fb46be feat(aibridge-core): 阶段2c assemblyai 适配器
e533bff feat(aibridge-core): 阶段2c deepgram 适配器

（阶段 2c 第二批，TTS）
ac0a3c8 feat(aibridge-core): 阶段2c 第二批收尾 注册 elevenlabs + cartesia 到工厂
5c2d659 feat(aibridge-core): 阶段2c cartesia 适配器
0b7c205 feat(aibridge-core): 阶段2c elevenlabs 适配器

（阶段 2b+2c 收尾，视频 + 免费 TTS）
03776d2 feat(aibridge-core): 阶段2b+2c 收尾 注册 kling + edge-tts 到工厂（edge-tts 免认证）
d6493c7 feat(aibridge-core): 阶段2c edge-tts 适配器（免费 TTS）
ee89f32 feat(aibridge-core): 阶段2b kling 适配器（可灵）

（阶段 2b 第二批，视频）
3f05c39 feat(aibridge-core): 阶段2b 第二批收尾 注册 runway + pika 到工厂
0124c5c feat(aibridge-core): 阶段2b pika 适配器
0247589 feat(aibridge-core): 阶段2b runway 适配器

（阶段 2b 第一批，独立协议）
04fd202 feat(aibridge-core): 阶段2b 第一批收尾 注册 anthropic + stability 到工厂
e6151da feat(aibridge-core): 阶段2b stability 适配器
954d153 feat(aibridge-core): 阶段2b anthropic 适配器

（阶段 2a 完成）
37a1b9a docs: AIBridge Rust 重构进度文档 + AGENTS.md 接手指引
3880168 feat(aibridge-core): 阶段2a 第三批收尾（emerging_models + chinese，阶段2a 完成）
43e923f feat(aibridge-core): 阶段2a chinese 适配器（中文模型聚合）
ba4a78b feat(aibridge-core): 阶段2a emerging_models 适配器
a0eb96d feat(aibridge-core): 阶段2a 第二批收尾（additional_models + more_models）
cb4536a feat(aibridge-core): 阶段2a 第一批收尾（azure + 聚合平台）

（阶段 0-1）
7b4a79d fix: 阶段1 Python/Node 流式桥接重构
e5df4f3 fix(aibridge-python): dylib 产物名改为 _aibridge
68954d2 feat: 阶段1.5 跨语言一致性测试 + 错误 code 统一
bf5cb1d feat(aibridge-core): 阶段1 收尾 注册四 MVP 适配器到工厂
60de645 feat(aibridge-core): 阶段1.0 OpenAI 兼容适配器地基
b81dab5 chore: 提交阶段0.6 五语言绑定依赖锁
0a802b9 feat(aibridge-dotnet): 阶段0.6 P/Invoke 绑定
268cd49 feat(aibridge-python): 阶段0.6 PyO3 绑定 + hello world
2ecfb58 feat(aibridge-node): 阶段0.6 napi-rs 绑定 + hello world
5118eeb feat(aibridge-jvm): 阶段0.6 JNA 绑定 + hello world
f36ee7c feat(aibridge-go): 阶段0.6 CGO 绑定 + hello world
8020c9c feat(aibridge-core): echo 适配器用于阶段0.6 管线验证
ddafa26 feat(aibridge-ffi): 阶段0.5 C ABI 层
414678d feat(aibridge-core): 阶段0.2-0.4 基础设施/数据模型/Adapter/Client/Router
1b7414e chore: 提交 Cargo.lock
350177c feat(aibridge): 阶段0.1 Cargo workspace 骨架
65fcf2b docs: AIBridge Rust 重构设计文档
```

## 9. 用户偏好（接手 agent 必读）

- 用户是**编程小白**，沟通用**大白话**，少术语多类比
- 以**行业专家身份替其做技术决策**，纯技术取舍直接拍板并解释理由
- 只在**影响成本/时间/兼容性的业务分叉**处征求其意见
- 用户**全权委托** Claude 端到端实施，实施阶段用多 agent 并行
- 所有回复、代码注释、文档用**中文**（技术标识符除外）
