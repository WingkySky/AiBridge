# AIBridge Rust 重构 · 进度与接手文档

> 本文档供任何 agent 接手 AIBridge Rust 重构工作使用。自包含，不依赖 Claude memory。
> 最后更新：2026-07-07

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
├── docs/                   # 设计文档 + 计划 + 本进度文档
└── examples/               # 五语言 hello world（echo adapter）
```

## 4. 当前进度（截至 2026-07-07）

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

### ✅ 阶段 2a：OpenAI 兼容族 + 部分独立协议（完成，23 provider）
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

### ⏳ 阶段 2b：独立协议 5 个（待做）
anthropic / stability / runway / pika / kling

### ⏳ 阶段 2c：音频 5 个（待做，二进制载荷）
edge-tts / elevenlabs / cartesia / deepgram / assemblyai

### ⏳ 阶段 3：发布（待做）
- CI 矩阵（平台 × 语言，交叉编译）
- 五语言包发布（PyPI `aibridge` / npm `aibridge` / Maven `io.aibridge:aibridge` / NuGet `AIBridge` / Go module `aibridge-go`）
- Python v1→v2 迁移指南
- 文档网站
- 旧版 v1 归档 + 打 v2.0.0 tag

## 5. 测试状态

- **aibridge-core**：810 单测全通过
- **aibridge-ffi**：39 单测全通过
- **五语言 hello world**（echo adapter）：Python/Node/Go/JVM 跑通，.NET 代码就绪待 dotnet
- **跨语言一致性测试**：tests/consistency/（四语言 chat/stream/speech/错误全一致）

## 6. 遗留问题（非阻塞）

1. **.NET hello world**：待装 dotnet sdk（`brew install --cask dotnet-sdk`，需 sudo），代码 + P/Invoke 已就绪
2. **一致性测试纳入 CI**：当前手动跑，待接入 CI matrix
3. **真实 IO 流式验证**：Python/Node 流式重构已完成（不阻塞推理），但真实 API key 验证待用户
4. **dylib 分发**：Go/JVM/.NET 依赖 libaibridge，JVM/.NET 打进包，Go 提供安装脚本（阶段 3 处理）

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
cargo test -p aibridge-core        # 期望 810 passed
cargo test -p aibridge-ffi         # 期望 39 passed
cargo build --workspace            # 0 warning
```

### 7.3 怎么继续阶段 2b/2c

**模式**：每批 2 个适配器，worktree 并行实现 + 收尾 agent cherry-pick 注册（同阶段 2a）。

1. **启动 2 个 worktree agent**（每个实现一个 adapter .rs）：
   - prompt 要点：读设计文档第 10 节 + Python 对应 `agn/adapters/<name>.py` + `openai_compat.rs`（地基）+ `volcengine_cv.rs`/`more_models.rs Cohere`（独立协议范例）；实现 adapter .rs；mockito 单测；**只 git add 自己的 .rs**（不 add mod.rs/factory.rs）；commit 返回 hash
   - `isolation: "worktree"` + `run_in_background: true`
   - worktree 初始可能在 main 分支（无 crates），需 `git checkout feat/aibridge-rust-rewrite` 或基于它建工作分支
2. **收尾 agent**：cherry-pick 2 个 commit + 注册 mod.rs（pub mod）+ factory.rs（match 分支 + 别名，参考 Python `agn/adapters/factory.py` 的 register）+ 更新 factory 测试 + `cargo test` 全量 + commit
3. **避免 4+ 并行**（会触发 API 限流 429），每批 2 个

### 7.4 阶段 2b 各适配器要点
- **anthropic**：Claude messages API，`POST /messages`，header `x-api-key`+`anthropic-version`，流式 SSE 事件（content_block_delta），`ANTHROPIC_MAPPING`。Python: `agn/adapters/anthropic.py`
- **stability**：Stability AI 图像协议，独立。Python: `agn/adapters/stability.py`
- **runway**：视频协议，独立。Python: `agn/adapters/runway.py`
- **pika**：视频协议，独立。Python: `agn/adapters/pika.py`
- **kling**：可灵视频/图像，独立。Python: `agn/adapters/kling.py`

### 7.5 阶段 2c 音频要点
- 二进制载荷（TTS 返 audio_data bytes，ASR 接受 file path/URL/bytes/base64）
- edge-tts 免认证（`requires_api_key=false`，加到 `client.rs` 的 `is_free_provider`）
- TTS 音色健康检查/推荐/自动降级（v1.3.3 特性，保留）
- Python: `agn/adapters/audio_adapters.py`（含全部 5 个）

### 7.6 关键约束（必须遵守）
- **中文注释**（项目强制规则，文件模块文档字符串 + 公开项文档注释）
- **错误 code 对齐** Rust `aibridge-core/src/error.rs` 的 `code()` 实际值（带 `_error` 后缀）
- **mockito 1.x 用法**：`let mut server = mockito::Server::new_async().await;`（不是 `async_try_start`）
- **aibridge-python lib name = `_aibridge`**（避免与 ffi 的 libaibridge dylib 冲突，不要改回 `aibridge`）
- **Python/Node 流式**已重构（PyO3 Coroutine / napi async fn），不要再改回 block_on
- **Adapter trait** 在 `crates/aibridge-core/src/adapter/base.rs`；工厂注册在 `adapter/factory.rs`（编译期 match，非运行时注册）
- **openai_compat.rs** 的 9 个方法已 pub，子适配器组合委托复用

### 7.7 关键命令
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
```

## 8. 提交历史（阶段 0-2a）

```
3880168 feat(aibridge-core): 阶段2a 第三批收尾（emerging_models + chinese，阶段2a 完成）
a0eb96d feat(aibridge-core): 阶段2a 第二批收尾（additional_models + more_models）
cb4536a feat(aibridge-core): 阶段2a 第一批收尾（azure + 聚合平台）
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
