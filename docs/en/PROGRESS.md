# AIBridge Rust Rewrite · Progress & Handover Document

> This document is for any agent taking over the AIBridge Rust rewrite work. Self-contained, not dependent on Claude memory.
> Last updated: 2026-07-08

---

## 1. Project Overview

Rewriting the Python `agn-sdk` (a multimodal AI unified interface SDK, v1.3.3, ~19,700 lines) in Rust as the cross-language SDK `aibridge`, supporting direct import in five languages.

- **Branch**: `feat/aibridge-rust-rewrite` (based on `main`; `main` is still the legacy Python version)
- **Brand name**: aibridge (formerly agn-sdk; available on both PyPI/npm)
- **Five languages**: Python (PyO3 directly to core) / JS-TS (napi-rs directly to core) / Go (CGO calling ffi) / JVM (JNA calling ffi) / .NET (P/Invoke calling ffi)

## 2. Key Documents (must read)

| Document | Path | Content |
|---|---|---|
| Design document | [docs/superpowers/specs/2026-07-07-aibridge-rust-rewrite-design.md](design.md) | Architecture, data models, FFI boundary, async bridging, error handling, adapter migration strategy |
| Implementation plan | [docs/superpowers/plans/2026-07-07-aibridge-implementation-plan.md](plan.md) | Phase 0-3 task breakdown, multi-agent orchestration strategy, milestones |
| Migration guide | [docs/migration-guide.md](migration-guide.md) | Python v1 (agn-sdk) → v2 (aibridge) breaking-change comparison and examples |
| v2 README | [README.md](index.md) | Five-language quick start + provider list |
| This progress document | docs/PROGRESS.md | Current progress + handover guide (this document) |

## 3. Architecture Overview

```
                    aibridge-core  (Rust, pure async logic)
                  ┌──────────┴──────────┐
        direct (native async)         C ABI (aibridge-ffi cdylib)
        ┌─────┴─────┐            ┌─────┬─────┬─────┐
   aibridge-    aibridge-     aibridge- aibridge- aibridge-
   python       node          go       jvm       dotnet
   (PyO3)       (napi-rs)     (CGO)    (JNA)     (P/Invoke)
```

### Monorepo layout
```
aibridge/
├── crates/
│   ├── aibridge-core/      # Rust core (pure logic, no FFI)
│   ├── aibridge-ffi/       # C ABI cdylib (for Go/JVM/.NET)
│   ├── aibridge-python/    # PyO3 binding (directly to core)
│   └── aibridge-node/      # napi-rs binding (directly to core)
├── bindings/
│   ├── go/                 # CGO calling ffi
│   ├── jvm/                # JNA calling ffi (Java)
│   └── dotnet/             # P/Invoke calling ffi (C#)
├── docs/                   # Design document + plan + migration guide + this progress document
└── examples/               # Five-language hello world (echo adapter)
```

## 4. Current Progress (as of 2026-07-08)

### ✅ Phase 0: Foundation (complete)
- Cargo workspace + 4-crate skeleton
- aibridge-core: error/config/http/retry/model/adapter/client/router
- aibridge-ffi: C ABI (global tokio runtime + handles + JSON boundary + cbindgen-generated include/aibridge.h)
- Five-language binding hello world running (Python/Node/Go/JVM; .NET code ready pending dotnet sdk)

### ✅ Phase 1: MVP four providers (complete)
- openai / agnes / Volcengine (volcengine_cv) / gemini
- Usable across five languages (including streaming chat_stream)
- Python/Node streaming refactored (PyO3 Coroutine / napi async fn, real IO without blocking)

### ✅ Phase 1.5: Quality wrap-up (complete)
- Cross-language consistency tests (four languages fully consistent in chat/stream/speech/errors)
- Unified error codes (aligned with the code() values in Rust error.rs)
- dylib artifact name conflict fixed (aibridge-python lib changed to `_aibridge`)

### ✅ Phase 2: Full adapter migration (complete, 38 real providers + echo mock)

Phase 2 migrated all v1 adapters in three batches; the compile-time match factory has registered all providers. Factory placeholder branches have been cleared.

#### Phase 2a: OpenAI-compatible family (complete, 24 providers)
| File | provider (including aliases) |
|---|---|
| openai.rs | openai |
| agnes.rs | agnes |
| volcengine_cv.rs | volcengine_cv (Volcengine) |
| gemini.rs | gemini |
| azure.rs | azure |
| aggregation_platforms.rs | siliconflow\|sf, togetherai\|together, fireworksai\|fireworks, cloudflareai\|cloudflare\|workersai |
| additional_models.rs | grok\|xaigrok, yi\|lingyiwanwu, sensenova\|shangtang, hunyuan\|tencent_hunyuan, groq |
| more_models.rs | deepseek, stepfun\|step, mistral, cohere, perplexity |
| emerging_models.rs | ideogram\|ideo, luma\|dream-machine\|lumalabs, llama\|meta-llama\|meta |
| chinese.rs | qwen, zhipu, doubao, ernie, kimi, minimax |

#### Phase 2b: Standalone protocols, 5 (complete)
| File | provider | Capabilities |
|---|---|---|
| anthropic.rs | anthropic | Claude messages API, streaming SSE, text conversation/multimodal |
| stability.rs | stability | Stability AI text-to-image/image-to-image |
| runway.rs | runway | Video generation (text-to-video/image-to-video/task polling) |
| pika.rs | pika | Video generation (text-to-video/image-to-video/task polling) |
| kling.rs | kling | Kling video generation (text-to-video/image-to-video/task polling) |

#### Phase 2c: Audio, 5 (complete, binary payloads)
| File | provider | Capabilities | Notes |
|---|---|---|---|
| edge_tts.rs | edge-tts (aliases edge_tts / edge) | Free TTS | No authentication required (`requires_api_key=false`) |
| elevenlabs.rs | elevenlabs (aliases eleven / 11labs) | TTS | High-quality voices/multilingual/cloning |
| cartesia.rs | cartesia (alias sonic) | TTS | Sonic low-latency streaming |
| deepgram.rs | deepgram (alias dg) | ASR | Token auth / REST |
| assemblyai.rs | assemblyai (aliases assembly / aai) | ASR | Key auth / REST |

#### Complete list of migrated providers (38 real + 1 mock)

**MVP (4)**: openai, agnes, volcengine_cv, gemini
**Compatible family (24)**: azure, siliconflow, togetherai, fireworksai, cloudflareai, grok, yi, sensenova, hunyuan, groq, deepseek, stepfun, mistral, cohere, perplexity, ideogram, luma, llama, qwen, zhipu, doubao, ernie, kimi, minimax
**Standalone protocols (5)**: anthropic, stability, runway, pika, kling
**Audio (5)**: edge-tts, elevenlabs, cartesia, deepgram, assemblyai
**Mock (1)**: echo (used for Phase 0.6 pipeline verification, permanent)

### ⏳ Phase 3: Release wrap-up (in progress)
- [x] Python v1→v2 migration guide ([docs/migration-guide.md](migration-guide.md))
- [x] v2 README ([README.md](index.md))
- [x] Progress document update (this document)
- [ ] CI matrix (platform × language, cross-compilation)
- [ ] Five-language package publishing (PyPI `aibridge` / npm `aibridge` / Maven `io.aibridge:aibridge` / NuGet `AIBridge` / Go module `aibridge-go`)
- [ ] Documentation website
- [ ] Archive legacy v1 + tag v2.0.0
- [ ] .NET hello world verification (pending dotnet sdk)
- [ ] Consistency tests integrated into CI (currently run manually)
- [ ] Real API key smoke test (user verification)

## 5. Test Status

- **aibridge-core**: 1448 unit tests all pass (including 38 provider mock HTTP tests + data models + errors + routing)
- **aibridge-ffi**: 39 unit tests all pass
- **Five-language hello world** (echo adapter): Python/Node/Go/JVM running; .NET code ready pending dotnet
- **Cross-language consistency tests**: tests/consistency/ (four languages fully consistent in chat/stream/speech/errors)

## 6. Outstanding Issues (non-blocking)

1. **.NET hello world**: pending dotnet sdk installation (`brew install --cask dotnet-sdk`, requires sudo); code + P/Invoke already ready
2. **Consistency tests into CI**: currently run manually, pending integration into CI matrix
3. **Real IO streaming verification**: Python/Node streaming refactoring is complete (does not block inference), but real API key verification is pending user
4. **dylib distribution**: Go/JVM/.NET depend on libaibridge; JVM/.NET bundle it into the package, Go provides an installation script (handled in Phase 3)
5. **Python binding capability exposure**: The Rust core has fully implemented 38 providers + six major capabilities; the Python binding (PyO3) currently exposes `chat/chat_stream/speech`, while `image_generate/video_*/embed/transcribe/list_models/list_voices` will be exposed in a later version (see migration guide Q1)

## 7. New Agent Handover Guide

### 7.1 Environment
- Rust 1.94 + cargo (crates.io mirror already configured in `~/.cargo/config.toml` to use rsproxy; required for networks in China)
- Python 3.14 + maturin (`pip install maturin`)
- Node 25 + npm
- Go 1.26
- Java 21 (Temurin)
- dotnet not installed (.NET binding code ready, hello world pending verification)

### 7.2 Verify Current State
```bash
git checkout feat/aibridge-rust-rewrite
cargo test -p aibridge-core        # expect 1448 passed
cargo test -p aibridge-ffi         # expect 39 passed
cargo build --workspace            # 0 warning
```

### 7.3 How to Continue Phase 3

Phase 3 is release wrap-up, with the main work being:

1. **CI matrix**: GitHub Actions platform (linux/macos/windows × amd64/arm64) × language. The aibridge-ffi dynamic library is a build artifact for Go/JVM/.NET packaging to consume. One workflow each for the Rust core + 5 bindings.
2. **Five-language package publishing**:
   - Python: `maturin build --release` produces a wheel, published to PyPI `aibridge`
   - Node: `napi build --release` produces a .node, published to npm `aibridge`
   - Go: provide a libaibridge installation script, Go module `aibridge-go`
   - JVM: bundle the dynamic library into the jar (by platform classifier), Maven `io.aibridge:aibridge`
   - .NET: bundle the dynamic library into the package (runtimes/{rid}/native/), NuGet `AIBridge`
3. **Python binding completion**: In `crates/aibridge-python/src/lib.rs`'s `#[pymethods] impl Client`, add `image_generate/video_create/video_poll/embed/transcribe/list_models/list_voices/recommend_voices`, following the pattern of the existing `chat`/`speech` (builder construction + RUNTIME.spawn + map_error).
4. **Documentation website**: mkdocs or similar, integrating the design document + migration guide + five-language API.
5. **v1 archiving**: After tagging the `main` branch (Python v1) with `v1.3.3`, archive it; README points to v2.

### 7.4 Key Constraints (must follow)
- **Chinese comments** (mandatory project rule, file/module docstrings + public item doc comments)
- **Error code alignment** with the actual `code()` values in Rust `aibridge-core/src/error.rs` (with the `_error` suffix)
- **mockito 1.x usage**: `let mut server = mockito::Server::new_async().await;` (not `async_try_start`)
- **aibridge-python lib name = `_aibridge`** (to avoid conflict with ffi's libaibridge dylib; do not change back to `aibridge`)
- **Python/Node streaming** has been refactored (PyO3 Coroutine / napi async fn); do not revert to block_on
- **Adapter trait** is in `crates/aibridge-core/src/adapter/base.rs`; factory registration is in `adapter/factory.rs` (compile-time match, not runtime registration)
- **openai_compat.rs**'s 9 methods are already pub; sub-adapters reuse via composition delegation
- **Dual-prefix environment variable compatibility**: `AIBRIDGE_*` new prefix + `AGN_*` old prefix coexist (see `config.rs merge_env`)

### 7.5 Key Commands
```bash
# Rust core
cargo test -p aibridge-core
cargo build --workspace
cargo clippy -p aibridge-core -- -D warnings

# Python binding
pip install maturin
maturin develop -m crates/aibridge-python/Cargo.toml
python examples/hello_python.py

# Node binding
cd crates/aibridge-node && npm install && napi build && cd ../..
node examples/hello_node.js

# Go binding
cargo build -p aibridge-ffi
cd bindings/go && CGO_ENABLED=1 DYLD_LIBRARY_PATH=../../target/debug go run ./example

# JVM binding
cd bindings/jvm && ./gradlew run

# .NET binding
cargo build -p aibridge-ffi
cd bindings/dotnet && dotnet run
```

## 8. Commit History (Phase 0-3)

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

## 9. User Preferences (required reading for taking-over agents)

- The user is a **programming beginner**; communicate in **plain language**, fewer jargon terms and more analogies
- Act as an **industry expert making technical decisions on their behalf**; for purely technical trade-offs, make the call directly and explain the reasoning
- Only seek their input at **business forks that affect cost/time/compatibility**
- The user has **fully delegated** end-to-end implementation to Claude, using multi-agent parallelism during the implementation phase
- All replies, code comments, and documentation use **Chinese** (except technical identifiers)
