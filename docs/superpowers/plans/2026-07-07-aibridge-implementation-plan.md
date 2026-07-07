# AIBridge 实现计划

> 日期：2026-07-07
> 依据设计：[docs/superpowers/specs/2026-07-07-aibridge-rust-rewrite-design.md](../superpowers/specs/2026-07-07-aibridge-rust-rewrite-design.md)
> 实施方式：Claude 全权委托；阶段 0 串行，阶段 1+ 多 agent 并行

---

## 实施总策略

- **阶段 0（地基）**：Claude 串行搭建，保证命名/依赖/配置一致性（并行易产生不一致）
- **阶段 1（MVP 四 provider）**：多 agent 并行（每 provider 一个 agent），先由 Claude 搭好 OpenAI 兼容地基
- **阶段 2（剩余适配器）**：多 agent 并行，按批次 2a/2b/2c
- **五语言绑定**：aibridge-core 与 aibridge-ffi 稳定后，多 agent 并行（每语言一个 agent）
- 每个任务带明确验收标准，作为 agent 完成判据与多 agent 编排的契约

---

## 阶段 0：地基（串行，2–3 周）

### 0.1 Cargo workspace 骨架
- workspace 根 `Cargo.toml` + 4 crate：aibridge-core / aibridge-ffi / aibridge-python / aibridge-node
- 统一依赖版本（workspace.dependencies）
- rust-toolchain.toml + 更新 .gitignore
- **验收**：`cargo build -p aibridge-core -p aibridge-ffi` 通过

### 0.2 aibridge-core 基础设施
- `error.rs`：AibridgeError 枚举（thiserror），对齐 Python v1 错误分类
- `config.rs`：ClientOptions + 环境变量加载（dotenv）
- `http.rs`：reqwest 封装（h2、超时、连接池）
- `retry.rs`：重试机制（指数退避，对应 tenacity）
- `util.rs`
- **验收**：核心层单测通过

### 0.3 数据模型层
- `model/{chat,image,video,audio,common,options}.rs`
- serde struct + Builder derive
- **验收**：模型序列化/反序列化测试通过，与 Python fixture 对照

### 0.4 Adapter trait + Client + Router
- `adapter::{Adapter trait, Capabilities, create_adapter}`
- `client::Client`、`router::Router`
- **验收**：trait 可实现，Client/Router 编译通过

### 0.5 aibridge-ffi C ABI 骨架
- 全局 tokio runtime（Lazy）
- 句柄管理（client/stream opaque）
- cbindgen 配置 + 生成 `aibridge.h`
- 基本函数：client_new/destroy/start、chat、chat_stream、stream_next/destroy、last_error、string/bytes_free
- **验收**：`cargo build -p aibridge-ffi` 产 cdylib，aibridge.h 生成

### 0.6 跨语言管线验证（openai chat stub）
- aibridge-python：PyO3 模块 + Client.chat async + 流式 AsyncIterator
- aibridge-node：napi 模块 + Client.chat async + 流式 AsyncIterable
- bindings/go：CGO 调 ffi + chat（goroutine+channel）
- bindings/jvm：JNA 调 ffi + chat（CompletableFuture）
- bindings/dotnet：P/Invoke 调 ffi + chat（Task）
- **验收**：五语言 hello world（含 async + 流式 + 错误）跑通 ← **最大技术风险点**

---

## 阶段 1：MVP 四 provider（多 agent 并行，3–4 周）

### 1.0 OpenAI 兼容地基（Claude 串行）
- `adapters/openai_compat.rs`：共享 HTTP 请求构造/响应解析/参数映射
- `OPENAI_COMPATIBLE_MAPPING` 常量
- **验收**：可被子适配器复用

### 1.1–1.4 四 provider 适配器（每 provider 一个 agent 并行）
| 任务 | provider | 协议 | 能力 |
|---|---|---|---|
| 1.1 | openai | 复用 compat | chat/image/embed/list_models |
| 1.2 | agnes | 复用 compat | chat/image/video/list_models |
| 1.3 | volcengine_cv | 独立 | image/video |
| 1.4 | gemini | 独立 | chat/image/embed/list_models |
- **验收**：每适配器单测通过（共享 mock fixture）

### 1.5 五语言绑定完善（多 agent 并行，每语言一个）
- 四 provider 能力暴露到各语言
- 跨语言一致性测试
- **验收**：五语言 × 四 provider 全跑通 ← **用户另一项目可接入里程碑 M1**

---

## 阶段 2：剩余适配器（多 agent 并行批次，4–6 周）

### 2a OpenAI 兼容族（6 agent 并行）
azure, aggregation_platforms, additional_models, more_models, emerging_models, chinese(兼容部分)

### 2b 独立协议（5 agent 并行）
anthropic, stability, runway, pika, kling

### 2c 音频（edge-tts/elevenlabs/cartesia/deepgram/assemblyai，二进制载荷）

---

## 阶段 3：发布收尾（2–3 周）

- 3.1 CI 矩阵（平台 × 语言，交叉编译）
- 3.2 各语言包发布（PyPI/npm/Maven/NuGet/Go module）
- 3.3 Python v1→v2 迁移指南
- 3.4 文档网站
- 3.5 旧版 v1 归档 + 打 v2.0.0 tag

---

## 多 agent 编排原则

- **适配器迁移**：每 provider 一个 agent，共享 fixture 与 OpenAiCompatAdapter 基础
- **语言绑定**：每语言一个 agent，依赖 core/ffi 稳定后启动
- **跨语言一致性测试**：单独 agent 汇总
- **依赖顺序**：阶段 0 串行；阶段 1.0 地基串行；1.1–1.4 适配器并行；1.5 绑定并行
- **每个 agent 任务契约**：设计文档引用 + 验收标准 + fixture 路径 + 命名规范

---

## 里程碑

- **M0**：阶段 0 完成，五语言 hello world 跑通（技术风险解除）
- **M1**：阶段 1 完成，四 provider 五语言可用（用户另一项目可接入）⭐
- **M2**：阶段 2 完成，全量适配器迁移
- **M3**：阶段 3 完成，v2.0.0 正式发布
