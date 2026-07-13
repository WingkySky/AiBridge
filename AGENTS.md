# AGN-SDK / AIBridge 项目引导 - Agent 执行前必读

> 本文件是 Agent 执行任务时必须首先加载的项目指南。
> v2.0.0 已合并至 `main`：项目当前主线是 **AIBridge**（Rust 核心 + 五语言原生绑定）。
> Python v1（`agn-sdk`）已归档，仅作历史参考。

---

## 1. 项目现状

- **当前主线**：AIBridge v2 - 跨语言 AI 统一接口 SDK（Rust 核心 + Python / JS-TS / Go / JVM / .NET 五语言原生绑定）
- **版本**：`2.0.0-alpha.1`（见 `Cargo.toml`），阶段 3 发布收尾中
- **能力**：chat（含流式）/ image / video / TTS / ASR / embed，38 个真实 provider + 1 个 mock（echo）
- **v1 状态**：Python v1（`agn-sdk`）已归档，不再迭代。老用户参考 [README_v1.md](README_v1.md) 与 [迁移指南](docs/migration-guide.md)

---

## 2. 必读文档（按优先级）

接手任何任务前，先读以下文档：

| 优先级 | 文档 | 路径 | 内容 |
|---|---|---|---|
| ★★★ | 进度文档 | [docs/PROGRESS.md](docs/PROGRESS.md) | 当前进度 + 完整接手指南 + monorepo 布局 + provider 迁移清单 |
| ★★★ | 设计文档 | [docs/superpowers/specs/2026-07-07-aibridge-rust-rewrite-design.md](docs/superpowers/specs/2026-07-07-aibridge-rust-rewrite-design.md) | 架构、数据模型、FFI 边界、异步桥接、错误处理、适配器迁移策略 |
| ★★ | 实现计划 | [docs/superpowers/plans/2026-07-07-aibridge-implementation-plan.md](docs/superpowers/plans/2026-07-07-aibridge-implementation-plan.md) | 阶段 0-3 任务分解、多 agent 编排、里程碑 |
| ★★ | 迁移指南 | [docs/migration-guide.md](docs/migration-guide.md) | Python v1 -> v2 破坏性升级对照与示例 |
| ★ | README | [README.md](README.md) | 五语言快速开始 + provider 列表（面向用户） |

> 文档网站（mkdocs-material）构建自 `docs/`，配置见 `mkdocs.yml`。
> `docs/0[1-5]-*.md` 是 Python v1 旧文档，已通过 `exclude_docs` 排除出网站，仅在 git 中保留作归档。

---

## 3. Monorepo 布局

```
agn-sdk/
├── crates/
│   ├── aibridge-core/      # Rust 核心（纯逻辑，无 FFI）：error/config/http/retry/model/adapter/client/router
│   ├── aibridge-ffi/       # C ABI cdylib（给 Go/JVM/.NET）：全局 tokio runtime + 句柄 + JSON 边界 + cbindgen
│   ├── aibridge-python/    # PyO3 绑定（直连 core，原生 async）
│   └── aibridge-node/      # napi-rs 绑定（直连 core，原生 async）
├── bindings/
│   ├── go/                 # CGO 调 ffi
│   ├── jvm/                # JNA 调 ffi（Java/Kotlin）
│   └── dotnet/             # P/Invoke 调 ffi（C#）
├── docs/                   # 设计文档 + 计划 + 迁移指南 + 进度文档（mkdocs 网站）
├── examples/               # 五语言 hello world（echo adapter）+ v1 Python 示例（归档）
├── agn/                    # Python v1 旧代码（归档，勿在此改 v2 逻辑）
├── tests/                  # v1 Python 测试（归档）
├── Cargo.toml              # Rust workspace 根
├── pyproject.toml          # v1 Python 包配置（归档）
├── mkdocs.yml              # 文档网站配置
├── README.md               # v2 项目说明（面向用户）
├── README_v1.md            # v1 Python 项目说明（归档）
└── AGENTS.md               # 本文件
```

### 架构要点

- **Python / JS-TS 直连 Rust 核心**：PyO3 / napi-rs 直连 `aibridge-core`，无 JSON 序列化边界，享真正原生 async
- **Go / JVM / .NET 走 C ABI**：通过 `aibridge-ffi` 的 C ABI（句柄 + JSON 边界 + 全局 tokio runtime），各语言用原生异步原语包装
- **五语言共享同一个 Rust 核心**，绑定层都薄，行为一致
- **新增 provider**：在 `crates/aibridge-core/src/adapters/` 实现 trait 并在工厂 match 中注册（详见设计文档与现有适配器）

---

## 4. 构建 / 测试速查

```bash
# Rust 核心 + ffi 全量构建
cargo build --workspace

# 核心单测（badge 显示 1448+）
cargo test -p aibridge-core

# Python 绑定（开发安装）
pip install maturin
maturin develop -m crates/aibridge-python/Cargo.toml
python examples/hello_python.py

# Node 绑定
cd crates/aibridge-node && npm install && napi build && cd ../..
node examples/hello_node.js

# Go / JVM / .NET 绑定需先产 libaibridge 动态库
cargo build -p aibridge-ffi
cd bindings/go && CGO_ENABLED=1 DYLD_LIBRARY_PATH=../../target/debug go run ./example
cd bindings/jvm && ./gradlew run
cd bindings/dotnet && dotnet run
```

> 各绑定的运行细节（环境变量、动态库加载路径）见 [docs/PROGRESS.md](docs/PROGRESS.md) 与各 `bindings/*/` 目录 README。

---

## 5. v1 归档说明（勿改 v2 逻辑时误入）

以下内容属于 Python v1，仅作历史参考，**不要在其上做 v2 改动**：

- `agn/` - v1 Python SDK 核心代码
- `tests/` - v1 Python 测试
- `pyproject.toml` - v1 Python 包配置（`agn-sdk` PyPI 包）
- `examples/*.py`（非 `hello_*`）- v1 Python 示例
- `docs/01-overview.md` ~ `docs/05-project-structure.md` - v1 设计文档
- `README_v1.md` - v1 项目说明

v1 -> v2 迁移对照见 [docs/migration-guide.md](docs/migration-guide.md)。

---

## 6. 工作规范要点

- **中文优先**：所有回复、注释、文档、提交信息使用中文（技术标识符除外）
- **不可变模式**：创建新对象而非原地修改
- **错误处理**：每层显式处理，统一用 `aibridge-core` 的错误类型（详见设计文档错误处理章节）
- **测试**：新功能先写测试（TDD），核心单测覆盖率 ≥ 80%
- **提交格式**：`<type>: <description>`，类型 feat/fix/refactor/docs/test/chore/perf/ci
- **归属**：全局已禁用 Co-authored-by，提交信息勿加 attribution 行

---

## 7. 接手新任务的建议流程

1. 读本文件 + [docs/PROGRESS.md](docs/PROGRESS.md) 了解现状与接手指南
2. 按任务类型读对应文档（改架构读设计文档；迁移 provider 读迁移指南 + 现有适配器）
3. 用 `echo` 适配器（免认证、不调网络）本地验证管线
4. 改完跑 `cargo test -p aibridge-core` + 对应绑定构建
5. 提交前确认未误改 v1 归档代码

---

**最后提醒**：本项目已全面转向 v2（Rust）。除非任务明确要求改 v1，否则所有改动都应在 `crates/` 与 `bindings/` 下进行。
