# AGN-SDK / AIBridge 项目引导 - Agent 执行前必读

> 本文件是 Agent 执行任务时必须首先加载的项目指南。
> v2.0.0 已合并至 `main`：项目当前主线是 **AIBridge**（Rust 核心 + 五语言原生绑定）。
> Python v1（`agn-sdk`）已归档并从仓库移除（完整代码见 git tag `v1.3.3`）。

---

## 1. 项目现状

- **当前主线**：AIBridge v2 - 跨语言 AI 统一接口 SDK（Rust 核心 + Python / JS-TS / Go / JVM / .NET 五语言原生绑定）
- **版本**：`2.1.1`（见 `Cargo.toml`），阶段 3 发布收尾中
- **能力**：chat（含流式）/ image / video / TTS / ASR / embed，38 个真实 provider + 1 个 mock（echo）
- **v1 状态**：Python v1（`agn-sdk`）已全量迁移至 v2，旧代码已于 2026-08-30 从仓库移除（git tag `v1.3.3` 可随时找回）。老用户参考 [迁移指南](docs/migration-guide.md)

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
> `docs/superpowers/` 是原始 spec/plan 归档（已复制为 `docs/design.md` 与 `docs/plan.md`），通过 `exclude_docs` 排除出网站。

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
├── examples/               # 五语言 hello world（echo adapter）+ 真实 provider 冒烟脚本
├── tests/                  # v2 测试：Python 绑定 e2e + 跨语言一致性探针（tests/consistency/）
├── Cargo.toml              # Rust workspace 根
├── pytest.ini              # Python 绑定测试配置
├── mkdocs.yml              # 文档网站配置
├── README.md               # v2 项目说明（面向用户）
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

## 5. v1 归档说明

Python v1（`agn-sdk`）已全量迁移至 v2（38 provider + 六大能力零丢失），相关旧文件已于 2026-08-30 从仓库移除：

- **已移除**：`agn/`（v1 包）、v1 pytest 测试（`tests/test_adapters/` 等）、v1 示例（`examples/basic_usage.py` 等 3 个）、根 `pyproject.toml`（v1 flit 打包配置）、`uv.lock`、`README_v1.md`、`docs/01~05`（v1 旧文档）
- **找回方式**：`git checkout v1.3.3`（tag 完整保留 v1 代码）
- **PyPI**：已发布的 `agn-sdk` 包不受影响（与仓库无关）
- v1 -> v2 迁移对照见 [docs/migration-guide.md](docs/migration-guide.md)

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
5. 提交前跑 `git status` 确认没有无关文件变动

---

**最后提醒**：本项目主线是 v2（Rust）。所有功能改动都应在 `crates/` 与 `bindings/` 下进行。
