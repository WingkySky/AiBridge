# AIBridge 阶段1.5 跨语言一致性测试

> 分支：`feat/aibridge-rust-rewrite`
> 日期：2026-07-07
> 范围：五语言错误 code 统一 + 四语言（Python/Node/Go/JVM）hello world 一致性验证。

## 1. 目标

1. **错误 code 统一**：五个语言绑定（Python / Node / Go / JVM / .NET）的错误码字符串完全对齐 Rust `aibridge-core/src/error.rs` 中 `AibridgeError::code()` 的实际返回值。
2. **跨语言一致性**：用 echo adapter（`provider="echo"`，免认证）跑 Python/Node/Go/JVM 的 hello world，断言四种能力的输出语义一致：
   - `chat`：`choices[0].message.content` == `"hello [echo]"`
   - `chat_stream`：3 个 chunk，拼接内容 == `"hello [echo]"`
   - `speech`：`audio_data` 长度 == 15
   - 错误：未知 provider 返回 code == `"provider_not_found"`
3. **.NET**：dotnet 工具链未安装，仅做代码级错误 code 对齐校验，不跑 hello。

## 2. 基准：Rust `AibridgeError::code()`

来源：`crates/aibridge-core/src/error.rs`（`code()` 方法，第 255-269 行）。

| 变体 | code() 返回值 |
|------|--------------|
| Authentication | `authentication_error` |
| RateLimit | `rate_limit_error` |
| Validation | `validation_error` |
| ModelNotFound | `model_not_found` |
| Api | `api_error` |
| Network | `network_error` |
| Timeout | `timeout_error` |
| UnsupportedCapability | `unsupported_capability` |
| ProviderNotFound | `provider_not_found` |
| VoiceNotAvailable | `voice_not_available` |
| ServiceUnavailable | `service_unavailable` |

FFI 层（`aibridge-ffi`）把上述 code 写入线程局部 `last_error` JSON 的 `code` 字段；运行时各绑定均从该字段读取，故运行时 code 天然对齐 Rust。本阶段修复的是各绑定**源码内硬编码的 code 字符串常量 / switch case / 子类构造传入的 code**，使其与 Rust 一致（避免兜底分支或子类 `Code` 属性与实际 last_error code 不一致）。

## 3. 错误 code 统一前后对比

### 3.1 Python（`crates/aibridge-python/src/lib.rs`）

- **机制**：`map_error()` 用 `err.code()` 拼接消息 `"[{code}] {err}"`，异常类层级映射。
- **改动**：无（已对齐）。

### 3.2 Node（`crates/aibridge-node/src/lib.rs` + `lib.js`）

- **机制**：`map_error()` 用 `err.code()` 编码 `[code] message`；`lib.js` 的 `withCode()` 解析出 `err.code` 属性。
- **改动**：无（已对齐）。

### 3.3 JVM（`bindings/jvm/src/main/java/io/aibridge/AibridgeException.java`）

- **机制**：`CODE_*` 常量 + `mapToException()` switch。常量已对齐 Rust。
- **改动**：无（已对齐）。

### 3.4 Go（`bindings/go/error.go`）— **已修改**

| 常量 | 修改前 | 修改后 |
|------|--------|--------|
| `errCodeAuthentication` | `"authentication"` | `"authentication_error"` |
| `errCodeRateLimit` | `"rate_limit"` | `"rate_limit_error"` |
| `errCodeNetwork` | `"network"` | `"network_error"` |
| `errCodeTimeout` | `"timeout"` | `"timeout_error"` |
| 其余常量 | （已对齐） | （不变） |

同时更新文件头注释与接口注释里的示例（`"rate_limit"` → `"rate_limit_error"`）。
注：这些常量当前未被业务代码引用（运行时 code 来自 FFI last_error JSON），但作为文档/兜底用途须与 Rust 一致。

### 3.5 .NET（`bindings/dotnet/AIBridge/AibridgeException.cs`）— **已修改**

| 位置 | 修改前 | 修改后 |
|------|--------|--------|
| `MapByCode` switch：authentication | `"authentication"` | `"authentication_error"` |
| `MapByCode` switch：rate_limit | `"rate_limit"` | `"rate_limit_error"` |
| `MapByCode` switch：timeout | `"timeout"` | `"timeout_error"` |
| `AuthenticationException` 构造 code | `"authentication"` | `"authentication_error"` |
| `RateLimitException` 构造 code | `"rate_limit"` | `"rate_limit_error"` |
| `TimeoutException_` 构造 code | `"timeout"` | `"timeout_error"` |

修改前 `MapByCode` 的 switch case 与子类构造传入的 code 字符串不一致（前者部分用短名，后者部分用短名），导致 `MapByCode` 命中后构造的子类 `Code` 属性与 last_error 的 code 不一致（例如 last_error code=`"rate_limit_error"`，但 `MapByCode` 旧 case `"rate_limit"` 不命中 → 落到状态码兜底分支 → `RateLimitException.Code` 仍为旧 `"rate_limit"`）。修改后两者完全对齐 Rust。

同时更新类头注释与 `Code` 属性文档注释。

## 4. 四语言 hello world 一致性验证

环境：Python 3.14 + maturin 1.14 / Node 25.5 / Go 1.26 / OpenJDK 21。
echo adapter：`chat` 回显最后一条 user 消息 + `" [echo]"`；`chat_stream` 产 3 chunk（role / 前半段 `"hello "` / 后半段 `"[echo]"` + `finish_reason="stop"`）；`speech` 返 15 字节 `b"mock-audio-data"`。

| 语言 | 脚本 | chat content | stream chunk 数 | stream 拼接 | speech 字节数 | 结果 |
|------|------|-------------|----------------|-------------|--------------|------|
| Python | `examples/hello_python.py` | `hello [echo]` | 3 | `hello [echo]` | 15 | ✅ 通过 |
| Node | `examples/hello_node.js` | `hello [echo]` | 3 | `hello [echo]` | 15 | ✅ 通过 |
| Go | `bindings/go/example/hello.go` | `hello [echo]` | 3 | `hello [echo]` | 15 | ✅ 通过 |
| JVM | `bindings/jvm` (`./gradlew run`，`Hello.java`) | `hello [echo]` | 3 | `hello [echo]` | 15 | ✅ 通过 |

四语言在 chat / chat_stream / speech 三项能力的输出语义完全一致。

## 5. 未知 provider 错误一致性验证

探针脚本（`tests/consistency/error_probe_*`）：构造 `Client(provider="nonexistent", api_key="dummy-key")`（假 key 跳过 key 校验，触发 `ProviderNotFound`），断言错误 code == `"provider_not_found"`。

| 语言 | 探针脚本 | 错误对象 | code 取值方式 | 实际 code | 结果 |
|------|---------|---------|--------------|----------|------|
| Python | `error_probe_python.py` | `ProviderNotFoundError` | 消息前缀 `[code]` | `provider_not_found` | ✅ |
| Node | `error_probe_node.js` | `Error` | `err.code` 属性 | `provider_not_found` | ✅ |
| Go | `error_probe_go.go` | `aibridge.AibridgeError` | `ae.Code()` | `provider_not_found` | ✅ |
| JVM | `ErrorProbe.java` | `AibridgeException` | `e.getCode()` | `provider_not_found` | ✅ |

四语言错误 code 完全一致，均等于 Rust `AibridgeError::ProviderNotFound.code()` 返回的 `"provider_not_found"`。

### 5.1 运行方式

```bash
# 前置：cargo build -p aibridge-ffi（产出 target/debug/libaibridge.dylib，须为 ffi 版本而非 PyO3 版本）

# Python（需先 maturin develop -m crates/aibridge-python/Cargo.toml）
.venv/bin/python tests/consistency/error_probe_python.py

# Node（需先在 crates/aibridge-node 下 npm install && npm run build）
node tests/consistency/error_probe_node.js

# Go
cd bindings/go
DYLD_LIBRARY_PATH=../../target/debug CGO_ENABLED=1 go run ../../tests/consistency/error_probe_go.go

# JVM（需先 ./gradlew classes 编译）
JVM_CLASSES=bindings/jvm/build/classes/java/main
JNA_JAR=$(find ~/.gradle/caches -name 'jna-5.15.0.jar' | head -1)
JACKSON_DATABIND=$(find ~/.gradle/caches -name 'jackson-databind-2.18.1.jar' | head -1)
JACKSON_CORE=$(find ~/.gradle/caches -name 'jackson-core-2.18.1.jar' | head -1)
JACKSON_ANN=$(find ~/.gradle/caches -name 'jackson-annotations-2.18.1.jar' | head -1)
mkdir -p /tmp/aibridge_probe
javac -cp "$JVM_CLASSES:$JNA_JAR" -d /tmp/aibridge_probe tests/consistency/ErrorProbe.java
DYLD_LIBRARY_PATH=target/debug java -Djna.library.path=target/debug \
  -cp "$JVM_CLASSES:$JNA_JAR:$JACKSON_DATABIND:$JACKSON_CORE:$JACKSON_ANN:/tmp/aibridge_probe" ErrorProbe
```

## 6. .NET 代码级对齐确认（未跑 hello）

dotnet 工具链未安装，仅做代码审查。`bindings/dotnet/AIBridge/AibridgeException.cs` 修改后：

- `MapByCode` switch 的 11 个 case 字符串与 Rust `code()` 11 个返回值逐一对应。
- 11 个子类构造函数传入的 code 字符串与各自 `MapByCode` case 一致。
- 状态码兜底分支（`FromStatus`）在 last_error 缺失时按 `AibridgeStatus` 映射子类，子类 `Code` 属性现已对齐 Rust。

## 7. 结论

- 五语言错误 code 已统一对齐 Rust `AibridgeError::code()` 实际返回值（Go / .NET 修改，Python / Node / JVM 原本对齐）。
- 四语言（Python / Node / Go / JVM）hello world 在 chat / chat_stream / speech / 未知 provider 错误四项上输出语义完全一致。
- .NET 代码级错误 code 已对齐（dotnet 工具链缺失，未跑运行时验证）。

## 8. 遗留问题

1. **dylib 产物名冲突**：`aibridge-ffi` 与 `aibridge-python` 的 `[lib] name = "aibridge"`，两者 `cargo build` 产物都落到 `target/debug/libaibridge.dylib`，互相覆盖。`maturin develop` 后再跑 Go/JVM 会因 dylib 变成 PyO3 版本（无 C FFI 符号）而链接失败。**当前规避**：跑 Go/JVM 前先 `touch crates/aibridge-ffi/src/lib.rs && cargo build -p aibridge-ffi` 重建 ffi 版本。**建议后续**：给 aibridge-python 的 cdylib 改名（如 `aibridge_python`）或在 workspace 层隔离产物路径。
2. **.NET 运行时验证缺失**：本机未装 dotnet SDK，.NET 仅做代码审查。建议在装了 dotnet 的环境补跑 `dotnet run` hello 与错误探针。
3. **Go 常量未被引用**：`bindings/go/error.go` 的 `errCode*` 常量当前无业务代码引用（运行时 code 来自 FFI last_error JSON）。建议后续在 Go 侧加一个 `MapByCode` 风格的子类型断言（或保留常量作文档）。
4. **一致性测试未纳入 CI**：本阶段的探针脚本为手动运行。建议后续接入 CI matrix（五语言并行跑 hello + 错误探针）。
