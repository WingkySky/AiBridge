# AGN-SDK 项目引导 - Agent 执行前必读

> 本文件是 Agent 执行任务时必须首先加载和理解的项目指南，包含项目目标、架构、规范、验收标准等关键信息。

> ⚠️ **Rust 重构进行中**：本项目正在用 Rust 重构为跨语言 SDK `aibridge`（分支 `feat/aibridge-rust-rewrite`）。
> 接手 Rust 重构工作请先读 **[docs/PROGRESS.md](docs/PROGRESS.md)**（完整进度 + 接手指南）+ [设计文档](docs/superpowers/specs/2026-07-07-aibridge-rust-rewrite-design.md) + [实现计划](docs/superpowers/plans/2026-07-07-aibridge-implementation-plan.md)。
> 本文档（AGENTS.md）描述的是 Python 旧版（agn-sdk v1），Rust 新版以 PROGRESS.md + 设计文档为准。

---

## 1. 项目概述

### 1.1 项目名称
**AGN-SDK** - 多模型统一接口 SDK

### 1.2 项目目标
打造一个面向开发者的、轻量级的、可扩展的多模态 AI 统一接口 SDK：

1. **统一接口**：一套 API 调用所有 AI 模型（文本对话、图像生成、视频生成、语音处理、文本嵌入）
2. **零学习成本**：熟悉 OpenAI API 的开发者可以直接上手
3. **高度可扩展**：新增模型只需要实现一个适配器
4. **异步优先**：全异步设计，高性能，适合高并发场景
5. **生产级可靠**：内置重试、错误映射、参数归一化等企业级特性

### 1.3 技术栈
- **语言**：Python 3.10+
- **核心依赖**：
  - httpx[http2]>=0.27.0 - 异步 HTTP 客户端
  - pydantic>=2.0.0 - 数据校验与模型定义
  - python-dotenv>=1.0.0 - 环境变量管理
  - tenacity>=8.0.0 - 重试机制
  - anyio>=4.0.0 - 异步工具

---

## 2. 项目架构

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────┐
│                    API 层 (Client)                       │
│  - chat() / image_generate() / video_create()           │
│  - transcribe() / speech() / embed()                    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    路由器层 (Router)                     │
│  - 模型路由、负载均衡、Fallback                          │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    适配器层 (Adapters)                   │
│  - 各模型提供商的具体实现                                │
│  - BaseAdapter (抽象基类)                               │
│  - AdapterFactory (适配器工厂)                          │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    核心层 (Core)                         │
│  - HTTP 客户端、重试机制、错误处理                        │
│  - 配置管理、工具函数                                    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    数据模型层 (Models)                   │
│  - Pydantic 模型定义（统一数据结构）                     │
└─────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
agn-sdk/
├── agn/                          # SDK 核心代码
│   ├── __init__.py               # SDK 入口，导出核心类
│   ├── client.py                 # 统一客户端（API 层）
│   ├── router.py                 # 路由器（路由层）
│   ├── adapters/                 # 适配器层
│   │   ├── __init__.py
│   │   ├── base.py               # 适配器基类（抽象接口）
│   │   ├── factory.py            # 适配器工厂
│   │   ├── agnes.py              # Agnes AI 适配器
│   │   ├── openai.py             # OpenAI 适配器
│   │   ├── azure.py              # Azure OpenAI 适配器
│   │   ├── gemini.py             # Google Gemini 适配器
│   │   ├── anthropic.py          # Anthropic Claude 适配器
│   │   ├── runway.py             # Runway 适配器
│   │   ├── pika.py               # Pika 适配器
│   │   ├── kling.py              # 可灵适配器
│   │   ├── stability.py          # Stability AI 适配器
│   │   ├── chinese.py            # 中文模型聚合适配器
│   │   └── ...                   # 其他适配器
│   ├── core/                     # 核心层
│   │   ├── __init__.py
│   │   ├── http_client.py        # HTTP 客户端
│   │   ├── retry.py              # 重试机制封装
│   │   ├── errors.py             # 错误定义与映射
│   │   ├── config.py             # 配置管理
│   │   └── utils.py              # 工具函数
│   └── models/                   # 数据模型
│       ├── __init__.py
│       ├── common.py             # 通用数据结构
│       ├── chat.py               # 文本对话相关模型
│       ├── image.py              # 图像生成相关模型
│       ├── video.py              # 视频生成相关模型
│       ├── audio.py              # 语音相关模型
│       └── options.py            # 统一选项类与参数映射
├── docs/                         # 项目文档
│   ├── 01-overview.md            # 项目概述与目标
│   ├── 02-architecture.md        # 技术架构设计
│   ├── 03-api-reference.md       # API 设计规范
│   ├── 04-models-roadmap.md      # 兼容模型路线图
│   └── 05-project-structure.md   # 项目结构与开发计划
├── tests/                        # 测试目录
│   ├── __init__.py
│   ├── conftest.py               # pytest 配置与 fixtures
│   ├── test_client.py            # Client 测试
│   ├── test_router.py            # Router 测试
│   ├── test_unified_interface.py # 统一接口测试
│   ├── test_adapters/            # 适配器测试
│   │   └── test_*.py
│   └── test_core/                # 核心层测试
│       └── test_*.py
├── examples/                     # 使用示例
│   ├── basic_usage.py
│   ├── multi_provider.py
│   └── audio_usage.py
├── pyproject.toml                # Python 项目配置
├── README.md                     # 项目说明
├── CHANGELOG.md                  # 变更日志
└── AGENTS.md                     # 本文件 - Agent 项目引导
```

### 2.3 各层职责

| 层级 | 文件位置 | 职责 |
|------|---------|------|
| **API 层** | `agn/client.py`, `agn/__init__.py` | 对外统一接口，用户直接调用 |
| **路由器层** | `agn/router.py` | 模型选择、路由分发、负载均衡、Fallback |
| **适配器层** | `agn/adapters/` | 各模型提供商的具体实现，参数转换和响应归一化 |
| **核心层** | `agn/core/` | 通用能力（HTTP、重试、错误、配置、工具） |
| **数据模型层** | `agn/models/` | Pydantic 数据模型定义，统一数据结构 |

---

## 3. 文档路径

详细文档存放在 `docs/` 目录下：

| 文档 | 路径 | 内容 |
|------|------|------|
| 项目概述 | [01-overview.md](file:///Users/skywing/agn-sdk/docs/01-overview.md) | 项目背景、目标、技术选型、设计原则 |
| 架构设计 | [02-architecture.md](file:///Users/skywing/agn-sdk/docs/02-architecture.md) | 分层架构、核心模块、数据模型、扩展机制 |
| API 参考 | [03-api-reference.md](file:///Users/skywing/agn-sdk/docs/03-api-reference.md) | API 设计规范、使用示例、参数说明 |
| 模型路线图 | [04-models-roadmap.md](file:///Users/skywing/agn-sdk/docs/04-models-roadmap.md) | 兼容模型规划 |
| 项目结构 | [05-project-structure.md](file:///Users/skywing/agn-sdk/docs/05-project-structure.md) | 目录结构、开发计划、编码规范、发布流程 |

**重要参考文件**：
- [base.py](file:///Users/skywing/agn-sdk/agn/adapters/base.py) - 适配器基类，新增适配器必须参考
- [errors.py](file:///Users/skywing/agn-sdk/agn/core/errors.py) - 标准错误类型定义
- [options.py](file:///Users/skywing/agn-sdk/agn/models/options.py) - 统一选项类与参数映射机制
- [conftest.py](file:///Users/skywing/agn-sdk/tests/conftest.py) - 测试 fixtures 参考

---

## 4. 开发规范

### 4.1 代码风格规范

#### 命名规范
- **类名**：PascalCase（首字母大写，每个单词首字母大写）
  ```python
  class Client: ...
  class AgnesAdapter: ...
  ```
- **方法名/函数名/变量名/参数名**：snake_case（全小写，下划线分隔）
  ```python
  async def image_generate(self, prompt: str) -> ImageGenerationResult: ...
  ```
- **常量名**：UPPER_SNAKE_CASE（全大写，下划线分隔）
  ```python
  DEFAULT_TIMEOUT = 300
  ```
- **私有方法/变量**：单下划线前缀 `_private_method`

#### 类型提示规范（强制）
- 所有函数和方法必须有完整的类型提示
- 使用 PEP 604 语法：`str | None` 而不是 `Optional[str]`
- 返回类型必须明确标注
- Pydantic 模型使用现代类型注解

```python
# 正确示例
async def chat(
    self,
    model: str,
    messages: list[ChatMessage],
    temperature: float = 0.7,
    **kwargs: Any,
) -> ChatCompletion:
    ...
```

### 4.2 代码格式化与检查

项目配置了以下工具，提交前必须通过检查：

```bash
# 代码格式化（black，行宽 88）
black agn/

# 代码检查（ruff）
ruff check agn/

# 类型检查（mypy，strict 模式）
mypy agn/
```

**配置参考**（pyproject.toml）：
- black：line-length=88，target-version=py310
- ruff：启用 E/W/F/I/UP/B/C4/ASYNC 规则
- mypy：strict=true，disallow_untyped_defs=true

### 4.3 注释与文档规范

**重要**：代码的功能模块必须有备注信息（用户规则要求）

- 每个文件开头必须有模块文档字符串（docstring），说明模块用途
- 每个类必须有类文档字符串，说明类的职责
- 每个公开方法必须有文档字符串，说明功能、参数、返回值
- 关键逻辑段落需要有行内注释说明
- 调整或修复代码时，必须检查是否删除了原有的功能模块备注，如果删掉需要补充回来

```python
"""
模块文档字符串：说明本模块的用途和主要内容
"""

class SomeClass:
    """
    类文档字符串：说明类的职责和主要功能
    
    属性:
        attr1: 属性1说明
        attr2: 属性2说明
    """
    
    async def some_method(self, param1: str, param2: int) -> ResultType:
        """
        方法文档字符串：说明方法功能
        
        Args:
            param1: 参数1说明
            param2: 参数2说明
            
        Returns:
            返回值说明
            
        Raises:
            SomeError: 异常说明
        """
        # 关键逻辑注释
        result = await self._do_something(param1)
        return result
```

### 4.4 导入规范
导入顺序：标准库 → 第三方库 → 项目内部
```python
# 1. 标准库
import asyncio
from typing import Any

# 2. 第三方库
import httpx
from pydantic import BaseModel

# 3. 项目内部
from agn.core.errors import AGNError
from agn.models.chat import ChatCompletion
```

### 4.5 适配器开发规范

新增 Provider 适配器必须遵循以下规范：

1. **继承基类**：必须继承 `BaseAdapter`
2. **注册工厂**：文件末尾必须调用 `AdapterFactory.register()`
3. **声明能力**：必须设置 `provider_type`、`provider_name`、`supported_capabilities`
4. **实现抽象方法**：必须实现 `start()`、`close()`、`chat()`、`image_generate()`、`video_create()`、`video_poll()`、`list_models()`
5. **参数映射**：使用 `_map_params()` 或自定义 `param_mapping` 处理参数转换
6. **错误处理**：使用 `_handle_http_error()` 统一处理 HTTP 错误
7. **响应归一化**：所有响应必须转换为标准 Pydantic 模型
8. **不支持能力**：不支持的能力直接继承基类默认实现（抛出 UnsupportedCapabilityError）

参考示例：
```python
"""
XX Provider 适配器

实现 XX AI 平台的 API 调用，支持对话、图像生成等能力。
"""

from .base import BaseAdapter
from .factory import AdapterFactory
from agn.models.chat import ChatCompletion
# ... 其他导入

class XXAdapter(BaseAdapter):
    """XX AI 适配器"""
    
    provider_type = "xx"
    provider_name = "XX AI"
    supported_capabilities = [Capabilities.CHAT, Capabilities.IMAGE_GENERATE]
    
    def __init__(self, config):
        super().__init__(config)
        # 初始化逻辑
    
    async def start(self) -> None:
        """启动适配器，初始化 HTTP 客户端"""
        # 实现逻辑
    
    async def close(self) -> None:
        """关闭适配器，释放资源"""
        # 实现逻辑
    
    # ... 实现其他抽象方法

AdapterFactory.register("xx", XXAdapter)
```

### 4.6 错误处理规范

1. **使用标准错误类型**：必须使用 `agn.core.errors` 中定义的错误类型
2. **错误映射**：每个适配器负责将 Provider 特定错误映射到标准错误
3. **保留原始错误**：使用 `from e` 保留错误链，便于排查
4. **清晰错误信息**：错误信息应包含原因和上下文

标准错误类型：
- `AGNError` - SDK 基础错误
- `AuthenticationError` - 认证错误（API Key 无效）
- `RateLimitError` - 限流错误
- `ValidationError` - 参数校验错误
- `ModelNotFoundError` - 模型不存在
- `APIError` - API 调用错误
- `TimeoutError` - 超时错误
- `UnsupportedCapabilityError` - 不支持的能力

---

## 5. 测试规范

### 5.1 测试框架
- **测试框架**：pytest + pytest-asyncio
- **异步测试**：`asyncio_mode = "auto"`
- **测试覆盖率**：目标 ≥ 80%

### 5.2 测试目录结构
- `tests/conftest.py` - 公共 fixtures
- `tests/test_client.py` - Client 测试
- `tests/test_router.py` - Router 测试
- `tests/test_adapters/` - 各适配器测试
- `tests/test_core/` - 核心模块测试

### 5.3 测试命令
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_adapters/test_agnes.py -v

# 运行测试并显示覆盖率
pytest --cov=agn --cov-report=term-missing

# 生成 HTML 覆盖率报告
pytest --cov=agn --cov-report=html
```

### 5.4 测试编写规范

1. **测试命名**：`test_<功能名>`，如 `test_chat_completion`
2. **异步测试**：测试函数使用 `async def` 定义
3. **使用 fixtures**：复用 conftest.py 中定义的 fixtures（mock_api_key, sample_chat_messages 等）
4. **Mock 外部调用**：HTTP 请求必须 mock，不调用真实 API
5. **覆盖边界情况**：正常流程、异常流程、边界条件都要覆盖
6. **测试独立**：测试之间不依赖，可单独运行

测试示例参考：
```python
"""
XX 适配器测试

测试 XXAdapter 的各项功能。
"""
import pytest
from agn.adapters.xx import XXAdapter

@pytest.mark.asyncio
async def test_chat(mock_provider_config, sample_chat_messages):
    """测试文本对话功能"""
    adapter = XXAdapter.from_config(mock_provider_config)
    await adapter.start()
    
    # Mock HTTP 响应...
    
    result = await adapter.chat(model="test-model", messages=sample_chat_messages)
    assert isinstance(result, ChatCompletion)
    assert len(result.choices) > 0
    
    await adapter.close()
```

---

## 6. 验收方式和标准

### 6.1 代码修改验收清单

完成任何代码修改后，必须逐项检查以下内容：

#### ✅ 代码质量检查
- [ ] 所有公开类、方法、模块都有文档字符串（备注）
- [ ] 所有函数/方法都有完整的类型提示
- [ ] 代码通过 black 格式化：`black agn/`
- [ ] 代码通过 ruff 检查：`ruff check agn/`
- [ ] 代码通过 mypy 类型检查：`mypy agn/`
- [ ] 没有删除原有的功能模块备注（如有删除必须补充）

#### ✅ 功能验收
- [ ] 新增功能按照现有架构模式实现
- [ ] 适配器继承 BaseAdapter 并注册到工厂
- [ ] 错误映射到标准错误类型
- [ ] 响应归一化为标准 Pydantic 模型
- [ ] 参数处理符合统一规范

#### ✅ 测试验收
- [ ] 为新功能编写了对应的单元测试
- [ ] 所有现有测试通过：`pytest`
- [ ] 新增测试覆盖正常流程和异常流程
- [ ] 测试覆盖率不降低

#### ✅ 文档验收（如需要）
- [ ] README 或相关文档已更新
- [ ] API 变更已在文档中反映
- [ ] 示例代码（如需要）已提供

### 6.2 新增适配器验收标准

新增 Provider 适配器必须满足：

1. **文件位置**：`agn/adapters/<provider_name>.py`
2. **类定义**：继承 `BaseAdapter`，类名 `<ProviderName>Adapter`
3. **类变量**：
   - `provider_type`：小写唯一标识
   - `provider_name`：显示名称
   - `supported_capabilities`：使用 `Capabilities` 常量声明支持的能力
4. **必须实现的方法**：
   - `start()` - 初始化 HTTP 客户端
   - `close()` - 释放资源
   - `chat()` - 文本对话（如支持）
   - `image_generate()` - 图像生成（如支持）
   - `video_create()` - 创建视频任务（如支持）
   - `video_poll()` - 查询视频状态（如支持）
   - `list_models()` - 列出可用模型
5. **注册**：文件末尾调用 `AdapterFactory.register()`
6. **测试**：在 `tests/test_adapters/` 下有对应测试文件
7. **导出**：在 `agn/adapters/__init__.py` 中导入（确保自动注册）

### 6.3 Bug 修复验收标准

1. **复现**：有复现步骤或测试用例证明 bug 存在
2. **修复**：修复代码不引入新问题，不破坏现有功能
3. **回归测试**：所有现有测试通过
4. **测试用例**：添加测试用例防止 bug 复发

### 6.4 命令验证流程

提交代码前，按顺序执行以下命令确保全部通过：

```bash
# 1. 代码格式化
black agn/ tests/

# 2. 代码检查
ruff check agn/ tests/

# 3. 类型检查
mypy agn/

# 4. 运行所有测试
pytest

# 5. （可选）查看覆盖率
pytest --cov=agn --cov-report=term-missing
```

---

## 7. 工作流程

### 7.1 开始任务前
1. 首先阅读本文件（AGENTS.md）理解项目规范
2. 查看相关文档和现有代码实现模式
3. 参考类似功能的实现（如新增适配器先看现有适配器）

### 7.2 开发中
1. 遵循编码规范
2. 边开发边写测试（TDD 优先）
3. 保持功能模块备注完整
4. 不破坏现有测试

### 7.3 完成任务后
1. 运行验收清单中的所有检查
2. 运行所有验证命令确保通过
3. 确认没有遗漏的备注信息

---

## 8. 常见问题参考

### Q: 新增一个最简单的适配器需要做什么？
A: 参考现有适配器（如 [agnes.py](file:///Users/skywing/agn-sdk/agn/adapters/agnes.py)），创建继承 BaseAdapter 的类，实现必要方法，注册到工厂，写测试。

### Q: 参数如何做映射？
A: 参考 [options.py](file:///Users/skywing/agn-sdk/agn/models/options.py) 中的 `ParameterMapping`，可以定义通用参数到厂商特定参数的映射规则。OpenAI 兼容的适配器可以直接使用 `OPENAI_COMPATIBLE_MAPPING`。

### Q: 如何处理不同模型的特有参数？
A: 使用 `**kwargs` 透传，适配器内部从 kwargs 中提取特有参数。

### Q: 视频生成的轮询机制怎么实现？
A: `video_create()` 返回 `VideoTask`（含 task_id），`video_poll()` 接收 task_id 返回 `VideoStatus`（含 status, video_url, progress, error）。

---

**最后提醒**：所有代码修改完成后，务必运行完整的测试和代码检查，确保符合本文件中的所有规范！
