# 🤖 AGN-SDK 多模型统一接口 SDK

[![Stars](https://img.shields.io/github/stars/your-org/agn-sdk?style=flat)](https://github.com/)
[![Forks](https://img.shields.io/github/forks/your-org/agn-sdk?style=flat)](https://github.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![Async](https://img.shields.io/badge/-Async-00-stroke?logo=asyncio&logoColor=white)

> **统一接口** | **5+ 服务商** | **异步优先** | **生产级可靠** | **类型安全**

---

<div align="center">

**🌐 Language / 语言**

[English](README.md) | [**中文**](README_zh.md)

</div>

---

**一套 API 调用所有 AI 模型 —— 无论是文本对话、图像生成、视频制作还是语音合成。**

采用异步优先设计、全类型安全、插件化适配器架构。如果你熟悉 OpenAI API，可以立即上手使用 AGN-SDK。

---

## ✨ 功能特性

### 支持的能力

| 能力 | 说明 | 状态 |
| --- | --- | --- |
| 💬 **文本对话** | 与 AI 模型进行多轮对话 | ✅ 稳定 |
| 🖼️ **图像生成** | 文生图功能 | ✅ 稳定 |
| 🎬 **视频生成** | 异步视频生成，支持轮询 | ✅ 稳定 |
| 🔊 **语音合成** | 文本转语音 | 🚧 计划中 |
| 🎤 **语音识别** | 音频转录 | 🚧 计划中 |
| 📊 **文本嵌入** | 文本向量嵌入 | 🚧 计划中 |

### 架构亮点

- **统一接口** — 一套 API 调用所有 AI 服务商（OpenAI、Azure、Anthropic、Gemini 等）
- **异步优先** — 全异步/await 支持，基于 `httpx` 和 `anyio`
- **适配器模式** — 只需实现一个适配器类即可添加新的服务商
- **类型安全** — 所有数据模型使用 Pydantic v2 定义，全程类型提示
- **生产级可靠** — 内置重试逻辑、错误映射、参数归一化
- **OpenAI 兼容** — 直接使用 OpenAI API 模式，学习成本极低

---

## 📦 支持的服务商

### V1.0（稳定版）

| 服务商 | 对话 | 生图 | 视频 | 基础地址 |
| ------ | --- | --- | --- | -------- |
| **Agnes AI** | ✅ | ✅ | ✅ | `https://api.agnes.ai/v1` |
| **OpenAI** | ✅ | ✅ | — | `https://api.openai.com/v1` |
| **Azure OpenAI** | ✅ | ✅ | — | Azure 端点 |

### V1.1+（计划中）

| 服务商 | 对话 | 生图 | 视频 |
| ------ | --- | --- | --- |
| Anthropic (Claude) | ✅ | — | — |
| Google Gemini | ✅ | ✅ | — |
| Runway | — | — | ✅ |
| Pika | — | — | ✅ |
| Stability AI | — | ✅ | — |
| 字节 Seedance | ✅ | ✅ | ✅ |

---

## 📦 项目结构

```
agn-sdk/
├── agn/                              # SDK 核心代码
│   ├── __init__.py                   # SDK 入口
│   ├── client.py                     # 统一客户端（API 层）
│   ├── router.py                     # 路由器（路由层）
│   ├── adapters/                     # 适配器实现
│   │   ├── base.py                   # BaseAdapter 抽象类
│   │   ├── factory.py                # 适配器工厂
│   │   ├── agnes.py                  # Agnes AI 适配器
│   │   ├── openai.py                 # OpenAI 适配器
│   │   ├── azure.py                  # Azure OpenAI 适配器
│   │   └── ...                       # 更多适配器
│   ├── core/                         # 核心工具
│   │   ├── http_client.py            # 异步 HTTP 客户端
│   │   ├── retry.py                  # 重试机制
│   │   ├── errors.py                 # 错误定义
│   │   ├── config.py                 # 配置管理
│   │   └── utils.py                  # 工具函数
│   └── models/                       # Pydantic 数据模型
│       ├── common.py                 # 通用结构
│       ├── chat.py                   # 对话模型
│       ├── image.py                  # 图像模型
│       ├── video.py                  # 视频模型
│       └── options.py                # 统一选项
├── docs/                             # 项目文档
│   ├── 01-overview.md                # 项目概述
│   ├── 02-architecture.md            # 架构设计
│   └── 03-api-reference.md           # API 参考
├── tests/                            # 测试套件
├── examples/                          # 使用示例
├── pyproject.toml                    # 项目配置
└── README.md                         # 项目文档（英文）
```

---

## 🚀 快速开始

3 步启动：

### 步骤 1：安装

```bash
# 从 PyPI 安装（即将上线）
pip install agn-sdk

# 或从源码安装（开发模式）
git clone https://github.com/your-org/agn-sdk.git
cd agn-sdk
pip install -e .
```

### 步骤 2：配置 API Key

```bash
# 选项 A — 环境变量（推荐）
export AGN_API_KEY='your-api-key'
export AGN_BASE_URL='https://api.agnes.ai/v1'  # 服务商特定的基础地址

# 选项 B — .env 文件（自动加载）
echo "AGN_API_KEY=your-api-key" > .env
echo "AGN_BASE_URL=https://api.agnes.ai/v1" >> .env

# 选项 C — 代码中传入
client = Client(provider="agnes", api_key="your-key", base_url="https://api.agnes.ai/v1")
```

### 步骤 3：调用 AI 模型

```python
import asyncio
from agn import Client

async def main():
    # 创建客户端
    client = Client(
        provider="agnes",
        api_key="your-api-key",
        base_url="https://api.agnes.ai/v1",
    )
    
    # 💬 文本对话
    response = await client.chat(
        model="claude-3-opus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        temperature=0.7,
    )
    print(response.choices[0].message.content)
    
    # 🖼️ 图像生成
    result = await client.image_generate(
        model="dall-e-3",
        prompt="A beautiful sunset over the ocean",
        size="1024x1024",
        quality="hd",
    )
    print(result.data[0].url)
    
    # 🎬 视频生成（异步 + 轮询）
    task = await client.video_create(
        model="video-gen-1",
        prompt="A cat walking in the garden",
        width=1280,
        height=720,
        num_frames=121,
    )
    
    # 轮询视频状态直至完成
    while True:
        status = await client.video_poll(task.task_id)
        print(f"Status: {status.status}, Progress: {status.progress}%")
        if status.status in ("completed", "failed"):
            break
    
    print(f"Video URL: {status.video_url}")

if __name__ == "__main__":
    asyncio.run(main())
```

✨ **完成！** 你现在拥有了一个统一的接口来调用所有支持的 AI 服务商。

---

## 📖 完整使用参考

### 文本对话

```python
response = await client.chat(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,        # 随机性 (0.0-2.0)
    max_tokens=1000,        # 最大响应 token 数
    top_p=1.0,              # 核采样
    frequency_penalty=0.0,  # 重复惩罚
    presence_penalty=0.0,  # 主题多样性
    stream=False,           # 流式响应
)
print(response.choices[0].message.content)
```

### 图像生成

```python
result = await client.image_generate(
    model="dall-e-3",
    prompt="A futuristic city with flying cars",
    size="1024x1024",      # 1024x1024, 1024x1792, 1792x1024
    quality="hd",           # standard 或 hd
    style="vivid",          # vivid 或 natural (DALL-E 3)
    n=1,                    # 生成图片数量
)
print(result.data[0].url)  # 或 result.data[0].b64_json
```

### 视频生成

```python
# 创建视频任务
task = await client.video_create(
    model="video-gen-1",
    prompt="A dramatic sword fight scene",
    width=1280,
    height=720,
    num_frames=121,         # 必须满足 8n+1（如 33、49、81、121、241）
    frame_rate=24,
    seed=42,                # 可选：用于可重复性
)
print(f"Task ID: {task.task_id}")

# 轮询直至完成
status = await client.video_poll(task.task_id)
while status.status == "in_progress":
    await asyncio.sleep(5)
    status = await client.video_poll(task.task_id)
    
print(f"Video URL: {status.video_url}")
```

---

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    API 层（Client）                     │
│            chat() / image_generate() / video_create()   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    路由层（Router）                      │
│              模型路由、负载均衡、故障转移                 │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    适配器层（Adapter）                   │
│       BaseAdapter → AgnesAdapter / OpenAIAdapter / ...   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    核心层（Core）                       │
│          HTTP 客户端、重试、错误、配置、工具              │
└─────────────────────────────────────────────────────────┘
```

- **API 层** — 统一的 `Client` 类，用户直接调用的接口
- **路由层** — 模型选择、路由分发、负载均衡
- **适配器层** — 服务商特定实现、参数映射、响应归一化
- **核心层** — 共享工具（HTTP、重试、错误、配置）

---

## 📋 适配器开发

添加新的 AI 服务商非常简单：

1. **创建适配器** — 继承 `BaseAdapter`，实现必要的方法
2. **注册工厂** — 调用 `AdapterFactory.register("provider_name", YourAdapter)`
3. **声明能力** — 设置 `supported_capabilities` 列表

```python
from agn.adapters.base import BaseAdapter
from agn.adapters.factory import AdapterFactory

class NewProviderAdapter(BaseAdapter):
    provider_type = "newprovider"
    provider_name = "New Provider"
    supported_capabilities = [Capabilities.CHAT, Capabilities.IMAGE_GENERATE]
    
    async def start(self) -> None:
        # 初始化 HTTP 客户端
        ...
    
    async def chat(self, model: str, messages: list[ChatMessage], **kwargs):
        # 实现对话逻辑
        ...
    
    # ... 实现其他方法

AdapterFactory.register("newprovider", NewProviderAdapter)
```

---

## 🧪 开发指南

```bash
# 克隆并设置
git clone https://github.com/your-org/agn-sdk.git
cd agn-sdk
python -m venv venv
source venv/bin/activate

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black agn/

# 代码检查
ruff check agn/

# 类型检查
mypy agn/
```

---

## 📜 许可证

MIT License
