# 🤖 AGN-SDK

[![Stars](https://img.shields.io/github/stars/your-org/agn-sdk?style=flat)](https://github.com/)
[![Forks](https://img.shields.io/github/forks/your-org/agn-sdk?style=flat)](https://github.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![Async](https://img.shields.io/badge/-Async-00-stroke?logo=asyncio&logoColor=white)

> **Unified API** | **5+ Providers** | **Async-First** | **Production-Ready** | **Type-Safe**

---

<div align="center">

**🌐 Language / 语言**

[**English**](README.md) | [中文](README_zh.md)

</div>

---

**A unified SDK that calls all AI models through one API — whether it's text chat, image generation, video creation, or speech synthesis.**

Built with async-first design, full type safety, and a pluggable adapter architecture. If you're familiar with the OpenAI API, you can use AGN-SDK immediately.

---

## ✨ Features

### Capabilities

| Capability | Description | Status |
| ----------- | ----------- | ------ |
| 💬 **Chat Completion** | Multi-turn conversations with AI models | ✅ Stable |
| 🖼️ **Image Generation** | Text-to-image generation | ✅ Stable |
| 🎬 **Video Creation** | Async video generation with polling | ✅ Stable |
| 🔊 **Speech Synthesis** | Text-to-speech generation | 🚧 Coming Soon |
| 🎤 **Speech Recognition** | Audio transcription | 🚧 Coming Soon |
| 📊 **Embeddings** | Text embedding vectors | 🚧 Coming Soon |

### Architecture Highlights

- **Unified Interface** — One API to rule all AI providers (OpenAI, Azure, Anthropic, Gemini, etc.)
- **Async-First Design** — Full async/await support, built on `httpx` and `anyio`
- **Adapter Pattern** — Add new providers by implementing a single adapter class
- **Type Safety** — All data models defined with Pydantic v2, full type hints throughout
- **Production-Ready** — Built-in retry logic, error mapping, parameter normalization
- **OpenAI Compatible** — Use OpenAI API patterns directly, minimal learning curve

---

## 📦 Supported Providers

### V1.0 (Stable)

| Provider | Chat | Image | Video | Base URL |
|----------|------|-------|-------|----------|
| **Agnes AI** | ✅ | ✅ | ✅ | `https://api.agnes.ai/v1` |
| **OpenAI** | ✅ | ✅ | — | `https://api.openai.com/v1` |
| **Azure OpenAI** | ✅ | ✅ | — | Azure endpoint |

### V1.1+ (Coming Soon)

| Provider | Chat | Image | Video |
|----------|------|-------|-------|
| Anthropic (Claude) | ✅ | — | — |
| Google Gemini | ✅ | ✅ | — |
| Runway | — | — | ✅ |
| Pika | — | — | ✅ |
| Stability AI | — | ✅ | — |
| ByteDance Seedance | ✅ | ✅ | ✅ |

---

## 📦 Project Structure

```
agn-sdk/
├── agn/                              # SDK core code
│   ├── __init__.py                   # SDK entry point
│   ├── client.py                     # Unified client (API layer)
│   ├── router.py                     # Router (routing layer)
│   ├── adapters/                     # Adapter implementations
│   │   ├── base.py                   # BaseAdapter abstract class
│   │   ├── factory.py                # Adapter factory
│   │   ├── agnes.py                  # Agnes AI adapter
│   │   ├── openai.py                 # OpenAI adapter
│   │   ├── azure.py                  # Azure OpenAI adapter
│   │   └── ...                       # More adapters
│   ├── core/                         # Core utilities
│   │   ├── http_client.py            # Async HTTP client
│   │   ├── retry.py                  # Retry mechanism
│   │   ├── errors.py                 # Error definitions
│   │   ├── config.py                 # Configuration
│   │   └── utils.py                  # Utilities
│   └── models/                       # Pydantic data models
│       ├── common.py                 # Common structures
│       ├── chat.py                   # Chat models
│       ├── image.py                  # Image models
│       ├── video.py                  # Video models
│       └── options.py                # Unified options
├── docs/                             # Documentation
│   ├── 01-overview.md                # Project overview
│   ├── 02-architecture.md            # Architecture design
│   └── 03-api-reference.md           # API reference
├── tests/                            # Test suite
├── examples/                         # Usage examples
├── pyproject.toml                    # Project config
└── README.md                         # Project docs (English)
```

---

## 🚀 Quick Start

Get started in 3 steps:

### Step 1: Install

```bash
# From PyPI (coming soon)
pip install agn-sdk

# Or install from source (development mode)
git clone https://github.com/your-org/agn-sdk.git
cd agn-sdk
pip install -e .
```

### Step 2: Configure API Key

```bash
# Option A — Environment variable (Recommended)
export AGN_API_KEY='your-api-key'
export AGN_BASE_URL='https://api.agnes.ai/v1'  # Provider-specific base URL

# Option B — .env file (auto-loaded)
echo "AGN_API_KEY=your-api-key" > .env
echo "AGN_BASE_URL=https://api.agnes.ai/v1" >> .env

# Option C — Pass via code
client = Client(provider="agnes", api_key="your-key", base_url="https://api.agnes.ai/v1")
```

### Step 3: Call AI Models

```python
import asyncio
from agn import Client

async def main():
    # Create client
    client = Client(
        provider="agnes",
        api_key="your-api-key",
        base_url="https://api.agnes.ai/v1",
    )
    
    # 💬 Chat Completion
    response = await client.chat(
        model="claude-3-opus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        temperature=0.7,
    )
    print(response.choices[0].message.content)
    
    # 🖼️ Image Generation
    result = await client.image_generate(
        model="dall-e-3",
        prompt="A beautiful sunset over the ocean",
        size="1024x1024",
        quality="hd",
    )
    print(result.data[0].url)
    
    # 🎬 Video Creation (async with polling)
    task = await client.video_create(
        model="video-gen-1",
        prompt="A cat walking in the garden",
        width=1280,
        height=720,
        num_frames=121,
    )
    
    # Poll video status until complete
    while True:
        status = await client.video_poll(task.task_id)
        print(f"Status: {status.status}, Progress: {status.progress}%")
        if status.status in ("completed", "failed"):
            break
    
    print(f"Video URL: {status.video_url}")

if __name__ == "__main__":
    asyncio.run(main())
```

✨ **That's it!** You now have a unified interface to all supported AI providers.

---

## 📖 Complete Usage Reference

### Chat Completion

```python
response = await client.chat(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,        # Randomness (0.0-2.0)
    max_tokens=1000,        # Max response tokens
    top_p=1.0,              # Nucleus sampling
    frequency_penalty=0.0,   # Repetition penalty
    presence_penalty=0.0,    # Topic diversity
    stream=False,            # Streaming response
)
print(response.choices[0].message.content)
```

### Image Generation

```python
result = await client.image_generate(
    model="dall-e-3",
    prompt="A futuristic city with flying cars",
    size="1024x1024",       # 1024x1024, 1024x1792, 1792x1024
    quality="hd",           # standard or hd
    style="vivid",          # vivid or natural (DALL-E 3)
    n=1,                    # Number of images
)
print(result.data[0].url)   # or result.data[0].b64_json
```

### Video Creation

```python
# Create video task
task = await client.video_create(
    model="video-gen-1",
    prompt="A dramatic sword fight scene",
    width=1280,
    height=720,
    num_frames=121,         # Must satisfy 8n+1 (e.g., 33, 49, 81, 121, 241)
    frame_rate=24,
    seed=42,                # Optional: for reproducibility
)
print(f"Task ID: {task.task_id}")

# Poll until complete
status = await client.video_poll(task.task_id)
while status.status == "in_progress":
    await asyncio.sleep(5)
    status = await client.video_poll(task.task_id)
    
print(f"Video URL: {status.video_url}")
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    API Layer (Client)                   │
│            chat() / image_generate() / video_create()   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Router Layer                           │
│          Model routing, load balancing, fallback        │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 Adapter Layer                           │
│    BaseAdapter → AgnesAdapter / OpenAIAdapter / ...     │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Core Layer                            │
│      HTTP client, retry, errors, config, utils          │
└─────────────────────────────────────────────────────────┘
```

- **API Layer** — Unified `Client` class, user-facing interface
- **Router Layer** — Model selection, routing, load balancing
- **Adapter Layer** — Provider-specific implementations, parameter mapping, response normalization
- **Core Layer** — Shared utilities (HTTP, retry, errors, config)

---

## 📋 Adapter Development

Adding a new AI provider is straightforward:

1. **Create adapter** — Inherit `BaseAdapter`, implement required methods
2. **Register factory** — Call `AdapterFactory.register("provider_name", YourAdapter)`
3. **Declare capabilities** — Set `supported_capabilities` list

```python
from agn.adapters.base import BaseAdapter
from agn.adapters.factory import AdapterFactory

class NewProviderAdapter(BaseAdapter):
    provider_type = "newprovider"
    provider_name = "New Provider"
    supported_capabilities = [Capabilities.CHAT, Capabilities.IMAGE_GENERATE]
    
    async def start(self) -> None:
        # Initialize HTTP client
        ...
    
    async def chat(self, model: str, messages: list[ChatMessage], **kwargs):
        # Implement chat logic
        ...
    
    # ... implement other methods

AdapterFactory.register("newprovider", NewProviderAdapter)
```

---

## 🧪 Development

```bash
# Clone and setup
git clone https://github.com/your-org/agn-sdk.git
cd agn-sdk
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black agn/

# Linting
ruff check agn/

# Type checking
mypy agn/
```

---

## 📜 License

MIT License
