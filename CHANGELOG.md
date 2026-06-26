# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.1] - 2026-06-26

### Fixed

- EdgeTTSAdapter 空音频检测：edge-tts 服务端未返回音频时不再静默返回空 `SpeechResult`，改为抛出 `APIError`（code=`NO_AUDIO_RECEIVED`），调用方可直接捕获异常而非靠文件大小事后发现

## [1.1.0] - 2026-06-26

### Added

- 支持免费 Provider 免认证使用：`BaseAdapter` 新增 `requires_api_key` 类变量，免费 Provider（如 Edge TTS）设为 `False` 即可不传 API Key
- 新增免费 Provider 场景测试（`test_client_init_free_provider_without_api_key` 等）

### Changed

- `ProviderConfig.api_key` 从 `str` 改为 `str | None`，免费 Provider 可不传
- `Client` API Key 校验逻辑改为条件式：仅 `requires_api_key=True` 时检查
- 所有适配器 `__init__` 的 `self.api_key = config.api_key` 改为 `or ""` 兜底为 `str`（36 处）

### Fixed

- 修复设计缺陷：原先所有 Provider 都被强制要求 `api_key`，导致 Edge TTS 等免费模型无法正常使用

## [1.0.0] - 2026-06-25

### Added

- 多模型统一接口 SDK 首个正式版本
- 统一 API：chat / image_generate / video_create / transcribe / speech / embed
- 分层架构：API 层 / 路由器层 / 适配器层 / 核心层 / 数据模型层
- 支持 Provider：Agnes / OpenAI / Azure / Gemini / Anthropic / Runway / Pika / Kling / Stability / 中文模型聚合平台 / Edge TTS / ElevenLabs / Cartesia / Deepgram / AssemblyAI / Volcengine 等
- 生产级特性：异步优先、重试机制、错误映射、参数归一化、负载均衡、Fallback
- 643 单元测试，mypy strict 0 错误
