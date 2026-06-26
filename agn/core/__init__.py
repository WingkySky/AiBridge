"""
AGN-SDK 核心层

包含 HTTP 客户端、重试机制、错误处理、配置管理等核心功能。
"""

from agn.core.config import Config, get_env, get_provider_config, load_env
from agn.core.errors import (
    AGNError,
    APIError,
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderNotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    TimeoutError,
    UnsupportedCapabilityError,
    ValidationError,
    VoiceNotAvailableError,
    map_http_status_to_error,
)

__all__ = [
    # 错误类型
    "AGNError",
    "APIError",
    "AuthenticationError",
    "ModelNotFoundError",
    "NetworkError",
    "ProviderNotFoundError",
    "RateLimitError",
    "ServiceUnavailableError",
    "TimeoutError",
    "UnsupportedCapabilityError",
    "ValidationError",
    "VoiceNotAvailableError",
    "map_http_status_to_error",
    # 配置
    "Config",
    "get_env",
    "get_provider_config",
    "load_env",
]
