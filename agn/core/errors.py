"""
AGN-SDK 错误类型定义

定义统一的错误类型，用于映射各 Provider 的特定错误。
"""

from typing import Any


class AGNError(Exception):
    """SDK 基础错误，所有 SDK 错误的基类"""

    def __init__(
        self,
        message: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.original_error = original_error

    def __repr__(self) -> str:
        parts = [f"AGNError({self.message!r}"]
        if self.code:
            parts.append(f", code={self.code!r}")
        parts.append(")")
        return "".join(parts)

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class AuthenticationError(AGNError):
    """认证错误 - API Key 无效、过期或没有权限"""

    def __init__(
        self,
        message: str = "Authentication failed",
        code: str = "AUTHENTICATION_ERROR",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, code, details, original_error)


class RateLimitError(AGNError):
    """限流错误 - 请求频率超过限制"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        code: str = "RATE_LIMIT_ERROR",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, code, details, original_error)


class ValidationError(AGNError):
    """参数校验错误 - 请求参数不合法"""

    def __init__(
        self,
        message: str = "Validation failed",
        code: str = "VALIDATION_ERROR",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, code, details, original_error)


class ModelNotFoundError(AGNError):
    """模型不存在错误 - 请求的模型不可用"""

    def __init__(
        self,
        message: str = "Model not found",
        code: str = "MODEL_NOT_FOUND",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, code, details, original_error)


class APIError(AGNError):
    """API 调用错误 - 来自 Provider API 的错误响应"""

    def __init__(
        self,
        message: str = "API call failed",
        code: str = "API_ERROR",
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, code, details, original_error)
        self.status_code = status_code


class TimeoutError(AGNError):
    """超时错误 - 请求超时"""

    def __init__(
        self,
        message: str = "Request timeout",
        code: str = "TIMEOUT_ERROR",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, code, details, original_error)


class NetworkError(AGNError):
    """网络错误 - 网络连接问题"""

    def __init__(
        self,
        message: str = "Network error",
        code: str = "NETWORK_ERROR",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, code, details, original_error)


class ProviderNotFoundError(AGNError):
    """Provider 不存在错误 - 请求的 Provider 类型不可用"""

    def __init__(
        self,
        message: str = "Provider not found",
        code: str = "PROVIDER_NOT_FOUND",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, code, details, original_error)


class UnsupportedCapabilityError(AGNError):
    """不支持的能力错误 - Provider 不支持请求的能力"""

    def __init__(
        self,
        message: str = "Unsupported capability",
        code: str = "UNSUPPORTED_CAPABILITY",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, code, details, original_error)


class VoiceNotAvailableError(AGNError):
    """音色不可用错误 - 请求的 voice 已下线或不存在，重试无意义，应换音色

    典型场景：edge-tts 服务端下线了某个 voice，list_voices() 中查不到。
    """

    def __init__(
        self,
        message: str = "Voice not available",
        code: str = "VOICE_NOT_AVAILABLE",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, code, details, original_error)


class ServiceUnavailableError(AGNError):
    """服务不可用错误 - Provider 服务端临时不可用（限流/抖动），可重试

    典型场景：edge-tts 服务端限流导致返回空音频，但 voice 本身仍有效。
    """

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        code: str = "SERVICE_UNAVAILABLE",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, code, details, original_error)


def map_http_status_to_error(status_code: int, response_body: Any = None) -> AGNError:
    """
    将 HTTP 状态码映射到对应的错误类型

    Args:
        status_code: HTTP 状态码
        response_body: 响应体内容

    Returns:
        对应的 AGNError 子类实例
    """
    details = {"status_code": status_code}
    if response_body:
        details["response"] = response_body

    if status_code == 401 or status_code == 403:
        return AuthenticationError(
            message="Authentication failed. Please check your API key.",
            details=details,
        )
    elif status_code == 429:
        return RateLimitError(
            message="Rate limit exceeded. Please slow down your requests.",
            details=details,
        )
    elif status_code == 400:
        return ValidationError(
            message="Invalid request. Please check your parameters.",
            details=details,
        )
    elif status_code == 404:
        return ModelNotFoundError(
            message="The requested model was not found.",
            details=details,
        )
    elif 400 <= status_code < 500:
        return APIError(
            message=f"Client error: {status_code}",
            status_code=status_code,
            details=details,
        )
    elif 500 <= status_code < 600:
        return APIError(
            message=f"Server error: {status_code}. Please try again later.",
            status_code=status_code,
            details=details,
        )
    else:
        return APIError(
            message=f"Unexpected status code: {status_code}",
            status_code=status_code,
            details=details,
        )
