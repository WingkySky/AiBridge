"""
AGN-SDK 错误测试
"""


from agn.core.errors import (
    AGNError,
    APIError,
    AuthenticationError,
    ModelNotFoundError,
    RateLimitError,
    TimeoutError,
    ValidationError,
    map_http_status_to_error,
)


class TestErrorTypes:
    """错误类型测试"""

    def test_agn_error_basic(self) -> None:
        """测试基础错误"""
        error = AGNError("Test error")
        assert error.message == "Test error"
        assert error.code is None
        assert str(error) == "Test error"

    def test_agn_error_with_code(self) -> None:
        """测试带错误码的错误"""
        error = AGNError("Test error", code="TEST_ERROR")
        assert error.message == "Test error"
        assert error.code == "TEST_ERROR"
        assert str(error) == "[TEST_ERROR] Test error"

    def test_authentication_error(self) -> None:
        """测试认证错误"""
        error = AuthenticationError()
        assert error.code == "AUTHENTICATION_ERROR"
        assert "Authentication" in str(error)

    def test_rate_limit_error(self) -> None:
        """测试限流错误"""
        error = RateLimitError()
        assert error.code == "RATE_LIMIT_ERROR"
        assert "Rate limit" in str(error)

    def test_timeout_error(self) -> None:
        """测试超时错误"""
        error = TimeoutError()
        assert error.code == "TIMEOUT_ERROR"
        assert "timeout" in str(error).lower()

    def test_validation_error(self) -> None:
        """测试验证错误"""
        error = ValidationError(message="Invalid parameter: name")
        assert error.code == "VALIDATION_ERROR"
        assert "name" in str(error)

    def test_model_not_found_error(self) -> None:
        """测试模型不存在错误"""
        error = ModelNotFoundError(message="Model 'unknown' not found")
        assert error.code == "MODEL_NOT_FOUND"
        assert "unknown" in str(error)

    def test_api_error(self) -> None:
        """测试 API 错误"""
        error = APIError(
            message="Server error",
            status_code=500,
        )
        assert error.code == "API_ERROR"
        assert error.status_code == 500


class TestMapHttpStatusToError:
    """HTTP 状态码映射测试"""

    def test_map_401(self) -> None:
        """测试 401 错误"""
        error = map_http_status_to_error(401)
        assert isinstance(error, AuthenticationError)

    def test_map_403(self) -> None:
        """测试 403 错误"""
        error = map_http_status_to_error(403)
        assert isinstance(error, AuthenticationError)

    def test_map_429(self) -> None:
        """测试 429 错误"""
        error = map_http_status_to_error(429)
        assert isinstance(error, RateLimitError)

    def test_map_400(self) -> None:
        """测试 400 错误"""
        error = map_http_status_to_error(400)
        assert isinstance(error, ValidationError)

    def test_map_404(self) -> None:
        """测试 404 错误"""
        error = map_http_status_to_error(404)
        assert isinstance(error, ModelNotFoundError)

    def test_map_500(self) -> None:
        """测试 500 错误"""
        error = map_http_status_to_error(500)
        assert isinstance(error, APIError)
        assert error.status_code == 500

    def test_map_503(self) -> None:
        """测试 503 错误"""
        error = map_http_status_to_error(503)
        assert isinstance(error, APIError)
        assert "try again later" in str(error)
