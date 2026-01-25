"""
Comprehensive Error Handling Tests (Bug #20)

Tests for custom error classes with correlation IDs and error codes.
Follows the Federation Constitution's error handling requirements.
"""

import pytest
import uuid
from typing import Dict, Any

import sys
from pathlib import Path

# Add utils to path
utils_path = Path(__file__).parent.parent.parent / "utils"
sys.path.insert(0, str(utils_path))

from custom_errors import (
    BaseOpenEvolveError,
    NetworkError,
    AuthenticationError,
    ValidationError,
    ServerError,
    RateLimitError,
    ConfigurationError,
    is_custom_error,
    get_error_code,
    get_error_correlation_id,
    create_error_from_response,
)


class TestBaseOpenEvolveError:
    """Test base error class functionality."""

    def test_error_has_required_attributes(self):
        """Test error has correlation_id, error_code, and context."""
        error = BaseOpenEvolveError("Test error")

        assert hasattr(error, 'message')
        assert hasattr(error, 'correlation_id')
        assert hasattr(error, 'error_code')
        assert hasattr(error, 'context')

    def test_error_generates_correlation_id(self):
        """Test error generates UUID correlation ID if not provided."""
        error = BaseOpenEvolveError("Test error")

        assert error.correlation_id is not None
        assert len(error.correlation_id) == 36  # UUID length

    def test_error_uses_provided_correlation_id(self):
        """Test error uses provided correlation ID."""
        test_id = str(uuid.uuid4())
        error = BaseOpenEvolveError("Test error", correlation_id=test_id)

        assert error.correlation_id == test_id

    def test_error_uses_class_name_as_default_code(self):
        """Test error uses class name as default error code."""
        error = BaseOpenEvolveError("Test error")

        assert error.error_code == "BaseOpenEvolveError"

    def test_error_uses_provided_error_code(self):
        """Test error uses provided error code."""
        error = BaseOpenEvolveError(
            "Test error",
            error_code="CUSTOM_ERROR"
        )

        assert error.error_code == "CUSTOM_ERROR"

    def test_error_includes_context(self):
        """Test error includes provided context."""
        context = {"key": "value", "number": 42}
        error = BaseOpenEvolveError("Test error", context=context)

        assert error.context == context

    def test_error_defaults_to_empty_context(self):
        """Test error defaults to empty context dict."""
        error = BaseOpenEvolveError("Test error")

        assert error.context == {}

    def test_error_string_representation(self):
        """Test error string representation includes key info."""
        error = BaseOpenEvolveError("Test error")

        error_str = str(error)

        assert "BaseOpenEvolveError" in error_str
        assert "Test error" in error_str
        assert "correlation_id" in error_str
        assert error.correlation_id in error_str

    def test_error_to_dict(self):
        """Test error can be converted to dictionary."""
        test_id = str(uuid.uuid4())
        error = BaseOpenEvolveError(
            "Test error",
            correlation_id=test_id,
            error_code="TEST_ERROR",
            context={"key": "value"}
        )

        error_dict = error.to_dict()

        assert error_dict["error_type"] == "BaseOpenEvolveError"
        assert error_dict["error_code"] == "TEST_ERROR"
        assert error_dict["message"] == "Test error"
        assert error_dict["correlation_id"] == test_id
        assert error_dict["context"]["key"] == "value"


class TestNetworkError:
    """Test NetworkError functionality."""

    def test_network_error_code(self):
        """Test NetworkError has correct error code."""
        error = NetworkError("Connection failed")

        assert error.error_code == "NETWORK_ERROR"

    def test_network_error_includes_url_in_context(self):
        """Test NetworkError includes URL in context."""
        error = NetworkError("Connection failed", url="https://api.example.com")

        assert error.url == "https://api.example.com"
        assert error.context["url"] == "https://api.example.com"

    def test_network_error_includes_status_code_in_context(self):
        """Test NetworkError includes status code in context."""
        error = NetworkError("Connection failed", status_code=503)

        assert error.status_code == 503
        assert error.context["status_code"] == 503

    def test_network_error_with_all_params(self):
        """Test NetworkError with all parameters."""
        test_id = str(uuid.uuid4())
        error = NetworkError(
            "Connection failed",
            correlation_id=test_id,
            url="https://api.example.com",
            status_code=503,
            context={"timeout": 30}
        )

        assert error.correlation_id == test_id
        assert error.url == "https://api.example.com"
        assert error.status_code == 503
        assert error.context["timeout"] == 30


class TestAuthenticationError:
    """Test AuthenticationError functionality."""

    def test_authentication_error_code(self):
        """Test AuthenticationError has correct error code."""
        error = AuthenticationError("Invalid credentials")

        assert error.error_code == "AUTHENTICATION_ERROR"

    def test_authentication_error_for_401(self):
        """Test AuthenticationError is thrown for 401."""
        error = AuthenticationError("Unauthorized")

        assert isinstance(error, BaseOpenEvolveError)
        assert error.error_code == "AUTHENTICATION_ERROR"

    def test_authentication_error_includes_auth_type(self):
        """Test AuthenticationError includes auth type."""
        error = AuthenticationError(
            "Invalid API key",
            auth_type="api_key"
        )

        assert error.auth_type == "api_key"
        assert error.context["auth_type"] == "api_key"


class TestValidationError:
    """Test ValidationError functionality."""

    def test_validation_error_code(self):
        """Test ValidationError has correct error code."""
        error = ValidationError("Invalid input")

        assert error.error_code == "VALIDATION_ERROR"

    def test_validation_error_for_400(self):
        """Test ValidationError is thrown for 400."""
        error = ValidationError("Bad request")

        assert isinstance(error, BaseOpenEvolveError)
        assert error.error_code == "VALIDATION_ERROR"

    def test_validation_error_includes_field(self):
        """Test ValidationError includes field name."""
        error = ValidationError(
            "Invalid email",
            field="email"
        )

        assert error.field == "email"
        assert error.context["field"] == "email"

    def test_validation_error_includes_validation_errors(self):
        """Test ValidationError includes list of validation errors."""
        validation_errors = [
            "Email is required",
            "Email must be valid format"
        ]

        error = ValidationError(
            "Validation failed",
            validation_errors=validation_errors
        )

        assert error.validation_errors == validation_errors
        assert error.context["validation_errors"] == validation_errors


class TestServerError:
    """Test ServerError functionality."""

    def test_server_error_code(self):
        """Test ServerError has correct error code."""
        error = ServerError("Internal server error")

        assert error.error_code == "SERVER_ERROR"

    def test_server_error_for_5xx(self):
        """Test ServerError is thrown for 5xx responses."""
        error = ServerError("Server error", status_code=500)

        assert isinstance(error, BaseOpenEvolveError)
        assert error.error_code == "SERVER_ERROR"
        assert error.status_code == 500

    def test_server_error_for_various_5xx_codes(self):
        """Test ServerError handles various 5xx codes."""
        codes = [500, 502, 503, 504]

        for code in codes:
            error = ServerError(f"Server error {code}", status_code=code)
            assert error.status_code == code


class TestRateLimitError:
    """Test RateLimitError functionality."""

    def test_rate_limit_error_code(self):
        """Test RateLimitError has correct error code."""
        error = RateLimitError("Rate limit exceeded")

        assert error.error_code == "RATE_LIMIT_ERROR"

    def test_rate_limit_error_for_429(self):
        """Test RateLimitError is thrown for 429."""
        error = RateLimitError("Too many requests")

        assert isinstance(error, BaseOpenEvolveError)
        assert error.error_code == "RATE_LIMIT_ERROR"

    def test_rate_limit_error_includes_retry_after(self):
        """Test RateLimitError includes retry_after."""
        error = RateLimitError(
            "Rate limit exceeded",
            retry_after=60
        )

        assert error.retry_after == 60
        assert error.context["retry_after"] == 60

    def test_rate_limit_error_includes_limit(self):
        """Test RateLimitError includes limit."""
        error = RateLimitError(
            "Rate limit exceeded",
            limit=100
        )

        assert error.limit == 100
        assert error.context["limit"] == 100


class TestConfigurationError:
    """Test ConfigurationError functionality."""

    def test_configuration_error_code(self):
        """Test ConfigurationError has correct error code."""
        error = ConfigurationError("Missing configuration")

        assert error.error_code == "CONFIGURATION_ERROR"

    def test_configuration_error_includes_config_key(self):
        """Test ConfigurationError includes config_key."""
        error = ConfigurationError(
            "Missing API key",
            config_key="API_KEY"
        )

        assert error.config_key == "API_KEY"
        assert error.context["config_key"] == "API_KEY"


class TestHelperFunctions:
    """Test error helper functions."""

    def test_is_custom_error_returns_true_for_custom_errors(self):
        """Test is_custom_error returns True for custom errors."""
        error = NetworkError("Test error")

        assert is_custom_error(error) is True

    def test_is_custom_error_returns_false_for_standard_errors(self):
        """Test is_custom_error returns False for standard errors."""
        error = ValueError("Test error")

        assert is_custom_error(error) is False

    def test_get_error_code_for_custom_error(self):
        """Test get_error_code returns code for custom errors."""
        error = NetworkError("Test error")

        code = get_error_code(error)

        assert code == "NETWORK_ERROR"

    def test_get_error_code_for_standard_error(self):
        """Test get_error_code returns None for standard errors."""
        error = ValueError("Test error")

        code = get_error_code(error)

        assert code is None

    def test_get_error_correlation_id_for_custom_error(self):
        """Test get_error_correlation_id returns ID for custom errors."""
        test_id = str(uuid.uuid4())
        error = NetworkError("Test error", correlation_id=test_id)

        correlation_id = get_error_correlation_id(error)

        assert correlation_id == test_id

    def test_get_error_correlation_id_for_standard_error(self):
        """Test get_error_correlation_id returns None for standard errors."""
        error = ValueError("Test error")

        correlation_id = get_error_correlation_id(error)

        assert correlation_id is None


class TestCreateErrorFromResponse:
    """Test create_error_from_response function."""

    def test_401_creates_authentication_error(self):
        """Test 401 status creates AuthenticationError."""
        error = create_error_from_response(401, "Unauthorized")

        assert isinstance(error, AuthenticationError)
        assert error.message == "Unauthorized"

    def test_429_creates_rate_limit_error(self):
        """Test 429 status creates RateLimitError."""
        error = create_error_from_response(429, "Too many requests")

        assert isinstance(error, RateLimitError)
        assert error.message == "Too many requests"

    def test_400_creates_validation_error(self):
        """Test 400 status creates ValidationError."""
        error = create_error_from_response(400, "Bad request")

        assert isinstance(error, ValidationError)
        assert error.message == "Bad request"

    def test_404_creates_validation_error(self):
        """Test 404 status creates ValidationError."""
        error = create_error_from_response(404, "Not found")

        assert isinstance(error, ValidationError)

    def test_500_creates_server_error(self):
        """Test 500 status creates ServerError."""
        error = create_error_from_response(500, "Internal server error")

        assert isinstance(error, ServerError)
        assert error.status_code == 500

    def test_503_creates_server_error(self):
        """Test 503 status creates ServerError."""
        error = create_error_from_response(503, "Service unavailable")

        assert isinstance(error, ServerError)
        assert error.status_code == 503

    def test_unknown_status_creates_network_error(self):
        """Test unknown status (outside 4xx/5xx) creates NetworkError."""
        # Use a status code that's not in the 4xx or 5xx ranges
        error = create_error_from_response(399, "Invalid status")

        assert isinstance(error, NetworkError)
        assert error.status_code == 399

        # Also test with status > 600
        error2 = create_error_from_response(999, "Unknown error")
        assert isinstance(error2, NetworkError)
        assert error2.status_code == 999

    def test_includes_correlation_id(self):
        """Test created error includes correlation ID."""
        test_id = str(uuid.uuid4())
        error = create_error_from_response(
            401,
            "Unauthorized",
            correlation_id=test_id
        )

        assert error.correlation_id == test_id


class TestErrorConsistency:
    """Test error code consistency across errors."""

    def test_all_errors_have_consistent_codes(self):
        """Test all error types have consistent error codes."""
        error_classes = [
            (NetworkError, "NETWORK_ERROR"),
            (AuthenticationError, "AUTHENTICATION_ERROR"),
            (ValidationError, "VALIDATION_ERROR"),
            (ServerError, "SERVER_ERROR"),
            (RateLimitError, "RATE_LIMIT_ERROR"),
            (ConfigurationError, "CONFIGURATION_ERROR"),
        ]

        for error_class, expected_code in error_classes:
            error = error_class("Test message")
            assert error.error_code == expected_code, \
                f"{error_class.__name__} has incorrect error code"

    def test_all_errors_include_correlation_id(self):
        """Test all error types include correlation ID."""
        error_classes = [
            NetworkError,
            AuthenticationError,
            ValidationError,
            ServerError,
            RateLimitError,
            ConfigurationError,
        ]

        for error_class in error_classes:
            error = error_class("Test message")
            assert error.correlation_id is not None, \
                f"{error_class.__name__} missing correlation ID"

    def test_all_errors_can_convert_to_dict(self):
        """Test all error types can convert to dict."""
        error_classes = [
            NetworkError,
            AuthenticationError,
            ValidationError,
            ServerError,
            RateLimitError,
            ConfigurationError,
        ]

        for error_class in error_classes:
            error = error_class("Test message")
            error_dict = error.to_dict()

            assert isinstance(error_dict, dict)
            assert "error_type" in error_dict
            assert "error_code" in error_dict
            assert "message" in error_dict
            assert "correlation_id" in error_dict
