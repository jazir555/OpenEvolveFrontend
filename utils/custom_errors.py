"""
Custom Error Classes for OpenEvolve

Provides specialized error types with correlation IDs and error codes.
"""

from typing import Optional, Dict, Any
import uuid


class BaseOpenEvolveError(Exception):
    """
    Base class for all OpenEvolve custom errors.

    All errors include:
    - correlation_id: UUID for tracking
    - error_code: Machine-readable error code
    - context: Additional error context
    """

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}

    def __str__(self):
        return f"[{self.error_code}] {self.message} (correlation_id: {self.correlation_id})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "correlation_id": self.correlation_id,
            "context": self.context
        }


class NetworkError(BaseOpenEvolveError):
    """
    Network-related errors (connection failures, timeouts, etc.).

    Error code: NETWORK_ERROR
    """

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        full_context = context or {}
        if url:
            full_context["url"] = url
        if status_code:
            full_context["status_code"] = status_code

        super().__init__(
            message=message,
            correlation_id=correlation_id,
            error_code="NETWORK_ERROR",
            context=full_context
        )
        self.url = url
        self.status_code = status_code


class AuthenticationError(BaseOpenEvolveError):
    """
    Authentication failures (401, invalid credentials, etc.).

    Error code: AUTHENTICATION_ERROR
    """

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        auth_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        full_context = context or {}
        if auth_type:
            full_context["auth_type"] = auth_type

        super().__init__(
            message=message,
            correlation_id=correlation_id,
            error_code="AUTHENTICATION_ERROR",
            context=full_context
        )
        self.auth_type = auth_type


class ValidationError(BaseOpenEvolveError):
    """
    Validation errors (invalid input, schema violations, etc.).

    Error code: VALIDATION_ERROR
    """

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        field: Optional[str] = None,
        validation_errors: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        full_context = context or {}
        if field:
            full_context["field"] = field
        if validation_errors:
            full_context["validation_errors"] = validation_errors

        super().__init__(
            message=message,
            correlation_id=correlation_id,
            error_code="VALIDATION_ERROR",
            context=full_context
        )
        self.field = field
        self.validation_errors = validation_errors or []


class ServerError(BaseOpenEvolveError):
    """
    Server-side errors (5xx responses).

    Error code: SERVER_ERROR
    """

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        status_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        full_context = context or {}
        if status_code:
            full_context["status_code"] = status_code

        super().__init__(
            message=message,
            correlation_id=correlation_id,
            error_code="SERVER_ERROR",
            context=full_context
        )
        self.status_code = status_code


class RateLimitError(BaseOpenEvolveError):
    """
    Rate limiting errors (429 responses).

    Error code: RATE_LIMIT_ERROR
    """

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        retry_after: Optional[int] = None,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        full_context = context or {}
        if retry_after is not None:
            full_context["retry_after"] = retry_after
        if limit:
            full_context["limit"] = limit

        super().__init__(
            message=message,
            correlation_id=correlation_id,
            error_code="RATE_LIMIT_ERROR",
            context=full_context
        )
        self.retry_after = retry_after
        self.limit = limit


class ConfigurationError(BaseOpenEvolveError):
    """
    Configuration errors (missing config, invalid settings).

    Error code: CONFIGURATION_ERROR
    """

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        config_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        full_context = context or {}
        if config_key:
            full_context["config_key"] = config_key

        super().__init__(
            message=message,
            correlation_id=correlation_id,
            error_code="CONFIGURATION_ERROR",
            context=full_context
        )
        self.config_key = config_key


# Helper functions


def is_custom_error(error: Exception) -> bool:
    """
    Check if error is a custom OpenEvolve error.

    Args:
        error: Exception to check

    Returns:
        True if error is BaseOpenEvolveError subclass
    """
    return isinstance(error, BaseOpenEvolveError)


def get_error_code(error: Exception) -> Optional[str]:
    """
    Get error code from exception.

    Args:
        error: Exception to get code from

    Returns:
        Error code or None if not a custom error
    """
    if is_custom_error(error):
        return error.error_code
    return None


def get_error_correlation_id(error: Exception) -> Optional[str]:
    """
    Get correlation ID from exception.

    Args:
        error: Exception to get correlation ID from

    Returns:
        Correlation ID or None if not a custom error
    """
    if is_custom_error(error):
        return error.correlation_id
    return None


def create_error_from_response(
    status_code: int,
    message: str,
    correlation_id: Optional[str] = None
) -> BaseOpenEvolveError:
    """
    Create appropriate error from HTTP response.

    Args:
        status_code: HTTP status code
        message: Error message
        correlation_id: Optional correlation ID

    Returns:
        Appropriate error instance
    """
    if status_code == 401:
        return AuthenticationError(message, correlation_id)
    elif status_code == 429:
        return RateLimitError(message, correlation_id)
    elif 400 <= status_code < 500 and status_code not in [401, 429]:
        return ValidationError(message, correlation_id)
    elif 500 <= status_code < 600:
        return ServerError(message, correlation_id, status_code=status_code)
    else:
        return NetworkError(message, correlation_id, status_code=status_code)
