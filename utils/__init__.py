"""
Utility Modules for OpenEvolve

This package provides utility modules following the Federation Constitution:
- Structured logging with correlation IDs
- Custom error classes with tracking
- UTC timestamp utilities
"""

from .structured_logging import (
    StructuredLogger,
    JsonFormatter,
    generate_correlation_id,
    set_correlation_id,
    get_correlation_id,
    clear_correlation_id,
    with_correlation_id,
)

from .custom_errors import (
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

from .timestamp_utils import (
    getCurrentTimestamp,
    toUtcISO,
    isValidUtcISO,
    calculateDuration,
    addDuration,
    parseUtcISO,
    formatDuration,
)

__all__ = [
    # Structured logging
    "StructuredLogger",
    "JsonFormatter",
    "generate_correlation_id",
    "set_correlation_id",
    "get_correlation_id",
    "clear_correlation_id",
    "with_correlation_id",
    # Custom errors
    "BaseOpenEvolveError",
    "NetworkError",
    "AuthenticationError",
    "ValidationError",
    "ServerError",
    "RateLimitError",
    "ConfigurationError",
    "is_custom_error",
    "get_error_code",
    "get_error_correlation_id",
    "create_error_from_response",
    # Timestamps
    "getCurrentTimestamp",
    "toUtcISO",
    "isValidUtcISO",
    "calculateDuration",
    "addDuration",
    "parseUtcISO",
    "formatDuration",
]
