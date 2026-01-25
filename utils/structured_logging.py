"""
Structured Logging Utility for OpenEvolve

Provides JSON Lines format logging with correlation IDs for distributed tracing.
"""

import logging
import json
import uuid
import time
from typing import Dict, Any, Optional
from datetime import datetime
from contextvars import ContextVar

# Context variable for correlation ID
_correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class StructuredLogger:
    """
    Structured logger that outputs JSON Lines format with correlation IDs.

    Follows the Law of Configuration Explicitness - no magic defaults.
    Follows the Law of UTC - all timestamps in UTC.
    """

    def __init__(self, name: str, log_level: int = logging.INFO):
        """
        Initialize structured logger.

        Args:
            name: Logger name (e.g., service name)
            log_level: Logging level (default: INFO)

        Raises:
            ValueError: If name is empty or None
        """
        if not name:
            raise ValueError("Logger name must be provided")

        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            # Create JSON formatter
            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            self.logger.addHandler(handler)

    def _build_log_dict(
        self,
        level: str,
        msg: str,
        correlation_id: Optional[str] = None,
        source_service: Optional[str] = None,
        target_service: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build structured log dictionary."""
        log_dict = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "message": msg,
            "logger": self.logger.name,
            "correlation_id": correlation_id or get_correlation_id(),
        }

        # Add optional fields
        if source_service:
            log_dict["source_service"] = source_service
        if target_service:
            log_dict["target_service"] = target_service

        # Add extra fields
        log_dict.update(kwargs)

        return log_dict

    def info(
        self,
        msg: str,
        correlation_id: Optional[str] = None,
        source_service: Optional[str] = None,
        target_service: Optional[str] = None,
        **kwargs
    ):
        """Log info level message."""
        log_dict = self._build_log_dict(
            "INFO", msg, correlation_id, source_service, target_service, **kwargs
        )
        # Log with extra fields for JsonFormatter to use
        self.logger.info(msg, extra={
            "structured_log": log_dict,
            "correlation_id": log_dict["correlation_id"],
            "source_service": source_service,
            "target_service": target_service,
            **kwargs
        })

    def warning(
        self,
        msg: str,
        correlation_id: Optional[str] = None,
        source_service: Optional[str] = None,
        target_service: Optional[str] = None,
        **kwargs
    ):
        """Log warning level message."""
        log_dict = self._build_log_dict(
            "WARNING", msg, correlation_id, source_service, target_service, **kwargs
        )
        self.logger.warning(msg, extra={
            "structured_log": log_dict,
            "correlation_id": log_dict["correlation_id"],
            "source_service": source_service,
            "target_service": target_service,
            **kwargs
        })

    def warn(
        self,
        msg: str,
        correlation_id: Optional[str] = None,
        source_service: Optional[str] = None,
        target_service: Optional[str] = None,
        **kwargs
    ):
        """Alias for warning."""
        self.warning(msg, correlation_id, source_service, target_service, **kwargs)

    def error(
        self,
        msg: str,
        correlation_id: Optional[str] = None,
        source_service: Optional[str] = None,
        target_service: Optional[str] = None,
        **kwargs
    ):
        """Log error level message."""
        log_dict = self._build_log_dict(
            "ERROR", msg, correlation_id, source_service, target_service, **kwargs
        )
        self.logger.error(msg, extra={
            "structured_log": log_dict,
            "correlation_id": log_dict["correlation_id"],
            "source_service": source_service,
            "target_service": target_service,
            **kwargs
        })

    def debug(
        self,
        msg: str,
        correlation_id: Optional[str] = None,
        source_service: Optional[str] = None,
        target_service: Optional[str] = None,
        **kwargs
    ):
        """Log debug level message."""
        log_dict = self._build_log_dict(
            "DEBUG", msg, correlation_id, source_service, target_service, **kwargs
        )
        self.logger.debug(msg, extra={
            "structured_log": log_dict,
            "correlation_id": log_dict["correlation_id"],
            "source_service": source_service,
            "target_service": target_service,
            **kwargs
        })


class JsonFormatter(logging.Formatter):
    """Custom formatter that outputs JSON."""

    def format(self, record):
        """Format log record as JSON."""
        # If there's a structured_log in extra, use it directly
        if hasattr(record, "structured_log"):
            log_dict = record.structured_log
        else:
            # Build basic log dict
            log_dict = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
            }

            # Add extra fields from record
            if hasattr(record, "correlation_id"):
                log_dict["correlation_id"] = record.correlation_id
            if hasattr(record, "source_service"):
                log_dict["source_service"] = record.source_service
            if hasattr(record, "target_service"):
                log_dict["target_service"] = record.target_service

        return json.dumps(log_dict)


def generate_correlation_id() -> str:
    """
    Generate a new UUID v4 correlation ID.

    Returns:
        UUID v4 string
    """
    return str(uuid.uuid4())


def set_correlation_id(correlation_id: str):
    """
    Set correlation ID in context.

    Args:
        correlation_id: Correlation ID to set
    """
    _correlation_id.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """
    Get correlation ID from context.

    Returns:
        Correlation ID or None if not set
    """
    return _correlation_id.get()


def clear_correlation_id():
    """Clear correlation ID from context."""
    _correlation_id.set(None)


def with_correlation_id(func):
    """
    Decorator to ensure correlation ID is set for function.

    Generates new correlation ID if not already set.
    """
    def wrapper(*args, **kwargs):
        if get_correlation_id() is None:
            set_correlation_id(generate_correlation_id())

        return func(*args, **kwargs)

    return wrapper
