"""Logging utilities for Adaptive MDAP."""

import logging
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import threading

# Thread-local storage for correlation IDs
_local = threading.local()


class StructuredLogFormatter(logging.Formatter):
    """JSON structured log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if available
        if hasattr(_local, 'correlation_id'):
            log_data["correlation_id"] = _local.correlation_id
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable log formatter with correlation ID support."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record in human-readable format."""
        correlation = getattr(_local, 'correlation_id', None)
        correlation_str = f"[{correlation}] " if correlation else ""
        
        return f"{datetime.utcnow().isoformat()} [{record.levelname}] {correlation_str}{record.name}: {record.getMessage()}"


def setup_logging(
    level: str = "INFO",
    structured: bool = True,
    log_file: Optional[str] = None
) -> None:
    """Set up logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARN, ERROR)
        structured: Use JSON structured logging
        log_file: Optional file path for logs
    """
    handlers: list = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if structured:
        console_handler.setFormatter(StructuredLogFormatter())
    else:
        console_handler.setFormatter(HumanReadableFormatter())
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        if structured:
            file_handler.setFormatter(StructuredLogFormatter())
        else:
            file_handler.setFormatter(HumanReadableFormatter())
        handlers.append(file_handler)
    
    # Configure root logger for adaptive_mdap
    logger = logging.getLogger("adaptive_mdap")
    logger.setLevel(getattr(logging, level.upper()))
    
    for handler in handlers:
        logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name (should be within adaptive_mdap namespace)
    
    Returns:
        Configured logger
    """
    if not name.startswith("adaptive_mdap"):
        name = f"adaptive_mdap.{name}"
    return logging.getLogger(name)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current thread."""
    _local.correlation_id = correlation_id


def get_correlation_id() -> Optional[str]:
    """Get correlation ID for current thread."""
    return getattr(_local, 'correlation_id', None)


def clear_correlation_id() -> None:
    """Clear correlation ID for current thread."""
    if hasattr(_local, 'correlation_id'):
        delattr(_local, 'correlation_id')


class LogContext:
    """Context manager for correlation ID."""
    
    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self.previous_id: Optional[str] = None
    
    def __enter__(self):
        self.previous_id = get_correlation_id()
        set_correlation_id(self.correlation_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_id:
            set_correlation_id(self.previous_id)
        else:
            clear_correlation_id()
        return False
