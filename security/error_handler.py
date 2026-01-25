"""
RESE Security: Error Handling and Recovery

Comprehensive error handling, graceful degradation, and recovery mechanisms.

Author: Agent M2 (Security and Reliability Specialist)
Created: 2025-12-31
"""

import sys
import traceback
import logging
import signal
import threading
import functools
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import json


# =============================================================================
# Error Types and Classification
# =============================================================================

class ErrorCategory(Enum):
    """Error categories for proper handling"""
    VALIDATION = "validation"           # Input validation errors
    EXECUTION = "execution"             # Runtime execution errors
    RESOURCE = "resource"               # Resource exhaustion
    DEPENDENCY = "dependency"           # Missing/broken dependencies
    NETWORK = "network"                 # Network-related errors
    DATABASE = "database"               # Database errors
    SECURITY = "security"               # Security violations
    INTEGRATION = "integration"         # External integration errors
    TIMEOUT = "timeout"                 # Operation timeout
    UNKNOWN = "unknown"                 # Unclassified errors


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"                       # Informational
    WARNING = "warning"                 # Warning, operation continues
    ERROR = "error"                     # Error, operation fails
    CRITICAL = "critical"               # Critical, system may be unstable
    FATAL = "fatal"                     # Fatal, requires immediate attention


@dataclass
class ErrorContext:
    """Context information for errors"""
    component: str                      # Component where error occurred
    operation: str                      # Operation being performed
    phase: Optional[str] = None         # Pipeline phase if applicable
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None       # User ID if applicable
    request_id: Optional[str] = None    # Request ID for tracing
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'component': self.component,
            'operation': self.operation,
            'phase': self.phase,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'request_id': self.request_id,
            'additional_data': self.additional_data
        }


@dataclass
class ErrorDetails:
    """Detailed error information"""
    error_type: str                     # Type of error
    error_message: str                  # Error message
    category: ErrorCategory             # Error category
    severity: ErrorSeverity             # Error severity
    context: ErrorContext               # Error context
    traceback_str: Optional[str] = None # Stack trace
    recovery_actions: List[str] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'error_type': self.error_type,
            'error_message': self.error_message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context.to_dict(),
            'traceback': self.traceback_str,
            'recovery_actions': self.recovery_actions,
            'related_errors': self.related_errors
        }


# =============================================================================
# Custom Exceptions
# =============================================================================

class RESEError(Exception):
    """Base exception for all RESE errors"""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        recovery_actions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext(
            component="unknown",
            operation="unknown"
        )
        self.recovery_actions = recovery_actions or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return ErrorDetails(
            error_type=type(self).__name__,
            error_message=self.message,
            category=self.category,
            severity=self.severity,
            context=self.context,
            traceback_str=traceback.format_exc(),
            recovery_actions=self.recovery_actions
        ).to_dict()


class ValidationError(RESEError):
    """Input validation error"""

    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )
        self.field = field


class ExecutionError(RESEError):
    """Runtime execution error"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class ResourceError(RESEError):
    """Resource exhaustion error"""

    def __init__(self, message: str, resource_type: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        self.resource_type = resource_type


class DependencyError(RESEError):
    """Missing/broken dependency error"""

    def __init__(self, message: str, dependency_name: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DEPENDENCY,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )
        self.dependency_name = dependency_name


class TimeoutError(RESEError):
    """Operation timeout error"""

    def __init__(self, message: str, timeout_seconds: float = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.WARNING,
            **kwargs
        )
        self.timeout_seconds = timeout_seconds


class SecurityError(RESEError):
    """Security violation error"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


# =============================================================================
# Error Handler
# =============================================================================

class ErrorHandler:
    """
    Centralized error handling for RESE system.

    Features:
    - Error classification and routing
    - Graceful degradation
    - Automatic recovery
    - Detailed logging
    - Error aggregation
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize error handler.

        Args:
            log_file: Optional log file path
        """
        self.log_file = log_file
        self.error_history: List[ErrorDetails] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            category: [] for category in ErrorCategory
        }

        # Setup logging
        self._setup_logging()

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        self.logger = logging.getLogger('RESE.error_handler')
        self.logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        raise_on_unhandled: bool = False
    ) -> ErrorDetails:
        """
        Handle an error with appropriate recovery strategies.

        Args:
            error: The exception to handle
            context: Error context
            raise_on_unhandled: Whether to raise if error can't be handled

        Returns:
            ErrorDetails with full error information
        """
        # Classify error
        error_details = self._classify_error(error, context)

        # Log error
        self._log_error(error_details)

        # Add to history
        self.error_history.append(error_details)
        self.error_counts[error_details.error_type] = \
            self.error_counts.get(error_details.error_type, 0) + 1

        # Attempt recovery
        recovered = self._attempt_recovery(error_details)

        if not recovered and raise_on_unhandled:
            raise error

        return error_details

    def _classify_error(self, error: Exception, context: ErrorContext) -> ErrorDetails:
        """
        Classify error into category and severity.

        Args:
            error: Exception to classify
            context: Error context

        Returns:
            ErrorDetails with classification
        """
        # Determine error type and category
        error_type = type(error).__name__
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.ERROR

        if isinstance(error, RESEError):
            # Already classified
            category = error.category
            severity = error.severity
            recovery_actions = error.recovery_actions
        elif isinstance(error, (ValueError, TypeError)):
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.ERROR
            recovery_actions = ["Validate input data", "Check data types"]
        elif isinstance(error, (MemoryError, RecursionError)):
            category = ErrorCategory.RESOURCE
            severity = ErrorSeverity.CRITICAL
            recovery_actions = ["Reduce data size", "Increase memory limits", "Check for infinite loops"]
        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            category = ErrorCategory.DEPENDENCY
            severity = ErrorSeverity.ERROR
            recovery_actions = ["Install missing dependencies", "Check Python environment"]
        elif isinstance(error, TimeoutError):
            category = ErrorCategory.TIMEOUT
            severity = ErrorSeverity.WARNING
            recovery_actions = ["Increase timeout", "Optimize operation", "Check system load"]
        elif isinstance(error, (ConnectionError, OSError)):
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.ERROR
            recovery_actions = ["Check network connection", "Retry operation", "Check service availability"]

        # Create error details
        return ErrorDetails(
            error_type=error_type,
            error_message=str(error),
            category=category,
            severity=severity,
            context=context,
            traceback_str=traceback.format_exc(),
            recovery_actions=recovery_actions if isinstance(error, RESEError) else []
        )

    def _log_error(self, error_details: ErrorDetails) -> None:
        """
        Log error with appropriate level.

        Args:
            error_details: Error details to log
        """
        log_message = f"[{error_details.category.value.upper()}] " \
                      f"{error_details.context.component}.{error_details.context.operation}: " \
                      f"{error_details.error_message}"

        # Map severity to logging level
        severity_to_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.CRITICAL
        }

        log_level = severity_to_level.get(error_details.severity, logging.ERROR)

        self.logger.log(log_level, log_message, extra={
            'error_details': error_details.to_dict()
        })

        # Log traceback if available
        if error_details.traceback_str:
            self.logger.debug(f"Traceback:\n{error_details.traceback_str}")

    def _attempt_recovery(self, error_details: ErrorDetails) -> bool:
        """
        Attempt to recover from error using registered strategies.

        Args:
            error_details: Error details

        Returns:
            True if recovery was successful
        """
        recovery_strategies = self.recovery_strategies.get(error_details.category, [])

        for strategy in recovery_strategies:
            try:
                if strategy(error_details):
                    self.logger.info(f"Recovery successful for {error_details.error_type}")
                    return True
            except Exception as e:
                self.logger.warning(f"Recovery strategy failed: {e}")

        return False

    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        strategy: Callable[[ErrorDetails], bool]
    ) -> None:
        """
        Register a recovery strategy for an error category.

        Args:
            category: Error category
            strategy: Recovery function that takes ErrorDetails and returns bool
        """
        self.recovery_strategies[category].append(strategy)

    def cleanup(self) -> None:
        """Cleanup resources before shutdown"""
        self.logger.info("Cleaning up error handler...")

        # Save error history to file
        if self.log_file:
            try:
                history_file = str(self.log_file).replace('.log', '_history.json')
                with open(history_file, 'w') as f:
                    json.dump([e.to_dict() for e in self.error_history], f, indent=2)
                self.logger.info(f"Error history saved to {history_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save error history: {e}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics.

        Returns:
            Dictionary with error statistics
        """
        return {
            'total_errors': len(self.error_history),
            'errors_by_type': self.error_counts,
            'errors_by_category': {
                category.value: sum(
                    1 for e in self.error_history if e.category == category
                )
                for category in ErrorCategory
            },
            'errors_by_severity': {
                severity.value: sum(
                    1 for e in self.error_history if e.severity == severity
                )
                for severity in ErrorSeverity
            }
        }


# =============================================================================
# Error Handling Decorators
# =============================================================================

def handle_errors(
    error_handler: ErrorHandler,
    context: Optional[ErrorContext] = None,
    reraise: bool = False
):
    """
    Decorator for automatic error handling.

    Args:
        error_handler: ErrorHandler instance
        context: Optional error context
        reraise: Whether to re-raise exceptions after handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create context if not provided
            func_context = context or ErrorContext(
                component=func.__module__,
                operation=func.__name__
            )

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_details = error_handler.handle_error(e, func_context, raise_on_unhandled=reraise)

                if reraise:
                    raise

                # Return error indicator or None
                return None

        return wrapper
    return decorator


def safe_execute(
    default_return: Any = None,
    log_errors: bool = True
):
    """
    Decorator for safe execution with default return on error.

    Args:
        default_return: Value to return on error
        log_errors: Whether to log errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logging.error(f"Error in {func.__name__}: {e}")
                return default_return

        return wrapper
    return decorator


def retry_on_error(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for automatic retry on error.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff multiplier between retries
        retry_on: Tuple of exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import time

            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e

                    if attempt < max_retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        logging.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {wait_time}s: {e}"
                        )
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Max retries exceeded for {func.__name__}")

            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# Context Managers
# =============================================================================

@contextmanager
def error_context(
    error_handler: ErrorHandler,
    component: str,
    operation: str,
    **context_kwargs
):
    """
    Context manager for error handling with automatic context.

    Args:
        error_handler: ErrorHandler instance
        component: Component name
        operation: Operation name
        **context_kwargs: Additional context data
    """
    context = ErrorContext(
        component=component,
        operation=operation,
        **context_kwargs
    )

    try:
        yield context
    except Exception as e:
        error_handler.handle_error(e, context)
        raise


@contextmanager
def graceful_degradation(fallback_func: Optional[Callable] = None):
    """
    Context manager for graceful degradation on error.

    Args:
        fallback_func: Optional fallback function to call on error
    """
    try:
        yield
    except Exception as e:
        logging.warning(f"Error occurred, degrading gracefully: {e}")
        if fallback_func:
            return fallback_func()


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.

    Opens circuit after too many failures, closes after recovery.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that counts as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open
        """
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half_open'
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.last_failure_time is None:
            return True
        elapsed = datetime.now().timestamp() - self.last_failure_time
        return elapsed >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'closed'

    def _on_failure(self) -> None:
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now().timestamp()

        if self.failure_count >= self.failure_threshold:
            self.state = 'open'


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Error types
    'RESEError',
    'ValidationError',
    'ExecutionError',
    'ResourceError',
    'DependencyError',
    'TimeoutError',
    'SecurityError',

    # Data structures
    'ErrorCategory',
    'ErrorSeverity',
    'ErrorContext',
    'ErrorDetails',

    # Handler
    'ErrorHandler',

    # Decorators
    'handle_errors',
    'safe_execute',
    'retry_on_error',

    # Context managers
    'error_context',
    'graceful_degradation',

    # Circuit breaker
    'CircuitBreaker',
]
