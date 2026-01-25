"""
Comprehensive Error Handling and Retry System

This module provides robust error handling for the adversarial testing system:
- Custom exception hierarchy
- Retry mechanisms with exponential backoff
- Circuit breaker pattern
- Dead letter queue
- Error recovery strategies
- Structured error logging
- Error aggregation and analysis

Author: OpenEvolve Resilience Team
Created: 2025-01-07
Version: 1.0.0
"""

import asyncio
import functools
import logging
import random
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Type, TypeVar, Union,
    Awaitable
)
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# EXCEPTION HIERARCHY
# =============================================================================

class AdversarialError(Exception):
    """Base exception for all adversarial testing errors"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "traceback": traceback.format_exc()
        }


class ConfigurationError(AdversarialError):
    """Configuration related errors"""
    pass


class ValidationError(AdversarialError):
    """Input validation errors"""
    pass


class APIError(AdversarialError):
    """External API errors"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_body = response_body


class TimeoutError(AdversarialError):
    """Operation timeout errors"""
    pass


class RateLimitError(APIError):
    """API rate limit errors"""
    pass


class ResourceExhaustedError(AdversarialError):
    """Resource exhaustion errors"""
    pass


class AttackGenerationError(AdversarialError):
    """Attack generation failures"""
    pass


class DefenseGenerationError(AdversarialError):
    """Defense generation failures"""
    pass


class LLMError(AdversarialError):
    """LLM-related errors"""
    pass


# =============================================================================
# ERROR SEVERITY
# =============================================================================

class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# ERROR HANDLER STRATEGY
# =============================================================================

class ErrorHandlerStrategy(ABC):
    """Base class for error handling strategies"""

    @abstractmethod
    def can_handle(self, exception: Exception) -> bool:
        """Check if this strategy can handle the exception"""
        pass

    @abstractmethod
    async def handle(
        self,
        exception: Exception,
        context: Dict[str, Any]
    ) -> Any:
        """Handle the exception"""
        pass


class RetryStrategy(ErrorHandlerStrategy):
    """Retry with exponential backoff"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

        # Exceptions that should trigger retry
        self.retryable_exceptions = (
            APIError,
            TimeoutError,
            RateLimitError,
            ConnectionError,
            TimeoutError,
        )

    def can_handle(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        return isinstance(exception, self.retryable_exceptions)

    async def handle(
        self,
        exception: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate retry delay"""
        attempt = context.get("attempt", 0)

        if attempt >= self.max_retries:
            raise exception

        # Calculate delay with exponential backoff
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        # Add jitter
        if self.jitter:
            delay = delay * (0.5 + random.random())

        logger.warning(
            f"Retryable error occurred (attempt {attempt + 1}/{self.max_retries}): "
            f"{exception}. Retrying in {delay:.2f}s"
        )

        await asyncio.sleep(delay)

        return {"should_retry": True, "delay": delay}


class FallbackStrategy(ErrorHandlerStrategy):
    """Use fallback value on error"""

    def __init__(self, fallback_value: Any = None):
        self.fallback_value = fallback_value

    def can_handle(self, exception: Exception) -> bool:
        """Handle all exceptions"""
        return True

    async def handle(
        self,
        exception: Exception,
        context: Dict[str, Any]
    ) -> Any:
        """Return fallback value"""
        logger.info(f"Using fallback value due to error: {exception}")
        return self.fallback_value


class CircuitBreakerStrategy(ErrorHandlerStrategy):
    """Circuit breaker pattern to prevent cascading failures"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open

    def can_handle(self, exception: Exception) -> bool:
        """Check if circuit is open"""
        if self.state == "open":
            # Check if recovery timeout has passed
            if self.last_failure_time:
                time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if time_since_failure > self.recovery_timeout:
                    self.state = "half-open"
                    logger.info("Circuit breaker entering half-open state")
                    return False

            # Circuit is open, don't attempt
            logger.warning("Circuit breaker is OPEN, rejecting request")
            return True

        return False

    async def handle(
        self,
        exception: Exception,
        context: Dict[str, Any]
    ) -> None:
        """Handle failure by opening circuit"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will remain open for {self.recovery_timeout}s"
            )

        raise exception

    def on_success(self):
        """Call on successful operation"""
        if self.state == "half-open":
            self.state = "closed"
            self.failure_count = 0
            logger.info("Circuit breaker closed after successful request")


class DeadLetterQueueStrategy(ErrorHandlerStrategy):
    """Send failed operations to dead letter queue"""

    def __init__(self, queue_path: str = "./dlq"):
        self.queue_path = queue_path
        from pathlib import Path
        Path(queue_path).mkdir(parents=True, exist_ok=True)

    def can_handle(self, exception: Exception) -> bool:
        """Handle all exceptions"""
        return True

    async def handle(
        self,
        exception: Exception,
        context: Dict[str, Any]
    ) -> None:
        """Add to dead letter queue"""
        timestamp = datetime.utcnow().isoformat()

        entry = {
            "timestamp": timestamp,
            "exception": {
                "type": type(exception).__name__,
                "message": str(exception),
                "details": exception.to_dict() if hasattr(exception, 'to_dict') else {}
            },
            "context": context
        }

        # Save to file
        from pathlib import Path
        filename = f"{timestamp.replace(':', '-').replace('.', '-')}.json"
        filepath = Path(self.queue_path) / filename

        with open(filepath, 'w') as f:
            json.dump(entry, f, indent=2)

        logger.info(f"Added to dead letter queue: {filepath}")


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[tuple] = None
):
    """
    Decorator for retrying functions with exponential backoff

    Example:
        @retry_with_exponential_backoff(max_retries=3)
        async def api_call():
            # Make API request
            return response
    """
    if retryable_exceptions is None:
        retryable_exceptions = (
            APIError,
            TimeoutError,
            RateLimitError,
            ConnectionError,
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt >= max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise

                    # Calculate delay
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Retryable error in {func.__name__} "
                        f"(attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt >= max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise

                    # Calculate delay
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Retryable error in {func.__name__} "
                        f"(attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    time.sleep(delay)

            raise last_exception

        # Check if function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker implementation

    Prevents cascading failures by stopping requests to a failing service
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open

    def __call__(self, func: F) -> F:
        """Decorator usage"""

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check if circuit is open
            if self.state == "open":
                if self.last_failure_time:
                    time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
                    if time_since_failure > self.recovery_timeout:
                        self.state = "half-open"
                        logger.info("Circuit breaker entering half-open state")
                    else:
                        raise APIError(
                            f"Circuit breaker is OPEN. Try again in "
                            f"{self.recovery_timeout - time_since_failure:.0f}s"
                        )

            try:
                result = await func(*args, **kwargs)

                # Success - reset or close circuit
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker closed after successful request")

                return result

            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.utcnow()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error(
                        f"Circuit breaker opened after {self.failure_count} failures. "
                        f"Will remain open for {self.recovery_timeout}s"
                    )

                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Check if circuit is open
            if self.state == "open":
                if self.last_failure_time:
                    time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
                    if time_since_failure > self.recovery_timeout:
                        self.state = "half-open"
                        logger.info("Circuit breaker entering half-open state")
                    else:
                        raise APIError(
                            f"Circuit breaker is OPEN. Try again in "
                            f"{self.recovery_timeout - time_since_failure:.0f}s"
                        )

            try:
                result = func(*args, **kwargs)

                # Success - reset or close circuit
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker closed after successful request")

                return result

            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.utcnow()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error(
                        f"Circuit breaker opened after {self.failure_count} failures. "
                        f"Will remain open for {self.recovery_timeout}s"
                    )

                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


# =============================================================================
# ERROR AGGREGATOR
# =============================================================================

@dataclass
class ErrorStats:
    """Error statistics"""
    total_errors: int = 0
    error_type_counts: Dict[str, int] = field(default_factory=dict)
    last_error_time: Optional[datetime] = None
    error_rate_per_minute: float = 0.0


class ErrorAggregator:
    """
    Aggregate and analyze errors

    Features:
    - Track error counts by type
    - Calculate error rates
    - Detect error patterns
    - Generate error reports
    """

    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes
        self.errors: List[Dict[str, Any]] = []

    def record_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record an error"""
        error_entry = {
            "timestamp": datetime.utcnow(),
            "type": type(exception).__name__,
            "message": str(exception),
            "context": context or {},
            "exception_data": exception.to_dict() if hasattr(exception, 'to_dict') else {}
        }

        self.errors.append(error_entry)

        # Clean old errors outside window
        self._cleanup_old_errors()

    def _cleanup_old_errors(self):
        """Remove errors outside the time window"""
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)
        self.errors = [e for e in self.errors if e["timestamp"] > cutoff]

    def get_stats(self) -> ErrorStats:
        """Get error statistics"""
        if not self.errors:
            return ErrorStats()

        # Count by type
        type_counts = defaultdict(int)
        for error in self.errors:
            type_counts[error["type"]] += 1

        # Calculate rate
        time_span = (self.errors[-1]["timestamp"] - self.errors[0]["timestamp"]).total_seconds() / 60
        error_rate = len(self.errors) / time_span if time_span > 0 else 0

        return ErrorStats(
            total_errors=len(self.errors),
            error_type_counts=dict(type_counts),
            last_error_time=self.errors[-1]["timestamp"],
            error_rate_per_minute=error_rate
        )

    def get_error_patterns(self) -> List[Dict[str, Any]]:
        """Analyze error patterns"""
        patterns = []

        # Group by type
        type_groups = defaultdict(list)
        for error in self.errors:
            type_groups[error["type"]].append(error)

        # Analyze each group
        for error_type, error_list in type_groups.items():
            if len(error_list) >= 3:  # Only analyze if 3+ occurrences
                patterns.append({
                    "type": error_type,
                    "count": len(error_list),
                    "frequency": len(error_list) / self.window_minutes,
                    "sample_message": error_list[0]["message"]
                })

        return sorted(patterns, key=lambda p: p["count"], reverse=True)

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        stats = self.get_stats()
        patterns = self.get_error_patterns()

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "window_minutes": self.window_minutes,
            "stats": {
                "total_errors": stats.total_errors,
                "error_rate_per_minute": stats.error_rate_per_minute,
                "error_type_counts": stats.error_type_counts
            },
            "patterns": patterns,
            "recent_errors": [
                {
                    "timestamp": e["timestamp"].isoformat(),
                    "type": e["type"],
                    "message": e["message"]
                }
                for e in self.errors[-10:]
            ]
        }


# =============================================================================
# SAFE EXECUTION
# =============================================================================

async def safe_execute(
    func: Callable[..., T],
    *args,
    error_handlers: Optional[List[ErrorHandlerStrategy]] = None,
    fallback_value: Any = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> T:
    """
    Safely execute a function with error handling

    Args:
        func: Function to execute
        *args: Function arguments
        error_handlers: List of error handlers
        fallback_value: Fallback value if all handlers fail
        context: Context for error handlers
        **kwargs: Function keyword arguments

    Returns:
        Function result or fallback value
    """
    context = context or {}
    error_handlers = error_handlers or []

    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"Error in {func.__name__}: {e}")

        # Try error handlers
        for handler in error_handlers:
            if handler.can_handle(e):
                try:
                    result = await handler.handle(e, context)
                    return result
                except Exception as handler_error:  # TODO: Catch specific exception instead of Exception
                    logger.error(f"Error handler failed: {handler_error}")

        # Return fallback
        if fallback_value is not None:
            logger.info(f"Using fallback value for {func.__name__}")
            return fallback_value

        # Re-raise if no fallback
        raise


# =============================================================================
# DEMO / MAIN
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("Error Handling and Retry System")
    print("=" * 60)

    # Demo 1: Retry decorator
    print("\n1. Retry Decorator Demo")
    print("-" * 40)

    call_count = [0]

    @retry_with_exponential_backoff(max_retries=3, base_delay=0.5)
    async def flaky_function():
        call_count[0] += 1
        print(f"  Attempt {call_count[0]}...")

        if call_count[0] < 3:
            raise APIError("Temporary failure")

        print("  ✓ Success!")
        return "Success!"

    result = asyncio.run(flaky_function())
    print(f"Result: {result}")

    # Demo 2: Circuit breaker
    async def demo_circuit_breaker():
        print("\n2. Circuit Breaker Demo")
        print("-" * 40)

        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=5.0)

        @circuit_breaker
        async def unreliable_service():
            raise APIError("Service unavailable")

        # Trigger failures
        try:
            await unreliable_service()
        except APIError as e:
            print(f"  ✓ Failure 1: {e}")

        try:
            await unreliable_service()
        except APIError as e:
            print(f"  ✓ Failure 2: {e}")

        # Circuit should be open now
        try:
            await unreliable_service()
        except APIError as e:
            print(f"  ✓ Circuit breaker triggered: {e}")

        print(f"  Circuit state: {circuit_breaker.state}")

    asyncio.run(demo_circuit_breaker())

    # Demo 3: Error aggregation
    print("\n3. Error Aggregation Demo")
    print("-" * 40)

    aggregator = ErrorAggregator(window_minutes=5)

    # Record various errors
    for i in range(5):
        aggregator.record_error(APIError(f"API error {i+1}"))

    for i in range(3):
        aggregator.record_error(ValidationError(f"Validation error {i+1}"))

    for i in range(2):
        aggregator.record_error(TimeoutError(f"Timeout {i+1}"))

    # Get stats
    stats = aggregator.get_stats()
    print(f"  Total errors: {stats.total_errors}")
    print(f"  Error rate: {stats.error_rate_per_minute:.2f}/min")
    print(f"  Type counts: {stats.error_type_counts}")

    # Get patterns
    patterns = aggregator.get_error_patterns()
    print(f"\n  Error patterns:")
    for pattern in patterns:
        print(f"    - {pattern['type']}: {pattern['count']} occurrences")

    # Generate report
    report = aggregator.generate_report()
    print(f"\n  Full report generated with {len(report['recent_errors'])} recent errors")

    # Demo 4: Safe execution
    print("\n4. Safe Execution Demo")
    print("-" * 40)

    async def risky_operation():
        raise APIError("Operation failed")

    handlers = [
        RetryStrategy(max_retries=2),
        FallbackStrategy(fallback_value="Fallback result")
    ]

    result = asyncio.run(safe_execute(
        risky_operation,
        error_handlers=handlers,
        context={"operation": "test"}
    ))

    print(f"  Result with fallback: {result}")

    print("\n" + "=" * 60)
    print("Error handling demo complete!")
