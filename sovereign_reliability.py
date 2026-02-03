"""
Sovereign-Grade Problem Decomposition System - Reliability & Error Handling
Implements comprehensive error handling, retry logic, and health monitoring.
"""

import logging
import time
import traceback
from typing import Callable, Any, Optional, Dict, List, Type, Union, Tuple
from functools import wraps
from datetime import datetime, timedelta
from enum import Enum
import random

from reliability_config import (
    RETRY_CONFIG,
    CIRCUIT_BREAKER_CONFIG,
    RATE_LIMITER_CONFIG,
)

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SovereignError(Exception):
    """Base exception for sovereign system."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        self.message = message
        self.severity = severity
        self.timestamp = datetime.now()
        super().__init__(message)


class AnalysisError(SovereignError):
    """Error during problem analysis."""


class DecompositionError(SovereignError):
    """Error during decomposition."""


class ValidationError(SovereignError):
    """Error during validation."""


class PersistenceError(SovereignError):
    """Error during database operations."""


class RetryStrategy:
    """Retry strategy with exponential backoff."""
    
    def __init__(
        self,
        max_attempts: int = RETRY_CONFIG["max_attempts"],
        initial_delay: float = RETRY_CONFIG["initial_delay"],
        max_delay: float = RETRY_CONFIG["max_delay"],
        exponential_base: float = RETRY_CONFIG["exponential_base"],
        jitter: bool = RETRY_CONFIG["jitter"],
    ):
        """
        Initialize retry strategy.
        
        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay


def with_retry(
    max_attempts: int = 3,
    retry_on: tuple = (Exception,),
    fallback: Optional[Callable] = None
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        retry_on: Tuple of exceptions to retry on
        fallback: Optional fallback function if all retries fail
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            strategy = RetryStrategy(max_attempts=max_attempts)
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = strategy.get_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            # All retries failed
            if fallback:
                logger.info(f"Using fallback for {func.__name__}")
                return fallback(*args, **kwargs)
            
            raise last_exception
        
        return wrapper
    return decorator


class ErrorHandler:
    """Centralized error handling and reporting."""
    
    def __init__(self):
        """Initialize error handler."""
        self.error_log: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> Dict[str, Any]:
        """
        Handle an error with logging and tracking.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            severity: Error severity level
            
        Returns:
            Dictionary with error information
        """
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'severity': severity.value,
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        # Log error
        self.error_log.append(error_info)
        
        # Track error counts
        error_type = error_info['type']
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {error_info['message']}", exc_info=True)
            self._send_alert(error_info)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(f"ERROR: {error_info['message']}", exc_info=True)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Warning: {error_info['message']}")
        else:
            self.logger.info(f"Info: {error_info['message']}")
        
        return error_info
    
    def _send_alert(self, error_info: Dict[str, Any]):
        """Sends an alert for a critical error."""
        # In a real implementation, this would integrate with an alerting system like PagerDuty or Slack.
        self.logger.critical(f"ALERT: Critical error detected: {error_info['message']}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            'total_errors': len(self.error_log),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_log[-10:] if self.error_log else []
        }
    
    def clear_errors(self) -> None:
        """Clear error log."""
        self.error_log.clear()
        self.error_counts.clear()


class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.checks: Dict[str, Callable] = {}
        self.last_check_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """
        Register a health check.
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy
        """
        self.checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary with health check results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_healthy': True,
            'checks': {}
        }
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                duration = time.time() - start_time
                
                check_result = {
                    'healthy': is_healthy,
                    'duration_ms': duration * 1000,
                    'timestamp': datetime.now().isoformat()
                }
                
                results['checks'][name] = check_result
                
                if not is_healthy:
                    results['overall_healthy'] = False
                    self.logger.warning(f"Health check '{name}' failed")
                
            except Exception as e:
                self.logger.error(f"Health check '{name}' raised exception: {e}")
                results['checks'][name] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results['overall_healthy'] = False
        
        self.last_check_results = results
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if not self.last_check_results:
            return {'status': 'unknown', 'message': 'No health checks run yet'}
        
        if self.last_check_results['overall_healthy']:
            return {'status': 'healthy', 'checks': self.last_check_results}
        else:
            failed_checks = [
                name for name, result in self.last_check_results['checks'].items()
                if not result.get('healthy', False)
            ]
            return {
                'status': 'unhealthy',
                'failed_checks': failed_checks,
                'checks': self.last_check_results
            }


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = CIRCUIT_BREAKER_CONFIG["failure_threshold"],
        timeout: float = CIRCUIT_BREAKER_CONFIG["timeout"],
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting to close circuit
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
        
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        if self.state == "half_open":
            self.state = "closed"
            self.logger.info("Circuit breaker closed")
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout


class RateLimiter:
    """Rate limiting for API calls and resource management."""
    
    def __init__(self, max_requests: int = RATE_LIMITER_CONFIG["max_requests"], time_window: float = RATE_LIMITER_CONFIG["time_window"]):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests per time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[datetime] = []
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self) -> bool:
        """Check if a request is allowed within rate limits."""
        now = datetime.now()
        # Remove old requests outside the time window
        self.requests = [
            req for req in self.requests
            if (now - req).total_seconds() < self.time_window
        ]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        else:
            self.logger.warning(f"Rate limit exceeded: {len(self.requests)}/{self.max_requests} requests in {self.time_window}s")
            return False
    
    def get_wait_time(self) -> float:
        """Get recommended wait time before next request."""
        if not self.requests:
            return 0.0
        
        oldest = self.requests[0]
        time_since_oldest = (datetime.now() - oldest).total_seconds()
        return max(0.0, self.time_window - time_since_oldest)


class AdaptiveRetryStrategy(RetryStrategy):
    """Adaptive retry strategy that learns from failure patterns."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        backoff_multiplier: float = 1.5
    ):
        """
        Initialize adaptive retry strategy.
        
        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter
            backoff_multiplier: Multiplier for adaptive backoff
        """
        super().__init__(max_attempts, initial_delay, max_delay, exponential_base, jitter)
        self.backoff_multiplier = backoff_multiplier
        self.failure_history: List[datetime] = []
        self.success_history: List[datetime] = []
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay with adaptive backoff based on failure patterns."""
        base_delay = super().get_delay(attempt)
        
        # Apply adaptive multiplier based on recent failure rate
        recent_failures = self._get_recent_failures()
        if recent_failures > 3:  # If many recent failures, increase delay
            adaptive_multiplier = 1.0 + (min(2.0, recent_failures / 5.0))  # Max 2x delay
            base_delay *= adaptive_multiplier
        
        return base_delay
    
    def _get_recent_failures(self, minutes: int = 5) -> int:
        """Get number of failures in the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return len([f for f in self.failure_history if f > cutoff])
    
    def record_failure(self) -> None:
        """Record a failure to inform adaptive behavior."""
        self.failure_history.append(datetime.now())
        # Keep at most 100 records
        if len(self.failure_history) > 100:
            self.failure_history = self.failure_history[-100:]
    
    def record_success(self) -> None:
        """Record a success to inform adaptive behavior."""
        self.success_history.append(datetime.now())
        # Keep at most 100 records
        if len(self.success_history) > 100:
            self.success_history = self.success_history[-100:]


class ResourcePool:
    """Resource pooling for connection and object reuse."""
    
    def __init__(self, create_func: Callable, max_size: int = 10, validate_func: Optional[Callable] = None):
        """
        Initialize resource pool.
        
        Args:
            create_func: Function to create new resources
            max_size: Maximum pool size
            validate_func: Optional function to validate resources
        """
        self.create_func = create_func
        self.max_size = max_size
        self.validate_func = validate_func
        self.pool = []
        self.logger = logging.getLogger(__name__)
        
    def acquire(self):
        """Acquire a resource from the pool."""
        if self.pool:
            # Try to find a valid resource
            for resource in self.pool:
                if not self.validate_func or self.validate_func(resource):
                    self.pool.remove(resource)
                    return resource
            # If no valid resources, create new
            return self.create_func()
        else:
            # Pool empty, create new
            return self.create_func()
    
    def release(self, resource):
        """Return a resource to the pool."""
        if len(self.pool) < self.max_size:
            self.pool.append(resource)
        else:
            # Pool is full, dispose of resource
            if hasattr(resource, 'close'):
                try:
                    resource.close()
                except Exception as exc:
                    self.logger.debug(f"Failed to close pooled resource: {exc}")


from tracing import initialize_tracer

tracer = initialize_tracer()


class ResilientComponent:
    """Base class for resilient components with built-in reliability features."""
    
    def __init__(self):
        self.error_handler = get_error_handler()
        self.health_monitor = get_health_monitor()
        self.rate_limiter = RateLimiter()
        self.adaptive_retry = AdaptiveRetryStrategy()
        self.resource_pool = None
        self.logger = logging.getLogger(__name__)
        self.tracer = tracer
    
    def safe_execute(
        self,
        operation: Callable,
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_on: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
        fallback: Optional[Callable] = None
    ) -> Any:
        """
        Safely execute an operation with comprehensive error handling.
        
        Args:
            operation: The operation to execute
            context: Context information for error handling
            max_retries: Maximum number of retry attempts
            retry_on: Exception types to retry on
            fallback: Optional fallback function
        
        Returns:
            Operation result or fallback result
        """
        with self.tracer.start_as_current_span(operation.__name__) as span:
            for attempt in range(max_retries + 1):
                try:
                    # Check rate limits
                    if not self.rate_limiter.is_allowed():
                        wait_time = self.rate_limiter.get_wait_time()
                        self.logger.warning(f"Rate limited, waiting {wait_time:.2f}s")
                        time.sleep(wait_time)
                    
                    result = operation()
                    self.adaptive_retry.record_success()
                    span.set_attribute("status", "success")
                    return result
                except retry_on as e:
                    self.adaptive_retry.record_failure()
                    span.record_exception(e)
                    if attempt == max_retries:
                        # Final attempt failed
                        error_info = self.error_handler.handle_error(
                            e, 
                            context=context,
                            severity=ErrorSeverity.HIGH if attempt > 0 else ErrorSeverity.MEDIUM
                        )
                        span.set_attribute("status", "failure")
                        
                        if fallback:
                            self.logger.info("Using fallback after all retries failed")
                            return fallback()
                        else:
                            raise e
                    else:
                        # Retry with adaptive delay
                        delay = self.adaptive_retry.get_delay(attempt)
                        self.logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed, retrying in {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
    
    def register_health_check(self, name: str):
        """Register a health check for this component."""
        def check():
            try:
                # Default health check - just return True if component is available
                return True
            except Exception as e:
                self.error_handler.handle_error(
                    e, 
                    context={'check': name},
                    severity=ErrorSeverity.HIGH
                )
                return False
        
        self.health_monitor.register_check(name, check)


# Global instances
_global_error_handler = ErrorHandler()
_global_health_monitor = HealthMonitor()


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    return _global_error_handler


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    return _global_health_monitor


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    return RateLimiter()


def register_core_health_checks():
    """Register core health checks for the system."""
    from health_checks import (
        check_database_connectivity,
        check_llm_service_availability,
        check_cache_health,
    )

    health_monitor = get_health_monitor()
    health_monitor.register_check("database_connectivity", check_database_connectivity)
    health_monitor.register_check(
        "llm_service_availability", check_llm_service_availability
    )
    health_monitor.register_check("cache_health", check_cache_health)


def with_error_handling(
    fallback: Optional[Callable] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    log_errors: bool = True
):
    """
    Decorator for automatic error handling.
    
    Args:
        fallback: Optional fallback function
        severity: Error severity for logging
        log_errors: Whether to log errors to the error handler
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    error_handler = get_error_handler()
                    error_handler.handle_error(
                        e,
                        context={'function': func.__name__},
                        severity=severity
                    )
                
                if fallback:
                    return fallback(*args, **kwargs)
                
                raise e
        
        return wrapper
    return decorator
