"""
External Configuration for Sovereign Reliability System
"""
from __future__ import annotations



class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""
    pass


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and operation is rejected."""
    pass


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class CircuitState:
    """States for circuit breaker pattern."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class BackoffStrategy:
    """Backoff strategy for retries.
    
    Implements exponential backoff with optional jitter.
    """
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """Initialize backoff strategy.
        
        Args:
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt.
        
        Args:
            attempt: Retry attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        import random
        delay = self.initial_delay * (self.exponential_base ** attempt)
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)
        return min(delay, self.max_delay)


# Retry settings (used by with_retry decorator)
RETRY_CONFIG = {
    "max_attempts": 3,
    "initial_delay": 1.0,  # seconds
    "max_delay": 10.0,     # seconds
    "exponential_base": 2.0,
    "jitter": True,
}

# Circuit breaker settings
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 5,
    "timeout": 30.0,  # seconds
}

# Rate limiter settings
RATE_LIMITER_CONFIG = {
    "max_requests": 100,
    "time_window": 60.0,  # seconds
}

HEALTH_CHECK_CONFIG = {
    'enabled': True,
    'interval': 60
}

class HealthChecker:
    """Stub class for HealthChecker."""
    pass
