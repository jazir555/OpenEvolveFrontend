"""
Circuit Breaker Pattern Implementation

Prevents cascading failures by temporarily disabling components that are failing.
When a component fails repeatedly, the circuit opens and calls are blocked.
After a timeout, the circuit enters half-open state to test recovery.
"""

import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from functools import wraps
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, calls blocked
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitStats:
    """Statistics for circuit breaker"""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    total_calls: int = 0
    rejected_calls: int = 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_calls == 0:
            return 0.0
        return self.failure_count / self.total_calls


class CircuitBreaker:
    """
    Circuit breaker for protecting components from cascading failures.
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Component failing, calls blocked immediately
    - HALF_OPEN: Testing recovery with limited calls
    
    Configuration:
    - failure_threshold: Number of failures before opening circuit
    - recovery_timeout: Seconds to wait before testing recovery
    - half_open_max_calls: Max calls in half-open state
    - success_threshold: Successes needed to close circuit
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        on_open: Optional[Callable] = None,
        on_close: Optional[Callable] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name (typically component name)
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds to wait before recovery test
            half_open_max_calls: Max calls in half-open state
            success_threshold: Successes needed to close circuit
            on_open: Callback when circuit opens
            on_close: Callback when circuit closes
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.on_open = on_open
        self.on_close = on_close
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._lock = threading.RLock()
        
        logger.info({
            "msg": f"Circuit breaker initialized for {name}",
            "failure_threshold": failure_threshold,
            "recovery_timeout": recovery_timeout
        })
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        with self._lock:
            return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics"""
        with self._lock:
            return CircuitStats(
                failure_count=self._stats.failure_count,
                success_count=self._stats.success_count,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                total_calls=self._stats.total_calls,
                rejected_calls=self._stats.rejected_calls
            )
    
    def can_execute(self) -> bool:
        """
        Check if execution is allowed.
        
        Returns:
            True if call should proceed, False if blocked
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._stats.last_failure_time:
                    elapsed = time.time() - self._stats.last_failure_time
                    if elapsed >= self.recovery_timeout:
                        logger.info({
                            "msg": f"Circuit {self.name} entering half-open state",
                            "open_duration": elapsed
                        })
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
                        self._half_open_successes = 0
                        return True
                
                # Still in timeout period
                self._stats.rejected_calls += 1
                return False
            
            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                else:
                    self._stats.rejected_calls += 1
                    return False
            
            return True
    
    def record_success(self):
        """Record a successful call"""
        with self._lock:
            self._stats.success_count += 1
            self._stats.last_success_time = time.time()
            self._stats.total_calls += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                
                # Check if we can close the circuit
                if self._half_open_successes >= self.success_threshold:
                    logger.info({
                        "msg": f"Circuit {self.name} closed (recovered)",
                        "successes": self._half_open_successes
                    })
                    self._state = CircuitState.CLOSED
                    self._stats.failure_count = 0
                    self._half_open_calls = 0
                    
                    if self.on_close:
                        try:
                            self.on_close(self.name)
                        except Exception as e:
                            logger.error(f"on_close callback failed: {e}")
    
    def record_failure(self):
        """Record a failed call"""
        with self._lock:
            self._stats.failure_count += 1
            self._stats.last_failure_time = time.time()
            self._stats.total_calls += 1
            
            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open state, reopen circuit
                logger.warning({
                    "msg": f"Circuit {self.name} re-opened (recovery failed)",
                    "failure_count": self._stats.failure_count
                })
                self._state = CircuitState.OPEN
                return
            
            # Check if we should open the circuit
            if (self._state == CircuitState.CLOSED and 
                self._stats.failure_count >= self.failure_threshold):
                logger.error({
                    "msg": f"Circuit {self.name} opened (threshold reached)",
                    "failure_count": self._stats.failure_count,
                    "threshold": self.failure_threshold
                })
                self._state = CircuitState.OPEN
                
                if self.on_open:
                    try:
                        self.on_open(self.name)
                    except Exception as e:
                        logger.error(f"on_open callback failed: {e}")
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to wrap a function with circuit breaker.
        
        Usage:
            @CircuitBreaker("my_component")
            def my_function():
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.can_execute():
                raise CircuitBreakerOpenError(
                    f"Circuit {self.name} is OPEN - calls rejected"
                )
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise
        
        return wrapper
    
    def reset(self):
        """Manually reset circuit to CLOSED state"""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._stats = CircuitStats()
            self._half_open_calls = 0
            self._half_open_successes = 0
            
            logger.info({
                "msg": f"Circuit {self.name} manually reset",
                "old_state": old_state.value
            })
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._stats.failure_count,
                "success_count": self._stats.success_count,
                "failure_rate": self._stats.failure_rate,
                "total_calls": self._stats.total_calls,
                "rejected_calls": self._stats.rejected_calls,
                "last_failure": self._stats.last_failure_time,
                "last_success": self._stats.last_success_time,
                "config": {
                    "failure_threshold": self.failure_threshold,
                    "recovery_timeout": self.recovery_timeout,
                    "half_open_max_calls": self.half_open_max_calls,
                    "success_threshold": self.success_threshold
                }
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        **kwargs
    ) -> CircuitBreaker:
        """
        Get existing circuit breaker or create new one.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Failures before opening
            recovery_timeout: Seconds before recovery test
            **kwargs: Additional circuit breaker options
            
        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    **kwargs
                )
            return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        with self._lock:
            return self._breakers.get(name)
    
    def remove(self, name: str):
        """Remove circuit breaker"""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        with self._lock:
            return {
                name: breaker.get_status()
                for name, breaker in self._breakers.items()
            }
    
    def get_open_circuits(self) -> List[str]:
        """Get list of circuit breakers that are currently open"""
        with self._lock:
            return [
                name for name, breaker in self._breakers.items()
                if breaker.state == CircuitState.OPEN
            ]


# Global registry instance
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    **kwargs
) -> CircuitBreaker:
    """
    Get or create a circuit breaker from global registry.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before recovery test
        **kwargs: Additional options
        
    Returns:
        CircuitBreaker instance
    """
    return _global_registry.get_or_create(
        name, failure_threshold, recovery_timeout, **kwargs
    )


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    **kwargs
):
    """
    Decorator for adding circuit breaker to functions.
    
    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before recovery test
        **kwargs: Additional circuit breaker options
        
    Usage:
        @circuit_breaker(failure_threshold=3, recovery_timeout=30)
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        breaker_name = name or func.__name__
        breaker = get_circuit_breaker(
            breaker_name,
            failure_threshold,
            recovery_timeout,
            **kwargs
        )
        return breaker(func)
    return decorator
