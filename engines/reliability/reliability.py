"""Reliability Patterns Module

Real, dependency-light (stdlib only) resilience patterns:
  - CircuitBreaker (closed / open / half-open state machine)
  - RetryManager (exponential backoff with jitter)
  - FallbackHandler (no bare except)
  - Bulkhead (concurrency isolation via semaphore)
  - HealthChecker (pluggable component health probes)

Public names preserved: CircuitBreaker, RetryManager, FallbackHandler,
Bulkhead, HealthChecker.
"""
from __future__ import annotations


import threading
import time
from typing import Any, Callable, Tuple, Dict, Optional


class CircuitBreakerOpen(Exception):
    """Raised when a call is attempted while the circuit is open."""


class CircuitBreaker:
    """Circuit breaker with closed / open / half-open states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        success_threshold: int = 2,
        expected_exceptions: Tuple[type, ...] = (Exception,),
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        self.expected_exceptions = expected_exceptions
        self.failures = 0
        self.successes = 0
        self.state = self.CLOSED
        self.opened_at: Optional[float] = None
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self.state == self.OPEN:
                if time.time() - (self.opened_at or 0) >= self.timeout_seconds:
                    self.state = self.HALF_OPEN
                    self.successes = 0
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            if self.state == self.HALF_OPEN:
                self.successes += 1
                if self.successes >= self.success_threshold:
                    self.state = self.CLOSED
                    self.successes = 0
            else:
                self.state = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state == self.HALF_OPEN:
                self.state = self.OPEN
                self.opened_at = time.time()
            elif self.failures >= self.failure_threshold:
                self.state = self.OPEN
                self.opened_at = time.time()

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self.allow_request():
            raise CircuitBreakerOpen(f"Circuit is {self.state}")
        try:
            result = func(*args, **kwargs)
        except self.expected_exceptions:
            self.record_failure()
            raise
        else:
            self.record_success()
            return result

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper

    def reset(self) -> None:
        with self._lock:
            self.failures = 0
            self.successes = 0
            self.state = self.CLOSED
            self.opened_at = None


class RetryManager:
    """Retry logic with exponential backoff and optional jitter."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0, base_delay: float = 0.1, jitter: float = 0.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.base_delay = base_delay
        self.jitter = jitter

    def execute_with_retry(self, operation: Callable[..., Any], exceptions: tuple = (Exception,), *args: Any, **kwargs: Any) -> Any:
        """Execute operation, retrying on the given exceptions with backoff."""
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except exceptions as exc:  # type: ignore[misc]
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.base_delay * (self.backoff_factor ** attempt)
                    if self.jitter:
                        delay += (time.time() % 1) * self.jitter
                    time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def retry(self, exceptions: tuple = (Exception,)) -> Callable:
        """Decorator form of execute_with_retry."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.execute_with_retry(func, exceptions, *args, **kwargs)
            return wrapper
        return decorator


class FallbackHandler:
    """Handler for fallback logic with explicit exception handling."""

    def execute(self, primary: Callable[..., Any], fallback: Callable[..., Any], exceptions: tuple = (Exception,)) -> Any:
        """Execute primary; on the given exceptions, return fallback() result."""
        try:
            return primary()
        except exceptions:
            return fallback()


class Bulkhead:
    """Bulkhead isolation pattern limiting concurrent executions."""

    def __init__(self, max_concurrent: int = 10, max_queue: int = 100):
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self._sem = threading.BoundedSemaphore(max_concurrent)

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self._sem.acquire(blocking=False):
            raise RuntimeError("Bulkhead capacity exceeded")
        try:
            return func(*args, **kwargs)
        finally:
            self._sem.release()


class HealthChecker:
    """Pluggable component health probes."""

    def __init__(self):
        self._probes: Dict[str, Callable[[], bool]] = {}

    def register(self, name: str, probe: Callable[[], bool]) -> None:
        self._probes[name] = probe

    def check(self, name: str) -> Dict[str, Any]:
        probe = self._probes.get(name)
        if probe is None:
            return {"component": name, "status": "unknown"}
        try:
            ok = bool(probe())
        except Exception as exc:  # surface probe errors as unhealthy
            return {"component": name, "status": "unhealthy", "error": str(exc)}
        return {"component": name, "status": "healthy" if ok else "unhealthy"}

    def check_all(self) -> Dict[str, Any]:
        """Return aggregate health of all registered components."""
        if not self._probes:
            return {"status": "healthy", "components": {}}
        components = {name: self.check(name)["status"] for name in self._probes}
        overall = "healthy" if all(s == "healthy" for s in components.values()) else "degraded"
        return {"status": overall, "components": components}
