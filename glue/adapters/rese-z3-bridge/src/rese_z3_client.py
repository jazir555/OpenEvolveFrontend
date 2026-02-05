"""
RESE-Z3 Bridge Client

HTTP client for communicating with the Z3 API server.

Following CLAUDE.md principles:
- Circuit Breaker Pattern: Detect Z3 failures
- Exponential Backoff Retry: Handle transient failures
- Timeout Enforcement: All requests bounded
- Structured Logging: JSON with correlation_id

Author: RESE Team
Created: 2026-02-04
"""

import json
import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


# =============================================================================
# EXCEPTIONS
# =============================================================================

class Z3ClientError(Exception):
    """Base exception for Z3 client errors"""
    pass


class Z3ClientConnectionError(Z3ClientError):
    """Connection error to Z3 server"""
    pass


class Z3ClientTimeoutError(Z3ClientError):
    """Request timeout to Z3 server"""
    pass


class Z3ClientCircuitBreakerOpenError(Z3ClientError):
    """Circuit breaker is open"""
    pass


# =============================================================================
# CIRCUIT BREAKER STATES
# =============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close after half-open
    timeout_ms: int = 60000     # Time to stay open before half-open
    half_open_max_calls: int = 3  # Max calls in half-open state


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    opened_at: Optional[float] = None
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker for Z3 client

    Law of Runtime Truth: Detect failures via actual calls
    Prevents cascading failures by stopping calls to failing service
    """

    def __init__(self, config: CircuitBreakerConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.stats = CircuitBreakerStats()

    def can_execute(self) -> bool:
        """Check if request can proceed through circuit breaker"""
        now = time.time()

        # Check if we should transition from OPEN to HALF_OPEN
        if self.stats.state == CircuitBreakerState.OPEN:
            if (self.stats.opened_at and
                now - self.stats.opened_at > (self.config.timeout_ms / 1000.0)):
                self._transition_to(CircuitBreakerState.HALF_OPEN)
                self.logger.info(json.dumps({
                    "level": "info",
                    "component": "CircuitBreaker",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": "Circuit breaker transitioned to HALF_OPEN",
                }))
            else:
                return False

        # Check if we should limit HALF_OPEN calls
        if self.stats.state == CircuitBreakerState.HALF_OPEN:
            if self.stats.success_count + self.stats.failure_count >= self.config.half_open_max_calls:
                return False

        return True

    def record_success(self):
        """Record successful call"""
        self.stats.total_successes += 1
        self.stats.total_calls += 1
        self.stats.last_success_time = time.time()

        if self.stats.state == CircuitBreakerState.HALF_OPEN:
            self.stats.success_count += 1
            if self.stats.success_count >= self.config.success_threshold:
                self._transition_to(CircuitBreakerState.CLOSED)
        elif self.stats.state == CircuitBreakerState.CLOSED:
            self.stats.failure_count = 0  # Reset failure count on success

    def record_failure(self):
        """Record failed call"""
        self.stats.total_failures += 1
        self.stats.total_calls += 1
        self.stats.last_failure_time = time.time()

        if self.stats.state == CircuitBreakerState.HALF_OPEN:
            self.stats.failure_count += 1
            if self.stats.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitBreakerState.OPEN)
        elif self.stats.state == CircuitBreakerState.CLOSED:
            self.stats.failure_count += 1
            if self.stats.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitBreakerState.OPEN)

    def _transition_to(self, new_state: CircuitBreakerState):
        """Transition to new state"""
        old_state = self.stats.state
        self.stats.state = new_state

        if new_state == CircuitBreakerState.OPEN:
            self.stats.opened_at = time.time()

        if new_state == CircuitBreakerState.CLOSED:
            self.stats.failure_count = 0
            self.stats.success_count = 0

        self.logger.info(json.dumps({
            "level": "info",
            "component": "CircuitBreaker",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": f"Circuit breaker transitioned: {old_state.value} -> {new_state.value}",
            "stats": self.stats.__dict__,
        }))

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "state": self.stats.state.value,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "last_failure_time": self.stats.last_failure_time,
            "last_success_time": self.stats.last_success_time,
            "opened_at": self.stats.opened_at,
            "total_calls": self.stats.total_calls,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
        }


# =============================================================================
# Z3 CLIENT
# =============================================================================

@dataclass
class Z3ClientConfig:
    """Z3 client configuration"""
    base_url: str = "http://localhost:8000"
    timeout_ms: int = 30000  # Request timeout (mandatory)
    max_retries: int = 3
    retry_backoff_ms: int = 1000
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)


class Z3Client:
    """
    HTTP client for Z3 API server

    Features:
    - Circuit breaker pattern
    - Exponential backoff retry
    - Request timeout enforcement
    - Structured logging with correlation_id
    - Connection pooling

    Law of Runtime Truth: Execute actual HTTP calls to Z3
    Law of Timeout: All requests bounded by timeout
    """

    def __init__(self, config: Z3ClientConfig):
        self.config = config

        # Setup logger
        self.logger = logging.getLogger("rese.z3.client")
        self.logger.setLevel(logging.INFO)

        # Setup circuit breaker
        self.circuit_breaker = CircuitBreaker(config.circuit_breaker, self.logger)

        # Setup HTTP session with retry
        self.session = self._create_session()

        self.logger.info(json.dumps({
            "level": "info",
            "component": "Z3Client",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Z3 client initialized",
            "config": {
                "base_url": config.base_url,
                "timeout_ms": config.timeout_ms,
                "max_retries": config.max_retries,
            },
        }))

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff_ms / 1000.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def check_health(self) -> Dict[str, Any]:
        """
        Check Z3 server health

        Returns:
            Health check response
        """
        try:
            url = f"{self.config.base_url}/health"
            response = self.session.get(
                url,
                timeout=self.config.timeout_ms / 1000.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.warning(json.dumps({
                "level": "warn",
                "component": "Z3Client",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Health check failed",
                "error": str(e),
            }))
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def solve(
        self,
        smtlib_content: str,
        correlation_id: str,
        timeout_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Send solve request to Z3 server

        Args:
            smtlib_content: SMT-LIB2 problem content
            correlation_id: Correlation ID for tracing
            timeout_ms: Optional timeout override

        Returns:
            Z3 solver response

        Raises:
            Z3ClientCircuitBreakerOpenError: If circuit breaker is open
            Z3ClientTimeoutError: If request times out
            Z3ClientConnectionError: If connection fails
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            self.logger.error(json.dumps({
                "level": "error",
                "component": "Z3Client",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Circuit breaker is OPEN, rejecting request",
                "correlation_id": correlation_id,
            }))
            raise Z3ClientCircuitBreakerOpenError("Circuit breaker is OPEN")

        timeout = timeout_ms or self.config.timeout_ms

        try:
            url = f"{self.config.base_url}/solve"
            payload = {
                "smtlib": smtlib_content,
                "timeout_ms": timeout,
                "correlation_id": correlation_id,
            }

            self.logger.debug(json.dumps({
                "level": "debug",
                "component": "Z3Client",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Sending solve request",
                "correlation_id": correlation_id,
                "timeout_ms": timeout,
            }))

            response = self.session.post(
                url,
                json=payload,
                timeout=timeout / 1000.0
            )
            response.raise_for_status()

            result = response.json()

            # Record success
            self.circuit_breaker.record_success()

            self.logger.debug(json.dumps({
                "level": "debug",
                "component": "Z3Client",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Solve request succeeded",
                "correlation_id": correlation_id,
                "result": result.get("status"),
            }))

            return result

        except requests.Timeout as e:
            self.circuit_breaker.record_failure()
            self.logger.error(json.dumps({
                "level": "error",
                "component": "Z3Client",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Solve request timed out",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            raise Z3ClientTimeoutError(f"Request timed out: {str(e)}")

        except requests.ConnectionError as e:
            self.circuit_breaker.record_failure()
            self.logger.error(json.dumps({
                "level": "error",
                "component": "Z3Client",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Connection error to Z3 server",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            raise Z3ClientConnectionError(f"Connection error: {str(e)}")

        except requests.HTTPError as e:
            self.circuit_breaker.record_failure()
            self.logger.error(json.dumps({
                "level": "error",
                "component": "Z3Client",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "HTTP error from Z3 server",
                "correlation_id": correlation_id,
                "status_code": e.response.status_code if e.response else None,
                "error": str(e),
            }))
            raise Z3ClientError(f"HTTP error: {str(e)}")

        except Exception as e:
            self.circuit_breaker.record_failure()
            self.logger.error(json.dumps({
                "level": "error",
                "component": "Z3Client",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Unexpected error in solve request",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            raise Z3ClientError(f"Unexpected error: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "config": {
                "base_url": self.config.base_url,
                "timeout_ms": self.config.timeout_ms,
                "max_retries": self.config.max_retries,
            },
        }

    def close(self):
        """Close HTTP session"""
        self.session.close()
