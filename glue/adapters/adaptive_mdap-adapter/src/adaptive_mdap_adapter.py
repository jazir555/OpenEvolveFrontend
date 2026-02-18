"""
Adaptive MDAP Adapter - Anti-Corruption Layer (ACL)

Federation Constitution Compliance:
- Law 1 (Air Gap): No imports from core-projects/
- Law 2 (Runtime Truth): Verified by probes before use
- Law 4 (Idempotency): All operations safe to retry
- Law 5 (Config Explicitness): Required env vars fail fast
- Law 6 (UTC): All timestamps in UTC ISO-8601

This adapter provides:
1. Canonical schema transformation (Python → Canonical → Consumer)
2. Circuit breaker for resilience
3. Retry logic with exponential backoff
4. Structured logging with correlation IDs
5. Health checks and metrics
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import json

# ============================================================================
# Configuration (Law 5: Configuration Explicitness)
# ============================================================================

class AdapterConfigError(Exception):
    """Raised when required configuration is missing."""
    pass


@dataclass
class AdaptiveMDAPAdapterConfig:
    """Configuration for Adaptive MDAP Adapter.

    Environment Variables:
        ADAPTIVE_MDAP_TIMEOUT_MS: Request timeout in milliseconds (REQUIRED)
        ADAPTIVE_MDAP_MAX_RETRIES: Maximum retry attempts (default: 3)
        ADAPTIVE_MDAP_RETRY_DELAY_MS: Initial retry delay in milliseconds (default: 100)
        ADAPTIVE_MDAP_CIRCUIT_BREAKER_THRESHOLD: Failures before opening circuit (default: 5)
        ADAPTIVE_MDAP_CIRCUIT_BREAKER_TIMEOUT_MS: Circuit open timeout in milliseconds (default: 60000)
        ADAPTIVE_MDAP_LOG_LEVEL: Logging level (default: INFO)
    """
    timeout_ms: int
    max_retries: int = 3
    retry_delay_ms: int = 100
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_ms: int = 60000
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AdaptiveMDAPAdapterConfig":
        """Load configuration from environment variables."""
        timeout_ms = os.getenv("ADAPTIVE_MDAP_TIMEOUT_MS")
        if timeout_ms is None:
            raise AdapterConfigError(
                "ADAPTIVE_MDAP_TIMEOUT_MS is required. "
                "Service cannot start without explicit timeout configuration."
            )

        try:
            return cls(
                timeout_ms=int(timeout_ms),
                max_retries=int(os.getenv("ADAPTIVE_MDAP_MAX_RETRIES", "3")),
                retry_delay_ms=int(os.getenv("ADAPTIVE_MDAP_RETRY_DELAY_MS", "100")),
                circuit_breaker_threshold=int(os.getenv("ADAPTIVE_MDAP_CIRCUIT_BREAKER_THRESHOLD", "5")),
                circuit_breaker_timeout_ms=int(os.getenv("ADAPTIVE_MDAP_CIRCUIT_BREAKER_TIMEOUT_MS", "60000")),
                log_level=os.getenv("ADAPTIVE_MDAP_LOG_LEVEL", "INFO")
            )
        except ValueError as e:
            raise AdapterConfigError(f"Invalid environment variable value: {e}")


# ============================================================================
# Canonical Data Models
# ============================================================================

class ProcessingDomain(Enum):
    """Processing domain types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    STRUCTURED_DATA = "structured_data"


class AdaptationMode(Enum):
    """Adaptation learning modes."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    INCREMENTAL = "incremental"
    CONTINUAL = "continual"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class CanonicalComplexityScore:
    """Canonical complexity score representation."""
    overall_score: float
    text_length_score: float = 0.0
    domain_rarity_score: float = 0.0
    depth_score: float = 0.0
    dependency_score: float = 0.0
    feature_weights: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CanonicalSubProblem:
    """Canonical sub-problem representation."""
    id: str
    description: str
    domain: str
    depth: int = 1
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalStrategy:
    """Canonical resource allocation strategy."""
    strategy: str
    n_agents: int
    k_ahead: int
    max_retries: int
    timeout_ms: int


@dataclass
class CanonicalRequest:
    """Canonical request representation."""
    task_id: str
    domain: ProcessingDomain
    input_data: Union[str, Dict[str, Any], List[Any]]
    timeout_ms: int
    adaptation_mode: Optional[AdaptationMode] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CanonicalResponse:
    """Canonical response representation."""
    task_id: str
    status: TaskStatus
    complexity_score: Optional[CanonicalComplexityScore] = None
    strategy: Optional[CanonicalStrategy] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    execution_time_ms: Optional[int] = None


# ============================================================================
# Circuit Breaker (Law 3: Failure Management)
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for preventing cascading failures."""
    threshold: int = 5
    timeout_ms: int = 60000
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    state: CircuitState = CircuitState.CLOSED

    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED

    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.threshold:
            self.state = CircuitState.OPEN

    def allow_request(self) -> bool:
        """Check if request should be allowed based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self.last_failure_time is None:
                return False

            elapsed_ms = (time.time() - self.last_failure_time) * 1000
            if elapsed_ms >= self.timeout_ms:
                self.state = CircuitState.HALF_OPEN
                return True
            return False

        # HALF_OPEN: allow a test request
        return True

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state


# ============================================================================
# Structured Logger (Federation Constitution - Observability)
# ============================================================================

class StructuredLogger:
    """Structured JSON logger with correlation tracking."""

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # JSON handler could be added here
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _log(self, level: str, msg: str, **kwargs):
        """Log with structured context."""
        log_entry = {
            "msg": msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        getattr(self.logger, level.lower())(json.dumps(log_entry))

    def info(self, msg: str, **kwargs):
        self._log("INFO", msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log("WARNING", msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log("ERROR", msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        self._log("DEBUG", msg, **kwargs)


# ============================================================================
# Main Adapter Class
# ============================================================================

class AdaptiveMDAPAdapter:
    """
    Adaptive MDAP Adapter - Anti-Corruption Layer Implementation

    This adapter transforms between external data formats and the canonical
    schema, providing isolation from changes in external systems.

    Architecture:
        External System → ACL (Normalization) → Canonical Schema → Consumer
    """

    def __init__(self, config: Optional[AdaptiveMDAPAdapterConfig] = None):
        """Initialize the adapter with configuration."""
        self.config = config or AdaptiveMDAPAdapterConfig.from_env()
        self.logger = StructuredLogger("AdaptiveMDAPAdapter", self.config.log_level)
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout_ms=self.config.circuit_breaker_timeout_ms
        )
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "circuit_breaker_trips": 0
        }

        # Import MDAP components (verified by probes)
        try:
            from adaptive_mdap import (
                TaskComplexityClassifier,
                AdaptiveMDAPAllocator,
                AdaptiveExecutionController,
                get_health_checker
            )
            self.classifier = TaskComplexityClassifier()
            self.allocator = AdaptiveMDAPAllocator()
            self.controller = AdaptiveExecutionController(
                classifier=self.classifier,
                allocator=self.allocator
            )
            self.health_checker = get_health_checker()
            self.mdap_available = True
            self.logger.info("Adaptive MDAP components loaded successfully")
        except ImportError as e:
            self.mdap_available = False
            self.logger.error(f"Adaptive MDAP not available: {e}")

    # ========================================================================
    # Canonical Transformation Methods (ACL Implementation)
    # ========================================================================

    def to_canonical_subproblem(self, external_subproblem: Any) -> CanonicalSubProblem:
        """Transform external subproblem to canonical format."""
        # Handle dict-like input
        if isinstance(external_subproblem, dict):
            return CanonicalSubProblem(
                id=str(external_subproblem.get("id", "")),
                description=str(external_subproblem.get("description", "")),
                domain=str(external_subproblem.get("domain", "general")),
                depth=int(external_subproblem.get("depth", 1)),
                dependencies=list(external_subproblem.get("dependencies", [])),
                metadata=dict(external_subproblem.get("metadata", {}))
            )

        # Handle object-like input (has attributes)
        return CanonicalSubProblem(
            id=str(getattr(external_subproblem, "id", "")),
            description=str(getattr(external_subproblem, "description", "")),
            domain=str(getattr(external_subproblem, "domain", "general")),
            depth=int(getattr(external_subproblem, "depth", 1)),
            dependencies=list(getattr(external_subproblem, "dependencies", [])),
            metadata=dict(getattr(external_subproblem, "metadata", {}))
        )

    def to_canonical_complexity(self, external_complexity: Any) -> CanonicalComplexityScore:
        """Transform external complexity score to canonical format."""
        if isinstance(external_complexity, dict):
            return CanonicalComplexityScore(
                overall_score=float(external_complexity.get("overall_score", 0.0)),
                text_length_score=float(external_complexity.get("text_length_score", 0.0)),
                domain_rarity_score=float(external_complexity.get("domain_rarity_score", 0.0)),
                depth_score=float(external_complexity.get("depth_score", 0.0)),
                dependency_score=float(external_complexity.get("dependency_score", 0.0)),
                feature_weights=dict(external_complexity.get("feature_weights", {}))
            )

        return CanonicalComplexityScore(
            overall_score=float(getattr(external_complexity, "overall_score", 0.0)),
            text_length_score=float(getattr(external_complexity, "text_length_score", 0.0)),
            domain_rarity_score=float(getattr(external_complexity, "domain_rarity_score", 0.0)),
            depth_score=float(getattr(external_complexity, "depth_score", 0.0)),
            dependency_score=float(getattr(external_complexity, "dependency_score", 0.0)),
            feature_weights=dict(getattr(external_complexity, "feature_weights", {}))
        )

    def to_canonical_strategy(self, external_strategy: Any) -> CanonicalStrategy:
        """Transform external strategy to canonical format."""
        # Extract strategy value
        if isinstance(external_strategy, dict):
            strategy_value = external_strategy.get("strategy", "DIRECT")
            if hasattr(strategy_value, "value"):
                strategy_value = strategy_value.value
        elif hasattr(external_strategy, "strategy"):
            strategy_val = external_strategy.strategy
            strategy_value = strategy_val.value if hasattr(strategy_val, "value") else str(strategy_val)
        else:
            strategy_value = "DIRECT"

        return CanonicalStrategy(
            strategy=str(strategy_value),
            n_agents=int(getattr(external_strategy, "n_agents", 1) if hasattr(external_strategy, "n_agents") else external_strategy.get("n_agents", 1)),
            k_ahead=int(getattr(external_strategy, "k_ahead", 1) if hasattr(external_strategy, "k_ahead") else external_strategy.get("k_ahead", 1)),
            max_retries=int(getattr(external_strategy, "max_retries", 1) if hasattr(external_strategy, "max_retries") else external_strategy.get("max_retries", 1)),
            timeout_ms=int(getattr(external_strategy, "timeout_ms", 30000) if hasattr(external_strategy, "timeout_ms") else external_strategy.get("timeout_ms", 30000))
        )

    # ========================================================================
    # Core Operations
    # ========================================================================

    def analyze_complexity(
        self,
        subproblem: Union[CanonicalSubProblem, Any],
        correlation_id: Optional[str] = None
    ) -> CanonicalResponse:
        """
        Analyze subproblem complexity.

        Args:
            subproblem: Subproblem to analyze (canonical or external format)
            correlation_id: Optional correlation ID for distributed tracing

        Returns:
            CanonicalResponse with complexity score
        """
        self.metrics["requests_total"] += 1
        correlation_id = correlation_id or self._generate_correlation_id()

        self.logger.info(
            "Complexity analysis requested",
            task_id=subproblem.id if isinstance(subproblem, CanonicalSubProblem) else getattr(subproblem, "id", "unknown"),
            correlation_id=correlation_id
        )

        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            self.metrics["circuit_breaker_trips"] += 1
            self.logger.warning(
                "Circuit breaker OPEN - request rejected",
                correlation_id=correlation_id
            )
            return CanonicalResponse(
                task_id=subproblem.id if isinstance(subproblem, CanonicalSubProblem) else getattr(subproblem, "id", "unknown"),
                status=TaskStatus.FAILED,
                error={"code": "CIRCUIT_BREAKER_OPEN", "message": "Service temporarily unavailable"},
                correlation_id=correlation_id
            )

        if not self.mdap_available:
            return CanonicalResponse(
                task_id=subproblem.id if isinstance(subproblem, CanonicalSubProblem) else getattr(subproblem, "id", "unknown"),
                status=TaskStatus.FAILED,
                error={"code": "MDAP_UNAVAILABLE", "message": "Adaptive MDAP not available"},
                correlation_id=correlation_id
            )

        start_time = time.time()

        try:
            # Transform to internal format if needed
            canonical_subproblem = (
                subproblem if isinstance(subproblem, CanonicalSubProblem)
                else self.to_canonical_subproblem(subproblem)
            )

            # Perform complexity analysis with retry
            complexity_score = self._execute_with_retry(
                lambda: self.classifier.compute_complexity(canonical_subproblem),
                correlation_id
            )

            # Transform to canonical format
            canonical_complexity = self.to_canonical_complexity(complexity_score)

            execution_time_ms = int((time.time() - start_time) * 1000)

            self.circuit_breaker.record_success()
            self.metrics["requests_success"] += 1

            self.logger.info(
                "Complexity analysis completed",
                task_id=canonical_subproblem.id,
                complexity_score=canonical_complexity.overall_score,
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id
            )

            return CanonicalResponse(
                task_id=canonical_subproblem.id,
                status=TaskStatus.COMPLETED,
                complexity_score=canonical_complexity,
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self.circuit_breaker.record_failure()
            self.metrics["requests_failed"] += 1

            self.logger.error(
                "Complexity analysis failed",
                error=str(e),
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id
            )

            return CanonicalResponse(
                task_id=canonical_subproblem.id if 'canonical_subproblem' in locals() else "unknown",
                status=TaskStatus.FAILED,
                error={"code": "ANALYSIS_FAILED", "message": str(e)},
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms
            )

    def allocate_resources(
        self,
        complexity_score: Union[CanonicalComplexityScore, Any],
        correlation_id: Optional[str] = None
    ) -> CanonicalResponse:
        """
        Allocate resources based on complexity score.

        Args:
            complexity_score: Complexity score (canonical or external format)
            correlation_id: Optional correlation ID for distributed tracing

        Returns:
            CanonicalResponse with allocation strategy
        """
        self.metrics["requests_total"] += 1
        correlation_id = correlation_id or self._generate_correlation_id()

        self.logger.info(
            "Resource allocation requested",
            correlation_id=correlation_id
        )

        if not self.circuit_breaker.allow_request():
            return CanonicalResponse(
                task_id="allocation",
                status=TaskStatus.FAILED,
                error={"code": "CIRCUIT_BREAKER_OPEN", "message": "Service temporarily unavailable"},
                correlation_id=correlation_id
            )

        if not self.mdap_available:
            return CanonicalResponse(
                task_id="allocation",
                status=TaskStatus.FAILED,
                error={"code": "MDAP_UNAVAILABLE", "message": "Adaptive MDAP not available"},
                correlation_id=correlation_id
            )

        start_time = time.time()

        try:
            # Transform to canonical if needed
            canonical_complexity = (
                complexity_score if isinstance(complexity_score, CanonicalComplexityScore)
                else self.to_canonical_complexity(complexity_score)
            )

            # Allocate resources with retry
            strategy = self._execute_with_retry(
                lambda: self.allocator.allocate_resources(canonical_complexity.overall_score),
                correlation_id
            )

            # Transform to canonical format
            canonical_strategy = self.to_canonical_strategy(strategy)

            execution_time_ms = int((time.time() - start_time) * 1000)

            self.circuit_breaker.record_success()
            self.metrics["requests_success"] += 1

            self.logger.info(
                "Resource allocation completed",
                strategy=canonical_strategy.strategy,
                n_agents=canonical_strategy.n_agents,
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id
            )

            return CanonicalResponse(
                task_id="allocation",
                status=TaskStatus.COMPLETED,
                strategy=canonical_strategy,
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self.circuit_breaker.record_failure()
            self.metrics["requests_failed"] += 1

            self.logger.error(
                "Resource allocation failed",
                error=str(e),
                correlation_id=correlation_id
            )

            return CanonicalResponse(
                task_id="allocation",
                status=TaskStatus.FAILED,
                error={"code": "ALLOCATION_FAILED", "message": str(e)},
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms
            )

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        circuit_state = self.circuit_breaker.get_state()

        health = {
            "status": "healthy" if circuit_state != CircuitState.OPEN else "degraded",
            "circuit_breaker_state": circuit_state.value,
            "mdap_available": self.mdap_available,
            "metrics": self.metrics.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if self.mdap_available:
            try:
                mdap_health = self.health_checker.check()
                health["mdap_health"] = mdap_health
            except Exception as e:
                health["mdap_health"] = {"status": "error", "error": str(e)}

        return health

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _execute_with_retry(self, func, correlation_id: str):
        """Execute function with exponential backoff retry."""
        delay_ms = self.config.retry_delay_ms
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    delay_seconds = delay_ms / 1000.0
                    self.logger.warning(
                        "Retry attempt",
                        attempt=attempt + 1,
                        max_retries=self.config.max_retries,
                        delay_ms=delay_ms,
                        correlation_id=correlation_id
                    )
                    time.sleep(delay_seconds)
                    delay_ms *= 2  # Exponential backoff

        raise last_exception

    def _generate_correlation_id(self) -> str:
        """Generate a correlation ID for distributed tracing."""
        import uuid
        return str(uuid.uuid4())


# ============================================================================
# Convenience Functions
# ============================================================================

_default_adapter: Optional[AdaptiveMDAPAdapter] = None


def get_adapter(config: Optional[AdaptiveMDAPAdapterConfig] = None) -> AdaptiveMDAPAdapter:
    """Get or create the singleton adapter instance."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = AdaptiveMDAPAdapter(config)
    return _default_adapter
