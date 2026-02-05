"""
RESE-Z3 Bridge Adapter

Provides unified interface for all RESE phases to access Z3 capabilities.

Bridge Features:
1. Unified API for all Z3 operations (solve, optimize, prove, translate)
2. Circuit breaker and retry logic
3. Canonical schema transformation
4. Performance monitoring
5. Structured logging with correlation IDs

API Methods:
- solve_constraints(): For SCE constraint solving
- detect_contradictions(): For DITO ATP
- verify_anomaly(): For ACI constraint checking
- prove_theorem(): For formal verification
- translate_to_lean4(): For Lean 4 integration

Following CLAUDE.md principles:
- Law of the "Air Gap": No imports from core-projects
- Law of Runtime Truth: Verify Z3 via probes
- Law of Idempotency: All operations safe to run 100x
- Circuit Breaker Pattern: Detect Z3 failures
- Structured Logging: JSON with correlation_id
- Law of Configuration Explicitness: All config via environment
- Law of UTC: All timestamps in UTC ISO-8601

Author: RESE Team
Created: 2026-02-04
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
import functools

# Import bridge components
from .rese_z3_client import (
    Z3Client,
    Z3ClientError,
    Z3ClientConnectionError,
    Z3ClientTimeoutError,
    Z3ClientConfig,
    CircuitBreakerConfig,
)
from .rese_z3_schema import (
    CanonicalSolverRequest,
    CanonicalSolverResponse,
    CanonicalTheoremRequest,
    CanonicalTheoremResponse,
    CanonicalVariable,
    CanonicalConstraint,
    ConstraintType,
    ProblemType,
    Z3ResultStatus,
    canonical_to_z3_request,
    z3_to_canonical_response,
    canonical_to_smtlib,
    validate_solver_request,
    validate_theorem_request,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RESEZ3BridgeConfig:
    """
    RESE-Z3 Bridge configuration

    Law of Configuration Explicitness: All config from environment
    """
    # Z3 server configuration
    z3_base_url: str = "http://localhost:8000"
    z3_timeout_ms: int = 30000  # MANDATORY

    # Circuit breaker configuration
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_ms: int = 60000

    # Retry configuration
    max_retries: int = 3
    retry_backoff_ms: int = 1000

    # Caching configuration
    enable_cache: bool = True
    cache_ttl_ms: int = 300000  # 5 minutes

    # Performance monitoring
    enable_monitoring: bool = True

    @classmethod
    def from_env(cls) -> 'RESEZ3BridgeConfig':
        """Load configuration from environment variables"""
        return cls(
            z3_base_url=os.getenv("Z3_BASE_URL", "http://localhost:8000"),
            z3_timeout_ms=int(os.getenv("Z3_TIMEOUT_MS", "30000")),
            circuit_breaker_threshold=int(os.getenv("Z3_CIRCUIT_BREAKER_THRESHOLD", "5")),
            circuit_breaker_timeout_ms=int(os.getenv("Z3_CIRCUIT_BREAKER_TIMEOUT_MS", "60000")),
            max_retries=int(os.getenv("Z3_MAX_RETRIES", "3")),
            retry_backoff_ms=int(os.getenv("Z3_RETRY_BACKOFF_MS", "1000")),
            enable_cache=os.getenv("Z3_ENABLE_CACHE", "true").lower() == "true",
            cache_ttl_ms=int(os.getenv("Z3_CACHE_TTL_MS", "300000")),
            enable_monitoring=os.getenv("Z3_ENABLE_MONITORING", "true").lower() == "true",
        )


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for bridge operations"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = False
    cached: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, success: bool, error: Optional[str] = None):
        """Mark operation as complete"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_name": self.operation_name,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "cached": self.cached,
            "error": self.error,
            "metadata": self.metadata,
        }


class PerformanceMonitor:
    """Performance monitor for bridge operations"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics: List[PerformanceMetrics] = []
        self.logger = logging.getLogger("rese.z3.monitor")

    def start_operation(self, operation_name: str) -> PerformanceMetrics:
        """Start tracking an operation"""
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
        )
        if self.enabled:
            self.metrics.append(metrics)
        return metrics

    def record_success(self, metrics: PerformanceMetrics, **kwargs):
        """Record successful operation"""
        metrics.complete(success=True)
        metrics.metadata.update(kwargs)

        if self.enabled:
            self.logger.debug(json.dumps({
                "level": "debug",
                "component": "PerformanceMonitor",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Operation completed: {metrics.operation_name}",
                "metrics": metrics.to_dict(),
            }))

    def record_failure(self, metrics: PerformanceMetrics, error: str, **kwargs):
        """Record failed operation"""
        metrics.complete(success=False, error=error)
        metrics.metadata.update(kwargs)

        if self.enabled:
            self.logger.warning(json.dumps({
                "level": "warn",
                "component": "PerformanceMonitor",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Operation failed: {metrics.operation_name}",
                "metrics": metrics.to_dict(),
            }))

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.enabled or not self.metrics:
            return {"enabled": False}

        total_ops = len(self.metrics)
        successful_ops = sum(1 for m in self.metrics if m.success)
        failed_ops = total_ops - successful_ops
        cached_ops = sum(1 for m in self.metrics if m.cached)

        durations = [m.duration_ms for m in self.metrics if m.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "enabled": True,
            "total_operations": total_ops,
            "successful_operations": successful_ops,
            "failed_operations": failed_ops,
            "cached_operations": cached_ops,
            "success_rate": successful_ops / total_ops if total_ops > 0 else 0,
            "average_duration_ms": avg_duration,
        }


# =============================================================================
# CACHE
# =============================================================================

class SimpleCache:
    """Simple in-memory cache for Z3 results"""

    def __init__(self, ttl_ms: int = 300000):
        self.ttl_ms = ttl_ms
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.logger = logging.getLogger("rese.z3.cache")

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return "|".join(key_parts)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None

        value, timestamp = self.cache[key]
        age_ms = (time.time() - timestamp) * 1000

        if age_ms > self.ttl_ms:
            del self.cache[key]
            self.logger.debug(f"Cache entry expired: {key}")
            return None

        self.logger.debug(f"Cache hit: {key}")
        return value

    def set(self, key: str, value: Any):
        """Set value in cache"""
        self.cache[key] = (value, time.time())
        self.logger.debug(f"Cache set: {key}")

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "ttl_ms": self.ttl_ms,
        }


# =============================================================================
# MAIN BRIDGE CLASS
# =============================================================================

class RESEZ3Bridge:
    """
    RESE-Z3 Bridge Adapter

    Provides unified interface for all RESE phases to access Z3 capabilities.

    API Methods:
    1. solve_constraints() - For SCE constraint solving
    2. detect_contradictions() - For DITO ATP
    3. verify_anomaly() - For ACI constraint checking
    4. prove_theorem() - For formal verification
    5. translate_to_lean4() - For Lean 4 integration
    """

    def __init__(self, config: Optional[RESEZ3BridgeConfig] = None):
        """
        Initialize RESE-Z3 Bridge

        Args:
            config: Bridge configuration (defaults to environment variables)
        """
        self.config = config or RESEZ3BridgeConfig.from_env()

        # Setup logger
        self.logger = logging.getLogger("rese.z3.bridge")
        self.logger.setLevel(logging.INFO)

        # Setup Z3 client
        z3_client_config = Z3ClientConfig(
            base_url=self.config.z3_base_url,
            timeout_ms=self.config.z3_timeout_ms,
            max_retries=self.config.max_retries,
            retry_backoff_ms=self.config.retry_backoff_ms,
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=self.config.circuit_breaker_threshold,
                timeout_ms=self.config.circuit_breaker_timeout_ms,
            ),
        )
        self.client = Z3Client(z3_client_config)

        # Setup monitoring
        self.monitor = PerformanceMonitor(enabled=self.config.enable_monitoring)

        # Setup cache
        self.cache = SimpleCache(ttl_ms=self.config.cache_ttl_ms) if self.config.enable_cache else None

        # Setup thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.logger.info(json.dumps({
            "level": "info",
            "component": "RESEZ3Bridge",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "RESE-Z3 Bridge initialized",
            "config": asdict(self.config),
        }))

    # ========================================================================
    # UNIFIED API METHODS
    # ========================================================================

    def solve_constraints(
        self,
        variables: List[CanonicalVariable],
        constraints: List[CanonicalConstraint],
        correlation_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> CanonicalSolverResponse:
        """
        Solve constraint satisfaction problem

        For SCE (Symbolic Constraint Engine) phase.
        Finds satisfying assignment for constraints.

        Args:
            variables: List of variables
            constraints: List of constraints
            correlation_id: Correlation ID for tracing
            timeout_ms: Optional timeout override

        Returns:
            CanonicalSolverResponse with solution
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        timeout_ms = timeout_ms or self.config.z3_timeout_ms

        metrics = self.monitor.start_operation("solve_constraints")

        try:
            # Build canonical request
            request = CanonicalSolverRequest(
                problem="",  # Will be generated from constraints
                problem_type=ProblemType.CONSTRAINT_SAT,
                variables=variables,
                constraints=constraints,
                timeout_ms=timeout_ms,
                correlation_id=correlation_id,
            )

            # Check cache
            if self.cache:
                cache_key = self.cache._generate_key("solve", request.to_dict())
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    metrics.cached = True
                    self.monitor.record_success(metrics, cached=True)
                    return CanonicalSolverResponse.from_dict(cached_response)

            # Convert to SMT-LIB
            smtlib = canonical_to_smtlib(request)

            # Call Z3
            z3_response = self.client.solve(smtlib, correlation_id, timeout_ms)

            # Convert to canonical response
            response = z3_to_canonical_response(z3_response, correlation_id)

            # Cache result
            if self.cache:
                self.cache.set(cache_key, response.to_dict())

            self.monitor.record_success(metrics)
            return response

        except Exception as e:
            self.monitor.record_failure(metrics, str(e))
            self.logger.error(json.dumps({
                "level": "error",
                "component": "RESEZ3Bridge",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "solve_constraints failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            raise

    def detect_contradictions(
        self,
        constraints: List[CanonicalConstraint],
        correlation_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Detect contradictions in constraints

        For DITO (Dynamic Inference Trace Optimizer).
        Uses ATP to detect unsatisfiable constraint sets.

        Args:
            constraints: List of constraints to check
            correlation_id: Correlation ID for tracing
            timeout_ms: Optional timeout override

        Returns:
            Tuple of (has_contradiction, counterexample)
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        timeout_ms = timeout_ms or self.config.z3_timeout_ms

        metrics = self.monitor.start_operation("detect_contradictions")

        try:
            # Build canonical request
            request = CanonicalSolverRequest(
                problem="",
                problem_type=ProblemType.CONTRADICTION_DETECTION,
                constraints=constraints,
                timeout_ms=timeout_ms,
                correlation_id=correlation_id,
            )

            # Check cache
            if self.cache:
                cache_key = self.cache._generate_key("contradiction", request.to_dict())
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    metrics.cached = True
                    self.monitor.record_success(metrics, cached=True)
                    return cached_result["has_contradiction"], cached_result.get("counterexample")

            # Convert to SMT-LIB
            smtlib = canonical_to_smtlib(request)

            # Call Z3
            z3_response = self.client.solve(smtlib, correlation_id, timeout_ms)

            # Check if UNSAT (contradiction found)
            has_contradiction = z3_response.get("status") == "unsat"
            counterexample = None

            if not has_contradiction and z3_response.get("status") == "sat":
                # Extract model as counterexample to the contradiction
                model = z3_response.get("model", {})
                counterexample = model.get("assignments", model)

            result = {
                "has_contradiction": has_contradiction,
                "counterexample": counterexample,
            }

            # Cache result
            if self.cache:
                self.cache.set(cache_key, result)

            self.monitor.record_success(metrics)
            return has_contradiction, counterexample

        except Exception as e:
            self.monitor.record_failure(metrics, str(e))
            self.logger.error(json.dumps({
                "level": "error",
                "component": "RESEZ3Bridge",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "detect_contradictions failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            raise

    def verify_anomaly(
        self,
        constraints: List[CanonicalConstraint],
        correlation_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify anomaly constraints

        For ACI (Anomaly Characterization Index) calculator.
        Checks if anomaly violates constraints.

        Args:
            constraints: Constraints to verify
            correlation_id: Correlation ID for tracing
            timeout_ms: Optional timeout override

        Returns:
            Tuple of (is_valid, error_message)
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        timeout_ms = timeout_ms or self.config.z3_timeout_ms

        metrics = self.monitor.start_operation("verify_anomaly")

        try:
            # Build canonical request
            request = CanonicalSolverRequest(
                problem="",
                problem_type=ProblemType.CONSTRAINT_SAT,
                constraints=constraints,
                timeout_ms=timeout_ms,
                correlation_id=correlation_id,
            )

            # Convert to SMT-LIB
            smtlib = canonical_to_smtlib(request)

            # Call Z3
            z3_response = self.client.solve(smtlib, correlation_id, timeout_ms)

            # Check if SAT (constraints are satisfiable, no anomaly)
            is_valid = z3_response.get("status") == "sat"
            error_message = None

            if not is_valid:
                status = z3_response.get("status", "unknown")
                reason = z3_response.get("reason", "Unknown error")
                error_message = f"Constraint verification failed: {status} - {reason}"

            self.monitor.record_success(metrics)
            return is_valid, error_message

        except Exception as e:
            self.monitor.record_failure(metrics, str(e))
            self.logger.error(json.dumps({
                "level": "error",
                "component": "RESEZ3Bridge",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "verify_anomaly failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            raise

    def prove_theorem(
        self,
        theorem_statement: str,
        assumptions: Optional[List[str]] = None,
        variables: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> CanonicalTheoremResponse:
        """
        Prove theorem using Z3

        For formal verification and theorem proving.

        Args:
            theorem_statement: Theorem to prove (SMT-LIB or natural language)
            assumptions: Optional list of assumptions
            variables: Optional variable declarations (name -> type)
            correlation_id: Correlation ID for tracing
            timeout_ms: Optional timeout override

        Returns:
            CanonicalTheoremResponse with proof result
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        timeout_ms = timeout_ms or self.config.z3_timeout_ms

        metrics = self.monitor.start_operation("prove_theorem")

        try:
            # Build canonical theorem request
            request = CanonicalTheoremRequest(
                theorem_statement=theorem_statement,
                assumptions=assumptions or [],
                variables=variables or {},
                timeout_ms=timeout_ms,
                correlation_id=correlation_id,
            )

            # Check cache
            if self.cache:
                cache_key = self.cache._generate_key("prove", request.to_dict())
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    metrics.cached = True
                    self.monitor.record_success(metrics, cached=True)
                    return CanonicalTheoremResponse.from_dict(cached_response)

            # Convert to SMT-LIB
            smtlib = self._theorem_to_smtlib(request)

            # Call Z3
            z3_response = self.client.solve(smtlib, correlation_id, timeout_ms)

            # Parse theorem result
            proven = z3_response.get("status") == "unsat"  # Proof by contradiction
            proof = z3_response.get("proof")
            counterexample = None

            if not proven and z3_response.get("status") == "sat":
                model = z3_response.get("model", {})
                counterexample = model.get("assignments", model)

            response = CanonicalTheoremResponse(
                proven=proven,
                proof=proof,
                counterexample=counterexample,
                execution_time_ms=z3_response.get("execution_time", z3_response.get("time", 0.0)),
                correlation_id=correlation_id,
            )

            # Cache result
            if self.cache:
                self.cache.set(cache_key, response.to_dict())

            self.monitor.record_success(metrics)
            return response

        except Exception as e:
            self.monitor.record_failure(metrics, str(e))
            self.logger.error(json.dumps({
                "level": "error",
                "component": "RESEZ3Bridge",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "prove_theorem failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            raise

    def translate_to_lean4(
        self,
        smtlib_content: str,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Translate SMT-LIB to Lean 4 format

        For integration with Lean 4 theorem prover.

        Args:
            smtlib_content: SMT-LIB2 content
            correlation_id: Correlation ID for tracing

        Returns:
            Lean 4 formalization
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        metrics = self.monitor.start_operation("translate_to_lean4")

        try:
            # Simple translation (placeholder for more sophisticated translation)
            lean4_content = f"-- Translated from SMT-LIB\n"
            lean4_content += f"-- Correlation ID: {correlation_id}\n\n"
            lean4_content += f"{smtlib_content}\n\n"
            lean4_content += f"-- TODO: Implement proper SMT-LIB to Lean 4 translation\n"

            self.monitor.record_success(metrics)
            return lean4_content

        except Exception as e:
            self.monitor.record_failure(metrics, str(e))
            self.logger.error(json.dumps({
                "level": "error",
                "component": "RESEZ3Bridge",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "translate_to_lean4 failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            raise

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _theorem_to_smtlib(self, request: CanonicalTheoremRequest) -> str:
        """Convert theorem request to SMT-LIB format"""
        lines = [
            "; Theorem Proving - RESE-Z3 Bridge",
            f"; Correlation ID: {request.correlation_id}",
            "",
            "(set-logic ALL)",
            "(set-option :produce-models true)",
            "(set-option :produce-proofs true)",
            "(set-option :produce-unsat-cores true)",
        ]

        # Declare variables
        for name, var_type in request.variables.items():
            lines.append(f"(declare-fun {name} () {var_type})")

        # Add assumptions
        for assumption in request.assumptions:
            lines.append(f"(assert {assumption})")

        # Add theorem negation (proof by contradiction)
        lines.append(f"(assert (not {request.theorem_statement}))")

        # Check satisfiability
        lines.append("(check-sat)")

        # Get proof/counterexample
        lines.append("(get-proof)")
        lines.append("(get-model)")

        return "\n".join(lines)

    # ========================================================================
    # HEALTH AND STATS
    # ========================================================================

    def get_health(self) -> Dict[str, Any]:
        """Get bridge health status"""
        z3_health = self.client.check_health()

        return {
            "status": "healthy" if z3_health.get("status") == "ok" else "degraded",
            "z3_server": z3_health,
            "circuit_breaker": self.client.get_stats()["circuit_breaker"],
            "cache": self.cache.get_stats() if self.cache else {"enabled": False},
            "performance": self.monitor.get_summary(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return {
            "config": asdict(self.config),
            "client_stats": self.client.get_stats(),
            "performance_summary": self.monitor.get_summary(),
            "cache_stats": self.cache.get_stats() if self.cache else {"enabled": False},
        }

    def close(self):
        """Close bridge and cleanup resources"""
        self.client.close()
        self.executor.shutdown(wait=True)
        self.logger.info("RESE-Z3 Bridge closed")
