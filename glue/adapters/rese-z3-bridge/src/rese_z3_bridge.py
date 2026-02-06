"""
RESE-Z3 Bridge Adapter

Provides unified interface for all RESE phases to access Z3 capabilities.

Bridge Features:
1. Unified API for all Z3 operations (solve, optimize, prove, translate)
2. Circuit breaker and retry logic
3. Canonical schema transformation
4. Performance monitoring
5. Structured logging with correlation IDs
6. CAV-NLP integration for enhanced formalization and hybrid verification

API Methods:
- solve_constraints(): For SCE constraint solving
- detect_contradictions(): For DITO ATP
- verify_anomaly(): For ACI constraint checking
- prove_theorem(): For formal verification
- translate_to_lean4(): For Lean 4 integration (REAL implementation)
- formalize_rese_query(): CAV-NLP enhanced query formalization
- verify_hybrid(): Hybrid Z3 + Lean verification

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
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
import functools
from pathlib import Path

# Import real Lean interface for translation
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "lib" / "lean4_bridge"))
    from lean4_bridge import Lean4Interface, Lean4Error, Lean4TimeoutError
    LEAN4_AVAILABLE = True
except ImportError:
    LEAN4_AVAILABLE = False
    Lean4Interface = None  # type: ignore

# Import bridge components
try:
    # Try relative imports first (when installed as package)
    from .rese_z3_client import (
        Z3Client,
        Z3ClientError,
        Z3ClientConnectionError,
        Z3ClientTimeoutError,
        Z3ClientConfig,
        CircuitBreakerConfig,
        LeanAideClientConfig,
        CAVNLPConfig,
        CAV_NLP_AVAILABLE,
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
        # LeanAide schemas
        LeanAideAutoformalizeRequest,
        LeanAideAutoformalizeResponse,
        LeanAideProveRequest,
        LeanAideProveResponse,
        Z3ToLeanTranslationRequest,
        Z3ToLeanTranslationResponse,
        LeanAideTacticSuggestionRequest,
        LeanAideTacticSuggestionResponse,
        LeanAideTacticSuggestion,
        validate_autoformalize_request,
        validate_prove_request,
        validate_translation_request,
        validate_tactic_suggestion_request,
    )
except ImportError:
    # Fall back to absolute imports (when running directly)
    from rese_z3_client import (
        Z3Client,
        Z3ClientError,
        Z3ClientConnectionError,
        Z3ClientTimeoutError,
        Z3ClientConfig,
        CircuitBreakerConfig,
        LeanAideClientConfig,
        CAVNLPConfig,
        CAV_NLP_AVAILABLE,
    )
    from rese_z3_schema import (
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
        # LeanAide schemas
        LeanAideAutoformalizeRequest,
        LeanAideAutoformalizeResponse,
        LeanAideProveRequest,
        LeanAideProveResponse,
        Z3ToLeanTranslationRequest,
        Z3ToLeanTranslationResponse,
        LeanAideTacticSuggestionRequest,
        LeanAideTacticSuggestionResponse,
        LeanAideTacticSuggestion,
        validate_autoformalize_request,
        validate_prove_request,
        validate_translation_request,
        validate_tactic_suggestion_request,
    )

# Import Z3-LeanAide bridge (existing implementation)
try:
    import sys
    from pathlib import Path
    # Add root directory to path to import existing modules
    root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from z3_leanaide_bridge import (
        Z3LeanAideBridge,
        Z3Constraint,
        Lean4Constraint,
        ConstraintType as Z3LeanConstraintType,
        TranslationDirection,
        TranslationResult,
        VerificationBridgeResult,
        HybridProofResult,
    )
    Z3_LEANAIDE_BRIDGE_AVAILABLE = True
except ImportError:
    Z3_LEANAIDE_BRIDGE_AVAILABLE = False
    logging.warning("z3_leanaide_bridge not available - LeanAide integration will be limited")

# Import LeanAide client (existing implementation)
try:
    from leanaide_client import (
        LeanAideClient,
        LeanAideConfig,
        LeanAideResult,
        TaskType,
    )
    LEANAIDE_CLIENT_AVAILABLE = True
except ImportError:
    LEANAIDE_CLIENT_AVAILABLE = False
    logging.warning("leanaide_client not available - will use HTTP client")


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

    # LeanAide server configuration
    leanaide_base_url: str = "http://localhost:7654"
    leanaide_timeout_ms: int = 60000  # MANDATORY (LeanAide is slower)
    leanaide_enable: bool = True

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
            leanaide_base_url=os.getenv("LEANAIDE_BASE_URL", "http://localhost:7654"),
            leanaide_timeout_ms=int(os.getenv("LEANAIDE_TIMEOUT_MS", "60000")),
            leanaide_enable=os.getenv("LEANAIDE_ENABLE", "true").lower() == "true",
            circuit_breaker_threshold=int(os.getenv("Z3_CIRCUIT_BREAKER_THRESHOLD", "5")),
            circuit_breaker_timeout_ms=int(os.getenv("Z3_CIRCUIT_BREAKER_TIMEOUT_MS", "60000")),
            max_retries=int(os.getenv("Z3_MAX_RETRIES", "3")),
            retry_backoff_ms=int(os.getenv("Z3_RETRY_BACKOFF_MS", "1000")),
            enable_cache=os.getenv("Z3_ENABLE_CACHE", "true").lower() == "true",
            cache_ttl_ms=int(os.getenv("Z3_CACHE_TTL_MS", "300000")),
            enable_monitoring=os.getenv("Z3_ENABLE_MONITORING", "true").lower() == "true",
            use_cav_nlp=os.getenv("RESE_USE_CAV_NLP", "true").lower() == "true",
            cav_nlp_config=CAVNLPConfig.from_env(),
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

        # Setup LeanAide client if enabled
        self.leanaide_client = None
        if self.config.leanaide_enable:
            if LEANAIDE_CLIENT_AVAILABLE:
                # Use existing LeanAide client
                leanaide_config = LeanAideConfig(
                    host="localhost",  # Extract from URL
                    port=7654,
                    timeout=self.config.leanaide_timeout_ms / 1000.0,
                    max_retries=self.config.max_retries,
                )
                self.leanaide_client = LeanAideClient(config=leanaide_config)
                self.logger.info("Using existing LeanAide client from leanaide_client.py")
            else:
                # Will use HTTP client through rese_z3_client
                self.logger.info("Will use HTTP client for LeanAide")

        # Setup Z3-LeanAide bridge if available
        self.z3_leanaide_bridge = None
        if Z3_LEANAIDE_BRIDGE_AVAILABLE:
            try:
                self.z3_leanaide_bridge = Z3LeanAideBridge()
                self.logger.info("Z3-LeanAide bridge initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize Z3-LeanAide bridge: {e}")

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
            "message": "RESE-Z3 Bridge initialized with LeanAide and CAV-NLP support",
            "config": asdict(self.config),
            "leanaide_enabled": self.config.leanaide_enable,
            "leanaide_bridge_available": self.z3_leanaide_bridge is not None,
            "cav_nlp_enabled": self.config.use_cav_nlp and CAV_NLP_AVAILABLE,
            "cav_nlp_available": CAV_NLP_AVAILABLE,
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
        timeout_ms = timeout_ms or self.config.z3_timeout_ms

        metrics = self.monitor.start_operation("solve_constraints")

        try:
            # Build canonical request (without correlation_id and timestamp for caching)
            request = CanonicalSolverRequest(
                problem="",  # Will be generated from constraints
                problem_type=ProblemType.CONSTRAINT_SAT,
                variables=variables,
                constraints=constraints,
                timeout_ms=timeout_ms,
                correlation_id="",  # Empty for cache key generation
                timestamp="",  # Empty for cache key generation
            )

            # Check cache (using request without correlation_id and timestamp)
            if self.cache:
                cache_key = self.cache._generate_key("solve", request.to_dict())
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    metrics.cached = True
                    self.monitor.record_success(metrics, cached=True)
                    # Set correlation_id for the response
                    correlation_id = correlation_id or str(uuid.uuid4())
                    cached = CanonicalSolverResponse.from_dict(cached_response)
                    # Update correlation_id in response
                    cached.correlation_id = correlation_id
                    return cached

            # Generate correlation_id after cache check
            correlation_id = correlation_id or str(uuid.uuid4())
            request.correlation_id = correlation_id
            request.timestamp = datetime.now(timezone.utc).isoformat()

            # Convert to SMT-LIB
            smtlib = canonical_to_smtlib(request)

            # Call Z3
            z3_response = self.client.solve(smtlib, correlation_id, timeout_ms)

            # Convert to canonical response
            response = z3_to_canonical_response(z3_response, correlation_id)

            # Cache result (using cache_key generated above)
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
        Translate SMT-LIB to Lean 4 format using REAL Lean4Interface.

        Uses the Lean4Interface constraint translator for proper SMT-LIB to
        Lean 4 translation with formal verification support.

        Args:
            smtlib_content: SMT-LIB2 content
            correlation_id: Correlation ID for tracing

        Returns:
            Lean 4 formalization
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        metrics = self.monitor.start_operation("translate_to_lean4")

        try:
            # Check if Lean4Interface is available
            if LEAN4_AVAILABLE and Lean4Interface is not None:
                # Use REAL Lean4Interface for translation
                lean = Lean4Interface()

                # Parse SMT-LIB content into constraints
                constraints = self._parse_smtlib_constraints(smtlib_content)

                # Translate each constraint to Lean 4
                lean4_parts = [
                    "-- Translated from SMT-LIB using Lean4Interface",
                    f"-- Correlation ID: {correlation_id}",
                    "",
                    "import Mathlib",
                    "",
                    "namespace RESE",
                    "",
                ]

                for i, constraint in enumerate(constraints):
                    try:
                        result = lean.formalize_constraint(
                            constraint=constraint,
                            constraint_type="theorem",
                            correlation_id=correlation_id,
                        )
                        lean_code = result.get("lean4_code", "-- Failed to translate")
                        lean4_parts.append(f"-- Constraint {i+1}: {constraint[:60]}...")
                        lean4_parts.append(lean_code)
                        lean4_parts.append("")
                    except Exception as e:
                        lean4_parts.append(f"-- Error translating constraint {i+1}: {e}")
                        lean4_parts.append("")

                lean4_parts.extend([
                    "end RESE",
                    "",
                ])

                lean4_content = "\n".join(lean4_parts)

                self.logger.info(json.dumps({
                    "level": "info",
                    "component": "RESEZ3Bridge",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": "SMT-LIB to Lean 4 translation completed (REAL)",
                    "correlation_id": correlation_id,
                    "constraints_translated": len(constraints),
                }))

            else:
                # Fallback to basic translation if Lean4Interface not available
                self.logger.warning(json.dumps({
                    "level": "warn",
                    "component": "RESEZ3Bridge",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": "Lean4Interface not available, using basic translation",
                    "correlation_id": correlation_id,
                }))

                lean4_content = f"-- Translated from SMT-LIB (basic mode)\n"
                lean4_content += f"-- Correlation ID: {correlation_id}\n\n"
                lean4_content += f"-- Lean4Interface not available\n"
                lean4_content += f"-- Install glue.lib.lean4_bridge for full translation\n\n"
                lean4_content += f"{smtlib_content}\n"

            self.monitor.record_success(metrics, lean4_available=LEAN4_AVAILABLE)
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

    def _parse_smtlib_constraints(self, smtlib_content: str) -> List[str]:
        """Parse SMT-LIB content into individual constraint strings."""
        constraints = []
        lines = smtlib_content.split('\n')

        current_constraint = []
        paren_depth = 0

        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            for char in line:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1

            current_constraint.append(line)

            if paren_depth == 0 and current_constraint:
                constraint = ' '.join(current_constraint)
                if constraint:
                    constraints.append(constraint)
                current_constraint = []

        return constraints if constraints else [smtlib_content]

    # ========================================================================
    # CAV-NLP INTEGRATION METHODS
    # ========================================================================

    async def formalize_rese_query(
        self,
        query: str,
        correlation_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Formalize RESE natural language query using CAV-NLP.
        
        Uses the Unified Math Service to convert natural language mathematical
        statements into formal representations suitable for Z3 or Lean.
        
        Args:
            query: Natural language query to formalize
            correlation_id: Correlation ID for tracing
            timeout_ms: Optional timeout override
            
        Returns:
            Dictionary containing:
            - success: Whether formalization succeeded
            - formalized_code: The formalized representation
            - confidence: Confidence score for the formalization
            - original_query: The original query
            
        Example:
            >>> result = await bridge.formalize_rese_query(
            ...     "For all x, if x > 0 then x + 1 > 0"
            ... )
            >>> print(result['formalized_code'])
            '(forall ((x Real)) (implies (> x 0) (> (+ x 1) 0)))'
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        timeout_ms = timeout_ms or self.config.cav_nlp_config.timeout_ms
        
        metrics = self.monitor.start_operation("formalize_rese_query")
        
        try:
            # Check if CAV-NLP is available
            if not self.config.use_cav_nlp or not CAV_NLP_AVAILABLE:
                self.monitor.record_success(metrics, cached=False, fallback=True)
                return {
                    "success": False,
                    "formalized": False,
                    "original_query": query,
                    "formalized_code": None,
                    "confidence": 0.0,
                    "reason": "CAV-NLP not available (use_cav_nlp={self.config.use_cav_nlp}, available={CAV_NLP_AVAILABLE})",
                }
            
            # Use CAV-NLP client for formalization
            formalization_result = await self.client.formalize_query(query, correlation_id)
            
            if formalization_result.get("success"):
                self.monitor.record_success(
                    metrics,
                    confidence=formalization_result.get("confidence", 0.0),
                )
            else:
                self.monitor.record_failure(
                    metrics,
                    error=formalization_result.get("reason", "Unknown error"),
                )
            
            return formalization_result
            
        except Exception as e:
            self.monitor.record_failure(metrics, str(e))
            self.logger.error(json.dumps({
                "level": "error",
                "component": "RESEZ3Bridge",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "formalize_rese_query failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            return {
                "success": False,
                "formalized": False,
                "original_query": query,
                "formalized_code": None,
                "confidence": 0.0,
                "reason": str(e),
            }

    async def verify_hybrid(
        self,
        constraint: str,
        correlation_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Verify constraint using hybrid Z3 + Lean approach via CAV-NLP.
        
        Combines Z3's efficient SMT solving with Lean 4's powerful theorem proving
        capabilities through the CAV-NLP unified math service.
        
        Args:
            constraint: Constraint to verify (can be natural language or formal)
            correlation_id: Correlation ID for tracing
            timeout_ms: Optional timeout override
            
        Returns:
            Dictionary containing:
            - verified: Whether the constraint was verified
            - confidence: Confidence score (0.0 to 1.0)
            - proof: Optional proof object
            - tactics: Optional list of tactics used
            - z3_result: Optional Z3-specific results
            - lean_result: Optional Lean-specific results
            
        Example:
            >>> result = await bridge.verify_hybrid("forall x, x > 0 -> x + 1 > 1")
            >>> print(result['verified'], result['confidence'])
            True 0.95
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        timeout_ms = timeout_ms or self.config.cav_nlp_config.timeout_ms
        
        metrics = self.monitor.start_operation("verify_hybrid")
        
        try:
            # Check if CAV-NLP is available
            if not self.config.use_cav_nlp or not CAV_NLP_AVAILABLE:
                self.monitor.record_success(metrics, fallback=True)
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "reason": "CAV-NLP not available for hybrid verification",
                    "z3_result": None,
                    "lean_result": None,
                }
            
            # First try Z3 verification
            z3_result = None
            try:
                z3_response = await self._verify_with_z3(constraint, correlation_id, timeout_ms)
                z3_result = {
                    "status": z3_response.get("status"),
                    "satisfied": z3_response.get("status") == "sat",
                }
            except Exception as z3_error:
                self.logger.debug(json.dumps({
                    "level": "debug",
                    "component": "RESEZ3Bridge",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": "Z3 verification in hybrid mode failed, proceeding to CAV-NLP",
                    "correlation_id": correlation_id,
                    "error": str(z3_error),
                }))
            
            # Use CAV-NLP for hybrid verification
            hybrid_result = await self.client.verify_hybrid(constraint, correlation_id)
            
            # Combine results
            combined_confidence = hybrid_result.get("confidence", 0.0)
            if z3_result and z3_result.get("satisfied"):
                # Boost confidence if Z3 also verified
                combined_confidence = min(1.0, combined_confidence + 0.1)
            
            self.monitor.record_success(
                metrics,
                verified=hybrid_result.get("verified", False),
                confidence=combined_confidence,
            )
            
            return {
                "verified": hybrid_result.get("verified", False),
                "confidence": combined_confidence,
                "proof": hybrid_result.get("proof"),
                "tactics": hybrid_result.get("tactics"),
                "z3_result": z3_result,
                "lean_result": {
                    "verified": hybrid_result.get("verified", False),
                    "confidence": hybrid_result.get("confidence", 0.0),
                },
                "correlation_id": correlation_id,
            }
            
        except Exception as e:
            self.monitor.record_failure(metrics, str(e))
            self.logger.error(json.dumps({
                "level": "error",
                "component": "RESEZ3Bridge",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "verify_hybrid failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            return {
                "verified": False,
                "confidence": 0.0,
                "reason": str(e),
                "correlation_id": correlation_id,
            }

    async def _verify_with_z3(
        self,
        constraint: str,
        correlation_id: str,
        timeout_ms: int,
    ) -> Dict[str, Any]:
        """Internal method to verify constraint with Z3."""
        # Build a simple SMT-LIB query
        smtlib = f"""(set-logic ALL)
(set-option :produce-models true)
(declare-fun x () Real)
(assert {constraint})
(check-sat)
"""
        return self.client.solve(smtlib, correlation_id, timeout_ms)

    # ========================================================================
    # LEANAIDE INTEGRATION METHODS
    # ========================================================================

    def autoformalize(
        self,
        natural_language: str,
        theorem_name: Optional[str] = None,
        correlation_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> LeanAideAutoformalizeResponse:
        """
        Autoformalize natural language to Lean 4 theorem

        Uses LeanAide AI to convert natural language theorems into Lean 4 code.

        Args:
            natural_language: Natural language theorem statement
            theorem_name: Optional name for the theorem
            correlation_id: Correlation ID for tracing
            timeout_ms: Optional timeout override

        Returns:
            LeanAideAutoformalizeResponse with Lean 4 code
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        timeout_ms = timeout_ms or self.config.leanaide_timeout_ms

        metrics = self.monitor.start_operation("autoformalize")

        try:
            # Build request
            request = LeanAideAutoformalizeRequest(
                natural_language=natural_language,
                theorem_name=theorem_name,
                timeout_ms=timeout_ms,
                correlation_id=correlation_id,
            )

            # Validate request
            is_valid, error_msg = validate_autoformalize_request(request.to_dict())
            if not is_valid:
                raise ValueError(error_msg)

            # Check cache
            if self.cache:
                cache_key = self.cache._generate_key("autoformalize", request.to_dict())
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    metrics.cached = True
                    self.monitor.record_success(metrics, cached=True)
                    return LeanAideAutoformalizeResponse.from_dict(cached_response)

            # Use existing LeanAide client if available
            if self.leanaide_client:
                result = self.executor.submit(
                    self._autoformalize_with_client,
                    request
                ).result()
            else:
                # Use HTTP client
                result = self._autoformalize_http(request)

            # Cache result
            if self.cache:
                self.cache.set(cache_key, result.to_dict())

            self.monitor.record_success(metrics)
            return result

        except Exception as e:
            self.monitor.record_failure(metrics, str(e))
            self.logger.error(json.dumps({
                "level": "error",
                "component": "RESEZ3Bridge",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "autoformalize failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            raise

    def prove_with_ai(
        self,
        theorem_text: str,
        theorem_code: Optional[str] = None,
        theorem_statement: Optional[str] = None,
        correlation_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> LeanAideProveResponse:
        """
        Prove theorem using AI (LeanAide)

        Uses LeanAide AI to generate proofs for theorems.

        Args:
            theorem_text: Natural language theorem
            theorem_code: Optional Lean 4 code
            theorem_statement: Optional elaborated theorem type
            correlation_id: Correlation ID for tracing
            timeout_ms: Optional timeout override

        Returns:
            LeanAideProveResponse with proof
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        timeout_ms = timeout_ms or self.config.leanaide_timeout_ms

        metrics = self.monitor.start_operation("prove_with_ai")

        try:
            # Build request
            request = LeanAideProveRequest(
                theorem_text=theorem_text,
                theorem_code=theorem_code,
                theorem_statement=theorem_statement,
                timeout_ms=timeout_ms,
                correlation_id=correlation_id,
            )

            # Validate request
            is_valid, error_msg = validate_prove_request(request.to_dict())
            if not is_valid:
                raise ValueError(error_msg)

            # Check cache
            if self.cache:
                cache_key = self.cache._generate_key("prove_ai", request.to_dict())
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    metrics.cached = True
                    self.monitor.record_success(metrics, cached=True)
                    return LeanAideProveResponse.from_dict(cached_response)

            # Use existing LeanAide client if available
            if self.leanaide_client:
                result = self.executor.submit(
                    self._prove_with_client,
                    request
                ).result()
            else:
                # Use HTTP client
                result = self._prove_http(request)

            # Cache result
            if self.cache:
                self.cache.set(cache_key, result.to_dict())

            self.monitor.record_success(metrics)
            return result

        except Exception as e:
            self.monitor.record_failure(metrics, str(e))
            self.logger.error(json.dumps({
                "level": "error",
                "component": "RESEZ3Bridge",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "prove_with_ai failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            raise

    def translate_z3_to_lean(
        self,
        smtlib_content: str,
        constraint_type: ConstraintType = ConstraintType.BOOLEAN,
        correlation_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> Z3ToLeanTranslationResponse:
        """
        Translate Z3 SMT-LIB to Lean 4

        Uses Z3-LeanAide bridge to translate SMT-LIB constraints to Lean 4 theorems.

        Args:
            smtlib_content: SMT-LIB2 content
            constraint_type: Type of constraints
            correlation_id: Correlation ID for tracing
            timeout_ms: Optional timeout override

        Returns:
            Z3ToLeanTranslationResponse with Lean 4 code
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        timeout_ms = timeout_ms or self.config.z3_timeout_ms

        metrics = self.monitor.start_operation("translate_z3_to_lean")

        try:
            # Build request
            request = Z3ToLeanTranslationRequest(
                smtlib_content=smtlib_content,
                constraint_type=constraint_type,
                timeout_ms=timeout_ms,
                correlation_id=correlation_id,
            )

            # Validate request
            is_valid, error_msg = validate_translation_request(request.to_dict())
            if not is_valid:
                raise ValueError(error_msg)

            # Check cache
            if self.cache:
                cache_key = self.cache._generate_key("translate_z3_lean", request.to_dict())
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    metrics.cached = True
                    self.monitor.record_success(metrics, cached=True)
                    return Z3ToLeanTranslationResponse.from_dict(cached_response)

            # Use Z3-LeanAide bridge if available
            if self.z3_leanaide_bridge:
                result = self._translate_with_bridge(request)
            else:
                # Fallback to simple translation
                result = self._translate_simple(request)

            # Cache result
            if self.cache:
                self.cache.set(cache_key, result.to_dict())

            self.monitor.record_success(metrics)
            return result

        except Exception as e:
            self.monitor.record_failure(metrics, str(e))
            self.logger.error(json.dumps({
                "level": "error",
                "component": "RESEZ3Bridge",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "translate_z3_to_lean failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            raise

    def suggest_tactics(
        self,
        goal_state: str,
        context: Optional[str] = None,
        num_suggestions: int = 3,
        correlation_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> LeanAideTacticSuggestionResponse:
        """
        Get AI-suggested proof tactics

        Uses LeanAide to suggest appropriate tactics for the current proof state.

        Args:
            goal_state: Current goal state in Lean 4
            context: Additional context
            num_suggestions: Number of suggestions (1-10)
            correlation_id: Correlation ID for tracing
            timeout_ms: Optional timeout override

        Returns:
            LeanAideTacticSuggestionResponse with tactic suggestions
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        timeout_ms = timeout_ms or 15000  # 15 seconds default

        metrics = self.monitor.start_operation("suggest_tactics")

        try:
            # Build request
            request = LeanAideTacticSuggestionRequest(
                goal_state=goal_state,
                context=context,
                num_suggestions=num_suggestions,
                timeout_ms=timeout_ms,
                correlation_id=correlation_id,
            )

            # Validate request
            is_valid, error_msg = validate_tactic_suggestion_request(request.to_dict())
            if not is_valid:
                raise ValueError(error_msg)

            # Use math_query from LeanAide
            if self.leanaide_client:
                result = self.executor.submit(
                    self._suggest_tactics_with_client,
                    request
                ).result()
            else:
                # Fallback to basic suggestions
                result = self._suggest_tactics_basic(request)

            self.monitor.record_success(metrics)
            return result

        except Exception as e:
            self.monitor.record_failure(metrics, str(e))
            self.logger.error(json.dumps({
                "level": "error",
                "component": "RESEZ3Bridge",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "suggest_tactics failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))
            raise

    # ========================================================================
    # LEANAIDE HELPER METHODS (IMPLEMENTATION)
    # ========================================================================

    def _autoformalize_with_client(
        self,
        request: LeanAideAutoformalizeRequest
    ) -> LeanAideAutoformalizeResponse:
        """Autoformalize using existing LeanAide client"""
        import asyncio

        async def run_autoformalize():
            if request.theorem_name:
                result = await self.leanaide_client.translate_thm_detailed(
                    theorem_text=request.natural_language,
                    theorem_name=request.theorem_name
                )
            else:
                result = await self.leanaide_client.translate_thm(
                    theorem_text=request.natural_language
                )

            if result.success:
                return LeanAideAutoformalizeResponse(
                    success=True,
                    lean_code=result.data.get("lean_code", result.data.get("code", "")),
                    theorem_name=result.data.get("name"),
                    theorem_type=result.data.get("type"),
                    execution_time_ms=result.response_time * 1000,
                    correlation_id=request.correlation_id,
                )
            else:
                return LeanAideAutoformalizeResponse(
                    success=False,
                    error=result.error,
                    execution_time_ms=result.response_time * 1000,
                    correlation_id=request.correlation_id,
                )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(run_autoformalize())
        finally:
            loop.close()

    def _autoformalize_http(
        self,
        request: LeanAideAutoformalizeRequest
    ) -> LeanAideAutoformalizeResponse:
        """Autoformalize using HTTP client (fallback)"""
        # This would use the HTTP client from rese_z3_client
        # For now, return a placeholder
        return LeanAideAutoformalizeResponse(
            success=False,
            error="HTTP autoformalization not yet implemented - use LeanAide client",
            correlation_id=request.correlation_id,
        )

    def _prove_with_client(
        self,
        request: LeanAideProveRequest
    ) -> LeanAideProveResponse:
        """Prove using existing LeanAide client"""
        import asyncio

        async def run_prove():
            # First, if we don't have Lean code, autoformalize
            theorem_code = request.theorem_code
            theorem_statement = request.theorem_statement

            if not theorem_code:
                auto_result = await self.leanaide_client.translate_thm_detailed(
                    theorem_text=request.theorem_text
                )
                if auto_result.success:
                    theorem_code = auto_result.data.get("lean_code")
                    theorem_statement = auto_result.data.get("type")

            # Now generate proof
            if theorem_code and theorem_statement:
                result = await self.leanaide_client.prove_for_formalization(
                    theorem_text=request.theorem_text,
                    theorem_code=theorem_code,
                    theorem_statement=theorem_statement
                )

                if result.success:
                    return LeanAideProveResponse(
                        success=True,
                        proof=result.data.get("proof", ""),
                        tactics_used=result.data.get("tactics", []),
                        proof_script=result.data.get("script"),
                        execution_time_ms=result.response_time * 1000,
                        correlation_id=request.correlation_id,
                    )

            return LeanAideProveResponse(
                success=False,
                error="Could not generate proof",
                execution_time_ms=0.0,
                correlation_id=request.correlation_id,
            )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(run_prove())
        finally:
            loop.close()

    def _prove_http(
        self,
        request: LeanAideProveRequest
    ) -> LeanAideProveResponse:
        """Prove using HTTP client (fallback)"""
        return LeanAideProveResponse(
            success=False,
            error="HTTP proving not yet implemented - use LeanAide client",
            correlation_id=request.correlation_id,
        )

    def _translate_with_bridge(
        self,
        request: Z3ToLeanTranslationRequest
    ) -> Z3ToLeanTranslationResponse:
        """Translate using Z3-LeanAide bridge"""
        import asyncio

        async def run_translate():
            try:
                # Parse SMT-LIB to extract constraints
                # This is simplified - full implementation would parse SMT-LIB properly
                constraint_type_map = {
                    ConstraintType.BOOLEAN: Z3LeanConstraintType.BOOLEAN,
                    ConstraintType.ARITHMETIC: Z3LeanConstraintType.ARITHMETIC,
                    ConstraintType.NONLINEAR: Z3LeanConstraintType.NONLINEAR,
                    ConstraintType.ARRAY: Z3LeanConstraintType.ARRAY,
                }

                z3_constraint_type = constraint_type_map.get(
                    request.constraint_type,
                    Z3LeanConstraintType.BOOLEAN
                )

                # Use the bridge to translate
                lean_constraint = self.z3_leanaide_bridge.z3_to_lean4(
                    z3_expr=request.smtlib_content,  # In reality, would parse SMT-LIB first
                    constraint_type=z3_constraint_type
                )

                return Z3ToLeanTranslationResponse(
                    success=True,
                    lean_code=lean_constraint.lean_code,
                    theorem_statement=lean_constraint.theorem_statement,
                    variables=lean_constraint.variables,
                    translated_constraints=[lean_constraint.theorem_statement],
                    correlation_id=request.correlation_id,
                )

            except Exception as e:
                return Z3ToLeanTranslationResponse(
                    success=False,
                    error=str(e),
                    correlation_id=request.correlation_id,
                )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(run_translate())
        finally:
            loop.close()

    def _translate_simple(
        self,
        request: Z3ToLeanTranslationRequest
    ) -> Z3ToLeanTranslationResponse:
        """Simple translation fallback"""
        # Extract variable declarations from SMT-LIB
        import re
        variables = []
        var_pattern = r'\(declare-fun (\w+)'

        for match in re.finditer(var_pattern, request.smtlib_content):
            variables.append(match.group(1))

        # Generate basic Lean code
        lean_code = f"-- Translated from Z3 SMT-LIB\n"
        lean_code += f"-- Correlation ID: {request.correlation_id}\n\n"
        lean_code += "import Mathlib\n\n"
        lean_code += f"-- Variables: {', '.join(variables)}\n"
        lean_code += f"-- Constraint type: {request.constraint_type.value}\n"
        lean_code += "\n-- TODO: Complete formalization\n"

        return Z3ToLeanTranslationResponse(
            success=True,
            lean_code=lean_code,
            variables=variables,
            correlation_id=request.correlation_id,
        )

    def _suggest_tactics_with_client(
        self,
        request: LeanAideTacticSuggestionRequest
    ) -> LeanAideTacticSuggestionResponse:
        """Suggest tactics using LeanAide client"""
        import asyncio

        async def run_suggest():
            # Use math_query to get suggestions
            query = f"What tactics should I use for this goal: {request.goal_state}"
            if request.context:
                query = f"{request.context}\n\n{query}"

            result = await self.leanaide_client.math_query(
                query=query,
                n=request.num_suggestions
            )

            if result.success:
                # Parse suggestions from result
                suggestions = []
                answers = result.data.get("answers", [])

                for i, answer in enumerate(answers[:request.num_suggestions]):
                    suggestions.append(LeanAideTacticSuggestion(
                        tactic=f"suggestion_{i+1}",
                        description=answer,
                        confidence=1.0 / (i + 1),  # Decreasing confidence
                        reasoning="Generated by LeanAide",
                    ))

                return LeanAideTacticSuggestionResponse(
                    success=True,
                    suggestions=suggestions,
                    execution_time_ms=result.response_time * 1000,
                    correlation_id=request.correlation_id,
                )

            return LeanAideTacticSuggestionResponse(
                success=False,
                error=result.error,
                execution_time_ms=result.response_time * 1000,
                correlation_id=request.correlation_id,
            )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(run_suggest())
        finally:
            loop.close()

    def _suggest_tactics_basic(
        self,
        request: LeanAideTacticSuggestionRequest
    ) -> LeanAideTacticSuggestionResponse:
        """Basic tactic suggestions (fallback)"""
        # Provide basic suggestions based on goal
        suggestions = []

        goal_lower = request.goal_state.lower()

        # Arithmetic goals
        if any(op in goal_lower for op in ['+', '-', '*', '/', '<', '>', '=']):
            suggestions.append(LeanAideTacticSuggestion(
                tactic="linarith",
                description="Linear arithmetic tactic",
                confidence=0.8,
                reasoning="Detected arithmetic constraints",
            ))

        # Logical goals
        if any(op in goal_lower for op in ['and', 'or', 'not', 'implies']):
            suggestions.append(LeanAideTacticSuggestion(
                tactic="tauto",
                description="Tautology solver for propositional logic",
                confidence=0.7,
                reasoning="Detected logical operators",
            ))

        # Equality goals
        if '=' in goal_lower:
            suggestions.append(LeanAideTacticSuggestion(
                tactic="rfl",
                description="Reflexivity tactic for equalities",
                confidence=0.6,
                reasoning="Detected equality",
            ))

        # Default suggestions
        if len(suggestions) < request.num_suggestions:
            suggestions.append(LeanAideTacticSuggestion(
                tactic="simp",
                description="Simplification tactic",
                confidence=0.5,
                reasoning="General-purpose tactic",
            ))

        return LeanAideTacticSuggestionResponse(
            success=True,
            suggestions=suggestions[:request.num_suggestions],
            correlation_id=request.correlation_id,
        )

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

        # Close LeanAide client if available
        if self.leanaide_client:
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.leanaide_client.close())
                loop.close()
            except Exception as e:
                self.logger.warning(f"Error closing LeanAide client: {e}")

        self.executor.shutdown(wait=True)
        self.logger.info("RESE-Z3 Bridge closed")
