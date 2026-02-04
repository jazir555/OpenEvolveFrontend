"""
RESE Phase IV Adapter: Anti-Corruption Layer (ACL)

Following CLAUDE.md §2.2: The "Anti-Corruption Layer" (ACL)

This adapter transforms canonical requests to Phase IV executor format and
transforms results back to canonical format, with:
- Circuit breaker protection
- Exponential backoff retry
- Timeout enforcement
- Structured logging
- Idempotent operations
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import time
import uuid

# Add schemas to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "schemas"))

from rese_phase4_schemas import (
    ArchitectureAssembly,
    Phase4Config,
    AssemblyStatus,
)

from phase4_executor import (
    ArchitectureAssemblyExecutor,
    StructuredLogger,
    CircuitBreaker,
)


# ============================================================================
# ADAPTER CIRCUIT BREAKER
# ============================================================================

class AdapterCircuitBreaker:
    """Circuit breaker for adapter-level protection."""

    def __init__(self, threshold: int = 5, timeout_ms: int = 60000):
        self.threshold = threshold
        self.timeout_ms = timeout_ms
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        if self.failure_count >= self.threshold:
            self.state = "open"

    def can_execute(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if self.last_failure_time:
                elapsed_ms = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() * 1000
                if elapsed_ms >= self.timeout_ms:
                    self.state = "half_open"
                    return True
            return False
        return True


# ============================================================================
# EXPONENTIAL BACKOFF RETRY
# ============================================================================

def execute_with_retry(
    func,
    max_retries: int = 3,
    base_delay_ms: int = 1000,
    max_delay_ms: int = 10000,
    logger: Optional[StructuredLogger] = None,
) -> Any:
    """
    Execute function with exponential backoff retry.

    Following CLAUDE.md §2.3: Transient Failure → Exponential Backoff Retry.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_error = e

            # Don't retry on certain errors (logic failures)
            if isinstance(e, (ValueError, TypeError, KeyError)):
                if logger:
                    logger.error("Non-retryable error, skipping retries", error=e)
                raise

            if attempt < max_retries:
                # Calculate delay with jitter
                delay_ms = min(base_delay_ms * (2 ** attempt), max_delay_ms)
                jitter_ms = int(delay_ms * 0.1 * (hash(str(attempt)) % 10))
                total_delay_ms = delay_ms + jitter_ms

                if logger:
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_retries}",
                        error=str(e),
                        delay_ms=total_delay_ms,
                    )

                time.sleep(total_delay_ms / 1000.0)
            else:
                if logger:
                    logger.error(f"Max retries exceeded ({max_retries})", error=e)
                raise

    raise last_error


# ============================================================================
# PHASE IV ADAPTER
# ============================================================================

class Phase4Adapter:
    """
    RESE Phase IV Adapter: Anti-Corruption Layer

    Responsibilities:
    1. Transform canonical requests to executor format
    2. Execute with circuit breaker protection
    3. Transform results to canonical format
    4. Handle failures according to CLAUDE.md laws
    """

    def __init__(self, config: Optional[Phase4Config] = None):
        """Initialize adapter with configuration."""
        self.config = config or Phase4Config.from_env()
        self.logger = StructuredLogger(
            "rese-phase4-adapter",
            self.config.correlation_id
        )
        self.circuit_breaker = AdapterCircuitBreaker(
            threshold=5,
            timeout_ms=60000
        )

        # Initialize executor
        self.executor = ArchitectureAssemblyExecutor(self.config)

        self.logger.info(
            "Phase IV Adapter initialized",
            config=self.config.to_dict(),
        )

    def assemble_architecture(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assemble architecture from phase outputs (canonical interface).

        Args:
            request: Canonical request containing:
                - phase1_result: Phase I output
                - phase2_result: Phase II output
                - phase3_result: Phase III output
                - phase1_patterns: Patterns from Phase I
                - phase2_patterns: Patterns from Phase II
                - phase3_patterns: Patterns from Phase III

        Returns:
            Canonical response with architecture assembly

        Raises:
            RuntimeError: If circuit breaker is open
            TimeoutError: If assembly exceeds timeout
            ValueError: If request is invalid
        """
        start_time = time.time()

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            error_msg = "Circuit breaker is open - too many recent failures"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Validate request
        self._validate_request(request)

        try:
            self.logger.info(
                "Processing architecture assembly request",
                request_id=request.get("request_id", "unknown"),
            )

            # Execute with retry
            assembly = execute_with_retry(
                lambda: self._execute_assembly(request),
                max_retries=3,
                base_delay_ms=1000,
                max_delay_ms=10000,
                logger=self.logger,
            )

            # Transform to canonical response
            response = self._to_canonical_response(assembly)

            # Record success
            self.circuit_breaker.record_success()

            elapsed = time.time() - start_time
            self.logger.info(
                "Architecture assembly completed",
                assembly_id=assembly.assembly_id,
                elapsed_seconds=elapsed,
            )

            return response

        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()
            self.logger.error("Architecture assembly failed", error=e)
            raise

    def _validate_request(self, request: Dict[str, Any]):
        """Validate incoming request (Law of Runtime Truth)."""
        if not isinstance(request, dict):
            raise ValueError("Request must be a dictionary")

        # At minimum, we need one phase result
        has_phase1 = "phase1_result" in request or "phase1_patterns" in request
        has_phase2 = "phase2_result" in request or "phase2_patterns" in request
        has_phase3 = "phase3_result" in request or "phase3_patterns" in request

        if not (has_phase1 or has_phase2 or has_phase3):
            raise ValueError(
                "Request must contain at least one phase result or pattern set"
            )

    def _execute_assembly(self, request: Dict[str, Any]) -> ArchitectureAssembly:
        """Execute assembly through executor."""
        return self.executor.execute(
            phase1_result=request.get("phase1_result"),
            phase2_result=request.get("phase2_result"),
            phase3_result=request.get("phase3_result"),
            phase1_patterns=request.get("phase1_patterns", []),
            phase2_patterns=request.get("phase2_patterns", []),
            phase3_patterns=request.get("phase3_patterns", []),
        )

    def _to_canonical_response(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """Transform assembly to canonical response format."""
        return {
            "response_id": str(uuid.uuid4()),
            "assembly": assembly.to_dict(),
            "status": "success",
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "validation_passed": assembly.status == AssemblyStatus.VALIDATED,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint.

        Returns adapter health status including circuit breaker state.
        """
        return {
            "status": "healthy" if self.circuit_breaker.state == "closed" else "degraded",
            "circuit_breaker_state": self.circuit_breaker.state,
            "failure_count": self.circuit_breaker.failure_count,
            "config": self.config.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "Phase4Adapter",
    "Phase4Config",
]
