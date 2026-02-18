"""
MAKER Engine Adapter - Anti-Corruption Layer (ACL)

Federation Constitution Compliance:
- Law 1 (Air Gap): No imports from core-projects/
- Law 2 (Runtime Truth): Verified by probes before use
- Law 4 (Idempotency): All operations safe to retry
- Law 5 (Config Explicitness): Required env vars fail fast
- Law 6 (UTC): All timestamps in UTC ISO-8601

This adapter provides the same ACL pattern as AdaptiveMDAPAdapter but
specifically for MAKER Engine operations.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import json

from adaptive_mdap_adapter import (
    AdaptiveMDAPAdapterConfig,
    StructuredLogger,
    CircuitBreaker,
    CircuitState
)


# ============================================================================
# Canonical Data Models for MAKER
# ============================================================================

class VotingMode(Enum):
    """MAKER voting modes."""
    SIMPLE = "simple"
    K_AHEAD = "k_ahead"
    WEIGHTED = "weighted"
    CONSENSUS = "consensus"


class RedFlagSeverity(Enum):
    """Red flag severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CanonicalMakerConfig:
    """Canonical MAKER configuration."""
    k_min: int = 2
    k_max: int = 8
    max_votes_per_step: int = 60
    max_steps: int = 1000
    timeout_seconds: int = 90
    checkpoint_interval: int = 25


@dataclass
class CanonicalMakerStep:
    """Canonical MAKER step representation."""
    step_id: str
    prompt_template: str
    task_type: str = "general"
    priority: int = 0
    system_prompt: Optional[str] = None
    expected_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalAgentVote:
    """Canonical agent vote representation."""
    agent_id: str
    vote: Any
    raw_text: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    red_flags: List[str] = field(default_factory=list)


@dataclass
class CanonicalMakerResult:
    """Canonical MAKER execution result."""
    success: bool
    steps_completed: int
    votes_cast: int
    red_flags_detected: int
    final_action: Optional[Any] = None
    agent_votes: List[CanonicalAgentVote] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    terminated_reason: str = ""
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    execution_time_ms: Optional[int] = None


# ============================================================================
# Main MAKER Adapter Class
# ============================================================================

class MakerAdapter:
    """
    MAKER Engine Adapter - Anti-Corruption Layer Implementation

    This adapter transforms between external MAKER data formats and the
    canonical schema, providing isolation from changes in the MAKER system.
    """

    def __init__(self, config: Optional[AdaptiveMDAPAdapterConfig] = None):
        """Initialize the MAKER adapter with configuration."""
        self.config = config or AdaptiveMDAPAdapterConfig.from_env()
        self.logger = StructuredLogger("MakerAdapter", self.config.log_level)
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout_ms=self.config.circuit_breaker_timeout_ms
        )
        self.metrics = {
            "maker_runs_total": 0,
            "maker_runs_success": 0,
            "maker_runs_failed": 0,
            "total_votes_cast": 0,
            "total_red_flags": 0
        }

        # Import MAKER components (verified by probes)
        try:
            from maker_engine import MakerEngine, MakerConfig
            from mdap_engine import RedFlagger, RedFlagRules

            self.MakerEngine = MakerEngine
            self.MakerConfig = MakerConfig
            self.RedFlagger = RedFlagger
            self.RedFlagRules = RedFlagRules
            self.maker_available = True
            self.logger.info("MAKER components loaded successfully")
        except ImportError as e:
            self.maker_available = False
            self.logger.error(f"MAKER engine not available: {e}")

    # ========================================================================
    # Canonical Transformation Methods (ACL Implementation)
    # ========================================================================

    def to_canonical_maker_config(self, external_config: Any) -> CanonicalMakerConfig:
        """Transform external MAKER config to canonical format."""
        if isinstance(external_config, dict):
            return CanonicalMakerConfig(
                k_min=int(external_config.get("k_min", 2)),
                k_max=int(external_config.get("k_max", 8)),
                max_votes_per_step=int(external_config.get("max_votes_per_step", 60)),
                max_steps=int(external_config.get("max_steps", 1000)),
                timeout_seconds=int(external_config.get("timeout_seconds", 90)),
                checkpoint_interval=int(external_config.get("checkpoint_interval", 25))
            )

        return CanonicalMakerConfig(
            k_min=int(getattr(external_config, "k_min", 2)),
            k_max=int(getattr(external_config, "k_max", 8)),
            max_votes_per_step=int(getattr(external_config, "max_votes_per_step", 60)),
            max_steps=int(getattr(external_config, "max_steps", 1000)),
            timeout_seconds=int(getattr(external_config, "timeout_seconds", 90)),
            checkpoint_interval=int(getattr(external_config, "checkpoint_interval", 25))
        )

    def to_canonical_maker_step(self, external_step: Any) -> CanonicalMakerStep:
        """Transform external MAKER step to canonical format."""
        if isinstance(external_step, dict):
            return CanonicalMakerStep(
                step_id=str(external_step.get("step_id", "")),
                prompt_template=str(external_step.get("prompt_template", "")),
                task_type=str(external_step.get("task_type", "general")),
                priority=int(external_step.get("priority", 0)),
                system_prompt=external_step.get("system_prompt"),
                expected_schema=external_step.get("expected_schema"),
                metadata=dict(external_step.get("metadata", {}))
            )

        return CanonicalMakerStep(
            step_id=str(getattr(external_step, "step_id", "")),
            prompt_template=str(getattr(external_step, "prompt_template", "")),
            task_type=str(getattr(external_step, "task_type", "general")),
            priority=int(getattr(external_step, "priority", 0)),
            system_prompt=getattr(external_step, "system_prompt", None),
            expected_schema=getattr(external_step, "expected_schema", None),
            metadata=dict(getattr(external_step, "metadata", {}))
        )

    # ========================================================================
    # Core Operations
    # ========================================================================

    def execute_maker_step(
        self,
        step: CanonicalMakerStep,
        current_state: Any,
        history: List[Dict[str, Any]],
        team: Any,
        correlation_id: Optional[str] = None
    ) -> CanonicalMakerResult:
        """
        Execute a single MAKER voting step.

        Args:
            step: MAKER step to execute (canonical format)
            current_state: Current execution state
            history: Execution history
            team: MAKER team configuration
            correlation_id: Optional correlation ID for distributed tracing

        Returns:
            CanonicalMakerResult with execution results
        """
        self.metrics["maker_runs_total"] += 1
        correlation_id = correlation_id or self._generate_correlation_id()

        self.logger.info(
            "MAKER step execution requested",
            step_id=step.step_id,
            task_type=step.task_type,
            correlation_id=correlation_id
        )

        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            return CanonicalMakerResult(
                success=False,
                steps_completed=0,
                votes_cast=0,
                red_flags_detected=0,
                terminated_reason="CIRCUIT_BREAKER_OPEN",
                error={"code": "CIRCUIT_BREAKER_OPEN", "message": "Service temporarily unavailable"},
                correlation_id=correlation_id
            )

        if not self.maker_available:
            return CanonicalMakerResult(
                success=False,
                steps_completed=0,
                votes_cast=0,
                red_flags_detected=0,
                terminated_reason="MAKER_UNAVAILABLE",
                error={"code": "MAKER_UNAVAILABLE", "message": "MAKER engine not available"},
                correlation_id=correlation_id
            )

        start_time = time.time()

        try:
            # Create MAKER config
            maker_config = self.MakerConfig(
                k_min=2,
                k_max=5,
                max_votes_per_step=30,
                timeout_seconds=self.config.timeout_ms // 1000
            )

            # Create MAKER engine
            engine = self.MakerEngine(team=team, config=maker_config)

            # Transform canonical step to MakerStep
            from maker_engine import MakerStep
            maker_step = MakerStep(
                step_id=step.step_id,
                prompt_template=step.prompt_template,
                expected_schema=step.expected_schema,
                task_type=step.task_type,
                priority=step.priority,
                system_prompt=step.system_prompt,
                metadata=step.metadata
            )

            # Execute MAKER step with retry
            action = self._execute_with_retry(
                lambda: engine._maker_step(maker_step, current_state, history),
                correlation_id
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            self.circuit_breaker.record_success()
            self.metrics["maker_runs_success"] += 1
            self.metrics["total_votes_cast"] += engine.metrics.get("votes_cast", 0)
            self.metrics["total_red_flags"] += engine.metrics.get("red_flags", 0)

            self.logger.info(
                "MAKER step execution completed",
                step_id=step.step_id,
                success=action is not None,
                votes_cast=engine.metrics.get("votes_cast", 0),
                red_flags=engine.metrics.get("red_flags", 0),
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id
            )

            return CanonicalMakerResult(
                success=action is not None,
                steps_completed=1,
                votes_cast=engine.metrics.get("votes_cast", 0),
                red_flags_detected=engine.metrics.get("red_flags", 0),
                final_action=action,
                metrics=engine.metrics,
                terminated_reason="completed" if action else "no_action_selected",
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self.circuit_breaker.record_failure()
            self.metrics["maker_runs_failed"] += 1

            self.logger.error(
                "MAKER step execution failed",
                step_id=step.step_id,
                error=str(e),
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id
            )

            return CanonicalMakerResult(
                success=False,
                steps_completed=0,
                votes_cast=0,
                red_flags_detected=0,
                terminated_reason=f"error: {str(e)}",
                error={"code": "MAKER_EXECUTION_FAILED", "message": str(e)},
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms
            )

    def check_red_flags(
        self,
        raw_text: str,
        candidate: Any,
        expected_schema: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check content for red flags using RedFlagger.

        Args:
            raw_text: Raw text to check
            candidate: Parsed candidate content
            expected_schema: Optional expected schema
            correlation_id: Optional correlation ID

        Returns:
            Tuple of (is_flagged, flag_reasons)
        """
        correlation_id = correlation_id or self._generate_correlation_id()

        self.logger.debug(
            "Red flag check requested",
            text_length=len(raw_text),
            correlation_id=correlation_id
        )

        if not self.maker_available:
            return False, ["MAKER_UNAVAILABLE"]

        try:
            flagger = self.RedFlagger(self.RedFlagRules())
            is_flagged, flag_reason = flagger.is_flagged(raw_text, candidate, expected_schema)

            if is_flagged:
                self.logger.warning(
                    "Red flag detected",
                    reason=flag_reason,
                    correlation_id=correlation_id
                )
                return True, [flag_reason] if flag_reason else ["RED_FLAG_DETECTED"]

            return False, []

        except Exception as e:
            self.logger.error(
                "Red flag check failed",
                error=str(e),
                correlation_id=correlation_id
            )
            return False, [f"CHECK_FAILED: {str(e)}"]

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        circuit_state = self.circuit_breaker.get_state()

        health = {
            "status": "healthy" if circuit_state != CircuitState.OPEN else "degraded",
            "circuit_breaker_state": circuit_state.value,
            "maker_available": self.maker_available,
            "metrics": self.metrics.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

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

_default_maker_adapter: Optional[MakerAdapter] = None


def get_maker_adapter(config: Optional[AdaptiveMDAPAdapterConfig] = None) -> MakerAdapter:
    """Get or create the singleton MAKER adapter instance."""
    global _default_maker_adapter
    if _default_maker_adapter is None:
        _default_maker_adapter = MakerAdapter(config)
    return _default_maker_adapter
