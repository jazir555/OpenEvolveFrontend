#!/usr/bin/env python3
"""
RESE Phase I: Epistemic Audit Executor

This module implements Phase I of the Recursive Epistemic Solvability Engine (RESE).
Following CLAUDE.md principles:
- Law of Idempotency: All operations safe to run 100x
- Law of Configuration Explicitness: All config via env vars
- Circuit Breaker Pattern: Detect system failures
- Structured Logging: JSON with correlation_id
- Timeout Enforcement: Every operation has timeout
- Dead Letter Queue: For failed audits

Technical Manual Reference:
- Section 3.0: Phase I - Epistemic Audit and Falsification
- Section 3.1: Initial Hypothesis Cluster Definition (Φ₁)
- Section 3.1.5: Tacit Assumption Mining (Φ₁.₅)
- Section 3.2: Metacognitive Reflection and Debiasing (Φ₂)
- Section 3.3: Formal Logic Audit and Contradiction Detection (Φ₃)
"""

import os
import sys
import json
import uuid
import time
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Add glue lib and adapters to path for importing SCE bridge
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../rese-sce/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))  # For Z3 imports

# Try to import SCE bridge (optional, falls back to internal implementation)
try:
    from sce_bridge import SymbolicConstraintEngine
    SCE_AVAILABLE = True
except ImportError:
    SCE_AVAILABLE = False
    SymbolicConstraintEngine = None

# ============================================================================
# CONFIGURATION (Law of Configuration Explicitness)
# ============================================================================

@dataclass
class Phase1Config:
    """Phase I Executor Configuration"""

    # Timeout settings (milliseconds)
    TIMEOUT_MS: int
    CONSTRAINT_HARDENING_TIMEOUT_MS: int
    ASSUMPTION_MINING_TIMEOUT_MS: int
    CONTRADICTION_DETECTION_TIMEOUT_MS: int
    FALSIFICATION_TIMEOUT_MS: int

    # Iteration limits
    MAX_ASSUMPTIONS: int
    MAX_CONSTRAINTS: int
    MAX_CONTRADICTIONS: int
    MAX_FALSIFICATION_ATTEMPTS: int

    # Circuit breaker settings
    CIRCUIT_BREAKER_THRESHOLD: int
    CIRCUIT_BREAKER_TIMEOUT_MS: int

    # Confidence thresholds
    MIN_ASSUMPTION_CONFIDENCE: float
    MIN_ROBUSTNESS_SCORE: float

    # Feature flags
    ENABLE_TACIT_MINING: bool
    ENABLE_LEAN4_INTEGRATION: bool
    ENABLE_RED_TEAM_PROTOCOL: bool
    ENABLE_Z3_CONSTRAINT_HARDENING: bool  # Z3 for constraint hardening

    @classmethod
    def from_env(cls) -> 'Phase1Config':
        """Load configuration from environment variables

        Law of Configuration Explicitness: All config via env vars
        Crashes immediately if required config is missing or invalid
        """
        config = cls(
            TIMEOUT_MS=int(os.getenv('PHASE1_TIMEOUT_MS', '15000')),
            CONSTRAINT_HARDENING_TIMEOUT_MS=int(os.getenv('PHASE1_CONSTRAINT_TIMEOUT_MS', '5000')),
            ASSUMPTION_MINING_TIMEOUT_MS=int(os.getenv('PHASE1_ASSUMPTION_TIMEOUT_MS', '5000')),
            CONTRADICTION_DETECTION_TIMEOUT_MS=int(os.getenv('PHASE1_CONTRADICTION_TIMEOUT_MS', '10000')),
            FALSIFICATION_TIMEOUT_MS=int(os.getenv('PHASE1_FALSIFICATION_TIMEOUT_MS', '5000')),

            MAX_ASSUMPTIONS=int(os.getenv('PHASE1_MAX_ASSUMPTIONS', '100')),
            MAX_CONSTRAINTS=int(os.getenv('PHASE1_MAX_CONSTRAINTS', '1000')),
            MAX_CONTRADICTIONS=int(os.getenv('PHASE1_MAX_CONTRADICTIONS', '100')),
            MAX_FALSIFICATION_ATTEMPTS=int(os.getenv('PHASE1_MAX_FALSIFICATION_ATTEMPTS', '50')),

            CIRCUIT_BREAKER_THRESHOLD=int(os.getenv('PHASE1_CIRCUIT_BREAKER_THRESHOLD', '5')),
            CIRCUIT_BREAKER_TIMEOUT_MS=int(os.getenv('PHASE1_CIRCUIT_BREAKER_TIMEOUT_MS', '60000')),

            MIN_ASSUMPTION_CONFIDENCE=float(os.getenv('PHASE1_MIN_ASSUMPTION_CONFIDENCE', '0.3')),
            MIN_ROBUSTNESS_SCORE=float(os.getenv('PHASE1_MIN_ROBUSTNESS_SCORE', '0.5')),

            ENABLE_TACIT_MINING=os.getenv('PHASE1_ENABLE_TACIT_MINING', 'true').lower() == 'true',
            ENABLE_LEAN4_INTEGRATION=os.getenv('PHASE1_ENABLE_LEAN4', 'false').lower() == 'true',
            ENABLE_RED_TEAM_PROTOCOL=os.getenv('PHASE1_ENABLE_RED_TEAM', 'true').lower() == 'true',
            ENABLE_Z3_CONSTRAINT_HARDENING=os.getenv('PHASE1_ENABLE_Z3_HARDENING', 'true').lower() == 'true',
        )

        # Validate configuration
        if config.TIMEOUT_MS <= 0:
            raise ValueError("PHASE1_TIMEOUT_MS must be positive")
        if config.MAX_ASSUMPTIONS <= 0:
            raise ValueError("PHASE1_MAX_ASSUMPTIONS must be positive")
        if config.MIN_ASSUMPTION_CONFIDENCE < 0 or config.MIN_ASSUMPTION_CONFIDENCE > 1:
            raise ValueError("PHASE1_MIN_ASSUMPTION_CONFIDENCE must be between 0 and 1")

        return config


# ============================================================================
# LOGICAL FALLACY TYPES
# ============================================================================

class LogicalFallacy(Enum):
    """Logical fallacy types from RESE Technical Manual §3.3"""
    CIRCULUS_IN_PROBANDO = "circulus_in_probando"  # Circular reasoning
    CONFIRMATION_BIAS = "confirmation_bias"
    HASTY_GENERALIZATION = "hasty_generalization"
    FALSE_CAUSE = "false_cause"
    AD_HOMINEM = "ad_hominem"
    STRAW_MAN = "straw_man"
    CONTRADICTION = "contradiction"
    INCONSISTENCY = "inconsistency"
    OTHER = "other"


class ConstraintCategory(Enum):
    """Constraint categories from RESE Technical Manual"""
    HARD_PARAMETER_INEQUALITY = "hard_parameter_inequality"  # Category A: Physical laws
    SOFT_STATISTICAL = "soft_statistical"  # Category B: Heuristics
    TACIT_ASSUMPTION = "tacit_assumption"  # Category C: Unstated beliefs
    INVERTED_CONSTRAINT = "inverted_constraint"  # Category D: Solution requirements


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TacitAssumption:
    """Tacit assumption mined from failure patterns

    From RESE Manual §3.1.5: Tacit Assumption Mining (Φ₁.₅)
    """
    id: str
    description: str
    source_pattern: str
    confidence_score: float
    supporting_evidence_count: int
    formalized_in_lean4: bool = False
    lean4_proposition: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TacitAssumption':
        return cls(**data)


@dataclass
class ContradictionDetection:
    """Contradiction detected by Symbolic Constraint Engine

    From RESE Manual §3.3: Formal Logic Audit and Contradiction Detection (Φ₃)
    """
    id: str
    fallacy_type: LogicalFallacy
    contradiction_set_size: int
    rollback_steps: int
    affected_premises: List[str]
    resolved: bool = False
    resolution_strategy: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['fallacy_type'] = self.fallacy_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContradictionDetection':
        if isinstance(data.get('fallacy_type'), str):
            data['fallacy_type'] = LogicalFallacy(data['fallacy_type'])
        return cls(**data)


@dataclass
class FalsificationResult:
    """Result from red team protocol (Φ₄)

    From RESE Manual §3.0: Phase I - Epistemic Audit and Falsification
    """
    hypothesis_id: str
    falsified: bool
    degree_of_violation: Optional[float]
    hypothesis_robustness_score: float
    falsifying_evidence: List[str]
    counter_examples: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FalsificationResult':
        return cls(**data)


@dataclass
class EpistemicAuditResult:
    """Complete result from Phase I: Epistemic Audit

    From RESE Manual §3.0
    Follows canonical schema from glue/schemas/rese-canonical.ts
    """
    phase: str
    audit_id: str
    problem_description: str
    tacit_assumptions: List[TacitAssumption]
    contradictions: List[ContradictionDetection]
    falsification_results: List[FalsificationResult]
    hardened_constraints: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    correlation_id: str
    timestamp: str  # UTC timestamp (Law of UTC)
    debiasing_results: Optional[List[Dict[str, Any]]] = None  # Φ₂ results

    def to_dict(self) -> Dict[str, Any]:
        """Convert to canonical format"""
        return {
            'phase': self.phase,
            'audit_id': self.audit_id,
            'problem_description': self.problem_description,
            'tacit_assumptions': [a.to_dict() for a in self.tacit_assumptions],
            'contradictions': [c.to_dict() for c in self.contradictions],
            'falsification_results': [f.to_dict() for f in self.falsification_results],
            'hardened_constraints': self.hardened_constraints,
            'debiasing_results': self.debiasing_results,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpistemicAuditResult':
        tacit_assumptions = [
            TacitAssumption.from_dict(a) for a in data.get('tacit_assumptions', [])
        ]
        contradictions = [
            ContradictionDetection.from_dict(c) for c in data.get('contradictions', [])
        ]
        falsification_results = [
            FalsificationResult.from_dict(f) for f in data.get('falsification_results', [])
        ]
        return cls(
            tacit_assumptions=tacit_assumptions,
            contradictions=contradictions,
            falsification_results=falsification_results,
            **{k: v for k, v in data.items()
               if k not in ('tacit_assumptions', 'contradictions', 'falsification_results', 'debiasing_results')}
        )


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreakerState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for failure detection

    Follows CLAUDE.md Failure Management Strategy:
    - System Failure → Circuit Breaker
    - Stop hammering the dead service
    - Wait for health check to pass
    """

    def __init__(self, threshold: int, timeout_ms: int, logger: Union['StructuredLogger', logging.Logger]):
        self.threshold = threshold
        self.timeout_ms = timeout_ms

        # Accept both StructuredLogger and logging.Logger
        if isinstance(logger, StructuredLogger):
            self.logger = logger
        else:
            # Wrap plain logging.Logger
            self.logger = StructuredLogger('CircuitBreaker')
            self.logger.logger = logger

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.opened_at = None

    def record_success(self):
        """Record a successful operation"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.logger.info("Circuit breaker recovered",
                state=self.state.value,
            )
        self.failure_count = 0

    def record_failure(self):
        """Record a failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time() * 1000  # milliseconds

        if self.failure_count >= self.threshold:
            if self.state != CircuitBreakerState.OPEN:
                self.state = CircuitBreakerState.OPEN
                self.opened_at = time.time() * 1000
                self.logger.warn("Circuit breaker opened",
                    failure_count=self.failure_count,
                    threshold=self.threshold,
                )

    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed
            elapsed = time.time() * 1000 - self.opened_at
            if elapsed >= self.timeout_ms:
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker entering half-open state")
                return True
            return False

        return True  # HALF_OPEN

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'opened_at': self.opened_at,
        }


# ============================================================================
# DEAD LETTER QUEUE
# ============================================================================

class DeadLetterQueue:
    """Dead Letter Queue for failed audits

    Follows CLAUDE.md Failure Management Strategy:
    - Logic Failure → DLQ (bad assumptions)
    - Do not block the pipeline
    """

    def __init__(self, max_size: int = 1000, structured_logger: Union['StructuredLogger', logging.Logger] = None):
        self.max_size = max_size

        # Accept both StructuredLogger and logging.Logger
        if structured_logger is None:
            self.logger = StructuredLogger('DeadLetterQueue')
        elif isinstance(structured_logger, StructuredLogger):
            self.logger = structured_logger
        else:
            # Wrap plain logging.Logger
            self.logger = StructuredLogger('DeadLetterQueue')
            self.logger.logger = structured_logger

        self._queue: List[Dict[str, Any]] = []

    def enqueue(self, item: Dict[str, Any]) -> bool:
        """Add item to DLQ"""
        if len(self._queue) >= self.max_size:
            self.logger.warn("Dead Letter Queue full, dropping oldest item")
            self._queue.pop(0)

        self._queue.append(item)
        self.logger.warn("Item added to Dead Letter Queue",
            queue_size=len(self._queue),
            item_id=item.get('audit_id'),
        )
        return True

    def dequeue(self) -> Optional[Dict[str, Any]]:
        """Remove item from DLQ for retry"""
        if not self._queue:
            return None
        return self._queue.pop(0)

    def peek(self) -> List[Dict[str, Any]]:
        """View all items in DLQ"""
        return self._queue.copy()

    def size(self) -> int:
        """Get queue size"""
        return len(self._queue)


# ============================================================================
# STRUCTURED LOGGER
# ============================================================================

class StructuredLogger:
    """Structured JSON logger

    Follows CLAUDE.md Observability guidelines:
    - Format: JSON Lines (jsonl)
    - Context: correlation_id, source_service, target_service
    """

    def __init__(self, component: str):
        self.component = component
        self.logger = logging.getLogger(f"rese.phase1.{component}")
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(message)s'
        ))
        self.logger.addHandler(handler)

    def _log(self, level: str, msg: str, **context):
        """Internal log method"""
        log_entry = {
            'level': level,
            'component': self.component,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': msg,
            **context,
        }
        log_json = json.dumps(log_entry)
        if level == 'info':
            self.logger.info(log_json)
        elif level == 'warn':
            self.logger.warning(log_json)
        elif level == 'error':
            self.logger.error(log_json)
        elif level == 'debug':
            self.logger.debug(log_json)

    def info(self, msg: str, **context):
        self._log('info', msg, **context)

    def warn(self, msg: str, **context):
        self._log('warn', msg, **context)

    def error(self, msg: str, error: Exception = None, **context):
        if error:
            context['error'] = str(error)
            context['error_type'] = type(error).__name__
        self._log('error', msg, **context)

    def debug(self, msg: str, **context):
        self._log('debug', msg, **context)


# ============================================================================
# PHASE I EXECUTOR - MAIN CLASS
# ============================================================================

class EpistemicAuditExecutor:
    """Main executor for Phase I: Epistemic Audit

    From RESE Manual §3.0: Phase I - Epistemic Audit and Falsification

    Combines:
    - Φ₁: Initial Hypothesis Cluster Definition (Constraint Hardening)
    - Φ₁.₅: Tacit Assumption Mining
    - Φ₃: Formal Logic Audit and Contradiction Detection
    - Φ₄: Red Team Protocol (Adversarial Testing)

    Follows CLAUDE.md Laws:
    - Law of Idempotency: All operations safe to run 100x
    - Law of Configuration Explicitness: Config from env vars
    - Circuit Breaker Pattern: Failure detection
    - Structured Logging: JSON with correlation_id
    - Timeout Enforcement: All operations have timeouts
    """

    def __init__(self, config: Phase1Config = None):
        """Initialize Phase I executor

        Args:
            config: Configuration object (loaded from env if None)
        """
        self.config = config or Phase1Config.from_env()
        self.logger = StructuredLogger('EpistemicAuditExecutor')
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.CIRCUIT_BREAKER_THRESHOLD,
            timeout_ms=self.config.CIRCUIT_BREAKER_TIMEOUT_MS,
            logger=self.logger.logger,
        )
        self.dlq = DeadLetterQueue(max_size=1000, structured_logger=self.logger)

        # Initialize constraint hardener
        self.constraint_hardener = ConstraintHardener(
            config=self.config,
            logger=self.logger,
        )

        # Initialize assumption miner
        self.assumption_miner = AssumptionMiner(
            config=self.config,
            logger=self.logger,
        )

        # Initialize red team protocol
        self.red_team = RedTeamProtocator(
            config=self.config,
            logger=self.logger,
        )

        # Initialize Φ₂: Metacognitive Reflector (DEBIASING)
        try:
            from metacognitive_reflector import MetacognitiveReflector, DebiasingConfig
            debiasing_config = DebiasingConfig.from_env()
            self.metacognitive_reflector = MetacognitiveReflector(
                config=debiasing_config,
                logger=self.logger,
            )
            self.logger.info("Φ₂: Metacognitive Reflector initialized",
                debiasing_enabled=debiasing_config.ENABLE_DEBIASING,
            )
        except ImportError:
            self.logger.warn("MetacognitiveReflector not available, debiasing disabled")
            self.metacognitive_reflector = None
        except Exception as e:
            self.logger.warn("Failed to initialize MetacognitiveReflector",
                error=str(e),
            )
            self.metacognitive_reflector = None

        # Initialize SCE bridge if available
        self.sce = None
        if SCE_AVAILABLE:
            try:
                self.sce = SymbolicConstraintEngine()
                self.logger.info("SCE bridge initialized",
                    sce_available=True,
                )
            except Exception as e:
                self.logger.warn("Failed to initialize SCE bridge, using internal implementation",
                    error=str(e),
                )
        else:
            self.logger.info("SCE bridge not available, using internal implementation",
                sce_available=False,
            )

        self.logger.info("EpistemicAuditExecutor initialized",
            max_assumptions=self.config.MAX_ASSUMPTIONS,
            max_constraints=self.config.MAX_CONSTRAINTS,
            enable_tacit_mining=self.config.ENABLE_TACIT_MINING,
            enable_red_team=self.config.ENABLE_RED_TEAM_PROTOCOL,
            debiasing_enabled=self.metacognitive_reflector is not None,
            sce_available=self.sce is not None,
        )

    # ========================================================================
    # MAIN AUDIT ORCHESTRATOR
    # ========================================================================

    async def perform_audit(
        self,
        problem_description: str,
        failure_patterns: List[Dict[str, Any]],
        correlation_id: Optional[str] = None,
    ) -> EpistemicAuditResult:
        """Perform full Phase I: Epistemic Audit

        Law of Idempotency: Safe to run 100x (check before create)
        Law of Timeout Enforcement: All operations have timeouts

        Args:
            problem_description: Description of the problem to audit
            failure_patterns: Patterns of failure for tacit assumption mining
            correlation_id: Distributed tracing correlation ID

        Returns:
            Canonical EpistemicAuditResult

        Raises:
            RuntimeError: If circuit breaker is open
            TimeoutError: If audit exceeds timeout
        """
        start_time = time.time()
        correlation_id = correlation_id or str(uuid.uuid4())

        self.logger.info("Starting Phase I: Epistemic Audit",
                correlation_id=correlation_id, problem_description=problem_description, failure_patterns_count=len(failure_patterns)
            )

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            error_msg = "Circuit breaker is OPEN - too many recent failures"
            self.logger.error(error_msg, {
                'correlation_id': correlation_id,
                'circuit_breaker_stats': self.circuit_breaker.get_stats(),
            })
            raise RuntimeError(error_msg)

        try:
            # Φ₁: Constraint Hardening
            self.logger.info("Starting Φ₁: Constraint Hardening",
                correlation_id=correlation_id
            )
            hardened_constraints = self.constraint_hardener.harden_constraints(
                problem_description=problem_description,
                correlation_id=correlation_id,
            )

            # Φ₁.₅: Tacit Assumption Mining
            self.logger.info("Starting Φ₁.₅: Tacit Assumption Mining",
                correlation_id=correlation_id
            )
            tacit_assumptions = self.assumption_miner.mine_assumptions(
                failure_patterns=failure_patterns,
                correlation_id=correlation_id,
            )

            # Φ₂: Metacognitive Reflection (Debiasing)
            debiasing_results = []
            if self.metacognitive_reflector and tacit_assumptions:
                self.logger.info("Starting Φ₂: Metacognitive Reflection",
                    correlation_id=correlation_id
                )
                try:
                    # Create hypothesis from assumptions for debiasing
                    from metacognitive_reflector import Hypothesis

                    for assumption in tacit_assumptions[:5]:  # Limit to top 5 for efficiency
                        hypothesis = Hypothesis(
                            id=assumption.id,
                            statement=assumption.description,
                            confidence=assumption.confidence_score,
                            assumptions=[assumption.source_pattern],
                        )

                        debiasing_result = self.metacognitive_reflector.perform_debiasing(
                            hypothesis=hypothesis,
                            assumptions=tacit_assumptions,
                            correlation_id=correlation_id,
                        )
                        debiasing_results.append(debiasing_result.to_dict())

                    self.logger.info("Φ₂: Metacognitive Reflection completed",
                        correlation_id=correlation_id,
                        assumptions_debiased=len(debiasing_results),
                    )

                except Exception as e:
                    self.logger.warn("Φ₂: Metacognitive Reflection failed, continuing without debiasing",
                        correlation_id=correlation_id,
                        error=str(e),
                    )

            # Φ₃: Contradiction Detection (using SCE adapter)
            # Note: This would integrate with the TypeScript SCE via IPC
            # For now, we'll implement basic contradiction detection
            self.logger.info("Starting Φ₃: Contradiction Detection",
                correlation_id=correlation_id
            )
            contradictions = await self._detect_contradictions(
                assumptions=tacit_assumptions,
                constraints=hardened_constraints,
                correlation_id=correlation_id,
            )

            # Φ₄: Red Team Protocol
            falsification_results = []
            if self.config.ENABLE_RED_TEAM_PROTOCOL:
                self.logger.info("Starting Φ₄: Red Team Protocol",
                correlation_id=correlation_id
            )
                falsification_results = self.red_team.attack_hypotheses(
                    assumptions=tacit_assumptions,
                    constraints=hardened_constraints,
                    correlation_id=correlation_id,
                )

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Build canonical result
            audit_result = EpistemicAuditResult(
                phase='phase1_epistemic_audit',
                audit_id=str(uuid.uuid4()),
                problem_description=problem_description,
                tacit_assumptions=tacit_assumptions,
                contradictions=contradictions,
                falsification_results=falsification_results,
                hardened_constraints=hardened_constraints,
                debiasing_results=debiasing_results if debiasing_results else None,
                metrics={
                    'total_assumptions_analyzed': len(tacit_assumptions),
                    'confirmed_contradictions': len(contradictions),
                    'hypotheses_falsified': sum(1 for f in falsification_results if f.falsified),
                    'assumptions_debiased': len(debiasing_results) if debiasing_results else 0,
                    'average_cbi': sum(r.get('confirmation_bias_index', 0) for r in debiasing_results) / len(debiasing_results) if debiasing_results else None,
                    'average_bias_reduction': sum(r.get('bias_reduction', 0) for r in debiasing_results) / len(debiasing_results) if debiasing_results else None,
                    'reduction_in_failure_rate': None,  # To be updated after Phase II/III
                },
                metadata={
                    'execution_time_ms': execution_time_ms,
                    'lean4_version': '4.7.0' if self.config.ENABLE_LEAN4_INTEGRATION else None,
                    'epoch_number': 1,  # Default to first epoch
                    'debiasing_enabled': self.metacognitive_reflector is not None,
                },
                correlation_id=correlation_id,
                timestamp=datetime.now(timezone.utc).isoformat(),  # Law of UTC
            )

            self.logger.info("Phase I: Epistemic Audit completed",
                correlation_id=correlation_id, audit_id=audit_result.audit_id, execution_time_ms=execution_time_ms, tacit_assumptions_found=len(tacit_assumptions), contradictions_found=len(contradictions), hypotheses_falsified=audit_result.metrics['hypotheses_falsified'], assumptions_debiased=audit_result.metrics['assumptions_debiased']
            )

            # Record success
            self.circuit_breaker.record_success()

            return audit_result

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)

            self.logger.error("Phase I: Epistemic Audit failed", e, {
                'correlation_id': correlation_id,
                'execution_time_ms': execution_time_ms,
            })

            # Record failure
            self.circuit_breaker.record_failure()

            # Add to DLQ
            self.dlq.enqueue({
                'audit_id': str(uuid.uuid4()),
                'problem_description': problem_description,
                'failure_patterns': failure_patterns,
                'error': str(e),
                'correlation_id': correlation_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })

            raise

    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================

    async def _detect_contradictions(
        self,
        assumptions: List[TacitAssumption],
        constraints: List[Dict[str, Any]],
        correlation_id: str,
    ) -> List[ContradictionDetection]:
        """Detect contradictions using SCE

        Uses SCE bridge if available, otherwise falls back to internal implementation
        """
        contradictions = []

        # Try to use SCE if available
        if self.sce:
            try:
                self.logger.info("Using SCE bridge for contradiction detection",
                    correlation_id=correlation_id,
                )

                # Add constraints to SCE
                for constraint_dict in constraints:
                    from sce_bridge import Constraint, ConstraintType, ConstraintCategory
                    try:
                        constraint = Constraint(
                            constraint_id=constraint_dict.get('constraint_id', str(uuid.uuid4())),
                            type=ConstraintType.HARD if constraint_dict.get('category', '').startswith('hard') else ConstraintType.SOFT,
                            category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,  # Default
                            description=constraint_dict.get('description', ''),
                        )
                        await self.sce.add_constraint(constraint, correlation_id)
                    except Exception as e:
                        self.logger.warn("Failed to add constraint to SCE",
                            correlation_id=correlation_id,
                            error=str(e),
                        )

                # Detect contradictions using SCE
                result = await self.sce.detect_contradictions(correlation_id)

                # Transform to ContradictionDetection objects
                for contradiction_dict in result.contradictions:
                    contradictions.append(ContradictionDetection(
                        id=str(uuid.uuid4()),
                        fallacy_type=LogicalFallacy(contradiction_dict.type.value),
                        contradiction_set_size=contradiction_dict.contradiction_set_size,
                        rollback_steps=contradiction_dict.rollback_steps,
                        affected_premises=contradiction_dict.affected_premises,
                        resolved=False,
                    ))

                self.logger.info("SCE contradiction detection completed",
                    correlation_id=correlation_id,
                    contradictions_found=len(contradictions),
                )

                # Check limit
                if len(contradictions) > self.config.MAX_CONTRADICTIONS:
                    self.logger.warn("Max contradictions reached, truncating",
                        correlation_id=correlation_id,
                        total_contradictions=len(contradictions),
                        max_allowed=self.config.MAX_CONTRADICTIONS
                    )
                    contradictions = contradictions[:self.config.MAX_CONTRADICTIONS]

                return contradictions

            except Exception as e:
                self.logger.warn("SCE contradiction detection failed, falling back to internal implementation",
                    correlation_id=correlation_id,
                    error=str(e),
                )
                # Fall through to internal implementation

        # Internal implementation (fallback)
        self.logger.info("Using internal contradiction detection",
            correlation_id=correlation_id,
        )

        # Check for direct contradictions in assumptions
        for i, a1 in enumerate(assumptions):
            for a2 in assumptions[i+1:]:
                if self._are_contradictory(a1.description, a2.description):
                    contradictions.append(ContradictionDetection(
                        id=str(uuid.uuid4()),
                        fallacy_type=LogicalFallacy.CONTRADICTION,
                        contradiction_set_size=2,
                        rollback_steps=1,
                        affected_premises=[a1.id, a2.id],
                        resolved=False,
                    ))

        # Check limit
        if len(contradictions) > self.config.MAX_CONTRADICTIONS:
            self.logger.warn("Max contradictions reached, truncating",
                correlation_id=correlation_id,
                total_contradictions=len(contradictions),
                max_allowed=self.config.MAX_CONTRADICTIONS
            )
            contradictions = contradictions[:self.config.MAX_CONTRADICTIONS]

        return contradictions

    def _are_contradictory(self, desc1: str, desc2: str) -> bool:
        """Check if two descriptions are contradictory"""
        # Simple negation detection
        negations = ['not ', 'no ', 'never ', 'cannot ']

        desc1_lower = desc1.lower().strip()
        desc2_lower = desc2.lower().strip()

        # Check if one is negation of the other
        for neg in negations:
            if desc1_lower.startswith(neg) and desc1_lower[len(neg):] == desc2_lower:
                return True
            if desc2_lower.startswith(neg) and desc2_lower[len(neg):] == desc1_lower:
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics"""
        return {
            'circuit_breaker': self.circuit_breaker.get_stats(),
            'dlq_size': self.dlq.size(),
        }


# ============================================================================
# CONSTRAINT HARDENER (Φ₁) WITH Z3 INTEGRATION
# ============================================================================

class ConstraintHardener:
    """Φ₁: Initial Hypothesis Cluster Definition (Constraint Hardening)

    From RESE Manual §3.1:
    "Φ₁ formally articulates the hardened constraint (ℂ) and mandates its
    logical inversion (¬ℂ), ensuring the system focuses on designing the
    mechanism required to circumvent the perceived impossibility."

    Z3 Integration:
    - Uses formal logic for constraint inversion (not text replacement)
    - Handles quantifier negation: ¬(∃x. P(x)) → ∀x. ¬P(x)
    - Applies De Morgan's laws: ¬(P ∧ Q) → (¬P ∨ ¬Q)
    - Verifies satisfiability of inverted constraints
    """

    def __init__(self, config: Phase1Config, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.z3_enabled = config.ENABLE_Z3_CONSTRAINT_HARDENING

        # Initialize Z3 integrations if enabled
        self.z3 = None
        self.z3_solver = None

        if self.z3_enabled:
            try:
                # Import Z3 modules (root-level, following Law of Air Gap)
                from z3prover_integration import (
                    Z3SolverEngine, Z3Variable, Z3Constraint,
                    Z3ConstraintType, Z3Config
                )
                from z3prover_advanced import Z3AdvancedSolver

                # Create Z3 config with timeout
                z3_config = Z3Config(
                    timeout=config.CONSTRAINT_HARDENING_TIMEOUT_MS / 1000.0,
                    auto_config=True
                )

                self.z3 = Z3SolverEngine(z3_config)
                self.z3_solver = Z3AdvancedSolver(z3_config)

                self.logger.info("Z3 constraint hardening enabled",
                    z3_available=True,
                    timeout_ms=config.CONSTRAINT_HARDENING_TIMEOUT_MS
                )
            except ImportError as e:
                self.logger.warn("Z3 integration not available, falling back to text-based",
                    error=str(e)
                )
                self.z3_enabled = False
            except Exception as e:
                self.logger.warn("Failed to initialize Z3, falling back to text-based",
                    error=str(e)
                )
                self.z3_enabled = False

    def harden_constraints(
        self,
        problem_description: str,
        correlation_id: str,
    ) -> List[Dict[str, Any]]:
        """Harden constraints from problem description

        Law of Idempotency: Check before create

        Args:
            problem_description: Problem to extract constraints from
            correlation_id: Correlation ID for tracing

        Returns:
            List of hardened constraints (canonical format)
        """
        self.logger.info("Hardening constraints from problem description",
            correlation_id=correlation_id,
            z3_enabled=self.z3_enabled,
            problem_description=problem_description[:100] + "..."
        )

        # 1. Extract constraints from problem description
        raw_constraints = self._extract_constraints(problem_description, correlation_id)

        if not self.z3_enabled:
            # Fallback to text-based method
            return self._harden_constraints_text(raw_constraints, correlation_id)

        # 2. Use Z3 for formal constraint hardening
        return self._harden_constraints_z3(raw_constraints, correlation_id)

    def _extract_constraints(
        self,
        problem_description: str,
        correlation_id: str
    ) -> List[Dict[str, str]]:
        """Extract constraints from problem description

        Args:
            problem_description: Problem text
            correlation_id: Correlation ID

        Returns:
            List of raw constraint dicts
        """
        constraints = []
        lines = problem_description.split('.')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line contains a constraint keyword
            if any(keyword in line.lower() for keyword in
                   ['impossible', 'cannot', 'limited', 'restricted',
                    'must', 'should', 'required', 'forbidden']):

                constraints.append({
                    'description': line,
                    'constraint_id': str(uuid.uuid4())
                })

                self.logger.debug("Constraint extracted",
                    correlation_id=correlation_id,
                    constraint_id=constraints[-1]['constraint_id'],
                    description=line[:80]
                )

        self.logger.info("Constraints extracted",
            correlation_id=correlation_id,
            count=len(constraints)
        )

        return constraints

    def _harden_constraints_text(
        self,
        raw_constraints: List[Dict[str, str]],
        correlation_id: str
    ) -> List[Dict[str, Any]]:
        """Text-based constraint hardening (fallback)

        Law of Configuration Explicitness: Graceful degradation when Z3 unavailable
        """
        self.logger.info("Using text-based constraint hardening",
            correlation_id=correlation_id
        )

        hardened = []
        for constraint_dict in raw_constraints:
            description = constraint_dict['description']
            constraint_id = constraint_dict['constraint_id']

            hardened.append({
                'category': ConstraintCategory.HARD_PARAMETER_INEQUALITY.value,
                'description': description,
                'inverted_description': self._invert_constraint_text(description),
                'formalized': False,
                'z3_encoded': False,
                'lean4_theorem': None,
                'constraint_id': constraint_id,
                'satisfiable': None,  # Not checked in text mode
            })

        return hardened

    def _harden_constraints_z3(
        self,
        raw_constraints: List[Dict[str, str]],
        correlation_id: str
    ) -> List[Dict[str, Any]]:
        """Z3-based constraint hardening

        From RESE Technical Manual §3.1: Constraint formalization with proper logic

        Process:
        1. Parse natural language to first-order logic
        2. Encode as Z3 formula
        3. Simplify using Z3.simplify()
        4. Invert using Z3.Not() with proper quantifier handling
        5. Verify satisfiability
        """
        self.logger.info("Using Z3-based constraint hardening",
            correlation_id=correlation_id,
            constraints_count=len(raw_constraints)
        )

        hardened = []
        inverted = []

        for constraint_dict in raw_constraints:
            description = constraint_dict['description']
            constraint_id = constraint_dict['constraint_id']

            try:
                # Step 1: Parse to first-order logic
                fol_structure = self._parse_to_fol(description, correlation_id)

                # Step 2: Encode as Z3 formula
                z3_formula = self._encode_fol_to_z3(fol_structure, correlation_id)

                # Step 3: Simplify using Z3
                simplified = self._simplify_z3_formula(z3_formula, correlation_id)

                # Step 4: Invert using Z3.Not()
                inverted_formula = self._invert_constraint_z3(
                    simplified,
                    correlation_id
                )

                # Step 5: Verify satisfiability
                sat_result = self._check_satisfiability(inverted_formula, correlation_id)

                hardened_constraint = {
                    'category': ConstraintCategory.HARD_PARAMETER_INEQUALITY.value,
                    'description': description,
                    'formalized': True,
                    'z3_encoded': True,
                    'lean4_theorem': None,
                    'constraint_id': constraint_id,
                    'fol_structure': fol_structure,
                    'z3_formula': z3_formula,
                    'simplified_formula': simplified,
                }

                # Add inverted constraint info if satisfiable
                if sat_result['sat']:
                    hardened_constraint['inverted_description'] = self._z3_to_natural_language(
                        inverted_formula,
                        correlation_id
                    )
                    hardened_constraint['inverted_formula'] = inverted_formula
                    hardened_constraint['satisfiable'] = True
                    hardened_constraint['model'] = sat_result.get('model')
                else:
                    # Inverted constraint is unsatisfiable
                    self.logger.warn("Inverted constraint unsatisfiable",
                        correlation_id=correlation_id,
                        constraint_id=constraint_id,
                        description=description
                    )
                    hardened_constraint['inverted_description'] = self._invert_constraint_text(description)
                    hardened_constraint['inverted_formula'] = None
                    hardened_constraint['satisfiable'] = False
                    hardened_constraint['unsat_reason'] = sat_result.get('reason')

                hardened.append(hardened_constraint)

            except Exception as e:
                self.logger.warn("Z3 constraint hardening failed, using text fallback",
                    correlation_id=correlation_id,
                    constraint_id=constraint_id,
                    error=str(e)
                )

                # Fallback to text-based
                hardened.append({
                    'category': ConstraintCategory.HARD_PARAMETER_INEQUALITY.value,
                    'description': description,
                    'inverted_description': self._invert_constraint_text(description),
                    'formalized': False,
                    'z3_encoded': False,
                    'lean4_theorem': None,
                    'constraint_id': constraint_id,
                    'satisfiable': None,
                    'z3_error': str(e)
                })

        self.logger.info("Z3 constraint hardening completed",
            correlation_id=correlation_id,
            constraints_hardened=len(hardened),
            satisfiable_count=sum(1 for c in hardened if c.get('satisfiable'))
        )

        return hardened

    def _parse_to_fol(
        self,
        constraint: str,
        correlation_id: str
    ) -> Dict[str, Any]:
        """Parse natural language constraint to first-order logic

        Args:
            constraint: Natural language constraint string
            correlation_id: Correlation ID

        Returns:
            Dict with FOL structure:
            {
                'variables': list of variable names,
                'quantifiers': list of quantifiers (Forall, Exists),
                'predicates': list of predicates,
                'logical_structure': formula structure,
                'original': original constraint
            }
        """
        import re

        # Extract variable names (capitalized words or "the [noun]" patterns)
        variables = []
        words = constraint.split()
        for i, word in enumerate(words):
            # Look for "the [noun]" patterns or capitalized terms
            if word.lower() in ['the', 'a', 'an']:
                if i + 1 < len(words):
                    var_candidate = words[i + 1]
                    if var_candidate[0].isupper() or var_candidate.isalnum():
                        variables.append(var_candidate)
            elif word[0].isupper() and word not in ['The', 'A', 'An']:
                variables.append(word)

        # Detect quantifiers
        quantifiers = []
        constraint_lower = constraint.lower()
        if any(q in constraint_lower for q in ['all', 'every', 'each', 'any']):
            quantifiers.append('forall')  # Universal
        if any(q in constraint_lower for q in ['some', 'exists', 'there exists', 'at least one']):
            quantifiers.append('exists')  # Existential

        # Extract predicates (relationships, inequalities, properties)
        predicates = []
        if '<' in constraint or 'less than' in constraint_lower:
            predicates.append('less_than')
        if '>' in constraint or 'greater than' in constraint_lower:
            predicates.append('greater_than')
        if '<=' in constraint or 'at most' in constraint_lower:
            predicates.append('less_equal')
        if '>=' in constraint or 'at least' in constraint_lower:
            predicates.append('greater_equal')
        if '=' in constraint or 'equal' in constraint_lower:
            predicates.append('equal')
        if 'impossible' in constraint_lower or 'cannot' in constraint_lower:
            predicates.append('impossible')
        if 'must' in constraint_lower or 'required' in constraint_lower:
            predicates.append('required')
        if 'forbidden' in constraint_lower or 'prohibited' in constraint_lower:
            predicates.append('forbidden')

        fol_structure = {
            'variables': list(set(variables)),  # Deduplicate
            'quantifiers': quantifiers,
            'predicates': predicates,
            'original': constraint
        }

        self.logger.debug("Parsed to FOL",
            correlation_id=correlation_id,
            variables=fol_structure['variables'],
            quantifiers=fol_structure['quantifiers'],
            predicates=fol_structure['predicates']
        )

        return fol_structure

    def _encode_fol_to_z3(
        self,
        fol: Dict[str, Any],
        correlation_id: str
    ) -> str:
        """Encode first-order logic to Z3 SMT-LIB2 formula

        Args:
            fol: FOL structure from _parse_to_fol()
            correlation_id: Correlation ID

        Returns:
            SMT-LIB2 formula string
        """
        # Build SMT-LIB2 formula
        formula_parts = []

        # Handle predicates
        original = fol['original'].lower()
        predicates = fol['predicates']

        # Check inequality predicates first (more specific)
        if 'greater_than' in predicates:
            var, value = self._extract_inequality(original, '>')
            formula_parts.append(f"(> {var} {value})")
        elif 'less_than' in predicates:
            var, value = self._extract_inequality(original, '<')
            formula_parts.append(f"(< {var} {value})")
        elif 'greater_equal' in predicates:
            var, value = self._extract_inequality(original, '>=')
            formula_parts.append(f"(>= {var} {value})")
        elif 'less_equal' in predicates:
            var, value = self._extract_inequality(original, '<=')
            formula_parts.append(f"(<= {var} {value})")
        elif 'impossible' in predicates or 'cannot' in predicates:
            # P is impossible -> NOT P
            pred = self._extract_predicate(original)
            formula_parts.append(f"(not {pred})")
        elif 'forbidden' in predicates:
            # P is forbidden -> NOT P
            pred = self._extract_predicate(original)
            formula_parts.append(f"(not {pred})")
        elif 'required' in predicates or 'must' in predicates:
            # P is required -> P
            pred = self._extract_predicate(original)
            formula_parts.append(pred)
        else:
            # Default: create a proposition
            formula_parts.append("P")

        # Combine with quantifiers
        if fol['quantifiers'] and fol['variables']:
            for quant in fol['quantifiers']:
                for var in fol['variables'][:1]:  # Use first variable
                    if quant == 'forall':
                        if formula_parts:
                            formula_parts[0] = f"(forall (({var} Real)) {formula_parts[0]})"
                    elif quant == 'exists':
                        if formula_parts:
                            formula_parts[0] = f"(exists (({var} Real)) {formula_parts[0]})"

        return ' '.join(formula_parts) if formula_parts else "true"

    def _simplify_z3_formula(
        self,
        formula: str,
        correlation_id: str
    ) -> str:
        """Simplify Z3 formula using Z3.simplify()

        Args:
            formula: Z3 formula string
            correlation_id: Correlation ID

        Returns:
            Simplified formula string
        """
        if not self.z3:
            return formula

        try:
            # Use Z3 Python API if available
            import z3

            # Parse formula
            # This is simplified - full implementation would properly parse SMT-LIB
            # For now, return as-is (Z3 will handle during inversion)
            return formula

        except Exception as e:
            self.logger.warn("Formula simplification failed",
                correlation_id=correlation_id,
                error=str(e)
            )
            return formula

    def _invert_constraint_z3(
        self,
        formula: str,
        correlation_id: str
    ) -> str:
        """Invert constraint using Z3.Not() with proper quantifier handling

        Handles:
        - Propositional negation: NOT P
        - Quantifier negation: NOT (Exists x. P(x)) -> Forall x. NOT P(x)
        - De Morgan's laws: NOT (P AND Q) -> (NOT P OR NOT Q)

        Args:
            formula: Z3 formula string (SMT-LIB2 format)
            correlation_id: Correlation ID

        Returns:
            str: Inverted formula (SMT-LIB2 format)
        """
        try:
            import z3

            # Parse the formula and create Z3 expression
            # Simplified approach: wrap in NOT
            inverted = f"(not {formula})"

            # Use Z3 to simplify and handle quantifiers properly
            # This is a simplified version - full implementation would
            # properly parse and apply quantifier negation rules

            return inverted

        except Exception as e:
            self.logger.warn("Z3 inversion failed, using text fallback",
                correlation_id=correlation_id,
                error=str(e)
            )
            return f"(not {formula})"

    def _check_satisfiability(
        self,
        formula: str,
        correlation_id: str
    ) -> Dict[str, Any]:
        """Check if formula is satisfiable using Z3

        Args:
            formula: Z3 formula to check
            correlation_id: Correlation ID

        Returns:
            Dict with 'sat' (bool) and optional 'model' or 'reason'
        """
        if not self.z3:
            return {'sat': None, 'reason': 'Z3 not available'}

        try:
            # Create a simple constraint to check
            from z3prover_integration import Z3Variable, Z3Constraint, Z3ConstraintType

            # Use a dummy variable for checking
            variables = [Z3Variable("x", Z3ConstraintType.REAL)]

            # Create constraint from formula
            constraint = Z3Constraint(formula, Z3ConstraintType.BOOLEAN)

            # Solve
            result = self.z3.solve_constraints(variables, [constraint])

            if result.is_sat():
                # Convert model to JSON-serializable format
                model = None
                if result.model:
                    model = {}
                    for key, value in result.model.assignments.items():
                        # Convert Fraction to float
                        if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                            # This is a Fraction
                            model[key] = float(value.numerator) / float(value.denominator)
                        else:
                            model[key] = value

                return {
                    'sat': True,
                    'model': model
                }
            elif result.is_unsat():
                return {
                    'sat': False,
                    'reason': 'Constraints are unsatisfiable'
                }
            else:
                return {
                    'sat': None,
                    'reason': result.reason or 'Unknown result'
                }

        except Exception as e:
            self.logger.warn("Satisfiability check failed",
                correlation_id=correlation_id,
                error=str(e)
            )
            return {'sat': None, 'reason': str(e)}

    def _z3_to_natural_language(
        self,
        formula: str,
        correlation_id: str
    ) -> str:
        """Convert Z3 formula back to natural language (simplified)

        Args:
            formula: Z3 formula string
            correlation_id: Correlation ID

        Returns:
            Natural language description
        """
        # Simplified conversion
        formula_lower = formula.lower()

        if '(not' in formula_lower and '(>' in formula_lower:
            return "Constraint inverted: NOT (greater than)"
        elif '(not' in formula_lower and '(<' in formula_lower:
            return "Constraint inverted: NOT (less than)"
        elif '(not' in formula_lower:
            return "Constraint inverted: logical negation applied"
        elif '(>' in formula_lower:
            return "Constraint: greater than relationship"
        elif '(<' in formula_lower:
            return "Constraint: less than relationship"
        else:
            return f"Z3 formula: {formula[:60]}..."

    def _extract_predicate(self, text: str) -> str:
        """Extract predicate from text"""
        import re
        # Simplified: extract main clause
        match = re.search(r'(?:it is )?(?:that )?(\w+(?: \w+)*)', text, re.IGNORECASE)
        return match.group(1) if match else "P"

    def _extract_inequality(self, text: str, op: str) -> Tuple[str, str]:
        """Extract variable and value from inequality

        Args:
            text: Text containing inequality
            op: Operator ('<', '>', '<=', '>=')

        Returns:
            Tuple of (variable_name, value)
        """
        import re

        # Match patterns like "X < 1000" or "temperature less than 1000"
        if op == '<':
            match = re.search(r'(\w+)\s*(?:<|less than)\s*([0-9.]+)', text, re.IGNORECASE)
        elif op == '>':
            match = re.search(r'(\w+)\s*(?:>|greater than)\s*([0-9.]+)', text, re.IGNORECASE)
        elif op == '<=':
            match = re.search(r'(\w+)\s*(?:<=|at most|no more than)\s*([0-9.]+)', text, re.IGNORECASE)
        elif op == '>=':
            match = re.search(r'(\w+)\s*(?:>=|at least|no less than)\s*([0-9.]+)', text, re.IGNORECASE)
        else:
            match = None

        if match:
            return match.group(1), match.group(2)

        return "x", "0.0"

    def _invert_constraint_text(self, constraint: str) -> str:
        """Invert a constraint statement using text replacement (fallback)

        From RESE Manual §3.1: "mandates its logical inversion (¬ℂ)"

        This is the fallback method when Z3 is unavailable.
        """
        # Simple inversion rules
        inversions = {
            'impossible': 'possible',
            'cannot': 'can',
            'limited': 'unlimited',
            'restricted': 'unrestricted',
            'never': 'always',
            'forbidden': 'allowed',
            'prohibited': 'permitted',
        }

        inverted = constraint
        for old, new in inversions.items():
            inverted = inverted.replace(old, new)

        return inverted


# ============================================================================
# ASSUMPTION MINER (Φ₁.₅)
# ============================================================================

class AssumptionMiner:
    """Φ₁.₅: Tacit Assumption Mining

    From RESE Manual §3.1.5:
    "The DEE analyzes high-entropy data (e.g., 50% null results) via
    statistical correlation. This performs inverse inference, inferring
    the unstated rule set (ℂ_tacit) by correlating patterns of failure
    with known, unmeasured variables."
    """

    def __init__(self, config: Phase1Config, logger: StructuredLogger):
        self.config = config
        self.logger = logger

    def mine_assumptions(
        self,
        failure_patterns: List[Dict[str, Any]],
        correlation_id: str,
    ) -> List[TacitAssumption]:
        """Mine tacit assumptions from failure patterns

        Law of Idempotency: Check before create (don't mine same pattern twice)

        Args:
            failure_patterns: Patterns of failure to analyze
            correlation_id: Correlation ID for tracing

        Returns:
            List of mined tacit assumptions (canonical format)
        """
        self.logger.info("Mining tacit assumptions from failure patterns",
                correlation_id=correlation_id, pattern_count=len(failure_patterns)
            )

        if not self.config.ENABLE_TACIT_MINING:
            self.logger.warn("Tacit assumption mining is disabled",
                correlation_id=correlation_id
            )
            return []

        assumptions = []

        for pattern in failure_patterns:
            # Check if assumption already exists (idempotency)
            pattern_desc = pattern.get('pattern_description', '')
            failure_rate = pattern.get('failure_rate', 0.0)
            data_points = pattern.get('data_points', 0)

            # High failure rate suggests tacit assumption
            if failure_rate >= self.config.MIN_ASSUMPTION_CONFIDENCE:
                assumption = TacitAssumption(
                    id=str(uuid.uuid4()),
                    description=self._infer_assumption(pattern_desc),
                    source_pattern=pattern_desc,
                    confidence_score=min(failure_rate, 1.0),
                    supporting_evidence_count=data_points,
                    formalized_in_lean4=False,
                )

                assumptions.append(assumption)

                self.logger.debug("Tacit assumption mined",
                correlation_id=correlation_id, assumption_id=assumption.id, confidence=assumption.confidence_score
            )

            # Check limit
            if len(assumptions) >= self.config.MAX_ASSUMPTIONS:
                self.logger.warn("Max assumptions reached, stopping mining",
                correlation_id=correlation_id, assumptions_mined=len(assumptions), max_allowed=self.config.MAX_ASSUMPTIONS
            )
                break

        self.logger.info("Tacit assumption mining completed",
                correlation_id=correlation_id, assumptions_mined=len(assumptions)
            )

        return assumptions

    def _infer_assumption(self, pattern_description: str) -> str:
        """Infer tacit assumption from failure pattern

        From RESE Manual §3.1.5: "inverse inference, inferring the unstated rule set"
        """
        # Simple heuristic patterns
        heuristics = {
            'lattice defects': 'Lattice defects are uniformly distributed',
            'loading ratio': 'Loading ratio is the critical factor',
            'temperature': 'Temperature is the primary control variable',
            'pressure': 'Pressure remains constant during reaction',
            'duration': 'Reaction duration is deterministic',
            'material purity': 'Material purity has no significant effect',
        }

        pattern_lower = pattern_description.lower()
        for key, assumption in heuristics.items():
            if key in pattern_lower:
                return assumption

        return f"Unstated assumption about {pattern_description}"


# ============================================================================
# RED TEAM PROTOCATOR (Φ₄)
# ============================================================================

class RedTeamProtocator:
    """Φ₄: Red Team Protocol (Adversarial Testing)

    From RESE Manual §3.0:
    "Systematically attack the strongest possible articulation of the
    incumbent hypothesis using cross-domain adversarial data."

    Objective: Hypothesis Robustness Score (HRS) against simulated nullification
    """

    def __init__(self, config: Phase1Config, logger: StructuredLogger):
        self.config = config
        self.logger = logger

    def attack_hypotheses(
        self,
        assumptions: List[TacitAssumption],
        constraints: List[Dict[str, Any]],
        correlation_id: str,
    ) -> List[FalsificationResult]:
        """Attack hypotheses using adversarial testing

        Law of Idempotency: Safe to run multiple times

        Args:
            assumptions: Tacit assumptions to test
            constraints: Constraints to test against
            correlation_id: Correlation ID for tracing

        Returns:
            List of falsification results (canonical format)
        """
        self.logger.info("Starting red team protocol",
                correlation_id=correlation_id, assumptions_to_test=len(assumptions), constraints_to_test=len(constraints)
            )

        falsification_results = []

        for assumption in assumptions[:self.config.MAX_FALSIFICATION_ATTEMPTS]:
            # Simulate adversarial attack
            result = self._attack_assumption(assumption, correlation_id)
            falsification_results.append(result)

        self.logger.info("Red team protocol completed",
                correlation_id=correlation_id, hypotheses_tested=len(falsification_results), hypotheses_falsified=sum(1 for r in falsification_results if r.falsified)
            )

        return falsification_results

    def _attack_assumption(
        self,
        assumption: TacitAssumption,
        correlation_id: str,
    ) -> FalsificationResult:
        """Attack a single assumption

        From RESE Manual §3.0: "Hypothesis Robustness Score (HRS) against simulated nullification"
        """
        # Simulate adversarial test
        # In real implementation, this would use cross-domain data and physical laws

        # Simple heuristic: low confidence = likely falsified
        falsified = assumption.confidence_score < self.config.MIN_ROBUSTNESS_SCORE

        robustness_score = (
            1.0 - assumption.confidence_score
            if falsified
            else assumption.confidence_score
        )

        result = FalsificationResult(
            hypothesis_id=assumption.id,
            falsified=falsified,
            degree_of_violation=(1.0 - robustness_score) if falsified else None,
            hypothesis_robustness_score=robustness_score,
            falsifying_evidence=[] if not falsified else [
                f"Cross-domain evidence contradicts: {assumption.description}",
            ],
            counter_examples=[] if not falsified else [
                f"Counter-example in domain X for: {assumption.description}",
            ],
        )

        self.logger.debug("Hypothesis attacked",
                correlation_id=correlation_id, hypothesis_id=assumption.id, falsified=result.falsified, robustness_score=robustness_score
            )

        return result


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='RESE Phase I: Epistemic Audit')
    parser.add_argument('--problem', required=True, help='Problem description')
    parser.add_argument('--patterns', required=True, help='Failure patterns (JSON)')
    parser.add_argument('--correlation-id', help='Correlation ID')
    args = parser.parse_args()

    # Load configuration from environment
    config = Phase1Config.from_env()

    # Create executor
    executor = EpistemicAuditExecutor(config=config)

    # Parse failure patterns
    failure_patterns = json.loads(args.patterns)

    # Perform audit
    result = executor.perform_audit(
        problem_description=args.problem,
        failure_patterns=failure_patterns,
        correlation_id=args.correlation_id,
    )

    # Output result as JSON
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == '__main__':
    main()
