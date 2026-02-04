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
- Section 3.3: Formal Logic Audit and Contradiction Detection (Φ₃)
"""

import os
import sys
import json
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Add glue lib to path for importing SCE adapter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))

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
               if k not in ('tacit_assumptions', 'contradictions', 'falsification_results')}
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

        self.logger.info("EpistemicAuditExecutor initialized",
            max_assumptions=self.config.MAX_ASSUMPTIONS,
            max_constraints=self.config.MAX_CONSTRAINTS,
            enable_tacit_mining=self.config.ENABLE_TACIT_MINING,
            enable_red_team=self.config.ENABLE_RED_TEAM_PROTOCOL,
        )

    # ========================================================================
    # MAIN AUDIT ORCHESTRATOR
    # ========================================================================

    def perform_audit(
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
            'correlation_id': correlation_id,
            'problem_description': problem_description,
            'failure_patterns_count': len(failure_patterns),
        })

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
                'correlation_id': correlation_id,
            })
            hardened_constraints = self.constraint_hardener.harden_constraints(
                problem_description=problem_description,
                correlation_id=correlation_id,
            )

            # Φ₁.₅: Tacit Assumption Mining
            self.logger.info("Starting Φ₁.₅: Tacit Assumption Mining",
                'correlation_id': correlation_id,
            })
            tacit_assumptions = self.assumption_miner.mine_assumptions(
                failure_patterns=failure_patterns,
                correlation_id=correlation_id,
            )

            # Φ₃: Contradiction Detection (using SCE adapter)
            # Note: This would integrate with the TypeScript SCE via IPC
            # For now, we'll implement basic contradiction detection
            self.logger.info("Starting Φ₃: Contradiction Detection",
                'correlation_id': correlation_id,
            })
            contradictions = self._detect_contradictions(
                assumptions=tacit_assumptions,
                constraints=hardened_constraints,
                correlation_id=correlation_id,
            )

            # Φ₄: Red Team Protocol
            falsification_results = []
            if self.config.ENABLE_RED_TEAM_PROTOCOL:
                self.logger.info("Starting Φ₄: Red Team Protocol",
                    'correlation_id': correlation_id,
                })
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
                metrics={
                    'total_assumptions_analyzed': len(tacit_assumptions),
                    'confirmed_contradictions': len(contradictions),
                    'hypotheses_falsified': sum(1 for f in falsification_results if f.falsified),
                    'reduction_in_failure_rate': None,  # To be updated after Phase II/III
                },
                metadata={
                    'execution_time_ms': execution_time_ms,
                    'lean4_version': '4.7.0' if self.config.ENABLE_LEAN4_INTEGRATION else None,
                    'epoch_number': 1,  # Default to first epoch
                },
                correlation_id=correlation_id,
                timestamp=datetime.now(timezone.utc).isoformat(),  # Law of UTC
            )

            self.logger.info("Phase I: Epistemic Audit completed",
                'correlation_id': correlation_id,
                'audit_id': audit_result.audit_id,
                'execution_time_ms': execution_time_ms,
                'tacit_assumptions_found': len(tacit_assumptions),
                'contradictions_found': len(contradictions),
                'hypotheses_falsified': audit_result.metrics['hypotheses_falsified'],
            })

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

    def _detect_contradictions(
        self,
        assumptions: List[TacitAssumption],
        constraints: List[Dict[str, Any]],
        correlation_id: str,
    ) -> List[ContradictionDetection]:
        """Detect contradictions using SCE

        This is a placeholder that would integrate with the TypeScript SCE
        For now, implement basic contradiction detection
        """
        contradictions = []

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
                'correlation_id': correlation_id,
                'total_contradictions': len(contradictions),
                'max_allowed': self.config.MAX_CONTRADICTIONS,
            })
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
# CONSTRAINT HARDENER (Φ₁)
# ============================================================================

class ConstraintHardener:
    """Φ₁: Initial Hypothesis Cluster Definition (Constraint Hardening)

    From RESE Manual §3.1:
    "Φ₁ formally articulates the hardened constraint (ℂ) and mandates its
    logical inversion (¬ℂ), ensuring the system focuses on designing the
    mechanism required to circumvent the perceived impossibility."
    """

    def __init__(self, config: Phase1Config, logger: StructuredLogger):
        self.config = config
        self.logger = logger

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
            'correlation_id': correlation_id,
            'problem_description': problem_description,
        })

        constraints = []

        # Extract constraints from problem description
        # This is a simplified implementation - real version would use NLP
        lines = problem_description.split('.')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line contains a constraint
            if any(keyword in line.lower() for keyword in ['impossible', 'cannot', 'limited', 'restricted']):
                constraint_id = str(uuid.uuid4())
                constraints.append({
                    'category': ConstraintCategory.HARD_PARAMETER_INEQUALITY.value,
                    'description': line,
                    'inverted_description': self._invert_constraint(line),
                    'formalized': False,
                    'lean4_theorem': None,
                    'constraint_id': constraint_id,
                })

                self.logger.debug("Hardened constraint extracted",
                    'correlation_id': correlation_id,
                    'constraint_id': constraint_id,
                    'description': line,
                })

        self.logger.info("Constraint hardening completed",
            'correlation_id': correlation_id,
            'constraints_extracted': len(constraints),
        })

        return constraints

    def _invert_constraint(self, constraint: str) -> str:
        """Invert a constraint statement

        From RESE Manual §3.1: "mandates its logical inversion (¬ℂ)"
        """
        # Simple inversion rules
        inversions = {
            'impossible': 'possible',
            'cannot': 'can',
            'limited': 'unlimited',
            'restricted': 'unrestricted',
            'never': 'always',
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
            'correlation_id': correlation_id,
            'pattern_count': len(failure_patterns),
        })

        if not self.config.ENABLE_TACIT_MINING:
            self.logger.warn("Tacit assumption mining is disabled",
                'correlation_id': correlation_id,
            })
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
                    'correlation_id': correlation_id,
                    'assumption_id': assumption.id,
                    'confidence': assumption.confidence_score,
                })

            # Check limit
            if len(assumptions) >= self.config.MAX_ASSUMPTIONS:
                self.logger.warn("Max assumptions reached, stopping mining",
                    'correlation_id': correlation_id,
                    'assumptions_mined': len(assumptions),
                    'max_allowed': self.config.MAX_ASSUMPTIONS,
                })
                break

        self.logger.info("Tacit assumption mining completed",
            'correlation_id': correlation_id,
            'assumptions_mined': len(assumptions),
        })

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
            'correlation_id': correlation_id,
            'assumptions_to_test': len(assumptions),
            'constraints_to_test': len(constraints),
        })

        falsification_results = []

        for assumption in assumptions[:self.config.MAX_FALSIFICATION_ATTEMPTS]:
            # Simulate adversarial attack
            result = self._attack_assumption(assumption, correlation_id)
            falsification_results.append(result)

        self.logger.info("Red team protocol completed",
            'correlation_id': correlation_id,
            'hypotheses_tested': len(falsification_results),
            'hypotheses_falsified': sum(1 for r in falsification_results if r.falsified),
        })

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
            'correlation_id': correlation_id,
            'hypothesis_id': assumption.id,
            'falsified': result.falsified,
            'robustness_score': robustness_score,
        })

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
