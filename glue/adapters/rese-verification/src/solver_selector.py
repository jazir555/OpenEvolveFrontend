"""
Adaptive Solver Selector

Selects the appropriate solver tier based on:
1. Problem classification (class, domain, complexity)
2. Historical performance data
3. Current system state (circuit breakers, load)
4. User preferences and constraints

Following CLAUDE.md principles:
- Law of Configuration Explicitness: Selection criteria via env vars
- Circuit Breaker: Skip unhealthy solvers
- Structured Logging: JSON with correlation_id
- Performance Monitoring: Track solver effectiveness

Author: RESE Team
Created: 2026-02-04
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

try:
    from .verification_result import (
        VerificationTier,
        ProblemClass,
        ProblemDomain,
    )
    from .problem_classifier import ClassifierConfig, ProblemClassifier
except ImportError:
    from verification_result import (
        VerificationTier,
        ProblemClass,
        ProblemDomain,
    )
    from problem_classifier import ClassifierConfig, ProblemClassifier


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SolverSelectorConfig:
    """Solver selector configuration"""
    # Solver preferences
    prefer_fast: bool = True  # Prefer faster solvers (Tier 1) when possible
    allow_parallel: bool = False  # Allow running multiple solvers in parallel
    max_parallel_solvers: int = 2  # Maximum parallel solvers

    # Performance thresholds
    min_confidence_threshold: float = 0.7  # Minimum confidence to accept result
    max_total_time_ms: float = 300000  # Maximum total verification time (5 minutes)

    # Circuit breaker weights
    z3_failure_threshold: int = 5
    leanaide_failure_threshold: int = 5
    lean4_failure_threshold: int = 3

    # Solver availability
    z3_available: bool = True
    leanaide_available: bool = True
    lean4_available: bool = True

    @classmethod
    def from_env(cls) -> 'SolverSelectorConfig':
        """Load configuration from environment variables"""
        return cls(
            prefer_fast=os.getenv("PREFER_FAST_SOLVER", "true").lower() == "true",
            allow_parallel=os.getenv("ALLOW_PARALLEL_SOLVERS", "false").lower() == "true",
            max_parallel_solvers=int(os.getenv("MAX_PARALLEL_SOLVERS", "2")),
            min_confidence_threshold=float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.7")),
            max_total_time_ms=float(os.getenv("MAX_TOTAL_TIME_MS", "300000")),
            z3_failure_threshold=int(os.getenv("Z3_FAILURE_THRESHOLD", "5")),
            leanaide_failure_threshold=int(os.getenv("LEANAIDE_FAILURE_THRESHOLD", "5")),
            lean4_failure_threshold=int(os.getenv("LEAN4_FAILURE_THRESHOLD", "3")),
            z3_available=os.getenv("Z3_AVAILABLE", "true").lower() == "true",
            leanaide_available=os.getenv("LEANAIDE_AVAILABLE", "true").lower() == "true",
            lean4_available=os.getenv("LEAN4_AVAILABLE", "true").lower() == "true",
        )


# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================

@dataclass
class SolverPerformance:
    """Performance metrics for a solver"""
    tier: VerificationTier
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    timeout_attempts: int = 0
    total_time_ms: float = 0.0
    average_time_ms: float = 0.0
    success_rate: float = 0.0
    last_failure_time: Optional[datetime] = None
    circuit_breaker_open: bool = False
    failure_count: int = 0

    def record_attempt(self, success: bool, timeout: bool, execution_time_ms: float):
        """Record a solver attempt"""
        self.total_attempts += 1
        self.total_time_ms += execution_time_ms

        if success:
            self.successful_attempts += 1
            self.failure_count = 0  # Reset failure count on success
            self.circuit_breaker_open = False
        else:
            self.failed_attempts += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)

            if timeout:
                self.timeout_attempts += 1

        # Update averages
        self.average_time_ms = self.total_time_ms / self.total_attempts
        self.success_rate = self.successful_attempts / self.total_attempts if self.total_attempts > 0 else 0.0

    def should_attempt(self, threshold: int) -> bool:
        """Check if solver should be attempted (circuit breaker)"""
        if not self.circuit_breaker_open:
            return True

        # Check if enough time has passed since last failure
        if self.last_failure_time:
            elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() * 1000
            if elapsed > 60000:  # 1 minute cooldown
                self.circuit_breaker_open = False
                self.failure_count = 0
                return True

        return False

    def check_circuit_breaker(self, threshold: int):
        """Check if circuit breaker should be opened"""
        if self.failure_count >= threshold:
            self.circuit_breaker_open = True


# =============================================================================
# SOLVER SELECTION STRATEGY
# =============================================================================

class SelectionStrategy(Enum):
    """Solver selection strategies"""
    FAST_FIRST = "fast_first"  # Try Tier 1, escalate if needed
    ACCURATE_FIRST = "accurate_first"  # Try highest tier first
    PARALLEL = "parallel"  # Run multiple tiers in parallel
    ADAPTIVE = "adaptive"  # Choose based on classification
    USER_SPECIFIED = "user_specified"  # User specifies tier


@dataclass
class SelectionResult:
    """Result of solver selection"""
    recommended_tier: VerificationTier
    alternative_tiers: List[VerificationTier] = field(default_factory=list)
    strategy: SelectionStrategy = SelectionStrategy.ADAPTIVE
    confidence: float = 0.0
    reasoning: str = ""
    should_escalate_automatically: bool = True
    max_tier: Optional[VerificationTier] = None  # Maximum tier to try


# =============================================================================
# SOLVER SELECTOR
# =============================================================================

class SolverSelector:
    """
    Adaptive solver selector for tiered verification.

    Selects the appropriate solver tier based on:
    1. Problem classification and complexity
    2. Historical performance data
    3. Current system state (circuit breakers, availability)
    4. User preferences and constraints
    """

    def __init__(
        self,
        config: Optional[SolverSelectorConfig] = None,
        classifier_config: Optional[ClassifierConfig] = None
    ):
        """
        Initialize solver selector.

        Args:
            config: Selector configuration
            classifier_config: Problem classifier configuration
        """
        self.config = config or SolverSelectorConfig.from_env()
        self.classifier = ProblemClassifier(classifier_config)
        self.logger = logging.getLogger("rese.verification.selector")

        # Performance tracking
        self.performance = {
            VerificationTier.TIER1_Z3: SolverPerformance(VerificationTier.TIER1_Z3),
            VerificationTier.TIER2_LEANAIDE: SolverPerformance(VerificationTier.TIER2_LEANAIDE),
            VerificationTier.TIER3_LEAN4: SolverPerformance(VerificationTier.TIER3_LEAN4),
        }

    def select_solver(
        self,
        problem: str,
        constraints: Optional[List[Any]] = None,
        variables: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        strategy: Optional[SelectionStrategy] = None,
        max_tier: Optional[VerificationTier] = None
    ) -> SelectionResult:
        """
        Select the appropriate solver tier.

        Args:
            problem: Problem statement
            constraints: Optional list of constraints
            variables: Optional list of variables
            metadata: Optional metadata
            strategy: Selection strategy (defaults to config)
            max_tier: Maximum tier to try (for user constraints)

        Returns:
            SelectionResult with recommended tier and alternatives
        """
        strategy = strategy or self._determine_strategy(metadata)
        max_tier = max_tier or self._determine_max_tier()

        # Classify problem
        problem_class, problem_domain, complexity = self.classifier.classify(
            problem,
            constraints,
            variables,
            metadata
        )

        # Select based on strategy
        if strategy == SelectionStrategy.FAST_FIRST:
            return self._select_fast_first(complexity, max_tier)
        elif strategy == SelectionStrategy.ACCURATE_FIRST:
            return self._select_accurate_first(complexity, max_tier)
        elif strategy == SelectionStrategy.PARALLEL:
            return self._select_parallel(complexity, max_tier)
        elif strategy == SelectionStrategy.USER_SPECIFIED:
            return self._select_user_specified(metadata, max_tier)
        else:  # ADAPTIVE
            return self._select_adaptive(
                problem_class,
                problem_domain,
                complexity,
                max_tier
            )

    def _determine_strategy(self, metadata: Optional[Dict[str, Any]]) -> SelectionStrategy:
        """Determine selection strategy from metadata or config"""
        if metadata and "selection_strategy" in metadata:
            strategy_str = metadata["selection_strategy"]
            try:
                return SelectionStrategy(strategy_str)
            except ValueError:
                pass

        # Default to fast_first if configured, else adaptive
        return SelectionStrategy.FAST_FIRST if self.config.prefer_fast else SelectionStrategy.ADAPTIVE

    def _determine_max_tier(self) -> VerificationTier:
        """Determine maximum available tier"""
        if self.config.lean4_available:
            return VerificationTier.TIER3_LEAN4
        elif self.config.leanaide_available:
            return VerificationTier.TIER2_LEANAIDE
        elif self.config.z3_available:
            return VerificationTier.TIER1_Z3
        else:
            # Fallback: assume all available
            return VerificationTier.TIER3_LEAN4

    def _select_fast_first(self, complexity: Dict[str, Any], max_tier: VerificationTier) -> SelectionResult:
        """Select solver using fast-first strategy"""
        estimated_tier = complexity.get("estimated_tier", 1)

        # Start with Tier 1 if available and not too complex
        if self.config.z3_available and estimated_tier == 1:
            recommended = VerificationTier.TIER1_Z3
            reasoning = "Problem is simple enough for Tier 1 (Z3)"
            alternatives = self._get_escalation_path(recommended, max_tier)
        elif self.config.leanaide_available and estimated_tier == 2:
            recommended = VerificationTier.TIER2_LEANAIDE
            reasoning = "Problem complexity suggests Tier 2 (LeanAide)"
            alternatives = self._get_escalation_path(recommended, max_tier)
        else:
            recommended = VerificationTier.TIER3_LEAN4
            reasoning = "Problem is complex, starting with Tier 3 (Lean 4)"
            alternatives = []

        return SelectionResult(
            recommended_tier=recommended,
            alternative_tiers=alternatives,
            strategy=SelectionStrategy.FAST_FIRST,
            confidence=0.8,
            reasoning=reasoning,
            should_escalate_automatically=True,
            max_tier=max_tier,
        )

    def _select_accurate_first(self, complexity: Dict[str, Any], max_tier: VerificationTier) -> SelectionResult:
        """Select solver using accurate-first strategy"""
        # Always start with highest available tier
        if self.config.lean4_available and max_tier == VerificationTier.TIER3_LEAN4:
            recommended = VerificationTier.TIER3_LEAN4
            reasoning = "Starting with most accurate solver (Tier 3: Lean 4)"
        elif self.config.leanaide_available and max_tier in [VerificationTier.TIER2_LEANAIDE, VerificationTier.TIER3_LEAN4]:
            recommended = VerificationTier.TIER2_LEANAIDE
            reasoning = "Starting with Tier 2 (LeanAide) for accuracy"
        else:
            recommended = VerificationTier.TIER1_Z3
            reasoning = "Using Tier 1 (Z3) as only available option"

        return SelectionResult(
            recommended_tier=recommended,
            alternative_tiers=[],
            strategy=SelectionStrategy.ACCURATE_FIRST,
            confidence=0.9,
            reasoning=reasoning,
            should_escalate_automatically=False,  # No escalation if starting with best
            max_tier=max_tier,
        )

    def _select_parallel(self, complexity: Dict[str, Any], max_tier: VerificationTier) -> SelectionResult:
        """Select solver using parallel strategy"""
        available = []

        if self.config.z3_available:
            available.append(VerificationTier.TIER1_Z3)
        if self.config.leanaide_available:
            available.append(VerificationTier.TIER2_LEANAIDE)
        if self.config.lean4_available and max_tier == VerificationTier.TIER3_LEAN4:
            available.append(VerificationTier.TIER3_LEAN4)

        # Run all available in parallel (up to max_parallel_solvers)
        tiers_to_run = available[:self.config.max_parallel_solvers]

        return SelectionResult(
            recommended_tier=tiers_to_run[0] if tiers_to_run else VerificationTier.TIER1_Z3,
            alternative_tiers=tiers_to_run[1:],
            strategy=SelectionStrategy.PARALLEL,
            confidence=0.95,  # High confidence with parallel verification
            reasoning=f"Running {len(tiers_to_run)} solvers in parallel",
            should_escalate_automatically=False,
            max_tier=max_tier,
        )

    def _select_user_specified(self, metadata: Optional[Dict[str, Any]], max_tier: VerificationTier) -> SelectionResult:
        """Select solver based on user specification"""
        if metadata and "preferred_tier" in metadata:
            tier_str = metadata["preferred_tier"]
            try:
                tier = VerificationTier(tier_str)
                return SelectionResult(
                    recommended_tier=tier,
                    alternative_tiers=self._get_escalation_path(tier, max_tier),
                    strategy=SelectionStrategy.USER_SPECIFIED,
                    confidence=1.0,  # User knows best
                    reasoning="User-specified tier",
                    should_escalate_automatically=metadata.get("auto_escalate", True),
                    max_tier=max_tier,
                )
            except ValueError:
                pass

        # Fallback to adaptive
        return SelectionResult(
            recommended_tier=VerificationTier.TIER1_Z3,
            alternative_tiers=[VerificationTier.TIER2_LEANAIDE, VerificationTier.TIER3_LEAN4],
            strategy=SelectionStrategy.USER_SPECIFIED,
            confidence=0.5,
            reasoning="User specified invalid tier, defaulting to Tier 1",
            should_escalate_automatically=True,
            max_tier=max_tier,
        )

    def _select_adaptive(
        self,
        problem_class: ProblemClass,
        problem_domain: ProblemDomain,
        complexity: Dict[str, Any],
        max_tier: VerificationTier
    ) -> SelectionResult:
        """Select solver using adaptive strategy"""
        estimated_tier = complexity.get("estimated_tier", 1)

        # Check circuit breakers
        if estimated_tier == 1 and self._is_circuit_breaker_open(VerificationTier.TIER1_Z3):
            estimated_tier = 2  # Escalate due to circuit breaker
        elif estimated_tier == 2 and self._is_circuit_breaker_open(VerificationTier.TIER2_LEANAIDE):
            estimated_tier = 3  # Escalate due to circuit breaker

        # Select based on estimated tier
        if estimated_tier == 1 and self.config.z3_available:
            recommended = VerificationTier.TIER1_Z3
            reasoning = "Adaptive: Problem suitable for Tier 1 (Z3)"
        elif estimated_tier == 2 and self.config.leanaide_available:
            recommended = VerificationTier.TIER2_LEANAIDE
            reasoning = "Adaptive: Problem complexity suggests Tier 2 (LeanAide)"
        elif self.config.lean4_available:
            recommended = VerificationTier.TIER3_LEAN4
            reasoning = "Adaptive: Problem requires Tier 3 (Lean 4)"
        else:
            # Fallback
            recommended = VerificationTier.TIER1_Z3
            reasoning = "Adaptive: Using Tier 1 as fallback"

        alternatives = self._get_escalation_path(recommended, max_tier)

        return SelectionResult(
            recommended_tier=recommended,
            alternative_tiers=alternatives,
            strategy=SelectionStrategy.ADAPTIVE,
            confidence=0.85,
            reasoning=reasoning,
            should_escalate_automatically=True,
            max_tier=max_tier,
        )

    def _get_escalation_path(self, start_tier: VerificationTier, max_tier: VerificationTier) -> List[VerificationTier]:
        """Get escalation path from start tier to max tier"""
        all_tiers = [
            VerificationTier.TIER1_Z3,
            VerificationTier.TIER2_LEANAIDE,
            VerificationTier.TIER3_LEAN4,
        ]

        start_idx = all_tiers.index(start_tier)
        max_idx = all_tiers.index(max_tier)

        # Return tiers after start_tier up to max_tier
        return all_tiers[start_idx + 1:max_idx + 1]

    def _is_circuit_breaker_open(self, tier: VerificationTier) -> bool:
        """Check if circuit breaker is open for a tier"""
        performance = self.performance.get(tier)
        if not performance:
            return False

        # Update circuit breaker state
        if tier == VerificationTier.TIER1_Z3:
            performance.check_circuit_breaker(self.config.z3_failure_threshold)
        elif tier == VerificationTier.TIER2_LEANAIDE:
            performance.check_circuit_breaker(self.config.leanaide_failure_threshold)
        elif tier == VerificationTier.TIER3_LEAN4:
            performance.check_circuit_breaker(self.config.lean4_failure_threshold)

        return performance.circuit_breaker_open

    def record_performance(
        self,
        tier: VerificationTier,
        success: bool,
        timeout: bool,
        execution_time_ms: float
    ):
        """Record solver performance"""
        performance = self.performance.get(tier)
        if performance:
            performance.record_attempt(success, timeout, execution_time_ms)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all solvers"""
        return {
            tier.value: {
                "total_attempts": p.total_attempts,
                "successful_attempts": p.successful_attempts,
                "failed_attempts": p.failed_attempts,
                "timeout_attempts": p.timeout_attempts,
                "success_rate": p.success_rate,
                "average_time_ms": p.average_time_ms,
                "circuit_breaker_open": p.circuit_breaker_open,
                "failure_count": p.failure_count,
            }
            for tier, p in self.performance.items()
        }

    def reset_performance_stats(self):
        """Reset all performance statistics"""
        for performance in self.performance.values():
            performance.total_attempts = 0
            performance.successful_attempts = 0
            performance.failed_attempts = 0
            performance.timeout_attempts = 0
            performance.total_time_ms = 0.0
            performance.average_time_ms = 0.0
            performance.success_rate = 0.0
            performance.last_failure_time = None
            performance.circuit_breaker_open = False
            performance.failure_count = 0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def select_solver(
    problem: str,
    constraints: Optional[List[Any]] = None,
    strategy: Optional[SelectionStrategy] = None,
    config: Optional[SolverSelectorConfig] = None
) -> SelectionResult:
    """
    Convenience function to select solver.

    Args:
        problem: Problem statement
        constraints: Optional list of constraints
        strategy: Selection strategy
        config: Selector configuration

    Returns:
        SelectionResult
    """
    selector = SolverSelector(config)
    return selector.select_solver(problem, constraints, strategy=strategy)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SolverSelectorConfig",
    "SolverPerformance",
    "SelectionStrategy",
    "SelectionResult",
    "SolverSelector",
    "select_solver",
]
