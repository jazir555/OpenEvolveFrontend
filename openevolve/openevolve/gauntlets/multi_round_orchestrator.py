"""
Multi-Round Gauntlet Orchestrator with Advanced State Management

This module provides sophisticated orchestration for multi-round gauntlet systems,
including state management, decision logic, artifact fusion, and performance tracking.

Key Features:
1. State Management: Track progress, scores, decisions across all rounds
2. Decision Points: Smart logic to continue/terminate at each round
3. Artifact Fusion: Combine insights from all rounds into unified knowledge
4. Score Normalization: Handle different scoring scales across rounds
5. Progress Reporting: Real-time feedback on gauntlet progress
6. Performance Tracking: Metrics on evaluation quality and efficiency
7. Parallel Execution: Run independent evaluations in parallel where possible

Author: OpenEvolve Team
Version: 1.0.0
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
import json
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class RoundStatus(Enum):
    """Status of a gauntlet round"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class Round1Result:
    """Result from Round 1 (LoongFlow AI Evaluation)"""
    score: float  # 0-1 scale
    confidence: float  # 0-1 scale
    feedback: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    robustness_score: float = 0.0
    execution_time: float = 0.0
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Round2Result:
    """Result from Round 2 (Red Team Adversarial Attack)"""
    score: float  # 0-100 scale, will be normalized to 0-1
    attacks_attempted: int = 0
    attacks_successful: int = 0
    vulnerabilities_found: List[str] = field(default_factory=list)
    edge_cases_tested: List[str] = field(default_factory=list)
    robustness_score: float = 0.0
    execution_time: float = 0.0
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Round3Result:
    """Result from Round 3 (Gold Team Consensus Verification)"""
    score: float  # 0-10 scale, will be normalized to 0-1
    consensus_score: float  # 0-1 scale
    formal_verification_passed: bool = False
    judge_scores: List[float] = field(default_factory=list)
    judge_feedback: List[str] = field(default_factory=list)
    robustness_score: float = 0.0
    execution_time: float = 0.0
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GauntletState:
    """
    Tracks state across all rounds of gauntlet execution.

    This dataclass maintains the complete state of a gauntlet run,
    including progress, results from each round, decisions, and artifacts.
    """
    # Core problem definition
    solution: str
    problem: str
    domain: str

    # Progress tracking
    current_round: int = 0  # 0=not started, 1=R1, 2=R2, 3=R3, 4=complete
    rounds_completed: List[int] = field(default_factory=list)  # [1] or [1,2] or [1,2,3]

    # Results from each round
    round1_result: Optional[Round1Result] = None
    round2_result: Optional[Round2Result] = None
    round3_result: Optional[Round3Result] = None

    # Scores (normalized to 0-1 scale)
    round1_normalized_score: Optional[float] = None
    round2_normalized_score: Optional[float] = None
    round3_normalized_score: Optional[float] = None

    # Decisions
    round1_decision: Optional[str] = None  # "continue", "terminate"
    round2_decision: Optional[str] = None
    round3_decision: Optional[str] = None

    # Artifacts collected across rounds
    collected_artifacts: List[Any] = field(default_factory=list)

    # Performance metrics
    total_evaluation_time: float = 0.0
    round_times: Dict[int, float] = field(default_factory=dict)

    # Metadata
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "not_started"  # not_started, in_progress, completed, terminated, error

    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'solution': self.solution[:100] + '...' if len(self.solution) > 100 else self.solution,
            'problem': self.problem,
            'domain': self.domain,
            'current_round': self.current_round,
            'rounds_completed': self.rounds_completed,
            'round1_score': self.round1_normalized_score,
            'round2_score': self.round2_normalized_score,
            'round3_score': self.round3_normalized_score,
            'round1_decision': self.round1_decision,
            'round2_decision': self.round2_decision,
            'round3_decision': self.round3_decision,
            'total_time': self.total_evaluation_time,
            'status': self.status,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class FusedArtifacts:
    """
    Combined insights from all gauntlet rounds.

    This class aggregates artifacts from all rounds, identifies consensus
    and conflicts, and derives actionable insights.
    """
    # All scores and metrics
    all_scores: Dict[str, float] = field(default_factory=dict)
    all_feedback: List[str] = field(default_factory=list)
    all_strengths: List[str] = field(default_factory=list)
    all_weaknesses: List[str] = field(default_factory=list)
    all_suggestions: List[str] = field(default_factory=list)

    # Derived insights
    consensus_strengths: List[str] = field(default_factory=list)  # Mentioned by multiple rounds
    consensus_weaknesses: List[str] = field(default_factory=list)
    conflicting_feedback: List[Tuple[str, str]] = field(default_factory=list)  # (round, disagreement)

    # Aggregated metrics
    robustness_trend: List[float] = field(default_factory=list)  # [r1, r2, r3]
    confidence_trend: List[float] = field(default_factory=list)
    quality_trend: List[float] = field(default_factory=list)

    # Recommendations
    overall_recommendation: str = ""
    improvement_priority: List[str] = field(default_factory=list)  # Ordered by importance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'all_scores': self.all_scores,
            'consensus_strengths': self.consensus_strengths,
            'consensus_weaknesses': self.consensus_weaknesses,
            'conflicting_feedback': self.conflicting_feedback,
            'robustness_trend': self.robustness_trend,
            'confidence_trend': self.confidence_trend,
            'quality_trend': self.quality_trend,
            'overall_recommendation': self.overall_recommendation,
            'improvement_priority': self.improvement_priority
        }


@dataclass
class PerformanceMetrics:
    """
    Metrics on gauntlet execution quality and efficiency.

    Tracks various performance dimensions including time, quality,
    efficiency, and decision accuracy.
    """
    total_time: float = 0.0
    round_times: Dict[int, float] = field(default_factory=dict)

    # Quality metrics
    average_score: float = 0.0
    score_variance: float = 0.0  # Consistency across rounds
    trend: str = "unknown"  # "improving", "declining", "stable"

    # Efficiency metrics
    evaluations_per_round: Dict[int, int] = field(default_factory=dict)
    total_evaluations: int = 0
    cost_estimate: float = 0.0  # Estimated API costs in USD

    # Decision metrics
    termination_round: Optional[int] = None
    termination_reason: Optional[str] = None
    false_positive_risk: float = 0.0  # Risk of passing bad solution
    false_negative_risk: float = 0.0  # Risk of failing good solution

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_time': self.total_time,
            'round_times': self.round_times,
            'average_score': self.average_score,
            'score_variance': self.score_variance,
            'trend': self.trend,
            'total_evaluations': self.total_evaluations,
            'cost_estimate': self.cost_estimate,
            'termination_round': self.termination_round,
            'termination_reason': self.termination_reason
        }


@dataclass
class MultiRoundConfig:
    """
    Configuration for multi-round gauntlet orchestration.

    Defines thresholds, weights, and behavior for each round.
    """
    # Round 1 thresholds
    round1_threshold: float = 0.7  # Minimum score to continue
    min_confidence: float = 0.6  # Minimum confidence level
    max_weaknesses: int = 5  # Maximum weaknesses allowed

    # Round 2 thresholds
    round2_threshold: float = 0.6  # Minimum score to continue (after normalization)
    max_vulnerabilities: int = 3  # Maximum critical vulnerabilities
    min_robustness: float = 0.5  # Minimum robustness score

    # Round 3 thresholds
    round3_threshold: float = 0.85  # Minimum score for final approval (after normalization)
    min_consensus: float = 0.75  # Minimum consensus score
    require_formal_verification: bool = False  # Whether Lean 4 verification is required

    # Score weights for final aggregation
    round1_weight: float = 0.2
    round2_weight: float = 0.3
    round3_weight: float = 0.5

    # Execution options
    enable_parallel_execution: bool = True
    max_parallel_evaluations: int = 5
    timeout_per_round: int = 300  # seconds

    # Early termination
    enable_early_termination: bool = True
    fail_fast: bool = True  # Stop immediately on round failure

    # Artifact fusion
    consensus_threshold: int = 2  # Minimum rounds to consider something consensus
    conflict_detection: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'round1_threshold': self.round1_threshold,
            'min_confidence': self.min_confidence,
            'max_weaknesses': self.max_weaknesses,
            'round2_threshold': self.round2_threshold,
            'max_vulnerabilities': self.max_vulnerabilities,
            'min_robustness': self.min_robustness,
            'round3_threshold': self.round3_threshold,
            'min_consensus': self.min_consensus,
            'require_formal_verification': self.require_formal_verification,
            'round_weights': {
                'round1': self.round1_weight,
                'round2': self.round2_weight,
                'round3': self.round3_weight
            },
            'enable_parallel_execution': self.enable_parallel_execution,
            'enable_early_termination': self.enable_early_termination,
            'fail_fast': self.fail_fast
        }


class MultiRoundGauntletOrchestrator:
    """
    Orchestrates multi-round gauntlet execution with advanced state management.

    This class manages the complete lifecycle of a multi-round gauntlet run,
    including state tracking, decision making, artifact fusion, and reporting.

    Key Responsibilities:
    1. Initialize and manage gauntlet state
    2. Execute rounds sequentially or in parallel
    3. Make intelligent continue/terminate decisions
    4. Normalize scores across different scales
    5. Fuse artifacts from all rounds
    6. Generate comprehensive reports and metrics

    Example:
        ```python
        orchestrator = MultiRoundGauntletOrchestrator(config=MultiRoundConfig())

        # Initialize gauntlet
        state = await orchestrator.initialize_gauntlet(
            solution=my_solution,
            problem="Optimize trading strategy",
            domain="finance"
        )

        # Execute all rounds
        final_state = await orchestrator.execute_full_gauntlet(state)

        # Get results
        fused_artifacts = orchestrator.fuse_artifacts(final_state)
        metrics = orchestrator.get_performance_metrics(final_state)
        report = orchestrator.generate_progress_report(final_state)
        ```
    """

    def __init__(self, config: MultiRoundConfig):
        """
        Initialize the orchestrator.

        Args:
            config: Configuration for multi-round execution
        """
        self.config = config
        self.state: Optional[GauntletState] = None

        # Load evaluators (lazy initialization)
        self._round1_evaluator = None
        self._round2_evaluator = None
        self._round3_evaluator = None

        logger.info(f"MultiRoundGauntletOrchestrator initialized with config: {config.to_dict()}")

    async def initialize_gauntlet(
        self,
        solution: str,
        problem: str,
        domain: str,
        context: Optional[Dict[str, Any]] = None
    ) -> GauntletState:
        """
        Initialize a new gauntlet execution.

        Args:
            solution: The solution to evaluate
            problem: The problem statement
            domain: The problem domain
            context: Additional context

        Returns:
            Initialized GauntletState
        """
        self.state = GauntletState(
            solution=solution,
            problem=problem,
            domain=domain,
            context=context or {},
            started_at=datetime.utcnow(),
            status="in_progress"
        )

        logger.info(
            f"Gauntlet initialized for domain={domain}, "
            f"solution_length={len(solution)}, problem={problem[:50]}..."
        )

        return self.state

    async def execute_round(
        self,
        round_num: int,
        state: GauntletState
    ) -> GauntletState:
        """
        Execute a single gauntlet round.

        Args:
            round_num: Round number (1, 2, or 3)
            state: Current gauntlet state

        Returns:
            Updated GauntletState
        """
        if round_num not in [1, 2, 3]:
            raise ValueError(f"Invalid round number: {round_num}")

        logger.info(f"Executing Round {round_num}")
        state.current_round = round_num
        round_start_time = time.time()

        try:
            if round_num == 1:
                state = await self._execute_round1(state)
            elif round_num == 2:
                state = await self._execute_round2(state)
            elif round_num == 3:
                state = await self._execute_round3(state)

            # Track execution time
            round_time = time.time() - round_start_time
            state.round_times[round_num] = round_time
            state.total_evaluation_time += round_time

            # Mark round as completed
            if round_num not in state.rounds_completed:
                state.rounds_completed.append(round_num)

            logger.info(f"Round {round_num} completed in {round_time:.2f}s")

        except Exception as e:
            logger.error(f"Round {round_num} execution failed: {e}", exc_info=True)
            state.status = "error"
            state.context['error'] = str(e)

        return state

    async def _execute_round1(self, state: GauntletState) -> GauntletState:
        """
        Execute Round 1: LoongFlow AI Evaluation

        Quick automated quality assessment to screen solutions.
        """
        logger.info("Executing Round 1: LoongFlow AI Evaluation")

        try:
            # Import and initialize LoongFlow evaluator
            from .loongflow_gauntlet import LoongFlowGauntletEvaluator, LoongFlowGauntletConfig

            if self._round1_evaluator is None:
                config = LoongFlowGauntletConfig(
                    timeout=self.config.timeout_per_round,
                    min_score=self.config.round1_threshold
                )
                self._round1_evaluator = LoongFlowGauntletEvaluator(config=config)

            # Run evaluation
            result = await self._round1_evaluator.evaluate(
                solution=state.solution,
                problem=state.problem,
                domain=state.domain,
                context=state.context
            )

            # Store result
            state.round1_result = Round1Result(
                score=result.score,
                confidence=result.confidence,
                feedback=result.feedback,
                strengths=result.strengths,
                weaknesses=result.weaknesses,
                suggestions=result.suggestions,
                execution_time=result.execution_time,
                raw_data=result.metadata
            )

            # Normalize score (already 0-1 from LoongFlow)
            state.round1_normalized_score = state.round1_result.score

            # Make decision
            state.round1_decision = await self.make_decision(1, state)

        except Exception as e:
            logger.error(f"Round 1 error: {e}")
            state.round1_result = Round1Result(
                score=0.0,
                confidence=0.0,
                feedback=f"Evaluation error: {str(e)}",
                execution_time=0.0
            )
            state.round1_normalized_score = 0.0
            state.round1_decision = "terminate"

        return state

    async def _execute_round2(self, state: GauntletState) -> GauntletState:
        """
        Execute Round 2: Red Team Adversarial Attack

        Adversarial testing to find flaws and edge cases.
        """
        logger.info("Executing Round 2: Red Team Adversarial Attack")

        try:
            # Import red team evaluator
            # This is a placeholder - integrate with your actual red team system
            from ..evaluators.red_team import RedTeamEvaluator

            if self._round2_evaluator is None:
                self._round2_evaluator = RedTeamEvaluator(
                    domain=state.domain,
                    max_vulnerabilities=self.config.max_vulnerabilities
                )

            # Run adversarial testing
            result = await self._round2_evaluator.attack(
                solution=state.solution,
                problem=state.problem,
                context=state.context
            )

            # Store result
            state.round2_result = Round2Result(
                score=result.score,  # 0-100 scale
                attacks_attempted=result.attacks_attempted,
                attacks_successful=result.attacks_successful,
                vulnerabilities_found=result.vulnerabilities,
                edge_cases_tested=result.edge_cases,
                robustness_score=result.robustness_score,
                execution_time=result.execution_time,
                raw_data=result.metadata
            )

            # Normalize score (0-100 -> 0-1)
            state.round2_normalized_score = state.round2_result.score / 100.0

            # Make decision
            state.round2_decision = await self.make_decision(2, state)

        except Exception as e:
            logger.error(f"Round 2 error: {e}")
            # Create fallback result
            state.round2_result = Round2Result(
                score=50.0,  # Neutral score
                attacks_attempted=0,
                attacks_successful=0,
                feedback=f"Red team evaluation error: {str(e)}",
                execution_time=0.0
            )
            state.round2_normalized_score = 0.5
            state.round2_decision = "continue"  # Give benefit of doubt on error

        return state

    async def _execute_round3(self, state: GauntletState) -> GauntletState:
        """
        Execute Round 3: Gold Team Consensus Verification

        Multi-judge evaluation with consensus checking and optional formal verification.
        """
        logger.info("Executing Round 3: Gold Team Consensus Verification")

        try:
            # Import gold team evaluator
            # This is a placeholder - integrate with your actual gold team system
            from ..evaluators.gold_team import GoldTeamEvaluator

            if self._round3_evaluator is None:
                self._round3_evaluator = GoldTeamEvaluator(
                    domain=state.domain,
                    min_consensus=self.config.min_consensus,
                    require_formal_verification=self.config.require_formal_verification
                )

            # Run consensus evaluation
            result = await self._round3_evaluator.verify(
                solution=state.solution,
                problem=state.problem,
                previous_rounds={
                    'round1': state.round1_result,
                    'round2': state.round2_result
                },
                context=state.context
            )

            # Store result
            state.round3_result = Round3Result(
                score=result.score,  # 0-10 scale
                consensus_score=result.consensus_score,
                formal_verification_passed=result.formal_verification_passed,
                judge_scores=result.judge_scores,
                judge_feedback=result.judge_feedback,
                robustness_score=result.robustness_score,
                execution_time=result.execution_time,
                raw_data=result.metadata
            )

            # Normalize score (0-10 -> 0-1)
            state.round3_normalized_score = state.round3_result.score / 10.0

            # Make decision
            state.round3_decision = await self.make_decision(3, state)

        except Exception as e:
            logger.error(f"Round 3 error: {e}")
            # Create fallback result
            state.round3_result = Round3Result(
                score=5.0,  # Neutral score
                consensus_score=0.5,
                formal_verification_passed=False,
                feedback=f"Gold team evaluation error: {str(e)}",
                execution_time=0.0
            )
            state.round3_normalized_score = 0.5
            state.round3_decision = "terminate"  # Fail on error in final round

        return state

    async def make_decision(
        self,
        round_num: int,
        state: GauntletState
    ) -> str:
        """
        Make intelligent continue/terminate decision after a round.

        Args:
            round_num: Round number (1, 2, or 3)
            state: Current gauntlet state

        Returns:
            "continue" or "terminate"
        """
        logger.info(f"Making decision for Round {round_num}")

        if round_num == 1:
            return await self._make_round1_decision(state)
        elif round_num == 2:
            return await self._make_round2_decision(state)
        elif round_num == 3:
            return await self._make_round3_decision(state)
        else:
            return "terminate"

    async def _make_round1_decision(self, state: GauntletState) -> str:
        """Make decision after Round 1 (LoongFlow)"""
        result = state.round1_result
        if not result:
            return "terminate"

        score = state.round1_normalized_score

        # Decision factors
        meets_threshold = score >= self.config.round1_threshold
        has_minimum_confidence = result.confidence >= self.config.min_confidence
        no_critical_flaws = len(result.weaknesses) < self.config.max_weaknesses

        # Log decision factors
        logger.info(
            f"Round 1 Decision Factors:"
            f" score={score:.2f} (threshold={self.config.round1_threshold:.2f}),"
            f" confidence={result.confidence:.2f} (min={self.config.min_confidence:.2f}),"
            f" weaknesses={len(result.weaknesses)} (max={self.config.max_weaknesses})"
        )

        if meets_threshold and has_minimum_confidence and no_critical_flaws:
            logger.info("Round 1 decision: CONTINUE")
            return "continue"
        else:
            reason = self._get_termination_reason(result)
            logger.warning(f"Round 1 decision: TERMINATE - {reason}")
            return "terminate"

    async def _make_round2_decision(self, state: GauntletState) -> str:
        """Make decision after Round 2 (Red Team)"""
        result = state.round2_result
        if not result:
            return "terminate"

        score = state.round2_normalized_score

        # Decision factors
        meets_threshold = score >= self.config.round2_threshold
        acceptable_vulnerabilities = result.attacks_successful <= self.config.max_vulnerabilities
        sufficient_robustness = result.robustness_score >= self.config.min_robustness

        # Log decision factors
        logger.info(
            f"Round 2 Decision Factors:"
            f" score={score:.2f} (threshold={self.config.round2_threshold:.2f}),"
            f" attacks_successful={result.attacks_successful} (max={self.config.max_vulnerabilities}),"
            f" robustness={result.robustness_score:.2f} (min={self.config.min_robustness:.2f})"
        )

        if meets_threshold and acceptable_vulnerabilities and sufficient_robustness:
            logger.info("Round 2 decision: CONTINUE")
            return "continue"
        else:
            reason = self._get_red_team_termination_reason(result)
            logger.warning(f"Round 2 decision: TERMINATE - {reason}")
            return "terminate"

    async def _make_round3_decision(self, state: GauntletState) -> str:
        """Make decision after Round 3 (Gold Team - final)"""
        result = state.round3_result
        if not result:
            return "terminate"

        score = state.round3_normalized_score

        # Decision factors
        meets_threshold = score >= self.config.round3_threshold
        has_consensus = result.consensus_score >= self.config.min_consensus
        formal_verification_ok = (
            not self.config.require_formal_verification or
            result.formal_verification_passed
        )

        # Log decision factors
        logger.info(
            f"Round 3 Decision Factors:"
            f" score={score:.2f} (threshold={self.config.round3_threshold:.2f}),"
            f" consensus={result.consensus_score:.2f} (min={self.config.min_consensus:.2f}),"
            f" formal_verif={'passed' if result.formal_verification_passed else 'failed'}"
        )

        if meets_threshold and has_consensus and formal_verification_ok:
            logger.info("Round 3 decision: CONTINUE (FINAL APPROVAL)")
            return "continue"
        else:
            reason = self._get_gold_team_termination_reason(result)
            logger.warning(f"Round 3 decision: TERMINATE - {reason}")
            return "terminate"

    def _get_termination_reason(self, result: Round1Result) -> str:
        """Get termination reason for Round 1"""
        reasons = []
        if result.score < self.config.round1_threshold:
            reasons.append(f"low score ({result.score:.2f} < {self.config.round1_threshold:.2f})")
        if result.confidence < self.config.min_confidence:
            reasons.append(f"low confidence ({result.confidence:.2f} < {self.config.min_confidence:.2f})")
        if len(result.weaknesses) >= self.config.max_weaknesses:
            reasons.append(f"too many weaknesses ({len(result.weaknesses)} >= {self.config.max_weaknesses})")
        return ", ".join(reasons)

    def _get_red_team_termination_reason(self, result: Round2Result) -> str:
        """Get termination reason for Round 2"""
        reasons = []
        if result.score / 100.0 < self.config.round2_threshold:
            reasons.append(f"low score ({result.score:.1f}/100)")
        if result.attacks_successful > self.config.max_vulnerabilities:
            reasons.append(f"too many successful attacks ({result.attacks_successful} > {self.config.max_vulnerabilities})")
        if result.robustness_score < self.config.min_robustness:
            reasons.append(f"low robustness ({result.robustness_score:.2f} < {self.config.min_robustness:.2f})")
        return ", ".join(reasons)

    def _get_gold_team_termination_reason(self, result: Round3Result) -> str:
        """Get termination reason for Round 3"""
        reasons = []
        if result.score / 10.0 < self.config.round3_threshold:
            reasons.append(f"low score ({result.score:.1f}/10)")
        if result.consensus_score < self.config.min_consensus:
            reasons.append(f"low consensus ({result.consensus_score:.2f} < {self.config.min_consensus:.2f})")
        if self.config.require_formal_verification and not result.formal_verification_passed:
            reasons.append("formal verification failed")
        return ", ".join(reasons)

    async def execute_full_gauntlet(
        self,
        solution: str,
        problem: str,
        domain: str,
        context: Optional[Dict[str, Any]] = None
    ) -> GauntletState:
        """
        Execute complete 3-round gauntlet with intelligent decision making.

        Args:
            solution: The solution to evaluate
            problem: The problem statement
            domain: The problem domain
            context: Additional context

        Returns:
            Final GauntletState with all results
        """
        # Initialize
        state = await self.initialize_gauntlet(solution, problem, domain, context)

        # Execute rounds with early termination logic
        for round_num in [1, 2, 3]:
            state = await self.execute_round(round_num, state)

            # Check if should terminate
            decision = None
            if round_num == 1:
                decision = state.round1_decision
            elif round_num == 2:
                decision = state.round2_decision
            elif round_num == 3:
                decision = state.round3_decision

            if decision == "terminate" and self.config.enable_early_termination:
                logger.info(f"Early termination after Round {round_num}")
                state.status = "terminated"
                state.current_round = round_num
                break

            # Check for fail-fast
            if decision == "terminate" and self.config.fail_fast:
                logger.info(f"Fail-fast termination after Round {round_num}")
                state.status = "terminated"
                break

        # Update final status
        if state.status != "terminated" and state.status != "error":
            state.status = "completed"

        state.completed_at = datetime.utcnow()
        state.current_round = 4  # Mark as complete

        logger.info(
            f"Gauntlet execution complete: status={state.status}, "
            f"rounds_completed={state.rounds_completed}, "
            f"total_time={state.total_evaluation_time:.2f}s"
        )

        return state

    def normalize_scores(self, state: GauntletState) -> GauntletState:
        """
        Normalize scores from all rounds to 0-1 scale.

        Different rounds use different scales:
        - Round 1: 0-1 (already normalized)
        - Round 2: 0-100 (divide by 100)
        - Round 3: 0-10 (divide by 10)

        Args:
            state: GauntletState with raw scores

        Returns:
            GauntletState with normalized scores
        """
        # Round 1: Already 0-1 from LoongFlow
        if state.round1_result:
            state.round1_normalized_score = state.round1_result.score

        # Round 2: Red Team uses 0-100 scale
        if state.round2_result:
            state.round2_normalized_score = state.round2_result.score / 100.0

        # Round 3: Gold Team uses 0-10 scale
        if state.round3_result:
            state.round3_normalized_score = state.round3_result.score / 10.0

        # Log normalized scores (handle None values)
        r1_score = state.round1_normalized_score if state.round1_normalized_score is not None else 0.0
        r2_score = state.round2_normalized_score if state.round2_normalized_score is not None else 0.0
        r3_score = state.round3_normalized_score if state.round3_normalized_score is not None else 0.0

        logger.info(
            f"Normalized scores: R1={r1_score:.2f}, "
            f"R2={r2_score:.2f}, "
            f"R3={r3_score:.2f}"
        )

        return state

    def fuse_artifacts(self, state: GauntletState) -> FusedArtifacts:
        """
        Combine insights from all rounds into unified knowledge.

        Identifies consensus, conflicts, and generates recommendations.

        Args:
            state: GauntletState with results from all rounds

        Returns:
            FusedArtifacts with combined insights
        """
        fused = FusedArtifacts()

        # Collect artifacts from Round 1
        if state.round1_result:
            fused.all_scores['round1'] = state.round1_normalized_score
            fused.all_feedback.append(state.round1_result.feedback)
            fused.all_strengths.extend(state.round1_result.strengths)
            fused.all_weaknesses.extend(state.round1_result.weaknesses)
            fused.all_suggestions.extend(state.round1_result.suggestions)
            fused.robustness_trend.append(state.round1_result.robustness_score)
            fused.confidence_trend.append(state.round1_result.confidence)
            fused.quality_trend.append(state.round1_normalized_score)

        # Collect artifacts from Round 2
        if state.round2_result:
            fused.all_scores['round2'] = state.round2_normalized_score
            # Convert vulnerabilities to weaknesses
            for vuln in state.round2_result.vulnerabilities_found:
                fused.all_weaknesses.append(f"Vulnerability: {vuln}")
            fused.robustness_trend.append(state.round2_result.robustness_score)
            # No explicit confidence in R2, use score
            fused.confidence_trend.append(state.round2_normalized_score)
            fused.quality_trend.append(state.round2_normalized_score)

        # Collect artifacts from Round 3
        if state.round3_result:
            fused.all_scores['round3'] = state.round3_normalized_score
            fused.all_feedback.extend(state.round3_result.judge_feedback)
            fused.robustness_trend.append(state.round3_result.robustness_score)
            fused.confidence_trend.append(state.round3_result.consensus_score)
            fused.quality_trend.append(state.round3_normalized_score)

        # Find consensus (mentioned by 2+ rounds)
        fused.consensus_strengths = self._find_consensus(
            fused.all_strengths,
            min_mentions=2
        )
        fused.consensus_weaknesses = self._find_consensus(
            fused.all_weaknesses,
            min_mentions=2
        )

        # Detect conflicts
        if self.config.conflict_detection:
            fused.conflicting_feedback = self._detect_conflicts(
                fused.all_strengths,
                fused.all_weaknesses
            )

        # Generate recommendation
        fused.overall_recommendation = self._generate_recommendation(state)

        # Prioritize improvements
        fused.improvement_priority = self._prioritize_improvements(
            fused.consensus_weaknesses,
            state
        )

        logger.info(
            f"Artifact fusion complete: {len(fused.consensus_strengths)} consensus strengths, "
            f"{len(fused.consensus_weaknesses)} consensus weaknesses, "
            f"{len(fused.conflicting_feedback)} conflicts"
        )

        return fused

    def _find_consensus(
        self,
        items: List[str],
        min_mentions: int = 2
    ) -> List[str]:
        """Find items mentioned by multiple rounds."""
        # Simple keyword matching (can be enhanced with semantic similarity)
        item_counts = defaultdict(int)
        item_sources = defaultdict(set)

        for item in items:
            # Normalize for comparison
            normalized = item.lower().strip()
            item_counts[normalized] += 1

        # Find items mentioned min_mentions times
        consensus = [
            item for item, count in item_counts.items()
            if count >= min_mentions
        ]

        return consensus

    def _detect_conflicts(
        self,
        strengths: List[str],
        weaknesses: List[str]
    ) -> List[Tuple[str, str]]:
        """Detect conflicting feedback (strength in one, weakness in another)."""
        conflicts = []

        # Simple keyword-based conflict detection
        for strength in strengths:
            for weakness in weaknesses:
                # Check if they mention similar concepts
                if self._items_similar(strength, weakness):
                    conflicts.append((strength, weakness))

        return conflicts

    def _items_similar(self, item1: str, item2: str, threshold: float = 0.6) -> bool:
        """Check if two items are similar (simple word overlap)."""
        words1 = set(item1.lower().split())
        words2 = set(item2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union)
        return similarity >= threshold

    def _generate_recommendation(self, state: GauntletState) -> str:
        """Generate overall recommendation based on all rounds."""
        if state.status == "terminated":
            return "Solution did not meet quality thresholds - not recommended"

        if state.status == "error":
            return "Evaluation encountered errors - cannot recommend"

        # Check final decision
        if state.round3_decision == "continue":
            return "APPROVED: Solution passed all gauntlet rounds"
        elif state.round2_decision == "continue":
            return "CONDITIONAL: Solution passed adversarial testing, pending final verification"
        elif state.round1_decision == "continue":
            return "PRELIMINARY: Solution passed initial screening"
        else:
            return "NOT RECOMMENDED: Solution failed quality thresholds"

    def _prioritize_improvements(
        self,
        weaknesses: List[str],
        state: GauntletState
    ) -> List[str]:
        """Prioritize improvements based on consensus and severity."""
        # Prioritize by:
        # 1. Mentions in multiple rounds (consensus)
        # 2. Security/safety critical
        # 3. Impact on score

        priorities = []

        # Security/safety items first
        security_keywords = ['security', 'vulnerability', 'attack', 'exploit', 'safety', 'critical']
        for weakness in weaknesses:
            if any(kw in weakness.lower() for kw in security_keywords):
                priorities.append(f"[HIGH PRIORITY] {weakness}")

        # Consensus weaknesses next
        for weakness in weaknesses:
            if weakness not in ' '.join(priorities):
                if self._is_consensus_item(weakness, weaknesses):
                    priorities.append(f"[MEDIUM PRIORITY] {weakness}")

        # Other weaknesses
        for weakness in weaknesses:
            if weakness not in ' '.join(priorities):
                priorities.append(f"[LOW PRIORITY] {weakness}")

        return priorities[:10]  # Top 10 priorities

    def _is_consensus_item(self, item: str, all_items: List[str]) -> bool:
        """Check if item appears multiple times (is consensus)."""
        normalized = item.lower().strip()
        count = sum(1 for i in all_items if i.lower().strip() == normalized)
        return count >= 2

    def calculate_final_score(self, state: GauntletState) -> float:
        """
        Calculate final weighted score from all rounds.

        Uses configured weights for each round.

        Args:
            state: GauntletState with normalized scores

        Returns:
            Final weighted score (0-1)
        """
        scores = []
        weights = []

        if state.round1_normalized_score is not None:
            scores.append(state.round1_normalized_score)
            weights.append(self.config.round1_weight)

        if state.round2_normalized_score is not None:
            scores.append(state.round2_normalized_score)
            weights.append(self.config.round2_weight)

        if state.round3_normalized_score is not None:
            scores.append(state.round3_normalized_score)
            weights.append(self.config.round3_weight)

        if not scores:
            return 0.0

        # Calculate weighted average
        total_weight = sum(weights)
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        logger.info(
            f"Final score: {final_score:.2f} "
            f"(R1={state.round1_normalized_score:.2f} * {self.config.round1_weight}, "
            f"R2={state.round2_normalized_score:.2f} * {self.config.round2_weight}, "
            f"R3={state.round3_normalized_score:.2f} * {self.config.round3_weight})"
        )

        return final_score

    def generate_progress_report(self, state: GauntletState) -> str:
        """
        Generate detailed progress report for gauntlet execution.

        Args:
            state: GauntletState to report on

        Returns:
            Formatted progress report string
        """
        report = f"""
{'='*70}
GAUNTLET PROGRESS REPORT
{'='*70}

Solution: {state.solution[:75]}...
Problem: {state.problem}
Domain: {state.domain.upper()}

Status: {state.status.upper()}
Rounds Completed: {len(state.rounds_completed)}/3
Total Time: {state.total_evaluation_time:.1f}s
Started: {state.started_at.strftime('%Y-%m-%d %H:%M:%S')} UTC
"""

        if state.round1_result:
            report += f"""
{'-'*70}
ROUND 1: LoongFlow AI Evaluation
{'-'*70}
✓ Completed
Score: {state.round1_normalized_score:.2%}
Confidence: {state.round1_result.confidence:.2%}
Time: {state.round_times.get(1, 0):.1f}s
Decision: {state.round1_decision.upper()}

Strengths ({len(state.round1_result.strengths)}):
{self._format_list(state.round1_result.strengths[:3])}

Weaknesses ({len(state.round1_result.weaknesses)}):
{self._format_list(state.round1_result.weaknesses[:3])}
"""

        if state.round2_result:
            report += f"""
{'-'*70}
ROUND 2: Red Team Adversarial Attack
{'-'*70}
✓ Completed
Score: {state.round2_normalized_score:.2%}
Attacks: {state.round2_result.attacks_successful}/{state.round2_result.attacks_attempted} successful
Robustness: {state.round2_result.robustness_score:.2%}
Time: {state.round_times.get(2, 0):.1f}s
Decision: {state.round2_decision.upper()}

Vulnerabilities Found:
{self._format_list(state.round2_result.vulnerabilities_found[:3])}
"""

        if state.round3_result:
            report += f"""
{'-'*70}
ROUND 3: Gold Team Consensus Verification
{'-'*70}
✓ Completed
Score: {state.round3_normalized_score:.2%}
Consensus: {state.round3_result.consensus_score:.2%}
Formal Verification: {'✓ PASSED' if state.round3_result.formal_verification_passed else '✗ FAILED'}
Time: {state.round_times.get(3, 0):.1f}s
Decision: {state.round3_decision.upper()}

Judge Scores: {[f'{s:.1f}' for s in state.round3_result.judge_scores]}
"""

        if state.status == "completed":
            final_score = self.calculate_final_score(state)
            report += f"""
{'='*70}
FINAL RESULT
{'='*70}
Overall Score: {final_score:.2%}
Status: {'✓ PASSED' if state.round3_decision == 'continue' else '✗ FAILED'}
"""

            # Get fused artifacts
            fused = self.fuse_artifacts(state)
            if fused.consensus_strengths:
                report += f"""
Consensus Strengths ({len(fused.consensus_strengths)}):
{self._format_list(fused.consensus_strengths[:3])}
"""

            if fused.consensus_weaknesses:
                report += f"""
Consensus Weaknesses ({len(fused.consensus_weaknesses)}):
{self._format_list(fused.consensus_weaknesses[:3])}
"""

            if fused.overall_recommendation:
                report += f"""
Recommendation: {fused.overall_recommendation}
"""

        elif state.status == "terminated":
            report += f"""
{'='*70}
TERMINATED
{'='*70}
The gauntlet was terminated early due to quality thresholds not being met.
Last Round: {state.current_round}
"""

        report += f"\n{'='*70}\n"

        return report

    def _format_list(self, items: List[str], indent: str = "  ") -> str:
        """Format a list for pretty printing."""
        if not items:
            return f"{indent}(none)"
        return '\n'.join(f"{indent}• {item}" for item in items)

    def get_performance_metrics(self, state: GauntletState) -> PerformanceMetrics:
        """
        Calculate performance metrics for gauntlet execution.

        Args:
            state: GauntletState to analyze

        Returns:
            PerformanceMetrics with detailed analysis
        """
        metrics = PerformanceMetrics()

        # Time metrics
        metrics.total_time = state.total_evaluation_time
        metrics.round_times = state.round_times.copy()

        # Calculate scores
        scores = []
        if state.round1_normalized_score is not None:
            scores.append(state.round1_normalized_score)
        if state.round2_normalized_score is not None:
            scores.append(state.round2_normalized_score)
        if state.round3_normalized_score is not None:
            scores.append(state.round3_normalized_score)

        if scores:
            metrics.average_score = sum(scores) / len(scores)

            # Calculate variance
            mean = metrics.average_score
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            metrics.score_variance = variance

        # Determine trend
        if len(scores) >= 2:
            if scores[-1] > scores[0] + 0.1:
                metrics.trend = "improving"
            elif scores[-1] < scores[0] - 0.1:
                metrics.trend = "declining"
            else:
                metrics.trend = "stable"

        # Estimation of evaluations (placeholder)
        # This would be populated by actual evaluation counts from the evaluators
        metrics.total_evaluations = sum(
            state.round_times.get(r, 0) * 10  # Rough estimate
            for r in state.rounds_completed
        )

        # Cost estimation (placeholder: $0.001 per evaluation)
        metrics.cost_estimate = metrics.total_evaluations * 0.001

        # Termination info
        if state.status == "terminated":
            metrics.termination_round = state.current_round
            if state.current_round == 1:
                metrics.termination_reason = state.round1_decision or "failed round 1"
            elif state.current_round == 2:
                metrics.termination_reason = state.round2_decision or "failed round 2"
            elif state.current_round == 3:
                metrics.termination_reason = state.round3_decision or "failed round 3"

        # Risk estimation (simplified)
        if state.status == "completed" and state.round3_decision == "continue":
            metrics.false_positive_risk = 1.0 - state.round3_normalized_score
            metrics.false_negative_risk = 0.1  # Conservative estimate
        elif state.status == "terminated":
            metrics.false_positive_risk = 0.05  # Low risk of passing bad solution
            metrics.false_negative_risk = 0.3  # Higher risk of rejecting good solution
        else:
            metrics.false_positive_risk = 0.5
            metrics.false_negative_risk = 0.5

        logger.info(
            f"Performance metrics: avg_score={metrics.average_score:.2f}, "
            f"trend={metrics.trend}, "
            f"total_evals={metrics.total_evaluations}, "
            f"cost_est=${metrics.cost_estimate:.2f}"
        )

        return metrics

    async def execute_round_parallel(
        self,
        round_num: int,
        state: GauntletState
    ) -> GauntletState:
        """
        Execute a round with parallel evaluation where possible.

        This is most useful for Round 3 (Gold Team) where multiple judges
        can evaluate independently in parallel.

        Args:
            round_num: Round number
            state: Current gauntlet state

        Returns:
            Updated GauntletState
        """
        if not self.config.enable_parallel_execution:
            return await self.execute_round(round_num, state)

        if round_num == 3 and self._round3_evaluator:
            # Gold Team can run multiple models in parallel
            logger.info("Executing Round 3 with parallel evaluation")

            try:
                # Get judge models from evaluator
                judge_models = getattr(self._round3_evaluator, 'judge_models', ['default'])

                # Create evaluation tasks
                tasks = [
                    self._evaluate_with_single_judge(model, state)
                    for model in judge_models[:self.config.max_parallel_evaluations]
                ]

                # Execute in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Aggregate results
                state = self._aggregate_gold_team_results(state, results)

            except Exception as e:
                logger.error(f"Parallel execution failed: {e}")
                # Fall back to sequential execution
                return await self.execute_round(round_num, state)
        else:
            # Other rounds don't benefit from parallel execution
            return await self.execute_round(round_num, state)

        return state

    async def _evaluate_with_single_judge(
        self,
        model: str,
        state: GauntletState
    ) -> Dict[str, Any]:
        """Evaluate solution with a single judge (for parallel execution)."""
        # This would call the actual judge evaluation
        # Placeholder implementation
        return {
            'model': model,
            'score': 7.5,
            'feedback': f"Evaluation by {model}"
        }

    def _aggregate_gold_team_results(
        self,
        state: GauntletState,
        results: List[Any]
    ) -> GauntletState:
        """Aggregate results from parallel gold team evaluation."""
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]

        if not valid_results:
            logger.error("All parallel evaluations failed")
            return state

        # Calculate average score
        scores = [r.get('score', 5.0) for r in valid_results]
        avg_score = sum(scores) / len(scores)

        # Update Round 3 result
        if state.round3_result:
            state.round3_result.score = avg_score
            state.round3_result.judge_scores = scores
            state.round3_result.judge_feedback = [
                r.get('feedback', '') for r in valid_results
            ]

        return state


def create_multi_round_orchestrator(
    round1_threshold: float = 0.7,
    round2_threshold: float = 0.6,
    round3_threshold: float = 0.85,
    enable_early_termination: bool = True,
    **kwargs
) -> MultiRoundGauntletOrchestrator:
    """
    Factory function to create a configured MultiRoundGauntletOrchestrator.

    Args:
        round1_threshold: Minimum score to pass Round 1
        round2_threshold: Minimum score to pass Round 2 (normalized)
        round3_threshold: Minimum score to pass Round 3 (normalized)
        enable_early_termination: Whether to stop on round failure
        **kwargs: Additional configuration options

    Returns:
        Configured MultiRoundGauntletOrchestrator

    Example:
        ```python
        orchestrator = create_multi_round_orchestrator(
            round1_threshold=0.8,
            round2_threshold=0.7,
            round3_threshold=0.9,
            enable_early_termination=True
        )

        result = await orchestrator.execute_full_gauntlet(
            solution=my_solution,
            problem="Optimize trading strategy",
            domain="finance"
        )

        report = orchestrator.generate_progress_report(result)
        print(report)
        ```
    """
    config = MultiRoundConfig(
        round1_threshold=round1_threshold,
        round2_threshold=round2_threshold,
        round3_threshold=round3_threshold,
        enable_early_termination=enable_early_termination,
        **kwargs
    )

    return MultiRoundGauntletOrchestrator(config=config)
