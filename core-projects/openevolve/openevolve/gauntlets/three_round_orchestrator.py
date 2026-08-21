"""
Three-Round Gauntlet Orchestrator
==================================

Implements progressive 3-round gauntlet evaluation:
- Round 1: LoongFlow AI (Quick Screen) - 20% weight
- Round 2: Red Team (Adversarial) - 30% weight
- Round 3: Gold Team (Consensus) - 50% weight

Progressive filtering allows early termination to save computational resources.
Solutions must pass each round's threshold to proceed to the next.

Author: OpenEvolve Gauntlet System
Date: 2026-01-30
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from openevolve.gauntlets.llm_judge import (
    GauntletJudge,
    consensus_score,
    describe_attacks,
    probe_solution,
    robustness_from_probes,
    successful_attacks,
    verify_solution,
)

# Configure logging
logger = logging.getLogger(__name__)

# Blend of LLM judge verdict vs. deterministic static analysis in each round
JUDGE_WEIGHT = 0.5
STATIC_WEIGHT = 0.5
VERIFICATION_WEIGHT = 0.2


class GauntletRound(Enum):
    """Gauntlet round identifiers"""
    LOONGFLOW = "round1_loongflow"
    RED_TEAM = "round2_red_team"
    GOLD_TEAM = "round3_gold_team"


@dataclass
class ThreeRoundConfig:
    """
    Configuration for 3-round gauntlet orchestrator.

    Attributes:
        round1_config: LoongFlow evaluator configuration
        round1_weight: Weight for Round 1 in final score (0.0-1.0)
        round1_threshold: Minimum score to pass Round 1 (0.0-1.0)
        round1_enabled: Whether Round 1 is enabled

        round2_config: Red Team evaluator configuration
        round2_weight: Weight for Round 2 in final score (0.0-1.0)
        round2_threshold: Minimum score to pass Round 2 (0.0-1.0)
        round2_enabled: Whether Round 2 is enabled

        round3_config: Gold Team evaluator configuration
        round3_weight: Weight for Round 3 in final score (0.0-1.0)
        round3_threshold: Minimum score to pass Round 3 (0.0-1.0)
        round3_enabled: Whether Round 3 is enabled

        enable_early_termination: Stop evaluation if fails early round
        enable_parallel_execution: Run rounds in parallel when possible
        aggregate_artifacts: Collect artifacts from all rounds
        generate_detailed_report: Generate comprehensive report
    """
    # Round 1 Configuration
    round1_config: Dict[str, Any] = field(default_factory=dict)
    round1_weight: float = 0.2
    round1_threshold: float = 0.5
    round1_enabled: bool = True

    # Round 2 Configuration
    round2_config: Dict[str, Any] = field(default_factory=dict)
    round2_weight: float = 0.3
    round2_threshold: float = 0.6
    round2_enabled: bool = True

    # Round 3 Configuration
    round3_config: Dict[str, Any] = field(default_factory=dict)
    round3_weight: float = 0.5
    round3_threshold: float = 0.7
    round3_enabled: bool = True

    # Global Configuration
    enable_early_termination: bool = True
    enable_parallel_execution: bool = False
    aggregate_artifacts: bool = True
    generate_detailed_report: bool = True

    def __post_init__(self):
        """Validate configuration"""
        # Validate weights sum to approximately 1.0 if all rounds enabled
        if self.round1_enabled and self.round2_enabled and self.round3_enabled:
            total_weight = self.round1_weight + self.round2_weight + self.round3_weight
            if not (0.9 <= total_weight <= 1.1):
                logger.warning(
                    f"Weights sum to {total_weight}, expected ~1.0. "
                    "Scores will be normalized."
                )

        # Validate thresholds are in valid range
        for name, threshold in [
            ("round1", self.round1_threshold),
            ("round2", self.round2_threshold),
            ("round3", self.round3_threshold)
        ]:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name}_threshold must be 0.0-1.0, got {threshold}")


@dataclass
class Round1Result:
    """
    Result from Round 1 (LoongFlow evaluation).

    Attributes:
        passed: Whether the solution passed this round
        score: Score achieved (0.0-1.0+)
        confidence: Confidence in the evaluation
        evaluation_time: Time taken for evaluation (seconds)
        feedback: Human-readable feedback
        artifacts: Additional artifacts from evaluation
        evaluator_type: Type of evaluator used (loongflow/fallback)
    """
    passed: bool
    score: float
    confidence: float
    evaluation_time: float
    feedback: str
    artifacts: List[Any] = field(default_factory=list)
    evaluator_type: str = "loongflow"
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "round": "round1_loongflow",
            "passed": self.passed,
            "score": self.score,
            "confidence": self.confidence,
            "evaluation_time": self.evaluation_time,
            "feedback": self.feedback,
            "artifacts_count": len(self.artifacts),
            "evaluator_type": self.evaluator_type,
            "timestamp": self.timestamp
        }


@dataclass
class Round2Result:
    """
    Result from Round 2 (Red Team adversarial evaluation).

    Attributes:
        passed: Whether the solution survived adversarial testing
        score: Score achieved (0.0-1.0+)
        attacks_attempted: Number of attack vectors attempted
        attacks_successful: Number of attacks that succeeded
        robustness_score: Measure of solution robustness (0.0-1.0)
        evaluation_time: Time taken for evaluation (seconds)
        feedback: Human-readable feedback
        artifacts: Additional artifacts from evaluation
        attack_details: Details of specific attacks
    """
    passed: bool
    score: float
    attacks_attempted: int
    attacks_successful: int
    robustness_score: float
    evaluation_time: float
    feedback: str
    artifacts: List[Any] = field(default_factory=list)
    attack_details: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "round": "round2_red_team",
            "passed": self.passed,
            "score": self.score,
            "attacks_attempted": self.attacks_attempted,
            "attacks_successful": self.attacks_successful,
            "robustness_score": self.robustness_score,
            "evaluation_time": self.evaluation_time,
            "feedback": self.feedback,
            "artifacts_count": len(self.artifacts),
            "attack_count": len(self.attack_details),
            "timestamp": self.timestamp
        }


@dataclass
class Round3Result:
    """
    Result from Round 3 (Gold Team consensus verification).

    Attributes:
        passed: Whether the solution achieved consensus approval
        score: Final consensus score (0.0-1.0+)
        consensus_score: Agreement level among evaluators (0.0-1.0)
        formal_verification_passed: Whether formal verification (Lean 4) passed
        evaluation_time: Time taken for evaluation (seconds)
        feedback: Human-readable feedback
        artifacts: Additional artifacts from evaluation
        evaluator_votes: Votes from individual evaluators
    """
    passed: bool
    score: float
    consensus_score: float
    formal_verification_passed: bool
    evaluation_time: float
    feedback: str
    artifacts: List[Any] = field(default_factory=list)
    evaluator_votes: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "round": "round3_gold_team",
            "passed": self.passed,
            "score": self.score,
            "consensus_score": self.consensus_score,
            "formal_verification_passed": self.formal_verification_passed,
            "evaluation_time": self.evaluation_time,
            "feedback": self.feedback,
            "artifacts_count": len(self.artifacts),
            "evaluator_count": len(self.evaluator_votes),
            "timestamp": self.timestamp
        }


@dataclass
class FullGauntletResult:
    """
    Complete result from 3-round gauntlet evaluation.

    Attributes:
        solution: The solution that was evaluated
        problem: Problem statement
        domain: Domain of the problem

        round1_result: Round 1 result (None if not reached)
        round2_result: Round 2 result (None if not reached)
        round3_result: Round 3 result (None if not reached)

        passed: Whether the solution passed all attempted rounds
        final_score: Final weighted aggregate score
        rounds_completed: Number of rounds completed (1, 2, or 3)
        termination_reason: Reason for early termination (if applicable)

        artifacts_from_all_rounds: Collected artifacts from all rounds
        total_time: Total evaluation time (seconds)
        timestamp: When evaluation was performed
        comprehensive_report: Generated comprehensive report
    """
    solution: str
    problem: str
    domain: str

    round1_result: Optional[Round1Result]
    round2_result: Optional[Round2Result]
    round3_result: Optional[Round3Result]

    passed: bool
    final_score: float
    rounds_completed: int
    termination_reason: Optional[str]

    artifacts_from_all_rounds: List[Any] = field(default_factory=list)
    total_time: float = 0.0
    timestamp: float = field(default_factory=lambda: time.time())
    comprehensive_report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "solution_hash": hash(self.solution) % 10000,  # Anonymized
            "problem": self.problem[:100] + "..." if len(self.problem) > 100 else self.problem,
            "domain": self.domain,
            "passed": self.passed,
            "final_score": self.final_score,
            "rounds_completed": self.rounds_completed,
            "termination_reason": self.termination_reason,
            "round1": self.round1_result.to_dict() if self.round1_result else None,
            "round2": self.round2_result.to_dict() if self.round2_result else None,
            "round3": self.round3_result.to_dict() if self.round3_result else None,
            "total_artifacts": len(self.artifacts_from_all_rounds),
            "total_time": self.total_time,
            "timestamp": self.timestamp
        }


class ThreeRoundGauntletOrchestrator:
    """
    Orchestrates 3-round progressive gauntlet evaluation.

    This class implements the enhanced gauntlet system with:
    - Progressive filtering (early termination on failure)
    - Weighted score aggregation
    - Configurable thresholds and weights
    - Comprehensive reporting
    - Artifact collection

    Example:
        ```python
        orchestrator = ThreeRoundGauntletOrchestrator(
            config=ThreeRoundConfig(
                round1_threshold=0.5,
                round2_threshold=0.6,
                round3_threshold=0.7
            )
        )

        result = await orchestrator.run_full_gauntlet(
            solution="def solve(): ...",
            problem="Optimize portfolio allocation",
            domain="finance"
        )

        print(f"Passed: {result.passed}")
        print(f"Score: {result.final_score}")
        print(f"Rounds: {result.rounds_completed}")
        ```
    """

    def __init__(self, config: ThreeRoundConfig):
        """
        Initialize the orchestrator.

        Args:
            config: Configuration for all three rounds
        """
        self.config = config
        self.round1_evaluator = None
        self.round2_evaluator = None
        self.round3_evaluator = None

        self._initialize_evaluators()

    def _initialize_evaluators(self):
        """Initialize evaluators for each round"""
        # Round 1: LoongFlow. The external adapter is optional; when it is not
        # importable the in-library evaluator (which has its own degraded mode) is
        # used so Rounds 2 and 3 still get to run.
        if self.config.round1_enabled:
            try:
                from evaluators.loongflow_adapter import create_loongflow_evaluator

                self.round1_evaluator = create_loongflow_evaluator(
                    llm_config=self.config.round1_config.get('llm_config', {}),
                    timeout=self.config.round1_config.get('timeout', 60)
                )
                logger.info("Round 1 (LoongFlow) evaluator initialized")
            except Exception as e:
                logger.warning(
                    f"External LoongFlow adapter unavailable ({e}); "
                    "using the in-library LoongFlow evaluator"
                )
                try:
                    from openevolve.gauntlets.loongflow_gauntlet import (
                        LoongFlowGauntletConfig,
                        LoongFlowGauntletEvaluator,
                    )

                    self.round1_evaluator = LoongFlowGauntletEvaluator(
                        LoongFlowGauntletConfig(
                            quality_threshold=self.config.round1_threshold,
                            # Screening decision is score-based; confidence is only
                            # informative when LoongFlow itself is degraded
                            confidence_threshold=0.0,
                            evaluation_timeout=self.config.round1_config.get('timeout', 60),
                        )
                    )
                except Exception as fallback_error:
                    # run_round1 reports "not initialized" and the round is skipped
                    self.round1_evaluator = None
                    logger.warning(f"Round 1 evaluator unavailable: {fallback_error}")

        try:
            # Round 2: Red Team - LLM judge + deterministic adversarial probes
            if self.config.round2_enabled:
                self.round2_evaluator = GauntletJudge(
                    llm_config=self.config.round2_config.get('llm_config')
                )
                logger.info(
                    "Round 2 (Red Team) judge initialized with models: "
                    f"{self.round2_evaluator.model_names}"
                )

            # Round 3: Gold Team - LLM judge(s) + static verification
            if self.config.round3_enabled:
                self.round3_evaluator = GauntletJudge(
                    llm_config=self.config.round3_config.get('llm_config')
                )
                logger.info(
                    "Round 3 (Gold Team) judge initialized with models: "
                    f"{self.round3_evaluator.model_names}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize evaluators: {e}", exc_info=True)
            raise

    async def run_full_gauntlet(
        self,
        solution: str,
        problem: str,
        domain: str
    ) -> FullGauntletResult:
        """
        Run complete 3-round gauntlet evaluation.

        Args:
            solution: Solution to evaluate
            problem: Problem statement
            domain: Problem domain

        Returns:
            FullGauntletResult with complete evaluation results
        """
        start_time = time.time()
        logger.info(f"Starting 3-round gauntlet for domain={domain}")

        # Initialize results
        round1_result = None
        round2_result = None
        round3_result = None

        all_artifacts = []
        termination_reason = None

        try:
            # Round 1: LoongFlow Quick Screen
            if self.config.round1_enabled:
                logger.info("Executing Round 1: LoongFlow AI Evaluation")
                round1_result = await self.run_round1(solution, problem, domain)

                if self.config.aggregate_artifacts:
                    all_artifacts.extend(round1_result.artifacts)

                # Check if should continue
                if not self.should_continue_to_round2(round1_result):
                    termination_reason = f"Failed Round 1 threshold: {round1_result.score} < {self.config.round1_threshold}"
                    logger.info(f"Early termination: {termination_reason}")

                    if self.config.enable_early_termination:
                        return FullGauntletResult(
                            solution=solution,
                            problem=problem,
                            domain=domain,
                            round1_result=round1_result,
                            round2_result=None,
                            round3_result=None,
                            passed=False,
                            final_score=round1_result.score,
                            rounds_completed=1,
                            termination_reason=termination_reason,
                            artifacts_from_all_rounds=all_artifacts,
                            total_time=time.time() - start_time,
                            comprehensive_report=self._generate_report(
                                round1_result, None, None, termination_reason
                            )
                        )

            # Round 2: Red Team Adversarial
            if self.config.round2_enabled:
                logger.info("Executing Round 2: Red Team Attack")
                round2_result = await self.run_round2(solution, problem, domain)

                if self.config.aggregate_artifacts:
                    all_artifacts.extend(round2_result.artifacts)

                # Check if should continue
                if not self.should_continue_to_round3(round2_result):
                    termination_reason = f"Failed Round 2 threshold: {round2_result.score} < {self.config.round2_threshold}"
                    logger.info(f"Early termination: {termination_reason}")

                    if self.config.enable_early_termination:
                        # Calculate partial final score
                        final_score = self.calculate_final_score(
                            round1_result, round2_result, None
                        )

                        return FullGauntletResult(
                            solution=solution,
                            problem=problem,
                            domain=domain,
                            round1_result=round1_result,
                            round2_result=round2_result,
                            round3_result=None,
                            passed=False,
                            final_score=final_score,
                            rounds_completed=2,
                            termination_reason=termination_reason,
                            artifacts_from_all_rounds=all_artifacts,
                            total_time=time.time() - start_time,
                            comprehensive_report=self._generate_report(
                                round1_result, round2_result, None, termination_reason
                            )
                        )

            # Round 3: Gold Team Consensus
            if self.config.round3_enabled:
                logger.info("Executing Round 3: Gold Team Verification")
                round3_result = await self.run_round3(
                    solution,
                    problem,
                    domain,
                    prior_findings=self._collect_prior_findings(round2_result)
                )

                if self.config.aggregate_artifacts:
                    all_artifacts.extend(round3_result.artifacts)

            # Calculate final score
            final_score = self.calculate_final_score(
                round1_result, round2_result, round3_result
            )

            # Determine if passed
            passed = (
                (round1_result is None or round1_result.passed) and
                (round2_result is None or round2_result.passed) and
                (round3_result is None or round3_result.passed)
            )

            logger.info(f"Gauntlet complete: passed={passed}, score={final_score:.3f}")

            return FullGauntletResult(
                solution=solution,
                problem=problem,
                domain=domain,
                round1_result=round1_result,
                round2_result=round2_result,
                round3_result=round3_result,
                passed=passed,
                final_score=final_score,
                rounds_completed=3,
                termination_reason=None,
                artifacts_from_all_rounds=all_artifacts,
                total_time=time.time() - start_time,
                comprehensive_report=self._generate_report(
                    round1_result, round2_result, round3_result, None
                )
            )

        except Exception as e:
            logger.error(f"Gauntlet execution failed: {e}", exc_info=True)
            termination_reason = f"Execution error: {str(e)}"

            return FullGauntletResult(
                solution=solution,
                problem=problem,
                domain=domain,
                round1_result=round1_result,
                round2_result=round2_result,
                round3_result=round3_result,
                passed=False,
                final_score=0.0,
                rounds_completed=self._count_completed_rounds(
                    round1_result, round2_result, round3_result
                ),
                termination_reason=termination_reason,
                artifacts_from_all_rounds=all_artifacts,
                total_time=time.time() - start_time,
                comprehensive_report=f"ERROR: {termination_reason}"
            )

    async def run_gauntlet(
        self,
        solution: str,
        problem: str,
        domain: str
    ) -> FullGauntletResult:
        """Compatibility wrapper for full gauntlet execution."""
        return await self.run_full_gauntlet(solution=solution, problem=problem, domain=domain)

    async def evaluate(
        self,
        solution: str,
        problem: str,
        domain: str
    ) -> FullGauntletResult:
        """Alias for run_full_gauntlet to satisfy evaluator interface."""
        return await self.run_full_gauntlet(solution=solution, problem=problem, domain=domain)

    async def run_round1(
        self,
        solution: str,
        problem: str,
        domain: str
    ) -> Round1Result:
        """
        Execute Round 1: LoongFlow AI evaluation.

        Args:
            solution: Solution to evaluate
            problem: Problem statement
            domain: Problem domain

        Returns:
            Round1Result with evaluation outcome
        """
        start_time = time.time()

        try:
            if self.round1_evaluator is None:
                # Fallback if evaluator not initialized
                return Round1Result(
                    passed=False,
                    score=0.0,
                    confidence=0.0,
                    evaluation_time=0.0,
                    feedback="Round 1 evaluator not initialized",
                    evaluator_type="error"
                )

            if not hasattr(self.round1_evaluator, 'evaluate_round'):
                # In-library LoongFlow evaluator interface
                return await self._run_round1_inlib(solution, problem, domain, start_time)

            # Create round rule
            round_rule = type('RoundRule', (), {
                'rule_id': 'round1_loongflow',
                'min_overall_confidence': self.config.round1_threshold,
                'per_judge_requirements': {"rule_id": "round1_loongflow"}
            })()

            # Create solution wrapper
            solution_wrapper = type('Solution', (), {
                'solution_content': solution,
                'content': solution
            })()

            # Create context
            context = {
                'problem': problem,
                'domain': domain,
                'criteria': ['correctness', 'quality', 'completeness'],
                'trace_id': f'gauntlet_r1_{int(time.time())}'
            }

            # Run evaluation
            logger.info("Starting LoongFlow evaluation")
            result = await self.round1_evaluator.evaluate_round(
                solution=solution_wrapper,
                round_rule=round_rule,
                context=context
            )

            return Round1Result(
                passed=result.passed,
                score=result.score,
                confidence=result.details.get('metrics', {}).get('confidence', 0.7),
                evaluation_time=time.time() - start_time,
                feedback=result.feedback,
                artifacts=[result.details],
                evaluator_type=result.details.get('evaluation_type', 'unknown')
            )

        except Exception as e:
            logger.error(f"Round 1 execution failed: {e}", exc_info=True)
            return Round1Result(
                passed=False,
                score=0.0,
                confidence=0.0,
                evaluation_time=time.time() - start_time,
                feedback=f"Round 1 failed: {str(e)}",
                evaluator_type="error"
            )

    async def _run_round1_inlib(
        self,
        solution: str,
        problem: str,
        domain: str,
        start_time: float
    ) -> Round1Result:
        """
        Run Round 1 with the in-library LoongFlow evaluator.

        When LoongFlow itself is unavailable the evaluator runs in a degraded
        mode; its verdict is then treated as advisory so a degraded quick screen
        cannot gate the Red Team and Gold Team rounds.

        Args:
            solution: Solution to evaluate
            problem: Problem statement
            domain: Problem domain
            start_time: Round start timestamp

        Returns:
            Round1Result with evaluation outcome
        """
        result = await self.round1_evaluator.evaluate(
            solution=solution,
            problem=problem,
            domain=domain
        )

        degraded = not self.round1_evaluator.is_available()
        feedback = result.feedback
        if degraded:
            feedback = f"[advisory: LoongFlow degraded]\n{feedback}"

        return Round1Result(
            passed=True if degraded else result.passed,
            score=result.overall_score,
            confidence=result.confidence,
            evaluation_time=time.time() - start_time,
            feedback=feedback,
            artifacts=[result.artifacts],
            evaluator_type="loongflow_fallback" if degraded else "loongflow"
        )

    async def run_round2(
        self,
        solution: str,
        problem: str,
        domain: str,
        language: str = "python"
    ) -> Round2Result:
        """
        Execute Round 2: Red Team adversarial evaluation.

        Combines two real signals:
        1. Deterministic adversarial probes (:func:`probe_solution`) that attack
           the candidate locally and report which attacks succeeded.
        2. An LLM judge (:class:`GauntletJudge`) prompted to critique and
           adversarially test the candidate. With no API key configured this runs
           on the offline mock backend and returns a deterministic verdict.

        Args:
            solution: Solution to evaluate
            problem: Problem statement
            domain: Problem domain
            language: Language of the candidate solution

        Returns:
            Round2Result with adversarial testing outcome
        """
        start_time = time.time()

        try:
            logger.info("Running Red Team adversarial evaluation")

            # 1. Deterministic attack vectors
            probes = probe_solution(solution, language=language)
            static_robustness = robustness_from_probes(probes)
            probe_findings = describe_attacks(probes)

            attack_details: List[Dict[str, Any]] = list(probes)
            attacks_attempted = len(probes)
            attacks_successful = len(successful_attacks(probes))

            # 2. LLM judge critique (offline mock when no model is configured)
            judge_score = None
            judge_feedback = "Red Team judge unavailable"
            judge_findings: List[str] = []

            if self.round2_evaluator is not None:
                verdict = await self.round2_evaluator.red_team(
                    solution=solution,
                    problem=problem,
                    domain=domain,
                    language=language,
                    prior_findings=probe_findings
                )
                judge_feedback = verdict.feedback
                judge_findings = verdict.findings

                if verdict.parsed:
                    judge_score = verdict.score

                # Every issue the judge reports is an additional attack that landed
                for finding in judge_findings:
                    attack_details.append({
                        "name": "llm_red_team_finding",
                        "description": finding,
                        "severity": "medium",
                        "successful": True,
                        "evidence": verdict.model
                    })
                attacks_attempted += max(1, len(judge_findings))
                attacks_successful += len(judge_findings)

            # 3. Blend the two signals: robustness_score stays the deterministic
            # probe outcome, the round score also reflects the judge verdict
            robustness_score = static_robustness
            if judge_score is None:
                score = static_robustness
            else:
                score = STATIC_WEIGHT * static_robustness + JUDGE_WEIGHT * judge_score

            passed = score >= self.config.round2_threshold

            feedback = (
                f"Red Team evaluation complete. "
                f"{attacks_attempted} attacks attempted, "
                f"{attacks_successful} successful. "
                f"Static robustness: {static_robustness:.2f}, "
                f"judge score: {'n/a' if judge_score is None else f'{judge_score:.2f}'}. "
                f"Robustness score: {robustness_score:.2f}. "
                f"Judge: {judge_feedback}"
            )

            return Round2Result(
                passed=passed,
                score=score,
                attacks_attempted=attacks_attempted,
                attacks_successful=attacks_successful,
                robustness_score=robustness_score,
                evaluation_time=time.time() - start_time,
                feedback=feedback,
                artifacts=[{
                    "static_robustness": static_robustness,
                    "judge_score": judge_score,
                    "probe_findings": probe_findings,
                    "judge_findings": judge_findings
                }],
                attack_details=attack_details
            )

        except Exception as e:
            logger.error(f"Round 2 execution failed: {e}", exc_info=True)
            return Round2Result(
                passed=False,
                score=0.0,
                attacks_attempted=0,
                attacks_successful=0,
                robustness_score=0.0,
                evaluation_time=time.time() - start_time,
                feedback=f"Round 2 failed: {str(e)}"
            )

    async def run_round3(
        self,
        solution: str,
        problem: str,
        domain: str,
        language: str = "python",
        prior_findings: Optional[List[str]] = None
    ) -> Round3Result:
        """
        Execute Round 3: Gold Team consensus verification.

        Every judge model in the ensemble votes on certification, and the
        candidate is statically verified (:func:`verify_solution`) in place of the
        Lean 4 proof, which is unavailable offline.

        Args:
            solution: Solution to evaluate
            problem: Problem statement
            domain: Problem domain
            language: Language of the candidate solution
            prior_findings: Findings carried over from earlier rounds

        Returns:
            Round3Result with consensus verification outcome
        """
        start_time = time.time()

        try:
            logger.info("Running Gold Team consensus verification")

            # 1. Static verification stands in for formal (Lean 4) verification
            verification = verify_solution(solution, language=language)
            formal_verification_passed = bool(verification["passed"])

            # 2. Collect one certification vote per judge model
            votes = []
            if self.round3_evaluator is not None:
                votes = await self.round3_evaluator.gold_team(
                    solution=solution,
                    problem=problem,
                    domain=domain,
                    language=language,
                    prior_findings=prior_findings
                )

            usable_votes = [vote for vote in votes if vote.parsed]

            if usable_votes:
                judge_score = sum(vote.score for vote in usable_votes) / len(usable_votes)
                agreement = consensus_score(usable_votes)
            else:
                # No judge available: fall back to the verification checks alone
                checks = verification["checks"]
                judge_score = (
                    sum(1.0 for passed in checks.values() if passed) / len(checks)
                    if checks else 0.0
                )
                agreement = judge_score

            consensus = agreement
            score = (
                (1.0 - VERIFICATION_WEIGHT) * judge_score
                + VERIFICATION_WEIGHT * (1.0 if formal_verification_passed else 0.0)
            )

            passed = score >= self.config.round3_threshold

            feedback = (
                f"Gold Team consensus: {consensus:.2f} across {len(usable_votes)} judge(s). "
                f"Judge score: {judge_score:.2f}. "
                f"Static verification: {'PASSED' if formal_verification_passed else 'FAILED'} "
                f"({verification['detail']}). "
                f"Final score: {score:.2f}"
            )

            return Round3Result(
                passed=passed,
                score=score,
                consensus_score=consensus,
                formal_verification_passed=formal_verification_passed,
                evaluation_time=time.time() - start_time,
                feedback=feedback,
                artifacts=[verification],
                evaluator_votes=[vote.to_dict() for vote in votes]
            )

        except Exception as e:
            logger.error(f"Round 3 execution failed: {e}", exc_info=True)
            return Round3Result(
                passed=False,
                score=0.0,
                consensus_score=0.0,
                formal_verification_passed=False,
                evaluation_time=time.time() - start_time,
                feedback=f"Round 3 failed: {str(e)}"
            )

    @staticmethod
    def _collect_prior_findings(round2_result: Optional[Round2Result]) -> List[str]:
        """
        Collect the issues Round 2 found so Round 3 can verify against them.

        Args:
            round2_result: Round 2 result (may be None)

        Returns:
            List of short finding descriptions
        """
        if round2_result is None:
            return []

        findings = []
        for attack in round2_result.attack_details:
            if not attack.get("successful"):
                continue
            detail = attack.get("evidence") or attack.get("description", "")
            findings.append(f"[{attack.get('severity', 'medium')}] {attack.get('name')}: {detail}")
        return findings

    def should_continue_to_round2(self, round1_result: Round1Result) -> bool:
        """
        Determine if solution should proceed to Round 2.

        Args:
            round1_result: Round 1 evaluation result

        Returns:
            True if solution should proceed to Round 2
        """
        if not self.config.enable_early_termination:
            return True  # Always continue if early termination disabled

        if round1_result.evaluator_type in ("loongflow_fallback", "error"):
            # A degraded/unavailable quick screen is advisory only: it must not
            # gate the Red Team and Gold Team rounds, which do evaluate for real
            logger.info(
                "Round 1 verdict is advisory "
                f"(evaluator={round1_result.evaluator_type}); continuing to Round 2"
            )
            return True

        return round1_result.passed and round1_result.score >= self.config.round1_threshold

    def should_continue_to_round3(self, round2_result: Round2Result) -> bool:
        """
        Determine if solution should proceed to Round 3.

        Args:
            round2_result: Round 2 evaluation result

        Returns:
            True if solution should proceed to Round 3
        """
        if not self.config.enable_early_termination:
            return True  # Always continue if early termination disabled

        return round2_result.passed and round2_result.score >= self.config.round2_threshold

    def calculate_final_score(
        self,
        round1: Optional[Round1Result],
        round2: Optional[Round2Result],
        round3: Optional[Round3Result]
    ) -> float:
        """
        Calculate weighted aggregate final score.

        Only the rounds that actually produced a result contribute, and the
        weights are renormalized over them, so skipping or terminating a round
        does not silently penalize the score.

        Args:
            round1: Round 1 result (may be None)
            round2: Round 2 result (may be None)
            round3: Round 3 result (may be None)

        Returns:
            Weighted aggregate score (0.0-1.0+)
        """
        weights = self.config

        contributions = []
        if round1 is not None:
            contributions.append((round1.score, weights.round1_weight))
        if round2 is not None:
            contributions.append((round2.score, weights.round2_weight))
        if round3 is not None:
            contributions.append((round3.score, weights.round3_weight))

        total_weight = sum(weight for _, weight in contributions)
        if not contributions or total_weight <= 0:
            return 0.0

        return sum(score * weight for score, weight in contributions) / total_weight

    def generate_comprehensive_report(self, full_result: FullGauntletResult) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            full_result: Complete gauntlet result

        Returns:
            Comprehensive report text
        """
        if self.config.generate_detailed_report:
            return full_result.comprehensive_report
        return self._generate_summary_report(full_result)

    def _generate_report(
        self,
        r1: Optional[Round1Result],
        r2: Optional[Round2Result],
        r3: Optional[Round3Result],
        termination: Optional[str]
    ) -> str:
        """Generate detailed report"""
        lines = [
            "=" * 80,
            "3-ROUND GAUNTLET EVALUATION REPORT",
            "=" * 80,
            f"Timestamp: {datetime.now(UTC).isoformat()}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 80,
        ]

        if termination:
            lines.append(f"Status: TERMINATED EARLY")
            lines.append(f"Reason: {termination}")
        else:
            lines.append(f"Status: COMPLETE")

        lines.extend([
            "",
            "ROUND RESULTS",
            "-" * 80
        ])

        # Round 1
        if r1:
            lines.extend([
                f"Round 1 (LoongFlow AI): {'PASSED' if r1.passed else 'FAILED'}",
                f"  Score: {r1.score:.3f}",
                f"  Confidence: {r1.confidence:.3f}",
                f"  Time: {r1.evaluation_time:.2f}s",
                f"  Evaluator: {r1.evaluator_type}",
                f"  Feedback: {r1.feedback[:100]}...",
                ""
            ])

        # Round 2
        if r2:
            lines.extend([
                f"Round 2 (Red Team): {'PASSED' if r2.passed else 'FAILED'}",
                f"  Score: {r2.score:.3f}",
                f"  Robustness: {r2.robustness_score:.3f}",
                f"  Attacks: {r2.attacks_successful}/{r2.attacks_attempted} successful",
                f"  Time: {r2.evaluation_time:.2f}s",
                f"  Feedback: {r2.feedback[:100]}...",
                ""
            ])

        # Round 3
        if r3:
            lines.extend([
                f"Round 3 (Gold Team): {'PASSED' if r3.passed else 'FAILED'}",
                f"  Score: {r3.score:.3f}",
                f"  Consensus: {r3.consensus_score:.3f}",
                f"  Formal Verification: {'PASSED' if r3.formal_verification_passed else 'N/A'}",
                f"  Time: {r3.evaluation_time:.2f}s",
                f"  Feedback: {r3.feedback[:100]}...",
                ""
            ])

        lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])

        return "\n".join(lines)

    def _generate_summary_report(self, full_result: FullGauntletResult) -> str:
        """Generate brief summary report"""
        return (
            f"Gauntlet: {full_result.rounds_completed} rounds, "
            f"score={full_result.final_score:.3f}, "
            f"passed={full_result.passed}"
        )

    def _count_completed_rounds(
        self,
        r1: Optional[Round1Result],
        r2: Optional[Round2Result],
        r3: Optional[Round3Result]
    ) -> int:
        """Count how many rounds were completed"""
        count = 0
        if r1 is not None:
            count += 1
        if r2 is not None:
            count += 1
        if r3 is not None:
            count += 1
        return count


# Factory functions for common configurations

def create_strict_config() -> ThreeRoundConfig:
    """
    Create strict configuration for high-stakes domains.

    Higher thresholds for all rounds, ensuring only high-quality
    solutions pass through.

    Returns:
        ThreeRoundConfig with strict settings
    """
    return ThreeRoundConfig(
        round1_threshold=0.7,
        round2_threshold=0.8,
        round3_threshold=0.9,
        enable_early_termination=True
    )


def create_lenient_config() -> ThreeRoundConfig:
    """
    Create lenient configuration for exploration.

    Lower thresholds, always runs all rounds for learning
    even if early rounds fail.

    Returns:
        ThreeRoundConfig with lenient settings
    """
    return ThreeRoundConfig(
        round1_threshold=0.3,
        round2_threshold=0.5,
        round3_threshold=0.6,
        enable_early_termination=False
    )


def create_balanced_config() -> ThreeRoundConfig:
    """
    Create balanced configuration for general use.

    Default thresholds with early termination enabled.

    Returns:
        ThreeRoundConfig with balanced settings
    """
    return ThreeRoundConfig()


def create_domain_config(domain: str) -> ThreeRoundConfig:
    """
    Create domain-specific configuration.

    Args:
        domain: Domain name (finance, science, engineering, web, etc.)

    Returns:
        ThreeRoundConfig tuned for domain
    """
    domain_configs = {
        'finance': create_strict_config(),  # High stakes
        'trading': create_strict_config(),
        'science': ThreeRoundConfig(
            round1_threshold=0.5,
            round2_threshold=0.6,
            round3_threshold=0.7
        ),  # Moderate
        'engineering': create_strict_config(),  # Safety-critical
        'pharma': create_strict_config(),  # Safety-critical
        'web': create_lenient_config(),  # Low risk
        'general': create_balanced_config()
    }

    return domain_configs.get(domain.lower(), create_balanced_config())
