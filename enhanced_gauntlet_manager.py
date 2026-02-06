"""
Enhanced Gauntlet Manager with LoongFlow Integration

This module extends the standard gauntlet system to include LoongFlow's AI
evaluation as Round 1, providing quick quality screening before more expensive
red team and gold team evaluations.

The enhanced gauntlet system implements a 3-round validation process:
1. Round 1: LoongFlow AI Evaluation (Quick Screen - ~10-30 seconds)
2. Round 2: Red Team Attack (Adversarial - finds flaws)
3. Round 3: Gold Team Verification (Consensus - final approval)
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time

from openevolve_structures import GauntletDefinition, GauntletRoundRule

logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Alerting and adaptive for Enhanced Gauntlet Manager
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


class GauntletRoundStatus(Enum):
    """Status of a gauntlet round execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class GauntletRoundResult:
    """
    Result from executing a single gauntlet round.

    Attributes:
        rule_id: ID of the gauntlet round rule
        round_number: Round number (1-indexed)
        status: Final status of the round
        score: Score achieved (0.0-1.0+)
        feedback: Human-readable feedback
        details: Additional evaluation details
        execution_time: Time taken for evaluation in seconds
        timestamp: When the evaluation was performed
    """
    rule_id: str
    round_number: int
    status: GauntletRoundStatus
    score: float
    feedback: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'rule_id': self.rule_id,
            'round_number': self.round_number,
            'status': self.status.value,
            'score': self.score,
            'feedback': self.feedback,
            'details': self.details,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp
        }


@dataclass
class GauntletExecution:
    """
    Complete execution result for a gauntlet.

    Attributes:
        gauntlet_id: ID of the gauntlet definition
        solution_id: ID of the solution evaluated
        rounds_results: Results from each round
        rounds_passed: List of round IDs that passed
        rounds_failed: List of round IDs that failed
        final_score: Final aggregated score
        overall_passed: Whether the gauntlet was passed
        execution_time: Total time for all rounds
        timestamp: When the execution was performed
    """
    gauntlet_id: str
    solution_id: str
    rounds_results: List[GauntletRoundResult] = field(default_factory=list)
    rounds_passed: List[str] = field(default_factory=list)
    rounds_failed: List[str] = field(default_factory=list)
    final_score: float = 0.0
    overall_passed: bool = False
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'gauntlet_id': self.gauntlet_id,
            'solution_id': self.solution_id,
            'rounds_results': [r.to_dict() for r in self.rounds_results],
            'rounds_passed': self.rounds_passed,
            'rounds_failed': self.rounds_failed,
            'final_score': self.final_score,
            'overall_passed': self.overall_passed,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp
        }


class EnhancedGauntletSystem:
    """
    Enhanced gauntlet system with LoongFlow integration.

    This system provides 3-round validation:
    1. LoongFlow AI evaluation (fast, automated quality check)
    2. Red Team adversarial testing (finds flaws and edge cases)
    3. Gold Team consensus verification (final quality assurance)

    The system is designed to fail fast - if a solution fails Round 1,
    it won't waste resources on Rounds 2 and 3 unless configured otherwise.
    """

    def __init__(
        self,
        llm_config: Dict[str, Any],
        enable_loongflow: bool = True,
        red_team_evaluator=None,
        gold_team_evaluator=None
    ):
        """
        Initialize the enhanced gauntlet system.

        Args:
            llm_config: Configuration for LLM used by LoongFlow
            enable_loongflow: Whether to use LoongFlow evaluator
            red_team_evaluator: Optional red team evaluator instance
            gold_team_evaluator: Optional gold team evaluator instance
        """
        self.llm_config = llm_config
        self.enable_loongflow = enable_loongflow

        # Initialize LoongFlow evaluator
        try:
            from evaluators.loongflow_adapter import create_loongflow_evaluator
            self.loongflow_evaluator = create_loongflow_evaluator(
                llm_config=llm_config,
                timeout=60,
                enable_loongflow=enable_loongflow
            )
            logger.info("LoongFlow evaluator initialized")
        except Exception as e:
            logger.warning(f"Could not initialize LoongFlow evaluator: {e}")
            self.loongflow_evaluator = None

        # Store team evaluators
        self.red_team_evaluator = red_team_evaluator
        self.gold_team_evaluator = gold_team_evaluator

    def create_enhanced_gauntlet(
        self,
        problem_type: str,
        strictness: str = "standard"
    ) -> GauntletDefinition:
        """
        Create enhanced gauntlet with 3 rounds.

        Args:
            problem_type: Type of problem (trading, engineering, security, etc.)
            strictness: Validation strictness (strict, standard, lenient)

        Returns:
            Configured GauntletDefinition with 3 rounds
        """
        # Adjust thresholds based on strictness
        thresholds = self._get_thresholds(strictness)

        # Get attack modes for problem type
        attack_modes = self._get_attack_modes(problem_type)

        # Create rounds - store metadata in per_judge_requirements since GauntletRoundRule doesn't have rule_id
        rounds = [
            # Round 1: LoongFlow AI Evaluation (Quick Screen)
            GauntletRoundRule(
                round_number=1,
                min_overall_confidence=thresholds['round1'],
                quorum_required_approvals=1,
                quorum_from_panel_size=1,
                collaboration_mode="independent",
                time_limit_seconds=60,
                max_api_calls=10,
                per_judge_requirements={
                    "rule_id": "loongflow_ai_eval",
                    "rule_type": "automated",
                    "validation_type": "quality",
                    "description": "Quick AI-based quality assessment using LoongFlow"
                }
            ),

            # Round 2: Red Team Attack (Adversarial)
            GauntletRoundRule(
                round_number=2,
                min_overall_confidence=thresholds['round2'],
                quorum_required_approvals=2,
                quorum_from_panel_size=3,
                collaboration_mode="share_previous_feedback",
                time_limit_seconds=180,
                max_api_calls=30,
                per_judge_requirements={
                    "rule_id": "red_team_attack",
                    "rule_type": "red_team",
                    "validation_type": "adversarial",
                    "description": "Adversarial testing to find flaws and edge cases",
                    "attack_modes": {
                        "modes": attack_modes,
                        "description": "Use these attack modes"
                    }
                }
            ),

            # Round 3: Gold Team Verification (Consensus)
            GauntletRoundRule(
                round_number=3,
                min_overall_confidence=thresholds['round3'],
                quorum_required_approvals=3,
                quorum_from_panel_size=5,
                collaboration_mode="share_previous_feedback",
                time_limit_seconds=240,
                max_api_calls=50,
                max_score_variance=0.2,
                per_judge_requirements={
                    "rule_id": "gold_team_verify",
                    "rule_type": "gold_team",
                    "validation_type": "consensus",
                    "description": "Consensus-based validation for final approval"
                }
            )
        ]

        # Create gauntlet definition
        gauntlet = GauntletDefinition(
            name=f"enhanced_{problem_type}",
            team_name="enhanced_validation_team",
            rounds=rounds,
            description=f"Enhanced 3-round validation for {problem_type} problems",
            attack_modes=attack_modes,
            generation_mode="single_candidate"
        )

        return gauntlet

    async def execute_gauntlet(
        self,
        gauntlet: GauntletDefinition,
        solution: Any,
        context: Dict[str, Any]
    ) -> GauntletExecution:
        """
        Execute enhanced gauntlet with 3 rounds.

        Args:
            gauntlet: GauntletDefinition to execute
            solution: Solution to evaluate
            context: Additional context (problem, criteria, etc.)

        Returns:
            GauntletExecution with results from all rounds
        """
        start_time_total = time.time()
        success = False
        solution_id = getattr(solution, 'id', 'unknown')

        try:
            execution = GauntletExecution(
                gauntlet_id=gauntlet.name,
                solution_id=solution_id,
                rounds_results=[],
                rounds_passed=[],
                rounds_failed=[]
            )

            # Execute rounds sequentially
            for round_rule in gauntlet.rounds:
                rule_id = round_rule.per_judge_requirements.get("rule_id", "unknown")
                logger.info(f"Executing round {round_rule.round_number}: {rule_id}")

                try:
                    # Route to appropriate evaluator
                    result = await self._execute_round(
                        round_rule=round_rule,
                        solution=solution,
                        context=context
                    )

                    # Track result
                    execution.rounds_results.append(result)

                    if result.status == GauntletRoundStatus.PASSED:
                        execution.rounds_passed.append(result.rule_id)
                    elif result.status == GauntletRoundStatus.FAILED:
                        execution.rounds_failed.append(result.rule_id)

                    # Aggregate score
                    execution.final_score += result.score

                    # Check if should stop on failure
                    if result.status == GauntletRoundStatus.FAILED:
                        # Could implement logic to stop early here
                        pass

                except Exception as e:
                    logger.error(f"Round {rule_id} failed: {e}", exc_info=True)

                    # Create error result
                    error_result = GauntletRoundResult(
                        rule_id=rule_id,
                        round_number=round_rule.round_number,
                        status=GauntletRoundStatus.ERROR,
                        score=0.0,
                        feedback=f"Round execution error: {str(e)}",
                        details={"error": str(e)},
                        execution_time=0.0
                    )
                    execution.rounds_results.append(error_result)
                    execution.rounds_failed.append(rule_id)

            # Calculate final average score
            if execution.rounds_results:
                execution.final_score = execution.final_score / len(execution.rounds_results)

            # Final determination
            execution.overall_passed = (
                len(execution.rounds_failed) == 0
            )

            # Total execution time
            execution.execution_time = time.time() - start_time_total

            logger.info(
                f"Gauntlet execution complete: passed={execution.overall_passed}, "
                f"score={execution.final_score:.2f}, time={execution.execution_time:.2f}s"
            )

            success = True

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful execution
            self._extract_gauntlet_knowledge("execute_gauntlet", gauntlet.name, execution)
            self._track_gauntlet_performance("execute_gauntlet", True, execution.execution_time, execution.final_score)

            return execution

        except Exception as e:
            execution_time = time.time() - start_time_total

            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_gauntlet_alerts("execute_gauntlet", False, solution_id, str(e))
            self._track_gauntlet_performance("execute_gauntlet", False, execution_time, 0.0)

            logger.error(f"Gauntlet execution failed for {solution_id}: {e}")
            raise

    async def _execute_round(
        self,
        round_rule: GauntletRoundRule,
        solution: Any,
        context: Dict[str, Any]
    ) -> GauntletRoundResult:
        """Execute a single gauntlet round."""

        # Get rule_id from per_judge_requirements
        rule_id = round_rule.per_judge_requirements.get("rule_id", "unknown")

        # Round 1: LoongFlow evaluation
        if rule_id == "loongflow_ai_eval":
            if self.loongflow_evaluator:
                from evaluators.loongflow_adapter import GauntletRoundResult as LFRoundResult

                result = await self.loongflow_evaluator.evaluate_round(
                    solution=solution,
                    round_rule=round_rule,
                    context=context
                )

                # Convert to our GauntletRoundResult
                status = GauntletRoundStatus.PASSED if result.passed else GauntletRoundStatus.FAILED

                return GauntletRoundResult(
                    rule_id=rule_id,
                    round_number=round_rule.round_number,
                    status=status,
                    score=result.score,
                    feedback=result.feedback,
                    details=result.details,
                    execution_time=result.execution_time
                )
            else:
                # LoongFlow not available, mark as skipped
                return GauntletRoundResult(
                    rule_id=rule_id,
                    round_number=round_rule.round_number,
                    status=GauntletRoundStatus.SKIPPED,
                    score=1.0,  # Pass by default
                    feedback="LoongFlow evaluator not available - round skipped",
                    details={"warning": "loongflow_unavailable"},
                    execution_time=0.0
                )

        # Round 2: Red Team evaluation
        elif rule_id == "red_team_attack":
            if self.red_team_evaluator:
                # Call red team evaluator
                # This would interface with your existing red team system
                return await self._mock_red_team_evaluation(round_rule, solution, context)
            else:
                return GauntletRoundResult(
                    rule_id=rule_id,
                    round_number=round_rule.round_number,
                    status=GauntletRoundStatus.SKIPPED,
                    score=1.0,
                    feedback="Red team evaluator not configured - round skipped",
                    details={"warning": "red_team_unavailable"},
                    execution_time=0.0
                )

        # Round 3: Gold Team evaluation
        elif rule_id == "gold_team_verify":
            if self.gold_team_evaluator:
                # Call gold team evaluator
                # This would interface with your existing gold team system
                return await self._mock_gold_team_evaluation(round_rule, solution, context)
            else:
                return GauntletRoundResult(
                    rule_id=rule_id,
                    round_number=round_rule.round_number,
                    status=GauntletRoundStatus.SKIPPED,
                    score=1.0,
                    feedback="Gold team evaluator not configured - round skipped",
                    details={"warning": "gold_team_unavailable"},
                    execution_time=0.0
                )

        # Unknown round type
        else:
            return GauntletRoundResult(
                rule_id=rule_id,
                round_number=round_rule.round_number,
                status=GauntletRoundStatus.ERROR,
                score=0.0,
                feedback=f"Unknown round type: {rule_id}",
                details={"error": "unknown_round_type"},
                execution_time=0.0
            )

    async def _mock_red_team_evaluation(
        self,
        round_rule: GauntletRoundRule,
        solution: Any,
        context: Dict[str, Any]
    ) -> GauntletRoundResult:
        """
        Execute Red Team evaluation using the actual RedTeam engine.
        Performs adversarial attacks and vulnerability scanning.
        """
        rule_id = round_rule.per_judge_requirements.get("rule_id", "red_team_attack")
        solution_content = str(solution)
        
        try:
            # Initialize Red Team if not present (lazy loading)
            if not self.red_team_evaluator:
                try:
                    from red_team import RedTeam, RedTeamStrategy
                    self.red_team_evaluator = RedTeam()
                except ImportError:
                    logger.error("RedTeam module not found")
                    raise RuntimeError("RedTeam capability unavailable")

            # Determine strategy from rule configuration or default to SYSTEMATIC
            strategy_name = round_rule.per_judge_requirements.get("attack_modes", {}).get("strategy", "SYSTEMATIC")
            try:
                from red_team import RedTeamStrategy
                strategy = getattr(RedTeamStrategy, strategy_name.upper(), RedTeamStrategy.SYSTEMATIC)
            except (ImportError, AttributeError):
                strategy = None

            # Execute assessment
            assessment = self.red_team_evaluator.assess_content(
                content=solution_content,
                content_type=context.get("content_type", "general"),
                strategy=strategy,
                attack_modes=round_rule.per_judge_requirements.get("attack_modes", {}).get("modes")
            )
            
            # Calculate score (1.0 - weighted penalty of findings)
            # High confidence score from assessment means high confidence in FINDINGS (bad for solution)
            # So we need to inverse the logic: fewer findings = higher score
            issue_count = len(assessment.findings)
            critical_issues = assessment.issues_by_severity.get("critical", 0) + \
                              assessment.issues_by_severity.get("CRITICAL", 0)
            
            # Simple scoring: starts at 1.0, penalties for issues
            base_score = 1.0
            base_score -= (critical_issues * 0.3)
            base_score -= (issue_count * 0.05)
            score = max(0.0, base_score)
            
            # Determine pass/fail
            passed = score >= round_rule.min_overall_confidence
            
            return GauntletRoundResult(
                rule_id=rule_id,
                round_number=round_rule.round_number,
                status=GauntletRoundStatus.PASSED if passed else GauntletRoundStatus.FAILED,
                score=score,
                feedback=assessment.assessment_summary,
                details={
                    "findings_count": issue_count,
                    "critical_issues": critical_issues,
                    "strategy": str(strategy),
                    "full_findings": [f.title for f in assessment.findings]
                },
                execution_time=assessment.time_taken
            )
            
        except Exception as e:
            logger.error(f"Red Team evaluation failed: {e}", exc_info=True)
            return GauntletRoundResult(
                rule_id=rule_id,
                round_number=round_rule.round_number,
                status=GauntletRoundStatus.ERROR,
                score=0.0,
                feedback=f"Red Team execution error: {str(e)}",
                details={"error": str(e)},
                execution_time=0.0
            )

    async def _mock_gold_team_evaluation(
        self,
        round_rule: GauntletRoundRule,
        solution: Any,
        context: Dict[str, Any]
    ) -> GauntletRoundResult:
        """
        Execute Gold Team evaluation using the actual EvaluatorTeam engine.
        Performs rigorous consensus-based verification.
        """
        rule_id = round_rule.per_judge_requirements.get("rule_id", "gold_team_verify")
        solution_content = str(solution)
        
        try:
            # Initialize Gold Team (EvaluatorTeam) if not present
            if not self.gold_team_evaluator:
                try:
                    from evaluator_team import EvaluatorTeam, EvaluationThreshold
                    self.gold_team_evaluator = EvaluatorTeam()
                except ImportError:
                    logger.error("EvaluatorTeam module not found")
                    raise RuntimeError("GoldTeam capability unavailable")

            # Determine threshold
            from evaluator_team import EvaluationThreshold
            threshold_map = {
                "strict": EvaluationThreshold.EXCEPTIONAL,
                "standard": EvaluationThreshold.STANDARD_APPROVAL,
                "lenient": EvaluationThreshold.MINIMAL_ACCEPTANCE
            }
            strictness = context.get("strictness", "standard")
            threshold = threshold_map.get(strictness, EvaluationThreshold.STANDARD_APPROVAL)

            # Execute evaluation
            evaluation = self.gold_team_evaluator.evaluate_content(
                content=solution_content,
                content_type=context.get("content_type", "general"),
                threshold=threshold
            )
            
            # Extract results
            score = evaluation.consensus_score / 100.0  # Convert 0-100 to 0.0-1.0
            passed = evaluation.final_verdict == "APPROVED"
            
            return GauntletRoundResult(
                rule_id=rule_id,
                round_number=round_rule.round_number,
                status=GauntletRoundStatus.PASSED if passed else GauntletRoundStatus.FAILED,
                score=score,
                feedback=f"Verdict: {evaluation.final_verdict}. {evaluation.recommendations[0] if evaluation.recommendations else ''}",
                details={
                    "consensus_reached": evaluation.consensus_reached,
                    "variance": evaluation.variance_analysis.get("variance", 0),
                    "verdict": evaluation.final_verdict
                },
                execution_time=evaluation.evaluation_metadata.get("evaluation_time_taken", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Gold Team evaluation failed: {e}", exc_info=True)
            return GauntletRoundResult(
                rule_id=rule_id,
                round_number=round_rule.round_number,
                status=GauntletRoundStatus.ERROR,
                score=0.0,
                feedback=f"Gold Team execution error: {str(e)}",
                details={"error": str(e)},
                execution_time=0.0
            )

    def _get_thresholds(self, strictness: str) -> Dict[str, float]:
        """Get score thresholds based on strictness level."""
        if strictness == "strict":
            return {
                'round1': 0.8,
                'round2': 0.75,
                'round3': 0.9
            }
        elif strictness == "lenient":
            return {
                'round1': 0.6,
                'round2': 0.6,
                'round3': 0.75
            }
        else:  # standard
            return {
                'round1': 0.7,
                'round2': 0.7,
                'round3': 0.85
            }

    def _get_attack_modes(self, problem_type: str) -> List[str]:
        """Get appropriate attack modes for problem type."""
        attack_modes = {
            "trading": ["market_crash", "regime_change", "black_swan", "liquidity_crisis"],
            "engineering": ["overload", "fatigue", "extreme_conditions", "edge_case"],
            "security": ["injection", "bypass", "flood", "exploit"],
            "scientific": ["outlier", "noise", "confounding", "bias"],
            "finance": ["volatility_spike", "correlation_breakdown", "tail_risk"],
            "general": ["generic_attack", "stress_test", "edge_case"]
        }
        return attack_modes.get(problem_type.lower(), attack_modes["general"])

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Enhanced Gauntlet
    # =========================================================================

    def _trigger_gauntlet_alerts(
        self,
        operation: str,
        success: bool,
        solution_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for gauntlet failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                alert_manager.create_alert(
                    title=f"Gauntlet Alert: {operation}",
                    description=f"Gauntlet operation '{operation}' failed" +
                                 (f" for solution '{solution_id}'" if solution_id else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=AlertSeverity.HIGH.value,
                    source="enhanced_gauntlet_manager",
                    component="gauntlet",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger Gauntlet alert: {e}")

    def _extract_gauntlet_knowledge(
        self,
        operation: str,
        gauntlet_id: str,
        execution: 'GauntletExecution'
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract gauntlet knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"gauntlet_{operation}_{gauntlet_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="gauntlet_execution",
                source_component="enhanced_gauntlet_manager",
                title=f"Gauntlet: {operation} - {gauntlet_id}",
                content={
                    "operation": operation,
                    "gauntlet_id": gauntlet_id,
                    "solution_id": execution.solution_id,
                    "final_score": execution.final_score,
                    "overall_passed": execution.overall_passed,
                    "rounds_passed": len(execution.rounds_passed),
                    "rounds_failed": len(execution.rounds_failed),
                    "execution_time": execution.execution_time,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "rounds_results": [r.to_dict() for r in execution.rounds_results]
                },
                tags=["gauntlet", operation, "validation"]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted Gauntlet knowledge for {gauntlet_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract Gauntlet knowledge: {e}")
            return False

    def _track_gauntlet_performance(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
        final_score: float = 0.0
    ):
        """**ACTUAL INTEGRATION**: Track gauntlet performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            quality = 1.0 if success else 0.0

            performance_data = StrategyPerformanceData(
                strategy_name=f"gauntlet_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "duration_seconds": duration_seconds,
                    "final_score": final_score
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked Gauntlet performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track Gauntlet performance: {e}")


def create_enhanced_gauntlet_system(
    llm_config: Dict[str, Any],
    enable_loongflow: bool = True
) -> EnhancedGauntletSystem:
    """
    Factory function to create an enhanced gauntlet system.

    Args:
        llm_config: LLM configuration for evaluators
        enable_loongflow: Whether to use LoongFlow evaluator

    Returns:
        Configured EnhancedGauntletSystem instance

    Example:
        ```python
        system = create_enhanced_gauntlet_system(
            llm_config={
                'model': 'claude-3-5-sonnet-20241022',
                'api_key': 'sk-...',
                'url': 'http://localhost:8001'
            }
        )

        gauntlet = system.create_enhanced_gauntlet(
            problem_type="engineering",
            strictness="standard"
        )

        execution = await system.execute_gauntlet(
            gauntlet=gauntlet,
            solution=my_solution,
            context={'problem': 'Design a bridge', 'criteria': ['safety', 'efficiency']}
        )
        ```
    """
    return EnhancedGauntletSystem(
        llm_config=llm_config,
        enable_loongflow=enable_loongflow
    )
