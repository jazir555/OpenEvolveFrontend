"""
Adaptive Gauntlet System

Implements gauntlet validation that adapts difficulty based on team performance,
problem complexity, and historical success rates.

Features:
- Dynamic threshold adjustment based on performance metrics
- Team-specific difficulty calibration
- Problem complexity awareness
- Learning from historical performance
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict

# ROMA-MDAP-MAKER (Robust Execution)
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeEngine,
        create_romamdapmaker_associative_config,
        ROMA_MDAP_MAKER_AVAILABLE
    )
    from roma_mdap_maker_reliability_ssot import get_validation_config
except ImportError:
    ROMA_MDAP_MAKER_AVAILABLE = False
    get_validation_config = None

from sovereign_data_models import (
    GauntletDefinition, GauntletRoundRule, ProblemDefinition, SubProblem,
    TeamPerformanceMetrics, ComplexityScore
)

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks historical performance for adaptive difficulty adjustment.
    """

    def __init__(self):
        """Initialize performance tracker."""
        # Team performance history
        self.team_performance: Dict[str, List[float]] = defaultdict(list)

        # Domain performance history
        self.domain_performance: Dict[str, List[float]] = defaultdict(list)

        # Problem type performance
        self.problem_type_performance: Dict[str, List[float]] = defaultdict(list)

        # Gauntlet effectiveness
        self.gauntlet_performance: Dict[str, List[float]] = defaultdict(list)

    def record_performance(
        self,
        team_id: str,
        domain: str,
        problem_type: str,
        gauntlet_name: str,
        score: float
    ):
        """
        Record a performance datapoint.

        Args:
            team_id: Team identifier
            domain: Problem domain
            problem_type: Type of problem
            gauntlet_name: Name of gauntlet used
            score: Performance score (0-1)
        """
        # Only record valid scores
        if not (0.0 <= score <= 1.0):
            logger.warning(f"Invalid score {score}, must be between 0.0 and 1.0")
            return

        self.team_performance[team_id].append(score)
        self.domain_performance[domain].append(score)
        self.problem_type_performance[problem_type].append(score)
        self.gauntlet_performance[gauntlet_name].append(score)

    def get_team_performance(
        self,
        team_id: str,
        window_size: int = 10
    ) -> Dict[str, float]:
        """
        Get team performance metrics.

        Args:
            team_id: Team identifier
            window_size: Number of recent performances to consider

        Returns:
            Dict with performance metrics
        """
        if team_id not in self.team_performance:
            return {
                "avg_score": 0.5,
                "recent_avg": 0.5,
                "trend": "stable",
                "total_attempts": 0
            }

        scores = self.team_performance[team_id]
        recent_scores = scores[-window_size:] if len(scores) >= window_size else scores

        avg_score = sum(scores) / len(scores) if scores else 0.5
        recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0.5

        # Determine trend
        if len(recent_scores) >= 3:
            first_half = recent_scores[:len(recent_scores)//2]
            second_half = recent_scores[len(recent_scores)//2:]

            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            if avg_second > avg_first * 1.1:
                trend = "improving"
            elif avg_second < avg_first * 0.9:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "avg_score": avg_score,
            "recent_avg": recent_avg,
            "trend": trend,
            "total_attempts": len(scores)
        }

    def get_domain_difficulty_multiplier(
        self,
        domain: str,
        problem_type: str
    ) -> float:
        """
        Get difficulty multiplier based on domain performance.

        Returns:
            Multiplier between 0.9 (harder) and 1.1 (easier)
        """
        domain_scores = self.domain_performance.get(domain, [])
        type_scores = self.problem_type_performance.get(problem_type, [])

        # If no data, return neutral multiplier
        if not domain_scores and not type_scores:
            return 1.0

        # Calculate average scores
        domain_avg = sum(domain_scores) / len(domain_scores) if domain_scores else 0.5
        type_avg = sum(type_scores) / len(type_scores) if type_scores else 0.5

        # Combined average
        combined_avg = (domain_avg + type_avg) / 2

        # Lower multiplier for high-performing domains (make it harder)
        # Higher multiplier for low-performing domains (make it easier)
        if combined_avg > 0.8:
            return 0.9  # High success, increase difficulty
        elif combined_avg > 0.6:
            return 0.95  # Good success, slightly increase difficulty
        elif combined_avg < 0.4:
            return 1.1  # Low success, decrease difficulty
        elif combined_avg < 0.6:
            return 1.05  # Moderate success, slightly decrease difficulty
        else:
            return 1.0  # Neutral


class AdaptiveGauntletSystem:
    """
    Gauntlet system that adapts difficulty based on performance.

    Features:
    - Adjusts validation thresholds based on historical performance
    - Makes gauntlets harder/easier based on success rates
    - Learns optimal difficulty for each team/problem type
    - Balances challenge with achievability
    """

    def __init__(self, performance_tracker: Optional[PerformanceTracker] = None):
        """
        Initialize with performance tracking for adaptation.

        Args:
            performance_tracker: Optional pre-populated performance tracker
        """
        self.performance_tracker = performance_tracker or PerformanceTracker()
        self.adaptation_history: List[Dict[str, Any]] = []

        # Initialize ROMA-MDAP-MAKER Engine for robust validation
        self.roma_engine = None
        if ROMA_MDAP_MAKER_AVAILABLE:
            try:
                # Use SSOT validation preset for standardized high-reliability config
                config = get_validation_config()
                self.roma_engine = ROMAMDAPMakerAssociativeEngine(config)
                logger.info("ROMAMDAPMakerAssociativeEngine initialized for AdaptiveGauntletSystem")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Failed to initialize ROMA engine: {e}")

    def create_adaptive_gauntlet(
        self,
        problem: ProblemDefinition,
        sub_problem: SubProblem,
        team_performance: Dict[str, TeamPerformanceMetrics]
    ) -> GauntletDefinition:
        """
        Create gauntlet with adaptive difficulty.

        Adaptation factors:
        - Team historical performance (better teams → harder gauntlets)
        - Problem complexity (harder problems → easier gauntlets initially)
        - Domain complexity (complex domains → adjusted thresholds)
        - Recent success rates (high success → increase difficulty)

        Args:
            problem: Parent problem definition
            sub_problem: Sub-problem to create gauntlet for
            team_performance: Performance metrics for teams

        Returns:
            Adapted GauntletDefinition
        """
        # Get team metrics (use solver team if available)
        solver_team = sub_problem.assigned_team or "default"
        team_metrics = team_performance.get(solver_team)

        # Calculate adaptive thresholds
        base_threshold = 0.7  # Base acceptance threshold

        # Adapt threshold based on multiple factors
        adapted_threshold = self.calculate_adaptive_threshold(
            base_threshold=base_threshold,
            team_metrics=team_metrics,
            problem_complexity=problem.complexity_score,
            sub_problem_complexity=sub_problem.complexity_score
        )

        # Create adaptive rounds
        rounds = self._create_adaptive_rounds(
            sub_problem=sub_problem,
            adapted_threshold=adapted_threshold,
            team_metrics=team_metrics
        )

        # Create gauntlet definition
        gauntlet = GauntletDefinition(
            gauntlet_id=f"adaptive_{sub_problem.id}",
            name=f"Adaptive Gauntlet for {sub_problem.title}",
            description=f"Adaptive validation gauntlet with threshold {adapted_threshold:.2f}",
            rounds=rounds,
            execution_order="adaptive",
            stop_on_first_failure=False,
            require_all_rounds=True,
            red_team_required=True,
            gold_team_required=True,
            blue_team_participation="observer",
            metadata={
                "adapted_threshold": adapted_threshold,
                "base_threshold": base_threshold,
                "adaptation_factors": self._get_adaptation_factors(
                    team_metrics, problem.complexity_score
                ),
                "created_at": datetime.now().isoformat()
            }
        )

        # Record adaptation
        self._record_adaptation(gauntlet, team_metrics, problem)

        return gauntlet

    def calculate_adaptive_threshold(
        self,
        base_threshold: float,
        team_metrics: Optional[TeamPerformanceMetrics],
        problem_complexity: ComplexityScore,
        sub_problem_complexity: Optional[ComplexityScore] = None
    ) -> float:
        """
        Calculate adaptive validation threshold.

        Formula:
        adaptive = base * performance_multiplier * complexity_multiplier

        Where:
        - performance_multiplier: 0.8-1.2 based on team's recent performance
        - complexity_multiplier: 0.9-1.1 based on problem complexity

        Args:
            base_threshold: Base threshold to adapt
            team_metrics: Team performance metrics
            problem_complexity: Problem complexity score
            sub_problem_complexity: Optional sub-problem complexity

        Returns:
            Adapted threshold (0.0-1.0)
        """
        # Performance multiplier
        performance_multiplier = 1.0
        if team_metrics:
            avg_quality = team_metrics.avg_quality_score

            if avg_quality > 0.9:
                performance_multiplier = 0.8  # High performers, raise bar
            elif avg_quality > 0.8:
                performance_multiplier = 0.9  # Good performers
            elif avg_quality > 0.6:
                performance_multiplier = 1.0  # Average performers
            elif avg_quality > 0.4:
                performance_multiplier = 1.1  # Below average
            else:
                performance_multiplier = 1.2  # Struggling, lower bar

        # Complexity multiplier
        overall_complexity = problem_complexity.overall_complexity

        if sub_problem_complexity:
            # Use sub-problem complexity if available
            overall_complexity = (
                problem_complexity.overall_complexity * 0.3 +
                sub_problem_complexity.overall_complexity * 0.7
            )

        # Higher complexity → lower threshold (easier to pass)
        # Lower complexity → higher threshold (harder to pass)
        if overall_complexity > 8.0:
            complexity_multiplier = 0.9
        elif overall_complexity > 6.0:
            complexity_multiplier = 0.95
        elif overall_complexity > 4.0:
            complexity_multiplier = 1.0
        elif overall_complexity > 2.0:
            complexity_multiplier = 1.05
        else:
            complexity_multiplier = 1.1

        # Calculate adaptive threshold
        adapted_threshold = base_threshold * performance_multiplier * complexity_multiplier

        # Clamp to valid range
        adapted_threshold = max(0.3, min(0.95, adapted_threshold))

        return adapted_threshold

    def adapt_round_difficulty(
        self,
        round_rule: GauntletRoundRule,
        performance_history: List[float]
    ) -> GauntletRoundRule:
        """
        Adapt individual round difficulty.

        If team consistently passing: increase difficulty (higher threshold)
        If team consistently failing: decrease difficulty (lower threshold)
        If performance volatile: keep stable

        Args:
            round_rule: Round rule to adapt
            performance_history: Recent performance scores

        Returns:
            Adapted GauntletRoundRule
        """
        if len(performance_history) < 3:
            # Not enough data, return unchanged
            return round_rule

        # Calculate recent performance
        recent_avg = sum(performance_history[-5:]) / min(5, len(performance_history))

        # Create adapted round
        adapted_round = GauntletRoundRule(
            rule_id=round_rule.rule_id,
            rule_type=round_rule.rule_type,
            description=round_rule.description,
            validation_type=round_rule.validation_type,
            min_score=round_rule.min_score,
            max_attempts=round_rule.max_attempts,
            evaluator=round_rule.evaluator,
            evaluation_prompt=round_rule.evaluation_prompt,
            success_criteria=round_rule.success_criteria.copy(),
            is_required=round_rule.is_required,
            can_fail_gracefully=round_rule.can_fail_gracefully,
            retry_on_failure=round_rule.retry_on_failure,
            metadata=round_rule.metadata.copy()
        )

        # Adapt threshold based on performance
        if recent_avg > 0.9:
            # Consistently passing, increase difficulty
            adapted_round.min_score = min(0.95, round_rule.min_score * 1.1)
            adapted_round.metadata["adaptation"] = "increased_difficulty"
        elif recent_avg < 0.5:
            # Consistently failing, decrease difficulty
            adapted_round.min_score = max(0.3, round_rule.min_score * 0.9)
            adapted_round.metadata["adaptation"] = "decreased_difficulty"
        else:
            # Volatile or average, keep stable
            adapted_round.metadata["adaptation"] = "stable"

        adapted_round.metadata["recent_performance_avg"] = recent_avg
        adapted_round.metadata["original_min_score"] = round_rule.min_score

        return adapted_round

    def _create_adaptive_rounds(
        self,
        sub_problem: SubProblem,
        adapted_threshold: float,
        team_metrics: Optional[TeamPerformanceMetrics]
    ) -> List[GauntletRoundRule]:
        """
        Create adaptive validation rounds.

        Args:
            sub_problem: Sub-problem to validate
            adapted_threshold: Adapted threshold
            team_metrics: Team performance metrics

        Returns:
            List of adaptive GauntletRoundRule objects
        """
        rounds = []

        # Round 1: Automated Quality Check
        rounds.append(GauntletRoundRule(
            rule_id=f"auto_quality_{sub_problem.id}",
            rule_type="automated",
            description="Automated quality and completeness check",
            validation_type="quality",
            min_score=adapted_threshold * 0.9,  # Slightly easier
            max_attempts=3,
            evaluator="automated",
            evaluation_prompt=f"""
            Evaluate the solution for sub-problem: {sub_problem.title}

            Check for:
            1. Completeness: Addresses all requirements
            2. Quality: Well-reasoned and well-explained
            3. Clarity: Clear and understandable
            4. Feasibility: Realistic and implementable

            Threshold: {adapted_threshold * 0.9:.2f}
            """,
            success_criteria=[
                "Addresses all acceptance criteria",
                "Meets quality standards",
                "Clear and actionable"
            ],
            is_required=True,
            can_fail_gracefully=False,
            retry_on_failure=True,
            metadata={
                "round_number": 1,
                "type": "automated",
                "adapted": True
            }
        ))

        # Round 2: Red Team Review (Adaptive)
        red_team_threshold = adapted_threshold
        if team_metrics and team_metrics.recent_performance:
            # Adapt based on recent performance
            recent_avg = sum(team_metrics.recent_performance[-3:]) / min(3, len(team_metrics.recent_performance))
            red_team_threshold = adapted_threshold * (0.9 if recent_avg < 0.6 else 1.0)

        rounds.append(GauntletRoundRule(
            rule_id=f"red_team_{sub_problem.id}",
            rule_type="red_team",
            description="Red team critique and vulnerability analysis",
            validation_type="acceptance",
            min_score=red_team_threshold,
            max_attempts=2,
            evaluator="red_team",
            evaluation_prompt=f"""
    Critically analyze this solution for sub-problem: {sub_problem.title}

    Focus on:
    1. Potential flaws or weaknesses
    2. Edge cases not considered
    3. Security or performance concerns
    4. Assumptions that may not hold

    Threshold: {red_team_threshold:.2f}
    """,
            success_criteria=[
                "Identifies potential issues",
                "Provides constructive feedback",
                "Actionable recommendations"
            ],
            is_required=True,
            can_fail_gracefully=False,
            retry_on_failure=True,
            metadata={
                "round_number": 2,
                "type": "red_team",
                "adapted": True,
                "adapted_threshold": red_team_threshold
            }
        ))

        # Round 3: Gold Team Verification
        rounds.append(GauntletRoundRule(
            rule_id=f"gold_team_{sub_problem.id}",
            rule_type="gold_team",
            description="Gold team verification and final approval",
            validation_type="acceptance",
            min_score=adapted_threshold,
            max_attempts=2,
            evaluator="gold_team",
            evaluation_prompt=f"""
            Verify and approve this solution for sub-problem: {sub_problem.title}

            Review:
            1. Addresses red team feedback
            2. Meets all acceptance criteria
            3. High quality and ready for integration
            4. No critical issues remaining

            Threshold: {adapted_threshold:.2f}
            """,
            success_criteria=[
                "All critical issues addressed",
                "Meets quality standards",
                "Ready for production"
            ],
            is_required=True,
            can_fail_gracefully=False,
            retry_on_failure=True,
            metadata={
                "round_number": 3,
                "type": "gold_team",
                "adapted": True
            }
        ))

        return rounds

    def _get_adaptation_factors(
        self,
        team_metrics: Optional[TeamPerformanceMetrics],
        problem_complexity: ComplexityScore
    ) -> Dict[str, Any]:
        """Get factors used in adaptation for logging."""
        factors = {
            "problem_complexity": problem_complexity.overall_complexity
        }

        if team_metrics:
            factors["team_avg_quality"] = team_metrics.avg_quality_score
            factors["team_success_rate"] = team_metrics.success_rate
            factors["team_trend"] = team_metrics.trend

        return factors

    def _record_adaptation(
        self,
        gauntlet: GauntletDefinition,
        team_metrics: Optional[TeamPerformanceMetrics],
        problem: ProblemDefinition
    ):
        """Record adaptation for analysis."""
        self.adaptation_history.append({
            "timestamp": datetime.now().isoformat(),
            "gauntlet_id": gauntlet.gauntlet_id,
            "gauntlet_name": gauntlet.name,
            "adapted_threshold": gauntlet.metadata.get("adapted_threshold"),
            "problem_id": problem.id,
            "problem_complexity": problem.complexity_score.overall_complexity,
            "team_metrics": {
                "avg_quality": team_metrics.avg_quality_score if team_metrics else None,
                "success_rate": team_metrics.success_rate if team_metrics else None,
                "trend": team_metrics.trend if team_metrics else None
            }
        })

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """
        Get summary of adaptations made.

        Returns:
            Summary statistics and recent adaptations
        """
        if not self.adaptation_history:
            return {
                "total_adaptations": 0,
                "avg_adapted_threshold": 0.7,
                "recent_adaptations": []
            }

        # Calculate statistics
        adapted_thresholds = [
            a["adapted_threshold"]
            for a in self.adaptation_history
            if a["adapted_threshold"] is not None
        ]

        avg_threshold = sum(adapted_thresholds) / len(adapted_thresholds) if adapted_thresholds else 0.7

        return {
            "total_adaptations": len(self.adaptation_history),
            "avg_adapted_threshold": avg_threshold,
            "min_threshold": min(adapted_thresholds) if adapted_thresholds else 0.7,
            "max_threshold": max(adapted_thresholds) if adapted_thresholds else 0.7,
            "recent_adaptations": self.adaptation_history[-10:]
        }

    def update_performance_from_result(
        self,
        gauntlet_id: str,
        team_id: str,
        domain: str,
        problem_type: str,
        passed: bool,
        score: float
    ):
        """
        Update performance tracking based on gauntlet result.

        Args:
            gauntlet_id: Gauntlet identifier
            team_id: Team that completed the gauntlet
            domain: Problem domain
            problem_type: Type of problem
            passed: Whether gauntlet was passed
            score: Final score achieved
        """
        self.performance_tracker.record_performance(
            team_id=team_id,
            domain=domain,
            problem_type=problem_type,
            gauntlet_name=gauntlet_id,
            score=score
        )

        logger.info(
            f"Recorded performance: team={team_id}, score={score:.2f}, "
            f"passed={passed}, domain={domain}, type={problem_type}"
        )
