"""
Learning & Adaptation Engine for Long-Horizon Agents

Implements online learning and strategy adaptation.
Follows CLAUDE.md principles:
- Law of Runtime Truth: Verify all learning outcomes
- Law of Idempotency: All operations replay-safe
- Law of UTC: All timestamps in UTC

Integrates with knowledge_engine for persistent learning.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import structlog
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
from enum import Enum

from .schemas.learning_schemas import (
    LearningOutcome,
    StrategyPerformance,
    ABTestResult,
    AdaptationAction
)


logger = structlog.get_logger()


class ExplorationStrategy(Enum):
    """Exploration strategies for learning"""
    EPSILON_GREEDY = "epsilon_greedy"
    SOFTMAX = "softmax"
    UCB = "ucb"  # Upper Confidence Bound
    THOMPSON_SAMPLING = "thompson_sampling"


class LearningEngine:
    """
    Online learning and strategy adaptation engine.

    Features:
    - Learning from outcomes (success/failure patterns)
    - Strategy adaptation based on feedback
    - A/B testing framework
    - Overfitting prevention (exploration vs exploitation)
    - Meta-learning across workflow instances

    All operations idempotent and UTC-based.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Learning Engine.

        Args:
            config: Optional configuration
                - exploration_rate: Initial exploration rate (default: 0.1)
                - exploration_decay: Decay factor for exploration (default: 0.995)
                - min_exploration_rate: Minimum exploration rate (default: 0.01)
                - learning_rate: Learning rate for updates (default: 0.1)
                - ab_test_sample_size: Sample size for A/B tests (default: 100)
        """
        self.config = config or self._load_default_config()

        # Strategy performance tracking
        self._strategy_performance: Dict[str, StrategyPerformance] = {}

        # Learning outcomes
        self._outcomes: List[LearningOutcome] = []

        # A/B tests
        self._ab_tests: Dict[str, ABTestResult] = {}

        # Adaptation history
        self._adaptations: List[AdaptationAction] = []

        # Exploration parameters
        self._exploration_rate = self.config.get('exploration_rate', 0.1)
        self._exploration_decay = self.config.get('exploration_decay', 0.995)
        self._min_exploration_rate = self.config.get('min_exploration_rate', 0.01)

        logger.info(
            "learning_engine_initialized",
            exploration_rate=self._exploration_rate,
            exploration_decay=self._exploration_decay
        )

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'exploration_rate': 0.1,
            'exploration_decay': 0.995,
            'min_exploration_rate': 0.01,
            'learning_rate': 0.1,
            'ab_test_sample_size': 100,
        }

    async def record_outcome(
        self,
        workflow_id: str,
        execution_id: str,
        lesson_type: str,
        lesson_description: str,
        success: bool,
        performance_score: float,
        strategy_used: str,
        parameters: Dict[str, Any],
        environmental_factors: Optional[Dict[str, Any]] = None,
        causal_factors: Optional[List[str]] = None,
        learned_by: str = "system"
    ) -> LearningOutcome:
        """
        Record a learning outcome (idempotent).

        Args:
            workflow_id: Workflow where learning occurred
            execution_id: Execution instance
            lesson_type: Type of lesson
            lesson_description: What was learned
            success: Whether outcome was successful
            performance_score: Performance score (0-1)
            strategy_used: Strategy that produced outcome
            parameters: Parameters used
            environmental_factors: Contextual factors
            causal_factors: Factors that caused outcome
            learned_by: Agent that learned

        Returns:
            LearningOutcome: Recorded outcome
        """
        outcome = LearningOutcome(
            outcome_id=self._generate_id('outcome'),
            workflow_id=workflow_id,
            execution_id=execution_id,
            lesson_type=lesson_type,
            lesson_description=lesson_description,
            success=success,
            performance_score=performance_score,
            strategy_used=strategy_used,
            parameters=parameters,
            environmental_factors=environmental_factors or {},
            causal_factors=causal_factors or [],
            learned_by=learned_by
        )

        # Check if already recorded (idempotency)
        existing = next(
            (o for o in self._outcomes if o.execution_id == execution_id),
            None
        )
        if existing:
            logger.info(
                "outcome_already_recorded",
                execution_id=execution_id,
                outcome_id=existing.outcome_id
            )
            return existing

        self._outcomes.append(outcome)

        # Update strategy performance
        await self._update_strategy_performance(
            strategy_used,
            success,
            performance_score
        )

        logger.info(
            "outcome_recorded",
            outcome_id=outcome.outcome_id,
            strategy=strategy_used,
            success=success,
            performance=performance_score
        )

        return outcome

    async def _update_strategy_performance(
        self,
        strategy_id: str,
        success: bool,
        performance_score: float
    ) -> None:
        """Update performance tracking for a strategy"""
        if strategy_id not in self._strategy_performance:
            self._strategy_performance[strategy_id] = StrategyPerformance(
                performance_id=self._generate_id('perf'),
                strategy_id=strategy_id,
                last_updated=datetime.now(timezone.utc),
                first_used=datetime.now(timezone.utc)
            )

        perf = self._strategy_performance[strategy_id]

        # Update totals
        perf.total_uses += 1
        if success:
            perf.successful_uses += 1
        else:
            perf.failed_uses += 1

        # Update average performance (exponential moving average)
        alpha = self.config.get('learning_rate', 0.1)
        perf.avg_performance_score = (
            alpha * performance_score +
            (1 - alpha) * perf.avg_performance_score
        )

        # Update recent performance
        perf.recent_performance.append(performance_score)
        if len(perf.recent_performance) > 100:
            perf.recent_performance.pop(0)

        # Detect trend
        if len(perf.recent_performance) >= 10:
            recent_avg = np.mean(perf.recent_performance[-10:])
            if recent_avg > perf.avg_performance_score + 0.1:
                perf.recent_trend = 'improving'
            elif recent_avg < perf.avg_performance_score - 0.1:
                perf.recent_trend = 'declining'
            else:
                perf.recent_trend = 'stable'

        perf.last_updated = datetime.now(timezone.utc)

    async def select_strategy(
        self,
        available_strategies: List[str],
        context: Optional[Dict[str, Any]] = None,
        exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    ) -> str:
        """
        Select a strategy using exploration-exploitation balance.

        Args:
            available_strategies: Strategies to choose from
            context: Optional context for decision
            exploration_strategy: Which exploration strategy to use

        Returns:
            Selected strategy ID
        """
        if not available_strategies:
            raise ValueError("No strategies available")

        # Explore with probability epsilon
        if np.random.random() < self._exploration_rate:
            # Explore: random selection
            selected = np.random.choice(available_strategies)

            logger.info(
                "strategy_selected_explore",
                strategy=selected,
                exploration_rate=self._exploration_rate
            )
        else:
            # Exploit: select best known strategy
            selected = await self._select_best_strategy(
                available_strategies,
                exploration_strategy
            )

            logger.info(
                "strategy_selected_exploit",
                strategy=selected,
                exploration_rate=self._exploration_rate
            )

        # Decay exploration rate
        self._exploration_rate = max(
            self._min_exploration_rate,
            self._exploration_rate * self._exploration_decay
        )

        return selected

    async def _select_best_strategy(
        self,
        available_strategies: List[str],
        exploration_strategy: ExplorationStrategy
    ) -> str:
        """Select best strategy using specified strategy"""
        # Get performance data for available strategies
        performances = {}
        for strategy_id in available_strategies:
            if strategy_id in self._strategy_performance:
                perf = self._strategy_performance[strategy_id]
                performances[strategy_id] = perf.avg_performance_score
            else:
                # Unseen strategy gets neutral score
                performances[strategy_id] = 0.5

        if exploration_strategy == ExplorationStrategy.EPSILON_GREEDY:
            # Select strategy with highest performance
            return max(performances.items(), key=lambda x: x[1])[0]

        elif exploration_strategy == ExplorationStrategy.SOFTMAX:
            # Softmax selection
            scores = np.array(list(performances.values()))
            exp_scores = np.exp(scores * 2)  # Temperature parameter
            probs = exp_scores / np.sum(exp_scores)

            selected_idx = np.random.choice(len(available_strategies), p=probs)
            return available_strategies[selected_idx]

        elif exploration_strategy == ExplorationStrategy.UCB:
            # Upper Confidence Bound
            ucb_scores = {}
            for strategy_id, score in performances.items():
                perf = self._strategy_performance.get(strategy_id)
                if perf and perf.total_uses > 0:
                    # UCB = score + exploration bonus
                    exploration_bonus = np.sqrt(2 * np.log(len(self._outcomes) + 1) / perf.total_uses)
                    ucb_scores[strategy_id] = score + exploration_bonus
                else:
                    ucb_scores[strategy_id] = 1.0  # Unseen strategies get high UCB

            return max(ucb_scores.items(), key=lambda x: x[1])[0]

        else:
            # Default to epsilon-greedy
            return max(performances.items(), key=lambda x: x[1])[0]

    async def run_ab_test(
        self,
        test_name: str,
        hypothesis: str,
        control_strategy: str,
        treatment_strategy: str,
        test_context: Dict[str, Any],
        sample_size: Optional[int] = None
    ) -> ABTestResult:
        """
        Run an A/B test comparing two strategies.

        Args:
            test_name: Test name
            hypothesis: Hypothesis being tested
            control_strategy: Control strategy ID
            treatment_strategy: Treatment strategy ID
            test_context: Context for test
            sample_size: Sample size (uses config default if None)

        Returns:
            ABTestResult: Test results
        """
        sample_size = sample_size or self.config.get('ab_test_sample_size', 100)

        # Get outcomes for each strategy
        control_outcomes = [
            o for o in self._outcomes
            if o.strategy_used == control_strategy
        ]

        treatment_outcomes = [
            o for o in self._outcomes
            if o.strategy_used == treatment_strategy
        ]

        # Use recent outcomes
        control_scores = [o.performance_score for o in control_outcomes[-sample_size:]]
        treatment_scores = [o.performance_score for o in treatment_outcomes[-sample_size:]]

        if not control_scores or not treatment_scores:
            raise ValueError("Insufficient data for A/B test")

        # Calculate metrics
        control_mean = np.mean(control_scores)
        treatment_mean = np.mean(treatment_scores)
        delta = treatment_mean - control_mean

        # Simple statistical test (t-test approximation)
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(treatment_scores, control_scores)

        # Determine significance
        is_significant = p_value < 0.05

        # Recommendation
        if is_significant and delta > 0:
            recommended = treatment_strategy
            confidence = 1 - p_value
        elif is_significant and delta < 0:
            recommended = control_strategy
            confidence = 1 - p_value
        else:
            # No significant difference
            recommended = control_strategy  # Default to control
            confidence = 0.5

        result = ABTestResult(
            test_id=self._generate_id('abtest'),
            test_name=test_name,
            hypothesis=hypothesis,
            control_strategy=control_strategy,
            treatment_strategy=treatment_strategy,
            control_performance=float(control_mean),
            treatment_performance=float(treatment_mean),
            performance_delta=float(delta),
            sample_size=len(control_scores) + len(treatment_scores),
            p_value=float(p_value) if p_value else None,
            is_significant=is_significant,
            test_context=test_context,
            recommended_strategy=recommended,
            recommendation_confidence=confidence,
            started_at=min(
                min(o.learned_at for o in control_outcomes),
                min(o.learned_at for o in treatment_outcomes)
            ),
            completed_at=datetime.now(timezone.utc),
            conducted_by='learning_engine'
        )

        self._ab_tests[result.test_id] = result

        logger.info(
            "ab_test_completed",
            test_name=test_name,
            winner=recommended,
            is_significant=is_significant
        )

        return result

    async def adapt_strategy(
        self,
        trigger_reason: str,
        target_component: str,
        new_value: Any,
        previous_value: Optional[Any],
        expected_improvement: str,
        adapted_by: str
    ) -> AdaptationAction:
        """
        Make an adaptation to agent behavior.

        Args:
            trigger_reason: Why adaptation is being made
            target_component: Component being adapted
            new_value: New value
            previous_value: Previous value
            expected_improvement: Expected improvement description
            adapted_by: Agent making adaptation

        Returns:
            AdaptationAction: Adaptation record
        """
        action = AdaptationAction(
            action_id=self._generate_id('adapt'),
            action_type='parameter_change',
            target_component=target_component,
            previous_value=previous_value,
            new_value=new_value,
            trigger_reason=trigger_reason,
            expected_improvement=expected_improvement,
            adapted_by=adapted_by
        )

        self._adaptations.append(action)

        logger.info(
            "adaptation_made",
            action_id=action.action_id,
            target_component=target_component,
            reason=trigger_reason
        )

        return action

    async def validate_adaptation(
        self,
        action_id: str,
        validation_result: Dict[str, Any]
    ) -> None:
        """
        Validate an adaptation's effectiveness.

        Args:
            action_id: Adaptation to validate
            validation_result: Validation results
        """
        action = next((a for a in self._adaptations if a.action_id == action_id), None)

        if not action:
            raise ValueError(f"Adaptation {action_id} not found")

        action.validated = True
        action.validation_result = validation_result

        logger.info(
            "adaptation_validated",
            action_id=action_id,
            result=validation_result
        )

    async def get_strategy_recommendations(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get strategy recommendations based on learning.

        Args:
            context: Optional context for recommendations

        Returns:
            List of recommendations
        """
        recommendations = []

        # Top performing strategies
        sorted_strategies = sorted(
            self._strategy_performance.items(),
            key=lambda x: x[1].avg_performance_score,
            reverse=True
        )

        for strategy_id, perf in sorted_strategies[:5]:  # Top 5
            if perf.total_uses >= 3:  # Minimum data
                recommendations.append({
                    'strategy_id': strategy_id,
                    'avg_performance': perf.avg_performance_score,
                    'success_rate': perf.success_rate,
                    'total_uses': perf.total_uses,
                    'trend': perf.recent_trend,
                    'confidence': min(perf.total_uses / 10, 1.0)  # More uses = more confidence
                })

        return recommendations

    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress"""
        total_outcomes = len(self._outcomes)
        successful_outcomes = sum(1 for o in self._outcomes if o.success)

        return {
            'total_outcomes': total_outcomes,
            'successful_outcomes': successful_outcomes,
            'success_rate': successful_outcomes / total_outcomes if total_outcomes > 0 else 0,
            'strategies_tracked': len(self._strategy_performance),
            'ab_tests_conducted': len(self._ab_tests),
            'adaptations_made': len(self._adaptations),
            'current_exploration_rate': self._exploration_rate
        }

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID with prefix"""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:16]}"
