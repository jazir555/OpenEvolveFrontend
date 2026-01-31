"""
Online Learning Module

Continuously learns from streaming workflow outcomes.
Supports strategy performance tracking, exploration vs exploitation,
and adaptation recommendations.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, UTC, timedelta
from collections import deque
import numpy as np
import asyncio
import json
import logging

from .schemas.long_horizon import (
    LearningOutcome,
    StrategyPerformance,
    AdaptationAction,
    AdaptationActionType,
    ExplorationStrategy,
    OutcomeType
)


logger = logging.getLogger(__name__)


class OnlineLearner:
    """
    Online learning from continuous workflow outcomes

    Tracks strategy performance over time, manages exploration vs exploitation,
    and recommends when to adapt strategies.

    Usage:
        learner = OnlineLearner(
            exploration_strategy=ExplorationStrategy.EPSILON_GREEDY,
            initial_epsilon=0.3
        )

        # Record outcomes as they happen
        await learner.record_outcome(outcome)

        # Get best strategy
        best = await learner.get_best_strategy(workflow_id="wf_123")

        # Check if we should explore
        if await learner.should_explore():
            strategy = await learner.select_exploration_strategy()
    """

    def __init__(
        self,
        exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY,
        initial_epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        performance_window: int = 100,
        adaptation_threshold: float = 0.15,
        decay_detection_window: int = 20
    ):
        """
        Initialize online learner

        Args:
            exploration_strategy: Strategy for exploration vs exploitation
            initial_epsilon: Initial exploration rate (for epsilon-greedy)
            epsilon_decay: Rate at which epsilon decays
            min_epsilon: Minimum exploration rate
            performance_window: How many outcomes to consider for moving average
            adaptation_threshold: Performance drop threshold triggering adaptation
            decay_detection_window: Window for detecting performance decay
        """
        self.exploration_strategy = exploration_strategy
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.performance_window = performance_window
        self.adaptation_threshold = adaptation_threshold
        self.decay_detection_window = decay_detection_window

        # Strategy performance tracking
        # Key: workflow_id -> strategy_id -> StrategyPerformance
        self.strategies: Dict[str, Dict[str, StrategyPerformance]] = {}

        # Outcome history (for replay and analysis)
        # Key: workflow_id -> deque of LearningOutcome
        self.outcome_history: Dict[str, deque] = {}

        # Exploration statistics
        self.total_decisions: int = 0
        self.explore_count: int = 0
        self.exploit_count: int = 0

    async def record_outcome(self, outcome: LearningOutcome) -> None:
        """
        Record a workflow outcome (idempotent)

        Args:
            outcome: Learning outcome to record

        Law of Idempotency: Safe to call multiple times with same outcome
        """
        workflow_id = outcome.workflow_id
        strategy_id = outcome.strategy_used

        # Initialize structures if needed
        if workflow_id not in self.strategies:
            self.strategies[workflow_id] = {}
            self.outcome_history[workflow_id] = deque(maxlen=1000)

        # Check for duplicate (idempotency)
        existing = any(
            o.outcome_id == outcome.outcome_id
            for o in self.outcome_history[workflow_id]
        )
        if existing:
            logger.debug(f"Duplicate outcome {outcome.outcome_id}, skipping")
            return

        # Initialize strategy performance if needed
        if strategy_id not in self.strategies[workflow_id]:
            self.strategies[workflow_id][strategy_id] = StrategyPerformance(
                strategy_id=strategy_id
            )

        # Extract performance score
        # Use fitness if available, otherwise derive from metrics
        if "fitness" in outcome.metrics:
            score = outcome.metrics["fitness"]
        elif outcome.outcome_type == OutcomeType.SUCCESS:
            score = 1.0
        elif outcome.outcome_type == OutcomeType.FAILURE:
            score = 0.0
        else:  # PARTIAL or ERROR
            score = outcome.metrics.get("score", 0.5)

        # Update strategy performance
        perf = self.strategies[workflow_id][strategy_id]
        perf.performance_history.append(score)
        perf.total_outcomes += 1

        # Update success rate
        if outcome.outcome_type == OutcomeType.SUCCESS:
            perf.success_rate = (
                (perf.success_rate * (perf.total_outcomes - 1) + 1.0) /
                perf.total_outcomes
            )
        else:
            perf.success_rate = (
                perf.success_rate * (perf.total_outcomes - 1)
            ) / perf.total_outcomes

        # Update moving average (exponential)
        alpha = 2.0 / (self.performance_window + 1)
        perf.moving_average = (
            alpha * score + (1 - alpha) * perf.moving_average
        )

        # Update variance (for confidence intervals)
        if len(perf.performance_history) > 1:
            perf.variance = np.var(perf.performance_history[-self.performance_window:])
            std_error = np.sqrt(perf.variance / len(perf.performance_history))
            # 95% confidence interval
            perf.confidence_interval = (
                max(0.0, perf.moving_average - 1.96 * std_error),
                min(1.0, perf.moving_average + 1.96 * std_error)
            )

        # Detect performance decay
        if len(perf.performance_history) >= self.decay_detection_window:
            recent = np.mean(perf.performance_history[-self.decay_detection_window:])
            older = np.mean(perf.performance_history[-2*self.decay_detection_window:-self.decay_detection_window])
            perf.decay_rate = older - recent  # Positive = degrading
        else:
            perf.decay_rate = 0.0

        perf.last_updated = datetime.now(UTC)

        # Store outcome
        self.outcome_history[workflow_id].append(outcome)

        logger.info(
            f"Recorded outcome {outcome.outcome_id}: "
            f"workflow={workflow_id}, strategy={strategy_id}, "
            f"score={score:.3f}"
        )

    async def get_best_strategy(
        self,
        workflow_id: str,
        time_window: Optional[timedelta] = None
    ) -> Optional[str]:
        """
        Get the best performing strategy for a workflow

        Args:
            workflow_id: Workflow to check
            time_window: Optional time window for recent performance

        Returns:
            Best strategy ID or None if no data
        """
        if workflow_id not in self.strategies:
            return None

        strategies = self.strategies[workflow_id]
        if not strategies:
            return None

        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now(UTC) - time_window
            recent_perf = {}
            for strat_id, perf in strategies.items():
                if perf.last_updated >= cutoff:
                    recent_perf[strat_id] = perf
            strategies = recent_perf

        if not strategies:
            return None

        # Select by moving average
        best = max(strategies.items(), key=lambda x: x[1].moving_average)
        return best[0]

    async def should_explore(self) -> bool:
        """
        Decide whether to explore or exploit

        Uses configured exploration strategy:
        - EPSILON_GREEDY: Explore with probability epsilon
        - UCB: Explore if upper confidence bound of best is uncertain
        - THOMPSON_SAMPLING: Sample from posterior distributions

        Returns:
            True if should explore, False if should exploit
        """
        self.total_decisions += 1

        if self.exploration_strategy == ExplorationStrategy.EPSILON_GREEDY:
            should = np.random.random() < self.epsilon

            # Decay epsilon
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon * self.epsilon_decay
            )

            if should:
                self.explore_count += 1
            else:
                self.exploit_count += 1

            return should

        elif self.exploration_strategy == ExplorationStrategy.UCB:
            # Upper Confidence Bound: Explore if best strategy is uncertain
            # This requires workflow context, so we'll return True to trigger selection
            return True

        elif self.exploration_strategy == ExplorationStrategy.THOMPSON_SAMPLING:
            # Thompson Sampling: Sample from posterior, return True to trigger
            return True

        else:
            # Default: epsilon-greedy
            return np.random.random() < self.epsilon

    async def select_exploration_strategy(
        self,
        workflow_id: str
    ) -> Optional[str]:
        """
        Select a strategy for exploration

        Args:
            workflow_id: Workflow context

        Returns:
            Selected strategy ID
        """
        if workflow_id not in self.strategies:
            return None

        strategies = list(self.strategies[workflow_id].keys())
        if not strategies:
            return None

        if self.exploration_strategy == ExplorationStrategy.EPSILON_GREEDY:
            # Random exploration
            return np.random.choice(strategies)

        elif self.exploration_strategy == ExplorationStrategy.UCB:
            # Upper Confidence Bound selection
            best_ucb = -float('inf')
            best_strat = None

            for strat_id, perf in self.strategies[workflow_id].items():
                # UCB = mean + sqrt(2*ln(n)/n_i)
                n = perf.total_outcomes
                if n == 0:
                    ucb = float('inf')
                else:
                    exploration_bonus = np.sqrt(2 * np.log(self.total_decisions) / n)
                    ucb = perf.moving_average + exploration_bonus

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_strat = strat_id

            return best_strat

        elif self.exploration_strategy == ExplorationStrategy.THOMPSON_SAMPLING:
            # Thompson Sampling: Sample from Beta distribution
            best_sample = -float('inf')
            best_strat = None

            for strat_id, perf in self.strategies[workflow_id].items():
                # Beta distribution parameters
                # Alpha = successes + 1, Beta = failures + 1
                successes = int(perf.success_rate * perf.total_outcomes)
                failures = perf.total_outcomes - successes

                sample = np.random.beta(successes + 1, failures + 1)

                if sample > best_sample:
                    best_sample = sample
                    best_strat = strat_id

            return best_strat

        else:
            return np.random.choice(strategies)

    async def adapt_strategy(
        self,
        workflow_id: str,
        current_performance: float
    ) -> Optional[AdaptationAction]:
        """
        Recommend if/how to adapt strategy

        Triggers adaptation when:
        1. Performance drops below threshold
        2. Performance decay detected
        3. Better strategy available

        Args:
            workflow_id: Workflow to check
            current_performance: Current performance score

        Returns:
            Adaptation action or None
        """
        if workflow_id not in self.strategies:
            return None

        strategies = self.strategies[workflow_id]
        if not strategies:
            return None

        current_strategy = await self.get_best_strategy(workflow_id)
        if not current_strategy:
            return None

        current_perf = strategies[current_strategy]

        # Check 1: Performance decay
        if current_perf.decay_rate > self.adaptation_threshold:
            # Significant degradation detected
            return AdaptationAction(
                action_type=AdaptationActionType.CHANGE_STRATEGY,
                description=f"Performance decaying at {current_perf.decay_rate:.3f} per iteration",
                parameters={
                    "from_strategy": current_strategy,
                    "reason": "performance_decay"
                },
                expected_improvement=0.2,
                confidence=0.8,
                rollback_plan=f"Revert to {current_strategy} if new strategy underperforms",
                priority=75.0
            )

        # Check 2: Better strategy available
        best_strat = await self.get_best_strategy(workflow_id)
        if best_strat and best_strat != current_strategy:
            best_perf = strategies[best_strat]
            improvement = (
                (best_perf.moving_average - current_perf.moving_average) /
                max(0.01, current_perf.moving_average)
            )

            if improvement > self.adaptation_threshold:
                return AdaptationAction(
                    action_type=AdaptationActionType.CHANGE_STRATEGY,
                    description=f"Switch to {best_strat} for {improvement:.1%} improvement",
                    parameters={
                        "from_strategy": current_strategy,
                        "to_strategy": best_strat,
                        "expected_improvement": improvement
                    },
                    expected_improvement=improvement,
                    confidence=0.85,
                    rollback_plan=f"Revert to {current_strategy} if improvement not realized",
                    priority=70.0
                )

        # Check 3: Parameter tuning opportunity
        if current_perf.total_outcomes > 50 and current_perf.variance > 0.1:
            # High variance suggests parameters need tuning
            return AdaptationAction(
                action_type=AdaptationActionType.TUNE_PARAMETERS,
                description="High performance variance suggests parameter tuning needed",
                parameters={
                    "strategy": current_strategy,
                    "current_variance": current_perf.variance,
                    "tuning_method": "bayesian_optimization"
                },
                expected_improvement=0.15,
                confidence=0.7,
                rollback_plan="Revert to current parameters if tuning degrades performance",
                priority=60.0
            )

        # No adaptation needed
        return None

    async def get_strategy_performance(
        self,
        workflow_id: str,
        strategy_id: str
    ) -> Optional[StrategyPerformance]:
        """
        Get performance data for a specific strategy

        Args:
            workflow_id: Workflow identifier
            strategy_id: Strategy identifier

        Returns:
            Strategy performance or None
        """
        if workflow_id not in self.strategies:
            return None

        return self.strategies[workflow_id].get(strategy_id)

    async def get_all_strategies(
        self,
        workflow_id: str
    ) -> Dict[str, StrategyPerformance]:
        """
        Get all strategies for a workflow

        Args:
            workflow_id: Workflow identifier

        Returns:
            Dict mapping strategy_id to performance
        """
        return self.strategies.get(workflow_id, {})

    async def get_outcome_history(
        self,
        workflow_id: str,
        limit: int = 100
    ) -> List[LearningOutcome]:
        """
        Get recent outcomes for a workflow

        Args:
            workflow_id: Workflow identifier
            limit: Maximum number of outcomes

        Returns:
            List of learning outcomes
        """
        if workflow_id not in self.outcome_history:
            return []

        history = list(self.outcome_history[workflow_id])
        return history[-limit:]

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall learning statistics

        Returns:
            Statistics dictionary
        """
        total_strategies = sum(
            len(strats)
            for strats in self.strategies.values()
        )
        total_outcomes = sum(
            perf.total_outcomes
            for strats in self.strategies.values()
            for perf in strats.values()
        )

        explore_rate = (
            self.explore_count / self.total_decisions
            if self.total_decisions > 0
            else 0.0
        )

        return {
            "total_workflows": len(self.strategies),
            "total_strategies": total_strategies,
            "total_outcomes": total_outcomes,
            "total_decisions": self.total_decisions,
            "explore_count": self.explore_count,
            "exploit_count": self.exploit_count,
            "explore_rate": explore_rate,
            "current_epsilon": self.epsilon,
            "exploration_strategy": self.exploration_strategy.value
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "exploration_strategy": self.exploration_strategy.value,
            "epsilon": self.epsilon,
            "performance_window": self.performance_window,
            "adaptation_threshold": self.adaptation_threshold,
            "strategies": {
                wf_id: {
                    s_id: perf.to_dict()
                    for s_id, perf in strats.items()
                }
                for wf_id, strats in self.strategies.items()
            },
            "statistics": asyncio.create_task(self.get_statistics()).result()
            if hasattr(asyncio.create_task(self.get_statistics()), 'result')
            else {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OnlineLearner':
        """Deserialize from dictionary"""
        learner = cls(
            exploration_strategy=ExplorationStrategy(data["exploration_strategy"]),
            performance_window=data["performance_window"],
            adaptation_threshold=data["adaptation_threshold"]
        )
        learner.epsilon = data["epsilon"]

        # Restore strategies
        # Note: This is a simplified restoration
        # Full restoration would need to reconstruct StrategyPerformance objects

        return learner
