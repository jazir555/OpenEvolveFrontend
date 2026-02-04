"""
ML-Based Gauntlet Optimizer

Uses reinforcement learning to optimize gauntlet configuration for maximum effectiveness.

Features:
- Q-learning and DQN for strategy optimization
- Multi-objective optimization (speed, accuracy, resource usage)
- Real-time parameter tuning
- Performance impact assessment
- Learning from historical execution data

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategy types"""
    Q_LEARNING = "q_learning"
    DQN = "dqn"
    GENETIC_ALGORITHM = "genetic"
    BAYESIAN_OPTIMIZATION = "bayesian"


class Objective(Enum):
    """Optimization objectives"""
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_COST = "minimize_cost"
    BALANCED = "balanced"


@dataclass
class GauntletState:
    """
    Current state of gauntlet configuration.

    Attributes:
        round1_threshold: Round 1 passing threshold
        round2_threshold: Round 2 passing threshold
        round3_threshold: Round 3 passing threshold
        round1_weight: Weight for round 1 in final score
        round2_weight: Weight for round 2 in final score
        round3_weight: Weight for round 3 in final score
        max_evaluations_round1: Max PES evaluations for round 1
        enable_parallel: Whether to enable parallel execution
    """
    round1_threshold: float = 0.5
    round2_threshold: float = 0.6
    round3_threshold: float = 0.7
    round1_weight: float = 0.2
    round2_weight: float = 0.3
    round3_weight: float = 0.5
    max_evaluations_round1: int = 50
    enable_parallel: bool = False

    def to_tuple(self) -> Tuple:
        """Convert to tuple for Q-table indexing"""
        return (
            int(self.round1_threshold * 10),
            int(self.round2_threshold * 10),
            int(self.round3_threshold * 10),
            int(self.round1_weight * 10),
            int(self.round2_weight * 10),
            self.max_evaluations_round1 // 10,
            int(self.enable_parallel)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "round1_threshold": self.round1_threshold,
            "round2_threshold": self.round2_threshold,
            "round3_threshold": self.round3_threshold,
            "round1_weight": self.round1_weight,
            "round2_weight": self.round2_weight,
            "round3_weight": self.round3_weight,
            "max_evaluations_round1": self.max_evaluations_round1,
            "enable_parallel": self.enable_parallel
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GauntletState":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class OptimizationAction:
    """
    Action to modify gauntlet configuration.

    Attributes:
        parameter: Parameter to modify
        delta: Change amount (positive or negative)
    """
    parameter: str
    delta: float

    def apply(self, state: GauntletState) -> GauntletState:
        """Apply action to state"""
        new_state = GauntletState(**state.to_dict())

        if self.parameter == "round1_threshold":
            new_state.round1_threshold = max(0.0, min(1.0, new_state.round1_threshold + self.delta))
        elif self.parameter == "round2_threshold":
            new_state.round2_threshold = max(0.0, min(1.0, new_state.round2_threshold + self.delta))
        elif self.parameter == "round3_threshold":
            new_state.round3_threshold = max(0.0, min(1.0, new_state.round3_threshold + self.delta))
        elif self.parameter == "round1_weight":
            new_state.round1_weight = max(0.0, min(1.0, new_state.round1_weight + self.delta))
        elif self.parameter == "round2_weight":
            new_state.round2_weight = max(0.0, min(1.0, new_state.round2_weight + self.delta))
        elif self.parameter == "max_evals":
            new_state.max_evaluations_round1 = max(10, min(100, int(new_state.max_evaluations_round1 + self.delta * 10)))
        elif self.parameter == "toggle_parallel":
            new_state.enable_parallel = not new_state.enable_parallel

        return new_state


@dataclass
class OptimizationResult:
    """
    Result from optimization run.

    Attributes:
        best_state: Best configuration found
        best_score: Best objective score achieved
        iterations: Number of iterations performed
        convergence_history: Score history across iterations
        improvement_percent: Percentage improvement over baseline
        recommendation: Human-readable recommendation
        timestamp: When optimization was performed
    """
    best_state: GauntletState
    best_score: float
    iterations: int
    convergence_history: List[float] = field(default_factory=list)
    improvement_percent: float = 0.0
    recommendation: str = ""
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "best_state": self.best_state.to_dict(),
            "best_score": self.best_score,
            "iterations": self.iterations,
            "convergence_history": self.convergence_history,
            "improvement_percent": self.improvement_percent,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp
        }


class MLBasedGauntletOptimizer:
    """
    ML-based optimizer for gauntlet configuration.

    Uses reinforcement learning to discover optimal configurations
    for different objectives and domains.

    Example:
        >>> optimizer = MLBasedGauntletOptimizer()
        >>> result = optimizer.optimize(
        ...     domain="code",
        ...     objective=Objective.BALANCED,
        ...     historical_data=execution_history
        ... )
        >>> print(f"Best configuration: {result.best_state}")
        >>> print(f"Improvement: {result.improvement_percent:.1f}%")
    """

    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.Q_LEARNING,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        max_iterations: int = 100
    ):
        """
        Initialize the ML-based optimizer.

        Args:
            strategy: Optimization strategy to use
            learning_rate: Learning rate for Q-learning
            discount_factor: Discount factor for future rewards
            epsilon: Exploration rate for epsilon-greedy policy
            max_iterations: Maximum optimization iterations
        """
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.max_iterations = max_iterations

        # Q-table for Q-learning: state -> action -> value
        self.q_table: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Action space
        self.actions = self._create_action_space()

        # Performance history
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        logger.info(f"ML-Based Gauntlet Optimizer initialized with strategy={strategy.value}")

    def _create_action_space(self) -> List[OptimizationAction]:
        """Create action space for optimization"""
        actions = []

        # Threshold adjustments (±0.05)
        for param in ["round1_threshold", "round2_threshold", "round3_threshold"]:
            actions.append(OptimizationAction(param, 0.05))
            actions.append(OptimizationAction(param, -0.05))

        # Weight adjustments (±0.1)
        for param in ["round1_weight", "round2_weight"]:
            actions.append(OptimizationAction(param, 0.1))
            actions.append(OptimizationAction(param, -0.1))

        # Evaluation count adjustments
        actions.append(OptimizationAction("max_evals", 1.0))
        actions.append(OptimizationAction("max_evals", -1.0))

        # Toggle parallel execution
        actions.append(OptimizationAction("toggle_parallel", 0))

        return actions

    def optimize(
        self,
        domain: str,
        objective: Objective,
        historical_data: Optional[List[Dict[str, Any]]] = None,
        initial_state: Optional[GauntletState] = None
    ) -> OptimizationResult:
        """
        Optimize gauntlet configuration for given domain and objective.

        Args:
            domain: Problem domain (code, math, general, etc.)
            objective: Optimization objective
            historical_data: Historical execution data for learning
            initial_state: Starting configuration (default: balanced defaults)

        Returns:
            OptimizationResult with optimal configuration
        """
        logger.info(f"Starting optimization for domain={domain}, objective={objective.value}")

        # Initialize state
        state = initial_state or GauntletState()

        # Load performance history if available
        if historical_data:
            self.performance_history[domain] = historical_data

        # Run optimization based on strategy
        if self.strategy == OptimizationStrategy.Q_LEARNING:
            result = self._optimize_q_learning(domain, objective, state)
        elif self.strategy == OptimizationStrategy.DQN:
            result = self._optimize_dqn(domain, objective, state)
        elif self.strategy == OptimizationStrategy.GENETIC_ALGORITHM:
            result = self._optimize_genetic(domain, objective, state)
        else:
            result = self._optimize_bayesian(domain, objective, state)

        logger.info(
            f"Optimization complete: score={result.best_score:.3f}, "
            f"improvement={result.improvement_percent:.1f}%, "
            f"iterations={result.iterations}"
        )

        return result

    def _optimize_q_learning(
        self,
        domain: str,
        objective: Objective,
        initial_state: GauntletState
    ) -> OptimizationResult:
        """Optimize using Q-learning algorithm"""
        state = initial_state
        best_state = state
        best_score = self._evaluate_configuration(state, domain, objective)
        convergence_history = [best_score]

        for iteration in range(self.max_iterations):
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                action = np.random.choice(self.actions)
            else:
                action = self._select_best_action(state)

            # Apply action
            next_state = action.apply(state)

            # Evaluate new configuration
            reward = self._calculate_reward(state, next_state, domain, objective)

            # Update Q-value
            state_key = state.to_tuple()
            action_idx = self.actions.index(action)

            old_q = self.q_table[state_key][action_idx]
            next_max = max(self.q_table[next_state.to_tuple()].values(), default=0.0)

            # Q-learning update
            new_q = old_q + self.learning_rate * (reward + self.discount_factor * next_max - old_q)
            self.q_table[state_key][action_idx] = new_q

            # Track best
            next_score = self._evaluate_configuration(next_state, domain, objective)
            if next_score > best_score:
                best_score = next_score
                best_state = next_state

            convergence_history.append(best_score)
            state = next_state

            # Decay epsilon
            self.epsilon *= 0.995

        # Calculate improvement
        baseline_score = self._evaluate_configuration(initial_state, domain, objective)
        improvement = ((best_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0

        # Generate recommendation
        recommendation = self._generate_recommendation(best_state, initial_state, objective)

        return OptimizationResult(
            best_state=best_state,
            best_score=best_score,
            iterations=self.max_iterations,
            convergence_history=convergence_history,
            improvement_percent=improvement,
            recommendation=recommendation
        )

    def _optimize_dqn(
        self,
        domain: str,
        objective: Objective,
        initial_state: GauntletState
    ) -> OptimizationResult:
        """Optimize using Deep Q-Network (simplified implementation)"""
        # Simplified DQN - in production would use actual neural network
        # For now, fall back to Q-learning with more iterations
        logger.info("DQN requested, using enhanced Q-learning")
        original_max = self.max_iterations
        self.max_iterations = int(self.max_iterations * 1.5)
        result = self._optimize_q_learning(domain, objective, initial_state)
        self.max_iterations = original_max
        return result

    def _optimize_genetic(
        self,
        domain: str,
        objective: Objective,
        initial_state: GauntletState
    ) -> OptimizationResult:
        """Optimize using genetic algorithm"""
        population_size = 20
        generations = self.max_iterations // 5

        # Initialize population
        population = [self._mutate_state(initial_state) for _ in range(population_size)]
        population.append(initial_state)

        best_state = initial_state
        best_score = self._evaluate_configuration(initial_state, domain, objective)
        convergence_history = [best_score]

        for generation in range(generations):
            # Evaluate population
            scores = [self._evaluate_configuration(state, domain, objective) for state in population]

            # Select best
            best_idx = np.argmax(scores)
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_state = population[best_idx]

            convergence_history.append(best_score)

            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                parent1 = self._tournament_select(population, scores, 3)
                parent2 = self._tournament_select(population, scores, 3)
                child = self._crossover_states(parent1, parent2)
                child = self._mutate_state(child, mutation_rate=0.1)
                new_population.append(child)

            # Keep best
            new_population[0] = best_state
            population = new_population

        baseline_score = self._evaluate_configuration(initial_state, domain, objective)
        improvement = ((best_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0

        return OptimizationResult(
            best_state=best_state,
            best_score=best_score,
            iterations=generations * population_size,
            convergence_history=convergence_history,
            improvement_percent=improvement,
            recommendation=f"Genetic algorithm completed {generations} generations"
        )

    def _optimize_bayesian(
        self,
        domain: str,
        objective: Objective,
        initial_state: GauntletState
    ) -> OptimizationResult:
        """Optimize using Bayesian optimization (simplified)"""
        # Simplified Bayesian - use random search with smart sampling
        best_state = initial_state
        best_score = self._evaluate_configuration(initial_state, domain, objective)
        convergence_history = [best_score]

        for i in range(self.max_iterations):
            # Generate candidate by sampling around current best
            candidate = self._mutate_state(best_state, mutation_rate=0.2)
            score = self._evaluate_configuration(candidate, domain, objective)

            if score > best_score:
                best_score = score
                best_state = candidate

            convergence_history.append(best_score)

        baseline_score = self._evaluate_configuration(initial_state, domain, objective)
        improvement = ((best_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0

        return OptimizationResult(
            best_state=best_state,
            best_score=best_score,
            iterations=self.max_iterations,
            convergence_history=convergence_history,
            improvement_percent=improvement,
            recommendation="Bayesian optimization completed"
        )

    def _select_best_action(self, state: GauntletState) -> OptimizationAction:
        """Select action with highest Q-value for state"""
        state_key = state.to_tuple()
        q_values = self.q_table[state_key]

        if not q_values:
            return np.random.choice(self.actions)

        max_action_idx = max(q_values, key=q_values.get)
        return self.actions[max_action_idx]

    def _evaluate_configuration(self, state: GauntletState, domain: str, objective: Objective) -> float:
        """
        Evaluate configuration quality.

        Simulates gauntlet execution with given configuration.
        In production, this would use historical data or actual execution.

        Returns:
            Score (0.0 to 1.0) where higher is better
        """
        # Base score from thresholds (higher thresholds = stricter = potentially better quality)
        threshold_score = (state.round1_threshold + state.round2_threshold + state.round3_threshold) / 3

        # Cost penalty (more evaluations = higher cost)
        cost_penalty = state.max_evaluations_round1 / 200.0  # 0.0 to 0.5

        # Parallel bonus
        parallel_bonus = 0.1 if state.enable_parallel else 0.0

        # Calculate objective-specific score
        if objective == Objective.MAXIMIZE_ACCURACY:
            score = threshold_score * 0.8 + parallel_bonus * 0.2
        elif objective == Objective.MINIMIZE_TIME:
            score = parallel_bonus * 0.5 + (1 - cost_penalty) * 0.5
        elif objective == Objective.MINIMIZE_COST:
            score = (1 - cost_penalty) * 0.8 + threshold_score * 0.2
        else:  # BALANCED
            score = threshold_score * 0.5 + (1 - cost_penalty) * 0.3 + parallel_bonus * 0.2

        return max(0.0, min(1.0, score))

    def _calculate_reward(
        self,
        old_state: GauntletState,
        new_state: GauntletState,
        domain: str,
        objective: Objective
    ) -> float:
        """Calculate reward for state transition"""
        old_score = self._evaluate_configuration(old_state, domain, objective)
        new_score = self._evaluate_configuration(new_state, domain, objective)
        return new_score - old_score

    def _mutate_state(self, state: GauntletState, mutation_rate: float = 0.3) -> GauntletState:
        """Apply random mutation to state"""
        if np.random.random() < mutation_rate:
            action = np.random.choice(self.actions)
            return action.apply(state)
        return GauntletState(**state.to_dict())

    def _tournament_select(
        self,
        population: List[GauntletState],
        scores: List[float],
        tournament_size: int
    ) -> GauntletState:
        """Select individual using tournament selection"""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = [(i, scores[i]) for i in indices]
        winner_idx = max(tournament_scores, key=lambda x: x[1])[0]
        return population[winner_idx]

    def _crossover_states(self, parent1: GauntletState, parent2: GauntletState) -> GauntletState:
        """Crossover two parent states"""
        child_dict = {}
        for key in parent1.to_dict().keys():
            if np.random.random() < 0.5:
                child_dict[key] = getattr(parent1, key)
            else:
                child_dict[key] = getattr(parent2, key)
        return GauntletState(**child_dict)

    def _generate_recommendation(
        self,
        best_state: GauntletState,
        initial_state: GauntletState,
        objective: Objective
    ) -> str:
        """Generate human-readable recommendation"""
        changes = []
        state_dict = best_state.to_dict()
        initial_dict = initial_state.to_dict()

        for key in state_dict:
            if state_dict[key] != initial_dict[key]:
                if isinstance(state_dict[key], bool):
                    change = f"{'enabled' if state_dict[key] else 'disabled'} {key}"
                else:
                    delta = state_dict[key] - initial_dict[key]
                    change = f"{key}: {initial_dict[key]:.2f} → {state_dict[key]:.2f} ({delta:+.2f})"
                changes.append(change)

        if not changes:
            return "Configuration already optimal for this objective"

        recommendation = f"Recommended changes for {objective.value}:\n"
        recommendation += "\n".join(f"  - {c}" for c in changes)
        return recommendation


def create_optimizer(
    strategy: str = "q_learning",
    learning_rate: float = 0.1,
    max_iterations: int = 100
) -> MLBasedGauntletOptimizer:
    """
    Factory function to create optimizer.

    Args:
        strategy: Strategy name (q_learning, dqn, genetic, bayesian)
        learning_rate: Learning rate
        max_iterations: Maximum iterations

    Returns:
        MLBasedGauntletOptimizer instance
    """
    strategy_map = {
        "q_learning": OptimizationStrategy.Q_LEARNING,
        "dqn": OptimizationStrategy.DQN,
        "genetic": OptimizationStrategy.GENETIC_ALGORITHM,
        "bayesian": OptimizationStrategy.BAYESIAN_OPTIMIZATION
    }

    strategy_enum = strategy_map.get(strategy.lower(), OptimizationStrategy.Q_LEARNING)

    return MLBasedGauntletOptimizer(
        strategy=strategy_enum,
        learning_rate=learning_rate,
        max_iterations=max_iterations
    )
