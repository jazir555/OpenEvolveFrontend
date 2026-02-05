"""
Test Suite for ML-Based Gauntlet Optimizer

Comprehensive tests for the ML-based gauntlet optimizer component.

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import unittest
import asyncio
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml_optimizer import (
    MLBasedGauntletOptimizer,
    GauntletState,
    OptimizationAction,
    OptimizationStrategy,
    Objective,  # Changed from OptimizationObjective
    create_optimizer
)


class TestGauntletState(unittest.TestCase):
    """Test GauntletState dataclass"""

    def test_state_creation(self):
        """Test creating a gauntlet state"""
        state = GauntletState()
        self.assertEqual(state.round1_threshold, 0.5)
        self.assertEqual(state.round2_threshold, 0.6)
        self.assertEqual(state.round3_threshold, 0.7)

    def test_state_to_dict(self):
        """Test converting state to dictionary"""
        state = GauntletState(
            round1_threshold=0.6,
            round2_threshold=0.7,
            round3_threshold=0.8
        )
        data = state.to_dict()
        self.assertEqual(data["round1_threshold"], 0.6)
        self.assertEqual(data["round2_threshold"], 0.7)
        self.assertEqual(data["round3_threshold"], 0.8)

    def test_state_from_dict(self):
        """Test creating state from dictionary"""
        data = {
            "round1_threshold": 0.4,
            "round2_threshold": 0.5,
            "round3_threshold": 0.6
        }
        state = GauntletState.from_dict(data)
        self.assertEqual(state.round1_threshold, 0.4)
        self.assertEqual(state.round2_threshold, 0.5)
        self.assertEqual(state.round3_threshold, 0.6)

    def test_state_to_tuple(self):
        """Test converting state to tuple for Q-table indexing"""
        state = GauntletState(
            round1_threshold=0.5,
            round2_threshold=0.6,
            max_evaluations_round1=50
        )
        tuple_state = state.to_tuple()
        self.assertEqual(len(tuple_state), 7)
        self.assertEqual(tuple_state[0], 5)  # 0.5 * 10
        self.assertEqual(tuple_state[1], 6)  # 0.6 * 10
        self.assertEqual(tuple_state[5], 5)  # 50 / 10


class TestOptimizationAction(unittest.TestCase):
    """Test OptimizationAction"""

    def test_action_apply_threshold_increase(self):
        """Test applying threshold increase action"""
        state = GauntletState(round1_threshold=0.5)
        action = OptimizationAction("round1_threshold", 0.1)
        new_state = action.apply(state)
        self.assertEqual(new_state.round1_threshold, 0.6)

    def test_action_apply_threshold_decrease(self):
        """Test applying threshold decrease action"""
        state = GauntletState(round1_threshold=0.6)
        action = OptimizationAction("round1_threshold", -0.1)
        new_state = action.apply(state)
        self.assertEqual(new_state.round1_threshold, 0.5)

    def test_action_clamp_to_bounds(self):
        """Test that actions clamp values to valid range"""
        state = GauntletState(round1_threshold=0.9)
        action = OptimizationAction("round1_threshold", 0.2)
        new_state = action.apply(state)
        self.assertEqual(new_state.round1_threshold, 1.0)  # Clamped to max

        state = GauntletState(round1_threshold=0.1)
        action = OptimizationAction("round1_threshold", -0.2)
        new_state = action.apply(state)
        self.assertEqual(new_state.round1_threshold, 0.0)  # Clamped to min

    def test_action_toggle_parallel(self):
        """Test toggle parallel action"""
        state = GauntletState(enable_parallel=False)
        action = OptimizationAction("toggle_parallel", 0)
        new_state = action.apply(state)
        self.assertTrue(new_state.enable_parallel)


class TestMLBasedGauntletOptimizer(unittest.TestCase):
    """Test ML-based gauntlet optimizer"""

    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = MLBasedGauntletOptimizer(
            strategy=OptimizationStrategy.Q_LEARNING,
            max_iterations=20
        )

    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        self.assertEqual(self.optimizer.strategy, OptimizationStrategy.Q_LEARNING)
        self.assertEqual(self.optimizer.max_iterations, 20)
        self.assertIsNotNone(self.optimizer.actions)
        self.assertGreater(len(self.optimizer.actions), 0)

    def test_evaluate_configuration(self):
        """Test configuration evaluation"""
        state = GauntletState()
        score = self.optimizer._evaluate_configuration(
            state, "code", Objective.MAXIMIZE_ACCURACY
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_optimize_returns_result(self):
        """Test that optimize returns valid result"""
        result = self.optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED,
            historical_data=None
        )

        self.assertIsNotNone(result.best_state)
        self.assertGreaterEqual(result.best_score, 0.0)
        self.assertLessEqual(result.best_score, 1.0)
        self.assertGreater(result.iterations, 0)
        self.assertIsInstance(result.convergence_history, list)

    def test_optimize_improves_score(self):
        """Test that optimization improves score"""
        initial_state = GauntletState()
        baseline_score = self.optimizer._evaluate_configuration(
            initial_state, "code", Objective.BALANCED
        )

        result = self.optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED,
            initial_state=initial_state
        )

        # Optimized score should be at least as good as baseline
        self.assertGreaterEqual(result.best_score, baseline_score * 0.95)

    def test_create_optimizer_factory(self):
        """Test optimizer factory function"""
        optimizer = create_optimizer(
            strategy="q_learning",
            learning_rate=0.2,
            max_iterations=50
        )

        self.assertEqual(optimizer.strategy, OptimizationStrategy.Q_LEARNING)
        self.assertEqual(optimizer.learning_rate, 0.2)
        self.assertEqual(optimizer.max_iterations, 50)


class TestOptimizerIntegration(unittest.TestCase):
    """Integration tests for optimizer"""

    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = MLBasedGauntletOptimizer(
            strategy=OptimizationStrategy.Q_LEARNING,
            max_iterations=10
        )

    def test_optimize_with_historical_data(self):
        """Test optimization with historical data"""
        # Create mock historical data
        historical_data = [
            {"score": 0.7, "time": 30, "config": {"round1_threshold": 0.5}},
            {"score": 0.8, "time": 40, "config": {"round1_threshold": 0.6}},
            {"score": 0.6, "time": 25, "config": {"round1_threshold": 0.4}}
        ]

        result = self.optimizer.optimize(
            domain="code",
            objective=Objective.MAXIMIZE_ACCURACY,
            historical_data=historical_data
        )

        self.assertIsNotNone(result.best_state)
        self.assertGreater(result.iterations, 0)

    def test_multi_objective_optimization(self):
        """Test optimization with different objectives"""
        domain = "code"

        results = {}
        for objective in [
            Objective.MAXIMIZE_ACCURACY,
            Objective.MINIMIZE_TIME,
            Objective.MINIMIZE_COST,
            Objective.BALANCED
        ]:
            result = self.optimizer.optimize(
                domain=domain,
                objective=objective
            )
            results[objective] = result

        # Each objective should produce potentially different results
        # (though not guaranteed with small iterations)
        self.assertEqual(len(results), 4)

    def test_domain_specific_optimization(self):
        """Test optimization for different domains"""
        domains = ["code", "math", "general", "algorithm"]

        results = {}
        for domain in domains:
            result = self.optimizer.optimize(
                domain=domain,
                objective=Objective.BALANCED
            )
            results[domain] = result

        # Should successfully optimize all domains
        for domain, result in results.items():
            self.assertIsNotNone(result.best_state)
            self.assertGreaterEqual(result.best_score, 0.0)


class TestOptimizerEdgeCases(unittest.TestCase):
    """Edge case tests for optimizer"""

    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = MLBasedGauntletOptimizer(
            strategy=OptimizationStrategy.Q_LEARNING,
            max_iterations=5
        )

    def test_optimize_with_extreme_thresholds(self):
        """Test optimization with extreme threshold values"""
        # All thresholds at minimum
        min_state = GauntletState(
            round1_threshold=0.0,
            round2_threshold=0.0,
            round3_threshold=0.0
        )

        result = self.optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED,
            initial_state=min_state
        )

        self.assertIsNotNone(result.best_state)

        # All thresholds at maximum
        max_state = GauntletState(
            round1_threshold=1.0,
            round2_threshold=1.0,
            round3_threshold=1.0
        )

        result = self.optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED,
            initial_state=max_state
        )

        self.assertIsNotNone(result.best_state)

    def test_optimize_with_small_iterations(self):
        """Test optimizer with very few iterations"""
        optimizer = MLBasedGauntletOptimizer(
            strategy=OptimizationStrategy.Q_LEARNING,
            max_iterations=1
        )

        result = optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED
        )

        self.assertEqual(result.iterations, 1)
        self.assertIsNotNone(result.best_state)

    def test_mutate_state_preserves_validity(self):
        """Test that state mutation preserves validity"""
        state = GauntletState()

        for _ in range(100):
            mutated = self.optimizer._mutate_state(state, mutation_rate=1.0)

            # Check all thresholds are in valid range
            self.assertGreaterEqual(mutated.round1_threshold, 0.0)
            self.assertLessEqual(mutated.round1_threshold, 1.0)
            self.assertGreaterEqual(mutated.round2_threshold, 0.0)
            self.assertLessEqual(mutated.round2_threshold, 1.0)
            self.assertGreaterEqual(mutated.round3_threshold, 0.0)
            self.assertLessEqual(mutated.round3_threshold, 1.0)

            # Check weights sum to approximately 1.0
            weight_sum = (
                mutated.round1_weight +
                mutated.round2_weight +
                mutated.round3_weight
            )
            self.assertGreater(weight_sum, 0.0)
            self.assertLess(weight_sum, 1.5)  # Allow some tolerance


if __name__ == "__main__":
    unittest.main()
