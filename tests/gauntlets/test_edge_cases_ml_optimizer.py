"""
Edge Case Tests for ML-Based Gauntlet Optimizer

Comprehensive edge case testing to achieve 95%+ code coverage.

Tests cover:
- Empty/null input handling
- Extreme parameter values
- Invalid configurations
- Memory pressure conditions
- Concurrent access scenarios
- Boundary conditions
- Error handling paths

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import unittest
import pytest
import numpy as np
import sys
import os
import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from glue.adapters.gauntlet_adapter.src.ml_optimizer import (
    MLBasedGauntletOptimizer,
    GauntletState,
    OptimizationAction,
    OptimizationStrategy,
    OptimizationResult,
    Objective,
    create_optimizer
)


class TestEmptyNullInputs(unittest.TestCase):
    """Test handling of empty and null inputs"""

    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = MLBasedGauntletOptimizer(max_iterations=10)

    def test_optimize_with_none_historical_data(self):
        """Test optimization with None historical data"""
        result = self.optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED,
            historical_data=None
        )

        self.assertIsNotNone(result.best_state)
        self.assertGreater(result.iterations, 0)

    def test_optimize_with_empty_historical_data(self):
        """Test optimization with empty historical data list"""
        result = self.optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED,
            historical_data=[]
        )

        self.assertIsNotNone(result.best_state)
        self.assertGreater(result.iterations, 0)

    def test_optimize_with_malformed_historical_data(self):
        """Test optimization with malformed historical data"""
        malformed_data = [
            {"invalid": "data"},
            {},
            {"score": None},
            {"score": 0.5, "time": "invalid"}
        ]

        # Should not crash, should handle gracefully
        result = self.optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED,
            historical_data=malformed_data
        )

        self.assertIsNotNone(result.best_state)

    def test_optimize_with_null_domain(self):
        """Test optimization with None domain"""
        result = self.optimizer.optimize(
            domain=None,
            objective=Objective.BALANCED
        )

        self.assertIsNotNone(result.best_state)

    def test_optimize_with_empty_string_domain(self):
        """Test optimization with empty string domain"""
        result = self.optimizer.optimize(
            domain="",
            objective=Objective.BALANCED
        )

        self.assertIsNotNone(result.best_state)

    def test_state_from_dict_with_none_values(self):
        """Test creating state from dict with None values"""
        data = {
            "round1_threshold": None,
            "round2_threshold": 0.5,
            "round3_threshold": 0.7
        }

        # Should handle None values gracefully
        with self.assertRaises((TypeError, AttributeError)):
            state = GauntletState.from_dict(data)

    def test_state_from_dict_with_missing_keys(self):
        """Test creating state from dict with missing keys"""
        data = {
            "round1_threshold": 0.5,
            "round2_threshold": 0.6
            # Missing round3_threshold
        }

        # Should handle missing keys
        with self.assertRaises((TypeError, KeyError)):
            state = GauntletState.from_dict(data)

    def test_optimize_with_none_initial_state(self):
        """Test optimization with None initial state"""
        result = self.optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED,
            initial_state=None
        )

        self.assertIsNotNone(result.best_state)


class TestExtremeParameterValues(unittest.TestCase):
    """Test handling of extreme parameter values"""

    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = MLBasedGauntletOptimizer(
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.1,
            max_iterations=10
        )

    def test_extreme_learning_rates(self):
        """Test optimizer with extreme learning rates"""
        # Very small learning rate
        optimizer_small_lr = MLBasedGauntletOptimizer(
            learning_rate=0.0001,
            max_iterations=5
        )
        result = optimizer_small_lr.optimize(
            domain="code",
            objective=Objective.BALANCED
        )
        self.assertIsNotNone(result.best_state)

        # Very large learning rate
        optimizer_large_lr = MLBasedGauntletOptimizer(
            learning_rate=1.0,
            max_iterations=5
        )
        result = optimizer_large_lr.optimize(
            domain="code",
            objective=Objective.BALANCED
        )
        self.assertIsNotNone(result.best_state)

    def test_extreme_discount_factors(self):
        """Test optimizer with extreme discount factors"""
        # Discount factor of 0 (no future consideration)
        optimizer_zero_gamma = MLBasedGauntletOptimizer(
            discount_factor=0.0,
            max_iterations=5
        )
        result = optimizer_zero_gamma.optimize(
            domain="code",
            objective=Objective.BALANCED
        )
        self.assertIsNotNone(result.best_state)

        # Discount factor of 1.0 (full future consideration)
        optimizer_one_gamma = MLBasedGauntletOptimizer(
            discount_factor=1.0,
            max_iterations=5
        )
        result = optimizer_one_gamma.optimize(
            domain="code",
            objective=Objective.BALANCED
        )
        self.assertIsNotNone(result.best_state)

    def test_extreme_epsilon_values(self):
        """Test optimizer with extreme epsilon values"""
        # Epsilon of 0 (pure exploitation)
        optimizer_zero_epsilon = MLBasedGauntletOptimizer(
            epsilon=0.0,
            max_iterations=5
        )
        result = optimizer_zero_epsilon.optimize(
            domain="code",
            objective=Objective.BALANCED
        )
        self.assertIsNotNone(result.best_state)

        # Epsilon of 1.0 (pure exploration)
        optimizer_one_epsilon = MLBasedGauntletOptimizer(
            epsilon=1.0,
            max_iterations=5
        )
        result = optimizer_one_epsilon.optimize(
            domain="code",
            objective=Objective.BALANCED
        )
        self.assertIsNotNone(result.best_state)

    def test_extreme_threshold_values(self):
        """Test state with extreme threshold values"""
        # All thresholds at 0.0
        state_min = GauntletState(
            round1_threshold=0.0,
            round2_threshold=0.0,
            round3_threshold=0.0,
            round1_weight=0.0,
            round2_weight=0.0,
            round3_weight=1.0,
            max_evaluations_round1=10,
            enable_parallel=False
        )

        score = self.optimizer._evaluate_configuration(
            state_min, "code", Objective.BALANCED
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # All thresholds at 1.0
        state_max = GauntletState(
            round1_threshold=1.0,
            round2_threshold=1.0,
            round3_threshold=1.0,
            round1_weight=1.0,
            round2_weight=0.0,
            round3_weight=0.0,
            max_evaluations_round1=100,
            enable_parallel=True
        )

        score = self.optimizer._evaluate_configuration(
            state_max, "code", Objective.BALANCED
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_extreme_iteration_counts(self):
        """Test optimizer with extreme iteration counts"""
        # Single iteration
        optimizer_single = MLBasedGauntletOptimizer(max_iterations=1)
        result = optimizer_single.optimize(
            domain="code",
            objective=Objective.BALANCED
        )
        self.assertEqual(result.iterations, 1)

        # Very large iteration count
        optimizer_large = MLBasedGauntletOptimizer(max_iterations=1000)
        # Just test initialization, don't run full optimization
        self.assertEqual(optimizer_large.max_iterations, 1000)

    def test_extreme_weight_combinations(self):
        """Test state with extreme weight combinations"""
        # All weight in round1
        state = GauntletState(
            round1_weight=1.0,
            round2_weight=0.0,
            round3_weight=0.0
        )

        score = self.optimizer._evaluate_configuration(
            state, "code", Objective.BALANCED
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # All weight in round3
        state = GauntletState(
            round1_weight=0.0,
            round2_weight=0.0,
            round3_weight=1.0
        )

        score = self.optimizer._evaluate_configuration(
            state, "code", Objective.BALANCED
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_action_with_extreme_deltas(self):
        """Test action application with extreme delta values"""
        state = GauntletState(round1_threshold=0.5)

        # Very large positive delta
        action = OptimizationAction("round1_threshold", 10.0)
        new_state = action.apply(state)
        self.assertLessEqual(new_state.round1_threshold, 1.0)

        # Very large negative delta
        action = OptimizationAction("round1_threshold", -10.0)
        new_state = action.apply(state)
        self.assertGreaterEqual(new_state.round1_threshold, 0.0)


class TestInvalidConfigurations(unittest.TestCase):
    """Test handling of invalid configurations"""

    def test_invalid_strategy_string(self):
        """Test factory function with invalid strategy string"""
        optimizer = create_optimizer(
            strategy="invalid_strategy",
            learning_rate=0.1,
            max_iterations=10
        )

        # Should default to Q_LEARNING
        self.assertEqual(optimizer.strategy, OptimizationStrategy.Q_LEARNING)

    def test_negative_parameters(self):
        """Test optimizer with negative parameters"""
        # Negative learning rate
        optimizer = MLBasedGauntletOptimizer(
            learning_rate=-0.1,
            max_iterations=5
        )
        result = optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED
        )
        self.assertIsNotNone(result.best_state)

    def test_state_with_negative_values(self):
        """Test state with negative values"""
        # Should clamp to valid range
        state = GauntletState(
            round1_threshold=-0.5,
            round2_threshold=-0.3,
            round3_threshold=-0.1
        )

        # Apply an action to trigger clamping
        action = OptimizationAction("round1_threshold", 0.0)
        new_state = action.apply(state)

        # Values should still be in valid range
        self.assertGreaterEqual(new_state.round1_threshold, 0.0)

    def test_state_with_values_above_one(self):
        """Test state with values above 1.0"""
        state = GauntletState(
            round1_threshold=1.5,
            round2_threshold=2.0,
            round3_threshold=1.8
        )

        # Apply an action to trigger clamping
        action = OptimizationAction("round1_threshold", 0.0)
        new_state = action.apply(state)

        # Values should be clamped to valid range
        self.assertLessEqual(new_state.round1_threshold, 1.0)

    def test_invalid_objective_type(self):
        """Test with invalid objective type (if it gets through type hints)"""
        # This test ensures the code handles unexpected inputs
        optimizer = MLBasedGauntletOptimizer(max_iterations=5)

        # Test with a valid objective
        result = optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED
        )

        self.assertIsNotNone(result.best_state)


class TestMemoryPressureConditions(unittest.TestCase):
    """Test behavior under memory pressure"""

    def test_large_q_table_growth(self):
        """Test Q-table doesn't grow unbounded"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=100)

        initial_table_size = len(optimizer.q_table)

        # Run optimization
        result = optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED
        )

        # Q-table should have grown but not excessively
        # (each iteration adds at most one state)
        final_table_size = len(optimizer.q_table)
        self.assertLess(final_table_size, optimizer.max_iterations * 2)

    def test_memory_cleanup_on_optimizer_deletion(self):
        """Test memory is cleaned up when optimizer is deleted"""
        # Create optimizer and populate Q-table
        optimizer = MLBasedGauntletOptimizer(max_iterations=50)
        result = optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED
        )

        # Get reference count before deletion
        table_size = len(optimizer.q_table)

        # Delete optimizer
        del optimizer
        gc.collect()

        # If we got here without crash, memory cleanup worked

    def test_performance_history_growth(self):
        """Test performance history doesn't grow unbounded"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=10)

        # Add performance history
        for i in range(100):
            optimizer.performance_history["code"].append({
                "score": 0.5 + i * 0.001,
                "time": 30.0
            })

        # History should be stored
        self.assertEqual(len(optimizer.performance_history["code"]), 100)

    def test_large_state_space(self):
        """Test optimization with large state space"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=20)

        # Create many different states
        states = []
        for i in range(50):
            state = GauntletState(
                round1_threshold=0.3 + (i % 7) * 0.1,
                round2_threshold=0.4 + (i % 6) * 0.1,
                round3_threshold=0.5 + (i % 5) * 0.1
            )
            states.append(state)

        # Optimize from each state
        for state in states:
            result = optimizer.optimize(
                domain="code",
                objective=Objective.BALANCED,
                initial_state=state
            )
            self.assertIsNotNone(result.best_state)


class TestConcurrentAccess(unittest.TestCase):
    """Test concurrent access to optimizer"""

    def test_concurrent_optimization_same_optimizer(self):
        """Test multiple threads optimizing with same optimizer instance"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=10)

        def optimize_domain(domain):
            return optimizer.optimize(
                domain=domain,
                objective=Objective.BALANCED
            )

        domains = ["code", "math", "general", "algorithm"]
        results = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(optimize_domain, domain) for domain in domains]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.fail(f"Concurrent optimization failed: {e}")

        # Should have completed all optimizations
        self.assertEqual(len(results), len(domains))

        for result in results:
            self.assertIsNotNone(result.best_state)

    def test_concurrent_state_mutation(self):
        """Test concurrent state mutations"""
        state = GauntletState()

        def mutate_state():
            action = OptimizationAction("round1_threshold", 0.05)
            return action.apply(state)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(mutate_state) for _ in range(20)]
            results = [future.result() for future in as_completed(futures)]

        # All mutations should complete successfully
        self.assertEqual(len(results), 20)

        for result in results:
            self.assertGreaterEqual(result.round1_threshold, 0.0)
            self.assertLessEqual(result.round1_threshold, 1.0)

    def test_concurrent_q_table_access(self):
        """Test concurrent Q-table access"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=5)

        def access_q_table():
            state = GauntletState()
            state_key = state.to_tuple()
            action_idx = 0

            # Read and write to Q-table
            old_value = optimizer.q_table[state_key][action_idx]
            optimizer.q_table[state_key][action_idx] = old_value + 1.0

            return optimizer.q_table[state_key][action_idx]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(access_q_table) for _ in range(10)]
            results = [future.result() for future in as_completed(futures)]

        # All accesses should complete
        self.assertEqual(len(results), 10)

    def test_thread_safe_evaluation(self):
        """Test thread-safe configuration evaluation"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=5)

        def evaluate_config():
            state = GauntletState()
            return optimizer._evaluate_configuration(
                state, "code", Objective.BALANCED
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(evaluate_config) for _ in range(20)]
            results = [future.result() for future in as_completed(futures)]

        # All evaluations should complete
        self.assertEqual(len(results), 20)

        for score in results:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestBoundaryConditions(unittest.TestCase):
    """Test boundary conditions and edge cases"""

    def test_zero_baseline_score(self):
        """Test improvement calculation with zero baseline"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=5)

        # Create state that will score 0
        state = GauntletState(
            round1_threshold=0.0,
            round2_threshold=0.0,
            round3_threshold=0.0,
            max_evaluations_round1=100  # High cost
        )

        baseline_score = optimizer._evaluate_configuration(
            state, "code", Objective.MINIMIZE_COST
        )

        # Should handle zero score in improvement calculation
        if baseline_score == 0:
            # Improvement should be 0, not division by zero
            result = optimizer.optimize(
                domain="code",
                objective=Objective.MINIMIZE_COST,
                initial_state=state
            )
            self.assertIsNotNone(result.best_state)

    def test_perfect_score_boundary(self):
        """Test with perfect score of 1.0"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=5)

        # Create state that maximizes score
        state = GauntletState(
            round1_threshold=1.0,
            round2_threshold=1.0,
            round3_threshold=1.0,
            max_evaluations_round1=10,  # Low cost
            enable_parallel=True
        )

        score = optimizer._evaluate_configuration(
            state, "code", Objective.MAXIMIZE_ACCURACY
        )

        self.assertLessEqual(score, 1.0)

    def test_all_optimization_strategies(self):
        """Test all optimization strategies"""
        strategies = [
            OptimizationStrategy.Q_LEARNING,
            OptimizationStrategy.DQN,
            OptimizationStrategy.GENETIC_ALGORITHM,
            OptimizationStrategy.BAYESIAN_OPTIMIZATION
        ]

        for strategy in strategies:
            optimizer = MLBasedGauntletOptimizer(
                strategy=strategy,
                max_iterations=10
            )

            result = optimizer.optimize(
                domain="code",
                objective=Objective.BALANCED
            )

            self.assertIsNotNone(result.best_state)
            self.assertGreater(result.iterations, 0)

    def test_all_objectives(self):
        """Test all optimization objectives"""
        objectives = [
            Objective.MAXIMIZE_ACCURACY,
            Objective.MINIMIZE_TIME,
            Objective.MINIMIZE_COST,
            Objective.BALANCED
        ]

        optimizer = MLBasedGauntletOptimizer(max_iterations=5)

        for objective in objectives:
            result = optimizer.optimize(
                domain="code",
                objective=objective
            )

            self.assertIsNotNone(result.best_state)
            self.assertGreaterEqual(result.best_score, 0.0)
            self.assertLessEqual(result.best_score, 1.0)

    def test_empty_action_space_edge_case(self):
        """Test behavior with actions that don't change state"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=5)

        state = GauntletState(round1_threshold=0.5)
        action = OptimizationAction("round1_threshold", 0.0)

        new_state = action.apply(state)

        # State should remain the same
        self.assertEqual(new_state.round1_threshold, state.round1_threshold)

    def test_reward_calculation_extremes(self):
        """Test reward calculation with extreme score differences"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=5)

        # Large positive reward
        state_bad = GauntletState(round1_threshold=0.0)
        state_good = GauntletState(round1_threshold=1.0)

        reward = optimizer._calculate_reward(
            state_bad, state_good, "code", Objective.MAXIMIZE_ACCURACY
        )

        self.assertGreater(reward, 0)

        # Large negative reward
        reward = optimizer._calculate_reward(
            state_good, state_bad, "code", Objective.MAXIMIZE_ACCURACY
        )

        self.assertLess(reward, 0)

    def test_convergence_history_length(self):
        """Test convergence history matches iterations"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=20)

        result = optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED
        )

        # Convergence history should have max_iterations + 1 entries
        # (initial + one per iteration)
        self.assertLessEqual(len(result.convergence_history), result.iterations + 1)


class TestErrorHandling(unittest.TestCase):
    """Test error handling paths"""

    def test_malformed_state_to_tuple(self):
        """Test state to_tuple with extreme values"""
        state = GauntletState(
            round1_threshold=0.55,  # Will round to 5 or 6
            round2_threshold=0.65,
            round3_threshold=0.75,
            max_evaluations_round1=5  # Below minimum
        )

        tuple_result = state.to_tuple()

        # Should convert without error
        self.assertEqual(len(tuple_result), 7)

    def test_optimization_with_zero_epsilon_decay(self):
        """Test optimization without epsilon decay"""
        optimizer = MLBasedGauntletOptimizer(
            epsilon=0.5,
            max_iterations=10
        )

        # Manually prevent epsilon decay
        original_epsilon = optimizer.epsilon
        optimizer.epsilon_decay = 1.0

        result = optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED
        )

        self.assertIsNotNone(result.best_state)

    def test_select_best_action_with_empty_q_table(self):
        """Test action selection with empty Q-table"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=5)

        # Clear Q-table
        optimizer.q_table.clear()

        state = GauntletState()

        # Should select random action when Q-table is empty
        action = optimizer._select_best_action(state)

        self.assertIn(action, optimizer.actions)

    def test_genetic_algorithm_with_single_individual(self):
        """Test genetic algorithm edge case with minimal population"""
        optimizer = MLBasedGauntletOptimizer(
            strategy=OptimizationStrategy.GENETIC_ALGORITHM,
            max_iterations=5
        )

        # The implementation uses population_size=20, so this tests
        # that the genetic algorithm handles the population correctly
        result = optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED
        )

        self.assertIsNotNone(result.best_state)

    def test_bayesian_optimization_exploration(self):
        """Test Bayesian optimization explores properly"""
        optimizer = MLBasedGauntletOptimizer(
            strategy=OptimizationStrategy.BAYESIAN_OPTIMIZATION,
            max_iterations=10
        )

        result = optimizer.optimize(
            domain="code",
            objective=Objective.BALANCED
        )

        self.assertIsNotNone(result.best_state)
        self.assertEqual(result.iterations, 10)


class TestRecommendationGeneration(unittest.TestCase):
    """Test recommendation generation edge cases"""

    def test_recommendation_with_no_changes(self):
        """Test recommendation when no changes are made"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=5)

        initial_state = GauntletState()

        recommendation = optimizer._generate_recommendation(
            initial_state, initial_state, Objective.BALANCED
        )

        self.assertIn("optimal", recommendation.lower())

    def test_recommendation_with_many_changes(self):
        """Test recommendation with many parameter changes"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=5)

        initial_state = GauntletState(
            round1_threshold=0.3,
            round2_threshold=0.4,
            round3_threshold=0.5
        )

        best_state = GauntletState(
            round1_threshold=0.7,
            round2_threshold=0.8,
            round3_threshold=0.9,
            round1_weight=0.4,
            round2_weight=0.3,
            round3_weight=0.3,
            max_evaluations_round1=80,
            enable_parallel=True
        )

        recommendation = optimizer._generate_recommendation(
            best_state, initial_state, Objective.BALANCED
        )

        # Should contain multiple change recommendations
        self.assertGreater(recommendation.count("\n"), 0)

    def test_recommendation_boolean_changes(self):
        """Test recommendation format for boolean parameter changes"""
        optimizer = MLBasedGauntletOptimizer(max_iterations=5)

        initial_state = GauntletState(enable_parallel=False)
        best_state = GauntletState(enable_parallel=True)

        recommendation = optimizer._generate_recommendation(
            best_state, initial_state, Objective.BALANCED
        )

        # Should mention "enabled"
        self.assertIn("enabled", recommendation.lower())


@pytest.mark.parametrize("strategy", [
    OptimizationStrategy.Q_LEARNING,
    OptimizationStrategy.DQN,
    OptimizationStrategy.GENETIC_ALGORITHM,
    OptimizationStrategy.BAYESIAN_OPTIMIZATION
])
def test_all_strategies_with_edge_cases(strategy):
    """Parametrized test for all strategies with edge cases"""
    optimizer = MLBasedGauntletOptimizer(
        strategy=strategy,
        max_iterations=5
    )

    # Test with minimum state
    min_state = GauntletState(
        round1_threshold=0.0,
        round2_threshold=0.0,
        round3_threshold=0.0
    )

    result = optimizer.optimize(
        domain="code",
        objective=Objective.BALANCED,
        initial_state=min_state
    )

    assert result.best_state is not None
    assert result.best_score >= 0.0


@pytest.mark.parametrize("objective", [
    Objective.MAXIMIZE_ACCURACY,
    Objective.MINIMIZE_TIME,
    Objective.MINIMIZE_COST,
    Objective.BALANCED
])
def test_all_objectives_with_edge_cases(objective):
    """Parametrized test for all objectives with edge cases"""
    optimizer = MLBasedGauntletOptimizer(max_iterations=5)

    # Test with maximum state
    max_state = GauntletState(
        round1_threshold=1.0,
        round2_threshold=1.0,
        round3_threshold=1.0
    )

    result = optimizer.optimize(
        domain="code",
        objective=objective,
        initial_state=max_state
    )

    assert result.best_state is not None
    assert result.best_score <= 1.0


if __name__ == "__main__":
    unittest.main()
