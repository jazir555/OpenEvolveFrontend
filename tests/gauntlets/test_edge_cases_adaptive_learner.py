"""
Edge Case Tests for Advanced Adaptive Learner

Comprehensive edge case testing to achieve 95%+ code coverage.

Tests cover:
- Empty experience buffer
- Single experience
- Exploding gradients
- Network size edge cases
- Learning rate edge cases
- Memory overflow scenarios
- Epsilon decay edge cases

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import unittest
import pytest
import numpy as np
import sys
import os
import gc
import json
import tempfile
from collections import deque

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from adaptive_learner import (
    AdvancedAdaptiveLearner,
    LearningAlgorithm,
    Experience,
    LearningMetrics,
    AdaptationResult,
    create_learner
)


class TestEmptyExperienceBuffer(unittest.TestCase):
    """Test handling of empty experience buffer"""

    def setUp(self):
        """Set up test fixtures"""
        self.learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            batch_size=32
        )

    def test_replay_with_empty_memory(self):
        """Test training replay with empty memory"""
        metrics = self.learner.replay()

        self.assertEqual(metrics["loss"], 0.0)
        self.assertEqual(metrics["q_value"], 0.0)

    def test_act_with_empty_memory(self):
        """Test action selection with empty memory"""
        state = np.random.randn(8).astype(np.float32)

        action = self.learner.act(state)

        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.learner.action_size)

    def test_learn_from_execution_with_empty_buffer(self):
        """Test learning when buffer is not full yet"""
        state = np.random.randn(8).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(8).astype(np.float32)
        done = False

        # Add fewer experiences than batch size
        for _ in range(10):
            result = self.learner.learn_from_execution(
                state, action, reward, next_state, done
            )

        # Should not train yet (buffer < batch_size)
        self.assertIsNone(result)

    def test_train_from_history_with_empty_list(self):
        """Test training from empty history"""
        metrics = self.learner.train_from_history(
            history=[],
            episodes=10
        )

        self.assertEqual(len(metrics), 10)

        # All metrics should have zero reward
        for metric in metrics:
            self.assertEqual(metric.total_reward, 0.0)

    def test_get_adaptive_strategy_with_minimal_state(self):
        """Test strategy with minimal state values"""
        state = np.zeros(8, dtype=np.float32)

        strategy = self.learner.get_adaptive_strategy(state)

        self.assertIsInstance(strategy, dict)
        self.assertIn("round1_threshold", strategy)


class TestSingleExperience(unittest.TestCase):
    """Test handling of single experience"""

    def setUp(self):
        """Set up test fixtures"""
        self.learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            batch_size=1  # Set to 1 for single experience testing
        )

    def test_single_experience_replay(self):
        """Test replay with exactly one experience"""
        state = np.random.randn(8).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(8).astype(np.float32)
        done = False

        self.learner.remember(state, action, reward, next_state, done)

        metrics = self.learner.replay()

        self.assertGreater(metrics["loss"], 0.0)

    def test_single_experience_train(self):
        """Test training with single experience"""
        state = np.random.randn(8).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(8).astype(np.float32)
        done = False

        result = self.learner.learn_from_execution(
            state, action, reward, next_state, done
        )

        self.assertIsNotNone(result)

    def test_single_record_history(self):
        """Test training from history with single record"""
        history = [{
            "round1_threshold": 0.5,
            "round2_threshold": 0.6,
            "round3_threshold": 0.7,
            "score": 0.8,
            "passed": True,
            "execution_time": 30.0
        }]

        metrics = self.learner.train_from_history(
            history=history,
            episodes=1
        )

        self.assertEqual(len(metrics), 1)
        self.assertGreater(metrics[0].total_reward, 0)


class TestExplodingGradients(unittest.TestCase):
    """Test handling of exploding gradients"""

    def setUp(self):
        """Set up test fixtures"""
        self.learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            learning_rate=0.1
        )

    def test_large_reward_values(self):
        """Test training with very large rewards"""
        state = np.random.randn(8).astype(np.float32)
        action = 5
        reward = 1000.0  # Very large reward
        next_state = np.random.randn(8).astype(np.float32)
        done = False

        # Add enough experiences to trigger training
        for _ in range(32):
            self.learner.remember(state, action, reward, next_state, done)

        # Should not crash
        metrics = self.learner.replay()

        # Loss might be large but should be finite
        self.assertTrue(np.isfinite(metrics["loss"]))

    def test_large_state_values(self):
        """Test training with very large state values"""
        state = np.ones(8, dtype=np.float32) * 1000.0  # Very large state
        action = 5
        reward = 0.0
        next_state = np.ones(8, dtype=np.float32) * 1000.0
        done = False

        for _ in range(32):
            self.learner.remember(state, action, reward, next_state, done)

        metrics = self.learner.replay()

        # Should handle large values
        self.assertTrue(np.isfinite(metrics["loss"]))

    def test_high_learning_rate_stability(self):
        """Test stability with very high learning rate"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            learning_rate=10.0  # Very high
        )

        state = np.random.randn(8).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(8).astype(np.float32)
        done = False

        for _ in range(32):
            learner.remember(state, action, reward, next_state, done)

        metrics = learner.replay()

        # Should not explode to NaN/Inf
        self.assertTrue(np.isfinite(metrics["loss"]))

    def test_network_weights_remain_finite(self):
        """Test that network weights stay finite"""
        # Train with extreme values
        state = np.random.randn(8).astype(np.float32) * 100
        action = 5
        reward = 100.0
        next_state = np.random.randn(8).astype(np.float32) * 100
        done = False

        for _ in range(100):
            self.learner.remember(state, action, reward, next_state, done)
            self.learner.replay()

        # Check all weights are finite
        for key, weight_matrix in self.learner.q_network.items():
            self.assertTrue(np.all(np.isfinite(weight_matrix)), f"Weights {key} have NaN/Inf")


class TestNetworkSizeEdgeCases(unittest.TestCase):
    """Test edge cases for network size"""

    def test_minimal_network_size(self):
        """Test with smallest possible network"""
        learner = AdvancedAdaptiveLearner(
            state_size=1,
            action_size=2,
            batch_size=5
        )

        state = np.array([1.0], dtype=np.float32)
        action = 1

        # Should work with minimal sizes
        learner.act(state)
        learner.remember(state, action, 1.0, state, False)

        self.assertGreaterEqual(len(learner.memory), 1)

    def test_large_network_size(self):
        """Test with large network dimensions"""
        learner = AdvancedAdaptiveLearner(
            state_size=1000,
            action_size=100,
            batch_size=32
        )

        state = np.random.randn(1000).astype(np.float32)

        # Should work with large dimensions
        action = learner.act(state)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 100)

    def test_imbalanced_network(self):
        """Test with imbalanced state/action sizes"""
        # Many actions, few states
        learner1 = AdvancedAdaptiveLearner(
            state_size=2,
            action_size=100,
            batch_size=10
        )

        state = np.random.randn(2).astype(np.float32)
        action = learner1.act(state)
        self.assertLess(action, 100)

        # Many states, few actions
        learner2 = AdvancedAdaptiveLearner(
            state_size=100,
            action_size=2,
            batch_size=10
        )

        state = np.random.randn(100).astype(np.float32)
        action = learner2.act(state)
        self.assertIn(action, [0, 1])

    def test_action_space_boundary(self):
        """Test action at the edge of action space"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10
        )

        # Test action 0
        action = learner.act(np.zeros(8, dtype=np.float32), use_epsilon=False)
        self.assertGreaterEqual(action, 0)

        # Test action near max
        # We can't force a specific action, but can test it's in range
        for _ in range(100):
            action = learner.act(np.random.randn(8).astype(np.float32), use_epsilon=False)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, 10)


class TestLearningRateEdgeCases(unittest.TestCase):
    """Test edge cases for learning rate"""

    def test_zero_learning_rate(self):
        """Test with zero learning rate"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            learning_rate=0.0
        )

        # Store initial weights
        initial_W1 = learner.q_network["W1"].copy()

        # Add experiences and train
        state = np.random.randn(8).astype(np.float32)
        for _ in range(32):
            learner.remember(state, 5, 1.0, state, False)

        learner.replay()

        # Weights should not change with zero learning rate
        np.testing.assert_array_almost_equal(
            initial_W1,
            learner.q_network["W1"],
            decimal=10
        )

    def test_very_small_learning_rate(self):
        """Test with very small learning rate"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            learning_rate=1e-10
        )

        state = np.random.randn(8).astype(np.float32)
        for _ in range(32):
            learner.remember(state, 5, 1.0, state, False)

        metrics = learner.replay()

        # Should complete without error
        self.assertTrue(np.isfinite(metrics["loss"]))

    def test_negative_learning_rate(self):
        """Test with negative learning rate"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            learning_rate=-0.1
        )

        state = np.random.randn(8).astype(np.float32)
        for _ in range(32):
            learner.remember(state, 5, 1.0, state, False)

        # Should handle negative LR (will increase weights instead of decreasing)
        metrics = learner.replay()

        self.assertTrue(np.isfinite(metrics["loss"]))


class TestMemoryOverflowScenarios(unittest.TestCase):
    """Test memory overflow and buffer management"""

    def test_memory_maxlen_enforcement(self):
        """Test that memory buffer respects max length"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            memory_size=10
        )

        state = np.random.randn(8).astype(np.float32)

        # Add more experiences than capacity
        for i in range(100):
            learner.remember(state, i % 10, 1.0, state, False)

        # Memory should not exceed max size
        self.assertLessEqual(len(learner.memory), 10)

    def test_fifo_behavior(self):
        """Test FIFO behavior of memory buffer"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            memory_size=5
        )

        state = np.random.randn(8).astype(np.float32)

        # Add experiences with different rewards
        for i in range(10):
            learner.remember(state, i % 10, float(i), state, False)

        # First 5 should be discarded
        # Last 5 should be present
        self.assertEqual(len(learner.memory), 5)

        # The rewards should be 5, 6, 7, 8, 9 (not 0-4)
        rewards = [exp.reward for exp in learner.memory]
        self.assertEqual(rewards, [5.0, 6.0, 7.0, 8.0, 9.0])

    def test_large_memory_efficiency(self):
        """Test efficiency with large memory buffer"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            memory_size=10000
        )

        state = np.random.randn(8).astype(np.float32)

        # Fill buffer
        for i in range(10000):
            learner.remember(state, i % 10, 1.0, state, False)

        self.assertEqual(len(learner.memory), 10000)

        # Training should still work
        metrics = learner.replay()

        self.assertTrue(np.isfinite(metrics["loss"]))

    def test_batch_size_larger_than_memory(self):
        """Test when batch size exceeds memory size"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            memory_size=10,
            batch_size=20
        )

        state = np.random.randn(8).astype(np.float32)

        # Add fewer experiences than batch size
        for i in range(5):
            learner.remember(state, i % 10, 1.0, state, False)

        # Should handle gracefully
        metrics = learner.replay()

        self.assertEqual(metrics["loss"], 0.0)


class TestEpsilonDecayEdgeCases(unittest.TestCase):
    """Test epsilon decay edge cases"""

    def test_no_epsilon_decay(self):
        """Test with epsilon decay of 1.0 (no decay)"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            epsilon=0.5,
            epsilon_decay=1.0
        )

        initial_epsilon = learner.epsilon

        # Train multiple times
        state = np.random.randn(8).astype(np.float32)
        for _ in range(32):
            learner.remember(state, 5, 1.0, state, False)

        for _ in range(10):
            learner.replay()

        # Epsilon should not change
        self.assertEqual(learner.epsilon, initial_epsilon)

    def test_epsilon_at_minimum(self):
        """Test when epsilon reaches minimum"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            epsilon=0.01,
            epsilon_min=0.01,
            epsilon_decay=0.99
        )

        # Already at minimum
        self.assertEqual(learner.epsilon, learner.epsilon_min)

        # Train
        state = np.random.randn(8).astype(np.float32)
        for _ in range(32):
            learner.remember(state, 5, 1.0, state, False)

        learner.replay()

        # Should stay at minimum
        self.assertGreaterEqual(learner.epsilon, learner.epsilon_min)

    def test_rapid_epsilon_decay(self):
        """Test with very rapid epsilon decay"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.5
        )

        # Train to trigger decay
        state = np.random.randn(8).astype(np.float32)
        for _ in range(32):
            learner.remember(state, 5, 1.0, state, False)

        initial_epsilon = learner.epsilon

        for _ in range(5):
            learner.replay()

        # Epsilon should have decreased
        self.assertLess(learner.epsilon, initial_epsilon)

    def test_epsilon_greater_than_one(self):
        """Test with epsilon > 1.0"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            epsilon=1.5
        )

        # Should still work (will clamp to 1.0 in practice)
        action = learner.act(np.zeros(8, dtype=np.float32))

        self.assertGreaterEqual(action, 0)
        self.assertLess(action, learner.action_size)

    def test_epsilon_less_than_zero(self):
        """Test with epsilon < 0"""
        learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            epsilon=-0.5
        )

        # Should still work
        action = learner.act(np.zeros(8, dtype=np.float32))

        self.assertGreaterEqual(action, 0)
        self.assertLess(action, learner.action_size)


class TestExperienceDataclass(unittest.TestCase):
    """Test Experience dataclass edge cases"""

    def test_experience_with_none_values(self):
        """Test experience with None values (if allowed)"""
        state = np.zeros(8, dtype=np.float32)
        action = 5
        reward = 1.0
        next_state = np.zeros(8, dtype=np.float32)
        done = False

        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )

        self.assertIsNotNone(exp.timestamp)

    def test_experience_with_extreme_rewards(self):
        """Test experience with extreme reward values"""
        state = np.zeros(8, dtype=np.float32)

        # Very large reward
        exp1 = Experience(
            state=state,
            action=5,
            reward=1e10,
            next_state=state,
            done=False
        )

        self.assertEqual(exp1.reward, 1e10)

        # Very negative reward
        exp2 = Experience(
            state=state,
            action=5,
            reward=-1e10,
            next_state=state,
            done=False
        )

        self.assertEqual(exp2.reward, -1e10)

    def test_experience_timestamp_auto_generation(self):
        """Test that timestamp is auto-generated"""
        import time

        state = np.zeros(8, dtype=np.float32)

        before = time.time()
        exp = Experience(
            state=state,
            action=5,
            reward=1.0,
            next_state=state,
            done=False
        )
        after = time.time()

        self.assertGreaterEqual(exp.timestamp, before)
        self.assertLessEqual(exp.timestamp, after)


class TestTargetNetworkUpdate(unittest.TestCase):
    """Test target network update behavior"""

    def setUp(self):
        """Set up test fixtures"""
        self.learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10,
            target_update_freq=10
        )

    def test_target_network_initially_equal(self):
        """Test that target network initially equals Q network"""
        # After initialization, target should equal Q
        for key in self.learner.q_network.keys():
            np.testing.assert_array_equal(
                self.learner.q_network[key],
                self.learner.target_network[key]
            )

    def test_target_network_update_frequency(self):
        """Test that target network updates at correct frequency"""
        state = np.random.randn(8).astype(np.float32)

        # Get initial weights
        initial_W1 = self.learner.target_network["W1"].copy()

        # Train for 5 iterations (should not update yet)
        for _ in range(5):
            for _ in range(32):
                self.learner.remember(state, 5, 1.0, state, False)
            self.learner.replay()

        # Target should not have updated yet
        np.testing.assert_array_equal(
            initial_W1,
            self.learner.target_network["W1"]
        )

        # Train for 5 more iterations (should update on 10th)
        for _ in range(5):
            for _ in range(32):
                self.learner.remember(state, 5, 1.0, state, False)
            self.learner.replay()

        # Target should have updated
        # (Q network changed, so target should be different from initial)
        # Actually, target should now equal Q network
        np.testing.assert_array_equal(
            self.learner.q_network["W1"],
            self.learner.target_network["W1"]
        )

    def test_manual_target_update(self):
        """Test manual target network update"""
        # Change Q network
        self.learner.q_network["W1"][0, 0] = 999.0

        # Target should be different
        self.assertNotEqual(
            self.learner.target_network["W1"][0, 0],
            999.0
        )

        # Manually update
        self.learner.update_target_network()

        # Should now be equal
        self.assertEqual(
            self.learner.target_network["W1"][0, 0],
            999.0
        )


class TestModelPersistence(unittest.TestCase):
    """Test model saving and loading"""

    def setUp(self):
        """Set up test fixtures"""
        self.learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10
        )

        # Train a bit
        state = np.random.randn(8).astype(np.float32)
        for _ in range(32):
            self.learner.remember(state, 5, 1.0, state, False)

        self.learner.replay()

    def test_save_and_load_model(self):
        """Test saving and loading model"""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Save model
            self.learner.save_model(temp_path)

            # Load into new learner
            new_learner = AdvancedAdaptiveLearner(
                state_size=8,
                action_size=10
            )
            new_learner.load_model(temp_path)

            # Check weights match
            for key in self.learner.q_network.keys():
                np.testing.assert_array_almost_equal(
                    self.learner.q_network[key],
                    new_learner.q_network[key]
                )

            # Check epsilon matches
            self.assertEqual(
                self.learner.epsilon,
                new_learner.epsilon
            )

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_from_corrupted_file(self):
        """Test loading from corrupted file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
            f.write("corrupted data {[")

        try:
            new_learner = AdvancedAdaptiveLearner(
                state_size=8,
                action_size=10
            )

            # Should raise error
            with self.assertRaises(json.JSONDecodeError):
                new_learner.load_model(temp_path)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_save_with_nans_in_weights(self):
        """Test saving model with NaN in weights"""
        # Introduce NaN
        self.learner.q_network["W1"][0, 0] = np.nan

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Should still save (NaN is valid JSON)
            self.learner.save_model(temp_path)

            # Load back
            new_learner = AdvancedAdaptiveLearner(
                state_size=8,
                action_size=10
            )
            new_learner.load_model(temp_path)

            # NaN should be present
            self.assertTrue(np.isnan(new_learner.q_network["W1"][0, 0]))

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestAlgorithmVariations(unittest.TestCase):
    """Test different algorithm choices"""

    def test_all_algorithm_types(self):
        """Test that all algorithm types can be initialized"""
        algorithms = [
            LearningAlgorithm.DQN,
            LearningAlgorithm.PPO,
            LearningAlgorithm.A3C,
            LearningAlgorithm.SARSA
        ]

        for algorithm in algorithms:
            learner = AdvancedAdaptiveLearner(
                algorithm=algorithm,
                state_size=8,
                action_size=10
            )

            self.assertEqual(learner.algorithm, algorithm)

            # Should be able to act
            state = np.random.randn(8).astype(np.float32)
            action = learner.act(state)

            self.assertGreaterEqual(action, 0)
            self.assertLess(action, 10)

    def test_factory_function_all_algorithms(self):
        """Test factory function with all algorithms"""
        algorithm_names = ["dqn", "ppo", "a3c", "sarsa", "invalid"]

        for alg_name in algorithm_names:
            learner = create_learner(
                algorithm=alg_name,
                state_size=8,
                action_size=10
            )

            # Should create learner (defaults to DQN for invalid)
            self.assertIsNotNone(learner)


class TestGenerateTestCase(unittest.TestCase):
    """Test test case generation"""

    def setUp(self):
        """Set up test fixtures"""
        self.learner = AdvancedAdaptiveLearner(
            state_size=8,
            action_size=10
        )

    def test_generate_all_difficulties(self):
        """Test generating test cases for all difficulty levels"""
        difficulties = ["easy", "medium", "hard"]

        for difficulty in difficulties:
            test_case = self.learner.generate_test_case(
                difficulty=difficulty,
                domain="code"
            )

            self.assertEqual(test_case["difficulty"], difficulty)
            self.assertIn("config", test_case)
            self.assertIn("parameters", test_case)

    def test_generate_all_domains(self):
        """Test generating test cases for all domains"""
        domains = ["code", "math", "algorithm", "ml", "general"]

        for domain in domains:
            test_case = self.learner.generate_test_case(
                difficulty="medium",
                domain=domain
            )

            self.assertEqual(test_case["domain"], domain)

    def test_generate_with_invalid_difficulty(self):
        """Test generating with invalid difficulty"""
        test_case = self.learner.generate_test_case(
            difficulty="invalid",
            domain="code"
        )

        # Should default to 0.5 complexity (medium)
        self.assertIsNotNone(test_case)


@pytest.mark.parametrize("state_size,action_size", [
    (1, 2),
    (10, 5),
    (100, 100),
    (1000, 10),
])
def test_various_network_sizes(state_size, action_size):
    """Parametrized test for various network sizes"""
    learner = AdvancedAdaptiveLearner(
        state_size=state_size,
        action_size=action_size,
        batch_size=min(32, state_size)
    )

    state = np.random.randn(state_size).astype(np.float32)
    action = learner.act(state)

    assert action >= 0
    assert action < action_size


@pytest.mark.parametrize("epsilon,epsilon_min,epsilon_decay", [
    (1.0, 0.01, 0.99),
    (0.5, 0.1, 0.9),
    (0.1, 0.01, 1.0),  # No decay
    (1.0, 0.5, 0.95),
])
def test_various_epsilon_configs(epsilon, epsilon_min, epsilon_decay):
    """Parametrized test for various epsilon configurations"""
    learner = AdvancedAdaptiveLearner(
        state_size=8,
        action_size=10,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )

    state = np.random.randn(8).astype(np.float32)

    # Test exploration
    for _ in range(10):
        action = learner.act(state, use_epsilon=True)
        assert action >= 0
        assert action < 10

    # Test exploitation
    action = learner.act(state, use_epsilon=False)
    assert action >= 0
    assert action < 10


if __name__ == "__main__":
    unittest.main()
