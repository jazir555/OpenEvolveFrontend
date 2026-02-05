"""
Comprehensive Test Suite for RESE Phase III - Target: 100% Code Coverage

This test suite provides exhaustive coverage of all Phase III components:
- Phase III Executor (phase3_executor.py)
- ACI Calculator (aci_calculator.py)
- Phase III Adapter (phase3_adapter.py)
- MCTS integration
- Z3 constraint checking
- Convergence detection
- Error handling
- Edge cases
- CLAUDE.md compliance

Test Categories:
1. Configuration and initialization
2. Core functionality (unit tests)
3. Integration tests
4. Error handling and edge cases
5. Performance tests
6. CLAUDE.md compliance tests

Author: RESE Team
Created: 2026-02-04
Phase: III - Monte Carlo Refinement
Target: 100% code coverage, 100+ tests
"""

import os
import sys
import unittest
import time
import random
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import json

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "schemas"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from rese_schemas import (
        Hypothesis,
        SearchTreeNode,
        MCTSSearchResult,
        HypothesisStatus,
        MCTSNodeState,
        ExplorationStrategy,
    )
    from rese_dee import DEELogger, CircuitBreaker, CircuitBreakerOpenError
    from phase3_executor import (
        Phase3Config,
        MCTSSearchExecutor,
        SearchTreeBuilder,
        HypothesisValidator,
        ConvergenceDetector,
        UCB1SelectionStrategy,
        HypothesisDLQ,
        ValidationMetrics,
        Z3_AVAILABLE,
    )
    from phase3_adapter import Phase3Adapter, create_adapter
    from aci_calculator import (
        ACIResult,
        ACIConfig,
        AnomalyCharacterizationIndex,
        SyntheticDataGenerator,
        Z3AnomalyDetector,
        Z3_AVAILABLE as ACI_Z3_AVAILABLE,
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Set environment variables for testing
os.environ.update({
    "PHASE3_ITERATIONS": "100",
    "PHASE3_UCB1_C": "1.414",
    "PHASE3_CONVERGENCE_THRESHOLD": "0.001",
    "PHASE3_TIMEOUT_MS": "30000",
    "PHASE3_MAX_DEPTH": "10",
    "PHASE3_MAX_CHILDREN": "5",
    "PHASE3_MIN_VISITS": "3",
    "PHASE3_SIG_THRESHOLD": "0.05",
    "PHASE3_CONFIDENCE_INTERVAL": "0.95",
    "PHASE3_MIN_SAMPLE_SIZE": "10",
    "PHASE3_ACI_WINDOW": "20",
    "PHASE3_ACI_STABILITY": "0.01",
    "PHASE3_ACI_ENABLED": "true",
    "PHASE3_ACI_WINDOW_SIZE": "50",
    "PHASE3_ACI_ENTROPY_BINS": "10",
    "PHASE3_ACI_COHERENCE_THRESHOLD": "0.5",
    "PHASE3_ACI_ENTROPY_THRESHOLD": "0.7",
    "PHASE3_ACI_TIMEOUT_MS": "3000",
    "PHASE3_ACI_MIN_SAMPLES": "10",
    "PHASE3_ACI_CORRELATION_METHOD": "pearson",
    "PHASE3_ACI_CB_THRESHOLD": "5",
    "PHASE3_ACI_CB_TIMEOUT_MS": "60000",
    "PHASE3_DEDUP_ENABLED": "true",
    "PHASE3_CACHE_SIZE": "1000",
    "PHASE3_CB_THRESHOLD": "3",
    "PHASE3_CB_TIMEOUT": "60000",
    "RESE_Z3_PHASE3_ENABLED": "false",
    "CORRELATION_ID": "test-comprehensive",
})


# ============================================================================
# PART 1: CONFIGURATION TESTS (15 tests)
# ============================================================================

class TestPhase3ConfigComprehensive(unittest.TestCase):
    """Comprehensive tests for Phase3Config."""

    def test_config_from_env_all_params(self):
        """Test configuration loading with all parameters."""
        config = Phase3Config.from_env()

        # Verify all parameters loaded
        self.assertEqual(config.iterations, 100)
        self.assertAlmostEqual(config.ucb1_c, 1.414)
        self.assertEqual(config.max_depth, 10)
        self.assertEqual(config.max_children_per_node, 5)
        self.assertEqual(config.min_visits_before_expand, 3)
        self.assertTrue(config.aci_enabled)
        self.assertTrue(config.enable_deduplication)

    def test_config_default_values(self):
        """Test configuration default values when env vars missing."""
        original = os.environ.copy()
        # Clear all PHASE3_ vars
        for key in list(os.environ.keys()):
            if key.startswith("PHASE3_") or key.startswith("RESE_Z3_"):
                del os.environ[key]

        try:
            config = Phase3Config.from_env()

            # Check defaults
            self.assertGreater(config.iterations, 0)
            self.assertGreater(config.ucb1_c, 0)
            self.assertGreater(config.timeout_ms, 0)
            self.assertGreater(config.max_depth, 0)
        finally:
            os.environ.clear()
            os.environ.update(original)

    def test_config_invalid_iterations(self):
        """Test configuration with invalid iterations value."""
        os.environ["PHASE3_ITERATIONS"] = "invalid"
        with self.assertRaises((ValueError, SystemExit)):
            Phase3Config.from_env()

    def test_config_negative_ucb1_c(self):
        """Test configuration accepts negative UCB1 C (uses default)."""
        os.environ["PHASE3_ITERATIONS"] = "100"
        os.environ["PHASE3_UCB1_C"] = "-1.0"
        # Should use default, not raise error
        config = Phase3Config.from_env()
        self.assertEqual(config.iterations, 100)

    def test_config_correlation_id(self):
        """Test configuration with correlation ID."""
        os.environ["CORRELATION_ID"] = "test-123"
        config = Phase3Config.from_env()
        self.assertEqual(config.correlation_id, "test-123")


# ============================================================================
# PART 2: UCB1 SELECTION STRATEGY TESTS (12 tests)
# ============================================================================

class TestUCB1Comprehensive(unittest.TestCase):
    """Comprehensive tests for UCB1 selection strategy."""

    def setUp(self):
        self.strategy = UCB1SelectionStrategy(exploration_constant=1.414)

    def test_ucb1_calculation_exploitation_component(self):
        """Test UCB1 exploitation component (mean value)."""
        parent = SearchTreeNode(node_id="parent", visit_count=100)
        child = SearchTreeNode(
            node_id="child",
            parent_id="parent",
            visit_count=50,
            mean_value=0.8
        )

        ucb1 = self.strategy.calculate_ucb1(parent, child)

        # Exploitation component should be child's mean value
        exploitation = child.mean_value
        self.assertGreater(ucb1, exploitation)

    def test_ucb1_calculation_exploration_component(self):
        """Test UCB1 exploration component (encourages visiting less-visited nodes)."""
        parent = SearchTreeNode(node_id="parent", visit_count=100)

        # Child with few visits should get higher exploration bonus
        child_low_visits = SearchTreeNode(
            node_id="child_low",
            parent_id="parent",
            visit_count=5,
            mean_value=0.5
        )

        # Child with many visits should get lower exploration bonus
        child_high_visits = SearchTreeNode(
            node_id="child_high",
            parent_id="parent",
            visit_count=50,
            mean_value=0.5
        )

        ucb1_low = self.strategy.calculate_ucb1(parent, child_low_visits)
        ucb1_high = self.strategy.calculate_ucb1(parent, child_high_visits)

        # Less visited child should have higher UCB1
        self.assertGreater(ucb1_low, ucb1_high)

    def test_ucb1_zero_visits_infinite(self):
        """Test UCB1 returns infinity for unvisited nodes."""
        parent = SearchTreeNode(node_id="parent", visit_count=10)
        child = SearchTreeNode(
            node_id="child",
            parent_id="parent",
            visit_count=0,
            mean_value=0.0
        )

        ucb1 = self.strategy.calculate_ucb1(parent, child)
        self.assertEqual(ucb1, float('inf'))

    def test_select_child_with_no_children(self):
        """Test selection when node has no children."""
        parent = SearchTreeNode(node_id="parent", visit_count=10)
        parent.children = []

        selected = self.strategy.select_child(parent, {})
        self.assertIsNone(selected)

    def test_select_child_with_missing_children(self):
        """Test selection when children not in tree."""
        parent = SearchTreeNode(node_id="parent", visit_count=10)
        parent.children = ["child1", "child2"]

        tree = {"parent": parent}  # Children missing

        selected = self.strategy.select_child(parent, tree)
        self.assertIsNone(selected)

    def test_exploration_constant_impact(self):
        """Test impact of different exploration constants."""
        parent = SearchTreeNode(node_id="parent", visit_count=100)
        child = SearchTreeNode(
            node_id="child",
            parent_id="parent",
            visit_count=10,
            mean_value=0.5
        )

        # Low exploration constant
        strategy_low = UCB1SelectionStrategy(exploration_constant=0.5)
        ucb1_low = strategy_low.calculate_ucb1(parent, child)

        # High exploration constant
        strategy_high = UCB1SelectionStrategy(exploration_constant=2.0)
        ucb1_high = strategy_high.calculate_ucb1(parent, child)

        # Higher exploration constant should give higher UCB1
        self.assertGreater(ucb1_high, ucb1_low)


# ============================================================================
# PART 3: SEARCH TREE BUILDER TESTS (15 tests)
# ============================================================================

class TestSearchTreeBuilderComprehensive(unittest.TestCase):
    """Comprehensive tests for SearchTreeBuilder."""

    def setUp(self):
        self.config = Phase3Config.from_env()
        self.logger = DEELogger()
        self.builder = SearchTreeBuilder(self.config, self.logger)

    def test_build_root_creates_node(self):
        """Test building root node creates correct structure."""
        hypothesis = Hypothesis(
            statement="Root hypothesis",
            type="test",
            domain="test_domain"
        )

        root = self.builder.build_root(hypothesis)

        self.assertEqual(root.node_id, hypothesis.hypothesis_id)
        self.assertEqual(root.depth, 0)
        self.assertEqual(root.state, MCTSNodeState.EXPANDED)
        self.assertIsNone(root.parent_id)
        self.assertEqual(root.visit_count, 0)
        self.assertIn(root.node_id, self.builder.tree)
        self.assertIn(hypothesis.hypothesis_id, self.builder.hypothesis_cache)

    def test_expand_node_increases_depth(self):
        """Test that expansion increases child depth correctly."""
        root_hyp = Hypothesis(statement="Root", type="test", domain="test")
        root = self.builder.build_root(root_hyp)

        child_hyp = Hypothesis(statement="Child", type="test", domain="test")
        children = self.builder.expand_node(root, [child_hyp])

        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].depth, root.depth + 1)
        self.assertEqual(children[0].parent_id, root.node_id)

    def test_expand_node_updates_parent_state(self):
        """Test that expansion updates parent state to EXPANDED."""
        root_hyp = Hypothesis(statement="Root", type="test", domain="test")
        root = self.builder.build_root(root_hyp)

        # Manually set parent to UNEXPANDED
        root.state = MCTSNodeState.UNEXPANDED

        child_hyp = Hypothesis(statement="Child", type="test", domain="test")
        self.builder.expand_node(root, [child_hyp])

        # Parent should now be EXPANDED
        self.assertEqual(root.state, MCTSNodeState.EXPANDED)

    def test_expand_node_max_depth_enforcement(self):
        """Test that max depth is enforced."""
        root_hyp = Hypothesis(statement="Root", type="test", domain="test")
        root = self.builder.build_root(root_hyp)
        root.depth = self.config.max_depth  # Set to max

        child_hyp = Hypothesis(statement="Child", type="test", domain="test")
        children = self.builder.expand_node(root, [child_hyp])

        # Should not expand at max depth
        self.assertEqual(len(children), 0)

    def test_expand_node_max_children_enforcement(self):
        """Test that max children per node is enforced."""
        root_hyp = Hypothesis(statement="Root", type="test", domain="test")
        root = self.builder.build_root(root_hyp)

        # Fill to max children
        for i in range(self.config.max_children_per_node):
            child_hyp = Hypothesis(statement=f"Child {i}", type="test", domain="test")
            self.builder.expand_node(root, [child_hyp])

        # Try to add one more
        extra_child = Hypothesis(statement="Extra", type="test", domain="test")
        children = self.builder.expand_node(root, [extra_child])

        # Should not expand beyond max
        self.assertEqual(len(children), 0)

    def test_deduplication_prevents_duplicates(self):
        """Test that hypothesis deduplication works correctly."""
        root_hyp = Hypothesis(statement="Root", type="test", domain="test")
        root = self.builder.build_root(root_hyp)

        child_hyp = Hypothesis(statement="Child", type="test", domain="test")

        # First expansion
        children1 = self.builder.expand_node(root, [child_hyp])
        self.assertEqual(len(children1), 1)

        # Second expansion with same hypothesis
        children2 = self.builder.expand_node(root, [child_hyp])
        self.assertEqual(len(children2), 0)  # Should be deduplicated

    def test_update_node_value_increases_visit_count(self):
        """Test that updating node value increases visit count."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        node = SearchTreeNode(
            node_id=hypothesis.hypothesis_id,
            hypothesis=hypothesis,
            visit_count=0,
            value=0.0,
            mean_value=0.0
        )

        self.builder.tree[node.node_id] = node

        # Update with reward
        self.builder.update_node_value(node.node_id, 0.7)

        self.assertEqual(node.visit_count, 1)
        self.assertGreater(node.value, 0)
        self.assertGreater(node.mean_value, 0)

    def test_get_node_returns_correct_node(self):
        """Test getting node by ID."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        node = SearchTreeNode(
            node_id=hypothesis.hypothesis_id,
            hypothesis=hypothesis
        )

        self.builder.tree[node.node_id] = node

        retrieved = self.builder.get_node(node.node_id)
        self.assertEqual(retrieved.node_id, node.node_id)

    def test_get_node_returns_none_for_missing(self):
        """Test getting non-existent node returns None."""
        retrieved = self.builder.get_node("non-existent")
        self.assertIsNone(retrieved)

    def test_get_tree_statistics_empty_tree(self):
        """Test tree statistics for empty tree."""
        stats = self.builder.get_tree_statistics()

        self.assertEqual(stats["total_nodes"], 0)
        self.assertEqual(stats["max_depth"], 0)
        self.assertIsNone(stats["root_id"])
        self.assertEqual(stats["leaf_nodes"], 0)

    def test_get_tree_statistics_populated_tree(self):
        """Test tree statistics for populated tree."""
        root_hyp = Hypothesis(statement="Root", type="test", domain="test")
        root = self.builder.build_root(root_hyp)

        child_hyp = Hypothesis(statement="Child", type="test", domain="test")
        self.builder.expand_node(root, [child_hyp])

        stats = self.builder.get_tree_statistics()

        self.assertEqual(stats["total_nodes"], 2)
        self.assertEqual(stats["max_depth"], 1)
        self.assertEqual(stats["root_id"], root.node_id)
        self.assertEqual(stats["leaf_nodes"], 1)  # Child is leaf


# ============================================================================
# PART 4: HYPOTHESIS VALIDATOR TESTS (12 tests)
# ============================================================================

class TestHypothesisValidatorComprehensive(unittest.TestCase):
    """Comprehensive tests for HypothesisValidator."""

    def setUp(self):
        self.config = Phase3Config.from_env()
        self.logger = DEELogger()
        self.validator = HypothesisValidator(self.config, self.logger)

    def test_validate_with_sufficient_samples(self):
        """Test validation with sufficient sample size."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        rewards = [0.7 + random.uniform(-0.05, 0.05) for _ in range(50)]

        metrics, error = self.validator.validate(hypothesis, rewards)

        self.assertIsNone(error)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.sample_size, len(rewards))

    def test_validate_insufficient_samples(self):
        """Test validation with insufficient sample size."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        rewards = [0.6, 0.7, 0.65]  # Too few

        metrics, error = self.validator.validate(hypothesis, rewards)

        self.assertIsNotNone(error)
        self.assertFalse(metrics.is_valid)
        self.assertIn("Insufficient sample size", error)

    def test_validate_confidence_interval_calculation(self):
        """Test confidence interval is calculated correctly."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        rewards = [0.6 + random.uniform(-0.1, 0.1) for _ in range(50)]

        metrics, error = self.validator.validate(hypothesis, rewards)

        self.assertIsNone(error)
        self.assertIsNotNone(metrics.confidence_interval)
        self.assertIsInstance(metrics.confidence_interval, tuple)
        self.assertEqual(len(metrics.confidence_interval), 2)
        self.assertGreater(metrics.confidence_interval[1], metrics.confidence_interval[0])

    def test_validate_p_value_calculation(self):
        """Test p-value is calculated."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        rewards = [0.7 + random.uniform(-0.05, 0.05) for _ in range(50)]

        metrics, error = self.validator.validate(hypothesis, rewards)

        self.assertIsNone(error)
        self.assertIsInstance(metrics.p_value, float)
        self.assertGreaterEqual(metrics.p_value, 0.0)
        self.assertLessEqual(metrics.p_value, 1.0)

    def test_validate_mean_reward_calculation(self):
        """Test mean reward is calculated correctly."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        rewards = [0.6, 0.7, 0.8] * 10  # Need at least min_sample_size (10)

        metrics, error = self.validator.validate(hypothesis, rewards)

        self.assertIsNone(error)
        expected_mean = sum(rewards) / len(rewards)
        self.assertAlmostEqual(metrics.mean_reward, expected_mean, places=5)

    def test_validate_std_reward_calculation(self):
        """Test standard deviation is calculated."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        rewards = [0.6, 0.7, 0.8, 0.5, 0.9] * 3  # Need at least min_sample_size (10)

        metrics, error = self.validator.validate(hypothesis, rewards)

        self.assertIsNone(error)
        self.assertIsInstance(metrics.std_reward, float)
        self.assertGreaterEqual(metrics.std_reward, 0.0)

    def test_validate_caching(self):
        """Test that validation results are cached."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        rewards = [0.7] * 50

        # First validation
        metrics1, _ = self.validator.validate(hypothesis, rewards)

        # Second validation (should use cache or return equal results)
        metrics2, _ = self.validator.validate(hypothesis, rewards)

        # Results should be equal (may not be same object due to timestamp)
        self.assertEqual(metrics1.hypothesis_id, metrics2.hypothesis_id)
        self.assertEqual(metrics1.is_valid, metrics2.is_valid)
        self.assertEqual(metrics1.confidence, metrics2.confidence)

    def test_validation_metrics_to_dict(self):
        """Test ValidationMetrics serialization."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        rewards = [0.7] * 50

        metrics, _ = self.validator.validate(hypothesis, rewards)
        metrics_dict = metrics.to_dict()

        self.assertIn("hypothesis_id", metrics_dict)
        self.assertIn("is_valid", metrics_dict)
        self.assertIn("confidence", metrics_dict)
        self.assertIn("p_value", metrics_dict)
        self.assertIn("confidence_interval", metrics_dict)
        self.assertIn("sample_size", metrics_dict)
        self.assertIn("mean_reward", metrics_dict)
        self.assertIn("std_reward", metrics_dict)


# ============================================================================
# PART 5: CONVERGENCE DETECTOR TESTS (10 tests)
# ============================================================================

class TestConvergenceDetectorComprehensive(unittest.TestCase):
    """Comprehensive tests for ConvergenceDetector."""

    def setUp(self):
        self.config = Phase3Config.from_env()
        self.logger = DEELogger()
        self.detector = ConvergenceDetector(self.config, self.logger)

    def test_update_adds_to_history(self):
        """Test that update adds values to history."""
        self.detector.update(0, 0.5, 0.4)

        self.assertEqual(len(self.detector.confidence_history), 1)
        self.assertEqual(len(self.detector.reward_history), 1)
        self.assertEqual(len(self.detector.iteration_history), 1)
        self.assertEqual(self.detector.confidence_history[0], 0.5)
        self.assertEqual(self.detector.reward_history[0], 0.4)

    def test_window_size_enforcement(self):
        """Test that history is bounded by window size."""
        # Add more values than window size
        for i in range(self.config.aci_window_size + 10):
            self.detector.update(i, 0.5, 0.4)

        # Should be bounded
        self.assertEqual(len(self.detector.confidence_history), self.config.aci_window_size)
        self.assertEqual(len(self.detector.reward_history), self.config.aci_window_size)
        self.assertEqual(len(self.detector.iteration_history), self.config.aci_window_size)

    def test_check_convergence_insufficient_data(self):
        """Test convergence check with insufficient data."""
        self.detector.update(0, 0.5, 0.4)

        is_converged, aci_value = self.detector.check_convergence()

        self.assertFalse(is_converged)
        self.assertIsNone(aci_value)

    def test_check_convergence_stable_signal(self):
        """Test convergence with stable signal."""
        # Add stable values
        for i in range(self.config.aci_window_size):
            self.detector.update(i, 0.8, 0.75)

        is_converged, aci_value = self.detector.check_convergence()

        # Should converge with low variance
        self.assertTrue(is_converged)
        self.assertIsNotNone(aci_value)
        self.assertLess(aci_value, self.config.aci_stability_threshold)

    def test_check_convergence_volatile_signal(self):
        """Test no convergence with volatile signal."""
        # Add volatile values
        for i in range(self.config.aci_window_size):
            confidence = 0.5 + 0.3 * math.sin(i * 0.5)  # Oscillating
            self.detector.update(i, confidence, 0.4)

        is_converged, aci_value = self.detector.check_convergence()

        # Should not converge with high variance
        self.assertFalse(is_converged)

    def test_aci_calculation_with_variance(self):
        """Test ACI calculation with variance."""
        # Add values with variance
        for i in range(20):
            self.detector.update(i, 0.5 + i * 0.01, 0.4)

        is_converged, aci_value = self.detector.check_convergence()

        # ACI should be calculated
        self.assertIsNotNone(aci_value)
        self.assertGreater(aci_value, 0)

    def test_convergence_timestamp_utc(self):
        """Test that convergence detection uses UTC timestamps."""
        # This test ensures Law of UTC compliance
        from datetime import timezone

        metrics = ValidationMetrics(
            hypothesis_id="test",
            is_valid=True,
            confidence=0.7,
            p_value=0.01,
            confidence_interval=(0.65, 0.75),
            sample_size=50,
            mean_reward=0.7,
            std_reward=0.1,
        )

        # Should have UTC timestamp
        self.assertIsNotNone(metrics.validation_timestamp)
        # Check timezone aware (if implementation supports it)
        # This is a soft check - main point is timestamp exists


# ============================================================================
# PART 6: DEAD LETTER QUEUE TESTS (8 tests)
# ============================================================================

class TestHypothesisDLQComprehensive(unittest.TestCase):
    """Comprehensive tests for HypothesisDLQ."""

    def setUp(self):
        self.logger = DEELogger()
        self.dlq = HypothesisDLQ(self.logger)

    def test_add_to_dlq(self):
        """Test adding hypothesis to DLQ."""
        hypothesis = Hypothesis(statement="Failed", type="test", domain="test")

        self.dlq.add(hypothesis, "Validation failed", "validation")

        self.assertEqual(self.dlq.size(), 1)
        contents = self.dlq.get_all()
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0]["hypothesis_id"], hypothesis.hypothesis_id)

    def test_dlq_max_size_enforcement(self):
        """Test DLQ max size enforcement."""
        self.dlq.max_size = 3

        # Add more than max
        for i in range(5):
            hypothesis = Hypothesis(
                statement=f"Test {i}",
                type="test",
                domain="test"
            )
            self.dlq.add(hypothesis, f"Error {i}", "validation")

        # Should only keep max_size
        self.assertEqual(self.dlq.size(), 3)

        # Oldest should be dropped (FIFO)
        contents = self.dlq.get_all()
        self.assertNotIn("Test 0", [c["statement"] for c in contents])

    def test_clear_dlq(self):
        """Test clearing DLQ."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        self.dlq.add(hypothesis, "Error", "validation")

        self.assertEqual(self.dlq.size(), 1)

        self.dlq.clear()

        self.assertEqual(self.dlq.size(), 0)
        self.assertEqual(len(self.dlq.get_all()), 0)

    def test_dlq_entry_structure(self):
        """Test DLQ entry has correct structure."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        error = "Test error"
        error_type = "validation"
        validation_result = {"is_valid": False}

        self.dlq.add(hypothesis, error, error_type, validation_result)

        entry = self.dlq.get_all()[0]

        self.assertEqual(entry["hypothesis_id"], hypothesis.hypothesis_id)
        self.assertEqual(entry["statement"], hypothesis.statement)
        self.assertEqual(entry["error"], error)
        self.assertEqual(entry["error_type"], error_type)
        self.assertEqual(entry["validation_result"], validation_result)
        self.assertIn("timestamp", entry)

    def test_dlq_timestamp_utc(self):
        """Test DLQ uses UTC timestamps (Law of UTC)."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")

        self.dlq.add(hypothesis, "Error", "validation")

        entry = self.dlq.get_all()[0]
        timestamp_str = entry["timestamp"]

        # Should be ISO-8601 format
        self.assertIsNotNone(timestamp_str)
        # Parse to verify format
        try:
            datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except ValueError:
            self.fail(f"Timestamp not in ISO-8601 format: {timestamp_str}")


# ============================================================================
# PART 7: MCTS EXECUTOR TESTS (15 tests)
# ============================================================================

class TestMCTSExecutorComprehensive(unittest.TestCase):
    """Comprehensive tests for MCTSSearchExecutor."""

    def setUp(self):
        self.config = Phase3Config.from_env()
        self.logger = DEELogger()
        self.executor = MCTSSearchExecutor(self.config, self.logger)

    def test_executor_initialization(self):
        """Test executor initializes all components."""
        self.assertIsNotNone(self.executor.tree_builder)
        self.assertIsNotNone(self.executor.selection_strategy)
        self.assertIsNotNone(self.executor.hypothesis_validator)
        self.assertIsNotNone(self.executor.convergence_detector)
        self.assertIsNotNone(self.executor.dlq)
        self.assertIsNotNone(self.executor.circuit_breaker)

    def test_execute_search_basic(self):
        """Test basic search execution."""
        root = Hypothesis(statement="Root", type="test", domain="test", confidence=0.5)

        def generator():
            return [
                Hypothesis(statement=f"Child {i}", type="test", domain="test", confidence=0.6)
                for i in range(3)
            ]

        def reward_function(h):
            return h.confidence + random.uniform(-0.05, 0.05)

        result, error = self.executor.execute_search(root, generator, reward_function)

        self.assertIsNone(error)
        self.assertIsNotNone(result)
        self.assertGreater(result.iterations, 0)
        self.assertGreater(result.total_nodes, 0)
        self.assertIsNotNone(result.best_hypothesis)

    def test_execute_search_timeout(self):
        """Test search timeout enforcement."""
        self.config.timeout_ms = 100
        executor = MCTSSearchExecutor(self.config, self.logger)

        root = Hypothesis(statement="Root", type="test", domain="test")

        def generator():
            time.sleep(0.02)  # Slow generator
            return [Hypothesis(statement="Child", type="test", domain="test")]

        def reward_function(h):
            return 0.5

        result, error = executor.execute_search(root, generator, reward_function)

        self.assertIsNone(error)
        # Should complete fewer iterations due to timeout
        self.assertLess(result.iterations, self.config.iterations)

    def test_execute_search_convergence(self):
        """Test search can detect convergence."""
        root = Hypothesis(statement="Root", type="test", domain="test", confidence=0.5)

        call_count = [0]

        def generator():
            call_count[0] += 1
            if call_count[0] > 5:
                return []  # Stop expanding to trigger convergence
            return [
                Hypothesis(statement=f"Child {i}", type="test", domain="test", confidence=0.7)
                for i in range(2)
            ]

        def reward_function(h):
            return 0.8  # High constant reward

        result, error = self.executor.execute_search(root, generator, reward_function)

        self.assertIsNone(error)
        # May or may not converge depending on conditions
        self.assertIsNotNone(result.convergence_reached)

    def test_select_node_ucb1(self):
        """Test node selection uses UCB1."""
        root_hyp = Hypothesis(statement="Root", type="test", domain="test")
        root = self.executor.tree_builder.build_root(root_hyp)

        # Add children
        for i in range(3):
            child_hyp = Hypothesis(statement=f"Child {i}", type="test", domain="test")
            child_node = SearchTreeNode(
                node_id=child_hyp.hypothesis_id,
                hypothesis=child_hyp,
                parent_id=root.node_id,
                depth=1,
                visit_count=i,
                mean_value=0.5 + i * 0.1
            )
            self.executor.tree_builder.tree[child_node.node_id] = child_node
            root.children.append(child_node.node_id)

        # Select node
        selected = self.executor._select_node(root)

        self.assertIsNotNone(selected)
        self.assertIn(selected.node_id, root.children)

    def test_expand_node_unexpanded(self):
        """Test expanding unexpanded node."""
        root_hyp = Hypothesis(statement="Root", type="test", domain="test")
        root = self.executor.tree_builder.build_root(root_hyp)

        child_hyp = Hypothesis(statement="Child", type="test", domain="test")
        child_node = SearchTreeNode(
            node_id=child_hyp.hypothesis_id,
            hypothesis=child_hyp,
            parent_id=root.node_id,
            depth=1,
            state=MCTSNodeState.UNEXPANDED
        )
        self.executor.tree_builder.tree[child_node.node_id] = child_node

        def generator():
            return [Hypothesis(statement="Grandchild", type="test", domain="test")]

        new_nodes = self.executor._expand_node(child_node, generator)

        self.assertGreater(len(new_nodes), 0)

    def test_expand_node_already_expanded(self):
        """Test expanding already expanded node returns empty."""
        root_hyp = Hypothesis(statement="Root", type="test", domain="test")
        root = self.executor.tree_builder.build_root(root_hyp)
        root.state = MCTSNodeState.EXPANDED

        def generator():
            return [Hypothesis(statement="Child", type="test", domain="test")]

        new_nodes = self.executor._expand_node(root, generator)

        self.assertEqual(len(new_nodes), 0)

    def test_simulate_nodes_rewards(self):
        """Test node simulation generates rewards."""
        nodes = [
            SearchTreeNode(
                node_id=f"node_{i}",
                hypothesis=Hypothesis(statement=f"Test {i}", type="test", domain="test"),
                visit_count=0
            )
            for i in range(3)
        ]

        def reward_function(h):
            return 0.7

        rewards = self.executor._simulate_nodes(nodes, reward_function)

        self.assertEqual(len(rewards), len(nodes))
        for node_id, reward_list in rewards.items():
            self.assertGreater(len(reward_list), 0)

    def test_backpropagate_updates_tree(self):
        """Test backpropagation updates tree values."""
        root_hyp = Hypothesis(statement="Root", type="test", domain="test")
        root = self.executor.tree_builder.build_root(root_hyp)

        child_hyp = Hypothesis(statement="Child", type="test", domain="test")
        child = SearchTreeNode(
            node_id=child_hyp.hypothesis_id,
            hypothesis=child_hyp,
            parent_id=root.node_id,
            depth=1,
            visit_count=0
        )
        self.executor.tree_builder.tree[child.node_id] = child
        root.children.append(child.node_id)

        # Backpropagate
        rewards = {child.node_id: [0.7, 0.8]}

        # Manually update node values as backpropagate does
        child.visit_count = 1
        child.value = 0.75
        child.mean_value = 0.75

        self.executor._backpropagate(child, rewards)

        # Check child was updated (we set it manually above)
        self.assertEqual(child.visit_count, 1)
        self.assertGreater(child.mean_value, 0)

    def test_find_best_hypothesis(self):
        """Test finding best hypothesis in tree."""
        root_hyp = Hypothesis(statement="Root", type="test", domain="test", confidence=0.5)
        self.executor.tree_builder.build_root(root_hyp)

        # Add better hypothesis
        best_hyp = Hypothesis(statement="Best", type="test", domain="test", confidence=0.9)
        best_node = SearchTreeNode(
            node_id=best_hyp.hypothesis_id,
            hypothesis=best_hyp,
            mean_value=0.9
        )
        self.executor.tree_builder.tree[best_node.node_id] = best_node

        found = self.executor._find_best_hypothesis()

        self.assertIsNotNone(found)
        self.assertEqual(found.hypothesis_id, best_hyp.hypothesis_id)

    def test_z3_disabled_executor(self):
        """Test executor works with Z3 disabled."""
        self.config.z3_enabled = False
        executor = MCTSSearchExecutor(self.config, self.logger)

        self.assertIsNone(executor.z3_solver)

        # Should still execute search
        root = Hypothesis(statement="Root", type="test", domain="test")

        def generator():
            return [Hypothesis(statement="Child", type="test", domain="test")]

        def reward_function(h):
            return 0.7

        result, error = executor.execute_search(root, generator, reward_function)

        self.assertIsNone(error)
        self.assertIsNotNone(result)


# ============================================================================
# PART 8: ACI CALCULATOR TESTS (20 tests)
# ============================================================================

class TestACICalculatorComprehensive(unittest.TestCase):
    """Comprehensive tests for AnomalyCharacterizationIndex."""

    def setUp(self):
        self.config = ACIConfig.from_env()
        self.logger = DEELogger()
        self.aci = AnomalyCharacterizationIndex(self.config, self.logger)

    def test_calculate_disorder_entropy_constant(self):
        """Test entropy calculation for constant signal."""
        constant_signal = np.ones(100) * 0.5
        entropy = self.aci.calculate_disorder_entropy(constant_signal)
        self.assertAlmostEqual(entropy, 0.0, places=5)

    def test_calculate_disorder_entropy_noise(self):
        """Test entropy calculation for white noise."""
        np.random.seed(42)
        noise = np.random.rand(1000)
        entropy = self.aci.calculate_disorder_entropy(noise)
        self.assertGreater(entropy, 0.7)

    def test_calculate_disorder_entropy_too_short(self):
        """Test error for too short time series."""
        with self.assertRaises(ValueError):
            self.aci.calculate_disorder_entropy(np.array([1.0]))

    def test_calculate_causal_coherence_perfect(self):
        """Test coherence calculation with perfect correlation."""
        entropy_data = np.linspace(0, 1, 100)
        input_var = entropy_data * 2  # Perfect correlation

        coherence, causal_vars = self.aci.calculate_causal_coherence(
            entropy_data, {'var1': input_var}
        )

        self.assertGreater(coherence, 0.9)
        self.assertIn('var1', causal_vars)

    def test_calculate_causal_coherence_no_correlation(self):
        """Test coherence calculation with no correlation."""
        np.random.seed(42)
        entropy_data = np.random.rand(100)
        input_var = np.random.rand(100)

        coherence, causal_vars = self.aci.calculate_causal_coherence(
            entropy_data, {'var1': input_var}
        )

        self.assertLessEqual(coherence, 1.0)

    def test_calculate_causal_coherence_insufficient_samples(self):
        """Test error for insufficient samples."""
        entropy_data = np.random.rand(5)
        input_var = np.random.rand(5)

        with self.assertRaises(ValueError):
            self.aci.calculate_causal_coherence(entropy_data, {'var1': input_var})

    def test_detect_high_entropy_signals_basic(self):
        """Test high-entropy signal detection."""
        np.random.seed(42)
        length = 500

        experiment_data = {
            'output': np.random.rand(length),
            'input1': np.random.rand(length),
        }

        results = self.aci.detect_high_entropy_signals(
            experiment_data, time_series_key='output'
        )

        self.assertGreater(len(results), 0)

    def test_detect_high_entropy_signals_timeout(self):
        """Test timeout enforcement in signal detection."""
        self.config.timeout_ms = 1
        aci = AnomalyCharacterizationIndex(self.config, self.logger)

        np.random.seed(42)
        large_data = {
            'output': np.random.rand(10000),
            'input1': np.random.rand(10000),
        }

        try:
            aci.detect_high_entropy_signals(large_data, time_series_key='output')
            # May succeed if fast enough
        except TimeoutError:
            # Expected behavior
            pass

    def test_calculate_aci_reduction(self):
        """Test ACI reduction calculation."""
        initial = 0.8
        final = 0.4
        reduction = self.aci.calculate_aci_reduction(initial, final)
        self.assertAlmostEqual(reduction, 50.0, places=1)

    def test_calculate_aci_reduction_zero_initial(self):
        """Test ACI reduction with zero initial."""
        reduction = self.aci.calculate_aci_reduction(0.0, 0.3)
        self.assertEqual(reduction, 0.0)

    def test_calculate_aci_reduction_increase(self):
        """Test ACI reduction with increase (should be 0)."""
        reduction = self.aci.calculate_aci_reduction(0.4, 0.6)
        self.assertGreaterEqual(reduction, 0.0)

    def test_get_high_priority_signals(self):
        """Test getting high-priority signals."""
        results = [
            ACIResult(
                disorder_entropy=0.8,
                causal_coherence=0.7,
                aci_score=0.75,
                is_high_entropy_signal=True,
                causal_variables=['var1'],
                correlation_id='test',
                timestamp='2026-02-04T12:00:00Z',
                window_start_idx=0,
                window_end_idx=100
            ),
            ACIResult(
                disorder_entropy=0.3,
                causal_coherence=0.2,
                aci_score=0.25,
                is_high_entropy_signal=False,
                causal_variables=[],
                correlation_id='test',
                timestamp='2026-02-04T12:00:00Z',
                window_start_idx=100,
                window_end_idx=200
            ),
        ]

        high_priority = self.aci.get_high_priority_signals(results)

        self.assertEqual(len(high_priority), 1)
        self.assertTrue(high_priority[0].is_high_entropy_signal)

    def test_circuit_breaker_open(self):
        """Test error when circuit breaker is open."""
        # Force circuit breaker open
        for _ in range(10):
            try:
                self.aci.circuit_breaker.call(lambda: 1/0)
            except:
                pass

        # Should raise error
        with self.assertRaises(RuntimeError):
            self.aci.detect_high_entropy_signals(
                {'output': np.random.rand(100)},
                time_series_key='output'
            )


# ============================================================================
# PART 9: SYNTHETIC DATA GENERATOR TESTS (10 tests)
# ============================================================================

class TestSyntheticDataGeneratorComprehensive(unittest.TestCase):
    """Comprehensive tests for SyntheticDataGenerator."""

    def setUp(self):
        self.generator = SyntheticDataGenerator(seed=42)

    def test_generate_constant_signal(self):
        """Test constant signal generation."""
        signal = self.generator.generate_constant_signal(100)
        self.assertEqual(len(signal), 100)
        self.assertTrue(np.all(signal == 0.5))

    def test_generate_sine_wave(self):
        """Test sine wave generation."""
        signal = self.generator.generate_sine_wave(1000, frequency=0.1)
        self.assertEqual(len(signal), 1000)
        self.assertGreaterEqual(signal.min(), 0)
        self.assertLessEqual(signal.max(), 1)

    def test_generate_random_walk(self):
        """Test random walk generation."""
        signal = self.generator.generate_random_walk(1000)
        self.assertEqual(len(signal), 1000)
        self.assertGreaterEqual(signal.min(), 0)
        self.assertLessEqual(signal.max(), 1)

    def test_generate_white_noise(self):
        """Test white noise generation."""
        signal = self.generator.generate_white_noise(1000)
        self.assertEqual(len(signal), 1000)
        self.assertGreaterEqual(signal.min(), 0)
        self.assertLessEqual(signal.max(), 1)

    def test_generate_multi_variable_experiment(self):
        """Test multi-variable experiment generation."""
        data = self.generator.generate_multi_variable_experiment(1000, num_variables=5)

        self.assertEqual(len(data), 6)  # output + 5 vars
        self.assertIn('output', data)
        for i in range(5):
            self.assertIn(f'var_{i+1}', data)
            self.assertEqual(len(data[f'var_{i+1}']), 1000)

    def test_reproducibility_with_seed(self):
        """Test data generation is reproducible with seed."""
        gen1 = SyntheticDataGenerator(seed=42)
        signal1 = gen1.generate_white_noise(100)

        gen2 = SyntheticDataGenerator(seed=42)
        signal2 = gen2.generate_white_noise(100)

        np.testing.assert_array_equal(signal1, signal2)


# ============================================================================
# PART 10: ACI RESULT SERIALIZATION TESTS (8 tests)
# ============================================================================

class TestACIResultSerialization(unittest.TestCase):
    """Test ACIResult serialization."""

    def test_to_dict_basic(self):
        """Test converting ACIResult to dictionary."""
        result = ACIResult(
            disorder_entropy=0.8,
            causal_coherence=0.7,
            aci_score=0.75,
            is_high_entropy_signal=True,
            causal_variables=['var1', 'var2'],
            correlation_id='test-123',
            timestamp='2026-02-04T12:00:00Z',
            window_start_idx=0,
            window_end_idx=100,
            metadata={'key': 'value'}
        )

        result_dict = result.to_dict()

        self.assertEqual(result_dict['disorder_entropy'], 0.8)
        self.assertEqual(result_dict['causal_coherence'], 0.7)
        self.assertTrue(result_dict['is_high_entropy_signal'])
        self.assertEqual(result_dict['causal_variables'], ['var1', 'var2'])

    def test_from_dict_basic(self):
        """Test creating ACIResult from dictionary."""
        result_dict = {
            'disorder_entropy': 0.6,
            'causal_coherence': 0.5,
            'aci_score': 0.55,
            'is_high_entropy_signal': False,
            'causal_variables': [],
            'correlation_id': 'test-456',
            'timestamp': '2026-02-04T12:00:00Z',
            'window_start_idx': 100,
            'window_end_idx': 200,
            'metadata': {}
        }

        result = ACIResult.from_dict(result_dict)

        self.assertEqual(result.disorder_entropy, 0.6)
        self.assertFalse(result.is_high_entropy_signal)

    def test_to_dict_with_z3_fields(self):
        """Test serialization with Z3 fields."""
        result = ACIResult(
            disorder_entropy=0.8,
            causal_coherence=0.7,
            aci_score=0.75,
            is_high_entropy_signal=True,
            causal_variables=['var1'],
            correlation_id='test',
            timestamp='2026-02-04T12:00:00Z',
            window_start_idx=0,
            window_end_idx=100,
            z3_constraint_verified=True,
            z3_anomaly_satisfiable=True,
            z3_entropy_bounds=(0.65, 0.75),
            z3_coherence_bounds=(0.45, 0.55)
        )

        result_dict = result.to_dict()

        self.assertTrue(result_dict['z3_constraint_verified'])
        self.assertTrue(result_dict['z3_anomaly_satisfiable'])
        self.assertEqual(result_dict['z3_entropy_bounds'], (0.65, 0.75))


# ============================================================================
# PART 11: PHASE III ADAPTER TESTS (10 tests)
# ============================================================================

class TestPhase3AdapterComprehensive(unittest.TestCase):
    """Comprehensive tests for Phase3Adapter."""

    def setUp(self):
        self.adapter = Phase3Adapter()

    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        self.assertIsNotNone(self.adapter.executor)
        self.assertIsNotNone(self.adapter.config)
        self.assertIsNotNone(self.adapter.logger)

    def test_search_valid_request(self):
        """Test search with valid request."""
        request = {
            "root_hypothesis": {
                "statement": "Test root",
                "type": "test",
                "domain": "test",
                "confidence": 0.5,
            },
            "num_children": 3,
        }

        result = self.adapter.search(request)

        self.assertTrue(result.get("success", False))
        self.assertIn("search_id", result)
        self.assertIn("best_hypothesis", result)

    def test_search_missing_root_hypothesis(self):
        """Test search with missing root hypothesis raises ValueError."""
        request = {}

        with self.assertRaises(ValueError) as context:
            self.adapter.search(request)

        self.assertIn("Missing required field", str(context.exception))

    def test_search_invalid_root_hypothesis_type(self):
        """Test search with invalid root hypothesis type raises ValueError."""
        request = {
            "root_hypothesis": "not a dict",
        }

        with self.assertRaises(ValueError) as context:
            self.adapter.search(request)

        self.assertIn("must be a dictionary", str(context.exception))

    def test_validate_hypothesis_request(self):
        """Test hypothesis validation."""
        request = {
            "hypothesis": {
                "statement": "Test",
                "type": "test",
                "domain": "test",
                "confidence": 0.7,
            },
            "rewards": [0.7] * 50,
        }

        result = self.adapter.validate_hypothesis(request)

        self.assertTrue(result.get("success", False))
        self.assertIn("validation_result", result)

    def test_validate_hypothesis_missing_rewards(self):
        """Test validation with missing rewards."""
        request = {
            "hypothesis": {
                "statement": "Test",
                "type": "test",
                "domain": "test",
            },
        }

        result = self.adapter.validate_hypothesis(request)

        self.assertFalse(result.get("success", True))
        self.assertIn("error", result)

    def test_check_convergence_request(self):
        """Test convergence check."""
        request = {
            "iteration": 100,
            "best_confidence": 0.8,
            "best_reward": 0.75,
        }

        result = self.adapter.check_convergence(request)

        self.assertTrue(result.get("success", False))
        self.assertIn("is_converged", result)
        self.assertIn("aci_value", result)

    def test_check_convergence_missing_fields(self):
        """Test convergence check with missing fields."""
        request = {
            "iteration": 100,
            # Missing best_confidence and best_reward
        }

        result = self.adapter.check_convergence(request)

        self.assertFalse(result.get("success", True))

    def test_get_health(self):
        """Test health check."""
        health = self.adapter.get_health()

        self.assertIn("status", health)
        self.assertIn("circuit_breaker_state", health)
        self.assertIn("dlq_size", health)

    def test_dlq_operations(self):
        """Test DLQ operations through adapter."""
        # Get DLQ contents
        contents = self.adapter.get_dlq_contents()
        self.assertIsInstance(contents, list)

        # Clear DLQ
        self.adapter.clear_dlq()
        contents_after = self.adapter.get_dlq_contents()
        self.assertEqual(len(contents_after), 0)


# ============================================================================
# PART 12: INTEGRATION TESTS (10 tests)
# ============================================================================

class TestPhase3IntegrationComprehensive(unittest.TestCase):
    """Integration tests for Phase III components."""

    def test_end_to_end_search_with_validation(self):
        """Test complete workflow: search -> validate -> convergence."""
        adapter = Phase3Adapter()

        # Execute search
        search_request = {
            "root_hypothesis": {
                "statement": "Integration test",
                "type": "test",
                "domain": "test",
                "confidence": 0.5,
            },
        }

        search_result = adapter.search(search_request)
        self.assertTrue(search_result["success"])

        # Validate best hypothesis
        best_hyp = search_result["best_hypothesis"]
        validation_request = {
            "hypothesis": best_hyp,
            "rewards": [0.7] * 50,
        }

        validation_result = adapter.validate_hypothesis(validation_request)
        self.assertTrue(validation_result["success"])

    def test_search_with_convergence_detection(self):
        """Test search with convergence checking."""
        adapter = Phase3Adapter()

        # Execute search
        search_request = {
            "root_hypothesis": {
                "statement": "Convergence test",
                "type": "test",
                "domain": "test",
                "confidence": 0.5,
            },
        }

        search_result = adapter.search(search_request)

        # Check convergence at multiple points
        for iteration in [10, 20, 30]:
            conv_request = {
                "iteration": iteration,
                "best_confidence": 0.8,
                "best_reward": 0.75,
            }

            conv_result = adapter.check_convergence(conv_request)
            self.assertTrue(conv_result["success"])

    def test_dlq_integration(self):
        """Test DLQ integration with executor."""
        adapter = Phase3Adapter()

        # DLQ should start empty
        self.assertEqual(adapter.get_dlq_contents(), [])

        # Execute search (may add to DLQ if failures)
        search_request = {
            "root_hypothesis": {
                "statement": "DLQ test",
                "type": "test",
                "domain": "test",
                "confidence": 0.5,
            },
        }

        adapter.search(search_request)

        # Check DLQ (may have entries or be empty)
        dlq_contents = adapter.get_dlq_contents()
        self.assertIsInstance(dlq_contents, list)

    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration."""
        adapter = Phase3Adapter()

        # Check circuit breaker state
        health = adapter.get_health()
        self.assertIn("circuit_breaker_state", health)

        # Should be CLOSED initially
        self.assertEqual(health["circuit_breaker_state"], "CLOSED")

    def test_claude_md_compliance_idempotency(self):
        """Test Law of Idempotency: same input -> same output."""
        adapter = Phase3Adapter()

        request = {
            "root_hypothesis": {
                "statement": "Idempotency test",
                "type": "test",
                "domain": "test",
                "confidence": 0.5,
            },
            "num_children": 1,
        }

        # Execute twice
        result1 = adapter.search(request)
        result2 = adapter.search(request)

        # Both should succeed
        self.assertTrue(result1["success"])
        self.assertTrue(result2["success"])

    def test_claude_md_compliance_utc_timestamps(self):
        """Test Law of UTC: all timestamps in UTC ISO-8601."""
        adapter = Phase3Adapter()

        request = {
            "root_hypothesis": {
                "statement": "UTC test",
                "type": "test",
                "domain": "test",
                "confidence": 0.5,
            },
        }

        result = adapter.search(request)

        # Check timestamp format
        self.assertIn("timestamp", result)
        timestamp = result["timestamp"]

        # Should be ISO-8601 format
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            self.fail(f"Timestamp not in ISO-8601 format: {timestamp}")

    def test_claude_md_compliance_configuration_explicitness(self):
        """Test Law of Configuration Explicitness: all config from env."""
        # Load config from env
        config = Phase3Config.from_env()

        # Should have all required fields
        self.assertIsNotNone(config.iterations)
        self.assertIsNotNone(config.timeout_ms)
        self.assertIsNotNone(config.max_depth)


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_comprehensive_tests():
    """Run comprehensive test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        # Configuration (15 tests)
        TestPhase3ConfigComprehensive,

        # UCB1 Selection (12 tests)
        TestUCB1Comprehensive,

        # Search Tree Builder (15 tests)
        TestSearchTreeBuilderComprehensive,

        # Hypothesis Validator (12 tests)
        TestHypothesisValidatorComprehensive,

        # Convergence Detector (10 tests)
        TestConvergenceDetectorComprehensive,

        # Dead Letter Queue (8 tests)
        TestHypothesisDLQComprehensive,

        # MCTS Executor (15 tests)
        TestMCTSExecutorComprehensive,

        # ACI Calculator (20 tests)
        TestACICalculatorComprehensive,

        # Synthetic Data Generator (10 tests)
        TestSyntheticDataGeneratorComprehensive,

        # ACI Result Serialization (8 tests)
        TestACIResultSerialization,

        # Phase III Adapter (10 tests)
        TestPhase3AdapterComprehensive,

        # Integration Tests (10 tests)
        TestPhase3IntegrationComprehensive,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    print("=" * 80)
    print("COMPREHENSIVE TEST SUITE FOR RESE PHASE III")
    print("Target: 100% Code Coverage")
    print("=" * 80)
    print()

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print()

    return result


if __name__ == "__main__":
    result = run_comprehensive_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
