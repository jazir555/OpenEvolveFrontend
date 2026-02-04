"""
Comprehensive tests for RESE Phase III MCTS Search Executor.

Tests all components:
- Configuration validation
- Search tree builder
- UCB1 selection strategy
- Hypothesis validator
- Convergence detector
- MC-NEST executor
- Phase III adapter

Following CLAUDE.md principles:
- Law of Runtime Truth: Test against actual execution
- Law of Idempotency: Test deduplication
- Circuit Breaker: Test failure handling
- Timeout: Test timeout enforcement

Author: RESE Team
Created: 2026-02-04
"""

import os
import sys
import unittest
import time
import random
from typing import List

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
    from rese_dee import DEELogger, CircuitBreaker
    from phase3_executor import (
        Phase3Config,
        MCTSSearchExecutor,
        SearchTreeBuilder,
        HypothesisValidator,
        ConvergenceDetector,
        UCB1SelectionStrategy,
        HypothesisDLQ,
        ValidationMetrics,
    )
    from phase3_adapter import Phase3Adapter, create_adapter
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


# Set environment variables for testing
os.environ["PHASE3_ITERATIONS"] = "50"
os.environ["PHASE3_UCB1_C"] = "1.414"
os.environ["PHASE3_CONVERGENCE_THRESHOLD"] = "0.001"
os.environ["PHASE3_TIMEOUT_MS"] = "30000"
os.environ["PHASE3_MAX_DEPTH"] = "10"
os.environ["PHASE3_MAX_CHILDREN"] = "5"
os.environ["PHASE3_MIN_VISITS"] = "3"
os.environ["PHASE3_SIG_THRESHOLD"] = "0.05"
os.environ["PHASE3_CONFIDENCE_INTERVAL"] = "0.95"
os.environ["PHASE3_MIN_SAMPLE_SIZE"] = "10"
os.environ["PHASE3_ACI_WINDOW"] = "20"
os.environ["PHASE3_ACI_STABILITY"] = "0.01"
os.environ["PHASE3_DEDUP_ENABLED"] = "true"
os.environ["PHASE3_CACHE_SIZE"] = "1000"
os.environ["PHASE3_CB_THRESHOLD"] = "3"
os.environ["PHASE3_CB_TIMEOUT"] = "60000"


class TestPhase3Config(unittest.TestCase):
    """Test Phase3Config."""

    def test_config_from_env(self):
        """Test configuration loading from environment."""
        config = Phase3Config.from_env()

        self.assertEqual(config.iterations, 50)
        self.assertAlmostEqual(config.ucb1_c, 1.414)
        self.assertEqual(config.max_depth, 10)
        self.assertEqual(config.max_children_per_node, 5)

    def test_config_defaults(self):
        """Test configuration defaults."""
        # Clear env vars
        original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("PHASE3_"):
                del os.environ[key]

        try:
            config = Phase3Config.from_env()

            # Check defaults
            self.assertGreater(config.iterations, 0)
            self.assertGreater(config.ucb1_c, 0)
            self.assertGreater(config.timeout_ms, 0)
        finally:
            # Restore env vars
            os.environ.update(original_env)


class TestUCB1SelectionStrategy(unittest.TestCase):
    """Test UCB1 selection strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = UCB1SelectionStrategy(exploration_constant=1.414)

    def test_select_best_child(self):
        """Test selecting best child using UCB1."""
        # Create parent node
        parent = SearchTreeNode(
            node_id="parent",
            visit_count=10,
            value=5.0
        )

        # Create children with different values
        child1 = SearchTreeNode(
            node_id="child1",
            parent_id="parent",
            visit_count=5,
            value=3.0,
            mean_value=0.6
        )
        child2 = SearchTreeNode(
            node_id="child2",
            parent_id="parent",
            visit_count=3,
            value=2.5,
            mean_value=0.83
        )
        child3 = SearchTreeNode(
            node_id="child3",
            parent_id="parent",
            visit_count=0,
            value=0.0,
            mean_value=0.0
        )

        parent.children = ["child1", "child2", "child3"]
        tree = {
            "parent": parent,
            "child1": child1,
            "child2": child2,
            "child3": child3,
        }

        # Select best child (should favor unvisited or high-value)
        best = self.strategy.select_child(parent, tree)

        self.assertIsNotNone(best)
        self.assertIn(best.node_id, ["child1", "child2", "child3"])

    def test_calculate_ucb1(self):
        """Test UCB1 calculation."""
        parent = SearchTreeNode(
            node_id="parent",
            visit_count=10
        )
        child = SearchTreeNode(
            node_id="child",
            parent_id="parent",
            visit_count=5,
            mean_value=0.7
        )

        ucb1 = self.strategy.calculate_ucb1(parent, child)

        # UCB1 should be > mean_value (exploration bonus)
        self.assertGreater(ucb1, child.mean_value)

    def test_ucb1_unvisited(self):
        """Test UCB1 for unvisited node (should be infinity)."""
        parent = SearchTreeNode(
            node_id="parent",
            visit_count=10
        )
        child = SearchTreeNode(
            node_id="child",
            parent_id="parent",
            visit_count=0,
            mean_value=0.0
        )

        ucb1 = self.strategy.calculate_ucb1(parent, child)

        # Unvisited nodes should have infinite UCB1
        self.assertEqual(ucb1, float('inf'))


class TestSearchTreeBuilder(unittest.TestCase):
    """Test search tree builder."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Phase3Config.from_env()
        self.logger = DEELogger()
        self.builder = SearchTreeBuilder(self.config, self.logger)

    def test_build_root(self):
        """Test building root node."""
        root_hypothesis = Hypothesis(
            statement="Root hypothesis",
            type="test",
            domain="test_domain"
        )

        root_node = self.builder.build_root(root_hypothesis)

        self.assertEqual(root_node.node_id, root_hypothesis.hypothesis_id)
        self.assertEqual(root_node.depth, 0)
        self.assertEqual(root_node.state, MCTSNodeState.EXPANDED)
        self.assertIn(root_node.node_id, self.builder.tree)

    def test_expand_node(self):
        """Test expanding node with children."""
        # Create root
        root_hypothesis = Hypothesis(statement="Root", type="test", domain="test")
        root_node = self.builder.build_root(root_hypothesis)

        # Create children
        children = [
            Hypothesis(statement=f"Child {i}", type="test", domain="test")
            for i in range(3)
        ]

        # Expand
        new_nodes = self.builder.expand_node(root_node, children)

        self.assertEqual(len(new_nodes), 3)
        self.assertEqual(len(root_node.children), 3)

        # Check children are in tree
        for node in new_nodes:
            self.assertIn(node.node_id, self.builder.tree)
            self.assertEqual(node.depth, 1)
            self.assertEqual(node.parent_id, root_node.node_id)

    def test_deduplication(self):
        """Test hypothesis deduplication (Law of Idempotency)."""
        # Create root
        root_hypothesis = Hypothesis(statement="Root", type="test", domain="test")
        root_node = self.builder.build_root(root_hypothesis)

        # Try to add duplicate children
        child_hypothesis = Hypothesis(statement="Child", type="test", domain="test")

        # First expansion should succeed
        new_nodes_1 = self.builder.expand_node(root_node, [child_hypothesis])
        self.assertEqual(len(new_nodes_1), 1)

        # Second expansion with same hypothesis should be deduplicated
        new_nodes_2 = self.builder.expand_node(root_node, [child_hypothesis])
        self.assertEqual(len(new_nodes_2), 0)  # Should be skipped

        # Tree should still have only 2 nodes
        self.assertEqual(len(self.builder.tree), 2)

    def test_max_depth_limit(self):
        """Test max depth limit enforcement."""
        self.config.max_depth = 2

        # Create root at depth 0
        root_hypothesis = Hypothesis(statement="Root", type="test", domain="test")
        root_node = self.builder.build_root(root_hypothesis)

        # Expand to depth 1
        child1 = Hypothesis(statement="Child 1", type="test", domain="test")
        child_nodes = self.builder.expand_node(root_node, [child1])
        self.assertEqual(len(child_nodes), 1)

        # Try to expand to depth 2 (should succeed)
        child2 = Hypothesis(statement="Child 2", type="test", domain="test")
        grandchild_nodes = self.builder.expand_node(child_nodes[0], [child2])
        self.assertEqual(len(grandchild_nodes), 1)

        # Try to expand to depth 3 (should fail - max depth reached)
        child3 = Hypothesis(statement="Child 3", type="test", domain="test")
        great_grandchild_nodes = self.builder.expand_node(grandchild_nodes[0], [child3])
        self.assertEqual(len(great_grandchild_nodes), 0)


class TestHypothesisValidator(unittest.TestCase):
    """Test hypothesis validator."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Phase3Config.from_env()
        self.logger = DEELogger()
        self.validator = HypothesisValidator(self.config, self.logger)

    def test_validate_valid_hypothesis(self):
        """Test validating a valid hypothesis."""
        hypothesis = Hypothesis(
            statement="Valid hypothesis",
            type="test",
            domain="test",
            confidence=0.7
        )

        # Generate rewards above threshold
        rewards = [0.7 + random.uniform(-0.05, 0.05) for _ in range(50)]

        validation_metrics, error = self.validator.validate(hypothesis, rewards)

        self.assertIsNone(error)
        self.assertTrue(validation_metrics.is_valid)
        self.assertGreater(validation_metrics.confidence, 0.5)
        self.assertEqual(validation_metrics.sample_size, len(rewards))

    def test_validate_insufficient_sample(self):
        """Test validation with insufficient sample size."""
        hypothesis = Hypothesis(
            statement="Test hypothesis",
            type="test",
            domain="test"
        )

        # Too few samples
        rewards = [0.6, 0.7, 0.65]

        validation_metrics, error = self.validator.validate(hypothesis, rewards)

        self.assertIsNotNone(error)
        self.assertFalse(validation_metrics.is_valid)
        self.assertIn("Insufficient sample size", error)

    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")

        # Generate rewards
        rewards = [0.6 + random.uniform(-0.1, 0.1) for _ in range(50)]

        validation_metrics, error = self.validator.validate(hypothesis, rewards)

        self.assertIsNone(error)
        self.assertIsNotNone(validation_metrics.confidence_interval)
        self.assertGreater(validation_metrics.confidence_interval[1], validation_metrics.confidence_interval[0])


class TestConvergenceDetector(unittest.TestCase):
    """Test convergence detector."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Phase3Config.from_env()
        self.logger = DEELogger()
        self.detector = ConvergenceDetector(self.config, self.logger)

    def test_no_convergence_early(self):
        """Test that convergence is not detected early."""
        # Add only a few data points
        for i in range(5):
            self.detector.update(i, 0.5 + i * 0.01, 0.4)

        is_converged, aci_value = self.detector.check_convergence()

        self.assertFalse(is_converged)
        self.assertIsNone(aci_value)

    def test_convergence_stable(self):
        """Test convergence detection with stable confidence."""
        # Add stable confidence values
        for i in range(150):
            self.detector.update(i, 0.8, 0.75)

        is_converged, aci_value = self.detector.check_convergence()

        # Should converge with stable values
        self.assertTrue(is_converged)
        self.assertIsNotNone(aci_value)
        self.assertLess(aci_value, self.config.aci_stability_threshold)

    def test_no_convergence_volatile(self):
        """Test that volatile confidence does not converge."""
        # Add volatile confidence values
        for i in range(150):
            confidence = 0.5 + 0.3 * (i % 10) / 10.0  # Oscillating
            self.detector.update(i, confidence, 0.4)

        is_converged, aci_value = self.detector.check_convergence()

        # Should not converge with volatile values
        self.assertFalse(is_converged)


class TestHypothesisDLQ(unittest.TestCase):
    """Test Dead Letter Queue."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = DEELogger()
        self.dlq = HypothesisDLQ(self.logger)

    def test_add_to_dlq(self):
        """Test adding failed hypothesis to DLQ."""
        hypothesis = Hypothesis(
            statement="Failed hypothesis",
            type="test",
            domain="test"
        )

        self.dlq.add(hypothesis, "Validation failed", "validation")

        self.assertEqual(self.dlq.size(), 1)
        self.assertEqual(len(self.dlq.get_all()), 1)

    def test_dlq_max_size(self):
        """Test DLQ max size enforcement."""
        self.dlq.max_size = 5

        # Add more hypotheses than max size
        for i in range(10):
            hypothesis = Hypothesis(
                statement=f"Hypothesis {i}",
                type="test",
                domain="test"
            )
            self.dlq.add(hypothesis, f"Error {i}", "validation")

        # Should only keep max_size hypotheses
        self.assertEqual(self.dlq.size(), 5)

    def test_clear_dlq(self):
        """Test clearing DLQ."""
        hypothesis = Hypothesis(statement="Test", type="test", domain="test")
        self.dlq.add(hypothesis, "Error", "validation")

        self.assertEqual(self.dlq.size(), 1)

        self.dlq.clear()

        self.assertEqual(self.dlq.size(), 0)


class TestMCTSSearchExecutor(unittest.TestCase):
    """Test MC-NEST executor."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Phase3Config.from_env()
        self.logger = DEELogger()
        self.executor = MCTSSearchExecutor(self.config, self.logger)

    def test_execute_search(self):
        """Test executing MC-NEST search."""
        # Create root hypothesis
        root_hypothesis = Hypothesis(
            statement="Root hypothesis",
            type="test",
            domain="test_domain",
            confidence=0.5
        )

        # Define hypothesis generator
        def hypothesis_generator():
            children = []
            for i in range(3):
                child = Hypothesis(
                    statement=f"Child {i}",
                    type="test",
                    domain="test_domain",
                    confidence=0.6
                )
                children.append(child)
            return children

        # Define reward function
        def reward_function(hypothesis):
            return hypothesis.confidence + random.uniform(-0.05, 0.05)

        # Execute search
        search_result, error = self.executor.execute_search(
            root_hypothesis=root_hypothesis,
            hypothesis_generator=hypothesis_generator,
            reward_function=reward_function,
        )

        self.assertIsNone(error)
        self.assertIsNotNone(search_result)
        self.assertIsNotNone(search_result.search_id)
        self.assertGreater(search_result.iterations, 0)
        self.assertGreater(search_result.total_nodes, 0)
        self.assertGreater(search_result.execution_time_ms, 0)

    def test_search_timeout(self):
        """Test search timeout enforcement."""
        # Set very short timeout
        self.config.timeout_ms = 100
        self.config.iterations = 10000  # Try to do many iterations

        root_hypothesis = Hypothesis(statement="Root", type="test", domain="test")

        def hypothesis_generator():
            time.sleep(0.01)  # Slow generation
            return [Hypothesis(statement="Child", type="test", domain="test")]

        def reward_function(hypothesis):
            return 0.5

        # Execute (should timeout quickly)
        search_result, error = self.executor.execute_search(
            root_hypothesis=root_hypothesis,
            hypothesis_generator=hypothesis_generator,
            reward_function=reward_function,
        )

        self.assertIsNone(error)
        self.assertIsNotNone(search_result)
        # Should complete fewer iterations due to timeout
        self.assertLess(search_result.iterations, self.config.iterations)


class TestPhase3Adapter(unittest.TestCase):
    """Test Phase III adapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.adapter = Phase3Adapter()

    def test_search_request(self):
        """Test search through adapter."""
        request = {
            "root_hypothesis": {
                "statement": "Test root hypothesis",
                "type": "test",
                "domain": "test_domain",
                "confidence": 0.5,
            },
            "num_children": 3,
        }

        result = self.adapter.search(request)

        self.assertTrue(result.get("success", False))
        self.assertIn("search_id", result)
        self.assertIn("best_hypothesis", result)
        self.assertIn("tree_statistics", result)

    def test_validate_hypothesis_request(self):
        """Test hypothesis validation through adapter."""
        # Generate rewards
        rewards = [0.7 + random.uniform(-0.05, 0.05) for _ in range(50)]

        request = {
            "hypothesis": {
                "statement": "Test hypothesis",
                "type": "test",
                "domain": "test",
                "confidence": 0.7,
            },
            "rewards": rewards,
        }

        result = self.adapter.validate_hypothesis(request)

        self.assertTrue(result.get("success", False))
        self.assertIn("validation_result", result)

    def test_check_convergence_request(self):
        """Test convergence check through adapter."""
        request = {
            "iteration": 100,
            "best_confidence": 0.8,
            "best_reward": 0.75,
        }

        result = self.adapter.check_convergence(request)

        self.assertTrue(result.get("success", False))
        self.assertIn("is_converged", result)
        self.assertIn("aci_value", result)

    def test_get_health(self):
        """Test health check."""
        health = self.adapter.get_health()

        self.assertIn("status", health)
        self.assertIn("circuit_breaker_state", health)
        self.assertIn("dlq_size", health)

    def test_invalid_request(self):
        """Test handling of invalid request."""
        request = {}  # Missing root_hypothesis

        result = self.adapter.search(request)

        # Should fail validation
        self.assertFalse(result.get("success", True))
        self.assertIn("error", result)


class TestIntegration(unittest.TestCase):
    """Integration tests for Phase III."""

    def test_end_to_end_search(self):
        """Test end-to-end search flow."""
        # Create adapter
        adapter = Phase3Adapter()

        # Execute search
        request = {
            "root_hypothesis": {
                "statement": "Integration test root hypothesis",
                "type": "integration_test",
                "domain": "test_domain",
                "confidence": 0.5,
            },
            "num_children": 5,
        }

        result = adapter.search(request)

        # Verify result
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["search_id"])
        self.assertGreater(result["tree_statistics"]["iterations"], 0)
        self.assertGreater(result["tree_statistics"]["total_nodes"], 0)
        self.assertGreater(result["execution_time_ms"], 0)

        # Verify best hypothesis
        best_hypothesis = result["best_hypothesis"]
        self.assertIsNotNone(best_hypothesis)
        self.assertIn("hypothesis_id", best_hypothesis)
        self.assertIn("confidence", best_hypothesis)

    def test_search_with_validation(self):
        """Test search followed by validation."""
        adapter = Phase3Adapter()

        # Execute search
        request = {
            "root_hypothesis": {
                "statement": "Validation test",
                "type": "test",
                "domain": "test",
                "confidence": 0.5,
            },
        }

        search_result = adapter.search(request)

        # Validate best hypothesis
        best_hypothesis = search_result["best_hypothesis"]
        rewards = [0.7 + random.uniform(-0.05, 0.05) for _ in range(50)]

        validation_request = {
            "hypothesis": best_hypothesis,
            "rewards": rewards,
        }

        validation_result = adapter.validate_hypothesis(validation_request)

        self.assertTrue(validation_result["success"])
        self.assertIn("validation_result", validation_result)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3Config))
    suite.addTests(loader.loadTestsFromTestCase(TestUCB1SelectionStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchTreeBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestHypothesisValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestConvergenceDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestHypothesisDLQ))
    suite.addTests(loader.loadTestsFromTestCase(TestMCTSSearchExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3Adapter))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    run_tests()
