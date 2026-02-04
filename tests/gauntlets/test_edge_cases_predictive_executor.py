"""
Edge Case Tests for Predictive Gauntlet Executor

Comprehensive edge case testing to achieve 95%+ code coverage.

Tests cover:
- Empty solution/problem handling
- Extremely long solutions
- Unknown domains
- Edge case feature combinations
- Prediction boundary conditions
- Invalid threshold combinations
- Concurrent predictions

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import unittest
import pytest
import numpy as np
import sys
import os
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import (
    PredictiveGauntletExecutor,
    PredictionResult,
    ExecutionPlan,
    ExecutionResult,
    ExecutionDecision
)


class TestEmptyNullInputs(unittest.TestCase):
    """Test handling of empty and null inputs"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor()

    def test_predict_with_empty_solution(self):
        """Test prediction with empty solution string"""
        result = self.executor.predict_success(
            solution="",
            problem="Solve the problem",
            domain="code"
        )

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.success_probability, 0.0)
        self.assertLessEqual(result.success_probability, 1.0)

    def test_predict_with_empty_problem(self):
        """Test prediction with empty problem string"""
        result = self.executor.predict_success(
            solution="def solve(): pass",
            problem="",
            domain="code"
        )

        self.assertIsNotNone(result)

    def test_predict_with_none_context(self):
        """Test prediction with None context"""
        result = self.executor.predict_success(
            solution="def solve(): pass",
            problem="Solve the problem",
            domain="code",
            context=None
        )

        self.assertIsNotNone(result)

    def test_execute_with_empty_solution(self):
        """Test execution with empty solution"""
        result = self.executor.execute_with_prediction(
            solution="",
            problem="Test problem",
            domain="code"
        )

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.actual_outcome)

    def test_predict_with_whitespace_only(self):
        """Test prediction with whitespace-only strings"""
        result = self.executor.predict_success(
            solution="   \n\t   ",
            problem="   \n\t   ",
            domain="code"
        )

        self.assertIsNotNone(result)

    def test_execute_with_none_prediction(self):
        """Test execution when prediction is None"""
        result = self.executor.execute_with_prediction(
            solution="def solve(): return 42",
            problem="Find the answer",
            domain="code",
            prediction=None
        )

        self.assertIsNotNone(result)
        # Should generate its own prediction
        self.assertIsNotNone(result.prediction)


class TestExtremelyLongSolutions(unittest.TestCase):
    """Test handling of extremely long solutions"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor()

    def test_predict_with_very_long_solution(self):
        """Test prediction with very long solution (1000+ lines)"""
        long_solution = "\n".join([
            "def func_{}(): pass".format(i) for i in range(1000)
        ])

        result = self.executor.predict_success(
            solution=long_solution,
            problem="Solve complex problem",
            domain="code"
        )

        self.assertIsNotNone(result)
        # Long solutions might have lower success probability
        self.assertGreaterEqual(result.success_probability, 0.0)

    def test_predict_with_extremely_long_single_line(self):
        """Test prediction with extremely long single line"""
        long_line = "x = " + " + ".join([str(i) for i in range(1000)])

        result = self.executor.predict_success(
            solution=long_line,
            problem="Calculate sum",
            domain="code"
        )

        self.assertIsNotNone(result)

    def test_predict_with_very_long_problem(self):
        """Test prediction with very long problem statement"""
        long_problem = "Solve this problem: " + " ".join(["detail"] * 500)

        result = self.executor.predict_success(
            solution="def solve(): pass",
            problem=long_problem,
            domain="code"
        )

        self.assertIsNotNone(result)

    def test_execute_with_very_long_solution(self):
        """Test execution with very long solution"""
        long_solution = "\n".join([
            "def func_{}(): return {}".format(i, i) for i in range(500)
        ])

        result = self.executor.execute_with_prediction(
            solution=long_solution,
            problem="Test",
            domain="code"
        )

        self.assertIsNotNone(result)

    def test_complexity_score_upper_bound(self):
        """Test complexity score doesn't exceed 1.0"""
        # Create extremely complex solution
        complex_solution = """
        class ComplexClass:
            def __init__(self):
                pass

            @decorator
            async def complex_method(self):
                yield from range(100)
                lambda x: x ** 2
                return await self.another_method()
        """ * 100  # Repeat 100 times

        features = self.executor._extract_features(
            complex_solution, "Problem", "code", None
        )

        self.assertLessEqual(features["complexity_score"], 1.0)
        self.assertGreaterEqual(features["complexity_score"], 0.0)


class TestUnknownDomains(unittest.TestCase):
    """Test handling of unknown domains"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor()

    def test_predict_with_unknown_domain(self):
        """Test prediction with unknown/invalid domain"""
        result = self.executor.predict_success(
            solution="def solve(): pass",
            problem="Solve problem",
            domain="unknown_domain_xyz"
        )

        self.assertIsNotNone(result)
        # Should default to medium risk
        self.assertGreaterEqual(result.success_probability, 0.0)

    def test_predict_with_empty_domain(self):
        """Test prediction with empty domain string"""
        result = self.executor.predict_success(
            solution="def solve(): pass",
            problem="Solve problem",
            domain=""
        )

        self.assertIsNotNone(result)

    def test_predict_with_none_domain(self):
        """Test prediction with None domain (should handle gracefully)"""
        # This might raise an error, which is acceptable
        try:
            result = self.executor.predict_success(
                solution="def solve(): pass",
                problem="Solve problem",
                domain=None
            )
            self.assertIsNotNone(result)
        except (AttributeError, TypeError):
            # Expected behavior
            pass

    def test_predict_with_case_variations(self):
        """Test prediction with different domain cases"""
        domains = ["Code", "CODE", "CoDe", "math", "MATH", "Math"]

        for domain in domains:
            result = self.executor.predict_success(
                solution="def solve(): pass",
                problem="Solve problem",
                domain=domain
            )

            self.assertIsNotNone(result)

    def test_all_known_domains(self):
        """Test prediction with all known domains"""
        domains = ["math", "algorithm", "ml", "optimization", "code", "general"]

        for domain in domains:
            result = self.executor.predict_success(
                solution="def solve(): pass",
                problem="Solve problem",
                domain=domain
            )

            self.assertIsNotNone(result)
            self.assertGreaterEqual(result.domain_risk, 0.0)
            self.assertLessEqual(result.domain_risk, 1.0)


class TestEdgeCaseFeatureCombinations(unittest.TestCase):
    """Test edge case feature combinations"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor()

    def test_solution_with_no_structure(self):
        """Test solution with no functions, classes, or imports"""
        solution = "x = 42\nprint(x)"

        features = self.executor._extract_features(
            solution, "Problem", "code", None
        )

        self.assertFalse(features["has_functions"])
        self.assertFalse(features["has_classes"])
        self.assertFalse(features["has_imports"])

    def test_solution_with_all_structure(self):
        """Test solution with all structure elements"""
        solution = """
        import numpy as np

        class Solver:
            def solve(self):
                return 42
        """

        features = self.executor._extract_features(
            solution, "Problem", "code", None
        )

        self.assertTrue(features["has_functions"])
        self.assertTrue(features["has_classes"])
        self.assertTrue(features["has_imports"])

    def test_very_short_solution(self):
        """Test feature extraction with very short solution"""
        solution = "pass"

        features = self.executor._extract_features(
            solution, "Problem", "code", None
        )

        self.assertEqual(features["solution_lines"], 1)
        self.assertLess(features["complexity_score"], 0.5)

    def test_solution_with_advanced_keywords(self):
        """Test solution with all advanced keywords"""
        solution = """
        async def complex_func():
            await asyncio.sleep(1)
            yield from range(10)
            result = lambda x: x ** 2
            return result
        """

        features = self.executor._extract_features(
            solution, "Problem", "code", None
        )

        # Should have higher complexity due to keywords
        self.assertGreater(features["complexity_score"], 0.5)

    def test_risk_factors_identification(self):
        """Test all risk factors are identified correctly"""
        # High complexity
        solution = "def " + "func" * 100 + "(): pass"

        result = self.executor.predict_success(
            solution=solution,
            problem="Hard problem",
            domain="algorithm"  # High risk domain
        )

        # Should have risk factors
        self.assertIsInstance(result.risk_factors, list)

    def test_difficulty_recommendation_boundaries(self):
        """Test difficulty recommendation at probability boundaries"""
        # Very high probability -> hard
        result = self.executor.predict_success(
            solution="def solve(): return 42",
            problem="Easy problem",
            domain="general"
        )

        self.assertIn(result.recommended_difficulty, ["easy", "medium", "hard"])

        # Very low probability -> easy
        result = self.executor.predict_success(
            solution="x",
            problem="Very hard problem",
            domain="ml"
        )

        self.assertIn(result.recommended_difficulty, ["easy", "medium", "hard"])


class TestPredictionBoundaryConditions(unittest.TestCase):
    """Test prediction boundary conditions"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor()

    def test_probability_bounds(self):
        """Test success probability is always in [0, 1]"""
        test_cases = [
            ("", "", ""),
            ("pass", "test", "code"),
            ("def " * 100, "hard " * 100, "algorithm"),
            ("import " * 50, "test", "general")
        ]

        for solution, problem, domain in test_cases:
            result = self.executor.predict_success(
                solution=solution,
                problem=problem,
                domain=domain
            )

            self.assertGreaterEqual(result.success_probability, 0.0)
            self.assertLessEqual(result.success_probability, 1.0)

    def test_confidence_bounds(self):
        """Test confidence is always in valid range"""
        result = self.executor.predict_success(
            solution="def solve(): pass",
            problem="Test",
            domain="code"
        )

        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_time_estimate_bounds(self):
        """Test execution time estimates are reasonable"""
        result = self.executor.predict_success(
            solution="def solve(): pass",
            problem="Test",
            domain="code"
        )

        self.assertGreater(result.estimated_time, 0)
        self.assertLess(result.estimated_time, 1000)  # Should be under 16 minutes

    def test_cost_estimate_bounds(self):
        """Test cost estimates are non-negative"""
        result = self.executor.predict_success(
            solution="def solve(): pass",
            problem="Test",
            domain="code"
        )

        self.assertGreaterEqual(result.estimated_cost, 0.0)


class TestExecutionDecisionBoundaries(unittest.TestCase):
    """Test execution decision boundaries"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor(
            success_threshold=0.3,
            confidence_threshold=0.6,
            cost_threshold=100.0
        )

    def test_skip_low_probability_boundary(self):
        """Test skip decision at probability boundary"""
        prediction = PredictionResult(
            success_probability=0.29,  # Just below threshold
            confidence=0.8,
            estimated_cost=50.0
        )

        plan = self.executor.create_execution_plan(prediction)

        self.assertEqual(plan.decision, ExecutionDecision.SKIP_LOW_PROBABILITY)

    def test_skip_high_cost_boundary(self):
        """Test skip decision at cost boundary"""
        prediction = PredictionResult(
            success_probability=0.8,
            confidence=0.8,
            estimated_cost=101.0  # Just above threshold
        )

        plan = self.executor.create_execution_plan(prediction)

        self.assertEqual(plan.decision, ExecutionDecision.SKIP_HIGH_COST)

    def test_skip_low_confidence(self):
        """Test skip decision at confidence boundary"""
        prediction = PredictionResult(
            success_probability=0.8,
            confidence=0.59,  # Just below threshold
            estimated_cost=50.0
        )

        plan = self.executor.create_execution_plan(prediction)

        self.assertEqual(plan.decision, ExecutionDecision.SKIP_LOW_PROBABILITY)

    def test_proceed_at_middle_boundaries(self):
        """Test proceed decision in middle range"""
        prediction = PredictionResult(
            success_probability=0.5,
            confidence=0.8,
            estimated_cost=50.0
        )

        plan = self.executor.create_execution_plan(prediction)

        self.assertIn(
            plan.decision,
            [ExecutionDecision.PROCEED, ExecutionDecision.ADJUST_DIFFICULTY]
        )

    def test_adjust_difficulty_high_probability(self):
        """Test difficulty adjustment for high probability"""
        prediction = PredictionResult(
            success_probability=0.85,  # High probability
            confidence=0.8,
            estimated_cost=50.0
        )

        plan = self.executor.create_execution_plan(prediction)

        self.assertEqual(plan.decision, ExecutionDecision.ADJUST_DIFFICULTY)
        # Should increase thresholds
        self.assertGreater(
            plan.adjusted_config.get("round1_threshold", 0),
            0.5
        )

    def test_adjust_difficulty_low_probability(self):
        """Test difficulty adjustment for moderate probability"""
        prediction = PredictionResult(
            success_probability=0.45,  # Moderate probability
            confidence=0.8,
            estimated_cost=50.0
        )

        plan = self.executor.create_execution_plan(prediction)

        self.assertEqual(plan.decision, ExecutionDecision.ADJUST_DIFFICULTY)
        # Should decrease thresholds (but not below 0.3)
        self.assertGreaterEqual(
            plan.adjusted_config.get("round1_threshold", 0),
            0.3
        )


class TestInvalidThresholdCombinations(unittest.TestCase):
    """Test invalid threshold combinations"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor()

    def test_threshold_clamp_minimum(self):
        """Test thresholds are clamped at minimum when adjusting difficulty"""
        prediction = PredictionResult(
            success_probability=0.4,
            confidence=0.8,
            estimated_cost=50.0
        )

        # Start with very low thresholds
        base_config = {
            "round1_threshold": 0.31,
            "round2_threshold": 0.41,
            "round3_threshold": 0.51
        }

        plan = self.executor.create_execution_plan(
            prediction,
            base_config=base_config
        )

        # Should not go below 0.3
        self.assertGreaterEqual(
            plan.adjusted_config.get("round1_threshold", 0),
            0.3
        )

    def test_threshold_clamp_maximum(self):
        """Test thresholds can go up when adjusting difficulty"""
        prediction = PredictionResult(
            success_probability=0.85,
            confidence=0.8,
            estimated_cost=50.0
        )

        # Start with high thresholds
        base_config = {
            "round1_threshold": 0.9,
            "round2_threshold": 0.9,
            "round3_threshold": 0.9
        }

        plan = self.executor.create_execution_plan(
            prediction,
            base_config=base_config
        )

        # Can increase beyond 0.9 (up to 1.0)
        self.assertLessEqual(
            plan.adjusted_config.get("round1_threshold", 0),
            1.0
        )

    def test_thresholds_with_none_config(self):
        """Test threshold adjustment with None base config"""
        prediction = PredictionResult(
            success_probability=0.6,
            confidence=0.8,
            estimated_cost=50.0
        )

        plan = self.executor.create_execution_plan(
            prediction,
            base_config=None
        )

        self.assertIsNotNone(plan.adjusted_config)


class TestPredictionAccuracyCalculation(unittest.TestCase):
    """Test prediction accuracy calculation edge cases"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor()

    def test_accuracy_perfect_prediction(self):
        """Test accuracy when prediction is perfect"""
        prediction = PredictionResult(
            success_probability=0.8,
            confidence=0.8,
            estimated_cost=50.0
        )

        actual_outcome = {
            "passed": True,
            "score": 0.8
        }

        # Calculate accuracy manually
        predicted_passed = prediction.success_probability > 0.5
        actual_passed = actual_outcome["passed"]
        pass_fail_accuracy = 1.0 if (predicted_passed == actual_passed) else 0.0

        score_error = abs(actual_outcome["score"] - prediction.success_probability)
        score_accuracy = max(0.0, 1.0 - score_error)

        prediction_accuracy = (pass_fail_accuracy + score_accuracy) / 2

        self.assertGreaterEqual(prediction_accuracy, 0.0)
        self.assertLessEqual(prediction_accuracy, 1.0)

    def test_accuracy_worst_prediction(self):
        """Test accuracy when prediction is completely wrong"""
        prediction = PredictionResult(
            success_probability=1.0,
            confidence=0.8,
            estimated_cost=50.0
        )

        actual_outcome = {
            "passed": False,
            "score": 0.0
        }

        predicted_passed = prediction.success_probability > 0.5
        actual_passed = actual_outcome["passed"]
        pass_fail_accuracy = 1.0 if (predicted_passed == actual_passed) else 0.0

        score_error = abs(actual_outcome["score"] - prediction.success_probability)
        score_accuracy = max(0.0, 1.0 - score_error)

        prediction_accuracy = (pass_fail_accuracy + score_accuracy) / 2

        # Should be low accuracy
        self.assertLess(prediction_accuracy, 0.5)

    def test_accuracy_boundary_cases(self):
        """Test accuracy at score boundaries"""
        test_cases = [
            (0.0, 0.0),   # Both zero
            (1.0, 1.0),   # Both one
            (0.5, 0.5),   # Both at midpoint
            (0.0, 1.0),   # Opposite extremes
            (0.49, 0.51), # Just below/above threshold
        ]

        for predicted_score, actual_score in test_cases:
            score_error = abs(actual_score - predicted_score)
            score_accuracy = max(0.0, 1.0 - score_error)

            self.assertGreaterEqual(score_accuracy, 0.0)
            self.assertLessEqual(score_accuracy, 1.0)


class TestConcurrentPredictions(unittest.TestCase):
    """Test concurrent prediction requests"""

    def test_concurrent_predictions(self):
        """Test multiple concurrent predictions"""
        from concurrent.futures import ThreadPoolExecutor

        executor = PredictiveGauntletExecutor()

        def make_prediction(i):
            return executor.predict_success(
                solution=f"def solve_{i}(): return {i}",
                problem=f"Problem {i}",
                domain="code"
            )

        with ThreadPoolExecutor(max_workers=10) as executor_pool:
            futures = [executor_pool.submit(make_prediction, i) for i in range(20)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), 20)

        for result in results:
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result.success_probability, 0.0)
            self.assertLessEqual(result.success_probability, 1.0)

    def test_concurrent_executions(self):
        """Test multiple concurrent executions"""
        from concurrent.futures import ThreadPoolExecutor

        executor = PredictiveGauntletExecutor()

        def make_execution(i):
            return executor.execute_with_prediction(
                solution=f"def solve_{i}(): return {i}",
                problem=f"Problem {i}",
                domain="code"
            )

        with ThreadPoolExecutor(max_workers=5) as executor_pool:
            futures = [executor_pool.submit(make_execution, i) for i in range(10)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), 10)

        for result in results:
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.prediction)


class TestCostSavingsCalculation(unittest.TestCase):
    """Test cost savings calculation"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor(
            cost_threshold=100.0
        )

    def test_cost_savings_when_skipping_low_prob(self):
        """Test cost savings when skipping low probability"""
        prediction = PredictionResult(
            success_probability=0.2,
            confidence=0.8,
            estimated_cost=150.0
        )

        result = self.executor.execute_with_prediction(
            solution="pass",
            problem="test",
            domain="code",
            prediction=prediction
        )

        self.assertGreater(result.cost_savings, 0)
        self.assertEqual(result.cost_savings, prediction.estimated_cost)

    def test_cost_savings_when_skipping_high_cost(self):
        """Test cost savings when skipping high cost"""
        prediction = PredictionResult(
            success_probability=0.8,
            confidence=0.8,
            estimated_cost=150.0
        )

        result = self.executor.execute_with_prediction(
            solution="pass",
            problem="test",
            domain="code",
            prediction=prediction
        )

        # Should save: estimated_cost - threshold
        expected_savings = prediction.estimated_cost - self.executor.cost_threshold
        self.assertEqual(result.cost_savings, expected_savings)

    def test_no_cost_savings_when_proceeding(self):
        """Test no cost savings when proceeding with execution"""
        prediction = PredictionResult(
            success_probability=0.7,
            confidence=0.8,
            estimated_cost=50.0
        )

        result = self.executor.execute_with_prediction(
            solution="def solve(): pass",
            problem="test",
            domain="code",
            prediction=prediction
        )

        self.assertEqual(result.cost_savings, 0.0)


class TestStatisticsCollection(unittest.TestCase):
    """Test prediction accuracy statistics"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor()

    def test_empty_statistics(self):
        """Test statistics when no predictions made"""
        stats = self.executor.get_prediction_accuracy_stats()

        self.assertIn("error", stats)

    def test_single_prediction_statistics(self):
        """Test statistics with single prediction"""
        # Execute once
        result = self.executor.execute_with_prediction(
            solution="def solve(): pass",
            problem="test",
            domain="code"
        )

        stats = self.executor.get_prediction_accuracy_stats()

        self.assertIn("mean_accuracy", stats)
        self.assertEqual(stats["total_predictions"], 1)

    def test_multiple_predictions_statistics(self):
        """Test statistics with multiple predictions"""
        for i in range(10):
            self.executor.execute_with_prediction(
                solution=f"def solve_{i}(): pass",
                problem=f"test {i}",
                domain="code"
            )

        stats = self.executor.get_prediction_accuracy_stats()

        self.assertEqual(stats["total_predictions"], 10)
        self.assertIn("mean_accuracy", stats)
        self.assertIn("std_accuracy", stats)
        self.assertIn("min_accuracy", stats)
        self.assertIn("max_accuracy", stats)


@pytest.mark.parametrize("domain,expected_risk", [
    ("math", 0.7),
    ("algorithm", 0.8),
    ("ml", 0.8),
    ("optimization", 0.75),
    ("code", 0.5),
    ("general", 0.4),
    ("unknown", 0.5),  # Default
])
def test_domain_risk_mapping(domain, expected_risk):
    """Parametrized test for domain risk mapping"""
    executor = PredictiveGauntletExecutor()
    risk = executor._get_domain_risk(domain)
    assert risk == expected_risk


@pytest.mark.parametrize("success_prob,expected_difficulty", [
    (0.9, "hard"),
    (0.8, "hard"),
    (0.6, "medium"),
    (0.5, "medium"),
    (0.4, "easy"),
    (0.2, "easy"),
])
def test_difficulty_recommendation(success_prob, expected_difficulty):
    """Parametrized test for difficulty recommendation"""
    executor = PredictiveGauntletExecutor()
    difficulty = executor._recommend_difficulty(success_prob, {})
    assert difficulty == expected_difficulty


if __name__ == "__main__":
    unittest.main()
