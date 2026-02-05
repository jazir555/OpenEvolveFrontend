"""
Test Suite for Predictive Gauntlet Executor

Comprehensive tests for the predictive gauntlet executor component.

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

from predictive_gauntlet_executor import (
    PredictiveGauntletExecutor,
    PredictionResult,
    ExecutionPlan,
    ExecutionResult,
    ExecutionDecision
)


class TestPredictionResult(unittest.TestCase):
    """Test PredictionResult dataclass"""

    def test_prediction_result_creation(self):
        """Test creating a prediction result"""
        result = PredictionResult(
            success_probability=0.75,
            confidence=0.8,
            risk_factors=["High complexity", "Challenging domain"]
        )

        self.assertEqual(result.success_probability, 0.75)
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(len(result.risk_factors), 2)

    def test_prediction_result_to_dict(self):
        """Test converting prediction result to dictionary"""
        result = PredictionResult(
            success_probability=0.75,
            confidence=0.8,
            recommended_difficulty="hard"
        )

        data = result.to_dict()
        self.assertEqual(data["success_probability"], 0.75)
        self.assertEqual(data["confidence"], 0.8)
        self.assertEqual(data["recommended_difficulty"], "hard")


class TestPredictiveGauntletExecutor(unittest.TestCase):
    """Test Predictive Gauntlet Executor"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor(
            success_threshold=0.3,
            confidence_threshold=0.6,
            cost_threshold=100.0
        )

    def test_executor_initialization(self):
        """Test executor initialization"""
        self.assertEqual(self.executor.success_threshold, 0.3)
        self.assertEqual(self.executor.confidence_threshold, 0.6)
        self.assertEqual(self.executor.cost_threshold, 100.0)

    def test_predict_success_returns_valid_result(self):
        """Test that predict_success returns valid result"""
        prediction = self.executor.predict_success(
            solution="def solve(): return optimal",
            problem="Optimize portfolio",
            domain="finance"
        )

        self.assertIsInstance(prediction, PredictionResult)
        self.assertGreaterEqual(prediction.success_probability, 0.0)
        self.assertLessEqual(prediction.success_probability, 1.0)
        self.assertGreaterEqual(prediction.confidence, 0.0)
        self.assertLessEqual(prediction.confidence, 1.0)
        self.assertIsInstance(prediction.risk_factors, list)
        self.assertIsInstance(prediction.recommended_difficulty, str)

    def test_predict_success_different_domains(self):
        """Test predictions for different domains"""
        solution = "def solve(): return optimal"
        problem = "Solve the problem"

        domains = ["code", "math", "algorithm", "general"]
        results = {}

        for domain in domains:
            prediction = self.executor.predict_success(
                solution=solution,
                problem=problem,
                domain=domain
            )
            results[domain] = prediction

        # Should produce valid predictions for all domains
        for domain, prediction in results.items():
            self.assertGreaterEqual(prediction.success_probability, 0.0)
            self.assertLessEqual(prediction.success_probability, 1.0)

    def test_create_execution_plan_proceed(self):
        """Test execution plan for proceeding"""
        prediction = PredictionResult(
            success_probability=0.7,
            confidence=0.8,
            estimated_cost=50.0
        )

        plan = self.executor.create_execution_plan(prediction)

        self.assertEqual(plan.decision, ExecutionDecision.PROCEED)
        self.assertIn("acceptable", plan.reasoning.lower())

    def test_create_execution_plan_skip_low_probability(self):
        """Test execution plan skips low probability"""
        prediction = PredictionResult(
            success_probability=0.2,  # Below threshold
            confidence=0.8,
            estimated_cost=50.0
        )

        plan = self.executor.create_execution_plan(prediction)

        self.assertEqual(plan.decision, ExecutionDecision.SKIP_LOW_PROBABILITY)
        self.assertIn("below threshold", plan.reasoning.lower())

    def test_create_execution_plan_skip_high_cost(self):
        """Test execution plan skips high cost"""
        prediction = PredictionResult(
            success_probability=0.7,
            confidence=0.8,
            estimated_cost=150.0  # Above threshold
        )

        plan = self.executor.create_execution_plan(prediction)

        self.assertEqual(plan.decision, ExecutionDecision.SKIP_HIGH_COST)
        self.assertIn("exceeds threshold", plan.reasoning.lower())

    def test_create_execution_plan_adjust_difficulty(self):
        """Test execution plan adjusts difficulty"""
        # High success probability should increase difficulty
        prediction = PredictionResult(
            success_probability=0.9,
            confidence=0.8,
            estimated_cost=50.0
        )

        plan = self.executor.create_execution_plan(prediction)

        self.assertEqual(plan.decision, ExecutionDecision.ADJUST_DIFFICULTY)
        self.assertIn("increasing difficulty", plan.reasoning.lower())
        self.assertGreater(
            plan.adjusted_config.get("round1_threshold", 0.5),
            0.5
        )

    def test_execute_with_prediction_skips(self):
        """Test execution skips when prediction is low"""
        prediction = PredictionResult(
            success_probability=0.2,
            confidence=0.8,
            estimated_cost=50.0
        )

        result = self.executor.execute_with_prediction(
            solution="def solve(): return optimal",
            problem="Test problem",
            domain="code",
            prediction=prediction
        )

        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.actual_outcome.get("skipped"), True)
        self.assertGreater(result.cost_savings, 0)

    def test_execute_with_prediction_proceeds(self):
        """Test execution proceeds when prediction is good"""
        prediction = PredictionResult(
            success_probability=0.7,
            confidence=0.8,
            estimated_cost=50.0
        )

        result = self.executor.execute_with_prediction(
            solution="def solve(): return optimal",
            problem="Test problem",
            domain="code",
            prediction=prediction
        )

        self.assertIsInstance(result, ExecutionResult)
        self.assertFalse(result.actual_outcome.get("skipped", False))

    def test_prediction_accuracy_tracking(self):
        """Test that prediction accuracy is tracked"""
        # Make several predictions and executions
        for i in range(5):
            prediction = self.executor.predict_success(
                solution=f"def solve_{i}(): return {i}",
                problem=f"Problem {i}",
                domain="code"
            )

            result = self.executor.execute_with_prediction(
                solution=f"def solve_{i}(): return {i}",
                problem=f"Problem {i}",
                domain="code",
                prediction=prediction
            )

        # Get accuracy stats
        stats = self.executor.get_prediction_accuracy_stats()

        self.assertIn("mean_accuracy", stats)
        self.assertIn("total_predictions", stats)
        self.assertEqual(stats["total_predictions"], 5)


class TestPredictiveExecutorEdgeCases(unittest.TestCase):
    """Edge case tests for predictive executor"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor()

    def test_predict_with_empty_solution(self):
        """Test prediction with empty solution"""
        prediction = self.executor.predict_success(
            solution="",
            problem="Test problem",
            domain="code"
        )

        # Should still return valid prediction
        self.assertGreaterEqual(prediction.success_probability, 0.0)
        self.assertLessEqual(prediction.success_probability, 1.0)

    def test_predict_with_very_long_solution(self):
        """Test prediction with very long solution"""
        long_solution = "def solve():\n" + "    pass\n" * 1000

        prediction = self.executor.predict_success(
            solution=long_solution,
            problem="Test problem",
            domain="code"
        )

        # Should still return valid prediction
        self.assertGreaterEqual(prediction.success_probability, 0.0)
        self.assertLessEqual(prediction.success_probability, 1.0)

    def test_predict_with_unknown_domain(self):
        """Test prediction with unknown domain"""
        prediction = self.executor.predict_success(
            solution="def solve(): return optimal",
            problem="Test problem",
            domain="unknown_domain_xyz"
        )

        # Should still return valid prediction
        self.assertGreaterEqual(prediction.success_probability, 0.0)
        self.assertLessEqual(prediction.success_probability, 1.0)

    def test_execute_without_prediction(self):
        """Test execution without providing prediction"""
        result = self.executor.execute_with_prediction(
            solution="def solve(): return optimal",
            problem="Test problem",
            domain="code",
            prediction=None  # Will be generated
        )

        self.assertIsInstance(result, ExecutionResult)
        self.assertIsNotNone(result.prediction)

    def test_extreme_thresholds(self):
        """Test with extreme threshold values"""
        executor = PredictiveGauntletExecutor(
            success_threshold=0.0,  # Always proceed
            confidence_threshold=0.0,
            cost_threshold=float('inf')
        )

        prediction = PredictionResult(
            success_probability=0.0,
            confidence=0.0,
            estimated_cost=0.0
        )

        plan = executor.create_execution_plan(prediction)

        # Should proceed even with zero values
        self.assertIn(plan.decision, [
            ExecutionDecision.PROCEED,
            ExecutionDecision.ADJUST_DIFFICULTY
        ])


class TestPredictiveExecutorIntegration(unittest.TestCase):
    """Integration tests for predictive executor"""

    def setUp(self):
        """Set up test fixtures"""
        self.executor = PredictiveGauntletExecutor()

    def test_full_workflow(self):
        """Test complete workflow from prediction to execution"""
        # Step 1: Predict
        prediction = self.executor.predict_success(
            solution="def solve(): return 42",
            problem="Return the answer to life",
            domain="code"
        )

        # Step 2: Create plan
        plan = self.executor.create_execution_plan(prediction)

        # Step 3: Execute
        result = self.executor.execute_with_prediction(
            solution="def solve(): return 42",
            problem="Return the answer to life",
            domain="code",
            prediction=prediction
        )

        # Verify workflow
        self.assertIsNotNone(prediction)
        self.assertIsNotNone(plan)
        self.assertIsNotNone(result)
        self.assertEqual(result.prediction, prediction)

    def test_batch_predictions(self):
        """Test making multiple predictions"""
        solutions = [
            "def solve1(): return 1",
            "def solve2(): return 2",
            "def solve3(): return 3"
        ]

        predictions = []
        for solution in solutions:
            prediction = self.executor.predict_success(
                solution=solution,
                problem="Test",
                domain="code"
            )
            predictions.append(prediction)

        # All predictions should be valid
        self.assertEqual(len(predictions), 3)
        for prediction in predictions:
            self.assertGreaterEqual(prediction.success_probability, 0.0)
            self.assertLessEqual(prediction.success_probability, 1.0)


if __name__ == "__main__":
    unittest.main()
