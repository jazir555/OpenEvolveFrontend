"""
Tests for the parallel evaluation framework.

These tests verify the evaluation module's ability to:
1. Evaluate single samples correctly
2. Handle errors gracefully without failing the entire evaluation
3. Run parallel evaluations efficiently
4. Isolate errors per sample
"""

import unittest
from typing import List
from unittest.mock import MagicMock, Mock

from ace.evaluation import EvaluationResult, evaluate_dataset, evaluate_single_sample


class TestEvaluateSingleSample(unittest.TestCase):
    """Test cases for evaluate_single_sample function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent = MagicMock()
        self.mock_skillbook = MagicMock()
        self.simple_checker = lambda pred, truth: pred.strip().lower() == truth.strip().lower()

    def test_evaluate_single_sample_success(self):
        """Test successful evaluation of a single sample."""
        # Mock agent output
        mock_output = Mock()
        mock_output.final_answer = "Paris"
        mock_output.skill_ids = ["geo-00001", "general-00042"]
        self.mock_agent.generate.return_value = mock_output

        sample = {
            "question": "What is the capital of France?",
            "context": "Geography",
            "target": "Paris",
        }

        result = evaluate_single_sample(
            index=0,
            sample=sample,
            agent=self.mock_agent,
            skillbook=self.mock_skillbook,
            answer_checker=self.simple_checker,
        )

        # Verify result structure
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.prediction, "Paris")
        self.assertEqual(result.ground_truth, "Paris")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.skill_ids_used, ["geo-00001", "general-00042"])
        self.assertIsNone(result.error)

        # Verify agent was called correctly
        self.mock_agent.generate.assert_called_once()
        call_kwargs = self.mock_agent.generate.call_args[1]
        self.assertEqual(call_kwargs["question"], "What is the capital of France?")
        self.assertEqual(call_kwargs["context"], "Geography")

    def test_evaluate_single_sample_failure(self):
        """Test evaluation when prediction is incorrect."""
        # Mock agent output with wrong answer
        mock_output = Mock()
        mock_output.final_answer = "London"
        mock_output.skill_ids = []
        self.mock_agent.generate.return_value = mock_output

        sample = {
            "question": "What is the capital of France?",
            "context": None,
            "target": "Paris",
        }

        result = evaluate_single_sample(
            index=5,
            sample=sample,
            agent=self.mock_agent,
            skillbook=self.mock_skillbook,
            answer_checker=self.simple_checker,
        )

        self.assertFalse(result.is_correct)
        self.assertEqual(result.prediction, "London")
        self.assertEqual(result.ground_truth, "Paris")
        self.assertIsNone(result.error)

    def test_evaluate_single_sample_missing_question(self):
        """Test evaluation when sample is missing required question field."""
        sample = {
            "context": "Some context",
            "target": "Answer",
        }

        result = evaluate_single_sample(
            index=2,
            sample=sample,
            agent=self.mock_agent,
            skillbook=self.mock_skillbook,
            answer_checker=self.simple_checker,
        )

        self.assertFalse(result.is_correct)
        self.assertIsNotNone(result.error)
        self.assertIn("question", result.error.lower())
        self.assertEqual(result.index, 2)

    def test_evaluate_single_sample_agent_exception(self):
        """Test evaluation when agent.generate raises an exception."""
        # Simulate agent failure
        self.mock_agent.generate.side_effect = RuntimeError("LLM API timeout")

        sample = {
            "question": "Test question",
            "context": None,
            "target": "Expected answer",
        }

        result = evaluate_single_sample(
            index=10,
            sample=sample,
            agent=self.mock_agent,
            skillbook=self.mock_skillbook,
            answer_checker=self.simple_checker,
        )

        # Verify error was handled gracefully
        self.assertFalse(result.is_correct)
        self.assertIsNotNone(result.error)
        self.assertIn("RuntimeError", result.error)
        self.assertEqual(result.index, 10)

    def test_evaluate_single_sample_with_optional_fields(self):
        """Test evaluation with optional context and custom kwargs."""
        mock_output = Mock()
        mock_output.final_answer = "42"
        mock_output.skill_ids = ["math-00001"]
        self.mock_agent.generate.return_value = mock_output

        sample = {
            "question": "What is 6 * 7?",
            "ground_truth": "42",  # Use 'ground_truth' instead of 'target'
        }

        result = evaluate_single_sample(
            index=0,
            sample=sample,
            agent=self.mock_agent,
            skillbook=self.mock_skillbook,
            answer_checker=self.simple_checker,
            temperature=0.5,  # Custom kwarg
            max_tokens=100,
        )

        self.assertTrue(result.is_correct)
        self.assertEqual(result.prediction, "42")

        # Verify custom kwargs were passed through
        call_kwargs = self.mock_agent.generate.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.5)
        self.assertEqual(call_kwargs["max_tokens"], 100)


class TestEvaluateDataset(unittest.TestCase):
    """Test cases for evaluate_dataset function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent = MagicMock()
        self.mock_skillbook = MagicMock()
        self.simple_checker = lambda pred, truth: pred.strip().lower() == truth.strip().lower()

    def test_evaluate_dataset_parallel(self):
        """Test parallel evaluation of multiple samples."""
        # Mock agent responses for different samples
        def mock_generate_side_effect(**kwargs):
            output = Mock()
            if "capital of France" in kwargs.get("question", ""):
                output.final_answer = "Paris"
                output.skill_ids = ["geo-00001"]
            elif "capital of Germany" in kwargs.get("question", ""):
                output.final_answer = "Berlin"
                output.skill_ids = ["geo-00002"]
            else:
                output.final_answer = "Unknown"
                output.skill_ids = []
            return output

        self.mock_agent.generate.side_effect = mock_generate_side_effect

        samples = [
            {"question": "What is the capital of France?", "context": None, "target": "Paris"},
            {"question": "What is the capital of Germany?", "context": None, "target": "Berlin"},
            {"question": "What is the capital of Spain?", "context": None, "target": "Madrid"},
        ]

        results = evaluate_dataset(
            samples=samples,
            agent=self.mock_agent,
            skillbook=self.mock_skillbook,
            answer_checker=self.simple_checker,
            max_workers=2,
            show_progress=False,
        )

        # Verify overall results
        self.assertEqual(results["total"], 3)
        self.assertEqual(results["correct"], 2)  # Paris and Berlin correct, Madrid wrong
        self.assertAlmostEqual(results["accuracy"], 2 / 3, places=2)

        # Verify errors list contains the incorrect prediction
        self.assertEqual(len(results["errors"]), 1)
        error = results["errors"][0]
        self.assertEqual(error["prediction"], "Unknown")
        self.assertEqual(error["ground_truth"], "Madrid")

        # Verify results list
        self.assertEqual(len(results["results"]), 3)
        self.assertIsInstance(results["results"][0], EvaluationResult)

    def test_evaluate_dataset_all_correct(self):
        """Test evaluation with 100% accuracy."""
        mock_output = Mock()
        mock_output.final_answer = "correct"
        mock_output.skill_ids = ["skill-001"]
        self.mock_agent.generate.return_value = mock_output

        samples = [
            {"question": f"Question {i}", "context": None, "target": "correct"}
            for i in range(5)
        ]

        results = evaluate_dataset(
            samples=samples,
            agent=self.mock_agent,
            skillbook=self.mock_skillbook,
            answer_checker=self.simple_checker,
            max_workers=3,
            show_progress=False,
        )

        self.assertEqual(results["total"], 5)
        self.assertEqual(results["correct"], 5)
        self.assertEqual(results["accuracy"], 1.0)
        self.assertEqual(len(results["errors"]), 0)

    def test_evaluate_dataset_all_incorrect(self):
        """Test evaluation with 0% accuracy."""
        mock_output = Mock()
        mock_output.final_answer = "wrong"
        mock_output.skill_ids = []
        self.mock_agent.generate.return_value = mock_output

        samples = [
            {"question": f"Question {i}", "context": None, "target": "correct"}
            for i in range(3)
        ]

        results = evaluate_dataset(
            samples=samples,
            agent=self.mock_agent,
            skillbook=self.mock_skillbook,
            answer_checker=self.simple_checker,
            max_workers=2,
            show_progress=False,
        )

        self.assertEqual(results["total"], 3)
        self.assertEqual(results["correct"], 0)
        self.assertEqual(results["accuracy"], 0.0)
        self.assertEqual(len(results["errors"]), 3)

    def test_evaluate_dataset_empty(self):
        """Test evaluation with empty sample list."""
        results = evaluate_dataset(
            samples=[],
            agent=self.mock_agent,
            skillbook=self.mock_skillbook,
            answer_checker=self.simple_checker,
            show_progress=False,
        )

        self.assertEqual(results["total"], 0)
        self.assertEqual(results["correct"], 0)
        self.assertEqual(results["accuracy"], 0.0)
        self.assertEqual(len(results["errors"]), 0)

    def test_error_isolation(self):
        """Test that errors in one sample don't affect others."""
        call_count = 0

        def mock_generate_with_failures(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second call
                raise RuntimeError("Simulated failure")
            output = Mock()
            output.final_answer = "success"
            output.skill_ids = []
            return output

        self.mock_agent.generate.side_effect = mock_generate_with_failures

        samples = [
            {"question": f"Question {i}", "context": None, "target": "success"}
            for i in range(5)
        ]

        results = evaluate_dataset(
            samples=samples,
            agent=self.mock_agent,
            skillbook=self.mock_skillbook,
            answer_checker=self.simple_checker,
            max_workers=3,
            show_progress=False,
        )

        # All samples should be evaluated despite one failure
        self.assertEqual(results["total"], 5)
        # 4 successful (1 failure)
        self.assertEqual(results["correct"], 4)

        # Verify error was captured
        self.assertEqual(len(results["errors"]), 1)
        error = results["errors"][0]
        self.assertIn("RuntimeError", error["error"])

    def test_evaluate_dataset_custom_kwargs(self):
        """Test that custom kwargs are passed to agent.generate()."""
        mock_output = Mock()
        mock_output.final_answer = "answer"
        mock_output.skill_ids = []
        self.mock_agent.generate.return_value = mock_output

        samples = [
            {"question": "Test", "context": None, "target": "answer"}
        ]

        evaluate_dataset(
            samples=samples,
            agent=self.mock_agent,
            skillbook=self.mock_skillbook,
            answer_checker=self.simple_checker,
            temperature=0.7,
            max_tokens=200,
            show_progress=False,
        )

        # Verify custom kwargs were passed through
        call_kwargs = self.mock_agent.generate.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.7)
        self.assertEqual(call_kwargs["max_tokens"], 200)


class TestEvaluationResult(unittest.TestCase):
    """Test cases for EvaluationResult dataclass."""

    def test_evaluation_result_repr(self):
        """Test string representation of EvaluationResult."""
        result_correct = EvaluationResult(
            index=0,
            prediction="Paris",
            ground_truth="Paris",
            is_correct=True,
            skill_ids_used=["geo-00001"],
        )
        repr_str = repr(result_correct)
        self.assertIn("✓", repr_str)
        self.assertIn("index=0", repr_str)

        result_incorrect = EvaluationResult(
            index=1,
            prediction="London",
            ground_truth="Paris",
            is_correct=False,
            skill_ids_used=[],
        )
        repr_str = repr(result_incorrect)
        self.assertIn("✗", repr_str)
        self.assertIn("index=1", repr_str)

    def test_evaluation_result_defaults(self):
        """Test EvaluationResult with default values."""
        result = EvaluationResult(
            index=0,
            prediction="answer",
            ground_truth="answer",
            is_correct=True,
        )
        self.assertEqual(result.skill_ids_used, [])
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
