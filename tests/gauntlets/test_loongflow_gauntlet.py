"""
Comprehensive Tests for LoongFlow Gauntlet Adapter

Tests the LoongFlow PES integration as a Round 1 evaluator in the
OpenEvolve gauntlet system.
"""

import pytest
import asyncio
from datetime import datetime, UTC
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from openevolve.gauntlets import (
    LoongFlowGauntletEvaluator,
    LoongFlowGauntletConfig,
    GauntletEvaluationResult,
)


class TestLoongFlowGauntletConfig:
    """Test configuration validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LoongFlowGauntletConfig()

        assert config.enable_planning is True
        assert config.enable_memory is True
        assert config.early_stopping is True
        assert config.plan_temperature == 0.7
        assert config.summary_temperature == 0.7
        assert config.evaluation_timeout == 30
        assert config.max_evaluations == 50
        assert config.quality_threshold == 0.5
        assert config.confidence_threshold == 0.6
        assert config.enable_detailed_feedback is True
        assert config.correctness_weight == 0.4
        assert config.efficiency_weight == 0.3
        assert config.robustness_weight == 0.2
        assert config.creativity_weight == 0.1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LoongFlowGauntletConfig(
            quality_threshold=0.7,
            max_evaluations=100,
            evaluation_timeout=60,
            enable_detailed_feedback=False
        )

        assert config.quality_threshold == 0.7
        assert config.max_evaluations == 100
        assert config.evaluation_timeout == 60
        assert config.enable_detailed_feedback is False

    def test_weight_validation(self):
        """Test that weights must sum to 1.0."""
        # Valid weights
        config = LoongFlowGauntletConfig(
            correctness_weight=0.5,
            efficiency_weight=0.3,
            robustness_weight=0.1,
            creativity_weight=0.1
        )
        assert config.correctness_weight == 0.5

        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            LoongFlowGauntletConfig(
                correctness_weight=0.5,
                efficiency_weight=0.5,
                robustness_weight=0.5,
                creativity_weight=0.5
            )

    def test_range_validation(self):
        """Test range validation for configuration fields."""
        # Valid ranges
        config = LoongFlowGauntletConfig(
            plan_temperature=0.0,
            evaluation_timeout=300,
            quality_threshold=1.0
        )
        assert config.plan_temperature == 0.0
        assert config.evaluation_timeout == 300
        assert config.quality_threshold == 1.0

        # Invalid ranges
        with pytest.raises(ValueError):
            LoongFlowGauntletConfig(plan_temperature=3.0)

        with pytest.raises(ValueError):
            LoongFlowGauntletConfig(evaluation_timeout=400)

        with pytest.raises(ValueError):
            LoongFlowGauntletConfig(quality_threshold=1.5)


class TestGauntletEvaluationResult:
    """Test evaluation result data structure."""

    def test_result_creation(self):
        """Test creating an evaluation result."""
        result = GauntletEvaluationResult(
            solution="def foo(): return 42",
            passed=True,
            overall_score=0.85,
            confidence=0.90,
            correctness_score=0.9,
            efficiency_score=0.8,
            robustness_score=0.85,
            creativity_score=0.7,
            pes_iterations=10,
            pes_evaluations=45,
            convergence_quality=0.88,
            feedback="Excellent solution",
            evaluation_time=5.2,
            timestamp=datetime.now(UTC)
        )

        assert result.solution == "def foo(): return 42"
        assert result.passed is True
        assert result.overall_score == 0.85
        assert result.confidence == 0.90
        assert result.correctness_score == 0.9
        assert result.efficiency_score == 0.8
        assert result.robustness_score == 0.85
        assert result.creativity_score == 0.7
        assert result.pes_iterations == 10
        assert result.pes_evaluations == 45
        assert result.convergence_quality == 0.88
        assert result.feedback == "Excellent solution"
        assert result.evaluation_time == 5.2

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        timestamp = datetime.now(UTC)
        result = GauntletEvaluationResult(
            solution="test solution",
            passed=True,
            overall_score=0.75,
            confidence=0.80,
            correctness_score=0.8,
            efficiency_score=0.7,
            robustness_score=0.75,
            creativity_score=0.6,
            pes_iterations=5,
            pes_evaluations=25,
            convergence_quality=0.7,
            feedback="Good solution",
            strengths=["Clear code"],
            weaknesses=["Slow"],
            suggestions=["Optimize"],
            evaluation_time=3.5,
            timestamp=timestamp
        )

        data = result.to_dict()

        assert data["solution"] == "test solution"
        assert data["passed"] is True
        assert data["overall_score"] == 0.75
        assert data["confidence"] == 0.80
        assert data["correctness_score"] == 0.8
        assert data["efficiency_score"] == 0.7
        assert data["robustness_score"] == 0.75
        assert data["creativity_score"] == 0.6
        assert data["pes_iterations"] == 5
        assert data["pes_evaluations"] == 25
        assert data["convergence_quality"] == 0.7
        assert data["feedback"] == "Good solution"
        assert data["strengths"] == ["Clear code"]
        assert data["weaknesses"] == ["Slow"]
        assert data["suggestions"] == ["Optimize"]
        assert data["evaluation_time"] == 3.5
        assert data["timestamp"] == timestamp.isoformat()

    def test_result_from_dict(self):
        """Test creating result from dictionary."""
        timestamp = datetime.now(UTC)
        data = {
            "solution": "test solution",
            "passed": True,
            "overall_score": 0.75,
            "confidence": 0.80,
            "correctness_score": 0.8,
            "efficiency_score": 0.7,
            "robustness_score": 0.75,
            "creativity_score": 0.6,
            "pes_iterations": 5,
            "pes_evaluations": 25,
            "convergence_quality": 0.7,
            "feedback": "Good solution",
            "strengths": ["Clear code"],
            "weaknesses": ["Slow"],
            "suggestions": ["Optimize"],
            "evaluation_time": 3.5,
            "timestamp": timestamp.isoformat(),
            "artifacts": {}
        }

        result = GauntletEvaluationResult.from_dict(data)

        assert result.solution == "test solution"
        assert result.passed is True
        assert result.overall_score == 0.75
        assert result.timestamp == timestamp

    def test_default_lists(self):
        """Test that feedback lists default to empty lists."""
        result = GauntletEvaluationResult(
            solution="test",
            passed=False,
            overall_score=0.5,
            confidence=0.5,
            correctness_score=0.5,
            efficiency_score=0.5,
            robustness_score=0.5,
            creativity_score=0.5,
            pes_iterations=0,
            pes_evaluations=0,
            convergence_quality=0.0,
            feedback="test"
        )

        assert result.strengths == []
        assert result.weaknesses == []
        assert result.suggestions == []
        assert result.artifacts == {}


class TestLoongFlowGauntletEvaluator:
    """Test the main LoongFlow gauntlet evaluator."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return LoongFlowGauntletConfig(
            quality_threshold=0.6,
            confidence_threshold=0.7,
            max_evaluations=30,
            enable_detailed_feedback=True
        )

    @pytest.fixture
    def evaluator(self, config):
        """Create an evaluator instance."""
        return LoongFlowGauntletEvaluator(config)

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.config is not None
        assert evaluator.loongflow_adapter is not None
        assert isinstance(evaluator.config, LoongFlowGauntletConfig)

    def test_get_config(self, evaluator, config):
        """Test getting configuration."""
        retrieved_config = evaluator.get_config()
        assert retrieved_config == config

    @pytest.mark.asyncio
    async def test_evaluate_solution_success(self, evaluator):
        """Test successful solution evaluation with proper mocking."""
        # Mock the LoongFlow adapter at the instance level
        original_evolve = evaluator.loongflow_adapter.evolve

        async def mock_evolve(*args, **kwargs):
            return {
                "best_solution": "def foo(): return 42",
                "best_fitness": 0.85,  # High fitness to pass threshold
                "total_evaluations": 25,
                "improvement_rate": 0.75,
                "iterations_performed": 8,
                "convergence_quality": 0.8,
                "strategy_used": "pes"
            }

        evaluator.loongflow_adapter.evolve = mock_evolve

        try:
            result = await evaluator.evaluate_solution(
                solution="def foo(): return 42",
                problem="Create a function that returns 42",
                domain="code"
            )

            # With best_fitness=0.85, should pass 0.6 threshold
            assert result.passed is True
            assert result.overall_score >= 0.6
            assert result.confidence >= 0.7
            assert result.pes_evaluations == 25
            assert result.pes_iterations == 8
            assert len(result.feedback) > 0
        finally:
            evaluator.loongflow_adapter.evolve = original_evolve

    @pytest.mark.asyncio
    async def test_evaluate_solution_failure(self, evaluator):
        """Test solution evaluation that fails thresholds."""
        # Mock low-quality result
        with patch.object(
            evaluator.loongflow_adapter,
            'evolve',
            new_callable=AsyncMock
        ) as mock_evolve:
            mock_evolve.return_value = {
                "best_solution": "def foo(): return 0",
                "best_fitness": 0.3,  # Below threshold
                "total_evaluations": 50,
                "improvement_rate": 0.1,
                "iterations_performed": 10,
                "convergence_quality": 0.2,
                "strategy_used": "pes"
            }

            result = await evaluator.evaluate_solution(
                solution="def foo(): return 0",
                problem="Create a function that returns 42",
                domain="code"
            )

            assert result.passed is False
            assert result.overall_score < 0.6
            assert "FAIL" in result.feedback

    @pytest.mark.asyncio
    async def test_evaluate_solution_error_handling(self, evaluator):
        """Test error handling during evaluation."""
        # Mock exception
        with patch.object(
            evaluator.loongflow_adapter,
            'evolve',
            new_callable=AsyncMock
        ) as mock_evolve:
            mock_evolve.side_effect = Exception("LoongFlow error")

            result = await evaluator.evaluate_solution(
                solution="def foo(): return 42",
                problem="Test problem",
                domain="code"
            )

            assert result.passed is False
            assert result.overall_score == 0.0
            assert result.confidence == 0.0
            assert "error" in result.feedback.lower()

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, evaluator):
        """Test batch evaluation of multiple solutions."""
        solutions = [
            "def foo(): return 1",
            "def foo(): return 2",
            "def foo(): return 3"
        ]

        # Mock the adapter
        with patch.object(
            evaluator.loongflow_adapter,
            'evolve',
            new_callable=AsyncMock
        ) as mock_evolve:
            mock_evolve.return_value = {
                "best_solution": "test",
                "best_fitness": 0.8,
                "total_evaluations": 20,
                "improvement_rate": 0.7,
                "iterations_performed": 7,
                "convergence_quality": 0.75,
                "strategy_used": "pes"
            }

            results = await evaluator.evaluate_batch(
                solutions=solutions,
                problem="Test problem",
                domain="code"
            )

            assert len(results) == 3
            assert all(isinstance(r, GauntletEvaluationResult) for r in results)
            assert mock_evolve.call_count == 3

    @pytest.mark.asyncio
    async def test_evaluate_batch_with_exception(self, evaluator):
        """Test batch evaluation with some failures."""
        solutions = ["sol1", "sol2", "sol3"]

        # Mock the evolve method to return different values for each call
        original_evolve = evaluator.loongflow_adapter.evolve
        call_count = [0]

        async def mock_evolve_with_exception(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"best_solution": "sol1", "best_fitness": 0.8, "total_evaluations": 20,
                        "improvement_rate": 0.7, "iterations_performed": 7, "convergence_quality": 0.75}
            elif call_count[0] == 2:
                raise Exception("Error")
            else:
                return {"best_solution": "sol3", "best_fitness": 0.9, "total_evaluations": 15,
                        "improvement_rate": 0.8, "iterations_performed": 5, "convergence_quality": 0.85}

        evaluator.loongflow_adapter.evolve = mock_evolve_with_exception

        try:
            results = await evaluator.evaluate_batch(
                solutions=solutions,
                problem="Test",
                domain="code"
            )

            assert len(results) == 3
            assert results[0].passed is True  # Success
            assert results[1].passed is False  # Exception
            assert results[2].passed is True  # Success
            assert "error" in results[1].feedback.lower()
        finally:
            evaluator.loongflow_adapter.evolve = original_evolve

    def test_is_available(self, evaluator):
        """Test checking if LoongFlow is available."""
        # This will depend on whether LoongFlow is actually installed
        available = evaluator.is_available()
        assert isinstance(available, bool)

    def test_calculate_overall_score(self, evaluator):
        """Test overall score calculation from dimensions."""
        scores = {
            "correctness": 0.8,
            "efficiency": 0.7,
            "robustness": 0.75,
            "creativity": 0.6
        }

        overall = evaluator._calculate_overall_score(scores)

        expected = (
            0.8 * 0.4 +  # correctness
            0.7 * 0.3 +  # efficiency
            0.75 * 0.2 +  # robustness
            0.6 * 0.1     # creativity
        )

        assert abs(overall - expected) < 0.001

    def test_check_thresholds_pass(self, evaluator):
        """Test threshold checking when solution passes."""
        assert evaluator._check_thresholds(0.8, 0.9) is True

    def test_check_thresholds_fail_score(self, evaluator):
        """Test threshold checking when score fails."""
        assert evaluator._check_thresholds(0.5, 0.9) is False

    def test_check_thresholds_fail_confidence(self, evaluator):
        """Test threshold checking when confidence fails."""
        assert evaluator._check_thresholds(0.8, 0.5) is False

    def test_check_thresholds_fail_both(self, evaluator):
        """Test threshold checking when both fail."""
        assert evaluator._check_thresholds(0.5, 0.5) is False

    @pytest.mark.asyncio
    async def test_calculate_confidence_with_loongflow(self, evaluator):
        """Test confidence calculation with LoongFlow available."""
        pes_result = {
            "iterations_performed": 15,
            "convergence_quality": 0.8
        }

        confidence = evaluator._calculate_confidence(pes_result, 0.85)

        # Should be reasonably high
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5

    @pytest.mark.asyncio
    async def test_calculate_confidence_fallback(self, evaluator):
        """Test confidence calculation without LoongFlow."""
        with patch.object(evaluator.loongflow_adapter, 'is_available', return_value=False):
            pes_result = {}

            confidence = evaluator._calculate_confidence(pes_result, 0.5)

            # Should be low in fallback mode
            assert confidence == 0.3

    @pytest.mark.asyncio
    async def test_assess_creativity(self, evaluator):
        """Test creativity assessment."""
        # Simple solution
        solution1 = "def foo(): return 42"
        creativity1 = await evaluator._assess_creativity(solution1, "test")
        assert 0.0 <= creativity1 <= 1.0

        # Complex solution with patterns
        solution2 = """
def generator():
    '''A generator function'''
    for i in range(10):
        yield i * 2

@decorator
def decorated():
    pass
"""
        creativity2 = await evaluator._assess_creativity(solution2, "test")
        assert creativity2 > creativity1  # Should be higher

    @pytest.mark.asyncio
    async def test_generate_feedback_passing(self, evaluator):
        """Test feedback generation for passing solution."""
        scores = {
            "correctness": 0.9,
            "efficiency": 0.85,
            "robustness": 0.8,
            "creativity": 0.7
        }

        feedback = await evaluator._generate_feedback(
            solution="def foo(): return 42",
            problem="Test problem",
            scores=scores,
            overall_score=0.84,
            passed=True
        )

        assert "feedback" in feedback
        assert "strengths" in feedback
        assert "weaknesses" in feedback
        assert "suggestions" in feedback
        assert len(feedback["strengths"]) > 0
        assert "PASS" in feedback["feedback"]

    @pytest.mark.asyncio
    async def test_generate_feedback_failing(self, evaluator):
        """Test feedback generation for failing solution."""
        scores = {
            "correctness": 0.3,
            "efficiency": 0.4,
            "robustness": 0.2,
            "creativity": 0.3
        }

        feedback = await evaluator._generate_feedback(
            solution="def foo(): return 0",
            problem="Test problem",
            scores=scores,
            overall_score=0.3,
            passed=False
        )

        assert "feedback" in feedback
        assert len(feedback["weaknesses"]) > 0
        assert len(feedback["suggestions"]) > 0
        assert "FAIL" in feedback["feedback"]


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    @pytest.fixture
    def config(self):
        """Create production-like configuration."""
        return LoongFlowGauntletConfig(
            quality_threshold=0.6,
            confidence_threshold=0.7,
            max_evaluations=50,
            evaluation_timeout=30,
            enable_detailed_feedback=True
        )

    @pytest.fixture
    def evaluator(self, config):
        """Create evaluator for integration tests."""
        return LoongFlowGauntletEvaluator(config)

    @pytest.mark.asyncio
    async def test_math_problem_evaluation(self, evaluator):
        """Test evaluating a math problem solution."""
        # Mock successful evolution
        with patch.object(
            evaluator.loongflow_adapter,
            'evolve',
            new_callable=AsyncMock
        ) as mock_evolve:
            mock_evolve.return_value = {
                "best_solution": "def solve(x): return x**2",
                "best_fitness": 0.92,
                "total_evaluations": 35,
                "improvement_rate": 0.85,
                "iterations_performed": 9,
                "convergence_quality": 0.9,
                "strategy_used": "pes"
            }

            result = await evaluator.evaluate_solution(
                solution="def solve(x): return x**2",
                problem="Square the input number",
                domain="math"
            )

            assert result.passed is True
            assert result.correctness_score > 0.8
            assert "math" in result.artifacts["domain"]

    @pytest.mark.asyncio
    async def test_code_problem_evaluation(self, evaluator):
        """Test evaluating a code problem solution."""
        # Mock the evolve method
        original_evolve = evaluator.loongflow_adapter.evolve

        async def mock_evolve(*args, **kwargs):
            return {
                "best_solution": "def sort(arr): return sorted(arr)",
                "best_fitness": 0.88,
                "total_evaluations": 28,
                "improvement_rate": 0.78,
                "iterations_performed": 8,
                "convergence_quality": 0.85,
                "strategy_used": "pes"
            }

        evaluator.loongflow_adapter.evolve = mock_evolve

        try:
            result = await evaluator.evaluate_solution(
                solution="def sort(arr): return sorted(arr)",
                problem="Sort an array of numbers",
                domain="code"
            )

            assert result.passed is True
            assert result.efficiency_score > 0.5
        finally:
            evaluator.loongflow_adapter.evolve = original_evolve

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, evaluator):
        """Test that evaluation meets performance targets."""
        import time

        with patch.object(
            evaluator.loongflow_adapter,
            'evolve',
            new_callable=AsyncMock
        ) as mock_evolve:
            # Simulate quick evaluation
            async def quick_evolve(*args, **kwargs):
                await asyncio.sleep(0.1)  # 100ms
                return {
                    "best_solution": "test",
                    "best_fitness": 0.8,
                    "total_evaluations": 20,
                    "improvement_rate": 0.7,
                    "iterations_performed": 5,
                    "convergence_quality": 0.75
                }

            mock_evolve.side_effect = quick_evolve

            start = time.time()
            result = await evaluator.evaluate_solution(
                solution="test",
                problem="test",
                domain="code"
            )
            elapsed = time.time() - start

            # Should complete quickly
            assert elapsed < 5.0  # Well under 30 second target
            assert result.evaluation_time < 5.0

    @pytest.mark.asyncio
    async def test_batch_performance(self, evaluator):
        """Test batch evaluation performance."""
        # Mock the evolve method
        original_evolve = evaluator.loongflow_adapter.evolve

        async def quick_evolve(*args, **kwargs):
            await asyncio.sleep(0.05)
            return {
                "best_solution": "test",
                "best_fitness": 0.75,
                "total_evaluations": 15,
                "improvement_rate": 0.65,
                "iterations_performed": 4,
                "convergence_quality": 0.7
            }

        evaluator.loongflow_adapter.evolve = quick_evolve

        try:
            solutions = [f"def solution_{i}(): return {i}" for i in range(10)]

            start = time.time()
            results = await evaluator.evaluate_batch(
                solutions=solutions,
                problem="Test batch",
                domain="code"
            )
            elapsed = time.time() - start

            # Batch should complete in reasonable time
            # With 10 solutions at 50ms each, should be < 5s
            assert elapsed < 5.0
            assert len(results) == 10
        finally:
            evaluator.loongflow_adapter.evolve = original_evolve


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
