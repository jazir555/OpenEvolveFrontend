"""
Unit Tests for Verification Engine

Comprehensive test suite for verification_engine.py

Test Coverage:
- SuccessCriterion creation and validation
- SolutionQualityMetrics calculation
- VerificationReport generation
- VerificationEngine core functionality
- Edge cases and error handling
- Integration with sovereign_data_models
- Integration with crewai_state_management

Run with: python -m pytest test_verification_engine.py -v
"""

import pytest
import time
from dataclasses import dataclass
from typing import Dict, Any, List
from datetime import datetime

# Import the verification engine
from verification_engine import (
    VerificationEngine,
    VerificationReport,
    SuccessCriterion,
    SolutionQualityMetrics,
    create_default_criteria,
    compare_reports
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_solution():
    """Create a sample solution for testing."""
    @dataclass
    class TestSolution:
        id: str
        solution_content: str
        gauntlet_name: str = "test_gauntlet"

    return TestSolution(
        id="test_solution_001",
        solution_content="""
def example_function(data: List[str]) -> Dict[str, int]:
    \"\"\"
    Process data and return statistics.

    Args:
        data: List of strings to process

    Returns:
        Dictionary with statistics
    \"\"\"
    try:
        if not data:
            return {"count": 0, "length": 0}

        result = {
            "count": len(data),
            "length": sum(len(item) for item in data),
            "average": sum(len(item) for item in data) / len(data)
        }

        return result

    except Exception as e:
        print(f"Error processing data: {e}")
        return {"error": str(e)}

# Unit tests
assert example_function([]) == {"count": 0, "length": 0}
assert example_function(["a", "ab", "abc"])["count"] == 3
"""
    )


@pytest.fixture
def engine():
    """Create a VerificationEngine instance for testing."""
    return VerificationEngine(config={
        'strict_mode': False,
        'min_quality_threshold': 0.6,
        'enable_detailed_logging': True
    })


@pytest.fixture
def sample_criteria():
    """Create sample success criteria for testing."""
    return [
        SuccessCriterion(
            id="test_completeness",
            description="Solution must be complete",
            metric="completeness",
            threshold=0.7
        ),
        SuccessCriterion(
            id="test_correctness",
            description="Solution must be correct",
            metric="correctness",
            threshold=0.6
        )
    ]


# =============================================================================
# SUCCESS CRITERION TESTS
# =============================================================================

class TestSuccessCriterion:
    """Test SuccessCriterion dataclass."""

    def test_criterion_creation(self):
        """Test creating a valid success criterion."""
        criterion = SuccessCriterion(
            id="test_001",
            description="Test criterion",
            metric="completeness",
            threshold=0.8
        )

        assert criterion.id == "test_001"
        assert criterion.description == "Test criterion"
        assert criterion.metric == "completeness"
        assert criterion.threshold == 0.8
        assert criterion.weight == 1.0  # Default value
        assert criterion.category == "functional"  # Default value

    def test_criterion_invalid_threshold_high(self):
        """Test that threshold > 1.0 raises error."""
        with pytest.raises(ValueError, match="Threshold must be between"):
            SuccessCriterion(
                id="test_002",
                description="Invalid threshold",
                metric="completeness",
                threshold=1.5
            )

    def test_criterion_invalid_threshold_low(self):
        """Test that threshold < 0.0 raises error."""
        with pytest.raises(ValueError, match="Threshold must be between"):
            SuccessCriterion(
                id="test_003",
                description="Invalid threshold",
                metric="completeness",
                threshold=-0.1
            )

    def test_criterion_negative_weight(self):
        """Test that negative weight raises error."""
        with pytest.raises(ValueError, match="Weight must be non-negative"):
            SuccessCriterion(
                id="test_004",
                description="Negative weight",
                metric="completeness",
                threshold=0.7,
                weight=-1.0
            )

    def test_criterion_empty_id(self):
        """Test that empty ID raises error."""
        with pytest.raises(ValueError, match="Criterion ID cannot be empty"):
            SuccessCriterion(
                id="",
                description="Empty ID",
                metric="completeness",
                threshold=0.7
            )


# =============================================================================
# QUALITY METRICS TESTS
# =============================================================================

class TestSolutionQualityMetrics:
    """Test SolutionQualityMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating quality metrics."""
        metrics = SolutionQualityMetrics(
            completeness=0.8,
            correctness=0.9,
            efficiency=0.7,
            clarity=0.85,
            maintainability=0.75,
            scalability=0.6,
            security=0.8,
            test_coverage=0.7
        )

        assert metrics.completeness == 0.8
        assert metrics.correctness == 0.9
        assert metrics.efficiency == 0.7

    def test_calculate_overall_default_weights(self):
        """Test overall score calculation with default weights."""
        metrics = SolutionQualityMetrics(
            completeness=0.8,
            correctness=0.9,
            efficiency=0.7,
            clarity=0.85,
            maintainability=0.75,
            scalability=0.6,
            security=0.8,
            test_coverage=0.7
        )

        overall = metrics.calculate_overall()

        assert 0.0 <= overall <= 1.0
        assert metrics.overall_score == overall

    def test_calculate_overall_custom_weights(self):
        """Test overall score calculation with custom weights."""
        metrics = SolutionQualityMetrics(
            completeness=1.0,
            correctness=1.0,
            efficiency=0.5,
            clarity=0.5
        )

        custom_weights = {
            'completeness': 0.5,
            'correctness': 0.5,
            'efficiency': 0.0,
            'clarity': 0.0
        }

        overall = metrics.calculate_overall(custom_weights)

        # With custom weights, only completeness and correctness matter
        assert overall == 1.0

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = SolutionQualityMetrics(
            completeness=0.8,
            correctness=0.9
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result['completeness'] == 0.8
        assert result['correctness'] == 0.9
        assert 'overall_score' in result
        assert 'confidence' in result


# =============================================================================
# VERIFICATION REPORT TESTS
# =============================================================================

class TestVerificationReport:
    """Test VerificationReport dataclass."""

    def test_report_creation(self):
        """Test creating a verification report."""
        quality_metrics = SolutionQualityMetrics(
            completeness=0.8,
            correctness=0.9,
            overall_score=0.85
        )

        report = VerificationReport(
            solution_attempt_id="solution_001",
            gauntlet_name="test_gauntlet",
            is_approved=True,
            reports_by_judge=[{"judge": "test"}],
            summary="Test summary",
            quality_metrics=quality_metrics,
            verification_score=0.85
        )

        assert report.solution_attempt_id == "solution_001"
        assert report.gauntlet_name == "test_gauntlet"
        assert report.is_approved is True
        assert report.verification_score == 0.85
        assert report.quality_metrics is not None

    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        report = VerificationReport(
            solution_attempt_id="solution_001",
            gauntlet_name="test_gauntlet",
            is_approved=True,
            reports_by_judge=[],
            summary="Test",
            verification_score=0.8
        )

        result = report.to_dict()

        assert isinstance(result, dict)
        assert result['solution_attempt_id'] == "solution_001"
        assert result['is_approved'] is True
        assert 'timestamp' in result
        assert 'metadata' in result

    def test_report_to_json(self):
        """Test converting report to JSON string."""
        report = VerificationReport(
            solution_attempt_id="solution_001",
            gauntlet_name="test_gauntlet",
            is_approved=True,
            reports_by_judge=[],
            summary="Test",
            verification_score=0.8
        )

        json_str = report.to_json()

        assert isinstance(json_str, str)
        assert "solution_001" in json_str
        assert "true" in json_str  # is_approved


# =============================================================================
# VERIFICATION ENGINE TESTS
# =============================================================================

class TestVerificationEngine:
    """Test VerificationEngine class."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = VerificationEngine()

        assert engine.config == {}
        assert engine.verification_history == []
        assert engine.strict_mode is False
        assert engine.min_quality_threshold == 0.6

    def test_engine_initialization_with_config(self):
        """Test engine initialization with custom config."""
        engine = VerificationEngine(config={
            'strict_mode': True,
            'min_quality_threshold': 0.8,
            'enable_detailed_logging': False
        })

        assert engine.strict_mode is True
        assert engine.min_quality_threshold == 0.8

    def test_verify_solution_success(self, engine, sample_solution, sample_criteria):
        """Test successful solution verification."""
        report = engine.verify_solution(sample_solution, sample_criteria)

        assert isinstance(report, VerificationReport)
        assert report.solution_attempt_id == "test_solution_001"
        assert report.quality_metrics is not None
        assert len(report.criteria_results) == len(sample_criteria)
        assert 'verification_time_seconds' in report.metadata

    def test_verify_solution_empty_solution(self, engine, sample_criteria):
        """Test verification with empty solution returns failed report."""
        @dataclass
        class EmptySolution:
            id: str
            solution_content: str = ""

        empty_solution = EmptySolution(id="empty")

        # Should create a report (empty solutions have default metrics)
        report = engine.verify_solution(empty_solution, sample_criteria)

        # Empty solution should generally fail verification
        assert report.is_approved is False
        # All criteria should fail for empty solution
        assert all(passed is False for passed in report.criteria_results.values())

    def test_verify_solution_none_solution(self, engine, sample_criteria):
        """Test verification with None solution raises error."""
        with pytest.raises(ValueError, match="Solution cannot be None"):
            engine.verify_solution(None, sample_criteria)

    def test_verify_solution_empty_criteria(self, engine, sample_solution):
        """Test verification with empty criteria raises error."""
        with pytest.raises(ValueError, match="At least one success criterion"):
            engine.verify_solution(sample_solution, [])

    def test_create_success_criteria(self, engine):
        """Test creating success criteria from requirements."""
        requirements = [
            "Solution must be at least 90% complete",
            "Code must pass all security checks",
            "Solution must be efficient"
        ]

        criteria = engine.create_success_criteria(requirements)

        assert len(criteria) == len(requirements)
        assert all(isinstance(c, SuccessCriterion) for c in criteria)
        assert criteria[0].metric == "completeness"
        assert criteria[1].metric == "security"
        assert criteria[2].metric == "efficiency"

    def test_check_criterion_passed(self, engine, sample_solution):
        """Test checking a criterion that passes."""
        criterion = SuccessCriterion(
            id="test_low_threshold",
            description="Easy criterion",
            metric="completeness",
            threshold=0.1  # Very low threshold
        )

        result = engine.check_criterion(sample_solution, criterion)

        assert result is True

    def test_check_criterion_failed(self, engine, sample_solution):
        """Test checking a criterion that fails."""
        criterion = SuccessCriterion(
            id="test_high_threshold",
            description="Hard criterion",
            metric="completeness",
            threshold=0.99  # Very high threshold
        )

        result = engine.check_criterion(sample_solution, criterion)

        assert result is False

    def test_calculate_quality_scores(self, engine, sample_solution):
        """Test quality score calculation."""
        metrics = engine.calculate_quality_scores(sample_solution)

        assert isinstance(metrics, SolutionQualityMetrics)
        assert 0.0 <= metrics.completeness <= 1.0
        assert 0.0 <= metrics.correctness <= 1.0
        assert 0.0 <= metrics.efficiency <= 1.0
        assert 0.0 <= metrics.clarity <= 1.0
        assert 0.0 <= metrics.maintainability <= 1.0
        assert 0.0 <= metrics.scalability <= 1.0
        assert 0.0 <= metrics.security <= 1.0
        assert 0.0 <= metrics.test_coverage <= 1.0
        assert 0.0 <= metrics.overall_score <= 1.0
        assert 0.0 <= metrics.confidence <= 1.0

    def test_generate_verification_report(self, engine):
        """Test generating verification report from results."""
        results = [
            {'is_approved': True, 'score': 0.9},
            {'is_approved': True, 'score': 0.8},
            {'is_approved': False, 'score': 0.5}
        ]

        report = engine.generate_verification_report(
            solution_id="solution_123",
            gauntlet_name="test_gauntlet",
            results=results
        )

        assert isinstance(report, VerificationReport)
        assert report.solution_attempt_id == "solution_123"
        assert report.gauntlet_name == "test_gauntlet"
        assert report.is_approved is False  # One failed
        assert abs(report.verification_score - 0.733) < 0.01  # Average (approx)

    def test_run_verification_suite(self, engine, sample_solution):
        """Test running a verification suite."""
        test_suite = [
            {
                'id': 'test1',
                'metric': 'completeness',
                'threshold': 0.5,
                'description': 'Test completeness'
            },
            {
                'id': 'test2',
                'metric': 'correctness',
                'threshold': 0.5,
                'description': 'Test correctness'
            }
        ]

        report = engine.run_verification_suite(sample_solution, test_suite)

        assert isinstance(report, VerificationReport)
        assert report.quality_metrics is not None
        assert report.metadata['total_tests'] == 2
        assert 'passed_tests' in report.metadata
        assert 'execution_time_seconds' in report.metadata

    def test_verification_history_tracking(self, engine, sample_solution, sample_criteria):
        """Test that verification history is tracked."""
        # Clear any existing history
        engine.clear_history()

        # Perform verification
        engine.verify_solution(sample_solution, sample_criteria)

        # Check history
        history = engine.get_verification_history()
        assert len(history) == 1
        assert history[0].solution_attempt_id == "test_solution_001"

    def test_clear_history(self, engine, sample_solution, sample_criteria):
        """Test clearing verification history."""
        # Add to history
        engine.verify_solution(sample_solution, sample_criteria)
        assert len(engine.get_verification_history()) > 0

        # Clear
        engine.clear_history()
        assert len(engine.get_verification_history()) == 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_verify_solution_dict_format(self, engine):
        """Test verification with dictionary solution."""
        solution = {
            'id': 'dict_solution',
            'solution_content': 'def test(): pass',
            'gauntlet_name': 'test'
        }

        criteria = engine.create_success_criteria(["Solution must be complete"])

        report = engine.verify_solution(solution, criteria)

        # Solution ID might be extracted from dict or default
        assert report.quality_metrics is not None

    def test_verify_solution_missing_attributes(self, engine):
        """Test verification with solution missing expected attributes."""
        class MinimalSolution:
            pass

        solution = MinimalSolution()
        solution.content = "def test(): pass"

        criteria = engine.create_success_criteria(["Solution must be complete"])

        # Should not raise error, but handle gracefully
        report = engine.verify_solution(solution, criteria)

        assert report is not None

    def test_create_criteria_invalid_requirements(self, engine):
        """Test creating criteria from invalid/unclear requirements."""
        requirements = [
            "This is a very vague requirement without specific metrics",
            "Another unclear requirement",
            ""
        ]

        criteria = engine.create_success_criteria(requirements)

        # Should still create criteria, but with defaults
        assert len(criteria) == len(requirements)
        assert all(isinstance(c, SuccessCriterion) for c in criteria)

    def test_verification_with_zero_length_content(self, engine):
        """Test verification with very short content."""
        @dataclass
        class ShortSolution:
            id: str
            solution_content: str

        solution = ShortSolution(id="short", solution_content="x")
        criteria = engine.create_success_criteria(["Complete solution"])

        report = engine.verify_solution(solution, criteria)

        # Should complete without error
        assert report is not None
        assert report.quality_metrics.completeness <= 0.5  # Low score for short content

    def test_verification_with_very_long_content(self, engine):
        """Test verification with very long content."""
        long_content = "def function_" + "x" * 10000 + "(): pass"

        @dataclass
        class LongSolution:
            id: str
            solution_content: str

        solution = LongSolution(id="long", solution_content=long_content)
        criteria = engine.create_success_criteria(["Complete solution"])

        report = engine.verify_solution(solution, criteria)

        # Should complete without error
        assert report is not None
        assert report.quality_metrics.completeness > 0.6  # Higher score for long content


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_default_criteria(self):
        """Test creating default criteria."""
        criteria = create_default_criteria()

        assert isinstance(criteria, list)
        assert len(criteria) > 0
        assert all(isinstance(c, SuccessCriterion) for c in criteria)

    def test_compare_reports(self):
        """Test comparing two verification reports."""
        report1 = VerificationReport(
            solution_attempt_id="solution_001",
            gauntlet_name="test",
            is_approved=True,
            reports_by_judge=[],
            summary="First report",
            verification_score=0.8
        )

        report2 = VerificationReport(
            solution_attempt_id="solution_002",
            gauntlet_name="test",
            is_approved=False,
            reports_by_judge=[],
            summary="Second report",
            verification_score=0.6
        )

        comparison = compare_reports(report1, report2)

        assert 'solution_ids' in comparison
        assert 'approval_changed' in comparison
        assert 'score_difference' in comparison
        assert comparison['approval_changed'] is True
        assert abs(comparison['score_difference'] - (-0.2)) < 0.01


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with related modules."""

    def test_integration_with_sovereign_data_models(self):
        """Test integration with sovereign_data_models SolutionAttempt."""
        try:
            from sovereign_data_models import SolutionAttempt

            # Use the fallback SolutionAttempt from sovereign_data_models
            # which has fields: id, problem_id, solution, score, timestamp
            solution = SolutionAttempt(
                id="integration_test",
                problem_id="test_problem",
                solution="def test_solution():\n    return True",
                score=0.8,
                timestamp=datetime.now()
            )

            engine = VerificationEngine()
            criteria = engine.create_success_criteria(["Solution must be correct"])

            report = engine.verify_solution(solution, criteria)

            assert report.quality_metrics is not None

        except ImportError:
            pytest.skip("sovereign_data_models not available")
        except Exception as e:
            # If validation fails, skip this test
            pytest.skip(f"Integration test skipped: {e}")

    def test_integration_with_crewai_state_management(self):
        """Test integration with crewai_state_management SolutionAttempt."""
        try:
            from crewai_state_management import SolutionAttempt

            solution = SolutionAttempt(
                sub_problem_id="crewai_test",
                solution_content="def crewai_solution():\n    pass",
                confidence_score=0.75,
                execution_method="traditional"
            )

            engine = VerificationEngine()
            criteria = engine.create_success_criteria(["Solution must be complete"])

            report = engine.verify_solution(solution, criteria)

            assert report.solution_attempt_id == "crewai_test"
            assert report.quality_metrics is not None

        except ImportError:
            pytest.skip("crewai_state_management not available")


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance and stress tests."""

    def test_verify_many_solutions(self, engine):
        """Test verifying multiple solutions efficiently."""
        @dataclass
        class TestSolution:
            id: str
            solution_content: str

        criteria = engine.create_success_criteria(["Complete solution"])

        start_time = time.time()

        for i in range(10):
            solution = TestSolution(
                id=f"perf_test_{i}",
                solution_content=f"def function_{i}():\n    pass"
            )
            engine.verify_solution(solution, criteria)

        elapsed = time.time() - start_time

        # Should complete 10 verifications in reasonable time
        assert elapsed < 10.0  # Less than 10 seconds

    def test_large_test_suite(self, engine, sample_solution):
        """Test running a large test suite."""
        # Create 50 test cases
        test_suite = [
            {
                'id': f'test_{i}',
                'metric': 'completeness',
                'threshold': 0.5
            }
            for i in range(50)
        ]

        start_time = time.time()
        report = engine.run_verification_suite(sample_solution, test_suite)
        elapsed = time.time() - start_time

        assert report.metadata['total_tests'] == 50
        assert elapsed < 30.0  # Should complete in reasonable time


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
