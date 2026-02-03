"""
Unit Tests for Success Criteria Management System

This test suite provides comprehensive coverage for the success_criteria module,
including unit tests for all major functionality and edge cases.

Run with: pytest test_success_criteria.py -v
Or: python test_success_criteria.py
"""

import pytest
import sys
from datetime import datetime, timezone
from typing import List

# Import the module to test
from success_criteria import (
    SuccessCriterion,
    CriteriaEvaluationResult,
    CriteriaReport,
    SuccessCriteriaManager,
    MetricType,
    CriterionStatus,
    merge_criteria,
    filter_criteria_by_type,
    filter_criteria_by_threshold,
    calculate_criteria_statistics,
    validate_criteria,
    generate_id
)

# Import SolutionAttempt with correct structure
try:
    from sovereign_data_models import SolutionAttempt
    from crewai_state_management import ExecutionMethod
    HAS_REAL_SOLUTION_ATTEMPT = True
except ImportError:
    HAS_REAL_SOLUTION_ATTEMPT = False
    # Create mock SolutionAttempt for testing
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class ExecutionMethod:
        MANUAL = "manual"
        AUTOMATED = "automated"

    @dataclass
    class SolutionAttempt:
        """Mock SolutionAttempt for testing."""
        sub_problem_id: str
        solution_content: str
        confidence_score: float = 0.5
        execution_method: str = ExecutionMethod.MANUAL
        agent_name: Optional[str] = None
        execution_time_seconds: float = 0.0
        token_usage: Optional[dict] = None
        voting_participants: Optional[int] = None
        red_flags: Optional[list] = None
        metadata: Optional[dict] = None
        created_at: Optional[str] = None

        @property
        def id(self):
            return self.sub_problem_id

        @property
        def solution(self):
            return self.solution_content


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_solution():
    """Create a sample solution for testing."""
    content = """
        This solution implements a secure authentication system with JWT tokens.
        It includes password hashing with bcrypt, input validation, and protection
        against SQL injection and XSS attacks. The code is well-documented and
        follows best practices for security and performance. Async operations
        are used for database queries, and caching is implemented for frequently
        accessed data. The system is designed to be scalable with a stateless
        architecture that supports horizontal scaling.
        """

    if HAS_REAL_SOLUTION_ATTEMPT:
        # Use TRADITIONAL execution method (actual value from ExecutionMethod enum)
        return SolutionAttempt(
            sub_problem_id="test_sol_001",
            solution_content=content,
            confidence_score=0.85,
            execution_method=ExecutionMethod.TRADITIONAL
        )
    else:
        return SolutionAttempt(
            sub_problem_id="test_sol_001",
            solution_content=content,
            confidence_score=0.85,
            execution_method="traditional"
        )


@pytest.fixture
def empty_solution():
    """Create an empty solution for edge case testing."""
    if HAS_REAL_SOLUTION_ATTEMPT:
        return SolutionAttempt(
            sub_problem_id="test_sol_empty",
            solution_content="",
            confidence_score=0.0,
            execution_method=ExecutionMethod.TRADITIONAL
        )
    else:
        return SolutionAttempt(
            sub_problem_id="test_sol_empty",
            solution_content="",
            confidence_score=0.0,
            execution_method="traditional"
        )


@pytest.fixture
def sample_criteria():
    """Create sample criteria for testing."""
    return [
        SuccessCriterion(
            id="crit_001",
            description="Must implement authentication with 90% accuracy",
            metric="correctness",
            threshold=0.90
        ),
        SuccessCriterion(
            id="crit_002",
            description="Should be secure and protect user data",
            metric="security",
            threshold=0.85
        ),
        SuccessCriterion(
            id="crit_003",
            description="Must be well-documented and maintainable",
            metric="maintainability",
            threshold=0.75
        ),
        SuccessCriterion(
            id="crit_004",
            description="Should perform efficiently",
            metric="performance",
            threshold=0.70
        ),
    ]


@pytest.fixture
def manager():
    """Create a SuccessCriteriaManager instance for testing."""
    return SuccessCriteriaManager()


# ============================================================================
# SuccessCriterion TESTS
# ============================================================================

class TestSuccessCriterion:
    """Test cases for SuccessCriterion dataclass."""

    def test_create_valid_criterion(self):
        """Test creating a valid criterion."""
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

    def test_criterion_validation_empty_id(self):
        """Test that empty ID raises ValueError."""
        with pytest.raises(ValueError, match="ID must be a non-empty string"):
            SuccessCriterion(
                id="",
                description="Test",
                metric="completeness",
                threshold=0.8
            )

    def test_criterion_validation_empty_description(self):
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="Description must be a non-empty string"):
            SuccessCriterion(
                id="test_001",
                description="",
                metric="completeness",
                threshold=0.8
            )

    def test_criterion_validation_invalid_threshold_type(self):
        """Test that invalid threshold type raises ValueError."""
        with pytest.raises(ValueError, match="Threshold must be a number"):
            SuccessCriterion(
                id="test_001",
                description="Test",
                metric="completeness",
                threshold="invalid"
            )

    def test_criterion_validation_threshold_out_of_range(self):
        """Test that threshold out of range raises ValueError."""
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            SuccessCriterion(
                id="test_001",
                description="Test",
                metric="completeness",
                threshold=1.5
            )

    def test_criterion_to_dict(self):
        """Test converting criterion to dictionary."""
        criterion = SuccessCriterion(
            id="test_001",
            description="Test criterion",
            metric="completeness",
            threshold=0.8
        )
        data = criterion.to_dict()
        assert data["id"] == "test_001"
        assert data["description"] == "Test criterion"
        assert data["metric"] == "completeness"
        assert data["threshold"] == 0.8

    def test_criterion_from_dict(self):
        """Test creating criterion from dictionary."""
        data = {
            "id": "test_001",
            "description": "Test criterion",
            "metric": "completeness",
            "threshold": 0.8
        }
        criterion = SuccessCriterion.from_dict(data)
        assert criterion.id == "test_001"
        assert criterion.description == "Test criterion"
        assert criterion.metric == "completeness"
        assert criterion.threshold == 0.8

    def test_criterion_equality(self):
        """Test criterion equality based on ID."""
        criterion1 = SuccessCriterion(
            id="test_001",
            description="First",
            metric="completeness",
            threshold=0.8
        )
        criterion2 = SuccessCriterion(
            id="test_001",
            description="Second",
            metric="performance",
            threshold=0.9
        )
        assert criterion1 == criterion2

    def test_criterion_hashable(self):
        """Test that criterion is hashable for use in sets."""
        criterion = SuccessCriterion(
            id="test_001",
            description="Test",
            metric="completeness",
            threshold=0.8
        )
        criteria_set = {criterion}
        assert len(criteria_set) == 1


# ============================================================================
# SuccessCriteriaManager TESTS
# ============================================================================

class TestSuccessCriteriaManager:
    """Test cases for SuccessCriteriaManager class."""

    def test_manager_initialization(self):
        """Test manager initialization with default config."""
        manager = SuccessCriteriaManager()
        assert manager.cache_enabled is True
        assert manager.default_threshold == 0.8
        assert manager.strict_validation is True

    def test_manager_initialization_with_config(self):
        """Test manager initialization with custom config."""
        config = {
            'cache_enabled': False,
            'default_threshold': 0.7,
            'strict_validation': False
        }
        manager = SuccessCriteriaManager(config)
        assert manager.cache_enabled is False
        assert manager.default_threshold == 0.7
        assert manager.strict_validation is False

    # -----------------------------------------------------------------------
    # create_criteria tests
    # -----------------------------------------------------------------------

    def test_create_criteria_from_requirements(self, manager):
        """Test creating criteria from a list of requirements."""
        requirements = [
            "The solution must implement authentication",
            "Performance should be optimized",
            "Security measures must be implemented"
        ]
        criteria = manager.create_criteria(requirements)
        assert len(criteria) == 3
        assert all(isinstance(c, SuccessCriterion) for c in criteria)

    def test_create_criteria_empty_list(self, manager):
        """Test that empty requirements list raises ValueError."""
        with pytest.raises(ValueError, match="Requirements list cannot be empty"):
            manager.create_criteria([])

    def test_create_criteria_invalid_requirement(self, manager):
        """Test handling of invalid requirements."""
        requirements = [
            "Valid requirement",
            "   ",  # Invalid (whitespace only)
            "Another valid requirement"
        ]
        # Should raise error because all requirements must be valid
        with pytest.raises(ValueError, match="All requirements must be non-empty strings"):
            manager.create_criteria(requirements)

    # -----------------------------------------------------------------------
    # define_criterion tests
    # -----------------------------------------------------------------------

    def test_define_criterion_basic(self, manager):
        """Test defining a basic criterion."""
        criterion = manager.define_criterion(
            "The solution must be secure",
            metric_type="security"
        )
        assert isinstance(criterion, SuccessCriterion)
        assert criterion.metric == "security"
        assert 0.0 <= criterion.threshold <= 1.0

    def test_define_criterion_auto_detect_metric_type(self, manager):
        """Test automatic metric type detection."""
        criterion = manager.define_criterion(
            "The system must be secure and protect user data"
        )
        # Should detect "security" from keywords
        assert "secur" in criterion.metric.lower() or criterion.metric == "security"

    def test_define_criterion_empty_requirement(self, manager):
        """Test that empty requirement raises ValueError."""
        with pytest.raises(ValueError, match="Requirement must be a non-empty string"):
            manager.define_criterion("")

    def test_define_criterion_with_percentage(self, manager):
        """Test threshold inference from percentage."""
        criterion = manager.define_criterion(
            "Must achieve 95% accuracy"
        )
        # Should infer 0.95 from "95%"
        assert criterion.threshold >= 0.94 and criterion.threshold <= 0.96

    # -----------------------------------------------------------------------
    # set_threshold tests
    # -----------------------------------------------------------------------

    def test_set_threshold_valid(self, manager, sample_criteria):
        """Test setting a valid threshold."""
        criterion = sample_criteria[0]
        updated = manager.set_threshold(criterion, 0.95)
        assert updated.threshold == 0.95
        assert updated.id == criterion.id

    def test_set_threshold_invalid_type(self, manager, sample_criteria):
        """Test that invalid threshold type raises ValueError."""
        criterion = sample_criteria[0]
        with pytest.raises(ValueError, match="Threshold must be a number"):
            manager.set_threshold(criterion, "invalid")

    def test_set_threshold_out_of_range(self, manager, sample_criteria):
        """Test that out-of-range threshold raises ValueError."""
        criterion = sample_criteria[0]
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            manager.set_threshold(criterion, 1.5)

    # -----------------------------------------------------------------------
    # check_criteria_satisfaction tests
    # -----------------------------------------------------------------------

    def test_check_criteria_satisfaction_basic(self, manager, sample_solution, sample_criteria):
        """Test basic criteria satisfaction checking."""
        results = manager.check_criteria_satisfaction(sample_solution, sample_criteria)
        assert isinstance(results, dict)
        assert len(results) == len(sample_criteria)
        assert all(isinstance(v, bool) for v in results.values())

    def test_check_criteria_satisfaction_empty_solution(self, manager, empty_solution, sample_criteria):
        """Test satisfaction checking with empty solution."""
        results = manager.check_criteria_satisfaction(empty_solution, sample_criteria[:1])
        assert isinstance(results, dict)
        # Empty solution should fail most criteria
        assert not all(results.values())

    def test_check_criteria_satisfaction_none_solution(self, manager, sample_criteria):
        """Test that None solution raises ValueError."""
        with pytest.raises(ValueError, match="Solution cannot be None"):
            manager.check_criteria_satisfaction(None, sample_criteria)

    def test_check_criteria_satisfaction_empty_criteria(self, manager, sample_solution):
        """Test that empty criteria list raises ValueError."""
        with pytest.raises(ValueError, match="Criteria list cannot be empty"):
            manager.check_criteria_satisfaction(sample_solution, [])

    # -----------------------------------------------------------------------
    # calculate_criteria_score tests
    # -----------------------------------------------------------------------

    def test_calculate_criteria_score_basic(self, manager, sample_solution, sample_criteria):
        """Test basic score calculation."""
        score = manager.calculate_criteria_score(sample_solution, sample_criteria)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_calculate_criteria_score_empty_solution(self, manager, empty_solution, sample_criteria):
        """Test score calculation with empty solution."""
        score = manager.calculate_criteria_score(empty_solution, sample_criteria[:1])
        assert score == 0.0

    def test_calculate_criteria_score_none_solution(self, manager, sample_criteria):
        """Test that None solution raises ValueError."""
        with pytest.raises(ValueError, match="Solution cannot be None"):
            manager.calculate_criteria_score(None, sample_criteria)

    # -----------------------------------------------------------------------
    # generate_criteria_report tests
    # -----------------------------------------------------------------------

    def test_generate_criteria_report_without_solution(self, manager, sample_criteria):
        """Test generating report without solution."""
        report = manager.generate_criteria_report(sample_criteria)
        assert isinstance(report, str)
        assert "SUCCESS CRITERIA REPORT" in report
        assert "Total criteria:" in report

    def test_generate_criteria_report_with_solution(self, manager, sample_solution, sample_criteria):
        """Test generating report with solution."""
        report = manager.generate_criteria_report(sample_criteria, sample_solution)
        assert isinstance(report, str)
        assert "SUCCESS CRITERIA REPORT" in report
        assert "Evaluation:" in report or "SATISFIED" in report or "FAILED" in report

    def test_generate_criteria_report_empty_criteria(self, manager):
        """Test that empty criteria list raises ValueError."""
        with pytest.raises(ValueError, match="Criteria list cannot be empty"):
            manager.generate_criteria_report([])

    # -----------------------------------------------------------------------
    # prioritize_criteria tests
    # -----------------------------------------------------------------------

    def test_prioritize_criteria_without_weights(self, manager, sample_criteria):
        """Test prioritization without weights (original order)."""
        prioritized = manager.prioritize_criteria(sample_criteria)
        assert len(prioritized) == len(sample_criteria)
        # Should maintain original order
        assert [c.id for c in prioritized] == [c.id for c in sample_criteria]

    def test_prioritize_criteria_with_weights(self, manager, sample_criteria):
        """Test prioritization with weights."""
        weights = {
            sample_criteria[0].id: 2.0,
            sample_criteria[1].id: 1.5,
        }
        prioritized = manager.prioritize_criteria(sample_criteria, weights)
        assert len(prioritized) == len(sample_criteria)
        # First criterion should be first (highest weight)
        assert prioritized[0].id == sample_criteria[0].id

    def test_prioritize_criteria_empty_list(self, manager):
        """Test that empty criteria list raises ValueError."""
        with pytest.raises(ValueError, match="Criteria list cannot be empty"):
            manager.prioritize_criteria([])

    def test_prioritize_criteria_invalid_weights(self, manager, sample_criteria):
        """Test that invalid weights type raises ValueError."""
        with pytest.raises(ValueError, match="Weights must be a dictionary"):
            manager.prioritize_criteria(sample_criteria, "invalid")


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_merge_criteria_keep_first(self):
        """Test merging criteria with 'keep_first' strategy."""
        criteria1 = [
            SuccessCriterion(id="crit_001", description="First", metric="test", threshold=0.8)
        ]
        criteria2 = [
            SuccessCriterion(id="crit_001", description="Second", metric="test", threshold=0.9)
        ]
        merged = merge_criteria(criteria1, criteria2, "keep_first")
        assert len(merged) == 1
        assert merged[0].description == "First"

    def test_merge_criteria_keep_second(self):
        """Test merging criteria with 'keep_second' strategy."""
        criteria1 = [
            SuccessCriterion(id="crit_001", description="First", metric="test", threshold=0.8)
        ]
        criteria2 = [
            SuccessCriterion(id="crit_001", description="Second", metric="test", threshold=0.9)
        ]
        merged = merge_criteria(criteria1, criteria2, "keep_second")
        assert len(merged) == 1
        assert merged[0].description == "Second"

    def test_merge_criteria_merge_strategy(self):
        """Test merging criteria with 'merge' strategy."""
        criteria1 = [
            SuccessCriterion(id="crit_001", description="First", metric="test", threshold=0.8)
        ]
        criteria2 = [
            SuccessCriterion(id="crit_001", description="Second", metric="test", threshold=0.9)
        ]
        merged = merge_criteria(criteria1, criteria2, "merge")
        assert len(merged) == 1
        # Should keep higher threshold
        assert merged[0].threshold == 0.9

    def test_merge_criteria_invalid_strategy(self):
        """Test that invalid strategy raises ValueError."""
        criteria1 = [SuccessCriterion(id="crit_001", description="Test", metric="test", threshold=0.8)]
        criteria2 = [SuccessCriterion(id="crit_002", description="Test2", metric="test", threshold=0.9)]
        with pytest.raises(ValueError, match="Invalid resolve_duplicates strategy"):
            merge_criteria(criteria1, criteria2, "invalid")

    def test_filter_criteria_by_type(self):
        """Test filtering criteria by type."""
        criteria = [
            SuccessCriterion(id="crit_001", description="Security", metric="security", threshold=0.8),
            SuccessCriterion(id="crit_002", description="Performance", metric="performance", threshold=0.7),
            SuccessCriterion(id="crit_003", description="Auth", metric="security", threshold=0.9),
        ]
        filtered = filter_criteria_by_type(criteria, "security")
        assert len(filtered) == 2
        assert all(c.metric == "security" for c in filtered)

    def test_filter_criteria_by_threshold(self):
        """Test filtering criteria by threshold range."""
        criteria = [
            SuccessCriterion(id="crit_001", description="High", metric="test", threshold=0.9),
            SuccessCriterion(id="crit_002", description="Medium", metric="test", threshold=0.7),
            SuccessCriterion(id="crit_003", description="Low", metric="test", threshold=0.5),
        ]
        filtered = filter_criteria_by_threshold(criteria, 0.7, 1.0)
        assert len(filtered) == 2

    def test_calculate_criteria_statistics(self):
        """Test calculating statistics for criteria."""
        criteria = [
            SuccessCriterion(id="crit_001", description="Test1", metric="security", threshold=0.9),
            SuccessCriterion(id="crit_002", description="Test2", metric="performance", threshold=0.7),
            SuccessCriterion(id="crit_003", description="Test3", metric="security", threshold=0.8),
        ]
        stats = calculate_criteria_statistics(criteria)
        assert stats["total"] == 3
        assert stats["avg_threshold"] == pytest.approx(0.8)
        assert stats["min_threshold"] == 0.7
        assert stats["max_threshold"] == 0.9
        assert stats["by_type"]["security"] == 2
        assert stats["by_type"]["performance"] == 1

    def test_calculate_criteria_statistics_empty(self):
        """Test statistics calculation with empty list."""
        stats = calculate_criteria_statistics([])
        assert stats["total"] == 0
        assert stats["avg_threshold"] == 0.0

    def test_validate_criteria_valid(self):
        """Test validation of valid criteria."""
        criteria = [
            SuccessCriterion(id="crit_001", description="Test1", metric="test", threshold=0.8),
            SuccessCriterion(id="crit_002", description="Test2", metric="test", threshold=0.9),
        ]
        is_valid, errors = validate_criteria(criteria)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_criteria_duplicate_ids(self):
        """Test validation catches duplicate IDs."""
        criteria = [
            SuccessCriterion(id="crit_001", description="Test1", metric="test", threshold=0.8),
            SuccessCriterion(id="crit_001", description="Test2", metric="test", threshold=0.9),
        ]
        is_valid, errors = validate_criteria(criteria)
        assert is_valid is False
        assert len(errors) > 0
        assert any("Duplicate" in error for error in errors)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for end-to-end workflows."""

    def test_full_criteria_workflow(self, manager):
        """Test complete workflow from creation to evaluation."""
        # 1. Create criteria from requirements
        requirements = [
            "Must implement secure authentication",
            "Should be performant and efficient",
            "Must be well-documented"
        ]
        criteria = manager.create_criteria(requirements)
        assert len(criteria) == 3

        # 2. Create a solution
        content = "This solution provides secure authentication with JWT tokens, " \
                 "is optimized for performance with caching, and includes " \
                 "comprehensive documentation and comments."

        if HAS_REAL_SOLUTION_ATTEMPT:
            solution = SolutionAttempt(
                sub_problem_id="integration_test",
                solution_content=content,
                confidence_score=0.8,
                execution_method=ExecutionMethod.TRADITIONAL
            )
        else:
            solution = SolutionAttempt(
                sub_problem_id="integration_test",
                solution_content=content,
                confidence_score=0.8,
                execution_method="traditional"
            )

        # 3. Check satisfaction
        satisfaction = manager.check_criteria_satisfaction(solution, criteria)
        assert len(satisfaction) == 3

        # 4. Calculate score
        score = manager.calculate_criteria_score(solution, criteria)
        assert 0.0 <= score <= 1.0

        # 5. Generate report
        report = manager.generate_criteria_report(criteria, solution)
        assert "SUCCESS CRITERIA REPORT" in report

        # 6. Prioritize with weights
        weights = {criteria[0].id: 2.0}
        prioritized = manager.prioritize_criteria(criteria, weights)
        assert len(prioritized) == 3

    def test_criteria_persistence_workflow(self, manager):
        """Test workflow with criteria persistence (dict conversion)."""
        # Create criterion
        criterion = manager.define_criterion("Test requirement", metric_type="completeness")

        # Convert to dict
        criterion_dict = criterion.to_dict()
        assert "id" in criterion_dict
        assert "description" in criterion_dict

        # Recreate from dict
        recreated = SuccessCriterion.from_dict(criterion_dict)
        assert recreated.id == criterion.id
        assert recreated.description == criterion.description


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest if available
    try:
        import pytest
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        print("pytest not installed, running basic tests...")
        print("Install pytest for full test coverage: pip install pytest")

        # Run basic tests
        print("\n" + "=" * 80)
        print("Running basic tests...")
        print("=" * 80)

        manager = SuccessCriteriaManager()
        print("✓ Manager initialized")

        # Test create_criteria
        criteria = manager.create_criteria([
            "Must implement authentication",
            "Should be secure"
        ])
        print(f"✓ Created {len(criteria)} criteria")

        # Test define_criterion
        criterion = manager.define_criterion("Test requirement")
        print("✓ Defined criterion")

        # Test set_threshold
        updated = manager.set_threshold(criterion, 0.9)
        print("✓ Set threshold")

        # Create test solution
        solution = SolutionAttempt(
            id="test",
            problem_id="test",
            solution="Test solution with security and performance features",
            score=0.8,
            timestamp=datetime.now(timezone.utc)
        )

        # Test evaluation
        satisfaction = manager.check_criteria_satisfaction(solution, criteria)
        print(f"✓ Checked satisfaction: {sum(satisfaction.values())}/{len(satisfaction)} satisfied")

        score = manager.calculate_criteria_score(solution, criteria)
        print(f"✓ Calculated score: {score:.2%}")

        report = manager.generate_criteria_report(criteria)
        print("✓ Generated report")

        print("\n" + "=" * 80)
        print("All basic tests passed!")
        print("=" * 80)
