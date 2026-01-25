"""
Comprehensive unit tests for ACI Reduction Validator (Δ₃)

Tests the complete 8-stage validation pipeline for non-circular
validation of RESE inventions through ACI reduction measurement.

Author: Agent E3 (Δ₃ Specialist)
Created: 2025-12-31
"""

import pytest
import numpy as np
import random
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# Try to import ACI reduction validator
try:
    from rese.phase4.aci_reduction_validator import (
        Delta3Validator,
        Problem,
        RESESolution,
        Delta3Config,
        ValidationResult,
        ValidationMetrics,
        ACIReductionMetrics,
        StatisticalTestResults,
        EffectSizeMetrics,
        ConfidenceIntervalMetrics,
        IndependenceCheckResult,
        PhaseTransitionResult,
        ACIMeasurement,
        ConstraintPartition,
        ValidationStatus,
        EffectSizeMagnitude,
        validate_rese_invention,
        validate_rese_batch,
        Delta3Error,
        DataLeakageError,
        CircularityError,
        IndependenceViolationError
    )
except ImportError:
    pytest.skip("ACI reduction validator module not available", allow_module_level=True)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def basic_config():
    """Create basic Delta3 config"""
    return Delta3Config(
        significance_level=0.05,
        min_aci_reduction=0.2,
        bootstrap_iterations=100,
        confidence_level=0.95,
        validation_threshold=0.7
    )


@pytest.fixture
def sample_problem():
    """Create sample problem"""
    return Problem(
        id="test_001",
        description="Optimize routing problem",
        constraints=["c1", "c2", "c3", "c4", "c5"],
        variables={"x": 0, "y": 0},
        objective="Minimize cost"
    )


@pytest.fixture
def sample_rese_solution():
    """Create sample RESE solution"""
    return RESESolution(
        problem_id="test_001",
        solution={"x": 1, "y": 2, "cost": 10},
        aci_history=[8.0, 6.0, 4.5, 3.0, 2.5, 2.0],
        stage_results={"stage1": "complete", "stage2": "complete"}
    )


# =============================================================================
# Problem and RESESolution Tests
# =============================================================================

class TestDataStructures:
    """Test data structures"""

    def test_problem_initialization(self):
        """Test Problem initialization"""
        problem = Problem(
            id="prob_001",
            description="Test problem",
            constraints=["c1", "c2"],
            variables={"x": 0},
            objective="Maximize value",
            domain="optimization"
        )

        assert problem.id == "prob_001"
        assert problem.description == "Test problem"
        assert len(problem.constraints) == 2
        assert problem.variables == {"x": 0}
        assert problem.objective == "Maximize value"
        assert problem.domain == "optimization"

    def test_rese_solution_initialization(self):
        """Test RESESolution initialization"""
        solution = RESESolution(
            problem_id="prob_001",
            solution={"x": 1},
            aci_history=[0.8, 0.6, 0.4],
            stage_results={"s1": {}, "s2": {}}
        )

        assert solution.problem_id == "prob_001"
        assert solution.solution == {"x": 1}
        assert len(solution.aci_history) == 3
        assert isinstance(solution.timestamp, datetime)

    def test_aci_measurement(self):
        """Test ACIMeasurement structure"""
        measurement = ACIMeasurement(
            timestamp=datetime.now(),
            aci_value=0.65,
            disorder_entropy=0.7,
            causal_coherence=0.6,
            num_constraints=5,
            stage="baseline"
        )

        assert measurement.aci_value == 0.65
        assert measurement.disorder_entropy == 0.7
        assert measurement.stage == "baseline"

    def test_constraint_partition(self):
        """Test ConstraintPartition structure"""
        partition = ConstraintPartition(
            training_constraints=["c1", "c2", "c3"],
            holdout_constraints=["c4", "c5"],
            partition_method="random",
            stratification={"ratio": 0.4}
        )

        assert len(partition.training_constraints) == 3
        assert len(partition.holdout_constraints) == 2
        assert partition.partition_method == "random"


# =============================================================================
# Delta3Config Tests
# =============================================================================

class TestDelta3Config:
    """Test Delta3Config functionality"""

    def test_default_values(self):
        """Test default configuration"""
        config = Delta3Config()

        assert config.significance_level == 0.05
        assert config.min_effect_size == 0.5
        assert config.bootstrap_iterations == 1000
        assert config.confidence_level == 0.95
        assert config.holdout_ratio == 0.2
        assert config.min_aci_reduction == 0.2
        assert config.validation_threshold == 0.7

    def test_custom_values(self):
        """Test custom configuration"""
        config = Delta3Config(
            significance_level=0.01,
            min_aci_reduction=0.3,
            validation_threshold=0.8
        )

        assert config.significance_level == 0.01
        assert config.min_aci_reduction == 0.3
        assert config.validation_threshold == 0.8


# =============================================================================
# ACI Measurement Tests
# =============================================================================

class TestACIMeasurement:
    """Test ACI measurement functionality"""

    def test_measure_aci_baseline(self, basic_config, sample_problem):
        """Test baseline ACI measurement"""
        validator = Delta3Validator(basic_config)

        baseline = validator._measure_aci_baseline(sample_problem)

        assert baseline.stage == "baseline"
        assert baseline.aci_value > 0
        assert baseline.disorder_entropy > 0
        assert baseline.causal_coherence >= 0

    def test_measure_aci_final(self, basic_config, sample_problem, sample_rese_solution):
        """Test final ACI measurement"""
        validator = Delta3Validator(basic_config)

        final = validator._measure_aci_final(sample_problem, sample_rese_solution)

        assert final.stage == "final"
        assert final.aci_value >= 0  # Non-negative

    def test_measure_aci_final_empty_history(self, basic_config, sample_problem):
        """Test final ACI measurement with empty history"""
        validator = Delta3Validator(basic_config)

        solution = RESESolution(
            problem_id="test_001",
            solution={},
            constraints=[],
            aci_history=[]  # Empty
        )

        final = validator._measure_aci_final(sample_problem, solution)

        # Should use fallback
        assert final is not None
        assert final.aci_value >= 0


# =============================================================================
# ACI Reduction Tests
# =============================================================================

class TestACIReduction:
    """Test ACI reduction calculation"""

    def test_calculate_aci_reduction(self, basic_config):
        """Test ACI reduction calculation"""
        validator = Delta3Validator(basic_config)

        baseline = ACIMeasurement(
            timestamp=datetime.now(),
            aci_value=8.0,
            disorder_entropy=2.0,
            causal_coherence=0.5,
            num_constraints=5,
            stage="baseline"
        )

        final = ACIMeasurement(
            timestamp=datetime.now(),
            aci_value=2.0,
            disorder_entropy=0.5,
            causal_coherence=0.8,
            num_constraints=5,
            stage="final"
        )

        reduction = validator._calculate_aci_reduction(baseline, final)

        assert reduction.baseline_aci == 8.0
        assert reduction.final_aci == 2.0
        assert reduction.absolute_reduction == 6.0
        assert reduction.relative_reduction == 0.75  # 75% reduction

    def test_calculate_aci_reduction_zero_baseline(self, basic_config):
        """Test ACI reduction with zero baseline"""
        validator = Delta3Validator(basic_config)

        baseline = ACIMeasurement(
            timestamp=datetime.now(),
            aci_value=0.0,
            disorder_entropy=0.0,
            causal_coherence=0.0,
            num_constraints=0,
            stage="baseline"
        )

        final = ACIMeasurement(
            timestamp=datetime.now(),
            aci_value=0.0,
            disorder_entropy=0.0,
            causal_coherence=0.0,
            num_constraints=0,
            stage="final"
        )

        reduction = validator._calculate_aci_reduction(baseline, final)

        # Should handle zero division
        assert reduction.relative_reduction == 0.0


# =============================================================================
# Validation Score Tests
# =============================================================================

class TestValidationScore:
    """Test validation score computation"""

    def test_compute_validation_score_valid(self, basic_config):
        """Test score for valid invention"""
        validator = Delta3Validator(basic_config)

        # Create metrics for valid invention
        metrics = ValidationMetrics(
            aci_reduction=ACIReductionMetrics(
                absolute_reduction=5.0,
                relative_reduction=0.7,
                baseline_aci=7.0,
                final_aci=2.0,
                meets_threshold=True
            ),
            statistical_tests=StatisticalTestResults(
                test_used="t_test",
                p_value=0.001,
                is_significant=True,
                test_statistic=3.5,
                degrees_of_freedom=98,
                critical_value=1.98,
                effect_size=0.8
            ),
            effect_sizes=EffectSizeMetrics(
                cohens_d=0.8,
                magnitude=EffectSizeMagnitude.LARGE,
                meets_threshold=True
            ),
            confidence_intervals=ConfidenceIntervalMetrics(
                ci_level=0.95,
                lower_bound=0.6,
                upper_bound=0.8,
                excludes_zero=True,
                width=0.2,
                method="bca"
            ),
            independence_check=IndependenceCheckResult(
                is_independent=True,
                data_leakage_detected=False,
                holdout_integrity=True,
                circularity_detected=False
            ),
            phase_transition=PhaseTransitionResult(
                phase_transition_detected=True,
                transition_point=10,
                aci_change=5.0,
                chaos_to_control=True,
                discontinuity_magnitude=2.0
            )
        )

        score = validator._compute_validation_score(metrics)

        assert score >= basic_config.validation_threshold

    def test_compute_validation_score_invalid_no_independence(self, basic_config):
        """Test score when independence violated"""
        validator = Delta3Validator(basic_config)

        metrics = ValidationMetrics(
            aci_reduction=ACIReductionMetrics(
                absolute_reduction=5.0,
                relative_reduction=0.7,
                baseline_aci=7.0,
                final_aci=2.0,
                meets_threshold=True
            ),
            statistical_tests=StatisticalTestResults(
                test_used="t_test",
                p_value=0.001,
                is_significant=True,
                test_statistic=3.5,
                degrees_of_freedom=98,
                critical_value=1.98
            ),
            effect_sizes=EffectSizeMetrics(
                cohens_d=0.8,
                magnitude=EffectSizeMagnitude.LARGE,
                meets_threshold=True
            ),
            confidence_intervals=ConfidenceIntervalMetrics(
                ci_level=0.95,
                lower_bound=0.6,
                upper_bound=0.8,
                excludes_zero=True,
                width=0.2,
                method="bca"
            ),
            independence_check=IndependenceCheckResult(
                is_independent=False,  # NOT independent
                data_leakage_detected=True,
                holdout_integrity=False,
                circularity_detected=True,
                issues=["Circular dependency detected"]
            ),
            phase_transition=PhaseTransitionResult(
                phase_transition_detected=True,
                chaos_to_control=True
            )
        )

        score = validator._compute_validation_score(metrics)

        # Should fail (no independence)
        assert score == 0.0


# =============================================================================
# Complete Validation Tests
# =============================================================================

class TestCompleteValidation:
    """Test complete validation pipeline"""

    def test_validate_success(self, basic_config, sample_problem, sample_rese_solution):
        """Test successful validation"""
        validator = Delta3Validator(basic_config)

        result = validator.validate(sample_problem, sample_rese_solution)

        assert isinstance(result, ValidationResult)
        assert isinstance(result.validation_score, float)
        assert 0.0 <= result.validation_score <= 1.0
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.confidence, float)

    def test_validate_with_error(self, basic_config, sample_problem):
        """Test validation with error handling"""
        validator = Delta3Validator(basic_config)

        # Create invalid solution
        invalid_solution = RESESolution(
            problem_id="test_001",
            solution=None,  # Invalid
            constraints=None,
            aci_history=None
        )

        result = validator.validate(sample_problem, invalid_solution)

        # Should return error result
        assert result.status == ValidationStatus.ERROR
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_compute_confidence(self, basic_config):
        """Test confidence computation"""
        validator = Delta3Validator(basic_config)

        # High score -> high confidence
        conf_high = validator._compute_confidence(0.95)
        assert conf_high >= 0.9

        # Low score -> low confidence
        conf_low = validator._compute_confidence(0.6)
        assert conf_low < 0.7

    def test_generate_decision_reason_valid(self, basic_config):
        """Test decision reason for valid invention"""
        validator = Delta3Validator(basic_config)

        metrics = ValidationMetrics(
            aci_reduction=ACIReductionMetrics(
                absolute_reduction=5.0,
                relative_reduction=0.7,
                baseline_aci=7.0,
                final_aci=2.0,
                meets_threshold=True
            ),
            statistical_tests=StatisticalTestResults(
                test_used="t_test",
                p_value=0.001,
                is_significant=True,
                test_statistic=3.5,
                degrees_of_freedom=98,
                critical_value=1.98
            ),
            effect_sizes=EffectSizeMetrics(
                cohens_d=0.8,
                magnitude=EffectSizeMagnitude.LARGE,
                meets_threshold=True
            ),
            confidence_intervals=ConfidenceIntervalMetrics(
                ci_level=0.95,
                lower_bound=0.6,
                upper_bound=0.8,
                excludes_zero=True,
                width=0.2,
                method="bca"
            ),
            independence_check=IndependenceCheckResult(
                is_independent=True,
                data_leakage_detected=False,
                holdout_integrity=True,
                circularity_detected=False
            ),
            phase_transition=PhaseTransitionResult(
                phase_transition_detected=False,
                chaos_to_control=False
            )
        )

        reason = validator._generate_decision_reason(
            is_valid=True,
            score=0.85,
            metrics=metrics
        )

        assert "Valid invention" in reason

    def test_generate_decision_reason_invalid(self, basic_config):
        """Test decision reason for invalid invention"""
        validator = Delta3Validator(basic_config)

        metrics = ValidationMetrics(
            aci_reduction=ACIReductionMetrics(
                absolute_reduction=0.5,
                relative_reduction=0.1,  # Below threshold
                baseline_aci=5.0,
                final_aci=4.5,
                meets_threshold=False
            ),
            statistical_tests=StatisticalTestResults(
                test_used="t_test",
                p_value=0.3,
                is_significant=False,
                test_statistic=1.0,
                degrees_of_freedom=98,
                critical_value=1.98
            ),
            effect_sizes=EffectSizeMetrics(
                cohens_d=0.2,
                magnitude=EffectSizeMagnitude.SMALL,
                meets_threshold=False
            ),
            confidence_intervals=ConfidenceIntervalMetrics(
                ci_level=0.95,
                lower_bound=-0.1,
                upper_bound=0.3,
                excludes_zero=False,
                width=0.4,
                method="normal"
            ),
            independence_check=IndependenceCheckResult(
                is_independent=True,
                data_leakage_detected=False,
                holdout_integrity=True,
                circularity_detected=False
            ),
            phase_transition=PhaseTransitionResult(
                phase_transition_detected=False,
                chaos_to_control=False
            )
        )

        reason = validator._generate_decision_reason(
            is_valid=False,
            score=0.3,
            metrics=metrics
        )

        assert "Invalid" in reason


# =============================================================================
# Constraint Partition Tests
# =============================================================================

class TestConstraintPartition:
    """Test constraint partitioning"""

    def test_create_constraint_partition(self, basic_config, sample_problem):
        """Test creating constraint partition"""
        validator = Delta3Validator(basic_config)

        partition = validator._create_constraint_partition(sample_problem)

        assert partition.partition_method == basic_config.holdout_method
        assert len(partition.training_constraints) > 0
        assert len(partition.holdout_constraints) > 0

        # Check total
        total = len(partition.training_constraints) + len(partition.holdout_constraints)
        assert total == len(sample_problem.constraints)

    def test_partition_stratification(self, basic_config, sample_problem):
        """Test partition stratification"""
        validator = Delta3Validator(basic_config)

        partition = validator._create_constraint_partition(sample_problem)

        assert 'ratio' in partition.stratification
        assert partition.stratification['ratio'] == basic_config.holdout_ratio


# =============================================================================
# ValidationResult Tests
# =============================================================================

class TestValidationResult:
    """Test ValidationResult functionality"""

    def test_validation_result_initialization(self):
        """Test ValidationResult initialization"""
        result = ValidationResult(
            is_valid=True,
            validation_score=0.85,
            confidence=0.9,
            status=ValidationStatus.VALID,
            metrics=None,  # type: ignore
            decision_reason="Valid invention",
            warnings=[],
            errors=[]
        )

        assert result.is_valid
        assert result.validation_score == 0.85
        assert result.confidence == 0.9
        assert result.status == ValidationStatus.VALID

    def test_validation_result_with_warnings(self):
        """Test ValidationResult with warnings"""
        result = ValidationResult(
            is_valid=True,
            validation_score=0.75,
            confidence=0.8,
            status=ValidationStatus.VALID,
            metrics=None,  # type: ignore
            decision_reason="Valid with warnings",
            warnings=["Small sample size", "High variance"],
            errors=[]
        )

        assert len(result.warnings) == 2


# =============================================================================
# Convenience Functions Tests
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_validate_rese_invention(self, sample_problem, sample_rese_solution):
        """Test validate_rese_invention convenience function"""
        result = validate_rese_invention(sample_problem, sample_rese_solution)

        assert isinstance(result, ValidationResult)

    def test_validate_rese_batch(self, basic_config):
        """Test batch validation"""
        # Create multiple problems and solutions
        problems = [
            Problem(
                id=f"prob_{i}",
                description=f"Problem {i}",
                constraints=[f"c{i}_1", f"c{i}_2"],
                variables={"x": i}
            )
            for i in range(3)
        ]

        solutions = [
            RESESolution(
                problem_id=f"prob_{i}",
                solution={"x": i + 1},
                aci_history=[5.0 - i, 3.0 - i, 1.0 - i]
            )
            for i in range(3)
        ]

        results = validate_rese_batch(problems, solutions, basic_config)

        assert len(results) == 3
        assert all(isinstance(r, ValidationResult) for r in results)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling"""

    def test_delta3_error(self):
        """Test Delta3Error"""
        with pytest.raises(Delta3Error):
            raise Delta3Error("Test error")

    def test_data_leakage_error(self):
        """Test DataLeakageError"""
        with pytest.raises(DataLeakageError):
            raise DataLeakageError("Data leakage detected")

    def test_circularity_error(self):
        """Test CircularityError"""
        with pytest.raises(CircularityError):
            raise CircularityError("Circular reasoning detected")

    def test_independence_violation_error(self):
        """Test IndependenceViolationError"""
        with pytest.raises(IndependenceViolationError):
            raise IndependenceViolationError("Independence violated")


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_empty_constraints(self, basic_config):
        """Test with empty constraints"""
        validator = Delta3Validator(basic_config)

        problem = Problem(
            id="empty",
            description="Empty problem",
            constraints=[],
            variables={}
        )

        solution = RESESolution(
            problem_id="empty",
            solution={},
            constraints=[],
            aci_history=[1.0]
        )

        result = validator.validate(problem, solution)

        # Should handle gracefully
        assert isinstance(result, ValidationResult)

    def test_single_constraint(self, basic_config):
        """Test with single constraint"""
        validator = Delta3Validator(basic_config)

        problem = Problem(
            id="single",
            description="Single constraint",
            constraints=["c1"],
            variables={"x": 0}
        )

        solution = RESESolution(
            problem_id="single",
            solution={"x": 1},
            constraints=["c1"],
            aci_history=[5.0, 2.5, 1.0]
        )

        result = validator.validate(problem, solution)

        assert isinstance(result, ValidationResult)

    def test_very_large_aci_reduction(self, basic_config):
        """Test with very large ACI reduction"""
        validator = Delta3Validator(basic_config)

        baseline = ACIMeasurement(
            timestamp=datetime.now(),
            aci_value=100.0,
            disorder_entropy=10.0,
            causal_coherence=0.1,
            num_constraints=100,
            stage="baseline"
        )

        final = ACIMeasurement(
            timestamp=datetime.now(),
            aci_value=1.0,
            disorder_entropy=0.1,
            causal_coherence=0.9,
            num_constraints=100,
            stage="final"
        )

        reduction = validator._calculate_aci_reduction(baseline, final)

        # Should handle large reduction
        assert reduction.relative_reduction > 0.9

    def test_no_aci_reduction(self, basic_config):
        """Test with no ACI reduction"""
        validator = Delta3Validator(basic_config)

        baseline = ACIMeasurement(
            timestamp=datetime.now(),
            aci_value=5.0,
            disorder_entropy=1.0,
            causal_coherence=0.5,
            num_constraints=10,
            stage="baseline"
        )

        final = ACIMeasurement(
            timestamp=datetime.now(),
            aci_value=5.0,
            disorder_entropy=1.0,
            causal_coherence=0.5,
            num_constraints=10,
            stage="final"
        )

        reduction = validator._calculate_aci_reduction(baseline, final)

        assert reduction.absolute_reduction == 0.0
        assert reduction.relative_reduction == 0.0
        assert not reduction.meets_threshold
