"""
Unit tests for RESE Phase IV Predictive Validator

Tests cover:
- Wilcoxon signed-rank test
- Mann-Whitney U test
- T-test (paired and independent)
- Bootstrap test
- Effect size calculation
- Confidence interval calculation
- Statistical significance assessment
- Prediction validation

Following CLAUDE.md principles:
- Law of Runtime Truth: Verify actual statistical behavior
- Law of Idempotency: Same inputs produce same results
"""

import pytest
import sys
import os
import math

# Add src and schemas to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "schemas"))

from src.predictive_validator import (
    PredictiveValidator,
    PredictiveValidationResult,
    StatisticalTest,
    StructuredLogger,
)
from rese_phase4_schemas import (
    ArchitectureAssembly,
    SynthesizedKnowledge,
    ParadigmShift,
    ParadigmShiftType,
    Phase4Config,
    AssemblyStatus,
    ValidationLevel,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return Phase4Config(
        assembly_timeout_ms=25000,
        validation_level=ValidationLevel.STANDARD,
        min_confidence_threshold=0.7,
        correlation_id="test-correlation-123",
    )


@pytest.fixture
def sample_assembly():
    """Create sample architecture assembly."""
    paradigm_shift = ParadigmShift(
        shift_type=ParadigmShiftType.STRUCTURAL,
        description="Test shift",
        confidence=0.85,
        validation_status="validated",
    )

    knowledge = SynthesizedKnowledge(
        knowledge_type="test",
        paradigm_shifts=[paradigm_shift],
        confidence=0.82,
        completeness=0.9,
        consistency=0.88,
    )

    return ArchitectureAssembly(
        synthesized_knowledge=knowledge,
        paradigm_shifts=[paradigm_shift],
        aci_reduction_achieved=0.35,
        confidence=0.82,
        validation_level=ValidationLevel.STANDARD,
        status=AssemblyStatus.VALIDATED,
    )


@pytest.fixture
def incumbent_aci_data():
    """Simulate incumbent paradigm ACI measurements."""
    return [0.85, 0.82, 0.88, 0.90, 0.87, 0.83, 0.86, 0.89, 0.84, 0.88]


@pytest.fixture
def new_aci_data():
    """Simulate new architecture ACI measurements (lower is better)."""
    return [0.55, 0.52, 0.58, 0.50, 0.56, 0.53, 0.54, 0.51, 0.57, 0.52]


# ============================================================================
# TEST: PREDICTIVE VALIDATOR INITIALIZATION
# ============================================================================

def test_predictive_validator_initialization(config):
    """Test PredictiveValidator initializes correctly."""
    validator = PredictiveValidator(config, test_type=StatisticalTest.WILCOXON)

    assert validator.config == config
    assert validator.logger is not None
    assert validator.test_type == StatisticalTest.WILCOXON
    assert validator.significance_level == 0.05
    assert validator.min_effect_size == 0.2


def test_predictive_validator_custom_config(config):
    """Test PredictiveValidator with custom configuration."""
    os.environ["PREDICTIVE_ALPHA"] = "0.01"
    os.environ["PREDICTIVE_MIN_EFFECT"] = "0.3"

    validator = PredictiveValidator(config)

    assert validator.significance_level == 0.01
    assert validator.min_effect_size == 0.3

    # Clean up
    del os.environ["PREDICTIVE_ALPHA"]
    del os.environ["PREDICTIVE_MIN_EFFECT"]


# ============================================================================
# TEST: VALIDATION - WILCOXON TEST
# ============================================================================

def test_validate_with_wilcoxon(sample_assembly, incumbent_aci_data, new_aci_data, config):
    """Test validation with Wilcoxon signed-rank test."""
    validator = PredictiveValidator(config, test_type=StatisticalTest.WILCOXON)
    result = validator.validate(sample_assembly, incumbent_aci_data, new_aci_data)

    assert isinstance(result, PredictiveValidationResult)
    assert result.is_valid is True  # Should detect significant improvement
    assert result.aci_reduction > 0
    assert result.incumbent_aci > result.new_aci
    assert result.test_used == StatisticalTest.WILCOXON

    # Check statistical significance
    assert result.statistical_significance["is_significant"] is True
    assert result.statistical_significance["p_value"] < 0.05


# ============================================================================
# TEST: VALIDATION - MANN-WHITNEY U TEST
# ============================================================================

def test_validate_with_mann_whitney(sample_assembly, incumbent_aci_data, new_aci_data, config):
    """Test validation with Mann-Whitney U test."""
    validator = PredictiveValidator(config, test_type=StatisticalTest.MANN_WHITNEY_U)
    result = validator.validate(sample_assembly, incumbent_aci_data, new_aci_data)

    assert isinstance(result, PredictiveValidationResult)
    assert result.test_used == StatisticalTest.MANN_WHITNEY_U
    assert result.is_valid is True


# ============================================================================
# TEST: VALIDATION - T-TESTS
# ============================================================================

def test_validate_with_t_test_paired(sample_assembly, incumbent_aci_data, new_aci_data, config):
    """Test validation with paired t-test."""
    validator = PredictiveValidator(config, test_type=StatisticalTest.T_TEST_PAIRED)
    result = validator.validate(sample_assembly, incumbent_aci_data, new_aci_data)

    assert isinstance(result, PredictiveValidationResult)
    assert result.test_used == StatisticalTest.T_TEST_PAIRED


def test_validate_with_t_test_independent(sample_assembly, incumbent_aci_data, new_aci_data, config):
    """Test validation with independent t-test."""
    validator = PredictiveValidator(config, test_type=StatisticalTest.T_TEST_INDEPENDENT)
    result = validator.validate(sample_assembly, incumbent_aci_data, new_aci_data)

    assert isinstance(result, PredictiveValidationResult)
    assert result.test_used == StatisticalTest.T_TEST_INDEPENDENT


# ============================================================================
# TEST: EFFECT SIZE CALCULATION
# ============================================================================

def test_effect_size_calculation(sample_assembly, incumbent_aci_data, new_aci_data, config):
    """Test effect size (Cohen's d) calculation."""
    validator = PredictiveValidator(config)
    result = validator.validate(sample_assembly, incumbent_aci_data, new_aci_data)

    # Cohen's d should be positive and substantial
    assert result.effect_size > 0
    # Large effect size is > 0.8
    assert result.effect_size > 0.8


# ============================================================================
# TEST: CONFIDENCE INTERVAL CALCULATION
# ============================================================================

def test_confidence_interval_calculation(sample_assembly, incumbent_aci_data, new_aci_data, config):
    """Test confidence interval calculation."""
    validator = PredictiveValidator(config)
    result = validator.validate(sample_assembly, incumbent_aci_data, new_aci_data)

    # Check confidence interval
    assert isinstance(result.confidence_interval, tuple)
    assert len(result.confidence_interval) == 2
    lower, upper = result.confidence_interval
    assert lower < upper
    assert lower < result.new_aci < upper


# ============================================================================
# TEST: PREDICTION VALIDATION
# ============================================================================

def test_prediction_validation(sample_assembly, incumbent_aci_data, new_aci_data, config):
    """Test prediction validation."""
    validator = PredictiveValidator(config)
    result = validator.validate(sample_assembly, incumbent_aci_data, new_aci_data)

    # Check predictions
    assert result.predictions_validated >= 0
    assert result.predictions_total >= 1
    assert result.predictions_validated <= result.predictions_total


# ============================================================================
# TEST: ERROR HANDLING
# ============================================================================

def test_validate_with_empty_measurements(sample_assembly, config):
    """Test error handling with empty measurements."""
    validator = PredictiveValidator(config)

    with pytest.raises(ValueError, match="Measurement lists cannot be empty"):
        validator.validate(sample_assembly, [], [])


def test_validate_with_insufficient_measurements(sample_assembly, config):
    """Test error handling with insufficient measurements."""
    validator = PredictiveValidator(config)

    with pytest.raises(ValueError, match="Need at least 3 measurements"):
        validator.validate(sample_assembly, [0.5, 0.6], [0.4, 0.3])


def test_validate_with_negative_measurements(sample_assembly, config):
    """Test error handling with negative measurements."""
    validator = PredictiveValidator(config)

    with pytest.raises(ValueError, match="ACI measurements must be non-negative"):
        validator.validate(sample_assembly, [0.5, -0.1, 0.6], [0.4, 0.3, 0.5])


def test_validate_with_nan_measurements(sample_assembly, config):
    """Test error handling with NaN measurements."""
    validator = PredictiveValidator(config)

    with pytest.raises(ValueError, match="contain NaN or Inf"):
        validator.validate(sample_assembly, [0.5, float('nan'), 0.6], [0.4, 0.3, 0.5])


# ============================================================================
# TEST: VALIDATION RESULT SERIALIZATION
# ============================================================================

def test_validation_result_to_dict(sample_assembly, incumbent_aci_data, new_aci_data, config):
    """Test PredictiveValidationResult serialization."""
    validator = PredictiveValidator(config)
    result = validator.validate(sample_assembly, incumbent_aci_data, new_aci_data)

    result_dict = result.to_dict()

    # Check all fields
    assert "validation_id" in result_dict
    assert "is_valid" in result_dict
    assert "aci_reduction" in result_dict
    assert "incumbent_aci" in result_dict
    assert "new_aci" in result_dict
    assert "effect_size" in result_dict
    assert "confidence_interval" in result_dict
    assert "statistical_significance" in result_dict
    assert "test_used" in result_dict
    assert "predictions_validated" in result_dict
    assert "predictions_total" in result_dict
    assert "metadata" in result_dict
    assert "validated_at" in result_dict

    # Check confidence interval structure
    assert "lower" in result_dict["confidence_interval"]
    assert "upper" in result_dict["confidence_interval"]


# ============================================================================
# TEST: STATISTICAL SIGNIFICANCE ASSESSMENT
# ============================================================================

def test_statistical_significance_assessment(sample_assembly, incumbent_aci_data, new_aci_data, config):
    """Test statistical significance assessment."""
    validator = PredictiveValidator(config)
    result = validator.validate(sample_assembly, incumbent_aci_data, new_aci_data)

    sig = result.statistical_significance

    # Check structure
    assert "is_significant" in sig
    assert "p_value" in sig
    assert "alpha" in sig
    assert "test_statistic" in sig
    assert "test_type" in sig

    # Check significance logic
    assert sig["is_significant"] == (sig["p_value"] < sig["alpha"])


# ============================================================================
# TEST: NORMAL CDF CALCULATION
# ============================================================================

def test_normal_cdf_calculation(config):
    """Test standard normal CDF calculation."""
    validator = PredictiveValidator(config)

    # Test standard values
    assert abs(validator._normal_cdf(0) - 0.5) < 0.01  # Mean
    assert abs(validator._normal_cdf(1.96) - 0.975) < 0.01  # 95% CI
    assert abs(validator._normal_cdf(-1.96) - 0.025) < 0.01  # Lower tail
    assert abs(validator._normal_cdf(3.0) - 0.9987) < 0.01  # High value


# ============================================================================
# TEST: IDEMPOTENCY
# ============================================================================

def test_validation_idempotency(sample_assembly, incumbent_aci_data, new_aci_data, config):
    """Test that validation is idempotent (Law of Idempotency)."""
    validator = PredictiveValidator(config)

    # Validate twice
    result1 = validator.validate(sample_assembly, incumbent_aci_data, new_aci_data)
    result2 = validator.validate(sample_assembly, incumbent_aci_data, new_aci_data)

    # Check results are consistent
    assert result1.is_valid == result2.is_valid
    assert result1.aci_reduction == result2.aci_reduction
    assert result1.effect_size == result2.effect_size
    assert result1.statistical_significance["p_value"] == result2.statistical_significance["p_value"]


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
