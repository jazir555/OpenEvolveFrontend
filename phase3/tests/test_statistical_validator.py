"""
Comprehensive unit tests for Statistical Validator Module

Tests all statistical validation methods including bootstrap
confidence intervals, significance testing, convergence detection,
and sequential analysis.

Author: Agent D2 (Γ₃ Specialist)
Created: 2025-12-31
"""

import pytest
import numpy as np
from scipy import stats
from typing import List, Tuple
import math

# Try to import statistical validator
try:
    from rese.phase3.statistical_validator import (
        StatisticalValidator,
        ValidationConfig,
        ValidationResult,
        ConfidenceInterval,
        SignificanceTestResult,
        ConvergenceResult,
        CIType,
        TestType,
        ConvergenceMethod,
        SequentialAnalyzer,
        quick_validation,
        compare_mcts_runs
    )
except ImportError:
    pytest.skip("Statistical validator module not available", allow_module_level=True)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    return np.random.normal(0.75, 0.05, 100).tolist()


@pytest.fixture
def basic_config():
    """Create basic validation config"""
    return ValidationConfig(
        num_bootstrap=100,
        confidence_level=0.95,
        convergence_window=20,
        verbose=False
    )


@pytest.fixture
def validator(basic_config):
    """Create validator instance"""
    return StatisticalValidator(config=basic_config)


# =============================================================================
# ValidationConfig Tests
# =============================================================================

class TestValidationConfig:
    """Test ValidationConfig functionality"""

    def test_default_values(self):
        """Test default configuration"""
        config = ValidationConfig()

        assert config.num_bootstrap == 1000
        assert config.ci_type == CIType.BCA
        assert config.confidence_level == 0.95
        assert config.convergence_method == ConvergenceMethod.COMBINED
        assert config.convergence_window == 20
        assert config.convergence_threshold == 0.001
        assert config.significance_level == 0.05
        assert config.test_type == TestType.T_TEST
        assert config.effect_size == 0.1
        assert config.power == 0.8

    def test_custom_values(self):
        """Test custom configuration"""
        config = ValidationConfig(
            num_bootstrap=500,
            confidence_level=0.99,
            convergence_threshold=0.01
        )

        assert config.num_bootstrap == 500
        assert config.confidence_level == 0.99
        assert config.convergence_threshold == 0.01


# =============================================================================
# Confidence Interval Tests
# =============================================================================

class TestConfidenceIntervals:
    """Test bootstrap confidence interval calculation"""

    def test_percentile_ci(self, validator, sample_data):
        """Test percentile method CI"""
        ci = validator.bootstrap_confidence_interval(
            sample_data,
            method=CIType.PERCENTILE
        )

        assert ci.lower < ci.upper
        assert ci.level == 0.95
        assert ci.method == CIType.PERCENTILE
        assert ci.width > 0

    def test_bca_ci(self, validator, sample_data):
        """Test BCa method CI"""
        ci = validator.bootstrap_confidence_interval(
            sample_data,
            method=CIType.BCA
        )

        assert ci.lower < ci.upper
        assert ci.level == 0.95
        assert ci.method == CIType.BCA

    def test_normal_ci(self, validator, sample_data):
        """Test normal approximation CI"""
        ci = validator.bootstrap_confidence_interval(
            sample_data,
            method=CIType.NORMAL
        )

        assert ci.lower < ci.upper
        assert ci.method == CIType.NORMAL

    def test_studentized_ci(self, validator, sample_data):
        """Test studentized (bootstrap-t) CI"""
        ci = validator.bootstrap_confidence_interval(
            sample_data,
            method=CIType.STUDENTIZED
        )

        assert ci.lower < ci.upper
        assert ci.method == CIType.STUDENTIZED

    def test_ci_with_small_sample(self, validator):
        """Test CI with small sample"""
        small_data = [1.0, 2.0, 3.0, 4.0, 5.0]

        ci = validator.bootstrap_confidence_interval(
            small_data,
            method=CIType.PERCENTILE
        )

        assert ci.lower < ci.upper

    def test_ci_with_empty_data(self, validator):
        """Test CI raises error with empty data"""
        with pytest.raises(ValueError):
            validator.bootstrap_confidence_interval([])

    def test_ci_different_confidence_levels(self, validator, sample_data):
        """Test CI with different confidence levels"""
        ci_90 = validator.bootstrap_confidence_interval(
            sample_data,
            confidence_level=0.90
        )

        ci_99 = validator.bootstrap_confidence_interval(
            sample_data,
            confidence_level=0.99
        )

        # Higher confidence -> wider interval
        assert ci_99.width > ci_90.width

    def test_percentile_ci_calculation(self, validator):
        """Test percentile CI calculation details"""
        bootstrap_stats = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        lower, upper = validator._percentile_ci(bootstrap_stats, 0.95)

        assert lower < upper
        assert lower > np.min(bootstrap_stats)
        assert upper < np.max(bootstrap_stats)

    def test_bca_ci_with_skewed_data(self, validator):
        """Test BCa CI handles skewed data"""
        skewed_data = np.random.exponential(1.0, 100).tolist()

        ci = validator.bootstrap_confidence_interval(
            skewed_data,
            method=CIType.BCA
        )

        assert ci.lower < ci.upper

    def test_normal_ci_calculation(self, validator, sample_data):
        """Test normal CI calculation"""
        bootstrap_stats = np.array(sample_data)

        lower, upper = validator._normal_ci(
            np.array(sample_data),
            bootstrap_stats,
            0.95
        )

        assert lower < upper
        assert abs(lower - np.mean(sample_data)) < 1.0
        assert abs(upper - np.mean(sample_data)) < 1.0


# =============================================================================
# Significance Testing Tests
# =============================================================================

class TestSignificanceTesting:
    """Test statistical significance testing"""

    def test_t_test(self, validator):
        """Test paired t-test"""
        np.random.seed(42)
        group_a = np.random.normal(0.75, 0.05, 50).tolist()
        group_b = np.random.normal(0.77, 0.05, 50).tolist()

        result = validator.significance_test(
            group_a,
            group_b,
            test_type=TestType.T_TEST
        )

        assert result.test_type == TestType.T_TEST
        assert isinstance(result.p_value, float)
        assert isinstance(result.statistic, float)
        assert isinstance(result.significant, bool)

    def test_wilcoxon_test(self, validator):
        """Test Wilcoxon signed-rank test"""
        np.random.seed(42)
        group_a = np.random.normal(0.75, 0.05, 50).tolist()
        group_b = np.random.normal(0.77, 0.05, 50).tolist()

        result = validator.significance_test(
            group_a,
            group_b,
            test_type=TestType.WILCOXON
        )

        assert result.test_type == TestType.WILCOXON
        assert isinstance(result.p_value, float)

    def test_mann_whitney_test(self, validator):
        """Test Mann-Whitney U test"""
        np.random.seed(42)
        group_a = np.random.normal(0.75, 0.05, 50).tolist()
        group_b = np.random.normal(0.80, 0.05, 50).tolist()

        result = validator.significance_test(
            group_a,
            group_b,
            test_type=TestType.MANN_WHITNEY
        )

        assert result.test_type == TestType.MANN_WHITNEY
        assert isinstance(result.p_value, float)

    def test_significant_difference(self, validator):
        """Test detection of significant difference"""
        np.random.seed(42)
        group_a = np.random.normal(0.5, 0.01, 100).tolist()
        group_b = np.random.normal(0.7, 0.01, 100).tolist()

        result = validator.significance_test(
            group_a,
            group_b,
            test_type=TestType.T_TEST,
            alpha=0.05
        )

        # Should be significant
        assert result.significant
        assert result.p_value < 0.05

    def test_no_significant_difference(self, validator):
        """Test when no significant difference"""
        np.random.seed(42)
        group_a = np.random.normal(0.5, 0.01, 100).tolist()
        group_b = np.random.normal(0.501, 0.01, 100).tolist()

        result = validator.significance_test(
            group_a,
            group_b,
            test_type=TestType.T_TEST,
            alpha=0.05
        )

        # Should not be significant
        assert not result.significant
        assert result.p_value >= 0.05

    def test_unequal_sample_sizes(self, validator):
        """Test with unequal sample sizes"""
        group_a = [1.0] * 50
        group_b = [1.1] * 100

        # Should not raise error
        result = validator.significance_test(group_a, group_b)
        assert result is not None

    def test_unknown_test_type(self, validator):
        """Test unknown test type raises error"""
        with pytest.raises(ValueError):
            validator.significance_test(
                [1.0, 2.0],
                [1.1, 2.1],
                test_type="unknown"
            )


# =============================================================================
# Convergence Detection Tests
# =============================================================================

class TestConvergenceDetection:
    """Test convergence detection methods"""

    def test_moving_window_convergence(self, validator):
        """Test moving window convergence detection"""
        # Create converging sequence
        value_history = []
        current = 0.5
        for i in range(100):
            improvement = 0.01 * np.exp(-i / 20)
            current += improvement
            value_history.append(current)

        result = validator.detect_convergence(
            value_history,
            method=ConvergenceMethod.MOVING_WINDOW,
            window=20,
            threshold=0.001
        )

        assert isinstance(result.converged, bool)
        assert result.method == ConvergenceMethod.MOVING_WINDOW
        assert isinstance(result.confidence, float)

    def test_gradient_convergence(self, validator):
        """Test gradient-based convergence detection"""
        # Create converging sequence
        value_history = [1.0 - np.exp(-i/20) for i in range(100)]

        result = validator.detect_convergence(
            value_history,
            method=ConvergenceMethod.GRADIENT,
            window=20,
            threshold=0.001
        )

        assert isinstance(result.converged, bool)
        assert 'gradient' in result.details

    def test_spc_convergence(self, validator):
        """Test statistical process control convergence"""
        # Stable sequence
        value_history = [1.0 + np.random.normal(0, 0.001) for _ in range(50)]

        result = validator.detect_convergence(
            value_history,
            method=ConvergenceMethod.SPC,
            window=20
        )

        assert isinstance(result.converged, bool)
        assert 'mean' in result.details
        assert 'std' in result.details

    def test_combined_convergence(self, validator):
        """Test combined convergence detection"""
        value_history = [1.0 - np.exp(-i/20) for i in range(100)]

        result = validator.detect_convergence(
            value_history,
            method=ConvergenceMethod.COMBINED,
            window=20,
            threshold=0.001
        )

        assert isinstance(result.converged, bool)
        assert 'moving_window' in result.details
        assert 'gradient' in result.details
        assert 'spc' in result.details

    def test_insufficient_data(self, validator):
        """Test convergence with insufficient data"""
        short_history = [1.0, 1.1, 1.2]

        result = validator.detect_convergence(
            short_history,
            window=20
        )

        assert not result.converged
        assert 'Insufficient data' in result.details.get('reason', '')

    def test_convergence_moving_window_details(self, validator):
        """Test moving window convergence details"""
        # Very stable sequence
        value_history = [1.0 for _ in range(30)]

        result = validator._convergence_moving_window(
            value_history,
            window=20,
            threshold=0.01
        )

        assert result.converged
        assert result.details['std'] < result.details['threshold']

    def test_convergence_gradient_details(self, validator):
        """Test gradient convergence details"""
        # Flat sequence
        value_history = [1.0 for _ in range(30)]

        result = validator._convergence_gradient(
            value_history,
            window=20,
            threshold=0.01
        )

        assert result.converged
        assert result.details['gradient'] < result.details['threshold']

    def test_convergence_spc_details(self, validator):
        """Test SPC convergence details"""
        # All points within limits
        value_history = [1.0 + np.random.normal(0, 0.0001) for _ in range(30)]

        result = validator._convergence_spc(value_history, window=20)

        assert result.converged
        assert 'upper_limit' in result.details
        assert 'lower_limit' in result.details


# =============================================================================
# Sample Size Tests
# =============================================================================

class TestSampleSize:
    """Test sample size calculation"""

    def test_required_sample_size_two_sample(self, validator):
        """Test sample size for two-sample test"""
        n = validator.required_sample_size(
            effect_size=0.5,
            alpha=0.05,
            power=0.8,
            test='two_sample'
        )

        assert n > 0
        assert isinstance(n, int)

    def test_required_sample_size_paired(self, validator):
        """Test sample size for paired test (more powerful)"""
        n_two_sample = validator.required_sample_size(
            effect_size=0.5,
            test='two_sample'
        )

        n_paired = validator.required_sample_size(
            effect_size=0.5,
            test='paired'
        )

        # Paired requires fewer samples
        assert n_paired < n_two_sample

    def test_required_sample_size_one_sample(self, validator):
        """Test sample size for one-sample test"""
        n = validator.required_sample_size(
            effect_size=0.5,
            test='one_sample'
        )

        assert n > 0

    def test_sample_size_scales_with_effect_size(self, validator):
        """Test sample size inversely related to effect size"""
        n_small_effect = validator.required_sample_size(
            effect_size=0.2,
            test='two_sample'
        )

        n_large_effect = validator.required_sample_size(
            effect_size=0.8,
            test='two_sample'
        )

        # Smaller effect requires larger sample
        assert n_small_effect > n_large_effect

    def test_sample_size_scales_with_power(self, validator):
        """Test sample size increases with power"""
        n_80_power = validator.required_sample_size(
            effect_size=0.5,
            power=0.8
        )

        n_90_power = validator.required_sample_size(
            effect_size=0.5,
            power=0.9
        )

        # Higher power requires larger sample
        assert n_90_power > n_80_power

    def test_unknown_test_type_raises_error(self, validator):
        """Test unknown test type raises error"""
        with pytest.raises(ValueError):
            validator.required_sample_size(
                effect_size=0.5,
                test='unknown'
            )


# =============================================================================
# Validation Result Tests
# =============================================================================

class TestValidationResult:
    """Test ValidationResult data structure"""

    def test_validation_result_creation(self, validator):
        """Test creating validation result"""
        ci = ConfidenceInterval(
            lower=0.7,
            upper=0.8,
            level=0.95,
            method=CIType.PERCENTILE,
            width=0.1
        )

        conv = ConvergenceResult(
            converged=True,
            method=ConvergenceMethod.MOVING_WINDOW,
            iteration=100,
            confidence=0.95,
            details={}
        )

        result = ValidationResult(
            confidence_interval=ci,
            convergence=conv,
            sample_size=100
        )

        assert result.confidence_interval == ci
        assert result.convergence == conv
        assert result.sample_size == 100

    def test_is_confident_narrow_ci(self):
        """Test is_confident with narrow CI"""
        ci = ConfidenceInterval(
            lower=0.75,
            upper=0.77,
            level=0.95,
            method=CIType.PERCENTILE,
            width=0.02  # Narrow
        )

        conv = ConvergenceResult(
            converged=True,
            method=ConvergenceMethod.MOVING_WINDOW,
            iteration=100,
            confidence=0.95,
            details={}
        )

        result = ValidationResult(
            confidence_interval=ci,
            convergence=conv
        )

        assert result.is_confident()

    def test_is_confident_wide_ci(self):
        """Test is_confident with wide CI"""
        ci = ConfidenceInterval(
            lower=0.5,
            upper=1.0,
            level=0.95,
            method=CIType.PERCENTILE,
            width=0.5  # Wide
        )

        conv = ConvergenceResult(
            converged=True,
            method=ConvergenceMethod.MOVING_WINDOW,
            iteration=100,
            confidence=0.95,
            details={}
        )

        result = ValidationResult(
            confidence_interval=ci,
            convergence=conv
        )

        assert not result.is_confident()

    def test_is_confident_not_converged(self):
        """Test is_confident when not converged"""
        ci = ConfidenceInterval(
            lower=0.75,
            upper=0.77,
            level=0.95,
            method=CIType.PERCENTILE,
            width=0.02
        )

        conv = ConvergenceResult(
            converged=False,
            method=ConvergenceMethod.MOVING_WINDOW,
            iteration=50,
            confidence=0.5,
            details={}
        )

        result = ValidationResult(
            confidence_interval=ci,
            convergence=conv
        )

        assert not result.is_confident()

    def test_summary_string(self):
        """Test summary string generation"""
        ci = ConfidenceInterval(
            lower=0.7,
            upper=0.8,
            level=0.95,
            method=CIType.PERCENTILE,
            width=0.1
        )

        conv = ConvergenceResult(
            converged=True,
            method=ConvergenceMethod.MOVING_WINDOW,
            iteration=100,
            confidence=0.95,
            details={}
        )

        result = ValidationResult(
            confidence_interval=ci,
            convergence=conv
        )

        summary = result.summary()

        assert 'Confidence Interval' in summary
        assert 'Convergence' in summary
        assert 'Confident' in summary


# =============================================================================
# Complete Validation Tests
# =============================================================================

class TestCompleteValidation:
    """Test complete validation pipeline"""

    def test_validate_mcts_results_basic(self, validator):
        """Test basic MCTS results validation"""
        results = [0.75, 0.76, 0.74, 0.77, 0.75]
        value_history = [0.5, 0.6, 0.7, 0.74, 0.75]

        validation = validator.validate_mcts_results(
            results=results,
            value_history=value_history
        )

        assert validation.confidence_interval is not None
        assert validation.convergence is not None
        assert validation.sample_size is not None
        assert validation.diagnostics is not None

    def test_validate_with_comparison(self, validator):
        """Test validation with comparison results"""
        results_a = [0.75, 0.76, 0.74]
        results_b = [0.70, 0.71, 0.69]
        value_history = [0.5, 0.6, 0.7]

        validation = validator.validate_mcts_results(
            results=results_a,
            value_history=value_history,
            comparison_results=results_b
        )

        assert validation.significance is not None

    def test_diagnostics_calculation(self, validator):
        """Test diagnostic statistics"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        diagnostics = validator._calculate_diagnostics(data)

        assert 'n' in diagnostics
        assert 'mean' in diagnostics
        assert 'std' in diagnostics
        assert 'min' in diagnostics
        assert 'max' in diagnostics
        assert 'median' in diagnostics
        assert 'q25' in diagnostics
        assert 'q75' in diagnostics
        assert 'skewness' in diagnostics
        assert 'kurtosis' in diagnostics
        assert 'cv' in diagnostics

        assert diagnostics['n'] == 5
        assert diagnostics['mean'] == 3.0


# =============================================================================
# Sequential Analyzer Tests
# =============================================================================

class TestSequentialAnalyzer:
    """Test sequential analysis for adaptive stopping"""

    def test_initialization(self, validator):
        """Test sequential analyzer initialization"""
        analyzer = SequentialAnalyzer(validator)

        assert analyzer.validator == validator
        assert len(analyzer.results_history) == 0

    def test_should_continue_minimum_simulations(self, validator):
        """Test minimum simulations before checking"""
        analyzer = SequentialAnalyzer(validator)

        should_continue, reason = analyzer.should_continue(
            new_result=0.75,
            min_simulations=100
        )

        assert should_continue
        assert 'Minimum simulations not reached' in reason

    def test_should_continue_maximum_simulations(self, validator):
        """Test stop at maximum simulations"""
        analyzer = SequentialAnalyzer(validator)

        # Add results up to max
        for _ in range(100):
            analyzer.results_history.append(0.75)

        should_continue, reason = analyzer.should_continue(
            new_result=0.75,
            max_simulations=100
        )

        assert not should_continue
        assert 'Maximum simulations reached' in reason

    def test_should_continue_ci_width(self, validator):
        """Test stop when CI narrow enough"""
        analyzer = SequentialAnalyzer(validator)

        # Add stable results
        for _ in range(150):
            analyzer.results_history.append(0.75)

        should_continue, reason = analyzer.should_continue(
            new_result=0.75,
            min_simulations=100,
            target_ci_width=0.1
        )

        # Should stop due to narrow CI
        assert not should_continue
        assert 'CI narrow enough' in reason

    def test_should_continue_convergence(self, validator):
        """Test stop when converged"""
        analyzer = SequentialAnalyzer(validator)

        # Add converging results
        for i in range(150):
            value = 1.0 - np.exp(-i/50)
            analyzer.results_history.append(value)

        should_continue, reason = analyzer.should_continue(
            new_result=0.98,
            min_simulations=100
        )

        # Should stop due to convergence
        assert not should_continue
        assert 'Converged' in reason

    def test_should_continue_check_interval(self, validator):
        """Test check interval behavior"""
        analyzer = SequentialAnalyzer(validator)

        # Add 99 results
        for _ in range(99):
            analyzer.results_history.append(0.75)

        should_continue, reason = analyzer.should_continue(
            new_result=0.75,
            min_simulations=50,
            check_interval=100
        )

        # Should continue (waiting for interval)
        assert should_continue
        assert 'Waiting for next check interval' in reason


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_quick_validation(self):
        """Test quick_validation convenience function"""
        results = [0.75, 0.76, 0.74, 0.77]
        value_history = [0.5, 0.6, 0.7, 0.75]

        validation = quick_validation(results, value_history)

        assert validation.confidence_interval is not None
        assert validation.convergence is not None

    def test_compare_mcts_runs(self):
        """Test compare_mcts_runs convenience function"""
        np.random.seed(42)
        results_a = np.random.normal(0.75, 0.05, 50).tolist()
        results_b = np.random.normal(0.78, 0.05, 50).tolist()

        result = compare_mcts_runs(results_a, results_b)

        assert result.test_type == TestType.T_TEST
        assert isinstance(result.p_value, float)
        assert isinstance(result.significant, bool)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_single_value_data(self, validator):
        """Test with single value"""
        single_value = [0.75]

        ci = validator.bootstrap_confidence_interval(
            single_value,
            method=CIType.PERCENTILE
        )

        # Should work but CI will be degenerate
        assert ci.lower == ci.upper

    def test_identical_values(self, validator):
        """Test with all identical values"""
        identical_data = [0.75] * 100

        ci = validator.bootstrap_confidence_interval(
            identical_data,
            method=CIType.PERCENTILE
        )

        assert ci.lower == ci.upper

    def test_nan_values(self, validator):
        """Test handling of NaN values"""
        data_with_nan = [0.75, np.nan, 0.76, np.nan, 0.77]

        # Should filter or handle NaNs
        try:
            diagnostics = validator._calculate_diagnostics(data_with_nan)
            # If it doesn't raise, check results
            assert 'n' in diagnostics
        except (ValueError, RuntimeError):
            # Expected behavior for some implementations
            pass

    def test_infinite_values(self, validator):
        """Test handling of infinite values"""
        data_with_inf = [0.75, np.inf, 0.76, -np.inf, 0.77]

        try:
            diagnostics = validator._calculate_diagnostics(data_with_inf)
            # Check if handled
            assert 'n' in diagnostics
        except (ValueError, RuntimeError):
            # Expected for invalid data
            pass

    def test_very_large_data(self, validator):
        """Test with very large dataset"""
        large_data = np.random.randn(10000).tolist()

        ci = validator.bootstrap_confidence_interval(
            large_data,
            num_bootstrap=100
        )

        assert ci.lower < ci.upper

    def test_very_small_values(self, validator):
        """Test with very small values"""
        tiny_data = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]

        ci = validator.bootstrap_confidence_interval(
            tiny_data,
            method=CIType.PERCENTILE
        )

        assert ci.lower < ci.upper

    def test_very_large_values(self, validator):
        """Test with very large values"""
        huge_data = [1e10, 2e10, 3e10, 4e10, 5e10]

        ci = validator.bootstrap_confidence_interval(
            huge_data,
            method=CIType.PERCENTILE
        )

        assert ci.lower < ci.upper
