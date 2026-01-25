"""
Statistical Testing Framework for Δ₃
======================================

Implements comprehensive statistical testing for ACI reduction validation.

This module provides:
- T-tests (paired, two-sample)
- Wilcoxon signed-rank test
- Bootstrap confidence intervals
- Effect size calculation (Cohen's d)
- Multiple testing corrections

Author: Agent E3 (Δ₃ Specialist)
Date: 2025-12-31
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from scipy import stats

from .types import ACIMeasurement

if TYPE_CHECKING:
    from .aci_reduction_validator import (
        Delta3Config,
        StatisticalTestResults,
        EffectSizeMetrics,
        ConfidenceIntervalMetrics,
        EffectSizeMagnitude
    )


# =============================================================================
# STATISTICAL TEST RUNNER
# =============================================================================

class StatisticalTestRunner:
    """
    Run statistical tests for ACI reduction validation.

    Implements paired t-test, Wilcoxon signed-rank, effect size calculation,
    and bootstrap confidence intervals.
    """

    def __init__(self, config: 'Delta3Config'):
        """
        Initialize statistical test runner.

        Args:
            config: Δ₃ configuration
        """
        self.config = config

    def test(
        self,
        aci_baseline: ACIMeasurement,
        aci_final: ACIMeasurement
    ) -> 'StatisticalTestResults':
        """
        Perform statistical test for ACI reduction.

        Strategy: Paired t-test with normality check
        Falls back to Wilcoxon signed-rank if data not normal

        Args:
            aci_baseline: Baseline ACI measurement
            aci_final: Final ACI measurement

        Returns:
            StatisticalTestResults
        """
        # Extract values (handle both scalar and array)
        baseline_vals = self._extract_values(aci_baseline.aci_value)
        final_vals = self._extract_values(aci_final.aci_value)

        # Check normality
        normal_baseline = self._check_normality(baseline_vals)
        normal_final = self._check_normality(final_vals)

        # Choose test based on normality and sample size
        if (normal_baseline and normal_final) or len(baseline_vals) >= 30:
            # Use paired t-test
            return self._paired_t_test(baseline_vals, final_vals)
        else:
            # Use Wilcoxon signed-rank test
            return self._wilcoxon_test(baseline_vals, final_vals)

    def _paired_t_test(
        self,
        baseline: np.ndarray,
        final: np.ndarray
    ) -> 'StatisticalTestResults':
        """
        Perform paired t-test.

        Tests if final ACI is significantly lower than baseline.

        Args:
            baseline: Baseline ACI values
            final: Final ACI values

        Returns:
            StatisticalTestResults
        """
        # Ensure equal length
        min_len = min(len(baseline), len(final))
        baseline = baseline[:min_len]
        final = final[:min_len]

        # Paired t-test
        t_statistic, p_value = stats.ttest_rel(final, baseline)

        # Degrees of freedom
        df = len(baseline) - 1

        # Critical value (two-tailed)
        alpha = self.config.significance_level
        critical_value = stats.t.ppf(1 - alpha/2, df)

        # Check if significant (final < baseline for reduction)
        is_significant = (p_value < alpha) and (np.mean(final) < np.mean(baseline))

        return StatisticalTestResults(
            test_used="paired_t_test",
            p_value=float(p_value),
            is_significant=is_significant,
            test_statistic=float(t_statistic),
            degrees_of_freedom=df,
            critical_value=float(critical_value)
        )

    def _wilcoxon_test(
        self,
        baseline: np.ndarray,
        final: np.ndarray
    ) -> 'StatisticalTestResults':
        """
        Perform Wilcoxon signed-rank test.

        Non-parametric alternative to paired t-test.

        Args:
            baseline: Baseline ACI values
            final: Final ACI values

        Returns:
            StatisticalTestResults
        """
        # Ensure equal length
        min_len = min(len(baseline), len(final))
        baseline = baseline[:min_len]
        final = final[:min_len]

        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(final, baseline)

        # Approximate degrees of freedom
        df = len(baseline) - 1

        # Critical value (approximate using normal distribution)
        alpha = self.config.significance_level
        critical_value = stats.norm.ppf(1 - alpha/2)

        # Check if significant
        is_significant = (p_value < alpha) and (np.mean(final) < np.mean(baseline))

        return StatisticalTestResults(
            test_used="wilcoxon_signed_rank",
            p_value=float(p_value),
            is_significant=is_significant,
            test_statistic=float(statistic),
            degrees_of_freedom=df,
            critical_value=float(critical_value)
        )

    def calculate_effect_size(
        self,
        aci_baseline: ACIMeasurement,
        aci_final: ACIMeasurement
    ) -> 'EffectSizeMetrics':
        """
        Calculate Cohen's d and other effect sizes.

        Args:
            aci_baseline: Baseline ACI measurement
            aci_final: Final ACI measurement

        Returns:
            EffectSizeMetrics
        """
        baseline_vals = self._extract_values(aci_baseline.aci_value)
        final_vals = self._extract_values(aci_final.aci_value)

        # Cohen's d
        mean_diff = np.mean(final_vals) - np.mean(baseline_vals)
        pooled_std = self._pooled_std(baseline_vals, final_vals)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        # Magnitude classification
        if abs(cohens_d) < 0.2:
            magnitude = EffectSizeMagnitude.NEGLIGIBLE
        elif abs(cohens_d) < 0.5:
            magnitude = EffectSizeMagnitude.SMALL
        elif abs(cohens_d) < 0.8:
            magnitude = EffectSizeMagnitude.MEDIUM
        elif abs(cohens_d) < 1.2:
            magnitude = EffectSizeMagnitude.LARGE
        else:
            magnitude = EffectSizeMagnitude.VERY_LARGE

        meets_threshold = abs(cohens_d) >= self.config.min_effect_size

        # Pearson's r (if paired)
        pearsons_r = None
        r_squared = None
        if len(baseline_vals) == len(final_vals) and len(baseline_vals) > 1:
            try:
                pearsons_r = float(np.corrcoef(baseline_vals, final_vals)[0, 1])
                r_squared = pearsons_r ** 2
            except:
                pass  # Correlation calculation failed

        return EffectSizeMetrics(
            cohens_d=float(cohens_d),
            magnitude=magnitude,
            meets_threshold=meets_threshold,
            pearsons_r=pearsons_r,
            r_squared=r_squared
        )

    def calculate_ci(
        self,
        aci_baseline: ACIMeasurement,
        aci_final: ACIMeasurement
    ) -> 'ConfidenceIntervalMetrics':
        """
        Calculate bootstrap confidence interval for ACI reduction.

        Args:
            aci_baseline: Baseline ACI measurement
            aci_final: Final ACI measurement

        Returns:
            ConfidenceIntervalMetrics
        """
        baseline_vals = self._extract_values(aci_baseline.aci_value)
        final_vals = self._extract_values(aci_final.aci_value)

        # Bootstrap
        n_bootstrap = self.config.bootstrap_iterations
        reductions = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            boot_baseline = np.random.choice(baseline_vals, size=len(baseline_vals), replace=True)
            boot_final = np.random.choice(final_vals, size=len(final_vals), replace=True)

            # Calculate reduction
            reduction = np.mean(boot_baseline) - np.mean(boot_final)
            reductions.append(reduction)

        reductions = np.array(reductions)

        # Percentiles
        alpha = 1 - self.config.confidence_level
        lower = np.percentile(reductions, alpha/2 * 100)
        upper = np.percentile(reductions, (1 - alpha/2) * 100)

        excludes_zero = (lower > 0) or (upper < 0)
        width = upper - lower

        return ConfidenceIntervalMetrics(
            ci_level=self.config.confidence_level,
            lower_bound=float(lower),
            upper_bound=float(upper),
            excludes_zero=excludes_zero,
            width=float(width),
            method="bootstrap"
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _extract_values(self, value: any) -> np.ndarray:
        """
        Extract numpy array from value.

        Handles scalars, lists, tuples, and numpy arrays.

        Args:
            value: Input value

        Returns:
            Numpy array
        """
        if isinstance(value, (list, tuple)):
            return np.array(value)
        elif isinstance(value, np.ndarray):
            return value
        else:
            # Scalar: return as single-element array
            return np.array([value])

    def _check_normality(self, data: np.ndarray, alpha: float = 0.05) -> bool:
        """
        Check if data is normally distributed using Shapiro-Wilk test.

        Args:
            data: Data to test
            alpha: Significance level

        Returns:
            True if data appears normally distributed
        """
        if len(data) < 3:
            return True  # Assume normal for very small samples

        try:
            _, p_value = stats.shapiro(data)
            return p_value > alpha
        except:
            return True  # Fallback: assume normal if test fails

    def _pooled_std(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate pooled standard deviation.

        Args:
            a: First array
            b: Second array

        Returns:
            Pooled standard deviation
        """
        n_a, n_b = len(a), len(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

        if n_a + n_b - 2 == 0:
            return 1.0  # Avoid division by zero

        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        return np.sqrt(max(0, pooled_var))  # Ensure non-negative


# =============================================================================
# ADDITIONAL STATISTICAL FUNCTIONS
# =============================================================================

def two_sample_t_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Perform two-sample t-test (independent samples).

    Args:
        sample1: First sample
        sample2: Second sample
        alpha: Significance level

    Returns:
        Tuple of (t_statistic, p_value, is_significant)
    """
    t_statistic, p_value = stats.ttest_ind(sample1, sample2)
    is_significant = p_value < alpha

    return float(t_statistic), float(p_value), is_significant


def mann_whitney_u_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).

    Args:
        sample1: First sample
        sample2: Second sample
        alpha: Significance level

    Returns:
        Tuple of (u_statistic, p_value, is_significant)
    """
    u_statistic, p_value = stats.mannwhitneyu(sample1, sample2)
    is_significant = p_value < alpha

    return float(u_statistic), float(p_value), is_significant


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Bonferroni correction for multiple testing.

    Args:
        p_values: List of p-values
        alpha: Original significance level

    Returns:
        List of booleans indicating which tests are significant after correction
    """
    corrected_alpha = alpha / len(p_values)
    return [p < corrected_alpha for p in p_values]


def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Benjamini-Hochberg FDR correction for multiple testing.

    Args:
        p_values: List of p-values
        alpha: False discovery rate

    Returns:
        List of booleans indicating which tests are significant after correction
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]

    # Find largest k such that p_k <= (k/n) * alpha
    significant = []
    for k, p_val in enumerate(sorted_p_values):
        threshold = (k + 1) / n * alpha
        significant.append(p_val <= threshold)

    # Reorder to original order
    reordered = [False] * n
    for idx, sig in zip(sorted_indices, significant):
        reordered[idx] = sig

    return reordered


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StatisticalTestRunner',
    'two_sample_t_test',
    'mann_whitney_u_test',
    'bonferroni_correction',
    'benjamini_hochberg_correction',
]
