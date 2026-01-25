"""
Statistical Validation Module for RESE Phase III (Monte Carlo Refinement)

Provides bootstrap confidence intervals, significance testing, convergence
detection, and sample size determination for MCTS results.

Author: Agent D2 (Γ₂/Γ₃ Specialist)
Created: 2025-12-31
Status: 🟢 Active Implementation
Dependencies:
    - rese.phase3.mcts_search (Γ₂ - MCTS search)
    - numpy, scipy (statistical computing)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import numpy as np
from scipy import stats
from scipy.stats import norm
import math

# Try to import MCTS module
try:
    from phase3.mcts_search import MCTSNode, MCTSState, MCTSSearch
except ImportError:
    MCTSNode = None
    MCTSState = None
    MCTSSearch = None


class CIType(Enum):
    """Types of confidence intervals"""
    PERCENTILE = "percentile"  # Basic percentile method
    BCA = "bca"  # Bias-corrected and accelerated
    NORMAL = "normal"  # Normal approximation
    STUDENTIZED = "studentized"  # Bootstrap-t


class TestType(Enum):
    """Types of statistical tests"""
    T_TEST = "t_test"  # Paired t-test
    WILCOXON = "wilcoxon"  # Wilcoxon signed-rank (non-parametric)
    MANN_WHITNEY = "mann_whitney"  # Mann-Whitney U test


class ConvergenceMethod(Enum):
    """Methods for convergence detection"""
    MOVING_WINDOW = "moving_window"  # Variance in moving window
    GRADIENT = "gradient"  # Rate of improvement
    SPC = "spc"  # Statistical process control
    COMBINED = "combined"  # Combination of methods


@dataclass
class ValidationConfig:
    """Configuration for statistical validation"""
    # Bootstrap parameters
    num_bootstrap: int = 1000
    ci_type: CIType = CIType.BCA
    confidence_level: float = 0.95

    # Convergence detection
    convergence_method: ConvergenceMethod = ConvergenceMethod.COMBINED
    convergence_window: int = 20
    convergence_threshold: float = 0.001

    # Significance testing
    significance_level: float = 0.05
    test_type: TestType = TestType.T_TEST

    # Sample size analysis
    effect_size: float = 0.1
    power: float = 0.8

    # Reporting
    verbose: bool = False


@dataclass
class ConfidenceInterval:
    """Confidence interval result"""
    lower: float
    upper: float
    level: float
    method: CIType
    width: float

    def __str__(self):
        return f"{self.level*100:.1f}% CI [{self.lower:.4f}, {self.upper:.4f}] (width={self.width:.4f})"


@dataclass
class SignificanceTestResult:
    """Result of statistical significance test"""
    test_type: TestType
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    interpretation: str

    def __str__(self):
        sig_str = "Significant" if self.significant else "Not significant"
        return f"{sig_str} (p={self.p_value:.4f}, {self.test_type.value})"


@dataclass
class ConvergenceResult:
    """Result of convergence detection"""
    converged: bool
    method: ConvergenceMethod
    iteration: int
    confidence: float
    details: Dict[str, float]

    def __str__(self):
        status = "Converged" if self.converged else "Not converged"
        return f"{status} at iteration {self.iteration} ({self.method.value})"


@dataclass
class ValidationResult:
    """Complete validation result for MCTS output"""
    confidence_interval: ConfidenceInterval
    convergence: ConvergenceResult
    sample_size: Optional[int] = None
    significance: Optional[SignificanceTestResult] = None
    diagnostics: Dict = field(default_factory=dict)

    def is_confident(self) -> bool:
        """Check if results are confident (narrow CI + converged)"""
        ci_narrow = self.confidence_interval.width < 0.05
        converged = self.convergence.converged
        return ci_narrow and converged

    def summary(self) -> str:
        """Generate summary string"""
        lines = [
            "=== MCTS Validation Summary ===",
            f"Confidence Interval: {self.confidence_interval}",
            f"Convergence: {self.convergence}",
        ]

        if self.significance:
            lines.append(f"Significance Test: {self.significance}")

        if self.sample_size:
            lines.append(f"Recommended Sample Size: {self.sample_size}")

        lines.append(f"Confident: {self.is_confident()}")

        return "\n".join(lines)


class StatisticalValidator:
    """
    Statistical validation for MCTS results.

    Provides:
    1. Bootstrap confidence intervals (percentile, BCa)
    2. Significance testing (t-test, Wilcoxon)
    3. Convergence detection (multiple methods)
    4. Sample size determination (power analysis)
    """

    def __init__(self, config: ValidationConfig = None):
        """
        Initialize statistical validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()

    def validate_mcts_results(self,
                             results: List[float],
                             value_history: List[float] = None,
                             comparison_results: List[float] = None) -> ValidationResult:
        """
        Perform complete validation of MCTS results.

        Args:
            results: List of MCTS result values
            value_history: History of best values over iterations (for convergence)
            comparison_results: Optional comparison results for significance testing

        Returns:
            ValidationResult with all validation metrics
        """
        # 1. Calculate confidence interval
        ci = self.bootstrap_confidence_interval(results)

        # 2. Detect convergence
        if value_history is not None:
            conv_result = self.detect_convergence(value_history)
        else:
            conv_result = ConvergenceResult(
                converged=False,
                method=ConvergenceMethod.MOVING_WINDOW,
                iteration=0,
                confidence=0.0,
                details={}
            )

        # 3. Significance testing (if comparison provided)
        sig_result = None
        if comparison_results is not None:
            sig_result = self.significance_test(results, comparison_results)

        # 4. Sample size analysis
        sample_size = self.required_sample_size(
            effect_size=self.config.effect_size,
            alpha=self.config.significance_level,
            power=self.config.power
        )

        # 5. Diagnostics
        diagnostics = self._calculate_diagnostics(results)

        return ValidationResult(
            confidence_interval=ci,
            convergence=conv_result,
            sample_size=sample_size,
            significance=sig_result,
            diagnostics=diagnostics
        )

    def bootstrap_confidence_interval(self,
                                     data: List[float],
                                     num_bootstrap: int = None,
                                     confidence_level: float = None,
                                     method: CIType = None) -> ConfidenceInterval:
        """
        Calculate bootstrap confidence interval.

        Args:
            data: Sample data
            num_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            method: CI method (percentile, BCA, normal)

        Returns:
            ConfidenceInterval object
        """
        num_bootstrap = num_bootstrap or self.config.num_bootstrap
        confidence_level = confidence_level or self.config.confidence_level
        method = method or self.config.ci_type

        data = np.array(data)
        n = len(data)

        if n == 0:
            raise ValueError("Cannot compute CI on empty data")

        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(num_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)

        # Calculate CI based on method
        if method == CIType.PERCENTILE:
            lower, upper = self._percentile_ci(bootstrap_means, confidence_level)

        elif method == CIType.BCA:
            lower, upper = self._bca_ci(data, bootstrap_means, confidence_level)

        elif method == CIType.NORMAL:
            lower, upper = self._normal_ci(data, bootstrap_means, confidence_level)

        elif method == CIType.STUDENTIZED:
            lower, upper = self._studentized_ci(data, confidence_level)

        else:
            raise ValueError(f"Unknown CI method: {method}")

        width = upper - lower

        return ConfidenceInterval(
            lower=lower,
            upper=upper,
            level=confidence_level,
            method=method,
            width=width
        )

    def _percentile_ci(self, bootstrap_stats: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """Basic percentile method"""
        alpha = 1.0 - confidence_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        return lower, upper

    def _bca_ci(self, data: np.ndarray, bootstrap_stats: np.ndarray,
                confidence_level: float) -> Tuple[float, float]:
        """
        Bias-corrected and accelerated (BCa) confidence interval.

        Adjusts for:
        1. Bias (estimate not centered)
        2. Acceleration (variance changes with sample)

        More accurate than basic percentile method.
        """
        # Handle edge cases
        if len(data) < 2 or np.std(data, ddof=1) == 0:
            # Fall back to percentile method for degenerate cases
            return self._percentile_ci(bootstrap_stats, confidence_level)

        # Calculate bias correction
        theta_hat = np.mean(data)
        prop_less = np.sum(data < theta_hat) / len(data)

        # Avoid z0 = infinity
        prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
        z0 = norm.ppf(prop_less)

        # Calculate acceleration (jackknife)
        n = len(data)
        jackknife_means = []
        for i in range(n):
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_means.append(np.mean(jackknife_sample))

        jackknife_means = np.array(jackknife_means)
        jackknife_mean = np.mean(jackknife_means)

        # Acceleration factor
        num = np.sum((jackknife_mean - jackknife_means) ** 3)
        denom = 6.0 * (np.sum((jackknife_mean - jackknife_means) ** 2)) ** 1.5

        if denom == 0:
            a = 0.0
        else:
            a = num / denom

        # Adjusted percentiles
        alpha = 1.0 - confidence_level
        z_alpha = norm.ppf(alpha / 2)
        z_1alpha = norm.ppf(1 - alpha / 2)

        # Adjusted percentiles for BCa
        def adjust(z):
            return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

        alpha1 = adjust(z_alpha)
        alpha2 = adjust(z_1alpha)

        # Clip to valid range BEFORE multiplying by 100
        alpha1 = np.clip(alpha1, 0.001, 0.999)
        alpha2 = np.clip(alpha2, 0.001, 0.999)

        # Convert to percentiles (should be in range 0.1 to 99.9)
        lower_percentile = 100.0 * alpha1
        upper_percentile = 100.0 * alpha2

        # Final safety check to ensure percentiles are valid
        lower_percentile = np.clip(lower_percentile, 0.0, 100.0)
        upper_percentile = np.clip(upper_percentile, 0.0, 100.0)

        lower = np.percentile(bootstrap_stats, lower_percentile)
        upper = np.percentile(bootstrap_stats, upper_percentile)

        return lower, upper

    def _normal_ci(self, data: np.ndarray, bootstrap_stats: np.ndarray,
                   confidence_level: float) -> Tuple[float, float]:
        """Normal approximation confidence interval"""
        se = np.std(bootstrap_stats, ddof=1)
        mean = np.mean(data)

        z = norm.ppf(1 - (1 - confidence_level) / 2)

        lower = mean - z * se
        upper = mean + z * se

        return lower, upper

    def _studentized_ci(self, data: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """
        Studentized (Bootstrap-t) confidence interval.

        More accurate but computationally expensive.
        """
        n = len(data)
        theta_hat = np.mean(data)

        # Double bootstrap for studentized statistics
        t_statistics = []

        for _ in range(self.config.num_bootstrap):
            # First-level bootstrap
            sample = np.random.choice(data, size=n, replace=True)
            theta_star = np.mean(sample)

            # Second-level bootstrap for variance estimation
            variances = []
            for _ in range(100):  # Fewer iterations for inner bootstrap
                inner_sample = np.random.choice(sample, size=n, replace=True)
                variances.append(np.var(inner_sample, ddof=1))

            se_star = np.sqrt(np.mean(variances))

            if se_star > 0:
                t_star = (theta_star - theta_hat) / se_star
                t_statistics.append(t_star)

        t_statistics = np.array(t_statistics)

        # Percentiles of t-distribution
        alpha = 1.0 - confidence_level
        t_lower = np.percentile(t_statistics, 100 * alpha / 2)
        t_upper = np.percentile(t_statistics, 100 * (1 - alpha / 2))

        # Original SE
        se = np.std(data, ddof=1) / np.sqrt(n)

        # CI
        lower = theta_hat - t_upper * se
        upper = theta_hat - t_lower * se

        return lower, upper

    def significance_test(self,
                         results_a: List[float],
                         results_b: List[float],
                         test_type: TestType = None,
                         alpha: float = None) -> SignificanceTestResult:
        """
        Test if two sets of results differ significantly.

        Args:
            results_a: First set of results
            results_b: Second set of results
            test_type: Type of test to perform
            alpha: Significance level

        Returns:
            SignificanceTestResult
        """
        test_type = test_type or self.config.test_type
        alpha = alpha or self.config.significance_level

        results_a = np.array(results_a)
        results_b = np.array(results_b)

        if len(results_a) != len(results_b):
            print(f"[Warning] Unequal sample sizes: {len(results_a)} vs {len(results_b)}")

        if test_type == TestType.T_TEST:
            # Paired t-test
            statistic, p_value = stats.ttest_rel(results_a, results_b)
            interpretation = "Paired t-test"

        elif test_type == TestType.WILCOXON:
            # Wilcoxon signed-rank test (non-parametric)
            statistic, p_value = stats.wilcoxon(results_a, results_b)
            interpretation = "Wilcoxon signed-rank test"

        elif test_type == TestType.MANN_WHITNEY:
            # Mann-Whitney U test (independent samples)
            statistic, p_value = stats.mannwhitneyu(results_a, results_b,
                                                     alternative='two-sided')
            interpretation = "Mann-Whitney U test"

        else:
            raise ValueError(f"Unknown test type: {test_type}")

        significant = p_value < alpha

        if significant:
            interpretation += f": Significant difference (p < {alpha})"
        else:
            interpretation += f": No significant difference (p >= {alpha})"

        return SignificanceTestResult(
            test_type=test_type,
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            alpha=alpha,
            interpretation=interpretation
        )

    def detect_convergence(self,
                          value_history: List[float],
                          method: ConvergenceMethod = None,
                          window: int = None,
                          threshold: float = None) -> ConvergenceResult:
        """
        Detect convergence in value history.

        Args:
            value_history: History of best values over iterations
            method: Convergence detection method
            window: Window size for moving methods
            threshold: Convergence threshold

        Returns:
            ConvergenceResult
        """
        method = method or self.config.convergence_method
        window = window or self.config.convergence_window
        threshold = threshold or self.config.convergence_threshold

        if len(value_history) < window:
            return ConvergenceResult(
                converged=False,
                method=method,
                iteration=len(value_history),
                confidence=0.0,
                details={'reason': 'Insufficient data'}
            )

        if method == ConvergenceMethod.MOVING_WINDOW:
            return self._convergence_moving_window(value_history, window, threshold)

        elif method == ConvergenceMethod.GRADIENT:
            return self._convergence_gradient(value_history, window, threshold)

        elif method == ConvergenceMethod.SPC:
            return self._convergence_spc(value_history, window)

        elif method == ConvergenceMethod.COMBINED:
            # Combine multiple methods
            results = []
            results.append(self._convergence_moving_window(value_history, window, threshold))
            results.append(self._convergence_gradient(value_history, window, threshold))
            results.append(self._convergence_spc(value_history, window))

            # Converged if all methods agree
            converged_all = all(r.converged for r in results)

            # Average confidence
            confidence = np.mean([r.confidence for r in results])

            # Earliest convergence iteration
            iteration = max(r.iteration for r in results)

            return ConvergenceResult(
                converged=converged_all,
                method=method,
                iteration=iteration,
                confidence=confidence,
                details={
                    'moving_window': results[0].converged,
                    'gradient': results[1].converged,
                    'spc': results[2].converged
                }
            )

        else:
            raise ValueError(f"Unknown convergence method: {method}")

    def _convergence_moving_window(self, value_history: List[float],
                                   window: int, threshold: float) -> ConvergenceResult:
        """Detect convergence using moving window variance"""
        recent = value_history[-window:]
        rolling_std = np.std(recent)

        converged = rolling_std < threshold

        # Confidence based on how far below threshold
        confidence = min(1.0, threshold / (rolling_std + 1e-10))

        return ConvergenceResult(
            converged=converged,
            method=ConvergenceMethod.MOVING_WINDOW,
            iteration=len(value_history),
            confidence=confidence,
            details={'std': rolling_std, 'threshold': threshold}
        )

    def _convergence_gradient(self, value_history: List[float],
                             window: int, threshold: float) -> ConvergenceResult:
        """Detect convergence using gradient (rate of improvement)"""
        recent = value_history[-window:]

        # Calculate average gradient
        gradients = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        avg_gradient = np.mean(np.abs(gradients))

        converged = avg_gradient < threshold

        # Confidence based on gradient
        confidence = min(1.0, threshold / (avg_gradient + 1e-10))

        return ConvergenceResult(
            converged=converged,
            method=ConvergenceMethod.GRADIENT,
            iteration=len(value_history),
            confidence=confidence,
            details={'gradient': avg_gradient, 'threshold': threshold}
        )

    def _convergence_spc(self, value_history: List[float],
                        window: int) -> ConvergenceResult:
        """Detect convergence using statistical process control"""
        recent = value_history[-window:]

        mean = np.mean(recent)
        std = np.std(recent)

        # 3-sigma control limits
        upper_limit = mean + 3 * std
        lower_limit = mean - 3 * std

        # Check if all points within limits
        within_limits = all(lower_limit <= x <= upper_limit for x in recent)

        # Confidence based on distance from limits
        max_distance = max(abs(x - mean) for x in recent)
        confidence = min(1.0, 3 * std / (max_distance + 1e-10))

        return ConvergenceResult(
            converged=within_limits,
            method=ConvergenceMethod.SPC,
            iteration=len(value_history),
            confidence=confidence,
            details={
                'mean': mean,
                'std': std,
                'upper_limit': upper_limit,
                'lower_limit': lower_limit
            }
        )

    def required_sample_size(self,
                            effect_size: float,
                            alpha: float = None,
                            power: float = None,
                            test: str = 'two_sample') -> int:
        """
        Calculate required sample size for statistical power.

        Args:
            effect_size: Minimum detectable effect (Cohen's d or similar)
            alpha: Type I error rate (significance level)
            power: Statistical power (1 - Type II error rate)
            test: Type of test ('two_sample', 'paired', 'one_sample')

        Returns:
            Required sample size
        """
        alpha = alpha or self.config.significance_level
        power = power or self.config.power

        # Z-values
        z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed
        z_beta = norm.ppf(power)

        if test == 'two_sample':
            # Two-sample t-test
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        elif test == 'paired':
            # Paired t-test (more powerful)
            n = ((z_alpha + z_beta) / effect_size) ** 2

        elif test == 'one_sample':
            # One-sample t-test
            n = ((z_alpha + z_beta) / effect_size) ** 2

        else:
            raise ValueError(f"Unknown test type: {test}")

        return int(np.ceil(n))

    def _calculate_diagnostics(self, data: List[float]) -> Dict:
        """Calculate diagnostic statistics"""
        data = np.array(data)

        return {
            'n': len(data),
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'cv': np.std(data) / (np.mean(data) + 1e-10)  # Coefficient of variation
        }


class SequentialAnalyzer:
    """
    Sequential analysis for adaptive MCTS stopping.

    Stops search when:
    1. Confidence interval is narrow enough, OR
    2. Maximum iterations reached, OR
    3. Convergence detected
    """

    def __init__(self, validator: StatisticalValidator = None):
        self.validator = validator or StatisticalValidator()
        self.results_history: List[float] = []

    def should_continue(self,
                       new_result: float,
                       min_simulations: int = 100,
                       max_simulations: int = 10000,
                       target_ci_width: float = 0.05,
                       check_interval: int = 100) -> Tuple[bool, str]:
        """
        Check if search should continue.

        Args:
            new_result: Latest MCTS result value
            min_simulations: Minimum simulations before checking
            max_simulations: Maximum allowed simulations
            target_ci_width: Target CI width for stopping
            check_interval: Check every N simulations

        Returns:
            (should_continue, reason)
        """
        self.results_history.append(new_result)
        n = len(self.results_history)

        # Must run minimum simulations
        if n < min_simulations:
            return True, f"Minimum simulations not reached ({n}/{min_simulations})"

        # Check only at intervals
        if n % check_interval != 0:
            return True, f"Waiting for next check interval ({n % check_interval} remaining)"

        # Check maximum
        if n >= max_simulations:
            return False, f"Maximum simulations reached ({n})"

        # Check CI width
        ci = self.validator.bootstrap_confidence_interval(
            self.results_history,
            confidence_level=0.95
        )

        if ci.width < target_ci_width:
            return False, f"CI narrow enough ({ci.width:.4f} < {target_ci_width})"

        # Check convergence
        conv_result = self.validator.detect_convergence(self.results_history)
        if conv_result.converged:
            return False, f"Converged ({conv_result.method.value})"

        # Continue
        return True, f"Continue (CI width: {ci.width:.4f})"


# Convenience functions
def quick_validation(results: List[float],
                    value_history: List[float] = None,
                    confidence_level: float = 0.95) -> ValidationResult:
    """
    Convenience function for quick validation.

    Args:
        results: List of MCTS result values
        value_history: Optional history of best values
        confidence_level: Confidence level for CI (e.g., 0.95)

    Returns:
        ValidationResult
    """
    config = ValidationConfig(confidence_level=confidence_level)
    validator = StatisticalValidator(config)

    return validator.validate_mcts_results(results, value_history)


def compare_mcts_runs(results_a: List[float],
                      results_b: List[float],
                      alpha: float = 0.05) -> SignificanceTestResult:
    """
    Compare two MCTS runs for statistical significance.

    Args:
        results_a: First set of results
        results_b: Second set of results
        alpha: Significance level

    Returns:
        SignificanceTestResult
    """
    validator = StatisticalValidator()

    return validator.significance_test(results_a, results_b, alpha=alpha)


# Example usage (for testing)
if __name__ == "__main__":
    print("Statistical Validator Module - Ready")
    print("=" * 60)

    # Example: Validate MCTS results
    print("\nExample: Bootstrap confidence intervals")
    print("-" * 60)

    # Simulate MCTS results (normal distribution)
    np.random.seed(42)
    true_mean = 0.75
    results = np.random.normal(true_mean, 0.05, 100).tolist()

    validator = StatisticalValidator()

    # Calculate CI using different methods
    print("Confidence Intervals (95%):")
    for method in [CIType.PERCENTILE, CIType.BCA, CIType.NORMAL]:
        ci = validator.bootstrap_confidence_interval(results, method=method)
        print(f"  {method.value:12s}: {ci}")

    print(f"\nTrue mean: {true_mean:.4f}")
    print(f"Sample mean: {np.mean(results):.4f}")

    # Example: Convergence detection
    print("\nExample: Convergence detection")
    print("-" * 60)

    # Simulate convergence history
    value_history = []
    current = 0.5
    for i in range(100):
        improvement = 0.01 * np.exp(-i / 20)  # Decreasing improvements
        current += improvement + np.random.normal(0, 0.001)
        value_history.append(current)

    conv_result = validator.detect_convergence(value_history)

    print(f"Convergence detected: {conv_result.converged}")
    print(f"Method: {conv_result.method.value}")
    print(f"Iteration: {conv_result.iteration}")
    print(f"Confidence: {conv_result.confidence:.2f}")

    # Example: Significance testing
    print("\nExample: Significance testing")
    print("-" * 60)

    # Two sets of results (slightly different)
    results_a = np.random.normal(0.75, 0.05, 50).tolist()
    results_b = np.random.normal(0.77, 0.05, 50).tolist()

    sig_result = validator.significance_test(results_a, results_b)

    print(f"Mean A: {np.mean(results_a):.4f}")
    print(f"Mean B: {np.mean(results_b):.4f}")
    print(f"Test result: {sig_result}")

    # Example: Sample size calculation
    print("\nExample: Sample size determination")
    print("-" * 60)

    for effect_size in [0.1, 0.2, 0.5, 1.0]:
        n = validator.required_sample_size(
            effect_size=effect_size,
            alpha=0.05,
            power=0.8
        )
        print(f"Effect size {effect_size:.1f}: n = {n}")

    print("\n" + "=" * 60)
    print("Statistical Validator Module - Test Complete")
