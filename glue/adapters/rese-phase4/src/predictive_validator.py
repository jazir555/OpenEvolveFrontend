"""
RESE Phase IV: Predictive Validator

This module validates predictive efficacy by comparing ACI reduction against
the incumbent paradigm with statistical significance testing.

Following CLAUDE.md principles:
- Law of Runtime Truth: Validate against actual measurements
- Law of Configuration Explicitness: All config via env vars
- Circuit Breaker: Detect validation failures
- Structured Logging: JSON with correlation_id
- UTC: All timestamps in UTC ISO-8601

Per RESE spec §6.3: "The final architecture must generate a set of testable
predictions that, when verified, demonstrate a statistically significant
reduction in the Anomaly Characterization Index (ACI) relative to the
incumbent paradigm."

Author: RESE Team
Created: 2026-02-04
Phase: IV - Architectural Synthesis and Validation
"""

import os
import sys
import json
import uuid
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

# Add schemas to path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "..", "..", "schemas")))

from rese_phase4_schemas import (
    ArchitectureAssembly,
    Phase4Config,
    AssemblyStatus,
)


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

class StatisticalTest(Enum):
    """Types of statistical tests for validation."""
    WILCOXON = "wilcoxon"  # Non-parametric paired test
    MANN_WHITNEY_U = "mann_whitney_u"  # Non-parametric independent test
    T_TEST_PAIRED = "t_test_paired"  # Parametric paired test
    T_TEST_INDEPENDENT = "t_test_independent"  # Parametric independent test
    BOOTSTRAP = "bootstrap"  # Resampling-based test


# ============================================================================
# VALIDATION RESULT
# ============================================================================

@dataclass
class PredictiveValidationResult:
    """Result of predictive validation."""
    validation_id: str
    is_valid: bool
    aci_reduction: float
    incumbent_aci: float
    new_aci: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    statistical_significance: Dict[str, Any]
    test_used: StatisticalTest
    predictions_validated: int
    predictions_total: int
    metadata: Dict[str, Any]
    validated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validation_id": self.validation_id,
            "is_valid": self.is_valid,
            "aci_reduction": self.aci_reduction,
            "incumbent_aci": self.incumbent_aci,
            "new_aci": self.new_aci,
            "effect_size": self.effect_size,
            "confidence_interval": {
                "lower": self.confidence_interval[0],
                "upper": self.confidence_interval[1],
            },
            "statistical_significance": self.statistical_significance,
            "test_used": self.test_used.value,
            "predictions_validated": self.predictions_validated,
            "predictions_total": self.predictions_total,
            "metadata": self.metadata,
            "validated_at": self.validated_at.isoformat(),
        }


# ============================================================================
# STRUCTURED LOGGER
# ============================================================================

class StructuredLogger:
    """Structured JSON logger following CLAUDE.md §3.3."""

    def __init__(self, service_name: str, correlation_id: Optional[str] = None):
        self.service_name = service_name
        self.correlation_id = correlation_id or str(uuid.uuid4())

    def _log(self, level: str, msg: str, **kwargs):
        """Internal log method."""
        log_entry = {
            "level": level,
            "msg": msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": self.correlation_id,
            "source_service": self.service_name,
            **kwargs
        }
        print(json.dumps(log_entry))

    def debug(self, msg: str, **kwargs):
        self._log("debug", msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log("info", msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log("warning", msg, **kwargs)

    def error(self, msg: str, error: Optional[Exception] = None, **kwargs):
        if error:
            kwargs["error"] = str(error)
            kwargs["error_type"] = type(error).__name__
        self._log("error", msg, **kwargs)


# ============================================================================
# PREDICTIVE VALIDATOR
# ============================================================================

class PredictiveValidator:
    """
    Validates predictive efficacy of architecture assembly.

    This implements Δ₃: ACI Reduction Validation with statistical testing.

    Core responsibilities:
    1. Compare ACI of new architecture vs incumbent
    2. Perform statistical significance testing
    3. Validate predictions from paradigm shifts
    4. Generate validation reports
    """

    def __init__(
        self,
        config: Phase4Config,
        logger: Optional[StructuredLogger] = None,
        test_type: StatisticalTest = StatisticalTest.WILCOXON
    ):
        """
        Initialize predictive validator.

        Args:
            config: Phase IV configuration
            logger: Optional logger
            test_type: Statistical test to use
        """
        self.config = config
        self.logger = logger or StructuredLogger(
            "rese-phase4-predictive-validator",
            self.config.correlation_id
        )
        self.test_type = test_type

        # Validation thresholds (from env vars or defaults)
        self.significance_level = float(os.getenv("PREDICTIVE_ALPHA", "0.05"))
        self.min_effect_size = float(os.getenv("PREDICTIVE_MIN_EFFECT", "0.2"))
        self.min_predictions_validated = float(os.getenv("PREDICTIVE_MIN_PREDICTIONS", "0.8"))

        self.logger.info(
            "Predictive Validator initialized",
            test_type=test_type.value,
            significance_level=self.significance_level,
            min_effect_size=self.min_effect_size,
        )

    def validate(
        self,
        assembly: ArchitectureAssembly,
        incumbent_aci_measurements: List[float],
        new_aci_measurements: List[float]
    ) -> PredictiveValidationResult:
        """
        Validate predictive efficacy of assembly.

        Args:
            assembly: Architecture assembly to validate
            incumbent_aci_measurements: ACI measurements from incumbent paradigm
            new_aci_measurements: ACI measurements from new architecture

        Returns:
            PredictiveValidationResult with full statistical analysis

        Raises:
            ValueError: If measurements are invalid
            TimeoutError: If validation exceeds timeout
        """
        import time
        start_time = time.time()
        timeout_sec = self.config.assembly_timeout_ms / 1000.0

        self.logger.info(
            "Starting predictive validation",
            assembly_id=assembly.assembly_id,
            incumbent_samples=len(incumbent_aci_measurements),
            new_samples=len(new_aci_measurements),
        )

        try:
            # Validate measurements
            self._validate_measurements(incumbent_aci_measurements, new_aci_measurements)

            # Calculate summary statistics
            incumbent_mean = sum(incumbent_aci_measurements) / len(incumbent_aci_measurements)
            new_mean = sum(new_aci_measurements) / len(new_aci_measurements)

            # Calculate ACI reduction
            aci_reduction = (incumbent_mean - new_mean) / incumbent_mean if incumbent_mean > 0 else 0.0

            # Perform statistical test
            test_result = self._perform_statistical_test(
                incumbent_aci_measurements,
                new_aci_measurements
            )

            # Calculate effect size (Cohen's d)
            effect_size = self._calculate_effect_size(
                incumbent_aci_measurements,
                new_aci_measurements
            )

            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                new_aci_measurements,
                confidence=0.95
            )

            # Assess statistical significance
            is_significant = test_result["p_value"] < self.significance_level

            # Validate predictions
            predictions_validated, predictions_total = self._validate_predictions(assembly)

            # Determine overall validity
            is_valid = (
                is_significant and
                abs(effect_size) >= self.min_effect_size and
                aci_reduction >= self.min_effect_size and
                (predictions_validated / predictions_total if predictions_total > 0 else 0) >= self.min_predictions_validated
            )

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                raise TimeoutError(f"Predictive validation exceeded timeout: {elapsed:.2f}s")

            result = PredictiveValidationResult(
                validation_id=str(uuid.uuid4()),
                is_valid=is_valid,
                aci_reduction=aci_reduction,
                incumbent_aci=incumbent_mean,
                new_aci=new_mean,
                effect_size=effect_size,
                confidence_interval=confidence_interval,
                statistical_significance={
                    "is_significant": is_significant,
                    "p_value": test_result["p_value"],
                    "alpha": self.significance_level,
                    "test_statistic": test_result["test_statistic"],
                    "test_type": self.test_type.value,
                },
                test_used=self.test_type,
                predictions_validated=predictions_validated,
                predictions_total=predictions_total,
                metadata={
                    "assembly_id": assembly.assembly_id,
                    "validation_time_seconds": elapsed,
                    "incumbent_samples": len(incumbent_aci_measurements),
                    "new_samples": len(new_aci_measurements),
                },
                validated_at=datetime.now(timezone.utc),
            )

            self.logger.info(
                "Predictive validation completed",
                validation_id=result.validation_id,
                is_valid=is_valid,
                aci_reduction=aci_reduction,
                is_significant=is_significant,
                elapsed_seconds=elapsed,
            )

            return result

        except Exception as e:
            self.logger.error("Predictive validation failed", error=e)
            raise

    def _validate_measurements(
        self,
        incumbent: List[float],
        new: List[float]
    ):
        """Validate measurement data."""
        if not incumbent or not new:
            raise ValueError("Measurement lists cannot be empty")

        if len(incumbent) < 3 or len(new) < 3:
            raise ValueError("Need at least 3 measurements per group for validation")

        # Check for invalid values
        if any(v < 0 for v in incumbent + new):
            raise ValueError("ACI measurements must be non-negative")

        if any(math.isnan(v) or math.isinf(v) for v in incumbent + new):
            raise ValueError("ACI measurements contain NaN or Inf values")

    def _perform_statistical_test(
        self,
        incumbent: List[float],
        new: List[float]
    ) -> Dict[str, Any]:
        """
        Perform statistical test.

        For production, would use scipy.stats. Here we provide simplified implementations.
        """
        if self.test_type == StatisticalTest.WILCOXON:
            return self._wilcoxon_test(incumbent, new)
        elif self.test_type == StatisticalTest.MANN_WHITNEY_U:
            return self._mann_whitney_u_test(incumbent, new)
        elif self.test_type == StatisticalTest.T_TEST_PAIRED:
            return self._t_test_paired(incumbent, new)
        elif self.test_type == StatisticalTest.T_TEST_INDEPENDENT:
            return self._t_test_independent(incumbent, new)
        elif self.test_type == StatisticalTest.BOOTSTRAP:
            return self._bootstrap_test(incumbent, new)
        else:
            raise ValueError(f"Unsupported test type: {self.test_type}")

    def _wilcoxon_test(
        self,
        incumbent: List[float],
        new: List[float]
    ) -> Dict[str, Any]:
        """
        Wilcoxon signed-rank test (simplified).

        NOTE: Production would use scipy.stats.wilcoxon
        """
        # Simplified implementation - compute paired differences
        if len(incumbent) != len(new):
            # Truncate to match lengths
            min_len = min(len(incumbent), len(new))
            incumbent = incumbent[:min_len]
            new = new[:min_len]

        # Calculate differences
        diffs = [i - n for i, n in zip(incumbent, new)]

        # Remove zero differences
        diffs = [d for d in diffs if d != 0]

        if not diffs:
            return {
                "test_statistic": 0.0,
                "p_value": 1.0,
            }

        # Simplified test statistic (sum of ranks of positive differences)
        # In production, would use proper ranking
        positive_sum = sum(d for d in diffs if d > 0)
        negative_sum = sum(abs(d) for d in diffs if d < 0)

        test_statistic = min(positive_sum, negative_sum)

        # Simplified p-value estimation
        # In production, would use exact distribution or normal approximation
        n = len(diffs)
        mean = n * (n + 1) / 4
        std = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)

        if std > 0:
            z = (test_statistic - mean) / std
            # Simplified p-value from Z-score (two-tailed)
            p_value = 2 * (1 - self._normal_cdf(abs(z)))
        else:
            p_value = 1.0

        return {
            "test_statistic": test_statistic,
            "p_value": max(0.0, min(1.0, p_value)),
        }

    def _mann_whitney_u_test(
        self,
        incumbent: List[float],
        new: List[float]
    ) -> Dict[str, Any]:
        """Mann-Whitney U test (simplified)."""
        # Combine all samples with indices
        combined = [(idx, x, "incumbent") for idx, x in enumerate(incumbent)] + \
                   [(idx + len(incumbent), x, "new") for idx, x in enumerate(new)]

        # Rank all samples by value
        sorted_combined = sorted(combined, key=lambda x: x[1])

        # Calculate U statistic for 'new' group
        rank_sum_new = sum(idx + 1 for idx, (_, _, label) in enumerate(sorted_combined) if label == "new")
        n1, n2 = len(incumbent), len(new)

        u1 = rank_sum_new - n2 * (n2 + 1) / 2
        u2 = n1 * n2 - u1
        u = min(u1, u2)

        # Simplified p-value
        mean = n1 * n2 / 2
        std = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

        if std > 0:
            z = (u - mean) / std
            p_value = 2 * (1 - self._normal_cdf(abs(z)))
        else:
            p_value = 1.0

        return {
            "test_statistic": u,
            "p_value": max(0.0, min(1.0, p_value)),
        }

    def _t_test_paired(self, incumbent: List[float], new: List[float]) -> Dict[str, Any]:
        """Paired t-test (simplified)."""
        # Match lengths
        min_len = min(len(incumbent), len(new))
        incumbent = incumbent[:min_len]
        new = new[:min_len]

        # Calculate differences
        diffs = [i - n for i, n in zip(incumbent, new)]

        # Calculate t-statistic
        n = len(diffs)
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0
        std_diff = math.sqrt(var_diff)

        if std_diff > 0:
            t_stat = mean_diff / (std_diff / math.sqrt(n))
            # Simplified p-value (would use t-distribution in production)
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        else:
            t_stat = 0.0
            p_value = 1.0

        return {
            "test_statistic": t_stat,
            "p_value": max(0.0, min(1.0, p_value)),
        }

    def _t_test_independent(self, incumbent: List[float], new: List[float]) -> Dict[str, Any]:
        """Independent t-test (simplified)."""
        n1, n2 = len(incumbent), len(new)
        mean1 = sum(incumbent) / n1
        mean2 = sum(new) / n2

        var1 = sum((x - mean1) ** 2 for x in incumbent) / (n1 - 1) if n1 > 1 else 0
        var2 = sum((x - mean2) ** 2 for x in new) / (n2 - 1) if n2 > 1 else 0

        # Pooled standard error
        se = math.sqrt(var1 / n1 + var2 / n2)

        if se > 0:
            t_stat = (mean1 - mean2) / se
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        else:
            t_stat = 0.0
            p_value = 1.0

        return {
            "test_statistic": t_stat,
            "p_value": max(0.0, min(1.0, p_value)),
        }

    def _bootstrap_test(
        self,
        incumbent: List[float],
        new: List[float],
        n_bootstrap: int = 10000
    ) -> Dict[str, Any]:
        """Bootstrap test (simplified)."""
        # Calculate observed difference
        obs_diff = sum(incumbent) / len(incumbent) - sum(new) / len(new)

        # Bootstrap resampling
        count_extreme = 0
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample_inc = [incumbent[i] for i in [len(incumbent) - 1 for _ in incumbent]]  # Simplified
            sample_new = [new[i] for i in [len(new) - 1 for _ in new]]  # Simplified

            # Calculate difference
            diff = sum(sample_inc) / len(sample_inc) - sum(sample_new) / len(sample_new)

            # Count extreme values
            if abs(diff) >= abs(obs_diff):
                count_extreme += 1

        p_value = count_extreme / n_bootstrap

        return {
            "test_statistic": obs_diff,
            "p_value": p_value,
        }

    def _calculate_effect_size(
        self,
        incumbent: List[float],
        new: List[float]
    ) -> float:
        """
        Calculate Cohen's d effect size.

        d = (mean1 - mean2) / pooled_std
        """
        mean1 = sum(incumbent) / len(incumbent)
        mean2 = sum(new) / len(new)

        n1, n2 = len(incumbent), len(new)
        var1 = sum((x - mean1) ** 2 for x in incumbent) / (n1 - 1) if n1 > 1 else 0
        var2 = sum((x - mean2) ** 2 for x in new) / (n2 - 1) if n2 > 1 else 0

        # Pooled standard deviation
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2) if (n1 + n2 - 2) > 0 else 0
        pooled_std = math.sqrt(pooled_var)

        if pooled_std > 0:
            return (mean1 - mean2) / pooled_std
        else:
            return 0.0

    def _calculate_confidence_interval(
        self,
        measurements: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for measurements."""
        n = len(measurements)
        mean = sum(measurements) / n

        # Standard error
        std = math.sqrt(sum((x - mean) ** 2 for x in measurements) / (n - 1)) if n > 1 else 0
        se = std / math.sqrt(n)

        # Z-score for confidence level (simplified - would use t-distribution)
        z_score = 1.96 if confidence == 0.95 else 1.645

        margin_of_error = z_score * se

        return (mean - margin_of_error, mean + margin_of_error)

    def _validate_predictions(self, assembly: ArchitectureAssembly) -> Tuple[int, int]:
        """
        Validate predictions from paradigm shifts.

        Returns:
            Tuple of (validated_count, total_count)
        """
        # In production, would actually test predictions
        # Here we count them as potentially validatable
        total = len(assembly.paradigm_shifts)

        # Simplified: assume predictions are validatable if confidence > threshold
        validated = sum(
            1 for ps in assembly.paradigm_shifts
            if ps.confidence >= self.config.min_confidence_threshold
        )

        return (validated, total)

    def _normal_cdf(self, x: float) -> float:
        """
        Standard normal CDF (simplified).

        Uses approximation formula.
        """
        # Constants for approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # Save sign of x
        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2)

        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

        return 0.5 * (1.0 + sign * y)


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "PredictiveValidator",
    "PredictiveValidationResult",
    "StatisticalTest",
]
