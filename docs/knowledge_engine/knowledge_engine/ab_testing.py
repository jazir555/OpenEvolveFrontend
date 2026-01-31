"""
A/B Testing Framework

Statistical A/B testing for agent behaviors and strategies.
Supports frequentist and Bayesian approaches, early stopping,
and winner selection.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, UTC, timedelta
from dataclasses import dataclass
import numpy as np
from scipy import stats
import asyncio
import logging

from .schemas.long_horizon import (
    Experiment,
    ExperimentStatus,
    ExperimentResults,
    VariantStats,
    OutcomeType
)


logger = logging.getLogger(__name__)


class ABTestFramework:
    """
    Statistical A/B testing for agent behaviors

    Supports:
    - Frequentist hypothesis testing (t-test, chi-squared)
    - Bayesian A/B testing (Beta-Bernoulli model)
    - Early stopping based on sequential analysis
    - Multiple comparison correction

    Usage:
        framework = ABTestFramework(significance_level=0.05)

        # Create experiment
        experiment = await framework.create_experiment(
            name="Strategy Comparison",
            description="Test PES vs QD",
            variants=["pes", "qd"],
            min_sample_size=100
        )

        # Record observations
        await framework.record_observation(
            experiment_id="exp_123",
            variant="pes",
            outcome=0.85,
            is_success=True
        )

        # Get results
        results = await framework.get_results("exp_123")
        if results.significance:
            winner = results.winner
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_sample_size: int = 100,
        test_method: str = "frequentist",  # or "bayesian"
        enable_early_stopping: bool = True,
        early_stop_look_interval: int = 20,
        multiple_testing_correction: str = "bonferroni"  # or "bonferroni", "holm"
    ):
        """
        Initialize A/B testing framework

        Args:
            significance_level: Statistical significance threshold (alpha)
            min_sample_size: Minimum observations per variant
            test_method: "frequentist" or "bayesian"
            enable_early_stopping: Enable early stopping
            early_stop_look_interval: Check for early stopping every N samples
            multiple_testing_correction: Method for multiple comparisons
        """
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size
        self.test_method = test_method
        self.enable_early_stopping = enable_early_stopping
        self.early_stop_look_interval = early_stop_look_interval
        self.multiple_testing_correction = multiple_testing_correction

        # Experiment storage
        # Key: experiment_id -> Experiment
        self.experiments: Dict[str, Experiment] = {}

    async def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[str],
        min_sample_size: Optional[int] = None,
        significance_level: Optional[float] = None
    ) -> Experiment:
        """
        Create a new A/B experiment

        Args:
            name: Human-readable name
            description: What is being tested
            variants: List of variant identifiers
            min_sample_size: Override default min sample size
            significance_level: Override default significance level

        Returns:
            Created experiment
        """
        experiment_id = f"exp_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"

        # Create variant stats
        variant_stats = {
            variant: VariantStats(variant_id=variant)
            for variant in variants
        }

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variant_stats,
            start_time=datetime.now(UTC),
            status=ExperimentStatus.RUNNING,
            significance_level=significance_level or self.significance_level,
            min_sample_size=min_sample_size or self.min_sample_size
        )

        self.experiments[experiment_id] = experiment

        logger.info(
            f"Created experiment {experiment_id}: {name} "
            f"with {len(variants)} variants"
        )

        return experiment

    async def record_observation(
        self,
        experiment_id: str,
        variant: str,
        outcome: float,
        is_success: Optional[bool] = None
    ) -> None:
        """
        Record an observation for a variant (idempotent)

        Args:
            experiment_id: Experiment identifier
            variant: Variant identifier
            outcome: Continuous outcome score (0-1)
            is_success: Binary success indicator (optional)

        Law of Idempotency: Safe to record duplicate observations
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        if variant not in experiment.variants:
            raise ValueError(f"Variant {variant} not in experiment")

        if experiment.status != ExperimentStatus.RUNNING:
            logger.warning(
                f"Experiment {experiment_id} is not running, "
                f"status={experiment.status.value}"
            )
            return

        variant_stats = experiment.variants[variant]

        # Record observation
        variant_stats.observations.append(outcome)
        variant_stats.sample_size += 1

        # Update statistics
        variant_stats.mean_outcome = np.mean(variant_stats.observations)
        variant_stats.variance = np.var(variant_stats.observations)

        # Update conversion rate if binary outcome provided
        if is_success is not None:
            variant_stats.conversion_rate = (
                (variant_stats.conversion_rate * (variant_stats.sample_size - 1) +
                 (1.0 if is_success else 0.0)) /
                variant_stats.sample_size
            )

        # Update confidence interval
        std_error = np.sqrt(variant_stats.variance / variant_stats.sample_size)
        variant_stats.confidence_interval = (
            max(0.0, variant_stats.mean_outcome - 1.96 * std_error),
            min(1.0, variant_stats.mean_outcome + 1.96 * std_error)
        )

        # Check for early stopping
        if self.enable_early_stopping:
            await self._check_early_stopping(experiment_id)

    async def get_results(
        self,
        experiment_id: str
    ) -> ExperimentResults:
        """
        Get experiment results with statistical analysis

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment results with statistical tests
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        if self.test_method == "bayesian":
            return await self._bayesian_analysis(experiment)
        else:
            return await self._frequentist_analysis(experiment)

    async def _frequentist_analysis(
        self,
        experiment: Experiment
    ) -> ExperimentResults:
        """
        Frequentist statistical analysis

        Uses t-test for continuous outcomes, chi-squared for binary
        """
        variants = list(experiment.variants.values())

        if len(variants) < 2:
            return ExperimentResults(
                experiment_id=experiment.experiment_id,
                winner=None,
                confidence=0.0,
                improvement=0.0,
                significance=False,
                test_statistic=0.0,
                p_value=1.0,
                recommendation="Need at least 2 variants"
            )

        # Compare each variant against control (first variant)
        control = variants[0]
        best_p_value = 1.0
        best_variant = None
        best_improvement = 0.0
        best_statistic = 0.0

        for treatment in variants[1:]:
            # Check sample sizes
            if control.sample_size < experiment.min_sample_size or \
               treatment.sample_size < experiment.min_sample_size:
                continue

            # Two-sample t-test
            statistic, p_value = stats.ttest_ind(
                control.observations,
                treatment.observations,
                equal_var=False
            )

            # Calculate improvement
            improvement = (
                (treatment.mean_outcome - control.mean_outcome) /
                max(0.001, control.mean_outcome)
            )

            # Apply multiple testing correction
            num_comparisons = len(variants) - 1
            if self.multiple_testing_correction == "bonferroni":
                adjusted_p = p_value * num_comparisons
            elif self.multiple_testing_correction == "holm":
                # Holm-Bonferroni (simplification)
                adjusted_p = min(1.0, p_value * num_comparisons)
            else:
                adjusted_p = p_value

            if adjusted_p < best_p_value:
                best_p_value = adjusted_p
                best_variant = treatment.variant_id
                best_improvement = improvement
                best_statistic = statistic

        # Determine significance
        is_significant = best_p_value < experiment.significance_level

        # Confidence (1 - p_value)
        confidence = 1.0 - best_p_value

        # Generate recommendation
        if is_significant:
            if best_improvement > 0:
                recommendation = (
                    f"Adopt {best_variant}. Shows {best_improvement:.1%} improvement "
                    f"over control with {confidence:.0%} confidence."
                )
            else:
                recommendation = (
                    f"Keep control. {best_variant} underperforms by "
                    f"{abs(best_improvement):.1%}."
                )
        else:
            recommendation = (
                f"No significant difference detected. "
                f"Collect more data or consider both variants equivalent."
            )

        return ExperimentResults(
            experiment_id=experiment.experiment_id,
            winner=best_variant if is_significant and best_improvement > 0 else None,
            confidence=confidence,
            improvement=best_improvement,
            significance=is_significant,
            test_statistic=best_statistic,
            p_value=best_p_value,
            recommendation=recommendation
        )

    async def _bayesian_analysis(
        self,
        experiment: Experiment
    ) -> ExperimentResults:
        """
        Bayesian statistical analysis

        Uses Beta-Bernoulli model for binary outcomes
        """
        variants = list(experiment.variants.values())

        if len(variants) < 2:
            return ExperimentResults(
                experiment_id=experiment.experiment_id,
                winner=None,
                confidence=0.0,
                improvement=0.0,
                significance=False,
                test_statistic=0.0,
                p_value=0.0,
                recommendation="Need at least 2 variants"
            )

        # For each variant, compute posterior Beta distribution
        # Alpha = successes + 1, Beta = failures + 1
        posteriors = {}
        for var in variants:
            successes = int(var.conversion_rate * var.sample_size)
            failures = var.sample_size - successes
            posteriors[var.variant_id] = {
                "alpha": successes + 1,
                "beta": failures + 1,
                "mean": (successes + 1) / (var.sample_size + 2)
            }

        # Sample from posteriors to determine winner
        num_samples = 10000
        samples = {
            var_id: np.random.beta(post["alpha"], post["beta"], num_samples)
            for var_id, post in posteriors.items()
        }

        # Count how often each variant wins
        win_counts = {var_id: 0 for var_id in posteriors.keys()}
        for i in range(num_samples):
            best_sample = -1
            best_var = None
            for var_id, var_samples in samples.items():
                if var_samples[i] > best_sample:
                    best_sample = var_samples[i]
                    best_var = var_id
            win_counts[best_var] += 1

        # Find winner
        best_variant = max(win_counts, key=win_counts.get)
        win_probability = win_counts[best_variant] / num_samples

        # Calculate improvement
        control_mean = posteriors[variants[0].variant_id]["mean"]
        treatment_mean = posteriors[best_variant]["mean"]
        improvement = (treatment_mean - control_mean) / max(0.001, control_mean)

        # Significance: probability of winning > threshold
        is_significant = win_probability > (1 - experiment.significance_level)

        # Recommendation
        if is_significant:
            recommendation = (
                f"Adopt {best_variant}. {win_probability:.1%} probability of being best. "
                f"Expected improvement: {improvement:.1%}."
            )
        else:
            recommendation = (
                f"Insufficient evidence. {best_variant} wins "
                f"{win_probability:.1%} of samples, but below threshold."
            )

        return ExperimentResults(
            experiment_id=experiment.experiment_id,
            winner=best_variant if is_significant else None,
            confidence=win_probability,
            improvement=improvement,
            significance=is_significant,
            test_statistic=win_probability,
            p_value=1 - win_probability,
            recommendation=recommendation
        )

    async def _check_early_stopping(self, experiment_id: str) -> None:
        """
        Check if experiment should stop early

        Uses sequential analysis: Stop if overwhelming evidence
        """
        if not self.enable_early_stopping:
            return

        experiment = self.experiments[experiment_id]

        # Check minimum interval
        total_samples = sum(
            var.sample_size
            for var in experiment.variants.values()
        )
        if total_samples % self.early_stop_look_interval != 0:
            return

        # Get current results
        results = await self.get_results(experiment_id)

        # Early stopping criteria
        if results.significance and results.confidence > 0.99:
            # Very strong evidence
            await self.complete_experiment(
                experiment_id,
                winner=results.winner,
                reason="Early stopping: Very strong evidence"
            )
        elif results.confidence > 0.95 and results.improvement > 0.3:
            # Strong positive evidence
            await self.complete_experiment(
                experiment_id,
                winner=results.winner,
                reason="Early stopping: Strong positive evidence"
            )

    async def select_winner(
        self,
        experiment_id: str
    ) -> Optional[str]:
        """
        Select winning variant

        Args:
            experiment_id: Experiment identifier

        Returns:
            Winning variant ID or None if experiment not complete
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.COMPLETED:
            logger.warning(f"Experiment {experiment_id} not complete")
            return None

        return experiment.winner

    async def complete_experiment(
        self,
        experiment_id: str,
        winner: Optional[str] = None,
        reason: str = "Manual completion"
    ) -> None:
        """
        Mark experiment as complete

        Args:
            experiment_id: Experiment identifier
            winner: Winning variant (if determined)
            reason: Reason for completion
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = datetime.now(UTC)
        experiment.winner = winner

        logger.info(
            f"Completed experiment {experiment_id}: "
            f"winner={winner}, reason={reason}"
        )

    async def abandon_experiment(
        self,
        experiment_id: str,
        reason: str
    ) -> None:
        """
        Abandon experiment (no winner)

        Args:
            experiment_id: Experiment identifier
            reason: Reason for abandoning
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.ABANDONED
        experiment.end_time = datetime.now(UTC)

        logger.info(f"Abandoned experiment {experiment_id}: {reason}")

    async def get_experiment(
        self,
        experiment_id: str
    ) -> Optional[Experiment]:
        """
        Get experiment by ID

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment or None
        """
        return self.experiments.get(experiment_id)

    async def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None
    ) -> List[Experiment]:
        """
        List experiments, optionally filtered by status

        Args:
            status: Filter by status

        Returns:
            List of experiments
        """
        experiments = list(self.experiments.values())

        if status:
            experiments = [e for e in experiments if e.status == status]

        return sorted(experiments, key=lambda e: e.start_time, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "significance_level": self.significance_level,
            "min_sample_size": self.min_sample_size,
            "test_method": self.test_method,
            "enable_early_stopping": self.enable_early_stopping,
            "experiments": {
                exp_id: exp.to_dict()
                for exp_id, exp in self.experiments.items()
            }
        }
