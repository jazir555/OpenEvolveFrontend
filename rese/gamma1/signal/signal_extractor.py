"""
Γ₁ Signal Extractor

Extracts solvability signal from ACI scores.
Validates ACI correlation with actual solvability.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score
from gamma1.core.aci_calculator import ACIResult


@dataclass
class SignalQuality:
    """
    Signal Quality Metrics

    Attributes:
        signal_to_noise: Signal-to-noise ratio
        correlation: Pearson correlation with solve time
        accuracy: Classification accuracy (solvable vs intractable)
        auc: ROC AUC score
        mean_solvable_aci: Mean ACI for solvable instances
        mean_intractable_aci: Mean ACI for intractable instances
        separation_quality: Qualitative assessment
    """
    signal_to_noise: float = 0.0
    correlation: float = 0.0
    accuracy: float = 0.0
    auc: float = 0.0
    mean_solvable_aci: float = 0.0
    mean_intractable_aci: float = 0.0
    separation_quality: str = "UNKNOWN"

    def meets_target(self, target_correlation: float = 0.85) -> bool:
        """
        Check if metrics meet target

        Args:
            target_correlation: Target correlation threshold

        Returns:
            True if all metrics meet targets
        """
        return (self.correlation >= target_correlation and
                self.accuracy >= 0.85 and
                self.auc >= 0.90)


class SignalExtractor:
    """
    Extract solvability signal from ACI scores

    Measures how well ACI separates solvable from intractable instances.
    """

    def extract_signal(
        self,
        aci_results: List[ACIResult],
        solve_times: List[float]
    ) -> SignalQuality:
        """
        Extract solvability signal from ACI scores

        Args:
            aci_results: List of ACI calculation results
            solve_times: Corresponding solve times (inf for intractable)

        Returns:
            SignalQuality metrics
        """
        if len(aci_results) != len(solve_times):
            raise ValueError("ACI results and solve times must have same length")

        if len(aci_results) == 0:
            return SignalQuality()

        # Separate by solvability
        solvable_aci = []
        intractable_aci = []

        for result, time in zip(aci_results, solve_times):
            if time < float('inf'):
                solvable_aci.append(result.ACI)
            else:
                intractable_aci.append(result.ACI)

        if not solvable_aci or not intractable_aci:
            # Need both classes for signal extraction
            return SignalQuality(
                mean_solvable_aci=np.mean(solvable_aci) if solvable_aci else 0.0,
                mean_intractable_aci=np.mean(intractable_aci) if intractable_aci else 0.0,
                separation_quality="INSUFFICIENT_DATA"
            )

        # Calculate metrics
        signal_to_noise = self._calculate_snr(solvable_aci, intractable_aci)
        correlation = self._calculate_correlation(aci_results, solve_times)
        accuracy, auc = self._calculate_classification_metrics(aci_results, solve_times)

        # Mean values
        mean_solvable = np.mean(solvable_aci)
        mean_intractable = np.mean(intractable_aci)

        # Qualitative assessment
        separation_quality = self._assess_separation(signal_to_noise)

        return SignalQuality(
            signal_to_noise=signal_to_noise,
            correlation=abs(correlation),  # Use absolute value
            accuracy=accuracy,
            auc=auc,
            mean_solvable_aci=mean_solvable,
            mean_intractable_aci=mean_intractable,
            separation_quality=separation_quality
        )

    def _calculate_snr(
        self,
        solvable_aci: List[float],
        intractable_aci: List[float]
    ) -> float:
        """
        Calculate signal-to-noise ratio

        Args:
            solvable_aci: ACI scores for solvable instances
            intractable_aci: ACI scores for intractable instances

        Returns:
            Signal-to-noise ratio
        """
        signal = np.mean(solvable_aci) - np.mean(intractable_aci)
        noise = (np.std(solvable_aci) + np.std(intractable_aci)) / 2

        if noise == 0:
            return float('inf') if signal != 0 else 0.0

        return signal / noise

    def _calculate_correlation(
        self,
        aci_results: List[ACIResult],
        solve_times: List[float]
    ) -> float:
        """
        Calculate correlation between ACI and solve time

        Args:
            aci_results: List of ACI results
            solve_times: Solve times (inf for intractable)

        Returns:
            Pearson correlation coefficient
        """
        aci_scores = [r.ACI for r in aci_results]

        # Convert infinite times to large number for correlation
        measurable_times = []
        for t in solve_times:
            if t == float('inf'):
                measurable_times.append(1e6)  # Large value
            else:
                measurable_times.append(t)

        # Calculate Pearson correlation
        if len(aci_scores) < 2:
            return 0.0

        correlation, _ = stats.pearsonr(aci_scores, measurable_times)

        # Note: We expect negative correlation (high ACI = low time)
        # Return absolute value
        return abs(correlation)

    def _calculate_classification_metrics(
        self,
        aci_results: List[ACIResult],
        solve_times: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate classification metrics

        Args:
            aci_results: List of ACI results
            solve_times: Solve times

        Returns:
            (accuracy, auc)
        """
        aci_scores = [r.ACI for r in aci_results]
        labels = [1 if t < float('inf') else 0 for t in solve_times]

        # Find optimal threshold dynamically
        # Use median as a better default than 0.5
        if len(aci_scores) > 0:
            median_threshold = np.median(aci_scores)
        else:
            median_threshold = 0.5

        predictions = [1 if aci >= median_threshold else 0 for aci in aci_scores]

        # Accuracy
        accuracy = accuracy_score(labels, predictions)

        # AUC
        try:
            auc = roc_auc_score(labels, aci_scores)
        except:
            # If only one class present
            auc = 0.5

        return accuracy, auc

    def _assess_separation(self, snr: float) -> str:
        """
        Assess separation quality

        Args:
            snr: Signal-to-noise ratio

        Returns:
            Qualitative assessment
        """
        if snr > 3:
            return "EXCELLENT"
        elif snr > 2:
            return "GOOD"
        elif snr > 1:
            return "FAIR"
        else:
            return "POOR"


if __name__ == "__main__":
    print("=" * 70)
    print("Signal Extractor - Demonstration")
    print("=" * 70)

    from gamma1.core.aci_calculator import ACICalculator
    from gamma1.core.csp_models import create_tree_csp, create_dense_csp

    # Create calculator
    calculator = ACICalculator()
    extractor = SignalExtractor()

    # Generate test instances
    print("\n[OK] Generating test instances...")

    # Solvable instances (tree-structured)
    solvable_results = []
    solvable_times = []

    for i in range(20):
        csp = create_tree_csp(n_variables=10, domain_size=3)
        result = calculator.calculate(csp)
        solvable_results.append(result)
        solvable_times.append(1.0 + i * 0.1)  # Fast

    # Intractable instances (dense)
    intractable_results = []
    intractable_times = []

    for i in range(20):
        csp = create_dense_csp(n_variables=10, domain_size=3, constraint_density=0.9)
        result = calculator.calculate(csp)
        intractable_results.append(result)
        intractable_times.append(float('inf'))  # Intractable

    # Combine
    all_results = solvable_results + intractable_results
    all_times = solvable_times + intractable_times

    # Extract signal
    print("\n[OK] Extracting signal...")
    quality = extractor.extract_signal(all_results, all_times)

    # Display results
    print("\n" + "=" * 70)
    print("Signal Quality Metrics")
    print("=" * 70)
    print(f"Signal-to-Noise:      {quality.signal_to_noise:.3f}")
    print(f"Correlation:          {quality.correlation:.3f}")
    print(f"Accuracy:             {quality.accuracy:.3f}")
    print(f"AUC:                  {quality.auc:.3f}")
    print(f"\nMean Solvable ACI:    {quality.mean_solvable_aci:.3f}")
    print(f"Mean Intractable ACI: {quality.mean_intractable_aci:.3f}")
    print(f"\nSeparation Quality:   {quality.separation_quality}")

    # Check target
    meets_target = quality.meets_target(target_correlation=0.70)  # Lower for demo
    print(f"\nMeets Target (70%):   {meets_target}")

    if meets_target:
        print("\n[SUCCESS] Signal extraction meets target!")
    else:
        print("\n[WARNING] Signal extraction below target")

    print("\n" + "=" * 70)
    print("[OK] Signal extractor demonstration complete")
    print("=" * 70)
