"""
Γ₁ Threshold Learner

Learns optimal ACI threshold for classifying solvable vs intractable instances.
"""

from typing import List, Tuple
import numpy as np
from gamma1.core.aci_calculator import ACIResult


class ThresholdLearner:
    """
    Learn optimal ACI threshold for classification

    Finds threshold that best separates solvable from intractable instances.
    """

    def learn_optimal_threshold(
        self,
        aci_results: List[ACIResult],
        solve_times: List[float],
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        step: float = 0.05
    ) -> Tuple[float, float]:
        """
        Learn optimal threshold for classification

        Args:
            aci_results: List of ACI results
            solve_times: Corresponding solve times (inf for intractable)
            threshold_range: (min, max) range to search
            step: Step size for threshold search

        Returns:
            (optimal_threshold, max_accuracy)
        """
        if len(aci_results) != len(solve_times):
            raise ValueError("ACI results and solve times must have same length")

        # Create labels
        labels = [1 if t < float('inf') else 0 for t in solve_times]
        aci_scores = [r.ACI for r in aci_results]

        if not labels:
            return (0.5, 0.0)

        # Search thresholds
        best_threshold = 0.5
        best_accuracy = 0.0

        for threshold in np.arange(threshold_range[0], threshold_range[1] + step, step):
            predictions = [1 if aci > threshold else 0 for aci in aci_scores]
            accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        return (best_threshold, best_accuracy)

    def get_accuracy_at_threshold(
        self,
        aci_results: List[ACIResult],
        solve_times: List[float],
        threshold: float
    ) -> float:
        """
        Calculate accuracy at specific threshold

        Args:
            aci_results: List of ACI results
            solve_times: Solve times
            threshold: Classification threshold

        Returns:
            Accuracy at threshold
        """
        labels = [1 if t < float('inf') else 0 for t in solve_times]
        aci_scores = [r.ACI for r in aci_results]

        predictions = [1 if aci > threshold else 0 for aci in aci_scores]
        accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels) if labels else 0.0

        return accuracy


if __name__ == "__main__":
    print("=" * 70)
    print("Threshold Learner - Demonstration")
    print("=" * 70)

    from gamma1.core.aci_calculator import ACICalculator
    from gamma1.core.csp_models import create_tree_csp, create_dense_csp

    calculator = ACICalculator()
    learner = ThresholdLearner()

    # Generate instances
    print("\n[OK] Generating instances...")
    results = []
    times = []

    # Solvable
    for _ in range(20):
        csp = create_tree_csp(n_variables=10, domain_size=3)
        result = calculator.calculate(csp)
        results.append(result)
        times.append(1.0)

    # Intractable
    for _ in range(20):
        csp = create_dense_csp(n_variables=10, domain_size=3)
        result = calculator.calculate(csp)
        results.append(result)
        times.append(float('inf'))

    # Learn threshold
    print("\n[OK] Learning optimal threshold...")
    optimal_threshold, max_accuracy = learner.learn_optimal_threshold(results, times)

    print(f"\nOptimal threshold: {optimal_threshold:.3f}")
    print(f"Max accuracy: {max_accuracy:.3f}")

    # Test at default threshold
    default_accuracy = learner.get_accuracy_at_threshold(results, times, 0.5)
    print(f"Accuracy at default (0.5): {default_accuracy:.3f}")

    print("\n" + "=" * 70)
    print("[OK] Threshold learner demonstration complete")
    print("=" * 70)
