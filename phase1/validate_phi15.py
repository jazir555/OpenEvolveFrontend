"""
Phi 1.5 Validation Script

Validate the Phi 1.5 system on synthetic data with known hidden constraints.
Measures assumption mining accuracy against a ground truth.

Author: Agent B1 (Phi 1/Phi 1.5 Specialist)
Created: 2025-12-31
Status: Green - Active Implementation
Target: >70% assumption mining accuracy
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from phase1.tacit_assumption_miner import (
    Phi15Engine, NullResult, TacitAssumption,
    ErrorType, AssumptionType
)


# ============================================================================
# Synthetic Data Generation
# ============================================================================

class SyntheticDataGenerator:
    """
    Generate synthetic null results with known hidden constraints.

    Creates realistic failure patterns where we know the ground truth
    tacit assumption, enabling validation of Φ₁.₅'s inference accuracy.
    """

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed"""
        np.random.seed(seed)

    def generate_case1_approximation(self, n_failures: int = 50) -> Tuple[List[NullResult], Dict]:
        """
        Case 1: Exact solutions fail, approximation needed.

        Hidden Constraint: "Must use exact algorithms"
        Tacit Assumption: "Approximation is acceptable"

        Ground truth: System repeatedly fails when trying to solve
        NP-hard problem exactly, indicating need for approximation.
        """
        null_results = []

        for i in range(n_failures):
            result = NullResult(
                attempt_id=f"approx_case_{i:03d}",
                timestamp=datetime.now() - timedelta(minutes=i),
                problem_type="np_optimization",
                approach_type="exact",
                constraints=["exact_solution_required", "optimal_solution"],
                error_type=ErrorType.TIMEOUT if i % 3 == 0 else ErrorType.INFEASIBILITY,
                error_message=f"Exact algorithm exceeded time limit (complexity: 2^{i/10})",
                state={
                    "iteration": 100 + i * 10,
                    "time_elapsed": 7200.0,  # 2 hours
                    "time_limit": 3600.0,
                    "problem_size": 50 + i
                },
                iteration=100 + i * 10,
                resources_used={
                    "cpu": 100.0,
                    "memory": 8000.0,
                    "time": 7200.0
                },
                metadata={
                    "case": "approximation",
                    "problem_class": "NP-hard",
                    "algorithm": "branch_and_bound"
                }
            )
            null_results.append(result)

        ground_truth = {
            "case": "approximation",
            "hidden_constraint": "Must use exact algorithms",
            "tacit_assumption": "Approximation is acceptable",
            "assumption_type": "METHODOLOGICAL",
            "pattern": "All exact approaches timeout or prove infeasible",
            "expected_keywords": ["exact", "timeout", "optimal", "time", "limit"]
        }

        return null_results, ground_truth

    def generate_case2_randomization(self, n_failures: int = 50) -> Tuple[List[NullResult], Dict]:
        """
        Case 2: Deterministic methods fail, randomization helps.

        Hidden Constraint: "Must be deterministic"
        Tacit Assumption: "Randomization can break symmetries"

        Ground truth: Deterministic algorithms get stuck in local optima,
        indicating need for randomized approaches.
        """
        null_results = []

        for i in range(n_failures):
            # Deterministic approaches fail in same patterns
            stuck_at_local_optima = (i % 5) * 0.2  # Always at same points

            result = NullResult(
                attempt_id=f"rand_case_{i:03d}",
                timestamp=datetime.now() - timedelta(minutes=i),
                problem_type="global_optimization",
                approach_type="deterministic",
                constraints=["deterministic_required", "reproducible"],
                error_type=ErrorType.OPTIMIZATION_FAILED,
                error_message=f"Converged to local optimum at {stuck_at_local_optima:.2f}",
                state={
                    "iteration": 1000 + i * 100,
                    "objective_value": stuck_at_local_optima,
                    "global_optimum": 1.0,
                    "stuck_count": 10 + i
                },
                iteration=1000 + i * 100,
                resources_used={
                    "cpu": 80.0,
                    "memory": 2000.0
                },
                metadata={
                    "case": "randomization",
                    "problem": "multi_modal_optimization",
                    "local_optima_pattern": True
                }
            )
            null_results.append(result)

        ground_truth = {
            "case": "randomization",
            "hidden_constraint": "Must be deterministic",
            "tacit_assumption": "Randomization can help escape local optima",
            "assumption_type": "METHODOLOGICAL",
            "pattern": "Deterministic methods consistently converge to same suboptimal points",
            "expected_keywords": ["deterministic", "local", "optimum", "converged", "stuck"]
        }

        return null_results, ground_truth

    def generate_case3_relaxation(self, n_failures: int = 50) -> Tuple[List[NullResult], Dict]:
        """
        Case 3: Problem is infeasible with current constraints.

        Hidden Constraint: "All constraints must be satisfied"
        Tacit Assumption: "Some constraints can be relaxed"

        Ground truth: Problem repeatedly proves infeasible, indicating
        overly restrictive constraint set.
        """
        null_results = []

        for i in range(n_failures):
            violated_constraint = f"constraint_{i % 5}"  # Same 5 constraints violated

            result = NullResult(
                attempt_id=f"relax_case_{i:03d}",
                timestamp=datetime.now() - timedelta(minutes=i),
                problem_type="constraint_satisfaction",
                approach_type="exact_csp",
                constraints=["constraint_0", "constraint_1", "constraint_2",
                          "constraint_3", "constraint_4"],
                error_type=ErrorType.INFEASIBILITY,
                error_message=f"Problem infeasible: {violated_constraint} contradicts others",
                state={
                    "iteration": i,
                    "constraint_violations": [violated_constraint],
                    "feasibility": "PROVEN_INFEASIBLE"
                },
                iteration=i,
                resources_used={
                    "cpu": 50.0,
                    "memory": 1000.0
                },
                metadata={
                    "case": "relaxation",
                    "violated_constraint": violated_constraint,
                    "constraint_set": "over_constrained"
                }
            )
            null_results.append(result)

        ground_truth = {
            "case": "relaxation",
            "hidden_constraint": "All constraints must be satisfied",
            "tacit_assumption": "Some constraints can be relaxed or treated as soft",
            "assumption_type": "CONSTRAINT",
            "pattern": "Same subset of constraints consistently violated",
            "expected_keywords": ["infeasible", "constraint", "violated", "contradicts"]
        }

        return null_results, ground_truth

    def generate_case4_scale(self, n_failures: int = 50) -> Tuple[List[NullResult], Dict]:
        """
        Case 4: Method works at small scale, fails at large scale.

        Hidden Constraint: "Problem is independent of scale"
        Tacit Assumption: "Scale matters - need different approach for large instances"

        Ground truth: Method succeeds for small instances but fails for
        large ones, indicating scale-dependent limitation.
        """
        null_results = []

        for i in range(n_failures):
            problem_size = 1000 + i * 100  # Increasing size

            result = NullResult(
                attempt_id=f"scale_case_{i:03d}",
                timestamp=datetime.now() - timedelta(minutes=i),
                problem_type="large_scale_optimization",
                approach_type="iterative_method",
                constraints=["memory_efficient"],
                error_type=ErrorType.NUMERICAL_INSTABILITY if i > 25 else ErrorType.TIMEOUT,
                error_message=f"Method unstable at scale {problem_size} (condition number: {1e10 + i*1e9})",
                state={
                    "iteration": 50,
                    "problem_size": problem_size,
                    "condition_number": 1e10 + i * 1e9,
                    "numerical_error": 1e-5 + i * 1e-4
                },
                iteration=50,
                resources_used={
                    "cpu": 90.0,
                    "memory": 16000.0
                },
                metadata={
                    "case": "scale",
                    "problem_size": problem_size,
                    "small_scale_works": True,
                    "large_scale_fails": True
                }
            )
            null_results.append(result)

        ground_truth = {
            "case": "scale",
            "hidden_constraint": "Problem is independent of scale",
            "tacit_assumption": "Scale affects feasibility - need specialized large-scale methods",
            "assumption_type": "REPRESENTATIONAL",
            "pattern": "Failures only occur beyond certain scale threshold",
            "expected_keywords": ["scale", "large", "instability", "condition", "size"]
        }

        return null_results, ground_truth

    def generate_all_cases(self) -> List[Tuple[List[NullResult], Dict]]:
        """Generate all validation cases"""
        cases = []

        cases.append(self.generate_case1_approximation())
        cases.append(self.generate_case2_randomization())
        cases.append(self.generate_case3_relaxation())
        cases.append(self.generate_case4_scale())

        return cases


# ============================================================================
# Validation Metrics
# ============================================================================

class Phi15Validator:
    """
    Validate Φ₁.₅ performance against ground truth.

    Measures:
    - Accuracy: Did we infer the correct tacit assumption?
    - Precision: Of inferred assumptions, how many are correct?
    - Recall: Of ground truth assumptions, how many did we find?
    - Confidence calibration: Are confidence scores meaningful?
    """

    def __init__(self):
        self.results = []

    def validate_case(self, null_results: List[NullResult],
                     ground_truth: Dict) -> Dict:
        """
        Validate Φ₁.₅ on a single test case.

        Args:
            null_results: Generated null results
            ground_truth: Ground truth tacit assumption

        Returns:
            Validation metrics
        """
        # Create engine and process
        engine = Phi15Engine()
        assumptions, paradigm_rec = engine.process_null_results(null_results)

        # Get top assumptions
        top_assumptions = engine.get_top_assumptions(k=5)

        # Check if ground truth was found
        found = False
        matched_assumption = None

        for assumption in top_assumptions:
            # Check for semantic similarity (simplified)
            if self._matches_ground_truth(assumption, ground_truth):
                found = True
                matched_assumption = assumption
                break

        # Compute metrics
        metrics = {
            'case': ground_truth['case'],
            'ground_truth': ground_truth['tacit_assumption'],
            'assumptions_generated': len(assumptions),
            'top_assumptions': len(top_assumptions),
            'ground_truth_found': found,
            'matched_assumption': matched_assumption.description if matched_assumption else None,
            'confidence': matched_assumption.confidence if matched_assumption else None,
            'paradigm_crisis': paradigm_rec.trigger
        }

        self.results.append(metrics)
        return metrics

    def _matches_ground_truth(self, assumption: TacitAssumption,
                            ground_truth: Dict) -> bool:
        """
        Check if assumption matches ground truth (simplified).

        Real implementation would use semantic similarity / embeddings.
        """
        gt_desc = ground_truth['tacit_assumption'].lower()
        ass_desc = assumption.description.lower()

        # Check for keyword overlap
        gt_words = set(gt_desc.split())
        ass_words = set(ass_desc.split())

        overlap = len(gt_words & ass_words) / max(len(gt_words), 1)

        # Also check expected keywords
        expected_keywords = ground_truth.get('expected_keywords', [])
        keyword_matches = sum(1 for kw in expected_keywords
                            if kw.lower() in ass_desc)

        # Consider it a match if sufficient overlap
        return overlap > 0.3 or keyword_matches >= 2

    def compute_summary_metrics(self) -> Dict:
        """Compute summary metrics across all test cases"""
        if not self.results:
            return {}

        total_cases = len(self.results)
        correct_cases = sum(1 for r in self.results if r['ground_truth_found'])

        accuracy = correct_cases / total_cases if total_cases > 0 else 0

        # Average confidence of correct matches
        confidences = [r['confidence'] for r in self.results
                      if r['ground_truth_found'] and r['confidence'] is not None]
        avg_confidence = np.mean(confidences) if confidences else 0

        summary = {
            'total_cases': total_cases,
            'correct_cases': correct_cases,
            'accuracy': accuracy,
            'target_accuracy': 0.7,
            'target_met': accuracy >= 0.7,
            'avg_confidence_correct': avg_confidence,
            'paradigm_crisis_count': sum(1 for r in self.results if r['paradigm_crisis'])
        }

        return summary

    def print_report(self) -> None:
        """Print validation report"""
        print("\n" + "="*70)
        print("Φ₁.₅ VALIDATION REPORT")
        print("="*70)

        # Per-case results
        print("\nPer-Case Results:")
        print("-"*70)
        for i, result in enumerate(self.results, 1):
            status = "✓ CORRECT" if result['ground_truth_found'] else "✗ INCORRECT"
            print(f"\nCase {i}: {result['case']}")
            print(f"  Status: {status}")
            print(f"  Ground Truth: {result['ground_truth']}")
            print(f"  Assumptions Generated: {result['assumptions_generated']}")
            if result['matched_assumption']:
                print(f"  Matched: {result['matched_assumption']}")
                print(f"  Confidence: {result['confidence']:.2f}")

        # Summary metrics
        print("\n" + "="*70)
        print("SUMMARY METRICS")
        print("="*70)

        summary = self.compute_summary_metrics()

        print(f"\nTotal Test Cases: {summary['total_cases']}")
        print(f"Correct Inferences: {summary['correct_cases']}")
        print(f"Accuracy: {summary['accuracy']:.1%}")
        print(f"Target Accuracy: 70%")
        print(f"Target Met: {'✓ YES' if summary['target_met'] else '✗ NO'}")

        if summary['avg_confidence_correct'] > 0:
            print(f"\nAverage Confidence (Correct): {summary['avg_confidence_correct']:.2f}")

        print(f"\nParadigm Crisis Detected: {summary['paradigm_crisis_count']} cases")

        # Final verdict
        print("\n" + "="*70)
        if summary['target_met']:
            print("✓ Φ₁.₅ VALIDATION PASSED - Target accuracy achieved!")
        else:
            print("✗ Φ₁.₅ VALIDATION FAILED - Below target accuracy")
        print("="*70 + "\n")


# ============================================================================
# Main Validation Script
# ============================================================================

def main():
    """Run validation suite"""
    print("Φ₁.₅ Validation Suite")
    print("="*70)
    print("Validating assumption mining on synthetic data with known ground truth")
    print(f"Target accuracy: >70%")
    print()

    # Generate test cases
    print("Generating synthetic test cases...")
    generator = SyntheticDataGenerator(seed=42)
    test_cases = generator.generate_all_cases()
    print(f"Generated {len(test_cases)} test cases")

    # Run validation
    print("\nRunning validation...")
    validator = Phi15Validator()

    for i, (null_results, ground_truth) in enumerate(test_cases, 1):
        print(f"\nValidating case {i}/{len(test_cases)}: {ground_truth['case']}")
        print(f"  Null results: {len(null_results)}")
        print(f"  Ground truth: {ground_truth['tacit_assumption']}")

        metrics = validator.validate_case(null_results, ground_truth)

        status = "✓" if metrics['ground_truth_found'] else "✗"
        print(f"  Result: {status}")

    # Print report
    validator.print_report()

    # Save results
    results_dir = Path("rese/data/validation")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"phi15_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'results': validator.results,
            'summary': validator.compute_summary_metrics(),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return validator.compute_summary_metrics()['target_met']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
