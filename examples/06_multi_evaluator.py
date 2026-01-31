"""
Evaluator for Multi-Objective Optimization

Evaluates multiple competing objectives and returns combined score.
"""

import sys
import importlib.util


def evaluate(program_path):
    """
    Multi-objective evaluation.

    Objectives:
    1. Maximize sum (x + y) - want large values
    2. Minimize distance from origin (x^2 + y^2) - want small values
    3. Satisfy constraints (0 <= x, y <= 10)

    Returns both individual metrics and combined score.
    """
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        return {"combined_score": 0.0, "error": "Failed to load"}

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {"combined_score": 0.0, "error": f"Load error: {e}"}

    if not hasattr(module, 'multi_objective_optimize'):
        return {"combined_score": 0.0, "error": "No multi_objective_optimize function"}

    try:
        # Test with different inputs
        test_cases = [
            (5, 5),   # Balanced
            (10, 0),  # Extreme sum
            (0, 10),  # Extreme sum
            (1, 1),   # Close to origin
            (10, 10), # Max values
        ]

        best_combined = 0.0
        best_metrics = {}

        for x, y in test_cases:
            f1, f2 = module.multi_objective_optimize(x, y)

            # Normalize f1 (sum) - maximum is 20 (10+10)
            sum_score = f1 / 20.0

            # Normalize f2 (distance) - maximum is 200 (10^2+10^2)
            # We want to minimize, so invert
            distance_score = 1.0 - (f2 / 200.0)

            # Constraint penalty (x, y should be in [0, 10])
            penalty = 0.0
            if not (0 <= x <= 10 and 0 <= y <= 10):
                penalty = 0.5

            # Combined score with weights
            # Weight sum optimization higher than distance minimization
            combined = (sum_score * 0.6) + (distance_score * 0.4) - penalty
            combined = max(0.0, combined)  # Ensure non-negative

            if combined > best_combined:
                best_combined = combined
                best_metrics = {
                    "sum_score": sum_score,
                    "distance_score": distance_score,
                    "penalty": penalty,
                    "x": x,
                    "y": y,
                    "f1": f1,
                    "f2": f2
                }

        return {
            "combined_score": best_combined,
            **best_metrics
        }

    except Exception as e:
        return {"combined_score": 0.0, "error": f"Eval error: {e}"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python multi_evaluator.py <program_path>")
        sys.exit(1)

    metrics = evaluate(sys.argv[1])

    print("Multi-Objective Evaluation:")
    print(f"  Combined Score: {metrics['combined_score']:.4f}")
    print(f"  Sum Score: {metrics.get('sum_score', 0):.4f}")
    print(f"  Distance Score: {metrics.get('distance_score', 0):.4f}")
    print(f"  Penalty: {metrics.get('penalty', 0):.2f}")
    print(f"  Best Test Point: x={metrics.get('x', 0)}, y={metrics.get('y', 0)}")
    print(f"  f1 (sum): {metrics.get('f1', 0):.2f}")
    print(f"  f2 (distance): {metrics.get('f2', 0):.2f}")

    if 'error' in metrics:
        print(f"  Error: {metrics['error']}")
