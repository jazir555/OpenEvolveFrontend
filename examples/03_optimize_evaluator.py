"""
Evaluator for 2D Optimization Example

Evaluates how well the optimize() function maximizes f(x,y).
The theoretical maximum is at x=2, y=3 with value f(2,3) = 13
"""

import sys
import importlib.util


def evaluate(program_path):
    """
    Evaluate the optimization function.

    The function to maximize is: f(x,y) = -(x^2 + y^2) + 4x + 6y
    Maximum at (x=2, y=3) with value 13
    """
    # Load program
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        return {"combined_score": 0.0, "error": "Failed to load"}

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {"combined_score": 0.0, "error": f"Load error: {e}"}

    if not hasattr(module, 'optimize'):
        return {"combined_score": 0.0, "error": "No optimize function"}

    try:
        # Test the function with different starting points
        # The best implementation should find x=2, y=3
        test_cases = [
            (0, 0),    # Far from optimal
            (1, 1),    # Medium distance
            (5, 5),    # Far in opposite direction
        ]

        best_value = float('-inf')
        best_params = None

        for x, y in test_cases:
            try:
                result = module.optimize(x, y)
                if result > best_value:
                    best_value = result
                    best_params = (x, y)
            except:
                pass

        # Theoretical maximum is 13 at (2, 3)
        theoretical_max = 13.0

        # Score based on how close to optimum
        # Normalize to 0-1 range
        score = max(0, best_value / theoretical_max)

        return {
            "combined_score": score,
            "best_value": best_value,
            "theoretical_max": theoretical_max,
            "distance_from_optimal": abs(best_value - theoretical_max)
        }

    except Exception as e:
        return {"combined_score": 0.0, "error": f"Eval error: {e}"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python optimize_evaluator.py <program_path>")
        sys.exit(1)

    metrics = evaluate(sys.argv[1])

    print("Optimization Results:")
    print(f"  Score: {metrics['combined_score']:.4f}")
    print(f"  Best Value: {metrics['best_value']:.2f}")
    print(f"  Theoretical Max: {metrics['theoretical_max']:.2f}")
    print(f"  Gap: {metrics['distance_from_optimal']:.2f}")

    if 'error' in metrics:
        print(f"  Error: {metrics['error']}")
