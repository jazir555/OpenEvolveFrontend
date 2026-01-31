"""
Evaluator for Basic Evolution Example

This evaluator tests how well the solve() function maximizes x^2.
"""

import sys
import importlib.util


def evaluate(program_path):
    """
    Evaluate a program and return performance metrics.

    Args:
        program_path: Path to the program file to evaluate

    Returns:
        Dictionary with metrics (must include 'score' or 'combined_score')
    """
    # Load the program
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        return {"combined_score": 0.0, "error": "Failed to load program"}

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {"combined_score": 0.0, "error": f"Failed to execute: {str(e)}"}

    # Check if solve function exists
    if not hasattr(module, 'solve'):
        return {"combined_score": 0.0, "error": "No solve() function found"}

    # Test the function
    try:
        result = module.solve()

        # The goal is to maximize x^2 where x in [0, 10]
        # Maximum possible value is 10^2 = 100
        # Higher is better, so score is just the result
        score = float(result)

        # Normalize to 0-1 range (100 is perfect)
        normalized_score = score / 100.0

        return {
            "combined_score": normalized_score,  # Primary metric for evolution
            "raw_value": score,                  # Actual x^2 value
            "success": score > 90                # Bonus for being close to optimal
        }

    except Exception as e:
        return {"combined_score": 0.0, "error": f"Execution error: {str(e)}"}


# For standalone testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <program_path>")
        sys.exit(1)

    program_path = sys.argv[1]
    metrics = evaluate(program_path)

    print("Evaluation Results:")
    print(f"  Combined Score: {metrics.get('combined_score', 0):.4f}")
    print(f"  Raw Value: {metrics.get('raw_value', 0):.2f}")
    print(f"  Success: {metrics.get('success', False)}")

    if 'error' in metrics:
        print(f"  Error: {metrics['error']}")
