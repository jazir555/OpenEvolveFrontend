"""
Evaluator for Function Evolution - Sorting Algorithm

This evaluator tests both correctness and speed of sorting algorithms.
"""

import sys
import time
import importlib.util


def evaluate(program_path):
    """
    Evaluate sorting algorithm performance.

    Tests:
    1. Correctness - Does it sort correctly?
    2. Speed - How fast does it sort?
    3. Edge cases - Empty arrays, single element, duplicates
    """
    # Load the program
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        return {"combined_score": 0.0, "error": "Failed to load"}

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {"combined_score": 0.0, "error": f"Load error: {e}"}

    if not hasattr(module, 'sort_array'):
        return {"combined_score": 0.0, "error": "No sort_array function"}

    # Test cases
    test_cases = [
        ([], []),                          # Empty
        ([1], [1]),                        # Single element
        ([3, 1, 2], [1, 2, 3]),            # Simple
        ([5, 2, 8, 1, 9], [1, 2, 5, 8, 9]), # Medium
        (list(range(10, 0, -1)), list(range(1, 11))), # Reverse sorted
        ([1, 1, 1, 1], [1, 1, 1, 1]),      # Duplicates
    ]

    # Performance test
    large_test = list(range(100, 0, -1))

    correctness_score = 0.0
    speed_score = 0.0
    total_tests = len(test_cases)

    for input_arr, expected in test_cases:
        try:
            result = module.sort_array(input_arr)
            if result == expected:
                correctness_score += 1.0
        except Exception as e:
            pass  # Failed test

    # Normalize correctness (0-1)
    correctness = correctness_score / total_tests

    # Measure speed on large test
    if correctness > 0.5:  # Only measure speed if mostly correct
        try:
            start = time.time()
            result = module.sort_array(large_test)
            duration = time.time() - start

            if result == sorted(large_test):
                # Faster is better - aim for < 0.001 seconds
                # Log scale for scoring
                if duration < 0.001:
                    speed_score = 1.0
                else:
                    speed_score = 1.0 / (1.0 + duration * 100)
        except:
            speed_score = 0.0

    # Combined score: 70% correctness, 30% speed
    combined = (correctness * 0.7) + (speed_score * 0.3)

    return {
        "combined_score": combined,
        "correctness": correctness,
        "speed": speed_score,
        "tests_passed": int(correctness_score)
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sort_evaluator.py <program_path>")
        sys.exit(1)

    metrics = evaluate(sys.argv[1])

    print("Sorting Algorithm Evaluation:")
    print(f"  Combined Score: {metrics['combined_score']:.4f}")
    print(f"  Correctness: {metrics['correctness']:.2%}")
    print(f"  Speed Score: {metrics['speed']:.4f}")
    print(f"  Tests Passed: {metrics['tests_passed']}/6")

    if 'error' in metrics:
        print(f"  Error: {metrics['error']}")
