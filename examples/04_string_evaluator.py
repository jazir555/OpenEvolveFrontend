"""
Evaluator for String Processing Example

Tests string processing capabilities like cleaning, formatting, etc.
"""

import sys
import importlib.util


def evaluate(program_path):
    """
    Evaluate string processing function.

    Tests various string operations:
    - Whitespace handling
    - Case conversion
    - Punctuation handling
    """
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        return {"combined_score": 0.0, "error": "Failed to load"}

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {"combined_score": 0.0, "error": f"Load error: {e}"}

    if not hasattr(module, 'process_string'):
        return {"combined_score": 0.0, "error": "No process_string function"}

    # Test cases
    test_cases = [
        # (input, expected_output, description)
        ("  hello  world  ", "hello world", "Remove extra spaces"),
        ("Hello World", "hello world", "Lowercase"),
        ("  TEST  ", "test", "Trim and lowercase"),
        ("hello\nworld", "hello world", "Normalize newlines"),
    ]

    passed = 0
    total = len(test_cases)

    for input_text, expected, description in test_cases:
        try:
            result = module.process_string(input_text)
            if result == expected:
                passed += 1
        except:
            pass  # Test failed

    # Calculate score
    score = passed / total if total > 0 else 0.0

    return {
        "combined_score": score,
        "tests_passed": passed,
        "total_tests": total,
        "pass_rate": score
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python string_evaluator.py <program_path>")
        sys.exit(1)

    metrics = evaluate(sys.argv[1])

    print("String Processing Evaluation:")
    print(f"  Score: {metrics['combined_score']:.4f}")
    print(f"  Tests Passed: {metrics['tests_passed']}/{metrics['total_tests']}")
    print(f"  Pass Rate: {metrics['pass_rate']:.2%}")

    if 'error' in metrics:
        print(f"  Error: {metrics['error']}")
