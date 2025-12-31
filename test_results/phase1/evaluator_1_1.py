
def evaluate(program_path):
    """Evaluate sorting function"""
    import importlib.util
    import time
    import sys

    # Load the program
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        return {"score": 0.0, "error": "Failed to load program"}

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {"score": 0.0, "error": f"Execution error: {str(e)}"}

    # Check if function exists
    if not hasattr(module, 'bubble_sort'):
        return {"score": 0.0, "error": "bubble_sort function not found"}

    sort_func = module.bubble_sort

    # Test cases
    test_cases = [
        ([3, 1, 2], [1, 2, 3]),
        ([5, 2, 8, 1, 9], [1, 2, 5, 8, 9]),
        ([1], [1]),
        ([], []),
        (list(range(10, 0, -1)), list(range(1, 11))),
    ]

    correctness_score = 0.0
    performance_score = 0.0

    # Test correctness
    for input_arr, expected in test_cases:
        try:
            input_copy = input_arr.copy()
            result = sort_func(input_copy)
            if result == expected:
                correctness_score += 1.0
        except Exception as e:
            print(f"Error on test case: {e}")

    correctness_score = correctness_score / len(test_cases)

    # Test performance on larger dataset
    if correctness_score >= 0.8:
        large_input = list(range(100, 0, -1))

        try:
            start = time.time()
            result = sort_func(large_input.copy())
            duration = time.time() - start

            if result == sorted(large_input):
                # Score based on speed (faster is better)
                # Bubble sort should take ~0.01-0.1s
                # Target is < 0.05s for good implementation
                if duration < 0.05:
                    performance_score = 1.0
                elif duration < 0.1:
                    performance_score = 0.8
                elif duration < 0.5:
                    performance_score = 0.5
                else:
                    performance_score = 0.2
            else:
                performance_score = 0.0
        except:
            performance_score = 0.0

    # Combined score: 70% correctness, 30% performance
    combined_score = 0.7 * correctness_score + 0.3 * performance_score

    return {
        "combined_score": combined_score,
        "correctness": correctness_score,
        "performance": performance_score
    }
