"""
Python API Example - Using OpenEvolve as a Library

This example demonstrates the Python API for programmatic control.
You have full control over the evolution process.

Problem: Evolve a string processing function
"""

# EVOLVE-BLOCK-START
def process_string(text):
    """Process text - initial simple implementation"""
    # Remove extra spaces
    result = ' '.join(text.split())
    return result
# EVOLVE-BLOCK-END


"""
HOW TO USE PYTHON API:
---------------------
"""

from openevolve import run_evolution, evolve_function, evolve_code
from openevolve.config import Config, LLMModelConfig
import os


# Example 1: Simple API call
def example_1_simple():
    """Basic usage with minimal setup"""
    result = run_evolution(
        initial_program='process_string.py',
        evaluator='string_evaluator.py',
        iterations=10
    )

    print(f"Best score: {result.best_score:.4f}")
    print(f"Best code:\n{result.best_code}")


# Example 2: Using code strings instead of files
def example_2_code_string():
    """Evolve code provided as a string"""

    initial_code = """
# EVOLVE-BLOCK-START
def calculate_sum(numbers):
    total = 0
    for n in numbers:
        total += n
    return total
# EVOLVE-BLOCK-END
"""

    def evaluator(program_path):
        """Test if sum function works correctly"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("prog", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        test = [1, 2, 3, 4, 5]
        result = module.calculate_sum(test)
        expected = 15

        return {
            "combined_score": 1.0 if result == expected else 0.0,
            "correct": result == expected
        }

    result = run_evolution(
        initial_program=initial_code,
        evaluator=evaluator,
        iterations=5
    )

    return result


# Example 3: Custom configuration
def example_3_custom_config():
    """Full control with custom configuration"""

    # Create config object
    config = Config()
    config.max_iterations = 20
    config.database.population_size = 50
    config.database.num_islands = 2

    # Configure LLM
    config.llm.models = [
        LLMModelConfig(
            name="gpt-4",
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=2048
        )
    ]

    # Run with custom config
    result = run_evolution(
        initial_program='process_string.py',
        evaluator='string_evaluator.py',
        config=config
    )

    return result


# Example 4: Using evolve_function helper
def example_4_function_helper():
    """Convenient wrapper for evolving functions"""

    def initial_function(arr):
        """Initial slow implementation"""
        result = []
        for item in arr:
            if item > 0:
                result.append(item * 2)
        return result

    # Test cases
    test_cases = [
        ([1, 2, 3], [2, 4, 6]),
        ([0, -1, 5], [10]),
        ([], []),
    ]

    result = evolve_function(
        initial_function,
        test_cases=test_cases,
        iterations=10
    )

    print(f"Evolved function score: {result.best_score:.4f}")
    return result


# Example 5: Accessing detailed results
def example_5_detailed_results():
    """Extract detailed information from evolution"""

    result = run_evolution(
        initial_program='process_string.py',
        evaluator='string_evaluator.py',
        iterations=10,
        output_dir='./evolution_results'
    )

    # Access detailed results
    print(f"Best program ID: {result.best_program.id if result.best_program else 'N/A'}")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Iterations: {result.best_program.iteration_found if result.best_program else 'N/A'}")
    print(f"All metrics: {result.metrics}")
    print(f"Output directory: {result.output_dir}")

    # Save best program
    if result.output_dir:
        best_file = os.path.join(result.output_dir, "best", "best_program.py")
        print(f"Best program saved to: {best_file}")

    return result


# Run examples
if __name__ == "__main__":
    print("="*80)
    print("Python API Examples")
    print("="*80)

    # Uncomment to run examples:
    # example_1_simple()
    # example_2_code_string()
    # example_3_custom_config()
    # example_4_function_helper()
    # example_5_detailed_results()

    print("\nTo run these examples, uncomment them in the main() function")
