"""
Demo script for Quality Calculator

This script demonstrates the usage of the quality_calculator module
with various examples and real-world scenarios.
"""

from datetime import datetime
from quality_calculator import (
    QualityCalculator,
    SolutionQualityMetrics,
    calculate_quality,
    analyze_code_quality,
    detect_code_smells,
    get_quality_calculator
)
from dataclasses import dataclass


@dataclass
class SolutionAttempt:
    """Mock SolutionAttempt for demo purposes."""
    id: str
    problem_id: str
    solution: str
    score: float
    timestamp: datetime


def example_1_basic_usage():
    """Example 1: Basic quality calculation."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Quality Calculation")
    print("="*70)

    calculator = QualityCalculator()

    solution = SolutionAttempt(
        id="sol1",
        problem_id="prob1",
        solution='''def calculate_sum(numbers):
    """Calculate the sum of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total''',
        score=0.8,
        timestamp=datetime.now()
    )

    requirements = [
        "Implement a function to calculate sum",
        "Handle lists of numbers",
        "Return the total"
    ]

    metrics = calculator.calculate_quality(solution, requirements)

    print(f"\nSolution Quality Metrics:")
    print(f"  Correctness:   {metrics.correctness:.2%}")
    print(f"  Completeness:  {metrics.completeness:.2%}")
    print(f"  Efficiency:    {metrics.efficiency:.2%}")
    print(f"  Maintainability: {metrics.maintainability:.2%}")

    overall = calculator.calculate_overall_score(metrics)
    print(f"\nOverall Score: {overall:.2%}")


def example_2_comparing_solutions():
    """Example 2: Comparing multiple solutions."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Comparing Multiple Solutions")
    print("="*70)

    calculator = QualityCalculator()

    requirements = [
        "Implement fibonacci calculator",
        "Use efficient algorithm",
        "Include error handling",
        "Add documentation"
    ]

    # Solution A: Naive recursive approach
    solution_a = SolutionAttempt(
        id="a",
        problem_id="fib",
        solution='''def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)''',
        score=0.4,
        timestamp=datetime.now()
    )

    # Solution B: Dynamic programming
    solution_b = SolutionAttempt(
        id="b",
        problem_id="fib",
        solution='''"""
Fibonacci calculator using dynamic programming.

This module provides an efficient O(n) implementation.
"""

def fibonacci(n: int) -> int:
    """Calculate the nth fibonacci number.

    Args:
        n: Non-negative integer

    Returns:
        The nth fibonacci number

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b''',
        score=0.95,
        timestamp=datetime.now()
    )

    metrics_a = calculator.calculate_quality(solution_a, requirements)
    metrics_b = calculator.calculate_quality(solution_b, requirements)

    print("\nSolution A (Naive Recursive):")
    print(f"  Correctness:    {metrics_a.correctness:.2%}")
    print(f"  Completeness:   {metrics_a.completeness:.2%}")
    print(f"  Efficiency:     {metrics_a.efficiency:.2%}")
    print(f"  Maintainability: {metrics_a.maintainability:.2%}")
    print(f"  Overall:        {calculator.calculate_overall_score(metrics_a):.2%}")

    print("\nSolution B (Dynamic Programming):")
    print(f"  Correctness:    {metrics_b.correctness:.2%}")
    print(f"  Completeness:   {metrics_b.completeness:.2%}")
    print(f"  Efficiency:     {metrics_b.efficiency:.2%}")
    print(f"  Maintainability: {metrics_b.maintainability:.2%}")
    print(f"  Overall:        {calculator.calculate_overall_score(metrics_b):.2%}")


def example_3_code_quality_analysis():
    """Example 3: Detailed code quality analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Detailed Code Quality Analysis")
    print("="*70)

    code = '''"""
Data processing pipeline.

This module processes large datasets efficiently.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process data with validation and transformation."""

    def __init__(self, config: Dict[str, any]):
        """Initialize the processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.processed_count = 0

    def process_batch(self, data: List[Dict]) -> List[Dict]:
        """Process a batch of data records.

        Args:
            data: List of data records

        Returns:
            Processed data records

        Raises:
            ValueError: If data validation fails
        """
        if not data:
            logger.warning("Empty data batch received")
            return []

        results = []
        for record in data:
            try:
                processed = self._process_record(record)
                results.append(processed)
                self.processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process record: {e}")

        return results

    def _process_record(self, record: Dict) -> Dict:
        """Process a single record.

        Args:
            record: Input record

        Returns:
            Processed record
        """
        # Transform record
        result = {
            "id": record.get("id"),
            "value": record.get("value", 0) * 2,
            "processed": True
        }
        return result


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate statistical measures.

    Args:
        data: List of numerical values

    Returns:
        Dictionary with mean, median, std
    """
    if not data:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}

    n = len(data)
    mean = sum(data) / n
    median = sorted(data)[n // 2]

    return {"mean": mean, "median": median, "std": 0.0}
'''

    analysis = analyze_code_quality(code)

    print("\nCode Quality Analysis:")
    print(f"  Complexity Score:     {analysis.complexity_score:.2%}")
    print(f"  Documentation Score:  {analysis.documentation_score:.2%}")
    print(f"  Naming Score:         {analysis.naming_score:.2%}")
    print(f"  Structure Score:      {analysis.structure_score:.2%}")

    print("\nCode Smells Detected:")
    if analysis.code_smells:
        for smell in analysis.code_smells:
            print(f"  - {smell}")
    else:
        print("  No code smells detected!")

    print("\nImprovement Suggestions:")
    if analysis.suggestions:
        for suggestion in analysis.suggestions:
            print(f"  - {suggestion}")
    else:
        print("  No suggestions - code looks good!")


def example_4_code_smell_detection():
    """Example 4: Code smell detection."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Code Smell Detection")
    print("="*70)

    bad_code = '''def BAD(x):
    y=0
    for i in range(100):
        for j in range(100):
            for k in range(100):
                if x==42:
                    y=y+1
                else:
                    try:
                        y=y-1
                    except:
                        pass
    print("DEBUG:",y)
    return y


GLOBAL_VAR = []  # This is bad


def function_with_too_many_parameters(a, b, c, d, e, f, g, h):
    pass
'''

    smells = detect_code_smells(bad_code)

    print("\nDetected Code Smells:")
    for smell in smells:
        print(f"  [X] {smell}")

    print(f"\nTotal: {len(smells)} code smells detected")


def example_5_custom_weights():
    """Example 5: Using custom weights for scoring."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Weights for Quality Scoring")
    print("="*70)

    # Default weights
    default_calculator = QualityCalculator()

    # Custom weights (emphasize correctness and efficiency)
    custom_weights = {
        "correctness": 0.50,    # High emphasis on correctness
        "completeness": 0.15,
        "efficiency": 0.25,     # High emphasis on efficiency
        "maintainability": 0.10
    }
    custom_calculator = QualityCalculator(weights=custom_weights)

    solution = SolutionAttempt(
        id="custom1",
        problem_id="prob1",
        solution='''def quick_sort(arr):
    """Sort array using quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)''',
        score=0.85,
        timestamp=datetime.now()
    )

    requirements = ["Implement efficient sorting", "Use divide and conquer"]

    metrics = default_calculator.calculate_quality(solution, requirements)

    default_overall = default_calculator.calculate_overall_score(metrics)
    custom_overall = custom_calculator.calculate_overall_score(metrics)

    print("\nSolution Metrics:")
    print(f"  Correctness:     {metrics.correctness:.2%}")
    print(f"  Completeness:    {metrics.completeness:.2%}")
    print(f"  Efficiency:      {metrics.efficiency:.2%}")
    print(f"  Maintainability: {metrics.maintainability:.2%}")

    print("\nOverall Scores:")
    print(f"  Default Weights: {default_overall:.2%}")
    print(f"  Custom Weights:  {custom_overall:.2%}")

    print("\nCustom Weight Configuration:")
    for key, value in custom_weights.items():
        print(f"  {key}: {value:.2%}")


def example_6_convenience_functions():
    """Example 6: Using convenience functions."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Convenience Functions")
    print("="*70)

    solution = SolutionAttempt(
        id="conv1",
        problem_id="prob1",
        solution='''def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"''',
        score=0.7,
        timestamp=datetime.now()
    )

    requirements = ["Create greeting function"]

    # Use convenience functions
    metrics = calculate_quality(solution, requirements)

    print("\nQuick Quality Calculation:")
    print(f"  Overall Score: {get_quality_calculator().calculate_overall_score(metrics):.2%}")

    print("\nQuick Code Analysis:")
    analysis = analyze_code_quality(solution.solution)
    print(f"  Documentation: {analysis.documentation_score:.2%}")
    print(f"  Naming: {analysis.naming_score:.2%}")


def example_7_edge_cases():
    """Example 7: Handling edge cases."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Edge Cases")
    print("="*70)

    calculator = QualityCalculator()

    # Empty solution
    empty_solution = SolutionAttempt(
        id="empty",
        problem_id="prob1",
        solution="",
        score=0.0,
        timestamp=datetime.now()
    )

    print("\nEmpty Solution:")
    metrics = calculator.calculate_quality(empty_solution, ["requirement1"])
    print(f"  All metrics should be 0: {metrics.to_dict()}")

    # Very long solution
    long_solution = SolutionAttempt(
        id="long",
        problem_id="prob1",
        solution="\n".join([f"# Line {i}" for i in range(1000)]),
        score=0.5,
        timestamp=datetime.now()
    )

    print("\nVery Long Solution (1000 lines):")
    metrics = calculator.calculate_quality(long_solution, ["Have some content"])
    print(f"  Completeness: {metrics.completeness:.2%}")

    # Solution with syntax errors
    invalid_solution = SolutionAttempt(
        id="invalid",
        problem_id="prob1",
        solution="def broken(\n    # Missing closing parenthesis",
        score=0.0,
        timestamp=datetime.now()
    )

    print("\nSolution with Syntax Errors:")
    try:
        metrics = calculator.calculate_quality(invalid_solution, ["requirement1"])
        print(f"  Handled gracefully: Correctness={metrics.correctness:.2%}")
    except Exception as e:
        print(f"  Error handled: {type(e).__name__}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("QUALITY CALCULATOR DEMONSTRATION")
    print("="*70)
    print("\nThis demo showcases the features of the quality_calculator module")
    print("including quality metrics, code analysis, and code smell detection.")

    examples = [
        example_1_basic_usage,
        example_2_comparing_solutions,
        example_3_code_quality_analysis,
        example_4_code_smell_detection,
        example_5_custom_weights,
        example_6_convenience_functions,
        example_7_edge_cases
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n[!] Error in {example.__name__}: {e}")

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nFor more information, see the quality_calculator module documentation.")


if __name__ == "__main__":
    main()
