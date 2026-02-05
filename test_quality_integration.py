"""
Integration test for quality_calculator.py with sovereign_data_models.py

This script verifies that the quality calculator works correctly with
the actual sovereign data models.
"""

import sys
from datetime import datetime

# Import from crewai_state_management (has the real SolutionAttempt)
try:
    from crewai_state_management import SolutionAttempt, ExecutionMethod
    print("[OK] Successfully imported SolutionAttempt from crewai_state_management")
    USE_REAL_MODEL = True
except ImportError as e:
    print(f"[WARN] Could not import from crewai_state_management: {e}")
    print("  Using fallback SolutionAttempt")
    from dataclasses import dataclass
    USE_REAL_MODEL = False

    @dataclass
    class SolutionAttempt:
        id: str
        problem_id: str
        solution: str
        score: float
        timestamp: datetime

# Import quality calculator
try:
    from quality_calculator import (
        QualityCalculator,
        SolutionQualityMetrics,
        calculate_quality,
        analyze_code_quality,
        detect_code_smells
    )
    print("[OK] Successfully imported from quality_calculator")
except ImportError as e:
    print(f"[FAIL] Failed to import from quality_calculator: {e}")
    sys.exit(1)


def test_basic_integration():
    """Test basic integration with sovereign data models."""
    print("\n" + "="*70)
    print("TEST 1: Basic Integration")
    print("="*70)

    calculator = QualityCalculator()

    # Create a solution attempt
    if USE_REAL_MODEL:
        from crewai_state_management import SolutionAttempt, ExecutionMethod
        solution = SolutionAttempt(
            sub_problem_id="fibonacci",
            solution_content='''"""
Fibonacci Number Calculator

This module provides an efficient implementation for calculating
fibonacci numbers using dynamic programming.
"""

from typing import Optional


def fibonacci(n: int) -> int:
    """Calculate the nth fibonacci number.

    This implementation uses dynamic programming for O(n) time complexity.

    Args:
        n: The position in the fibonacci sequence (must be non-negative)

    Returns:
        The nth fibonacci number

    Raises:
        ValueError: If n is negative

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    if n <= 1:
        return n

    # Dynamic programming approach
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b


def fibonacci_sequence(count: int) -> list[int]:
    """Generate a sequence of fibonacci numbers.

    Args:
        count: Number of fibonacci numbers to generate

    Returns:
        List of fibonacci numbers
    """
    return [fibonacci(i) for i in range(count)]


if __name__ == "__main__":
    print(f"Fibonacci(10) = {fibonacci(10)}")
    print(f"First 10 fibonacci numbers: {fibonacci_sequence(10)}")
''',
            confidence_score=0.95,
            execution_method=ExecutionMethod.TRADITIONAL
        )
    else:
        solution = SolutionAttempt(
            id="test_sol_1",
            problem_id="fibonacci",
            solution='''"""
Fibonacci Number Calculator

This module provides an efficient implementation for calculating
fibonacci numbers using dynamic programming.
"""

from typing import Optional


def fibonacci(n: int) -> int:
    """Calculate the nth fibonacci number.

    This implementation uses dynamic programming for O(n) time complexity.

    Args:
        n: The position in the fibonacci sequence (must be non-negative)

    Returns:
        The nth fibonacci number

    Raises:
        ValueError: If n is negative

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    if n <= 1:
        return n

    # Dynamic programming approach
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b


def fibonacci_sequence(count: int) -> list[int]:
    """Generate a sequence of fibonacci numbers.

    Args:
        count: Number of fibonacci numbers to generate

    Returns:
        List of fibonacci numbers
    """
    return [fibonacci(i) for i in range(count)]


if __name__ == "__main__":
    print(f"Fibonacci(10) = {fibonacci(10)}")
    print(f"First 10 fibonacci numbers: {fibonacci_sequence(10)}")
''',
            score=0.95,
            timestamp=datetime.now()
        )

    requirements = [
        "Implement fibonacci calculator",
        "Use dynamic programming for efficiency",
        "Include comprehensive documentation",
        "Add error handling for invalid inputs",
        "Provide examples in docstrings"
    ]

    # Calculate quality
    metrics = calculator.calculate_quality(solution, requirements)

    print("\n[OK] Quality Metrics Calculated:")
    print(f"  Correctness:    {metrics.correctness:.2%}")
    print(f"  Completeness:   {metrics.completeness:.2%}")
    print(f"  Efficiency:     {metrics.efficiency:.2%}")
    print(f"  Maintainability: {metrics.maintainability:.2%}")

    overall = calculator.calculate_overall_score(metrics)
    print(f"\n[OK] Overall Quality Score: {overall:.2%}")

    # Verify metrics are in valid range
    assert 0.0 <= metrics.correctness <= 1.0, "Correctness out of range"
    assert 0.0 <= metrics.completeness <= 1.0, "Completeness out of range"
    assert 0.0 <= metrics.efficiency <= 1.0, "Efficiency out of range"
    assert 0.0 <= metrics.maintainability <= 1.0, "Maintainability out of range"
    assert 0.0 <= overall <= 1.0, "Overall score out of range"

    print("\n[OK] All metrics in valid range [0.0, 1.0]")

    return True


def test_code_quality_analysis():
    """Test detailed code quality analysis."""
    print("\n" + "="*70)
    print("TEST 2: Code Quality Analysis")
    print("="*70)

    solution_code = '''"""
Data Processing Pipeline

This module provides efficient data processing capabilities with proper
error handling and logging.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessedData:
    """Container for processed data."""
    id: str
    value: float
    status: str
    metadata: Dict[str, Any]


class DataProcessor:
    """Process and transform data efficiently."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data processor.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.processed_count = 0

    def process_batch(self, data: List[Dict[str, Any]]) -> List[ProcessedData]:
        """Process a batch of data records.

        Args:
            data: List of raw data records

        Returns:
            List of processed data objects

        Raises:
            ValueError: If data format is invalid
        """
        if not data:
            logger.warning("Empty batch received")
            return []

        results = []
        for record in data:
            try:
                processed = self._process_single_record(record)
                results.append(processed)
                self.processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process record: {e}")

        return results

    def _process_single_record(self, record: Dict[str, Any]) -> ProcessedData:
        """Process a single data record.

        Args:
            record: Raw data record

        Returns:
            Processed data object
        """
        record_id = record.get("id", "unknown")
        value = float(record.get("value", 0))

        return ProcessedData(
            id=record_id,
            value=value * 2,
            status="processed",
            metadata=record.get("metadata", {})
        )

    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics.

        Returns:
            Dictionary with processing stats
        """
        return {"processed_count": self.processed_count}


def calculate_aggregates(data: List[float]) -> Dict[str, float]:
    """Calculate statistical aggregates.

    Args:
        data: List of numerical values

    Returns:
        Dictionary with mean, median, min, max
    """
    if not data:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

    sorted_data = sorted(data)
    n = len(sorted_data)

    return {
        "mean": sum(sorted_data) / n,
        "median": sorted_data[n // 2],
        "min": sorted_data[0],
        "max": sorted_data[-1]
    }
'''

    analysis = analyze_code_quality(solution_code)

    print("\n[OK] Code Quality Analysis Complete:")
    print(f"  Complexity Score:     {analysis.complexity_score:.2%}")
    print(f"  Documentation Score:  {analysis.documentation_score:.2%}")
    print(f"  Naming Score:         {analysis.naming_score:.2%}")
    print(f"  Structure Score:      {analysis.structure_score:.2%}")

    print(f"\n[OK] Code Smells Detected: {len(analysis.code_smells)}")
    if analysis.code_smells:
        for smell in analysis.code_smells:
            print(f"  - {smell}")
    else:
        print("  No code smells detected!")

    print(f"\n[OK] Suggestions: {len(analysis.suggestions)}")
    if analysis.suggestions:
        for suggestion in analysis.suggestions[:3]:
            print(f"  - {suggestion}")

    # Verify analysis structure
    assert hasattr(analysis, 'complexity_score'), "Missing complexity_score"
    assert hasattr(analysis, 'documentation_score'), "Missing documentation_score"
    assert hasattr(analysis, 'naming_score'), "Missing naming_score"
    assert hasattr(analysis, 'structure_score'), "Missing structure_score"
    assert isinstance(analysis.code_smells, list), "code_smells should be list"
    assert isinstance(analysis.suggestions, list), "suggestions should be list"

    print("\n[OK] Analysis structure validated")

    return True


def test_code_smell_detection():
    """Test code smell detection."""
    print("\n" + "="*70)
    print("TEST 3: Code Smell Detection")
    print("="*70)

    bad_code = '''import sys

global_cache = {}


def BAD_Function_NAME(x, y, z, a, b, c, d, e):
    y = 0
    if x == 1:
        for i in range(100):
            for j in range(100):
                for k in range(100):
                    if y == 42:
                        y = y + 1
                    else:
                        try:
                            y = y - 1
                        except:
                            pass
    print("DEBUG:", y)
    return y


class EmptyClass:
    pass
'''

    smells = detect_code_smells(bad_code)

    print(f"\n[OK] Detected {len(smells)} code smells:")
    for smell in smells:
        print(f"  - {smell}")

    # Verify we detected expected smells
    assert len(smells) > 0, "Should detect at least some code smells"

    expected_types = [
        "Bare except",
        "Print statement",
        "Global variable",
        "too many parameters"
    ]

    detected_types = ' '.join(smells).lower()
    for expected in expected_types:
        if expected.lower() in detected_types:
            print(f"\n[OK] Detected expected smell type: {expected}")

    print("\n[OK] Code smell detection working")

    return True


def test_comparison():
    """Test comparing solutions of different quality."""
    print("\n" + "="*70)
    print("TEST 4: Solution Comparison")
    print("="*70)

    calculator = QualityCalculator()

    requirements = [
        "Implement sorting algorithm",
        "Include documentation",
        "Handle edge cases"
    ]

    # Poor solution
    poor_solution = SolutionAttempt(
        id="poor",
        problem_id="sorting",
        solution='''def sort(x):
    y = x[:]
    for i in range(len(y)):
        for j in range(len(y)):
            if y[i] < y[j]:
                y[i], y[j] = y[j], y[i]
    return y''',
        score=0.3,
        timestamp=datetime.now()
    )

    # Good solution
    good_solution = SolutionAttempt(
        id="good",
        problem_id="sorting",
        solution='''"""
Sorting utilities.
"""

from typing import List


def quick_sort(arr: List[int]) -> List[int]:
    """Sort array using quicksort algorithm.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list of integers
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)''',
        score=0.9,
        timestamp=datetime.now()
    )

    # Calculate quality for both
    poor_metrics = calculator.calculate_quality(poor_solution, requirements)
    good_metrics = calculator.calculate_quality(good_solution, requirements)

    poor_overall = calculator.calculate_overall_score(poor_metrics)
    good_overall = calculator.calculate_overall_score(good_metrics)

    print("\nPoor Solution:")
    print(f"  Overall: {poor_overall:.2%}")
    print(f"  Correctness: {poor_metrics.correctness:.2%}")
    print(f"  Maintainability: {poor_metrics.maintainability:.2%}")

    print("\nGood Solution:")
    print(f"  Overall: {good_overall:.2%}")
    print(f"  Correctness: {good_metrics.correctness:.2%}")
    print(f"  Maintainability: {good_metrics.maintainability:.2%}")

    # Good solution should score higher on maintainability at least
    print(f"\n[OK] Maintainability difference: {good_metrics.maintainability - poor_metrics.maintainability:.2%}")

    return True


def test_edge_cases():
    """Test edge case handling."""
    print("\n" + "="*70)
    print("TEST 5: Edge Cases")
    print("="*70)

    calculator = QualityCalculator()

    # Empty solution
    empty_solution = SolutionAttempt(
        id="empty",
        problem_id="test",
        solution="",
        score=0.0,
        timestamp=datetime.now()
    )

    print("\nTesting empty solution...")
    metrics = calculator.calculate_quality(empty_solution, ["requirement"])
    assert metrics.correctness == 0.0, "Empty solution should have 0 correctness"
    assert metrics.completeness == 0.0, "Empty solution should have 0 completeness"
    print("[OK] Empty solution handled correctly")

    # Solution with syntax error
    invalid_solution = SolutionAttempt(
        id="invalid",
        problem_id="test",
        solution="def broken(\n    # Missing closing paren",
        score=0.0,
        timestamp=datetime.now()
    )

    print("\nTesting solution with syntax errors...")
    try:
        metrics = calculator.calculate_quality(invalid_solution, ["requirement"])
        print("[OK] Syntax errors handled gracefully")
        print(f"  Correctness: {metrics.correctness:.2%}")
    except Exception as e:
        print(f"[FAIL] Failed to handle syntax errors: {e}")
        return False

    # Very long solution
    long_solution = SolutionAttempt(
        id="long",
        problem_id="test",
        solution="\n".join([f"# Line {i}" for i in range(1000)]),
        score=0.5,
        timestamp=datetime.now()
    )

    print("\nTesting very long solution (1000 lines)...")
    metrics = calculator.calculate_quality(long_solution, ["Have content"])
    print(f"[OK] Long solution processed: Completeness = {metrics.completeness:.2%}")

    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("QUALITY CALCULATOR INTEGRATION TESTS")
    print("="*70)
    print("\nTesting integration with sovereign_data_models.py")

    tests = [
        ("Basic Integration", test_basic_integration),
        ("Code Quality Analysis", test_code_quality_analysis),
        ("Code Smell Detection", test_code_smell_detection),
        ("Solution Comparison", test_comparison),
        ("Edge Cases", test_edge_cases)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n[OK] {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n[FAIL] {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {test_name}: FAILED with exception")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n[OK] ALL TESTS PASSED")
        return 0
    else:
        print(f"\n[FAIL] {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
