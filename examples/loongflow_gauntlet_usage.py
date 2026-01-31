"""
Example Usage of LoongFlow Gauntlet Adapter

Demonstrates various ways to use the LoongFlow gauntlet evaluator
for Round 1 screening in the OpenEvolve gauntlet system.
"""

import asyncio
from openevolve.gauntlets import (
    LoongFlowGauntletEvaluator,
    LoongFlowGauntletConfig,
    GauntletEvaluationResult,
)


async def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create configuration
    config = LoongFlowGauntletConfig(
        quality_threshold=0.6,
        confidence_threshold=0.7
    )

    # Initialize evaluator
    evaluator = LoongFlowGauntletEvaluator(config)

    # Evaluate a solution
    result = await evaluator.evaluate_solution(
        solution="def solve(): return 42",
        problem="Create a function that returns the answer to life, the universe, and everything",
        domain="code"
    )

    # Display results
    print(f"\nSolution: {result.solution[:50]}...")
    print(f"Passed: {result.passed}")
    print(f"Overall Score: {result.overall_score:.1%}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"\nScore Breakdown:")
    print(f"  - Correctness: {result.correctness_score:.1%}")
    print(f"  - Efficiency: {result.efficiency_score:.1%}")
    print(f"  - Robustness: {result.robustness_score:.1%}")
    print(f"  - Creativity: {result.creativity_score:.1%}")
    print(f"\nEvaluation Time: {result.evaluation_time:.2f}s")
    print(f"\nFeedback:\n{result.feedback}")

    return result


async def example_batch_evaluation():
    """Batch evaluation example."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Evaluation")
    print("=" * 60)

    config = LoongFlowGauntletConfig(
        quality_threshold=0.5,
        max_evaluations=30
    )

    evaluator = LoongFlowGauntletEvaluator(config)

    # Multiple solutions to evaluate
    solutions = [
        "def solution1(): return 1",
        "def solution2(): return 2",
        "def solution3(): return 3",
        "def solution4(): return 4",
        "def solution5(): return 5"
    ]

    print(f"\nEvaluating {len(solutions)} solutions...")

    # Batch evaluate
    results = await evaluator.evaluate_batch(
        solutions=solutions,
        problem="Return a number",
        domain="code"
    )

    # Display results
    print(f"\n{'Solution':<12} {'Score':<10} {'Passed':<8} {'Time':<10}")
    print("-" * 40)
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"Solution {i:<4} {result.overall_score:<10.1%} {status:<8} {result.evaluation_time:<10.2f}s")

    # Summary statistics
    passed_count = sum(1 for r in results if r.passed)
    avg_score = sum(r.overall_score for r in results) / len(results)
    total_time = sum(r.evaluation_time for r in results)

    print(f"\nSummary:")
    print(f"  Passed: {passed_count}/{len(results)}")
    print(f"  Average Score: {avg_score:.1%}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg Time: {total_time/len(results):.2f}s")

    return results


async def example_strict_evaluation():
    """Strict evaluation example."""
    print("\n" + "=" * 60)
    print("Example 3: Strict Evaluation (High Quality Filter)")
    print("=" * 60)

    config = LoongFlowGauntletConfig(
        quality_threshold=0.8,
        confidence_threshold=0.8,
        correctness_weight=0.5,
        efficiency_weight=0.2,
        robustness_weight=0.2,
        creativity_weight=0.1
    )

    evaluator = LoongFlowGauntletEvaluator(config)

    result = await evaluator.evaluate_solution(
        solution="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        problem="Implement Fibonacci sequence",
        domain="code"
    )

    print(f"\nStrict Mode Results:")
    print(f"Passed: {result.passed}")
    print(f"Score: {result.overall_score:.1%}")
    print(f"Required: {config.quality_threshold:.1%}")

    if not result.passed:
        print(f"\n❌ Solution failed strict quality requirements")
        print(f"   Score: {result.overall_score:.1%} < {config.quality_threshold:.1%}")

    return result


async def example_creativity_focused():
    """Creativity-focused evaluation example."""
    print("\n" + "=" * 60)
    print("Example 4: Creativity-Focused Evaluation")
    print("=" * 60)

    config = LoongFlowGauntletConfig(
        quality_threshold=0.6,
        correctness_weight=0.3,
        efficiency_weight=0.2,
        robustness_weight=0.2,
        creativity_weight=0.3  # Higher creativity weight
    )

    evaluator = LoongFlowGauntletEvaluator(config)

    # Creative solution using generators
    creative_solution = """
def fibonacci_generator(n):
    '''Generate Fibonacci numbers using a generator'''
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

def get_nth_fibonacci(n):
    return list(fibonacci_generator(n))[-1]
"""

    result = await evaluator.evaluate_solution(
        solution=creative_solution,
        problem="Implement Fibonacci sequence creatively",
        domain="code"
    )

    print(f"\nCreativity-Focused Results:")
    print(f"Overall Score: {result.overall_score:.1%}")
    print(f"Creativity Score: {result.creativity_score:.1%}")
    print(f"Correctness Score: {result.correctness_score:.1%}")

    print(f"\nCreativity Feedback:")
    if result.strengths:
        print("Strengths:")
        for strength in result.strengths:
            print(f"  ✓ {strength}")

    return result


async def example_math_problem():
    """Math problem evaluation example."""
    print("\n" + "=" * 60)
    print("Example 5: Math Problem Evaluation")
    print("=" * 60)

    config = LoongFlowGauntletConfig(
        quality_threshold=0.7,
        domain="math"
    )

    evaluator = LoongFlowGauntletEvaluator(config)

    # Math solution
    math_solution = """
def optimize_packing(n_circles):
    '''Optimize circle packing in unit square using hexagonal pattern'''
    import math

    # Calculate optimal radius
    radius = 1.0 / (math.ceil(math.sqrt(n_circles)) + 1)

    # Generate hexagonal positions
    positions = []
    for i in range(n_circles):
        row = i // int(math.sqrt(n_circles))
        col = i % int(math.sqrt(n_circles))
        x = (col + 1) * radius
        y = (row + 1) * radius
        positions.append((x, y))

    return positions, radius, sum([radius] * n_circles)
"""

    result = await evaluator.evaluate_solution(
        solution=math_solution,
        problem="Optimize packing of 26 circles in a unit square to maximize sum of radii",
        domain="math"
    )

    print(f"\nMath Problem Results:")
    print(f"Passed: {result.passed}")
    print(f"Overall Score: {result.overall_score:.1%}")
    print(f"Correctness: {result.correctness_score:.1%}")
    print(f"Creativity: {result.creativity_score:.1%}")
    print(f"PES Iterations: {result.pes_iterations}")
    print(f"PES Evaluations: {result.pes_evaluations}")

    return result


async def example_error_handling():
    """Error handling example."""
    print("\n" + "=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)

    config = LoongFlowGauntletConfig()
    evaluator = LoongFlowGauntletEvaluator(config)

    # Check availability
    if not evaluator.is_available():
        print("⚠️  LoongFlow not available, using fallback mode")

    # Try evaluating
    result = await evaluator.evaluate_solution(
        solution="def test(): pass",
        problem="Test problem",
        domain="general"
    )

    # Handle gracefully
    if result.passed:
        print("✅ Evaluation succeeded")
    else:
        print("❌ Evaluation failed or solution rejected")
        if "error" in result.feedback.lower():
            print(f"   Error: {result.feedback}")

    return result


async def example_custom_weights():
    """Custom scoring weights example."""
    print("\n" + "=" * 60)
    print("Example 7: Custom Scoring Weights")
    print("=" * 60)

    # Different weight configurations for different use cases
    configs = {
        "Correctness-Focused": LoongFlowGauntletConfig(
            correctness_weight=0.7,
            efficiency_weight=0.1,
            robustness_weight=0.1,
            creativity_weight=0.1
        ),
        "Balanced": LoongFlowGauntletConfig(
            correctness_weight=0.25,
            efficiency_weight=0.25,
            robustness_weight=0.25,
            creativity_weight=0.25
        ),
        "Efficiency-Focused": LoongFlowGauntletConfig(
            correctness_weight=0.2,
            efficiency_weight=0.6,
            robustness_weight=0.1,
            creativity_weight=0.1
        )
    }

    solution = "def quicksort(arr): return sorted(arr)"
    problem = "Sort an array"

    print("\nComparing different weight configurations:\n")
    print(f"{'Configuration':<20} {'Score':<10} {'Correct':<10} {'Efficiency':<10}")
    print("-" * 50)

    for name, config in configs.items():
        evaluator = LoongFlowGauntletEvaluator(config)
        result = await evaluator.evaluate_solution(solution, problem, "code")

        print(f"{name:<20} {result.overall_score:<10.1%} {result.correctness_score:<10.1%} {result.efficiency_score:<10.1%}")


async def example_result_serialization():
    """Result serialization example."""
    print("\n" + "=" * 60)
    print("Example 8: Result Serialization")
    print("=" * 60)

    config = LoongFlowGauntletConfig()
    evaluator = LoongFlowGauntletEvaluator(config)

    result = await evaluator.evaluate_solution(
        solution="def example(): return 42",
        problem="Test problem",
        domain="code"
    )

    # Serialize to dict
    result_dict = result.to_dict()

    print(f"\nSerialized Result Keys: {list(result_dict.keys())}")
    print(f"Solution (truncated): {result_dict['solution'][:50]}...")
    print(f"Passed: {result_dict['passed']}")
    print(f"Overall Score: {result_dict['overall_score']:.1%}")
    print(f"Timestamp: {result_dict['timestamp']}")

    # Deserialize from dict
    restored_result = GauntletEvaluationResult.from_dict(result_dict)

    print(f"\nRestored Result:")
    print(f"Matches original: {restored_result.solution == result.solution}")
    print(f"Score matches: {restored_result.overall_score == result.overall_score}")

    return result_dict


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LOONGFLOW GAUNTLET ADAPTER - USAGE EXAMPLES")
    print("=" * 60)

    try:
        # Run examples
        await example_basic_usage()
        await example_batch_evaluation()
        await example_strict_evaluation()
        await example_creativity_focused()
        await example_math_problem()
        await example_error_handling()
        await example_custom_weights()
        await example_result_serialization()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
