"""
Example usage of the Gauntlet Benchmark Suite

This script demonstrates various ways to use the benchmarking suite
for performance testing and regression detection.

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import sys
import json
from pathlib import Path

# Add benchmark module to path
sys.path.insert(0, str(Path(__file__).parent))

from gauntlet_benchmarks import (
    GauntletBenchmarkSuite,
    BaselineMetrics,
    PerformanceTargets,
    BenchmarkStatus
)


def example_basic_usage():
    """Example 1: Basic usage with defaults"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create suite with default settings
    suite = GauntletBenchmarkSuite()

    # Run all benchmarks
    results = suite.run_all_benchmarks()

    # Save to JSON
    results.to_json("example_results.json")

    # Print summary
    print(f"\nResults: {results.summary}")
    print(f"Grade: {results.summary['performance_grade']}")

    return results


def example_custom_baselines():
    """Example 2: Custom baseline metrics"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Baselines")
    print("=" * 60)

    # Define custom baselines (e.g., after optimizations)
    custom_baselines = BaselineMetrics(
        ml_optimizer_iterations_per_second=60.0,  # Improved from 50.0
        prediction_latency_ms=80.0,  # Improved from 100.0
        training_episodes_per_second=12.0,  # Improved from 10.0
        planning_time_ms=150.0  # Improved from 200.0
    )

    # Create suite with custom baselines
    suite = GauntletBenchmarkSuite(baselines=custom_baselines)

    # Run benchmarks
    results = suite.run_all_benchmarks()

    print(f"\nWith improved baselines: {results.summary['performance_grade']}")

    return results


def example_custom_targets():
    """Example 3: Custom performance targets"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Performance Targets")
    print("=" * 60)

    # Stricter targets for production
    strict_targets = PerformanceTargets(
        ml_optimizer_speed_tolerance=0.10,  # Only 10% slower allowed
        prediction_latency_tolerance=0.15,  # Only 15% slower allowed
        training_speed_tolerance=0.15,  # Only 15% slower allowed
        min_prediction_accuracy=0.80,  # Higher accuracy required
        min_improvement_percent=15.0  # Higher improvement required
    )

    suite = GauntletBenchmarkSuite(targets=strict_targets, num_runs=5)
    results = suite.run_all_benchmarks()

    print(f"\nWith strict targets: {results.summary['overall_status']}")

    return results


def example_specific_component():
    """Example 4: Benchmark specific component"""
    print("\n" + "=" * 60)
    print("Example 4: Specific Component Benchmarking")
    print("=" * 60)

    suite = GauntletBenchmarkSuite(num_runs=5)

    # Only benchmark ML Optimizer
    try:
        from ml_optimizer import MLBasedGauntletOptimizer, Objective, OptimizationStrategy

        optimizer = MLBasedGauntletOptimizer(
            strategy=OptimizationStrategy.Q_LEARNING
        )

        # Run specific benchmark
        suite._benchmark_ml_optimizer_speed(optimizer)
        suite._benchmark_ml_optimizer_memory(optimizer)

        # Get results for ML Optimizer only
        ml_results = [r for r in suite.results if r.component == "ml_optimizer"]

        print(f"\nML Optimizer Results:")
        for result in ml_results:
            print(f"  {result.metric_name}: {result.value:.2f} {result.unit} - {result.status.value}")

    except ImportError:
        print("ML Optimizer not available for demonstration")


def example_regression_detection():
    """Example 5: Performance regression detection"""
    print("\n" + "=" * 60)
    print("Example 5: Regression Detection")
    print("=" * 60)

    # Load previous results
    try:
        with open("example_results.json", "r") as f:
            previous_data = json.load(f)
    except FileNotFoundError:
        print("No previous results found. Run Example 1 first.")
        return

    # Extract previous baselines
    prev_results = previous_data.get("results", [])

    # Run new benchmarks
    suite = GauntletBenchmarkSuite(num_runs=5)
    new_results = suite.run_all_benchmarks()

    # Compare results
    print("\nRegression Analysis:")

    for new_result in new_results.results:
        # Find matching previous result
        prev_result = next(
            (r for r in prev_results if r["metric_name"] == new_result.metric_name),
            None
        )

        if prev_result:
            change = ((new_result.value - prev_result["value"]) / prev_result["value"]) * 100

            if change > 10:
                print(f"  ⚠ REGRESSION: {new_result.metric_name}")
                print(f"    Previous: {prev_result['value']:.2f}")
                print(f"    Current:  {new_result.value:.2f}")
                print(f"    Change:   +{change:.1f}%")
            elif change < -10:
                print(f"  [OK] IMPROVEMENT: {new_result.metric_name}")
                print(f"    Previous: {prev_result['value']:.2f}")
                print(f"    Current:  {new_result.value:.2f}")
                print(f"    Change:   {change:.1f}%")


def example_ci_integration():
    """Example 6: CI/CD integration pattern"""
    print("\n" + "=" * 60)
    print("Example 6: CI/CD Integration")
    print("=" * 60)

    # Run benchmarks
    suite = GauntletBenchmarkSuite(num_runs=5)
    results = suite.run_all_benchmarks()

    # Save for CI artifacts
    results.to_json("ci_benchmark_results.json")

    # Determine exit code
    if results.failed > 0:
        print(f"\n[FAIL] CI FAILED: {results.failed} benchmarks failed")
        print("Failing checks...")

        # List failures
        failures = [r for r in results.results if r.status == BenchmarkStatus.FAIL]
        for failure in failures:
            print(f"  - {failure.name}: {failure.value:.2f} vs baseline {failure.baseline:.2f}")

        return 1  # Non-zero exit code for CI
    else:
        print(f"\n[OK] CI PASSED: All {results.passed} benchmarks passed")
        return 0


def example_load_baseline_from_config():
    """Example 7: Load baselines from config file"""
    print("\n" + "=" * 60)
    print("Example 7: Load Baselines from Config")
    print("=" * 60)

    try:
        with open("baseline_config.json", "r") as f:
            config = json.load(f)

        baseline_data = config["baseline_metrics"]

        # Create BaselineMetrics from config
        baselines = BaselineMetrics(
            ml_optimizer_iterations_per_second=baseline_data["ml_optimizer"]["iterations_per_second"],
            ml_optimizer_memory_mb=baseline_data["ml_optimizer"]["memory_mb"],
            ml_optimizer_convergence_rate=baseline_data["ml_optimizer"]["convergence_rate"],
            ml_optimizer_improvement_percent=baseline_data["ml_optimizer"]["improvement_percent"],
            prediction_latency_ms=baseline_data["predictive_executor"]["prediction_latency_ms"],
            prediction_accuracy=baseline_data["predictive_executor"]["prediction_accuracy"],
            cost_savings_percent=baseline_data["predictive_executor"]["cost_savings_percent"],
            training_episodes_per_second=baseline_data["adaptive_learner"]["training_episodes_per_second"],
            training_memory_mb=baseline_data["adaptive_learner"]["training_memory_mb"],
            loss_convergence_rate=baseline_data["adaptive_learner"]["loss_convergence_rate"],
            prediction_accuracy_learner=baseline_data["adaptive_learner"]["prediction_accuracy_learner"],
            planning_time_ms=baseline_data["intelligent_orchestrator"]["planning_time_ms"],
            execution_time_vs_baseline=baseline_data["intelligent_orchestrator"]["execution_time_vs_baseline"],
            resource_utilization=baseline_data["intelligent_orchestrator"]["resource_utilization"]
        )

        suite = GauntletBenchmarkSuite(baselines=baselines, num_runs=5)
        results = suite.run_all_benchmarks()

        print(f"\nUsing baselines from config: {results.summary['performance_grade']}")

    except FileNotFoundError:
        print("baseline_config.json not found")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "GAUNTLET BENCHMARK SUITE EXAMPLES" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Baselines", example_custom_baselines),
        ("Custom Targets", example_custom_targets),
        ("Specific Component", example_specific_component),
        ("Regression Detection", example_regression_detection),
        ("CI/CD Integration", example_ci_integration),
        ("Load from Config", example_load_baseline_from_config)
    ]

    print("Available Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. Run all examples")
    print("\n")

    choice = input("Select example to run (0-7): ").strip()

    if choice == "0":
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\nError in {name}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        name, func = examples[int(choice) - 1]
        try:
            func()
        except Exception as e:
            print(f"\nError: {e}")
    else:
        print("Invalid choice. Running basic usage example...")
        example_basic_usage()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
