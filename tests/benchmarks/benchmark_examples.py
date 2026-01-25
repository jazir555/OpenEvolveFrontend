"""
Benchmark Usage Examples

This script demonstrates various ways to use the benchmark suite.

Author: OpenEvolve Framework
Date: 2025-01-07
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_engine.engine import KnowledgeEngine
from tests.benchmarks.kg_performance_benchmarks import KnowledgeGraphPerformanceBenchmarks


async def example_1_basic_usage():
    """Example 1: Basic benchmark usage."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Benchmark Usage")
    print("="*60)

    # Initialize
    engine = KnowledgeEngine()
    benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

    # Run a single benchmark
    result = await benchmarks.benchmark_knowledge_addition(
        num_artifacts=500,
        batch_size=25
    )

    # Access results
    if result.success:
        print(f"\n✓ Benchmark completed successfully")
        print(f"  Throughput: {result.metrics['artifacts_per_second']:.2f} artifacts/sec")
        print(f"  Duration: {result.metrics['duration_seconds']:.2f}s")
        print(f"  Memory: {result.metrics['memory_used_gb']:.2f} GB")
    else:
        print(f"\n✗ Benchmark failed: {result.error}")

    await engine.cleanup_kggen_pipeline()


async def example_2_custom_configuration():
    """Example 2: Using custom benchmark parameters."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Benchmark Configuration")
    print("="*60)

    engine = KnowledgeEngine()
    benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

    # Test multiple configurations
    configs = [
        {"num_artifacts": 100, "batch_size": 10},
        {"num_artifacts": 100, "batch_size": 50},
        {"num_artifacts": 100, "batch_size": 100}
    ]

    print("\nTesting different batch sizes:")
    for config in configs:
        result = await benchmarks.benchmark_knowledge_addition(**config)

        if result.success:
            batch_size = config["batch_size"]
            throughput = result.metrics["artifacts_per_second"]
            print(f"  Batch size {batch_size:3d}: {throughput:8.2f} artifacts/sec")

    await engine.cleanup_kggen_pipeline()


async def example_3_comparative_analysis():
    """Example 3: Comparing different scenarios."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Comparative Analysis")
    print("="*60)

    engine = KnowledgeEngine()
    benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

    # Compare different duplicate rates
    print("\nDeduplication performance by duplicate rate:")
    duplicate_rates = [0.1, 0.3, 0.5, 0.7]

    for rate in duplicate_rates:
        result = await benchmarks.benchmark_deduplication(
            num_entities=500,
            duplicate_rate=rate
        )

        if result.success:
            f1 = result.metrics["f1_score"]
            duration = result.metrics["duration_seconds"]
            print(f"  Rate {rate:.1f}: F1={f1:.3f}, Time={duration:.2f}s")

    await engine.cleanup_kggen_pipeline()


async def example_4_scalability_analysis():
    """Example 4: Analyzing scalability."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Scalability Analysis")
    print("="*60)

    engine = KnowledgeEngine()
    benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

    # Test across different scales
    print("\nGraph algorithm scalability:")
    result = await benchmarks.benchmark_graph_algorithms(
        graph_sizes=[100, 250, 500, 1000]
    )

    if result.success:
        print("\n  Size    | Time (s) | Memory (MB)")
        print("  " + "-"*35)

        for size, metrics in result.metrics.items():
            duration = metrics["duration_seconds"]
            memory = metrics["memory_mb"]
            print(f"  {size:7} | {duration:8.2f} | {memory:11.1f}")

    await engine.cleanup_kggen_pipeline()


async def example_5_end_to_end_workflow():
    """Example 5: End-to-end workflow benchmarking."""
    print("\n" + "="*60)
    print("EXAMPLE 5: End-to-End Workflow Benchmarking")
    print("="*60)

    engine = KnowledgeEngine()
    benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

    # Run all workflow scenarios
    result = await benchmarks.benchmark_end_to_end_workflows(
        scenarios=[
            "entity_relationship_workflow",
            "batch_processing_workflow",
            "query_workflow"
        ]
    )

    if result.success:
        print("\nWorkflow Performance Summary:")
        print("-" * 50)

        for scenario, metrics in result.metrics.items():
            if metrics["success"]:
                duration = metrics["duration_seconds"]
                print(f"  {scenario:30} | {duration:6.2f}s")

    await engine.cleanup_kggen_pipeline()


async def example_6_custom_metrics_collection():
    """Example 6: Collecting and analyzing custom metrics."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Metrics Collection")
    print("="*60)

    engine = KnowledgeEngine()
    benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

    # Run multiple benchmarks
    print("\nRunning comprehensive benchmark suite...")

    await benchmarks.benchmark_knowledge_addition(num_artifacts=200)
    await benchmarks.benchmark_knowledge_retrieval(num_queries=50)
    await benchmarks.benchmark_deduplication(num_entities=200)

    # Analyze results
    print("\nMetrics Summary:")
    print("-" * 50)

    for name, result in benchmarks.results.items():
        if result.success:
            # Extract key metric
            metrics = result.metrics

            if "throughput" in str(metrics):
                for k, v in metrics.items():
                    if "throughput" in k.lower():
                        print(f"  {name:30} | {v:8.2f} {k}")
                        break
            elif "duration" in metrics:
                print(f"  {name:30} | {metrics['duration_seconds']:8.2f}s")

    # Save results
    benchmarks.generate_report("custom_benchmark_report.md")

    await engine.cleanup_kggen_pipeline()


async def example_7_performance_validation():
    """Example 7: Validating performance against thresholds."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Performance Validation")
    print("="*60)

    engine = KnowledgeEngine()
    benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

    # Define performance thresholds
    thresholds = {
        "min_throughput": 100,  # artifacts/sec
        "max_latency_p95": 500,  # milliseconds
        "min_accuracy": 0.85,  # 85%
    }

    print(f"\nPerformance Thresholds:")
    print(f"  Min Throughput: {thresholds['min_throughput']} artifacts/sec")
    print(f"  Max P95 Latency: {thresholds['max_latency_p95']}ms")
    print(f"  Min Accuracy: {thresholds['min_accuracy']*100:.0f}%")

    # Run benchmarks
    print("\nRunning validation benchmarks...")
    addition_result = await benchmarks.benchmark_knowledge_addition(num_artifacts=200)
    retrieval_result = await benchmarks.benchmark_knowledge_retrieval(num_queries=50)
    dedup_result = await benchmarks.benchmark_deduplication(num_entities=200)

    # Validate against thresholds
    print("\nValidation Results:")
    print("-" * 50)

    # Check throughput
    if addition_result.success:
        throughput = addition_result.metrics["artifacts_per_second"]
        status = "✓ PASS" if throughput >= thresholds["min_throughput"] else "✗ FAIL"
        print(f"  Throughput: {throughput:.2f}/sec {status}")

    # Check latency
    if retrieval_result.success:
        avg_latency = retrieval_result.metrics["avg_latency_ms"]
        status = "✓ PASS" if avg_latency <= thresholds["max_latency_p95"] else "✗ FAIL"
        print(f"  Latency: {avg_latency:.2f}ms {status}")

    # Check accuracy
    if dedup_result.success:
        accuracy = dedup_result.metrics["accuracy"]
        status = "✓ PASS" if accuracy >= thresholds["min_accuracy"] else "✗ FAIL"
        print(f"  Accuracy: {accuracy*100:.1f}% {status}")

    await engine.cleanup_kggen_pipeline()


async def example_8_concurrent_stress_test():
    """Example 8: Stress testing with concurrent operations."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Concurrent Stress Test")
    print("="*60)

    engine = KnowledgeEngine()
    benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

    # Test increasing concurrent load
    print("\nConcurrent Load Testing:")
    print("  Clients | Throughput | Error Rate | Duration")
    print("  " + "-"*50)

    for num_clients in [5, 10, 20]:
        result = await benchmarks.benchmark_concurrent_operations(
            num_concurrent=num_clients,
            operations_per_client=20
        )

        if result.success:
            throughput = result.metrics["throughput_ops_per_sec"]
            error_rate = result.metrics["error_rate"] * 100
            duration = result.metrics["duration_seconds"]
            print(f"  {num_clients:7} | {throughput:10.2f} | {error_rate:9.2f}% | {duration:8.2f}s")

    await engine.cleanup_kggen_pipeline()


async def run_all_examples():
    """Run all examples."""
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Custom Configuration", example_2_custom_configuration),
        ("Comparative Analysis", example_3_comparative_analysis),
        ("Scalability Analysis", example_4_scalability_analysis),
        ("End-to-End Workflow", example_5_end_to_end_workflow),
        ("Custom Metrics", example_6_custom_metrics_collection),
        ("Performance Validation", example_7_performance_validation),
        ("Concurrent Stress Test", example_8_concurrent_stress_test),
    ]

    print("\n╔" + "="*58 + "╗")
    print("║" + " "*15 + "BENCHMARK USAGE EXAMPLES" + " "*16 + "║")
    print("╚" + "="*58 + "╝")

    for i, (name, example_func) in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}/8: {name}")
        print(f"{'='*60}")

        try:
            await example_func()
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark usage examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=range(1, 9),
        help="Run specific example (1-8)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples"
    )

    args = parser.parse_args()

    if args.all:
        asyncio.run(run_all_examples())
    elif args.example:
        example_functions = [
            example_1_basic_usage,
            example_2_custom_configuration,
            example_3_comparative_analysis,
            example_4_scalability_analysis,
            example_5_end_to_end_workflow,
            example_6_custom_metrics_collection,
            example_7_performance_validation,
            example_8_concurrent_stress_test,
        ]
        asyncio.run(example_functions[args.example - 1]())
    else:
        # Default to example 1
        asyncio.run(example_1_basic_usage())
