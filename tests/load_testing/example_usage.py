"""
Example usage of the Load Testing Framework

This script demonstrates how to use the load testing framework
with different configurations and scenarios.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.load_testing.kg_load_tests import KnowledgeGraphLoadTest
from tests.load_testing.analyze_results import LoadTestAnalyzer


async def example_basic_usage():
    """
    Example 1: Basic usage with default configuration.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)

    # Import and initialize your knowledge engine
    # from knowledge_engine.engine import KnowledgeEngine
    # engine = KnowledgeEngine()

    # For demonstration, create a mock engine
    class MockEngine:
        async def search(self, query, search_type="hybrid"):
            await asyncio.sleep(0.1)  # Simulate work
            return {"results": []}

        async def add_knowledge(self, source, content, metadata=None):
            await asyncio.sleep(0.05)  # Simulate work
            return {"id": "123"}

        async def get_graph_stats(self):
            await asyncio.sleep(0.05)
            return {"nodes": 100, "edges": 200}

    engine = MockEngine()

    # Create load tester
    load_test = KnowledgeGraphLoadTest(engine)

    # Run a simple read-heavy test
    result = await load_test.run_read_heavy_test(
        num_users=50,
        spawn_rate=5,
        test_duration=30
    )

    print(f"\nResult: {'PASSED' if result.passed else 'FAILED'}")


async def example_custom_configuration():
    """
    Example 2: Using custom configuration.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)

    class MockEngine:
        async def search(self, query, search_type="hybrid"):
            await asyncio.sleep(0.1)
            return {"results": []}

        async def add_knowledge(self, source, content, metadata=None):
            await asyncio.sleep(0.05)
            return {"id": "123"}

        async def get_graph_stats(self):
            await asyncio.sleep(0.05)
            return {"nodes": 100, "edges": 200}

    engine = MockEngine()
    load_test = KnowledgeGraphLoadTest(engine)

    # Custom configuration
    config = {
        "target_throughput": 150,  # Higher target
        "max_error_rate": 0.005    # Stricter error threshold
    }

    result = await load_test.run_read_heavy_test(
        num_users=100,
        spawn_rate=10,
        test_duration=60,
        config=config
    )

    print(f"\nThroughput: {result.metrics['throughput_ops_per_sec']:.2f} ops/sec")
    print(f"Error Rate: {result.metrics['error_rate']:.2%}")


async def example_multiple_tests():
    """
    Example 3: Running multiple test scenarios.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Multiple Test Scenarios")
    print("="*60)

    class MockEngine:
        async def search(self, query, search_type="hybrid"):
            await asyncio.sleep(0.1)
            return {"results": []}

        async def add_knowledge(self, source, content, metadata=None):
            await asyncio.sleep(0.05)
            return {"id": "123"}

        async def get_graph_stats(self):
            await asyncio.sleep(0.05)
            return {"nodes": 100, "edges": 200}

    engine = MockEngine()
    load_test = KnowledgeGraphLoadTest(engine)

    # Run multiple tests
    tests = [
        ("Read-Heavy", lambda: load_test.run_read_heavy_test(50, 5, 30)),
        ("Write-Heavy", lambda: load_test.run_write_heavy_test(25, 3, 30)),
        ("Spike Test", lambda: load_test.run_spike_test(10, 50, 20)),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nRunning {name}...")
        result = await test_func()
        results.append(result)
        print(f"Status: {'PASSED' if result.passed else 'FAILED'}")

        await asyncio.sleep(2)  # Cool down between tests

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for r in results if r.passed)
    print(f"Passed: {passed}/{len(results)}")


async def example_with_analysis():
    """
    Example 4: Run tests and analyze results.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Test with Analysis")
    print("="*60)

    class MockEngine:
        async def search(self, query, search_type="hybrid"):
            await asyncio.sleep(0.1)
            return {"results": []}

        async def add_knowledge(self, source, content, metadata=None):
            await asyncio.sleep(0.05)
            return {"id": "123"}

        async def get_graph_stats(self):
            await asyncio.sleep(0.05)
            return {"nodes": 100, "edges": 200}

    engine = MockEngine()
    load_test = KnowledgeGraphLoadTest(engine)

    # Run tests
    print("\nRunning tests...")
    await load_test.run_read_heavy_test(50, 5, 30)
    await load_test.run_write_heavy_test(25, 3, 30)
    await load_test.run_spike_test(10, 50, 20)

    # Save results
    output_file = "example_results.json"
    load_test.save_results(output_file)
    print(f"\nResults saved to: {output_file}")

    # Analyze results
    print("\nAnalyzing results...")
    analyzer = LoadTestAnalyzer(output_file)

    # Get throughput analysis
    throughput = analyzer.analyze_throughput()
    if "average_throughput" in throughput:
        print(f"\nAverage Throughput: {throughput['average_throughput']:.2f} ops/sec")

    # Identify bottlenecks
    bottlenecks = analyzer.identify_bottlenecks()
    if bottlenecks:
        print(f"\nBottlenecks Found: {len(bottlenecks)}")
        for bottleneck in bottlenecks:
            print(f"  [{bottleneck['severity']}] {bottleneck['issue']}")
    else:
        print("\nNo bottlenecks detected!")

    # Generate recommendations
    recommendations = analyzer.generate_recommendations()
    if recommendations:
        print(f"\nRecommendations: {len(recommendations)}")
        for rec in recommendations[:3]:  # Show top 3
            print(f"  [{rec['priority']}] {rec['recommendation']}")

    # Generate full report
    report_file = "example_report.txt"
    analyzer.generate_report(report_file)
    print(f"\nFull report saved to: {report_file}")


async def example_error_handling():
    """
    Example 5: Handling errors and edge cases.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Error Handling")
    print("="*60)

    # Engine that simulates failures
    class FailingEngine:
        def __init__(self):
            self.call_count = 0

        async def search(self, query, search_type="hybrid"):
            self.call_count += 1

            # Simulate 10% failure rate
            if self.call_count % 10 == 0:
                raise Exception("Simulated search failure")

            await asyncio.sleep(0.1)
            return {"results": []}

        async def add_knowledge(self, source, content, metadata=None):
            await asyncio.sleep(0.05)
            return {"id": "123"}

        async def get_graph_stats(self):
            await asyncio.sleep(0.05)
            return {"nodes": 100, "edges": 200}

    engine = FailingEngine()
    load_test = KnowledgeGraphLoadTest(engine)

    # Run test with looser error tolerance
    config = {
        "max_error_rate": 0.15  # Allow 15% error rate
    }

    result = await load_test.run_read_heavy_test(
        num_users=20,
        spawn_rate=5,
        test_duration=20,
        config=config
    )

    print(f"\nTest completed despite errors")
    print(f"Error Rate: {result.metrics['error_rate']:.2%}")
    print(f"Status: {'PASSED' if result.passed else 'FAILED'}")

    if result.errors:
        print(f"\nErrors logged:")
        for error in result.errors:
            print(f"  - {error}")


async def main():
    """
    Run all examples.
    """
    print("\n" + "="*70)
    print("LOAD TESTING FRAMEWORK - EXAMPLES")
    print("="*70)

    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Configuration", example_custom_configuration),
        ("Multiple Tests", example_multiple_tests),
        ("Test with Analysis", example_with_analysis),
        ("Error Handling", example_error_handling),
    ]

    for i, (name, func) in enumerate(examples, 1):
        try:
            await func()
        except Exception as e:
            print(f"\nExample failed: {e}")
            import traceback
            traceback.print_exc()

        if i < len(examples):
            print("\n" + "."*70)
            await asyncio.sleep(1)

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
