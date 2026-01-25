"""
Quick Test Script for Benchmark Suite

This script performs a quick validation of the benchmark system
to ensure all components work correctly.

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


async def test_benchmark_system():
    """Test the benchmark system with minimal data."""

    print("╔" + "="*58 + "╗")
    print("║" + " "*20 + "BENCHMARK SYSTEM TEST" + " "*20 + "║")
    print("╚" + "="*58 + "╝")

    try:
        # Initialize
        print("\n[1/5] Initializing Knowledge Engine...")
        engine = KnowledgeEngine()
        benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)
        print("✓ Engine initialized successfully")

        # Test 1: Knowledge Addition
        print("\n[2/5] Testing Knowledge Addition Benchmark...")
        result = await benchmarks.benchmark_knowledge_addition(
            num_artifacts=50,
            batch_size=10
        )
        assert result.success, f"Knowledge addition failed: {result.error}"
        assert result.metrics["artifacts_per_second"] > 0, "Invalid throughput"
        print("✓ Knowledge addition benchmark passed")

        # Test 2: Knowledge Retrieval
        print("\n[3/5] Testing Knowledge Retrieval Benchmark...")
        result = await benchmarks.benchmark_knowledge_retrieval(
            num_queries=20,
            query_types=["keyword", "graph"]
        )
        assert result.success, f"Knowledge retrieval failed: {result.error}"
        assert result.metrics["avg_latency_ms"] > 0, "Invalid latency"
        print("✓ Knowledge retrieval benchmark passed")

        # Test 3: Deduplication
        print("\n[4/5] Testing Deduplication Benchmark...")
        result = await benchmarks.benchmark_deduplication(
            num_entities=100,
            duplicate_rate=0.3
        )
        assert result.success, f"Deduplication failed: {result.error}"
        assert "f1_score" in result.metrics, "Missing F1 score metric"
        print("✓ Deduplication benchmark passed")

        # Test 4: Graph Algorithms
        print("\n[5/5] Testing Graph Algorithms Benchmark...")
        result = await benchmarks.benchmark_graph_algorithms(
            graph_sizes=[50, 100]
        )
        assert result.success, f"Graph algorithms failed: {result.error}"
        assert len(result.metrics) >= 2, "Missing graph size results"
        print("✓ Graph algorithms benchmark passed")

        # Generate Test Report
        print("\n[6/6] Generating Test Report...")
        test_report_path = "test_benchmark_report.md"
        benchmarks.generate_report(
            output_path=test_report_path,
            include_raw_data=True
        )
        print(f"✓ Test report saved to {test_report_path}")

        # Cleanup
        await engine.cleanup_kggen_pipeline()

        # Summary
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nThe benchmark system is working correctly.")
        print("You can now run full benchmarks with:")
        print("  python run_benchmarks.py --quick")
        print("  python run_benchmarks.py --all")
        print("="*60)

        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def quick_smoke_test():
    """Run a very quick smoke test."""
    print("\nRunning smoke test...")

    try:
        engine = KnowledgeEngine()
        benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

        # Add a few entities
        for i in range(10):
            await engine.entity_graph.add_entity(
                f"test_entity_{i}",
                {"test": True}
            )

        # Verify entities were added
        entities = engine.entity_graph.get_entities()
        assert len(entities) >= 10, "Entities not added correctly"

        print("✓ Smoke test passed")
        await engine.cleanup_kggen_pipeline()
        return True

    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test benchmark system")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run quick smoke test only"
    )

    args = parser.parse_args()

    if args.smoke:
        success = asyncio.run(quick_smoke_test())
    else:
        success = asyncio.run(test_benchmark_system())

    sys.exit(0 if success else 1)
