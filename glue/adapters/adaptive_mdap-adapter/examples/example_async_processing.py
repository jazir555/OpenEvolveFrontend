#!/usr/bin/env python3
"""
Example: Async Processing with Concurrent Operations

This example demonstrates how to use the async adapter for concurrent
complexity analysis, achieving 3-5x performance improvements.

Usage:
    cd examples
    python example_async_processing.py
"""

import os
import sys
import asyncio
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set environment variables
os.environ.setdefault("ADAPTIVE_MDAP_TIMEOUT_MS", "5000")

from src import get_async_adapter, CanonicalSubProblem


async def main():
    """Demonstrate async concurrent processing."""
    print("=" * 70)
    print("  EXAMPLE: Async Concurrent Processing")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now(timezone.utc).isoformat()}\n")

    # Get async adapter
    adapter = get_async_adapter()

    # Create 10 sub-problems to analyze
    print("Creating 10 sub-problems for analysis...\n")

    subproblems = [
        CanonicalSubProblem(
            id=f"async_task_{i}",
            description=f"Analyze system component {i}",
            domain="distributed_systems",
            depth=2
        )
        for i in range(10)
    ]

    print(f"Created {len(subproblems)} sub-problems")

    # Sequential processing (baseline)
    print("\n" + "-" * 70)
    print("Sequential Processing (Baseline)")
    print("-" * 70)

    start = asyncio.get_event_loop().time()
    sequential_results = []

    for sp in subproblems[:3]:  # Only 3 for sequential demo
        result = await adapter.analyze_complexity_async(sp, use_cache=False)
        sequential_results.append(result)
        if result.complexity_score:
            print(f"  Task {sp.id}: complexity={result.complexity_score.overall_score:.3f}")
        else:
            print(f"  Task {sp.id}: No complexity score available (graceful degradation)")

    sequential_duration = (asyncio.get_event_loop().time() - start) * 1000

    print(f"\nSequential Time: {sequential_duration:.0f}ms")
    print(f"Average: {sequential_duration / len(sequential_results):.0f}ms per task")

    # Concurrent processing (3x-5x faster)
    print("\n" + "-" * 70)
    print("Concurrent Processing (3-5x Faster)")
    print("-" * 70)

    start = asyncio.get_event_loop().time()

    # Process all 10 concurrently with max 5 concurrent
    results = await adapter.batch_analyze_complexity(
        subproblems,
        max_concurrency=5
    )

    concurrent_duration = (asyncio.get_event_loop().time() - start) * 1000

    print(f"\nProcessed {len(results)} tasks concurrently")
    print(f"Concurrent Time: {concurrent_duration:.0f}ms")
    print(f"Average: {concurrent_duration / len(results):.0f}ms per task")

    # Show speedup
    speedup = sequential_duration / (concurrent_duration / len(subproblems) * 3)
    print(f"\nSpeedup: {speedup:.1f}x faster")

    # Show cache statistics
    print("\n" + "-" * 70)
    print("Cache Statistics")
    print("-" * 70)

    cache_stats = adapter.get_cache_stats()
    print(f"Hit Rate: {cache_stats['hit_rate']:.1%}")
    print(f"Cache Size: {cache_stats['size']}/{cache_stats['max_size']}")

    print("\n" + "=" * 70)
    print("  EXAMPLE COMPLETE")
    print("=" * 70)
    print(f"\nEnd Time: {datetime.now(timezone.utc).isoformat()}\n")


if __name__ == "__main__":
    asyncio.run(main())
