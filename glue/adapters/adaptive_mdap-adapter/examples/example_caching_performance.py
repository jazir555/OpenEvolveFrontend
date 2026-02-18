#!/usr/bin/env python3
"""
Example: Caching and Performance Optimization

This example demonstrates the performance improvements achieved through
response caching, connection pooling, and async processing.

Usage:
    cd examples
    python example_caching_performance.py
"""

import os
import sys
import asyncio
import time
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Set environment variables
os.environ.setdefault("ADAPTIVE_MDAP_TIMEOUT_MS", "5000")
os.environ.setdefault("ADAPTIVE_MDAP_CACHE_SIZE", "1000")
os.environ.setdefault("ADAPTIVE_MDAP_CACHE_TTL", "300")

from src import get_async_adapter, get_performance_monitor, CanonicalSubProblem


def time_function(func, *args, **kwargs):
    """Time a function execution."""
    start = time.time()
    result = func(*args, **kwargs)
    duration = (time.time() - start) * 1000  # Convert to ms
    return result, duration


async def time_async_function(func, *args, **kwargs):
    """Time an async function execution."""
    start = asyncio.get_event_loop().time()
    result = await func(*args, **kwargs)
    duration = (asyncio.get_event_loop().time() - start) * 1000  # Convert to ms
    return result, duration


def main():
    """Demonstrate caching and performance optimizations."""
    print("=" * 70)
    print("  EXAMPLE: Caching and Performance Optimization")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now(timezone.utc).isoformat()}\n")

    # Get async adapter and performance monitor
    adapter = get_async_adapter()
    monitor = get_performance_monitor()

    # Phase 1: Baseline - No caching
    print("Phase 1: Baseline Performance (No Cache)")
    print("-" * 70)

    # Create test subproblems
    test_problems = [
        CanonicalSubProblem(
            id=f"baseline_{i}",
            description=f"Test problem {i}",
            domain="test",
            depth=1
        )
        for i in range(5)
    ]

    print("\nRunning 5 analyses WITHOUT caching...")

    start = time.time()
    for sp in test_problems:
        result, duration = asyncio.run(
            time_async_function(
                adapter.analyze_complexity_async,
                sp,
                use_cache=False
            )
        )
        monitor.record("baseline_analysis", duration)
        print(f"  {sp.id}: {duration:.0f}ms")
    baseline_total = (time.time() - start) * 1000

    print(f"\nBaseline Total: {baseline_total:.0f}ms")
    print(f"Baseline Average: {baseline_total / len(test_problems):.0f}ms per operation")

    baseline_stats = monitor.get_stats("baseline_analysis")
    print(f"Baseline P95: {baseline_stats['p95_ms']:.0f}ms")

    # Phase 2: With caching - First run (cache misses)
    print("\n" + "=" * 70)
    print("Phase 2: Caching Performance - First Run (Cache Misses)")
    print("=" * 70)

    cache_problems = [
        CanonicalSubProblem(
            id=f"cache_{i}",
            description=f"Cache test problem {i}",
            domain="test",
            depth=1
        )
        for i in range(5)
    ]

    print("\nRunning 5 analyses WITH caching (first run)...")

    start = time.time()
    for sp in cache_problems:
        result, duration = asyncio.run(
            time_async_function(
                adapter.analyze_complexity_async,
                sp,
                use_cache=True
            )
        )
        monitor.record("cached_analysis_first", duration)
        print(f"  {sp.id}: {duration:.0f}ms")
    first_cached_total = (time.time() - start) * 1000

    print(f"\nFirst Run Total: {first_cached_total:.0f}ms")
    print(f"First Run Average: {first_cached_total / len(cache_problems):.0f}ms per operation")

    # Phase 3: With caching - Second run (cache hits)
    print("\n" + "=" * 70)
    print("Phase 3: Caching Performance - Second Run (Cache Hits)")
    print("=" * 70)

    print("\nRunning SAME 5 analyses (should hit cache)...")

    start = time.time()
    for sp in cache_problems:
        result, duration = asyncio.run(
            time_async_function(
                adapter.analyze_complexity_async,
                sp,
                use_cache=True
            )
        )
        monitor.record("cached_analysis_hit", duration)
        print(f"  {sp.id}: {duration:.0f}ms")
    second_cached_total = (time.time() - start) * 1000

    print(f"\nSecond Run Total: {second_cached_total:.0f}ms")
    print(f"Second Run Average: {second_cached_total / len(cache_problems):.0f}ms per operation")

    hit_stats = monitor.get_stats("cached_analysis_hit")

    # Calculate cache speedup
    speedup = baseline_total / second_cached_total
    cache_savings = ((baseline_total - second_cached_total) / baseline_total) * 100

    print(f"\nCache Speedup: {speedup:.1f}x faster")
    print(f"Cache Savings: {cache_savings:.1f}% reduction in time")

    # Phase 4: Cache statistics
    print("\n" + "=" * 70)
    print("Phase 4: Cache Statistics")
    print("=" * 70)

    cache_stats = adapter.get_cache_stats()

    print(f"\nCache Size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"Hit Rate: {cache_stats['hit_rate']:.1%}")
    print(f"Total Hits: {cache_stats['total_hits']}")
    print(f"Total Misses: {cache_stats['total_misses']}")

    # Phase 5: Concurrent processing comparison
    print("\n" + "=" * 70)
    print("Phase 5: Concurrent vs Sequential Processing")
    print("=" * 70)

    concurrent_problems = [
        CanonicalSubProblem(
            id=f"concurrent_{i}",
            description=f"Concurrent test {i}",
            domain="test",
            depth=1
        )
        for i in range(10)
    ]

    print(f"\nProcessing {len(concurrent_problems)} operations...")

    # Sequential
    print("\nSequential processing...")
    start = time.time()
    for sp in concurrent_problems[:3]:  # Only 3 for sequential
        result, _ = asyncio.run(
            time_async_function(
                adapter.analyze_complexity_async,
                sp,
                use_cache=False
            )
        )
    sequential_time = (time.time() - start) * 1000

    print(f"Sequential Time: {sequential_time:.0f}ms")

    # Concurrent
    print("\nConcurrent processing (max_concurrency=5)...")
    start = time.time()
    results = asyncio.run(
        adapter.batch_analyze_complexity(
            concurrent_problems,
            max_concurrency=5
        )
    )
    concurrent_time = (time.time() - start) * 1000

    print(f"Concurrent Time: {concurrent_time:.0f}ms")
    print(f"Operations: {len(results)}")

    concurrent_speedup = (sequential_time / 3) / (concurrent_time / len(concurrent_problems))
    print(f"\nConcurrent Speedup: {concurrent_speedup:.1f}x faster")

    # Phase 6: Overall performance summary
    print("\n" + "=" * 70)
    print("Phase 6: Performance Summary")
    print("=" * 70)

    all_stats = monitor.get_all_stats()

    print("\nOperation Statistics:")
    for operation, stats in all_stats.items():
        print(f"\n{operation}:")
        print(f"  Count: {stats['count']}")
        print(f"  Average: {stats['avg_ms']:.0f}ms")
        print(f"  Min: {stats['min_ms']:.0f}ms")
        print(f"  Max: {stats['max_ms']:.0f}ms")
        print(f"  P50: {stats['p50_ms']:.0f}ms")
        print(f"  P95: {stats['p95_ms']:.0f}ms")
        print(f"  P99: {stats['p99_ms']:.0f}ms")

    print("\n" + "=" * 70)
    print("  KEY TAKEAWAYS")
    print("=" * 70)

    print(f"\n1. Caching Speedup: {speedup:.1f}x faster (on cache hits)")
    print(f"2. Cache Savings: {cache_savings:.1f}% reduction in response time")
    print(f"3. Concurrent Speedup: {concurrent_speedup:.1f}x faster")
    print(f"4. Cache Hit Rate: {cache_stats['hit_rate']:.1%}")

    print("\nRecommendations:")
    print("  - Enable caching for repeated analyses")
    print("  - Use concurrent processing for batch operations")
    print("  - Monitor cache hit rate to optimize cache size and TTL")

    print("\n" + "=" * 70)
    print("  EXAMPLE COMPLETE")
    print("=" * 70)
    print(f"\nEnd Time: {datetime.now(timezone.utc).isoformat()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
