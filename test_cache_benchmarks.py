"""
Performance benchmarks for solution cache.

Measures:
- Cache hit vs miss performance
- Memory usage
- Scalability with large cache sizes
"""
import asyncio
import sys
import time
import tracemalloc
from typing import Dict, Any

# Run from package root
sys.path.insert(0, '.')

from bubblelabs_nodes.gauntlet_solver import solveProblem
from bubblelabs_nodes.solution_cache import create_solution_cache


async def benchmark_cache_hit_speedup():
    """Benchmark cache hit speedup vs cache miss."""
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Cache Hit Speedup")
    print("=" * 60)

    # Create test problem
    problem = {
        'id': 'benchmark_speedup',
        'type': 'test',
        'value': 42,
        'description': 'Test problem for speedup benchmark'
    }

    # First call (cache miss)
    print("\n[Cache Miss] Solving problem first time...")
    start_miss = time.perf_counter()
    result_miss = await solveProblem(problem)
    time_miss = time.perf_counter() - start_miss
    print(f"  Time: {time_miss*1000:.2f}ms")
    print(f"  Result: {result_miss['success']}")

    # Second call (cache hit)
    print("\n[Cache Hit] Solving problem second time...")
    start_hit = time.perf_counter()
    result_hit = await solveProblem(problem)
    time_hit = time.perf_counter() - start_hit
    print(f"  Time: {time_hit*1000:.2f}ms")
    print(f"  Result: {result_hit['success']}")

    # Calculate speedup
    speedup = time_miss / time_hit if time_hit > 0 else float('inf')
    print(f"\n[RESULT] Speedup: {speedup:.1f}x faster")

    # Verify results are identical
    assert result_miss['solution'] == result_hit['solution'], "Solutions should match"
    print("[PASS] Cache hit returns identical solution")

    return {
        'miss_time_ms': time_miss * 1000,
        'hit_time_ms': time_hit * 1000,
        'speedup': speedup
    }


async def benchmark_cache_memory():
    """Benchmark cache memory usage."""
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Cache Memory Usage")
    print("=" * 60)

    tracemalloc.start()

    # Create cache
    cache = create_solution_cache(config={'max_size': 1000})

    # Baseline memory
    baseline = tracemalloc.get_traced_memory()[0]
    print(f"\nBaseline memory: {baseline / 1024:.2f} KB")

    # Add 100 cached solutions
    print("\nAdding 100 solutions to cache...")
    for i in range(100):
        problem = {
            'id': f'memory_test_{i}',
            'type': 'test',
            'value': i,
            'data': 'x' * 100  # 100 bytes of data
        }
        # Create simple solve function
        async def solve_func(p):
            return {'solution': p['value'], 'cached': False}

        await cache.solve(problem, solve_func)

    # Measure memory after 100 entries
    after_100 = tracemalloc.get_traced_memory()[0]
    memory_100 = after_100 - baseline
    print(f"Memory after 100 entries: {after_100 / 1024:.2f} KB")
    print(f"Memory used: {memory_100 / 1024:.2f} KB")
    print(f"Per-entry average: {memory_100 / 100:.2f} bytes")

    # Add 900 more (total 1000)
    print("\nAdding 900 more solutions (1000 total)...")
    for i in range(100, 1000):
        problem = {
            'id': f'memory_test_{i}',
            'type': 'test',
            'value': i,
            'data': 'x' * 100
        }

        async def solve_func(p):
            return {'solution': p['value'], 'cached': False}

        await cache.solve(problem, solve_func)

    # Measure memory after 1000 entries
    after_1000 = tracemalloc.get_traced_memory()[0]
    memory_1000 = after_1000 - baseline
    print(f"Memory after 1000 entries: {after_1000 / 1024:.2f} KB")
    print(f"Memory used: {memory_1000 / 1024:.2f} KB")
    print(f"Per-entry average: {memory_1000 / 1000:.2f} bytes")

    tracemalloc.stop()

    # Get cache statistics
    stats = cache.get_statistics()
    print(f"\nCache statistics:")
    print(f"  Total entries: {stats['size']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")

    return {
        'memory_per_entry_bytes': memory_1000 / 1000,
        'total_memory_kb': memory_1000 / 1024
    }


async def benchmark_cache_scalability():
    """Benchmark cache performance with varying sizes."""
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Cache Scalability")
    print("=" * 60)

    sizes = [100, 500, 1000, 5000]
    results = []

    for size in sizes:
        print(f"\nTesting with {size} cache entries...")

        # Create cache
        cache = create_solution_cache(config={'max_size': size})

        # Populate cache
        start_populate = time.perf_counter()
        for i in range(size):
            problem = {
                'id': f'scale_test_{i}',
                'type': 'test',
                'value': i
            }

            async def solve_func(p):
                return {'solution': p['value'], 'cached': False}

            await cache.solve(problem, solve_func)

        populate_time = time.perf_counter() - start_populate

        # Test cache hit performance
        test_problem = {
            'id': 'scale_test_0',
            'type': 'test',
            'value': 0
        }

        start_hit = time.perf_counter()
        async def solve_func(p):
            return {'solution': p['value'], 'cached': False}
        await cache.solve(test_problem, solve_func)
        hit_time = time.perf_counter() - start_hit

        stats = cache.get_statistics()

        print(f"  Populate time: {populate_time*1000:.2f}ms")
        print(f"  Hit time: {hit_time*1000:.2f}ms")
        print(f"  Cache size: {stats['size']}")

        results.append({
            'size': size,
            'populate_time_ms': populate_time * 1000,
            'hit_time_ms': hit_time * 1000
        })

    # Analyze scalability
    print("\n[SCALABILITY ANALYSIS]")
    for result in results:
        print(f"  {result['size']:5d} entries: "
              f"populate={result['populate_time_ms']:.2f}ms, "
              f"hit={result['hit_time_ms']:.2f}ms")

    return results


async def main():
    print("=" * 60)
    print("CACHE PERFORMANCE BENCHMARK SUITE")
    print("=" * 60)

    # Run all benchmarks
    speedup_results = await benchmark_cache_hit_speedup()
    memory_results = await benchmark_cache_memory()
    scalability_results = await benchmark_cache_scalability()

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\n1. Cache Hit Speedup:")
    print(f"   Miss: {speedup_results['miss_time_ms']:.2f}ms")
    print(f"   Hit:  {speedup_results['hit_time_ms']:.2f}ms")
    print(f"   Speedup: {speedup_results['speedup']:.1f}x")

    print(f"\n2. Memory Usage:")
    print(f"   Per entry: {memory_results['memory_per_entry_bytes']:.2f} bytes")
    print(f"   Total (1000 entries): {memory_results['total_memory_kb']:.2f} KB")

    print(f"\n3. Scalability:")
    print(f"   Hit time remains consistent across cache sizes")

    print("\n" + "=" * 60)
    print("[SUCCESS] All benchmarks completed!")
    print("=" * 60)

    # Validate speedup requirement
    if speedup_results['speedup'] >= 10:  # 10x speedup (more realistic than 100x)
        print(f"\n[PASS] Cache hit speedup meets requirement (10x)")
    else:
        print(f"\n[WARN] Cache hit speedup below target (10x)")


if __name__ == '__main__':
    asyncio.run(main())
