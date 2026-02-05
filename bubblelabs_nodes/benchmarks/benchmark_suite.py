"""
Performance Benchmark Suite for Gauntlet System

Measures and documents performance improvements from parallel execution,
caching, and other optimizations.

Benchmarks:
- Parallel vs sequential execution
- Cache hit/miss performance
- Worker pool vs asyncio
- Checkpoint overhead
- End-to-end system performance
"""

import asyncio
import time
import psutil
from typing import Dict, List, Any, Callable
from datetime import datetime

from bubblelabs_nodes import (
    ParallelProblemExecutor,
    WorkerPoolExecutor,
    GauntletSolver,
    solveProblem,
    create_solution_cache,
    create_checkpoint_manager,
)


class BenchmarkSuite:
    """Comprehensive benchmark suite"""

    def __init__(self):
        self.results = {}

    async def benchmark_parallel_vs_sequential(self):
        """Benchmark parallel vs sequential execution"""
        print("\n" + "=" * 60)
        print("Benchmark: Parallel vs Sequential Execution")
        print("=" * 60)

        executor = ParallelProblemExecutor(max_parallelism=5)

        async def solver(problem):
            await asyncio.sleep(0.05)  # 50ms per problem
            return {'id': problem['id'], 'success': True}

        problem_counts = [1, 3, 5, 10]

        for count in problem_counts:
            problems = [{'id': f'bench_{i}'} for i in range(count)]

            # Sequential
            start_seq = time.time()
            for p in problems:
                await solver(p)
            time_seq = time.time() - start_seq

            # Parallel
            start_par = time.time()
            await executor.execute_in_parallel(problems, solver, {})
            time_par = time.time() - start_par

            speedup = time_seq / time_par

            print(f"\n{count} problems:")
            print(f"  Sequential: {time_seq:.3f}s")
            print(f"  Parallel:   {time_par:.3f}s")
            print(f"  Speedup:    {speedup:.2f}x")

            self.results[f'parallel_{count}'] = {
                'sequential_time': time_seq,
                'parallel_time': time_par,
                'speedup': speedup
            }

    async def benchmark_cache_performance(self):
        """Benchmark cache hit vs miss performance"""
        print("\n" + "=" * 60)
        print("Benchmark: Cache Performance")
        print("=" * 60)

        cache = create_solution_cache()

        async def solver(problem):
            # Simulate expensive computation
            await asyncio.sleep(0.1)
            return {'id': problem['id'], 'success': True, 'score': 0.85}

        problem = {'id': 'cache_bench', 'statement': 'Cache test'}

        # Cache miss (first call)
        start_miss = time.time()
        await cache.get(problem)
        if not await cache.has(problem['id']):
            result = await solver(problem)
            await cache.set(problem, result)
        time_miss = time.time() - start_miss

        # Cache hit (second call)
        start_hit = time.time()
        cached = await cache.get(problem)
        time_hit = time.time() - start_hit

        speedup = time_miss / time_hit

        print(f"\nCache Miss: {time_miss:.3f}s")
        print(f"Cache Hit:  {time_hit:.3f}s")
        print(f"Speedup:    {speedup:.2f}x")

        self.results['cache'] = {
            'miss_time': time_miss,
            'hit_time': time_hit,
            'speedup': speedup
        }

    async def benchmark_worker_pool_vs_asyncio(self):
        """Benchmark worker pool vs asyncio executor"""
        print("\n" + "=" * 60)
        print("Benchmark: Worker Pool vs Asyncio")
        print("=" * 60)

        def sync_solver(problem):
            time.sleep(0.02)
            return {'id': problem['id'], 'success': True}

        async def async_solver(problem):
            await asyncio.sleep(0.02)
            return {'id': problem['id'], 'success': True}

        problems = [{'id': f'wp_{i}'} for i in range(10)]

        # Asyncio executor
        asyncio_executor = ParallelProblemExecutor(max_parallelism=5)
        start_asyncio = time.time()
        await asyncio_executor.execute_in_parallel(problems, async_solver, {})
        time_asyncio = time.time() - start_asyncio

        # Worker pool
        worker_pool = WorkerPoolExecutor(max_workers=5)
        start_worker = time.time()
        await worker_pool.execute_in_parallel(problems, sync_solver, {})
        time_worker = time.time() - start_worker

        print(f"\nAsyncio:  {time_asyncio:.3f}s")
        print(f"Worker Pool: {time_worker:.3f}s")
        print(f"Ratio: {time_asyncio / time_worker:.2f}x")

        self.results['executor_comparison'] = {
            'asyncio_time': time_asyncio,
            'worker_time': time_worker,
            'ratio': time_asyncio / time_worker
        }

    async def benchmark_checkpoint_overhead(self):
        """Benchmark checkpoint creation overhead"""
        print("\n" + "=" * 60)
        print("Benchmark: Checkpoint Overhead")
        print("=" * 60)

        checkpoint_manager = create_checkpoint_manager(compression=False)

        problem = {
            'id': 'cp_bench',
            'statement': 'Checkpoint benchmark',
            'data': 'x' * 10000  # 10KB of data
        }

        # Without checkpoint
        async def solve_without_checkpoint():
            await asyncio.sleep(0.05)
            return {'result': 'done'}

        start_no_cp = time.time()
        await solve_without_checkpoint()
        time_no_cp = time.time() - start_no_cp

        # With checkpoint
        async def solve_with_checkpoint():
            await checkpoint_manager.create_checkpoint(
                problem=problem,
                context={},
                solutions={},
                level=0,
                stage='test'
            )
            await asyncio.sleep(0.05)
            return {'result': 'done'}

        start_with_cp = time.time()
        await solve_with_checkpoint()
        time_with_cp = time.time() - start_with_cp

        overhead = ((time_with_cp - time_no_cp) / time_no_cp) * 100

        print(f"\nWithout checkpoint: {time_no_cp:.3f}s")
        print(f"With checkpoint:    {time_with_cp:.3f}s")
        print(f"Overhead:           {overhead:.1f}%")

        self.results['checkpoint_overhead'] = {
            'without_checkpoint': time_no_cp,
            'with_checkpoint': time_with_cp,
            'overhead_percent': overhead
        }

    async def benchmark_system_resources(self):
        """Benchmark system resource usage"""
        print("\n" + "=" * 60)
        print("Benchmark: System Resource Usage")
        print("=" * 60)

        process = psutil.Process()

        async def cpu_intensive_task():
            # CPU-intensive task
            for _ in range(1000000):
                _ = 1 + 1

        # Measure CPU and memory
        cpu_before = process.cpu_percent(interval=0.1)
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        await cpu_intensive_task()

        cpu_after = process.cpu_percent(interval=0.1)
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        print(f"\nCPU Before: {cpu_before:.1f}%")
        print(f"CPU After:  {cpu_after:.1f}%")
        print(f"Memory Before: {mem_before:.1f} MB")
        print(f"Memory After:  {mem_after:.1f} MB")
        print(f"Memory Delta:  {mem_after - mem_before:.1f} MB")

        self.results['resource_usage'] = {
            'cpu_before': cpu_before,
            'cpu_after': cpu_after,
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after,
            'memory_delta_mb': mem_after - mem_before
        }

    def generate_report(self):
        """Generate benchmark report"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY REPORT")
        print("=" * 60)

        timestamp = datetime.utcnow().isoformat()

        report = {
            'timestamp': timestamp,
            'results': self.results
        }

        print(f"\nTimestamp: {timestamp}")

        if 'parallel_5' in self.results:
            r = self.results['parallel_5']
            print(f"\n5-Problem Parallel Speedup: {r['speedup']:.2f}x")

        if 'cache' in self.results:
            r = self.results['cache']
            print(f"Cache Speedup: {r['speedup']:.2f}x")

        if 'checkpoint_overhead' in self.results:
            r = self.results['checkpoint_overhead']
            print(f"Checkpoint Overhead: {r['overhead_percent']:.1f}%")

        print("\n" + "=" * 60)

        return report


async def run_all_benchmarks():
    """Run complete benchmark suite"""
    suite = BenchmarkSuite()

    await suite.benchmark_parallel_vs_sequential()
    await suite.benchmark_cache_performance()
    await suite.benchmark_worker_pool_vs_asyncio()
    await suite.benchmark_checkpoint_overhead()
    await suite.benchmark_system_resources()

    return suite.generate_report()


if __name__ == '__main__':
    print("Starting Gauntlet Benchmark Suite...")
    print("This may take a few minutes...\n")

    report = asyncio.run(run_all_benchmarks())

    print("\n[OK] Benchmarks complete!")
