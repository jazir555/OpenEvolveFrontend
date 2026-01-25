"""
Performance Benchmarking Suite for OpenEvolve Frontend

This module provides comprehensive benchmarking capabilities to measure
the impact of performance optimizations.

Author: Performance Benchmarking Suite
Version: 1.0.0
"""

import time
import statistics
from typing import Callable, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Benchmark Result Classes
# ============================================================================

@dataclass
class BenchmarkIteration:
    """Single iteration benchmark result"""
    iteration: int
    execution_time: float
    memory_mb: float = 0.0
    success: bool = True
    error: str = None


@dataclass
class BenchmarkStats:
    """Statistical summary of benchmark results"""
    name: str
    total_time: float
    mean_time: float
    median_time: float
    std_dev: float
    min_time: float
    max_time: float
    iterations: int
    success_rate: float
    total_memory_mb: float

    def __str__(self):
        return (
            f"Benchmark: {self.name}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Mean: {self.mean_time*1000:.2f}ms\n"
            f"  Median: {self.median_time*1000:.2f}ms\n"
            f"  Std Dev: {self.std_dev*1000:.2f}ms\n"
            f"  Min: {self.min_time*1000:.2f}ms\n"
            f"  Max: {self.max_time*1000:.2f}ms\n"
            f"  Total: {self.total_time:.2f}s\n"
            f"  Success Rate: {self.success_rate*100:.1f}%\n"
            f"  Memory: {self.total_memory_mb:.1f}MB"
        )


@dataclass
class ComparisonResult:
    """Result of comparing two benchmarks"""
    old_stats: BenchmarkStats
    new_stats: BenchmarkStats
    improvement_percent: float
    speedup_factor: float
    memory_saving_percent: float

    def __str__(self):
        return (
            f"Performance Comparison: {self.old_stats.name} vs {self.new_stats.name}\n"
            f"{'='*70}\n"
            f"  Old Mean: {self.old_stats.mean_time*1000:.2f}ms\n"
            f"  New Mean: {self.new_stats.mean_time*1000:.2f}ms\n"
            f"  Improvement: {self.improvement_percent:+.1f}%\n"
            f"  Speedup: {self.speedup_factor:.2f}x\n"
            f"  Memory Saving: {self.memory_saving_percent:+.1f}%\n"
            f"{'='*70}"
        )


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """
    Comprehensive benchmarking runner with statistical analysis.

    Usage:
        runner = BenchmarkRunner()

        # Benchmark a function
        stats = runner.benchmark(
            func=my_function,
            args=(arg1, arg2),
            kwargs={'key': 'value'},
            iterations=100,
            warmup_iterations=10
        )

        # Compare two functions
        comparison = runner.compare(
            old_func=old_implementation,
            new_func=new_implementation,
            args=(arg1,),
            iterations=100
        )

        print(comparison)
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize benchmark runner.

        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self.history: List[BenchmarkStats] = []

    def benchmark(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict = None,
        iterations: int = 100,
        warmup_iterations: int = 10,
        name: str = None,
        measure_memory: bool = False
    ) -> BenchmarkStats:
        """
        Benchmark a function with statistical analysis.

        Args:
            func: Function to benchmark
            args: Positional arguments
            kwargs: Keyword arguments
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations (not measured)
            name: Benchmark name (defaults to function name)
            measure_memory: Whether to measure memory usage

        Returns:
            BenchmarkStats with comprehensive statistics
        """
        kwargs = kwargs or {}
        name = name or func.__name__

        if self.verbose:
            print(f"Benchmarking: {name}")
            print(f"  Warmup iterations: {warmup_iterations}")
            print(f"  Benchmark iterations: {iterations}")

        # Warmup
        for i in range(warmup_iterations):
            try:
                func(*args, **kwargs)
            except (TypeError, ValueError, KeyError, AttributeError, IndexError,
                    RuntimeError, ArithmeticError, MemoryError) as e:
                logger.warning(f"Warmup iteration {i+1} failed: {type(e).__name__}: {e}")

        # Benchmark iterations
        results: List[BenchmarkIteration] = []

        for i in range(iterations):
            start_time = time.perf_counter()
            start_memory = self._get_memory_mb() if measure_memory else 0

            try:
                result = func(*args, **kwargs)

                end_time = time.perf_counter()
                end_memory = self._get_memory_mb() if measure_memory else 0

                results.append(BenchmarkIteration(
                    iteration=i,
                    execution_time=end_time - start_time,
                    memory_mb=end_memory - start_memory if measure_memory else 0,
                    success=True
                ))

            except (TypeError, ValueError, KeyError, AttributeError, IndexError,
                    RuntimeError, ArithmeticError, MemoryError, OverflowError) as e:
                end_time = time.perf_counter()
                results.append(BenchmarkIteration(
                    iteration=i,
                    execution_time=end_time - start_time,
                    success=False,
                    error=f"{type(e).__name__}: {e}"
                ))

        # Calculate statistics
        successful_results = [r for r in results if r.success]
        execution_times = [r.execution_time for r in successful_results]

        if not execution_times:
            raise RuntimeError(f"All {iterations} iterations failed for {name}")

        stats = BenchmarkStats(
            name=name,
            total_time=sum(execution_times),
            mean_time=statistics.mean(execution_times),
            median_time=statistics.median(execution_times),
            std_dev=statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            min_time=min(execution_times),
            max_time=max(execution_times),
            iterations=iterations,
            success_rate=len(successful_results) / iterations,
            total_memory_mb=sum(r.memory_mb for r in successful_results) if measure_memory else 0
        )

        self.history.append(stats)

        if self.verbose:
            print(f"\n{stats}\n")

        return stats

    def compare(
        self,
        old_func: Callable,
        new_func: Callable,
        args: Tuple = (),
        kwargs: Dict = None,
        iterations: int = 100,
        warmup_iterations: int = 10,
        old_name: str = None,
        new_name: str = None
    ) -> ComparisonResult:
        """
        Compare two functions and calculate improvement.

        Args:
            old_func: Original (slow) function
            new_func: Optimized (fast) function
            args: Arguments to pass to both functions
            kwargs: Keyword arguments to pass to both functions
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            old_name: Name for old function
            new_name: Name for new function

        Returns:
            ComparisonResult with improvement metrics
        """
        kwargs = kwargs or {}
        old_name = old_name or old_func.__name__
        new_name = new_name or new_func.__name__

        if self.verbose:
            print(f"\nComparing: {old_name} vs {new_name}")
            print("="*70)

        # Benchmark both
        old_stats = self.benchmark(
            func=old_func,
            args=args,
            kwargs=kwargs,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            name=old_name
        )

        new_stats = self.benchmark(
            func=new_func,
            args=args,
            kwargs=kwargs,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            name=new_name
        )

        # Calculate improvement
        improvement_percent = (
            (old_stats.mean_time - new_stats.mean_time) / old_stats.mean_time * 100
        )

        speedup_factor = old_stats.mean_time / new_stats.mean_time

        memory_saving_percent = 0
        if old_stats.total_memory_mb > 0:
            memory_saving_percent = (
                (old_stats.total_memory_mb - new_stats.total_memory_mb) /
                old_stats.total_memory_mb * 100
            )

        comparison = ComparisonResult(
            old_stats=old_stats,
            new_stats=new_stats,
            improvement_percent=improvement_percent,
            speedup_factor=speedup_factor,
            memory_saving_percent=memory_saving_percent
        )

        if self.verbose:
            print(f"\n{comparison}\n")

        return comparison

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0

    def get_history(self) -> List[BenchmarkStats]:
        """Get all benchmark results from history"""
        return self.history.copy()

    def clear_history(self):
        """Clear benchmark history"""
        self.history.clear()


# ============================================================================
# Specific Benchmarks
# ============================================================================

def benchmark_config_access():
    """Benchmark config access patterns"""
    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        temperature: float = 0.7
        max_tokens: int = 2048
        top_p: float = 1.0
        frequency_penalty: float = 0.0
        presence_penalty: float = 0.0
        max_iterations: int = 100

    # OLD WAY: Direct attribute access in loop
    def old_way(config):
        result = 0
        for i in range(config.max_iterations):
            temp = config.temperature
            tokens = config.max_tokens
            top_p = config.top_p
            freq_pen = config.frequency_penalty
            pres_pen = config.presence_penalty
            result += temp + tokens + top_p
        return result

    # NEW WAY: Cached config access
    def new_way(config):
        from performance_optimizations import cache_config_loop

        cached = cache_config_loop(config, [
            'max_iterations', 'temperature', 'max_tokens',
            'top_p', 'frequency_penalty', 'presence_penalty'
        ])

        result = 0
        for i in range(cached['max_iterations']):
            temp = cached['temperature']
            tokens = cached['max_tokens']
            top_p = cached['top_p']
            freq_pen = cached['frequency_penalty']
            pres_pen = cached['presence_penalty']
            result += temp + tokens + top_p
        return result

    config = MockConfig()
    runner = BenchmarkRunner(verbose=True)

    comparison = runner.compare(
        old_func=old_way,
        new_func=new_way,
        args=(config,),
        iterations=100,
        warmup_iterations=10,
        old_name="Direct Config Access",
        new_name="Cached Config Access"
    )

    return comparison


def benchmark_membership_testing():
    """Benchmark list vs set membership testing"""
    import random

    # Create test data
    large_list = list(range(10000))
    test_items = [random.randint(0, 20000) for _ in range(1000)]

    # OLD WAY: List membership (O(n))
    def old_way(items, search_list):
        found = 0
        for item in items:
            if item in search_list:  # O(n) lookup
                found += 1
        return found

    # NEW WAY: Set membership (O(1))
    def new_way(items, search_list):
        from performance_optimizations import optimize_for_membership

        search_set = optimize_for_membership(search_list)
        found = 0
        for item in items:
            if item in search_set:  # O(1) lookup
                found += 1
        return found

    runner = BenchmarkRunner(verbose=True)

    comparison = runner.compare(
        old_func=old_way,
        new_func=new_way,
        args=(test_items, large_list),
        iterations=100,
        warmup_iterations=10,
        old_name="List Membership (O(n))",
        new_name="Set Membership (O(1))"
    )

    return comparison


def benchmark_string_operations():
    """Benchmark string concatenation patterns"""
    items = [f"item_{i}" for i in range(1000)]

    # OLD WAY: String concatenation in loop
    def old_way(items):
        result = ""
        for item in items:
            result += item + ", "  # Creates new string each time
        return result

    # NEW WAY: Join operation
    def new_way(items):
        return ", ".join(items)  # Single allocation

    runner = BenchmarkRunner(verbose=True)

    comparison = runner.compare(
        old_func=old_way,
        new_func=new_way,
        args=(items,),
        iterations=100,
        warmup_iterations=10,
        old_name="String Concatenation",
        new_name="String Join"
    )

    return comparison


def benchmark_dict_creation():
    """Benchmark dict creation patterns"""
    items = list(range(1000))

    # OLD WAY: Create dict in loop
    def old_way(items):
        results = []
        for item in items:
            result = {
                'value': item,
                'doubled': item * 2,
                'squared': item ** 2
            }
            results.append(result)
        return results

    # NEW WAY: Use list of tuples, convert later
    def new_way(items):
        results = []
        for item in items:
            result = (item, item * 2, item ** 2)  # Faster tuple
            results.append(result)
        return results

    runner = BenchmarkRunner(verbose=True)

    comparison = runner.compare(
        old_func=old_way,
        new_func=new_way,
        args=(items,),
        iterations=100,
        warmup_iterations=10,
        old_name="Dict Creation",
        new_name="Tuple Creation"
    )

    return comparison


# ============================================================================
# Main Entry Point
# ============================================================================

def run_all_benchmarks() -> Dict[str, ComparisonResult]:
    """Run all predefined benchmarks"""
    print("\n" + "="*70)
    print("OPENEVOLVE FRONTEND PERFORMANCE BENCHMARK SUITE")
    print("="*70 + "\n")

    results = {}

    print("\n1. Config Access Benchmark")
    print("-"*70)
    try:
        results['config_access'] = benchmark_config_access()
    except (ImportError, AttributeError, NameError, RuntimeError) as e:
        logger.error(f"Config access benchmark failed: {type(e).__name__}: {e}")

    print("\n2. Membership Testing Benchmark")
    print("-"*70)
    try:
        results['membership'] = benchmark_membership_testing()
    except (ImportError, AttributeError, NameError, RuntimeError) as e:
        logger.error(f"Membership testing benchmark failed: {type(e).__name__}: {e}")

    print("\n3. String Operations Benchmark")
    print("-"*70)
    try:
        results['strings'] = benchmark_string_operations()
    except (ImportError, AttributeError, NameError, RuntimeError) as e:
        logger.error(f"String operations benchmark failed: {type(e).__name__}: {e}")

    print("\n4. Dict Creation Benchmark")
    print("-"*70)
    try:
        results['dicts'] = benchmark_dict_creation()
    except (ImportError, AttributeError, NameError, RuntimeError) as e:
        logger.error(f"Dict creation benchmark failed: {type(e).__name__}: {e}")

    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    for name, comparison in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Speedup: {comparison.speedup_factor:.2f}x")
        print(f"  Improvement: {comparison.improvement_percent:+.1f}%")

    print("\n" + "="*70)

    return results


if __name__ == "__main__":
    # Run all benchmarks when executed directly
    results = run_all_benchmarks()
