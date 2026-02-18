"""
Performance Benchmarks for Adaptive MDAP/MAKER Adapter

Measures and reports on adapter performance characteristics.
"""

import os
import sys
import time
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from .adaptive_mdap_adapter import (
    get_adapter,
    CanonicalSubProblem,
    CanonicalComplexityScore,
    TaskStatus
)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    ops_per_second: float
    success_rate: float
    timestamp: str


class BenchmarkSuite:
    """Performance benchmark suite for the adapter."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.adapter = get_adapter()
        self.results: List[BenchmarkResult] = []

    def _measure_operation(
        self,
        operation: Callable[[], Any],
        iterations: int = 100
    ) -> List[float]:
        """
        Measure operation execution time.

        Returns:
            List of execution times in milliseconds
        """
        times = []

        for _ in range(iterations):
            start = time.perf_counter()

            try:
                operation()
                times.append((time.perf_counter() - start) * 1000)
            except Exception:
                # Failed operations are not counted
                pass

        return times

    def benchmark_complexity_analysis(self, iterations: int = 100) -> BenchmarkResult:
        """
        Benchmark complexity analysis performance.

        Args:
            iterations: Number of iterations

        Returns:
            Benchmark result
        """
        print(f"Benchmarking complexity analysis ({iterations} iterations)...")

        def operation():
            subproblem = CanonicalSubProblem(
                id="bench-test",
                description="Test problem for benchmarking",
                domain="benchmarking",
                depth=2,
                dependencies=["dep1", "dep2"]
            )
            self.adapter.analyze_complexity(subproblem)

        times = self._measure_operation(operation, iterations)

        if not times:
            return BenchmarkResult(
                name="complexity_analysis",
                iterations=iterations,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                p50_time_ms=0,
                p95_time_ms=0,
                p99_time_ms=0,
                ops_per_second=0,
                success_rate=0,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        return BenchmarkResult(
            name="complexity_analysis",
            iterations=len(times),
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            p50_time_ms=statistics.median(times),
            p95_time_ms=statistics.quantiles(times, n=20)[18],  # 95th percentile
            p99_time_ms=statistics.quantiles(times, n=100)[98],  # 99th percentile
            ops_per_second=1000 / statistics.mean(times) if times else 0,
            success_rate=len(times) / iterations,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def benchmark_resource_allocation(self, iterations: int = 100) -> BenchmarkResult:
        """
        Benchmark resource allocation performance.

        Args:
            iterations: Number of iterations

        Returns:
            Benchmark result
        """
        print(f"Benchmarking resource allocation ({iterations} iterations)...")

        def operation():
            complexity = CanonicalComplexityScore(overall_score=0.5)
            self.adapter.allocate_resources(complexity)

        times = self._measure_operation(operation, iterations)

        if not times:
            return BenchmarkResult(
                name="resource_allocation",
                iterations=iterations,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                p50_time_ms=0,
                p95_time_ms=0,
                p99_time_ms=0,
                ops_per_second=0,
                success_rate=0,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        return BenchmarkResult(
            name="resource_allocation",
            iterations=len(times),
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            p50_time_ms=statistics.median(times),
            p95_time_ms=statistics.quantiles(times, n=20)[18],
            p99_time_ms=statistics.quantiles(times, n=100)[98],
            ops_per_second=1000 / statistics.mean(times) if times else 0,
            success_rate=len(times) / iterations,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def benchmark_health_check(self, iterations: int = 1000) -> BenchmarkResult:
        """
        Benchmark health check performance.

        Args:
            iterations: Number of iterations

        Returns:
            Benchmark result
        """
        print(f"Benchmarking health check ({iterations} iterations)...")

        def operation():
            self.adapter.health_check()

        times = self._measure_operation(operation, iterations)

        if not times:
            return BenchmarkResult(
                name="health_check",
                iterations=iterations,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                p50_time_ms=0,
                p95_time_ms=0,
                p99_time_ms=0,
                ops_per_second=0,
                success_rate=0,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        return BenchmarkResult(
            name="health_check",
            iterations=len(times),
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            p50_time_ms=statistics.median(times),
            p95_time_ms=statistics.quantiles(times, n=20)[18],
            p99_time_ms=statistics.quantiles(times, n=100)[98],
            ops_per_second=1000 / statistics.mean(times) if times else 0,
            success_rate=len(times) / iterations,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """
        Run all benchmarks.

        Returns:
            List of benchmark results
        """
        print("=" * 70)
        print("ADAPTIVE MDAP/MAKER ADAPTER BENCHMARK SUITE")
        print("=" * 70)
        print()

        results = []

        # Run benchmarks
        results.append(self.benchmark_health_check(iterations=1000))
        print()

        results.append(self.benchmark_complexity_analysis(iterations=100))
        print()

        results.append(self.benchmark_resource_allocation(iterations=100))
        print()

        # Store results
        self.results = results

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: List[BenchmarkResult]):
        """Print benchmark summary."""
        print("=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print()

        for result in results:
            print(f"{result.name.replace('_', ' ').title()}")
            print("-" * 70)
            print(f"  Iterations:     {result.iterations}")
            print(f"  Success Rate:   {result.success_rate*100:.1f}%")
            print(f"  Avg Time:       {result.avg_time_ms:.2f} ms")
            print(f"  Min Time:       {result.min_time_ms:.2f} ms")
            print(f"  Max Time:       {result.max_time_ms:.2f} ms")
            print(f"  Median (p50):   {result.p50_time_ms:.2f} ms")
            print(f"  95th (p95):     {result.p95_time_ms:.2f} ms")
            print(f"  99th (p99):     {result.p99_time_ms:.2f} ms")
            print(f"  Throughput:     {result.ops_per_second:.1f} ops/sec")
            print()

        # Performance classification
        print("Performance Classification:")
        print("-" * 70)

        for result in results:
            ops_per_sec = result.ops_per_second
            if ops_per_sec > 1000:
                classification = "EXCELLENT"
            elif ops_per_sec > 500:
                classification = "GOOD"
            elif ops_per_sec > 100:
                classification = "ACCEPTABLE"
            else:
                classification = "NEEDS OPTIMIZATION"

            print(f"  {result.name}: {classification} ({ops_per_sec:.1f} ops/sec)")

    def export_results(self, filepath: str = "benchmark_results.json"):
        """
        Export benchmark results to JSON.

        Args:
            filepath: Output file path
        """
        import json

        export_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": [
                {
                    "name": r.name,
                    "iterations": r.iterations,
                    "total_time_ms": r.total_time_ms,
                    "avg_time_ms": r.avg_time_ms,
                    "min_time_ms": r.min_time_ms,
                    "max_time_ms": r.max_time_ms,
                    "p50_time_ms": r.p50_time_ms,
                    "p95_time_ms": r.p95_time_ms,
                    "p99_time_ms": r.p99_time_ms,
                    "ops_per_second": r.ops_per_second,
                    "success_rate": r.success_rate
                }
                for r in self.results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Benchmark results exported to: {filepath}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run adapter benchmarks")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--export", type=str, default="benchmark_results.json", help="Export results to file")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks (fewer iterations)")

    args = parser.parse_args()

    suite = BenchmarkSuite()

    if args.quick:
        suite.benchmark_health_check(iterations=100)
    else:
        suite.run_all_benchmarks()
        suite.export_results(args.export)
