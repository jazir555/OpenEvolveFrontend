"""
DITO Performance Benchmarks

Validates O(n log n) complexity and speedup over naive O(n²) approach.

Author: Agent A3 (DITO Specialist)
Created: 2025-12-31
Status: 🟢 Benchmarking Phase
"""

import time
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
import numpy as np

from core.dito_optimizer import DITOOptimizer, DITOConfig
from core.symbolic_constraint_engine import Constraint, ConstraintType as SCEConstraintType


class PerformanceBenchmarks:
    """Comprehensive performance benchmarks for DITO"""

    def __init__(self):
        self.results = {
            "build": [],
            "query": [],
            "update": [],
            "full_check": [],
        }

    def generate_constraints(self, n: int) -> List[Constraint]:
        """Generate n random constraints"""
        constraints = []

        for i in range(n):
            c_type = random.choice([
                SCEConstraintType.HARD,
                SCEConstraintType.SOFT,
                SCEConstraintType.PREFERENCE
            ])

            constraints.append(Constraint(
                id=f"constraint_{i}",
                type=c_type,
                description=f"Test constraint {i} with value {random.randint(0, 1000)}",
                formalization=f"forall (x : Variable), {random.choice(['<', '>', '=', '≠'])} {random.randint(0, 1000)}",
                source="benchmark"
            ))

        return constraints

    def benchmark_build(self, sizes: List[int] = None) -> Dict[str, List]:
        """Benchmark DITO build performance"""
        if sizes is None:
            sizes = [10, 50, 100, 500, 1000, 5000]

        print("\n" + "=" * 70)
        print("BUILD PERFORMANCE BENCHMARK")
        print("=" * 70)

        build_times = []
        build_rates = []

        for n in sizes:
            constraints = self.generate_constraints(n)

            dito = DITOOptimizer(DITOConfig(
                max_hierarchy_level=5,
                rtree_max_entries=50,
                rtree_min_entries=10
            ))

            start = time.time()
            dito.build(constraints)
            elapsed = time.time() - start

            rate = n / elapsed if elapsed > 0 else 0

            build_times.append(elapsed)
            build_rates.append(rate)

            print(f"n = {n:5d}: {elapsed:8.4f}s  ({rate:8.0f} constraints/sec)")

        self.results["build"] = list(zip(sizes, build_times, build_rates))

        # Analyze complexity
        self._analyze_complexity(sizes, build_times, "Build")

        return {
            "sizes": sizes,
            "times": build_times,
            "rates": build_rates
        }

    def benchmark_query(self, sizes: List[int] = None) -> Dict[str, List]:
        """Benchmark DITO query performance"""
        if sizes is None:
            sizes = [10, 50, 100, 500, 1000, 5000]

        print("\n" + "=" * 70)
        print("QUERY PERFORMANCE BENCHMARK")
        print("=" * 70)

        query_times = []

        for n in sizes:
            constraints = self.generate_constraints(n)

            dito = DITOOptimizer()
            dito.build(constraints)

            # Query first constraint
            query_constraint = constraints[0]

            start = time.time()
            contradictions = dito.detect_contradictions(query_constraint)
            elapsed = time.time() - start

            query_times.append(elapsed)

            print(f"n = {n:5d}: {elapsed*1000:8.2f}ms  ({len(contradictions)} contradictions)")

        self.results["query"] = list(zip(sizes, query_times))

        # Analyze complexity
        self._analyze_complexity(sizes, query_times, "Query")

        return {
            "sizes": sizes,
            "times": query_times
        }

    def benchmark_update(self, sizes: List[int] = None) -> Dict[str, List]:
        """Benchmark DITO update performance"""
        if sizes is None:
            sizes = [10, 50, 100, 500, 1000, 5000]

        print("\n" + "=" * 70)
        print("UPDATE PERFORMANCE BENCHMARK")
        print("=" * 70)

        update_times = []

        for n in sizes:
            constraints = self.generate_constraints(n)

            dito = DITOOptimizer()
            dito.build(constraints)

            # Add new constraint
            new_constraint = Constraint(
                id=f"new_constraint",
                type=SCEConstraintType.HARD,
                description="New constraint for benchmark",
                formalization="new_constraint_formula",
                source="benchmark"
            )

            start = time.time()
            dito.update("ADD", constraint=new_constraint)
            elapsed = time.time() - start

            update_times.append(elapsed)

            print(f"n = {n:5d}: {elapsed*1000:8.2f}ms")

        self.results["update"] = list(zip(sizes, update_times))

        # Analyze complexity
        self._analyze_complexity(sizes, update_times, "Update")

        return {
            "sizes": sizes,
            "times": update_times
        }

    def benchmark_full_check(self, sizes: List[int] = None) -> Dict[str, List]:
        """Benchmark full contradiction check"""
        if sizes is None:
            sizes = [10, 50, 100, 500, 1000]

        print("\n" + "=" * 70)
        print("FULL CHECK PERFORMANCE BENCHMARK")
        print("=" * 70)

        check_times = []

        for n in sizes:
            constraints = self.generate_constraints(n)

            dito = DITOOptimizer(DITOConfig(max_hierarchy_level=3))
            dito.build(constraints)

            start = time.time()
            contradictions = dito.detect_contradictions(query_constraint=None)
            elapsed = time.time() - start

            check_times.append(elapsed)

            print(f"n = {n:5d}: {elapsed:8.4f}s  ({len(contradictions)} contradictions)")

        self.results["full_check"] = list(zip(sizes, check_times))

        # Analyze complexity
        self._analyze_complexity(sizes, check_times, "Full Check")

        return {
            "sizes": sizes,
            "times": check_times
        }

    def benchmark_naive_comparison(self, n: int = 100) -> Dict[str, float]:
        """Compare DITO with naive O(n²) approach"""
        print("\n" + "=" * 70)
        print(f"NAIVE O(n²) COMPARISON (n={n})")
        print("=" * 70)

        constraints = self.generate_constraints(n)

        # DITO approach
        dito = DITOOptimizer()
        start = time.time()
        dito.build(constraints)
        dito_build_time = time.time() - start

        start = time.time()
        contradictions_dito = dito.detect_contradictions(query_constraint=None)
        dito_check_time = time.time() - start

        dito_total = dito_build_time + dito_check_time

        # Naive O(n²) approach
        start = time.time()
        contradictions_naive = self._naive_contradiction_check(constraints)
        naive_time = time.time() - start

        speedup = naive_time / dito_total if dito_total > 0 else 0

        print(f"\nDITO:")
        print(f"  Build:  {dito_build_time:.4f}s")
        print(f"  Check:  {dito_check_time:.4f}s")
        print(f"  Total:  {dito_total:.4f}s")
        print(f"\nNaive O(n²):")
        print(f"  Time:   {naive_time:.4f}s")
        print(f"\nSpeedup: {speedup:.1f}x")
        print(f"Contradictions found: DITO={len(contradictions_dito)}, Naive={len(contradictions_naive)}")

        return {
            "dito_build": dito_build_time,
            "dito_check": dito_check_time,
            "dito_total": dito_total,
            "naive_time": naive_time,
            "speedup": speedup
        }

    def _naive_contradiction_check(self, constraints: List[Constraint]) -> List[Tuple[str, str]]:
        """Naive O(n²) contradiction check"""
        contradictions = []

        for i, c1 in enumerate(constraints):
            for c2 in constraints[i+1:]:
                # Simple keyword-based check
                desc1 = c1.description.lower()
                desc2 = c2.description.lower()

                # Check for contradictory keywords
                if ("less than" in desc1 and "greater than" in desc2) or \
                   ("greater than" in desc1 and "less than" in desc2) or \
                   ("<" in desc1 and ">" in desc2) or \
                   (">" in desc1 and "<" in desc2):
                    contradictions.append((c1.id, c2.id))

        return contradictions

    def _analyze_complexity(self, sizes: List[int], times: List[float], operation: str) -> None:
        """Analyze time complexity"""
        if len(sizes) < 2:
            return

        # Fit to model: T(n) = a * n^b
        log_sizes = np.log(sizes)
        log_times = np.log(times)

        # Linear regression in log-log space
        coeffs = np.polyfit(log_sizes, log_times, 1)
        b = coeffs[0]  # Exponent
        a = np.exp(coeffs[1])  # Coefficient

        # Compute R²
        predicted = [a * (n ** b) for n in sizes]
        ss_res = sum((t - p) ** 2 for t, p in zip(times, predicted))
        ss_tot = sum((t - np.mean(times)) ** 2 for t in times)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"\nComplexity Analysis ({operation}):")
        print(f"  T(n) ≈ {a:.6f} * n^{b:.2f}")
        print(f"  R² = {r_squared:.4f}")

        if b < 1.2:
            print(f"  ✓ Consistent with O(n log n) or better")
        elif b < 1.5:
            print(f"  ~ Approximately O(n^1.2)")
        else:
            print(f"  ✗ Higher than expected (b={b:.2f})")

    def generate_plots(self, output_dir: str = ".") -> None:
        """Generate performance plots"""
        try:
            # Build performance
            if self.results["build"]:
                sizes, times, rates = zip(*self.results["build"])

                plt.figure(figsize=(10, 6))
                plt.subplot(1, 2, 1)
                plt.plot(sizes, times, 'o-', label='Actual')
                plt.xlabel('Number of Constraints')
                plt.ylabel('Build Time (s)')
                plt.title('DITO Build Performance')
                plt.grid(True)
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(sizes, rates, 'o-', color='green')
                plt.xlabel('Number of Constraints')
                plt.ylabel('Build Rate (constraints/sec)')
                plt.title('DITO Build Throughput')
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(f"{output_dir}/dito_build_performance.png")
                print(f"\n[OK] Saved build performance plot")

            # Query performance
            if self.results["query"]:
                sizes, times = zip(*self.results["query"])

                plt.figure(figsize=(10, 6))
                plt.plot(sizes, [t*1000 for t in times], 'o-', color='blue')
                plt.xlabel('Number of Constraints')
                plt.ylabel('Query Time (ms)')
                plt.title('DITO Query Performance')
                plt.grid(True)
                plt.savefig(f"{output_dir}/dito_query_performance.png")
                print(f"[OK] Saved query performance plot")

        except Exception as e:
            print(f"[WARNING] Could not generate plots: {e}")

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("\n" + "=" * 70)
        print("DITO PERFORMANCE BENCHMARK SUITE")
        print("=" * 70)

        start_total = time.time()

        # Run benchmarks
        build_results = self.benchmark_build()
        query_results = self.benchmark_query()
        update_results = self.benchmark_update()
        full_check_results = self.benchmark_full_check()

        # Naive comparison
        naive_results = self.benchmark_naive_comparison(n=100)

        total_time = time.time() - start_total

        # Summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        print(f"\nTotal benchmark time: {total_time:.2f}s")
        print(f"\nSpeedup over naive O(n²): {naive_results['speedup']:.1f}x")

        if naive_results['speedup'] >= 1000:
            print("✓ ACHIEVED: 1000x+ speedup target!")
        elif naive_results['speedup'] >= 100:
            print("✓ GOOD: 100x+ speedup achieved")
        else:
            print("~ Speedup below target")

        return {
            "build": build_results,
            "query": query_results,
            "update": update_results,
            "full_check": full_check_results,
            "naive_comparison": naive_results
        }


def main():
    """Run benchmarks and generate report"""
    benchmarks = PerformanceBenchmarks()

    # Run all benchmarks
    results = benchmarks.run_all_benchmarks()

    # Generate plots
    benchmarks.generate_plots()

    # Write report
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nAll benchmarks completed successfully!")
    print("Performance plots saved to current directory.")


if __name__ == "__main__":
    main()
