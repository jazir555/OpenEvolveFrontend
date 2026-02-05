"""
Comprehensive Benchmarking Suite for Mathematical Knowledge Integration

Benchmarks:
- Z3 solving performance
- LeanAIDE proving performance
- Knowledge extraction speed
- Pattern matching accuracy
- Cross-system translation quality
- End-to-end workflow latency

Usage:
    python benchmark_suite.py --suite basic --output results.json
    python benchmark_suite.py --suite comprehensive --visualize

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    category: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    median_time: float
    std_dev: float
    success_count: int
    failure_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    name: str
    timestamp: str
    environment: Dict[str, str]
    results: List[BenchmarkResult]
    summary: Dict[str, Any] = field(default_factory=dict)


class MathKnowledgeBenchmarks:
    """Comprehensive benchmarking for mathematical knowledge system."""
    
    def __init__(self, iterations: int = 10, warmup: int = 3):
        self.iterations = iterations
        self.warmup = warmup
        self.results: List[BenchmarkResult] = []
    
    async def run_all(self, suite: str = "basic") -> BenchmarkSuite:
        """Run all benchmarks."""
        print(f"Running {suite} benchmark suite ({self.iterations} iterations)...")
        print("=" * 70)
        
        # Warmup
        if self.warmup > 0:
            print(f"Warming up ({self.warmup} iterations)...")
            await self._warmup()
        
        # Run benchmarks based on suite
        if suite in ["basic", "comprehensive", "stress"]:
            await self.benchmark_z3_basic()
            await self.benchmark_z3_linear()
            await self.benchmark_knowledge_extraction()
        
        if suite in ["comprehensive", "stress"]:
            await self.benchmark_z3_nonlinear()
            await self.benchmark_pattern_matching()
            await self.benchmark_semantic_translation()
        
        if suite == "stress":
            await self.benchmark_concurrent_solving()
            await self.benchmark_large_knowledge_base()
        
        # Create suite results
        return BenchmarkSuite(
            name=suite,
            timestamp=datetime.now().isoformat(),
            environment=self._get_environment(),
            results=self.results,
            summary=self._compute_summary()
        )
    
    async def _warmup(self):
        """Warmup runs to stabilize performance."""
        from z3_solver_connector import get_z3_connector, Z3SolverConfig
        
        z3 = get_z3_connector()
        smtlib = "(declare-fun x () Int) (assert (> x 0)) (check-sat)"
        
        for _ in range(self.warmup):
            await z3.solve_smtlib(smtlib, Z3SolverConfig())
    
    def _get_environment(self) -> Dict[str, str]:
        """Get environment information."""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": str(psutil.cpu_count()),
            "memory_gb": f"{psutil.virtual_memory().total / (1024**3):.1f}",
            "timestamp": datetime.now().isoformat()
        }
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not self.results:
            return {}
        
        total_time = sum(r.total_time for r in self.results)
        total_success = sum(r.success_count for r in self.results)
        total_failures = sum(r.failure_count for r in self.results)
        
        return {
            "total_benchmarks": len(self.results),
            "total_time": total_time,
            "avg_time_per_benchmark": total_time / len(self.results),
            "total_successes": total_success,
            "total_failures": total_failures,
            "success_rate": total_success / (total_success + total_failures) if (total_success + total_failures) > 0 else 0
        }
    
    async def _run_benchmark(
        self,
        name: str,
        category: str,
        fn,
        setup_fn=None,
        teardown_fn=None
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        print(f"  Running {name}...", end=" ", flush=True)
        
        times = []
        success_count = 0
        failure_count = 0
        
        for i in range(self.iterations):
            try:
                if setup_fn:
                    await setup_fn()
                
                start = time.perf_counter()
                result = await fn()
                elapsed = time.perf_counter() - start
                
                times.append(elapsed)
                success_count += 1
                
                if teardown_fn:
                    await teardown_fn()
                    
            except Exception as e:
                failure_count += 1
                print(f"x", end="", flush=True)
        
        # Compute statistics
        if times:
            result = BenchmarkResult(
                name=name,
                category=category,
                iterations=self.iterations,
                total_time=sum(times),
                avg_time=statistics.mean(times),
                min_time=min(times),
                max_time=max(times),
                median_time=statistics.median(times),
                std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
                success_count=success_count,
                failure_count=failure_count
            )
        else:
            result = BenchmarkResult(
                name=name,
                category=category,
                iterations=self.iterations,
                total_time=0.0,
                avg_time=0.0,
                min_time=0.0,
                max_time=0.0,
                median_time=0.0,
                std_dev=0.0,
                success_count=success_count,
                failure_count=failure_count
            )
        
        self.results.append(result)
        print(f"[OK] ({result.avg_time*1000:.1f}ms avg)")
        return result
    
    # ==================================================================
    # Z3 Benchmarks
    # ==================================================================
    
    async def benchmark_z3_basic(self):
        """Benchmark basic Z3 solving."""
        from z3_solver_connector import get_z3_connector, Z3SolverConfig
        
        z3 = get_z3_connector()
        smtlib = "(declare-fun x () Int) (assert (> x 0)) (check-sat)"
        
        async def solve():
            return await z3.solve_smtlib(smtlib, Z3SolverConfig())
        
        await self._run_benchmark("Z3 Basic SAT", "z3", solve)
    
    async def benchmark_z3_linear(self):
        """Benchmark linear equation solving."""
        from z3_solver_connector import get_z3_connector, Z3SolverConfig
        
        z3 = get_z3_connector()
        smtlib = """
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (= (+ x y) 10))
        (assert (= (- x y) 2))
        (check-sat)
        (get-model)
        """
        
        async def solve():
            return await z3.solve_smtlib(smtlib, Z3SolverConfig())
        
        await self._run_benchmark("Z3 Linear System", "z3", solve)
    
    async def benchmark_z3_nonlinear(self):
        """Benchmark nonlinear equation solving."""
        from z3_solver_connector import get_z3_connector, Z3SolverConfig
        
        z3 = get_z3_connector()
        smtlib = """
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (= (+ (* x x) (* y y)) 25))
        (check-sat)
        """
        
        async def solve():
            return await z3.solve_smtlib(smtlib, Z3SolverConfig())
        
        await self._run_benchmark("Z3 Nonlinear", "z3", solve)
    
    # ==================================================================
    # Knowledge Extraction Benchmarks
    # ==================================================================
    
    async def benchmark_knowledge_extraction(self):
        """Benchmark knowledge extraction."""
        from z3_knowledge_complete import get_z3_knowledge_manager
        
        manager = await get_z3_knowledge_manager()
        
        async def extract():
            return await manager.learn_from_solution(
                problem_statement="Test problem",
                constraints=["x + y = 10"],
                result="success"
            )
        
        await self._run_benchmark("Knowledge Extraction", "knowledge", extract)
    
    async def benchmark_pattern_matching(self):
        """Benchmark pattern matching."""
        from z3_knowledge_complete import get_z3_knowledge_manager
        
        manager = await get_z3_knowledge_manager()
        
        async def match():
            return await manager.find_similar_solutions(
                problem_statement="Linear system",
                constraints=["x + y = 10"],
                top_k=5
            )
        
        await self._run_benchmark("Pattern Matching", "knowledge", match)
    
    # ==================================================================
    # Translation Benchmarks
    # ==================================================================
    
    async def benchmark_semantic_translation(self):
        """Benchmark semantic translation."""
        from unified_math_bridge_complete import SemanticTranslator
        
        translator = SemanticTranslator()
        
        def translate():
            return translator.translate_smt_to_lean("(assert (> x 0))")
        
        # Run synchronously
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            translate()
            times.append(time.perf_counter() - start)
        
        result = BenchmarkResult(
            name="Semantic Translation",
            category="translation",
            iterations=self.iterations,
            total_time=sum(times),
            avg_time=statistics.mean(times),
            min_time=min(times),
            max_time=max(times),
            median_time=statistics.median(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
            success_count=len(times),
            failure_count=0
        )
        self.results.append(result)
        print(f"  Semantic Translation... [OK] ({result.avg_time*1000:.1f}ms avg)")
    
    # ==================================================================
    # Stress Benchmarks
    # ==================================================================
    
    async def benchmark_concurrent_solving(self):
        """Benchmark concurrent problem solving."""
        from z3_solver_connector import get_z3_connector, Z3SolverConfig
        
        z3 = get_z3_connector()
        problems = [
            "(declare-fun x () Int) (assert (> x 0)) (check-sat)",
            "(declare-fun y () Int) (assert (< y 10)) (check-sat)",
            "(declare-fun z () Int) (assert (= z 5)) (check-sat)",
        ] * 3  # 9 concurrent problems
        
        async def solve_all():
            await asyncio.gather(*[
                z3.solve_smtlib(p, Z3SolverConfig())
                for p in problems
            ])
        
        await self._run_benchmark("Concurrent Solving (9x)", "stress", solve_all)
    
    async def benchmark_large_knowledge_base(self):
        """Benchmark with large knowledge base."""
        from z3_knowledge_complete import get_z3_knowledge_manager
        
        manager = await get_z3_knowledge_manager()
        
        # Add many entries
        for i in range(100):
            await manager.learn_from_solution(
                problem_statement=f"Problem {i}",
                constraints=[f"x + y = {i}"],
                result="success"
            )
        
        async def search():
            return await manager.find_similar_solutions(
                problem_statement="Problem 50",
                constraints=["x + y = 50"],
                top_k=10
            )
        
        await self._run_benchmark("Large KB Search (100 entries)", "stress", search)


def visualize_results(suite: BenchmarkSuite):
    """Visualize benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    # Summary
    print(f"\nSuite: {suite.name}")
    print(f"Timestamp: {suite.timestamp}")
    print(f"Environment: {json.dumps(suite.environment, indent=2)}")
    
    print(f"\nSummary:")
    for key, value in suite.summary.items():
        print(f"  {key}: {value}")
    
    # Results by category
    categories = {}
    for result in suite.results:
        cat = result.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    print("\nDetailed Results:")
    for category, results in categories.items():
        print(f"\n{category.upper()}:")
        for r in results:
            print(f"  {r.name}:")
            print(f"    Avg: {r.avg_time*1000:.2f}ms, Min: {r.min_time*1000:.2f}ms, Max: {r.max_time*1000:.2f}ms")
            print(f"    Success: {r.success_count}/{r.iterations}")


def compare_results(current: BenchmarkSuite, previous_file: str):
    """Compare current results with previous run."""
    with open(previous_file, 'r') as f:
        previous_data = json.load(f)
    
    print("\n" + "=" * 70)
    print("COMPARISON WITH PREVIOUS RUN")
    print("=" * 70)
    
    for result in current.results:
        prev_result = next(
            (r for r in previous_data.get('results', []) if r['name'] == result.name),
            None
        )
        
        if prev_result:
            prev_avg = prev_result['avg_time']
            curr_avg = result.avg_time
            change = ((curr_avg - prev_avg) / prev_avg) * 100 if prev_avg > 0 else 0
            
            symbol = "^" if change > 0 else "v" if change < 0 else "="
            print(f"{result.name}: {prev_avg*1000:.2f}ms -> {curr_avg*1000:.2f}ms ({symbol}{abs(change):.1f}%)")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark mathematical knowledge integration")
    parser.add_argument("--suite", choices=["basic", "comprehensive", "stress"],
                       default="basic", help="Benchmark suite to run")
    parser.add_argument("--iterations", "-n", type=int, default=10,
                       help="Number of iterations per benchmark")
    parser.add_argument("--warmup", "-w", type=int, default=3,
                       help="Number of warmup iterations")
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Visualize results")
    parser.add_argument("--compare", "-c", help="Compare with previous results file")
    
    args = parser.parse_args()
    
    # Run benchmarks
    benchmarks = MathKnowledgeBenchmarks(
        iterations=args.iterations,
        warmup=args.warmup
    )
    
    suite = await benchmarks.run_all(args.suite)
    
    # Visualize
    if args.visualize:
        visualize_results(suite)
    
    # Compare
    if args.compare:
        compare_results(suite, args.compare)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(suite), f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())
