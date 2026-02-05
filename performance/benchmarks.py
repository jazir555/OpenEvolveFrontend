"""
RESE Performance Benchmark Suite

Comprehensive benchmarks for all RESE components.

Author: Agent M1
"""

import time
import numpy as np
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a benchmark run"""
    name: str
    component: str
    input_size: int
    time_seconds: float
    operations_per_second: float
    memory_mb: float
    passed: bool
    metadata: Dict = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "component": self.component,
            "input_size": self.input_size,
            "time_seconds": self.time_seconds,
            "operations_per_second": self.operations_per_second,
            "memory_mb": self.memory_mb,
            "passed": self.passed,
            "metadata": self.metadata or {}
        }


class PerformanceBenchmark:
    """
    Performance benchmark suite for RESE components.

    Targets:
    - SCE: <1s for 10K constraints
    - DITO: <10s for 100K constraints
    - Phi1.5: <10s for 1K failures
    - I_mech: <30s for domain pairs
    - Gamma1: <5s for ACI calculation
    - MCTS: <60s for 1000 iterations
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def benchmark(self,
                  func: Callable,
                  name: str,
                  component: str,
                  input_size: int,
                  target_time: float,
                  **kwargs) -> BenchmarkResult:
        """
        Run a single benchmark.

        Args:
            func: Function to benchmark
            name: Benchmark name
            component: Component being tested
            input_size: Size of input data
            target_time: Target time in seconds
            **kwargs: Additional arguments for func

        Returns:
            BenchmarkResult
        """
        print(f"\n{'='*70}")
        print(f"Benchmark: {name}")
        print(f"Component: {component}")
        print(f"Input Size: {input_size}")
        print(f"Target: <{target_time}s")
        print(f"{'='*70}")

        # Warmup
        try:
            func(**{k: min(v, 10) for k, v in kwargs.items()})
        except:
            pass

        # Benchmark
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            result = func(**kwargs)
            success = True
        except Exception as e:
            print(f"Error: {e}")
            result = None
            success = False

        end_time = time.time()
        end_memory = self._get_memory_usage()

        elapsed = end_time - start_time
        memory_used = end_memory - start_memory

        # Calculate throughput
        ops_per_sec = input_size / elapsed if elapsed > 0 else 0

        # Check if target met
        passed = success and elapsed <= target_time

        result = BenchmarkResult(
            name=name,
            component=component,
            input_size=input_size,
            time_seconds=elapsed,
            operations_per_second=ops_per_sec,
            memory_mb=memory_used,
            passed=passed,
            metadata={
                "target_time": target_time,
                "result": str(result)[:100] if result else None
            }
        )

        self.results.append(result)

        # Print results
        print(f"\nResults:")
        print(f"  Time:       {elapsed:.4f}s")
        print(f"  Throughput: {ops_per_sec:.0f} ops/sec")
        print(f"  Memory:     {memory_used:.2f} MB")
        print(f"  Status:     {'[OK] PASS' if passed else '[FAIL] FAIL'}")

        if not passed:
            if not success:
                print(f"  Reason:     Function failed")
            else:
                print(f"  Reason:     Exceeded target by {elapsed - target_time:.4f}s")

        return result

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def generate_report(self, output_path: str = None) -> str:
        """
        Generate comprehensive benchmark report.

        Args:
            output_path: Optional file path to save report

        Returns:
            Report as string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("RESE PERFORMANCE BENCHMARK REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        total_passed = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)

        lines.append(f"Summary: {total_passed}/{total_tests} benchmarks passed")
        lines.append("")

        # Component breakdown
        components = {}
        for result in self.results:
            if result.component not in components:
                components[result.component] = []
            components[result.component].append(result)

        lines.append("Component Breakdown:")
        lines.append("-" * 80)

        for component, results in components.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)

            # Get average performance
            avg_time = np.mean([r.time_seconds for r in results])
            avg_throughput = np.mean([r.operations_per_second for r in results])

            lines.append(f"\n{component}:")
            lines.append(f"  Passed: {passed}/{total}")
            lines.append(f"  Avg Time: {avg_time:.4f}s")
            lines.append(f"  Avg Throughput: {avg_throughput:.0f} ops/sec")

            # Show individual benchmarks
            for r in results:
                status = "[OK]" if r.passed else "[FAIL]"
                lines.append(f"    {status} {r.name}: {r.time_seconds:.4f}s @ {r.input_size} items")

        lines.append("")
        lines.append("=" * 80)

        report = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(report)

        return report

    def save_results(self, output_path: str) -> None:
        """Save results to JSON file"""
        data = {
            "benchmark_results": [r.to_dict() for r in self.results],
            "summary": {
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed)
            }
        }

        Path(output_path).write_text(json.dumps(data, indent=2))

    def load_results(self, input_path: str) -> None:
        """Load results from JSON file"""
        data = json.loads(Path(input_path).read_text())
        self.results = [
            BenchmarkResult(**r) for r in data["benchmark_results"]
        ]


def create_sce_benchmark() -> Callable:
    """Create SCE benchmark function"""
    from performance.sce_optimizer import SCEOptimizer
    from core.symbolic_constraint_engine import Constraint, ConstraintType

    def sce_benchmark(num_constraints: int = 10000):
        optimizer = SCEOptimizer(use_parallel=True, num_threads=4)

        # Create constraints
        constraints = []
        for i in range(num_constraints):
            c = Constraint(
                id=f"constraint_{i}",
                type=ConstraintType.HARD,
                description=f"Constraint {i} with value {i % 100}",
                formalization=f"forall (x : int), x {['<', '>', '='][i % 3]} {i % 100}",
                source="benchmark"
            )
            constraints.append(c)

        # Add all constraints
        start = time.time()
        for c in constraints:
            optimizer.add_constraint(c)
        add_time = time.time() - start

        # Detect conflicts
        conflicts = optimizer.detect_conflicts()

        return {
            "constraints_added": num_constraints,
            "add_time": add_time,
            "conflicts_found": len(conflicts)
        }

    return sce_benchmark


def create_dito_benchmark() -> Callable:
    """Create DITO benchmark function"""
    from core.dito_optimizer import DITOOptimizer, DITOConfig
    from core.symbolic_constraint_engine import Constraint, ConstraintType

    def dito_benchmark(num_constraints: int = 100000):
        config = DITOConfig(
            parallel_enabled=True,
            num_threads=4
        )
        dito = DITOOptimizer(config)

        # Create constraints
        constraints = []
        for i in range(num_constraints):
            c = Constraint(
                id=f"dito_constraint_{i}",
                type=ConstraintType.HARD,
                description=f"Temperature must be {'less' if i % 2 == 0 else 'greater'} than {i % 1000}",
                formalization=f"forall (T : Temperature), T {['<', '>'][i % 2]} {i % 1000}",
                source="benchmark"
            )
            constraints.append(c)

        # Build DITO
        build_result = dito.build(constraints)

        # Detect contradictions
        contradictions = dito.detect_contradictions()

        return {
            "constraints_processed": build_result["constraints_processed"],
            "build_time": build_result["build_time"],
            "contradictions_found": len(contradictions)
        }

    return dito_benchmark


def create_phi15_benchmark() -> Callable:
    """Create Phi1.5 benchmark function"""
    from phase1.tacit_assumption_miner import (
        Phi15Engine, NullResult, ErrorType
    )
    from datetime import datetime

    def phi15_benchmark(num_failures: int = 1000):
        engine = Phi15Engine()

        # Create null results
        null_results = []
        for i in range(num_failures):
            nr = NullResult(
                attempt_id=f"attempt_{i}",
                timestamp=datetime.now(),
                problem_type=f"problem_type_{i % 10}",
                approach_type=f"approach_{i % 5}",
                constraints=[f"constraint_{j}" for j in range(i % 10)],
                error_type=list(ErrorType)[i % len(ErrorType)],
                error_message=f"Error {i}: Optimization failed with code {i % 100}",
                state={"iteration": i, "error_magnitude": i % 10},
                iteration=i,
                resources_used={"cpu": i % 100, "memory": i % 1000}
            )
            null_results.append(nr)

        # Process null results
        start = time.time()
        assumptions, paradigm_rec = engine.process_null_results(null_results)
        process_time = time.time() - start

        return {
            "failures_processed": num_failures,
            "assumptions_inferred": len(assumptions),
            "process_time": process_time,
            "paradigm_crisis": paradigm_rec.trigger
        }

    return phi15_benchmark


def create_gamma1_benchmark() -> Callable:
    """Create Gamma1 ACI benchmark function"""
    from gamma1.core.csp_models import CSPInstance, Variable, Domain, Constraint
    from gamma1.core.aci_calculator import ACICalculator

    def gamma1_benchmark(num_variables: int = 100):
        calculator = ACICalculator()

        # Create CSP instance
        variables = []
        for i in range(num_variables):
            var = Variable(
                name=f"x{i}",
                domain=Domain([f"value_{j}" for j in range(5)])
            )
            variables.append(var)

        constraints = []
        for i in range(num_variables - 1):
            c = Constraint(
                variables=[f"x{i}", f"x{i+1}"],
                allowed_tuples=[(f"value_{j}", f"value_{j}") for j in range(5)]
            )
            constraints.append(c)

        csp = CSPInstance(variables=variables, constraints=constraints)

        # Calculate ACI
        start = time.time()
        result = calculator.calculate(csp)
        calc_time = time.time() - start

        return {
            "variables": num_variables,
            "aci_score": result.ACI,
            "calculation_time": calc_time,
            "confidence": result.confidence
        }

    return gamma1_benchmark


def create_mcts_benchmark() -> Callable:
    """Create MCTS benchmark function"""
    from phase3.mcts_search import (
        MCTSConfig, MCTSSearch, MCTSState
    )

    def mcts_benchmark(num_iterations: int = 1000):
        config = MCTSConfig(
            max_iterations=num_iterations,
            num_workers=1,
            verbose=False
        )

        mcts = MCTSSearch(config)

        # Simple test problem
        class TestState(MCTSState):
            def __init__(self, value=0, depth=0):
                super().__init__()
                self.value_val = value
                self.depth_val = depth
                self.variables = {"value": value}
                self.unassigned = list(range(10 - depth))
                self.domains = {}
                self.depth = depth

            def is_terminal(self):
                return self.depth_val >= 10

        initial_state = TestState(value=0, depth=0)

        def simple_actions(state):
            if state.depth_val >= 10:
                return []
            return [-1, 0, 1]

        def simple_transition(state, action):
            new_value = state.value_val + action
            new_depth = state.depth_val + 1
            return TestState(new_value, new_depth)

        def simple_value(state):
            return state.value_val

        # Run MCTS
        best_node, info = mcts.search(
            initial_state,
            simple_actions,
            simple_transition,
            simple_value
        )

        return {
            "iterations": num_iterations,
            "elapsed_time": info["elapsed_time"],
            "best_value": info["best_value"],
            "tree_size": info["tree_size"]
        }

    return mcts_benchmark


if __name__ == "__main__":
    print("RESE Performance Benchmark Suite")
    print("=" * 70)

    # Create benchmark suite
    suite = PerformanceBenchmark()

    # Run benchmarks
    suite.benchmark(
        func=create_sce_benchmark(),
        name="SCE - Add and detect conflicts",
        component="SCE",
        input_size=10000,
        target_time=1.0,
        num_constraints=10000
    )

    suite.benchmark(
        func=create_gamma1_benchmark(),
        name="Gamma1 - ACI Calculation",
        component="Gamma1",
        input_size=100,
        target_time=5.0,
        num_variables=100
    )

    suite.benchmark(
        func=create_phi15_benchmark(),
        name="Phi1.5 - Process null results",
        component="Phi1.5",
        input_size=1000,
        target_time=10.0,
        num_failures=1000
    )

    suite.benchmark(
        func=create_mcts_benchmark(),
        name="MCTS - Search iterations",
        component="MCTS",
        input_size=1000,
        target_time=60.0,
        num_iterations=1000
    )

    # Generate report
    report = suite.generate_report()
    print("\n" + report)

    # Save results
    suite.save_results("benchmark_results.json")
    print("\nResults saved to benchmark_results.json")
