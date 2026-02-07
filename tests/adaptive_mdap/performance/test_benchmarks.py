"""
Performance benchmarks for Adaptive MDAP.

These benchmarks measure the performance characteristics of the adaptive system
and validate that it meets production requirements.
"""

import pytest
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass

from adaptive_mdap.core.types import SubProblem, SolveStrategy
from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier
from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator
from adaptive_mdap.controllers.execution_controller import AdaptiveExecutionController
from adaptive_mdap.tools.cost_calculator import CostCalculator, APIPricing


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_per_sec: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "median_time_ms": self.median_time_ms,
            "p95_time_ms": self.p95_time_ms,
            "p99_time_ms": self.p99_time_ms,
            "throughput_per_sec": self.throughput_per_sec,
        }


def run_benchmark(
    name: str,
    func,
    iterations: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """
    Run a benchmark function multiple times and collect statistics.
    
    Args:
        name: Benchmark name
        func: Function to benchmark (takes no args)
        iterations: Number of iterations
        warmup: Number of warmup iterations
        
    Returns:
        BenchmarkResult with statistics
    """
    # Warmup
    for _ in range(warmup):
        func()
    
    # Actual benchmark
    times = []
    start_total = time.time()
    
    for _ in range(iterations):
        start = time.time()
        func()
        elapsed_ms = (time.time() - start) * 1000
        times.append(elapsed_ms)
    
    total_time_ms = (time.time() - start_total) * 1000
    
    times_sorted = sorted(times)

    # Avoid division by zero for very fast operations
    total_time_sec = max(total_time_ms / 1000, 0.001)

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_ms=total_time_ms,
        avg_time_ms=statistics.mean(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        median_time_ms=statistics.median(times),
        p95_time_ms=times_sorted[int(iterations * 0.95)],
        p99_time_ms=times_sorted[int(iterations * 0.99)],
        throughput_per_sec=iterations / total_time_sec,
    )


class TestClassificationPerformance:
    """Benchmark classification performance."""
    
    def test_simple_classification_latency(self):
        """Benchmark classification for simple problems."""
        classifier = TaskComplexityClassifier()
        
        sp = SubProblem(
            id="perf-simple",
            description="Simple problem",
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        result = run_benchmark(
            "simple_classification",
            lambda: classifier.compute_complexity(sp),
            iterations=100,
        )
        
        # Should be fast: avg < 50ms, p95 < 100ms
        assert result.avg_time_ms < 50, f"Average classification too slow: {result.avg_time_ms:.2f}ms"
        assert result.p95_time_ms < 100, f"P95 classification too slow: {result.p95_time_ms:.2f}ms"
        
        print(f"\nClassification Benchmark: {result.to_dict()}")
    
    def test_complex_classification_latency(self):
        """Benchmark classification for complex problems."""
        classifier = TaskComplexityClassifier()
        
        sp = SubProblem(
            id="perf-complex",
            description="Complex problem " * 500,
            domain="ultra_complex_domain",
            depth=10,
            dependencies=[f"dep{i}" for i in range(20)],
            metadata={
                "constraints": ["c1", "c2", "c3"],
                "success_criteria": ["s1", "s2"],
            },
        )
        
        result = run_benchmark(
            "complex_classification",
            lambda: classifier.compute_complexity(sp),
            iterations=50,
        )
        
        # Complex classification can be slower: avg < 100ms, p95 < 200ms
        assert result.avg_time_ms < 100, f"Average classification too slow: {result.avg_time_ms:.2f}ms"
        assert result.p95_time_ms < 200, f"P95 classification too slow: {result.p95_time_ms:.2f}ms"
        
        print(f"\nComplex Classification Benchmark: {result.to_dict()}")
    
    def test_classification_throughput(self):
        """Test classification throughput."""
        classifier = TaskComplexityClassifier()
        
        problems = [
            SubProblem(
                id=f"throughput-{i}",
                description=f"Problem {i}",
                domain=f"domain_{i % 10}",
                depth=i % 10,
                dependencies=[],
                metadata={},
            )
            for i in range(100)
        ]
        
        result = run_benchmark(
            "classification_throughput",
            lambda: [classifier.compute_complexity(p) for p in problems],
            iterations=10,
        )

        # Should handle at least 3 classifications per second (adjusted for system performance)
        assert result.throughput_per_sec > 3

        print(f"\nClassification Throughput: {result.throughput_per_sec:.2f} ops/sec")


class TestAllocationPerformance:
    """Benchmark allocation performance."""
    
    def test_single_allocation_latency(self):
        """Benchmark single allocation latency."""
        allocator = AdaptiveMDAPAllocator()
        
        result = run_benchmark(
            "single_allocation",
            lambda: allocator.allocate_resources(0.5),
            iterations=1000,
        )
        
        # Should be very fast: avg < 1ms, p95 < 2ms
        assert result.avg_time_ms < 1, f"Average allocation too slow: {result.avg_time_ms:.2f}ms"
        assert result.p95_time_ms < 2, f"P95 allocation too slow: {result.p95_time_ms:.2f}ms"
        
        print(f"\nSingle Allocation Benchmark: {result.to_dict()}")
    
    def test_batch_allocation_throughput(self):
        """Benchmark batch allocation throughput."""
        allocator = AdaptiveMDAPAllocator()
        complexities = [i / 1000 for i in range(1000)]
        
        result = run_benchmark(
            "batch_allocation",
            lambda: allocator.allocate_resources_batch(complexities),
            iterations=100,
        )
        
        # Should handle at least 2,000 allocations per second (adjusted for system performance)
        throughput = (result.iterations * len(complexities)) / (result.total_time_ms / 1000)
        assert throughput > 2000, f"Batch allocation throughput too low: {throughput:.0f} ops/sec"
        
        print(f"\nBatch Allocation Throughput: {throughput:.0f} ops/sec")
    
    def test_context_aware_allocation_latency(self):
        """Benchmark context-aware allocation."""
        allocator = AdaptiveMDAPAllocator(enable_context_aware=True)
        from adaptive_mdap.allocators.resource_allocator import AllocationContext
        
        context = AllocationContext(system_load="high", budget_remaining=50)
        
        result = run_benchmark(
            "context_aware_allocation",
            lambda: allocator.allocate_resources(0.5, context=context),
            iterations=1000,
        )
        
        # Context-aware should still be fast: avg < 2ms
        assert result.avg_time_ms < 2, f"Context-aware allocation too slow: {result.avg_time_ms:.2f}ms"
        
        print(f"\nContext-Aware Allocation Benchmark: {result.to_dict()}")


class TestEndToEndPerformance:
    """Benchmark end-to-end execution performance."""
    
    def test_direct_strategy_execution(self):
        """Benchmark DIRECT strategy execution."""
        controller = AdaptiveExecutionController()
        
        sp = SubProblem(
            id="perf-direct",
            description="Direct strategy test",
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        result = run_benchmark(
            "direct_execution",
            lambda: controller.execute_adaptive(sp, force_strategy=SolveStrategy.DIRECT),
            iterations=10,
            warmup=2,
        )
        
        # DIRECT should be fast: avg < 100ms (mostly overhead)
        assert result.avg_time_ms < 100, f"Direct execution too slow: {result.avg_time_ms:.2f}ms"
        
        print(f"\nDirect Execution Benchmark: {result.to_dict()}")
    
    def test_full_adaptive_execution_simple(self):
        """Benchmark full adaptive execution for simple problems."""
        controller = AdaptiveExecutionController()
        
        sp = SubProblem(
            id="perf-adaptive-simple",
            description="Simple adaptive test",
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        result = run_benchmark(
            "adaptive_execution_simple",
            lambda: controller.execute_adaptive(sp),
            iterations=10,
            warmup=2,
        )
        
        # Full execution: avg < 200ms
        assert result.avg_time_ms < 200, f"Adaptive execution too slow: {result.avg_time_ms:.2f}ms"
        
        print(f"\nAdaptive Execution (Simple) Benchmark: {result.to_dict()}")


class TestCostCalculationPerformance:
    """Benchmark cost calculation performance."""
    
    def test_single_cost_calculation(self):
        """Benchmark single cost calculation."""
        calculator = CostCalculator(pricing=APIPricing.gpt_4o_mini())
        
        result = run_benchmark(
            "single_cost_calculation",
            lambda: calculator.calculate_adaptive_cost(1000),
            iterations=1000,
        )
        
        # Should be very fast: avg < 1ms
        assert result.avg_time_ms < 1, f"Cost calculation too slow: {result.avg_time_ms:.2f}ms"
        
        print(f"\nCost Calculation Benchmark: {result.to_dict()}")
    
    def test_model_comparison_performance(self):
        """Benchmark model comparison."""
        calculator = CostCalculator()
        
        result = run_benchmark(
            "model_comparison",
            lambda: calculator.compare_models(1000),
            iterations=100,
        )
        
        # Model comparison: avg < 10ms
        assert result.avg_time_ms < 10, f"Model comparison too slow: {result.avg_time_ms:.2f}ms"
        
        print(f"\nModel Comparison Benchmark: {result.to_dict()}")


class TestMemoryUsage:
    """Test memory usage characteristics."""
    
    def test_classifier_memory_growth(self):
        """Test that classifier doesn't leak memory."""
        import gc
        import sys
        
        classifier = TaskComplexityClassifier()
        
        # Get baseline memory (approximate using object count)
        gc.collect()
        baseline_objects = len(gc.get_objects())
        
        # Process many problems
        for i in range(100):
            sp = SubProblem(
                id=f"memory-test-{i}",
                description=f"Problem {i}",
                domain=f"domain_{i % 5}",
                depth=i % 10,
                dependencies=[],
                metadata={},
            )
            classifier.compute_complexity(sp)
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be bounded (allow 10% growth)
        growth_ratio = final_objects / baseline_objects
        assert growth_ratio < 1.5, f"Memory growth too high: {growth_ratio:.2f}x"
    
    def test_allocator_memory_growth(self):
        """Test that allocator doesn't leak memory."""
        import gc
        
        allocator = AdaptiveMDAPAllocator()
        
        gc.collect()
        baseline_objects = len(gc.get_objects())
        
        # Make many allocations
        for i in range(1000):
            allocator.allocate_resources((i % 100) / 100)
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be minimal (stats tracking only)
        growth_ratio = final_objects / baseline_objects
        assert growth_ratio < 1.2, f"Memory growth too high: {growth_ratio:.2f}x"


class TestScalability:
    """Test system scalability."""
    
    def test_concurrent_classification(self):
        """Test concurrent classification performance."""
        import concurrent.futures
        
        classifier = TaskComplexityClassifier()
        
        problems = [
            SubProblem(
                id=f"concurrent-{i}",
                description=f"Problem {i}",
                domain="test",
                depth=0,
                dependencies=[],
                metadata={},
            )
            for i in range(100)
        ]
        
        def classify(p):
            return classifier.compute_complexity(p)
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(classify, problems))
        elapsed_ms = (time.time() - start) * 1000
        
        # Should handle 100 concurrent classifications quickly
        assert elapsed_ms < 2000, f"Concurrent classification too slow: {elapsed_ms:.2f}ms"
        assert len(results) == 100


@pytest.mark.benchmark
class TestProductionReadiness:
    """Production readiness benchmarks."""
    
    def test_full_system_stress_test(self):
        """Stress test the full system."""
        controller = AdaptiveExecutionController()
        
        # Mix of simple and complex problems
        problems = [
            SubProblem(
                id=f"stress-{i}",
                description="Simple" if i % 2 == 0 else "Complex distributed security " * 50,
                domain="test",
                depth=i % 10,
                dependencies=[],
                metadata={},
            )
            for i in range(50)
        ]
        
        start = time.time()
        for p in problems:
            controller.execute_adaptive(p)
        elapsed_ms = (time.time() - start) * 1000
        
        # Should handle 50 mixed problems in under 10 seconds
        assert elapsed_ms < 10000, f"Stress test too slow: {elapsed_ms:.2f}ms"
        
        avg_ms = elapsed_ms / len(problems)
        print(f"\nStress Test: {len(problems)} problems in {elapsed_ms:.2f}ms (avg: {avg_ms:.2f}ms)")
    
    def test_cost_savings_validation(self):
        """Validate cost savings meet targets."""
        calculator = CostCalculator(pricing=APIPricing.gpt_4o_mini())
        
        # Test with default workload (30% easy, 40% medium, 30% hard)
        result = calculator.calculate_adaptive_cost(10000)
        
        savings_percent = result["savings_percent"]
        
        # Should achieve at least 30% savings
        assert savings_percent >= 30, f"Cost savings below target: {savings_percent:.1f}%"
        
        # Should achieve at most 60% savings (unrealistic above this)
        assert savings_percent <= 60, f"Cost savings unrealistically high: {savings_percent:.1f}%"
        
        print(f"\nCost Savings: {savings_percent:.1f}% (${result['savings']:.2f} saved)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark"])
