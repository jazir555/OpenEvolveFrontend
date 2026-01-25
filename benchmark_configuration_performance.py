"""
Configuration Performance Benchmarking Suite

This module provides comprehensive performance benchmarking for configuration access patterns
to identify bottlenecks and validate optimization improvements.

Usage:
    python benchmark_configuration_performance.py
"""
import timeit
import statistics
import sys
from typing import Dict, Any, List, Callable
import json

# Import UnifiedConfiguration
try:
    from unified_configuration import (
        UnifiedConfiguration,
        create_unified_config,
        create_standard_evolution_config,
        create_adversarial_testing_config
    )
    from parameter_manager import ParameterManager
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Configuration modules not available: {e}")
    CONFIG_AVAILABLE = False
    # Create dummy class for testing
    class UnifiedConfiguration:
        def __init__(self, params):
            self._parameters = params
        @property
        def max_iterations(self):
            return self._parameters.get('max_iterations', 10)
        def get(self, name, default=None):
            return self._parameters.get(name, default)
        def to_dict(self):
            return self._parameters.copy()


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""

    def __init__(self):
        self.results: Dict[str, List[float]] = {}

    def benchmark(self, name: str, func: Callable, iterations: int = 1000) -> Dict[str, float]:
        """
        Run a benchmark with statistical analysis.

        Args:
            name: Benchmark name
            func: Function to benchmark
            iterations: Number of iterations

        Returns:
            Dictionary with mean, median, min, max times
        """
        print(f"\n{'='*60}")
        print(f"Benchmark: {name}")
        print(f"{'='*60}")

        # Run benchmark
        times = timeit.repeat(
            stmt=func,
            repeat=10,
            number=iterations
        )

        # Calculate statistics
        stats = {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
            'iterations': iterations
        }

        self.results[name] = times

        # Print results
        print(f"Iterations: {iterations:,}")
        print(f"Mean:      {stats['mean']*1000:.4f} ms")
        print(f"Median:    {stats['median']*1000:.4f} ms")
        print(f"Min:       {stats['min']*1000:.4f} ms")
        print(f"Max:       {stats['max']*1000:.4f} ms")
        print(f"Std Dev:   {stats['std_dev']*1000:.4f} ms")
        print(f"Per iter:  {stats['mean']/iterations*1000000:.4f} us")

        return stats

    def compare(self, baseline: str, optimized: str) -> float:
        """
        Compare two benchmarks and calculate improvement.

        Args:
            baseline: Baseline benchmark name
            optimized: Optimized benchmark name

        Returns:
            Improvement percentage (positive = faster)
        """
        if baseline not in self.results or optimized not in self.results:
            print(f"ERROR: Missing benchmarks for comparison")
            return 0.0

        baseline_mean = statistics.mean(self.results[baseline])
        optimized_mean = statistics.mean(self.results[optimized])

        improvement = ((baseline_mean - optimized_mean) / baseline_mean) * 100

        print(f"\n{'='*60}")
        print(f"Comparison: {baseline} vs {optimized}")
        print(f"{'='*60}")
        print(f"Baseline:  {baseline_mean*1000:.4f} ms")
        print(f"Optimized: {optimized_mean*1000:.4f} ms")
        print(f"Improvement: {improvement:+.2f}%")

        return improvement

    def summary(self):
        """Print summary of all benchmarks"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*60}")

        for name, times in self.results.items():
            mean_time = statistics.mean(times) * 1000
            print(f"{name:50s}: {mean_time:.4f} ms")


# =============================================================================
# BENCHMARK TEST CASES
# =============================================================================

def test_property_access():
    """Benchmark: Property access (current implementation)"""
    config = UnifiedConfiguration({'max_iterations': 10, 'temperature': 0.7}, validate=False)
    for _ in range(100):
        _ = config.max_iterations
        _ = config.temperature
        _ = config.population_size


def test_property_access_cached():
    """Benchmark: Property access with external caching"""
    config = UnifiedConfiguration({'max_iterations': 10, 'temperature': 0.7}, validate=False)
    # Cache properties outside loop
    max_iter = config.max_iterations
    temp = config.temperature
    pop_size = config.population_size
    for _ in range(100):
        _ = max_iter
        _ = temp
        _ = pop_size


def test_get_method():
    """Benchmark: get() method access"""
    config = UnifiedConfiguration({'max_iterations': 10, 'temperature': 0.7}, validate=False)
    for _ in range(100):
        _ = config.get('max_iterations')
        _ = config.get('temperature')
        _ = config.get('population_size')


def test_dict_access():
    """Benchmark: Direct dictionary access"""
    config = UnifiedConfiguration({'max_iterations': 10, 'temperature': 0.7}, validate=False)
    params = config.to_dict()
    for _ in range(100):
        _ = params['max_iterations']
        _ = params['temperature']
        _ = params['population_size']


def test_dict_item_access():
    """Benchmark: __getitem__ access"""
    config = UnifiedConfiguration({'max_iterations': 10, 'temperature': 0.7}, validate=False)
    for _ in range(100):
        _ = config['max_iterations']
        _ = config['temperature']
        _ = config['population_size']


def test_config_creation():
    """Benchmark: Configuration creation overhead"""
    for _ in range(10):
        config = UnifiedConfiguration({'max_iterations': 10, 'temperature': 0.7}, validate=False)


def test_config_creation_with_defaults():
    """Benchmark: Configuration creation with defaults applied"""
    for _ in range(10):
        config = create_standard_evolution_config()


def test_config_merge():
    """Benchmark: Configuration merging"""
    config1 = create_standard_evolution_config()
    config2 = create_adversarial_testing_config()
    for _ in range(10):
        merged = config1.merge(config2.to_dict())


def test_loop_with_config_access():
    """Benchmark: Config access in tight loop (common pattern)"""
    config = UnifiedConfiguration({
        'max_iterations': 100,
        'temperature': 0.7,
        'population_size': 50
    }, validate=False)

    results = []
    for i in range(config.max_iterations):
        # Simulating computation with config access
        temp = config.temperature
        pop = config.population_size
        results.append(i * temp)


def test_loop_optimized():
    """Benchmark: Optimized loop with cached config values"""
    config = UnifiedConfiguration({
        'max_iterations': 100,
        'temperature': 0.7,
        'population_size': 50
    }, validate=False)

    # Cache config values outside loop
    max_iter = config.max_iterations
    temp = config.temperature
    pop = config.population_size

    results = []
    for i in range(max_iter):
        # Use cached values
        results.append(i * temp)


def test_multiple_instance_creation():
    """Benchmark: Multiple config instances (potential memory leak)"""
    for _ in range(50):
        config = UnifiedConfiguration({'max_iterations': 10}, validate=False)
        _ = config.max_iterations


def test_single_instance_reuse():
    """Benchmark: Single config instance reuse (optimized pattern)"""
    config = UnifiedConfiguration({'max_iterations': 10}, validate=False)
    for _ in range(50):
        _ = config.max_iterations


def test_to_dict_conversion():
    """Benchmark: to_dict() conversion"""
    config = create_standard_evolution_config()
    for _ in range(50):
        params = config.to_dict()


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

def run_all_benchmarks():
    """Run all benchmarks and generate report"""
    if not CONFIG_AVAILABLE:
        print("Configuration modules not available. Skipping benchmarks.")
        return

    print("=" * 60)
    print("  CONFIGURATION PERFORMANCE BENCHMARK")
    print("=" * 60)

    benchmark = PerformanceBenchmark()

    # 1. Property Access Patterns
    print("\n" + "="*60)
    print("PROPERTY ACCESS PATTERNS")
    print("="*60)

    benchmark.benchmark("Property Access", test_property_access, iterations=1000)
    benchmark.benchmark("Property Access (Cached)", test_property_access_cached, iterations=1000)
    benchmark.benchmark("get() Method", test_get_method, iterations=1000)
    benchmark.benchmark("Direct Dict Access", test_dict_access, iterations=1000)
    benchmark.benchmark("__getitem__ Access", test_dict_item_access, iterations=1000)

    # 2. Configuration Creation
    print("\n" + "="*60)
    print("CONFIGURATION CREATION")
    print("="*60)

    benchmark.benchmark("Config Creation", test_config_creation, iterations=100)
    benchmark.benchmark("Config with Defaults", test_config_creation_with_defaults, iterations=100)

    # 3. Configuration Operations
    print("\n" + "="*60)
    print("CONFIGURATION OPERATIONS")
    print("="*60)

    benchmark.benchmark("Config Merge", test_config_merge, iterations=100)
    benchmark.benchmark("to_dict() Conversion", test_to_dict_conversion, iterations=100)

    # 4. Loop Performance (Real-world patterns)
    print("\n" + "="*60)
    print("LOOP PERFORMANCE (Real-world patterns)")
    print("="*60)

    benchmark.benchmark("Loop with Config Access", test_loop_with_config_access, iterations=100)
    benchmark.benchmark("Loop Optimized (Cached)", test_loop_optimized, iterations=100)

    # 5. Memory Efficiency
    print("\n" + "="*60)
    print("MEMORY EFFICIENCY")
    print("="*60)

    benchmark.benchmark("Multiple Instances", test_multiple_instance_creation, iterations=100)
    benchmark.benchmark("Single Instance Reuse", test_single_instance_reuse, iterations=100)

    # 6. Comparisons
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISONS")
    print("="*60)

    benchmark.compare("Property Access (Cached)", "Property Access")
    benchmark.compare("Direct Dict Access", "Property Access")
    benchmark.compare("Loop Optimized (Cached)", "Loop with Config Access")
    benchmark.compare("Single Instance Reuse", "Multiple Instances")

    # 7. Summary
    benchmark.summary()

    # 8. Recommendations
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)

    print_recommendations(benchmark)


def print_recommendations(benchmark: PerformanceBenchmark):
    """Print optimization recommendations based on benchmark results"""
    recommendations = [
        {
            'pattern': 'Loop Config Access',
            'issue': 'Accessing config properties inside loops',
            'solution': 'Cache frequently accessed parameters outside loops',
            'example': '''
# BEFORE (slow)
for i in range(config.max_iterations):
    temp = config.temperature  # Property access every iteration

# AFTER (fast)
temp = config.temperature  # Cache once
for i in range(config.max_iterations):
    # Use cached temp
'''
        },
        {
            'pattern': 'Multiple Instances',
            'issue': 'Creating new config instances repeatedly',
            'solution': 'Reuse single config instance or use config.to_dict()',
            'example': '''
# BEFORE (slow)
def process_items(items):
    for item in items:
        config = UnifiedConfiguration()  # New instance every iteration!
        process(item, config)

# AFTER (fast)
def process_items(items):
    config = UnifiedConfiguration()  # One instance
    for item in items:
        process(item, config)
'''
        },
        {
            'pattern': 'Dict Conversion',
            'issue': 'Frequent to_dict() conversions',
            'solution': 'Convert once and reuse the dictionary',
            'example': '''
# BEFORE (slow)
for item in items:
    params = config.to_dict()  # Converts every iteration
    process(item, params)

# AFTER (fast)
params = config.to_dict()  # Convert once
for item in items:
    process(item, params)
'''
        }
    ]

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['pattern']}")
        print(f"   Issue: {rec['issue']}")
        print(f"   Solution: {rec['solution']}")
        print(f"   Example:{rec['example']}")


if __name__ == "__main__":
    run_all_benchmarks()
