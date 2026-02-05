#!/usr/bin/env python3
"""
Compare ParameterManager vs UnifiedConfiguration

This script benchmarks and compares the performance and functionality
of ParameterManager vs UnifiedConfiguration to demonstrate the benefits
of migration.
"""

import sys
import os
import time
import statistics
import gc
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # PHASE 1 MIGRATION: ParameterManager deprecated for comparison
    # This script compares old vs new, so we keep both for benchmarking
    from parameter_manager import ParameterManager  # DEPRECATED - kept for comparison only
    from unified_configuration import (
        UnifiedConfiguration,
        create_unified_config,
        ConfigurationValidationError
    )
    print("[INFO] This script compares deprecated ParameterManager vs new UnifiedConfiguration")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Stores benchmark results"""
    name: str
    avg_time: float
    min_time: float
    max_time: float
    median_time: float
    success: bool
    error: str = None
    details: Dict[str, Any] = None


class ParameterManagerComparator:
    """Compares ParameterManager vs UnifiedConfiguration performance"""

    def __init__(self):
        self.iterations = 100  # Number of iterations for benchmarks (reduced for faster testing)
        self.warmup_iterations = 10  # Warmup iterations (reduced for faster testing)

    def time_execution(self, func, *args, **kwargs) -> List[float]:
        """
        Time function execution multiple times

        Args:
            func: Function to time
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            List of execution times in seconds
        """
        times = []

        # Warmup
        for _ in range(self.warmup_iterations):
            try:
                func(*args, **kwargs)
            except (RuntimeError, OSError, ValueError) as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in compare_parameter_managers.py: {e}", exc_info=True)
                raise

        # Actual timing
        for _ in range(self.iterations):
            gc.collect()  # Garbage collect between runs
            start_time = time.perf_counter()
            try:
                func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except (RuntimeError, OSError, ValueError) as e:
                times.append(float('inf'))  # Mark failed runs

        return times

    def test_parameter_manager_creation(self) -> BenchmarkResult:
        """Test ParameterManager creation performance"""
        def create_pm():
            return ParameterManager()

        try:
            times = self.time_execution(create_pm)
            finite_times = [t for t in times if t != float('inf')]

            if not finite_times:
                return BenchmarkResult(
                    name="ParameterManager Creation",
                    avg_time=float('inf'),
                    min_time=float('inf'),
                    max_time=float('inf'),
                    median_time=float('inf'),
                    success=False,
                    error="All runs failed"
                )

            return BenchmarkResult(
                name="ParameterManager Creation",
                avg_time=statistics.mean(finite_times),
                min_time=min(finite_times),
                max_time=max(finite_times),
                median_time=statistics.median(finite_times),
                success=True,
                details={
                    'success_rate': len(finite_times) / len(times) * 100,
                    'total_runs': len(times),
                    'successful_runs': len(finite_times)
                }
            )

        except (RuntimeError, OSError, ValueError) as e:
            return BenchmarkResult(
                name="ParameterManager Creation",
                avg_time=float('inf'),
                min_time=float('inf'),
                max_time=float('inf'),
                median_time=float('inf'),
                success=False,
                error=str(e)
            )

    def test_unified_config_creation(self) -> BenchmarkResult:
        """Test UnifiedConfiguration creation performance"""
        def create_uc():
            return create_unified_config()

        try:
            times = self.time_execution(create_uc)
            finite_times = [t for t in times if t != float('inf')]

            if not finite_times:
                return BenchmarkResult(
                    name="UnifiedConfiguration Creation",
                    avg_time=float('inf'),
                    min_time=float('inf'),
                    max_time=float('inf'),
                    median_time=float('inf'),
                    success=False,
                    error="All runs failed"
                )

            return BenchmarkResult(
                name="UnifiedConfiguration Creation",
                avg_time=statistics.mean(finite_times),
                min_time=min(finite_times),
                max_time=max(finite_times),
                median_time=statistics.median(finite_times),
                success=True,
                details={
                    'success_rate': len(finite_times) / len(times) * 100,
                    'total_runs': len(times),
                    'successful_runs': len(finite_times)
                }
            )

        except (RuntimeError, OSError, ValueError) as e:
            return BenchmarkResult(
                name="UnifiedConfiguration Creation",
                avg_time=float('inf'),
                min_time=float('inf'),
                max_time=float('inf'),
                median_time=float('inf'),
                success=False,
                error=str(e)
            )

    def test_parameter_manager_get_defaults(self) -> BenchmarkResult:
        """Test ParameterManager get_defaults performance"""
        def get_defaults():
            pm = ParameterManager()
            return pm.get_defaults()

        try:
            times = self.time_execution(get_defaults)
            finite_times = [t for t in times if t != float('inf')]

            if not finite_times:
                return BenchmarkResult(
                    name="ParameterManager get_defaults",
                    avg_time=float('inf'),
                    min_time=float('inf'),
                    max_time=float('inf'),
                    median_time=float('inf'),
                    success=False,
                    error="All runs failed"
                )

            return BenchmarkResult(
                name="ParameterManager get_defaults",
                avg_time=statistics.mean(finite_times),
                min_time=min(finite_times),
                max_time=max(finite_times),
                median_time=statistics.median(finite_times),
                success=True,
                details={
                    'success_rate': len(finite_times) / len(times) * 100,
                    'defaults_count': len(finite_times[0]) if finite_times else 0
                }
            )

        except (RuntimeError, OSError, ValueError) as e:
            return BenchmarkResult(
                name="ParameterManager get_defaults",
                avg_time=float('inf'),
                min_time=float('inf'),
                max_time=float('inf'),
                median_time=float('inf'),
                success=False,
                error=str(e)
            )

    def test_unified_config_property_access(self) -> BenchmarkResult:
        """Test UnifiedConfiguration property access performance"""
        def get_properties():
            config = create_unified_config()
            _ = config.max_iterations
            _ = config.temperature
            _ = config.population_size
            _ = config.seed
            _ = config.api_key
            return config

        try:
            times = self.time_execution(get_properties)
            finite_times = [t for t in times if t != float('inf')]

            if not finite_times:
                return BenchmarkResult(
                    name="UnifiedConfiguration Property Access",
                    avg_time=float('inf'),
                    min_time=float('inf'),
                    max_time=float('inf'),
                    median_time=float('inf'),
                    success=False,
                    error="All runs failed"
                )

            return BenchmarkResult(
                name="UnifiedConfiguration Property Access",
                avg_time=statistics.mean(finite_times),
                min_time=min(finite_times),
                max_time=max(finite_times),
                median_time=statistics.median(finite_times),
                success=True,
                details={
                    'success_rate': len(finite_times) / len(times) * 100,
                    'properties_accessed': 5
                }
            )

        except (RuntimeError, OSError, ValueError) as e:
            return BenchmarkResult(
                name="UnifiedConfiguration Property Access",
                avg_time=float('inf'),
                min_time=float('inf'),
                max_time=float('inf'),
                median_time=float('inf'),
                success=False,
                error=str(e)
            )

    def test_parameter_manager_validation(self) -> BenchmarkResult:
        """Test ParameterManager validation performance"""
        def validate_config():
            pm = ParameterManager()
            test_config = {'max_iterations': 20, 'temperature': 0.8}
            return pm.validate(test_config)

        try:
            times = self.time_execution(validate_config)
            finite_times = [t for t in times if t != float('inf')]

            if not finite_times:
                return BenchmarkResult(
                    name="ParameterManager Validation",
                    avg_time=float('inf'),
                    min_time=float('inf'),
                    max_time=float('inf'),
                    median_time=float('inf'),
                    success=False,
                    error="All runs failed"
                )

            return BenchmarkResult(
                name="ParameterManager Validation",
                avg_time=statistics.mean(finite_times),
                min_time=min(finite_times),
                max_time=max(finite_times),
                median_time=statistics.median(finite_times),
                success=True,
                details={
                    'success_rate': len(finite_times) / len(times) * 100,
                    'config_size': 2
                }
            )

        except (RuntimeError, OSError, ValueError) as e:
            return BenchmarkResult(
                name="ParameterManager Validation",
                avg_time=float('inf'),
                min_time=float('inf'),
                max_time=float('inf'),
                median_time=float('inf'),
                success=False,
                error=str(e)
            )

    def test_unified_config_validation(self) -> BenchmarkResult:
        """Test UnifiedConfiguration validation performance"""
        def validate_config():
            config = create_unified_config({'max_iterations': 20, 'temperature': 0.8})
            return config.validate()

        try:
            times = self.time_execution(validate_config)
            finite_times = [t for t in times if t != float('inf')]

            if not finite_times:
                return BenchmarkResult(
                    name="UnifiedConfiguration Validation",
                    avg_time=float('inf'),
                    min_time=float('inf'),
                    max_time=float('inf'),
                    median_time=float('inf'),
                    success=False,
                    error="All runs failed"
                )

            return BenchmarkResult(
                name="UnifiedConfiguration Validation",
                avg_time=statistics.mean(finite_times),
                min_time=min(finite_times),
                max_time=max(finite_times),
                median_time=statistics.median(finite_times),
                success=True,
                details={
                    'success_rate': len(finite_times) / len(times) * 100,
                    'config_size': 2
                }
            )

        except (RuntimeError, OSError, ValueError) as e:
            return BenchmarkResult(
                name="UnifiedConfiguration Validation",
                avg_time=float('inf'),
                min_time=float('inf'),
                max_time=float('inf'),
                median_time=float('inf'),
                success=False,
                error=str(e)
            )

    def test_complex_parameter_access(self) -> BenchmarkResult:
        """Test complex parameter access scenarios"""
        def access_parameters():
            # Test different types of parameter access
            config = create_unified_config({
                'max_iterations': 50,
                'temperature': 0.9,
                'population_size': 100,
                'seed': 123,
                'api_key': 'test-key',
                'max_tokens': 2048
            })

            # Mix of access methods
            prop_access = config.max_iterations
            get_access = config.get('temperature')
            dict_access = config['population_size']
            category_access = config.get_category_params('core_evolution')

            return prop_access, get_access, dict_access, category_access

        try:
            times = self.time_execution(access_parameters)
            finite_times = [t for t in times if t != float('inf')]

            if not finite_times:
                return BenchmarkResult(
                    name="Complex Parameter Access",
                    avg_time=float('inf'),
                    min_time=float('inf'),
                    max_time=float('inf'),
                    median_time=float('inf'),
                    success=False,
                    error="All runs failed"
                )

            return BenchmarkResult(
                name="Complex Parameter Access",
                avg_time=statistics.mean(finite_times),
                min_time=min(finite_times),
                max_time=max(finite_times),
                median_time=statistics.median(finite_times),
                success=True,
                details={
                    'success_rate': len(finite_times) / len(times) * 100,
                    'access_methods': 4,
                    'parameters_accessed': 6
                }
            )

        except (RuntimeError, OSError, ValueError) as e:
            return BenchmarkResult(
                name="Complex Parameter Access",
                avg_time=float('inf'),
                min_time=float('inf'),
                max_time=float('inf'),
                median_time=float('inf'),
                success=False,
                error=str(e)
            )

    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks"""
        print("Running ParameterManager vs UnifiedConfiguration benchmarks...")
        print(f"Iterations per test: {self.iterations:,}")
        print(f"Warmup iterations: {self.warmup:,}")
        print()

        benchmarks = [
            self.test_parameter_manager_creation,
            self.test_unified_config_creation,
            self.test_parameter_manager_get_defaults,
            self.test_unified_config_property_access,
            self.test_parameter_manager_validation,
            self.test_unified_config_validation,
            self.test_complex_parameter_access,
        ]

        results = []

        for benchmark in benchmarks:
            print(f"Running {benchmark.__name__}...")
            result = benchmark()
            results.append(result)
            print(f"  Result: {'[OK]' if result.success else '[FAIL]'}")

        return results

    def format_benchmark_report(self, results: List[BenchmarkResult]) -> str:
        """Format benchmark results into a readable report"""
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("PARAMETER MANAGER vs UNIFIEDCONFIGURATION PERFORMANCE COMPARISON")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary table
        report_lines.append("BENCHMARK RESULTS:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Benchmark':<45} | {'Avg (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Status'}")
        report_lines.append("-" * 80)

        for result in results:
            avg_ms = result.avg_time * 1000
            min_ms = result.min_time * 1000
            max_ms = result.max_time * 1000

            status = "[OK] PASS" if result.success else "[FAIL] FAIL"

            # Truncate long names
            name = result.name[:44]

            report_lines.append(
                f"{name:<45} | {avg_ms:>9.3f} | {min_ms:>9.3f} | {max_ms:>9.3f} | {status}"
            )

        report_lines.append("")

        # Detailed analysis
        report_lines.append("PERFORMANCE ANALYSIS:")
        report_lines.append("-" * 80)

        # Compare creation
        pm_creation = next(r for r in results if "ParameterManager Creation" in r.name)
        uc_creation = next(r for r in results if "UnifiedConfiguration Creation" in r.name)

        if pm_creation.success and uc_creation.success:
            creation_ratio = pm_creation.avg_time / uc_creation.avg_time
            report_lines.append(f"Creation Performance: UnifiedConfiguration {creation_ratio:.2f}x "
                              f"{'faster' if creation_ratio > 1 else 'slower'} than ParameterManager")

        # Compare property access vs defaults
        pm_defaults = next(r for r in results if "ParameterManager get_defaults" in r.name)
        uc_props = next(r for r in results if "UnifiedConfiguration Property Access" in r.name)

        if pm_defaults.success and uc_props.success:
            props_ratio = pm_defaults.avg_time / uc_props.avg_time
            report_lines.append(f"Access Performance: Property access {props_ratio:.2f}x "
                              f"{'faster' if props_ratio > 1 else 'slower'} than get_defaults()")

        # Compare validation
        pm_validation = next(r for r in results if "ParameterManager Validation" in r.name)
        uc_validation = next(r for r in results if "UnifiedConfiguration Validation" in r.name)

        if pm_validation.success and uc_validation.success:
            validation_ratio = pm_validation.avg_time / uc_validation.avg_time
            report_lines.append(f"Validation Performance: UnifiedConfiguration {validation_ratio:.2f}x "
                              f"{'faster' if validation_ratio > 1 else 'slower'} than ParameterManager")

        report_lines.append("")

        # Error analysis
        failed_benchmarks = [r for r in results if not r.success]
        if failed_benchmarks:
            report_lines.append("FAILED BENCHMARKS:")
            report_lines.append("-" * 40)
            for result in failed_benchmarks:
                report_lines.append(f"[FAIL] {result.name}: {result.error}")
            report_lines.append("")

        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 40)

        successful_pm = sum(1 for r in results if "ParameterManager" in r.name and r.success)
        successful_uc = sum(1 for r in results if "UnifiedConfiguration" in r.name and r.success)

        if successful_uc > successful_pm:
            report_lines.append("[OK] UnifiedConfiguration has more successful benchmarks")

        # Feature comparison
        feature_lines = [
            "[OK] UnifiedConfiguration provides:",
            "  - Type-safe property access",
            "  - Flexible get/set methods",
            "  - Merging capabilities",
            "  - File I/O operations",
            "  - Preset configuration functions",
            "  - Better error handling",
            "  - Unified interface across modules"
        ]

        if successful_uc >= successful_pm:
            feature_lines.extend([
                "",
                "[OK] Migrating to UnifiedConfiguration provides:",
                "  - More features with similar performance",
                "  - Better maintainability",
                "  - Reduced code duplication",
                "  - Consistent configuration interface"
            ])

        report_lines.extend(feature_lines)

        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def save_report(self, results: List[BenchmarkResult], output_path: str = None) -> str:
        """Save benchmark report to file"""
        if output_path is None:
            output_path = "parameter_manager_comparison_report.txt"

        report = self.format_benchmark_report(results)

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"\nBenchmark report saved to: {output_path}")
        return output_path


def main():
    """Main comparison function"""
    try:
        comparator = ParameterManagerComparator()
        results = comparator.run_benchmarks()
        report_path = comparator.save_report(results)

        # Print summary
        print(f"\n{comparator.format_benchmark_report(results)}")

        # Return exit code
        if all(r.success for r in results):
            return 0
        else:
            print(f"\n⚠ {len([r for r in results if not r.success])} benchmark(s) failed")
            return 1

    except (RuntimeError, OSError, ValueError) as e:
        print(f"\n[FAIL] Benchmarking failed: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())