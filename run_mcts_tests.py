#!/usr/bin/env python3
"""
MCTS Test Runner

Comprehensive test runner for LeanAide MCTS implementation with:
- Category-based test execution
- Coverage reporting
- Performance benchmarking
- Result visualization

Usage:
    python run_mcts_tests.py                    # Run all tests
    python run_mcts_tests.py --category unit    # Run unit tests only
    python run_mcts_tests.py --benchmark        # Run performance benchmarks
    python run_mcts_tests.py --coverage         # Generate coverage report
    python run_mcts_tests.py --verbose          # Verbose output
"""

import sys
import argparse
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import json


class MCTSTestRunner:
    """Test runner for MCTS tests"""

    def __init__(self, args):
        self.args = args
        self.results = {
            "unit": [],
            "integration": [],
            "performance": [],
            "edge_cases": []
        }

    def run_all(self):
        """Run all test categories"""
        print("=" * 80)
        print("LeanAide MCTS Test Suite")
        print("=" * 80)

        categories = ["unit", "integration", "performance", "edge_cases"]

        for category in categories:
            if self.args.category and self.args.category != category:
                continue

            self.run_category(category)

        self.print_summary()

    def run_category(self, category: str):
        """Run tests in a specific category"""
        print(f"\n{'=' * 80}")
        print(f"Running {category.upper()} tests")
        print("=" * 80)

        start_time = time.time()

        try:
            if category == "unit":
                result = self.run_unit_tests()
            elif category == "integration":
                result = self.run_integration_tests()
            elif category == "performance":
                result = self.run_performance_tests()
            elif category == "edge_cases":
                result = self.run_edge_case_tests()
            else:
                print(f"Unknown category: {category}")
                return

            elapsed = time.time() - start_time
            self.results[category] = result

            print(f"\n[OK] {category.upper()} tests completed in {elapsed:.2f}s")

        except Exception as e:
            print(f"\n[FAIL] {category.upper()} tests failed: {e}")
            self.results[category] = {"status": "failed", "error": str(e)}

    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests"""
        tests = [
            "TestMCTSNode",
            "TestMCTSSelection",
            "TestMCTSExpansion",
            "TestMCTSSimulation",
            "TestMCTSBackpropagation"
        ]

        return self._run_pytest_tests(tests, "unit")

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        tests = [
            "TestMCTSIntegration",
            "TestLeanProofMCTS",
            "TestMCTSWithTheorems"
        ]

        return self._run_pytest_tests(tests, "integration")

    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        print("\nRunning performance benchmarks...")

        benchmarks = [
            ("test_convergence_rate", 100),
            ("test_scalability_with_iterations", 500),
            ("test_tree_size_growth", 200)
        ]

        results = {}
        for test_name, timeout in benchmarks:
            print(f"  Benchmark: {test_name}...")
            start = time.time()

            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", "test_leanaide_mcts.py",
                     f"-k", test_name,
                     "-v", "--tb=short"],
                    capture_output=True,
                    timeout=timeout,
                    text=True
                )

                elapsed = time.time() - start
                results[test_name] = {
                    "status": "passed" if result.returncode == 0 else "failed",
                    "time": elapsed,
                    "output": result.stdout[-500:]  # Last 500 chars
                }

                print(f"    Status: {results[test_name]['status']}")
                print(f"    Time: {elapsed:.2f}s")

            except subprocess.TimeoutExpired:
                results[test_name] = {
                    "status": "timeout",
                    "time": timeout
                }
                print(f"    Status: timeout after {timeout}s")

        return results

    def run_edge_case_tests(self) -> Dict[str, Any]:
        """Run edge case tests"""
        tests = [
            "test_empty_tactic_list",
            "test_single_tactic_available",
            "test_immediate_proof_found",
            "test_no_proof_possible",
            "test_timeout_handling"
        ]

        return self._run_pytest_tests(tests, "edge_cases")

    def _run_pytest_tests(self, test_names: List[str], category: str) -> Dict[str, Any]:
        """Run pytest tests and return results"""
        pytest_args = [
            "python", "-m", "pytest",
            "test_leanaide_mcts.py",
            "-v",
            "--tb=short"
        ]

        if not self.args.verbose:
            pytest_args.append("--quiet")

        # Add test filters
        for test_name in test_names:
            pytest_args.extend(["-k", test_name])

        print(f"\nRunning: {' '.join(pytest_args)}")

        start_time = time.time()
        result = subprocess.run(
            pytest_args,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time

        # Parse output
        passed = result.stdout.count("PASSED")
        failed = result.stdout.count("FAILED")
        errors = result.stdout.count("ERROR")

        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "time": elapsed,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "output": result.stdout if self.args.verbose else ""
        }

    def run_coverage(self):
        """Generate coverage report"""
        print("\n" + "=" * 80)
        print("Generating Coverage Report")
        print("=" * 80)

        try:
            result = subprocess.run(
                ["python", "-m", "pytest",
                 "test_leanaide_mcts.py",
                 "--cov=leanaide_mcts",
                 "--cov-report=html",
                 "--cov-report=term"],
                capture_output=True,
                text=True
            )

            print(result.stdout)

            if result.returncode == 0:
                print("\n[OK] Coverage report generated in htmlcov/index.html")
            else:
                print("\n[FAIL] Coverage generation failed")

        except FileNotFoundError:
            print("\n[FAIL] pytest-cov not installed. Install with: pip install pytest-cov")

    def run_benchmark(self):
        """Run performance benchmarks"""
        print("\n" + "=" * 80)
        print("Performance Benchmarks")
        print("=" * 80)

        # Import and run benchmarks
        try:
            from test_leanaide_mcts import TestMCTSPerformance

            benchmarks = [
                ("Convergence Rate", TestMCTSPerformance().test_convergence_rate),
                ("Scalability", TestMCTSPerformance().test_scalability_with_iterations),
                ("Tree Growth", TestMCTSPerformance().test_tree_size_growth)
            ]

            results = {}
            for name, test_func in benchmarks:
                print(f"\nBenchmark: {name}")
                start = time.time()

                try:
                    test_func()
                    elapsed = time.time() - start
                    results[name] = {"time": elapsed, "status": "passed"}
                    print(f"  [OK] Passed in {elapsed:.2f}s")

                except Exception as e:
                    elapsed = time.time() - start
                    results[name] = {"time": elapsed, "status": "failed", "error": str(e)}
                    print(f"  [FAIL] Failed: {e}")

            # Print summary
            print("\n" + "-" * 80)
            print("Benchmark Summary:")
            for name, result in results.items():
                status = result["status"]
                time_str = f"{result['time']:.2f}s"
                print(f"  {name:30s} {status:10s} {time_str}")

        except ImportError as e:
            print(f"[FAIL] Failed to import test module: {e}")

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)

        total_passed = 0
        total_failed = 0
        total_time = 0.0

        for category, results in self.results.items():
            if not results:
                continue

            if isinstance(results, dict):
                if "passed" in results:
                    passed = results.get("passed", 0)
                    failed = results.get("failed", 0)
                    time_taken = results.get("time", 0)

                    total_passed += passed
                    total_failed += failed
                    total_time += time_taken

                    status = "[OK] PASSED" if results["status"] == "passed" else "[FAIL] FAILED"
                    print(f"\n{category.upper():20s}: {status}")
                    print(f"  Passed: {passed}, Failed: {failed}, Time: {time_taken:.2f}s")

        print("\n" + "-" * 80)
        print(f"TOTAL: {total_passed} passed, {total_failed} failed, {total_time:.2f}s")

        # Save results to file
        self.save_results()

    def save_results(self):
        """Save test results to JSON file"""
        output_file = Path("mcts_test_results.json")

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LeanAide MCTS Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mcts_tests.py                    # Run all tests
  python run_mcts_tests.py --category unit    # Run unit tests only
  python run_mcts_tests.py --benchmark        # Run performance benchmarks
  python run_mcts_tests.py --coverage         # Generate coverage report
  python run_mcts_tests.py --verbose          # Verbose output
        """
    )

    parser.add_argument(
        "--category",
        choices=["unit", "integration", "performance", "edge_cases"],
        help="Run specific test category"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Check if test file exists
    test_file = Path("test_leanaide_mcts.py")
    if not test_file.exists():
        print(f"[FAIL] Test file not found: {test_file}")
        sys.exit(1)

    # Run tests
    runner = MCTSTestRunner(args)

    if args.coverage:
        runner.run_coverage()
    elif args.benchmark:
        runner.run_benchmark()
    else:
        runner.run_all()


if __name__ == "__main__":
    main()
