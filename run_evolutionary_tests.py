#!/usr/bin/env python3
"""
Test Runner for Evolutionary LeanAide Test Suite

This script provides convenient ways to run the evolutionary LeanAide tests:
- Run all tests
- Run specific test categories
- Generate coverage reports
- Performance benchmarking
- Selective test execution

Author: OpenEvolve
Created: 2025-12-30
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class TestRunner:
    """Test runner for evolutionary LeanAide tests."""

    def __init__(self, test_file: str = "test_leanaide_evolutionary.py"):
        self.test_file = test_file
        self.results_dir = Path("test_results_evolutionary")
        self.results_dir.mkdir(exist_ok=True)

    def run_tests(
        self,
        markers: Optional[str] = None,
        verbose: bool = True,
        capture_output: bool = False,
        parallel: bool = False,
        coverage: bool = False,
        extra_args: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Run tests with specified configuration.

        Args:
            markers: Pytest markers to filter tests (e.g., "evolution", "unit")
            verbose: Enable verbose output
            capture_output: Capture test output
            parallel: Run tests in parallel
            coverage: Generate coverage report
            extra_args: Additional pytest arguments

        Returns:
            Dictionary with test results
        """
        # Build pytest command
        cmd = ["python", "-m", "pytest", self.test_file]

        # Add markers
        if markers:
            cmd.extend(["-m", markers])

        # Add verbosity
        if verbose:
            cmd.append("-v")

        # Capture output
        if capture_output:
            cmd.append("--capture=no")  # Show prints

        # Parallel execution
        if parallel:
            cmd.extend(["-n", "auto"])  # Requires pytest-xdist

        # Coverage
        if coverage:
            cmd.extend([
                "--cov=.",
                "--cov-report=html",
                "--cov-report=term-missing"
            ])

        # Extra arguments
        if extra_args:
            cmd.extend(extra_args)

        # Run tests
        print(f"Running: {' '.join(cmd)}")
        print("=" * 80)

        start_time = time.time()

        result = subprocess.run(
            cmd,
            capture_output=False,
            text=False
        )

        elapsed_time = time.time() - start_time

        # Collect results
        test_results = {
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }

        return test_results

    def run_all_tests(
        self,
        verbose: bool = True,
        coverage: bool = False
    ) -> Dict[str, any]:
        """Run all tests."""
        print("\n" + "=" * 80)
        print("RUNNING ALL EVOLUTIONARY LEANAIDE TESTS")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers=None,
            verbose=verbose,
            coverage=coverage
        )

    def run_evolution_tests(self, verbose: bool = True) -> Dict[str, any]:
        """Run evolution tests only."""
        print("\n" + "=" * 80)
        print("RUNNING EVOLUTION TESTS")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers="evolution",
            verbose=verbose
        )

    def run_decomposition_tests(self, verbose: bool = True) -> Dict[str, any]:
        """Run decomposition tests only."""
        print("\n" + "=" * 80)
        print("RUNNING DECOMPOSITION TESTS")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers="decomposition",
            verbose=verbose
        )

    def run_adversarial_tests(self, verbose: bool = True) -> Dict[str, any]:
        """Run adversarial tests only."""
        print("\n" + "=" * 80)
        print("RUNNING ADVERSARIAL TESTS")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers="adversarial",
            verbose=verbose
        )

    def run_selfplay_tests(self, verbose: bool = True) -> Dict[str, any]:
        """Run self-play tests only."""
        print("\n" + "=" * 80)
        print("RUNNING SELF-PLAY TESTS")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers="selfplay",
            verbose=verbose
        )

    def run_strategy_tests(self, verbose: bool = True) -> Dict[str, any]:
        """Run strategy tests only."""
        print("\n" + "=" * 80)
        print("RUNNING STRATEGY TESTS")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers="strategy",
            verbose=verbose
        )

    def run_workflow_tests(self, verbose: bool = True) -> Dict[str, any]:
        """Run workflow integration tests only."""
        print("\n" + "=" * 80)
        print("RUNNING WORKFLOW INTEGRATION TESTS")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers="workflow",
            verbose=verbose
        )

    def run_unit_tests(self, verbose: bool = True) -> Dict[str, any]:
        """Run unit tests only."""
        print("\n" + "=" * 80)
        print("RUNNING UNIT TESTS")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers="unit",
            verbose=verbose
        )

    def run_integration_tests(self, verbose: bool = True) -> Dict[str, any]:
        """Run integration tests only."""
        print("\n" + "=" * 80)
        print("RUNNING INTEGRATION TESTS")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers="integration",
            verbose=verbose
        )

    def run_fast_tests(self, verbose: bool = True) -> Dict[str, any]:
        """Run fast tests (excluding slow tests)."""
        print("\n" + "=" * 80)
        print("RUNNING FAST TESTS")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers="not slow",
            verbose=verbose
        )

    def run_server_tests(self, verbose: bool = True) -> Dict[str, any]:
        """Run tests requiring LeanAide server."""
        print("\n" + "=" * 80)
        print("RUNNING SERVER TESTS (requires LeanAide server)")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers="server",
            verbose=verbose
        )

    def run_mock_tests(self, verbose: bool = True) -> Dict[str, any]:
        """Run mock-based (offline) tests."""
        print("\n" + "=" * 80)
        print("RUNNING MOCK TESTS (offline)")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers="mock",
            verbose=verbose
        )

    def generate_coverage_report(self) -> Dict[str, any]:
        """Generate coverage report."""
        print("\n" + "=" * 80)
        print("GENERATING COVERAGE REPORT")
        print("=" * 80 + "\n")

        return self.run_tests(
            coverage=True,
            verbose=True
        )

    def run_performance_benchmark(self) -> Dict[str, any]:
        """Run performance benchmarks."""
        print("\n" + "=" * 80)
        print("RUNNING PERFORMANCE BENCHMARKS")
        print("=" * 80 + "\n")

        return self.run_tests(
            markers="slow",
            verbose=True
        )

    def run_specific_tests(
        self,
        test_names: List[str],
        verbose: bool = True
    ) -> Dict[str, any]:
        """Run specific tests by name."""
        print("\n" + "=" * 80)
        print(f"RUNNING SPECIFIC TESTS: {', '.join(test_names)}")
        print("=" * 80 + "\n")

        # Build pytest command with specific test names
        cmd = ["python", "-m", "pytest", "-v"]
        cmd.extend(test_names)

        print(f"Running: {' '.join(cmd)}")
        print("=" * 80)

        start_time = time.time()

        result = subprocess.run(
            cmd,
            capture_output=False,
            text=False
        )

        elapsed_time = time.time() - start_time

        return {
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }

    def save_results(self, results: Dict[str, any], filename: str):
        """Save test results to file."""
        output_path = self.results_dir / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    def print_summary(self, results: List[Dict[str, any]]):
        """Print summary of test results."""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80 + "\n")

        total_passed = 0
        total_failed = 0
        total_time = 0.0

        for i, result in enumerate(results, 1):
            print(f"Test Suite {i}:")
            print(f"  Command: {result['command']}")
            print(f"  Return Code: {result['return_code']}")
            print(f"  Elapsed Time: {result['elapsed_time']:.2f}s")
            print(f"  Timestamp: {result['timestamp']}")

            if result['return_code'] == 0:
                total_passed += 1
                print("  Status: PASSED")
            else:
                total_failed += 1
                print("  Status: FAILED")

            total_time += result['elapsed_time']
            print()

        print(f"Total Suites Passed: {total_passed}/{len(results)}")
        print(f"Total Time: {total_time:.2f}s")
        print("=" * 80)


def print_usage():
    """Print usage information."""
    print("""
Evolutionary LeanAide Test Runner
=================================

Usage:
    python run_evolutionary_tests.py [OPTIONS]

Options:
    --all                    Run all tests
    --evolution              Run evolution tests only
    --decomposition          Run decomposition tests only
    --adversarial            Run adversarial tests only
    --selfplay               Run self-play tests only
    --strategy               Run strategy tests only
    --workflow               Run workflow integration tests only
    --unit                   Run unit tests only
    --integration            Run integration tests only
    --fast                   Run fast tests (exclude slow)
    --server                 Run tests requiring server
    --mock                   Run mock tests (offline)
    --coverage               Generate coverage report
    --benchmark              Run performance benchmarks
    --quiet                  Less verbose output
    --parallel               Run tests in parallel
    --help                   Show this help message

Examples:
    # Run all tests
    python run_evolutionary_tests.py --all

    # Run only evolution tests
    python run_evolutionary_tests.py --evolution

    # Run unit tests with coverage
    python run_evolutionary_tests.py --unit --coverage

    # Run tests in parallel
    python run_evolutionary_tests.py --all --parallel

    # Run fast tests only
    python run_evolutionary_tests.py --fast

    # Generate coverage report
    python run_evolutionary_tests.py --coverage
    """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for evolutionary LeanAide tests",
        add_help=False
    )

    # Test category options
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--evolution", action="store_true", help="Run evolution tests")
    parser.add_argument("--decomposition", action="store_true", help="Run decomposition tests")
    parser.add_argument("--adversarial", action="store_true", help="Run adversarial tests")
    parser.add_argument("--selfplay", action="store_true", help="Run self-play tests")
    parser.add_argument("--strategy", action="store_true", help="Run strategy tests")
    parser.add_argument("--workflow", action="store_true", help="Run workflow tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--fast", action="store_true", help="Run fast tests")
    parser.add_argument("--server", action="store_true", help="Run server tests")
    parser.add_argument("--mock", action="store_true", help="Run mock tests")

    # Execution options
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--save", action="store_true", help="Save test results to file")
    parser.add_argument("--help", action="store_true", help="Show help message")

    args = parser.parse_args()

    # Show help
    if args.help or len(sys.argv) == 1:
        print_usage()
        return 0

    # Create test runner
    runner = TestRunner()

    # Determine which tests to run
    results = []

    if args.all:
        results.append(runner.run_all_tests(verbose=not args.quiet, coverage=args.coverage))
    elif args.evolution:
        results.append(runner.run_evolution_tests(verbose=not args.quiet))
    elif args.decomposition:
        results.append(runner.run_decomposition_tests(verbose=not args.quiet))
    elif args.adversarial:
        results.append(runner.run_adversarial_tests(verbose=not args.quiet))
    elif args.selfplay:
        results.append(runner.run_selfplay_tests(verbose=not args.quiet))
    elif args.strategy:
        results.append(runner.run_strategy_tests(verbose=not args.quiet))
    elif args.workflow:
        results.append(runner.run_workflow_tests(verbose=not args.quiet))
    elif args.unit:
        results.append(runner.run_unit_tests(verbose=not args.quiet))
    elif args.integration:
        results.append(runner.run_integration_tests(verbose=not args.quiet))
    elif args.fast:
        results.append(runner.run_fast_tests(verbose=not args.quiet))
    elif args.server:
        results.append(runner.run_server_tests(verbose=not args.quiet))
    elif args.mock:
        results.append(runner.run_mock_tests(verbose=not args.quiet))
    elif args.coverage:
        results.append(runner.generate_coverage_report())
    elif args.benchmark:
        results.append(runner.run_performance_benchmark())
    else:
        # Default: run all tests
        print("No specific test category selected. Running all tests...")
        results.append(runner.run_all_tests(verbose=not args.quiet, coverage=args.coverage))

    # Save results if requested
    if args.save and results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        runner.save_results(results[0], filename)

    # Print summary
    if len(results) > 0:
        runner.print_summary(results)

    # Return exit code based on results
    if any(r['return_code'] != 0 for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
