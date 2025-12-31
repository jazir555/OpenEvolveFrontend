#!/usr/bin/env python3
"""
MDAP-Evolution Test Runner

Runs comprehensive tests for MDAP-enhanced evolutionary computation.

Usage:
    python run_evolution_mdap_tests.py                    # Run all tests
    python run_evolution_mdap_tests.py --category unit    # Run unit tests only
    python run_evolution_mdap_tests.py --category integration --category performance
    python run_evolution_mdup_tests.py --benchmark         # Run performance benchmarks

Author: OpenEvolve Frontend Team
Date: 2025-12-30
"""

import sys
import time
import argparse
from typing import List, Optional, Dict
import subprocess

# Test categories
TEST_CATEGORIES = [
    "unit",           # Unit tests for components
    "integration",    # Integration tests for complete workflows
    "comparison",     # Comparison tests (pure vs MDAP-enhanced)
    "workflow",       # Workflow integration tests
    "performance",    # Performance benchmarks
    "edge"            # Edge case tests
]


def run_test_file(
    test_file: str,
    verbose: bool = True,
    extra_args: Optional[List[str]] = None
) -> int:
    """Run a single test file"""
    cmd = [sys.executable, test_file]
    if extra_args:
        cmd.extend(extra_args)

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def run_tests(
    categories: Optional[List[str]] = None,
    verbose: bool = True,
    benchmark: bool = False
) -> Dict[str, int]:
    """
    Run evolution-MDAP tests.

    Args:
        categories: List of test categories to run
        verbose: Whether to use verbose output
        benchmark: Whether to run performance benchmarks

    Returns:
        Dictionary mapping test file to return code
    """
    results = {}

    # Main test file
    test_file = "test_leanaide_evolution_mdap.py"

    if categories:
        # Run specific categories
        args = []
        for cat in categories:
            args.extend(["--category", cat])

        if not verbose:
            args.append("--quiet")

        print(f"\n{'=' * 70}")
        print(f"Running test categories: {', '.join(categories)}")
        print(f"{'=' * 70}\n")

        returncode = run_test_file(test_file, verbose, args)
        results[test_file] = returncode

    else:
        # Run all tests
        print(f"\n{'=' * 70}")
        print("Running ALL Evolution-MDAP Tests")
        print(f"{'=' * 70}\n")

        returncode = run_test_file(test_file, verbose)
        results[test_file] = returncode

    # Performance benchmarks (if requested)
    if benchmark:
        print(f"\n{'=' * 70}")
        print("Running Performance Benchmarks")
        print(f"{'=' * 70}\n")

        benchmark_file = "test_leanaide_evolution_mdap.py"
        args = ["--category", "performance"]

        returncode = run_test_file(benchmark_file, verbose, args)
        results[benchmark_file] = returncode

    return results


def print_summary(results: Dict[str, int]):
    """Print test summary"""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for rc in results.values() if rc == 0)
    failed = total - passed

    for test_file, returncode in results.items():
        status = "PASSED" if returncode == 0 else "FAILED"
        symbol = "✓" if returncode == 0 else "✗"
        print(f"{symbol} {test_file}: {status}")

    print(f"\nTotal: {total}, Passed: {passed}, Failed: {failed}")
    print("=" * 70)

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Run LeanAide MDAP-Evolution Tests"
    )

    parser.add_argument(
        "--category",
        "-c",
        action="append",
        choices=TEST_CATEGORIES,
        help="Test category to run (can specify multiple)"
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    parser.add_argument(
        "--benchmark",
        "-b",
        action="store_true",
        help="Run performance benchmarks"
    )

    args = parser.parse_args()

    # Run tests
    start_time = time.time()
    results = run_tests(
        categories=args.category,
        verbose=not args.quiet,
        benchmark=args.benchmark
    )
    elapsed = time.time() - start_time

    # Print summary
    print(f"\nTotal time: {elapsed:.2f}s")
    exit_code = print_summary(results)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
