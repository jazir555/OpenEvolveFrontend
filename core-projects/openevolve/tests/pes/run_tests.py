#!/usr/bin/env python
"""
Run all PES tests with coverage reporting

This script runs the complete PES test suite and generates coverage reports.
Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --verbose          # Run with verbose output
    python run_tests.py --coverage         # Generate coverage report
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_pytest(
    tests_dir: Path,
    verbose: bool = False,
    coverage: bool = False,
    pattern: str = None,
    marker: str = None
) -> int:
    """
    Run pytest with specified options

    Args:
        tests_dir: Directory containing tests
        verbose: Enable verbose output
        coverage: Generate coverage report
        pattern: File pattern to match
        marker: Pytest marker to filter tests

    Returns:
        Exit code from pytest
    """
    cmd = ["python", "-m", "pytest"]

    # Add verbose flag
    if verbose:
        cmd.append("-v")

    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=openevolve",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=60"  # Require at least 60% coverage
        ])

    # Add pattern if specified
    if pattern:
        cmd.append("-k")
        cmd.append(pattern)

    # Add marker if specified
    if marker:
        cmd.append("-m")
        cmd.append(marker)

    # Add tests directory
    cmd.append(str(tests_dir))

    # Run pytest
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=tests_dir.parent.parent.parent)

    return result.returncode


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(
        description="Run PES test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit             # Run only unit tests
  python run_tests.py --integration      # Run only integration tests
  python run_tests.py --verbose          # Verbose output
  python run_tests.py --coverage         # Generate coverage report
  python run_tests.py -k "test_database" # Run tests matching pattern
        """
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests (not integration tests)"
    )

    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )

    parser.add_argument(
        "--pattern", "-k",
        type=str,
        help="Run tests matching pattern"
    )

    parser.add_argument(
        "--no-cov",
        action="store_true",
        help="Disable coverage even if --coverage is default"
    )

    args = parser.parse_args()

    # Determine tests directory
    tests_dir = Path(__file__).parent / "pes"

    if not tests_dir.exists():
        print(f"Error: Tests directory not found: {tests_dir}")
        return 1

    # Determine what to run
    if args.unit:
        pattern = "not integration"
    elif args.integration:
        pattern = "integration"
    else:
        pattern = args.pattern

    # Run tests
    exit_code = run_pytest(
        tests_dir=tests_dir,
        verbose=args.verbose,
        coverage=args.coverage,
        pattern=pattern
    )

    # Print summary
    print(f"\n{'='*60}")
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print(f"{'='*60}\n")

    if args.coverage:
        coverage_dir = tests_dir.parent.parent.parent / "htmlcov"
        if coverage_dir.exists():
            print(f"Coverage report: {coverage_dir / 'index.html'}")
            print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
