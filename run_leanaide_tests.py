#!/usr/bin/env python3
"""
LeanAide Test Runner

A convenient script to run LeanAide integration tests with various options.

Usage:
    python run_leanaide_tests.py                    # Run all tests
    python run_leanaide_tests.py --unit             # Run unit tests only
    python run_leanaide_tests.py --integration      # Run integration tests only
    python run_leanaide_tests.py --mock             # Run offline (mock) tests
    python run_leanaide_tests.py --server           # Run server-required tests
    python run_leanaide_tests.py --fast             # Run fast tests only (no slow)
    python run_leanaide_tests.py --coverage         # Run with coverage report
    python run_leanaide_tests.py --verbose          # Verbose output
    python run_leanaide_tests.py --help             # Show help

Author: OpenEvolve
Created: 2025-12-30
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LeanAide integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all tests
  %(prog)s --unit --verbose     Run unit tests with verbose output
  %(prog)s --mock --coverage    Run mock tests with coverage
  %(prog)s --integration --fast Run integration tests, skip slow ones
        """
    )

    parser.add_argument(
        "--unit", "-u",
        action="store_true",
        help="Run unit tests only"
    )

    parser.add_argument(
        "--integration", "-i",
        action="store_true",
        help="Run integration tests only"
    )

    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="Run mock (offline) tests only"
    )

    parser.add_argument(
        "--server", "-s",
        action="store_true",
        help="Run tests requiring LeanAide server"
    )

    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Skip slow tests"
    )

    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for test results"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all tests without running them"
    )

    return parser.parse_args()


def build_pytest_command(args):
    """Build pytest command based on arguments."""
    cmd = ["python", "-m", "pytest"]

    # Test file
    test_file = Path(__file__).parent / "test_leanaide_integration.py"
    cmd.append(str(test_file))

    # Verbose
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # Show print output
    cmd.append("-s")

    # Markers
    markers = []
    if args.unit:
        markers.append("unit")
    elif args.integration:
        markers.append("integration")
    elif args.mock:
        markers.append("mock")
    elif args.server:
        markers.append("server")

    if markers:
        cmd.extend(["-m", " and ".join(markers)])

    # Skip slow tests
    if args.fast:
        cmd.append("-m")
        cmd.append("not slow")

    # Coverage
    if args.coverage:
        cmd.extend([
            "--cov=leanaide_client",
            "--cov=leanaide_mcp_tools",
            "--cov=leanaide_CREWAI_bridge",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])

    # Parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])

    # Output file
    if args.output:
        cmd.extend(["--tb=short", f"--junitxml={args.output}"])

    # List tests
    if args.list:
        cmd.append("--collect-only")

    return cmd


def check_dependencies():
    """Check if required dependencies are installed."""
    required = ["pytest", "pytest-asyncio"]
    optional = ["pytest-cov", "pytest-xdist"]

    missing_required = []
    missing_optional = []

    for package in required:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_required.append(package)

    for package in optional:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_optional.append(package)

    if missing_required:
        print("[FAIL] Missing required packages:")
        for pkg in missing_required:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing_required))
        return False

    if missing_optional:
        print("[WARN]  Missing optional packages:")
        for pkg in missing_optional:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing_optional))
        print()

    return True


def print_banner():
    """Print test suite banner."""
    print("=" * 70)
    print("LeanAide Integration Test Suite")
    print("=" * 70)
    print()


def print_summary(args):
    """Print test run summary."""
    print("Test Configuration:")
    print(f"  Unit tests:        {'Yes' if args.unit else 'No'}")
    print(f"  Integration tests: {'Yes' if args.integration else 'No'}")
    print(f"  Mock tests:        {'Yes' if args.mock else 'No'}")
    print(f"  Server tests:      {'Yes' if args.server else 'No'}")
    print(f"  Skip slow tests:   {'Yes' if args.fast else 'No'}")
    print(f"  Coverage:          {'Yes' if args.coverage else 'No'}")
    print(f"  Parallel:          {'Yes' if args.parallel else 'No'}")
    print(f"  Verbose:           {'Yes' if args.verbose else 'No'}")
    print()


def run_tests(cmd):
    """Run the pytest command."""
    print("Running command:")
    print(" ".join(cmd))
    print()
    print("-" * 70)
    print()

    result = subprocess.run(cmd)

    print()
    print("-" * 70)
    print()

    if result.returncode == 0:
        print("[OK] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed.")
        return result.returncode


def main():
    """Main entry point."""
    args = parse_arguments()

    print_banner()
    print_summary(args)

    # Check dependencies
    if not check_dependencies():
        return 1

    # Build command
    cmd = build_pytest_command(args)

    # Run tests
    return run_tests(cmd)


if __name__ == "__main__":
    sys.exit(main())
