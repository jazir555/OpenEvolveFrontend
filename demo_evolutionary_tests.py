#!/usr/bin/env python3
"""
Demo: Using the Evolutionary LeanAide Test Suite

This script demonstrates how to use the comprehensive test suite
for evolutionary LeanAide components.

Author: OpenEvolve
Created: 2025-12-30
"""

import sys
import subprocess
from pathlib import Path


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def print_section(title: str):
    """Print a formatted section."""
    print(f"\n{title}")
    print("-" * len(title))


def run_command(cmd: list, description: str):
    """Run a command and display results."""
    print(f"\n$ {' '.join(cmd)}")
    print(f"# {description}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\n[OK] SUCCESS: {description}")
    else:
        print(f"\n[FAIL] FAILED: {description}")

    return result.returncode == 0


def demo_basic_usage():
    """Demonstrate basic test usage."""
    print_header("DEMO 1: Basic Test Usage")

    # Run a few fast unit tests
    print_section("Running fast unit tests")
    run_command(
        ["python", "-m", "pytest", "test_leanaide_evolutionary.py", "-v", "-m", "unit", "-k", "test_strategy_creation", "--tb=short"],
        "Run a simple test (strategy creation)"
    )


def demo_evolution_tests():
    """Demonstrate evolution test category."""
    print_header("DEMO 2: Evolution Tests")

    print_section("Testing evolutionary components")

    # Test population statistics
    run_command(
        ["python", "-m", "pytest", "test_leanaide_evolutionary.py", "-v",
         "-m", "evolution", "-k", "test_population_statistics", "--tb=short"],
        "Test population statistics calculation"
    )


def demo_decomposition_tests():
    """Demonstrate decomposition test category."""
    print_header("DEMO 3: Decomposition Tests")

    print_section("Testing decomposition components")

    # Test component extraction
    run_command(
        ["python", "-m", "pytest", "test_leanaide_evolutionary.py", "-v",
         "-m", "decomposition", "-k", "test_extract_simple_components", "--tb=short"],
        "Test mathematical component extraction"
    )


def demo_test_runner():
    """Demonstrate test runner usage."""
    print_header("DEMO 4: Using Test Runner")

    print_section("Running tests via test runner script")

    # Run evolution tests via test runner
    run_command(
        ["python", "run_evolutionary_tests.py", "--evolution", "--quiet"],
        "Run evolution tests using test runner"
    )


def demo_coverage():
    """Demonstrate coverage generation."""
    print_header("DEMO 5: Coverage Reporting")

    print_section("Generating test coverage report")

    # Note: This requires pytest-cov
    print("Note: This requires pytest-cov to be installed")
    print("Install with: pip install pytest pytest-cov pytest-asyncio\n")

    run_command(
        ["python", "-m", "pytest", "test_leanaide_evolutionary.py",
         "--cov=.", "--cov-report=term-missing", "-v", "-m", "evolution", "-k", "test_strategy", "--tb=short"],
        "Generate coverage report for evolution tests"
    )


def demo_selective_testing():
    """Demonstrate selective test execution."""
    print_header("DEMO 6: Selective Test Execution")

    print_section("Running specific tests by category")

    # Run only unit tests
    run_command(
        ["python", "-m", "pytest", "test_leanaide_evolutionary.py", "-v",
         "-m", "unit", "-k", "evolution", "--tb=short"],
        "Run only evolution unit tests"
    )


def demo_parallel_execution():
    """Demonstrate parallel test execution."""
    print_header("DEMO 7: Parallel Execution")

    print_section("Running tests in parallel")

    print("Note: This requires pytest-xdist to be installed")
    print("Install with: pip install pytest-xdist\n")

    # Run tests in parallel
    run_command(
        ["python", "-m", "pytest", "test_leanaide_evolutionary.py", "-v",
         "-n", "auto", "-m", "unit", "-k", "test_strategy", "--tb=short"],
        "Run tests in parallel (if pytest-xdist is available)"
    )


def demo_offline_testing():
    """Demonstrate offline (mock) testing."""
    print_header("DEMO 8: Offline Testing")

    print_section("Running tests without LeanAide server")

    # Run mock tests
    run_command(
        ["python", "-m", "pytest", "test_leanaide_evolutionary.py", "-v",
         "-m", "mock", "-k", "test_strategy", "--tb=short"],
        "Run mock tests (offline, no server required)"
    )


def main():
    """Run all demos."""
    print_header("Evolutionary LeanAide Test Suite - Demo")
    print("\nThis demo shows various ways to use the test suite.")
    print("Each demo runs a small subset of tests for demonstration.\n")

    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Evolution Tests", demo_evolution_tests),
        ("Decomposition Tests", demo_decomposition_tests),
        ("Test Runner", demo_test_runner),
        ("Coverage Reporting", demo_coverage),
        ("Selective Testing", demo_selective_testing),
        ("Parallel Execution", demo_parallel_execution),
        ("Offline Testing", demo_offline_testing),
    ]

    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  0. All demos")

    choice = input("\nSelect demo (0-8, or press Enter for all): ").strip()

    selected_demos = []
    if choice == "" or choice == "0":
        selected_demos = demos
    elif choice.isdigit() and 1 <= int(choice) <= len(demos):
        selected_demos = [demos[int(choice) - 1]]
    else:
        print("Invalid choice. Running all demos.\n")
        selected_demos = demos

    # Run selected demos
    for name, demo_func in selected_demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\nError in {name} demo: {e}")

    # Summary
    print_header("Demo Complete")
    print("\nTo run the full test suite:")
    print("  python run_evolutionary_tests.py --all")
    print("\nOr with pytest:")
    print("  pytest test_leanaide_evolutionary.py -v")
    print("\nFor more information, see:")
    print("  - LEANAIDE_EVOLUTIONARY_TEST_SUITE_GUIDE.md (complete guide)")
    print("  - LEANAIDE_QUICK_TEST_REFERENCE.md (quick reference)")
    print("  - README_EVOLUTIONARY_TESTS.md (overview)")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
