"""
Edge Case Test Runner for Gauntlet Components

Runs all edge case tests and generates coverage report.

Usage:
    python run_edge_case_tests.py
    python run_edge_case_tests.py --component ml_optimizer
    python run_edge_case_tests.py --coverage

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import sys
import os
import unittest
import argparse
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_tests(component=None, coverage=False):
    """
    Run edge case tests.

    Args:
        component: Specific component to test (ml_optimizer, predictive_executor, adaptive_learner, websocket)
        coverage: Whether to generate coverage report
    """
    # Import test modules
    test_modules = []

    if component is None or component == "ml_optimizer":
        from tests.gauntlets import test_edge_cases_ml_optimizer
        test_modules.append(test_edge_cases_ml_optimizer)

    if component is None or component == "predictive_executor":
        from tests.gauntlets import test_edge_cases_predictive_executor
        test_modules.append(test_edge_cases_predictive_executor)

    if component is None or component == "adaptive_learner":
        from tests.gauntlets import test_edge_cases_adaptive_learner
        test_modules.append(test_edge_cases_adaptive_learner)

    if component is None or component == "websocket":
        from tests.gauntlets import test_edge_cases_websocket
        test_modules.append(test_edge_cases_websocket)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for module in test_modules:
        suite.addTests(loader.loadTestsFromModule(module))

    # Run tests
    if coverage:
        try:
            import coverage

            # Initialize coverage
            cov = coverage.Coverage(
                source=[
                    "glue/adapters/gauntlet-adapter/src/ml_optimizer.py",
                    "glue/adapters/gauntlet-adapter/src/predictive_gauntlet_executor.py",
                    "glue/adapters/gauntlet-adapter/src/adaptive_learner.py",
                    "api/gauntlets_websocket.py"
                ],
                omit=["*/tests/*", "*/test_*.py"]
            )

            cov.start()

            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)

            cov.stop()

            # Generate coverage report
            print("\n" + "="*80)
            print("COVERAGE REPORT")
            print("="*80)

            cov.report()

            # Generate HTML report
            html_dir = Path(__file__).parent / "coverage_html"
            cov.html_report(directory=str(html_dir))

            print(f"\nHTML coverage report generated in: {html_dir}")

            return result.wasSuccessful()

        except ImportError:
            print("Coverage package not installed. Install with: pip install coverage")
            print("Running tests without coverage...")

            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)

            return result.wasSuccessful()
    else:
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return result.wasSuccessful()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run edge case tests for gauntlet components"
    )

    parser.add_argument(
        "--component",
        choices=["ml_optimizer", "predictive_executor", "adaptive_learner", "websocket"],
        help="Test specific component (default: all)"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )

    args = parser.parse_args()

    print("="*80)
    print("EDGE CASE TEST SUITE FOR GAUNTLET COMPONENTS")
    print("="*80)
    print(f"Component: {args.component or 'All'}")
    print(f"Coverage: {args.coverage}")
    print("="*80)
    print()

    success = run_tests(args.component, args.coverage)

    print()
    print("="*80)
    if success:
        print("ALL TESTS PASSED [OK]")
    else:
        print("SOME TESTS FAILED [FAIL]")
    print("="*80)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
