"""
MDAP/MAKER Test Runner

Comprehensive test runner for LeanAide MDAP/MAKER integration tests.

Usage:
    python run_mdap_tests.py                    # Run all tests
    python run_mdap_tests.py unit               # Run unit tests only
    python run_mdap_tests.py integration        # Run integration tests only
    python run_mdap_tests.py maker              # Run MAKER tests only
    python run_mdap_tests.py edge               # Run edge case tests
    python run_mdap_tests.py --coverage         # Run with coverage report
    python run_mdap_tests.py --benchmark        # Run performance benchmarks

Author: OpenEvolve Frontend Team
Version: 1.0.0
Date: 2025-12-30
"""

import argparse
import json
import logging
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_CATEGORIES = {
    "unit": "Unit Tests",
    "integration": "Integration Tests",
    "maker": "MAKER Tests",
    "workflow": "Workflow Tests",
    "redflag": "Red-Flagging Tests",
    "edge": "Edge Case Tests",
    "performance": "Performance Tests"
}

# =============================================================================
# TEST SUITE BUILDER
# =============================================================================

class MDAPTestRunner:
    """Custom test runner for MDAP/MAKER tests"""

    def __init__(self, verbosity: int = 2):
        self.verbosity = verbosity
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_tests(
        self,
        category: Optional[str] = None,
        pattern: Optional[str] = None,
        coverage: bool = False
    ) -> unittest.TestResult:
        """Run tests with optional filtering"""

        # Import test module
        try:
            import test_leanaide_mdap
        except ImportError as e:
            logger.error(f"Failed to import test module: {e}")
            return None

        # Build test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        if category:
            # Run specific category
            suite = self._build_category_suite(category, loader)
        elif pattern:
            # Run tests matching pattern
            suite = self._build_pattern_suite(pattern, loader)
        else:
            # Run all tests
            suite = loader.loadTestsFromModule(sys.modules['test_leanaide_mdap'])

        # Run tests
        self.start_time = time.time()

        if coverage:
            result = self._run_with_coverage(suite)
        else:
            runner = unittest.TextTestRunner(verbosity=self.verbosity)
            result = runner.run(suite)

        self.end_time = time.time()

        # Store results
        self._store_results(result, category or "all")

        return result

    def _build_category_suite(
        self,
        category: str,
        loader: unittest.TestLoader
    ) -> unittest.TestSuite:
        """Build test suite for specific category"""

        suite = unittest.TestSuite()

        # Map categories to test classes
        category_map = {
            "unit": [
                "TestMDAPStepConfiguration",
                "TestMDAPTaskConfiguration",
                "TestRedFlagging",
                "TestMDAPCache",
                "TestUtilityFunctions",
                "TestMDAPConfig"
            ],
            "integration": [
                "TestMDAPOrchestrator",
                "TestROMAMDAPMakerIntegration",
                "TestSubProblemStructure"
            ],
            "maker": [
                "TestMAKERWorkflowIntegration"
            ],
            "workflow": [
                "TestWorkflowIntegration"
            ],
            "redflag": [
                "TestRedFlagging"
            ],
            "edge": [
                "TestEdgeCases",
                "TestROMAEdgeCases"
            ],
            "performance": [
                "TestPerformance"
            ]
        }

        # Get test classes for category
        test_classes = category_map.get(category, [])

        if not test_classes:
            logger.warning(f"Unknown category: {category}")
            logger.info(f"Available categories: {list(category_map.keys())}")
            return suite

        # Add tests
        import test_leanaide_mdap
        for test_class_name in test_classes:
            test_class = getattr(test_leanaide_mdap, test_class_name, None)
            if test_class is not None:
                tests = loader.loadTestsFromTestCase(test_class)
                suite.addTests(tests)
            else:
                logger.warning(f"Test class not found: {test_class_name}")

        return suite

    def _build_pattern_suite(
        self,
        pattern: str,
        loader: unittest.TestLoader
    ) -> unittest.TestSuite:
        """Build test suite for pattern matching"""

        suite = unittest.TestSuite()

        # Load all tests
        import test_leanaide_mdap
        all_tests = loader.loadTestsFromModule(test_leanaide_mdap)

        # Filter by pattern
        for test_group in all_tests:
            for test in test_group:
                if pattern.lower() in str(test).lower():
                    suite.addTest(test)

        return suite

    def _run_with_coverage(self, suite: unittest.TestSuite) -> unittest.TestResult:
        """Run tests with coverage tracking"""

        try:
            import coverage

            # Initialize coverage
            cov = coverage.Coverage()
            cov.start()

            # Run tests
            runner = unittest.TextTestRunner(verbosity=self.verbosity)
            result = runner.run(suite)

            # Stop coverage
            cov.stop()

            # Generate report
            print("\n" + "=" * 80)
            print("COVERAGE REPORT")
            print("=" * 80)

            cov.report()

            # Generate HTML report
            html_dir = "test_results/coverage_html"
            os.makedirs(html_dir, exist_ok=True)
            cov.html_report(directory=html_dir)
            print(f"\nHTML coverage report generated: {html_dir}/index.html")

            return result

        except ImportError:
            logger.warning("Coverage module not installed. Running tests without coverage.")
            runner = unittest.TextTestRunner(verbosity=self.verbosity)
            return runner.run(suite)

    def _store_results(self, result: unittest.TestResult, category: str):
        """Store test results"""

        self.results[category] = {
            "tests_run": result.testsRun,
            "successes": result.testsRun - len(result.failures) - len(result.errors),
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "success": result.wasSuccessful(),
            "duration": self.end_time - self.start_time if self.start_time and self.end_time else 0
        }

    def print_summary(self):
        """Print test summary"""

        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        for category, results in self.results.items():
            status = "✓ PASSED" if results["success"] else "✗ FAILED"
            print(f"\n{category.upper()}: {status}")
            print(f"  Tests run: {results['tests_run']}")
            print(f"  Successes: {results['successes']}")
            print(f"  Failures: {results['failures']}")
            print(f"  Errors: {results['errors']}")
            print(f"  Skipped: {results['skipped']}")
            print(f"  Duration: {results['duration']:.2f}s")

        print("\n" + "=" * 80)

    def save_results_json(self, filepath: str = "test_results/mdap_test_results.json"):
        """Save test results to JSON file"""

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Test results saved to: {filepath}")

    def generate_report(self, filepath: str = "test_results/mdap_test_report.txt"):
        """Generate detailed test report"""

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MDAP/MAKER TEST REPORT\n")
            f.write("=" * 80 + "\n\n")

            for category, results in self.results.items():
                f.write(f"{category.upper()}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Status: {'PASSED' if results['success'] else 'FAILED'}\n")
                f.write(f"Tests run: {results['tests_run']}\n")
                f.write(f"Successes: {results['successes']}\n")
                f.write(f"Failures: {results['failures']}\n")
                f.write(f"Errors: {results['errors']}\n")
                f.write(f"Skipped: {results['skipped']}\n")
                f.write(f"Duration: {results['duration']:.2f}s\n\n")

        logger.info(f"Test report generated: {filepath}")


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

class PerformanceBenchmark:
    """Performance benchmarking for MDAP/MAKER"""

    def __init__(self):
        self.results = {}

    def benchmark_cache_performance(self):
        """Benchmark cache performance"""

        logger.info("Running cache performance benchmark...")

        from test_leanaide_mdap import TestPerformance

        suite = unittest.TestLoader().loadTestsFromName(
            'test_leanaide_mdap.TestPerformance.test_cache_performance'
        )

        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        self.results["cache_performance"] = {
            "passed": result.wasSuccessful(),
            "duration": result._testRunEntered
        }

    def benchmark_token_counting(self):
        """Benchmark token counting performance"""

        logger.info("Running token counting benchmark...")

        from test_leanaide_mdap import TestPerformance

        suite = unittest.TestLoader().loadTestsFromName(
            'test_leanaide_mdap.TestPerformance.test_token_count_performance'
        )

        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        self.results["token_counting"] = {
            "passed": result.wasSuccessful(),
            "duration": result._testRunEntered
        }

    def benchmark_schema_validation(self):
        """Benchmark schema validation performance"""

        logger.info("Running schema validation benchmark...")

        from test_leanaide_mdap import TestPerformance

        suite = unittest.TestLoader().loadTestsFromName(
            'test_leanaide_mdap.TestPerformance.test_schema_validation_performance'
        )

        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        self.results["schema_validation"] = {
            "passed": result.wasSuccessful(),
            "duration": result._testRunEntered
        }

    def run_all_benchmarks(self):
        """Run all performance benchmarks"""

        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARKS")
        print("=" * 80 + "\n")

        self.benchmark_cache_performance()
        self.benchmark_token_counting()
        self.benchmark_schema_validation()

        # Print results
        print("\nBenchmark Results:")
        for benchmark, result in self.results.items():
            status = "✓" if result["passed"] else "✗"
            print(f"  {status} {benchmark}: {result['duration']:.4f}s")

    def save_benchmark_results(
        self,
        filepath: str = "test_results/mdap_benchmark_results.json"
    ):
        """Save benchmark results to JSON"""

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Benchmark results saved to: {filepath}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        description="MDAP/MAKER Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     Run all tests
  %(prog)s unit                Run unit tests only
  %(prog)s integration         Run integration tests only
  %(prog)s --coverage          Run all tests with coverage report
  %(prog)s --benchmark         Run performance benchmarks
  %(prog)s RedFlag             Run tests matching "RedFlag"
  %(prog)s unit --coverage     Run unit tests with coverage

Available categories:
  unit          Unit Tests
  integration   Integration Tests
  maker         MAKER Tests
  workflow      Workflow Tests
  redflag       Red-Flagging Tests
  edge          Edge Case Tests
  performance   Performance Tests
        """
    )

    parser.add_argument(
        "category",
        nargs="?",
        choices=list(TEST_CATEGORIES.keys()),
        help="Test category to run"
    )

    parser.add_argument(
        "pattern",
        nargs="?",
        help="Pattern to match test names"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage report"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet output"
    )

    parser.add_argument(
        "--output-dir",
        default="test_results",
        help="Output directory for test results (default: test_results)"
    )

    return parser.parse_args()


def main():
    """Main entry point"""

    # Parse arguments
    args = parse_arguments()

    # Set verbosity
    verbosity = 2
    if args.verbose:
        verbosity = 3
    elif args.quiet:
        verbosity = 0

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run benchmarks if requested
    if args.benchmark:
        benchmark = PerformanceBenchmark()
        benchmark.run_all_benchmarks()
        benchmark.save_benchmark_results(
            os.path.join(args.output_dir, "mdap_benchmark_results.json")
        )
        return 0 if all(r["passed"] for r in benchmark.results.values()) else 1

    # Run tests
    runner = MDAPTestRunner(verbosity=verbosity)

    # Determine what to run
    if args.pattern:
        # Pattern match
        logger.info(f"Running tests matching pattern: {args.pattern}")
        result = runner.run_tests(pattern=args.pattern, coverage=args.coverage)
    elif args.category:
        # Specific category
        logger.info(f"Running test category: {args.category}")
        result = runner.run_tests(category=args.category, coverage=args.coverage)
    else:
        # All tests
        logger.info("Running all tests")
        result = runner.run_tests(coverage=args.coverage)

    # Print summary
    runner.print_summary()

    # Save results
    runner.save_results_json(
        os.path.join(args.output_dir, "mdap_test_results.json")
    )
    runner.generate_report(
        os.path.join(args.output_dir, "mdap_test_report.txt")
    )

    # Exit with appropriate code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
