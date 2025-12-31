#!/usr/bin/env python3
"""
MCTS-MDAP Test Runner

Comprehensive test runner for MCTS-MDAP integration tests.
Supports running specific test categories, generating coverage reports,
and performance benchmarking.

Usage:
    python run_mcts_mdap_tests.py                    # Run all tests
    python run_mcts_mdap_tests.py --category unit    # Run unit tests only
    python run_mcts_mdap_tests.py --coverage         # Generate coverage report
    python run_mcts_mdap_tests.py --benchmark        # Run performance benchmarks
    python run_mcts_mdap_tests.py --verbose          # Verbose output
    python run_mcts_mdap_tests.py --help            # Show help

Author: OpenEvolve
Created: 2025-12-30
"""

import argparse
import json
import logging
import os
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mcts_mdap_test_results.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

class TestConfig:
    """Configuration for test runner."""

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        enable_slow_tests: bool = False,
        enable_integration: bool = True,
        verbose: bool = False,
        generate_coverage: bool = False,
        run_benchmarks: bool = False,
        output_dir: str = "test_results"
    ):
        self.categories = categories or ["unit", "integration", "workflow", "edge_cases"]
        self.enable_slow_tests = enable_slow_tests
        self.enable_integration = enable_integration
        self.verbose = verbose
        self.generate_coverage = generate_coverage
        self.run_benchmarks = run_benchmarks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)


# =============================================================================
# TEST SUITE BUILDER
# =============================================================================

def build_test_suite(config: TestConfig) -> unittest.TestSuite:
    """
    Build test suite based on configuration.

    Args:
        config: Test configuration

    Returns:
        Configured test suite
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    try:
        from test_leanaide_mcts_mdap import (
            TestMDAPMCTSNode,
            TestMDAPMCTSExpansion,
            TestMDAPMCTSSimulation,
            TestMDAPMCTSOrchestration,
            TestMCTSDAPIntegration,
            TestMDAPMCTSWorkflow,
            TestMCTSDAPPerformance,
            TestMDAPMCTSEdgeCases
        )
    except ImportError as e:
        logger.error(f"Failed to import test modules: {e}")
        logger.error("Make sure test_leanaide_mcts_mdap.py is in the current directory")
        sys.exit(1)

    # Add tests by category
    if "unit" in config.categories:
        logger.info("Adding unit tests...")
        suite.addTests(loader.loadTestsFromTestCase(TestMDAPMCTSNode))
        suite.addTests(loader.loadTestsFromTestCase(TestMDAPMCTSExpansion))
        suite.addTests(loader.loadTestsFromTestCase(TestMDAPMCTSSimulation))
        suite.addTests(loader.loadTestsFromTestCase(TestMDAPMCTSOrchestration))

    if "integration" in config.categories and config.enable_integration:
        logger.info("Adding integration tests...")
        suite.addTests(loader.loadTestsFromTestCase(TestMCTSDAPIntegration))

    if "workflow" in config.categories and config.enable_integration:
        logger.info("Adding workflow tests...")
        suite.addTests(loader.loadTestsFromTestCase(TestMDAPMCTSWorkflow))

    if "performance" in config.categories:
        if config.enable_slow_tests:
            logger.info("Adding performance tests...")
            suite.addTests(loader.loadTestsFromTestCase(TestMCTSDAPPerformance))
        else:
            logger.warning("Performance tests skipped (use --enable-slow to enable)")

    if "edge_cases" in config.categories:
        logger.info("Adding edge case tests...")
        suite.addTests(loader.loadTestsFromTestCase(TestMDAPMCTSEdgeCases))

    return suite


# =============================================================================
# TEST RUNNER
# =============================================================================

class MCTSDAPTestRunner(unittest.TextTestRunner):
    """Custom test runner with additional reporting."""

    def __init__(self, config: TestConfig, *args, **kwargs):
        self.config = config
        super().__init__(*args, **kwargs)

    def run(self, test):
        """
        Run test suite with enhanced reporting.

        Args:
            test: Test suite to run

        Returns:
            Test result
        """
        start_time = time.time()
        result = super().run(test)
        elapsed_time = time.time() - start_time

        # Generate enhanced report
        self._generate_report(result, elapsed_time)

        return result

    def _generate_report(self, result: unittest.TestResult, elapsed_time: float):
        """Generate test report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": elapsed_time,
            "tests_run": result.testsRun,
            "successes": result.testsRun - len(result.failures) - len(result.errors),
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / max(1, result.testsRun),
            "failure_details": [
                {
                    "test": str(test[0]),
                    "traceback": test[1]
                }
                for test in result.failures
            ],
            "error_details": [
                {
                    "test": str(test[0]),
                    "traceback": test[1]
                }
                for test in result.errors
            ]
        }

        # Save JSON report
        json_path = self.config.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nTest report saved to: {json_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Tests run:      {report['tests_run']}")
        print(f"Successes:      {report['successes']}")
        print(f"Failures:       {report['failures']}")
        print(f"Errors:         {report['errors']}")
        print(f"Skipped:        {report['skipped']}")
        print(f"Success rate:   {report['success_rate']:.1%}")
        print(f"Elapsed time:   {elapsed_time:.2f}s")
        print("=" * 80)


# =============================================================================
# COVERAGE ANALYSIS
# =============================================================================

def run_coverage_analysis(config: TestConfig):
    """
    Run tests with coverage analysis.

    Args:
        config: Test configuration
    """
    try:
        import coverage
    except ImportError:
        logger.error("Coverage package not installed. Install with: pip install coverage")
        sys.exit(1)

    # Create coverage object
    cov = coverage.Coverage(
        source=[
            "leanaide_mcts",
            "mdap_engine",
            "mdap_maker_complete"
        ],
        omit=[
            "*/test_*",
            "*/__init__.py"
        ]
    )

    # Start coverage
    cov.start()

    # Run tests
    suite = build_test_suite(config)
    runner = MCTSDAPTestRunner(config, verbosity=2 if config.verbose else 1)
    runner.run(suite)

    # Stop coverage
    cov.stop()

    # Generate reports
    logger.info("\nGenerating coverage reports...")

    # Terminal report
    print("\n" + "=" * 80)
    print("COVERAGE REPORT")
    print("=" * 80)
    cov.report()

    # HTML report
    html_dir = config.output_dir / "coverage_html"
    cov.html_report(directory=str(html_dir))
    logger.info(f"HTML coverage report: {html_dir}/index.html")

    # XML report (for CI)
    xml_path = config.output_dir / "coverage.xml"
    cov.xml_report(outfile=str(xml_path))
    logger.info(f"XML coverage report: {xml_path}")


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

def run_performance_benchmarks(config: TestConfig):
    """
    Run performance benchmarks comparing MCTS vs MDAP-MCTS.

    Args:
        config: Test configuration
    """
    logger.info("Running performance benchmarks...")

    try:
        from leanaide_mcts import MCTSConfig, ProofState, search_proof_with_mcts
        from mdap_engine import MDAPConfig, MDAPOrchestrator
        from workflow_structures import Team, ModelConfig
        from test_leanaide_mcts_mdap import search_with_mdap_mcts
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        sys.exit(1)

    # Benchmark configuration
    benchmarks = [
        {
            "name": "Simple Proof (Nat.add_zero)",
            "theorem": "forall (n : Nat), n + 0 = n",
            "iterations": 500,
            "time_budget": 30.0
        },
        {
            "name": "Medium Proof (Nat.add_comm)",
            "theorem": "forall (a b : Nat), a + b = b + a",
            "iterations": 1000,
            "time_budget": 60.0
        }
    ]

    results = []

    for benchmark in benchmarks:
        logger.info(f"\nBenchmark: {benchmark['name']}")

        # Create state
        state = ProofState(
            goals=[benchmark["theorem"]],
            context=[],
            depth=0
        )

        # Pure MCTS
        logger.info("  Running pure MCTS...")
        config_mcts = MCTSConfig(
            max_iterations=benchmark["iterations"],
            time_budget=benchmark["time_budget"]
        )

        start = time.time()
        result_mcts = search_proof_with_mcts(state, config_mcts)
        time_mcts = time.time() - start

        logger.info(f"    Success: {result_mcts.success}")
        logger.info(f"    Time: {time_mcts:.2f}s")
        logger.info(f"    Win rate: {result_mcts.win_rate:.2%}")

        # MDAP-MCTS (if team available)
        try:
            team = Team(
                team_id="benchmark_team",
                name="Benchmark Team",
                members=[
                    ModelConfig(
                        model_id="mock",
                        api_key="test",
                        api_base="http://test",
                        temperature=0.0
                    )
                ]
            )

            mdap_config = MDAPConfig(k_min=2, k_max=5)

            logger.info("  Running MDAP-MCTS...")
            start = time.time()
            result_mdap = search_with_mdap_mcts(
                state,
                config_mcts,
                mdap_config,
                team
            )
            time_mdap = time.time() - start

            logger.info(f"    Success: {result_mdap.success}")
            logger.info(f"    Time: {time_mdap:.2f}s")
            logger.info(f"    Win rate: {result_mdap.win_rate:.2%}")

            # Calculate overhead
            overhead = ((time_mdap - time_mcts) / max(time_mcts, 0.001)) * 100
            logger.info(f"    Overhead: {overhead:.1f}%")

        except Exception as e:
            logger.warning(f"  MDAP-MCTS skipped: {e}")
            time_mdap = None
            overhead = None

        # Store results
        results.append({
            "benchmark": benchmark["name"],
            "mcts_success": result_mcts.success,
            "mcts_time": time_mcts,
            "mcts_win_rate": result_mcts.win_rate,
            "mdap_success": result_mdap.success if 'result_mdap' in locals() else None,
            "mdap_time": time_mdap,
            "mdap_win_rate": result_mdap.win_rate if 'result_mdap' in locals() else None,
            "overhead_percent": overhead
        })

    # Save benchmark results
    benchmark_path = config.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(benchmark_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nBenchmark results saved to: {benchmark_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Benchmark':<40} {'MCTS':<10} {'MDAP-MCTS':<10} {'Overhead':<10}")
    print("-" * 80)
    for r in results:
        mcts_str = f"{r['mcts_time']:.1f}s ({r['mcts_win_rate']:.0%})"
        mdap_str = f"{r['mdap_time']:.1f}s" if r['mdap_time'] else "N/A"
        overhead_str = f"{r['overhead_percent']:.0f}%" if r['overhead_percent'] else "N/A"
        print(f"{r['benchmark']:<40} {mcts_str:<10} {mdap_str:<10} {overhead_str:<10}")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MCTS-MDAP Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mcts_mdap_tests.py                     Run all tests
  python run_mcts_mdap_tests.py --category unit     Run unit tests only
  python run_mcts_mdap_tests.py --coverage          Generate coverage report
  python run_mcts_mdap_tests.py --benchmark         Run performance benchmarks
  python run_mcts_mdap_tests.py --verbose           Verbose output

Categories:
  unit          Unit tests for individual components
  integration   Integration tests for MCTS-MDAP workflows
  workflow      Workflow integration tests (Stage 3A/3B)
  performance   Performance comparison tests
  edge_cases    Edge case and error handling tests
        """
    )

    parser.add_argument(
        "--category",
        "-c",
        action="append",
        choices=["unit", "integration", "workflow", "performance", "edge_cases"],
        help="Test category to run (can specify multiple)"
    )

    parser.add_argument(
        "--enable-slow",
        action="store_true",
        help="Enable slow tests (performance tests)"
    )

    parser.add_argument(
        "--no-integration",
        action="store_true",
        help="Skip integration tests"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )

    parser.add_argument(
        "--benchmark",
        "-b",
        action="store_true",
        help="Run performance benchmarks"
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="test_results",
        help="Output directory for test results (default: test_results)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Create configuration
    config = TestConfig(
        categories=args.category,
        enable_slow_tests=args.enable_slow,
        enable_integration=not args.no_integration,
        verbose=args.verbose,
        generate_coverage=args.coverage,
        run_benchmarks=args.benchmark,
        output_dir=args.output_dir
    )

    print("=" * 80)
    print("MCTS-MDAP TEST RUNNER")
    print("=" * 80)
    print(f"Categories:      {', '.join(config.categories)}")
    print(f"Slow tests:      {config.enable_slow_tests}")
    print(f"Integration:     {config.enable_integration}")
    print(f"Verbose:         {config.verbose}")
    print(f"Coverage:        {config.generate_coverage}")
    print(f"Benchmarks:      {config.run_benchmarks}")
    print(f"Output dir:      {config.output_dir}")
    print("=" * 80)

    start_time = time.time()

    try:
        # Run coverage analysis if requested
        if config.generate_coverage:
            run_coverage_analysis(config)

        # Run benchmarks if requested
        elif config.run_benchmarks:
            run_performance_benchmarks(config)

        # Run standard tests
        else:
            suite = build_test_suite(config)
            runner = MCTSDAPTestRunner(config, verbosity=2 if config.verbose else 1)
            result = runner.run(suite)

            # Exit with appropriate code
            elapsed = time.time() - start_time
            logger.info(f"\nTotal time: {elapsed:.2f}s")

            sys.exit(0 if result.wasSuccessful() else 1)

    except KeyboardInterrupt:
        logger.warning("\nTests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
