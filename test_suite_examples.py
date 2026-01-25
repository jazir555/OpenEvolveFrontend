"""
Usage Examples for test_suite.py

This file demonstrates various ways to use the test suite framework
for common testing scenarios.
"""

from pathlib import Path
from test_suite import (
    TestSuite,
    TestSuiteConfig,
    TestType,
    Priority,
    create_test_suite,
    run_tests,
    flaky,
    slow,
    requires_network,
)


# ============================================================================
# EXAMPLE 1: Basic Test Suite Creation
# ============================================================================

def example_1_basic_suite():
    """
    Create and run a basic test suite.

    This is the simplest way to get started with the test suite framework.
    """
    # Create test suite with default configuration
    suite = create_test_suite(
        name="basic_tests",
        test_dirs=["tests"],
    )

    # Run all discovered tests
    result = suite.run()

    # Print results
    print(f"Total tests: {result.total_tests}")
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")
    print(f"Success rate: {result.success_rate:.1f}%")

    return result


# ============================================================================
# EXAMPLE 2: Custom Configuration
# ============================================================================

def example_2_custom_config():
    """
    Create a test suite with custom configuration.

    Demonstrates advanced configuration options.
    """
    # Create custom configuration
    config = TestSuiteConfig(
        test_dirs=[Path("tests/unit"), Path("tests/integration")],
        test_patterns=["test_*.py", "*_test.py"],
        parallel_workers=8,  # Run 8 tests in parallel
        enable_coverage=True,
        coverage_threshold=80.0,  # Require 80% code coverage
        timeout=300,  # 5 minute timeout per test
        max_retries=3,  # Retry failed tests up to 3 times
        retry_flaky_tests=True,
        verbose=True,
        json_report=Path("reports/test_results.json"),
        html_report=Path("reports/test_results.html"),
        history_file=Path("reports/test_history.json"),
    )

    # Create suite with custom config
    suite = TestSuite(config=config, name="custom_suite")

    # Run tests
    result = suite.run()

    return result


# ============================================================================
# EXAMPLE 3: Test Filtering
# ============================================================================

def example_3_test_filtering():
    """
    Filter and run specific tests.

    Demonstrates various filtering options.
    """
    suite = create_test_suite()

    # Filter by test type
    unit_tests = suite.filter_tests(test_type=TestType.UNIT)
    print(f"Found {len(unit_tests)} unit tests")

    # Filter by priority
    critical_tests = suite.filter_tests(priority=Priority.CRITICAL)
    print(f"Found {len(critical_tests)} critical tests")

    # Filter by tags
    api_tests = suite.filter_tests(tags={"api"})
    print(f"Found {len(api_tests)} API tests")

    # Filter by pattern
    user_tests = suite.filter_tests(pattern=r"user.*")
    print(f"Found {len(user_tests)} user-related tests")

    # Combine filters
    fast_api_tests = suite.filter_tests(
        test_type=TestType.API,
        tags={"fast"},
    )
    print(f"Found {len(fast_api_tests)} fast API tests")

    # Run filtered tests
    result = suite.run(test_names=unit_tests)

    return result


# ============================================================================
# EXAMPLE 4: Parallel Execution
# ============================================================================

def example_4_parallel_execution():
    """
    Run tests in parallel for faster execution.

    Demonstrates parallel test execution.
    """
    # Create config with multiple workers
    config = TestSuiteConfig(
        test_dirs=["tests"],
        parallel_workers=8,  # Run 8 tests concurrently
        enable_coverage=False,  # Coverage can be tricky with parallel execution
    )

    suite = TestSuite(config=config, name="parallel_suite")

    # Run tests in parallel
    result = suite.run()

    print(f"Completed {result.total_tests} tests in {result.duration:.2f}s")
    print(f"Average: {result.duration / result.total_tests:.2f}s per test")

    return result


# ============================================================================
# EXAMPLE 5: Using Decorators
# ============================================================================

# Example test functions using decorators

@flaky(max_runs=3, min_passes=1)
def test_network_api():
    """
    Flaky test that may fail due to network issues.

    Will be retried up to 3 times, needs 1 pass to succeed.
    """
    import requests

    response = requests.get("https://api.example.com/data")
    assert response.status_code == 200


@slow
def test_large_dataset_processing():
    """
    Slow test that processes large datasets.

    Marked as slow to allow filtering in CI/CD pipelines.
    """
    data = range(1000000)
    result = sum(data)
    assert result == 499999500000


@requires_network
def test_external_service():
    """
    Test that requires network access.

    Can be skipped in environments without network access.
    """
    import socket

    # Try to connect to external service
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("example.com", 80))
    sock.close()

    assert result == 0


# ============================================================================
# EXAMPLE 6: CI/CD Integration
# ============================================================================

def example_6_ci_cd_integration():
    """
    Example CI/CD pipeline integration.

    Demonstrates how to use the test suite in a CI/CD pipeline.
    """
    import sys

    # Create configuration for CI environment
    config = TestSuiteConfig(
        test_dirs=["tests"],
        parallel_workers=4,
        enable_coverage=True,
        coverage_threshold=70.0,
        stop_on_first_failure=False,  # Run all tests even if some fail
        verbose=True,
        json_report=Path("artifacts/test_results.json"),
        html_report=Path("artifacts/test_results.html"),
        history_file=Path("artifacts/test_history.json"),
    )

    suite = TestSuite(config=config, name="ci_pipeline")

    # Run tests
    result = suite.run()

    # Generate reports
    print("\n" + "=" * 70)
    print("CI/CD Pipeline Test Results")
    print("=" * 70)
    print(f"Total Tests: {result.total_tests}")
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")
    print(f"Errors: {result.errors}")
    print(f"Success Rate: {result.success_rate:.1f}%")
    print(f"Duration: {result.duration:.2f}s")
    print("=" * 70)

    # Exit with appropriate code
    if result.was_successful:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)


# ============================================================================
# EXAMPLE 7: Test History and Trends
# ============================================================================

def example_7_test_history():
    """
    Analyze test history to identify flaky and slow tests.

    Demonstrates historical analysis capabilities.
    """
    # Create suite with history tracking
    config = TestSuiteConfig(
        test_dirs=["tests"],
        history_file=Path("history/test_history.json"),
    )

    suite = TestSuite(config=config)

    # Run tests to populate history
    result = suite.run()

    # Analyze flaky tests (pass rate < 80%)
    flaky_tests = suite.get_flaky_tests(threshold=0.8)

    if flaky_tests:
        print(f"\nFlaky tests detected (pass rate < 80%):")
        for test_name in flaky_tests:
            history = suite.get_trends(test_name)
            print(f"  - {test_name}: {history.pass_rate:.1f}% pass rate")

    # Analyze slow tests (avg duration > 5s)
    slow_tests = suite.get_slow_tests(threshold=5.0)

    if slow_tests:
        print(f"\nSlow tests detected (avg duration > 5s):")
        for test_name in slow_tests:
            history = suite.get_trends(test_name)
            print(f"  - {test_name}: {history.avg_duration:.2f}s average")

    return result


# ============================================================================
# EXAMPLE 8: Quick Test Execution
# ============================================================================

def example_8_quick_execution():
    """
    Quick and simple test execution using convenience functions.

    Demonstrates the run_tests() convenience function.
    """
    # Run all tests
    result = run_tests(verbose=True)

    print(f"Success rate: {result.success_rate:.1f}%")

    # Run only unit tests
    unit_result = run_tests(test_type=TestType.UNIT)

    print(f"Unit tests: {unit_result.passed}/{unit_result.total_tests} passed")

    # Run tests matching pattern
    api_result = run_tests(test_pattern=r"api.*")

    print(f"API tests: {api_result.passed}/{api_result.total_tests} passed")

    # Run in parallel
    parallel_result = run_tests(parallel=True)

    print(f"Parallel execution completed in {parallel_result.duration:.2f}s")


# ============================================================================
# EXAMPLE 9: Selective Test Execution
# ============================================================================

def example_9_selective_execution():
    """
    Select and run specific tests based on various criteria.

    Demonstrates advanced test selection.
    """
    suite = create_test_suite()

    # Run only critical priority tests
    critical_tests = suite.filter_tests(priority=Priority.CRITICAL)
    result = suite.run(test_names=critical_tests)

    print(f"Critical tests: {result.passed}/{result.total_tests} passed")

    # Run integration tests only
    integration_tests = suite.filter_tests(test_type=TestType.INTEGRATION)
    result = suite.run(test_names=integration_tests)

    print(f"Integration tests: {result.passed}/{result.total_tests} passed")

    # Run fast unit tests (not marked as slow)
    all_tests = list(suite._tests.keys())
    fast_tests = [
        name for name in all_tests
        if not suite._tests[name].slow
        and suite._tests[name].test_type == TestType.UNIT
    ]
    result = suite.run(test_names=fast_tests)

    print(f"Fast unit tests: {result.passed}/{result.total_tests} passed")


# ============================================================================
# EXAMPLE 10: Custom Test Discovery
# ============================================================================

def example_10_custom_discovery():
    """
    Custom test discovery patterns.

    Demonstrates flexible test discovery options.
    """
    # Create configuration with custom patterns
    config = TestSuiteConfig(
        test_dirs=[
            Path("tests"),
            Path("integration"),
            Path("e2e"),
        ],
        test_patterns=[
            "test_*.py",      # Match test_*.py
            "*_test.py",      # Match *_test.py
            "test_*.py",      # Match test_*.py in subdirectories
        ],
        exclude_patterns=[
            "*/test_*.py",    # Exclude test_ directories
            "*/tests/*",      # Exclude tests directories
        ],
    )

    suite = TestSuite(config=config)

    # Discover tests with custom patterns
    count = suite.discover_tests()

    print(f"Discovered {count} tests with custom patterns")

    # Run discovered tests
    result = suite.run()

    return result


# ============================================================================
# EXAMPLE 11: Report Generation
# ============================================================================

def example_11_report_generation():
    """
    Generate various test reports.

    Demonstrates report generation capabilities.
    """
    # Create configuration with multiple report formats
    config = TestSuiteConfig(
        test_dirs=["tests"],
        json_report=Path("reports/json/test_results.json"),
        html_report=Path("reports/html/test_results.html"),
        history_file=Path("reports/history/test_history.json"),
        log_file=Path("logs/test_execution.log"),
    )

    # Create reports directory
    config.json_report.parent.mkdir(parents=True, exist_ok=True)
    config.html_report.parent.mkdir(parents=True, exist_ok=True)
    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    config.history_file.parent.mkdir(parents=True, exist_ok=True)

    suite = TestSuite(config=config)

    # Run tests and generate reports
    result = suite.run()

    print(f"\nReports generated:")
    print(f"  - JSON: {config.json_report}")
    print(f"  - HTML: {config.html_report}")
    print(f"  - History: {config.history_file}")
    print(f"  - Log: {config.log_file}")

    return result


# ============================================================================
# EXAMPLE 12: Test Suite for Different Environments
# ============================================================================

def example_12_environment_specific():
    """
    Create test suites for different environments.

    Demonstrates environment-specific configuration.
    """
    import os

    environment = os.getenv("TEST_ENV", "development")

    # Environment-specific configurations
    configs = {
        "development": TestSuiteConfig(
            test_dirs=["tests/unit"],
            parallel_workers=2,
            enable_coverage=False,
            verbose=True,
        ),
        "staging": TestSuiteConfig(
            test_dirs=["tests/unit", "tests/integration"],
            parallel_workers=4,
            enable_coverage=True,
            coverage_threshold=70.0,
        ),
        "production": TestSuiteConfig(
            test_dirs=["tests"],
            parallel_workers=8,
            enable_coverage=True,
            coverage_threshold=80.0,
            retry_flaky_tests=True,
            json_report=Path(f"reports/production/test_results_{environment}.json"),
        ),
    }

    # Get configuration for current environment
    config = configs.get(environment, configs["development"])

    suite = TestSuite(config=config, name=f"{environment}_tests")

    result = suite.run()

    print(f"\n{environment.upper()} Environment Test Results:")
    print(f"  Total: {result.total_tests}")
    print(f"  Passed: {result.passed}")
    print(f"  Success Rate: {result.success_rate:.1f}%")

    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Test Suite Usage Examples")
    print("=" * 70)

    # Run examples (comment out as needed)
    print("\nExample 1: Basic Test Suite")
    example_1_basic_suite()

    print("\nExample 2: Custom Configuration")
    # example_2_custom_config()  # Requires test files

    print("\nExample 3: Test Filtering")
    # example_3_test_filtering()  # Requires test files

    print("\nExample 4: Parallel Execution")
    # example_4_parallel_execution()  # Requires test files

    print("\nExample 5: Using Decorators")
    print("  See test function definitions above")

    print("\nExample 6: CI/CD Integration")
    # example_6_ci_cd_integration()  # Requires test files

    print("\nExample 7: Test History and Trends")
    # example_7_test_history()  # Requires test files

    print("\nExample 8: Quick Execution")
    # example_8_quick_execution()  # Requires test files

    print("\nExample 9: Selective Execution")
    # example_9_selective_execution()  # Requires test files

    print("\nExample 10: Custom Discovery")
    # example_10_custom_discovery()  # Requires test files

    print("\nExample 11: Report Generation")
    # example_11_report_generation()  # Requires test files

    print("\nExample 12: Environment-Specific")
    # example_12_environment_specific()  # Requires test files

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("\nNote: Some examples are commented out as they require test files.")
    print("Uncomment the examples you want to run.")
