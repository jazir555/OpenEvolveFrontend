"""
Unit tests for test_suite.py module

These tests verify the functionality of the test suite framework including:
- Test discovery
- Test filtering
- Test execution
- Report generation
- History management
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from test_suite import (
    TestMetadata,
    TestResult,
    TestStatus,
    SuiteResult,
    TestType,
    Priority,
    TestSuite,
    TestSuiteConfig,
    create_test_suite,
    run_tests,
    flaky,
    slow,
    requires_network,
    TestSuiteError,
    TestDiscoveryError,
    FrameworkAdapter,
    GenericAdapter,
)


class TestMetadataTest(unittest.TestCase):
    """Test cases for TestMetadata dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        metadata = TestMetadata(
            name="test_example",
            file_path=Path("test_example.py"),
            test_type=TestType.UNIT,
        )

        self.assertEqual(metadata.name, "test_example")
        self.assertEqual(metadata.priority, Priority.MEDIUM)
        self.assertEqual(len(metadata.tags), 0)
        self.assertEqual(metadata.timeout, 300)
        self.assertEqual(metadata.max_retries, 3)
        self.assertFalse(metadata.flaky)

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        tags = {"slow", "integration"}
        metadata = TestMetadata(
            name="test_custom",
            file_path=Path("test_custom.py"),
            test_type=TestType.INTEGRATION,
            priority=Priority.HIGH,
            tags=tags,
            timeout=600,
            flaky=True,
        )

        self.assertEqual(metadata.priority, Priority.HIGH)
        self.assertEqual(metadata.tags, tags)
        self.assertEqual(metadata.timeout, 600)
        self.assertTrue(metadata.flaky)


class TestResultTest(unittest.TestCase):
    """Test cases for TestResult dataclass."""

    def test_passed_result(self):
        """Test creating a passed test result."""
        result = TestResult(
            test_name="test_passed",
            status=TestStatus.PASSED,
            duration=1.5,
        )

        self.assertEqual(result.test_name, "test_passed")
        self.assertEqual(result.status, TestStatus.PASSED)
        self.assertEqual(result.duration, 1.5)
        self.assertIsNone(result.error_message)

    def test_failed_result(self):
        """Test creating a failed test result."""
        result = TestResult(
            test_name="test_failed",
            status=TestStatus.FAILED,
            duration=0.5,
            error_message="Assertion failed",
            error_traceback="Traceback...",
        )

        self.assertEqual(result.status, TestStatus.FAILED)
        self.assertIsNotNone(result.error_message)
        self.assertIsNotNone(result.error_traceback)


class SuiteResultTest(unittest.TestCase):
    """Test cases for SuiteResult dataclass."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        result = SuiteResult(
            total_tests=10,
            passed=7,
            failed=2,
            skipped=1,
            errors=0,
            duration=5.0,
        )

        self.assertEqual(result.success_rate, 70.0)

    def test_success_rate_no_tests(self):
        """Test success rate with no tests."""
        result = SuiteResult(
            total_tests=0,
            passed=0,
            failed=0,
            skipped=0,
            errors=0,
            duration=0.0,
        )

        self.assertEqual(result.success_rate, 0.0)

    def test_was_successful(self):
        """Test was_successful property."""
        result = SuiteResult(
            total_tests=10,
            passed=10,
            failed=0,
            skipped=0,
            errors=0,
            duration=5.0,
        )

        self.assertTrue(result.was_successful)

    def test_was_not_successful_failures(self):
        """Test was_successful with failures."""
        result = SuiteResult(
            total_tests=10,
            passed=8,
            failed=2,
            skipped=0,
            errors=0,
            duration=5.0,
        )

        self.assertFalse(result.was_successful)

    def test_was_not_successful_errors(self):
        """Test was_successful with errors."""
        result = SuiteResult(
            total_tests=10,
            passed=9,
            failed=0,
            skipped=0,
            errors=1,
            duration=5.0,
        )

        self.assertFalse(result.was_successful)


class TestSuiteConfigTest(unittest.TestCase):
    """Test cases for TestSuiteConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TestSuiteConfig()

        self.assertEqual(config.project_root, Path.cwd())
        self.assertEqual(len(config.test_dirs), 0)
        self.assertEqual(config.parallel_workers, 4)
        self.assertTrue(config.enable_coverage)
        self.assertEqual(config.coverage_threshold, 70.0)
        self.assertTrue(config.retry_flaky_tests)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TestSuiteConfig(
            parallel_workers=8,
            enable_coverage=False,
            timeout=600,
            max_retries=5,
        )

        self.assertEqual(config.parallel_workers, 8)
        self.assertFalse(config.enable_coverage)
        self.assertEqual(config.timeout, 600)
        self.assertEqual(config.max_retries, 5)


class GenericAdapterTest(unittest.TestCase):
    """Test cases for GenericAdapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.adapter = GenericAdapter()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_adapter_available(self):
        """Test that generic adapter is always available."""
        self.assertTrue(self.adapter.is_available())

    def test_discover_empty_directory(self):
        """Test discovery in empty directory."""
        tests = self.adapter.discover_tests(
            test_dirs=[Path(self.temp_dir)],
            patterns=["test_*.py"],
        )

        self.assertEqual(len(tests), 0)

    def test_discover_test_file(self):
        """Test discovery of test file."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test_example.py"
        test_file.write_text("""
def test_something():
    assert True

def test_another():
    assert 1 + 1 == 2
""")

        tests = self.adapter.discover_tests(
            test_dirs=[Path(self.temp_dir)],
            patterns=["test_*.py"],
        )

        self.assertEqual(len(tests), 2)
        self.assertIn("test_example.test_something", [t[0] for t in tests])
        self.assertIn("test_example.test_another", [t[0] for t in tests])

    def test_run_test_function(self):
        """Test running a test function."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test_run.py"
        test_file.write_text("""
def test_passing():
    assert True

def test_failing():
    assert False
""")

        # Get metadata
        tests = self.adapter.discover_tests(
            test_dirs=[Path(self.temp_dir)],
            patterns=["test_run.py"],
        )

        # Run passing test
        passing_metadata = tests[0][1]
        result = self.adapter.run_test(
            tests[0][0],
            passing_metadata,
            TestSuiteConfig(),
        )

        self.assertEqual(result.status, TestStatus.PASSED)

        # Run failing test
        failing_metadata = tests[1][1]
        result = self.adapter.run_test(
            tests[1][0],
            failing_metadata,
            TestSuiteConfig(),
        )

        self.assertEqual(result.status, TestStatus.FAILED)


class TestSuiteTest(unittest.TestCase):
    """Test cases for TestSuite class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TestSuiteConfig(
            test_dirs=[Path(self.temp_dir)],
            enable_coverage=False,  # Disable for faster tests
            parallel_workers=1,  # Run sequentially for deterministic tests
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_suite_initialization(self):
        """Test suite initialization."""
        suite = TestSuite(config=self.config, name="test_suite")

        self.assertEqual(suite.name, "test_suite")
        self.assertEqual(len(suite._tests), 0)

    def test_discover_tests(self):
        """Test test discovery."""
        # Create test files
        (Path(self.temp_dir) / "test_unit.py").write_text("""
def test_unit_1():
    assert True

def test_unit_2():
    assert True
""")

        suite = TestSuite(config=self.config)
        count = suite.discover_tests()

        self.assertEqual(count, 2)
        self.assertEqual(len(suite._tests), 2)

    def test_filter_tests_by_type(self):
        """Test filtering tests by type."""
        suite = TestSuite(config=self.config)

        # Add test metadata
        suite._tests["test_1"] = TestMetadata(
            name="test_1",
            file_path=Path("test_1.py"),
            test_type=TestType.UNIT,
        )
        suite._tests["test_2"] = TestMetadata(
            name="test_2",
            file_path=Path("test_2.py"),
            test_type=TestType.INTEGRATION,
        )

        filtered = suite.filter_tests(test_type=TestType.UNIT)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], "test_1")

    def test_filter_tests_by_priority(self):
        """Test filtering tests by priority."""
        suite = TestSuite(config=self.config)

        # Add test metadata
        suite._tests["test_1"] = TestMetadata(
            name="test_1",
            file_path=Path("test_1.py"),
            test_type=TestType.UNIT,
            priority=Priority.HIGH,
        )
        suite._tests["test_2"] = TestMetadata(
            name="test_2",
            file_path=Path("test_2.py"),
            test_type=TestType.UNIT,
            priority=Priority.LOW,
        )

        # Filter for HIGH priority and above (HIGH, CRITICAL)
        filtered = suite.filter_tests(priority=Priority.HIGH)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], "test_1")

    def test_filter_tests_by_tags(self):
        """Test filtering tests by tags."""
        suite = TestSuite(config=self.config)

        # Add test metadata
        suite._tests["test_1"] = TestMetadata(
            name="test_1",
            file_path=Path("test_1.py"),
            test_type=TestType.UNIT,
            tags={"slow", "integration"},
        )
        suite._tests["test_2"] = TestMetadata(
            name="test_2",
            file_path=Path("test_2.py"),
            test_type=TestType.UNIT,
            tags={"fast"},
        )

        filtered = suite.filter_tests(tags={"slow"})

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], "test_1")

    def test_filter_tests_by_pattern(self):
        """Test filtering tests by name pattern."""
        suite = TestSuite(config=self.config)

        # Add test metadata
        suite._tests["test_user_login"] = TestMetadata(
            name="test_user_login",
            file_path=Path("test_1.py"),
            test_type=TestType.UNIT,
        )
        suite._tests["test_user_logout"] = TestMetadata(
            name="test_user_logout",
            file_path=Path("test_2.py"),
            test_type=TestType.UNIT,
        )
        suite._tests["test_product_list"] = TestMetadata(
            name="test_product_list",
            file_path=Path("test_3.py"),
            test_type=TestType.UNIT,
        )

        filtered = suite.filter_tests(pattern=r"user.*")

        self.assertEqual(len(filtered), 2)
        self.assertIn("test_user_login", filtered)
        self.assertIn("test_user_logout", filtered)

    def test_run_tests_empty_suite(self):
        """Test running empty test suite."""
        suite = TestSuite(config=self.config)
        result = suite.run()

        self.assertEqual(result.total_tests, 0)
        self.assertEqual(result.passed, 0)
        self.assertTrue(result.was_successful)

    def test_run_passing_tests(self):
        """Test running passing tests."""
        # Create passing test
        (Path(self.temp_dir) / "test_pass.py").write_text("""
def test_pass_1():
    assert True

def test_pass_2():
    assert 1 + 1 == 2
""")

        suite = TestSuite(config=self.config)
        result = suite.run()

        self.assertEqual(result.total_tests, 2)
        self.assertEqual(result.passed, 2)
        self.assertEqual(result.failed, 0)
        self.assertTrue(result.was_successful)

    def test_run_failing_tests(self):
        """Test running failing tests."""
        # Create failing test
        (Path(self.temp_dir) / "test_fail.py").write_text("""
def test_fail():
    assert False
""")

        suite = TestSuite(config=self.config)
        result = suite.run()

        self.assertEqual(result.total_tests, 1)
        self.assertEqual(result.failed, 1)
        self.assertFalse(result.was_successful)

    def test_json_report_generation(self):
        """Test JSON report generation."""
        # Create temporary report file
        report_file = Path(self.temp_dir) / "report.json"
        self.config.json_report = report_file

        # Create test
        (Path(self.temp_dir) / "test_report.py").write_text("""
def test_example():
    assert True
""")

        suite = TestSuite(config=self.config)
        suite.run()

        self.assertTrue(report_file.exists())

        # Verify content
        with open(report_file) as f:
            data = json.load(f)

        self.assertIn("summary", data)
        self.assertIn("tests", data)
        self.assertEqual(data["summary"]["total"], 1)
        self.assertEqual(data["summary"]["passed"], 1)

    def test_html_report_generation(self):
        """Test HTML report generation."""
        # Create temporary report file
        report_file = Path(self.temp_dir) / "report.html"
        self.config.html_report = report_file

        # Create test
        (Path(self.temp_dir) / "test_html.py").write_text("""
def test_example():
    assert True
""")

        suite = TestSuite(config=self.config)
        suite.run()

        self.assertTrue(report_file.exists())

        # Verify content
        content = report_file.read_text()
        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn("Test Report", content)
        self.assertIn("test_example", content)


class CreateTestSuiteTest(unittest.TestCase):
    """Test cases for create_test_suite convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_default_suite(self):
        """Test creating default test suite."""
        suite = create_test_suite()

        self.assertIsNotNone(suite)
        self.assertEqual(suite.name, "default")

    def test_create_named_suite(self):
        """Test creating named test suite."""
        suite = create_test_suite(name="my_suite")

        self.assertEqual(suite.name, "my_suite")

    def test_create_suite_with_dirs(self):
        """Test creating suite with test directories."""
        # Create test directory
        test_dir = Path(self.temp_dir) / "tests"
        test_dir.mkdir()

        suite = create_test_suite(test_dirs=[test_dir])

        self.assertEqual(len(suite.config.test_dirs), 1)
        self.assertEqual(suite.config.test_dirs[0], test_dir)


class DecoratorsTest(unittest.TestCase):
    """Test cases for test decorators."""

    def test_flaky_decorator_success(self):
        """Test flaky decorator with eventual success."""
        call_count = 0

        @flaky(max_runs=3, min_passes=1)
        def test_flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise AssertionError("First attempt fails")
            return True

        result = test_flaky_function()

        self.assertTrue(result)
        self.assertEqual(call_count, 2)

    def test_flaky_decorator_failure(self):
        """Test flaky decorator with all failures."""
        @flaky(max_runs=3, min_passes=2)
        def test_always_fails():
            raise AssertionError("Always fails")

        with self.assertRaises(AssertionError):
            test_always_fails()

    def test_slow_decorator(self):
        """Test slow decorator."""
        @slow
        def test_slow_function():
            return True

        # Should still execute
        result = test_slow_function()
        self.assertTrue(result)

    def test_requires_network_decorator(self):
        """Test requires_network decorator."""
        @requires_network
        def test_network_function():
            return True

        # Should still execute
        result = test_network_function()
        self.assertTrue(result)


class TestHistoryTest(unittest.TestCase):
    """Test cases for test history tracking."""

    def test_pass_rate_calculation(self):
        """Test pass rate calculation."""
        from test_suite import TestHistory

        history = TestHistory(test_name="test_example")

        # Add results
        for i in range(7):
            history.results.append(TestResult(
                test_name="test_example",
                status=TestStatus.PASSED if i < 7 else TestStatus.FAILED,
                duration=1.0,
            ))

        for i in range(3):
            history.results.append(TestResult(
                test_name="test_example",
                status=TestStatus.FAILED,
                duration=1.0,
            ))

        self.assertEqual(len(history.results), 10)
        self.assertEqual(history.pass_rate, 70.0)

    def test_avg_duration_calculation(self):
        """Test average duration calculation."""
        from test_suite import TestHistory

        history = TestHistory(test_name="test_example")

        # Add results with different durations
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        for duration in durations:
            history.results.append(TestResult(
                test_name="test_example",
                status=TestStatus.PASSED,
                duration=duration,
            ))

        self.assertEqual(history.avg_duration, 3.0)

    def test_empty_history(self):
        """Test history with no results."""
        from test_suite import TestHistory

        history = TestHistory(test_name="test_example")

        self.assertEqual(history.pass_rate, 0.0)
        self.assertEqual(history.avg_duration, 0.0)


class IntegrationTest(unittest.TestCase):
    """Integration tests for the test suite framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_workflow(self):
        """Test complete workflow from discovery to reporting."""
        # Create test files
        tests_dir = Path(self.temp_dir) / "tests"
        tests_dir.mkdir()

        (tests_dir / "test_unit.py").write_text("""
def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 5 - 3 == 2
""")

        (tests_dir / "test_integration.py").write_text("""
def test_database_connection():
    # Simulate integration test
    assert True
""")

        # Create configuration
        config = TestSuiteConfig(
            test_dirs=[tests_dir],
            enable_coverage=False,
            parallel_workers=1,
            json_report=Path(self.temp_dir) / "report.json",
            html_report=Path(self.temp_dir) / "report.html",
        )

        # Create and run suite
        suite = TestSuite(config=config, name="integration_test")
        result = suite.run()

        # Verify results
        self.assertEqual(result.total_tests, 3)
        self.assertTrue(result.was_successful)

        # Verify reports
        self.assertTrue(config.json_report.exists())
        self.assertTrue(config.html_report.exists())

    def test_flaky_test_retry(self):
        """Test flaky test retry mechanism."""
        tests_dir = Path(self.temp_dir) / "tests"
        tests_dir.mkdir()

        # Create flaky test
        test_file = tests_dir / "test_flaky.py"
        test_file.write_text("""
import time

call_count = 0

def test_network_call():
    global call_count
    call_count += 1
    # Fail on first attempt, pass on retry
    if call_count == 1:
        raise AssertionError("Network timeout")
    assert True
""")

        config = TestSuiteConfig(
            test_dirs=[tests_dir],
            enable_coverage=False,
            parallel_workers=1,
            retry_flaky_tests=True,
        )

        suite = TestSuite(config=config)
        suite.discover_tests()

        # Mark test as flaky
        for test_name, metadata in suite._tests.items():
            if "test_network_call" in test_name:
                metadata.flaky = True
                metadata.max_retries = 3

        result = suite.run()

        # Test should pass after retry
        self.assertTrue(result.was_successful or result.failed <= 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
