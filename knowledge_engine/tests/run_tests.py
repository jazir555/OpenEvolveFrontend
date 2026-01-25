#!/usr/bin/env python
"""
Test Execution Script for Knowledge Engine

Following CLAUDE.md principles:
- All tests independent (can run in any order)
- Clear failure messages
- Setup/teardown properly handled
- Structured logging of test execution

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --contracts        # Run contract tests only
    python run_tests.py --integration      # Run integration tests only
    python run_tests.py --performance      # Run performance tests only
    python run_tests.py --coverage         # Run with coverage report
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Test runner for Knowledge Engine test suite."""

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.results = {
            "start_time": datetime.now().isoformat(),
            "tests_run": [],
            "summary": {}
        }

    def run_test_suite(
        self,
        test_pattern: str = "test_*.py",
        extra_args: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run test suite with given pattern.

        Args:
            test_pattern: Glob pattern for test files
            extra_args: Additional arguments to pass to pytest

        Returns:
            Test results dictionary
        """
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / test_pattern),
            "-v",
            "--tb=short",
            "--strict-markers",
            "-W", "ignore::DeprecationWarning"
        ]

        if extra_args:
            cmd.extend(extra_args)

        logger.info(json.dumps({
            "msg": "Running test suite",
            "command": " ".join(cmd),
            "level": "INFO"
        }))

        # Run tests
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.test_dir.parent)
        )

        # Parse results
        test_results = self._parse_pytest_output(result.stdout, result.stderr)
        test_results["return_code"] = result.returncode

        self.results["tests_run"].append({
            "pattern": test_pattern,
            "results": test_results
        })

        return test_results

    def run_with_coverage(self) -> Dict[str, Any]:
        """
        Run tests with coverage reporting.

        Returns:
            Test results with coverage information
        """
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "test_*.py"),
            "-v",
            "--cov=../knowledge_engine",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=json:coverage.json",
            "--tb=short"
        ]

        logger.info(json.dumps({
            "msg": "Running tests with coverage",
            "level": "INFO"
        }))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.test_dir.parent)
        )

        coverage_results = self._parse_coverage_output(result.stdout)
        coverage_results["return_code"] = result.returncode

        return coverage_results

    def run_specific_test(self, test_file: str) -> Dict[str, Any]:
        """
        Run a specific test file.

        Args:
            test_file: Name of test file to run

        Returns:
            Test results
        """
        test_path = self.test_dir / test_file

        if not test_path.exists():
            logger.error(json.dumps({
                "msg": "Test file not found",
                "file": test_file,
                "level": "ERROR"
            }))
            return {"error": f"Test file not found: {test_file}"}

        cmd = [
            sys.executable, "-m", "pytest",
            str(test_path),
            "-v",
            "--tb=short"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.test_dir.parent)
        )

        return self._parse_pytest_output(result.stdout, result.stderr)

    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """
        Parse pytest output to extract test results.

        Args:
            stdout: Standard output from pytest
            stderr: Standard error from pytest

        Returns:
            Parsed results dictionary
        """
        results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration": 0.0,
            "output": stdout,
            "errors_output": stderr
        }

        # Parse pytest summary
        lines = stdout.split('\n')
        for line in lines:
            if " passed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed":
                        results["passed"] = int(parts[i-1])
                    elif part == "failed":
                        results["failed"] = int(parts[i-1])
                    elif part == "skipped":
                        results["skipped"] = int(parts[i-1])
            elif "failed" in line and " error" in line:
                # Parse error count
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "error" or part == "errors":
                        results["errors"] = int(parts[i-1])
            elif "in" in line and ("second" in line or "seconds" in line):
                # Parse duration
                try:
                    duration_str = line.split("in ")[1].split(" second")[0]
                    results["duration"] = float(duration_str)
                except (IndexError, ValueError):
                    pass

        return results

    def _parse_coverage_output(self, stdout: str) -> Dict[str, Any]:
        """
        Parse coverage output.

        Args:
            stdout: Standard output from pytest-cov

        Returns:
            Coverage results
        """
        coverage_data = {
            "total_coverage": 0.0,
            "covered_lines": 0,
            "total_lines": 0,
            "missing_lines": 0,
            "files": {}
        }

        lines = stdout.split('\n')
        for line in lines:
            if "TOTAL" in line and "%" in line:
                # Parse total coverage
                try:
                    coverage_str = line.split()[0].replace('%', '')
                    coverage_data["total_coverage"] = float(coverage_str)
                except (IndexError, ValueError):
                    pass

        return coverage_data

    def generate_report(self) -> str:
        """
        Generate test execution report.

        Returns:
            Formatted report string
        """
        self.results["end_time"] = datetime.now().isoformat()

        report = []
        report.append("=" * 80)
        report.append("KNOWLEDGE ENGINE TEST EXECUTION REPORT")
        report.append("=" * 80)
        report.append(f"Start Time: {self.results['start_time']}")
        report.append(f"End Time: {self.results['end_time']}")
        report.append("")

        # Summary for each test suite
        for suite in self.results["tests_run"]:
            pattern = suite["pattern"]
            results = suite["results"]

            report.append(f"Test Suite: {pattern}")
            report.append("-" * 40)

            if "error" in results:
                report.append(f"ERROR: {results['error']}")
            else:
                report.append(f"Passed: {results['passed']}")
                report.append(f"Failed: {results['failed']}")
                report.append(f"Skipped: {results['skipped']}")
                report.append(f"Errors: {results['errors']}")
                report.append(f"Duration: {results['duration']:.2f}s")

                # Calculate success rate
                total = results['passed'] + results['failed']
                if total > 0:
                    success_rate = (results['passed'] / total) * 100
                    report.append(f"Success Rate: {success_rate:.1f}%")

                if results['failed'] > 0:
                    report.append("\nFailed Tests:")
                    # Extract failed test names from output
                    output_lines = results['output'].split('\n')
                    for line in output_lines:
                        if 'FAILED' in line:
                            report.append(f"  - {line.strip()}")

            report.append("")

        report.append("=" * 80)

        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Knowledge Engine test suite"
    )

    parser.add_argument(
        "--contracts",
        action="store_true",
        help="Run contract tests only"
    )

    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only"
    )

    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance tests only"
    )

    parser.add_argument(
        "--errors",
        action="store_true",
        help="Run error handling tests only"
    )

    parser.add_argument(
        "--quality",
        action="store_true",
        help="Run data quality tests only"
    )

    parser.add_argument(
        "--security",
        action="store_true",
        help="Run security tests only"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage report"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (skip slow tests)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output report to file"
    )

    args = parser.parse_args()

    # Determine test directory
    test_dir = Path(__file__).parent

    # Create test runner
    runner = TestRunner(test_dir)

    # Run tests based on arguments
    if args.contracts:
        logger.info("Running contract tests...")
        results = runner.run_specific_test("test_contracts.py")
    elif args.integration:
        logger.info("Running integration tests...")
        results = runner.run_specific_test("test_integration_e2e.py")
    elif args.performance:
        logger.info("Running performance tests...")
        results = runner.run_specific_test("test_performance.py")
    elif args.errors:
        logger.info("Running error handling tests...")
        results = runner.run_specific_test("test_errors.py")
    elif args.quality:
        logger.info("Running data quality tests...")
        results = runner.run_specific_test("test_quality.py")
    elif args.security:
        logger.info("Running security tests...")
        results = runner.run_specific_test("test_security.py")
    elif args.coverage:
        logger.info("Running tests with coverage...")
        results = runner.run_with_coverage()
    else:
        # Run all tests
        logger.info("Running all tests...")
        extra_args = []
        if args.quick:
            extra_args = ["-m", "not slow"]
        results = runner.run_test_suite(extra_args=extra_args)

    # Generate and display report
    report = runner.generate_report()

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {args.output}")
    else:
        print(report)

    # Exit with appropriate code
    if any("failed" in suite.get("results", {}) and suite["results"]["failed"] > 0
           for suite in runner.results["tests_run"]):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
