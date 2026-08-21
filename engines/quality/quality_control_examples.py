"""
Quality Control Module - Usage Examples

This file demonstrates how to use the quality_control module in various scenarios.
Per CLAUDE.md, these examples show the "Runtime Truth" of the system.

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""
from __future__ import annotations


from quality_control import (
    CodeQualityChecker,
    run_quality_checks,
    IssueSeverity,
    IssueType
)
import json


def example_1_basic_usage():
    """
    Example 1: Basic usage - Run all quality checks on a project.

    This is the simplest way to run quality checks using the run_quality_checks function.
    """
    print("Example 1: Basic Usage")
    print("-" * 70)

    result = run_quality_checks(
        project_root=".",
        config={'check_coverage': False}  # Disable coverage for faster execution
    )

    print(f"Quality Score: {result['quality_score']:.2%}")
    print(f"Total Issues: {result['total_issues']}")
    print(f"  Critical: {result['critical_issues']}")
    print(f"  High: {result['high_issues']}")
    print(f"  Medium: {result['medium_issues']}")
    print(f"  Low: {result['low_issues']}")
    print()


def example_2_custom_thresholds():
    """
    Example 2: Custom thresholds for quality checks.

    Adjust the sensitivity of quality checks by setting custom thresholds.
    """
    print("Example 2: Custom Thresholds")
    print("-" * 70)

    result = run_quality_checks(
        project_root=".",
        config={
            'max_cyclomatic_complexity': 15,  # Allow more complex functions
            'max_function_length': 100,  # Allow longer functions
            'max_nesting_depth': 5,  # Allow deeper nesting
            'min_coverage': 80.0,  # Require 80% test coverage
            'check_coverage': False
        }
    )

    print(f"Quality Score with custom thresholds: {result['quality_score']:.2%}")
    print()


def example_3_security_only():
    """
    Example 3: Run only security checks.

    Focus on security vulnerabilities without other quality checks.
    """
    print("Example 3: Security Checks Only")
    print("-" * 70)

    result = run_quality_checks(
        project_root=".",
        config={
            'check_code_smells': False,
            'check_complexity': False,
            'check_duplication': False,
            'check_coverage': False,
            'check_security': True
        }
    )

    print(f"Security Issues Found: {result['security_issues']}")
    print(f"  Critical: {result['critical_issues']}")
    print(f"  High: {result['high_issues']}")

    # Print details of security issues
    if result['security_issues'] > 0:
        print("\nSecurity Issue Details:")
        for issue in result['issues'][:5]:  # Show first 5 issues
            print(f"  - {issue['file_path']}:{issue['line_number']}")
            print(f"    {issue['message']}")
            print(f"    Suggestion: {issue['suggestion']}")
    print()


def example_4_specific_paths():
    """
    Example 4: Check specific directories or files.

    Run quality checks only on specific paths instead of the entire project.
    """
    print("Example 4: Specific Paths")
    print("-" * 70)

    result = run_quality_checks(
        project_root=".",
        paths=['src/', 'lib/'],  # Only check these directories
        config={'check_coverage': False}
    )

    print(f"Checked specific paths - Quality Score: {result['quality_score']:.2%}")
    print(f"Total Issues: {result['total_issues']}")
    print()


def example_5_detailed_report():
    """
    Example 5: Generate and save detailed quality report.

    Create a comprehensive report and save it to a JSON file.
    """
    print("Example 5: Detailed Report Generation")
    print("-" * 70)

    checker = CodeQualityChecker(
        project_root=".",
        config={'check_coverage': False}
    )

    # Run all checks
    report = checker.run_all_checks()

    # Print summary
    print(f"Quality Score: {report.metrics.quality_score:.2%}")
    print(f"Files Analyzed: {report.metrics.total_files}")
    print(f"Total Issues: {report.metrics.total_issues}")
    print(f"\nBreakdown:")
    print(f"  Code Smells: {report.metrics.code_smell_count}")
    print(f"  Security Issues: {report.metrics.security_count}")
    print(f"  Complexity Issues: {report.metrics.complexity_score}")
    print(f"  Duplication: {report.metrics.duplication_percent}")

    # Save report to file
    report.save_to_file("quality_report.json")
    print("\nReport saved to: quality_report.json")

    # Show sample issues
    if report.issues:
        print(f"\nSample Issues (showing first 3):")
        for issue in report.issues[:3]:
            print(f"\n  [{issue.severity.value.upper()}] {issue.issue_type.value}")
            print(f"  File: {issue.file_path}:{issue.line_number}")
            print(f"  Message: {issue.message}")
            print(f"  Rule: {issue.rule_id}")
            if issue.suggestion:
                print(f"  Suggestion: {issue.suggestion}")
    print()


def example_6_checker_instance():
    """
    Example 6: Using CodeQualityChecker class directly.

    Use the CodeQualityChecker class for more control over the checking process.
    """
    print("Example 6: Direct Checker Usage")
    print("-" * 70)

    # Initialize checker with custom configuration
    checker = CodeQualityChecker(
        project_root=".",
        config={
            'max_cyclomatic_complexity': 10,
            'max_function_length': 50,
            'check_coverage': False
        }
    )

    # Run individual check types
    print("Running code smell checks...")
    smell_issues = checker.check_code_smells()
    print(f"  Found {len(smell_issues)} code smell issues")

    print("Running security checks...")
    security_issues = checker.check_security_issues()
    print(f"  Found {len(security_issues)} security issues")

    print("Running complexity checks...")
    complexity_issues = checker.check_complexity()
    print(f"  Found {len(complexity_issues)} complexity issues")

    # Run all checks at once
    print("\nRunning all checks...")
    report = checker.run_all_checks()

    print(f"\nOverall Quality Score: {report.metrics.quality_score:.2%}")
    print(f"Total Issues: {len(report.issues)}")
    print()


def example_7_filter_issues():
    """
    Example 7: Filter and analyze issues by type or severity.

    Analyze specific types of issues or severity levels.
    """
    print("Example 7: Issue Filtering")
    print("-" * 70)

    checker = CodeQualityChecker(project_root=".", config={'check_coverage': False})
    report = checker.run_all_checks()

    # Filter by severity
    critical_issues = [i for i in report.issues if i.severity == IssueSeverity.CRITICAL]
    high_issues = [i for i in report.issues if i.severity == IssueSeverity.HIGH]

    print(f"Critical Issues: {len(critical_issues)}")
    for issue in critical_issues[:3]:
        print(f"  - {issue.file_path}:{issue.line_number} - {issue.message}")

    print(f"\nHigh Priority Issues: {len(high_issues)}")

    # Filter by type
    security_issues = [i for i in report.issues if i.issue_type == IssueType.SECURITY]
    complexity_issues = [i for i in report.issues if i.issue_type == IssueType.COMPLEXITY]

    print(f"\nSecurity Issues: {len(security_issues)}")
    print(f"Complexity Issues: {len(complexity_issues)}")
    print()


def example_8_ci_cd_integration():
    """
    Example 8: CI/CD Pipeline Integration.

    Example of how to integrate quality checks into a CI/CD pipeline.
    Returns exit code based on quality score.
    """
    print("Example 8: CI/CD Integration")
    print("-" * 70)

    # Run quality checks with production thresholds
    result = run_quality_checks(
        project_root=".",
        config={
            'max_cyclomatic_complexity': 10,
            'min_coverage': 80.0,
            'check_coverage': False  # Enable in real CI/CD
        }
    )

    # Define quality gate threshold
    QUALITY_GATE_THRESHOLD = 0.80

    print(f"Quality Score: {result['quality_score']:.2%}")
    print(f"Quality Gate Threshold: {QUALITY_GATE_THRESHOLD:.2%}")

    if result['quality_score'] >= QUALITY_GATE_THRESHOLD:
        print("[OK] Quality gate PASSED")
        exit_code = 0
    else:
        print("[FAIL] Quality gate FAILED")
        print(f"  Score {result['quality_score']:.2%} is below threshold {QUALITY_GATE_THRESHOLD:.2%}")
        exit_code = 1

    # Check for critical security issues
    if result['critical_issues'] > 0:
        print(f"[FAIL] Found {result['critical_issues']} critical security issues")
        exit_code = 1

    print(f"\nExit Code: {exit_code}")
    print()
    return exit_code


def example_9_incremental_checks():
    """
    Example 9: Incremental checks on changed files only.

    Run quality checks only on files that have changed since last commit.
    """
    print("Example 9: Incremental Checks")
    print("-" * 70)

    # Get list of changed files (simulated)
    # In real usage, use git to get changed files:
    # import subprocess
    # result = subprocess.run(['git', 'diff', '--name-only', 'HEAD'],
    #                          capture_output=True, text=True)
    # changed_files = result.stdout.strip().split('\n')

    changed_files = ['src/main.py', 'lib/utils.py']  # Simulated

    print(f"Checking {len(changed_files)} changed files...")

    result = run_quality_checks(
        project_root=".",
        paths=changed_files,
        config={'check_coverage': False}
    )

    print(f"Quality Score for changes: {result['quality_score']:.2%}")
    print(f"Issues in changed files: {result['total_issues']}")
    print()


def example_10_custom_exception_handling():
    """
    Example 10: Handle quality check exceptions gracefully.

    Demonstrate proper error handling for quality checks.
    """
    print("Example 10: Exception Handling")
    print("-" * 70)

    from quality_control import QualityCheckConfigError, QualityCheckExecutionError

    try:
        # Try to check non-existent directory
        result = run_quality_checks(project_root="/nonexistent/path")
    except QualityCheckConfigError as e:
        print(f"Configuration Error: {e}")
        print("Please check the project path and try again.")
    except QualityCheckExecutionError as e:
        print(f"Execution Error: {e}")
        print("Quality check failed during execution.")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    print()


def example_11_metrics_analysis():
    """
    Example 11: Analyze quality metrics over time.

    Track quality metrics to identify trends.
    """
    print("Example 11: Metrics Analysis")
    print("-" * 70)

    # Run checks and get metrics
    checker = CodeQualityChecker(project_root=".", config={'check_coverage': False})
    report = checker.run_all_checks()

    metrics = report.metrics

    print("Current Quality Metrics:")
    print(f"  Quality Score: {metrics.quality_score:.2%}")
    print(f"  Total Files: {metrics.total_files}")
    print(f"  Total Issues: {metrics.total_issues}")
    print(f"  Code Smells: {metrics.code_smell_count}")
    print(f"  Security Issues: {metrics.security_count}")
    print(f"  Complexity Issues: {metrics.complexity_score}")
    print(f"  Duplication: {metrics.duplication_percent}")

    # Calculate issue density
    if metrics.total_files > 0:
        issue_density = metrics.total_issues / metrics.total_files
        print(f"  Issue Density: {issue_density:.2f} issues per file")

    # Calculate security ratio
    if metrics.total_issues > 0:
        security_ratio = metrics.security_count / metrics.total_issues
        print(f"  Security Issue Ratio: {security_ratio:.1%}")

    print()


def example_12_language_specific_checks():
    """
    Example 12: Check specific programming languages.

    Run checks on specific file types only.
    """
    print("Example 12: Language-Specific Checks")
    print("-" * 70)

    # Check only Python files
    print("Checking Python files...")
    checker = CodeQualityChecker(project_root=".")

    # Discover only Python files
    python_files = list(Path(".").rglob("*.py"))
    python_paths = [str(f) for f in python_files if 'test' not in str(f)]

    print(f"Found {len(python_paths)} Python files")

    if python_paths:
        result = run_quality_checks(
            project_root=".",
            paths=python_paths[:10],  # Limit for demo
            config={'check_coverage': False}
        )

        print(f"Python Code Quality Score: {result['quality_score']:.2%}")
    print()


def main():
    """Run all examples."""
    import sys
    from pathlib import Path

    print("=" * 70)
    print("Quality Control Module - Usage Examples")
    print("=" * 70)
    print()

    # Check if we're in a valid project
    if not Path(".").exists():
        print("Error: Current directory does not exist")
        sys.exit(1)

    # Run examples
    try:
        example_1_basic_usage()
        # example_2_custom_thresholds()
        # example_3_security_only()
        # example_4_specific_paths()
        # example_5_detailed_report()
        # example_6_checker_instance()
        # example_7_filter_issues()
        # example_8_ci_cd_integration()
        # example_9_incremental_checks()
        # example_10_custom_exception_handling()
        # example_11_metrics_analysis()
        # example_12_language_specific_checks()

    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
