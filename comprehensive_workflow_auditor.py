#!/usr/bin/env python3
"""
Comprehensive Workflow Bubble Auditor and Tester
Audits ALL BubbleLab bubbles for security, code quality, and test coverage
"""

import os
import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime


@dataclass
class SecurityIssue:
    """Represents a security issue found in a bubble"""
    file_path: str
    line_number: int
    severity: str  # Critical, High, Medium, Low
    category: str  # env_validation, api_key, rate_limiting, etc.
    issue: str
    impact: str
    recommendation: str
    code_snippet: str = ""


@dataclass
class CodeQualityIssue:
    """Represents a code quality issue"""
    file_path: str
    line_number: int
    category: str
    issue: str
    recommendation: str


@dataclass
class BubbleTestStatus:
    """Test status for a bubble"""
    file_path: str
    bubble_type: str  # service, tool, workflow
    has_test: bool
    test_file_path: str = ""
    test_coverage: float = 0.0
    missing_tests: List[str] = field(default_factory=list)


@dataclass
class BubbleAuditReport:
    """Complete audit report for a bubble"""
    file_path: str
    bubble_type: str
    bubble_name: str
    security_issues: List[SecurityIssue] = field(default_factory=list)
    quality_issues: List[CodeQualityIssue] = field(default_factory=list)
    test_status: BubbleTestStatus = None
    lines_of_code: int = 0
    complexity_score: int = 0


class ComprehensiveWorkflowAuditor:
    """Audits all workflow bubbles comprehensively"""

    def __init__(self, bubblelab_root: str):
        self.bubblelab_root = Path(bubblelab_root)
        self.bubbles_dir = self.bubblelab_root / "packages" / "bubble-core" / "src" / "bubbles"

        self.reports: Dict[str, BubbleAuditReport] = {}
        self.all_security_issues: List[SecurityIssue] = []
        self.all_quality_issues: List[CodeQualityIssue] = []

        # Security patterns
        self.env_patterns = [
            r"process\.env\.\w+",
            r"import\.meta\.env\.\w+",
        ]

        self.api_key_patterns = [
            r"api_?key",
            r"apikey",
            r"API_?KEY",
            r"authorization",
            r"bearer",
            r"token",
        ]

        self.dangerous_patterns = [
            (r"eval\s*\(", "Use of eval() is dangerous"),
            (r"innerHTML\s*=", "innerHTML usage allows XSS"),
            (r"dangerouslySetInnerHTML", "dangerouslySetInnerHTML allows XSS"),
            (r"exec\s*\(", "Use of exec() is dangerous"),
            (r"\.exec\(\s*user", "Executing user input is dangerous"),
        ]

        # Quality patterns
        self.quality_patterns = [
            (r"catch\s*\(\s*\w+\s*\)\s*\{\s*\}", "Empty catch block"),
            (r"console\.(log|warn|error)", "Console logging in production"),
            (r"TODO|FIXME|HACK|XXX", "Todo/Fixme comment"),
            (r"any\s*[,\)]", "Use of 'any' type"),
        ]

    def find_all_bubbles(self) -> Dict[str, List[str]]:
        """Find all bubble files by type"""
        bubbles = {
            "service": [],
            "tool": [],
            "workflow": [],
        }

        if not self.bubbles_dir.exists():
            print(f"Warning: Bubbles directory not found at {self.bubbles_dir}")
            return bubbles

        for bubble_type in bubbles.keys():
            type_dir = self.bubbles_dir / f"{bubble_type}-bubble"
            if type_dir.exists():
                bubbles[bubble_type] = list(type_dir.glob("*.ts"))
                # Filter out test files
                bubbles[bubble_type] = [
                    f for f in bubbles[bubble_type]
                    if "test" not in f.name.lower() and "spec" not in f.name.lower()
                ]

        return bubbles

    def check_test_coverage(self, bubble_path: Path) -> BubbleTestStatus:
        """Check if bubble has test coverage"""
        test_path = bubble_path.parent / f"{bubble_path.stem}.test.ts"
        spec_path = bubble_path.parent / f"{bubble_path.stem}.spec.ts"

        has_test = test_path.exists() or spec_path.exists()
        test_file = str(test_path) if test_path.exists() else str(spec_path) if spec_path.exists() else ""

        # Determine bubble type
        if "service-bubble" in str(bubble_path):
            bubble_type = "service"
        elif "tool-bubble" in str(bubble_path):
            bubble_type = "tool"
        elif "workflow-bubble" in str(bubble_path):
            bubble_type = "workflow"
        else:
            bubble_type = "unknown"

        status = BubbleTestStatus(
            file_path=str(bubble_path),
            bubble_type=bubble_type,
            has_test=has_test,
            test_file_path=test_file,
        )

        if has_test:
            # Analyze test file for coverage
            try:
                with open(test_file, "r", encoding="utf-8") as f:
                    test_content = f.read()

                # Check for common test patterns
                required_tests = [
                    "Environment Validation",
                    "Authentication",
                    "Rate Limiting",
                    "Input Validation",
                    "Error Handling",
                ]

                missing = []
                for required in required_tests:
                    if required.lower() not in test_content.lower():
                        missing.append(required)

                status.missing_tests = missing
                status.test_coverage = len(required_tests) - len(missing) / len(required_tests) * 100

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"Error analyzing test file {test_file}: {e}")

        return status

    def audit_security(self, content: str, file_path: str) -> List[SecurityIssue]:
        """Audit file for security issues"""
        issues = []
        lines = content.split("\n")

        # Check for environment variable usage without validation
        env_usage = []
        for i, line in enumerate(lines, 1):
            for pattern in self.env_patterns:
                if re.search(pattern, line):
                    env_usage.append((i, line.strip()))

        # Check if env vars are validated
        has_env_validation = any(
            "validate" in line.lower() or "check" in line.lower()
            for line in lines
        )

        if env_usage and not has_env_validation:
            issues.append(SecurityIssue(
                file_path=file_path,
                line_number=env_usage[0][0],
                severity="High",
                category="env_validation",
                issue="Environment variables used without validation",
                impact="Application may crash or behave unexpectedly with missing/invalid env vars",
                recommendation="Add environment variable validation at startup",
                code_snippet=env_usage[0][1]
            ))

        # Check for API keys in code
        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("//") or line.strip().startswith("*"):
                continue

            # Check for hardcoded API keys
            if re.search(r'(api_?key|apikey)\s*[:=]\s*["\'][^"\']+["\']', line, re.IGNORECASE):
                # Check if it's obviously a placeholder
                if not re.search(r'(your_|<.*>|xxx|placeholder|example)', line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        file_path=file_path,
                        line_number=i,
                        severity="Critical",
                        category="api_key",
                        issue="Potential hardcoded API key detected",
                        impact="API keys exposed in source code",
                        recommendation="Move API keys to environment variables",
                        code_snippet=line.strip()
                    ))

        # Check for dangerous patterns
        for i, line in enumerate(lines, 1):
            for pattern, message in self.dangerous_patterns:
                if re.search(pattern, line):
                    issues.append(SecurityIssue(
                        file_path=file_path,
                        line_number=i,
                        severity="High",
                        category="code_injection",
                        issue=message,
                        impact="Code injection or XSS vulnerability",
                        recommendation="Remove or sanitize this pattern",
                        code_snippet=line.strip()
                    ))

        # Check for rate limiting
        if any(keyword in content.lower() for keyword in ["api", "fetch", "axios", "http"]):
            has_rate_limit = any(
                keyword in content.lower()
                for keyword in ["ratelimit", "rate-limit", "throttle", "limit"]
            )
            if not has_rate_limit:
                issues.append(SecurityIssue(
                    file_path=file_path,
                    line_number=1,
                    severity="Medium",
                    category="rate_limiting",
                    issue="No rate limiting detected for API calls",
                    impact="API abuse and potential quota exhaustion",
                    recommendation="Implement rate limiting for all API calls"
                ))

        # Check for input validation
        if "function" in content or "=>" in content:
            has_validation = any(
                keyword in content.lower()
                for keyword in ["validate", "sanitize", "escape", "zod", "yup", "joi"]
            )
            if not has_validation:
                issues.append(SecurityIssue(
                    file_path=file_path,
                    line_number=1,
                    severity="High",
                    category="input_validation",
                    issue="No input validation detected",
                    impact="Vulnerable to injection attacks",
                    recommendation="Add input validation for all user inputs"
                ))

        # Check for error handling
        functions = content.count("function") + content.count("=>")
        try_catches = content.count("try")
        if functions > 0 and try_catches == 0:
            issues.append(SecurityIssue(
                file_path=file_path,
                line_number=1,
                severity="Medium",
                category="error_handling",
                issue="No error handling detected",
                impact="Errors may expose sensitive information",
                recommendation="Add try-catch blocks and proper error handling"
            ))

        # Check for structured logging
        has_console_log = "console.log" in content
        has_structured_logging = any(
            keyword in content.lower()
            for keyword in ["logger.", "winston", "pino", "log4js", "structured"]
        )
        if has_console_log and not has_structured_logging:
            issues.append(SecurityIssue(
                file_path=file_path,
                line_number=1,
                severity="Low",
                category="logging",
                issue="Using console.log instead of structured logging",
                impact="Poor observability and potential information leakage",
                recommendation="Use structured logging with correlation IDs"
            ))

        # Check for timeout handling
        if any(keyword in content.lower() for keyword in ["fetch", "axios", "http"]):
            has_timeout = any(
                keyword in content.lower()
                for keyword in ["timeout", "abortcontroller", "abort"]
            )
            if not has_timeout:
                issues.append(SecurityIssue(
                    file_path=file_path,
                    line_number=1,
                    severity="High",
                    category="timeout",
                    issue="No timeout configured for network requests",
                    impact="Application may hang indefinitely",
                    recommendation="Add timeouts to all network requests"
                ))

        return issues

    def audit_quality(self, content: str, file_path: str) -> List[CodeQualityIssue]:
        """Audit file for code quality issues"""
        issues = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            for pattern, message in self.quality_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeQualityIssue(
                        file_path=file_path,
                        line_number=i,
                        category="code_quality",
                        issue=message,
                        recommendation="Review and improve code quality"
                    ))

        # Check for proper error messages
        if "throw new Error" in content:
            throw_lines = [(i, line) for i, line in enumerate(lines, 1) if "throw new Error" in line]
            for line_num, line in throw_lines:
                if not re.search(r'throw new Error\s*\(\s*["\']', line):
                    issues.append(CodeQualityIssue(
                        file_path=file_path,
                        line_number=line_num,
                        category="error_handling",
                        issue="Error message may expose sensitive information",
                        recommendation="Use sanitized error messages"
                    ))

        # Check for resource cleanup
        if any(keyword in content.lower() for keyword in ["connection", "stream", "file"]):
            has_cleanup = any(
                keyword in content.lower()
                for keyword in ["close()", "disconnect()", "finally", "cleanup"]
            )
            if not has_cleanup:
                issues.append(CodeQualityIssue(
                    file_path=file_path,
                    line_number=1,
                    category="resource_management",
                    issue="No resource cleanup detected",
                    recommendation="Ensure proper resource cleanup in finally blocks"
                ))

        return issues

    def calculate_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity

        # Count decision points
        complexity += content.count("if")
        complexity += content.count("else")
        complexity += content.count("elif")
        complexity += content.count("for")
        complexity += content.count("while")
        complexity += content.count("case")
        complexity += content.count("catch")
        complexity += content.count("&&")
        complexity += content.count("||")

        return complexity

    def audit_bubble(self, bubble_path: Path) -> BubbleAuditReport:
        """Perform comprehensive audit on a bubble"""
        print(f"Auditing: {bubble_path.name}")

        try:
            with open(bubble_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"Error reading {bubble_path}: {e}")
            return None

        # Determine bubble type
        if "service-bubble" in str(bubble_path):
            bubble_type = "service"
        elif "tool-bubble" in str(bubble_path):
            bubble_type = "tool"
        elif "workflow-bubble" in str(bubble_path):
            bubble_type = "workflow"
        else:
            bubble_type = "unknown"

        # Audit security
        security_issues = self.audit_security(content, str(bubble_path))
        self.all_security_issues.extend(security_issues)

        # Audit quality
        quality_issues = self.audit_quality(content, str(bubble_path))
        self.all_quality_issues.extend(quality_issues)

        # Check test coverage
        test_status = self.check_test_coverage(bubble_path)

        # Calculate metrics
        lines_of_code = len([line for line in content.split("\n") if line.strip()])
        complexity = self.calculate_complexity(content)

        report = BubbleAuditReport(
            file_path=str(bubble_path),
            bubble_type=bubble_type,
            bubble_name=bubble_path.stem,
            security_issues=security_issues,
            quality_issues=quality_issues,
            test_status=test_status,
            lines_of_code=lines_of_code,
            complexity_score=complexity,
        )

        self.reports[str(bubble_path)] = report
        return report

    def audit_all_bubbles(self):
        """Audit all bubbles"""
        bubbles = self.find_all_bubbles()

        print(f"\nFound {len(bubbles['service'])} service bubbles")
        print(f"Found {len(bubbles['tool'])} tool bubbles")
        print(f"Found {len(bubbles['workflow'])} workflow bubbles")
        print(f"Total: {sum(len(v) for v in bubbles.values())} bubbles\n")

        # Audit in priority order
        print("=" * 80)
        print("AUDITING WORKFLOW BUBBLES (Highest Priority)")
        print("=" * 80)
        for bubble in sorted(bubbles["workflow"]):
            self.audit_bubble(bubble)

        print("\n" + "=" * 80)
        print("AUDITING SERVICE BUBBLES")
        print("=" * 80)
        for bubble in sorted(bubbles["service"]):
            self.audit_bubble(bubble)

        print("\n" + "=" * 80)
        print("AUDITING TOOL BUBBLES")
        print("=" * 80)
        for bubble in sorted(bubbles["tool"]):
            self.audit_bubble(bubble)

    def generate_reports(self):
        """Generate all reports"""
        self.generate_bug_report()
        self.generate_test_status_report()
        self.generate_summary_report()

    def generate_bug_report(self):
        """Generate comprehensive bug report"""
        report_path = Path("ALL_WORKFLOW_BUGS.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# ALL WORKFLOW BUBBLES - COMPREHENSIVE BUG REPORT\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            # Security Issues
            f.write("## SECURITY ISSUES\n\n")
            f.write(f"Total Security Issues: {len(self.all_security_issues)}\n\n")

            # Group by severity
            by_severity = defaultdict(list)
            for issue in self.all_security_issues:
                by_severity[issue.severity].append(issue)

            for severity in ["Critical", "High", "Medium", "Low"]:
                if severity in by_severity:
                    f.write(f"### {severity} Severity ({len(by_severity[severity])} issues)\n\n")
                    for issue in sorted(by_severity[severity], key=lambda x: x.file_path):
                        f.write(f"#### {issue.category}\n\n")
                        f.write(f"**File:** `{issue.file_path}:{issue.line_number}`\n\n")
                        f.write(f"**Issue:** {issue.issue}\n\n")
                        f.write(f"**Impact:** {issue.impact}\n\n")
                        f.write(f"**Recommendation:** {issue.recommendation}\n\n")
                        if issue.code_snippet:
                            f.write(f"**Code:**\n```\n{issue.code_snippet}\n```\n\n")
                        f.write("---\n\n")

            # Code Quality Issues
            f.write("## CODE QUALITY ISSUES\n\n")
            f.write(f"Total Quality Issues: {len(self.all_quality_issues)}\n\n")

            by_category = defaultdict(list)
            for issue in self.all_quality_issues:
                by_category[issue.category].append(issue)

            for category in sorted(by_category.keys()):
                f.write(f"### {category} ({len(by_category[category])} issues)\n\n")
                for issue in sorted(by_category[category], key=lambda x: x.file_path):
                    f.write(f"**File:** `{issue.file_path}:{issue.line_number}`\n\n")
                    f.write(f"**Issue:** {issue.issue}\n\n")
                    f.write(f"**Recommendation:** {issue.recommendation}\n\n")
                    f.write("---\n\n")

        print(f"Generated bug report: {report_path}")

    def generate_test_status_report(self):
        """Generate test status report"""
        report_path = Path("WORKFLOW_TEST_STATUS.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# WORKFLOW TEST STATUS REPORT\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            # Group by type
            by_type = defaultdict(list)
            for report in self.reports.values():
                by_type[report.bubble_type].append(report)

            for bubble_type in ["workflow", "service", "tool"]:
                if bubble_type not in by_type:
                    continue

                f.write(f"## {bubble_type.upper()} BUBBLES\n\n")

                reports = sorted(by_type[bubble_type], key=lambda x: x.bubble_name)

                with_tests = sum(1 for r in reports if r.test_status.has_test)
                without_tests = len(reports) - with_tests

                f.write(f"Total: {len(reports)}\n")
                f.write(f"With Tests: {with_tests} ({with_tests/len(reports)*100:.1f}%)\n")
                f.write(f"Without Tests: {without_tests} ({without_tests/len(reports)*100:.1f}%)\n\n")

                f.write("| Bubble | Has Test | Test File | Coverage | Missing Tests |\n")
                f.write("|--------|----------|-----------|----------|---------------|\n")

                for report in reports:
                    status = report.test_status
                    has_test = "[OK]" if status.has_test else "[FAIL]"
                    test_file = status.test_file_path.split("/")[-1] if status.test_file_path else "N/A"
                    coverage = f"{status.test_coverage:.1f}%" if status.has_test else "N/A"
                    missing = ", ".join(status.missing_tests) if status.missing_tests else "-"

                    f.write(f"| {report.bubble_name} | {has_test} | {test_file} | {coverage} | {missing} |\n")

                f.write("\n")

        print(f"Generated test status report: {report_path}")

    def generate_summary_report(self):
        """Generate summary report with statistics"""
        report_path = Path("WORKFLOW_TEST_SUMMARY.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# WORKFLOW TEST SUMMARY - STATISTICS & ANALYSIS\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            # Overall statistics
            f.write("## OVERALL STATISTICS\n\n")
            f.write(f"Total Bubbles Audited: {len(self.reports)}\n\n")

            by_type = defaultdict(list)
            for report in self.reports.values():
                by_type[report.bubble_type].append(report)

            for bubble_type in ["workflow", "service", "tool"]:
                if bubble_type in by_type:
                    reports = by_type[bubble_type]
                    with_tests = sum(1 for r in reports if r.test_status.has_test)
                    total_issues = sum(len(r.security_issues) + len(r.quality_issues) for r in reports)
                    avg_complexity = sum(r.complexity_score for r in reports) / len(reports)

                    f.write(f"### {bubble_type.upper()} BUBBLES\n\n")
                    f.write(f"- Total: {len(reports)}\n")
                    f.write(f"- With Tests: {with_tests} ({with_tests/len(reports)*100:.1f}%)\n")
                    f.write(f"- Total Issues: {total_issues}\n")
                    f.write(f"- Avg Complexity: {avg_complexity:.1f}\n\n")

            # Security statistics
            f.write("## SECURITY ISSUES SUMMARY\n\n")
            f.write(f"Total Security Issues: {len(self.all_security_issues)}\n\n")

            by_severity = defaultdict(int)
            by_category = defaultdict(int)

            for issue in self.all_security_issues:
                by_severity[issue.severity] += 1
                by_category[issue.category] += 1

            f.write("### By Severity\n\n")
            for severity in ["Critical", "High", "Medium", "Low"]:
                count = by_severity.get(severity, 0)
                if count > 0:
                    f.write(f"- {severity}: {count}\n")

            f.write("\n### By Category\n\n")
            for category, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {category}: {count}\n")

            # Quality statistics
            f.write("\n## CODE QUALITY ISSUES SUMMARY\n\n")
            f.write(f"Total Quality Issues: {len(self.all_quality_issues)}\n\n")

            by_category = defaultdict(int)
            for issue in self.all_quality_issues:
                by_category[issue.category] += 1

            for category, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {category}: {count}\n")

            # Top problematic files
            f.write("\n## TOP PROBLEMATIC FILES\n\n")
            sorted_reports = sorted(
                self.reports.values(),
                key=lambda r: len(r.security_issues) + len(r.quality_issues),
                reverse=True
            )[:10]

            f.write("| File | Type | Security Issues | Quality Issues | Total |\n")
            f.write("|------|------|-----------------|----------------|-------|\n")

            for report in sorted_reports:
                total = len(report.security_issues) + len(report.quality_issues)
                f.write(f"| {report.bubble_name} | {report.bubble_type} | {len(report.security_issues)} | {len(report.quality_issues)} | {total} |\n")

            # Recommendations
            f.write("\n## RECOMMENDATIONS\n\n")

            critical_count = by_severity.get("Critical", 0)
            high_count = by_severity.get("High", 0)

            if critical_count > 0:
                f.write(f"### URGENT: {critical_count} Critical Issues\n\n")
                f.write("Address critical security issues immediately:\n")
                f.write("- Hardcoded API keys\n")
                f.write("- Code injection vulnerabilities\n")
                f.write("- Missing authentication\n\n")

            if high_count > 0:
                f.write(f"### HIGH PRIORITY: {high_count} High Issues\n\n")
                f.write("- Missing input validation\n")
                f.write("- Missing timeout handling\n")
                f.write("- Missing rate limiting\n\n")

            without_tests = sum(1 for r in self.reports.values() if not r.test_status.has_test)
            if without_tests > 0:
                f.write(f"### TESTING: {without_tests} Bubbles Without Tests\n\n")
                f.write("Create comprehensive test suites for all bubbles:\n")
                f.write("- Environment validation tests\n")
                f.write("- Authentication tests\n")
                f.write("- Rate limiting tests\n")
                f.write("- Input validation tests\n")
                f.write("- Error handling tests\n")
                f.write("- Integration tests\n\n")

        print(f"Generated summary report: {report_path}")


def main():
    """Main entry point"""
    print("=" * 80)
    print("COMPREHENSIVE WORKFLOW BUBBLE AUDITOR")
    print("=" * 80)
    print()

    # Initialize auditor
    bubblelab_root = "./BubbleLab"
    auditor = ComprehensiveWorkflowAuditor(bubblelab_root)

    # Audit all bubbles
    auditor.audit_all_bubbles()

    # Generate reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    auditor.generate_reports()

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print("\nGenerated reports:")
    print("  - ALL_WORKFLOW_BUGS.md (All bugs and issues)")
    print("  - WORKFLOW_TEST_STATUS.md (Test coverage status)")
    print("  - WORKFLOW_TEST_SUMMARY.md (Statistics and summary)")


if __name__ == "__main__":
    main()
