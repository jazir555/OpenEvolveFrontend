"""
Crash Analyzer for Fuzzing Results

Analyzes crashes from fuzzing, deduplicates them, assesses severity,
and generates actionable reports.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
from .fuzzing import Vulnerability, VulnerabilitySeverity, FuzzResult

logger = logging.getLogger(__name__)


@dataclass
class CrashPattern:
    """Represents a recurring crash pattern"""
    pattern_id: str
    pattern_name: str
    frequency: int
    severity: VulnerabilitySeverity
    example_vulnerabilities: List[Vulnerability] = field(default_factory=list)
    suspected_root_cause: str = ""
    suggested_fix: str = ""


@dataclass
class CrashReport:
    """Comprehensive crash analysis report"""
    report_id: str
    generated_at: datetime
    total_crashes: int
    unique_crashes: int
    vulnerabilities_by_severity: Dict[VulnerabilitySeverity, List[Vulnerability]]
    crash_patterns: List[CrashPattern]
    recommendations: List[str]
    reproducible_crashes: int
    top_crash_types: Dict[str, int]


class CrashAnalyzer:
    """
    Analyzes fuzzing crashes to identify patterns and root causes.
    """

    def __init__(self):
        # Known crash patterns and their indicators
        self.crash_signatures = {
            'null_pointer': [
                'AttributeError: \'NoneType\'',
                'NullPointerException',
                'Cannot read property',
                'null is not an object',
            ],
            'buffer_overflow': [
                'IndexError',
                'BufferOverflowError',
                'stack overflow',
                'out of bounds',
            ],
            'type_confusion': [
                'TypeError',
                'incompatible type',
                'expected .* got .*',
            ],
            'resource_exhaustion': [
                'MemoryError',
                'OutOfMemoryError',
                'ResourceExhausted',
                'too many open files',
            ],
            'division_by_zero': [
                'ZeroDivisionError',
                'division by zero',
                'divide by zero',
            ],
            'format_string': [
                'format string',
                '%n',
                'printf',
            ],
            'injection': [
                'SQL injection',
                'command injection',
                'path traversal',
                'eval\\(',
                'exec\\(',
            ],
        }

    def analyze(self, fuzz_result: FuzzResult) -> CrashReport:
        """
        Analyze fuzzing results and generate comprehensive report.

        Args:
            fuzz_result: Results from fuzzing session

        Returns:
            CrashReport with analysis
        """
        logger.info("Analyzing fuzzing crashes...")

        # Deduplicate and categorize vulnerabilities
        vulnerabilities_by_severity = self._categorize_by_severity(
            fuzz_result.vulnerabilities
        )

        # Identify crash patterns
        crash_patterns = self._identify_patterns(fuzz_result.vulnerabilities)

        # Count reproducible crashes
        reproducible = sum(1 for v in fuzz_result.vulnerabilities if v.reproducible)

        # Get top crash types
        top_crash_types = self._get_crash_type_counts(fuzz_result.vulnerabilities)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            vulnerabilities_by_severity,
            crash_patterns,
            top_crash_types
        )

        report = CrashReport(
            report_id=f"crash_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow(),
            total_crashes=fuzz_result.crashes_found,
            unique_crashes=fuzz_result.unique_crashes,
            vulnerabilities_by_severity=vulnerabilities_by_severity,
            crash_patterns=crash_patterns,
            recommendations=recommendations,
            reproducible_crashes=reproducible,
            top_crash_types=top_crash_types,
        )

        logger.info(
            f"Analysis complete: {report.unique_crashes} unique vulnerabilities, "
            f"{len(crash_patterns)} patterns identified"
        )

        return report

    def _categorize_by_severity(
        self,
        vulnerabilities: List[Vulnerability]
    ) -> Dict[VulnerabilitySeverity, List[Vulnerability]]:
        """Categorize vulnerabilities by severity"""
        categorized = {
            VulnerabilitySeverity.CRITICAL: [],
            VulnerabilitySeverity.HIGH: [],
            VulnerabilitySeverity.MEDIUM: [],
            VulnerabilitySeverity.LOW: [],
            VulnerabilitySeverity.INFO: [],
        }

        for vuln in vulnerabilities:
            categorized[vuln.severity].append(vuln)

        return categorized

    def _identify_patterns(
        self,
        vulnerabilities: List[Vulnerability]
    ) -> List[CrashPattern]:
        """Identify recurring crash patterns"""
        pattern_counts: Dict[str, tuple[int, List[Vulnerability]]] = {}

        for vuln in vulnerabilities:
            # Match against known patterns
            for pattern_name, signatures in self.crash_signatures.items():
                for sig in signatures:
                    if self._matches_signature(vuln, sig):
                        if pattern_name not in pattern_counts:
                            pattern_counts[pattern_name] = (0, [])
                        count, vulns = pattern_counts[pattern_name]
                        pattern_counts[pattern_name] = (count + 1, vulns + [vuln])
                        break

        # Create CrashPattern objects
        patterns = []
        for pattern_name, (count, vulns) in pattern_counts.items():
            # Determine severity from most severe example
            severity = max(
                (v.severity for v in vulns),
                key=lambda s: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'].index(s.value)
            )

            pattern = CrashPattern(
                pattern_id=f"pattern_{pattern_name}",
                pattern_name=pattern_name.replace('_', ' ').title(),
                frequency=count,
                severity=severity,
                example_vulnerabilities=vulns[:3],  # Top 3 examples
                suspected_root_cause=self._infer_root_cause(pattern_name, vulns),
                suggested_fix=self._suggest_fix(pattern_name),
            )
            patterns.append(pattern)

        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)

        return patterns

    def _matches_signature(self, vuln: Vulnerability, signature: str) -> bool:
        """Check if vulnerability matches a crash signature"""
        # Check crash type
        if signature.lower() in vuln.crash_type.lower():
            return True

        # Check description
        if re.search(signature, vuln.description, re.IGNORECASE):
            return True

        # Check stack trace
        if vuln.stack_trace and re.search(signature, vuln.stack_trace, re.IGNORECASE):
            return True

        return False

    def _infer_root_cause(self, pattern_name: str, vulnerabilities: List[Vulnerability]) -> str:
        """Infer root cause from pattern and examples"""
        causes = {
            'null_pointer': "Dereferencing null/None values without proper validation",
            'buffer_overflow': "Accessing memory/array boundaries without bounds checking",
            'type_confusion': "Type confusion between different data types",
            'resource_exhaustion': "Unbounded resource consumption (memory, file handles, etc.)",
            'division_by_zero': "Division operation without checking for zero denominator",
            'format_string': "User input directly used in format strings",
            'injection': "User input not sanitized before use in sensitive operations",
        }
        return causes.get(pattern_name, "Unknown root cause")

    def _suggest_fix(self, pattern_name: str) -> str:
        """Suggest fix for a crash pattern"""
        fixes = {
            'null_pointer': "Add null checks before dereferencing: if value is not None",
            'buffer_overflow': "Add bounds checking: if 0 <= index < len(array)",
            'type_confusion': "Add type validation: isinstance(value, ExpectedType)",
            'resource_exhaustion': "Add resource limits and cleanup logic",
            'division_by_zero': "Check denominator: if denominator != 0",
            'format_string': "Use parameterized formatting: f\"{value}\" instead of format strings",
            'injection': "Sanitize/escape user input before use: html.escape(input)",
        }
        return fixes.get(pattern_name, "Add input validation and error handling")

    def _get_crash_type_counts(self, vulnerabilities: List[Vulnerability]) -> Dict[str, int]:
        """Count crashes by type"""
        counts = {}
        for vuln in vulnerabilities:
            crash_type = vuln.crash_type
            counts[crash_type] = counts.get(crash_type, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def _generate_recommendations(
        self,
        by_severity: Dict[VulnerabilitySeverity, List[Vulnerability]],
        patterns: List[CrashPattern],
        top_types: Dict[str, int]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Critical/high severity fixes
        critical_count = len(by_severity[VulnerabilitySeverity.CRITICAL])
        high_count = len(by_severity[VulnerabilitySeverity.HIGH])

        if critical_count > 0:
            recommendations.append(
                f"🚨 URGENT: Fix {critical_count} critical vulnerabilities before deployment"
            )

        if high_count > 0:
            recommendations.append(
                f"[WARN] HIGH PRIORITY: Address {high_count} high-severity issues"
            )

        # Pattern-based recommendations
        for pattern in patterns[:3]:
            if pattern.frequency >= 3:
                recommendations.append(
                    f"📊 Pattern '{pattern.pattern_name}' appears {pattern.frequency} times: "
                    f"{pattern.suggested_fix}"
                )

        # Input validation
        if any('injection' in p.pattern_name.lower() for p in patterns):
            recommendations.append(
                "🔒 SECURITY: Implement comprehensive input validation and sanitization"
            )

        # Error handling
        error_types = ['TypeError', 'ValueError', 'IndexError', 'KeyError']
        error_count = sum(top_types.get(t, 0) for t in error_types)
        if error_count > 0:
            recommendations.append(
                f"🛡️ Add robust error handling for {error_count} error cases"
            )

        # Testing recommendations
        if len(by_severity[VulnerabilitySeverity.MEDIUM]) > 5:
            recommendations.append(
                "🧪 Expand test coverage with fuzzing and edge case testing"
            )

        return recommendations

    def deduplicate_vulnerabilities(
        self,
        vulnerabilities: List[Vulnerability]
    ) -> List[Vulnerability]:
        """
        Deduplicate vulnerabilities based on similarity.

        Args:
            vulnerabilities: List to deduplicate

        Returns:
            Deduplicated list
        """
        unique: List[Vulnerability] = []
        seen_signatures: Set[str] = set()

        for vuln in vulnerabilities:
            # Create signature from crash type and first line of description
            signature = f"{vuln.crash_type}:{vuln.description.split(chr(10))[0][:100]}"

            if signature not in seen_signatures:
                unique.append(vuln)
                seen_signatures.add(signature)

        return unique


class CrashReporter:
    """
    Generates human-readable crash reports in various formats.
    """

    def __init__(self):
        self.analyzer = CrashAnalyzer()

    def generate_report(
        self,
        fuzz_result: FuzzResult,
        format: str = 'text'
    ) -> str:
        """
        Generate a crash report.

        Args:
            fuzz_result: Fuzzing results to analyze
            format: Output format ('text', 'markdown', 'json')

        Returns:
            Formatted report
        """
        analysis = self.analyzer.analyze(fuzz_result)

        if format == 'text':
            return self._format_text(analysis)
        elif format == 'markdown':
            return self._format_markdown(analysis)
        elif format == 'json':
            return self._format_json(analysis)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _format_text(self, report: CrashReport) -> str:
        """Format report as plain text"""
        lines = [
            "=" * 60,
            f"CRASH ANALYSIS REPORT - {report.report_id}",
            f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "=" * 60,
            "",
            f"SUMMARY",
            f"-------",
            f"Total crashes: {report.total_crashes}",
            f"Unique crashes: {report.unique_crashes}",
            f"Reproducible: {report.reproducible_crashes}",
            "",
            f"VULNERABILITIES BY SEVERITY",
            f"--------------------------",
        ]

        for severity, vulns in report.vulnerabilities_by_severity.items():
            if vulns:
                lines.append(f"{severity.value.upper()}: {len(vulns)}")

        lines.extend([
            "",
            f"TOP CRASH TYPES",
            f"---------------",
        ])

        for crash_type, count in list(report.top_crash_types.items())[:5]:
            lines.append(f"  {crash_type}: {count}")

        if report.crash_patterns:
            lines.extend([
                "",
                f"CRASH PATTERNS",
                f"--------------",
            ])

            for pattern in report.crash_patterns[:3]:
                lines.extend([
                    f"",
                    f"Pattern: {pattern.pattern_name} ({pattern.frequency} occurrences)",
                    f"Severity: {pattern.severity.value.upper()}",
                    f"Root Cause: {pattern.suspected_root_cause}",
                    f"Suggested Fix: {pattern.suggested_fix}",
                ])

        if report.recommendations:
            lines.extend([
                "",
                f"RECOMMENDATIONS",
                f"---------------",
            ])

            for rec in report.recommendations:
                lines.append(f"  {rec}")

        lines.append("")
        return "\n".join(lines)

    def _format_markdown(self, report: CrashReport) -> str:
        """Format report as Markdown"""
        lines = [
            f"# Crash Analysis Report: {report.report_id}",
            "",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Summary",
            "",
            f"- **Total crashes:** {report.total_crashes}",
            f"- **Unique crashes:** {report.unique_crashes}",
            f"- **Reproducible:** {report.reproducible_crashes}",
            "",
            "## Vulnerabilities by Severity",
            "",
        ]

        for severity, vulns in report.vulnerabilities_by_severity.items():
            if vulns:
                emoji = {
                    VulnerabilitySeverity.CRITICAL: "🚨",
                    VulnerabilitySeverity.HIGH: "[WARN]",
                    VulnerabilitySeverity.MEDIUM: "⚡",
                    VulnerabilitySeverity.LOW: "ℹ️",
                    VulnerabilitySeverity.INFO: "📝",
                }
                lines.append(f"- {emoji.get(severity, '*')} **{severity.value.upper()}:** {len(vulns)}")

        if report.top_crash_types:
            lines.extend([
                "",
                "## Top Crash Types",
                "",
            ])
            for crash_type, count in list(report.top_crash_types.items())[:5]:
                lines.append(f"{count}. `{crash_type}`")

        if report.crash_patterns:
            lines.extend([
                "",
                "## Crash Patterns",
                "",
            ])
            for pattern in report.crash_patterns[:3]:
                lines.extend([
                    f"### {pattern.pattern_name} ({pattern.frequency} occurrences)",
                    "",
                    f"- **Severity:** {pattern.severity.value.upper()}",
                    f"- **Root Cause:** {pattern.suspected_root_cause}",
                    f"- **Suggested Fix:** {pattern.suggested_fix}",
                    "",
                ])

        if report.recommendations:
            lines.extend([
                "",
                "## Recommendations",
                "",
            ])
            for rec in report.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)

    def _format_json(self, report: CrashReport) -> str:
        """Format report as JSON"""
        import json

        def serialize_vuln(v):
            return {
                'id': v.vulnerability_id,
                'severity': v.severity.value,
                'title': v.title,
                'description': v.description,
                'crash_type': v.crash_type,
                'input_data': str(v.input_data)[:200],
                'reproducible': v.reproducible,
            }

        return json.dumps({
            'report_id': report.report_id,
            'generated_at': report.generated_at.isoformat(),
            'summary': {
                'total_crashes': report.total_crashes,
                'unique_crashes': report.unique_crashes,
                'reproducible_crashes': report.reproducible_crashes,
            },
            'vulnerabilities_by_severity': {
                sev.value: [serialize_vuln(v) for v in vulns]
                for sev, vulns in report.vulnerabilities_by_severity.items()
            },
            'crash_patterns': [
                {
                    'name': p.pattern_name,
                    'frequency': p.frequency,
                    'severity': p.severity.value,
                    'root_cause': p.suspected_root_cause,
                    'suggested_fix': p.suggested_fix,
                }
                for p in report.crash_patterns
            ],
            'recommendations': report.recommendations,
        }, indent=2)


# Convenience functions
async def analyze_crashes(fuzz_result: FuzzResult) -> CrashReport:
    """Analyze fuzzing crashes and generate report"""
    analyzer = CrashAnalyzer()
    return analyzer.analyze(fuzz_result)


def generate_crash_report(fuzz_result: FuzzResult, format: str = 'text') -> str:
    """Generate formatted crash report"""
    reporter = CrashReporter()
    return reporter.generate_report(fuzz_result, format=format)
