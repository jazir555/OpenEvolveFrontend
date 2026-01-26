#!/usr/bin/env python3
"""
Technical Debt Analyzer for BubbleLab Bubbles
Analyzes code quality issues and generates refactoring recommendations
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DebtIssue:
    """Represents a single technical debt issue"""
    file: str
    line: int
    type: str
    severity: str  # 'high', 'medium', 'low'
    description: str
    suggestion: str
    code_snippet: str


@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    file_path: str
    lines: int
    functions: List[Dict[str, Any]]
    issues: List[DebtIssue]
    metrics: Dict[str, int]


class TechnicalDebtAnalyzer:
    """Analyzes TypeScript files for technical debt patterns"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.issues: List[DebtIssue] = []
        self.file_analyses: List[FileAnalysis] = []

        # Configuration thresholds
        self.LONG_METHOD_THRESHOLD = 50
        self.LONG_FUNCTION_THRESHOLD = 100
        self.COMPLEXITY_THRESHOLD = 10
        self.PARAMETER_COUNT_THRESHOLD = 7
        self.DUPLICATION_THRESHOLD = 5

    def analyze_files(self, pattern: str = "**/*.ts") -> List[FileAnalysis]:
        """Analyze all TypeScript files matching pattern"""
        files = list(self.base_path.glob(pattern))
        # Exclude test files
        files = [f for f in files if not any(x in f.name for x in ['test', 'spec'])]

        print(f"Analyzing {len(files)} files...")

        for file_path in files:
            analysis = self.analyze_file(file_path)
            if analysis:
                self.file_analyses.append(analysis)

        return self.file_analyses

    def analyze_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single file for technical debt"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

        lines = content.split('\n')
        issues = []
        functions = []
        metrics = defaultdict(int)

        # Track current function scope
        current_function = None
        function_lines = []

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Detect function/method definitions
            func_match = re.match(
                r'(?:export\s+)?(?:async\s+)?(?:function\s+)?(\w+)\s*\(',
                stripped
            ) or re.match(
                r'(?:export\s+)?(?:async\s+)?(\w+)\s*=\s*(?:async\s+)?\(',
                stripped
            )

            if func_match and not stripped.startswith('//'):
                # Save previous function
                if current_function and function_lines:
                    functions.append({
                        'name': current_function,
                        'start_line': current_function['start_line'],
                        'line_count': len(function_lines),
                        'complexity': self._calculate_complexity(function_lines)
                    })

                    # Check for long methods
                    if len(function_lines) > self.LONG_METHOD_THRESHOLD:
                        severity = 'high' if len(function_lines) > self.LONG_FUNCTION_THRESHOLD else 'medium'
                        issues.append(DebtIssue(
                            file=str(file_path.relative_to(self.base_path)),
                            line=current_function['start_line'],
                            type='long_method',
                            severity=severity,
                            description=f"Function '{current_function}' is {len(function_lines)} lines long",
                            suggestion=f"Extract logical sections into separate helper functions. Consider breaking down into smaller, focused functions.",
                            code_snippet=function_lines[0][:80] if function_lines else ''
                        ))
                        metrics['long_methods'] += 1

                # Start new function
                current_function = {
                    'name': func_match.group(1),
                    'start_line': line_num
                }
                function_lines = []
            elif current_function:
                function_lines.append(line)

            # Check for magic numbers (numbers >= 10 not in comments)
            numbers = re.findall(r'\b\d{2,}\b', stripped)
            for num in numbers:
                if not stripped.startswith('//') and 'const' not in stripped and 'type' not in stripped:
                    issues.append(DebtIssue(
                        file=str(file_path.relative_to(self.base_path)),
                        line=line_num,
                        type='magic_number',
                        severity='low',
                        description=f"Magic number: {num}",
                        suggestion=f"Extract to named constant (e.g., {num.upper()}_VALUE)",
                        code_snippet=stripped[:80]
                    ))
                    metrics['magic_numbers'] += 1

            # Check for complex conditionals
            logical_ops = len(re.findall(r'&&|\|\|', stripped))
            if logical_ops >= 3 and not stripped.startswith('//'):
                issues.append(DebtIssue(
                    file=str(file_path.relative_to(self.base_path)),
                    line=line_num,
                    type='complex_conditional',
                    severity='medium',
                    description=f"Complex conditional with {logical_ops} logical operators",
                    suggestion="Extract condition to named variable or function for better readability",
                    code_snippet=stripped[:80]
                ))
                metrics['complex_conditionals'] += 1

            # Check for deep nesting
            indent_level = len(line) - len(line.lstrip())
            if indent_level > 24 and any(kw in stripped for kw in ['if', 'else', 'for', 'while']):
                issues.append(DebtIssue(
                    file=str(file_path.relative_to(self.base_path)),
                    line=line_num,
                    type='deep_nesting',
                    severity='high',
                    description=f"Deep nesting ({indent_level // 2} levels)",
                    suggestion="Use early returns or extract to separate function (Guard Clause pattern)",
                    code_snippet=stripped[:80]
                ))
                metrics['deep_nesting'] += 1

            # Check for hardcoded URLs
            url_match = re.search(r'["\'](https?://[^"\']+)["\']', stripped)
            if url_match and 'const' not in stripped and '=' not in stripped:
                issues.append(DebtIssue(
                    file=str(file_path.relative_to(self.base_path)),
                    line=line_num,
                    type='hardcoded_url',
                    severity='medium',
                    description=f"Hardcoded URL: {url_match.group(1)[:50]}...",
                    suggestion="Extract to configuration constant",
                    code_snippet=stripped[:80]
                ))
                metrics['hardcoded_urls'] += 1

            # Check for poorly named variables
            var_match = re.search(r'\b(let|const)\s+(tmp|temp|data|item|obj|val|stuff)\b', stripped)
            if var_match:
                issues.append(DebtIssue(
                    file=str(file_path.relative_to(self.base_path)),
                    line=line_num,
                    type='poor_naming',
                    severity='low',
                    description=f"Unclear variable name: {var_match.group(2)}",
                    suggestion="Use more descriptive variable name that indicates purpose",
                    code_snippet=stripped[:80]
                ))
                metrics['poor_naming'] += 1

            # Check for TODO/FIXME comments
            if any(kw in stripped.upper() for kw in ['TODO', 'FIXME', 'HACK', 'XXX']):
                issues.append(DebtIssue(
                    file=str(file_path.relative_to(self.base_path)),
                    line=line_num,
                    type='technical_debt_marker',
                    severity='medium',
                    description=f"Technical debt marker in comment",
                    suggestion="Address the marked issue or create a ticket",
                    code_snippet=stripped[:80]
                ))
                metrics['debt_markers'] += 1

            # Check for console.log statements (should use proper logging)
            if 'console.log' in stripped and not stripped.startswith('//'):
                issues.append(DebtIssue(
                    file=str(file_path.relative_to(self.base_path)),
                    line=line_num,
                    type='console_log',
                    severity='low',
                    description="Console.log statement found",
                    suggestion="Use proper logging library (e.g., logger.info, logger.debug)",
                    code_snippet=stripped[:80]
                ))
                metrics['console_logs'] += 1

            # Check for any type usage
            if re.search(r':\s*any\b', stripped) and not stripped.startswith('//'):
                issues.append(DebtIssue(
                    file=str(file_path.relative_to(self.base_path)),
                    line=line_num,
                    type='any_type',
                    severity='medium',
                    description="Usage of 'any' type reduces type safety",
                    suggestion="Use specific type or unknown with type guards",
                    code_snippet=stripped[:80]
                ))
                metrics['any_types'] += 1

        # Save last function
        if current_function and function_lines:
            functions.append({
                'name': current_function['name'],
                'start_line': current_function['start_line'],
                'line_count': len(function_lines),
                'complexity': self._calculate_complexity(function_lines)
            })

        return FileAnalysis(
            file_path=str(file_path.relative_to(self.base_path)),
            lines=len(lines),
            functions=functions,
            issues=issues,
            metrics=dict(metrics)
        )

    def _calculate_complexity(self, function_lines: List[str]) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        for line in function_lines:
            stripped = line.strip()
            if not stripped.startswith('//'):
                complexity += len(re.findall(r'\bif\b|\belse\b|\bfor\b|\bwhile\b|\bcase\b|\bcatch\b', stripped))
                complexity += len(re.findall(r'&&|\|\|', stripped))
        return complexity

    def find_code_duplication(self) -> List[Dict[str, Any]]:
        """Find potential code duplication patterns"""
        duplication_patterns = []
        code_blocks = defaultdict(list)

        for analysis in self.file_analyses:
            try:
                with open(self.base_path / analysis.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                    # Extract code blocks (5+ consecutive lines)
                    for i in range(len(lines) - 5):
                        block = '\n'.join(lines[i:i+5])
                        # Normalize for comparison
                        normalized = re.sub(r'\s+', ' ', block).strip()
                        normalized = re.sub(r"'[^']*'", "'X'", normalized)  # Replace strings
                        normalized = re.sub(r'"[^"]*"', '"X"', normalized)
                        normalized = re.sub(r'\b\d+\b', 'N', normalized)  # Replace numbers

                        if len(normalized) > 50:  # Only significant blocks
                            code_blocks[normalized].append({
                                'file': analysis.file_path,
                                'start_line': i + 1,
                                'block': block
                            })
            except Exception as e:
                print(f"Error analyzing duplication in {analysis.file_path}: {e}")

        # Find duplicates
        for pattern, occurrences in code_blocks.items():
            if len(occurrences) >= self.DUPLICATION_THRESHOLD:
                files = list(set([occ['file'] for occ in occurrences]))
                if len(files) > 1:  # Cross-file duplication
                    duplication_patterns.append({
                        'pattern': pattern[:100] + '...',
                        'occurrences': len(occurrences),
                        'files': files,
                        'examples': occurrences[:3]
                    })

        return sorted(duplication_patterns, key=lambda x: x['occurrences'], reverse=True)

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive technical debt report"""
        total_issues = sum(len(a.issues) for a in self.file_analyses)

        # Count by severity
        severity_counts = Counter()
        type_counts = Counter()
        file_issue_counts = Counter()

        for analysis in self.file_analyses:
            for issue in analysis.issues:
                severity_counts[issue.severity] += 1
                type_counts[issue.type] += 1
                file_issue_counts[analysis.file_path] += 1

        # Find duplication
        duplications = self.find_code_duplication()

        return {
            'summary': {
                'total_files_analyzed': len(self.file_analyses),
                'total_issues': total_issues,
                'total_lines': sum(a.lines for a in self.file_analyses),
                'severity_breakdown': dict(severity_counts),
                'issue_types': dict(type_counts),
            },
            'top_files': [
                {
                    'file': file,
                    'issues': count,
                    'details': next((a for a in self.file_analyses if a.file_path == file), None)
                }
                for file, count in file_issue_counts.most_common(20)
            ],
            'code_duplication': duplications[:10],
            'detailed_issues': [asdict(issue) for analysis in self.file_analyses for issue in analysis.issues],
            'file_analyses': [
                {
                    'file': analysis.file_path,
                    'lines': analysis.lines,
                    'function_count': len(analysis.functions),
                    'issues': len(analysis.issues),
                    'metrics': analysis.metrics
                }
                for analysis in self.file_analyses
            ]
        }

    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save report to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {output_path}")

    def print_summary(self, report: Dict[str, Any]):
        """Print report summary to console"""
        print("\n" + "="*80)
        print("TECHNICAL DEBT ANALYSIS REPORT")
        print("="*80)

        summary = report['summary']
        print(f"\nFiles Analyzed: {summary['total_files_analyzed']}")
        print(f"Total Lines: {summary['total_lines']:,}")
        print(f"Total Issues: {summary['total_issues']:,}")

        print("\nSeverity Breakdown:")
        for severity in ['high', 'medium', 'low']:
            count = summary['severity_breakdown'].get(severity, 0)
            if count > 0:
                print(f"  {severity.upper()}: {count:,}")

        print("\nTop Issue Types:")
        for issue_type, count in sorted(dict(summary['issue_types']).items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {issue_type}: {count:,}")

        print("\n" + "-"*80)
        print("TOP 20 FILES BY ISSUE COUNT")
        print("-"*80)
        for idx, entry in enumerate(report['top_files'], 1):
            details = entry['details']
            if details:
                print(f"\n{idx}. {entry['file']}")
                print(f"   Issues: {entry['issues']:,} | Lines: {details.lines:,} | Functions: {len(details.functions)}")
                if details.metrics:
                    top_metric = max(details.metrics.items(), key=lambda x: x[1])
                    print(f"   Top Issue: {top_metric[0].replace('_', ' ').title()} ({top_metric[1]})")

        if report['code_duplication']:
            print("\n" + "-"*80)
            print("CODE DUPLICATION PATTERNS")
            print("-"*80)
            for idx, dup in enumerate(report['code_duplication'][:5], 1):
                print(f"\n{idx}. Pattern appears {dup['occurrences']} times in {len(dup['files'])} files:")
                print(f"   Files: {', '.join([Path(f).name for f in dup['files'][:3]])}")


def main():
    """Main execution"""
    base_path = Path(__file__).parent.parent / "BubbleLab" / "packages" / "bubble-core" / "src" / "bubbles"

    if not base_path.exists():
        print(f"Error: Path not found: {base_path}")
        return

    analyzer = TechnicalDebtAnalyzer(base_path)

    # Analyze all TypeScript files
    analyzer.analyze_files("**/*.ts")

    # Generate report
    report = analyzer.generate_report()

    # Print summary
    analyzer.print_summary(report)

    # Save detailed report
    output_path = Path(__file__).parent / "technical_debt_report.json"
    analyzer.save_report(report, str(output_path))

    # Generate human-readable markdown report
    md_report_path = Path(__file__).parent / "TECHNICAL_DEBT_REPORT.md"
    generate_markdown_report(report, str(md_report_path))
    print(f"Markdown report saved to: {md_report_path}")


def generate_markdown_report(report: Dict[str, Any], output_path: str):
    """Generate human-readable markdown report"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Technical Debt Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        summary = report['summary']
        f.write("## Summary\n\n")
        f.write(f"- **Files Analyzed:** {summary['total_files_analyzed']}\n")
        f.write(f"- **Total Lines:** {summary['total_lines']:,}\n")
        f.write(f"- **Total Issues:** {summary['total_issues']:,}\n\n")

        f.write("### Severity Breakdown\n\n")
        for severity in ['high', 'medium', 'low']:
            count = summary['severity_breakdown'].get(severity, 0)
            if count > 0:
                f.write(f"- **{severity.upper()}:** {count:,}\n")

        f.write("\n### Top Issue Categories\n\n")
        for issue_type, count in sorted(dict(summary['issue_types']).items(), key=lambda x: x[1], reverse=True)[:15]:
            f.write(f"- **{issue_type.replace('_', ' ').title()}:** {count:,}\n")

        f.write("\n## Top Files Requiring Attention\n\n")
        for idx, entry in enumerate(report['top_files'][:20], 1):
            details = entry['details']
            f.write(f"\n### {idx}. {entry['file']}\n\n")
            f.write(f"- **Issues:** {entry['issues']:,}\n")
            f.write(f"- **Lines:** {details.lines:,}\n")
            f.write(f"- **Functions:** {len(details.functions)}\n")

            if details.metrics:
                f.write("\n**Issue Breakdown:**\n")
                for metric, count in sorted(details.metrics.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        f.write(f"- {metric.replace('_', ' ').title()}: {count}\n")

        f.write("\n## Refactoring Recommendations\n\n")
        f.write("### High Priority\n\n")
        f.write("1. **Extract Long Methods:** Break down functions over 100 lines\n")
        f.write("2. **Reduce Deep Nesting:** Apply Guard Clause pattern\n")
        f.write("3. **Remove Code Duplication:** Extract common patterns to utilities\n")

        f.write("\n### Medium Priority\n\n")
        f.write("1. **Replace Magic Numbers:** Use named constants\n")
        f.write("2. **Extract Complex Conditionals:** Create descriptive variable names\n")
        f.write("3. **Remove Hardcoded URLs:** Move to configuration\n")

        f.write("\n### Low Priority\n\n")
        f.write("1. **Improve Naming:** Use descriptive variable names\n")
        f.write("2. **Remove Console.log:** Use proper logging\n")
        f.write("3. **Replace 'any' Types:** Use specific types\n")


if __name__ == "__main__":
    main()
