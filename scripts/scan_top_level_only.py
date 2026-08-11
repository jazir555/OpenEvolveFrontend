#!/usr/bin/env python3
"""
Focused Security Scanner - Top Level Only

Scans ONLY Python files in the top-level Frontend directory (not subdirectories).
This matches the original bug report scope.

Usage:
    python scan_top_level_only.py [--output-dir PATH]
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import re

def get_top_level_python_files(target_dir: Path) -> list:
    """Get only Python files in the top-level directory (no recursion)."""
    python_files = []

    # Only files directly in target_dir, not subdirectories
    for item in target_dir.iterdir():
        if item.is_file() and item.suffix == '.py':
            python_files.append(item)

    return sorted(python_files)

def run_bandit(files: list, output_dir: Path) -> dict:
    """Run bandit on specified files."""
    print(f"\n[*] Running Bandit security scanner on {len(files)} files...")

    # Prepare bandit command
    bandit_cmd = [
        sys.executable, '-m', 'bandit',
        '-f', 'json',
        '-o', str(output_dir / 'bandit_report.json'),
        '-r'  # Even though we're not using recursion, this flag is needed
    ]

    # Add only the top-level files
    bandit_cmd.extend([str(f) for f in files])

    print(f"[*] Command: {' '.join(bandit_cmd[:5])} ... ({len(files)} files)")

    try:
        result = subprocess.run(
            bandit_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
            # Load the report
            report_path = output_dir / 'bandit_report.json'
            if report_path.exists():
                with open(report_path, 'r') as f:
                    return json.load(f)
        else:
            print(f"[!] Bandit exited with code {result.returncode}")
            if result.stderr:
                print(f"[!] Error: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("[!] Bandit scan timed out after 5 minutes")
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"[!] Error running bandit: {e}")

    return {"results": [], "errors": []}

def analyze_with_custom_rules(files: list) -> dict:
    """Run custom analysis for issues bandit might miss."""
    print("\n[*] Running custom analysis...")

    findings = {
        'syntax_errors': [],
        'bare_except': [],
        'try_except_pass': [],
        'pickle_usage': [],
        'hardcoded_tmp': [],
        'missing_imports': []
    }

    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Check for syntax errors
            try:
                compile(content, str(filepath), 'exec')
            except SyntaxError as e:
                findings['syntax_errors'].append({
                    'file': str(filepath.name),
                    'line': e.lineno,
                    'error': str(e.msg)
                })

            # Analyze each line
            for i, line in enumerate(lines, 1):
                # Bare except
                if re.search(r'except:\s*$', line):
                    findings['bare_except'].append({
                        'file': str(filepath.name),
                        'line': i,
                        'code': line.strip()
                    })

                # Try/except/pass
                if re.search(r'except:\s*pass', line):
                    findings['try_except_pass'].append({
                        'file': str(filepath.name),
                        'line': i,
                        'code': line.strip()
                    })

                # Pickle usage
                if 'pickle' in line and ('import' in line or 'pickle.' in line):
                    findings['pickle_usage'].append({
                        'file': str(filepath.name),
                        'line': i,
                        'code': line.strip()
                    })

                # Hardcoded /tmp
                if re.search(r'["\']\/tmp\/', line):
                    findings['hardcoded_tmp'].append({
                        'file': str(filepath.name),
                        'line': i,
                        'code': line.strip()
                    })

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"[!] Error analyzing {filepath.name}: {e}")

    return findings

def generate_report(bandit_results: dict, custom_findings: dict, files: list, output_dir: Path):
    """Generate comprehensive security report."""

    report_lines = []
    report_lines.append("# OpenEvolve-BubbleLab Security Report")
    report_lines.append(f"# Top-Level Directory Only")
    report_lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append(f"**Files Scanned:** {len(files)}")
    report_lines.append(f"**Scan Scope:** Top-level directory ONLY (no subdirectories)")
    report_lines.append("")

    # Bandit results
    bandit_issues = bandit_results.get('results', [])
    bandit_errors = bandit_results.get('errors', [])

    report_lines.append("### Bandit Security Scanner Results")
    report_lines.append("")
    report_lines.append(f"- **Security Issues Found:** {len(bandit_issues)}")
    report_lines.append(f"- **Files with Errors:** {len(bandit_errors)}")
    report_lines.append("")

    # Group by severity
    high_severity = [i for i in bandit_issues if i.get('issue_severity') == 'HIGH']
    medium_severity = [i for i in bandit_issues if i.get('issue_severity') == 'MEDIUM']
    low_severity = [i for i in bandit_issues if i.get('issue_severity') == 'LOW']

    report_lines.append("**By Severity:**")
    report_lines.append(f"- HIGH: {len(high_severity)}")
    report_lines.append(f"- MEDIUM: {len(medium_severity)}")
    report_lines.append(f"- LOW: {len(low_severity)}")
    report_lines.append("")

    # Group by issue type
    issue_types = {}
    for issue in bandit_issues:
        issue_name = issue.get('issue_text', 'Unknown')
        issue_type = issue.get('test_id', 'Unknown')
        key = f"{issue_type} ({issue_name})"
        if key not in issue_types:
            issue_types[key] = []
        issue_types[key].append(issue)

    if issue_types:
        report_lines.append("**By Issue Type:**")
        for issue_type, issues in sorted(issue_types.items(), key=lambda x: len(x[1]), reverse=True):
            report_lines.append(f"- {issue_type}: {len(issues)}")
        report_lines.append("")

    # Custom findings
    report_lines.append("### Custom Analysis Results")
    report_lines.append("")
    report_lines.append(f"- **Syntax Errors:** {len(custom_findings['syntax_errors'])}")
    report_lines.append(f"- **Bare Except Clauses:** {len(custom_findings['bare_except'])}")
    report_lines.append(f"- **Try/Except/Pass Patterns:** {len(custom_findings['try_except_pass'])}")
    report_lines.append(f"- **Pickle Usage:** {len(custom_findings['pickle_usage'])}")
    report_lines.append(f"- **Hardcoded /tmp Paths:** {len(custom_findings['hardcoded_tmp'])}")
    report_lines.append("")

    # Top issues by file
    report_lines.append("## Files with Most Issues")
    report_lines.append("")

    issues_by_file = {}
    for issue in bandit_issues:
        filename = Path(issue.get('filename', '')).name
        if filename not in issues_by_file:
            issues_by_file[filename] = []
        issues_by_file[filename].append(issue)

    # Sort by issue count
    sorted_files = sorted(issues_by_file.items(), key=lambda x: len(x[1]), reverse=True)

    for filename, issues in sorted_files[:20]:
        report_lines.append(f"### {filename}")
        report_lines.append(f"**Issues:** {len(issues)}")
        report_lines.append("")

        # Group by severity for this file
        high = [i for i in issues if i.get('issue_severity') == 'HIGH']
        medium = [i for i in issues if i.get('issue_severity') == 'MEDIUM']
        low = [i for i in issues if i.get('issue_severity') == 'LOW']

        if high:
            report_lines.append(f"- HIGH: {len(high)}")
            for issue in high[:3]:  # Show first 3
                report_lines.append(f"  - Line {issue.get('line_number', '?')}: {issue.get('issue_text', 'Unknown')}")
            if len(high) > 3:
                report_lines.append(f"  - ... and {len(high) - 3} more")

        if medium:
            report_lines.append(f"- MEDIUM: {len(medium)}")
            for issue in medium[:3]:
                report_lines.append(f"  - Line {issue.get('line_number', '?')}: {issue.get('issue_text', 'Unknown')}")
            if len(medium) > 3:
                report_lines.append(f"  - ... and {len(medium) - 3} more")

        if low:
            report_lines.append(f"- LOW: {len(low)}")
        report_lines.append("")

    # Detailed findings - Syntax Errors
    if custom_findings['syntax_errors']:
        report_lines.append("## Syntax Errors (Critical)")
        report_lines.append("")
        report_lines.append("These files cannot be executed and must be fixed first:")
        report_lines.append("")
        for error in custom_findings['syntax_errors']:
            report_lines.append(f"### {error['file']}")
            report_lines.append(f"- **Line:** {error['line']}")
            report_lines.append(f"- **Error:** {error['error']}")
            report_lines.append("")

    # Detailed findings - Bare Except
    if custom_findings['bare_except']:
        report_lines.append("## Bare Except Clauses")
        report_lines.append("")
        report_lines.append("Generic exception handlers that catch everything:")
        report_lines.append("")
        for finding in custom_findings['bare_except'][:30]:
            report_lines.append(f"**{finding['file']}:{finding['line']}**")
            report_lines.append(f"```python")
            report_lines.append(finding['code'])
            report_lines.append(f"```")
            report_lines.append("")

        if len(custom_findings['bare_except']) > 30:
            report_lines.append(f"*... and {len(custom_findings['bare_except']) - 30} more*")
            report_lines.append("")

    # Detailed findings - Pickle
    if custom_findings['pickle_usage']:
        report_lines.append("## Pickle Usage (Security Risk)")
        report_lines.append("")
        report_lines.append("Insecure deserialization - should use JSON instead:")
        report_lines.append("")
        for finding in custom_findings['pickle_usage'][:30]:
            report_lines.append(f"**{finding['file']}:{finding['line']}**")
            report_lines.append(f"```python")
            report_lines.append(finding['code'])
            report_lines.append(f"```")
            report_lines.append("")

        if len(custom_findings['pickle_usage']) > 30:
            report_lines.append(f"*... and {len(custom_findings['pickle_usage']) - 30} more*")
            report_lines.append("")

    # Detailed findings - Hardcoded /tmp
    if custom_findings['hardcoded_tmp']:
        report_lines.append("## Hardcoded Temp Paths")
        report_lines.append("")
        report_lines.append("Predictable temp directories - should use tempfile module:")
        report_lines.append("")
        for finding in custom_findings['hardcoded_tmp'][:30]:
            report_lines.append(f"**{finding['file']}:{finding['line']}**")
            report_lines.append(f"```python")
            report_lines.append(finding['code'])
            report_lines.append(f"```")
            report_lines.append("")

        if len(custom_findings['hardcoded_tmp']) > 30:
            report_lines.append(f"*... and {len(custom_findings['hardcoded_tmp']) - 30} more*")
            report_lines.append("")

    # Fix recommendations
    report_lines.append("## Recommended Fixes")
    report_lines.append("")
    report_lines.append("### 1. Syntax Errors (Highest Priority)")
    report_lines.append("")
    report_lines.append("Fix syntax errors first - these files cannot be imported or executed.")
    report_lines.append("")
    report_lines.append("### 2. High Severity Security Issues")
    report_lines.append("")
    report_lines.append(f"Address {len(high_severity)} HIGH severity security issues.")
    report_lines.append("")
    report_lines.append("### 3. Bare Except Clauses")
    report_lines.append("")
    report_lines.append(f"Replace {len(custom_findings['bare_except'])} bare except clauses with specific exception types.")
    report_lines.append("")
    report_lines.append("```python")
    report_lines.append("# Before")
    report_lines.append("try:")
    report_lines.append("    risky_operation()")
    report_lines.append("except:")
    report_lines.append("    pass")
    report_lines.append("")
    report_lines.append("# After")
    report_lines.append("import logging")
    report_lines.append("logger = logging.getLogger(__name__)")
    report_lines.append("")
    report_lines.append("try:")
    report_lines.append("    risky_operation()")
    report_lines.append("except (ValueError, TypeError) as e:")
    report_lines.append("    logger.error(f\"Expected error: {e}\")")
    report_lines.append("except Exception as e:")
    report_lines.append("    logger.error(f\"Unexpected error: {e}\", exc_info=True)")
    report_lines.append("    raise")
    report_lines.append("```")
    report_lines.append("")

    report_lines.append("### 4. Pickle Usage")
    report_lines.append("")
    report_lines.append(f"Replace {len(custom_findings['pickle_usage'])} pickle usage with JSON.")
    report_lines.append("")
    report_lines.append("```python")
    report_lines.append("# Before (insecure)")
    report_lines.append("import pickle")
    report_lines.append("data = pickle.load(open('data.pkl', 'rb'))")
    report_lines.append("")
    report_lines.append("# After (secure)")
    report_lines.append("import json")
    report_lines.append("data = json.load(open('data.json', 'r'))")
    report_lines.append("```")
    report_lines.append("")

    report_lines.append("### 5. Hardcoded Temp Paths")
    report_lines.append("")
    report_lines.append(f"Replace {len(custom_findings['hardcoded_tmp'])} hardcoded /tmp paths with tempfile module.")
    report_lines.append("")
    report_lines.append("```python")
    report_lines.append("# Before (insecure)")
    report_lines.append("temp_dir = '/tmp/myapp'")
    report_lines.append("")
    report_lines.append("# After (secure)")
    report_lines.append("import tempfile")
    report_lines.append("temp_dir = tempfile.mkdtemp(prefix='myapp_')")
    report_lines.append("```")
    report_lines.append("")

    # Save report
    report_path = output_dir / f'SECURITY_REPORT_TOP_LEVEL_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    return report_path

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Scan top-level Python files for security issues')
    parser.add_argument('--target-dir', type=str, default='.', help='Target directory (default: current)')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for reports')

    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Top-Level Security Scanner")
    print("=" * 80)
    print(f"\nTarget Directory: {target_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Scope: Top-level Python files ONLY (no subdirectories)")
    print("=" * 80)

    # Get top-level Python files only
    files = get_top_level_python_files(target_dir)

    if not files:
        print("\n[!] No Python files found in top-level directory")
        return

    print(f"\n[*] Found {len(files)} Python files in top-level directory")
    print("\nFiles to scan:")
    for f in files:
        print(f"  - {f.name}")

    # Run bandit
    bandit_results = run_bandit(files, output_dir)

    # Run custom analysis
    custom_findings = analyze_with_custom_rules(files)

    # Generate report
    report_path = generate_report(bandit_results, custom_findings, files, output_dir)

    print("\n" + "=" * 80)
    print("[OK] Scan Complete")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  Files scanned: {len(files)}")
    print(f"  Security issues: {len(bandit_results.get('results', []))}")
    print(f"  Syntax errors: {len(custom_findings['syntax_errors'])}")
    print(f"  Bare except clauses: {len(custom_findings['bare_except'])}")
    print(f"  Pickle usage: {len(custom_findings['pickle_usage'])}")
    print(f"  Hardcoded /tmp paths: {len(custom_findings['hardcoded_tmp'])}")
    print(f"\nReport saved to: {report_path}")
    print(f"Bandit JSON saved to: {output_dir / 'bandit_report.json'}")
    print("\n[OK] Done!")

if __name__ == '__main__':
    main()
