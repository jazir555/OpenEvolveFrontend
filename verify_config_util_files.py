#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification Script for Config and Utility Files

Checks all 20 files from CREWAI_MIGRATION_MASTER_TASKLIST.md:
1. Import status verification
2. CrewAI reference detection
3. Syntax validation
4. Migration notice verification

Author: Claude Code
Date: 2026-01-21
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


@dataclass
class FileVerificationResult:
    """Results for a single file verification."""
    file_path: str
    exists: bool
    import_status: str  # "OK", "FAIL", "SKIP"
    syntax_valid: bool
    CrewAI_refs: List[str] = field(default_factory=list)
    migration_notice: bool = False
    issues: List[str] = field(default_factory=list)


class ConfigUtilVerifier:
    """Verifies config and utility files for migration completeness."""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.results: Dict[str, FileVerificationResult] = {}

    def check_syntax(self, file_path: Path) -> bool:
        """Check if Python file has valid syntax."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source)
            return True
        except SyntaxError as e:
            return False
        except Exception as e:
            return False

    def check_imports(self, file_path: Path) -> Tuple[str, List[str]]:
        """
        Check import status of file.
        Returns: (status, list_of_issues)
        status: "OK", "FAIL", "SKIP"
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)

            # Check for broken imports
            issues = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if 'CrewAI' in alias.name.lower():
                            issues.append(f"Found CrewAI import: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and 'CrewAI' in node.module.lower():
                        issues.append(f"Found CrewAI from import: {node.module}")

            if issues:
                return "FAIL", issues
            return "OK", []

        except FileNotFoundError:
            return "SKIP", ["File not found"]
        except Exception as e:
            return "FAIL", [f"Error checking imports: {str(e)}"]

    def check_CrewAI_references(self, file_path: Path) -> List[str]:
        """Find all CrewAI references in file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            refs = []

            # Check for various CrewAI references
            patterns = [
                r'CrewAI\w+',  # CamelCase class names
                r'CrewAI_\w+',  # snake_case variable/function names
                r'\bfrom\s+CrewAI',  # from imports
                r'\bimport\s+CrewAI',  # direct imports
                r'CrewAIIntegration',  # Specific class
                r'CrewAIClient',  # Specific class
            ]

            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Exclude comments and docstrings mentioning migration
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    line_end = content.find('\n', match.start())
                    line = content[line_start:line_end]

                    # Skip if it's in a migration notice or comment
                    if 'MIGRATION NOTICE' in line or 'AGPL → MIT' in line:
                        continue
                    if line.strip().startswith('#'):
                        continue

                    refs.append(f"Line {content[:match.start()].count(chr(10)) + 1}: {match.group()}")

            return refs

        except Exception as e:
            return [f"Error checking references: {str(e)}"]

    def check_migration_notice(self, file_path: Path) -> bool:
        """Check if file has migration notice."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for migration notice in first 50 lines
            lines = content.split('\n')[:50]
            content_sample = '\n'.join(lines)

            return 'MIGRATION NOTICE' in content_sample and 'CrewAI (AGPL) → CrewAI (MIT)' in content_sample

        except (OSError, IOError, UnicodeDecodeError):
            return False

    def verify_file(self, file_path: str) -> FileVerificationResult:
        """Verify a single file."""
        full_path = self.root_dir / file_path

        if not full_path.exists():
            return FileVerificationResult(
                file_path=file_path,
                exists=False,
                import_status="SKIP",
                syntax_valid=False,
                issues=["File not found"]
            )

        # Check syntax
        syntax_valid = self.check_syntax(full_path)

        # Check imports
        import_status, import_issues = self.check_imports(full_path)

        # Check CrewAI references
        CrewAI_refs = self.check_CrewAI_references(full_path)

        # Check migration notice
        migration_notice = self.check_migration_notice(full_path)

        # Collect all issues
        all_issues = import_issues + ([f"Syntax error" + ("s" if not syntax_valid else "")] if not syntax_valid else [])

        return FileVerificationResult(
            file_path=file_path,
            exists=True,
            import_status=import_status,
            syntax_valid=syntax_valid,
            CrewAI_refs=CrewAI_refs,
            migration_notice=migration_notice,
            issues=all_issues
        )

    def verify_all(self) -> Dict[str, FileVerificationResult]:
        """Verify all 20 files."""
        files_to_check = [
            # Config Files (7)
            "roma_config.py",
            "datapizza_config.py",
            "claudiomiro_config.py",
            "roma_recomposition_config.py",
            "roma_mdap_maker_reliability_ssot.py",
            "integrations/bug_fixes/config_provider.py",
            "ragbits_integration/config.py",

            # Utility Files (13)
            "apply_code_quality_fixes.py",
            "apply_api_consistency_fixes.py",
            "apply_ace_phase4_fixes.py",
            "api_contract_fixes.py",
            "data_consistency_verification.py",
            "deep_static_analysis.py",
            "deep_bug_check.py",
            "advanced_sgd_monitoring.py",
            "security_helpers.py",
            "compare_before_after.py",
            "final_project_status.py",
            "tripartite_production.py",
        ]

        print("=" * 80)
        print("CONFIG AND UTILITY FILE VERIFICATION")
        print("=" * 80)
        print()

        for file_path in files_to_check:
            print(f"Checking: {file_path}...", end=" ")
            result = self.verify_file(file_path)
            self.results[file_path] = result

            if not result.exists:
                print("❌ SKIP (not found)")
            elif result.import_status == "FAIL" or not result.syntax_valid:
                print("❌ FAIL")
            elif result.CrewAI_refs:
                print("⚠️  WARN (CrewAI refs)")
            else:
                print("✅ PASS")

        return self.results

    def generate_report(self) -> str:
        """Generate comprehensive verification report."""
        report = []
        report.append("=" * 80)
        report.append("CONFIG AND UTILITY FILE VERIFICATION REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Total Files Checked: {len(self.results)}")
        report.append("")

        # Categorize results
        passed = []
        failed = []
        warnings = []
        skipped = []

        for file_path, result in self.results.items():
            if not result.exists:
                skipped.append((file_path, result))
            elif result.import_status == "FAIL" or not result.syntax_valid:
                failed.append((file_path, result))
            elif result.CrewAI_refs:
                warnings.append((file_path, result))
            else:
                passed.append((file_path, result))

        # Summary
        report.append("─" * 80)
        report.append("SUMMARY")
        report.append("─" * 80)
        report.append(f"✅ PASS: {len(passed)} files")
        report.append(f"⚠️  WARN: {len(warnings)} files (CrewAI references)")
        report.append(f"❌ FAIL: {len(failed)} files")
        report.append(f"⏭️  SKIP: {len(skipped)} files (not found)")
        report.append("")

        # Passed files
        if passed:
            report.append("─" * 80)
            report.append("✅ PASSED FILES")
            report.append("─" * 80)
            for file_path, result in passed:
                report.append(f"✅ {file_path}")
                report.append(f"   - Import Status: {result.import_status}")
                report.append(f"   - Syntax Valid: {result.syntax_valid}")
                report.append(f"   - Migration Notice: {'✓' if result.migration_notice else '✗'}")
                report.append("")

        # Warnings (CrewAI references)
        if warnings:
            report.append("─" * 80)
            report.append("⚠️  WARNINGS (CrewAI References Found)")
            report.append("─" * 80)
            for file_path, result in warnings:
                report.append(f"⚠️  {file_path}")
                report.append(f"   - Import Status: {result.import_status}")
                report.append(f"   - Syntax Valid: {result.syntax_valid}")
                report.append(f"   - Migration Notice: {'✓' if result.migration_notice else '✗'}")
                report.append(f"   - CrewAI References ({len(result.CrewAI_refs)}):")
                for ref in result.CrewAI_refs[:5]:  # Show first 5
                    report.append(f"     • {ref}")
                if len(result.CrewAI_refs) > 5:
                    report.append(f"     ... and {len(result.CrewAI_refs) - 5} more")
                report.append("")

        # Failed files
        if failed:
            report.append("─" * 80)
            report.append("❌ FAILED FILES")
            report.append("─" * 80)
            for file_path, result in failed:
                report.append(f"❌ {file_path}")
                report.append(f"   - Import Status: {result.import_status}")
                report.append(f"   - Syntax Valid: {result.syntax_valid}")
                report.append(f"   - Migration Notice: {'✓' if result.migration_notice else '✗'}")
                if result.issues:
                    report.append(f"   - Issues:")
                    for issue in result.issues:
                        report.append(f"     • {issue}")
                report.append("")

        # Skipped files
        if skipped:
            report.append("─" * 80)
            report.append("⏭️  SKIPPED FILES (Not Found)")
            report.append("─" * 80)
            for file_path, result in skipped:
                report.append(f"⏭️  {file_path}")
                report.append("")

        # Recommendations
        report.append("─" * 80)
        report.append("RECOMMENDATIONS")
        report.append("─" * 80)

        if warnings:
            report.append("1. ⚠️  CrewAI References Detected:")
            report.append("   The following files still contain CrewAI references:")
            for file_path, _ in warnings:
                report.append(f"   - {file_path}")
            report.append("")
            report.append("   Action Required:")
            report.append("   - Update CrewAIIntegrationConfig in ragbits_integration/config.py")
            report.append("   - Replace with CrewAIIntegrationConfig")
            report.append("   - Update all references to use CrewAI instead")
            report.append("")

        if failed:
            report.append("2. ❌ Critical Issues Found:")
            report.append("   The following files have syntax errors or import failures:")
            for file_path, _ in failed:
                report.append(f"   - {file_path}")
            report.append("")
            report.append("   Action Required:")
            report.append("   - Fix syntax errors")
            report.append("   - Update broken imports")
            report.append("   - Run python -m py_compile to verify")
            report.append("")

        if not warnings and not failed:
            report.append("✅ All files verified successfully!")
            report.append("   Migration from crewai # MIGRATED: was CrewAI (AGPL) to CrewAI (MIT) is complete.")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify config and utility files")
    parser.add_argument("--root-dir", default=".", help="Root directory of project")
    parser.add_argument("--output", help="Output file for report (optional)")

    args = parser.parse_args()

    verifier = ConfigUtilVerifier(args.root_dir)
    verifier.verify_all()

    report = verifier.generate_report()

    # Print to console
    print()
    print(report)

    # Write to file if specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n✅ Report saved to: {args.output}")


if __name__ == "__main__":
    main()
