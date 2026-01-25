#!/usr/bin/env python3
"""
Verification Script for Core CrewAI Infrastructure Files

This script verifies that all core CrewAI files mentioned in
CREWAI_MIGRATION_MASTER_TASKLIST.md are properly implemented.

Author: Claude Code
Date: 2026-01-21
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class FileVerificationResult:
    """Results for a single file verification"""
    def __init__(self, filename: str):
        self.filename = filename
        self.exists = False
        self.can_parse = False
        self.import_errors = []
        self.hephaestus_imports = []
        self.critical_classes = {}
        self.syntax_errors = []
        self.bug_fixes_present = []

    def to_dict(self) -> dict:
        return {
            'file': self.filename,
            'exists': self.exists,
            'can_parse': self.can_parse,
            'import_errors': self.import_errors,
            'hephaestus_imports': self.hephaestus_imports,
            'critical_classes': self.critical_classes,
            'syntax_errors': self.syntax_errors,
            'bug_fixes_present': self.bug_fixes_present,
        }


def verify_file(filepath: Path, expected_classes: List[str]) -> FileVerificationResult:
    """Verify a single file"""
    result = FileVerificationResult(filepath.name)

    # Check if file exists
    if not filepath.exists():
        result.exists = False
        return result
    result.exists = True

    # Try to parse the file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content, filename=str(filepath))
        result.can_parse = True
    except SyntaxError as e:
        result.syntax_errors.append(f"Line {e.lineno}: {e.msg}")
        result.can_parse = False
        return result
    except Exception as e:
        result.syntax_errors.append(str(e))
        result.can_parse = False
        return result

    # Check for Hephaestus imports
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                module = node.module.lower() if node.module else ""
                if 'hephaestus' in module:
                    result.hephaestus_imports.append(f"Line {node.lineno}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if 'hephaestus' in alias.name.lower():
                        result.hephaestus_imports.append(f"Line {node.lineno}: import {alias.name}")

    # Find all class definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            result.critical_classes[class_name] = True

    # Check specific bug fixes
    content_lower = content.lower()

    # Bug fix: ValidationResult/Feedback/QualityScores in workflow_structures.py
    if filepath.name == 'workflow_structures.py':
        if 'class ValidationResult' in content:
            result.bug_fixes_present.append('ValidationResult dataclass')
        if 'class Feedback' in content:
            result.bug_fixes_present.append('Feedback dataclass')
        if 'class QualityScores' in content:
            result.bug_fixes_present.append('QualityScores dataclass')

    # Bug fix: Re-exports in sovereign_data_models.py
    if filepath.name == 'sovereign_data_models.py':
        if 'from workflow_structures import ValidationResult' in content:
            result.bug_fixes_present.append('ValidationResult re-export')
        if 'from workflow_structures import Feedback' in content:
            result.bug_fixes_present.append('Feedback re-export')
        if 'from workflow_structures import QualityScores' in content:
            result.bug_fixes_present.append('QualityScores re-export')
        if 'from crewai_state_management import SolutionAttempt' in content:
            result.bug_fixes_present.append('SolutionAttempt re-export')
        if 'def generate_id' in content:
            result.bug_fixes_present.append('generate_id function')

    return result


def print_results(results: List[FileVerificationResult]):
    """Print verification results"""
    print("\n" + "="*80)
    print("CORE CREWAI INFRASTRUCTURE VERIFICATION REPORT")
    print("="*80 + "\n")

    pass_count = 0
    fail_count = 0

    for result in results:
        print(f"\n{'='*80}")
        print(f"FILE: {result.filename}")
        print(f"{'='*80}")

        # Determine pass/fail
        is_pass = (
            result.exists and
            result.can_parse and
            len(result.hephaestus_imports) == 0 and
            len(result.syntax_errors) == 0
        )

        if is_pass:
            print("✅ STATUS: PASS")
            pass_count += 1
        else:
            print("❌ STATUS: FAIL")
            fail_count += 1

        # File existence
        print(f"\n📁 File Exists: {'✅ Yes' if result.exists else '❌ No'}")

        if not result.exists:
            continue

        # Parse status
        print(f"🔍 Can Parse: {'✅ Yes' if result.can_parse else '❌ No'}")

        # Hephaestus imports
        if result.hephaestus_imports:
            print(f"\n⚠️  Hephaestus Imports Found ({len(result.hephaestus_imports)}):")
            for imp in result.hephaestus_imports:
                print(f"   ❌ {imp}")
        else:
            print(f"\n✅ No Hephaestus imports (only in comments/docs)")

        # Syntax errors
        if result.syntax_errors:
            print(f"\n❌ Syntax Errors ({len(result.syntax_errors)}):")
            for err in result.syntax_errors:
                print(f"   ❌ {err}")
        else:
            print(f"\n✅ No syntax errors")

        # Critical classes
        print(f"\n📦 Classes Found ({len(result.critical_classes)}):")
        if result.critical_classes:
            for cls in sorted(result.critical_classes.keys()):
                print(f"   ✅ {cls}")
        else:
            print("   ⚠️  No classes found")

        # Bug fixes
        if result.bug_fixes_present:
            print(f"\n🐛 Bug Fixes Present ({len(result.bug_fixes_present)}):")
            for fix in result.bug_fixes_present:
                print(f"   ✅ {fix}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Files: {len(results)}")
    print(f"✅ Passed: {pass_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"Success Rate: {pass_count/len(results)*100:.1f}%")

    # Regression checks
    print("\n" + "="*80)
    print("REGRESSION CHECKS")
    print("="*80)

    all_hephaestus_free = all(len(r.hephaestus_imports) == 0 for r in results if r.exists)
    all_parseable = all(r.can_parse for r in results if r.exists)
    no_syntax_errors = all(len(r.syntax_errors) == 0 for r in results if r.exists)

    print(f"✅ All files Hephaestus-free: {'✅ Yes' if all_hephaestus_free else '❌ No'}")
    print(f"✅ All files parseable: {'✅ Yes' if all_parseable else '❌ No'}")
    print(f"✅ No syntax errors: {'✅ Yes' if no_syntax_errors else '❌ No'}")


def main():
    """Main verification function"""
    base_path = Path('/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

    # Files to verify with expected critical classes
    files_to_verify = [
        ('crewai_client.py', ['CrewAIClient']),
        ('crewai_integration.py', []),
        ('crewai_state_management.py', ['WorkflowState', 'SubProblem', 'SolutionAttempt', 'DecompositionPlan']),
        ('crewai_unified_flow.py', []),
        ('crewai_unified_bridge.py', []),
        ('crewai_mdap_maker_engine.py', []),
        ('crewai_mdap_integrator.py', []),
        ('crewai_zero_error_workflow.py', []),
        ('sovereign_data_models.py', ['ValidationResult', 'Feedback', 'QualityScores', 'generate_id']),
        ('workflow_structures.py', ['ValidationResult', 'Feedback', 'QualityScores']),
    ]

    results = []
    for filename, expected_classes in files_to_verify:
        filepath = base_path / filename
        result = verify_file(filepath, expected_classes)
        results.append(result)

    print_results(results)

    # Exit with appropriate code
    if all(r.exists and r.can_parse and len(r.hephaestus_imports) == 0 for r in results):
        print("\n✅ ALL VERIFICATIONS PASSED!")
        return 0
    else:
        print("\n❌ SOME VERIFICATIONS FAILED!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
