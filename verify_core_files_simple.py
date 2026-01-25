#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


def verify_file(filepath: Path, filename: str) -> dict:
    """Verify a single file"""
    result = {
        'file': filename,
        'exists': False,
        'can_parse': False,
        'hephaestus_imports': [],
        'classes_found': [],
        'syntax_errors': [],
        'bug_fixes': [],
    }

    # Check if file exists
    if not filepath.exists():
        return result
    result['exists'] = True

    # Try to parse the file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content, filename=str(filepath))
        result['can_parse'] = True
    except SyntaxError as e:
        result['syntax_errors'].append(f"Line {e.lineno}: {e.msg}")
        return result
    except Exception as e:
        result['syntax_errors'].append(str(e))
        return result

    # Check for Hephaestus imports
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                module = node.module.lower() if node.module else ""
                if 'hephaestus' in module:
                    result['hephaestus_imports'].append(f"Line {node.lineno}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if 'hephaestus' in alias.name.lower():
                        result['hephaestus_imports'].append(f"Line {node.lineno}: import {alias.name}")

    # Find all class definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            result['classes_found'].append(node.name)

    # Check specific bug fixes
    if filename == 'workflow_structures.py':
        if 'class ValidationResult' in content:
            result['bug_fixes'].append('ValidationResult dataclass')
        if 'class Feedback' in content:
            result['bug_fixes'].append('Feedback dataclass')
        if 'class QualityScores' in content:
            result['bug_fixes'].append('QualityScores dataclass')

    if filename == 'sovereign_data_models.py':
        if 'from workflow_structures import ValidationResult' in content:
            result['bug_fixes'].append('ValidationResult re-export')
        if 'from workflow_structures import Feedback' in content:
            result['bug_fixes'].append('Feedback re-export')
        if 'from workflow_structures import QualityScores' in content:
            result['bug_fixes'].append('QualityScores re-export')
        if 'from crewai_state_management import SolutionAttempt' in content:
            result['bug_fixes'].append('SolutionAttempt re-export')
        if 'def generate_id' in content:
            result['bug_fixes'].append('generate_id function')

    return result


def main():
    """Main verification function"""
    import os
    base_path = Path(os.getcwd())

    # Files to verify
    files_to_verify = [
        'crewai_client.py',
        'crewai_integration.py',
        'crewai_state_management.py',
        'crewai_unified_flow.py',
        'crewai_unified_bridge.py',
        'crewai_mdap_maker_engine.py',
        'crewai_mdap_integrator.py',
        'crewai_zero_error_workflow.py',
        'sovereign_data_models.py',
        'workflow_structures.py',
    ]

    results = []
    for filename in files_to_verify:
        filepath = base_path / filename
        result = verify_file(filepath, filename)
        results.append(result)

    # Print results
    print("\n" + "="*80)
    print("CORE CREWAI INFRASTRUCTURE VERIFICATION REPORT")
    print("="*80 + "\n")

    pass_count = 0
    fail_count = 0

    for result in results:
        print(f"\n{'='*80}")
        print(f"FILE: {result['file']}")
        print(f"{'='*80}")

        # Determine pass/fail
        is_pass = (
            result['exists'] and
            result['can_parse'] and
            len(result['hephaestus_imports']) == 0 and
            len(result['syntax_errors']) == 0
        )

        if is_pass:
            print("[PASS] STATUS: PASS")
            pass_count += 1
        else:
            print("[FAIL] STATUS: FAIL")
            fail_count += 1

        # File existence
        print(f"\n[?] File Exists: {result['exists']}")

        if not result['exists']:
            continue

        # Parse status
        print(f"[?] Can Parse: {result['can_parse']}")

        # Hephaestus imports
        if result['hephaestus_imports']:
            print(f"\n[!] Hephaestus Imports Found ({len(result['hephaestus_imports'])}):")
            for imp in result['hephaestus_imports']:
                print(f"   [X] {imp}")
        else:
            print(f"\n[OK] No Hephaestus imports (only in comments/docs)")

        # Syntax errors
        if result['syntax_errors']:
            print(f"\n[X] Syntax Errors ({len(result['syntax_errors'])}):")
            for err in result['syntax_errors']:
                print(f"   [X] {err}")
        else:
            print(f"\n[OK] No syntax errors")

        # Critical classes
        print(f"\n[*] Classes Found ({len(result['classes_found'])}):")
        if result['classes_found']:
            for cls in sorted(result['classes_found']):
                print(f"   [+] {cls}")
        else:
            print("   [!] No classes found")

        # Bug fixes
        if result['bug_fixes']:
            print(f"\n[BUG] Bug Fixes Present ({len(result['bug_fixes'])}):")
            for fix in result['bug_fixes']:
                print(f"   [+] {fix}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Files: {len(results)}")
    print(f"Passed: {pass_count}")
    print(f"Failed: {fail_count}")
    print(f"Success Rate: {pass_count/len(results)*100:.1f}%")

    # Regression checks
    print("\n" + "="*80)
    print("REGRESSION CHECKS")
    print("="*80)

    all_hephaestus_free = all(len(r['hephaestus_imports']) == 0 for r in results if r['exists'])
    all_parseable = all(r['can_parse'] for r in results if r['exists'])
    no_syntax_errors = all(len(r['syntax_errors']) == 0 for r in results if r['exists'])

    print(f"[OK] All files Hephaestus-free: {all_hephaestus_free}")
    print(f"[OK] All files parseable: {all_parseable}")
    print(f"[OK] No syntax errors: {no_syntax_errors}")

    # Specific class checks
    print("\n" + "="*80)
    print("CRITICAL CLASS VERIFICATION")
    print("="*80)

    # Check crewai_state_management.py
    state_result = next((r for r in results if r['file'] == 'crewai_state_management.py'), None)
    if state_result:
        required_classes = ['WorkflowState', 'SubProblem', 'SolutionAttempt', 'DecompositionPlan']
        found_classes = [cls for cls in required_classes if cls in state_result['classes_found']]
        missing_classes = [cls for cls in required_classes if cls not in state_result['classes_found']]
        print(f"\ncrewai_state_management.py:")
        print(f"  Required: {required_classes}")
        print(f"  Found: {found_classes}")
        if missing_classes:
            print(f"  [X] Missing: {missing_classes}")
        else:
            print(f"  [OK] All required classes present")

    # Check sovereign_data_models.py
    sovereign_result = next((r for r in results if r['file'] == 'sovereign_data_models.py'), None)
    if sovereign_result:
        print(f"\nsovereign_data_models.py:")
        for fix in sovereign_result['bug_fixes']:
            print(f"  [+] {fix}")

    # Check workflow_structures.py
    workflow_result = next((r for r in results if r['file'] == 'workflow_structures.py'), None)
    if workflow_result:
        print(f"\nworkflow_structures.py:")
        for fix in workflow_result['bug_fixes']:
            print(f"  [+] {fix}")

    # Check crewai_client.py
    client_result = next((r for r in results if r['file'] == 'crewai_client.py'), None)
    if client_result:
        has_client = 'CrewAIClient' in client_result['classes_found']
        print(f"\ncrewai_client.py:")
        print(f"  CrewAIClient class: {'[OK] Found' if has_client else '[X] Missing'}")

    print("\n" + "="*80)

    # Exit with appropriate code
    if all(r['exists'] and r['can_parse'] and len(r['hephaestus_imports']) == 0 for r in results):
        print("\n[SUCCESS] ALL VERIFICATIONS PASSED!")
        return 0
    else:
        print("\n[FAILURE] SOME VERIFICATIONS FAILED!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
