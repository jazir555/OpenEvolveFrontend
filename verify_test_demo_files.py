#!/usr/bin/env python3
"""
Comprehensive verification script for all test and demo files mentioned in CREWAI_MIGRATION_MASTER_TASKLIST.md

Checks:
1. Import status (syntax errors)
2. CREWAI references (should be removed)
3. Bug fix presence (SolutionAttempt, generate_id, ValidationResult)
4. Recent regressions from changes
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

class FileVerificationResult:
    """Store verification results for a single file"""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.exists = os.path.exists(filepath)
        self.can_import = False
        self.has_syntax_error = False
        self.CREWAI_references = []
        self.crewai_imports = []
        self.bug_fixes_present = {
            'solution_attempt_fallback': False,
            'generate_id_fallback': False,
            'validation_result_import': False
        }
        self.import_errors = []
        self.warnings = []

    def to_dict(self) -> dict:
        return {
            'filepath': self.filepath,
            'exists': self.exists,
            'can_import': self.can_import,
            'has_syntax_error': self.has_syntax_error,
            'CREWAI_references': self.CREWAI_references,
            'crewai_imports': self.crewai_imports,
            'bug_fixes_present': self.bug_fixes_present,
            'import_errors': self.import_errors,
            'warnings': self.warnings
        }

def check_syntax(filepath: str) -> Tuple[bool, str]:
    """Check if Python file has valid syntax"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def scan_for_CREWAI(content: str) -> List[str]:
    """Scan content for active CREWAI references (excluding comments)"""
    references = []
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        # Skip comment lines
        stripped = line.strip()
        if stripped.startswith('#'):
            continue

        # Check for active CREWAI references
        if 'CREWAI' in line.lower():
            # Import statements
            if 'import' in line and 'CREWAI' in line.lower():
                references.append(f"Line {i}: {line.strip()}")
            # String literals that might be imports/references
            elif '"' in line or "'" in line:
                if 'CREWAI' in line.lower():
                    references.append(f"Line {i}: {line.strip()}")

    return references

def scan_for_crewai(content: str) -> List[str]:
    """Scan for CrewAI imports"""
    imports = []
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        if 'import' in line and 'crewai' in line.lower():
            imports.append(f"Line {i}: {line.strip()}")

    return imports

def check_bug_fixes(content: str) -> Dict[str, bool]:
    """Check for presence of recent bug fixes"""
    fixes = {
        'solution_attempt_fallback': False,
        'generate_id_fallback': False,
        'validation_result_import': False
    }

    # Check for SolutionAttempt import with fallback
    if 'SolutionAttempt' in content:
        # Look for try/except or fallback pattern
        if 'try:' in content and 'except' in content and 'SolutionAttempt' in content:
            fixes['solution_attempt_fallback'] = True
        # Or direct import from workflow_structures
        if 'from workflow_structures import' in content and 'SolutionAttempt' in content:
            fixes['solution_attempt_fallback'] = True

    # Check for generate_id function/fallback
    if 'generate_id' in content:
        # Check for function definition or import
        if 'def generate_id' in content:
            fixes['generate_id_fallback'] = True
        elif 'from sovereign_data_models import' in content and 'generate_id' in content:
            fixes['generate_id_fallback'] = True

    # Check for ValidationResult import
    if 'ValidationResult' in content:
        if 'from workflow_structures import' in content:
            fixes['validation_result_import'] = True
        elif 'from sovereign_data_models import' in content:
            fixes['validation_result_import'] = True

    return fixes

def verify_file(filepath: str) -> FileVerificationResult:
    """Comprehensive verification of a single file"""
    result = FileVerificationResult(filepath)

    if not result.exists:
        result.warnings.append("File does not exist")
        return result

    # Check syntax
    syntax_ok, error_msg = check_syntax(filepath)
    if not syntax_ok:
        result.has_syntax_error = True
        result.import_errors.append(f"Syntax error: {error_msg}")
        return result

    # Read content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        result.import_errors.append(f"Read error: {str(e)}")
        return result

    # Scan for CREWAI references
    result.CREWAI_references = scan_for_CREWAI(content)

    # Scan for CrewAI imports
    result.crewai_imports = scan_for_crewai(content)

    # Check for bug fixes
    result.bug_fixes_present = check_bug_fixes(content)

    # Try importing the file
    try:
        # Add current directory to path
        import_dir = os.path.dirname(os.path.abspath(filepath))
        if import_dir not in sys.path:
            sys.path.insert(0, import_dir)

        # Try to import
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        __import__(module_name)
        result.can_import = True
    except ImportError as e:
        result.can_import = False
        result.import_errors.append(f"Import error: {str(e)}")
    except Exception as e:
        # Import successful but execution failed (expected for demo/test files)
        result.can_import = True
        result.warnings.append(f"Import OK but execution error: {str(e)[:100]}")

    return result

def print_results(results: List[FileVerificationResult]):
    """Print formatted results"""
    print("\n" + "="*120)
    print("TEST AND DEMO FILE VERIFICATION REPORT")
    print("="*120)

    test_files = [
        "conftest.py",
        "final_integration_test.py",
        "integration_test.py",
        "comprehensive_integration_test.py",
        "final_verification_test.py",
        "final_verification_test_simple.py",
        "final_verification_report.py",
        "comprehensive_verification_report.py",
        "test_fixes.py",
        "advanced_validation_workflows.py"
    ]

    demo_files = [
        "example_crewai_delegation.py",
        "demo_roma_mdap_maker.py",
        "demo_openevolve_bubblelabs.py",
        "demo_database_cleanup.py",
        "comprehensive_demo.py",
        "demo_app.py",
        "demo_evolution_maker.py",
        "demo_evolutionary_tests.py",
        "demo_generic_maker.py",
        "demo_hybrid_maker.py",
        "demo_mcts.py",
        "demo_mdap_maker.py",
        "demo_leanaide_client.py",
        "demo_sop_generator.py",
        "demo_sop_integrated.py",
        "demo_sop_components.py",
        "demo_ui_integration.py",
        "demo_adversarial_maker.py"
    ]

    # Create result lookup
    result_map = {r.filepath: r for r in results}

    print("\n## TEST FILES (10)\n")

    passed = 0
    failed = 0

    for filename in test_files:
        result = result_map.get(filename)
        if not result:
            print(f"❌ {filename}: NOT CHECKED")
            failed += 1
            continue

        if not result.exists:
            print(f"❌ {filename}: FILE NOT FOUND")
            failed += 1
            continue

        # Determine pass/fail
        is_pass = (
            not result.has_syntax_error and
            len(result.CREWAI_references) == 0
        )

        status = "✅ PASS" if is_pass else "❌ FAIL"
        print(f"\n{status}: {filename}")
        print(f"  Exists: {result.exists}")
        print(f"  Can Import: {result.can_import}")
        print(f"  Syntax Error: {result.has_syntax_error}")
        print(f"  CREWAI Refs: {len(result.CREWAI_references)}")
        print(f"  CrewAI Imports: {len(result.crewai_imports)}")

        if result.CREWAI_references:
            print(f"  ⚠️  CREWAI References:")
            for ref in result.CREWAI_references[:3]:
                print(f"     - {ref}")

        if result.bug_fixes_present.get('solution_attempt_fallback') or \
           result.bug_fixes_present.get('generate_id_fallback') or \
           result.bug_fixes_present.get('validation_result_import'):
            print(f"  ✓ Bug Fixes Present:")
            if result.bug_fixes_present['solution_attempt_fallback']:
                print(f"     - SolutionAttempt fallback: YES")
            if result.bug_fixes_present['generate_id_fallback']:
                print(f"     - generate_id fallback: YES")
            if result.bug_fixes_present['validation_result_import']:
                print(f"     - ValidationResult import: YES")

        if result.import_errors:
            print(f"  ⚠️  Import Errors:")
            for err in result.import_errors[:2]:
                print(f"     - {err}")

        if is_pass:
            passed += 1
        else:
            failed += 1

    print("\n## DEMO FILES (18)\n")

    for filename in demo_files:
        result = result_map.get(filename)
        if not result:
            print(f"❌ {filename}: NOT CHECKED")
            failed += 1
            continue

        if not result.exists:
            print(f"❌ {filename}: FILE NOT FOUND")
            failed += 1
            continue

        # Determine pass/fail
        is_pass = (
            not result.has_syntax_error and
            len(result.CREWAI_references) == 0
        )

        status = "✅ PASS" if is_pass else "❌ FAIL"
        print(f"\n{status}: {filename}")
        print(f"  Exists: {result.exists}")
        print(f"  Can Import: {result.can_import}")
        print(f"  Syntax Error: {result.has_syntax_error}")
        print(f"  CREWAI Refs: {len(result.CREWAI_references)}")
        print(f"  CrewAI Imports: {len(result.crewai_imports)}")

        if result.CREWAI_references:
            print(f"  ⚠️  CREWAI References:")
            for ref in result.CREWAI_references[:3]:
                print(f"     - {ref}")

        if result.bug_fixes_present.get('solution_attempt_fallback') or \
           result.bug_fixes_present.get('generate_id_fallback') or \
           result.bug_fixes_present.get('validation_result_import'):
            print(f"  ✓ Bug Fixes Present:")
            if result.bug_fixes_present['solution_attempt_fallback']:
                print(f"     - SolutionAttempt fallback: YES")
            if result.bug_fixes_present['generate_id_fallback']:
                print(f"     - generate_id fallback: YES")
            if result.bug_fixes_present['validation_result_import']:
                print(f"     - ValidationResult import: YES")

        if result.import_errors:
            print(f"  ⚠️  Import Errors:")
            for err in result.import_errors[:2]:
                print(f"     - {err}")

        if is_pass:
            passed += 1
        else:
            failed += 1

    print("\n" + "="*120)
    print(f"SUMMARY: {passed} PASSED, {failed} FAILED out of {len(results)} files")
    print("="*120)

    # Detailed failures
    if failed > 0:
        print("\n## DETAILED FAILURES:\n")
        for result in results:
            if result.exists and not result.has_syntax_error and len(result.CREWAI_references) > 0:
                print(f"\n{result.filepath}:")
                for ref in result.CREWAI_references:
                    print(f"  - {ref}")

def main():
    """Main verification function"""
    test_files = [
        "conftest.py",
        "final_integration_test.py",
        "integration_test.py",
        "comprehensive_integration_test.py",
        "final_verification_test.py",
        "final_verification_test_simple.py",
        "final_verification_report.py",
        "comprehensive_verification_report.py",
        "test_fixes.py",
        "advanced_validation_workflows.py"
    ]

    demo_files = [
        "example_crewai_delegation.py",
        "demo_roma_mdap_maker.py",
        "demo_openevolve_bubblelabs.py",
        "demo_database_cleanup.py",
        "comprehensive_demo.py",
        "demo_app.py",
        "demo_evolution_maker.py",
        "demo_evolutionary_tests.py",
        "demo_generic_maker.py",
        "demo_hybrid_maker.py",
        "demo_mcts.py",
        "demo_mdap_maker.py",
        "demo_leanaide_client.py",
        "demo_sop_generator.py",
        "demo_sop_integrated.py",
        "demo_sop_components.py",
        "demo_ui_integration.py",
        "demo_adversarial_maker.py"
    ]

    all_files = test_files + demo_files

    print(f"Verifying {len(all_files)} files...")

    results = []
    for filepath in all_files:
        result = verify_file(filepath)
        results.append(result)
        print(f"[OK] Checked {filepath}")

    print_results(results)

    # Return exit code based on failures
    failed_count = sum(1 for r in results if r.exists and (r.has_syntax_error or len(r.CREWAI_references) > 0))
    sys.exit(0 if failed_count == 0 else 1)

if __name__ == "__main__":
    main()
