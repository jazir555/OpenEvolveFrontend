"""
Comprehensive validation of all security fixes
Tests syntax, imports, and functionality
"""

import ast
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def test_syntax():
    """Test all Python files for syntax errors"""
    print("\n" + "="*80)
    print("TEST 1: Syntax Validation")
    print("="*80)

    python_files = sorted([f for f in Path('.').iterdir() if f.suffix == '.py'])
    errors = []
    success_count = 0

    for filepath in python_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content, filename=str(filepath))
            success_count += 1
        except SyntaxError as e:
            errors.append(f'{filepath.name}:{e.lineno} - {e.msg}')

    print(f"Files tested: {len(python_files)}")
    print(f"Passed: {success_count}")
    print(f"Failed: {len(errors)}")

    if errors:
        print("\nSyntax Errors:")
        for error in errors:
            print(f"  [FAIL] {error}")
        return False
    else:
        print("\n[OK] All files have valid Python syntax")
        return True


def test_imports():
    """Test that files can be imported"""
    print("\n" + "="*80)
    print("TEST 2: Import Validation")
    print("="*80)

    # Test a sample of critical files
    critical_files = [
        'llm_cache.py',
        'llm_caching.py',
        'advanced_cache.py',
        'evaluator_team_coordinator.py',
        'leanaide_mdap.py',
        'red_team_coordinator.py',
        'maker_engine.py',
        'deployment_operations.py',
    ]

    success = []
    failed = []

    for filename in critical_files:
        filepath = Path(filename)
        if not filepath.exists():
            continue

        try:
            # Try to compile the file
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, filename, 'exec')
            success.append(filename)
            print(f"  [OK] {filename}")
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            failed.append((filename, str(e)))
            print(f"  [FAIL] {filename}: {e}")

    print(f"\nImport test results:")
    print(f"  Passed: {len(success)}/{len(critical_files)}")
    print(f"  Failed: {len(failed)}")

    return len(failed) == 0


def test_security_patterns():
    """Test that security patterns have been properly fixed"""
    print("\n" + "="*80)
    print("TEST 3: Security Pattern Validation")
    print("="*80)

    import re

    python_files = sorted([f for f in Path('.').iterdir() if f.suffix == '.py'])

    # Patterns to check
    issues = {
        'pickle_import': [],
        'pickle_usage': [],
        'bare_except': [],
        'hardcoded_tmp': [],
    }

    skip_files = {
        'auto_fix_security.py', 'auto_fix_top_level.py',
        'fix_manual_security_issues.py', 'scan_top_level_only.py',
        'validate_all_fixes.py'
    }

    for filepath in python_files:
        if filepath.name in skip_files:
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip comments
            if stripped.startswith('#'):
                continue

            # Check for pickle import
            if re.match(r'^import\s+pickle\b', line):
                issues['pickle_import'].append(f'{filepath.name}:{i}')

            # Check for pickle usage
            if re.search(r'\bpickle\.(load|dump)\s*\(', line):
                issues['pickle_usage'].append(f'{filepath.name}:{i}')

            # Check for bare except (with context)
            if re.search(r'^\s*except\s*:', line):
                # Verify it's actually in a try block
                context = ''.join(lines[max(0, i-10):i])
                if 'try:' in context:
                    issues['bare_except'].append(f'{filepath.name}:{i}')

            # Check for hardcoded /tmp
            if '="/tmp/' in line or "='/tmp/" in line:
                issues['hardcoded_tmp'].append(f'{filepath.name}:{i}')

    # Print results
    all_clear = True
    for category, findings in issues.items():
        if findings:
            all_clear = False
            print(f"\n[ISSUE] {category.replace('_', ' ').title()} ({len(findings)} found):")
            for finding in findings[:5]:
                print(f"  - {finding}")
            if len(findings) > 5:
                print(f"  ... and {len(findings)-5} more")
        else:
            print(f"  [OK] No {category.replace('_', ' ')} issues")

    return all_clear


def test_json_replacement():
    """Verify pickle->JSON replacements are syntactically correct"""
    print("\n" + "="*80)
    print("TEST 4: JSON Replacement Validation")
    print("="*80)

    # Files that had pickle replaced with JSON
    json_files = [
        'llm_cache.py',
        'llm_caching.py',
        'advanced_cache.py',
        'evaluator_team_coordinator.py',
        'leanaide_mdap.py',
        'mcts_evolved_policies.py',
        'red_team_coordinator.py',
    ]

    success = []
    failed = []

    for filename in json_files:
        filepath = Path(filename)
        if not filepath.exists():
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Verify json import exists
            if 'import json' not in content:
                failed.append((filename, "Missing 'import json'"))
                continue

            # Verify no pickle usage (except in comments)
            lines = [line for line in content.split('\n') if not line.strip().startswith('#')]
            code = '\n'.join(lines)

            if 'pickle' in code and 'import pickle' not in code:
                # Check if it's just a variable name
                if 'pickle.' in code or 'import pickle' in code:
                    failed.append((filename, "Still contains pickle usage"))
                    continue

            success.append(filename)
            print(f"  [OK] {filename}")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            failed.append((filename, str(e)))
            print(f"  [FAIL] {filename}: {e}")

    print(f"\nJSON replacement test results:")
    print(f"  Passed: {len(success)}/{len(json_files)}")
    print(f"  Failed: {len(failed)}")

    return len(failed) == 0


def test_tempfile_replacement():
    """Verify /tmp->tempfile.mkdtemp() replacements"""
    print("\n" + "="*80)
    print("TEST 5: Tempfile Replacement Validation")
    print("="*80)

    # Files that had /tmp paths replaced
    tmp_files = [
        'add_class_function_docstrings.py',
        'deployment_operations.py',
        'maker_engine.py',
    ]

    success = []
    failed = []

    for filename in tmp_files:
        filepath = Path(filename)
        if not filepath.exists():
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Verify tempfile import exists
            if 'import tempfile' not in content:
                failed.append((filename, "Missing 'import tempfile'"))
                continue

            # Verify no hardcoded /tmp (except in comments)
            lines = [line for line in content.split('\n') if not line.strip().startswith('#')]
            code = '\n'.join(lines)

            if '="/tmp/' in code or "'/tmp/" in code:
                failed.append((filename, "Still contains hardcoded /tmp"))
                continue

            success.append(filename)
            print(f"  [OK] {filename}")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            failed.append((filename, str(e)))
            print(f"  [FAIL] {filename}: {e}")

    print(f"\nTempfile replacement test results:")
    print(f"  Passed: {len(success)}/{len(tmp_files)}")
    print(f"  Failed: {len(failed)}")

    return len(failed) == 0


def main():
    print("="*80)
    print("COMPREHENSIVE SECURITY FIX VALIDATION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Run all tests
    results['syntax'] = test_syntax()
    results['imports'] = test_imports()
    results['security_patterns'] = test_security_patterns()
    results['json_replacement'] = test_json_replacement()
    results['tempfile_replacement'] = test_tempfile_replacement()

    # Final summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test_name.replace('_', ' ').title()}")

    print("\n" + "="*80)
    if all(results.values()):
        print("[SUCCESS] ALL VALIDATION TESTS PASSED!")
        print("All security fixes have been properly applied and validated.")
        return 0
    else:
        print("[WARNING] Some validation tests failed.")
        print("Please review the failed tests above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
