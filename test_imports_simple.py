#!/usr/bin/env python3
"""
Simple Import Testing Script for RESE Framework
Tests all Python files for import errors
"""

import os
import sys
import ast
import importlib
import traceback
from pathlib import Path
from datetime import datetime
import json


def find_python_files(base_path):
    """Find all Python files"""
    return list(Path(base_path).rglob("*.py"))

def check_syntax(file_path):
    """Check Python file syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def get_module_name(file_path, base_path):
    """Get module name from file path"""
    relative = file_path.relative_to(base_path.parent)
    return str(relative.with_suffix('')).replace(os.sep, '.')

def test_import(file_path, base_path):
    """Test importing a single file"""
    result = {
        'file': str(file_path),
        'relative': str(file_path.relative_to(base_path)),
        'success': False,
        'error': None,
        'syntax_error': False
    }

    # Check syntax first
    syntax_ok, syntax_err = check_syntax(file_path)
    if not syntax_ok:
        result['syntax_error'] = True
        result['error'] = f"Syntax Error: {syntax_err}"
        return result

    # Try import
    try:
        module_name = get_module_name(file_path, base_path)
        sys.path.insert(0, str(base_path.parent))

        if module_name in sys.modules:
            del sys.modules[module_name]

        importlib.import_module(module_name)
        result['success'] = True

    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"

    finally:
        if str(base_path.parent) in sys.path:
            sys.path.remove(str(base_path.parent))

    return result

def main():
    """Main test runner"""
    base_path = Path.cwd() / "rese"

    print("=" * 80)
    print("RESE Framework Import Testing")
    print("=" * 80)

    # Find all Python files
    python_files = find_python_files(base_path)
    python_files = [f for f in python_files if "__pycache__" not in str(f)]

    total = len(python_files)
    print(f"\nFound {total} Python files to test\n")

    results = []
    successful = 0
    failed = 0
    syntax_errors = 0

    # Test each file
    for i, file_path in enumerate(python_files, 1):
        relative = file_path.relative_to(base_path)
        print(f"[{i}/{total}] {relative}", end=" ")

        result = test_import(file_path, base_path)
        results.append(result)

        if result['success']:
            successful += 1
            print("[OK]")
        else:
            failed += 1
            if result['syntax_error']:
                syntax_errors += 1
            print("[FAIL]")
            if result['error']:
                print(f"    {result['error'][:100]}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Files:        {total}")
    print(f"Successful:         {successful}")
    print(f"Failed:             {failed}")
    print(f"Syntax Errors:      {syntax_errors}")
    print(f"Success Rate:       {(successful/total*100):.2f}%")
    print("=" * 80)

    # Save detailed report
    failed_results = [r for r in results if not r['success']]
    syntax_error_results = [r for r in failed_results if r['syntax_error']]

    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total': total,
            'successful': successful,
            'failed': failed,
            'syntax_errors': syntax_errors,
            'success_rate': f"{(successful/total*100):.2f}%"
        },
        'failed_imports': [
            {'file': r['relative'], 'error': r['error']}
            for r in failed_results
        ],
        'syntax_errors': [
            {'file': r['relative'], 'error': r['error']}
            for r in syntax_error_results
        ]
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"import_test_report_{timestamp}.json"

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {report_file}")

    # Print failed imports
    if failed_results:
        print(f"\nFailed Imports ({len(failed_results)}):")
        for r in failed_results[:20]:  # Show first 20
            print(f"  - {r['relative']}")
            print(f"    {r['error']}")
        if len(failed_results) > 20:
            print(f"  ... and {len(failed_results) - 20} more")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
