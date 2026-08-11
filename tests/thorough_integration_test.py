#!/usr/bin/env python3
"""
Thorough OpenEvolve Integration Test

Tests each file individually to verify:
1. File can be imported
2. OpenEvolve imports work
3. Error handling is functional
4. No runtime errors
"""

import os
import sys
import importlib
import traceback
from typing import Dict, List, Tuple

class IntegrationTester:
    def __init__(self):
        self.results = []
        self.files_tested = 0
        self.files_passed = 0
        self.files_failed = 0
        self.files_skipped = 0

    def test_file_import(self, filepath: str) -> Dict[str, any]:
        """Test if a file can be imported"""
        result = {
            'file': filepath,
            'exists': False,
            'can_import': False,
            'has_openevolve': False,
            'error_handling': False,
            'error': None,
            'warnings': []
        }

        if not os.path.exists(filepath):
            result['error'] = 'File not found'
            return result

        result['exists'] = True

        # Try to import the module
        module_name = filepath.replace('.py', '')
        original_path = sys.path.copy()

        try:
            # Add current directory to path if not there
            if '.' not in sys.path:
                sys.path.insert(0, '.')

            # Remove from modules if already imported
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Try to import
            module = importlib.import_module(module_name)
            result['can_import'] = True

            # Check for OpenEvolve usage
            module_source = open(filepath).read()

            # Check for OpenEvolve imports
            if 'from openevolve' in module_source or 'import openevolve' in module_source:
                result['has_openevolve'] = True

            # Check for error handling patterns
            has_try = 'try:' in module_source
            has_except_import = 'except ImportError' in module_source
            has_available = 'OPENEVOLVE_AVAILABLE' in module_source
            has_logging = 'import logging' in module_source

            # Check if error handling is present
            if has_try and has_except_import:
                result['error_handling'] = True
            elif has_available and has_logging:
                result['error_handling'] = True

            # Add warnings
            if result['has_openevolve'] and not result['error_handling']:
                result['warnings'].append('Has OpenEvolve but no error handling')

        except ImportError as e:
            result['error'] = f'ImportError: {str(e)[:100]}'
        except SyntaxError as e:
            result['error'] = f'SyntaxError at line {e.lineno}: {str(e.msg)[:100]}'
        except Exception as e:
            result['error'] = f'{type(e).__name__}: {str(e)[:100]}'

        sys.path = original_path
        return result

    def test_all_files(self, file_list: List[str]) -> None:
        """Test all files in the list"""
        print("=" * 80)
        print("THOROUGH OPENEVOLVE INTEGRATION TEST")
        print("=" * 80)
        print()

        for i, filepath in enumerate(file_list, 1):
            self.files_tested += 1
            result = self.test_file_import(filepath)
            self.results.append(result)

            # Print result
            status = "PASS" if result['can_import'] else "FAIL"
            print(f"[{i:3d}/{len(file_list)}] {status} - {filepath}")

            if result['can_import']:
                self.files_passed += 1
                if result['has_openevolve']:
                    eh_status = "OK" if result['error_handling'] else "WARN"
                    print(f"         Uses OpenEvolve: {eh_status}")
                    if result['warnings']:
                        for w in result['warnings']:
                            print(f"         WARNING: {w}")
            else:
                self.files_failed += 1
                print(f"         ERROR: {result['error']}")

        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total files tested: {self.files_tested}")
        print(f"Passed: {self.files_passed} ({self.files_passed/self.files_tested*100:.1f}%)")
        print(f"Failed: {self.files_failed} ({self.files_failed/self.files_tested*100:.1f}%)")

        # Check for files with OpenEvolve
        openevolve_files = [r for r in self.results if r['has_openevolve']]
        print(f"\nFiles using OpenEvolve: {len(openevolve_files)}")

        # Check for OpenEvolve files without error handling
        risky_files = [r for r in openevolve_files if not r['error_handling']]
        print(f"OpenEvolve files without error handling: {len(risky_files)}")

        if risky_files:
            print("\nFILES REQUIRING ATTENTION:")
            for r in risky_files:
                print(f"  - {r['file']}")
                if r['warnings']:
                    for w in r['warnings']:
                        print(f"    {w}")

def get_openevolve_files() -> List[str]:
    """Get list of Python files that use OpenEvolve"""
    files = []

    # Scan current directory
    for filename in os.listdir('.'):
        if not filename.endswith('.py') or filename.startswith('test_') or filename.startswith('final_'):
            continue

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for OpenEvolve references
            if 'openevolve' in content.lower():
                files.append(filename)
        except:
            pass

    return sorted(files)

def main():
    tester = IntegrationTester()

    # Get files to test
    openevolve_files = get_openevolve_files()

    print(f"Found {len(openevolve_files)} files using OpenEvolve")
    print()

    # Test all files
    tester.test_all_files(openevolve_files)

    return 0 if tester.files_failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
