#!/usr/bin/env python3
"""
Analyze OpenEvolve integration across all project files.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

def analyze_file(filepath: str) -> Dict[str, any]:
    """Analyze a single file for OpenEvolve integration."""
    result = {
        'path': filepath,
        'exists': False,
        'has_openevolve_import': False,
        'has_try_except': False,
        'has_available_flag': False,
        'has_logging': False,
        'imports': [],
        'issues': []
    }

    if not os.path.exists(filepath):
        result['issues'].append('File not found')
        return result

    result['exists'] = True

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        # Check for OpenEvolve imports
        for i, line in enumerate(lines):
            if 'from openevolve' in line or 'import openevolve' in line:
                # Skip wrapper imports
                if 'openevolve_integration' in line or 'openevolve_structures' in line:
                    continue
                result['imports'].append((i+1, line.strip()))
                result['has_openevolve_import'] = True

        # Check for error handling patterns
        result['has_try_except'] = 'try:' in content and 'except ImportError' in content
        result['has_available_flag'] = 'OPENEVOLVE_AVAILABLE' in content
        result['has_logging'] = 'import logging' in content

        # Identify issues
        if result['has_openevolve_import'] and not result['has_try_except']:
            result['issues'].append('No try/except error handling')

        if result['has_available_flag'] and not result['has_logging']:
            result['issues'].append('Has OPENEVOLVE_AVAILABLE but no logging import')

    except Exception as e:
        result['issues'].append(f'Error reading file: {e}')

    return result

def main():
    """Main analysis function."""
    print("OpenEvolve Integration Analysis")
    print("=" * 70)
    print()

    # Get all Python files
    py_files = [f for f in os.listdir('.') if f.endswith('.py') and os.path.isfile(f)]

    # Analyze all files
    results = []
    for filename in sorted(py_files):
        result = analyze_file(filename)
        if result['has_openevolve_import'] or result['issues']:
            results.append(result)

    # Categorize results
    with_error_handling = [r for r in results if r['has_try_except'] and r['has_openevolve_import']]
    without_error_handling = [r for r in results if r['has_openevolve_import'] and not r['has_try_except']]
    with_issues = [r for r in results if r['issues'] and not r['has_openevolve_import']]

    # Print summary
    print(f"Total files analyzed: {len(py_files)}")
    print(f"Files with OpenEvolve imports: {len([r for r in results if r['has_openevolve_import']])}")
    print(f"With error handling: {len(with_error_handling)}")
    print(f"Without error handling: {len(without_error_handling)}")
    print(f"With other issues: {len(with_issues)}")
    print()

    # Files WITHOUT error handling (need attention)
    if without_error_handling:
        print("FILES REQUIRING ATTENTION (No error handling):")
        print("-" * 70)
        for r in without_error_handling:
            print(f"\n{r['path']}:")
            for line_num, line in r['imports'][:3]:
                print(f"  Line {line_num}: {line[:65]}...")
            if r['issues']:
                for issue in r['issues']:
                    print(f"  WARNING: {issue}")

    print()
    print("=" * 70)
    print("Analysis complete.")

if __name__ == '__main__':
    main()
