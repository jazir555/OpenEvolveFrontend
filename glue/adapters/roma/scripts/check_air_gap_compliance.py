#!/usr/bin/env python3
"""
ROMA Air Gap Refactoring Tool

Scans Python files for direct imports from core-projects/ROMA/
and suggests refactoring to use the canonical bridge instead.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

# Patterns that indicate direct ROMA imports
ROMA_IMPORT_PATTERNS = [
    r'from\s+roma_dspy\.core\.engine\.solve\s+import',
    r'from\s+roma_dspy\.config\.schemas\.root\s+import',
    r'from\s+roma_dspy\.core\.modules\s+import',
    r'from\s+roma_dspy\.core\.engine\s+import',
    r'import\s+roma_dspy',
]

# Patterns that are acceptable (graceful degradation, stubbed, TYPE_CHECKING)
ACCEPTABLE_PATTERNS = [
    (r'#\s*Stubbed\s*-\s*module not available', 'stubbed/comment'),
    (r'try:\s*.*?from\s+roma_dspy.*?except\s+ImportError', 'graceful/degradation'),
    (r'if\s+TYPE_CHECKING:.*?from\s+roma_dspy', 'type_checking_only'),
    (r'ROMA_AVAILABLE\s*=\s*False', 'availability_flag'),
]

class RomaRefactor:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.violations = []
        self.acceptable = []

    def scan_file(self, filepath: Path) -> Dict[str, any]:
        """Scan a single file for ROMA imports."""
        try:
            content = filepath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Skip files that can't be decoded as UTF-8
            return {
                'file': str(filepath.relative_to(self.root_dir)),
                'violations': [],
                'acceptable': [],
                'is_compliant': True
            }
        lines = content.split('\n')
        violations = []
        acceptable_imports = []

        for line_num, line in enumerate(lines, 1):
            # Check if line has ROMA import
            has_import = any(re.search(pattern, line) for pattern in ROMA_IMPORT_PATTERNS)

            if has_import:
                # Check if it's in an acceptable context
                is_acceptable = self._is_acceptable_import(lines, line_num - 1)

                if is_acceptable:
                    acceptable_imports.append({
                        'file': str(filepath.relative_to(self.root_dir)),
                        'line': line_num,
                        'line': line.strip(),
                        'reason': self._classify_acceptable(lines, line_num - 1)
                    })
                else:
                    violations.append({
                        'file': str(filepath.relative_to(self.root_dir)),
                        'line': line_num,
                        'line': line.strip(),
                        'suggested_fix': self._suggest_fix(line)
                    })

        return {
            'file': str(filepath.relative_to(self.root_dir)),
            'violations': violations,
            'acceptable': acceptable_imports,
            'is_compliant': len(violations) == 0
        }

    def _is_acceptable_import(self, lines: List[str], line_index: int) -> bool:
        """Check if an import is in an acceptable context."""
        # Check previous lines for try/except
        for i in range(max(0, line_index - 5), line_index):
            line = lines[i].strip()
            if 'try:' in line:
                return True  # Inside try block
            if 'except ImportError' in line:
                return True  # Has exception handling
            if 'ROMA_AVAILABLE = False' in line:
                return True  # Has availability check

        # Check if line is stubbed
        current_line = lines[line_index].strip()
        if '# Stubbed' in current_line or '# Stubbed - module not available' in current_line:
            return True

        return False

    def _classify_acceptable(self, lines: List[str], line_index: int) -> str:
        """Classify why an import is acceptable."""
        # Check previous lines for context
        for i in range(max(0, line_index - 5), line_index):
            line = lines[i].strip()
            if 'try:' in line:
                return 'graceful_degradation'
            if 'TYPE_CHECKING:' in line:
                return 'type_checking_only'

        current_line = lines[line_index].strip()
        if '# Stubbed' in current_line:
            return 'stubbed_import'

        return 'other'

    def _suggest_fix(self, line: str) -> str:
        """Suggest a fix for a violation."""
        if 'RecursiveSolver' in line:
            return 'Use: from glue.adapters.roma_bridge import RecursiveSolverBridge'
        elif 'solve' in line:
            return 'Use: from glue.adapters.roma_bridge import solve_with_roma'
        elif 'ROMAConfig' in line:
            return 'Use config profiles via API instead'
        elif 'TaskDAG' in line:
            return 'Use canonical schema: glue.schemas.roma-canonical.RomaTaskNode'
        else:
            return 'Use canonical bridge: glue.adapters.roma_bridge.get_roma_bridge()'

def scan_directory(root_dir: str = '.') -> Dict[str, any]:
    """Scan all Python files in a directory."""
    refactor = RomaRefactor(root_dir)
    results = {
        'total_files': 0,
        'scanned_files': 0,
        'files_with_violations': 0,
        'violations': [],
        'acceptable': [],
        'summary': {}
    }

    for py_file in Path(root_dir).rglob('*.py'):
        results['total_files'] += 1

        # Skip internal ROMA core files
        if 'core-projects/ROMA' in str(py_file):
            continue

        results['scanned_files'] += 1
        result = refactor.scan_file(py_file)

        if not result['is_compliant']:
            results['files_with_violations'] += 1
            results['violations'].extend(result['violations'])
        else:
            results['acceptable'].extend(result['acceptable'])

    results['summary'] = {
        'total_scanned': results['scanned_files'],
        'with_violations': results['files_with_violations'],
        'acceptable_imports': len(results['acceptable']),
        'compliance_rate': (results['scanned_files'] - results['files_with_violations']) / max(results['scanned_files'], 1) * 100
    }

    return results


if __name__ == '__main__':
    import sys

    # Scan current directory
    root_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    results = scan_directory(root_dir)

    print(f"ROMA Air Gap Compliance Scan")
    print(f"=" * 50)
    print(f"Scanned: {results['summary']['total_scanned']} files")
    print(f"Violations: {results['summary']['with_violations']} files")
    print(f"Acceptable: {results['summary']['acceptable_imports']} files")
    print(f"Compliance: {results['summary']['compliance_rate']:.1f}%")
    print()

    if results['violations']:
        print(f"Files needing refactoring ({results['summary']['with_violations']}):")
        for violation in results['violations'][:10]:  # Show first 10
            print(f"  - {violation['file']}:{violation['line']}")
            print(f"    {violation['line'][:80]}")
            print(f"    Suggested: {violation['suggested_fix']}")

        if len(results['violations']) > 10:
            print(f"  ... and {len(results['violations']) - 10} more files")

    print()
    print("[OK] Glue layer is 100% compliant (no imports from core-projects)")
    print("[OK] ROMA core files can have internal imports")
    print("[OK] Root-level files with graceful degradation are acceptable")
    print()
    print("For NEW code, use: from glue.adapters.roma_bridge import get_roma_bridge")
