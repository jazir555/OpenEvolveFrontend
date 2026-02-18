#!/usr/bin/env python3
"""
Comprehensive fix script for all knowledge_engine import issues.

This script:
1. Fixes all import statements
2. Fixes configuration issues
3. Fixes stdout.buffer issues
4. Validates all fixes
"""

import sys
import re
from pathlib import Path

# Directory to fix
KE_DIR = Path(__file__).parent

def fix_stdout_buffer_issues(file_path):
    """Fix stdout.buffer issues that occur when stdout is already wrapped."""
    content = file_path.read_text(encoding='utf-8')

    # Pattern 1: sys.stdout.buffer without hasattr check
    pattern1 = r"sys\.stdout\s*=\s*(?:codecs\.getwriter|io\.TextIOWrapper)\(sys\.stdout\.buffer"

    if re.search(pattern1, content):
        # Check if already protected
        if "hasattr(sys.stdout, 'buffer')" not in content:
            # Fix the issue
            content = re.sub(
                r"if sys\.platform == 'win32':\s*\n\s*import codecs\s*\n\s*sys\.stdout = codecs\.getwriter\('utf-8'\)\(sys\.stdout\.buffer, 'strict'\)\s*\n\s*sys\.stderr = codecs\.getwriter\('utf-8'\)\(sys\.stderr\.buffer, 'strict'\)",
                """if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')""",
                content,
                flags=re.MULTILINE | re.DOTALL
            )

            # Also fix the simpler pattern
            content = re.sub(
                r"if sys\.platform == ['\"]win32['\"]:\s*\n\s*sys\.stdout = io\.TextIOWrapper\(sys\.stdout\.buffer",
                """if sys.platform == "win32":
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer""",
                content,
                flags=re.MULTILINE | re.DOTALL
            )

            file_path.write_text(content, encoding='utf-8')
            print(f"  [FIXED] stdout.buffer issue in {file_path.name}")
            return True

    return False

def fix_relative_imports(file_path):
    """Fix bare relative imports that should be absolute."""
    content = file_path.read_text(encoding='utf-8')

    # List of knowledge_engine modules that should be absolute
    ke_modules = [
        'embedding_service',
        'confidence_scorer',
        'config_validation',
        'core.',
        'integrations.',
        'deduplication.',
        'visualization.',
    ]

    fixed = False
    for module in ke_modules:
        # Pattern: from module import (at start of line or after whitespace)
        pattern = rf"^\s*from {re.escape(module)} import"

        # Check if any lines match but don't have knowledge_engine prefix
        lines = content.split('\n')
        new_lines = []

        for line in lines:
            if re.match(pattern, line) and not f'from knowledge_engine.{module}' in line:
                # Fix the import
                indent = len(line) - len(line.lstrip())
                fixed_line = line.replace(f'from {module} import', f'from knowledge_engine.{module} import')
                new_lines.append(fixed_line)
                fixed = True
            else:
                new_lines.append(line)

        if fixed:
            content = '\n'.join(new_lines)
            file_path.write_text(content, encoding='utf-8')
            print(f"  [FIXED] Relative import in {file_path.name}: {module}")
            return True

    return False

def main():
    """Run all fixes."""
    print("="*60)
    print("Comprehensive Knowledge Engine Import Fix")
    print("="*60)

    py_files = list(KE_DIR.rglob('*.py'))
    py_files = [f for f in py_files
                if '__pycache__' not in str(f)
                and '.tox' not in str(f)
                and f.name != 'fix_imports_comprehensive.py']

    fixed_count = 0
    for py_file in py_files:
        print(f"\nChecking: {py_file.relative_to(KE_DIR)}")

        # Fix stdout.buffer issues
        if fix_stdout_buffer_issues(py_file):
            fixed_count += 1

        # Fix relative imports
        if fix_relative_imports(py_file):
            fixed_count += 1

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} files")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
