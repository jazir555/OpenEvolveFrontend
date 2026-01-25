#!/usr/bin/env python3
"""
Quick Import Checker for OpenEvolve Frontend Root Files
Checks only root-level Python files for broken imports
"""

import sys
import os
import ast
import importlib
from pathlib import Path
from typing import List, Dict, Tuple

BROKEN_IMPORTS = []
MISSING_MODULES = []

class ImportChecker(ast.NodeVisitor):
    """AST visitor to extract all imports from a Python file"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.imports = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(('import', alias.name, node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module if node.module else ''
        for alias in node.names:
            self.imports.append(('from', f'{module}.{alias.name}' if module else alias.name, node.lineno))
        self.generic_visit(node)


def check_module_exists(module_name: str) -> Tuple[bool, str]:
    """Check if a module can be imported"""
    try:
        if module_name.startswith('.'):
            return True, "Relative import"
        importlib.import_module(module_name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        return False, f"Error: {str(e)}"


def check_file(filepath: Path) -> List[Dict]:
    """Check a single Python file for broken imports"""
    issues = []

    try:
        filepath = filepath.resolve()
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        checker = ImportChecker(str(filepath))
        checker.visit(tree)

        for import_type, import_name, lineno in checker.imports:
            # Skip standard library and common third-party modules
            base = import_name.split('.')[0]
            if base in {
                'os', 'sys', 'json', 're', 'datetime', 'pathlib', 'typing',
                'logging', 'threading', 'collections', 'functools', 'dataclasses',
                'copy', 'uuid', 'hashlib', 'time', 'base64', 'io', 'tempfile',
                'enum', 'contextlib', 'queue', 'multiprocessing', 'math', 'random',
                'string', 'secrets', 'gc', 'tracemalloc', 'weakref', 'pickle',
                'sqlite3', 'statistics', 'decimal', 'fractions', 'itertools',
                'asyncio', 'concurrent', 'unittest', 'test',
                'numpy', 'pandas', 'plotly', 'requests', 'streamlit',
                'fastapi', 'uvicorn', 'pydantic', 'click', 'typer',
                'aiohttp', 'PIL', 'cv2', 'torch', 'tensorflow',
                'sklearn', 'scipy', 'matplotlib', 'networkx',
                'openai', 'anthropic', 'langchain', 'chromadb', 'dspy'
            }:
                continue

            # Extract base module name
            if import_type == 'from':
                base_module = import_name.split('.')[1] if import_name.startswith('.') else import_name.split('.')[0]
            else:
                base_module = import_name.split('.')[0] if '.' in import_name else import_name

            exists, reason = check_module_exists(base_module)

            if not exists:
                issues.append({
                    'file': filepath.name,
                    'line': lineno,
                    'import_type': import_type,
                    'import_name': import_name,
                    'error': reason
                })
                if base_module not in MISSING_MODULES:
                    MISSING_MODULES.append(base_module)

    except SyntaxError as e:
        issues.append({
            'file': filepath.name,
            'line': e.lineno,
            'import_type': 'SYNTAX_ERROR',
            'import_name': 'N/A',
            'error': f"Syntax error: {e.msg}"
        })
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        issues.append({
            'file': filepath.name,
            'line': 0,
            'import_type': 'PARSE_ERROR',
            'import_name': 'N/A',
            'error': str(e)[:100]
        })

    return issues


def main():
    print("=" * 80)
    print("OPENEVOLVE FRONTEND - ROOT IMPORT CHECKER")
    print("=" * 80)
    print()

    # Only check root-level Python files
    root_files = [f for f in Path('.').glob('*.py') if f.name != 'check_root_imports.py']

    print(f"Checking {len(root_files)} root-level Python files\n")

    all_issues = {}
    for filepath in root_files:
        issues = check_file(filepath)
        if issues:
            all_issues[filepath.name] = issues

    if all_issues:
        print(f"\n{'='*80}")
        print(f"BROKEN IMPORTS FOUND IN {len(all_issues)} FILES")
        print(f"{'='*80}\n")

        for filename, issues in sorted(all_issues.items()):
            print(f"\n{filename}:")
            for issue in issues:
                print(f"  Line {issue['line']}: {issue['import_type']} {issue['import_name']}")
                print(f"    Error: {issue['error']}")

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Files with issues: {len(all_issues)}")
        print(f"Total broken imports: {sum(len(issues) for issues in all_issues.values())}")

        if MISSING_MODULES:
            print(f"\nMISSING UNIQUE MODULES ({len(set(MISSING_MODULES))}):")
            for module in sorted(set(MISSING_MODULES)):
                print(f"  - {module}")

        return 1
    else:
        print("\n" + "="*80)
        print("NO BROKEN IMPORTS IN ROOT FILES!")
        print("="*80)
        return 0


if __name__ == '__main__':
    sys.exit(main())
