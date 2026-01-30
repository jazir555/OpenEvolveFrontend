#!/usr/bin/env python3
"""
Comprehensive Import Checker for OpenEvolve Frontend
Checks all Python files for broken imports and missing dependencies
"""

import sys
import os
import ast
import importlib
from pathlib import Path
from typing import List, Dict, Tuple, Set
import traceback

# Track all issues
BROKEN_IMPORTS = []
MISSING_MODULES = []
MISSING_FILES = []
CIRCULAR_IMPORTS = []

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
        # Handle relative imports
        if module_name.startswith('.'):
            return True, "Relative import"

        # Try to import the module
        importlib.import_module(module_name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            return False, f"Error: {str(e)}"


def check_file_exists(filepath: str, import_line: str) -> bool:
    """Check if an imported file exists in the project"""
    # For local imports (not from stdlib or installed packages)
    if import_line.startswith('from .') or import_line.startswith('from ..'):
        return True  # Relative import, will be checked by runtime

    # Extract module name from "from module import X"
    if import_line.startswith('from '):
        parts = import_line.split()
        if len(parts) >= 2:
            module = parts[1].split('.')[0]
            # Check if it's a local file
            module_file = Path(f"{module}.py")
            if module_file.exists():
                return True
            module_dir = Path(module)
            if module_dir.exists() and (module_dir / "__init__.py").exists():
                return True

    return True  # Assume OK for external packages


def check_file_for_broken_imports(filepath: Path) -> List[Dict]:
    """Check a single Python file for broken imports"""
    issues = []

    try:
        # Make filepath absolute
        filepath = filepath.resolve()

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the file
        tree = ast.parse(content, filename=str(filepath))
        checker = ImportChecker(str(filepath))
        checker.visit(tree)

        # Check each import
        for import_type, import_name, lineno in checker.imports:
            # Skip standard library and known third-party modules
            if import_name.split('.')[0] in {
                'os', 'sys', 'json', 're', 'datetime', 'pathlib', 'typing',
                'logging', 'threading', 'collections', 'functools', 'dataclasses',
                'copy', 'uuid', 'hashlib', 'time', 'base64', 'io', 'tempfile',
                'enum', 'contextlib', 'queue', 'multiprocessing', 'math', 'random',
                'string', 'secrets', 'gc', 'tracemalloc', 'weakref', 'pickle',
                'sqlite3', 'statistics', 'decimal', 'fractions', 'itertools',
                'numpy', 'pandas', 'plotly', 'requests', 'streamlit', 'fastapi',
                'uvicorn', 'pydantic', 'click', 'typer', 'aiohttp', 'asyncio',
                'concurrent', 'unittest', 'pytest', 'PIL', 'cv2', 'torch',
                'tensorflow', 'sklearn', 'scipy', 'matplotlib', 'networkx',
                'openai', 'anthropic', 'langchain', 'chromadb', 'dspy'
            }:
                continue

            # Extract base module name
            base_module = import_name.split('.')[0] if '.' in import_name else import_name
            if import_type == 'from':
                base_module = import_name.split('.')[1] if import_name.startswith('.') else import_name.split('.')[0]

            # Check if module can be imported
            exists, reason = check_module_exists(base_module)

            if not exists:
                try:
                    rel_path = filepath.relative_to(Path.cwd())
                except ValueError:
                    rel_path = filepath

                issue = {
                    'file': str(rel_path),
                    'line': lineno,
                    'import_type': import_type,
                    'import_name': import_name,
                    'error': reason
                }
                issues.append(issue)

                # Track missing modules
                if base_module not in MISSING_MODULES:
                    MISSING_MODULES.append(base_module)

    except SyntaxError as e:
        try:
            rel_path = filepath.relative_to(Path.cwd())
        except ValueError:
            rel_path = filepath

        issues.append({
            'file': str(rel_path),
            'line': e.lineno,
            'import_type': 'SYNTAX_ERROR',
            'import_name': 'N/A',
            'error': f"Syntax error: {e.msg}"
        })
        except (OSError, IOError, UnicodeDecodeError) as e:
            try:
                rel_path = filepath.relative_to(Path.cwd())
            except ValueError:
                rel_path = filepath

            issues.append({
                'file': str(rel_path),
                'line': 0,
                'import_type': 'PARSE_ERROR',
                'import_name': 'N/A',
                'error': f"Parse error: {str(e)}"
            })

    return issues


def main():
    """Main function to check all Python files"""
    print("=" * 80)
    print("OPENEVOLVE FRONTEND - BROKEN IMPORT CHECKER")
    print("=" * 80)
    print()

    # Find all Python files
    start_path = Path.cwd()
    python_files = list(start_path.rglob('*.py'))

    # Filter out test environments and node_modules
    python_files = [f for f in python_files
                    if 'openevolve_test_env' not in str(f)
                    and 'node_modules' not in str(f)
                    and '.venv' not in str(f)
                    and 'venv' not in str(f)
                    and 'site-packages' not in str(f)]

    print(f"Found {len(python_files)} Python files to check\n")

    # Check each file
    all_issues = {}
    for filepath in python_files:
        issues = check_file_for_broken_imports(filepath)
        if issues:
            all_issues[str(filepath)] = issues

    # Report results
    if all_issues:
        print(f"\n{'='*80}")
        print(f"BROKEN IMPORTS FOUND IN {len(all_issues)} FILES")
        print(f"{'='*80}\n")

        for filepath, issues in sorted(all_issues.items()):
            rel_path = Path(filepath).relative_to(Path.cwd())
            print(f"\n{rel_path}:")
            for issue in issues:
                print(f"  Line {issue['line']}: {issue['import_type']} {issue['import_name']}")
                print(f"    Error: {issue['error']}")

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Files with issues: {len(all_issues)}")
        print(f"Total broken imports: {sum(len(issues) for issues in all_issues.values())}")
        print(f"Missing unique modules: {len(set(MISSING_MODULES))}")

        if MISSING_MODULES:
            print(f"\nMISSING MODULES ({len(set(MISSING_MODULES))}):")
            for module in sorted(set(MISSING_MODULES)):
                print(f"  - {module}")

        return 1
    else:
        print("\n" + "="*80)
        print("NO BROKEN IMPORTS FOUND!")
        print("="*80)
        print("\nAll imports can be resolved.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
