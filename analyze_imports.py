#!/usr/bin/env python3
"""
Comprehensive import analysis for RESE codebase.
Identifies all import statements and creates dependency graph.
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import json


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to extract import information."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.imports = []
        self.from_imports = []
        self.relative_imports = []

    def visit_Import(self, node: ast.Import):
        """Handle 'import X' statements."""
        for alias in node.names:
            self.imports.append({
                'module': alias.name,
                'alias': alias.asname,
                'line': node.lineno
            })

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Handle 'from X import Y' statements."""
        module = node.module or ''
        level = node.level  # Number of dots for relative imports

        import_info = {
            'module': module,
            'level': level,
            'names': [alias.name for alias in node.names],
            'line': node.lineno
        }

        if level > 0:
            self.relative_imports.append(import_info)
        else:
            self.from_imports.append(import_info)


def analyze_file(filepath: str) -> dict:
    """Analyze a single Python file for imports."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=filepath)
        analyzer = ImportAnalyzer(filepath)
        analyzer.visit(tree)

        # Get module name from filepath
        rel_path = os.path.relpath(filepath)
        module_name = rel_path.replace('.py', '').replace(os.sep, '.')

        return {
            'filepath': filepath,
            'module': module_name,
            'imports': analyzer.imports,
            'from_imports': analyzer.from_imports,
            'relative_imports': analyzer.relative_imports,
            'has_syntax_error': False
        }
    except SyntaxError as e:
        return {
            'filepath': filepath,
            'module': filepath,
            'imports': [],
            'from_imports': [],
            'relative_imports': [],
            'has_syntax_error': True,
            'error': str(e)
        }
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        return {
            'filepath': filepath,
            'module': filepath,
            'imports': [],
            'from_imports': [],
            'relative_imports': [],
            'has_syntax_error': True,
            'error': str(e)
        }


def build_dependency_graph(files_data: List[dict]) -> Dict:
    """Build dependency graph from analyzed files."""
    graph = defaultdict(set)
    module_to_file = {}

    # Map modules to files
    for file_data in files_data:
        if not file_data['has_syntax_error']:
            module_to_file[file_data['module']] = file_data['filepath']

    # Build dependencies
    for file_data in files_data:
        if file_data['has_syntax_error']:
            continue

        current_module = file_data['module']

        # Process from_imports
        for imp in file_data['from_imports']:
            if imp['module']:
                # Check if it's a relative import
                if imp['level'] > 0:
                    # Resolve relative import
                    base_path = os.path.dirname(file_data['filepath'])
                    parts = current_module.split('.')
                    for _ in range(imp['level']):
                        parts.pop()
                    base = '.'.join(parts)
                    target_module = f"{base}.{imp['module']}" if imp['module'] else base
                else:
                    target_module = imp['module']

                # Check if it's internal (rese.*)
                if target_module.startswith('rese') or target_module.startswith('.'):
                    graph[current_module].add(target_module)

        # Process regular imports
        for imp in file_data['imports']:
            if imp['module'].startswith('rese'):
                graph[current_module].add(imp['module'])

    return dict(graph), module_to_file


def detect_circular_dependencies(graph: Dict) -> List[List[str]]:
    """Detect circular dependencies using DFS."""
    cycles = []
    visited = set()
    rec_stack = set()
    path = []

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
                return True

        path.pop()
        rec_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            dfs(node)

    return cycles


def check_missing_modules(files_data: List[dict]) -> Dict[str, List[str]]:
    """Check for missing internal modules."""
    missing = defaultdict(list)
    defined_modules = set()

    # Get all defined modules
    for file_data in files_data:
        if not file_data['has_syntax_error']:
            defined_modules.add(file_data['module'])

            # Also add __init__ as potential module
            if file_data['filepath'].endswith('__init__.py'):
                parent = file_data['module'].rsplit('.', 1)[0]
                defined_modules.add(parent)

    # Check imports
    for file_data in files_data:
        if file_data['has_syntax_error']:
            continue

        for imp in file_data['from_imports']:
            if imp['module'] and imp['module'].startswith('rese'):
                if imp['module'] not in defined_modules:
                    missing[file_data['filepath']].append(imp['module'])

        for imp in file_data['imports']:
            if imp['module'].startswith('rese'):
                if imp['module'] not in defined_modules:
                    missing[file_data['filepath']].append(imp['module'])

    return dict(missing)


def find_all_python_files(root_dir: str) -> List[str]:
    """Find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip __pycache__ and test directories if needed
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.pytest_cache', '.git']]

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return python_files


def main():
    """Main analysis function."""
    rese_dir = 'C:/Users/mmeadow/Documents/OpenEvolve/Frontend/rese'

    print("=" * 80)
    print("RESE IMPORT ANALYSIS")
    print("=" * 80)

    # Find all Python files
    print("\n[1/6] Finding all Python files...")
    python_files = find_all_python_files(rese_dir)
    print(f"Found {len(python_files)} Python files")

    # Analyze each file
    print("\n[2/6] Analyzing imports...")
    files_data = []
    for filepath in python_files:
        data = analyze_file(filepath)
        files_data.append(data)

    # Count errors
    syntax_errors = [f for f in files_data if f['has_syntax_error']]
    if syntax_errors:
        print(f"\nWARNING: {len(syntax_errors)} files have syntax errors:")
        for f in syntax_errors:
            print(f"  - {f['filepath']}: {f.get('error', 'Unknown error')}")

    # Build dependency graph
    print("\n[3/6] Building dependency graph...")
    graph, module_to_file = build_dependency_graph(files_data)
    print(f"Built graph with {len(graph)} modules")

    # Detect cycles
    print("\n[4/6] Detecting circular dependencies...")
    cycles = detect_circular_dependencies(graph)
    if cycles:
        print(f"WARNING: Found {len(cycles)} circular dependencies:")
        for i, cycle in enumerate(cycles, 1):
            print(f"\n  Cycle {i}:")
            print("  -> " + "\n  -> ".join(cycle))
    else:
        print("OK: No circular dependencies detected")

    # Check missing modules
    print("\n[5/6] Checking for missing modules...")
    missing = check_missing_modules(files_data)
    if missing:
        print(f"WARNING: Found {len(missing)} files importing missing modules:")
        for filepath, modules in missing.items():
            print(f"\n  {filepath}:")
            for module in modules:
                print(f"    - {module}")
    else:
        print("OK: All internal modules resolved")

    # Generate report
    print("\n[6/6] Generating report...")

    report = {
        'summary': {
            'total_files': len(python_files),
            'syntax_errors': len(syntax_errors),
            'circular_dependencies': len(cycles),
            'missing_imports': len(missing),
            'total_modules': len(graph)
        },
        'dependency_graph': graph,
        'circular_dependencies': cycles,
        'missing_modules': missing,
        'files_with_errors': syntax_errors
    }

    # Save report
    output_file = 'C:/Users/mmeadow/Documents/OpenEvolve/Frontend/rese_import_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nOK: Report saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total Python files: {report['summary']['total_files']}")
    print(f"Syntax errors: {report['summary']['syntax_errors']}")
    print(f"Circular dependencies: {report['summary']['circular_dependencies']}")
    print(f"Missing module imports: {report['summary']['missing_imports']}")
    print(f"Total modules: {report['summary']['total_modules']}")

    return report


if __name__ == '__main__':
    report = main()
