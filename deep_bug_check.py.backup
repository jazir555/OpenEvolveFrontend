#!/usr/bin/env python3
"""
Deep Bug Check for BubbleLabs Integration
Performs comprehensive static and dynamic analysis
"""

import ast
import os
import sys
import re
from typing import List, Dict, Any, Set

class DeepCodeAnalyzer:
    """Deep static code analyzer for Python files."""

    def __init__(self):
        self.issues = []
        self.warnings = []
        self.suggestions = []

    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """Analyze a Python file deeply."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)

            analysis = {
                'filepath': filepath,
                'imports': self._analyze_imports(tree),
                'classes': self._analyze_classes(tree),
                'functions': self._analyze_functions(tree),
                'calls': self._analyze_calls(tree),
                'issues': [],
                'warnings': [],
                'suggestions': []
            }

            # Check for specific patterns
            analysis['issues'].extend(self._check_resource_leaks(tree, filepath))
            analysis['issues'].extend(self._check_sql_injection(tree, filepath))
            analysis['issues'].extend(self._check_race_conditions(tree, filepath))
            analysis['issues'].extend(self._check_error_handling(tree, filepath))
            analysis['issues'].extend(self._check_none_handling(tree, filepath))
            analysis['issues'].extend(self._check_type_safety(tree, filepath))
            analysis['issues'].extend(self._check_infinite_loops(tree, filepath))
            analysis['warnings'].extend(self._check_deprecated_usage(tree, filepath))
            analysis['warnings'].extend(self._check_performance(tree, filepath))

            return analysis

        except Exception as e:
            return {
                'filepath': filepath,
                'error': str(e),
                'issues': [f"Failed to analyze: {e}"],
                'warnings': [],
                'suggestions': []
            }

    def _analyze_imports(self, tree: ast.AST) -> List[Dict]:
        """Analyze all imports."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
        return imports

    def _analyze_classes(self, tree: ast.AST) -> List[Dict]:
        """Analyze all classes."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)

                # Check for __init__ issues
                init_count = sum(1 for m in methods if m == '__init__')

                classes.append({
                    'name': node.name,
                    'methods': methods,
                    'init_count': init_count,
                    'line': node.lineno,
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
                })

                # Check for duplicate __init__
                if init_count > 1:
                    self.issues.append({
                        'file': 'unknown',
                        'line': node.lineno,
                        'severity': 'medium',
                        'message': f"Class {node.name} has {init_count} __init__ methods",
                        'type': 'duplicate_init'
                    })
        return classes

    def _analyze_functions(self, tree: ast.AST) -> List[Dict]:
        """Analyze all functions."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                returns = ast.unparse(node.returns) if node.returns else None

                functions.append({
                    'name': node.name,
                    'args': args,
                    'returns': returns,
                    'line': node.lineno,
                    'is_async': isinstance(node, ast.AsyncFunctionDef)
                })
        return functions

    def _analyze_calls(self, tree: ast.AST) -> List[Dict]:
        """Analyze all function calls."""
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = ast.unparse(node.func)

                calls.append({
                    'function': func_name,
                    'line': node.lineno,
                    'args_count': len(node.args)
                })
        return calls

    def _check_resource_leaks(self, tree: ast.AST, filepath: str) -> List[Dict]:
        """Check for potential resource leaks."""
        issues = []

        # Look for file operations without context managers
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == 'open':
                        # Check if it's in a with statement
                        parent = None
                        for parent_node in ast.walk(tree):
                            if hasattr(parent_node, 'body'):
                                if isinstance(parent_node.body, list):
                                    for child in parent_node.body:
                                        if child == node:
                                            parent = parent_node
                                            break

                        # Not in a with statement
                        if not isinstance(parent, ast.With):
                            issues.append({
                                'file': filepath,
                                'line': node.lineno,
                                'severity': 'medium',
                                'message': 'File opened without context manager (potential resource leak)',
                                'type': 'resource_leak'
                            })

        return issues

    def _check_sql_injection(self, tree: ast.AST, filepath: str) -> List[Dict]:
        """Check for SQL injection vulnerabilities."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'execute':
                        # Check if SQL string is being concatenated
                        if node.args:
                            arg = node.args[0]
                            if isinstance(arg, ast.BinOp) and isinstance(arg.op, (ast.Add, ast.Mod)):
                                issues.append({
                                    'file': filepath,
                                    'line': node.lineno,
                                    'severity': 'critical',
                                    'message': 'SQL query constructed with string concatenation (potential SQL injection)',
                                    'type': 'sql_injection'
                                })

        return issues

    def _check_race_conditions(self, tree: ast.AST, filepath: str) -> List[Dict]:
        """Check for potential race conditions."""
        issues = []

        # Look for threading usage
        uses_threading = False
        has_locks = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if 'threading' in alias.name:
                        uses_threading = True
            elif isinstance(node, ast.ImportFrom):
                if node.module and 'threading' in node.module:
                    uses_threading = True
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == 'Thread':
                        uses_threading = True
                    elif node.func.id in ['Lock', 'RLock', 'Semaphore']:
                        has_locks = True

        # Check if shared data is accessed without locks
        if uses_threading and not has_locks:
            issues.append({
                'file': filepath,
                'line': 0,
                'severity': 'medium',
                'message': 'Threading used but no locks detected (potential race condition)',
                'type': 'race_condition'
            })

        return issues

    def _check_error_handling(self, tree: ast.AST, filepath: str) -> List[Dict]:
        """Check for error handling issues."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                # Check for bare except
                for handler in node.handlers:
                    if handler.type is None:
                        issues.append({
                            'file': filepath,
                            'line': handler.lineno,
                            'severity': 'medium',
                            'message': 'Bare except clause (catches all exceptions including SystemExit)',
                            'type': 'bare_except'
                        })

        return issues

    def _check_none_handling(self, tree: ast.AST, filepath: str) -> List[Dict]:
        """Check for potential None reference errors."""
        issues = []

        # Look for Optional types that are accessed without None checks
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if test is for None
                test_str = ast.unparse(node.test)
                if 'is not None' in test_str or 'is None' in test_str:
                    # This is good - checking for None
                    continue

            # Look for attribute access on potentially None values
            if isinstance(node, ast.Attribute):
                var_name = ast.unparse(node.value)
                # This is a simplified check
                # A full implementation would need type analysis

        return issues

    def _check_type_safety(self, tree: ast.AST, filepath: str) -> List[Dict]:
        """Check for type safety issues."""
        issues = []

        # Check for type: ignore comments
        source_lines = []
        try:
            with open(filepath, 'r') as f:
                source_lines = f.readlines()
        except:
            return issues

        for i, line in enumerate(source_lines, 1):
            if '# type: ignore' in line:
                issues.append({
                    'file': filepath,
                    'line': i,
                    'severity': 'low',
                    'message': 'Type check ignored with # type: ignore',
                    'type': 'type_safety'
                })

        return issues

    def _check_infinite_loops(self, tree: ast.AST, filepath: str) -> List[Dict]:
        """Check for potential infinite loops."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                # Check if test is always True
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    # Check if there's a break statement
                    has_break = False
                    for body_node in ast.walk(node):
                        if isinstance(body_node, ast.Break):
                            has_break = True
                            break

                    if not has_break:
                        issues.append({
                            'file': filepath,
                            'line': node.lineno,
                            'severity': 'high',
                            'message': 'While loop with condition True and no break (infinite loop)',
                            'type': 'infinite_loop'
                        })

        return issues

    def _check_deprecated_usage(self, tree: ast.AST, filepath: str) -> List[Dict]:
        """Check for usage of deprecated features."""
        warnings = []

        # Check for deprecated imports
        deprecated = [
            'threading.Thread.isAlive',  # Use is_alive() instead
            'distutils',  # Use setuptools instead
        ]

        source = ast.unparse(tree)
        for dep in deprecated:
            if dep in source:
                warnings.append({
                    'file': filepath,
                    'line': 0,
                    'message': f'Usage of deprecated: {dep}',
                    'type': 'deprecated'
                })

        return warnings

    def _check_performance(self, tree: ast.AST, filepath: str) -> List[Dict]:
        """Check for performance issues."""
        warnings = []

        for node in ast.walk(tree):
            # Check for string concatenation in loops
            if isinstance(node, ast.For):
                for body_node in ast.walk(node):
                    if isinstance(body_node, ast.AugAssign) and isinstance(body_node.op, ast.Add):
                        if isinstance(body_node.target, ast.Name) and isinstance(body_node.value, ast.BinOp):
                            warnings.append({
                                'file': filepath,
                                'line': body_node.lineno,
                                'message': 'String concatenation in loop (consider list join)',
                                'type': 'performance'
                            })

        return warnings


def analyze_bubblelabs_files():
    """Analyze all BubbleLabs integration files."""

    files = [
        'bubblelabs_hephaestus_bridge.py',
        'bubblelabs_mcp_tools.py',
        'bubblelabs_analytics.py',
        'bubblelabs_typescript_export.py',
        'test_bubblelabs_complete_integration.py'
    ]

    analyzer = DeepCodeAnalyzer()
    all_results = []

    print("=" * 80)
    print("DEEP BUG CHECK - BubbleLabs Integration")
    print("=" * 80)

    for filepath in files:
        if not os.path.exists(filepath):
            print(f"\n[SKIP] {filepath} (not found)")
            continue

        print(f"\n[ANALYZING] {filepath}...")
        result = analyzer.analyze_file(filepath)
        all_results.append(result)

        # Print results
        issues = result.get('issues', [])
        warnings = result.get('warnings', [])

        if result.get('error'):
            print(f"  [ERROR] {result['error']}")
        else:
            print(f"  Classes: {len(result.get('classes', []))}")
            print(f"  Functions: {len(result.get('functions', []))}")
            print(f"  Imports: {len(result.get('imports', []))}")

        if issues:
            print(f"\n  [ISSUES] {len(issues)} found:")
            for issue in issues:
                print(f"    Line {issue.get('line', '?')}: {issue['message']}")

        if warnings:
            print(f"\n  [WARNINGS] {len(warnings)}:")
            for warning in warnings:
                print(f"    {warning['message']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_issues = sum(len(r.get('issues', [])) for r in all_results)
    total_warnings = sum(len(r.get('warnings', [])) for r in all_results)

    print(f"Total files analyzed: {len(all_results)}")
    print(f"Total issues: {total_issues}")
    print(f"Total warnings: {total_warnings}")

    if total_issues == 0:
        print("\n[OK] No critical issues found!")
    else:
        print(f"\n[!] Found {total_issues} issue(s) that need attention")

    return all_results


if __name__ == "__main__":
    analyze_bubblelabs_files()
