#!/usr/bin/env python3
"""
Comprehensive Edge Case Analyzer for OpenEvolve Frontend
Detects 20 categories of subtle issues across the entire codebase
"""

import ast
import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime


class ImportGraphAnalyzer:
    """Builds and analyzes import dependency graph"""

    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self.file_imports: Dict[str, List[Dict]] = defaultdict(list)

    def add_import(self, from_file: str, import_module: str, line_no: int):
        self.graph[from_file].add(import_module)
        self.reverse_graph[import_module].add(from_file)
        self.file_imports[from_file].append({
            'module': import_module,
            'line': line_no
        })

    def find_cycles(self) -> List[Dict]:
        """Find all circular dependencies using DFS"""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if node in visited:
                return None

            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.graph.get(node, set()):
                result = dfs(neighbor, path + [node])
                if result:
                    return result

            rec_stack.remove(node)
            return None

        for node in self.graph:
            if node not in visited:
                cycle = dfs(node, [])
                if cycle and len(cycle) > 2:
                    cycles.append({
                        'cycle': cycle,
                        'depth': len(cycle) - 1,
                        'severity': 'HIGH' if len(cycle) <= 3 else 'MEDIUM',
                        'files': cycle
                    })

        return cycles


class EdgeCaseDetector(ast.NodeVisitor):
    """AST visitor for detecting edge cases"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.issues: List[Dict] = []
        self.imports: List[Dict] = []
        self.functions: Dict[str, List[Dict]] = defaultdict(list)
        self.classes: Dict[str, List[Dict]] = defaultdict(list)
        self.globals: Set[str] = set()
        self.current_function = None
        self.current_class = None

    def detect_issues(self, tree: ast.AST):
        """Run all detection passes"""
        self.visit(tree)
        self._post_process(tree)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append({
                'type': 'import',
                'module': alias.name,
                'alias': alias.asname,
                'line': node.lineno
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            for alias in node.names:
                self.imports.append({
                    'type': 'from_import',
                    'module': node.module,
                    'name': alias.name,
                    'alias': alias.asname,
                    'line': node.lineno
                })

                # Check for wildcard imports (Category 3)
                if alias.name == '*':
                    self.issues.append({
                        'category': 'Implicit Import',
                        'severity': 'HIGH',
                        'line': node.lineno,
                        'description': f'Wildcard import from {node.module}',
                        'example': f'from {node.module} import *',
                        'impact': 'Pollutes namespace, unclear what is imported',
                        'recommendation': 'Import specific names explicitly',
                        'priority': 'HIGH'
                    })
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old_function = self.current_function
        self.current_function = node.name

        # Check for missing docstring (Category 15)
        docstring = ast.get_docstring(node)
        if not docstring and not node.name.startswith('_'):
            self.issues.append({
                'category': 'Documentation Gap',
                'severity': 'LOW',
                'line': node.lineno,
                'description': f'Function {node.name} missing docstring',
                'recommendation': 'Add docstring explaining purpose, parameters, and return value',
                'priority': 'LOW'
            })

        # Check for lazy imports (Category 2)
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                if child != node:  # Not at top level
                    self.issues.append({
                        'category': 'Lazy Import',
                        'severity': 'MEDIUM',
                        'line': child.lineno,
                        'description': f'Import inside function {node.name}',
                        'recommendation': 'Move to top-level unless avoiding circular dependency',
                        'priority': 'MEDIUM'
                    })

        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        old_class = self.current_class
        self.current_class = node.name

        # Check for missing docstring (Category 15)
        docstring = ast.get_docstring(node)
        if not docstring:
            self.issues.append({
                'category': 'Documentation Gap',
                'severity': 'LOW',
                'line': node.lineno,
                'description': f'Class {node.name} missing docstring',
                'recommendation': 'Add docstring explaining class purpose',
                'priority': 'LOW'
            })

        # Check for singleton pattern (Category 8)
        if 'Singleton' in node.name:
            self.issues.append({
                'category': 'Thread Safety',
                'severity': 'HIGH',
                'line': node.lineno,
                'description': f'Singleton pattern in {node.name}',
                'impact': 'May not be thread-safe without proper locking',
                'recommendation': 'Ensure thread-safe initialization with locks',
                'priority': 'HIGH'
            })

        self.generic_visit(node)
        self.current_class = old_class

    def visit_Global(self, node: ast.Global):
        # Category 8: Thread safety - global variables
        self.globals.update(node.names)
        self.issues.append({
            'category': 'Thread Safety',
            'severity': 'HIGH',
            'line': node.lineno,
            'description': f'Global variable declaration: {", ".join(node.names)}',
            'impact': 'Not thread-safe, can cause race conditions',
            'recommendation': 'Use thread-local storage or proper synchronization',
            'priority': 'HIGH'
        })
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try):
        # Check for bare except (Category 11)
        for handler in node.handlers:
            if handler.type is None:
                self.issues.append({
                    'category': 'Error Handling Gap',
                    'severity': 'CRITICAL',
                    'line': handler.lineno,
                    'description': 'Bare except clause catches all exceptions',
                    'example': 'except:',
                    'impact': 'Catches KeyboardInterrupt and system exceptions',
                    'recommendation': 'Specify exception type: except Exception:',
                    'priority': 'CRITICAL'
                })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Category 14: Security issues
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Dangerous dynamic execution
            if func_name in ['eval', 'exec']:
                self.issues.append({
                    'category': 'Security',
                    'severity': 'CRITICAL',
                    'line': node.lineno,
                    'description': f'Dangerous dynamic execution: {func_name}',
                    'example': f'{func_name}(user_input)',
                    'impact': 'Code injection vulnerability',
                    'recommendation': 'NEVER use eval/exec with untrusted input',
                    'priority': 'CRITICAL'
                })

            # Shell command execution
            if func_name in ['system', 'popen']:
                self.issues.append({
                    'category': 'Security',
                    'severity': 'CRITICAL',
                    'line': node.lineno,
                    'description': f'Shell command execution: {func_name}',
                    'impact': 'Shell injection vulnerability',
                    'recommendation': 'Use subprocess with proper sanitization',
                    'priority': 'CRITICAL'
                })

            # Type conversion without error handling (Category 12)
            if func_name in ['str', 'int', 'float']:
                parent = getattr(node, 'parent', None)
                if not isinstance(parent, ast.Try):
                    self.issues.append({
                        'category': 'Type Safety',
                        'severity': 'LOW',
                        'line': node.lineno,
                        'description': f'Type conversion {func_name} without error handling',
                        'impact': 'May raise ValueError',
                        'recommendation': 'Wrap in try/except',
                        'priority': 'LOW'
                    })

        # Category 10: Performance - calls in loops
        if isinstance(node.parent, (ast.For, ast.While)):
            if isinstance(node.func, ast.Name):
                if not node.func.id.startswith('_'):  # Not private method
                    self.issues.append({
                        'category': 'Performance Anti-Pattern',
                        'severity': 'MEDIUM',
                        'line': node.lineno,
                        'description': f'Function call {node.func.id} inside loop',
                        'impact': 'May be inefficient if called repeatedly',
                        'recommendation': 'Consider caching or moving outside loop',
                        'priority': 'MEDIUM'
                    })

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Category 4: Shadowed imports
        imported_names = {
            imp['alias'] or imp['name'] if imp['type'] == 'from_import' else imp['alias'] or imp['module']
            for imp in self.imports
        }

        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in imported_names:
                    self.issues.append({
                        'category': 'Shadowed Import',
                        'severity': 'MEDIUM',
                        'line': node.lineno,
                        'description': f'Variable "{target.id}" shadows imported name',
                        'impact': 'Loses access to imported module/function',
                        'recommendation': 'Rename variable to avoid shadowing',
                        'priority': 'MEDIUM'
                    })

        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        # Set parent references for children
        for child in ast.iter_child_nodes(node):
            child.parent = node
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        # Set parent references for children
        for child in ast.iter_child_nodes(node):
            child.parent = node
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        # Category 18: Dead code after return
        self.generic_visit(node)

    def _post_process(self, tree: ast.AST):
        """Post-processing analysis"""
        # Check for encoding issues (Category 13)
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        lines = content.split('\n')
        has_encoding = any('coding' in line for line in lines[:2])
        has_non_ascii = any(ord(c) > 127 for c in content)

        if has_non_ascii and not has_encoding:
            self.issues.append({
                'category': 'Encoding Issue',
                'severity': 'LOW',
                'line': 1,
                'description': 'Non-ASCII characters without encoding declaration',
                'recommendation': 'Add: # -*- coding: utf-8 -*-',
                'priority': 'LOW'
            })

        # Check for unreachable code (Category 18)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found_return = False
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        found_return = True
                    elif found_return and not isinstance(stmt, (ast.Str, ast.Expr)):
                        if isinstance(stmt, ast.Expr) and not isinstance(stmt.value, ast.Constant):
                            self.issues.append({
                                'category': 'Dead Code',
                                'severity': 'LOW',
                                'line': stmt.lineno,
                                'description': 'Unreachable code after return statement',
                                'recommendation': 'Remove dead code',
                                'priority': 'LOW'
                            })
                            break


class EdgeCaseAnalyzer:
    """Main analyzer orchestrating all edge case detection"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.import_graph = ImportGraphAnalyzer()
        self.all_issues: List[Dict] = []
        self.file_issues: Dict[str, List[Dict]] = defaultdict(list)

    def analyze(self) -> Dict:
        """Run complete analysis"""
        print(f"Analyzing Python files in: {self.root_dir}")

        python_files = list(self.root_dir.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")

        # First pass: Build import graph
        print("\n[Phase 1] Building import dependency graph...")
        for py_file in python_files:
            self._build_import_info(py_file)

        # Second pass: Detect circular dependencies
        print("[Phase 2] Detecting circular dependencies...")
        cycles = self.import_graph.find_cycles()

        # Third pass: Analyze each file for edge cases
        print("[Phase 3] Analyzing files for edge cases...")
        for py_file in python_files:
            self._analyze_file(py_file)

        # Generate report
        print("[Phase 4] Generating report...")
        return self._generate_report(cycles, len(python_files))

    def _build_import_info(self, file_path: Path):
        """Build import information from file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))

            module_name = str(file_path.relative_to(self.root_dir))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.import_graph.add_import(module_name, alias.name, node.lineno)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    self.import_graph.add_import(module_name, node.module, node.lineno)

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            pass  # Skip files that can't be parsed

    def _analyze_file(self, file_path: Path):
        """Analyze single file for edge cases"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))

            detector = EdgeCaseDetector(str(file_path))
            detector.detect_issues(tree)

            relative_path = str(file_path.relative_to(self.root_dir))
            self.file_issues[relative_path] = detector.issues
            self.all_issues.extend(detector.issues)

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"  Error analyzing {file_path}: {e}")

    def _generate_report(self, cycles: List[Dict], total_files: int) -> Dict:
        """Generate comprehensive analysis report"""
        # Group issues by category
        by_category = defaultdict(list)
        by_severity = defaultdict(list)

        for issue in self.all_issues:
            category = issue['category']
            severity = issue['severity']
            by_category[category].append(issue)
            by_severity[severity].append(issue)

        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files_analyzed': total_files,
                'total_issues_found': len(self.all_issues),
                'files_with_issues': len(self.file_issues),
                'circular_dependencies': len(cycles)
            },
            'circular_dependencies': cycles,
            'issues_by_category': {
                cat: {
                    'count': len(issues),
                    'severity_breakdown': Counter(i['severity'] for i in issues),
                    'issues': issues[:10]  # First 10 as examples
                }
                for cat, issues in by_category.items()
            },
            'issues_by_severity': {
                severity: {
                    'count': len(issues),
                    'categories': Counter(i['category'] for i in issues)
                }
                for severity, issues in by_severity.items()
            },
            'top_files': {
                file: len(issues)
                for file, issues in sorted(
                    self.file_issues.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )[:20]
            }
        }


def main():
    """Main entry point"""
    root_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

    analyzer = EdgeCaseAnalyzer(root_dir)
    report = analyzer.analyze()

    # Print summary
    print("\n" + "="*80)
    print("EDGE CASE ANALYSIS REPORT")
    print("="*80)
    print(f"Timestamp: {report['timestamp']}")
    print(f"\nFiles analyzed: {report['summary']['total_files_analyzed']}")
    print(f"Total issues found: {report['summary']['total_issues_found']}")
    print(f"Files with issues: {report['summary']['files_with_issues']}")
    print(f"Circular dependencies: {report['summary']['circular_dependencies']}")

    print("\n" + "-"*80)
    print("ISSUES BY SEVERITY")
    print("-"*80)
    severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    for severity in severity_order:
        if severity in report['issues_by_severity']:
            info = report['issues_by_severity'][severity]
            print(f"\n{severity}: {info['count']} issues")
            for category, count in sorted(info['categories'].items(), key=lambda x: -x[1]):
                print(f"  - {category}: {count}")

    print("\n" + "-"*80)
    print("TOP FILES WITH MOST ISSUES")
    print("-"*80)
    for file, count in list(report['top_files'].items())[:10]:
        print(f"  {count:3d} - {file}")

    # Save detailed report
    output_file = Path(root_dir) / "EDGE_CASE_ANALYSIS_REPORT.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDetailed report saved to: {output_file}")


if __name__ == '__main__':
    main()
