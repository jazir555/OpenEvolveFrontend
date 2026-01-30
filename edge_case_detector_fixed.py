#!/usr/bin/env python3
"""
Fixed Comprehensive Edge Case Analyzer
Properly handles AST parent references
"""

__all__ = [
    'set_parent_references',
    'ImportGraphBuilder',
    'EdgeCaseAnalyzer',
    'analyze_project_edge_cases'
]

import ast
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime


def set_parent_references(node: ast.AST, parent: Optional[ast.AST] = None):
    """Recursively set parent references for all nodes"""
    if hasattr(node, 'parent'):
        return  # Already processed

    node.parent = parent
    for child in ast.iter_child_nodes(node):
        set_parent_references(child, node)


class ImportGraphBuilder(ast.NodeVisitor):
    """Build import dependency graph"""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.file_imports: Dict[str, List[Dict]] = defaultdict(list)

    def build_from_file(self, file_path: Path):
        """Extract imports from a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))
                set_parent_references(tree)
                self.visit(tree)

            module_name = str(file_path.relative_to(self.root_dir))
            return module_name
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            return None

    def visit_Import(self, node: ast.Import):
        # Will be processed by caller
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # Will be processed by caller
        self.generic_visit(node)


class CircularDependencyDetector:
    """Detect circular dependencies in import graph"""

    def __init__(self, graph: Dict[str, Set[str]]):
        self.graph = graph

    def find_all_cycles(self) -> List[Dict]:
        """Find all circular dependencies"""
        cycles = []
        visited = set()

        for node in self.graph:
            if node not in visited:
                self._dfs(node, [], [], visited, cycles)

        return cycles

    def _dfs(self, node: str, path: List[str], path_set: Set[str],
             visited: Set[str], cycles: List[Dict]):
        """DFS to find cycles"""
        if node in path_set:
            # Found cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append({
                'cycle': cycle,
                'depth': len(cycle) - 1,
                'severity': 'HIGH' if len(cycle) <= 3 else 'MEDIUM'
            })
            return

        if node in visited:
            return

        visited.add(node)
        path.append(node)
        path_set.add(node)

        for neighbor in self.graph.get(node, set()):
            self._dfs(neighbor, path, path_set, visited, cycles)

        path.pop()
        path_set.remove(node)


class EdgeCaseDetector(ast.NodeVisitor):
    """Detect edge cases in Python code"""

    def __init__(self, file_path: str, root_dir: Path):
        self.file_path = str(Path(file_path).relative_to(root_dir))
        self.issues: List[Dict] = []
        self.imports: List[Dict] = []
        self.imported_names: Set[str] = set()

    def detect_all(self, tree: ast.AST):
        """Run all edge case detections"""
        set_parent_references(tree)
        self.visit(tree)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append({
                'type': 'import',
                'module': alias.name,
                'alias': alias.asname,
                'line': node.lineno
            })
            name = alias.asname or alias.name
            self.imported_names.add(name)
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

                name = alias.asname or alias.name
                self.imported_names.add(name)

                # Check for wildcard imports (Category 3)
                if alias.name == '*':
                    self.issues.append({
                        'category': 'Implicit Import',
                        'severity': 'HIGH',
                        'line': node.lineno,
                        'file': self.file_path,
                        'description': f'Wildcard import from {node.module}',
                        'example': f'from {node.module} import *',
                        'impact': 'Pollutes namespace, unclear what is imported',
                        'recommendation': 'Import specific names explicitly',
                        'priority': 'HIGH'
                    })
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Check for missing docstring (Category 15)
        docstring = ast.get_docstring(node)
        if not docstring and not node.name.startswith('_'):
            self.issues.append({
                'category': 'Documentation Gap',
                'severity': 'LOW',
                'line': node.lineno,
                'file': self.file_path,
                'description': f'Function {node.name} missing docstring',
                'recommendation': 'Add docstring explaining purpose, parameters, and return value',
                'priority': 'LOW'
            })

        # Check for lazy imports (Category 2)
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                if child != node:
                    # Check if inside try/except (acceptable for lazy imports)
                    in_try_except = False
                    parent = child
                    while parent != node:
                        if isinstance(parent, ast.Try):
                            in_try_except = True
                            break
                        parent = getattr(parent, 'parent', None)

                    if not in_try_except:
                        self.issues.append({
                            'category': 'Lazy Import',
                            'severity': 'MEDIUM',
                            'line': child.lineno,
                            'file': self.file_path,
                            'description': f'Import inside function {node.name}',
                            'recommendation': 'Move to top-level unless avoiding circular dependency',
                            'priority': 'MEDIUM'
                        })
                break

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        # Check for missing docstring (Category 15)
        docstring = ast.get_docstring(node)
        if not docstring:
            self.issues.append({
                'category': 'Documentation Gap',
                'severity': 'LOW',
                'line': node.lineno,
                'file': self.file_path,
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
                'file': self.file_path,
                'description': f'Singleton pattern in {node.name}',
                'impact': 'May not be thread-safe without proper locking',
                'recommendation': 'Ensure thread-safe initialization with locks',
                'priority': 'HIGH'
            })

        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        # Category 8: Thread safety - global variables
        self.issues.append({
            'category': 'Thread Safety',
            'severity': 'HIGH',
            'line': node.lineno,
            'file': self.file_path,
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
                    'file': self.file_path,
                    'description': 'Bare except clause catches all exceptions',
                    'example': 'except:',
                    'impact': 'Catches KeyboardInterrupt and system exceptions',
                    'recommendation': 'Specify exception type: except Exception:',
                    'priority': 'CRITICAL'
                })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check if inside loop (Category 10: Performance)
        parent = getattr(node, 'parent', None)
        in_loop = False
        while parent:
            if isinstance(parent, (ast.For, ast.While)):
                in_loop = True
                break
            parent = getattr(parent, 'parent', None)

        # Category 14: Security issues
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Dangerous dynamic execution
            if func_name in ['eval', 'exec']:
                self.issues.append({
                    'category': 'Security',
                    'severity': 'CRITICAL',
                    'line': node.lineno,
                    'file': self.file_path,
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
                    'file': self.file_path,
                    'description': f'Shell command execution: {func_name}',
                    'impact': 'Shell injection vulnerability',
                    'recommendation': 'Use subprocess with proper sanitization',
                    'priority': 'CRITICAL'
                })

            # Type conversion without error handling (Category 12)
            if func_name in ['str', 'int', 'float']:
                # Check if wrapped in try/except
                in_try = False
                parent = getattr(node, 'parent', None)
                while parent:
                    if isinstance(parent, ast.Try):
                        in_try = True
                        break
                    parent = getattr(parent, 'parent', None)

                if not in_try:
                    self.issues.append({
                        'category': 'Type Safety',
                        'severity': 'LOW',
                        'line': node.lineno,
                        'file': self.file_path,
                        'description': f'Type conversion {func_name} without error handling',
                        'impact': 'May raise ValueError',
                        'recommendation': 'Wrap in try/except',
                        'priority': 'LOW'
                    })

            # Function call in loop (Category 10)
            if in_loop and not func_name.startswith('_'):
                self.issues.append({
                    'category': 'Performance Anti-Pattern',
                    'severity': 'MEDIUM',
                    'line': node.lineno,
                    'file': self.file_path,
                    'description': f'Function call {func_name} inside loop',
                    'impact': 'May be inefficient if called repeatedly',
                    'recommendation': 'Consider caching or moving outside loop',
                    'priority': 'MEDIUM'
                })

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Category 4: Shadowed imports
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in self.imported_names:
                    self.issues.append({
                        'category': 'Shadowed Import',
                        'severity': 'MEDIUM',
                        'line': node.lineno,
                        'file': self.file_path,
                        'description': f'Variable "{target.id}" shadows imported name',
                        'impact': 'Loses access to imported module/function',
                        'recommendation': 'Rename variable to avoid shadowing',
                        'priority': 'MEDIUM'
                    })

        self.generic_visit(node)


def analyze_encoding_issues(file_path: Path, root_dir: Path) -> List[Dict]:
    """Check for encoding issues (Category 13)"""
    issues = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        lines = content.split('\n')
        has_encoding = any('coding' in line for line in lines[:2])
        has_non_ascii = any(ord(c) > 127 for c in content)

        if has_non_ascii and not has_encoding:
            rel_path = str(file_path.relative_to(root_dir))
            issues.append({
                'category': 'Encoding Issue',
                'severity': 'LOW',
                'line': 1,
                'file': rel_path,
                'description': 'Non-ASCII characters without encoding declaration',
                'recommendation': 'Add: # -*- coding: utf-8 -*-',
                'priority': 'LOW'
            })
    except Exception:  # TODO: Catch specific exception instead of Exception
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in {__name__}", exc_info=True)
        raise  # Re-raise the exception

    return issues


def analyze_for_dead_code(tree: ast.AST, file_path: str, root_dir: Path) -> List[Dict]:
    """Check for dead code (Category 18)"""
    issues = []
    rel_path = str(Path(file_path).relative_to(root_dir))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found_return = False
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    found_return = True
                elif found_return:
                    # Check if unreachable
                    if not isinstance(stmt, (ast.Str, ast.Expr, ast.Pass)):
                        if isinstance(stmt, ast.Expr):
                            if not isinstance(stmt.value, ast.Constant):
                                issues.append({
                                    'category': 'Dead Code',
                                    'severity': 'LOW',
                                    'line': stmt.lineno,
                                    'file': rel_path,
                                    'description': 'Unreachable code after return statement',
                                    'recommendation': 'Remove dead code',
                                    'priority': 'LOW'
                                })

    return issues


class ComprehensiveAnalyzer:
    """Main analyzer orchestrating all analysis"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.all_issues: List[Dict] = []
        self.file_issues: Dict[str, List[Dict]] = defaultdict(list)

    def analyze(self) -> Dict:
        """Run complete analysis"""
        print(f"Analyzing Python files in: {self.root_dir}")

        python_files = list(self.root_dir.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")

        # Phase 1: Build import graph
        print("\n[Phase 1] Building import dependency graph...")
        for py_file in python_files:
            self._build_import_info(py_file)

        # Phase 2: Detect circular dependencies
        print("[Phase 2] Detecting circular dependencies...")
        detector = CircularDependencyDetector(self.graph)
        cycles = detector.find_all_cycles()

        # Phase 3: Analyze files for edge cases
        print("[Phase 3] Analyzing files for edge cases...")
        analyzed = 0
        for py_file in python_files:
            try:
                issues = self._analyze_file(py_file)
                if issues:
                    rel_path = str(py_file.relative_to(self.root_dir))
                    self.file_issues[rel_path] = issues
                    self.all_issues.extend(issues)
                analyzed += 1

                if analyzed % 1000 == 0:
                    print(f"  Analyzed {analyzed}/{len(python_files)} files...")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in {__name__}", exc_info=True)
                raise  # Re-raise the exception

        # Generate report
        print("[Phase 4] Generating report...")
        return self._generate_report(cycles, len(python_files))

    def _build_import_info(self, file_path: Path):
        """Extract import information from file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))
                set_parent_references(tree)

            module_name = str(file_path.relative_to(self.root_dir))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.graph[module_name].add(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    self.graph[module_name].add(node.module)

        except Exception:  # TODO: Catch specific exception instead of Exception
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in {__name__}", exc_info=True)
            raise  # Re-raise the exception

    def _analyze_file(self, file_path: Path) -> List[Dict]:
        """Analyze single file for edge cases"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))

            detector = EdgeCaseDetector(str(file_path), self.root_dir)
            detector.detect_all(tree)

            # Additional checks
            encoding_issues = analyze_encoding_issues(file_path, self.root_dir)
            dead_code = analyze_for_dead_code(tree, str(file_path), self.root_dir)

            return detector.issues + encoding_issues + dead_code

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            return []

    def _generate_report(self, cycles: List[Dict], total_files: int) -> Dict:
        """Generate comprehensive report"""
        # Group by category and severity
        by_category = defaultdict(list)
        by_severity = defaultdict(list)

        for issue in self.all_issues:
            by_category[issue['category']].append(issue)
            by_severity[issue['severity']].append(issue)

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
                    'severity_breakdown': dict(Counter(i['severity'] for i in issues)),
                    'examples': issues[:5]
                }
                for cat, issues in sorted(by_category.items())
            },
            'issues_by_severity': {
                severity: {
                    'count': len(issues),
                    'categories': dict(Counter(i['category'] for i in issues))
                }
                for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
                if severity in by_severity
            },
            'top_files': {
                file: len(issues)
                for file, issues in sorted(
                    self.file_issues.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )[:30]
            }
        }


def main():
    """Main entry point"""
    root_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

    analyzer = ComprehensiveAnalyzer(root_dir)
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
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        if severity in report['issues_by_severity']:
            info = report['issues_by_severity'][severity]
            print(f"\n{severity}: {info['count']} issues")
            for category, count in sorted(info['categories'].items(), key=lambda x: -x[1])[:5]:
                print(f"  - {category}: {count}")

    print("\n" + "-"*80)
    print("TOP ISSUES BY CATEGORY")
    print("-"*80)
    for category, info in sorted(report['issues_by_category'].items(), key=lambda x: -x[1]['count'])[:10]:
        print(f"\n{category}: {info['count']} issues")
        severity_str = ", ".join(f"{s}:{c}" for s, c in info['severity_breakdown'].items())
        print(f"  Severity: {severity_str}")

    print("\n" + "-"*80)
    print("TOP FILES WITH MOST ISSUES")
    print("-"*80)
    for file, count in list(report['top_files'].items())[:15]:
        print(f"  {count:3d} - {file}")

    # Save detailed report
    output_file = Path(root_dir) / "EDGE_CASE_ANALYSIS_REPORT.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDetailed report saved to: {output_file}")


if __name__ == '__main__':
    main()
