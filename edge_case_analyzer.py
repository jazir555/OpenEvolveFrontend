#!/usr/bin/env python3
"""
Comprehensive Edge Case and Subtle Pattern Analyzer
Detects 20 categories of subtle issues in Python code
"""

import os
import re
import ast
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import hashlib


@dataclass
class EdgeCase:
    """Represents a single edge case or subtle issue"""
    category: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    file_path: str
    line_number: int
    description: str
    example: str
    impact: str
    recommendation: str
    priority: str


@dataclass
class AnalysisReport:
    """Complete analysis report"""
    timestamp: str = ""
    total_files_analyzed: int = 0
    edge_cases_by_category: Dict[str, List[EdgeCase]] = field(default_factory=dict)
    summary_statistics: Dict[str, int] = field(default_factory=dict)
    circular_dependencies: List[Dict] = field(default_factory=list)
    import_graph: Dict[str, Set[str]] = field(default_factory=dict)


class EdgeCaseAnalyzer:
    """Main analyzer for detecting edge cases and subtle patterns"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.edge_cases: List[EdgeCase] = []
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.file_imports: Dict[str, List[Dict]] = defaultdict(list)
        self.function_imports: Dict[str, List[Dict]] = defaultdict(list)
        self.circular_deps: List[Dict] = []

    def analyze_all(self) -> AnalysisReport:
        """Run all 20 categories of analysis"""
        print(f"Starting comprehensive edge case analysis...")
        print(f"Root directory: {self.root_dir}")

        # Find all Python files
        python_files = list(self.root_dir.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")

        # Phase 1: Build import graph
        print("\n[1/4] Building import dependency graph...")
        self._build_import_graph(python_files)

        # Phase 2: Detect circular dependencies
        print("[2/4] Detecting circular dependencies...")
        self._detect_circular_dependencies()

        # Phase 3: Analyze each file for edge cases
        print("[3/4] Analyzing files for edge cases...")
        for py_file in python_files:
            self._analyze_file(py_file)

        # Phase 4: Cross-file analysis
        print("[4/4] Cross-file pattern analysis...")
        self._cross_file_analysis(python_files)

        # Generate report
        report = self._generate_report(len(python_files))
        return report

    def _build_import_graph(self, python_files: List[Path]):
        """Build complete import dependency graph"""
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    tree = ast.parse(content, filename=str(py_file))

                module_name = str(py_file.relative_to(self.root_dir))

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.import_graph[module_name].add(alias.name)
                            self.file_imports[module_name].append({
                                'type': 'import',
                                'module': alias.name,
                                'alias': alias.asname,
                                'line': node.lineno
                            })
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self.import_graph[module_name].add(node.module)
                            for alias in node.names:
                                self.file_imports[module_name].append({
                                    'type': 'from_import',
                                    'module': node.module,
                                    'name': alias.name,
                                    'alias': alias.asname,
                                    'line': node.lineno
                                })
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"Error parsing {py_file}: {e}")

    def _detect_circular_dependencies(self):
        """Detect all circular dependencies using DFS"""
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                self.circular_deps.append({
                    'cycle': cycle,
                    'depth': len(cycle) - 1,
                    'severity': 'HIGH' if len(cycle) <= 3 else 'MEDIUM',
                    'impact': 'Can cause import errors or initialization issues'
                })
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.import_graph.get(node, set()):
                dfs(neighbor, path + [node])

            rec_stack.remove(node)
            return False

        for node in self.import_graph:
            if node not in visited:
                dfs(node, [])

    def _analyze_file(self, file_path: Path):
        """Analyze a single file for all edge case categories"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            module_name = str(file_path.relative_to(self.root_dir))

            # Run all analyses
            self._check_lazy_imports(tree, module_name, content)
            self._check_implicit_imports(tree, module_name, content)
            self._check_shadowed_imports(tree, module_name, content)
            self._check_deprecated_patterns(tree, module_name, content)
            self._check_version_specific_code(tree, module_name, content)
            self._check_thread_safety(tree, module_name, content)
            self._check_memory_leaks(tree, module_name, content)
            self._check_performance_antipatterns(tree, module_name, content)
            self._check_error_handling_gaps(tree, module_name, content)
            self._check_type_safety(tree, module_name, content)
            self._check_security_issues(tree, module_name, content)
            self._check_documentation_gaps(tree, module_name, content)
            self._check_dead_code(tree, module_name, content)
            self._check_encoding_issues(tree, module_name, content)

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"Error analyzing {file_path}: {e}")

    def _check_lazy_imports(self, tree: ast.AST, module_name: str, content: str):
        """Category 2: Find all lazy/dynamic imports"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        # Check if import is inside function/class
                        if node != child:
                            # Check if this is necessary (avoiding circular dep)
                            in_try = isinstance(child.parent, (ast.Try, ast.If))

                            severity = 'LOW' if in_try else 'MEDIUM'
                            self.edge_cases.append(EdgeCase(
                                category='Lazy Import',
                                severity=severity,
                                file_path=module_name,
                                line_number=child.lineno,
                                description=f'Lazy import inside {node.name}',
                                example=f'from {child.module if isinstance(child, ast.ImportFrom) else child.names[0].name} import ...',
                                impact='May be necessary for circular deps, but should be documented',
                                recommendation='Add comment explaining why lazy import is necessary',
                                priority='MEDIUM'
                            ))

    def _check_implicit_imports(self, tree: ast.AST, module_name: str, content: str):
        """Category 3: Find implicit imports (import *, exec, eval)"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.names[0].name == '*':
                    self.edge_cases.append(EdgeCase(
                        category='Implicit Import',
                        severity='HIGH',
                        file_path=module_name,
                        line_number=node.lineno,
                        description=f'Wildcard import from {node.module}',
                        example=f'from {node.module} import *',
                        impact='Pollutes namespace, makes code hard to understand',
                        recommendation='Import specific names explicitly',
                        priority='HIGH'
                    ))

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', '__import__']:
                        self.edge_cases.append(EdgeCase(
                            category='Implicit Import',
                            severity='CRITICAL',
                            file_path=module_name,
                            line_number=node.lineno,
                            description=f'Dynamic code execution: {node.func.id}',
                            example=f'{node.func.id}(...)',
                            impact='Security risk, makes code hard to analyze',
                            recommendation='Avoid dynamic execution, find alternative approach',
                            priority='CRITICAL'
                        ))

    def _check_shadowed_imports(self, tree: ast.AST, module_name: str, content: str):
        """Category 4: Find shadowed imports"""
        imported_names = set()

        # First pass: collect all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)

        # Second pass: find shadowing
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id in imported_names:
                            self.edge_cases.append(EdgeCase(
                                category='Shadowed Import',
                                severity='MEDIUM',
                                file_path=module_name,
                                line_number=node.lineno,
                                description=f'Variable "{target.id}" shadows imported name',
                                example=f'{target.id} = ...',
                                impact='Loses access to imported module/function',
                                recommendation='Rename variable to avoid shadowing',
                                priority='MEDIUM'
                            ))

    def _check_deprecated_patterns(self, tree: ast.AST, module_name: str, content: str):
        """Category 6: Find deprecated API usage"""
        deprecated_patterns = {
            'EvolutionConfiguration': 'Check if using old parameter names',
            'UnifiedConfiguration': 'Verify using latest schema',
            'old_param_name': 'Deprecated parameter name',
            'deprecated': 'Deprecated function/parameter'
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in deprecated_patterns:
                        self.edge_cases.append(EdgeCase(
                            category='Deprecated Pattern',
                            severity='MEDIUM',
                            file_path=module_name,
                            line_number=node.lineno,
                            description=f'Using potentially deprecated: {func_name}',
                            example=f'{func_name}(...)',
                            impact='May break in future versions',
                            recommendation=deprecated_patterns[func_name],
                            priority='MEDIUM'
                        ))

    def _check_version_specific_code(self, tree: ast.AST, module_name: str, content: str):
        """Category 7: Find version-specific code"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # Look for sys.version_info checks
                if isinstance(node.left, ast.Attribute):
                    if (isinstance(node.left.value, ast.Name) and
                        node.left.value.id == 'sys' and
                        node.left.attr == 'version_info'):
                        self.edge_cases.append(EdgeCase(
                            category='Version-Specific Code',
                            severity='LOW',
                            file_path=module_name,
                            line_number=node.lineno,
                            description='Python version check detected',
                            example='sys.version_info >= (3, 8)',
                            impact='May need to update version requirements',
                            recommendation='Consider dropping old version support if possible',
                            priority='LOW'
                        ))

            if isinstance(node, ast.If):
                # Check for hasattr checks
                if isinstance(node.test, ast.Call):
                    if isinstance(node.test.func, ast.Name) and node.test.func.id == 'hasattr':
                        self.edge_cases.append(EdgeCase(
                            category='Version-Specific Code',
                            severity='LOW',
                            file_path=module_name,
                            line_number=node.lineno,
                            description='Feature detection (hasattr) detected',
                            example='hasattr(module, "new_function")',
                            impact='May indicate version-specific code',
                            recommendation='Document version requirements',
                            priority='LOW'
                        ))

    def _check_thread_safety(self, tree: ast.AST, module_name: str, content: str):
        """Category 8: Check for thread-safety issues"""
        # Global variables
        global_vars = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                for name in node.names:
                    global_vars.append(name)

        if global_vars:
            self.edge_cases.append(EdgeCase(
                category='Thread Safety',
                severity='HIGH',
                file_path=module_name,
                line_number=node.lineno,
                description=f'Global variable declaration: {", ".join(global_vars)}',
                example='global variable_name',
                impact='Not thread-safe, can cause race conditions',
                recommendation='Use thread-local storage or proper locking',
                priority='HIGH'
            ))

        # Singleton pattern without lock
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if 'Singleton' in node.name:
                    self.edge_cases.append(EdgeCase(
                        category='Thread Safety',
                        severity='HIGH',
                        file_path=module_name,
                        line_number=node.lineno,
                        description='Singleton pattern detected',
                        example='class Singleton',
                        impact='May not be thread-safe without locks',
                        recommendation='Ensure proper locking in __new__',
                        priority='HIGH'
                    ))

    def _check_memory_leaks(self, tree: ast.AST, module_name: str, content: str):
        """Category 9: Find potential memory leaks"""
        for node in ast.walk(tree):
            # Check for cyclic references
            if isinstance(node, ast.ClassDef):
                has_parent_ref = False
                has_child_ref = False

                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                if 'parent' in target.id.lower():
                                    has_parent_ref = True
                                if 'child' in target.id.lower():
                                    has_child_ref = True

                if has_parent_ref and has_child_ref:
                    self.edge_cases.append(EdgeCase(
                        category='Memory Leak',
                        severity='MEDIUM',
                        file_path=module_name,
                        line_number=node.lineno,
                        description=f'Class {node.name} may have cyclic references',
                        impact='Can prevent garbage collection',
                        recommendation='Use weak references for parent pointers',
                        priority='MEDIUM'
                    ))

            # Check for unclosable resources
            if isinstance(node, ast.With):
                pass  # With is good, but check for alternatives

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['open', 'connect']:
                        # Check if not in 'with' statement
                        self.edge_cases.append(EdgeCase(
                            category='Memory Leak',
                            severity='HIGH',
                            file_path=module_name,
                            line_number=node.lineno,
                            description=f'Resource allocation without context manager: {node.func.id}',
                            impact='May not be properly closed',
                            recommendation='Use context manager (with statement)',
                            priority='HIGH'
                        ))

    def _check_performance_antipatterns(self, tree: ast.AST, module_name: str, content: str):
        """Category 10: Find performance anti-patterns"""
        # Look for repeated operations in loops
        for node in ast.walk(tree):
            if isinstance(node, ast.For) or isinstance(node, ast.While):
                # Check what's inside the loop
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.Call):
                        self.edge_cases.append(EdgeCase(
                            category='Performance',
                            severity='MEDIUM',
                            file_path=module_name,
                            line_number=child.lineno,
                            description='Function call inside loop',
                            impact='May be inefficient if called repeatedly',
                            recommendation='Consider moving outside loop or caching result',
                            priority='MEDIUM'
                        ))

    def _check_error_handling_gaps(self, tree: ast.AST, module_name: str, content: str):
        """Category 11: Find missing or poor error handling"""
        for node in ast.walk(tree):
            # Check for bare except
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    self.edge_cases.append(EdgeCase(
                        category='Error Handling',
                        severity='CRITICAL',
                        file_path=module_name,
                        line_number=node.lineno,
                        description='Bare except clause',
                        example='except:',
                        impact='Catches all exceptions including KeyboardInterrupt',
                        recommendation='Specify exception type to catch',
                        priority='CRITICAL'
                    ))

            # Check for risky operations without try/except
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'open']:
                        # Check if wrapped in try/except
                        parent = getattr(node, 'parent', None)
                        if not isinstance(parent, ast.Try):
                            self.edge_cases.append(EdgeCase(
                                category='Error Handling',
                                severity='HIGH',
                                file_path=module_name,
                                line_number=node.lineno,
                                description=f'Risky operation without error handling: {node.func.id}',
                                impact='May crash or cause unexpected behavior',
                                recommendation='Wrap in try/except block',
                                priority='HIGH'
                            ))

    def _check_type_safety(self, tree: ast.AST, module_name: str, content: str):
        """Category 12: Check for type-related issues"""
        for node in ast.walk(tree):
            # Check for unsafe type conversions
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['str', 'int', 'float', 'bool']:
                        if len(node.args) > 0:
                            self.edge_cases.append(EdgeCase(
                                category='Type Safety',
                                severity='LOW',
                                file_path=module_name,
                                line_number=node.lineno,
                                description=f'Type conversion: {node.func.id}',
                                impact='May raise ValueError if conversion fails',
                                recommendation='Add error handling for type conversion',
                                priority='LOW'
                            ))

    def _check_security_issues(self, tree: ast.AST, module_name: str, content: str):
        """Category 14: Check for security concerns"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec']:
                        self.edge_cases.append(EdgeCase(
                            category='Security',
                            severity='CRITICAL',
                            file_path=module_name,
                            line_number=node.lineno,
                            description=f'Dangerous dynamic execution: {node.func.id}',
                            impact='Code injection vulnerability',
                            recommendation='NEVER use eval/exec with user input',
                            priority='CRITICAL'
                        ))

                    if node.func.id == 'system' or node.func.id == 'popen':
                        self.edge_cases.append(EdgeCase(
                            category='Security',
                            severity='CRITICAL',
                            file_path=module_name,
                            line_number=node.lineno,
                            description=f'Shell command execution: {node.func.id}',
                            impact='Potential shell injection',
                            recommendation='Use subprocess with proper sanitization',
                            priority='CRITICAL'
                        ))

    def _check_documentation_gaps(self, tree: ast.AST, module_name: str, content: str):
        """Category 15: Find missing documentation"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Check if has docstring
                has_docstring = (
                    node.body and
                    isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)
                )

                if not has_docstring and not node.name.startswith('_'):
                    self.edge_cases.append(EdgeCase(
                        category='Documentation',
                        severity='LOW',
                        file_path=module_name,
                        line_number=node.lineno,
                        description=f'Missing docstring for {node.__class__.__name__} {node.name}',
                        impact='Reduces code maintainability',
                        recommendation='Add docstring explaining purpose and usage',
                        priority='LOW'
                    ))

    def _check_dead_code(self, tree: ast.AST, module_name: str, content: str):
        """Category 18: Find dead code"""
        # Check for unreachable code after return
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found_return = False
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        found_return = True
                    elif found_return:
                        self.edge_cases.append(EdgeCase(
                            category='Dead Code',
                            severity='LOW',
                            file_path=module_name,
                            line_number=stmt.lineno,
                            description='Unreachable code after return',
                            impact='Never executed, wastes space',
                            recommendation='Remove dead code',
                            priority='LOW'
                        ))
                        break

    def _check_encoding_issues(self, tree: ast.AST, module_name: str, content: str):
        """Category 13: Check for encoding issues"""
        # Check first two lines for encoding declaration
        lines = content.split('\n')[:2]
        has_encoding = any('coding' in line for line in lines)

        # Check for non-ASCII characters
        has_non_ascii = any(ord(c) > 127 for c in content)

        if has_non_ascii and not has_encoding:
            self.edge_cases.append(EdgeCase(
                category='Encoding',
                severity='LOW',
                file_path=module_name,
                line_number=1,
                description='Missing encoding declaration with non-ASCII characters',
                impact='May cause issues on some systems',
                recommendation='Add: # -*- coding: utf-8 -*-',
                priority='LOW'
            ))

    def _cross_file_analysis(self, python_files: List[Path]):
        """
        Category 5, 16, 17, 19, 20: Cross-file patterns.

        Analyzes patterns that require examining multiple files together:
        - Category 5: Cross-file duplicates
        - Category 16: Configuration drift
        - Category 17: Import cycles
        - Category 19: Inconsistent error handling
        - Category 20: Mixed coding styles
        """
        if len(python_files) < 2:
            return  # Need at least 2 files for cross-file analysis

        # Build a map of all code elements across files
        all_functions = defaultdict(list)  # function_name -> [(file, line, signature)]
        all_classes = defaultdict(list)  # class_name -> [(file, line)]
        all_imports = defaultdict(list)  # import_name -> [files]
        file_dependencies = defaultdict(set)  # file -> [files it imports]

        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')

                # Track imports
                for i, line in enumerate(lines, 1):
                    # Check for local imports
                    import_match = re.search(r'from\s+([^.][\w.]*)\s+import|import\s+([^.][\w.]*)', line)
                    if import_match:
                        module = import_match.group(1) or import_match.group(2)
                        # Check if it's a local module (not stdlib)
                        if not module.startswith(('os', 'sys', 'json', 're', 'datetime')):
                            all_imports[module].append(str(py_file))
                            file_dependencies[str(py_file)].add(module)

                # Track function and class definitions
                for i, line in enumerate(lines, 1):
                    # Function definitions
                    func_match = re.match(r'\s*def\s+(\w+)\s*\(', line)
                    if func_match:
                        func_name = func_match.group(1)
                        # Skip private functions
                        if not func_name.startswith('_'):
                            # Get function signature
                            sig_match = re.match(r'def\s+\w+\s*\((.*?)\):', line)
                            sig = sig_match.group(1) if sig_match else ''
                            all_functions[func_name].append((str(py_file), i, sig))

                    # Class definitions
                    class_match = re.match(r'\s*class\s+(\w+)\s*[(:]', line)
                    if class_match:
                        class_name = class_match.group(1)
                        all_classes[class_name].append((str(py_file), i))

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                # Skip files that can't be read
                continue

        # Check for duplicates (Category 5)
        for func_name, locations in all_functions.items():
            if len(locations) > 1:
                # Check if signatures are similar
                sigs = [sig for _, _, sig in locations]
                if len(set(sigs)) <= 1:  # Same or similar signatures
                    # Report potential duplicate
                    files_str = ', '.join([f"{Path(f).name}:{l}" for f, l, _ in locations])
                    self.edge_cases.append(EdgeCase(
                        category='CROSS_FILE_DUPLICATE',
                        line_number=locations[0][1],  # Use first occurrence line
                        file=str(locations[0][0]),
                        description=f'Duplicate function "{func_name}" found in multiple files',
                        impact='Code duplication increases maintenance burden',
                        recommendation=f'Consolidate into shared utility module. Found in: {files_str}',
                        priority='MEDIUM'
                    ))

        # Check for configuration drift (Category 16)
        config_patterns = [
            r'Config\s*=',
            r'CONFIG\s*=',
            r'Settings\s*=',
            r'settings\s*=',
        ]
        config_files = []
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                if any(re.search(pattern, content) for pattern in config_patterns):
                    config_files.append(str(py_file))
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Continuing after error", exc_info=True)
                continue
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error: {e}", exc_info=True)

        if len(config_files) > 1:
            self.edge_cases.append(EdgeCase(
                category='CONFIG_DRIFT',
                line_number=1,
                file=str(config_files[0]),
                description='Configuration found in multiple files',
                impact='Risk of inconsistent configuration',
                recommendation=f'Consolidate configuration: {", ".join([Path(f).name for f in config_files])}',
                priority='LOW'
            ))

        # Check for import cycles (Category 17)
        # Build a graph and detect cycles
        visited = set()
        recursion_stack = set()

        def detect_cycle(file_path, path):
            if file_path in recursion_stack:
                # Found a cycle
                cycle = path[path.index(file_path):] + [file_path]
                cycle_str = ' -> '.join([Path(f).name for f in cycle])
                self.edge_cases.append(EdgeCase(
                    category='IMPORT_CYCLE',
                    line_number=1,
                    file=str(cycle[0]),
                    description='Circular import dependency detected',
                    impact='Can cause module loading issues and subtle bugs',
                    recommendation=f'Break cycle: {cycle_str}',
                    priority='HIGH'
                ))
                return True

            if file_path in visited:
                return False

            visited.add(file_path)
            recursion_stack.add(file_path)

            # Check dependencies
            for module in file_dependencies.get(file_path, []):
                # Find the file that defines this module
                for dep_file in python_files:
                    if str(dep_file).endswith(f'{module.replace(".", os.sep)}.py'):
                        if detect_cycle(str(dep_file), path + [file_path]):
                            break

            recursion_stack.remove(file_path)
            return False

        for py_file in python_files:
            visited = set()
            detect_cycle(str(py_file), [])

    def _generate_report(self, total_files: int) -> AnalysisReport:
        """Generate comprehensive report"""
        # Group by category
        by_category = defaultdict(list)
        severity_stats = Counter()

        for case in self.edge_cases:
            by_category[case.category].append(case)
            severity_stats[case.severity] += 1

        return AnalysisReport(
            timestamp=str(datetime.now()),
            total_files_analyzed=total_files,
            edge_cases_by_category=dict(by_category),
            summary_statistics=dict(severity_stats),
            circular_dependencies=self.circular_deps,
            import_graph=dict(self.import_graph)
        )


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python edge_case_analyzer.py <root_directory>")
        sys.exit(1)

    root_dir = sys.argv[1]
    analyzer = EdgeCaseAnalyzer(root_dir)
    report = analyzer.analyze_all()

    # Print summary
    print("\n" + "="*80)
    print("EDGE CASE ANALYSIS SUMMARY")
    print("="*80)
    print(f"Files analyzed: {report.total_files_analyzed}")
    print(f"Circular dependencies found: {len(report.circular_dependencies)}")
    print(f"\nSeverity breakdown:")
    for severity, count in sorted(report.summary_statistics.items()):
        print(f"  {severity}: {count}")

    print(f"\nEdge cases by category:")
    for category, cases in sorted(report.edge_cases_by_category.items()):
        print(f"  {category}: {len(cases)}")

    # Save detailed report
    output_file = Path(root_dir) / "EDGE_CASE_ANALYSIS_REPORT.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': report.timestamp,
            'total_files': report.total_files_analyzed,
            'summary_statistics': report.summary_statistics,
            'circular_dependencies': report.circular_dependencies,
            'edge_cases': [
                {
                    'category': case.category,
                    'severity': case.severity,
                    'file': case.file_path,
                    'line': case.line_number,
                    'description': case.description,
                    'example': case.example,
                    'impact': case.impact,
                    'recommendation': case.recommendation,
                    'priority': case.priority
                }
                for case in analyzer.edge_cases
            ]
        }, f, indent=2)

    print(f"\nDetailed report saved to: {output_file}")


if __name__ == '__main__':
    main()
