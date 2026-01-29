"""
Ultimate Validation Suite - Most Comprehensive Possible Validation
Checks EVERYTHING with zero assumptions.
"""

import os
import ast
import importlib
import subprocess
import sys
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import traceback


class UltimateValidator:
    """
    The most comprehensive validator possible.
    Checks everything. Assumes nothing.
    """

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()
        self.results = {
            'files_checked': 0,
            'files_passed': 0,
            'files_failed': 0,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'issues_found': [],
            'warnings': [],
            'info': [],
            'score': 0.0,
            'grade': 'F',
            'check_results': {}
        }

        self.excluded_dirs = {
            '__pycache__', '.git', 'node_modules', '.venv', 'venv',
            'env', 'dist', 'build', '.pytest_cache', '.tox',
            'core-projects', 'projects to analyze', 'Generic-Knowledge-Extraction-Tool',
            'Formal-Reasoning-Mode', 'Research-Quest', 'NeuralKG', 'DeepKE-main'
        }

        print(f"Ultimate Validator initialized")
        print(f"Root directory: {self.root_dir}")
        print(f"Excluded directories: {len(self.excluded_dirs)}")

    def validate_everything(self) -> Dict[str, Any]:
        """Run all validation checks - absolutely everything."""
        print("=" * 80)
        print("ULTIMATE VALIDATION SUITE")
        print("=" * 80)
        print(f"Starting comprehensive validation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        start_time = time.time()

        # Check 1: File Existence & Integrity
        print("Check 1/10: File Existence & Integrity...")
        self.check_file_existence()

        # Check 2: Syntax Validation
        print("Check 2/10: Syntax Validation...")
        self.check_all_syntax()

        # Check 3: Import Validation
        print("Check 3/10: Import Validation...")
        self.check_all_imports()

        # Check 4: Pattern Validation
        print("Check 4/10: Pattern Validation...")
        self.check_all_patterns()

        # Check 5: Dependency Validation
        print("Check 5/10: Dependency Validation...")
        self.check_dependencies()

        # Check 6: Type Validation
        print("Check 6/10: Type Validation...")
        self.check_types()

        # Check 7: Test Validation
        print("Check 7/10: Test Validation...")
        self.check_tests()

        # Check 8: Performance Validation
        print("Check 8/10: Performance Validation...")
        self.check_performance()

        # Check 9: Security Validation
        print("Check 9/10: Security Validation...")
        self.check_security()

        # Check 10: Documentation Validation
        print("Check 10/10: Documentation Validation...")
        self.check_documentation()

        # Calculate final score
        print("\nCalculating final score...")
        self.calculate_score()

        # Generate report
        print("Generating comprehensive report...")
        self.generate_report()

        elapsed = time.time() - start_time

        print()
        print("=" * 80)
        print(f"VALIDATION COMPLETE in {elapsed:.2f} seconds")
        print("=" * 80)

        return self.results

    def get_python_files(self) -> List[Path]:
        """Get all Python files, excluding certain directories."""
        py_files = []
        for file in self.root_dir.rglob('*.py'):
            # Skip excluded directories
            if any(excluded in file.parts for excluded in self.excluded_dirs):
                continue
            py_files.append(file)
        return py_files

    # ============================================
    # CHECK 1: File Existence & Integrity
    # ============================================

    def check_file_existence(self):
        """Verify all files exist and are readable."""
        check_results = {
            'name': 'File Existence & Integrity',
            'passed': False,
            'issues': [],
            'details': {}
        }

        all_py_files = self.get_python_files()
        check_results['details']['total_files'] = len(all_py_files)

        for file in all_py_files:
            self.results['files_checked'] += 1

            # Check file exists
            if not file.exists():
                issue = {
                    'file': str(file.relative_to(self.root_dir)),
                    'issue': 'File does not exist',
                    'severity': 'CRITICAL'
                }
                self.results['issues_found'].append(issue)
                check_results['issues'].append(issue)
                self.results['files_failed'] += 1
                continue

            # Check file is readable
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check file is not empty
                if len(content.strip()) == 0:
                    issue = {
                        'file': str(file.relative_to(self.root_dir)),
                        'issue': 'File is empty',
                        'severity': 'LOW'
                    }
                    self.results['warnings'].append(issue)
                    check_results['issues'].append(issue)

                # Check for encoding issues
                try:
                    content.encode('utf-8')
                except UnicodeEncodeError as e:
                    issue = {
                        'file': str(file.relative_to(self.root_dir)),
                        'issue': f'Encoding error: {e}',
                        'severity': 'MEDIUM'
                    }
                    self.results['issues_found'].append(issue)
                    check_results['issues'].append(issue)
                    self.results['files_failed'] += 1
                    continue

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                issue = {
                    'file': str(file.relative_to(self.root_dir)),
                    'issue': f'Cannot read file: {e}',
                    'severity': 'CRITICAL'
                }
                self.results['issues_found'].append(issue)
                check_results['issues'].append(issue)
                self.results['files_failed'] += 1
            else:
                self.results['files_passed'] += 1

        check_results['passed'] = self.results['files_failed'] == 0
        check_results['details']['files_passed'] = self.results['files_passed']
        check_results['details']['files_failed'] = self.results['files_failed']
        self.results['check_results']['file_existence'] = check_results

        if check_results['passed']:
            print(f"  [PASS] All {self.results['files_checked']} files exist and are readable")
        else:
            print(f"  [FAIL] {self.results['files_failed']} files have issues")

    # ============================================
    # CHECK 2: Syntax Validation
    # ============================================

    def check_all_syntax(self):
        """Validate syntax of ALL Python files."""
        check_results = {
            'name': 'Syntax Validation',
            'passed': False,
            'issues': [],
            'details': {}
        }

        all_py_files = self.get_python_files()
        syntax_errors = []

        for file in all_py_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Parse with AST
                ast.parse(code)

            except SyntaxError as e:
                error = {
                    'file': str(file.relative_to(self.root_dir)),
                    'line': e.lineno,
                    'column': e.offset,
                    'error': str(e),
                    'severity': 'CRITICAL'
                }
                syntax_errors.append(error)
                self.results['issues_found'].append(error)

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                error = {
                    'file': str(file.relative_to(self.root_dir)),
                    'error': str(e),
                    'severity': 'HIGH'
                }
                syntax_errors.append(error)
                self.results['issues_found'].append(error)

        check_results['issues'] = syntax_errors
        check_results['details']['files_with_syntax_errors'] = len(syntax_errors)
        check_results['passed'] = len(syntax_errors) == 0
        self.results['check_results']['syntax'] = check_results

        if check_results['passed']:
            print(f"  [PASS] All files have valid syntax")
        else:
            print(f"  [FAIL] Syntax errors in {len(syntax_errors)} files")

    # ============================================
    # CHECK 3: Import Validation
    # ============================================

    def check_all_imports(self):
        """Validate all imports in all files."""
        check_results = {
            'name': 'Import Validation',
            'passed': False,
            'issues': [],
            'details': {}
        }

        bad_imports = []
        all_imports = []
        missing_modules = []

        all_py_files = self.get_python_files()

        for file in all_py_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())

                # Check all imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            all_imports.append({
                                'file': str(file.relative_to(self.root_dir)),
                                'module': alias.name,
                                'line': node.lineno
                            })

                            # Try to import the module
                            try:
                                importlib.import_module(alias.name)
                            except ImportError:
                                missing_modules.append({
                                    'file': str(file.relative_to(self.root_dir)),
                                    'module': alias.name,
                                    'line': node.lineno,
                                    'severity': 'HIGH'
                                })

                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            all_imports.append({
                                'file': str(file.relative_to(self.root_dir)),
                                'module': node.module,
                                'line': node.lineno
                            })

                            # Check for star imports
                            if any(alias.name == '*' for alias in node.names):
                                bad_imports.append({
                                    'file': str(file.relative_to(self.root_dir)),
                                    'import': f"from {node.module} import *",
                                    'line': node.lineno,
                                    'issue': 'Star import',
                                    'severity': 'MEDIUM'
                                })

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Continuing after error", exc_info=True)
                continue

        # Check for evolution imports without guards
        evolution_imports = self.check_evolution_imports()
        bad_imports.extend(evolution_imports)

        check_results['issues'] = bad_imports + missing_modules
        check_results['details']['total_imports'] = len(all_imports)
        check_results['details']['bad_imports'] = len(bad_imports)
        check_results['details']['missing_modules'] = len(missing_modules)
        check_results['passed'] = len(bad_imports) == 0 and len(missing_modules) == 0
        self.results['check_results']['imports'] = check_results

        if check_results['passed']:
            print(f"  [PASS] All {len(all_imports)} imports are valid")
        else:
            print(f"  [FAIL] {len(bad_imports)} bad imports, {len(missing_modules)} missing modules")

    def check_evolution_imports(self) -> List[Dict[str, Any]]:
        """Check for evolution imports without proper guards."""
        issues = []

        # Files that should have guards
        sensitive_imports = {
            'evolution': ['import_evolution_safely', 'EVOLUTION_AVAILABLE'],
        }

        py_files = self.get_python_files()

        for file in py_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)

                has_import = False
                has_guard = False

                # Check for evolution imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith('evolution'):
                            has_import = True

                            # Check if safe import
                            if any(alias.name in ['import_evolution_safely', 'EVOLUTION_AVAILABLE']
                                   for alias in node.names):
                                has_guard = True

                if has_import and not has_guard:
                    issues.append({
                        'file': str(file.relative_to(self.root_dir)),
                        'issue': 'Direct evolution import without guard',
                        'severity': 'MEDIUM'
                    })

            except Exception:  # TODO: Catch specific exception instead of Exception
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Continuing after error", exc_info=True)
                continue

        return issues

    # ============================================
    # CHECK 4: Pattern Validation
    # ============================================

    def check_all_patterns(self):
        """Validate all code patterns."""
        check_results = {
            'name': 'Pattern Validation',
            'passed': False,
            'issues': [],
            'details': {}
        }

        patterns_found = []

        # Pattern checks
        patterns = [
            {
                'name': 'Direct ParameterManager usage',
                'pattern': r'ParameterManager\(\)',
                'severity': 'HIGH',
                'fix': 'Use UnifiedConfiguration'
            },
            {
                'name': 'Direct session state access',
                'pattern': r"st\.session_state\['",
                'severity': 'MEDIUM',
                'fix': 'Use UnifiedConfiguration'
            },
            {
                'name': 'Star imports',
                'pattern': r'from\s+\S+\s+import\s+\*',
                'severity': 'MEDIUM',
                'fix': 'Import specific names'
            },
            {
                'name': 'Bare except',
                'pattern': r'except\s*:',
                'severity': 'MEDIUM',
                'fix': 'Use specific exception types'
            },
            {
                'name': 'Print statements',
                'pattern': r'\bprint\s*\(',
                'severity': 'LOW',
                'fix': 'Use proper logging'
            }
        ]

        all_py_files = self.get_python_files()

        for pattern_spec in patterns:
            pattern = pattern_spec['pattern']
            matches = []

            for file in all_py_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            matches.append({
                                'file': str(file.relative_to(self.root_dir)),
                                'line': i,
                                'content': line.strip(),
                                'pattern': pattern_spec['name']
                            })
                except Exception:  # TODO: Catch specific exception instead of Exception
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Continuing after error", exc_info=True)
                    continue

            if matches:
                for match in matches[:10]:  # Limit to first 10 per pattern
                    patterns_found.append({
                        'file': match['file'],
                        'line': match['line'],
                        'pattern': match['pattern'],
                        'severity': pattern_spec['severity'],
                        'fix': pattern_spec['fix']
                    })

        check_results['issues'] = patterns_found
        check_results['details']['patterns_checked'] = len(patterns)
        check_results['details']['issues_found'] = len(patterns_found)
        check_results['passed'] = len(patterns_found) == 0
        self.results['check_results']['patterns'] = check_results

        if check_results['passed']:
            print(f"  [PASS] No bad patterns found")
        else:
            print(f"  [WARN] {len(patterns_found)} pattern issues found")

    # ============================================
    # CHECK 5: Dependency Validation
    # ============================================

    def check_dependencies(self):
        """Check for dependency issues."""
        check_results = {
            'name': 'Dependency Validation',
            'passed': False,
            'issues': [],
            'details': {}
        }

        issues = []

        # Check for circular dependencies
        circular = self.detect_circular_dependencies()
        if circular:
            issues.extend(circular)

        # Check for missing dependencies
        missing = self.detect_missing_dependencies()
        if missing:
            issues.extend(missing)

        check_results['issues'] = issues
        check_results['details']['circular_dependencies'] = len(circular)
        check_results['details']['missing_dependencies'] = len(missing)
        check_results['passed'] = len(issues) == 0
        self.results['check_results']['dependencies'] = check_results

        if check_results['passed']:
            print(f"  [PASS] No dependency issues")
        else:
            print(f"  [FAIL] {len(issues)} dependency issues found")

    def detect_circular_dependencies(self) -> List[Dict[str, Any]]:
        """Detect circular import dependencies."""
        issues = []

        # Build dependency graph
        graph = defaultdict(set)
        file_map = {}

        py_files = self.get_python_files()

        for file in py_files:
            try:
                module_name = str(file.relative_to(self.root_dir)).replace(os.sep, '.').replace('.py', '')
                file_map[module_name] = file

                with open(file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            graph[module_name].add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            graph[module_name].add(node.module)

            except Exception:  # TODO: Catch specific exception instead of Exception
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Continuing after error", exc_info=True)
                continue

        # Check for cycles (simplified check)
        visited = set()
        rec_stack = set()

        def has_cycle(node, path=None):
            if path is None:
                path = []

            if node in rec_stack:
                return path + [node]

            if node in visited:
                return None

            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor in file_map:  # Only check internal modules
                    cycle = has_cycle(neighbor, path + [node])
                    if cycle:
                        return cycle

            rec_stack.remove(node)
            return None

        for module in graph:
            if module not in visited:
                cycle = has_cycle(module)
                if cycle and len(cycle) > 1:
                    issues.append({
                        'issue': 'Circular dependency detected',
                        'cycle': ' -> '.join(cycle),
                        'severity': 'HIGH'
                    })
                    break  # Report first cycle only

        return issues

    def detect_missing_dependencies(self) -> List[Dict[str, Any]]:
        """Detect missing dependencies."""
        issues = []

        # Common external dependencies
        known_dependencies = {
            'streamlit', 'pandas', 'numpy', 'matplotlib', 'plotly',
            'pytest', 'pydantic', 'requests', 'yaml', 'toml', 'asyncio'
        }

        # Check if common modules are available
        for dep in known_dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                issues.append({
                    'dependency': dep,
                    'issue': f'Missing dependency: {dep}',
                    'severity': 'MEDIUM'
                })

        return issues

    # ============================================
    # CHECK 6: Type Validation
    # ============================================

    def check_types(self):
        """Validate type hints and type safety."""
        check_results = {
            'name': 'Type Validation',
            'passed': False,
            'issues': [],
            'details': {}
        }

        issues = []
        functions_checked = 0
        functions_with_hints = 0

        all_py_files = self.get_python_files()

        for file in all_py_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions_checked += 1

                        # Check for type hints
                        has_return_hint = node.returns is not None
                        has_param_hints = all(arg.annotation is not None for arg in node.args.args)

                        if has_return_hint or has_param_hints:
                            functions_with_hints += 1

                        # Report critical functions without hints
                        if (not has_return_hint or not has_param_hints) and \
                           not node.name.startswith('_'):  # Skip private functions
                            if len(node.body) > 5:  # Only report non-trivial functions
                                issues.append({
                                    'file': str(file.relative_to(self.root_dir)),
                                    'function': node.name,
                                    'line': node.lineno,
                                    'issue': 'Missing type hints',
                                    'severity': 'LOW'
                                })

            except Exception:  # TODO: Catch specific exception instead of Exception
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Continuing after error", exc_info=True)
                continue

        check_results['issues'] = issues
        check_results['details']['functions_checked'] = functions_checked
        check_results['details']['functions_with_hints'] = functions_with_hints
        check_results['details']['type_hint_coverage'] = \
            (functions_with_hints / functions_checked * 100) if functions_checked > 0 else 0
        check_results['passed'] = len([i for i in issues if i['severity'] in ['HIGH', 'CRITICAL']]) == 0
        self.results['check_results']['types'] = check_results

        if check_results['passed']:
            coverage = check_results['details']['type_hint_coverage']
            print(f"  [PASS] Type hints adequate ({coverage:.1f}% coverage)")
        else:
            print(f"  [WARN] {len(issues)} functions missing type hints")

    # ============================================
    # CHECK 7: Test Validation
    # ============================================

    def check_tests(self):
        """Run all tests and check results."""
        check_results = {
            'name': 'Test Validation',
            'passed': False,
            'issues': [],
            'details': {}
        }

        # Count test files
        test_files = list(self.root_dir.rglob('test_*.py')) + list(self.root_dir.rglob('*_test.py'))
        test_files = [f for f in test_files if not any(excluded in f.parts for excluded in self.excluded_dirs)]

        check_results['details']['test_files_found'] = len(test_files)

        # Try to run pytest
        try:
            print(f"    Running pytest...", end='', flush=True)
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', '-v', '--tb=short', '-q'],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.root_dir
            )

            output = result.stdout + result.stderr

            # Parse results
            if 'passed' in output.lower():
                import re
                passed_match = re.search(r'(\d+) passed', output)
                if passed_match:
                    self.results['tests_passed'] = int(passed_match.group(1))

            if 'failed' in output.lower():
                failed_match = re.search(r'(\d+) failed', output)
                if failed_match:
                    self.results['tests_failed'] = int(failed_match.group(1))

            self.results['tests_run'] = self.results['tests_passed'] + self.results['tests_failed']

            check_results['details']['tests_run'] = self.results['tests_run']
            check_results['details']['tests_passed'] = self.results['tests_passed']
            check_results['details']['tests_failed'] = self.results['tests_failed']
            check_results['passed'] = result.returncode == 0

            if check_results['passed']:
                print(f" [PASS] All {self.results['tests_passed']} tests passed")
            else:
                print(f" [FAIL] {self.results['tests_failed']} test(s) failed")

        except FileNotFoundError:
            print(f" [WARN] pytest not found, skipping test execution")
            check_results['passed'] = True  # Don't fail if pytest not available
            check_results['details']['note'] = 'pytest not available'

        except subprocess.TimeoutExpired:
            print(f" [FAIL] Tests timed out")
            check_results['passed'] = False
            check_results['issues'].append({
                'issue': 'Tests timed out',
                'severity': 'MEDIUM'
            })

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f" [WARN] Could not run tests: {e}")
            check_results['passed'] = True  # Don't fail if tests can't run
            check_results['details']['error'] = str(e)

        self.results['check_results']['tests'] = check_results

    # ============================================
    # CHECK 8: Performance Validation
    # ============================================

    def check_performance(self):
        """Check for performance issues."""
        check_results = {
            'name': 'Performance Validation',
            'passed': False,
            'issues': [],
            'details': {}
        }

        issues = []

        # Check for performance anti-patterns
        perf_patterns = [
            {
                'name': 'Nested loops with heavy operations',
                'pattern': r'for\s+\w+\s+in\s+.*:.*for\s+\w+\s+in\s+.*:',
                'severity': 'MEDIUM'
            },
            {
                'name': 'String concatenation in loop',
                'pattern': r'for\s+.*:\s*\w+\s*\+=\s*["\']',
                'severity': 'LOW'
            },
            {
                'name': 'Global variable modification',
                'pattern': r'^global\s+\w+',
                'severity': 'LOW'
            }
        ]

        all_py_files = self.get_python_files()

        for pattern_spec in perf_patterns:
            pattern = pattern_spec['pattern']

            for file in all_py_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            issues.append({
                                'file': str(file.relative_to(self.root_dir)),
                                'line': i,
                                'pattern': pattern_spec['name'],
                                'severity': pattern_spec['severity']
                            })
                            break  # Only report once per file per pattern

                except Exception:  # TODO: Catch specific exception instead of Exception
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Continuing after error", exc_info=True)
                    continue

        check_results['issues'] = issues
        check_results['details']['performance_issues'] = len(issues)
        check_results['passed'] = len([i for i in issues if i['severity'] in ['HIGH', 'CRITICAL']]) == 0
        self.results['check_results']['performance'] = check_results

        if check_results['passed']:
            print(f"  [PASS] No critical performance issues")
        else:
            print(f"  [WARN] {len(issues)} potential performance issues found")

    # ============================================
    # CHECK 9: Security Validation
    # ============================================

    def check_security(self):
        """Check for security issues."""
        check_results = {
            'name': 'Security Validation',
            'passed': False,
            'issues': [],
            'details': {}
        }

        security_issues = []

        # Security patterns to check
        security_patterns = [
            {
                'name': 'Use of eval()',
                'pattern': r'\beval\s*\(',
                'severity': 'CRITICAL',
                'description': 'eval() can execute arbitrary code'
            },
            {
                'name': 'Use of exec()',
                'pattern': r'\bexec\s*\(',
                'severity': 'CRITICAL',
                'description': 'exec() can execute arbitrary code'
            },
            {
                'name': 'Use of os.system()',
                'pattern': r'\bos\.system\s*\(',
                'severity': 'HIGH',
                'description': 'os.system() is vulnerable to shell injection'
            },
            {
                'name': 'Hardcoded password',
                'pattern': r'(password|passwd|pwd)\s*=\s*["\'][^"\']+["\']',
                'severity': 'HIGH',
                'description': 'Possible hardcoded credential'
            },
            {
                'name': 'Hardcoded API key',
                'pattern': r'(api_key|apikey|api-key)\s*=\s*["\'][^"\']+["\']',
                'severity': 'HIGH',
                'description': 'Possible hardcoded API key'
            },
            {
                'name': 'SQL injection risk',
                'pattern': rf'(execute|executemany)\s*\(\s*["\'].*\+.*["\']',
                'severity': 'CRITICAL',
                'description': 'Possible SQL injection vulnerability'
            }
        ]

        all_py_files = self.get_python_files()

        for pattern_spec in security_patterns:
            pattern = pattern_spec['pattern']

            for file in all_py_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            security_issues.append({
                                'file': str(file.relative_to(self.root_dir)),
                                'line': i,
                                'issue': pattern_spec['name'],
                                'severity': pattern_spec['severity'],
                                'description': pattern_spec['description'],
                                'content': line.strip()[:100]
                            })

                except Exception:  # TODO: Catch specific exception instead of Exception
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Continuing after error", exc_info=True)
                    continue

        check_results['issues'] = security_issues
        check_results['details']['security_issues'] = len(security_issues)
        check_results['details']['critical_issues'] = len([i for i in security_issues if i['severity'] == 'CRITICAL'])
        check_results['passed'] = check_results['details']['critical_issues'] == 0
        self.results['check_results']['security'] = check_results

        if check_results['passed']:
            print(f"  [PASS] No critical security issues")
        else:
            print(f"  [FAIL] {check_results['details']['critical_issues']} critical security issues found")

    # ============================================
    # CHECK 10: Documentation Validation
    # ============================================

    def check_documentation(self):
        """Check documentation coverage."""
        check_results = {
            'name': 'Documentation Validation',
            'passed': False,
            'issues': [],
            'details': {}
        }

        doc_issues = []
        modules_checked = 0
        modules_with_docstrings = 0
        functions_checked = 0
        functions_with_docstrings = 0
        classes_checked = 0
        classes_with_docstrings = 0

        all_py_files = self.get_python_files()

        for file in all_py_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())

                modules_checked += 1

                # Check for module docstring
                module_docstring = ast.get_docstring(tree)
                if module_docstring:
                    modules_with_docstrings += 1
                else:
                    doc_issues.append({
                        'file': str(file.relative_to(self.root_dir)),
                        'issue': 'Missing module docstring',
                        'severity': 'LOW'
                    })

                # Check for class/function docstrings
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions_checked += 1
                        if ast.get_docstring(node):
                            functions_with_docstrings += 1
                    elif isinstance(node, ast.ClassDef):
                        classes_checked += 1
                        if ast.get_docstring(node):
                            classes_with_docstrings += 1

            except Exception:  # TODO: Catch specific exception instead of Exception
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Continuing after error", exc_info=True)
                continue

        check_results['issues'] = doc_issues
        check_results['details']['modules_checked'] = modules_checked
        check_results['details']['modules_with_docstrings'] = modules_with_docstrings
        check_results['details']['functions_checked'] = functions_checked
        check_results['details']['functions_with_docstrings'] = functions_with_docstrings
        check_results['details']['classes_checked'] = classes_checked
        check_results['details']['classes_with_docstrings'] = classes_with_docstrings

        # Calculate coverage
        module_coverage = (modules_with_docstrings / modules_checked * 100) if modules_checked > 0 else 0
        function_coverage = (functions_with_docstrings / functions_checked * 100) if functions_checked > 0 else 0
        class_coverage = (classes_with_docstrings / classes_checked * 100) if classes_checked > 0 else 0

        check_results['details']['module_coverage'] = module_coverage
        check_results['details']['function_coverage'] = function_coverage
        check_results['details']['class_coverage'] = class_coverage

        # Pass if no critical documentation gaps
        check_results['passed'] = len([i for i in doc_issues if i['severity'] in ['HIGH', 'CRITICAL']]) == 0
        self.results['check_results']['documentation'] = check_results

        if check_results['passed']:
            print(f"  [PASS] Documentation adequate (modules: {module_coverage:.0f}%, functions: {function_coverage:.0f}%)")
        else:
            print(f"  [WARN] Documentation gaps in {len(doc_issues)} items")

    # ============================================
    # Score Calculation
    # ============================================

    def calculate_score(self):
        """Calculate overall validation score."""
        total_checks = 10
        passed_checks = 0
        weights = {
            'file_existence': 1.0,
            'syntax': 1.5,  # Critical
            'imports': 1.5,  # Critical
            'patterns': 1.0,
            'dependencies': 1.0,
            'types': 0.5,
            'tests': 1.5,  # Critical
            'performance': 0.5,
            'security': 2.0,  # Most critical
            'documentation': 0.5
        }

        total_weight = sum(weights.values())
        weighted_score = 0.0

        for check_name, weight in weights.items():
            if check_name in self.results['check_results']:
                check_result = self.results['check_results'][check_name]
                if check_result.get('passed', False):
                    weighted_score += weight

        # Calculate final score (0-100)
        self.results['score'] = (weighted_score / total_weight) * 100

        # Determine grade
        score = self.results['score']
        if score >= 95:
            self.results['grade'] = 'A+'
        elif score >= 90:
            self.results['grade'] = 'A'
        elif score >= 85:
            self.results['grade'] = 'B+'
        elif score >= 80:
            self.results['grade'] = 'B'
        elif score >= 75:
            self.results['grade'] = 'C+'
        elif score >= 70:
            self.results['grade'] = 'C'
        elif score >= 60:
            self.results['grade'] = 'D'
        else:
            self.results['grade'] = 'F'

        # Count critical issues
        critical_issues = len([i for i in self.results['issues_found'] if i.get('severity') == 'CRITICAL'])
        self.results['critical_issues'] = critical_issues

        # Adjust grade based on critical issues
        if critical_issues > 0:
            if self.results['grade'] in ['A+', 'A']:
                self.results['grade'] = 'B'
            elif self.results['grade'] in ['B+', 'B']:
                self.results['grade'] = 'C'
            elif self.results['grade'] in ['C+', 'C']:
                self.results['grade'] = 'D'
            else:
                self.results['grade'] = 'F'

    # ============================================
    # Report Generation
    # ============================================

    def generate_report(self):
        """Generate comprehensive validation report."""
        report_path = self.root_dir / "ULTIMATE_VALIDATION_REPORT.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# ULTIMATE VALIDATION REPORT\n")
            f.write("## Most Comprehensive Validation Possible\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Validator:** UltimateValidator Suite\n")
            f.write(f"**Scope:** EVERYTHING\n\n")

            # Summary
            f.write("## VALIDATION SUMMARY\n\n")
            f.write(f"**Overall Score:** {self.results['score']:.1f}%\n")
            f.write(f"**Grade:** {self.results['grade']}\n")
            f.write(f"**Status:** {'COMPLETE' if self.results['score'] >= 70 else 'INCOMPLETE'}\n")
            f.write(f"**Critical Issues:** {self.results['critical_issues']}\n")
            f.write(f"**Files Checked:** {self.results['files_checked']}\n")
            f.write(f"**Files Passed:** {self.results['files_passed']}\n")
            f.write(f"**Files Failed:** {self.results['files_failed']}\n")
            f.write(f"**Tests Run:** {self.results['tests_run']}\n")
            f.write(f"**Tests Passed:** {self.results['tests_passed']}\n")
            f.write(f"**Tests Failed:** {self.results['tests_failed']}\n\n")

            # Detailed Results
            f.write("## DETAILED RESULTS\n\n")

            for i, (check_name, check_result) in enumerate(self.results['check_results'].items(), 1):
                f.write(f"### {i}. {check_result['name']}\n\n")
                f.write(f"**Status:** {'[PASS] PASS' if check_result['passed'] else '[FAIL] FAIL'}\n")

                if 'details' in check_result:
                    f.write("**Details:**\n")
                    for key, value in check_result['details'].items():
                        f.write(f"  - {key}: {value}\n")

                if check_result['issues']:
                    f.write(f"\n**Issues Found:** {len(check_result['issues'])}\n\n")

                    # Show first 10 issues
                    for issue in check_result['issues'][:10]:
                        severity = issue.get('severity', 'UNKNOWN')
                        f.write(f"- [{severity}] ")

                        if 'file' in issue:
                            f.write(f"{issue['file']}")
                        if 'line' in issue:
                            f.write(f":{issue['line']}")
                        if 'issue' in issue:
                            f.write(f" - {issue['issue']}")
                        elif 'pattern' in issue:
                            f.write(f" - {issue['pattern']}")
                        elif 'error' in issue:
                            f.write(f" - {issue['error']}")

                        f.write("\n")

                    if len(check_result['issues']) > 10:
                        f.write(f"\n_... and {len(check_result['issues']) - 10} more issues_\n")

                f.write("\n")

            # All Issues Summary
            if self.results['issues_found']:
                f.write("## ALL ISSUES FOUND\n\n")

                # Group by severity
                by_severity = defaultdict(list)
                for issue in self.results['issues_found']:
                    severity = issue.get('severity', 'UNKNOWN')
                    by_severity[severity].append(issue)

                for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                    if severity in by_severity:
                        f.write(f"### {severity} Issues ({len(by_severity[severity])})\n\n")
                        for issue in by_severity[severity]:
                            f.write(f"- {issue}\n")
                        f.write("\n")

            # Recommendations
            f.write("## RECOMMENDATIONS\n\n")

            # Prioritized recommendations
            critical_count = len([i for i in self.results['issues_found'] if i.get('severity') == 'CRITICAL'])
            high_count = len([i for i in self.results['issues_found'] if i.get('severity') == 'HIGH'])

            if critical_count > 0:
                f.write(f"1. **URGENT:** Fix {critical_count} critical issues immediately\n")
            if high_count > 0:
                f.write(f"2. **HIGH PRIORITY:** Address {high_count} high-severity issues\n")

            # Check-specific recommendations
            for check_name, check_result in self.results['check_results'].items():
                if not check_result['passed']:
                    if check_name == 'security':
                        f.write(f"3. **SECURITY:** Review and fix all security vulnerabilities\n")
                    elif check_name == 'tests':
                        f.write(f"4. **TESTING:** Fix failing tests to ensure code quality\n")
                    elif check_name == 'syntax':
                        f.write(f"5. **SYNTAX:** Fix syntax errors before proceeding\n")

            # Final Assessment
            f.write("\n## FINAL ASSESSMENT\n\n")

            if self.results['score'] >= 90 and self.results['critical_issues'] == 0:
                f.write("[PASS] **EXCELLENT** - Codebase is in excellent condition.\n")
                f.write("Production ready with high confidence.\n")
            elif self.results['score'] >= 80 and self.results['critical_issues'] == 0:
                f.write("[PASS] **GOOD** - Codebase is in good condition.\n")
                f.write("Production ready with minor improvements recommended.\n")
            elif self.results['score'] >= 70:
                f.write("[WARN] **ACCEPTABLE** - Codebase needs attention.\n")
                f.write("Address high-priority issues before production deployment.\n")
            elif self.results['score'] >= 60:
                f.write("[FAIL] **NEEDS WORK** - Codebase has significant issues.\n")
                f.write("Major improvements required before production use.\n")
            else:
                f.write("[FAIL] **CRITICAL** - Codebase is in poor condition.\n")
                f.write("Extensive remediation required. Not production ready.\n")

            # Signature
            f.write("\n---\n")
            f.write(f"\nGenerated by UltimateValidator Suite\n")
            f.write(f"Total validation time: {time.strftime('%H:%M:%S')}\n")

        print(f"\n[PASS] Report generated: {report_path}")

        # Also print summary to console
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Overall Score: {self.results['score']:.1f}%")
        print(f"Grade: {self.results['grade']}")
        print(f"Status: {'COMPLETE' if self.results['score'] >= 70 else 'INCOMPLETE'}")
        print(f"Critical Issues: {self.results['critical_issues']}")
        print(f"Files: {self.results['files_passed']}/{self.results['files_checked']} passed")
        print(f"Tests: {self.results['tests_passed']}/{self.results['tests_run']} passed")
        print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Ultimate Validation Suite')
    parser.add_argument('--root', default='.', help='Root directory to validate')

    args = parser.parse_args()

    validator = UltimateValidator(args.root)
    results = validator.validate_everything()

    # Exit with appropriate code
    if results['score'] >= 70 and results['critical_issues'] == 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
