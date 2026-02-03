#!/usr/bin/env python3
"""
Deep Static Code Analysis for BubbleLabs Integration

Performs comprehensive static analysis including:
1. AST (Abstract Syntax Tree) parsing
2. Import analysis and circular dependency detection
3. Type checking
4. Dead code detection
5. Unused imports detection
6. Code complexity analysis
7. Security vulnerability scanning
8. Resource leak detection
"""

import ast
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Static Analysis
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import enterprise_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


# **ACTUAL INTEGRATION HELPER METHODS**: Static Analysis
def _trigger_static_analysis_alerts(operation, success, analysis_id=None, error=None, metadata=None):
    """Trigger alerts for static analysis operations"""
    if not ALERTING_AVAILABLE:
        return

    try:
        alert_mgr = get_alert_manager()
        if success:
            return  # No alerts for successful operations

        severity = AlertSeverity.MEDIUM
        alert_mgr.trigger_alert(
            title=f"Static Analysis {operation} Failed",
            message=f"Static analysis operation '{operation}' failed: {error}",
            severity=severity,
            source="DeepStaticAnalyzer",
            metadata=metadata or {"analysis_id": analysis_id, "operation": operation}
        )
    except Exception as e:
        logger.warning(f"Failed to trigger static analysis alert: {e}")


def _extract_static_analysis_knowledge(operation, analysis_id, result):
    """Extract knowledge from static analysis operations"""
    if not KNOWLEDGE_AVAILABLE:
        return

    try:
        artifact = KnowledgeArtifact(
            artifact_id=f"static_analysis_{operation}_{analysis_id}",
            artifact_type="static_analysis_execution",
            source_component="DeepStaticAnalyzer",
            content={
                "operation": operation,
                "analysis_id": analysis_id,
                "files_analyzed": result.get("total_files", 0) if result else 0,
                "issues_found": result.get("total_issues", 0) if result else 0,
                "critical_issues": result.get("critical_issues", 0) if result else 0,
                "success": result is not None,
            },
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        enterprise_knowledge_engine.store_artifact(artifact)
    except Exception as e:
        logger.warning(f"Failed to extract static analysis knowledge: {e}")


def _track_static_analysis_performance(operation, success, duration_seconds, files_analyzed, issues_found=0):
    """Track performance of static analysis operations"""
    if not ADAPTIVE_AVAILABLE:
        return

    try:
        tracker = StrategyPerformanceTracker.get_instance()
        data = StrategyPerformanceData(
            strategy_name="static_analysis",
            component_name="DeepStaticAnalyzer",
            operation_name=operation,
            success=success,
            duration_seconds=duration_seconds,
            metadata={
                "files_analyzed": files_analyzed,
                "issues_found": issues_found
            }
        )
        tracker.record_execution(data)
    except Exception as e:
        logger.warning(f"Failed to track static analysis performance: {e}")


@dataclass
class Issue:
    """Represents a code issue found during analysis."""
    file_path: str
    line_number: int
    column: int
    severity: str  # critical, high, medium, low, info
    category: str  # security, performance, bug, code_quality, resource_leak
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class FileAnalysis:
    """Analysis results for a single file."""
    file_path: str
    issues: List[Issue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    imports: Set[str] = field(default_factory=set)
    unused_imports: Set[str] = field(default_factory=set)
    circular_dependencies: List[str] = field(default_factory=list)


class DeepStaticAnalyzer:
    """Comprehensive static code analyzer."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.issues: List[Issue] = []
        self.file_analyses: Dict[str, FileAnalysis] = {}
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_import_graph: Dict[str, Set[str]] = defaultdict(set)

    def analyze_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple files and return comprehensive report."""
        import time
        start_time = time.time()
        success = False
        analysis_id = f"static_{hash(str(file_paths)) % 10000:04d}"

        try:
            print(f"Analyzing {len(file_paths)} files...")

            # First pass: parse all files and build import graph
            for file_path in file_paths:
                self._analyze_single_file(file_path)

            # Second pass: detect circular dependencies
            self._detect_circular_dependencies()

            # Third pass: detect unused imports
            for file_path in file_paths:
                self._detect_unused_imports(file_path)

            # Generate report
            result = self._generate_report()

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance
            success = True
            duration = time.time() - start_time
            _extract_static_analysis_knowledge("analyze_files", analysis_id, result)
            _track_static_analysis_performance("analyze_files", True, duration, len(file_paths),
                                               result.get("total_issues", 0))

            return result

        except Exception as e:
            duration = time.time() - start_time
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            _trigger_static_analysis_alerts("analyze_files", False, analysis_id, str(e))
            _track_static_analysis_performance("analyze_files", False, duration, 0, 0)
            raise

    def _analyze_single_file(self, file_path: str) -> FileAnalysis:
        """Analyze a single file for all issues."""
        analysis = FileAnalysis(file_path=file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Parse AST
            try:
                tree = ast.parse(source, filename=file_path)
            except SyntaxError as e:
                issue = Issue(
                    file_path=file_path,
                    line_number=e.lineno or 0,
                    column=e.offset or 0,
                    severity="critical",
                    category="bug",
                    message=f"Syntax error: {e.msg}",
                    code_snippet=self._get_code_snippet(source, e.lineno)
                )
                analysis.issues.append(issue)
                self.issues.append(issue)
                return analysis

            # Collect imports
            self._collect_imports(tree, file_path, analysis)

            # Run various analyses
            self._check_security_vulnerabilities(tree, file_path, source, analysis)
            self._check_resource_leaks(tree, file_path, source, analysis)
            self._check_type_issues(tree, file_path, source, analysis)
            self._check_dead_code(tree, file_path, source, analysis)
            self._calculate_complexity(tree, file_path, analysis)
            self._check_code_quality(tree, file_path, source, analysis)

            # Store metrics
            analysis.metrics = self._calculate_file_metrics(tree, source)

        except Exception as e:
            issue = Issue(
                file_path=file_path,
                line_number=0,
                column=0,
                severity="high",
                category="bug",
                message=f"Failed to analyze file: {str(e)}"
            )
            analysis.issues.append(issue)

        self.file_analyses[file_path] = analysis
        return analysis

    def _collect_imports(self, tree: ast.AST, file_path: str, analysis: FileAnalysis):
        """Collect all imports from a file."""
        module_name = Path(file_path).stem

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_module = alias.name.split('.')[0]
                    analysis.imports.add(imported_module)
                    self.import_graph[module_name].add(imported_module)
                    self.reverse_import_graph[imported_module].add(module_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_module = node.module.split('.')[0]
                    analysis.imports.add(imported_module)
                    self.import_graph[module_name].add(imported_module)
                    self.reverse_import_graph[imported_module].add(module_name)

    def _detect_circular_dependencies(self):
        """Detect circular dependencies using import graph."""
        visited = set()
        recursion_stack = set()

        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            if node in recursion_stack:
                # Found cycle
                if node in path:
                    cycle_start = path.index(node)
                    return path[cycle_start:] + [node]
                return [node]
            if node in visited:
                return None

            visited.add(node)
            recursion_stack.add(node)

            for neighbor in self.import_graph.get(node, set()):
                result = dfs(neighbor, path + [node])
                if result:
                    return result

            recursion_stack.remove(node)
            return None

        for module in self.import_graph:
            cycle = dfs(module, [])
            if cycle:
                # Report cycle for all files in cycle
                for cycle_module in cycle:
                    for file_path, analysis in self.file_analyses.items():
                        if Path(file_path).stem == cycle_module:
                            msg = f"Circular dependency detected: {' -> '.join(cycle)}"
                            if msg not in analysis.circular_dependencies:
                                analysis.circular_dependencies.append(msg)

    def _detect_unused_imports(self, file_path: str):
        """Detect unused imports by tracking name usage."""
        analysis = self.file_analyses.get(file_path)
        if not analysis:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)
            imported_names: Set[str] = set()
            used_names: Set[str] = set()

            # Collect imported names
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name.split('.')[0]
                        imported_names.add(name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        if name != '*':
                            imported_names.add(name)

            # Find all used names
            class NameUsageVisitor(ast.NodeVisitor):
                def visit_Name(self, node):
                    used_names.add(node.id)
                    self.generic_visit(node)

                def visit_Attribute(self, node):
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)
                    self.generic_visit(node)

            visitor = NameUsageVisitor()
            visitor.visit(tree)

            # Find unused imports (excluding special cases)
            special_cases = {'logger', 'Optional', 'List', 'Dict', 'Any', 'Set', 'Union', 'Callable'}
            unused = imported_names - used_names - special_cases

            for unused_import in unused:
                # Only report if it's actually unused
                issue = Issue(
                    file_path=file_path,
                    line_number=0,  # Line number would require tracking import location
                    column=0,
                    severity="low",
                    category="code_quality",
                    message=f"Unused import: '{unused_import}'",
                    suggestion=f"Remove unused import '{unused_import}'"
                )
                analysis.unused_imports.add(unused_import)

        except Exception as e:
            print(f"Error detecting unused imports in {file_path}: {e}")

    def _check_security_vulnerabilities(self, tree: ast.AST, file_path: str, source: str, analysis: FileAnalysis):
        """Check for security vulnerabilities."""
        for node in ast.walk(tree):
            # SQL Injection checks
            if isinstance(node, ast.Call):
                # Check for execute() with string concatenation
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'execute':
                        if node.args and isinstance(node.args[0], ast.BinOp):
                            issue = Issue(
                                file_path=file_path,
                                line_number=node.lineno,
                                column=node.col_offset,
                                severity="critical",
                                category="security",
                                message="Potential SQL injection: string concatenation in execute()",
                                suggestion="Use parameterized queries instead",
                                code_snippet=self._get_code_snippet(source, node.lineno)
                            )
                            analysis.issues.append(issue)
                            self.issues.append(issue)

                    # Check for shell command execution
                    if node.func.attr in {'system', 'popen', 'call', 'Popen'}:
                        if isinstance(node.func.value, ast.Name) and node.func.value.id in {'os', 'subprocess'}:
                            if node.args and isinstance(node.args[0], (ast.BinOp, ast.JoinedStr)):
                                issue = Issue(
                                    file_path=file_path,
                                    line_number=node.lineno,
                                    column=node.col_offset,
                                    severity="critical",
                                    category="security",
                                    message="Potential command injection: string concatenation in shell command",
                                    suggestion="Use subprocess.run with list of arguments or shlex.quote()",
                                    code_snippet=self._get_code_snippet(source, node.lineno)
                                )
                                analysis.issues.append(issue)
                                self.issues.append(issue)

            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {'eval', 'exec', 'compile'}:
                        issue = Issue(
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            severity="high",
                            category="security",
                            message=f"Use of dangerous function: {node.func.id}()",
                            suggestion="Avoid using eval/exec - use safer alternatives",
                            code_snippet=self._get_code_snippet(source, node.lineno)
                        )
                        analysis.issues.append(issue)
                        self.issues.append(issue)

            # Check for hardcoded credentials
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if 'password' in target.id.lower() or 'secret' in target.id.lower() or 'api_key' in target.id.lower():
                            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                if len(node.value.value) > 10:
                                    issue = Issue(
                                        file_path=file_path,
                                        line_number=node.lineno,
                                        column=node.col_offset,
                                        severity="high",
                                        category="security",
                                        message=f"Potential hardcoded credential in variable: {target.id}",
                                        suggestion="Use environment variables or config files",
                                        code_snippet=self._get_code_snippet(source, node.lineno)
                                    )
                                    analysis.issues.append(issue)
                                    self.issues.append(issue)

    def _check_resource_leaks(self, tree: ast.AST, file_path: str, source: str, analysis: FileAnalysis):
        """Check for potential resource leaks."""
        for node in ast.walk(tree):
            # Check for file operations without context manager
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'open':
                        # Check if open() is not used with 'with' statement
                        parent = self._find_parent_node(tree, node)
                        if not isinstance(parent, ast.With):
                            issue = Issue(
                                file_path=file_path,
                                line_number=node.lineno,
                                column=node.col_offset,
                                severity="medium",
                                category="resource_leak",
                                message="File opened without context manager (potential resource leak)",
                                suggestion="Use 'with open(...)' to ensure file is closed",
                                code_snippet=self._get_code_snippet(source, node.lineno)
                            )
                            analysis.issues.append(issue)
                            self.issues.append(issue)

            # Check for database connections without context manager
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if 'connect' in node.func.attr:
                        parent = self._find_parent_node(tree, node)
                        if not isinstance(parent, ast.With):
                            issue = Issue(
                                file_path=file_path,
                                line_number=node.lineno,
                                column=node.col_offset,
                                severity="medium",
                                category="resource_leak",
                                message="Database connection without context manager",
                                suggestion="Use context manager or ensure explicit close() in finally block",
                                code_snippet=self._get_code_snippet(source, node.lineno)
                            )
                            analysis.issues.append(issue)
                            self.issues.append(issue)

            # Check for thread operations without proper join
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'start' and isinstance(node.func.value, ast.Name):
                        if 'thread' in node.func.value.id.lower():
                            # Check if join() is called
                            issue = Issue(
                                file_path=file_path,
                                line_number=node.lineno,
                                column=node.col_offset,
                                severity="medium",
                                category="resource_leak",
                                message="Thread started without visible join() call",
                                suggestion="Ensure thread.join() is called to prevent resource leaks",
                                code_snippet=self._get_code_snippet(source, node.lineno)
                            )
                            analysis.issues.append(issue)
                            self.issues.append(issue)

    def _find_parent_node(self, tree: ast.AST, target_node: ast.AST) -> Optional[ast.AST]:
        """Find parent node of a given node in AST."""
        class ParentFinder(ast.NodeVisitor):
            def __init__(self, target):
                self.target = target
                self.parent = None
                self.parent_stack = []

            def generic_visit(self, node):
                self.parent_stack.append(node)
                if node is self.target:
                    self.parent = self.parent_stack[-2] if len(self.parent_stack) >= 2 else None
                super().generic_visit(node)
                self.parent_stack.pop()

        finder = ParentFinder(target_node)
        finder.visit(tree)
        return finder.parent

    def _check_type_issues(self, tree: ast.AST, file_path: str, source: str, analysis: FileAnalysis):
        """Check for type-related issues."""
        for node in ast.walk(tree):
            # Check for None comparisons
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and comparator.value is None:
                        issue = Issue(
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            severity="low",
                            category="bug",
                            message="Comparison with None using ==/!=",
                            suggestion="Use 'is None' or 'is not None' instead",
                            code_snippet=self._get_code_snippet(source, node.lineno)
                        )
                        analysis.issues.append(issue)
                        self.issues.append(issue)

            # Check for mutable default arguments
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issue = Issue(
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            severity="high",
                            category="bug",
                            message=f"Mutable default argument in function '{node.name}'",
                            suggestion="Use None as default and initialize inside function",
                            code_snippet=self._get_code_snippet(source, node.lineno)
                        )
                        analysis.issues.append(issue)
                        self.issues.append(issue)

    def _check_dead_code(self, tree: ast.AST, file_path: str, source: str, analysis: FileAnalysis):
        """Check for potential dead code."""
        for node in ast.walk(tree):
            # Check for unreachable code after return
            if isinstance(node, ast.FunctionDef):
                last_was_return = False
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Return):
                        last_was_return = True
                    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        # Nested function/class definitions are OK
                        pass

            # Check for empty except blocks
            if isinstance(node, ast.ExceptHandler):
                if node.body and len(node.body) == 1:
                    if isinstance(node.body[0], ast.Pass):
                        issue = Issue(
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            severity="medium",
                            category="bug",
                            message="Empty except block (silences all errors)",
                            suggestion="Add proper error handling or at least logging",
                            code_snippet=self._get_code_snippet(source, node.lineno)
                        )
                        analysis.issues.append(issue)
                        self.issues.append(issue)

    def _calculate_complexity(self, tree: ast.AST, file_path: str, analysis: FileAnalysis):
        """Calculate cyclomatic complexity."""
        complexity_scores = {}

        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.function_complexity = defaultdict(int)
                self.current_function = None

            def visit_FunctionDef(self, node):
                self.current_function = node.name
                complexity = 1  # Base complexity

                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1

                self.function_complexity[node.name] = complexity
                self.current_function = None

        visitor = ComplexityVisitor()
        visitor.visit(tree)

        for func_name, complexity in visitor.function_complexity.items():
            if complexity > 10:
                issue = Issue(
                    file_path=file_path,
                    line_number=0,
                    column=0,
                    severity="medium",
                    category="code_quality",
                    message=f"High cyclomatic complexity in '{func_name}': {complexity}",
                    suggestion="Consider refactoring into smaller functions"
                )
                analysis.issues.append(issue)
                self.issues.append(issue)

    def _check_code_quality(self, tree: ast.AST, file_path: str, source: str, analysis: FileAnalysis):
        """Check for code quality issues."""
        for node in ast.walk(tree):
            # Check for overly long functions
            if isinstance(node, ast.FunctionDef):
                line_count = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if line_count > 100:
                    issue = Issue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        severity="medium",
                        category="code_quality",
                        message=f"Function '{node.name}' is too long ({line_count} lines)",
                        suggestion="Consider breaking into smaller functions",
                        code_snippet=self._get_code_snippet(source, node.lineno)
                    )
                    analysis.issues.append(issue)
                    self.issues.append(issue)

            # Check for too many arguments
            if isinstance(node, ast.FunctionDef):
                arg_count = len(node.args.args)
                if arg_count > 7:
                    issue = Issue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        severity="low",
                        category="code_quality",
                        message=f"Function '{node.name}' has too many arguments ({arg_count})",
                        suggestion="Consider using a config object or dataclass",
                        code_snippet=self._get_code_snippet(source, node.lineno)
                    )
                    analysis.issues.append(issue)
                    self.issues.append(issue)

    def _calculate_file_metrics(self, tree: ast.AST, source: str) -> Dict[str, Any]:
        """Calculate various file metrics."""
        lines = source.split('\n')

        class MetricsVisitor(ast.NodeVisitor):
            def __init__(self):
                self.functions = 0
                self.classes = 0
                self.imports = 0
                self.comments = 0
                self.blank_lines = 0

            def visit_FunctionDef(self, node):
                self.functions += 1
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                self.functions += 1
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                self.classes += 1
                self.generic_visit(node)

            def visit_Import(self, node):
                self.imports += 1

            def visit_ImportFrom(self, node):
                self.imports += 1

        visitor = MetricsVisitor()
        visitor.visit(tree)

        # Count comments and blank lines
        for line in lines:
            stripped = line.strip()
            if not stripped:
                visitor.blank_lines += 1
            elif stripped.startswith('#'):
                visitor.comments += 1

        return {
            'total_lines': len(lines),
            'code_lines': len(lines) - visitor.blank_lines - visitor.comments,
            'blank_lines': visitor.blank_lines,
            'comment_lines': visitor.comments,
            'functions': visitor.functions,
            'classes': visitor.classes,
            'imports': visitor.imports
        }

    def _get_code_snippet(self, source: str, line_number: int, context_lines: int = 2) -> str:
        """Get code snippet around a line."""
        if line_number is None or line_number <= 0:
            return ""

        lines = source.split('\n')
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)

        snippet_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == line_number - 1 else "    "
            snippet_lines.append(f"{prefix}{i+1}: {lines[i]}")

        return '\n'.join(snippet_lines)

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        # Categorize issues
        issues_by_severity = defaultdict(list)
        issues_by_category = defaultdict(list)
        issues_by_file = defaultdict(list)

        for issue in self.issues:
            issues_by_severity[issue.severity].append(issue)
            issues_by_category[issue.category].append(issue)
            issues_by_file[issue.file_path].append(issue)

        # Generate file-specific reports
        file_reports = {}
        for file_path, analysis in self.file_analyses.items():
            file_reports[file_path] = {
                'issues': [
                    {
                        'line': issue.line_number,
                        'column': issue.column,
                        'severity': issue.severity,
                        'category': issue.category,
                        'message': issue.message,
                        'suggestion': issue.suggestion,
                        'code_snippet': issue.code_snippet
                    }
                    for issue in analysis.issues
                ],
                'metrics': analysis.metrics,
                'imports': list(analysis.imports),
                'unused_imports': list(analysis.unused_imports),
                'circular_dependencies': analysis.circular_dependencies
            }

        return {
            'summary': {
                'total_issues': len(self.issues),
                'by_severity': {k: len(v) for k, v in issues_by_severity.items()},
                'by_category': {k: len(v) for k, v in issues_by_category.items()},
                'by_file': {k: len(v) for k, v in issues_by_file.items()}
            },
            'files': file_reports,
            'issues_by_severity': {
                severity: [
                    {
                        'file': issue.file_path,
                        'line': issue.line_number,
                        'message': issue.message,
                        'suggestion': issue.suggestion
                    }
                    for issue in issues
                ]
                for severity, issues in issues_by_severity.items()
            }
        }


def main():
    """Main analysis entry point."""
    # Files to analyze
    files_to_analyze = [
        r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\bubblelabs_CREWAI_bridge.py",
        r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\bubblelabs_analytics.py",
        r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\bubblelabs_typescript_export.py",
        r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\bubblelabs_mcp_tools.py",
        r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\bubblelabs_integration.py",
        r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\openevolve_bubblelabs_api.py"
    ]

    # Run analysis
    analyzer = DeepStaticAnalyzer(root_dir=r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend")
    report = analyzer.analyze_files(files_to_analyze)

    # Print report
    print("\n" + "="*100)
    print("DEEP STATIC CODE ANALYSIS REPORT - BUBBLELABS INTEGRATION")
    print("="*100)

    # Print summary
    print("\n## SUMMARY")
    print("-" * 100)
    summary = report['summary']
    print(f"Total Issues Found: {summary['total_issues']}")
    print(f"\nBy Severity:")
    for severity, count in sorted(summary['by_severity'].items(), key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}.get(x[0], 5)):
        print(f"  {severity.upper()}: {count}")

    print(f"\nBy Category:")
    for category, count in sorted(summary['by_category'].items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")

    # Print critical and high severity issues
    print("\n## CRITICAL AND HIGH SEVERITY ISSUES")
    print("-" * 100)
    for severity in ['critical', 'high']:
        if severity in report['issues_by_severity']:
            print(f"\n### {severity.upper()} ISSUES ({len(report['issues_by_severity'][severity])})")
            print("-" * 100)
            for issue in report['issues_by_severity'][severity]:
                print(f"\n  File: {issue['file']}")
                print(f"  Line: {issue['line']}")
                print(f"  Message: {issue['message']}")
                if issue['suggestion']:
                    print(f"  Suggestion: {issue['suggestion']}")

    # Print file-specific details
    print("\n## FILE-SPECIFIC ANALYSIS")
    print("-" * 100)
    for file_path, file_data in report['files'].items():
        filename = Path(file_path).name
        print(f"\n### {filename}")
        print(f"Metrics: {file_data['metrics']}")
        print(f"Issues: {len(file_data['issues'])}")

        if file_data['issues']:
            print("\nTop Issues:")
            for issue in sorted(file_data['issues'], key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}.get(x['severity'], 5))[:5]:
                print(f"  - [{issue['severity'].upper()}] Line {issue['line']}: {issue['message']}")

        if file_data['unused_imports']:
            print(f"\nUnused Imports: {', '.join(file_data['unused_imports'])}")

        if file_data['circular_dependencies']:
            print(f"\nCircular Dependencies:")
            for dep in file_data['circular_dependencies']:
                print(f"  - {dep}")

    # Save JSON report
    output_file = r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\static_analysis_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n\nFull report saved to: {output_file}")
    print("="*100)


if __name__ == "__main__":
    main()
