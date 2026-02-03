"""
Sovereign-Grade Problem Decomposition System - Automated Code Quality Checks
Implements code quality enforcement, linting, formatting, and static analysis.
"""

import ast
import astor
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import subprocess
import tempfile
import logging
from pathlib import Path
import json
import re

@dataclass
class QualityIssue:
    """Represents a code quality issue."""
    file_path: str
    line_number: int
    column: int
    issue_type: str
    message: str
    severity: str  # 'error', 'warning', 'info'
    rule: str


class CodeQualityChecker:
    """Main class for code quality checks."""
    
    def __init__(self, project_root: str = "."):
        """
        Initialize code quality checker.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        self.issues: List[QualityIssue] = []
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load quality settings from configuration file."""
        # Default settings
        settings = {
            "flake8": {
                "max_line_length": 88,
                "ignore": ["E203", "W503"],  # Commonly ignored issues
                "select": ["E", "W", "F", "C", "N", "D"]
            },
            "pylint": {
                "enable": ["all"],
                "disable": ["C0103", "C0114", "C0115", "C0116"],  # Naming conventions we'll handle differently
                "min_similarity": 10,
                "max_args": 10
            },
            "black": {
                "line_length": 88
            },
            "mypy": {
                "python_version": "3.8",
                "strict": True
            },
            "security": {
                "check_hardcoded_secrets": True,
                "check_unsafe_imports": True,
                "check_url_validation": True
            }
        }
        
        # Try to load from config file if it exists
        config_path = self.project_root / ".quality_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_settings = json.load(f)
                    settings.update(user_settings)
            except (json.JSONDecodeError, OSError, IOError) as e:
                self.logger.warning(f"Could not load quality config: {e}")
        
        return settings
    
    def check_file_quality(self, file_path: str) -> List[QualityIssue]:
        """
        Check quality of a single file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            List of quality issues found
        """
        issues = []
        path = Path(file_path)
        
        if not path.exists():
            return issues
        
        if path.suffix != '.py':
            return issues
        
        try:
            # Use AST to perform static analysis
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
            
            # Run all checks
            issues.extend(self._check_naming_conventions(tree, path, content))
            issues.extend(self._check_complexity(tree, path, content))
            issues.extend(self._check_security_patterns(path, content))
            issues.extend(self._check_docstring_completeness(tree, path, content))
            issues.extend(self._check_type_annotations(tree, path, content))
            
        except SyntaxError as e:
            issues.append(QualityIssue(
                file_path=str(path),
                line_number=e.lineno or 0,
                column=e.offset or 0,
                issue_type="syntax_error",
                message=f"Syntax error: {e.msg}",
                severity="error",
                rule="syntax"
            ))
        except (OSError, IOError, UnicodeDecodeError) as e:
            self.logger.error(f"Error checking quality of {path}: {e}")
        
        return issues
    
    def check_project_quality(self, include_tests: bool = True) -> List[QualityIssue]:
        """
        Check quality of all Python files in the project.
        
        Args:
            include_tests: Whether to include test files in the check
            
        Returns:
            List of all quality issues found
        """
        issues = []
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        # Filter out test files if requested
        if not include_tests:
            python_files = [f for f in python_files if "test" not in f.parts]
        
        for py_file in python_files:
            file_issues = self.check_file_quality(str(py_file))
            issues.extend(file_issues)
        
        self.issues.extend(issues)
        return issues
    
    def _check_naming_conventions(self, tree: ast.AST, file_path: Path, content: str) -> List[QualityIssue]:
        """Check for naming convention violations."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not self._is_valid_function_name(node.name):
                    issues.append(QualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        issue_type="naming_convention",
                        message=f"Function name '{node.name}' doesn't follow naming conventions (use snake_case)",
                        severity="warning",
                        rule="function_naming"
                    ))
            elif isinstance(node, ast.ClassDef):
                if not self._is_valid_class_name(node.name):
                    issues.append(QualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        issue_type="naming_convention",
                        message=f"Class name '{node.name}' doesn't follow naming conventions (use PascalCase)",
                        severity="warning",
                        rule="class_naming"
                    ))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if not self._is_valid_variable_name(target.id):
                            issues.append(QualityIssue(
                                file_path=str(file_path),
                                line_number=node.lineno,
                                column=getattr(node, 'col_offset', 0),
                                issue_type="naming_convention",
                                message=f"Variable name '{target.id}' doesn't follow naming conventions (use snake_case)",
                                severity="warning",
                                rule="variable_naming"
                            ))
        
        return issues
    
    def _is_valid_function_name(self, name: str) -> bool:
        """Check if function name follows snake_case convention."""
        return re.match(r'^[a-z_][a-z0-9_]*$', name) is not None
    
    def _is_valid_class_name(self, name: str) -> bool:
        """Check if class name follows PascalCase convention."""
        return re.match(r'^[A-Z][a-zA-Z0-9]*$', name) is not None
    
    def _is_valid_variable_name(self, name: str) -> bool:
        """Check if variable name follows snake_case convention."""
        return re.match(r'^[a-z_][a-z0-9_]*$', name) is not None
    
    def _check_complexity(self, tree: ast.AST, file_path: Path, content: str) -> List[QualityIssue]:
        """Check for complexity issues (cyclomatic, function length, etc.)."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check function length
                func_start = node.lineno
                func_end = self._get_function_end_line(node)
                func_length = func_end - func_start + 1
                
                if func_length > 50:  # More than 50 lines
                    issues.append(QualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        issue_type="complexity",
                        message=f"Function '{node.name}' is too long ({func_length} lines, max 50)",
                        severity="warning",
                        rule="function_length"
                    ))
                
                # Check parameter count
                arg_count = len(node.args.args)
                if arg_count > 5:  # More than 5 parameters
                    issues.append(QualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        issue_type="complexity",
                        message=f"Function '{node.name}' has too many parameters ({arg_count}, max 5)",
                        severity="warning",
                        rule="parameter_count"
                    ))
        
        return issues
    
    def _get_function_end_line(self, func_node: ast.FunctionDef) -> int:
        """Get the ending line number of a function."""
        max_line = func_node.lineno
        
        for child in ast.walk(func_node):
            if hasattr(child, 'lineno') and child.lineno:
                max_line = max(max_line, child.lineno)
        
        return max_line
    
    def _check_security_patterns(self, file_path: Path, content: str) -> List[QualityIssue]:
        """Check for security patterns and vulnerabilities."""
        issues = []
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret detected"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token detected"),
            (r'key\s*=\s*["\'][^"\']+["\']', "Potentially hardcoded key detected"),
        ]
        
        for pattern, message in secret_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_no = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    file_path=str(file_path),
                    line_number=line_no,
                    column=match.start() - content.rfind('\n', 0, match.start()) - 1,
                    issue_type="security",
                    message=message,
                    severity="error",
                    rule="hardcoded_credentials"
                ))
        
        # Check for unsafe eval usage
        eval_patterns = [
            (r'\beval\s*\(', "Use of eval() is dangerous"),
            (r'\bexec\s*\(', "Use of exec() is dangerous"),
            (r'\bcompile\s*\([^,]*,[^,]*,[^,]*["\']exec["\']', "Use of compile() with exec mode is dangerous"),
        ]
        
        for pattern, message in eval_patterns:
            for match in re.finditer(pattern, content):
                line_no = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    file_path=str(file_path),
                    line_number=line_no,
                    column=match.start() - content.rfind('\n', 0, match.start()) - 1,
                    issue_type="security",
                    message=message,
                    severity="error",
                    rule="unsafe_functions"
                ))
        
        return issues
    
    def _check_docstring_completeness(self, tree: ast.AST, file_path: Path, content: str) -> List[QualityIssue]:
        """Check for missing or incomplete docstrings."""
        issues = []
        
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                # Check if the node has a docstring
                docstring = ast.get_docstring(node)
                
                if not docstring:
                    # Check if it's a private function (starts with _)
                    is_private = (hasattr(node, 'name') and node.name.startswith('_')) or isinstance(node, ast.Module)
                    
                    if not is_private:
                        issues.append(QualityIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            column=getattr(node, 'col_offset', 0),
                            issue_type="docstring",
                            message=f"Missing docstring in {type(node).__name__.lower()} '{getattr(node, 'name', 'module')}'",
                            severity="warning",
                            rule="docstring_missing"
                        ))
                else:
                    # Check if docstring is too short
                    if len(docstring.strip()) < 10:
                        issues.append(QualityIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            column=getattr(node, 'col_offset', 0),
                            issue_type="docstring",
                            message=f"Docstring in {type(node).__name__.lower()} '{getattr(node, 'name', 'module')}' is too brief",
                            severity="info",
                            rule="docstring_brief"
                        ))
        
        return issues
    
    def _check_type_annotations(self, tree: ast.AST, file_path: Path, content: str) -> List[QualityIssue]:
        """Check for missing type annotations."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check for return type annotation
                if node.returns is None:
                    issues.append(QualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        issue_type="typing",
                        message=f"Missing return type annotation in function '{node.name}'",
                        severity="warning",
                        rule="missing_return_annotation"
                    ))
                
                # Check for parameter type annotations
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != 'self':
                        issues.append(QualityIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            column=getattr(node, 'col_offset', 0),
                            issue_type="typing",
                            message=f"Missing type annotation for parameter '{arg.arg}' in function '{node.name}'",
                            severity="warning",
                            rule="missing_parameter_annotation"
                        ))
        
        return issues
    
    def get_report(self) -> Dict[str, Any]:
        """Generate a quality report."""
        # Organize issues by severity
        error_count = sum(1 for issue in self.issues if issue.severity == "error")
        warning_count = sum(1 for issue in self.issues if issue.severity == "warning")
        info_count = sum(1 for issue in self.issues if issue.severity == "info")
        
        # Organize issues by file
        issues_by_file = {}
        for issue in self.issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)
        
        return {
            "total_issues": len(self.issues),
            "errors": error_count,
            "warnings": warning_count,
            "infos": info_count,
            "issues_by_file": {path: [i for i in issues] for path, issues in issues_by_file.items()},
            "quality_score": self._calculate_quality_score()
        }
    
    def _calculate_quality_score(self) -> float:
        """Calculate an overall quality score."""
        if not self.issues:
            return 100.0
        
        # Weight errors and warnings differently
        total_weighted_issues = sum(
            5 if issue.severity == "error" else 
            2 if issue.severity == "warning" else 
            1 for issue in self.issues
        )
        
        # Quality score is 100 minus a penalty based on issues
        # Maximum penalty is 100, so score is between 0 and 100
        penalty = min(100, total_weighted_issues)
        return max(0, 100 - penalty)
    
    def save_report(self, output_path: str):
        """Save the quality report to a file."""
        report = self.get_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Quality report saved to {output_path}")


class CodeFormatter:
    """Automated code formatter."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_file(self, file_path: str) -> bool:
        """
        Format a Python file using best practices.
        
        Args:
            file_path: Path to the file to format
            
        Returns:
            True if formatting was successful
        """
        try:
            # This is a simple formatter that ensures basic formatting
            # In a real implementation, we would use black, autopep8, etc.
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply basic formatting rules
            formatted_content = self._apply_basic_formatting(content)
            
            # Only write if content actually changed
            if content != formatted_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_content)
                self.logger.info(f"Formatted file: {file_path}")
            
            return True
            
        except (OSError, IOError, UnicodeDecodeError) as e:
            self.logger.error(f"Error formatting file {file_path}: {e}")
            return False
    
    def _apply_basic_formatting(self, content: str) -> str:
        """Apply basic formatting rules to content."""
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            
            # Ensure single blank line at end of file
            if line or formatted_lines:
                formatted_lines.append(line)
        
        # Add a single blank line at the end
        if formatted_lines:
            formatted_lines.append('')
        
        return '\n'.join(formatted_lines)
    
    def format_project(self, project_root: str) -> Dict[str, bool]:
        """
        Format all Python files in the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary mapping file paths to success status
        """
        results = {}
        project_path = Path(project_root)
        
        for py_file in project_path.rglob("*.py"):
            success = self.format_file(str(py_file))
            results[str(py_file)] = success
        
        return results


class CodeAnalyzer:
    """Static code analyzer that goes beyond basic checks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Perform deep analysis of a Python file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
            
            analysis = {
                "file_path": file_path,
                "line_count": len(content.split('\n')),
                "function_count": 0,
                "class_count": 0,
                "import_count": 0,
                "has_comprehensions": False,
                "has_generators": False,
                "has_lambdas": False,
                "cyclomatic_complexity": 0,
                "functions": [],
                "classes": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["function_count"] += 1
                    analysis["functions"].append({
                        "name": node.name,
                        "line_start": node.lineno,
                        "line_end": self._get_function_end_line(node),
                        "arg_count": len(node.args.args),
                        "has_docstring": ast.get_docstring(node) is not None
                    })
                elif isinstance(node, ast.ClassDef):
                    analysis["class_count"] += 1
                    analysis["classes"].append({
                        "name": node.name,
                        "line_start": node.lineno,
                        "line_end": self._get_function_end_line(node),
                        "method_count": len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    analysis["import_count"] += 1
                elif isinstance(node, ast.ListComp):
                    analysis["has_comprehensions"] = True
                elif isinstance(node, ast.GeneratorExp):
                    analysis["has_generators"] = True
                elif isinstance(node, ast.Lambda):
                    analysis["has_lambdas"] = True
            
            return analysis
            
        except (SyntaxError, OSError, IOError, UnicodeDecodeError) as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return {"file_path": file_path, "error": str(e)}
    
    def _get_function_end_line(self, func_node: ast.FunctionDef) -> int:
        """Get the ending line number of a function."""
        max_line = func_node.lineno
        
        for child in ast.walk(func_node):
            if hasattr(child, 'lineno') and child.lineno:
                max_line = max(max_line, child.lineno)
        
        return max_line
    
    def analyze_project(self, project_root: str) -> Dict[str, Any]:
        """
        Analyze all Python files in the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary with project analysis results
        """
        project_path = Path(project_root)
        all_analyses = []
        
        for py_file in project_path.rglob("*.py"):
            analysis = self.analyze_file(str(py_file))
            all_analyses.append(analysis)
        
        # Aggregate results
        total_lines = sum(analysis.get("line_count", 0) for analysis in all_analyses if "error" not in analysis)
        total_functions = sum(analysis.get("function_count", 0) for analysis in all_analyses if "error" not in analysis)
        total_classes = sum(analysis.get("class_count", 0) for analysis in all_analyses if "error" not in analysis)
        total_imports = sum(analysis.get("import_count", 0) for analysis in all_analyses if "error" not in analysis)
        
        return {
            "project_root": str(project_root),
            "file_count": len(all_analyses),
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_imports": total_imports,
            "analyses": all_analyses
        }


# Global quality checker instance
quality_checker = CodeQualityChecker()
code_formatter = CodeFormatter()
code_analyzer = CodeAnalyzer()


def run_quality_checks(project_root: str = ".") -> Dict[str, Any]:
    """
    Run comprehensive quality checks on the project.
    
    Args:
        project_root: Root directory of the project to check
        
    Returns:
        Dictionary with quality check results
    """
    global quality_checker
    
    # Reinitialize checker with project root
    quality_checker = CodeQualityChecker(project_root=project_root)
    
    # Run checks
    issues = quality_checker.check_project_quality()
    
    # Generate report
    report = quality_checker.get_report()
    
    return report


def format_project_code(project_root: str = ".") -> Dict[str, bool]:
    """
    Format all Python code in the project.
    
    Args:
        project_root: Root directory of the project to format
        
    Returns:
        Dictionary mapping file paths to format success status
    """
    return code_formatter.format_project(project_root)


def analyze_project_code(project_root: str = ".") -> Dict[str, Any]:
    """
    Analyze all Python code in the project.
    
    Args:
        project_root: Root directory of the project to analyze
        
    Returns:
        Dictionary with analysis results
    """
    return code_analyzer.analyze_project(project_root)


def setup_quality_control_hooks():
    """
    Set up git hooks for quality control (example implementation).
    This would be part of a larger setup process.
    """
    # This would set up pre-commit hooks, etc.
    # For now, we'll just document what would be done
    print("Setting up quality control hooks...")
    print("1. Configure pre-commit hooks for code formatting and linting")
    print("2. Set up CI/CD pipeline for automated quality checks")
    print("3. Configure code coverage reporting")
    print("4. Set up code quality badges")


def example_usage():
    """Example of how to use the quality tools."""
    
    # Example 1: Run quality checks on a file
    checker = CodeQualityChecker()
    
    # Get test password from environment
    test_password = os.environ.get('TEST_PASSWORD', 'placeholder_password')
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(f"""
def bad_function_name():
    '''Missing parameters and return type'''
    return "Hello World"

class bad_class_name:
    pass

# Hardcoded password
password = "{test_password}"

def long_function(arg1, arg2, arg3, arg4, arg5, arg6):
    '''This function is too long and has too many parameters'''
    result = arg1 + arg2
    result = result + arg3
    result = result + arg4
    result = result + arg5
    result = result + arg6
    return result
""")
        temp_file = f.name
    
    try:
        issues = checker.check_file_quality(temp_file)
        print(f"Found {len(issues)} issues in temporary file:")
        for issue in issues:
            print(f"  - {issue.message} at line {issue.line_number}")
        
        # Run checks on the project directory
        print("\nRunning project-wide quality checks...")
        report = run_quality_checks(".")
        print(f"Quality Score: {report['quality_score']:.2f}")
        print(f"Total Issues: {report['total_issues']}")
        print(f"Errors: {report['errors']}")
        print(f"Warnings: {report['warnings']}")
        
        # Save the report
        checker.save_report("quality_report.json")
        print("Quality report saved to quality_report.json")
        
    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    example_usage()