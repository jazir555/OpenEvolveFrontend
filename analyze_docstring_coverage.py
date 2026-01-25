#!/usr/bin/env python3
"""
Docstring Coverage Analyzer

Analyzes Python files for missing docstrings and generates a comprehensive report.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class DocstringAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze docstring coverage."""

    def __init__(self, filename: str):
        """Initialize analyzer with filename."""
        self.filename = filename
        self.missing_module_docstring = False
        self.missing_class_docstrings = []
        self.missing_function_docstrings = []
        self.classes_found = 0
        self.functions_found = 0

    def visit_Module(self, node: ast.Module):
        """Check module docstring."""
        self.classes_found = 0
        self.functions_found = 0

        docstring = ast.get_docstring(node)
        if not docstring:
            self.missing_module_docstring = True

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Check class docstring."""
        self.classes_found += 1
        docstring = ast.get_docstring(node)
        if not docstring:
            self.missing_class_docstrings.append(node.name)

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Check function/method docstring."""
        # Skip private methods (starting with _)
        if not node.name.startswith('_'):
            self.functions_found += 1
            docstring = ast.get_docstring(node)
            if not docstring:
                self.missing_function_docstrings.append(node.name)

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Check async function/method docstring."""
        # Skip private methods
        if not node.name.startswith('_'):
            self.functions_found += 1
            docstring = ast.get_docstring(node)
            if not docstring:
                self.missing_function_docstrings.append(f"{node.name} (async)")

        self.generic_visit(node)


def analyze_file(filepath: Path) -> Dict:
    """Analyze a single Python file for docstring coverage."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source, filename=str(filepath))
        analyzer = DocstringAnalyzer(str(filepath))
        analyzer.visit(tree)

        return {
            'file': str(filepath),
            'missing_module': analyzer.missing_module_docstring,
            'missing_classes': analyzer.missing_class_docstrings,
            'missing_functions': analyzer.missing_function_docstrings,
            'total_classes': analyzer.classes_found,
            'total_functions': analyzer.functions_found,
            'has_issues': (analyzer.missing_module_docstring or
                          analyzer.missing_class_docstrings or
                          analyzer.missing_function_docstrings)
        }
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        return {
            'file': str(filepath),
            'error': str(e),
            'has_issues': True
        }


def analyze_directory(root_dir: Path) -> List[Dict]:
    """Analyze all Python files in directory."""
    results = []

    for py_file in root_dir.rglob('*.py'):
        # Skip test files and __pycache__
        if '__pycache__' in str(py_file):
            continue

        result = analyze_file(py_file)
        if result.get('has_issues'):
            results.append(result)

    return results


def generate_report(results: List[Dict]) -> str:
    """Generate comprehensive coverage report."""
    lines = []
    lines.append("=" * 80)
    lines.append("DOCSTRING COVERAGE REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Summary statistics
    total_files = len(results)
    total_missing_modules = sum(1 for r in results if r.get('missing_module'))
    total_missing_classes = sum(len(r.get('missing_classes', [])) for r in results)
    total_missing_functions = sum(len(r.get('missing_functions', [])) for r in results)

    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Files with issues: {total_files}")
    lines.append(f"Missing module docstrings: {total_missing_modules}")
    lines.append(f"Missing class docstrings: {total_missing_classes}")
    lines.append(f"Missing function docstrings: {total_missing_functions}")
    lines.append("")

    # Priority 1: Core Functions (evolution, adversarial, integrated_workflow)
    lines.append("PRIORITY 1: CRITICAL FILES (Core Functions)")
    lines.append("-" * 80)

    priority1_files = [
        'evolution.py',
        'adversarial.py',
        'integrated_workflow.py',
        'end_to_end_invention_planner.py',
        'maker_engine.py',
        'mdap_engine.py',
        'decomposition_engine.py',
        'problem_analyzer.py'
    ]

    for result in results:
        filename = Path(result['file']).name
        if filename in priority1_files:
            lines.append(f"\n{filename}:")
            if result.get('missing_module'):
                lines.append("  - MISSING module docstring")
            if result.get('missing_classes'):
                for cls in result['missing_classes']:
                    lines.append(f"  - MISSING class docstring: {cls}")
            if result.get('missing_functions'):
                for func in result['missing_functions'][:5]:  # Show first 5
                    lines.append(f"  - MISSING function docstring: {func}")
                if len(result['missing_functions']) > 5:
                    lines.append(f"  - ... and {len(result['missing_functions']) - 5} more")

    # Priority 2: Integration files
    lines.append("\n\nPRIORITY 2: INTEGRATION FILES")
    lines.append("-" * 80)

    integration_keywords = ['integration', 'bridge', 'adapter', 'client', 'mcp_tools']

    for result in results:
        filename = Path(result['file']).name.lower()
        if any(kw in filename for kw in integration_keywords):
            if not any(Path(result['file']).name == f for f in priority1_files):
                lines.append(f"\n{Path(result['file']).name}:")
                if result.get('missing_module'):
                    lines.append("  - MISSING module docstring")
                if result.get('missing_classes'):
                    for cls in result['missing_classes']:
                        lines.append(f"  - MISSING class docstring: {cls}")
                if result.get('missing_functions'):
                    lines.append(f"  - {len(result['missing_functions'])} missing function docstrings")

    # All other files
    lines.append("\n\nPRIORITY 3: OTHER FILES")
    lines.append("-" * 80)

    other_files = []
    for result in results:
        filename = Path(result['file']).name
        if (filename not in priority1_files and
            not any(kw in filename.lower() for kw in integration_keywords)):
            other_files.append(result)

    for result in other_files[:20]:  # Show first 20
        lines.append(f"\n{Path(result['file']).name}:")
        if result.get('missing_module'):
            lines.append("  - MISSING module docstring")
        if result.get('missing_classes'):
            lines.append(f"  - {len(result['missing_classes'])} missing class docstrings")
        if result.get('missing_functions'):
            lines.append(f"  - {len(result['missing_functions'])} missing function docstrings")

    if len(other_files) > 20:
        lines.append(f"\n... and {len(other_files) - 20} more files")

    return "\n".join(lines)


def main():
    """Main entry point."""
    root_dir = Path.cwd()

    print("Analyzing Python files for docstring coverage...")
    print(f"Working directory: {root_dir}")
    results = analyze_directory(root_dir)

    # Generate report
    report = generate_report(results)

    # Save report
    report_path = root_dir / "DOCSTRING_COVERAGE_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print(f"\nFound {len(results)} files with documentation issues")
    print(f"Total missing docstrings: "
          f"{sum(1 for r in results if r.get('missing_module'))} modules, "
          f"{sum(len(r.get('missing_classes', [])) for r in results)} classes, "
          f"{sum(len(r.get('missing_functions', [])) for r in results)} functions")

    # Also print report to console
    print("\n" + "=" * 80)
    print(report)


if __name__ == "__main__":
    main()
