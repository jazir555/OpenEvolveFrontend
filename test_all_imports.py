#!/usr/bin/env python3
"""
Comprehensive Import Testing Script for RESE Framework

This script tests all Python files in the rese/ directory to verify:
1. Files can be imported without errors
2. Dependencies are satisfied
3. No syntax errors exist
4. Basic functionality can be tested

Author: RESE Validation Framework
Date: 2026-01-01
"""

import os
import sys
import ast
import importlib
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ImportResult:
    """Result of importing a single module"""
    file_path: str
    module_name: str
    success: bool
    error_type: str = ""
    error_message: str = ""
    error_details: str = ""
    syntax_error: bool = False
    missing_dependencies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "module_name": self.module_name,
            "success": self.success,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "syntax_error": self.syntax_error,
            "missing_dependencies": self.missing_dependencies,
            "warnings": self.warnings
        }


class ImportTester:
    """Comprehensive import testing framework"""

    def __init__(self, base_path: str = "/c/Users/mmeadow/Documents/OpenEvolve/Frontend/rese"):
        self.base_path = Path(base_path)
        self.results: List[ImportResult] = []
        self.total_files = 0
        self.successful = 0
        self.failed = 0
        self.syntax_errors = 0

    def find_all_python_files(self) -> List[Path]:
        """Find all Python files in the target directory"""
        python_files = list(self.base_path.rglob("*.py"))
        # Exclude __pycache__ and test directories if needed
        python_files = [f for f in python_files if "__pycache__" not in str(f)]
        return sorted(python_files)

    def check_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """Check if Python file has valid syntax"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Error reading file: {str(e)}"

    def get_module_name(self, file_path: Path) -> str:
        """Convert file path to Python module name"""
        relative_path = file_path.relative_to(self.base_path.parent)
        module_name = str(relative_path.with_suffix('')).replace(os.sep, '.')
        return module_name

    def extract_imports(self, file_path: Path) -> List[str]:
        """Extract import statements from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])
            return imports
        except:
            return []

    def check_dependencies(self, imports: List[str]) -> List[str]:
        """Check if imported modules are available"""
        missing = []
        for imp in imports:
            if imp.startswith('.'):
                continue  # Relative import
            try:
                importlib.import_module(imp)
            except ImportError:
                missing.append(imp)
        return missing

    def test_import(self, file_path: Path) -> ImportResult:
        """Test importing a single Python file"""
        module_name = self.get_module_name(file_path)

        result = ImportResult(
            file_path=str(file_path),
            module_name=module_name,
            success=False
        )

        # Step 1: Check syntax
        syntax_ok, syntax_error = self.check_syntax(file_path)
        if not syntax_ok:
            result.syntax_error = True
            result.error_type = "SyntaxError"
            result.error_message = syntax_error
            return result

        # Step 2: Extract imports and check dependencies
        imports = self.extract_imports(file_path)
        missing_deps = self.check_dependencies(imports)
        if missing_deps:
            result.missing_dependencies = missing_deps
            result.warnings.append(f"Missing dependencies: {', '.join(missing_deps)}")

        # Step 3: Try to import the module
        try:
            # Add parent directory to sys.path
            sys.path.insert(0, str(self.base_path.parent))

            # Clear module from cache if it exists
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Import the module
            importlib.import_module(module_name)

            result.success = True

        except SyntaxError as e:
            result.syntax_error = True
            result.error_type = "SyntaxError"
            result.error_message = f"Line {e.lineno}: {e.msg}"
            result.error_details = traceback.format_exc()

        except ImportError as e:
            result.error_type = "ImportError"
            result.error_message = str(e)
            result.error_details = traceback.format_exc()

        except ModuleNotFoundError as e:
            result.error_type = "ModuleNotFoundError"
            result.error_message = str(e)
            result.error_details = traceback.format_exc()

        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)
            result.error_details = traceback.format_exc()

        finally:
            # Clean up sys.path
            if str(self.base_path.parent) in sys.path:
                sys.path.remove(str(self.base_path.parent))

        return result

    def run_tests(self) -> None:
        """Run import tests on all Python files"""
        python_files = self.find_all_python_files()
        self.total_files = len(python_files)

        print(f"\n{'='*80}")
        print(f"RESE Framework Import Testing")
        print(f"{'='*80}")
        print(f"Found {self.total_files} Python files to test")
        print(f"{'='*80}\n")

        for i, file_path in enumerate(python_files, 1):
            print(f"[{i}/{self.total_files}] Testing: {file_path.relative_to(self.base_path)}", end=" ")

            result = self.test_import(file_path)
            self.results.append(result)

            if result.success:
                self.successful += 1
                print("[OK] PASS")
            else:
                self.failed += 1
                if result.syntax_error:
                    self.syntax_errors += 1
                print("[FAIL] FAIL")
                print(f"  Error Type: {result.error_type}")
                print(f"  Message: {result.error_message}")

        print(f"\n{'='*80}")
        print("Testing Complete")
        print(f"{'='*80}\n")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        failed_results = [r for r in self.results if not r.success]
        syntax_error_results = [r for r in failed_results if r.syntax_error]
        missing_dep_files = [r for r in self.results if r.missing_dependencies]

        # Group failures by error type
        failures_by_type = {}
        for result in failed_results:
            if result.error_type not in failures_by_type:
                failures_by_type[result.error_type] = []
            failures_by_type[result.error_type].append(result)

        report = {
            "summary": {
                "total_files": self.total_files,
                "successful": self.successful,
                "failed": self.failed,
                "syntax_errors": self.syntax_errors,
                "success_rate": f"{(self.successful / self.total_files * 100):.2f}%" if self.total_files > 0 else "0%"
            },
            "failures_by_type": {
                error_type: len(results)
                for error_type, results in failures_by_type.items()
            },
            "files_with_missing_dependencies": len(missing_dep_files),
            "failed_imports": [
                {
                    "file": r.file_path,
                    "module": r.module_name,
                    "error_type": r.error_type,
                    "error_message": r.error_message
                }
                for r in failed_results
            ],
            "syntax_errors": [
                {
                    "file": r.file_path,
                    "module": r.module_name,
                    "error_message": r.error_message
                }
                for r in syntax_error_results
            ],
            "missing_dependencies_summary": self._summarize_missing_dependencies(missing_dep_files)
        }

        return report

    def _summarize_missing_dependencies(self, results: List[ImportResult]) -> Dict[str, List[str]]:
        """Summarize missing dependencies across all files"""
        dep_map = {}
        for result in results:
            for dep in result.missing_dependencies:
                if dep not in dep_map:
                    dep_map[dep] = []
                dep_map[dep].append(result.file_path)
        return dep_map

    def print_report(self) -> None:
        """Print detailed test report"""
        report = self.generate_report()

        print(f"\n{'='*80}")
        print(f"IMPORT TEST REPORT")
        print(f"{'='*80}\n")

        # Summary
        print("SUMMARY:")
        print(f"  Total Files Tested: {report['summary']['total_files']}")
        print(f"  Successful: {report['summary']['successful']}")
        print(f"  Failed: {report['summary']['failed']}")
        print(f"  Syntax Errors: {report['summary']['syntax_errors']}")
        print(f"  Success Rate: {report['summary']['success_rate']}")

        # Failures by type
        if report['failures_by_type']:
            print(f"\nFAILURES BY ERROR TYPE:")
            for error_type, count in sorted(report['failures_by_type'].items()):
                print(f"  {error_type}: {count}")

        # Missing dependencies
        if report['missing_dependencies_summary']:
            print(f"\nMISSING DEPENDENCIES:")
            for dep, files in sorted(report['missing_dependencies_summary'].items()):
                print(f"  {dep}: {len(files)} file(s)")

        # Syntax errors
        if report['syntax_errors']:
            print(f"\nSYNTAX ERRORS ({len(report['syntax_errors'])}):")
            for error in report['syntax_errors']:
                print(f"  File: {error['file']}")
                print(f"  Error: {error['error_message']}\n")

        # Failed imports
        if report['failed_imports']:
            print(f"\nFAILED IMPORTS ({len(report['failed_imports'])}):")
            for imp in report['failed_imports'][:10]:  # Show first 10
                print(f"  File: {imp['file']}")
                print(f"  Module: {imp['module']}")
                print(f"  Error: {imp['error_type']} - {imp['error_message']}\n")

            if len(report['failed_imports']) > 10:
                print(f"  ... and {len(report['failed_imports']) - 10} more failures")

        print(f"\n{'='*80}\n")

    def save_report(self, output_file: str = None) -> str:
        """Save report to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"import_test_report_{timestamp}.json"

        report = self.generate_report()
        report['timestamp'] = datetime.now().isoformat()
        report['all_results'] = [r.to_dict() for r in self.results]

        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        return str(output_path)


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test all Python imports in RESE framework")
    parser.add_argument(
        "--path",
        default="/c/Users/mmeadow/Documents/OpenEvolve/Frontend/rese",
        help="Path to RESE directory"
    )
    parser.add_argument(
        "--output",
        help="Output JSON report file path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    tester = ImportTester(base_path=args.path)

    try:
        tester.run_tests()
        tester.print_report()

        output_file = tester.save_report(args.output)
        print(f"Detailed report saved to: {output_file}")

        # Return exit code based on success
        if tester.failed > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
