#!/usr/bin/env python3
"""
Script to test the validity of integration imports across the codebase.
Checks for:
1. Correct module paths
2. Helper method naming patterns
3. Circular import issues
4. Incomplete integrations
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

class IntegrationChecker:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.issues = []
        self.integration_files = []
        self.module_files = {}

        # Expected import patterns
        self.expected_imports = {
            'alerting_system': r'from\s+alerting_system\s+import',
            'knowledge_engine': r'from\s+knowledge_engine\.enterprise_knowledge_engine\s+import',
            'strategy_selector': r'from\s+adaptive_strategy_selector\s+import'
        }

        # Helper method patterns
        self.helper_patterns = {
            'alerts': r'_trigger_\w+_alerts',
            'knowledge': r'_extract_\w+_knowledge',
            'performance': r'_track_\w+_performance'
        }

    def find_integration_files(self) -> List[Path]:
        """Find all integration files."""
        files = []
        for file_path in self.root_dir.glob("*_integration.py"):
            if file_path.is_file():
                files.append(file_path)
        return sorted(files)

    def find_module_files(self) -> Dict[str, Path]:
        """Find the core integration module files."""
        modules = {}

        # Check for alerting_system.py
        alerting = self.root_dir / "alerting_system.py"
        if alerting.exists():
            modules['alerting_system'] = alerting

        # Check for adaptive_strategy_selector.py
        strategy = self.root_dir / "adaptive_strategy_selector.py"
        if strategy.exists():
            modules['adaptive_strategy_selector'] = strategy

        # Check for knowledge_engine/enterprise_knowledge_engine.py
        knowledge = self.root_dir / "knowledge_engine" / "enterprise_knowledge_engine.py"
        if knowledge.exists():
            modules['knowledge_engine.enterprise_knowledge_engine'] = knowledge

        return modules

    def extract_imports(self, file_path: Path) -> List[Tuple[str, int, str]]:
        """Extract all import statements from a file."""
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line.startswith('from ') and ' import ' in line:
                    imports.append(('from', line_num, line))
                elif line.startswith('import '):
                    imports.append(('import', line_num, line))
        except Exception as e:
            self.issues.append({
                'file': str(file_path),
                'type': 'ERROR',
                'message': f"Failed to read file: {e}"
            })

        return imports

    def check_import_path(self, file_path: Path, import_type: int, line_num: int, statement: str) -> Optional[Dict]:
        """Check if an import statement uses the correct module path."""
        # Check for alerting_system
        if 'alerting_system' in statement:
            if 'from alerting_system import' not in statement:
                return {
                    'file': str(file_path),
                    'line': line_num,
                    'type': 'IMPORT_PATH_ERROR',
                    'message': f"Incorrect alerting_system import: {statement}",
                    'expected': "from alerting_system import ..."
                }

        # Check for knowledge_engine.enterprise_knowledge_engine
        if 'knowledge_engine' in statement and 'enterprise_knowledge_engine' in statement:
            if 'from knowledge_engine.enterprise_knowledge_engine import' not in statement:
                return {
                    'file': str(file_path),
                    'line': line_num,
                    'type': 'IMPORT_PATH_ERROR',
                    'message': f"Incorrect knowledge_engine import: {statement}",
                    'expected': "from knowledge_engine.enterprise_knowledge_engine import ..."
                }

        # Check for adaptive_strategy_selector
        if 'adaptive_strategy_selector' in statement and 'import' in statement:
            if 'from adaptive_strategy_selector import' not in statement:
                return {
                    'file': str(file_path),
                    'line': line_num,
                    'type': 'IMPORT_PATH_ERROR',
                    'message': f"Incorrect adaptive_strategy_selector import: {statement}",
                    'expected': "from adaptive_strategy_selector import ..."
                }

        return None

    def extract_helper_methods(self, file_path: Path) -> Dict[str, List[Tuple[str, int]]]:
        """Extract helper method definitions from a file."""
        helpers = {
            'alerts': [],
            'knowledge': [],
            'performance': []
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the AST
            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    method_name = node.name

                    # Check for helper method patterns
                    if re.match(self.helper_patterns['alerts'], method_name):
                        helpers['alerts'].append((method_name, node.lineno))
                    elif re.match(self.helper_patterns['knowledge'], method_name):
                        helpers['knowledge'].append((method_name, node.lineno))
                    elif re.match(self.helper_patterns['performance'], method_name):
                        helpers['performance'].append((method_name, node.lineno))

        except SyntaxError as e:
            self.issues.append({
                'file': str(file_path),
                'type': 'SYNTAX_ERROR',
                'message': f"Syntax error at line {e.lineno}: {e.msg}"
            })
        except Exception as e:
            self.issues.append({
                'file': str(file_path),
                'type': 'ERROR',
                'message': f"Failed to parse file: {e}"
            })

        return helpers

    def check_helper_method_calls(self, file_path: Path, helpers: Dict[str, List[Tuple[str, int]]]) -> List[Dict]:
        """Check if helper methods are actually called."""
        issues = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            for category, methods in helpers.items():
                for method_name, line_num in methods:
                    # Check if method is called anywhere in the file
                    pattern = rf'{method_name}\s*\('
                    if not re.search(pattern, content):
                        issues.append({
                            'file': str(file_path),
                            'line': line_num,
                            'type': 'UNUSED_HELPER',
                            'message': f"Helper method '{method_name}' defined but never called",
                            'category': category
                        })

        except Exception as e:
            issues.append({
                'file': str(file_path),
                'type': 'ERROR',
                'message': f"Failed to check method calls: {e}"
            })

        return issues

    def check_integration_completeness(self, file_path: Path, imports: List[Tuple], helpers: Dict[str, List]) -> List[Dict]:
        """Check if integration has both imports and helper methods."""
        issues = []

        has_alerting_import = any('alerting_system' in imp[2] for imp in imports)
        has_knowledge_import = any('knowledge_engine' in imp[2] and 'enterprise_knowledge_engine' in imp[2] for imp in imports)
        has_strategy_import = any('adaptive_strategy_selector' in imp[2] for imp in imports)

        has_alert_helpers = len(helpers['alerts']) > 0
        has_knowledge_helpers = len(helpers['knowledge']) > 0
        has_performance_helpers = len(helpers['performance']) > 0

        # Check for incomplete integrations
        if has_alerting_import and not has_alert_helpers:
            issues.append({
                'file': str(file_path),
                'type': 'INCOMPLETE_INTEGRATION',
                'message': "Imports alerting_system but has no _trigger_*_alerts helper methods"
            })

        if has_knowledge_import and not has_knowledge_helpers:
            issues.append({
                'file': str(file_path),
                'type': 'INCOMPLETE_INTEGRATION',
                'message': "Imports knowledge_engine.enterprise_knowledge_engine but has no _extract_*_knowledge helper methods"
            })

        if has_strategy_import and not has_performance_helpers:
            issues.append({
                'file': str(file_path),
                'type': 'INCOMPLETE_INTEGRATION',
                'message': "Imports adaptive_strategy_selector but has no _track_*_performance helper methods"
            })

        return issues

    def check_circular_imports(self) -> List[Dict]:
        """Check for potential circular import issues."""
        issues = []
        import_map = {}

        # Build import map
        for file_path in self.integration_files:
            imports = self.extract_imports(file_path)
            imported_modules = []

            for imp_type, line_num, statement in imports:
                if 'from alerting_system import' in statement:
                    imported_modules.append('alerting_system')
                elif 'from knowledge_engine.enterprise_knowledge_engine import' in statement:
                    imported_modules.append('knowledge_engine.enterprise_knowledge_engine')
                elif 'from adaptive_strategy_selector import' in statement:
                    imported_modules.append('adaptive_strategy_selector')

            import_map[str(file_path.name)] = imported_modules

        # Check if modules import each other
        module_deps = {}
        for file_name, imports in import_map.items():
            module_name = file_name.replace('_integration.py', '')
            module_deps[module_name] = imports

        # Check for circular dependencies
        for module, deps in module_deps.items():
            for dep in deps:
                dep_name = dep.replace('knowledge_engine.enterprise_knowledge_engine', 'enterprise_knowledge_engine')
                if dep_name in module_deps and module in module_deps[dep_name]:
                    issues.append({
                        'type': 'CIRCULAR_IMPORT',
                        'message': f"Potential circular import between {module} and {dep_name}",
                        'modules': [module, dep_name]
                    })

        return issues

    def run_checks(self) -> Dict:
        """Run all checks and return results."""
        print(">> Finding integration files...")
        self.integration_files = self.find_integration_files()
        print(f"   Found {len(self.integration_files)} integration files")

        print("\n>> Finding core module files...")
        self.module_files = self.find_module_files()
        print(f"   Found {len(self.module_files)} module files")

        if not self.module_files.get('alerting_system'):
            self.issues.append({
                'type': 'CRITICAL',
                'message': "alerting_system.py not found in root directory"
            })

        if not self.module_files.get('knowledge_engine.enterprise_knowledge_engine'):
            self.issues.append({
                'type': 'CRITICAL',
                'message': "knowledge_engine/enterprise_knowledge_engine.py not found"
            })

        if not self.module_files.get('adaptive_strategy_selector'):
            self.issues.append({
                'type': 'CRITICAL',
                'message': "adaptive_strategy_selector.py not found in root directory"
            })

        print("\n>> Checking integration files...")
        results = {
            'total_files': len(self.integration_files),
            'files_with_issues': 0,
            'import_errors': [],
            'helper_method_issues': [],
            'incomplete_integrations': [],
            'unused_helpers': [],
            'circular_imports': [],
            'valid_integrations': []
        }

        for file_path in self.integration_files:
            print(f"   Checking {file_path.name}...")

            file_issues = []

            # Extract imports and helpers
            imports = self.extract_imports(file_path)
            helpers = self.extract_helper_methods(file_path)

            # Check import paths
            for imp_type, line_num, statement in imports:
                error = self.check_import_path(file_path, imp_type, line_num, statement)
                if error:
                    file_issues.append(error)
                    results['import_errors'].append(error)

            # Check for unused helpers
            unused = self.check_helper_method_calls(file_path, helpers)
            if unused:
                file_issues.extend(unused)
                results['unused_helpers'].extend(unused)

            # Check completeness
            incomplete = self.check_integration_completeness(file_path, imports, helpers)
            if incomplete:
                file_issues.extend(incomplete)
                results['incomplete_integrations'].extend(incomplete)

            if file_issues:
                results['files_with_issues'] += 1
            else:
                results['valid_integrations'].append(str(file_path))

        # Check for circular imports
        print("\n>> Checking for circular imports...")
        circular = self.check_circular_imports()
        if circular:
            results['circular_imports'] = circular

        return results

    def print_report(self, results: Dict):
        """Print a formatted report."""
        print("\n" + "="*80)
        print("INTEGRATION VALIDITY REPORT")
        print("="*80)

        print(f"\n[SUMMARY]")
        print(f"   Total integration files: {results['total_files']}")
        print(f"   Files with issues: {results['files_with_issues']}")
        print(f"   Valid integrations: {len(results['valid_integrations'])}")

        # Critical issues
        critical_issues = [i for i in self.issues if i.get('type') == 'CRITICAL']
        if critical_issues:
            print(f"\n[CRITICAL ISSUES] ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"   [!] {issue['message']}")

        # Import path errors
        if results['import_errors']:
            print(f"\n[IMPORT PATH ERRORS] ({len(results['import_errors'])}):")
            for error in results['import_errors']:
                print(f"   [X] {Path(error['file']).name}:{error['line']}")
                print(f"      {error['message']}")
                print(f"      Expected: {error['expected']}")

        # Incomplete integrations
        if results['incomplete_integrations']:
            print(f"\n[INCOMPLETE INTEGRATIONS] ({len(results['incomplete_integrations'])}):")
            for issue in results['incomplete_integrations']:
                print(f"   [!] {Path(issue['file']).name}")
                print(f"      {issue['message']}")

        # Unused helper methods
        if results['unused_helpers']:
            print(f"\n[UNUSED HELPER METHODS] ({len(results['unused_helpers'])}):")
            for issue in results['unused_helpers']:
                print(f"   [?] {Path(issue['file']).name}:{issue['line']}")
                print(f"      {issue['message']}")

        # Circular imports
        if results['circular_imports']:
            print(f"\n[CIRCULAR IMPORT WARNINGS] ({len(results['circular_imports'])}):")
            for issue in results['circular_imports']:
                print(f"   [CIRC] {issue['message']}")
                print(f"      Modules: {', '.join(issue['modules'])}")

        # Valid integrations
        if results['valid_integrations']:
            print(f"\n[VALID INTEGRATIONS] ({len(results['valid_integrations'])}):")
            for file_path in results['valid_integrations'][:10]:  # Show first 10
                print(f"   [OK] {Path(file_path).name}")
            if len(results['valid_integrations']) > 10:
                print(f"      ... and {len(results['valid_integrations']) - 10} more")

        print("\n" + "="*80)

        # Exit code
        total_errors = (
            len(critical_issues) +
            len(results['import_errors']) +
            len(results['incomplete_integrations'])
        )
        sys.exit(1 if total_errors > 0 else 0)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Check integration validity')
    parser.add_argument('--dir', default='.', help='Root directory to scan')
    args = parser.parse_args()

    checker = IntegrationChecker(args.dir)
    results = checker.run_checks()
    checker.print_report(results)


if __name__ == '__main__':
    main()
