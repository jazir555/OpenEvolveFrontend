"""
Comprehensive Gap Scanner

Scans for:
1. TODO/FIXME comments
2. Placeholder implementations
3. Missing error handling
4. Incomplete features
5. Hardcoded values
6. Missing documentation
7. Import issues
8. Integration gaps
9. Test coverage gaps
10. Configuration gaps
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple


class ComprehensiveGapScanner:
    """Comprehensive scanner for remaining gaps."""
    
    def __init__(self):
        self.gaps = []
        self.warnings = []
        self.base_path = Path(__file__).parent
    
    def scan_all(self):
        """Run all scans."""
        print("="*70)
        print("COMPREHENSIVE GAP SCANNER")
        print("="*70)
        
        self.scan_todo_comments()
        self.scan_placeholder_implementations()
        self.scan_missing_error_handling()
        self.scan_hardcoded_values()
        self.scan_missing_docstrings()
        self.scan_import_issues()
        self.scan_integration_gaps()
        self.scan_test_coverage()
        self.scan_configuration_gaps()
        self.scan_api_completeness()
        
        self.print_summary()
    
    def scan_todo_comments(self):
        """Scan for TODO/FIXME comments."""
        print("\n1. Scanning for TODO/FIXME Comments")
        
        todo_pattern = re.compile(r'#\s*(TODO|FIXME|XXX|HACK|BUG)\s*:?\s*(.+)', re.IGNORECASE)
        
        py_files = list(self.base_path.glob('*.py'))
        found_todos = []
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for match in todo_pattern.finditer(content):
                        line_num = content[:match.start()].count('\n') + 1
                        found_todos.append((file_path.name, line_num, match.group(1), match.group(2).strip()))
            except Exception as e:
                pass
        
        if found_todos:
            for file, line, tag, text in found_todos[:10]:  # Show first 10
                self.gaps.append(f"TODO in {file}:{line} - {tag}: {text[:50]}")
                print(f"   [GAP] {file}:{line} - {tag}: {text[:50]}")
        else:
            print("   [OK] No TODO/FIXME comments found")
    
    def scan_placeholder_implementations(self):
        """Scan for placeholder implementations."""
        print("\n2. Scanning for Placeholder Implementations")
        
        placeholders = [
            (r'pass\s*$', 'Empty pass statement'),
            (r'return\s+None\s*$', 'Returns None without implementation'),
            (r'raise\s+NotImplementedError', 'NotImplementedError'),
            (r'#\s*Not implemented', 'Not implemented comment'),
            (r'print\s*\(\s*["\']Not implemented', 'Print not implemented'),
        ]
        
        py_files = list(self.base_path.glob('*.py'))
        found_placeholders = []
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines, 1):
                        for pattern, desc in placeholders:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Check if it's in a function/method
                                found_placeholders.append((file_path.name, i, desc))
                                break
            except Exception as e:
                pass
        
        if found_placeholders:
            for file, line, desc in found_placeholders[:10]:
                self.warnings.append(f"Placeholder in {file}:{line} - {desc}")
                print(f"   [WARN] {file}:{line} - {desc}")
        else:
            print("   [OK] No obvious placeholders found")
    
    def scan_missing_error_handling(self):
        """Scan for functions without error handling."""
        print("\n3. Scanning for Missing Error Handling")
        
        py_files = list(self.base_path.glob('*.py'))
        functions_without_try = []
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check if function has try-except
                        has_try = any(isinstance(n, ast.Try) for n in ast.walk(node))
                        
                        # Skip small functions and private functions
                        if len(node.body) > 5 and not node.name.startswith('_'):
                            if not has_try:
                                functions_without_try.append((file_path.name, node.name, node.lineno))
            except Exception as e:
                pass
        
        if functions_without_try:
            for file, func, line in functions_without_try[:10]:
                self.warnings.append(f"No error handling in {file}:{line} - {func}()")
                print(f"   [WARN] {file}:{line} - {func}() lacks try-except")
        else:
            print("   [OK] All significant functions have error handling")
    
    def scan_hardcoded_values(self):
        """Scan for hardcoded values that should be configurable."""
        print("\n4. Scanning for Hardcoded Values")
        
        # Patterns for hardcoded values that might need configuration
        patterns = [
            (r'timeout\s*=\s*\d+', 'Hardcoded timeout'),
            (r'port\s*=\s*\d+', 'Hardcoded port'),
            (r'host\s*=\s*["\'][^"\']+["\']', 'Hardcoded host'),
            (r'max_retries\s*=\s*\d+', 'Hardcoded retry count'),
            (r'batch_size\s*=\s*\d+', 'Hardcoded batch size'),
        ]
        
        py_files = list(self.base_path.glob('*.py'))
        found_hardcoded = []
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines, 1):
                        for pattern, desc in patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Skip if it's using config or env
                                if 'config' not in line.lower() and 'env' not in line.lower() and 'getenv' not in line.lower():
                                    found_hardcoded.append((file_path.name, i, desc))
                                    break
            except Exception as e:
                pass
        
        if found_hardcoded:
            for file, line, desc in found_hardcoded[:10]:
                self.warnings.append(f"Hardcoded value in {file}:{line} - {desc}")
                print(f"   [WARN] {file}:{line} - {desc}")
        else:
            print("   [OK] No concerning hardcoded values found")
    
    def scan_missing_docstrings(self):
        """Scan for missing docstrings."""
        print("\n5. Scanning for Missing Docstrings")
        
        py_files = list(self.base_path.glob('*.py'))
        missing_docs = []
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        # Check if it has a docstring
                        has_docstring = (node.body and 
                                       isinstance(node.body[0], ast.Expr) and 
                                       isinstance(node.body[0].value, ast.Constant) and 
                                       isinstance(node.body[0].value.value, str))
                        
                        # Skip private and small functions
                        if not node.name.startswith('_') and len(node.body) > 3:
                            if not has_docstring:
                                missing_docs.append((file_path.name, type(node).__name__, node.name, node.lineno))
            except Exception as e:
                pass
        
        if missing_docs:
            for file, type_, name, line in missing_docs[:10]:
                self.warnings.append(f"Missing docstring in {file}:{line} - {type_} {name}")
                print(f"   [WARN] {file}:{line} - {type_} '{name}' missing docstring")
        else:
            print("   [OK] All public functions/classes have docstrings")
    
    def scan_import_issues(self):
        """Scan for potential import issues."""
        print("\n6. Scanning for Import Issues")
        
        # Check if all imports work
        critical_modules = [
            'z3_solver_connector',
            'leanaide_real_connector',
            'z3_knowledge_complete',
            'unified_math_bridge_complete',
            'math_knowledge_config',
            'math_mcp_tools',
            'math_api_complete',
            'math_knowledge_cli',
        ]
        
        import_errors = []
        for module in critical_modules:
            try:
                __import__(module)
            except Exception as e:
                import_errors.append((module, str(e)))
        
        if import_errors:
            for module, error in import_errors:
                self.gaps.append(f"Import error in {module}: {error}")
                print(f"   [GAP] {module}: {error}")
        else:
            print("   [OK] All critical modules import successfully")
    
    def scan_integration_gaps(self):
        """Scan for integration gaps between components."""
        print("\n7. Scanning for Integration Gaps")
        
        # Check if components can work together
        gaps_found = []
        
        # Check 1: Z3 to Knowledge flow
        try:
            from z3_solver_connector import get_z3_connector
            from z3_knowledge_complete import get_z3_knowledge_manager
            
            # Both should exist
            if not get_z3_connector or not get_z3_knowledge_manager:
                gaps_found.append("Z3-Knowledge integration gap")
        except Exception as e:
            gaps_found.append(f"Z3-Knowledge integration: {e}")
        
        # Check 2: Bridge to Solver flow
        try:
            from unified_math_bridge_complete import get_unified_bridge_complete
            from z3_solver_connector import get_z3_connector
            
            if not get_unified_bridge_complete or not get_z3_connector:
                gaps_found.append("Bridge-Solver integration gap")
        except Exception as e:
            gaps_found.append(f"Bridge-Solver integration: {e}")
        
        # Check 3: API to Components flow
        try:
            from math_api_complete import math_api
            if not math_api:
                gaps_found.append("API not properly initialized")
        except Exception as e:
            gaps_found.append(f"API initialization: {e}")
        
        if gaps_found:
            for gap in gaps_found:
                self.gaps.append(gap)
                print(f"   [GAP] {gap}")
        else:
            print("   [OK] All integrations appear functional")
    
    def scan_test_coverage(self):
        """Scan for test coverage gaps."""
        print("\n8. Scanning for Test Coverage Gaps")
        
        # List of core files that should have tests
        core_files = [
            'z3_solver_connector.py',
            'leanaide_real_connector.py',
            'z3_knowledge_complete.py',
            'unified_math_bridge_complete.py',
            'math_api_complete.py',
            'math_knowledge_cli.py',
        ]
        
        test_files = list(self.base_path.glob('test_*.py'))
        tested_modules = set()
        
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                    for core in core_files:
                        module_name = core.replace('.py', '')
                        if module_name in content:
                            tested_modules.add(core)
            except Exception as e:
                pass
        
        untested = set(core_files) - tested_modules
        
        if untested:
            for file in untested:
                self.warnings.append(f"No dedicated test file for {file}")
                print(f"   [WARN] No dedicated test file for {file}")
        else:
            print("   [OK] Core modules have test coverage")
    
    def scan_configuration_gaps(self):
        """Scan for configuration gaps."""
        print("\n9. Scanning for Configuration Gaps")
        
        gaps_found = []
        
        # Check if config has all necessary fields
        try:
            from math_knowledge_config import MathKnowledgeConfig
            
            config = MathKnowledgeConfig()
            
            required_sections = ['database', 'z3', 'leanaide', 'api', 'monitoring']
            for section in required_sections:
                if not hasattr(config, section):
                    gaps_found.append(f"Missing config section: {section}")
            
            # Check if config can be loaded from env
            if not hasattr(config, 'from_env'):
                gaps_found.append("Config missing from_env method")
                
        except Exception as e:
            gaps_found.append(f"Config check failed: {e}")
        
        if gaps_found:
            for gap in gaps_found:
                self.warnings.append(gap)
                print(f"   [WARN] {gap}")
        else:
            print("   [OK] Configuration complete")
    
    def scan_api_completeness(self):
        """Scan for API completeness."""
        print("\n10. Scanning for API Completeness")
        
        try:
            from math_api_complete import math_api
            
            if not math_api:
                self.gaps.append("API not created")
                print("   [GAP] API not created")
                return
            
            # Check for expected endpoints
            routes = [(r.path, list(r.methods) if hasattr(r, 'methods') else []) 
                      for r in math_api.routes if hasattr(r, 'path')]
            
            expected = [
                ('/health', ['GET', 'HEAD']),
                ('/solve/z3', ['POST']),
                ('/solve/lean', ['POST']),
                ('/solve/unified', ['POST']),
                ('/knowledge/learn', ['POST']),
                ('/knowledge/search', ['POST']),
            ]
            
            missing = []
            for path, methods in expected:
                found = any(path == r[0] for r in routes)
                if not found:
                    missing.append(path)
            
            if missing:
                for m in missing:
                    self.gaps.append(f"Missing API endpoint: {m}")
                    print(f"   [GAP] Missing endpoint: {m}")
            else:
                print("   [OK] All expected API endpoints present")
                
        except Exception as e:
            self.gaps.append(f"API completeness check failed: {e}")
            print(f"   [GAP] API check failed: {e}")
    
    def print_summary(self):
        """Print scan summary."""
        print("\n" + "="*70)
        print("GAP SCAN SUMMARY")
        print("="*70)
        print(f"\nCritical Gaps: {len(self.gaps)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.gaps:
            print("\nCritical Gaps Found:")
            for gap in self.gaps:
                print(f"   ❌ {gap}")
        
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"   [WARN] {warning}")
        
        print("\n" + "="*70)
        if not self.gaps and not self.warnings:
            print("SUCCESS: NO GAPS FOUND - SYSTEM COMPLETE")
        elif not self.gaps:
            print("WARNING: ONLY WARNINGS - NO CRITICAL GAPS")
        else:
            print(f"FAILED: {len(self.gaps)} CRITICAL GAPS NEED ATTENTION")
        print("="*70)


def main():
    scanner = ComprehensiveGapScanner()
    scanner.scan_all()
    return 0 if not scanner.gaps else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
