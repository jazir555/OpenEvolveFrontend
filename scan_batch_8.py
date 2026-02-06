#!/usr/bin/env python3
"""
Batch 8 Import Error Scanner
Scans Python files for syntax errors, import errors, and circular imports.
"""

import ast
import json
import os
import py_compile
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import tempfile
import traceback


class ImportErrorScanner:
    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
        self.total_files = 0
        self.files_with_errors = 0
        # Track imports for circular import detection
        self.file_imports: Dict[str, Set[str]] = {}
        self.project_root = Path("C:/Users/mmeadow/Documents/OpenEvolve/Frontend")
        
    def scan_files(self, file_list_path: str) -> Dict[str, Any]:
        """Scan all files from the provided list."""
        with open(file_list_path, 'r', encoding='utf-8', errors='ignore') as f:
            files = [line.strip() for line in f if line.strip()]
        
        self.total_files = len(files)
        print(f"Scanning {self.total_files} files...")
        
        # First pass: collect all imports for circular import detection
        for filepath in files:
            self._collect_imports(filepath)
        
        # Second pass: check each file
        for i, filepath in enumerate(files, 1):
            if i % 50 == 0:
                print(f"  Processed {i}/{self.total_files} files...")
            self._scan_file(filepath)
        
        # Check for circular imports
        self._detect_circular_imports()
        
        return {
            "total_files": self.total_files,
            "errors_found": len(self.errors),
            "errors": self.errors
        }
    
    def _collect_imports(self, filepath: str):
        """Collect imports from a file for circular import analysis."""
        try:
            if not os.path.exists(filepath):
                return
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return
            
            tree = ast.parse(content)
            imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    if node.level > 0:
                        # Relative import - resolve relative to file location
                        rel_path = Path(filepath).parent
                        for _ in range(node.level - 1):
                            rel_path = rel_path.parent
                        if module:
                            module_path = str(rel_path / module.replace('.', '/'))
                        else:
                            module_path = str(rel_path)
                        imports.add(module_path)
                    else:
                        imports.add(module)
            
            self.file_imports[filepath] = imports
            
        except Exception:
            pass  # Will be caught in main scan
    
    def _detect_circular_imports(self):
        """Detect circular import chains."""
        for filepath, imports in self.file_imports.items():
            for imp in imports:
                # Check if the import target imports back to this file
                for other_file, other_imports in self.file_imports.items():
                    if other_file == filepath:
                        continue
                    
                    # Check if other_file imports from filepath and filepath imports from other_file
                    filepath_normalized = filepath.replace('\\', '/').replace('.py', '').replace('/', '.')
                    other_file_normalized = other_file.replace('\\', '/').replace('.py', '').replace('/', '.')
                    
                    # Simple circular check: if both import each other
                    if (imp.replace('/', '.') in other_file_normalized or 
                        other_file_normalized.endswith(imp.replace('/', '.'))):
                        if any(filepath_normalized.endswith(oi.replace('/', '.')) for oi in other_imports):
                            # Avoid duplicate reports
                            error_key = f"{min(filepath, other_file)}|{max(filepath, other_file)}"
                            already_reported = any(
                                e.get('error_key') == error_key for e in self.errors 
                                if e['error_type'] == 'circular_import'
                            )
                            if not already_reported:
                                self.errors.append({
                                    'file': filepath,
                                    'error_type': 'circular_import',
                                    'line_number': 0,
                                    'message': f'Potential circular import with {other_file}',
                                    'suggested_fix': 'Consider restructuring imports or using TYPE_CHECKING',
                                    'error_key': error_key
                                })
    
    def _scan_file(self, filepath: str):
        """Scan a single file for errors."""
        if not os.path.exists(filepath):
            self.errors.append({
                'file': filepath,
                'error_type': 'other',
                'line_number': 0,
                'message': 'File does not exist',
                'suggested_fix': 'Remove from batch list or locate correct path'
            })
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            self.errors.append({
                'file': filepath,
                'error_type': 'other',
                'line_number': 0,
                'message': f'Cannot read file: {str(e)}',
                'suggested_fix': 'Check file permissions and encoding'
            })
            return
        
        if not content.strip():
            return  # Empty files are valid
        
        # Check 1: AST parsing for syntax errors
        self._check_ast(filepath, content)
        
        # Check 2: py_compile for compilation errors
        self._check_py_compile(filepath, content)
        
        # Check 3: Check for common import issues
        self._check_import_issues(filepath, content)
    
    def _check_ast(self, filepath: str, content: str):
        """Check file with AST parser."""
        try:
            ast.parse(content)
        except SyntaxError as e:
            self.errors.append({
                'file': filepath,
                'error_type': 'syntax_error',
                'line_number': e.lineno or 0,
                'message': f'Syntax error: {e.msg}',
                'suggested_fix': self._suggest_syntax_fix(e, content)
            })
        except Exception as e:
            self.errors.append({
                'file': filepath,
                'error_type': 'syntax_error',
                'line_number': 0,
                'message': f'Parse error: {str(e)}',
                'suggested_fix': 'Check for invalid Python syntax'
            })
    
    def _check_py_compile(self, filepath: str, content: str):
        """Check file with py_compile."""
        try:
            # Write to temp file and compile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                py_compile.compile(temp_path, doraise=True)
            finally:
                os.unlink(temp_path)
                
        except py_compile.PyCompileError as e:
            # Only add if not already reported by AST check
            if not any(e['file'] == filepath and e['error_type'] == 'syntax_error' for e in self.errors):
                self.errors.append({
                    'file': filepath,
                    'error_type': 'syntax_error',
                    'line_number': 0,
                    'message': f'Compilation error: {str(e)}',
                    'suggested_fix': 'Check for syntax errors or invalid characters'
                })
    
    def _check_import_issues(self, filepath: str, content: str):
        """Check for common import-related issues."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for relative imports
            if stripped.startswith('from .') or stripped.startswith('from ...'):
                level = 0
                for c in stripped[5:]:
                    if c == '.':
                        level += 1
                    else:
                        break
                
                # Check if relative import level is valid
                file_dir = Path(filepath).parent
                parent_levels = len(file_dir.parts) - len(self.project_root.parts)
                
                if level > parent_levels + 1:
                    self.errors.append({
                        'file': filepath,
                        'error_type': 'import_error',
                        'line_number': i,
                        'message': f'Relative import level {level} exceeds package depth',
                        'suggested_fix': 'Reduce relative import depth or use absolute imports'
                    })
            
            # Check for common typos in module names
            common_typos = {
                'crewai': ['crewAI', 'crew_ai', 'CrewAI'],
                'openevolve': ['openEvolve', 'OpenEvolve', 'open_evolve'],
                'datapizza': ['data_pizza', 'Datapizza', 'DataPizza'],
                'leanaide': ['lean_aide', 'Leanaide', 'LeanAide'],
                'numpy': ['num_py', 'Numpy', 'numPy'],
                'pandas': ['panda', 'Pandas'],
            }
            
            if stripped.startswith(('import ', 'from ')):
                for correct, typos in common_typos.items():
                    for typo in typos:
                        if typo in stripped and correct not in stripped.lower():
                            self.errors.append({
                                'file': filepath,
                                'error_type': 'import_error',
                                'line_number': i,
                                'message': f'Possible typo: "{typo}" should be "{correct}"',
                                'suggested_fix': f'Use correct module name: "{correct}"'
                            })
            
            # Check for star imports (potential namespace issues)
            if ' import *' in stripped and not stripped.startswith('#'):
                self.errors.append({
                    'file': filepath,
                    'error_type': 'import_error',
                    'line_number': i,
                    'message': 'Wildcard import used - potential namespace pollution',
                    'suggested_fix': 'Import specific symbols or use explicit module namespace'
                })
    
    def _suggest_syntax_fix(self, error: SyntaxError, content: str) -> str:
        """Suggest a fix for a syntax error."""
        msg = error.msg.lower()
        
        if 'unexpected eof' in msg:
            return 'Check for missing closing parentheses, brackets, or quotes'
        elif 'invalid syntax' in msg:
            lines = content.split('\n')
            if error.lineno and error.lineno <= len(lines):
                line = lines[error.lineno - 1]
                if '->' in line and 'def ' not in line and ':' not in line:
                    return 'Arrow syntax requires function definition context'
                if any(kw in line for kw in ['async', 'await']) and 'import ' not in line:
                    return 'Check async/await syntax - may need to be inside async function'
            return 'Check for invalid Python syntax or missing colons'
        elif 'eol while scanning string literal' in msg:
            return 'Check for unclosed string quotes'
        elif 'indent' in msg:
            return 'Check indentation consistency (spaces vs tabs)'
        elif 'invalid character' in msg:
            return 'Remove non-ASCII or invalid characters'
        else:
            return f'Fix syntax error: {error.msg}'


def main():
    scanner = ImportErrorScanner()
    
    file_list_path = r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_8.txt'
    output_path = r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend\import_errors_batch_8.json'
    
    print("Starting Batch 8 import error scan...")
    print("=" * 60)
    
    result = scanner.scan_files(file_list_path)
    
    # Remove error_key from output
    for error in result['errors']:
        error.pop('error_key', None)
    
    # Write JSON report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print("=" * 60)
    print(f"Scan complete!")
    print(f"  Total files scanned: {result['total_files']}")
    print(f"  Errors found: {result['errors_found']}")
    print(f"  Report saved to: {output_path}")
    
    # Print summary by error type
    if result['errors_found'] > 0:
        print("\nErrors by type:")
        error_counts = {}
        for error in result['errors']:
            error_type = error['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        for error_type, count in sorted(error_counts.items()):
            print(f"  - {error_type}: {count}")


if __name__ == '__main__':
    main()
