#!/usr/bin/env python3
"""
Scan Python files for import errors (Batch 2)

This script scans Python files for:
1. Syntax errors (using ast module)
2. Compilation errors (using py_compile)
3. Import issues (circular imports, missing imports, relative imports, typos)
"""

import ast
import json
import os
import py_compile
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class ImportErrorScanner:
    """Scanner for detecting import errors in Python files."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.errors: List[Dict] = []
        self.processed_files: Set[str] = set()
        self.file_imports: Dict[str, Set[str]] = {}
        self.all_modules: Set[str] = set()
        
    def scan_file(self, filepath: str) -> Optional[Dict]:
        """Scan a single Python file for import errors."""
        file_path = Path(filepath).resolve()
        
        if not file_path.exists():
            return {
                "file": str(file_path),
                "error_type": "other",
                "line_number": 0,
                "message": f"File not found: {file_path}",
                "suggested_fix": "Check file path"
            }
        
        if not file_path.suffix == '.py':
            return None  # Skip non-Python files
            
        self.processed_files.add(str(file_path))
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
        except Exception as e:
            return {
                "file": str(file_path),
                "error_type": "other",
                "line_number": 0,
                "message": f"Cannot read file: {str(e)}",
                "suggested_fix": "Check file permissions and encoding"
            }
        
        errors = []
        
        # 1. Check for syntax errors with ast
        syntax_error = self._check_syntax_ast(file_path, source)
        if syntax_error:
            errors.append(syntax_error)
            # If there's a syntax error, we can't proceed with AST analysis
            return errors[0] if len(errors) == 1 else errors
        
        # 2. Check for compilation errors with py_compile
        compile_error = self._check_py_compile(file_path)
        if compile_error:
            errors.append(compile_error)
        
        # 3. Parse imports and analyze them
        try:
            tree = ast.parse(source)
            import_errors = self._analyze_imports(file_path, tree, source)
            errors.extend(import_errors)
            
            # Store imports for circular import detection
            self._store_imports(file_path, tree)
            
        except SyntaxError:
            # Already caught above
            pass
        except Exception as e:
            errors.append({
                "file": str(file_path),
                "error_type": "other",
                "line_number": 0,
                "message": f"Failed to parse AST: {str(e)}",
                "suggested_fix": "Check for unusual Python syntax"
            })
        
        # 4. Check for common typos in imports
        typo_errors = self._check_import_typos(file_path, source)
        errors.extend(typo_errors)
        
        # 5. Check for relative import issues
        relative_errors = self._check_relative_imports(file_path, source)
        errors.extend(relative_errors)
        
        if errors:
            return errors[0] if len(errors) == 1 else errors
        return None
    
    def _check_syntax_ast(self, file_path: Path, source: str) -> Optional[Dict]:
        """Check for syntax errors using the ast module."""
        try:
            ast.parse(source)
            return None
        except SyntaxError as e:
            return {
                "file": str(file_path),
                "error_type": "syntax_error",
                "line_number": e.lineno or 0,
                "message": f"Syntax error: {e.msg}",
                "suggested_fix": f"Check line {e.lineno}: {e.text.strip() if e.text else 'N/A'}"
            }
        except Exception as e:
            return {
                "file": str(file_path),
                "error_type": "syntax_error",
                "line_number": 0,
                "message": f"Parse error: {str(e)}",
                "suggested_fix": "Check file for encoding issues or binary content"
            }
    
    def _check_py_compile(self, file_path: Path) -> Optional[Dict]:
        """Check for compilation errors using py_compile."""
        try:
            py_compile.compile(str(file_path), doraise=True)
            return None
        except py_compile.PyCompileError as e:
            # Extract line number from error if possible
            line_num = 0
            msg = str(e)
            
            # Try to extract line number from error message
            match = re.search(r'line\s+(\d+)', msg, re.IGNORECASE)
            if match:
                line_num = int(match.group(1))
            
            return {
                "file": str(file_path),
                "error_type": "syntax_error",
                "line_number": line_num,
                "message": f"Compilation error: {msg}",
                "suggested_fix": "Fix syntax error before running"
            }
    
    def _analyze_imports(self, file_path: Path, tree: ast.AST, source: str) -> List[Dict]:
        """Analyze imports for potential issues."""
        errors = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                    level = node.level  # Relative import level
                    
                    # Check for relative imports
                    if level > 0:
                        # Validate relative import can resolve
                        if not self._validate_relative_import(file_path, module, level):
                            errors.append({
                                "file": str(file_path),
                                "error_type": "import_error",
                                "line_number": node.lineno,
                                "message": f"Relative import '{'.' * level}{module}' may not resolve",
                                "suggested_fix": f"Check that the module exists at the relative path"
                            })
                    else:
                        # Absolute import - check if it's a local module
                        if module and self._is_local_module(module):
                            if not self._module_exists(module):
                                errors.append({
                                    "file": str(file_path),
                                    "error_type": "import_error",
                                    "line_number": node.lineno,
                                    "message": f"Module '{module}' not found in project",
                                    "suggested_fix": f"Check if module name is correct or file exists"
                                })
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name
                        if self._is_local_module(name) and not self._module_exists(name):
                            errors.append({
                                "file": str(file_path),
                                "error_type": "import_error",
                                "line_number": node.lineno,
                                "message": f"Module '{name}' not found in project",
                                "suggested_fix": f"Check if module name is correct or file exists"
                            })
        
        return errors
    
    def _store_imports(self, file_path: Path, tree: ast.AST):
        """Store imports for circular import detection."""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
        
        self.file_imports[str(file_path)] = imports
        
        # Add module to all_modules
        module_name = self._path_to_module(file_path)
        if module_name:
            self.all_modules.add(module_name)
    
    def _check_import_typos(self, file_path: Path, source: str) -> List[Dict]:
        """Check for common typos in import statements."""
        errors = []
        
        # Common typo patterns
        typo_patterns = [
            (r'from\s+\.+\s+import', "Relative import with spaces"),
            (r'import\s+\.', "Invalid import syntax"),
            (r'from\s+\s+import', "Empty module name in import"),
        ]
        
        lines = source.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, desc in typo_patterns:
                if re.search(pattern, line):
                    errors.append({
                        "file": str(file_path),
                        "error_type": "syntax_error",
                        "line_number": i,
                        "message": f"Possible import typo: {desc}",
                        "suggested_fix": "Check import statement syntax"
                    })
        
        return errors
    
    def _check_relative_imports(self, file_path: Path, source: str) -> List[Dict]:
        """Check for relative import issues."""
        errors = []
        
        # Find relative imports
        relative_import_pattern = r'^(from\s+(\.+)(\w*)\s+import)'
        
        lines = source.split('\n')
        for i, line in enumerate(lines, 1):
            match = re.match(relative_import_pattern, line.strip())
            if match:
                dots = match.group(2)
                module = match.group(3)
                level = len(dots)
                
                # Check if relative import is too deep for file location
                file_depth = len(file_path.parent.relative_to(self.project_root).parts)
                
                if level > file_depth:
                    errors.append({
                        "file": str(file_path),
                        "error_type": "import_error",
                        "line_number": i,
                        "message": f"Relative import goes beyond package root ({level} levels up, but file is only {file_depth} levels deep)",
                        "suggested_fix": "Use absolute imports or reduce relative import depth"
                    })
        
        return errors
    
    def _validate_relative_import(self, file_path: Path, module: Optional[str], level: int) -> bool:
        """Validate if a relative import can resolve."""
        try:
            # Calculate the target directory
            parent = file_path.parent
            for _ in range(level - 1):
                parent = parent.parent
                if parent == self.project_root.parent:
                    return False
            
            if module:
                # Check if module exists as file or package
                module_path = parent / (module.replace('.', '/') + '.py')
                package_path = parent / module.replace('.', '/')
                
                return module_path.exists() or (package_path.exists() and (package_path / '__init__.py').exists())
            else:
                # Just checking if we can go up that many levels
                return True
        except:
            return False
    
    def _is_local_module(self, module_name: str) -> bool:
        """Check if a module name appears to be a local project module."""
        # Check if it's not a standard library or third-party module
        stdlib_modules = {
            'abc', 'argparse', 'ast', 'asyncio', 'base64', 'binascii', 'bisect',
            'builtins', 'bz2', 'calendar', 'cmath', 'collections', 'concurrent',
            'configparser', 'contextlib', 'copy', 'csv', 'ctypes', 'datetime',
            'decimal', 'difflib', 'dis', 'enum', 'errno', 'faulthandler',
            'filecmp', 'fnmatch', 'functools', 'gc', 'getopt', 'getpass',
            'gettext', 'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html',
            'http', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools',
            'json', 'keyword', 'linecache', 'locale', 'logging', 'lzma',
            'math', 'mimetypes', 'multiprocessing', 'numbers', 'operator',
            'optparse', 'os', 'pathlib', 'pickle', 'pkgutil', 'platform',
            'posixpath', 'pprint', 'profile', 'pstats', 'pwd', 'queue',
            'random', 're', 'reprlib', 'resource', 'select', 'selectors',
            'shlex', 'shutil', 'signal', 'socket', 'sqlite3', 'ssl', 'stat',
            'statistics', 'string', 'struct', 'subprocess', 'sys', 'tarfile',
            'tempfile', 'textwrap', 'threading', 'time', 'timeit', 'token',
            'tokenize', 'traceback', 'types', 'typing', 'unicodedata',
            'unittest', 'urllib', 'uuid', 'warnings', 'weakref', 'xml',
            'xmlrpc', 'zipfile', 'zipimport', 'zlib'
        }
        
        # Common third-party modules
        third_party = {
            'numpy', 'pandas', 'requests', 'flask', 'django', 'fastapi',
            'pydantic', 'sqlalchemy', 'pytest', 'click', 'jinja2',
            'yaml', 'toml', ' PIL', 'matplotlib', 'sklearn', 'tensorflow',
            'torch', 'cv2', 'bs4', 'lxml', 'httpx', 'aiohttp', 'tornado',
            'celery', 'redis', 'mongoengine', 'pymongo', 'psycopg2',
            'boto3', 'botocore', 'google', 'azure', 'z3', 'streamlit',
            'crewai', 'langchain', 'openai', 'anthropic'
        }
        
        first_part = module_name.split('.')[0]
        
        if first_part in stdlib_modules or first_part in third_party:
            return False
        
        return True
    
    def _module_exists(self, module_name: str) -> bool:
        """Check if a module exists in the project."""
        # Convert module name to path
        module_path = module_name.replace('.', '/')
        
        # Check as a Python file
        py_file = self.project_root / (module_path + '.py')
        if py_file.exists():
            return True
        
        # Check as a package
        pkg_dir = self.project_root / module_path
        if pkg_dir.exists() and (pkg_dir / '__init__.py').exists():
            return True
        
        return False
    
    def _path_to_module(self, file_path: Path) -> Optional[str]:
        """Convert a file path to module name."""
        try:
            rel_path = file_path.relative_to(self.project_root)
            if rel_path.name == '__init__.py':
                return str(rel_path.parent).replace('/', '.').replace('\\', '.')
            else:
                return str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')
        except:
            return None
    
    def detect_circular_imports(self) -> List[Dict]:
        """Detect circular imports between files."""
        circular = []
        
        # Build dependency graph
        for file_path, imports in self.file_imports.items():
            for imp in imports:
                # Check if this import creates a cycle
                if self._creates_cycle(file_path, imp):
                    circular.append({
                        "file": file_path,
                        "error_type": "circular_import",
                        "line_number": 0,
                        "message": f"Potential circular import with '{imp}'",
                        "suggested_fix": "Refactor to break circular dependency, consider using TYPE_CHECKING or lazy imports"
                    })
        
        return circular
    
    def _creates_cycle(self, file_path: str, import_name: str) -> bool:
        """Check if importing a module creates a circular dependency."""
        # Find the file that defines this import
        imported_file = None
        for fp, mods in self.file_imports.items():
            if fp != file_path:
                mod_name = self._path_to_module(Path(fp))
                if mod_name == import_name or (import_name and mod_name and mod_name.endswith(import_name)):
                    imported_file = fp
                    break
        
        if not imported_file:
            return False
        
        # Check if the imported file imports back to the original file
        original_module = self._path_to_module(Path(file_path))
        if not original_module:
            return False
        
        for imp in self.file_imports.get(imported_file, set()):
            if imp == original_module or original_module.startswith(imp + '.'):
                return True
        
        return False
    
    def scan_batch(self, file_list_path: str) -> Dict:
        """Scan all files in a batch list."""
        # Read file list
        with open(file_list_path, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        
        print(f"Scanning {len(files)} files from {file_list_path}...")
        
        errors_found = 0
        
        for i, filepath in enumerate(files, 1):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(files)} files scanned...")
            
            result = self.scan_file(filepath)
            if result:
                if isinstance(result, list):
                    for err in result:
                        self.errors.append(err)
                        errors_found += 1
                else:
                    self.errors.append(result)
                    errors_found += 1
        
        # Detect circular imports
        print("Detecting circular imports...")
        circular_errors = self.detect_circular_imports()
        self.errors.extend(circular_errors)
        errors_found += len(circular_errors)
        
        return {
            "total_files": len(files),
            "errors_found": errors_found,
            "errors": self.errors
        }


def main():
    project_root = r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend"
    batch_file = r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_2.txt"
    output_file = r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\import_errors_batch_2.json"
    
    scanner = ImportErrorScanner(project_root)
    report = scanner.scan_batch(batch_file)
    
    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Scan Complete!")
    print(f"{'='*60}")
    print(f"Total files scanned: {report['total_files']}")
    print(f"Errors found: {report['errors_found']}")
    print(f"\nReport saved to: {output_file}")
    
    # Print summary by error type
    if report['errors']:
        print(f"\nError breakdown:")
        error_types = {}
        for err in report['errors']:
            etype = err.get('error_type', 'unknown')
            error_types[etype] = error_types.get(etype, 0) + 1
        
        for etype, count in sorted(error_types.items()):
            print(f"  - {etype}: {count}")
    
    return report


if __name__ == "__main__":
    main()
