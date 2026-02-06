#!/usr/bin/env python3
"""
Scan Python files in batch 3 for import errors and syntax issues.
"""

import ast
import json
import os
import py_compile
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional


def check_syntax_with_ast(filepath: str) -> Optional[Dict[str, Any]]:
    """Check file for syntax errors using AST."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
        ast.parse(source)
        return None
    except SyntaxError as e:
        return {
            "error_type": "syntax_error",
            "line_number": e.lineno or 0,
            "message": f"SyntaxError: {e.msg}",
            "suggested_fix": f"Check syntax around line {e.lineno}, column {e.offset}"
        }
    except Exception as e:
        return {
            "error_type": "other",
            "line_number": 0,
            "message": f"AST parsing error: {str(e)}",
            "suggested_fix": "Check file encoding and content"
        }


def check_compile(filepath: str) -> Optional[Dict[str, Any]]:
    """Check file compiles with py_compile."""
    try:
        py_compile.compile(filepath, doraise=True)
        return None
    except py_compile.PyCompileError as e:
        line_num = 0
        msg = str(e)
        # Try to extract line number from error
        if "line" in msg.lower():
            try:
                parts = msg.lower().split("line")
                if len(parts) > 1:
                    num_part = parts[1].split()[0].strip()
                    line_num = int(num_part)
            except:
                pass
        return {
            "error_type": "syntax_error",
            "line_number": line_num,
            "message": f"Compile error: {msg}",
            "suggested_fix": "Fix syntax errors before checking imports"
        }


def extract_imports(filepath: str) -> List[Dict[str, Any]]:
    """Extract all imports from a file."""
    imports = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "line": node.lineno,
                        "alias": alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level
                imports.append({
                    "type": "from_import",
                    "module": module,
                    "level": level,
                    "line": node.lineno,
                    "names": [alias.name for alias in node.names]
                })
    except:
        pass
    return imports


def check_import_issues(filepath: str, imports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check for common import issues."""
    issues = []
    file_dir = os.path.dirname(filepath)
    project_root = r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend"
    
    for imp in imports:
        if imp["type"] == "from_import":
            module = imp["module"] or ""
            level = imp["level"]
            
            # Check relative imports
            if level > 0:
                # Try to resolve relative import
                parts = file_dir.replace(project_root, "").strip(os.sep).split(os.sep)
                if level <= len(parts):
                    target_parts = parts[:-level] if level < len(parts) else []
                    if module:
                        target_parts.extend(module.split("."))
                    target_path = os.path.join(project_root, *target_parts) + ".py"
                    target_init = os.path.join(project_root, *target_parts, "__init__.py")
                    
                    if not os.path.exists(target_path) and not os.path.exists(target_init):
                        issues.append({
                            "error_type": "import_error",
                            "line_number": imp["line"],
                            "message": f"Relative import cannot resolve: {'.' * level}{module}",
                            "suggested_fix": f"Check if module exists at expected path"
                        })
            else:
                # Check absolute imports that might be local
                first_part = module.split(".")[0] if module else ""
                if first_part and not is_stdlib_module(first_part):
                    # Check if it's a local module
                    local_path = os.path.join(project_root, first_part + ".py")
                    local_init = os.path.join(project_root, first_part, "__init__.py")
                    
                    if os.path.exists(local_path) or os.path.exists(local_init):
                        # It's a local module, check full path
                        full_module_path = os.path.join(project_root, *module.split(".")) + ".py"
                        full_init_path = os.path.join(project_root, *module.split("."), "__init__.py")
                        
                        if not os.path.exists(full_module_path) and not os.path.exists(full_init_path):
                            issues.append({
                                "error_type": "import_error",
                                "line_number": imp["line"],
                                "message": f"Module not found: {module}",
                                "suggested_fix": f"Check if module '{module}' exists"
                            })
        
        elif imp["type"] == "import":
            module = imp["module"]
            first_part = module.split(".")[0]
            
            if not is_stdlib_module(first_part):
                local_path = os.path.join(project_root, first_part + ".py")
                local_init = os.path.join(project_root, first_part, "__init__.py")
                
                if os.path.exists(local_path) or os.path.exists(local_init):
                    # It's a local module, check full path
                    full_module_path = os.path.join(project_root, *module.split(".")) + ".py"
                    full_init_path = os.path.join(project_root, *module.split("."), "__init__.py")
                    
                    if not os.path.exists(full_module_path) and not os.path.exists(full_init_path):
                        issues.append({
                            "error_type": "import_error",
                            "line_number": imp["line"],
                            "message": f"Module not found: {module}",
                            "suggested_fix": f"Check if module '{module}' exists"
                        })
    
    return issues


def is_stdlib_module(module_name: str) -> bool:
    """Check if a module is likely a Python stdlib module."""
    stdlib_modules = {
        'abc', 'argparse', 'ast', 'asyncio', 'base64', 'bisect', 'builtins',
        'calendar', 'collections', 'concurrent', 'configparser', 'contextlib',
        'copy', 'csv', 'ctypes', 'dataclasses', 'datetime', 'decimal', 'difflib',
        'dis', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
        'fnmatch', 'fractions', 'functools', 'gc', 'getopt', 'getpass', 'gettext',
        'glob', 'graphlib', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac', 'html',
        'http', 'idlelib', 'imaplib', 'imghdr', 'imp', 'importlib', 'inspect',
        'io', 'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3', 'linecache',
        'locale', 'logging', 'lzma', 'mailbox', 'mailcap', 'marshal', 'math',
        'mimetypes', 'mmap', 'modulefinder', 'multiprocessing', 'netrc', 'nis',
        'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev',
        'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform',
        'plistlib', 'poplib', 'posix', 'posixpath', 'pprint', 'profile',
        'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue',
        'quopri', 'random', 're', 'readline', 'reprlib', 'resource', 'rlcompleter',
        'runpy', 'sched', 'secrets', 'select', 'selectors', 'shelve', 'shlex',
        'shutil', 'signal', 'site', 'smtpd', 'smtplib', 'sndhdr', 'socket',
        'socketserver', 'spwd', 'sqlite3', 'ssl', 'stat', 'statistics', 'string',
        'stringprep', 'struct', 'subprocess', 'sunau', 'symtable', 'sys',
        'sysconfig', 'syslog', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile',
        'termios', 'test', 'textwrap', 'threading', 'time', 'timeit', 'trace',
        'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types',
        'typing', 'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 'venv',
        'warnings', 'wave', 'weakref', 'webbrowser', 'winreg', 'winsound',
        'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport',
        'zlib', '_thread', '__future__', 'zoneinfo', 'tomllib', 'typing_extensions'
    }
    return module_name in stdlib_modules or module_name.startswith('_')


def scan_file(filepath: str) -> List[Dict[str, Any]]:
    """Scan a single file for errors."""
    errors = []
    
    # Check if file exists
    if not os.path.exists(filepath):
        return [{
            "file": filepath,
            "error_type": "other",
            "line_number": 0,
            "message": "File does not exist",
            "suggested_fix": "Remove from batch list or restore file"
        }]
    
    # Check syntax with AST
    ast_error = check_syntax_with_ast(filepath)
    if ast_error:
        errors.append({"file": filepath, **ast_error})
        return errors  # Don't continue if syntax error
    
    # Check compilation
    compile_error = check_compile(filepath)
    if compile_error:
        errors.append({"file": filepath, **compile_error})
        return errors
    
    # Extract and check imports
    imports = extract_imports(filepath)
    import_issues = check_import_issues(filepath, imports)
    for issue in import_issues:
        errors.append({"file": filepath, **issue})
    
    return errors


def main():
    batch_file = r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_3.txt"
    output_file = r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\import_errors_batch_3.json"
    
    # Read batch file
    with open(batch_file, 'r', encoding='utf-8') as f:
        files = [line.strip() for line in f if line.strip()]
    
    print(f"Scanning {len(files)} files from batch 3...")
    
    all_errors = []
    total_scanned = 0
    
    for i, filepath in enumerate(files, 1):
        if not filepath.endswith('.py'):
            continue
        
        total_scanned += 1
        if i % 50 == 0:
            print(f"  Scanned {i}/{len(files)} files...")
        
        errors = scan_file(filepath)
        all_errors.extend(errors)
    
    # Create report
    report = {
        "total_files": total_scanned,
        "errors_found": len(all_errors),
        "errors": all_errors
    }
    
    # Write JSON report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nScan complete!")
    print(f"Total files scanned: {total_scanned}")
    print(f"Errors found: {len(all_errors)}")
    print(f"Report saved to: {output_file}")
    
    # Print summary by error type
    if all_errors:
        error_types = {}
        for err in all_errors:
            et = err["error_type"]
            error_types[et] = error_types.get(et, 0) + 1
        
        print("\nError breakdown:")
        for et, count in sorted(error_types.items()):
            print(f"  {et}: {count}")


if __name__ == "__main__":
    main()
