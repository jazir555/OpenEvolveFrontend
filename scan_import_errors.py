#!/usr/bin/env python3
"""
Scan Python files for import errors and syntax issues.
"""

import ast
import json
import os
import py_compile
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def get_imports_from_ast(tree: ast.AST, file_path: str) -> List[Dict[str, Any]]:
    """Extract all imports from an AST tree."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    'type': 'import',
                    'module': alias.name,
                    'alias': alias.asname,
                    'line': node.lineno,
                    'level': 0
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            level = node.level
            for alias in node.names:
                imports.append({
                    'type': 'from_import',
                    'module': module,
                    'name': alias.name,
                    'alias': alias.asname,
                    'line': node.lineno,
                    'level': level
                })
    return imports


def check_syntax_with_ast(file_path: str) -> Optional[Dict[str, Any]]:
    """Check a file for syntax errors using ast.parse."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        ast.parse(source)
        return None
    except SyntaxError as e:
        return {
            'error_type': 'syntax_error',
            'line_number': e.lineno or 0,
            'message': str(e),
            'suggested_fix': f"Check syntax around line {e.lineno}: {e.text}"
        }
    except UnicodeDecodeError as e:
        return {
            'error_type': 'encoding_error',
            'line_number': 0,
            'message': f"Encoding error: {e}",
            'suggested_fix': "Check file encoding - should be UTF-8"
        }
    except Exception as e:
        return {
            'error_type': 'other',
            'line_number': 0,
            'message': f"Error parsing file: {e}",
            'suggested_fix': "Check file for corruption"
        }


def check_compile(file_path: str) -> Optional[Dict[str, Any]]:
    """Check a file for compilation errors using py_compile."""
    try:
        py_compile.compile(file_path, doraise=True)
        return None
    except py_compile.PyCompileError as e:
        line_num = 0
        msg = str(e)
        # Try to extract line number from error message
        match = re.search(r'line\s+(\d+)', msg, re.IGNORECASE)
        if match:
            line_num = int(match.group(1))
        return {
            'error_type': 'compilation_error',
            'line_number': line_num,
            'message': msg,
            'suggested_fix': "Fix syntax or encoding issues"
        }


def check_circular_imports(file_path: str, all_files: set, project_root: str) -> List[Dict[str, Any]]:
    """Check for potential circular imports by analyzing import statements."""
    errors = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        tree = ast.parse(source)
        imports = get_imports_from_ast(tree, file_path)
        
        file_dir = os.path.dirname(file_path)
        file_module = os.path.basename(file_path)[:-3]  # Remove .py
        
        for imp in imports:
            if imp['type'] == 'from_import' and imp['level'] > 0:
                # Relative import
                # Check if the target file exists
                target_path = file_dir
                for _ in range(imp['level'] - 1):
                    target_path = os.path.dirname(target_path)
                
                if imp['module']:
                    target_path = os.path.join(target_path, imp['module'].replace('.', os.sep))
                
                # Check various possible locations
                possible_paths = [
                    target_path + '.py',
                    os.path.join(target_path, '__init__.py'),
                    os.path.join(os.path.dirname(target_path), imp['module'].split('.')[-1] + '.py') if imp['module'] else ''
                ]
                
                if not any(os.path.exists(p) for p in possible_paths if p):
                    errors.append({
                        'error_type': 'relative_import_error',
                        'line_number': imp['line'],
                        'message': f"Relative import may not resolve: from {'.' * imp['level']}{imp['module']} import {imp.get('name', '*')}",
                        'suggested_fix': f"Ensure module exists at expected path: {target_path}"
                    })
    except Exception:
        pass  # Already handled in syntax check
    
    return errors


def check_common_import_issues(file_path: str) -> List[Dict[str, Any]]:
    """Check for common import issues like typos."""
    errors = []
    common_typos = {
        'stearnlet': 'streamlit',
        'streamilt': 'streamlit',
        'numpu': 'numpy',
        'numyp': 'numpy',
        'panda': 'pandas',
        'pands': 'pandas',
        'sqlachemy': 'sqlalchemy',
        'sqlalchmey': 'sqlalchemy',
        'fastpi': 'fastapi',
        'pydatic': 'pydantic',
        'reqeusts': 'requests',
        'urllib2': 'urllib',
        'ConfigParser': 'configparser',  # Python 2 vs 3
        'cPickle': 'pickle',  # Python 2 vs 3
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            # Check for import statements with typos
            if line.strip().startswith(('import ', 'from ')):
                for typo, correction in common_typos.items():
                    if typo in line:
                        errors.append({
                            'error_type': 'typo_in_import',
                            'line_number': line_num,
                            'message': f"Possible typo: '{typo}' should be '{correction}'",
                            'suggested_fix': f"Change '{typo}' to '{correction}'"
                        })
            
            # Check for Python 2 style imports
            if 'print ' in line and not line.strip().startswith('#'):
                if not line.strip().startswith('print('):
                    errors.append({
                        'error_type': 'python2_style',
                        'line_number': line_num,
                        'message': "Python 2 style print statement detected",
                        'suggested_fix': "Use print() function instead of print statement"
                    })
    except Exception:
        pass
    
    return errors


def check_import_resolution(file_path: str, project_root: str) -> List[Dict[str, Any]]:
    """Check if imports can be resolved within the project."""
    errors = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        tree = ast.parse(source)
        imports = get_imports_from_ast(tree, file_path)
        
        for imp in imports:
            module_name = imp['module']
            if not module_name:
                continue
            
            # Skip standard library and known third-party packages
            stdlib_modules = {
                'os', 'sys', 'json', 're', 'time', 'datetime', 'collections', 
                'typing', 'pathlib', 'abc', 'enum', 'logging', 'hashlib',
                'random', 'string', 'math', 'functools', 'itertools',
                'inspect', 'types', 'warnings', 'traceback', 'uuid',
                'copy', 'pickle', 'csv', 'io', 'contextlib', 'dataclasses',
                'asyncio', 'threading', 'multiprocessing', 'queue', 'socket',
                'http', 'urllib', 'base64', 'binascii', 'bisect', 'calendar',
                'codecs', 'decimal', 'fractions', 'numbers', 'statistics',
                'textwrap', 'unicodedata', 'html', 'xml', 'xmlrpc',
                'ftplib', 'imaplib', 'smtplib', 'poplib', 'telnetlib',
                'uuid', 'secrets', 'hmac', 'tempfile', 'shutil', 'glob',
                'fnmatch', 'filecmp', 'linecache', 'macpath', 'os.path',
                'dbm', 'sqlite3', 'zlib', 'gzip', 'bz2', 'lzma', 'zipfile',
                'tarfile', 'configparser', 'optparse', 'getopt', 'pprint',
                'reprlib', 'string', 'difflib', 'textwrap', 'readline',
                'rlcompleter', 'site', 'sysconfig', 'builtins', '__future__',
                'ast', 'py_compile', 'compileall', 'dis', 'tokenize',
                'pyclbr', 'importlib', 'pkgutil', 'modulefinder', 'runpy',
                'importlib.metadata', 'importlib_resources'
            }
            
            third_party_modules = {
                'numpy', 'pandas', 'streamlit', 'fastapi', 'pydantic', 'sqlalchemy',
                'requests', 'flask', 'django', 'pytest', 'matplotlib', 'seaborn',
                'plotly', 'bokeh', 'altair', 'torch', 'tensorflow', 'sklearn',
                'transformers', 'openai', 'anthropic', 'crewai', 'z3', 'networkx',
                'scipy', 'pillow', 'cv2', 'pygraphistry', 'neo4j', 'qdrant',
                'chromadb', 'redis', 'pymongo', 'psycopg2', 'asyncpg', 'alembic',
                'jinja2', 'markupsafe', 'werkzeug', 'click', 'typer', 'rich',
                'tqdm', 'colorama', 'termcolor', 'tabulate', 'prettytable',
                'yaml', 'toml', 'configobj', 'python-dotenv', 'environs',
                'pydantic_settings', 'httpx', 'aiohttp', 'websockets', 'grpc',
                'boto3', 'botocore', 'azure', 'google', 'gcsfs', 's3fs',
                'docker', 'kubernetes', 'helm', 'git', 'github', 'gitlab',
                'tox', 'nox', 'pre_commit', 'black', 'isort', 'flake8', 'mypy',
                'pylint', 'bandit', 'safety', 'pip', 'setuptools', 'wheel',
                'twine', 'build', 'hatch', 'poetry', 'pipenv', 'conda',
                'jupyter', 'ipython', 'nbformat', 'nbconvert', 'jupyterlab',
                'tornado', 'starlette', 'uvicorn', 'gunicorn', 'hypercorn',
                'daphne', 'channels', 'celery', 'rq', 'huey', 'dramatiq',
                'kombu', 'amqp', 'pika', 'redis', 'memcached', 'valkey'
            }
            
            base_module = module_name.split('.')[0]
            
            if base_module in stdlib_modules or base_module in third_party_modules:
                continue
            
            # Check if it's a local module
            if imp['type'] == 'from_import' and imp['level'] > 0:
                # Relative import - skip, handled elsewhere
                continue
            
            # Try to resolve local module
            module_path = module_name.replace('.', os.sep)
            possible_paths = [
                os.path.join(project_root, module_path + '.py'),
                os.path.join(project_root, module_path, '__init__.py'),
                os.path.join(os.path.dirname(file_path), module_path + '.py'),
                os.path.join(os.path.dirname(file_path), module_path, '__init__.py'),
            ]
            
            if not any(os.path.exists(p) for p in possible_paths):
                # Might be an unresolved import
                errors.append({
                    'error_type': 'unresolved_import',
                    'line_number': imp['line'],
                    'message': f"Cannot resolve import: {module_name}",
                    'suggested_fix': f"Ensure module '{module_name}' exists in project or is installed"
                })
    except Exception:
        pass
    
    return errors


def scan_file(file_path: str, all_files: set, project_root: str) -> List[Dict[str, Any]]:
    """Scan a single file for all types of import errors."""
    errors = []
    
    # Check syntax with AST
    syntax_error = check_syntax_with_ast(file_path)
    if syntax_error:
        errors.append({
            'file': file_path,
            **syntax_error
        })
        # If syntax error exists, skip other checks
        return errors
    
    # Check compilation
    compile_error = check_compile(file_path)
    if compile_error:
        errors.append({
            'file': file_path,
            **compile_error
        })
    
    # Check circular imports
    circular_errors = check_circular_imports(file_path, all_files, project_root)
    for err in circular_errors:
        errors.append({
            'file': file_path,
            **err
        })
    
    # Check common issues
    common_errors = check_common_import_issues(file_path)
    for err in common_errors:
        errors.append({
            'file': file_path,
            **err
        })
    
    # Check import resolution
    import_errors = check_import_resolution(file_path, project_root)
    for err in import_errors:
        errors.append({
            'file': file_path,
            **err
        })
    
    return errors


def main():
    batch_file = r'c:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_1.txt'
    output_file = r'c:\Users\mmeadow\Documents\OpenEvolve\Frontend\import_errors_batch_1.json'
    project_root = r'c:\Users\mmeadow\Documents\OpenEvolve\Frontend'
    
    # Read file list
    with open(batch_file, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    all_files = set(files)
    all_errors = []
    
    print(f"Scanning {len(files)} files for import errors...")
    
    for i, file_path in enumerate(files, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(files)} files scanned...")
        
        if not os.path.exists(file_path):
            all_errors.append({
                'file': file_path,
                'error_type': 'file_not_found',
                'line_number': 0,
                'message': f"File does not exist: {file_path}",
                'suggested_fix': "Remove from batch or create the file"
            })
            continue
        
        if not file_path.endswith('.py'):
            continue
        
        errors = scan_file(file_path, all_files, project_root)
        all_errors.extend(errors)
    
    # Generate report
    report = {
        'total_files': len(files),
        'errors_found': len(all_errors),
        'errors': all_errors
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nScan complete!")
    print(f"Total files scanned: {len(files)}")
    print(f"Errors found: {len(all_errors)}")
    print(f"Report saved to: {output_file}")
    
    # Print summary by error type
    if all_errors:
        error_types = {}
        for err in all_errors:
            et = err['error_type']
            error_types[et] = error_types.get(et, 0) + 1
        
        print("\nErrors by type:")
        for et, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {et}: {count}")


if __name__ == '__main__':
    main()
