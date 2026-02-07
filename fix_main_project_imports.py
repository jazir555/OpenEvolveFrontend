#!/usr/bin/env python3
"""Fix imports only in the main project (not core-projects)."""

import os
import ast
from pathlib import Path
from collections import defaultdict

# Get main project files only (not in core-projects)
main_project_files = []
for root, dirs, files in os.walk('.', topdown=True):
    # Skip core-projects and other external directories
    dirs[:] = [d for d in dirs if d not in [
        '__pycache__', '.venv', 'node_modules', '.git', 
        'openevolve_test_env', 'core-projects', '.pytest_cache',
        'DeepKE_repo', 'projects to analyze'
    ]]
    
    for f in files:
        if f.endswith('.py'):
            main_project_files.append(os.path.join(root, f))

print(f"Main project files: {len(main_project_files)}")

# Get all real modules in main project
real_modules = set()
for f in main_project_files:
    path = Path(f)
    # Get module name
    if path.name == '__init__.py':
        module_name = path.parent.name
    else:
        module_name = path.stem
    real_modules.add(module_name)

print(f"Real modules in main project: {len(real_modules)}")

# Standard library
STDLIB = {'abc', 'argparse', 'ast', 'asyncio', 'base64', 'bisect', 'builtins',
    'calendar', 'collections', 'concurrent', 'configparser', 'contextlib', 'copy',
    'csv', 'ctypes', 'dataclasses', 'datetime', 'decimal', 'difflib', 'dis', 'email',
    'enum', 'errno', 'faulthandler', 'fnmatch', 'functools', 'gc', 'getopt', 'getpass',
    'gettext', 'glob', 'graphlib', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http',
    'idlelib', 'imaplib', 'imghdr', 'imp', 'inspect', 'io', 'ipaddress', 'itertools',
    'json', 'keyword', 'lib2to3', 'linecache', 'locale', 'logging', 'mailbox', 'math',
    'mimetypes', 'multiprocessing', 'netrc', 'numbers', 'operator', 'optparse', 'os',
    'pathlib', 'pdb', 'pickle', 'pickletools', 'pkgutil', 'platform', 'plistlib',
    'poplib', 'posixpath', 'pprint', 'profile', 'pstats', 'pty', 'pwd', 'py_compile',
    'pyclbr', 'pydoc', 'queue', 'quopri', 'random', 're', 'readline', 'reprlib',
    'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select', 'selectors',
    'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib', 'sndhdr',
    'socket', 'socketserver', 'sqlite3', 'ssl', 'stat', 'statistics', 'string',
    'stringprep', 'struct', 'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig',
    'syslog', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile', 'termios', 'test',
    'textwrap', 'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize',
    'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types',
    'typing', 'typing_extensions', 'unicodedata', 'unittest', 'urllib', 'uu',
    'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser', 'winreg',
    'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile',
    'zipimport', 'zlib', 'zoneinfo', '_thread', 'importlib', 'inspect', 'types',
    '__future__'}

# Known external libraries used in main project
EXTERNAL = {'pydantic', 'openai', 'z3', 'fastapi', 'sqlalchemy', 'chromadb', 
    'qdrant_client', 'sentence_transformers', 'sklearn', 'scipy', 'numpy', 'pandas',
    'matplotlib', 'psutil', 'sympy', 'dotenv'}

# Find real missing imports in main project
missing = defaultdict(list)

for filepath in main_project_files:
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
        except:
            continue
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ''
                level = node.level
                
                if level > 0:
                    continue  # Skip relative imports for now
                
                top_module = module.split('.')[0] if module else ''
                
                # Skip stdlib and known external
                if top_module in STDLIB or top_module in EXTERNAL:
                    continue
                
                # Check if module exists
                if module:
                    parts = module.split('.')
                    
                    # Check as file
                    file_path = Path(*parts).with_suffix('.py')
                    pkg_path = Path(*parts) / '__init__.py'
                    
                    if not (file_path.exists() or pkg_path.exists()):
                        # Not found, check if similar exists
                        if top_module not in real_modules:
                            missing[module].append(filepath)
                        else:
                            # Top module exists but subpath doesn't
                            # This might be a real issue
                            missing[module].append(filepath)
    except:
        pass

print(f"\n=== REAL MISSING IMPORTS IN MAIN PROJECT ===")
print(f"Total unique: {len(missing)}")

for mod, files in sorted(missing.items(), key=lambda x: len(x[1]), reverse=True)[:50]:
    count = len(files)
    print(f"\n{mod}: {count} references")
    print(f"  Example: {files[0]}")
    
    # Check if there's a similar real module
    top = mod.split('.')[0]
    if top in real_modules:
        print(f"  -> Top module '{top}' EXISTS but path '{mod}' doesn't")

# Save report
import json
with open('main_project_missing.json', 'w') as f:
    json.dump({k: v for k, v in missing.items()}, f, indent=2)

print(f"\nSaved to main_project_missing.json")
