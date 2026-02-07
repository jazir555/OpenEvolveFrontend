#!/usr/bin/env python3
"""Ultimate import check - finds all import resolution errors."""

import os
import sys
import ast
import subprocess
from pathlib import Path
from collections import defaultdict

# Get all Python files
py_files = []
for root, dirs, files in os.walk('.', topdown=True):
    dirs[:] = [d for d in dirs if d not in [
        '__pycache__', '.venv', 'node_modules', '.git',
        'openevolve_test_env', 'core-projects', '.pytest_cache'
    ]]
    for file in files:
        if file.endswith('.py'):
            py_files.append(os.path.join(root, file))

print(f"Scanning {len(py_files)} Python files...")

# Track issues
syntax_errors = []
import_errors = []

for filepath in py_files:
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        
        # Check syntax
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            syntax_errors.append({
                'file': filepath,
                'line': e.lineno,
                'error': str(e)
            })
            continue
        
        # Find all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_errors.append({
                        'file': filepath,
                        'line': node.lineno,
                        'type': 'import',
                        'module': alias.name,
                        'name': None
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                level = node.level
                for alias in node.names:
                    import_errors.append({
                        'file': filepath,
                        'line': node.lineno,
                        'type': 'from',
                        'module': module,
                        'name': alias.name,
                        'level': level
                    })
    except Exception as e:
        pass

print(f"\nFound {len(syntax_errors)} syntax errors")
print(f"Found {len(import_errors)} import statements")

# Categorize imports
standard_lib = []
third_party = []
project_local = []
unknown = []

stdlib_modules = {
    'abc', 'argparse', 'ast', 'asyncio', 'base64', 'bisect', 'builtins',
    'calendar', 'collections', 'concurrent', 'configparser', 'contextlib',
    'copy', 'csv', 'ctypes', 'dataclasses', 'datetime', 'decimal', 'difflib',
    'dis', 'email', 'enum', 'errno', 'faulthandler', 'fnmatch', 'functools',
    'gc', 'getopt', 'getpass', 'gettext', 'glob', 'graphlib', 'gzip', 'hashlib',
    'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib', 'imghdr', 'imp',
    'inspect', 'io', 'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3',
    'linecache', 'locale', 'logging', 'mailbox', 'math', 'mimetypes',
    'multiprocessing', 'netrc', 'numbers', 'operator', 'optparse', 'os',
    'pathlib', 'pdb', 'pickle', 'pickletools', 'pkgutil', 'platform',
    'plistlib', 'poplib', 'posixpath', 'pprint', 'profile', 'pstats', 'pty',
    'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri', 'random',
    're', 'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy',
    'sched', 'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil',
    'signal', 'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver',
    'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep',
    'struct', 'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig',
    'syslog', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile', 'termios',
    'test', 'textwrap', 'threading', 'time', 'timeit', 'tkinter', 'token',
    'tokenize', 'trace', 'traceback', 'tracemalloc', 'tty', 'turtle',
    'turtledemo', 'types', 'typing', 'unicodedata', 'unittest', 'urllib',
    'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser',
    'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp',
    'zipfile', 'zipimport', 'zlib', '_thread', '__future__', 'zoneinfo',
    'tomllib', 'typing_extensions'
}

common_third_party = {
    'numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn', 'plotly',
    'requests', 'flask', 'django', 'fastapi', 'pydantic', 'sqlalchemy',
    'boto3', 'botocore', 'openai', 'anthropic', 'google', 'z3', 'yaml',
    'toml', 'click', 'rich', 'typer', 'streamlit', 'gradio', 'torch',
    'tensorflow', 'jax', 'transformers', 'datasets', 'accelerate',
    'wandb', 'mlflow', 'optuna', 'ray', 'dask', 'pytest', 'pillow',
    'cv2', 'scipy', 'networkx', 'igraph', 'pygraphistry', 'nebula3',
    'neo4j', 'pymongo', 'redis', 'celery', 'docker', 'kubernetes',
    'jinja2', 'markupsafe', 'werkzeug', 'uvicorn', 'starlette',
    'anyio', 'sniffio', 'idna', 'chardet', 'certifi', 'urllib3',
    'attrs', 'cattrs', 'structlog', 'loguru', 'tqdm', 'dateutil',
    'pytz', 'packaging', 'setuptools', 'pip', 'wheel', 'setuptools',
    'pkg_resources', 'six', 'future', 'past', 'copyreg', 'pickle5',
    'multidict', 'yarl', 'async_timeout', 'aiohttp', 'aiofiles',
    'httpx', 'httptools', 'websockets', 'wsproto', 'h11', 'h2',
    'priority', 'hyperframe', 'brotli', 'brotlicffi', 'zstandard',
    'lz4', 'blosc', 'msgpack', 'ujson', 'orjson', 'cbor2',
    'base58', 'base62', 'crc32c', 'xxhash', 'murmurhash',
    'regex', 'pcre', 'lxml', 'html5lib', 'beautifulsoup4',
    'soupsieve', 'cssselect', 'parsel', 'pyquery', 'scrapy',
    'selenium', 'playwright', 'puppeteer', 'mechanize',
    'newspaper3k', 'trafilatura', 'readability', 'goose3',
    'sumy', 'nltk', 'spacy', 'stanza', 'transformers',
    'tokenizers', 'sentencepiece', 'sacremoses', 'subword_nmt',
    'mosestokenizer', 'indic_nlp_library', 'polyglot',
    'textblob', 'pattern', 'vaderSentiment', 'afinn',
    'langdetect', 'langid', 'fasttext', 'polyglot',
    'pycld2', 'pycld3', 'langdetect', 'ftfy', 'unicodedata2',
    'charset_normalizer', 'cchardet', 'chardet',
    'bloom_filter', 'pybloom_live', 'datasketch',
    'simhash', 'minhash', 'hyperloglog', 'bloom',
    'mmh3', 'farmhash', 'cityhash', 'metrohash',
    'bitarray', 'bitstring', 'bitstruct',
    'construct', 'kaitaistruct', 'protobuf', 'google',
    'grpc', 'thrift', 'avro', 'fastavro', 'parquet',
    'pyarrow', 'feather', 'orc', 'hdf5', 'netcdf4',
    'xarray', 'zarr', 'dask', 'cupy', 'numba',
    'cython', 'cffi', 'ctypes', 'swig', 'boost',
    'eigen', 'pybind11', 'nanobind', 'shiboken',
    'pyside2', 'pyside6', 'pyqt5', 'pyqt6', 'wx',
    'tkinter', 'dearpygui', 'imgui', 'pyglet',
    'arcade', 'pygame', 'panda3d', 'ursina',
    'moderngl', 'pyopengl', 'vispy', 'mayavi',
    'vtk', 'itk', 'simpleitk', 'nibabel', 'nilearn',
    'dipy', 'mne', 'pymvpa', 'nitime', 'nitools'
}

for imp in import_errors:
    if imp['type'] == 'import':
        top = imp['module'].split('.')[0]
    else:
        if imp['level'] > 0:
            # Relative import - project local
            project_local.append(imp)
            continue
        top = (imp['module'] or '').split('.')[0]
    
    if not top:
        continue
        
    if top in stdlib_modules:
        standard_lib.append(imp)
    elif top in common_third_party:
        third_party.append(imp)
    else:
        # Check if it looks like a project import
        project_indicators = [
            'openevolve', 'leanaide', 'bubblelab', 'roma', 'z3_',
            'knowledge', 'gauntlet', 'decomposition', 'recomposition',
            'solution_', 'workflow_', 'quality_', 'sovereign_',
            'glue', 'crewai_', 'mcp_', 'security_', 'api_',
            'analytics', 'monitoring', 'validation', 'verification',
            'parameter_', 'config_', 'utils', 'helpers',
            'strategies', 'templates', 'models', 'types',
            'ace', 'adaptive', 'adversarial', 'evolution',
            'tests', 'examples', 'docs'
        ]
        
        if any(top.lower().startswith(p) or p in top.lower() for p in project_indicators):
            project_local.append(imp)
        else:
            unknown.append(imp)

print(f"\n=== IMPORT BREAKDOWN ===")
print(f"Standard library: {len(standard_lib)}")
print(f"Third-party: {len(third_party)}")
print(f"Project local: {len(project_local)}")
print(f"Unknown: {len(unknown)}")

# Check if project local imports can be resolved
print(f"\n=== CHECKING PROJECT IMPORTS ===")
unresolved = []

for imp in project_local:
    if imp['type'] == 'import':
        module = imp['module']
    else:
        if imp['level'] > 0:
            # Relative import - resolve it
            file_dir = os.path.dirname(imp['file'])
            parts = []
            for _ in range(imp['level'] - 1):
                file_dir = os.path.dirname(file_dir)
            if imp['module']:
                module = os.path.join(file_dir, imp['module'].replace('.', os.sep))
            else:
                module = file_dir
        else:
            module = imp['module']
    
    if not module:
        continue
    
    # Check various forms
    found = False
    checks = [
        module.replace('.', os.sep) + '.py',
        os.path.join(module.replace('.', os.sep), '__init__.py'),
        module + '.py',
    ]
    
    for check in checks:
        if os.path.exists(check):
            found = True
            break
    
    if not found:
        unresolved.append(imp)

print(f"Unresolved project imports: {len(unresolved)}")

# Group by module
by_module = defaultdict(list)
for imp in unresolved:
    if imp['type'] == 'import':
        mod = imp['module']
    else:
        mod = imp['module'] or '(relative)'
    by_module[mod].append(imp)

print(f"\n=== TOP UNRESOLVED MODULES ===")
sorted_modules = sorted(by_module.items(), key=lambda x: len(x[1]), reverse=True)
for mod, imps in sorted_modules[:50]:
    print(f"\n{mod} ({len(imps)} references):")
    for imp in imps[:3]:
        print(f"  {imp['file']}:{imp['line']}")
    if len(imps) > 3:
        print(f"  ... and {len(imps)-3} more")

# Save results
import json
with open('ultimate_import_check.json', 'w') as f:
    json.dump({
        'syntax_errors': syntax_errors,
        'unresolved_imports': unresolved,
        'by_module': {k: [{'file': i['file'], 'line': i['line'], 'name': i.get('name')} for i in v[:10]] 
                      for k, v in sorted_modules[:200]}
    }, f, indent=2)

print(f"\n\nResults saved to ultimate_import_check.json")
print(f"Total syntax errors: {len(syntax_errors)}")
print(f"Total unresolved imports: {len(unresolved)}")
