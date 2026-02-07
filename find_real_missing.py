#!/usr/bin/env python3
"""Find only the REAL missing project modules (not external libraries)."""

import os
import ast
from pathlib import Path
from collections import defaultdict

# Python standard library
STDLIB = {
    '__future__', 'abc', 'argparse', 'ast', 'asyncio', 'base64', 'bisect', 'builtins',
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
    'zipimport', 'zlib', 'zoneinfo', '_thread', 'importlib', 'inspect', 'types'
}

# External/third-party libraries
EXTERNAL = {
    'crewai', 'crewai_tools', 'dspy', 'ragbits', 'graphistry', 'neuromancer',
    'datapizza', 'deprecated', 'causallearn', 'outlines', 'rich', 'sklearn',
    'scipy', 'fastapi', 'uvicorn', 'starlette', 'lmql', 'itertools', 'backend',
    'src', 'opentelemetry', 'loguru', 'chromadb', 'tqdm', 'IPython', 'copy',
    'langchain_core', 'hydra', 'uuid', 'textual', 'sentence_transformers',
    'langgraph', 'inspect_ai', 'detllm', 'PIL', 'sqlalchemy', 'crytic_compile',
    'cryptography', 'z3', 'qdrant_client', 'jose', 'jwt', 'hmac', 'pytest',
    'pytest_asyncio', 'coverage', 'docx', 'PyPDF2', 'openpyxl', 'psutil',
    'sympy', 'pygraphistry', 'neo4j', 'PAMI', 'rdkit', 'biopython', 'karateclub',
    'textstat', 'e2b', 'rlm', 'steer', 'pami_research_quest_curie_globalchem_adapter',
    'lmql_dspy_adapter', 'gamma1', 'phase1', 'phase3_health', 'physicsnemo',
    'gamma', 'agentjson', 'OneKE', 'neuralkg', 'win32api', 'croniter', 'zulip',
    'strawberry', 'nlp_layer', 'intelligent_orchestrator', 'sysconfig', 'mpmath',
    'lean4_atp_bridge', 'phase4', 'tree_update_reminder', 'mlflow', 'wandb',
    'comet_ml', 'sacred', 'optuna', 'ray', 'kubernetes', 'docker', 'celery',
    'redis', 'boto3', 'azure', 'gcp', 'torch', 'tensorflow', 'jax', 'transformers',
    'datasets', 'accelerate', 'peft', 'trl', 'bitsandbytes', 'vllm', 'guidance',
    'instructor', 'marvin', 'llama_cpp', 'ctransformers', 'auto_gptq', 'auto_awq',
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'plotly', 'altair', 'bokeh',
    'holoviews', 'datashader', 'panel', 'param', 'streamlit', 'gradio', 'dash',
    'voila', 'ipywidgets', 'jinja2', 'markupsafe', 'werkzeug', 'anyio', 'sniffio',
    'idna', 'chardet', 'certifi', 'urllib3', 'attrs', 'cattrs', 'dateutil',
    'pytz', 'packaging', 'setuptools', 'six', 'future', 'multidict', 'yarl',
    'async_timeout', 'aiohttp', 'aiofiles', 'httpx', 'httptools', 'websockets',
    'wsproto', 'h11', 'h2', 'priority', 'hyperframe', 'brotli', 'brotlicffi',
    'zstandard', 'lz4', 'blosc', 'msgpack', 'ujson', 'orjson', 'cbor2',
    'base58', 'base62', 'crc32c', 'xxhash', 'mmh3', 'farmhash', 'cityhash',
    'metrohash', 'bitarray', 'bitstring', 'bitstruct', 'construct', 'kaitaistruct',
    'protobuf', 'avro', 'fastavro', 'parquet', 'pyarrow', 'feather', 'orc',
    'hdf5', 'netcdf4', 'xarray', 'zarr', 'cupy', 'numba', 'cython', 'cffi',
    'pybind11', 'nanobind', 'bio', 'biopython'
}

# Get all real Python files
real_modules = {}
for root, dirs, files in os.walk('.', topdown=True):
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.venv', 'node_modules', '.git', 'openevolve_test_env']]
    
    for f in files:
        if f.endswith('.py'):
            path = Path(root) / f
            module_name = f[:-3]
            if module_name not in real_modules:
                real_modules[module_name] = []
            real_modules[module_name].append(str(path))

print(f"Found {len(real_modules)} real module names")

# Find imports that are truly missing from the project
real_missing = defaultdict(list)

for root, dirs, files in os.walk('.', topdown=True):
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.venv', 'node_modules', '.git', 'openevolve_test_env']]
    
    for f in files:
        if not f.endswith('.py'):
            continue
            
        filepath = Path(root) / f
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                source = file.read()
            
            try:
                tree = ast.parse(source)
            except:
                continue
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    level = node.level
                    
                    if level > 0:
                        continue  # Skip relative for now
                    
                    top_module = module.split('.')[0] if module else ''
                    
                    # Skip stdlib and external
                    if top_module in STDLIB or top_module in EXTERNAL:
                        continue
                    
                    # Check if module exists
                    if module:
                        parts = module.split('.')
                        check_paths = [
                            Path(*parts).with_suffix('.py'),
                            Path(*parts) / '__init__.py'
                        ]
                        
                        found = False
                        for check in check_paths:
                            if check.exists():
                                found = True
                                break
                        
                        if not found:
                            real_missing[module].append(str(filepath))
        except:
            pass

print(f"\n=== REAL MISSING PROJECT MODULES ===")
print(f"Total unique: {len(real_missing)}")
print(f"\nTop 100 by reference count:")

for mod, files in sorted(real_missing.items(), key=lambda x: len(x[1]), reverse=True)[:100]:
    count = len(files)
    print(f"\n{mod}: {count} references")
    print(f"  Example: {files[0]}")
    
    # Suggest real module if similar exists
    parts = mod.split('.')
    for part in parts:
        if part in real_modules:
            print(f"  -> FOUND: {part} at {real_modules[part]}")
            break

# Save report
import json
with open('real_missing_modules.json', 'w') as f:
    json.dump({mod: files for mod, files in real_missing.items()}, f, indent=2)

print(f"\n\nSaved to real_missing_modules.json")
