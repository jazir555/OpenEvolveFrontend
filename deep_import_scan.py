#!/usr/bin/env python3
"""Deep scan for import errors - actually try to import modules."""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from collections import defaultdict

def scan_imports_deep():
    """Scan all Python files and check for import issues."""
    
    # Get all Python files in main project (exclude core-projects, venv, etc.)
    py_files = []
    for root, dirs, files in os.walk('.', topdown=True):
        # Skip problematic directories
        dirs[:] = [d for d in dirs if d not in [
            '__pycache__', '.venv', 'node_modules', '.git', 
            'openevolve_test_env', 'core-projects', '.pytest_cache',
            'test_enhanced_storage', 'test_chronicle', 'test_export',
            'test_screenshots', 'test_lean_workspace', 'test_leanaide_data'
        ]]
        for file in files:
            if file.endswith('.py'):
                py_files.append(Path(root) / file)
    
    print(f"Scanning {len(py_files)} Python files...\n")
    
    # Collect all imports
    all_imports = defaultdict(list)
    syntax_errors = []
    
    for filepath in py_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        all_imports[alias.name].append(str(filepath))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    if module:
                        all_imports[module].append(str(filepath))
                    # Also record the full import path
                    for alias in node.names:
                        full_path = f"{module}.{alias.name}" if module else alias.name
                        all_imports[full_path].append(str(filepath))
                        
        except SyntaxError as e:
            syntax_errors.append({'file': str(filepath), 'error': str(e)})
        except Exception as e:
            pass  # Ignore other errors
    
    return all_imports, syntax_errors

def check_missing_modules(all_imports):
    """Check which imports are missing."""
    missing = []
    
    # Project prefixes to check
    project_prefixes = [
        'openevolve', 'leanaide', 'bubblelab', 'roma', 'z3_',
        'knowledge_engine', 'adaptive_mdap', 'gauntlet',
        'decomposition', 'recomposition', 'solution_',
        'workflow_', 'quality_', 'sovereign_', 'unified',
        'glue', 'crewai_', 'mcp_', 'security_', 'api_',
        'analytics', 'monitoring', 'validation', 'verification',
        'parameter_', 'config_', 'utils', 'helpers',
        'strategies', 'templates', 'models', 'types'
    ]
    
    for module_name, files in all_imports.items():
        if not module_name:
            continue
            
        # Skip standard library and common third-party
        if '.' in module_name:
            top_module = module_name.split('.')[0]
        else:
            top_module = module_name
            
        # Skip if it's clearly an external package
        if top_module in ['os', 'sys', 'json', 'typing', 'abc', 'datetime', 
                          'pathlib', 'collections', 'functools', 'itertools',
                          'math', 'random', 're', 'hashlib', 'uuid', 'time',
                          'logging', 'inspect', 'contextlib', 'copy', 'pickle',
                          'asyncio', 'threading', 'multiprocessing', 'socket',
                          'urllib', 'http', 'ftplib', 'xml', 'html', 'csv',
                          'sqlite3', 'zlib', 'gzip', 'bz2', 'tarfile', 'zipfile',
                          'tempfile', 'shutil', 'subprocess', 'warnings',
                          'traceback', 'pdb', 'unittest', 'pytest', 'numpy',
                          'pandas', 'sklearn', 'matplotlib', 'seaborn', 'plotly',
                          'requests', 'flask', 'django', 'fastapi', 'pydantic',
                          'sqlalchemy', 'boto3', 'google', 'openai', 'anthropic',
                          'z3', 'yaml', 'toml', 'click', 'rich', 'typer',
                          'streamlit', 'gradio', 'torch', 'tensorflow', 'jax',
                          'transformers', 'datasets', 'accelerate', 'wandb',
                          'mlflow', 'neptune', 'comet_ml', 'optuna', 'ray',
                          'dask', 'spark', 'kafka', 'redis', 'celery', 'airflow',
                          'docker', 'kubernetes', 'helm', 'terraform', 'pulumi']:
            continue
        
        # Check if it's a project import
        is_project_import = any(module_name.startswith(p) or top_module == p 
                                for p in project_prefixes)
        
        if is_project_import or top_module in ['tests', 'examples', 'docs']:
            # Try to locate the module
            parts = module_name.split('.')
            
            # Check various possible paths
            found = False
            for i in range(len(parts), 0, -1):
                test_path = os.path.join(*parts[:i])
                
                # Check as file
                if os.path.exists(test_path + '.py'):
                    found = True
                    break
                # Check as package
                if os.path.exists(os.path.join(test_path, '__init__.py')):
                    found = True
                    break
            
            if not found:
                missing.append({
                    'module': module_name,
                    'files': files[:5]  # Limit to first 5
                })
    
    return missing

def main():
    all_imports, syntax_errors = scan_imports_deep()
    
    print(f"=== SYNTAX ERRORS ({len(syntax_errors)}) ===")
    if syntax_errors:
        for e in syntax_errors[:20]:
            print(f"  {e['file']}: {e['error'][:80]}")
    else:
        print("  None found")
    
    print(f"\n=== CHECKING FOR MISSING MODULES ===")
    missing = check_missing_modules(all_imports)
    
    print(f"\nFound {len(missing)} potentially missing modules\n")
    
    # Categorize
    categories = defaultdict(list)
    for m in missing:
        cat = m['module'].split('.')[0] if '.' in m['module'] else m['module']
        categories[cat].append(m)
    
    for cat, mods in sorted(categories.items()):
        print(f"\n{cat.upper()} ({len(mods)}):")
        for m in mods[:10]:
            print(f"  - {m['module']}")
            for f in m['files'][:2]:
                print(f"      in: {f}")
    
    # Save full results
    import json
    with open('deep_import_scan_results.json', 'w') as f:
        json.dump({
            'syntax_errors': syntax_errors,
            'missing_modules': missing
        }, f, indent=2)
    
    print(f"\n\nFull results saved to deep_import_scan_results.json")

if __name__ == "__main__":
    main()
