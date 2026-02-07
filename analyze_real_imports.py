#!/usr/bin/env python3
"""Analyze real import errors and find actual module locations."""

import os
import ast
from pathlib import Path
from collections import defaultdict

# Get all real Python files
real_files = {}
for root, dirs, files in os.walk('.', topdown=True):
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.venv', 'node_modules', '.git', 'openevolve_test_env']]
    
    for f in files:
        if f.endswith('.py') and f != '__init__.py':
            path = Path(root) / f
            module_name = f[:-3]  # Remove .py
            full_path = str(path)
            
            # Store by module name
            if module_name not in real_files:
                real_files[module_name] = []
            real_files[module_name].append(full_path)

print(f"Found {len(real_files)} unique module names")

# Find all imports and check if they resolve
import_errors = []

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
                        # Relative import - check if target exists
                        file_dir = filepath.parent
                        # Go up 'level' directories
                        target_dir = file_dir
                        for _ in range(level - 1):
                            target_dir = target_dir.parent
                        
                        if module:
                            # Check if module exists relative to target_dir
                            parts = module.split('.')
                            check_path = target_dir
                            for part in parts:
                                check_path = check_path / part
                            
                            if not (check_path.exists() or (check_path.parent / (check_path.name + '.py')).exists()):
                                import_errors.append({
                                    'file': str(filepath),
                                    'line': node.lineno,
                                    'import': f"from {'.' * level}{module} import ...",
                                    'type': 'relative',
                                    'module': module,
                                    'level': level
                                })
                        else:
                            # Just 'from . import name' - check if names exist in __init__.py
                            init_file = target_dir / '__init__.py'
                            if init_file.exists():
                                for alias in node.names:
                                    name = alias.name
                                    # Check if name is exported from __init__.py
                                    try:
                                        with open(init_file, 'r', encoding='utf-8', errors='ignore') as f:
                                            init_content = f.read()
                                        if f'__all__' in init_content:
                                            # Has __all__, check if name is in it
                                            if name not in init_content:
                                                import_errors.append({
                                                    'file': str(filepath),
                                                    'line': node.lineno,
                                                    'import': f"from {'.' * level} import {name}",
                                                    'type': 'relative_export',
                                                    'name': name,
                                                    'init_file': str(init_file)
                                                })
                                    except:
                                        pass
                    else:
                        # Absolute import
                        top_module = module.split('.')[0] if module else ''
                        
                        # Check if top_module exists
                        if top_module and top_module not in real_files:
                            # Could be a package
                            pkg_init = Path(top_module) / '__init__.py'
                            if not pkg_init.exists():
                                import_errors.append({
                                    'file': str(filepath),
                                    'line': node.lineno,
                                    'import': f"from {module} import ...",
                                    'type': 'absolute',
                                    'module': module,
                                    'top_module': top_module
                                })
                        
        except Exception as e:
            pass

print(f"\nFound {len(import_errors)} import errors")

# Group by module
by_module = defaultdict(list)
for err in import_errors:
    if err['type'] == 'absolute':
        key = err.get('top_module', err.get('module', 'unknown'))
    else:
        key = err.get('module', 'relative')
    by_module[key].append(err)

# Find real modules that might satisfy these
print("\n=== TOP 50 MISSING MODULES ===")
for mod, errors in sorted(by_module.items(), key=lambda x: len(x[1]), reverse=True)[:50]:
    count = len(errors)
    first_file = errors[0]['file']
    print(f"{mod}: {count} references")
    print(f"  Example: {first_file}:{errors[0]['line']}")
    
    # Check if a real file with similar name exists
    if mod in real_files:
        print(f"  -> FOUND REAL FILE: {real_files[mod]}")
    else:
        # Check partial matches
        for real_mod in real_files:
            if mod in real_mod or real_mod in mod:
                print(f"  -> SIMILAR: {real_mod} at {real_files[real_mod]}")
                break

# Save detailed report
import json
with open('real_import_errors.json', 'w') as f:
    json.dump({
        'errors': import_errors[:500],
        'by_module': {k: len(v) for k, v in by_module.items()}
    }, f, indent=2)

print(f"\nDetailed report saved to real_import_errors.json")
