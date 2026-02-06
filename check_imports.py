#!/usr/bin/env python3
"""Check for import errors in the project."""

import os
import sys
import ast
from pathlib import Path
from collections import defaultdict

def find_imports(filepath):
    """Find all imports in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        
        tree = ast.parse(source)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                imports.append(module)
        
        return imports
    except:
        return []

def scan_project_imports():
    """Scan all imports in the project."""
    print("Scanning project imports...")
    
    # Get all Python files
    py_files = []
    for root, dirs, files in os.walk('.', topdown=True):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.venv', 'node_modules', '.git', 'openevolve_test_env']]
        for file in files:
            if file.endswith('.py'):
                py_files.append(Path(root) / file)
    
    print(f"Found {len(py_files)} Python files")
    
    # Collect all imports
    all_imports = defaultdict(list)
    project_modules = set()
    
    for filepath in py_files:
        module_path = str(filepath).replace('/', '.').replace('\\', '.')[:-3]
        if module_path.startswith('.'):
            module_path = module_path[1:]
        project_modules.add(module_path)
        
        imports = find_imports(filepath)
        for imp in imports:
            all_imports[imp].append(str(filepath))
    
    # Find potential issues
    issues = []
    
    # Check for imports that reference non-existent project modules
    project_prefixes = [
        'openevolve', 'leanaide', 'bubblelab', 'roma', 'z3_', 'crewai_',
        'knowledge_engine', 'adaptive_mdap', 'gauntlet', 'decomposition',
        'recomposition', 'solution_', 'workflow_', 'quality_',
        'sovereign_', 'unified', 'glue'
    ]
    
    for imp, files in all_imports.items():
        if not imp:
            continue
            
        # Check if it's a project import
        is_project_import = any(imp.startswith(p) for p in project_prefixes)
        
        if is_project_import:
            # Check if module exists
            # Try different path variations
            possible_paths = [
                imp.replace('.', '/') + '.py',
                imp.replace('.', '\\') + '.py',
                imp.replace('.', '/') + '/__init__.py',
                imp.replace('.', '\\') + '\\__init__.py',
            ]
            
            exists = any(Path(p).exists() for p in possible_paths)
            
            if not exists:
                # Could be a submodule - check if parent package exists
                parts = imp.split('.')
                for i in range(len(parts), 0, -1):
                    parent = '.'.join(parts[:i])
                    parent_paths = [
                        parent.replace('.', '/') + '.py',
                        parent.replace('.', '\\') + '.py',
                        parent.replace('.', '/') + '/__init__.py',
                        parent.replace('.', '\\') + '\\__init__.py',
                    ]
                    if any(Path(p).exists() for p in parent_paths):
                        exists = True
                        break
                
                if not exists:
                    issues.append({
                        'type': 'missing_module',
                        'module': imp,
                        'files': files[:3]  # Show first 3 files
                    })
    
    return issues, all_imports

def main():
    issues, all_imports = scan_project_imports()
    
    print(f"\n=== IMPORT ANALYSIS ===")
    print(f"Total unique imports: {len(all_imports)}")
    
    if issues:
        print(f"\n=== POTENTIAL IMPORT ISSUES ({len(issues)}) ===")
        for issue in issues[:50]:  # Show first 50
            print(f"\n  Missing module: {issue['module']}")
            print(f"    Referenced in:")
            for f in issue['files']:
                print(f"      - {f}")
    else:
        print("\n✓ No obvious import issues found!")
    
    # Save results
    with open('import_issues.json', 'w') as f:
        import json
        json.dump(issues, f, indent=2)
    
    print(f"\nResults saved to import_issues.json")

if __name__ == "__main__":
    main()
