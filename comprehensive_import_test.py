"""
Comprehensive import test for all Lean-integrated files.
"""

import os
import sys
import importlib
from pathlib import Path

# Import bootstrap for proper paths
import lean_bootstrap

def get_lean_files():
    """Get all Python files with Lean integration."""
    lean_files = []
    root = Path('.')
    
    skip_dirs = {
        '__pycache__', '.git', 'node_modules', '.venv', 'venv',
        '.pytest_cache', '.mypy_cache', 'docs', 'test_results',
        'backups', 'core-projects'
    }
    
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith('.')]
        
        for filename in filenames:
            if not filename.endswith('.py'):
                continue
            if filename.startswith('test_'):
                continue
            
            filepath = Path(dirpath) / filename
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check if file has Lean integration
                if ('leanaide_client' in content or 
                    'lean4_integration' in content or
                    'LEAN_AVAILABLE' in content):
                    
                    rel_path = filepath.relative_to(root)
                    module = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')
                    lean_files.append((module, filepath))
            except:
                pass
    
    return lean_files

def test_import(module_name):
    """Test importing a module."""
    try:
        mod = importlib.import_module(module_name)
        lean_avail = getattr(mod, 'LEAN_AVAILABLE', 'NOT_SET')
        return True, lean_avail, None
    except Exception as e:
        return False, None, str(e)[:100]

def main():
    print("="*70)
    print("COMPREHENSIVE LEAN INTEGRATION IMPORT TEST")
    print("="*70)
    print()
    
    files = get_lean_files()
    print(f"Testing {len(files)} files with Lean integration...")
    print()
    
    success = 0
    failed = 0
    lean_true = 0
    lean_false = 0
    lean_not_set = 0
    
    failed_files = []
    lean_false_files = []
    
    for i, (module, filepath) in enumerate(files, 1):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(files)}...")
        
        ok, lean_avail, error = test_import(module)
        
        if ok:
            success += 1
            if lean_avail is True:
                lean_true += 1
            elif lean_avail is False:
                lean_false += 1
                lean_false_files.append(module)
            else:
                lean_not_set += 1
        else:
            failed += 1
            failed_files.append((module, error))
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"Total files: {len(files)}")
    print(f"  Successfully imported: {success}")
    print(f"  Failed to import: {failed}")
    print()
    print(f"LEAN_AVAILABLE=True: {lean_true}")
    print(f"LEAN_AVAILABLE=False: {lean_false}")
    print(f"LEAN_AVAILABLE not set: {lean_not_set}")
    print()
    
    if lean_false_files:
        print("FILES WITH LEAN_AVAILABLE=FALSE:")
        for f in lean_false_files[:10]:
            print(f"  - {f}")
        if len(lean_false_files) > 10:
            print(f"  ... and {len(lean_false_files) - 10} more")
        print()
    
    if failed_files:
        print("FILES WITH IMPORT ERRORS:")
        for f, err in failed_files[:10]:
            print(f"  - {f}: {err}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
        print()
    
    print("="*70)
    
    if failed == 0 and lean_false == 0:
        print("STATUS: ALL CHECKS PASSED")
        return 0
    else:
        print(f"STATUS: ISSUES FOUND - {failed} import failures, {lean_false} LEAN_AVAILABLE=False")
        return 1

if __name__ == "__main__":
    sys.exit(main())
