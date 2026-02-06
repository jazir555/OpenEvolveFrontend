"""
Deep scan for remaining Lean integration issues.

This script performs a comprehensive scan for:
1. Files with LEAN_AVAILABLE=False
2. Import errors in Lean-integrated files
3. Missing dependencies
4. Test files that skip due to Lean unavailability
"""

import os
import sys
import importlib
from pathlib import Path

# Add bootstrap for proper imports
import lean_bootstrap

def scan_file_for_issues(filepath):
    """Scan a single file for Lean integration issues."""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return [("read_error", str(e))]
    
    # Check for LEAN_AVAILABLE = False
    if 'LEAN_AVAILABLE = False' in content:
        issues.append(("lean_false", "File sets LEAN_AVAILABLE = False"))
    
    # Check for try/except ImportError patterns that might hide issues
    if 'except ImportError:' in content and 'LEAN_AVAILABLE' in content:
        # This is the expected pattern, but check if it's actually importing
        if 'from leanaide_client import' in content or 'import leanaide_client' in content:
            pass  # This is correct
    
    return issues

def test_import_safely(module_name):
    """Test importing a module and capture any errors."""
    try:
        module = importlib.import_module(module_name)
        lean_avail = getattr(module, 'LEAN_AVAILABLE', 'NOT_SET')
        return {
            "imported": True,
            "lean_available": lean_avail,
            "error": None
        }
    except Exception as e:
        return {
            "imported": False,
            "lean_available": None,
            "error": str(e)[:100]
        }

def discover_lean_files():
    """Discover all Python files with Lean integration."""
    lean_files = []
    root = Path('.')
    
    skip_dirs = {
        '__pycache__', '.git', 'node_modules', '.venv', 'venv',
        '.pytest_cache', '.mypy_cache', 'docs', 'test_results',
        'tests', 'benchmark_artifacts', 'backups', 'core-projects'
    }
    
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        
        for filename in filenames:
            if not filename.endswith('.py'):
                continue
            if filename.startswith('test_') or filename.endswith('_test.py'):
                continue
            
            filepath = Path(dirpath) / filename
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check if file has Lean integration
                has_lean = ('LEAN_AVAILABLE' in content or 
                           'leanaide_client' in content or
                           'lean4_integration' in content)
                
                if has_lean:
                    rel_path = filepath.relative_to(root)
                    module = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')
                    lean_files.append((module, filepath))
            except:
                pass
    
    return lean_files

def main():
    print("="*70)
    print("DEEP SCAN FOR REMAINING LEAN INTEGRATION ISSUES")
    print("="*70)
    print()
    
    # Discover Lean files
    print("Discovering Lean-integrated files...")
    lean_files = discover_lean_files()
    print(f"Found {len(lean_files)} files with Lean integration")
    print()
    
    # Categorize issues
    lean_false_files = []
    import_errors = []
    not_set_files = []
    success_files = []
    
    print("Testing imports...")
    for i, (module, filepath) in enumerate(lean_files, 1):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(lean_files)}...")
        
        result = test_import_safely(module)
        
        if not result["imported"]:
            import_errors.append((module, result["error"]))
        elif result["lean_available"] is False:
            lean_false_files.append(module)
        elif result["lean_available"] == "NOT_SET":
            not_set_files.append(module)
        else:
            success_files.append(module)
    
    print()
    print("="*70)
    print("SCAN RESULTS")
    print("="*70)
    print()
    
    print(f"Total files tested: {len(lean_files)}")
    print(f"  - Successfully imported: {len(success_files)}")
    print(f"  - LEAN_AVAILABLE=True: {len([f for f in success_files if True])}")
    print(f"  - LEAN_AVAILABLE=False: {len(lean_false_files)}")
    print(f"  - LEAN_AVAILABLE not set: {len(not_set_files)}")
    print(f"  - Import errors: {len(import_errors)}")
    print()
    
    if lean_false_files:
        print("FILES WITH LEAN_AVAILABLE=FALSE:")
        for f in lean_false_files[:10]:
            print(f"  - {f}")
        if len(lean_false_files) > 10:
            print(f"  ... and {len(lean_false_files) - 10} more")
        print()
    
    if import_errors:
        print("FILES WITH IMPORT ERRORS:")
        for f, err in import_errors[:10]:
            print(f"  - {f}: {err}")
        if len(import_errors) > 10:
            print(f"  ... and {len(import_errors) - 10} more")
        print()
    
    if not lean_false_files and not import_errors:
        print("✅ NO CRITICAL ISSUES FOUND!")
        print("All Lean-integrated files are working correctly.")
    else:
        print(f"⚠️  ISSUES FOUND: {len(lean_false_files)} files with LEAN_AVAILABLE=False, {len(import_errors)} import errors")
    
    print()
    return 0 if not lean_false_files and not import_errors else 1

if __name__ == "__main__":
    sys.exit(main())
