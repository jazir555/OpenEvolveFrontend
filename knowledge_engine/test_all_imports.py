#!/usr/bin/env python3
"""
Comprehensive import test for knowledge_engine.
Tests all Python modules for import errors.
"""
import sys
import traceback
from pathlib import Path
import importlib.util

# Add knowledge_engine to path
sys.path.insert(0, str(Path(__file__).parent))

def test_import(module_path):
    """Test if a module can be imported."""
    try:
        # Convert file path to module path
        rel_path = module_path.relative_to(Path(__file__).parent)
        module_name = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return False, f"Cannot create spec for {module_name}"

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return True, module_name
    except Exception as e:
        return False, f"{module_path.name}: {str(e)}\n{traceback.format_exc()}"

def main():
    """Test all Python files."""
    ke_dir = Path(__file__).parent
    py_files = list(ke_dir.rglob('*.py'))

    # Exclude __pycache__ and test files
    py_files = [f for f in py_files
                if '__pycache__' not in str(f)
                and f.name != 'test_all_imports.py']

    print(f"Testing {len(py_files)} Python files...\n")

    failures = []
    successes = []

    for py_file in sorted(py_files):
        success, result = test_import(py_file)
        if success:
            successes.append(result)
            print(f"[OK] {result}")
        else:
            failures.append((py_file, result))
            print(f"[FAIL] {py_file}")
            print(f"  {result[:200]}...")  # First 200 chars of error

    print(f"\n{'='*60}")
    print(f"Results: {len(successes)} passed, {len(failures)} failed")

    if failures:
        print(f"\n{'='*60}")
        print("FAILED IMPORTS:")
        print('='*60)
        for py_file, error in failures:
            print(f"\n{py_file}:")
            print(error)

    return len(failures)

if __name__ == '__main__':
    sys.exit(main())
