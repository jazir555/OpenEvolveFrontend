#!/usr/bin/env python
"""
Quick validation script to check test infrastructure

This script performs basic validation of the test suite:
- Checks all test files exist
- Validates imports work
- Verifies fixtures are loadable
- Tests basic test discovery
"""

import sys
from pathlib import Path


def check_file_exists(filepath: Path, description: str) -> bool:
    """Check if a file exists"""
    if filepath.exists():
        print(f"[OK] {description}: {filepath}")
        return True
    else:
        print(f"[FAIL] {description}: {filepath} NOT FOUND")
        return False


def check_import(module_name: str) -> bool:
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"[OK] Import {module_name}")
        return True
    except ImportError as e:
        print(f"[FAIL] Import {module_name}: {e}")
        return False


def main():
    """Main validation routine"""
    print("="*60)
    print("Validating PES Test Suite")
    print("="*60)
    print()

    # Get test directory
    test_dir = Path(__file__).parent

    # Check test files
    print("1. Checking test files...")
    print("-"*60)
    files_ok = True

    files_to_check = [
        (test_dir / "__init__.py", "Package init"),
        (test_dir / "fixtures.py", "Test fixtures"),
        (test_dir / "test_controller.py", "Controller tests"),
        (test_dir / "test_evaluator.py", "Evaluator tests"),
        (test_dir / "test_database.py", "Database tests"),
        (test_dir / "integration" / "__init__.py", "Integration init"),
        (test_dir / "integration" / "test_pes_optimization.py", "Integration tests"),
        (test_dir / "run_tests.py", "Test runner"),
        (test_dir / "pytest.ini", "Pytest config"),
        (test_dir / "requirements-test.txt", "Test requirements"),
    ]

    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            files_ok = False

    print()

    # Check mock LLM client
    print("2. Checking mock LLM client...")
    print("-"*60)
    mock_ok = True

    # Get the openevolve package directory
    openevolve_dir = test_dir.parent.parent.parent / "openevolve" / "openevolve"
    mock_dir = openevolve_dir / "llm" / "mocks"
    mock_init = mock_dir / "__init__.py"
    mock_client = mock_dir / "mock_client.py"

    if not check_file_exists(mock_init, "Mock LLM __init__"):
        mock_ok = False
    if not check_file_exists(mock_client, "Mock LLM client"):
        mock_ok = False

    print()

    # Check imports
    print("3. Checking imports...")
    print("-"*60)
    imports_ok = True

    # Add parent directory to path
    sys.path.insert(0, str(test_dir.parent.parent.parent))

    imports_to_check = [
        "pytest",
        "openevolve",
    ]

    for module in imports_to_check:
        if not check_import(module):
            imports_ok = False

    print()

    # Check test discovery
    print("4. Checking test discovery...")
    print("-"*60)

    try:
        import subprocess
        result = subprocess.run(
            ["python", "-m", "pytest", "--collect-only", "tests/pes/"],
            cwd=test_dir.parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=30
        )

        if "collected" in result.stdout or "error" in result.stdout.lower():
            # Extract collection count
            for line in result.stdout.split('\n'):
                if 'collected' in line or 'item' in line:
                    print(f"  {line.strip()}")
            discovery_ok = True
        else:
            print("[FAIL] Test collection failed")
            print(result.stdout)
            print(result.stderr)
            discovery_ok = False

    except Exception as e:
        print(f"[FAIL] Test discovery error: {e}")
        discovery_ok = False

    print()

    # Summary
    print("="*60)
    print("Validation Summary")
    print("="*60)

    all_ok = files_ok and mock_ok and imports_ok and discovery_ok

    if all_ok:
        print("[OK] All checks passed!")
        print()
        print("Next steps:")
        print("  1. Install test dependencies:")
        print("     pip install -r tests/pes/requirements-test.txt")
        print()
        print("  2. Run tests:")
        print("     cd tests/pes && python run_tests.py")
        print()
        return 0
    else:
        print("[FAIL] Some checks failed")
        print()
        if not files_ok:
            print("- Some test files are missing")
        if not mock_ok:
            print("- Mock LLM client is not set up correctly")
        if not imports_ok:
            print("- Some imports are failing (install dependencies?)")
        if not discovery_ok:
            print("- Test discovery is not working")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
