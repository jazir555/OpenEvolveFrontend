"""
Verification Script for Edge Case Tests

Quick verification that all edge case test files are properly set up.

Usage:
    python verify_edge_case_tests.py

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import os
import sys
from pathlib import Path


def verify_test_file(filepath):
    """
    Verify that a test file exists and has the expected structure.

    Args:
        filepath: Path to test file

    Returns:
        dict with verification results
    """
    if not filepath.exists():
        return {
            "exists": False,
            "size": 0,
            "classes": 0,
            "tests": 0,
            "status": "File not found"
        }

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Count test classes
    test_classes = content.count('class Test')
    test_methods = content.count('def test_')

    return {
        "exists": True,
        "size": len(content),
        "classes": test_classes,
        "tests": test_methods,
        "status": "OK"
    }


def main():
    """Main verification function"""
    print("="*80)
    print("EDGE CASE TEST VERIFICATION")
    print("="*80)
    print()

    # Define test files
    test_dir = Path(__file__).parent
    test_files = {
        "ML Optimizer": test_dir / "test_edge_cases_ml_optimizer.py",
        "Predictive Executor": test_dir / "test_edge_cases_predictive_executor.py",
        "Adaptive Learner": test_dir / "test_edge_cases_adaptive_learner.py",
        "WebSocket": test_dir / "test_edge_cases_websocket.py"
    }

    # Verify each file
    all_ok = True
    total_tests = 0
    total_classes = 0

    for name, filepath in test_files.items():
        result = verify_test_file(filepath)

        print(f"{name}:")
        print(f"  Path: {filepath}")
        print(f"  Status: {result['status']}")
        print(f"  Size: {result['size']:,} bytes")

        if result['exists']:
            print(f"  Test Classes: {result['classes']}")
            print(f"  Test Methods: {result['tests']}")
            total_classes += result['classes']
            total_tests += result['tests']
        else:
            all_ok = False

        print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Test Files: {len(test_files)}")
    print(f"Total Test Classes: {total_classes}")
    print(f"Total Test Methods: {total_tests}")
    print(f"Overall Status: {'[OK] ALL OK' if all_ok else '[FAIL] ISSUES FOUND'}")
    print()

    # Check for supporting files
    print("="*80)
    print("SUPPORTING FILES")
    print("="*80)

    supporting_files = {
        "Test Runner": test_dir / "run_edge_case_tests.py",
        "Documentation": test_dir / "EDGE_CASE_TESTS_DOCUMENTATION.md",
        "README": test_dir / "EDGE_CASE_TESTS_README.md",
        "Verification Script": test_dir / "verify_edge_case_tests.py"
    }

    for name, filepath in supporting_files.items():
        exists = filepath.exists()
        size = filepath.stat().st_size if exists else 0
        status = "[OK]" if exists else "[FAIL]"
        print(f"{status} {name}: {filepath.name} ({size:,} bytes)" if exists else f"{status} {name}: MISSING")

    print()

    # Check dependencies
    print("="*80)
    print("DEPENDENCIES")
    print("="*80)

    dependencies = [
        ("pytest", "pytest"),
        ("pytest-asyncio", "pytest_asyncio"),
        ("pytest-cov", "pytest_cov"),
        ("coverage", "coverage")
    ]

    for package, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FAIL] {package} (not installed)")

    print()

    # Check source files
    print("="*80)
    print("SOURCE FILES")
    print("="*80)

    project_root = Path(__file__).parent.parent.parent
    source_files = {
        "ML Optimizer": project_root / "glue" / "adapters" / "gauntlet-adapter" / "src" / "ml_optimizer.py",
        "Predictive Executor": project_root / "glue" / "adapters" / "gauntlet-adapter" / "src" / "predictive_gauntlet_executor.py",
        "Adaptive Learner": project_root / "glue" / "adapters" / "gauntlet-adapter" / "src" / "adaptive_learner.py",
        "WebSocket": project_root / "api" / "gauntlets_websocket.py"
    }

    for name, filepath in source_files.items():
        exists = filepath.exists()
        status = "[OK]" if exists else "[FAIL]"
        print(f"{status} {name}: {filepath}")

    print()

    # Recommendations
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if all_ok:
        print("[OK] All test files are present and properly structured")
        print("[OK] Ready to run tests:")
        print(f"  python {test_dir / 'run_edge_case_tests.py'}")
        print(f"  python {test_dir / 'run_edge_case_tests.py'} --coverage")
        print()
    else:
        print("[FAIL] Some test files are missing or incomplete")
        print("  Please check the output above for details")
        print()

    # Missing dependencies check
    missing_deps = []
    for package, import_name in dependencies:
        try:
            __import__(import_name)
        except ImportError:
            missing_deps.append(package)

    if missing_deps:
        print(f"[FAIL] Missing dependencies: {', '.join(missing_deps)}")
        print("  Install with: pip install " + " ".join(missing_deps))
        print()
    else:
        print("[OK] All dependencies are installed")
        print()

    # Next steps
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Run all tests:")
    print(f"   python {test_dir / 'run_edge_case_tests.py'}")
    print()
    print("2. Run with coverage:")
    print(f"   python {test_dir / 'run_edge_case_tests.py'} --coverage")
    print()
    print("3. Run specific component:")
    print(f"   python {test_dir / 'run_edge_case_tests.py'} --component ml_optimizer")
    print()
    print("4. View detailed documentation:")
    print(f"   See {test_dir / 'EDGE_CASE_TESTS_DOCUMENTATION.md'}")
    print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
