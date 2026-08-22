#!/usr/bin/env python3
"""
Quick Verification Script

Verifies that the OpenEvolve API service and frontend client are properly set up.
Run this before starting full integration testing.

Usage: python scripts/verify_setup.py
"""

import sys
import os
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"[OK] {description}")
        return True
    else:
        print(f"[FAIL] {description} - NOT FOUND")
        return False


def check_import(module_name: str) -> bool:
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"[OK] {module_name} - available")
        return True
    except ImportError:
        print(f"[FAIL] {module_name} - NOT AVAILABLE")
        return False


def main():
    """Run verification checks"""
    print("="*60)
    print("OpenEvolve Integration - Quick Verification")
    print("="*60)

    service_dir = Path(__file__).parent.parent
    os.chdir(service_dir)

    results = []

    # Check backend files
    print("\n📦 Backend Files:")
    print("-" * 60)

    backend_files = [
        ("main.py", "FastAPI application entry point"),
        ("models/__init__.py", "Pydantic models"),
        ("core/evolution.py", "Evolution engine"),
        ("core/adversarial.py", "Adversarial engine"),
        ("core/sovereign.py", "Sovereign engine"),
        ("api/workflows.py", "Workflows API"),
        ("api/execution.py", "Execution API"),
        ("api/teams.py", "Teams API"),
        ("api/gauntlets.py", "Gauntlets API"),
        ("services/execution_service.py", "Execution service"),
    ]

    for filepath, description in backend_files:
        results.append(check_file_exists(filepath, description))

    # Check test files
    print("\n🧪 Test Files:")
    print("-" * 60)

    test_files = [
        ("tests/__init__.py", "Test package"),
        ("tests/conftest.py", "Pytest configuration"),
        ("tests/test_api_integration.py", "Integration tests"),
        ("scripts/run_tests.py", "Test runner script"),
        ("TESTING_GUIDE.md", "Testing documentation"),
    ]

    for filepath, description in test_files:
        results.append(check_file_exists(filepath, description))

    # Check documentation
    print("\n📚 Documentation:")
    print("-" * 60)

    docs_files = [
        ("README.md", "Service overview"),
        ("API_DOCUMENTATION.md", "API reference"),
        ("QUICK_REFERENCE.md", "Quick start guide"),
        ("TESTING_GUIDE.md", "Testing guide"),
    ]

    for filepath, description in docs_files:
        results.append(check_file_exists(filepath, description))

    # Check Python dependencies
    print("\n🐍 Python Dependencies:")
    print("-" * 60)

    dependencies = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "httpx",
        "structlog",
        "pytest",
        "pytest_asyncio",
    ]

    for dep in dependencies:
        results.append(check_import(dep))

    # Check frontend files
    print("\n⚛️  Frontend Files:")
    print("-" * 60)

    frontend_dir = service_dir.parent.parent / "apps" / "bubble-studio"

    frontend_files = [
        ("src/services/openevolveApi.ts", "OpenEvolve API client", frontend_dir),
        ("src/types/openevolve.ts", "TypeScript type definitions", frontend_dir),
        ("src/services/__tests__/openevolveApi.test.ts", "Frontend tests", frontend_dir),
    ]

    for filepath, description, base_dir in frontend_files:
        full_path = base_dir / filepath
        results.append(check_file_exists(str(full_path), description))

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100

    print(f"\nPassed: {passed}/{total} ({percentage:.1f}%)")

    if passed == total:
        print("\n🎉 All checks passed! Ready for integration testing.")
        print("\nNext steps:")
        print("1. Start the service: make dev")
        print("2. Run tests: python scripts/run_tests.py")
        return 0
    else:
        print(f"\n[WARN]  {total - passed} check(s) failed.")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
