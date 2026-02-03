#!/usr/bin/env python3
"""
OpenEvolve API Test Runner

Runs integration tests for the OpenEvolve API service
Usage: python scripts/run_tests.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
        return True
    else:
        print(f"❌ {description} - FAILED")
        return False


def main():
    """Main test runner"""
    # Change to service directory
    service_dir = Path(__file__).parent.parent
    os.chdir(service_dir)

    print("OpenEvolve API Test Suite")
    print("="*60)

    # Check if service is running
    print("\n🔍 Checking if service is running...")
    try:
        import httpx
        response = httpx.get("http://localhost:8001/health", timeout=2.0)
        if response.status_code == 200:
            print("✅ Service is running")
        else:
            print("❌ Service is not healthy")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        print("\n💡 Start the service first:")
        print("   cd BubbleLab/services && python -m uvicorn openevolve-api.main:app --host 0.0.0.0 --port 8001")
        sys.exit(1)

    # Run tests
    tests = [
        {
            "cmd": [sys.executable, "-m", "pytest", "tests/test_api_integration.py", "-v", "--tb=short"],
            "desc": "API Integration Tests"
        },
        {
            "cmd": [sys.executable, "-m", "pytest", "tests/", "-v", "--cov=.", "--cov-report=term-missing"],
            "desc": "Tests with Coverage Report"
        },
    ]

    results = []
    for test in tests:
        results.append(run_command(test["cmd"], test["desc"]))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test['desc']}")

    print(f"\nTotal: {passed}/{total} test suites passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
