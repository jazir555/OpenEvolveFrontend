#!/usr/bin/env python3
"""
Performance Test Runner

Runs all performance tests and generates a comprehensive report.
"""

import subprocess
import sys
from pathlib import Path
import time


def run_command(cmd: list, description: str) -> tuple:
    """Run a command and return (success, output)"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )
    elapsed = time.time() - start

    success = result.returncode == 0

    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Status: {'✓ PASS' if success else '✗ FAIL'}")

    return success, result.stdout + result.stderr


def main():
    """Run all performance tests"""
    results = {}
    test_dir = Path(__file__).parent

    print("\n" + "="*80)
    print(" PERFORMANCE TEST SUITE")
    print("="*80)

    # Test 1: Compression Performance
    success, output = run_command(
        [sys.executable, "-m", "pytest", "test_compression_performance.py", "-v", "--tb=short"],
        "Compression Performance Tests (Bug #16)"
    )
    results['compression'] = success

    # Test 2: Concurrent File Deletion
    success, output = run_command(
        [sys.executable, "-m", "pytest", "test_concurrent_files_performance.py", "-v", "--tb=short"],
        "Concurrent File Deletion Tests (Bug #15)"
    )
    results['concurrent_deletion'] = success

    # Test 3: Atomic Writes
    success, output = run_command(
        [sys.executable, "-m", "pytest", "test_atomic_writes_performance.py", "-v", "--tb=short"],
        "Atomic Write Tests (Bug #17)"
    )
    results['atomic_writes'] = success

    # Test 4: Pagination
    success, output = run_command(
        [sys.executable, "-m", "pytest", "test_pagination_performance.py", "-v", "--tb=short"],
        "Pagination Performance Tests (Bug #18)"
    )
    results['pagination'] = success

    # Test 5: Comprehensive Benchmarks
    success, output = run_command(
        [sys.executable, "-m", "pytest", "test_performance_benchmarks.py", "-v", "-s", "--tb=short"],
        "Comprehensive Performance Benchmarks"
    )
    results['benchmarks'] = success

    # Print Summary
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80)

    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:30s} {status}")

    total = len(results)
    passed = sum(1 for s in results.values() if s)
    failed = total - passed

    print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    if failed > 0:
        print("\n⚠ Some tests failed. Review the output above for details.")
        return 1
    else:
        print("\n✓ All performance tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
