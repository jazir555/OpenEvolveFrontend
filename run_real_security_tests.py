"""
Real Security Tests Runner
Runs all REAL security tests and generates coverage report.

Usage:
    python run_real_security_tests.py
"""

import subprocess
import sys
import os
from datetime import datetime


TEST_FILES = [
    # NEW REAL SECURITY TESTS
    "real_sql_injection_tests.py",
    "real_security_headers_tests.py", 
    "real_xss_prevention_tests.py",
    "real_rate_limiting_tests.py",
    "real_audit_logging_tests.py",
    # FIXED EXISTING TESTS
    "test_input_validation_fixed.py",
]


def run_tests():
    """Run all real security tests."""
    print("=" * 80)
    print("REAL SECURITY TESTS - CRITICAL GAP FIXES")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    results = {}
    
    for test_file in TEST_FILES:
        if not os.path.exists(test_file):
            print(f"⚠️  SKIPPED: {test_file} (file not found)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running: {test_file}")
        print('='*60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--tb=short",
            "-x",  # Stop on first failure
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per file
            )
            
            results[test_file] = {
                "returncode": result.returncode,
                "passed": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            
            # Print output
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            if result.returncode == 0:
                print(f"✅ PASSED: {test_file}")
            else:
                print(f"❌ FAILED: {test_file}")
                
        except subprocess.TimeoutExpired:
            results[test_file] = {
                "returncode": -1,
                "passed": False,
                "stdout": "",
                "stderr": "TIMEOUT",
            }
            print(f"⏱️  TIMEOUT: {test_file}")
        except Exception as e:
            results[test_file] = {
                "returncode": -1,
                "passed": False,
                "stdout": "",
                "stderr": str(e),
            }
            print(f"💥 ERROR: {test_file} - {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r["passed"])
    failed = sum(1 for r in results.values() if not r["passed"])
    
    print(f"Total test files: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()
    
    for test_file, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{status}: {test_file}")
    
    print()
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 80)
    
    return failed == 0


def generate_coverage_report():
    """Generate coverage report for security tests."""
    print("\n" + "=" * 80)
    print("GENERATING COVERAGE REPORT")
    print("=" * 80)
    
    cmd = [
        sys.executable, "-m", "pytest",
    ] + TEST_FILES + [
        "--cov=input_validation",
        "--cov=security_framework",
        "--cov-report=term-missing",
        "--cov-report=html:security_coverage_html",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Coverage generation failed: {e}")


if __name__ == "__main__":
    success = run_tests()
    
    # Optionally generate coverage
    if "--coverage" in sys.argv:
        generate_coverage_report()
    
    sys.exit(0 if success else 1)
