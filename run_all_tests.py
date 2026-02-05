#!/usr/bin/env python3
"""
Test Runner - TRUE 100% Testing Framework
=========================================

Runs all fixed test suites and generates a comprehensive report.

Usage:
    python run_all_tests.py
    python run_all_tests.py --verbose
    python run_all_tests.py --quick

Test Suites:
    - test_compliance.py: 73 security compliance tests (OWASP, NIST, ISO 27001, GDPR, PCI DSS)
    - test_all_bugs_fixed.py: 136 comprehensive bug fix verification tests
    - test_sovereign_gauntlets.py: 17 gauntlet system tests
    - test_input_validation_comprehensive.py: 25 input validation tests

Total: 251+ tests
"""

import subprocess
import sys
import argparse
from datetime import datetime

TEST_SUITES = [
    ("test_compliance.py", "Security Compliance Tests (OWASP, NIST, ISO 27001, GDPR, PCI DSS)"),
    ("test_all_bugs_fixed.py", "Comprehensive Bug Fix Verification (136 tests)"),
    ("test_sovereign_gauntlets.py", "Sovereign Gauntlet System Tests"),
    ("test_input_validation_comprehensive.py", "Input Validation & Security Tests"),
]


def run_test_suite(test_file, verbose=False):
    """Run a single test suite and return results."""
    cmd = ["python", "-m", "pytest", test_file, "-v" if verbose else "-q", "--tb=no"]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Parse results
        output = result.stdout + result.stderr
        
        # Extract counts
        passed = 0
        failed = 0
        errors = 0
        
        for line in output.split('\n'):
            if 'passed' in line:
                # Extract number passed
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'passed' in part:
                            passed = int(parts[i-1])
                        if 'failed' in part:
                            failed = int(parts[i-1])
                        if 'error' in part:
                            errors = int(parts[i-1])
                except:
                    pass
        
        return {
            'file': test_file,
            'returncode': result.returncode,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'output': output
        }
    except subprocess.TimeoutExpired:
        return {
            'file': test_file,
            'returncode': -1,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'output': 'TIMEOUT'
        }
    except Exception as e:
        return {
            'file': test_file,
            'returncode': -1,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'output': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Run all test suites')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick mode (collect only)')
    args = parser.parse_args()

    print("=" * 80)
    print("TESTING FRAMEWORK - TRUE 100% FIX VERIFICATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_passed = 0
    total_failed = 0
    total_errors = 0
    results = []

    for test_file, description in TEST_SUITES:
        print(f"Running: {description}")
        print(f"  File: {test_file}")
        
        if args.quick:
            # Just collect tests
            cmd = ["python", "-m", "pytest", test_file, "--collect-only", "-q"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            # Count collected
            collected = result.stdout.count("<Function")
            print(f"  Collected: {collected} tests")
            total_passed += collected
        else:
            result = run_test_suite(test_file, args.verbose)
            results.append(result)
            
            total_passed += result['passed']
            total_failed += result['failed']
            total_errors += result['errors']
            
            status = "PASS" if result['returncode'] == 0 else "FAIL"
            print(f"  Passed: {result['passed']}")
            if result['failed'] > 0:
                print(f"  Failed: {result['failed']}")
            if result['errors'] > 0:
                print(f"  Errors: {result['errors']}")
            print(f"  Status: {status}")
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if args.quick:
        print(f"Total Tests Collected: {total_passed}")
    else:
        print(f"Total Passed: {total_passed}")
        print(f"Total Failed: {total_failed}")
        print(f"Total Errors: {total_errors}")
        print(f"Success Rate: {total_passed/(total_passed+total_failed+total_errors)*100:.1f}%")
    
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Return appropriate exit code
    if args.quick:
        return 0 if total_passed > 0 else 1
    else:
        return 0 if total_failed == 0 and total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
