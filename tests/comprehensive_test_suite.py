"""
Comprehensive Test Suite for Sovereign-Grade Problem Decomposition System
Complete integration of all test suites
"""

import unittest
import sys
import os
from datetime import datetime
import time
import json

# Add the project root to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all test modules
from additional_unit_tests import (
    TestDataModels, TestAnalyzer, TestDecompositionEngine, 
    TestPersistence, TestInputValidation, TestAuthSystem,
    TestSolutionOrchestrator, TestTeamCoordination
)
from integration_and_performance_tests import (
    TestIntegrationWorkflows, TestPerformanceAndStress,
    TestErrorHandling, TestSecurity
)
from gauntlet_tests import TestGauntletSystem, TestGauntletIntegration


def create_comprehensive_test_suite():
    """Create a comprehensive test suite combining all tests"""
    suite = unittest.TestSuite()
    
    # Add all test cases from each module
    # Basic unit tests
    suite.addTest(unittest.makeSuite(TestDataModels))
    suite.addTest(unittest.makeSuite(TestAnalyzer))
    suite.addTest(unittest.makeSuite(TestDecompositionEngine))
    suite.addTest(unittest.makeSuite(TestPersistence))
    suite.addTest(unittest.makeSuite(TestInputValidation))
    suite.addTest(unittest.makeSuite(TestAuthSystem))
    suite.addTest(unittest.makeSuite(TestSolutionOrchestrator))
    suite.addTest(unittest.makeSuite(TestTeamCoordination))
    
    # Integration and performance tests
    suite.addTest(unittest.makeSuite(TestIntegrationWorkflows))
    suite.addTest(unittest.makeSuite(TestPerformanceAndStress))
    suite.addTest(unittest.makeSuite(TestErrorHandling))
    suite.addTest(unittest.makeSuite(TestSecurity))
    
    # Gauntlet system tests
    suite.addTest(unittest.makeSuite(TestGauntletSystem))
    suite.addTest(unittest.makeSuite(TestGauntletIntegration))
    
    return suite


def run_comprehensive_tests():
    """Run the comprehensive test suite"""
    print("="*80)
    print("COMPREHENSIVE TEST SUITE FOR SOVEREIGN-GRADE SYSTEM")
    print("="*80)
    print(f"Test execution started at: {datetime.now().isoformat()}")
    print("-"*80)
    
    # Create the comprehensive test suite
    suite = create_comprehensive_test_suite()
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        buffer=True  # Capture stdout/stderr during tests
    )
    
    start_time = time.time()
    result = runner.run(suite)
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST REPORT")
    print("="*80)
    print(f"Test execution completed at: {datetime.now().isoformat()}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("-"*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Expected failures: {len(result.expectedFailures)}")
    print(f"Unexpected successes: {len(result.unexpectedSuccesses)}")
    print(f"Skipped: {len(result.skipped)}")
    print("-"*80)
    
    # Calculate success rate
    total_tests = result.testsRun
    failed_tests = len(result.failures) + len(result.errors)
    successful_tests = total_tests - failed_tests
    
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 100.0
    print(f"Success rate: {success_rate:.2f}% ({successful_tests}/{total_tests})")
    
    # Performance indicators
    tests_per_second = total_tests / total_time if total_time > 0 else float('inf')
    print(f"Tests per second: {tests_per_second:.2f}")
    
    print("="*80)
    
    # Print failures and errors if any
    if result.failures or result.errors:
        print("\nFAILURE DETAILS:")
        for test_case, traceback in result.failures:
            print(f"\nFAILED: {test_case}")
            print(traceback)
        
        print("\nERROR DETAILS:")
        for test_case, traceback in result.errors:
            print(f"\nERROR: {test_case}")
            print(traceback)
    else:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    
    print("="*80)
    
    return result


def run_minimal_smoke_tests():
    """Run minimal smoke tests to verify basic functionality"""
    print("Running minimal smoke tests...")
    
    smoke_suite = unittest.TestSuite()
    
    # Add only critical smoke tests
    smoke_suite.addTest(TestDataModels('test_generate_id_uniqueness'))
    smoke_suite.addTest(TestDataModels('test_complexity_score_validation'))
    smoke_suite.addTest(TestAuthSystem('test_hash_password_and_verify'))
    smoke_suite.addTest(TestPersistence('test_create_and_retrieve_problem'))
    
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(smoke_suite)
    
    print(f"\nSmoke Tests - Passed: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")
    return result


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive test suite for Sovereign-Grade System')
    parser.add_argument('--smoke', action='store_true', help='Run minimal smoke tests only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    
    args = parser.parse_args()
    
    if args.smoke:
        result = run_minimal_smoke_tests()
    else:
        result = run_comprehensive_tests()
    
    # Exit with appropriate code
    total_failures = len(result.failures) + len(result.errors)
    sys.exit(1 if total_failures > 0 else 0)


if __name__ == "__main__":
    main()