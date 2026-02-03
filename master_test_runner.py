#!/usr/bin/env python3
"""
Master Test Runner for Sovereign-Grade Problem Decomposition System
Combines all test suites and provides comprehensive reporting
"""


import unittest
import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
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
from extended_unit_tests import (
    TestEdgeCases, TestDomainSpecificDecomposition,
    TestIntegrationScenarios, TestSecurityScenarios,
    TestPerformanceBoundaries
)


class TestResultReporter:
    """Generates comprehensive test reports"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results = {}
    
    def start_test_run(self):
        """Record test run start time"""
        self.start_time = datetime.now()
        print("="*80)
        print("SOVEREIGN-GRADE SYSTEM - COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Test run started at: {self.start_time.isoformat()}")
    
    def end_test_run(self, test_result: unittest.TestResult):
        """Record test run end time and generate report"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"Test run ended at: {self.end_time.isoformat()}")
        print(f"Total duration: {duration:.2f} seconds")
        print("-"*80)
        print(f"Tests run: {test_result.testsRun}")
        print(f"Failures: {len(test_result.failures)}")
        print(f"Errors: {len(test_result.errors)}")
        print(f"Success rate: {self._calculate_success_rate(test_result):.2f}%")
        
        if test_result.failures or test_result.errors:
            print("\nFAILURE DETAILS:")
            for test, traceback in test_result.failures:
                print(f"\nFAILED: {test}")
                print(traceback)
            
            print("\nERROR DETAILS:")
            for test, traceback in test_result.errors:
                print(f"\nERROR: {test}")
                print(traceback)
        else:
            print("\n🎉 ALL TESTS PASSED! 🎉")
        
        print("="*80)
    
    def _calculate_success_rate(self, result: unittest.TestResult) -> float:
        """Calculate success rate percentage"""
        if result.testsRun == 0:
            return 0.0
        successful = result.testsRun - len(result.failures) - len(result.errors)
        return (successful / result.testsRun) * 100


def create_comprehensive_test_suite():
    """Create the complete test suite combining all tests"""
    suite = unittest.TestSuite()
    
    # Add all test cases from different modules
    # Basic unit tests
    suite.addTest(unittest.makeSuite(TestDataModels))
    suite.addTest(unittest.makeSuite(TestAnalyzer))
    suite.addTest(unittest.makeSuite(TestDecompositionEngine))
    suite.addTest(unittest.makeSuite(TestPersistence))
    suite.addTest(unittest.makeSuite(TestInputValidation))
    suite.addTest(unittest.makeSuite(TestAuthSystem))
    suite.addTest(unittest.makeSuite(TestSolutionOrchestrator))
    suite.addTest(unittest.makeSuite(TestTeamCoordination))
    
    # Extended unit tests
    suite.addTest(unittest.makeSuite(TestEdgeCases))
    suite.addTest(unittest.makeSuite(TestDomainSpecificDecomposition))
    suite.addTest(unittest.makeSuite(TestIntegrationScenarios))
    suite.addTest(unittest.makeSuite(TestSecurityScenarios))
    suite.addTest(unittest.makeSuite(TestPerformanceBoundaries))
    
    # Integration and performance tests
    suite.addTest(unittest.makeSuite(TestIntegrationWorkflows))
    suite.addTest(unittest.makeSuite(TestPerformanceAndStress))
    suite.addTest(unittest.makeSuite(TestErrorHandling))
    suite.addTest(unittest.makeSuite(TestSecurity))
    
    # Gauntlet system tests
    suite.addTest(unittest.makeSuite(TestGauntletSystem))
    suite.addTest(unittest.makeSuite(TestGauntletIntegration))
    
    return suite


def run_comprehensive_test_suite():
    """Run the complete comprehensive test suite"""
    reporter = TestResultReporter()
    reporter.start_test_run()
    
    # Create the test suite
    suite = create_comprehensive_test_suite()
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        buffer=True  # Capture stdout/stderr during tests
    )
    
    # Execute tests
    result = runner.run(suite)
    
    # Generate final report
    reporter.end_test_run(result)
    
    return result


def run_smoke_tests():
    """Run minimal smoke tests for quick validation"""
    print("Running smoke tests...")
    
    smoke_suite = unittest.TestSuite()
    
    # Add critical smoke tests
    smoke_suite.addTest(TestDataModels('test_generate_id_uniqueness'))
    smoke_suite.addTest(TestDataModels('test_complexity_score_validation'))
    smoke_suite.addTest(TestDataModels('test_constraint_validation'))
    smoke_suite.addTest(TestAuthSystem('test_hash_password_and_verify'))
    smoke_suite.addTest(TestPersistence('test_create_and_retrieve_problem'))
    smoke_suite.addTest(TestAnalyzer('test_analyze_problem'))
    
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(smoke_suite)
    
    print(f"\nSmoke Tests - Passed: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")
    return result


def run_performance_benchmarks():
    """Run performance benchmark tests"""
    print("\nRunning performance benchmarks...")
    
    performance_suite = unittest.TestSuite()
    
    # Add performance-related tests
    performance_suite.addTest(TestPerformanceAndStress('test_llm_cache_performance'))
    performance_suite.addTest(TestPerformanceAndStress('test_parallel_processing_performance'))
    performance_suite.addTest(TestPerformanceAndStress('test_concurrent_user_simulation'))
    performance_suite.addTest(TestPerformanceBoundaries('test_memory_efficiency'))
    performance_suite.addTest(TestPerformanceBoundaries('test_concurrent_operations'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(performance_suite)
    
    print(f"\nPerformance Benchmarks - Passed: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")
    return result


def main():
    """Main entry point for test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive test suite for Sovereign-Grade System')
    parser.add_argument('--suite', choices=['all', 'smoke', 'extended', 'performance'], 
                       default='all', help='Test suite to run')
    parser.add_argument('--output-format', choices=['console', 'json', 'xml'], 
                       default='console', help='Output format for results')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Quiet output')
    
    args = parser.parse_args()
    
    if args.suite == 'smoke':
        result = run_smoke_tests()
    elif args.suite == 'performance':
        result = run_performance_benchmarks()
    elif args.suite == 'extended':
        # Run just extended tests
        extended_suite = unittest.TestSuite()
        extended_suite.addTest(unittest.makeSuite(TestEdgeCases))
        extended_suite.addTest(unittest.makeSuite(TestDomainSpecificDecomposition))
        extended_suite.addTest(unittest.makeSuite(TestIntegrationScenarios))
        extended_suite.addTest(unittest.makeSuite(TestSecurityScenarios))
        extended_suite.addTest(unittest.makeSuite(TestPerformanceBoundaries))
        
        runner = unittest.TextTestRunner(verbosity=2 if not args.quiet else 0)
        result = runner.run(extended_suite)
    else:  # all
        result = run_comprehensive_test_suite()
    
    # Exit with appropriate code based on test results
    total_failures = len(result.failures) + len(result.errors)
    exit_code = 1 if total_failures > 0 else 0
    
    if not args.quiet:
        print(f"\nFinal Result: {'PASSED' if exit_code == 0 else 'FAILED'} ({result.testsRun - total_failures}/{result.testsRun} passed)")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()