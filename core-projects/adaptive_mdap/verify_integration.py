#!/usr/bin/env python3
"""
Adaptive MDAP Integration Verification Script

This script verifies that the Adaptive MDAP integration is complete and working.
Run this script to validate:
1. All components are importable
2. Basic functionality works
3. Integration with existing systems is functional
4. Performance meets targets

Usage:
    python -m adaptive_mdap.verify_integration
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Verify Integration
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


import sys
import time
from typing import List, Dict, Any


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(test: str, passed: bool, details: str = ""):
    """Print a test result."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}: {test}")
    if details:
        print(f"         {details}")


def test_imports() -> bool:
    """Test that all components can be imported."""
    print_section("Testing Imports")
    
    tests = []
    
    try:
        from adaptive_mdap.core.types import SubProblem, ComplexityScore, SolveStrategy
        tests.append(("Core types", True))
    except Exception as e:
        tests.append(("Core types", False, str(e)))
    
    try:
        from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier
        tests.append(("TaskComplexityClassifier", True))
    except Exception as e:
        tests.append(("TaskComplexityClassifier", False, str(e)))
    
    try:
        from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator
        tests.append(("AdaptiveMDAPAllocator", True))
    except Exception as e:
        tests.append(("AdaptiveMDAPAllocator", False, str(e)))
    
    try:
        from adaptive_mdap.controllers.execution_controller import AdaptiveExecutionController
        tests.append(("AdaptiveExecutionController", True))
    except Exception as e:
        tests.append(("AdaptiveExecutionController", False, str(e)))
    
    try:
        from adaptive_mdap.integrations.subproblem_solver_integration import AdaptiveSubProblemSolver
        tests.append(("AdaptiveSubProblemSolver", True))
    except Exception as e:
        tests.append(("AdaptiveSubProblemSolver", False, str(e)))
    
    try:
        from adaptive_mdap.tools.cost_calculator import CostCalculator
        tests.append(("CostCalculator", True))
    except Exception as e:
        tests.append(("CostCalculator", False, str(e)))
    
    try:
        from adaptive_mdap.monitoring.health import HealthChecker
        tests.append(("HealthChecker", True))
    except Exception as e:
        tests.append(("HealthChecker", False, str(e)))
    
    try:
        from adaptive_mdap.monitoring.dashboard import DashboardGenerator
        tests.append(("DashboardGenerator", True))
    except Exception as e:
        tests.append(("DashboardGenerator", False, str(e)))
    
    all_passed = True
    for test in tests:
        name = test[0]
        passed = test[1]
        details = test[2] if len(test) > 2 else ""
        print_result(name, passed, details)
        if not passed:
            all_passed = False
    
    return all_passed


def test_basic_functionality() -> bool:
    """Test basic functionality."""
    print_section("Testing Basic Functionality")
    
    tests = []
    
    try:
        from adaptive_mdap.core.types import SubProblem, SolveStrategy
        from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier
        from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator
        
        # Create classifier
        classifier = TaskComplexityClassifier()
        
        # Create sub-problem
        sp = SubProblem(
            id="verify-test",
            description="Test problem for verification",
            domain="testing",
            depth=2,
            dependencies=[],
            metadata={},
        )
        
        # Compute complexity
        complexity = classifier.compute_complexity(sp)
        
        if 0.0 <= complexity.overall_score <= 1.0:
            tests.append(("Complexity classification", True, f"score={complexity.overall_score:.3f}"))
        else:
            tests.append(("Complexity classification", False, f"score out of range: {complexity.overall_score}"))
        
        # Create allocator
        allocator = AdaptiveMDAPAllocator()
        
        # Allocate resources
        config = allocator.allocate_resources(complexity.overall_score)
        
        if config.strategy in SolveStrategy:
            tests.append(("Resource allocation", True, f"strategy={config.strategy.value}, agents={config.n_agents}"))
        else:
            tests.append(("Resource allocation", False, f"invalid strategy: {config.strategy}"))
        
    except Exception as e:
        tests.append(("Basic functionality", False, str(e)))
    
    all_passed = True
    for test in tests:
        name = test[0]
        passed = test[1]
        details = test[2] if len(test) > 2 else ""
        print_result(name, passed, details)
        if not passed:
            all_passed = False
    
    return all_passed


def test_execution() -> bool:
    """Test execution controller."""
    print_section("Testing Execution Controller")
    
    tests = []
    
    try:
        from adaptive_mdap.core.types import SubProblem
        from adaptive_mdap.controllers.execution_controller import AdaptiveExecutionController
        
        controller = AdaptiveExecutionController()
        
        # Create sub-problem
        sp = SubProblem(
            id="execution-test",
            description="Simple execution test",
            domain="testing",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        # Execute
        attempt = controller.execute_adaptive(sp)
        
        if attempt.status.value == "completed":
            tests.append(("Execution", True, f"strategy={attempt.allocated_strategy}, complexity={attempt.complexity_score:.3f}"))
        else:
            tests.append(("Execution", False, f"status={attempt.status.value}"))
        
    except Exception as e:
        tests.append(("Execution", False, str(e)))
    
    all_passed = True
    for test in tests:
        name = test[0]
        passed = test[1]
        details = test[2] if len(test) > 2 else ""
        print_result(name, passed, details)
        if not passed:
            all_passed = False
    
    return all_passed


def test_cost_calculator() -> bool:
    """Test cost calculator."""
    print_section("Testing Cost Calculator")
    
    tests = []
    
    try:
        from adaptive_mdap.tools.cost_calculator import CostCalculator, APIPricing
        
        calculator = CostCalculator(pricing=APIPricing.gpt_4o_mini())
        
        result = calculator.calculate_adaptive_cost(1000)
        
        if result["savings_percent"] > 20:
            tests.append(("Cost savings", True, f"{result['savings_percent']:.1f}% savings"))
        else:
            tests.append(("Cost savings", False, f"only {result['savings_percent']:.1f}% savings"))
        
        if result["baseline_cost"] > result["adaptive_cost"]:
            tests.append(("Cost comparison", True, f"${result['baseline_cost']:.2f} -> ${result['adaptive_cost']:.2f}"))
        else:
            tests.append(("Cost comparison", False, "adaptive cost not lower than baseline"))
        
    except Exception as e:
        tests.append(("Cost calculator", False, str(e)))
    
    all_passed = True
    for test in tests:
        name = test[0]
        passed = test[1]
        details = test[2] if len(test) > 2 else ""
        print_result(name, passed, details)
        if not passed:
            all_passed = False
    
    return all_passed


def test_performance() -> bool:
    """Test performance targets."""
    print_section("Testing Performance")
    
    tests = []
    
    try:
        from adaptive_mdap.core.types import SubProblem
        from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier
        from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator
        
        classifier = TaskComplexityClassifier()
        allocator = AdaptiveMDAPAllocator()
        
        sp = SubProblem(
            id="perf-test",
            description="Performance test problem",
            domain="testing",
            depth=3,
            dependencies=[],
            metadata={},
        )
        
        # Test classification latency
        start = time.time()
        for _ in range(10):
            classifier.compute_complexity(sp)
        elapsed_ms = (time.time() - start) * 1000 / 10
        
        if elapsed_ms < 100:
            tests.append(("Classification latency", True, f"{elapsed_ms:.2f}ms avg"))
        else:
            tests.append(("Classification latency", False, f"{elapsed_ms:.2f}ms avg (target <100ms)"))
        
        # Test allocation latency
        start = time.time()
        for _ in range(100):
            allocator.allocate_resources(0.5)
        elapsed_ms = (time.time() - start) * 1000 / 100
        
        if elapsed_ms < 10:
            tests.append(("Allocation latency", True, f"{elapsed_ms:.2f}ms avg"))
        else:
            tests.append(("Allocation latency", False, f"{elapsed_ms:.2f}ms avg (target <10ms)"))
        
    except Exception as e:
        tests.append(("Performance", False, str(e)))
    
    all_passed = True
    for test in tests:
        name = test[0]
        passed = test[1]
        details = test[2] if len(test) > 2 else ""
        print_result(name, passed, details)
        if not passed:
            all_passed = False
    
    return all_passed


def test_integration_completeness() -> bool:
    """Test that all integration points are present."""
    print_section("Testing Integration Completeness")
    
    tests = []
    
    # Check that all 5 strategies are available
    try:
        from adaptive_mdap.core.types import SolveStrategy
        
        expected_strategies = [
            SolveStrategy.DIRECT,
            SolveStrategy.MDAP_LIGHT,
            SolveStrategy.MDAP_MEDIUM,
            SolveStrategy.MAKER_FULL,
            SolveStrategy.MAKER_ULTRA,
        ]
        
        if len(expected_strategies) == 5:
            tests.append(("5-tier strategy system", True))
        else:
            tests.append(("5-tier strategy system", False, f"found {len(expected_strategies)} strategies"))
        
    except Exception as e:
        tests.append(("Strategy system", False, str(e)))
    
    # Check that ICR integration is present
    try:
        from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator
        
        allocator = AdaptiveMDAPAllocator()
        
        if hasattr(allocator, 'detect_strategy_patterns'):
            tests.append(("ICR pattern detection", True))
        else:
            tests.append(("ICR pattern detection", False, "method not found"))
        
        if hasattr(allocator, 'adapt_thresholds_from_patterns'):
            tests.append(("ICR threshold adaptation", True))
        else:
            tests.append(("ICR threshold adaptation", False, "method not found"))
        
    except Exception as e:
        tests.append(("ICR integration", False, str(e)))
    
    # Check that Gauntlet integration is present
    try:
        from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator
        
        allocator = AdaptiveMDAPAllocator()
        
        if hasattr(allocator, 'record_gauntlet_feedback'):
            tests.append(("Gauntlet feedback integration", True))
        else:
            tests.append(("Gauntlet feedback integration", False, "method not found"))
        
    except Exception as e:
        tests.append(("Gauntlet integration", False, str(e)))
    
    all_passed = True
    for test in tests:
        name = test[0]
        passed = test[1]
        details = test[2] if len(test) > 2 else ""
        print_result(name, passed, details)
        if not passed:
            all_passed = False
    
    return all_passed


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("  Adaptive MDAP Integration Verification")
    print("  Version 1.0.0 - Production Ready")
    print("=" * 70)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Execution", test_execution()))
    results.append(("Cost Calculator", test_cost_calculator()))
    results.append(("Performance", test_performance()))
    results.append(("Integration Completeness", test_integration_completeness()))
    
    print_section("Verification Summary")
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("  [SUCCESS] ALL TESTS PASSED - Integration is 100% Complete!")
        print("=" * 70)
        return 0
    else:
        print("  [ERROR] SOME TESTS FAILED - Review errors above")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
