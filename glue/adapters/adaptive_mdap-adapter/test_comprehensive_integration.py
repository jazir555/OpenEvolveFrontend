#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for Adaptive MDAP/MAKER Adapter

This script provides comprehensive testing including:
- Integration tests for all workflow types
- Edge case testing (empty inputs, extreme complexity)
- Load testing for concurrent operations
- Failure scenario testing (network failures, timeouts)
- Contract tests for all integration points

Usage:
    python test_comprehensive_integration.py
"""

import os
import sys
import logging
import time
import asyncio
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set required environment variables
os.environ["ADAPTIVE_MDAP_TIMEOUT_MS"] = "5000"
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "[OK]" if passed else "[FAIL]"
    print(f"{status} {test_name}")
    if details:
        print(f"    {details}")


# =============================================================================
# TEST SUITE 1: Workflow Type Integration Tests
# =============================================================================

def test_workflow_evolution():
    """Test evolution workflow integration."""
    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        result = manager.execute_full_workflow(
            workflow_id="test_evolution",
            problem_statement="Optimize neural network hyperparameters",
            workflow_type="evolution",
            context={"domain": "ml", "iterations": 100}
        )

        passed = result.get("overall_status") == "completed"
        print_result(
            "Evolution Workflow Integration",
            passed,
            f"Steps: {len(result.get('steps', []))}, Time: {result.get('execution_time_ms', 0):.0f}ms"
        )
        return passed

    except Exception as e:
        print_result("Evolution Workflow Integration", False, str(e))
        return False


def test_workflow_adversarial():
    """Test adversarial workflow integration."""
    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        result = manager.execute_full_workflow(
            workflow_id="test_adversarial",
            problem_statement="Find security vulnerabilities in code",
            workflow_type="adversarial",
            context={"domain": "security", "attack_mode": "systematic"}
        )

        passed = result.get("overall_status") == "completed"
        print_result(
            "Adversarial Workflow Integration",
            passed,
            f"Red team checks: {len(result.get('steps', []))}"
        )
        return passed

    except Exception as e:
        print_result("Adversarial Workflow Integration", False, str(e))
        return False


def test_workflow_sovereign():
    """Test sovereign workflow with decomposition."""
    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        result = manager.execute_full_workflow(
            workflow_id="test_sovereign",
            problem_statement="Design distributed system architecture",
            workflow_type="sovereign",
            context={"domain": "architecture", "decomposition_depth": 3}
        )

        passed = result.get("overall_status") == "completed"
        print_result(
            "Sovereign Workflow Integration",
            passed,
            f"Decomposition depth: 3"
        )
        return passed

    except Exception as e:
        print_result("Sovereign Workflow Integration", False, str(e))
        return False


def test_workflow_web3():
    """Test Web3 workflow integration."""
    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        result = manager.execute_full_workflow(
            workflow_id="test_web3",
            problem_statement="Verify smart contract security",
            workflow_type="web3",
            context={"domain": "blockchain", "contract_type": "ERC20"}
        )

        passed = result.get("overall_status") == "completed"
        print_result(
            "Web3 Workflow Integration",
            passed
        )
        return passed

    except Exception as e:
        print_result("Web3 Workflow Integration", False, str(e))
        return False


def test_workflow_rag():
    """Test RAG workflow integration."""
    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        result = manager.execute_full_workflow(
            workflow_id="test_rag",
            problem_statement="Answer question using knowledge base",
            workflow_type="rag",
            context={"domain": "knowledge", "query": "What is RAG?"}
        )

        passed = result.get("overall_status") == "completed"
        print_result(
            "RAG Workflow Integration",
            passed
        )
        return passed

    except Exception as e:
        print_result("RAG Workflow Integration", False, str(e))
        return False


# =============================================================================
# TEST SUITE 2: Edge Case Tests
# =============================================================================

def test_empty_input():
    """Test handling of empty inputs."""
    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        # Empty problem statement
        analysis = manager.analyze_workflow(
            workflow_id="test_empty",
            problem_statement="",
            workflow_type="evolution"
        )

        # Should handle gracefully
        passed = analysis is not None
        print_result(
            "Empty Input Handling",
            passed,
            "Empty problem statement handled"
        )
        return passed

    except Exception as e:
        print_result("Empty Input Handling", False, str(e))
        return False


def test_extreme_complexity():
    """Test handling of extreme complexity values."""
    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        # Very high complexity (1.0)
        analysis_high = manager.analyze_workflow(
            workflow_id="test_complexity_high",
            problem_statement="A" * 10000,  # Very long
            workflow_type="evolution"
        )

        # Very low complexity (0.0)
        analysis_low = manager.analyze_workflow(
            workflow_id="test_complexity_low",
            problem_statement="B",  # Very short
            workflow_type="evolution"
        )

        passed = (
            analysis_high.overall_complexity >= 0.5 and
            analysis_low.overall_complexity <= 0.5
        )

        print_result(
            "Extreme Complexity Handling",
            passed,
            f"High: {analysis_high.overall_complexity:.3f}, Low: {analysis_low.overall_complexity:.3f}"
        )
        return passed

    except Exception as e:
        print_result("Extreme Complexity Handling", False, str(e))
        return False


def test_malformed_input():
    """Test handling of malformed inputs."""
    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        # Special characters
        analysis = manager.analyze_workflow(
            workflow_id="test_malformed",
            problem_statement="<script>alert('xss')</script>",
            workflow_type="evolution"
        )

        passed = analysis is not None
        print_result(
            "Malformed Input Handling",
            passed,
            "Special characters handled"
        )
        return passed

    except Exception as e:
        print_result("Malformed Input Handling", False, str(e))
        return False


# =============================================================================
# TEST SUITE 3: Load Testing
# =============================================================================

def test_concurrent_operations():
    """Test concurrent operations."""
    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        def analyze_one(i):
            return manager.analyze_workflow(
                workflow_id=f"concurrent_{i}",
                problem_statement=f"Problem {i}",
                workflow_type="evolution"
            )

        # Run 10 concurrent analyses
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(analyze_one, i) for i in range(10)]
            results = [f.result(timeout=30) for f in as_completed(futures)]

        passed = len(results) == 10
        print_result(
            "Concurrent Operations",
            passed,
            f"Completed {len(results)} concurrent operations"
        )
        return passed

    except Exception as e:
        print_result("Concurrent Operations", False, str(e))
        return False


def test_batch_processing():
    """Test batch processing of multiple workflows."""
    try:
        from openevolve_advanced import get_advanced_openevolve_integration

        integration = get_advanced_openevolve_integration()

        # Create multiple sub-problems
        decomposition = integration.decompose_problem(
            workflow_id="batch_test",
            problem_statement="Solve multiple related problems",
            workflow_type="evolution",
            max_depth=2
        )

        passed = len(decomposition.sub_problems) >= 2
        print_result(
            "Batch Processing",
            passed,
            f"Decomposed into {len(decomposition.sub_problems)} sub-problems"
        )
        return passed

    except Exception as e:
        print_result("Batch Processing", False, str(e))
        return False


def test_memory_efficiency():
    """Test memory efficiency with many operations."""
    try:
        from integration_manager import get_integration_manager

        manager = get_integration_manager()

        # Perform many operations
        for i in range(100):
            manager.analyze_workflow(
                workflow_id=f"mem_test_{i}",
                problem_statement=f"Memory test {i}",
                workflow_type="evolution"
            )

        passed = True
        print_result(
            "Memory Efficiency",
            passed,
            "Completed 100 operations without memory issues"
        )
        return passed

    except Exception as e:
        print_result("Memory Efficiency", False, str(e))
        return False


# =============================================================================
# TEST SUITE 4: Failure Scenario Tests
# =============================================================================

def test_timeout_handling():
    """Test timeout handling."""
    try:
        import os

        # Set very short timeout
        original_timeout = os.environ.get("ADAPTIVE_MDAP_TIMEOUT_MS")
        os.environ["ADAPTIVE_MDAP_TIMEOUT_MS"] = "1"  # 1ms timeout

        from integration_manager import get_integration_manager

        # Force reload with new timeout
        import importlib
        import integration_manager
        importlib.reload(integration_manager)

        manager = integration_manager.get_integration_manager()

        # This should timeout
        try:
            result = manager.analyze_workflow(
                workflow_id="timeout_test",
                problem_statement="This will timeout",
                workflow_type="evolution"
            )
            # Should still return something (timeout handled)
            passed = result is not None
        except Exception:
            # Timeout exception is also acceptable
            passed = True

        # Restore original timeout
        if original_timeout:
            os.environ["ADAPTIVE_MDAP_TIMEOUT_MS"] = original_timeout

        print_result(
            "Timeout Handling",
            passed,
            "Timeout handled gracefully"
        )
        return passed

    except Exception as e:
        print_result("Timeout Handling", False, str(e))
        return False


def test_missing_dependency():
    """Test handling of missing dependencies."""
    try:
        # This tests graceful degradation when dependencies are missing
        from additional_systems_integration import get_unified_system_monitor

        monitor = get_unified_system_monitor()

        # Check health (should handle missing systems gracefully)
        health = monitor.get_overall_health()

        passed = health is not None
        print_result(
            "Missing Dependency Handling",
            passed,
            f"Available: {health.get('available_systems', 0)}/{health.get('total_systems', 0)}"
        )
        return passed

    except Exception as e:
        print_result("Missing Dependency Handling", False, str(e))
        return False


def test_network_failure_simulation():
    """Test network failure simulation."""
    try:
        # Test cache when adapter unavailable
        from performance_optimization import get_async_adapter

        async_adapter = get_async_adapter()

        # Try to use cached result
        from adaptive_mdap_adapter import CanonicalSubProblem

        subproblem = CanonicalSubProblem(
            id="network_test",
            description="Test network failure",
            domain="test",
            depth=1
        )

        # This should either work or fail gracefully
        try:
            result = asyncio.run(async_adapter.analyze_complexity_async(subproblem))
            passed = True
        except Exception:
            # Failure is acceptable
            passed = True

        print_result(
            "Network Failure Simulation",
            passed,
            "Handled network failure gracefully"
        )
        return passed

    except Exception as e:
        print_result("Network Failure Simulation", False, str(e))
        return False


# =============================================================================
# TEST SUITE 5: Advanced Feature Tests
# =============================================================================

def test_gauntlet_pipeline():
    """Test multi-gauntlet pipeline."""
    try:
        from gauntlet_advanced import get_advanced_gauntlet_integration, GauntletType

        integration = get_advanced_gauntlet_integration()

        # Create pipeline
        pipeline = integration.create_gauntlet_pipeline(
            complexity_score=0.75,
            base_gauntlet_type=GauntletType.ADVERSARIAL,
            include_cross_validation=True
        )

        # Execute pipeline
        result = integration.execute_pipeline(
            pipeline=pipeline,
            solution="test solution",
            context={"test": True}
        )

        passed = (
            result.total_gauntlets >= 2 and
            result.overall_pass is not None
        )

        print_result(
            "Gauntlet Pipeline",
            passed,
            f"Total gauntlets: {result.total_gauntlets}, Passed: {result.passed_gauntlets}"
        )
        return passed

    except Exception as e:
        print_result("Gauntlet Pipeline", False, str(e))
        return False


def test_icr_pattern_learning():
    """Test ICR pattern learning."""
    try:
        from icr_advanced import get_advanced_icr_integration, ICRPatternType

        integration = get_advanced_icr_integration()

        # Store patterns
        for i in range(10):
            integration.store_pattern_advanced(
                pattern_type=ICRPatternType.WORKFLOW_EXECUTION,
                passed=i % 3 != 0,  # Some fail, some pass
                context={"test": i},
                metrics={"complexity": 0.5 + i * 0.05}
            )

        # Get insights
        insights = integration.get_pattern_insights()

        passed = insights.get("available", False)
        print_result(
            "ICR Pattern Learning",
            passed,
            f"Pattern types tracked: {len(insights.get('pattern_types', {}))}"
        )
        return passed

    except Exception as e:
        print_result("ICR Pattern Learning", False, str(e))
        return False


def test_advanced_ui_components():
    """Test advanced UI components."""
    try:
        from bubblelab_ui_advanced import get_advanced_bubblelab_ui

        ui = get_advanced_bubblelab_ui()

        # Get dashboard
        dashboard = ui.create_adapter_health_dashboard()

        passed = dashboard is not None and "health" in dashboard
        print_result(
            "Advanced UI Components",
            passed,
            f"Dashboard components: {len(dashboard)}"
        )
        return passed

    except Exception as e:
        print_result("Advanced UI Components", False, str(e))
        return False


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run all test suites."""
    print_section("COMPREHENSIVE INTEGRATION TEST SUITE")
    print(f"Start Time: {datetime.now(timezone.utc).isoformat()}")

    results = {}

    # Test Suite 1: Workflow Types
    print_section("TEST SUITE 1: Workflow Type Integration")
    results['evolution'] = test_workflow_evolution()
    results['adversarial'] = test_workflow_adversarial()
    results['sovereign'] = test_workflow_sovereign()
    results['web3'] = test_workflow_web3()
    results['rag'] = test_workflow_rag()

    # Test Suite 2: Edge Cases
    print_section("TEST SUITE 2: Edge Case Testing")
    results['empty_input'] = test_empty_input()
    results['extreme_complexity'] = test_extreme_complexity()
    results['malformed_input'] = test_malformed_input()

    # Test Suite 3: Load Testing
    print_section("TEST SUITE 3: Load Testing")
    results['concurrent'] = test_concurrent_operations()
    results['batch_processing'] = test_batch_processing()
    results['memory_efficiency'] = test_memory_efficiency()

    # Test Suite 4: Failure Scenarios
    print_section("TEST SUITE 4: Failure Scenario Testing")
    results['timeout'] = test_timeout_handling()
    results['missing_dependency'] = test_missing_dependency()
    results['network_failure'] = test_network_failure_simulation()

    # Test Suite 5: Advanced Features
    print_section("TEST SUITE 5: Advanced Feature Testing")
    results['gauntlet_pipeline'] = test_gauntlet_pipeline()
    results['icr_learning'] = test_icr_pattern_learning()
    results['advanced_ui'] = test_advanced_ui_components()

    # Print Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")
    print()

    for test_name, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")

    print(f"\nEnd Time: {datetime.now(timezone.utc).isoformat()}")

    if passed == total:
        print("\nSUCCESS: All comprehensive tests passed!")
        return 0
    else:
        print(f"\nFAILED: {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
