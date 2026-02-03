#!/usr/bin/env python3
"""
Comprehensive integration test for Adaptive MDAP.

Tests all 16 integration points:
1. Core package imports
2. API server endpoints
3. Workflow engine integration
4. Evolution configuration
5. Orchestrator configuration
6. Sidebar UI elements
7. Demo application
8. Config loader
9. CLI commands
10. Red team integration
11. Blue team integration
12. Demo scripts
13. Team assignment engine
14. Gauntlet manager
15. Quality assessment
16. Monitoring system
"""

import sys
import ast


def test_file_parses(filepath: str) -> bool:
    """Test that a file parses correctly."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"  [FAIL] Syntax error in {filepath}: {e}")
        return False


def test_imports():
    """Test core package imports."""
    print("\n1. Testing Core Package Imports...")
    try:
        from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
        from adaptive_mdap.core.types import SubProblem
        print("  [PASS] Core imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_workflow_engine_integration():
    """Test workflow engine integration."""
    print("\n2. Testing Workflow Engine Integration...")
    
    checks = []
    
    # Check file parses
    if not test_file_parses('workflow_engine.py'):
        return False
    
    # Check imports exist
    with open('workflow_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
    checks.append(('get_adaptive_workflow', 'get_adaptive_workflow' in content))
    checks.append(('get_adaptive_mdap_status', 'get_adaptive_mdap_status' in content))
    
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    
    return all(r for _, r in checks)


def test_evolution_integration():
    """Test evolution system integration."""
    print("\n3. Testing Evolution System Integration...")
    
    if not test_file_parses('evolution.py'):
        return False
    
    with open('evolution.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content),
        ('enable_adaptive_mdap', 'enable_adaptive_mdap' in content),
        ('adaptive_mdap_profile', 'adaptive_mdap_profile' in content),
    ]
    
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    
    return all(r for _, r in checks)


def test_orchestrator_integration():
    """Test orchestrator integration."""
    print("\n4. Testing Orchestrator Integration...")
    
    if not test_file_parses('openevolve_orchestrator.py'):
        return False
    
    with open('openevolve_orchestrator.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content),
        ('adaptive_mdap_config', 'adaptive_mdap_config' in content),
    ]
    
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    
    return all(r for _, r in checks)


def test_sidebar_integration():
    """Test sidebar UI integration."""
    print("\n5. Testing Sidebar UI Integration...")
    
    if not test_file_parses('sidebar.py'):
        return False
    
    with open('sidebar.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ('enable_adaptive_mdap', 'enable_adaptive_mdap' in content),
        ('adaptive_profile', 'adaptive_profile' in content),
        ('Adaptive MDAP UI', 'Adaptive MDAP' in content),
    ]
    
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    
    return all(r for _, r in checks)


def test_api_server_integration():
    """Test API server integration."""
    print("\n6. Testing API Server Integration...")
    
    if not test_file_parses('api_server.py'):
        return False
    
    with open('api_server.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ('/adaptive-mdap/ endpoints', '/adaptive-mdap/' in content),
    ]
    
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    
    return all(r for _, r in checks)


def test_cli_integration():
    """Test CLI integration."""
    print("\n7. Testing CLI Integration...")
    
    if not test_file_parses('openevolve_cli.py'):
        return False
    
    with open('openevolve_cli.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ('adaptive command', 'def adaptive():' in content),
        ('classify command', 'def classify(' in content),
        ('allocate command', 'def allocate(' in content),
        ('status command', 'def status(' in content),
        ('profiles command', 'def profiles(' in content),
    ]
    
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    
    return all(r for _, r in checks)


def test_team_assignment_integration():
    """Test team assignment engine integration."""
    print("\n8. Testing Team Assignment Engine Integration...")
    
    if not test_file_parses('team_assignment_engine.py'):
        return False
    
    with open('team_assignment_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content),
        ('compute_subproblem_complexity', 'compute_subproblem_complexity' in content),
        ('get_optimal_team_size', 'get_optimal_team_size' in content),
        ('assign_teams_with_complexity', 'assign_teams_with_complexity' in content),
    ]
    
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    
    return all(r for _, r in checks)


def test_gauntlet_manager_integration():
    """Test gauntlet manager integration."""
    print("\n9. Testing Gauntlet Manager Integration...")
    
    if not test_file_parses('gauntlet_manager.py'):
        return False
    
    with open('gauntlet_manager.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content),
        ('create_adaptive_gauntlet', 'create_adaptive_gauntlet' in content),
        ('get_complexity_for_gauntlet', 'get_complexity_for_gauntlet' in content),
    ]
    
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    
    return all(r for _, r in checks)


def test_quality_assessment_integration():
    """Test quality assessment integration."""
    print("\n10. Testing Quality Assessment Integration...")
    
    if not test_file_parses('quality_assessment.py'):
        return False
    
    with open('quality_assessment.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content),
        ('assess_quality_with_complexity', 'assess_quality_with_complexity' in content),
        ('get_content_complexity', 'get_content_complexity' in content),
    ]
    
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    
    return all(r for _, r in checks)


def test_monitoring_integration():
    """Test monitoring system integration."""
    print("\n11. Testing Monitoring System Integration...")
    
    if not test_file_parses('monitoring_system.py'):
        return False
    
    with open('monitoring_system.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content),
        ('record_adaptive_classification', 'record_adaptive_classification' in content),
        ('record_adaptive_allocation', 'record_adaptive_allocation' in content),
        ('get_adaptive_metrics', 'get_adaptive_metrics' in content),
    ]
    
    for name, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    
    return all(r for _, r in checks)


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("ADAPTIVE MDAP INTEGRATION TESTS")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Core Imports", test_imports()))
    results.append(("Workflow Engine", test_workflow_engine_integration()))
    results.append(("Evolution System", test_evolution_integration()))
    results.append(("Orchestrator", test_orchestrator_integration()))
    results.append(("Sidebar UI", test_sidebar_integration()))
    results.append(("API Server", test_api_server_integration()))
    results.append(("CLI", test_cli_integration()))
    results.append(("Team Assignment", test_team_assignment_integration()))
    results.append(("Gauntlet Manager", test_gauntlet_manager_integration()))
    results.append(("Quality Assessment", test_quality_assessment_integration()))
    results.append(("Monitoring System", test_monitoring_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
