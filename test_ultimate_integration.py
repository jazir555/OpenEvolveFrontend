"""
Test suite for the ultimate integration functions that support both
native OpenEvolve features AND the comprehensive workflow system.
"""

import pytest
from typing import Dict, Any
import time

# Import the ultimate functions
from evolution import (
    run_ultimate_comprehensive_evolution,
    run_native_openevolve_with_workflow_enhancement,
    get_comprehensive_evolution_capabilities
)

from adversarial import (
    run_ultimate_adversarial_testing,
    run_native_openevolve_adversarial_only,
    get_adversarial_testing_capabilities,
    create_comprehensive_adversarial_config
)


def test_ultimate_comprehensive_evolution():
    """Test the ultimate comprehensive evolution function"""
    print("Testing ultimate comprehensive evolution...")
    
    test_content = "def authenticate(user, password): return user == 'admin' and password == '123'"
    
    result = run_ultimate_comprehensive_evolution(
        content=test_content,
        content_type="code_python",
        evolution_mode="adversarial",
        use_decomposition=True,
        enable_all_features=True,
        max_iterations=5,  # Keep it small for testing
        population_size=10
    )
    
    # Validate result structure
    assert isinstance(result, dict)
    assert "success" in result
    assert "operation_id" in result
    assert "system_architecture" in result
    assert result["system_architecture"] == "tripartite_ai_with_openevolve"
    assert "implementation_level" in result
    assert result["implementation_level"] == "ultimate_comprehensive"
    
    # Validate phases
    assert "workflow_phases" in result
    phases = result["workflow_phases"]
    expected_phases = [
        "phase_1_initialization",
        "phase_2_adversarial_testing", 
        "phase_3_evolutionary_optimization",
        "phase_4_evaluator_integration",
        "phase_5_model_management",
        "phase_6_quality_assurance"
    ]
    
    for phase in expected_phases:
        assert phase in phases
        assert "status" in phases[phase]
        assert "duration" in phases[phase]
    
    # Validate metrics
    assert "openevolve_metrics" in result
    assert "workflow_metrics" in result
    assert "hybrid_metrics" in result
    
    # Validate advanced features
    assert "advanced_features" in result
    features = result["advanced_features"]
    assert "problem_decomposition" in features
    assert "meta_learning" in features
    assert "transfer_learning" in features
    
    print(f"[OK] Ultimate comprehensive evolution test passed")
    print(f"   Overall score: {result.get('overall_score', 0):.4f}")
    print(f"   Duration: {result.get('total_duration', 0):.2f}s")
    
    return True


def test_native_openevolve_with_workflow_enhancement():
    """Test native OpenEvolve with workflow enhancement"""
    print("Testing native OpenEvolve with workflow enhancement...")
    
    test_content = "# TODO: Implement secure authentication system"
    
    result = run_native_openevolve_with_workflow_enhancement(
        content=test_content,
        content_type="document_technical",
        evolution_mode="standard",
        workflow_enhancement=True,
        max_iterations=3,
        temperature=0.7
    )
    
    # Validate result structure
    assert isinstance(result, dict)
    assert "success" in result
    assert "native_openevolve_result" in result
    assert "workflow_enhancements" in result
    assert "combined_metrics" in result
    assert "final_content" in result
    
    # Validate combined metrics
    metrics = result["combined_metrics"]
    assert "native_openevolve_score" in metrics
    assert "workflow_enhancement_applied" in metrics
    assert "final_quality_score" in metrics
    assert "hybrid_approach" in metrics
    assert metrics["hybrid_approach"] is True
    
    print(f"[OK] Native OpenEvolve + workflow enhancement test passed")
    print(f"   Native score: {metrics.get('native_openevolve_score', 0):.4f}")
    print(f"   Enhancement applied: {metrics.get('workflow_enhancement_applied', False)}")
    
    return True


def test_ultimate_adversarial_testing():
    """Test the ultimate adversarial testing function"""
    print("Testing ultimate adversarial testing...")
    
    test_content = "SELECT * FROM users WHERE username = '" + "admin" + "' AND password = '" + "password" + "'"
    
    result = run_ultimate_adversarial_testing(
        content=test_content,
        content_type="code_sql",
        use_native_openevolve=True,
        use_workflow_system=True,
        adversarial_rounds=3,
        attack_strength=0.8,
        defense_strength=1.0
    )
    
    # Validate result structure
    assert isinstance(result, dict)
    assert "success" in result
    assert "operation_id" in result
    assert "system_type" in result
    assert result["system_type"] == "ultimate_adversarial_hybrid"
    
    # Validate testing phases
    assert "testing_phases" in result
    phases = result["testing_phases"]
    expected_phases = [
        "phase_1_native_adversarial",
        "phase_2_workflow_testing",
        "phase_3_hybrid_analysis",
        "phase_4_comprehensive_validation"
    ]
    
    for phase in expected_phases:
        assert phase in phases
        assert "status" in phases[phase]
        assert "duration" in phases[phase]
    
    # Validate metrics
    assert "native_openevolve_results" in result
    assert "workflow_results" in result
    assert "hybrid_metrics" in result
    
    # Validate hybrid metrics
    hybrid_metrics = result["hybrid_metrics"]
    assert "combined_robustness_score" in hybrid_metrics
    assert "improvement_ratio" in hybrid_metrics
    assert "testing_effectiveness" in hybrid_metrics
    assert "overall_security_score" in hybrid_metrics
    
    print(f"[OK] Ultimate adversarial testing test passed")
    print(f"   Security score: {hybrid_metrics.get('overall_security_score', 0):.4f}")
    print(f"   Testing effectiveness: {hybrid_metrics.get('testing_effectiveness', 0):.4f}")
    
    return True


def test_native_openevolve_adversarial_only():
    """Test pure native OpenEvolve adversarial evolution"""
    print("Testing native OpenEvolve adversarial only...")
    
    test_content = "function validateInput(input) { return input; }"
    
    result = run_native_openevolve_adversarial_only(
        content=test_content,
        content_type="code_javascript",
        max_iterations=5,
        adversarial_rounds=3,
        attack_strength=1.0
    )
    
    # Validate result structure
    assert isinstance(result, dict)
    assert "success" in result
    assert "approach" in result
    assert result["approach"] == "native_openevolve_only"
    assert "openevolve_result" in result
    assert "metrics" in result
    
    # Validate metrics
    if result["success"]:
        metrics = result["metrics"]
        assert "adversarial_score" in metrics
        assert "robustness_improvement" in metrics
        assert "iterations_completed" in metrics
        assert "total_duration" in metrics
    
    print(f"[OK] Native OpenEvolve adversarial only test passed")
    print(f"   Success: {result.get('success', False)}")
    
    return True


def test_comprehensive_adversarial_config():
    """Test comprehensive adversarial configuration creation"""
    print("Testing comprehensive adversarial configuration...")
    
    config = create_comprehensive_adversarial_config(
        adversarial_rounds=7,
        attack_strength=1.2,
        defense_strength=0.9,
        use_decomposition=True,
        enable_all_features=True
    )
    
    # Validate configuration
    assert hasattr(config, 'adversarial_rounds')
    assert config.adversarial_rounds == 7
    assert hasattr(config, 'attack_strength')
    assert config.attack_strength == 1.2
    assert hasattr(config, 'defense_strength')
    assert config.defense_strength == 0.9
    
    print(f"[OK] Comprehensive adversarial config test passed")
    print(f"   Rounds: {config.adversarial_rounds}")
    print(f"   Attack strength: {config.attack_strength}")
    
    return True


def test_evolution_capabilities():
    """Test evolution capabilities reporting"""
    print("Testing evolution capabilities...")
    
    capabilities = get_comprehensive_evolution_capabilities()
    
    # Validate capabilities structure
    assert isinstance(capabilities, dict)
    assert "native_openevolve" in capabilities
    assert "workflow_system" in capabilities
    assert "combined_capabilities" in capabilities
    
    # Validate native OpenEvolve capabilities
    native = capabilities["native_openevolve"]
    assert "available" in native
    assert "evolution_modes" in native
    assert "parameters_supported" in native
    assert "advanced_features" in native
    
    # Validate workflow system capabilities
    workflow = capabilities["workflow_system"]
    assert "available" in workflow
    assert "team_components" in workflow
    assert "workflow_phases" in workflow
    
    # Validate combined capabilities
    combined = capabilities["combined_capabilities"]
    assert "ultimate_comprehensive_evolution" in combined
    assert "hybrid_approaches" in combined
    assert "full_parameter_support" in combined
    
    print(f"[OK] Evolution capabilities test passed")
    print(f"   Native OpenEvolve available: {native.get('available', False)}")
    print(f"   Workflow system available: {workflow.get('available', False)}")
    
    return True


def test_adversarial_capabilities():
    """Test adversarial capabilities reporting"""
    print("Testing adversarial capabilities...")
    
    capabilities = get_adversarial_testing_capabilities()
    
    # Validate capabilities structure
    assert isinstance(capabilities, dict)
    assert "native_openevolve_adversarial" in capabilities
    assert "workflow_adversarial" in capabilities
    assert "hybrid_capabilities" in capabilities
    
    # Validate native adversarial capabilities
    native = capabilities["native_openevolve_adversarial"]
    assert "available" in native
    assert "features" in native
    assert "parameters_supported" in native
    
    # Validate workflow adversarial capabilities
    workflow = capabilities["workflow_adversarial"]
    assert "available" in workflow
    assert "team_components" in workflow
    assert "testing_phases" in workflow
    
    print(f"[OK] Adversarial capabilities test passed")
    print(f"   Native adversarial available: {native.get('available', False)}")
    print(f"   Workflow adversarial available: {workflow.get('available', False)}")
    
    return True


def test_integration_compatibility():
    """Test that the integration maintains compatibility with existing functions"""
    print("Testing integration compatibility...")
    
    # Test that existing functions still work
    from evolution import run_comprehensive_evolution, create_evolution_configuration_from_session
    from adversarial import run_comprehensive_adversarial_testing, create_adversarial_configuration_from_session
    
    # Test existing evolution function
    try:
        config = create_evolution_configuration_from_session()
        assert hasattr(config, 'evolution_mode')
        assert hasattr(config, 'max_iterations')
        print("   [OK] Existing evolution configuration works")
    except Exception as e:
        print(f"   [FAIL] Evolution configuration error: {e}")
        return False
    
    # Test existing adversarial function
    try:
        adv_config = create_adversarial_configuration_from_session()
        assert hasattr(adv_config, 'adversarial_rounds')
        assert hasattr(adv_config, 'attack_strength')
        print("   [OK] Existing adversarial configuration works")
    except Exception as e:
        print(f"   [FAIL] Adversarial configuration error: {e}")
        return False
    
    print(f"[OK] Integration compatibility test passed")
    
    return True


if __name__ == "__main__":
    print("🚀 Running Ultimate Integration Tests...")
    print("=" * 60)
    
    tests = [
        test_ultimate_comprehensive_evolution,
        test_native_openevolve_with_workflow_enhancement,
        test_ultimate_adversarial_testing,
        test_native_openevolve_adversarial_only,
        test_comprehensive_adversarial_config,
        test_evolution_capabilities,
        test_adversarial_capabilities,
        test_integration_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"🎯 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL ULTIMATE INTEGRATION TESTS PASSED!")
    else:
        print(f"[WARN] {failed} tests failed")