"""
Test to verify causal-learn integration is fully optional.

This test simulates an environment where causal-learn library is not installed
and verifies the system works correctly without it.
"""

import sys
import os

# Block causal-learn from being imported to simulate it not being installed
sys.modules['causallearn'] = None
sys.modules['causallearn.search'] = None
sys.modules['causallearn.search.ConstraintBased'] = None
sys.modules['causallearn.search.ScoreBased'] = None
sys.modules['causallearn.search.FCMBased'] = None
sys.modules['causallearn.search.FCMBased.lingam'] = None
sys.modules['causallearn.search.Granger'] = None
sys.modules['causallearn.utils'] = None
sys.modules['causallearn.utils.cit'] = None

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knowledge_engine'))


def test_causal_learn_integration_optional():
    """Test 1: Verify CausalLearnIntegration handles missing library gracefully"""
    print("\n" + "="*60)
    print("TEST 1: CausalLearnIntegration without causal-learn library")
    print("="*60)
    
    from knowledge_engine.integrations.causal_learn_integration import (
        CausalLearnIntegration,
        CausalDiscoveryEngine
    )
    
    # Test CausalLearnIntegration
    integration = CausalLearnIntegration()
    assert hasattr(integration, 'is_available'), "Missing is_available method"
    assert hasattr(integration, 'discover_structure'), "Missing discover_structure method"
    
    # Should report not available
    available = integration.is_available()
    print(f"CausalLearnIntegration available: {available}")
    assert not available, "Should not be available when library not installed"
    
    # discover_structure should return error gracefully
    import numpy as np
    result = integration.discover_structure(
        data=np.random.randn(10, 3),
        algorithm='pc'
    )
    print(f"Result when unavailable: {result}")
    assert result.get('status') == 'error', "Should return error status"
    assert 'not available' in result.get('message', '').lower(), "Should indicate unavailability"
    
    print("[PASS] CausalLearnIntegration works without causal-learn library")
    return True


def test_unified_extractor_optional():
    """Test 2: Verify UnifiedKnowledgeExtractor works without causal-learn library"""
    print("\n" + "="*60)
    print("TEST 2: UnifiedKnowledgeExtractor without causal-learn library")
    print("="*60)
    
    from knowledge_engine.integrations.unified_knowledge_extraction import (
        UnifiedKnowledgeExtractor,
        ExtractionResult
    )
    
    extractor = UnifiedKnowledgeExtractor()
    
    # Should initialize without error
    print(f"Available modules: {extractor.get_available_modules()}")
    
    # Causal methods should exist
    assert hasattr(extractor, 'discover_causal_structure'), "Missing discover_causal_structure"
    assert hasattr(extractor, 'identify_confounders'), "Missing identify_confounders"
    
    # Calling causal methods should return error result, not raise exception
    import numpy as np
    result = extractor.discover_causal_structure(
        data=np.random.randn(10, 3),
        variable_names=['A', 'B', 'C'],
        algorithm='pc'
    )
    
    assert isinstance(result, ExtractionResult), "Should return ExtractionResult"
    print(f"Causal discovery result status: {result.status}")
    
    # The module exists but library is not available
    if 'causal_learn' in extractor.get_available_modules():
        print("[OK] causal_learn module exists but library not available")
        # Check if it's actually usable
        status = extractor.get_module_status()
        print(f"Module status: {status}")
    
    print("[PASS] UnifiedKnowledgeExtractor works without causal-learn library")
    return True


def test_master_engine_optional():
    """Test 3: Verify Master Engine works without causal-learn library"""
    print("\n" + "="*60)
    print("TEST 3: Master Engine without causal-learn library")
    print("="*60)
    
    from knowledge_engine.master_engine import ComponentRegistry, CAUSAL_LEARN_AVAILABLE
    
    # CAUSAL_LEARN_AVAILABLE refers to the integration module being importable
    print(f"CAUSAL_LEARN_AVAILABLE (integration): {CAUSAL_LEARN_AVAILABLE}")
    
    # Component registry should initialize without error
    registry = ComponentRegistry()
    
    # causal_learn component should exist
    component = registry.get_component('causal_learn')
    assert component is not None, "causal_learn component should exist"
    print("[OK] causal_learn component exists in registry")
    
    # Check if component is available (has is_available method)
    if hasattr(component, 'is_available'):
        lib_available = component.is_available()
        print(f"Causal-learn library available: {lib_available}")
        assert not lib_available, "Library should not be available"
    else:
        print("[OK] Component is mock (library not available)")
    
    # Capabilities should be registered
    capabilities = registry.capabilities.get('causal_learn', [])
    print(f"Registered capabilities: {capabilities}")
    assert 'causal_discovery' in capabilities, "causal_discovery capability should be registered"
    
    print("[PASS] Master Engine works without causal-learn library")
    return True


def test_orchestrator_optional():
    """Test 4: Verify KnowledgeOrchestrator works without causal-learn library"""
    print("\n" + "="*60)
    print("TEST 4: KnowledgeOrchestrator without causal-learn library")
    print("="*60)
    
    from knowledge_engine.orchestration.knowledge_orchestrator import (
        OrchestratorConfig,
        ComponentType
    )
    
    config = OrchestratorConfig()
    
    # CAUSAL_LEARN should be in component types
    assert hasattr(ComponentType, 'CAUSAL_LEARN'), "CAUSAL_LEARN should be in ComponentType"
    
    # Component config should exist
    assert ComponentType.CAUSAL_LEARN in config.components, "CAUSAL_LEARN should be in components"
    
    # Can enable/disable without error
    config.enable_component(ComponentType.CAUSAL_LEARN)
    config.disable_component(ComponentType.CAUSAL_LEARN)
    
    print("[PASS] KnowledgeOrchestrator works without causal-learn library")
    return True


def test_analytics_engine_optional():
    """Test 5: Verify AdvancedAnalyticsEngine works without causal-learn library"""
    print("\n" + "="*60)
    print("TEST 5: AdvancedAnalyticsEngine without causal-learn library")
    print("="*60)
    
    from knowledge_engine.advanced_analytics_engine import (
        AdvancedAnalyticsEngine,
        CausalDiscoveryEngine,
    )
    
    print(f"CausalDiscoveryEngine: {CausalDiscoveryEngine}")
    
    # Should be None when integrations not available
    assert CausalDiscoveryEngine is None, "CausalDiscoveryEngine should be None (integrations unavailable)"
    
    # Engine should initialize without error
    engine = AdvancedAnalyticsEngine()
    print(f"Analytics engine initialized: {engine}")
    
    # Causal integration should not be in integrations dict
    assert 'causal' not in engine.integrations, "Causal should not be in integrations when unavailable"
    
    print("[PASS] AdvancedAnalyticsEngine works without causal-learn library")
    return True


def test_imports_optional():
    """Test 6: Verify all imports work without causal-learn library"""
    print("\n" + "="*60)
    print("TEST 6: All imports work without causal-learn library")
    print("="*60)
    
    # Test knowledge_engine.integrations import
    from knowledge_engine.integrations import (
        CausalLearnIntegration,
        CausalDiscoveryEngine,
        CAUSAL_LEARN_AVAILABLE
    )
    
    # CAUSAL_LEARN_AVAILABLE refers to integration module being importable
    print(f"CAUSAL_LEARN_AVAILABLE (integration): {CAUSAL_LEARN_AVAILABLE}")
    
    # Integration should be importable (module exists)
    assert CAUSAL_LEARN_AVAILABLE, "Integration module should be available"
    assert CausalLearnIntegration is not None, "CausalLearnIntegration class should exist"
    assert CausalDiscoveryEngine is not None, "CausalDiscoveryEngine class should exist"
    
    # But library should not be available
    integration = CausalLearnIntegration()
    lib_available = integration.is_available()
    print(f"Library available: {lib_available}")
    assert not lib_available, "Library should not be available"
    
    print("[PASS] All imports work correctly without causal-learn library")
    return True


def test_integration_vs_library():
    """Test 7: Distinguish between integration availability and library availability"""
    print("\n" + "="*60)
    print("TEST 7: Integration vs Library Availability")
    print("="*60)
    
    from knowledge_engine.integrations import CAUSAL_LEARN_AVAILABLE as INTEGRATION_AVAILABLE
    from knowledge_engine.integrations.causal_learn_integration import CausalLearnIntegration
    
    # Integration module should be available
    print(f"Integration module available: {INTEGRATION_AVAILABLE}")
    assert INTEGRATION_AVAILABLE, "Integration module should be available"
    
    # Create integration instance
    integration = CausalLearnIntegration()
    
    # Library should not be available
    library_available = integration.is_available()
    print(f"Causal-learn library available: {library_available}")
    assert not library_available, "Library should not be available"
    
    # All methods should exist but return errors
    methods = ['discover_structure', 'get_available_algorithms', 'is_available']
    for method in methods:
        assert hasattr(integration, method), f"Missing method: {method}"
        print(f"[OK] Method '{method}' exists")
    
    # Calling methods should not raise exceptions
    result = integration.get_available_algorithms()
    print(f"get_available_algorithms() returned: {result}")
    assert result == [], "Should return empty list when library unavailable"
    
    print("[PASS] Integration vs Library availability correctly distinguished")
    return True


def run_all_tests():
    """Run all optionality tests"""
    print("\n" + "="*70)
    print("CAUSAL-LEARN OPTIONALITY TEST SUITE")
    print("="*70)
    print("Testing that causal-learn is fully optional...")
    print("\nNote: Testing WITHOUT causal-learn library installed")
    
    tests = [
        ("Integration vs Library", test_integration_vs_library),
        ("Imports", test_imports_optional),
        ("CausalLearnIntegration", test_causal_learn_integration_optional),
        ("UnifiedKnowledgeExtractor", test_unified_extractor_optional),
        ("Master Engine", test_master_engine_optional),
        ("KnowledgeOrchestrator", test_orchestrator_optional),
        ("AdvancedAnalyticsEngine", test_analytics_engine_optional),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n[FAIL] TEST FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, passed, _ in results if passed)
    failed = sum(1 for _, passed, _ in results if not passed)
    
    for name, passed, error in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")
    
    print("-"*70)
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n[OK] Causal-learn is FULLY OPTIONAL - system works without the library!")
        print("     Integration code exists but gracefully handles missing library.")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
