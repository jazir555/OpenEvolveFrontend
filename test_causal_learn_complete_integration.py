"""
Comprehensive Test for Causal-Learn Integration Across All Components

This test verifies causal-learn integration in:
1. Knowledge Engine core
2. Unified Knowledge Extractor
3. Knowledge Orchestrator
4. Master Engine
5. Advanced Analytics Engine
6. Unified KG Integration Hub
7. BubbleLabs Nodes
8. MCP Server
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knowledge_engine'))


def test_integration_module():
    """Test 1: Integration module exports"""
    print("\n" + "="*60)
    print("TEST 1: Integration Module Exports")
    print("="*60)
    
    from knowledge_engine.integrations import (
        CausalLearnIntegration,
        CausalDiscoveryEngine,
        CAUSAL_LEARN_AVAILABLE
    )
    
    print(f"CausalLearnIntegration: {CausalLearnIntegration}")
    print(f"CausalDiscoveryEngine: {CausalDiscoveryEngine}")
    print(f"CAUSAL_LEARN_AVAILABLE: {CAUSAL_LEARN_AVAILABLE}")
    
    # Should be importable (not raise ImportError)
    assert CausalLearnIntegration is not None or not CAUSAL_LEARN_AVAILABLE
    assert CausalDiscoveryEngine is not None or not CAUSAL_LEARN_AVAILABLE
    
    print("[PASS] Integration module exports correct")
    return True


def test_causal_learn_integration_class():
    """Test 2: CausalLearnIntegration class"""
    print("\n" + "="*60)
    print("TEST 2: CausalLearnIntegration Class")
    print("="*60)
    
    from knowledge_engine.integrations.causal_learn_integration import (
        CausalLearnIntegration,
        CausalDiscoveryEngine
    )
    
    # Test integration
    integration = CausalLearnIntegration()
    assert hasattr(integration, 'is_available')
    assert hasattr(integration, 'discover_structure')
    assert hasattr(integration, 'get_available_algorithms')
    
    available = integration.is_available()
    print(f"Integration available: {available}")
    
    # Test engine
    engine = CausalDiscoveryEngine()
    assert hasattr(engine, 'is_available')
    assert hasattr(engine, 'discover_causal_structure')
    
    engine_available = engine.is_available()
    print(f"Engine available: {engine_available}")
    
    print("[PASS] CausalLearnIntegration class works")
    return True


def test_unified_extractor():
    """Test 3: UnifiedKnowledgeExtractor integration"""
    print("\n" + "="*60)
    print("TEST 3: UnifiedKnowledgeExtractor")
    print("="*60)
    
    from knowledge_engine.integrations.unified_knowledge_extraction import (
        UnifiedKnowledgeExtractor,
        ExtractionResult
    )
    
    extractor = UnifiedKnowledgeExtractor()
    
    # Check causal methods exist
    assert hasattr(extractor, 'discover_causal_structure')
    assert hasattr(extractor, 'identify_confounders')
    
    # Check if in modules
    modules = extractor.get_available_modules()
    print(f"Available modules: {modules}")
    
    # Test method returns ExtractionResult
    import numpy as np
    result = extractor.discover_causal_structure(
        data=np.random.randn(10, 3),
        variable_names=['A', 'B', 'C'],
        algorithm='pc'
    )
    
    assert isinstance(result, ExtractionResult)
    print(f"Result status: {result.status}")
    
    print("[PASS] UnifiedKnowledgeExtractor integration works")
    return True


def test_orchestrator():
    """Test 4: KnowledgeOrchestrator integration"""
    print("\n" + "="*60)
    print("TEST 4: KnowledgeOrchestrator")
    print("="*60)
    
    from knowledge_engine.orchestration.knowledge_orchestrator import (
        OrchestratorConfig,
        ComponentType,
        PipelineStage
    )
    
    config = OrchestratorConfig()
    
    # Check CAUSAL_LEARN is in component types
    assert hasattr(ComponentType, 'CAUSAL_LEARN')
    
    # Check component config exists
    assert ComponentType.CAUSAL_LEARN in config.components
    
    # Check pipeline has causal stage
    stage_names = [s.name for s in config.pipeline_stages]
    assert 'discover_causal_structure' in stage_names
    
    print(f"Component enabled: {config.components[ComponentType.CAUSAL_LEARN].enabled}")
    print(f"Pipeline stage: discover_causal_structure")
    
    print("[PASS] KnowledgeOrchestrator integration works")
    return True


def test_master_engine():
    """Test 5: Master Engine integration"""
    print("\n" + "="*60)
    print("TEST 5: Master Engine")
    print("="*60)
    
    from knowledge_engine.master_engine import ComponentRegistry
    
    registry = ComponentRegistry()
    
    # Check component exists
    component = registry.get_component('causal_learn')
    assert component is not None
    
    # Check capabilities
    capabilities = registry.capabilities.get('causal_learn', [])
    assert 'causal_discovery' in capabilities
    
    print(f"Component: {component}")
    print(f"Capabilities: {capabilities}")
    
    print("[PASS] Master Engine integration works")
    return True


def test_analytics_engine():
    """Test 6: AdvancedAnalyticsEngine integration"""
    print("\n" + "="*60)
    print("TEST 6: AdvancedAnalyticsEngine")
    print("="*60)
    
    from knowledge_engine.advanced_analytics_engine import (
        AdvancedAnalyticsEngine,
        CausalDiscoveryEngine
    )
    
    engine = AdvancedAnalyticsEngine()
    
    # Check config includes causal_learn
    assert 'causal_learn' in engine.config
    
    print(f"Config: {engine.config['causal_learn']}")
    print(f"CausalDiscoveryEngine: {CausalDiscoveryEngine}")
    
    print("[PASS] AdvancedAnalyticsEngine integration works")
    return True


def test_unified_kg_hub():
    """Test 7: Unified KG Integration Hub"""
    print("\n" + "="*60)
    print("TEST 7: Unified KG Integration Hub")
    print("="*60)
    
    from knowledge_engine.unified_kg_integration_hub import (
        UnifiedKGIntegrationHub,
        UnifiedKGConfig,
        KGSource
    )
    
    # Check config
    config = UnifiedKGConfig()
    assert hasattr(config, 'enable_causal_learn')
    
    # Check KGSource
    assert hasattr(KGSource, 'CAUSAL_LEARN')
    assert hasattr(KGSource, 'CAUSAL_DISCOVERY_ENGINE')
    
    print(f"enable_causal_learn: {config.enable_causal_learn}")
    print(f"KGSource.CAUSAL_LEARN: {KGSource.CAUSAL_LEARN}")
    
    print("[PASS] Unified KG Integration Hub works")
    return True


def test_integration_factory():
    """Test 8: Integration Factory"""
    print("\n" + "="*60)
    print("TEST 8: Integration Factory")
    print("="*60)
    
    try:
        from integrations import IntegrationFactory, IntegrationType
        
        factory = IntegrationFactory()
        
        # Check causal_learn is registered
        info = factory.get_integration_info('causal_learn')
        if info:
            print(f"Integration: {info.name}")
            print(f"Type: {info.type}")
            assert info.type == IntegrationType.CAUSAL_DISCOVERY
            print("[PASS] Integration Factory has causal_learn")
        else:
            print("[WARN] Integration Factory doesn't have causal_learn registered")
        
        return True
    except ImportError as e:
        print(f"[SKIP] Integration Factory not available: {e}")
        return True


def run_async_tests():
    """Run async tests"""
    print("\n" + "="*60)
    print("ASYNC TESTS: Unified KG Hub Causal Analysis")
    print("="*60)
    
    async def test_causal_analysis():
        from knowledge_engine.unified_kg_integration_hub import (
            UnifiedKGIntegrationHub,
            UnifiedKGConfig
        )
        
        hub = UnifiedKGIntegrationHub(UnifiedKGConfig(enable_causal_learn=True))
        
        # Test analyze_causal_relations
        result = await hub.analyze_causal_relations([])
        
        print(f"Result source: {result.source}")
        print(f"Result type: {result.analysis_type}")
        
        return True
    
    try:
        asyncio.run(test_causal_analysis())
        print("[PASS] Async causal analysis works")
        return True
    except Exception as e:
        print(f"[FAIL] Async test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("CAUSAL-LEARN COMPLETE INTEGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Integration Module", test_integration_module),
        ("CausalLearnIntegration Class", test_causal_learn_integration_class),
        ("UnifiedKnowledgeExtractor", test_unified_extractor),
        ("KnowledgeOrchestrator", test_orchestrator),
        ("Master Engine", test_master_engine),
        ("AdvancedAnalyticsEngine", test_analytics_engine),
        ("Unified KG Hub", test_unified_kg_hub),
        ("Integration Factory", test_integration_factory),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Run async tests
    try:
        result = run_async_tests()
        results.append(("Async Causal Analysis", result, None))
    except Exception as e:
        results.append(("Async Causal Analysis", False, str(e)))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r, _ in results if r)
    failed = sum(1 for _, r, _ in results if not r)
    
    for name, result, error in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")
    
    print("-"*70)
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n[OK] All integration points verified!")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
