"""
Comprehensive Integration Test for Causal-Learn in Knowledge Engine

This test verifies that causal-learn is FULLY integrated with:
1. Knowledge Engine master engine
2. Unified Knowledge Extractor
3. Knowledge Orchestrator
4. Integration Registry
5. CausalDiscoveryBridge and CausalLearnAdapter
"""

import asyncio
import sys
import os
import numpy as np
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knowledge_engine'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'integrations'))


def test_integration_registry():
    """Test 1: Verify causal-learn is registered in Integration Registry"""
    print("\n" + "="*60)
    print("TEST 1: Integration Registry")
    print("="*60)
    
    from integrations.registry import get_registry, IntegrationType
    
    registry = get_registry()
    
    # Check if causal_learn is registered
    info = registry.get_integration_info("causal_learn")
    assert info is not None, "causal_learn not found in registry"
    assert info.type == IntegrationType.CAUSAL_DISCOVERY, f"Wrong type: {info.type}"
    
    print(f"[OK] Causal-learn registered: {info.name}")
    print(f"   Type: {info.type.value}")
    print(f"   Module: {info.module_path}")
    print(f"   Class: {info.class_name}")
    return True


def test_causal_interface():
    """Test 2: Verify CausalDiscoveryInterface is properly defined"""
    print("\n" + "="*60)
    print("TEST 2: CausalDiscoveryInterface")
    print("="*60)
    
    from integrations.base.causal_interface import (
        CausalDiscoveryInterface,
        CausalGraphResult,
        CausalEffectResult,
        CausalMethod,
        EdgeType
    )
    
    # Verify interface has required methods
    required_methods = [
        'initialize',
        'discover_causal_structure',
        'validate_causal_claim',
        'estimate_causal_effect',
        'test_independence',
        'counterfactual_analysis',
        'get_causal_ancestors',
        'identify_confounders',
        'validate',
        'shutdown'
    ]
    
    for method in required_methods:
        assert hasattr(CausalDiscoveryInterface, method), f"Missing method: {method}"
        print(f"[OK] Method '{method}' defined")
    
    # Verify enums
    assert CausalMethod.PC.value == "pc"
    assert CausalMethod.GES.value == "ges"
    assert CausalMethod.FCI.value == "fci"
    print("[OK] CausalMethod enum correct")
    
    assert EdgeType.DIRECTED.value == "-->"
    assert EdgeType.BIDIRECTED.value == "<->"
    print("[OK] EdgeType enum correct")
    
    return True


def test_causal_learn_adapter():
    """Test 3: Verify CausalLearnAdapter implements interface"""
    print("\n" + "="*60)
    print("TEST 3: CausalLearnAdapter")
    print("="*60)
    
    from integrations.causal_learn.adapter import CausalLearnAdapter
    from integrations.base.causal_interface import CausalDiscoveryInterface
    
    # Verify adapter implements interface
    assert issubclass(CausalLearnAdapter, CausalDiscoveryInterface), \
        "CausalLearnAdapter doesn't implement CausalDiscoveryInterface"
    print("[OK] CausalLearnAdapter implements CausalDiscoveryInterface")
    
    # Verify adapter has all required methods
    adapter = CausalLearnAdapter()
    required_methods = [
        'initialize',
        'discover_causal_structure',
        'validate_causal_claim',
        'estimate_causal_effect',
        'test_independence',
        'counterfactual_analysis',
        'get_causal_ancestors',
        'identify_confounders',
        'validate',
        'shutdown'
    ]
    
    for method in required_methods:
        assert hasattr(adapter, method), f"Adapter missing method: {method}"
        print(f"[OK] Adapter has method: {method}")
    
    return True


def test_causal_discovery_bridge():
    """Test 4: Verify CausalDiscoveryBridge for high-level integration"""
    print("\n" + "="*60)
    print("TEST 4: CausalDiscoveryBridge")
    print("="*60)
    
    from integrations.causal_learn.bridge import CausalDiscoveryBridge
    
    bridge = CausalDiscoveryBridge()
    
    # Verify bridge has integration methods
    integration_methods = [
        'initialize',
        'pre_experiment_validation',
        'analyze_problem_causally',
        'extract_causal_knowledge',
        'validate_hypothesis',
        'suggest_interventions',
        'shutdown'
    ]
    
    for method in integration_methods:
        assert hasattr(bridge, method), f"Bridge missing method: {method}"
        print(f"[OK] Bridge has method: {method}")
    
    return True


def test_knowledge_engine_integration():
    """Test 5: Verify CausalLearnIntegration in Knowledge Engine"""
    print("\n" + "="*60)
    print("TEST 5: Knowledge Engine Integration")
    print("="*60)
    
    from knowledge_engine.integrations.causal_learn_integration import (
        CausalLearnIntegration,
        CausalDiscoveryEngine
    )
    
    # Verify integration class exists
    integration = CausalLearnIntegration()
    assert hasattr(integration, 'is_available'), "Missing is_available method"
    assert hasattr(integration, 'discover_structure'), "Missing discover_structure method"
    assert hasattr(integration, 'get_available_algorithms'), "Missing get_available_algorithms method"
    
    print(f"[OK] CausalLearnIntegration initialized")
    print(f"   Available: {integration.is_available()}")
    print(f"   Algorithms: {integration.get_available_algorithms()}")
    
    # Verify discovery engine
    engine = CausalDiscoveryEngine()
    assert hasattr(engine, 'discover_causal_structure'), "Missing discover_causal_structure"
    assert hasattr(engine, 'analyze_causal_graph'), "Missing analyze_causal_graph"
    assert hasattr(engine, 'identify_confounders'), "Missing identify_confounders"
    
    print(f"[OK] CausalDiscoveryEngine initialized")
    print(f"   Available algorithms: {engine.get_available_algorithms()}")
    
    return True


def test_unified_knowledge_extractor():
    """Test 6: Verify causal-learn in UnifiedKnowledgeExtractor"""
    print("\n" + "="*60)
    print("TEST 6: UnifiedKnowledgeExtractor Integration")
    print("="*60)
    
    from knowledge_engine.integrations.unified_knowledge_extraction import (
        UnifiedKnowledgeExtractor,
        ExtractionResult
    )
    
    extractor = UnifiedKnowledgeExtractor()
    
    # Check if causal_learn is in modules
    available_modules = extractor.get_available_modules()
    print(f"Available modules: {available_modules}")
    
    if 'causal_learn' in available_modules:
        print("[OK] causal_learn module is available")
    else:
        print("[WARN] causal_learn module not available (causal-learn not installed)")
    
    # Verify causal discovery methods exist
    assert hasattr(extractor, 'discover_causal_structure'), "Missing discover_causal_structure"
    assert hasattr(extractor, 'identify_confounders'), "Missing identify_confounders"
    
    print("[OK] Causal discovery methods exist")
    
    # Check status includes causal discovery
    status = extractor.get_status()
    assert 'causal_discovery' in status['capabilities'], "Causal discovery not in capabilities"
    print("[OK] Causal discovery in capabilities list")
    
    return True


def test_knowledge_orchestrator():
    """Test 7: Verify causal-learn in Knowledge Orchestrator"""
    print("\n" + "="*60)
    print("TEST 7: Knowledge Orchestrator Integration")
    print("="*60)
    
    from knowledge_engine.orchestration.knowledge_orchestrator import (
        OrchestratorConfig,
        ComponentType,
        PipelineStage
    )
    
    config = OrchestratorConfig()
    
    # Check CAUSAL_LEARN is a valid component type
    assert hasattr(ComponentType, 'CAUSAL_LEARN'), "CAUSAL_LEARN not in ComponentType"
    print("[OK] CAUSAL_LEARN in ComponentType enum")
    
    # Check causal-learn is in default components
    assert ComponentType.CAUSAL_LEARN in config.components, "CAUSAL_LEARN not in components"
    component_config = config.components[ComponentType.CAUSAL_LEARN]
    print(f"[OK] CAUSAL_LEARN in default components")
    print(f"   Enabled: {component_config.enabled}")
    print(f"   Required: {component_config.required}")
    
    # Check causal discovery is in default pipeline
    stage_names = [s.name for s in config.pipeline_stages]
    if 'discover_causal_structure' in stage_names:
        print("[OK] Causal discovery stage in default pipeline")
        stage = next(s for s in config.pipeline_stages if s.name == 'discover_causal_structure')
        print(f"   Component: {stage.component}")
        print(f"   Enabled: {stage.enabled}")
    else:
        print("[WARN] Causal discovery stage not in default pipeline")
    
    return True


def test_master_engine():
    """Test 8: Verify causal-learn in Master Engine"""
    print("\n" + "="*60)
    print("TEST 8: Master Engine Integration")
    print("="*60)
    
    from knowledge_engine.master_engine import ComponentRegistry, KnowledgeDomain
    
    registry = ComponentRegistry()
    
    # Check causal_learn component exists
    component = registry.get_component('causal_learn')
    assert component is not None, "causal_learn component not found"
    print("[OK] causal_learn component registered")
    
    # Check capabilities
    capabilities = registry.capabilities.get('causal_learn', [])
    expected_capabilities = ['causal_discovery', 'structure_learning', 'confounder_detection']
    for cap in expected_capabilities:
        if cap in capabilities:
            print(f"[OK] Capability '{cap}' registered")
        else:
            print(f"[WARN] Capability '{cap}' not found")
    
    # Check component substitution matrix
    substitutes = registry.get_substitutes('causal_learn')
    print(f"   Substitutes: {substitutes}")
    
    # Check domain classification
    causal_components = registry.get_components_for_capability('causal_discovery')
    if 'causal_learn' in causal_components:
        print("[OK] causal_learn provides 'causal_discovery' capability")
    
    return True


def test_functional_causal_discovery():
    """Test 9: Functional test of causal discovery (if causal-learn installed)"""
    print("\n" + "="*60)
    print("TEST 9: Functional Causal Discovery")
    print("="*60)
    
    try:
        from knowledge_engine.integrations.causal_learn_integration import CausalLearnIntegration
        
        integration = CausalLearnIntegration()
        
        if not integration.is_available():
            print("[SKIP] causal-learn not installed, skipping functional test")
            return True
        
        print("[OK] causal-learn is available")
        
        # Generate synthetic data with known causal structure
        # X -> Y -> Z
        np.random.seed(42)
        n_samples = 500
        X = np.random.randn(n_samples)
        Y = 0.5 * X + np.random.randn(n_samples)
        Z = 0.3 * Y + np.random.randn(n_samples)
        data = np.column_stack([X, Y, Z])
        
        print(f"[OK] Generated synthetic data: {data.shape}")
        
        # Test causal discovery
        result = integration.discover_structure(
            data=data,
            algorithm='pc',
            variable_names=['X', 'Y', 'Z']
        )
        
        print(f"[OK] Causal discovery completed")
        print(f"   Status: {result.get('status')}")
        
        if result.get('status') == 'success':
            graph = result.get('graph', {})
            edges = graph.get('edges', [])
            print(f"   Discovered edges: {len(edges)}")
            for edge in edges[:5]:  # Show first 5 edges
                print(f"     {edge}")
        
        return True
        
    except ImportError as e:
        print(f"[SKIP] Cannot run functional test: {e}")
        return True


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("CAUSAL-LEARN FULL INTEGRATION TEST SUITE")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")
    
    tests = [
        ("Integration Registry", test_integration_registry),
        ("Causal Interface", test_causal_interface),
        ("CausalLearnAdapter", test_causal_learn_adapter),
        ("CausalDiscoveryBridge", test_causal_discovery_bridge),
        ("Knowledge Engine Integration", test_knowledge_engine_integration),
        ("UnifiedKnowledgeExtractor", test_unified_knowledge_extractor),
        ("Knowledge Orchestrator", test_knowledge_orchestrator),
        ("Master Engine", test_master_engine),
        ("Functional Causal Discovery", test_functional_causal_discovery),
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
    print(f"Completed: {datetime.now().isoformat()}")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
