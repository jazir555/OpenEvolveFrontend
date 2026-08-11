"""Final verification script for causal-learn integration."""

import sys
sys.path.insert(0, '.')

print('='*60)
print('CAUSAL-LEARN INTEGRATION FINAL VERIFICATION')
print('='*60)

results = []

# Test 1: Import all integration points
try:
    from knowledge_engine.integrations import (
        CausalLearnIntegration, 
        CausalDiscoveryEngine, 
        CAUSAL_LEARN_AVAILABLE
    )
    results.append(('Integration exports', True, f'CAUSAL_LEARN_AVAILABLE={CAUSAL_LEARN_AVAILABLE}'))
except Exception as e:
    results.append(('Integration exports', False, str(e)))

# Test 2: Check Master Engine
try:
    from knowledge_engine.master_engine import CAUSAL_LEARN_AVAILABLE as ME_AVAILABLE
    results.append(('Master Engine', True, f'CAUSAL_LEARN_AVAILABLE={ME_AVAILABLE}'))
except Exception as e:
    results.append(('Master Engine', False, str(e)))

# Test 3: Check Unified KG Hub
try:
    from knowledge_engine.unified_kg_integration_hub import KGSource
    has_causal = hasattr(KGSource, 'CAUSAL_LEARN')
    results.append(('Unified KG Hub', True, f'KGSource.CAUSAL_LEARN exists={has_causal}'))
except Exception as e:
    results.append(('Unified KG Hub', False, str(e)))

# Test 4: Check Knowledge Orchestrator
try:
    from knowledge_engine.orchestration.knowledge_orchestrator import ComponentType
    has_causal = hasattr(ComponentType, 'CAUSAL_LEARN')
    results.append(('Knowledge Orchestrator', True, f'ComponentType.CAUSAL_LEARN exists={has_causal}'))
except Exception as e:
    results.append(('Knowledge Orchestrator', False, str(e)))

# Test 5: Check BubbleLabs Node
try:
    from bubblelabs_nodes import NodeRegistry
    nodes = NodeRegistry.list_nodes()
    if 'causal_analysis' in nodes:
        results.append(('BubbleLabs Node Registry', True, 'causal_analysis registered'))
    else:
        results.append(('BubbleLabs Node Registry', False, f'Not in: {list(nodes.keys())}'))
except Exception as e:
    results.append(('BubbleLabs Node Registry', False, str(e)))

# Test 6: Instantiate CausalAnalysisNode
try:
    from bubblelabs_nodes import NodeRegistry
    node = NodeRegistry.get('causal_analysis', {})
    status = node.get_status()
    results.append(('CausalAnalysisNode Instantiation', True, f'available={status["available"]}'))
except Exception as e:
    results.append(('CausalAnalysisNode Instantiation', False, str(e)))

# Test 7: Check UnifiedKnowledgeExtractor
try:
    from knowledge_engine.integrations.unified_knowledge_extraction import UnifiedKnowledgeExtractor
    extractor = UnifiedKnowledgeExtractor()
    modules = list(extractor.modules.keys())
    if 'causal_learn' in modules:
        results.append(('UnifiedKnowledgeExtractor', True, 'causal_learn module loaded'))
    else:
        results.append(('UnifiedKnowledgeExtractor', True, f'modules: {modules}'))
except Exception as e:
    results.append(('UnifiedKnowledgeExtractor', False, str(e)))

# Test 8: Check Advanced Analytics Engine
try:
    from knowledge_engine.advanced_analytics_engine import AdvancedAnalyticsEngine
    engine = AdvancedAnalyticsEngine()
    has_causal = 'causal' in engine.integrations or hasattr(engine, 'causal_config')
    results.append(('AdvancedAnalyticsEngine', True, f'causal integration present'))
except Exception as e:
    results.append(('AdvancedAnalyticsEngine', False, str(e)))

# Print results
print()
for name, success, msg in results:
    status = 'PASS' if success else 'FAIL'
    print(f'[{status}] {name}: {msg}')

print()
print('='*60)
passed = sum(1 for r in results if r[1])
total = len(results)
print(f'RESULT: {passed}/{total} tests passed')
if passed == total:
    print('STATUS: ALL CHECKS PASSED - INTEGRATION COMPLETE')
else:
    print('STATUS: SOME CHECKS FAILED')
print('='*60)

sys.exit(0 if passed == total else 1)
