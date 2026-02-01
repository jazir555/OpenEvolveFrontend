"""Verify all Knowledge Engine bubbles can be imported and initialized."""
import sys
sys.path.insert(0, '..')

print('='*70)
print('VERIFYING ALL KNOWLEDGE ENGINE BUBBLES')
print('='*70)

bubbles = [
    ('knowledge_extraction_node', 'KnowledgeExtractionNode'),
    ('knowledge_query_node', 'KnowledgeQueryNode'),
    ('knowledge_reasoning_node', 'KnowledgeReasoningNode'),
    ('knowledge_integration_node', 'KnowledgeIntegrationNode'),
    ('entity_profile_node', 'EntityProfileNode'),
    ('temporal_knowledge_node', 'TemporalKnowledgeNode'),
    ('pattern_mining_node', 'PatternMiningNode'),
    ('semantic_search_node', 'SemanticSearchNode'),
    ('causal_analysis_node', 'CausalAnalysisNode'),
    ('knowledge_evolution_node', 'KnowledgeEvolutionNode'),
    ('deduplication_node', 'DeduplicationNode'),
    ('contradiction_detection_node', 'ContradictionDetectionNode'),
    ('knowledge_analytics_node', 'KnowledgeAnalyticsNode'),
    ('knowledge_validation_node', 'KnowledgeValidationNode'),
    ('knowledge_import_export_node', 'KnowledgeImportExportNode'),
]

success = 0
for module, class_name in bubbles:
    try:
        # Dynamic import
        module_obj = __import__(f'bubblelabs_nodes.{module}', fromlist=[class_name])
        node_class = getattr(module_obj, class_name)
        node = node_class()
        name = node.get_display_name()
        print(f'[OK] {class_name}: {name}')
        success += 1
    except Exception as e:
        print(f'[FAIL] {class_name}: {e}')

print()
print(f'Result: {success}/{len(bubbles)} bubbles imported successfully')
print('='*70)

# List all bubble files with sizes
print()
print('Bubble Files:')
import os
for module, _ in bubbles:
    path = f'../bubblelabs_nodes/{module}.py'
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f'  {module}.py: {size:,} bytes')
