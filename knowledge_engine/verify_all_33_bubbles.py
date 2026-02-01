"""Verify ALL 33 Knowledge Engine bubbles can be imported and initialized."""
import sys
sys.path.insert(0, '..')

print('='*70)
print('VERIFYING ALL 33 KNOWLEDGE ENGINE BUBBLES')
print('='*70)

bubbles = [
    # Original 21 Bubbles
    # Category 1: Core Knowledge Operations (5)
    ('knowledge_extraction_node', 'KnowledgeExtractionNode'),
    ('knowledge_query_node', 'KnowledgeQueryNode'),
    ('knowledge_reasoning_node', 'KnowledgeReasoningNode'),
    ('knowledge_integration_node', 'KnowledgeIntegrationNode'),
    ('entity_profile_node', 'EntityProfileNode'),
    # Category 2: Advanced Analytics (5)
    ('temporal_knowledge_node', 'TemporalKnowledgeNode'),
    ('pattern_mining_node', 'PatternMiningNode'),
    ('semantic_search_node', 'SemanticSearchNode'),
    ('causal_analysis_node', 'CausalAnalysisNode'),
    ('knowledge_evolution_node', 'KnowledgeEvolutionNode'),
    # Category 3: Quality & Management (5)
    ('deduplication_node', 'DeduplicationNode'),
    ('contradiction_detection_node', 'ContradictionDetectionNode'),
    ('knowledge_analytics_node', 'KnowledgeAnalyticsNode'),
    ('knowledge_validation_node', 'KnowledgeValidationNode'),
    ('knowledge_import_export_node', 'KnowledgeImportExportNode'),
    # Category 4: Intelligent Operations (6)
    ('knowledge_learning_node', 'KnowledgeLearningNode'),
    ('quality_assurance_node', 'QualityAssuranceNode'),
    ('knowledge_summarization_node', 'KnowledgeSummarizationNode'),
    ('change_detection_node', 'ChangeDetectionNode'),
    ('knowledge_enrichment_node', 'KnowledgeEnrichmentNode'),
    ('knowledge_alerting_node', 'KnowledgeAlertingNode'),
    # NEW: 12 Additional Bubbles
    # Category 5: User Interface (2)
    ('natural_language_interface_node', 'NaturalLanguageInterfaceNode'),
    ('knowledge_visualization_node', 'KnowledgeVisualizationNode'),
    # Category 6: Intelligence & Recommendations (2)
    ('recommendation_engine_node', 'RecommendationEngineNode'),
    ('explainability_node', 'ExplainabilityNode'),
    # Category 7: Production & Operations (4)
    ('version_control_node', 'VersionControlNode'),
    ('backup_recovery_node', 'BackupRecoveryNode'),
    ('security_compliance_node', 'SecurityComplianceNode'),
    ('streaming_ingestion_node', 'StreamingIngestionNode'),
    # Category 8: Advanced AI (4)
    ('bias_detection_node', 'BiasDetectionNode'),
    ('uncertainty_quantification_node', 'UncertaintyQuantificationNode'),
    ('knowledge_federation_node', 'KnowledgeFederationNode'),
    ('workflow_orchestration_node', 'WorkflowOrchestrationNode'),
]

success = 0
failures = []

for module, class_name in bubbles:
    try:
        module_obj = __import__(f'bubblelabs_nodes.{module}', fromlist=[class_name])
        node_class = getattr(module_obj, class_name)
        node = node_class()
        name = node.get_display_name()
        print(f'[OK] {class_name}: {name}')
        success += 1
    except Exception as e:
        print(f'[FAIL] {class_name}: {e}')
        failures.append((class_name, str(e)))

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

total_size = sum(os.path.getsize(f'../bubblelabs_nodes/{m}.py') for m, _ in bubbles if os.path.exists(f'../bubblelabs_nodes/{m}.py'))
print(f'\nTotal Size: {total_size:,} bytes ({total_size/1024:.1f} KB) ({total_size/(1024*1024):.1f} MB)')

if failures:
    print()
    print('Failures:')
    for name, error in failures:
        print(f'  {name}: {error}')
else:
    print()
    print('All 33 bubbles created and verified successfully!')
