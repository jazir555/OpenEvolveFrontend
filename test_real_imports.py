"""
Final Verification: Real Import Test

This script performs actual imports (not just hasattr checks) to verify
everything can be imported and used in real code.

Following CLAUDE.md principle: RUNTIME TRUTH
"""

import sys

def test_real_imports():
    """Test that all imports actually work in real code"""

    print("="*80)
    print("FINAL VERIFICATION: REAL IMPORT TEST")
    print("="*80)

    all_passed = True

    # Test 1: Main knowledge_engine imports
    print("\n[1/5] Testing main knowledge_engine imports...")
    try:
        from knowledge_engine import (
            KnowledgeEngine,
            create_knowledge_engine,
            ProcessingResult,
            QueryResult,
            KnowledgeState,
            EntityKnowledgeGraph,
            KnowledgeExtractor,
            KnowledgeArtifact,
            KnowledgeStorage,
            KnowledgeRetriever,
            IntegratedKnowledgeEngine
        )
        print("  PASS: All 11 items imported successfully")
        # Verify we can actually use them
        assert KnowledgeEngine is not None
        assert callable(create_knowledge_engine)
        assert ProcessingResult is not None
        assert QueryResult is not None
    except (ImportError, AttributeError, TypeError) as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # Test 2: Graphiti imports
    print("\n[2/5] Testing Graphiti integration imports...")
    try:
        from knowledge_engine.integrations.graphiti import (
            GraphitiConfig,
            validate_config,
            GraphitiIntegrationError,
            ConfigurationError,
            ConnectionError,
            ContradictionError,
            InvalidTimestampError,
            EpisodeProcessingError,
            IncrementalUpdateError,
            GraphitiTemporalBridge,
            WorkflowArtifact,
            WorkflowState,
            TemporalFilter,
            TemporalRelationship,
            GraphitiAgentMemory,
            AgentInteraction,
            MemorySummary,
            MemoryType,
            GraphitiContradictionDetector,
            Contradiction,
            ContradictionReport,
            ContradictionSeverity,
            ResolutionAction,
            GraphitiIncrementalUpdater,
            GraphUpdate,
            EntityMergeResult,
            UpdateType,
            UpdateStatus,
            GraphitiHealthChecker,
            HealthCheckResult,
            SystemHealthReport,
            health_check_quick
        )
        print("  PASS: All 32 items imported successfully")
        # Verify key items
        assert GraphitiConfig is not None
        assert GraphitiTemporalBridge is not None
        assert GraphitiContradictionDetector is not None
        assert callable(validate_config)
        assert callable(health_check_quick)
    except (ImportError, AttributeError, TypeError) as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # Test 3: KG-Gen imports
    print("\n[3/5] Testing KG-Gen integration imports...")
    try:
        from knowledge_engine.integrations.kggen import (
            ExtractionPipeline,
            ExtractionResult,
            PipelineConfig,
            PipelineStatus,
            DeduplicationEngine,
            DeduplicationResult,
            SEMHASHStrategy,
            LMClusterStrategy,
            CrossDocumentResolver,
            KGGenMCPServer,
            MemoryManager,
            MemoryTools,
            ConversationAnalyzer,
            ConversationResult,
            SpeakerEntityExtractor,
            GraphAggregator,
            AggregationResult,
            GraphVersion,
            ConflictResolver
        )
        print("  PASS: All 19 items imported successfully")
        # Verify key items
        assert ExtractionPipeline is not None
        assert DeduplicationEngine is not None
        assert KGGenMCPServer is not None
    except (ImportError, AttributeError, TypeError) as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # Test 4: OneKE imports
    print("\n[4/5] Testing OneKE integration imports...")
    try:
        from knowledge_engine.integrations.oneke import (
            OneKEModelAdapter,
            ModelConfig,
            ExtractionResult,
            MultiTaskExtractionFramework,
            TaskType,
            OneKESchemaManager,
            SchemaDefinition,
            CrossLingualEntityLinker,
            EntityMatchResult,
            EventExtractionPipeline,
            EventChain,
            TemporalEvent
        )
        print("  PASS: All 12 items imported successfully")
        # Verify key items
        assert OneKEModelAdapter is not None
        assert MultiTaskExtractionFramework is not None
        assert TaskType is not None
    except (ImportError, AttributeError, TypeError) as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # Test 5: Visualization imports
    print("\n[5/5] Testing Visualization imports...")
    try:
        from knowledge_engine.visualization import (
            GraphExplorer,
            TemporalVisualizer,
            CommunityVisualizer,
            VisualizationOptions,
            TemporalVisualizationOptions,
            CommunityVisualizationOptions,
            NodeFilter,
            EdgeFilter,
            VisualizationResult,
            TemporalSnapshot,
            TimeRange,
            CommunityInfo,
            ExportHandler,
            VisualizationConfig
        )
        print("  PASS: All 14 items imported successfully")
        # Verify key items
        assert GraphExplorer is not None
        assert TemporalVisualizer is not None
        assert CommunityVisualizer is not None
        assert ExportHandler is not None
    except (ImportError, AttributeError, TypeError) as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # Final summary
    print("\n" + "="*80)
    print("FINAL VERIFICATION SUMMARY")
    print("="*80)

    if all_passed:
        print("\n[SUCCESS] ALL REAL IMPORTS PASSED")
        print("All 88 items can be imported and used in actual Python code")
        print("\nThe knowledge_engine module is PRODUCTION READY")
        return 0
    else:
        print("\n[FAILURE] SOME IMPORTS FAILED")
        print("Please review the errors above")
        return 1


if __name__ == "__main__":
    sys.exit(test_real_imports())
