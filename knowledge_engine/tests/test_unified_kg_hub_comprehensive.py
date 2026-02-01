"""
Comprehensive Tests for Unified Knowledge Graph Integration Hub.

Tests all 30+ integrations and their interactions.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from knowledge_engine.unified_kg_integration_hub import (
    UnifiedKGIntegrationHub,
    UnifiedKGConfig,
    KnowledgeTriple,
    KGSource,
    IntegrationRegistry,
    ExtractionResult,
    AnalysisResult,
    create_unified_hub,
    quick_extract,
)


class TestKGSourceComprehensive:
    """Comprehensive tests for KGSource enum."""
    
    def test_all_sources_defined(self):
        """Test that all 30+ sources are defined."""
        sources = list(KGSource)
        
        # Knowledge Extraction (6)
        assert KGSource.DEEPKE in sources
        assert KGSource.ONEKE in sources
        assert KGSource.KG_GEN in sources
        assert KGSource.AI_KG in sources
        assert KGSource.AGENTJSON in sources
        assert KGSource.UNIFIED_EXTRACTION in sources
        
        # Neural & Embeddings (3)
        assert KGSource.NEUROML in sources
        assert KGSource.KARATECLUB in sources
        
        # Reasoning & Verification (4)
        assert KGSource.Z3 in sources
        assert KGSource.LEANAIDE in sources
        assert KGSource.LEANAIDE_PROOF in sources
        assert KGSource.DSPY in sources
        
        # Temporal & Causal (2)
        assert KGSource.GRAPHITI in sources
        assert KGSource.CAUSAL_LEARN in sources
        
        # Agent & Workflow (5)
        assert KGSource.OPENEVOLVE in sources
        assert KGSource.CREWAI in sources
        assert KGSource.LOONGFLOW in sources
        assert KGSource.RESEARCH_QUEST in sources
        assert KGSource.AGENTIC_CONTEXT in sources
        
        # Domain Specific (3)
        assert KGSource.GLOBAL_CHEM in sources
        assert KGSource.LAGRANGE_MAPPER in sources
        assert KGSource.PAMI in sources
        
        # Data & Retrieval (2)
        assert KGSource.RAGBITS in sources
        assert KGSource.MEMORY_FUSION in sources
        
        # Integration & Gateway (1)
        assert KGSource.MCP_GATEWAY in sources
    
    def test_source_values(self):
        """Test source enum values."""
        assert KGSource.DEEPKE.value == "deepke"
        assert KGSource.Z3.value == "z3"
        assert KGSource.OPENEVOLVE.value == "openevolve"
        assert KGSource.GRAPHITI.value == "graphiti"


class TestUnifiedKGConfigComprehensive:
    """Comprehensive tests for UnifiedKGConfig."""
    
    def test_all_integrations_configurable(self):
        """Test that all 30+ integrations can be configured."""
        config = UnifiedKGConfig()
        
        # Knowledge Extraction
        assert hasattr(config, 'enable_deepke')
        assert hasattr(config, 'enable_oneke')
        assert hasattr(config, 'enable_kg_gen')
        assert hasattr(config, 'enable_ai_kg')
        assert hasattr(config, 'enable_agentjson')
        assert hasattr(config, 'enable_unified_extraction')
        
        # Neural & Embeddings
        assert hasattr(config, 'enable_neuralkg')
        assert hasattr(config, 'enable_karateclub')
        assert hasattr(config, 'enable_neuromancer')
        
        # Reasoning & Verification
        assert hasattr(config, 'enable_z3')
        assert hasattr(config, 'enable_leanaide')
        assert hasattr(config, 'enable_leanaide_proof')
        assert hasattr(config, 'enable_dspy')
        
        # Temporal & Causal
        assert hasattr(config, 'enable_graphiti')
        assert hasattr(config, 'enable_causal_learn')
        
        # Agent & Workflow
        assert hasattr(config, 'enable_openevolve')
        assert hasattr(config, 'enable_crewai')
        assert hasattr(config, 'enable_loongflow')
        assert hasattr(config, 'enable_research_quest')
        assert hasattr(config, 'enable_agentic_context')
        
        # Domain Specific
        assert hasattr(config, 'enable_global_chem')
        assert hasattr(config, 'enable_lagrange_mapper')
        assert hasattr(config, 'enable_pami')
        
        # Data & Retrieval
        assert hasattr(config, 'enable_ragbits')
        assert hasattr(config, 'enable_memory_fusion')
        
        # Integration & Gateway
        assert hasattr(config, 'enable_mcp_gateway')
    
    def test_selective_disable(self):
        """Test selectively disabling integrations."""
        config = UnifiedKGConfig(
            enable_deepke=False,
            enable_z3=False,
            enable_openevolve=False,
            enable_causal_learn=True
        )
        
        assert config.enable_deepke is False
        assert config.enable_z3 is False
        assert config.enable_openevolve is False
        assert config.enable_causal_learn is True
        assert config.enable_oneke is True  # Default


class TestIntegrationRegistry:
    """Tests for IntegrationRegistry."""
    
    @pytest.mark.asyncio
    async def test_registry_creation(self):
        """Test registry initialization."""
        registry = IntegrationRegistry()
        
        assert len(registry._initializers) > 20  # At least 20 integrations
    
    @pytest.mark.asyncio
    async def test_get_uninitialized_integration(self):
        """Test getting an integration that hasn't been initialized."""
        registry = IntegrationRegistry()
        
        # Should return None or integration without error
        result = await registry.get("nonexistent_integration")
        assert result is None


class TestUnifiedKGIntegrationHubComprehensive:
    """Comprehensive tests for UnifiedKGIntegrationHub."""
    
    @pytest.mark.asyncio
    async def test_hub_with_minimal_config(self):
        """Test hub with minimal configuration."""
        config = UnifiedKGConfig(
            enable_deepke=False,
            enable_oneke=False,
            enable_kg_gen=False,
            enable_neuralkg=False,
            enable_karateclub=True,  # Keep one for testing
        )
        
        hub = UnifiedKGIntegrationHub(config)
        result = await hub.initialize()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_comprehensive_extraction(self):
        """Test extraction with multiple extractors."""
        config = UnifiedKGConfig(
            enable_deepke=False,  # Disable to avoid heavy dependencies
            enable_oneke=False,
            enable_kg_gen=False,
        )
        
        hub = await create_unified_hub(config)
        
        # Should work even with no extractors
        triples = await hub.extract_knowledge("Test text")
        assert isinstance(triples, list)
    
    @pytest.mark.asyncio
    async def test_causal_analysis(self):
        """Test causal relation analysis."""
        config = UnifiedKGConfig(enable_causal_learn=False)
        hub = await create_unified_hub(config)
        
        result = await hub.analyze_causal_relations([])
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "causal_discovery"
    
    @pytest.mark.asyncio
    async def test_pattern_mining(self):
        """Test pattern mining."""
        config = UnifiedKGConfig(enable_pami=False)
        hub = await create_unified_hub(config)
        
        result = await hub.mine_patterns(min_support=0.1)
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "pattern_mining"
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self):
        """Test comprehensive health check."""
        hub = await create_unified_hub()
        
        health = await hub.health_check()
        
        assert "hub_status" in health
        assert "initialized_integrations" in health
        assert "integration_count" in health
        assert "statistics" in health
        assert "total_triples" in health["statistics"]
        assert "total_entities" in health["statistics"]
        assert "total_relations" in health["statistics"]
        assert "total_patterns" in health["statistics"]
    
    def test_triple_merge_comprehensive(self):
        """Test comprehensive triple merging."""
        hub = UnifiedKGIntegrationHub()
        
        # Test case sensitivity
        triples = [
            KnowledgeTriple("Alice", "knows", "Bob", 0.8, KGSource.DEEPKE),
            KnowledgeTriple("alice", "knows", "bob", 0.9, KGSource.ONEKE),
            KnowledgeTriple("ALICE", "KNOWS", "BOB", 0.7, KGSource.KG_GEN),
            KnowledgeTriple("Charlie", "knows", "Dave", 0.95, KGSource.AI_KG),
        ]
        
        merged = hub._merge_triples(triples)
        
        # Should merge all Alice-Bob variants into one
        assert len(merged) == 2
        
        # Should keep highest confidence
        alice_bob = [t for t in merged if t.subject.lower() == "alice"][0]
        assert alice_bob.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_export_import_comprehensive(self):
        """Test comprehensive export/import."""
        hub = await create_unified_hub()
        
        # Add comprehensive test data
        hub.triples.append(KnowledgeTriple("A", "rel", "B", 0.9, KGSource.DEEPKE))
        hub.entities["A"] = {"type": "person"}
        hub.relations["rel"] = {"type": "association"}
        hub.patterns.append({"pattern": "test", "support": 0.5})
        
        # Export
        exported = hub.export_knowledge(format="dict")
        
        assert "triples" in exported
        assert "entities" in exported
        assert "relations" in exported
        assert "patterns" in exported
        assert "export_info" in exported
        
        # Import to new hub
        new_hub = await create_unified_hub()
        result = new_hub.import_knowledge(exported)
        
        assert result is True
        assert len(new_hub.triples) == 1
        assert len(new_hub.entities) == 1
        assert len(new_hub.patterns) == 1


class TestKnowledgeTripleAdvanced:
    """Advanced tests for KnowledgeTriple."""
    
    def test_triple_with_all_sources(self):
        """Test creating triples with all source types."""
        for source in KGSource:
            triple = KnowledgeTriple(
                subject="Test",
                predicate="test_relation",
                object="Test2",
                confidence=0.9,
                source=source,
                metadata={"source_type": source.value}
            )
            
            assert triple.source == source
            data = triple.to_dict()
            assert data["source"] == source.value
    
    def test_triple_serialization_roundtrip(self):
        """Test full serialization roundtrip."""
        original = KnowledgeTriple(
            subject="Alice",
            predicate="works_at",
            object="Acme Corp",
            confidence=0.95,
            source=KGSource.ONEKE,
            metadata={
                "extractor_version": "1.0",
                "document_id": "doc_123",
                "nested": {"key": "value"}
            }
        )
        
        # Serialize
        data = original.to_dict()
        
        # Deserialize
        restored = KnowledgeTriple.from_dict(data)
        
        assert restored.subject == original.subject
        assert restored.predicate == original.predicate
        assert restored.object == original.object
        assert restored.confidence == original.confidence
        assert restored.source == original.source
        assert restored.metadata == original.metadata


class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when integrations fail."""
        config = UnifiedKGConfig()
        hub = UnifiedKGIntegrationHub(config)
        
        # Should not raise even with unavailable integrations
        result = await hub.initialize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_invalid_analysis_type(self):
        """Test handling of invalid analysis type."""
        hub = await create_unified_hub()
        
        result = await hub.analyze_graph("invalid_type")
        
        # Should handle gracefully
        assert isinstance(result, AnalysisResult)
    
    def test_import_invalid_data(self):
        """Test import with invalid data."""
        hub = UnifiedKGIntegrationHub()
        
        result = hub.import_knowledge("invalid json", format="json")
        
        assert result is False
    
    def test_convert_invalid_triple(self):
        """Test converting invalid data to triple."""
        hub = UnifiedKGIntegrationHub()
        
        result = hub._convert_to_triple("invalid", KGSource.DEEPKE)
        
        assert result is None


class TestPerformance:
    """Performance-related tests."""
    
    @pytest.mark.asyncio
    async def test_large_triple_merge(self):
        """Test merging large numbers of triples."""
        hub = UnifiedKGIntegrationHub()
        
        # Create 1000 triples
        triples = [
            KnowledgeTriple(
                subject=f"Entity{i}",
                predicate="rel",
                object=f"Entity{i+1}",
                confidence=0.5 + (i % 50) / 100
            )
            for i in range(1000)
        ]
        
        import time
        start = time.time()
        merged = hub._merge_triples(triples)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds
        assert len(merged) == 1000  # All unique


# Mark tests that require heavy dependencies
pytestmark = [
    pytest.mark.asyncio,
]
