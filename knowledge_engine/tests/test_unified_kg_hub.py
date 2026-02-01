"""
Tests for Unified Knowledge Graph Integration Hub.

Following CLAUDE.md principles:
- Test all integration paths
- Test fallback behavior
- Test error handling
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

# Import the module under test
from knowledge_engine.unified_kg_integration_hub import (
    UnifiedKGIntegrationHub,
    UnifiedKGConfig,
    KnowledgeTriple,
    KGSource,
    create_unified_hub,
    quick_extract,
)


class TestKnowledgeTriple:
    """Tests for KnowledgeTriple dataclass."""
    
    def test_triple_creation(self):
        """Test creating a knowledge triple."""
        triple = KnowledgeTriple(
            subject="Alice",
            predicate="knows",
            object="Bob",
            confidence=0.95,
            source=KGSource.DEEPKE
        )
        
        assert triple.subject == "Alice"
        assert triple.predicate == "knows"
        assert triple.object == "Bob"
        assert triple.confidence == 0.95
        assert triple.source == KGSource.DEEPKE
    
    def test_triple_to_dict(self):
        """Test converting triple to dictionary."""
        triple = KnowledgeTriple(
            subject="Alice",
            predicate="knows",
            object="Bob",
            confidence=0.95,
            source=KGSource.DEEPKE,
            metadata={"extractor": "test"}
        )
        
        data = triple.to_dict()
        
        assert data["subject"] == "Alice"
        assert data["predicate"] == "knows"
        assert data["object"] == "Bob"
        assert data["confidence"] == 0.95
        assert data["source"] == "deepke"
        assert data["metadata"]["extractor"] == "test"
    
    def test_triple_from_dict(self):
        """Test creating triple from dictionary."""
        data = {
            "subject": "Alice",
            "predicate": "knows",
            "object": "Bob",
            "confidence": 0.95,
            "source": "deepke",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"extractor": "test"}
        }
        
        triple = KnowledgeTriple.from_dict(data)
        
        assert triple.subject == "Alice"
        assert triple.predicate == "knows"
        assert triple.object == "Bob"
        assert triple.confidence == 0.95
        assert triple.source == KGSource.DEEPKE


class TestUnifiedKGConfig:
    """Tests for UnifiedKGConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = UnifiedKGConfig()
        
        assert config.enable_deepke is True
        assert config.enable_oneke is True
        assert config.enable_kg_gen is True
        assert config.enable_neuralkg is True
        assert config.enable_graphiti is True
        assert config.default_backend == "memory"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = UnifiedKGConfig(
            enable_deepke=False,
            enable_z3=False,
            default_backend="memgraph"
        )
        
        assert config.enable_deepke is False
        assert config.enable_oneke is True
        assert config.enable_z3 is False
        assert config.default_backend == "memgraph"


class TestUnifiedKGIntegrationHub:
    """Tests for UnifiedKGIntegrationHub."""
    
    @pytest.mark.asyncio
    async def test_hub_initialization(self):
        """Test hub initialization."""
        hub = UnifiedKGIntegrationHub()
        
        assert hub._initialized is False
        
        result = await hub.initialize()
        
        assert result is True
        assert hub._initialized is True
    
    @pytest.mark.asyncio
    async def test_hub_initialization_idempotent(self):
        """Test that initialization is idempotent."""
        hub = UnifiedKGIntegrationHub()
        
        await hub.initialize()
        result = await hub.initialize()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_extract_knowledge_empty_text(self):
        """Test extraction with empty text."""
        hub = await create_unified_hub()
        
        triples = await hub.extract_knowledge("")
        
        assert isinstance(triples, list)
        assert len(triples) == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        hub = await create_unified_hub()
        
        health = await hub.health_check()
        
        assert "hub_status" in health
        assert "integrations" in health
        assert "statistics" in health
        assert "total_triples" in health["statistics"]
    
    @pytest.mark.asyncio
    async def test_export_import(self):
        """Test knowledge export and import."""
        hub = await create_unified_hub()
        
        # Add some test data
        triple = KnowledgeTriple(
            subject="Alice",
            predicate="knows",
            object="Bob",
            source=KGSource.KG_GEN
        )
        hub.triples.append(triple)
        
        # Export
        exported = hub.export_knowledge(format="dict")
        
        assert "triples" in exported
        assert len(exported["triples"]) == 1
        
        # Create new hub and import
        new_hub = await create_unified_hub()
        result = new_hub.import_knowledge(exported, format="dict")
        
        assert result is True
        assert len(new_hub.triples) == 1
    
    @pytest.mark.asyncio
    async def test_export_json(self):
        """Test JSON export."""
        hub = await create_unified_hub()
        
        triple = KnowledgeTriple(
            subject="Alice",
            predicate="knows",
            object="Bob"
        )
        hub.triples.append(triple)
        
        exported = hub.export_knowledge(format="json")
        
        assert isinstance(exported, str)
        assert "Alice" in exported
        assert "knows" in exported
        assert "Bob" in exported
    
    @pytest.mark.asyncio
    async def test_analyze_graph_without_karateclub(self):
        """Test graph analysis when KarateClub is not available."""
        config = UnifiedKGConfig(enable_karateclub=False)
        hub = await create_unified_hub(config)
        
        result = await hub.analyze_graph("community_detection")
        
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_evolve_knowledge_without_openevolve(self):
        """Test evolution when OpenEvolve is not available."""
        config = UnifiedKGConfig(enable_openevolve=False)
        hub = await create_unified_hub(config)
        
        result = await hub.evolve_knowledge()
        
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_verify_knowledge(self):
        """Test knowledge verification."""
        hub = await create_unified_hub()
        
        # Add test triples
        hub.triples.append(KnowledgeTriple(
            subject="Alice",
            predicate="knows",
            object="Bob"
        ))
        
        result = await hub.verify_knowledge()
        
        assert "verified" in result
        assert "contradictions" in result
        assert "uncertain" in result
    
    def test_triple_merge(self):
        """Test triple merging logic."""
        hub = UnifiedKGIntegrationHub()
        
        triples = [
            KnowledgeTriple("Alice", "knows", "Bob", 0.8, KGSource.DEEPKE),
            KnowledgeTriple("alice", "knows", "bob", 0.9, KGSource.ONEKE),
            KnowledgeTriple("Charlie", "knows", "Dave", 0.7, KGSource.KG_GEN),
        ]
        
        merged = hub._merge_triples(triples)
        
        # Should merge the first two (case-insensitive)
        assert len(merged) == 2
        
        # Should keep highest confidence
        alice_bob = [t for t in merged if t.subject.lower() == "alice"][0]
        assert alice_bob.confidence == 0.9


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_create_unified_hub(self):
        """Test create_unified_hub convenience function."""
        hub = await create_unified_hub()
        
        assert isinstance(hub, UnifiedKGIntegrationHub)
        assert hub._initialized is True
    
    @pytest.mark.asyncio
    async def test_quick_extract(self):
        """Test quick_extract convenience function."""
        # Note: This will use mock extractors if real ones aren't available
        triples = await quick_extract("Alice knows Bob.")
        
        assert isinstance(triples, list)


class TestKGSource:
    """Tests for KGSource enum."""
    
    def test_kg_source_values(self):
        """Test KGSource enum values."""
        assert KGSource.NEURALKG.value == "neuralkg"
        assert KGSource.DEEPKE.value == "deepke"
        assert KGSource.ONEKE.value == "oneke"
        assert KGSource.KG_GEN.value == "kg_gen"
        assert KGSource.GRAPHITI.value == "graphiti"
        assert KGSource.UNKNOWN.value == "unknown"
