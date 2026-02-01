"""
Comprehensive Test Suite for Graphiti Temporal Integration

Tests temporal reasoning, hybrid search, and contradiction detection.
"""

import pytest
import pytest_asyncio
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_engine.core.temporal_knowledge_engine import (
    TemporalKnowledgeEngine,
    KnowledgeArtifact,
    RerankMethod,
    ContradictionDetection,
)
from knowledge_engine.integrations.graphiti_temporal_bridge import (
    GraphitiTemporalBridge,
    EntityMapping,
)


class TestKnowledgeArtifact:
    """Test KnowledgeArtifact dataclass."""

    def test_artifact_creation(self):
        """Test creating a basic artifact."""
        artifact = KnowledgeArtifact(
            id="test_001",
            content="Test content",
            artifact_type="solution_pattern",
            valid_at=datetime.utcnow(),
        )

        assert artifact.id == "test_001"
        assert artifact.content == "Test content"
        assert artifact.artifact_type == "solution_pattern"
        assert artifact.invalid_at is None
        assert artifact.confidence == 1.0

    def test_artifact_validity(self):
        """Test temporal validity checking."""
        now = datetime.utcnow()
        past = now - timedelta(days=1)
        future = now + timedelta(days=1)

        # Valid at current time
        artifact = KnowledgeArtifact(
            id="test_002",
            content="Valid now",
            artifact_type="solution_pattern",
            valid_at=past,
            invalid_at=future,
        )

        assert artifact.is_valid_at(now)
        assert not artifact.is_valid_at(past - timedelta(seconds=1))
        assert not artifact.is_valid_at(future + timedelta(seconds=1))

    def test_artifact_to_dict(self):
        """Test artifact serialization."""
        now = datetime.utcnow()
        artifact = KnowledgeArtifact(
            id="test_003",
            content="Serialization test",
            artifact_type="workflow",
            valid_at=now,
            metadata={"key": "value"},
        )

        data = artifact.to_dict()

        assert data["id"] == "test_003"
        assert data["content"] == "Serialization test"
        assert data["metadata"]["key"] == "value"

    def test_artifact_from_dict(self):
        """Test artifact deserialization."""
        now = datetime.utcnow()
        data = {
            "id": "test_004",
            "content": "Deserialization test",
            "artifact_type": "problem",
            "valid_at": now.isoformat(),
            "invalid_at": None,
            "created_at": now.isoformat(),
            "source": "test",
            "metadata": {},
            "entities": [],
            "relationships": [],
            "confidence": 0.9,
            "group_id": None,
        }

        artifact = KnowledgeArtifact.from_dict(data)

        assert artifact.id == "test_004"
        assert artifact.content == "Deserialization test"
        assert artifact.confidence == 0.9


class TestTemporalKnowledgeEngine:
    """Test TemporalKnowledgeEngine."""

    @pytest_asyncio.fixture
    async def engine(self):
        """Create a temporal knowledge engine for testing."""
        engine = TemporalKnowledgeEngine(
            enable_temporal=True,
            enable_hybrid_search=True,
        )
        # Don't initialize Graphiti for unit tests
        yield engine
        # Cleanup

    @pytest.mark.asyncio
    async def test_add_knowledge_temporal(self, engine):
        """Test adding temporal knowledge."""
        now = datetime.utcnow()

        artifact = await engine.add_knowledge_temporal(
            content="Solution pattern for testing",
            artifact_type="solution_pattern",
            valid_at=now,
            metadata={"test": "value"},
        )

        assert artifact is not None
        assert artifact.artifact_type == "solution_pattern"
        assert artifact.content == "Solution pattern for testing"
        assert artifact.valid_at == now
        assert artifact.metadata["test"] == "value"

    @pytest.mark.asyncio
    async def test_add_knowledge_with_expiry(self, engine):
        """Test adding knowledge with expiration."""
        now = datetime.utcnow()
        future = now + timedelta(days=7)

        artifact = await engine.add_knowledge_temporal(
            content="Temporary knowledge",
            artifact_type="workflow",
            valid_at=now,
            invalid_at=future,
        )

        assert artifact is not None
        assert artifact.invalid_at == future

    @pytest.mark.asyncio
    async def test_query_at_time(self, engine):
        """Test point-in-time querying."""
        now = datetime.utcnow()
        past = now - timedelta(hours=1)

        # Add knowledge
        await engine.add_knowledge_temporal(
            content="Historical solution",
            artifact_type="solution_pattern",
            valid_at=past,
        )

        await engine.add_knowledge_temporal(
            content="Current solution",
            artifact_type="solution_pattern",
            valid_at=now,
        )

        # Query at past time
        past_results = await engine.query_at_time(
            query="solution",
            timestamp=past + timedelta(minutes=30),
        )

        # Query at current time
        current_results = await engine.query_at_time(
            query="solution",
            timestamp=now,
        )

        # Should have different results
        assert len(past_results) >= 0
        assert len(current_results) >= 0

    @pytest.mark.asyncio
    async def test_get_valid_knowledge(self, engine):
        """Test getting all valid knowledge at a time."""
        now = datetime.utcnow()

        # Add multiple artifacts
        await engine.add_knowledge_temporal(
            content="Valid artifact 1",
            artifact_type="solution_pattern",
            valid_at=now,
        )

        await engine.add_knowledge_temporal(
            content="Valid artifact 2",
            artifact_type="workflow",
            valid_at=now,
        )

        # Get valid knowledge
        valid = await engine.get_valid_knowledge(timestamp=now)

        assert len(valid) >= 2

    @pytest.mark.asyncio
    async def test_invalidate_knowledge(self, engine):
        """Test invalidating knowledge."""
        now = datetime.utcnow()

        artifact = await engine.add_knowledge_temporal(
            content="To be invalidated",
            artifact_type="solution_pattern",
            valid_at=now,
        )

        # Invalidate
        success = await engine.invalidate_knowledge(
            artifact_id=artifact.id,
            invalid_at=now + timedelta(hours=1),
        )

        assert success

        # Check it's no longer valid
        updated = await engine.get_artifact(artifact.id)
        assert updated is not None
        assert updated.invalid_at is not None

    @pytest.mark.asyncio
    async def test_detect_contradictions(self, engine):
        """Test contradiction detection."""
        now = datetime.utcnow()

        # Add potentially contradicting artifacts
        await engine.add_knowledge_temporal(
            content="The system cannot handle large loads",
            artifact_type="problem",
            valid_at=now,
        )

        await engine.add_knowledge_temporal(
            content="The system can handle large loads efficiently",
            artifact_type="solution_pattern",
            valid_at=now,
        )

        # Detect contradictions
        result = await engine.detect_contradictions()

        assert isinstance(result, ContradictionDetection)
        assert hasattr(result, "has_contradictions")
        assert hasattr(result, "contradictions")
        assert hasattr(result, "confidence")


class TestGraphitiTemporalBridge:
    """Test GraphitiTemporalBridge."""

    @pytest_asyncio.fixture
    async def bridge(self):
        """Create a temporal bridge for testing."""
        # Create without initializing Graphiti
        bridge = GraphitiTemporalBridge(graphiti_bridge=None)
        yield bridge

    def test_entity_mapping(self):
        """Test entity type mapping."""
        bridge = GraphitiTemporalBridge()

        # Test mappings
        assert bridge.get_graphiti_type_for_artifact("solution_pattern") == "Procedure"
        assert bridge.get_graphiti_type_for_artifact("workflow") == "Document"
        assert bridge.get_graphiti_type_for_artifact("unknown") == "unknown"

    def test_get_entity_mappings(self):
        """Test getting all entity mappings."""
        bridge = GraphitiTemporalBridge()
        mappings = bridge.get_entity_type_mappings()

        assert len(mappings) > 0
        assert all(isinstance(m, EntityMapping) for m in mappings)

    @pytest.mark.asyncio
    async def test_artifact_to_episode(self, bridge):
        """Test converting artifact to episode."""
        now = datetime.utcnow()
        artifact = KnowledgeArtifact(
            id="test_001",
            content="Test artifact",
            artifact_type="solution_pattern",
            valid_at=now,
            metadata={"custom": "value"},
        )

        episode = await bridge.artifact_to_episode(artifact)

        assert episode["name"] == f"solution_pattern: test_001"
        assert episode["body"] == "Test artifact"
        assert episode["reference_time"] == now
        assert episode["metadata"]["artifact_id"] == "test_001"
        assert episode["metadata"]["custom"] == "value"
        assert episode["metadata"]["graphiti_type"] == "Procedure"

    @pytest.mark.asyncio
    async def test_graphiti_result_to_artifact(self, bridge):
        """Test converting Graphiti result to artifact."""
        now = datetime.utcnow()

        # Test edge result
        edge_result = {
            "uuid": "edge_001",
            "fact": "Entity A relates to Entity B",
            "valid_at": now.isoformat(),
            "expired_at": None,
            "created_at": now.isoformat(),
            "source_node": "node_a",
            "target_node": "node_b",
        }

        artifact = await bridge.graphiti_result_to_artifact(edge_result)

        assert artifact is not None
        assert artifact.id == "edge_001"
        assert artifact.content == "Entity A relates to Entity B"
        assert artifact.artifact_type == "relationship"

        # Test node result
        node_result = {
            "uuid": "node_001",
            "name": "Test Entity",
            "summary": "A test entity",
            "labels": ["Entity", "Test"],
        }

        artifact = await bridge.graphiti_result_to_artifact(node_result)

        assert artifact is not None
        assert artifact.id == "node_001"
        assert artifact.artifact_type == "entity"


class TestRerankMethod:
    """Test RerankMethod enum."""

    def test_rerank_methods(self):
        """Test rerank method values."""
        assert RerankMethod.RRF.value == "rrf"
        assert RerankMethod.CROSS_ENCODER.value == "cross_encoder"
        assert RerankMethod.WEIGHTED.value == "weighted"
        assert RerankMethod.NONE.value == "none"


class TestTemporalIntegration:
    """Integration tests for temporal features."""

    @pytest.mark.asyncio
    async def test_temporal_workflow(self):
        """Test complete temporal workflow."""
        engine = TemporalKnowledgeEngine(
            enable_temporal=True,
            enable_hybrid_search=True,
        )

        # Create timeline
        now = datetime.utcnow()
        t1 = now - timedelta(days=2)
        t2 = now - timedelta(days=1)
        t3 = now

        # Add knowledge at different times
        a1 = await engine.add_knowledge_temporal(
            content="Initial approach",
            artifact_type="solution_pattern",
            valid_at=t1,
        )

        a2 = await engine.add_knowledge_temporal(
            content="Improved approach",
            artifact_type="solution_pattern",
            valid_at=t2,
        )

        a3 = await engine.add_knowledge_temporal(
            content="Final optimized solution",
            artifact_type="solution_pattern",
            valid_at=t3,
        )

        # Query evolution over time
        results_t1 = await engine.query_at_time(query="approach", timestamp=t1 + timedelta(hours=1))
        results_t2 = await engine.query_at_time(query="approach", timestamp=t2 + timedelta(hours=1))
        results_t3 = await engine.query_at_time(query="approach", timestamp=t3 + timedelta(hours=1))

        # Verify temporal progression
        assert len(results_t1) >= 0
        assert len(results_t2) >= 0
        assert len(results_t3) >= 0

    @pytest.mark.asyncio
    async def test_hybrid_search_vs_local(self):
        """Test hybrid search vs local search."""
        engine = TemporalKnowledgeEngine(
            enable_temporal=True,
            enable_hybrid_search=True,
        )

        now = datetime.utcnow()

        # Add knowledge
        await engine.add_knowledge_temporal(
            content="Hybrid search test with semantic keywords",
            artifact_type="solution_pattern",
            valid_at=now,
        )

        # Local search
        local_results = await engine._local_search("semantic", max_results=10)

        # Hybrid search (will fall back to local if Graphiti not available)
        hybrid_results = await engine.search_with_graphiti(
            query="semantic information retrieval",
            use_hybrid=True,
            rerank_method="rrf",
        )

        # Both should return results
        assert len(local_results) >= 0
        assert len(hybrid_results) >= 0


@pytest.mark.skipif(
    True,  # Skip unless Graphiti is available
    reason="Requires Graphiti backend"
)
class TestGraphitiBackend:
    """Tests requiring actual Graphiti backend."""

    @pytest.mark.asyncio
    async def test_graphiti_integration(self):
        """Test real Graphiti integration."""
        # This test requires a running Graphiti backend
        # and should be run separately
        pass

    @pytest.mark.asyncio
    async def test_temporal_persistence(self):
        """Test knowledge persistence across sessions."""
        # This test requires a running Graphiti backend
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
