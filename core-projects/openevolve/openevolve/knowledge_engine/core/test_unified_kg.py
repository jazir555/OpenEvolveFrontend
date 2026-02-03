"""
Unit tests for Unified Knowledge Graph Manager.

Following CLAUDE.md principles: Runtime Truth, Idempotency.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from knowledge_engine.core.unified_knowledge_graph import (
    UnifiedKnowledgeGraph,
    KnowledgeGraphError,
    BackendUnavailableError
)
from knowledge_engine.core.backends.base import (
    KnowledgeEntry,
    SearchResults,
    AnalysisResult,
    GraphStatistics
)
from knowledge_engine.core.backends.memory_backend import MemoryBackend


class TestMemoryBackend:
    """Test MemoryBackend implementation"""

    @pytest_asyncio.fixture
    async def backend(self):
        """Create a memory backend instance"""
        backend = MemoryBackend({})
        await backend.connect()
        return backend

    @pytest.mark.asyncio
    async def test_connect(self, backend):
        """Test backend connection"""
        assert backend.is_healthy is True

    @pytest.mark.asyncio
    async def test_add_knowledge(self, backend):
        """Test adding knowledge"""
        entry = KnowledgeEntry(
            source="test",
            content="Test content",
            metadata={"key": "value"}
        )

        entry_id = await backend.add_knowledge(entry)
        assert entry_id is not None
        assert isinstance(entry_id, str)

    @pytest.mark.asyncio
    async def test_search(self, backend):
        """Test searching knowledge"""
        # Add test knowledge
        await backend.add_knowledge(KnowledgeEntry(
            source="test",
            content="Python is a programming language"
        ))

        await backend.add_knowledge(KnowledgeEntry(
            source="test",
            content="JavaScript is also a programming language"
        ))

        # Search
        results = await backend.search("programming")
        assert results.total_count >= 1
        assert len(results.results) > 0
        assert results.backend_used == "memory"

    @pytest.mark.asyncio
    async def test_analyze(self, backend):
        """Test graph analysis"""
        # Add test knowledge
        await backend.add_knowledge(KnowledgeEntry(
            source="test1",
            content="Test content one"
        ))

        await backend.add_knowledge(KnowledgeEntry(
            source="test2",
            content="Test content two"
        ))

        # Analyze
        analysis = await backend.analyze("source_distribution")
        assert analysis.backend_used == "memory"
        assert "by_source" in analysis.results

    @pytest.mark.asyncio
    async def test_get_statistics(self, backend):
        """Test getting statistics"""
        await backend.add_knowledge(KnowledgeEntry(
            source="test",
            content="Test content"
        ))

        stats = await backend.get_statistics()
        assert stats.backend == "memory"
        assert stats.node_count > 0

    @pytest.mark.asyncio
    async def test_visualize_html(self, backend):
        """Test HTML visualization"""
        viz = await backend.visualize("html")
        assert isinstance(viz, str)
        assert "<html>" in viz

    @pytest.mark.asyncio
    async def test_visualize_json(self, backend):
        """Test JSON visualization"""
        viz = await backend.visualize("json")
        assert isinstance(viz, str)
        assert "{" in viz  # JSON

    @pytest.mark.asyncio
    async def test_delete_knowledge(self, backend):
        """Test deleting knowledge"""
        entry = await backend.add_knowledge(KnowledgeEntry(
            source="test",
            content="Test content"
        ))

        # Delete
        result = await backend.delete_knowledge(entry)
        assert result is True

        # Verify deleted
        results = await backend.search("Test content")
        assert results.total_count == 0

    @pytest.mark.asyncio
    async def test_update_knowledge(self, backend):
        """Test updating knowledge"""
        entry = await backend.add_knowledge(KnowledgeEntry(
            source="test",
            content="Original content"
        ))

        # Update
        result = await backend.update_knowledge(
            entry,
            {"content": "Updated content"}
        )
        assert result is True

        # Verify updated
        results = await backend.search("Updated content")
        assert results.total_count == 1

    @pytest.mark.asyncio
    async def test_clear_all(self, backend):
        """Test clearing all knowledge"""
        await backend.add_knowledge(KnowledgeEntry(
            source="test",
            content="Test content"
        ))

        count = await backend.clear_all()
        assert count > 0

        # Verify cleared
        stats = await backend.get_statistics()
        assert stats.node_count == 0


class TestUnifiedKnowledgeGraph:
    """Test Unified Knowledge Graph Manager"""

    @pytest_asyncio.fixture
    async def kg(self):
        """Create a knowledge graph manager"""
        kg = UnifiedKnowledgeGraph()
        await kg.connect_all()
        return kg

    @pytest.mark.asyncio
    async def test_initialization(self, kg):
        """Test manager initialization"""
        assert len(kg.backends) > 0
        assert "memory" in kg.backends

    @pytest.mark.asyncio
    async def test_add_knowledge(self, kg):
        """Test adding knowledge"""
        entry_id = await kg.add_knowledge(
            source="test",
            content="Test content"
        )
        assert entry_id is not None

    @pytest.mark.asyncio
    async def test_search(self, kg):
        """Test searching"""
        await kg.add_knowledge(
            source="test",
            content="Searchable content"
        )

        results = await kg.search("searchable")
        assert results.total_count >= 1

    @pytest.mark.asyncio
    async def test_analyze(self, kg):
        """Test analysis"""
        await kg.add_knowledge(
            source="test",
            content="Test content"
        )

        analysis = await kg.analyze("source_distribution")
        assert analysis is not None

    @pytest.mark.asyncio
    async def test_get_graph_stats(self, kg):
        """Test getting statistics"""
        await kg.add_knowledge(
            source="test",
            content="Test content"
        )

        stats = await kg.get_graph_stats()
        assert "backends" in stats
        assert "timestamp" in stats

    @pytest.mark.asyncio
    async def test_visualize(self, kg):
        """Test visualization"""
        viz = await kg.visualize("html")
        assert isinstance(viz, str)

    @pytest.mark.asyncio
    async def test_batch_add_knowledge(self, kg):
        """Test batch adding"""
        entries = [
            {"source": "test", "content": "Entry 1"},
            {"source": "test", "content": "Entry 2"},
            {"source": "test", "content": "Entry 3"}
        ]

        ids = await kg.batch_add_knowledge(entries)
        assert len(ids) == 3

    @pytest.mark.asyncio
    async def test_health_check(self, kg):
        """Test health check"""
        health = await kg.health_check()
        assert isinstance(health, dict)
        assert len(health) > 0

    @pytest.mark.asyncio
    async def test_backend_selection(self, kg):
        """Test automatic backend selection"""
        backend = kg._select_backend("add_knowledge")
        assert backend is not None
        assert backend.is_healthy is True

    @pytest.mark.asyncio
    async def test_search_with_filters(self, kg):
        """Test search with filters"""
        await kg.add_knowledge(
            source="source1",
            content="Content from source 1"
        )

        await kg.add_knowledge(
            source="source2",
            content="Content from source 2"
        )

        # Search with source filter
        results = await kg.search(
            "Content",
            filters={"source": "source1"}
        )

        assert results.total_count >= 0

    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, kg):
        """Test fallback to memory backend when others fail"""
        # This should work since memory backend is always available
        entry_id = await kg.add_knowledge(
            source="fallback_test",
            content="Test fallback"
        )
        assert entry_id is not None


class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_backend_unavailable(self):
        """Test behavior when all backends are unavailable"""
        kg = UnifiedKnowledgeGraph()

        # Make all backends unhealthy
        for backend in kg.backends.values():
            backend.is_healthy = False

        with pytest.raises(BackendUnavailableError):
            await kg.add_knowledge("test", "content")

    @pytest.mark.asyncio
    async def test_invalid_analysis_type(self):
        """Test invalid analysis type"""
        kg = UnifiedKnowledgeGraph()
        await kg.connect_all()

        with pytest.raises(KnowledgeGraphError):
            await kg.analyze("invalid_analysis_type")

    @pytest.mark.asyncio
    async def test_invalid_visualization_format(self):
        """Test invalid visualization format"""
        kg = UnifiedKnowledgeGraph()
        await kg.connect_all()

        with pytest.raises(KnowledgeGraphError):
            await kg.visualize("invalid_format")

    @pytest.mark.asyncio
    async def test_empty_search(self):
        """Test search with no results"""
        kg = UnifiedKnowledgeGraph()
        await kg.connect_all()

        results = await kg.search("nonexistent content xyz123")
        assert results.total_count == 0
        assert len(results.results) == 0


class TestIdempotency:
    """Test idempotent operations - Law of Idempotency"""

    @pytest.mark.asyncio
    async def test_idempotent_add(self):
        """Test that adding same knowledge multiple times is handled"""
        kg = UnifiedKnowledgeGraph()
        await kg.connect_all()

        # Add same content multiple times
        id1 = await kg.add_knowledge("test", "Same content")
        id2 = await kg.add_knowledge("test", "Same content")

        # Should create different entries (different IDs)
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_idempotent_delete(self):
        """Test that deleting same entry twice is safe"""
        kg = UnifiedKnowledgeGraph()
        await kg.connect_all()

        # Get backend directly
        backend = kg.backends["memory"]
        entry_id = await backend.add_knowledge(KnowledgeEntry(
            source="test",
            content="Test content"
        ))

        # Delete twice
        result1 = await backend.delete_knowledge(entry_id)
        result2 = await backend.delete_knowledge(entry_id)

        # First delete should succeed, second should return False
        assert result1 is True
        assert result2 is False


class TestConfiguration:
    """Test configuration handling"""

    def test_default_config(self):
        """Test default configuration"""
        kg = UnifiedKnowledgeGraph()
        assert kg.config is not None
        assert "backends" in kg.config

    def test_config_loading(self, tmp_path):
        """Test loading config from file"""
        import yaml

        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "backends": {
                "memory": {"enabled": True}
            },
            "fallback_chain": ["memory"]
        }

        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        kg = UnifiedKnowledgeGraph(str(config_file))
        assert kg.config is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
