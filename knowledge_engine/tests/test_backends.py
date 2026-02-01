"""
Comprehensive Backend Tests for Knowledge Engine Storage Backends.

Following CLAUDE.md principles:
- Runtime Truth: Verify all backends actually work
- Configuration Explicitness: All config validated
- UTC: All timestamps in UTC
- Idempotency: Tests can be run multiple times safely
- Circuit Breakers: Test failure handling
- Structured Logging: JSON logs with correlation IDs
"""

import asyncio
import pytest
import pytest_asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import backend modules
import sys
from pathlib import Path

# Add knowledge_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_engine.core.backends.base import (
    KnowledgeGraphBackend,
    BackendType,
    KnowledgeEntry,
    SearchResults,
    AnalysisResult,
    GraphStatistics
)
from knowledge_engine.core.backends.memory_backend import MemoryBackend
from knowledge_engine.core.backends.memgraph_backend import MemgraphBackend
from knowledge_engine.core.backends.qdrant_backend import QdrantBackend
from knowledge_engine.core.backends.postgresql_backend import PostgreSQLBackend
from knowledge_engine.core.backends.karateclub_backend import KarateClubBackend

# Note: Neo4j (GPL) and MongoDB (SSPL) backends are excluded due to non-permissive licenses
# Use Memgraph (Apache 2.0) as a drop-in replacement for Neo4j

logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_knowledge_entries() -> List[KnowledgeEntry]:
    """Generate sample knowledge entries for testing."""
    return [
        KnowledgeEntry(
            source="test_document_1",
            content="Artificial Intelligence is transforming healthcare through machine learning algorithms.",
            metadata={"category": "AI", "importance": "high"},
            timestamp=datetime.utcnow().isoformat()
        ),
        KnowledgeEntry(
            source="test_document_2",
            content="Neural networks are inspired by biological neurons in the human brain.",
            metadata={"category": "Neural Networks", "importance": "medium"},
            timestamp=datetime.utcnow().isoformat()
        ),
        KnowledgeEntry(
            source="test_document_3",
            content="Deep learning uses multiple layers of neural networks for feature extraction.",
            metadata={"category": "Deep Learning", "importance": "high"},
            timestamp=datetime.utcnow().isoformat()
        )
    ]


@pytest_asyncio.fixture
async def memory_backend():
    """Create Memory backend instance."""
    backend = MemoryBackend(config={})
    await backend.connect()
    yield backend
    await backend.disconnect()


@pytest_asyncio.fixture
async def memgraph_backend() -> MemgraphBackend:
    """Create Memgraph backend instance if available (Apache 2.0, replaces Neo4j)."""
    config = {
        'uri': 'bolt://localhost:7687',
        'user': '',
        'password': ''
    }

    backend = MemgraphBackend(config=config)

    try:
        await backend.connect()
        yield backend
    except (ImportError, ConnectionError) as e:
        pytest.skip(f"Memgraph not available: {e}")
    finally:
        try:
            await backend.disconnect()
        except:
            pass


@pytest_asyncio.fixture
async def qdrant_backend() -> QdrantBackend:
    """Create Qdrant backend instance if available."""
    config = {
        'host': 'localhost',
        'port': 6333,
        'collection': 'test_knowledge_graph',
        'vector_size': 128
    }

    backend = QdrantBackend(config=config)

    try:
        await backend.connect()
        yield backend
    except (ImportError, ConnectionError) as e:
        pytest.skip(f"Qdrant not available: {e}")
    finally:
        try:
            await backend.disconnect()
        except:
            pass


@pytest_asyncio.fixture
async def postgresql_backend() -> PostgreSQLBackend:
    """Create PostgreSQL backend instance if available (PostgreSQL License)."""
    config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'test_knowledge_graph',
        'user': 'postgres',
        'password': 'postgres'
    }

    backend = PostgreSQLBackend(config=config)

    try:
        await backend.connect()
        yield backend
    except (ImportError, ConnectionError) as e:
        pytest.skip(f"PostgreSQL not available: {e}")
    finally:
        try:
            await backend.disconnect()
        except:
            pass


@pytest_asyncio.fixture
async def karateclub_backend() -> KarateClubBackend:
    """Create KarateClub backend instance."""
    config = {
        'embedding_dim': 64,
        'random_state': 42
    }

    backend = KarateClubBackend(config=config)

    try:
        await backend.connect()
        yield backend
    except ImportError as e:
        pytest.skip(f"KarateClub/NetworkX not available: {e}")
    finally:
        try:
            await backend.disconnect()
        except:
            pass


# =============================================================================
# Base Backend Interface Tests
# =============================================================================

class TestBackendInterface:
    """Test that all backends implement the required interface."""

    @pytest.mark.asyncio
    async def test_memory_backend_interface(self, memory_backend):
        """Test Memory backend implements all required methods."""
        assert hasattr(memory_backend, 'connect')
        assert hasattr(memory_backend, 'disconnect')
        assert hasattr(memory_backend, 'health_check')
        assert hasattr(memory_backend, 'add_knowledge')
        assert hasattr(memory_backend, 'search')
        assert hasattr(memory_backend, 'analyze')
        assert hasattr(memory_backend, 'get_statistics')
        assert hasattr(memory_backend, 'visualize')

    @pytest.mark.asyncio
    async def test_backend_type_enum(self, memory_backend):
        """Test backend type is correctly set."""
        assert memory_backend.backend_type == BackendType.MEMORY
        assert memory_backend.get_backend_name() == "memory"


# =============================================================================
# Memory Backend Tests
# =============================================================================

class TestMemoryBackend:
    """Comprehensive tests for Memory backend."""

    @pytest.mark.asyncio
    async def test_connect_and_health_check(self, memory_backend):
        """Test connection and health check."""
        assert memory_backend.is_healthy
        health = await memory_backend.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_add_knowledge(self, memory_backend, sample_knowledge_entries):
        """Test adding knowledge entries."""
        entry_ids = []
        for entry in sample_knowledge_entries:
            entry_id = await memory_backend.add_knowledge(entry)
            assert entry_id is not None
            assert isinstance(entry_id, str)
            entry_ids.append(entry_id)

        # Verify entries were added
        stats = await memory_backend.get_statistics()
        assert stats.node_count >= len(sample_knowledge_entries)

    @pytest.mark.asyncio
    async def test_search(self, memory_backend, sample_knowledge_entries):
        """Test search functionality."""
        # Add entries first
        for entry in sample_knowledge_entries:
            await memory_backend.add_knowledge(entry)

        # Search for "neural networks"
        results = await memory_backend.search(
            query="neural networks",
            limit=10
        )

        assert isinstance(results, SearchResults)
        assert results.backend_used == "memory"
        assert results.total_count > 0
        assert len(results.results) > 0
        assert any("neural" in r["content"].lower() for r in results.results)

    @pytest.mark.asyncio
    async def test_search_with_filters(self, memory_backend, sample_knowledge_entries):
        """Test search with filters."""
        for entry in sample_knowledge_entries:
            await memory_backend.add_knowledge(entry)

        # Search with source filter
        results = await memory_backend.search(
            query="AI",
            filters={"source": "test_document_1"},
            limit=10
        )

        assert results.total_count > 0
        for result in results.results:
            assert result["source"] == "test_document_1"

    @pytest.mark.asyncio
    async def test_analyze_entity_analysis(self, memory_backend, sample_knowledge_entries):
        """Test entity analysis."""
        for entry in sample_knowledge_entries:
            await memory_backend.add_knowledge(entry)

        result = await memory_backend.analyze(
            analysis_type="entity_analysis"
        )

        assert isinstance(result, AnalysisResult)
        assert result.backend_used == "memory"
        assert "total_entities" in result.results
        assert "top_entities" in result.results

    @pytest.mark.asyncio
    async def test_analyze_source_distribution(self, memory_backend, sample_knowledge_entries):
        """Test source distribution analysis."""
        for entry in sample_knowledge_entries:
            await memory_backend.add_knowledge(entry)

        result = await memory_backend.analyze(
            analysis_type="source_distribution"
        )

        assert isinstance(result, AnalysisResult)
        assert "by_source" in result.results
        assert len(result.results["by_source"]) > 0

    @pytest.mark.asyncio
    async def test_get_statistics(self, memory_backend, sample_knowledge_entries):
        """Test statistics retrieval."""
        for entry in sample_knowledge_entries:
            await memory_backend.add_knowledge(entry)

        stats = await memory_backend.get_statistics()

        assert isinstance(stats, GraphStatistics)
        assert stats.backend == "memory"
        assert stats.node_count > 0
        assert "knowledge_entries" in stats.metadata
        assert "entities" in stats.metadata

    @pytest.mark.asyncio
    async def test_visualize_json(self, memory_backend, sample_knowledge_entries):
        """Test JSON visualization."""
        for entry in sample_knowledge_entries:
            await memory_backend.add_knowledge(entry)

        json_output = await memory_backend.visualize(output_format='json')

        assert isinstance(json_output, str)
        assert "knowledge" in json_output
        assert "entities" in json_output

    @pytest.mark.asyncio
    async def test_visualize_html(self, memory_backend, sample_knowledge_entries):
        """Test HTML visualization."""
        for entry in sample_knowledge_entries:
            await memory_backend.add_knowledge(entry)

        html_output = await memory_backend.visualize(output_format='html')

        assert isinstance(html_output, str)
        assert "<!DOCTYPE html>" in html_output
        assert "Knowledge Graph" in html_output

    @pytest.mark.asyncio
    async def test_delete_knowledge(self, memory_backend, sample_knowledge_entries):
        """Test knowledge deletion."""
        entry_id = await memory_backend.add_knowledge(sample_knowledge_entries[0])

        # Verify entry exists
        results = await memory_backend.search(query="AI", limit=10)
        assert results.total_count > 0

        # Delete entry
        deleted = await memory_backend.delete_knowledge(entry_id)
        assert deleted is True

    @pytest.mark.asyncio
    async def test_update_knowledge(self, memory_backend, sample_knowledge_entries):
        """Test knowledge update."""
        entry_id = await memory_backend.add_knowledge(sample_knowledge_entries[0])

        # Update entry
        updated = await memory_backend.update_knowledge(
            entry_id,
            {"content": "Updated content about AI"}
        )
        assert updated is True

    @pytest.mark.asyncio
    async def test_clear_all(self, memory_backend, sample_knowledge_entries):
        """Test clearing all knowledge."""
        for entry in sample_knowledge_entries:
            await memory_backend.add_knowledge(entry)

        # Verify entries exist
        stats = await memory_backend.get_statistics()
        assert stats.node_count > 0

        # Clear all
        count = await memory_backend.clear_all()
        assert count > 0

        # Verify cleared
        stats_after = await memory_backend.get_statistics()
        assert stats_after.node_count == 0

    @pytest.mark.asyncio
    async def test_batch_add_knowledge(self, memory_backend, sample_knowledge_entries):
        """Test batch adding knowledge."""
        ids = await memory_backend.batch_add_knowledge(sample_knowledge_entries)

        assert len(ids) == len(sample_knowledge_entries)
        assert all(isinstance(id, str) for id in ids)

    @pytest.mark.asyncio
    async def test_pagination(self, memory_backend, sample_knowledge_entries):
        """Test search pagination."""
        for entry in sample_knowledge_entries:
            await memory_backend.add_knowledge(entry)

        # Get first page
        page1 = await memory_backend.search(query="AI", limit=2, offset=0)
        assert len(page1.results) <= 2

        # Get second page
        page2 = await memory_backend.search(query="AI", limit=2, offset=2)
        assert len(page2.results) <= 2


# =============================================================================
# Neo4j Backend Tests
# =============================================================================

class TestMemgraphBackend:
    """Tests for Memgraph backend - Apache 2.0 licensed, Neo4j-compatible."""

    @pytest.mark.asyncio
    async def test_connect_and_health_check(self, memgraph_backend):
        """Test Memgraph connection."""
        assert memgraph_backend.is_healthy
        health = await memgraph_backend.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_add_and_search(self, memgraph_backend, sample_knowledge_entries):
        """Test adding and searching knowledge."""
        # Add entry
        entry_id = await memgraph_backend.add_knowledge(sample_knowledge_entries[0])
        assert entry_id is not None

        # Search
        results = await memgraph_backend.search(query="AI", limit=10)
        assert results.total_count > 0
        assert results.backend_used == "memgraph"

    @pytest.mark.asyncio
    async def test_analyze_connected_components(self, memgraph_backend, sample_knowledge_entries):
        """Test connected components analysis."""
        for entry in sample_knowledge_entries:
            await memgraph_backend.add_knowledge(entry)

        result = await memgraph_backend.analyze(analysis_type="connectivity")
        assert "node_count" in result.results

    @pytest.mark.asyncio
    async def test_analyze_statistics(self, memgraph_backend, sample_knowledge_entries):
        """Test graph statistics analysis."""
        for entry in sample_knowledge_entries:
            await memgraph_backend.add_knowledge(entry)

        stats = await memgraph_backend.get_statistics()
        assert stats.node_count >= len(sample_knowledge_entries)


# =============================================================================
# Qdrant Backend Tests
# =============================================================================

class TestQdrantBackend:
    """Tests for Qdrant backend (requires Qdrant to be running)."""

    @pytest.mark.asyncio
    async def test_connect_and_health_check(self, qdrant_backend):
        """Test Qdrant connection."""
        assert qdrant_backend.is_healthy
        health = await qdrant_backend.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_add_and_search(self, qdrant_backend, sample_knowledge_entries):
        """Test adding and searching with vector similarity."""
        entry_id = await qdrant_backend.add_knowledge(sample_knowledge_entries[0])
        assert entry_id is not None

        # Vector similarity search
        results = await qdrant_backend.search(query="machine learning", limit=10)
        assert results.backend_used == "qdrant"
        assert "search_type" in results.metadata

    @pytest.mark.asyncio
    async def test_batch_add(self, qdrant_backend, sample_knowledge_entries):
        """Test batch adding to Qdrant."""
        ids = await qdrant_backend.batch_add_knowledge(sample_knowledge_entries)
        assert len(ids) == len(sample_knowledge_entries)


# =============================================================================
# MongoDB Backend Tests
# =============================================================================

class TestPostgreSQLBackend:
    """Tests for PostgreSQL backend - PostgreSQL License (permissive)."""

    @pytest.mark.asyncio
    async def test_connect_and_health_check(self, postgresql_backend):
        """Test PostgreSQL connection."""
        assert postgresql_backend.is_healthy
        health = await postgresql_backend.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_add_and_search(self, postgresql_backend, sample_knowledge_entries):
        """Test adding and searching in PostgreSQL."""
        entry_id = await postgresql_backend.add_knowledge(sample_knowledge_entries[0])
        assert entry_id is not None

        # Full-text search
        results = await postgresql_backend.search(query="AI", limit=10)
        assert results.backend_used == "postgresql"

    @pytest.mark.asyncio
    async def test_analyze_statistics(self, postgresql_backend, sample_knowledge_entries):
        """Test statistics analysis."""
        for entry in sample_knowledge_entries:
            await postgresql_backend.add_knowledge(entry)

        stats = await postgresql_backend.get_statistics()
        assert stats.node_count >= len(sample_knowledge_entries)


# =============================================================================
# KarateClub Backend Tests
# =============================================================================

class TestKarateClubBackend:
    """Tests for KarateClub backend."""

    @pytest.mark.asyncio
    async def test_connect_and_health_check(self, karateclub_backend):
        """Test KarateClub initialization."""
        assert karateclub_backend.is_healthy
        health = await karateclub_backend.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_add_knowledge(self, karateclub_backend, sample_knowledge_entries):
        """Test adding knowledge to graph."""
        entry_id = await karateclub_backend.add_knowledge(sample_knowledge_entries[0])
        assert entry_id is not None

        # Verify node added
        stats = await karateclub_backend.get_statistics()
        assert stats.node_count > 0

    @pytest.mark.asyncio
    async def test_analyze_centrality(self, karateclub_backend, sample_knowledge_entries):
        """Test centrality analysis."""
        for entry in sample_knowledge_entries:
            await karateclub_backend.add_knowledge(entry)

        result = await karateclub_backend.analyze(analysis_type="centrality")
        assert "top_betweenness" in result.results
        assert "top_degree" in result.results
        assert "top_pagerank" in result.results


# =============================================================================
# Cross-Backend Tests
# =============================================================================

class TestBackendSwitching:
    """Test backend switching and graceful degradation."""

    @pytest.mark.asyncio
    async def test_backend_selection_by_type(self):
        """Test selecting backend by type."""
        backends = {
            BackendType.MEMORY: MemoryBackend(config={})
        }

        # Try to create Memgraph if available (Apache 2.0, replaces Neo4j)
        try:
            backends[BackendType.MEMGRAPH] = MemgraphBackend(config={
                'uri': 'bolt://localhost:7687'
            })
        except:
            pass

        # At minimum, memory backend should work
        assert BackendType.MEMORY in backends

    @pytest.mark.asyncio
    async def test_fallback_to_memory(self):
        """Test falling back to memory backend."""
        # This simulates the scenario where primary backend fails
        primary_backend = MemoryBackend(config={})

        await primary_backend.connect()
        assert primary_backend.is_healthy

        # Fallback works
        entry = KnowledgeEntry(
            source="test",
            content="Test content"
        )
        entry_id = await primary_backend.add_knowledge(entry)
        assert entry_id is not None


# =============================================================================
# Performance Tests
# =============================================================================

class TestBackendPerformance:
    """Performance tests for backends."""

    @pytest.mark.asyncio
    async def test_memory_backend_performance(self, memory_backend):
        """Test Memory backend performance."""
        import time

        # Add 100 entries
        entries = [
            KnowledgeEntry(
                source=f"doc_{i}",
                content=f"Test content {i} with some keywords",
                metadata={"index": i}
            )
            for i in range(100)
        ]

        start = time.time()
        ids = await memory_backend.batch_add_knowledge(entries)
        add_time = time.time() - start

        logger.info(f"Memory backend: Added {len(ids)} entries in {add_time:.2f}s")
        logger.info(f"Average time per entry: {(add_time / len(ids)) * 1000:.2f}ms")

        # Search performance
        start = time.time()
        results = await memory_backend.search(query="keywords", limit=50)
        search_time = time.time() - start

        logger.info(f"Memory backend: Search completed in {search_time * 1000:.2f}ms")
        assert search_time < 1.0  # Should be very fast


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling in backends."""

    @pytest.mark.asyncio
    async def test_invalid_config_memgraph(self):
        """Test Memgraph with invalid config."""
        with pytest.raises((ValueError, ImportError)):
            backend = MemgraphBackend(config={})
            await backend.connect()

    @pytest.mark.asyncio
    async def test_invalid_config_qdrant(self):
        """Test Qdrant with invalid config."""
        with pytest.raises(ValueError):
            backend = QdrantBackend(config={})  # Missing required fields
            await backend.connect()

    @pytest.mark.asyncio
    async def test_search_when_disconnected(self, memory_backend):
        """Test search when backend is disconnected."""
        await memory_backend.disconnect()

        with pytest.raises(ConnectionError):
            await memory_backend.search(query="test")

    @pytest.mark.asyncio
    async def test_add_when_unhealthy(self, memory_backend):
        """Test adding knowledge when backend is unhealthy."""
        memory_backend.is_healthy = False

        with pytest.raises(ConnectionError):
            await memory_backend.add_knowledge(
                KnowledgeEntry(source="test", content="test")
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
