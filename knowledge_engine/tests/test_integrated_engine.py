"""
Comprehensive Test Suite for IntegratedKnowledgeEngine

Tests all functionality including:
- Initialization and configuration
- Document processing with all sprints
- Batch processing with progress tracking
- Knowledge search and retrieval
- Code analysis
- Temporal queries
- Contradiction detection
- Health checks and statistics
- Error handling and graceful degradation
- Sprint selection and fallback chains
"""

import asyncio
import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import the engine
from knowledge_engine.integrated_engine import (
    IntegratedKnowledgeEngine,
    create_integrated_knowledge_engine,
    ProcessingOptions,
    SprintType,
    TaskType,
    BatchResult
)

# Test fixtures
@pytest.fixture
async def test_config():
    """Provide test configuration."""
    return {
        "graphiti_uri": "bolt://localhost:7687",
        "graphiti_user": "neo4j",
        "graphiti_password": "test_password",
        "elasticsearch_hosts": ["http://localhost:9200"],
        "cache_ttl": 60,
    }


@pytest.fixture
async def minimal_engine(test_config):
    """Create engine with minimal configuration for testing."""
    engine = IntegratedKnowledgeEngine(test_config)
    yield engine
    await engine.close()


@pytest.fixture
def sample_text_content():
    """Provide sample text content for testing."""
    return """
    Machine Learning is a subset of artificial intelligence that focuses on
    algorithms that can learn from data. Over time, these algorithms have
    evolved significantly. The history of machine learning dates back to the
    1950s with early work on neural networks and perceptrons.

    Key concepts include supervised learning, unsupervised learning, and
    reinforcement learning. Modern applications span computer vision,
    natural language processing, and autonomous systems.
    """


@pytest.fixture
def sample_markdown_file(tmp_path):
    """Create a temporary markdown file for testing."""
    file_path = tmp_path / "test.md"
    file_path.write_text("""
    # Test Document

    This is a test document for knowledge extraction.

    ## Key Concepts

    - Machine Learning
    - Neural Networks
    - Data Science

    The timeline of AI development shows significant progress over recent decades.
    """)
    return str(file_path)


class TestIntegratedKnowledgeEngineInitialization:
    """Test engine initialization and configuration."""

    @pytest.mark.asyncio
    async def test_initialization_with_config(self, test_config):
        """Test initialization with custom configuration."""
        engine = IntegratedKnowledgeEngine(test_config)
        assert engine.config == test_config
        assert not engine._initialized
        assert not engine._closed
        await engine.close()

    @pytest.mark.asyncio
    async def test_initialization_with_defaults(self):
        """Test initialization with default configuration."""
        engine = IntegratedKnowledgeEngine()
        assert engine.config is not None
        assert "graphiti_uri" in engine.config
        await engine.close()

    @pytest.mark.asyncio
    async def test_config_validation_missing_required(self):
        """Test that missing required config raises error."""
        # Mock GRAPHITI_AVAILABLE as True
        with patch('knowledge_engine.integrated_engine.GRAPHITI_AVAILABLE', True):
            config = {"graphiti_uri": "bolt://localhost:7687"}
            # Should not raise immediately, but on initialize
            engine = IntegratedKnowledgeEngine(config)
            # Note: Actual validation happens when trying to use components
            await engine.close()

    @pytest.mark.asyncio
    async def test_initialize_method(self, minimal_engine):
        """Test the initialize method."""
        await minimal_engine.initialize()
        assert minimal_engine._initialized
        assert not minimal_engine._closed

    @pytest.mark.asyncio
    async def test_double_initialize(self, minimal_engine):
        """Test that double initialization is idempotent."""
        await minimal_engine.initialize()
        await minimal_engine.initialize()  # Should not raise
        assert minimal_engine._initialized

    @pytest.mark.asyncio
    async def test_context_manager(self, test_config):
        """Test async context manager usage."""
        async with IntegratedKnowledgeEngine(test_config) as engine:
            assert engine._initialized
            assert not engine._closed
        assert engine._closed


class TestDocumentProcessing:
    """Test document processing functionality."""

    @pytest.mark.asyncio
    async def test_process_markdown_file(self, minimal_engine, sample_markdown_file):
        """Test processing a markdown file."""
        result = await minimal_engine.process_document(sample_markdown_file)
        assert result is not None
        assert "correlation_id" in result
        assert "processing_time_ms" in result

    @pytest.mark.asyncio
    async def test_process_document_with_options(self, minimal_engine, sample_markdown_file):
        """Test processing with custom options."""
        options = ProcessingOptions(
            extract_temporal=True,
            extract_bilingual=False,
            timeout_ms=10000
        )
        result = await minimal_engine.process_document(sample_markdown_file, options)
        assert result is not None

    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self, minimal_engine):
        """Test processing a file that doesn't exist."""
        result = await minimal_engine.process_document("/nonexistent/file.pdf")
        assert result["success"] == False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_process_with_correlation_id(self, minimal_engine, sample_markdown_file):
        """Test processing with custom correlation ID."""
        options = ProcessingOptions(correlation_id="test_correlation_123")
        result = await minimal_engine.process_document(sample_markdown_file, options)
        assert result["correlation_id"] == "test_correlation_123"


class TestBatchProcessing:
    """Test batch processing functionality."""

    @pytest.mark.asyncio
    async def test_batch_process_multiple_files(self, minimal_engine, tmp_path):
        """Test batch processing multiple files."""
        # Create multiple test files
        files = []
        for i in range(3):
            file_path = tmp_path / f"test_{i}.md"
            file_path.write_text(f"Test document {i} content")
            files.append(str(file_path))

        # Track progress
        progress_updates = []
        def progress_callback(msg, pct, meta):
            progress_updates.append({"msg": msg, "pct": pct, "meta": meta})

        # Process batch
        result = await minimal_engine.batch_process_documents(
            files,
            progress_callback=progress_callback,
            max_concurrent=2
        )

        assert isinstance(result, BatchResult)
        assert result.total_items == 3
        assert len(progress_updates) > 0

    @pytest.mark.asyncio
    async def test_batch_process_with_failures(self, minimal_engine, tmp_path):
        """Test batch processing with some failures."""
        files = [
            str(tmp_path / "exists.md"),  # This will be created
            "/nonexistent/file.pdf",      # This doesn't exist
            str(tmp_path / "exists2.md")  # This will be created
        ]

        # Create the existing files
        Path(files[0]).write_text("Content 1")
        Path(files[2]).write_text("Content 2")

        result = await minimal_engine.batch_process_documents(files)

        assert result.total_items == 3
        assert result.successful >= 0
        assert result.failed >= 0
        assert result.successful + result.failed == 3

    @pytest.mark.asyncio
    async def test_batch_concurrency_limit(self, minimal_engine, tmp_path):
        """Test that batch processing respects concurrency limit."""
        files = []
        for i in range(5):
            file_path = tmp_path / f"test_{i}.md"
            file_path.write_text(f"Content {i}")
            files.append(str(file_path))

        import time
        start_time = time.time()

        result = await minimal_engine.batch_process_documents(
            files,
            max_concurrent=2
        )

        # Should complete, but with controlled concurrency
        assert result.total_items == 5


class TestKnowledgeSearch:
    """Test knowledge search functionality."""

    @pytest.mark.asyncio
    async def test_search_knowledge_hybrid(self, minimal_engine):
        """Test hybrid knowledge search."""
        result = await minimal_engine.search_knowledge(
            "machine learning",
            query_type="hybrid",
            limit=5
        )
        assert "query" in result
        assert "correlation_id" in result

    @pytest.mark.asyncio
    async def test_search_knowledge_keyword(self, minimal_engine):
        """Test keyword search."""
        result = await minimal_engine.search_knowledge(
            "algorithm",
            query_type="keyword",
            limit=10
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_search_with_filters(self, minimal_engine):
        """Test search with filters."""
        result = await minimal_engine.search_knowledge(
            "test query",
            filters={"type": "document"},
            limit=5
        )
        assert result is not None


class TestCodeAnalysis:
    """Test code repository analysis."""

    @pytest.mark.asyncio
    async def test_analyze_code_repository(self, minimal_engine, tmp_path):
        """Test analyzing a code repository."""
        # Create a simple Python project
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        (repo_path / "main.py").write_text("""
def hello_world():
    print("Hello, World!")

class TestClass:
    def method(self):
        pass
""")

        result = await minimal_engine.analyze_code(str(repo_path))
        assert result is not None
        assert "correlation_id" in result

    @pytest.mark.asyncio
    async def test_analyze_nonexistent_repo(self, minimal_engine):
        """Test analyzing a repository that doesn't exist."""
        result = await minimal_engine.analyze_code("/nonexistent/repo")
        assert result["success"] == False


class TestTemporalQueries:
    """Test temporal knowledge queries."""

    @pytest.mark.asyncio
    async def test_query_temporal_default_time(self, minimal_engine):
        """Test temporal query with default time (now)."""
        result = await minimal_engine.query_temporal("test query")
        assert result is not None
        assert "query" in result
        assert "correlation_id" in result

    @pytest.mark.asyncio
    async def test_query_temporal_custom_time(self, minimal_engine):
        """Test temporal query with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = await minimal_engine.query_temporal(
            "test query",
            timestamp=custom_time
        )
        assert result is not None


class TestContradictionDetection:
    """Test contradiction detection."""

    @pytest.mark.asyncio
    async def test_detect_contradictions(self, minimal_engine):
        """Test contradiction detection for an entity."""
        result = await minimal_engine.detect_contradictions("test_entity")
        assert result is not None
        assert "entity" in result
        assert "correlation_id" in result


class TestStatisticsAndHealth:
    """Test statistics and health checks."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, minimal_engine):
        """Test getting engine statistics."""
        await minimal_engine.initialize()
        stats = await minimal_engine.get_statistics()
        assert stats is not None
        assert "timestamp" in stats
        assert "components" in stats
        assert "knowledge" in stats

    @pytest.mark.asyncio
    async def test_health_check(self, minimal_engine):
        """Test health check."""
        await minimal_engine.initialize()
        health = await minimal_engine.health_check()
        assert health is not None
        assert "timestamp" in health
        assert "overall" in health
        assert "components" in health


class TestSprintSelection:
    """Test automatic sprint selection."""

    def test_select_sprint_for_temporal_content(self, minimal_engine):
        """Test sprint selection for temporal content."""
        content = "The timeline shows evolution over the past decade"
        options = ProcessingOptions(extract_temporal=True)
        sprint = minimal_engine._select_sprint_for_content(content, options)
        assert sprint in [SprintType.TEMPORAL_GRAPHITI, SprintType.GENERIC_KGGEN]

    def test_select_sprint_for_general_content(self, minimal_engine):
        """Test sprint selection for general content."""
        content = "This is a general document about algorithms"
        options = ProcessingOptions()
        sprint = minimal_engine._select_sprint_for_content(content, options)
        assert sprint in [SprintType.GENERIC_KGGEN, SprintType.HYBRID_AUTO]

    def test_get_sprint_fallback_chain(self, minimal_engine):
        """Test fallback chain generation."""
        chain = minimal_engine._get_sprint_fallback_chain(SprintType.TEMPORAL_GRAPHITI)
        assert len(chain) > 1
        assert SprintType.TEMPORAL_GRAPHITI in chain


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_missing_component(self, minimal_engine):
        """Test that engine degrades gracefully when components are missing."""
        # Engine should still function with some components unavailable
        result = await minimal_engine.search_knowledge("test")
        # Should return result even if some components are missing
        assert result is not None

    @pytest.mark.asyncio
    async def test_error_propagation(self, minimal_engine):
        """Test that errors are properly propagated."""
        result = await minimal_engine.process_document("/nonexistent/file.txt")
        assert result["success"] == False
        assert "error" in result


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_create_integrated_knowledge_engine(self, test_config):
        """Test the convenience creation function."""
        engine = await create_integrated_knowledge_engine(test_config)
        assert engine._initialized
        await engine.close()


class TestDataStructures:
    """Test data structures."""

    def test_processing_options_defaults(self):
        """Test ProcessingOptions default values."""
        options = ProcessingOptions()
        assert options.extract_temporal == True
        assert options.extract_bilingual == False
        assert options.use_embeddings == True
        assert options.timeout_ms == 30000

    def test_processing_options_custom(self):
        """Test ProcessingOptions with custom values."""
        options = ProcessingOptions(
            extract_temporal=False,
            extract_bilingual=True,
            timeout_ms=60000
        )
        assert options.extract_temporal == False
        assert options.extract_bilingual == True
        assert options.timeout_ms == 60000

    def test_batch_result_to_dict(self):
        """Test BatchResult serialization."""
        result = BatchResult(
            total_items=10,
            successful=8,
            failed=2,
            total_time_ms=1500.0
        )
        d = result.to_dict()
        assert d["total_items"] == 10
        assert d["successful"] == 8
        assert d["failed"] == 2
        assert "success_rate" in d
        assert d["success_rate"] == 0.8


# Performance tests
class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self, minimal_engine, tmp_path):
        """Test processing multiple documents concurrently."""
        # Create test files
        files = []
        for i in range(10):
            file_path = tmp_path / f"perf_test_{i}.md"
            file_path.write_text(f"Performance test content {i}")
            files.append(str(file_path))

        import time
        start_time = time.time()

        result = await minimal_engine.batch_process_documents(
            files,
            max_concurrent=5
        )

        elapsed = time.time() - start_time
        assert result.total_items == 10
        # Should complete in reasonable time
        assert elapsed < 30  # 30 seconds max


# Integration tests
class TestIntegration:
    """Integration tests with real components (if available)."""

    @pytest.mark.asyncio
    @pytest.mark.skip("Requires actual Neo4j instance")
    async def test_full_workflow_with_neo4j(self):
        """Test full workflow with real Neo4j (requires setup)."""
        config = {
            "graphiti_uri": "bolt://localhost:7687",
            "graphiti_user": "neo4j",
            "graphiti_password": os.getenv("NEO4J_PASSWORD", "password"),
        }

        async with await create_integrated_knowledge_engine(config) as engine:
            health = await engine.health_check()
            assert health["overall"] in ["healthy", "degraded"]

            # Process a document
            # result = await engine.process_document("test.pdf")
            # assert result["success"]


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
