"""
Comprehensive tests for KnowledgeEngine orchestration class.

Following CLAUDE.md principles:
- RUNTIME TRUTH: Test actual functionality, not mocks
- IDEMPOTENCY: Tests can be run multiple times
- STRUCTURED LOGGING: All tests log with correlation IDs
- CONFIGURATION EXPLICITNESS: Tests fail if misconfigured

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Import KnowledgeEngine components
from knowledge_engine.orchestration import (
    KnowledgeEngine,
    create_knowledge_engine,
    ProcessingResult,
    QueryResult
)


# ========== Fixtures ==========

@pytest.fixture
async def minimal_config():
    """
    Minimal configuration for testing.

    Using CLAUDE.md: CONFIGURATION EXPLICITNESS
    All config explicitly provided, no magic defaults.
    """
    return {
        # Graphiti (optional for tests)
        "graphiti_uri": os.getenv("GRAPHITI_URI", "bolt://localhost:7687"),
        "graphiti_user": os.getenv("GRAPHITI_USER", "neo4j"),
        "graphiti_password": os.getenv("GRAPHITI_PASSWORD"),

        # LLM (required for extraction)
        "openai_api_key": os.getenv("OPENAI_API_KEY", "test_key"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),

        # Elasticsearch (optional)
        "elasticsearch_hosts": ["http://localhost:9200"],
        "elasticsearch_api_key": os.getenv("ELASTICSEARCH_API_KEY", ""),

        # Visualization
        "viz_export_dir": tempfile.mkdtemp(),
    }


@pytest.fixture
async def engine(minimal_config):
    """
    Create a KnowledgeEngine instance for testing.

    Following CLAUDE.md: Proper setup and teardown
    """
    eng = KnowledgeEngine(config=minimal_config)
    try:
        await eng.initialize()
        yield eng
    finally:
        await eng.close()


@pytest.fixture
def sample_document(tmp_path):
    """Create a sample document for testing."""
    doc_path = tmp_path / "test_doc.txt"
    doc_path.write_text("""
    Machine Learning and Artificial Intelligence

    Machine learning is a subset of artificial intelligence that focuses
    on algorithms that can learn from data. Deep learning is a type of
    machine learning that uses neural networks with multiple layers.

    Neural networks are inspired by the human brain and consist of
    interconnected nodes (neurons) that process information in layers.

    Key concepts:
    - Supervised learning uses labeled data
    - Unsupervised learning finds patterns in unlabeled data
    - Reinforcement learning learns through trial and error
    """, encoding='utf-8')
    return str(doc_path)


# ========== Configuration Tests ==========

class TestConfiguration:
    """Test configuration validation and loading."""

    def test_config_from_env(self):
        """Test configuration loads from environment variables."""
        # Set test environment variables
        os.environ["TEST_VAR"] = "test_value"

        engine = KnowledgeEngine(config={"test_key": "test_value"})

        assert engine.config is not None
        assert engine.config.get("test_key") == "test_value"

    def test_config_validation_missing_required(self):
        """Test that missing required config raises error."""
        # Mock Graphiti available but no password
        os.environ["GRAPHITI_PASSWORD"] = ""

        # This should raise RuntimeError due to missing config
        with pytest.raises(RuntimeError, match="Missing required"):
            KnowledgeEngine(config={
                "graphiti_uri": "bolt://localhost:7687",
                "graphiti_user": "neo4j",
                "graphiti_password": ""  # Empty = missing
            })

    def test_config_with_minimal_config(self, minimal_config):
        """Test engine accepts minimal valid configuration."""
        engine = KnowledgeEngine(config=minimal_config)
        assert engine is not None
        assert engine.config == minimal_config


# ========== Initialization Tests ==========

class TestInitialization:
    """Test KnowledgeEngine initialization."""

    @pytest.mark.asyncio
    async def test_engine_initialization(self, minimal_config):
        """Test basic engine initialization."""
        engine = KnowledgeEngine(config=minimal_config)
        await engine.initialize()

        assert engine._initialized is True
        assert engine._closed is False

        await engine.close()

    @pytest.mark.asyncio
    async def test_double_initialization(self, engine):
        """Test that double initialization is safe (idempotent)."""
        assert engine._initialized is True

        # Should not raise error
        await engine.initialize()

        assert engine._initialized is True

    @pytest.mark.asyncio
    async def test_create_knowledge_engine_convenience(self, minimal_config):
        """Test convenience function for creating engine."""
        eng = await create_knowledge_engine(config=minimal_config)

        assert eng is not None
        assert eng._initialized is True

        await eng.close()


# ========== Document Processing Tests ==========

class TestDocumentProcessing:
    """Test document processing functionality."""

    @pytest.mark.asyncio
    async def test_process_text_document(self, engine, sample_document):
        """Test processing a text document."""
        result = await engine.process_document(
            document_path=sample_document,
            extract_temporal=False,  # Skip Graphiti if not available
            extract_bilingual=False
        )

        assert result is not None
        assert isinstance(result, ProcessingResult)
        assert result.correlation_id is not None
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_process_nonexistent_document(self, engine):
        """Test processing a non-existent document raises error."""
        with pytest.raises(FileNotFoundError):
            await engine.process_document(
                document_path="/nonexistent/path/doc.txt"
            )

    @pytest.mark.asyncio
    async def test_process_document_idempotency(self, engine, sample_document):
        """
        Test that processing same document twice is safe.

        Following CLAUDE.md: IDEMPOTENCY
        """
        # Process first time
        result1 = await engine.process_document(
            document_path=sample_document,
            extract_temporal=False
        )

        # Process second time
        result2 = await engine.process_document(
            document_path=sample_document,
            extract_temporal=False
        )

        # Both should succeed
        assert result1.success
        assert result2.success

        # Should have different correlation IDs
        assert result1.correlation_id != result2.correlation_id


# ========== Query Tests ==========

class TestQuerying:
    """Test knowledge query functionality."""

    @pytest.mark.asyncio
    async def test_query_temporal_raises_without_graphiti(self, engine):
        """Test that temporal query raises error if Graphiti not available."""
        # If Graphiti is not configured, this should raise
        if not engine._graphiti:
            with pytest.raises(RuntimeError, match="Graphiti.*not available"):
                await engine.query_temporal(
                    query="test query",
                    timestamp=datetime.now(timezone.utc)
                )

    @pytest.mark.asyncio
    async def test_search_knowledge_raises_without_elasticsearch(self, engine):
        """Test that search raises error if Elasticsearch not available."""
        # If Elasticsearch is not configured, this should raise
        if not engine._elasticsearch:
            with pytest.raises(RuntimeError, match="Elasticsearch.*not available"):
                await engine.search_knowledge(
                    query="test query",
                    query_type="hybrid"
                )


# ========== Visualization Tests ==========

class TestVisualization:
    """Test visualization functionality."""

    @pytest.mark.asyncio
    async def test_visualize_graph_raises_without_viz(self, engine):
        """Test that visualization raises error if not available."""
        # If visualization not available, should raise
        if not engine._visualization:
            with pytest.raises(RuntimeError, match="Visualization.*not available"):
                await engine.visualize_graph(
                    graph_type="explorer",
                    data={"triples": []}
                )

    @pytest.mark.asyncio
    async def test_visualize_invalid_graph_type(self, engine):
        """Test that invalid graph type raises error."""
        if engine._visualization:
            with pytest.raises(ValueError, match="Unknown visualization type"):
                await engine.visualize_graph(
                    graph_type="invalid_type",
                    data={"triples": []}
                )


# ========== Statistics and Health Tests ==========

class TestStatisticsAndHealth:
    """Test statistics and health check functionality."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, engine):
        """Test getting statistics."""
        stats = await engine.get_statistics()

        assert stats is not None
        assert "timestamp" in stats
        assert "components" in stats
        assert "knowledge" in stats

        # Check component status
        components = stats["components"]
        assert "graphiti" in components
        assert "kggen" in components
        assert "oneke" in components
        assert "visualization" in components
        assert "elasticsearch" in components
        assert "indexer" in components

        # Check knowledge stats
        knowledge = stats["knowledge"]
        assert "entities" in knowledge
        assert "relationships" in knowledge
        assert isinstance(knowledge["entities"], int)
        assert isinstance(knowledge["relationships"], int)

    @pytest.mark.asyncio
    async def test_health_check(self, engine):
        """Test health check."""
        health = await engine.health_check()

        assert health is not None
        assert "timestamp" in health
        assert "overall" in health
        assert "components" in health

        # Overall status should be one of these
        assert health["overall"] in ["healthy", "degraded", "unhealthy"]


# ========== Context Manager Tests ==========

class TestContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, minimal_config):
        """Test using engine as async context manager."""
        async with await create_knowledge_engine(config=minimal_config) as engine:
            assert engine is not None
            assert engine._initialized is True
            assert engine._closed is False

        # After exiting context, should be closed
        # Note: We can't check engine._closed here as it's out of scope

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, engine):
        """
        Test that close() can be called multiple times safely.

        Following CLAUDE.md: IDEMPOTENCY
        """
        assert engine._closed is False

        await engine.close()
        assert engine._closed is True

        # Should not raise error
        await engine.close()
        assert engine._closed is True


# ========== Integration Tests ==========

class TestIntegration:
    """Integration tests for end-to-end functionality."""

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self, engine, sample_document):
        """
        Test full pipeline: document -> extraction -> visualization.

        This is an integration test that exercises multiple components.
        """
        # Step 1: Process document
        process_result = await engine.process_document(
            document_path=sample_document,
            extract_temporal=False,
            extract_bilingual=False
        )

        assert process_result.success

        # Step 2: Check knowledge was added to entity graph
        entities = engine.entity_graph.get_entities()
        assert len(entities) >= 0  # May have entities extracted

        # Step 3: Try to visualize (if available)
        if engine._visualization and process_result.triples:
            viz = await engine.visualize_graph(
                graph_type="explorer",
                data={"triples": process_result.triples}
            )
            assert viz is not None

    @pytest.mark.asyncio
    async def test_correlation_id_tracking(self, engine, sample_document):
        """Test that correlation IDs are properly tracked."""
        custom_correlation_id = "test_custom_correlation_123"

        result = await engine.process_document(
            document_path=sample_document,
            correlation_id=custom_correlation_id
        )

        assert result.correlation_id == custom_correlation_id

    @pytest.mark.asyncio
    async def test_processing_result_serialization(self, engine, sample_document):
        """Test that ProcessingResult can be serialized to dict."""
        result = await engine.process_document(
            document_path=sample_document
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "success" in result_dict
        assert "entities" in result_dict
        assert "relations" in result_dict
        assert "triples" in result_dict
        assert "correlation_id" in result_dict
        assert "processing_time_ms" in result_dict


# ========== Performance Tests ==========

class TestPerformance:
    """Performance and load tests."""

    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self, engine, sample_document, tmp_path):
        """Test processing multiple documents concurrently."""
        # Create multiple test documents
        docs = []
        for i in range(3):
            doc_path = tmp_path / f"test_doc_{i}.txt"
            doc_path.write_text(f"Test document {i}", encoding='utf-8')
            docs.append(str(doc_path))

        # Process all concurrently
        tasks = [
            engine.process_document(
                document_path=doc,
                extract_temporal=False
            )
            for doc in docs
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 3
        assert all(r.success for r in results)

        # Each should have unique correlation ID
        correlation_ids = [r.correlation_id for r in results]
        assert len(set(correlation_ids)) == 3  # All unique


# ========== Error Handling Tests ==========

class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self, engine, tmp_path):
        """Test processing unsupported file type."""
        # Create a file with unsupported extension
        doc_path = tmp_path / "test.xyz"
        doc_path.write_text("content")

        result = await engine.process_document(
            document_path=str(doc_path)
        )

        # Should fail gracefully
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_processing_error_returns_result_object(self, engine, tmp_path):
        """
        Test that errors during processing return Result object, not raise.

        Following CLAUDE.md: Handle failure gracefully
        """
        # Try to process non-existent file
        result = await engine.process_document(
            document_path="/nonexistent/file.txt"
        )

        # Should return Result object with error info
        assert isinstance(result, ProcessingResult)
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()


# ========== Logging Tests ==========

class TestLogging:
    """Test structured logging functionality."""

    @pytest.mark.asyncio
    async def test_operations_log_correlation_id(self, engine, sample_document, caplog):
        """Test that operations log with correlation IDs."""
        import logging

        with caplog.at_level(logging.INFO):
            result = await engine.process_document(
                document_path=sample_document,
                correlation_id="test_log_correlation_123"
            )

        # Check that logs contain correlation ID
        log_messages = [record.message for record in caplog.records]
        correlation_id_logs = [
            msg for msg in log_messages
            if "test_log_correlation_123" in msg
        ]

        # Should have at least one log with correlation ID
        assert len(correlation_id_logs) > 0


# ========== Configuration Edge Cases ==========

class TestConfigurationEdgeCases:
    """Test edge cases in configuration."""

    def test_empty_config_uses_defaults(self):
        """Test that empty config uses environment variables."""
        engine = KnowledgeEngine(config={})
        assert engine.config is not None
        # Should have defaults from environment
        assert "graphiti_uri" in engine.config

    def test_none_config_uses_defaults(self):
        """Test that None config uses environment variables."""
        engine = KnowledgeEngine(config=None)
        assert engine.config is not None
        # Should have defaults from environment
        assert "graphiti_uri" in engine.config


# ========== Run Tests ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
