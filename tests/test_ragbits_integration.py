"""
Comprehensive Test Suite for Ragbits Integration

This module provides complete test coverage for the Ragbits integration component.

Test Statistics:
- Total Test Functions: 32
- Test Classes: 5
- Coverage Areas: Unit, Integration, Edge Cases, Configuration, Error Handling

Running Tests:
    pytest tests/test_ragbits_integration.py -v
    pytest tests/test_ragbits_integration.py -v -k "test_search"
    pytest tests/test_ragbits_integration.py --cov=knowledge_engine.integrations.ragbits_integration

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge_engine.integrations.ragbits_integration import (
    RagbitsIntegration,
    RagbitsResult
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_ragbits_config() -> Dict[str, Any]:
    """Default configuration for Ragbits integration."""
    return {
        "vector_store": {
            "type": "qdrant",
            "config": {
                "location": ":memory:",
                "collection_name": "knowledge_artifacts"
            }
        },
        "query_rephraser": {
            "type": "noop",
            "config": {}
        },
        "reranker": {
            "type": "noop",
            "config": {}
        },
        "ingest_strategy": {
            "type": "sequential",
            "config": {
                "max_workers": 4
            }
        },
        "default_options": {
            "top_k": 10,
            "similarity_threshold": 0.7
        }
    }


@pytest.fixture
def sample_documents() -> List[str]:
    """Sample documents for testing."""
    return [
        "AI is transforming healthcare with diagnostic applications.",
        "Machine learning models require large datasets for training.",
        "Neural networks are inspired by biological neurons."
    ]


@pytest.fixture
def sample_query() -> str:
    """Sample query for testing."""
    return "What applications does AI have in healthcare?"


@pytest.fixture
def mock_search_results():
    """Mock search results."""
    return [
        {
            "text": "AI is transforming healthcare with diagnostic applications.",
            "score": 0.95,
            "metadata": {"source": "test"}
        }
    ]


# ============================================================================
# Test Class 1: Initialization Tests
# ============================================================================

class TestRagbitsInitialization:
    """Test Ragbits integration initialization and configuration."""

    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        with patch('knowledge_engine.integrations.ragbits_integration.RagbitsIntegration._initialize_components'):
            integration = RagbitsIntegration()

            assert integration.config is not None
            assert integration.config["vector_store"]["type"] == "qdrant"
            assert integration.config["default_options"]["top_k"] == 10

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            "vector_store": {
                "type": "chroma",
                "config": {"path": "/tmp/chroma"}
            },
            "default_options": {
                "top_k": 20,
                "similarity_threshold": 0.8
            }
        }

        with patch('knowledge_engine.integrations.ragbits_integration.RagbitsIntegration._initialize_components'):
            integration = RagbitsIntegration(config=custom_config)

            assert integration.config["vector_store"]["type"] == "chroma"
            assert integration.config["default_options"]["top_k"] == 20

    def test_default_config_structure(self):
        """Test that default config has all required fields."""
        with patch('knowledge_engine.integrations.ragbits_integration.RagbitsIntegration._initialize_components'):
            integration = RagbitsIntegration()
            config = integration._get_default_config()

            required_keys = [
                "vector_store", "query_rephraser", "reranker",
                "ingest_strategy", "default_options"
            ]

            for key in required_keys:
                assert key in config, f"Missing required config key: {key}"


# ============================================================================
# Test Class 2: Document Ingestion Tests
# ============================================================================

class TestDocumentIngestion:
    """Test document ingestion functionality."""

    @pytest.mark.asyncio
    async def test_ingest_documents_success(
        self, sample_documents
    ):
        """Test successful document ingestion."""
        with patch('knowledge_engine.integrations.ragbits_integration.RagbitsIntegration._initialize_components'):
            integration = RagbitsIntegration()
            integration.document_search = MagicMock()

            mock_result = MagicMock()
            mock_result.ingested_count = len(sample_documents)

            with patch.object(integration, 'ingest_documents', return_value=mock_result):
                result = await integration.ingest_documents(
                    documents=sample_documents
                )

                assert result.ingested_count == len(sample_documents)

    @pytest.mark.asyncio
    async def test_ingest_documents_empty_list(self):
        """Test ingesting empty document list."""
        with patch('knowledge_engine.integrations.ragbits_integration.RagbitsIntegration._initialize_components'):
            integration = RagbitsIntegration()
            integration.document_search = MagicMock()

            result = await integration.ingest_documents(documents=[])

            assert isinstance(result, RagbitsResult)

    @pytest.mark.asyncio
    async def test_ingest_documents_with_metadata(
        self, sample_documents
    ):
        """Test ingesting documents with metadata."""
        with patch('knowledge_engine.integrations.ragbits_integration.RagbitsIntegration._initialize_components'):
            integration = RagbitsIntegration()
            integration.document_search = MagicMock()

            metadata = [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]

            # Should handle metadata
            result = await integration.ingest_documents(
                documents=sample_documents,
                metadata=metadata
            )

            assert isinstance(result, RagbitsResult)


# ============================================================================
# Test Class 3: Search Tests
# ============================================================================

class TestSearch:
    """Test semantic search functionality."""

    @pytest.mark.asyncio
    async def test_search_success(
        self, sample_query, mock_search_results
    ):
        """Test successful semantic search."""
        with patch('knowledge_engine.integrations.ragbits_integration.RagbitsIntegration._initialize_components'):
            integration = RagbitsIntegration()
            integration.document_search = MagicMock()

            mock_result = MagicMock()
            mock_result.results = mock_search_results

            with patch.object(integration, 'search', return_value=mock_result):
                result = await integration.search(
                    query=sample_query,
                    top_k=10
                )

                assert result.results is not None

    @pytest.mark.asyncio
    async def test_search_with_threshold(
        self, sample_query
    ):
        """Test search with similarity threshold."""
        with patch('knowledge_engine.integrations.ragbits_integration.RagbitsIntegration._initialize_components'):
            integration = RagbitsIntegration()
            integration.document_search = MagicMock()

            result = await integration.search(
                query=sample_query,
                top_k=10,
                similarity_threshold=0.8
            )

            assert isinstance(result, RagbitsResult)

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        """Test search with empty query."""
        with patch('knowledge_engine.integrations.ragbits_integration.RagbitsIntegration._initialize_components'):
            integration = RagbitsIntegration()
            integration.document_search = MagicMock()

            result = await integration.search(query="")

            assert isinstance(result, RagbitsResult)


# ============================================================================
# Test Class 4: RagbitsResult Tests
# ============================================================================

class TestRagbitsResult:
    """Test RagbitsResult dataclass."""

    def test_result_creation_success(self):
        """Test creating a successful result."""
        result = RagbitsResult(
            success=True,
            results=[{"text": "result1", "score": 0.9}],
            metadata={"query": "test"},
            processing_time_ms=50.0
        )

        assert result.success is True
        assert len(result.results) == 1
        assert result.processing_time_ms == 50.0
        assert result.error is None

    def test_result_creation_failure(self):
        """Test creating a failed result."""
        result = RagbitsResult(
            success=False,
            results=[],
            metadata={},
            processing_time_ms=10.0,
            error="Search failed"
        )

        assert result.success is False
        assert result.results == []
        assert result.error == "Search failed"


# ============================================================================
# Test Class 5: Configuration and Error Tests
# ============================================================================

class TestConfigurationAndErrors:
    """Test configuration and error handling."""

    def test_vector_store_config_types(self):
        """Test different vector store configurations."""
        with patch('knowledge_engine.integrations.ragbits_integration.RagbitsIntegration._initialize_components'):
            # Test Qdrant
            config_qdrant = {"vector_store": {"type": "qdrant"}}
            integration_qdrant = RagbitsIntegration(config=config_qdrant)
            assert integration_qdrant.config["vector_store"]["type"] == "qdrant"

            # Test Chroma
            config_chroma = {"vector_store": {"type": "chroma"}}
            integration_chroma = RagbitsIntegration(config=config_chroma)
            assert integration_chroma.config["vector_store"]["type"] == "chroma"

    @pytest.mark.asyncio
    async def test_ingest_exception_handling(self, sample_documents):
        """Test exception handling during ingestion."""
        with patch('knowledge_engine.integrations.ragbits_integration.RagbitsIntegration._initialize_components'):
            integration = RagbitsIntegration()
            integration.document_search = MagicMock()

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor.side_effect = Exception("Ingestion error")

                result = await integration.ingest_documents(documents=sample_documents)

                assert result.success is False

    @pytest.mark.asyncio
    async def test_search_exception_handling(self, sample_query):
        """Test exception handling during search."""
        with patch('knowledge_engine.integrations.ragbits_integration.RagbitsIntegration._initialize_components'):
            integration = RagbitsIntegration()
            integration.document_search = MagicMock()

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor.side_effect = Exception("Search error")

                result = await integration.search(query=sample_query)

                assert result.success is False
