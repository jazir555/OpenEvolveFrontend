
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from bubblelabs_integration import BubbleLabsIntegration
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestBubbleLabsRagbitsIntegration:
    """
    Test suite for BubbleLabs integration with Ragbits.
    """

    @pytest.fixture
    def mock_ragbits_integration(self):
        """Mock RagbitsIntegration."""
        mock = MagicMock()
        mock.search_solutions = AsyncMock(return_value=[{"content": "test", "score": 0.9}])
        mock.retrieve_similar_solutions = AsyncMock(return_value=[
            MagicMock(to_dict=lambda: {"problem": "p", "similarity_score": 0.8})
        ])
        mock.ragbits_integration = MagicMock()
        mock.ragbits_integration.ingest_documents = AsyncMock(return_value=MagicMock(success=True, results=[{"ingested_count": 1}]))
        return mock

    @pytest.fixture
    def bubblelabs_integration(self):
        """BubbleLabsIntegration instance."""
        with patch('bubblelabs_integration._get_api_server_managers', return_value=(MagicMock(), MagicMock())):
            integration = BubbleLabsIntegration()
            return integration

    @pytest.mark.asyncio
    async def test_get_ragbits_integration_lazy_init(self, bubblelabs_integration, mock_ragbits_integration):
        """Test lazy initialization of Ragbits integration."""
        with patch('bubblelabs_integration.get_roma_ragbits_integration', return_value=mock_ragbits_integration) as mock_get:
            # First call should initialize
            integration = bubblelabs_integration.get_ragbits_integration()
            assert integration == mock_ragbits_integration
            mock_get.assert_called_once()
            
            # Second call should use cached instance
            integration2 = bubblelabs_integration.get_ragbits_integration()
            assert integration2 == mock_ragbits_integration
            mock_get.assert_called_once()  # Still called once

    @pytest.mark.asyncio
    async def test_search_knowledge_success(self, bubblelabs_integration, mock_ragbits_integration):
        """Test searching knowledge base."""
        with patch('bubblelabs_integration.get_roma_ragbits_integration', return_value=mock_ragbits_integration):
            # Ensure integration is initialized
            bubblelabs_integration.get_ragbits_integration()
            
            results = await bubblelabs_integration.search_knowledge("test query", top_k=3)
            
            assert len(results) == 1
            assert results[0]["content"] == "test"
            mock_ragbits_integration.search_solutions.assert_called_with("test query", 3)

    @pytest.mark.asyncio
    async def test_retrieve_similar_solutions(self, bubblelabs_integration, mock_ragbits_integration):
        """Test retrieving similar solutions."""
        with patch('bubblelabs_integration.get_roma_ragbits_integration', return_value=mock_ragbits_integration):
            # Ensure integration is initialized
            bubblelabs_integration.get_ragbits_integration()
            
            results = await bubblelabs_integration.retrieve_similar_solutions("problem", top_k=2)
            
            assert len(results) == 1
            mock_ragbits_integration.retrieve_similar_solutions.assert_called_with("problem", 2, None)

    @pytest.mark.asyncio
    async def test_index_solution(self, bubblelabs_integration, mock_ragbits_integration):
        """Test indexing a solution."""
        with patch('bubblelabs_integration.get_roma_ragbits_integration', return_value=mock_ragbits_integration):
            # Ensure integration is initialized
            bubblelabs_integration.get_ragbits_integration()
            
            result = await bubblelabs_integration.index_solution({"content": "new solution"})
            
            assert result == "indexed_1_docs"
            mock_ragbits_integration.ragbits_integration.ingest_documents.assert_called()

    @pytest.mark.asyncio
    async def test_unavailable_ragbits(self, bubblelabs_integration):
        """Test behavior when Ragbits is unavailable."""
        # Patch RAGBITS_AVAILABLE to False in bubblelabs_integration module
        with patch('bubblelabs_integration.RAGBITS_AVAILABLE', False):
            # Force re-check
            if hasattr(bubblelabs_integration, '_ragbits_integration'):
                del bubblelabs_integration._ragbits_integration
                
            integration = bubblelabs_integration.get_ragbits_integration()
            assert integration is None
            
            # Methods should return empty/None safely
            results = await bubblelabs_integration.search_knowledge("query")
            assert results == []
            
            index_res = await bubblelabs_integration.index_solution({})
            assert index_res is None
