"""
Graphiti Integration Test Suite

Comprehensive tests for Graphiti adapter and bridge integration with OpenEvolve.
Tests cover initialization, CRUD operations, search, community detection, and
graceful degradation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

# Import test targets
from integrations.graphiti.adapter import GraphitiAdapter, GRAPHITI_AVAILABLE
from integrations.graphiti.bridge import GraphitiBridge, get_bridge
from integrations.base.knowledge_interface import (
    TemporalFilter,
    ConfigurationError,
    ConnectionError,
    ValidationError,
    StorageError,
    SearchError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Mock Graphiti configuration."""
    return {
        'uri': 'bolt://localhost:7687',
        'user': 'neo4j',
        'password': 'test_password',
        'backend': 'neo4j',
        'store_raw_episode_content': True,
        'max_coroutines': 4,
    }


@pytest.fixture
def mock_episode_data():
    """Mock episode data for testing."""
    return {
        'name': 'test_episode',
        'body': 'This is a test episode about project kickoff',
        'reference_time': datetime.now(),
        'metadata': {'project': 'test_project', 'team_size': 5},
        'source': 'openevolve',
        'group_id': 'test_group',
    }


@pytest.fixture
def mock_search_results():
    """Mock search results."""
    return {
        'edges': [
            {
                'uuid': 'edge1',
                'fact': 'is related to',
                'source_node': 'node1',
                'target_node': 'node2',
                'created_at': datetime.now().isoformat(),
                'valid_at': datetime.now().isoformat(),
                'expired_at': None,
            }
        ],
        'nodes': [
            {
                'uuid': 'node1',
                'name': 'Project Kickoff',
                'summary': 'Initial project meeting',
                'labels': ['Event'],
            },
            {
                'uuid': 'node2',
                'name': 'Team',
                'summary': 'Project team members',
                'labels': ['Group'],
            }
        ],
        'context': ['Test context for search'],
    }


@pytest.fixture
def mock_community_results():
    """Mock community detection results."""
    return {
        'communities': [
            {
                'uuid': 'comm1',
                'name': 'Project Management',
                'summary': 'Project planning and execution activities',
            },
            {
                'uuid': 'comm2',
                'name': 'Development',
                'summary': 'Software development activities',
            },
        ],
        'community_edges': [
            {
                'uuid': 'ce1',
                'fact': 'collaborates with',
            }
        ],
        'metrics': {
            'num_communities': 2,
            'num_edges': 1,
        },
    }


# ============================================================================
# Adapter Tests
# ============================================================================

class TestGraphitiAdapter:
    """Test suite for GraphitiAdapter."""

    @pytest.mark.asyncio
    async def test_adapter_initialization_without_graphiti(self):
        """Test graceful degradation when Graphiti unavailable."""
        with patch('integrations.graphiti.adapter.GRAPHITI_AVAILABLE', False):
            adapter = GraphitiAdapter()

            with pytest.raises(ConfigurationError) as exc_info:
                await adapter.initialize(mock_config())

            assert "Graphiti is not available" in str(exc_info.value)

    @pytest.mark.skipif(not GRAPHITI_AVAILABLE, reason="Graphiti not installed")
    @pytest.mark.asyncio
    async def test_adapter_initialization_with_graphiti(self, mock_config):
        """Test successful adapter initialization with mocked Graphiti."""
        adapter = GraphitiAdapter()

        # Mock Graphiti class
        with patch('integrations.graphiti.adapter.Graphiti') as mock_graphiti:
            mock_instance = AsyncMock()
            mock_graphiti.return_value = mock_instance
            mock_instance.build_indices_and_constraints = AsyncMock()

            success = await adapter.initialize(mock_config)

            assert success is True
            assert adapter.is_initialized is True
            assert adapter.backend_type == 'neo4j'
            mock_graphiti.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_falkordb_backend(self):
        """Test adapter initialization with FalkorDB backend."""
        adapter = GraphitiAdapter()

        config = mock_config()
        config['backend'] = 'falkordb'

        with patch('integrations.graphiti.adapter.GRAPHITI_AVAILABLE', True):
            with patch('integrations.graphiti.adapter.Graphiti') as mock_graphiti:
                with patch('integrations.graphiti.adapter.FalkorDBDriver') as mock_falkor:
                    mock_instance = AsyncMock()
                    mock_graphiti.return_value = mock_instance
                    mock_instance.build_indices_and_constraints = AsyncMock()

                    success = await adapter.initialize(config)

                    assert success is True
                    assert adapter.backend_type == 'falkordb'

    @pytest.mark.asyncio
    async def test_add_episode_not_initialized(self):
        """Test adding episode when adapter not initialized."""
        adapter = GraphitiAdapter()

        with pytest.raises(StorageError) as exc_info:
            await adapter.add_episode(
                name='test',
                body='test body',
                reference_time=datetime.now()
            )

        assert "not initialized" in str(exc_info.value)

    @pytest.mark.skipif(not GRAPHITI_AVAILABLE, reason="Graphiti not installed")
    @pytest.mark.asyncio
    async def test_add_episode_success(self, mock_config, mock_episode_data):
        """Test successful episode addition."""
        adapter = GraphitiAdapter()

        with patch('integrations.graphiti.adapter.Graphiti') as mock_graphiti:
            # Setup mocks
            mock_instance = AsyncMock()
            mock_graphiti.return_value = mock_instance
            mock_instance.build_indices_and_constraints = AsyncMock()

            # Mock add_episode result
            mock_result = Mock()
            mock_result.episode.uuid = 'ep1'
            mock_result.episode.name = mock_episode_data['name']
            mock_result.episode.created_at = datetime.now()
            mock_result.episode.valid_at = datetime.now()
            mock_result.nodes = []
            mock_result.edges = []
            mock_result.communities = []

            mock_instance.add_episode = AsyncMock(return_value=mock_result)

            # Initialize
            await adapter.initialize(mock_config)

            # Add episode
            result = await adapter.add_episode(**mock_episode_data)

            assert result['uuid'] == 'ep1'
            assert result['name'] == mock_episode_data['name']
            assert 'nodes' in result
            assert 'edges' in result
            mock_instance.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_not_initialized(self):
        """Test searching when adapter not initialized."""
        adapter = GraphitiAdapter()

        with pytest.raises(SearchError) as exc_info:
            await adapter.search("test query")

        assert "not initialized" in str(exc_info.value)

    @pytest.mark.skipif(not GRAPHITI_AVAILABLE, reason="Graphiti not installed")
    @pytest.mark.asyncio
    async def test_search_success(self, mock_config, mock_search_results):
        """Test successful search."""
        adapter = GraphitiAdapter()

        with patch('integrations.graphiti.adapter.Graphiti') as mock_graphiti:
            # Setup mocks
            mock_instance = AsyncMock()
            mock_graphiti.return_value = mock_instance
            mock_instance.build_indices_and_constraints = AsyncMock()

            # Mock search result
            mock_search = Mock()
            mock_search.edges = [
                Mock(uuid='e1', fact='related to',
                     source_node_uuid='n1', target_node_uuid='n2',
                     created_at=datetime.now(), valid_at=datetime.now(),
                     expired_at=None)
            ]
            mock_search.nodes = [
                Mock(uuid='n1', name='Node1', summary='Summary1', labels=['A']),
                Mock(uuid='n2', name='Node2', summary='Summary2', labels=['B']),
            ]
            mock_search.context = ['context']

            mock_instance.search_ = AsyncMock(return_value=mock_search)

            # Initialize
            await adapter.initialize(mock_config)

            # Search
            results = await adapter.search("test query")

            assert len(results['nodes']) == 2
            assert len(results['edges']) == 1
            assert 'context' in results
            mock_instance.search_.assert_called_once()

    @pytest.mark.skipif(not GRAPHITI_AVAILABLE, reason="Graphiti not installed")
    @pytest.mark.asyncio
    async def test_search_with_temporal_filters(self, mock_config):
        """Test search with temporal filtering."""
        adapter = GraphitiAdapter()

        with patch('integrations.graphiti.adapter.Graphiti') as mock_graphiti:
            mock_instance = AsyncMock()
            mock_graphiti.return_value = mock_instance
            mock_instance.build_indices_and_constraints = AsyncMock()

            mock_search = Mock()
            mock_search.edges = []
            mock_search.nodes = []
            mock_search.context = []

            mock_instance.search_ = AsyncMock(return_value=mock_search)

            await adapter.initialize(mock_config)

            # Search with temporal filter
            await adapter.search(
                "test query",
                temporal_filters={
                    'filter_type': TemporalFilter.TIME_RANGE,
                    'start_time': datetime.now() - timedelta(days=7),
                    'end_time': datetime.now(),
                }
            )

            mock_instance.search_.assert_called_once()

    @pytest.mark.skipif(not GRAPHITI_AVAILABLE, reason="Graphiti not installed")
    @pytest.mark.asyncio
    async def test_community_detection_success(
        self, mock_config, mock_community_results
    ):
        """Test successful community detection."""
        adapter = GraphitiAdapter()

        with patch('integrations.graphiti.adapter.Graphiti') as mock_graphiti:
            mock_instance = AsyncMock()
            mock_graphiti.return_value = mock_instance
            mock_instance.build_indices_and_constraints = AsyncMock()

            # Mock build_communities result
            mock_communities = [
                Mock(uuid='c1', name='Comm1', summary='Summary1')
            ]
            mock_edges = [Mock(uuid='e1', fact='connects')]

            mock_instance.build_communities = AsyncMock(
                return_value=(mock_communities, mock_edges)
            )

            await adapter.initialize(mock_config)

            results = await adapter.get_community_detections()

            assert len(results['communities']) == 1
            assert len(results['community_edges']) == 1
            assert 'metrics' in results
            mock_instance.build_communities.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_not_initialized(self):
        """Test validation when adapter not initialized."""
        adapter = GraphitiAdapter()

        results = await adapter.validate()

        assert results['is_valid'] is False
        assert 'initialized' in results['checks']
        assert results['checks']['initialized'] is False

    @pytest.mark.skipif(not GRAPHITI_AVAILABLE, reason="Graphiti not installed")
    @pytest.mark.asyncio
    async def test_validate_success(self, mock_config):
        """Test successful validation."""
        adapter = GraphitiAdapter()

        with patch('integrations.graphiti.adapter.Graphiti') as mock_graphiti:
            mock_instance = AsyncMock()
            mock_graphiti.return_value = mock_instance
            mock_instance.build_indices_and_constraints = AsyncMock()
            mock_instance.search = AsyncMock(return_value=Mock())

            await adapter.initialize(mock_config)
            results = await adapter.validate()

            assert results['is_valid'] is True
            assert results['checks']['backend_connected'] is True
            assert results['checks']['search_operational'] is True

    @pytest.mark.asyncio
    async def test_shutdown_not_initialized(self):
        """Test shutdown when adapter not initialized."""
        adapter = GraphitiAdapter()

        # Should return True without error
        result = await adapter.shutdown()
        assert result is True

    @pytest.mark.skipif(not GRAPHITI_AVAILABLE, reason="Graphiti not installed")
    @pytest.mark.asyncio
    async def test_shutdown_success(self, mock_config):
        """Test successful shutdown."""
        adapter = GraphitiAdapter()

        with patch('integrations.graphiti.adapter.Graphiti') as mock_graphiti:
            mock_instance = AsyncMock()
            mock_graphiti.return_value = mock_instance
            mock_instance.build_indices_and_constraints = AsyncMock()
            mock_instance.close = AsyncMock()

            await adapter.initialize(mock_config)
            result = await adapter.shutdown()

            assert result is True
            assert adapter.is_initialized is False
            mock_instance.close.assert_called_once()


# ============================================================================
# Bridge Tests
# ============================================================================

class TestGraphitiBridge:
    """Test suite for GraphitiBridge."""

    def test_singleton_pattern(self):
        """Test that bridge implements singleton pattern."""
        bridge1 = GraphitiBridge()
        bridge2 = GraphitiBridge()

        assert bridge1 is bridge2

    @pytest.mark.asyncio
    async def test_get_bridge_singleton(self):
        """Test get_bridge returns singleton."""
        bridge1 = await get_bridge()
        bridge2 = await get_bridge()

        assert bridge1 is bridge2

    @pytest.mark.asyncio
    async def test_load_config_success(self, tmp_path):
        """Test successful config loading."""
        # Create temp config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
project:
  name: Graphiti
  enabled: true

connection:
  backend: neo4j
  uri: bolt://localhost:7687
  user: neo4j
  password: test_password

integration:
  auto_start: true
  cache_enabled: true
  cache_ttl: 3600
"""
        config_file.write_text(config_content)

        bridge = GraphitiBridge()
        success = await bridge.load_config(str(config_file))

        assert success is True
        assert bridge._config is not None
        assert bridge._cache_enabled is True
        assert bridge._cache_ttl == 3600

    @pytest.mark.asyncio
    async def test_load_config_file_not_found(self):
        """Test config loading with non-existent file."""
        bridge = GraphitiBridge()
        success = await bridge.load_config("nonexistent.yaml")

        assert success is False

    def test_resolve_env_var(self):
        """Test environment variable resolution."""
        bridge = GraphitiBridge()

        # Test env var
        with patch.dict('os.environ', {'TEST_VAR': 'resolved_value'}):
            result = bridge._resolve_env_var('${TEST_VAR}')
            assert result == 'resolved_value'

        # Test non-env var
        result = bridge._resolve_env_var('plain_value')
        assert result == 'plain_value'

    @pytest.mark.asyncio
    async def test_initialize_without_config(self):
        """Test initialization without config."""
        bridge = GraphitiBridge()

        # Should return False gracefully
        result = await bridge.initialize()
        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_with_auto_start_disabled(self, tmp_path):
        """Test initialization with auto_start disabled."""
        # Create config with auto_start: false
        config_file = tmp_path / "config.yaml"
        config_content = """
project:
  enabled: true

connection:
  backend: neo4j
  uri: bolt://localhost:7687
  user: neo4j
  password: test

integration:
  auto_start: false
"""
        config_file.write_text(config_content)

        bridge = GraphitiBridge()
        await bridge.load_config(str(config_file))

        result = await bridge.initialize()
        assert result is False  # Should not initialize

    @pytest.mark.asyncio
    async def test_add_episode_without_adapter(self, mock_episode_data):
        """Test adding episode when adapter not initialized."""
        bridge = GraphitiBridge()

        # Should return empty dict without error
        result = await bridge.add_episode(**mock_episode_data)
        assert result == {}

    @pytest.mark.asyncio
    async def test_search_without_adapter(self):
        """Test search when adapter not initialized."""
        bridge = GraphitiBridge()

        results = await bridge.search("test query")

        assert results['edges'] == []
        assert results['nodes'] == []
        assert results['context'] == []

    @pytest.mark.asyncio
    async def test_search_caching(self, tmp_path, mock_search_results):
        """Test search caching functionality."""
        # Create config with caching enabled
        config_file = tmp_path / "config.yaml"
        config_content = """
project:
  enabled: true

connection:
  backend: neo4j
  uri: bolt://localhost:7687
  user: neo4j
  password: test

integration:
  cache_enabled: true
  cache_ttl: 3600
"""
        config_file.write_text(config_content)

        bridge = GraphitiBridge()
        await bridge.load_config(str(config_file))

        # Mock adapter
        bridge._adapter = AsyncMock()
        bridge._adapter.is_initialized = True
        bridge._adapter.search = AsyncMock(return_value=mock_search_results)

        # First search
        results1 = await bridge.search("test query")

        # Second search (should use cache)
        results2 = await bridge.search("test query")

        assert results1 == results2
        # Should only call adapter once due to caching
        bridge._adapter.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_community_detections_without_adapter(self):
        """Test community detection when adapter not initialized."""
        bridge = GraphitiBridge()

        results = await bridge.get_community_detections()

        assert results['communities'] == []
        assert results['community_edges'] == []
        assert results['metrics'] == {}

    @pytest.mark.asyncio
    async def test_validate_without_adapter(self):
        """Test validation when adapter not initialized."""
        bridge = GraphitiBridge()

        results = await bridge.validate()

        assert results['is_valid'] is False
        assert len(results['issues']) > 0

    @pytest.mark.asyncio
    async def test_shutdown_without_adapter(self):
        """Test shutdown when adapter not initialized."""
        bridge = GraphitiBridge()

        # Should return True without error
        result = await bridge.shutdown()
        assert result is True

    def test_is_available(self):
        """Test availability check."""
        bridge = GraphitiBridge()

        with patch('integrations.graphiti.adapter.GRAPHITI_AVAILABLE', True):
            assert bridge.is_available is True

        with patch('integrations.graphiti.adapter.GRAPHITI_AVAILABLE', False):
            assert bridge.is_available is False

    def test_is_initialized(self):
        """Test initialization status check."""
        bridge = GraphitiBridge()

        assert bridge.is_initialized is False

        # Mock adapter
        bridge._adapter = Mock()
        bridge._adapter.is_initialized = True

        assert bridge.is_initialized is True


# ============================================================================
# Integration Tests
# ============================================================================

class TestGraphitiIntegration:
    """Integration tests for Graphiti with OpenEvolve."""

    @pytest.mark.asyncio
    async def test_full_workflow_mock(self, mock_episode_data):
        """Test complete workflow with mocked Graphiti."""
        if not GRAPHITI_AVAILABLE:
            pytest.skip("Graphiti not installed")

        # Get bridge
        bridge = await get_bridge()

        # Mock adapter
        with patch('integrations.graphiti.adapter.Graphiti') as mock_graphiti:
            mock_instance = AsyncMock()
            mock_graphiti.return_value = mock_instance
            mock_instance.build_indices_and_constraints = AsyncMock()

            # Mock add_episode
            mock_result = Mock()
            mock_result.episode.uuid = 'ep1'
            mock_result.episode.name = mock_episode_data['name']
            mock_result.episode.created_at = datetime.now()
            mock_result.episode.valid_at = datetime.now()
            mock_result.nodes = []
            mock_result.edges = []
            mock_result.communities = []

            mock_instance.add_episode = AsyncMock(return_value=mock_result)

            # Mock search
            mock_search = Mock()
            mock_search.edges = []
            mock_search.nodes = []
            mock_search.context = []
            mock_instance.search_ = AsyncMock(return_value=mock_search)

            # Initialize
            bridge._adapter = GraphitiAdapter()
            await bridge._adapter.initialize(mock_config())

            # Add episode
            result = await bridge.add_episode(**mock_episode_data)
            assert result['uuid'] == 'ep1'

            # Search
            search_results = await bridge.search("test")
            assert 'edges' in search_results
            assert 'nodes' in search_results

            # Validate
            validation = await bridge.validate()
            assert validation['is_valid'] is True

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations on bridge."""
        bridge = GraphitiBridge()

        # Mock adapter
        bridge._adapter = AsyncMock()
        bridge._adapter.is_initialized = True
        bridge._adapter.search = AsyncMock(return_value={
            'edges': [], 'nodes': [], 'context': []
        })

        # Run concurrent searches
        tasks = [bridge.search(f"query_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all('edges' in r for r in results)

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when Graphiti unavailable."""
        with patch('integrations.graphiti.adapter.GRAPHITI_AVAILABLE', False):
            bridge = GraphitiBridge()

            # Should not raise errors
            await bridge.initialize()

            result = await bridge.add_episode(
                name='test',
                body='test',
                reference_time=datetime.now()
            )
            assert result == {}

            results = await bridge.search("test")
            assert results['edges'] == []

            validation = await bridge.validate()
            assert validation['is_valid'] is False


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
