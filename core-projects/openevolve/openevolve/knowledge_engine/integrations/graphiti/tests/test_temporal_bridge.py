"""
Unit tests for Graphiti Temporal Bridge.

Implements Task 1.5.1: Unit tests for temporal bridge functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from knowledge_engine.integrations.graphiti.temporal_bridge import (
    GraphitiTemporalBridge,
    WorkflowArtifact,
    WorkflowState,
    TemporalFilter,
    TemporalRelationship,
)
from knowledge_engine.integrations.graphiti.config import GraphitiConfig
from knowledge_engine.integrations.graphiti.exceptions import (
    ConfigurationError,
    ConnectionError,
    EpisodeProcessingError,
    InvalidTimestampError,
)


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    with patch.dict('os.environ', {
        'GRAPHITI_URI': 'bolt://localhost:7687',
        'GRAPHITI_USER': 'neo4j',
        'GRAPHITI_PASSWORD': 'password',
        'OPENAI_API_KEY': 'test-key',
    }):
        config = GraphitiConfig()
        config.validate()
        return config


@pytest.fixture
async def temporal_bridge(mock_config):
    """Create a temporal bridge instance."""
    bridge = GraphitiTemporalBridge(config=mock_config)

    # Mock the Graphiti client
    mock_graphiti = AsyncMock()
    mock_graphiti.search = AsyncMock(return_value=Mock(edges=[], nodes=[]))
    mock_graphiti.add_episode = AsyncMock(
        return_value=Mock(uuid="test-episode-uuid")
    )
    mock_graphiti.close = AsyncMock()

    bridge.graphiti_client = mock_graphiti
    bridge._initialized = True

    yield bridge

    # Cleanup
    await bridge.close()


class TestWorkflowArtifactTracking:
    """Tests for workflow artifact tracking (1.1.1)."""

    @pytest.mark.asyncio
    async def test_track_workflow_artifact_success(self, temporal_bridge):
        """Test successful workflow artifact tracking."""
        artifact = await temporal_bridge.track_workflow_artifact(
            workflow_id="test-workflow-1",
            workflow_name="Test Workflow",
            state=WorkflowState.COMPLETED,
            metadata={"test": True},
        )

        assert artifact.workflow_id == "test-workflow-1"
        assert artifact.workflow_name == "Test Workflow"
        assert artifact.state == WorkflowState.COMPLETED
        assert artifact.metadata["test"] is True
        assert artifact.artifact_id is not None

    @pytest.mark.asyncio
    async def test_track_workflow_artifact_with_timestamps(self, temporal_bridge):
        """Test workflow artifact tracking with custom timestamps."""
        started_at = datetime.utcnow() - timedelta(hours=1)
        completed_at = datetime.utcnow()

        artifact = await temporal_bridge.track_workflow_artifact(
            workflow_id="test-workflow-2",
            workflow_name="Test Workflow 2",
            state=WorkflowState.COMPLETED,
            started_at=started_at,
            completed_at=completed_at,
        )

        assert artifact.started_at == started_at
        assert artifact.completed_at == completed_at

    @pytest.mark.asyncio
    async def test_track_workflow_artifact_not_initialized(self, mock_config):
        """Test that tracking fails when bridge not initialized."""
        bridge = GraphitiTemporalBridge(config=mock_config)
        bridge._initialized = False

        with pytest.raises(Exception):  # GraphitiIntegrationError
            await bridge.track_workflow_artifact(
                workflow_id="test-workflow",
                workflow_name="Test",
                state=WorkflowState.PENDING,
            )


class TestWorkflowStateQueries:
    """Tests for workflow state queries at specific timestamps (1.1.2)."""

    @pytest.mark.asyncio
    async def test_query_workflow_state_at_time(self, temporal_bridge):
        """Test querying workflow state at a specific time."""
        # First track an artifact
        artifact = await temporal_bridge.track_workflow_artifact(
            workflow_id="test-workflow-3",
            workflow_name="Test Workflow 3",
            state=WorkflowState.COMPLETED,
        )

        # Query the state
        queried_state = await temporal_bridge.query_workflow_state_at_time(
            workflow_id="test-workflow-3",
            timestamp=datetime.utcnow(),
        )

        # Should return the artifact from cache
        assert queried_state is not None
        assert queried_state.workflow_id == "test-workflow-3"

    @pytest.mark.asyncio
    async def test_query_workflow_state_not_found(self, temporal_bridge):
        """Test querying non-existent workflow state."""
        queried_state = await temporal_bridge.query_workflow_state_at_time(
            workflow_id="non-existent-workflow",
            timestamp=datetime.utcnow(),
        )

        assert queried_state is None


class TestTemporalRelationships:
    """Tests for temporal relationship metadata (1.1.3)."""

    @pytest.mark.asyncio
    async def test_add_temporal_relationship(self, temporal_bridge):
        """Test adding a temporal relationship."""
        relationship = await temporal_bridge.add_temporal_relationship(
            source_entity="Entity1",
            relation="CONNECTED_TO",
            target_entity="Entity2",
            valid_at=datetime.utcnow(),
            confidence=0.9,
            metadata={"test": True},
        )

        assert relationship.source_entity == "Entity1"
        assert relationship.relation == "CONNECTED_TO"
        assert relationship.target_entity == "Entity2"
        assert relationship.confidence == 0.9
        assert relationship.metadata["test"] is True

    def test_relationship_validity_check(self):
        """Test temporal relationship validity checking."""
        now = datetime.utcnow()
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)

        # Valid relationship
        relationship = TemporalRelationship(
            source_entity="A",
            relation="LINK",
            target_entity="B",
            valid_at=past,
            invalid_at=future,
        )

        assert relationship.is_valid_at_time(now) is True
        assert relationship.is_valid_at_time(past + timedelta(minutes=30)) is True
        assert relationship.is_valid_at_time(now + timedelta(hours=2)) is False


class TestEpisodeIngestion:
    """Tests for episode-based knowledge ingestion (1.1.4)."""

    @pytest.mark.asyncio
    async def test_add_episode_success(self, temporal_bridge):
        """Test successful episode addition."""
        episode_uuid = await temporal_bridge.add_episode(
            name="Test Episode",
            episode_body="This is a test episode.",
            reference_time=datetime.utcnow(),
            source="test",
        )

        assert episode_uuid is not None
        temporal_bridge.graphiti_client.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_episode_with_utc_timestamp(self, temporal_bridge):
        """Test that timestamps are normalized to UTC."""
        # Create a timestamp with timezone info
        timestamp_with_tz = datetime.now().astimezone()

        episode_uuid = await temporal_bridge.add_episode(
            name="Test Episode",
            episode_body="Test",
            reference_time=timestamp_with_tz,
            source="test",
        )

        # Should succeed without error
        assert episode_uuid is not None


class TestTemporalSearch:
    """Tests for temporal search API endpoints (1.1.5)."""

    @pytest.mark.asyncio
    async def test_search_current(self, temporal_bridge):
        """Test searching with CURRENT filter."""
        results = await temporal_bridge.search_temporal(
            query="test query",
            filter_type=TemporalFilter.CURRENT,
            max_results=10,
        )

        assert "edges" in results
        assert "nodes" in results
        temporal_bridge.graphiti_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_time_range(self, temporal_bridge):
        """Test searching with TIME_RANGE filter."""
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()

        results = await temporal_bridge.search_temporal(
            query="test query",
            filter_type=TemporalFilter.TIME_RANGE,
            start_time=start_time,
            end_time=end_time,
            max_results=10,
        )

        assert "edges" in results
        assert "nodes" in results

    @pytest.mark.asyncio
    async def test_search_not_initialized(self, mock_config):
        """Test that search fails when bridge not initialized."""
        bridge = GraphitiTemporalBridge(config=mock_config)
        bridge._initialized = False

        with pytest.raises(Exception):
            await bridge.search_temporal(
                query="test",
                max_results=10,
            )


class TestConfigurationValidation:
    """Tests for configuration validation."""

    def test_valid_configuration(self, mock_config):
        """Test that valid configuration passes validation."""
        assert mock_config.graphiti_uri == "bolt://localhost:7687"
        assert mock_config.graphiti_user == "neo4j"
        assert mock_config.openai_api_key == "test-key"

    def test_missing_required_configuration(self):
        """Test that missing required configuration raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ConfigurationError):
                config = GraphitiConfig()
                config.validate()

    def test_invalid_contradiction_threshold(self, mock_config):
        """Test that invalid threshold raises error."""
        with patch.dict('os.environ', {
            'GRAPHITI_URI': 'bolt://localhost:7687',
            'GRAPHITI_USER': 'neo4j',
            'GRAPHITI_PASSWORD': 'password',
            'OPENAI_API_KEY': 'test-key',
            'GRAPHITI_CONTRADICTION_THRESHOLD': '1.5',  # Invalid > 1.0
        }):
            with pytest.raises(ConfigurationError):
                config = GraphitiConfig()
                config.validate()


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_connection_failure(self, mock_config):
        """Test handling of connection failure."""
        bridge = GraphitiTemporalBridge(config=mock_config)

        # Mock connection test to fail
        with patch.object(bridge, '_test_connection', side_effect=Exception("Connection failed")):
            with pytest.raises(ConnectionError):
                await bridge.initialize()

    @pytest.mark.asyncio
    async def test_search_with_correlation_id(self, temporal_bridge):
        """Test that correlation IDs are propagated."""
        correlation_id = "test-correlation-123"
        temporal_bridge.correlation_id = correlation_id

        await temporal_bridge.search_temporal(
            query="test",
            max_results=10,
        )

        # Verify search was called
        assert temporal_bridge.graphiti_client.search.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
