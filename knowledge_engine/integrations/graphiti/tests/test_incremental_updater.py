"""
Unit tests for Graphiti Incremental Updater.

Implements Task 1.5.3: Unit tests for incremental updater functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from knowledge_engine.integrations.graphiti.incremental_updater import (
    GraphitiIncrementalUpdater,
    UpdateType,
    UpdateStatus,
    GraphUpdate,
    EntityMergeResult,
)
from knowledge_engine.integrations.graphiti.config import GraphitiConfig
from knowledge_engine.integrations.graphiti.exceptions import IncrementalUpdateError


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    with patch.dict('os.environ', {
        'GRAPHITI_URI': 'bolt://localhost:7687',
        'GRAPHITI_USER': 'neo4j',
        'GRAPHITI_PASSWORD': 'password',
        'OPENAI_API_KEY': 'test-key',
        'GRAPHITI_INCREMENTAL_UPDATES_ENABLED': 'true',
    }):
        config = GraphitiConfig()
        config.validate()
        return config


@pytest.fixture
def mock_temporal_bridge(mock_config):
    """Create a mock temporal bridge."""
    bridge = Mock()
    bridge._initialized = True
    bridge.add_episode = AsyncMock(return_value="episode-uuid-123")
    bridge.search_temporal = AsyncMock(
        return_value={"edges": [], "nodes": []}
    )
    return bridge


@pytest.fixture
def incremental_updater(mock_config, mock_temporal_bridge):
    """Create an incremental updater instance."""
    updater = GraphitiIncrementalUpdater(config=mock_config)
    updater.set_bridge(mock_temporal_bridge)
    return updater


class TestEntityOperations:
    """Tests for entity add and update operations."""

    @pytest.mark.asyncio
    async def test_add_entity(self, incremental_updater):
        """Test adding an entity."""
        update = await incremental_updater.add_entity(
            entity_name="TestEntity",
            entity_type="TestType",
            attributes={"key": "value"},
        )

        assert update.update_type == UpdateType.ENTITY_ADD
        assert update.status == UpdateStatus.COMPLETED
        assert "TestEntity" in update.affected_entities

    @pytest.mark.asyncio
    async def test_add_entity_with_timestamp(self, incremental_updater):
        """Test adding an entity with custom timestamp."""
        timestamp = datetime.utcnow() - timedelta(hours=1)

        update = await incremental_updater.add_entity(
            entity_name="TestEntity",
            entity_type="TestType",
            timestamp=timestamp,
        )

        assert update.status == UpdateStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_update_entity(self, incremental_updater):
        """Test updating an entity."""
        update = await incremental_updater.update_entity(
            entity_name="TestEntity",
            new_attributes={"updated_key": "updated_value"},
        )

        assert update.update_type == UpdateType.ENTITY_UPDATE
        assert update.status == UpdateStatus.COMPLETED
        assert "TestEntity" in update.affected_entities

    @pytest.mark.asyncio
    async def test_add_entity_disabled(self, mock_config):
        """Test that adding entity fails when disabled."""
        with patch.dict('os.environ', {
            'GRAPHITI_URI': 'bolt://localhost:7687',
            'GRAPHITI_USER': 'neo4j',
            'GRAPHITI_PASSWORD': 'password',
            'OPENAI_API_KEY': 'test-key',
            'GRAPHITI_INCREMENTAL_UPDATES_ENABLED': 'false',
        }):
            config = GraphitiConfig()
            config.validate()

            updater = GraphitiIncrementalUpdater(config=config)

            with pytest.raises(IncrementalUpdateError):
                await updater.add_entity(
                    entity_name="TestEntity",
                    entity_type="TestType",
                )


class TestEdgeInvalidation:
    """Tests for edge invalidation operations."""

    @pytest.mark.asyncio
    async def test_invalidate_edge(self, incremental_updater):
        """Test invalidating an edge."""
        update = await incremental_updater.invalidate_edge(
            source_entity="Entity1",
            relation="CONNECTED_TO",
            target_entity="Entity2",
            reason="Test invalidation",
        )

        assert update.update_type == UpdateType.EDGE_INVALIDATE
        assert update.status == UpdateStatus.COMPLETED
        assert "Entity1" in update.affected_entities
        assert "Entity2" in update.affected_entities
        assert update.metadata["reason"] == "Test invalidation"

    @pytest.mark.asyncio
    async def test_invalidate_edge_with_timestamp(self, incremental_updater):
        """Test invalidating an edge with custom invalidation time."""
        invalidation_time = datetime.utcnow() + timedelta(hours=1)

        update = await incremental_updater.invalidate_edge(
            source_entity="Entity1",
            relation="CONNECTED_TO",
            target_entity="Entity2",
            invalidation_time=invalidation_time,
        )

        assert update.status == UpdateStatus.COMPLETED


class TestEntityMerging:
    """Tests for entity merging operations."""

    @pytest.mark.asyncio
    async def test_find_duplicate_entities(self, incremental_updater):
        """Test finding duplicate entities."""
        # Mock search results
        incremental_updater.temporal_bridge.search_temporal = AsyncMock(
            return_value={
                "edges": [
                    {"source": "Entity1", "target": "Entity2"},
                    {"source": "Entity2", "target": "Entity3"},
                ],
                "nodes": [
                    {"name": "Entity1"},
                    {"name": "Entity2"},
                ]
            }
        )

        duplicates = await incremental_updater.find_duplicate_entities(
            similarity_threshold=0.85,
        )

        # Should return list of tuples
        assert isinstance(duplicates, list)
        # May or may not find duplicates depending on similarity

    @pytest.mark.asyncio
    async def test_merge_entities(self, incremental_updater):
        """Test merging entities."""
        result = await incremental_updater.merge_entities(
            primary_entity="Entity1",
            entities_to_merge=["Entity2"],
        )

        assert result.primary_entity == "Entity1"
        assert result.merged_entities == ["Entity2"]
        assert result.similarity_score >= 0.0

    @pytest.mark.asyncio
    async def test_merge_multiple_entities(self, incremental_updater):
        """Test merging multiple entities."""
        result = await incremental_updater.merge_entities(
            primary_entity="Entity1",
            entities_to_merge=["Entity2", "Entity3", "Entity4"],
        )

        assert len(result.merged_entities) == 3
        assert "Entity2" in result.merged_entities
        assert "Entity3" in result.merged_entities
        assert "Entity4" in result.merged_entities


class TestCommunityRebuilding:
    """Tests for community rebuilding operations."""

    @pytest.mark.asyncio
    async def test_schedule_community_rebuild(self, incremental_updater):
        """Test scheduling a community rebuild."""
        await incremental_updater.schedule_community_rebuild(
            reason="Test rebuild"
        )

        assert incremental_updater._community_rebuild_needed is True

    @pytest.mark.asyncio
    async def test_rebuild_communities_if_needed(self, incremental_updater):
        """Test rebuilding communities when needed."""
        # Schedule rebuild
        await incremental_updater.schedule_community_rebuild(
            reason="Test rebuild"
        )

        # Rebuild
        update = await incremental_updater.rebuild_communities_if_needed()

        assert update is not None
        assert update.update_type == UpdateType.COMMUNITY_REBUILD
        assert update.status == UpdateStatus.COMPLETED
        assert incremental_updater._community_rebuild_needed is False

    @pytest.mark.asyncio
    async def test_rebuild_not_needed(self, incremental_updater):
        """Test that rebuild is skipped when not needed."""
        # Don't schedule rebuild
        update = await incremental_updater.rebuild_communities_if_needed()

        assert update is None

    @pytest.mark.asyncio
    async def test_rebuild_respects_min_time(self, incremental_updater):
        """Test that rebuild respects minimum time since last rebuild."""
        # Schedule rebuild
        await incremental_updater.schedule_community_rebuild(
            reason="Test rebuild"
        )

        # Set last rebuild time to now
        incremental_updater._last_rebuild_time = datetime.utcnow()

        # Try to rebuild with 1 hour minimum
        update = await incremental_updater.rebuild_communities_if_needed(
            min_time_since_last_rebuild=timedelta(hours=1),
        )

        # Should not rebuild
        assert update is None


class TestUpdateHistory:
    """Tests for update history and statistics."""

    @pytest.mark.asyncio
    async def test_get_update_history(self, incremental_updater):
        """Test getting update history."""
        # Add some updates
        await incremental_updater.add_entity("Entity1", "Type1")
        await incremental_updater.add_entity("Entity2", "Type2")

        # Get history
        history = await incremental_updater.get_update_history(limit=10)

        assert len(history) >= 2
        assert all(isinstance(u, GraphUpdate) for u in history)

    @pytest.mark.asyncio
    async def test_get_update_history_by_type(self, incremental_updater):
        """Test filtering update history by type."""
        # Add updates
        await incremental_updater.add_entity("Entity1", "Type1")
        await incremental_updater.invalidate_edge("E1", "REL", "E2")

        # Get history filtered by type
        history = await incremental_updater.get_update_history(
            limit=10,
            update_type=UpdateType.ENTITY_ADD,
        )

        assert all(u.update_type == UpdateType.ENTITY_ADD for u in history)

    @pytest.mark.asyncio
    async def test_get_pending_updates(self, incremental_updater):
        """Test getting pending updates."""
        # All updates should be processed immediately
        pending = await incremental_updater.get_pending_updates()

        assert isinstance(pending, list)

    @pytest.mark.asyncio
    async def test_get_statistics(self, incremental_updater):
        """Test getting update statistics."""
        # Add some updates
        await incremental_updater.add_entity("Entity1", "Type1")
        await incremental_updater.update_entity("Entity2", {"key": "value"})

        # Get statistics
        stats = await incremental_updater.get_statistics()

        assert "total_updates" in stats
        assert "by_status" in stats
        assert "by_type" in stats
        assert "pending_count" in stats
        assert stats["total_updates"] >= 2


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_merge_entities_without_bridge(self, mock_config):
        """Test that merging fails gracefully without bridge."""
        updater = GraphitiIncrementalUpdater(config=mock_config)
        # Don't set bridge

        with pytest.raises(IncrementalUpdateError):
            await updater.merge_entities(
                primary_entity="Entity1",
                entities_to_merge=["Entity2"],
            )

    @pytest.mark.asyncio
    async def test_calculate_entity_similarity(self, incremental_updater):
        """Test entity similarity calculation."""
        # Test exact match
        sim1 = incremental_updater._calculate_entity_similarity(
            "Entity1",
            "Entity1"
        )
        assert sim1 == 1.0

        # Test different entities
        sim2 = incremental_updater._calculate_entity_similarity(
            "Entity1",
            "Entity2"
        )
        assert 0.0 <= sim2 < 1.0


class TestGraphUpdateSerialization:
    """Tests for GraphUpdate serialization."""

    def test_graph_update_to_dict(self):
        """Test converting GraphUpdate to dictionary."""
        update = GraphUpdate(
            update_type=UpdateType.ENTITY_ADD,
            status=UpdateStatus.COMPLETED,
            affected_entities=["Entity1"],
            affected_edges=["edge1"],
        )

        data = update.to_dict()

        assert data["update_type"] == "entity_add"
        assert data["status"] == "completed"
        assert data["affected_entities"] == ["Entity1"]
        assert "created_at" in data

    def test_entity_merge_result_to_dict(self):
        """Test converting EntityMergeResult to dictionary."""
        result = EntityMergeResult(
            primary_entity="Entity1",
            merged_entities=["Entity2", "Entity3"],
            similarity_score=0.9,
        )

        data = result.to_dict()

        assert data["primary_entity"] == "Entity1"
        assert data["merged_entities"] == ["Entity2", "Entity3"]
        assert data["similarity_score"] == 0.9
        assert "merged_at" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
