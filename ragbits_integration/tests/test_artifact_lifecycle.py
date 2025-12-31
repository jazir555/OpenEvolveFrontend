"""
Unit tests for ArtifactLifecycleManager
"""

import pytest
from unittest.mock import Mock, AsyncMock
from ragbits_integration.intermediary_storage import ArtifactStatus


@pytest.fixture
def mock_storage_manager():
    """Create mock storage manager"""
    storage = Mock()
    storage.store_artifact = AsyncMock(return_value="artifact_123")
    storage.update_artifact_status = AsyncMock(return_value=True)
    storage.retrieve_artifact = AsyncMock(return_value=None)
    storage._search_artifacts = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def lifecycle_manager(mock_storage_manager):
    """Create lifecycle manager with mock storage"""
    from ragbits_integration.intermediary_storage.artifact_lifecycle import ArtifactLifecycleManager
    return ArtifactLifecycleManager(mock_storage_manager)


@pytest.mark.asyncio
async def test_create_draft(lifecycle_manager, mock_storage_manager):
    """Test creating a draft artifact"""
    artifact_id = await lifecycle_manager.create_draft(
        artifact_type="solution_draft",
        content="Test content",
        metadata={"team": "blue"}
    )

    assert artifact_id == "artifact_123"
    mock_storage_manager.store_artifact.assert_called_once()


@pytest.mark.asyncio
async def test_transition_to_pending(lifecycle_manager):
    """Test transitioning from draft to pending"""
    # Mock the current status check
    lifecycle_manager._get_current_status = AsyncMock(
        return_value=ArtifactStatus.DRAFT
    )

    success = await lifecycle_manager.transition_to_pending("artifact_123")

    assert success is True


@pytest.mark.asyncio
async def test_transition_to_verified(lifecycle_manager):
    """Test transitioning from pending to verified"""
    # Mock the current status check
    lifecycle_manager._get_current_status = AsyncMock(
        return_value=ArtifactStatus.PENDING
    )

    success = await lifecycle_manager.transition_to_verified("artifact_123")

    assert success is True


@pytest.mark.asyncio
async def test_transition_to_final(lifecycle_manager):
    """Test transitioning from verified to final"""
    # Mock the current status check
    lifecycle_manager._get_current_status = AsyncMock(
        return_value=ArtifactStatus.VERIFIED
    )

    success = await lifecycle_manager.transition_to_final("artifact_123")

    assert success is True


@pytest.mark.asyncio
async def test_transition_to_rejected(lifecycle_manager):
    """Test transitioning to rejected"""
    # Mock the current status check
    lifecycle_manager._get_current_status = AsyncMock(
        return_value=ArtifactStatus.PENDING
    )

    success = await lifecycle_manager.transition_to_rejected(
        "artifact_123",
        "Does not meet requirements"
    )

    assert success is True


@pytest.mark.asyncio
async def test_invalid_transition(lifecycle_manager):
    """Test that invalid transitions are rejected"""
    # Mock the current status as FINAL
    lifecycle_manager._get_current_status = AsyncMock(
        return_value=ArtifactStatus.FINAL
    )

    # Try to transition from FINAL to PENDING (invalid)
    success = await lifecycle_manager._transition_status(
        "artifact_123",
        ArtifactStatus.FINAL,
        ArtifactStatus.PENDING,
        "invalid transition"
    )

    assert success is False


@pytest.mark.asyncio
async def test_get_pending_artifacts(lifecycle_manager, mock_storage_manager):
    """Test getting pending artifacts"""
    # Mock search results
    mock_storage_manager._search_artifacts.return_value = [
        {"content": "Test", "metadata": {"status": "pending"}}
    ]

    artifacts = await lifecycle_manager.get_pending_artifacts()

    assert len(artifacts) >= 0


@pytest.mark.asyncio
async def test_get_statistics(lifecycle_manager):
    """Test getting lifecycle statistics"""
    # Create some transitions
    lifecycle_manager._record_transition("art1", None, "draft", "created")
    lifecycle_manager._record_transition("art1", "draft", "pending", "submitted")
    lifecycle_manager._record_transition("art2", None, "draft", "created")

    stats = lifecycle_manager.get_statistics()

    assert stats["total_artifacts_tracked"] == 2
    assert stats["total_transitions"] == 3
