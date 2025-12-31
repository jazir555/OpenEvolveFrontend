"""
Unit tests for IntermediaryStorageManager
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock


# Mock RAGBits DocumentSearch
class MockDocumentSearch:
    """Mock DocumentSearch for testing"""

    def __init__(self):
        self.storage = {}

    async def ingest_text(self, text: str, metadata: dict):
        """Mock ingest method"""
        artifact_id = metadata.get("artifact_id", "mock_id")
        self.storage[artifact_id] = {
            "text": text,
            "metadata": metadata
        }

    async def search(self, query: str, filters: dict = None, top_k: int = 5):
        """Mock search method"""
        results = []
        for artifact_id, data in self.storage.items():
            # Filter matching
            if filters:
                match = True
                for key, value in filters.items():
                    if data["metadata"].get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # Create mock result
            mock_result = Mock()
            mock_result.text_representation = data["text"]
            mock_result.metadata = data["metadata"]
            mock_result.metadata["similarity"] = 0.85
            results.append(mock_result)

            if len(results) >= top_k:
                break

        return results


@pytest.fixture
def mock_document_search():
    """Create mock document search"""
    return MockDocumentSearch()


@pytest.fixture
def storage_manager(mock_document_search):
    """Create storage manager with mock document search"""
    from ragbits_integration.intermediary_storage.storage_manager import IntermediaryStorageManager
    return IntermediaryStorageManager(mock_document_search)


@pytest.mark.asyncio
async def test_store_artifact(storage_manager):
    """Test storing an artifact"""
    artifact_id = await storage_manager.store_artifact(
        artifact_type="solution_draft",
        content="Test solution content",
        metadata={
            "stage": "stage_3",
            "team": "blue",
            "sub_problem_id": "sub_1"
        }
    )

    assert artifact_id is not None
    assert artifact_id.startswith("solution_draft_")

    # Verify artifact is in cache
    assert artifact_id in storage_manager._artifact_cache


@pytest.mark.asyncio
async def test_retrieve_artifact(storage_manager):
    """Test retrieving an artifact"""
    # Store an artifact first
    artifact_id = await storage_manager.store_artifact(
        artifact_type="solution_draft",
        content="Test solution content",
        metadata={"team": "blue"}
    )

    # Retrieve it
    retrieved = await storage_manager.retrieve_artifact(artifact_id)

    assert retrieved is not None
    assert retrieved["content"] == "Test solution content"
    assert retrieved["metadata"]["team"] == "blue"


@pytest.mark.asyncio
async def test_retrieve_context_for_stage(storage_manager):
    """Test gathering context for a stage"""
    # Store some artifacts
    await storage_manager.store_artifact(
        artifact_type="content_analysis",
        content="Problem analysis: High complexity",
        metadata={"stage": "stage_0"}
    )

    await storage_manager.store_artifact(
        artifact_type="decomposition_plan",
        content="5 sub-problems identified",
        metadata={"stage": "stage_1", "is_current": True}
    )

    # Gather context for Stage 3
    context = await storage_manager.retrieve_context_for_stage(
        stage="stage_3_blue_team_solution",
        sub_problem_id="sub_1"
    )

    assert context["stage"] == "stage_3_blue_team_solution"
    assert "artifacts" in context
    assert "similar_historical" in context


@pytest.mark.asyncio
async def test_update_artifact_status(storage_manager):
    """Test updating artifact status"""
    # Store artifact in draft status
    artifact_id = await storage_manager.store_artifact(
        artifact_type="solution_draft",
        content="Test solution",
        metadata={"status": "draft"}
    )

    # Update to pending
    success = await storage_manager.update_artifact_status(artifact_id, "pending")

    assert success is True


@pytest.mark.asyncio
async def test_get_artifact_chain(storage_manager):
    """Test getting artifact chain"""
    # Store linked artifacts
    solution_id = await storage_manager.store_artifact(
        artifact_type="solution_draft",
        content="Solution content",
        metadata={"team": "blue"}
    )

    critique_id = await storage_manager.store_artifact(
        artifact_type="critique",
        content="Critique content",
        metadata={"team": "red"},
        links_to=[solution_id]
    )

    # Get chain starting from critique
    chain = await storage_manager.get_artifact_chain(critique_id)

    assert len(chain) >= 1
    assert any(item["artifact_id"] == critique_id for item in chain)


@pytest.mark.asyncio
async def test_cache_stats(storage_manager):
    """Test cache statistics"""
    # Store some artifacts
    await storage_manager.store_artifact(
        artifact_type="test",
        content="Test content",
        metadata={"stage": "test"}
    )

    stats = storage_manager.get_cache_stats()

    assert stats["cached_artifacts"] > 0
    assert len(stats["artifact_ids"]) > 0


@pytest.mark.asyncio
async def test_clear_cache(storage_manager):
    """Test clearing cache"""
    # Store artifact
    await storage_manager.store_artifact(
        artifact_type="test",
        content="Test content",
        metadata={"stage": "test"}
    )

    # Verify cache has items
    assert len(storage_manager._artifact_cache) > 0

    # Clear cache
    storage_manager.clear_cache()

    # Verify cache is empty
    assert len(storage_manager._artifact_cache) == 0


@pytest.mark.asyncio
async def test_get_artifacts_by_stage(storage_manager):
    """Test getting artifacts by stage"""
    # Store artifacts for different stages
    await storage_manager.store_artifact(
        artifact_type="solution",
        content="Solution 1",
        metadata={"stage": "stage_3", "status": "draft"}
    )

    await storage_manager.store_artifact(
        artifact_type="solution",
        content="Solution 2",
        metadata={"stage": "stage_3", "status": "verified"}
    )

    # Get all stage_3 artifacts
    artifacts = await storage_manager.get_artifacts_by_stage("stage_3")

    assert len(artifacts) >= 2


@pytest.mark.asyncio
async def test_get_artifacts_by_subproblem(storage_manager):
    """Test getting artifacts by sub-problem"""
    sub_problem_id = "test_sub_1"

    # Store artifacts for sub-problem
    await storage_manager.store_artifact(
        artifact_type="solution_draft",
        content="Solution",
        metadata={"sub_problem_id": sub_problem_id}
    )

    await storage_manager.store_artifact(
        artifact_type="critique",
        content="Critique",
        metadata={"sub_problem_id": sub_problem_id}
    )

    # Get all artifacts for sub-problem
    artifacts = await storage_manager.get_artifacts_by_sub_problem(sub_problem_id)

    assert len(artifacts) >= 2
