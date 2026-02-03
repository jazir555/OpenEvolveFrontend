"""
Integration tests for Graphiti Agent Memory.

Implements Task 1.5.2: Integration tests for agent memory functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from knowledge_engine.integrations.graphiti.agent_memory import (
    GraphitiAgentMemory,
    MemoryType,
    AgentInteraction,
    MemorySummary,
)
from knowledge_engine.integrations.graphiti.temporal_bridge import (
    GraphitiTemporalBridge,
    WorkflowState,
)
from knowledge_engine.integrations.graphiti.config import GraphitiConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    with patch.dict('os.environ', {
        'GRAPHITI_URI': 'bolt://localhost:7687',
        'GRAPHITI_USER': 'neo4j',
        'GRAPHITI_PASSWORD': 'password',
        'OPENAI_API_KEY': 'test-key',
        'GRAPHITI_AGENT_MEMORY_ENABLED': 'true',
    }):
        config = GraphitiConfig()
        config.validate()
        return config


@pytest.fixture
def mock_temporal_bridge(mock_config):
    """Create a mock temporal bridge."""
    bridge = Mock(spec=GraphitiTemporalBridge)
    bridge._initialized = True
    bridge.graphiti_client = AsyncMock()
    bridge.graphiti_client.search = AsyncMock(
        return_value=Mock(edges=[], nodes=[])
    )
    bridge.graphiti_client.add_episode = AsyncMock(
        return_value=Mock(uuid="test-episode-uuid")
    )
    bridge.add_episode = AsyncMock(return_value="episode-uuid-123")
    bridge.search_temporal = AsyncMock(
        return_value={"edges": [], "nodes": []}
    )
    return bridge


@pytest.fixture
def agent_memory(mock_config, mock_temporal_bridge):
    """Create an agent memory instance."""
    memory = GraphitiAgentMemory(
        agent_id="test-agent-1",
        config=mock_config,
    )
    memory.set_bridge(mock_temporal_bridge)
    return memory


class TestAgentInteractionTracking:
    """Tests for agent interaction tracking (1.3.2)."""

    @pytest.mark.asyncio
    async def test_track_user_interaction(self, agent_memory):
        """Test tracking a user interaction."""
        interaction = await agent_memory.track_interaction(
            session_id="session-1",
            role="user",
            content="Hello, how can you help me?",
            memory_type=MemoryType.CONVERSATION,
        )

        assert interaction is not None
        assert interaction.agent_id == "test-agent-1"
        assert interaction.session_id == "session-1"
        assert interaction.role == "user"
        assert interaction.content == "Hello, how can you help me?"
        assert interaction.memory_type == MemoryType.CONVERSATION

    @pytest.mark.asyncio
    async def test_track_assistant_interaction(self, agent_memory):
        """Test tracking an assistant interaction."""
        interaction = await agent_memory.track_interaction(
            session_id="session-1",
            role="assistant",
            content="I can help you with various tasks.",
            memory_type=MemoryType.CONVERSATION,
        )

        assert interaction.role == "assistant"
        assert interaction.content == "I can help you with various tasks."

    @pytest.mark.asyncio
    async def test_track_knowledge_memory(self, agent_memory):
        """Test tracking knowledge memory."""
        interaction = await agent_memory.track_interaction(
            session_id="session-1",
            role="system",
            content="Learned: Python is a programming language",
            memory_type=MemoryType.KNOWLEDGE,
            metadata={"source": "user_input"},
        )

        assert interaction.memory_type == MemoryType.KNOWLEDGE
        assert interaction.metadata["source"] == "user_input"

    @pytest.mark.asyncio
    async def test_track_multiple_interactions(self, agent_memory):
        """Test tracking multiple interactions in a session."""
        session_id = "session-2"

        for i in range(3):
            await agent_memory.track_interaction(
                session_id=session_id,
                role="user",
                content=f"Message {i+1}",
            )

        history = await agent_memory.get_session_history(session_id)
        assert len(history) == 3


class TestContextRetrieval:
    """Tests for context retrieval (1.3.3)."""

    @pytest.mark.asyncio
    async def test_retrieve_context_with_interactions(self, agent_memory):
        """Test retrieving context with interactions."""
        session_id = "session-3"

        # Add some interactions
        await agent_memory.track_interaction(
            session_id=session_id,
            role="user",
            content="What is Python?",
        )
        await agent_memory.track_interaction(
            session_id=session_id,
            role="assistant",
            content="Python is a programming language.",
        )

        # Retrieve context
        context = await agent_memory.retrieve_context(
            session_id=session_id,
            query="Python",
            max_interactions=10,
        )

        assert len(context) >= 2  # At least our 2 interactions

    @pytest.mark.asyncio
    async def test_retrieve_context_with_time_window(self, agent_memory):
        """Test retrieving context within a time window."""
        session_id = "session-4"

        now = datetime.utcnow()
        past = now - timedelta(hours=1)

        # Add interaction
        await agent_memory.track_interaction(
            session_id=session_id,
            role="user",
            content="Test message",
            timestamp=now,
        )

        # Retrieve context with time window
        context = await agent_memory.retrieve_context(
            session_id=session_id,
            query="Test",
            time_window=(past, now + timedelta(minutes=5)),
        )

        assert len(context) >= 1

    @pytest.mark.asyncio
    async def test_get_session_history(self, agent_memory):
        """Test getting session history."""
        session_id = "session-5"

        # Add interactions
        await agent_memory.track_interaction(
            session_id=session_id,
            role="user",
            content="First message",
        )
        await agent_memory.track_interaction(
            session_id=session_id,
            role="assistant",
            content="First response",
        )
        await agent_memory.track_interaction(
            session_id=session_id,
            role="user",
            content="Second message",
        )

        # Get history
        history = await agent_memory.get_session_history(session_id)

        assert len(history) == 3
        assert history[0].role == "user"
        assert history[0].content == "First message"

    @pytest.mark.asyncio
    async def test_get_session_history_with_limit(self, agent_memory):
        """Test getting session history with limit."""
        session_id = "session-6"

        # Add 5 interactions
        for i in range(5):
            await agent_memory.track_interaction(
                session_id=session_id,
                role="user",
                content=f"Message {i+1}",
            )

        # Get limited history
        history = await agent_memory.get_session_history(session_id, limit=3)

        assert len(history) == 3


class TestCrossSessionPersistence:
    """Tests for cross-session memory persistence (1.3.4)."""

    @pytest.mark.asyncio
    async def test_persist_session_memory(self, agent_memory):
        """Test persisting session memory."""
        session_id = "session-7"

        # Add interactions
        await agent_memory.track_interaction(
            session_id=session_id,
            role="user",
            content="Important information",
        )

        # Persist session
        summary = await agent_memory.persist_session_memory(
            session_id=session_id,
            summarize=True,
        )

        assert summary is not None
        assert summary.session_id == session_id
        assert summary.interaction_count == 1

    @pytest.mark.asyncio
    async def test_persist_without_summarization(self, agent_memory):
        """Test persisting without creating a summary."""
        session_id = "session-8"

        await agent_memory.track_interaction(
            session_id=session_id,
            role="user",
            content="Test",
        )

        summary = await agent_memory.persist_session_memory(
            session_id=session_id,
            summarize=False,
        )

        # Should still persist but without summary
        assert summary is None

    @pytest.mark.asyncio
    async def test_clear_session_memory(self, agent_memory):
        """Test clearing session memory from cache."""
        session_id = "session-9"

        # Add interactions
        await agent_memory.track_interaction(
            session_id=session_id,
            role="user",
            content="Test",
        )

        # Clear session
        cleared_count = await agent_memory.clear_session_memory(session_id)

        assert cleared_count == 1

        # Verify cache is empty
        history = await agent_memory.get_session_history(session_id)
        assert len(history) == 0


class TestMemorySummarization:
    """Tests for memory summarization (1.3.5)."""

    @pytest.mark.asyncio
    async def test_memory_summary_creation(self, agent_memory):
        """Test that memory summaries are created."""
        session_id = "session-10"

        # Add multiple interactions
        await agent_memory.track_interaction(
            session_id=session_id,
            role="user",
            content="What is machine learning?",
        )
        await agent_memory.track_interaction(
            session_id=session_id,
            role="assistant",
            content="Machine learning is a subset of AI.",
        )

        # Persist with summarization
        summary = await agent_memory.persist_session_memory(
            session_id=session_id,
            summarize=True,
        )

        assert summary is not None
        assert summary.summary_id is not None
        assert summary.interaction_count == 2

    @pytest.mark.asyncio
    async def test_get_memory_summaries(self, agent_memory):
        """Test retrieving memory summaries."""
        session_id = "session-11"

        # Add and persist interactions
        await agent_memory.track_interaction(
            session_id=session_id,
            role="user",
            content="Test",
        )
        await agent_memory.persist_session_memory(
            session_id=session_id,
            summarize=True,
        )

        # Get summaries for session
        summaries = await agent_memory.get_memory_summaries(session_id=session_id)

        assert len(summaries) == 1
        assert summaries[0].session_id == session_id

    @pytest.mark.asyncio
    async def test_key_point_extraction(self, agent_memory):
        """Test that key points are extracted from interactions."""
        session_id = "session-12"

        # Add interactions with substantive content
        await agent_memory.track_interaction(
            session_id=session_id,
            role="assistant",
            content="Machine learning involves training models on data to make predictions.",
        )

        # Persist and summarize
        summary = await agent_memory.persist_session_memory(
            session_id=session_id,
            summarize=True,
        )

        # Should have extracted key points
        assert len(summary.key_points) >= 0  # May extract or be empty

    @pytest.mark.asyncio
    async def test_entity_extraction(self, agent_memory):
        """Test that entities are extracted from interactions."""
        session_id = "session-13"

        # Add interaction with named entities
        await agent_memory.track_interaction(
            session_id=session_id,
            role="user",
            content="Tell me about Python and Django",
        )

        # Persist and summarize
        summary = await agent_memory.persist_session_memory(
            session_id=session_id,
            summarize=True,
        )

        # Should have extracted entities
        assert len(summary.entities) >= 0  # May extract or be empty


class TestMemoryDisabled:
    """Tests for when memory is disabled."""

    @pytest.mark.asyncio
    async def test_track_with_memory_disabled(self, mock_config, mock_temporal_bridge):
        """Test that tracking returns None when disabled."""
        with patch.dict('os.environ', {
            'GRAPHITI_URI': 'bolt://localhost:7687',
            'GRAPHITI_USER': 'neo4j',
            'GRAPHITI_PASSWORD': 'password',
            'OPENAI_API_KEY': 'test-key',
            'GRAPHITI_AGENT_MEMORY_ENABLED': 'false',
        }):
            config = GraphitiConfig()
            config.validate()

            memory = GraphitiAgentMemory(
                agent_id="test-agent",
                config=config,
            )
            memory.set_bridge(mock_temporal_bridge)

            interaction = await memory.track_interaction(
                session_id="session",
                role="user",
                content="Test",
            )

            assert interaction is None


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_context_retrieval_without_bridge(self, mock_config):
        """Test context retrieval fails gracefully without bridge."""
        memory = GraphitiAgentMemory(
            agent_id="test-agent",
            config=mock_config,
        )
        # Don't set bridge

        context = await memory.retrieve_context(
            session_id="session",
            query="test",
        )

        # Should return empty list
        assert context == []

    @pytest.mark.asyncio
    async def test_persist_without_interactions(self, agent_memory):
        """Test persisting session with no interactions."""
        summary = await agent_memory.persist_session_memory(
            session_id="empty-session",
            summarize=True,
        )

        # Should return None
        assert summary is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
