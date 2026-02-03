"""
Graphiti Agent Memory System

Implements Sprint 1 Task 1.3: Agent memory with interaction tracking,
context retrieval, cross-session persistence, and summarization.

Following CLAUDE.md principles:
- IDEMPOTENCY: Memory operations safe to run multiple times
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

from .config import GraphitiConfig
from .exceptions import GraphitiIntegrationError


logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of agent memories."""
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    PREFERENCE = "preference"
    PROCEDURE = "procedure"
    EXPERIENCE = "experience"


@dataclass
class AgentInteraction:
    """
    Represents a single agent interaction.
    """
    agent_id: str
    session_id: str
    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    role: str = "user"
    content: str = ""
    memory_type: MemoryType = MemoryType.CONVERSATION
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["memory_type"] = self.memory_type.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentInteraction":
        """Create from dictionary."""
        if "memory_type" in data and isinstance(data["memory_type"], str):
            data["memory_type"] = MemoryType(data["memory_type"])
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class MemorySummary:
    """
    Summarized memory for long-term storage.
    """
    agent_id: str
    session_id: str
    memory_type: MemoryType
    summary_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime = field(default_factory=datetime.utcnow)
    interaction_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["memory_type"] = self.memory_type.value
        for field_name in ["start_time", "end_time", "created_at"]:
            if data[field_name]:
                data[field_name] = data[field_name].isoformat()
        return data


class GraphitiAgentMemory:
    """
    Agent memory system using Graphiti for persistence.

    Implements:
    - 1.3.1: GraphitiAgentMemory class
    - 1.3.2: Agent interaction tracking
    - 1.3.3: Context retrieval for agent conversations
    - 1.3.4: Cross-session memory persistence
    - 1.3.5: Memory summarization for long-term storage
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[GraphitiConfig] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize the agent memory system.

        Args:
            agent_id: Unique identifier for this agent
            config: Graphiti configuration
            correlation_id: Request correlation ID
        """
        self.agent_id = agent_id
        self.config = config or GraphitiConfig()
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.temporal_bridge = None  # Will be set via set_bridge

        # In-memory cache for active sessions
        self._session_interactions: Dict[str, List[AgentInteraction]] = {}
        self._session_summaries: Dict[str, List[MemorySummary]] = {}
        self._lock = asyncio.Lock()

        logger.info(
            json.dumps({
                "msg": "GraphitiAgentMemory created",
                "correlation_id": self.correlation_id,
                "agent_id": agent_id,
                "enabled": self.config.agent_memory_enabled,
            })
        )

    def set_bridge(self, bridge: "GraphitiTemporalBridge") -> None:
        """
        Set the temporal bridge instance.

        Args:
            bridge: GraphitiTemporalBridge instance
        """
        self.temporal_bridge = bridge

    # ===== 1.3.1 & 1.3.2: Agent Interaction Tracking =====

    async def track_interaction(
        self,
        session_id: str,
        role: str,
        content: str,
        memory_type: MemoryType = MemoryType.CONVERSATION,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> AgentInteraction:
        """
        Track an agent interaction.

        Args:
            session_id: Session identifier
            role: Interaction role (user, assistant, system)
            content: Interaction content
            memory_type: Type of memory
            metadata: Additional metadata
            timestamp: Interaction timestamp (defaults to now UTC)

        Returns:
            Created AgentInteraction

        Raises:
            GraphitiIntegrationError: If bridge not initialized
        """
        if not self.config.agent_memory_enabled:
            logger.debug("Agent memory tracking disabled")
            return None

        # Validate timestamp
        timestamp = timestamp or datetime.utcnow()
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(tz=None).replace(tzinfo=None)

        interaction = AgentInteraction(
            agent_id=self.agent_id,
            session_id=session_id,
            role=role,
            content=content,
            timestamp=timestamp,
            memory_type=memory_type,
            metadata=metadata or {},
            correlation_id=self.correlation_id,
        )

        # Store in cache (idempotent)
        async with self._lock:
            if session_id not in self._session_interactions:
                self._session_interactions[session_id] = []
            self._session_interactions[session_id].append(interaction)

        # Persist to Graphiti if bridge available
        if self.temporal_bridge and self.temporal_bridge._initialized:
            await self._persist_interaction(interaction)

        logger.info(
            json.dumps({
                "msg": "Agent interaction tracked",
                "correlation_id": self.correlation_id,
                "agent_id": self.agent_id,
                "session_id": session_id,
                "role": role,
                "memory_type": memory_type.value,
            })
        )

        return interaction

    async def _persist_interaction(self, interaction: AgentInteraction) -> None:
        """
        Persist interaction to Graphiti.

        Args:
            interaction: Interaction to persist
        """
        try:
            # Construct episode body
            episode_body = f"{interaction.role}: {interaction.content}"

            # Add as episode
            await self.temporal_bridge.add_episode(
                name=f"Agent {self.agent_id} - {interaction.session_id}",
                episode_body=episode_body,
                reference_time=interaction.timestamp,
                source=f"agent_memory:{interaction.agent_id}",
                metadata={
                    "interaction_id": interaction.interaction_id,
                    "session_id": interaction.session_id,
                    "role": interaction.role,
                    "memory_type": interaction.memory_type.value,
                },
            )

        except Exception as e:
            logger.error(
                json.dumps({
                    "msg": "Failed to persist interaction",
                    "correlation_id": self.correlation_id,
                    "interaction_id": interaction.interaction_id,
                    "error": str(e),
                })
            )

    # ===== 1.3.3: Context Retrieval for Agent Conversations =====

    async def retrieve_context(
        self,
        session_id: str,
        query: str,
        max_interactions: int = 10,
        time_window: Optional[tuple[datetime, datetime]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for agent conversation.

        Args:
            session_id: Session identifier
            query: Context query
            max_interactions: Maximum interactions to retrieve
            time_window: Optional time window for retrieval

        Returns:
            List of relevant interactions with context
        """
        if not self.temporal_bridge or not self.temporal_bridge._initialized:
            logger.warning("Temporal bridge not initialized for context retrieval")
            return []

        # Retrieve from cache first
        async with self._lock:
            session_interactions = self._session_interactions.get(session_id, [])

        # Filter by time window if specified
        if time_window:
            start_time, end_time = time_window
            session_interactions = [
                i for i in session_interactions
                if start_time <= i.timestamp <= end_time
            ]

        # Get most recent interactions
        session_interactions = session_interactions[-max_interactions:]

        # Search Graphiti for related knowledge
        try:
            search_results = await self.temporal_bridge.search_temporal(
                query=query,
                max_results=max_interactions,
            )

            # Combine session interactions with search results
            context = [
                {
                    "type": "interaction",
                    "interaction_id": i.interaction_id,
                    "role": i.role,
                    "content": i.content,
                    "timestamp": i.timestamp.isoformat(),
                    "memory_type": i.memory_type.value,
                }
                for i in session_interactions
            ]

            # Add search results
            for edge in search_results.get("edges", []):
                context.append({
                    "type": "knowledge",
                    "fact": edge.get("fact", ""),
                    "source": edge.get("source", ""),
                    "target": edge.get("target", ""),
                    "relation": edge.get("relation", ""),
                    "score": edge.get("score", 0.0),
                })

            logger.info(
                json.dumps({
                    "msg": "Context retrieved",
                    "correlation_id": self.correlation_id,
                    "session_id": session_id,
                    "context_count": len(context),
                })
            )

            return context

        except Exception as e:
            logger.error(
                json.dumps({
                    "msg": "Failed to retrieve context",
                    "correlation_id": self.correlation_id,
                    "session_id": session_id,
                    "error": str(e),
                })
            )
            return []

    async def get_session_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[AgentInteraction]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Optional limit on number of interactions

        Returns:
            List of interactions in chronological order
        """
        async with self._lock:
            interactions = self._session_interactions.get(session_id, [])

        # Sort by timestamp
        interactions.sort(key=lambda x: x.timestamp)

        # Apply limit
        if limit:
            interactions = interactions[-limit:]

        return interactions

    # ===== 1.3.4: Cross-Session Memory Persistence =====

    async def persist_session_memory(
        self,
        session_id: str,
        summarize: bool = True,
    ) -> Optional[MemorySummary]:
        """
        Persist session memory to long-term storage.

        Args:
            session_id: Session to persist
            summarize: Whether to create a summary

        Returns:
            MemorySummary if summarization enabled
        """
        async with self._lock:
            interactions = self._session_interactions.get(session_id, [])

        if not interactions:
            logger.info(
                json.dumps({
                    "msg": "No interactions to persist",
                    "correlation_id": self.correlation_id,
                    "session_id": session_id,
                })
            )
            return None

        # Create summary if requested
        summary = None
        if summarize:
            summary = await self._create_memory_summary(session_id, interactions)

            # Store summary
            async with self._lock:
                if session_id not in self._session_summaries:
                    self._session_summaries[session_id] = []
                self._session_summaries[session_id].append(summary)

        # Persist to Graphiti
        if self.temporal_bridge and self.temporal_bridge._initialized:
            try:
                # Create episode from interactions
                episode_body = self._construct_session_episode(interactions, summary)

                start_time = interactions[0].timestamp
                end_time = interactions[-1].timestamp

                await self.temporal_bridge.add_episode(
                    name=f"Agent {self.agent_id} Session: {session_id}",
                    episode_body=episode_body,
                    reference_time=end_time,
                    source=f"agent_memory:{self.agent_id}",
                    metadata={
                        "session_id": session_id,
                        "interaction_count": len(interactions),
                        "summary_id": summary.summary_id if summary else None,
                    },
                )

                logger.info(
                    json.dumps({
                        "msg": "Session memory persisted",
                        "correlation_id": self.correlation_id,
                        "session_id": session_id,
                        "interaction_count": len(interactions),
                    })
                )

            except Exception as e:
                logger.error(
                    json.dumps({
                        "msg": "Failed to persist session memory",
                        "correlation_id": self.correlation_id,
                        "session_id": session_id,
                        "error": str(e),
                    })
                )

        return summary

    def _construct_session_episode(
        self,
        interactions: List[AgentInteraction],
        summary: Optional[MemorySummary] = None,
    ) -> str:
        """
        Construct episode body from session interactions.

        Args:
            interactions: List of interactions
            summary: Optional summary

        Returns:
            Episode body text
        """
        parts = [f"Agent: {self.agent_id}"]

        if summary:
            parts.append("\nSummary:")
            parts.append(summary.summary)
            parts.append("\nKey Points:")
            for point in summary.key_points:
                parts.append(f"  - {point}")

        parts.append("\nConversation:")
        for interaction in interactions:
            parts.append(f"{interaction.role}: {interaction.content}")

        return "\n".join(parts)

    # ===== 1.3.5: Memory Summarization =====

    async def _create_memory_summary(
        self,
        session_id: str,
        interactions: List[AgentInteraction],
    ) -> MemorySummary:
        """
        Create a memory summary from interactions.

        Args:
            session_id: Session identifier
            interactions: List of interactions to summarize

        Returns:
            MemorySummary
        """
        # Extract key information
        all_content = " ".join([i.content for i in interactions])
        start_time = interactions[0].timestamp
        end_time = interactions[-1].timestamp

        # Simple summarization (in production, use LLM)
        # Extract entities and key points
        entities = self._extract_entities(all_content)
        key_points = self._extract_key_points(interactions)

        summary_text = self._generate_summary_text(interactions)

        summary = MemorySummary(
            agent_id=self.agent_id,
            session_id=session_id,
            memory_type=MemoryType.CONVERSATION,
            interaction_count=len(interactions),
            summary=summary_text,
            key_points=key_points,
            entities=entities,
            start_time=start_time,
            end_time=end_time,
            correlation_id=self.correlation_id,
        )

        return summary

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text.

        Args:
            text: Text to extract from

        Returns:
            List of entity names
        """
        # Simple extraction (in production, use NER)
        # Look for capitalized words that might be entities
        import re
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        # Return unique entities
        return list(set(entities))[:20]  # Limit to top 20

    def _extract_key_points(self, interactions: List[AgentInteraction]) -> List[str]:
        """
        Extract key points from interactions.

        Args:
            interactions: List of interactions

        Returns:
            List of key points
        """
        # Simple extraction (in production, use LLM)
        key_points = []

        for interaction in interactions:
            if interaction.role == "assistant":
                # Extract sentences that might be key points
                sentences = interaction.content.split(". ")
                for sentence in sentences[:3]:  # Top 3 sentences
                    if len(sentence) > 20:  # Filter short sentences
                        key_points.append(sentence.strip())

        return key_points[:10]  # Limit to top 10

    def _generate_summary_text(self, interactions: List[AgentInteraction]) -> str:
        """
        Generate summary text from interactions.

        Args:
            interactions: List of interactions

        Returns:
            Summary text
        """
        if not interactions:
            return ""

        # Simple summarization (in production, use LLM)
        first_interaction = interactions[0]
        last_interaction = interactions[-1]
        duration = (last_interaction.timestamp - first_interaction.timestamp).total_seconds()

        summary_parts = [
            f"Session with {len(interactions)} interactions",
            f"Duration: {duration:.0f} seconds",
            f"Started: {first_interaction.timestamp.isoformat()}",
            f"Ended: {last_interaction.timestamp.isoformat()}",
        ]

        return ". ".join(summary_parts)

    async def get_memory_summaries(
        self,
        session_id: Optional[str] = None,
    ) -> List[MemorySummary]:
        """
        Get memory summaries.

        Args:
            session_id: Optional session filter

        Returns:
            List of memory summaries
        """
        async with self._lock:
            if session_id:
                return self._session_summaries.get(session_id, [])

            # Return all summaries flattened
            all_summaries = []
            for summaries in self._session_summaries.values():
                all_summaries.extend(summaries)
            return all_summaries

    async def clear_session_memory(self, session_id: str) -> int:
        """
        Clear session memory from cache.

        Args:
            session_id: Session to clear

        Returns:
            Number of interactions cleared
        """
        async with self._lock:
            interactions = self._session_interactions.pop(session_id, [])
            self._session_summaries.pop(session_id, None)

        logger.info(
            json.dumps({
                "msg": "Session memory cleared",
                "correlation_id": self.correlation_id,
                "session_id": session_id,
                "cleared_count": len(interactions),
            })
        )

        return len(interactions)
