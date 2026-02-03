"""
Agent Memory MCP Server - Production Grade

Task 2.3: Agent Memory MCP Server
- 2.3.1: Integrate KG-Gen MCP server
- 2.3.2: Add add_memories tool to unified MCP
- 2.3.3: Add retrieve_relevant_memories tool
- 2.3.4: Add visualize_memories tool
- 2.3.5: Implement memory aggregation across sessions
- 2.3.6: Add memory persistence and backup

Following CLAUDE.md Principles:
- IDEMPOTENCY: Memory operations safe to retry
- CONFIGURATION EXPLICITNESS: All config via env vars
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid
import pickle

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memories."""
    FACT = "fact"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    CONVERSATION = "conversation"
    PROCEDURAL = "procedural"


@dataclass
class Memory:
    """
    A single memory entry.

    All timestamps in UTC (LAW OF UTC).
    """
    memory_id: str
    content: str
    memory_type: MemoryType
    session_id: str

    # Metadata
    importance: float = 0.5  # 0.0 to 1.0
    confidence: float = 1.0  # 0.0 to 1.0
    embedding: Optional[List[float]] = None

    # Temporal
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0

    # Source tracking
    source: str = "manual"
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Linked memories
    related_memory_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "session_id": self.session_id,
            "importance": self.importance,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "related_memory_ids": self.related_memory_ids
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            session_id=data["session_id"],
            importance=data.get("importance", 0.5),
            confidence=data.get("confidence", 1.0),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            last_accessed=data.get("last_accessed", datetime.now(timezone.utc).isoformat()),
            access_count=data.get("access_count", 0),
            source=data.get("source", "manual"),
            correlation_id=data.get("correlation_id", str(uuid.uuid4())),
            related_memory_ids=data.get("related_memory_ids", [])
        )


@dataclass
class MemoryQuery:
    """
    Query for retrieving memories.
    """
    query_text: str
    session_id: Optional[str] = None
    memory_types: Optional[List[MemoryType]] = None
    min_importance: float = 0.0
    min_confidence: float = 0.0
    max_results: int = 10
    time_window_hours: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_text": self.query_text,
            "session_id": self.session_id,
            "memory_types": [mt.value for mt in self.memory_types] if self.memory_types else None,
            "min_importance": self.min_importance,
            "min_confidence": self.min_confidence,
            "max_results": self.max_results,
            "time_window_hours": self.time_window_hours
        }


@dataclass
class MemoryStoreConfig:
    """
    Memory store configuration.

    LAW OF CONFIGURATION EXPLICITNESS.
    """
    # Storage
    persistence_enabled: bool = field(
        default_factory=lambda: os.getenv("KGGEN_MEMORY_PERSISTENCE", "true").lower() == "true"
    )
    storage_path: str = field(
        default_factory=lambda: os.getenv("KGGEN_MEMORY_STORAGE_PATH", "./data/kggen_memories")
    )

    # Retrieval
    embedding_model: str = field(
        default_factory=lambda: os.getenv("KGGEN_MEMORY_EMBEDDING_MODEL", "text-embedding-ada-002")
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("KGGEN_SIMILARITY_THRESHOLD", "0.75"))
    )
    max_memories_per_session: int = field(
        default_factory=lambda: int(os.getenv("KGGEN_MAX_MEMORIES", "10000"))
    )

    # Backup
    backup_enabled: bool = field(
        default_factory=lambda: os.getenv("KGGEN_BACKUP_ENABLED", "true").lower() == "true"
    )
    backup_interval_hours: int = field(
        default_factory=lambda: int(os.getenv("KGGEN_BACKUP_INTERVAL_HOURS", "24"))
    )
    backup_retention_days: int = field(
        default_factory=lambda: int(os.getenv("KGGEN_BACKUP_RETENTION_DAYS", "30"))
    )

    # Aggregation
    aggregation_enabled: bool = field(
        default_factory=lambda: os.getenv("KGGEN_AGGREGATION_ENABLED", "true").lower() == "true"
    )

    def validate(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(f"Invalid similarity_threshold: {self.similarity_threshold}")
        if self.max_memories_per_session <= 0:
            raise ValueError(f"Invalid max_memories_per_session: {self.max_memories_per_session}")
        logger.info("MemoryStoreConfig validated", extra={"config": asdict(self)})


class MemoryManager:
    """
    Manages memory storage, retrieval, and persistence.

    Task 2.3.5: Implement memory aggregation across sessions.
    Task 2.3.6: Add memory persistence and backup.
    """

    def __init__(self, config: Optional[MemoryStoreConfig] = None):
        """
        Initialize memory manager.

        Args:
            config: Memory store configuration
        """
        self.config = config or MemoryStoreConfig()
        self.config.validate()

        # In-memory storage
        self._memories: Dict[str, Memory] = {}
        self._session_memories: Dict[str, List[str]] = {}
        self._embeddings: Dict[str, List[float]] = {}

        # Initialize storage
        if self.config.persistence_enabled:
            self._init_storage()

        logger.info(
            "MemoryManager initialized",
            extra={"config": asdict(self.config)}
        )

    def _init_storage(self) -> None:
        """Initialize storage directory."""
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing memories
        self._load_memories()

    def _load_memories(self) -> None:
        """Load memories from disk."""
        try:
            memories_file = self.storage_path / "memories.pkl"

            if memories_file.exists():
                with open(memories_file, 'rb') as f:
                    data = pickle.load(f)

                    self._memories = data.get("_memories", {})
                    self._session_memories = data.get("_session_memories", {})
                    self._embeddings = data.get("_embeddings", {})

                logger.info(f"Loaded {len(self._memories)} memories from disk")
        except Exception as e:
            logger.error(f"Error loading memories: {e}")

    def _save_memories(self) -> None:
        """Save memories to disk."""
        if not self.config.persistence_enabled:
            return

        try:
            memories_file = self.storage_path / "memories.pkl"

            with open(memories_file, 'wb') as f:
                pickle.dump({
                    "_memories": self._memories,
                    "_session_memories": self._session_memories,
                    "_embeddings": self._embeddings
                }, f)

            logger.debug(f"Saved {len(self._memories)} memories to disk")

        except Exception as e:
            logger.error(f"Error saving memories: {e}")

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        session_id: str,
        importance: float = 0.5,
        confidence: float = 1.0,
        source: str = "manual",
        correlation_id: Optional[str] = None
    ) -> Memory:
        """
        Add a memory.

        LAW OF IDEMPOTENCY: If memory with same content exists, update it.

        Args:
            content: Memory content
            memory_type: Type of memory
            session_id: Session identifier
            importance: Importance score (0.0 to 1.0)
            confidence: Confidence score (0.0 to 1.0)
            source: Source identifier
            correlation_id: Optional correlation ID

        Returns:
            Memory object
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        # Check for existing memory (idempotency)
        existing = self._find_memory_by_content(content, session_id)
        if existing:
            # Update existing
            existing.importance = max(existing.importance, importance)
            existing.last_accessed = datetime.now(timezone.utc).isoformat()
            existing.access_count += 1

            logger.info(
                f"Updated existing memory: {existing.memory_id}",
                extra={"correlation_id": correlation_id}
            )

            self._save_memories()
            return existing

        # Create new memory
        memory_id = f"mem-{uuid.uuid4().hex[:16]}"
        memory = Memory(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            importance=importance,
            confidence=confidence,
            source=source,
            correlation_id=correlation_id
        )

        # Store
        self._memories[memory_id] = memory

        # Index by session
        if session_id not in self._session_memories:
            self._session_memories[session_id] = []

        self._session_memories[session_id].append(memory_id)

        # Generate embedding
        embedding = await self._get_embedding(content)
        self._embeddings[memory_id] = embedding

        logger.info(
            f"Added memory: {memory_id}",
            extra={
                "correlation_id": correlation_id,
                "memory_type": memory_type.value,
                "session_id": session_id
            }
        )

        # Persist
        self._save_memories()

        return memory

    def _find_memory_by_content(self, content: str, session_id: str) -> Optional[Memory]:
        """
        Find existing memory by content.

        Args:
            content: Content to search for
            session_id: Session ID

        Returns:
            Memory if found
        """
        for mem_id in self._session_memories.get(session_id, []):
            memory = self._memories.get(mem_id)
            if memory and memory.content == content:
                return memory
        return None

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        # Simple fallback embedding
        # Production: use actual LLM embeddings
        import numpy as np

        normalized = text.lower().strip()
        embedding = [float(ord(c)) / 255.0 for c in normalized[:128]]

        # Pad to 128 dimensions
        embedding = embedding[:128]
        embedding.extend([0.0] * (128 - len(embedding)))

        return embedding

    async def retrieve_relevant_memories(
        self,
        query: MemoryQuery,
        correlation_id: Optional[str] = None
    ) -> List[Memory]:
        """
        Retrieve relevant memories.

        Task 2.3.3: Add retrieve_relevant_memories tool.

        Args:
            query: Memory query
            correlation_id: Optional correlation ID

        Returns:
            List of relevant memories
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        logger.info(
            f"Retrieving memories for query: {query.query_text}",
            extra={"correlation_id": correlation_id}
        )

        # Get query embedding
        query_embedding = await self._get_embedding(query.query_text)

        # Score memories
        scored_memories: List[tuple[Memory, float]] = []

        for memory in self._memories.values():
            # Apply filters
            if query.session_id and memory.session_id != query.session_id:
                continue

            if query.memory_types and memory.memory_type not in query.memory_types:
                continue

            if memory.importance < query.min_importance:
                continue

            if memory.confidence < query.min_confidence:
                continue

            # Time window filter
            if query.time_window_hours:
                from datetime import timedelta

                created = datetime.fromisoformat(memory.created_at)
                if datetime.now(timezone.utc) - created > timedelta(hours=query.time_window_hours):
                    continue

            # Calculate similarity
            memory_embedding = self._embeddings.get(memory.memory_id)
            if memory_embedding:
                similarity = self._cosine_similarity(query_embedding, memory_embedding)
            else:
                similarity = 0.0

            # Combine with importance
            score = (similarity * 0.7) + (memory.importance * 0.3)

            scored_memories.append((memory, score))

        # Sort by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Get top results
        results = [mem for mem, score in scored_memories[:query.max_results]]

        # Update access stats
        for memory in results:
            memory.last_accessed = datetime.now(timezone.utc).isoformat()
            memory.access_count += 1

        logger.info(
            f"Retrieved {len(results)} memories",
            extra={"correlation_id": correlation_id}
        )

        self._save_memories()
        return results

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity."""
        import numpy as np

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    async def aggregate_session_memories(
        self,
        session_id: str,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aggregate memories across a session.

        Task 2.3.5: Implement memory aggregation across sessions.

        Args:
            session_id: Session identifier
            correlation_id: Optional correlation ID

        Returns:
            Aggregation statistics
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        memory_ids = self._session_memories.get(session_id, [])
        memories = [self._memories[mid] for mid in memory_ids if mid in self._memories]

        if not memories:
            return {
                "session_id": session_id,
                "total_memories": 0,
                "by_type": {},
                "avg_importance": 0.0,
                "avg_confidence": 0.0
            }

        # Aggregate by type
        by_type: Dict[str, int] = {}
        total_importance = 0.0
        total_confidence = 0.0

        for memory in memories:
            by_type[memory.memory_type.value] = by_type.get(memory.memory_type.value, 0) + 1
            total_importance += memory.importance
            total_confidence += memory.confidence

        result = {
            "session_id": session_id,
            "total_memories": len(memories),
            "by_type": by_type,
            "avg_importance": total_importance / len(memories),
            "avg_confidence": total_confidence / len(memories)
        }

        logger.info(
            f"Aggregated session memories: {session_id}",
            extra={"correlation_id": correlation_id, "result": result}
        )

        return result

    async def backup_memories(self, correlation_id: Optional[str] = None) -> bool:
        """
        Backup memories to disk.

        Task 2.3.6: Add memory persistence and backup.

        Args:
            correlation_id: Optional correlation ID

        Returns:
            True if backup successful
        """
        if not self.config.backup_enabled:
            logger.info("Backup not enabled")
            return True

        correlation_id = correlation_id or str(uuid.uuid4())

        try:
            # Create backup filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_file = self.storage_path / f"backup_{timestamp}.pkl"

            # Save backup
            with open(backup_file, 'wb') as f:
                pickle.dump({
                    "_memories": self._memories,
                    "_session_memories": self._session_memories,
                    "_embeddings": self._embeddings
                }, f)

            logger.info(
                f"Created backup: {backup_file}",
                extra={"correlation_id": correlation_id}
            )

            # Clean old backups
            self._cleanup_old_backups()

            return True

        except Exception as e:
            logger.error(
                f"Backup failed: {e}",
                extra={"correlation_id": correlation_id}
            )
            return False

    def _cleanup_old_backups(self) -> None:
        """Clean up old backups beyond retention period."""
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.backup_retention_days)

        for backup_file in self.storage_path.glob("backup_*.pkl"):
            try:
                # Extract timestamp from filename
                timestamp_str = backup_file.stem.split("_")[1]
                file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                if file_time < cutoff:
                    backup_file.unlink()
                    logger.info(f"Deleted old backup: {backup_file}")

            except Exception as e:
                logger.error(f"Error cleaning backup {backup_file}: {e}")

    async def get_all_memories(self, session_id: Optional[str] = None) -> List[Memory]:
        """
        Get all memories, optionally filtered by session.

        Args:
            session_id: Optional session filter

        Returns:
            List of memories
        """
        if session_id:
            memory_ids = self._session_memories.get(session_id, [])
            return [self._memories[mid] for mid in memory_ids if mid in self._memories]

        return list(self._memories.values())


class MemoryTools:
    """
    MCP tools for memory operations.

    Task 2.3.2: Add add_memories tool to unified MCP.
    """

    def __init__(self, manager: MemoryManager):
        """
        Initialize memory tools.

        Args:
            manager: Memory manager instance
        """
        self.manager = manager

    async def add_memories(
        self,
        memories: List[Dict[str, Any]],
        session_id: str,
        correlation_id: Optional[str] = None
    ) -> List[Memory]:
        """
        Add multiple memories.

        Args:
            memories: List of memory dictionaries
            session_id: Session ID
            correlation_id: Optional correlation ID

        Returns:
            List of created/updated memories
        """
        results = []

        for mem_dict in memories:
            memory = await self.manager.add_memory(
                content=mem_dict["content"],
                memory_type=MemoryType(mem_dict.get("memory_type", "fact")),
                session_id=session_id,
                importance=mem_dict.get("importance", 0.5),
                confidence=mem_dict.get("confidence", 1.0),
                source=mem_dict.get("source", "mcp_tool"),
                correlation_id=correlation_id
            )
            results.append(memory)

        return results


class KGGenMCPServer:
    """
    KG-Gen MCP Server for agent memory.

    Task 2.3.1: Integrate KG-Gen MCP server.

    Provides MCP tools for:
    - Adding memories
    - Retrieving relevant memories
    - Visualizing memories
    """

    def __init__(self, config: Optional[MemoryStoreConfig] = None):
        """
        Initialize MCP server.

        Args:
            config: Memory store configuration
        """
        self.config = config or MemoryStoreConfig()
        self.memory_manager = MemoryManager(self.config)
        self.memory_tools = MemoryTools(self.memory_manager)

        logger.info("KGGenMCPServer initialized")

    async def add_memories(
        self,
        memories: List[Dict[str, Any]],
        session_id: str,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        MCP tool: Add memories.

        Args:
            memories: List of memory data
            session_id: Session ID
            correlation_id: Optional correlation ID

        Returns:
            Result dictionary
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        added = await self.memory_tools.add_memories(
            memories=memories,
            session_id=session_id,
            correlation_id=correlation_id
        )

        return {
            "success": True,
            "count": len(added),
            "memories": [m.to_dict() for m in added],
            "correlation_id": correlation_id
        }

    async def retrieve_relevant_memories(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        max_results: int = 10,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        MCP tool: Retrieve relevant memories.

        Task 2.3.3: Add retrieve_relevant_memories tool.

        Args:
            query_text: Query text
            session_id: Optional session ID
            max_results: Maximum results
            correlation_id: Optional correlation ID

        Returns:
            Retrieved memories
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        query = MemoryQuery(
            query_text=query_text,
            session_id=session_id,
            max_results=max_results
        )

        memories = await self.memory_manager.retrieve_relevant_memories(
            query=query,
            correlation_id=correlation_id
        )

        return {
            "success": True,
            "count": len(memories),
            "memories": [m.to_dict() for m in memories],
            "correlation_id": correlation_id
        }

    async def visualize_memories(
        self,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        MCP tool: Visualize memories.

        Task 2.3.4: Add visualize_memories tool.

        Args:
            session_id: Optional session ID
            correlation_id: Optional correlation ID

        Returns:
            Visualization data
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        memories = await self.memory_manager.get_all_memories(session_id=session_id)

        # Group by type
        by_type: Dict[str, List[Dict]] = {}
        for memory in memories:
            mem_type = memory.memory_type.value
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append(memory.to_dict())

        # Statistics
        stats = {
            "total_memories": len(memories),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "avg_importance": sum(m.importance for m in memories) / max(len(memories), 1),
            "avg_confidence": sum(m.confidence for m in memories) / max(len(memories), 1)
        }

        return {
            "success": True,
            "statistics": stats,
            "memories_by_type": by_type,
            "correlation_id": correlation_id
        }

    async def close(self) -> None:
        """Cleanup and save."""
        await self.memory_manager.backup_memories()
        logger.info("KGGenMCPServer closed")
