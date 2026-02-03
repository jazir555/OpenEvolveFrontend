"""
Base backend interface for Unified Knowledge Graph Manager.

This module defines the abstract interface that all knowledge graph backends must implement.
Following the Law of the Air Gap - each backend is isolated and provides a canonical interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Supported backend types (all permissive licenses)"""
    POSTGRESQL = "postgresql"  # PostgreSQL License
    MEMGRAPH = "memgraph"      # Apache 2.0
    QDRANT = "qdrant"          # Apache 2.0
    REDIS = "redis"            # BSD
    KARATECLUB = "karateclub"  # MIT
    MEMORY = "memory"          # MIT


class OperationType(Enum):
    """Types of operations for backend selection"""
    ADD_KNOWLEDGE = "add_knowledge"
    SEARCH = "search"
    ANALYZE = "analyze"
    VISUALIZE = "visualize"
    STATS = "stats"


@dataclass
class KnowledgeEntry:
    """Canonical representation of a knowledge entry"""
    source: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    timestamp: Optional[str] = None
    id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class SearchResults:
    """Canonical search results format"""
    query: str
    results: List[Dict[str, Any]]
    total_count: int
    backend_used: str
    search_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisResult:
    """Canonical analysis result format"""
    analysis_type: str
    target: str
    results: Dict[str, Any]
    backend_used: str
    analysis_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GraphStatistics:
    """Canonical graph statistics format"""
    node_count: int
    edge_count: int
    backend: str
    metadata: Dict[str, Any]
    timestamp: str


class KnowledgeGraphBackend(ABC):
    """
    Abstract base class for all knowledge graph backends.

    All backends must implement these methods to provide a consistent interface.
    Following CLAUDE.md principles:
    - Runtime Truth: Verify connections on initialization
    - Configuration Explicitness: All config via environment variables or params
    - UTC: All timestamps in UTC
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backend with configuration.

        Args:
            config: Backend-specific configuration dictionary
        """
        self.config = config
        self.backend_type: BackendType = BackendType.MEMORY
        self.is_healthy: bool = False
        self._connection_pool: Optional[Any] = None
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the backend.

        Returns:
            bool: True if connection successful, False otherwise

        Raises:
            ConnectionError: If connection fails critically
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection and cleanup resources.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if backend is healthy and responsive.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass

    @abstractmethod
    async def add_knowledge(self, entry: KnowledgeEntry) -> str:
        """
        Add knowledge entry to the backend.

        Args:
            entry: KnowledgeEntry to add

        Returns:
            str: ID of the added entry

        Raises:
            ValueError: If entry is invalid
            ConnectionError: If backend is unavailable
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> SearchResults:
        """
        Search knowledge in the backend.

        Args:
            query: Search query string
            filters: Optional filters to apply
            limit: Maximum number of results
            offset: Result offset for pagination

        Returns:
            SearchResults: Canonical search results

        Raises:
            ConnectionError: If backend is unavailable
        """
        pass

    @abstractmethod
    async def analyze(
        self,
        analysis_type: str,
        target: Optional[str] = None
    ) -> AnalysisResult:
        """
        Perform analysis on the knowledge graph.

        Args:
            analysis_type: Type of analysis to perform
            target: Optional target entity/graph for analysis

        Returns:
            AnalysisResult: Canonical analysis results

        Raises:
            ValueError: If analysis_type is not supported
            ConnectionError: If backend is unavailable
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> GraphStatistics:
        """
        Get statistics about the knowledge graph.

        Returns:
            GraphStatistics: Canonical statistics

        Raises:
            ConnectionError: If backend is unavailable
        """
        pass

    @abstractmethod
    async def visualize(
        self,
        output_format: str = 'html',
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate visualization of the knowledge graph.

        Args:
            output_format: Format for visualization (html, json, dot, etc.)
            options: Optional visualization parameters

        Returns:
            str: Visualization data or file path

        Raises:
            ValueError: If output_format is not supported
            ConnectionError: If backend is unavailable
        """
        pass

    async def batch_add_knowledge(
        self,
        entries: List[KnowledgeEntry]
    ) -> List[str]:
        """
        Add multiple knowledge entries efficiently.

        Default implementation adds entries sequentially.
        Backends should override this for better performance.

        Args:
            entries: List of KnowledgeEntry objects

        Returns:
            List[str]: List of entry IDs
        """
        ids = []
        for entry in entries:
            entry_id = await self.add_knowledge(entry)
            ids.append(entry_id)
        return ids

    async def batch_search(
        self,
        queries: List[str],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[SearchResults]:
        """
        Perform multiple searches efficiently.

        Default implementation searches sequentially.
        Backends should override this for better performance.

        Args:
            queries: List of search queries
            filters: Optional filters to apply to all searches
            limit: Maximum results per query

        Returns:
            List[SearchResults]: List of search results
        """
        results = []
        for query in queries:
            result = await self.search(query, filters, limit)
            results.append(result)
        return results

    async def delete_knowledge(self, entry_id: str) -> bool:
        """
        Delete a knowledge entry by ID.

        Args:
            entry_id: ID of entry to delete

        Returns:
            bool: True if deleted, False if not found

        Raises:
            ConnectionError: If backend is unavailable
        """
        # Default implementation - backends should override
        raise NotImplementedError(f"{self.__class__.__name__} does not support deletion")

    async def update_knowledge(
        self,
        entry_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a knowledge entry by ID.

        Args:
            entry_id: ID of entry to update
            updates: Dictionary of fields to update

        Returns:
            bool: True if updated, False if not found

        Raises:
            ConnectionError: If backend is unavailable
        """
        # Default implementation - backends should override
        raise NotImplementedError(f"{self.__class__.__name__} does not support updates")

    async def clear_all(self) -> int:
        """
        Clear all knowledge from the backend.

        WARNING: This is a destructive operation.

        Returns:
            int: Number of entries cleared

        Raises:
            ConnectionError: If backend is unavailable
        """
        # Default implementation - backends should override
        raise NotImplementedError(f"{self.__class__.__name__} does not support clear operation")

    def get_backend_type(self) -> BackendType:
        """Get the backend type enum value"""
        return self.backend_type

    def get_backend_name(self) -> str:
        """Get the backend type as string"""
        return self.backend_type.value

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
