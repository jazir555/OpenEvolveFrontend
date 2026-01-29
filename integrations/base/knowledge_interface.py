"""
Base Knowledge Graph Interface for OpenEvolve

This module defines the abstract interface that all knowledge graph implementations must follow.
It provides a consistent API for temporal knowledge management across different backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class TemporalFilter(Enum):
    """Temporal filtering options for knowledge retrieval."""
    CURRENT = "current"  # Only currently valid knowledge
    HISTORICAL = "historical"  # All historical knowledge
    FUTURE = "future"  # Future projected knowledge
    TIME_RANGE = "time_range"  # Knowledge within a specific time range


class KnowledgeGraphInterface(ABC):
    """
    Abstract base class for knowledge graph implementations.

    This interface defines the contract that all knowledge graph adapters must implement,
    ensuring consistency across different backend technologies (Graphiti, Neo4j, etc.).
    """

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the knowledge graph with the given configuration.

        Args:
            config: Configuration dictionary containing connection details,
                   API keys, and other initialization parameters.

        Returns:
            True if initialization was successful, False otherwise.

        Raises:
            ConfigurationError: If configuration is invalid
            ConnectionError: If connection to backend fails
        """
        pass

    @abstractmethod
    async def add_episode(
        self,
        name: str,
        body: str,
        reference_time: datetime,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "openevolve",
        group_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add an episode (temporal knowledge unit) to the graph.

        Args:
            name: Name/identifier for the episode
            body: Content/body of the episode
            reference_time: When the episode occurred (temporal context)
            metadata: Additional metadata about the episode
            source: Source of the episode (e.g., "workflow", "user", "external")
            group_id: Optional group/partition identifier

        Returns:
            Dictionary containing episode details including:
            - uuid: Unique identifier for the episode
            - nodes: List of entities extracted
            - edges: List of relationships discovered
            - created_at: Timestamp when episode was added

        Raises:
            ValidationError: If episode data is invalid
            StorageError: If storage operation fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        temporal_filters: Optional[Dict[str, Any]] = None,
        num_results: int = 10,
        group_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search the knowledge graph with optional temporal filtering.

        Args:
            query: Search query string
            temporal_filters: Optional temporal filtering parameters
                - filter_type: TemporalFilter enum value
                - start_time: Start of time range (for TIME_RANGE)
                - end_time: End of time range (for TIME_RANGE)
            num_results: Maximum number of results to return
            group_ids: Optional list of group IDs to search within

        Returns:
            Dictionary containing search results:
            - edges: List of matching edges/relationships
            - nodes: List of matching nodes/entities
            - context: Relevant episodic context
            - scores: Relevance scores for results

        Raises:
            SearchError: If search operation fails
        """
        pass

    @abstractmethod
    async def get_community_detections(
        self,
        group_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get or compute community detections in the knowledge graph.

        Args:
            group_ids: Optional list of group IDs to analyze

        Returns:
            Dictionary containing community information:
            - communities: List of detected communities
            - community_edges: Relationships between communities
            - metrics: Quality metrics for communities

        Raises:
            AnalysisError: If community detection fails
        """
        pass

    @abstractmethod
    async def validate(self) -> Dict[str, Any]:
        """
        Validate the knowledge graph state and connections.

        Returns:
            Dictionary containing validation results:
            - is_valid: Overall validation status
            - checks: Individual check results
            - issues: List of any issues found
            - metrics: Performance and health metrics

        Raises:
            ValidationError: If validation itself fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the knowledge graph connection.

        Performs cleanup and closes connections to the backend.

        Returns:
            True if shutdown was successful, False otherwise

        Raises:
            ShutdownError: If shutdown fails
        """
        pass

    @abstractmethod
    async def get_episodes(
        self,
        reference_time: datetime,
        last_n: int = 10,
        group_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent episodes for context.

        Args:
            reference_time: Time reference for retrieval
            last_n: Number of recent episodes to retrieve
            group_ids: Optional list of group IDs to filter by

        Returns:
            List of episode dictionaries

        Raises:
            RetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def add_triplet(
        self,
        source_entity: Dict[str, Any],
        relationship: Dict[str, Any],
        target_entity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add a knowledge triplet (entity-relationship-entity) directly.

        Args:
            source_entity: Source entity dictionary with 'name' and optional attributes
            relationship: Relationship dictionary with 'fact' and metadata
            target_entity: Target entity dictionary with 'name' and optional attributes

        Returns:
            Dictionary containing added triplet details

        Raises:
            ValidationError: If triplet data is invalid
            StorageError: If storage operation fails
        """
        pass

    @abstractmethod
    async def remove_episode(self, episode_uuid: str) -> bool:
        """
        Remove an episode from the graph.

        Args:
            episode_uuid: UUID of the episode to remove

        Returns:
            True if removal was successful

        Raises:
            RemovalError: If removal fails
        """
        pass


class KnowledgeGraphError(Exception):
    """Base exception for knowledge graph operations."""
    pass


class ConfigurationError(KnowledgeGraphError):
    """Raised when configuration is invalid."""
    pass


class ConnectionError(KnowledgeGraphError):
    """Raised when connection to backend fails."""
    pass


class ValidationError(KnowledgeGraphError):
    """Raised when data validation fails."""
    pass


class StorageError(KnowledgeGraphError):
    """Raised when storage operations fail."""
    pass


class SearchError(KnowledgeGraphError):
    """Raised when search operations fail."""
    pass


class AnalysisError(KnowledgeGraphError):
    """Raised when analysis operations fail."""
    pass


class ShutdownError(KnowledgeGraphError):
    """Raised when shutdown operations fail."""
    pass


class RetrievalError(KnowledgeGraphError):
    """Raised when retrieval operations fail."""
    pass


class RemovalError(KnowledgeGraphError):
    """Raised when removal operations fail."""
    pass
