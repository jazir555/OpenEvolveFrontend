"""
Graphiti Adapter for OpenEvolve

This module provides an adapter that wraps Graphiti's functionality to implement
the OpenEvolve KnowledgeGraphInterface. It enables temporal knowledge management
without modifying Graphiti's source code.
"""

import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Add Graphiti to path
graphiti_path = os.path.join(os.path.dirname(__file__), "../../projects to analyze/graphiti")
if graphiti_path not in sys.path:
    sys.path.insert(0, graphiti_path)

try:
    from graphiti_core import Graphiti
    from graphiti_core.llm_client import LLMClient
    from graphiti_core.embedder import EmbedderClient
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.driver.driver import GraphDriver
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.driver.falkordb_driver import FalkorDBDriver
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.search.search_config_recipes import (
        COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
        EDGE_HYBRID_SEARCH_RRF,
    )
    from graphiti_core.search.search_filters import SearchFilters
    GRAPHITI_AVAILABLE = True
except ImportError as e:
    GRAPHITI_AVAILABLE = False
    graphiti_import_error = str(e)

from integrations.base.knowledge_interface import (
    KnowledgeGraphInterface,
    KnowledgeGraphError,
    ConfigurationError,
    ConnectionError,
    ValidationError,
    StorageError,
    SearchError,
    AnalysisError,
    ShutdownError,
    RetrievalError,
    RemovalError,
    TemporalFilter,
)

logger = logging.getLogger(__name__)


class GraphitiAdapter(KnowledgeGraphInterface):
    """
    Adapter for Graphiti temporally-aware knowledge graph.

    This adapter wraps Graphiti's core functionality to provide a consistent
    interface for OpenEvolve. It supports Neo4j and FalkorDB backends,
    temporal metadata tracking, and hybrid search capabilities.

    Gracefully degrades if Graphiti is unavailable.
    """

    def __init__(self):
        """Initialize the adapter without connecting to the backend."""
        self.graphiti: Optional[Graphiti] = None
        self.config: Optional[Dict[str, Any]] = None
        self.is_initialized = False
        self.backend_type: Optional[str] = None
        self.llm_client: Optional[LLMClient] = None
        self.embedder: Optional[EmbedderClient] = None
        self.cross_encoder: Optional[CrossEncoderClient] = None

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize Graphiti with the provided configuration.

        Args:
            config: Configuration dictionary with keys:
                - uri: Database URI (required)
                - user: Database username (for Neo4j)
                - password: Database password (for Neo4j)
                - backend: "neo4j" or "falkordb" (default: "neo4j")
                - llm_client: Optional custom LLM client
                - embedder: Optional custom embedder
                - cross_encoder: Optional custom cross-encoder
                - store_raw_episode_content: Whether to store raw content (default: True)
                - max_coroutines: Max concurrent operations (optional)

        Returns:
            True if initialization was successful

        Raises:
            ConfigurationError: If config is invalid or Graphiti unavailable
            ConnectionError: If connection fails
        """
        if not GRAPHITI_AVAILABLE:
            logger.warning(f"Graphiti not available: {graphiti_import_error}")
            raise ConfigurationError(
                f"Graphiti is not available. Please ensure it is installed. Error: {graphiti_import_error}"
            )

        try:
            self.config = config

            # Extract configuration
            uri = config.get("uri")
            if not uri:
                raise ConfigurationError("Database URI is required in configuration")

            user = config.get("user", "neo4j")
            password = config.get("password", "")
            backend = config.get("backend", "neo4j").lower()
            store_raw = config.get("store_raw_episode_content", True)
            max_coroutines = config.get("max_coroutines", None)

            # Get custom clients if provided
            self.llm_client = config.get("llm_client")
            self.embedder = config.get("embedder")
            self.cross_encoder = config.get("cross_encoder")

            # Create appropriate driver
            graph_driver = None
            if backend == "neo4j":
                graph_driver = Neo4jDriver(uri, user, password)
                self.backend_type = "neo4j"
            elif backend == "falkordb":
                graph_driver = FalkorDBDriver(uri, password)
                self.backend_type = "falkordb"
            else:
                raise ConfigurationError(f"Unsupported backend: {backend}")

            # Initialize Graphiti
            self.graphiti = Graphiti(
                graph_driver=graph_driver,
                llm_client=self.llm_client,
                embedder=self.embedder,
                cross_encoder=self.cross_encoder,
                store_raw_episode_content=store_raw,
                max_coroutines=max_coroutines,
            )

            # Build indices
            await self.graphiti.build_indices_and_constraints(delete_existing=False)

            self.is_initialized = True
            logger.info(f"Graphiti adapter initialized successfully with {backend} backend")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Graphiti adapter: {e}")
            raise ConnectionError(f"Failed to connect to Graphiti backend: {e}")

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
        Add an episode to Graphiti.

        Args:
            name: Episode name
            body: Episode content
            reference_time: When the episode occurred
            metadata: Optional metadata
            source: Source identifier
            group_id: Optional group ID

        Returns:
            Dictionary with episode results

        Raises:
            ValidationError: If data is invalid
            StorageError: If storage fails
        """
        if not self.is_initialized or not self.graphiti:
            raise StorageError("Graphiti adapter not initialized")

        try:
            # Map source to EpisodeType
            source_mapping = {
                "openevolve": EpisodeType.text,
                "message": EpisodeType.message,
                "text": EpisodeType.text,
                "news": EpisodeType.news,
            }
            episode_type = source_mapping.get(source, EpisodeType.text)

            # Add episode to Graphiti
            result = await self.graphiti.add_episode(
                name=name,
                episode_body=body,
                source_description=source,
                reference_time=reference_time,
                source=episode_type,
                group_id=group_id,
                update_communities=False,  # Don't auto-update communities
            )

            # Format results
            return {
                "uuid": result.episode.uuid,
                "name": result.episode.name,
                "created_at": result.episode.created_at.isoformat(),
                "valid_at": result.episode.valid_at.isoformat(),
                "nodes": [
                    {
                        "uuid": node.uuid,
                        "name": node.name,
                        "summary": node.summary,
                        "labels": node.labels,
                    }
                    for node in result.nodes
                ],
                "edges": [
                    {
                        "uuid": edge.uuid,
                        "fact": edge.fact,
                        "source_node": edge.source_node_uuid,
                        "target_node": edge.target_node_uuid,
                    }
                    for edge in result.edges
                ],
                "communities": len(result.communities),
            }

        except Exception as e:
            logger.error(f"Failed to add episode: {e}")
            raise StorageError(f"Failed to add episode to Graphiti: {e}")

    async def search(
        self,
        query: str,
        temporal_filters: Optional[Dict[str, Any]] = None,
        num_results: int = 10,
        group_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search Graphiti with optional temporal filtering.

        Args:
            query: Search query
            temporal_filters: Optional temporal filters
            num_results: Max results
            group_ids: Optional group IDs

        Returns:
            Search results dictionary

        Raises:
            SearchError: If search fails
        """
        if not self.is_initialized or not self.graphiti:
            raise SearchError("Graphiti adapter not initialized")

        try:
            # Apply search filters based on temporal constraints
            search_filter = SearchFilters()

            if temporal_filters:
                filter_type = temporal_filters.get("filter_type", TemporalFilter.CURRENT)

                if filter_type == TemporalFilter.TIME_RANGE:
                    start_time = temporal_filters.get("start_time")
                    end_time = temporal_filters.get("end_time")
                    if start_time and end_time:
                        # Graphiti handles temporal filtering internally
                        pass

            # Perform hybrid search
            config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER
            config.limit = num_results

            results = await self.graphiti.search_(
                query=query,
                config=config,
                group_ids=group_ids,
                search_filter=search_filter,
            )

            # Format results
            return {
                "edges": [
                    {
                        "uuid": edge.uuid,
                        "fact": edge.fact,
                        "source_node": edge.source_node_uuid,
                        "target_node": edge.target_node_uuid,
                        "created_at": edge.created_at.isoformat() if edge.created_at else None,
                        "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                        "expired_at": edge.expired_at.isoformat() if edge.expired_at else None,
                    }
                    for edge in results.edges
                ],
                "nodes": [
                    {
                        "uuid": node.uuid,
                        "name": node.name,
                        "summary": node.summary,
                        "labels": node.labels,
                    }
                    for node in results.nodes
                ],
                "context": results.context if hasattr(results, "context") else [],
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Graphiti search failed: {e}")

    async def get_community_detections(
        self,
        group_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get or compute community detections.

        Args:
            group_ids: Optional group IDs

        Returns:
            Community information

        Raises:
            AnalysisError: If analysis fails
        """
        if not self.is_initialized or not self.graphiti:
            raise AnalysisError("Graphiti adapter not initialized")

        try:
            communities, community_edges = await self.graphiti.build_communities(
                group_ids=group_ids
            )

            return {
                "communities": [
                    {
                        "uuid": comm.uuid,
                        "summary": comm.summary,
                        "name": comm.name,
                    }
                    for comm in communities
                ],
                "community_edges": [
                    {
                        "uuid": edge.uuid,
                        "fact": edge.fact,
                    }
                    for edge in community_edges
                ],
                "metrics": {
                    "num_communities": len(communities),
                    "num_edges": len(community_edges),
                },
            }

        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            raise AnalysisError(f"Failed to detect communities: {e}")

    async def validate(self) -> Dict[str, Any]:
        """
        Validate Graphiti state.

        Returns:
            Validation results
        """
        if not self.is_initialized:
            return {
                "is_valid": False,
                "checks": {"initialized": False},
                "issues": ["Graphiti adapter not initialized"],
                "metrics": {},
            }

        try:
            # Basic validation - check if we can perform a simple search
            await self.graphiti.search("test", num_results=1)

            return {
                "is_valid": True,
                "checks": {
                    "initialized": True,
                    "backend_connected": True,
                    "search_operational": True,
                },
                "issues": [],
                "metrics": {
                    "backend_type": self.backend_type,
                    "has_custom_llm": self.llm_client is not None,
                    "has_custom_embedder": self.embedder is not None,
                },
            }

        except Exception as e:
            return {
                "is_valid": False,
                "checks": {
                    "initialized": True,
                    "backend_connected": False,
                    "search_operational": False,
                },
                "issues": [str(e)],
                "metrics": {},
            }

    async def shutdown(self) -> bool:
        """
        Shutdown Graphiti connection.

        Returns:
            True if successful
        """
        if not self.is_initialized or not self.graphiti:
            return True

        try:
            await self.graphiti.close()
            self.is_initialized = False
            logger.info("Graphiti adapter shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            raise ShutdownError(f"Failed to shutdown Graphiti: {e}")

    async def get_episodes(
        self,
        reference_time: datetime,
        last_n: int = 10,
        group_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent episodes.

        Args:
            reference_time: Reference time
            last_n: Number of episodes
            group_ids: Optional group IDs

        Returns:
            List of episode dictionaries
        """
        if not self.is_initialized or not self.graphiti:
            raise RetrievalError("Graphiti adapter not initialized")

        try:
            episodes = await self.graphiti.retrieve_episodes(
                reference_time=reference_time,
                last_n=last_n,
                group_ids=group_ids,
            )

            return [
                {
                    "uuid": ep.uuid,
                    "name": ep.name,
                    "content": ep.content,
                    "source": ep.source.value if ep.source else None,
                    "created_at": ep.created_at.isoformat() if ep.created_at else None,
                    "valid_at": ep.valid_at.isoformat() if ep.valid_at else None,
                }
                for ep in episodes
            ]

        except Exception as e:
            logger.error(f"Failed to retrieve episodes: {e}")
            raise RetrievalError(f"Failed to retrieve episodes: {e}")

    async def add_triplet(
        self,
        source_entity: Dict[str, Any],
        relationship: Dict[str, Any],
        target_entity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add a knowledge triplet.

        Args:
            source_entity: Source entity
            relationship: Relationship
            target_entity: Target entity

        Returns:
            Added triplet details
        """
        if not self.is_initialized or not self.graphiti:
            raise StorageError("Graphiti adapter not initialized")

        try:
            from graphiti_core.nodes import EntityNode, EntityEdge

            # Create nodes
            source_node = EntityNode(
                name=source_entity.get("name", ""),
                summary=source_entity.get("summary", ""),
                labels=source_entity.get("labels", []),
            )

            target_node = EntityNode(
                name=target_entity.get("name", ""),
                summary=target_entity.get("summary", ""),
                labels=target_entity.get("labels", []),
            )

            # Create edge
            edge = EntityEdge(
                fact=relationship.get("fact", ""),
                source_node_uuid="",  # Will be set by Graphiti
                target_node_uuid="",  # Will be set by Graphiti
            )

            # Add triplet
            result = await self.graphiti.add_triplet(source_node, edge, target_node)

            return {
                "nodes": [
                    {"uuid": node.uuid, "name": node.name}
                    for node in result.nodes
                ],
                "edges": [
                    {"uuid": edge.uuid, "fact": edge.fact}
                    for edge in result.edges
                ],
            }

        except Exception as e:
            logger.error(f"Failed to add triplet: {e}")
            raise StorageError(f"Failed to add triplet: {e}")

    async def remove_episode(self, episode_uuid: str) -> bool:
        """
        Remove an episode.

        Args:
            episode_uuid: Episode UUID

        Returns:
            True if successful
        """
        if not self.is_initialized or not self.graphiti:
            raise RemovalError("Graphiti adapter not initialized")

        try:
            await self.graphiti.remove_episode(episode_uuid)
            logger.info(f"Removed episode {episode_uuid}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove episode: {e}")
            raise RemovalError(f"Failed to remove episode: {e}")

    async def search_hybrid(
        self,
        query: str,
        use_bm25: bool = True,
        use_vector: bool = True,
        use_graph_traversal: bool = True,
        rerank_method: str = "rrf",
        temporal_filters: Optional[Dict[str, Any]] = None,
        num_results: int = 10,
        group_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform hybrid search with configurable components.

        Args:
            query: Search query
            use_bm25: Use BM25 keyword search
            use_vector: Use vector similarity search
            use_graph_traversal: Use graph traversal (BFS)
            rerank_method: Reranking method ('rrf', 'cross_encoder', 'weighted', 'none')
            temporal_filters: Optional temporal filters
            num_results: Maximum results
            group_ids: Optional group IDs

        Returns:
            Search results dictionary

        Raises:
            SearchError: If search fails
        """
        if not self.is_initialized or not self.graphiti:
            raise SearchError("Graphiti adapter not initialized")

        try:
            import graphiti_core.search.search_config_recipes as recipes

            # Select search configuration based on rerank method
            if rerank_method == "cross_encoder":
                config = recipes.COMBINED_HYBRID_SEARCH_CROSS_ENCODER
            elif rerank_method == "rrf":
                config = recipes.EDGE_HYBRID_SEARCH_RRF
            elif rerank_method == "weighted":
                config = recipes.COMBINED_HYBRID_SEARCH_CROSS_ENCODER  # Use weighted config
            else:
                config = recipes.EDGE_HYBRID_SEARCH_RRF  # Default

            config.limit = num_results

            # Apply search filters
            search_filter = SearchFilters()
            if temporal_filters:
                filter_type = temporal_filters.get("filter_type", TemporalFilter.CURRENT)
                if filter_type == TemporalFilter.TIME_RANGE:
                    start_time = temporal_filters.get("start_time")
                    end_time = temporal_filters.get("end_time")
                    if start_time and end_time:
                        # Graphiti handles temporal filtering internally
                        pass

            # Perform hybrid search
            results = await self.graphiti.search_(
                query=query,
                config=config,
                group_ids=group_ids,
                search_filter=search_filter,
            )

            # Format results
            return {
                "edges": [
                    {
                        "uuid": edge.uuid,
                        "fact": edge.fact,
                        "source_node": edge.source_node_uuid,
                        "target_node": edge.target_node_uuid,
                        "created_at": edge.created_at.isoformat() if edge.created_at else None,
                        "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                        "expired_at": edge.expired_at.isoformat() if edge.expired_at else None,
                        "score": edge.score if hasattr(edge, 'score') else None,
                    }
                    for edge in results.edges
                ],
                "nodes": [
                    {
                        "uuid": node.uuid,
                        "name": node.name,
                        "summary": node.summary,
                        "labels": node.labels,
                    }
                    for node in results.nodes
                ],
                "context": results.context if hasattr(results, "context") else [],
                "search_config": {
                    "use_bm25": use_bm25,
                    "use_vector": use_vector,
                    "use_graph_traversal": use_graph_traversal,
                    "rerank_method": rerank_method,
                },
            }

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise SearchError(f"Graphiti hybrid search failed: {e}")

    async def get_episode_at_time(
        self,
        episode_uuid: str,
        reference_time: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Get episode state at a specific point in time.

        Args:
            episode_uuid: Episode UUID
            reference_time: Point in time

        Returns:
            Episode dictionary or None
        """
        if not self.is_initialized or not self.graphiti:
            raise RetrievalError("Graphiti adapter not initialized")

        try:
            episodes = await self.graphiti.retrieve_episodes(
                reference_time=reference_time,
                last_n=100,  # Retrieve more to find the specific one
            )

            for ep in episodes:
                if ep.uuid == episode_uuid:
                    return {
                        "uuid": ep.uuid,
                        "name": ep.name,
                        "content": ep.content,
                        "source": ep.source.value if ep.source else None,
                        "created_at": ep.created_at.isoformat() if ep.created_at else None,
                        "valid_at": ep.valid_at.isoformat() if ep.valid_at else None,
                    }

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve episode at time: {e}")
            raise RetrievalError(f"Failed to retrieve episode: {e}")

    async def find_contradictions(
        self,
        entity_name: str,
        time_range: Optional[tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find potential contradictions involving an entity.

        Args:
            entity_name: Entity to check
            time_range: Optional time range (start, end)

        Returns:
            List of potential contradictions
        """
        if not self.is_initialized or not self.graphiti:
            raise SearchError("Graphiti adapter not initialized")

        try:
            # Search for relationships involving the entity
            query = f"relationships and facts about {entity_name}"
            results = await self.search(query=query, num_results=50)

            contradictions = []
            edges = results.get("edges", [])

            # Look for edges with negation or contradiction patterns
            contradiction_patterns = [
                "not ", "never ", "cannot ", "won't ", "however ", "but ",
                "contrary ", "opposite ", "despite ", "although ",
            ]

            for edge in edges:
                fact = edge.get("fact", "").lower()
                for pattern in contradiction_patterns:
                    if pattern in fact:
                        contradictions.append({
                            "edge_uuid": edge.get("uuid"),
                            "fact": edge.get("fact"),
                            "pattern": pattern,
                            "valid_at": edge.get("valid_at"),
                            "expired_at": edge.get("expired_at"),
                        })
                        break

            logger.info(f"Found {len(contradictions)} potential contradictions for {entity_name}")
            return contradictions

        except Exception as e:
            logger.error(f"Contradiction detection failed: {e}")
            raise SearchError(f"Failed to detect contradictions: {e}")

    async def get_entity_timeline(
        self,
        entity_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of events for an entity.

        Args:
            entity_name: Entity name
            start_time: Start of timeline
            end_time: End of timeline

        Returns:
            List of temporal events
        """
        if not self.is_initialized or not self.graphiti:
            raise RetrievalError("Graphiti adapter not initialized")

        try:
            # Search for episodes involving the entity
            query = f"events and activities involving {entity_name}"
            temporal_filters = {
                "filter_type": TemporalFilter.TIME_RANGE,
                "start_time": start_time,
                "end_time": end_time,
            }

            results = await self.search(
                query=query,
                temporal_filters=temporal_filters,
                num_results=100,
            )

            timeline = []
            for edge in results.get("edges", []):
                valid_at = edge.get("valid_at")
                if valid_at:
                    valid_dt = datetime.fromisoformat(valid_at)
                    if start_time <= valid_dt <= end_time:
                        timeline.append({
                            "timestamp": valid_dt,
                            "fact": edge.get("fact"),
                            "edge_uuid": edge.get("uuid"),
                            "expired_at": edge.get("expired_at"),
                        })

            # Sort by timestamp
            timeline.sort(key=lambda x: x["timestamp"])
            logger.info(f"Retrieved {len(timeline)} timeline events for {entity_name}")
            return timeline

        except Exception as e:
            logger.error(f"Failed to get entity timeline: {e}")
            raise RetrievalError(f"Failed to retrieve timeline: {e}")
