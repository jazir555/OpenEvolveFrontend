"""
Graphiti Temporal Bridge for KnowledgeEngine Integration

This module provides a high-level bridge that integrates Graphiti's temporal
capabilities with the KnowledgeEngine, enabling seamless conversion between
KnowledgeArtifacts and Graphiti episodes.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from ...integrations.graphiti.bridge import GraphitiBridge
from ...integrations.graphiti.adapter import GraphitiAdapter
from ...integrations.base.knowledge_interface import TemporalFilter
from ..core.temporal_knowledge_engine import (
    KnowledgeArtifact,
    RerankMethod,
    ContradictionDetection,
)

logger = logging.getLogger(__name__)


@dataclass
class EntityMapping:
    """Mapping between KnowledgeEngine entity types and Graphiti custom types."""
    ke_type: str
    graphiti_type: str
    description: str


class GraphitiTemporalBridge:
    """
    Bridge between Graphiti and KnowledgeEngine for temporal operations.

    This bridge handles:
    - KnowledgeArtifact to Episode conversion
    - Search result transformation
    - Entity type mapping
    - Temporal query translation
    """

    # Entity type mappings
    ENTITY_MAPPINGS = [
        EntityMapping("solution_pattern", "Procedure", "Reusable solution patterns"),
        EntityMapping("critique_insight", "Requirement", "Insights from critiques"),
        EntityMapping("team_performance", "Preference", "Team performance metrics"),
        EntityMapping("problem", "Event", "Problem descriptions"),
        EntityMapping("workflow", "Document", "Workflow definitions"),
        EntityMapping("agent", "Organization", "AI agents"),
        EntityMapping("tool", "Technology", "Tools and technologies"),
        EntityMapping("technique", "Methodology", "Techniques and methodologies"),
    ]

    def __init__(
        self,
        graphiti_bridge: Optional[GraphitiBridge] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the temporal bridge.

        Args:
            graphiti_bridge: Optional existing Graphiti bridge
            config_path: Optional path to Graphiti config
        """
        self.graphiti_bridge = graphiti_bridge
        self.config_path = config_path or "integrations/graphiti/config.yaml"

        if self.graphiti_bridge is None:
            self.graphiti_bridge = GraphitiBridge()

    async def initialize(self) -> bool:
        """
        Initialize the bridge.

        Returns:
            True if successful
        """
        try:
            await self.graphiti_bridge.load_config(self.config_path)
            await self.graphiti_bridge.initialize()
            logger.info("GraphitiTemporalBridge initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize GraphitiTemporalBridge: {e}")
            return False

    async def artifact_to_episode(
        self,
        artifact: KnowledgeArtifact,
    ) -> Dict[str, Any]:
        """
        Convert KnowledgeArtifact to Graphiti episode format.

        Args:
            artifact: KnowledgeArtifact to convert

        Returns:
            Episode dictionary
        """
        # Map artifact type to Graphiti custom type
        graphiti_type = self._map_entity_type(artifact.artifact_type)

        # Create episode metadata
        episode_metadata = {
            "artifact_id": artifact.id,
            "artifact_type": artifact.artifact_type,
            "graphiti_type": graphiti_type,
            "source": artifact.source,
            "confidence": artifact.confidence,
            "entity_count": len(artifact.entities),
            "relationship_count": len(artifact.relationships),
            **artifact.metadata,
        }

        # Construct episode
        episode = {
            "name": f"{artifact.artifact_type}: {artifact.id[:8]}",
            "body": artifact.content,
            "reference_time": artifact.valid_at,
            "metadata": episode_metadata,
            "source": artifact.source,
            "group_id": artifact.group_id,
        }

        return episode

    def _map_entity_type(self, artifact_type: str) -> str:
        """
        Map KnowledgeEngine artifact type to Graphiti custom type.

        Args:
            artifact_type: KE artifact type

        Returns:
            Graphiti custom type
        """
        for mapping in self.ENTITY_MAPPINGS:
            if mapping.ke_type == artifact_type:
                return mapping.graphiti_type
        return artifact_type  # Default to original type

    async def add_artifact(
        self,
        artifact: KnowledgeArtifact,
    ) -> Dict[str, Any]:
        """
        Add a KnowledgeArtifact to Graphiti.

        Args:
            artifact: KnowledgeArtifact to add

        Returns:
            Result dictionary
        """
        if not self.graphiti_bridge or not self.graphiti_bridge.is_initialized:
            logger.warning("Graphiti bridge not initialized")
            return {"success": False, "error": "Bridge not initialized"}

        try:
            # Convert artifact to episode
            episode = await self.artifact_to_episode(artifact)

            # Add episode to Graphiti
            result = await self.graphiti_bridge.add_episode(
                name=episode["name"],
                body=episode["body"],
                reference_time=episode["reference_time"],
                metadata=episode["metadata"],
                source=episode["source"],
                group_id=episode["group_id"],
            )

            logger.info(f"Added artifact {artifact.id} to Graphiti")
            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"Failed to add artifact to Graphiti: {e}")
            return {"success": False, "error": str(e)}

    async def graphiti_result_to_artifact(
        self,
        result: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> Optional[KnowledgeArtifact]:
        """
        Convert Graphiti search result to KnowledgeArtifact.

        Args:
            result: Graphiti result dictionary
            timestamp: Reference timestamp (defaults to now)

        Returns:
            KnowledgeArtifact or None
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        try:
            # Handle edge result
            if "fact" in result:
                return KnowledgeArtifact(
                    id=result.get("uuid", ""),
                    content=result.get("fact", ""),
                    artifact_type="relationship",
                    valid_at=datetime.fromisoformat(result["valid_at"]) if result.get("valid_at") else timestamp,
                    invalid_at=datetime.fromisoformat(result["expired_at"]) if result.get("expired_at") else None,
                    created_at=datetime.fromisoformat(result["created_at"]) if result.get("created_at") else None,
                    source="graphiti",
                    metadata={
                        "source_node": result.get("source_node"),
                        "target_node": result.get("target_node"),
                        "score": result.get("score"),
                    },
                )

            # Handle node result
            elif "name" in result:
                return KnowledgeArtifact(
                    id=result.get("uuid", ""),
                    content=result.get("summary", result.get("name", "")),
                    artifact_type="entity",
                    valid_at=timestamp,
                    source="graphiti",
                    metadata={
                        "labels": result.get("labels", []),
                        "name": result.get("name"),
                    },
                )

            return None

        except Exception as e:
            logger.warning(f"Failed to convert Graphiti result to artifact: {e}")
            return None

    async def search_with_temporal_filters(
        self,
        query: str,
        filter_type: TemporalFilter = TemporalFilter.CURRENT,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_results: int = 10,
        group_ids: Optional[List[str]] = None,
        use_hybrid: bool = True,
        rerank_method: RerankMethod = RerankMethod.RRF,
    ) -> List[KnowledgeArtifact]:
        """
        Search with temporal filtering.

        Args:
            query: Search query
            filter_type: Temporal filter type
            start_time: Start time for range queries
            end_time: End time for range queries
            max_results: Maximum results
            group_ids: Optional group IDs
            use_hybrid: Use hybrid search
            rerank_method: Reranking method

        Returns:
            List of KnowledgeArtifacts
        """
        if not self.graphiti_bridge or not self.graphiti_bridge.is_initialized:
            logger.warning("Graphiti bridge not initialized")
            return []

        try:
            # Build temporal filters
            temporal_filters = None
            if filter_type != TemporalFilter.CURRENT:
                temporal_filters = {
                    "filter_type": filter_type,
                }
                if start_time:
                    temporal_filters["start_time"] = start_time
                if end_time:
                    temporal_filters["end_time"] = end_time

            # Perform search
            if use_hybrid and self.graphiti_bridge.adapter:
                results = await self.graphiti_bridge.adapter.search_hybrid(
                    query=query,
                    use_bm25=True,
                    use_vector=True,
                    use_graph_traversal=True,
                    rerank_method=rerank_method.value,
                    temporal_filters=temporal_filters,
                    num_results=max_results,
                    group_ids=group_ids,
                )
            else:
                results = await self.graphiti_bridge.search(
                    query=query,
                    temporal_filters=temporal_filters,
                    num_results=max_results,
                    group_ids=group_ids,
                )

            # Convert results to artifacts
            artifacts = []
            for edge in results.get("edges", []):
                artifact = await self.graphiti_result_to_artifact(edge)
                if artifact:
                    artifacts.append(artifact)

            for node in results.get("nodes", []):
                artifact = await self.graphiti_result_to_artifact(node)
                if artifact:
                    artifacts.append(artifact)

            logger.info(f"Found {len(artifacts)} artifacts for query: {query}")
            return artifacts

        except Exception as e:
            logger.error(f"Search with temporal filters failed: {e}")
            return []

    async def query_at_point_in_time(
        self,
        query: str,
        timestamp: datetime,
        max_results: int = 10,
        group_ids: Optional[List[str]] = None,
    ) -> List[KnowledgeArtifact]:
        """
        Query knowledge as it was at a specific point in time.

        Args:
            query: Search query
            timestamp: Point in time
            max_results: Maximum results
            group_ids: Optional group IDs

        Returns:
            List of KnowledgeArtifacts valid at the given time
        """
        return await self.search_with_temporal_filters(
            query=query,
            filter_type=TemporalFilter.TIME_RANGE,
            start_time=timestamp - timedelta(hours=1),
            end_time=timestamp + timedelta(hours=1),
            max_results=max_results,
            group_ids=group_ids,
        )

    async def detect_contradictions(
        self,
        entity_name: str,
        time_range: Optional[tuple[datetime, datetime]] = None,
    ) -> ContradictionDetection:
        """
        Detect contradictions in knowledge about an entity.

        Args:
            entity_name: Entity to check
            time_range: Optional time range (start, end)

        Returns:
            ContradictionDetection result
        """
        if not self.graphiti_bridge or not self.graphiti_bridge.is_initialized:
            return ContradictionDetection(
                has_contradictions=False,
                contradictions=[],
                timestamp=datetime.utcnow(),
                confidence=0.0,
            )

        try:
            adapter = self.graphiti_bridge.adapter
            if not adapter:
                raise ValueError("Adapter not initialized")

            contradictions = await adapter.find_contradictions(
                entity_name=entity_name,
                time_range=time_range,
            )

            has_contradictions = len(contradictions) > 0
            confidence = min(1.0, len(contradictions) / 10.0) if contradictions else 0.0

            return ContradictionDetection(
                has_contradictions=has_contradictions,
                contradictions=contradictions,
                timestamp=datetime.utcnow(),
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Contradiction detection failed: {e}")
            return ContradictionDetection(
                has_contradictions=False,
                contradictions=[],
                timestamp=datetime.utcnow(),
                confidence=0.0,
            )

    async def get_entity_timeline(
        self,
        entity_name: str,
        start_time: datetime,
        end_time: datetime,
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
        if not self.graphiti_bridge or not self.graphiti_bridge.is_initialized:
            logger.warning("Graphiti bridge not initialized")
            return []

        try:
            adapter = self.graphiti_bridge.adapter
            if not adapter:
                raise ValueError("Adapter not initialized")

            timeline = await adapter.get_entity_timeline(
                entity_name=entity_name,
                start_time=start_time,
                end_time=end_time,
            )

            return timeline

        except Exception as e:
            logger.error(f"Failed to get entity timeline: {e}")
            return []

    async def get_valid_knowledge_at_time(
        self,
        timestamp: datetime,
        max_results: int = 100,
        group_ids: Optional[List[str]] = None,
    ) -> List[KnowledgeArtifact]:
        """
        Get all valid knowledge at a specific time.

        Args:
            timestamp: Point in time
            max_results: Maximum results
            group_ids: Optional group IDs

        Returns:
            List of valid KnowledgeArtifacts
        """
        # This is a broad query to find all knowledge valid at the time
        return await self.query_at_point_in_time(
            query="*",  # Broad query
            timestamp=timestamp,
            max_results=max_results,
            group_ids=group_ids,
        )

    def get_entity_type_mappings(self) -> List[EntityMapping]:
        """
        Get all entity type mappings.

        Returns:
            List of EntityMapping objects
        """
        return self.ENTITY_MAPPINGS

    def get_graphiti_type_for_artifact(self, artifact_type: str) -> str:
        """
        Get Graphiti custom type for a KnowledgeEngine artifact type.

        Args:
            artifact_type: KE artifact type

        Returns:
            Graphiti custom type
        """
        return self._map_entity_type(artifact_type)


async def get_temporal_bridge(
    config_path: Optional[str] = None,
) -> GraphitiTemporalBridge:
    """
    Get or create the temporal bridge singleton.

    Args:
        config_path: Optional path to Graphiti config

    Returns:
        GraphitiTemporalBridge instance
    """
    bridge = GraphitiTemporalBridge(config_path=config_path)
    await bridge.initialize()
    return bridge
