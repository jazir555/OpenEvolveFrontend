"""
Temporal Knowledge Engine with Graphiti Integration

This module extends the KnowledgeEngine to support temporal reasoning,
hybrid search, and contradiction detection using Graphiti's capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from knowledge_engine.core import KnowledgeEngine, KnowledgeState, EntityKnowledgeGraph

# Optional Graphiti integration
try:
    from knowledge_engine.integrations.graphiti.bridge import GraphitiBridge
    from knowledge_engine.integrations.graphiti.adapter import GraphitiAdapter
    GRAPHITI_AVAILABLE = True
except ImportError:
    GraphitiBridge = None
    GraphitiAdapter = None
    GRAPHITI_AVAILABLE = False

# Optional temporal filter
try:
    from knowledge_engine.integrations.base.knowledge_interface import TemporalFilter
    TEMPORAL_FILTER_AVAILABLE = True
except ImportError:
    TemporalFilter = None
    TEMPORAL_FILTER_AVAILABLE = False

logger = logging.getLogger(__name__)


class RerankMethod(Enum):
    """Reranking methods for hybrid search."""
    RRF = "rrf"  # Reciprocal Rank Fusion
    CROSS_ENCODER = "cross_encoder"  # Cross-encoder reranking
    WEIGHTED = "weighted"  # Weighted combination
    NONE = "none"  # No reranking


@dataclass
class KnowledgeArtifact:
    """
    Represents a knowledge artifact in the system.

    This is the canonical representation that bridges KnowledgeEngine
    and Graphiti's episodic knowledge model.
    """
    id: str
    content: str
    artifact_type: str  # solution_pattern, critique_insight, team_performance, etc.
    valid_at: datetime
    invalid_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    source: str = "openevolve"
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    group_id: Optional[str] = None

    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if artifact is valid at a given time."""
        if self.valid_at and timestamp < self.valid_at:
            return False
        if self.invalid_at and timestamp >= self.invalid_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "artifact_type": self.artifact_type,
            "valid_at": self.valid_at.isoformat() if self.valid_at else None,
            "invalid_at": self.invalid_at.isoformat() if self.invalid_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source": self.source,
            "metadata": self.metadata,
            "entities": self.entities,
            "relationships": self.relationships,
            "confidence": self.confidence,
            "group_id": self.group_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeArtifact":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            artifact_type=data["artifact_type"],
            valid_at=datetime.fromisoformat(data["valid_at"]) if data.get("valid_at") else datetime.utcnow(),
            invalid_at=datetime.fromisoformat(data["invalid_at"]) if data.get("invalid_at") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            source=data.get("source", "openevolve"),
            metadata=data.get("metadata", {}),
            entities=data.get("entities", []),
            relationships=data.get("relationships", []),
            confidence=data.get("confidence", 1.0),
            group_id=data.get("group_id"),
        )


@dataclass
class ContradictionDetection:
    """Result of contradiction detection."""
    has_contradictions: bool
    contradictions: List[Dict[str, Any]]
    timestamp: datetime
    confidence: float


class TemporalKnowledgeEngine(KnowledgeEngine):
    """
    Extended KnowledgeEngine with temporal reasoning capabilities.

    This engine integrates with Graphiti to provide:
    - Temporal knowledge tracking (valid_at, invalid_at timestamps)
    - Point-in-time queries
    - Hybrid search (BM25 + Vector + Graph traversal)
    - Contradiction detection
    - Timeline reconstruction
    """

    # Entity type mappings between KnowledgeEngine and Graphiti
    ENTITY_TYPE_MAPPINGS = {
        "solution_pattern": "Procedure",
        "critique_insight": "Requirement",
        "team_performance": "Preference",
        "problem": "Event",
        "workflow": "Document",
        "agent": "Organization",
        "tool": "Technology",
        "technique": "Methodology",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        graphiti_config_path: Optional[str] = None,
        enable_temporal: bool = True,
        enable_hybrid_search: bool = True,
        default_rerank_method: RerankMethod = RerankMethod.RRF,
    ):
        """
        Initialize the TemporalKnowledgeEngine.

        Args:
            config: Configuration dictionary
            graphiti_config_path: Path to Graphiti config (optional)
            enable_temporal: Enable temporal tracking
            enable_hybrid_search: Enable hybrid search
            default_rerank_method: Default reranking method
        """
        super().__init__(config=config)

        self.enable_temporal = enable_temporal
        self.enable_hybrid_search = enable_hybrid_search
        self.default_rerank_method = default_rerank_method

        # Graphiti integration
        self.graphiti_bridge: Optional[GraphitiBridge] = None
        self.graphiti_config_path = graphiti_config_path or "integrations/graphiti/config.yaml"

        # Artifact storage
        self.artifacts: Dict[str, KnowledgeArtifact] = {}
        self.artifact_lock = asyncio.Lock()

        # Initialize Graphiti if available
        asyncio.create_task(self._initialize_graphiti())

    async def _initialize_graphiti(self):
        """Initialize Graphiti bridge."""
        try:
            self.graphiti_bridge = GraphitiBridge()
            await self.graphiti_bridge.load_config(self.graphiti_config_path)
            await self.graphiti_bridge.initialize()
            logger.info("Graphiti bridge initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Graphiti bridge: {e}")
            self.graphiti_bridge = None

    async def add_knowledge_temporal(
        self,
        content: str,
        artifact_type: str,
        valid_at: datetime,
        invalid_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "openevolve",
        group_id: Optional[str] = None,
    ) -> Optional[KnowledgeArtifact]:
        """
        Add knowledge with temporal metadata.

        Args:
            content: Knowledge content
            artifact_type: Type of artifact
            valid_at: When knowledge becomes valid
            invalid_at: When knowledge becomes invalid (optional)
            metadata: Additional metadata
            source: Source identifier
            group_id: Optional group ID

        Returns:
            Created KnowledgeArtifact or None
        """
        if not self.enable_temporal:
            logger.warning("Temporal tracking disabled, using current time")
            valid_at = datetime.utcnow()

        artifact_id = f"{artifact_type}_{valid_at.isoformat()}_{hash(content) % 10000}"

        artifact = KnowledgeArtifact(
            id=artifact_id,
            content=content,
            artifact_type=artifact_type,
            valid_at=valid_at,
            invalid_at=invalid_at,
            created_at=datetime.utcnow(),
            source=source,
            metadata=metadata or {},
            group_id=group_id,
        )

        # Store locally
        async with self.artifact_lock:
            self.artifacts[artifact_id] = artifact

        # Add to Graphiti if available
        if self.graphiti_bridge and self.graphiti_bridge.is_initialized:
            try:
                await self._add_artifact_to_graphiti(artifact)
            except Exception as e:
                logger.warning(f"Failed to add artifact to Graphiti: {e}")

        logger.info(f"Added temporal knowledge artifact: {artifact_id}")
        return artifact

    async def _add_artifact_to_graphiti(self, artifact: KnowledgeArtifact):
        """
        Add KnowledgeArtifact to Graphiti as an episode.

        Args:
            artifact: KnowledgeArtifact to add
        """
        if not self.graphiti_bridge or not self.graphiti_bridge.is_initialized:
            return

        # Map artifact type to Graphiti custom type
        graphiti_type = self.ENTITY_TYPE_MAPPINGS.get(
            artifact.artifact_type,
            artifact.artifact_type
        )

        # Add episode
        await self.graphiti_bridge.add_episode(
            name=f"{artifact.artifact_type}: {artifact.id[:8]}",
            body=artifact.content,
            reference_time=artifact.valid_at,
            metadata={
                "artifact_id": artifact.id,
                "artifact_type": artifact.artifact_type,
                "graphiti_type": graphiti_type,
                "source": artifact.source,
                **artifact.metadata,
            },
            source=artifact.source,
            group_id=artifact.group_id,
        )

    async def query_at_time(
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
            timestamp: Point in time to query
            max_results: Maximum results
            group_ids: Optional group IDs

        Returns:
            List of KnowledgeArtifacts valid at the given time
        """
        results = []

        # Query local artifacts
        async with self.artifact_lock:
            for artifact in self.artifacts.values():
                if artifact.is_valid_at(timestamp):
                    # Simple relevance check (can be enhanced with embeddings)
                    if query.lower() in artifact.content.lower():
                        results.append(artifact)

        # Query Graphiti if available
        if self.graphiti_bridge and self.graphiti_bridge.is_initialized:
            try:
                temporal_filters = {
                    "filter_type": TemporalFilter.TIME_RANGE,
                    "start_time": timestamp - timedelta(hours=1),
                    "end_time": timestamp + timedelta(hours=1),
                }

                graphiti_results = await self.graphiti_bridge.search(
                    query=query,
                    temporal_filters=temporal_filters,
                    num_results=max_results,
                    group_ids=group_ids,
                )

                # Convert Graphiti results to artifacts
                for edge in graphiti_results.get("edges", []):
                    artifact = self._graphiti_edge_to_artifact(edge, timestamp)
                    if artifact and artifact not in results:
                        results.append(artifact)

            except Exception as e:
                logger.warning(f"Graphiti temporal query failed: {e}")

        # Sort by relevance and limit
        results = results[:max_results]
        logger.info(f"Query at time {timestamp}: found {len(results)} artifacts")
        return results

    def _graphiti_edge_to_artifact(
        self,
        edge: Dict[str, Any],
        timestamp: datetime
    ) -> Optional[KnowledgeArtifact]:
        """Convert Graphiti edge to KnowledgeArtifact."""
        try:
            valid_at = datetime.fromisoformat(edge.get("valid_at")) if edge.get("valid_at") else timestamp
            invalid_at = datetime.fromisoformat(edge.get("expired_at")) if edge.get("expired_at") else None

            return KnowledgeArtifact(
                id=edge.get("uuid", ""),
                content=edge.get("fact", ""),
                artifact_type="relationship",
                valid_at=valid_at,
                invalid_at=invalid_at,
                created_at=datetime.fromisoformat(edge.get("created_at")) if edge.get("created_at") else None,
                source="graphiti",
                metadata={"edge_data": edge},
            )
        except Exception as e:
            logger.warning(f"Failed to convert Graphiti edge to artifact: {e}")
            return None

    async def get_timeline(
        self,
        entity: str,
        start_time: datetime,
        end_time: datetime,
        group_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of events for an entity.

        Args:
            entity: Entity name
            start_time: Start of timeline
            end_time: End of timeline
            group_ids: Optional group IDs

        Returns:
            List of temporal events
        """
        timeline = []

        # Collect from local artifacts
        async with self.artifact_lock:
            for artifact in self.artifacts.values():
                if entity in artifact.entities:
                    if artifact.valid_at >= start_time and artifact.valid_at <= end_time:
                        timeline.append({
                            "timestamp": artifact.valid_at,
                            "event_type": artifact.artifact_type,
                            "description": artifact.content[:100],
                            "artifact_id": artifact.id,
                            "source": "local",
                        })

        # Query Graphiti if available
        if self.graphiti_bridge and self.graphiti_bridge.is_initialized:
            try:
                # Search for entity-related episodes
                query = f"events involving {entity}"
                temporal_filters = {
                    "filter_type": TemporalFilter.TIME_RANGE,
                    "start_time": start_time,
                    "end_time": end_time,
                }

                results = await self.graphiti_bridge.search(
                    query=query,
                    temporal_filters=temporal_filters,
                    num_results=50,
                    group_ids=group_ids,
                )

                for edge in results.get("edges", []):
                    valid_at = datetime.fromisoformat(edge.get("valid_at")) if edge.get("valid_at") else datetime.utcnow()
                    if start_time <= valid_at <= end_time:
                        timeline.append({
                            "timestamp": valid_at,
                            "event_type": "graphiti_relationship",
                            "description": edge.get("fact", "")[:100],
                            "artifact_id": edge.get("uuid", ""),
                            "source": "graphiti",
                        })

            except Exception as e:
                logger.warning(f"Failed to get timeline from Graphiti: {e}")

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        logger.info(f"Timeline for {entity}: {len(timeline)} events")
        return timeline

    async def search_with_graphiti(
        self,
        query: str,
        use_hybrid: bool = True,
        rerank_method: str = "rrf",
        temporal_filters: Optional[Dict[str, Any]] = None,
        max_results: int = 10,
        group_ids: Optional[List[str]] = None,
    ) -> List[KnowledgeArtifact]:
        """
        Search using Graphiti's hybrid search capabilities.

        Args:
            query: Search query
            use_hybrid: Use hybrid search (BM25 + Vector + BFS)
            rerank_method: Reranking method ('rrf', 'cross_encoder', 'weighted', 'none')
            temporal_filters: Optional temporal filters
            max_results: Maximum results
            group_ids: Optional group IDs

        Returns:
            List of KnowledgeArtifacts
        """
        if not self.enable_hybrid_search or not self.graphiti_bridge or not self.graphiti_bridge.is_initialized:
            logger.info("Hybrid search disabled or Graphiti unavailable, falling back to local search")
            return await self._local_search(query, max_results)

        try:
            # Perform search through Graphiti
            results = await self.graphiti_bridge.search(
                query=query,
                temporal_filters=temporal_filters,
                num_results=max_results,
                group_ids=group_ids,
            )

            # Convert results to artifacts
            artifacts = []
            for edge in results.get("edges", []):
                artifact = self._graphiti_edge_to_artifact(edge, datetime.utcnow())
                if artifact:
                    artifacts.append(artifact)

            logger.info(f"Graphiti hybrid search: {len(artifacts)} results")
            return artifacts[:max_results]

        except Exception as e:
            logger.warning(f"Graphiti search failed: {e}, falling back to local search")
            return await self._local_search(query, max_results)

    async def _local_search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[KnowledgeArtifact]:
        """
        Perform local search across stored artifacts.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of KnowledgeArtifacts
        """
        results = []
        query_lower = query.lower()

        async with self.artifact_lock:
            for artifact in self.artifacts.values():
                # Simple keyword matching
                if (query_lower in artifact.content.lower() or
                    query_lower in artifact.artifact_type.lower() or
                    any(query_lower in str(e).lower() for e in artifact.entities)):
                    results.append(artifact)

        return results[:max_results]

    async def detect_contradictions(
        self,
        knowledge_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
    ) -> ContradictionDetection:
        """
        Detect contradictions in the knowledge base.

        Args:
            knowledge_id: Optional specific knowledge ID to check
            group_ids: Optional group IDs to scope detection

        Returns:
            ContradictionDetection result
        """
        contradictions = []

        # Get artifacts to check
        artifacts_to_check = []
        if knowledge_id:
            async with self.artifact_lock:
                artifact = self.artifacts.get(knowledge_id)
                if artifact:
                    artifacts_to_check.append(artifact)
        else:
            async with self.artifact_lock:
                artifacts_to_check = list(self.artifacts.values())

        # Check for contradictions
        for i, artifact1 in enumerate(artifacts_to_check):
            for artifact2 in artifacts_to_check[i+1:]:
                contradiction = await _check_contradiction(artifact1, artifact2)
                if contradiction:
                    contradictions.append(contradiction)

        has_contradictions = len(contradictions) > 0
        confidence = min(1.0, len(contradictions) / max(1, len(artifacts_to_check)))

        result = ContradictionDetection(
            has_contradictions=has_contradictions,
            contradictions=contradictions,
            timestamp=datetime.utcnow(),
            confidence=confidence,
        )

        logger.info(f"Contradiction detection: {len(contradictions)} contradictions found")
        return result

    async def artifact_to_episode(self, artifact: KnowledgeArtifact) -> Dict[str, Any]:
        """
        Convert KnowledgeArtifact to Graphiti episode format.

        Args:
            artifact: KnowledgeArtifact to convert

        Returns:
            Episode dictionary
        """
        return {
            "name": f"{artifact.artifact_type}: {artifact.id[:8]}",
            "body": artifact.content,
            "reference_time": artifact.valid_at,
            "metadata": {
                "artifact_id": artifact.id,
                "artifact_type": artifact.artifact_type,
                "source": artifact.source,
                "confidence": artifact.confidence,
                **artifact.metadata,
            },
            "source": artifact.source,
            "group_id": artifact.group_id,
        }

    async def get_artifact(
        self,
        artifact_id: str,
    ) -> Optional[KnowledgeArtifact]:
        """
        Get artifact by ID.

        Args:
            artifact_id: Artifact ID

        Returns:
            KnowledgeArtifact or None
        """
        async with self.artifact_lock:
            return self.artifacts.get(artifact_id)

    async def get_artifacts_by_type(
        self,
        artifact_type: str,
        valid_at: Optional[datetime] = None,
    ) -> List[KnowledgeArtifact]:
        """
        Get artifacts by type.

        Args:
            artifact_type: Type of artifact
            valid_at: Optional point in time to check validity

        Returns:
            List of KnowledgeArtifacts
        """
        results = []
        async with self.artifact_lock:
            for artifact in self.artifacts.values():
                if artifact.artifact_type == artifact_type:
                    if valid_at is None or artifact.is_valid_at(valid_at):
                        results.append(artifact)
        return results

    async def invalidate_knowledge(
        self,
        artifact_id: str,
        invalid_at: datetime,
    ) -> bool:
        """
        Invalidate a knowledge artifact at a specific time.

        Args:
            artifact_id: Artifact ID
            invalid_at: Time when knowledge becomes invalid

        Returns:
            True if successful
        """
        async with self.artifact_lock:
            artifact = self.artifacts.get(artifact_id)
            if artifact:
                artifact.invalid_at = invalid_at
                logger.info(f"Invalidated artifact {artifact_id} at {invalid_at}")
                return True
        return False

    async def get_valid_knowledge(
        self,
        timestamp: Optional[datetime] = None,
    ) -> List[KnowledgeArtifact]:
        """
        Get all valid knowledge at a given time.

        Args:
            timestamp: Point in time (defaults to now)

        Returns:
            List of valid KnowledgeArtifacts
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        results = []
        async with self.artifact_lock:
            for artifact in self.artifacts.values():
                if artifact.is_valid_at(timestamp):
                    results.append(artifact)

        return results


async def _check_contradiction(
    artifact1: KnowledgeArtifact,
    artifact2: KnowledgeArtifact,
) -> Optional[Dict[str, Any]]:
    """
    Check if two artifacts contradict each other.

    Args:
        artifact1: First artifact
        artifact2: Second artifact

    Returns:
        Contradiction details or None
    """
    # Check temporal overlap
    if not _temporal_overlap(artifact1, artifact2):
        return None

    # Simple heuristic: if artifacts have opposite keywords
    content1 = artifact1.content.lower()
    content2 = artifact2.content.lower()

    # Check for negation patterns
    contradiction_patterns = [
        ("not ", ""),
        ("never ", "always "),
        ("cannot ", "can "),
        ("won't ", "will "),
        ("false", "true"),
    ]

    for pattern1, pattern2 in contradiction_patterns:
        if pattern1 in content1 and pattern2 in content2:
            return {
                "type": "temporal_contradiction",
                "artifact1_id": artifact1.id,
                "artifact2_id": artifact2.id,
                "reason": f"Found contradictory patterns: '{pattern1}' vs '{pattern2}'",
                "severity": "medium",
            }

    return None


def _temporal_overlap(
    artifact1: KnowledgeArtifact,
    artifact2: KnowledgeArtifact,
) -> bool:
    """Check if two artifacts have temporal overlap."""
    # Artifact1 is valid during Artifact2's validity
    if artifact1.valid_at <= (artifact2.invalid_at or datetime.max) and \
       (artifact1.invalid_at or datetime.max) >= artifact2.valid_at:
        return True
    return False
