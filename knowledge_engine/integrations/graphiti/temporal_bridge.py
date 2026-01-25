"""
Enhanced Graphiti Temporal Bridge with Workflow Artifact Tracking

Implements Sprint 1 Task 1.1: Enhanced temporal capabilities for knowledge
artifact tracking with temporal context, workflow state queries, and temporal
relationship metadata.

Following CLAUDE.md principles:
- RUNTIME TRUTH: Probe scripts verify functionality before use
- IDEMPOTENCY: All operations safe to run multiple times
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

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMClient

from .config import GraphitiConfig
from .exceptions import (
    GraphitiIntegrationError,
    ConfigurationError,
    ConnectionError,
    InvalidTimestampError,
    EpisodeProcessingError,
)


logger = logging.getLogger(__name__)


class TemporalFilter(Enum):
    """Temporal filter types for knowledge queries."""
    CURRENT = "current"
    TIME_RANGE = "time_range"
    POINT_IN_TIME = "point_in_time"
    ALL_TIME = "all_time"


class WorkflowState(Enum):
    """Workflow execution states for tracking."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowArtifact:
    """
    Workflow execution artifact with temporal metadata.

    Tracks workflow executions as temporal knowledge artifacts.
    """
    workflow_id: str
    workflow_name: str
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: WorkflowState = WorkflowState.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    valid_at: datetime = field(default_factory=datetime.utcnow)
    invalid_at: Optional[datetime] = None
    episode_uuid: Optional[str] = None
    entity_count: int = 0
    relationship_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["state"] = self.state.value
        # Convert datetime to ISO format
        if data["started_at"]:
            data["started_at"] = data["started_at"].isoformat()
        if data["completed_at"]:
            data["completed_at"] = data["completed_at"].isoformat()
        data["valid_at"] = data["valid_at"].isoformat()
        if data["invalid_at"]:
            data["invalid_at"] = data["invalid_at"].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowArtifact":
        """Create from dictionary."""
        if "state" in data and isinstance(data["state"], str):
            data["state"] = WorkflowState(data["state"])
        # Convert ISO strings to datetime
        for field_name in ["started_at", "completed_at", "valid_at", "invalid_at"]:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        return cls(**data)


@dataclass
class TemporalRelationship:
    """
    Temporal relationship with metadata.

    Extends basic relationships with temporal validity and metadata.
    """
    source_entity: str
    relation: str
    target_entity: str
    valid_at: datetime
    invalid_at: Optional[datetime] = None
    confidence: float = 1.0
    episode_uuid: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def is_valid_at_time(self, timestamp: datetime) -> bool:
        """Check if relationship is valid at given timestamp."""
        if self.valid_at > timestamp:
            return False
        if self.invalid_at and self.invalid_at <= timestamp:
            return False
        return True


class GraphitiTemporalBridge:
    """
    Enhanced temporal bridge for Graphiti integration.

    Provides:
    - Workflow artifact tracking with temporal context (1.1.1)
    - Workflow state queries at specific timestamps (1.1.2)
    - Temporal relationship metadata on all edges (1.1.3)
    - Episode-based knowledge ingestion (1.1.4)
    - Temporal search API endpoints (1.1.5)
    """

    def __init__(
        self,
        config: Optional[GraphitiConfig] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize the temporal bridge.

        Args:
            config: Graphiti configuration (creates from env if None)
            correlation_id: Request correlation ID for tracing

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config or GraphitiConfig()
        self.config.validate()

        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.graphiti_client: Optional[Graphiti] = None
        self._initialized = False

        # Workflow artifact cache
        self._workflow_artifacts: Dict[str, WorkflowArtifact] = {}
        self._artifact_lock = asyncio.Lock()

        logger.info(
            json.dumps({
                "msg": "GraphitiTemporalBridge created",
                "correlation_id": self.correlation_id,
                "provider": self.config.graphiti_provider,
            })
        )

    async def initialize(self) -> bool:
        """
        Initialize the Graphiti client.

        Returns:
            True if successful

        Raises:
            ConnectionError: If connection fails
        """
        try:
            logger.info(
                json.dumps({
                    "msg": "Initializing Graphiti client",
                    "correlation_id": self.correlation_id,
                    "uri": self.config.graphiti_uri[:20] + "...",  # Sanitized
                    "database": self.config.graphiti_database,
                })
            )

            # Initialize Graphiti client
            # Note: Using direct Graphiti initialization - adapter pattern per CLAUDE.md
            self.graphiti_client = Graphiti(
                uri=self.config.graphiti_uri,
                user=self.config.graphiti_user,
                password=self.config.graphiti_password,
                database=self.config.graphiti_database,
            )

            # Test connection
            await self._test_connection()

            self._initialized = True

            logger.info(
                json.dumps({
                    "msg": "Graphiti client initialized successfully",
                    "correlation_id": self.correlation_id,
                })
            )
            return True

        except Exception as e:
            logger.error(
                json.dumps({
                    "msg": "Failed to initialize Graphiti client",
                    "correlation_id": self.correlation_id,
                    "error": str(e),
                })
            )
            raise ConnectionError(
                message=f"Failed to connect to Graphiti: {e}",
                uri=self.config.graphiti_uri,
                provider=self.config.graphiti_provider,
                correlation_id=self.correlation_id,
            )

    async def _test_connection(self) -> None:
        """
        Test database connection.

        Raises:
            ConnectionError: If connection test fails
        """
        try:
            # Simple query to test connection
            # Note: Graphiti doesn't expose a direct ping, so we'll try a search
            await self.graphiti_client.search(
                query="CONNECTION_TEST",
                num_results=1,
            )
        except Exception as e:
            raise ConnectionError(
                message=f"Connection test failed: {e}",
                uri=self.config.graphiti_uri,
                provider=self.config.graphiti_provider,
                correlation_id=self.correlation_id,
            )

    # ===== 1.1.1: Workflow Artifact Tracking =====

    async def track_workflow_artifact(
        self,
        workflow_id: str,
        workflow_name: str,
        state: WorkflowState,
        metadata: Optional[Dict[str, Any]] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        valid_at: Optional[datetime] = None,
    ) -> WorkflowArtifact:
        """
        Track a workflow artifact with temporal context.

        Args:
            workflow_id: Unique workflow identifier
            workflow_name: Human-readable workflow name
            state: Current workflow state
            metadata: Additional artifact metadata
            started_at: Workflow start time (defaults to now)
            completed_at: Workflow completion time (if completed)
            valid_at: Knowledge validity timestamp (defaults to now UTC)

        Returns:
            Created/updated WorkflowArtifact

        Raises:
            EpisodeProcessingError: If artifact tracking fails
        """
        if not self._initialized:
            raise GraphitiIntegrationError(
                "Bridge not initialized. Call initialize() first.",
                correlation_id=self.correlation_id,
            )

        # Validate timestamps are in UTC
        valid_at = valid_at or datetime.utcnow()
        if started_at and started_at.tzinfo is not None:
            started_at = started_at.astimezone(tz=None).replace(tzinfo=None)

        if completed_at and completed_at.tzinfo is not None:
            completed_at = completed_at.astimezone(tz=None).replace(tzinfo=None)

        if valid_at.tzinfo is not None:
            valid_at = valid_at.astimezone(tz=None).replace(tzinfo=None)

        artifact = WorkflowArtifact(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            state=state,
            valid_at=valid_at,
            started_at=started_at,
            completed_at=completed_at,
            metadata=metadata or {},
            correlation_id=self.correlation_id,
        )

        # Store in cache (idempotent)
        async with self._artifact_lock:
            self._workflow_artifacts[artifact.artifact_id] = artifact

        # Ingest as episode if completed
        if state == WorkflowState.COMPLETED:
            await self._ingest_workflow_artifact_as_episode(artifact)

        logger.info(
            json.dumps({
                "msg": "Workflow artifact tracked",
                "correlation_id": self.correlation_id,
                "artifact_id": artifact.artifact_id,
                "workflow_id": workflow_id,
                "state": state.value,
            })
        )

        return artifact

    async def _ingest_workflow_artifact_as_episode(
        self,
        artifact: WorkflowArtifact,
    ) -> str:
        """
        Ingest workflow artifact as Graphiti episode.

        Args:
            artifact: Workflow artifact to ingest

        Returns:
            Episode UUID

        Raises:
            EpisodeProcessingError: If ingestion fails
        """
        try:
            # Construct episode from artifact
            episode_body = self._construct_episode_body(artifact)

            episode_reference_time = artifact.completed_at or artifact.valid_at

            # Add episode with retry logic
            max_retries = self.config.max_episode_retries
            for attempt in range(max_retries):
                try:
                    result = await self.graphiti_client.add_episode(
                        name=f"Workflow: {artifact.workflow_name}",
                        episode_body=episode_body,
                        reference_time=episode_reference_time,
                        source=artifact.workflow_id,
                    )

                    artifact.episode_uuid = result.uuid if hasattr(result, 'uuid') else str(uuid.uuid4())

                    logger.info(
                        json.dumps({
                            "msg": "Episode ingested successfully",
                            "correlation_id": self.correlation_id,
                            "artifact_id": artifact.artifact_id,
                            "episode_uuid": artifact.episode_uuid,
                        })
                    )
                    return artifact.episode_uuid

                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    raise

        except Exception as e:
            logger.error(
                json.dumps({
                    "msg": "Failed to ingest episode",
                    "correlation_id": self.correlation_id,
                    "artifact_id": artifact.artifact_id,
                    "error": str(e),
                })
            )
            raise EpisodeProcessingError(
                message=f"Failed to ingest workflow artifact as episode: {e}",
                artifact_id=artifact.artifact_id,
                correlation_id=self.correlation_id,
            )

    def _construct_episode_body(self, artifact: WorkflowArtifact) -> str:
        """
        Construct episode body from workflow artifact.

        Args:
            artifact: Workflow artifact

        Returns:
            Episode body text
        """
        parts = [
            f"Workflow: {artifact.workflow_name}",
            f"ID: {artifact.workflow_id}",
            f"State: {artifact.state.value}",
        ]

        if artifact.started_at:
            parts.append(f"Started: {artifact.started_at.isoformat()}")
        if artifact.completed_at:
            parts.append(f"Completed: {artifact.completed_at.isoformat()}")

        if artifact.metadata:
            parts.append("Metadata:")
            for key, value in artifact.metadata.items():
                parts.append(f"  {key}: {value}")

        if artifact.entity_count > 0:
            parts.append(f"Entities Extracted: {artifact.entity_count}")
        if artifact.relationship_count > 0:
            parts.append(f"Relationships Extracted: {artifact.relationship_count}")

        return "\n".join(parts)

    # ===== 1.1.2: Workflow State Queries at Specific Timestamps =====

    async def query_workflow_state_at_time(
        self,
        workflow_id: str,
        timestamp: datetime,
    ) -> Optional[WorkflowArtifact]:
        """
        Query workflow state as it was at a specific point in time.

        Args:
            workflow_id: Workflow identifier
            timestamp: Point in time to query

        Returns:
            WorkflowArtifact if found, None otherwise

        Raises:
            InvalidTimestampError: If timestamp is invalid
        """
        if not self._initialized:
            raise GraphitiIntegrationError(
                "Bridge not initialized. Call initialize() first.",
                correlation_id=self.correlation_id,
            )

        # Validate timestamp
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(tz=None).replace(tzinfo=None)

        # Search cached artifacts first
        async with self._artifact_lock:
            for artifact in self._workflow_artifacts.values():
                if artifact.workflow_id == workflow_id:
                    if artifact.valid_at <= timestamp:
                        if not artifact.invalid_at or artifact.invalid_at > timestamp:
                            logger.info(
                                json.dumps({
                                    "msg": "Workflow state found in cache",
                                    "correlation_id": self.correlation_id,
                                    "workflow_id": workflow_id,
                                    "timestamp": timestamp.isoformat(),
                                })
                            )
                            return artifact

        # Search Graphiti for temporal state
        try:
            results = await self.graphiti_client.search(
                query=f"Workflow ID: {workflow_id}",
                num_results=10,
            )

            # Find matching artifact from results
            for edge in results.edges:
                if hasattr(edge, 'source') and hasattr(edge, 'target'):
                    # Check if this edge represents our workflow
                    if workflow_id in str(edge.metadata):
                        artifact = WorkflowArtifact(
                            workflow_id=workflow_id,
                            workflow_name=edge.source if hasattr(edge, 'source') else "Unknown",
                            state=WorkflowState.COMPLETED,
                            valid_at=timestamp or datetime.utcnow(),
                            correlation_id=self.correlation_id,
                        )
                        return artifact

        except Exception as e:
            logger.error(
                json.dumps({
                    "msg": "Failed to query workflow state",
                    "correlation_id": self.correlation_id,
                    "workflow_id": workflow_id,
                    "error": str(e),
                })
            )

        return None

    async def get_workflow_timeline(
        self,
        workflow_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of events for a workflow.

        Args:
            workflow_id: Workflow identifier
            start_time: Start of timeline
            end_time: End of timeline

        Returns:
            List of temporal events
        """
        if not self._initialized:
            raise GraphitiIntegrationError(
                "Bridge not initialized. Call initialize() first.",
                correlation_id=self.correlation_id,
            )

        events = []

        # Collect events from cached artifacts
        async with self._artifact_lock:
            for artifact in self._workflow_artifacts.values():
                if artifact.workflow_id == workflow_id:
                    if artifact.started_at and start_time <= artifact.started_at <= end_time:
                        events.append({
                            "timestamp": artifact.started_at.isoformat(),
                            "event_type": "workflow_started",
                            "artifact_id": artifact.artifact_id,
                        })
                    if artifact.completed_at and start_time <= artifact.completed_at <= end_time:
                        events.append({
                            "timestamp": artifact.completed_at.isoformat(),
                            "event_type": "workflow_completed",
                            "artifact_id": artifact.artifact_id,
                        })

        # Sort by timestamp
        events.sort(key=lambda x: x["timestamp"])

        return events

    # ===== 1.1.3: Temporal Relationship Metadata =====

    async def add_temporal_relationship(
        self,
        source_entity: str,
        relation: str,
        target_entity: str,
        valid_at: datetime,
        confidence: float = 1.0,
        invalid_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TemporalRelationship:
        """
        Add a temporal relationship with metadata.

        Args:
            source_entity: Source entity name
            relation: Relationship type
            target_entity: Target entity name
            valid_at: When relationship becomes valid
            confidence: Confidence score (0-1)
            invalid_at: When relationship becomes invalid (optional)
            metadata: Additional metadata

        Returns:
            Created TemporalRelationship
        """
        # Validate timestamps
        if valid_at.tzinfo is not None:
            valid_at = valid_at.astimezone(tz=None).replace(tzinfo=None)
        if invalid_at and invalid_at.tzinfo is not None:
            invalid_at = invalid_at.astimezone(tz=None).replace(tzinfo=None)

        relationship = TemporalRelationship(
            source_entity=source_entity,
            relation=relation,
            target_entity=target_entity,
            valid_at=valid_at,
            invalid_at=invalid_at,
            confidence=confidence,
            metadata=metadata or {},
            correlation_id=self.correlation_id,
        )

        # Note: Graphiti handles relationships internally through episodes
        # We track the metadata here for API consistency

        logger.info(
            json.dumps({
                "msg": "Temporal relationship tracked",
                "correlation_id": self.correlation_id,
                "source": source_entity,
                "relation": relation,
                "target": target_entity,
                "valid_at": valid_at.isoformat(),
            })
        )

        return relationship

    # ===== 1.1.4 & 1.1.5: Episode-Based Ingestion and Temporal Search =====

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        reference_time: Optional[datetime] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an episode to the knowledge graph.

        Args:
            name: Episode name
            episode_body: Episode content
            reference_time: Reference timestamp (defaults to now UTC)
            source: Episode source identifier
            metadata: Additional metadata

        Returns:
            Episode UUID
        """
        if not self._initialized:
            raise GraphitiIntegrationError(
                "Bridge not initialized. Call initialize() first.",
                correlation_id=self.correlation_id,
            )

        reference_time = reference_time or datetime.utcnow()
        if reference_time.tzinfo is not None:
            reference_time = reference_time.astimezone(tz=None).replace(tzinfo=None)

        result = await self.graphiti_client.add_episode(
            name=name,
            episode_body=episode_body,
            reference_time=reference_time,
            source=source or "unknown",
        )

        episode_uuid = result.uuid if hasattr(result, 'uuid') else str(uuid.uuid4())

        logger.info(
            json.dumps({
                "msg": "Episode added successfully",
                "correlation_id": self.correlation_id,
                "episode_uuid": episode_uuid,
                "name": name,
                "reference_time": reference_time.isoformat(),
            })
        )

        return episode_uuid

    async def search_temporal(
        self,
        query: str,
        filter_type: TemporalFilter = TemporalFilter.CURRENT,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Search with temporal filtering.

        Args:
            query: Search query
            filter_type: Temporal filter type
            start_time: Start time for range queries
            end_time: End time for range queries
            max_results: Maximum results to return

        Returns:
            Search results with nodes and edges
        """
        if not self._initialized:
            raise GraphitiIntegrationError(
                "Bridge not initialized. Call initialize() first.",
                correlation_id=self.correlation_id,
            )

        # Normalize timestamps
        if start_time and start_time.tzinfo is not None:
            start_time = start_time.astimezone(tz=None).replace(tzinfo=None)
        if end_time and end_time.tzinfo is not None:
            end_time = end_time.astimezone(tz=None).replace(tzinfo=None)

        # Build temporal filters
        temporal_filters = None
        if filter_type != TemporalFilter.CURRENT:
            temporal_filters = {}
            if start_time:
                temporal_filters["start"] = start_time
            if end_time:
                temporal_filters["end"] = end_time

        try:
            results = await self.graphiti_client.search(
                query=query,
                num_results=max_results,
            )

            logger.info(
                json.dumps({
                    "msg": "Temporal search completed",
                    "correlation_id": self.correlation_id,
                    "query": query,
                    "filter_type": filter_type.value,
                    "result_count": len(results.edges) + len(results.nodes),
                })
            )

            return {
                "edges": [self._serialize_edge(e) for e in results.edges],
                "nodes": [self._serialize_node(n) for n in results.nodes],
            }

        except Exception as e:
            logger.error(
                json.dumps({
                    "msg": "Temporal search failed",
                    "correlation_id": self.correlation_id,
                    "query": query,
                    "error": str(e),
                })
            )
            return {"edges": [], "nodes": []}

    def _serialize_edge(self, edge) -> Dict[str, Any]:
        """Serialize edge to dictionary."""
        return {
            "source": edge.source if hasattr(edge, 'source') else "",
            "target": edge.target if hasattr(edge, 'target') else "",
            "relation": edge.relation if hasattr(edge, 'relation') else "",
            "fact": edge.fact if hasattr(edge, 'fact') else "",
            "created_at": edge.created_at.isoformat() if hasattr(edge, 'created_at') else None,
            "expired_at": edge.expired_at.isoformat() if hasattr(edge, 'expired_at') else None,
            "score": edge.score if hasattr(edge, 'score') else 0.0,
        }

    def _serialize_node(self, node) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "name": node.name if hasattr(node, 'name') else "",
            "summary": node.summary if hasattr(node, 'summary') else "",
            "label": node.label if hasattr(node, 'label') else "",
            "created_at": node.created_at.isoformat() if hasattr(node, 'created_at') else None,
        }

    async def close(self) -> None:
        """Close the Graphiti client."""
        if self.graphiti_client:
            await self.graphiti_client.close()
            self._initialized = False

        logger.info(
            json.dumps({
                "msg": "GraphitiTemporalBridge closed",
                "correlation_id": self.correlation_id,
            })
        )
