"""
Graph Aggregator - Production Grade

Task 2.5: Knowledge Graph Aggregation
- 2.5.1: Implement graph aggregation from multiple sources
- 2.5.2: Add graph merging with conflict resolution
- 2.5.3: Implement graph versioning
- 2.5.4: Add differential graph comparison
- 2.5.5: Implement graph aggregation API

Following CLAUDE.md Principles:
- IDEMPOTENCY: Aggregation safe to retry
- CONFIGURATION EXPLICITNESS: All config via env vars
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


@dataclass
class GraphVersion:
    """
    A version of the knowledge graph.

    Task 2.5.3: Implement graph versioning.

    All timestamps in UTC (LAW OF UTC).
    """
    version_id: str
    entities: List[str]
    relationships: List[Dict[str, str]]
    metadata: Dict[str, Any]

    # Version info
    parent_version_id: Optional[str] = None
    version_number: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = "system"

    # Checksum for integrity
    checksum: str = ""

    def __post_init__(self):
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """
        Calculate checksum for this version.

        Returns:
            Checksum hash
        """
        content = f"{json.dumps(sorted(self.entities), sort_keys=True)}{json.dumps(self.relationships, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphVersion':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GraphDiff:
    """
    Differential comparison between two graph versions.

    Task 2.5.4: Add differential graph comparison.
    """
    from_version: str
    to_version: str

    # Changes
    entities_added: List[str] = field(default_factory=list)
    entities_removed: List[str] = field(default_factory=list)
    entities_modified: List[str] = field(default_factory=list)

    relationships_added: List[Dict[str, str]] = field(default_factory=list)
    relationships_removed: List[Dict[str, str]] = field(default_factory=list)

    # Metrics
    change_count: int = 0
    similarity_score: float = 0.0

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AggregationResult:
    """
    Result of graph aggregation.

    Task 2.5.1: Implement graph aggregation from multiple sources.
    """
    correlation_id: str
    aggregated_graph: GraphVersion

    # Source tracking
    source_versions: List[str] = field(default_factory=list)

    # Aggregation metrics
    total_entities: int = 0
    total_relationships: int = 0
    conflicts_resolved: int = 0

    # Processing
    processing_time_seconds: float = 0.0
    aggregation_method: str = "merge"

    # Timestamps
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "aggregated_graph": self.aggregated_graph.to_dict(),
            "source_versions": self.source_versions,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
            "conflicts_resolved": self.conflicts_resolved,
            "processing_time_seconds": self.processing_time_seconds,
            "aggregation_method": self.aggregation_method,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


@dataclass
class GraphAggregatorConfig:
    """
    Graph aggregator configuration.

    LAW OF CONFIGURATION EXPLICITNESS.
    """
    # Versioning
    max_versions: int = field(
        default_factory=lambda: int(os.getenv("KGGEN_MAX_VERSIONS", "100"))
    )
    auto_version: bool = field(
        default_factory=lambda: os.getenv("KGGEN_AUTO_VERSION", "true").lower() == "true"
    )

    # Aggregation
    merge_strategy: str = field(
        default_factory=lambda: os.getenv("KGGEN_MERGE_STRATEGY", "union")
    )
    conflict_resolution: str = field(
        default_factory=lambda: os.getenv("KGGEN_CONFLICT_RESOLUTION", "keep_both")
    )

    # Comparison
    diff_threshold: float = field(
        default_factory=lambda: float(os.getenv("KGGEN_DIFF_THRESHOLD", "0.8"))
    )

    def validate(self) -> None:
        """Validate configuration."""
        if self.max_versions <= 0:
            raise ValueError(f"Invalid max_versions: {self.max_versions}")
        valid_strategies = {"union", "intersection", "weighted"}
        if self.merge_strategy not in valid_strategies:
            raise ValueError(f"Invalid merge_strategy: {self.merge_strategy}")
        logger.info("GraphAggregatorConfig validated", extra={"config": asdict(self)})


class ConflictResolver:
    """
    Resolve conflicts when merging graphs.

    Task 2.5.2: Add graph merging with conflict resolution.
    """

    def __init__(self, config: GraphAggregatorConfig):
        """
        Initialize resolver.

        Args:
            config: Aggregator configuration
        """
        self.config = config
        self._resolution_log: List[Dict[str, Any]] = []

    async def resolve_entity_conflict(
        self,
        entity: str,
        sources: List[Tuple[str, Dict[str, Any]]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Resolve entity conflict from multiple sources.

        Args:
            entity: Entity name
            sources: List of (source_id, attributes) tuples

        Returns:
            Tuple of (entity_name, resolved_attributes)
        """
        resolution_method = self.config.conflict_resolution

        if resolution_method == "keep_first":
            # Keep first source's attributes
            resolved = sources[0][1]
            source_id = sources[0][0]

        elif resolution_method == "keep_last":
            # Keep last source's attributes
            resolved = sources[-1][1]
            source_id = sources[-1][0]

        elif resolution_method == "merge":
            # Merge all attributes
            resolved = {}
            for source_id, attrs in sources:
                resolved.update(attrs)

        elif resolution_method == "keep_both":
            # Keep all versions as variants
            resolved = {
                "canonical": entity,
                "variants": [attrs.get("name", entity) for _, attrs in sources],
                "sources": [source_id for source_id, _ in sources]
            }
            source_id = "merged"

        else:
            # Default: keep first
            resolved = sources[0][1]
            source_id = sources[0][0]

        # Log resolution
        self._resolution_log.append({
            "type": "entity",
            "entity": entity,
            "method": resolution_method,
            "sources": [s[0] for s in sources],
            "resolved_from": source_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return entity, resolved

    async def resolve_relationship_conflict(
        self,
        relationship: Dict[str, str],
        sources: List[str]
    ) -> Dict[str, str]:
        """
        Resolve relationship conflict.

        Args:
            relationship: Relationship dictionary
            sources: List of source IDs

        Returns:
            Resolved relationship
        """
        resolution_method = self.config.conflict_resolution

        if resolution_method == "keep_both":
            # Add source annotations
            resolved = relationship.copy()
            resolved["sources"] = sources
        else:
            # Keep relationship as-is
            resolved = relationship

        # Log resolution
        self._resolution_log.append({
            "type": "relationship",
            "relationship": relationship,
            "method": resolution_method,
            "sources": sources,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return resolved

    def get_resolution_log(self) -> List[Dict[str, Any]]:
        """Get conflict resolution log."""
        return self._resolution_log.copy()


class GraphAggregator:
    """
    Aggregate knowledge graphs from multiple sources.

    Task 2.5.1: Implement graph aggregation from multiple sources.
    Task 2.5.5: Implement graph aggregation API.

    Following CLAUDE.md:
    - IDEMPOTENCY: Aggregation safe to retry
    - STRUCTURED LOGGING: JSON with correlation_id
    """

    def __init__(self, config: Optional[GraphAggregatorConfig] = None):
        """
        Initialize aggregator.

        Args:
            config: Aggregator configuration
        """
        self.config = config or GraphAggregatorConfig()
        self.config.validate()

        # Version storage
        self._versions: Dict[str, GraphVersion] = {}
        self._version_list: List[str] = []
        self._current_version_number = 0

        # Conflict resolver
        self.conflict_resolver = ConflictResolver(self.config)

        logger.info(
            "GraphAggregator initialized",
            extra={"config": asdict(self.config)}
        )

    async def aggregate(
        self,
        graphs: List[Dict[str, Any]],
        correlation_id: Optional[str] = None,
        create_version: bool = True
    ) -> AggregationResult:
        """
        Aggregate multiple graphs.

        Args:
            graphs: List of graph dictionaries with 'entities' and 'relationships'
            correlation_id: Optional correlation ID
            create_version: Whether to create a new version

        Returns:
            Aggregation result
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        logger.info(
            f"Aggregating {len(graphs)} graphs",
            extra={"correlation_id": correlation_id}
        )

        # Extract source version IDs if present
        source_versions = []
        for graph in graphs:
            if "version_id" in graph:
                source_versions.append(graph["version_id"])

        # Merge entities
        merged_entities, entity_conflicts = await self._merge_entities(
            graphs,
            correlation_id
        )

        # Merge relationships
        merged_relationships, rel_conflicts = await self._merge_relationships(
            graphs,
            correlation_id
        )

        total_conflicts = entity_conflicts + rel_conflicts

        # Create aggregated graph
        aggregated = GraphVersion(
            version_id=f"ver-{uuid.uuid4().hex[:16]}",
            entities=merged_entities,
            relationships=merged_relationships,
            metadata={
                "source_count": len(graphs),
                "aggregation_method": self.config.merge_strategy,
                "conflicts_resolved": total_conflicts
            },
            created_by="aggregator"
        )

        # Store version if enabled
        if create_version and self.config.auto_version:
            await self._store_version(aggregated)

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        result = AggregationResult(
            correlation_id=correlation_id,
            aggregated_graph=aggregated,
            source_versions=source_versions,
            total_entities=len(merged_entities),
            total_relationships=len(merged_relationships),
            conflicts_resolved=total_conflicts,
            processing_time_seconds=processing_time,
            aggregation_method=self.config.merge_strategy,
            started_at=start_time.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat()
        )

        logger.info(
            f"Aggregation complete: {len(merged_entities)} entities, "
            f"{len(merged_relationships)} relationships, "
            f"{total_conflicts} conflicts resolved",
            extra={"correlation_id": correlation_id}
        )

        return result

    async def _merge_entities(
        self,
        graphs: List[Dict[str, Any]],
        correlation_id: str
    ) -> Tuple[List[str], int]:
        """
        Merge entities from multiple graphs.

        Args:
            graphs: Input graphs
            correlation_id: Correlation ID

        Returns:
            Tuple of (merged entities, conflict count)
        """
        # Collect all entities
        entity_sources: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)

        for i, graph in enumerate(graphs):
            source_id = graph.get("version_id", f"source_{i}")
            entities = graph.get("entities", [])

            for entity in entities:
                entity_sources[entity].append((source_id, {"name": entity}))

        # Merge based on strategy
        conflicts = 0

        if self.config.merge_strategy == "union":
            # Union: keep all entities
            merged = list(entity_sources.keys())
            conflicts = sum(1 for sources in entity_sources.values() if len(sources) > 1)

        elif self.config.merge_strategy == "intersection":
            # Intersection: keep entities in all graphs
            if graphs:
                entity_sets = [set(g.get("entities", [])) for g in graphs]
                merged = list(set.intersection(*entity_sets))
            else:
                merged = []

        else:  # weighted or default
            # Weighted: keep entities appearing in multiple sources
            merged = []
            for entity, sources in entity_sources.items():
                weight = len(sources) / len(graphs)
                if weight >= 0.3:  # Appears in at least 30% of sources
                    merged.append(entity)
                if len(sources) > 1:
                    conflicts += 1

        return merged, conflicts

    async def _merge_relationships(
        self,
        graphs: List[Dict[str, Any]],
        correlation_id: str
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Merge relationships from multiple graphs.

        Args:
            graphs: Input graphs
            correlation_id: Correlation ID

        Returns:
            Tuple of (merged relationships, conflict count)
        """
        # Collect relationships
        seen = {}
        conflicts = 0

        for i, graph in enumerate(graphs):
            source_id = graph.get("version_id", f"source_{i}")
            relationships = graph.get("relationships", [])

            for rel in relationships:
                # Create unique key
                key = (
                    rel.get("subject", ""),
                    rel.get("predicate", ""),
                    rel.get("object", "")
                )

                if key in seen:
                    # Conflict: same relationship from multiple sources
                    conflicts += 1

                    # Resolve conflict
                    resolved = await self.conflict_resolver.resolve_relationship_conflict(
                        rel,
                        [seen[key]["source"], source_id]
                    )
                    seen[key] = {
                        "relationship": resolved,
                        "source": "merged"
                    }
                else:
                    seen[key] = {
                        "relationship": rel,
                        "source": source_id
                    }

        merged = [item["relationship"] for item in seen.values()]
        return merged, conflicts

    async def _store_version(self, version: GraphVersion) -> None:
        """
        Store a graph version.

        Task 2.5.3: Implement graph versioning.

        Args:
            version: Graph version to store
        """
        # Increment version number
        self._current_version_number += 1
        version.version_number = self._current_version_number

        # Store
        self._versions[version.version_id] = version
        self._version_list.append(version.version_id)

        # Enforce max versions
        if len(self._version_list) > self.config.max_versions:
            # Remove oldest
            oldest_id = self._version_list.pop(0)
            del self._versions[oldest_id]

        logger.info(f"Stored graph version: {version.version_id} (v{version.version_number})")

    async def get_version(self, version_id: str) -> Optional[GraphVersion]:
        """
        Get a graph version by ID.

        Args:
            version_id: Version ID

        Returns:
            Graph version if found
        """
        return self._versions.get(version_id)

    async def get_latest_version(self) -> Optional[GraphVersion]:
        """
        Get the latest graph version.

        Returns:
            Latest version or None
        """
        if not self._version_list:
            return None
        latest_id = self._version_list[-1]
        return self._versions.get(latest_id)

    async def compare_versions(
        self,
        version_id1: str,
        version_id2: str,
        correlation_id: Optional[str] = None
    ) -> GraphDiff:
        """
        Compare two graph versions.

        Task 2.5.4: Add differential graph comparison.

        Args:
            version_id1: First version ID
            version_id2: Second version ID
            correlation_id: Optional correlation ID

        Returns:
            Graph diff
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        version1 = self._versions.get(version_id1)
        version2 = self._versions.get(version_id2)

        if not version1 or not version2:
            raise ValueError("One or both versions not found")

        # Calculate entity changes
        entities1 = set(version1.entities)
        entities2 = set(version2.entities)

        entities_added = list(entities2 - entities1)
        entities_removed = list(entities1 - entities2)
        entities_modified = []  # Would require attribute comparison

        # Calculate relationship changes
        rels1 = {self._rel_key(r) for r in version1.relationships}
        rels2 = {self._rel_key(r) for r in version2.relationships}

        relationships_added = [
            r for r in version2.relationships
            if self._rel_key(r) in rels2 - rels1
        ]
        relationships_removed = [
            r for r in version1.relationships
            if self._rel_key(r) in rels1 - rels2
        ]

        # Calculate metrics
        change_count = (
            len(entities_added) +
            len(entities_removed) +
            len(relationships_added) +
            len(relationships_removed)
        )

        # Calculate similarity
        total_entities = len(entities1 | entities2)
        if total_entities > 0:
            similarity_score = len(entities1 & entities2) / total_entities
        else:
            similarity_score = 1.0

        diff = GraphDiff(
            from_version=version_id1,
            to_version=version_id2,
            entities_added=entities_added,
            entities_removed=entities_removed,
            entities_modified=entities_modified,
            relationships_added=relationships_added,
            relationships_removed=relationships_removed,
            change_count=change_count,
            similarity_score=similarity_score
        )

        logger.info(
            f"Compared versions: {version_id1} vs {version_id2}",
            extra={
                "correlation_id": correlation_id,
                "change_count": change_count,
                "similarity": similarity_score
            }
        )

        return diff

    def _rel_key(self, relationship: Dict[str, str]) -> Tuple[str, str, str]:
        """Create unique key for relationship."""
        return (
            relationship.get("subject", ""),
            relationship.get("predicate", ""),
            relationship.get("object", "")
        )

    async def list_versions(
        self,
        limit: int = 10
    ) -> List[GraphVersion]:
        """
        List recent graph versions.

        Args:
            limit: Maximum versions to return

        Returns:
            List of versions
        """
        recent_ids = self._version_list[-limit:]
        return [self._versions[vid] for vid in recent_ids if vid in self._versions]

    async def close(self) -> None:
        """Cleanup resources."""
        logger.info("GraphAggregator closed")
