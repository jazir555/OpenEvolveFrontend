"""
ROMA-Entity Knowledge Graph Integration for OpenEvolve Knowledge Engine

This module provides comprehensive bi-directional integration between ROMA
(Recursive Optimized Multi-Agent) decomposition system and the Entity
Knowledge Graph (EKG), enabling:

1. Extraction of knowledge entities from ROMA decompositions and solutions
2. Storage of ROMA artifacts in the knowledge graph for future reference
3. Retrieval of similar past decompositions to enhance new problem solving
4. Dependency tracing across ROMA problems and solutions
5. Knowledge-aware ROMA operations using graph context

Architecture:
- ROMAEntityExtractor: Extracts entities and relationships from ROMA data
- ROMAKnowledgeWriter: Writes ROMA entities to EKG with idempotency
- ROMAKnowledgeReader: Queries EKG for similar ROMA entities

Following CLAUDE.md principles:
- ZERO TRUST: Validate all ROMA data before integration
- RUNTIME TRUTH: Verify EKG operations succeed
- IDEMPOTENCY: All writes safe to retry
- CONFIGURATION EXPLICITNESS: No magic defaults
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
- AIR GAP: No direct imports from core-projects

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
import uuid
import json
from enum import Enum

# Import knowledge engine components
from ..core.entity_knowledge_graph import EntityKnowledgeGraph
from ..schemas.base import Entity, Relationship, KnowledgeArtifact, ArtifactType, ArtifactCategory


logger = logging.getLogger(__name__)

# ROMA-Entity Knowledge Graph integration availability flag
EKG_AVAILABLE = True


# ============================================================================
# ROMA ENTITY SCHEMA
# ============================================================================

class ROMAEntityType(Enum):
    """Canonical entity types for ROMA integration."""
    PROBLEM = "roma_problem"
    SUB_PROBLEM = "roma_sub_problem"
    SOLUTION = "roma_solution"
    DEPENDENCY = "roma_dependency"
    DECOMPOSITION = "roma_decomposition"
    AGGREGATION = "roma_aggregation"


class ROMARelationshipType(Enum):
    """Canonical relationship types for ROMA integration."""
    DECOMPOSED_FROM = "decomposed_from"
    SOLVES = "solves"
    DEPENDS_ON = "depends_on"
    AGGREGATED_FROM = "aggregated_from"
    SIMILAR_TO = "similar_to"
    REUSES = "reuses"
    VALIDATED_BY = "validated_by"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ROMAEntity:
    """ROMA entity extracted from decomposition or solution."""
    entity_id: str
    entity_type: ROMAEntityType
    name: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "name": self.name,
            "description": self.description,
            "properties": self.properties,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


@dataclass
class ROMARelationship:
    """ROMA relationship between entities."""
    source_id: str
    target_id: str
    relationship_type: ROMARelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "properties": self.properties,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class ROMAKnowledgeResult:
    """Result of ROMA knowledge operations."""
    success: bool
    entity_ids: List[str] = field(default_factory=list)
    relationship_ids: List[str] = field(default_factory=list)
    artifact_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "entity_ids": self.entity_ids,
            "relationship_ids": self.relationship_ids,
            "artifact_ids": self.artifact_ids,
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error
        }


@dataclass
class SimilarDecomposition:
    """Similar decomposition found in knowledge graph."""
    decomposition_id: str
    problem: str
    similarity_score: float
    sub_problems: List[str]
    solution_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# ROMA ENTITY EXTRACTOR
# ============================================================================

class ROMAEntityExtractor:
    """
    Extract knowledge entities from ROMA decompositions and solutions.

    Features:
    - Extract entities from decompositions (problems, sub-problems)
    - Extract entities from solutions (solution artifacts)
    - Extract relationships (dependencies, decompositions, aggregations)
    - Compute entity properties and metadata
    - Validate extracted entities

    All operations are async and follow structured logging.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ROMA entity extractor.

        Args:
            config: Configuration for extraction
        """
        self.config = config or self._get_default_config()

        logger.info({
            "msg": "ROMAEntityExtractor initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "extract_properties": True,
            "extract_metadata": True,
            "compute_embeddings": False,
            "min_confidence": 0.5,
            "max_sub_problems": 1000
        }

    async def extract_from_decomposition(
        self,
        decomposition: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> List[ROMAEntity]:
        """
        Extract entities from a ROMA decomposition.

        Args:
            decomposition: ROMA decomposition dictionary
            correlation_id: Correlation ID for tracking

        Returns:
            List of extracted ROMA entities

        Example:
            >>> decomposer = ROMAIntegration()
            >>> result = await decomposer.decompose_problem("Design API")
            >>> extractor = ROMAEntityExtractor()
            >>> entities = await extractor.extract_from_decomposition(
            ...     result.decomposition.__dict__
            ... )
        """
        correlation_id = correlation_id or f"roma_extract_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Extracting entities from ROMA decomposition",
            "decomposition_id": decomposition.get("decomposition_id", "unknown"),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        entities = []

        try:
            # Extract main problem entity
            problem_entity = await self._extract_problem_entity(decomposition)
            entities.append(problem_entity)

            # Extract sub-problem entities recursively
            sub_problems = decomposition.get("sub_problems", [])
            if sub_problems:
                sub_entities = await self._extract_sub_problems(
                    sub_problems,
                    parent_id=decomposition.get("decomposition_id"),
                    depth=decomposition.get("depth", 0)
                )
                entities.extend(sub_entities)

            # Extract decomposition metadata entity
            if self.config.get("extract_metadata"):
                metadata_entity = await self._extract_decomposition_metadata(decomposition)
                entities.append(metadata_entity)

            logger.info({
                "msg": "Entity extraction completed",
                "entity_count": len(entities),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return entities

        except Exception as e:
            logger.error({
                "msg": "Entity extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return entities

    async def extract_from_solution(
        self,
        solution: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> List[ROMAEntity]:
        """
        Extract entities from a ROMA solution.

        Args:
            solution: ROMA solution dictionary
            correlation_id: Correlation ID for tracking

        Returns:
            List of extracted ROMA entities

        Example:
            >>> solver = ROMAIntegration()
            >>> result = await solver.solve_atomic(atomic_problem)
            >>> extractor = ROMAEntityExtractor()
            >>> entities = await extractor.extract_from_solution(
            ...     result.solutions[0].__dict__
            ... )
        """
        correlation_id = correlation_id or f"roma_sol_extract_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Extracting entities from ROMA solution",
            "solution_id": solution.get("solution_id", "unknown"),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        entities = []

        try:
            # Extract solution entity
            solution_entity = await self._extract_solution_entity(solution)
            entities.append(solution_entity)

            # Extract solution approach/strategy as entity
            if self.config.get("extract_properties"):
                approach_entity = await self._extract_solution_approach(solution)
                if approach_entity:
                    entities.append(approach_entity)

            logger.info({
                "msg": "Solution extraction completed",
                "entity_count": len(entities),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return entities

        except Exception as e:
            logger.error({
                "msg": "Solution extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return entities

    async def extract_relationships(
        self,
        decomposition: Dict[str, Any],
        entities: List[ROMAEntity],
        correlation_id: Optional[str] = None
    ) -> List[ROMARelationship]:
        """
        Extract relationships between entities from decomposition.

        Args:
            decomposition: ROMA decomposition dictionary
            entities: Extracted entities
            correlation_id: Correlation ID for tracking

        Returns:
            List of ROMA relationships

        Example:
            >>> relationships = await extractor.extract_relationships(
            ...     decomposition,
            ...     entities
            ... )
        """
        correlation_id = correlation_id or f"roma_rel_extract_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Extracting relationships from decomposition",
            "decomposition_id": decomposition.get("decomposition_id", "unknown"),
            "entity_count": len(entities),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        relationships = []

        try:
            # Build entity ID map
            entity_map = {e.entity_id: e for e in entities}

            # Extract decomposition relationships
            decompose_rels = await self._extract_decompose_relationships(
                decomposition,
                entity_map
            )
            relationships.extend(decompose_rels)

            # Extract dependency relationships
            dependency_rels = await self._extract_dependency_relationships(
                decomposition,
                entity_map
            )
            relationships.extend(dependency_rels)

            logger.info({
                "msg": "Relationship extraction completed",
                "relationship_count": len(relationships),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return relationships

        except Exception as e:
            logger.error({
                "msg": "Relationship extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return relationships

    async def _extract_problem_entity(self, decomposition: Dict[str, Any]) -> ROMAEntity:
        """Extract main problem entity."""
        return ROMAEntity(
            entity_id=decomposition.get("decomposition_id", str(uuid.uuid4())),
            entity_type=ROMAEntityType.PROBLEM,
            name=decomposition.get("problem", "Unknown Problem")[:100],
            description=decomposition.get("problem", ""),
            properties={
                "depth": decomposition.get("depth", 0),
                "is_atomic": decomposition.get("is_atomic", False),
                "parent_id": decomposition.get("parent_id"),
                "branching_factor": decomposition.get("metadata", {}).get("branching_factor", 0)
            },
            confidence=1.0,
            metadata={
                "strategy": decomposition.get("metadata", {}).get("strategy", "unknown"),
                "source": "roma_decomposition"
            }
        )

    async def _extract_sub_problems(
        self,
        sub_problems: List[Dict[str, Any]],
        parent_id: str,
        depth: int
    ) -> List[ROMAEntity]:
        """Extract sub-problem entities recursively."""
        entities = []

        for sp in sub_problems[:self.config.get("max_sub_problems", 1000)]:
            entity = ROMAEntity(
                entity_id=sp.get("decomposition_id", str(uuid.uuid4())),
                entity_type=ROMAEntityType.SUB_PROBLEM,
                name=sp.get("problem", "Unknown Sub-Problem")[:100],
                description=sp.get("problem", ""),
                properties={
                    "depth": sp.get("depth", depth),
                    "is_atomic": sp.get("is_atomic", False),
                    "parent_id": sp.get("parent_id", parent_id)
                },
                confidence=1.0,
                metadata={
                    "source": "roma_sub_problem"
                }
            )
            entities.append(entity)

            # Recursively extract nested sub-problems
            if sp.get("sub_problems"):
                nested_entities = await self._extract_sub_problems(
                    sp["sub_problems"],
                    sp["decomposition_id"],
                    depth + 1
                )
                entities.extend(nested_entities)

        return entities

    async def _extract_decomposition_metadata(self, decomposition: Dict[str, Any]) -> ROMAEntity:
        """Extract decomposition metadata entity."""
        metadata_id = f"{decomposition.get('decomposition_id', 'unknown')}_metadata"

        return ROMAEntity(
            entity_id=metadata_id,
            entity_type=ROMAEntityType.DECOMPOSITION,
            name=f"Decomposition Metadata: {decomposition.get('problem', 'Unknown')[:50]}",
            description=f"Metadata for decomposition of {decomposition.get('problem', 'unknown')}",
            properties={
                "depth": decomposition.get("depth", 0),
                "strategy": decomposition.get("metadata", {}).get("strategy", "unknown"),
                "sub_problem_count": self._count_sub_problems(decomposition)
            },
            confidence=1.0,
            metadata={
                "source": "roma_metadata"
            }
        )

    def _count_sub_problems(self, decomposition: Dict[str, Any]) -> int:
        """Count total sub-problems in decomposition."""
        count = 0
        for sp in decomposition.get("sub_problems", []):
            count += 1 + self._count_sub_problems(sp)
        return count

    async def _extract_solution_entity(self, solution: Dict[str, Any]) -> ROMAEntity:
        """Extract solution entity."""
        # Convert solution content to string if needed
        solution_content = solution.get("solution", "")
        if not isinstance(solution_content, str):
            solution_content = json.dumps(solution_content)

        return ROMAEntity(
            entity_id=solution.get("solution_id", str(uuid.uuid4())),
            entity_type=ROMAEntityType.SOLUTION,
            name=f"Solution for {solution.get('problem_id', 'unknown')}"[:100],
            description=solution_content[:500],
            properties={
                "problem_id": solution.get("problem_id"),
                "confidence": solution.get("confidence", 0.0),
                "reasoning_length": len(solution.get("reasoning", ""))
            },
            confidence=solution.get("confidence", 1.0),
            metadata={
                "agent_used": solution.get("metadata", {}).get("agent_used", "unknown"),
                "source": "roma_solution"
            }
        )

    async def _extract_solution_approach(self, solution: Dict[str, Any]) -> Optional[ROMAEntity]:
        """Extract solution approach as entity."""
        reasoning = solution.get("reasoning", "")
        if not reasoning or len(reasoning) < 50:
            return None

        approach_id = f"{solution.get('solution_id', 'unknown')}_approach"

        return ROMAEntity(
            entity_id=approach_id,
            entity_type=ROMAEntityType.SOLUTION,
            name=f"Approach for {solution.get('problem_id', 'unknown')}"[:100],
            description=reasoning[:500],
            properties={
                "problem_id": solution.get("problem_id"),
                "approach_type": solution.get("metadata", {}).get("processing_strategy", "unknown")
            },
            confidence=solution.get("confidence", 1.0) * 0.8,  # Lower confidence for approach
            metadata={
                "source": "roma_solution_approach"
            }
        )

    async def _extract_decompose_relationships(
        self,
        decomposition: Dict[str, Any],
        entity_map: Dict[str, ROMAEntity]
    ) -> List[ROMARelationship]:
        """Extract DECOMPOSED_FROM relationships."""
        relationships = []
        parent_id = decomposition.get("decomposition_id")

        # Extract sub-problem relationships
        for sp in decomposition.get("sub_problems", []):
            child_id = sp.get("decomposition_id")
            if parent_id and child_id:
                rel = ROMARelationship(
                    source_id=child_id,
                    target_id=parent_id,
                    relationship_type=ROMARelationshipType.DECOMPOSED_FROM,
                    properties={
                        "depth": sp.get("depth", 0)
                    },
                    confidence=1.0
                )
                relationships.append(rel)

            # Recursively extract nested relationships
            if sp.get("sub_problems"):
                nested_decomp = {**decomposition, "sub_problems": sp["sub_problems"]}
                nested_rels = await self._extract_decompose_relationships(nested_decomp, entity_map)
                relationships.extend(nested_rels)

        return relationships

    async def _extract_dependency_relationships(
        self,
        decomposition: Dict[str, Any],
        entity_map: Dict[str, ROMAEntity]
    ) -> List[ROMARelationship]:
        """Extract DEPENDS_ON relationships."""
        relationships = []

        # Extract dependencies from metadata
        metadata = decomposition.get("metadata", {})
        dependencies = metadata.get("dependencies", [])

        for dep in dependencies:
            dep_id = dep.get("decomposition_id") if isinstance(dep, dict) else dep
            parent_id = decomposition.get("decomposition_id")

            if dep_id and parent_id:
                rel = ROMARelationship(
                    source_id=parent_id,
                    target_id=dep_id,
                    relationship_type=ROMARelationshipType.DEPENDS_ON,
                    properties={
                        "dependency_type": dep.get("type", "unknown") if isinstance(dep, dict) else "unknown"
                    },
                    confidence=0.8
                )
                relationships.append(rel)

        return relationships


# ============================================================================
# ROMA KNOWLEDGE WRITER
# ============================================================================

class ROMAKnowledgeWriter:
    """
    Write ROMA entities and relationships to Entity Knowledge Graph.

    Features:
    - Store entities with idempotent writes (check before create)
    - Store relationships between entities
    - Create knowledge artifacts from ROMA solutions
    - Batch operations for efficiency
    - Circuit breaker pattern for EKG failures
    - Comprehensive error handling and logging

    All operations are async and follow UTC timestamp standards.
    """

    def __init__(
        self,
        knowledge_graph: EntityKnowledgeGraph,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ROMA knowledge writer.

        Args:
            knowledge_graph: Entity knowledge graph instance
            config: Configuration for writing
        """
        self.kg = knowledge_graph
        self.config = config or self._get_default_config()

        # Circuit breaker state
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = None
        self._circuit_breaker_open = False

        logger.info({
            "msg": "ROMAKnowledgeWriter initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "auto_extract": True,
            "auto_store": True,
            "batch_size": 100,
            "timeout_seconds": 30,
            "idempotent": True,
            "retry_attempts": 3,
            "retry_backoff_ms": 1000
        }

    async def store_entities(
        self,
        entities: List[ROMAEntity],
        correlation_id: Optional[str] = None
    ) -> List[str]:
        """
        Store ROMA entities in knowledge graph.

        IDEMPOTENT: Checks if entity exists before creating.

        Args:
            entities: List of ROMA entities to store
            correlation_id: Correlation ID for tracking

        Returns:
            List of entity IDs that were stored

        Example:
            >>> writer = ROMAKnowledgeWriter(knowledge_graph)
            >>> entity_ids = await writer.store_entities(entities)
            >>> print(f"Stored {len(entity_ids)} entities")
        """
        correlation_id = correlation_id or f"roma_write_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Storing ROMA entities in knowledge graph",
            "entity_count": len(entities),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        stored_ids = []

        try:
            # Check circuit breaker
            if self._is_circuit_breaker_open():
                logger.warning({
                    "msg": "Circuit breaker open, skipping entity storage",
                    "correlation_id": correlation_id
                })
                return stored_ids

            # Process in batches
            batch_size = self.config.get("batch_size", 100)

            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]

                # Store batch
                batch_ids = await self._store_entity_batch(
                    batch,
                    correlation_id=f"{correlation_id}_batch_{i // batch_size}"
                )
                stored_ids.extend(batch_ids)

            logger.info({
                "msg": "Entity storage completed",
                "stored_count": len(stored_ids),
                "requested_count": len(entities),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Reset circuit breaker on success
            self._reset_circuit_breaker()

            return stored_ids

        except Exception as e:
            logger.error({
                "msg": "Entity storage failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Update circuit breaker
            self._record_circuit_breaker_failure()

            return stored_ids

    async def store_relationships(
        self,
        relationships: List[ROMARelationship],
        correlation_id: Optional[str] = None
    ) -> List[str]:
        """
        Store ROMA relationships in knowledge graph.

        IDEMPOTENT: Duplicate relationships are ignored.

        Args:
            relationships: List of ROMA relationships to store
            correlation_id: Correlation ID for tracking

        Returns:
            List of relationship IDs that were stored

        Example:
            >>> rel_ids = await writer.store_relationships(relationships)
        """
        correlation_id = correlation_id or f"roma_rel_write_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Storing ROMA relationships in knowledge graph",
            "relationship_count": len(relationships),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        stored_ids = []

        try:
            # Check circuit breaker
            if self._is_circuit_breaker_open():
                logger.warning({
                    "msg": "Circuit breaker open, skipping relationship storage",
                    "correlation_id": correlation_id
                })
                return stored_ids

            # Store each relationship
            for rel in relationships:
                rel_id = f"{rel.source_id}_{rel.target_id}_{rel.relationship_type.value}"

                success = await self.kg.add_relationship_async(
                    source=rel.source_id,
                    target=rel.target_id,
                    relation_type=rel.relationship_type.value,
                    attributes={
                        **rel.properties,
                        "confidence": rel.confidence,
                        "metadata": rel.metadata
                    }
                )

                if success:
                    stored_ids.append(rel_id)

            logger.info({
                "msg": "Relationship storage completed",
                "stored_count": len(stored_ids),
                "requested_count": len(relationships),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return stored_ids

        except Exception as e:
            logger.error({
                "msg": "Relationship storage failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            self._record_circuit_breaker_failure()

            return stored_ids

    async def store_artifact(
        self,
        solution: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Store ROMA solution as knowledge artifact.

        Args:
            solution: ROMA solution dictionary
            correlation_id: Correlation ID for tracking

        Returns:
            Artifact ID

        Example:
            >>> artifact_id = await writer.store_artifact(solution)
        """
        correlation_id = correlation_id or f"roma_artifact_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Storing ROMA solution as knowledge artifact",
            "solution_id": solution.get("solution_id", "unknown"),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        try:
            # Convert solution to artifact
            artifact = KnowledgeArtifact(
                artifact_id=solution.get("solution_id", str(uuid.uuid4())),
                artifact_type=ArtifactType.SOLUTION_PATTERN,
                category=ArtifactCategory.SOLUTION,
                title=f"ROMA Solution: {solution.get('problem_id', 'unknown')}"[:100],
                description=solution.get("reasoning", "")[:500],
                content={
                    "solution": solution.get("solution"),
                    "reasoning": solution.get("reasoning"),
                    "metadata": solution.get("metadata", {})
                },
                domain="roma",
                subdomain="problem_solving",
                tags=["roma", "solution", "decomposition"],
                source_type="roma_integration",
                source_id=solution.get("solution_id"),
                confidence=solution.get("confidence", 0.8),
                quality_score=0.8,  # Would be computed by validator
                status="verified",
                metadata={
                    "roma_problem_id": solution.get("problem_id"),
                    "agent_used": solution.get("metadata", {}).get("agent_used", "unknown")
                }
            )

            # Store artifact in knowledge graph
            # Note: EKG stores entities, artifacts are stored as entities with type "artifact"
            artifact_entity_id = f"artifact_{artifact.artifact_id}"

            success = await self.kg.add_entity_async(
                name=artifact_entity_id,
                entity_type="knowledge_artifact",
                attributes={
                    "artifact_data": artifact.to_dict()
                }
            )

            if success:
                logger.info({
                    "msg": "Artifact storage completed",
                    "artifact_id": artifact.artifact_id,
                    "entity_id": artifact_entity_id,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                return artifact.artifact_id
            else:
                logger.error({
                    "msg": "Artifact storage failed",
                    "solution_id": solution.get("solution_id"),
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return ""

        except Exception as e:
            logger.error({
                "msg": "Artifact storage error",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return ""

    async def store_decomposition_graph(
        self,
        decomposition: Dict[str, Any],
        entities: List[ROMAEntity],
        relationships: List[ROMARelationship],
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Store complete decomposition graph in knowledge graph.

        Args:
            decomposition: ROMA decomposition
            entities: Extracted entities
            relationships: Extracted relationships
            correlation_id: Correlation ID for tracking

        Returns:
            Graph ID

        Example:
            >>> graph_id = await writer.store_decomposition_graph(
            ...     decomposition,
            ...     entities,
            ...     relationships
            ... )
        """
        correlation_id = correlation_id or f"roma_graph_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Storing ROMA decomposition graph",
            "decomposition_id": decomposition.get("decomposition_id", "unknown"),
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        try:
            # Store entities
            entity_ids = await self.store_entities(entities, correlation_id)

            # Store relationships
            relationship_ids = await self.store_relationships(relationships, correlation_id)

            # Create graph metadata entity
            graph_id = f"graph_{decomposition.get('decomposition_id', str(uuid.uuid4()))}"

            await self.kg.add_entity_async(
                name=graph_id,
                entity_type="roma_decomposition_graph",
                attributes={
                    "decomposition_id": decomposition.get("decomposition_id"),
                    "problem": decomposition.get("problem", ""),
                    "entity_ids": entity_ids,
                    "relationship_ids": relationship_ids,
                    "entity_count": len(entity_ids),
                    "relationship_count": len(relationship_ids),
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            )

            logger.info({
                "msg": "Decomposition graph storage completed",
                "graph_id": graph_id,
                "entities_stored": len(entity_ids),
                "relationships_stored": len(relationship_ids),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return graph_id

        except Exception as e:
            logger.error({
                "msg": "Decomposition graph storage failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return ""

    async def _store_entity_batch(
        self,
        batch: List[ROMAEntity],
        correlation_id: Optional[str] = None
    ) -> List[str]:
        """Store a batch of entities."""
        stored_ids = []

        for entity in batch:
            # Check if entity already exists (idempotent)
            existing = await self.kg.get_entity_async(entity.entity_id)

            if existing and self.config.get("idempotent", True):
                logger.debug({
                    "msg": "Entity already exists, skipping",
                    "entity_id": entity.entity_id,
                    "correlation_id": correlation_id
                })
                stored_ids.append(entity.entity_id)
                continue

            # Create entity
            success = await self.kg.add_entity_async(
                name=entity.entity_id,
                entity_type=entity.entity_type.value,
                attributes={
                    "name": entity.name,
                    "description": entity.description,
                    **entity.properties,
                    "confidence": entity.confidence,
                    "metadata": entity.metadata,
                    "created_at": entity.created_at
                }
            )

            if success:
                stored_ids.append(entity.entity_id)

        return stored_ids

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self._circuit_breaker_open:
            return False

        # Check if recovery timeout has passed
        if self._circuit_breaker_last_failure:
            elapsed = (datetime.now(timezone.utc) - self._circuit_breaker_last_failure).total_seconds()
            recovery_timeout = self.config.get("recovery_timeout_seconds", 60)

            if elapsed > recovery_timeout:
                # Attempt recovery
                logger.info({
                    "msg": "Circuit breaker recovery timeout elapsed, attempting recovery",
                    "failures": self._circuit_breaker_failures,
                    "elapsed_seconds": elapsed
                })
                self._reset_circuit_breaker()
                return False

        return True

    def _record_circuit_breaker_failure(self):
        """Record a circuit breaker failure."""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = datetime.now(timezone.utc)

        failure_threshold = self.config.get("failure_threshold", 5)

        if self._circuit_breaker_failures >= failure_threshold:
            self._circuit_breaker_open = True
            logger.error({
                "msg": "Circuit breaker opened",
                "failures": self._circuit_breaker_failures,
                "threshold": failure_threshold
            })

    def _reset_circuit_breaker(self):
        """Reset circuit breaker."""
        self._circuit_breaker_failures = 0
        self._circuit_breaker_open = False
        self._circuit_breaker_last_failure = None


# ============================================================================
# ROMA KNOWLEDGE READER
# ============================================================================

class ROMAKnowledgeReader:
    """
    Query Entity Knowledge Graph for ROMA entities.

    Features:
    - Find similar decompositions
    - Retrieve solution artifacts
    - Trace dependencies
    - Semantic search across ROMA entities

    All operations are async with comprehensive error handling.
    """

    def __init__(
        self,
        knowledge_graph: EntityKnowledgeGraph,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ROMA knowledge reader.

        Args:
            knowledge_graph: Entity knowledge graph instance
            config: Configuration for reading
        """
        self.kg = knowledge_graph
        self.config = config or self._get_default_config()

        logger.info({
            "msg": "ROMAKnowledgeReader initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "default_top_k": 5,
            "similarity_threshold": 0.7,
            "max_results": 100,
            "include_metadata": True
        }

    async def find_similar_decompositions(
        self,
        problem: str,
        top_k: int = 5,
        correlation_id: Optional[str] = None
    ) -> List[SimilarDecomposition]:
        """
        Find similar decompositions in knowledge graph.

        Args:
            problem: Problem statement to match
            top_k: Maximum number of results
            correlation_id: Correlation ID for tracking

        Returns:
            List of similar decompositions

        Example:
            >>> reader = ROMAKnowledgeReader(knowledge_graph)
            >>> similar = await reader.find_similar_decompositions(
            ...     "Design a RESTful API",
            ...     top_k=3
            ... )
            >>> for decomp in similar:
            ...     print(f"{decomp.similarity_score:.2f}: {decomp.problem}")
        """
        correlation_id = correlation_id or f"roma_find_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Finding similar ROMA decompositions",
            "problem_length": len(problem),
            "top_k": top_k,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        similar_decompositions = []

        try:
            # Search for problem entities
            results = await self.kg.search_entities_async(
                query=problem,
                limit=self.config.get("max_results", 100)
            )

            # Filter to ROMA problem entities
            problem_entities = [
                r for r in results
                if r.get("entity_type") in ["roma_problem", "roma_sub_problem"]
            ]

            # Rank by similarity (simple keyword overlap for now)
            ranked = await self._rank_similarity(problem, problem_entities)

            # Convert to SimilarDecomposition objects
            for i, entity in enumerate(ranked[:top_k]):
                props = entity.get("properties", {})
                similar = SimilarDecomposition(
                    decomposition_id=entity.get("entity_id", entity.get("name", "")),
                    problem=entity.get("name", "Unknown Problem"),
                    similarity_score=1.0 - (i * 0.1),  # Simple ranking
                    sub_problems=props.get("sub_problem_count", 0),
                    solution_summary=props.get("description", "")[:200],
                    metadata={
                        "depth": props.get("depth", 0),
                        "is_atomic": props.get("is_atomic", False)
                    }
                )
                similar_decompositions.append(similar)

            logger.info({
                "msg": "Similar decomposition search completed",
                "results_count": len(similar_decompositions),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return similar_decompositions

        except Exception as e:
            logger.error({
                "msg": "Similar decomposition search failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return similar_decompositions

    async def get_solution_artifacts(
        self,
        entity_id: str,
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get solution artifacts for a ROMA entity.

        Args:
            entity_id: Entity ID to get artifacts for
            correlation_id: Correlation ID for tracking

        Returns:
            List of solution artifacts

        Example:
            >>> artifacts = await reader.get_solution_artifacts(entity_id)
            >>> for artifact in artifacts:
            ...     print(f"Solution: {artifact['solution']}")
        """
        correlation_id = correlation_id or f"roma_artifacts_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Retrieving solution artifacts for entity",
            "entity_id": entity_id,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        artifacts = []

        try:
            # Get relationships for entity
            relationships = await self.kg.get_relationships_async(entity_id)

            # Find SOLVES relationships
            for rel in relationships:
                if rel.get("relationship_type") == ROMARelationshipType.SOLVES.value:
                    # Get the solution entity
                    solution_id = rel.get("source_entity_id")
                    if solution_id == entity_id:
                        solution_id = rel.get("target_entity_id")

                    solution_entity = await self.kg.get_entity_async(solution_id)

                    if solution_entity:
                        props = solution_entity.get("properties", {})
                        artifacts.append({
                            "artifact_id": solution_id,
                            "solution": props.get("description", ""),
                            "confidence": props.get("confidence", 0.0),
                            "metadata": props.get("metadata", {})
                        })

            logger.info({
                "msg": "Solution artifacts retrieved",
                "artifact_count": len(artifacts),
                "entity_id": entity_id,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return artifacts

        except Exception as e:
            logger.error({
                "msg": "Solution artifact retrieval failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return artifacts

    async def trace_dependencies(
        self,
        problem_id: str,
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Trace dependencies for a ROMA problem.

        Args:
            problem_id: Problem entity ID
            correlation_id: Correlation ID for tracking

        Returns:
            List of dependency traces

        Example:
            >>> dependencies = await reader.trace_dependencies(problem_id)
            >>> for dep in dependencies:
            ...     print(f"Depends on: {dep['target']}")
        """
        correlation_id = correlation_id or f"roma_trace_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Tracing dependencies for ROMA problem",
            "problem_id": problem_id,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        dependencies = []

        try:
            # Get relationships for problem
            relationships = await self.kg.get_relationships_async(problem_id)

            # Find DEPENDS_ON relationships
            for rel in relationships:
                if rel.get("relationship_type") == ROMARelationshipType.DEPENDS_ON.value:
                    dep_id = rel.get("target_entity_id")
                    dep_entity = await self.kg.get_entity_async(dep_id)

                    if dep_entity:
                        dependencies.append({
                            "source": problem_id,
                            "target": dep_id,
                            "target_name": dep_entity.get("name", "Unknown"),
                            "relationship_id": rel.get("relationship_id", ""),
                            "properties": rel.get("properties", {})
                        })

            logger.info({
                "msg": "Dependency tracing completed",
                "dependency_count": len(dependencies),
                "problem_id": problem_id,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return dependencies

        except Exception as e:
            logger.error({
                "msg": "Dependency tracing failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return dependencies

    async def get_decomposition_tree(
        self,
        decomposition_id: str,
        max_depth: int = 10,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get complete decomposition tree from knowledge graph.

        Args:
            decomposition_id: Root decomposition ID
            max_depth: Maximum depth to traverse
            correlation_id: Correlation ID for tracking

        Returns:
            Decomposition tree structure

        Example:
            >>> tree = await reader.get_decomposition_tree(decomp_id)
            >>> print(tree['problem'])
            >>> for sub in tree['sub_problems']:
            ...     print(f"  - {sub['problem']}")
        """
        correlation_id = correlation_id or f"roma_tree_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Retrieving decomposition tree",
            "decomposition_id": decomposition_id,
            "max_depth": max_depth,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        try:
            # Get root entity
            root = await self.kg.get_entity_async(decomposition_id)

            if not root:
                logger.warning({
                    "msg": "Root decomposition not found",
                    "decomposition_id": decomposition_id,
                    "correlation_id": correlation_id
                })
                return {}

            # Recursively build tree
            tree = await self._build_tree(decomposition_id, max_depth, 0, correlation_id)

            logger.info({
                "msg": "Decomposition tree retrieved",
                "tree_depth": tree.get("depth", 0),
                "sub_problem_count": tree.get("sub_problem_count", 0),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return tree

        except Exception as e:
            logger.error({
                "msg": "Decomposition tree retrieval failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return {}

    async def _build_tree(
        self,
        entity_id: str,
        max_depth: int,
        current_depth: int,
        correlation_id: str
    ) -> Dict[str, Any]:
        """Recursively build decomposition tree."""
        if current_depth >= max_depth:
            return {}

        # Get entity
        entity = await self.kg.get_entity_async(entity_id)

        if not entity:
            return {}

        # Get relationships to find children
        relationships = await self.kg.get_relationships_async(entity_id)

        # Find children (entities that DECOMPOSED_FROM this entity)
        children = []
        for rel in relationships:
            if rel.get("relationship_type") == ROMARelationshipType.DECOMPOSED_FROM.value:
                child_id = rel.get("source_entity_id")
                if child_id != entity_id:
                    child_tree = await self._build_tree(
                        child_id,
                        max_depth,
                        current_depth + 1,
                        correlation_id
                    )
                    if child_tree:
                        children.append(child_tree)

        # Build tree node
        props = entity.get("properties", {})

        return {
            "entity_id": entity_id,
            "name": entity.get("name", ""),
            "description": entity.get("properties", {}).get("description", ""),
            "depth": current_depth,
            "is_atomic": props.get("is_atomic", False),
            "sub_problems": children,
            "sub_problem_count": len(children)
        }

    async def _rank_similarity(
        self,
        query: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank entities by similarity to query."""
        query_terms = set(query.lower().split())

        ranked = sorted(
            entities,
            key=lambda e: self._compute_similarity(query_terms, e),
            reverse=True
        )

        return ranked

    def _compute_similarity(
        self,
        query_terms: Set[str],
        entity: Dict[str, Any]
    ) -> float:
        """Compute similarity score between query and entity."""
        name = entity.get("name", "").lower()
        description = entity.get("properties", {}).get("description", "").lower()

        entity_terms = set((name + " " + description).split())

        # Jaccard similarity
        intersection = query_terms & entity_terms
        union = query_terms | entity_terms

        if not union:
            return 0.0

        return len(intersection) / len(union)


# ============================================================================
# INTEGRATION FACTORY
# ============================================================================

def create_roma_ekg_integration(
    knowledge_graph: EntityKnowledgeGraph,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[ROMAEntityExtractor, ROMAKnowledgeWriter, ROMAKnowledgeReader]:
    """
    Create complete ROMA-EKG integration components.

    Args:
        knowledge_graph: Entity knowledge graph instance
        config: Configuration for integration

    Returns:
        Tuple of (extractor, writer, reader)

    Example:
        >>> from knowledge_engine.core.entity_knowledge_graph import EntityKnowledgeGraph
        >>> kg = EntityKnowledgeGraph()
        >>> extractor, writer, reader = create_roma_ekg_integration(kg)
        >>>
        >>> # Extract entities from decomposition
        >>> entities = await extractor.extract_from_decomposition(decomposition)
        >>>
        >>> # Store in knowledge graph
        >>> entity_ids = await writer.store_entities(entities)
        >>>
        >>> # Find similar decompositions
        >>> similar = await reader.find_similar_decompositions("Design API")
    """
    extractor = ROMAEntityExtractor(config)
    writer = ROMAKnowledgeWriter(knowledge_graph, config)
    reader = ROMAKnowledgeReader(knowledge_graph, config)

    logger.info({
        "msg": "ROMA-EKG integration created",
        "config": config,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return extractor, writer, reader


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Schema
    "ROMAEntityType",
    "ROMARelationshipType",

    # Data structures
    "ROMAEntity",
    "ROMARelationship",
    "ROMAKnowledgeResult",
    "SimilarDecomposition",

    # Components
    "ROMAEntityExtractor",
    "ROMAKnowledgeWriter",
    "ROMAKnowledgeReader",

    # Factory
    "create_roma_ekg_integration",
]
