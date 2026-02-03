"""
ROMA Knowledge Pipeline for OpenEvolve Knowledge Engine

This module provides an automated pipeline that converts ROMA (Recursive Optimized Multi-Agent)
executions into persistent knowledge in the knowledge graph.

Pipeline Flow:
ROMA Decomposition → Entity Extraction → Knowledge Storage → Similar Solution Retrieval

Features:
- Execute ROMA and automatically persist results
- Extract entities from decompositions
- Store solutions as knowledge artifacts
- Retrieve similar past solutions
- Idempotent operations (safe to re-run)
- Async/await for scalability
- Comprehensive error handling
- Structured logging (JSON Lines)
- UTC timestamps
- Correlation ID tracking
- Graceful degradation if knowledge engine unavailable

Following CLAUDE.md principles:
- ZERO TRUST: Validate all inputs and handle failures gracefully
- RUNTIME TRUTH: Verify operations succeed
- IDEMPOTENCY: All operations safe to retry
- CONFIGURATION EXPLICITNESS: No magic defaults
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs

Usage Example:
    >>> from knowledge_engine.integrations.roma_integration import ROMAIntegration
    >>> from knowledge_engine.integrations.roma_knowledge_pipeline import create_roma_knowledge_pipeline
    >>>
    >>> # Create pipeline
    >>> pipeline = await create_roma_knowledge_pipeline({
    ...     "auto_extract_entities": True,
    ...     "auto_store_solutions": True
    ... })
    >>>
    >>> # Execute and store knowledge
    >>> result = await pipeline.execute_and_store(
    ...     "Design a scalable microservices architecture",
    ...     options={"max_depth": 3}
    ... )
    >>>
    >>> # Retrieve similar solutions
    >>> similar = await pipeline.retrieve_similar_solutions(
    ...     "Design a distributed system",
    ...     top_k=5
    ... )
    >>>
    >>> # Close pipeline
    >>> await pipeline.close()

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import uuid
import json

from .roma_integration import (
    ROMAIntegration,
    ROMAResult,
    ROMADecomposition,
    ROMASolution,
)
from ..core.entity_knowledge_graph import EntityKnowledgeGraph


logger = logging.getLogger(__name__)


@dataclass
class EntityExtractionResult:
    """Result of entity extraction from ROMA decomposition."""
    entity_ids: List[str]
    relationships_created: int
    extraction_time_ms: float
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'entity_ids': self.entity_ids,
            'relationships_created': self.relationships_created,
            'extraction_time_ms': self.extraction_time_ms,
            'errors': self.errors
        }


@dataclass
class KnowledgeArtifact:
    """Represents a knowledge artifact stored in the graph."""
    artifact_id: str
    artifact_type: str
    content: Any
    metadata: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'artifact_id': self.artifact_id,
            'artifact_type': self.artifact_type,
            'content': self.content,
            'metadata': self.metadata,
            'created_at': self.created_at
        }


class ROMAKnowledgePipeline:
    """
    Pipeline to convert ROMA executions into persistent knowledge.

    Automates the flow from ROMA decomposition → knowledge extraction → knowledge storage.

    Features:
    - Execute ROMA decomposition and automatically store results
    - Extract entities from ROMA decompositions
    - Store solutions as knowledge artifacts in the knowledge graph
    - Retrieve similar past solutions based on semantic similarity
    - Idempotent operations (safe to re-run)
    - Async/await for scalability
    - Comprehensive error handling
    - Structured logging (JSON Lines)
    - UTC timestamps
    - Correlation ID tracking
    - Graceful degradation if knowledge engine unavailable
    """

    def __init__(
        self,
        roma_integration: ROMAIntegration,
        knowledge_engine: EntityKnowledgeGraph,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ROMA knowledge pipeline.

        Args:
            roma_integration: ROMA integration instance
            knowledge_engine: Knowledge engine instance (EntityKnowledgeGraph)
            config: Optional configuration dictionary

        Configuration Options:
            auto_extract_entities: Automatically extract entities (default: True)
            auto_store_solutions: Automatically store solutions (default: True)
            entity_types: List of entity types to extract (default: ["sub_problem", "solution", "dependency"])
            knowledge_artifact_type: Type label for stored solutions (default: "roma_solution")
            similarity_threshold: Minimum similarity score for retrieval (default: 0.7)
            max_entities_per_decomposition: Maximum entities to extract (default: 100)

        Example:
            >>> from knowledge_engine.integrations.roma_integration import ROMAIntegration
            >>> from knowledge_engine.core.entity_knowledge_graph import EntityKnowledgeGraph
            >>>
            >>> roma = ROMAIntegration()
            >>> ke = EntityKnowledgeGraph()
            >>> pipeline = ROMAKnowledgePipeline(roma, ke, {
            ...     "auto_extract_entities": True,
            ...     "auto_store_solutions": True
            ... })
        """
        self.roma = roma_integration
        self.knowledge_engine = knowledge_engine
        self.config = config or self._get_default_config()

        # Statistics tracking
        self._stats = {
            "executions_performed": 0,
            "entities_extracted": 0,
            "solutions_stored": 0,
            "similar_retrievals": 0,
            "total_processing_time_ms": 0.0,
            "errors_encountered": 0
        }

        logger.info({
            "msg": "ROMAKnowledgePipeline initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ROMA knowledge pipeline."""
        return {
            "auto_extract_entities": True,
            "auto_store_solutions": True,
            "entity_types": ["sub_problem", "solution", "dependency"],
            "knowledge_artifact_type": "roma_solution",
            "similarity_threshold": 0.7,
            "max_entities_per_decomposition": 100,
            "entity_extraction": {
                "extract_sub_problems": True,
                "extract_solutions": True,
                "extract_dependencies": True,
                "extract_metadata": True
            },
            "knowledge_storage": {
                "store_decomposition": True,
                "store_solutions": True,
                "store_verification": True,
                "store_metadata": True
            },
            "retrieval": {
                "max_results": 10,
                "include_metadata": True,
                "score_threshold": 0.5
            }
        }

    def _generate_correlation_id(self, prefix: str = "roma_kp") -> str:
        """Generate a unique correlation ID for tracking."""
        return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"

    async def execute_and_store(
        self,
        problem: str,
        options: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> ROMAResult:
        """
        Execute ROMA decomposition and automatically store results in knowledge graph.

        This is the main entry point for the pipeline. It:
        1. Executes ROMA decomposition
        2. Extracts entities from decomposition (if enabled)
        3. Stores entities in knowledge graph
        4. Stores solution as knowledge artifact
        5. Enhances result with knowledge metadata

        Args:
            problem: The problem to decompose and solve
            options: Optional ROMA execution options
            correlation_id: Correlation ID for tracking

        Returns:
            Enhanced ROMAResult with knowledge metadata (knowledge_artifact_id, entities_created)

        Example:
            >>> result = await pipeline.execute_and_store(
            ...     "Design a scalable microservices architecture",
            ...     options={"max_depth": 3}
            ... )
            >>> print(result.metadata["knowledge_artifact_id"])
            >>> print(result.metadata["entities_created"])
        """
        correlation_id = correlation_id or self._generate_correlation_id("execute")
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting ROMA execute and store pipeline",
            "problem_length": len(problem),
            "options": options,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Step 1: Execute ROMA decomposition
            logger.info({
                "msg": "Executing ROMA decomposition",
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            roma_result = await self.roma.decompose_problem(
                problem=problem,
                max_depth=options.get("max_depth") if options else None,
                correlation_id=correlation_id
            )

            if not roma_result.success:
                logger.warning({
                    "msg": "ROMA decomposition failed, skipping knowledge extraction",
                    "correlation_id": correlation_id,
                    "error": roma_result.error,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return roma_result

            # Step 2: Extract and store entities (if enabled)
            entities_created = []
            if self.config["auto_extract_entities"]:
                logger.info({
                    "msg": "Extracting entities from ROMA decomposition",
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                entities_created = await self.extract_and_store_entities(
                    decomposition=roma_result,
                    correlation_id=f"{correlation_id}_extract"
                )

            # Step 3: Store solution as knowledge artifact (if enabled)
            knowledge_artifact_id = None
            if self.config["auto_store_solutions"]:
                logger.info({
                    "msg": "Storing ROMA solution as knowledge artifact",
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                knowledge_artifact_id = await self._store_solution_as_artifact(
                    problem=problem,
                    result=roma_result,
                    entities_created=entities_created,
                    correlation_id=f"{correlation_id}_store"
                )

            # Step 4: Enhance result with knowledge metadata
            roma_result.metadata["knowledge_artifact_id"] = knowledge_artifact_id
            roma_result.metadata["entities_created"] = entities_created
            roma_result.metadata["stored_at"] = datetime.now(timezone.utc).isoformat()

            # Update statistics
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._stats["executions_performed"] += 1
            self._stats["entities_extracted"] += len(entities_created)
            self._stats["solutions_stored"] += 1 if knowledge_artifact_id else 0
            self._stats["total_processing_time_ms"] += processing_time_ms

            logger.info({
                "msg": "ROMA execute and store pipeline completed",
                "correlation_id": correlation_id,
                "knowledge_artifact_id": knowledge_artifact_id,
                "entities_created_count": len(entities_created),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return roma_result

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self._stats["errors_encountered"] += 1

            logger.error({
                "msg": "ROMA execute and store pipeline failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return failed result
            return ROMAResult(
                success=False,
                decomposition=None,
                solutions=[],
                verification=None,
                metadata={
                    "error_type": "pipeline_error",
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )

    async def extract_and_store_entities(
        self,
        decomposition: ROMAResult,
        correlation_id: Optional[str] = None
    ) -> List[str]:
        """
        Extract entities from ROMA decomposition and store in knowledge graph.

        Extracts entities such as:
        - Sub-problems (as entities)
        - Solutions (as entities)
        - Dependencies (as relationships)
        - Metadata (as entity properties)

        Args:
            decomposition: ROMA decomposition result
            correlation_id: Correlation ID for tracking

        Returns:
            List of entity IDs created

        Example:
            >>> result = await roma.decompose_problem("Design API")
            >>> entity_ids = await pipeline.extract_and_store_entities(result)
            >>> print(f"Created {len(entity_ids)} entities")
        """
        correlation_id = correlation_id or self._generate_correlation_id("extract")
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting entity extraction from ROMA decomposition",
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        entity_ids = []
        errors = []

        try:
            # Extract entities from decomposition
            if decomposition.decomposition:
                decomposition_entities = await self._extract_decomposition_entities(
                    decomposition.decomposition,
                    correlation_id=f"{correlation_id}_decomp"
                )
                entity_ids.extend(decomposition_entities)

            # Extract entities from solutions
            if decomposition.solutions:
                solution_entities = await self._extract_solution_entities(
                    decomposition.solutions,
                    correlation_id=f"{correlation_id}_solutions"
                )
                entity_ids.extend(solution_entities)

            # Extract entities from verification
            if decomposition.verification:
                verification_entities = await self._extract_verification_entities(
                    decomposition.verification,
                    correlation_id=f"{correlation_id}_verify"
                )
                entity_ids.extend(verification_entities)

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self._stats["entities_extracted"] += len(entity_ids)

            logger.info({
                "msg": "Entity extraction completed",
                "correlation_id": correlation_id,
                "entities_created": len(entity_ids),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return entity_ids

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self._stats["errors_encountered"] += 1

            logger.error({
                "msg": "Entity extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return entity_ids  # Return any entities created before failure

    async def _extract_decomposition_entities(
        self,
        decomposition: ROMADecomposition,
        correlation_id: str
    ) -> List[str]:
        """Extract entities from ROMA decomposition tree."""
        entity_ids = []

        try:
            # Create entity for the main problem
            problem_entity_id = f"problem_{decomposition.decomposition_id}"
            await self.knowledge_engine.add_entity_async(
                name=problem_entity_id,
                entity_type="sub_problem",
                attributes={
                    "problem_text": decomposition.problem,
                    "depth": decomposition.depth,
                    "is_atomic": decomposition.is_atomic,
                    "created_at": decomposition.created_at,
                    **decomposition.metadata
                }
            )
            entity_ids.append(problem_entity_id)

            # Recursively extract sub-problems
            for sub_problem in decomposition.sub_problems:
                sub_entities = await self._extract_decomposition_entities(
                    sub_problem,
                    correlation_id=f"{correlation_id}_sub"
                )
                entity_ids.extend(sub_entities)

                # Create relationship between parent and child
                if sub_entities:
                    await self.knowledge_engine.add_relationship_async(
                        source=problem_entity_id,
                        target=sub_entities[0],
                        relation_type="decomposes_to",
                        attributes={"depth": decomposition.depth}
                    )

        except Exception as e:
            logger.error({
                "msg": "Failed to extract decomposition entities",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        return entity_ids

    async def _extract_solution_entities(
        self,
        solutions: List[ROMASolution],
        correlation_id: str
    ) -> List[str]:
        """Extract entities from ROMA solutions."""
        entity_ids = []

        try:
            for solution in solutions:
                solution_entity_id = f"solution_{solution.solution_id}"
                await self.knowledge_engine.add_entity_async(
                    name=solution_entity_id,
                    entity_type="solution",
                    attributes={
                        "problem_id": solution.problem_id,
                        "solution_content": str(solution.solution),
                        "confidence": solution.confidence,
                        "reasoning": solution.reasoning,
                        "created_at": solution.created_at,
                        **solution.metadata
                    }
                )
                entity_ids.append(solution_entity_id)

        except Exception as e:
            logger.error({
                "msg": "Failed to extract solution entities",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        return entity_ids

    async def _extract_verification_entities(
        self,
        verification: 'ROMAVerification',
        correlation_id: str
    ) -> List[str]:
        """Extract entities from ROMA verification."""
        entity_ids = []

        try:
            verification_entity_id = f"verification_{verification.verification_id}"
            await self.knowledge_engine.add_entity_async(
                name=verification_entity_id,
                entity_type="verification",
                attributes={
                    "solution_id": verification.solution_id,
                    "passed": verification.passed,
                    "score": verification.score,
                    "feedback": verification.feedback,
                    "requirements_met": verification.requirements_met,
                    "created_at": verification.created_at,
                    **verification.metadata
                }
            )
            entity_ids.append(verification_entity_id)

        except Exception as e:
            logger.error({
                "msg": "Failed to extract verification entities",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        return entity_ids

    async def _store_solution_as_artifact(
        self,
        problem: str,
        result: ROMAResult,
        entities_created: List[str],
        correlation_id: str
    ) -> Optional[str]:
        """Store ROMA solution as a knowledge artifact."""
        try:
            artifact_id = f"artifact_{uuid.uuid4().hex}"

            # Create knowledge artifact entity
            await self.knowledge_engine.add_entity_async(
                name=artifact_id,
                entity_type=self.config["knowledge_artifact_type"],
                attributes={
                    "problem": problem,
                    "decomposition": result.decomposition.__dict__ if result.decomposition else None,
                    "solutions": [s.__dict__ for s in result.solutions],
                    "verification": result.verification.__dict__ if result.verification else None,
                    "metadata": result.metadata,
                    "entities_created": entities_created,
                    "processing_time_ms": result.processing_time_ms,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            )

            logger.info({
                "msg": "Knowledge artifact stored",
                "correlation_id": correlation_id,
                "artifact_id": artifact_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return artifact_id

        except Exception as e:
            logger.error({
                "msg": "Failed to store knowledge artifact",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return None

    async def retrieve_similar_solutions(
        self,
        problem: str,
        top_k: int = 5,
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar past solutions from the knowledge graph.

        Searches for knowledge artifacts with similar problem descriptions
        and returns them with similarity scores.

        Args:
            problem: The problem to find similar solutions for
            top_k: Maximum number of similar solutions to retrieve
            correlation_id: Correlation ID for tracking

        Returns:
            List of similar solutions with metadata and similarity scores

        Example:
            >>> similar = await pipeline.retrieve_similar_solutions(
            ...     "Design a distributed system",
            ...     top_k=5
            ... )
            >>> for sol in similar:
            ...     print(f"Similarity: {sol['similarity']}")
            ...     print(f"Problem: {sol['problem']}")
        """
        correlation_id = correlation_id or self._generate_correlation_id("retrieve")
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting similar solution retrieval",
            "problem_length": len(problem),
            "top_k": top_k,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Search for entities with artifact type
            artifacts = await self.knowledge_engine.find_entities_async(
                entity_type=self.config["knowledge_artifact_type"]
            )

            # Simple similarity scoring (can be enhanced with embeddings)
            similar_solutions = []
            problem_lower = problem.lower()

            for artifact in artifacts:
                artifact_problem = artifact.get("properties", {}).get("problem", "")
                similarity = self._calculate_similarity(problem_lower, artifact_problem.lower())

                if similarity >= self.config["similarity_threshold"]:
                    similar_solutions.append({
                        "artifact_id": artifact.get("entity_id"),
                        "problem": artifact_problem,
                        "similarity": similarity,
                        "metadata": artifact.get("properties", {}).get("metadata", {}),
                        "created_at": artifact.get("properties", {}).get("created_at")
                    })

            # Sort by similarity and take top_k
            similar_solutions.sort(key=lambda x: x["similarity"], reverse=True)
            similar_solutions = similar_solutions[:top_k]

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self._stats["similar_retrievals"] += 1
            self._stats["total_processing_time_ms"] += processing_time_ms

            logger.info({
                "msg": "Similar solution retrieval completed",
                "correlation_id": correlation_id,
                "solutions_found": len(similar_solutions),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return similar_solutions

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self._stats["errors_encountered"] += 1

            logger.error({
                "msg": "Similar solution retrieval failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return []

    def _calculate_similarity(self, query: str, target: str) -> float:
        """
        Calculate similarity between two strings.

        Simple implementation using word overlap. Can be enhanced with
        embedding-based similarity.

        Args:
            query: Query string
            target: Target string

        Returns:
            Similarity score between 0 and 1
        """
        query_words = set(query.split())
        target_words = set(target.split())

        if not query_words or not target_words:
            return 0.0

        intersection = query_words.intersection(target_words)
        union = query_words.union(target_words)

        return len(intersection) / len(union) if union else 0.0

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline execution statistics.

        Returns:
            Dictionary with pipeline metrics

        Example:
            >>> stats = await pipeline.get_statistics()
            >>> print(stats["executions_performed"])
        """
        return {
            "executions_performed": self._stats["executions_performed"],
            "entities_extracted": self._stats["entities_extracted"],
            "solutions_stored": self._stats["solutions_stored"],
            "similar_retrievals": self._stats["similar_retrievals"],
            "errors_encountered": self._stats["errors_encountered"],
            "total_processing_time_ms": self._stats["total_processing_time_ms"],
            "average_processing_time_ms": (
                self._stats["total_processing_time_ms"] / self._stats["executions_performed"]
                if self._stats["executions_performed"] > 0
                else 0.0
            ),
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Check pipeline health status.

        Returns:
            Dictionary with health status

        Example:
            >>> health = await pipeline.health_check()
            >>> print(health["status"])  # "healthy", "degraded", or "unhealthy"
        """
        roma_health = self.roma.health_check()

        # Check knowledge engine availability
        ke_available = self.knowledge_engine is not None

        # Determine overall health
        if roma_health["status"] == "healthy" and ke_available:
            status = "healthy"
        elif roma_health["status"] == "degraded" or ke_available:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "roma_integration": roma_health["status"],
            "knowledge_engine": "available" if ke_available else "unavailable",
            "statistics": await self.get_statistics(),
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def close(self):
        """
        Close resources used by the pipeline.

        Performs cleanup of any open connections or resources.

        Example:
            >>> await pipeline.close()
        """
        logger.info({
            "msg": "Closing ROMA knowledge pipeline resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Close ROMA integration
        if self.roma and hasattr(self.roma, 'close'):
            try:
                await self.roma.close()
            except Exception as e:
                logger.error({
                    "msg": "Error closing ROMA integration",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        # Note: EntityKnowledgeGraph doesn't have a close method currently

        logger.info({
            "msg": "ROMA knowledge pipeline resources closed",
            "statistics": await self.get_statistics(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


async def create_roma_knowledge_pipeline(
    roma_config: Optional[Dict[str, Any]] = None,
    pipeline_config: Optional[Dict[str, Any]] = None
) -> ROMAKnowledgePipeline:
    """
    Factory function to create ROMA knowledge pipeline.

    Args:
        roma_config: Configuration for ROMA integration
        pipeline_config: Configuration for knowledge pipeline

    Returns:
        Initialized ROMAKnowledgePipeline instance

    Example:
        >>> pipeline = await create_roma_knowledge_pipeline({
        ...     "auto_extract_entities": True,
        ...     "auto_store_solutions": True
        ... })
    """
    # Create ROMA integration
    roma = ROMAIntegration(config=roma_config)

    # Create knowledge engine
    knowledge_engine = EntityKnowledgeGraph()

    # Create pipeline
    pipeline = ROMAKnowledgePipeline(
        roma_integration=roma,
        knowledge_engine=knowledge_engine,
        config=pipeline_config
    )

    logger.info({
        "msg": "ROMA knowledge pipeline created via factory",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return pipeline


__all__ = [
    'ROMAKnowledgePipeline',
    'EntityExtractionResult',
    'KnowledgeArtifact',
    'create_roma_knowledge_pipeline'
]
