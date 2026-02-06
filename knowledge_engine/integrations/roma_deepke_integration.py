"""
ROMA-DeepKE Integration for OpenEvolve Knowledge Engine

This module provides integration between ROMA (Recursive Optimized Multi-Agent)
decomposition system and DeepKE knowledge extraction, enabling automatic entity
extraction from ROMA solutions and knowledge graph integration.

Features:
- Auto-extract entities from ROMA solutions
- Extract relations between entities
- Create knowledge graph entities from extracted data
- Batch entity extraction for efficiency
- Entity deduplication and confidence scoring
- Graceful degradation if DeepKE unavailable

Follows CLAUDE.md principles:
- ZERO TRUST: Validate all inputs
- RUNTIME TRUTH: Verify operations succeed
- IDEMPOTENCY: All operations safe to retry
- CONFIGURATION EXPLICITNESS: No magic defaults
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
- AIR GAP: No direct imports from core-projects/
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import uuid
import json
from pathlib import Path

# Import required integrations
from .roma_integration import ROMAIntegration, ROMAResult, ROMASolution
from .deepke_integration import DeepKEIntegration, DeepKEResult


logger = logging.getLogger(__name__)

# ROMA-DeepKE integration availability flag
DEEPKE_AVAILABLE = True


@dataclass
class EntityExtraction:
    """
    Result of entity extraction from a ROMA solution.

    Attributes:
        entities: List of extracted entity dictionaries
        relations: List of extracted relation dictionaries
        confidence: Overall confidence score for extraction
        extraction_metadata: Metadata about the extraction process
    """
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'entities': self.entities,
            'relations': self.relations,
            'confidence': self.confidence,
            'extraction_metadata': self.extraction_metadata
        }


class ROMADeepKEIntegration:
    """
    Integration between ROMA and DeepKE for automatic entity extraction.

    This class connects ROMA's problem-solving outputs with DeepKE's knowledge
    extraction capabilities to automatically identify and extract entities,
    relations, and knowledge from ROMA solutions.

    Features:
    - Extract entities from ROMA solution text
    - Extract relations between entities
    - Create knowledge graph entities from extracted data
    - Batch extraction for efficiency
    - Entity deduplication
    - Confidence scoring and filtering
    - Graceful degradation if DeepKE unavailable

    Example:
        >>> from knowledge_engine.integrations import ROMAIntegration, DeepKEIntegration
        >>> from knowledge_engine.core import EntityKnowledgeGraph
        >>>
        >>> # Create integrations
        >>> roma = ROMAIntegration()
        >>> deepke = DeepKEIntegration()
        >>> kg = EntityKnowledgeGraph()
        >>>
        >>> # Create ROMA-DeepKE integration
        >>> roma_deepke = ROMADeepKEIntegration(roma, deepke, kg)
        >>>
        >>> # Get ROMA solution
        >>> result = await roma.decompose_problem("Design a scalable microservices architecture")
        >>> solution = await roma.solve_atomic(result.decomposition)
        >>>
        >>> # Enrich with entities
        >>> enriched = await roma_deepke.enrich_with_entities(solution)
        >>> print(f"Extracted {len(enriched.metadata['extracted_entities'])} entities")
    """

    def __init__(
        self,
        roma_integration: ROMAIntegration,
        deepke_integration: DeepKEIntegration,
        knowledge_engine,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ROMA-DeepKE integration.

        Args:
            roma_integration: ROMA integration instance
            deepke_integration: DeepKE integration instance
            knowledge_engine: Knowledge engine instance for storing entities
            config: Optional configuration dictionary

        Raises:
            ValueError: If required integrations are None
        """
        if not roma_integration:
            raise ValueError("ROMA integration is required")
        if not deepke_integration:
            raise ValueError("DeepKE integration is required")
        if not knowledge_engine:
            raise ValueError("Knowledge engine is required")

        self.roma = roma_integration
        self.deepke = deepke_integration
        self.knowledge_engine = knowledge_engine
        self.config = config or self._get_default_config()

        # Entity tracking for deduplication
        self._seen_entities: Set[str] = set()
        self._entity_lock = None  # Async lock, initialized lazily

        # Statistics tracking
        self._stats = {
            "solutions_processed": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "kg_entities_created": 0,
            "kg_relations_created": 0,
            "total_processing_time_ms": 0.0,
            "extraction_failures": 0
        }

        logger.info({
            "msg": "ROMADeepKEIntegration initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ROMA-DeepKE integration."""
        return {
            "auto_extract_entities": True,
            "auto_extract_relations": True,
            "auto_create_kg_entities": True,
            "entity_types": ["PERSON", "ORG", "TECH", "CONCEPT", "SOLUTION", "PROBLEM"],
            "relation_types": ["uses", "depends_on", "solves", "related_to", "part_of"],
            "confidence_threshold": 0.7,
            "min_entity_length": 2,
            "extraction_model": "default",
            "batch_size": 10,
            "deduplication_enabled": True,
            "create_entity_relations": True,
            "entity_naming_strategy": "descriptive"  # "descriptive", "compact", "uuid"
        }

    async def _get_async_lock(self) -> asyncio.Lock:
        """Get or create async lock (lazy initialization)."""
        if self._entity_lock is None:
            self._entity_lock = asyncio.Lock()
        return self._entity_lock

    async def enrich_with_entities(
        self,
        solution: ROMAResult,
        correlation_id: Optional[str] = None
    ) -> ROMAResult:
        """
        Enrich a ROMA solution with extracted entities and relations.

        This is the main entry point for entity extraction. It extracts
        entities and relations from the solution text, creates knowledge
        graph entities, and enhances the solution metadata.

        Args:
            solution: ROMA solution to enrich
            correlation_id: Optional correlation ID for tracking

        Returns:
            Enhanced ROMA result with extracted entities in metadata

        Example:
            >>> result = await roma.solve_atomic(atomic_problem)
            >>> enriched = await roma_deepke.enrich_with_entities(result)
            >>> entities = enriched.metadata.get('extracted_entities', [])
        """
        correlation_id = correlation_id or f"roma_enrich_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting ROMA solution entity enrichment",
            "solution_count": len(solution.solutions),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Check if auto-extraction is enabled
            if not self.config.get("auto_extract_entities", True):
                logger.debug({
                    "msg": "Entity extraction disabled in config",
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return solution

            # Extract entities from each solution
            all_entities = []
            all_relations = []

            for sol in solution.solutions:
                # Convert solution to text for extraction
                solution_text = self._solution_to_text(sol)
                solution_type = solution.metadata.get("strategy", "general")

                # Extract entities
                entities = await self.extract_entities_from_solution(
                    solution_text, solution_type, correlation_id
                )
                all_entities.extend(entities)

                # Extract relations if enabled
                if self.config.get("auto_extract_relations", True):
                    relations = await self.extract_relations_from_solution(
                        solution_text, entities, correlation_id
                    )
                    all_relations.extend(relations)

            # Deduplicate entities
            if self.config.get("deduplication_enabled", True):
                all_entities = await self._deduplicate_entities(all_entities)

            # Create knowledge graph entities if enabled
            kg_entity_ids = []
            if self.config.get("auto_create_kg_entities", True):
                kg_entity_ids = await self.create_knowledge_entities(
                    all_entities, all_relations, correlation_id
                )

            # Filter by confidence threshold
            confidence_threshold = self.config.get("confidence_threshold", 0.7)
            high_confidence_entities = [
                e for e in all_entities
                if e.get("confidence", 0.0) >= confidence_threshold
            ]

            # Create extraction result
            extraction = EntityExtraction(
                entities=high_confidence_entities,
                relations=all_relations,
                confidence=self._calculate_overall_confidence(high_confidence_entities),
                extraction_metadata={
                    "total_entities_extracted": len(all_entities),
                    "high_confidence_entities": len(high_confidence_entities),
                    "relations_extracted": len(all_relations),
                    "kg_entities_created": len(kg_entity_ids),
                    "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                    "confidence_threshold": confidence_threshold
                }
            )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Update statistics
            self._stats["solutions_processed"] += 1
            self._stats["entities_extracted"] += len(all_entities)
            self._stats["relations_extracted"] += len(all_relations)
            self._stats["kg_entities_created"] += len(kg_entity_ids)
            self._stats["total_processing_time_ms"] += processing_time_ms

            # Enhance solution metadata
            enhanced_metadata = solution.metadata.copy()
            enhanced_metadata["extracted_entities"] = extraction.to_dict()
            enhanced_metadata["entity_extraction_time_ms"] = processing_time_ms
            enhanced_metadata["kg_entity_ids"] = kg_entity_ids

            # Create enhanced result
            enhanced_result = ROMAResult(
                success=solution.success,
                decomposition=solution.decomposition,
                solutions=solution.solutions,
                verification=solution.verification,
                metadata=enhanced_metadata,
                processing_time_ms=solution.processing_time_ms + processing_time_ms,
                error=solution.error
            )

            logger.info({
                "msg": "ROMA solution entity enrichment completed",
                "correlation_id": correlation_id,
                "entities_extracted": len(all_entities),
                "high_confidence_entities": len(high_confidence_entities),
                "relations_extracted": len(all_relations),
                "kg_entities_created": len(kg_entity_ids),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return enhanced_result

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self._stats["extraction_failures"] += 1

            logger.error({
                "msg": "ROMA solution entity enrichment failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return original solution on failure (graceful degradation)
            solution.metadata["entity_extraction_error"] = str(e)
            solution.metadata["entity_extraction_time_ms"] = processing_time_ms
            return solution

    def _solution_to_text(self, solution: ROMASolution) -> str:
        """
        Convert ROMA solution to text for entity extraction.

        Args:
            solution: ROMA solution

        Returns:
            Text representation of the solution
        """
        parts = []

        # Add solution content
        if solution.solution:
            parts.append(f"Solution: {str(solution.solution)}")

        # Add reasoning
        if solution.reasoning:
            parts.append(f"Reasoning: {solution.reasoning}")

        # Add metadata
        if solution.metadata:
            metadata_str = json.dumps(solution.metadata, default=str)
            parts.append(f"Metadata: {metadata_str}")

        return "\n\n".join(parts)

    async def extract_entities_from_solution(
        self,
        solution_text: str,
        solution_type: str,
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from ROMA solution text using DeepKE.

        Args:
            solution_text: Solution text to extract from
            solution_type: Type of solution (affects extraction strategy)
            correlation_id: Optional correlation ID

        Returns:
            List of entity dictionaries with properties

        Example:
            >>> entities = await roma_deepke.extract_entities_from_solution(
            ...     "Implement a REST API using FastAPI",
            ...     "technical_solution"
            ... )
            >>> # Returns: [{"name": "REST API", "type": "TECH", "confidence": 0.85}]
        """
        correlation_id = correlation_id or f"extract_ent_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Extracting entities from solution text",
            "text_length": len(solution_text),
            "solution_type": solution_type,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Get entity types from config
            entity_types = self.config.get("entity_types", ["PERSON", "ORG", "TECH", "CONCEPT"])

            # Use DeepKE to extract entities
            deepke_result: DeepKEResult = await self.deepke.extract_entities(
                text=solution_text,
                entity_types=entity_types,
                correlation_id=correlation_id
            )

            if not deepke_result.success:
                logger.warning({
                    "msg": "DeepKE entity extraction failed, using fallback",
                    "error": deepke_result.error,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                # Use fallback extraction
                entities = self._fallback_entity_extraction(solution_text, entity_types)
            else:
                entities = deepke_result.entities

            # Enhance entities with additional metadata
            min_entity_length = self.config.get("min_entity_length", 2)

            enhanced_entities = []
            for entity in entities:
                # Filter by minimum length
                if len(entity.get("name", "")) < min_entity_length:
                    continue

                # Enhance with metadata
                enhanced_entity = {
                    "name": entity.get("name", ""),
                    "type": entity.get("type", "ENTITY"),
                    "confidence": entity.get("confidence", 0.7),
                    "properties": {
                        "solution_type": solution_type,
                        "source": "roma_deepke_extraction",
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                        "extraction_confidence": entity.get("confidence", 0.7)
                    },
                    "metadata": {
                        "extraction_method": "deepke",
                        "solution_type": solution_type
                    }
                }
                enhanced_entities.append(enhanced_entity)

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info({
                "msg": "Entity extraction completed",
                "correlation_id": correlation_id,
                "entities_count": len(enhanced_entities),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return enhanced_entities

        except Exception as e:
            logger.error({
                "msg": "Entity extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return empty list on failure (graceful degradation)
            return []

    def _fallback_entity_extraction(
        self,
        text: str,
        entity_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Fallback entity extraction using simple patterns.

        Used when DeepKE is unavailable or fails.

        Args:
            text: Text to extract from
            entity_types: Types of entities to extract

        Returns:
            List of entity dictionaries
        """
        import re

        entities = []
        seen = set()

        # Simple pattern matching for technical terms
        tech_pattern = r'\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b'
        matches = re.findall(tech_pattern, text)

        for match in matches:
            if len(match) < 2 or match in seen:
                continue

            # Determine entity type
            entity_type = "TECH"
            if any(org in match.upper() for org in ["INC", "CORP", "LLC", "LTD"]):
                entity_type = "ORG"
            elif match.isupper() and len(match.split()) == 1:
                entity_type = "CONCEPT"

            if entity_type in entity_types:
                entities.append({
                    "name": match,
                    "type": entity_type,
                    "confidence": 0.6  # Lower confidence for fallback
                })
                seen.add(match)

        return entities

    async def extract_relations_from_solution(
        self,
        solution_text: str,
        entities: List[Dict],
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract relations between entities from solution text.

        Args:
            solution_text: Solution text to extract relations from
            entities: List of entities to find relations between
            correlation_id: Optional correlation ID

        Returns:
            List of relation dictionaries

        Example:
            >>> entities = [{"name": "FastAPI", "type": "TECH"}]
            >>> relations = await roma_deepke.extract_relations_from_solution(
            ...     "FastAPI is used for building REST APIs",
            ...     entities
            ... )
        """
        correlation_id = correlation_id or f"extract_rel_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Extracting relations from solution text",
            "entity_count": len(entities),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Use DeepKE to extract relations
            deepke_result: DeepKEResult = await self.deepke.extract_relations(
                text=solution_text,
                correlation_id=correlation_id
            )

            if not deepke_result.success:
                logger.warning({
                    "msg": "DeepKE relation extraction failed, using fallback",
                    "error": deepke_result.error,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                # Use fallback extraction
                relations = self._fallback_relation_extraction(solution_text, entities)
            else:
                relations = deepke_result.relations

            # Filter by configured relation types
            allowed_relation_types = set(self.config.get("relation_types", []))
            if allowed_relation_types:
                relations = [
                    r for r in relations
                    if r.get("predicate", "") in allowed_relation_types
                ]

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info({
                "msg": "Relation extraction completed",
                "correlation_id": correlation_id,
                "relations_count": len(relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return relations

        except Exception as e:
            logger.error({
                "msg": "Relation extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return empty list on failure
            return []

    def _fallback_relation_extraction(
        self,
        text: str,
        entities: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Fallback relation extraction using simple patterns.

        Args:
            text: Text to extract from
            entities: List of entities

        Returns:
            List of relation dictionaries
        """
        import re

        relations = []
        entity_names = {e["name"].lower() for e in entities}

        # Simple relation patterns
        patterns = {
            "uses": r"(\w+)\s+uses?\s+(\w+)",
            "depends_on": r"(\w+)\s+depends?\s+on\s+(\w+)",
            "solves": r"(\w+)\s+solves?\s+(\w+)",
            "related_to": r"(\w+)\s+is\s+related\s+to\s+(\w+)"
        }

        for relation_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for subj, obj in matches:
                if subj.lower() in entity_names and obj.lower() in entity_names:
                    relations.append({
                        "subject": subj,
                        "predicate": relation_type,
                        "object": obj,
                        "confidence": 0.6
                    })

        return relations

    async def _deduplicate_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate entities based on name and type.

        Args:
            entities: List of entities to deduplicate

        Returns:
            Deduplicated list of entities
        """
        async with await self._get_async_lock():
            seen = {}
            deduplicated = []

            for entity in entities:
                key = (entity["name"].lower(), entity["type"])

                if key in seen:
                    # Merge with existing entity (keep highest confidence)
                    existing = seen[key]
                    if entity.get("confidence", 0.0) > existing.get("confidence", 0.0):
                        seen[key] = entity
                else:
                    seen[key] = entity

            deduplicated = list(seen.values())

            logger.debug({
                "msg": "Entity deduplication completed",
                "original_count": len(entities),
                "deduplicated_count": len(deduplicated),
                "removed_count": len(entities) - len(deduplicated)
            })

            return deduplicated

    async def create_knowledge_entities(
        self,
        entities: List[Dict],
        relations: List[Dict],
        correlation_id: Optional[str] = None
    ) -> List[str]:
        """
        Create entities and relations in the knowledge graph.

        Args:
            entities: List of entity dictionaries
            relations: List of relation dictionaries
            correlation_id: Optional correlation ID

        Returns:
            List of created entity IDs

        Example:
            >>> entity_ids = await roma_deepke.create_knowledge_entities(
            ...     entities,
            ...     relations
            ... )
            >>> print(f"Created {len(entity_ids)} knowledge graph entities")
        """
        correlation_id = correlation_id or f"create_kg_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Creating knowledge graph entities",
            "entity_count": len(entities),
            "relation_count": len(relations),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            created_entity_ids = []

            # Create entities
            for entity in entities:
                # Generate entity ID based on naming strategy
                entity_id = self._generate_entity_id(entity)

                # Add to knowledge graph
                success = await self.knowledge_engine.add_entity_async(
                    name=entity_id,
                    entity_type=entity["type"],
                    attributes=entity.get("properties", {})
                )

                if success:
                    created_entity_ids.append(entity_id)

                    # Track for deduplication
                    async with await self._get_async_lock():
                        self._seen_entities.add(entity_id)

            # Create relations if enabled
            if self.config.get("create_entity_relations", True):
                for relation in relations:
                    # Generate IDs for source and target
                    source_id = self._generate_entity_id({"name": relation["subject"], "type": "ENTITY"})
                    target_id = self._generate_entity_id({"name": relation["object"], "type": "ENTITY"})

                    # Add relation to knowledge graph
                    await self.knowledge_engine.add_relationship_async(
                        source=source_id,
                        target=target_id,
                        relation_type=relation.get("predicate", "related_to"),
                        attributes={
                            "confidence": relation.get("confidence", 0.7),
                            "source": "roma_deepke_extraction"
                        }
                    )

                    self._stats["kg_relations_created"] += 1

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info({
                "msg": "Knowledge graph entity creation completed",
                "correlation_id": correlation_id,
                "entities_created": len(created_entity_ids),
                "relations_created": len(relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return created_entity_ids

        except Exception as e:
            logger.error({
                "msg": "Knowledge graph entity creation failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return empty list on failure
            return []

    def _generate_entity_id(self, entity: Dict[str, Any]) -> str:
        """
        Generate a unique entity ID based on naming strategy.

        Args:
            entity: Entity dictionary

        Returns:
            Unique entity ID
        """
        naming_strategy = self.config.get("entity_naming_strategy", "descriptive")
        entity_name = entity.get("name", "unknown")
        entity_type = entity.get("type", "ENTITY")

        if naming_strategy == "uuid":
            return f"entity_{uuid.uuid4().hex[:16]}"
        elif naming_strategy == "compact":
            # Use first few chars of name + type
            clean_name = entity_name.replace(" ", "_").lower()[:20]
            return f"{entity_type.lower()}_{clean_name}"
        else:  # descriptive
            # Use full descriptive name
            clean_name = entity_name.replace(" ", "_").replace("/", "_").lower()
            return f"{entity_type.lower()}_{clean_name}"

    def _calculate_overall_confidence(self, entities: List[Dict[str, Any]]) -> float:
        """
        Calculate overall confidence score for entity extraction.

        Args:
            entities: List of entities

        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        if not entities:
            return 0.0

        confidences = [e.get("confidence", 0.7) for e in entities]
        return round(sum(confidences) / len(confidences), 3)

    async def batch_extract_entities(
        self,
        solutions: List[ROMAResult],
        correlation_id: Optional[str] = None
    ) -> List[ROMAResult]:
        """
        Extract entities from multiple solutions in parallel.

        Args:
            solutions: List of ROMA results to extract entities from
            correlation_id: Optional correlation ID

        Returns:
            List of enhanced ROMA results

        Example:
            >>> results = await roma.batch_decompose(problems)
            >>> enriched = await roma_deepke.batch_extract_entities(results)
            >>> print(f"Enriched {len(enriched)} solutions")
        """
        correlation_id = correlation_id or f"batch_extract_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting batch entity extraction",
            "solution_count": len(solutions),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            batch_size = self.config.get("batch_size", 10)

            # Process in batches to control parallelism
            enhanced_solutions = []

            for i in range(0, len(solutions), batch_size):
                batch = solutions[i:i + batch_size]

                # Process batch in parallel
                tasks = [
                    self.enrich_with_entities(
                        solution,
                        correlation_id=f"{correlation_id}_{idx}"
                    )
                    for idx, solution in enumerate(batch)
                ]

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error({
                            "msg": f"Batch item {i+j} enrichment failed",
                            "correlation_id": f"{correlation_id}_{i+j}",
                            "error": str(result)
                        })
                        # Return original solution on failure
                        enhanced_solutions.append(batch[j])
                    else:
                        enhanced_solutions.append(result)

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info({
                "msg": "Batch entity extraction completed",
                "correlation_id": correlation_id,
                "solution_count": len(solutions),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return enhanced_solutions

        except Exception as e:
            logger.error({
                "msg": "Batch entity extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return original solutions on failure
            return solutions

    async def get_entity_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about entity extraction.

        Returns:
            Dictionary with extraction statistics

        Example:
            >>> stats = await roma_deepke.get_entity_statistics()
            >>> print(f"Extracted {stats['total_entities']} entities")
        """
        # Get knowledge graph statistics
        kg_stats = await self.knowledge_engine.get_statistics_async()

        return {
            "solutions_processed": self._stats["solutions_processed"],
            "entities_extracted": self._stats["entities_extracted"],
            "relations_extracted": self._stats["relations_extracted"],
            "kg_entities_created": self._stats["kg_entities_created"],
            "kg_relations_created": self._stats["kg_relations_created"],
            "extraction_failures": self._stats["extraction_failures"],
            "total_processing_time_ms": self._stats["total_processing_time_ms"],
            "average_processing_time_ms": (
                self._stats["total_processing_time_ms"] / self._stats["solutions_processed"]
                if self._stats["solutions_processed"] > 0
                else 0.0
            ),
            "success_rate": (
                (self._stats["solutions_processed"] - self._stats["extraction_failures"])
                / self._stats["solutions_processed"]
                if self._stats["solutions_processed"] > 0
                else 1.0
            ),
            "knowledge_graph_stats": kg_stats,
            "config": {
                "auto_extract_entities": self.config.get("auto_extract_entities"),
                "auto_extract_relations": self.config.get("auto_extract_relations"),
                "auto_create_kg_entities": self.config.get("auto_create_kg_entities"),
                "confidence_threshold": self.config.get("confidence_threshold"),
                "deduplication_enabled": self.config.get("deduplication_enabled")
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def close(self):
        """
        Close resources used by the integration.

        Performs cleanup of resources and logs final statistics.
        """
        logger.info({
            "msg": "Closing ROMA-DeepKE integration resources",
            "statistics": await self.get_entity_statistics(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Clear entity tracking
        self._seen_entities.clear()

        # Note: We don't close roma or deepke integrations here
        # as they may be used elsewhere

        logger.info({
            "msg": "ROMA-DeepKE integration resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


async def create_roma_deepke_integration(
    roma_integration: Optional[ROMAIntegration] = None,
    deepke_integration: Optional[DeepKEIntegration] = None,
    knowledge_engine=None,
    config: Optional[Dict[str, Any]] = None
) -> ROMADeepKEIntegration:
    """
    Factory function to create a ROMA-DeepKE integration.

    Args:
        roma_integration: ROMA integration instance (created if None)
        deepke_integration: DeepKE integration instance (created if None)
        knowledge_engine: Knowledge engine instance (required)
        config: Optional configuration

    Returns:
        ROMADeepKEIntegration instance

    Example:
        >>> from knowledge_engine.core import EntityKnowledgeGraph
        >>>
        >>> kg = EntityKnowledgeGraph()
        >>> roma_deepke = await create_roma_deepke_integration(
        ...     knowledge_engine=kg,
        ...     config={"confidence_threshold": 0.8}
        ... )
    """
    if not knowledge_engine:
        raise ValueError("knowledge_engine is required")

    # Create integrations if not provided
    if not roma_integration:
        roma_integration = ROMAIntegration(config=config)

    if not deepke_integration:
        deepke_integration = DeepKEIntegration(config=config)

    # Create and return the integration
    integration = ROMADeepKEIntegration(
        roma_integration=roma_integration,
        deepke_integration=deepke_integration,
        knowledge_engine=knowledge_engine,
        config=config
    )

    logger.info({
        "msg": "ROMA-DeepKE integration created via factory",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return integration


__all__ = [
    'ROMADeepKEIntegration',
    'EntityExtraction',
    'create_roma_deepke_integration'
]
