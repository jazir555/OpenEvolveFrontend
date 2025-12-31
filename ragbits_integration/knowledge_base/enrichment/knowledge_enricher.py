"""
Knowledge Enricher

Enriches extracted knowledge with additional context,
relationships, and metadata.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ragbits_integration.knowledge_base.extraction.knowledge_extractor import (
    KnowledgeEntity,
    KnowledgeEntityType
)

logger = logging.getLogger(__name__)


@dataclass
class EnrichedEntity:
    """An enriched knowledge entity"""
    original_entity: KnowledgeEntity
    additional_context: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    success_rate: Optional[float] = None
    usage_count: int = 0
    last_used: Optional[float] = None
    quality_score: float = 0.5
    enrichment_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_entity": self.original_entity.to_dict(),
            "additional_context": self.additional_context,
            "related_patterns": self.related_patterns,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
            "quality_score": self.quality_score,
            "enrichment_metadata": self.enrichment_metadata
        }


@dataclass
class EnrichmentResult:
    """Result of knowledge enrichment"""
    enriched_entities: List[EnrichedEntity]
    enrichment_summary: Dict[str, Any]
    processing_time_ms: float
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())


class KnowledgeEnricher:
    """
    Enriches knowledge entities with additional information.

    Usage:
        enricher = KnowledgeEnricher(storage_manager, hephaestus_client)

        result = await enricher.enrich_entities(
            entities=extracted_entities,
            artifact_type="solution"
        )
    """

    def __init__(self, storage_manager=None, hephaestus_client=None):
        """
        Initialize knowledge enricher.

        Args:
            storage_manager: Storage manager for context retrieval
            hephaestus_client: Optional LLM client for enrichment
        """
        self.storage_manager = storage_manager
        self.hephaestus_client = hephaestus_client

        # Enrichment statistics
        self.enrichment_stats = {
            "entities_enriched": 0,
            "context_added": 0,
            "relationships_found": 0
        }

        logger.info("KnowledgeEnricher initialized")

    async def enrich_entities(
        self,
        entities: List[KnowledgeEntity],
        artifact_type: str,
        add_context: bool = True,
        find_relationships: bool = True
    ) -> EnrichmentResult:
        """
        Enrich a list of knowledge entities.

        Args:
            entities: Entities to enrich
            artifact_type: Type of source artifact
            add_context: Whether to add contextual information
            find_relationships: Whether to find related entities

        Returns:
            Enrichment result
        """
        import time
        start_time = time.time()

        enriched = []

        for entity in entities:
            enriched_entity = await self._enrich_single_entity(
                entity,
                artifact_type,
                add_context,
                find_relationships
            )
            enriched.append(enriched_entity)

        processing_time = (time.time() - start_time) * 1000

        # Update statistics
        self.enrichment_stats["entities_enriched"] += len(entities)
        self.enrichment_stats["context_added"] += sum(
            len(e.additional_context) for e in enriched
        )
        self.enrichment_stats["relationships_found"] += sum(
            len(e.related_patterns) for e in enriched
        )

        return EnrichmentResult(
            enriched_entities=enriched,
            enrichment_summary={
                "total_entities": len(entities),
                "context_added": sum(len(e.additional_context) for e in enriched),
                "relationships_found": sum(len(e.related_patterns) for e in enriched),
                "average_quality_score": sum(e.quality_score for e in enriched) / len(enriched)
                if enriched else 0
            },
            processing_time_ms=processing_time
        )

    async def _enrich_single_entity(
        self,
        entity: KnowledgeEntity,
        artifact_type: str,
        add_context: bool,
        find_relationships: bool
    ) -> EnrichedEntity:
        """Enrich a single entity"""
        enriched = EnrichedEntity(
            original_entity=entity,
            quality_score=self._calculate_base_quality_score(entity)
        )

        # Add contextual information
        if add_context and self.storage_manager:
            context = await self._get_contextual_info(entity)
            enriched.additional_context.extend(context)

        # Find related patterns
        if find_relationships:
            related = await self._find_related_patterns(entity)
            enriched.related_patterns.extend(related)

        # Calculate success rate if available
        if self.storage_manager:
            enriched.success_rate = await self._estimate_success_rate(entity)

        return enriched

    async def _get_contextual_info(
        self,
        entity: KnowledgeEntity
    ) -> List[str]:
        """Get contextual information for entity"""
        context = []

        if not self.storage_manager:
            return context

        try:
            # Search for similar entities in storage
            results = await self.storage_manager.search_similar(
                query=entity.content[:200],
                limit=3,
                artifact_type="knowledge"
            )

            for result in results:
                if result.get("content"):
                    context.append(result["content"][:200])

        except Exception as e:
            logger.error(f"Error getting context: {e}")

        return context

    async def _find_related_patterns(
        self,
        entity: KnowledgeEntity
    ) -> List[str]:
        """Find related patterns and entities"""
        related = []

        # Entity-type specific relationships
        type_relationships = {
            KnowledgeEntityType.SOLUTION_PATTERN: [
                KnowledgeEntityType.BEST_PRACTICE,
                KnowledgeEntityType.TECHNIQUE
            ],
            KnowledgeEntityType.BEST_PRACTICE: [
                KnowledgeEntityType.PRINCIPLE,
                KnowledgeEntityType.SOLUTION_PATTERN
            ],
            KnowledgeEntityType.ANTI_PATTERN: [
                KnowledgeEntityType.SOLUTION_PATTERN,
                KnowledgeEntityType.LESSON_LEARNED
            ]
        }

        related_types = type_relationships.get(entity.entity_type, [])

        # In a real implementation, would search knowledge base
        # For now, return empty list
        return related

    async def _estimate_success_rate(
        self,
        entity: KnowledgeEntity
    ) -> Optional[float]:
        """Estimate success rate based on historical usage"""
        # In a real implementation, would query historical data
        # For now, return None
        return None

    def _calculate_base_quality_score(
        self,
        entity: KnowledgeEntity
    ) -> float:
        """Calculate base quality score for entity"""
        score = 0.5

        # Content length factor
        if len(entity.content) > 100:
            score += 0.1
        if len(entity.content) > 300:
            score += 0.1

        # Confidence factor
        score += (entity.confidence - 0.5) * 0.3

        # Tags factor
        if len(entity.tags) >= 3:
            score += 0.1

        # Related entities factor
        if len(entity.related_entities) >= 2:
            score += 0.1

        return min(1.0, max(0.0, score))

    def get_statistics(self) -> Dict[str, Any]:
        """Get enrichment statistics"""
        return self.enrichment_stats.copy()
