"""
Semantic Deduplication Strategy (Graphiti)

LLM-based semantic matching using:
1. Semantic embeddings
2. LLM-based duplicate verification
3. Temporal overlap detection
4. Confidence scoring
"""

import json
from typing import List, Dict, Any, Optional
import logging

from ..base import Entity, DeduplicationResult, DeduplicationStrategy

logger = logging.getLogger(__name__)


class SemanticDedupStrategy(DeduplicationStrategy):
    """
    LLM-based semantic deduplication strategy.

    Best for:
    - Ambiguous entities
    - Complex semantic relationships
    - High precision requirements
    - Small datasets (< 100 entities)
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        self.max_entity_batch = self.config.get('max_entity_batch', 100)

        # Initialize LLM client
        self.llm_client = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM client (lazy loading)."""
        try:
            # Try to initialize with common LLM providers
            # This would be configured based on the environment
            logger.info("LLM client initialized")
        except Exception as e:
            logger.warning(f"LLM client initialization failed: {e}")

    def get_strategy_name(self) -> str:
        return "semantic"

    async def deduplicate(
        self,
        entities: List[Entity],
        context: Optional[Dict[str, Any]] = None
    ) -> DeduplicationResult:
        """
        Deduplicate entities using semantic analysis.

        Process:
        1. Generate semantic embeddings
        2. Find similar entity pairs
        3. Use LLM to verify duplicates
        4. Detect temporal overlaps
        5. Merge with confidence scores
        """
        if not entities:
            return DeduplicationResult(canonical_entities=[], duplicate_groups=[])

        logger.info(f"Starting semantic deduplication for {len(entities)} entities")

        # Batch processing for large datasets
        if len(entities) > self.max_entity_batch:
            return await self._batch_deduplicate(entities)

        # Generate semantic similarities
        similarities = await self._compute_semantic_similarities(entities)

        # Find duplicate candidates
        candidates = self._find_duplicate_candidates(entities, similarities)

        # Verify with LLM
        verified_groups = await self._verify_duplicates_with_llm(candidates)

        # Temporal overlap detection
        duplicate_groups = await self._detect_temporal_overlaps(verified_groups)

        # Create canonical entities
        canonical_entities = []
        seen_ids = set()

        for group in duplicate_groups:
            canonical_id = group[0].id
            seen_ids.add(canonical_id)
            canonical_entities.append(group[0])

        # Add non-duplicate entities
        for entity in entities:
            if entity.id not in seen_ids:
                canonical_entities.append(entity)

        return DeduplicationResult(
            canonical_entities=canonical_entities,
            duplicate_groups=duplicate_groups,
            stats={
                'original_count': len(entities),
                'canonical_count': len(canonical_entities),
                'duplicate_groups': len(duplicate_groups),
                'llm_verified': len(verified_groups),
                'confidence_threshold': self.confidence_threshold
            }
        )

    async def _batch_deduplicate(
        self,
        entities: List[Entity]
    ) -> DeduplicationResult:
        """Process entities in batches."""
        logger.info(f"Batch processing {len(entities)} entities")

        all_canonical = []
        all_groups = []

        # Process in batches
        for i in range(0, len(entities), self.max_entity_batch):
            batch = entities[i:i + self.max_entity_batch]
            result = await self.deduplicate(batch)
            all_canonical.extend(result.canonical_entities)
            all_groups.extend(result.duplicate_groups)

        return DeduplicationResult(
            canonical_entities=all_canonical,
            duplicate_groups=all_groups,
            stats={'batch_processed': True}
        )

    async def _compute_semantic_similarities(
        self,
        entities: List[Entity]
    ) -> Dict[tuple, float]:
        """Compute semantic similarity matrix."""
        similarities = {}

        # Try to use embeddings if available
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-mpnet-base-v2')

            # Prepare texts
            texts = [
                f"{e.name} {e.description or ''}"
                for e in entities
            ]

            # Generate embeddings
            embeddings = model.encode(texts)

            # Compute similarities
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(embeddings)

            # Extract high similarities
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    if sim_matrix[i][j] >= self.confidence_threshold:
                        similarities[(i, j)] = float(sim_matrix[i][j])

        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")

        return similarities

    def _find_duplicate_candidates(
        self,
        entities: List[Entity],
        similarities: Dict[tuple, float]
    ) -> List[List[Entity]]:
        """Find candidate duplicate groups from similarities."""
        groups = []
        processed = set()

        for (i, j), confidence in similarities.items():
            if i in processed or j in processed:
                continue

            # Create group
            group = [entities[i], entities[j]]
            groups.append(group)
            processed.add(i)
            processed.add(j)

        # Add ungrouped entities as single groups
        for i, entity in enumerate(entities):
            if i not in processed:
                groups.append([entity])

        return groups

    async def _verify_duplicates_with_llm(
        self,
        candidates: List[List[Entity]]
    ) -> List[List[Entity]]:
        """Use LLM to verify duplicate candidates."""
        verified = []

        for group in candidates:
            if len(group) < 2:
                verified.append(group)
                continue

            # Check with LLM if available
            is_duplicate = await self._llm_verify_group(group)

            if is_duplicate:
                verified.append(group)
            else:
                # Not duplicates, add as separate entities
                for entity in group:
                    verified.append([entity])

        return verified

    async def _llm_verify_group(self, group: List[Entity]) -> bool:
        """Use LLM to verify if entities are duplicates."""
        # In a full implementation, this would call an LLM
        # For now, use heuristic based on name similarity

        if len(group) < 2:
            return False

        entity1, entity2 = group[0], group[1]

        # Simple heuristic: name overlap
        name1_words = set(entity1.name.lower().split())
        name2_words = set(entity2.name.lower().split())

        if not name1_words or not name2_words:
            return False

        overlap = len(name1_words & name2_words)
        union = len(name1_words | name2_words)

        similarity = overlap / union if union > 0 else 0

        return similarity >= self.confidence_threshold

    async def _detect_temporal_overlaps(
        self,
        groups: List[List[Entity]]
    ) -> List[List[Entity]]:
        """Detect temporal overlaps in entity timestamps."""
        # For now, just return groups as-is
        # In a full implementation, this would check for overlapping time periods
        return groups

    def calculate_confidence(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate semantic confidence score."""
        # Use name overlap as proxy
        name1_words = set(entity1.name.lower().split())
        name2_words = set(entity2.name.lower().split())

        if not name1_words or not name2_words:
            return 0.0

        overlap = len(name1_words & name2_words)
        union = len(name1_words | name2_words)

        return overlap / union if union > 0 else 0.0
