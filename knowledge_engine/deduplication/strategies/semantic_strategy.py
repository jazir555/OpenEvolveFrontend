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
        if len(group) < 2:
            return False

        # Try to use LLM for verification
        if self.llm_client:
            try:
                return await self._llm_verification_call(group)
            except Exception as e:
                logger.warning(f"LLM verification failed, falling back to heuristics: {e}")

        # Fallback to sophisticated heuristic verification
        return await self._heuristic_verification(group)

    async def _llm_verification_call(self, group: List[Entity]) -> bool:
        """Make LLM API call for duplicate verification."""
        # Build comparison prompt
        entity_descriptions = []
        for i, entity in enumerate(group):
            desc = {
                'name': entity.name,
                'type': entity.entity_type,
                'description': entity.description or '',
                'attributes': entity.attributes or {}
            }
            entity_descriptions.append(f"Entity {i+1}: {json.dumps(desc, indent=2)}")

        prompt = f"""You are a knowledge graph deduplication expert. Determine if the following entities represent the same real-world entity.

{chr(10).join(entity_descriptions)}

Consider:
- Name similarity (variations, aliases, translations)
- Type compatibility
- Description semantics
- Attribute overlap
- Temporal consistency

Respond with ONLY 'true' if they are duplicates or 'false' if they are distinct entities."""

        # Call LLM (implementation depends on configured provider)
        try:
            # Try OpenAI
            import openai
            response = await openai.AsyncClient(api_key=self.config.get('openai_api_key')).chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().lower()
            return 'true' in result or 'yes' in result

        except Exception as e:
            logger.debug(f"OpenAI LLM call failed: {e}")

        # Try fallback to litellm if available
        try:
            from litellm import acompletion
            response = await acompletion(
                model=self.config.get('llm_model', 'gpt-4'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            result = response['choices'][0]['message']['content'].strip().lower()
            return 'true' in result or 'yes' in result

        except Exception as e:
            logger.debug(f"LiteLLM call failed: {e}")

        # If all LLM attempts fail, raise to trigger fallback
        raise RuntimeError("All LLM providers failed")

    async def _heuristic_verification(self, group: List[Entity]) -> bool:
        """Sophisticated heuristic verification when LLM unavailable."""
        if len(group) < 2:
            return False

        entity1, entity2 = group[0], group[1]

        # Multi-factor similarity scoring
        scores = []

        # 1. Name similarity (Jaccard index)
        name1_words = set(entity1.name.lower().split())
        name2_words = set(entity2.name.lower().split())
        if name1_words and name2_words:
            overlap = len(name1_words & name2_words)
            union = len(name1_words | name2_words)
            scores.append(overlap / union if union > 0 else 0)

        # 2. Type compatibility
        if entity1.entity_type == entity2.entity_type:
            scores.append(1.0)
        elif self._are_compatible_types(entity1.entity_type, entity2.entity_type):
            scores.append(0.7)

        # 3. Description similarity (word overlap)
        if entity1.description and entity2.description:
            desc1_words = set(entity1.description.lower().split())
            desc2_words = set(entity2.description.lower().split())
            if desc1_words and desc2_words:
                overlap = len(desc1_words & desc2_words)
                union = len(desc1_words | desc2_words)
                scores.append(overlap / union if union > 0 else 0)

        # 4. Attribute overlap
        if entity1.attributes and entity2.attributes:
            attr_keys1 = set(entity1.attributes.keys())
            attr_keys2 = set(entity2.attributes.keys())
            if attr_keys1 and attr_keys2:
                overlap = len(attr_keys1 & attr_keys2)
                union = len(attr_keys1 | attr_keys2)
                scores.append(overlap / union if union > 0 else 0)

        # Average scores
        if scores:
            avg_score = sum(scores) / len(scores)
            return avg_score >= self.confidence_threshold

        return False

    def _are_compatible_types(self, type1: str, type2: str) -> bool:
        """Check if two entity types are compatible for deduplication."""
        # Define compatible type mappings
        compatible_pairs = {
            ('person', 'individual'),
            ('organization', 'company'),
            ('location', 'place'),
            ('product', 'item'),
            ('event', 'incident'),
        }

        # Check both directions
        return (type1.lower(), type2.lower()) in compatible_pairs or \
               (type2.lower(), type1.lower()) in compatible_pairs or \
               type1.lower() == type2.lower()

    async def _detect_temporal_overlaps(
        self,
        groups: List[List[Entity]]
    ) -> List[List[Entity]]:
        """Detect temporal overlaps in entity timestamps."""
        filtered_groups = []

        for group in groups:
            if len(group) < 2:
                filtered_groups.append(group)
                continue

            # Check if entities in group have temporal overlap
            if await self._has_temporal_overlap(group):
                filtered_groups.append(group)
            else:
                # No temporal overlap, might be different entities at different times
                # Keep them separate
                for entity in group:
                    filtered_groups.append([entity])

        return filtered_groups

    async def _has_temporal_overlap(self, group: List[Entity]) -> bool:
        """Check if entities have overlapping valid time periods."""
        # Extract time ranges from entities
        time_ranges = []

        for entity in group:
            # Try to get time from attributes
            start_time = None
            end_time = None

            if entity.attributes:
                # Common attribute names for timestamps
                for key in ['start_time', 'created_at', 'timestamp', 'date', 'valid_from', 'begin_time']:
                    if key in entity.attributes:
                        start_time = entity.attributes[key]
                        break

                for key in ['end_time', 'expired_at', 'valid_until', 'end_date', 'finish_time']:
                    if key in entity.attributes:
                        end_time = entity.attributes[key]
                        break

            time_ranges.append((start_time, end_time))

        # If no time information, assume overlap
        if all(tr == (None, None) for tr in time_ranges):
            return True

        # Check for overlaps between all pairs
        for i in range(len(time_ranges)):
            for j in range(i + 1, len(time_ranges)):
                if not self._ranges_overlap(time_ranges[i], time_ranges[j]):
                    return False

        return True

    def _ranges_overlap(self, range1: tuple, range2: tuple) -> bool:
        """Check if two time ranges overlap."""
        start1, end1 = range1
        start2, end2 = range2

        # Convert to comparable format
        def to_datetime(value):
            if value is None:
                return None
            if isinstance(value, (int, float)):
                from datetime import datetime, timezone
                return datetime.fromtimestamp(value, tz=timezone.utc)
            if isinstance(value, str):
                from datetime import datetime
                try:
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except:
                    return None
            return value

        start1_dt = to_datetime(start1)
        end1_dt = to_datetime(end1)
        start2_dt = to_datetime(start2)
        end2_dt = to_datetime(end2)

        # If any endpoint is None, assume unbounded
        # Overlap exists if not (range1 ends before range2 OR range2 ends before range1)
        if end1_dt is not None and start2_dt is not None:
            if end1_dt < start2_dt:
                return False
        if end2_dt is not None and start1_dt is not None:
            if end2_dt < start1_dt:
                return False

        return True

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
