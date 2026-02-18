"""
Entity Standardization Deduplication Strategy (ai-knowledge-graph)

Entity normalization using:
1. Text normalization
2. Frequency-based grouping
3. Root word analysis
4. Subset detection for hierarchical relationships
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import logging

from ..base import Entity, DeduplicationResult, DeduplicationStrategy

logger = logging.getLogger(__name__)


class EntityStandardizationStrategy(DeduplicationStrategy):
    """
    Entity standardization deduplication strategy.

    Best for:
    - Medium datasets (100-1000 entities)
    - Entity normalization needs
    - Hierarchical entity detection
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.use_llm_for_resolution = self.config.get('use_llm_for_resolution', False)
        self.stem_length = self.config.get('stem_length', 4)

    def get_strategy_name(self) -> str:
        return "standardization"

    async def deduplicate(
        self,
        entities: List[Entity],
        context: Optional[Dict[str, Any]] = None
    ) -> DeduplicationResult:
        """
        Deduplicate entities using standardization approach.

        Process:
        1. Normalize entity names
        2. Frequency-based grouping
        3. Root word analysis
        4. Subset detection
        """
        logger.info(f"Starting entity standardization for {len(entities)} entities")

        # Step 1: Normalize and group by exact match
        exact_groups = await self._group_by_exact_match(entities)

        # Step 2: Root word analysis
        root_groups = await self._group_by_root_words(exact_groups)

        # Step 3: Subset detection
        duplicate_groups = await self._detect_subsets(root_groups)

        # Step 4: LLM-assisted resolution (optional)
        if self.use_llm_for_resolution:
            duplicate_groups = await self._llm_assisted_resolution(duplicate_groups)

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
                'exact_groups': len(exact_groups),
                'root_groups': len(root_groups)
            }
        )

    async def _group_by_exact_match(
        self,
        entities: List[Entity]
    ) -> List[List[Entity]]:
        """Group entities by exact normalized name match."""
        groups_dict = defaultdict(list)

        for entity in entities:
            normalized = self._normalize_entity_name(entity.name)
            groups_dict[normalized].append(entity)

        # Return groups with more than one entity
        return [g for g in groups_dict.values() if len(g) > 1]

    async def _group_by_root_words(
        self,
        groups: List[List[Entity]]
    ) -> List[List[Entity]]:
        """Group entities by root word analysis."""
        merged_groups = []

        # Create root word map
        root_map = defaultdict(list)

        for group in groups:
            # Get root words from first entity in group
            entity = group[0]
            roots = self._extract_root_words(entity.name)

            for root in roots:
                root_map[root].append(group)

        # Merge groups with shared roots
        processed = set()

        for root, root_groups in root_map.items():
            if len(root_groups) > 1:
                # Merge these groups
                merged = []
                seen_ids = set()

                for g in root_groups:
                    for entity in g:
                        if entity.id not in seen_ids:
                            merged.append(entity)
                            seen_ids.add(entity.id)

                if len(merged) > 1:
                    merged_groups.append(merged)

        # Add groups that weren't merged
        for group in groups:
            group_id = id(group)
            if group_id not in processed:
                merged_groups.append(group)

        return merged_groups

    async def _detect_subsets(
        self,
        groups: List[List[Entity]]
    ) -> List[List[Entity]]:
        """Detect subset relationships (hierarchical entities)."""
        duplicate_groups = []

        for group in groups:
            if len(group) < 2:
                duplicate_groups.append(group)
                continue

            # Check for subset relationships
            subsets = self._find_subsets(group)

            if subsets:
                # Merge subsets
                merged = []
                seen_ids = set()

                for subset in subsets:
                    for entity in subset:
                        if entity.id not in seen_ids:
                            merged.append(entity)
                            seen_ids.add(entity.id)

                duplicate_groups.append(merged)
            else:
                duplicate_groups.append(group)

        return duplicate_groups

    def _find_subsets(
        self,
        entities: List[Entity]
    ) -> List[List[Entity]]:
        """Find entities where one name is a subset of another."""
        subsets = []

        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1:]:
                name1 = entity1.name.lower()
                name2 = entity2.name.lower()

                # Check if one is subset of another
                if name1 in name2 or name2 in name1:
                    subsets.append([entity1, entity2])

        return subsets

    async def _llm_assisted_resolution(
        self,
        groups: List[List[Entity]]
    ) -> List[List[Entity]]:
        """
        Use LLM to resolve ambiguous duplicates.

        Makes intelligent decisions about whether entities in a group
        represent the same real-world entity or different entities.
        """
        resolved_groups = []

        for group in groups:
            if len(group) < 2:
                resolved_groups.append(group)
                continue

            # Try LLM-based resolution
            should_merge = await self._llm_merge_decision(group)

            if should_merge:
                # Entities are duplicates, keep them together
                resolved_groups.append(group)
            else:
                # Entities are distinct, separate them
                for entity in group:
                    resolved_groups.append([entity])

        return resolved_groups

    async def _llm_merge_decision(self, group: List[Entity]) -> bool:
        """
        Ask LLM whether entities should be merged.

        Returns True if entities represent the same real-world entity.
        """
        # Check if LLM is configured
        openai_key = self.config.get('openai_api_key')
        if not openai_key:
            # No LLM available, use heuristic
            return self._heuristic_merge_decision(group)

        try:
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

            prompt = f"""You are an expert in knowledge graph deduplication. Analyze these entities and determine if they represent the SAME real-world entity.

{chr(10).join(entity_descriptions)}

Consider these factors in your decision:
1. Name similarity (including variations, aliases, abbreviations)
2. Type compatibility (Person vs Individual, Organization vs Company)
3. Attribute overlap (locations, dates, identifiers)
4. Description semantics
5. Temporal consistency (do they exist at the same time?)

Important: These entities MIGHT be duplicates that were extracted from different sources or at different times.

Respond with ONLY 'true' if they should be merged as duplicates, or 'false' if they are distinct entities."""

            # Call OpenAI API
            import openai
            client = openai.AsyncClient(api_key=openai_key)

            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )

            result = response.choices[0].message.content.strip().lower()
            return 'true' in result or 'yes' in result

        except Exception as e:
            logger.warning(f"LLM merge decision failed: {e}, falling back to heuristics")
            return self._heuristic_merge_decision(group)

    def _heuristic_merge_decision(self, group: List[Entity]) -> bool:
        """
        Heuristic-based merge decision when LLM unavailable.

        Multi-factor scoring approach.
        """
        if len(group) < 2:
            return False

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                sim = self._calculate_pairwise_similarity(group[i], group[j])
                similarities.append(sim)

        # Average similarity
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)

            # Decision threshold
            # Higher threshold for standardization (stricter)
            return avg_similarity >= self.merge_threshold

        return False

    def _calculate_pairwise_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate similarity between two entities (multi-factor)."""
        scores = []
        weights = []

        # 1. Name similarity (weight: 0.4)
        name1_norm = self._normalize_entity_name(entity1.name)
        name2_norm = self._normalize_entity_name(entity2.name)

        if name1_norm and name2_norm:
            if name1_norm == name2_norm:
                name_sim = 1.0
            else:
                # Jaccard similarity
                words1 = set(name1_norm.split())
                words2 = set(name2_norm.split())
                if words1 and words2:
                    overlap = len(words1 & words2)
                    union = len(words1 | words2)
                    name_sim = overlap / union if union > 0 else 0
                else:
                    name_sim = 0.0
            scores.append(name_sim)
            weights.append(0.4)

        # 2. Type compatibility (weight: 0.3)
        if entity1.entity_type == entity2.entity_type:
            type_sim = 1.0
        else:
            # Check compatible types
            compatible = {
                ('person', 'individual'),
                ('organization', 'company'),
                ('location', 'place'),
            }
            type_sim = 1.0 if (entity1.entity_type.lower(), entity2.entity_type.lower()) in compatible else 0.0

        scores.append(type_sim)
        weights.append(0.3)

        # 3. Attribute overlap (weight: 0.3)
        if entity1.attributes and entity2.attributes:
            keys1 = set(entity1.attributes.keys())
            keys2 = set(entity2.attributes.keys())

            if keys1 and keys2:
                overlap = len(keys1 & keys2)
                union = len(keys1 | keys2)
                attr_sim = overlap / union if union > 0 else 0
                scores.append(attr_sim)
                weights.append(0.3)

        # Weighted average
        if scores and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                return sum(s * w for s, w in zip(scores, weights)) / total_weight

        return 0.0

    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity name for comparison.

        1. Lowercase
        2. Remove punctuation
        3. Remove extra whitespace
        4. Remove special characters
        """
        # Lowercase
        name = name.lower()

        # Remove punctuation
        name = re.sub(r'[^\w\s]', ' ', name)

        # Remove extra whitespace
        name = ' '.join(name.split())

        return name

    def _extract_root_words(self, name: str) -> Set[str]:
        """
        Extract root words for grouping.

        Uses first N characters of each word (default: 4)
        """
        normalized = self._normalize_entity_name(name)
        words = normalized.split()

        roots = set()
        for word in words:
            if len(word) >= self.stem_length:
                root = word[:self.stem_length]
                roots.add(root)

        return roots

    def calculate_confidence(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate confidence based on name overlap and root words."""
        name1 = self._normalize_entity_name(entity1.name)
        name2 = self._normalize_entity_name(entity2.name)

        # Exact match
        if name1 == name2:
            return 1.0

        # Subset relationship
        if name1 in name2 or name2 in name1:
            return 0.9

        # Root word overlap
        roots1 = self._extract_root_words(entity1.name)
        roots2 = self._extract_root_words(entity2.name)

        if roots1 and roots2:
            overlap = len(roots1 & roots2)
            union = len(roots1 | roots2)

            if union > 0:
                return overlap / union

        return 0.0
