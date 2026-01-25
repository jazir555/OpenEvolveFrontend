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
        """Use LLM to resolve ambiguous duplicates."""
        # In a full implementation, this would call an LLM
        # For now, just return the groups as-is
        logger.info("LLM-assisted resolution not implemented, returning original groups")
        return groups

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
