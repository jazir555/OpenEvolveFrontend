"""
AI-Knowledge-Graph Entity Standardization Integration

This module integrates ai-knowledge-graph's entity standardization pipeline,
providing multi-level entity resolution and deduplication capabilities.
"""

import asyncio
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

import networkx as nx

logger = logging.getLogger(__name__)


class Entity:
    """Represents a knowledge graph entity."""

    def __init__(
        self,
        name: str,
        entity_type: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.entity_type = entity_type
        self.attributes = attributes or {}
        self.variants: List[str] = []
        self.canonical: Optional[str] = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.name == other.name
        return False

    def __repr__(self):
        return f"Entity({self.name})"


class Triple:
    """Represents a knowledge graph triple (subject, predicate, object)."""

    def __init__(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        source: str = "extracted"
    ):
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.confidence = confidence
        self.source = source  # "extracted" or "inferred"

    def to_tuple(self) -> Tuple[str, str, str]:
        """Convert to tuple format."""
        return (self.subject, self.predicate, self.object)

    def __eq__(self, other):
        if isinstance(other, Triple):
            return self.to_tuple() == other.to_tuple()
        return False

    def __hash__(self):
        return hash(self.to_tuple())

    def __repr__(self):
        return f"Triple({self.subject}, {self.predicate}, {self.object})"


class StandardizationResult:
    """Result of entity standardization process."""

    def __init__(
        self,
        canonical_entities: List[Entity],
        variant_mappings: Dict[str, List[str]],
        removed_self_refs: int,
        statistics: Dict[str, Any]
    ):
        self.canonical_entities = canonical_entities
        self.variant_mappings = variant_mappings  # canonical -> [variants]
        self.removed_self_refs = removed_self_refs
        self.statistics = statistics
        self.timestamp = datetime.now().isoformat()


class AIKGEntityStandardizer:
    """
    Integrates ai-knowledge-graph's entity standardization pipeline.

    Features:
    - Multi-level standardization
    - Frequency-based grouping
    - Root word analysis
    - LLM-assisted resolution
    - Self-reference filtering
    """

    # Common stopwords for entity normalization
    DEFAULT_STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the entity standardizer.

        Args:
            config: Configuration dictionary with options:
                - use_llm_for_entities: Whether to use LLM for entity resolution
                - stopword_removal: Whether to remove stopwords
                - root_word_analysis: Whether to use root word analysis
                - self_reference_filtering: Whether to filter self-references
                - llm_client: Optional LLM client for advanced resolution
        """
        self.use_llm = config.get('use_llm_for_entities', False)
        self.stopword_removal = config.get('stopword_removal', True)
        self.root_word_analysis = config.get('root_word_analysis', True)
        self.self_reference_filtering = config.get('self_reference_filtering', True)
        self.llm_client = config.get('llm_client')

        self.stopwords = self._load_stopwords()
        self.variant_mappings: Dict[str, Set[str]] = defaultdict(set)

        logger.info(f"AIKGEntityStandardizer initialized with LLM: {self.use_llm}")

    def _load_stopwords(self) -> Set[str]:
        """Load stopwords for text normalization."""
        stopwords = self.DEFAULT_STOPWORDS.copy()
        # Add domain-specific stopwords if needed
        return stopwords

    async def standardize_entities(
        self,
        entities: List[Entity],
        triples: List[Triple]
    ) -> StandardizationResult:
        """
        Standardize entities using multi-level approach.

        Levels:
        1. Text normalization (lowercasing, stopword removal)
        2. Frequency-based grouping
        3. Root word analysis (4-char prefix matching)
        4. LLM-assisted resolution (optional)
        5. Self-reference filtering

        Args:
            entities: List of entities to standardize
            triples: List of triples for context

        Returns:
            StandardizationResult with canonical entities and mappings
        """
        logger.info(f"Starting standardization for {len(entities)} entities")

        stats = {
            'original_entities': len(entities),
            'original_triples': len(triples),
            'duplicates_found': 0,
            'variants_resolved': 0
        }

        # Level 1: Text normalization
        normalized_entities = await self._normalize_entities(entities)

        # Level 2: Frequency-based grouping
        frequency_groups = await self.group_by_frequency(normalized_entities)
        logger.info(f"Frequency grouping produced {len(frequency_groups)} groups")

        # Level 3: Root word analysis
        if self.root_word_analysis:
            root_groups = await self.analyze_root_words(normalized_entities)
            # Merge root groups with frequency groups
            frequency_groups = self._merge_groups(frequency_groups, root_groups)
            logger.info(f"After root word analysis: {len(frequency_groups)} groups")

        # Level 4: LLM-assisted resolution (optional)
        if self.use_llm and self.llm_client:
            canonical_entities = await self.resolve_with_llm(frequency_groups)
        else:
            # Select most frequent entity as canonical
            canonical_entities = self._select_canonical_entities(frequency_groups)

        stats['canonical_entities'] = len(canonical_entities)
        stats['duplicates_found'] = stats['original_entities'] - stats['canonical_entities']

        # Build variant mappings
        variant_mappings = await self._build_variant_mappings(
            canonical_entities, entities
        )
        stats['variants_resolved'] = sum(len(v) for v in variant_mappings.values())

        # Level 5: Filter self-references
        filtered_triples = triples
        if self.self_reference_filtering:
            filtered_triples = await self.filter_self_references(triples)
            stats['self_references_removed'] = len(triples) - len(filtered_triples)
        else:
            stats['self_references_removed'] = 0

        # Track variant mappings
        for canonical, variants in variant_mappings.items():
            await self.track_variants(
                next(e for e in canonical_entities if e.name == canonical),
                [Entity(v) for v in variants]
            )

        logger.info(
            f"Standardization complete: {stats['original_entities']} -> "
            f"{stats['canonical_entities']} entities"
        )

        return StandardizationResult(
            canonical_entities=canonical_entities,
            variant_mappings={k: list(v) for k, v in variant_mappings.items()},
            removed_self_refs=stats.get('self_references_removed', 0),
            statistics=stats
        )

    async def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        Steps:
        1. Unicode normalization (NFKC)
        2. Lowercase conversion
        3. Remove stopwords
        4. Remove special characters
        5. Normalize whitespace

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Step 1: Unicode normalization
        text = unicodedata.normalize('NFKC', text)

        # Step 2: Lowercase
        text = text.lower()

        # Step 3: Remove stopwords (if enabled)
        if self.stopword_removal:
            words = text.split()
            words = [w for w in words if w not in self.stopwords]
            text = ' '.join(words)

        # Step 4: Remove special characters (keep alphanumeric and spaces)
        text = re.sub(r'[^\w\s]', ' ', text)

        # Step 5: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    async def _normalize_entities(self, entities: List[Entity]) -> List[Entity]:
        """Normalize all entity names."""
        normalized = []
        for entity in entities:
            normalized_name = await self.normalize_text(entity.name)
            # Create new entity with normalized name
            normalized_entity = Entity(
                name=normalized_name,
                entity_type=entity.entity_type,
                attributes=entity.attributes.copy()
            )
            # Store original name as variant
            normalized_entity.variants = [entity.name]
            normalized.append(normalized_entity)
        return normalized

    async def group_by_frequency(
        self,
        entities: List[Entity]
    ) -> Dict[str, List[Entity]]:
        """
        Group entities by frequency patterns.

        Process:
        1. Count entity occurrences (including variants)
        2. Group by identical normalized names
        3. Return grouped candidates

        Args:
            entities: List of normalized entities

        Returns:
            Dictionary mapping normalized name to list of entities
        """
        groups = defaultdict(list)

        # Group by exact match after normalization
        for entity in entities:
            groups[entity.name].append(entity)

        # Convert to regular dict
        return dict(groups)

    async def analyze_root_words(
        self,
        entities: List[Entity]
    ) -> Dict[str, List[Entity]]:
        """
        Analyze entities by root words (4-char prefix).

        Examples:
        - "capitalism" and "capitalist" -> grouped
        - "programming" and "programmer" -> grouped

        Args:
            entities: List of entities to analyze

        Returns:
            Dictionary mapping root word to list of entities
        """
        root_groups = defaultdict(list)

        for entity in entities:
            # Extract 4-character prefix (or shorter if entity name is short)
            root = entity.name[:4] if len(entity.name) >= 4 else entity.name
            root_groups[root].append(entity)

        return dict(root_groups)

    def _merge_groups(
        self,
        freq_groups: Dict[str, List[Entity]],
        root_groups: Dict[str, List[Entity]]
    ) -> Dict[str, List[Entity]]:
        """Merge frequency groups with root word groups."""
        merged = defaultdict(list)

        # Start with frequency groups
        for key, entities in freq_groups.items():
            merged[key].extend(entities)

        # Add root word group insights
        # (Only add if entities are not already in a group)
        seen = set()
        for entities in freq_groups.values():
            for entity in entities:
                seen.add(entity.name)

        for root, entities in root_groups.items():
            if len(entities) > 1:
                # Find if any of these entities are already grouped
                for entity in entities:
                    if entity.name not in seen:
                        merged[root].append(entity)
                        seen.add(entity.name)

        return dict(merged)

    def _select_canonical_entities(
        self,
        groups: Dict[str, List[Entity]]
    ) -> List[Entity]:
        """
        Select canonical entity from each group.

        Strategy: Select entity with most variants (highest frequency).
        """
        canonical_entities = []

        for key, entities in groups.items():
            if not entities:
                continue

            # Select entity with most variants as canonical
            canonical = max(entities, key=lambda e: len(e.variants))

            # Aggregate all variants
            all_variants = []
            for entity in entities:
                all_variants.extend(entity.variants)

            canonical.variants = list(set(all_variants))
            canonical_entities.append(canonical)

        return canonical_entities

    async def resolve_with_llm(
        self,
        entity_groups: Dict[str, List[Entity]]
    ) -> List[Entity]:
        """
        Use LLM to resolve ambiguous entity mappings.

        Args:
            entity_groups: Groups of potentially duplicate entities

        Returns:
            List of canonical entities
        """
        if not self.llm_client:
            logger.warning("LLM resolution requested but no LLM client available")
            return self._select_canonical_entities(entity_groups)

        canonical_entities = []

        for group_key, entities in entity_groups.items():
            if len(entities) <= 1:
                canonical_entities.extend(entities)
                continue

            # Prepare LLM prompt
            entity_names = [e.name for e in entities]
            prompt = self._build_resolution_prompt(entity_names)

            try:
                # Call LLM
                response = await self.llm_client(prompt)
                selected = self._parse_llm_response(response, entity_names)

                # Create canonical entity
                canonical = Entity(
                    name=selected,
                    entity_type=entities[0].entity_type,
                    attributes=entities[0].attributes.copy()
                )
                canonical.variants = list(set(
                    v for e in entities for v in e.variants
                ))
                canonical_entities.append(canonical)

            except Exception as e:
                logger.error(f"LLM resolution failed for group {group_key}: {e}")
                # Fallback to frequency-based selection
                canonical_entities.extend(self._select_canonical_entities({group_key: entities}))

        return canonical_entities

    def _build_resolution_prompt(self, entity_names: List[str]) -> str:
        """Build prompt for LLM entity resolution."""
        return f"""Given the following entity names that may refer to the same concept:

{chr(10).join(f"- {name}" for name in entity_names)}

Select the most canonical/standard name from this list. The canonical name should be:
1. Most commonly used in academic/technical literature
2. Most descriptive and clear
3. Standard terminology in the domain

Return only the selected canonical name, nothing else."""

    def _parse_llm_response(self, response: str, candidates: List[str]) -> str:
        """Parse LLM response and validate against candidates."""
        response = response.strip()

        # Direct match
        if response in candidates:
            return response

        # Fuzzy match
        for candidate in candidates:
            if response.lower() in candidate.lower() or candidate.lower() in response.lower():
                return candidate

        # Default to first candidate if no match
        logger.warning(f"LLM response '{response}' not in candidates, using first")
        return candidates[0]

    async def filter_self_references(
        self,
        triples: List[Triple]
    ) -> List[Triple]:
        """
        Remove triples where subject = object.

        Example to remove:
        - (Python, related_to, Python)  # Remove
        - (Python, similar_to, Java)     # Keep

        Args:
            triples: List of triples to filter

        Returns:
            Filtered list of triples
        """
        filtered = []
        for triple in triples:
            # Normalize for comparison
            subj_norm = await self.normalize_text(triple.subject)
            obj_norm = await self.normalize_text(triple.object)

            if subj_norm != obj_norm:
                filtered.append(triple)

        removed = len(triples) - len(filtered)
        if removed > 0:
            logger.info(f"Removed {removed} self-referential triples")

        return filtered

    async def _build_variant_mappings(
        self,
        canonical_entities: List[Entity],
        original_entities: List[Entity]
    ) -> Dict[str, List[str]]:
        """Build mappings from canonical names to variant names."""
        mappings = {}

        for canonical in canonical_entities:
            # Use stored variants
            mappings[canonical.name] = canonical.variants.copy()

        return mappings

    async def track_variants(
        self,
        canonical: Entity,
        variants: List[Entity]
    ):
        """
        Track canonical-to-variant mappings.

        Args:
            canonical: The canonical entity
            variants: List of variant entities
        """
        for variant in variants:
            self.variant_mappings[canonical.name].add(variant.name)

    def get_variants(self, canonical_name: str) -> List[str]:
        """Get all variants for a canonical entity."""
        return list(self.variant_mappings.get(canonical_name, set()))

    def resolve_to_canonical(self, entity_name: str) -> Optional[str]:
        """Resolve a variant name to its canonical form."""
        normalized = entity_name.lower()

        # Check direct mappings
        for canonical, variants in self.variant_mappings.items():
            if normalized in [v.lower() for v in variants]:
                return canonical

        # Check if already canonical
        if normalized in [c.lower() for c in self.variant_mappings.keys()]:
            for canonical in self.variant_mappings.keys():
                if canonical.lower() == normalized:
                    return canonical

        return None
