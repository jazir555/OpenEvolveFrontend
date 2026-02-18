"""
SEMHASH Deduplication Strategy (kg-gen)

Fast rule-based deduplication using:
1. Unicode normalization (NFKC)
2. Lowercasing and stopword removal
3. Singularization
4. Deterministic similarity matching
"""

import unicodedata
import re
from typing import List, Dict, Any, Optional, Set
import logging

from ..base import Entity, DeduplicationResult, DeduplicationStrategy

logger = logging.getLogger(__name__)


class SemHashStrategy(DeduplicationStrategy):
    """
    Fast rule-based deduplication strategy.

    Best for:
    - Small datasets (< 100 entities)
    - Fast deduplication needs
    - Exact/near-exact duplicate detection
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.95)
        self.use_llm_for_entities = self.config.get('use_llm_for_entities', False)

        # Common stopwords
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> Set[str]:
        """Load common stopwords for removal."""
        return {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is',
            'was', 'are', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'it', 'its'
        }

    def get_strategy_name(self) -> str:
        return "semhash"

    async def deduplicate(
        self,
        entities: List[Entity],
        context: Optional[Dict[str, Any]] = None
    ) -> DeduplicationResult:
        """
        Deduplicate entities using SEMHASH approach.

        Process:
        1. Normalize entity names (NFKC, lowercase, remove stopwords)
        2. Calculate similarity matrix
        3. Group duplicates above threshold
        4. Return canonical entities
        """
        start_time = self.logger.info(f"Starting SEMHASH deduplication for {len(entities)} entities")

        # Preprocess entities
        normalized = await self.preprocess_entities(entities)

        # Create normalized name map
        name_map = {}
        for entity in normalized:
            norm_name = self._normalize_text(entity.name)
            name_map[entity.id] = {
                'normalized': norm_name,
                'original': entity
            }

        # Find duplicate groups
        duplicate_groups = []
        processed_ids = set()

        for entity1 in normalized:
            if entity1.id in processed_ids:
                continue

            group = [entity1]
            norm_name1 = name_map[entity1.id]['normalized']

            for entity2 in normalized:
                if entity2.id in processed_ids or entity2.id == entity1.id:
                    continue

                norm_name2 = name_map[entity2.id]['normalized']

                # Calculate similarity
                similarity = self._calculate_similarity(norm_name1, norm_name2)

                if similarity >= self.similarity_threshold:
                    group.append(entity2)
                    processed_ids.add(entity2.id)

            processed_ids.add(entity1.id)

            if len(group) > 1:
                duplicate_groups.append(group)

        # Create canonical entities
        canonical_entities = []
        for entity in normalized:
            is_duplicate = False
            for group in duplicate_groups:
                if entity.id in [e.id for e in group[1:]]:  # Not the first
                    is_duplicate = True
                    break

            if not is_duplicate:
                canonical_entities.append(entity)

        return DeduplicationResult(
            canonical_entities=canonical_entities,
            duplicate_groups=duplicate_groups,
            stats={
                'original_count': len(entities),
                'canonical_count': len(canonical_entities),
                'duplicate_groups': len(duplicate_groups),
                'similarity_threshold': self.similarity_threshold
            }
        )

    async def preprocess_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Normalize entity text for comparison.

        Performs advanced text normalization:
        1. Singularization (with inflect if available)
        2. Unicode normalization
        3. Stopword removal
        4. Abbreviation expansion
        5. Whitespace normalization
        """
        processed = []

        # Try to load inflect for singularization
        inflect = None
        try:
            import inflect
            inflect_engine = inflect.engine()
        except ImportError:
            inflect_engine = None

        # Load abbreviation mapping
        abbreviations = self._get_abbreviation_map()

        for entity in entities:
            # Create a copy with normalized name
            normalized_entity = Entity(
                id=entity.id,
                name=self._normalize_entity_text(entity.name, inflect_engine, abbreviations),
                entity_type=entity.entity_type,
                description=entity.description,
                attributes=entity.attributes,
                confidence=entity.confidence
            )
            processed.append(normalized_entity)

        return processed

    def _normalize_entity_text(
        self,
        text: str,
        inflect_engine=None,
        abbreviations: Dict[str, str] = None
    ) -> str:
        """
        Fully normalize entity text.

        Args:
            text: Original text
            inflect_engine: Optional inflect engine for singularization
            abbreviations: Optional abbreviation expansion map

        Returns:
            Normalized text
        """
        if not text:
            return text

        # 1. Expand abbreviations
        if abbreviations:
            words = text.split()
            expanded = []
            for word in words:
                lower_word = word.lower()
                if lower_word in abbreviations:
                    expanded.append(abbreviations[lower_word])
                else:
                    expanded.append(word)
            text = ' '.join(expanded)

        # 2. Singularize with inflect if available
        if inflect_engine:
            try:
                words = text.split()
                singularized = []
                for word in words:
                    # Check if it's a plural noun
                    if inflect_engine.singular_noun(word):
                        singularized.append(inflect_engine.singular_noun(word))
                    else:
                        singularized.append(word)
                text = ' '.join(singularized)
            except Exception:
                pass  # Fall back to original if singularization fails

        # 3. Apply base normalization
        text = self._normalize_text(text)

        return text

    def _get_abbreviation_map(self) -> Dict[str, str]:
        """Get common abbreviation expansion map."""
        return {
            # Business/organizations
            'corp': 'corporation',
            'inc': 'incorporated',
            'llc': 'limited liability company',
            'ltd': 'limited',
            'co': 'company',
            'dept': 'department',
            'div': 'division',
            'assn': 'association',
            'inst': 'institute',

            # Technology
            'tech': 'technology',
            'sys': 'system',
            'app': 'application',
            'prog': 'program',
            'dev': 'development',
            'eng': 'engineering',
            'mgr': 'manager',
            'rep': 'representative',

            # Common terms
            'info': 'information',
            'doc': 'document',
            'msg': 'message',
            'req': 'request',
            'resp': 'response',
            'config': 'configuration',
            'param': 'parameter',
            'arg': 'argument',

            # Locations
            'ave': 'avenue',
            'st': 'street',
            'blvd': 'boulevard',
            'rd': 'road',
            'mt': 'mount',
            'ft': 'fort',
            'univ': 'university',
            'dept': 'department',

            # Time periods
            'qtr': 'quarter',
            'fy': 'fiscal year',
            'yoy': 'year over year',
            'mom': 'month over month',
        }

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text using SEMHASH approach.

        1. Unicode NFKC normalization
        2. Lowercase
        3. Remove punctuation
        4. Remove stopwords
        5. Remove extra whitespace
        """
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)

        # Lowercase
        text = text.lower()

        # Remove punctuation (keep alphanumeric and spaces)
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]

        # Rejoin
        return ' '.join(tokens)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two normalized texts.

        Uses a hybrid approach:
        - Jaccard similarity for word overlap
        - Sequence matching for ordering
        """
        if not text1 or not text2:
            return 0.0

        # Tokenize
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        if union == 0:
            return 0.0

        jaccard = intersection / union

        # Add sequence matching bonus
        sequence_bonus = self._sequence_similarity(text1, text2)

        # Weighted combination
        return 0.7 * jaccard + 0.3 * sequence_bonus

    def _sequence_similarity(self, text1: str, text2: str) -> float:
        """Calculate sequence similarity using difflib."""
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, text1, text2).ratio()
        except Exception as e:
            self.logger.warning(f"Sequence matching failed: {e}")
            return 0.0

    def calculate_confidence(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate confidence score for entity similarity."""
        norm_name1 = self._normalize_text(entity1.name)
        norm_name2 = self._normalize_text(entity2.name)
        return self._calculate_similarity(norm_name1, norm_name2)
