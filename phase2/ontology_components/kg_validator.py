"""
Knowledge Graph Validator

Validate ontology mappings using external knowledge graphs (ConceptNet, WordNet).

Agent: G2 (Ψ₂ Specialist)
Created: 2025-12-31
"""

import time
import logging
from typing import Dict, Optional, Tuple
import sqlite3

logger = logging.getLogger(__name__)


class KGValidator:
    """
    Knowledge graph validator for ontology mappings.

    Queries external KGs to validate concept relationships.
    """

    def __init__(
        self,
        cache_size: int = 10000,
        timeout: float = 5.0,
        use_conceptnet: bool = True,
        use_wordnet: bool = True
    ):
        """
        Initialize KG validator

        Args:
            cache_size: Maximum cache size
            timeout: API timeout in seconds
            use_conceptnet: Use ConceptNet API
            use_wordnet: Use WordNet
        """
        self.cache_size = cache_size
        self.timeout = timeout
        self.use_conceptnet = use_conceptnet
        self.use_wordnet = use_wordnet

        # ConceptNet endpoint
        self.conceptnet_api = "http://api.conceptnet.io"

        # WordNet (lazy loading)
        self.wordnet = None

        # Initialize
        self._initialize_wordnet()

    def _initialize_wordnet(self):
        """Initialize WordNet"""
        if not self.use_wordnet:
            return

        try:
            import nltk
            from nltk.corpus import wordnet as wn

            # Download WordNet if not available
            try:
                wn.synsets('test')
            except:
                logger.info("Downloading WordNet...")
                nltk.download('wordnet')
                nltk.download('omw-1.4')

            self.wordnet = wn
            logger.info("WordNet initialized")

        except ImportError:
            logger.warning("nltk not installed, WordNet unavailable")
            self.use_wordnet = False

    def validate_relation(
        self,
        concept1: str,
        concept2: str
    ) -> Optional[float]:
        """
        Validate relationship between two concepts

        Args:
            concept1: First concept
            concept2: Second concept

        Returns:
            Confidence score [0, 1] or None if no evidence
        """
        scores = []

        # Query ConceptNet
        if self.use_conceptnet:
            conceptnet_score = self._query_conceptnet(concept1, concept2)
            if conceptnet_score is not None:
                scores.append(conceptnet_score)

        # Query WordNet
        if self.use_wordnet and self.wordnet:
            wordnet_score = self._query_wordnet(concept1, concept2)
            if wordnet_score is not None:
                scores.append(wordnet_score)

        # Combine scores
        if not scores:
            return None

        # Return maximum score
        return max(scores)

    def _query_conceptnet(
        self,
        concept1: str,
        concept2: str
    ) -> Optional[float]:
        """
        Query ConceptNet for relationship

        Args:
            concept1: First concept
            concept2: Second concept

        Returns:
            Confidence score or None
        """
        try:
            import requests

            # Normalize concepts
            c1_normalized = concept1.lower().replace(' ', '_')
            c2_normalized = concept2.lower().replace(' ', '_')

            # Query ConceptNet API
            url = f"{self.conceptnet_api}/c/en/{c1_normalized}"
            params = {
                'limit': 100,
                'offset': 0
            }

            response = requests.get(url, params=params, timeout=self.timeout)

            if response.status_code != 200:
                return None

            data = response.json()

            # Search for relationship to concept2
            for edge in data.get('edges', []):
                end_label = edge.get('end', {}).get('label', '').lower()
                relation = edge.get('rel', {}).get('label', '')
                weight = edge.get('weight', 0.0)

                # Check if this is about concept2
                if c2_normalized.replace('_', ' ') in end_label:
                    # Boost score based on relationship type
                    boost = self._relation_boost(relation)
                    return boost * weight

            return None

        except Exception as e:
            logger.warning(f"ConceptNet query failed: {e}")
            return None

    def _query_wordnet(
        self,
        concept1: str,
        concept2: str
    ) -> Optional[float]:
        """
        Query WordNet for relationship

        Args:
            concept1: First concept
            concept2: Second concept

        Returns:
            Confidence score or None
        """
        try:
            # Get synsets
            synsets1 = self.wordnet.synsets(concept1)
            synsets2 = self.wordnet.synsets(concept2)

            if not synsets1 or not synsets2:
                return None

            # Compute path similarity
            best_score = 0.0

            for s1 in synsets1:
                for s2 in synsets2:
                    # Path similarity
                    path_sim = self.wordnet.path_similarity(s1, s2)
                    if path_sim is not None:
                        best_score = max(best_score, path_sim)

                    # Wu-Palmer similarity
                    wup_sim = self.wordnet.wup_similarity(s1, s2)
                    if wup_sim is not None:
                        best_score = max(best_score, wup_sim)

            return best_score if best_score > 0 else None

        except Exception as e:
            logger.warning(f"WordNet query failed: {e}")
            return None

    def _relation_boost(self, relation: str) -> float:
        """
        Get confidence boost based on relationship type

        Args:
            relation: Relationship type from ConceptNet

        Returns:
            Boost multiplier
        """
        # Strong relationships
        if relation in ['Synonym', 'Antonym', 'SimilarTo']:
            return 1.0

        # Moderate relationships
        elif relation in ['RelatedTo', 'IsA', 'PartOf', 'HasProperty']:
            return 0.8

        # Weak relationships
        elif relation in ['UsedFor', 'Causes', 'CreatedBy']:
            return 0.6

        # Default
        else:
            return 0.5

    def get_related_concepts(
        self,
        concept: str,
        limit: int = 10
    ) -> list:
        """
        Get related concepts from KGs

        Args:
            concept: Concept to query
            limit: Maximum number of related concepts

        Returns:
            List of (related_concept, score) tuples
        """
        related = []

        # Query ConceptNet
        if self.use_conceptnet:
            conceptnet_related = self._conceptnet_related(concept, limit)
            related.extend(conceptnet_related)

        # Sort by score
        related.sort(key=lambda x: x[1], reverse=True)

        return related[:limit]

    def _conceptnet_related(
        self,
        concept: str,
        limit: int
    ) -> list:
        """
        Get related concepts from ConceptNet

        Args:
            concept: Concept to query
            limit: Maximum results

        Returns:
            List of (related_concept, score) tuples
        """
        try:
            import requests

            # Normalize concept
            c_normalized = concept.lower().replace(' ', '_')

            # Query
            url = f"{self.conceptnet_api}/c/en/{c_normalized}"
            params = {'limit': limit * 2}  # Get more, filter later

            response = requests.get(url, params=params, timeout=self.timeout)

            if response.status_code != 200:
                return []

            data = response.json()

            # Extract related concepts
            related = []
            for edge in data.get('edges', []):
                end_label = edge.get('end', {}).get('label', '')
                weight = edge.get('weight', 0.0)
                relation = edge.get('rel', {}).get('label', '')

                if end_label:
                    boost = self._relation_boost(relation)
                    related.append((end_label, boost * weight))

            return related

        except Exception as e:
            logger.warning(f"ConceptNet related query failed: {e}")
            return []

    def batch_validate(
        self,
        concept_pairs: list
    ) -> Dict[Tuple[str, str], float]:
        """
        Validate multiple concept pairs

        Args:
            concept_pairs: List of (concept1, concept2) tuples

        Returns:
            Dictionary mapping (concept1, concept2) -> score
        """
        results = {}

        for c1, c2 in concept_pairs:
            score = self.validate_relation(c1, c2)
            if score is not None:
                results[(c1, c2)] = score

        return results

    def is_synonym(
        self,
        concept1: str,
        concept2: str
    ) -> bool:
        """
        Check if two concepts are synonyms

        Args:
            concept1: First concept
            concept2: Second concept

        Returns:
            True if concepts are synonyms
        """
        score = self.validate_relation(concept1, concept2)

        # High threshold for synonym
        return score is not None and score > 0.8

    def get_concept_definition(
        self,
        concept: str
    ) -> Optional[str]:
        """
        Get definition of concept from WordNet

        Args:
            concept: Concept to define

        Returns:
            Definition string or None
        """
        if not self.use_wordnet or not self.wordnet:
            return None

        try:
            synsets = self.wordnet.synsets(concept)

            if not synsets:
                return None

            # Return first definition
            return synsets[0].definition()

        except Exception as e:
            logger.warning(f"WordNet definition failed: {e}")
            return None


class FallbackKGValidator:
    """
    Fallback KG validator using simple heuristics.

    Used when external KGs are not available.
    """

    def validate_relation(
        self,
        concept1: str,
        concept2: str
    ) -> Optional[float]:
        """
        Validate relationship using simple heuristics

        Args:
            concept1: First concept
            concept2: Second concept

        Returns:
            Confidence score or None
        """
        # String similarity
        from difflib import SequenceMatcher

        similarity = SequenceMatcher(None, concept1.lower(), concept2.lower()).ratio()

        # High similarity -> likely related
        if similarity > 0.8:
            return similarity

        # Check word overlap
        words1 = set(concept1.lower().split())
        words2 = set(concept2.lower().split())

        if words1 & words2:
            overlap = len(words1 & words2) / len(words1 | words2)
            return overlap

        # Return low similarity score instead of None
        return 0.0  # Return 0.0 instead of None to indicate no relation

    def is_synonym(
        self,
        word1: str,
        word2: str
    ) -> bool:
        """
        Check if two words are synonyms using simple heuristics

        Args:
            word1: First word
            word2: Second word

        Returns:
            True if likely synonyms, False otherwise
        """
        # Simple string-based fallback
        if word1.lower() == word2.lower():
            return True

        # Check string similarity
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, word1.lower(), word2.lower()).ratio()

        # Very high similarity might indicate synonyms
        if similarity > 0.9:
            return True

        # Check common synonym patterns
        synonym_pairs = [
            ('car', 'automobile'),
            ('fast', 'quick'),
            ('big', 'large'),
            ('small', 'little'),
            ('happy', 'glad'),
            ('sad', 'unhappy'),
        ]

        lower_w1 = word1.lower().strip()
        lower_w2 = word2.lower().strip()

        for pair in synonym_pairs:
            if (lower_w1 == pair[0] and lower_w2 == pair[1]) or \
               (lower_w1 == pair[1] and lower_w2 == pair[0]):
                return True

        return False

    def get_related_concepts(
        self,
        concept: str,
        limit: int = 10
    ) -> list:
        """No related concepts available in fallback"""
        return []


if __name__ == "__main__":
    # Demo
    print("Knowledge Graph Validator")
    print("=" * 50)

    validator = KGValidator(
        use_conceptnet=True,
        use_wordnet=True
    )

    # Test cases
    test_cases = [
        ("velocity", "speed"),
        ("fast", "rapid"),
        ("pressure", "force"),
        ("flow", "current"),
    ]

    print("\nValidating concept relationships:")
    for c1, c2 in test_cases:
        score = validator.validate_relation(c1, c2)
        print(f"  {c1:15} ↔ {c2:15}: {score:.3f}" if score else f"  {c1:15} ↔ {c2:15}: None")

    # Get related concepts
    print("\nRelated concepts for 'velocity':")
    related = validator.get_related_concepts("velocity", limit=5)
    for concept, score in related:
        print(f"  {concept:20}: {score:.3f}")

    print("\n✅ KG Validator working!")
