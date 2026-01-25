"""
Lexical Matcher

String-based similarity for ontology mapping.

Agent: G2 (Ψ₂ Specialist)
Created: 2025-12-31
"""

import numpy as np
from typing import List, Tuple
from difflib import SequenceMatcher


class LexicalMatcher:
    """
    Lexical similarity matcher using string similarity algorithms.

    Methods:
    - Jaro-Winkler similarity
    - Levenshtein distance
    - N-gram overlap
    """

    def __init__(self, threshold: float = 0.3, method: str = 'jaro-winkler'):
        """
        Initialize lexical matcher

        Args:
            threshold: Minimum similarity score
            method: Similarity method ('jaro-winkler', 'levenshtein', 'ngram')
        """
        self.threshold = threshold
        self.method = method

    def similarity(self, s1: str, s2: str) -> float:
        """
        Compute similarity between two strings

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score [0, 1]
        """
        if self.method == 'jaro-winkler':
            return self._jaro_winkler_similarity(s1, s2)
        elif self.method == 'levenshtein':
            return self._levenshtein_similarity(s1, s2)
        elif self.method == 'ngram':
            return self._ngram_similarity(s1, s2, n=3)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """
        Compute Jaro-Winkler similarity

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score [0, 1]
        """
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        if len(s1) == 0 or len(s2) == 0:
            return 0.0

        # Match distance
        match_distance = max(len(s1), len(s2)) // 2 - 1
        if match_distance < 0:
            match_distance = 0

        # Find matches
        s1_matches = [False] * len(s1)
        s2_matches = [False] * len(s2)

        matches = 0
        transpositions = 0

        for i in range(len(s1)):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len(s2))

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        # Count transpositions
        k = 0
        for i in range(len(s1)):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        # Jaro similarity
        jaro = (
            (matches / len(s1) +
             matches / len(s2) +
             (matches - transpositions / 2) / matches) / 3
        )

        # Winkler modification
        prefix = 0
        for i in range(min(len(s1), len(s2))):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        prefix = min(prefix, 4)
        winkler = jaro + prefix * 0.1 * (1 - jaro)

        return winkler

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """
        Compute normalized Levenshtein similarity

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score [0, 1]
        """
        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))

        if max_len == 0:
            return 1.0

        return 1.0 - (distance / max_len)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Compute Levenshtein edit distance

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance (number of operations)
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]

            for j, c2 in enumerate(s2):
                # Cost of substitution
                cost = 0 if c1 == c2 else 1

                # Minimum of deletion, insertion, substitution
                current_row.append(min(
                    previous_row[j + 1] + 1,  # deletion
                    current_row[j] + 1,       # insertion
                    previous_row[j] + cost    # substitution
                ))

            previous_row = current_row

        return previous_row[-1]

    def _ngram_similarity(self, s1: str, s2: str, n: int = 3) -> float:
        """
        Compute n-gram overlap similarity

        Args:
            s1: First string
            s2: Second string
            n: N-gram size

        Returns:
            Similarity score [0, 1]
        """
        # Generate n-grams
        ngrams1 = self._generate_ngrams(s1, n)
        ngrams2 = self._generate_ngrams(s2, n)

        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0

        # Compute overlap
        intersection = len(set(ngrams1) & set(ngrams2))
        union = len(set(ngrams1) | set(ngrams2))

        return intersection / union if union > 0 else 0.0

    def _generate_ngrams(self, s: str, n: int) -> List[str]:
        """
        Generate character n-grams

        Args:
            s: Input string
            n: N-gram size

        Returns:
            List of n-grams
        """
        return [s[i:i+n] for i in range(len(s) - n + 1)]

    def match_best(
        self,
        source: str,
        targets: List[str]
    ) -> Tuple[str, float]:
        """
        Find best matching target for source

        Args:
            source: Source string
            targets: List of target strings

        Returns:
            Tuple of (best_match, score)
        """
        best_match = None
        best_score = 0.0

        for target in targets:
            score = self.similarity(source, target)
            if score > best_score:
                best_score = score
                best_match = target

        return best_match, best_score

    def match_all(
        self,
        source: str,
        targets: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Match source against all targets

        Args:
            source: Source string
            targets: List of target strings

        Returns:
            List of (target, score) tuples, sorted by score
        """
        matches = [(target, self.similarity(source, target)) for target in targets]
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches


if __name__ == "__main__":
    # Demo
    print("Lexical Matcher")
    print("=" * 50)

    matcher = LexicalMatcher(threshold=0.3, method='jaro-winkler')

    # Test cases
    test_cases = [
        ("velocity", "velocity"),
        ("velocity", "velocity_x"),
        ("velocity", "speed"),
        ("flow_rate", "current"),
        ("pressure", "voltage"),
    ]

    print("\nSimilarity scores:")
    for s1, s2 in test_cases:
        score = matcher.similarity(s1, s2)
        print(f"  {s1:15} ↔ {s2:15}: {score:.3f}")

    print("\n✅ Lexical Matcher working!")
