"""
Semantic Matcher

Word/sentence embedding-based similarity for ontology mapping.

Agent: G2 (Ψ₂ Specialist)
Created: 2025-12-31
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple


class SemanticMatcher:
    """
    Semantic similarity matcher using sentence embeddings.

    Uses sentence-transformers for computing semantic similarity.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        threshold: float = 0.5,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize semantic matcher

        Args:
            model_name: Sentence transformer model name
            threshold: Minimum similarity score
            cache_dir: Cache directory for models
        """
        self.model_name = model_name
        self.threshold = threshold
        self.cache_dir = cache_dir or 'rese/phase2/ontology_cache/models'

        # Lazy loading: load model only when needed
        self.model = None
        self.embedding_cache: Dict[str, np.ndarray] = {}

    def _load_model(self):
        """
        Load sentence transformer model

        Lazily loads model on first use to avoid startup overhead
        """
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            # Create cache directory
            os.makedirs(self.cache_dir, exist_ok=True)

            # Load model
            print(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            print(f"Model loaded successfully")

        except ImportError:
            print("WARNING: sentence-transformers not installed")
            print("Install with: pip install sentence-transformers")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: List of text strings

        Returns:
            Embeddings array of shape (len(texts), embedding_dim)
        """
        self._load_model()

        # Check cache
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Encode uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)

            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                self.embedding_cache[text] = embedding

            # Combine cached and new embeddings
            all_embeddings = [None] * len(texts)
            for i, emb in cached_embeddings:
                all_embeddings[i] = emb
            for i, emb in zip(uncached_indices, new_embeddings):
                all_embeddings[i] = emb

            return np.array(all_embeddings)
        else:
            # All cached
            all_embeddings = [None] * len(texts)
            for i, emb in cached_embeddings:
                all_embeddings[i] = emb
            return np.array(all_embeddings)

    def similarity(self, s1: str, s2: str) -> float:
        """
        Compute semantic similarity between two strings

        Args:
            s1: First string
            s2: Second string

        Returns:
            Cosine similarity [0, 1]
        """
        embeddings = self.encode([s1, s2])
        return self._cosine_similarity(embeddings[0], embeddings[1])

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Cosine similarity [-1, 1]
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(v1, v2) / (norm1 * norm2)

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
        # Batch encode for efficiency
        all_texts = [source] + targets
        embeddings = self.encode(all_texts)

        source_emb = embeddings[0]
        target_embs = embeddings[1:]

        # Compute similarities
        matches = [
            (target, self._cosine_similarity(source_emb, target_emb))
            for target, target_emb in zip(targets, target_embs)
        ]

        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def similarity_matrix(
        self,
        sources: List[str],
        targets: List[str]
    ) -> np.ndarray:
        """
        Compute similarity matrix between sources and targets

        Args:
            sources: List of source strings
            targets: List of target strings

        Returns:
            Similarity matrix of shape (len(sources), len(targets))
        """
        source_embs = self.encode(sources)
        target_embs = self.encode(targets)

        # Normalize embeddings
        source_norms = np.linalg.norm(source_embs, axis=1, keepdims=True)
        target_norms = np.linalg.norm(target_embs, axis=1, keepdims=True)

        source_embs = source_embs / (source_norms + 1e-8)
        target_embs = target_embs / (target_norms + 1e-8)

        # Compute cosine similarity matrix
        similarity_matrix = np.dot(source_embs, target_embs.T)

        return similarity_matrix

    def save_cache(self, filepath: str):
        """
        Save embedding cache to file

        Args:
            filepath: Output file path
        """
        import pickle

        with open(filepath, 'wb') as f:
            pickle.dump(self.embedding_cache, f)

        print(f"Saved cache to {filepath}")

    def load_cache(self, filepath: str):
        """
        Load embedding cache from file

        Args:
            filepath: Input file path
        """
        import pickle

        with open(filepath, 'rb') as f:
            self.embedding_cache = pickle.load(f)

        print(f"Loaded cache from {filepath}")


class FallbackSemanticMatcher:
    """
    Fallback semantic matcher using basic word overlap.

    Used when sentence-transformers is not available.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize fallback matcher

        Args:
            threshold: Minimum similarity score
        """
        self.threshold = threshold

    def similarity(self, s1: str, s2: str) -> float:
        """
        Compute word overlap similarity

        Args:
            s1: First string
            s2: Second string

        Returns:
            Jaccard similarity [0, 1]
        """
        # Tokenize
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Fallback encode (returns dummy embeddings)

        Args:
            texts: List of texts

        Returns:
            Dummy embeddings
        """
        # Return TF-IDF like sparse vectors
        from collections import Counter
        import numpy as np

        # Build vocabulary
        all_words = set()
        for text in texts:
            all_words.update(text.lower().split())

        word_to_idx = {word: i for i, word in enumerate(sorted(all_words))}

        # Create TF vectors
        embeddings = np.zeros((len(texts), len(word_to_idx)))
        for i, text in enumerate(texts):
            words = text.lower().split()
            word_counts = Counter(words)
            for word, count in word_counts.items():
                if word in word_to_idx:
                    embeddings[i, word_to_idx[word]] = count

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        return embeddings


if __name__ == "__main__":
    # Demo
    print("Semantic Matcher")
    print("=" * 50)

    try:
        matcher = SemanticMatcher(
            model_name='all-MiniLM-L6-v2',
            threshold=0.5
        )

        # Test cases
        test_cases = [
            ("velocity", "speed"),
            ("fast", "rapid"),
            ("flow rate", "current"),
            ("pressure", "voltage"),
            ("pipe resistance", "electrical resistance"),
        ]

        print("\nSemantic similarity scores:")
        for s1, s2 in test_cases:
            score = matcher.similarity(s1, s2)
            print(f"  {s1:20} ↔ {s2:25}: {score:.3f}")

        print("\n✅ Semantic Matcher working!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nUsing fallback matcher...")

        matcher = FallbackSemanticMatcher()

        for s1, s2 in test_cases:
            score = matcher.similarity(s1, s2)
            print(f"  {s1:20} ↔ {s2:25}: {score:.3f}")
