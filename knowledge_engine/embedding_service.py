"""
Real Embedding Service for Knowledge Engine

Provides actual embedding generation using sentence-transformers or fallback implementations.
Supports multiple embedding models and caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "all-MiniLM-L6-v2"
    dimensions: int = 384
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_dir: Optional[str] = None
    device: str = "cpu"  # cpu, cuda, mps
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "cache_dir": self.cache_dir,
            "device": self.device
        }


class EmbeddingService:
    """
    Real embedding service with sentence-transformers integration.
    Falls back to TF-IDF based embeddings if the library is not available.
    """
    
    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "all-distilroberta-v1": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "paraphrase-mpnet-base-v2": 768,
    }
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._init_model()
        
    def _init_model(self):
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                cache_folder=self.config.cache_dir
            )
            
            # Update dimensions based on actual model
            self.config.dimensions = self._model.get_sentence_embedding_dimension()
            
            logger.info({
                "msg": "Embedding model initialized",
                "model": self.config.model_name,
                "dimensions": self.config.dimensions,
                "device": self.config.device
            })
            
        except ImportError:
            logger.warning({
                "msg": "sentence-transformers not available, using TF-IDF fallback",
                "install": "pip install sentence-transformers"
            })
            self._init_tfidf_fallback()
        except Exception as e:
            logger.error({
                "msg": "Failed to load embedding model, using fallback",
                "error": str(e)
            })
            self._init_tfidf_fallback()
    
    def _init_tfidf_fallback(self):
        """Initialize TF-IDF based fallback embedding."""
        self._model = None
        self._vectorizer = None
        self._tfidf_vocab: Dict[str, int] = {}
        self._tfidf_idf: Dict[int, float] = {}
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=self.config.dimensions,
                stop_words='english',
                lowercase=True
            )
            logger.info({
                "msg": "TF-IDF fallback initialized",
                "max_features": self.config.dimensions
            })
        except ImportError:
            logger.warning({
                "msg": "scikit-learn not available, using hash-based fallback"
            })
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embedding vector
        """
        if not text:
            return np.zeros(self.config.dimensions, dtype=np.float32)
        
        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        # Generate embedding
        if self._model is not None:
            # Use sentence-transformers
            embedding = self._model.encode(
                text,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False
            )
        elif self._vectorizer is not None:
            # Use TF-IDF
            embedding = self._embed_tfidf(text)
        else:
            # Use hash-based fallback
            embedding = self._embed_hash(text)
        
        # Ensure correct dimensions
        if len(embedding) != self.config.dimensions:
            embedding = self._pad_or_truncate(embedding)
        
        # Cache the result
        self._cache[cache_key] = embedding
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts
            
        Returns:
            numpy array of shape (len(texts), dimensions)
        """
        if not texts:
            return np.zeros((0, self.config.dimensions), dtype=np.float32)
        
        # Filter out empty texts
        valid_texts = [t if t else "" for t in texts]
        
        if self._model is not None:
            # Batch encode with sentence-transformers
            embeddings = self._model.encode(
                valid_texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False
            )
            return embeddings.astype(np.float32)
        else:
            # Process individually
            embeddings = [self.embed_text(text) for text in valid_texts]
            return np.array(embeddings, dtype=np.float32)
    
    def _embed_tfidf(self, text: str) -> np.ndarray:
        """Generate TF-IDF embedding."""
        if self._vectorizer is None:
            return self._embed_hash(text)
        
        try:
            # Transform single document
            tfidf_matrix = self._vectorizer.fit_transform([text])
            embedding = tfidf_matrix.toarray()[0]
            
            # Pad or truncate to match dimensions
            if len(embedding) < self.config.dimensions:
                embedding = np.pad(
                    embedding,
                    (0, self.config.dimensions - len(embedding)),
                    mode='constant'
                )
            elif len(embedding) > self.config.dimensions:
                embedding = embedding[:self.config.dimensions]
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error({"msg": "TF-IDF embedding failed", "error": str(e)})
            return self._embed_hash(text)
    
    def _embed_hash(self, text: str) -> np.ndarray:
        """
        Generate hash-based embedding as final fallback.
        Creates deterministic embeddings based on character n-grams.
        """
        vector = np.zeros(self.config.dimensions, dtype=np.float32)
        
        if not text:
            return vector
        
        # Character n-gram features
        text = text.lower()
        n_grams = [2, 3, 4]  # bi-grams, tri-grams, 4-grams
        
        for n in n_grams:
            for i in range(len(text) - n + 1):
                ngram = text[i:i + n]
                # Hash to position in vector
                hash_val = hashlib.md5(ngram.encode()).hexdigest()
                idx = int(hash_val, 16) % self.config.dimensions
                # Weight by position (earlier in text = more important)
                weight = 1.0 / (1 + i * 0.1)
                vector[idx] += weight
        
        # Add word-level features
        words = text.split()
        for i, word in enumerate(words[:100]):  # Limit to first 100 words
            hash_val = hashlib.md5(word.encode()).hexdigest()
            idx = int(hash_val, 16) % self.config.dimensions
            weight = 1.0 / (1 + i * 0.05)
            vector[idx] += weight * 2  # Words weighted more than n-grams
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _pad_or_truncate(self, embedding: np.ndarray) -> np.ndarray:
        """Pad or truncate embedding to match configured dimensions."""
        if len(embedding) < self.config.dimensions:
            return np.pad(
                embedding,
                (0, self.config.dimensions - len(embedding)),
                mode='constant'
            )
        elif len(embedding) > self.config.dimensions:
            return embedding[:self.config.dimensions]
        return embedding
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top matches to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = [
            (i, self.compute_similarity(query_embedding, emb))
            for i, emb in enumerate(candidate_embeddings)
        ]
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (
            self._cache_hits / total_requests * 100
            if total_requests > 0 else 0
        )
        
        return {
            "model": self.config.model_name,
            "dimensions": self.config.dimensions,
            "device": self.config.device,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "using_sentence_transformers": self._model is not None,
            "using_tfidf": self._vectorizer is not None
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info({"msg": "Embedding cache cleared"})


# Convenience functions
def create_embedding_service(
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    cache_dir: Optional[str] = None
) -> EmbeddingService:
    """
    Create an embedding service with specified configuration.
    
    Args:
        model_name: Name of the sentence-transformers model
        device: Device to run on (cpu, cuda, mps)
        cache_dir: Directory to cache models
        
    Returns:
        Configured EmbeddingService instance
    """
    config = EmbeddingConfig(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        dimensions=EmbeddingService.MODEL_DIMENSIONS.get(model_name, 384)
    )
    return EmbeddingService(config)


def get_default_embedding_service() -> EmbeddingService:
    """Get or create the default embedding service singleton."""
    if not hasattr(get_default_embedding_service, '_instance'):
        get_default_embedding_service._instance = EmbeddingService()
    return get_default_embedding_service._instance


# Example usage
if __name__ == "__main__":
    # Test the embedding service
    service = create_embedding_service()
    
    # Single text embedding
    text = "This is a test sentence about machine learning."
    embedding = service.embed_text(text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding)}")
    
    # Batch embedding
    texts = [
        "Machine learning is fascinating.",
        "Deep learning uses neural networks.",
        "Python is a programming language."
    ]
    embeddings = service.embed_batch(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # Compute similarity
    sim = service.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between text 0 and 1: {sim:.4f}")
    
    # Stats
    print("\nService stats:")
    print(json.dumps(service.get_stats(), indent=2))
