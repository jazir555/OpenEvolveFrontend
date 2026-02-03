"""
Embedding Generator for OpenEvolve Knowledge Engine

This module provides functionality for generating embeddings for knowledge artifacts
to enable semantic search and similarity analysis.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding generation operation."""
    success: bool
    embedding: Optional[List[float]] = None
    text_used: Optional[str] = None
    model_used: Optional[str] = None
    processing_time_ms: float = 0.0
    error: Optional[str] = None


class EmbeddingGenerator:
    """
    Generator for embeddings of knowledge artifacts.
    
    Provides methods for:
    - Generating embeddings for text content
    - Batch embedding generation
    - Similarity computation
    - Embedding normalization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedding generator.
        
        Args:
            config: Configuration for embedding generation
        """
        self.config = config or self._get_default_config()
        
        # Initialize embedding model
        self.model = None
        self.tokenizer = None
        self.model_name = self.config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.normalize = self.config.get("normalize", True)
        self.batch_size = self.config.get("batch_size", 32)
        
        # Initialize the model
        self._initialize_model()
        
        logger.info({
            "msg": "EmbeddingGenerator initialized",
            "model_name": self.model_name,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "normalize": True,
            "batch_size": 32,
            "max_length": 512,
            "pooling_strategy": "mean"
        }
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model {self.model_name} loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not available, using mock embeddings")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.model = None
    
    def generate_embedding(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            EmbeddingResult with the generated embedding
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Generating embedding for text",
            "text_length": len(text),
            "timestamp": start_time.isoformat()
        })
        
        try:
            if self.model:
                # Generate embedding using the loaded model
                embedding = self.model.encode([text])[0].tolist()
                
                if self.normalize:
                    embedding = self._normalize_embedding(embedding)
                
                processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                result = EmbeddingResult(
                    success=True,
                    embedding=embedding,
                    text_used=text,
                    model_used=self.model_name,
                    processing_time_ms=processing_time_ms
                )
                
                logger.info({
                    "msg": "Embedding generated successfully",
                    "embedding_dim": len(embedding),
                    "processing_time_ms": processing_time_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return result
            else:
                # Fallback: generate mock embedding
                embedding = self._generate_mock_embedding(text)
                
                processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                result = EmbeddingResult(
                    success=True,
                    embedding=embedding,
                    text_used=text,
                    model_used="mock_model",
                    processing_time_ms=processing_time_ms
                )
                
                logger.warning({
                    "msg": "Generated mock embedding (model not available)",
                    "embedding_dim": len(embedding),
                    "processing_time_ms": processing_time_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return result
                
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Embedding generation failed",
                "text_length": len(text),
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return EmbeddingResult(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of EmbeddingResult objects
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Generating batch embeddings",
            "text_count": len(texts),
            "batch_size": self.batch_size,
            "timestamp": start_time.isoformat()
        })
        
        results = []
        
        try:
            if self.model:
                # Process in batches to manage memory
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    
                    # Generate embeddings for the batch
                    batch_embeddings = self.model.encode(batch)
                    
                    # Process each embedding in the batch
                    for j, text in enumerate(batch):
                        embedding = batch_embeddings[j].tolist()
                        
                        if self.normalize:
                            embedding = self._normalize_embedding(embedding)
                        
                        results.append(EmbeddingResult(
                            success=True,
                            embedding=embedding,
                            text_used=text,
                            model_used=self.model_name,
                            processing_time_ms=0.0  # Will calculate total at the end
                        ))
            else:
                # Fallback: generate mock embeddings
                for text in texts:
                    embedding = self._generate_mock_embedding(text)
                    results.append(EmbeddingResult(
                        success=True,
                        embedding=embedding,
                        text_used=text,
                        model_used="mock_model",
                        processing_time_ms=0.0
                    ))
            
            total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Update processing time for all results
            for result in results:
                result.processing_time_ms = total_processing_time / len(results) if results else 0.0
            
            logger.info({
                "msg": "Batch embeddings generated successfully",
                "text_count": len(texts),
                "successful_count": len([r for r in results if r.success]),
                "total_processing_time_ms": total_processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return results
            
        except Exception as e:
            total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Batch embedding generation failed",
                "text_count": len(texts),
                "error": str(e),
                "total_processing_time_ms": total_processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Return error results for all texts
            error_results = []
            for text in texts:
                error_results.append(EmbeddingResult(
                    success=False,
                    text_used=text,
                    error=str(e),
                    processing_time_ms=total_processing_time / len(texts) if texts else 0.0
                ))
            
            return error_results
    
    def generate_knowledge_artifact_embedding(self, artifact: Dict[str, Any]) -> Optional[List[float]]:
        """
        Generate embedding for a knowledge artifact.
        
        Args:
            artifact: Knowledge artifact dictionary
            
        Returns:
            Embedding vector or None if generation failed
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Generating embedding for knowledge artifact",
            "artifact_type": artifact.get("type", "unknown"),
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Construct text from artifact components
            text_parts = []
            
            # Add content
            content = artifact.get("content", "")
            if content:
                text_parts.append(content)
            
            # Add type
            artifact_type = artifact.get("type", "")
            if artifact_type:
                text_parts.append(f"Type: {artifact_type}")
            
            # Add source
            source = artifact.get("source", "")
            if source:
                text_parts.append(f"Source: {source}")
            
            # Add context
            context = artifact.get("context", "")
            if context:
                text_parts.append(f"Context: {context}")
            
            # Add metadata
            metadata = artifact.get("metadata", {})
            if metadata:
                text_parts.append(f"Metadata: {str(metadata)}")
            
            # Combine all parts
            combined_text = " ".join(text_parts)
            
            # Generate embedding
            result = self.generate_embedding(combined_text)
            
            if result.success:
                processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                logger.info({
                    "msg": "Knowledge artifact embedding generated successfully",
                    "artifact_type": artifact.get("type", "unknown"),
                    "embedding_dim": len(result.embedding) if result.embedding else 0,
                    "processing_time_ms": processing_time_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return result.embedding
            else:
                logger.error({
                    "msg": "Knowledge artifact embedding generation failed",
                    "artifact_type": artifact.get("type", "unknown"),
                    "error": result.error,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return None
                
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Knowledge artifact embedding generation failed with exception",
                "artifact_type": artifact.get("type", "unknown"),
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return None
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (cosine similarity)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0
    
    def find_similar_embeddings(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find the most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top similar embeddings to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for i, candidate_emb in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate_emb)
            similarities.append((i, similarity))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return similarities[:top_k]
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize an embedding vector to unit length."""
        vec = np.array(embedding)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return embedding
        return (vec / norm).tolist()
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate a mock embedding for testing purposes."""
        # Create a deterministic mock embedding based on text content
        import hashlib
        
        # Use text hash to create a pseudo-random but consistent embedding
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hex to numbers and normalize
        embedding = []
        for i in range(0, 32, 2):  # Generate 16 pairs of hex chars
            if i + 1 < len(text_hash):
                hex_pair = text_hash[i:i+2]
                val = int(hex_pair, 16) / 255.0  # Normalize to 0-1
                embedding.append(val)
        
        # Pad if needed
        while len(embedding) < 32:
            embedding.append(0.0)
        
        # Trim if too long
        embedding = embedding[:32]
        
        return embedding
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "normalize_embeddings": self.normalize,
            "batch_size": self.batch_size,
            "model_loaded": self.model is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }