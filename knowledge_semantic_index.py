"""
Knowledge Semantic Index - Semantic Indexing Layer using Vector Embeddings

This module provides a semantic indexing layer that uses vector embeddings for meaning-based
search. Results are filtered through hierarchical and graph indexes first to ensure only
relevant semantically similar data is retrieved.

Key Components:
- SemanticIndex: Main interface for semantic indexing operations
- EmbeddingStore: Manages vector embeddings with local caching and optional vector DB integration
- SemanticQuery: Context-aware query with hierarchical and graph pre-filtering
- generate_embedding: Embedding generation with OpenAI and sentence-transformers fallbacks
- semantic_search: Multi-stage filtering pipeline for semantic search

Integration:
- Uses patterns from langchain_chroma_integration.py for vector storage
- Uses env_helpers for API key management
- Uses thread_safety_utils patterns for thread-safe operations
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from contextlib import contextmanager

import numpy as np
from numpy.linalg import norm

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# OPTIONAL DEPENDENCIES WITH FALLBACKS
# =============================================================================

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.debug("OpenAI package not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.debug("sentence-transformers not available")

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.debug("ChromaDB not available")

try:
    import qdrant_client
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.debug("Qdrant not available")

# Import existing patterns from codebase
from env_helpers import env_var_api_key, env_var_str, env_var_path, env_var_int
from thread_safety_utils import get_session_lock


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SemanticIndexConfig:
    """Configuration for semantic indexing system."""
    
    # Storage configuration
    cache_dir: str = "./knowledge_semantic_cache"
    embedding_model: str = "text-embedding-3-small"  # OpenAI preferred
    fallback_model: str = "all-MiniLM-L6-v2"  # sentence-transformers fallback
    
    # Vector dimensions for each model
    embedding_dimensions: Dict[str, int] = field(default_factory=lambda: {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
    })
    
    # Search configuration
    default_top_k: int = 10
    similarity_threshold: float = 0.7
    max_batch_size: int = 100
    
    # Caching configuration
    cache_enabled: bool = True
    cache_ttl_hours: int = 168  # 1 week
    
    # Backend configuration (sqlite, chroma, qdrant)
    vector_backend: str = "sqlite"  # sqlite, chroma, qdrant
    qdrant_url: Optional[str] = None
    qdrant_collection: str = "semantic_index"
    
    # Threading
    thread_safe: bool = True
    
    def __post_init__(self):
        """Initialize derived paths and validate configuration."""
        self.cache_path = Path(self.cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Validate backend
        valid_backends = ["sqlite", "chroma", "qdrant"]
        if self.vector_backend not in valid_backends:
            logger.warning(f"Invalid backend '{self.vector_backend}', using 'sqlite'")
            self.vector_backend = "sqlite"
        
        # Check backend availability
        if self.vector_backend == "chroma" and not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, falling back to sqlite")
            self.vector_backend = "sqlite"
        
        if self.vector_backend == "qdrant" and not QDRANT_AVAILABLE:
            logger.warning("Qdrant not available, falling back to sqlite")
            self.vector_backend = "sqlite"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class SemanticQuery:
    """
    Query with context-aware filtering for semantic search.
    
    Pre-filters:
    - hierarchy_level: Filter by hierarchical level (e.g., "root", "child", "leaf")
    - graph_nodes: Filter by graph node IDs
    - graph_connectivity: Filter by graph connectivity (connected nodes only)
    
    Ranking:
    - semantic similarity (primary)
    - recency (secondary)
    - importance (tertiary)
    """
    
    query_text: str
    
    # Hierarchical filters
    hierarchy_level: Optional[str] = None  # e.g., "root", "level_1", "level_2"
    parent_id: Optional[str] = None  # Filter by parent in hierarchy
    
    # Graph filters
    graph_node_ids: Optional[List[str]] = None  # Specific graph nodes to search
    graph_connectivity: Optional[Dict[str, Any]] = None  # Connectivity constraints
    
    # Ranking parameters
    top_k: int = 10
    similarity_threshold: float = 0.7
    recency_weight: float = 0.2  # Weight for recency in re-ranking
    importance_weight: float = 0.1  # Weight for importance in re-ranking
    
    # Time filters
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    
    # Metadata filters
    metadata_filters: Optional[Dict[str, Any]] = None
    
    def validate(self) -> List[str]:
        """Validate query parameters."""
        errors = []
        
        if not self.query_text or not self.query_text.strip():
            errors.append("query_text cannot be empty")
        
        if self.top_k < 1:
            errors.append(f"top_k must be at least 1, got {self.top_k}")
        
        if not 0.0 <= self.similarity_threshold <= 1.0:
            errors.append(f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}")
        
        if not 0.0 <= self.recency_weight <= 1.0:
            errors.append(f"recency_weight must be between 0.0 and 1.0, got {self.recency_weight}")
        
        if not 0.0 <= self.importance_weight <= 1.0:
            errors.append(f"importance_weight must be between 0.0 and 1.0, got {self.importance_weight}")
        
        # Check weight sum
        total_weight = self.recency_weight + self.importance_weight
        if total_weight > 1.0:
            errors.append(f"Sum of recency_weight and importance_weight must be <= 1.0, got {total_weight}")
        
        return errors


@dataclass
class SemanticResult:
    """Result from semantic search with metadata."""
    
    id: str
    content: str
    similarity_score: float  # 0-1
    
    # Hierarchical context
    hierarchy_level: Optional[str] = None
    parent_id: Optional[str] = None
    
    # Graph context
    graph_node_id: Optional[str] = None
    connected_nodes: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: Optional[datetime] = None
    importance: float = 0.5  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "similarity_score": self.similarity_score,
            "hierarchy_level": self.hierarchy_level,
            "parent_id": self.parent_id,
            "graph_node_id": self.graph_node_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "importance": self.importance,
        }


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

class EmbeddingGenerator:
    """
    Generates embeddings using multiple backends with caching.
    
    Priority:
    1. OpenAI text-embedding-3-small (preferred for quality)
    2. Sentence-transformers (fallback for local/offline)
    """
    
    def __init__(self, config: Optional[SemanticIndexConfig] = None):
        self.config = config or SemanticIndexConfig()
        self._openai_client: Optional[Any] = None
        self._local_model: Optional[Any] = None
        self._model_lock = threading.RLock()
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.RLock()
        
    def _get_openai_client(self) -> Optional[Any]:
        """Get or create OpenAI client."""
        if not OPENAI_AVAILABLE:
            return None
        
        with self._model_lock:
            if self._openai_client is None:
                api_key = env_var_api_key("OPENAI_API_KEY", provider="OpenAI")
                if api_key:
                    try:
                        self._openai_client = openai.OpenAI(api_key=api_key)
                        logger.info("OpenAI client initialized for embeddings")
                    except Exception as e:
                        logger.warning(f"Failed to initialize OpenAI client: {e}")
            return self._openai_client
    
    def _get_local_model(self) -> Optional[Any]:
        """Get or create local sentence-transformers model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        
        with self._model_lock:
            if self._local_model is None:
                try:
                    self._local_model = SentenceTransformer(self.config.fallback_model)
                    logger.info(f"Loaded local embedding model: {self.config.fallback_model}")
                except Exception as e:
                    logger.warning(f"Failed to load local model: {e}")
            return self._local_model
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text embedding."""
        content = f"{text}:{model}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def _check_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Check if embedding is in memory cache."""
        if not self.config.cache_enabled:
            return None
        with self._cache_lock:
            return self._cache.get(cache_key)
    
    def _add_to_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """Add embedding to memory cache."""
        if not self.config.cache_enabled:
            return
        with self._cache_lock:
            self._cache[cache_key] = embedding
            # Simple LRU: limit cache size
            if len(self._cache) > 10000:
                # Remove oldest entries (simple approach)
                keys_to_remove = list(self._cache.keys())[:1000]
                for key in keys_to_remove:
                    del self._cache[key]
    
    def generate(
        self,
        text: Union[str, List[str]],
        use_cache: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text(s).
        
        Args:
            text: Single text or list of texts
            use_cache: Whether to use caching
            
        Returns:
            Embedding vector(s) as numpy array(s)
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Try OpenAI first
        embeddings = self._generate_with_openai(texts, use_cache)
        
        # Fall back to local model if OpenAI fails
        if embeddings is None:
            embeddings = self._generate_with_local(texts, use_cache)
        
        if embeddings is None:
            raise RuntimeError("No embedding backend available. Install openai or sentence-transformers.")
        
        return embeddings[0] if is_single else embeddings
    
    def _generate_with_openai(
        self,
        texts: List[str],
        use_cache: bool
    ) -> Optional[List[np.ndarray]]:
        """Generate embeddings using OpenAI API."""
        client = self._get_openai_client()
        if client is None:
            return None
        
        try:
            # Check cache first
            if use_cache:
                cached = []
                uncached_texts = []
                uncached_indices = []
                
                for i, text in enumerate(texts):
                    cache_key = self._get_cache_key(text, self.config.embedding_model)
                    cached_emb = self._check_cache(cache_key)
                    if cached_emb is not None:
                        cached.append((i, cached_emb))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                
                if not uncached_texts:
                    # All cached
                    results = [None] * len(texts)
                    for idx, emb in cached:
                        results[idx] = emb
                    return results
            else:
                uncached_texts = texts
                uncached_indices = list(range(len(texts)))
                cached = []
            
            # Call OpenAI API
            response = client.embeddings.create(
                model=self.config.embedding_model,
                input=uncached_texts
            )
            
            # Process results
            results = [None] * len(texts)
            
            # Add cached results
            for idx, emb in cached:
                results[idx] = emb
            
            # Add new results
            for i, embedding_data in enumerate(response.data):
                idx = uncached_indices[i]
                embedding = np.array(embedding_data.embedding, dtype=np.float32)
                results[idx] = embedding
                
                # Update cache
                if use_cache:
                    cache_key = self._get_cache_key(uncached_texts[i], self.config.embedding_model)
                    self._add_to_cache(cache_key, embedding)
            
            return results
            
        except Exception as e:
            logger.warning(f"OpenAI embedding generation failed: {e}")
            return None
    
    def _generate_with_local(
        self,
        texts: List[str],
        use_cache: bool
    ) -> Optional[List[np.ndarray]]:
        """Generate embeddings using local sentence-transformers model."""
        model = self._get_local_model()
        if model is None:
            return None
        
        try:
            # Check cache first
            if use_cache:
                cached = []
                uncached_texts = []
                uncached_indices = []
                
                for i, text in enumerate(texts):
                    cache_key = self._get_cache_key(text, self.config.fallback_model)
                    cached_emb = self._check_cache(cache_key)
                    if cached_emb is not None:
                        cached.append((i, cached_emb))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                
                if not uncached_texts:
                    results = [None] * len(texts)
                    for idx, emb in cached:
                        results[idx] = emb
                    return results
            else:
                uncached_texts = texts
                uncached_indices = list(range(len(texts)))
                cached = []
            
            # Generate embeddings
            embeddings = model.encode(uncached_texts, convert_to_numpy=True)
            
            # Process results
            results = [None] * len(texts)
            
            # Add cached results
            for idx, emb in cached:
                results[idx] = emb
            
            # Add new results
            for i, embedding in enumerate(embeddings):
                idx = uncached_indices[i]
                embedding = np.array(embedding, dtype=np.float32)
                results[idx] = embedding
                
                # Update cache
                if use_cache:
                    cache_key = self._get_cache_key(uncached_texts[i], self.config.fallback_model)
                    self._add_to_cache(cache_key, embedding)
            
            return results
            
        except Exception as e:
            logger.warning(f"Local embedding generation failed: {e}")
            return None
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        with self._cache_lock:
            self._cache.clear()


def generate_embedding(
    text: Union[str, List[str]],
    config: Optional[SemanticIndexConfig] = None,
    use_cache: bool = True
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Generate embeddings for text(s) using available backends.
    
    Args:
        text: Single text or list of texts to embed
        config: Optional configuration
        use_cache: Whether to use caching
        
    Returns:
        Embedding vector(s) as numpy array(s)
        
    Example:
        >>> embedding = generate_embedding("This is a test")
        >>> embeddings = generate_embedding(["text1", "text2"])
    """
    generator = EmbeddingGenerator(config)
    return generator.generate(text, use_cache)


# =============================================================================
# EMBEDDING STORE
# =============================================================================

class EmbeddingStore:
    """
    Manages vector embeddings with local caching and optional vector DB integration.
    
    Features:
    - Local caching (SQLite or pickle)
    - Optional: Qdrant/Chroma integration
    - Incremental updates
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[SemanticIndexConfig] = None):
        self.config = config or SemanticIndexConfig()
        self._lock = threading.RLock()
        self._db_connection: Optional[sqlite3.Connection] = None
        self._chroma_collection: Optional[Any] = None
        self._qdrant_client: Optional[Any] = None
        
        # Initialize backend
        self._initialize_backend()
    
    def _initialize_backend(self) -> None:
        """Initialize the storage backend."""
        if self.config.vector_backend == "sqlite":
            self._initialize_sqlite()
        elif self.config.vector_backend == "chroma":
            self._initialize_chroma()
        elif self.config.vector_backend == "qdrant":
            self._initialize_qdrant()
    
    def _initialize_sqlite(self) -> None:
        """Initialize SQLite backend."""
        db_path = self.config.cache_path / "embeddings.db"
        self._db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
        
        # Create tables
        cursor = self._db_connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                hierarchy_level TEXT,
                parent_id TEXT,
                graph_node_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                importance REAL DEFAULT 0.5,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                node_id TEXT,
                connected_node_id TEXT,
                PRIMARY KEY (node_id, connected_node_id)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hierarchy ON embeddings(hierarchy_level)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_parent ON embeddings(parent_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_graph_node ON embeddings(graph_node_id)
        """)
        
        self._db_connection.commit()
        logger.info(f"SQLite embedding store initialized: {db_path}")
    
    def _initialize_chroma(self) -> None:
        """Initialize ChromaDB backend."""
        if not CHROMADB_AVAILABLE:
            raise RuntimeError("ChromaDB not available")
        
        client = chromadb.PersistentClient(path=str(self.config.cache_path / "chroma"))
        
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.fallback_model
        )
        
        try:
            self._chroma_collection = client.get_collection(
                name=self.config.qdrant_collection,
                embedding_function=embedding_func
            )
        except Exception:
            self._chroma_collection = client.create_collection(
                name=self.config.qdrant_collection,
                embedding_function=embedding_func
            )
        
        logger.info("ChromaDB embedding store initialized")
    
    def _initialize_qdrant(self) -> None:
        """Initialize Qdrant backend."""
        if not QDRANT_AVAILABLE:
            raise RuntimeError("Qdrant not available")
        
        url = self.config.qdrant_url or "http://localhost:6333"
        self._qdrant_client = qdrant_client.QdrantClient(url=url)
        
        # Get embedding dimension
        dim = self.config.embedding_dimensions.get(
            self.config.embedding_model,
            self.config.embedding_dimensions[self.config.fallback_model]
        )
        
        # Create collection if it doesn't exist
        try:
            self._qdrant_client.create_collection(
                collection_name=self.config.qdrant_collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
        except Exception:
            pass  # Collection already exists
        
        logger.info(f"Qdrant embedding store initialized: {url}")
    
    def add(
        self,
        id: str,
        content: str,
        embedding: np.ndarray,
        hierarchy_level: Optional[str] = None,
        parent_id: Optional[str] = None,
        graph_node_id: Optional[str] = None,
        connected_nodes: Optional[List[str]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add an embedding to the store.
        
        Args:
            id: Unique identifier
            content: Original text content
            embedding: Vector embedding
            hierarchy_level: Hierarchical level (e.g., "root", "level_1")
            parent_id: Parent ID in hierarchy
            graph_node_id: Associated graph node ID
            connected_nodes: List of connected graph node IDs
            importance: Importance score (0-1)
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        with self._lock:
            if self.config.vector_backend == "sqlite":
                return self._add_sqlite(
                    id, content, embedding, hierarchy_level, parent_id,
                    graph_node_id, connected_nodes, importance, metadata
                )
            elif self.config.vector_backend == "chroma":
                return self._add_chroma(
                    id, content, embedding, hierarchy_level, parent_id,
                    graph_node_id, importance, metadata
                )
            elif self.config.vector_backend == "qdrant":
                return self._add_qdrant(
                    id, content, embedding, hierarchy_level, parent_id,
                    graph_node_id, importance, metadata
                )
            return False
    
    def _add_sqlite(
        self,
        id: str,
        content: str,
        embedding: np.ndarray,
        hierarchy_level: Optional[str],
        parent_id: Optional[str],
        graph_node_id: Optional[str],
        connected_nodes: Optional[List[str]],
        importance: float,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Add to SQLite backend."""
        try:
            cursor = self._db_connection.cursor()
            
            # Insert embedding
            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings 
                (id, content, embedding, hierarchy_level, parent_id, graph_node_id, importance, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    id,
                    content,
                    embedding.tobytes(),
                    hierarchy_level,
                    parent_id,
                    graph_node_id,
                    importance,
                    json.dumps(metadata) if metadata else None
                )
            )
            
            # Insert graph edges
            if connected_nodes and graph_node_id:
                for connected_id in connected_nodes:
                    cursor.execute(
                        "INSERT OR IGNORE INTO graph_edges (node_id, connected_node_id) VALUES (?, ?)",
                        (graph_node_id, connected_id)
                    )
            
            self._db_connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embedding to SQLite: {e}")
            return False
    
    def _add_chroma(
        self,
        id: str,
        content: str,
        embedding: np.ndarray,
        hierarchy_level: Optional[str],
        parent_id: Optional[str],
        graph_node_id: Optional[str],
        importance: float,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Add to ChromaDB backend."""
        try:
            chroma_metadata = {
                "hierarchy_level": hierarchy_level,
                "parent_id": parent_id,
                "graph_node_id": graph_node_id,
                "importance": importance,
            }
            if metadata:
                # Flatten metadata for ChromaDB (only str, int, float, bool)
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        chroma_metadata[key] = value
                    else:
                        chroma_metadata[key] = str(value)
            
            self._chroma_collection.add(
                ids=[id],
                documents=[content],
                embeddings=[embedding.tolist()],
                metadatas=[chroma_metadata]
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embedding to ChromaDB: {e}")
            return False
    
    def _add_qdrant(
        self,
        id: str,
        content: str,
        embedding: np.ndarray,
        hierarchy_level: Optional[str],
        parent_id: Optional[str],
        graph_node_id: Optional[str],
        importance: float,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Add to Qdrant backend."""
        try:
            payload = {
                "content": content,
                "hierarchy_level": hierarchy_level,
                "parent_id": parent_id,
                "graph_node_id": graph_node_id,
                "importance": importance,
            }
            if metadata:
                payload.update(metadata)
            
            self._qdrant_client.upsert(
                collection_name=self.config.qdrant_collection,
                points=[PointStruct(
                    id=id,
                    vector=embedding.tolist(),
                    payload=payload
                )]
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embedding to Qdrant: {e}")
            return False
    
    def get_by_hierarchy(self, hierarchy_level: Optional[str] = None, parent_id: Optional[str] = None) -> List[Tuple[str, str, np.ndarray]]:
        """
        Get embeddings filtered by hierarchy.
        
        Returns:
            List of (id, content, embedding) tuples
        """
        with self._lock:
            if self.config.vector_backend != "sqlite":
                logger.warning("Hierarchy filtering only supported with SQLite backend")
                return []
            
            try:
                cursor = self._db_connection.cursor()
                
                if hierarchy_level and parent_id:
                    cursor.execute(
                        "SELECT id, content, embedding FROM embeddings WHERE hierarchy_level = ? AND parent_id = ?",
                        (hierarchy_level, parent_id)
                    )
                elif hierarchy_level:
                    cursor.execute(
                        "SELECT id, content, embedding FROM embeddings WHERE hierarchy_level = ?",
                        (hierarchy_level,)
                    )
                elif parent_id:
                    cursor.execute(
                        "SELECT id, content, embedding FROM embeddings WHERE parent_id = ?",
                        (parent_id,)
                    )
                else:
                    cursor.execute("SELECT id, content, embedding FROM embeddings")
                
                results = []
                for row in cursor.fetchall():
                    id, content, embedding_bytes = row
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    results.append((id, content, embedding))
                
                return results
                
            except Exception as e:
                logger.error(f"Failed to get embeddings by hierarchy: {e}")
                return []
    
    def get_by_graph_nodes(self, node_ids: List[str]) -> List[Tuple[str, str, np.ndarray]]:
        """
        Get embeddings for specific graph nodes.
        
        Returns:
            List of (id, content, embedding) tuples
        """
        with self._lock:
            if self.config.vector_backend != "sqlite":
                logger.warning("Graph filtering only supported with SQLite backend")
                return []
            
            try:
                cursor = self._db_connection.cursor()
                placeholders = ','.join('?' * len(node_ids))
                cursor.execute(
                    f"SELECT id, content, embedding FROM embeddings WHERE graph_node_id IN ({placeholders})",
                    node_ids
                )
                
                results = []
                for row in cursor.fetchall():
                    id, content, embedding_bytes = row
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    results.append((id, content, embedding))
                
                return results
                
            except Exception as e:
                logger.error(f"Failed to get embeddings by graph nodes: {e}")
                return []
    
    def get_connected_nodes(self, node_id: str) -> List[str]:
        """Get IDs of nodes connected to a given node."""
        with self._lock:
            if self.config.vector_backend != "sqlite":
                return []
            
            try:
                cursor = self._db_connection.cursor()
                cursor.execute(
                    "SELECT connected_node_id FROM graph_edges WHERE node_id = ?",
                    (node_id,)
                )
                return [row[0] for row in cursor.fetchall()]
                
            except Exception as e:
                logger.error(f"Failed to get connected nodes: {e}")
                return []
    
    def get_all(self) -> List[Tuple[str, str, np.ndarray, Dict[str, Any]]]:
        """
        Get all embeddings with metadata.
        
        Returns:
            List of (id, content, embedding, metadata) tuples
        """
        with self._lock:
            if self.config.vector_backend != "sqlite":
                logger.warning("get_all only supported with SQLite backend")
                return []
            
            try:
                cursor = self._db_connection.cursor()
                cursor.execute(
                    "SELECT id, content, embedding, hierarchy_level, parent_id, graph_node_id, importance, metadata FROM embeddings"
                )
                
                results = []
                for row in cursor.fetchall():
                    id, content, embedding_bytes, hierarchy_level, parent_id, graph_node_id, importance, metadata_json = row
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    metadata = {
                        "hierarchy_level": hierarchy_level,
                        "parent_id": parent_id,
                        "graph_node_id": graph_node_id,
                        "importance": importance,
                    }
                    if metadata_json:
                        metadata.update(json.loads(metadata_json))
                    results.append((id, content, embedding, metadata))
                
                return results
                
            except Exception as e:
                logger.error(f"Failed to get all embeddings: {e}")
                return []
    
    def delete(self, id: str) -> bool:
        """Delete an embedding by ID."""
        with self._lock:
            if self.config.vector_backend == "sqlite":
                try:
                    cursor = self._db_connection.cursor()
                    cursor.execute("DELETE FROM embeddings WHERE id = ?", (id,))
                    self._db_connection.commit()
                    return cursor.rowcount > 0
                except Exception as e:
                    logger.error(f"Failed to delete embedding: {e}")
                    return False
            elif self.config.vector_backend == "chroma":
                try:
                    self._chroma_collection.delete(ids=[id])
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete embedding from ChromaDB: {e}")
                    return False
            elif self.config.vector_backend == "qdrant":
                try:
                    self._qdrant_client.delete(
                        collection_name=self.config.qdrant_collection,
                        points_selector=[id]
                    )
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete embedding from Qdrant: {e}")
                    return False
            return False
    
    def clear(self) -> bool:
        """Clear all embeddings."""
        with self._lock:
            if self.config.vector_backend == "sqlite":
                try:
                    cursor = self._db_connection.cursor()
                    cursor.execute("DELETE FROM embeddings")
                    cursor.execute("DELETE FROM graph_edges")
                    self._db_connection.commit()
                    return True
                except Exception as e:
                    logger.error(f"Failed to clear embeddings: {e}")
                    return False
            elif self.config.vector_backend == "chroma":
                try:
                    self._chroma_collection.delete()
                    return True
                except Exception as e:
                    logger.error(f"Failed to clear ChromaDB collection: {e}")
                    return False
            elif self.config.vector_backend == "qdrant":
                try:
                    self._qdrant_client.delete_collection(self.config.qdrant_collection)
                    # Recreate collection
                    dim = self.config.embedding_dimensions.get(
                        self.config.embedding_model,
                        self.config.embedding_dimensions[self.config.fallback_model]
                    )
                    self._qdrant_client.create_collection(
                        collection_name=self.config.qdrant_collection,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                    )
                    return True
                except Exception as e:
                    logger.error(f"Failed to clear Qdrant collection: {e}")
                    return False
            return False
    
    def close(self) -> None:
        """Close the store connection."""
        with self._lock:
            if self._db_connection:
                self._db_connection.close()
                self._db_connection = None


# =============================================================================
# SEMANTIC INDEX
# =============================================================================

class SemanticIndex:
    """
    Semantic indexing using vector embeddings.
    
    Filters results through hierarchical/graph context first,
    then ranks by semantic similarity to query.
    
    Usage:
        index = SemanticIndex()
        
        # Add documents
        index.add_document(
            id="doc1",
            content="Machine learning is a subset of AI...",
            hierarchy_level="root",
            graph_node_id="ml_concepts"
        )
        
        # Search with context-aware filtering
        query = SemanticQuery(
            query_text="artificial intelligence",
            hierarchy_level="root",  # Pre-filter by hierarchy
            graph_node_ids=["ml_concepts", "ai_concepts"],  # Pre-filter by graph
            top_k=5
        )
        results = index.search(query)
    """
    
    def __init__(self, config: Optional[SemanticIndexConfig] = None):
        self.config = config or SemanticIndexConfig()
        self.store = EmbeddingStore(self.config)
        self.generator = EmbeddingGenerator(self.config)
        self._lock = threading.RLock()
    
    def add_document(
        self,
        id: str,
        content: str,
        hierarchy_level: Optional[str] = None,
        parent_id: Optional[str] = None,
        graph_node_id: Optional[str] = None,
        connected_nodes: Optional[List[str]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Add a document to the semantic index.
        
        Args:
            id: Unique document identifier
            content: Document text content
            hierarchy_level: Hierarchical level (e.g., "root", "level_1")
            parent_id: Parent document ID in hierarchy
            graph_node_id: Associated graph node ID
            connected_nodes: List of connected graph node IDs
            importance: Importance score (0-1)
            metadata: Additional metadata
            embedding: Pre-computed embedding (optional)
            
        Returns:
            True if successful
        """
        with self._lock:
            # Generate embedding if not provided
            if embedding is None:
                try:
                    embedding = self.generator.generate(content)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for document {id}: {e}")
                    return False
            
            # Add to store
            return self.store.add(
                id=id,
                content=content,
                embedding=embedding,
                hierarchy_level=hierarchy_level,
                parent_id=parent_id,
                graph_node_id=graph_node_id,
                connected_nodes=connected_nodes,
                importance=importance,
                metadata=metadata
            )
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> List[bool]:
        """
        Add multiple documents in batches for efficiency.
        
        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to process per batch
            
        Returns:
            List of success booleans
        """
        with self._lock:
            results = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Generate embeddings for batch
                texts = [doc["content"] for doc in batch]
                try:
                    embeddings = self.generator.generate(texts)
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch: {e}")
                    results.extend([False] * len(batch))
                    continue
                
                # Add each document with its embedding
                for doc, emb in zip(batch, embeddings):
                    success = self.store.add(
                        id=doc["id"],
                        content=doc["content"],
                        embedding=emb,
                        hierarchy_level=doc.get("hierarchy_level"),
                        parent_id=doc.get("parent_id"),
                        graph_node_id=doc.get("graph_node_id"),
                        connected_nodes=doc.get("connected_nodes"),
                        importance=doc.get("importance", 0.5),
                        metadata=doc.get("metadata")
                    )
                    results.append(success)
            
            return results
    
    def search(self, query: SemanticQuery) -> List[SemanticResult]:
        """
        Search with multi-stage filtering pipeline.
        
        Pipeline:
        1. Pre-filter by hierarchy level
        2. Pre-filter by graph connectivity
        3. Rank by semantic similarity
        4. Re-rank by recency and importance
        
        Args:
            query: SemanticQuery with filtering and ranking parameters
            
        Returns:
            List of SemanticResult objects
        """
        with self._lock:
            # Validate query
            errors = query.validate()
            if errors:
                logger.error(f"Invalid query: {errors}")
                return []
            
            # Stage 1: Pre-filter by hierarchy
            candidates = self._filter_by_hierarchy(query)
            
            # Stage 2: Pre-filter by graph connectivity
            candidates = self._filter_by_graph(query, candidates)
            
            # Stage 3: Rank by semantic similarity
            if not candidates:
                return []
            
            ranked = self._rank_by_similarity(query, candidates)
            
            # Stage 4: Re-rank by recency and importance
            final_results = self._rerank(query, ranked)
            
            return final_results[:query.top_k]
    
    def _filter_by_hierarchy(self, query: SemanticQuery) -> List[Tuple[str, str, np.ndarray, Dict[str, Any]]]:
        """Pre-filter candidates by hierarchy constraints."""
        if self.config.vector_backend != "sqlite":
            # For non-SQLite backends, get all and filter in memory
            all_docs = self.store.get_all()
            
            if query.hierarchy_level:
                all_docs = [
                    doc for doc in all_docs
                    if doc[3].get("hierarchy_level") == query.hierarchy_level
                ]
            
            if query.parent_id:
                all_docs = [
                    doc for doc in all_docs
                    if doc[3].get("parent_id") == query.parent_id
                ]
            
            return all_docs
        
        # Use SQLite filtering
        docs = self.store.get_by_hierarchy(query.hierarchy_level, query.parent_id)
        
        # Enrich with metadata
        results = []
        for id, content, embedding in docs:
            # Get full metadata
            cursor = self.store._db_connection.cursor()
            cursor.execute(
                "SELECT graph_node_id, importance, metadata FROM embeddings WHERE id = ?",
                (id,)
            )
            row = cursor.fetchone()
            if row:
                graph_node_id, importance, metadata_json = row
                metadata = {
                    "graph_node_id": graph_node_id,
                    "importance": importance,
                }
                if metadata_json:
                    metadata.update(json.loads(metadata_json))
                results.append((id, content, embedding, metadata))
        
        return results
    
    def _filter_by_graph(
        self,
        query: SemanticQuery,
        candidates: List[Tuple[str, str, np.ndarray, Dict[str, Any]]]
    ) -> List[Tuple[str, str, np.ndarray, Dict[str, Any]]]:
        """Pre-filter candidates by graph connectivity constraints."""
        if not query.graph_node_ids and not query.graph_connectivity:
            return candidates
        
        if query.graph_node_ids:
            # Filter to only specified graph nodes
            candidates = [
                c for c in candidates
                if c[3].get("graph_node_id") in query.graph_node_ids
            ]
        
        if query.graph_connectivity:
            # Additional connectivity filtering logic
            required_connections = query.graph_connectivity.get("required_connections", [])
            if required_connections:
                filtered = []
                for c in candidates:
                    node_id = c[3].get("graph_node_id")
                    if node_id:
                        connected = self.store.get_connected_nodes(node_id)
                        if any(conn in required_connections for conn in connected):
                            filtered.append(c)
                candidates = filtered
        
        return candidates
    
    def _rank_by_similarity(
        self,
        query: SemanticQuery,
        candidates: List[Tuple[str, str, np.ndarray, Dict[str, Any]]]
    ) -> List[Tuple[str, str, np.ndarray, Dict[str, Any], float]]:
        """Rank candidates by cosine similarity to query."""
        # Generate query embedding
        query_embedding = self.generator.generate(query.query_text)
        
        # Calculate cosine similarity for each candidate
        scored = []
        for id, content, embedding, metadata in candidates:
            similarity = cosine_similarity(query_embedding, embedding)
            if similarity >= query.similarity_threshold:
                scored.append((id, content, embedding, metadata, similarity))
        
        # Sort by similarity (descending)
        scored.sort(key=lambda x: x[4], reverse=True)
        
        return scored
    
    def _rerank(
        self,
        query: SemanticQuery,
        candidates: List[Tuple[str, str, np.ndarray, Dict[str, Any], float]]
    ) -> List[SemanticResult]:
        """Re-rank by combining similarity with recency and importance."""
        results = []
        
        for id, content, embedding, metadata, similarity in candidates:
            # Get additional scores
            importance = metadata.get("importance", 0.5)
            
            # Calculate recency score if timestamp available
            recency_score = 0.5  # Default
            created_at_str = metadata.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    # Score based on how recent (1.0 = now, 0.0 = old)
                    age_days = (datetime.now() - created_at).days
                    recency_score = max(0.0, 1.0 - (age_days / 365.0))  # Decay over 1 year
                except:
                    pass
            
            # Combined score
            similarity_weight = 1.0 - query.recency_weight - query.importance_weight
            combined_score = (
                similarity_weight * similarity +
                query.recency_weight * recency_score +
                query.importance_weight * importance
            )
            
            # Build result
            result = SemanticResult(
                id=id,
                content=content,
                similarity_score=combined_score,
                hierarchy_level=metadata.get("hierarchy_level"),
                parent_id=metadata.get("parent_id"),
                graph_node_id=metadata.get("graph_node_id"),
                importance=importance,
                metadata=metadata
            )
            results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results
    
    def delete_document(self, id: str) -> bool:
        """Delete a document from the index."""
        with self._lock:
            return self.store.delete(id)
    
    def clear(self) -> bool:
        """Clear all documents from the index."""
        with self._lock:
            return self.store.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self._lock:
            if self.config.vector_backend == "sqlite":
                try:
                    cursor = self.store._db_connection.cursor()
                    cursor.execute("SELECT COUNT(*) FROM embeddings")
                    count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(DISTINCT hierarchy_level) FROM embeddings")
                    hierarchy_levels = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(DISTINCT graph_node_id) FROM embeddings")
                    graph_nodes = cursor.fetchone()[0]
                    
                    return {
                        "document_count": count,
                        "hierarchy_levels": hierarchy_levels,
                        "graph_nodes": graph_nodes,
                        "backend": self.config.vector_backend,
                        "embedding_model": self.config.embedding_model,
                    }
                except Exception as e:
                    logger.error(f"Failed to get stats: {e}")
                    return {"error": str(e)}
            else:
                return {
                    "backend": self.config.vector_backend,
                    "embedding_model": self.config.embedding_model,
                }
    
    def close(self) -> None:
        """Close the index and release resources."""
        with self._lock:
            self.store.close()


# =============================================================================
# SEMANTIC SEARCH FUNCTION
# =============================================================================

def semantic_search(
    query_text: str,
    documents: Optional[List[Dict[str, Any]]] = None,
    hierarchy_level: Optional[str] = None,
    graph_node_ids: Optional[List[str]] = None,
    top_k: int = 10,
    similarity_threshold: float = 0.7,
    config: Optional[SemanticIndexConfig] = None,
    index: Optional[SemanticIndex] = None
) -> List[SemanticResult]:
    """
    Search with multi-stage filtering pipeline.
    
    This is a convenience function that either uses an existing index or
    searches a provided list of documents.
    
    Args:
        query_text: The search query
        documents: Optional list of documents to search (each with 'id', 'content', etc.)
        hierarchy_level: Pre-filter by hierarchy level
        graph_node_ids: Pre-filter by graph node IDs
        top_k: Number of results to return
        similarity_threshold: Minimum similarity threshold
        config: Optional configuration
        index: Optional existing SemanticIndex instance
        
    Returns:
        List of SemanticResult objects
        
    Example:
        >>> results = semantic_search(
        ...     query_text="machine learning algorithms",
        ...     hierarchy_level="root",
        ...     top_k=5
        ... )
        >>> for r in results:
        ...     print(f"{r.id}: {r.similarity_score:.3f}")
    """
    # Create or use index
    if index is None:
        index = SemanticIndex(config)
        
        # Add documents if provided
        if documents:
            index.add_documents(documents)
    
    # Build query
    query = SemanticQuery(
        query_text=query_text,
        hierarchy_level=hierarchy_level,
        graph_node_ids=graph_node_ids,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )
    
    # Search
    return index.search(query)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (0-1)
    """
    if a.ndim == 1 and b.ndim == 1:
        return float(np.dot(a, b) / (norm(a) * norm(b)))
    elif a.ndim == 1 and b.ndim == 2:
        return float(np.dot(a, b.T) / (norm(a) * norm(b, axis=1)))
    elif a.ndim == 2 and b.ndim == 1:
        return float(np.dot(a, b) / (norm(a, axis=1) * norm(b)))
    else:
        return float(np.dot(a, b.T) / (np.outer(norm(a, axis=1), norm(b, axis=1))))


def batch_process_embeddings(
    texts: List[str],
    batch_size: int = 32,
    config: Optional[SemanticIndexConfig] = None
) -> List[np.ndarray]:
    """
    Process embeddings in batches for efficiency.
    
    Args:
        texts: List of texts to embed
        batch_size: Batch size
        config: Optional configuration
        
    Returns:
        List of embedding vectors
    """
    generator = EmbeddingGenerator(config)
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = generator.generate(batch)
        results.extend(embeddings)
    
    return results


# =============================================================================
# INITIALIZATION
# =============================================================================

def create_semantic_index(
    cache_dir: str = "./knowledge_semantic_cache",
    embedding_model: str = "text-embedding-3-small",
    vector_backend: str = "sqlite"
) -> SemanticIndex:
    """
    Create a new semantic index with the specified configuration.
    
    Args:
        cache_dir: Directory for caching
        embedding_model: Embedding model to use
        vector_backend: Backend type (sqlite, chroma, qdrant)
        
    Returns:
        Configured SemanticIndex instance
    """
    config = SemanticIndexConfig(
        cache_dir=cache_dir,
        embedding_model=embedding_model,
        vector_backend=vector_backend
    )
    return SemanticIndex(config)


# Auto-initialize on import (optional)
_default_index: Optional[SemanticIndex] = None

def get_default_index() -> SemanticIndex:
    """Get or create the default semantic index."""
    global _default_index
    if _default_index is None:
        _default_index = create_semantic_index()
    return _default_index


if __name__ == "__main__":
    # Example usage
    print("🔍 Knowledge Semantic Index Demo")
    print("=" * 50)
    
    # Create index
    index = create_semantic_index()
    
    # Add documents
    documents = [
        {
            "id": "doc1",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "hierarchy_level": "root",
            "graph_node_id": "ml",
            "importance": 0.9
        },
        {
            "id": "doc2",
            "content": "Deep learning uses neural networks with multiple layers to extract high-level features.",
            "hierarchy_level": "level_1",
            "parent_id": "doc1",
            "graph_node_id": "deep_learning",
            "importance": 0.8
        },
        {
            "id": "doc3",
            "content": "Natural language processing helps computers understand and generate human language.",
            "hierarchy_level": "root",
            "graph_node_id": "nlp",
            "importance": 0.85
        }
    ]
    
    index.add_documents(documents)
    
    # Search
    query = SemanticQuery(
        query_text="neural networks and AI",
        top_k=3
    )
    results = index.search(query)
    
    print(f"\nFound {len(results)} results:")
    for r in results:
        print(f"  - {r.id}: {r.similarity_score:.3f} ({r.content[:50]}...)")
    
    # Search with hierarchy filter
    query_filtered = SemanticQuery(
        query_text="machine learning",
        hierarchy_level="root",
        top_k=2
    )
    results_filtered = index.search(query_filtered)
    
    print(f"\nFiltered results (root level only):")
    for r in results_filtered:
        print(f"  - {r.id}: {r.similarity_score:.3f}")
    
    # Cleanup
    index.clear()
    index.close()
    
    print("\n[OK] Demo completed")
