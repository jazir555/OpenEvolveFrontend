"""
Knowledge Engine Hierarchical Integration - 4-Layer Unified Indexing System

Integrates four specialized indexes into a cohesive knowledge engine:
1. Hierarchical Index - Importance-based memory organization (CORE -> GRANULAR)
2. Graph Index - Logical relationship preservation system
3. Hash Index - Deduplication layer using multiple hash strategies
4. Semantic Index - Vector embedding-based semantic search

Key Features:
- Transparent indexing: knowledge added automatically goes through all 4 indexes
- Query-time curation: queries automatically use context assembler
- Backwards compatibility: existing KnowledgeEngine API still works
- Configurable indexing levels (can disable layers if needed)
- Automatic maintenance jobs (promote/demote, deduplicate)
- Thread-safe operations with proper error handling and fallbacks

Usage:
    >>> from knowledge_engine_hierarchical_integration import (
    ...     EnhancedKnowledgeEngine, create_enhanced_knowledge_engine
    ... )
    >>> engine = create_enhanced_knowledge_engine(
    ...     storage_path="./knowledge_db",
    ...     enable_hierarchical=True,
    ...     enable_graph=True,
    ...     enable_hash=True,
    ...     enable_semantic=True
    ... )
    >>> engine.add_knowledge_with_indexing(content="Important concept")
    >>> results = engine.query_with_context_curation(query="search term")

Author: OpenEvolve AI
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union, Iterator
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# OPTIONAL IMPORTS WITH FALLBACKS
# =============================================================================

# Import the four index modules
try:
    from knowledge_hierarchical_index import (
        HierarchicalIndex, MemoryNode as HierarchicalMemoryNode,
        MemoryLevel, ImportanceScorer
    )
    HIERARCHICAL_AVAILABLE = True
except ImportError as e:
    HIERARCHICAL_AVAILABLE = False
    logger.warning(f"Hierarchical index not available: {e}")

try:
    from knowledge_graph_index import (
        GraphIndex, MemoryNode as GraphMemoryNode,
        RelationshipType, RelationshipEdge, TraversalMode
    )
    GRAPH_AVAILABLE = True
except ImportError as e:
    GRAPH_AVAILABLE = False
    logger.warning(f"Graph index not available: {e}")

try:
    from knowledge_hash_index import (
        HashIndex, HashIndexConfig, compute_md5_hash,
        compute_simhash, compute_minhash
    )
    HASH_AVAILABLE = True
except ImportError as e:
    HASH_AVAILABLE = False
    logger.warning(f"Hash index not available: {e}")

try:
    from knowledge_semantic_index import (
        SemanticIndex, SemanticIndexConfig, SemanticQuery,
        generate_embedding, cosine_similarity
    )
    SEMANTIC_AVAILABLE = True
except ImportError as e:
    SEMANTIC_AVAILABLE = False
    logger.warning(f"Semantic index not available: {e}")

# Import existing knowledge engine patterns
try:
    from knowledge_base import KnowledgeBase, KnowledgeQuery, KnowledgeArtifact
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False
    logger.warning("KnowledgeBase not available, using fallback")

try:
    from sovereign_data_models import KnowledgeArtifact, generate_id
    DATA_MODELS_AVAILABLE = True
except ImportError:
    DATA_MODELS_AVAILABLE = False
    logger.warning("Data models not available, using fallback")


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class IndexingLevel(Enum):
    """Configuration for which indexes to enable."""
    ALL = "all"
    HIERARCHICAL_ONLY = "hierarchical_only"
    GRAPH_ONLY = "graph_only"
    HASH_ONLY = "hash_only"
    SEMANTIC_ONLY = "semantic_only"
    HYBRID = "hybrid"  # Hierarchical + Graph + Hash
    NO_SEMANTIC = "no_semantic"  # All except semantic (for performance)


class MaintenanceJobType(Enum):
    """Types of automatic maintenance jobs."""
    PROMOTE_DEMOTE = "promote_demote"
    DEDUPLICATE = "deduplicate"
    PRUNE_OLD = "prune_old"
    UPDATE_EDGES = "update_edges"
    REINDEX = "reindex"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EnhancedKnowledgeEngineConfig:
    """Configuration for Enhanced Knowledge Engine with 4-layer indexing."""
    
    # Storage paths
    storage_path: str = "./knowledge_engine_v2"
    hierarchical_db_path: Optional[str] = None
    graph_db_path: Optional[str] = None
    hash_db_path: Optional[str] = None
    semantic_cache_dir: Optional[str] = None
    
    # Index enablement
    enable_hierarchical: bool = True
    enable_graph: bool = True
    enable_hash: bool = True
    enable_semantic: bool = True
    
    # Performance settings
    max_query_results: int = 50
    default_top_k: int = 10
    similarity_threshold: float = 0.7
    
    # Maintenance settings
    enable_auto_maintenance: bool = True
    maintenance_interval_hours: int = 24
    auto_deduplicate: bool = True
    auto_promote_demote: bool = True
    
    # Context curation settings
    enable_context_curation: bool = True
    max_context_tokens: int = 4000
    recency_weight: float = 0.3
    importance_weight: float = 0.3
    relevance_weight: float = 0.4
    
    # Thread safety
    thread_safe: bool = True
    
    # API keys (for semantic index)
    openai_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived paths."""
        base_path = Path(self.storage_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        if self.hierarchical_db_path is None:
            self.hierarchical_db_path = str(base_path / "hierarchical_index.db")
        if self.graph_db_path is None:
            self.graph_db_path = str(base_path / "graph_index.db")
        if self.hash_db_path is None:
            self.hash_db_path = str(base_path / "hash_index.db")
        if self.semantic_cache_dir is None:
            self.semantic_cache_dir = str(base_path / "semantic_cache")
        
        # Check availability and adjust
        if not HIERARCHICAL_AVAILABLE:
            self.enable_hierarchical = False
        if not GRAPH_AVAILABLE:
            self.enable_graph = False
        if not HASH_AVAILABLE:
            self.enable_hash = False
        if not SEMANTIC_AVAILABLE:
            self.enable_semantic = False


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class UnifiedKnowledgeEntry:
    """
    Unified knowledge entry that spans all 4 indexes.
    
    This dataclass tracks a single piece of knowledge across all indexing systems,
    maintaining references to its position in each index.
    """
    
    # Core identification
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    content_type: str = "text"  # text, code, json, etc.
    
    # Index references
    hierarchical_node_id: Optional[str] = None
    graph_node_id: Optional[str] = None
    hash_signature: Optional[str] = None
    semantic_embedding_id: Optional[str] = None
    
    # Metadata
    title: Optional[str] = None
    domain: str = "general"
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    confidence: float = 0.5
    
    # Temporal tracking
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    # Importance tracking (from hierarchical index)
    importance_score: float = 0.5
    memory_level: Optional[str] = None
    
    # Graph relationships
    related_entries: List[str] = field(default_factory=list)
    parent_entry_id: Optional[str] = None
    child_entry_ids: List[str] = field(default_factory=list)
    
    # Additional flexible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "content_type": self.content_type,
            "hierarchical_node_id": self.hierarchical_node_id,
            "graph_node_id": self.graph_node_id,
            "hash_signature": self.hash_signature,
            "semantic_embedding_id": self.semantic_embedding_id,
            "title": self.title,
            "domain": self.domain,
            "tags": self.tags,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "importance_score": self.importance_score,
            "memory_level": self.memory_level,
            "related_entries": self.related_entries,
            "parent_entry_id": self.parent_entry_id,
            "child_entry_ids": self.child_entry_ids,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedKnowledgeEntry":
        """Create from dictionary."""
        def parse_datetime(value):
            if isinstance(value, str):
                return datetime.fromisoformat(value)
            return value
        
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            content=data.get("content", ""),
            content_type=data.get("content_type", "text"),
            hierarchical_node_id=data.get("hierarchical_node_id"),
            graph_node_id=data.get("graph_node_id"),
            hash_signature=data.get("hash_signature"),
            semantic_embedding_id=data.get("semantic_embedding_id"),
            title=data.get("title"),
            domain=data.get("domain", "general"),
            tags=data.get("tags", []),
            source=data.get("source"),
            confidence=data.get("confidence", 0.5),
            created_at=parse_datetime(data.get("created_at", datetime.now())),
            updated_at=parse_datetime(data.get("updated_at", datetime.now())),
            access_count=data.get("access_count", 0),
            last_accessed=parse_datetime(data.get("last_accessed")),
            importance_score=data.get("importance_score", 0.5),
            memory_level=data.get("memory_level"),
            related_entries=data.get("related_entries", []),
            parent_entry_id=data.get("parent_entry_id"),
            child_entry_ids=data.get("child_entry_ids", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CuratedQueryResult:
    """Query result with curation information."""
    
    entry: UnifiedKnowledgeEntry
    relevance_score: float = 0.0
    semantic_similarity: float = 0.0
    graph_distance: Optional[int] = None
    hierarchical_level: Optional[str] = None
    importance_score: float = 0.0
    recency_score: float = 0.0
    combined_score: float = 0.0
    
    # Context information
    related_entries: List[str] = field(default_factory=list)
    context_path: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "relevance_score": self.relevance_score,
            "semantic_similarity": self.semantic_similarity,
            "graph_distance": self.graph_distance,
            "hierarchical_level": self.hierarchical_level,
            "importance_score": self.importance_score,
            "recency_score": self.recency_score,
            "combined_score": self.combined_score,
            "related_entries": self.related_entries,
            "context_path": self.context_path,
        }


@dataclass
class MaintenanceJobResult:
    """Result of a maintenance job."""
    job_type: MaintenanceJobType
    success: bool
    items_processed: int = 0
    items_affected: int = 0
    details: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# THREAD SAFETY UTILITIES
# =============================================================================

class ThreadSafeIndexManager:
    """Manages thread-safe access to indexes."""
    
    def __init__(self):
        self._locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
    
    @contextmanager
    def acquire_lock(self, index_name: str):
        """Acquire lock for a specific index."""
        lock = self._locks[index_name]
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
    
    def get_lock(self, index_name: str) -> threading.RLock:
        """Get the lock for a specific index."""
        return self._locks[index_name]


# =============================================================================
# ENHANCED KNOWLEDGE ENGINE
# =============================================================================

class EnhancedKnowledgeEngine:
    """
    Enhanced knowledge engine with 4-layer hierarchical indexing.
    Backwards compatible with existing KnowledgeEngine API.
    
    The 4 layers work together:
    1. Hierarchical Index - Organizes by importance (CORE -> GRANULAR)
    2. Graph Index - Preserves logical relationships between entries
    3. Hash Index - Deduplicates content using multiple hash strategies
    4. Semantic Index - Enables meaning-based search with embeddings
    
    Integration features:
    - Transparent indexing: knowledge added automatically goes through all 4 indexes
    - Query-time curation: queries automatically use context assembler
    - Backwards compatibility: existing API still works
    - Configurable indexing levels (can disable layers if needed)
    - Automatic maintenance jobs (promote/demote, deduplicate)
    """
    
    def __init__(self, config: Optional[EnhancedKnowledgeEngineConfig] = None):
        """
        Initialize the enhanced knowledge engine.
        
        Args:
            config: Configuration for the engine. Uses defaults if None.
        """
        self.config = config or EnhancedKnowledgeEngineConfig()
        
        # Initialize thread safety
        self._lock_manager = ThreadSafeIndexManager()
        self._master_lock = threading.RLock()
        
        # Initialize indexes
        self._hierarchical_index: Optional[Any] = None
        self._graph_index: Optional[Any] = None
        self._hash_index: Optional[Any] = None
        self._semantic_index: Optional[Any] = None
        
        # Initialize entry registry
        self._entries: Dict[str, UnifiedKnowledgeEntry] = {}
        self._entry_registry_path = Path(self.config.storage_path) / "entry_registry.json"
        
        # Initialize legacy knowledge base for backwards compatibility
        self._legacy_kb: Optional[Any] = None
        
        # Statistics tracking
        self._stats = {
            "total_entries": 0,
            "queries_served": 0,
            "maintenance_jobs_run": 0,
            "deduplications": 0,
            "promotions": 0,
            "demotions": 0,
        }
        
        # Initialize all components
        self._initialize_indexes()
        self._load_entry_registry()
        
        logger.info(f"EnhancedKnowledgeEngine initialized with {len(self._entries)} entries")
    
    def _initialize_indexes(self):
        """Initialize all 4 index layers."""
        # Initialize hierarchical index
        if self.config.enable_hierarchical and HIERARCHICAL_AVAILABLE:
            try:
                self._hierarchical_index = HierarchicalIndex(
                    db_path=self.config.hierarchical_db_path
                )
                logger.info("Hierarchical index initialized")
            except Exception as e:
                logger.error(f"Failed to initialize hierarchical index: {e}")
                self._hierarchical_index = None
        
        # Initialize graph index
        if self.config.enable_graph and GRAPH_AVAILABLE:
            try:
                self._graph_index = GraphIndex(db_path=self.config.graph_db_path)
                logger.info("Graph index initialized")
            except Exception as e:
                logger.error(f"Failed to initialize graph index: {e}")
                self._graph_index = None
        
        # Initialize hash index
        if self.config.enable_hash and HASH_AVAILABLE:
            try:
                hash_config = HashIndexConfig(db_path=self.config.hash_db_path)
                self._hash_index = HashIndex(config=hash_config)
                logger.info("Hash index initialized")
            except Exception as e:
                logger.error(f"Failed to initialize hash index: {e}")
                self._hash_index = None
        
        # Initialize semantic index
        if self.config.enable_semantic and SEMANTIC_AVAILABLE:
            try:
                semantic_config = SemanticIndexConfig(
                    cache_dir=self.config.semantic_cache_dir
                )
                self._semantic_index = SemanticIndex(config=semantic_config)
                logger.info("Semantic index initialized")
            except Exception as e:
                logger.error(f"Failed to initialize semantic index: {e}")
                self._semantic_index = None
        
        # Initialize legacy knowledge base
        if KNOWLEDGE_BASE_AVAILABLE:
            try:
                legacy_path = Path(self.config.storage_path) / "legacy_kb.json"
                self._legacy_kb = KnowledgeBase(storage_path=str(legacy_path))
                logger.info("Legacy KnowledgeBase initialized for backwards compatibility")
            except Exception as e:
                logger.error(f"Failed to initialize legacy knowledge base: {e}")
                self._legacy_kb = None
    
    def _load_entry_registry(self):
        """Load the entry registry from disk."""
        if self._entry_registry_path.exists():
            try:
                with open(self._entry_registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry_data in data.get("entries", []):
                        entry = UnifiedKnowledgeEntry.from_dict(entry_data)
                        self._entries[entry.entry_id] = entry
                    self._stats = data.get("stats", self._stats)
                logger.info(f"Loaded {len(self._entries)} entries from registry")
            except Exception as e:
                logger.error(f"Failed to load entry registry: {e}")
    
    def _save_entry_registry(self):
        """Save the entry registry to disk."""
        try:
            data = {
                "entries": [e.to_dict() for e in self._entries.values()],
                "stats": self._stats,
                "saved_at": datetime.now().isoformat(),
            }
            with open(self._entry_registry_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save entry registry: {e}")
    
    # ========================================================================
    # CORE API: Add Knowledge with Indexing
    # ========================================================================
    
    def add_knowledge_with_indexing(
        self,
        content: str,
        title: Optional[str] = None,
        domain: str = "general",
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        content_type: str = "text",
        parent_id: Optional[str] = None,
        related_ids: Optional[List[str]] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_deduplication: bool = False,
    ) -> UnifiedKnowledgeEntry:
        """
        Add knowledge through all 4 indexes simultaneously.
        
        This method:
        1. Checks for duplicates using hash index
        2. Adds to hierarchical index with importance scoring
        3. Adds to graph index with relationships
        4. Adds to semantic index for vector search
        5. Registers in unified entry registry
        
        Args:
            content: The knowledge content to add
            title: Optional title for the entry
            domain: Domain classification
            tags: Searchable tags
            source: Source of the knowledge
            content_type: Type of content (text, code, json, etc.)
            parent_id: Parent entry ID for hierarchical organization
            related_ids: Related entry IDs for graph relationships
            importance: Explicit importance score (0.0-1.0)
            metadata: Additional metadata
            skip_deduplication: If True, don't check for duplicates
            
        Returns:
            UnifiedKnowledgeEntry with references to all indexes
        """
        with self._master_lock:
            # Create entry
            entry = UnifiedKnowledgeEntry(
                entry_id=str(uuid.uuid4()),
                content=content,
                title=title,
                domain=domain,
                tags=tags or [],
                source=source,
                content_type=content_type,
                parent_entry_id=parent_id,
                related_entries=related_ids or [],
                importance_score=importance or 0.5,
                metadata=metadata or {},
            )
            
            # Step 1: Hash Index - Check for duplicates
            if self._hash_index and not skip_deduplication:
                with self._lock_manager.acquire_lock("hash"):
                    existing = self._hash_index.find_duplicate(content)
                    if existing:
                        logger.info(f"Duplicate detected, merging with existing entry")
                        entry = self._merge_with_existing(entry, existing)
                        self._stats["deduplications"] += 1
                        return entry
                    
                    # Add to hash index
                    hash_sig = self._hash_index.add_content(
                        content=content,
                        content_id=entry.entry_id
                    )
                    entry.hash_signature = hash_sig
            
            # Step 2: Hierarchical Index - Importance-based organization
            if self._hierarchical_index:
                with self._lock_manager.acquire_lock("hierarchical"):
                    h_node = HierarchicalMemoryNode(
                        content=content,
                        domain=domain,
                        tags=tags or [],
                        parent_id=parent_id,
                        user_importance=importance or 0.5,
                    )
                    h_node_id = self._hierarchical_index.add_node(h_node)
                    entry.hierarchical_node_id = h_node_id
                    entry.memory_level = h_node.level.to_string()
                    entry.importance_score = h_node.importance_score
            
            # Step 3: Graph Index - Relationship preservation
            if self._graph_index:
                with self._lock_manager.acquire_lock("graph"):
                    g_node = GraphMemoryNode(
                        node_id=entry.entry_id,  # Use same ID for consistency
                        content=content,
                        metadata={
                            "domain": domain,
                            "tags": tags or [],
                            "title": title,
                            "source": source,
                        }
                    )
                    self._graph_index.add_node(g_node)
                    entry.graph_node_id = g_node.node_id
                    
                    # Add relationships
                    if parent_id and parent_id in self._entries:
                        self._graph_index.add_relationship(
                            source_id=parent_id,
                            target_id=entry.entry_id,
                            relationship_type=RelationshipType.PART_OF
                        )
                    
                    for related_id in (related_ids or []):
                        if related_id in self._entries:
                            self._graph_index.add_relationship(
                                source_id=entry.entry_id,
                                target_id=related_id,
                                relationship_type=RelationshipType.SEMANTIC
                            )
            
            # Step 4: Semantic Index - Vector embedding
            if self._semantic_index:
                with self._lock_manager.acquire_lock("semantic"):
                    try:
                        embedding_id = self._semantic_index.add_content(
                            content=content,
                            content_id=entry.entry_id,
                            metadata=entry.to_dict()
                        )
                        entry.semantic_embedding_id = embedding_id
                    except Exception as e:
                        logger.warning(f"Failed to add to semantic index: {e}")
            
            # Step 5: Register in unified registry
            self._entries[entry.entry_id] = entry
            self._stats["total_entries"] = len(self._entries)
            
            # Add to legacy knowledge base for backwards compatibility
            if self._legacy_kb and DATA_MODELS_AVAILABLE:
                self._add_to_legacy_kb(entry)
            
            # Save registry
            self._save_entry_registry()
            
            logger.info(f"Added knowledge entry {entry.entry_id} with {len([x for x in [entry.hierarchical_node_id, entry.graph_node_id, entry.hash_signature, entry.semantic_embedding_id] if x])} indexes")
            
            return entry
    
    def _merge_with_existing(
        self,
        new_entry: UnifiedKnowledgeEntry,
        existing_hash_entry: Dict[str, Any]
    ) -> UnifiedKnowledgeEntry:
        """Merge new entry with existing duplicate."""
        existing_id = existing_hash_entry.get("content_id")
        if existing_id and existing_id in self._entries:
            existing = self._entries[existing_id]
            
            # Update access count and timestamps
            existing.access_count += 1
            existing.last_accessed = datetime.now()
            
            # Merge tags
            existing.tags = list(set(existing.tags + new_entry.tags))
            
            # Update importance if new entry has higher importance
            if new_entry.importance_score > existing.importance_score:
                existing.importance_score = new_entry.importance_score
            
            # Merge metadata
            existing.metadata.update(new_entry.metadata)
            
            # Save updates
            self._save_entry_registry()
            
            return existing
        
        return new_entry
    
    def _add_to_legacy_kb(self, entry: UnifiedKnowledgeEntry):
        """Add entry to legacy knowledge base for backwards compatibility."""
        try:
            artifact = KnowledgeArtifact(
                artifact_id=entry.entry_id,
                artifact_type=entry.content_type,
                title=entry.title or entry.content[:50],
                description=entry.content,
                domain=entry.domain,
                confidence=entry.confidence,
                tags=entry.tags,
                metadata=entry.metadata
            )
            self._legacy_kb.store_artifact(artifact)
        except Exception as e:
            logger.warning(f"Failed to add to legacy KB: {e}")
    
    # ========================================================================
    # CORE API: Query with Context Curation
    # ========================================================================
    
    def query_with_context_curation(
        self,
        query: str,
        top_k: int = 10,
        domain: Optional[str] = None,
        min_importance: float = 0.0,
        use_semantic: bool = True,
        use_graph: bool = True,
        recency_weight: Optional[float] = None,
        importance_weight: Optional[float] = None,
        relevance_weight: Optional[float] = None,
    ) -> List[CuratedQueryResult]:
        """
        Query with automatic context rot prevention.
        
        This method performs a multi-stage query:
        1. Semantic search for meaning-based matching
        2. Graph traversal for relationship context
        3. Hierarchical filtering by importance
        4. Combined ranking with recency, importance, and relevance
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            domain: Filter by domain
            min_importance: Minimum importance threshold
            use_semantic: Whether to use semantic search
            use_graph: Whether to use graph relationships
            recency_weight: Weight for recency in ranking (uses config default if None)
            importance_weight: Weight for importance in ranking
            relevance_weight: Weight for semantic relevance in ranking
            
        Returns:
            List of CuratedQueryResult sorted by combined score
        """
        with self._master_lock:
            self._stats["queries_served"] += 1
            
            # Use config defaults if not specified
            recency_weight = recency_weight or self.config.recency_weight
            importance_weight = importance_weight or self.config.importance_weight
            relevance_weight = relevance_weight or self.config.relevance_weight
            
            results: Dict[str, CuratedQueryResult] = {}
            
            # Stage 1: Semantic Search (if enabled)
            if use_semantic and self._semantic_index:
                with self._lock_manager.acquire_lock("semantic"):
                    try:
                        semantic_results = self._semantic_index.search(
                            query=query,
                            top_k=top_k * 2,  # Get more for filtering
                            threshold=self.config.similarity_threshold
                        )
                        
                        for sr in semantic_results:
                            entry_id = sr.get("content_id")
                            if entry_id in self._entries:
                                entry = self._entries[entry_id]
                                
                                # Apply domain filter
                                if domain and entry.domain != domain:
                                    continue
                                
                                # Apply importance filter
                                if entry.importance_score < min_importance:
                                    continue
                                
                                results[entry_id] = CuratedQueryResult(
                                    entry=entry,
                                    semantic_similarity=sr.get("similarity", 0.0),
                                    importance_score=entry.importance_score,
                                    relevance_score=sr.get("similarity", 0.0),
                                )
                    except Exception as e:
                        logger.warning(f"Semantic search failed: {e}")
            
            # Stage 2: Graph Traversal (if enabled and we have initial results)
            if use_graph and self._graph_index and results:
                with self._lock_manager.acquire_lock("graph"):
                    try:
                        for entry_id, result in list(results.items()):
                            if result.entry.graph_node_id:
                                # Find related entries through graph
                                related = self._graph_index.traverse_relationships(
                                    node_id=result.entry.graph_node_id,
                                    depth=2
                                )
                                
                                for rel in related:
                                    rel_id = rel.get("node_id")
                                    if rel_id and rel_id in self._entries and rel_id not in results:
                                        rel_entry = self._entries[rel_id]
                                        results[rel_id] = CuratedQueryResult(
                                            entry=rel_entry,
                                            graph_distance=rel.get("distance", 1),
                                            importance_score=rel_entry.importance_score,
                                            relevance_score=0.3,  # Lower base relevance for related
                                        )
                                
                                result.related_entries = [r.get("node_id") for r in related]
                    except Exception as e:
                        logger.warning(f"Graph traversal failed: {e}")
            
            # Stage 3: Fallback to hierarchical if no semantic results
            if not results and self._hierarchical_index:
                with self._lock_manager.acquire_lock("hierarchical"):
                    try:
                        # Search hierarchical index
                        h_results = self._hierarchical_index.search(
                            query=query,
                            top_k=top_k * 2
                        )
                        
                        for hr in h_results:
                            h_node_id = hr.get("node_id")
                            # Find entry with this hierarchical node ID
                            for entry in self._entries.values():
                                if entry.hierarchical_node_id == h_node_id:
                                    if domain and entry.domain != domain:
                                        continue
                                    if entry.importance_score < min_importance:
                                        continue
                                    
                                    results[entry.entry_id] = CuratedQueryResult(
                                        entry=entry,
                                        hierarchical_level=entry.memory_level,
                                        importance_score=entry.importance_score,
                                        relevance_score=hr.get("score", 0.5),
                                    )
                                    break
                    except Exception as e:
                        logger.warning(f"Hierarchical search failed: {e}")
            
            # Stage 4: Calculate combined scores and rank
            now = datetime.now()
            for result in results.values():
                # Recency score (exponential decay)
                if result.entry.last_accessed:
                    days_since_access = (now - result.entry.last_accessed).days
                    result.recency_score = max(0.0, 1.0 - (days_since_access / 30.0))
                else:
                    days_since_creation = (now - result.entry.created_at).days
                    result.recency_score = max(0.0, 1.0 - (days_since_creation / 30.0))
                
                # Combined score
                result.combined_score = (
                    relevance_weight * result.relevance_score +
                    importance_weight * result.importance_score +
                    recency_weight * result.recency_score
                )
            
            # Sort by combined score and return top_k
            sorted_results = sorted(
                results.values(),
                key=lambda r: r.combined_score,
                reverse=True
            )[:top_k]
            
            # Update access statistics
            for result in sorted_results:
                result.entry.access_count += 1
                result.entry.last_accessed = now
            
            self._save_entry_registry()
            
            return sorted_results
    
    # ========================================================================
    # BACKWARDS COMPATIBILITY API
    # ========================================================================
    
    def store_artifact(self, artifact: Any):
        """
        Backwards compatible artifact storage.
        
        Delegates to add_knowledge_with_indexing while maintaining
        compatibility with existing KnowledgeBase.store_artifact() API.
        """
        # Extract data from artifact (handles both dict and object)
        if hasattr(artifact, 'to_dict'):
            data = artifact.to_dict()
        elif hasattr(artifact, '__dict__'):
            data = artifact.__dict__
        else:
            data = artifact
        
        return self.add_knowledge_with_indexing(
            content=data.get('description', data.get('content', '')),
            title=data.get('title'),
            domain=data.get('domain', 'general'),
            tags=data.get('tags', []),
            content_type=data.get('artifact_type', 'text'),
            metadata=data.get('metadata', {})
        )
    
    def retrieve_artifacts(self, query: Any) -> List[Any]:
        """
        Backwards compatible artifact retrieval.
        
        Accepts KnowledgeQuery objects or dicts and returns
        results in the same format as KnowledgeBase.retrieve_artifacts().
        """
        # Extract query parameters
        if hasattr(query, 'to_dict'):
            query_dict = query.to_dict()
        elif hasattr(query, '__dict__'):
            query_dict = query.__dict__
        else:
            query_dict = query
        
        # Build search query from filters
        search_terms = []
        if query_dict.get('domain'):
            search_terms.append(query_dict['domain'])
        if query_dict.get('problem_type'):
            search_terms.append(query_dict['problem_type'])
        if query_dict.get('tags'):
            search_terms.extend(query_dict['tags'])
        
        query_text = ' '.join(search_terms) if search_terms else '*'
        
        results = self.query_with_context_curation(
            query=query_text,
            top_k=query_dict.get('max_results', 10),
            domain=query_dict.get('domain'),
            min_importance=query_dict.get('min_confidence', 0.0)
        )
        
        # Convert to artifact-like objects
        artifacts = []
        for result in results:
            entry = result.entry
            if DATA_MODELS_AVAILABLE:
                artifact = KnowledgeArtifact(
                    artifact_id=entry.entry_id,
                    artifact_type=entry.content_type,
                    title=entry.title or entry.content[:50],
                    description=entry.content,
                    domain=entry.domain,
                    confidence=entry.confidence,
                    tags=entry.tags,
                    metadata={
                        **entry.metadata,
                        'importance_score': entry.importance_score,
                        'memory_level': entry.memory_level,
                    }
                )
                artifacts.append(artifact)
            else:
                # Return as dict if data models not available
                artifacts.append(entry.to_dict())
        
        return artifacts
    
    def find_similar_problems(self, problem: Any, n_results: int = 5) -> List[Any]:
        """
        Backwards compatible similar problem finding.
        
        Uses semantic search to find similar entries.
        """
        # Extract problem description
        if hasattr(problem, 'to_dict'):
            problem_dict = problem.to_dict()
        elif hasattr(problem, '__dict__'):
            problem_dict = problem.__dict__
        else:
            problem_dict = problem
        
        query_text = problem_dict.get('description', problem_dict.get('title', ''))
        domain = problem_dict.get('domain')
        
        results = self.query_with_context_curation(
            query=query_text,
            top_k=n_results,
            domain=domain
        )
        
        # Convert to SimilarProblem-like objects
        similar_problems = []
        for result in results:
            entry = result.entry
            similar_problems.append({
                'problem_id': entry.entry_id,
                'title': entry.title or entry.content[:50],
                'similarity_score': result.combined_score,
                'domain': entry.domain,
                'strategy_used': entry.metadata.get('strategy', 'unknown'),
                'quality_achieved': entry.confidence,
                'why_similar': f"Semantic similarity: {result.semantic_similarity:.2f}",
                'lessons_applicable': entry.tags,
            })
        
        return similar_problems
    
    # ========================================================================
    # MIGRATION API
    # ========================================================================
    
    def migrate_from_legacy(
        self,
        legacy_kb: Any,
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Migrate existing knowledge base to new indexing system.
        
        Args:
            legacy_kb: Existing KnowledgeBase instance or path to legacy storage
            batch_size: Number of entries to process per batch
            
        Returns:
            Migration statistics
        """
        stats = {
            "total_found": 0,
            "successfully_migrated": 0,
            "failed": 0,
            "duplicates_skipped": 0,
        }
        
        # Load legacy knowledge base
        if isinstance(legacy_kb, str):
            if KNOWLEDGE_BASE_AVAILABLE:
                legacy_kb = KnowledgeBase(storage_path=legacy_kb)
            else:
                logger.error("KnowledgeBase not available for migration")
                return stats
        
        # Migrate artifacts
        if hasattr(legacy_kb, 'artifacts'):
            artifacts = legacy_kb.artifacts
            stats["total_found"] = len(artifacts)
            
            for i, artifact in enumerate(artifacts):
                try:
                    result = self.add_knowledge_with_indexing(
                        content=artifact.description if hasattr(artifact, 'description') else str(artifact),
                        title=artifact.title if hasattr(artifact, 'title') else None,
                        domain=artifact.domain if hasattr(artifact, 'domain') else "general",
                        tags=artifact.tags if hasattr(artifact, 'tags') else [],
                        content_type=artifact.artifact_type if hasattr(artifact, 'artifact_type') else "text",
                        metadata=artifact.metadata if hasattr(artifact, 'metadata') else {}
                    )
                    
                    if result.hash_signature and result.entry_id in self._entries:
                        stats["successfully_migrated"] += 1
                    else:
                        stats["duplicates_skipped"] += 1
                        
                except Exception as e:
                    logger.error(f"Failed to migrate artifact: {e}")
                    stats["failed"] += 1
                
                # Progress logging
                if (i + 1) % batch_size == 0:
                    logger.info(f"Migrated {i + 1}/{len(artifacts)} artifacts")
        
        logger.info(f"Migration complete: {stats}")
        return stats
    
    # ========================================================================
    # MAINTENANCE API
    # ========================================================================
    
    def run_maintenance_job(
        self,
        job_type: MaintenanceJobType,
        **kwargs
    ) -> MaintenanceJobResult:
        """
        Run an automatic maintenance job.
        
        Args:
            job_type: Type of maintenance job to run
            **kwargs: Job-specific parameters
            
        Returns:
            MaintenanceJobResult with details of the job execution
        """
        import time
        start_time = time.time()
        
        result = MaintenanceJobResult(
            job_type=job_type,
            success=False,
        )
        
        try:
            if job_type == MaintenanceJobType.PROMOTE_DEMOTE:
                result = self._maintenance_promote_demote(**kwargs)
            elif job_type == MaintenanceJobType.DEDUPLICATE:
                result = self._maintenance_deduplicate(**kwargs)
            elif job_type == MaintenanceJobType.PRUNE_OLD:
                result = self._maintenance_prune_old(**kwargs)
            elif job_type == MaintenanceJobType.UPDATE_EDGES:
                result = self._maintenance_update_edges(**kwargs)
            elif job_type == MaintenanceJobType.REINDEX:
                result = self._maintenance_reindex(**kwargs)
            else:
                result.errors.append(f"Unknown job type: {job_type}")
        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Maintenance job {job_type} failed: {e}")
        
        result.execution_time_seconds = time.time() - start_time
        self._stats["maintenance_jobs_run"] += 1
        
        return result
    
    def _maintenance_promote_demote(
        self,
        promotion_threshold: float = 0.8,
        demotion_threshold: float = 0.2
    ) -> MaintenanceJobResult:
        """Run promotion/demotion maintenance."""
        result = MaintenanceJobResult(
            job_type=MaintenanceJobType.PROMOTE_DEMOTE,
            success=True,
        )
        
        if not self._hierarchical_index:
            result.errors.append("Hierarchical index not available")
            return result
        
        with self._lock_manager.acquire_lock("hierarchical"):
            for entry in self._entries.values():
                if entry.hierarchical_node_id:
                    result.items_processed += 1
                    
                    # Check if promotion/demotion needed
                    if entry.importance_score >= promotion_threshold:
                        promoted = self._hierarchical_index.promote_node(
                            entry.hierarchical_node_id
                        )
                        if promoted:
                            result.items_affected += 1
                            result.details.append(f"Promoted {entry.entry_id}")
                            self._stats["promotions"] += 1
                    
                    elif entry.importance_score <= demotion_threshold:
                        demoted = self._hierarchical_index.demote_node(
                            entry.hierarchical_node_id
                        )
                        if demoted:
                            result.items_affected += 1
                            result.details.append(f"Demoted {entry.entry_id}")
                            self._stats["demotions"] += 1
        
        return result
    
    def _maintenance_deduplicate(
        self,
        similarity_threshold: float = 0.95
    ) -> MaintenanceJobResult:
        """Run deduplication maintenance."""
        result = MaintenanceJobResult(
            job_type=MaintenanceJobType.DEDUPLICATE,
            success=True,
        )
        
        if not self._hash_index:
            result.errors.append("Hash index not available")
            return result
        
        with self._lock_manager.acquire_lock("hash"):
            # Find duplicates using hash index
            duplicates = self._hash_index.find_all_duplicates(
                threshold=similarity_threshold
            )
            
            for dup_group in duplicates:
                result.items_processed += len(dup_group)
                # Merge duplicates (keep first, merge metadata into it)
                if len(dup_group) > 1:
                    primary_id = dup_group[0]
                    for dup_id in dup_group[1:]:
                        if dup_id in self._entries:
                            # Merge metadata
                            self._entries[primary_id].metadata.update(
                                self._entries[dup_id].metadata
                            )
                            # Remove duplicate
                            del self._entries[dup_id]
                            result.items_affected += 1
        
        if result.items_affected > 0:
            self._save_entry_registry()
        
        return result
    
    def _maintenance_prune_old(
        self,
        max_age_days: int = 365,
        min_access_count: int = 1
    ) -> MaintenanceJobResult:
        """Run pruning maintenance for old entries."""
        result = MaintenanceJobResult(
            job_type=MaintenanceJobType.PRUNE_OLD,
            success=True,
        )
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        with self._master_lock:
            to_remove = []
            for entry_id, entry in self._entries.items():
                result.items_processed += 1
                
                # Check if entry is old and rarely accessed
                is_old = entry.created_at < cutoff_date
                is_rarely_accessed = entry.access_count < min_access_count
                
                if is_old and is_rarely_accessed:
                    to_remove.append(entry_id)
            
            # Remove entries
            for entry_id in to_remove:
                self._remove_entry(entry_id)
                result.items_affected += 1
                result.details.append(f"Pruned {entry_id}")
        
        if result.items_affected > 0:
            self._save_entry_registry()
        
        return result
    
    def _maintenance_update_edges(self) -> MaintenanceJobResult:
        """Run graph edge update maintenance."""
        result = MaintenanceJobResult(
            job_type=MaintenanceJobType.UPDATE_EDGES,
            success=True,
        )
        
        if not self._graph_index:
            result.errors.append("Graph index not available")
            return result
        
        with self._lock_manager.acquire_lock("graph"):
            # Re-extract relationships for all entries
            for entry in self._entries.values():
                result.items_processed += 1
                if entry.graph_node_id:
                    # Update relationships based on content analysis
                    # This is a placeholder - actual implementation would
                    # use NLP to extract relationships
                    pass
        
        return result
    
    def _maintenance_reindex(self) -> MaintenanceJobResult:
        """Run full reindexing maintenance."""
        result = MaintenanceJobResult(
            job_type=MaintenanceJobType.REINDEX,
            success=True,
        )
        
        # Re-add all entries to all indexes
        entries_copy = list(self._entries.values())
        
        for entry in entries_copy:
            result.items_processed += 1
            try:
                # Re-add to hierarchical index
                if self._hierarchical_index and entry.hierarchical_node_id:
                    # Remove and re-add
                    pass  # Implementation depends on index capabilities
                
                # Re-add to semantic index
                if self._semantic_index and entry.semantic_embedding_id:
                    self._semantic_index.update_embedding(entry.entry_id, entry.content)
                    result.items_affected += 1
                    
            except Exception as e:
                result.errors.append(f"Failed to reindex {entry.entry_id}: {e}")
        
        return result
    
    def _remove_entry(self, entry_id: str):
        """Remove entry from all indexes."""
        if entry_id not in self._entries:
            return
        
        entry = self._entries[entry_id]
        
        # Remove from hierarchical index
        if self._hierarchical_index and entry.hierarchical_node_id:
            self._hierarchical_index.remove_node(entry.hierarchical_node_id)
        
        # Remove from graph index
        if self._graph_index and entry.graph_node_id:
            self._graph_index.remove_node(entry.graph_node_id)
        
        # Remove from hash index
        if self._hash_index and entry.hash_signature:
            self._hash_index.remove_content(entry.entry_id)
        
        # Remove from semantic index
        if self._semantic_index and entry.semantic_embedding_id:
            self._semantic_index.remove_content(entry.entry_id)
        
        # Remove from registry
        del self._entries[entry_id]
        self._stats["total_entries"] = len(self._entries)
    
    # ========================================================================
    # UTILITY API
    # ========================================================================
    
    def get_entry(self, entry_id: str) -> Optional[UnifiedKnowledgeEntry]:
        """Get a specific entry by ID."""
        return self._entries.get(entry_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = self._stats.copy()
        stats.update({
            "hierarchical_index_enabled": self._hierarchical_index is not None,
            "graph_index_enabled": self._graph_index is not None,
            "hash_index_enabled": self._hash_index is not None,
            "semantic_index_enabled": self._semantic_index is not None,
            "total_entries": len(self._entries),
        })
        return stats
    
    def get_entries_by_domain(self, domain: str) -> List[UnifiedKnowledgeEntry]:
        """Get all entries in a specific domain."""
        return [e for e in self._entries.values() if e.domain == domain]
    
    def get_entries_by_tag(self, tag: str) -> List[UnifiedKnowledgeEntry]:
        """Get all entries with a specific tag."""
        return [e for e in self._entries.values() if tag in e.tags]
    
    def close(self):
        """Close all indexes and save state."""
        self._save_entry_registry()
        
        # Close individual indexes
        for name, index in [
            ("hierarchical", self._hierarchical_index),
            ("graph", self._graph_index),
            ("hash", self._hash_index),
            ("semantic", self._semantic_index),
        ]:
            if index and hasattr(index, 'close'):
                try:
                    index.close()
                    logger.info(f"Closed {name} index")
                except Exception as e:
                    logger.error(f"Error closing {name} index: {e}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_enhanced_knowledge_engine(
    storage_path: str,
    enable_hierarchical: bool = True,
    enable_graph: bool = True,
    enable_hash: bool = True,
    enable_semantic: bool = True,
    openai_api_key: Optional[str] = None,
    **kwargs
) -> EnhancedKnowledgeEngine:
    """
    Factory for creating enhanced engine with all 4 indexes.
    
    This factory function provides a convenient way to create an
    EnhancedKnowledgeEngine with the desired configuration.
    
    Args:
        storage_path: Base path for all index storage
        enable_hierarchical: Enable hierarchical importance index
        enable_graph: Enable graph relationship index
        enable_hash: Enable hash-based deduplication index
        enable_semantic: Enable semantic vector index
        openai_api_key: API key for OpenAI embeddings (semantic index)
        **kwargs: Additional configuration options
        
    Returns:
        Configured EnhancedKnowledgeEngine instance
        
    Example:
        >>> engine = create_enhanced_knowledge_engine(
        ...     storage_path="./my_knowledge",
        ...     enable_hierarchical=True,
        ...     enable_graph=True,
        ...     enable_hash=True,
        ...     enable_semantic=True,
        ...     openai_api_key=os.getenv("OPENAI_API_KEY")
        ... )
        >>> entry = engine.add_knowledge_with_indexing("Important fact")
        >>> results = engine.query_with_context_curation("fact")
    """
    config = EnhancedKnowledgeEngineConfig(
        storage_path=storage_path,
        enable_hierarchical=enable_hierarchical,
        enable_graph=enable_graph,
        enable_hash=enable_hash,
        enable_semantic=enable_semantic,
        openai_api_key=openai_api_key,
        **{k: v for k, v in kwargs.items() if k in EnhancedKnowledgeEngineConfig.__dataclass_fields__}
    )
    
    engine = EnhancedKnowledgeEngine(config=config)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("EnhancedKnowledgeEngine Created")
    logger.info(f"  Storage: {storage_path}")
    logger.info(f"  Hierarchical Index: {'ENABLED' if enable_hierarchical else 'DISABLED'}")
    logger.info(f"  Graph Index: {'ENABLED' if enable_graph else 'DISABLED'}")
    logger.info(f"  Hash Index: {'ENABLED' if enable_hash else 'DISABLED'}")
    logger.info(f"  Semantic Index: {'ENABLED' if enable_semantic else 'DISABLED'}")
    logger.info("=" * 60)
    
    return engine


# =============================================================================
# BACKWARDS COMPATIBILITY ALIAS
# =============================================================================

# Alias for backwards compatibility with code expecting KnowledgeEngineV2
KnowledgeEngineV2 = EnhancedKnowledgeEngine


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("EnhancedKnowledgeEngine - 4-Layer Hierarchical Indexing System")
    print("=" * 70)
    
    # Create engine with all indexes enabled
    engine = create_enhanced_knowledge_engine(
        storage_path="./test_enhanced_engine",
        enable_hierarchical=True,
        enable_graph=True,
        enable_hash=True,
        enable_semantic=False,  # Disabled for testing without API key
    )
    
    # Test adding knowledge
    print("\n--- Testing add_knowledge_with_indexing ---")
    entry1 = engine.add_knowledge_with_indexing(
        content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        title="Machine Learning Definition",
        domain="AI",
        tags=["machine-learning", "AI", "definition"],
        importance=0.8
    )
    print(f"Added entry: {entry1.entry_id}")
    print(f"  - Hierarchical node: {entry1.hierarchical_node_id}")
    print(f"  - Graph node: {entry1.graph_node_id}")
    print(f"  - Hash signature: {entry1.hash_signature}")
    print(f"  - Memory level: {entry1.memory_level}")
    
    entry2 = engine.add_knowledge_with_indexing(
        content="Deep learning uses neural networks with multiple layers to model complex patterns.",
        title="Deep Learning Overview",
        domain="AI",
        tags=["deep-learning", "neural-networks", "AI"],
        parent_id=entry1.entry_id,
        importance=0.75
    )
    print(f"\nAdded entry: {entry2.entry_id}")
    print(f"  - Parent: {entry2.parent_entry_id}")
    
    # Test query
    print("\n--- Testing query_with_context_curation ---")
    results = engine.query_with_context_curation(
        query="artificial intelligence learning",
        top_k=5
    )
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  - {r.entry.title or r.entry.content[:40]}... (score: {r.combined_score:.3f})")
    
    # Test stats
    print("\n--- Engine Statistics ---")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test maintenance
    print("\n--- Testing Maintenance Job ---")
    result = engine.run_maintenance_job(MaintenanceJobType.PROMOTE_DEMOTE)
    print(f"Job completed: {result.job_type.value}")
    print(f"  Items processed: {result.items_processed}")
    print(f"  Items affected: {result.items_affected}")
    
    # Cleanup
    engine.close()
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
