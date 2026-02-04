"""
Knowledge Graph Index - Logical Relationship Preservation System

This module implements a graph indexing system for preserving logical relationships
between memories. It prevents "context rot" by maintaining connections between ideas
across messages through a persistent graph structure.

Key Features:
- Graph-based index with typed relationships (CAUSAL, TEMPORAL, SEMANTIC, etc.)
- Automatic relationship extraction from content
- Graph traversal (BFS, DFS) for context recovery
- Path finding between distant memories
- Community detection for memory clustering
- SQLite persistence with thread-safe operations
- NetworkX integration for advanced graph operations

Usage:
    >>> from knowledge_graph_index import GraphIndex, RelationshipType
    >>> index = GraphIndex("memory_graph.db")
    >>> node_id = index.add_node("System design requires careful planning")
    >>> index.add_node("Therefore we need modular architecture", 
    ...                relationships=[(node_id, RelationshipType.CAUSAL)])
    >>> related = index.traverse_relationships(node_id, depth=2)
"""

import logging
import sqlite3
import threading
import json
import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from collections import defaultdict
from contextlib import contextmanager
import uuid
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("networkx not available. Install with: pip install networkx")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# =============================================================================
# Enums and Constants
# =============================================================================

class RelationshipType(Enum):
    """
    Types of relationships between memory nodes.
    
    These relationship types capture different dimensions of logical connection
    between ideas, enabling rich semantic traversal of the knowledge graph.
    """
    CAUSAL = "causal"           # A causes B (because, therefore, leads to)
    TEMPORAL = "temporal"       # Time-based (then, after, before, during)
    SEMANTIC = "semantic"       # Meaning similarity (similar, related to)
    REFERENTIAL = "referential" # Reference/link (refers to, about, regarding)
    SEQUENTIAL = "sequential"   # Order-based (step 1 → step 2, next)
    CONTRADICTORY = "contradictory"  # Opposition (however, but, unlike)
    SUPPORTING = "supporting"   # Evidence/support (for example, evidence)
    PART_OF = "part_of"         # Composition (component of, subset)
    DEPENDS_ON = "depends_on"   # Dependency (requires, needs)
    GENERALIZES = "generalizes" # Abstraction (general case of)
    SPECIFIES = "specifies"     # Specialization (specific instance of)
    EQUIVALENT = "equivalent"   # Synonymy (same as, equivalent to)


class TraversalMode(Enum):
    """Traversal modes for graph navigation."""
    BFS = "breadth_first"       # Breadth-first search
    DFS = "depth_first"         # Depth-first search
    WEIGHTED = "weighted"       # Weighted by relationship strength
    TEMPORAL = "temporal"       # Follow temporal ordering
    CAUSAL = "causal"           # Follow causal chains


class NodeType(Enum):
    """Types of memory nodes."""
    CONCEPT = "concept"         # Abstract concept
    FACT = "fact"               # Concrete fact
    DECISION = "decision"       # Decision point
    QUESTION = "question"       # Open question
    HYPOTHESIS = "hypothesis"   # Proposed explanation
    CONCLUSION = "conclusion"   # Derived conclusion
    ACTION = "action"           # Action item
    OBSERVATION = "observation" # Recorded observation


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MemoryNode:
    """
    Represents a memory/concept node in the graph.
    
    Attributes:
        node_id: Unique identifier for the node
        content: The actual memory content/text
        node_type: Type of memory (concept, fact, etc.)
        timestamp: When the memory was created
        metadata: Additional context (source, confidence, etc.)
        embedding: Optional vector embedding for semantic similarity
        importance: Importance score (0.0 - 1.0)
        access_count: Number of times accessed
        last_accessed: Last access timestamp
    """
    node_id: str
    content: str
    node_type: NodeType = NodeType.CONCEPT
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "content": self.content,
            "node_type": self.node_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "embedding": self.embedding,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        """Create node from dictionary."""
        return cls(
            node_id=data["node_id"],
            content=data["content"],
            node_type=NodeType(data.get("node_type", "concept")),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
        )
    
    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class RelationshipEdge:
    """
    Represents a relationship between two memory nodes.
    
    Attributes:
        edge_id: Unique identifier for the edge
        source_id: ID of the source node
        target_id: ID of the target node
        relationship_type: Type of relationship
        weight: Strength of the relationship (0.0 - 1.0)
        timestamp: When the relationship was created
        metadata: Additional context (extractor confidence, evidence, etc.)
        bidirectional: Whether the relationship applies both ways
    """
    edge_id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization."""
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "weight": self.weight,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "bidirectional": self.bidirectional
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipEdge":
        """Create edge from dictionary."""
        return cls(
            edge_id=data["edge_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship_type=RelationshipType(data["relationship_type"]),
            weight=data.get("weight", 1.0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False)
        )
    
    def reverse(self) -> "RelationshipEdge":
        """Create a reversed version of this edge."""
        # Determine inverse relationship type
        inverse_map = {
            RelationshipType.CAUSAL: RelationshipType.DEPENDS_ON,
            RelationshipType.DEPENDS_ON: RelationshipType.CAUSAL,
            RelationshipType.PART_OF: RelationshipType.GENERALIZES,
            RelationshipType.GENERALIZES: RelationshipType.PART_OF,
            RelationshipType.SPECIFIES: RelationshipType.GENERALIZES,
            RelationshipType.CONTRADICTORY: RelationshipType.CONTRADICTORY,
            RelationshipType.EQUIVALENT: RelationshipType.EQUIVALENT,
        }
        inverse_type = inverse_map.get(self.relationship_type, self.relationship_type)
        
        return RelationshipEdge(
            edge_id=f"{self.edge_id}_rev",
            source_id=self.target_id,
            target_id=self.source_id,
            relationship_type=inverse_type,
            weight=self.weight,
            timestamp=self.timestamp,
            metadata=self.metadata.copy(),
            bidirectional=self.bidirectional
        )


@dataclass
class TraversalResult:
    """Result of a graph traversal operation."""
    nodes: List[MemoryNode]
    edges: List[RelationshipEdge]
    path: List[str]  # Node IDs in traversal order
    depth_reached: int
    total_weight: float
    context_summary: str = ""  # Generated summary of found context


@dataclass
class PathResult:
    """Result of a path finding operation."""
    path: List[str]  # Node IDs
    edges: List[RelationshipEdge]
    total_weight: float
    path_length: int
    relationship_chain: List[RelationshipType]  # Types of relationships along path


@dataclass
class CommunityResult:
    """Result of community detection."""
    communities: List[List[str]]  # Lists of node IDs
    modularity: float
    community_count: int
    largest_community_size: int


# =============================================================================
# Relationship Extractor
# =============================================================================

class RelationshipExtractor:
    """
    Extracts relationships from content using linguistic patterns and embeddings.
    
    This class analyzes text content to identify logical connections between
    ideas, automatically suggesting relationship types based on:
    - Cue phrases ("because", "therefore", "then", etc.)
    - Semantic similarity (via embeddings)
    - Structural patterns (lists, sequences)
    
    Example:
        >>> extractor = RelationshipExtractor()
        >>> rels = extractor.extract_from_text(
        ...     "We should use microservices because they enable scalability",
        ...     source_id="node_1"
        ... )
    """
    
    # Cue phrases mapped to relationship types
    CAUSAL_PATTERNS = {
        "because": RelationshipType.CAUSAL,
        "therefore": RelationshipType.CAUSAL,
        "thus": RelationshipType.CAUSAL,
        "hence": RelationshipType.CAUSAL,
        "leads to": RelationshipType.CAUSAL,
        "results in": RelationshipType.CAUSAL,
        "causes": RelationshipType.CAUSAL,
        "due to": RelationshipType.CAUSAL,
        "since": RelationshipType.CAUSAL,
        "as a result": RelationshipType.CAUSAL,
        "consequently": RelationshipType.CAUSAL,
    }
    
    TEMPORAL_PATTERNS = {
        "then": RelationshipType.TEMPORAL,
        "after": RelationshipType.TEMPORAL,
        "before": RelationshipType.TEMPORAL,
        "during": RelationshipType.TEMPORAL,
        "while": RelationshipType.TEMPORAL,
        "meanwhile": RelationshipType.TEMPORAL,
        "next": RelationshipType.SEQUENTIAL,
        "first": RelationshipType.SEQUENTIAL,
        "finally": RelationshipType.SEQUENTIAL,
        "subsequently": RelationshipType.TEMPORAL,
        "previously": RelationshipType.TEMPORAL,
        "lately": RelationshipType.TEMPORAL,
    }
    
    REFERENTIAL_PATTERNS = {
        "refers to": RelationshipType.REFERENTIAL,
        "about": RelationshipType.REFERENTIAL,
        "regarding": RelationshipType.REFERENTIAL,
        "concerning": RelationshipType.REFERENTIAL,
        "with respect to": RelationshipType.REFERENTIAL,
        "in relation to": RelationshipType.REFERENTIAL,
        "as mentioned": RelationshipType.REFERENTIAL,
        "see": RelationshipType.REFERENTIAL,
    }
    
    CONTRADICTORY_PATTERNS = {
        "however": RelationshipType.CONTRADICTORY,
        "but": RelationshipType.CONTRADICTORY,
        "although": RelationshipType.CONTRADICTORY,
        "whereas": RelationshipType.CONTRADICTORY,
        "unlike": RelationshipType.CONTRADICTORY,
        "in contrast": RelationshipType.CONTRADICTORY,
        "nevertheless": RelationshipType.CONTRADICTORY,
        "conversely": RelationshipType.CONTRADICTORY,
    }
    
    SUPPORTING_PATTERNS = {
        "for example": RelationshipType.SUPPORTING,
        "for instance": RelationshipType.SUPPORTING,
        "such as": RelationshipType.SUPPORTING,
        "evidence": RelationshipType.SUPPORTING,
        "demonstrates": RelationshipType.SUPPORTING,
        "shows": RelationshipType.SUPPORTING,
    }
    
    DEPENDENCY_PATTERNS = {
        "requires": RelationshipType.DEPENDS_ON,
        "needs": RelationshipType.DEPENDS_ON,
        "depends on": RelationshipType.DEPENDS_ON,
        "relies on": RelationshipType.DEPENDS_ON,
        "contingent on": RelationshipType.DEPENDS_ON,
    }
    
    ALL_PATTERNS = {
        **CAUSAL_PATTERNS,
        **TEMPORAL_PATTERNS,
        **REFERENTIAL_PATTERNS,
        **CONTRADICTORY_PATTERNS,
        **SUPPORTING_PATTERNS,
        **DEPENDENCY_PATTERNS,
    }
    
    def __init__(self, embedding_model: Optional[Any] = None):
        """
        Initialize the relationship extractor.
        
        Args:
            embedding_model: Optional embedding model for semantic similarity
        """
        self.embedding_model = embedding_model
        self.extraction_stats = {
            "total_extractions": 0,
            "by_type": defaultdict(int),
            "confidence_scores": []
        }
    
    def extract_from_text(
        self,
        content: str,
        source_id: Optional[str] = None,
        target_candidates: Optional[List[Tuple[str, str]]] = None
    ) -> List[Tuple[str, RelationshipType, float, str]]:
        """
        Extract relationships from text content.
        
        Args:
            content: Text content to analyze
            source_id: ID of the source node (optional)
            target_candidates: List of (node_id, content) tuples to match against
            
        Returns:
            List of (target_id, relationship_type, confidence, evidence) tuples
        """
        relationships = []
        content_lower = content.lower()
        
        # Pattern-based extraction
        for pattern, rel_type in self.ALL_PATTERNS.items():
            if pattern in content_lower:
                confidence = self._calculate_pattern_confidence(pattern, content_lower)
                evidence = f"Pattern '{pattern}' found in content"
                
                # If we have target candidates, try to match
                if target_candidates:
                    for target_id, target_content in target_candidates:
                        if self._is_related(content, target_content, pattern):
                            relationships.append((target_id, rel_type, confidence, evidence))
                else:
                    # Store as potential relationship to be resolved later
                    relationships.append((None, rel_type, confidence, evidence))
        
        # Semantic similarity extraction (if embedding model available)
        if self.embedding_model and target_candidates:
            semantic_rels = self._extract_semantic_relationships(
                content, source_id, target_candidates
            )
            relationships.extend(semantic_rels)
        
        self.extraction_stats["total_extractions"] += len(relationships)
        for _, rel_type, _, _ in relationships:
            self.extraction_stats["by_type"][rel_type] += 1
        
        return relationships
    
    def extract_between_nodes(
        self,
        source_content: str,
        target_content: str,
        source_id: str,
        target_id: str
    ) -> Optional[RelationshipEdge]:
        """
        Extract relationship between two specific nodes.
        
        Args:
            source_content: Content of source node
            target_content: Content of target node
            source_id: ID of source node
            target_id: ID of target node
            
        Returns:
            RelationshipEdge if relationship found, None otherwise
        """
        combined = f"{source_content} {target_content}".lower()
        
        best_rel = None
        best_confidence = 0.0
        
        for pattern, rel_type in self.ALL_PATTERNS.items():
            if pattern in combined:
                confidence = self._calculate_pattern_confidence(pattern, combined)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_rel = rel_type
        
        # Check for semantic similarity
        if self.embedding_model:
            semantic_sim = self._calculate_semantic_similarity(
                source_content, target_content
            )
            if semantic_sim > 0.8 and semantic_sim > best_confidence:
                best_confidence = semantic_sim
                best_rel = RelationshipType.SEMANTIC
        
        if best_rel and best_confidence > 0.3:
            return RelationshipEdge(
                edge_id=str(uuid.uuid4()),
                source_id=source_id,
                target_id=target_id,
                relationship_type=best_rel,
                weight=best_confidence,
                metadata={"extraction_method": "automatic", "evidence": combined[:200]}
            )
        
        return None
    
    def _calculate_pattern_confidence(self, pattern: str, content: str) -> float:
        """Calculate confidence score for a pattern match."""
        # Base confidence
        confidence = 0.7
        
        # Increase for exact word boundaries
        if re.search(r'\b' + re.escape(pattern) + r'\b', content):
            confidence += 0.15
        
        # Increase for early occurrence (more likely to be key relationship)
        position = content.find(pattern)
        if position >= 0:
            position_factor = 1.0 - (position / len(content))
            confidence += 0.1 * position_factor
        
        return min(confidence, 1.0)
    
    def _is_related(self, source: str, target: str, pattern: str) -> bool:
        """Check if two contents are related based on pattern context."""
        # Simple heuristic: shared keywords around pattern
        source_words = set(source.lower().split())
        target_words = set(target.lower().split())
        shared = source_words & target_words
        return len(shared) >= 2  # At least 2 shared words
    
    def _extract_semantic_relationships(
        self,
        content: str,
        source_id: Optional[str],
        target_candidates: List[Tuple[str, str]]
    ) -> List[Tuple[str, RelationshipType, float, str]]:
        """Extract relationships based on semantic similarity."""
        relationships = []
        
        if not self.embedding_model:
            return relationships
        
        try:
            content_embedding = self.embedding_model.encode(content)
            
            for target_id, target_content in target_candidates:
                if target_id == source_id:
                    continue
                    
                target_embedding = self.embedding_model.encode(target_content)
                similarity = self._cosine_similarity(content_embedding, target_embedding)
                
                if similarity > 0.75:  # High similarity threshold
                    relationships.append(
                        (target_id, RelationshipType.SEMANTIC, similarity, 
                         f"Semantic similarity: {similarity:.3f}")
                    )
        except Exception as e:
            logger.warning(f"Semantic extraction failed: {e}")
        
        return relationships
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self.embedding_model:
            return 0.0
        
        try:
            emb1 = self.embedding_model.encode(text1)
            emb2 = self.embedding_model.encode(text2)
            return self._cosine_similarity(emb1, emb2)
        except Exception:
            return 0.0
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if not NUMPY_AVAILABLE:
            return 0.0
        
        try:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except Exception:
            return 0.0
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """Get summary of extraction statistics."""
        return {
            "total_extractions": self.extraction_stats["total_extractions"],
            "by_type": dict(self.extraction_stats["by_type"]),
            "avg_confidence": (
                sum(self.extraction_stats["confidence_scores"]) / 
                len(self.extraction_stats["confidence_scores"])
                if self.extraction_stats["confidence_scores"] else 0.0
            )
        }


# =============================================================================
# Graph Index Main Class
# =============================================================================

class GraphIndex:
    """
    Graph-based index preserving logical connections between memories.
    
    This is the main class for managing a graph structure where:
    - Nodes represent memories, concepts, or ideas
    - Edges represent logical relationships (causal, temporal, semantic, etc.)
    
    The index provides:
    - Persistent storage via SQLite
    - Thread-safe concurrent access
    - Graph traversal and path finding
    - Community detection
    - NetworkX integration for advanced operations
    
    Attributes:
        db_path: Path to SQLite database file
        extractor: RelationshipExtractor instance
        _lock: Thread lock for concurrent access
        _local: Thread-local storage for connections
    
    Example:
        >>> index = GraphIndex("memory_graph.db")
        >>> n1 = index.add_node("User needs fast response time")
        >>> n2 = index.add_node("We should implement caching")
        >>> index.add_edge(n1, n2, RelationshipType.CAUSAL)
        >>> context = index.traverse_relationships(n1, depth=2)
    """
    
    def __init__(
        self,
        db_path: str = "./knowledge_graph_index.db",
        embedding_model: Optional[Any] = None,
        enable_networkx: bool = True
    ):
        """
        Initialize the graph index.
        
        Args:
            db_path: Path to SQLite database
            embedding_model: Optional embedding model for semantic extraction
            enable_networkx: Whether to enable NetworkX integration
        """
        self.db_path = db_path
        self.extractor = RelationshipExtractor(embedding_model)
        self.enable_networkx = enable_networkx and NETWORKX_AVAILABLE
        
        # Thread safety
        self._lock = threading.RLock()
        self._local = threading.local()
        
        # In-memory cache for hot nodes
        self._node_cache: Dict[str, MemoryNode] = {}
        self._edge_cache: Dict[str, RelationshipEdge] = {}
        self._cache_size = 1000
        
        # Initialize database
        self._init_database()
        
        # NetworkX graph (lazy loaded)
        self._nx_graph = None
        
        logger.info(f"GraphIndex initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    node_type TEXT DEFAULT 'concept',
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    embedding TEXT,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT
                )
            """)
            
            # Edges table with adjacency list structure
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    bidirectional INTEGER DEFAULT 0,
                    FOREIGN KEY (source_id) REFERENCES nodes(node_id),
                    FOREIGN KEY (target_id) REFERENCES nodes(node_id)
                )
            """)
            
            # Indexes for efficient traversal
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_edges_source 
                ON edges(source_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_edges_target 
                ON edges(target_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_edges_type 
                ON edges(relationship_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_edges_source_target 
                ON edges(source_id, target_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_nodes_timestamp 
                ON nodes(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_nodes_type 
                ON nodes(node_type)
            """)
            
            # Graph metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT NOT NULL
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        
        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            raise e
    
    def _generate_node_id(self, content: str) -> str:
        """Generate unique node ID from content hash."""
        hash_input = f"{content}_{datetime.now().isoformat()}_{uuid.uuid4()}"
        return f"node_{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"
    
    def _cache_node(self, node: MemoryNode):
        """Add node to cache with LRU eviction."""
        if len(self._node_cache) >= self._cache_size:
            # Simple LRU: remove oldest
            oldest = next(iter(self._node_cache))
            del self._node_cache[oldest]
        self._node_cache[node.node_id] = node
    
    def _get_cached_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get node from cache."""
        return self._node_cache.get(node_id)
    
    # ========================================================================
    # Node Operations
    # ========================================================================
    
    def add_node(
        self,
        content: str,
        node_type: NodeType = NodeType.CONCEPT,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        relationships: Optional[List[Tuple[str, RelationshipType, Optional[Dict]]]] = None
    ) -> str:
        """
        Add a new memory node to the graph.
        
        Args:
            content: The memory content/text
            node_type: Type of memory node
            node_id: Optional custom ID (generated if not provided)
            metadata: Additional metadata
            embedding: Optional vector embedding
            importance: Importance score (0.0 - 1.0)
            relationships: List of (target_id, rel_type, metadata) to create
            
        Returns:
            The node ID
        """
        with self._lock:
            if node_id is None:
                node_id = self._generate_node_id(content)
            
            node = MemoryNode(
                node_id=node_id,
                content=content,
                node_type=node_type,
                timestamp=datetime.now(),
                metadata=metadata or {},
                embedding=embedding,
                importance=importance
            )
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO nodes 
                    (node_id, content, node_type, timestamp, metadata, embedding, 
                     importance, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    node.node_id,
                    node.content,
                    node.node_type.value,
                    node.timestamp.isoformat(),
                    json.dumps(node.metadata),
                    json.dumps(node.embedding) if node.embedding else None,
                    node.importance,
                    node.access_count,
                    node.last_accessed.isoformat() if node.last_accessed else None
                ))
                conn.commit()
            
            self._cache_node(node)
            
            # Create initial relationships if provided
            if relationships:
                for target_id, rel_type, rel_metadata in relationships:
                    self.add_edge(node_id, target_id, rel_type, metadata=rel_metadata)
            
            # Auto-extract relationships from content
            if self.extractor:
                self._auto_extract_relationships(node)
            
            logger.debug(f"Added node: {node_id}")
            return node_id
    
    def get_node(self, node_id: str, update_access: bool = True) -> Optional[MemoryNode]:
        """
        Retrieve a node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            update_access: Whether to update access statistics
            
        Returns:
            MemoryNode if found, None otherwise
        """
        # Check cache first
        cached = self._get_cached_node(node_id)
        if cached and not update_access:
            return cached
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
                )
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                node = self._row_to_node(row)
                
                if update_access:
                    node.touch()
                    cursor.execute("""
                        UPDATE nodes 
                        SET access_count = ?, last_accessed = ?
                        WHERE node_id = ?
                    """, (node.access_count, node.last_accessed.isoformat(), node_id))
                    conn.commit()
                
                self._cache_node(node)
                return node
    
    def update_node(
        self,
        node_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None
    ) -> bool:
        """
        Update an existing node.
        
        Args:
            node_id: ID of node to update
            content: New content (optional)
            metadata: Metadata updates (merged with existing)
            importance: New importance score
            
        Returns:
            True if updated successfully
        """
        with self._lock:
            node = self.get_node(node_id, update_access=False)
            if not node:
                return False
            
            if content is not None:
                node.content = content
            if metadata is not None:
                node.metadata.update(metadata)
            if importance is not None:
                node.importance = importance
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE nodes 
                    SET content = ?, metadata = ?, importance = ?
                    WHERE node_id = ?
                """, (
                    node.content,
                    json.dumps(node.metadata),
                    node.importance,
                    node_id
                ))
                conn.commit()
            
            self._cache_node(node)
            return True
    
    def delete_node(self, node_id: str, cascade: bool = True) -> bool:
        """
        Delete a node and optionally its connected edges.
        
        Args:
            node_id: ID of node to delete
            cascade: Whether to delete connected edges
            
        Returns:
            True if deleted successfully
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if cascade:
                    cursor.execute(
                        "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                        (node_id, node_id)
                    )
                
                cursor.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
            
            # Clear from cache
            if node_id in self._node_cache:
                del self._node_cache[node_id]
            
            return deleted
    
    def find_nodes_by_content(
        self,
        query: str,
        limit: int = 10
    ) -> List[MemoryNode]:
        """
        Find nodes by content substring match.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching nodes
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM nodes 
                    WHERE content LIKE ?
                    ORDER BY importance DESC, access_count DESC
                    LIMIT ?
                """, (f"%{query}%", limit))
                
                return [self._row_to_node(row) for row in cursor.fetchall()]
    
    def find_nodes_by_type(
        self,
        node_type: NodeType,
        limit: int = 100
    ) -> List[MemoryNode]:
        """Find nodes by type."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM nodes 
                    WHERE node_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (node_type.value, limit))
                
                return [self._row_to_node(row) for row in cursor.fetchall()]
    
    def _row_to_node(self, row: sqlite3.Row) -> MemoryNode:
        """Convert database row to MemoryNode."""
        return MemoryNode(
            node_id=row["node_id"],
            content=row["content"],
            node_type=NodeType(row["node_type"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None
        )
    
    # ========================================================================
    # Edge Operations
    # ========================================================================
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        weight: float = 1.0,
        edge_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        bidirectional: bool = False
    ) -> str:
        """
        Create a relationship edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship
            weight: Relationship strength (0.0 - 1.0)
            edge_id: Optional custom ID
            metadata: Additional metadata
            bidirectional: Whether to create reverse edge
            
        Returns:
            The edge ID
        """
        with self._lock:
            if edge_id is None:
                edge_id = str(uuid.uuid4())
            
            edge = RelationshipEdge(
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship_type,
                weight=weight,
                metadata=metadata or {},
                bidirectional=bidirectional
            )
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO edges 
                    (edge_id, source_id, target_id, relationship_type, weight, 
                     timestamp, metadata, bidirectional)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    edge.relationship_type.value,
                    edge.weight,
                    edge.timestamp.isoformat(),
                    json.dumps(edge.metadata),
                    1 if edge.bidirectional else 0
                ))
                
                # Create reverse edge if bidirectional
                if bidirectional:
                    reverse = edge.reverse()
                    cursor.execute("""
                        INSERT OR REPLACE INTO edges 
                        (edge_id, source_id, target_id, relationship_type, weight, 
                         timestamp, metadata, bidirectional)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        reverse.edge_id,
                        reverse.source_id,
                        reverse.target_id,
                        reverse.relationship_type.value,
                        reverse.weight,
                        reverse.timestamp.isoformat(),
                        json.dumps(reverse.metadata),
                        1
                    ))
                
                conn.commit()
            
            self._edge_cache[edge_id] = edge
            
            # Invalidate NetworkX cache
            self._nx_graph = None
            
            logger.debug(f"Added edge: {edge_id} ({relationship_type.value})")
            return edge_id
    
    def get_edge(self, edge_id: str) -> Optional[RelationshipEdge]:
        """Get edge by ID."""
        if edge_id in self._edge_cache:
            return self._edge_cache[edge_id]
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM edges WHERE edge_id = ?", (edge_id,)
                )
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                edge = self._row_to_edge(row)
                self._edge_cache[edge_id] = edge
                return edge
    
    def get_edges_from_node(
        self,
        node_id: str,
        relationship_type: Optional[RelationshipType] = None
    ) -> List[RelationshipEdge]:
        """
        Get all edges originating from a node.
        
        Args:
            node_id: Source node ID
            relationship_type: Filter by type (optional)
            
        Returns:
            List of outgoing edges
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if relationship_type:
                    cursor.execute("""
                        SELECT * FROM edges 
                        WHERE source_id = ? AND relationship_type = ?
                        ORDER BY weight DESC
                    """, (node_id, relationship_type.value))
                else:
                    cursor.execute("""
                        SELECT * FROM edges 
                        WHERE source_id = ?
                        ORDER BY weight DESC
                    """, (node_id,))
                
                return [self._row_to_edge(row) for row in cursor.fetchall()]
    
    def get_edges_to_node(
        self,
        node_id: str,
        relationship_type: Optional[RelationshipType] = None
    ) -> List[RelationshipEdge]:
        """Get all edges targeting a node."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if relationship_type:
                    cursor.execute("""
                        SELECT * FROM edges 
                        WHERE target_id = ? AND relationship_type = ?
                        ORDER BY weight DESC
                    """, (node_id, relationship_type.value))
                else:
                    cursor.execute("""
                        SELECT * FROM edges 
                        WHERE target_id = ?
                        ORDER BY weight DESC
                    """, (node_id,))
                
                return [self._row_to_edge(row) for row in cursor.fetchall()]
    
    def get_connected_nodes(
        self,
        node_id: str,
        relationship_type: Optional[RelationshipType] = None
    ) -> List[Tuple[MemoryNode, RelationshipEdge]]:
        """
        Get all nodes connected to a given node.
        
        Returns:
            List of (node, edge) tuples
        """
        edges = self.get_edges_from_node(node_id, relationship_type)
        result = []
        
        for edge in edges:
            target = self.get_node(edge.target_id)
            if target:
                result.append((target, edge))
        
        return result
    
    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by ID."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM edges WHERE edge_id = ?", (edge_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
            
            if edge_id in self._edge_cache:
                del self._edge_cache[edge_id]
            
            self._nx_graph = None
            return deleted
    
    def _row_to_edge(self, row: sqlite3.Row) -> RelationshipEdge:
        """Convert database row to RelationshipEdge."""
        return RelationshipEdge(
            edge_id=row["edge_id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relationship_type=RelationshipType(row["relationship_type"]),
            weight=row["weight"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            bidirectional=bool(row["bidirectional"])
        )
    
    # ========================================================================
    # Auto-extraction
    # ========================================================================
    
    def _auto_extract_relationships(self, node: MemoryNode):
        """Automatically extract relationships from new node content."""
        # Get recent nodes for matching
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT node_id, content FROM nodes 
                WHERE node_id != ?
                ORDER BY timestamp DESC
                LIMIT 50
            """, (node.node_id,))
            
            candidates = [(row["node_id"], row["content"]) for row in cursor.fetchall()]
        
        # Extract relationships
        extracted = self.extractor.extract_from_text(
            node.content,
            source_id=node.node_id,
            target_candidates=candidates
        )
        
        # Add extracted edges
        for target_id, rel_type, confidence, evidence in extracted:
            if target_id and confidence > 0.5:
                self.add_edge(
                    source_id=node.node_id,
                    target_id=target_id,
                    relationship_type=rel_type,
                    weight=confidence,
                    metadata={"auto_extracted": True, "evidence": evidence}
                )
    
    # ========================================================================
    # Graph Traversal
    # ========================================================================
    
    def traverse_relationships(
        self,
        start_node_id: str,
        depth: int = 2,
        mode: TraversalMode = TraversalMode.BFS,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_weight: float = 0.0
    ) -> TraversalResult:
        """
        Navigate the graph to find connected context across time.
        
        Args:
            start_node_id: Starting node ID
            depth: Maximum traversal depth
            mode: Traversal mode (BFS, DFS, etc.)
            relationship_types: Filter by relationship types
            min_weight: Minimum edge weight to follow
            
        Returns:
            TraversalResult with found nodes, edges, and path
        """
        with self._lock:
            if mode == TraversalMode.BFS:
                return self._traverse_bfs(
                    start_node_id, depth, relationship_types, min_weight
                )
            elif mode == TraversalMode.DFS:
                return self._traverse_dfs(
                    start_node_id, depth, relationship_types, min_weight
                )
            elif mode == TraversalMode.WEIGHTED:
                return self._traverse_weighted(
                    start_node_id, depth, relationship_types, min_weight
                )
            else:
                return self._traverse_bfs(
                    start_node_id, depth, relationship_types, min_weight
                )
    
    def _traverse_bfs(
        self,
        start_node_id: str,
        max_depth: int,
        relationship_types: Optional[List[RelationshipType]],
        min_weight: float
    ) -> TraversalResult:
        """Breadth-first traversal."""
        from collections import deque
        
        visited = {start_node_id}
        queue = deque([(start_node_id, 0)])
        path = [start_node_id]
        all_nodes = {start_node_id: self.get_node(start_node_id)}
        all_edges = []
        total_weight = 0.0
        max_reached = 0
        
        while queue:
            current_id, current_depth = queue.popleft()
            max_reached = max(max_reached, current_depth)
            
            if current_depth >= max_depth:
                continue
            
            edges = self.get_edges_from_node(current_id)
            
            for edge in edges:
                if edge.weight < min_weight:
                    continue
                if relationship_types and edge.relationship_type not in relationship_types:
                    continue
                
                target_id = edge.target_id
                if target_id not in visited:
                    visited.add(target_id)
                    queue.append((target_id, current_depth + 1))
                    path.append(target_id)
                    all_nodes[target_id] = self.get_node(target_id)
                    all_edges.append(edge)
                    total_weight += edge.weight
        
        nodes_list = [n for n in all_nodes.values() if n is not None]
        context_summary = self._generate_context_summary(nodes_list, all_edges)
        
        return TraversalResult(
            nodes=nodes_list,
            edges=all_edges,
            path=path,
            depth_reached=max_reached,
            total_weight=total_weight,
            context_summary=context_summary
        )
    
    def _traverse_dfs(
        self,
        start_node_id: str,
        max_depth: int,
        relationship_types: Optional[List[RelationshipType]],
        min_weight: float
    ) -> TraversalResult:
        """Depth-first traversal."""
        visited = {start_node_id}
        path = [start_node_id]
        all_nodes = {start_node_id: self.get_node(start_node_id)}
        all_edges = []
        total_weight = 0.0
        max_reached = 0
        
        def dfs(node_id: str, depth: int):
            nonlocal max_reached, total_weight
            max_reached = max(max_reached, depth)
            
            if depth >= max_depth:
                return
            
            edges = self.get_edges_from_node(node_id)
            
            for edge in edges:
                if edge.weight < min_weight:
                    continue
                if relationship_types and edge.relationship_type not in relationship_types:
                    continue
                
                target_id = edge.target_id
                if target_id not in visited:
                    visited.add(target_id)
                    path.append(target_id)
                    all_nodes[target_id] = self.get_node(target_id)
                    all_edges.append(edge)
                    total_weight += edge.weight
                    
                    dfs(target_id, depth + 1)
        
        dfs(start_node_id, 0)
        
        nodes_list = [n for n in all_nodes.values() if n is not None]
        context_summary = self._generate_context_summary(nodes_list, all_edges)
        
        return TraversalResult(
            nodes=nodes_list,
            edges=all_edges,
            path=path,
            depth_reached=max_reached,
            total_weight=total_weight,
            context_summary=context_summary
        )
    
    def _traverse_weighted(
        self,
        start_node_id: str,
        max_depth: int,
        relationship_types: Optional[List[RelationshipType]],
        min_weight: float
    ) -> TraversalResult:
        """Weighted traversal prioritizing stronger relationships."""
        import heapq
        
        visited = {start_node_id}
        # Priority queue: (-weight, node_id, depth)
        pq = [(-1.0, start_node_id, 0)]
        path = [start_node_id]
        all_nodes = {start_node_id: self.get_node(start_node_id)}
        all_edges = []
        total_weight = 0.0
        max_reached = 0
        
        while pq:
            neg_weight, current_id, current_depth = heapq.heappop(pq)
            max_reached = max(max_reached, current_depth)
            
            if current_depth >= max_depth:
                continue
            
            edges = self.get_edges_from_node(current_id)
            # Sort by weight descending
            edges.sort(key=lambda e: e.weight, reverse=True)
            
            for edge in edges:
                if edge.weight < min_weight:
                    continue
                if relationship_types and edge.relationship_type not in relationship_types:
                    continue
                
                target_id = edge.target_id
                if target_id not in visited:
                    visited.add(target_id)
                    heapq.heappush(pq, (-edge.weight, target_id, current_depth + 1))
                    path.append(target_id)
                    all_nodes[target_id] = self.get_node(target_id)
                    all_edges.append(edge)
                    total_weight += edge.weight
        
        nodes_list = [n for n in all_nodes.values() if n is not None]
        context_summary = self._generate_context_summary(nodes_list, all_edges)
        
        return TraversalResult(
            nodes=nodes_list,
            edges=all_edges,
            path=path,
            depth_reached=max_reached,
            total_weight=total_weight,
            context_summary=context_summary
        )
    
    def _generate_context_summary(
        self,
        nodes: List[MemoryNode],
        edges: List[RelationshipEdge]
    ) -> str:
        """Generate a natural language summary of traversal results."""
        if not nodes:
            return "No context found."
        
        parts = [f"Found {len(nodes)} connected memories across {len(edges)} relationships."]
        
        # Group by relationship type
        rel_counts = defaultdict(int)
        for edge in edges:
            rel_counts[edge.relationship_type.value] += 1
        
        if rel_counts:
            rel_summary = ", ".join(
                f"{count} {rel_type}" for rel_type, count in sorted(rel_counts.items())
            )
            parts.append(f"Relationship types: {rel_summary}.")
        
        # Mention most important node
        if nodes:
            most_important = max(nodes, key=lambda n: n.importance)
            parts.append(
                f"Key concept: '{most_important.content[:50]}...' "
                f"(importance: {most_important.importance:.2f})"
            )
        
        return " ".join(parts)
    
    # ========================================================================
    # Path Finding
    # ========================================================================
    
    def find_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 10,
        algorithm: str = "shortest"
    ) -> Optional[PathResult]:
        """
        Find paths between distant memories (message 5 → message 500).
        
        Args:
            start_node_id: Starting node ID
            end_node_id: Target node ID
            max_depth: Maximum search depth
            algorithm: Path finding algorithm ("shortest", "strongest", "all")
            
        Returns:
            PathResult if path found, None otherwise
        """
        if algorithm == "shortest":
            return self._find_shortest_path(start_node_id, end_node_id, max_depth)
        elif algorithm == "strongest":
            return self._find_strongest_path(start_node_id, end_node_id, max_depth)
        else:
            return self._find_shortest_path(start_node_id, end_node_id, max_depth)
    
    def _find_shortest_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int
    ) -> Optional[PathResult]:
        """Find shortest path using BFS."""
        from collections import deque
        
        if start_node_id == end_node_id:
            node = self.get_node(start_node_id)
            if node:
                return PathResult(
                    path=[start_node_id],
                    edges=[],
                    total_weight=1.0,
                    path_length=0,
                    relationship_chain=[]
                )
            return None
        
        queue = deque([(start_node_id, [start_node_id], [], 0.0)])
        visited = {start_node_id}
        
        while queue:
            current_id, path, edges, total_weight = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            for edge in self.get_edges_from_node(current_id):
                target_id = edge.target_id
                
                if target_id == end_node_id:
                    final_path = path + [target_id]
                    final_edges = edges + [edge]
                    return PathResult(
                        path=final_path,
                        edges=final_edges,
                        total_weight=total_weight + edge.weight,
                        path_length=len(final_path) - 1,
                        relationship_chain=[e.relationship_type for e in final_edges]
                    )
                
                if target_id not in visited:
                    visited.add(target_id)
                    queue.append((
                        target_id,
                        path + [target_id],
                        edges + [edge],
                        total_weight + edge.weight
                    ))
        
        return None
    
    def _find_strongest_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int
    ) -> Optional[PathResult]:
        """Find path with maximum cumulative weight."""
        import heapq
        
        if start_node_id == end_node_id:
            node = self.get_node(start_node_id)
            if node:
                return PathResult(
                    path=[start_node_id],
                    edges=[],
                    total_weight=1.0,
                    path_length=0,
                    relationship_chain=[]
                )
            return None
        
        # Priority queue: (-cumulative_weight, node_id, path, edges)
        pq = [(-1.0, start_node_id, [start_node_id], [])]
        visited = set()
        
        while pq:
            neg_weight, current_id, path, edges = heapq.heappop(pq)
            cumulative_weight = -neg_weight
            
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if len(path) > max_depth:
                continue
            
            for edge in self.get_edges_from_node(current_id):
                target_id = edge.target_id
                new_weight = cumulative_weight * edge.weight
                new_path = path + [target_id]
                new_edges = edges + [edge]
                
                if target_id == end_node_id:
                    return PathResult(
                        path=new_path,
                        edges=new_edges,
                        total_weight=new_weight,
                        path_length=len(new_path) - 1,
                        relationship_chain=[e.relationship_type for e in new_edges]
                    )
                
                if target_id not in visited:
                    heapq.heappush(pq, (-new_weight, target_id, new_path, new_edges))
        
        return None
    
    # ========================================================================
    # Community Detection
    # ========================================================================
    
    def detect_communities(
        self,
        algorithm: str = "label_propagation",
        min_community_size: int = 3
    ) -> CommunityResult:
        """
        Detect communities for clustering related memories.
        
        Args:
            algorithm: Community detection algorithm
            min_community_size: Minimum nodes per community
            
        Returns:
            CommunityResult with detected communities
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available, using basic clustering")
            return self._basic_clustering(min_community_size)
        
        nx_graph = self._build_networkx_graph()
        
        if algorithm == "label_propagation":
            communities = self._label_propagation_clustering(nx_graph)
        elif algorithm == "louvain":
            communities = self._louvain_clustering(nx_graph)
        else:
            communities = self._connected_components_clustering(nx_graph)
        
        # Filter by size
        communities = [c for c in communities if len(c) >= min_community_size]
        
        # Calculate modularity
        modularity = self._calculate_modularity(nx_graph, communities)
        
        largest_size = max((len(c) for c in communities), default=0)
        
        return CommunityResult(
            communities=communities,
            modularity=modularity,
            community_count=len(communities),
            largest_community_size=largest_size
        )
    
    def _build_networkx_graph(self) -> "nx.DiGraph":
        """Build NetworkX graph from database."""
        if self._nx_graph is not None:
            return self._nx_graph
        
        G = nx.DiGraph()
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Add nodes
                cursor.execute("SELECT * FROM nodes")
                for row in cursor.fetchall():
                    G.add_node(
                        row["node_id"],
                        content=row["content"],
                        node_type=row["node_type"],
                        importance=row["importance"]
                    )
                
                # Add edges
                cursor.execute("SELECT * FROM edges")
                for row in cursor.fetchall():
                    G.add_edge(
                        row["source_id"],
                        row["target_id"],
                        relationship_type=row["relationship_type"],
                        weight=row["weight"]
                    )
        
        self._nx_graph = G
        return G
    
    def _label_propagation_clustering(
        self,
        G: "nx.DiGraph"
    ) -> List[List[str]]:
        """Label propagation community detection."""
        try:
            # Convert to undirected for community detection
            undirected = G.to_undirected()
            communities = nx.community.label_propagation_communities(undirected)
            return [list(c) for c in communities]
        except Exception as e:
            logger.warning(f"Label propagation failed: {e}, using connected components")
            return self._connected_components_clustering(G)
    
    def _louvain_clustering(self, G: "nx.DiGraph") -> List[List[str]]:
        """Louvain community detection."""
        try:
            import community as community_louvain
            undirected = G.to_undirected()
            partition = community_louvain.best_partition(undirected)
            
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
            
            return list(communities.values())
        except ImportError:
            logger.warning("python-louvain not available, using label propagation")
            return self._label_propagation_clustering(G)
        except Exception as e:
            logger.warning(f"Louvain clustering failed: {e}")
            return self._connected_components_clustering(G)
    
    def _connected_components_clustering(
        self,
        G: "nx.DiGraph"
    ) -> List[List[str]]:
        """Simple connected components clustering."""
        undirected = G.to_undirected()
        return [list(c) for c in nx.connected_components(undirected)]
    
    def _basic_clustering(self, min_size: int) -> CommunityResult:
        """Basic clustering without NetworkX."""
        # Simple connected components via BFS
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT node_id FROM nodes")
                all_nodes = {row["node_id"] for row in cursor.fetchall()}
        
        visited = set()
        communities = []
        
        for start_node in all_nodes:
            if start_node in visited:
                continue
            
            component = []
            stack = [start_node]
            
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                
                # Get neighbors
                for edge in self.get_edges_from_node(node):
                    if edge.target_id not in visited:
                        stack.append(edge.target_id)
                for edge in self.get_edges_to_node(node):
                    if edge.source_id not in visited:
                        stack.append(edge.source_id)
            
            if len(component) >= min_size:
                communities.append(component)
        
        return CommunityResult(
            communities=communities,
            modularity=0.0,  # Cannot calculate without NetworkX
            community_count=len(communities),
            largest_community_size=max((len(c) for c in communities), default=0)
        )
    
    def _calculate_modularity(
        self,
        G: "nx.DiGraph",
        communities: List[List[str]]
    ) -> float:
        """Calculate modularity score."""
        try:
            undirected = G.to_undirected()
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
            return nx.community.modularity(undirected, communities)
        except Exception:
            return 0.0
    
    # ========================================================================
    # Export/Import
    # ========================================================================
    
    def to_networkx(self) -> Optional["nx.DiGraph"]:
        """
        Export the graph to NetworkX format.
        
        Returns:
            NetworkX DiGraph or None if NetworkX not available
        """
        if not NETWORKX_AVAILABLE:
            logger.error("NetworkX not available")
            return None
        
        return self._build_networkx_graph()
    
    def export_to_json(self, filepath: str):
        """
        Export the entire graph to JSON format.
        
        Args:
            filepath: Output file path
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM nodes")
                nodes = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT * FROM edges")
                edges = [dict(row) for row in cursor.fetchall()]
            
            data = {
                "metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "version": "1.0"
                },
                "nodes": nodes,
                "edges": edges
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported graph to {filepath}")
    
    def import_from_json(self, filepath: str, merge: bool = False):
        """
        Import graph from JSON format.
        
        Args:
            filepath: Input file path
            merge: Whether to merge with existing data or replace
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        with self._lock:
            if not merge:
                # Clear existing data
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM edges")
                    cursor.execute("DELETE FROM nodes")
                    conn.commit()
            
            # Import nodes
            for node_data in data.get("nodes", []):
                try:
                    self.add_node(
                        content=node_data["content"],
                        node_type=NodeType(node_data.get("node_type", "concept")),
                        node_id=node_data["node_id"],
                        metadata=json.loads(node_data.get("metadata", "{}")),
                        embedding=json.loads(node_data["embedding"]) if node_data.get("embedding") else None,
                        importance=node_data.get("importance", 0.5)
                    )
                except Exception as e:
                    logger.warning(f"Failed to import node {node_data.get('node_id')}: {e}")
            
            # Import edges
            for edge_data in data.get("edges", []):
                try:
                    self.add_edge(
                        source_id=edge_data["source_id"],
                        target_id=edge_data["target_id"],
                        relationship_type=RelationshipType(edge_data["relationship_type"]),
                        weight=edge_data.get("weight", 1.0),
                        edge_id=edge_data["edge_id"],
                        metadata=json.loads(edge_data.get("metadata", "{}")),
                        bidirectional=bool(edge_data.get("bidirectional", 0))
                    )
                except Exception as e:
                    logger.warning(f"Failed to import edge {edge_data.get('edge_id')}: {e}")
        
        logger.info(f"Imported graph from {filepath}")
    
    def export_to_graphml(self, filepath: str):
        """
        Export to GraphML format for use with other tools.
        
        Args:
            filepath: Output file path
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for GraphML export")
        
        G = self._build_networkx_graph()
        nx.write_graphml(G, filepath)
        logger.info(f"Exported graph to GraphML: {filepath}")
    
    # ========================================================================
    # Statistics and Maintenance
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM nodes")
                node_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM edges")
                edge_count = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT relationship_type, COUNT(*) 
                    FROM edges 
                    GROUP BY relationship_type
                """)
                rel_type_counts = {row[0]: row[1] for row in cursor.fetchall()}
                
                cursor.execute("""
                    SELECT node_type, COUNT(*) 
                    FROM nodes 
                    GROUP BY node_type
                """)
                node_type_counts = {row[0]: row[1] for row in cursor.fetchall()}
                
                cursor.execute("""
                    SELECT AVG(importance), MAX(importance), MIN(importance)
                    FROM nodes
                """)
                importance_stats = cursor.fetchone()
                
                return {
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "relationship_type_counts": rel_type_counts,
                    "node_type_counts": node_type_counts,
                    "avg_importance": importance_stats[0],
                    "max_importance": importance_stats[1],
                    "min_importance": importance_stats[2],
                    "avg_degree": edge_count / node_count if node_count > 0 else 0,
                    "density": edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0
                }
    
    def vacuum(self):
        """Optimize database storage."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
                logger.info("Database vacuum completed")
    
    def cleanup_orphaned_edges(self) -> int:
        """Remove edges pointing to non-existent nodes."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM edges 
                    WHERE source_id NOT IN (SELECT node_id FROM nodes)
                    OR target_id NOT IN (SELECT node_id FROM nodes)
                """)
                deleted = cursor.rowcount
                conn.commit()
                logger.info(f"Cleaned up {deleted} orphaned edges")
                return deleted
    
    def close(self):
        """Close database connections."""
        with self._lock:
            if hasattr(self._local, 'connection') and self._local.connection:
                self._local.connection.close()
                self._local.connection = None
                logger.info("Database connection closed")


# =============================================================================
# Convenience Functions
# =============================================================================

def traverse_relationships(
    index: GraphIndex,
    start_node_id: str,
    depth: int = 2,
    mode: TraversalMode = TraversalMode.BFS,
    relationship_types: Optional[List[RelationshipType]] = None
) -> TraversalResult:
    """
    Navigate the graph to find connected context across time.
    
    This is a convenience function that wraps GraphIndex.traverse_relationships().
    
    Args:
        index: GraphIndex instance
        start_node_id: Starting node ID
        depth: Maximum traversal depth
        mode: Traversal mode (BFS, DFS, weighted)
        relationship_types: Optional filter for relationship types
        
    Returns:
        TraversalResult with found nodes, edges, and paths
        
    Example:
        >>> index = GraphIndex("memories.db")
        >>> result = traverse_relationships(
        ...     index, 
        ...     "node_abc123", 
        ...     depth=3,
        ...     mode=TraversalMode.CAUSAL
        ... )
        >>> for node in result.nodes:
        ...     print(f"Found: {node.content}")
    """
    return index.traverse_relationships(
        start_node_id=start_node_id,
        depth=depth,
        mode=mode,
        relationship_types=relationship_types
    )


def find_memory_path(
    index: GraphIndex,
    start_node_id: str,
    end_node_id: str,
    max_depth: int = 10
) -> Optional[PathResult]:
    """
    Find the path connecting two distant memories.
    
    Args:
        index: GraphIndex instance
        start_node_id: Starting memory node ID
        end_node_id: Target memory node ID
        max_depth: Maximum search depth
        
    Returns:
        PathResult if path found, None otherwise
        
    Example:
        >>> path = find_memory_path(index, "msg_005", "msg_500")
        >>> if path:
        ...     print(f"Path length: {path.path_length}")
        ...     for edge in path.edges:
        ...         print(f"  {edge.relationship_type.value}")
    """
    return index.find_path(start_node_id, end_node_id, max_depth)


def extract_relationships_from_text(text: str) -> List[Tuple[RelationshipType, str, float]]:
    """
    Extract relationship indicators from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of (relationship_type, pattern, confidence) tuples
        
    Example:
        >>> extract_relationships_from_text(
        ...     "We need caching because performance is critical"
        ... )
        [(RelationshipType.CAUSAL, "because", 0.85)]
    """
    extractor = RelationshipExtractor()
    results = extractor.extract_from_text(text)
    return [(rel_type, evidence, conf) for _, rel_type, conf, evidence in results]


# =============================================================================
# Context Recovery Helper
# =============================================================================

class ContextRecovery:
    """
    Helper class for recovering lost context across conversation history.
    
    This addresses "context rot" by using the graph index to find
    related memories even when they're far apart in the conversation.
    """
    
    def __init__(self, index: GraphIndex):
        """
        Initialize context recovery.
        
        Args:
            index: GraphIndex instance
        """
        self.index = index
    
    def recover_context(
        self,
        current_node_id: str,
        lookback_depth: int = 5,
        min_importance: float = 0.3
    ) -> Dict[str, Any]:
        """
        Recover relevant context for a given node.
        
        Args:
            current_node_id: Current conversation point
            lookback_depth: How far to search in the graph
            min_importance: Minimum importance threshold
            
        Returns:
            Dictionary with recovered context
        """
        # Get immediate context
        traversal = self.index.traverse_relationships(
            start_node_id=current_node_id,
            depth=lookback_depth,
            mode=TraversalMode.WEIGHTED
        )
        
        # Filter by importance
        important_nodes = [
            n for n in traversal.nodes 
            if n.importance >= min_importance and n.node_id != current_node_id
        ]
        
        # Sort by relevance (combination of weight and importance)
        important_nodes.sort(
            key=lambda n: (n.importance * (n.access_count + 1)),
            reverse=True
        )
        
        # Find causal chain
        causal_traversal = self.index.traverse_relationships(
            start_node_id=current_node_id,
            depth=lookback_depth,
            mode=TraversalMode.CAUSAL,
            relationship_types=[RelationshipType.CAUSAL, RelationshipType.DEPENDS_ON]
        )
        
        # Get community information
        node = self.index.get_node(current_node_id)
        communities = self.index.detect_communities(min_community_size=2)
        node_communities = [
            i for i, comm in enumerate(communities.communities)
            if current_node_id in comm
        ]
        
        return {
            "current_node": node.to_dict() if node else None,
            "related_memories": [n.to_dict() for n in important_nodes[:10]],
            "causal_chain": [n.to_dict() for n in causal_traversal.nodes],
            "context_summary": traversal.context_summary,
            "communities": node_communities,
            "total_connected": len(traversal.nodes),
            "max_depth_reached": traversal.depth_reached
        }
    
    def find_similar_memories(
        self,
        content: str,
        top_k: int = 5
    ) -> List[MemoryNode]:
        """
        Find memories similar to given content.
        
        Args:
            content: Content to match
            top_k: Number of results
            
        Returns:
            List of similar memory nodes
        """
        # Use content search
        return self.index.find_nodes_by_content(content, limit=top_k)
    
    def trace_decision_reasoning(
        self,
        decision_node_id: str
    ) -> Optional[PathResult]:
        """
        Trace back the reasoning that led to a decision.
        
        Args:
            decision_node_id: ID of decision node
            
        Returns:
            PathResult with reasoning chain if found
        """
        node = self.index.get_node(decision_node_id)
        if not node or node.node_type != NodeType.DECISION:
            return None
        
        # Find strongest supporting path
        edges = self.index.get_edges_to_node(decision_node_id)
        supporting = [
            e for e in edges 
            if e.relationship_type in [RelationshipType.SUPPORTING, RelationshipType.CAUSAL]
        ]
        
        if not supporting:
            return None
        
        # Follow strongest support back
        strongest = max(supporting, key=lambda e: e.weight)
        return self.index.find_path(
            strongest.source_id,
            decision_node_id,
            algorithm="strongest"
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("Knowledge Graph Index - Demo")
    print("=" * 60)
    
    # Create index
    index = GraphIndex(":memory:")  # In-memory for demo
    
    # Add nodes
    n1 = index.add_node(
        "System design requires careful planning",
        node_type=NodeType.CONCEPT,
        importance=0.9
    )
    print(f"Added node 1: {n1}")
    
    n2 = index.add_node(
        "We should implement microservices architecture",
        node_type=NodeType.DECISION,
        importance=0.85
    )
    print(f"Added node 2: {n2}")
    
    n3 = index.add_node(
        "Microservices enable independent scaling of components",
        node_type=NodeType.FACT,
        importance=0.8
    )
    print(f"Added node 3: {n3}")
    
    n4 = index.add_node(
        "Therefore we need a message queue for service communication",
        node_type=NodeType.CONCLUSION,
        importance=0.75
    )
    print(f"Added node 4: {n4}")
    
    # Create relationships
    index.add_edge(n1, n2, RelationshipType.CAUSAL, weight=0.9)
    index.add_edge(n2, n3, RelationshipType.SUPPORTING, weight=0.85)
    index.add_edge(n3, n4, RelationshipType.CAUSAL, weight=0.8)
    index.add_edge(n2, n4, RelationshipType.DEPENDS_ON, weight=0.7)
    
    print("\nCreated relationships between nodes")
    
    # Traverse
    print("\n--- Traversal (BFS from node 1) ---")
    result = index.traverse_relationships(n1, depth=3)
    print(f"Found {len(result.nodes)} nodes, {len(result.edges)} edges")
    print(f"Path: {' -> '.join(result.path)}")
    print(f"Summary: {result.context_summary}")
    
    # Path finding
    print("\n--- Path Finding ---")
    path = index.find_path(n1, n4)
    if path:
        print(f"Path from node 1 to node 4:")
        print(f"  Length: {path.path_length}")
        print(f"  Weight: {path.total_weight:.3f}")
        print(f"  Relationships: {[r.value for r in path.relationship_chain]}")
    
    # Community detection
    print("\n--- Community Detection ---")
    communities = index.detect_communities()
    print(f"Found {communities.community_count} communities")
    print(f"Modularity: {communities.modularity:.3f}")
    
    # Statistics
    print("\n--- Statistics ---")
    stats = index.get_statistics()
    print(f"Nodes: {stats['node_count']}")
    print(f"Edges: {stats['edge_count']}")
    print(f"Avg degree: {stats['avg_degree']:.2f}")
    
    # Context recovery
    print("\n--- Context Recovery ---")
    recovery = ContextRecovery(index)
    context = recovery.recover_context(n4)
    print(f"Recovered {context['total_connected']} connected memories")
    print(f"Context: {context['context_summary']}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
