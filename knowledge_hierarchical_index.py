"""
Knowledge Hierarchical Index - Importance-Based Memory Organization

Implements a hierarchical indexing system for organizing memories by importance/depth.
High-level principles and core facts stay at the top, granular details at the leaves.

Key Features:
- Hierarchical organization (CORE → IMPORTANT → CONTEXTUAL → GRANULAR)
- Dynamic importance scoring with multiple factors
- Automatic promotion/demotion based on memory usage patterns
- Thread-safe operations
- SQLite persistence with JSON export capability
- Integration with existing KnowledgeArtifact patterns
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
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS - Memory Hierarchy Levels
# ============================================================================

class MemoryLevel(IntEnum):
    """
    Hierarchical levels for memory organization.
    Lower values = higher importance, closer to root.
    """
    CORE = 0          # High-level principles, facts that never change
    IMPORTANT = 1     # Key concepts, domain knowledge
    CONTEXTUAL = 2    # Conversation state, recent decisions
    GRANULAR = 3      # One-off details, specific examples
    
    @classmethod
    def from_string(cls, value: str) -> MemoryLevel:
        """Create MemoryLevel from string representation."""
        mapping = {
            "core": cls.CORE,
            "important": cls.IMPORTANT,
            "contextual": cls.CONTEXTUAL,
            "granular": cls.GRANULAR,
        }
        return mapping.get(value.lower(), cls.CONTEXTUAL)
    
    def to_string(self) -> str:
        """Convert MemoryLevel to string representation."""
        return self.name.lower()


# ============================================================================
# DATA CLASSES - Memory Node
# ============================================================================

@dataclass
class MemoryNode:
    """
    Represents a memory node with importance scoring and hierarchical position.
    
    Attributes:
        node_id: Unique identifier for this memory node
        content: The actual memory content (string or dict)
        level: Current hierarchical level in the index
        importance_score: Computed importance score (0.0 - 1.0)
        
        # Usage tracking
        access_count: Number of times this memory has been accessed
        last_accessed: Timestamp of last access
        created_at: When this memory was first created
        
        # Relationship tracking
        parent_id: Optional parent node ID for tree structure
        child_ids: List of child node IDs
        related_ids: List of related (peer) node IDs
        
        # Scoring factors
        frequency_score: Score based on access frequency
        centrality_score: Score based on connection to other memories
        decay_factor: Time-based decay multiplier
        user_importance: Explicit user-defined importance (0.0 - 1.0)
        semantic_density: Measure of content information density
        
        # Metadata
        tags: Searchable tags for categorization
        domain: Domain classification
        metadata: Additional flexible metadata
    """
    
    # Core identification
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Union[str, Dict[str, Any]] = field(default_factory=dict)
    level: MemoryLevel = MemoryLevel.CONTEXTUAL
    
    # Scoring
    importance_score: float = 0.5
    
    # Usage tracking
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    
    # Relationships
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    related_ids: List[str] = field(default_factory=list)
    
    # Scoring components
    frequency_score: float = 0.5
    centrality_score: float = 0.5
    decay_factor: float = 1.0
    user_importance: float = 0.5
    semantic_density: float = 0.5
    
    # Categorization
    tags: List[str] = field(default_factory=list)
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and initialize computed fields."""
        # Ensure level is MemoryLevel enum
        if isinstance(self.level, int):
            self.level = MemoryLevel(self.level)
        elif isinstance(self.level, str):
            self.level = MemoryLevel.from_string(self.level)
            
        # Validate score bounds
        self.importance_score = max(0.0, min(1.0, self.importance_score))
        self.frequency_score = max(0.0, min(1.0, self.frequency_score))
        self.centrality_score = max(0.0, min(1.0, self.centrality_score))
        self.decay_factor = max(0.0, min(1.0, self.decay_factor))
        self.user_importance = max(0.0, min(1.0, self.user_importance))
        self.semantic_density = max(0.0, min(1.0, self.semantic_density))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MemoryNode to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "content": self.content,
            "level": self.level.value,
            "level_name": self.level.name,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "created_at": self.created_at.isoformat(),
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "related_ids": self.related_ids,
            "frequency_score": self.frequency_score,
            "centrality_score": self.centrality_score,
            "decay_factor": self.decay_factor,
            "user_importance": self.user_importance,
            "semantic_density": self.semantic_density,
            "tags": self.tags,
            "domain": self.domain,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryNode:
        """Create MemoryNode from dictionary."""
        # Parse datetime fields
        last_accessed = data.get("last_accessed")
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)
        
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        return cls(
            node_id=data.get("node_id", str(uuid.uuid4())),
            content=data.get("content", {}),
            level=MemoryLevel(data.get("level", MemoryLevel.CONTEXTUAL.value)),
            importance_score=data.get("importance_score", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=last_accessed or datetime.now(),
            created_at=created_at or datetime.now(),
            parent_id=data.get("parent_id"),
            child_ids=data.get("child_ids", []),
            related_ids=data.get("related_ids", []),
            frequency_score=data.get("frequency_score", 0.5),
            centrality_score=data.get("centrality_score", 0.5),
            decay_factor=data.get("decay_factor", 1.0),
            user_importance=data.get("user_importance", 0.5),
            semantic_density=data.get("semantic_density", 0.5),
            tags=data.get("tags", []),
            domain=data.get("domain", "general"),
            metadata=data.get("metadata", {}),
        )
    
    def record_access(self) -> None:
        """Record an access to this memory node, updating tracking fields."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def add_child(self, child_id: str) -> None:
        """Add a child node ID if not already present."""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
    
    def remove_child(self, child_id: str) -> None:
        """Remove a child node ID."""
        if child_id in self.child_ids:
            self.child_ids.remove(child_id)
    
    def add_related(self, related_id: str) -> None:
        """Add a related node ID if not already present."""
        if related_id not in self.related_ids and related_id != self.node_id:
            self.related_ids.append(related_id)
    
    def calculate_age_days(self) -> float:
        """Calculate the age of this memory in days."""
        return (datetime.now() - self.created_at).total_seconds() / 86400.0


# ============================================================================
# IMPORTANCE SCORER
# ============================================================================

class ImportanceScorer:
    """
    Scores memory importance based on multiple factors:
    - Frequency of access
    - Connection to other memories (centrality)
    - Time since creation (decay)
    - Explicit user importance markers
    - Semantic density of content
    
    Weights can be configured to adjust the relative importance of each factor.
    """
    
    DEFAULT_WEIGHTS = {
        "frequency": 0.25,
        "centrality": 0.25,
        "recency": 0.20,  # Inverse of decay
        "user_importance": 0.15,
        "semantic_density": 0.15,
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the importance scorer.
        
        Args:
            weights: Optional custom weights for scoring factors.
                    Must sum to 1.0 if provided.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        # Validate weights sum to approximately 1.0
        total_weight = sum(self.weights.values())
        if not 0.99 <= total_weight <= 1.01:
            logger.warning(f"Importance scorer weights sum to {total_weight}, normalizing")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def calculate_frequency_score(self, access_count: int, 
                                  time_period_days: float = 30.0) -> float:
        """
        Calculate frequency score based on access count.
        
        Uses logarithmic scaling to prevent runaway scores for frequently
        accessed items while still rewarding usage.
        
        Args:
            access_count: Number of times memory has been accessed
            time_period_days: Time window for normalization
            
        Returns:
            Frequency score between 0.0 and 1.0
        """
        if access_count == 0:
            return 0.1  # Baseline for unused memories
        
        # Logarithmic scaling: diminishing returns after many accesses
        import math
        score = min(1.0, 0.1 + 0.9 * (math.log10(access_count + 1) / 3.0))
        return score
    
    def calculate_centrality_score(self, connection_count: int,
                                    total_memories: int) -> float:
        """
        Calculate centrality score based on connections to other memories.
        
        Higher scores for memories that are well-connected in the knowledge graph.
        
        Args:
            connection_count: Total number of connections (parent + children + related)
            total_memories: Total number of memories in the system
            
        Returns:
            Centrality score between 0.0 and 1.0
        """
        if total_memories <= 1:
            return 0.5
        
        # Normalize by expected connections (assuming sparse graph)
        expected_connections = min(5, total_memories - 1)
        if expected_connections == 0:
            return 0.5
            
        ratio = connection_count / expected_connections
        score = min(1.0, 0.3 + 0.7 * (ratio / (1 + ratio)))  # Sigmoid-like
        return score
    
    def calculate_decay_factor(self, age_days: float, 
                               half_life_days: float = 30.0) -> float:
        """
        Calculate time-based decay factor.
        
        Uses exponential decay model. Memories lose relevance over time
        unless refreshed through access.
        
        Args:
            age_days: Age of memory in days
            half_life_days: Time for importance to halve without access
            
        Returns:
            Decay factor between 0.0 and 1.0 (1.0 = no decay)
        """
        import math
        decay = math.exp(-0.693 * age_days / half_life_days)
        return max(0.1, min(1.0, decay))  # Minimum 0.1 to prevent elimination
    
    def calculate_semantic_density(self, content: Union[str, Dict[str, Any]]) -> float:
        """
        Calculate semantic density of content.
        
        Higher density indicates more information-rich content.
        Considers:
        - Information-to-length ratio
        - Presence of key indicators (numbers, technical terms, etc.)
        - Structural complexity
        
        Args:
            content: Memory content to analyze
            
        Returns:
            Semantic density score between 0.0 and 1.0
        """
        if isinstance(content, dict):
            # Convert dict to string representation for analysis
            content_str = json.dumps(content)
        else:
            content_str = str(content)
        
        if not content_str or len(content_str) < 10:
            return 0.3  # Low density for very short content
        
        # Calculate base metrics
        words = content_str.split()
        unique_words = set(w.lower() for w in words)
        
        if len(words) == 0:
            return 0.3
        
        # Lexical diversity (unique words / total words)
        lexical_diversity = len(unique_words) / len(words)
        
        # Information density indicators
        indicators = 0
        indicators += len([w for w in words if w.isdigit()]) * 0.05  # Numbers
        indicators += len([w for w in words if len(w) > 8]) * 0.02   # Long/technical words
        indicators += content_str.count(":") * 0.1                    # Key-value pairs
        indicators += content_str.count("{") * 0.05                   # Structure
        
        # Normalize indicators
        indicators = min(1.0, indicators / 10)
        
        # Combine scores (lexical diversity weighted more for short texts)
        if len(words) < 50:
            score = 0.4 * lexical_diversity + 0.4 * indicators + 0.2 * min(1.0, len(words) / 50)
        else:
            score = 0.3 * lexical_diversity + 0.5 * indicators + 0.2 * min(1.0, 100 / len(words))
        
        return max(0.1, min(1.0, score))
    
    def score_node(self, node: MemoryNode, total_memories: int = 100) -> float:
        """
        Calculate comprehensive importance score for a memory node.
        
        Args:
            node: MemoryNode to score
            total_memories: Total memories in system for centrality calculation
            
        Returns:
            Composite importance score between 0.0 and 1.0
        """
        # Calculate individual factors
        frequency = self.calculate_frequency_score(node.access_count)
        
        connection_count = len(node.child_ids) + len(node.related_ids)
        if node.parent_id:
            connection_count += 1
        centrality = self.calculate_centrality_score(connection_count, total_memories)
        
        age_days = node.calculate_age_days()
        decay = self.calculate_decay_factor(age_days)
        recency = 1.0 - (1.0 - decay) * 0.5  # Convert decay to recency score
        
        semantic = self.calculate_semantic_density(node.content)
        
        # Weighted combination
        score = (
            self.weights["frequency"] * frequency +
            self.weights["centrality"] * centrality +
            self.weights["recency"] * recency +
            self.weights["user_importance"] * node.user_importance +
            self.weights["semantic_density"] * semantic
        )
        
        # Update node's component scores
        node.frequency_score = frequency
        node.centrality_score = centrality
        node.decay_factor = decay
        node.semantic_density = semantic
        
        return max(0.0, min(1.0, score))


# ============================================================================
# HIERARCHICAL INDEX
# ============================================================================

class HierarchicalIndex:
    """
    Organizes memories in a hierarchy by importance/depth.
    
    Levels:
    - CORE (0): High-level principles, facts that never change
    - IMPORTANT (1): Key concepts, domain knowledge
    - CONTEXTUAL (2): Conversation state, recent decisions
    - GRANULAR (3): One-off details, specific examples
    
    Features:
    - Thread-safe operations
    - SQLite persistence
    - Automatic importance scoring
    - Promotion/demotion based on scores
    - Hierarchical querying
    - Domain-based organization
    """
    
    # Level boundaries for automatic promotion/demotion
    LEVEL_BOUNDARIES = {
        MemoryLevel.CORE: (0.85, 1.0),
        MemoryLevel.IMPORTANT: (0.65, 0.85),
        MemoryLevel.CONTEXTUAL: (0.35, 0.65),
        MemoryLevel.GRANULAR: (0.0, 0.35),
    }
    
    def __init__(self, storage_path: str = "./knowledge_hierarchical_index.db",
                 use_sqlite: bool = True):
        """
        Initialize the hierarchical index.
        
        Args:
            storage_path: Path to SQLite database or JSON file
            use_sqlite: If True, use SQLite; otherwise use JSON file
        """
        self.storage_path = storage_path
        self.use_sqlite = use_sqlite
        self.nodes: Dict[str, MemoryNode] = {}
        self.scorer = ImportanceScorer()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Domain-based indices
        self._domain_index: Dict[str, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._level_index: Dict[MemoryLevel, Set[str]] = defaultdict(set)
        
        # Initialize storage
        if use_sqlite:
            self._init_sqlite()
        
        # Load existing data
        self._load_from_storage()
        
        logger.info(f"HierarchicalIndex initialized with {len(self.nodes)} nodes")
    
    def _init_sqlite(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            
            # Main nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    node_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    importance_score REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    parent_id TEXT,
                    child_ids TEXT,
                    related_ids TEXT,
                    frequency_score REAL DEFAULT 0.5,
                    centrality_score REAL DEFAULT 0.5,
                    decay_factor REAL DEFAULT 1.0,
                    user_importance REAL DEFAULT 0.5,
                    semantic_density REAL DEFAULT 0.5,
                    tags TEXT,
                    domain TEXT DEFAULT 'general',
                    metadata TEXT,
                    FOREIGN KEY (parent_id) REFERENCES memory_nodes(node_id)
                )
            """)
            
            # Index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_level ON memory_nodes(level)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_domain ON memory_nodes(domain)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON memory_nodes(importance_score DESC)
            """)
            
            conn.commit()
    
    def _load_from_storage(self) -> None:
        """Load nodes from persistent storage."""
        if self.use_sqlite and Path(self.storage_path).exists():
            self._load_from_sqlite()
        elif not self.use_sqlite and Path(self.storage_path).exists():
            self._load_from_json()
    
    def _load_from_sqlite(self) -> None:
        """Load nodes from SQLite database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM memory_nodes")
                
                for row in cursor.fetchall():
                    node = self._row_to_node(row)
                    self._index_node(node)
                    
            logger.info(f"Loaded {len(self.nodes)} nodes from SQLite")
        except sqlite3.Error as e:
            logger.error(f"Error loading from SQLite: {e}")
    
    def _load_from_json(self) -> None:
        """Load nodes from JSON file."""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for node_data in data.get("nodes", []):
                node = MemoryNode.from_dict(node_data)
                self._index_node(node)
                
            logger.info(f"Loaded {len(self.nodes)} nodes from JSON")
        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.error(f"Error loading from JSON: {e}")
    
    def _row_to_node(self, row: Tuple) -> MemoryNode:
        """Convert SQLite row to MemoryNode."""
        (node_id, content, level, importance_score, access_count,
         last_accessed, created_at, parent_id, child_ids, related_ids,
         frequency_score, centrality_score, decay_factor, user_importance,
         semantic_density, tags, domain, metadata) = row
        
        # Parse JSON fields
        content = json.loads(content) if isinstance(content, str) else content
        child_ids = json.loads(child_ids) if child_ids else []
        related_ids = json.loads(related_ids) if related_ids else []
        tags = json.loads(tags) if tags else []
        metadata = json.loads(metadata) if metadata else {}
        
        return MemoryNode(
            node_id=node_id,
            content=content,
            level=MemoryLevel(level),
            importance_score=importance_score,
            access_count=access_count,
            last_accessed=datetime.fromisoformat(last_accessed),
            created_at=datetime.fromisoformat(created_at),
            parent_id=parent_id,
            child_ids=child_ids,
            related_ids=related_ids,
            frequency_score=frequency_score,
            centrality_score=centrality_score,
            decay_factor=decay_factor,
            user_importance=user_importance,
            semantic_density=semantic_density,
            tags=tags,
            domain=domain,
            metadata=metadata,
        )
    
    def _index_node(self, node: MemoryNode) -> None:
        """Add node to in-memory indices."""
        self.nodes[node.node_id] = node
        self._level_index[node.level].add(node.node_id)
        self._domain_index[node.domain].add(node.node_id)
        
        for tag in node.tags:
            self._tag_index[tag].add(node.node_id)
    
    def _unindex_node(self, node: MemoryNode) -> None:
        """Remove node from in-memory indices."""
        if node.node_id in self.nodes:
            del self.nodes[node.node_id]
        
        self._level_index[node.level].discard(node.node_id)
        self._domain_index[node.domain].discard(node.node_id)
        
        for tag in node.tags:
            self._tag_index[tag].discard(node.node_id)
    
    def _save_to_storage(self) -> None:
        """Save nodes to persistent storage."""
        if self.use_sqlite:
            self._save_to_sqlite()
        else:
            self._save_to_json()
    
    def _save_to_sqlite(self) -> None:
        """Save nodes to SQLite database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.cursor()
                
                for node in self.nodes.values():
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_nodes VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    """, (
                        node.node_id,
                        json.dumps(node.content),
                        node.level.value,
                        node.importance_score,
                        node.access_count,
                        node.last_accessed.isoformat(),
                        node.created_at.isoformat(),
                        node.parent_id,
                        json.dumps(node.child_ids),
                        json.dumps(node.related_ids),
                        node.frequency_score,
                        node.centrality_score,
                        node.decay_factor,
                        node.user_importance,
                        node.semantic_density,
                        json.dumps(node.tags),
                        node.domain,
                        json.dumps(node.metadata),
                    ))
                
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving to SQLite: {e}")
    
    def _save_to_json(self) -> None:
        """Save nodes to JSON file."""
        try:
            data = {
                "metadata": {
                    "version": "1.0",
                    "timestamp": datetime.now().isoformat(),
                    "node_count": len(self.nodes),
                },
                "nodes": [node.to_dict() for node in self.nodes.values()],
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except (IOError, OSError) as e:
            logger.error(f"Error saving to JSON: {e}")
    
    # ========================================================================
    # PUBLIC API - Memory Management
    # ========================================================================
    
    def add_memory(self, 
                   content: Union[str, Dict[str, Any]],
                   level: Union[MemoryLevel, str] = MemoryLevel.CONTEXTUAL,
                   tags: Optional[List[str]] = None,
                   domain: str = "general",
                   user_importance: float = 0.5,
                   parent_id: Optional[str] = None,
                   related_ids: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> MemoryNode:
        """
        Add a new memory to the hierarchical index.
        
        Args:
            content: The memory content (string or dict)
            level: Initial hierarchical level
            tags: Searchable tags
            domain: Domain classification
            user_importance: User-defined importance (0.0 - 1.0)
            parent_id: Optional parent node ID
            related_ids: Optional list of related node IDs
            metadata: Additional metadata
            
        Returns:
            The created MemoryNode
        """
        with self._lock:
            # Convert level string to enum if needed
            if isinstance(level, str):
                level = MemoryLevel.from_string(level)
            
            # Create node
            node = MemoryNode(
                content=content,
                level=level,
                tags=tags or [],
                domain=domain,
                user_importance=user_importance,
                parent_id=parent_id,
                related_ids=related_ids or [],
                metadata=metadata or {},
            )
            
            # Calculate initial importance score
            node.importance_score = self.scorer.score_node(node, len(self.nodes) + 1)
            
            # Update parent if specified
            if parent_id and parent_id in self.nodes:
                self.nodes[parent_id].add_child(node.node_id)
            
            # Index and save
            self._index_node(node)
            self._save_to_storage()
            
            logger.debug(f"Added memory node {node.node_id} at level {level.name}")
            return node
    
    def get_memory(self, node_id: str, record_access: bool = True) -> Optional[MemoryNode]:
        """
        Retrieve a memory node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            record_access: Whether to record this as an access
            
        Returns:
            MemoryNode if found, None otherwise
        """
        with self._lock:
            node = self.nodes.get(node_id)
            
            if node and record_access:
                node.record_access()
                self._save_to_storage()
                
            return node
    
    def update_memory(self, node_id: str, **updates) -> Optional[MemoryNode]:
        """
        Update an existing memory node.
        
        Args:
            node_id: ID of node to update
            **updates: Fields to update
            
        Returns:
            Updated MemoryNode if found, None otherwise
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                return None
            
            # Update allowed fields
            if "content" in updates:
                node.content = updates["content"]
            if "level" in updates:
                old_level = node.level
                node.level = updates["level"] if isinstance(updates["level"], MemoryLevel) else MemoryLevel.from_string(updates["level"])
                # Update level index
                self._level_index[old_level].discard(node_id)
                self._level_index[node.level].add(node_id)
            if "tags" in updates:
                # Remove from old tag indices
                for tag in node.tags:
                    self._tag_index[tag].discard(node_id)
                node.tags = updates["tags"]
                # Add to new tag indices
                for tag in node.tags:
                    self._tag_index[tag].add(node_id)
            if "domain" in updates:
                self._domain_index[node.domain].discard(node_id)
                node.domain = updates["domain"]
                self._domain_index[node.domain].add(node_id)
            if "user_importance" in updates:
                node.user_importance = max(0.0, min(1.0, updates["user_importance"]))
            if "metadata" in updates:
                node.metadata.update(updates["metadata"])
            
            # Recalculate importance
            node.importance_score = self.scorer.score_node(node, len(self.nodes))
            
            self._save_to_storage()
            return node
    
    def delete_memory(self, node_id: str) -> bool:
        """
        Delete a memory node.
        
        Args:
            node_id: ID of node to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                return False
            
            # Remove from parent's children
            if node.parent_id and node.parent_id in self.nodes:
                self.nodes[node.parent_id].remove_child(node_id)
            
            # Remove references from other nodes
            for other in self.nodes.values():
                other.remove_child(node_id)
                if node_id in other.related_ids:
                    other.related_ids.remove(node_id)
            
            # Unindex and delete
            self._unindex_node(node)
            self._save_to_storage()
            
            logger.debug(f"Deleted memory node {node_id}")
            return True
    
    # ========================================================================
    # PUBLIC API - Querying
    # ========================================================================
    
    def query_by_level(self, level: Union[MemoryLevel, str], 
                       limit: Optional[int] = None) -> List[MemoryNode]:
        """
        Query memories by hierarchical level.
        
        Args:
            level: Level to query (MemoryLevel enum or string)
            limit: Maximum number of results
            
        Returns:
            List of MemoryNodes at the specified level
        """
        if isinstance(level, str):
            level = MemoryLevel.from_string(level)
        
        with self._lock:
            node_ids = list(self._level_index[level])
            nodes = [self.nodes[nid] for nid in node_ids]
            
            # Sort by importance score (descending)
            nodes.sort(key=lambda n: n.importance_score, reverse=True)
            
            if limit:
                nodes = nodes[:limit]
            
            return nodes
    
    def query_by_domain(self, domain: str, 
                        level: Optional[Union[MemoryLevel, str]] = None,
                        limit: Optional[int] = None) -> List[MemoryNode]:
        """
        Query memories by domain.
        
        Args:
            domain: Domain to query
            level: Optional level filter
            limit: Maximum number of results
            
        Returns:
            List of MemoryNodes in the domain
        """
        with self._lock:
            node_ids = list(self._domain_index.get(domain, set()))
            nodes = [self.nodes[nid] for nid in node_ids]
            
            if level:
                if isinstance(level, str):
                    level = MemoryLevel.from_string(level)
                nodes = [n for n in nodes if n.level == level]
            
            # Sort by importance score
            nodes.sort(key=lambda n: n.importance_score, reverse=True)
            
            if limit:
                nodes = nodes[:limit]
            
            return nodes
    
    def query_by_tags(self, tags: List[str], 
                      match_all: bool = False,
                      limit: Optional[int] = None) -> List[MemoryNode]:
        """
        Query memories by tags.
        
        Args:
            tags: List of tags to search for
            match_all: If True, all tags must match; otherwise any tag matches
            limit: Maximum number of results
            
        Returns:
            List of MemoryNodes matching the tags
        """
        with self._lock:
            if not tags:
                return []
            
            # Get node IDs for each tag
            tag_sets = [self._tag_index.get(tag, set()) for tag in tags]
            
            if match_all:
                # Intersection of all sets
                node_ids = set.intersection(*tag_sets) if tag_sets else set()
            else:
                # Union of all sets
                node_ids = set.union(*tag_sets) if tag_sets else set()
            
            nodes = [self.nodes[nid] for nid in node_ids]
            nodes.sort(key=lambda n: n.importance_score, reverse=True)
            
            if limit:
                nodes = nodes[:limit]
            
            return nodes
    
    def search_content(self, query: str, 
                       limit: Optional[int] = None) -> List[Tuple[MemoryNode, float]]:
        """
        Search memory content for query string.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of (MemoryNode, relevance_score) tuples
        """
        with self._lock:
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            results = []
            for node in self.nodes.values():
                content_str = json.dumps(node.content).lower() if isinstance(node.content, dict) else str(node.content).lower()
                content_words = set(content_str.split())
                
                # Calculate Jaccard similarity
                intersection = query_words.intersection(content_words)
                union = query_words.union(content_words)
                
                if intersection:  # Only include if there's some overlap
                    similarity = len(intersection) / len(union)
                    
                    # Boost by importance score
                    relevance = 0.6 * similarity + 0.4 * node.importance_score
                    results.append((node, relevance))
            
            # Sort by relevance
            results.sort(key=lambda x: x[1], reverse=True)
            
            if limit:
                results = results[:limit]
            
            return results
    
    def get_tree_structure(self, root_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get hierarchical tree structure of memories.
        
        Args:
            root_id: Optional root node ID (if None, returns forest)
            
        Returns:
            Tree structure as nested dictionary
        """
        with self._lock:
            if root_id:
                root = self.nodes.get(root_id)
                if not root:
                    return {}
                return self._build_tree(root)
            else:
                # Find all root nodes (no parent)
                roots = [n for n in self.nodes.values() if n.parent_id is None]
                return {
                    "roots": [self._build_tree(root) for root in roots],
                    "total_nodes": len(self.nodes)
                }
    
    def _build_tree(self, node: MemoryNode, visited: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Recursively build tree structure."""
        if visited is None:
            visited = set()
        
        if node.node_id in visited:
            return {"node_id": node.node_id, "circular": True}
        
        visited.add(node.node_id)
        
        return {
            "node_id": node.node_id,
            "level": node.level.name,
            "importance_score": node.importance_score,
            "content_preview": str(node.content)[:100] + "..." if len(str(node.content)) > 100 else str(node.content),
            "children": [
                self._build_tree(self.nodes[child_id], visited.copy())
                for child_id in node.child_ids
                if child_id in self.nodes
            ]
        }
    
    # ========================================================================
    # PUBLIC API - Hierarchy Maintenance
    # ========================================================================
    
    def recalculate_importance(self, node_id: Optional[str] = None) -> None:
        """
        Recalculate importance scores.
        
        Args:
            node_id: If provided, only recalculate for this node;
                    otherwise recalculate for all nodes
        """
        with self._lock:
            if node_id:
                node = self.nodes.get(node_id)
                if node:
                    node.importance_score = self.scorer.score_node(node, len(self.nodes))
            else:
                for node in self.nodes.values():
                    node.importance_score = self.scorer.score_node(node, len(self.nodes))
            
            self._save_to_storage()
    
    def promote_demote_memories(self, 
                                auto_apply: bool = True,
                                dry_run: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Automatically adjust hierarchy levels based on importance scores.
        
        Args:
            auto_apply: If True, automatically apply promotions/demotions
            dry_run: If True, return proposed changes without applying
            
        Returns:
            Dictionary with 'promotions' and 'demotions' lists
        """
        with self._lock:
            changes = {
                "promotions": [],
                "demotions": [],
            }
            
            for node in self.nodes.values():
                current_level = node.level
                current_score = node.importance_score
                
                # Determine appropriate level based on score
                appropriate_level = self._determine_level_for_score(current_score)
                
                if appropriate_level.value < current_level.value:
                    # Should be promoted (lower value = higher level)
                    change = {
                        "node_id": node.node_id,
                        "from_level": current_level.name,
                        "to_level": appropriate_level.name,
                        "importance_score": current_score,
                        "content_preview": str(node.content)[:50],
                    }
                    changes["promotions"].append(change)
                    
                    if auto_apply and not dry_run:
                        self._change_level(node, appropriate_level)
                        
                elif appropriate_level.value > current_level.value:
                    # Should be demoted
                    change = {
                        "node_id": node.node_id,
                        "from_level": current_level.name,
                        "to_level": appropriate_level.name,
                        "importance_score": current_score,
                        "content_preview": str(node.content)[:50],
                    }
                    changes["demotions"].append(change)
                    
                    if auto_apply and not dry_run:
                        self._change_level(node, appropriate_level)
            
            if auto_apply and not dry_run:
                self._save_to_storage()
            
            return changes
    
    def _determine_level_for_score(self, score: float) -> MemoryLevel:
        """Determine appropriate level for a given importance score."""
        for level, (min_score, max_score) in self.LEVEL_BOUNDARIES.items():
            if min_score <= score <= max_score:
                return level
        return MemoryLevel.GRANULAR  # Default fallback
    
    def _change_level(self, node: MemoryNode, new_level: MemoryLevel) -> None:
        """Change a node's level, updating indices."""
        old_level = node.level
        
        # Update level index
        self._level_index[old_level].discard(node.node_id)
        self._level_index[new_level].add(node.node_id)
        
        # Update node
        node.level = new_level
        
        logger.info(f"Changed node {node.node_id} from {old_level.name} to {new_level.name}")
    
    def promote_node(self, node_id: str, levels: int = 1) -> Optional[MemoryNode]:
        """
        Manually promote a node up the hierarchy.
        
        Args:
            node_id: ID of node to promote
            levels: Number of levels to promote (default 1)
            
        Returns:
            Updated MemoryNode if found, None otherwise
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                return None
            
            new_level_value = max(0, node.level.value - levels)
            new_level = MemoryLevel(new_level_value)
            
            self._change_level(node, new_level)
            self._save_to_storage()
            
            return node
    
    def demote_node(self, node_id: str, levels: int = 1) -> Optional[MemoryNode]:
        """
        Manually demote a node down the hierarchy.
        
        Args:
            node_id: ID of node to demote
            levels: Number of levels to demote (default 1)
            
        Returns:
            Updated MemoryNode if found, None otherwise
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                return None
            
            new_level_value = min(3, node.level.value + levels)
            new_level = MemoryLevel(new_level_value)
            
            self._change_level(node, new_level)
            self._save_to_storage()
            
            return node
    
    # ========================================================================
    # PUBLIC API - Statistics and Export
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the hierarchical index.
        
        Returns:
            Dictionary with various statistics
        """
        with self._lock:
            stats = {
                "total_nodes": len(self.nodes),
                "by_level": {
                    level.name: len(self._level_index[level])
                    for level in MemoryLevel
                },
                "by_domain": {
                    domain: len(nodes)
                    for domain, nodes in self._domain_index.items()
                },
                "average_importance": 0.0,
                "top_nodes": [],
            }
            
            if self.nodes:
                avg_importance = sum(n.importance_score for n in self.nodes.values()) / len(self.nodes)
                stats["average_importance"] = round(avg_importance, 3)
                
                # Top 5 nodes by importance
                top_nodes = sorted(self.nodes.values(), 
                                   key=lambda n: n.importance_score, 
                                   reverse=True)[:5]
                stats["top_nodes"] = [
                    {
                        "node_id": n.node_id,
                        "level": n.level.name,
                        "importance_score": n.importance_score,
                        "access_count": n.access_count,
                    }
                    for n in top_nodes
                ]
            
            return stats
    
    def export_to_json(self, file_path: str) -> None:
        """
        Export the entire index to a JSON file.
        
        Args:
            file_path: Path to export file
        """
        with self._lock:
            data = {
                "metadata": {
                    "version": "1.0",
                    "export_timestamp": datetime.now().isoformat(),
                    "total_nodes": len(self.nodes),
                },
                "statistics": self.get_statistics(),
                "nodes": [node.to_dict() for node in self.nodes.values()],
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported {len(self.nodes)} nodes to {file_path}")
    
    def import_from_json(self, file_path: str, 
                         merge: bool = False) -> int:
        """
        Import memories from a JSON file.
        
        Args:
            file_path: Path to import file
            merge: If True, merge with existing nodes; if False, replace
            
        Returns:
            Number of nodes imported
        """
        with self._lock:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not merge:
                # Clear existing nodes
                self.nodes.clear()
                self._level_index.clear()
                self._domain_index.clear()
                self._tag_index.clear()
            
            imported = 0
            for node_data in data.get("nodes", []):
                try:
                    node = MemoryNode.from_dict(node_data)
                    self._index_node(node)
                    imported += 1
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to import node: {e}")
            
            self._save_to_storage()
            logger.info(f"Imported {imported} nodes from {file_path}")
            return imported
    
    def clear_all(self) -> None:
        """Clear all memories from the index."""
        with self._lock:
            self.nodes.clear()
            self._level_index.clear()
            self._domain_index.clear()
            self._tag_index.clear()
            self._save_to_storage()
            logger.info("Cleared all memories from index")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_hierarchical_index(storage_path: str = "./knowledge_hierarchical_index.db",
                               use_sqlite: bool = True) -> HierarchicalIndex:
    """
    Factory function to create a HierarchicalIndex instance.
    
    Args:
        storage_path: Path to storage file
        use_sqlite: Whether to use SQLite (True) or JSON (False)
        
    Returns:
        Configured HierarchicalIndex instance
    """
    return HierarchicalIndex(storage_path=storage_path, use_sqlite=use_sqlite)


def promote_demote_memories(index: HierarchicalIndex,
                            auto_apply: bool = True,
                            dry_run: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    Standalone function to promote/demote memories in an index.
    
    Args:
        index: HierarchicalIndex instance
        auto_apply: Whether to automatically apply changes
        dry_run: Whether to simulate without applying
        
    Returns:
        Dictionary with promotion and demotion details
    """
    return index.promote_demote_memories(auto_apply=auto_apply, dry_run=dry_run)


# ============================================================================
# Integration with existing KnowledgeArtifact patterns
# ============================================================================

def convert_artifact_to_node(artifact: Any,
                              level: MemoryLevel = MemoryLevel.CONTEXTUAL) -> MemoryNode:
    """
    Convert a KnowledgeArtifact to a MemoryNode.
    
    Compatible with workflow_structures.KnowledgeArtifact and
    knowledge_engine.schemas.base.KnowledgeArtifact.
    
    Args:
        artifact: KnowledgeArtifact to convert
        level: Target memory level
        
    Returns:
        MemoryNode populated from artifact data
    """
    # Extract fields from artifact (handle different artifact types)
    artifact_id = getattr(artifact, 'id', None) or getattr(artifact, 'artifact_id', str(uuid.uuid4()))
    content = getattr(artifact, 'content', {})
    domain = getattr(artifact, 'domain', 'general')
    tags = getattr(artifact, 'tags', [])
    
    # Handle timestamps
    created_at = getattr(artifact, 'timestamp', None) or getattr(artifact, 'created_at', None)
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    elif created_at is None:
        created_at = datetime.now()
    
    # Handle effectiveness/confidence as user_importance proxy
    effectiveness = getattr(artifact, 'effectiveness_score', None) or getattr(artifact, 'confidence', 0.5)
    
    return MemoryNode(
        node_id=str(artifact_id),
        content=content,
        level=level,
        tags=list(tags) if tags else [],
        domain=str(domain) if domain else 'general',
        user_importance=effectiveness if isinstance(effectiveness, (int, float)) else 0.5,
        created_at=created_at,
        metadata={
            "source_artifact": True,
            "artifact_type": getattr(artifact, 'artifact_type', 'unknown'),
        }
    )


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create index
    index = HierarchicalIndex(storage_path=":memory:", use_sqlite=True)
    
    # Add some memories at different levels
    core_mem = index.add_memory(
        content="Always validate user input before processing",
        level=MemoryLevel.CORE,
        tags=["security", "validation"],
        domain="security",
        user_importance=0.95
    )
    
    important_mem = index.add_memory(
        content={"pattern": "Circuit Breaker", "use_case": "API resilience"},
        level=MemoryLevel.IMPORTANT,
        tags=["patterns", "resilience"],
        domain="architecture",
        parent_id=core_mem.node_id
    )
    
    contextual_mem = index.add_memory(
        content="Current conversation is about database optimization",
        level=MemoryLevel.CONTEXTUAL,
        tags=["conversation", "database"],
        domain="database"
    )
    
    granular_mem = index.add_memory(
        content="The specific query took 150ms to execute",
        level=MemoryLevel.GRANULAR,
        tags=["performance", "metrics"],
        domain="database"
    )
    
    # Access some memories to update frequency scores
    for _ in range(10):
        index.get_memory(core_mem.node_id)
    for _ in range(5):
        index.get_memory(important_mem.node_id)
    
    # Recalculate importance
    index.recalculate_importance()
    
    # Check for promotions/demotions
    changes = index.promote_demote_memories(dry_run=True)
    print("Proposed changes:", json.dumps(changes, indent=2))
    
    # Get statistics
    stats = index.get_statistics()
    print("Statistics:", json.dumps(stats, indent=2))
    
    # Query by level
    core_memories = index.query_by_level(MemoryLevel.CORE)
    print(f"\nCore memories: {len(core_memories)}")
    
    # Query by domain
    db_memories = index.query_by_domain("database")
    print(f"Database memories: {len(db_memories)}")
    
    # Search
    results = index.search_content("security")
    print(f"\nSearch results for 'security': {len(results)}")
    
    # Tree structure
    tree = index.get_tree_structure()
    print("\nTree structure:", json.dumps(tree, indent=2))
