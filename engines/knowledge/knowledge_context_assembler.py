"""
Knowledge Context Assembler - Unified Context Curation System

This module implements a unified context assembler that prevents "context rot" in long
conversations (50,000+ words) by curating context through four specialized indexes before
feeding to LLM. Instead of providing raw transcripts, it produces a structured "state of
the union" view.

Four-Stage Pipeline:
    1. Hierarchical Stage: Retrieve memories by importance level (CORE -> IMPORTANT -> CONTEXTUAL -> GRANULAR)
    2. Graph Stage: Follow relationship paths to connect distant memories
    3. Deduplication Stage: Use hash index to remove/merge near-duplicates
    4. Semantic Stage: Re-rank remaining by semantic similarity to current query

Key Features:
    - Importance-weighted token budget allocation
    - Cross-reference linking (shows how memories connect)
    - Temporal decay with exception for CORE memories
    - Dynamic token budget management
    - Context freshness scoring
    - Thread-safe operations
    - SQLite persistence

Integration:
    - knowledge_hierarchical_index: Importance-based memory organization
    - knowledge_graph_index: Logical relationship preservation
    - knowledge_hash_index: Deduplication layer
    - knowledge_semantic_index: Semantic similarity ranking

Example:
    >>> assembler = UnifiedContextAssembler()
    >>> context = assembler.assemble(
    ...     query="How do we handle error recovery?",
    ...     conversation_history=history,
    ...     max_tokens=4000
    ... )
    >>> print(context.to_llm_format())

Author: OpenEvolve AI
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Protocol, Set, 
    Tuple, TypeVar, Union, Iterator
)
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# OPTIONAL DEPENDENCIES WITH GRACEFUL FALLBACKS
# =============================================================================

# Import hierarchical index
try:
    from knowledge_hierarchical_index import (
        HierarchicalIndex, MemoryLevel, MemoryNode as HierarchicalMemoryNode
    )
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    logger.warning("knowledge_hierarchical_index not available")
    MemoryLevel = None
    HierarchicalMemoryNode = None
    HierarchicalIndex = None

# Import graph index
try:
    from knowledge_graph_index import (
        GraphIndex, RelationshipType, MemoryNode as GraphMemoryNode,
        RelationshipEdge, TraversalResult
    )
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    logger.warning("knowledge_graph_index not available")
    RelationshipType = None
    GraphMemoryNode = None
    RelationshipEdge = None
    TraversalResult = None
    GraphIndex = None

# Import hash index
try:
    from knowledge_hash_index import (
        HashIndex, HashEntry, HashIndexConfig, compute_simhash, hamming_distance
    )
    HASH_AVAILABLE = True
except ImportError:
    HASH_AVAILABLE = False
    logger.warning("knowledge_hash_index not available")
    HashEntry = None
    HashIndexConfig = None
    HashIndex = None

# Import semantic index
try:
    from knowledge_semantic_index import (
        SemanticIndex, SemanticQuery, SemanticResult, EmbeddingGenerator,
        SemanticIndexConfig, generate_embedding
    )
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logger.warning("knowledge_semantic_index not available")
    SemanticQuery = None
    SemanticResult = None
    EmbeddingGenerator = None
    SemanticIndexConfig = None
    SemanticIndex = None

# **LEAN INTEGRATION**: Formal verification context
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ContextAssemblyStage(Enum):
    """Stages in the context assembly pipeline."""
    HIERARCHICAL = "hierarchical"
    GRAPH = "graph"
    DEDUPLICATION = "deduplication"
    SEMANTIC = "semantic"
    FORMATTING = "formatting"


class ContextSectionType(Enum):
    """Types of sections in assembled context."""
    CORE_PRINCIPLES = "core_principles"
    KEY_RELATIONSHIPS = "key_relationships"
    RECENT_DETAILS = "recent_details"
    CONNECTION_SUMMARY = "connection_summary"
    QUERY_CONTEXT = "query_context"


class TokenBudgetStrategy(Enum):
    """Strategies for allocating token budgets."""
    IMPORTANCE_WEIGHTED = "importance_weighted"
    LEVEL_BALANCED = "level_balanced"
    QUERY_FOCUSED = "query_focused"
    TEMPORAL_DECAY = "temporal_decay"


# Default configuration constants
DEFAULT_MAX_TOKENS = 4000
DEFAULT_CORE_TOKEN_RATIO = 0.25  # 25% for CORE level
DEFAULT_IMPORTANT_TOKEN_RATIO = 0.30  # 30% for IMPORTANT level
DEFAULT_CONTEXTUAL_TOKEN_RATIO = 0.30  # 30% for CONTEXTUAL level
DEFAULT_GRANULAR_TOKEN_RATIO = 0.15  # 15% for GRANULAR level

TOKENS_PER_WORD = 1.3  # Approximate tokens per word for GPT models
MAX_CONTEXT_AGE_DAYS = 30  # Maximum age for non-CORE memories


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ContextAssemblerConfig:
    """
    Configuration for the Unified Context Assembler.
    
    Attributes:
        db_path: Path to SQLite database for persistence
        max_tokens: Maximum tokens in assembled context
        token_budget_strategy: How to allocate token budgets
        enable_hierarchical: Whether to use hierarchical index
        enable_graph: Whether to use graph index
        enable_deduplication: Whether to use hash index for deduplication
        enable_semantic: Whether to use semantic index
        core_token_ratio: Token ratio for CORE level memories
        important_token_ratio: Token ratio for IMPORTANT level memories
        contextual_token_ratio: Token ratio for CONTEXTUAL level memories
        granular_token_ratio: Token ratio for GRANULAR level memories
        temporal_decay_enabled: Whether to apply temporal decay
        max_context_age_days: Maximum age for non-CORE memories
        cross_reference_enabled: Whether to include cross-references
        freshness_scoring_enabled: Whether to calculate freshness scores
        thread_safe: Whether to enable thread safety
        cache_enabled: Whether to enable caching
        cache_ttl_seconds: Cache time-to-live in seconds
    """
    
    # Storage
    db_path: str = "./knowledge_context_assembler.db"
    
    # Token budgets
    max_tokens: int = DEFAULT_MAX_TOKENS
    token_budget_strategy: TokenBudgetStrategy = TokenBudgetStrategy.IMPORTANCE_WEIGHTED
    
    # Stage enablement
    enable_hierarchical: bool = True
    enable_graph: bool = True
    enable_deduplication: bool = True
    enable_semantic: bool = True
    
    # Token allocation ratios (must sum to <= 1.0)
    core_token_ratio: float = DEFAULT_CORE_TOKEN_RATIO
    important_token_ratio: float = DEFAULT_IMPORTANT_TOKEN_RATIO
    contextual_token_ratio: float = DEFAULT_CONTEXTUAL_TOKEN_RATIO
    granular_token_ratio: float = DEFAULT_GRANULAR_TOKEN_RATIO
    
    # Temporal settings
    temporal_decay_enabled: bool = True
    max_context_age_days: int = MAX_CONTEXT_AGE_DAYS
    core_exempt_from_decay: bool = True
    
    # Feature toggles
    cross_reference_enabled: bool = True
    freshness_scoring_enabled: bool = True
    connection_summary_enabled: bool = True
    
    # Performance
    thread_safe: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    
    # Integration paths
    hierarchical_index_path: str = "./knowledge_hierarchical_index.db"
    graph_index_path: str = "./knowledge_graph_index.db"
    hash_index_path: str = "./knowledge_hash_index.db"
    semantic_cache_dir: str = "./knowledge_semantic_cache"
    
    def __post_init__(self):
        """Validate configuration."""
        total_ratio = (
            self.core_token_ratio + self.important_token_ratio +
            self.contextual_token_ratio + self.granular_token_ratio
        )
        if total_ratio > 1.0:
            logger.warning(f"Token ratios sum to {total_ratio:.2f}, normalizing")
            factor = 1.0 / total_ratio
            self.core_token_ratio *= factor
            self.important_token_ratio *= factor
            self.contextual_token_ratio *= factor
            self.granular_token_ratio *= factor
    
    def get_level_budget(self, level: Any) -> int:
        """Get token budget for a specific memory level."""
        if not HIERARCHICAL_AVAILABLE:
            return self.max_tokens // 4
        
        if level == MemoryLevel.CORE:
            return int(self.max_tokens * self.core_token_ratio)
        elif level == MemoryLevel.IMPORTANT:
            return int(self.max_tokens * self.important_token_ratio)
        elif level == MemoryLevel.CONTEXTUAL:
            return int(self.max_tokens * self.contextual_token_ratio)
        elif level == MemoryLevel.GRANULAR:
            return int(self.max_tokens * self.granular_token_ratio)
        return self.max_tokens // 4


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ContextItem:
    """
    A single item in the assembled context.
    
    Represents a memory that has passed through all pipeline stages
    and is ready for inclusion in the final context.
    """
    
    # Core identification
    item_id: str
    content: str
    
    # Source information
    source_level: Optional[Any] = None  # MemoryLevel from hierarchical index
    source_node_id: Optional[str] = None  # Node ID from graph index
    
    # Scoring
    importance_score: float = 0.5
    semantic_similarity: float = 0.5
    freshness_score: float = 0.5
    composite_score: float = 0.5
    
    # Relationships
    related_item_ids: List[str] = field(default_factory=list)
    relationship_types: List[str] = field(default_factory=list)
    
    # Temporal
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    
    # Metadata
    section_type: ContextSectionType = ContextSectionType.RECENT_DETAILS
    tags: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate composite score if not set."""
        if self.composite_score == 0.5:
            self.composite_score = (
                0.4 * self.importance_score +
                0.4 * self.semantic_similarity +
                0.2 * self.freshness_score
            )
    
    def estimate_tokens(self) -> int:
        """Estimate token count for this item."""
        word_count = len(self.content.split())
        return int(word_count * TOKENS_PER_WORD)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "importance_score": self.importance_score,
            "semantic_similarity": self.semantic_similarity,
            "freshness_score": self.freshness_score,
            "composite_score": self.composite_score,
            "section_type": self.section_type.value,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class AssembledContext:
    """
    Final structured context ready for LLM consumption.
    
    Contains:
    - Core principles (from hierarchy level 0/CORE)
    - Key relationships (from graph paths)
    - Recent relevant details (deduplicated)
    - Connection summaries (how it all links together)
    """
    
    # Core sections
    core_principles: List[ContextItem] = field(default_factory=list)
    key_relationships: List[ContextItem] = field(default_factory=list)
    recent_details: List[ContextItem] = field(default_factory=list)
    connection_summary: str = ""
    
    # Query context
    query: str = ""
    query_embedding: Optional[np.ndarray] = None
    
    # Metadata
    total_tokens: int = 0
    item_count: int = 0
    freshness_score: float = 0.0
    assembly_timestamp: datetime = field(default_factory=datetime.now)
    pipeline_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Source tracking
    source_ids: Set[str] = field(default_factory=set)
    cross_references: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_item(self, item: ContextItem, section: ContextSectionType) -> None:
        """Add an item to the appropriate section."""
        item.section_type = section
        
        if section == ContextSectionType.CORE_PRINCIPLES:
            self.core_principles.append(item)
        elif section == ContextSectionType.KEY_RELATIONSHIPS:
            self.key_relationships.append(item)
        elif section == ContextSectionType.RECENT_DETAILS:
            self.recent_details.append(item)
        
        self.source_ids.add(item.item_id)
        self.item_count += 1
        self.total_tokens += item.estimate_tokens()
    
    def to_llm_format(self, include_metadata: bool = False) -> str:
        """
        Convert to formatted string for LLM consumption.
        
        Format:
        === CORE PRINCIPLES ===
        [Core items]
        
        === KEY RELATIONSHIPS ===
        [Relationship items with cross-references]
        
        === RECENT RELEVANT DETAILS ===
        [Recent items]
        
        === CONNECTION SUMMARY ===
        [How everything connects]
        """
        sections = []
        
        # Core Principles
        if self.core_principles:
            sections.append("=== CORE PRINCIPLES ===")
            for item in sorted(self.core_principles, 
                             key=lambda x: x.composite_score, reverse=True):
                sections.append(f"* {item.content}")
                if item.cross_references:
                    sections.append(f"  [See also: {', '.join(item.cross_references)}]")
        
        # Key Relationships
        if self.key_relationships:
            sections.append("\n=== KEY RELATIONSHIPS ===")
            for item in sorted(self.key_relationships,
                             key=lambda x: x.composite_score, reverse=True):
                sections.append(f"* {item.content}")
                if item.related_item_ids:
                    sections.append(f"  [Related to: {', '.join(item.related_item_ids[:3])}]")
        
        # Recent Details
        if self.recent_details:
            sections.append("\n=== RECENT RELEVANT DETAILS ===")
            for item in sorted(self.recent_details,
                             key=lambda x: x.created_at, reverse=True)[:10]:
                sections.append(f"* {item.content}")
        
        # Connection Summary
        if self.connection_summary:
            sections.append(f"\n=== CONNECTION SUMMARY ===\n{self.connection_summary}")
        
        # Metadata footer
        if include_metadata:
            sections.append(f"\n---")
            sections.append(f"Context assembled: {self.assembly_timestamp.isoformat()}")
            sections.append(f"Items: {self.item_count}, Tokens: {self.total_tokens}")
            sections.append(f"Freshness: {self.freshness_score:.2f}")
        
        return "\n".join(sections)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "core_principles": [item.to_dict() for item in self.core_principles],
            "key_relationships": [item.to_dict() for item in self.key_relationships],
            "recent_details": [item.to_dict() for item in self.recent_details],
            "connection_summary": self.connection_summary,
            "query": self.query,
            "total_tokens": self.total_tokens,
            "item_count": self.item_count,
            "freshness_score": self.freshness_score,
            "assembly_timestamp": self.assembly_timestamp.isoformat(),
            "pipeline_stats": self.pipeline_stats,
        }


@dataclass
class PipelineStageResult:
    """Result from a single pipeline stage."""
    stage: ContextAssemblyStage
    items: List[ContextItem] = field(default_factory=list)
    items_removed: int = 0
    items_added: int = 0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "items_count": len(self.items),
            "items_removed": self.items_removed,
            "items_added": self.items_added,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class AssemblyPipelineResult:
    """Complete result from the assembly pipeline."""
    context: AssembledContext
    stage_results: List[PipelineStageResult] = field(default_factory=list)
    total_processing_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "stage_results": [sr.to_dict() for sr in self.stage_results],
            "total_processing_time_ms": self.total_processing_time_ms,
            "success": self.success,
            "error_message": self.error_message,
        }


# =============================================================================
# STAGE IMPLEMENTATIONS
# =============================================================================

class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""
    
    def __init__(self, config: ContextAssemblerConfig):
        self.config = config
        self.stage_type: ContextAssemblyStage = ContextAssemblyStage.FORMATTING
    
    @abstractmethod
    def process(
        self,
        items: List[ContextItem],
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        **kwargs
    ) -> PipelineStageResult:
        """
        Process items through this stage.
        
        Args:
            items: Input items from previous stage
            query: Current query string
            query_embedding: Pre-computed query embedding
            **kwargs: Additional stage-specific arguments
            
        Returns:
            PipelineStageResult with processed items
        """
        pass
    
    def _calculate_freshness_score(self, item: ContextItem) -> float:
        """Calculate freshness score based on age and access patterns."""
        if not self.config.temporal_decay_enabled:
            return 1.0
        
        age_days = (datetime.now() - item.created_at).total_seconds() / 86400
        
        # CORE memories are exempt from decay
        if self.config.core_exempt_from_decay and item.source_level:
            if HIERARCHICAL_AVAILABLE and item.source_level == MemoryLevel.CORE:
                return 1.0
        
        # Exponential decay
        half_life = self.config.max_context_age_days / 3
        decay = np.exp(-0.693 * age_days / half_life) if half_life > 0 else 1.0
        
        # Boost for recently accessed
        access_boost = min(0.2, item.last_accessed.timestamp() / (time.time() + 1))
        
        return min(1.0, decay + access_boost)


class HierarchicalStage(PipelineStage):
    """
    Stage 1: Retrieve memories by hierarchical level.
    
    Retrieves memories in priority order:
    1. CORE - High-level principles (never change)
    2. IMPORTANT - Key concepts and domain knowledge
    3. CONTEXTUAL - Conversation state and recent decisions
    4. GRANULAR - One-off details and specific examples
    """
    
    def __init__(self, config: ContextAssemblerConfig, hierarchical_index: Optional[Any] = None):
        super().__init__(config)
        self.stage_type = ContextAssemblyStage.HIERARCHICAL
        self.hierarchical_index = hierarchical_index
        
        if self.hierarchical_index is None and HIERARCHICAL_AVAILABLE:
            try:
                self.hierarchical_index = HierarchicalIndex(
                    storage_path=config.hierarchical_index_path
                )
            except Exception as e:
                logger.warning(f"Failed to initialize hierarchical index: {e}")
    
    def process(
        self,
        items: List[ContextItem],
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        **kwargs
    ) -> PipelineStageResult:
        """Retrieve memories by hierarchical level."""
        start_time = time.time()
        
        if not self.hierarchical_index or not HIERARCHICAL_AVAILABLE:
            return PipelineStageResult(
                stage=self.stage_type,
                items=items,
                metadata={"note": "Hierarchical index not available"}
            )
        
        result_items = []
        metadata = {"levels_queried": [], "items_by_level": {}}
        
        # Query each level with its budget
        levels = [MemoryLevel.CORE, MemoryLevel.IMPORTANT, 
                 MemoryLevel.CONTEXTUAL, MemoryLevel.GRANULAR]
        
        for level in levels:
            try:
                budget = self.config.get_level_budget(level)
                nodes = self.hierarchical_index.query_by_level(level, limit=100)
                
                # Filter by temporal decay (except CORE)
                if self.config.temporal_decay_enabled and level != MemoryLevel.CORE:
                    cutoff = datetime.now() - timedelta(days=self.config.max_context_age_days)
                    nodes = [n for n in nodes if n.created_at > cutoff]
                
                # Sort by importance score and take top within budget
                nodes.sort(key=lambda n: n.importance_score, reverse=True)
                
                current_tokens = 0
                level_items = []
                
                for node in nodes:
                    item = ContextItem(
                        item_id=node.node_id,
                        content=str(node.content),
                        source_level=level,
                        importance_score=node.importance_score,
                        created_at=node.created_at,
                        last_accessed=node.last_accessed,
                        tags=node.tags
                    )
                    
                    item_tokens = item.estimate_tokens()
                    if current_tokens + item_tokens > budget:
                        break
                    
                    current_tokens += item_tokens
                    level_items.append(item)
                
                result_items.extend(level_items)
                metadata["levels_queried"].append(level.name)
                metadata["items_by_level"][level.name] = len(level_items)
                
            except Exception as e:
                logger.warning(f"Error querying level {level}: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        return PipelineStageResult(
            stage=self.stage_type,
            items=result_items,
            items_added=len(result_items),
            processing_time_ms=processing_time,
            metadata=metadata
        )


class GraphStage(PipelineStage):
    """
    Stage 2: Follow relationship paths to connect distant memories.
    
    Uses graph traversal to:
    - Find related memories that might not be semantically similar
    - Follow causal chains
    - Connect memories across time through relationships
    - Build context webs around key concepts
    """
    
    def __init__(self, config: ContextAssemblerConfig, graph_index: Optional[Any] = None):
        super().__init__(config)
        self.stage_type = ContextAssemblyStage.GRAPH
        self.graph_index = graph_index
        
        if self.graph_index is None and GRAPH_AVAILABLE:
            try:
                self.graph_index = GraphIndex(db_path=config.graph_index_path)
            except Exception as e:
                logger.warning(f"Failed to initialize graph index: {e}")
    
    def process(
        self,
        items: List[ContextItem],
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        **kwargs
    ) -> PipelineStageResult:
        """Follow relationship paths to enrich context."""
        start_time = time.time()
        
        if not self.graph_index or not GRAPH_AVAILABLE:
            return PipelineStageResult(
                stage=self.stage_type,
                items=items,
                metadata={"note": "Graph index not available"}
            )
        
        # Track all items and their relationships
        enriched_items = items.copy()
        added_item_ids = set(item.item_id for item in items)
        relationship_chains = defaultdict(list)
        
        # For each item, traverse its relationships
        for item in items:
            try:
                # Find the corresponding graph node
                # Try to get by node_id if it matches
                node = self.graph_index.get_node(item.item_id, update_access=False)
                
                if not node:
                    # Try to find by content similarity
                    matches = self.graph_index.find_nodes_by_content(
                        item.content[:100], limit=1
                    )
                    if matches:
                        node = matches[0]
                
                if node:
                    # Traverse relationships
                    traversal = self.graph_index.traverse_relationships(
                        node.node_id, depth=2, max_nodes=10
                    )
                    
                    # Add discovered nodes
                    for related_node in traversal.nodes:
                        if related_node.node_id not in added_item_ids:
                            related_item = ContextItem(
                                item_id=related_node.node_id,
                                content=related_node.content,
                                source_node_id=node.node_id,
                                importance_score=related_node.importance,
                                created_at=related_node.timestamp
                            )
                            enriched_items.append(related_item)
                            added_item_ids.add(related_node.node_id)
                            relationship_chains[item.item_id].append(related_node.node_id)
                    
                    # Update original item with relationships
                    item.related_item_ids = [n.node_id for n in traversal.nodes 
                                            if n.node_id != item.item_id][:5]
                    
            except Exception as e:
                logger.warning(f"Error traversing graph for item {item.item_id}: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        return PipelineStageResult(
            stage=self.stage_type,
            items=enriched_items,
            items_added=len(enriched_items) - len(items),
            processing_time_ms=processing_time,
            metadata={
                "relationship_chains_found": len(relationship_chains),
                "avg_chain_length": np.mean([len(v) for v in relationship_chains.values()]) 
                                   if relationship_chains else 0
            }
        )


class DeduplicationStage(PipelineStage):
    """
    Stage 3: Remove or merge near-duplicate memories.
    
    Uses hash-based deduplication to:
    - Remove exact duplicates
    - Merge near-duplicates using SimHash
    - Preserve the most detailed/informative version
    - Track merge history
    """
    
    def __init__(self, config: ContextAssemblerConfig, hash_index: Optional[Any] = None):
        super().__init__(config)
        self.stage_type = ContextAssemblyStage.DEDUPLICATION
        self.hash_index = hash_index
        
        if self.hash_index is None and HASH_AVAILABLE:
            try:
                hash_config = HashIndexConfig(db_path=config.hash_index_path)
                self.hash_index = HashIndex(config=hash_config)
            except Exception as e:
                logger.warning(f"Failed to initialize hash index: {e}")
    
    def process(
        self,
        items: List[ContextItem],
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        **kwargs
    ) -> PipelineStageResult:
        """Deduplicate items using hash index."""
        start_time = time.time()
        
        if not self.hash_index or not HASH_AVAILABLE:
            return PipelineStageResult(
                stage=self.stage_type,
                items=items,
                metadata={"note": "Hash index not available"}
            )
        
        unique_items = []
        removed_count = 0
        merged_groups = []
        
        # Group items by content hash
        content_groups = defaultdict(list)
        
        for item in items:
            try:
                # Compute simhash for near-duplicate detection
                simhash = compute_simhash(item.content)
                
                # Check against existing groups
                found_duplicate = False
                for existing_hash, group in list(content_groups.items()):
                    distance = hamming_distance(simhash, existing_hash)
                    if distance <= 3:  # Near-duplicate threshold
                        content_groups[existing_hash].append(item)
                        found_duplicate = True
                        break
                
                if not found_duplicate:
                    content_groups[simhash].append(item)
                    
            except Exception as e:
                logger.warning(f"Error computing hash for item {item.item_id}: {e}")
                unique_items.append(item)
        
        # For each group, select the best item
        for simhash, group in content_groups.items():
            if len(group) == 1:
                unique_items.append(group[0])
            else:
                # Select best item (highest composite score, longest content)
                best_item = max(group, 
                    key=lambda x: (x.composite_score, len(x.content), x.importance_score))
                
                # Merge tags and cross-references
                for item in group:
                    if item != best_item:
                        best_item.tags = list(set(best_item.tags + item.tags))
                        best_item.cross_references.extend(item.cross_references)
                        removed_count += 1
                
                merged_groups.append({
                    "kept": best_item.item_id,
                    "merged": [item.item_id for item in group if item != best_item],
                    "count": len(group)
                })
                
                unique_items.append(best_item)
        
        processing_time = (time.time() - start_time) * 1000
        
        return PipelineStageResult(
            stage=self.stage_type,
            items=unique_items,
            items_removed=removed_count,
            processing_time_ms=processing_time,
            metadata={
                "groups_merged": len(merged_groups),
                "merge_details": merged_groups[:5]  # Limit details
            }
        )


class SemanticStage(PipelineStage):
    """
    Stage 4: Re-rank remaining items by semantic similarity to query.
    
    Uses vector embeddings to:
    - Calculate semantic similarity between query and each item
    - Re-rank items by relevance
    - Filter items below similarity threshold
    - Generate connection summaries
    """
    
    def __init__(self, config: ContextAssemblerConfig, semantic_index: Optional[Any] = None):
        super().__init__(config)
        self.stage_type = ContextAssemblyStage.SEMANTIC
        self.semantic_index = semantic_index
        self.embedding_generator: Optional[Any] = None
        
        if SEMANTIC_AVAILABLE:
            try:
                semantic_config = SemanticIndexConfig(
                    cache_dir=config.semantic_cache_dir
                )
                self.embedding_generator = EmbeddingGenerator(semantic_config)
            except Exception as e:
                logger.warning(f"Failed to initialize embedding generator: {e}")
    
    def process(
        self,
        items: List[ContextItem],
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        **kwargs
    ) -> PipelineStageResult:
        """Re-rank items by semantic similarity to query."""
        start_time = time.time()
        
        if not self.embedding_generator or not SEMANTIC_AVAILABLE:
            # Fallback: use keyword matching
            return self._keyword_ranking(items, query)
        
        try:
            # Generate query embedding if not provided
            if query_embedding is None:
                query_embedding = self.embedding_generator.generate(query)
            
            # Calculate similarity for each item
            scored_items = []
            
            for item in items:
                try:
                    item_embedding = self.embedding_generator.generate(item.content)
                    similarity = self._cosine_similarity(query_embedding, item_embedding)
                    item.semantic_similarity = float(similarity)
                    
                    # Update composite score with semantic weight
                    item.composite_score = (
                        0.3 * item.importance_score +
                        0.5 * item.semantic_similarity +
                        0.2 * self._calculate_freshness_score(item)
                    )
                    
                    scored_items.append(item)
                    
                except Exception as e:
                    logger.warning(f"Error embedding item {item.item_id}: {e}")
                    scored_items.append(item)
            
            # Sort by composite score
            scored_items.sort(key=lambda x: x.composite_score, reverse=True)
            
            processing_time = (time.time() - start_time) * 1000
            
            return PipelineStageResult(
                stage=self.stage_type,
                items=scored_items,
                processing_time_ms=processing_time,
                metadata={
                    "query_embedding_generated": True,
                    "avg_semantic_similarity": np.mean([i.semantic_similarity for i in scored_items])
                                      if scored_items else 0
                }
            )
            
        except Exception as e:
            logger.warning(f"Semantic ranking failed: {e}, falling back to keyword")
            return self._keyword_ranking(items, query)
    
    def _keyword_ranking(self, items: List[ContextItem], query: str) -> PipelineStageResult:
        """Fallback keyword-based ranking."""
        query_words = set(query.lower().split())
        
        for item in items:
            item_words = set(item.content.lower().split())
            overlap = len(query_words & item_words)
            item.semantic_similarity = min(1.0, overlap / len(query_words)) if query_words else 0.5
            item.composite_score = (
                0.4 * item.importance_score +
                0.4 * item.semantic_similarity +
                0.2 * self._calculate_freshness_score(item)
            )
        
        items.sort(key=lambda x: x.composite_score, reverse=True)
        
        return PipelineStageResult(
            stage=self.stage_type,
            items=items,
            metadata={"ranking_method": "keyword_fallback"}
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except:
            return 0.0


# =============================================================================
# **LEAN INTEGRATION**: Verification Context Stage
# =============================================================================

class VerificationContextStage:
    """
    **LEAN INTEGRATION**: Verification context stage.
    
    Adds formal verification context to assembled context using Lean.
    """
    
    def __init__(self, config: ContextAssemblerConfig):
        self.config = config
        self.stage_type = ContextAssemblyStage.FORMATTING
        self._lean_client = None
        if LEAN_AVAILABLE:
            try:
                self._lean_client = LeanAideClient()
            except Exception as e:
                logger.warning(f"Failed to initialize Lean client: {e}")
    
    async def add_verification_context(
        self,
        context: AssembledContext,
        query: str
    ) -> AssembledContext:
        """
        Add formal verification context to assembled context.
        
        Args:
            context: The assembled context
            query: Current query string
            
        Returns:
            Context enhanced with verification information
        """
        if not LEAN_AVAILABLE or not self._lean_client:
            context.connection_summary += "\n[Formal verification not available]"
            return context
        
        try:
            # Extract mathematical content from context
            content_to_verify = self._extract_verifiable_content(context)
            
            if content_to_verify:
                # Autoformalize and verify
                formalized = await self._lean_client.autoformalize(content_to_verify)
                result = await self._lean_client.verify(formalized)
                
                # Add verification context
                verification_summary = f"""
\n=== FORMAL VERIFICATION CONTEXT ===
Verifiable content detected in context.
Formalization status: {'Verified' if result.verified else 'Unverified'}
Confidence: {result.confidence if hasattr(result, 'confidence') else 0.0:.2f}
Proof available: {'Yes' if hasattr(result, 'proof_code') and result.proof_code else 'No'}
====================================
"""
                context.connection_summary += verification_summary
                
                # Add verification metadata
                context.pipeline_stats["formal_verification"] = {
                    "verified": result.verified if hasattr(result, 'verified') else False,
                    "confidence": result.confidence if hasattr(result, 'confidence') else 0.0,
                    "method": "lean_autoformalize"
                }
            
            return context
            
        except Exception as e:
            logger.warning(f"Failed to add verification context: {e}")
            context.connection_summary += "\n[Formal verification error: check logs]"
            return context
    
    def _extract_verifiable_content(self, context: AssembledContext) -> str:
        """Extract content that can be formally verified."""
        # Combine core principles for verification
        verifiable_parts = []
        for item in context.core_principles[:3]:  # Top 3 core items
            content = item.content
            # Look for mathematical statements, theorems, proofs
            if any(keyword in content.lower() for keyword in [
                'theorem', 'proof', 'lemma', 'proposition', 'definition',
                'forall', 'exists', 'implies', 'iff', '->', '∀', '∃'
            ]):
                verifiable_parts.append(content)
        
        return "\n".join(verifiable_parts) if verifiable_parts else ""


# =============================================================================
# CONTEXT ASSEMBLY PIPELINE
# =============================================================================

class ContextAssemblyPipeline:
    """
    Four-stage pipeline for context curation.
    
    Pipeline Flow:
    1. Hierarchical Stage: Retrieve by importance level
    2. Graph Stage: Follow relationship paths
    3. Deduplication Stage: Remove near-duplicates
    4. Semantic Stage: Re-rank by query similarity
    
    Each stage filters and enriches the context items,
    producing a curated set ready for LLM consumption.
    """
    
    def __init__(self, config: Optional[ContextAssemblerConfig] = None):
        """
        Initialize the pipeline with all stages.
        
        Args:
            config: Configuration for the pipeline
        """
        self.config = config or ContextAssemblerConfig()
        
        # Initialize stages
        self.stages: List[PipelineStage] = []
        
        if self.config.enable_hierarchical:
            self.stages.append(HierarchicalStage(self.config))
        
        if self.config.enable_graph:
            self.stages.append(GraphStage(self.config))
        
        if self.config.enable_deduplication:
            self.stages.append(DeduplicationStage(self.config))
        
        if self.config.enable_semantic:
            self.stages.append(SemanticStage(self.config))
        
        logger.info(f"Initialized ContextAssemblyPipeline with {len(self.stages)} stages")
    
    def execute(
        self,
        query: str,
        initial_items: Optional[List[ContextItem]] = None,
        query_embedding: Optional[np.ndarray] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> AssemblyPipelineResult:
        """
        Execute the full pipeline.
        
        Args:
            query: Current user query
            initial_items: Optional pre-existing items to process
            query_embedding: Pre-computed query embedding
            conversation_history: Optional conversation history
            
        Returns:
            AssemblyPipelineResult with assembled context
        """
        start_time = time.time()
        
        try:
            # Start with initial items or empty list
            items = initial_items or []
            stage_results = []
            
            # Execute each stage
            for stage in self.stages:
                try:
                    result = stage.process(
                        items=items,
                        query=query,
                        query_embedding=query_embedding,
                        conversation_history=conversation_history
                    )
                    items = result.items
                    stage_results.append(result)
                    
                    logger.debug(f"Stage {result.stage.value}: "
                               f"{len(items)} items, "
                               f"{result.processing_time_ms:.1f}ms")
                    
                except Exception as e:
                    logger.error(f"Stage {stage.stage_type.value} failed: {e}")
                    # Continue with current items
            
            # Build final context
            context = self._build_assembled_context(items, query, stage_results)
            
            total_time = (time.time() - start_time) * 1000
            
            return AssemblyPipelineResult(
                context=context,
                stage_results=stage_results,
                total_processing_time_ms=total_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return AssemblyPipelineResult(
                context=AssembledContext(query=query),
                stage_results=stage_results if 'stage_results' in locals() else [],
                total_processing_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e)
            )
    
    def _build_assembled_context(
        self,
        items: List[ContextItem],
        query: str,
        stage_results: List[PipelineStageResult]
    ) -> AssembledContext:
        """Build the final assembled context from processed items."""
        context = AssembledContext(
            query=query,
            pipeline_stats={
                "stages_executed": len(stage_results),
                "total_items_processed": len(items)
            }
        )
        
        # Calculate token budgets for each section
        core_budget = int(self.config.max_tokens * 0.25)
        relationship_budget = int(self.config.max_tokens * 0.25)
        details_budget = int(self.config.max_tokens * 0.40)
        summary_budget = int(self.config.max_tokens * 0.10)
        
        used_tokens = 0
        
        # Add CORE principles first
        if HIERARCHICAL_AVAILABLE:
            core_items = [i for i in items if i.source_level == MemoryLevel.CORE]
        else:
            core_items = []
        for item in sorted(core_items, key=lambda x: x.composite_score, reverse=True):
            item_tokens = item.estimate_tokens()
            if used_tokens + item_tokens > core_budget:
                break
            context.add_item(item, ContextSectionType.CORE_PRINCIPLES)
            used_tokens += item_tokens
        
        # Add key relationships
        relationship_items = [i for i in items if i.related_item_ids]
        for item in sorted(relationship_items, key=lambda x: x.composite_score, reverse=True):
            item_tokens = item.estimate_tokens()
            if used_tokens + item_tokens > core_budget + relationship_budget:
                break
            context.add_item(item, ContextSectionType.KEY_RELATIONSHIPS)
            used_tokens += item_tokens
        
        # Add recent details
        remaining_budget = self.config.max_tokens - used_tokens - summary_budget
        used_items = set(id(item) for item in context.core_principles + context.key_relationships)
        detail_items = [i for i in items if id(i) not in used_items]
        for item in sorted(detail_items, 
                          key=lambda x: (x.composite_score, x.created_at), 
                          reverse=True):
            item_tokens = item.estimate_tokens()
            if used_tokens + item_tokens > self.config.max_tokens - summary_budget:
                break
            context.add_item(item, ContextSectionType.RECENT_DETAILS)
            used_tokens += item_tokens
        
        # Generate connection summary
        if self.config.connection_summary_enabled:
            context.connection_summary = self._generate_connection_summary(context)
        
        # Calculate overall freshness score
        if items:
            context.freshness_score = np.mean([i.freshness_score for i in items])
        
        return context
    
    def _generate_connection_summary(self, context: AssembledContext) -> str:
        """Generate a summary of how context items connect."""
        parts = []
        
        # Count connections
        total_connections = sum(len(i.related_item_ids) for i in 
                               context.core_principles + context.key_relationships)
        
        if total_connections > 0:
            parts.append(f"Context contains {total_connections} logical connections "
                        f"between {context.item_count} items.")
        
        # Note key themes
        all_tags = []
        for item in context.core_principles + context.key_relationships:
            all_tags.extend(item.tags)
        
        if all_tags:
            from collections import Counter
            top_tags = Counter(all_tags).most_common(3)
            parts.append(f"Key themes: {', '.join(tag for tag, _ in top_tags)}.")
        
        # Time range
        all_times = [i.created_at for i in context.core_principles + 
                    context.key_relationships + context.recent_details if i.created_at]
        if all_times:
            time_range = max(all_times) - min(all_times)
            if time_range.days > 0:
                parts.append(f"Context spans {time_range.days} days of conversation.")
        
        return " ".join(parts) if parts else "No significant connections found."


# =============================================================================
# UNIFIED CONTEXT ASSEMBLER
# =============================================================================

class UnifiedContextAssembler:
    """
    Assembles curated context for LLM by processing through 4 indexes.
    
    This is the main class that orchestrates the context assembly process,
    combining the four indexes to produce a structured "state of the union"
    instead of a raw transcript.
    
    Four Indexes:
    1. Hierarchical Index - Filters by importance level
    2. Graph Index - Preserves logical relationships
    3. Hash Index - Removes near-duplicates
    4. Semantic Index - Ranks by relevance to query
    
    Usage:
        >>> assembler = UnifiedContextAssembler()
        >>> context = assembler.assemble(
        ...     query="How do we handle errors?",
        ...     max_tokens=4000
        ... )
        >>> print(context.to_llm_format())
    
    Attributes:
        config: Configuration for the assembler
        pipeline: ContextAssemblyPipeline instance
        indexes: Dictionary of index instances
        _lock: Thread lock for concurrent access
        _cache: In-memory cache for recent assemblies
    """
    
    def __init__(self, config: Optional[ContextAssemblerConfig] = None):
        """
        Initialize the Unified Context Assembler.
        
        Args:
            config: Configuration object. Uses defaults if not provided.
        """
        self.config = config or ContextAssemblerConfig()
        self.pipeline = ContextAssemblyPipeline(self.config)
        
        # Initialize indexes
        self.indexes: Dict[str, Any] = {}
        self._initialize_indexes()
        
        # Thread safety
        self._lock = threading.RLock() if self.config.thread_safe else contextmanager(lambda: (yield))
        
        # Cache
        self._cache: Dict[str, Tuple[AssembledContext, float]] = {}
        self._cache_lock = threading.RLock()
        
        # Statistics
        self._stats = {
            "total_assemblies": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_processing_time_ms": 0.0
        }
        
        logger.info("UnifiedContextAssembler initialized")
    
    def _initialize_indexes(self) -> None:
        """Initialize all four indexes with graceful fallbacks."""
        # Hierarchical Index
        if self.config.enable_hierarchical and HIERARCHICAL_AVAILABLE:
            try:
                self.indexes["hierarchical"] = HierarchicalIndex(
                    storage_path=self.config.hierarchical_index_path
                )
                logger.info("Hierarchical index initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize hierarchical index: {e}")
        
        # Graph Index
        if self.config.enable_graph and GRAPH_AVAILABLE:
            try:
                self.indexes["graph"] = GraphIndex(
                    db_path=self.config.graph_index_path
                )
                logger.info("Graph index initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize graph index: {e}")
        
        # Hash Index
        if self.config.enable_deduplication and HASH_AVAILABLE:
            try:
                hash_config = HashIndexConfig(db_path=self.config.hash_index_path)
                self.indexes["hash"] = HashIndex(config=hash_config)
                logger.info("Hash index initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize hash index: {e}")
        
        # Semantic Index
        if self.config.enable_semantic and SEMANTIC_AVAILABLE:
            try:
                semantic_config = SemanticIndexConfig(
                    cache_dir=self.config.semantic_cache_dir
                )
                self.indexes["semantic"] = SemanticIndex(semantic_config)
                logger.info("Semantic index initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic index: {e}")
    
    def assemble(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> AssembledContext:
        """
        Assemble curated context for the given query.
        
        This is the main entry point for context assembly. It runs the
        four-stage pipeline and returns a structured context object.
        
        Args:
            query: The current user query
            conversation_history: Optional list of previous conversation turns
            max_tokens: Override max tokens (uses config default if None)
            use_cache: Whether to use caching
            **kwargs: Additional arguments passed to pipeline
            
        Returns:
            AssembledContext ready for LLM consumption
            
        Example:
            >>> context = assembler.assemble(
            ...     query="How do we handle error recovery?",
            ...     max_tokens=4000
            ... )
            >>> print(context.to_llm_format())
        """
        start_time = time.time()
        
        # Override max_tokens if provided
        if max_tokens:
            original_max = self.config.max_tokens
            self.config.max_tokens = max_tokens
        
        try:
            # Check cache
            cache_key = None
            if use_cache and self.config.cache_enabled:
                cache_key = self._get_cache_key(query, conversation_history)
                cached = self._get_from_cache(cache_key)
                if cached:
                    self._stats["cache_hits"] += 1
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return cached
                self._stats["cache_misses"] += 1
            
            # Execute pipeline
            result = self.pipeline.execute(
                query=query,
                conversation_history=conversation_history,
                **kwargs
            )
            
            if not result.success:
                logger.error(f"Pipeline failed: {result.error_message}")
                return result.context
            
            # Update statistics
            self._stats["total_assemblies"] += 1
            processing_time = (time.time() - start_time) * 1000
            
            # Update average processing time
            n = self._stats["total_assemblies"]
            self._stats["avg_processing_time_ms"] = (
                (self._stats["avg_processing_time_ms"] * (n - 1) + processing_time) / n
            )
            
            # Cache result
            if cache_key and self.config.cache_enabled:
                self._add_to_cache(cache_key, result.context)
            
            return result.context
            
        finally:
            # Restore original max_tokens
            if max_tokens:
                self.config.max_tokens = original_max
    
    def assemble_with_formatting(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        include_metadata: bool = False,
        **kwargs
    ) -> str:
        """
        Assemble context and return formatted string for LLM.
        
        Convenience method that calls assemble() and formats the result.
        
        Args:
            query: The current user query
            conversation_history: Optional conversation history
            max_tokens: Override max tokens
            include_metadata: Include metadata in output
            **kwargs: Additional arguments
            
        Returns:
            Formatted context string
        """
        context = self.assemble(
            query=query,
            conversation_history=conversation_history,
            max_tokens=max_tokens,
            **kwargs
        )
        return context.to_llm_format(include_metadata=include_metadata)
    
    def get_index(self, index_name: str) -> Optional[Any]:
        """Get an index by name."""
        return self.indexes.get(index_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get assembler statistics."""
        return {
            **self._stats,
            "indexes_available": list(self.indexes.keys()),
            "cache_size": len(self._cache),
            "config": {
                "max_tokens": self.config.max_tokens,
                "stages_enabled": {
                    "hierarchical": self.config.enable_hierarchical,
                    "graph": self.config.enable_graph,
                    "deduplication": self.config.enable_deduplication,
                    "semantic": self.config.enable_semantic,
                }
            }
        }
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        with self._cache_lock:
            self._cache.clear()
        logger.info("Context assembler cache cleared")
    
    def _get_cache_key(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Generate cache key for query."""
        import hashlib
        key_data = query.lower().strip()
        if conversation_history:
            key_data += json.dumps(conversation_history[-3:], sort_keys=True)  # Last 3 turns
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def _get_from_cache(self, cache_key: str) -> Optional[AssembledContext]:
        """Get cached result if valid."""
        with self._cache_lock:
            if cache_key in self._cache:
                context, timestamp = self._cache[cache_key]
                age = time.time() - timestamp
                if age < self.config.cache_ttl_seconds:
                    return context
                else:
                    del self._cache[cache_key]
            return None
    
    def _add_to_cache(self, cache_key: str, context: AssembledContext) -> None:
        """Add result to cache."""
        with self._cache_lock:
            self._cache[cache_key] = (context, time.time())
            # Simple LRU: limit cache size
            if len(self._cache) > 100:
                oldest = next(iter(self._cache))
                del self._cache[oldest]


# =============================================================================
# CONTEXT ROT PREVENTER
# =============================================================================

class ContextRotPreventer:
    """
    High-level interface specifically designed to prevent context rot
    in long conversations (50,000+ words).
    
    Context rot occurs when LLMs lose track of important information
    from earlier in long conversations. This class provides:
    
    - Automatic context assembly before each LLM call
    - Freshness scoring to detect stale context
    - Alerting when context needs refresh
    - Session-based context management
    - Conversation summarization triggers
    
    Usage:
        >>> preventer = ContextRotPreventer()
        >>> preventer.start_session("session_123")
        >>> 
        >>> # Before each LLM call
        >>> context = preventer.prepare_context(
        ...     query="How do we handle this?",
        ...     conversation_so_far=messages
        ... )
        >>> 
        >>> # Check if we need a summary
        >>> if preventer.needs_summary():
        ...     summary = preventer.generate_summary()
    """
    
    def __init__(self, config: Optional[ContextAssemblerConfig] = None):
        """
        Initialize the Context Rot Preventer.
        
        Args:
            config: Configuration object
        """
        self.config = config or ContextAssemblerConfig()
        self.assembler = UnifiedContextAssembler(self.config)
        
        # Session management
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None
        self.session_turn_count: int = 0
        self.session_total_tokens: int = 0
        
        # Conversation tracking
        self.conversation_history: List[Dict[str, Any]] = []
        self.assembled_contexts: List[AssembledContext] = []
        
        # Rot detection
        self.freshness_history: List[float] = []
        self.rot_threshold: float = 0.3  # Alert if freshness drops below this
        
        logger.info("ContextRotPreventer initialized")
    
    def start_session(self, session_id: str) -> None:
        """Start a new conversation session."""
        self.current_session_id = session_id
        self.session_start_time = datetime.now()
        self.session_turn_count = 0
        self.session_total_tokens = 0
        self.conversation_history.clear()
        self.assembled_contexts.clear()
        self.freshness_history.clear()
        logger.info(f"Started context session: {session_id}")
    
    def prepare_context(
        self,
        query: str,
        conversation_so_far: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None
    ) -> AssembledContext:
        """
        Prepare curated context for the next LLM call.
        
        This method:
        1. Updates conversation history
        2. Runs the context assembly pipeline
        3. Tracks freshness scores
        4. Detects potential context rot
        
        Args:
            query: The upcoming user query
            conversation_so_far: Previous conversation turns
            max_tokens: Maximum tokens for context
            
        Returns:
            AssembledContext ready for LLM
        """
        # Update tracking
        self.session_turn_count += 1
        
        if conversation_so_far:
            self.conversation_history = conversation_so_far
        
        # Add query to history
        self.conversation_history.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Assemble context
        context = self.assembler.assemble(
            query=query,
            conversation_history=self.conversation_history,
            max_tokens=max_tokens
        )
        
        # Track freshness
        self.freshness_history.append(context.freshness_score)
        self.assembled_contexts.append(context)
        self.session_total_tokens += context.total_tokens
        
        # Check for rot
        if self._detect_rot(context):
            logger.warning("Context rot detected! Consider generating summary.")
        
        return context
    
    def prepare_context_string(
        self,
        query: str,
        conversation_so_far: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        include_metadata: bool = False
    ) -> str:
        """
        Prepare context and return as formatted string.
        
        Convenience method that calls prepare_context() and formats output.
        """
        context = self.prepare_context(query, conversation_so_far, max_tokens)
        return context.to_llm_format(include_metadata=include_metadata)
    
    def needs_summary(self) -> bool:
        """
        Check if conversation needs summarization.
        
        Triggers:
        - Turn count exceeds threshold
        - Freshness score drops below threshold
        - Total conversation tokens exceed limit
        """
        if self.session_turn_count > 20:
            return True
        
        if self.freshness_history and self.freshness_history[-1] < self.rot_threshold:
            return True
        
        if self.session_total_tokens > 50000:
            return True
        
        return False
    
    def generate_summary(self, max_tokens: int = 2000) -> str:
        """
        Generate a summary of the conversation for context reset.
        
        This creates a condensed version of key points that can be
        used to start a fresh conversation context.
        """
        # Get all core principles from assembled contexts
        all_core = []
        for ctx in self.assembled_contexts:
            all_core.extend(ctx.core_principles)
        
        # Get all key relationships
        all_relationships = []
        for ctx in self.assembled_contexts:
            all_relationships.extend(ctx.key_relationships)
        
        # Build summary
        summary_parts = [
            "=== CONVERSATION SUMMARY ===",
            f"Session: {self.current_session_id}",
            f"Turns: {self.session_turn_count}, "
            f"Duration: {datetime.now() - self.session_start_time if self.session_start_time else 'N/A'}",
            ""
        ]
        
        # Add core principles
        if all_core:
            summary_parts.append("Key Principles Established:")
            unique_principles = {p.content: p for p in all_core}
            for principle in list(unique_principles.values())[:10]:
                summary_parts.append(f"  * {principle.content}")
            summary_parts.append("")
        
        # Add relationships
        if all_relationships:
            summary_parts.append("Key Connections Made:")
            unique_rels = {r.content: r for r in all_relationships}
            for rel in list(unique_rels.values())[:10]:
                summary_parts.append(f"  * {rel.content}")
            summary_parts.append("")
        
        # Add session stats
        avg_freshness = np.mean(self.freshness_history) if self.freshness_history else 0
        summary_parts.append(f"Context Freshness: {avg_freshness:.2f}")
        
        return "\n".join(summary_parts)
    
    def reset_with_summary(self) -> str:
        """
        Reset the conversation with a summary.
        
        Returns:
            Summary string to use as new conversation context
        """
        summary = self.generate_summary()
        
        # Reset tracking but keep session
        self.session_turn_count = 0
        self.session_total_tokens = 0
        self.conversation_history = []
        self.assembled_contexts = []
        self.freshness_history = []
        
        # Add summary as first context
        self.conversation_history.append({
            "role": "system",
            "content": summary,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info("Conversation reset with summary")
        return summary
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        return {
            "session_id": self.current_session_id,
            "session_start": self.session_start_time.isoformat() if self.session_start_time else None,
            "turn_count": self.session_turn_count,
            "total_tokens": self.session_total_tokens,
            "avg_freshness": np.mean(self.freshness_history) if self.freshness_history else 0,
            "current_freshness": self.freshness_history[-1] if self.freshness_history else 0,
            "needs_summary": self.needs_summary(),
            "conversation_length": len(self.conversation_history),
        }
    
    def _detect_rot(self, context: AssembledContext) -> bool:
        """Detect if context rot is occurring."""
        # Check freshness score
        if context.freshness_score < self.rot_threshold:
            return True
        
        # Check for declining freshness trend
        if len(self.freshness_history) >= 5:
            recent = self.freshness_history[-5:]
            # Check if consistently declining
            if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                if recent[-1] < 0.5:
                    return True
        
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def assemble_context(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    config: Optional[ContextAssemblerConfig] = None,
    return_format: str = "object"
) -> Union[AssembledContext, str, Dict[str, Any]]:
    """
    Main entry point: takes query + conversation history, returns curated context.
    
    This is the primary function for assembling context. It runs the full
    four-stage pipeline and returns the result in the requested format.
    
    Args:
        query: Current user query
        conversation_history: Previous conversation turns
        max_tokens: Maximum tokens for assembled context
        config: Optional configuration
        return_format: "object", "string", or "dict"
        
    Returns:
        AssembledContext in requested format
        
    Example:
        >>> context = assemble_context(
        ...     query="How do we handle error recovery?",
        ...     conversation_history=[{"role": "user", "content": "..."}],
        ...     max_tokens=4000,
        ...     return_format="string"
        ... )
    """
    config = config or ContextAssemblerConfig()
    config.max_tokens = max_tokens
    
    assembler = UnifiedContextAssembler(config)
    context = assembler.assemble(
        query=query,
        conversation_history=conversation_history
    )
    
    if return_format == "string":
        return context.to_llm_format()
    elif return_format == "dict":
        return context.to_dict()
    else:
        return context


def create_context_preventer(
    session_id: Optional[str] = None,
    config: Optional[ContextAssemblerConfig] = None
) -> ContextRotPreventer:
    """
    Create a ContextRotPreventer with optional session.
    
    Convenience function for creating a preventer and optionally
    starting a session in one call.
    
    Args:
        session_id: Optional session ID to start
        config: Optional configuration
        
    Returns:
        Configured ContextRotPreventer
    """
    preventer = ContextRotPreventer(config)
    if session_id:
        preventer.start_session(session_id)
    return preventer


def quick_assemble(
    query: str,
    memories: List[str],
    max_tokens: int = 2000
) -> str:
    """
    Quick assembly from a list of memory strings.
    
    Simplified interface that doesn't require index setup.
    Useful for quick prototyping or testing.
    
    Args:
        query: Current query
        memories: List of memory strings
        max_tokens: Maximum tokens
        
    Returns:
        Formatted context string
    """
    config = ContextAssemblerConfig(
        max_tokens=max_tokens,
        enable_hierarchical=False,  # No indexes needed
        enable_graph=False,
        enable_deduplication=True,
        enable_semantic=False
    )
    
    # Create items from memories
    items = [
        ContextItem(
            item_id=f"mem_{i}",
            content=mem,
            importance_score=0.5,
            composite_score=0.5
        )
        for i, mem in enumerate(memories)
    ]
    
    # Simple deduplication
    dedup_stage = DeduplicationStage(config)
    result = dedup_stage.process(items, query)
    
    # Build simple context
    context = AssembledContext(query=query)
    for item in result.items:
        context.add_item(item, ContextSectionType.RECENT_DETAILS)
    
    return context.to_llm_format()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Demonstration
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Knowledge Context Assembler - Demonstration")
    print("=" * 60)
    
    # Example 1: Quick assembly
    print("\n1. Quick Assembly Example:")
    memories = [
        "System should use microservices architecture for scalability",
        "Error handling must be implemented at service boundaries",
        "Logging should be centralized for observability",
        "System should use microservices for better scaling",  # Near-duplicate
        "Database connections should be pooled",
    ]
    
    result = quick_assemble(
        query="How do we handle errors in our system?",
        memories=memories,
        max_tokens=500
    )
    print(result)
    
    # Example 2: Full assembler
    print("\n2. Full Assembler Example:")
    config = ContextAssemblerConfig(
        max_tokens=1000,
        enable_hierarchical=False,  # Would need DB setup
        enable_graph=False,
        enable_deduplication=True,
        enable_semantic=False
    )
    
    context = assemble_context(
        query="How should we architect the system?",
        conversation_history=[
            {"role": "user", "content": "We need a scalable architecture"},
            {"role": "assistant", "content": "Microservices would be best"}
        ],
        max_tokens=800,
        config=config,
        return_format="string"
    )
    print(context)
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
