"""
Hybrid Retrieval System for Knowledge Management

This module implements a hybrid retrieval system that uses 4 strategies in parallel
to retrieve the most relevant memories for a given query. The system returns only
the top N (default 10-20) most relevant memories, keeping context size manageable.

Key Features:
- 4 parallel retrieval strategies (hierarchical, graph, semantic, recency)
- Combined scoring with configurable weights
- Diversity enforcement to avoid redundant results
- Thread-safe execution with connection pooling
- Result caching for frequently accessed queries
- Performance metrics tracking
"""

import asyncio
import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import heapq

# Configure logging
logger = logging.getLogger(__name__)


class RetrievalStrategyType(Enum):
    """Types of retrieval strategies."""
    HIERARCHICAL = "hierarchical"
    GRAPH = "graph"
    SEMANTIC = "semantic"
    RECENCY = "recency"


@dataclass
class Memory:
    """Base memory representation."""
    id: str
    content: str
    importance: int = 5  # 1-10 scale
    timestamp: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None  # For semantic search
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Memory):
            return False
        return self.id == other.id


@dataclass
class RetrievedMemory:
    """
    Single retrieved memory with scoring information.
    
    Attributes:
        memory: The underlying memory object
        strategy_scores: Individual scores from each strategy
        combined_score: Final combined relevance score
        retrieval_reason: Explanation of why this was selected
        retrieval_latency_ms: Time taken to retrieve this memory
    """
    memory: Memory
    strategy_scores: Dict[RetrievalStrategyType, float] = field(default_factory=dict)
    combined_score: float = 0.0
    retrieval_reason: str = ""
    retrieval_latency_ms: float = 0.0
    
    def __hash__(self):
        return hash(self.memory.id)
    
    def __eq__(self, other):
        if not isinstance(other, RetrievedMemory):
            return False
        return self.memory.id == other.memory.id
    
    def __lt__(self, other):
        """For priority queue ordering (higher score = higher priority)."""
        return self.combined_score > other.combined_score


@dataclass
class RetrievalWeights:
    """Configurable weights for each retrieval strategy."""
    hierarchical: float = 0.25
    graph: float = 0.25
    semantic: float = 0.30
    recency: float = 0.20
    
    def validate(self) -> bool:
        """Validate that weights sum to approximately 1.0."""
        total = self.hierarchical + self.graph + self.semantic + self.recency
        return 0.99 <= total <= 1.01
    
    def normalize(self) -> "RetrievalWeights":
        """Normalize weights to sum to 1.0."""
        total = self.hierarchical + self.graph + self.semantic + self.recency
        if total == 0:
            return RetrievalWeights(0.25, 0.25, 0.30, 0.20)
        return RetrievalWeights(
            hierarchical=self.hierarchical / total,
            graph=self.graph / total,
            semantic=self.semantic / total,
            recency=self.recency / total
        )


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""
    
    def __init__(self, strategy_type: RetrievalStrategyType):
        self.strategy_type = strategy_type
        self._lock = threading.RLock()
        self._metrics = {
            "total_queries": 0,
            "total_time_ms": 0.0,
            "avg_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        context: Dict[str, Any],
        limit: int
    ) -> List[Tuple[Memory, float]]:
        """
        Retrieve memories based on this strategy.
        
        Args:
            query: The search query
            context: Additional context (current conversation, user preferences, etc.)
            limit: Maximum number of results to return
            
        Returns:
            List of (memory, score) tuples, score in [0, 1]
        """
        raise NotImplementedError
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this strategy."""
        with self._lock:
            return self._metrics.copy()
    
    def _update_metrics(self, query_time_ms: float):
        """Update performance metrics."""
        with self._lock:
            self._metrics["total_queries"] += 1
            self._metrics["total_time_ms"] += query_time_ms
            self._metrics["avg_time_ms"] = (
                self._metrics["total_time_ms"] / self._metrics["total_queries"]
            )


class HierarchicalRetrieval(RetrievalStrategy):
    """
    Retrieve memories based on importance hierarchy.
    
    Higher importance memories (7-10) are critical context.
    Medium importance (4-6) provide supporting information.
    Lower importance (1-3) add detail when needed.
    """
    
    def __init__(self):
        super().__init__(RetrievalStrategyType.HIERARCHICAL)
        self._importance_index: Dict[int, Set[str]] = defaultdict(set)
        self._memories: Dict[str, Memory] = {}
    
    def index_memory(self, memory: Memory):
        """Add a memory to the importance index."""
        with self._lock:
            self._importance_index[memory.importance].add(memory.id)
            self._memories[memory.id] = memory
    
    def remove_memory(self, memory_id: str):
        """Remove a memory from the index."""
        with self._lock:
            if memory_id in self._memories:
                memory = self._memories[memory_id]
                self._importance_index[memory.importance].discard(memory_id)
                del self._memories[memory_id]
    
    def retrieve(
        self, 
        query: str, 
        context: Dict[str, Any],
        limit: int
    ) -> List[Tuple[Memory, float]]:
        """
        Retrieve memories by importance hierarchy.
        
        Strategy:
        1. Prioritize high importance (8-10)
        2. Include medium importance (5-7) if query suggests depth
        3. Add low importance (1-4) only if explicitly requested
        """
        start_time = time.time()
        results = []
        
        with self._lock:
            # Get query depth hint from context
            query_depth = context.get("query_depth", "standard")
            
            # High importance always included (score 0.9-1.0)
            for importance in range(10, 7, -1):
                for mem_id in self._importance_index[importance]:
                    memory = self._memories[mem_id]
                    score = importance / 10.0
                    results.append((memory, score))
            
            # Medium importance for detailed queries
            if query_depth in ["detailed", "comprehensive"]:
                for importance in range(7, 4, -1):
                    for mem_id in self._importance_index[importance]:
                        memory = self._memories[mem_id]
                        score = (importance / 10.0) * 0.8  # Slightly lower weight
                        results.append((memory, score))
            
            # Sort by score and limit
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:limit]
        
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_metrics(elapsed_ms)
        
        return results


class GraphRetrieval(RetrievalStrategy):
    """
    Retrieve memories based on graph relationships.
    
    Traverses relationship edges to find connected memories.
    Supports multiple relationship types: related_to, depends_on, 
    part_of, caused_by, etc.
    """
    
    def __init__(self, max_depth: int = 2):
        super().__init__(RetrievalStrategyType.GRAPH)
        self.max_depth = max_depth
        self._graph: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._memories: Dict[str, Memory] = {}
        self._node_embeddings: Dict[str, List[float]] = {}
    
    def add_relationship(
        self, 
        from_id: str, 
        to_id: str, 
        relation_type: str = "related_to",
        bidirectional: bool = True
    ):
        """Add a relationship between two memories."""
        with self._lock:
            self._graph[from_id][relation_type].append(to_id)
            if bidirectional:
                reverse_type = self._get_reverse_relation(relation_type)
                self._graph[to_id][reverse_type].append(from_id)
    
    def _get_reverse_relation(self, relation_type: str) -> str:
        """Get the reverse relationship type."""
        reversals = {
            "depends_on": "required_by",
            "required_by": "depends_on",
            "part_of": "contains",
            "contains": "part_of",
            "caused_by": "causes",
            "causes": "caused_by",
            "precedes": "follows",
            "follows": "precedes",
            "related_to": "related_to"
        }
        return reversals.get(relation_type, "related_to")
    
    def index_memory(self, memory: Memory):
        """Index a memory node."""
        with self._lock:
            self._memories[memory.id] = memory
    
    def retrieve(
        self, 
        query: str, 
        context: Dict[str, Any],
        limit: int
    ) -> List[Tuple[Memory, float]]:
        """
        Retrieve memories through graph traversal.
        
        Starts from seed nodes (recently accessed or explicitly mentioned)
        and traverses relationships up to max_depth.
        """
        start_time = time.time()
        results = []
        visited = set()
        
        with self._lock:
            # Get seed nodes from context
            seed_ids = context.get("active_memory_ids", [])
            if not seed_ids and self._memories:
                # Use most recently accessed as fallback
                seed_ids = [
                    mid for mid, mem in sorted(
                        self._memories.items(),
                        key=lambda x: x[1].last_accessed,
                        reverse=True
                    )[:3]
                ]
            
            # BFS traversal with decaying scores
            queue = [(seed_id, 0, 1.0) for seed_id in seed_ids]  # (node_id, depth, score)
            
            while queue and len(results) < limit * 2:  # Get extra for scoring
                node_id, depth, base_score = queue.pop(0)
                
                if node_id in visited or depth > self.max_depth:
                    continue
                visited.add(node_id)
                
                if node_id in self._memories:
                    memory = self._memories[node_id]
                    # Score decays with depth
                    score = base_score * (0.7 ** depth)
                    results.append((memory, score))
                
                # Add neighbors to queue
                if depth < self.max_depth:
                    for rel_type, neighbors in self._graph.get(node_id, {}).items():
                        rel_weight = self._get_relation_weight(rel_type)
                        for neighbor_id in neighbors:
                            if neighbor_id not in visited:
                                queue.append(
                                    (neighbor_id, depth + 1, base_score * rel_weight)
                                )
        
        # Sort by score and deduplicate
        results.sort(key=lambda x: x[1], reverse=True)
        seen_ids = set()
        unique_results = []
        for memory, score in results:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                unique_results.append((memory, score))
        
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_metrics(elapsed_ms)
        
        return unique_results[:limit]
    
    def _get_relation_weight(self, relation_type: str) -> float:
        """Get weight for different relationship types."""
        weights = {
            "related_to": 0.9,
            "depends_on": 0.95,
            "required_by": 0.95,
            "part_of": 0.85,
            "contains": 0.85,
            "caused_by": 0.9,
            "causes": 0.9,
            "precedes": 0.8,
            "follows": 0.8
        }
        return weights.get(relation_type, 0.8)


class SemanticRetrieval(RetrievalStrategy):
    """
    Retrieve memories based on vector similarity.
    
    Uses cosine similarity between query embedding and memory embeddings.
    Supports approximate nearest neighbor search for scalability.
    """
    
    def __init__(self, embedding_dim: int = 384):
        super().__init__(RetrievalStrategyType.SEMANTIC)
        self.embedding_dim = embedding_dim
        self._vectors: Dict[str, List[float]] = {}
        self._memories: Dict[str, Memory] = {}
        self._cache: Dict[str, List[Tuple[str, float]]] = {}
        self._cache_lock = threading.RLock()
    
    def index_memory(self, memory: Memory, vector: Optional[List[float]] = None):
        """Index a memory with its vector embedding."""
        with self._lock:
            self._memories[memory.id] = memory
            if vector:
                self._vectors[memory.id] = vector
            elif memory.vector:
                self._vectors[memory.id] = memory.vector
    
    def _compute_similarity(
        self, 
        vec1: List[float], 
        vec2: List[float]
    ) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for query string.
        
        In production, this would call an embedding service.
        For now, we return a simple hash-based embedding.
        """
        # Simple deterministic embedding for demonstration
        # In production: return embedding_model.encode(query)
        hash_val = hashlib.sha256(query.encode()).digest()
        embedding = []
        for i in range(self.embedding_dim):
            byte_val = hash_val[i % len(hash_val)]
            embedding.append((byte_val / 255.0) * 2 - 1)  # Normalize to [-1, 1]
        return embedding
    
    def retrieve(
        self, 
        query: str, 
        context: Dict[str, Any],
        limit: int
    ) -> List[Tuple[Memory, float]]:
        """
        Retrieve memories by semantic similarity.
        
        Computes cosine similarity between query embedding and
        all memory embeddings, returns top matches.
        """
        start_time = time.time()
        
        # Check cache
        cache_key = hashlib.sha256(query.encode()).hexdigest()[:16]
        with self._cache_lock:
            if cache_key in self._cache:
                cached_results = self._cache[cache_key]
                results = []
                for mem_id, score in cached_results[:limit]:
                    if mem_id in self._memories:
                        results.append((self._memories[mem_id], score))
                self._metrics["cache_hits"] += 1
                return results
            self._metrics["cache_misses"] += 1
        
        query_embedding = self._get_query_embedding(query)
        similarities = []
        
        with self._lock:
            for mem_id, mem_vector in self._vectors.items():
                similarity = self._compute_similarity(query_embedding, mem_vector)
                # Normalize to [0, 1]
                similarity = (similarity + 1) / 2
                similarities.append((mem_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Build results
            results = []
            for mem_id, score in similarities[:limit]:
                if mem_id in self._memories:
                    results.append((self._memories[mem_id], score))
            
            # Cache results
            with self._cache_lock:
                self._cache[cache_key] = similarities[:limit * 2]
        
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_metrics(elapsed_ms)
        
        return results


class RecencyRetrieval(RetrievalStrategy):
    """
    Retrieve memories based on recency and access patterns.
    
    Combines:
    - Time since creation
    - Time since last access
    - Access frequency
    - Temporal relevance to query
    """
    
    def __init__(self, decay_half_life_hours: float = 24.0):
        super().__init__(RetrievalStrategyType.RECENCY)
        self.decay_half_life = decay_half_life_hours * 3600  # Convert to seconds
        self._memories: Dict[str, Memory] = {}
        self._temporal_index: List[Tuple[float, str]] = []  # (timestamp, id)
    
    def index_memory(self, memory: Memory):
        """Index a memory by timestamp."""
        with self._lock:
            self._memories[memory.id] = memory
            self._temporal_index.append((memory.timestamp, memory.id))
            # Keep sorted
            self._temporal_index.sort(reverse=True)
    
    def update_access(self, memory_id: str):
        """Update last accessed time for a memory."""
        with self._lock:
            if memory_id in self._memories:
                self._memories[memory_id].last_accessed = time.time()
                self._memories[memory_id].access_count += 1
    
    def retrieve(
        self, 
        query: str, 
        context: Dict[str, Any],
        limit: int
    ) -> List[Tuple[Memory, float]]:
        """
        Retrieve memories based on recency.
        
        Score = recency_score * 0.5 + frequency_score * 0.3 + relevance_score * 0.2
        """
        start_time = time.time()
        results = []
        current_time = time.time()
        
        # Get temporal context
        query_timestamp = context.get("reference_timestamp", current_time)
        temporal_window = context.get("temporal_window_hours", 168)  # 1 week default
        
        with self._lock:
            for memory in self._memories.values():
                # Recency score (exponential decay)
                time_since_creation = query_timestamp - memory.timestamp
                recency_score = self._compute_decay_score(time_since_creation)
                
                # Last access recency
                time_since_access = query_timestamp - memory.last_accessed
                access_recency = self._compute_decay_score(time_since_access)
                
                # Frequency score (log scale to prevent dominance)
                frequency_score = min(
                    1.0, 
                    (memory.access_count / 10.0) if memory.access_count > 0 else 0.1
                )
                
                # Temporal relevance (within query window)
                in_window = time_since_creation < (temporal_window * 3600)
                window_boost = 1.3 if in_window else 1.0
                
                # Combined score
                score = (
                    recency_score * 0.35 +
                    access_recency * 0.25 +
                    frequency_score * 0.25
                ) * window_boost
                
                if score > 0.1:  # Filter very low scores
                    results.append((memory, min(1.0, score)))
            
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
        
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_metrics(elapsed_ms)
        
        return results[:limit]
    
    def _compute_decay_score(self, time_delta: float) -> float:
        """Compute exponential decay score."""
        if time_delta < 0:
            return 1.0
        return 0.5 ** (time_delta / self.decay_half_life)


class ScoreCombiner:
    """
    Combines scores from multiple retrieval strategies.
    
    Features:
    - Weighted average combination
    - Boost for multi-strategy hits
    - Diversity bonus to avoid similar results
    - Configurable normalization
    """
    
    def __init__(
        self,
        weights: Optional[RetrievalWeights] = None,
        multi_strategy_boost: float = 0.15,
        diversity_weight: float = 0.1
    ):
        self.weights = weights or RetrievalWeights()
        self.multi_strategy_boost = multi_strategy_boost
        self.diversity_weight = diversity_weight
    
    def combine(
        self,
        strategy_results: Dict[RetrievalStrategyType, List[Tuple[Memory, float]]],
        limit: int
    ) -> List[RetrievedMemory]:
        """
        Combine results from multiple strategies.
        
        Args:
            strategy_results: Map from strategy type to list of (memory, score)
            limit: Maximum number of results to return
            
        Returns:
            List of RetrievedMemory objects with combined scores
        """
        # Collect all memories and their scores
        memory_scores: Dict[str, Dict[str, Any]] = {}
        
        for strategy_type, results in strategy_results.items():
            for memory, score in results:
                if memory.id not in memory_scores:
                    memory_scores[memory.id] = {
                        "memory": memory,
                        "scores": {},
                        "strategies": set()
                    }
                memory_scores[memory.id]["scores"][strategy_type] = score
                memory_scores[memory.id]["strategies"].add(strategy_type)
        
        # Calculate combined scores
        retrieved_memories = []
        for mem_id, data in memory_scores.items():
            strategy_scores = data["scores"]
            num_strategies = len(data["strategies"])
            
            # Weighted average
            weighted_sum = 0.0
            weight_total = 0.0
            
            for strategy_type, score in strategy_scores.items():
                weight = getattr(self.weights, strategy_type.value)
                weighted_sum += score * weight
                weight_total += weight
            
            base_score = weighted_sum / weight_total if weight_total > 0 else 0.0
            
            # Multi-strategy boost
            strategy_boost = self.multi_strategy_boost * (num_strategies - 1)
            
            combined_score = min(1.0, base_score + strategy_boost)
            
            # Build retrieval reason
            reasons = []
            for strategy_type in sorted(data["strategies"], key=lambda x: x.value):
                score = strategy_scores[strategy_type]
                reasons.append(f"{strategy_type.value}({score:.2f})")
            
            retrieved_memories.append(RetrievedMemory(
                memory=data["memory"],
                strategy_scores=strategy_scores,
                combined_score=combined_score,
                retrieval_reason=f"Combined: {', '.join(reasons)}"
            ))
        
        # Apply diversity re-ranking
        retrieved_memories = self._apply_diversity(retrieved_memories, limit)
        
        # Sort by final combined score
        retrieved_memories.sort(key=lambda x: x.combined_score, reverse=True)
        
        return retrieved_memories[:limit]
    
    def _apply_diversity(
        self, 
        memories: List[RetrievedMemory], 
        limit: int
    ) -> List[RetrievedMemory]:
        """
        Apply diversity re-ranking using Maximal Marginal Relevance.
        
        Selects items that are both relevant and diverse from already selected items.
        """
        if len(memories) <= limit:
            return memories
        
        selected = []
        remaining = memories.copy()
        
        while len(selected) < limit and remaining:
            if not selected:
                # First item: pick highest relevance
                best = max(remaining, key=lambda x: x.combined_score)
            else:
                # MMR: lambda * Relevance - (1-lambda) * max_similarity(selected)
                best_mmr_score = -float('inf')
                best = None
                
                for candidate in remaining:
                    relevance = candidate.combined_score
                    max_sim = max(
                        self._compute_similarity(candidate, s) 
                        for s in selected
                    )
                    mmr_score = (
                        0.7 * relevance - 
                        0.3 * max_sim * self.diversity_weight
                    )
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best = candidate
            
            selected.append(best)
            remaining.remove(best)
        
        return selected
    
    def _compute_similarity(
        self, 
        m1: RetrievedMemory, 
        m2: RetrievedMemory
    ) -> float:
        """Compute similarity between two memories for diversity calculation."""
        # Use tag overlap and content similarity
        tags1 = m1.memory.tags
        tags2 = m2.memory.tags
        
        if not tags1 or not tags2:
            return 0.0
        
        intersection = len(tags1 & tags2)
        union = len(tags1 | tags2)
        
        return intersection / union if union > 0 else 0.0


class RetrievalCache:
    """LRU cache for retrieval results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: Dict[str, Tuple[List[RetrievedMemory], float]] = {}
        self._lock = threading.RLock()
    
    def get(self, query: str, limit: int) -> Optional[List[RetrievedMemory]]:
        """Get cached results if available and not expired."""
        cache_key = self._make_key(query, limit)
        with self._lock:
            if cache_key in self._cache:
                results, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self.ttl:
                    return results
                else:
                    del self._cache[cache_key]
        return None
    
    def put(self, query: str, limit: int, results: List[RetrievedMemory]):
        """Cache retrieval results."""
        cache_key = self._make_key(query, limit)
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(
                    self._cache.keys(), 
                    key=lambda k: self._cache[k][1]
                )
                del self._cache[oldest_key]
            
            self._cache[cache_key] = (results, time.time())
    
    def invalidate(self, memory_id: str):
        """Invalidate cache entries containing a specific memory."""
        with self._lock:
            keys_to_remove = []
            for key, (results, _) in self._cache.items():
                if any(r.memory.id == memory_id for r in results):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self._cache[key]
    
    def _make_key(self, query: str, limit: int) -> str:
        """Create cache key from query and limit."""
        return hashlib.sha256(f"{query}:{limit}".encode()).hexdigest()[:32]


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval operations."""
    total_queries: int = 0
    total_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_query_time_ms: float = 0.0
    strategy_times: Dict[RetrievalStrategyType, float] = field(default_factory=dict)
    
    def record_query(self, duration_ms: float, cache_hit: bool):
        """Record a query metric."""
        self.total_queries += 1
        self.total_time_ms += duration_ms
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        self.avg_query_time_ms = self.total_time_ms / self.total_queries
    
    def record_strategy_time(
        self, 
        strategy: RetrievalStrategyType, 
        duration_ms: float
    ):
        """Record time taken by a specific strategy."""
        if strategy not in self.strategy_times:
            self.strategy_times[strategy] = 0.0
        self.strategy_times[strategy] += duration_ms


class HybridRetriever:
    """
    Hybrid retrieval using 4 strategies in parallel.
    
    This class orchestrates the four retrieval strategies:
    1. Hierarchical retrieval (by importance level)
    2. Graph traversal (relationship-based)
    3. Semantic search (vector similarity)
    4. Recency/Temporal (recently accessed)
    
    Returns top N (default 10-20) most relevant memories.
    Only these go into context (~5KB), rest stays in storage.
    
    Thread Safety:
        All operations are thread-safe using fine-grained locking.
        Parallel strategy execution uses ThreadPoolExecutor.
    
    Example:
        >>> retriever = HybridRetriever()
        >>> retriever.index_memory(memory)
        >>> results = retriever.retrieve("query about topic", limit=15)
        >>> for r in results:
        ...     print(f"{r.memory.content}: {r.combined_score:.3f}")
    """
    
    def __init__(
        self,
        default_limit: int = 15,
        max_workers: int = 4,
        weights: Optional[RetrievalWeights] = None,
        cache_size: int = 1000,
        cache_ttl_seconds: float = 300,
        enable_metrics: bool = True
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            default_limit: Default number of memories to retrieve
            max_workers: Maximum number of threads for parallel execution
            weights: Strategy weights (uses defaults if not provided)
            cache_size: Maximum number of cached queries
            cache_ttl_seconds: Cache entry time-to-live
            enable_metrics: Whether to track performance metrics
        """
        self.default_limit = default_limit
        self.max_workers = max_workers
        self.weights = weights or RetrievalWeights()
        self.enable_metrics = enable_metrics
        
        # Initialize strategies
        self._hierarchical = HierarchicalRetrieval()
        self._graph = GraphRetrieval()
        self._semantic = SemanticRetrieval()
        self._recency = RecencyRetrieval()
        
        self._strategies: Dict[RetrievalStrategyType, RetrievalStrategy] = {
            RetrievalStrategyType.HIERARCHICAL: self._hierarchical,
            RetrievalStrategyType.GRAPH: self._graph,
            RetrievalStrategyType.SEMANTIC: self._semantic,
            RetrievalStrategyType.RECENCY: self._recency
        }
        
        # Score combiner
        self._combiner = ScoreCombiner(self.weights)
        
        # Cache
        self._cache = RetrievalCache(cache_size, cache_ttl_seconds)
        
        # Metrics
        self._metrics = RetrievalMetrics()
        self._metrics_lock = threading.RLock()
        
        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(
            f"HybridRetriever initialized with limit={default_limit}, "
            f"max_workers={max_workers}"
        )
    
    def index_memory(self, memory: Memory, vector: Optional[List[float]] = None):
        """
        Index a memory across all strategies.
        
        Args:
            memory: The memory to index
            vector: Optional pre-computed embedding vector
        """
        self._hierarchical.index_memory(memory)
        self._graph.index_memory(memory)
        self._semantic.index_memory(memory, vector)
        self._recency.index_memory(memory)
        
        # Invalidate cache as index has changed
        self._cache.invalidate(memory.id)
        
        logger.debug(f"Indexed memory {memory.id}")
    
    def add_graph_relationship(
        self, 
        from_id: str, 
        to_id: str, 
        relation_type: str = "related_to",
        bidirectional: bool = True
    ):
        """Add a relationship between two memories for graph retrieval."""
        self._graph.add_relationship(from_id, to_id, relation_type, bidirectional)
    
    def update_access_time(self, memory_id: str):
        """Update the access time for a memory (call when memory is used)."""
        self._recency.update_access(memory_id)
    
    def retrieve(
        self, 
        query: str, 
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[RetrievedMemory]:
        """
        Run 4 strategies in parallel, merge results, return top N.
        
        Each memory gets a combined relevance score from all strategies.
        
        Args:
            query: The search query string
            limit: Maximum number of memories to return (default: self.default_limit)
            context: Additional context for retrieval (e.g., current conversation)
            use_cache: Whether to use cached results
            
        Returns:
            List of RetrievedMemory objects sorted by combined relevance score
        """
        limit = limit or self.default_limit
        context = context or {}
        
        # Check cache
        if use_cache:
            cached = self._cache.get(query, limit)
            if cached is not None:
                if self.enable_metrics:
                    with self._metrics_lock:
                        self._metrics.record_query(0.0, cache_hit=True)
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached
        
        start_time = time.time()
        
        # Run strategies in parallel
        strategy_results = self._execute_strategies_parallel(query, context, limit)
        
        # Combine scores
        combined_results = self._combiner.combine(strategy_results, limit * 2)
        
        # Take top N
        final_results = combined_results[:limit]
        
        # Update access times for retrieved memories
        for result in final_results:
            self.update_access_time(result.memory.id)
        
        # Cache results
        if use_cache:
            self._cache.put(query, limit, final_results)
        
        # Record metrics
        elapsed_ms = (time.time() - start_time) * 1000
        if self.enable_metrics:
            with self._metrics_lock:
                self._metrics.record_query(elapsed_ms, cache_hit=False)
        
        logger.info(
            f"Retrieved {len(final_results)} memories for query "
            f"'{query[:50]}...' in {elapsed_ms:.2f}ms"
        )
        
        return final_results
    
    def _execute_strategies_parallel(
        self, 
        query: str, 
        context: Dict[str, Any],
        limit: int
    ) -> Dict[RetrievalStrategyType, List[Tuple[Memory, float]]]:
        """
        Execute all retrieval strategies in parallel using ThreadPoolExecutor.
        
        Returns:
            Dictionary mapping strategy types to their results
        """
        results = {}
        futures = {}
        
        # Submit all strategy tasks
        with self._executor as executor:
            for strategy_type, strategy in self._strategies.items():
                future = executor.submit(
                    self._execute_strategy_with_timing,
                    strategy,
                    strategy_type,
                    query,
                    context,
                    limit
                )
                futures[future] = strategy_type
            
            # Collect results as they complete
            for future in as_completed(futures):
                strategy_type = futures[future]
                try:
                    result, duration_ms = future.result()
                    results[strategy_type] = result
                    
                    if self.enable_metrics:
                        with self._metrics_lock:
                            self._metrics.record_strategy_time(strategy_type, duration_ms)
                    
                    logger.debug(
                        f"{strategy_type.value} retrieved {len(result)} items "
                        f"in {duration_ms:.2f}ms"
                    )
                except Exception as e:
                    logger.error(f"Strategy {strategy_type.value} failed: {e}")
                    results[strategy_type] = []
        
        return results
    
    def _execute_strategy_with_timing(
        self,
        strategy: RetrievalStrategy,
        strategy_type: RetrievalStrategyType,
        query: str,
        context: Dict[str, Any],
        limit: int
    ) -> Tuple[List[Tuple[Memory, float]], float]:
        """Execute a single strategy and measure timing."""
        start = time.time()
        result = strategy.retrieve(query, context, limit)
        duration_ms = (time.time() - start) * 1000
        return result, duration_ms
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive retrieval metrics."""
        with self._metrics_lock:
            metrics = {
                "total_queries": self._metrics.total_queries,
                "avg_query_time_ms": self._metrics.avg_query_time_ms,
                "cache_hit_rate": (
                    self._metrics.cache_hits / self._metrics.total_queries
                    if self._metrics.total_queries > 0 else 0.0
                ),
                "cache_hits": self._metrics.cache_hits,
                "cache_misses": self._metrics.cache_misses,
                "strategy_times": {
                    k.value: v for k, v in self._metrics.strategy_times.items()
                },
                "strategy_metrics": {
                    name.value: strat.get_metrics()
                    for name, strat in self._strategies.items()
                }
            }
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics to zero."""
        with self._metrics_lock:
            self._metrics = RetrievalMetrics()
    
    def get_context_size_estimate(self, memories: List[RetrievedMemory]) -> int:
        """
        Estimate the size of memories in bytes.
        
        Used to ensure context stays within limits (~5KB default).
        """
        total_bytes = 0
        for mem in memories:
            # Estimate: content length + metadata overhead
            content_bytes = len(mem.memory.content.encode('utf-8'))
            metadata_bytes = 200  # Approximate overhead per memory
            total_bytes += content_bytes + metadata_bytes
        return total_bytes
    
    def adjust_limit_for_context(
        self, 
        target_bytes: int = 5120,
        max_memories: int = 20
    ) -> int:
        """
        Calculate appropriate limit based on average memory size.
        
        Args:
            target_bytes: Target context size in bytes (default 5KB)
            max_memories: Hard maximum number of memories
            
        Returns:
            Recommended memory limit
        """
        # Get sample of recent memories
        sample_results = self.retrieve("*", limit=100, use_cache=False)
        if not sample_results:
            return self.default_limit
        
        avg_size = self.get_context_size_estimate(sample_results) / len(sample_results)
        recommended = int(target_bytes / avg_size) if avg_size > 0 else max_memories
        
        return min(recommended, max_memories)
    
    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        logger.info("HybridRetriever shut down")


# Convenience functions for common use cases

def create_default_retriever(
    default_limit: int = 15,
    weights: Optional[RetrievalWeights] = None
) -> HybridRetriever:
    """Create a retriever with sensible defaults."""
    return HybridRetriever(
        default_limit=default_limit,
        weights=weights or RetrievalWeights()
    )


def create_accuracy_focused_retriever(default_limit: int = 15) -> HybridRetriever:
    """Create a retriever optimized for accuracy (high semantic weight)."""
    weights = RetrievalWeights(
        hierarchical=0.15,
        graph=0.20,
        semantic=0.50,
        recency=0.15
    )
    return HybridRetriever(default_limit=default_limit, weights=weights)


def create_recency_focused_retriever(default_limit: int = 15) -> HybridRetriever:
    """Create a retriever optimized for recent context."""
    weights = RetrievalWeights(
        hierarchical=0.15,
        graph=0.20,
        semantic=0.25,
        recency=0.40
    )
    return HybridRetriever(default_limit=default_limit, weights=weights)


def create_relationship_focused_retriever(default_limit: int = 15) -> HybridRetriever:
    """Create a retriever optimized for relationship traversal."""
    weights = RetrievalWeights(
        hierarchical=0.15,
        graph=0.45,
        semantic=0.25,
        recency=0.15
    )
    return HybridRetriever(default_limit=default_limit, weights=weights)


# Example usage and testing

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create retriever
    retriever = create_default_retriever(default_limit=10)
    
    # Index some sample memories
    for i in range(20):
        memory = Memory(
            id=f"mem_{i}",
            content=f"This is memory number {i} about topic {i % 5}",
            importance=5 + (i % 5),
            tags={f"topic_{i % 5}", "sample"},
            vector=[0.1 * (i % 10)] * 384  # Sample vector
        )
        retriever.index_memory(memory)
    
    # Add some relationships
    retriever.add_graph_relationship("mem_0", "mem_1", "related_to")
    retriever.add_graph_relationship("mem_1", "mem_2", "depends_on")
    retriever.add_graph_relationship("mem_5", "mem_10", "part_of")
    
    # Perform retrieval
    results = retriever.retrieve("topic 1", limit=10)
    
    print("\n=== Retrieval Results ===")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result.combined_score:.3f}] {result.memory.content}")
        print(f"   Reason: {result.retrieval_reason}")
        print()
    
    # Show metrics
    metrics = retriever.get_metrics()
    print("\n=== Metrics ===")
    print(f"Total queries: {metrics['total_queries']}")
    print(f"Avg query time: {metrics['avg_query_time_ms']:.2f}ms")
    print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
    
    retriever.close()
