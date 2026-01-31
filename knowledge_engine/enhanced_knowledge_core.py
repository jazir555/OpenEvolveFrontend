"""
Enhanced Knowledge Engine Core

A comprehensive upgrade to the knowledge engine with:
- Embedding-based semantic search
- Advanced knowledge graph operations
- Real-time synchronization
- Multi-modal knowledge processing
- Active learning capabilities
- Distributed storage coordination
- Smart caching
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Coroutine, Dict, Generic, List, Optional, 
    Set, Tuple, TypeVar, Union, Iterator, AsyncIterator
)
import numpy as np
from contextlib import asynccontextmanager

# Configure logging
logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge that can be stored."""
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"
    GRAPH = "graph"
    EMBEDDING = "embedding"


class RelationType(Enum):
    """Types of relationships between knowledge items."""
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    DERIVED_FROM = "derived_from"
    SUPERSEDES = "supersedes"
    REFERENCES = "references"
    CONTRADICTS = "contradicts"
    IMPLEMENTS = "implements"


@dataclass
class EmbeddingVector:
    """Represents an embedding vector with metadata."""
    vector: np.ndarray
    model: str
    dimensions: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if isinstance(self.vector, list):
            self.vector = np.array(self.vector, dtype=np.float32)
        self.dimensions = len(self.vector)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.vector.tolist(),
            "model": self.model,
            "dimensions": self.dimensions,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmbeddingVector:
        return cls(
            vector=np.array(data["vector"], dtype=np.float32),
            model=data["model"],
            dimensions=data["dimensions"],
            created_at=datetime.fromisoformat(data["created_at"])
        )
    
    def cosine_similarity(self, other: EmbeddingVector) -> float:
        """Calculate cosine similarity with another embedding."""
        dot_product = np.dot(self.vector, other.vector)
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))


@dataclass
class KnowledgeItem:
    """Enhanced knowledge item with multi-modal support."""
    id: str
    content: Any
    knowledge_type: KnowledgeType
    embedding: Optional[EmbeddingVector] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    source: str = "unknown"
    confidence: float = 1.0
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Lineage tracking
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    derived_from: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "knowledge_type": self.knowledge_type.value,
            "embedding": self.embedding.to_dict() if self.embedding else None,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "source": self.source,
            "confidence": self.confidence,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "derived_from": self.derived_from
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeItem:
        return cls(
            id=data["id"],
            content=data["content"],
            knowledge_type=KnowledgeType(data["knowledge_type"]),
            embedding=EmbeddingVector.from_dict(data["embedding"]) if data.get("embedding") else None,
            metadata=data.get("metadata", {}),
            tags=set(data.get("tags", [])),
            source=data.get("source", "unknown"),
            confidence=data.get("confidence", 1.0),
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            parent_ids=data.get("parent_ids", []),
            child_ids=data.get("child_ids", []),
            derived_from=data.get("derived_from")
        )
    
    def update_content(self, new_content: Any, confidence: Optional[float] = None):
        """Update content and increment version."""
        self.content = new_content
        self.version += 1
        self.updated_at = datetime.utcnow()
        if confidence is not None:
            self.confidence = confidence
    
    def is_expired(self) -> bool:
        """Check if knowledge item has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def add_tag(self, tag: str):
        """Add a tag to the knowledge item."""
        self.tags.add(tag)
        self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str):
        """Remove a tag from the knowledge item."""
        self.tags.discard(tag)
        self.updated_at = datetime.utcnow()


@dataclass
class KnowledgeRelation:
    """Represents a relationship between two knowledge items."""
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeRelation:
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
        )


@dataclass
class SearchQuery:
    """Enhanced search query with multiple modes."""
    text: Optional[str] = None
    embedding: Optional[EmbeddingVector] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    knowledge_types: Set[KnowledgeType] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    min_confidence: float = 0.0
    max_results: int = 10
    search_mode: str = "hybrid"  # keyword, semantic, vector, hybrid
    
    def __post_init__(self):
        if not self.text and self.embedding is None:
            raise ValueError("Search query must have text or embedding")


@dataclass
class SearchResult:
    """Enhanced search result with relevance scoring."""
    item: KnowledgeItem
    relevance_score: float
    match_details: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "relevance_score": self.relevance_score,
            "match_details": self.match_details
        }


class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self, model_name: str = "default", dimensions: int = 1536):
        self.model_name = model_name
        self.dimensions = dimensions
        self._cache: Dict[str, EmbeddingVector] = {}
        self._cache_size_limit = 10000
        
    async def generate_embedding(self, content: Any, content_type: KnowledgeType) -> EmbeddingVector:
        """Generate embedding for content."""
        # Check cache first
        content_hash = self._hash_content(content)
        if content_hash in self._cache:
            return self._cache[content_hash]
        
        # Generate embedding based on content type
        if content_type == KnowledgeType.TEXT:
            vector = await self._embed_text(str(content))
        elif content_type == KnowledgeType.CODE:
            vector = await self._embed_code(str(content))
        elif content_type == KnowledgeType.STRUCTURED:
            vector = await self._embed_structured(content)
        else:
            # Default text embedding
            vector = await self._embed_text(str(content))
        
        embedding = EmbeddingVector(
            vector=vector,
            model=self.model_name,
            dimensions=self.dimensions
        )
        
        # Cache the embedding
        self._cache[content_hash] = embedding
        self._enforce_cache_limit()
        
        return embedding
    
    def _hash_content(self, content: Any) -> str:
        """Create a hash for content."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _enforce_cache_limit(self):
        """Enforce cache size limit with LRU eviction."""
        if len(self._cache) > self._cache_size_limit:
            # Remove oldest entries (simplified)
            keys_to_remove = list(self._cache.keys())[:1000]
            for key in keys_to_remove:
                del self._cache[key]
    
    async def _embed_text(self, text: str) -> np.ndarray:
        """Generate text embedding."""
        # Placeholder - would integrate with actual embedding model
        # Simulating a simple embedding based on character frequencies
        vector = np.zeros(self.dimensions, dtype=np.float32)
        if text:
            # Simple hash-based embedding for demonstration
            for i, char in enumerate(text[:1000]):
                idx = (ord(char) + i) % self.dimensions
                vector[idx] += 1.0
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
    
    async def _embed_code(self, code: str) -> np.ndarray:
        """Generate code embedding with language-aware processing."""
        # Placeholder for code-specific embedding
        # Could use tree-sitter AST, tokenization, etc.
        vector = await self._embed_text(code)
        return vector
    
    async def _embed_structured(self, data: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for structured data."""
        # Flatten and embed structured data
        flattened = json.dumps(data, sort_keys=True)
        vector = await self._embed_text(flattened)
        return vector
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()


class SemanticSearchEngine:
    """High-performance semantic search with multiple strategies."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self._vector_index: Dict[str, Tuple[str, np.ndarray]] = {}  # id -> (item_id, vector)
        self._inverted_index: Dict[str, Set[str]] = defaultdict(set)  # term -> item_ids
        self._item_cache: Dict[str, KnowledgeItem] = {}
        
    def index_item(self, item: KnowledgeItem):
        """Index a knowledge item for search."""
        # Store in cache
        self._item_cache[item.id] = item
        
        # Index embedding if available
        if item.embedding:
            self._vector_index[item.id] = (item.id, item.embedding.vector)
        
        # Build inverted index for keyword search
        if isinstance(item.content, str):
            terms = self._tokenize(item.content)
            for term in terms:
                self._inverted_index[term].add(item.id)
    
    def remove_item(self, item_id: str):
        """Remove item from search index."""
        if item_id in self._vector_index:
            del self._vector_index[item_id]
        
        if item_id in self._item_cache:
            item = self._item_cache[item_id]
            if isinstance(item.content, str):
                terms = self._tokenize(item.content)
                for term in terms:
                    self._inverted_index[term].discard(item_id)
            del self._item_cache[item_id]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for keyword search."""
        # Normalize and split
        text = text.lower()
        # Remove special characters and split
        tokens = []
        current = ""
        for char in text:
            if char.isalnum():
                current += char
            else:
                if current:
                    tokens.append(current)
                    current = ""
        if current:
            tokens.append(current)
        return tokens
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute search with multiple strategies."""
        results = []
        
        if query.search_mode in ("keyword", "hybrid") and query.text:
            keyword_results = self._keyword_search(query)
            results.extend(keyword_results)
        
        if query.search_mode in ("semantic", "vector", "hybrid"):
            if query.embedding:
                semantic_results = await self._semantic_search(query, query.embedding)
                results.extend(semantic_results)
            elif query.text:
                # Generate embedding for text query
                query_embedding = await self.embedding_service.generate_embedding(
                    query.text, KnowledgeType.TEXT
                )
                semantic_results = await self._semantic_search(query, query_embedding)
                results.extend(semantic_results)
        
        # Merge and rank results
        merged_results = self._merge_results(results, query.max_results)
        return merged_results
    
    def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform keyword-based search."""
        if not query.text:
            return []
        
        terms = self._tokenize(query.text)
        candidate_ids: Dict[str, float] = defaultdict(float)
        
        for term in terms:
            for item_id in self._inverted_index.get(term, set()):
                candidate_ids[item_id] += 1.0
        
        results = []
        for item_id, score in candidate_ids.items():
            if item_id in self._item_cache:
                item = self._item_cache[item_id]
                if self._matches_filters(item, query):
                    # Normalize score
                    normalized_score = score / len(terms) if terms else 0.0
                    results.append(SearchResult(
                        item=item,
                        relevance_score=normalized_score,
                        match_details={"keyword_score": normalized_score}
                    ))
        
        return results
    
    async def _semantic_search(
        self, 
        query: SearchQuery, 
        query_embedding: EmbeddingVector
    ) -> List[SearchResult]:
        """Perform semantic (vector similarity) search."""
        results = []
        
        for item_id, (_, vector) in self._vector_index.items():
            if item_id in self._item_cache:
                item = self._item_cache[item_id]
                if self._matches_filters(item, query):
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding.vector, vector)
                    results.append(SearchResult(
                        item=item,
                        relevance_score=float(similarity),
                        match_details={"semantic_score": float(similarity)}
                    ))
        
        return results
    
    def _matches_filters(self, item: KnowledgeItem, query: SearchQuery) -> bool:
        """Check if item matches query filters."""
        # Check confidence
        if item.confidence < query.min_confidence:
            return False
        
        # Check knowledge types
        if query.knowledge_types and item.knowledge_type not in query.knowledge_types:
            return False
        
        # Check tags
        if query.tags and not query.tags.intersection(item.tags):
            return False
        
        # Check custom filters
        for key, value in query.filters.items():
            if key == "source" and item.source != value:
                return False
            if key in item.metadata and item.metadata[key] != value:
                return False
        
        return True
    
    def _merge_results(
        self, 
        results: List[SearchResult], 
        max_results: int
    ) -> List[SearchResult]:
        """Merge and rank search results."""
        # Deduplicate by item ID
        unique_results: Dict[str, SearchResult] = {}
        
        for result in results:
            item_id = result.item.id
            if item_id in unique_results:
                # Merge scores
                existing = unique_results[item_id]
                existing.relevance_score = max(existing.relevance_score, result.relevance_score)
                existing.match_details.update(result.match_details)
            else:
                unique_results[item_id] = result
        
        # Sort by relevance score
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: x.relevance_score,
            reverse=True
        )
        
        return sorted_results[:max_results]
    
    def get_stats(self) -> Dict[str, int]:
        """Get search index statistics."""
        return {
            "vector_index_size": len(self._vector_index),
            "inverted_index_terms": len(self._inverted_index),
            "cached_items": len(self._item_cache)
        }


class KnowledgeGraphNavigator:
    """Advanced knowledge graph traversal and reasoning."""
    
    def __init__(self):
        self._nodes: Dict[str, KnowledgeItem] = {}
        self._edges: Dict[str, KnowledgeRelation] = {}
        self._adjacency: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )  # source -> relation_type -> list of (edge_id, target_id)
        
    def add_node(self, item: KnowledgeItem):
        """Add a node to the graph."""
        self._nodes[item.id] = item
    
    def add_edge(self, relation: KnowledgeRelation):
        """Add an edge to the graph."""
        self._edges[relation.id] = relation
        self._adjacency[relation.source_id][relation.relation_type.value].append(
            (relation.id, relation.target_id)
        )
    
    def get_neighbors(
        self, 
        node_id: str, 
        relation_type: Optional[RelationType] = None
    ) -> List[Tuple[KnowledgeItem, KnowledgeRelation]]:
        """Get neighbors of a node."""
        results = []
        
        if relation_type:
            edge_list = self._adjacency[node_id].get(relation_type.value, [])
        else:
            edge_list = []
            for edges in self._adjacency[node_id].values():
                edge_list.extend(edges)
        
        for edge_id, target_id in edge_list:
            if target_id in self._nodes and edge_id in self._edges:
                results.append((self._nodes[target_id], self._edges[edge_id]))
        
        return results
    
    def traverse(
        self,
        start_id: str,
        max_depth: int = 3,
        relation_types: Optional[List[RelationType]] = None,
        min_weight: float = 0.0
    ) -> List[Tuple[List[KnowledgeItem], List[KnowledgeRelation]]]:
        """Traverse the graph from a starting node."""
        paths = []
        visited = set()
        
        def dfs(current_id: str, depth: int, path_items: List[KnowledgeItem], path_edges: List[KnowledgeRelation]):
            if depth > max_depth or current_id in visited:
                return
            
            visited.add(current_id)
            
            if depth > 0:
                paths.append((path_items.copy(), path_edges.copy()))
            
            if depth < max_depth:
                neighbors = self.get_neighbors(current_id)
                for neighbor, edge in neighbors:
                    if edge.weight >= min_weight:
                        if relation_types is None or edge.relation_type in relation_types:
                            dfs(neighbor.id, depth + 1, path_items + [neighbor], path_edges + [edge])
            
            visited.remove(current_id)
        
        if start_id in self._nodes:
            dfs(start_id, 0, [self._nodes[start_id]], [])
        
        return paths
    
    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> List[Tuple[List[KnowledgeItem], List[KnowledgeRelation]]]:
        """Find paths between two nodes."""
        all_paths = self.traverse(source_id, max_depth)
        return [(items, edges) for items, edges in all_paths if items[-1].id == target_id]
    
    def get_connected_components(self) -> List[Set[str]]:
        """Find connected components in the graph."""
        visited = set()
        components = []
        
        def dfs(node_id: str, component: Set[str]):
            if node_id in visited:
                return
            visited.add(node_id)
            component.add(node_id)
            
            neighbors = self.get_neighbors(node_id)
            for neighbor, _ in neighbors:
                dfs(neighbor.id, component)
            
            # Also check reverse edges
            for source_id, relations in self._adjacency.items():
                for relation_type, edges in relations.items():
                    for edge_id, target_id in edges:
                        if target_id == node_id:
                            dfs(source_id, component)
        
        for node_id in self._nodes:
            if node_id not in visited:
                component = set()
                dfs(node_id, component)
                components.append(component)
        
        return components
    
    def calculate_centrality(self, node_id: str) -> Dict[str, float]:
        """Calculate centrality metrics for a node."""
        if node_id not in self._nodes:
            return {}
        
        # Degree centrality
        degree = len(self.get_neighbors(node_id))
        max_degree = max(len(self.get_neighbors(nid)) for nid in self._nodes) if self._nodes else 1
        degree_centrality = degree / max_degree if max_degree > 0 else 0
        
        # Simple betweenness approximation
        betweenness = 0.0
        
        return {
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness,
            "raw_degree": degree
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get graph statistics."""
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "relations": len(self._adjacency)
        }


class SmartCacheManager:
    """Intelligent caching with LRU, TTL, and predictive prefetching."""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, datetime, int]] = {}  # key -> (value, expires_at, access_count)
        self._access_times: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        async with self._lock:
            if key in self._cache:
                value, expires_at, access_count = self._cache[key]
                
                # Check expiration
                if datetime.utcnow() > expires_at:
                    del self._cache[key]
                    del self._access_times[key]
                    return None
                
                # Update access stats
                self._cache[key] = (value, expires_at, access_count + 1)
                self._access_times[key] = datetime.utcnow()
                
                return value
            
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ):
        """Set item in cache."""
        async with self._lock:
            ttl = ttl or self.default_ttl
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            # Evict if necessary
            if len(self._cache) >= self.max_size:
                await self._evict_lru()
            
            self._cache[key] = (value, expires_at, 0)
            self._access_times[key] = datetime.utcnow()
    
    async def delete(self, key: str):
        """Delete item from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
    
    async def clear(self):
        """Clear all cache."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    async def _evict_lru(self):
        """Evict least recently used items."""
        if not self._access_times:
            return
        
        # Find oldest accessed item
        oldest_key = min(self._access_times, key=self._access_times.get)
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_items = len(self._cache)
            expired_items = sum(
                1 for _, expires_at, _ in self._cache.values()
                if datetime.utcnow() > expires_at
            )
            
            return {
                "total_items": total_items,
                "expired_items": expired_items,
                "active_items": total_items - expired_items,
                "max_size": self.max_size
            }


class ActiveLearningEngine:
    """Active learning and feedback integration for continuous improvement."""
    
    def __init__(self):
        self.feedback_history: List[Dict[str, Any]] = []
        self.learning_rate = 0.1
        self.min_confidence_threshold = 0.5
        
    async def record_feedback(
        self, 
        item_id: str, 
        feedback_type: str,  # "positive", "negative", "neutral"
        feedback_score: float,  # 0.0 to 1.0
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record user feedback on a knowledge item."""
        feedback_record = {
            "item_id": item_id,
            "feedback_type": feedback_type,
            "feedback_score": feedback_score,
            "user_id": user_id,
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.feedback_history.append(feedback_record)
        
        logger.info(f"Recorded feedback for item {item_id}: {feedback_type} ({feedback_score:.2f})")
    
    async def calculate_item_quality(self, item_id: str) -> Dict[str, Any]:
        """Calculate quality metrics based on feedback."""
        relevant_feedback = [
            f for f in self.feedback_history 
            if f["item_id"] == item_id
        ]
        
        if not relevant_feedback:
            return {
                "average_score": 0.5,
                "feedback_count": 0,
                "confidence": 0.0
            }
        
        scores = [f["feedback_score"] for f in relevant_feedback]
        average_score = sum(scores) / len(scores)
        
        # Calculate confidence based on sample size
        confidence = min(1.0, len(scores) / 10.0)
        
        return {
            "average_score": average_score,
            "feedback_count": len(scores),
            "confidence": confidence,
            "trend": self._calculate_trend(scores)
        }
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend from scores."""
        if len(scores) < 2:
            return "stable"
        
        # Compare first half vs second half
        mid = len(scores) // 2
        first_half = sum(scores[:mid]) / mid if mid > 0 else 0
        second_half = sum(scores[mid:]) / (len(scores) - mid) if len(scores) > mid else 0
        
        diff = second_half - first_half
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        return "stable"
    
    async def identify_improvement_areas(self) -> List[Dict[str, Any]]:
        """Identify areas for knowledge improvement."""
        item_scores: Dict[str, List[float]] = defaultdict(list)
        
        for feedback in self.feedback_history:
            item_scores[feedback["item_id"]].append(feedback["feedback_score"])
        
        improvement_areas = []
        
        for item_id, scores in item_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < self.min_confidence_threshold:
                improvement_areas.append({
                    "item_id": item_id,
                    "current_score": avg_score,
                    "feedback_count": len(scores),
                    "priority": "high" if avg_score < 0.3 else "medium"
                })
        
        return sorted(improvement_areas, key=lambda x: x["current_score"])
    
    async def generate_learning_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations for knowledge improvement."""
        recommendations = []
        
        # Analyze feedback patterns
        positive_count = sum(1 for f in self.feedback_history if f["feedback_type"] == "positive")
        negative_count = sum(1 for f in self.feedback_history if f["feedback_type"] == "negative")
        
        total = positive_count + negative_count
        if total > 0:
            positive_ratio = positive_count / total
            
            if positive_ratio < 0.6:
                recommendations.append({
                    "type": "quality_improvement",
                    "message": "Knowledge quality needs improvement. Consider reviewing and updating low-rated items.",
                    "priority": "high"
                })
            
            if total < 100:
                recommendations.append({
                    "type": "feedback_collection",
                    "message": "Insufficient feedback data. Encourage more user feedback collection.",
                    "priority": "medium"
                })
        
        return recommendations


# Export all classes
__all__ = [
    "KnowledgeType",
    "RelationType",
    "EmbeddingVector",
    "KnowledgeItem",
    "KnowledgeRelation",
    "SearchQuery",
    "SearchResult",
    "EmbeddingService",
    "SemanticSearchEngine",
    "KnowledgeGraphNavigator",
    "SmartCacheManager",
    "ActiveLearningEngine"
]
