"""
Unified Knowledge Graph Implementation

High-level unified interface for knowledge graph operations,
integrating with the Unified KG Integration Hub.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
import asyncio
from collections import defaultdict

# Optional imports with fallbacks
try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    nx = None

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False
    np = None

# Optional imports from graph module (may not be available)
try:
    from .models import KnowledgeNode, KnowledgeEdge, NodeProperties, EdgeProperties
    _models_available = True
except ImportError:
    _models_available = False
    KnowledgeNode = None
    KnowledgeEdge = None
    NodeProperties = None
    EdgeProperties = None

try:
    from .schema import NodeType, EdgeType
    _schema_available = True
except ImportError:
    _schema_available = False
    NodeType = None
    EdgeType = None

logger = logging.getLogger(__name__)


@dataclass
class UnifiedTriple:
    """Unified triple representation for integration hub."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedTriple":
        """Create from dictionary."""
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "unknown"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
            metadata=data.get("metadata", {})
        )


@dataclass
class GraphStatistics:
    """Statistics for the knowledge graph."""
    node_count: int = 0
    edge_count: int = 0
    node_types: Dict[str, int] = field(default_factory=dict)
    edge_types: Dict[str, int] = field(default_factory=dict)
    sources: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    last_updated: Optional[datetime] = None


class UnifiedKnowledgeGraph:
    """
    Unified Knowledge Graph for Integration Hub.
    
    Provides a high-level interface for knowledge graph operations
    with support for multiple backends (NetworkX, in-memory, etc.)
    and integration with the Unified KG Integration Hub.
    
    Features:
    - Triple storage and retrieval
    - Entity and relationship management
    - Graph traversal and search
    - Statistics and analytics
    - Export/import capabilities
    - Async operations support
    """
    
    def __init__(self, backend: str = "networkx", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified knowledge graph.
        
        Args:
            backend: Storage backend ("networkx", "memory")
            config: Backend-specific configuration
        """
        self.backend = backend
        self.config = config or {}
        self._initialized = False
        
        # In-memory storage (always available)
        self._triples: List[UnifiedTriple] = []
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._relations: Dict[str, Dict[str, Any]] = {}
        
        # NetworkX graph (if available)
        self._nx_graph = None
        if backend == "networkx" and NX_AVAILABLE:
            self._nx_graph = nx.DiGraph()
        
        # Statistics
        self._stats = GraphStatistics()
        self._stats.last_updated = datetime.now(timezone.utc)
        
        # Indices for fast lookup
        self._entity_index: Dict[str, Set[int]] = defaultdict(set)
        self._predicate_index: Dict[str, Set[int]] = defaultdict(set)
        
        logger.info(f"UnifiedKnowledgeGraph initialized with {backend} backend")
    
    async def initialize(self) -> bool:
        """Initialize the knowledge graph."""
        if self._initialized:
            return True
        
        try:
            # Initialize indices
            self._rebuild_indices()
            self._initialized = True
            logger.info("UnifiedKnowledgeGraph initialization complete")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize UnifiedKnowledgeGraph: {e}")
            return False
    
    def _rebuild_indices(self):
        """Rebuild lookup indices."""
        self._entity_index.clear()
        self._predicate_index.clear()
        
        for i, triple in enumerate(self._triples):
            self._entity_index[triple.subject.lower()].add(i)
            self._entity_index[triple.object.lower()].add(i)
            self._predicate_index[triple.predicate.lower()].add(i)
    
    # ========================================================================
    # Triple Operations
    # ========================================================================
    
    def add_triple(self, triple: UnifiedTriple) -> bool:
        """
        Add a triple to the knowledge graph.
        
        Args:
            triple: Triple to add
            
        Returns:
            True if added successfully
        """
        try:
            self._triples.append(triple)
            idx = len(self._triples) - 1
            
            # Update indices
            self._entity_index[triple.subject.lower()].add(idx)
            self._entity_index[triple.object.lower()].add(idx)
            self._predicate_index[triple.predicate.lower()].add(idx)
            
            # Update entities
            if triple.subject not in self._entities:
                self._entities[triple.subject] = {
                    "name": triple.subject,
                    "first_seen": triple.timestamp,
                    "sources": set()
                }
            self._entities[triple.subject]["sources"].add(triple.source)
            
            if triple.object not in self._entities:
                self._entities[triple.object] = {
                    "name": triple.object,
                    "first_seen": triple.timestamp,
                    "sources": set()
                }
            self._entities[triple.object]["sources"].add(triple.source)
            
            # Update relations
            rel_key = f"{triple.predicate}"
            if rel_key not in self._relations:
                self._relations[rel_key] = {
                    "predicate": triple.predicate,
                    "count": 0,
                    "sources": set()
                }
            self._relations[rel_key]["count"] += 1
            self._relations[rel_key]["sources"].add(triple.source)
            
            # Update NetworkX graph if available
            if self._nx_graph is not None:
                self._nx_graph.add_edge(
                    triple.subject,
                    triple.object,
                    predicate=triple.predicate,
                    confidence=triple.confidence,
                    source=triple.source,
                    timestamp=triple.timestamp
                )
            
            self._update_stats()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add triple: {e}")
            return False
    
    def add_triples(self, triples: List[UnifiedTriple]) -> int:
        """
        Add multiple triples.
        
        Args:
            triples: List of triples to add
            
        Returns:
            Number of triples added successfully
        """
        count = 0
        for triple in triples:
            if self.add_triple(triple):
                count += 1
        return count
    
    def get_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[UnifiedTriple]:
        """
        Get triples matching criteria.
        
        Args:
            subject: Filter by subject (optional)
            predicate: Filter by predicate (optional)
            object: Filter by object (optional)
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching triples
        """
        results = self._triples
        
        # Use indices for filtering when possible
        if subject:
            indices = self._entity_index.get(subject.lower(), set())
            results = [self._triples[i] for i in indices]
        
        if predicate:
            indices = self._predicate_index.get(predicate.lower(), set())
            if subject:
                # Intersect with subject results
                pred_triples = {self._triples[i] for i in indices}
                results = [t for t in results if t in pred_triples]
            else:
                results = [self._triples[i] for i in indices]
        
        if object:
            if subject or predicate:
                results = [t for t in results if t.object.lower() == object.lower()]
            else:
                indices = self._entity_index.get(object.lower(), set())
                results = [self._triples[i] for i in indices]
        
        # Filter by confidence
        results = [t for t in results if t.confidence >= min_confidence]
        
        return results
    
    def remove_triple(self, subject: str, predicate: str, object: str) -> bool:
        """
        Remove a specific triple.
        
        Args:
            subject: Subject of triple to remove
            predicate: Predicate of triple to remove
            object: Object of triple to remove
            
        Returns:
            True if removed successfully
        """
        try:
            for i, triple in enumerate(self._triples):
                if (triple.subject == subject and 
                    triple.predicate == predicate and 
                    triple.object == object):
                    self._triples.pop(i)
                    self._rebuild_indices()
                    self._update_stats()
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove triple: {e}")
            return False
    
    # ========================================================================
    # Entity Operations
    # ========================================================================
    
    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Get entity by name."""
        return self._entities.get(name)
    
    def get_related_entities(
        self,
        entity: str,
        relation: Optional[str] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Get entities related to a given entity.
        
        Args:
            entity: Entity name
            relation: Filter by relation (optional)
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            List of related entities with relation info
        """
        results = []
        entity_lower = entity.lower()
        
        for triple in self._triples:
            if direction in ("outgoing", "both") and triple.subject.lower() == entity_lower:
                if relation is None or triple.predicate == relation:
                    results.append({
                        "entity": triple.object,
                        "relation": triple.predicate,
                        "direction": "outgoing",
                        "confidence": triple.confidence
                    })
            
            if direction in ("incoming", "both") and triple.object.lower() == entity_lower:
                if relation is None or triple.predicate == relation:
                    results.append({
                        "entity": triple.subject,
                        "relation": triple.predicate,
                        "direction": "incoming",
                        "confidence": triple.confidence
                    })
        
        return results
    
    def get_all_entities(self) -> List[str]:
        """Get all entity names."""
        return list(self._entities.keys())
    
    # ========================================================================
    # Graph Analytics
    # ========================================================================
    
    def get_statistics(self) -> GraphStatistics:
        """Get graph statistics."""
        self._update_stats()
        return self._stats
    
    def _update_stats(self):
        """Update graph statistics."""
        self._stats.node_count = len(self._entities)
        self._stats.edge_count = len(self._triples)
        self._stats.last_updated = datetime.now(timezone.utc)
        
        # Calculate source distribution
        sources = defaultdict(int)
        for triple in self._triples:
            sources[triple.source] += 1
        self._stats.sources = dict(sources)
        
        # Calculate average confidence
        if self._triples:
            self._stats.avg_confidence = sum(t.confidence for t in self._triples) / len(self._triples)
    
    def find_paths(
        self,
        source: str,
        target: str,
        max_length: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two entities.
        
        Args:
            source: Source entity
            target: Target entity
            max_length: Maximum path length
            
        Returns:
            List of paths, each path is a list of edge info dicts
        """
        if not NX_AVAILABLE or self._nx_graph is None:
            # Fallback to simple BFS
            return self._find_paths_bfs(source, target, max_length)
        
        try:
            paths = []
            for path in nx.all_simple_paths(
                self._nx_graph,
                source,
                target,
                cutoff=max_length
            ):
                path_edges = []
                for i in range(len(path) - 1):
                    edge_data = self._nx_graph.edges[path[i], path[i+1]]
                    path_edges.append({
                        "from": path[i],
                        "to": path[i+1],
                        "predicate": edge_data.get("predicate", "related_to"),
                        "confidence": edge_data.get("confidence", 1.0)
                    })
                paths.append(path_edges)
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def _find_paths_bfs(
        self,
        source: str,
        target: str,
        max_length: int
    ) -> List[List[Dict[str, Any]]]:
        """Fallback BFS path finding."""
        paths = []
        visited = {source}
        queue = [(source, [])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) >= max_length:
                continue
            
            # Find outgoing edges
            for triple in self._triples:
                if triple.subject == current and triple.object not in visited:
                    new_path = path + [{
                        "from": triple.subject,
                        "to": triple.object,
                        "predicate": triple.predicate,
                        "confidence": triple.confidence
                    }]
                    
                    if triple.object == target:
                        paths.append(new_path)
                    else:
                        visited.add(triple.object)
                        queue.append((triple.object, new_path))
        
        return paths
    
    def get_neighbors(self, entity: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get neighborhood of an entity.
        
        Args:
            entity: Center entity
            depth: Neighborhood depth
            
        Returns:
            Neighborhood subgraph info
        """
        if depth < 1:
            return {"center": entity, "nodes": [entity], "edges": []}
        
        nodes = {entity}
        edges = []
        current_frontier = {entity}
        
        for _ in range(depth):
            next_frontier = set()
            for e in current_frontier:
                for triple in self._triples:
                    if triple.subject == e:
                        nodes.add(triple.object)
                        next_frontier.add(triple.object)
                        edges.append({
                            "from": triple.subject,
                            "to": triple.object,
                            "predicate": triple.predicate
                        })
                    elif triple.object == e:
                        nodes.add(triple.subject)
                        next_frontier.add(triple.subject)
                        edges.append({
                            "from": triple.subject,
                            "to": triple.object,
                            "predicate": triple.predicate
                        })
            current_frontier = next_frontier
        
        return {
            "center": entity,
            "nodes": list(nodes),
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
    
    # ========================================================================
    # Export/Import
    # ========================================================================
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary."""
        return {
            "triples": [t.to_dict() for t in self._triples],
            "entities": {k: dict(v, sources=list(v["sources"]) if "sources" in v else []) 
                        for k, v in self._entities.items()},
            "relations": {k: dict(v, sources=list(v["sources"]) if "sources" in v else []) 
                         for k, v in self._relations.items()},
            "statistics": {
                "node_count": self._stats.node_count,
                "edge_count": self._stats.edge_count,
                "avg_confidence": self._stats.avg_confidence,
                "export_timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    
    def import_from_dict(self, data: Dict[str, Any]) -> int:
        """
        Import graph from dictionary.
        
        Args:
            data: Dictionary with graph data
            
        Returns:
            Number of triples imported
        """
        count = 0
        if "triples" in data:
            for triple_data in data["triples"]:
                triple = UnifiedTriple.from_dict(triple_data)
                if self.add_triple(triple):
                    count += 1
        return count
    
    def export_to_networkx(self) -> Optional[Any]:
        """Export to NetworkX graph (if available)."""
        return self._nx_graph
    
    # ========================================================================
    # Async Operations
    # ========================================================================
    
    async def add_triple_async(self, triple: UnifiedTriple) -> bool:
        """Async version of add_triple."""
        return await asyncio.to_thread(self.add_triple, triple)
    
    async def get_triples_async(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[UnifiedTriple]:
        """Async version of get_triples."""
        return await asyncio.to_thread(
            self.get_triples,
            subject,
            predicate,
            object,
            min_confidence
        )
    
    async def find_paths_async(
        self,
        source: str,
        target: str,
        max_length: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """Async version of find_paths."""
        return await asyncio.to_thread(self.find_paths, source, target, max_length)
    
    # ========================================================================
    # Health Check
    # ========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """Check graph health."""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "backend": self.backend,
            "networkx_available": NX_AVAILABLE,
            "numpy_available": NP_AVAILABLE,
            "triples_count": len(self._triples),
            "entities_count": len(self._entities),
            "relations_count": len(self._relations),
            "stats": {
                "node_count": self._stats.node_count,
                "edge_count": self._stats.edge_count,
                "avg_confidence": self._stats.avg_confidence
            }
        }
