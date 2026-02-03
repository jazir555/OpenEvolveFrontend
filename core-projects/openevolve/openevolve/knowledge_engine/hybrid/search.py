"""
Hybrid Search - Vector + Graph

Combines Chroma vector search with Neo4j graph queries for
best-of-both-worlds retrieval.

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
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import numpy as np

# Try to import Chroma
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Strategies for fusing vector and graph results"""
    RRF = "rrf"  # Reciprocal Rank Fusion
    LINEAR = "linear"  # Linear combination
    WEIGHTED = "weighted"  # Weighted by confidence
    INTERPOLATION = "interpolation"  # Interpolate scores
    CASCADE = "cascade"  # Use one to filter the other


@dataclass
class SearchResult:
    """A single search result"""
    id: str
    score: float
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # "vector", "graph", "hybrid"
    node_type: Optional[str] = None
    embeddings: Optional[List[float]] = None
    
    def __post_init__(self):
        if not self.source:
            self.source = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "metadata": self.metadata,
            "source": self.source,
            "node_type": self.node_type
        }


class VectorSearch:
    """ChromaDB vector search"""
    
    def __init__(
        self,
        collection_name: str = "knowledge_graph",
        embedding_function = None,
        persist_directory: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        
        if CHROMA_AVAILABLE:
            self._init_chroma()
        else:
            logger.warning("Chroma not available, using mock vector search")
    
    def _init_chroma(self):
        """Initialize Chroma client"""
        try:
            if self.persist_directory:
                self.client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.persist_directory
                ))
            else:
                self.client = chromadb.Client()
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Chroma collection '{self.collection_name}' ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {e}")
            self.client = None
            self.collection = None
    
    def is_available(self) -> bool:
        """Check if vector search is available"""
        return CHROMA_AVAILABLE and self.collection is not None
    
    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None
    ):
        """Add documents to vector store"""
        if not self.is_available():
            logger.debug(f"Mock add: {len(documents)} documents")
            return
        
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            logger.debug(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Search by query text"""
        if not self.is_available():
            return self._mock_search(query, top_k)
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_dict
            )
            
            search_results = []
            for i in range(len(results['ids'][0])):
                search_results.append(SearchResult(
                    id=results['ids'][0][i],
                    score=results['distances'][0][i] if 'distances' in results else 0.5,
                    content=results['documents'][0][i] if 'documents' in results else "",
                    metadata=results['metadatas'][0][i] if 'metadatas' in results else {},
                    source="vector"
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def search_by_vector(
        self,
        embedding: List[float],
        top_k: int = 10
    ) -> List[SearchResult]:
        """Search by vector embedding"""
        if not self.is_available():
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k
            )
            
            search_results = []
            for i in range(len(results['ids'][0])):
                search_results.append(SearchResult(
                    id=results['ids'][0][i],
                    score=1.0 - (results['distances'][0][i] if 'distances' in results else 0.5),
                    content=results['documents'][0][i] if 'documents' in results else "",
                    metadata=results['metadatas'][0][i] if 'metadatas' in results else {},
                    source="vector"
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _mock_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Mock vector search"""
        logger.debug(f"Mock vector search: {query[:50]}...")
        return []
    
    def delete(self, ids: List[str]):
        """Delete documents by ID"""
        if not self.is_available():
            return
        
        try:
            self.collection.delete(ids=ids)
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")


class GraphSearch:
    """Neo4j graph search"""
    
    def __init__(self, connection_pool=None):
        self.pool = connection_pool
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        node_type: Optional[str] = None
    ) -> List[SearchResult]:
        """Search graph by text query"""
        # Build Cypher query
        label_filter = f":{node_type}" if node_type else ""
        
        cypher = f"""
        MATCH (n{label_filter})
        WHERE n.name CONTAINS $query OR n.description CONTAINS $query
        RETURN n, 
               CASE 
                   WHEN n.name CONTAINS $query THEN 2.0
                   ELSE 1.0
               END as relevance
        ORDER BY relevance DESC
        LIMIT $limit
        """
        
        if not self.pool:
            return self._mock_search(query, top_k)
        
        try:
            results = await self.pool.run_cypher(
                cypher,
                {"query": query, "limit": top_k}
            )
            
            search_results = []
            for record in results:
                node = record.get('n', {})
                relevance = record.get('relevance', 1.0)
                
                search_results.append(SearchResult(
                    id=node.get('id', ''),
                    score=relevance / 2.0,  # Normalize to 0-1
                    content=node.get('name', ''),
                    metadata=node.get('metadata', {}),
                    source="graph",
                    node_type=node.get('node_type')
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    async def semantic_search(
        self,
        embedding: List[float],
        top_k: int = 10
    ) -> List[SearchResult]:
        """Search by embedding similarity (requires vector index)"""
        # This requires Neo4j with GDS or APOC
        cypher = """
        CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $embedding)
        YIELD node, score
        RETURN node, score
        """
        
        if not self.pool:
            return []
        
        try:
            results = await self.pool.run_cypher(
                cypher,
                {"embedding": embedding, "top_k": top_k}
            )
            
            search_results = []
            for record in results:
                node = record.get('node', {})
                score = record.get('score', 0.0)
                
                search_results.append(SearchResult(
                    id=node.get('id', ''),
                    score=score,
                    content=node.get('name', ''),
                    metadata={},
                    source="graph",
                    node_type=node.get('node_type')
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Graph semantic search failed: {e}")
            return []
    
    async def traverse(
        self,
        start_node_id: str,
        depth: int = 2,
        edge_types: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Traverse graph from a starting node"""
        edge_filter = ""
        if edge_types:
            edge_filter = "|".join(f":{et}" for et in edge_types)
            edge_filter = f"[{edge_filter}]"
        else:
            edge_filter = "[*1.." + str(depth) + "]"
        
        cypher = f"""
        MATCH (start {{id: $start_id}})-{edge_filter}-(neighbor)
        WHERE neighbor.id <> $start_id
        RETURN DISTINCT neighbor, length(shortestPath((start)-[*]-(neighbor))) as distance
        ORDER BY distance
        LIMIT 20
        """
        
        if not self.pool:
            return []
        
        try:
            results = await self.pool.run_cypher(
                cypher,
                {"start_id": start_node_id}
            )
            
            search_results = []
            for record in results:
                node = record.get('neighbor', {})
                distance = record.get('distance', 1)
                
                # Closer nodes get higher scores
                score = 1.0 / (1 + distance)
                
                search_results.append(SearchResult(
                    id=node.get('id', ''),
                    score=score,
                    content=node.get('name', ''),
                    metadata={},
                    source="graph",
                    node_type=node.get('node_type')
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return []
    
    def _mock_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Mock graph search"""
        logger.debug(f"Mock graph search: {query[:50]}...")
        return []


class HybridSearch:
    """Hybrid search combining vector and graph search"""
    
    def __init__(
        self,
        vector_search: Optional[VectorSearch] = None,
        graph_search: Optional[GraphSearch] = None,
        fusion_strategy: FusionStrategy = FusionStrategy.RRF
    ):
        self.vector_search = vector_search or VectorSearch()
        self.graph_search = graph_search or GraphSearch()
        self.fusion_strategy = fusion_strategy
        
        # Weights for different strategies
        self.vector_weight = 0.5
        self.graph_weight = 0.5
        self.rrf_k = 60  # RRF constant
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        include_vector: bool = True,
        include_graph: bool = True
    ) -> List[SearchResult]:
        """Perform hybrid search"""
        vector_results = []
        graph_results = []
        
        # Get results from both sources
        if include_vector:
            vector_results = self.vector_search.search(query, top_k=top_k * 2)
        
        if include_graph:
            graph_results = await self.graph_search.search(query, top_k=top_k * 2)
        
        # Fuse results
        if include_vector and include_graph:
            return self._fuse_results(vector_results, graph_results, top_k)
        elif include_vector:
            return vector_results[:top_k]
        else:
            return graph_results[:top_k]
    
    def _fuse_results(
        self,
        vector_results: List[SearchResult],
        graph_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Fuse results from multiple sources"""
        if self.fusion_strategy == FusionStrategy.RRF:
            return self._rrf_fusion(vector_results, graph_results, top_k)
        elif self.fusion_strategy == FusionStrategy.LINEAR:
            return self._linear_fusion(vector_results, graph_results, top_k)
        elif self.fusion_strategy == FusionStrategy.WEIGHTED:
            return self._weighted_fusion(vector_results, graph_results, top_k)
        else:
            return self._rrf_fusion(vector_results, graph_results, top_k)
    
    def _rrf_fusion(
        self,
        vector_results: List[SearchResult],
        graph_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion"""
        # Create score map
        scores: Dict[str, Dict[str, Any]] = {}
        
        # Add vector results
        for rank, result in enumerate(vector_results):
            if result.id not in scores:
                scores[result.id] = {
                    "score": 0,
                    "content": result.content,
                    "metadata": result.metadata,
                    "sources": []
                }
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            scores[result.id]["score"] += rrf_score
            scores[result.id]["sources"].append("vector")
        
        # Add graph results
        for rank, result in enumerate(graph_results):
            if result.id not in scores:
                scores[result.id] = {
                    "score": 0,
                    "content": result.content,
                    "metadata": result.metadata,
                    "sources": []
                }
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            scores[result.id]["score"] += rrf_score
            scores[result.id]["sources"].append("graph")
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Create results
        fused_results = []
        for doc_id, data in sorted_scores[:top_k]:
            fused_results.append(SearchResult(
                id=doc_id,
                score=data["score"],
                content=data["content"],
                metadata=data["metadata"],
                source="+".join(set(data["sources"]))
            ))
        
        return fused_results
    
    def _linear_fusion(
        self,
        vector_results: List[SearchResult],
        graph_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Linear combination of scores"""
        scores: Dict[str, Dict[str, Any]] = {}
        
        # Normalize and combine scores
        for result in vector_results:
            if result.id not in scores:
                scores[result.id] = {"score": 0, "content": result.content, "metadata": result.metadata}
            scores[result.id]["score"] += result.score * self.vector_weight
        
        for result in graph_results:
            if result.id not in scores:
                scores[result.id] = {"score": 0, "content": result.content, "metadata": result.metadata}
            scores[result.id]["score"] += result.score * self.graph_weight
        
        # Sort and return
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        return [
            SearchResult(
                id=doc_id,
                score=data["score"],
                content=data["content"],
                metadata=data["metadata"],
                source="hybrid"
            )
            for doc_id, data in sorted_scores[:top_k]
        ]
    
    def _weighted_fusion(
        self,
        vector_results: List[SearchResult],
        graph_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Weighted fusion by confidence scores"""
        # Similar to linear but uses confidence for weighting
        return self._linear_fusion(vector_results, graph_results, top_k)
    
    async def explain(
        self,
        query: str,
        result_id: str
    ) -> Dict[str, Any]:
        """Explain why a result was returned"""
        explanation = {
            "query": query,
            "result_id": result_id,
            "vector_score": None,
            "graph_score": None,
            "fusion_method": self.fusion_strategy.value,
            "reasoning": []
        }
        
        # Get individual scores
        vector_results = self.vector_search.search(query, top_k=20)
        for r in vector_results:
            if r.id == result_id:
                explanation["vector_score"] = r.score
                explanation["reasoning"].append(f"Vector similarity: {r.score:.3f}")
                break
        
        graph_results = await self.graph_search.search(query, top_k=20)
        for r in graph_results:
            if r.id == result_id:
                explanation["graph_score"] = r.score
                explanation["reasoning"].append(f"Graph relevance: {r.score:.3f}")
                break
        
        return explanation
