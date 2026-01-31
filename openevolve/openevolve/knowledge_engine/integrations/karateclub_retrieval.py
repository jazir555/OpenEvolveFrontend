"""
KarateClub Retrieval - Embedding-based Knowledge Retrieval

Use KarateClub embeddings for enhanced knowledge retrieval from graphs.

Follows CLAUDE.md principles:
- Runtime Truth: Validates embeddings at retrieval time
- Configuration Explicitness: All retrieval parameters via config
- Law of Idempotency: Safe to run multiple times
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import pickle

import networkx as nx
import numpy as np

from knowledge_engine.integrations.karateclub_analytics import (
    KarateClubAnalytics,
    NodeEmbeddingResult,
    GraphEmbeddingResult
)

logger = logging.getLogger(__name__)


@dataclass
class SimilarNode:
    """Similar node result"""
    node: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarGraph:
    """Similar graph result"""
    graph_id: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridResult:
    """Hybrid retrieval result"""
    query: str
    embedding_results: List[SimilarNode]
    keyword_results: List[SimilarNode]
    combined_results: List[SimilarNode]
    alpha: float
    execution_time_ms: float = 0.0


@dataclass
class EmbeddingIndex:
    """Embedding index for fast similarity search"""
    embeddings: Dict[str, List[float]]
    embedding_dim: int
    algorithm: str
    index_type: str  # 'faiss', 'annoy', 'brute'
    index: Any = None  # FAISS or Annoy index
    node_list: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class KarateClubRetrieval:
    """
    Use KarateClub embeddings for enhanced retrieval.

    Features:
    - Node similarity search
    - Graph similarity search
    - Hybrid retrieval (embeddings + keywords)
    - Efficient indexing (FAISS, Annoy, or brute-force)
    """

    def __init__(
        self,
        analytics_engine: KarateClubAnalytics,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize KarateClub Retrieval.

        Args:
            analytics_engine: KarateClubAnalytics instance
            config: Optional configuration dict
        """
        self.analytics = analytics_engine
        self.config = config or self._default_config()

        self.node_embeddings: Dict[str, NodeEmbeddingResult] = {}
        self.graph_embeddings: Dict[str, GraphEmbeddingResult] = {}
        self.indices: Dict[str, EmbeddingIndex] = {}

        logger.info("KarateClub Retrieval initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'embedding_model': 'node2vec',
            'dimensions': 128,
            'similarity_metric': 'cosine',  # cosine, euclidean, dot
            'index_type': 'brute',  # faiss, annoy, brute
            'top_k': 10,
            'save_embeddings': True,
            'cache_dir': '/tmp/karateclub_embeddings'
        }

    async def generate_embeddings_for_kg(
        self,
        graph: nx.Graph,
        index_name: str = 'default',
        algorithm: Optional[str] = None,
        dimensions: Optional[int] = None
    ) -> EmbeddingIndex:
        """
        Generate embeddings for knowledge graph retrieval.

        Process:
        1. Generate node embeddings (node2vec or specified algorithm)
        2. Generate graph embeddings (graph2vec or specified algorithm)
        3. Build index for similarity search
        4. Store for fast retrieval

        Args:
            graph: NetworkX graph
            index_name: Name for this embedding index
            algorithm: Embedding algorithm to use
            dimensions: Embedding dimensions

        Returns:
            EmbeddingIndex with embeddings and search index
        """
        start_time = datetime.utcnow()

        logger.info(f"Generating embeddings for KG '{index_name}'")

        # Use defaults if not specified
        if not algorithm:
            algorithm = self.config['embedding_model']
        if not dimensions:
            dimensions = self.config['dimensions']

        try:
            # 1. Generate node embeddings
            node_result = await self.analytics.generate_node_embeddings(
                graph,
                algorithm=algorithm,
                dimensions=dimensions
            )

            if not node_result.embeddings:
                raise ValueError("Failed to generate node embeddings")

            # 2. Build embedding index
            node_list = list(node_result.embeddings.keys())
            embedding_matrix = np.array([
                node_result.embeddings[node] for node in node_list
            ])

            # 3. Create search index
            index_type = self.config['index_type']
            index = None

            if index_type == 'faiss':
                index = self._build_faiss_index(embedding_matrix)
            elif index_type == 'annoy':
                index = self._build_annoy_index(embedding_matrix, dimensions)
            else:
                # Brute-force - no index needed
                index = None

            # 4. Create EmbeddingIndex
            embedding_index = EmbeddingIndex(
                embeddings=node_result.embeddings,
                embedding_dim=dimensions,
                algorithm=algorithm,
                index_type=index_type,
                index=index,
                node_list=node_list
            )

            # 5. Store index
            self.indices[index_name] = embedding_index
            self.node_embeddings[index_name] = node_result

            # 6. Optionally save to disk
            if self.config.get('save_embeddings', False):
                await self._save_embeddings(index_name, embedding_index)

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"Generated embeddings in {elapsed_ms:.2f}ms")

            return embedding_index

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    async def retrieve_similar_nodes(
        self,
        query_node: str,
        index_name: str = 'default',
        top_k: Optional[int] = None
    ) -> List[SimilarNode]:
        """
        Retrieve similar nodes using embeddings.

        Args:
            query_node: Query node ID
            index_name: Embedding index to use
            top_k: Number of similar nodes to return

        Returns:
            List of similar nodes with similarity scores
        """
        start_time = datetime.utcnow()

        if top_k is None:
            top_k = self.config['top_k']

        logger.info(f"Retrieving {top_k} similar nodes to '{query_node}'")

        # Get index
        if index_name not in self.indices:
            raise ValueError(f"Index '{index_name}' not found. Generate embeddings first.")

        embedding_index = self.indices[index_name]

        # Get query node embedding
        query_node_str = str(query_node)
        if query_node_str not in embedding_index.embeddings:
            raise ValueError(f"Node '{query_node}' not in embeddings")

        query_embedding = np.array(embedding_index.embeddings[query_node_str])

        try:
            # Retrieve similar nodes
            if embedding_index.index_type == 'faiss' and embedding_index.index is not None:
                similar_nodes = self._search_faiss(
                    embedding_index.index,
                    query_embedding,
                    embedding_index.node_list,
                    top_k
                )
            elif embedding_index.index_type == 'annoy' and embedding_index.index is not None:
                similar_nodes = self._search_annoy(
                    embedding_index.index,
                    query_embedding,
                    embedding_index.node_list,
                    top_k
                )
            else:
                # Brute-force
                similar_nodes = self._search_brute_force(
                    query_embedding,
                    embedding_index.embeddings,
                    top_k
                )

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"Retrieved similar nodes in {elapsed_ms:.2f}ms")

            return similar_nodes

        except Exception as e:
            logger.error(f"Failed to retrieve similar nodes: {e}")
            raise

    async def retrieve_similar_graphs(
        self,
        query_graph: nx.Graph,
        index_name: str = 'default',
        top_k: Optional[int] = None
    ) -> List[SimilarGraph]:
        """
        Retrieve similar subgraphs.

        Args:
            query_graph: Query graph
            index_name: Graph embedding index to use
            top_k: Number of similar graphs to return

        Returns:
            List of similar graphs with similarity scores
        """
        start_time = datetime.utcnow()

        if top_k is None:
            top_k = self.config['top_k']

        logger.info(f"Retrieving {top_k} similar graphs")

        try:
            # Generate embedding for query graph
            query_result = await self.analytics.generate_graph_embeddings(
                [query_graph],
                dimensions=self.config['dimensions']
            )

            if not query_result.embeddings or len(query_result.embeddings) == 0:
                raise ValueError("Failed to generate query graph embedding")

            query_embedding = np.array(query_result.embeddings[0])

            # Get graph embeddings index
            if index_name not in self.graph_embeddings:
                raise ValueError(f"Graph embedding index '{index_name}' not found")

            graph_result = self.graph_embeddings[index_name]

            # Compute similarities
            similarities = []
            for i, graph_embedding in enumerate(graph_result.embeddings):
                similarity = self._cosine_similarity(query_embedding, np.array(graph_embedding))
                similarities.append((
                    f"graph_{i}",
                    similarity,
                    {'index': i}
                ))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Return top-k
            top_similar = [
                SimilarGraph(
                    graph_id=graph_id,
                    similarity=float(sim),
                    metadata=metadata
                )
                for graph_id, sim, metadata in similarities[:top_k]
            ]

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"Retrieved similar graphs in {elapsed_ms:.2f}ms")

            return top_similar

        except Exception as e:
            logger.error(f"Failed to retrieve similar graphs: {e}")
            raise

    async def hybrid_retrieval(
        self,
        query: str,
        graph: nx.Graph,
        index_name: str = 'default',
        alpha: float = 0.5,
        top_k: Optional[int] = None
    ) -> HybridResult:
        """
        Hybrid retrieval combining embeddings and keywords.

        Formula: score = alpha * embedding_sim + (1 - alpha) * keyword_sim

        Args:
            query: Query string
            graph: Knowledge graph
            index_name: Embedding index to use
            alpha: Weight for embedding similarity (0-1)
            top_k: Number of results to return

        Returns:
            HybridResult with combined scores
        """
        start_time = datetime.utcnow()

        if top_k is None:
            top_k = self.config['top_k']

        alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]

        logger.info(f"Hybrid retrieval with alpha={alpha}")

        try:
            # 1. Keyword-based retrieval (simple text matching)
            keyword_results = self._keyword_search(query, graph, top_k * 2)

            # 2. Embedding-based retrieval
            # Use query node if found, otherwise search all
            embedding_results = []

            if index_name in self.indices:
                embedding_index = self.indices[index_name]

                # Compute similarity to all nodes
                query_lower = query.lower()

                for node_id, node_data in graph.nodes(data=True):
                    node_str = str(node_id)

                    # Check if query matches node name or content
                    match_score = 0.0
                    if query_lower in node_str.lower():
                        match_score = 0.8
                    elif 'content' in node_data and query_lower in str(node_data['content']).lower():
                        match_score = 0.6

                    if match_score > 0:
                        # Get embedding similarity if available
                        if node_str in embedding_index.embeddings:
                            # For now, use match score as placeholder
                            # In real implementation, you'd compute actual embedding similarity
                            embedding_results.append(
                                SimilarNode(
                                    node=node_str,
                                    similarity=match_score,
                                    metadata={'method': 'hybrid'}
                                )
                            )

            # 3. Combine scores
            combined_scores = {}

            # Add embedding scores
            for result in embedding_results:
                combined_scores[result.node] = alpha * result.similarity

            # Add keyword scores
            for result in keyword_results:
                if result.node in combined_scores:
                    combined_scores[result.node] += (1 - alpha) * result.similarity
                else:
                    combined_scores[result.node] = (1 - alpha) * result.similarity

            # Sort and get top-k
            sorted_results = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            combined_results = [
                SimilarNode(
                    node=node,
                    similarity=float(score),
                    metadata={'method': 'hybrid', 'alpha': alpha}
                )
                for node, score in sorted_results
            ]

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return HybridResult(
                query=query,
                embedding_results=embedding_results[:top_k],
                keyword_results=keyword_results[:top_k],
                combined_results=combined_results,
                alpha=alpha,
                execution_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            raise

    def _keyword_search(
        self,
        query: str,
        graph: nx.Graph,
        top_k: int
    ) -> List[SimilarNode]:
        """Simple keyword-based search"""
        query_lower = query.lower()
        results = []

        for node_id, node_data in graph.nodes(data=True):
            node_str = str(node_id)
            score = 0.0

            # Match in node ID
            if query_lower in node_str.lower():
                score += 0.5

            # Match in content
            if 'content' in node_data:
                content = str(node_data['content']).lower()
                if query_lower in content:
                    score += 0.5

            # Match in metadata
            if 'metadata' in node_data:
                metadata = str(node_data['metadata']).lower()
                if query_lower in metadata:
                    score += 0.3

            if score > 0:
                results.append(
                    SimilarNode(
                        node=node_str,
                        similarity=score,
                        metadata={'method': 'keyword'}
                    )
                )

        # Sort by score
        results.sort(key=lambda x: x.similarity, reverse=True)

        return results[:top_k]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search"""
        try:
            import faiss

            index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product (cosine for normalized vectors)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)

            return index

        except ImportError:
            logger.warning("FAISS not installed. Falling back to brute-force.")
            return None

    def _build_annoy_index(self, embeddings: np.ndarray, dimensions: int):
        """Build Annoy index for fast similarity search"""
        try:
            from annoy import AnnoyIndex

            index = AnnoyIndex(dimensions, 'angular')

            for i, embedding in enumerate(embeddings):
                index.add_item(i, embedding)

            index.build(10)  # 10 trees

            return index

        except ImportError:
            logger.warning("Annoy not installed. Falling back to brute-force.")
            return None

    def _search_faiss(
        self,
        index: Any,
        query_embedding: np.ndarray,
        node_list: List[str],
        top_k: int
    ) -> List[SimilarNode]:
        """Search using FAISS index"""
        import faiss

        # Normalize query
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = index.search(query_embedding, top_k + 1)  # +1 to skip self

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(node_list) and idx >= 0:
                results.append(
                    SimilarNode(
                        node=node_list[idx],
                        similarity=float(score),
                        metadata={'method': 'faiss'}
                    )
                )

        return results[1:]  # Skip first (self)

    def _search_annoy(
        self,
        index: Any,
        query_embedding: np.ndarray,
        node_list: List[str],
        top_k: int
    ) -> List[SimilarNode]:
        """Search using Annoy index"""
        indices = index.get_nns_by_vector(
            query_embedding,
            top_k + 1,  # +1 to skip self
            include_distances=True
        )

        results = []
        for idx, dist in zip(indices[0], indices[1]):
            if idx < len(node_list) and idx >= 0:
                # Convert distance to similarity (angular distance to cosine)
                similarity = 1.0 - (dist / np.pi)
                results.append(
                    SimilarNode(
                        node=node_list[idx],
                        similarity=float(similarity),
                        metadata={'method': 'annoy'}
                    )
                )

        return results[1:]  # Skip first (self)

    def _search_brute_force(
        self,
        query_embedding: np.ndarray,
        embeddings: Dict[str, List[float]],
        top_k: int
    ) -> List[SimilarNode]:
        """Brute-force similarity search"""
        similarities = []

        for node, embedding in embeddings.items():
            sim = self._cosine_similarity(query_embedding, np.array(embedding))
            similarities.append((node, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Convert to results
        results = [
            SimilarNode(
                node=node,
                similarity=float(sim),
                metadata={'method': 'brute_force'}
            )
            for node, sim in similarities[:top_k + 1]
        ]

        return results[1:]  # Skip first (self)

    async def _save_embeddings(self, index_name: str, embedding_index: EmbeddingIndex):
        """Save embeddings to disk"""
        try:
            cache_dir = Path(self.config.get('cache_dir', '/tmp/karateclub_embeddings'))
            cache_dir.mkdir(parents=True, exist_ok=True)

            filepath = cache_dir / f"{index_name}.pkl"

            with open(filepath, 'wb') as f:
                pickle.dump(embedding_index, f)

            logger.info(f"Saved embeddings to {filepath}")

        except Exception as e:
            logger.warning(f"Failed to save embeddings: {e}")

    async def load_embeddings(self, index_name: str) -> Optional[EmbeddingIndex]:
        """Load embeddings from disk"""
        try:
            cache_dir = Path(self.config.get('cache_dir', '/tmp/karateclub_embeddings'))
            filepath = cache_dir / f"{index_name}.pkl"

            if not filepath.exists():
                return None

            with open(filepath, 'rb') as f:
                embedding_index = pickle.load(f)

            self.indices[index_name] = embedding_index

            logger.info(f"Loaded embeddings from {filepath}")

            return embedding_index

        except Exception as e:
            logger.warning(f"Failed to load embeddings: {e}")
            return None
