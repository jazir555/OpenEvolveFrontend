"""
Graph Embedder

Graph embedding (Node2Vec) for structural similarity.

Agent: G2 (Ψ₂ Specialist)
Created: 2025-12-31
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional
from collections import defaultdict


class GraphEmbedder:
    """
    Graph embedding using Node2Vec for structural similarity.

    Generates node embeddings that preserve graph structure.
    """

    def __init__(
        self,
        dimensions: int = 64,
        walk_length: int = 40,
        num_walks: int = 20,
        p: float = 1.0,
        q: float = 1.0,
        window_size: int = 5,
        min_count: int = 1,
        workers: int = 1
    ):
        """
        Initialize graph embedder

        Args:
            dimensions: Embedding dimensionality
            walk_length: Length of random walks
            num_walks: Number of walks per node
            p: Return parameter (1.0 = unbiased)
            q: In-out parameter (1.0 = unbiased)
            window_size: Context window size for Word2Vec
            min_count: Minimum count for Word2Vec
            workers: Number of worker threads
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.min_count = min_count
        self.workers = workers

        # Node2Vec model (lazy loading)
        self.model = None

    def fit_transform(
        self,
        graph: nx.Graph,
        nodes: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Fit Node2Vec model and return embeddings

        Args:
            graph: NetworkX graph
            nodes: List of nodes to embed (default: all nodes)

        Returns:
            Dictionary mapping node -> embedding vector
        """
        if nodes is None:
            nodes = list(graph.nodes())

        # Generate random walks
        walks = self._generate_random_walks(graph, nodes)

        # Train Word2Vec on walks
        embeddings = self._train_word2vec(walks)

        return embeddings

    def _generate_random_walks(
        self,
        graph: nx.Graph,
        nodes: List[str]
    ) -> List[List[str]]:
        """
        Generate biased random walks through graph

        Args:
            graph: NetworkX graph
            nodes: List of starting nodes

        Returns:
            List of random walks (each walk is a list of node IDs)
        """
        walks = []

        for node in nodes:
            for _ in range(self.num_walks):
                walk = self._random_walk(graph, node, self.walk_length, self.p, self.q)
                walks.append(walk)

        return walks

    def _random_walk(
        self,
        graph: nx.Graph,
        start_node: str,
        length: int,
        p: float,
        q: float
    ) -> List[str]:
        """
        Generate a single biased random walk

        Args:
            graph: NetworkX graph
            start_node: Starting node
            length: Walk length
            p: Return parameter
            q: In-out parameter

        Returns:
            List of node IDs in walk
        """
        walk = [str(start_node)]
        prev_node = None
        curr_node = start_node

        for _ in range(length - 1):
            neighbors = list(graph.neighbors(curr_node))

            if not neighbors:
                break

            if prev_node is None:
                # First step: uniform random
                next_node = np.random.choice(neighbors)
            else:
                # Biased walk
                next_node = self._biased_choice(
                    graph,
                    prev_node,
                    curr_node,
                    neighbors,
                    p,
                    q
                )

            walk.append(str(next_node))
            prev_node = curr_node
            curr_node = next_node

        return walk

    def _biased_choice(
        self,
        graph: nx.Graph,
        prev_node: str,
        curr_node: str,
        neighbors: List[str],
        p: float,
        q: float
    ) -> str:
        """
        Choose next node with biased probabilities

        Args:
            graph: NetworkX graph
            prev_node: Previous node
            curr_node: Current node
            neighbors: List of neighbor nodes
            p: Return parameter
            q: In-out parameter

        Returns:
            Next node
        """
        # Compute unnormalized probabilities
        probabilities = []

        for next_node in neighbors:
            if next_node == prev_node:
                # Return to previous node
                prob = 1.0 / p
            elif graph.has_edge(prev_node, next_node):
                # Distance 1: common neighbor
                prob = 1.0
            else:
                # Distance 2: no edge to prev_node
                prob = 1.0 / q

            probabilities.append(prob)

        # Normalize
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()

        # Sample
        next_node = np.random.choice(neighbors, p=probabilities)
        return next_node

    def _train_word2vec(
        self,
        walks: List[List[str]]
    ) -> Dict[str, np.ndarray]:
        """
        Train Word2Vec on random walks

        Args:
            walks: List of random walks

        Returns:
            Dictionary mapping node -> embedding
        """
        try:
            from gensim.models import Word2Vec

            # Train Word2Vec
            model = Word2Vec(
                sentences=walks,
                vector_size=self.dimensions,
                window=self.window_size,
                min_count=self.min_count,
                workers=self.workers,
                sg=1,  # Skip-gram
                epochs=10
            )

            # Extract embeddings
            embeddings = {}
            for node in model.wv.key_to_index:
                embeddings[node] = model.wv[node]

            return embeddings

        except ImportError:
            print("WARNING: gensim not installed")
            print("Install with: pip install gensim")
            return self._fallback_embeddings(walks, self.dimensions)

    def _fallback_embeddings(
        self,
        walks: List[List[str]],
        dimensions: int
    ) -> Dict[str, np.ndarray]:
        """
        Fallback embeddings using simple node features

        Args:
            walks: Random walks
            dimensions: Embedding dimension

        Returns:
            Dictionary mapping node -> embedding
        """
        # Build co-occurrence matrix from walks
        cooccur = defaultdict(lambda: defaultdict(int))

        for walk in walks:
            for i, node1 in enumerate(walk):
                for j, node2 in enumerate(walk):
                    if i != j and abs(i - j) <= 5:  # Window size 5
                        cooccur[node1][node2] += 1

        # Create node -> index mapping
        all_nodes = list(cooccur.keys())
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}

        # Build co-occurrence matrix
        import numpy as np
        cooccur_matrix = np.zeros((len(all_nodes), len(all_nodes)))

        for node1, neighbors in cooccur.items():
            for node2, count in neighbors.items():
                i = node_to_idx[node1]
                j = node_to_idx[node2]
                cooccur_matrix[i, j] = count

        # Use SVD to get embeddings
        U, S, V = np.linalg.svd(cooccur_matrix, full_matrices=False)

        # Take top-k singular vectors
        k = min(dimensions, len(S))
        embeddings_matrix = U[:, :k] * np.sqrt(S[:k])

        # Pad to requested dimensions if needed
        if k < dimensions:
            padded = np.zeros((len(all_nodes), dimensions))
            padded[:, :k] = embeddings_matrix
            embeddings_matrix = padded

        # Convert to dictionary
        embeddings = {}
        for node, idx in node_to_idx.items():
            embeddings[node] = embeddings_matrix[idx, :]

        return embeddings

    def similarity(
        self,
        graph1: nx.Graph,
        graph2: nx.Graph,
        node1: str,
        node2: str
    ) -> float:
        """
        Compute structural similarity between two nodes

        Args:
            graph1: First graph
            graph2: Second graph
            node1: Node in first graph
            node2: Node in second graph

        Returns:
            Cosine similarity
        """
        # Get embeddings
        emb1 = self.fit_transform(graph1, [node1]).get(node1)
        emb2 = self.fit_transform(graph2, [node2]).get(node2)

        if emb1 is None or emb2 is None:
            return 0.0

        # Cosine similarity
        return self._cosine_similarity(emb1, emb2)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Cosine similarity [-1, 1]
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(v1, v2) / (norm1 * norm2)


class FallbackGraphEmbedder:
    """
    Fallback graph embedder using structural features.

    Used when Node2Vec is not available.
    """

    def __init__(self, dimensions: int = 64):
        """
        Initialize fallback embedder

        Args:
            dimensions: Embedding dimension
        """
        self.dimensions = dimensions

    def fit_transform(
        self,
        graph: nx.Graph,
        nodes: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings using structural features

        Args:
            graph: NetworkX graph
            nodes: List of nodes to embed

        Returns:
            Dictionary mapping node -> embedding
        """
        if nodes is None:
            nodes = list(graph.nodes())

        embeddings = {}

        for node in nodes:
            # Structural features
            features = self._extract_features(graph, node)
            embeddings[node] = features

        return embeddings

    def _extract_features(self, graph: nx.Graph, node: str) -> np.ndarray:
        """
        Extract structural features for node

        Args:
            graph: NetworkX graph
            node: Node to extract features for

        Returns:
            Feature vector
        """
        # Basic features
        degree = graph.degree(node)
        clustering = nx.clustering(graph, node) if nx.is_connected(graph) else 0.0

        # Neighbor features
        neighbors = list(graph.neighbors(node))
        neighbor_degrees = [graph.degree(n) for n in neighbors]
        avg_neighbor_degree = np.mean(neighbor_degrees) if neighbor_degrees else 0.0

        # Distance features (to other nodes)
        if nx.is_connected(graph):
            distances = nx.single_source_shortest_path_length(graph, node)
            avg_distance = np.mean(list(distances.values()))
            max_distance = max(distances.values())
        else:
            avg_distance = 0.0
            max_distance = 0.0

        # Centrality
        try:
            betweenness = nx.betweenness_centrality(graph, normalized=True)[node]
        except:
            betweenness = 0.0

        # Create feature vector
        features = np.array([
            degree,
            clustering,
            avg_neighbor_degree,
            avg_distance,
            max_distance,
            betweenness
        ])

        # Pad/truncate to target dimension
        if len(features) < self.dimensions:
            features = np.pad(features, (0, self.dimensions - len(features)))
        else:
            features = features[:self.dimensions]

        return features

    def similarity(
        self,
        graph1: nx.Graph,
        graph2: nx.Graph,
        node1: str,
        node2: str
    ) -> float:
        """
        Compute structural similarity between two nodes

        Args:
            graph1: First graph
            graph2: Second graph
            node1: Node in first graph
            node2: Node in second graph

        Returns:
            Cosine similarity
        """
        emb1 = self.fit_transform(graph1, [node1]).get(node1)
        emb2 = self.fit_transform(graph2, [node2]).get(node2)

        if emb1 is None or emb2 is None:
            return 0.0

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(emb1, emb2) / (norm1 * norm2)


if __name__ == "__main__":
    # Demo
    print("Graph Embedder")
    print("=" * 50)

    # Create test graphs
    G1 = nx.Graph()
    G1.add_edges_from([
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'D'),
        ('B', 'D')
    ])

    G2 = nx.Graph()
    G2.add_edges_from([
        ('W', 'X'),
        ('X', 'Y'),
        ('Y', 'Z'),
        ('X', 'Z')
    ])

    print("\nGraph 1 nodes:", list(G1.nodes()))
    print("Graph 2 nodes:", list(G2.nodes()))

    # Create embedder
    embedder = GraphEmbedder(
        dimensions=32,
        walk_length=20,
        num_walks=10
    )

    # Get embeddings
    print("\nGenerating embeddings...")
    emb1 = embedder.fit_transform(G1)
    emb2 = embedder.fit_transform(G2)

    print(f"Graph 1 embeddings: {len(emb1)} nodes")
    print(f"Graph 2 embeddings: {len(emb2)} nodes")

    # Compare nodes
    print("\nStructural similarities:")
    for node1 in ['A', 'B', 'C']:
        for node2 in ['W', 'X', 'Y']:
            sim = embedder.similarity(G1, G2, node1, node2)
            print(f"  {node1} ↔ {node2}: {sim:.3f}")

    print("\n✅ Graph Embedder working!")
