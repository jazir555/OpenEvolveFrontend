"""
Base Visualization Interface

Abstract interface for graph visualization and ML analytics integrations.
This provides a common contract for all visualization adapters (pygraphistry, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class VisualizationInterface(ABC):
    """
    Abstract interface for visualization systems.

    All visualization adapters must implement these methods to ensure
    compatibility with OpenEvolve's knowledge extraction systems.
    """

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the visualization system with configuration.

        Args:
            config: Configuration dictionary (API keys, GPU settings, etc.)

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    async def visualize_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        layout: str = "force_directed",
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Visualize a graph with nodes and edges.

        Args:
            nodes: List of node dictionaries with 'id' and attributes
            edges: List of edge dictionaries with 'source', 'target', and attributes
            layout: Layout algorithm ('force_directed', 'circular', 'hierarchical')
            output_path: Optional path to save visualization

        Returns:
            URL or path to visualization, or None if failed
        """
        pass

    @abstractmethod
    async def compute_embeddings(
        self,
        nodes: List[Dict[str, Any]],
        method: str = "umap",
        n_components: int = 2
    ) -> Optional[np.ndarray]:
        """
        Compute node embeddings for dimensionality reduction.

        Args:
            nodes: List of node dictionaries with features
            method: Embedding method ('umap', 'pca', 'tsne')
            n_components: Number of dimensions for output

        Returns:
            Embedding array (n_nodes x n_components)
        """
        pass

    @abstractmethod
    async def cluster_nodes(
        self,
        embeddings: np.ndarray,
        method: str = "dbscan",
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Cluster nodes based on embeddings.

        Args:
            embeddings: Node embeddings (n_nodes x n_features)
            method: Clustering method ('dbscan', 'kmeans', 'hierarchical')
            **kwargs: Additional parameters for clustering algorithm

        Returns:
            Cluster labels array (n_nodes,)
        """
        pass

    @abstractmethod
    async def create_interactive_dashboard(
        self,
        data: Dict[str, Any],
        dashboard_type: str = "graph"
    ) -> Optional[str]:
        """
        Create an interactive dashboard for data exploration.

        Args:
            data: Data dictionary with nodes, edges, embeddings, etc.
            dashboard_type: Type of dashboard ('graph', 'clusters', 'patterns')

        Returns:
            URL or path to dashboard
        """
        pass

    @abstractmethod
    async def validate(self) -> Dict[str, Any]:
        """
        Validate the integration is working correctly.

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'version': str,
                'capabilities': List[str],
                'errors': List[str]
            }
        """
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Cleanly shutdown the visualization system.

        Returns:
            True if shutdown successful
        """
        pass

    async def compute_graph_statistics(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute basic graph statistics.

        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries

        Returns:
            Dictionary with graph statistics
        """
        stats = {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "avg_degree": 0.0,
            "density": 0.0,
        }

        if len(nodes) > 0:
            stats["avg_degree"] = (2 * len(edges)) / len(nodes)

        return stats

    async def filter_nodes(
        self,
        nodes: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter nodes based on attribute criteria.

        Args:
            nodes: List of node dictionaries
            filters: Filter criteria (e.g., {'node_type': 'pattern', 'complexity': {'$gt': 5}})

        Returns:
            Filtered list of nodes
        """
        filtered = []

        for node in nodes:
            match = True
            for key, value in filters.items():
                if key not in node:
                    match = False
                    break

                if isinstance(value, dict):
                    # Handle operators
                    for op, op_val in value.items():
                        if op == '$gt' and not node[key] > op_val:
                            match = False
                            break
                        elif op == '$lt' and not node[key] < op_val:
                            match = False
                            break
                        elif op == '$eq' and node[key] != op_val:
                            match = False
                            break
                        elif op == '$ne' and node[key] == op_val:
                            match = False
                            break
                elif node[key] != value:
                    match = False
                    break

            if match:
                filtered.append(node)

        return filtered

    async def enrich_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        enrichments: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Enrich graph nodes and edges with computed attributes.

        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            enrichments: Dictionary specifying enrichments to compute

        Returns:
            Tuple of (enriched_nodes, enriched_edges)
        """
        # Compute node degrees
        if enrichments.get('compute_degree', False):
            node_degree = {}
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                node_degree[source] = node_degree.get(source, 0) + 1
                node_degree[target] = node_degree.get(target, 0) + 1

            for node in nodes:
                node_id = node.get('id')
                node['degree'] = node_degree.get(node_id, 0)

        # Compute node centrality (simplified)
        if enrichments.get('compute_centrality', False):
            max_degree = max((node.get('degree', 0) for node in nodes), default=1)
            for node in nodes:
                degree = node.get('degree', 0)
                node['centrality'] = degree / max_degree if max_degree > 0 else 0

        return nodes, edges
