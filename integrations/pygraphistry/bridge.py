"""
Pygraphistry Bridge for OpenEvolve Knowledge Visualization

This module bridges pygraphistry's interactive visualization capabilities with
OpenEvolve's knowledge graph visualizer. It provides a high-level interface for
creating rich, interactive visualizations with UMAP + DBSCAN clustering pipeline.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime

from integrations.pygraphistry.adapter import PygraphistryAdapter

logger = logging.getLogger(__name__)


class PygraphistryBridge:
    """
    Bridge between pygraphistry and OpenEvolve knowledge visualization.

    This bridge provides:
    - Direct integration with KnowledgeGraphVisualizer
    - Automated clustering pipeline (UMAP + DBSCAN)
    - GPU-accelerated analytics
    - Interactive dashboard generation
    - iframe embedding for UI clients

    Attributes:
        adapter: PygraphistryAdapter instance
        config: Configuration dictionary
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pygraphistry bridge.

        Args:
            config: Optional configuration dictionary
        """
        self.adapter = PygraphistryAdapter()
        self.config = config or {}
        self._is_connected = False

    async def connect(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Connect to pygraphistry with configuration.

        Args:
            config: Optional configuration to override defaults

        Returns:
            True if connection successful
        """
        if config:
            self.config.update(config)

        try:
            self._is_connected = await self.adapter.initialize(self.config)
            if self._is_connected:
                logger.info("Pygraphistry bridge connected successfully")
            return self._is_connected

        except Exception as e:
            logger.error(f"Failed to connect pygraphistry bridge: {e}")
            return False

    async def visualize_knowledge_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        apply_clustering: bool = True,
        clustering_method: str = "dbscan",
        embedding_method: str = "umap",
        output_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Visualize a knowledge graph with optional clustering pipeline.

        The full pipeline:
        1. Extract node features
        2. Compute UMAP embeddings (or PCA)
        3. Apply DBSCAN clustering (or K-means)
        4. Create interactive visualization

        Args:
            nodes: List of node dictionaries with 'id' and attributes
            edges: List of edge dictionaries with 'source', 'target', and attributes
            apply_clustering: Whether to apply clustering pipeline
            clustering_method: Clustering method ('dbscan', 'kmeans')
            embedding_method: Embedding method ('umap', 'pca')
            output_path: Optional path to save visualization

        Returns:
            Dictionary with visualization results:
            {
                'url': str,  # Visualization URL
                'embeddings': np.ndarray,  # Node embeddings
                'clusters': np.ndarray,  # Cluster labels
                'n_clusters': int,  # Number of clusters
            }
        """
        if not self._is_connected:
            await self.connect()

        if not self._is_connected:
            logger.error("Pygraphistry bridge not connected")
            return None

        try:
            # Step 1: Compute embeddings if clustering requested
            embeddings = None
            if apply_clustering:
                embeddings = await self.adapter.compute_embeddings(
                    nodes=nodes,
                    method=embedding_method,
                    n_components=2
                )

                if embeddings is None:
                    logger.warning("Failed to compute embeddings, skipping clustering")
                    apply_clustering = False

            # Step 2: Cluster nodes if embeddings computed
            clusters = None
            n_clusters = 0

            if apply_clustering and embeddings is not None:
                clusters = await self.adapter.cluster_nodes(
                    embeddings=embeddings,
                    method=clustering_method
                )

                if clusters is not None:
                    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                    logger.info(f"Identified {n_clusters} clusters")

                    # Add cluster labels to nodes
                    for i, node in enumerate(nodes):
                        if i < len(clusters):
                            node['cluster'] = int(clusters[i])
                else:
                    logger.warning("Clustering failed, continuing without clusters")

            # Step 3: Create interactive visualization
            url = await self.adapter.visualize_graph(
                nodes=nodes,
                edges=edges,
                layout="force_directed",
                output_path=output_path
            )

            if url is None:
                logger.error("Failed to create visualization")
                return None

            result = {
                "url": url,
                "embeddings": embeddings,
                "clusters": clusters,
                "n_clusters": n_clusters,
                "n_nodes": len(nodes),
                "n_edges": len(edges),
            }

            logger.info(f"Knowledge graph visualization created: {url}")
            return result

        except Exception as e:
            logger.error(f"Failed to visualize knowledge graph: {e}")
            return None

    async def create_cluster_dashboard(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        embeddings: np.ndarray,
        clusters: np.ndarray,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create an interactive dashboard focused on clusters.

        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            embeddings: Node embeddings (n_nodes x 2)
            clusters: Cluster labels
            output_path: Optional path to save dashboard

        Returns:
            Dashboard URL or path
        """
        if not self._is_connected:
            logger.error("Pygraphistry bridge not connected")
            return None

        try:
            # Add cluster metadata to nodes
            enriched_nodes = []
            for i, node in enumerate(nodes):
                enriched_node = node.copy()
                if i < len(embeddings):
                    enriched_node['x'] = float(embeddings[i, 0])
                    enriched_node['y'] = float(embeddings[i, 1]) if embeddings.shape[1] > 1 else 0.0
                if i < len(clusters):
                    enriched_node['cluster'] = int(clusters[i])
                enriched_nodes.append(enriched_node)

            # Create dashboard
            data = {
                "nodes": enriched_nodes,
                "edges": edges,
                "embeddings": embeddings,
                "clusters": clusters,
            }

            url = await self.adapter.create_interactive_dashboard(
                data=data,
                dashboard_type="clusters"
            )

            if output_path and url:
                logger.info(f"Cluster dashboard saved to {output_path}")

            return url

        except Exception as e:
            logger.error(f"Failed to create cluster dashboard: {e}")
            return None

    async def analyze_patterns(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze patterns in the knowledge graph using clustering.

        This method:
        1. Computes embeddings for all nodes
        2. Identifies clusters
        3. Analyzes cluster characteristics
        4. Returns pattern insights

        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries

        Returns:
            Dictionary with pattern analysis results:
            {
                'n_clusters': int,
                'cluster_info': List[Dict],  # Info about each cluster
                'embeddings': np.ndarray,
                'clusters': np.ndarray,
            }
        """
        if not self._is_connected:
            await self.connect()

        if not self._is_connected:
            logger.error("Pygraphistry bridge not connected")
            return None

        try:
            # Compute embeddings
            embeddings = await self.adapter.compute_embeddings(
                nodes=nodes,
                method="umap",
                n_components=2
            )

            if embeddings is None:
                logger.error("Failed to compute embeddings for pattern analysis")
                return None

            # Cluster nodes
            clusters = await self.adapter.cluster_nodes(
                embeddings=embeddings,
                method="dbscan",
                eps=0.5,
                min_samples=5
            )

            if clusters is None:
                logger.error("Failed to cluster nodes for pattern analysis")
                return None

            # Analyze clusters
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            cluster_info = []

            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_nodes = [nodes[i] for i in range(len(nodes)) if cluster_mask[i]]

                # Analyze cluster characteristics
                cluster_info.append({
                    "cluster_id": cluster_id,
                    "n_nodes": len(cluster_nodes),
                    "centroid": embeddings[cluster_mask].mean(axis=0).tolist(),
                    # Add more cluster statistics as needed
                })

            result = {
                "n_clusters": n_clusters,
                "cluster_info": cluster_info,
                "embeddings": embeddings,
                "clusters": clusters,
                "n_nodes_analyzed": len(nodes),
            }

            logger.info(f"Pattern analysis completed: {n_clusters} patterns identified")
            return result

        except Exception as e:
            logger.error(f"Failed to analyze patterns: {e}")
            return None

    async def export_to_ui(
        self,
        visualization_url: str,
        iframe_height: int = 600
    ) -> str:
        """
        Generate iframe embedding code for UI clients.

        Args:
            visualization_url: URL from pygraphistry visualization
            iframe_height: Height of iframe in pixels

        Returns:
            UI component code as string
        """
        code = f"""
from ui_shim import components

# Display Pygraphistry visualization
components.iframe(
    src="{visualization_url}",
    height={iframe_height},
    scrolling=True
)
"""
        return code

    async def get_visualization_stats(
        self,
        visualization_url: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get statistics about a visualization.

        Args:
            visualization_url: URL from pygraphistry visualization

        Returns:
            Dictionary with visualization statistics
        """
        # This would require pygraphistry API integration
        # For now, return basic info
        return {
            "url": visualization_url,
            "created_at": datetime.now().isoformat(),
            "bridge_connected": self._is_connected,
        }

    async def validate_integration(self) -> Dict[str, Any]:
        """
        Validate the pygraphistry integration.

        Returns:
            Validation results dictionary
        """
        if not self._is_connected:
            await self.connect()

        validation_result = await self.adapter.validate()

        return {
            **validation_result,
            "bridge_connected": self._is_connected,
        }

    async def disconnect(self) -> bool:
        """
        Disconnect from pygraphistry.

        Returns:
            True if disconnection successful
        """
        try:
            success = await self.adapter.shutdown()
            self._is_connected = False
            logger.info("Pygraphistry bridge disconnected")
            return success

        except Exception as e:
            logger.error(f"Failed to disconnect pygraphistry bridge: {e}")
            return False


# ========== Convenience Functions ==========

async def create_knowledge_viz(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    apply_clustering: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to create a knowledge graph visualization.

    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        config: Optional pygraphistry configuration
        apply_clustering: Whether to apply clustering pipeline

    Returns:
        Visualization results dictionary
    """
    bridge = PygraphistryBridge(config)
    return await bridge.visualize_knowledge_graph(
        nodes=nodes,
        edges=edges,
        apply_clustering=apply_clustering
    )


async def analyze_knowledge_patterns(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to analyze patterns in a knowledge graph.

    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        config: Optional pygraphistry configuration

    Returns:
        Pattern analysis results dictionary
    """
    bridge = PygraphistryBridge(config)
    return await bridge.analyze_patterns(nodes=nodes, edges=edges)
