"""
Enhanced Knowledge Graph Visualizer with PyGraphistry Integration

This module extends the basic KnowledgeGraphVisualizer to include
PyGraphistry-powered interactive visualization and ML analytics capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from knowledge_graph_visualizer import KnowledgeGraphVisualizer
from integrations.pygraphistry.bridge import PygraphistryBridge


class KnowledgeGraphVisualizerWithPyGraphistry(KnowledgeGraphVisualizer):
    """
    Enhanced Knowledge Graph Visualizer with PyGraphistry integration.

    This class extends the basic KnowledgeGraphVisualizer to include
    PyGraphistry-powered interactive visualization and ML analytics capabilities
    such as GPU-accelerated graph analytics with UMAP + DBSCAN clustering pipeline.
    """

    def __init__(self, db_path: str = "./knowledge_artifacts.db", use_pygraphistry: bool = False):
        """
        Initialize the enhanced knowledge graph visualizer.

        Args:
            db_path: Path to artifact database
            use_pygraphistry: Whether to use PyGraphistry for visualization
        """
        super().__init__(db_path)
        self.use_pygraphistry = use_pygraphistry
        self.pygraphistry_bridge = None
        
        if self.use_pygraphistry:
            self.pygraphistry_bridge = PygraphistryBridge()

    def visualize_interactive(
        self,
        output_path: str = "knowledge_graph.html",
        layout: str = "spring",
        filter_node_type: Optional[str] = None,
        filter_attribute: Optional[str] = None,
        filter_value: Optional[Any] = None,
        color_by: str = "node_type",
        size_by: str = "usage_count",
        show_labels: bool = True,
        apply_clustering: bool = True,
        clustering_method: str = "dbscan",
        embedding_method: str = "umap"
    ) -> bool:
        """
        Create an interactive visualization of the knowledge graph.

        If PyGraphistry is enabled, uses PyGraphistry for advanced visualization
        with clustering and ML analytics. Otherwise, falls back to Plotly.

        Args:
            output_path: Path to save the HTML file
            layout: Layout algorithm ('spring', 'circular', 'random', 'shell')
            filter_node_type: Optional node type to filter by
            filter_attribute: Optional attribute to filter by
            filter_value: Optional value for attribute filter
            color_by: Node attribute to color nodes by
            size_by: Node attribute to size nodes by
            show_labels: Whether to show node labels
            apply_clustering: Whether to apply clustering pipeline (PyGraphistry only)
            clustering_method: Clustering method ('dbscan', 'kmeans') (PyGraphistry only)
            embedding_method: Embedding method ('umap', 'pca') (PyGraphistry only)

        Returns:
            True if visualization created successfully
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("Graph is empty. Call build_graph() first.")
            return False

        # Convert NetworkX graph to the format expected by PyGraphistry
        nodes_list = []
        edges_list = []

        # Convert nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node_dict = {"id": str(node_id)}
            node_dict.update({k: v for k, v in node_data.items() if v is not None})
            nodes_list.append(node_dict)

        # Convert edges
        for source, target, edge_data in self.graph.edges(data=True):
            edge_dict = {
                "source": str(source),
                "target": str(target)
            }
            edge_dict.update({k: v for k, v in edge_data.items() if v is not None})
            edges_list.append(edge_dict)

        if self.use_pygraphistry and self.pygraphistry_bridge:
            # Use PyGraphistry for advanced visualization
            try:
                result = self.pygraphistry_bridge.visualize_knowledge_graph(
                    nodes=nodes_list,
                    edges=edges_list,
                    apply_clustering=apply_clustering,
                    clustering_method=clustering_method,
                    embedding_method=embedding_method,
                    output_path=output_path
                )
                
                if result and 'url' in result:
                    print(f"PyGraphistry visualization saved to {output_path}")
                    return True
                else:
                    print("PyGraphistry visualization failed, falling back to Plotly...")
                    # Fall back to the original implementation
                    return super().visualize_interactive(
                        output_path=output_path,
                        layout=layout,
                        filter_node_type=filter_node_type,
                        filter_attribute=filter_attribute,
                        filter_value=filter_value,
                        color_by=color_by,
                        size_by=size_by,
                        show_labels=show_labels
                    )
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"PyGraphistry visualization failed: {e}, falling back to Plotly...")
                # Fall back to the original implementation
                return super().visualize_interactive(
                    output_path=output_path,
                    layout=layout,
                    filter_node_type=filter_node_type,
                    filter_attribute=filter_attribute,
                    filter_value=filter_value,
                    color_by=color_by,
                    size_by=size_by,
                    show_labels=show_labels
                )
        else:
            # Use the original Plotly implementation
            return super().visualize_interactive(
                output_path=output_path,
                layout=layout,
                filter_node_type=filter_node_type,
                filter_attribute=filter_attribute,
                filter_value=filter_value,
                color_by=color_by,
                size_by=size_by,
                show_labels=show_labels
            )

    async def analyze_patterns_with_pygraphistry(
        self
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze patterns in the knowledge graph using PyGraphistry's clustering capabilities.

        Returns:
            Dictionary with pattern analysis results, or None if failed
        """
        if not self.use_pygraphistry or not self.pygraphistry_bridge:
            print("PyGraphistry not enabled. Call with use_pygraphistry=True")
            return None

        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("Graph is empty. Call build_graph() first.")
            return None

        # Convert NetworkX graph to the format expected by PyGraphistry
        nodes_list = []
        edges_list = []

        # Convert nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node_dict = {"id": str(node_id)}
            node_dict.update({k: v for k, v in node_data.items() if v is not None})
            nodes_list.append(node_dict)

        # Convert edges
        for source, target, edge_data in self.graph.edges(data=True):
            edge_dict = {
                "source": str(source),
                "target": str(target)
            }
            edge_dict.update({k: v for k, v in edge_data.items() if v is not None})
            edges_list.append(edge_dict)

        try:
            result = await self.pygraphistry_bridge.analyze_patterns(
                nodes=nodes_list,
                edges=edges_list
            )
            return result
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"Pattern analysis with PyGraphistry failed: {e}")
            return None

    async def connect_pygraphistry(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Connect to PyGraphistry with the provided configuration.

        Args:
            config: Optional configuration for PyGraphistry connection

        Returns:
            True if connection successful
        """
        if self.pygraphistry_bridge:
            return await self.pygraphistry_bridge.connect(config)
        return False


# ========== Convenience Functions ==========

def create_enhanced_visualizer(
    db_path: str = "./knowledge_artifacts.db",
    use_pygraphistry: bool = True,
    output_path: str = "knowledge_graph.html",
    max_nodes: int = 500
) -> bool:
    """
    Convenience function to create a knowledge graph visualization with PyGraphistry.

    Args:
        db_path: Path to artifact database
        use_pygraphistry: Whether to use PyGraphistry for visualization
        output_path: Path to save the HTML file
        max_nodes: Maximum number of nodes to include

    Returns:
        True if visualization created successfully
    """
    visualizer = KnowledgeGraphVisualizerWithPyGraphistry(
        db_path=db_path,
        use_pygraphistry=use_pygraphistry
    )
    
    visualizer.build_graph(max_nodes=max_nodes)
    return visualizer.visualize_interactive(output_path)


async def analyze_knowledge_patterns(
    db_path: str = "./knowledge_artifacts.db",
    config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to analyze knowledge patterns using PyGraphistry.

    Args:
        db_path: Path to artifact database
        config: Optional PyGraphistry configuration

    Returns:
        Pattern analysis results dictionary
    """
    visualizer = KnowledgeGraphVisualizerWithPyGraphistry(
        db_path=db_path,
        use_pygraphistry=True
    )
    
    if config:
        await visualizer.connect_pygraphistry(config)
    
    visualizer.build_graph()
    return await visualizer.analyze_patterns_with_pygraphistry()