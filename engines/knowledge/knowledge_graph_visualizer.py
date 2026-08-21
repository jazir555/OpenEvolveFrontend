"""
Knowledge Graph Visualizer - Stage 6 Knowledge Extraction

This module creates interactive visualizations of knowledge artifacts.
It uses NetworkX for graph operations and Plotly for interactive visualization.
"""
from __future__ import annotations


import time
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import numpy as np
from collections import defaultdict

try:
    from workflow_structures import (
        SolutionPatternArtifact,
        TeamPerformanceArtifact,
        GauntletEffectivenessArtifact,
        KnowledgeArtifact,
    )
    from workflow_knowledge_extractor import KnowledgeArtifactManager
except ImportError as e:
    print(f"Warning: Could not import from workflow_structures or workflow_knowledge_extractor: {e}")
    # Define minimal classes for fallback
    SolutionPatternArtifact = object
    TeamPerformanceArtifact = object
    GauntletEffectivenessArtifact = object
    KnowledgeArtifact = object
    KnowledgeArtifactManager = object

class KnowledgeGraphVisualizer:
    """
    Visualizes knowledge artifacts as an interactive graph.

    Features:
    - Interactive graph visualization with Plotly
    - Node filtering by attributes
    - Community detection
    - Path finding and subgraph extraction
    - Export to multiple formats (HTML, JSON, Graphviz)
    - Optional PyGraphistry integration for advanced visualization

    Attributes:
        artifact_manager: Manager for accessing artifacts
        graph: NetworkX graph structure
        use_pygraphistry: Whether to use PyGraphistry for visualization
        pygraphistry_bridge: Bridge to PyGraphistry functionality
    """

    def __init__(self, db_path: str = "./knowledge_artifacts.db", use_pygraphistry: bool = False):
        """
        Initialize the knowledge graph visualizer.

        Args:
            db_path: Path to artifact database
            use_pygraphistry: Whether to use PyGraphistry for visualization
        """
        self.artifact_manager = KnowledgeArtifactManager(db_path)
        self.node_positions = None
        self.use_pygraphistry = use_pygraphistry
        self.pygraphistry_bridge = None

        # Initialize empty graph
        try:
            import networkx as nx
            self.graph = nx.DiGraph()
        except ImportError:
            print("networkx not available. Install with: pip install networkx")
            self.graph = None

        if self.use_pygraphistry:
            try:
                from integrations.pygraphistry.bridge import PygraphistryBridge
                self.pygraphistry_bridge = PygraphistryBridge()
            except ImportError:
                print("PyGraphistry not available. Install with: pip install graphistry")
                self.use_pygraphistry = False

    def build_graph(
        self,
        include_solution_patterns: bool = True,
        include_team_performance: bool = True,
        include_gauntlet_effectiveness: bool = True,
        max_nodes: int = 1000
    ) -> Dict[str, Any]:
        """
        Build a knowledge graph from artifacts.

        Args:
            include_solution_patterns: Include solution pattern artifacts
            include_team_performance: Include team performance artifacts
            include_gauntlet_effectiveness: Include gauntlet effectiveness artifacts
            max_nodes: Maximum number of nodes to include

        Returns:
            Dictionary with graph statistics
        """
        if self.graph is None:
            return {"status": "error", "message": "networkx not available"}

        try:
            import networkx as nx
        except ImportError:
            return {"status": "error", "message": "networkx not installed"}

        # Clear existing graph and create new one
        self.graph.clear()

        node_count = 0

        # Add solution patterns
        if include_solution_patterns:
            patterns = self.artifact_manager.list_solution_patterns(limit=max_nodes // 3)
            for pattern in patterns:
                if node_count >= max_nodes:
                    break

                self.graph.add_node(
                    pattern.artifact_id,
                    node_type="solution_pattern",
                    domain=pattern.domain,
                    complexity=pattern.complexity,
                    success_rate=pattern.success_rate,
                    confidence=pattern.confidence,
                    usage_count=pattern.usage_count,
                    decomposition_strategy=pattern.decomposition_strategy,
                )
                node_count += 1

                # Add edges to related patterns
                for related_id in pattern.related_patterns:
                    self.graph.add_edge(pattern.artifact_id, related_id, relationship="related")

        # Add team performance
        if include_team_performance:
            team_artifacts = self.artifact_manager.list_team_performance(limit=max_nodes // 3)
            for artifact in team_artifacts:
                if node_count >= max_nodes:
                    break

                self.graph.add_node(
                    artifact.artifact_id,
                    node_type="team_performance",
                    team_id=artifact.team_id,
                    velocity=artifact.velocity,
                    confidence=artifact.confidence,
                )
                node_count += 1

        # Add gauntlet effectiveness
        if include_gauntlet_effectiveness:
            gauntlet_artifacts = self.artifact_manager.list_gauntlet_effectiveness(limit=max_nodes // 3)
            for artifact in gauntlet_artifacts:
                if node_count >= max_nodes:
                    break

                self.graph.add_node(
                    artifact.artifact_id,
                    node_type="gauntlet_effectiveness",
                    gauntlet_id=artifact.gauntlet_id,
                    gauntlet_type=artifact.gauntlet_type,
                    catch_rate=artifact.catch_rate,
                    effectiveness_score=artifact.get_effectiveness_score(),
                )
                node_count += 1

        # Add cross-artifact relationships
        self._add_cross_relationships()

        # Calculate graph statistics
        stats = {
            "status": "success",
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph) if not self.graph.is_directed() else nx.is_weakly_connected(self.graph),
        }

        if stats["nodes"] > 0:
            stats["avg_degree"] = sum(dict(self.graph.degree()).values()) / stats["nodes"]

        return stats

    def _add_cross_relationships(self):
        """Add relationships between different artifact types."""
        # Connect patterns from the same workflow
        patterns_by_workflow = defaultdict(list)
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("node_type") == "solution_pattern":
                source_workflow = node_id.split("_")[1] if "_" in node_id else ""
                if source_workflow:
                    patterns_by_workflow[source_workflow].append(node_id)

        for workflow_id, pattern_ids in patterns_by_workflow.items():
            if len(pattern_ids) > 1:
                for i, id1 in enumerate(pattern_ids):
                    for id2 in pattern_ids[i+1:]:
                        if not self.graph.has_edge(id1, id2):
                            self.graph.add_edge(id1, id2, relationship="same_workflow")

        # Connect team performance to patterns they solved
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("node_type") == "team_performance":
                team_id = node_data.get("team_id", "")
                # Connect to patterns this team might have created
                for pattern_id in self.graph.nodes():
                    if self.graph.nodes[pattern_id].get("node_type") == "solution_pattern":
                        # Heuristic: connect if pattern was created around the same time
                        # In a real system, this would use explicit metadata
                        pass

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
                # Since PyGraphistryBridge methods are async, we need to handle this carefully
                # If we're in an async context, we should await; otherwise, we need to run in a new loop
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    # We're already in an async context, so we can't use run_until_complete
                    print("PyGraphistry visualization skipped: already in async context")
                    # Fall back to the original implementation
                    return self._visualize_with_plotly(
                        output_path=output_path,
                        layout=layout,
                        filter_node_type=filter_node_type,
                        filter_attribute=filter_attribute,
                        filter_value=filter_value,
                        color_by=color_by,
                        size_by=size_by,
                        show_labels=show_labels
                    )
                except RuntimeError:
                    # No running loop, we can create one
                    result = asyncio.run(
                        self.pygraphistry_bridge.visualize_knowledge_graph(
                            nodes=nodes_list,
                            edges=edges_list,
                            apply_clustering=apply_clustering,
                            clustering_method=clustering_method,
                            embedding_method=embedding_method,
                            output_path=output_path
                        )
                    )

                    if result and 'url' in result:
                        print(f"PyGraphistry visualization saved to {output_path}")
                        return True
                    else:
                        print("PyGraphistry visualization failed, falling back to Plotly...")
                        # Fall back to the original implementation
                        return self._visualize_with_plotly(
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
                return self._visualize_with_plotly(
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
            return self._visualize_with_plotly(
                output_path=output_path,
                layout=layout,
                filter_node_type=filter_node_type,
                filter_attribute=filter_attribute,
                filter_value=filter_value,
                color_by=color_by,
                size_by=size_by,
                show_labels=show_labels
            )

    def _visualize_with_plotly(
        self,
        output_path: str = "knowledge_graph.html",
        layout: str = "spring",
        filter_node_type: Optional[str] = None,
        filter_attribute: Optional[str] = None,
        filter_value: Optional[Any] = None,
        color_by: str = "node_type",
        size_by: str = "usage_count",
        show_labels: bool = True
    ) -> bool:
        """
        Create an interactive visualization of the knowledge graph using Plotly.

        Args:
            output_path: Path to save the HTML file
            layout: Layout algorithm ('spring', 'circular', 'random', 'shell')
            filter_node_type: Optional node type to filter by
            filter_attribute: Optional attribute to filter by
            filter_value: Optional value for attribute filter
            color_by: Node attribute to color nodes by
            size_by: Node attribute to size nodes by
            show_labels: Whether to show node labels

        Returns:
            True if visualization created successfully
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("Graph is empty. Call build_graph() first.")
            return False

        try:
            import plotly.graph_objects as go
        except ImportError:
            print("plotly not available. Cannot create visualization.")
            return False

        # Filter nodes if requested
        nodes_to_show = list(self.graph.nodes())
        if filter_node_type:
            nodes_to_show = [
                n for n in nodes_to_show
                if self.graph.nodes[n].get("node_type") == filter_node_type
            ]
        if filter_attribute and filter_value is not None:
            nodes_to_show = [
                n for n in nodes_to_show
                if self.graph.nodes[n].get(filter_attribute) == filter_value
            ]

        # Create subgraph
        subgraph = self.graph.subgraph(nodes_to_show)

        # Calculate layout
        if self.node_positions is None or len(self.node_positions) != len(nodes_to_show):
            self.node_positions = self._calculate_layout(subgraph, layout)

        # Extract node and edge data
        node_trace = self._create_node_trace(subgraph, nodes_to_show, color_by, size_by, show_labels)
        edge_trace = self._create_edge_trace(subgraph, nodes_to_show)

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title="Knowledge Graph Visualization",
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           annotations=[
                               dict(
                                   text=f"Nodes: {len(nodes_to_show)}, Edges: {subgraph.number_of_edges()}",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002,
                                   xanchor='left', yanchor='bottom',
                                   font=dict(size=12)
                               )
                           ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))

        # Save to HTML
        fig.write_html(output_path)
        print(f"Interactive visualization saved to {output_path}")
        return True

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

    def _calculate_layout(self, graph, layout: str = "spring") -> Dict[str, Tuple[float, float]]:
        """Calculate node positions for layout."""
        import networkx as nx

        if layout == "spring":
            pos = nx.spring_layout(graph, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "random":
            pos = nx.random_layout(graph)
        elif layout == "shell":
            pos = nx.shell_layout(graph)
        else:
            pos = nx.spring_layout(graph)

        return pos

    def _create_node_trace(self, graph, nodes_to_show: List[str], color_by: str, size_by: str, show_labels: bool):
        """Create node trace for Plotly."""
        import plotly.graph_objects as go

        # Get positions
        x = []
        y = []
        for node in nodes_to_show:
            pos = self.node_positions.get(node, (0, 0))
            x.append(pos[0])
            y.append(pos[1])

        # Get colors
        node_colors = []
        for node in nodes_to_show:
            value = graph.nodes[node].get(color_by, "unknown")
            node_colors.append(str(value))

        # Get sizes
        node_sizes = []
        for node in nodes_to_show:
            value = graph.nodes[node].get(size_by, 1)
            # Normalize size
            if isinstance(value, (int, float)):
                size = max(10, min(50, value * 10))
            else:
                size = 20
            node_sizes.append(size)

        # Create hover text
        hover_text = []
        for node in nodes_to_show:
            node_data = graph.nodes[node]
            hover = f"ID: {node}<br>"
            hover += f"Type: {node_data.get('node_type', 'unknown')}<br>"
            for key, value in node_data.items():
                if key != "node_type" and not key.startswith("_"):
                    hover += f"{key}: {value}<br>"
            hover_text.append(hover)

        # Create trace
        node_trace = go.Scatter(
            x=x, y=y,
            mode='markers+text' if show_labels else 'markers',
            text=nodes_to_show if show_labels else None,
            textposition="bottom center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_by),
                line=dict(width=1, color='#888')
            ),
            hovertext=hover_text,
            hoverinfo='text'
        )

        return node_trace

    def _create_edge_trace(self, graph, nodes_to_show: List[str]):
        """Create edge trace for Plotly."""
        import plotly.graph_objects as go

        x_edges = []
        y_edges = []

        for edge in graph.edges():
            if edge[0] in nodes_to_show and edge[1] in nodes_to_show:
                pos0 = self.node_positions.get(edge[0], (0, 0))
                pos1 = self.node_positions.get(edge[1], (0, 0))
                x_edges.extend([pos0[0], pos1[0], None])
                y_edges.extend([pos0[1], pos1[1], None])

        edge_trace = go.Scatter(
            x=x_edges, y=y_edges,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none'
        )

        return edge_trace

    def detect_communities(self, method: str = "louvain") -> Dict[str, List[str]]:
        """
        Detect communities in the knowledge graph.

        Args:
            method: Community detection method ('louvain', 'label_propagation')

        Returns:
            Dictionary mapping community IDs to node lists
        """
        if self.graph is None:
            return {}

        try:
            import networkx as nx
        except ImportError:
            return {}

        communities = defaultdict(list)

        if method == "louvain":
            try:
                import networkx.algorithms.community as nx_comm
                communities_generator = nx_comm.louvain_communities(self.graph.to_undirected())
                for i, community in enumerate(communities_generator):
                    communities[f"community_{i}"] = list(community)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                # Fallback to label propagation
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error: {e}", exc_info=True)
                method = "label_propagation"

        if method == "label_propagation":
            try:
                labels = nx.algorithms.community.label_propagation_communities(self.graph.to_undirected())
                for i, community in enumerate(labels):
                    communities[f"community_{i}"] = list(community)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in knowledge_graph_visualizer.py: {e}", exc_info=True)
                raise

        return dict(communities)

    def find_communities(self, method: str = "louvain") -> Dict[str, List[str]]:
        """
        Find communities in the knowledge graph.

        This is an alias for detect_communities() for MASTER_TASKLIST compatibility.

        Args:
            method: Community detection method ('louvain', 'label_propagation')

        Returns:
            Dictionary mapping community IDs to node lists
        """
        return self.detect_communities(method)

    def find_shortest_path(
        self,
        source: str,
        target: str,
        relationship_type: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            relationship_type: Optional relationship type to filter by

        Returns:
            List of node IDs in the path, or None if no path exists
        """
        if self.graph is None:
            return None

        try:
            import networkx as nx
        except ImportError:
            return None

        # Filter edges by relationship type if specified
        if relationship_type:
            subgraph = nx.DiGraph()
            for u, v, data in self.graph.edges(data=True):
                if data.get("relationship") == relationship_type:
                    subgraph.add_edge(u, v)
        else:
            subgraph = self.graph

        try:
            path = nx.shortest_path(subgraph, source, target)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def extract_subgraph(
        self,
        center_node: str,
        radius: int = 1,
        node_type: Optional[str] = None
    ) -> Optional['KnowledgeGraphVisualizer']:
        """
        Extract a subgraph around a central node.

        Args:
            center_node: ID of the central node
            radius: Radius of the subgraph (in hops)
            node_type: Optional node type filter

        Returns:
            New KnowledgeGraphVisualizer with the subgraph
        """
        if self.graph is None:
            return None

        try:
            import networkx as nx
        except ImportError:
            return None

        # Get nodes within radius
        subgraph_nodes = {center_node}
        current_layer = {center_node}

        for _ in range(radius):
            next_layer = set()
            for node in current_layer:
                neighbors = set(self.graph.neighbors(node))
                next_layer.update(neighbors)
            subgraph_nodes.update(next_layer)
            current_layer = next_layer

        # Filter by node type if specified
        if node_type:
            subgraph_nodes = {
                n for n in subgraph_nodes
                if self.graph.nodes[n].get("node_type") == node_type
            }

        # Create subgraph
        subgraph = self.graph.subgraph(subgraph_nodes)

        # Create new visualizer with subgraph
        new_viz = KnowledgeGraphVisualizer()
        new_viz.graph = subgraph

        return new_viz

    def export_to_json(self, output_path: str) -> bool:
        """
        Export graph to JSON format.

        Args:
            output_path: Path to save the JSON file

        Returns:
            True if export successful
        """
        if self.graph is None:
            return False

        try:
            import networkx as nx
        except ImportError:
            return False

        # Convert to JSON-serializable format
        from networkx.readwrite import json_graph

        graph_data = json_graph.node_link_data(self.graph)

        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)

        print(f"Graph exported to JSON: {output_path}")
        return True

    def export_to_graphviz(self, output_path: str) -> bool:
        """
        Export graph to Graphviz DOT format.

        Args:
            output_path: Path to save the DOT file

        Returns:
            True if export successful
        """
        if self.graph is None:
            return False

        try:
            import networkx as nx
        except ImportError:
            return False

        # Write to DOT format
        nx.drawing.nx_pydot.write_dot(self.graph, output_path)
        print(f"Graph exported to Graphviz DOT: {output_path}")
        return True

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the graph.

        Returns:
            Dictionary with graph statistics
        """
        if self.graph is None:
            return {}

        try:
            import networkx as nx
        except ImportError:
            return {}

        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
        }

        # Node type distribution
        node_types = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            node_types[data.get("node_type", "unknown")] += 1
        stats["node_type_distribution"] = dict(node_types)

        # Degree statistics
        degrees = [d for n, d in self.graph.degree()]
        if degrees:
            stats["avg_degree"] = np.mean(degrees)
            stats["max_degree"] = max(degrees)
            stats["min_degree"] = min(degrees)

        # Connected components
        stats["connected_components"] = nx.number_weakly_connected_components(self.graph)

        # If undirected
        undirected = self.graph.to_undirected()
        if nx.is_connected(undirected):
            stats["avg_shortest_path_length"] = nx.average_shortest_path_length(undirected)
            stats["diameter"] = nx.diameter(undirected)

        return stats


# ========== Convenience Functions ==========

def visualize_knowledge_graph(
    db_path: str = "./knowledge_artifacts.db",
    output_path: str = "knowledge_graph.html",
    max_nodes: int = 500,
    use_pygraphistry: bool = False
) -> bool:
    """
    Convenience function to create a knowledge graph visualization.

    Args:
        db_path: Path to artifact database
        output_path: Path to save the HTML file
        max_nodes: Maximum number of nodes to include
        use_pygraphistry: Whether to use PyGraphistry for visualization

    Returns:
        True if visualization created successfully
    """
    visualizer = KnowledgeGraphVisualizer(db_path, use_pygraphistry=use_pygraphistry)
    visualizer.build_graph(max_nodes=max_nodes)
    return visualizer.visualize_interactive(output_path)

def export_knowledge_graph(
    db_path: str = "./knowledge_artifacts.db",
    output_path: str = "knowledge_graph.json",
    format: str = "json"
) -> bool:
    """
    Convenience function to export knowledge graph.

    Args:
        db_path: Path to artifact database
        output_path: Path to save the file
        format: Export format ('json' or 'graphviz')

    Returns:
        True if export successful
    """
    visualizer = KnowledgeGraphVisualizer(db_path)
    visualizer.build_graph()

    if format == "json":
        return visualizer.export_to_json(output_path)
    elif format == "graphviz":
        return visualizer.export_to_graphviz(output_path)
    else:
        print(f"Unknown format: {format}")
        return False

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
    visualizer = KnowledgeGraphVisualizer(db_path, use_pygraphistry=True)

    if config:
        await visualizer.connect_pygraphistry(config)

    visualizer.build_graph()
    return await visualizer.analyze_patterns_with_pygraphistry()
