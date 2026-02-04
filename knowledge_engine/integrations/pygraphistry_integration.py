"""
PyGraphistry Integration Module for OpenEvolve Knowledge Engine

This module provides advanced graph visualization and analysis capabilities by 
integrating PyGraphistry's GPU-accelerated graph visualization without modifying 
any core files.

Business Logic:
    - Visualize knowledge graphs with interactive, GPU-accelerated rendering
    - Analyze graph structure and compute metrics
    - Detect communities and patterns visually
    - Generate visual reports of knowledge extraction results
    - Integrate with Neo4j/Memgraph for database visualization

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

import sys
import os
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

# Add PyGraphistry to Python path for import
pygraphistry_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
    'pygraphistry'
)
if pygraphistry_path not in sys.path:
    sys.path.insert(0, pygraphistry_path)


class LayoutAlgorithm(Enum):
    """Available layout algorithms for graph visualization."""
    FORCE_ATLAS2 = "force_atlas2"
    CIRCULAR = "circular"
    HIERARCHICAL = "hierarchical"
    GRID = "grid"
    RANDOM = "random"
    SPRING = "spring"


class ColorScheme(Enum):
    """Available color schemes for node/edge coloring."""
    CATEGORY = "categorical"  # For categorical data
    SEQUENTIAL = "sequential"  # For sequential/numeric data
    DIVERGING = "diverging"  # For diverging data
    CUSTOM = "custom"  # User-defined colors


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""
    layout: LayoutAlgorithm = LayoutAlgorithm.FORCE_ATLAS2
    color_scheme: ColorScheme = ColorScheme.CATEGORY
    node_size_column: Optional[str] = None
    edge_weight_column: Optional[str] = None
    node_color_column: Optional[str] = None
    edge_color_column: Optional[str] = None
    width: int = 800
    height: int = 600
    title: Optional[str] = None
    description: Optional[str] = None
    background_color: str = "#ffffff"
    show_labels: bool = True
    label_column: Optional[str] = None
    hover_columns: List[str] = field(default_factory=list)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphMetrics:
    """Graph analytics metrics."""
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    max_degree: int = 0
    clustering_coefficient: float = 0.0
    connected_components: int = 0
    diameter: Optional[int] = None
    avg_shortest_path: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_count': self.node_count,
            'edge_count': self.edge_count,
            'density': self.density,
            'avg_degree': self.avg_degree,
            'max_degree': self.max_degree,
            'clustering_coefficient': self.clustering_coefficient,
            'connected_components': self.connected_components,
            'diameter': self.diameter,
            'avg_shortest_path': self.avg_shortest_path
        }


@dataclass
class VisualizationResult:
    """Result of a visualization operation."""
    status: str
    url: Optional[str] = None
    html: Optional[str] = None
    iframe: Optional[str] = None
    metrics: Optional[GraphMetrics] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'url': self.url,
            'html': self.html,
            'iframe': self.iframe,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class PyGraphistryIntegration:
    """
    Main PyGraphistry Integration class for the Knowledge Engine.
    
    Provides GPU-accelerated graph visualization and analytics capabilities.
    
    Business Capabilities:
        1. Visualize knowledge graphs from various sources
        2. Generate interactive dashboards for graph exploration
        3. Compute and display graph metrics
        4. Export visualizations as HTML/embeddable iframes
        5. Integrate with Neo4j/Memgraph for live database visualization
    
    Example:
        >>> pg = PyGraphistryIntegration()
        >>> result = pg.visualize_knowledge_graph(nodes, edges)
        >>> print(result.url)  # Open in browser
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PyGraphistry Integration.
        
        Args:
            api_key: PyGraphistry API key (or from environment variable)
            config: Configuration dictionary
        """
        self.config = config or {}
        self._api_key = api_key or os.environ.get('GRAPHISTRY_API_KEY')
        self._visualizer = GraphistryVisualizer(api_key=self._api_key)
        self._analyzer = GraphAnalyzer()
        self._initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize PyGraphistry with API key.
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
            
        if not self._api_key:
            logger.warning("No PyGraphistry API key provided. Visualizations will be limited.")
            
        self._initialized = self._visualizer.initialize()
        return self._initialized
    
    def is_available(self) -> bool:
        """Check if PyGraphistry is available."""
        return self._visualizer.is_available()
    
    def visualize_knowledge_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        config: Optional[VisualizationConfig] = None,
        output_format: str = 'url'
    ) -> VisualizationResult:
        """
        Visualize a knowledge graph.
        
        Args:
            nodes: List of node dictionaries with at least 'id' key
            edges: List of edge dictionaries with 'source' and 'target' keys
            config: Visualization configuration
            output_format: 'url', 'html', 'iframe', or 'all'
            
        Returns:
            VisualizationResult with URL or HTML content
            
        Example:
            >>> nodes = [{'id': 'A', 'label': 'Entity A', 'type': 'PERSON'}]
            >>> edges = [{'source': 'A', 'target': 'B', 'relation': 'KNOWS'}]
            >>> result = pg.visualize_knowledge_graph(nodes, edges)
        """
        if not self.initialize():
            return VisualizationResult(
                status='error',
                error_message='PyGraphistry not initialized'
            )
            
        return self._visualizer.visualize(nodes, edges, config, output_format)
    
    def visualize_from_neo4j(
        self,
        query: str,
        connection_params: Dict[str, str],
        config: Optional[VisualizationConfig] = None
    ) -> VisualizationResult:
        """
        Visualize graph data directly from Neo4j/Memgraph.
        
        Args:
            query: Cypher query returning nodes and relationships
            connection_params: Connection parameters (uri, username, password)
            config: Visualization configuration
            
        Returns:
            VisualizationResult
        """
        if not self.initialize():
            return VisualizationResult(
                status='error',
                error_message='PyGraphistry not initialized'
            )
            
        try:
            # Import Neo4j driver
            from neo4j import GraphDatabase
            
            # Connect and run query
            driver = GraphDatabase.driver(
                connection_params.get('uri', 'bolt://localhost:7687'),
                auth=(
                    connection_params.get('username', 'neo4j'),
                    connection_params.get('password', '')
                )
            )
            
            nodes = []
            edges = []
            
            with driver.session() as session:
                result = session.run(query)
                
                for record in result:
                    # Process nodes
                    if 'nodes' in record:
                        for node in record['nodes']:
                            node_dict = {
                                'id': str(node.id),
                                'labels': list(node.labels),
                                **dict(node)
                            }
                            nodes.append(node_dict)
                    
                    # Process relationships
                    if 'relationships' in record:
                        for rel in record['relationships']:
                            edge_dict = {
                                'source': str(rel.start_node.id),
                                'target': str(rel.end_node.id),
                                'type': rel.type,
                                **dict(rel)
                            }
                            edges.append(edge_dict)
            
            driver.close()
            
            # Visualize extracted data
            return self._visualizer.visualize(nodes, edges, config)
            
        except Exception as e:
            logger.error(f"Neo4j visualization failed: {e}")
            return VisualizationResult(
                status='error',
                error_message=f'Neo4j visualization failed: {str(e)}'
            )
    
    def analyze_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        compute_paths: bool = False
    ) -> GraphMetrics:
        """
        Compute graph analytics metrics.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            compute_paths: Whether to compute path-based metrics (slower)
            
        Returns:
            GraphMetrics object
        """
        return self._analyzer.analyze(nodes, edges, compute_paths)
    
    def detect_communities(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        algorithm: str = 'louvain'
    ) -> Dict[str, Any]:
        """
        Detect communities in the graph.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            algorithm: Community detection algorithm
            
        Returns:
            Dictionary with community assignments
        """
        return self._analyzer.detect_communities(nodes, edges, algorithm)
    
    def create_visualization_dashboard(
        self,
        graphs: List[Dict[str, Any]],
        dashboard_title: str = "Knowledge Graph Dashboard"
    ) -> Dict[str, Any]:
        """
        Create a multi-view visualization dashboard.
        
        Args:
            graphs: List of graph dictionaries with 'nodes', 'edges', 'title'
            dashboard_title: Overall dashboard title
            
        Returns:
            Dictionary with dashboard configuration and URLs
        """
        if not self.initialize():
            return {'status': 'error', 'message': 'PyGraphistry not initialized'}
        
        dashboard = {
            'title': dashboard_title,
            'created_at': datetime.now().isoformat(),
            'visualizations': []
        }
        
        for i, graph_data in enumerate(graphs):
            result = self._visualizer.visualize(
                graph_data['nodes'],
                graph_data['edges'],
                config=graph_data.get('config'),
                output_format='url'
            )
            
            dashboard['visualizations'].append({
                'index': i,
                'title': graph_data.get('title', f'Graph {i+1}'),
                'url': result.url,
                'metrics': result.metrics.to_dict() if result.metrics else None
            })
        
        dashboard['status'] = 'success'
        return dashboard
    
    def export_to_html(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        filepath: str,
        config: Optional[VisualizationConfig] = None
    ) -> Dict[str, Any]:
        """
        Export visualization to HTML file.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            filepath: Path to save HTML file
            config: Visualization configuration
            
        Returns:
            Dictionary with export status
        """
        result = self._visualizer.visualize(nodes, edges, config, output_format='html')
        
        if result.status == 'success' and result.html:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result.html)
                return {
                    'status': 'success',
                    'filepath': filepath,
                    'size_bytes': len(result.html)
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Failed to write file: {str(e)}'
                }
        
        return {
            'status': 'error',
            'message': result.error_message or 'Failed to generate HTML'
        }
    
    def get_visualization_settings(self) -> Dict[str, Any]:
        """
        Get available visualization settings and options.
        
        Returns:
            Dictionary with settings information
        """
        return {
            'layout_algorithms': [alg.value for alg in LayoutAlgorithm],
            'color_schemes': [scheme.value for scheme in ColorScheme],
            'available': self.is_available(),
            'api_configured': self._api_key is not None
        }


class GraphistryVisualizer:
    """
    Graph visualization engine using PyGraphistry.
    
    Handles the actual rendering and visualization logic.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._available = False
        self._graphistry_module = None
    
    def initialize(self) -> bool:
        """Initialize PyGraphistry module."""
        try:
            import graphistry
            self._graphistry_module = graphistry
            
            if self._api_key:
                graphistry.register(api_key=self._api_key)
            else:
                graphistry.register()
                
            self._available = True
            logger.info("PyGraphistry initialized successfully")
            return True
            
        except ImportError:
            logger.warning("PyGraphistry not installed. Install with: pip install graphistry")
            self._available = False
            return False
        except Exception as e:
            logger.error(f"PyGraphistry initialization failed: {e}")
            self._available = False
            return False
    
    def is_available(self) -> bool:
        """Check if PyGraphistry is available."""
        return self._available
    
    def visualize(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        config: Optional[VisualizationConfig] = None,
        output_format: str = 'url'
    ) -> VisualizationResult:
        """
        Create visualization from nodes and edges.
        """
        if not self._available:
            return VisualizationResult(
                status='error',
                error_message='PyGraphistry not available'
            )
        
        config = config or VisualizationConfig()
        
        try:
            # Convert to DataFrames
            nodes_df = pd.DataFrame(nodes) if nodes else pd.DataFrame()
            edges_df = pd.DataFrame(edges) if edges else pd.DataFrame()
            
            if nodes_df.empty:
                return VisualizationResult(
                    status='error',
                    error_message='No nodes to visualize'
                )
            
            # Ensure required columns exist
            if 'id' not in nodes_df.columns:
                nodes_df['id'] = range(len(nodes_df))
            
            if not edges_df.empty:
                if 'source' not in edges_df.columns or 'target' not in edges_df.columns:
                    return VisualizationResult(
                        status='error',
                        error_message='Edges must have source and target columns'
                    )
            
            # Build graph
            g = self._graphistry_module.bind(
                source='source',
                destination='target',
                node='id'
            )
            
            # Add node attributes
            if config.node_color_column and config.node_color_column in nodes_df.columns:
                g = g.bind(point_color=config.node_color_column)
            
            if config.node_size_column and config.node_size_column in nodes_df.columns:
                g = g.bind(point_size=config.node_size_column)
            
            if config.label_column and config.label_column in nodes_df.columns:
                g = g.bind(point_label=config.label_column)
            
            # Add edge attributes
            if config.edge_color_column and config.edge_color_column in edges_df.columns:
                g = g.bind(edge_color=config.edge_color_column)
            
            if config.edge_weight_column and config.edge_weight_column in edges_df.columns:
                g = g.bind(edge_weight=config.edge_weight_column)
            
            # Create plot
            plot = g.plot(edges_df if not edges_df.empty else None, nodes_df)
            
            # Generate output
            result = VisualizationResult(status='success')
            
            if output_format in ['url', 'all']:
                result.url = plot.url if hasattr(plot, 'url') else str(plot)
            
            if output_format in ['iframe', 'all']:
                result.iframe = self._generate_iframe(result.url, config) if result.url else None
            
            if output_format in ['html', 'all']:
                result.html = self._generate_html(result.url, config) if result.url else None
            
            # Compute metrics
            analyzer = GraphAnalyzer()
            result.metrics = analyzer.analyze(nodes, edges)
            
            result.metadata = {
                'node_count': len(nodes_df),
                'edge_count': len(edges_df),
                'layout': config.layout.value,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return VisualizationResult(
                status='error',
                error_message=f'Visualization failed: {str(e)}'
            )
    
    def _generate_iframe(self, url: str, config: VisualizationConfig) -> str:
        """Generate iframe HTML for embedding."""
        return f'''
        <iframe 
            src="{url}" 
            width="{config.width}" 
            height="{config.height}"
            frameborder="0"
            allowfullscreen
        ></iframe>
        '''
    
    def _generate_html(self, url: str, config: VisualizationConfig) -> str:
        """Generate complete HTML page with visualization."""
        title = config.title or "Knowledge Graph Visualization"
        return f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; }}
        .container {{ max-width: {config.width}px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .description {{ color: #666; margin-bottom: 20px; }}
        .visualization {{ border: 1px solid #ddd; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        {f'<p class="description">{config.description}</p>' if config.description else ''}
        <div class="visualization">
            <iframe src="{url}" width="{config.width}" height="{config.height}" frameborder="0"></iframe>
        </div>
    </div>
</body>
</html>'''


class GraphAnalyzer:
    """
    Graph analytics engine for computing metrics and detecting patterns.
    """
    
    def analyze(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        compute_paths: bool = False
    ) -> GraphMetrics:
        """
        Compute comprehensive graph metrics.
        """
        try:
            import networkx as nx
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            node_ids = set()
            for node in nodes:
                node_id = str(node.get('id', node.get('name', '')))
                if node_id:
                    G.add_node(node_id, **node)
                    node_ids.add(node_id)
            
            # Add edges
            for edge in edges:
                source = str(edge.get('source', ''))
                target = str(edge.get('target', ''))
                if source and target and source in node_ids and target in node_ids:
                    G.add_edge(source, target, **edge)
            
            metrics = GraphMetrics()
            metrics.node_count = G.number_of_nodes()
            metrics.edge_count = G.number_of_edges()
            
            if metrics.node_count > 0:
                # Basic metrics
                degrees = [d for n, d in G.degree()]
                metrics.avg_degree = sum(degrees) / len(degrees) if degrees else 0
                metrics.max_degree = max(degrees) if degrees else 0
                
                # Density
                metrics.density = nx.density(G)
                
                # Clustering coefficient
                try:
                    metrics.clustering_coefficient = nx.average_clustering(G)
                except:
                    metrics.clustering_coefficient = 0.0
                
                # Connected components
                metrics.connected_components = nx.number_connected_components(G)
                
                # Path-based metrics (expensive for large graphs)
                if compute_paths and metrics.node_count < 1000:
                    try:
                        metrics.diameter = nx.diameter(G)
                        metrics.avg_shortest_path = nx.average_shortest_path_length(G)
                    except:
                        pass
            
            return metrics
            
        except ImportError:
            logger.warning("NetworkX not installed. Basic metrics only.")
            return self._compute_basic_metrics(nodes, edges)
        except Exception as e:
            logger.error(f"Graph analysis failed: {e}")
            return GraphMetrics()
    
    def _compute_basic_metrics(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> GraphMetrics:
        """Compute basic metrics without NetworkX."""
        metrics = GraphMetrics()
        metrics.node_count = len(nodes)
        metrics.edge_count = len(edges)
        
        if metrics.node_count > 0:
            # Simple degree calculation
            degree_counts = {}
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                if source:
                    degree_counts[source] = degree_counts.get(source, 0) + 1
                if target:
                    degree_counts[target] = degree_counts.get(target, 0) + 1
            
            if degree_counts:
                degrees = list(degree_counts.values())
                metrics.avg_degree = sum(degrees) / len(degrees)
                metrics.max_degree = max(degrees)
            
            # Simple density
            max_edges = metrics.node_count * (metrics.node_count - 1) / 2
            if max_edges > 0:
                metrics.density = metrics.edge_count / max_edges
        
        return metrics
    
    def detect_communities(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        algorithm: str = 'louvain'
    ) -> Dict[str, Any]:
        """
        Detect communities in the graph.
        """
        try:
            import networkx as nx
            import networkx.algorithms.community as nx_comm
            
            # Create graph
            G = nx.Graph()
            
            node_ids = set()
            for node in nodes:
                node_id = str(node.get('id', node.get('name', '')))
                if node_id:
                    G.add_node(node_id)
                    node_ids.add(node_id)
            
            for edge in edges:
                source = str(edge.get('source', ''))
                target = str(edge.get('target', ''))
                if source and target and source in node_ids and target in node_ids:
                    G.add_edge(source, target)
            
            communities = []
            
            if algorithm == 'louvain':
                try:
                    import community as community_louvain
                    partition = community_louvain.best_partition(G)
                    communities = self._partition_to_communities(partition)
                except ImportError:
                    # Fall back to greedy modularity
                    communities_iter = nx_comm.greedy_modularity_communities(G)
                    communities = [list(c) for c in communities_iter]
            
            elif algorithm == 'greedy':
                communities_iter = nx_comm.greedy_modularity_communities(G)
                communities = [list(c) for c in communities_iter]
            
            elif algorithm == 'label_propagation':
                communities_iter = nx_comm.label_propagation_communities(G)
                communities = [list(c) for c in communities_iter]
            
            # Create node-to-community mapping
            node_communities = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_communities[node] = i
            
            return {
                'status': 'success',
                'algorithm': algorithm,
                'num_communities': len(communities),
                'communities': communities,
                'node_communities': node_communities,
                'modularity': nx_comm.modularity(G, communities) if communities else 0.0
            }
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'algorithm': algorithm
            }
    
    def _partition_to_communities(self, partition: Dict[str, int]) -> List[List[str]]:
        """Convert partition dict to communities list."""
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        return list(communities.values())


# Convenience functions for quick usage
def visualize_knowledge_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    title: Optional[str] = None
) -> VisualizationResult:
    """
    Quick visualization of a knowledge graph.
    
    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries  
        api_key: PyGraphistry API key
        title: Visualization title
        
    Returns:
        VisualizationResult
    """
    pg = PyGraphistryIntegration(api_key=api_key)
    config = VisualizationConfig(title=title)
    return pg.visualize_knowledge_graph(nodes, edges, config)


def analyze_graph_structure(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]]
) -> GraphMetrics:
    """
    Quick graph structure analysis.
    
    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        
    Returns:
        GraphMetrics
    """
    analyzer = GraphAnalyzer()
    return analyzer.analyze(nodes, edges)


__all__ = [
    'PyGraphistryIntegration',
    'GraphistryVisualizer',
    'GraphAnalyzer',
    'GraphMetrics',
    'VisualizationConfig',
    'VisualizationResult',
    'LayoutAlgorithm',
    'ColorScheme',
    'visualize_knowledge_graph',
    'analyze_graph_structure'
]
