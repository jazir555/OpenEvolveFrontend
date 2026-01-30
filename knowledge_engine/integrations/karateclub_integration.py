"""
Karate Club Integration Module for OpenEvolve Knowledge Engine

This module provides advanced graph analysis capabilities by integrating
Karate Club's state-of-the-art algorithms without modifying any core files.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Add Karate Club to Python path for import
karateclub_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'karateclub')
if karateclub_path not in sys.path:
    sys.path.insert(0, karateclub_path)


class KarateClubIntegration:
    """
    Main Karate Club Integration class for the Knowledge Engine.
    
    Provides graph analysis, community detection, and node embedding capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Karate Club Integration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._analyzer = KarateClubGraphAnalyzer()
    
    def is_available(self) -> bool:
        """Check if Karate Club is available."""
        return self._analyzer.is_available()
    
    def analyze_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a knowledge graph.
        
        Args:
            graph_data: Graph data with nodes and edges
            
        Returns:
            Dictionary with analysis results
        """
        return self._analyzer.analyze_graph(graph_data)
    
    def detect_communities(self, graph_data: Dict[str, Any], algorithm: str = 'louvain') -> Dict[str, Any]:
        """
        Detect communities in a graph.
        
        Args:
            graph_data: Graph data
            algorithm: Algorithm to use
            
        Returns:
            Dictionary with communities
        """
        return self._analyzer.detect_communities(graph_data, algorithm)


class KarateClubGraphAnalyzer:
    """
    Advanced graph analyzer that leverages Karate Club's algorithms.
    
    This class integrates community detection, node embeddings, and graph embeddings
    for comprehensive knowledge graph analysis.
    """
    
    def __init__(self):
        """Initialize Karate Club modules for graph analysis."""
        self._initialize_karateclub_modules()
    
    def _initialize_karateclub_modules(self):
        """Initialize all Karate Club modules with proper error handling."""
        try:
            # Import Karate Club modules
            # Note: Louvain and Leiden are in separate packages (python-louvain, leidenalg), not karateclub
            # LabelPropagation is in non_overlapping submodule
            from karateclub.community_detection.non_overlapping import LabelPropagation, EdMot, SCD, GEMSEC
            from karateclub.community_detection.overlapping import BigClam, DANMF, EgoNetSplitter, NNSED, SymmNMF
            from karateclub.node_embedding.neighbourhood import Node2Vec, DeepWalk
            from karateclub.graph_embedding import Graph2Vec, SF, FeatherGraph, FGSD, GL2Vec, IGE
            
            # Initialize community detectors (only those that actually exist in karateclub)
            self.community_detectors = {
                'label_propagation': LabelPropagation(),
                'bigclam': BigClam(),
                'danmf': DANMF(),
                'ego_splitter': EgoNetSplitter(),
                'nnsed': NNSED(),
                'symmnmf': SymmNMF(),
                'edmot': EdMot(),
                'scd': SCD(),
                'gemsec': GEMSEC()
            }
            
            # Initialize node embedders (only those that exist in karateclub)
            self.node_embedders = {
                'node2vec': Node2Vec(),
                'deepwalk': DeepWalk()
            }
            
            # Initialize graph embedders
            self.graph_embedders = {
                'graph2vec': Graph2Vec(),
                'sf': SF(),
                'feathergraph': FeatherGraph(),
                'fgsd': FGSD(),
                'gl2vec': GL2Vec(),
                'ige': IGE()
            }
            
            self._karateclub_available = True
            
        except ImportError as e:
            print(f"Warning: Could not import Karate Club modules: {e}")
            print("Karate Club integration will be disabled.")
            self._karateclub_available = False
            self.community_detectors = {}
            self.node_embedders = {}
            self.graph_embedders = {}
    
    def is_available(self) -> bool:
        """Check if Karate Club integration is available."""
        return self._karateclub_available
    
    def analyze_graph(self, graph_data: Dict[str, Any], analysis_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze knowledge graph using Karate Club algorithms.
        
        Args:
            graph_data: Knowledge graph data in OpenEvolve format
            analysis_config: Configuration for analysis process
            
        Returns:
            Dictionary containing graph analysis results
        """
        if not self.is_available():
            return {
                'status': 'error',
                'message': 'Karate Club integration not available',
                'analysis_results': {}
            }
        
        try:
            # Set default configuration
            config = {
                'community_detection': {
                    'enabled': True,
                    'algorithms': ['louvain', 'leiden', 'label_propagation'],
                    'overlapping_algorithms': ['bigclam', 'cfinder']
                },
                'node_embeddings': {
                    'enabled': True,
                    'algorithms': ['node2vec', 'deepwalk', 'graphsage'],
                    'dimensions': 128
                },
                'graph_embeddings': {
                    'enabled': True,
                    'algorithms': ['graph2vec', 'sf'],
                    'dimensions': 128
                },
                'calculate_metrics': True
            }
            
            if analysis_config:
                self._deep_update_config(config, analysis_config)
            
            # Convert OpenEvolve graph format to Karate Club format
            graph = self._convert_to_karate_format(graph_data)
            
            if graph is None:
                return {
                    'status': 'error',
                    'message': 'Graph conversion failed',
                    'analysis_results': {}
                }
            
            results = {}
            
            # Community detection
            if config['community_detection']['enabled']:
                results['communities'] = self._detect_communities(graph, config['community_detection'])
            
            # Node embeddings
            if config['node_embeddings']['enabled']:
                results['node_embeddings'] = self._generate_node_embeddings(graph, config['node_embeddings'])
            
            # Graph embeddings
            if config['graph_embeddings']['enabled']:
                results['graph_embeddings'] = self._generate_graph_embeddings(graph, config['graph_embeddings'])
            
            # Calculate graph metrics
            if config['calculate_metrics']:
                results['metrics'] = self._calculate_graph_metrics(graph)
            
            return {
                'status': 'success',
                'analysis_results': results,
                'config_used': config
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Karate Club analysis failed: {str(e)}',
                'analysis_results': {}
            }
    
    def _deep_update_config(self, base_config: Dict[str, Any], update_config: Dict[str, Any]):
        """Deep update configuration dictionary."""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._deep_update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _convert_to_karate_format(self, graph_data: Dict[str, Any]) -> Any:
        """Convert OpenEvolve graph format to Karate Club format."""
        try:
            # Import required modules
            import networkx as nx
            from karateclub.utils import convert_graph
            
            # Create NetworkX graph from OpenEvolve format
            G = nx.Graph()
            
            # Add nodes
            nodes = graph_data.get('nodes', [])
            for node in nodes:
                node_id = node.get('id')
                if node_id:
                    G.add_node(node_id)
                    # Add node attributes
                    for key, value in node.items():
                        if key != 'id':
                            G.nodes[node_id][key] = value
            
            # Add edges
            edges = graph_data.get('edges', [])
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                if source and target:
                    G.add_edge(source, target)
                    # Add edge attributes
                    for key, value in edge.items():
                        if key not in ['source', 'target']:
                            G.edges[(source, target)][key] = value
            
            # Convert to Karate Club format
            return convert_graph(G)
            
        except Exception as e:
            print(f"Warning: Graph conversion failed: {e}")
            return None
    
    def _detect_communities(self, graph: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect communities using multiple algorithms."""
        communities = {}
        
        # Non-overlapping community detection
        for algo_name in config['algorithms']:
            if algo_name in self.community_detectors:
                try:
                    detector = self.community_detectors[algo_name]
                    community_labels = detector.fit_predict(graph)
                    communities[f'non_overlapping_{algo_name}'] = {
                        'labels': community_labels.tolist(),
                        'num_communities': len(set(community_labels)),
                        'algorithm': algo_name
                    }
                except Exception as e:
                    print(f"Warning: {algo_name} community detection failed: {e}")
        
        # Overlapping community detection
        for algo_name in config['overlapping_algorithms']:
            if algo_name in self.community_detectors:
                try:
                    detector = self.community_detectors[algo_name]
                    community_membership = detector.fit_predict(graph)
                    communities[f'overlapping_{algo_name}'] = {
                        'membership': community_membership.tolist(),
                        'num_communities': len(community_membership[0]) if len(community_membership) > 0 else 0,
                        'algorithm': algo_name
                    }
                except Exception as e:
                    print(f"Warning: {algo_name} overlapping community detection failed: {e}")
        
        # Generate ensemble community detection
        if len(communities) > 1:
            communities['ensemble'] = self._ensemble_community_detection(communities)
        
        return communities
    
    def _generate_node_embeddings(self, graph: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate node embeddings using multiple algorithms."""
        embeddings = {}
        
        for algo_name in config['algorithms']:
            if algo_name in self.node_embedders:
                try:
                    embedder = self.node_embedders[algo_name]
                    
                    # Set dimensions if supported
                    if hasattr(embedder, 'dimensions'):
                        embedder.dimensions = config['dimensions']
                    
                    # Generate embeddings
                    embedding_result = embedder.fit_transform(graph)
                    
                    embeddings[algo_name] = {
                        'embeddings': embedding_result.tolist(),
                        'dimensions': config['dimensions'],
                        'algorithm': algo_name,
                        'num_nodes': len(embedding_result)
                    }
                except Exception as e:
                    print(f"Warning: {algo_name} node embedding failed: {e}")
        
        # Generate ensemble embeddings
        if len(embeddings) > 1:
            embeddings['ensemble'] = self._ensemble_node_embeddings(embeddings)
        
        return embeddings
    
    def _generate_graph_embeddings(self, graph: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate graph-level embeddings."""
        embeddings = {}
        
        for algo_name in config['algorithms']:
            if algo_name in self.graph_embedders:
                try:
                    embedder = self.graph_embedders[algo_name]
                    
                    # Set dimensions if supported
                    if hasattr(embedder, 'dimensions'):
                        embedder.dimensions = config['dimensions']
                    
                    # Generate embeddings (note: graph embedders expect list of graphs)
                    embedding_result = embedder.fit_transform([graph])
                    
                    embeddings[algo_name] = {
                        'embeddings': embedding_result.tolist(),
                        'dimensions': config['dimensions'],
                        'algorithm': algo_name
                    }
                except Exception as e:
                    print(f"Warning: {algo_name} graph embedding failed: {e}")
        
        return embeddings
    
    def _calculate_graph_metrics(self, graph: Any) -> Dict[str, Any]:
        """Calculate comprehensive graph metrics."""
        try:
            import networkx as nx
            from karateclub.utils import convert_graph
            
            # Convert back to NetworkX for metric calculation
            G = convert_graph(graph, to_networkx=True)
            
            metrics = {
                'basic_metrics': {
                    'num_nodes': G.number_of_nodes(),
                    'num_edges': G.number_of_edges(),
                    'density': nx.density(G),
                    'is_directed': nx.is_directed(G),
                    'is_weighted': False  # TODO: Check for weighted edges
                },
                'degree_metrics': {
                    'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
                    'max_degree': max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0,
                    'min_degree': min(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
                },
                'connectivity_metrics': {
                    'is_connected': nx.is_connected(G),
                    'num_connected_components': nx.number_connected_components(G),
                    'average_clustering': nx.average_clustering(G)
                }
            }
            
            # Calculate additional metrics if graph is large enough
            if G.number_of_nodes() > 10:
                metrics['centrality_metrics'] = {
                    'degree_centrality': self._calculate_centrality(G, nx.degree_centrality),
                    'betweenness_centrality': self._calculate_centrality(G, nx.betweenness_centrality),
                    'closeness_centrality': self._calculate_centrality(G, nx.closeness_centrality)
                }
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Graph metric calculation failed: {e}")
            return {}
    
    def _calculate_centrality(self, G: Any, centrality_func: callable) -> Dict[str, Any]:
        """Calculate centrality metrics."""
        try:
            centrality = centrality_func(G)
            return {
                'values': centrality,
                'average': sum(centrality.values()) / len(centrality),
                'max': max(centrality.values()) if centrality else 0,
                'min': min(centrality.values()) if centrality else 0
            }
        except Exception as e:
            print(f"Warning: Centrality calculation failed: {e}")
            return {}
    
    def _ensemble_community_detection(self, communities: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble community detection from multiple algorithms."""
        try:
            # Get all non-overlapping community results
            non_overlapping_results = {}
            for name, data in communities.items():
                if name.startswith('non_overlapping_'):
                    non_overlapping_results[name] = data
            
            if len(non_overlapping_results) < 2:
                return {
                    'method': 'single_algorithm',
                    'source': next(iter(communities.keys()))
                }
            
            # Use the most consistent community structure
            # For now, use the result with the highest number of communities
            best_result = max(non_overlapping_results.items(), 
                            key=lambda x: x[1]['num_communities'])
            
            return {
                'method': 'consensus_based',
                'source': best_result[0],
                'labels': best_result[1]['labels'],
                'num_communities': best_result[1]['num_communities'],
                'algorithms_used': list(non_overlapping_results.keys())
            }
            
        except Exception as e:
            print(f"Warning: Ensemble community detection failed: {e}")
            return {}
    
    def _ensemble_node_embeddings(self, embeddings: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble node embeddings from multiple algorithms."""
        try:
            # Get all embedding matrices
            embedding_matrices = []
            for name, data in embeddings.items():
                if 'embeddings' in data and len(data['embeddings']) > 0:
                    embedding_matrices.append(np.array(data['embeddings']))
            
            if len(embedding_matrices) < 2:
                return {
                    'method': 'single_algorithm',
                    'source': next(iter(embeddings.keys()))
                }
            
            # Simple averaging ensemble
            stacked = np.stack(embedding_matrices)
            average_embedding = np.mean(stacked, axis=0)
            
            return {
                'method': 'averaging',
                'embeddings': average_embedding.tolist(),
                'dimensions': average_embedding.shape[1],
                'algorithms_used': list(embeddings.keys())
            }
            
        except Exception as e:
            print(f"Warning: Ensemble node embeddings failed: {e}")
            return {}
    
    def analyze_knowledge_graph(self, graph_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze knowledge graph and return enhanced analysis.
        
        This method provides a simplified interface for knowledge graph analysis
        that returns results in OpenEvolve format.
        
        Args:
            graph_data: Knowledge graph data
            config: Analysis configuration
            
        Returns:
            Enhanced analysis results
        """
        result = self.analyze_graph(graph_data, config)
        
        if result['status'] == 'success':
            # Convert to OpenEvolve analysis format
            analysis = {
                'graph_analysis': result['analysis_results'],
                'analysis_stats': self._calculate_analysis_stats(result['analysis_results']),
                'metadata': {
                    'analysis_timestamp': self._get_current_timestamp(),
                    'analysis_method': 'karateclub',
                    'config_used': result['config_used']
                }
            }
            
            return analysis
        else:
            return {
                'graph_analysis': {},
                'analysis_stats': {},
                'metadata': {
                    'analysis_timestamp': self._get_current_timestamp(),
                    'analysis_method': 'karateclub',
                    'error': result['message']
                }
            }
    
    def _calculate_analysis_stats(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics about the analysis process."""
        stats = {
            'community_detection': {
                'algorithms_used': [],
                'num_communities_detected': 0
            },
            'node_embeddings': {
                'algorithms_used': [],
                'embedding_dimensions': []
            },
            'graph_embeddings': {
                'algorithms_used': [],
                'embedding_dimensions': []
            }
        }
        
        # Count community detection results
        if 'communities' in analysis_results:
            for name, data in analysis_results['communities'].items():
                if not name.startswith('overlapping_'):
                    stats['community_detection']['algorithms_used'].append(name)
                    if isinstance(data, dict) and 'num_communities' in data:
                        stats['community_detection']['num_communities_detected'] += data['num_communities']
        
        # Count node embedding results
        if 'node_embeddings' in analysis_results:
            for name, data in analysis_results['node_embeddings'].items():
                stats['node_embeddings']['algorithms_used'].append(name)
                if isinstance(data, dict) and 'dimensions' in data:
                    stats['node_embeddings']['embedding_dimensions'].append(data['dimensions'])
        
        # Count graph embedding results
        if 'graph_embeddings' in analysis_results:
            for name, data in analysis_results['graph_embeddings'].items():
                stats['graph_embeddings']['algorithms_used'].append(name)
                if isinstance(data, dict) and 'dimensions' in data:
                    stats['graph_embeddings']['embedding_dimensions'].append(data['dimensions'])
        
        return stats
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()