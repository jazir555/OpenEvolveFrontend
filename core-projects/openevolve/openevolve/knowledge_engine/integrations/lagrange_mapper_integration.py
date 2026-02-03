"""
Lagrange Mapper Integration Module for OpenEvolve Knowledge Engine

This module provides topological data analysis capabilities by integrating
lagrange-mapper's attractor landscape mapping for understanding LLM output spaces
and knowledge embedding topologies.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime
from collections import Counter

# Add lagrange-mapper to Python path for import
lagrange_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'lagrange-mapper')
if lagrange_path not in sys.path:
    sys.path.insert(0, lagrange_path)


class LagrangeMapperIntegration:
    """
    Main Lagrange Mapper Integration class for the Knowledge Engine.
    
    Provides topological data analysis and attractor landscape mapping.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Lagrange Mapper Integration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._analyzer = LagrangeAttractorAnalyzer()
    
    def is_available(self) -> bool:
        """Check if Lagrange Mapper is available."""
        return self._analyzer.is_available()
    
    def analyze_landscape(self, embeddings: np.ndarray, n_clusters: int = 8) -> Dict[str, Any]:
        """
        Analyze embedding landscape.
        
        Args:
            embeddings: Embedding matrix
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with landscape analysis
        """
        return self._analyzer.analyze_embedding_landscape(embeddings, n_clusters=n_clusters)


class LagrangeAttractorAnalyzer:
    """
    Attractor landscape analyzer using topological data analysis.
    
    This class provides capabilities for:
    - Mapping attractor landscapes in knowledge spaces
    - Identifying clusters and stable regions in embeddings
    - Analyzing topological structure of knowledge graphs
    """
    
    def __init__(self):
        """Initialize lagrange-mapper modules."""
        self._lagrange_available = False
        self._sklearn_available = False
        self._initialize_lagrange()
    
    def _initialize_lagrange(self):
        """Initialize lagrange-mapper with proper error handling."""
        try:
            # Check for required dependencies
            try:
                from sklearn.cluster import KMeans
                from sklearn.decomposition import PCA
                self._sklearn_available = True
            except ImportError:
                print("Warning: scikit-learn not available. Some features will be limited.")
            
            self._lagrange_available = True
            print("Lagrange-mapper integration initialized")
            
        except Exception as e:
            print(f"Warning: Could not initialize lagrange-mapper: {e}")
    
    def is_available(self) -> bool:
        """Check if lagrange-mapper integration is available."""
        return self._lagrange_available
    
    def analyze_embedding_landscape(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        n_clusters: int = 8,
        reduction_method: str = 'pca',
        reduction_dims: int = 2
    ) -> Dict[str, Any]:
        """
        Analyze the landscape of embeddings to identify attractors/clusters.
        
        Args:
            embeddings: Matrix of embeddings (n_samples x n_features)
            labels: Optional labels for each embedding
            n_clusters: Number of clusters to identify
            reduction_method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            reduction_dims: Dimensions to reduce to for visualization
            
        Returns:
            Dictionary containing landscape analysis
        """
        if not self.is_available():
            return {
                'status': 'error',
                'message': 'Lagrange-mapper integration not available',
                'landscape': {}
            }
        
        try:
            # Ensure embeddings is numpy array
            embeddings = np.array(embeddings)
            
            # Validate input
            if embeddings.ndim != 2:
                return {
                    'status': 'error',
                    'message': 'Embeddings must be 2D array (n_samples x n_features)',
                    'landscape': {}
                }
            
            n_samples = embeddings.shape[0]
            
            if labels is None:
                labels = [f'sample_{i}' for i in range(n_samples)]
            
            # Determine optimal number of clusters if not specified
            if n_clusters is None or n_clusters <= 0:
                n_clusters = min(8, max(2, n_samples // 10))
            
            # Cluster the embeddings
            cluster_result = self._cluster_embeddings(embeddings, n_clusters)
            
            # Dimensionality reduction for visualization
            reduced_embeddings = self._reduce_dimensions(
                embeddings, reduction_method, reduction_dims
            )
            
            # Analyze cluster properties
            cluster_analysis = self._analyze_clusters(
                embeddings, cluster_result['labels'], labels
            )
            
            # Calculate attractor strengths
            attractors = self._calculate_attractor_strengths(
                embeddings, cluster_result['labels'], cluster_result['centers']
            )
            
            return {
                'status': 'success',
                'landscape': {
                    'n_samples': n_samples,
                    'n_features': embeddings.shape[1],
                    'n_clusters': n_clusters,
                    'cluster_labels': cluster_result['labels'].tolist(),
                    'cluster_centers': cluster_result['centers'].tolist(),
                    'reduced_embeddings': reduced_embeddings.tolist() if reduced_embeddings is not None else None,
                    'clusters': cluster_analysis,
                    'attractors': attractors
                },
                'parameters': {
                    'n_clusters': n_clusters,
                    'reduction_method': reduction_method,
                    'reduction_dims': reduction_dims
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Landscape analysis failed: {str(e)}',
                'landscape': {}
            }
    
    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int
    ) -> Dict[str, Any]:
        """Cluster embeddings using K-means or alternative methods."""
        try:
            if self._sklearn_available:
                from sklearn.cluster import KMeans
                
                # Ensure n_clusters doesn't exceed samples
                n_clusters = min(n_clusters, embeddings.shape[0])
                
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10
                )
                labels = kmeans.fit_predict(embeddings)
                centers = kmeans.cluster_centers_
                
                return {
                    'labels': labels,
                    'centers': centers,
                    'inertia': kmeans.inertia_
                }
            else:
                # Fallback to simple clustering
                return self._simple_clustering(embeddings, n_clusters)
                
        except Exception as e:
            print(f"Warning: Clustering failed, using fallback: {e}")
            return self._simple_clustering(embeddings, n_clusters)
    
    def _simple_clustering(
        self,
        embeddings: np.ndarray,
        n_clusters: int
    ) -> Dict[str, Any]:
        """Simple k-means-like clustering without sklearn."""
        n_samples = embeddings.shape[0]
        n_clusters = min(n_clusters, n_samples)
        
        # Random initialization of centers
        np.random.seed(42)
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        centers = embeddings[indices].copy()
        
        # Iterative optimization
        for _ in range(20):
            # Assign points to nearest center
            distances = np.linalg.norm(
                embeddings[:, np.newaxis] - centers[np.newaxis, :],
                axis=2
            )
            labels = np.argmin(distances, axis=1)
            
            # Update centers
            new_centers = np.array([
                embeddings[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                for k in range(n_clusters)
            ])
            
            # Check convergence
            if np.allclose(centers, new_centers, atol=1e-4):
                break
            
            centers = new_centers
        
        # Recalculate final labels
        distances = np.linalg.norm(
            embeddings[:, np.newaxis] - centers[np.newaxis, :],
            axis=2
        )
        labels = np.argmin(distances, axis=1)
        
        return {
            'labels': labels,
            'centers': centers,
            'inertia': np.sum(np.min(distances, axis=1) ** 2)
        }
    
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str,
        n_dims: int
    ) -> Optional[np.ndarray]:
        """Reduce dimensions for visualization."""
        try:
            if method == 'pca' and self._sklearn_available:
                from sklearn.decomposition import PCA
                
                # Adjust dimensions
                n_dims = min(n_dims, embeddings.shape[0], embeddings.shape[1])
                
                pca = PCA(n_components=n_dims)
                return pca.fit_transform(embeddings)
            
            elif method == 'tsne' and self._sklearn_available:
                from sklearn.manifold import TSNE
                
                # Adjust dimensions
                n_dims = min(n_dims, embeddings.shape[0] - 1)
                
                tsne = TSNE(n_components=n_dims, random_state=42)
                return tsne.fit_transform(embeddings)
            
            else:
                # Simple projection or no reduction
                if embeddings.shape[1] > n_dims:
                    # Project onto first n_dims dimensions
                    return embeddings[:, :n_dims]
                return embeddings
                
        except Exception as e:
            print(f"Warning: Dimensionality reduction failed: {e}")
            return embeddings[:, :min(n_dims, embeddings.shape[1])]
    
    def _analyze_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        sample_labels: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze properties of each cluster."""
        clusters = []
        n_clusters = len(np.unique(labels))
        
        for k in range(n_clusters):
            mask = labels == k
            cluster_embeddings = embeddings[mask]
            cluster_samples = [sample_labels[i] for i in range(len(labels)) if mask[i]]
            
            # Calculate statistics
            centroid = cluster_embeddings.mean(axis=0)
            
            # Calculate spread (average distance from centroid)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            spread = float(np.mean(distances))
            
            # Calculate density (inverse of spread)
            density = 1.0 / (1.0 + spread)
            
            clusters.append({
                'cluster_id': k,
                'size': int(np.sum(mask)),
                'spread': spread,
                'density': density,
                'samples': cluster_samples[:10]  # Limit samples listed
            })
        
        return clusters
    
    def _calculate_attractor_strengths(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Calculate attractor strengths for each cluster."""
        attractors = []
        n_clusters = len(centers)
        
        for k in range(n_clusters):
            mask = labels == k
            cluster_embeddings = embeddings[mask]
            
            if len(cluster_embeddings) == 0:
                continue
            
            # Calculate distance to center
            distances = np.linalg.norm(cluster_embeddings - centers[k], axis=1)
            
            # Attractor strength based on:
            # 1. Tightness (inverse of average distance)
            # 2. Size (number of samples)
            tightness = 1.0 / (1.0 + np.mean(distances))
            size = len(cluster_embeddings)
            
            # Combined attractor strength
            strength = tightness * np.log1p(size)
            
            attractors.append({
                'cluster_id': k,
                'strength': float(strength),
                'tightness': float(tightness),
                'size': size,
                'center': centers[k].tolist()
            })
        
        # Sort by strength
        attractors.sort(key=lambda x: x['strength'], reverse=True)
        
        return attractors
    
    def find_attractor_basins(
        self,
        embeddings: np.ndarray,
        attractor_centers: np.ndarray,
        resolution: int = 50
    ) -> Dict[str, Any]:
        """
        Find basins of attraction around attractor centers.
        
        Args:
            embeddings: Embedding space
            attractor_centers: Centers of attractors
            resolution: Grid resolution for basin computation
            
        Returns:
            Dictionary containing basin information
        """
        try:
            # Reduce to 2D for visualization if needed
            if embeddings.shape[1] > 2:
                reduced = self._reduce_dimensions(embeddings, 'pca', 2)
                centers_reduced = self._reduce_dimensions(
                    np.vstack([embeddings, attractor_centers]), 'pca', 2
                )[-len(attractor_centers):]
            else:
                reduced = embeddings
                centers_reduced = attractor_centers
            
            # Compute Voronoi-like regions (basins)
            # For each point, find nearest attractor
            basins = {i: [] for i in range(len(attractor_centers))}
            
            for i, point in enumerate(reduced):
                distances = np.linalg.norm(centers_reduced - point, axis=1)
                nearest_attractor = np.argmin(distances)
                basins[nearest_attractor].append(i)
            
            # Calculate basin properties
            basin_info = []
            for k, indices in basins.items():
                if indices:
                    basin_points = reduced[indices]
                    basin_info.append({
                        'attractor_id': k,
                        'size': len(indices),
                        'coverage': len(indices) / len(embeddings),
                        'centroid': basin_points.mean(axis=0).tolist(),
                        'spread': float(np.std(basin_points, axis=0).mean())
                    })
            
            return {
                'status': 'success',
                'basins': basin_info,
                'num_basins': len(basin_info)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Basin computation failed: {str(e)}',
                'basins': []
            }
    
    def analyze_knowledge_topology(
        self,
        graph_data: Dict[str, Any],
        embedding_dim: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze topological structure of a knowledge graph.
        
        Args:
            graph_data: Knowledge graph with nodes and edges
            embedding_dim: Dimension for node embeddings
            
        Returns:
            Dictionary containing topological analysis
        """
        try:
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            if not nodes or not edges:
                return {
                    'status': 'error',
                    'message': 'Graph must have nodes and edges',
                    'topology': {}
                }
            
            # Build adjacency matrix
            node_ids = {node['id']: i for i, node in enumerate(nodes)}
            n = len(nodes)
            
            adjacency = np.zeros((n, n))
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                if source in node_ids and target in node_ids:
                    i, j = node_ids[source], node_ids[target]
                    adjacency[i, j] = 1
            
            # Create simple node embeddings based on graph structure
            # Use spectral embedding (eigenvectors of Laplacian)
            degrees = np.sum(adjacency, axis=1)
            laplacian = np.diag(degrees) - adjacency
            
            # Compute eigenvectors
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                
                # Use smallest non-zero eigenvectors as embeddings
                embedding_dim = min(embedding_dim, n - 1)
                node_embeddings = eigenvectors[:, 1:embedding_dim+1]
                
            except np.linalg.LinAlgError:
                # Fallback to random embeddings
                node_embeddings = np.random.randn(n, embedding_dim) / np.sqrt(embedding_dim)
            
            # Analyze topology
            landscape = self.analyze_embedding_landscape(
                node_embeddings,
                labels=[node.get('id') for node in nodes],
                n_clusters=min(8, n),
                reduction_method='pca',
                reduction_dims=2
            )
            
            # Add graph-specific metrics
            if landscape['status'] == 'success':
                landscape['landscape']['graph_metrics'] = {
                    'num_nodes': n,
                    'num_edges': len(edges),
                    'density': len(edges) / (n * (n - 1)) if n > 1 else 0,
                    'avg_degree': float(np.mean(degrees)),
                    'connected_components': self._count_connected_components(adjacency)
                }
            
            return landscape
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Topology analysis failed: {str(e)}',
                'topology': {}
            }
    
    def _count_connected_components(self, adjacency: np.ndarray) -> int:
        """Count connected components in graph."""
        n = adjacency.shape[0]
        visited = np.zeros(n, dtype=bool)
        components = 0
        
        for start in range(n):
            if not visited[start]:
                components += 1
                # BFS
                queue = [start]
                visited[start] = True
                
                while queue:
                    node = queue.pop(0)
                    neighbors = np.where(adjacency[node] > 0)[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
        
        return components
    
    def detect_landscape_transitions(
        self,
        embeddings_t1: np.ndarray,
        embeddings_t2: np.ndarray,
        labels_t1: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Detect transitions in the landscape between two time points.
        
        Args:
            embeddings_t1: Embeddings at time t1
            embeddings_t2: Embeddings at time t2
            labels_t1: Optional cluster labels at t1
            
        Returns:
            Dictionary containing transition analysis
        """
        try:
            # Ensure same number of samples
            n_samples = min(embeddings_t1.shape[0], embeddings_t2.shape[0])
            embeddings_t1 = embeddings_t1[:n_samples]
            embeddings_t2 = embeddings_t2[:n_samples]
            
            # Analyze landscapes at both times
            landscape_t1 = self.analyze_embedding_landscape(
                embeddings_t1,
                n_clusters=8 if labels_t1 is None else len(set(labels_t1))
            )
            
            landscape_t2 = self.analyze_embedding_landscape(
                embeddings_t2,
                n_clusters=landscape_t1['landscape']['n_clusters'] if landscape_t1['status'] == 'success' else 8
            )
            
            if landscape_t1['status'] != 'success' or landscape_t2['status'] != 'success':
                return {
                    'status': 'error',
                    'message': 'Failed to analyze one or both landscapes',
                    'transitions': {}
                }
            
            # Compare attractors
            attractors_t1 = landscape_t1['landscape']['attractors']
            attractors_t2 = landscape_t2['landscape']['attractors']
            
            # Track changes
            transitions = {
                'attractors_created': [],
                'attractors_destroyed': [],
                'attractors_persisted': [],
                'strength_changes': []
            }
            
            # Match attractors between time points (simplified)
            for a2 in attractors_t2:
                matched = False
                for a1 in attractors_t1:
                    # Check if attractor persisted (based on cluster ID matching)
                    if a2['cluster_id'] == a1['cluster_id']:
                        matched = True
                        transitions['attractors_persisted'].append({
                            'cluster_id': a2['cluster_id'],
                            'strength_t1': a1['strength'],
                            'strength_t2': a2['strength'],
                            'strength_change': a2['strength'] - a1['strength']
                        })
                        break
                
                if not matched:
                    transitions['attractors_created'].append(a2)
            
            # Find destroyed attractors
            for a1 in attractors_t1:
                if not any(a2['cluster_id'] == a1['cluster_id'] for a2 in attractors_t2):
                    transitions['attractors_destroyed'].append(a1)
            
            return {
                'status': 'success',
                'transitions': transitions,
                'n_attractors_t1': len(attractors_t1),
                'n_attractors_t2': len(attractors_t2),
                'stability': len(transitions['attractors_persisted']) / max(len(attractors_t1), 1)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Transition detection failed: {str(e)}',
                'transitions': {}
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of lagrange-mapper integration."""
        return {
            'available': self.is_available(),
            'sklearn_available': self._sklearn_available,
            'timestamp': datetime.now().isoformat()
        }
