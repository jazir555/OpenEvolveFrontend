"""
Pygraphistry Adapter for OpenEvolve

This module provides an adapter that wraps Pygraphistry's interactive visualization
and ML analytics capabilities to implement the OpenEvolve VisualizationInterface.
It enables GPU-accelerated graph analytics with UMAP + DBSCAN clustering pipeline.

Zero modifications to pygraphistry source - uses a decoupled adapter pattern.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime

# Add local pygraphistry directory to path if it exists
local_pygraphistry = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pygraphistry")
if os.path.exists(local_pygraphistry) and local_pygraphistry not in sys.path:
    sys.path.insert(0, local_pygraphistry)

# Try importing pygraphistry - graceful degradation
try:
    import graphistry
    PYGRAPHISTRY_AVAILABLE = True
except ImportError:
    PYGRAPHISTRY_AVAILABLE = False
    graphistry = None

# Try importing optional ML libraries
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import cudf
    import cuml
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

from integrations.base.visualization_interface import VisualizationInterface

logger = logging.getLogger(__name__)


class PygraphistryAdapter(VisualizationInterface):
    """
    Adapter for Pygraphistry interactive graph visualization and ML analytics.

    This adapter wraps Pygraphistry's core functionality to provide:
    - Interactive web-based graph visualization
    - GPU-accelerated analytics with cuML (optional)
    - UMAP dimensionality reduction
    - DBSCAN clustering
    - iframe embedding support for UI clients

    Gracefully degrades if pygraphistry or ML libraries are unavailable.

    Attributes:
        config: Configuration dictionary
        is_initialized: Whether adapter has been initialized
        use_gpu: Whether to use GPU acceleration
        cache_enabled: Whether to cache visualization results
    """

    def __init__(self):
        """Initialize the adapter without connecting to pygraphistry."""
        self.config: Optional[Dict[str, Any]] = None
        self.is_initialized = False
        self.use_gpu = False
        self.cache_enabled = False
        self.cache_ttl = 3600
        self._visualization_cache = {}

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize pygraphistry with the provided configuration.

        Args:
            config: Configuration dictionary with keys:
                - api_key: Pygraphistry API key (or set GRAPHISTRY_API_KEY env var)
                - username: Pygraphistry username (optional, for auth)
                - password: Pygraphistry password (optional, for auth)
                - gpu_acceleration: Enable GPU with cuML (default: False)
                - umap_clustering: Enable UMAP embeddings (default: True)
                - dbscan_clustering: Enable DBSCAN clustering (default: True)
                - ui_embedding: Enable iframe support for UI clients (default: True)
                - auto_start: Auto-start visualization server (default: True)
                - cache_enabled: Cache visualization results (default: True)
                - cache_ttl: Cache TTL in seconds (default: 3600)
                - max_workers: Max parallel workers (default: 4)
                - timeout: Request timeout in seconds (default: 30)
                - batch_size: Batch size for processing (default: 100)

        Returns:
            True if initialization was successful

        Raises:
            ConfigurationError: If config is invalid or pygraphistry unavailable
        """
        if not PYGRAPHISTRY_AVAILABLE:
            logger.warning("Pygraphistry not available. Install with: pip install pygraphistry")
            return False

        try:
            self.config = config
            self.cache_enabled = config.get("cache_enabled", True)
            self.cache_ttl = config.get("cache_ttl", 3600)
            self.use_gpu = config.get("gpu_acceleration", False)

            # Validate GPU availability if requested
            if self.use_gpu and not CUML_AVAILABLE:
                logger.warning("GPU acceleration requested but cuML not available. Falling back to CPU.")
                self.use_gpu = False

            # Configure pygraphistry authentication
            api_key = config.get("api_key") or os.environ.get("GRAPHISTRY_API_KEY")
            if api_key:
                graphistry.register(api_key=api_key)
                logger.info("Pygraphistry authenticated with API key")
            else:
                # Pygraphistry can work without API key for local visualizations
                logger.info("No API key provided - using local visualization mode")

            username = config.get("username")
            password = config.get("password")
            if username and password:
                graphistry.authenticate(username=username, password=password)
                logger.info("Pygraphistry authenticated with username/password")

            self.is_initialized = True
            logger.info("Pygraphistry adapter initialized successfully")

            # Log capabilities
            capabilities = self._get_capabilities()
            logger.info(f"Available capabilities: {', '.join(capabilities)}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Pygraphistry adapter: {e}")
            return False

    async def visualize_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        layout: str = "force_directed",
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Visualize a graph with nodes and edges using pygraphistry.

        Args:
            nodes: List of node dictionaries with 'id' and attributes
            edges: List of edge dictionaries with 'source', 'target', and attributes
            layout: Layout algorithm ('force_directed', 'circular', 'hierarchical')
            output_path: Optional path to save visualization HTML

        Returns:
            URL or path to visualization, or None if failed
        """
        if not self.is_initialized or not PYGRAPHISTRY_AVAILABLE:
            logger.error("Pygraphistry adapter not initialized")
            return None

        try:
            # Convert to pandas DataFrame format expected by pygraphistry
            import pandas as pd

            nodes_df = pd.DataFrame(nodes)
            edges_df = pd.DataFrame(edges)

            # Ensure required columns exist
            if 'id' not in nodes_df.columns:
                nodes_df['id'] = nodes_df.index.astype(str)

            if 'source' not in edges_df.columns:
                edges_df['source'] = edges_df.iloc[:, 0].astype(str)
            if 'target' not in edges_df.columns:
                edges_df['target'] = edges_df.iloc[:, 1].astype(str)

            # Create pygraphistry plot
            plotter = graphistry.bind(
                source='source',
                destination='target',
                node='id'
            )

            # Add data
            plotter = plotter.nodes(nodes_df).edges(edges_df)

            # Apply layout
            if layout == "force_directed":
                # Pygraphistry's default is force-directed
                pass
            elif layout == "circular":
                plotter = plotter.layout_settings(layout_method='circular')
            elif layout == "hierarchical":
                plotter = plotter.layout_settings(layout_method='hierarchical')

            # Generate visualization
            url = plotter.plot(render=False)

            # Save to HTML if requested
            if output_path:
                plotter.save(output_path)
                logger.info(f"Visualization saved to {output_path}")
                return output_path

            logger.info(f"Visualization created: {url}")
            return url

        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            return None

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
            method: Embedding method ('umap', 'pca')
            n_components: Number of dimensions for output

        Returns:
            Embedding array (n_nodes x n_components), or None if failed
        """
        if not self.is_initialized:
            logger.error("Pygraphistry adapter not initialized")
            return None

        # Extract features from nodes
        import pandas as pd

        nodes_df = pd.DataFrame(nodes)

        # Filter numeric columns for embedding
        numeric_cols = nodes_df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            logger.warning("No numeric features found for embedding")
            return None

        features = nodes_df[numeric_cols].fillna(0).values

        try:
            if method == "umap":
                if not UMAP_AVAILABLE:
                    logger.warning("UMAP not available. Install with: pip install umap-learn")
                    return None

                if self.use_gpu and CUML_AVAILABLE:
                    # Use cuML UMAP for GPU acceleration
                    embedder = cuml.UMAP(
                        n_components=n_components,
                        n_neighbors=15,
                        min_dist=0.1
                    )
                else:
                    # Use CPU UMAP
                    embedder = UMAP(
                        n_components=n_components,
                        n_neighbors=15,
                        min_dist=0.1
                    )

                embeddings = embedder.fit_transform(features)

            elif method == "pca":
                if not SKLEARN_AVAILABLE:
                    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")
                    return None

                embedder = PCA(n_components=n_components)
                embeddings = embedder.fit_transform(features)

            else:
                logger.error(f"Unknown embedding method: {method}")
                return None

            logger.info(f"Computed {method} embeddings: shape={embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            return None

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
            method: Clustering method ('dbscan', 'kmeans')
            **kwargs: Additional parameters for clustering algorithm
                - eps: DBSCAN epsilon (default: 0.5)
                - min_samples: DBSCAN min samples (default: 5)
                - n_clusters: K-means number of clusters (default: 3)

        Returns:
            Cluster labels array (n_nodes,), or None if failed
        """
        if not self.is_initialized:
            logger.error("Pygraphistry adapter not initialized")
            return None

        try:
            if method == "dbscan":
                if not SKLEARN_AVAILABLE and not CUML_AVAILABLE:
                    logger.warning("Neither scikit-learn nor cuML available for DBSCAN")
                    return None

                eps = kwargs.get("eps", 0.5)
                min_samples = kwargs.get("min_samples", 5)

                if self.use_gpu and CUML_AVAILABLE:
                    # Use cuML DBSCAN for GPU acceleration
                    clusterer = cuml.DBSCAN(eps=eps, min_samples=min_samples)
                else:
                    # Use scikit-learn DBSCAN
                    clusterer = DBSCAN(eps=eps, min_samples=min_samples)

                labels = clusterer.fit_predict(embeddings)

            elif method == "kmeans":
                if not SKLEARN_AVAILABLE and not CUML_AVAILABLE:
                    logger.warning("Neither scikit-learn nor cuML available for K-means")
                    return None

                n_clusters = kwargs.get("n_clusters", 3)

                if self.use_gpu and CUML_AVAILABLE:
                    # Use cuML K-means for GPU acceleration
                    clusterer = cuml.KMeans(n_clusters=n_clusters)
                else:
                    # Use scikit-learn K-means
                    from sklearn.cluster import KMeans
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)

                labels = clusterer.fit_predict(embeddings)

            else:
                logger.error(f"Unknown clustering method: {method}")
                return None

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"Clustered nodes into {n_clusters} clusters using {method}")
            return labels

        except Exception as e:
            logger.error(f"Failed to cluster nodes: {e}")
            return None

    async def create_interactive_dashboard(
        self,
        data: Dict[str, Any],
        dashboard_type: str = "graph"
    ) -> Optional[str]:
        """
        Create an interactive dashboard for data exploration.

        Args:
            data: Data dictionary with nodes, edges, embeddings, clusters
            dashboard_type: Type of dashboard ('graph', 'clusters', 'patterns')

        Returns:
            URL or path to dashboard, or None if failed
        """
        if not self.is_initialized or not PYGRAPHISTRY_AVAILABLE:
            logger.error("Pygraphistry adapter not initialized")
            return None

        try:
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            embeddings = data.get("embeddings")
            clusters = data.get("clusters")

            import pandas as pd

            nodes_df = pd.DataFrame(nodes)

            # Add embeddings to nodes if available
            if embeddings is not None:
                nodes_df['x'] = embeddings[:, 0]
                nodes_df['y'] = embeddings[:, 1] if embeddings.shape[1] > 1 else 0

            # Add cluster labels if available
            if clusters is not None:
                nodes_df['cluster'] = clusters.astype(str)

            # Create pygraphistry plot
            plotter = graphistry.bind(
                source='source',
                destination='target',
                node='id'
            )

            plotter = plotter.nodes(nodes_df).edges(pd.DataFrame(edges))

            # Encode by cluster if available
            if 'cluster' in nodes_df.columns:
                plotter = plotter.encode_point_color('cluster')

            # Generate dashboard URL
            url = plotter.plot(render=False)

            logger.info(f"Interactive dashboard created: {url}")
            return url

        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return None

    async def validate(self) -> Dict[str, Any]:
        """
        Validate the pygraphistry integration is working correctly.

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'version': str,
                'capabilities': List[str],
                'errors': List[str]
            }
        """
        checks = {
            "pygraphistry_installed": PYGRAPHISTRY_AVAILABLE,
            "umap_available": UMAP_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "cuml_available": CUML_AVAILABLE,
            "initialized": self.is_initialized,
            "gpu_enabled": self.use_gpu,
        }

        errors = []
        if not PYGRAPHISTRY_AVAILABLE:
            errors.append("pygraphistry not installed")

        capabilities = self._get_capabilities()

        return {
            "valid": self.is_initialized and PYGRAPHISTRY_AVAILABLE,
            "version": self._get_version(),
            "capabilities": capabilities,
            "checks": checks,
            "errors": errors,
        }

    async def shutdown(self) -> bool:
        """
        Cleanly shutdown the pygraphistry adapter.

        Returns:
            True if shutdown successful
        """
        if not self.is_initialized:
            return True

        try:
            # Clear cache
            self._visualization_cache.clear()

            self.is_initialized = False
            logger.info("Pygraphistry adapter shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            return False

    def _get_capabilities(self) -> List[str]:
        """Get list of available capabilities based on installed libraries."""
        capabilities = []

        if PYGRAPHISTRY_AVAILABLE:
            capabilities.append("interactive_viz")

        if UMAP_AVAILABLE:
            capabilities.append("umap")

        if SKLEARN_AVAILABLE:
            capabilities.append("dbscan")
            capabilities.append("kmeans")
            capabilities.append("pca")

        if CUML_AVAILABLE:
            capabilities.append("gpu_acceleration")

        return capabilities

    def _get_version(self) -> str:
        """Get pygraphistry version."""
        if PYGRAPHISTRY_AVAILABLE:
            try:
                import graphistry
                return getattr(graphistry, "__version__", "unknown")
            except:
                return "unknown"
        return "not_installed"
