"""
Unified Knowledge Graph Manager

Provides a consistent interface across all knowledge graph backends.
Follows CLAUDE.md principles: Zero Trust, Runtime Truth, Configuration Explicitness.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import yaml
from pathlib import Path
import json

from .backends.base import (
    KnowledgeGraphBackend,
    BackendType,
    OperationType,
    KnowledgeEntry,
    SearchResults,
    AnalysisResult,
    GraphStatistics
)
from .backends.neo4j_backend import Neo4jBackend
from .backends.qdrant_backend import QdrantBackend
from .backends.mongodb_backend import MongoDBBackend
from .backends.karateclub_backend import KarateClubBackend
from .backends.memory_backend import MemoryBackend

# KarateClub Analytics Integration
from ..integrations.karateclub_analytics import KarateClubAnalytics
from ..integrations.karateclub_retrieval import KarateClubRetrieval
import networkx as nx

logger = logging.getLogger(__name__)


class KnowledgeGraphError(Exception):
    """Base exception for knowledge graph operations"""
    pass


class BackendUnavailableError(KnowledgeGraphError):
    """Raised when all backends are unavailable"""
    pass


class UnifiedKnowledgeGraph:
    """
    Unified interface for knowledge graph operations across all backends.

    This manager provides:
    - Automatic backend selection based on operation type
    - Fallback chain for high availability
    - Unified result formatting
    - Health monitoring and circuit breaking
    - Performance monitoring

    Configuration (YAML):
        backends:
          neo4j:
            enabled: true
            uri: bolt://localhost:7687
            user: neo4j
            password: password
          qdrant:
            enabled: true
            host: localhost
            port: 6333
          mongodb:
            enabled: true
            uri: mongodb://localhost:27017
          karateclub:
            enabled: true
            embedding_dim: 128

        fallback_chain:
          - neo4j
          - qdrant
          - mongodb
          - memory

        operations:
          add_knowledge: [neo4j, mongodb]
          search: [qdrant, neo4j, mongodb]
          analyze: [karateclub, neo4j]
          visualize: [neo4j, karateclub]
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Unified Knowledge Graph Manager.

        Args:
            config_path: Path to YAML configuration file
        """
        self.backends: Dict[str, KnowledgeGraphBackend] = {}
        self.fallback_chain: List[str] = []
        self.operation_backends: Dict[str, List[str]] = {}
        self.performance_metrics: Dict[str, List[float]] = {}

        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()

        # Initialize backends
        self._initialize_backends()

        # Initialize KarateClub Analytics if enabled
        self.karateclub = None
        if "karateclub" in self.backends or self.config.get("karateclub", {}).get("enabled", False):
            try:
                karateclub_config = self.config.get("karateclub", self.config.get("backends", {}).get("karateclub", {}))
                self.karateclub = KarateClubAnalytics()
                logger.info("KarateClub Analytics initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize KarateClub Analytics: {e}")

        logger.info("UnifiedKnowledgeGraph initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "backends": {
                "memory": {
                    "enabled": True
                }
            },
            "fallback_chain": ["memory"],
            "operations": {
                "add_knowledge": ["memory"],
                "search": ["memory"],
                "analyze": ["memory"],
                "visualize": ["memory"]
            }
        }

    def _initialize_backends(self):
        """Initialize all enabled backends"""
        backend_classes = {
            "neo4j": Neo4jBackend,
            "qdrant": QdrantBackend,
            "mongodb": MongoDBBackend,
            "karateclub": KarateClubBackend,
            "memory": MemoryBackend
        }

        backends_config = self.config.get("backends", {})

        for backend_name, backend_config in backends_config.items():
            if backend_config.get("enabled", False):
                try:
                    if backend_name in backend_classes:
                        backend = backend_classes[backend_name](backend_config)
                        self.backends[backend_name] = backend
                        logger.info(f"Initialized {backend_name} backend")
                except Exception as e:
                    logger.error(f"Failed to initialize {backend_name} backend: {e}")

        # Set fallback chain
        self.fallback_chain = self.config.get("fallback_chain", ["memory"])

        # Set operation-specific backends
        self.operation_backends = self.config.get("operations", {})

        logger.info(f"Initialized {len(self.backends)} backends: {list(self.backends.keys())}")

    async def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all enabled backends - Runtime Truth principle.

        Returns:
            Dict mapping backend names to connection success
        """
        connection_results = {}

        connection_tasks = {}
        for name, backend in self.backends.items():
            connection_tasks[name] = backend.connect()

        # Run all connections in parallel
        results = await asyncio.gather(
            *connection_tasks.values(),
            return_exceptions=True
        )

        for name, result in zip(connection_tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to connect to {name}: {result}")
                connection_results[name] = False
            else:
                connection_results[name] = result

        successful = sum(1 for v in connection_results.values() if v)
        logger.info(f"Connected to {successful}/{len(connection_results)} backends")

        return connection_results

    async def disconnect_all(self):
        """Disconnect from all backends"""
        disconnect_tasks = [backend.disconnect() for backend in self.backends.values()]
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        logger.info("Disconnected from all backends")

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all backends.

        Returns:
            Dict mapping backend names to health status
        """
        health_results = {}

        for name, backend in self.backends.items():
            try:
                is_healthy = await backend.health_check()
                health_results[name] = is_healthy
            except Exception as e:
                logger.warning(f"Health check failed for {name}: {e}")
                health_results[name] = False

        return health_results

    def _select_backend(self, operation: str) -> Optional[KnowledgeGraphBackend]:
        """
        Select the best available backend for an operation.

        Args:
            operation: Operation type (add_knowledge, search, analyze, visualize)

        Returns:
            Best available backend or None
        """
        # Get operation-specific backends or use fallback chain
        backend_names = self.operation_backends.get(operation, self.fallback_chain)

        # Find first healthy backend
        for name in backend_names:
            if name in self.backends:
                backend = self.backends[name]
                if backend.is_healthy:
                    return backend

        # If no healthy backend in preferred list, check all backends
        for name, backend in self.backends.items():
            if backend.is_healthy:
                logger.warning(f"Using fallback backend {name} for operation {operation}")
                return backend

        logger.error(f"No healthy backend available for operation: {operation}")
        return None

    async def add_knowledge(
        self,
        source: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        use_graph: bool = True
    ) -> str:
        """
        Add knowledge to the graph.

        Args:
            source: Knowledge source identifier
            content: Knowledge content
            metadata: Optional metadata dictionary
            use_graph: Whether to use graph backend (vs document storage)

        Returns:
            ID of the added knowledge entry

        Raises:
            BackendUnavailableError: If all backends are unavailable
        """
        backend = self._select_backend("add_knowledge")
        if not backend:
            raise BackendUnavailableError("No backend available for add_knowledge")

        entry = KnowledgeEntry(
            source=source,
            content=content,
            metadata=metadata
        )

        try:
            entry_id = await backend.add_knowledge(entry)
            logger.info(f"Added knowledge: {entry_id} using {backend.get_backend_name()}")
            return entry_id
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            # Try fallback
            for fallback_name in self.fallback_chain:
                if fallback_name in self.backends and fallback_name != backend.get_backend_name():
                    fallback_backend = self.backends[fallback_name]
                    if fallback_backend.is_healthy:
                        try:
                            entry_id = await fallback_backend.add_knowledge(entry)
                            logger.info(f"Added knowledge using fallback {fallback_name}")
                            return entry_id
                        except Exception as fallback_error:
                            logger.warning(f"Fallback {fallback_name} also failed: {fallback_error}")
            raise KnowledgeGraphError(f"Failed to add knowledge: {e}")

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        use_graph: bool = True,
        limit: int = 10,
        offset: int = 0
    ) -> SearchResults:
        """
        Search knowledge in the graph.

        Args:
            query: Search query string
            filters: Optional filters (source, tags, date_after, etc.)
            use_graph: Whether to use graph-based search
            limit: Maximum number of results
            offset: Result offset for pagination

        Returns:
            SearchResults object with results and metadata

        Raises:
            BackendUnavailableError: If all backends are unavailable
        """
        backend = self._select_backend("search")
        if not backend:
            raise BackendUnavailableError("No backend available for search")

        try:
            results = await backend.search(query, filters, limit, offset)

            # Track performance
            backend_name = backend.get_backend_name()
            if backend_name not in self.performance_metrics:
                self.performance_metrics[backend_name] = []
            self.performance_metrics[backend_name].append(results.search_time_ms)

            logger.info(
                f"Search completed: {results.total_count} results "
                f"in {results.search_time_ms:.2f}ms using {backend_name}"
            )

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise KnowledgeGraphError(f"Search failed: {e}")

    async def analyze(
        self,
        analysis_type: str,
        target: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze the knowledge graph.

        Supported analysis types:
        - connected_components: Find connected components
        - entity_connections: Find most connected entities
        - knowledge_by_source: Analyze knowledge distribution by source
        - community_detection: Detect communities (KarateClub)
        - centrality: Calculate centrality measures
        - graph_statistics: General graph statistics

        Args:
            analysis_type: Type of analysis to perform
            target: Optional target entity/graph for analysis

        Returns:
            AnalysisResult with analysis findings

        Raises:
            BackendUnavailableError: If all backends are unavailable
        """
        backend = self._select_backend("analyze")
        if not backend:
            raise BackendUnavailableError("No backend available for analyze")

        try:
            result = await backend.analyze(analysis_type, target)

            logger.info(
                f"Analysis '{analysis_type}' completed in {result.analysis_time_ms:.2f}ms "
                f"using {backend.get_backend_name()}"
            )

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise KnowledgeGraphError(f"Analysis failed: {e}")

    async def analyze_with_karateclub(
        self,
        analysis_type: str,
        target: Union[str, nx.Graph],
        **params
    ) -> AnalysisResult:
        """
        Analyze graph using KarateClub algorithms (51 algorithms total).

        Analysis types:
        - communities: Detect communities using 10 algorithms
        - node_embeddings: Generate node embeddings using 32 algorithms
        - graph_embeddings: Generate graph embeddings using 10 algorithms
        - metrics: Compute comprehensive graph metrics
        - structure: Full structural analysis

        Args:
            analysis_type: Type of analysis (communities, node_embeddings, etc.)
            target: NetworkX graph or node identifier
            **params: Algorithm-specific parameters

        Returns:
            AnalysisResult with findings

        Example:
            >>> # Community detection
            >>> result = await kg.analyze_with_karateclub("communities", graph, algorithm="gemsec")
            >>>
            >>> # Node embeddings
            >>> result = await kg.analyze_with_karateclub("node_embeddings", graph, algorithm="node2vec")
            >>>
            >>> # Full structural analysis
            >>> result = await kg.analyze_with_karateclub("structure", graph)
        """
        if not self.karateclub:
            raise KnowledgeGraphError("KarateClub Analytics not initialized")

        try:
            if analysis_type == "communities":
                result = await self.karateclub.detect_communities(target, **params)

                # Convert to AnalysisResult
                return AnalysisResult(
                    status_code=200,
                    analysis_type="community_detection",
                    target=str(target),
                    results={
                        "num_communities": result.num_communities,
                        "communities": result.communities,
                        "modularity": result.modularity,
                        "algorithm": result.algorithm
                    },
                    backend_used="karateclub",
                    analysis_time_ms=result.execution_time_ms
                )

            elif analysis_type == "node_embeddings":
                result = await self.karateclub.generate_node_embeddings(target, **params)

                return AnalysisResult(
                    status_code=200,
                    analysis_type="node_embedding",
                    target=str(target),
                    results={
                        "num_nodes": result.num_nodes,
                        "embedding_dim": result.embedding_dim,
                        "algorithm": result.algorithm,
                        "embeddings_sample": dict(list(result.embeddings.items())[:5])
                    },
                    backend_used="karateclub",
                    analysis_time_ms=result.execution_time_ms
                )

            elif analysis_type == "graph_embeddings":
                # target should be list of graphs
                result = await self.karateclub.generate_graph_embeddings(target, **params)

                return AnalysisResult(
                    status_code=200,
                    analysis_type="graph_embedding",
                    target="multiple_graphs",
                    results={
                        "num_graphs": result.num_graphs,
                        "embedding_dim": result.embedding_dim,
                        "algorithm": result.algorithm
                    },
                    backend_used="karateclub",
                    analysis_time_ms=result.execution_time_ms
                )

            elif analysis_type == "metrics":
                result = await self.karateclub.compute_graph_metrics(target)

                return AnalysisResult(
                    status_code=200,
                    analysis_type="graph_metrics",
                    target=str(target),
                    results={
                        "num_nodes": result.num_nodes,
                        "num_edges": result.num_edges,
                        "density": result.density,
                        "avg_clustering": result.avg_clustering,
                        "is_connected": result.is_connected,
                        "num_components": result.num_components,
                        "diameter": result.diameter,
                        "avg_path_length": result.avg_path_length
                    },
                    backend_used="karateclub",
                    analysis_time_ms=result.execution_time_ms
                )

            elif analysis_type == "structure":
                result = await self.karateclub.analyze_graph_structure(target, **params)

                return AnalysisResult(
                    status_code=200,
                    analysis_type="structure_analysis",
                    target=str(target),
                    results={
                        "communities": {
                            "num_communities": result.communities.num_communities,
                            "modularity": result.communities.modularity,
                            "algorithm": result.communities.algorithm
                        },
                        "metrics": {
                            "num_nodes": result.metrics.num_nodes,
                            "density": result.metrics.density,
                            "avg_clustering": result.metrics.avg_clustering
                        },
                        "centrality_sample": dict(list(result.centrality.get('pagerank', {}).items())[:5])
                    },
                    backend_used="karateclub",
                    analysis_time_ms=result.execution_time_ms
                )

            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")

        except Exception as e:
            logger.error(f"KarateClub analysis failed: {e}")
            raise KnowledgeGraphError(f"KarateClub analysis failed: {e}")

    async def get_similar_knowledge(
        self,
        query: str,
        graph: nx.Graph,
        top_k: int = 10,
        index_name: str = 'default'
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar knowledge using KarateClub embeddings.

        This is a convenience method that uses KarateClub retrieval.

        Args:
            query: Query node ID or search term
            graph: NetworkX graph to search
            top_k: Number of similar items to return
            index_name: Embedding index to use

        Returns:
            List of similar nodes with metadata

        Example:
            >>> similar = await kg.get_similar_knowledge("machine_learning", graph, top_k=10)
            >>> for node in similar:
            ...     print(f"{node['node']}: {node['similarity']:.3f}")
        """
        if not self.karateclub:
            raise KnowledgeGraphError("KarateClub Analytics not initialized")

        try:
            # Create retrieval engine
            retrieval = KarateClubRetrieval(self.karateclub)

            # Generate embeddings if not exists
            if index_name not in retrieval.indices:
                await retrieval.generate_embeddings_for_kg(graph, index_name=index_name)

            # Retrieve similar nodes
            similar_nodes = await retrieval.retrieve_similar_nodes(
                query,
                index_name=index_name,
                top_k=top_k
            )

            # Convert to dict format
            return [
                {
                    'node': node.node,
                    'similarity': node.similarity,
                    'metadata': node.metadata
                }
                for node in similar_nodes
            ]

        except Exception as e:
            logger.error(f"Failed to retrieve similar knowledge: {e}")
            raise KnowledgeGraphError(f"Failed to retrieve similar knowledge: {e}")

    async def visualize(
        self,
        output_format: str = 'html',
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate visualization of the knowledge graph.

        Args:
            output_format: Output format (html, json, dot)
            options: Optional visualization parameters

        Returns:
            Visualization data or HTML string

        Raises:
            BackendUnavailableError: If all backends are unavailable
        """
        backend = self._select_backend("visualize")
        if not backend:
            raise BackendUnavailableError("No backend available for visualize")

        try:
            visualization = await backend.visualize(output_format, options)
            logger.info(f"Generated visualization using {backend.get_backend_name()}")
            return visualization

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            raise KnowledgeGraphError(f"Visualization failed: {e}")

    async def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics across all backends.

        Returns:
            Dictionary with statistics from all healthy backends
        """
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "backends": {}
        }

        for name, backend in self.backends.items():
            if backend.is_healthy:
                try:
                    backend_stats = await backend.get_statistics()
                    stats["backends"][name] = {
                        "node_count": backend_stats.node_count,
                        "edge_count": backend_stats.edge_count,
                        "metadata": backend_stats.metadata
                    }
                except Exception as e:
                    logger.warning(f"Failed to get stats from {name}: {e}")
                    stats["backends"][name] = {"error": str(e)}

        # Add performance metrics
        if self.performance_metrics:
            stats["performance"] = {}
            for backend_name, times in self.performance_metrics.items():
                if times:
                    stats["performance"][backend_name] = {
                        "avg_time_ms": sum(times) / len(times),
                        "min_time_ms": min(times),
                        "max_time_ms": max(times),
                        "total_operations": len(times)
                    }

        return stats

    async def batch_add_knowledge(
        self,
        entries: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Batch add multiple knowledge entries.

        Args:
            entries: List of dictionaries with keys: source, content, metadata

        Returns:
            List of entry IDs
        """
        backend = self._select_backend("add_knowledge")
        if not backend:
            raise BackendUnavailableError("No backend available for batch add")

        knowledge_entries = [
            KnowledgeEntry(
                source=entry["source"],
                content=entry["content"],
                metadata=entry.get("metadata")
            )
            for entry in entries
        ]

        try:
            ids = await backend.batch_add_knowledge(knowledge_entries)
            logger.info(f"Batch added {len(ids)} entries using {backend.get_backend_name()}")
            return ids
        except Exception as e:
            logger.error(f"Batch add failed: {e}")
            raise KnowledgeGraphError(f"Batch add failed: {e}")

    async def export_knowledge(self, backend_name: str, output_path: str):
        """
        Export all knowledge from a specific backend.

        Args:
            backend_name: Name of backend to export from
            output_path: Path to save export file
        """
        if backend_name not in self.backends:
            raise ValueError(f"Backend {backend_name} not found")

        backend = self.backends[backend_name]

        if not backend.is_healthy:
            raise BackendUnavailableError(f"Backend {backend_name} not healthy")

        # Export as JSON
        visualization = await backend.visualize("json")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write(visualization)

        logger.info(f"Exported knowledge from {backend_name} to {output_path}")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect_all()
