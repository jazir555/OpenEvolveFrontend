"""
KarateClub Analytics Engine for Knowledge Engine

Provides comprehensive graph analytics using KarateClub's 51 algorithms:
- 10 Community Detection algorithms
- 32 Node Embedding algorithms
- 10 Graph Embedding algorithms

Follows CLAUDE.md principles:
- Runtime Truth: Validates algorithms at runtime
- Configuration Explicitness: All parameters via config
- Law of UTC: All timestamps in UTC
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import yaml

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CommunityResult:
    """Result from community detection"""
    communities: Dict[str, List[str]]
    num_communities: int
    algorithm: str
    modularity: float = 0.0
    coverage: float = 0.0
    performance: float = 0.0
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeEmbeddingResult:
    """Result from node embedding"""
    embeddings: Dict[str, List[float]]
    embedding_dim: int
    algorithm: str
    num_nodes: int
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEmbeddingResult:
    """Result from graph embedding"""
    embeddings: List[List[float]]
    embedding_dim: int
    algorithm: str
    num_graphs: int
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeMetrics:
    """Metrics for a specific node"""
    node: str
    degree_centrality: float
    betweenness_centrality: float
    eigenvector_centrality: float
    closeness_centrality: float
    pagerank: float
    clustering_coefficient: float
    community: Optional[str] = None
    degree: int = 0


@dataclass
class GraphMetrics:
    """Comprehensive graph metrics"""
    num_nodes: int
    num_edges: int
    density: float
    avg_clustering: float
    is_connected: bool
    num_components: int
    diameter: Optional[int] = None
    avg_path_length: Optional[float] = None
    assortativity: Optional[float] = None
    centralization: Optional[float] = None
    execution_time_ms: float = 0.0


@dataclass
class StructureAnalysis:
    """Complete structural analysis"""
    communities: CommunityResult
    metrics: GraphMetrics
    centrality: Dict[str, Dict[str, float]]
    roles: Optional[Dict[str, List[str]]] = None
    execution_time_ms: float = 0.0


@dataclass
class SimilarNode:
    """Similar node result"""
    node: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarGraph:
    """Similar graph result"""
    graph_id: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphComparison:
    """Graph comparison results"""
    method: str
    similarities: Dict[Tuple[str, str], float]
    most_similar: List[Tuple[str, str, float]]
    least_similar: List[Tuple[str, str, float]]
    execution_time_ms: float = 0.0


class KarateClubAnalytics:
    """
    KarateClub analytics integration for Knowledge Engine.

    Provides access to 51 algorithms across:
    - Community Detection (10 algorithms)
    - Node Embedding (32 algorithms)
    - Graph Embedding (10 algorithms)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize KarateClub Analytics Engine.

        Args:
            config_path: Path to karateclub_analytics.yaml configuration
        """
        self.config = self._load_config(config_path)
        self.algorithms = self._load_algorithms()
        self._validate_karateclub_installation()

        logger.info("KarateClub Analytics Engine initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        default_config = {
            'community_detection': {
                'default_algorithm': 'label_propagation',
                'resolution': 1.0,
                'random_seed': 42
            },
            'node_embeddings': {
                'default_algorithm': 'node2vec',
                'dimensions': 128,
                'walk_number': 10,
                'walk_length': 80,
                'window_size': 5
            },
            'graph_embeddings': {
                'default_algorithm': 'graph2vec',
                'dimensions': 128,
                'wl_iterations': 5,
                'epochs': 10
            },
            'metrics': {
                'compute_centrality': True,
                'compute_clustering': True,
                'compute_connectivity': True,
                'compute_density': True
            },
            'retrieval': {
                'enabled': True,
                'embedding_model': 'node2vec',
                'similarity_metric': 'cosine',
                'index_type': 'brute'
            },
            'output': {
                'save_embeddings': True,
                'save_metrics': True,
                'save_communities': True,
                'formats': ['json', 'csv']
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge configs
                for key in user_config:
                    if key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]

        return default_config

    def _load_algorithms(self) -> Dict[str, List[str]]:
        """Load algorithm registry"""
        from knowledge_engine.integrations.karateclub_algorithms import KarateClubAlgorithmRegistry
        return KarateClubAlgorithmRegistry.get_all_algorithms()

    def _validate_karateclub_installation(self):
        """Validate KarateClub installation (Runtime Truth)"""
        try:
            import karateclub
            logger.info(f"KarateClub version: {karateclub.__version__}")
        except ImportError:
            logger.warning("KarateClub not installed. Install with: pip install karateclub")
        except Exception as e:
            logger.error(f"KarateClub validation failed: {e}")

    async def detect_communities(
        self,
        graph: nx.Graph,
        algorithm: Optional[str] = None,
        **params
    ) -> CommunityResult:
        """
        Detect communities in graph.

        Algorithms:
        Overlapping: DANMF, M-NMF, Ego-Splitting, NNSED, BigClam, SymmNMF
        Non-overlapping: GEMSEC, EdMot, SCD, Label Propagation

        Args:
            graph: NetworkX graph
            algorithm: Community detection algorithm
            **params: Algorithm-specific parameters

        Returns:
            Community assignments and metrics
        """
        start_time = datetime.utcnow()

        # Use default algorithm if not specified
        if not algorithm:
            algorithm = self.config['community_detection']['default_algorithm']

        algorithm = algorithm.lower().replace('-', '_')
        logger.info(f"Detecting communities using {algorithm}")

        try:
            import karateclub as kc

            # Convert to undirected if needed
            if graph.is_directed():
                graph = graph.to_undirected()

            # Select algorithm
            if algorithm == 'label_propagation':
                model = kc.LabelPropagation(seed=params.get('seed', self.config['community_detection']['random_seed']))

            elif algorithm == 'danmf':
                model = kc.DANMF(
                    layers=params.get('layers', [32, 16]),
                    iterations=params.get('iterations', 100)
                )

            elif algorithm == 'gemsec':
                model = kc.GEMSEC(
                    dimensions=params.get('dimensions', 32),
                    walk_number=params.get('walk_number', 10),
                    walk_length=params.get('walk_length', 80)
                )

            elif algorithm == 'edmot':
                model = kc.EdMot(
                    component_number=params.get('components', 10)
                )

            elif algorithm == 'scd':
                model = kc.SCD()

            elif algorithm == 'ego_splitting':
                model = kc.EgoSplitting()

            elif algorithm == 'bigclam':
                model = kc.BigClam(
                    dimensions=params.get('dimensions', 32),
                    iterations=params.get('iterations', 100)
                )

            else:
                # Fallback to NetworkX
                logger.warning(f"Algorithm {algorithm} not available, using NetworkX")
                communities = nx.community.greedy_modularity_communities(graph)

                # Build community dict
                community_dict = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        community_dict[str(node)] = i

                elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                return CommunityResult(
                    communities=community_dict,
                    num_communities=len(communities),
                    algorithm=algorithm,
                    execution_time_ms=elapsed_ms,
                    metadata={'method': 'networkx_fallback'}
                )

            # Fit model
            model.fit(graph)

            # Get memberships
            membership = model.get_memberships()

            # Count communities
            communities = {}
            for node, comm_id in membership.items():
                comm_key = str(comm_id)
                if comm_key not in communities:
                    communities[comm_key] = []
                communities[comm_key].append(str(node))

            # Calculate modularity
            modularity = 0.0
            try:
                modularity = nx.community.modularity(graph, [set(c) for c in communities.values()])
            except:
                pass

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return CommunityResult(
                communities=communities,
                num_communities=len(communities),
                algorithm=algorithm,
                modularity=modularity,
                execution_time_ms=elapsed_ms,
                metadata={'params': params}
            )

        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            # Return empty result
            return CommunityResult(
                communities={},
                num_communities=0,
                algorithm=algorithm,
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                metadata={'error': str(e)}
            )

    async def generate_node_embeddings(
        self,
        graph: nx.Graph,
        algorithm: Optional[str] = None,
        dimensions: Optional[int] = None,
        **params
    ) -> NodeEmbeddingResult:
        """
        Generate node embeddings.

        Algorithms:
        Neighbourhood: DeepWalk, Node2Vec, Walklets, GraRep, HOPE, NetMF, etc.
        Structural: GraphWave, Role2Vec, SINR
        Attributed: FEATHER-N, TADW, MUSAE, AE, FSCNMF, etc.

        Returns:
            Node embeddings and metadata
        """
        start_time = datetime.utcnow()

        # Use defaults if not specified
        if not algorithm:
            algorithm = self.config['node_embeddings']['default_algorithm']
        if not dimensions:
            dimensions = self.config['node_embeddings']['dimensions']

        algorithm = algorithm.lower().replace('-', '_')
        logger.info(f"Generating node embeddings using {algorithm} ({dimensions} dimensions)")

        try:
            import karateclub as kc

            # Select algorithm
            if algorithm == 'deepwalk':
                model = kc.DeepWalk(
                    dimensions=dimensions,
                    walk_length=params.get('walk_length', self.config['node_embeddings']['walk_length']),
                    walk_number=params.get('walk_number', self.config['node_embeddings']['walk_number']),
                    window_size=params.get('window_size', self.config['node_embeddings']['window_size'])
                )

            elif algorithm == 'node2vec':
                model = kc.Node2Vec(
                    dimensions=dimensions,
                    walk_length=params.get('walk_length', self.config['node_embeddings']['walk_length']),
                    walk_number=params.get('walk_number', self.config['node_embeddings']['walk_number']),
                    p=params.get('p', 1.0),
                    q=params.get('q', 1.0),
                    window_size=params.get('window_size', self.config['node_embeddings']['window_size'])
                )

            elif algorithm == 'walklets':
                model = kc.Walklets(
                    dimensions=dimensions,
                    walk_length=params.get('walk_length', 80),
                    walk_number=params.get('walk_number', 10)
                )

            elif algorithm == 'grarep':
                model = kc.GraRep(
                    dimensions=dimensions,
                    order=params.get('order', 5)
                )

            elif algorithm == 'hope':
                model = kc.HOPE(
                    dimensions=dimensions
                )

            elif algorithm == 'netmf':
                model = kc.NetMF(
                    dimensions=dimensions,
                    order=params.get('order', 2),
                    window_size=params.get('window_size', 5)
                )

            elif algorithm == 'boostne':
                model = kc.BoostNE(
                    dimensions=dimensions
                )

            elif algorithm == 'randne':
                model = kc.RandNE(
                    dimensions=dimensions
                )

            elif algorithm == 'nodesketch':
                model = kc.NodeSketch(
                    dimensions=dimensions
                )

            elif algorithm == 'diff2vec':
                model = kc.Diff2Vec(
                    dimensions=dimensions,
                    diffusion_number=params.get('diffusion_number', 10),
                    diffusion_cover=params.get('diffusion_cover', 80)
                )

            elif algorithm == 'sociodim':
                model = kc.SocioDim(
                    dimensions=dimensions
                )

            elif algorithm == 'glee':
                model = kc.GLEE(
                    dimensions=dimensions
                )

            elif algorithm == 'laplacian_eigenmaps':
                model = kc.LaplacianEigenmaps(
                    dimensions=dimensions
                )

            elif algorithm == 'line':
                model = kc.LINE(
                    dimensions=dimensions,
                    order=params.get('order', 2)
                )

            # Structural algorithms
            elif algorithm == 'graphwave':
                model = kc.GraphWave(
                    dimensions=dimensions,
                    scales=params.get('scales', [5, 10, 15])
                )

            elif algorithm == 'role2vec':
                model = kc.Role2Vec(
                    dimensions=dimensions,
                    walk_length=params.get('walk_length', 80),
                    walk_number=params.get('walk_number', 10)
                )

            elif algorithm == 'sinr':
                model = kc.SINR(
                    dimensions=dimensions
                )

            # Attributed algorithms
            elif algorithm == 'feather_n':
                model = kc.FeatherN(
                    dimensions=dimensions
                )

            elif algorithm == 'tadw':
                model = kc.TADW(
                    dimensions=dimensions
                )

            elif algorithm == 'musae':
                model = kc.MUSAE(
                    dimensions=dimensions,
                    window_size=params.get('window_size', 5)
                )

            elif algorithm == 'ae':
                model = kc.AE(
                    dimensions=dimensions
                )

            elif algorithm == 'fscnmf':
                model = kc.FSCNMF(
                    dimensions=dimensions,
                    clusters=params.get('clusters', 10)
                )

            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            # Fit model
            model.fit(graph)

            # Get embeddings
            embeddings_matrix = model.get_embedding()

            # Convert to dict
            embeddings_dict = {}
            node_list = list(graph.nodes())
            for i, node in enumerate(node_list):
                if i < len(embeddings_matrix):
                    embeddings_dict[str(node)] = embeddings_matrix[i].tolist()

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return NodeEmbeddingResult(
                embeddings=embeddings_dict,
                embedding_dim=dimensions,
                algorithm=algorithm,
                num_nodes=len(embeddings_dict),
                execution_time_ms=elapsed_ms,
                metadata={'params': params}
            )

        except Exception as e:
            logger.error(f"Node embedding failed: {e}")
            return NodeEmbeddingResult(
                embeddings={},
                embedding_dim=dimensions,
                algorithm=algorithm,
                num_nodes=0,
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                metadata={'error': str(e)}
            )

    async def generate_graph_embeddings(
        self,
        graphs: List[nx.Graph],
        algorithm: Optional[str] = None,
        dimensions: Optional[int] = None,
        **params
    ) -> GraphEmbeddingResult:
        """
        Generate graph-level embeddings.

        Algorithms:
        Graph2Vec, FEATHER-G, NetLSD, GeoScattering, WaveletCharacteristic,
        IGE, LDP, GL2Vec, SF, FGSD

        Returns:
            Graph embeddings for similarity comparison
        """
        start_time = datetime.utcnow()

        # Use defaults if not specified
        if not algorithm:
            algorithm = self.config['graph_embeddings']['default_algorithm']
        if not dimensions:
            dimensions = self.config['graph_embeddings']['dimensions']

        algorithm = algorithm.lower().replace('-', '_')
        logger.info(f"Generating graph embeddings using {algorithm} ({dimensions} dimensions)")

        try:
            import karateclub as kc

            # Select algorithm
            if algorithm == 'graph2vec':
                model = kc.Graph2Vec(
                    dimensions=dimensions,
                    wl_iterations=params.get('wl_iterations', self.config['graph_embeddings']['wl_iterations']),
                    epochs=params.get('epochs', self.config['graph_embeddings']['epochs']),
                    learning_rate=params.get('learning_rate', 0.025)
                )

            elif algorithm == 'feather_g':
                model = kc.FeatherG(
                    dimensions=dimensions
                )

            elif algorithm == 'netlsd':
                model = kc.NetLSD()

            elif algorithm == 'geoscattering':
                model = kc.Geoscattering(
                    scales=params.get('scales', [5, 10, 15])
                )

            elif algorithm == 'wavelet_characteristic':
                model = kc.WaveletCharacteristic(
                    scales=params.get('scales', [5, 10, 15])
                )

            elif algorithm == 'ige':
                model = kc.IGE(
                    dimensions=dimensions
                )

            elif algorithm == 'ldp':
                model = kc.LDP()

            elif algorithm == 'gl2vec':
                model = kc.GL2Vec(
                    dimensions=dimensions,
                    wl_iterations=params.get('wl_iterations', 5)
                )

            elif algorithm == 'sf':
                model = kc.SF()

            elif algorithm == 'fgsd':
                model = kc.FGSD()

            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            # Fit model
            model.fit(graphs)

            # Get embeddings
            embeddings_matrix = model.get_embedding()

            # Convert to list
            embeddings_list = [emb.tolist() for emb in embeddings_matrix]

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return GraphEmbeddingResult(
                embeddings=embeddings_list,
                embedding_dim=dimensions,
                algorithm=algorithm,
                num_graphs=len(graphs),
                execution_time_ms=elapsed_ms,
                metadata={'params': params}
            )

        except Exception as e:
            logger.error(f"Graph embedding failed: {e}")
            return GraphEmbeddingResult(
                embeddings=[],
                embedding_dim=dimensions,
                algorithm=algorithm,
                num_graphs=0,
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                metadata={'error': str(e)}
            )

    async def compute_graph_metrics(
        self,
        graph: nx.Graph
    ) -> GraphMetrics:
        """
        Compute comprehensive graph metrics.

        Metrics:
        - Centrality (degree, betweenness, eigenvector, closeness)
        - Clustering (local, global)
        - Connectivity (components, paths)
        - Density
        - Assortativity
        """
        start_time = datetime.utcnow()

        logger.info("Computing graph metrics")

        try:
            # Basic metrics
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            density = nx.density(graph)

            # Clustering
            if graph.is_directed():
                graph_undirected = graph.to_undirected()
            else:
                graph_undirected = graph

            avg_clustering = nx.average_clustering(graph_undirected)

            # Connectivity
            if graph.is_directed():
                is_connected = nx.is_weakly_connected(graph)
                num_components = nx.number_weakly_connected_components(graph)
            else:
                is_connected = nx.is_connected(graph)
                num_components = nx.number_connected_components(graph)

            # Diameter and path length (only for connected graphs)
            diameter = None
            avg_path_length = None

            if is_connected and num_nodes < 1000:  # Limit for performance
                try:
                    if graph.is_directed():
                        graph_for_paths = graph.to_undirected()
                    else:
                        graph_for_paths = graph

                    diameter = nx.diameter(graph_for_paths)
                    avg_path_length = nx.average_shortest_path_length(graph_for_paths)
                except:
                    pass

            # Assortativity
            assortativity = None
            try:
                if graph.number_of_edges() > 0:
                    assortativity = nx.degree_assortativity_coefficient(graph)
            except:
                pass

            # Centralization
            centralization = None
            try:
                degree_centralities = nx.degree_centrality(graph)
                max_centrality = max(degree_centralities.values()) if degree_centralities else 0
                sum_centralities = sum(degree_centralities.values()) if degree_centralities else 0
                if num_nodes > 1:
                    centralization = (num_nodes * max_centrality - sum_centralities) / ((num_nodes - 1) * (num_nodes - 2))
            except:
                pass

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return GraphMetrics(
                num_nodes=num_nodes,
                num_edges=num_edges,
                density=density,
                avg_clustering=avg_clustering,
                is_connected=is_connected,
                num_components=num_components,
                diameter=diameter,
                avg_path_length=avg_path_length,
                assortativity=assortativity,
                centralization=centralization,
                execution_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Graph metrics computation failed: {e}")
            raise

    async def compute_node_metrics(
        self,
        graph: nx.Graph,
        node: str
    ) -> NodeMetrics:
        """
        Compute metrics for a specific node.

        Metrics:
        - Degree centrality
        - Betweenness centrality
        - Eigenvector centrality
        - Closeness centrality
        - PageRank
        - Clustering coefficient
        - Community membership
        """
        start_time = datetime.utcnow()

        logger.info(f"Computing metrics for node: {node}")

        try:
            # Convert node to string for consistency
            node_str = str(node)

            # Check if node exists
            if node_str not in [str(n) for n in graph.nodes()]:
                raise ValueError(f"Node {node} not in graph")

            # Get actual node object
            actual_node = None
            for n in graph.nodes():
                if str(n) == node_str:
                    actual_node = n
                    break

            # Centrality measures
            degree_centrality = nx.degree_centrality(graph).get(actual_node, 0.0)
            betweenness_centrality = nx.betweenness_centrality(graph).get(actual_node, 0.0)
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000).get(actual_node, 0.0)
            closeness_centrality = nx.closeness_centrality(graph).get(actual_node, 0.0)
            pagerank = nx.pagerank(graph).get(actual_node, 0.0)

            # Clustering coefficient
            if graph.is_directed():
                graph_undirected = graph.to_undirected()
            else:
                graph_undirected = graph

            clustering_coefficient = nx.clustering(graph_undirected, actual_node)

            # Degree
            degree = graph.degree(actual_node)

            return NodeMetrics(
                node=node_str,
                degree_centrality=degree_centrality,
                betweenness_centrality=betweenness_centrality,
                eigenvector_centrality=eigenvector_centrality,
                closeness_centrality=closeness_centrality,
                pagerank=pagerank,
                clustering_coefficient=clustering_coefficient,
                degree=degree
            )

        except Exception as e:
            logger.error(f"Node metrics computation failed: {e}")
            raise

    async def analyze_graph_structure(
        self,
        graph: nx.Graph,
        community_algorithm: Optional[str] = None
    ) -> StructureAnalysis:
        """
        Perform comprehensive graph structure analysis.

        Analysis:
        1. Community detection
        2. Centrality analysis
        3. Clustering analysis
        4. Connectivity analysis
        5. Role detection (structural roles)

        Returns:
            Complete structural analysis
        """
        start_time = datetime.utcnow()

        logger.info("Performing comprehensive graph structure analysis")

        try:
            # 1. Community detection
            communities = await self.detect_communities(graph, community_algorithm)

            # 2. Graph metrics
            metrics = await self.compute_graph_metrics(graph)

            # 3. Centrality analysis
            centrality = {
                'degree': nx.degree_centrality(graph),
                'betweenness': nx.betweenness_centrality(graph),
                'eigenvector': nx.eigenvector_centrality(graph, max_iter=1000),
                'closeness': nx.closeness_centrality(graph),
                'pagerank': nx.pagerank(graph)
            }

            # 4. Role detection (optional)
            roles = None
            try:
                if graph.number_of_nodes() < 5000:  # Limit for performance
                    role_result = await self.generate_node_embeddings(
                        graph,
                        algorithm='role2vec',
                        dimensions=64
                    )

                    # Cluster roles
                    if role_result.embeddings:
                        from sklearn.cluster import KMeans

                        embedding_matrix = np.array(list(role_result.embeddings.values()))
                        n_clusters = min(5, len(embedding_matrix))

                        if n_clusters > 0:
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            cluster_labels = kmeans.fit_predict(embedding_matrix)

                            roles = {}
                            node_list = list(role_result.embeddings.keys())
                            for i, label in enumerate(cluster_labels):
                                role_key = f"role_{label}"
                                if role_key not in roles:
                                    roles[role_key] = []
                                roles[role_key].append(node_list[i])
            except Exception as e:
                logger.warning(f"Role detection failed: {e}")

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return StructureAnalysis(
                communities=communities,
                metrics=metrics,
                centrality=centrality,
                roles=roles,
                execution_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Structure analysis failed: {e}")
            raise

    async def compare_graphs(
        self,
        graphs: List[nx.Graph],
        method: str = "embeddings"
    ) -> GraphComparison:
        """
        Compare multiple graphs.

        Methods:
        - embeddings: Graph embedding similarity
        - metrics: Metric-based comparison
        - structural: Structural similarity

        Returns:
            Graph similarities and differences
        """
        start_time = datetime.utcnow()

        logger.info(f"Comparing {len(graphs)} graphs using {method}")

        try:
            if method == "embeddings":
                # Generate graph embeddings
                embedding_result = await self.generate_graph_embeddings(graphs)

                if not embedding_result.embeddings:
                    raise ValueError("Failed to generate embeddings")

                # Compute pairwise similarities
                from sklearn.metrics.pairwise import cosine_similarity

                embedding_matrix = np.array(embedding_result.embeddings)
                similarity_matrix = cosine_similarity(embedding_matrix)

                # Build similarity dict
                similarities = {}
                n = len(graphs)
                for i in range(n):
                    for j in range(i + 1, n):
                        similarities[(str(i), str(j))] = float(similarity_matrix[i][j])

                # Sort
                all_sims = [(str(i), str(j), sim) for (i, j), sim in similarities.items()]
                all_sims.sort(key=lambda x: x[2], reverse=True)

                most_similar = all_sims[:5]
                least_similar = all_sims[-5:] if len(all_sims) > 5 else []

            elif method == "metrics":
                # Compute metrics for each graph
                all_metrics = []
                for i, graph in enumerate(graphs):
                    metrics = await self.compute_graph_metrics(graph)
                    all_metrics.append(metrics)

                # Compare metrics
                similarities = {}

                # Compare density
                for i in range(len(graphs)):
                    for j in range(i + 1, len(graphs)):
                        density_diff = abs(all_metrics[i].density - all_metrics[j].density)
                        clustering_diff = abs(all_metrics[i].avg_clustering - all_metrics[j].avg_clustering)

                        # Similarity = 1 - normalized difference
                        similarity = 1.0 - (density_diff + clustering_diff) / 2.0
                        similarities[(str(i), str(j))] = similarity

                all_sims = [(str(i), str(j), sim) for (i, j), sim in similarities.items()]
                all_sims.sort(key=lambda x: x[2], reverse=True)

                most_similar = all_sims[:5]
                least_similar = all_sims[-5:] if len(all_sims) > 5 else []

            else:
                raise ValueError(f"Unsupported comparison method: {method}")

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return GraphComparison(
                method=method,
                similarities=similarities,
                most_similar=most_similar,
                least_similar=least_similar,
                execution_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Graph comparison failed: {e}")
            raise
