"""
KarateClub Backend Adapter for Unified Knowledge Graph Manager.

Provides graph analytics using KarateClub library.
Follows CLAUDE.md principles: Runtime Truth, Configuration Explicitness, UTC.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import numpy as np

from .base import (
    KnowledgeGraphBackend,
    BackendType,
    KnowledgeEntry,
    SearchResults,
    AnalysisResult,
    GraphStatistics
)

logger = logging.getLogger(__name__)


class KarateClubBackend(KnowledgeGraphBackend):
    """
    KarateClub backend adapter for graph analytics.

    This backend uses KarateClub for graph embedding and analysis operations.
    It maintains an in-memory graph structure and applies KarateClub algorithms.

    Note: This is primarily an analytics backend. For storage, use Neo4j or MongoDB.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend_type = BackendType.KARATECLUB
        self.graph = None  # NetworkX graph
        self.node_embeddings = None
        self._validate_config()

    def _validate_config(self):
        """Validate configuration"""
        # KarateClub mostly uses default configurations
        self.embedding_dim = self.config.get('embedding_dim', 128)
        self.random_state = self.config.get('random_state', 42)

        logger.info(f"KarateClub backend configured with embedding_dim={self.embedding_dim}")

    async def connect(self) -> bool:
        """Initialize KarateClub - Runtime Truth"""
        try:
            import networkx as nx

            # Create empty graph
            self.graph = nx.DiGraph()

            # Try importing KarateClub to verify installation
            try:
                import karateclub
                version = getattr(karateclub, '__version__', 'unknown')
                logger.info(f"KarateClub version: {version}")
            except ImportError:
                logger.warning("karateclub package not fully installed. Install with: pip install karateclub")
                # Still allow connection for basic graph operations

            self.is_healthy = True
            logger.info("KarateClub backend initialized")

            return True

        except ImportError:
            logger.error("networkx package not installed. Install with: pip install networkx")
            raise ImportError("networkx package required for KarateClubBackend")
        except Exception as e:
            logger.error(f"Failed to initialize KarateClub: {e}")
            raise ConnectionError(f"KarateClub initialization failed: {e}")

    async def disconnect(self) -> None:
        """Cleanup KarateClub resources"""
        self.graph = None
        self.node_embeddings = None
        self.is_healthy = False
        logger.info("Disconnected from KarateClub backend")

    async def health_check(self) -> bool:
        """Check KarateClub health"""
        return self.is_healthy and self.graph is not None

    async def add_knowledge(self, entry: KnowledgeEntry) -> str:
        """Add knowledge as graph node"""
        if not self.is_healthy:
            raise ConnectionError("KarateClub backend not healthy")

        start_time = datetime.utcnow()

        try:
            import networkx as nx

            # Create node ID
            node_id = f"{entry.source}_{hash(entry.content) % 10000}"

            # Add node to graph
            self.graph.add_node(
                node_id,
                source=entry.source,
                content=entry.content,
                metadata=entry.metadata or {},
                timestamp=entry.timestamp,
                node_type="knowledge"
            )

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"Added knowledge to KarateClub graph in {elapsed_ms:.2f}ms: {node_id}")

            return node_id

        except Exception as e:
            logger.error(f"Failed to add knowledge to KarateClub: {e}")
            raise ConnectionError(f"KarateClub add_knowledge failed: {e}")

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> SearchResults:
        """Search in KarateClub graph using NetworkX"""
        if not self.is_healthy:
            raise ConnectionError("KarateClub backend not healthy")

        start_time = datetime.utcnow()

        try:
            import networkx as nx

            results = []

            # Simple content-based search
            for node_id, node_data in self.graph.nodes(data=True):
                # Apply filters
                if filters and "source" in filters:
                    if node_data.get("source") != filters["source"]:
                        continue

                # Content matching
                if query.lower() in node_data.get("content", "").lower():
                    results.append({
                        "id": node_id,
                        "source": node_data.get("source"),
                        "content": node_data.get("content"),
                        "metadata": node_data.get("metadata", {}),
                        "timestamp": node_data.get("timestamp")
                    })

            # Apply pagination
            paginated_results = results[offset:offset + limit]

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return SearchResults(
                query=query,
                results=paginated_results,
                total_count=len(results),
                backend_used="karateclub",
                search_time_ms=elapsed_ms,
                metadata={"filters": filters}
            )

        except Exception as e:
            logger.error(f"KarateClub search failed: {e}")
            raise ConnectionError(f"KarateClub search failed: {e}")

    async def analyze(
        self,
        analysis_type: str,
        target: Optional[str] = None
    ) -> AnalysisResult:
        """
        Perform graph analysis using KarateClub algorithms.

        Supported analysis types:
        - community_detection: Detect communities in the graph
        - node_embedding: Generate node embeddings
        - centrality: Calculate centrality measures
        - role_detection: Detect structural roles
        """
        if not self.is_healthy:
            raise ConnectionError("KarateClub backend not healthy")

        start_time = datetime.utcnow()

        try:
            import networkx as nx

            if len(self.graph.nodes) == 0:
                return AnalysisResult(
                    analysis_type=analysis_type,
                    target=target or "graph",
                    results={"error": "Graph is empty"},
                    backend_used="karateclub",
                    analysis_time_ms=0
                )

            if analysis_type == "community_detection":
                # Detect communities using Label Propagation
                try:
                    from karateclub import LabelPropagation

                    # Convert to undirected for community detection
                    undirected_graph = self.graph.to_undirected()

                    # Run community detection
                    model = LabelPropagation()
                    model.fit(undirected_graph)

                    # Get community assignments
                    membership = model.get_memberships()

                    # Count communities
                    communities = {}
                    for node, community_id in membership.items():
                        if community_id not in communities:
                            communities[community_id] = []
                        communities[community_id].append(node)

                    results = {
                        "num_communities": len(communities),
                        "communities": {
                            str(cid): members for cid, members in communities.items()
                        }
                    }

                except Exception as e:
                    logger.warning(f"KarateClub community detection failed: {e}")
                    # Fallback to NetworkX
                    communities = nx.community.greedy_modularity_communities(self.graph.to_undirected())
                    results = {
                        "num_communities": len(communities),
                        "communities": {str(i): list(c) for i, c in enumerate(communities)}
                    }

            elif analysis_type == "node_embedding":
                # Generate node embeddings
                try:
                    from karateclub import DeepWalk, Node2Vec

                    # Choose algorithm based on graph size
                    if len(self.graph.nodes) < 1000:
                        model = DeepWalk(dimensions=self.embedding_dim, walk_length=30)
                    else:
                        model = Node2Vec(dimensions=self.embedding_dim)

                    # Fit model
                    model.fit(self.graph)

                    # Get embeddings
                    self.node_embeddings = model.get_embedding()

                    # Convert to dict
                    embeddings_dict = {}
                    node_list = list(self.graph.nodes())
                    for i, node in enumerate(node_list):
                        if i < len(self.node_embeddings):
                            embeddings_dict[str(node)] = self.node_embeddings[i].tolist()

                    results = {
                        "embedding_dim": self.embedding_dim,
                        "num_nodes": len(embeddings_dict),
                        "embeddings_sample": dict(list(embeddings_dict.items())[:5])
                    }

                except Exception as e:
                    logger.warning(f"KarateClub node embedding failed: {e}")
                    results = {"error": str(e), "fallback": "Use Node2Vec manually"}

            elif analysis_type == "centrality":
                # Calculate centrality measures
                betweenness = nx.betweenness_centrality(self.graph)
                degree = nx.degree_centrality(self.graph)
                pagerank = nx.pagerank(self.graph)

                # Get top nodes
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
                top_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]
                top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]

                results = {
                    "top_betweenness": [{"node": n, "score": s} for n, s in top_betweenness],
                    "top_degree": [{"node": n, "score": s} for n, s in top_degree],
                    "top_pagerank": [{"node": n, "score": s} for n, s in top_pagerank]
                }

            elif analysis_type == "role_detection":
                # Detect structural roles
                try:
                    from karateclub import Role2Vec

                    # Fit role detection model
                    model = Role2Vec(dimensions=self.embedding_dim)
                    model.fit(self.graph)

                    # Get role embeddings
                    role_embeddings = model.get_embedding()

                    results = {
                        "num_roles": len(role_embeddings),
                        "embedding_dim": self.embedding_dim,
                        "message": "Role embeddings generated successfully"
                    }

                except Exception as e:
                    logger.warning(f"KarateClub role detection failed: {e}")
                    results = {"error": str(e)}

            elif analysis_type == "graph_statistics":
                # General graph statistics
                results = {
                    "num_nodes": self.graph.number_of_nodes(),
                    "num_edges": self.graph.number_of_edges(),
                    "density": nx.density(self.graph),
                    "is_directed": self.graph.is_directed(),
                    "is_connected": nx.is_weakly_connected(self.graph) if self.graph.is_directed() else nx.is_connected(self.graph)
                }

                # Add average clustering if undirected
                if not self.graph.is_directed():
                    results["avg_clustering"] = nx.average_clustering(self.graph.to_undirected())

            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return AnalysisResult(
                analysis_type=analysis_type,
                target=target or "graph",
                results=results,
                backend_used="karateclub",
                analysis_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"KarateClub analysis failed: {e}")
            raise ConnectionError(f"KarateClub analysis failed: {e}")

    async def get_statistics(self) -> GraphStatistics:
        """Get KarateClub graph statistics"""
        if not self.is_healthy:
            raise ConnectionError("KarateClub backend not healthy")

        try:
            import networkx as nx

            return GraphStatistics(
                node_count=self.graph.number_of_nodes(),
                edge_count=self.graph.number_of_edges(),
                backend="karateclub",
                metadata={
                    "density": nx.density(self.graph),
                    "is_directed": self.graph.is_directed(),
                    "num_selfloops": nx.number_of_selfloops(self.graph)
                },
                timestamp=datetime.utcnow().isoformat()
            )

        except Exception as e:
            logger.error(f"Failed to get KarateClub statistics: {e}")
            raise ConnectionError(f"KarateClub statistics failed: {e}")

    async def visualize(
        self,
        output_format: str = 'html',
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate visualization from KarateClub graph"""
        if not self.is_healthy:
            raise ConnectionError("KarateClub backend not healthy")

        try:
            import networkx as nx

            if output_format == 'json':
                # Export graph as JSON
                from networkx.readwrite import json_graph

                data = json_graph.node_link_data(self.graph)
                return json.dumps(data, indent=2)

            elif output_format == 'html':
                # Generate HTML with D3.js visualization
                from networkx.readwrite import json_graph

                # Convert to node-link format
                graph_data = json_graph.node_link_data(self.graph)

                # Limit to 100 nodes for performance
                if len(graph_data["nodes"]) > 100:
                    graph_data["nodes"] = graph_data["nodes"][:100]
                    graph_data["links"] = [l for l in graph_data["links"] if l["source"] < 100 and l["target"] < 100]

                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>KarateClub Graph Visualization</title>
                    <script src="https://d3js.org/d3.v7.min.js"></script>
                    <style>
                        body {{ margin: 0; }}
                        svg {{ width: 100vw; height: 100vh; }}
                        .node {{ stroke: #fff; stroke-width: 1.5px; }}
                        .link {{ stroke: #999; stroke-opacity: 0.6; }}
                    </style>
                </head>
                <body>
                    <svg></svg>
                    <script>
                        var graphData = {json.dumps(graph_data)};

                        var svg = d3.select("svg"),
                            width = +svg.attr("width"),
                            height = +svg.attr("height");

                        var simulation = d3.forceSimulation(graphData.nodes)
                            .force("link", d3.forceLink(graphData.links).id(d => d.id))
                            .force("charge", d3.forceManyBody().strength(-300))
                            .force("center", d3.forceCenter(width / 2, height / 2));

                        var link = svg.append("g")
                            .attr("class", "links")
                            .selectAll("line")
                            .data(graphData.links)
                            .enter().append("line")
                            .attr("class", "link");

                        var node = svg.append("g")
                            .attr("class", "nodes")
                            .selectAll("circle")
                            .data(graphData.nodes)
                            .enter().append("circle")
                            .attr("class", "node")
                            .attr("r", 5)
                            .call(d3.drag()
                                .on("start", dragstarted)
                                .on("drag", dragged)
                                .on("end", dragended));

                        node.append("title")
                            .text(d => d.id);

                        simulation.on("tick", () => {{
                            link
                                .attr("x1", d => d.source.x)
                                .attr("y1", d => d.source.y)
                                .attr("x2", d => d.target.x)
                                .attr("y2", d => d.target.y);

                            node
                                .attr("cx", d => d.x)
                                .attr("cy", d => d.y);
                        }});

                        function dragstarted(event, d) {{
                            if (!event.active) simulation.alphaTarget(0.3).restart();
                            d.fx = d.x;
                            d.fy = d.y;
                        }}

                        function dragged(event, d) {{
                            d.fx = event.x;
                            d.fy = event.y;
                        }}

                        function dragended(event, d) {{
                            if (!event.active) simulation.alphaTarget(0);
                            d.fx = null;
                            d.fy = null;
                        }}
                    </script>
                </body>
                </html>
                """
                return html

            else:
                raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"KarateClub visualization failed: {e}")
            raise ConnectionError(f"KarateClub visualization failed: {e}")

    async def clear_all(self) -> int:
        """Clear all knowledge from KarateClub graph"""
        if not self.is_healthy:
            raise ConnectionError("KarateClub backend not healthy")

        try:
            import networkx as nx

            count = self.graph.number_of_nodes()

            # Recreate empty graph
            self.graph = nx.DiGraph()
            self.node_embeddings = None

            logger.warning(f"Cleared {count} nodes from KarateClub graph")
            return count

        except Exception as e:
            logger.error(f"KarateClub clear failed: {e}")
            raise ConnectionError(f"KarateClub clear failed: {e}")
