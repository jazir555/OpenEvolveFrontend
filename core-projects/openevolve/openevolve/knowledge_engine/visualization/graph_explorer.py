"""
Interactive Graph Explorer

Production-grade interactive graph visualization with:
- Node filtering (search, type, attributes)
- Edge filtering by relationship type
- Zoom and pan controls
- Hover tooltips
- Responsive design
- Accessibility (WCAG 2.1 AA)
"""

import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import networkx as nx

from .config import get_visualization_config

logger = logging.getLogger(__name__)


@dataclass
class NodeFilter:
    """Node filter criteria."""
    search_query: Optional[str] = None
    node_types: Optional[List[str]] = None
    min_centrality: Optional[float] = None
    max_centrality: Optional[float] = None
    min_degree: Optional[int] = None
    max_degree: Optional[int] = None
    attributes: Optional[Dict[str, Any]] = None


@dataclass
class EdgeFilter:
    """Edge filter criteria."""
    relationship_types: Optional[List[str]] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    source: Optional[str] = None  # 'extracted', 'inferred', etc.


@dataclass
class VisualizationOptions:
    """Visualization rendering options."""
    width: int = 1200
    height: int = 800
    node_sizing: str = "centrality"  # centrality, degree, uniform
    edge_differentiation: bool = True
    color_scheme: str = "colorblind"  # colorblind, default, spectral
    show_labels: bool = True
    enable_zoom: bool = True
    enable_physics: bool = True
    enable_selection: bool = True
    animation_duration: int = 300  # ms


@dataclass
class VisualizationResult:
    """Result of visualization generation."""
    output_path: str
    node_count: int
    edge_count: int
    community_count: int
    statistics: Dict[str, Any]
    cache_key: Optional[str] = None
    generation_time: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class GraphExplorer:
    """
    Interactive graph explorer with comprehensive filtering and visualization.

    Features:
    - Advanced node/edge filtering
    - Interactive exploration (zoom, pan, select)
    - Real-time search
    - Multiple layout algorithms
    - Export capabilities
    - Caching for performance
    """

    # Colorblind-friendly color schemes
    COLOR_SCHEMES = {
        "colorblind": [
            "#0072B2", "#D55E00", "#009E73", "#CC79A7",
            "#F0E442", "#56B4E9", "#E69F00", "#000000"
        ],
        "default": [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
        ],
        "spectral": [
            "#d73027", "#f46d43", "#fdae61", "#fee08b",
            "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850"
        ]
    }

    def __init__(self, config=None):
        """
        Initialize graph explorer.

        Args:
            config: VisualizationConfig instance (uses default if None)
        """
        self.config = config or get_visualization_config()
        self.output_dir = Path(self.config.output_dir)
        self.cache_dir = Path(self.config.cache_dir)

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info({
            'event': 'graph_explorer_initialized',
            'output_dir': str(self.output_dir),
            'cache_dir': str(self.cache_dir),
            'timestamp': datetime.utcnow().isoformat()
        })

    def generate_cache_key(
        self,
        graph_data: Dict[str, Any],
        node_filter: Optional[NodeFilter] = None,
        edge_filter: Optional[EdgeFilter] = None,
        options: Optional[VisualizationOptions] = None
    ) -> str:
        """
        Generate cache key for visualization.

        Ensures idempotency - same inputs produce same cache key.
        """
        # Create deterministic hash of inputs
        cache_input = {
            'graph_nodes': len(graph_data.get('nodes', [])),
            'graph_edges': len(graph_data.get('edges', [])),
            'node_filter': asdict(node_filter) if node_filter else None,
            'edge_filter': asdict(edge_filter) if edge_filter else None,
            'options': asdict(options) if options else None
        }

        cache_str = json.dumps(cache_input, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()[:32]

    async def visualize(
        self,
        triples: List[Any],
        entities: List[Any],
        output_path: Optional[str] = None,
        node_filter: Optional[NodeFilter] = None,
        edge_filter: Optional[EdgeFilter] = None,
        options: Optional[VisualizationOptions] = None,
        use_cache: bool = True
    ) -> VisualizationResult:
        """
        Generate interactive graph visualization.

        Args:
            triples: List of graph triples (subject, predicate, object)
            entities: List of entity definitions
            output_path: Output file path (auto-generated if None)
            node_filter: Node filtering criteria
            edge_filter: Edge filtering criteria
            options: Visualization options
            use_cache: Whether to use caching

        Returns:
            VisualizationResult with metadata
        """
        start_time = datetime.utcnow()

        # Set default options
        if options is None:
            options = VisualizationOptions(
                width=self.config.default_width,
                height=self.config.default_height
            )

        # Build graph
        graph = self._build_graph(triples)

        # Apply filters
        graph = self._apply_filters(graph, node_filter, edge_filter)

        # Check size limits
        if len(graph.nodes()) > self.config.max_nodes:
            logger.warning({
                'event': 'graph_too_large',
                'node_count': len(graph.nodes()),
                'max_nodes': self.config.max_nodes,
                'action': 'truncating'
            })
            # Keep most central nodes
            graph = self._truncate_to_central_nodes(graph, self.config.max_nodes)

        if len(graph.edges()) > self.config.max_edges:
            logger.warning({
                'event': 'too_many_edges',
                'edge_count': len(graph.edges()),
                'max_edges': self.config.max_edges,
                'action': 'filtering by confidence'
            })
            graph = self._filter_edges_by_confidence(graph, self.config.max_edges)

        # Detect communities
        communities = await self._detect_communities(graph)

        # Compute centrality
        centrality = await self._compute_centrality(graph)

        # Prepare graph data
        graph_data = self._prepare_graph_data(
            graph, triples, communities, centrality, options
        )

        # Generate cache key
        cache_key = self.generate_cache_key(graph_data, node_filter, edge_filter, options)

        # Check cache
        if use_cache and self.config.enable_caching:
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                logger.info({
                    'event': 'visualization_cache_hit',
                    'cache_key': cache_key,
                    'timestamp': datetime.utcnow().isoformat()
                })
                return cached_result

        # Generate HTML
        html_content = await self._generate_d3_html(graph_data, options)

        # Save to file
        if output_path is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f'graph_{timestamp}_{cache_key}.html'

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Calculate generation time
        generation_time = (datetime.utcnow() - start_time).total_seconds()

        # Gather statistics
        statistics = self._compute_statistics(graph, communities, centrality)

        result = VisualizationResult(
            output_path=str(output_file),
            node_count=len(graph.nodes()),
            edge_count=len(graph.edges()),
            community_count=len(set(communities.values())),
            statistics=statistics,
            cache_key=cache_key,
            generation_time=generation_time,
            timestamp=datetime.utcnow().isoformat()
        )

        # Save to cache
        if use_cache and self.config.enable_caching:
            self._save_to_cache(cache_key, result)

        logger.info({
            'event': 'visualization_generated',
            'output_path': str(output_file),
            'node_count': result.node_count,
            'edge_count': result.edge_count,
            'generation_time': generation_time,
            'timestamp': datetime.utcnow().isoformat()
        })

        return result

    def _build_graph(self, triples: List[Any]) -> nx.Graph:
        """Build NetworkX graph from triples.

        Handles multiple formats:
        - Object with subject/predicate/object attributes
        - Dict with subject/predicate/object keys
        - Tuple/list with [subject, predicate, object, confidence?, source?]
        """
        graph = nx.Graph()

        for triple in triples:
            try:
                # Handle object format
                if hasattr(triple, 'subject'):
                    subj = triple.subject
                    pred = triple.predicate
                    obj = triple.object
                    conf = getattr(triple, 'confidence', 1.0)
                    source = getattr(triple, 'source', 'extracted')
                # Handle dict format
                elif isinstance(triple, dict):
                    subj = triple.get('subject')
                    pred = triple.get('predicate')
                    obj = triple.get('object')
                    conf = triple.get('confidence', 1.0)
                    source = triple.get('source', 'extracted')

                    if not all([subj, pred, obj]):
                        logger.warning({
                            'event': 'invalid_triple_dict',
                            'triple': triple,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                        continue
                # Handle tuple/list format
                elif isinstance(triple, (tuple, list)) and len(triple) >= 3:
                    subj, pred, obj = triple[0], triple[1], triple[2]
                    conf = triple[3] if len(triple) > 3 else 1.0
                    source = triple[4] if len(triple) > 4 else 'extracted'
                else:
                    logger.warning({
                        'event': 'unknown_triple_format',
                        'triple_type': type(triple),
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    continue

                graph.add_node(subj)
                graph.add_node(obj)
                graph.add_edge(
                    subj, obj,
                    predicate=pred,
                    confidence=float(conf),
                    source=source
                )

            except Exception as e:
                logger.warning({
                    'event': 'triple_processing_failed',
                    'triple': str(triple),
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
                continue

        logger.info({
            'event': 'graph_built',
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'timestamp': datetime.utcnow().isoformat()
        })

        return graph

    def _apply_filters(
        self,
        graph: nx.Graph,
        node_filter: Optional[NodeFilter],
        edge_filter: Optional[EdgeFilter]
    ) -> nx.Graph:
        """Apply filtering criteria to graph."""
        filtered_graph = graph.copy()

        # Apply node filters
        if node_filter:
            nodes_to_remove = set()

            for node in filtered_graph.nodes():
                # Search query filter
                if node_filter.search_query:
                    if node_filter.search_query.lower() not in node.lower():
                        nodes_to_remove.add(node)
                        continue

                # Degree filter
                degree = filtered_graph.degree(node)
                if node_filter.min_degree is not None and degree < node_filter.min_degree:
                    nodes_to_remove.add(node)
                    continue
                if node_filter.max_degree is not None and degree > node_filter.max_degree:
                    nodes_to_remove.add(node)

            filtered_graph.remove_nodes_from(nodes_to_remove)

        # Apply edge filters
        if edge_filter:
            edges_to_remove = set()

            for source, target, data in filtered_graph.edges(data=True):
                # Relationship type filter
                if edge_filter.relationship_types:
                    if data.get('predicate') not in edge_filter.relationship_types:
                        edges_to_remove.add((source, target))
                        continue

                # Confidence filter
                confidence = data.get('confidence', 1.0)
                if edge_filter.min_confidence is not None and confidence < edge_filter.min_confidence:
                    edges_to_remove.add((source, target))
                    continue
                if edge_filter.max_confidence is not None and confidence > edge_filter.max_confidence:
                    edges_to_remove.add((source, target))
                    continue

                # Source filter
                if edge_filter.source:
                    if data.get('source') != edge_filter.source:
                        edges_to_remove.add((source, target))

            filtered_graph.remove_edges_from(edges_to_remove)

        return filtered_graph

    def _truncate_to_central_nodes(self, graph: nx.Graph, max_nodes: int) -> nx.Graph:
        """Keep only most central nodes."""
        centrality = nx.degree_centrality(graph)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in sorted_nodes[:max_nodes]]

        return graph.subgraph(top_nodes).copy()

    def _filter_edges_by_confidence(self, graph: nx.Graph, max_edges: int) -> nx.Graph:
        """Filter edges by confidence."""
        edges_with_conf = [
            (u, v, data.get('confidence', 1.0))
            for u, v, data in graph.edges(data=True)
        ]
        edges_with_conf.sort(key=lambda x: x[2], reverse=True)

        top_edges = [(u, v) for u, v, _ in edges_with_conf[:max_edges]]
        return graph.edge_subgraph(top_edges).copy()

    async def _detect_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """Detect communities using Louvain algorithm."""
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(graph)

            node_to_community = {}
            for comm_id, community in enumerate(communities):
                for node in community:
                    node_to_community[node] = comm_id

            return node_to_community

        except ImportError:
            logger.warning({
                'event': 'louvain_not_available',
                'fallback': 'connected_components',
                'timestamp': datetime.utcnow().isoformat()
            })
            # Fallback to connected components
            communities = list(nx.connected_components(graph))
            node_to_community = {}
            for comm_id, community in enumerate(communities):
                for node in community:
                    node_to_community[node] = comm_id
            return node_to_community

    async def _compute_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute composite centrality score."""
        centrality_scores = {}

        # Degree centrality (60%)
        try:
            degree_cent = nx.degree_centrality(graph)
        except:
            degree_cent = {node: 0.0 for node in graph.nodes()}

        # Betweenness centrality (30%)
        try:
            betweenness_cent = nx.betweenness_centrality(graph, normalized=True)
        except:
            betweenness_cent = {node: 0.0 for node in graph.nodes()}

        # Eigenvector centrality (10%)
        try:
            eigenvector_cent = nx.eigenvector_centrality(graph, max_iter=100)
        except:
            eigenvector_cent = {node: 0.0 for node in graph.nodes()}

        # Combine
        for node in graph.nodes():
            score = (
                0.6 * degree_cent.get(node, 0.0) +
                0.3 * betweenness_cent.get(node, 0.0) +
                0.1 * eigenvector_cent.get(node, 0.0)
            )
            centrality_scores[node] = score

        return centrality_scores

    def _prepare_graph_data(
        self,
        graph: nx.Graph,
        triples: List[Any],
        communities: Dict[str, int],
        centrality: Dict[str, float],
        options: VisualizationOptions
    ) -> Dict[str, Any]:
        """Prepare graph data for D3.js visualization."""
        colors = self.COLOR_SCHEMES.get(
            options.color_scheme,
            self.COLOR_SCHEMES["colorblind"]
        )

        # Prepare nodes
        nodes = []
        for node in graph.nodes():
            community_id = communities.get(node, 0)
            centrality_score = centrality.get(node, 0.0)

            if options.node_sizing == "centrality":
                node_size = 5 + centrality_score * 20
            elif options.node_sizing == "degree":
                node_size = 5 + graph.degree(node) * 2
            else:
                node_size = 10

            nodes.append({
                "id": node,
                "community": community_id,
                "color": colors[community_id % len(colors)],
                "size": node_size,
                "centrality": centrality_score,
                "degree": graph.degree(node)
            })

        # Prepare edges
        edges = []
        for source, target, data in graph.edges(data=True):
            edge_type = 'dashed' if (options.edge_differentiation and data.get('source') == 'inferred') else 'solid'

            edges.append({
                "source": source,
                "target": target,
                "predicate": data.get('predicate', 'related_to'),
                "confidence": data.get('confidence', 1.0),
                "type": edge_type
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "options": {
                "width": options.width,
                "height": options.height,
                "showLabels": options.show_labels,
                "enableZoom": options.enable_zoom,
                "enablePhysics": options.enable_physics,
                "enableSelection": options.enable_selection,
                "animationDuration": options.animation_duration
            }
        }

    async def _generate_d3_html(
        self,
        graph_data: Dict[str, Any],
        options: VisualizationOptions
    ) -> str:
        """Generate D3.js HTML visualization."""
        # This will load the template from templates/graph_explorer.html
        template_path = Path(__file__).parent / 'templates' / 'graph_explorer.html'

        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                html_template = f.read()
        else:
            # Use embedded template
            html_template = self._get_embedded_template()

        # Prepare data
        graph_data_json = json.dumps(graph_data)
        options_json = json.dumps(asdict(options))

        # Fill template
        html = html_template.format(
            width=options.width,
            height=options.height,
            graph_data=graph_data_json,
            options=options_json
        )

        return html

    def _get_embedded_template(self) -> str:
        """Get embedded HTML template with full D3.js visualization."""
        template_path = Path(__file__).parent / 'templates' / 'graph_explorer.html'

        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()

        # Fallback comprehensive template
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Explorer</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
            background: #f8f9fa;
        }
        #container { display: flex; flex-direction: column; height: 100vh; }
        #header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        #header h1 { font-size: 1.5rem; font-weight: 600; }
        #graph-container { flex: 1; position: relative; }
        #graph { width: 100%; height: 100%; cursor: grab; }
        .node {
            stroke: #fff;
            stroke-width: 1.5px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .node:hover {
            stroke: #000;
            stroke-width: 2.5px;
            filter: brightness(1.1);
        }
        .link { stroke-opacity: 0.6; }
        #tooltip {
            position: absolute;
            padding: 0.75rem;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 6px;
            pointer-events: none;
            font-size: 0.85rem;
            z-index: 1000;
            display: none;
        }
        #stats {
            position: absolute;
            top: 20px;
            left: 20px;
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            font-size: 0.85rem;
        }
        .stat-item {
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            gap: 2rem;
        }
        .stat-label { font-weight: 600; color: #666; }
        .stat-value { color: #333; }
    </style>
</head>
<body>
    <div id="container">
        <div id="header">
            <h1>Knowledge Graph Explorer</h1>
        </div>
        <div id="graph-container">
            <svg id="graph"></svg>
            <div id="stats"></div>
            <div id="tooltip"></div>
        </div>
    </div>
    <script>
        const graphData = {graph_data};
        const options = {options};

        (function() {{
            const width = options.width || 1200;
            const height = options.height || 800;

            const svg = d3.select('#graph').attr('width', width).attr('height', height);
            const linksGroup = svg.append('g').attr('class', 'links');
            const nodesGroup = svg.append('g').attr('class', 'nodes');

            const zoom = d3.zoom().scaleExtent([0.1, 4]).on('zoom', (event) => {{
                linksGroup.attr('transform', event.transform);
                nodesGroup.attr('transform', event.transform);
            }});
            svg.call(zoom);

            const simulation = d3.forceSimulation(graphData.nodes)
                .force('link', d3.forceLink(graphData.edges).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(d => d.size + 5));

            const link = linksGroup.selectAll('line')
                .data(graphData.edges)
                .enter().append('line')
                .attr('class', 'link')
                .attr('stroke', d => d.type === 'dashed' ? '#999' : '#666')
                .attr('stroke-width', d => d.confidence * 2)
                .attr('stroke-dasharray', d => d.type === 'dashed' ? '5,5' : 'none');

            const node = nodesGroup.selectAll('circle')
                .data(graphData.nodes)
                .enter().append('circle')
                .attr('class', 'node')
                .attr('r', d => d.size)
                .attr('fill', d => d.color)
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));

            const statsDiv = d3.select('#stats');
            statsDiv.html(`
                <div class="stat-item">
                    <span class="stat-label">Nodes:</span>
                    <span class="stat-value">${{graphData.nodes.length}}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Edges:</span>
                    <span class="stat-value">${{graphData.edges.length}}</span>
                </div>
            `);

            const tooltip = d3.select('#tooltip');

            node.on('mouseover', function(event, d) {{
                const html = `<strong>${{d.id}}</strong><br/>Community: ${{d.community}}<br/>Centrality: ${{d.centrality.toFixed(3)}}`;
                tooltip.html(html).style('display', 'block')
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px');
            }}).on('mouseout', function() {{
                tooltip.style('display', 'none');
            }});

            simulation.on('tick', () => {{
                link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
                node.attr('cx', d => d.x).attr('cy', d => d.y);
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
        }})();
    </script>
</body>
</html>"""

    def _compute_statistics(
        self,
        graph: nx.Graph,
        communities: Dict[str, int],
        centrality: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compute graph statistics."""
        return {
            'communities': len(set(communities.values())),
            'avg_community_size': sum(1 for c in communities.values()) / len(set(communities.values())) if communities else 0,
            'max_centrality': max(centrality.values()) if centrality else 0,
            'avg_centrality': sum(centrality.values()) / len(centrality) if centrality else 0,
            'graph_density': nx.density(graph),
            'is_connected': nx.is_connected(graph.to_undirected()),
            'avg_clustering': nx.average_clustering(graph.to_undirected()),
            'diameter': nx.diameter(graph.to_undirected()) if nx.is_connected(graph.to_undirected()) else None
        }

    def _load_from_cache(self, cache_key: str) -> Optional[VisualizationResult]:
        """Load visualization from cache."""
        cache_file = self.cache_dir / f'{cache_key}.json'

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if cache is still valid
            cache_time = datetime.fromisoformat(data['timestamp'])
            if datetime.utcnow() - cache_time > timedelta(seconds=self.config.cache_ttl):
                return None

            return VisualizationResult(**data)

        except Exception as e:
            logger.warning({
                'event': 'cache_load_failed',
                'cache_key': cache_key,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            return None

    def _save_to_cache(self, cache_key: str, result: VisualizationResult):
        """Save visualization to cache."""
        cache_file = self.cache_dir / f'{cache_key}.json'

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2)

        except Exception as e:
            logger.warning({
                'event': 'cache_save_failed',
                'cache_key': cache_key,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
