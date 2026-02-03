"""
Community-Based Graph Visualization

Production-grade community visualization with:
- Community color coding
- Force-directed layouts per community
- Inter-community relationships
- Community hierarchy display
- Community filtering
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict

import networkx as nx

from .config import get_visualization_config

logger = logging.getLogger(__name__)


@dataclass
class CommunityInfo:
    """Information about a community."""
    id: int
    nodes: List[str]
    internal_edges: int
    external_edges: int
    density: float
    centrality: float
    color: str


@dataclass
class CommunityVisualizationOptions:
    """Options for community visualization."""
    width: int = 1200
    height: int = 800
    layout_algorithm: str = "force_community"  # force_community, force_global, hierarchical
    color_scheme: str = "colorblind"
    show_community_labels: bool = True
    show_inter_community_edges: bool = True
    enable_community_filtering: bool = True
    community_spacing: float = 1.5  # Multiplier for spacing between communities
    node_sizing: str = "centrality"


class CommunityVisualizer:
    """
    Community-based graph visualization.

    Features:
    - Automatic community detection (Louvain)
    - Community-centric layouts
    - Inter-community relationship visualization
    - Community hierarchy
    - Filtering by community
    """

    COLOR_SCHEMES = {
        "colorblind": [
            "#0072B2", "#D55E00", "#009E73", "#CC79A7",
            "#F0E442", "#56B4E9", "#E69F00", "#000000"
        ],
        "default": [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
        ],
        "pastel": [
            "#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4",
            "#fed9a6", "#ffffcc", "#e5d8bd", "#fddaec"
        ]
    }

    def __init__(self, config=None):
        """Initialize community visualizer."""
        self.config = config or get_visualization_config()
        self.output_dir = Path(self.config.output_dir) / 'community'
        self.cache_dir = Path(self.config.cache_dir) / 'community'

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info({
            'event': 'community_visualizer_initialized',
            'output_dir': str(self.output_dir),
            'timestamp': datetime.utcnow().isoformat()
        })

    async def visualize_communities(
        self,
        triples: List[Any],
        entities: List[Any],
        output_path: Optional[str] = None,
        options: Optional[CommunityVisualizationOptions] = None
    ) -> Dict[str, Any]:
        """
        Generate community visualization.

        Args:
            triples: List of graph triples
            entities: List of entity definitions
            output_path: Output file path
            options: Visualization options

        Returns:
            Result dictionary with metadata
        """
        if options is None:
            options = CommunityVisualizationOptions(
                width=self.config.default_width,
                height=self.config.default_height
            )

        start_time = datetime.utcnow()

        # Build graph
        graph = self._build_graph(triples)

        # Detect communities
        communities = await self._detect_communities(graph)

        # Analyze communities
        community_info = self._analyze_communities(graph, communities, options)

        # Compute community hierarchy
        hierarchy = self._compute_community_hierarchy(graph, communities)

        # Compute inter-community edges
        inter_community_edges = self._compute_inter_community_edges(graph, communities)

        # Generate layout
        layout = self._generate_community_layout(
            graph, communities, community_info, options
        )

        # Prepare visualization data
        viz_data = {
            'nodes': self._prepare_nodes(graph, communities, layout, options),
            'edges': self._prepare_edges(graph, communities, options),
            'communities': [asdict(c) for c in community_info],
            'hierarchy': hierarchy,
            'inter_community_edges': inter_community_edges,
            'options': asdict(options)
        }

        # Generate HTML
        html_content = await self._generate_community_html(viz_data, options)

        # Save to file
        if output_path is None:
            timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f'community_{timestamp_str}.html'

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        generation_time = (datetime.utcnow() - start_time).total_seconds()

        logger.info({
            'event': 'community_visualization_generated',
            'output_path': str(output_file),
            'num_communities': len(community_info),
            'generation_time': generation_time,
            'timestamp': datetime.utcnow().isoformat()
        })

        return {
            'output_path': str(output_file),
            'num_communities': len(community_info),
            'generation_time': generation_time,
            'timestamp': datetime.utcnow().isoformat()
        }

    def _build_graph(self, triples: List[Any]) -> nx.Graph:
        """Build NetworkX graph from triples.

        Handles multiple formats:
        - Object with subject/predicate/object attributes
        - Dict with subject/predicate/object keys
        - Tuple/list with [subject, predicate, object, confidence?]
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
                # Handle dict format
                elif isinstance(triple, dict):
                    subj = triple.get('subject')
                    pred = triple.get('predicate')
                    obj = triple.get('object')
                    conf = triple.get('confidence', 1.0)

                    if not all([subj, pred, obj]):
                        continue
                # Handle tuple/list format
                elif isinstance(triple, (tuple, list)) and len(triple) >= 3:
                    subj, pred, obj = triple[0], triple[1], triple[2]
                    conf = triple[3] if len(triple) > 3 else 1.0
                else:
                    continue

                graph.add_edge(subj, obj, predicate=pred, confidence=float(conf))

            except Exception as e:
                logger.warning({
                    'event': 'community_triple_processing_failed',
                    'triple': str(triple),
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
                continue

        return graph

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

        except Exception as e:
            logger.warning({
                'event': 'community_detection_failed',
                'error': str(e),
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

    def _analyze_communities(
        self,
        graph: nx.Graph,
        communities: Dict[str, int],
        options: CommunityVisualizationOptions
    ) -> List[CommunityInfo]:
        """Analyze communities and extract statistics."""
        # Group nodes by community
        community_nodes: Dict[int, Set[str]] = {}
        for node, comm_id in communities.items():
            if comm_id not in community_nodes:
                community_nodes[comm_id] = set()
            community_nodes[comm_id].add(node)

        # Get color scheme
        colors = self.COLOR_SCHEMES.get(
            options.color_scheme,
            self.COLOR_SCHEMES["colorblind"]
        )

        # Analyze each community
        community_info_list = []

        for comm_id, nodes in community_nodes.items():
            # Create subgraph for this community
            subgraph = graph.subgraph(nodes)

            # Count internal edges
            internal_edges = subgraph.number_of_edges()

            # Count external edges
            external_edges = 0
            for node in nodes:
                for neighbor in graph.neighbors(node):
                    if neighbor not in nodes:
                        external_edges += 1
            external_edges //= 2  # Each edge counted twice

            # Compute density
            density = nx.density(subgraph) if len(nodes) > 1 else 0

            # Compute centrality (average degree centrality)
            centrality_scores = nx.degree_centrality(subgraph)
            avg_centrality = sum(centrality_scores.values()) / len(centrality_scores) if centrality_scores else 0

            community_info = CommunityInfo(
                id=comm_id,
                nodes=list(nodes),
                internal_edges=internal_edges,
                external_edges=external_edges,
                density=density,
                centrality=avg_centrality,
                color=colors[comm_id % len(colors)]
            )

            community_info_list.append(community_info)

        # Sort by centrality
        community_info_list.sort(key=lambda c: c.centrality, reverse=True)

        return community_info_list

    def _compute_community_hierarchy(
        self,
        graph: nx.Graph,
        communities: Dict[str, int]
    ) -> Dict[str, Any]:
        """Compute community hierarchy based on inter-community connections."""
        # Create community graph
        community_graph = nx.Graph()

        # Add edges between communities
        for source, target in graph.edges():
            comm_source = communities.get(source, -1)
            comm_target = communities.get(target, -1)

            if comm_source != comm_target:
                if community_graph.has_edge(comm_source, comm_target):
                    community_graph[comm_source][comm_target]['weight'] += 1
                else:
                    community_graph.add_edge(comm_source, comm_target, weight=1)

        # Compute hierarchy levels using community centrality
        community_centrality = nx.degree_centrality(community_graph)

        # Build hierarchy tree
        hierarchy = {
            'levels': {},
            'connections': []
        }

        # Group communities by centrality (3 levels: core, intermediate, peripheral)
        for comm_id, centrality in community_centrality.items():
            if centrality > 0.6:
                level = 'core'
            elif centrality > 0.3:
                level = 'intermediate'
            else:
                level = 'peripheral'

            if level not in hierarchy['levels']:
                hierarchy['levels'][level] = []
            hierarchy['levels'][level].append(comm_id)

        # Add inter-community connections
        for source, target, data in community_graph.edges(data=True):
            hierarchy['connections'].append({
                'source': source,
                'target': target,
                'weight': data['weight']
            })

        return hierarchy

    def _compute_inter_community_edges(
        self,
        graph: nx.Graph,
        communities: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Compute edges between communities."""
        inter_edges = []

        for source, target, data in graph.edges(data=True):
            comm_source = communities.get(source, -1)
            comm_target = communities.get(target, -1)

            if comm_source != comm_target:
                inter_edges.append({
                    'source': source,
                    'target': target,
                    'source_community': comm_source,
                    'target_community': comm_target,
                    'predicate': data.get('predicate', 'related_to'),
                    'confidence': data.get('confidence', 1.0)
                })

        return inter_edges

    def _generate_community_layout(
        self,
        graph: nx.Graph,
        communities: Dict[str, int],
        community_info: List[CommunityInfo],
        options: CommunityVisualizationOptions
    ) -> Dict[str, tuple]:
        """Generate layout for community visualization."""
        layout = {}

        if options.layout_algorithm == "force_community":
            layout = self._force_directed_community_layout(
                graph, communities, community_info, options
            )
        elif options.layout_algorithm == "force_global":
            layout = nx.spring_layout(graph, weight='confidence')
        elif options.layout_algorithm == "hierarchical":
            layout = self._hierarchical_layout(
                graph, communities, community_info, options
            )
        else:
            layout = nx.spring_layout(graph)

        return layout

    def _force_directed_community_layout(
        self,
        graph: nx.Graph,
        communities: Dict[str, int],
        community_info: List[CommunityInfo],
        options: CommunityVisualizationOptions
    ) -> Dict[str, tuple]:
        """Generate force-directed layout with community spacing."""
        # First, create community-level layout
        community_centers = self._compute_community_centers(community_info, options)

        # Then, layout nodes within each community
        layout = {}

        for comm_info in community_info:
            center_x, center_y = community_centers[comm_info.id]

            # Create subgraph for this community
            subgraph_nodes = comm_info.nodes
            subgraph = graph.subgraph(subgraph_nodes)

            # Layout within community (tighter layout)
            sub_layout = nx.spring_layout(
                subgraph,
                center=(center_x, center_y),
                scale=100.0  # Smaller scale for tighter clustering
            )

            layout.update(sub_layout)

        return layout

    def _compute_community_centers(
        self,
        community_info: List[CommunityInfo],
        options: CommunityVisualizationOptions
    ) -> Dict[int, tuple]:
        """Compute center positions for each community."""
        # Create community graph for layout
        num_communities = len(community_info)

        # Arrange communities in a circle
        import math
        centers = {}

        radius = 300 * options.community_spacing
        for i, comm_info in enumerate(community_info):
            angle = 2 * math.pi * i / num_communities
            x = 600 + radius * math.cos(angle)
            y = 400 + radius * math.sin(angle)
            centers[comm_info.id] = (x, y)

        return centers

    def _hierarchical_layout(
        self,
        graph: nx.Graph,
        communities: Dict[str, int],
        community_info: List[CommunityInfo],
        options: CommunityVisualizationOptions
    ) -> Dict[str, tuple]:
        """Generate hierarchical layout based on community importance."""
        layout = {}

        # Sort communities by importance (centrality)
        sorted_communities = sorted(community_info, key=lambda c: c.centrality, reverse=True)

        # Position communities in layers
        y = 100
        for level, communities_in_level in enumerate(['core', 'intermediate', 'peripheral']):
            relevant_communities = [c for c in sorted_communities if c.id in [ci.id for ci in community_info]]

            if not relevant_communities:
                continue

            # Distribute horizontally
            x_step = 1200 / (len(relevant_communities) + 1)
            for i, comm_info in enumerate(relevant_communities[:8]):  # Limit to 8 per level
                x = x_step * (i + 1)

                # Layout nodes within community
                subgraph = graph.subgraph(comm_info.nodes)
                sub_layout = nx.spring_layout(
                    subgraph,
                    center=(x, y),
                    scale=80.0
                )
                layout.update(sub_layout)

            y += 250

        return layout

    def _prepare_nodes(
        self,
        graph: nx.Graph,
        communities: Dict[str, int],
        layout: Dict[str, tuple],
        options: CommunityVisualizationOptions
    ) -> List[Dict[str, Any]]:
        """Prepare node data for visualization."""
        nodes = []

        # Compute centrality for sizing
        centrality = nx.degree_centrality(graph)

        for node in graph.nodes():
            comm_id = communities.get(node, 0)

            # Calculate node size
            if options.node_sizing == "centrality":
                size = 5 + centrality.get(node, 0) * 20
            else:
                size = 10

            x, y = layout.get(node, (0, 0))

            nodes.append({
                'id': node,
                'community': comm_id,
                'x': x,
                'y': y,
                'size': size,
                'centrality': centrality.get(node, 0),
                'degree': graph.degree(node)
            })

        return nodes

    def _prepare_edges(
        self,
        graph: nx.Graph,
        communities: Dict[str, int],
        options: CommunityVisualizationOptions
    ) -> List[Dict[str, Any]]:
        """Prepare edge data for visualization."""
        edges = []

        for source, target, data in graph.edges(data=True):
            source_comm = communities.get(source, -1)
            target_comm = communities.get(target, -1)

            edges.append({
                'source': source,
                'target': target,
                'predicate': data.get('predicate', 'related_to'),
                'confidence': data.get('confidence', 1.0),
                'is_inter_community': source_comm != target_comm,
                'source_community': source_comm,
                'target_community': target_comm
            })

        return edges

    async def _generate_community_html(
        self,
        viz_data: Dict[str, Any],
        options: CommunityVisualizationOptions
    ) -> str:
        """Generate community visualization HTML."""
        template_path = Path(__file__).parent / 'templates' / 'community_viz.html'

        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                html_template = f.read()
        else:
            html_template = self._get_embedded_template()

        viz_data_json = json.dumps(viz_data)

        # Use format to substitute the data
        html = html_template.replace('{viz_data}', viz_data_json)
        html = html.replace('{width}', str(options.width))
        html = html.replace('{height}', str(options.height))

        return html

    def _get_embedded_template(self) -> str:
        """Get embedded HTML template with full D3.js community visualization."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Community Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8f9fa; overflow: hidden; }
        #container { display: flex; flex-direction: column; height: 100vh; }
        #header { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 1rem 2rem; }
        #main-content { flex: 1; display: flex; position: relative; }
        #graph-container { flex: 1; position: relative; }
        #graph { width: 100%; height: 100%; cursor: grab; }
        #sidebar { width: 320px; background: white; border-left: 1px solid #ddd; overflow-y: auto; box-shadow: -2px 0 8px rgba(0,0,0,0.1); }
        #sidebar h2 { padding: 1rem; font-size: 1.1rem; color: #333; border-bottom: 1px solid #eee; }
        .community-info { padding: 1rem; border-bottom: 1px solid #eee; }
        .community-info h3 { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem; font-size: 1rem; }
        .community-color { width: 20px; height: 20px; border-radius: 50%; border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-size: 0.85rem; }
        .stat { background: #f8f9fa; padding: 0.5rem; border-radius: 4px; }
        .stat-label { color: #666; margin-bottom: 0.25rem; }
        .stat-value { color: #333; font-weight: 600; }
        .node { stroke: #fff; stroke-width: 1.5px; cursor: pointer; }
        .link { stroke-opacity: 0.6; }
        .link.inter-community { stroke-dasharray: 5,5; stroke: #999; }
    </style>
</head>
<body>
    <div id="container">
        <div id="header"><h1>Community Visualization</h1></div>
        <div id="main-content">
            <div id="graph-container">
                <svg id="graph"></svg>
            </div>
            <div id="sidebar">
                <h2>Communities</h2>
                <div id="communities-list"></div>
            </div>
        </div>
    </div>
    <script>
        const vizData = {viz_data};
        const width = {width};
        const height = {height};

        (function() {{
            const svg = d3.select('#graph').attr('width', width).attr('height', height);
            const interLinksGroup = svg.append('g').attr('class', 'inter-links');
            const linksGroup = svg.append('g').attr('class', 'links');
            const nodesGroup = svg.append('g').attr('class', 'nodes');

            const zoom = d3.zoom().scaleExtent([0.1, 4]).on('zoom', (event) => {{
                interLinksGroup.attr('transform', event.transform);
                linksGroup.attr('transform', event.transform);
                nodesGroup.attr('transform', event.transform);
            }});
            svg.call(zoom);

            const nodePositions = new Map();
            vizData.nodes.forEach(node => {{
                nodePositions.set(node.id, {{ x: node.x, y: node.y }});
            }});

            const interLinks = interLinksGroup.selectAll('line')
                .data(vizData.inter_community_edges)
                .enter().append('line')
                .attr('class', 'link inter-community')
                .attr('stroke', '#999')
                .attr('stroke-width', 1)
                .attr('stroke-dasharray', '5,5')
                .attr('x1', d => nodePositions.get(d.source)?.x || 0)
                .attr('y1', d => nodePositions.get(d.source)?.y || 0)
                .attr('x2', d => nodePositions.get(d.target)?.x || 0)
                .attr('y2', d => nodePositions.get(d.target)?.y || 0);

            const intraLinks = vizData.edges.filter(e => !e.is_inter_community);
            const links = linksGroup.selectAll('line')
                .data(intraLinks)
                .enter().append('line')
                .attr('class', 'link')
                .attr('stroke', '#666')
                .attr('stroke-width', d => d.confidence * 2)
                .attr('x1', d => nodePositions.get(d.source)?.x || 0)
                .attr('y1', d => nodePositions.get(d.source)?.y || 0)
                .attr('x2', d => nodePositions.get(d.target)?.x || 0)
                .attr('y2', d => nodePositions.get(d.target)?.y || 0);

            const node = nodesGroup.selectAll('circle')
                .data(vizData.nodes)
                .enter().append('circle')
                .attr('class', 'node')
                .attr('r', d => d.size)
                .attr('fill', d => d.color)
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
        }})();
    </script>
</body>
</html>"""
