"""
AI-Knowledge-Graph Visualization Integration

This module integrates ai-knowledge-graph's visualization capabilities,
generating D3.js interactive graph visualizations.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


class VisualizationOptions:
    """Options for graph visualization."""

    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        node_sizing: str = "centrality",  # centrality, degree, uniform
        edge_differentiation: bool = True,
        color_scheme: str = "colorblind",  # colorblind, default, spectral
        show_labels: bool = True,
        enable_zoom: bool = True,
        enable_physics: bool = True
    ):
        self.width = width
        self.height = height
        self.node_sizing = node_sizing
        self.edge_differentiation = edge_differentiation
        self.color_scheme = color_scheme
        self.show_labels = show_labels
        self.enable_zoom = enable_zoom
        self.enable_physics = enable_physics


class VisualizationResult:
    """Result of visualization generation."""

    def __init__(
        self,
        output_path: str,
        node_count: int,
        edge_count: int,
        community_count: int,
        statistics: Dict[str, Any]
    ):
        self.output_path = output_path
        self.node_count = node_count
        self.edge_count = edge_count
        self.community_count = community_count
        self.statistics = statistics
        self.timestamp = datetime.now().isoformat()


class AIKGVisualizer:
    """
    Generates D3.js interactive graph visualizations.

    Features:
    - Community detection (Louvain)
    - Centrality-based node sizing
    - Color-coded communities
    - Edge type differentiation
    - Interactive exploration
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

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visualizer.

        Args:
            config: Configuration dictionary with options:
                - output_dir: Directory for output files
                - community_algorithm: Algorithm for community detection
                - default_options: Default VisualizationOptions
        """
        self.output_dir = Path(config.get('output_dir', 'data/visualizations'))
        self.community_algorithm = config.get('community_algorithm', 'louvain')
        self.default_options = config.get('default_options', {})

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"AIKGVisualizer initialized with output_dir: {self.output_dir}")

    async def visualize_graph(
        self,
        triples: List,
        entities: List,
        output_path: str,
        options: Optional[VisualizationOptions] = None
    ) -> VisualizationResult:
        """
        Generate interactive HTML visualization.

        Features:
        - D3.js force-directed layout
        - Community detection and coloring
        - Node sizing by centrality
        - Edge styling (solid/dashed for original/inferred)
        - Interactive zoom/pan
        - Hover information

        Args:
            triples: List of triples
            entities: List of entities
            output_path: Path for output HTML file
            options: Visualization options

        Returns:
            VisualizationResult with metadata
        """
        if options is None:
            options = VisualizationOptions(**self.default_options)

        logger.info(f"Generating visualization for {len(triples)} triples, {len(entities)} entities")

        # Build NetworkX graph
        graph = self._build_graph(triples)

        # Detect communities
        communities = await self.detect_communities(graph)
        logger.info(f"Detected {len(set(communities.values()))} communities")

        # Compute centrality for node sizing
        centrality = await self.compute_centrality(graph)

        # Prepare graph data for D3.js
        graph_data = self._prepare_graph_data(
            graph, triples, communities, centrality, options
        )

        # Generate D3.js HTML
        html_content = await self.generate_d3_html(graph_data, options)

        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Visualization saved to: {output_file}")

        # Gather statistics
        statistics = {
            'communities': len(set(communities.values())),
            'avg_community_size': sum(1 for c in communities.values()) / len(set(communities.values()))
            if communities else 0,
            'max_centrality': max(centrality.values()) if centrality else 0,
            'graph_density': nx.density(graph),
            'is_connected': nx.is_connected(graph.to_undirected())
        }

        return VisualizationResult(
            output_path=str(output_file),
            node_count=len(graph.nodes()),
            edge_count=len(graph.edges()),
            community_count=len(set(communities.values())),
            statistics=statistics
        )

    def _build_graph(self, triples: List) -> nx.Graph:
        """Build NetworkX graph from triples."""
        graph = nx.Graph()

        for triple in triples:
            graph.add_node(triple.subject)
            graph.add_node(triple.object)
            graph.add_edge(
                triple.subject,
                triple.object,
                predicate=triple.predicate,
                confidence=getattr(triple, 'confidence', 1.0),
                source=getattr(triple, 'source', 'extracted')
            )

        return graph

    async def detect_communities(
        self,
        graph: nx.Graph
    ) -> Dict[str, int]:
        """
        Detect communities using Louvain algorithm.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary mapping node name to community ID
        """
        try:
            import networkx.algorithms.community as nx_comm

            # Use Louvain algorithm
            communities = nx_comm.louvain_communities(graph)

            # Build node -> community mapping
            node_to_community = {}
            for comm_id, community in enumerate(communities):
                for node in community:
                    node_to_community[node] = comm_id

            return node_to_community

        except ImportError:
            logger.warning("Louvain algorithm not available, using connected components")
            # Fallback to connected components
            communities = list(nx.connected_components(graph))
            node_to_community = {}
            for comm_id, community in enumerate(communities):
                for node in community:
                    node_to_community[node] = comm_id
            return node_to_community

    async def compute_centrality(
        self,
        graph: nx.Graph
    ) -> Dict[str, float]:
        """
        Compute centrality metrics for node sizing.

        Metrics:
        - Degree centrality (60% weight)
        - Betweenness centrality (30% weight)
        - Eigenvector centrality (10% weight)

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary mapping node name to centrality score
        """
        centrality_scores = {}

        # Degree centrality
        try:
            degree_cent = nx.degree_centrality(graph)
        except:
            degree_cent = {node: 0.0 for node in graph.nodes()}

        # Betweenness centrality (may be slow for large graphs)
        try:
            betweenness_cent = nx.betweenness_centrality(graph, normalized=True)
        except:
            betweenness_cent = {node: 0.0 for node in graph.nodes()}

        # Eigenvector centrality (may not converge for all graphs)
        try:
            eigenvector_cent = nx.eigenvector_centrality(graph, max_iter=100)
        except:
            eigenvector_cent = {node: 0.0 for node in graph.nodes()}

        # Combine with weighted average
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
        triples: List,
        communities: Dict[str, int],
        centrality: Dict[str, float],
        options: VisualizationOptions
    ) -> Dict[str, Any]:
        """Prepare graph data for D3.js visualization."""

        # Get color scheme
        colors = self.COLOR_SCHEMES.get(
            options.color_scheme,
            self.COLOR_SCHEMES["colorblind"]
        )

        # Prepare nodes
        nodes = []
        for node in graph.nodes():
            community_id = communities.get(node, 0)
            centrality_score = centrality.get(node, 0.0)

            # Calculate node size based on centrality
            if options.node_sizing == "centrality":
                node_size = 5 + centrality_score * 20
            elif options.node_sizing == "degree":
                node_size = 5 + graph.degree(node) * 2
            else:  # uniform
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
            # Determine edge style based on source
            if options.edge_differentiation and data.get('source') == 'inferred':
                edge_type = 'dashed'
            else:
                edge_type = 'solid'

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
                "enablePhysics": options.enable_physics
            }
        }

    async def generate_d3_html(
        self,
        graph_data: Dict[str, Any],
        options: VisualizationOptions
    ) -> str:
        """
        Generate D3.js HTML visualization.

        Components:
        - D3.js library (CDN)
        - Force simulation
        - Zoom/pan behavior
        - Node rendering with communities
        - Edge rendering with types
        - Hover tooltips
        - Legend and statistics
        """
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            overflow: hidden;
        }}
        #graph {{
            width: {width}px;
            height: {height}px;
            border: 1px solid #ccc;
            background: #fafafa;
        }}
        .node {{
            stroke: #fff;
            stroke-width: 1.5px;
            cursor: pointer;
        }}
        .node:hover {{
            stroke: #000;
            stroke-width: 2.5px;
        }}
        .link {{
            stroke-opacity: 0.6;
        }}
        .link.solid {{
            stroke-dasharray: none;
        }}
        .link.dashed {{
            stroke-dasharray: 5, 5;
        }}
        .label {{
            font-size: 10px;
            pointer-events: none;
            text-anchor: middle;
        }}
        #tooltip {{
            position: absolute;
            padding: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: #fff;
            border-radius: 5px;
            pointer-events: none;
            display: none;
            z-index: 1000;
        }}
        #legend {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.9);
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        #stats {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.9);
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div id="graph"></div>
    <div id="tooltip"></div>
    <div id="legend">
        <strong>Communities</strong><br/>
        <div id="legend-items"></div>
    </div>
    <div id="stats">
        <strong>Statistics</strong><br/>
        <div id="stats-content"></div>
    </div>

    <script>
        // Graph data
        const graphData = {graph_data_json};

        // Setup SVG
        const width = {width};
        const height = {height};

        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        // Zoom behavior
        {zoom_code}

        // Create container for zoom
        const container = svg.append("g");

        svg.call(zoom);

        // Force simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.edges).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => d.size + 5));

        // Create edges
        const link = container.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(graphData.edges)
            .enter().append("line")
            .attr("class", d => `link ${{d.type}}`)
            .attr("stroke", "#999")
            .attr("stroke-width", d => Math.sqrt(d.confidence) * 2);

        // Create nodes
        const node = container.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(graphData.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => d.size)
            .attr("fill", d => d.color)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        // Add labels (if enabled)
        {labels_code}

        // Tooltip
        const tooltip = d3.select("#tooltip");

        node.on("mouseover", function(event, d) {{
            tooltip.style("display", "block")
                .html(`<strong>${{d.id}}</strong><br/>
                       Community: ${{d.community}}<br/>
                       Centrality: ${{d.centrality.toFixed(3)}}<br/>
                       Degree: ${{d.degree}}`);
        }}).on("mousemove", function(event) {{
            tooltip.style("left", (event.pageX + 10) + "px")
                  .style("top", (event.pageY - 10) + "px");
        }}).on("mouseout", function() {{
            tooltip.style("display", "none");
        }});

        // Update positions on tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            {labels_update_code}
        }});

        // Drag functions
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

        // Legend
        const communities = [...new Set(graphData.nodes.map(d => d.community))];
        const legendItems = d3.select("#legend-items");
        communities.forEach(comm => {{
            const color = graphData.nodes.find(d => d.community === comm).color;
            legendItems.append("div")
                .style("display", "flex")
                .style("align-items", "center")
                .style("margin", "5px 0")
                .html(`<div style="width: 15px; height: 15px; background: ${{color}}; margin-right: 5px; border-radius: 50%;"></div>
                       Community ${{comm}}`);
        }});

        // Statistics
        const stats = d3.select("#stats-content");
        stats.html(`
            Nodes: ${{graphData.nodes.length}}<br/>
            Edges: ${{graphData.edges.length}}<br/>
            Communities: ${{communities.length}}<br/>
            Density: ${{(graphData.edges.length / (graphData.nodes.length * (graphData.nodes.length - 1) / 2)).toFixed(3)}}<br/>
            Avg Degree: ${{(graphData.edges.length * 2 / graphData.nodes.length).toFixed(2)}}
        `);
    </script>
</body>
</html>"""

        # Prepare graph data JSON
        graph_data_json = json.dumps(graph_data)

        # Generate zoom code
        zoom_code = """
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {
                container.attr("transform", event.transform);
            });
        """ if options.enable_zoom else "const zoom = null;"

        # Generate labels code
        if options.show_labels:
            labels_code = """
            const label = container.append("g")
                .attr("class", "labels")
                .selectAll("text")
                .data(graphData.nodes)
                .enter().append("text")
                .attr("class", "label")
                .text(d => d.id);
            """

            labels_update_code = """
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y + d.size + 12);
            """
        else:
            labels_code = "const label = null;"
            labels_update_code = ""

        # Fill template
        html = html_template.format(
            width=options.width,
            height=options.height,
            graph_data_json=graph_data_json,
            zoom_code=zoom_code,
            labels_code=labels_code,
            labels_update_code=labels_update_code
        )

        return html

    async def export_graph_data(
        self,
        triples: List,
        format: str = "json"
    ) -> str:
        """
        Export graph data for external visualization.

        Formats:
        - json: Structured JSON
        - gexf: Gephi format
        - graphml: NetworkX format
        - csv: Edge list

        Args:
            triples: List of triples
            format: Export format

        Returns:
            Exported data as string
        """
        graph = self._build_graph(triples)

        if format == "json":
            return json.dumps(nx.node_link_data(graph), indent=2)

        elif format == "gexf":
            # Note: NetworkX's gexf writer expects a file path
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.gexf', delete=False) as f:
                nx.write_gexf(graph, f.name)
                with open(f.name, 'r') as rf:
                    return rf.read()

        elif format == "graphml":
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
                nx.write_graphml(graph, f.name)
                with open(f.name, 'r') as rf:
                    return rf.read()

        elif format == "csv":
            # Export edge list
            import io
            output = io.StringIO()
            output.write("source,target,predicate,confidence,source\n")
            for source, target, data in graph.edges(data=True):
                output.write(f'"{source}","{target}",{data.get("predicate", "related_to")},'
                           f'{data.get("confidence", 1.0)},{data.get("source", "extracted")}\n')
            return output.getvalue()

        else:
            raise ValueError(f"Unsupported export format: {format}")
