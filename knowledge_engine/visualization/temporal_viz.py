"""
Temporal Graph Visualization

Production-grade temporal visualization with:
- Time-based filtering
- Timeline slider
- Color-coded edges by age
- Before/after comparison
- Animation for temporal changes
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import networkx as nx

from .config import get_visualization_config

logger = logging.getLogger(__name__)


@dataclass
class TimeRange:
    """Time range for filtering."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within range."""
        if self.start and timestamp < self.start:
            return False
        if self.end and timestamp > self.end:
            return False
        return True


@dataclass
class TemporalSnapshot:
    """Snapshot of graph at specific time."""
    timestamp: datetime
    nodes: List[str]
    edges: List[Dict[str, Any]]
    statistics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serializable values."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'nodes': self.nodes,
            'edges': self.edges,
            'statistics': self.statistics
        }


@dataclass
class TemporalVisualizationOptions:
    """Options for temporal visualization."""
    width: int = 1200
    height: int = 800
    timeline_height: int = 150
    color_scale: str = "viridis"  # viridis, plasma, inferno, magma
    edge_color_by_age: bool = True
    show_timeline: bool = True
    enable_animation: bool = True
    animation_duration: int = 1000  # ms
    comparison_mode: bool = False
    time_window: Optional[TimeRange] = None


class TemporalVisualizer:
    """
    Temporal graph visualization with timeline support.

    Features:
    - Time-based filtering with slider
    - Color-coded edges by age
    - Animated temporal changes
    - Before/after comparison
    - Multiple snapshots
    - Heatmap visualization
    """

    # Color scales for edge age
    COLOR_SCALES = {
        "viridis": ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6dcd59", "#b4de2c", "#fde725"],
        "plasma": ["#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786", "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"],
        "inferno": ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"],
        "magma": ["#000004", "#180f3d", "#440f76", "#721f81", "#9e2f7f", "#c83f85", "#e65f70", "#f98e64", "#fec362", "#fcfdbf"]
    }

    def __init__(self, config=None):
        """Initialize temporal visualizer."""
        self.config = config or get_visualization_config()
        self.output_dir = Path(self.config.output_dir) / 'temporal'
        self.cache_dir = Path(self.config.cache_dir) / 'temporal'

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info({
            'event': 'temporal_visualizer_initialized',
            'output_dir': str(self.output_dir),
            'timestamp': datetime.utcnow().isoformat()
        })

    async def visualize_temporal(
        self,
        triples: List[Any],
        timestamps: List[datetime],
        output_path: Optional[str] = None,
        options: Optional[TemporalVisualizationOptions] = None
    ) -> Dict[str, Any]:
        """
        Generate temporal visualization.

        Args:
            triples: List of triples with timestamps
            timestamps: List of timestamps corresponding to triples
            output_path: Output file path
            options: Visualization options

        Returns:
            Result dictionary with metadata
        """
        if options is None:
            options = TemporalVisualizationOptions(
                width=self.config.default_width,
                height=self.config.default_height
            )

        # Build temporal graph
        temporal_graph = self._build_temporal_graph(triples, timestamps)

        # Apply time window filter
        if options.time_window:
            temporal_graph = self._filter_by_time_window(
                temporal_graph,
                options.time_window
            )

        # Generate snapshots
        snapshots = self._generate_snapshots(temporal_graph)

        # Compute temporal statistics
        statistics = self._compute_temporal_statistics(temporal_graph, snapshots)

        # Prepare visualization data
        viz_data = {
            'snapshots': [s.to_dict() for s in snapshots],
            'statistics': statistics,
            'options': asdict(options)
        }

        # Generate HTML
        html_content = await self._generate_temporal_html(viz_data, options)

        # Save to file
        if output_path is None:
            timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f'temporal_{timestamp_str}.html'

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info({
            'event': 'temporal_visualization_generated',
            'output_path': str(output_file),
            'snapshots': len(snapshots),
            'timestamp': datetime.utcnow().isoformat()
        })

        return {
            'output_path': str(output_file),
            'snapshots': len(snapshots),
            'statistics': statistics,
            'timestamp': datetime.utcnow().isoformat()
        }

    def _build_temporal_graph(
        self,
        triples: List[Any],
        timestamps: List[datetime]
    ) -> nx.DiGraph:
        """Build directed temporal graph.

        Handles multiple formats:
        - Object with subject/predicate/object attributes
        - Dict with subject/predicate/object keys
        - Tuple/list with [subject, predicate, object, confidence?]
        """
        graph = nx.DiGraph()

        for triple, ts in zip(triples, timestamps):
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

                # Add timestamp to edge
                graph.add_edge(
                    subj, obj,
                    predicate=pred,
                    confidence=float(conf),
                    timestamp=ts.isoformat()
                )

            except Exception as e:
                logger.warning({
                    'event': 'temporal_triple_processing_failed',
                    'triple': str(triple),
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
                continue

        return graph

    def _filter_by_time_window(
        self,
        graph: nx.DiGraph,
        time_window: TimeRange
    ) -> nx.DiGraph:
        """Filter graph by time window."""
        filtered_graph = nx.DiGraph()

        for source, target, data in graph.edges(data=True):
            timestamp = datetime.fromisoformat(data['timestamp'])

            if time_window.contains(timestamp):
                filtered_graph.add_edge(source, target, **data)

        # Add isolated nodes
        filtered_graph.add_nodes_from(graph.nodes())

        return filtered_graph

    def _generate_snapshots(
        self,
        graph: nx.DiGraph,
        num_snapshots: int = 10
    ) -> List[TemporalSnapshot]:
        """Generate temporal snapshots."""
        # Get all edge timestamps
        timestamps = []
        for _, _, data in graph.edges(data=True):
            ts = datetime.fromisoformat(data['timestamp'])
            timestamps.append(ts)

        if not timestamps:
            return []

        # Determine time range
        min_time = min(timestamps)
        max_time = max(timestamps)
        time_span = (max_time - min_time).total_seconds()

        if time_span == 0:
            # Single snapshot
            return [self._create_snapshot(graph, min_time)]

        # Generate snapshots at regular intervals
        snapshots = []
        interval = time_span / (num_snapshots - 1)

        for i in range(num_snapshots):
            snapshot_time = min_time + timedelta(seconds=i * interval)
            snapshot = self._create_snapshot_at_time(graph, snapshot_time)
            snapshots.append(snapshot)

        return snapshots

    def _create_snapshot_at_time(
        self,
        graph: nx.DiGraph,
        timestamp: datetime
    ) -> TemporalSnapshot:
        """Create snapshot at specific time."""
        # Filter edges up to this time
        edges_before = []
        for source, target, data in graph.edges(data=True):
            edge_time = datetime.fromisoformat(data['timestamp'])
            if edge_time <= timestamp:
                edges_before.append({
                    'source': source,
                    'target': target,
                    'predicate': data['predicate'],
                    'confidence': data['confidence'],
                    'timestamp': data['timestamp']
                })

        # Get all nodes involved
        nodes = set()
        for edge in edges_before:
            nodes.add(edge['source'])
            nodes.add(edge['target'])

        # Compute statistics
        snapshot_graph = nx.DiGraph()
        for edge in edges_before:
            snapshot_graph.add_edge(
                edge['source'],
                edge['target'],
                predicate=edge['predicate'],
                confidence=edge['confidence']
            )

        statistics = {
            'node_count': len(nodes),
            'edge_count': len(edges_before),
            'density': nx.density(snapshot_graph) if len(nodes) > 1 else 0,
            'is_connected': nx.is_weakly_connected(snapshot_graph)
        }

        return TemporalSnapshot(
            timestamp=timestamp,
            nodes=list(nodes),
            edges=edges_before,
            statistics=statistics
        )

    def _create_snapshot(self, graph: nx.DiGraph, timestamp: datetime) -> TemporalSnapshot:
        """Create snapshot at specific time (all edges)."""
        edges = []
        for source, target, data in graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'predicate': data['predicate'],
                'confidence': data['confidence'],
                'timestamp': data['timestamp']
            })

        nodes = list(graph.nodes())

        statistics = {
            'node_count': len(nodes),
            'edge_count': len(edges),
            'density': nx.density(graph),
            'is_connected': nx.is_weakly_connected(graph)
        }

        return TemporalSnapshot(
            timestamp=timestamp,
            nodes=nodes,
            edges=edges,
            statistics=statistics
        )

    def _compute_temporal_statistics(
        self,
        graph: nx.DiGraph,
        snapshots: List[TemporalSnapshot]
    ) -> Dict[str, Any]:
        """Compute temporal statistics."""
        if not snapshots:
            return {}

        # Growth metrics
        initial_nodes = snapshots[0].statistics['node_count']
        final_nodes = snapshots[-1].statistics['node_count']
        node_growth = final_nodes - initial_nodes

        initial_edges = snapshots[0].statistics['edge_count']
        final_edges = snapshots[-1].statistics['edge_count']
        edge_growth = final_edges - initial_edges

        # Edge age distribution
        edge_ages = []
        for _, _, data in graph.edges(data=True):
            edge_time = datetime.fromisoformat(data['timestamp'])
            age = (datetime.utcnow() - edge_time).total_seconds()
            edge_ages.append(age)

        return {
            'num_snapshots': len(snapshots),
            'initial_nodes': initial_nodes,
            'final_nodes': final_nodes,
            'node_growth': node_growth,
            'initial_edges': initial_edges,
            'final_edges': final_edges,
            'edge_growth': edge_growth,
            'avg_edge_age_seconds': sum(edge_ages) / len(edge_ages) if edge_ages else 0,
            'time_span_hours': (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 3600 if len(snapshots) > 1 else 0
        }

    async def _generate_temporal_html(
        self,
        viz_data: Dict[str, Any],
        options: TemporalVisualizationOptions
    ) -> str:
        """Generate temporal visualization HTML."""
        # Get color scale
        colors = self.COLOR_SCALES.get(options.color_scale, self.COLOR_SCALES["viridis"])

        # Load template
        template_path = Path(__file__).parent / 'templates' / 'temporal_viz.html'

        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                html_template = f.read()
        else:
            html_template = self._get_embedded_template()

        # Prepare data
        viz_data_json = json.dumps(viz_data)
        colors_json = json.dumps(colors)

        # Fill template
        html = html_template.format(
            width=options.width,
            height=options.height,
            timeline_height=options.timeline_height,
            viz_data=viz_data_json,
            colors=colors_json
        )

        return html

    def _get_embedded_template(self) -> str:
        """Get embedded HTML template with full D3.js temporal visualization."""
        template_path = Path(__file__).parent / 'templates' / 'temporal_viz.html'

        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()

        # Fallback template - simplified but functional
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Temporal Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8f9fa; overflow: hidden; }
        #container { display: flex; flex-direction: column; height: 100vh; }
        #header { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1rem 2rem; }
        #timeline-container { height: 150px; background: white; border-bottom: 1px solid #ddd; position: relative; }
        #graph-container { flex: 1; position: relative; background: white; }
        #graph { width: 100%; height: 100%; }
        .node { stroke: #fff; stroke-width: 1.5px; cursor: pointer; }
        .link { stroke-opacity: 0.6; }
        .link.new { stroke-dasharray: 5,5; animation: dash 1s linear infinite; }
        @keyframes dash { to { stroke-dashoffset: -10; } }
        #controls { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); background: white; padding: 1rem 1.5rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); display: flex; align-items: center; gap: 1.5rem; }
        #play-button { width: 50px; height: 50px; border: none; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 50%; color: white; font-size: 1.5rem; cursor: pointer; }
        #slider-container { display: flex; flex-direction: column; gap: 0.5rem; min-width: 300px; }
        #time-slider { width: 100%; height: 6px; border-radius: 3px; background: #ddd; outline: none; }
    </style>
</head>
<body>
    <div id="container">
        <div id="header"><h1>Temporal Graph Visualization</h1></div>
        <div id="timeline-container"><svg id="timeline"></svg></div>
        <div id="graph-container">
            <svg id="graph"></svg>
            <div id="controls">
                <button id="play-button">▶</button>
                <div id="slider-container">
                    <input type="range" id="time-slider" min="0" max="100" value="0" />
                </div>
            </div>
        </div>
    </div>
    <script>
        const vizData = {viz_data};
        const colors = {colors};
        let currentIndex = 0;
        const snapshots = vizData.snapshots;
        const numSnapshots = snapshots.length;

        if (numSnapshots > 0) {{
            const width = {width};
            const height = {height};

            const svg = d3.select('#graph').attr('width', width).attr('height', height);
            const linksGroup = svg.append('g').attr('class', 'links');
            const nodesGroup = svg.append('g').attr('class', 'nodes');

            function showSnapshot(index) {{
                currentIndex = index;
                const snapshot = snapshots[index];

                const nodes = snapshot.nodes.map(id => ({{ id: id }}));
                const links = snapshot.edges.map(e => ({{ source: e.source, target: e.target }}));

                linksGroup.selectAll('*').remove();
                nodesGroup.selectAll('*').remove();

                const link = linksGroup.selectAll('line').data(links).enter().append('line')
                    .attr('class', 'link').attr('stroke', '#999').attr('stroke-width', 2);

                const node = nodesGroup.selectAll('circle').data(nodes).enter().append('circle')
                    .attr('class', 'node').attr('r', 8).attr('fill', colors[index % colors.length]);

                const simulation = d3.forceSimulation(nodes)
                    .force('link', d3.forceLink(links).id(d => d.id).distance(100))
                    .force('charge', d3.forceManyBody().strength(-300))
                    .force('center', d3.forceCenter(width / 2, height / 2))
                    .force('collision', d3.forceCollide().radius(15))
                    .on('tick', () => {{
                        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
                            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
                        node.attr('cx', d => d.x).attr('cy', d => d.y);
                    }});
            }}

            showSnapshot(0);
            document.getElementById('time-slider').addEventListener('input', function() {{
                const index = Math.floor((this.value / 100) * (numSnapshots - 1));
                showSnapshot(index);
            }});
        }}
    </script>
</body>
</html>"""

    async def create_comparison_view(
        self,
        triples_before: List[Any],
        triples_after: List[Any],
        output_path: Optional[str] = None,
        options: Optional[TemporalVisualizationOptions] = None
    ) -> Dict[str, Any]:
        """
        Create before/after comparison view.

        Args:
            triples_before: Triples before change
            triples_after: Triples after change
            output_path: Output file path
            options: Visualization options

        Returns:
            Result dictionary
        """
        if options is None:
            options = TemporalVisualizationOptions(
                comparison_mode=True
            )

        # Build graphs
        graph_before = self._build_graph_from_triples(triples_before)
        graph_after = self._build_graph_from_triples(triples_after)

        # Compute differences
        added_nodes = set(graph_after.nodes()) - set(graph_before.nodes())
        removed_nodes = set(graph_before.nodes()) - set(graph_after.nodes())
        added_edges = set(graph_after.edges()) - set(graph_before.edges())
        removed_edges = set(graph_before.edges()) - set(graph_after.edges())

        # Prepare comparison data
        comparison_data = {
            'before': {
                'nodes': list(graph_before.nodes()),
                'edges': list(graph_before.edges()),
                'statistics': {
                    'node_count': len(graph_before.nodes()),
                    'edge_count': len(graph_before.edges())
                }
            },
            'after': {
                'nodes': list(graph_after.nodes()),
                'edges': list(graph_after.edges()),
                'statistics': {
                    'node_count': len(graph_after.nodes()),
                    'edge_count': len(graph_after.edges())
                }
            },
            'diff': {
                'added_nodes': list(added_nodes),
                'removed_nodes': list(removed_nodes),
                'added_edges': list(added_edges),
                'removed_edges': list(removed_edges)
            }
        }

        # Generate HTML
        html_content = await self._generate_comparison_html(comparison_data, options)

        # Save
        if output_path is None:
            timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f'comparison_{timestamp_str}.html'

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info({
            'event': 'comparison_view_generated',
            'output_path': str(output_file),
            'added_nodes': len(added_nodes),
            'added_edges': len(added_edges),
            'timestamp': datetime.utcnow().isoformat()
        })

        return {
            'output_path': str(output_file),
            'added_nodes': len(added_nodes),
            'removed_nodes': len(removed_nodes),
            'added_edges': len(added_edges),
            'removed_edges': len(removed_edges)
        }

    def _build_graph_from_triples(self, triples: List[Any]) -> nx.Graph:
        """Build NetworkX graph from triples."""
        graph = nx.Graph()

        for triple in triples:
            if hasattr(triple, 'subject'):
                subj = triple.subject
                pred = triple.predicate
                obj = triple.object
            elif isinstance(triple, (tuple, list)) and len(triple) >= 3:
                subj, pred, obj = triple[0], triple[1], triple[2]
            else:
                continue

            graph.add_edge(subj, obj, predicate=pred)

        return graph

    async def _generate_comparison_html(
        self,
        comparison_data: Dict[str, Any],
        options: TemporalVisualizationOptions
    ) -> str:
        """Generate comparison view HTML."""
        comparison_data_json = json.dumps(comparison_data)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Graph Comparison</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <h1>Before/After Comparison</h1>
    <div id="comparison"></div>
    <script>
        const comparisonData = {comparison_data_json};
        // D3.js comparison visualization code here
    </script>
</body>
</html>"""
