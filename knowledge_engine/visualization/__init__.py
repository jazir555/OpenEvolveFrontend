"""
Visualization module for OpenEvolve Knowledge Engine.

This module provides visualization capabilities for knowledge graphs,
evolution results, and system metrics.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    width: int = 800
    height: int = 600
    theme: str = "light"
    interactive: bool = True
    export_format: str = "html"


@dataclass
class GraphLayout:
    """Layout configuration for graph visualization"""
    layout_type: str = "force_directed"  # force_directed, hierarchical, circular
    node_size: int = 20
    edge_width: int = 2
    show_labels: bool = True
    label_size: int = 10


@dataclass
class ChartData:
    """Data for chart visualization"""
    x_values: List[Any]
    y_values: List[Any]
    labels: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Export Handler
# ============================================================================

class ExportHandler:
    """
    Handler for exporting visualizations to various formats.

    Supports HTML, PNG, SVG, JSON formats.
    """

    def __init__(self, export_dir: Optional[str] = None):
        """
        Initialize export handler.

        Args:
            export_dir: Directory to save exported files
        """
        self.export_dir = Path(export_dir) if export_dir else Path("./visualizations")
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_html(
        self,
        fig: Any,
        filename: str,
        include_plotly: str = "cdn"
    ) -> str:
        """
        Export visualization to HTML.

        Args:
            fig: Plotly figure or compatible object
            filename: Output filename
            include_plotly: How to include plotly library

        Returns:
            Path to exported file
        """
        output_path = self.export_dir / f"{filename}.html"

        if PLOTLY_AVAILABLE and hasattr(fig, 'write_html'):
            fig.write_html(
                str(output_path),
                include_plotlyjs=include_plotly
            )
        else:
            # Fallback: create simple HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{filename}</title>
            </head>
            <body>
                <h1>{filename}</h1>
                <div id="visualization"></div>
                <p>Visualization data: {json.dumps({'type': 'placeholder'})}</p>
            </body>
            </html>
            """
            output_path.write_text(html_content)

        return str(output_path)

    def export_png(
        self,
        fig: Any,
        filename: str,
        scale: float = 1.0
    ) -> str:
        """
        Export visualization to PNG.

        Args:
            fig: Figure object
            filename: Output filename
            scale: Scale factor for resolution

        Returns:
            Path to exported file
        """
        output_path = self.export_dir / f"{filename}.png"

        if PLOTLY_AVAILABLE and hasattr(fig, 'write_image'):
            try:
                fig.write_image(str(output_path), scale=scale)
            except Exception as e:
                # kaleido package might not be installed
                pass
        elif MATPLOTLIB_AVAILABLE:
            fig.savefig(str(output_path), dpi=100 * scale, bbox_inches='tight')

        return str(output_path)

    def export_svg(
        self,
        fig: Any,
        filename: str
    ) -> str:
        """
        Export visualization to SVG.

        Args:
            fig: Figure object
            filename: Output filename

        Returns:
            Path to exported file
        """
        output_path = self.export_dir / f"{filename}.svg"

        if PLOTLY_AVAILABLE and hasattr(fig, 'write_image'):
            try:
                fig.write_image(str(output_path), format='svg')
            except Exception:
                pass
        elif MATPLOTLIB_AVAILABLE:
            fig.savefig(str(output_path), format='svg', bbox_inches='tight')

        return str(output_path)

    def export_json(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> str:
        """
        Export visualization data to JSON.

        Args:
            data: Data to export
            filename: Output filename

        Returns:
            Path to exported file
        """
        output_path = self.export_dir / f"{filename}.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return str(output_path)


# ============================================================================
# Knowledge Graph Visualizer
# ============================================================================

class KnowledgeGraphVisualizer:
    """
    Visualize knowledge graphs with various layouts.

    Supports entity-relationship graphs, concept maps, and
    knowledge flow visualizations.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        layout: Optional[GraphLayout] = None
    ):
        """
        Initialize knowledge graph visualizer.

        Args:
            config: Visualization configuration
            layout: Graph layout configuration
        """
        self.config = config or VisualizationConfig()
        self.layout = layout or GraphLayout()
        self.export_handler = ExportHandler()

    def visualize_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        layout_type: Optional[str] = None
    ) -> Any:
        """
        Create a graph visualization.

        Args:
            nodes: List of node dictionaries with at least 'id' field
            edges: List of edge dictionaries with 'source' and 'target' fields
            layout_type: Override default layout type

        Returns:
            Figure object (plotly or matplotlib)
        """
        layout_type = layout_type or self.layout.layout_type

        if PLOTLY_AVAILABLE:
            return self._create_plotly_graph(nodes, edges, layout_type)
        elif NETWORKX_AVAILABLE and MATPLOTLIB_AVAILABLE:
            return self._create_networkx_graph(nodes, edges, layout_type)
        else:
            # Fallback: return data structure
            return {
                "nodes": nodes,
                "edges": edges,
                "layout": layout_type
            }

    def _create_plotly_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        layout_type: str
    ) -> Any:
        """Create plotly graph visualization."""
        # Create networkx graph for layout
        if NETWORKX_AVAILABLE:
            G = nx.DiGraph()
            for node in nodes:
                G.add_node(node['id'], **node)
            for edge in edges:
                G.add_edge(edge['source'], edge['target'], **edge)

            # Calculate layout
            if layout_type == "force_directed":
                pos = nx.spring_layout(G)
            elif layout_type == "circular":
                pos = nx.circular_layout(G)
            elif layout_type == "hierarchical":
                pos = nx.multipartite_layout(G)
            else:
                pos = nx.spring_layout(G)

            # Extract node positions
            node_x = [pos[node['id']][0] for node in nodes]
            node_y = [pos[node['id']][1] for node in nodes]

            # Create edges
            edge_x = []
            edge_y = []
            for edge in edges:
                x0, y0 = pos[edge['source']]
                x1, y1 = pos[edge['target']]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            # Create plotly figure
            fig = go.Figure()

            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=self.layout.edge_width, color='#888'),
                hoverinfo='none',
                name='edges'
            ))

            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text' if self.layout.show_labels else 'markers',
                marker=dict(
                    size=self.layout.node_size,
                    color=[node.get('color', '#1f77b4') for node in nodes],
                    line=dict(width=1, color='white')
                ),
                text=[node.get('label', node['id']) for node in nodes] if self.layout.show_labels else None,
                textposition='middle center',
                textfont=dict(size=self.layout.label_size),
                hovertemplate='<b>%{text}</b><br>%{customdata}',
                customdata=[json.dumps(node, default=str) for node in nodes],
                name='nodes'
            ))

            fig.update_layout(
                title='Knowledge Graph',
                showlegend=False,
                width=self.config.width,
                height=self.config.height,
                template=self.config.theme
            )

            return fig
        else:
            # Simple scatter plot fallback
            node_ids = [node['id'] for node in nodes]
            return go.Figure(data=go.Scatter(
                x=list(range(len(nodes))),
                y=list(range(len(nodes))),
                text=node_ids,
                mode='markers+text'
            ))

    def _create_networkx_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        layout_type: str
    ) -> Any:
        """Create matplotlib graph using networkx."""
        G = nx.DiGraph()

        for node in nodes:
            G.add_node(node['id'], **node)

        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)

        # Calculate layout
        if layout_type == "force_directed":
            pos = nx.spring_layout(G)
        elif layout_type == "circular":
            pos = nx.circular_layout(G)
        elif layout_type == "hierarchical":
            pos = nx.multipartite_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Draw graph
        fig, ax = plt.subplots(figsize=(
            self.config.width / 100,
            self.config.height / 100
        ))

        nx.draw(
            G, pos,
            with_labels=self.layout.show_labels,
            node_size=self.layout.node_size * 10,
            width=self.layout.edge_width,
            ax=ax
        )

        ax.set_title('Knowledge Graph')
        return fig


# ============================================================================
# Metrics Visualizer
# ============================================================================

class MetricsVisualizer:
    """
    Visualize system metrics and performance data.

    Creates charts for execution time, success rates, resource usage,
    and other system metrics.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize metrics visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.export_handler = ExportHandler()

    def create_line_chart(
        self,
        data: ChartData,
        title: str = "Line Chart",
        x_label: str = "X",
        y_label: str = "Y"
    ) -> Any:
        """
        Create a line chart.

        Args:
            data: Chart data
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label

        Returns:
            Figure object
        """
        if PLOTLY_AVAILABLE:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=data.x_values,
                y=data.y_values,
                mode='lines+markers',
                name=title,
                line=dict(color=data.colors[0] if data.colors else '#1f77b4'),
                text=data.labels,
                hovertemplate='%{x}<br>%{y}: %{text}'
            ))

            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                width=self.config.width,
                height=self.config.height,
                template=self.config.theme
            )

            return fig
        else:
            return {"type": "line_chart", "data": data}

    def create_bar_chart(
        self,
        data: ChartData,
        title: str = "Bar Chart",
        x_label: str = "Categories",
        y_label: str = "Values"
    ) -> Any:
        """
        Create a bar chart.

        Args:
            data: Chart data
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label

        Returns:
            Figure object
        """
        if PLOTLY_AVAILABLE:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=data.x_values,
                y=data.y_values,
                name=title,
                marker_color=data.colors[0] if data.colors else '#1f77b4',
                text=data.labels,
                textposition='auto'
            ))

            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                width=self.config.width,
                height=self.config.height,
                template=self.config.theme
            )

            return fig
        else:
            return {"type": "bar_chart", "data": data}

    def create_pie_chart(
        self,
        data: ChartData,
        title: str = "Pie Chart"
    ) -> Any:
        """
        Create a pie chart.

        Args:
            data: Chart data
            title: Chart title

        Returns:
            Figure object
        """
        if PLOTLY_AVAILABLE:
            fig = go.Figure()

            fig.add_trace(go.Pie(
                labels=data.labels or data.x_values,
                values=data.y_values,
                name=title
            ))

            fig.update_layout(
                title=title,
                width=self.config.width,
                height=self.config.height,
                template=self.config.theme
            )

            return fig
        else:
            return {"type": "pie_chart", "data": data}


# ============================================================================
# Evolution Visualizer
# ============================================================================

class EvolutionVisualizer:
    """
    Visualize knowledge evolution and learning progress.

    Shows how knowledge artifacts, system performance, and
    learning metrics evolve over time.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize evolution visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.metrics_viz = MetricsVisualizer(config)

    def visualize_learning_progress(
        self,
        iterations: List[int],
        success_rates: List[float],
        confidence_scores: List[float]
    ) -> Any:
        """
        Visualize learning progress over iterations.

        Args:
            iterations: Iteration numbers
            success_rates: Success rate at each iteration
            confidence_scores: Confidence score at each iteration

        Returns:
            Figure object
        """
        if PLOTLY_AVAILABLE:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=iterations,
                y=success_rates,
                mode='lines+markers',
                name='Success Rate',
                line=dict(color='green')
            ))

            fig.add_trace(go.Scatter(
                x=iterations,
                y=confidence_scores,
                mode='lines+markers',
                name='Confidence Score',
                line=dict(color='blue')
            ))

            fig.update_layout(
                title='Learning Progress',
                xaxis_title='Iteration',
                yaxis_title='Score',
                width=self.config.width,
                height=self.config.height,
                template=self.config.theme
            )

            return fig
        else:
            return {
                "iterations": iterations,
                "success_rates": success_rates,
                "confidence_scores": confidence_scores
            }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_graph_visualization(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    output_file: Optional[str] = None
) -> Union[str, Dict[str, Any]]:
    """
    Convenience function to create and optionally export a graph visualization.

    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        output_file: Optional filename to export visualization

    Returns:
        Exported file path or data dictionary
    """
    visualizer = KnowledgeGraphVisualizer()
    fig = visualizer.visualize_graph(nodes, edges)

    if output_file:
        handler = ExportHandler()
        return handler.export_html(fig, output_file)

    return fig


# Export all components
__all__ = [
    'VisualizationConfig',
    'GraphLayout',
    'ChartData',
    'ExportHandler',
    'KnowledgeGraphVisualizer',
    'MetricsVisualizer',
    'EvolutionVisualizer',
    'create_graph_visualization'
]
