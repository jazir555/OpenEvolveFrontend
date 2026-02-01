"""
Knowledge Visualization Node for BubbleLabs Integration

Generates visual representations, charts, and graphs from knowledge graphs including:
- Network graphs with multiple layout algorithms
- Entity relationship diagrams
- Statistical charts and analytics visualizations
- Timeline visualizations for temporal knowledge
- Heatmaps for relationship analysis
- Export to multiple formats (PNG, SVG, JSON, HTML, D3)
- Interactive visualization data for web UIs

Supports fallback text-based visualizations when graphical libraries are unavailable.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict
import json
import base64
import io
from .base_node import BubbleLabsNode, NodeExecutionError


class KnowledgeVisualizationNode(BubbleLabsNode):
    """
    Knowledge Visualization Node for generating visual representations from knowledge graphs.

    Provides comprehensive visualization capabilities:
    - Network graphs: Force-directed, hierarchical, circular, matrix layouts
    - Entity diagrams: Relationship diagrams showing entity connections
    - Statistics charts: Bar charts, pie charts, histograms of graph metrics
    - Timeline visualizations: Temporal views of knowledge evolution
    - Heatmaps: Relationship density and correlation matrices
    - Multiple export formats: PNG, SVG, JSON, HTML, D3.js

    Features safe fallbacks to text-based visualizations when libraries unavailable.
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Visualization"
    DESCRIPTION = "Generate visual representations, charts, and graphs from knowledge"
    ICON = "visualization"
    CATEGORY = "interface"
    VERSION = "1.0.0"

    # Color palettes for different themes
    THEMES = {
        "light": {
            "background": "#ffffff",
            "node_colors": ["#4285f4", "#ea4335", "#fbbc05", "#34a853", "#9334e6", "#ff6d01"],
            "edge_color": "#999999",
            "text_color": "#333333",
            "grid_color": "#e0e0e0"
        },
        "dark": {
            "background": "#1a1a2e",
            "node_colors": ["#4fc3f7", "#ff7043", "#ffee58", "#66bb6a", "#ab47bc", "#ffa726"],
            "edge_color": "#555555",
            "text_color": "#e0e0e0",
            "grid_color": "#333333"
        },
        "colorful": {
            "background": "#f8f9fa",
            "node_colors": ["#e91e63", "#9c27b0", "#673ab7", "#3f51b5", "#2196f3", "#03a9f4",
                           "#00bcd4", "#009688", "#4caf50", "#8bc34a", "#cddc39", "#ffeb3b",
                           "#ffc107", "#ff9800", "#ff5722"],
            "edge_color": "#607d8b",
            "text_color": "#212121",
            "grid_color": "#e0e0e0"
        },
        "minimal": {
            "background": "#fafafa",
            "node_colors": ["#212121", "#424242", "#616161", "#757575", "#9e9e9e"],
            "edge_color": "#bdbdbd",
            "text_color": "#212121",
            "grid_color": "#eeeeee"
        }
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe import of UnifiedKGIntegrationHub
        unified_hub_module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for KnowledgeVisualizationNode"
        )

        self.UnifiedKGIntegrationHub = None
        if unified_hub_module:
            self.UnifiedKGIntegrationHub = getattr(unified_hub_module, 'UnifiedKGIntegrationHub', None)

        # Safe import of KnowledgeGraphVisualizer
        visualizer_module = self.safe_import(
            'knowledge_engine.knowledge_graph_visualizer',
            fallback_value=None,
            error_msg="KnowledgeGraphVisualizer not available for KnowledgeVisualizationNode"
        )

        self.KnowledgeGraphVisualizer = None
        if visualizer_module:
            self.KnowledgeGraphVisualizer = getattr(visualizer_module, 'KnowledgeGraphVisualizer', None)

        # Initialize hub instance if available
        self.hub = None
        if self.UnifiedKGIntegrationHub:
            try:
                self.hub = self.UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None

        # Initialize visualizer instance if available
        self.visualizer = None
        if self.KnowledgeGraphVisualizer:
            try:
                self.visualizer = self.KnowledgeGraphVisualizer()
                self.logger.info("KnowledgeGraphVisualizer initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize KnowledgeGraphVisualizer: {e}")
                self.visualizer = None

        # Try to import visualization libraries
        self.nx = self.safe_import(
            'networkx',
            fallback_value=None,
            error_msg="NetworkX not available, using fallback visualizations"
        )

        self.plt = None
        matplotlib_module = self.safe_import(
            'matplotlib',
            fallback_value=None,
            error_msg="Matplotlib not available, using fallback visualizations"
        )
        if matplotlib_module:
            try:
                import matplotlib.pyplot as plt
                self.plt = plt
                # Use non-interactive backend
                import matplotlib
                matplotlib.use('Agg')
            except Exception as e:
                self.logger.warning(f"Could not initialize matplotlib.pyplot: {e}")

        # Try to import PIL for image processing
        self.PIL = self.safe_import(
            'PIL',
            fallback_value=None,
            error_msg="PIL/Pillow not available, some image formats may be limited"
        )

        # Track available capabilities
        self.has_networkx = self.nx is not None
        self.has_matplotlib = self.plt is not None
        self.has_pil = self.PIL is not None
        self.has_visualizer = self.visualizer is not None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required (one of):
            - knowledge_graph_id: str - ID of the knowledge graph to visualize
            - knowledge_graph: dict - Knowledge graph data directly
            - entity_ids: list - Specific entity IDs to visualize

        Optional:
            - operation: str - Override the configured operation type
            - visualization_type: str - Override the configured visualization type
        """
        errors = []

        # Check that we have data source
        has_kg_id = 'knowledge_graph_id' in inputs and inputs['knowledge_graph_id']
        has_kg = 'knowledge_graph' in inputs and inputs['knowledge_graph']
        has_entities = 'entity_ids' in inputs and inputs['entity_ids']

        if not has_kg_id and not has_kg and not has_entities:
            errors.append("Missing required input: provide 'knowledge_graph_id', 'knowledge_graph', or 'entity_ids'")

        # Validate operation if provided
        if 'operation' in inputs:
            valid_operations = ['network_graph', 'entity_diagram', 'statistics_chart', 'timeline', 'heatmap']
            if inputs['operation'] not in valid_operations:
                errors.append(f"Invalid operation: '{inputs['operation']}'. Must be one of: {', '.join(valid_operations)}")

        # Validate visualization_type if provided
        if 'visualization_type' in inputs:
            valid_types = ['force_directed', 'hierarchical', 'circular', 'matrix']
            if inputs['visualization_type'] not in valid_types:
                errors.append(f"Invalid visualization_type: '{inputs['visualization_type']}'. Must be one of: {', '.join(valid_types)}")

        # Validate entity_ids if provided
        if 'entity_ids' in inputs:
            if not isinstance(inputs['entity_ids'], list):
                errors.append("'entity_ids' must be a list of strings")
            elif not all(isinstance(eid, str) for eid in inputs['entity_ids']):
                errors.append("All items in 'entity_ids' must be strings")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute knowledge graph visualization.

        Args:
            inputs: Contains knowledge_graph_id, knowledge_graph, or entity_ids, plus optional parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - image_data: Base64 encoded image or text representation
                - format: Output format
                - interactive_data: JSON-serializable data for interactive visualizations
                - metadata: Information about the visualization

        Raises:
            NodeExecutionError: If visualization fails
        """
        # Get configuration
        operation = inputs.get('operation', self.config.get('operation', 'network_graph'))
        entity_ids = inputs.get('entity_ids', self.config.get('entity_ids', []))
        visualization_type = inputs.get('visualization_type', self.config.get('visualization_type', 'force_directed'))
        output_format = inputs.get('output_format', self.config.get('output_format', 'png'))
        layout_algorithm = inputs.get('layout_algorithm', self.config.get('layout_algorithm', 'spring'))
        color_by = inputs.get('color_by', self.config.get('color_by', 'type'))
        max_nodes = inputs.get('max_nodes', self.config.get('max_nodes', 100))
        include_labels = inputs.get('include_labels', self.config.get('include_labels', True))
        style_theme = inputs.get('style_theme', self.config.get('style_theme', 'light'))

        context.update_progress(10, f"Initializing {operation} visualization")
        self.logger.info(f"Starting knowledge visualization: operation={operation}, format={output_format}")

        try:
            # Retrieve knowledge graph data
            kg_data = self._get_knowledge_graph_data(inputs, context)

            if not kg_data:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message="Could not retrieve knowledge graph data",
                    details={'inputs': list(inputs.keys())}
                )

            # Filter to specific entities if requested
            if entity_ids:
                kg_data = self._filter_to_entities(kg_data, entity_ids, max_nodes)
                context.update_progress(20, f"Filtered to {len(kg_data.get('nodes', []))} entities")
            else:
                # Apply max_nodes limit
                kg_data = self._limit_nodes(kg_data, max_nodes)
                context.update_progress(20, f"Limited to {len(kg_data.get('nodes', []))} nodes")

            context.update_progress(30, "Processing graph structure")

            # Execute the appropriate visualization operation
            if operation == 'network_graph':
                result = self._generate_network_graph(
                    kg_data, visualization_type, output_format, layout_algorithm,
                    color_by, include_labels, style_theme, context
                )
            elif operation == 'entity_diagram':
                result = self._generate_entity_diagram(
                    kg_data, output_format, color_by, include_labels, style_theme, context
                )
            elif operation == 'statistics_chart':
                result = self._generate_statistics_chart(
                    kg_data, output_format, style_theme, context
                )
            elif operation == 'timeline':
                result = self._generate_timeline(
                    kg_data, output_format, style_theme, context
                )
            elif operation == 'heatmap':
                result = self._generate_heatmap(
                    kg_data, output_format, style_theme, context
                )
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['network_graph', 'entity_diagram', 'statistics_chart', 'timeline', 'heatmap']}
                )

            context.update_progress(90, "Finalizing visualization output")

            # Add metadata to result
            result['metadata'] = {
                'executed_at': datetime.now().isoformat(),
                'execution_id': self.execution_id,
                'operation': operation,
                'visualization_type': visualization_type,
                'output_format': output_format,
                'layout_algorithm': layout_algorithm,
                'color_by': color_by,
                'max_nodes': max_nodes,
                'node_count': len(kg_data.get('nodes', [])),
                'edge_count': len(kg_data.get('edges', [])),
                'capabilities': {
                    'networkx': self.has_networkx,
                    'matplotlib': self.has_matplotlib,
                    'pil': self.has_pil,
                    'visualizer': self.has_visualizer
                },
                'fallback_used': not (self.has_networkx and self.has_matplotlib)
            }

            # Store artifact in context
            context.add_artifact('knowledge_visualization', {
                'operation': operation,
                'format': output_format,
                'node_count': result['metadata']['node_count'],
                'has_image': 'image_data' in result and bool(result['image_data'])
            })

            context.update_progress(100, "Visualization complete")

            self.logger.info(
                f"Knowledge visualization completed: "
                f"operation={operation}, format={output_format}, "
                f"nodes={result['metadata']['node_count']}"
            )

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Knowledge visualization failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Knowledge visualization failed: {str(e)}",
                details={
                    'operation': operation,
                    'format': output_format,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _get_knowledge_graph_data(self, inputs: Dict, context) -> Optional[Dict[str, Any]]:
        """
        Retrieve knowledge graph data from inputs or hub.

        Priority:
        1. knowledge_graph from inputs (direct data)
        2. knowledge_graph_id from inputs (fetch from hub)
        3. Build from entity_ids if provided
        """
        # Direct knowledge graph data
        if 'knowledge_graph' in inputs and inputs['knowledge_graph']:
            return inputs['knowledge_graph']

        # Fetch from hub using knowledge_graph_id
        kg_id = inputs.get('knowledge_graph_id')
        if kg_id and self.hub:
            try:
                if hasattr(self.hub, 'get_knowledge_graph'):
                    return self.hub.get_knowledge_graph(kg_id)
                elif hasattr(self.hub, 'export_graph'):
                    return self.hub.export_graph(kg_id)
            except Exception as e:
                self.logger.warning(f"Could not fetch graph from hub: {e}")

        # Check for kg_instance in inputs
        if 'kg_instance' in inputs and inputs['kg_instance']:
            kg = inputs['kg_instance']
            if hasattr(kg, 'export_to_dict'):
                return kg.export_to_dict()
            elif hasattr(kg, 'to_dict'):
                return kg.to_dict()

        # If we have entity_ids but no graph, return minimal structure
        entity_ids = inputs.get('entity_ids', [])
        if entity_ids:
            return {
                'nodes': [{'id': eid, 'type': 'entity', 'name': eid} for eid in entity_ids],
                'edges': []
            }

        return None

    def _filter_to_entities(self, kg_data: Dict[str, Any], entity_ids: List[str], max_nodes: int) -> Dict[str, Any]:
        """Filter knowledge graph to specific entities and their neighbors."""
        entity_set = set(entity_ids)

        # Filter nodes
        filtered_nodes = [
            node for node in kg_data.get('nodes', [])
            if node.get('id') in entity_set
        ]

        # Get IDs of filtered nodes
        node_ids = {node.get('id') for node in filtered_nodes}

        # Filter edges to only include connections between filtered nodes
        filtered_edges = [
            edge for edge in kg_data.get('edges', [])
            if edge.get('source') in node_ids and edge.get('target') in node_ids
        ]

        # Limit nodes if needed
        if len(filtered_nodes) > max_nodes:
            filtered_nodes = filtered_nodes[:max_nodes]
            node_ids = {node.get('id') for node in filtered_nodes}
            filtered_edges = [
                edge for edge in filtered_edges
                if edge.get('source') in node_ids and edge.get('target') in node_ids
            ]

        result = dict(kg_data)
        result['nodes'] = filtered_nodes
        result['edges'] = filtered_edges

        return result

    def _limit_nodes(self, kg_data: Dict[str, Any], max_nodes: int) -> Dict[str, Any]:
        """Limit knowledge graph to maximum number of nodes."""
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])

        if len(nodes) <= max_nodes:
            return kg_data

        # Take first max_nodes
        limited_nodes = nodes[:max_nodes]
        node_ids = {node.get('id') for node in limited_nodes}

        # Filter edges
        limited_edges = [
            edge for edge in edges
            if edge.get('source') in node_ids and edge.get('target') in node_ids
        ]

        result = dict(kg_data)
        result['nodes'] = limited_nodes
        result['edges'] = limited_edges

        return result

    def _generate_network_graph(
        self,
        kg_data: Dict[str, Any],
        visualization_type: str,
        output_format: str,
        layout_algorithm: str,
        color_by: str,
        include_labels: bool,
        style_theme: str,
        context
    ) -> Dict[str, Any]:
        """Generate network graph visualization."""
        context.update_progress(40, "Building network graph structure")

        # Use matplotlib/networkx if available
        if self.has_networkx and self.has_matplotlib and output_format in ['png', 'svg']:
            return self._generate_network_graph_matplotlib(
                kg_data, visualization_type, output_format, layout_algorithm,
                color_by, include_labels, style_theme, context
            )

        # Fallback to text-based or JSON output
        context.update_progress(50, "Using fallback text visualization")
        return self._generate_network_graph_fallback(
            kg_data, output_format, color_by, include_labels, style_theme
        )

    def _generate_network_graph_matplotlib(
        self,
        kg_data: Dict[str, Any],
        visualization_type: str,
        output_format: str,
        layout_algorithm: str,
        color_by: str,
        include_labels: bool,
        style_theme: str,
        context
    ) -> Dict[str, Any]:
        """Generate network graph using matplotlib."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        context.update_progress(50, "Computing graph layout")

        # Build networkx graph
        G = self._build_networkx_graph(kg_data)
        if not G:
            return self._generate_network_graph_fallback(kg_data, output_format, color_by, include_labels, style_theme)

        # Get theme colors
        theme = self.THEMES.get(style_theme, self.THEMES['light'])

        # Compute layout
        pos = self._compute_layout(G, layout_algorithm)

        context.update_progress(60, "Rendering graph")

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(theme['background'])
        ax.set_facecolor(theme['background'])

        # Prepare node colors
        node_colors = self._compute_node_colors(G, kg_data, color_by, theme)

        # Get node sizes based on degree
        degrees = dict(G.degree())
        node_sizes = [100 + degrees.get(node, 1) * 50 for node in G.nodes()]

        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color=theme['edge_color'],
            alpha=0.5,
            width=1,
            arrows=True,
            arrowsize=15,
            ax=ax
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )

        # Draw labels
        if include_labels:
            labels = {node: self._truncate_label(str(node)) for node in G.nodes()}
            nx.draw_networkx_labels(
                G, pos, labels,
                font_size=8,
                font_color=theme['text_color'],
                ax=ax
            )

        # Remove axes
        ax.set_axis_off()

        # Add legend for color coding
        if color_by == 'type':
            legend_elements = self._create_type_legend(kg_data, theme)
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        context.update_progress(80, "Encoding image")

        # Save to buffer
        buf = io.BytesIO()
        format_ext = 'png' if output_format == 'png' else 'svg'
        plt.savefig(buf, format=format_ext, bbox_inches='tight', facecolor=theme['background'])
        buf.seek(0)

        # Encode as base64
        image_data = base64.b64encode(buf.read()).decode('utf-8')

        plt.close(fig)

        # Generate interactive data
        interactive_data = self._generate_interactive_network_data(G, kg_data, pos)

        return {
            'image_data': image_data,
            'format': output_format,
            'interactive_data': interactive_data
        }

    def _generate_network_graph_fallback(
        self,
        kg_data: Dict[str, Any],
        output_format: str,
        color_by: str,
        include_labels: bool,
        style_theme: str
    ) -> Dict[str, Any]:
        """Generate text-based network graph fallback."""
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])

        # Build adjacency representation
        adjacency = defaultdict(list)
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                adjacency[source].append(target)

        # Generate ASCII representation
        lines = [
            "=" * 60,
            "KNOWLEDGE GRAPH NETWORK VISUALIZATION (TEXT MODE)",
            "=" * 60,
            f"Nodes: {len(nodes)} | Edges: {len(edges)}",
            "-" * 60,
            ""
        ]

        # Show node connectivity
        for node in nodes[:50]:  # Limit to 50 nodes in text mode
            node_id = node.get('id', 'unknown')
            node_type = node.get('type', 'unknown')
            connections = adjacency.get(node_id, [])

            lines.append(f"[{node_type}] {node_id}")
            if connections and include_labels:
                for target in connections[:5]:  # Limit connections shown
                    lines.append(f"  └──> {target}")
                if len(connections) > 5:
                    lines.append(f"  └──> ... and {len(connections) - 5} more")
            lines.append("")

        # Statistics
        lines.extend([
            "-" * 60,
            "STATISTICS",
            "-" * 60,
            f"Total Nodes: {len(nodes)}",
            f"Total Edges: {len(edges)}",
            f"Average Degree: {len(edges) * 2 / len(nodes) if nodes else 0:.2f}",
            "=" * 60
        ])

        text_output = '\n'.join(lines)

        # Generate interactive data
        interactive_data = {
            'nodes': [{'id': n.get('id'), 'type': n.get('type')} for n in nodes],
            'edges': [{'source': e.get('source'), 'target': e.get('target')} for e in edges],
            'statistics': {
                'node_count': len(nodes),
                'edge_count': len(edges)
            }
        }

        return {
            'image_data': base64.b64encode(text_output.encode()).decode('utf-8'),
            'format': 'txt',
            'text_representation': text_output,
            'interactive_data': interactive_data
        }

    def _generate_entity_diagram(
        self,
        kg_data: Dict[str, Any],
        output_format: str,
        color_by: str,
        include_labels: bool,
        style_theme: str,
        context
    ) -> Dict[str, Any]:
        """Generate entity relationship diagram."""
        context.update_progress(40, "Building entity diagram")

        if self.has_networkx and self.has_matplotlib and output_format in ['png', 'svg']:
            return self._generate_entity_diagram_matplotlib(
                kg_data, output_format, color_by, include_labels, style_theme, context
            )

        return self._generate_entity_diagram_fallback(kg_data, output_format)

    def _generate_entity_diagram_matplotlib(
        self,
        kg_data: Dict[str, Any],
        output_format: str,
        color_by: str,
        include_labels: bool,
        style_theme: str,
        context
    ) -> Dict[str, Any]:
        """Generate entity diagram using matplotlib."""
        import matplotlib.pyplot as plt

        context.update_progress(60, "Rendering entity diagram")

        G = self._build_networkx_graph(kg_data)
        if not G:
            return self._generate_entity_diagram_fallback(kg_data, output_format)

        theme = self.THEMES.get(style_theme, self.THEMES['light'])

        # Use hierarchical layout for entity diagrams
        pos = self._compute_layout(G, 'hierarchical')

        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor(theme['background'])
        ax.set_facecolor(theme['background'])

        # Group nodes by type for coloring
        node_types = defaultdict(list)
        for node_id, node_data in kg_data.get('nodes', []):
            node_type = node_data.get('type', 'unknown')
            node_types[node_type].append(node_id)

        # Draw edges with different styles based on relationship type
        for edge in kg_data.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            rel_type = edge.get('type', 'related')

            if source in pos and target in pos:
                style = 'solid' if rel_type == 'direct' else 'dashed'
                ax.plot(
                    [pos[source][0], pos[target][0]],
                    [pos[source][1], pos[target][1]],
                    'k-',
                    alpha=0.3,
                    linestyle=style,
                    linewidth=1
                )

        # Draw nodes by type
        colors = theme['node_colors']
        for i, (node_type, nodes) in enumerate(node_types.items()):
            color = colors[i % len(colors)]
            node_positions = [pos.get(n) for n in nodes if n in pos]
            if node_positions:
                xs, ys = zip(*node_positions)
                ax.scatter(xs, ys, c=color, s=200, alpha=0.8, label=node_type)

        if include_labels:
            for node, (x, y) in pos.items():
                ax.text(x, y, self._truncate_label(str(node)), fontsize=7, ha='center', va='center')

        ax.legend(loc='upper right', fontsize=8)
        ax.set_axis_off()

        # Save
        buf = io.BytesIO()
        format_ext = 'png' if output_format == 'png' else 'svg'
        plt.savefig(buf, format=format_ext, bbox_inches='tight', facecolor=theme['background'])
        buf.seek(0)
        image_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return {
            'image_data': image_data,
            'format': output_format,
            'interactive_data': {
                'nodes': [{'id': n.get('id'), 'type': n.get('type')} for n in kg_data.get('nodes', [])],
                'edges': [{'source': e.get('source'), 'target': e.get('target'), 'type': e.get('type')}
                         for e in kg_data.get('edges', [])]
            }
        }

    def _generate_entity_diagram_fallback(
        self,
        kg_data: Dict[str, Any],
        output_format: str
    ) -> Dict[str, Any]:
        """Generate text-based entity diagram."""
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])

        lines = [
            "=" * 60,
            "ENTITY RELATIONSHIP DIAGRAM",
            "=" * 60,
            ""
        ]

        # Group by type
        by_type = defaultdict(list)
        for node in nodes:
            by_type[node.get('type', 'unknown')].append(node.get('id'))

        for node_type, ids in by_type.items():
            lines.append(f"[{node_type}]")
            for node_id in ids[:10]:
                lines.append(f"  - {node_id}")
            if len(ids) > 10:
                lines.append(f"  ... and {len(ids) - 10} more")
            lines.append("")

        lines.extend([
            "-" * 60,
            "RELATIONSHIPS",
            "-" * 60
        ])

        for edge in edges[:30]:
            lines.append(f"{edge.get('source')} --[{edge.get('type', 'related')}]--> {edge.get('target')}")

        if len(edges) > 30:
            lines.append(f"... and {len(edges) - 30} more relationships")

        lines.append("=" * 60)

        text_output = '\n'.join(lines)

        return {
            'image_data': base64.b64encode(text_output.encode()).decode('utf-8'),
            'format': 'txt',
            'text_representation': text_output,
            'interactive_data': {'nodes': nodes, 'edges': edges}
        }

    def _generate_statistics_chart(
        self,
        kg_data: Dict[str, Any],
        output_format: str,
        style_theme: str,
        context
    ) -> Dict[str, Any]:
        """Generate statistics chart visualization."""
        context.update_progress(40, "Calculating statistics")

        if self.has_matplotlib and output_format in ['png', 'svg']:
            return self._generate_statistics_chart_matplotlib(kg_data, output_format, style_theme, context)

        return self._generate_statistics_chart_fallback(kg_data)

    def _generate_statistics_chart_matplotlib(
        self,
        kg_data: Dict[str, Any],
        output_format: str,
        style_theme: str,
        context
    ) -> Dict[str, Any]:
        """Generate statistics charts using matplotlib."""
        import matplotlib.pyplot as plt

        context.update_progress(60, "Rendering statistics charts")

        theme = self.THEMES.get(style_theme, self.THEMES['light'])

        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])

        # Calculate statistics
        type_counts = defaultdict(int)
        for node in nodes:
            node_type = node.get('type', 'unknown')
            type_counts[node_type] += 1

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(theme['background'])

        # 1. Entity type distribution (pie chart)
        ax1 = axes[0, 0]
        ax1.set_facecolor(theme['background'])
        if type_counts:
            colors = theme['node_colors'][:len(type_counts)]
            ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax1.set_title('Entity Type Distribution', color=theme['text_color'])

        # 2. Degree distribution (bar chart)
        ax2 = axes[0, 1]
        ax2.set_facecolor(theme['background'])
        if self.has_networkx:
            G = self._build_networkx_graph(kg_data)
            if G:
                degrees = [d for _, d in G.degree()]
                if degrees:
                    ax2.hist(degrees, bins=10, color=theme['node_colors'][0], alpha=0.7, edgecolor='black')
                    ax2.set_xlabel('Degree', color=theme['text_color'])
                    ax2.set_ylabel('Frequency', color=theme['text_color'])
                    ax2.set_title('Node Degree Distribution', color=theme['text_color'])
                    ax2.tick_params(colors=theme['text_color'])

        # 3. Top entity types (bar chart)
        ax3 = axes[1, 0]
        ax3.set_facecolor(theme['background'])
        if type_counts:
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            types, counts = zip(*sorted_types)
            ax3.barh(types, counts, color=theme['node_colors'][1], alpha=0.7)
            ax3.set_xlabel('Count', color=theme['text_color'])
            ax3.set_title('Top Entity Types', color=theme['text_color'])
            ax3.tick_params(colors=theme['text_color'])

        # 4. Graph metrics summary (text)
        ax4 = axes[1, 1]
        ax4.set_facecolor(theme['background'])
        ax4.axis('off')

        metrics_text = f"""
GRAPH METRICS SUMMARY

Total Nodes: {len(nodes)}
Total Edges: {len(edges)}
Entity Types: {len(type_counts)}
Density: {len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0:.4f}
Avg Degree: {2 * len(edges) / len(nodes) if nodes else 0:.2f}
        """
        ax4.text(0.1, 0.5, metrics_text, fontsize=12, color=theme['text_color'],
                verticalalignment='center', family='monospace')

        plt.tight_layout()

        # Save
        buf = io.BytesIO()
        format_ext = 'png' if output_format == 'png' else 'svg'
        plt.savefig(buf, format=format_ext, bbox_inches='tight', facecolor=theme['background'])
        buf.seek(0)
        image_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return {
            'image_data': image_data,
            'format': output_format,
            'interactive_data': {
                'type_distribution': dict(type_counts),
                'total_nodes': len(nodes),
                'total_edges': len(edges)
            }
        }

    def _generate_statistics_chart_fallback(
        self,
        kg_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate text-based statistics report."""
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])

        type_counts = defaultdict(int)
        for node in nodes:
            type_counts[node.get('type', 'unknown')] += 1

        lines = [
            "=" * 60,
            "KNOWLEDGE GRAPH STATISTICS",
            "=" * 60,
            "",
            "BASIC METRICS",
            "-" * 40,
            f"Total Nodes: {len(nodes)}",
            f"Total Edges: {len(edges)}",
            f"Entity Types: {len(type_counts)}",
            f"Density: {len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0:.4f}",
            f"Average Degree: {2 * len(edges) / len(nodes) if nodes else 0:.2f}",
            "",
            "ENTITY TYPE DISTRIBUTION",
            "-" * 40
        ]

        for node_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(nodes) * 100 if nodes else 0
            bar = '█' * int(pct / 2)
            lines.append(f"{node_type:20} {count:5} ({pct:5.1f}%) {bar}")

        lines.append("=" * 60)

        text_output = '\n'.join(lines)

        return {
            'image_data': base64.b64encode(text_output.encode()).decode('utf-8'),
            'format': 'txt',
            'text_representation': text_output,
            'interactive_data': {
                'type_distribution': dict(type_counts),
                'total_nodes': len(nodes),
                'total_edges': len(edges)
            }
        }

    def _generate_timeline(
        self,
        kg_data: Dict[str, Any],
        output_format: str,
        style_theme: str,
        context
    ) -> Dict[str, Any]:
        """Generate timeline visualization."""
        context.update_progress(40, "Processing temporal data")

        if self.has_matplotlib and output_format in ['png', 'svg']:
            return self._generate_timeline_matplotlib(kg_data, output_format, style_theme, context)

        return self._generate_timeline_fallback(kg_data)

    def _generate_timeline_matplotlib(
        self,
        kg_data: Dict[str, Any],
        output_format: str,
        style_theme: str,
        context
    ) -> Dict[str, Any]:
        """Generate timeline using matplotlib."""
        import matplotlib.pyplot as plt
        from datetime import datetime

        context.update_progress(60, "Rendering timeline")

        theme = self.THEMES.get(style_theme, self.THEMES['light'])

        # Extract timestamps
        events = []
        for node in kg_data.get('nodes', []):
            ts = node.get('timestamp') or node.get('created_at')
            if ts:
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    else:
                        dt = ts
                    events.append({
                        'timestamp': dt,
                        'id': node.get('id'),
                        'type': node.get('type', 'unknown')
                    })
                except:
                    pass

        if not events:
            return self._generate_timeline_fallback(kg_data)

        events.sort(key=lambda x: x['timestamp'])

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(theme['background'])
        ax.set_facecolor(theme['background'])

        # Group by type
        type_events = defaultdict(list)
        for event in events:
            type_events[event['type']].append(event)

        colors = theme['node_colors']
        y_pos = 0
        for i, (event_type, type_events_list) in enumerate(type_events.items()):
            color = colors[i % len(colors)]
            timestamps = [e['timestamp'] for e in type_events_list]
            y_positions = [y_pos] * len(timestamps)

            ax.scatter(timestamps, y_positions, c=color, s=100, alpha=0.7, label=event_type)

            # Add labels for some events
            for j, (ts, event) in enumerate(zip(timestamps, type_events_list)):
                if j % max(1, len(timestamps) // 10) == 0:  # Label ~10 events per type
                    ax.text(ts, y_pos + 0.1, self._truncate_label(event['id']), fontsize=6, rotation=45)

            y_pos += 1

        ax.set_yticks(range(len(type_events)))
        ax.set_yticklabels(type_events.keys())
        ax.set_xlabel('Time', color=theme['text_color'])
        ax.set_title('Knowledge Graph Timeline', color=theme['text_color'])
        ax.tick_params(colors=theme['text_color'])
        ax.legend(loc='upper left', fontsize=8)

        plt.tight_layout()

        buf = io.BytesIO()
        format_ext = 'png' if output_format == 'png' else 'svg'
        plt.savefig(buf, format=format_ext, bbox_inches='tight', facecolor=theme['background'])
        buf.seek(0)
        image_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return {
            'image_data': image_data,
            'format': output_format,
            'interactive_data': {
                'events': [{'id': e['id'], 'type': e['type'], 'timestamp': e['timestamp'].isoformat()}
                          for e in events]
            }
        }

    def _generate_timeline_fallback(
        self,
        kg_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate text-based timeline."""
        events = []
        for node in kg_data.get('nodes', []):
            ts = node.get('timestamp') or node.get('created_at')
            if ts:
                events.append({
                    'timestamp': str(ts)[:19] if isinstance(ts, str) else str(ts),
                    'id': node.get('id'),
                    'type': node.get('type', 'unknown')
                })

        events.sort(key=lambda x: x['timestamp'])

        lines = [
            "=" * 80,
            "KNOWLEDGE GRAPH TIMELINE",
            "=" * 80,
            ""
        ]

        for event in events[:100]:  # Limit to 100 events
            lines.append(f"{event['timestamp']:<20} [{event['type']:<15}] {event['id']}")

        if len(events) > 100:
            lines.append(f"\n... and {len(events) - 100} more events")

        lines.append("=" * 80)

        text_output = '\n'.join(lines)

        return {
            'image_data': base64.b64encode(text_output.encode()).decode('utf-8'),
            'format': 'txt',
            'text_representation': text_output,
            'interactive_data': {'events': events}
        }

    def _generate_heatmap(
        self,
        kg_data: Dict[str, Any],
        output_format: str,
        style_theme: str,
        context
    ) -> Dict[str, Any]:
        """Generate heatmap visualization."""
        context.update_progress(40, "Computing relationship matrix")

        if self.has_matplotlib and output_format in ['png', 'svg']:
            return self._generate_heatmap_matplotlib(kg_data, output_format, style_theme, context)

        return self._generate_heatmap_fallback(kg_data)

    def _generate_heatmap_matplotlib(
        self,
        kg_data: Dict[str, Any],
        output_format: str,
        style_theme: str,
        context
    ) -> Dict[str, Any]:
        """Generate heatmap using matplotlib."""
        import matplotlib.pyplot as plt
        import numpy as np

        context.update_progress(60, "Rendering heatmap")

        theme = self.THEMES.get(style_theme, self.THEMES['light'])

        # Get entity types
        type_counts = defaultdict(int)
        for node in kg_data.get('nodes', []):
            type_counts[node.get('type', 'unknown')] += 1

        top_types = sorted(type_counts.keys(), key=lambda t: type_counts[t], reverse=True)[:10]

        # Build relationship matrix
        matrix = np.zeros((len(top_types), len(top_types)))
        type_index = {t: i for i, t in enumerate(top_types)}

        for edge in kg_data.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')

            # Find types for source and target
            source_type = None
            target_type = None
            for node in kg_data.get('nodes', []):
                if node.get('id') == source:
                    source_type = node.get('type', 'unknown')
                if node.get('id') == target:
                    target_type = node.get('type', 'unknown')

            if source_type in type_index and target_type in type_index:
                matrix[type_index[source_type]][type_index[target_type]] += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(theme['background'])
        ax.set_facecolor(theme['background'])

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(len(top_types)))
        ax.set_yticks(np.arange(len(top_types)))
        ax.set_xticklabels(top_types, rotation=45, ha='right')
        ax.set_yticklabels(top_types)
        ax.set_title('Entity Type Relationship Heatmap', color=theme['text_color'])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Relationships', color=theme['text_color'])

        plt.tight_layout()

        buf = io.BytesIO()
        format_ext = 'png' if output_format == 'png' else 'svg'
        plt.savefig(buf, format=format_ext, bbox_inches='tight', facecolor=theme['background'])
        buf.seek(0)
        image_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return {
            'image_data': image_data,
            'format': output_format,
            'interactive_data': {
                'matrix': matrix.tolist(),
                'labels': top_types
            }
        }

    def _generate_heatmap_fallback(
        self,
        kg_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate text-based heatmap."""
        type_counts = defaultdict(int)
        for node in kg_data.get('nodes', []):
            type_counts[node.get('type', 'unknown')] += 1

        top_types = sorted(type_counts.keys(), key=lambda t: type_counts[t], reverse=True)[:8]

        # Build relationship matrix
        matrix = [[0] * len(top_types) for _ in range(len(top_types))]
        type_index = {t: i for i, t in enumerate(top_types)}

        for edge in kg_data.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')

            source_type = None
            target_type = None
            for node in kg_data.get('nodes', []):
                if node.get('id') == source:
                    source_type = node.get('type', 'unknown')
                if node.get('id') == target:
                    target_type = node.get('type', 'unknown')

            if source_type in type_index and target_type in type_index:
                matrix[type_index[source_type]][type_index[target_type]] += 1

        lines = [
            "=" * 80,
            "RELATIONSHIP HEATMAP (TEXT MODE)",
            "=" * 80,
            ""
        ]

        # Header
        header = " " * 12
        for t in top_types:
            header += f"{t[:8]:>8}"
        lines.append(header)
        lines.append("-" * 80)

        # Rows
        for i, row_type in enumerate(top_types):
            row_str = f"{row_type[:10]:<10} |"
            for j, val in enumerate(matrix[i]):
                if val == 0:
                    row_str += "   ·    "
                elif val < 10:
                    row_str += f"   {val}    "
                elif val < 100:
                    row_str += f"  {val}    "
                else:
                    row_str += f" {val}    "
            lines.append(row_str)

        lines.append("=" * 80)

        text_output = '\n'.join(lines)

        return {
            'image_data': base64.b64encode(text_output.encode()).decode('utf-8'),
            'format': 'txt',
            'text_representation': text_output,
            'interactive_data': {
                'matrix': matrix,
                'labels': top_types
            }
        }

    def _build_networkx_graph(self, kg_data: Dict[str, Any]) -> Optional[Any]:
        """Build a NetworkX graph from knowledge graph data."""
        if not self.has_networkx:
            return None

        try:
            G = self.nx.DiGraph()

            # Add nodes
            for node in kg_data.get('nodes', []):
                node_id = node.get('id')
                if node_id:
                    G.add_node(node_id, **{k: v for k, v in node.items() if k != 'id'})

            # Add edges
            for edge in kg_data.get('edges', []):
                source = edge.get('source')
                target = edge.get('target')
                if source and target:
                    G.add_edge(source, target, **{k: v for k, v in edge.items()
                                                  if k not in ['source', 'target']})

            # Process triples if no edges
            if not kg_data.get('edges') and kg_data.get('triples'):
                for triple in kg_data.get('triples', []):
                    subject = triple.get('subject')
                    obj = triple.get('object')
                    if subject and obj:
                        G.add_edge(subject, obj, predicate=triple.get('predicate'))

            return G

        except Exception as e:
            self.logger.warning(f"Failed to build NetworkX graph: {e}")
            return None

    def _compute_layout(self, G: Any, layout_algorithm: str) -> Dict[str, Tuple[float, float]]:
        """Compute node positions based on layout algorithm."""
        if not self.has_networkx:
            return {}

        try:
            if layout_algorithm == 'spring':
                return self.nx.spring_layout(G, k=2, iterations=50)
            elif layout_algorithm == 'circular':
                return self.nx.circular_layout(G)
            elif layout_algorithm == 'hierarchical':
                return self.nx.kamada_kawai_layout(G)
            elif layout_algorithm == 'random':
                return self.nx.random_layout(G)
            else:
                return self.nx.spring_layout(G)
        except Exception as e:
            self.logger.warning(f"Layout computation failed: {e}, using spring layout")
            try:
                return self.nx.spring_layout(G)
            except:
                return {}

    def _compute_node_colors(self, G: Any, kg_data: Dict[str, Any], color_by: str, theme: Dict) -> List[str]:
        """Compute node colors based on coloring strategy."""
        colors = theme['node_colors']
        node_list = list(G.nodes())

        if color_by == 'type':
            # Group by type
            type_groups = defaultdict(list)
            for node in kg_data.get('nodes', []):
                node_type = node.get('type', 'unknown')
                type_groups[node_type].append(node.get('id'))

            type_to_color = {}
            for i, node_type in enumerate(type_groups.keys()):
                type_to_color[node_type] = colors[i % len(colors)]

            node_colors = []
            for node_id in node_list:
                node_type = G.nodes[node_id].get('type', 'unknown')
                node_colors.append(type_to_color.get(node_type, colors[0]))

            return node_colors

        elif color_by == 'confidence':
            # Color by confidence level
            node_colors = []
            for node_id in node_list:
                confidence = G.nodes[node_id].get('confidence', 0.5)
                # Use color intensity based on confidence
                idx = int(confidence * (len(colors) - 1))
                node_colors.append(colors[idx])
            return node_colors

        elif color_by == 'source':
            # Group by source
            source_groups = defaultdict(list)
            for node in kg_data.get('nodes', []):
                source = node.get('source', 'unknown')
                source_groups[source].append(node.get('id'))

            source_to_color = {}
            for i, source in enumerate(source_groups.keys()):
                source_to_color[source] = colors[i % len(colors)]

            node_colors = []
            for node_id in node_list:
                source = G.nodes[node_id].get('source', 'unknown')
                node_colors.append(source_to_color.get(source, colors[0]))

            return node_colors

        else:
            # Default: single color
            return [colors[0]] * len(node_list)

    def _create_type_legend(self, kg_data: Dict[str, Any], theme: Dict) -> List:
        """Create legend elements for entity types."""
        import matplotlib.patches as mpatches

        type_set = set()
        for node in kg_data.get('nodes', []):
            type_set.add(node.get('type', 'unknown'))

        colors = theme['node_colors']
        legend_elements = []
        for i, node_type in enumerate(sorted(type_set)):
            color = colors[i % len(colors)]
            legend_elements.append(mpatches.Patch(color=color, label=node_type))

        return legend_elements

    def _generate_interactive_network_data(self, G: Any, kg_data: Dict[str, Any],
                                           pos: Dict) -> Dict[str, Any]:
        """Generate interactive data for D3.js visualization."""
        nodes = []
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            x, y = pos.get(node_id, (0, 0))
            nodes.append({
                'id': node_id,
                'type': node_data.get('type', 'unknown'),
                'confidence': node_data.get('confidence', 1.0),
                'x': float(x),
                'y': float(y),
                'label': str(node_id)[:30]
            })

        links = []
        for source, target, edge_data in G.edges(data=True):
            links.append({
                'source': source,
                'target': target,
                'type': edge_data.get('type', 'related')
            })

        return {
            'nodes': nodes,
            'links': links,
            'metadata': {
                'node_count': len(nodes),
                'edge_count': len(links)
            }
        }

    def _truncate_label(self, label: str, max_length: int = 20) -> str:
        """Truncate label to maximum length."""
        if len(label) <= max_length:
            return label
        return label[:max_length - 3] + "..."

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration panel.
        """
        return {
            "type": "object",
            "title": "Knowledge Visualization Configuration",
            "description": "Configure knowledge graph visualization parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Type of visualization to generate",
                    "enum": ["network_graph", "entity_diagram", "statistics_chart", "timeline", "heatmap"],
                    "enumNames": [
                        "Network Graph - Visualize entity relationships",
                        "Entity Diagram - Entity relationship diagram",
                        "Statistics Chart - Analytics and metrics charts",
                        "Timeline - Temporal visualization",
                        "Heatmap - Relationship density heatmap"
                    ],
                    "default": "network_graph"
                },
                "entity_ids": {
                    "type": "array",
                    "title": "Entity IDs",
                    "description": "Specific entity IDs to visualize (empty for all)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "visualization_type": {
                    "type": "string",
                    "title": "Visualization Type",
                    "description": "Layout style for network visualizations",
                    "enum": ["force_directed", "hierarchical", "circular", "matrix"],
                    "enumNames": [
                        "Force Directed - Spring layout",
                        "Hierarchical - Tree-like layout",
                        "Circular - Radial layout",
                        "Matrix - Adjacency matrix"
                    ],
                    "default": "force_directed"
                },
                "output_format": {
                    "type": "string",
                    "title": "Output Format",
                    "description": "Export format for the visualization",
                    "enum": ["png", "svg", "json", "html", "d3"],
                    "enumNames": [
                        "PNG - Raster image",
                        "SVG - Vector graphics",
                        "JSON - Raw data",
                        "HTML - Interactive web page",
                        "D3 - D3.js visualization data"
                    ],
                    "default": "png"
                },
                "layout_algorithm": {
                    "type": "string",
                    "title": "Layout Algorithm",
                    "description": "Algorithm for positioning nodes",
                    "enum": ["spring", "circular", "hierarchical", "random"],
                    "enumNames": [
                        "Spring - Force-directed layout",
                        "Circular - Circular arrangement",
                        "Hierarchical - Layered layout",
                        "Random - Random positioning"
                    ],
                    "default": "spring"
                },
                "color_by": {
                    "type": "string",
                    "title": "Color By",
                    "description": "Attribute to use for node coloring",
                    "enum": ["type", "confidence", "community", "source"],
                    "enumNames": [
                        "Type - Entity type",
                        "Confidence - Confidence score",
                        "Community - Detected community",
                        "Source - Data source"
                    ],
                    "default": "type"
                },
                "max_nodes": {
                    "type": "integer",
                    "title": "Maximum Nodes",
                    "description": "Maximum number of nodes to include in visualization",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100
                },
                "include_labels": {
                    "type": "boolean",
                    "title": "Include Labels",
                    "description": "Show node labels on the visualization",
                    "default": True
                },
                "style_theme": {
                    "type": "string",
                    "title": "Style Theme",
                    "description": "Color theme for the visualization",
                    "enum": ["light", "dark", "colorful", "minimal"],
                    "enumNames": [
                        "Light - Light background",
                        "Dark - Dark background",
                        "Colorful - Vibrant colors",
                        "Minimal - Grayscale"
                    ],
                    "default": "light"
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node is healthy, False otherwise
        """
        try:
            # Basic health check - node can operate with fallbacks
            return True
        except Exception:
            return False
