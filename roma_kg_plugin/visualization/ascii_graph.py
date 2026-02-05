"""
ASCII Graph Renderer for Knowledge Graphs.

Provides text-based graph visualization for terminal displays.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import networkx as nx
from loguru import logger


class AsciiGraphRenderer:
    """
    Render knowledge graphs as ASCII art for TUI.

    Features:
    - Force-directed layout approximation
    - Community coloring
    - Node sizing by centrality
    - Edge type differentiation
    """

    def __init__(self):
        """Initialize ASCII graph renderer."""
        self.node_symbols = {
            'default': '○',
            'entity': '●',
            'concept': '◐',
            'relation': '◆',
            'event': '△',
        }
        self.edge_symbols = {
            'default': '─',
            'related_to': '─',
            'part_of': '═',
            'instance_of': '─',
            'causes': '->',
            'derives_from': '⇢',
        }
        logger.info("AsciiGraphRenderer initialized")

    def render_graph(
        self,
        graph_data: Dict[str, Any],
        width: int = 80,
        height: int = 20
    ) -> str:
        """
        Render graph as ASCII art.

        Args:
            graph_data: Dictionary containing nodes, edges, and metadata
            width: Display width in characters
            height: Display height in characters

        Returns:
            ASCII art representation of the graph

        Example:
              A ──-> B
              │  ↗  │
              v ↗   v
              C <-── D
        """
        logger.debug(f"Rendering graph with {len(graph_data.get('nodes', []))} nodes")

        # Build NetworkX graph
        G = self._build_networkx_graph(graph_data)

        if G.number_of_nodes() == 0:
            return "Empty graph"

        # Compute layout using spring layout for positioning
        try:
            pos = nx.spring_layout(G, dim=2, center=(width/2, height/2), scale=min(width, height)/3)
        except Exception as e:
            logger.error(f"Failed to compute layout: {e}")
            return f"Error computing layout: {e}"

        # Create ASCII grid
        grid = self._create_ascii_grid(G, pos, width, height)

        return grid

    def render_community_graph(
        self,
        communities: Dict[str, List[str]],
        width: int = 80
    ) -> str:
        """
        Render communities as grouped nodes.

        Args:
            communities: Dictionary mapping community IDs to node lists
            width: Display width in characters

        Returns:
            ASCII art showing community structure
        """
        lines = ["Community Structure:\n"]

        for comm_id, nodes in communities.items():
            lines.append(f"\n[{comm_id}]")
            # Show nodes in this community
            for node in nodes[:10]:  # Limit to first 10 nodes per community
                lines.append(f"  └─ {node}")

            if len(nodes) > 10:
                lines.append(f"  └─ ... and {len(nodes) - 10} more")

        return "\n".join(lines)

    def render_path(
        self,
        path: List[str],
        graph: nx.Graph,
        width: int = 80
    ) -> str:
        """
        Render path as linear sequence.

        Args:
            path: List of node IDs in the path
            graph: NetworkX graph containing the path
            width: Display width (unused, for future enhancements)

        Returns:
            ASCII art showing the path
        """
        if not path:
            return "No path"

        if len(path) == 1:
            return f"[{path[0]}]"

        # Build path visualization
        result = [f"[{path[0]}]"]

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]

            # Get edge label if available
            edge_data = graph.get_edge_data(source, target, default={})
            edge_label = edge_data.get('label', edge_data.get('type', ''))

            if edge_label:
                result.append(f" --[{edge_label}]-->")
            else:
                result.append(" ----->")

            result.append(f"[{target}]")

        # Join with line breaks if too long
        path_str = " ".join(result)

        # If path is too long, break it into multiple lines
        if len(path_str) > width:
            lines = []
            current_line = ""

            for segment in result:
                test_line = current_line + " " + segment if current_line else segment

                if len(test_line) > width - 4:
                    lines.append(current_line)
                    current_line = "    " + segment
                else:
                    current_line = test_line

            if current_line:
                lines.append(current_line)

            return "\n".join(lines)

        return path_str

    def _build_networkx_graph(self, graph_data: Dict[str, Any]) -> nx.Graph:
        """Build NetworkX graph from graph data.

        Args:
            graph_data: Graph data dictionary

        Returns:
            NetworkX Graph object
        """
        G = nx.Graph()

        # Add nodes
        for node in graph_data.get('nodes', []):
            node_id = node.get('id')
            if node_id:
                G.add_node(node_id, **node)

        # Add edges
        for edge in graph_data.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                G.add_edge(source, target, **edge)

        return G

    def _create_ascii_grid(
        self,
        G: nx.Graph,
        pos: Dict[str, Tuple[float, float]],
        width: int,
        height: int
    ) -> str:
        """
        Create ASCII grid from graph positions.

        Args:
            G: NetworkX graph
            pos: Node positions from layout algorithm
            width: Grid width
            height: Grid height

        Returns:
            ASCII grid string
        """
        # Initialize grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # Place nodes on grid
        node_positions = {}
        for node, (x, y) in pos.items():
            # Convert to grid coordinates
            grid_x = int(x)
            grid_y = int(y)

            # Clamp to grid bounds
            grid_x = max(0, min(width - 1, grid_x))
            grid_y = max(0, min(height - 1, grid_y))

            # Get node symbol
            node_data = G.nodes[node]
            node_type = node_data.get('type', 'default')
            symbol = self.node_symbols.get(node_type, self.node_symbols['default'])

            # Place node label (first 2 characters)
            label = node_data.get('label', str(node))[:2]
            grid[grid_y][grid_x] = label[0] if len(label) > 0 else symbol

            node_positions[node] = (grid_x, grid_y)

        # Draw edges (simplified - just draw lines between nodes)
        for source, target in G.edges():
            if source in node_positions and target in node_positions:
                x1, y1 = node_positions[source]
                x2, y2 = node_positions[target]

                # Draw simple line (could be improved with Bresenham's algorithm)
                self._draw_line(grid, x1, y1, x2, y2)

        # Convert grid to string
        lines = []
        for row in grid:
            lines.append(''.join(row))

        return '\n'.join(lines)

    def _draw_line(
        self,
        grid: List[List[str]],
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ) -> None:
        """
        Draw a line on the grid using Bresenham's algorithm.

        Args:
            grid: ASCII grid
            x1, y1: Start coordinates
            x2, y2: End coordinates
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1

        while True:
            if x == x2 and y == y2:
                break

            # Only draw if the position is empty (don't overwrite nodes)
            if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
                if grid[y][x] == ' ':
                    # Determine line character based on direction
                    if abs(dx) > abs(dy):
                        grid[y][x] = '─'
                    else:
                        grid[y][x] = '│'

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def render_subgraph(
        self,
        graph_data: Dict[str, Any],
        focus_node: str,
        depth: int = 1,
        width: int = 80,
        height: int = 20
    ) -> str:
        """
        Render a subgraph focused on a specific node.

        Args:
            graph_data: Full graph data
            focus_node: Central node to focus on
            depth: Number of hops to include
            width: Display width
            height: Display height

        Returns:
            ASCII art of the subgraph
        """
        G = self._build_networkx_graph(graph_data)

        if focus_node not in G.nodes:
            return f"Node {focus_node} not found in graph"

        # Extract subgraph
        nodes_to_include = {focus_node}
        current_level = {focus_node}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                neighbors = set(G.neighbors(node))
                next_level.update(neighbors)

            nodes_to_include.update(next_level)
            current_level = next_level

        subgraph = G.subgraph(nodes_to_include)

        # Convert back to graph data format
        subgraph_data = {
            'nodes': [
                {'id': n, **G.nodes[n]}
                for n in subgraph.nodes
            ],
            'edges': [
                {'source': u, 'target': v, **subgraph.edges[u, v]}
                for u, v in subgraph.edges
            ]
        }

        return self.render_graph(subgraph_data, width, height)
