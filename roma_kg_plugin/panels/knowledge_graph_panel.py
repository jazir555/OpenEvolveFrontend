"""
ROMA Knowledge Graph Plugin - Visualization Panel

Provides interactive knowledge graph visualization and exploration capabilities
within the ROMA terminal user interface.

This module follows the Air Gap principle - no direct imports from ROMA core.
All dependencies are injected through the constructor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from prompt_toolkit.layout import (
    Align,
    Dimension,
    FloatContainer,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.widgets import Box, Button, Label, TextArea

# Import from plugin's own visualization module (Air Gap compliant)
from ..visualization.ascii_graph import AsciiGraphRenderer


class KnowledgeGraphPanel:
    """
    ROMA TUI panel for knowledge graph visualization (Plugin Version).

    This is a plugin component that extends ROMA's TUI without modifying
    ROMA core files. All dependencies are injected.

    Features:
    - Interactive graph display
    - Node/edge details
    - Community browser
    - Search integration
    - Export functionality

    Layout:
    ┌─────────────────────────────────────────────────┐
    │ Knowledge Graph Explorer                        │
    ├──────────────┬──────────────────────────────────┤
    │ Graph Stats  │ Interactive Graph                │
    │              │                                  │
    │ - Nodes: 123  │     [Node A]───→[Node B]        │
    │ - Edges: 456  │        │         │              │
    │ - Comm: 12    │        ↓         ↓              │
    │              │     [Node C]   [Node D]           │
    ├──────────────┴──────────────────────────────────┤
    │ Search: [______________] [Filter] [Export]       │
    │ Selected: Node A - Related to B, C              │
    └─────────────────────────────────────────────────┘
    """

    def __init__(self, roma_client: Any, kg_manager: Any):
        """Initialize knowledge graph panel.

        Args:
            roma_client: ROMA API client instance
            kg_manager: Knowledge graph manager instance
        """
        self.client = roma_client
        self.kg = kg_manager
        self.current_graph = None
        self.selected_node = None
        self.filter_state = {
            "node_types": [],
            "edge_types": [],
            "communities": [],
            "min_centrality": 0.0,
        }
        self.renderer = AsciiGraphRenderer()

        # UI components
        self.search_text = TextArea(text="", height=1, prompt="Search: ")
        self.status_label = Label(text="Ready")
        self.details_label = Label(text="Select a node to view details")

        # Graph display area
        self.graph_display = TextArea(
            text="No graph loaded",
            read_only=True,
            scrollbar=True,
        )

        # Statistics display
        self.stats_label = Label(text="Nodes: 0 | Edges: 0 | Communities: 0")

        logger.info("KnowledgeGraphPanel initialized")

    def create_layout(self) -> Layout:
        """Create TUI layout for knowledge graph panel.

        Returns:
            Layout configured for knowledge graph visualization
        """
        # Create statistics panel (left side)
        stats_panel = HSplit(
            [
                Label(text="┌─ Graph Statistics ─┐", style="class:title"),
                Box(self.stats_label, padding=0, style="class:stats"),
                Label(text=""),
                Label(text="┌─ Filters ─┐", style="class:title"),
                Label(text="Active filters:"),
                Label(text=f"  Node types: {len(self.filter_state['node_types'])}"),
                Label(text=f"  Edge types: {len(self.filter_state['edge_types'])}"),
                Label(text=f"  Communities: {len(self.filter_state['communities'])}"),
            ],
            width=Dimension(min=25, max=30),
        )

        # Create graph display panel (center)
        graph_panel = HSplit(
            [
                Label(text="┌─ Interactive Graph ─┐", style="class:title"),
                Box(self.graph_display, padding=0, style="class:graph"),
            ]
        )

        # Create details panel (bottom)
        details_panel = HSplit(
            [
                Label(text="┌─ Details ─┐", style="class:title"),
                Box(self.details_label, padding=0, style="class:details"),
            ]
        )

        # Create search bar (bottom)
        search_panel = HSplit(
            [
                Label(text="┌─ Search & Actions ─┐", style="class:title"),
                VSplit(
                    [
                        Box(self.search_text, padding=0),
                        Button(text="Search", handler=self._on_search),
                        Button(text="Filter", handler=self._on_filter),
                        Button(text="Export", handler=self._on_export),
                    ],
                    padding=1,
                ),
            ]
        )

        # Main layout
        main_layout = VSplit(
            [
                stats_panel,
                graph_panel,
            ],
            padding=1,
        )

        root_container = HSplit(
            [
                Label(text="Knowledge Graph Explorer", style="class:header"),
                main_layout,
                details_panel,
                search_panel,
                Box(self.status_label, padding=0, style="class:status"),
            ],
            padding=0,
        )

        return Layout(FloatContainer(root_container, floats=[]))

    async def display_graph(self, graph_data: Dict[str, Any]) -> None:
        """Display knowledge graph in TUI.

        Args:
            graph_data: Dictionary containing nodes, edges, and metadata
        """
        logger.info(f"Displaying graph with {len(graph_data.get('nodes', []))} nodes")

        self.current_graph = graph_data

        # Update statistics
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        communities = graph_data.get("communities", {})

        stats_text = (
            f"Nodes: {len(nodes)} | Edges: {len(edges)} | "
            f"Communities: {len(communities)}"
        )
        self.stats_label.text = stats_text

        # Render ASCII graph
        try:
            ascii_graph = self.renderer.render_graph(
                graph_data,
                width=80,
                height=20,
            )
            self.graph_display.text = ascii_graph
        except Exception as e:
            logger.error(f"Failed to render graph: {e}")
            self.graph_display.text = f"Error rendering graph: {e}"

        self.status_label.text = f"Loaded graph with {len(nodes)} nodes"

    async def show_node_details(self, node_id: str) -> None:
        """Show detailed information about selected node.

        Args:
            node_id: ID of node to display
        """
        logger.info(f"Showing details for node: {node_id}")

        self.selected_node = node_id

        if not self.current_graph:
            self.details_label.text = "No graph loaded"
            return

        # Find node in current graph
        nodes = self.current_graph.get("nodes", [])
        node = next((n for n in nodes if n.get("id") == node_id), None)

        if not node:
            self.details_label.text = f"Node {node_id} not found"
            return

        # Build details text
        details = []
        details.append(f"ID: {node.get('id', 'N/A')}")
        details.append(f"Type: {node.get('type', 'N/A')}")
        details.append(f"Label: {node.get('label', 'N/A')}")

        # Add attributes
        attributes = node.get("attributes", {})
        if attributes:
            details.append("\nAttributes:")
            for key, value in attributes.items():
                details.append(f"  {key}: {value}")

        # Add connections
        edges = self.current_graph.get("edges", [])
        connections = [e for e in edges if e.get("source") == node_id or e.get("target") == node_id]
        details.append(f"\nConnections: {len(connections)}")

        # Add community info
        communities = self.current_graph.get("communities", {})
        for comm_id, comm_nodes in communities.items():
            if node_id in comm_nodes:
                details.append(f"\nCommunity: {comm_id}")
                break

        self.details_label.text = "\n".join(details)
        self.status_label.text = f"Selected node: {node_id}"

    async def show_community_browse(self) -> None:
        """Browse communities and their members."""
        logger.info("Browsing communities")

        if not self.current_graph:
            self.status_label.text = "No graph loaded"
            return

        communities = self.current_graph.get("communities", {})

        if not communities:
            self.status_label.text = "No communities found"
            return

        # Build community display
        community_text = ["Community Overview:\n"]

        for comm_id, nodes in communities.items():
            community_text.append(f"\nCommunity {comm_id}:")
            community_text.append(f"  Members: {len(nodes)}")

            # Show first few members
            for node_id in nodes[:5]:
                nodes_list = self.current_graph.get("nodes", [])
                node = next((n for n in nodes_list if n.get("id") == node_id), None)
                if node:
                    label = node.get("label", node.get("id", "Unknown"))
                    community_text.append(f"    - {label}")

            if len(nodes) > 5:
                community_text.append(f"    ... and {len(nodes) - 5} more")

        self.graph_display.text = "\n".join(community_text)
        self.status_label.text = f"Displaying {len(communities)} communities"

    async def show_graph_statistics(self) -> None:
        """Display comprehensive graph statistics."""
        logger.info("Showing graph statistics")

        if not self.current_graph:
            self.status_label.text = "No graph loaded"
            return

        nodes = self.current_graph.get("nodes", [])
        edges = self.current_graph.get("edges", [])
        communities = self.current_graph.get("communities", {})

        # Calculate statistics
        node_types = {}
        for node in nodes:
            node_type = node.get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1

        edge_types = {}
        for edge in edges:
            edge_type = edge.get("type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        # Build statistics display
        stats_text = ["Graph Statistics:\n"]
        stats_text.append(f"Total Nodes: {len(nodes)}")
        stats_text.append(f"Total Edges: {len(edges)}")
        stats_text.append(f"Total Communities: {len(communities)}")

        stats_text.append("\nNode Types:")
        for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
            stats_text.append(f"  {node_type}: {count}")

        stats_text.append("\nEdge Types:")
        for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
            stats_text.append(f"  {edge_type}: {count}")

        stats_text.append("\nCommunity Sizes:")
        for comm_id, comm_nodes in communities.items():
            stats_text.append(f"  Community {comm_id}: {len(comm_nodes)} nodes")

        self.graph_display.text = "\n".join(stats_text)
        self.status_label.text = "Graph statistics displayed"

    async def search_graph(self, query: str) -> None:
        """Search graph and display results.

        Args:
            query: Search query string
        """
        logger.info(f"Searching graph for: {query}")

        if not self.current_graph:
            self.status_label.text = "No graph loaded"
            return

        if not query.strip():
            self.status_label.text = "Please enter a search query"
            return

        query = query.lower()
        nodes = self.current_graph.get("nodes", [])

        # Search nodes
        results = []
        for node in nodes:
            node_id = node.get("id", "").lower()
            label = node.get("label", "").lower()
            node_type = node.get("type", "").lower()

            if query in node_id or query in label or query in node_type:
                results.append(node)

        # Display results
        if results:
            results_text = [f"Search Results for '{query}':\n"]
            results_text.append(f"Found {len(results)} nodes\n")

            for node in results[:20]:  # Limit to first 20 results
                label = node.get("label", node.get("id", "Unknown"))
                node_type = node.get("type", "Unknown")
                results_text.append(f"- {label} ({node_type})")

            if len(results) > 20:
                results_text.append(f"\n... and {len(results) - 20} more")

            self.graph_display.text = "\n".join(results_text)
            self.status_label.text = f"Found {len(results)} results"
        else:
            self.graph_display.text = f"No results found for '{query}'"
            self.status_label.text = "No results found"

    async def export_graph(self, format: str = "json") -> None:
        """Export graph to file.

        Args:
            format: Export format (json, gexf, csv)
        """
        logger.info(f"Exporting graph in {format} format")

        if not self.current_graph:
            self.status_label.text = "No graph loaded"
            return

        # This would integrate with the export functionality
        # For now, just update status
        self.status_label.text = f"Exporting graph as {format}..."

        # Implementation would depend on export utilities
        # Example:
        # from roma_dspy.tui.utils.export import export_knowledge_graph
        # result = await export_knowledge_graph(self.current_graph, format)
        # self.status_label.text = f"Exported to {result['path']}"

    def _on_search(self) -> None:
        """Handle search button click."""
        query = self.search_text.text
        if query:
            # In a real implementation, this would trigger the async search
            self.status_label.text = f"Searching for: {query}..."

    def _on_filter(self) -> None:
        """Handle filter button click."""
        self.status_label.text = "Filter dialog would open here"

    def _on_export(self) -> None:
        """Handle export button click."""
        self.status_label.text = "Export dialog would open here"

    def update_filter(self, filter_type: str, value: Any) -> None:
        """Update filter state.

        Args:
            filter_type: Type of filter to update
            value: New filter value
        """
        if filter_type in self.filter_state:
            self.filter_state[filter_type] = value
            logger.info(f"Updated filter {filter_type}: {value}")
