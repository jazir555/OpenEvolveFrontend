"""
Interactive Graph Exploration for ROMA TUI.

Provides interactive traversal and exploration capabilities for knowledge graphs.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from loguru import logger


class InteractiveGraphExplorer:
    """
    Interactive graph exploration in ROMA TUI.

    Features:
    - Graph traversal
    - Path finding
    - Neighborhood exploration
    - Node expansion/collapse
    """

    def __init__(self, kg_manager: Any):
        """Initialize graph explorer.

        Args:
            kg_manager: Knowledge graph manager instance
        """
        self.kg = kg_manager
        self.exploration_stack: List[str] = []
        self.current_focus: Optional[str] = None
        self.expanded_nodes: set = set()
        self.node_cache: Dict[str, Any] = {}

        logger.info("InteractiveGraphExplorer initialized")

    async def explore_neighborhood(
        self,
        node_id: str,
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        Explore neighborhood around node.

        Args:
            node_id: Central node ID
            depth: Exploration depth (1-hop, 2-hop, etc.)

        Returns:
            Dictionary containing neighborhood data including:
            - center: The central node
            - neighbors: Direct neighbors (depth 1)
            - two_hop: Two-hop neighbors (depth 2)
            - paths: Paths to neighbors
        """
        logger.info(f"Exploring neighborhood of {node_id} at depth {depth}")

        # Get graph from knowledge manager
        graph = await self._get_graph()
        if node_id not in graph.nodes:
            logger.error(f"Node {node_id} not found in graph")
            return {'error': f'Node {node_id} not found'}

        result = {
            'center': node_id,
            'depth': depth,
            'neighbors': [],
            'two_hop': [],
            'paths': {},
        }

        # Get direct neighbors (depth 1)
        neighbors = list(graph.neighbors(node_id))
        result['neighbors'] = neighbors

        # Get two-hop neighbors (depth 2)
        if depth >= 2:
            two_hop = set()
            for neighbor in neighbors:
                two_hop.update(graph.neighbors(neighbor))

            # Remove center and direct neighbors
            two_hop.discard(node_id)
            two_hop.discard(neighbors)

            result['two_hop'] = list(two_hop)

        # Get paths to neighbors
        for neighbor in neighbors:
            try:
                path = nx.shortest_path(graph, node_id, neighbor)
                result['paths'][neighbor] = path
            except nx.NetworkXNoPath:
                logger.warning(f"No path from {node_id} to {neighbor}")

        # Update current focus
        self.current_focus = node_id
        self.exploration_stack.append(node_id)

        return result

    async def find_shortest_path(
        self,
        source: str,
        target: str
    ) -> Dict[str, Any]:
        """
        Find and display shortest path between nodes.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Dictionary containing:
            - path: List of node IDs in the path
            - length: Number of edges in the path
            - edges: List of edges along the path
        """
        logger.info(f"Finding shortest path from {source} to {target}")

        graph = await self._get_graph()

        if source not in graph.nodes:
            return {'error': f'Source node {source} not found'}
        if target not in graph.nodes:
            return {'error': f'Target node {target} not found'}

        try:
            path = nx.shortest_path(graph, source, target)
            path_length = len(path) - 1

            # Get edges along the path
            edges = []
            for i in range(len(path) - 1):
                edge_data = graph.get_edge_data(path[i], path[i + 1], default={})
                edges.append({
                    'source': path[i],
                    'target': path[i + 1],
                    'data': edge_data
                })

            return {
                'path': path,
                'length': path_length,
                'edges': edges,
                'exists': True
            }

        except nx.NetworkXNoPath:
            return {
                'path': [],
                'length': 0,
                'edges': [],
                'exists': False,
                'message': 'No path exists between nodes'
            }

    async def find_all_paths(
        self,
        source: str,
        target: str,
        max_paths: int = 10,
        max_length: int = 5
    ) -> Dict[str, Any]:
        """
        Find all simple paths between nodes.

        Args:
            source: Source node ID
            target: Target node ID
            max_paths: Maximum number of paths to return
            max_length: Maximum path length

        Returns:
            Dictionary with all found paths
        """
        logger.info(f"Finding all paths from {source} to {target}")

        graph = await self._get_graph()

        if source not in graph.nodes or target not in graph.nodes:
            return {'error': 'One or both nodes not found'}

        try:
            paths = list(nx.all_simple_paths(
                graph,
                source,
                target,
                cutoff=max_length
            ))[:max_paths]

            return {
                'paths': paths,
                'count': len(paths),
                'shortest_length': min(len(p) for p in paths) if paths else 0
            }

        except Exception as e:
            logger.error(f"Error finding paths: {e}")
            return {'error': str(e)}

    async def find_communities(
        self,
        node_id: str
    ) -> Dict[str, Any]:
        """
        Find and display community membership.

        Args:
            node_id: Node to analyze

        Returns:
            Dictionary containing community information
        """
        logger.info(f"Finding communities for node {node_id}")

        # This would integrate with community detection
        # For now, return placeholder
        return {
            'node': node_id,
            'community': None,
            'community_members': [],
            'modularity': 0.0
        }

    async def expand_node(self, node_id: str) -> Dict[str, Any]:
        """
        Expand node to show connections.

        Args:
            node_id: Node to expand

        Returns:
            Expanded node data with connections
        """
        logger.info(f"Expanding node {node_id}")

        graph = await self._get_graph()

        if node_id not in graph.nodes:
            return {'error': f'Node {node_id} not found'}

        self.expanded_nodes.add(node_id)

        # Get all neighbors
        neighbors = list(graph.neighbors(node_id))

        # Get connections with edge data
        connections = []
        for neighbor in neighbors:
            edge_data = graph.get_edge_data(node_id, neighbor, default={})
            neighbor_data = graph.nodes.get(neighbor, {})
            connections.append({
                'target': neighbor,
                'edge_data': edge_data,
                'node_data': neighbor_data
            })

        return {
            'node': node_id,
            'connections': connections,
            'degree': len(neighbors)
        }

    async def collapse_node(self, node_id: str) -> Dict[str, Any]:
        """
        Collapse node to hide connections.

        Args:
            node_id: Node to collapse

        Returns:
            Collapse confirmation
        """
        logger.info(f"Collapsing node {node_id}")

        if node_id in self.expanded_nodes:
            self.expanded_nodes.remove(node_id)

        return {
            'node': node_id,
            'collapsed': True,
            'expanded_nodes': list(self.expanded_nodes)
        }

    async def navigate_back(self) -> Optional[str]:
        """
        Navigate back to previous node in exploration stack.

        Returns:
            Previous node ID, or None if stack is empty
        """
        if len(self.exploration_stack) > 1:
            self.exploration_stack.pop()  # Remove current
            previous = self.exploration_stack[-1]
            self.current_focus = previous
            logger.info(f"Navigated back to {previous}")
            return previous

        logger.warning("No previous node to navigate to")
        return None

    async def find_connected_components(
        self
    ) -> Dict[str, Any]:
        """
        Find all connected components in the graph.

        Returns:
            Dictionary with component information
        """
        logger.info("Finding connected components")

        graph = await self._get_graph()

        components = list(nx.connected_components(graph))

        return {
            'num_components': len(components),
            'components': [
                {
                    'id': i,
                    'size': len(comp),
                    'nodes': list(comp)
                }
                for i, comp in enumerate(components)
            ],
            'largest_component_size': max(len(c) for c in components) if components else 0
        }

    async def find_bridges(self) -> List[Dict[str, Any]]:
        """
        Find all bridge edges in the graph.

        Bridges are edges whose removal would disconnect the graph.

        Returns:
            List of bridge edges
        """
        logger.info("Finding bridge edges")

        graph = await self._get_graph()

        bridges = list(nx.bridges(graph))

        return [
            {
                'source': u,
                'target': v,
                'edge_data': graph.get_edge_data(u, v, default={})
            }
            for u, v in bridges
        ]

    async def get_node_statistics(
        self,
        node_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a node.

        Args:
            node_id: Node to analyze

        Returns:
            Dictionary with node statistics
        """
        logger.info(f"Getting statistics for node {node_id}")

        graph = await self._get_graph()

        if node_id not in graph.nodes:
            return {'error': f'Node {node_id} not found'}

        node_data = graph.nodes[node_id]

        # Compute centrality measures
        degree = graph.degree(node_id)

        # Get neighbors
        neighbors = list(graph.neighbors(node_id))

        return {
            'node_id': node_id,
            'data': node_data,
            'degree': degree,
            'neighbors': neighbors,
            'is_expanded': node_id in self.expanded_nodes
        }

    async def _get_graph(self) -> nx.Graph:
        """
        Get NetworkX graph from knowledge manager.

        Returns:
            NetworkX Graph object
        """
        # This would integrate with the actual knowledge graph manager
        # For now, return an empty graph
        return nx.Graph()

    def reset_exploration(self) -> None:
        """Reset exploration state."""
        self.exploration_stack.clear()
        self.current_focus = None
        self.expanded_nodes.clear()
        logger.info("Exploration state reset")

    def get_exploration_history(self) -> List[str]:
        """
        Get exploration history.

        Returns:
            List of visited node IDs
        """
        return self.exploration_stack.copy()
