"""
ROMA Knowledge Graph Plugin - Commands

Provides command-line interface for knowledge graph operations.

This module follows the Air Gap principle - all dependencies are injected.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger


class KnowledgeGraphCommands:
    """
    ROMA TUI commands for knowledge graph interaction (Plugin Version).

    This is a plugin component that extends ROMA's command system without
    modifying ROMA core files. All dependencies are injected.

    Commands:
    /kg search <query> - Search knowledge graph
    /kg explore <node> - Explore neighborhood
    /kg path <from> <to> - Find shortest path
    /kg communities - List all communities
    /kg stats - Show graph statistics
    /kg export <format> - Export graph
    /kg timeline <entity> - Show temporal timeline
    /kg analyze <type> - Run graph analysis
    """

    def __init__(
        self,
        roma_client: Optional[Any] = None,
        kg_engine: Optional[Any] = None
    ):
        """
        Initialize knowledge graph commands with dependency injection.

        Args:
            roma_client: ROMA client instance (injected)
            kg_engine: Knowledge graph engine instance (injected)
        """
        self.roma_client = roma_client
        self.kg_engine = kg_engine
        self.command_history: List[str] = []

        # Optional: panel and explorer references (injected when needed)
        self.panel: Optional[Any] = None
        self.explorer: Optional[Any] = None

        logger.info("KnowledgeGraphCommands initialized (plugin mode)")

    async def handle_command(self, command: str) -> str:
        """
        Handle knowledge graph command.

        Args:
            command: Command string (e.g., "/kg search python")

        Returns:
            Result message
        """
        self.command_history.append(command)

        parts = command.strip().split()
        if len(parts) < 2:
            return "Usage: /kg <command> [args...]"

        cmd = parts[1].lower()
        args = parts[2:] if len(parts) > 2 else []

        # Route to appropriate handler
        handlers = {
            'search': self._handle_search_command,
            'explore': self._handle_explore_command,
            'path': self._handle_path_command,
            'communities': self._handle_communities_command,
            'stats': self._handle_stats_command,
            'export': self._handle_export_command,
            'timeline': self._handle_timeline_command,
            'analyze': self._handle_analyze_command,
        }

        if cmd not in handlers:
            return f"Unknown command: {cmd}. Available: {', '.join(handlers.keys())}"

        try:
            return await handlers[cmd](args)
        except Exception as e:
            logger.error(f"Error handling command {cmd}: {e}")
            return f"Error: {e}"

    async def handle_search_command(self, query: str) -> str:
        """
        Handle /kg search command.

        Args:
            query: Search query string

        Returns:
            Result message
        """
        logger.info(f"Handling search command: {query}")

        if not query:
            return "Usage: /kg search <query>"

        await self.panel.search_graph(query)

        return f"Searching for: {query}"

    async def handle_explore_command(self, node_id: str) -> str:
        """
        Handle /kg explore command.

        Args:
            node_id: Node to explore

        Returns:
            Result message
        """
        logger.info(f"Handling explore command: {node_id}")

        if not node_id:
            return "Usage: /kg explore <node_id> [depth]"

        # Get neighborhood
        result = await self.explorer.explore_neighborhood(node_id, depth=2)

        if 'error' in result:
            return result['error']

        # Show node details
        await self.panel.show_node_details(node_id)

        return f"Explored neighborhood of {node_id}: {len(result['neighbors'])} neighbors found"

    async def handle_path_command(self, source: str, target: str = None) -> str:
        """
        Handle /kg path command.

        Args:
            source: Source node
            target: Target node

        Returns:
            Result message
        """
        logger.info(f"Handling path command: {source} -> {target}")

        if not source or not target:
            return "Usage: /kg path <source> <target>"

        result = await self.explorer.find_shortest_path(source, target)

        if 'error' in result:
            return result['error']

        if not result['exists']:
            return f"No path exists between {source} and {target}"

        path_str = " → ".join(result['path'])

        return f"Path found (length {result['length']}): {path_str}"

    async def handle_communities_command(self) -> str:
        """
        Handle /kg communities command.

        Returns:
            Result message
        """
        logger.info("Handling communities command")

        await self.panel.show_community_browse()

        return "Displaying communities"

    async def handle_stats_command(self) -> str:
        """
        Handle /kg stats command.

        Returns:
            Result message
        """
        logger.info("Handling stats command")

        await self.panel.show_graph_statistics()

        return "Graph statistics displayed"

    async def handle_export_command(self, format: str = "json") -> str:
        """
        Handle /kg export command.

        Args:
            format: Export format (json, gexf, csv)

        Returns:
            Result message
        """
        logger.info(f"Handling export command: {format}")

        await self.panel.export_graph(format)

        return f"Graph exported as {format}"

    async def handle_timeline_command(self, entity: str) -> str:
        """
        Handle /kg timeline command.

        Args:
            entity: Entity to show timeline for

        Returns:
            Result message
        """
        logger.info(f"Handling timeline command: {entity}")

        if not entity:
            return "Usage: /kg timeline <entity>"

        # This would integrate with temporal knowledge tracking
        return f"Timeline for {entity}: [Not yet implemented]"

    async def handle_analyze_command(self, analysis_type: str) -> str:
        """
        Handle /kg analyze command.

        Args:
            analysis_type: Type of analysis to run (centrality, community, etc.)

        Returns:
            Result message
        """
        logger.info(f"Handling analyze command: {analysis_type}")

        if not analysis_type:
            return "Usage: /kg analyze <type>"

        analyses = {
            'centrality': 'Running centrality analysis...',
            'community': 'Running community detection...',
            'connectivity': 'Analyzing connectivity...',
            'components': 'Finding connected components...',
        }

        if analysis_type not in analyses:
            available = ', '.join(analyses.keys())
            return f"Unknown analysis type. Available: {available}"

        return analyses[analysis_type]

    async def _handle_search_command(self, args: List[str]) -> str:
        """Internal handler for search command."""
        query = ' '.join(args)
        return await self.handle_search_command(query)

    async def _handle_explore_command(self, args: List[str]) -> str:
        """Internal handler for explore command."""
        if not args:
            return "Usage: /kg explore <node_id> [depth]"

        node_id = args[0]
        depth = int(args[1]) if len(args) > 1 else 2

        result = await self.explorer.explore_neighborhood(node_id, depth)

        if 'error' in result:
            return result['error']

        await self.panel.show_node_details(node_id)

        return f"Explored {node_id} (depth {depth}): {len(result['neighbors'])} neighbors"

    async def _handle_path_command(self, args: List[str]) -> str:
        """Internal handler for path command."""
        if len(args) < 2:
            return "Usage: /kg path <source> <target>"

        source = args[0]
        target = args[1]

        return await self.handle_path_command(source, target)

    async def _handle_communities_command(self, args: List[str]) -> str:
        """Internal handler for communities command."""
        return await self.handle_communities_command()

    async def _handle_stats_command(self, args: List[str]) -> str:
        """Internal handler for stats command."""
        return await self.handle_stats_command()

    async def _handle_export_command(self, args: List[str]) -> str:
        """Internal handler for export command."""
        format = args[0] if args else "json"
        return await self.handle_export_command(format)

    async def _handle_timeline_command(self, args: List[str]) -> str:
        """Internal handler for timeline command."""
        entity = ' '.join(args) if args else ""
        return await self.handle_timeline_command(entity)

    async def _handle_analyze_command(self, args: List[str]) -> str:
        """Internal handler for analyze command."""
        analysis_type = args[0] if args else ""
        return await self.handle_analyze_command(analysis_type)

    def get_command_history(self) -> List[str]:
        """
        Get command history.

        Returns:
            List of previous commands
        """
        return self.command_history.copy()

    def get_available_commands(self) -> List[Dict[str, str]]:
        """
        Get list of available commands with descriptions.

        Returns:
            List of command dictionaries
        """
        return [
            {
                'command': '/kg search <query>',
                'description': 'Search knowledge graph for matching nodes'
            },
            {
                'command': '/kg explore <node> [depth]',
                'description': 'Explore neighborhood around a node'
            },
            {
                'command': '/kg path <from> <to>',
                'description': 'Find shortest path between two nodes'
            },
            {
                'command': '/kg communities',
                'description': 'List and browse all communities'
            },
            {
                'command': '/kg stats',
                'description': 'Show comprehensive graph statistics'
            },
            {
                'command': '/kg export <format>',
                'description': 'Export graph (json, gexf, csv)'
            },
            {
                'command': '/kg timeline <entity>',
                'description': 'Show temporal timeline for an entity'
            },
            {
                'command': '/kg analyze <type>',
                'description': 'Run graph analysis (centrality, community, etc.)'
            },
        ]
